import { flowCosts } from '../src/engine/flow'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import * as ort from 'onnxruntime-web'
import { Rng } from '../src/engine/rng'
import { NOTE_GRAPHS, NoteDecoder, NoteState, type Gesture, type NoteSessions, type NoteSong, type Placed } from '../src/engine/notes'

type Sample = { z: number[]; mask: boolean[] | null; top_p: number; choice: number }
type Step = {
  position: number
  hand_logits: number[]
  end?: boolean
  hand?: number
  follow_logits?: number[]
  follow?: number
  prev_idx: number
  prev_time: number
  pointer_logits?: number[]
  mine_tokens?: number[]
  same_time?: number
  same_ok?: boolean
  cand?: number
  gesture?: Record<string, number[]>
  placed?: Record<string, number>
  swing_notes?: { time: number; x: number; y: number; direction: number }[]
  samples: Sample[]
  draws: number[]
  window_mask: boolean[] | null
  cell_mask: boolean[] | null
  parity_mask: boolean[] | null
  forced: { src: number; cand: number; mirror: boolean } | null
  forced_gesture: Record<string, number> | null
}

const args = process.argv.slice(2)
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : fallback
}
const parityDir = flag('--parity', '/mnt/storage/BeatFlow/parity')
const modelsDir = flag('--models', '/mnt/storage/BeatSaberModelTrainer/runs/export/notes')
const suffix = flag('--suffix', '')
const tol = Number(flag('--tol', '1e-3'))
const songs = flag('--songs', 'bereal,tweety').split(',')

ort.env.logLevel = 'error'

function bin<T extends Float32Array | Float64Array | Int32Array>(path: string, Ctor: new (b: ArrayBuffer) => T): T {
  const buf = readFileSync(path)
  return new Ctor(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
}

async function session(name: string): Promise<ort.InferenceSession> {
  const manifestPath = join(modelsDir, 'manifest.json')
  let bytes: Uint8Array
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))[`${name}.onnx`] as { files: string[]; bytes: number }
    bytes = new Uint8Array(manifest.bytes)
    let offset = 0
    for (const file of manifest.files) {
      const part = readFileSync(join(modelsDir, file))
      bytes.set(part, offset)
      offset += part.length
    }
  } else {
    const own = join(modelsDir, `${name}${suffix}.onnx`)
    bytes = readFileSync(existsSync(own) ? own : join(modelsDir, `${name}.onnx`))
  }
  return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
}

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs ${b.length}`)
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

const PLACED_KEYS: Record<string, keyof Placed> = {
  hand: 'hand',
  gap: 'gap',
  parity: 'parity',
  cut: 'cut',
  dot: 'dot',
  cell: 'cell',
  stack: 'stack',
  chain: 'chain',
  chain_slices: 'chainSlices',
  chain_tail: 'chainTail',
  arc_out: 'arcOut',
  follow: 'follow',
  token: 'token',
  cand: 'cand',
}
const GESTURE_KEYS: Record<string, keyof Gesture> = {
  parity: 'parity',
  cut: 'cut',
  dot: 'dot',
  cell: 'cell',
  stack: 'stack',
  chain: 'chain',
  chain_slices: 'chainSlices',
  chain_tail: 'chainTail',
  arc_out: 'arcOut',
}

class ReplayRng extends Rng {
  draws: number[] = []
  random(): number {
    if (!this.draws.length) throw new Error('rng draw with no recorded value')
    return this.draws.shift()!
  }
}

async function run(sessions: NoteSessions, song: string) {
  const dir = join(parityDir, song, 'notes')
  const meta = JSON.parse(readFileSync(join(dir, 'song.json'), 'utf8'))
  const critic = meta.knobs?.flow_lambda ? await ort.InferenceSession.create(readFileSync(join(modelsDir, '../critic.onnx')), { executionProviders: ['wasm'] }) : null
  const trace = JSON.parse(readFileSync(join(dir, 'trace.json'), 'utf8')) as Step[]
  const expected = JSON.parse(readFileSync(join(dir, 'result.json'), 'utf8'))
  const hs = bin(join(dir, 'h.f32.bin'), Float32Array)
  const input: NoteSong = {
    features: bin(join(dir, 'features.f32.bin'), Float32Array),
    sectionLabel: bin(join(dir, 'section_label.i32.bin'), Int32Array),
    candTimes: bin(join(dir, 'cand_times.f64.bin'), Float64Array),
    candFeats: bin(join(dir, 'cand_feats.f32.bin'), Float32Array),
    candTokens: bin(join(dir, 'cand_tokens.i32.bin'), Int32Array),
    duration: meta.duration,
    musicStart: meta.music_start,
    musicEnd: meta.music_end,
    introStem: meta.intro_stem,
    sustains: meta.sustains,
    segments: meta.segments,
    wallsPlan: [],
  }
  const rng = new ReplayRng(0)
  let pending: Sample[] = []
  let stepNo = 0
  const diffs: Record<string, number> = {}
  const note = (k: string, v: number) => (diffs[k] = Math.max(diffs[k] ?? 0, v))
  const failures: string[] = []
  const fail = (msg: string) => {
    failures.push(`step ${stepNo}: ${msg}`)
  }
  const agree = { window: 0, cell: 0, parity: 0, forced: 0, samples: 0, placed: 0 }
  const options = {
    difficulty: meta.difficulty,
    seed: meta.seed,
    temperature: meta.temperature,
    topP: meta.top_p,
    cond: meta.cond as [number, number],
    condScale: meta.cond_scale,
    rng,
    pick: (z: Float64Array, mask: boolean[] | null, topP: number) => {
      const s = pending.shift()
      if (!s) throw new Error(`step ${stepNo}: more sampling calls than recorded`)
      const dz = maxAbs(z, s.z)
      note('adjusted_logits', dz)
      if (dz > 2 * tol) fail(`adjusted logits differ by ${dz.toExponential(2)} (${z.length} classes)`)
      if (!same(mask, s.mask)) fail(`sampling mask differs for ${z.length} classes`)
      if (topP !== s.top_p) fail(`top_p ${topP} vs ${s.top_p}`)
      agree.samples++
      return s.choice
    },
  }
  const state = new NoteState(input, options)
  const dec = new NoteDecoder(sessions, input, options)
  let t0 = performance.now()
  await dec.prepare(input)
  const prepMs = performance.now() - t0
  let stepMs = 0
  let loopMs = 0
  for (const r of trace) {
    stepNo = r.position
    pending = [...r.samples]
    rng.draws = [...r.draws]
    const tLoop = performance.now()
    const { h, handLogits } = await dec.step()
    stepMs += performance.now() - tLoop
    note('h', maxAbs(h, hs.subarray(r.position * 512, (r.position + 1) * 512)))
    note('hand_logits', maxAbs(handLogits, r.hand_logits))
    const hand = state.chooseHand(handLogits)
    if (!same(state.forced, r.forced)) fail(`forced ${JSON.stringify(state.forced)} vs ${JSON.stringify(r.forced)}`)
    else agree.forced++
    if (hand === null || r.hand === undefined) {
      if (!(hand === null && r.end && r.hand === undefined)) fail(`hand ${hand} vs ${r.hand} (end ${r.end})`)
      break
    }
    if (hand !== r.hand) fail(`hand ${hand} vs ${r.hand}`)
    const fl = await dec.followLogits(hand)
    note('follow_logits', maxAbs(fl, r.follow_logits!))
    const follow = state.chooseFollow(fl)
    if (follow !== r.follow) fail(`follow ${follow} vs ${r.follow}`)
    const pl = await dec.pointerLogits(hand, follow)
    note('pointer_logits', maxAbs(pl, r.pointer_logits!))
    if (!same(Array.from(dec.mineTokens.data as BigInt64Array, Number), r.mine_tokens)) fail('mine_tokens differ')
    if (dec.prevIdx !== r.prev_idx || state.prevIdx !== r.prev_idx) fail(`prev_idx ${dec.prevIdx}/${state.prevIdx} vs ${r.prev_idx}`)
    if (Math.abs(state.prevTime - r.prev_time) > 1e-9) fail(`prev_time ${state.prevTime} vs ${r.prev_time}`)
    const window = dec.window()
    const wm = state.windowMask(window)
    if (same(wm, r.window_mask)) agree.window++
    else fail('window mask differs')
    const cand = state.chooseCandidate(pl, window)
    if (cand === null || r.cand === undefined) {
      if (!(cand === null && r.end && r.cand === undefined)) fail(`cand ${cand} vs ${r.cand}`)
      break
    }
    if (cand !== r.cand) fail(`cand ${cand} vs ${r.cand}`)
    const gl = await dec.gestureLogits(cand)
    for (const [k, v] of Object.entries(r.gesture!)) note(k, maxAbs(gl[GESTURE_KEYS[k]], v))
    const cm = state.cellMask(state.time)
    if (same(cm, r.cell_mask)) agree.cell++
    else fail(`cell mask ${JSON.stringify(cm.map(Number))} vs ${JSON.stringify(r.cell_mask!.map(Number))}`)
    const pm = state.parityMask(state.time)
    if (same(pm, r.parity_mask)) agree.parity++
    else fail('parity mask differs')
    const fg = state.forced ? state.forcedGesture() : null
    const fgExpected = r.forced_gesture ? Object.fromEntries(Object.entries(r.forced_gesture).map(([k, v]) => [GESTURE_KEYS[k], v])) : null
    if (!same(fg, fgExpected)) fail(`forced gesture ${JSON.stringify(fg)} vs ${JSON.stringify(fgExpected)}`)
    state.chooseGesture(gl, critic && !(state.forced && state.forcedGesture()) ? await flowCosts(critic, state.swings, state.hand, state.time, meta.difficulty) : undefined)
    const placed = state.history[state.history.length - 1]
    const placedExpected = Object.fromEntries(Object.entries(r.placed!).map(([k, v]) => [PLACED_KEYS[k], v]))
    if (same(placed, placedExpected)) agree.placed++
    else fail(`placed ${JSON.stringify(placed)} vs ${JSON.stringify(placedExpected)}`)
    const sw = state.swings[state.swings.length - 1]
    if (!same(sw.notes.map((n) => [n.time, n.x, n.y, n.direction]), r.swing_notes!.map((n) => [n.time, n.x, n.y, n.direction]))) fail('swing notes differ')
    dec.place(placed, state.time)
    if (pending.length) fail(`${pending.length} recorded sampling calls not consumed`)
    if (rng.draws.length) fail(`${rng.draws.length} recorded rng draws not consumed`)
    loopMs += performance.now() - tLoop
  }
  const result = state.result()
  const notesOk = same(
    result.notes.map((n) => [n.time, n.x, n.y, n.hand, n.direction]),
    expected.notes.map((n: Record<string, number>) => [n.time, n.x, n.y, n.hand, n.direction]),
  )
  const arcsOk = same(
    result.arcs.map((a) => [a.time, a.tailTime, a.hand, a.x, a.y, a.direction, a.tailX, a.tailY, a.tailDirection]),
    expected.arcs.map((a: Record<string, number>) => [a.time, a.tail_time, a.hand, a.x, a.y, a.direction, a.tail_x, a.tail_y, a.tail_direction]),
  )
  if (!notesOk) failures.push(`final notes differ (${result.notes.length} vs ${expected.notes.length})`)
  if (!arcsOk) failures.push(`final arcs differ (${result.arcs.length} vs ${expected.arcs.length})`)
  const logitKeys = Object.keys(diffs).filter((k) => k !== 'adjusted_logits')
  const worst = Math.max(...logitKeys.map((k) => diffs[k]))
  const steps = state.swings.length
  console.log(`\n${song} ${meta.difficulty} seed ${meta.seed}: ${trace.length} steps replayed, ${steps} swings, ${result.notes.length} notes, ${result.arcs.length} arcs, copied ${state.copied}`)
  console.log(`  prepare ${prepMs.toFixed(0)} ms; step graph ${(stepMs / steps).toFixed(1)} ms/step; full loop ${(loopMs / steps).toFixed(1)} ms/step`)
  for (const k of Object.keys(diffs)) console.log(`  ${k.padEnd(18)} ${diffs[k].toExponential(2)}`)
  console.log(`  logits max abs diff ${worst.toExponential(2)} ${worst <= tol ? `OK (<= ${tol})` : `(> ${tol})`}`)
  console.log(`  masks: window ${agree.window}/${steps}, cell ${agree.cell}/${steps}, parity ${agree.parity}/${steps}, forced ${agree.forced}/${trace.length}, sampling calls ${agree.samples}, placed ${agree.placed}/${steps}`)
  console.log(`  final notes ${notesOk ? 'equal' : 'DIFFER'}, arcs ${arcsOk ? 'equal' : 'DIFFER'}`)
  if (failures.length) console.log(`  FAILURES (${failures.length}):\n    ${failures.slice(0, 20).join('\n    ')}`)
  dec.dispose()
  if (critic) await critic.release()
  return failures.length === 0 && worst <= tol
}

const sessions = {} as NoteSessions
const tLoad = performance.now()
for (const name of NOTE_GRAPHS) sessions[name] = await session(name)
console.log(`loaded ${NOTE_GRAPHS.length} graphs from ${modelsDir}${suffix ? ` (${suffix})` : ''} in ${((performance.now() - tLoad) / 1000).toFixed(1)} s`)
let ok = true
for (const song of songs) ok = (await run(sessions, song)) && ok
console.log(ok ? '\nPARITY OK' : '\nPARITY FAILED')
process.exit(ok ? 0 : 1)
