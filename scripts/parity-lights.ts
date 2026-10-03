import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import JSZip from 'jszip'
import * as ort from 'onnxruntime-web'
import type { Note, Wall } from '../src/engine/types'
import type { LightEvent } from '../src/engine/package'
import {
  VOCAB, SONG_FEATURES, NOTE_FEATURES, CONTRAST_FLOOR, FLASH_CAP, canvasTargets, decode, flashCap, noteContext, runCanvas, sample, sectionKeep,
  shapeContrast, sortEvents, type Conditions, type Intensity, type LightsSong,
} from '../src/engine/lights'
import { audioFeatures, coverFeatures, coverPalette, styleFromPixels, type EnvironmentReference, type StyleAudio } from '../src/engine/style'

const ROOT = resolve(import.meta.dirname, '..')
const PARITY = resolve(ROOT, 'parity')
const args = new Set(process.argv.slice(2))
const doSample = args.has('--sample')
const modelArg = process.argv.find((a) => a.startsWith('--model='))?.slice(8) ?? 'canvas.int8.onnx'
const exact = modelArg !== 'canvas.int8.onnx'
const LEAD = 1.5
const CHANNELS = VOCAB.length
const FEATURES = SONG_FEATURES + NOTE_FEATURES

type Npz = Record<string, { shape: number[]; data: Float32Array | Float64Array | Int16Array }>

function float16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exp = (bits >> 10) & 0x1f
  const frac = bits & 0x3ff
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024)
  if (exp === 31) return frac ? NaN : sign * Infinity
  return sign * 2 ** (exp - 15) * (1 + frac / 1024)
}

async function readNpz(path: string): Promise<Npz> {
  const zip = await JSZip.loadAsync(readFileSync(path))
  const out: Npz = {}
  for (const name of Object.keys(zip.files)) {
    const raw = (await zip.files[name].async('uint8array')).slice()
    const major = raw[6]
    const headerLen = major === 1 ? raw[8] | (raw[9] << 8) : raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24)
    const start = (major === 1 ? 10 : 12) + headerLen
    const header = new TextDecoder().decode(raw.subarray(major === 1 ? 10 : 12, start))
    const descr = /'descr':\s*'([^']+)'/.exec(header)![1]
    const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)![1].match(/\d+/g) ?? []).map(Number)
    const body = raw.buffer.slice(raw.byteOffset + start, raw.byteOffset + raw.byteLength)
    let data: Float32Array | Float64Array | Int16Array
    if (descr === '<f4') data = new Float32Array(body)
    else if (descr === '<f8') data = new Float64Array(body)
    else if (descr === '<i2') data = new Int16Array(body)
    else if (descr === '<f2') data = Float32Array.from(new Uint16Array(body), float16)
    else throw new Error(`unsupported dtype ${descr} in ${name}`)
    out[name.replace(/\.npy$/, '')] = { shape, data }
  }
  return out
}

const json = (path: string) => JSON.parse(readFileSync(path, 'utf8'))

type Row = { song: string; diff: string; check: string; value: string; limit: string; ok: boolean }
const rows: Row[] = []
const report = (song: string, diff: string, check: string, value: number | string, limit: string, ok: boolean) =>
  rows.push({ song, diff, check, value: typeof value === 'number' ? value.toPrecision(4) : value, limit, ok })

async function loadSession(file: string): Promise<ort.InferenceSession> {
  const base = resolve(ROOT, 'public/models/lights')
  const manifest = json(resolve(base, 'manifest.json'))[file]
  const bytes = manifest ? Buffer.concat(manifest.files.map((f: string) => readFileSync(resolve(base, f)))) : readFileSync(file)
  return ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
}

type Fixture = [number, number, number, number]

function compareEvents(mine: LightEvent[], theirs: Fixture[]): { count: boolean; maxTime: number; mismatches: number } {
  const a = sortEvents(mine)
  const b = sortEvents(theirs.map(([time, type, value, float]) => ({ time, type, value, float })))
  let maxTime = 0
  let mismatches = Math.abs(a.length - b.length)
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const dt = Math.abs(a[i].time - b[i].time)
    maxTime = Math.max(maxTime, dt)
    if (dt > 1e-6 || a[i].type !== b[i].type || a[i].value !== b[i].value || Math.abs(a[i].float - b[i].float) > 1e-9) mismatches++
  }
  return { count: a.length === b.length, maxTime, mismatches }
}

function lightsSong(folder: string): LightsSong {
  const audio = json(resolve(folder, 'style-audio.json'))
  const style = json(resolve(folder, 'style.json'))
  const frames = audio.stem_rms[0].length
  const energy = new Float32Array(frames)
  for (const stem of audio.stem_rms) for (let i = 0; i < frames; i++) energy[i] = Math.fround(energy[i] + stem[i])
  return { sections: style.shaping.sections, energy }
}

function styleAudio(folder: string): StyleAudio {
  const audio = json(resolve(folder, 'style-audio.json'))
  return { tempo: audio.tempo_bpm, stemRms: audio.stem_rms.map((s: number[]) => Float32Array.from(s)), vocalPeriodicity: Float32Array.from(audio.vocal_periodicity) }
}

async function checkStyle(song: string, folder: string, ref: EnvironmentReference) {
  const fixture = json(resolve(folder, 'style.json'))
  const cover = json(resolve(folder, 'cover64.json'))
  const pixels = { rgb64: Uint8Array.from(cover.rgb64), rgb48: Uint8Array.from(cover.rgb48) }
  const audio = styleAudio(folder)
  const af = audioFeatures(audio)
  const afDiff = Math.max(...af.map((v, i) => Math.abs(v - fixture.audio_features[i])))
  report(song, '-', 'audio_features max abs', afDiff, '1e-3', afDiff <= 1e-3)
  const cf = coverFeatures(pixels.rgb48)
  const cfDiff = Math.max(...cf.map((v, i) => Math.abs(v - fixture.cover_features[i])))
  report(song, '-', 'cover_features max abs', cfDiff, '0.03', cfDiff <= 0.03)
  const palette = coverPalette(pixels.rgb64)
  const palDiff = Math.max(...palette.map((c, i) => Math.max(...c.rgb.map((v, k) => Math.abs(v - fixture.cover_palette[i].rgb[k])), Math.abs(c.share - fixture.cover_palette[i].share))))
  report(song, '-', 'palette (rgb, share) max abs', palDiff, '0.02', palDiff <= 0.02)
  const style = await styleFromPixels(pixels, audio, song, ref)
  const tagDiff = Math.max(...Object.keys(fixture.tags).map((k) => Math.abs((style.tags[k] ?? 0) - fixture.tags[k])))
  report(song, '-', 'tags max abs', tagDiff, '0.05', tagDiff <= 0.05)
  const names = ['saberA', 'saberB', 'env0', 'env1', 'env0boost', 'env1boost', 'obstacles'] as const
  const colDiff = Math.max(...names.flatMap((n) => (['r', 'g', 'b'] as const).map((ch) => Math.abs(style.colors[n][ch] - fixture.style.colors[n][ch]))))
  report(song, '-', 'colours max abs', colDiff, '0.02', colDiff <= 0.02)
  report(song, '-', 'colour rule', style.source.rule as string, fixture.style.colors.source.rule, style.source.rule === fixture.style.colors.source.rule)
  report(song, '-', 'environment', style.environment, fixture.style.environment, style.environment === fixture.style.environment)
  report(song, '-', 'category', style.category, fixture.style.category, style.category === fixture.style.category)
  report(song, '-', 'top3', style.top3.join(','), fixture.style.character.top3.join(','), style.top3.join(',') === fixture.style.character.top3.join(','))
}

async function checkDifficulty(song: string, diff: string, entry: { canvas: string; events: string; seed: number; shape_seed: number; T: number }, session: ort.InferenceSession) {
  const folder = resolve(PARITY, song)
  const npz = await readNpz(resolve(folder, entry.canvas))
  const ev = json(resolve(folder, entry.events))
  const style = json(resolve(folder, 'lights/style.json'))
  const T = entry.T
  const x = npz.x.data as Float32Array
  const times = npz.times.data as Float64Array
  const notes: Note[] = ev.notes.map((n: Note) => ({ time: n.time, x: n.x, y: n.y, hand: n.hand, direction: n.direction, angle: n.angle }))
  const walls: Wall[] = ev.walls
  const intensity = ev.intensity as Intensity

  const ctx = noteContext(notes, times)
  let ctxDiff = 0
  for (let i = 0; i < T; i++) for (let k = 0; k < NOTE_FEATURES; k++) ctxDiff = Math.max(ctxDiff, Math.abs(ctx[i * NOTE_FEATURES + k] - x[i * FEATURES + SONG_FEATURES + k]))
  report(song, diff, 'note_context max abs', ctxDiff, '1e-6', ctxDiff <= 1e-6)

  const given = canvasTargets(times, walls)
  const fixtureGiven = npz.given.data as Int16Array
  let givenBad = 0
  for (let i = 0; i < T * CHANNELS; i++) if (given[i] !== fixtureGiven[i]) givenBad++
  report(song, diff, 'given tokens mismatches', givenBad, '0', givenBad === 0)

  const conds = ev.conditions as Conditions
  const tokens0 = new Int32Array(T * CHANNELS)
  for (let i = 0; i < T; i++) for (let c = 0; c < CHANNELS; c++) tokens0[i * CHANNELS + c] = c === 0 ? fixtureGiven[i * CHANNELS] : VOCAB[c]
  const W = Math.min(T, 768)
  const logits = await runCanvas(session, x, tokens0, 0, W, conds)
  let maxAbs = 0
  let agree = 0
  let n = 0
  for (let c = 0; c < CHANNELS; c++) {
    const ref = npz[`logits0_${c}`].data as Float32Array
    const V = VOCAB[c]
    for (let i = 0; i < W; i++) {
      let am = 0
      let ar = 0
      for (let k = 0; k < V; k++) {
        maxAbs = Math.max(maxAbs, Math.abs(logits[c][i * V + k] - ref[i * V + k]))
        if (logits[c][i * V + k] > logits[c][i * V + am]) am = k
        if (ref[i * V + k] > ref[i * V + ar]) ar = k
      }
      agree += am === ar ? 1 : 0
      n++
    }
  }
  const absLimit = modelArg.includes('int8') ? 0.25 : modelArg.includes('fp16') ? 0.05 : 2e-3
  report(song, diff, `logits0 max abs (${modelArg})`, maxAbs, `${absLimit}`, maxAbs <= absLimit)
  report(song, diff, 'logits0 argmax agreement', agree / n, '>= 0.99', agree / n >= 0.99)

  const fixtureTokens = Int32Array.from(npz.tokens.data as Int16Array)
  const decoded = decode(fixtureTokens, times, LEAD)
  const d = compareEvents(decoded.events, ev.decoded)
  report(song, diff, `decoded events (${decoded.events.length} vs ${ev.decoded.length})`, d.mismatches, '0 mismatches', d.count && d.mismatches === 0)
  const boostsOk = decoded.boosts.length === ev.boosts.length && decoded.boosts.every(([t, on], i) => Math.abs(t - ev.boosts[i][0]) <= 1e-6 && on === ev.boosts[i][1])
  report(song, diff, `boosts (${decoded.boosts.length} vs ${ev.boosts.length})`, boostsOk ? 'exact' : 'differ', 'exact', boostsOk)

  const songRef = lightsSong(resolve(folder, 'lights'))
  const keep = sectionKeep(songRef, CONTRAST_FLOOR[intensity])
  const keepDiff = Math.max(...keep.map((v, i) => Math.abs(v - style.shaping.keep_p[intensity][i])))
  report(song, diff, 'keep_p max abs', keepDiff, '1e-6', keepDiff <= 1e-6)
  const shaped = shapeContrast(decoded.events, songRef, CONTRAST_FLOOR[intensity], LEAD, entry.shape_seed)
  const s = compareEvents(shaped, ev.shaped)
  report(song, diff, `shaped events (${shaped.length} vs ${ev.shaped.length})`, s.mismatches, '0 mismatches', s.count && s.mismatches === 0)
  const capped = flashCap(shaped, FLASH_CAP[intensity])
  const cp = compareEvents(capped, ev.capped)
  report(song, diff, `capped events (${capped.length} vs ${ev.capped.length})`, cp.mismatches, '0 mismatches', cp.count && cp.mismatches === 0)

  if (doSample) {
    const t0 = Date.now()
    const tokens = await sample(session, x, given, conds, entry.seed)
    const seconds = (Date.now() - t0) / 1000
    const nonzero = VOCAB.map((_, c) => {
      let k = 0
      for (let i = 0; i < T; i++) if (tokens[i * CHANNELS + c] > 0) k++
      return k
    })
    const refNonzero = ev.counts.tokens_nonzero as number[]
    const ratios = nonzero.slice(1).map((v, i) => v / Math.max(1, refNonzero[i + 1]))
    const worst = Math.max(...ratios.map((r) => Math.abs(r - 1)))
    report(song, diff, `sampled non-zero per channel ${nonzero.slice(1).join('/')} vs ${refNonzero.slice(1).join('/')} (${seconds.toFixed(0)} s)`, worst, 'within 0.2', worst <= 0.2)
    const mine = decode(tokens, times, LEAD).events.filter((e) => e.type <= 4).length
    const theirs = (ev.decoded as Fixture[]).filter((e) => e[1] <= 4).length
    const rate = mine / theirs
    report(song, diff, `sampled light events ${mine} vs ${theirs}`, Math.abs(rate - 1), 'within 0.2', Math.abs(rate - 1) <= 0.2)
    let same = 0
    for (let i = 0; i < T * CHANNELS; i++) if (tokens[i] === fixtureTokens[i]) same++
    report(song, diff, 'sampled tokens identical to fixture', same / (T * CHANNELS), exact ? '1 (same PCG64 stream)' : 'info', !exact || same === T * CHANNELS)
  }
}

const index = json(resolve(PARITY, 'lights.json'))
const ref = json(resolve(ROOT, 'public/data/environment-reference.json')) as EnvironmentReference
const session = await loadSession(modelArg)
for (const [song, entry] of Object.entries(index.songs) as [string, { difficulties: Record<string, never> }][]) {
  await checkStyle(song, resolve(PARITY, song, 'lights'), ref)
  for (const [diff, d] of Object.entries(entry.difficulties)) await checkDifficulty(song, diff, d, session)
}

const widths = [0, 1, 2, 3, 4].map((k) => Math.max(...rows.map((r) => [r.song, r.diff, r.check, r.value, r.limit][k].length)))
for (const r of rows) {
  const cells = [r.song, r.diff, r.check, r.value, r.limit].map((c, k) => c.padEnd(widths[k]))
  console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${cells.join('  ')}`)
}
const failed = rows.filter((r) => !r.ok).length
console.log(`${rows.length - failed}/${rows.length} checks passed`)
process.exit(failed ? 1 : 0)
