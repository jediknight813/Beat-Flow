import * as ort from 'onnxruntime-web'
import type { Difficulty, Note, Wall } from './types'
import type { LightEvent } from './package'
import { Pcg64 } from './pcg64'
import { wallKind } from './wallRules'

export const VOCAB = [73, 26, 26, 26, 26, 26, 2, 2, 6, 6, 3]
export const SONG_FEATURES = 369
export const NOTE_FEATURES = 17
const CHANNELS = VOCAB.length
const FEATURES = SONG_FEATURES + NOTE_FEATURES
const WINDOW = 768
const OVERLAP = 128
const STEPS = 16
const HORIZON = 1.5
const DUR_BUCKETS = [0.1, 0.5, 1.5]
const SPEED_VALUES = [0, 1, 3, 6, 8]
const BRIGHTNESS = [0.5, 1.0, 1.4]
const DIFFICULTY_INDEX: Record<Difficulty, number> = { Expert: 3, ExpertPlus: 4 }

export type Intensity = 'calm' | 'normal' | 'intense'
export const INTENSITY_BUCKET: Record<Intensity, number> = { calm: 3, normal: 4, intense: 5 }
export const CONTRAST_FLOOR: Record<Intensity, number> = { calm: 0.7, normal: 0.5, intense: 0.35 }
export const FLASH_CAP: Record<Intensity, number> = { calm: 8, normal: 12, intense: 18 }

export type LightsSong = {
  sections: { start: number; end: number }[]
  energy: Float32Array
}

export type Conditions = {
  diff: number
  walls_c: number
  lights_c: number
  lights_q: number
  walls_q: number
  notes_known: number
}

export type LightsInput = {
  features: Float32Array
  times: ArrayLike<number>
  notes: Note[]
  walls: Wall[]
  difficulty: Difficulty
  song: LightsSong
}

export type LightsResult = { events: LightEvent[]; boosts: [number, boolean][] }

function searchLeft(sorted: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

function searchRight(sorted: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] <= v) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function noteContext(notes: Note[], times: ArrayLike<number>): Float32Array {
  const T = times.length
  const out = new Float32Array(T * NOTE_FEATURES)
  if (!notes.length) return out
  const sorted = [...notes].sort((a, b) => a.time - b.time)
  const all = sorted.map((n) => n.time)
  for (let c = 0; c < 12; c++) {
    const tc = sorted.filter((n) => n.x * 3 + n.y === c).map((n) => n.time)
    if (!tc.length) continue
    for (let i = 0; i < T; i++) {
      const j = searchLeft(tc, times[i])
      if (j < tc.length) out[i * NOTE_FEATURES + c] = Math.min(1, Math.max(0, 1 - (tc[j] - times[i]) / HORIZON))
    }
  }
  const windows = [0.5, 1.0, 2.0]
  for (let i = 0; i < T; i++) {
    const t = times[i]
    const lo = searchLeft(all, t)
    windows.forEach((w, k) => (out[i * NOTE_FEATURES + 12 + k] = (searchLeft(all, t + w) - lo) / (w * 8)))
  }
  for (const h of [0, 1]) {
    const hand = sorted.filter((n) => n.hand === h)
    if (!hand.length) continue
    const th = hand.map((n) => n.time)
    for (let i = 0; i < T; i++) {
      const j = searchRight(th, times[i]) - 1
      out[i * NOTE_FEATURES + 15 + h] = j >= 0 ? hand[j].x / 3 : 0.5 - 0.5 * (h === 0 ? 1 : 0)
    }
  }
  return out
}

export function wallToken(w: Wall): number {
  const kind = wallKind(w)
  if (kind === 'other') return 0
  const lane = Math.min(4, Math.max(-1, w.x)) + 1
  const dur = DUR_BUCKETS.filter((d) => d < w.duration).length
  return 1 + (['side', 'crouch', 'dodge'].indexOf(kind) * 6 + lane) * 4 + dur
}

function nearestToken(times: ArrayLike<number>, t: number): number {
  const T = times.length
  const i = Math.min(T - 1, Math.max(0, searchLeft(times, t - 1e-3)))
  return i > 0 && Math.abs(times[i - 1] - t) < Math.abs(times[i] - t) ? i - 1 : i
}

export function canvasTargets(times: ArrayLike<number>, walls: Wall[]): Int32Array {
  const given = new Int32Array(times.length * CHANNELS)
  if (!times.length) return given
  for (const w of walls) {
    const tok = wallToken(w)
    if (tok) given[nearestToken(times, w.time) * CHANNELS] = tok
  }
  return given
}

export function conditions(difficulty: Difficulty, walls: Wall[], notes: Note[], intensity: Intensity): Conditions {
  return { diff: DIFFICULTY_INDEX[difficulty], walls_c: walls.length ? 4 : 1, lights_c: INTENSITY_BUCKET[intensity], lights_q: 4, walls_q: 4, notes_known: notes.length ? 1 : 0 }
}

export function canvasInput(features: Float32Array, notes: Note[], times: ArrayLike<number>): Float32Array {
  const T = times.length
  const ctx = noteContext(notes, times)
  const x = new Float32Array(T * FEATURES)
  for (let i = 0; i < T; i++) {
    x.set(features.subarray(i * SONG_FEATURES, (i + 1) * SONG_FEATURES), i * FEATURES)
    x.set(ctx.subarray(i * NOTE_FEATURES, (i + 1) * NOTE_FEATURES), i * FEATURES + SONG_FEATURES)
  }
  return x
}

export async function runCanvas(session: ort.InferenceSession, x: Float32Array, tokens: Int32Array, start: number, end: number, conds: Conditions): Promise<Float32Array[]> {
  const W = end - start
  const tok = new BigInt64Array(W * CHANNELS)
  for (let i = 0; i < W * CHANNELS; i++) tok[i] = BigInt(tokens[start * CHANNELS + i])
  const scalar = (v: number) => new ort.Tensor('int64', new BigInt64Array([BigInt(v)]), [1])
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor('float32', x.subarray(start * FEATURES, end * FEATURES), [1, W, FEATURES]),
    tokens: new ort.Tensor('int64', tok, [1, W, CHANNELS]),
  }
  for (const k of ['diff', 'walls_c', 'lights_c', 'lights_q', 'walls_q', 'notes_known'] as const) feeds[k] = scalar(conds[k])
  const out = await session.run(feeds)
  return VOCAB.map((_, c) => out[`logits_${c}`].data as Float32Array)
}

function sampleToken(logits: Float32Array, offset: number, size: number, u: number): number {
  let m = -Infinity
  for (let k = 0; k < size; k++) m = Math.max(m, logits[offset + k])
  const e = new Float32Array(size)
  let sum = 0
  for (let k = 0; k < size; k++) {
    e[k] = Math.exp(logits[offset + k] - m)
    sum = Math.fround(sum + e[k])
  }
  let cum = 0
  for (let k = 0; k < size; k++) {
    cum = Math.fround(cum + Math.fround(e[k] / sum))
    if (cum > u) return k
  }
  return 0
}

export async function sample(session: ort.InferenceSession, x: Float32Array, given: Int32Array, conds: Conditions, seed: number, onStep?: (fraction: number) => void): Promise<Int32Array> {
  const T = x.length / FEATURES
  const rng = new Pcg64(seed)
  const tokens = new Int32Array(T * CHANNELS)
  for (let i = 0; i < T; i++) {
    tokens[i * CHANNELS] = given[i * CHANNELS]
    for (let c = 1; c < CHANNELS; c++) tokens[i * CHANNELS + c] = VOCAB[c]
  }
  const windows: [number, number][] = []
  for (let start = 0; ; start = Math.min(T, start + WINDOW) - OVERLAP) {
    const end = Math.min(T, start + WINDOW)
    windows.push([start, end])
    if (end === T) break
  }
  for (const [wi, [start, end]] of windows.entries()) {
    const W = end - start
    const open = new Uint8Array(W * CHANNELS)
    let openCount = 0
    for (let i = 0; i < W * CHANNELS; i++) {
      if (i % CHANNELS === 0 || tokens[start * CHANNELS + i] !== VOCAB[i % CHANNELS]) continue
      open[i] = 1
      openCount++
    }
    const total = openCount
    for (let s = 0; s < STEPS; s++) {
      if (!openCount) break
      const logits = await runCanvas(session, x, tokens, start, end, conds)
      const targetOpen = Math.floor(total * Math.cos((Math.PI / 2) * (s + 1) / STEPS))
      const cands: { order: number; pos: number; c: number; tok: number }[] = []
      for (let c = 1; c < CHANNELS; c++) {
        const pos: number[] = []
        for (let i = 0; i < W; i++) if (open[i * CHANNELS + c]) pos.push(i)
        if (!pos.length) continue
        const choice = pos.map((p) => sampleToken(logits[c], p * VOCAB[c], VOCAB[c], rng.random()))
        pos.forEach((p, k) => cands.push({ order: rng.random(), pos: p, c, tok: choice[k] }))
      }
      cands.sort((a, b) => b.order - a.order)
      const commit = Math.max(1, openCount - targetOpen)
      for (const { pos, c, tok } of cands.slice(0, commit)) {
        tokens[(start + pos) * CHANNELS + c] = tok
        open[pos * CHANNELS + c] = 0
        openCount--
      }
      onStep?.((wi * STEPS + s + 1) / (windows.length * STEPS))
    }
  }
  return tokens
}

export function decode(tokens: Int32Array, times: ArrayLike<number>, lead = 0): LightsResult {
  const events: LightEvent[] = []
  const boosts: [number, boolean][] = []
  for (let i = 0; i < times.length; i++) {
    const t = times[i] + lead
    const row = tokens.subarray(i * CHANNELS, (i + 1) * CHANNELS)
    for (let g = 0; g < 5; g++) {
      const tok = row[1 + g]
      if (tok === 1) events.push({ time: t, type: g, value: 0, float: 1 })
      else if (tok >= 2) {
        const mc = Math.floor((tok - 2) / 3)
        const bright = (tok - 2) % 3
        const mode = Math.floor(mc / 2)
        const colour = mc % 2
        events.push({ time: t, type: g, value: (colour === 0 ? 5 : 1) + mode, float: BRIGHTNESS[bright] })
      }
    }
    if (row[6] === 1) events.push({ time: t, type: 8, value: 0, float: 1 })
    if (row[7] === 1) events.push({ time: t, type: 9, value: 0, float: 1 })
    if (row[8]) events.push({ time: t, type: 12, value: SPEED_VALUES[row[8] - 1], float: 1 })
    if (row[9]) events.push({ time: t, type: 13, value: SPEED_VALUES[row[9] - 1], float: 1 })
    if (row[10]) boosts.push([t, row[10] === 1])
  }
  return { events, boosts }
}

export function sectionKeep(song: LightsSong, floor: number): number[] {
  const { sections, energy } = song
  const e = sections.map((s) => {
    const lo = Math.min(energy.length, Math.floor(s.start * 50))
    const hi = Math.min(energy.length, Math.max(Math.floor(s.start * 50) + 1, Math.floor(s.end * 50)))
    let sum = 0
    for (let i = lo; i < hi; i++) sum += energy[i]
    return hi > lo ? sum / (hi - lo) : 0
  })
  const order = e.map((_, i) => i).sort((a, b) => e[a] - e[b])
  const rank = new Array<number>(e.length).fill(0)
  order.forEach((i, r) => (rank[i] = r / Math.max(1, e.length - 1)))
  return rank.map((r) => floor + (1 - floor) * r)
}

const ON_OR_FLASH = new Set([1, 2, 5, 6, 9, 10])
const FLASH = new Set([2, 6, 10])

export function shapeContrast(events: LightEvent[], song: LightsSong, floor: number, lead: number, seed: number): LightEvent[] {
  if (song.sections.length < 3) return events
  const keep = sectionKeep(song, floor)
  const starts = song.sections.map((s) => s.start + lead)
  const rng = new Pcg64(seed)
  return events.filter((ev) => {
    if (ev.type > 4 || !ON_OR_FLASH.has(ev.value)) return true
    const i = Math.max(0, searchRight(starts, ev.time) - 1)
    return rng.random() <= keep[i]
  })
}

export function sortEvents(events: LightEvent[]): LightEvent[] {
  return [...events].sort((a, b) => a.time - b.time || a.type - b.type || a.value - b.value || a.float - b.float)
}

export function flashCap(events: LightEvent[], perSecond: number): LightEvent[] {
  let recent: number[] = []
  const out: LightEvent[] = []
  for (const e of sortEvents(events)) {
    if (e.type <= 4 && FLASH.has(e.value)) {
      recent = recent.filter((t) => e.time - t < 1)
      if (recent.length >= perSecond) continue
      recent.push(e.time)
    }
    out.push(e)
  }
  return out
}

export async function sampleLights(session: ort.InferenceSession, input: LightsInput, seed: number, intensity: Intensity, onProgress?: (fraction: number) => void): Promise<LightsResult> {
  const x = canvasInput(input.features, input.notes, input.times)
  const given = canvasTargets(input.times, input.walls)
  const conds = conditions(input.difficulty, input.walls, input.notes, intensity)
  const tokens = await sample(session, x, given, conds, seed, onProgress)
  const { events, boosts } = decode(tokens, input.times)
  const shaped = shapeContrast(events, input.song, CONTRAST_FLOOR[intensity], 0, seed + 1)
  return { events: flashCap(shaped, FLASH_CAP[intensity]), boosts }
}
