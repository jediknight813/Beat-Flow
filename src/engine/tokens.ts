import { meanF32, rint, searchsorted } from './dsp'
import { FPS, MELS, STEMS } from './features'
import { barPositions, extendGrid, medianPeriod } from './grid'
import type { Section } from './sections'
import type { Syllable } from './vocals'

export const SLOTS = 12
export const MEL_FEATURES = STEMS * MELS
export const FEATURES = MEL_FEATURES + 8 + 2 * SLOTS + 3 + 10 + 4

export type SongInfo = { downbeats: number[]; duration: number; syllables: Syllable[]; sections: Section[] }

export type FrameArrays = { frames: number; mel: Float32Array; onset: Float32Array; rms: Float32Array; pitch: Float32Array; periodicity: Float32Array }

export type SongTokens = {
  count: number
  features: Float32Array
  times: Float64Array
  sectionIndex: Int32Array
  sectionLabel: Int32Array
  grid: Float64Array
  firstIndex: number
}

function sectionSlice(s: Section): [number, number] {
  const a = Math.trunc(s.start * FPS)
  return [a, Math.max(a + 1, Math.trunc(s.end * FPS))]
}

function maxOf(x: Float32Array, lo: number, hi: number): number {
  let m = -Infinity
  for (let i = lo; i < hi; i++) if (x[i] > m) m = x[i]
  return m
}

function boolMean(x: Uint8Array, lo: number, hi: number): number {
  let s = 0
  for (let i = lo; i < hi; i++) s += x[i]
  return s / (hi - lo)
}

export function songTokens(info: SongInfo, arrays: FrameArrays, beats: number[]): SongTokens {
  const { frames, mel, onset, rms, pitch, periodicity } = arrays
  const { duration, sections } = info
  const { grid } = extendGrid(beats, duration)
  const startList: number[] = []
  const indexList: number[] = []
  const allStarts: number[] = []
  for (let g = 0; g + 1 < grid.length; g++) {
    const d = grid[g + 1] - grid[g]
    for (let k = 0; k < 4; k++) allStarts.push(grid[g] + (d * k) / 4)
  }
  const allEnds = [...allStarts.slice(1), grid[grid.length - 1]]
  const endList: number[] = []
  for (let i = 0; i < allStarts.length; i++) {
    if (allEnds[i] > 0 && allStarts[i] < duration) {
      startList.push(allStarts[i])
      endList.push(allEnds[i])
      indexList.push(i)
    }
  }
  const T = startList.length
  const starts = Float64Array.from(startList)
  const out = new Float32Array(T * FEATURES)
  const melodic = new Float32Array(frames)
  const drums = onset.subarray(0, frames)
  for (let t = 0; t < frames; t++) melodic[t] = Math.max(onset[2 * frames + t], onset[3 * frames + t])
  const voiced = new Uint8Array(frames)
  const vocalOn = new Uint8Array(frames)
  for (let t = 0; t < frames; t++) {
    voiced[t] = periodicity[t] > 0.5 ? 1 : 0
    vocalOn[t] = voiced[t] && rms[3 * frames + t] > 0.02 ? 1 : 0
  }
  const energy = new Float32Array(frames)
  for (let t = 0; t < frames; t++) {
    let e = rms[t]
    for (let s = 1; s < STEMS; s++) e = Math.fround(e + rms[s * frames + t])
    energy[t] = e
  }
  const syll = info.syllables.map((s) => s.time)
  const down = info.downbeats
  const bounds = sections.map((s) => s.start)
  const secEnergy = sections.map((s) => {
    const [a, b] = sectionSlice(s)
    return meanF32(energy, a, Math.min(b, frames) - a)
  })
  const secVocal = sections.map((s) => {
    const [a, b] = sectionSlice(s)
    return boolMean(vocalOn, a, Math.min(b, frames))
  })
  const beatOf = indexList.map((i) => Math.floor(i / 4))
  const beatTimes = beatOf.map((b) => grid[Math.min(Math.max(b, 0), grid.length - 1)])
  const barPos = barPositions(beatTimes, beatOf, down, medianPeriod(beats))
  const sectionIndex = new Int32Array(T)
  for (let t = 0; t < T; t++) sectionIndex[t] = Math.min(Math.max(searchsorted(bounds, starts[t], 'right') - 1, 0), sections.length - 1)
  const voicedPitch = new Float32Array(frames)
  for (let t = 0; t < T; t++) {
    const row = out.subarray(t * FEATURES, (t + 1) * FEATURES)
    const st = starts[t]
    const en = endList[t]
    const a = Math.min(Math.max(Math.floor(st * FPS), 0), frames - 1)
    const b = Math.min(Math.max(Math.max(a + 1, Math.ceil(en * FPS)), 1), frames)
    const n = b - a
    for (let k = 0; k < MEL_FEATURES; k++) row[k] = meanF32(mel, k * frames + a, n)
    let c = MEL_FEATURES
    for (let s = 0; s < STEMS; s++) {
      row[c + s] = maxOf(onset, s * frames + a, s * frames + b)
      row[c + 4 + s] = meanF32(rms, s * frames + a, n)
    }
    c += 8
    const step = (en - st) / SLOTS
    const f: number[] = []
    for (let k = 0; k <= SLOTS; k++) {
      const edge = k === SLOTS ? en : k * step + st
      f.push(Math.min(Math.max(rint(edge * FPS), 0), frames - 1))
    }
    for (let k = 0; k < SLOTS; k++) {
      const s1 = Math.max(f[k] + 1, f[k + 1])
      row[c + k] = maxOf(drums, f[k], s1)
      row[c + SLOTS + k] = maxOf(melodic, f[k], s1)
    }
    c += 2 * SLOTS
    let p = 0
    for (let i = a; i < b; i++) if (voiced[i] && Number.isFinite(pitch[i])) voicedPitch[p++] = pitch[i]
    row[c] = p ? Math.fround(Math.fround(meanF32(voicedPitch, 0, p) - 60) / 12) : 0
    row[c + 1] = boolMean(voiced, a, b)
    let count = 0
    for (const s of syll) if (s >= st && s < en) count++
    row[c + 2] = count
    c += 3
    row[c + (indexList[t] % 4)] = 1
    row[c + 4 + barPos[t]] = 1
    let near = Infinity
    for (const d of down) near = Math.min(near, Math.abs(d - st))
    row[c + 8] = down.length && near < 0.03 ? 1 : 0
    row[c + 9] = Math.log(60 / Math.max(1e-3, (en - st) * 4) / 120)
    c += 10
    const si = sectionIndex[t]
    const sec = sections[si]
    const span = Math.max(1e-3, sec.end - sec.start)
    row[c] = (st - sec.start) / span
    row[c + 1] = secEnergy[si]
    row[c + 2] = secVocal[si]
    row[c + 3] = Math.min(8, sec.end - st) / 8
  }
  for (let i = 0; i < out.length; i++) if (Number.isNaN(out[i])) out[i] = 0
  return {
    count: T,
    features: out,
    times: starts,
    sectionIndex,
    sectionLabel: Int32Array.from(sectionIndex, (s) => sections[s].label),
    grid,
    firstIndex: indexList[0],
  }
}
