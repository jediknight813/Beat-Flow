import { interp, percentile, rint, roundDecimals, searchsorted } from './dsp'
import { FPS } from './features'
import { MAPPER_CONVENTION, medianPeriod } from './grid'
import type { Syllable } from './vocals'

export const MERGE = 0.015
export const CAND_FEATURES = 18
const SUBDIVS = [1, 2, 4, 3, 6]
const KICK_BANDS = 9

export type Candidates = { count: number; times: Float64Array; features: Float32Array; tokens: Int32Array }

export function gridLines(grid: ArrayLike<number>): number[] {
  const lines: number[] = Array.from(grid)
  for (const div of [2, 3, 4]) for (let k = 1; k < div; k++) for (let i = 0; i + 1 < grid.length; i++) lines.push(grid[i] + ((grid[i + 1] - grid[i]) * k) / div)
  const rounded = lines.map((t) => roundDecimals(t, 4)).sort((a, b) => a - b)
  return rounded.filter((t, i) => i === 0 || t !== rounded[i - 1])
}

export function gridPositions(times: ArrayLike<number>, grid: ArrayLike<number>): Float64Array {
  const n = grid.length
  const idx = Array.from({ length: n }, (_, i) => i)
  return Float64Array.from(times, (t) => {
    if (t < grid[0]) return (t - grid[0]) / (grid[1] - grid[0])
    if (t > grid[n - 1]) return n - 1 + (t - grid[n - 1]) / (grid[n - 1] - grid[n - 2])
    return interp(t, grid, idx)
  })
}

export function kickCurve(mel: Float32Array, frames: number): Float32Array {
  const low = new Float32Array(frames)
  for (let t = 0; t < frames; t++) {
    let s = mel[t]
    for (let b = 1; b < KICK_BANDS; b++) s = Math.fround(s + mel[b * frames + t])
    low[t] = s
  }
  const kick = new Float32Array(frames)
  for (let t = 1; t < frames; t++) kick[t] = Math.max(0, Math.fround(low[t] - low[t - 1]))
  return kick
}

export function candidates(
  grid: ArrayLike<number>,
  attacks: ArrayLike<number>,
  syllables: Syllable[],
  onset: Float32Array,
  mel: Float32Array,
  frames: number,
  duration: number,
  tokenTimes: ArrayLike<number>,
): Candidates {
  const events: [number, number][] = []
  for (let i = 0; i < attacks.length; i++) events.push([attacks[i] + MAPPER_CONVENTION, 0])
  for (const s of syllables) events.push([s.time, 1])
  for (const g of gridLines(grid)) events.push([g, 2])
  const sorted = events.filter(([t]) => t >= 0 && t < duration).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const times: number[] = []
  const flags: number[][] = []
  for (const [t, k] of sorted) {
    const last = times.length - 1
    if (last >= 0 && t - times[last] <= MERGE) {
      flags[last][k] = 1
      if (k === 0) times[last] = t
    } else {
      const f = [0, 0, 0]
      f[k] = 1
      times.push(t)
      flags.push(f)
    }
  }
  const C = times.length
  const features = new Float32Array(C * CAND_FEATURES)
  const kick = kickCurve(mel, frames)
  const p95 = percentile(kick, 95) + 1e-6
  const period = medianPeriod(grid)
  const tokens = new Int32Array(C)
  const positions = gridPositions(times, grid)
  for (let c = 0; c < C; c++) {
    const row = features.subarray(c * CAND_FEATURES, (c + 1) * CAND_FEATURES)
    row.set(flags[c])
    const f = Math.min(Math.max(rint(times[c] * FPS), 1), frames - 2)
    let salience = -Infinity
    for (let s = 0; s < 4; s++) {
      const o = s * frames + f
      const v = Math.max(onset[o - 1], onset[o], onset[o + 1])
      salience = Math.max(salience, v)
      row[3 + s] = Math.fround(Math.min(Math.max(v, -3), 6) / 3)
    }
    row[7] = Math.fround(Math.min(Math.max(salience, -3), 6) / 3)
    const frac = positions[c] - Math.floor(positions[c])
    let sub = 5
    for (let j = 0; j < SUBDIVS.length; j++) {
      const div = SUBDIVS[j]
      if (((((Math.abs(frac * div - rint(frac * div)) / div) * 1000 * period) / 1000) <= 0.012)) {
        sub = j
        break
      }
    }
    row[8 + sub] = 1
    row[14] = Math.sin(2 * Math.PI * frac)
    row[15] = Math.cos(2 * Math.PI * frac)
    const k = Math.max(kick[f - 1], kick[f], kick[f + 1]) / p95
    row[16] = Math.min(Math.max(k, 0), 3)
    row[17] = flags[c][0] + flags[c][1] > 0 ? 1 : 0
    tokens[c] = Math.min(Math.max(searchsorted(tokenTimes, times[c], 'right') - 1, 0), tokenTimes.length - 1)
  }
  return { count: C, times: Float64Array.from(times), features, tokens }
}
