import { interp, median, rint, searchsorted } from './dsp'

export const HALF_WIDTH = 8
export const OUTLIER_MS = 25
export const MAPPER_CONVENTION = -0.0098
export const MIN_GRID_BPM = 90

export function diffs(x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(Math.max(0, x.length - 1))
  for (let i = 1; i < x.length; i++) out[i - 1] = x[i] - x[i - 1]
  return out
}

export function medianPeriod(beats: ArrayLike<number>): number {
  return median(diffs(beats))
}

export function repair(beats: ArrayLike<number>): number[] {
  const b = Array.from(beats)
  if (b.length < 4) return b
  const period = medianPeriod(b)
  const out = [b[0]]
  for (let i = 1; i < b.length; i++) {
    const last = out[out.length - 1]
    const gap = b[i] - last
    if (gap < 0.6 * period) continue
    const missing = rint(gap / period) - 1
    if (missing >= 1 && Math.abs(gap / (missing + 1) - period) < 0.15 * period) {
      for (let k = 1; k <= missing; k++) out.push(last + (gap * k) / (missing + 1))
    }
    out.push(b[i])
  }
  return out
}

export function smooth(beats: ArrayLike<number>, halfWidth = HALF_WIDTH, outlierMs = OUTLIER_MS): number[] {
  const n = beats.length
  const refined = Array.from(beats)
  if (n < 5) return refined
  let weights = new Float64Array(n).fill(1)
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - halfWidth)
      const hi = Math.min(n, i + halfWidth + 1)
      let s0 = 0
      let s1 = 0
      let s2 = 0
      let t0 = 0
      let t1 = 0
      for (let j = lo; j < hi; j++) {
        const u = j - i
        const w = weights[j] * (1 - (Math.abs(u) / (halfWidth + 1)) ** 3) ** 3
        const y = beats[j] - beats[i]
        s0 += w
        s1 += w * u
        s2 += w * u * u
        t0 += w * y
        t1 += w * u * y
      }
      if (s0 < 1e-6) continue
      refined[i] = beats[i] + (s2 * t0 - s1 * t1) / (s0 * s2 - s1 * s1)
    }
    const next = new Float64Array(n)
    for (let i = 0; i < n; i++) next[i] = Math.abs(beats[i] - refined[i]) * 1000 > outlierMs ? 0.05 : 1
    weights = next
  }
  return refined
}

export function alignToAttacks(beats: ArrayLike<number>, attacks: ArrayLike<number>, convention = MAPPER_CONVENTION, halfWidth = 8, div = 2, window = 0.06): number[] {
  const n = beats.length
  if (n < 4 || attacks.length < 6) return Array.from(beats, (b) => b + convention)
  const period = medianPeriod(beats)
  const idx = Array.from({ length: n }, (_, i) => i)
  const pos: number[] = []
  const off: number[] = []
  for (let k = 0; k < attacks.length; k++) {
    const p = interp(attacks[k], beats, idx, NaN, NaN)
    if (Number.isNaN(p)) continue
    const o = ((p * div - rint(p * div)) / div) * period
    if (Math.abs(o) < window) {
      pos.push(p)
      off.push(o)
    }
  }
  const fallback = off.length ? median(off) : 0
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const near: number[] = []
    for (let k = 0; k < pos.length; k++) if (Math.abs(pos[k] - i) <= halfWidth) near.push(off[k])
    out.push(beats[i] + (near.length >= 6 ? median(near) : fallback) + convention)
  }
  return out
}

export function doubleGrid(beats: ArrayLike<number>): number[] {
  const out = Array.from(beats)
  for (let i = 1; i < beats.length; i++) out.push((beats[i - 1] + beats[i]) / 2)
  return out.sort((a, b) => a - b)
}

export function songGrid(beats: ArrayLike<number>, attacks?: ArrayLike<number>): number[] {
  let g = smooth(repair(beats))
  if (attacks) g = alignToAttacks(g, attacks)
  if (g.length > 2 && 60 / medianPeriod(g) < MIN_GRID_BPM) g = doubleGrid(g)
  return g
}

export function arange(start: number, stop: number, step: number): number[] {
  const n = Math.max(0, Math.ceil((stop - start) / step))
  if (!n) return []
  const delta = start + step - start
  const out = [start]
  if (n > 1) out.push(start + step)
  for (let i = 2; i < n; i++) out.push(start + i * delta)
  return out
}

export function extendGrid(beats: ArrayLike<number>, duration: number): { grid: Float64Array; lead: number } {
  const n = beats.length
  const first = beats[1] - beats[0]
  const last = beats[n - 1] - beats[n - 2]
  const before = arange(beats[0] - first, -first, -first)
    .reverse()
    .filter((t) => t > -first)
  const after = arange(beats[n - 1] + last, duration + last, last)
  return { grid: Float64Array.from([...before, ...Array.from(beats), ...after]), lead: before.length }
}

export function barPositions(beatTimes: ArrayLike<number>, beatIndex: ArrayLike<number>, downbeats: ArrayLike<number>, period: number): Int32Array {
  const out = new Int32Array(beatTimes.length)
  for (let i = 0; i < beatTimes.length; i++) {
    if (!downbeats.length) {
      out[i] = ((beatIndex[i] % 4) + 4) % 4
      continue
    }
    const prev = searchsorted(downbeats, beatTimes[i] + 0.03) - 1
    const r = prev >= 0 ? rint((beatTimes[i] - downbeats[prev]) / period) : 0
    out[i] = ((r % 4) + 4) % 4
  }
  return out
}
