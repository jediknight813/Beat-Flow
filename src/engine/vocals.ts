export const FPS = 50
const PRE_MAX = 1
const POST_MAX = 1
const PRE_AVG = 5
const POST_AVG = 6
const WAIT = 1
const DELTA = 0.07
const TINY32 = 1.1754943508222875e-38

export type Syllable = { time: number; strength: number }

function sum32(x: Float32Array, lo: number, hi: number): number {
  const n = hi - lo
  if (n < 8) {
    let res = 0
    for (let i = lo; i < hi; i++) res = Math.fround(res + x[i])
    return res
  }
  const r = new Float32Array(8)
  for (let j = 0; j < 8; j++) r[j] = x[lo + j]
  const blocks = n - (n % 8)
  for (let i = 8; i < blocks; i += 8) for (let j = 0; j < 8; j++) r[j] = Math.fround(r[j] + x[lo + i + j])
  let res = Math.fround(Math.fround(Math.fround(r[0] + r[1]) + Math.fround(r[2] + r[3])) + Math.fround(Math.fround(r[4] + r[5]) + Math.fround(r[6] + r[7])))
  for (let i = lo + blocks; i < hi; i++) res = Math.fround(res + x[i])
  return res
}

function mean32(x: Float32Array, lo: number, hi: number): number {
  return Math.fround(sum32(x, lo, hi) / (hi - lo))
}

function max32(x: Float32Array, lo: number, hi: number): number {
  let m = -Infinity
  for (let i = lo; i < hi; i++) if (x[i] > m) m = x[i]
  return m
}

export function peakPick(x: Float32Array, preMax = PRE_MAX, postMax = POST_MAX, preAvg = PRE_AVG, postAvg = POST_AVG, delta = DELTA, wait = WAIT): number[] {
  const n = x.length
  const peaks: number[] = []
  if (!n) return peaks
  const first = x[0] >= max32(x, 0, Math.min(postMax, n)) && x[0] >= mean32(x, 0, Math.min(postAvg, n)) + delta
  if (first) peaks.push(0)
  let i = first ? wait + 1 : 1
  while (i < n) {
    const maxn = max32(x, Math.max(0, i - preMax), Math.min(i + postMax, n))
    if (x[i] !== maxn) {
      i += 1
      continue
    }
    const avgn = mean32(x, Math.max(0, i - preAvg), Math.min(i + postAvg, n))
    if (x[i] < avgn + delta) {
      i += 1
      continue
    }
    peaks.push(i)
    i += wait + 1
  }
  return peaks
}

export function onsetDetect(envelope: Float32Array): number[] {
  const n = envelope.length
  let min = Infinity
  for (let i = 0; i < n; i++) if (envelope[i] < min) min = envelope[i]
  const shifted = new Float32Array(n)
  for (let i = 0; i < n; i++) shifted[i] = Math.fround(envelope[i] - min)
  const scale = Math.fround(max32(shifted, 0, n) + TINY32)
  for (let i = 0; i < n; i++) shifted[i] = Math.fround(shifted[i] / scale)
  return peakPick(shifted)
}

export function zscore32(x: Float32Array): Float32Array {
  const n = x.length
  const mean = mean32(x, 0, n)
  const dev = new Float32Array(n)
  for (let i = 0; i < n; i++) dev[i] = Math.fround(Math.fround(x[i] - mean) * Math.fround(x[i] - mean))
  const std = Math.fround(Math.sqrt(mean32(dev, 0, n)))
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.fround(x[i] - mean) / (std + 1e-6)
  return out
}

export function syllables(envelope: Float32Array, active: Uint8Array): Syllable[] {
  const z = zscore32(envelope)
  const out: Syllable[] = []
  for (const p of onsetDetect(envelope)) {
    if (p >= active.length) continue
    let voiced = false
    for (let f = Math.max(0, p - 2); f < Math.min(active.length, p + 3); f++) if (active[f]) voiced = true
    if (voiced) out.push({ time: Math.round((p / FPS) * 1000) / 1000, strength: Math.round(z[p] * 1000) / 1000 })
  }
  return out
}
