export type Matrix = { rows: number; cols: number; data: Float32Array }

export type PadMode = 'constant' | 'reflect'

export type StftOptions = { nFft: number; hop: number; winLength?: number; padMode?: PadMode }

export type MelOptions = {
  sr: number
  nFft: number
  nMels: number
  fMin: number
  fMax: number
  norm: 'slaney' | null
}

export type MelBank = { nMels: number; nBins: number; start: Int32Array; weights: Float32Array[] }

export function hann(n: number): Float64Array {
  const w = new Float64Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)
  return w
}

export class RealFFT {
  readonly n: number
  private readonly half: number
  private readonly rev: Uint32Array
  private readonly tc: Float64Array
  private readonly ts: Float64Array
  private readonly wc: Float64Array
  private readonly ws: Float64Array
  private readonly zr: Float64Array
  private readonly zi: Float64Array

  constructor(n: number) {
    if (n < 4 || (n & (n - 1)) !== 0) throw new Error(`fft size must be a power of two >= 4, got ${n}`)
    this.n = n
    const m = n / 2
    this.half = m
    this.rev = new Uint32Array(m)
    let bits = 0
    while (1 << bits < m) bits++
    for (let i = 0; i < m; i++) {
      let r = 0
      for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b)
      this.rev[i] = r
    }
    this.tc = new Float64Array(m / 2)
    this.ts = new Float64Array(m / 2)
    for (let i = 0; i < m / 2; i++) {
      this.tc[i] = Math.cos((2 * Math.PI * i) / m)
      this.ts[i] = -Math.sin((2 * Math.PI * i) / m)
    }
    this.wc = new Float64Array(m + 1)
    this.ws = new Float64Array(m + 1)
    for (let i = 0; i <= m; i++) {
      this.wc[i] = Math.cos((2 * Math.PI * i) / n)
      this.ws[i] = -Math.sin((2 * Math.PI * i) / n)
    }
    this.zr = new Float64Array(m)
    this.zi = new Float64Array(m)
  }

  forward(input: ArrayLike<number>, outRe: Float64Array, outIm: Float64Array): void {
    const m = this.half
    const zr = this.zr
    const zi = this.zi
    const rev = this.rev
    for (let i = 0; i < m; i++) {
      const r = rev[i]
      zr[r] = input[2 * i]
      zi[r] = input[2 * i + 1]
    }
    for (let size = 2; size <= m; size <<= 1) {
      const halfSize = size >> 1
      const step = m / size
      for (let start = 0; start < m; start += size) {
        for (let k = 0; k < halfSize; k++) {
          const c = this.tc[k * step]
          const s = this.ts[k * step]
          const a = start + k
          const b = a + halfSize
          const tr = zr[b] * c - zi[b] * s
          const ti = zr[b] * s + zi[b] * c
          zr[b] = zr[a] - tr
          zi[b] = zi[a] - ti
          zr[a] += tr
          zi[a] += ti
        }
      }
    }
    for (let k = 0; k <= m; k++) {
      const k1 = k === m ? 0 : k
      const k2 = k === 0 ? 0 : m - k
      const er = 0.5 * (zr[k1] + zr[k2])
      const ei = 0.5 * (zi[k1] - zi[k2])
      const or = 0.5 * (zi[k1] + zi[k2])
      const oi = -0.5 * (zr[k1] - zr[k2])
      const c = this.wc[k]
      const s = this.ws[k]
      outRe[k] = er + or * c - oi * s
      outIm[k] = ei + or * s + oi * c
    }
  }
}

const fftCache = new Map<number, RealFFT>()

export function realFFT(n: number): RealFFT {
  let f = fftCache.get(n)
  if (!f) {
    f = new RealFFT(n)
    fftCache.set(n, f)
  }
  return f
}

export function frameCount(length: number, hop: number): number {
  return 1 + Math.floor(length / hop)
}

function padded(y: ArrayLike<number>, i: number, mode: PadMode): number {
  const n = y.length
  if (i >= 0 && i < n) return y[i]
  if (mode === 'constant') return 0
  if (i < 0) return y[-i]
  return y[2 * (n - 1) - i]
}

export function spectrogram(y: ArrayLike<number>, opts: StftOptions, power: number): Matrix {
  const { nFft, hop } = opts
  const winLength = opts.winLength ?? nFft
  const padMode = opts.padMode ?? 'constant'
  const window = hann(winLength)
  const offset = Math.floor((nFft - winLength) / 2)
  const pad = nFft >> 1
  const frames = frameCount(y.length, hop)
  const bins = nFft / 2 + 1
  const fft = realFFT(nFft)
  const buf = new Float64Array(nFft)
  const re = new Float64Array(bins)
  const im = new Float64Array(bins)
  const data = new Float32Array(bins * frames)
  const n = y.length
  for (let t = 0; t < frames; t++) {
    const base = t * hop - pad + offset
    buf.fill(0)
    if (base >= 0 && base + winLength <= n) {
      for (let j = 0; j < winLength; j++) buf[offset + j] = window[j] * y[base + j]
    } else {
      for (let j = 0; j < winLength; j++) buf[offset + j] = window[j] * padded(y, base + j, padMode)
    }
    fft.forward(buf, re, im)
    for (let b = 0; b < bins; b++) {
      const p = re[b] * re[b] + im[b] * im[b]
      data[b * frames + t] = power === 2 ? p : power === 1 ? Math.sqrt(p) : Math.pow(p, power / 2)
    }
  }
  return { rows: bins, cols: frames, data }
}

export function hzToMel(hz: number): number {
  const fSp = 200 / 3
  const minLogHz = 1000
  const minLogMel = minLogHz / fSp
  const logStep = Math.log(6.4) / 27
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logStep : hz / fSp
}

export function melToHz(mel: number): number {
  const fSp = 200 / 3
  const minLogHz = 1000
  const minLogMel = minLogHz / fSp
  const logStep = Math.log(6.4) / 27
  return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : fSp * mel
}

export function melFrequencies(n: number, fMin: number, fMax: number): Float64Array {
  const lo = hzToMel(fMin)
  const hi = hzToMel(fMax)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = melToHz(lo + ((hi - lo) * i) / (n - 1))
  return out
}

export function melFilterbank(opts: MelOptions): MelBank {
  const { sr, nFft, nMels, fMin, fMax, norm } = opts
  const nBins = nFft / 2 + 1
  const melF = melFrequencies(nMels + 2, fMin, fMax)
  const start = new Int32Array(nMels)
  const weights: Float32Array[] = []
  for (let m = 0; m < nMels; m++) {
    const row = new Float64Array(nBins)
    const enorm = norm === 'slaney' ? 2 / (melF[m + 2] - melF[m]) : 1
    let lo = -1
    let hi = -1
    for (let b = 0; b < nBins; b++) {
      const f = (b * sr) / nFft
      const lower = (f - melF[m]) / (melF[m + 1] - melF[m])
      const upper = (melF[m + 2] - f) / (melF[m + 2] - melF[m + 1])
      const w = Math.max(0, Math.min(lower, upper))
      if (w > 0) {
        row[b] = Math.fround(Math.fround(w) * enorm)
        if (lo < 0) lo = b
        hi = b
      }
    }
    start[m] = lo < 0 ? 0 : lo
    weights.push(lo < 0 ? new Float32Array(0) : new Float32Array(row.subarray(lo, hi + 1)))
  }
  return { nMels, nBins, start, weights }
}

export function applyMel(bank: MelBank, spec: Matrix): Matrix {
  const T = spec.cols
  const data = new Float32Array(bank.nMels * T)
  const acc = new Float64Array(T)
  for (let m = 0; m < bank.nMels; m++) {
    const w = bank.weights[m]
    acc.fill(0)
    for (let k = 0; k < w.length; k++) {
      const wb = w[k]
      const row = (bank.start[m] + k) * T
      for (let t = 0; t < T; t++) acc[t] += wb * spec.data[row + t]
    }
    data.set(acc, m * T)
  }
  return { rows: bank.nMels, cols: T, data }
}

const bankCache = new Map<string, MelBank>()

export function melBank(opts: MelOptions): MelBank {
  const key = `${opts.sr}/${opts.nFft}/${opts.nMels}/${opts.fMin}/${opts.fMax}/${opts.norm}`
  let bank = bankCache.get(key)
  if (!bank) {
    bank = melFilterbank(opts)
    bankCache.set(key, bank)
  }
  return bank
}

export function melSpectrogram(y: ArrayLike<number>, stft: StftOptions, mel: MelOptions, power = 2): Matrix {
  return applyMel(melBank(mel), spectrogram(y, stft, power))
}

export function powerToDb(S: Matrix, topDb = 80): Matrix {
  const d = S.data
  let max = -Infinity
  for (let i = 0; i < d.length; i++) {
    const v = 10 * Math.log10(Math.max(1e-10, d[i]))
    d[i] = v
    if (v > max) max = v
  }
  const floor = max - topDb
  for (let i = 0; i < d.length; i++) if (d[i] < floor) d[i] = floor
  return S
}

export function maxFilterRows(S: Matrix, size: number): Matrix {
  const { rows, cols } = S
  const out = new Float32Array(rows * cols)
  const half = size >> 1
  for (let r = 0; r < rows; r++) {
    for (let t = 0; t < cols; t++) {
      let m = -Infinity
      for (let k = -half; k <= size - half - 1; k++) {
        let rr = r + k
        if (rr < 0) rr = -rr - 1
        if (rr >= rows) rr = 2 * rows - rr - 1
        const v = S.data[rr * cols + t]
        if (v > m) m = v
      }
      out[r * cols + t] = m
    }
  }
  return { rows, cols, data: out }
}

export type OnsetOptions = { sr: number; hop: number; nFft?: number; lag?: number; maxSize?: number }

export function onsetStrength(y: ArrayLike<number>, opts: OnsetOptions): Float32Array {
  const nFft = opts.nFft ?? 2048
  const lag = opts.lag ?? 1
  const maxSize = opts.maxSize ?? 1
  const S = powerToDb(melSpectrogram(y, { nFft, hop: opts.hop }, { sr: opts.sr, nFft, nMels: 128, fMin: 0, fMax: 8000, norm: 'slaney' }))
  const ref = maxSize > 1 ? maxFilterRows(S, maxSize) : S
  const T = S.cols
  const out = new Float32Array(T)
  const padWidth = lag + Math.floor(nFft / (2 * opts.hop))
  for (let t = 0; t + lag < T; t++) {
    const o = t + padWidth
    if (o >= T) break
    let sum = 0
    for (let m = 0; m < S.rows; m++) {
      const d = S.data[m * T + t + lag] - ref.data[m * T + t]
      if (d > 0) sum += d
    }
    out[o] = sum / S.rows
  }
  return out
}

export function rms(y: ArrayLike<number>, frameLength: number, hop: number): Float32Array {
  const frames = frameCount(y.length, hop)
  const out = new Float32Array(frames)
  const half = frameLength >> 1
  const n = y.length
  for (let t = 0; t < frames; t++) {
    const base = t * hop - half
    const lo = Math.max(0, base)
    const hi = Math.min(n, base + frameLength)
    let sum = 0
    for (let i = lo; i < hi; i++) sum += y[i] * y[i]
    out[t] = Math.sqrt(sum / frameLength)
  }
  return out
}

export function mean(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return x.length ? s / x.length : 0
}

export function std(x: ArrayLike<number>, ddof = 0): number {
  const m = mean(x)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] - m) * (x[i] - m)
  return Math.sqrt(s / (x.length - ddof))
}

export function zscore(x: ArrayLike<number>, ddof = 0, eps = 1e-6): Float32Array {
  const m = mean(x)
  const s = std(x, ddof) + eps
  const out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) / s
  return out
}

export function median(x: ArrayLike<number>): number {
  const n = x.length
  if (!n) return NaN
  const s = Float64Array.from(x as ArrayLike<number>).sort()
  const m = n >> 1
  return n % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function percentile(x: ArrayLike<number>, p: number): number {
  const s = Float64Array.from(x as ArrayLike<number>).sort()
  const pos = (p / 100) * (s.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(s.length - 1, lo + 1)
  const g = pos - lo
  const d = s[hi] - s[lo]
  return g >= 0.5 ? s[hi] - d * (1 - g) : s[lo] + d * g
}

export function medianFilter(x: ArrayLike<number>, size: number, mode: 'reflect' | 'symmetric' = 'symmetric'): Float32Array {
  const n = x.length
  const out = new Float32Array(n)
  const half = size >> 1
  const win: number[] = []
  for (let i = 0; i < n; i++) {
    win.length = 0
    for (let k = -half; k <= size - half - 1; k++) {
      let j = i + k
      if (mode === 'symmetric') {
        if (j < 0) j = -j - 1
        if (j >= n) j = 2 * n - j - 1
      } else {
        if (j < 0) j = -j
        if (j >= n) j = 2 * (n - 1) - j
      }
      win.push(x[j])
    }
    out[i] = median(win)
  }
  return out
}

export function findPeaks(x: ArrayLike<number>, height: number, distance: number): number[] {
  const n = x.length
  const peaks: number[] = []
  let i = 1
  while (i < n - 1) {
    if (x[i - 1] < x[i]) {
      let j = i
      while (j + 1 < n && x[j + 1] === x[i]) j++
      if (j + 1 < n && x[j + 1] < x[i]) {
        peaks.push((i + j) >> 1)
        i = j
      }
    }
    i++
  }
  const kept = peaks.filter((p) => x[p] >= height)
  const order = kept.map((_, k) => k).sort((a, b) => x[kept[a]] - x[kept[b]] || a - b)
  const keep = new Uint8Array(kept.length).fill(1)
  for (let q = order.length - 1; q >= 0; q--) {
    const j = order[q]
    if (!keep[j]) continue
    let k = j - 1
    while (k >= 0 && kept[j] - kept[k] < distance) keep[k--] = 0
    k = j + 1
    while (k < kept.length && kept[k] - kept[j] < distance) keep[k++] = 0
  }
  return kept.filter((_, k) => keep[k])
}

export type PeakPickOptions = { preMax: number; postMax: number; preAvg: number; postAvg: number; delta: number; wait: number }

export function peakPick(x: ArrayLike<number>, o: PeakPickOptions): number[] {
  const n = x.length
  const out: number[] = []
  if (!n) return out
  const maxOf = (a: number, b: number) => {
    let m = -Infinity
    for (let i = a; i < b; i++) if (x[i] > m) m = x[i]
    return m
  }
  const meanOf = (a: number, b: number) => {
    let s = 0
    for (let i = a; i < b; i++) s += x[i]
    return s / (b - a)
  }
  let first = x[0] >= maxOf(0, Math.min(o.postMax, n))
  first &&= x[0] >= meanOf(0, Math.min(o.postAvg, n)) + o.delta
  if (first) out.push(0)
  let i = first ? o.wait + 1 : 1
  while (i < n) {
    if (x[i] !== maxOf(Math.max(0, i - o.preMax), Math.min(i + o.postMax, n))) {
      i++
      continue
    }
    if (x[i] < meanOf(Math.max(0, i - o.preAvg), Math.min(i + o.postAvg, n)) + o.delta) {
      i++
      continue
    }
    out.push(i)
    i += o.wait + 1
  }
  return out
}

export function rint(x: number): number {
  const f = Math.floor(x)
  const d = x - f
  if (d < 0.5) return f
  if (d > 0.5) return f + 1
  return f % 2 === 0 ? f : f + 1
}

export function roundDecimals(x: number, decimals: number): number {
  const s = Math.pow(10, decimals)
  return rint(x * s) / s
}

export function searchsorted(a: ArrayLike<number>, v: number, side: 'left' | 'right' = 'left'): number {
  let lo = 0
  let hi = a.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (side === 'left' ? a[mid] < v : a[mid] <= v) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function interp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>, left = fp[0], right = fp[fp.length - 1]): number {
  const n = xp.length
  if (x < xp[0]) return left
  if (x > xp[n - 1]) return right
  const j = searchsorted(xp, x, 'right')
  if (j >= n) return fp[n - 1]
  if (j === 0) return fp[0]
  const x0 = xp[j - 1]
  if (x === x0) return fp[j - 1]
  const slope = (fp[j] - fp[j - 1]) / (xp[j] - x0)
  return slope * (x - x0) + fp[j - 1]
}

export function sumF32(a: ArrayLike<number>, start: number, n: number): number {
  if (n < 8) {
    let res = 0
    for (let i = 0; i < n; i++) res = Math.fround(res + a[start + i])
    return res
  }
  if (n <= 128) {
    const r = new Float64Array(8)
    for (let j = 0; j < 8; j++) r[j] = a[start + j]
    let i = 8
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = Math.fround(r[j] + a[start + i + j])
    let res = Math.fround(
      Math.fround(Math.fround(r[0] + r[1]) + Math.fround(r[2] + r[3])) + Math.fround(Math.fround(r[4] + r[5]) + Math.fround(r[6] + r[7])),
    )
    for (; i < n; i++) res = Math.fround(res + a[start + i])
    return res
  }
  let n2 = Math.floor(n / 2)
  n2 -= n2 % 8
  return Math.fround(sumF32(a, start, n2) + sumF32(a, start + n2, n - n2))
}

export function meanF32(a: ArrayLike<number>, start: number, n: number): number {
  return Math.fround(sumF32(a, start, n) / n)
}

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

export function roundFloat16(x: number): number {
  f32[0] = x
  const bits = u32[0]
  const sign = bits & 0x80000000 ? -1 : 1
  const exp = (bits >>> 23) & 0xff
  const mant = bits & 0x7fffff
  if (exp === 0xff) return f32[0]
  const e = exp - 127
  if (e >= -14) {
    if (e > 15) return sign * Infinity
    let q = mant >>> 13
    const rem = mant & 0x1fff
    if (rem > 0x1000 || (rem === 0x1000 && (q & 1))) q++
    let ee = e
    if (q === 0x400) {
      q = 0
      ee++
      if (ee > 15) return sign * Infinity
    }
    return sign * (1 + q / 1024) * Math.pow(2, ee)
  }
  const v = Math.abs(f32[0]) / Math.pow(2, -24)
  return sign * rint(v) * Math.pow(2, -24)
}

export function toFloat16(a: ArrayLike<number>): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = roundFloat16(a[i])
  return out
}

function besselI0(x: number): number {
  let sum = 1
  let term = 1
  const h = x / 2
  for (let k = 1; k < 200; k++) {
    term *= (h / k) * (h / k)
    sum += term
    if (term < sum * 1e-17) break
  }
  return sum
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b]
  return a
}

type Polyphase = { L: number; M: number; taps: Float64Array[]; offset: Int32Array }

const polyCache = new Map<string, Polyphase>()

const PASSBAND = 0.9137
const ATTENUATION = 136

function designPolyphase(from: number, to: number): Polyphase {
  const g = gcd(from, to)
  const L = to / g
  const M = from / g
  const fh = L * from
  const nyq = Math.min(from, to) / 2
  const fc = ((PASSBAND + 1) / 2) * nyq
  const tr = (1 - PASSBAND) * nyq
  const beta = 0.1102 * (ATTENUATION - 8.7)
  const dw = (2 * Math.PI * tr) / fh
  let N = Math.ceil((ATTENUATION - 8) / (2.285 * dw))
  if (N % 2 === 0) N++
  const c = (N - 1) >> 1
  const h = new Float64Array(N)
  const i0b = besselI0(beta)
  const scale = (2 * fc) / fh
  for (let j = -c; j <= c; j++) {
    const x = (2 * fc * j) / fh
    const sinc = j === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)
    const r = j / c
    const w = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0b
    h[j + c] = L * scale * sinc * w
  }
  const taps: Float64Array[] = []
  const offset = new Int32Array(L)
  for (let p = 0; p < L; p++) {
    const d = Math.ceil((p - c) / L)
    offset[p] = d
    const j0 = p - d * L
    const count = Math.floor((j0 + c) / L) + 1
    const row = new Float64Array(count)
    for (let k = 0; k < count; k++) row[k] = h[j0 - k * L + c]
    taps.push(row)
  }
  return { L, M, taps, offset }
}

export function resampleLength(n: number, from: number, to: number): number {
  return Math.ceil((n * to) / from)
}

export function resample(x: ArrayLike<number>, from: number, to: number): Float32Array {
  if (from === to) return Float32Array.from(x as ArrayLike<number>)
  const key = `${from}>${to}`
  let poly = polyCache.get(key)
  if (!poly) {
    poly = designPolyphase(from, to)
    polyCache.set(key, poly)
  }
  const { L, M, taps, offset } = poly
  const nIn = x.length
  const nOut = resampleLength(nIn, from, to)
  const out = new Float32Array(nOut)
  for (let n = 0; n < nOut; n++) {
    const pos = n * M
    const p = pos % L
    const q = (pos - p) / L
    const row = taps[p]
    const i0 = q + offset[p]
    const kStart = Math.max(0, -i0)
    const kEnd = Math.min(row.length, nIn - i0)
    let acc = 0
    for (let k = kStart; k < kEnd; k++) acc += row[k] * x[i0 + k]
    out[n] = acc
  }
  return out
}
