import { ort } from './models'
import { hann, realFFT, resample } from './dsp'
import { SR } from './features'

export const DEMUCS_SR = 44100
export const SOURCES = ['drums', 'bass', 'other', 'vocals'] as const
const NFFT = 4096
const HOP = 1024
const SEGMENT = 343980
const STRIDE = 257985
const FRAMES = Math.ceil(SEGMENT / HOP)
const PAD = (HOP / 2) * 3
const BINS = NFFT / 2
const PADDED = PAD + FRAMES * HOP + PAD
const SCALE = Math.sqrt(NFFT)

const WINDOW = hann(NFFT)

class ComplexFFT {
  readonly n: number
  private readonly rev: Uint32Array
  private readonly cos: Float64Array
  private readonly sin: Float64Array

  constructor(n: number) {
    this.n = n
    let bits = 0
    while (1 << bits < n) bits++
    this.rev = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      let r = 0
      for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b)
      this.rev[i] = r
    }
    this.cos = new Float64Array(n / 2)
    this.sin = new Float64Array(n / 2)
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n)
      this.sin[i] = Math.sin((2 * Math.PI * i) / n)
    }
  }

  inverse(re: Float64Array, im: Float64Array): void {
    const n = this.n
    for (let i = 0; i < n; i++) {
      const r = this.rev[i]
      if (r > i) {
        let t = re[i]
        re[i] = re[r]
        re[r] = t
        t = im[i]
        im[i] = im[r]
        im[r] = t
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1
      const step = n / size
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const c = this.cos[k * step]
          const s = this.sin[k * step]
          const a = start + k
          const b = a + half
          const tr = re[b] * c - im[b] * s
          const ti = re[b] * s + im[b] * c
          re[b] = re[a] - tr
          im[b] = im[a] - ti
          re[a] += tr
          im[a] += ti
        }
      }
    }
  }
}

const IFFT = new ComplexFFT(NFFT)

function reflect(i: number, n: number): number {
  if (i < 0) return -i
  if (i >= n) return 2 * (n - 1) - i
  return i
}

export function stftCac(chunk: Float32Array[]): Float32Array {
  const out = new Float32Array(4 * BINS * FRAMES)
  const fft = realFFT(NFFT)
  const buf = new Float64Array(NFFT)
  const re = new Float64Array(BINS + 1)
  const im = new Float64Array(BINS + 1)
  const outer = PADDED + NFFT
  chunk.forEach((x, c) => {
    const padded = new Float64Array(outer)
    for (let i = 0; i < outer; i++) padded[i] = x[reflect(reflect(i - NFFT / 2, PADDED) - PAD, SEGMENT)]
    for (let f = 0; f < FRAMES; f++) {
      const base = (f + 2) * HOP
      for (let j = 0; j < NFFT; j++) buf[j] = padded[base + j] * WINDOW[j]
      fft.forward(buf, re, im)
      const reRow = (2 * c) * BINS * FRAMES
      const imRow = (2 * c + 1) * BINS * FRAMES
      for (let k = 0; k < BINS; k++) {
        out[reRow + k * FRAMES + f] = re[k] / SCALE
        out[imRow + k * FRAMES + f] = im[k] / SCALE
      }
    }
  })
  return out
}

const TOTAL = NFFT + HOP * (FRAMES + 3)
const ENVELOPE = (() => {
  const env = new Float64Array(TOTAL)
  for (let j = 0; j < FRAMES + 4; j++) for (let n = 0; n < NFFT; n++) env[HOP * j + n] += WINDOW[n] * WINDOW[n]
  return env
})()

function istftPair(specs: (Float64Array | null)[]): Float64Array[] {
  const ys = specs.map(() => new Float64Array(TOTAL))
  const env = ENVELOPE
  const re = new Float64Array(NFFT)
  const im = new Float64Array(NFFT)
  for (let f = 0; f < FRAMES; f++) {
    re.fill(0)
    im.fill(0)
    const [a, b] = specs
    for (let k = 0; k < BINS; k++) {
      const ar = a ? a[(2 * k) * FRAMES + f] : 0
      const ai = a && k ? a[(2 * k + 1) * FRAMES + f] : 0
      const br = b ? b[(2 * k) * FRAMES + f] : 0
      const bi = b && k ? b[(2 * k + 1) * FRAMES + f] : 0
      re[k] = ar - bi
      im[k] = ai + br
      if (k > 0) {
        re[NFFT - k] = ar + bi
        im[NFFT - k] = br - ai
      }
    }
    IFFT.inverse(re, im)
    const base = HOP * (f + 2)
    const gain = SCALE / NFFT
    for (let n = 0; n < NFFT; n++) {
      const w = WINDOW[n] * gain
      ys[0][base + n] += re[n] * w
      if (ys[1]) ys[1][base + n] += im[n] * w
    }
  }
  const offset = NFFT / 2 + PAD
  return ys.map((y) => {
    const out = new Float64Array(SEGMENT)
    for (let n = 0; n < SEGMENT; n++) out[n] = y[offset + n] / env[offset + n]
    return out
  })
}

function monoSpec(specOut: Float32Array, source: number): Float64Array {
  const plane = BINS * FRAMES
  const base = source * 4 * plane
  const out = new Float64Array(2 * plane)
  for (let k = 0; k < BINS; k++) {
    for (let f = 0; f < FRAMES; f++) {
      const i = k * FRAMES + f
      out[(2 * k) * FRAMES + f] = 0.5 * (specOut[base + i] + specOut[base + 2 * plane + i])
      out[(2 * k + 1) * FRAMES + f] = 0.5 * (specOut[base + plane + i] + specOut[base + 3 * plane + i])
    }
  }
  return out
}

export async function separate(
  session: ort.InferenceSession,
  stereo: [Float32Array, Float32Array],
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<Float32Array[]> {
  const [left, right] = stereo
  const N = left.length
  let sum = 0
  for (let i = 0; i < N; i++) sum += 0.5 * (left[i] + right[i])
  const mean = sum / N
  let ss = 0
  for (let i = 0; i < N; i++) ss += (0.5 * (left[i] + right[i]) - mean) ** 2
  const std = Math.sqrt(ss / Math.max(1, N - 1)) + 1e-8
  const out = SOURCES.map(() => new Float32Array(N))
  const weightSum = new Float32Array(N)
  const half = SEGMENT / 2
  const weight = (i: number) => (i < half ? i + 1 : SEGMENT - i) / half
  const offsets: number[] = []
  for (let o = 0; o < N; o += STRIDE) offsets.push(o)
  for (const [ci, offset] of offsets.entries()) {
    if (signal?.aborted) throw new DOMException('cancelled', 'AbortError')
    const length = Math.min(N - offset, SEGMENT)
    const delta = SEGMENT - length
    const start = offset - Math.floor(delta / 2)
    const chunk = [new Float32Array(SEGMENT), new Float32Array(SEGMENT)]
    const lo = Math.max(0, start)
    const hi = Math.min(N, start + SEGMENT)
    for (let i = lo; i < hi; i++) {
      chunk[0][i - start] = (left[i] - mean) / std
      chunk[1][i - start] = (right[i] - mean) / std
    }
    const mix = new Float32Array(2 * SEGMENT)
    mix.set(chunk[0])
    mix.set(chunk[1], SEGMENT)
    const spec = stftCac(chunk)
    const res = await session.run({
      mix: new ort.Tensor('float32', mix, [1, 2, SEGMENT]),
      spec: new ort.Tensor('float32', spec, [1, 4, BINS, FRAMES]),
    })
    const specOut = (await res.spec_out.getData()) as Float32Array
    const waveOut = (await res.wave_out.getData()) as Float32Array
    for (const t of Object.values(res)) t.dispose?.()
    const waves: Float64Array[] = []
    for (let s = 0; s < SOURCES.length; s += 2) waves.push(...istftPair([monoSpec(specOut, s), monoSpec(specOut, s + 1)]))
    const trim = Math.floor(delta / 2)
    for (let s = 0; s < SOURCES.length; s++) {
      const w0 = waveOut.subarray((2 * s) * SEGMENT, (2 * s + 1) * SEGMENT)
      const w1 = waveOut.subarray((2 * s + 1) * SEGMENT, (2 * s + 2) * SEGMENT)
      const dst = out[s]
      const wave = waves[s]
      for (let i = 0; i < length; i++) {
        const j = trim + i
        dst[offset + i] += weight(i) * (wave[j] + 0.5 * (w0[j] + w1[j]))
      }
    }
    for (let i = 0; i < length; i++) weightSum[offset + i] += weight(i)
    onProgress?.((ci + 1) / offsets.length)
  }
  return out.map((stem) => {
    for (let i = 0; i < N; i++) stem[i] = (stem[i] / weightSum[i]) * std + mean
    return resample(stem, DEMUCS_SR, SR)
  })
}
