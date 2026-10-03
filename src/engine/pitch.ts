import { Rng } from './rng'
import { ort } from './models'
import { crepeFrames, CREPE_WINDOW, HOP } from './features'

export const PITCH_BINS = 360
export const CENTS_PER_BIN = 20
export const CENTS_OFFSET = 1997.3794084376191
export const MIN_BIN = 62
export const MAX_BIN = 308
export const CREPE_BATCH = 64
export const VITERBI_BATCH = 256
const TRANSITION_WIDTH = 12
const LOG_TINY = Math.log(1.1754943508222875e-38)

export type PitchTrack = {
  bins: Int32Array
  midi: Float32Array
  periodicity: Float32Array
}

function logTransition(): Float64Array {
  const table = new Float64Array(PITCH_BINS * (2 * TRANSITION_WIDTH - 1))
  for (let k = 0; k < PITCH_BINS; k++) {
    let sum = 0
    for (let j = 0; j < PITCH_BINS; j++) sum += Math.max(TRANSITION_WIDTH - Math.abs(k - j), 0)
    for (let o = -TRANSITION_WIDTH + 1; o < TRANSITION_WIDTH; o++) {
      table[k * (2 * TRANSITION_WIDTH - 1) + o + TRANSITION_WIDTH - 1] = Math.log((TRANSITION_WIDTH - Math.abs(o)) / sum + 1.1754943508222875e-38)
    }
  }
  return table
}

const LOG_TRANS = logTransition()

function logSoftmax(probs: Float32Array, frame: number, out: Float64Array): void {
  const base = frame * PITCH_BINS
  let max = -Infinity
  for (let b = MIN_BIN; b < MAX_BIN; b++) if (probs[base + b] > max) max = probs[base + b]
  let sum = 0
  const e = new Float32Array(PITCH_BINS)
  for (let b = MIN_BIN; b < MAX_BIN; b++) {
    e[b] = Math.fround(Math.exp(Math.fround(probs[base + b] - max)))
    sum = Math.fround(sum + e[b])
  }
  for (let b = 0; b < PITCH_BINS; b++) {
    const p = b >= MIN_BIN && b < MAX_BIN ? Math.fround(e[b] / sum) : 0
    out[b] = p > 0 ? Math.log(p) : LOG_TINY
  }
}

export function viterbiBatch(probs: Float32Array, start: number, length: number, bins: Int32Array): void {
  const W = 2 * TRANSITION_WIDTH - 1
  const value = new Float64Array(length * PITCH_BINS)
  const ptr = new Uint16Array(length * PITCH_BINS)
  const lp = new Float64Array(PITCH_BINS)
  const logInit = Math.log(1 / PITCH_BINS + 1.1754943508222875e-38)
  logSoftmax(probs, start, lp)
  for (let b = 0; b < PITCH_BINS; b++) value[b] = lp[b] + logInit
  for (let t = 1; t < length; t++) {
    logSoftmax(probs, start + t, lp)
    const prev = (t - 1) * PITCH_BINS
    const cur = t * PITCH_BINS
    let farK = 0
    for (let k = 1; k < PITCH_BINS; k++) if (value[prev + k] > value[prev + farK]) farK = k
    const farCost = value[prev + farK] + LOG_TINY
    for (let j = 0; j < PITCH_BINS; j++) {
      let best = -Infinity
      let bestK = 0
      const k0 = Math.max(0, j - TRANSITION_WIDTH + 1)
      const k1 = Math.min(PITCH_BINS - 1, j + TRANSITION_WIDTH - 1)
      for (let k = k0; k <= k1; k++) {
        const cost = value[prev + k] + LOG_TRANS[k * W + (j - k) + TRANSITION_WIDTH - 1]
        if (cost > best) {
          best = cost
          bestK = k
        }
      }
      if (farCost > best || (farCost === best && farK < bestK)) {
        best = farCost
        bestK = farK
      }
      value[cur + j] = lp[j] + best
      ptr[cur + j] = bestK
    }
  }
  const last = (length - 1) * PITCH_BINS
  let state = 0
  for (let b = 1; b < PITCH_BINS; b++) if (value[last + b] > value[last + state]) state = b
  bins[start + length - 1] = state
  for (let t = length - 2; t >= 0; t--) {
    state = ptr[(t + 1) * PITCH_BINS + state]
    bins[start + t] = state
  }
}

export function decodeBins(probs: Float32Array, totalFrames: number): Int32Array {
  const bins = new Int32Array(totalFrames)
  for (let start = 0; start < totalFrames; start += VITERBI_BATCH) viterbiBatch(probs, start, Math.min(VITERBI_BATCH, totalFrames - start), bins)
  return bins
}

export function medianFilter3(x: Float32Array): Float32Array {
  const n = x.length
  const out = Float32Array.from(x)
  if (n < 2) return out
  for (let i = 1; i < n - 1; i++) {
    const a = x[i - 1]
    const b = x[i]
    const c = x[i + 1]
    out[i] = Math.max(Math.min(a, b), Math.min(Math.max(a, b), c))
  }
  out[0] = Math.min(x[0], x[1])
  out[n - 1] = Math.min(x[n - 2], x[n - 1])
  return out
}

export function binsToMidi(bins: Int32Array, rng?: Rng): Float32Array {
  const midi = new Float32Array(bins.length)
  for (let i = 0; i < bins.length; i++) {
    const dither = rng ? (rng.next() + rng.next() - 1) * CENTS_PER_BIN : 0
    const cents = CENTS_PER_BIN * bins[i] + CENTS_OFFSET + dither
    const hz = 10 * Math.pow(2, cents / 1200)
    midi[i] = 69 + 12 * Math.log2(Math.max(hz, 1e-3) / 440)
  }
  return midi
}

export function decodePitch(probs: Float32Array, totalFrames: number, frames: number, rng?: Rng): PitchTrack {
  const bins = decodeBins(probs, totalFrames)
  const raw = new Float32Array(totalFrames)
  for (let t = 0; t < totalFrames; t++) raw[t] = probs[t * PITCH_BINS + bins[t]]
  const periodicity = fit(medianFilter3(raw), frames)
  const midi = fit(binsToMidi(bins, rng), frames)
  return { bins: fitInt(bins, frames), midi, periodicity }
}

export function vocalActive(periodicity: Float32Array, vocalRms: Float32Array, frames: number): Uint8Array {
  const active = new Uint8Array(frames)
  for (let t = 0; t < frames; t++) active[t] = periodicity[t] > 0.5 && vocalRms[t] > 0.02 ? 1 : 0
  return active
}

export function maskPitch(midi: Float32Array, active: Uint8Array): Float32Array {
  const out = new Float32Array(midi.length)
  for (let t = 0; t < midi.length; t++) out[t] = active[t] ? midi[t] : NaN
  return out
}

function fit(a: Float32Array, frames: number): Float32Array {
  if (a.length >= frames) return a.slice(0, frames)
  const out = new Float32Array(frames)
  out.set(a)
  out.fill(a[a.length - 1], a.length)
  return out
}

function fitInt(a: Int32Array, frames: number): Int32Array {
  if (a.length >= frames) return a.slice(0, frames)
  const out = new Int32Array(frames)
  out.set(a)
  out.fill(a[a.length - 1], a.length)
  return out
}

export async function crepeProbs(session: ort.InferenceSession, vocal16: Float32Array, onProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<{ probs: Float32Array; totalFrames: number }> {
  const totalFrames = Math.floor(vocal16.length / HOP) + 1
  const probs = new Float32Array(totalFrames * PITCH_BINS)
  for (let start = 0; start < totalFrames; start += CREPE_BATCH) {
    if (signal?.aborted) throw new DOMException('cancelled', 'AbortError')
    const count = Math.min(CREPE_BATCH, totalFrames - start)
    const res = await session.run({ frames: new ort.Tensor('float32', crepeFrames(vocal16, start, count), [count, CREPE_WINDOW]) })
    probs.set((await res.probs.getData()) as Float32Array, start * PITCH_BINS)
    for (const t of Object.values(res)) t.dispose?.()
    onProgress?.(Math.min(1, (start + count) / totalFrames))
  }
  return { probs, totalFrames }
}
