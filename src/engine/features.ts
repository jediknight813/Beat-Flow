import { findPeaks, melSpectrogram, onsetStrength, resample, rms, roundDecimals, toFloat16, zscore } from './dsp'
import { syllables, type Syllable } from './vocals'

export const FPS = 50
export const HOP = 320
export const SR = 16000
export const MELS = 80
export const STEMS = 4
export const ATTACK_HOP = 64
export const BEAT_SR = 22050
export const BEAT_HOP = 441
export const BEAT_MELS = 128
export const CREPE_WINDOW = 1024

export type StemFeatures = { frames: number; mel: Float32Array; onset: Float32Array; rms: Float32Array }

export function frameTotal(samples: number): number {
  return Math.ceil(samples / HOP)
}

export function fit(a: Float32Array, frames: number): Float32Array {
  if (a.length >= frames) return a.slice(0, frames)
  const out = new Float32Array(frames)
  out.set(a)
  out.fill(a.length ? a[a.length - 1] : 0, a.length)
  return out
}

export function logMel(x: Float32Array, frames: number): Float32Array {
  const S = melSpectrogram(x, { nFft: 1024, hop: HOP, winLength: 400 }, { sr: SR, nFft: 1024, nMels: MELS, fMin: 0, fMax: 8000, norm: 'slaney' })
  const out = new Float32Array(MELS * frames)
  const T = S.cols
  for (let m = 0; m < MELS; m++) {
    const row = out.subarray(m * frames, (m + 1) * frames)
    for (let t = 0; t < frames; t++) row[t] = Math.log(S.data[m * T + Math.min(t, T - 1)] + 1e-6)
  }
  return out
}

export function onsetEnvelope(x: Float32Array): Float32Array {
  return onsetStrength(x, { sr: SR, hop: HOP })
}

export function frameRms(x: Float32Array, frames: number): Float32Array {
  return fit(rms(x, 1024, HOP), frames)
}

export function stemFeatures(stems: Float32Array[], mix16: Float32Array): StemFeatures {
  const frames = frameTotal(mix16.length)
  const mixRms = frameRms(mix16, frames)
  let peak = 0
  for (const v of mixRms) if (v > peak) peak = v
  const scale = peak + 1e-8
  const mel = new Float32Array(STEMS * MELS * frames)
  const onset = new Float32Array(STEMS * frames)
  const level = new Float32Array(STEMS * frames)
  stems.forEach((s, i) => {
    mel.set(logMel(s, frames), i * MELS * frames)
    onset.set(zscore(fit(onsetEnvelope(s), frames)), i * frames)
    const r = frameRms(s, frames)
    for (let t = 0; t < frames; t++) level[i * frames + t] = r[t] / scale
  })
  return { frames, mel, onset, rms: level }
}

export function storedFeatures(f: StemFeatures): StemFeatures {
  return { frames: f.frames, mel: toFloat16(f.mel), onset: toFloat16(f.onset), rms: toFloat16(f.rms) }
}

export function attackEnvelope(pcm16: Float32Array): Float32Array {
  return zscore(onsetStrength(pcm16, { sr: SR, hop: ATTACK_HOP, nFft: 512, lag: 1, maxSize: 3 }))
}

export function attacks(pcm16: Float32Array): number[] {
  const distance = Math.trunc((0.06 * SR) / ATTACK_HOP)
  return findPeaks(attackEnvelope(pcm16), 2, distance).map((p) => roundDecimals((p * ATTACK_HOP) / SR, 4))
}

export function beatSpectrogram(mix16: Float32Array): { frames: number; data: Float32Array } {
  const y = resample(mix16, SR, BEAT_SR)
  const S = melSpectrogram(
    y,
    { nFft: 1024, hop: BEAT_HOP, padMode: 'reflect' },
    { sr: BEAT_SR, nFft: 1024, nMels: BEAT_MELS, fMin: 30, fMax: 11000, norm: null },
    1,
  )
  const F = S.cols
  const data = new Float32Array(F * BEAT_MELS)
  for (let m = 0; m < BEAT_MELS; m++) for (let t = 0; t < F; t++) data[t * BEAT_MELS + m] = Math.log1p(1000 * (S.data[m * F + t] / 32))
  return { frames: F, data }
}

export function crepeFrames(vocal16: Float32Array, start = 0, count = Math.floor(vocal16.length / HOP) + 1 - start): Float32Array {
  const out = new Float32Array(count * CREPE_WINDOW)
  const n = vocal16.length
  const half = CREPE_WINDOW / 2
  for (let f = 0; f < count; f++) {
    const base = (start + f) * HOP - half
    const row = out.subarray(f * CREPE_WINDOW, (f + 1) * CREPE_WINDOW)
    for (let j = 0; j < CREPE_WINDOW; j++) {
      const i = base + j
      row[j] = i >= 0 && i < n ? vocal16[i] : 0
    }
    let sum = 0
    for (let j = 0; j < CREPE_WINDOW; j++) sum += row[j]
    const m = sum / CREPE_WINDOW
    let ss = 0
    for (let j = 0; j < CREPE_WINDOW; j++) ss += (row[j] - m) * (row[j] - m)
    const sd = Math.max(1e-10, Math.sqrt(ss / (CREPE_WINDOW - 1)))
    for (let j = 0; j < CREPE_WINDOW; j++) row[j] = (row[j] - m) / sd
  }
  return out
}

export function vocalSyllables(vocal16: Float32Array, active: Uint8Array): Syllable[] {
  return syllables(onsetEnvelope(vocal16), active)
}
