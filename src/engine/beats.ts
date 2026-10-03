import { ort } from './models'

export const BEAT_FPS = 50
export const CHUNK_FRAMES = 1500
export const CHUNK_BORDER = 6
export const MEL_BINS = 128
const EMPTY_LOGIT = -1000

export type Logits = { beat: Float32Array; downbeat: Float32Array }
export type LogitsProvider = (chunk: Float32Array, frames: number) => Promise<Float32Array[]>

export function chunkStarts(frames: number): number[] {
  const stride = CHUNK_FRAMES - 2 * CHUNK_BORDER
  const starts: number[] = []
  for (let s = -CHUNK_BORDER; s < frames - CHUNK_BORDER; s += stride) starts.push(s)
  if (frames > stride) starts[starts.length - 1] = frames - (CHUNK_FRAMES - CHUNK_BORDER)
  return starts
}

export function sliceChunk(spect: Float32Array, frames: number, bins: number, start: number): { chunk: Float32Array; frames: number } {
  const lo = Math.max(start, 0)
  const hi = Math.min(start + CHUNK_FRAMES, frames)
  const left = Math.max(0, -start)
  const right = Math.max(0, Math.min(CHUNK_BORDER, start + CHUNK_FRAMES - frames))
  const length = left + (hi - lo) + right
  const chunk = new Float32Array(length * bins)
  chunk.set(spect.subarray(lo * bins, hi * bins), left * bins)
  return { chunk, frames: length }
}

export function stitchChunks(chunks: Float32Array[][], starts: number[], frames: number): Logits {
  const beat = new Float32Array(frames).fill(EMPTY_LOGIT)
  const downbeat = new Float32Array(frames).fill(EMPTY_LOGIT)
  for (let c = chunks.length - 1; c >= 0; c--) {
    const [b, d] = chunks[c]
    const start = starts[c] + CHUNK_BORDER
    const end = Math.min(starts[c] + CHUNK_FRAMES - CHUNK_BORDER, frames)
    for (let t = start, i = CHUNK_BORDER; t < end; t++, i++) {
      beat[t] = b[i]
      downbeat[t] = d[i]
    }
  }
  return { beat, downbeat }
}

export async function framewiseLogits(spect: Float32Array, frames: number, bins: number, run: LogitsProvider): Promise<Logits> {
  const starts = chunkStarts(frames)
  const chunks: Float32Array[][] = []
  for (const start of starts) {
    const piece = sliceChunk(spect, frames, bins, start)
    chunks.push(await run(piece.chunk, piece.frames))
  }
  return stitchChunks(chunks, starts, frames)
}

export function peakFrames(logits: Float32Array): number[] {
  const n = logits.length
  const peaks: number[] = []
  for (let t = 0; t < n; t++) {
    const x = logits[t]
    if (!(x > 0)) continue
    let max = -Infinity
    for (let u = Math.max(0, t - 3); u <= Math.min(n - 1, t + 3); u++) if (logits[u] > max) max = logits[u]
    if (x === max) peaks.push(t)
  }
  return peaks
}

export function deduplicatePeaks(peaks: number[], width = 1): number[] {
  if (!peaks.length) return []
  const out: number[] = []
  let p = peaks[0]
  let c = 1
  for (let i = 1; i < peaks.length; i++) {
    const p2 = peaks[i]
    if (p2 - p <= width) {
      c += 1
      p += (p2 - p) / c
    } else {
      out.push(p)
      p = p2
      c = 1
    }
  }
  out.push(p)
  return out
}

export function postprocessBeats(logits: Logits): { beats: number[]; downbeats: number[] } {
  const beats = deduplicatePeaks(peakFrames(logits.beat)).map((f) => f / BEAT_FPS)
  let downbeats = deduplicatePeaks(peakFrames(logits.downbeat)).map((f) => f / BEAT_FPS)
  if (beats.length) {
    downbeats = downbeats.map((d) => {
      let best = 0
      for (let i = 1; i < beats.length; i++) if (Math.abs(beats[i] - d) < Math.abs(beats[best] - d)) best = i
      return beats[best]
    })
    downbeats = [...new Set(downbeats)].sort((a, b) => a - b)
  }
  return { beats, downbeats }
}

export async function detectBeats(spect: Float32Array, frames: number, run: LogitsProvider): Promise<{ beats: number[]; downbeats: number[] }> {
  return postprocessBeats(await framewiseLogits(spect, frames, MEL_BINS, run))
}

export function beatThisProvider(session: ort.InferenceSession): LogitsProvider {
  return async (chunk, frames) => {
    const res = await session.run({ spect: new ort.Tensor('float32', chunk, [1, frames, MEL_BINS]) })
    const beat = Float32Array.from((await res.beat.getData()) as Float32Array)
    const downbeat = Float32Array.from((await res.downbeat.getData()) as Float32Array)
    for (const t of Object.values(res)) t.dispose?.()
    return [beat, downbeat]
  }
}
