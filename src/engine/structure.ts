const FPS = 50
const MEL_FEATURES = 320
const CONTEXT = 16
const INTRO_EASE = 6

export type Sustain = [number, number]
export type Segment = { start: number; end: number; lag: number; lagTokens: number }
export type MusicBounds = { musicStart: number; musicEnd: number | null; introStem: number }

function medianPositive(x: Float32Array): number {
  const v = Array.from(x).filter((e) => e > 0)
  if (!v.length) return NaN
  v.sort((a, b) => a - b)
  const n = v.length
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
}

export function musicBounds(rms: Float32Array, frames: number): MusicBounds {
  const e = new Float32Array(frames)
  for (let s = 0; s < 4; s++) for (let t = 0; t < frames; t++) e[t] += rms[s * frames + t]
  const threshold = 0.1 * medianPositive(e)
  let first = -1
  let last = -1
  for (let t = 0; t < frames; t++) {
    if (e[t] > threshold) {
      if (first < 0) first = t
      last = t
    }
  }
  const musicStart = first < 0 ? 0 : first / FPS
  const a = Math.trunc(musicStart * FPS)
  const b = Math.min(frames, a + INTRO_EASE * FPS)
  let introStem = 0
  if (b > a) {
    let best = -Infinity
    for (let s = 0; s < 4; s++) {
      let sum = 0
      for (let t = a; t < b; t++) sum += rms[s * frames + t]
      if (sum > best) {
        best = sum
        introStem = s
      }
    }
  }
  return { musicStart, musicEnd: last < 0 ? null : last / FPS, introStem }
}

export function findPeaks(x: Float32Array, height: number, distance: number): number[] {
  const n = x.length
  const peaks: number[] = []
  let i = 1
  while (i < n - 1) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1
      while (ahead < n - 1 && x[ahead] === x[i]) ahead++
      if (x[ahead] < x[i]) {
        const mid = Math.floor((i + ahead - 1) / 2)
        if (x[mid] >= height) peaks.push(mid)
        i = ahead
        continue
      }
    }
    i++
  }
  const keep = new Uint8Array(peaks.length).fill(1)
  const order = peaks.map((_, j) => j).sort((a, b) => x[peaks[b]] - x[peaks[a]] || a - b)
  for (const j of order) {
    if (!keep[j]) continue
    for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < distance; k--) keep[k] = 0
    for (let k = j + 1; k < peaks.length && peaks[k] - peaks[j] < distance; k++) keep[k] = 0
  }
  return peaks.filter((_, j) => keep[j])
}

export function sustains(periodicity: Float32Array, rms: Float32Array, onset: Float32Array, syllableTimes: number[], frames: number, minVocal = 0.6, minOther = 0.8): Sustain[] {
  const out: Sustain[] = []
  const vocal = new Uint8Array(frames)
  for (let t = 0; t < frames; t++) vocal[t] = periodicity[t] > 0.5 && rms[3 * frames + t] > 0.02 ? 1 : 0
  const syl = [...syllableTimes].sort((a, b) => a - b)
  for (let i = 0; i < syl.length; i++) {
    const a = syl[i]
    const b = i + 1 < syl.length ? syl[i + 1] : syl[syl.length - 1] + 10
    const f0 = Math.trunc(a * FPS)
    const f1 = Math.min(frames, Math.trunc(Math.min(b, a + 6) * FPS))
    const length = Math.max(0, f1 - f0)
    let held = length
    for (let f = f0; f < f1; f++) {
      if (!vocal[f]) {
        held = f - f0
        break
      }
    }
    if (held / FPS >= minVocal) out.push([a, a + held / FPS])
  }
  const other = rms.subarray(2 * frames, 3 * frames)
  const med = medianPositive(other)
  const loud = new Uint8Array(frames)
  if (!Number.isNaN(med)) for (let t = 0; t < frames; t++) loud[t] = other[t] > 0.5 * med ? 1 : 0
  const p = new Float32Array(frames)
  let mean = 0
  for (let t = 0; t < frames; t++) {
    p[t] = Math.max(onset[2 * frames + t], 0)
    mean += p[t]
  }
  mean /= frames
  let variance = 0
  for (let t = 0; t < frames; t++) variance += (p[t] - mean) ** 2
  const edges = [...findPeaks(p, mean + 1.5 * Math.sqrt(variance / frames), 5), frames]
  let start = 0
  for (const e of edges) {
    if (e > start) {
      let sum = 0
      for (let t = start; t < e; t++) sum += loud[t]
      if (sum / (e - start) > 0.9 && (e - start) / FPS >= minOther) out.push([start / FPS, e / FPS])
    }
    start = e
  }
  return out.sort((a, b) => a[0] - b[0] || a[1] - b[1])
}

export function similarityMatrix(features: Float32Array, tokens: number, width: number, context = CONTEXT): Float32Array {
  const T = tokens
  const mel = new Float64Array(T * MEL_FEATURES)
  const mean = new Float64Array(MEL_FEATURES)
  for (let i = 0; i < T; i++) for (let j = 0; j < MEL_FEATURES; j++) mean[j] += features[i * width + j]
  for (let j = 0; j < MEL_FEATURES; j++) mean[j] /= T
  for (let i = 0; i < T; i++) for (let j = 0; j < MEL_FEATURES; j++) mel[i * MEL_FEATURES + j] = features[i * width + j] - mean[j]
  const dim = 2 * MEL_FEATURES
  const desc = new Float64Array(T * dim)
  const window = 2 * context + 1
  const wDesc = Math.sqrt(0.7)
  const wLocal = Math.sqrt(0.3)
  const acc = new Float64Array(MEL_FEATURES)
  for (let i = 0; i < T; i++) {
    acc.fill(0)
    for (let m = Math.max(0, i - context); m <= Math.min(T - 1, i + context); m++) for (let j = 0; j < MEL_FEATURES; j++) acc[j] += mel[m * MEL_FEATURES + j]
    let normDesc = 0
    let normLocal = 0
    for (let j = 0; j < MEL_FEATURES; j++) {
      acc[j] /= window
      normDesc += acc[j] * acc[j]
      normLocal += mel[i * MEL_FEATURES + j] ** 2
    }
    normDesc = Math.sqrt(normDesc) + 1e-6
    normLocal = Math.sqrt(normLocal) + 1e-6
    for (let j = 0; j < MEL_FEATURES; j++) {
      desc[i * dim + j] = (wDesc * acc[j]) / normDesc
      desc[i * dim + MEL_FEATURES + j] = (wLocal * mel[i * MEL_FEATURES + j]) / normLocal
    }
  }
  const S = new Float32Array(T * T)
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      let dot = 0
      const a = i * dim
      const b = j * dim
      for (let c = 0; c < dim; c++) dot += desc[a + c] * desc[b + c]
      S[i * T + j] = dot
      S[j * T + i] = dot
    }
  }
  return S
}

export function repeatSegments(features: Float32Array, tokens: number, width: number, times: ArrayLike<number>, minSeconds = 4, threshold = 0.88, hole = 6): Segment[] {
  const T = tokens
  const S = similarityMatrix(features, T, width)
  const bestLen = new Float64Array(T)
  const bestLag = new Int32Array(T)
  let minLagTok = 0
  while (minLagTok < T && times[minLagTok] < times[0] + minSeconds) minLagTok++
  for (let L = Math.max(1, minLagTok); L < T; L++) {
    const idx: number[] = []
    for (let k = 0; k + L < T; k++) if (S[(k + L) * T + k] >= threshold) idx.push(k)
    if (idx.length < 8) continue
    let a = 0
    for (let i = 1; i <= idx.length; i++) {
      if (i < idx.length && idx[i] - idx[i - 1] <= hole + 1) continue
      const k0 = idx[a] + L
      const k1 = idx[i - 1] + L
      a = i
      const length = times[k1] - times[k0]
      if (length < minSeconds) continue
      for (let k = k0; k <= k1; k++) {
        if (length > bestLen[k] + 1e-6) {
          bestLen[k] = length
          bestLag[k] = L
        }
      }
    }
  }
  const out: Segment[] = []
  let k = 0
  while (k < T) {
    if (bestLag[k] === 0) {
      k++
      continue
    }
    const L = bestLag[k]
    let j = k
    while (j + 1 < T && bestLag[j + 1] === L) j++
    if (times[j] - times[k] >= minSeconds) out.push({ start: times[k], end: times[j], lag: times[k] - times[k - L], lagTokens: L })
    k = j + 1
  }
  return out
}
