import { MT19937 } from './mt19937'

export type Section = { start: number; end: number; label: number }

const FPS = 50
const N_MELS = 80
const N_MFCC = 13
const N_CHROMA = 12
const WIDTH = 3
const TINY32 = 1.1754943508222875e-38

export function roundHalfEven(x: number): number {
  const r = Math.round(x)
  if (Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0) return r - 1
  return r
}

function hzToMel(f: number): number {
  const fsp = 200 / 3
  const minLogHz = 1000
  const logstep = Math.log(6.4) / 27
  return f >= minLogHz ? minLogHz / fsp + Math.log(f / minLogHz) / logstep : f / fsp
}

function melToHz(m: number): number {
  const fsp = 200 / 3
  const minLogHz = 1000
  const minLogMel = minLogHz / fsp
  const logstep = Math.log(6.4) / 27
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m
}

export function melFrequencies(nMels: number, fmin: number, fmax: number): Float64Array {
  const lo = hzToMel(fmin)
  const hi = hzToMel(fmax)
  const step = (hi - lo) / (nMels - 1)
  const out = new Float64Array(nMels)
  for (let i = 0; i < nMels; i++) out[i] = melToHz(lo + i * step)
  out[nMels - 1] = melToHz(hi)
  return out
}

function chromaBands(): Int32Array {
  const freqs = melFrequencies(N_MELS, 0, 8000)
  const out = new Int32Array(N_MELS).fill(-1)
  for (let i = 0; i < N_MELS; i++) {
    const f = freqs[i]
    if (f >= 100 && f <= 5000) out[i] = ((roundHalfEven(12 * Math.log2(f / 440) + 69) % 12) + 12) % 12
  }
  return out
}

function median(values: Float64Array, n: number): number {
  const v = values.subarray(0, n)
  v.sort()
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
}

function medianOf(values: number[]): number {
  return median(Float64Array.from(values), values.length)
}

function syncMedian(feat: Float64Array, dim: number, frames: number, bounds: number[]): Float64Array {
  const segments = bounds.length - 1
  const out = new Float64Array(dim * segments)
  const buf = new Float64Array(frames)
  for (let s = 0; s < segments; s++) {
    const lo = bounds[s]
    const hi = bounds[s + 1]
    for (let c = 0; c < dim; c++) {
      for (let t = lo; t < hi; t++) buf[t - lo] = feat[c * frames + t]
      out[c * segments + s] = median(buf, hi - lo)
    }
  }
  return out
}

function reflectIndex(i: number, n: number): number {
  while (i < 0 || i >= n) {
    if (i < 0) i = -i - 1
    if (i >= n) i = 2 * n - i - 1
  }
  return i
}

function medianFilterAxis(x: Float64Array, rows: number, cols: number, size: number, alongRows: boolean): Float64Array {
  const out = new Float64Array(rows * cols)
  const half = size >> 1
  const buf = new Float64Array(size)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let w = -half; w <= half; w++) {
        const r = alongRows ? reflectIndex(i + w, rows) : i
        const c = alongRows ? j : reflectIndex(j + w, cols)
        buf[w + half] = x[r * cols + c]
      }
      out[i * cols + j] = median(buf, size)
    }
  }
  return out
}

function recurrenceAffinity(tonal: Float64Array, dim: number, t: number): Float64Array {
  const k = 2 * Math.ceil(Math.sqrt(t - 2 * WIDTH + 1))
  const neighbours = Math.min(t - 1, k + 2 * WIDTH)
  const dist = new Float64Array(t * t)
  for (let i = 0; i < t; i++) {
    for (let j = i + 1; j < t; j++) {
      let d = 0
      for (let c = 0; c < dim; c++) {
        const diff = tonal[c * t + i] - tonal[c * t + j]
        d += diff * diff
      }
      d = Math.sqrt(d)
      dist[i * t + j] = d
      dist[j * t + i] = d
    }
  }
  const links = new Float64Array(t * t)
  const order = new Int32Array(t)
  for (let i = 0; i < t; i++) {
    let n = 0
    for (let j = 0; j < t; j++) if (j !== i) order[n++] = j
    const row = order.subarray(0, n)
    row.sort((a, b) => dist[i * t + a] - dist[i * t + b] || a - b)
    const kept: number[] = []
    for (let m = 0; m < neighbours; m++) {
      const j = row[m]
      if (Math.abs(i - j) < WIDTH) continue
      const d = dist[i * t + j]
      if (d === 0) continue
      kept.push(j)
    }
    kept.sort((a, b) => dist[i * t + a] - dist[i * t + b] || a - b)
    for (let m = 0; m < Math.min(k, kept.length); m++) links[i * t + kept[m]] = dist[i * t + kept[m]]
  }
  const rec = new Float64Array(t * t)
  const distToK = new Float64Array(t)
  const kth: number[] = []
  for (let i = 0; i < t; i++) {
    const row: number[] = []
    for (let j = 0; j < t; j++) {
      if (links[i * t + j] > 0 && links[j * t + i] > 0) {
        const d = Math.min(links[i * t + j], links[j * t + i])
        rec[i * t + j] = d
        row.push(d)
      }
    }
    if (row.length) {
      row.sort((a, b) => a - b)
      distToK[i] = row[Math.min(k, row.length) - 1]
      kth.push(distToK[i])
    } else distToK[i] = NaN
  }
  const bw = medianOf(kth)
  for (let i = 0; i < t * t; i++) rec[i] = rec[i] > 0 ? Math.exp(-rec[i] / bw) : 0
  return rec
}

export function symmetricEigen(a: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const V = Float64Array.from(a)
  const d = new Float64Array(n)
  const e = new Float64Array(n)
  for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j]
  for (let i = n - 1; i > 0; i--) {
    let scale = 0
    let h = 0
    for (let k = 0; k < i; k++) scale += Math.abs(d[k])
    if (scale === 0) {
      e[i] = d[i - 1]
      for (let j = 0; j < i; j++) {
        d[j] = V[(i - 1) * n + j]
        V[i * n + j] = 0
        V[j * n + i] = 0
      }
    } else {
      for (let k = 0; k < i; k++) {
        d[k] /= scale
        h += d[k] * d[k]
      }
      let f = d[i - 1]
      let g = Math.sqrt(h)
      if (f > 0) g = -g
      e[i] = scale * g
      h -= f * g
      d[i - 1] = f - g
      for (let j = 0; j < i; j++) e[j] = 0
      for (let j = 0; j < i; j++) {
        f = d[j]
        V[j * n + i] = f
        g = e[j] + V[j * n + j] * f
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k * n + j] * d[k]
          e[k] += V[k * n + j] * f
        }
        e[j] = g
      }
      f = 0
      for (let j = 0; j < i; j++) {
        e[j] /= h
        f += e[j] * d[j]
      }
      const hh = f / (h + h)
      for (let j = 0; j < i; j++) e[j] -= hh * d[j]
      for (let j = 0; j < i; j++) {
        f = d[j]
        g = e[j]
        for (let k = j; k <= i - 1; k++) V[k * n + j] -= f * e[k] + g * d[k]
        d[j] = V[(i - 1) * n + j]
        V[i * n + j] = 0
      }
    }
    d[i] = h
  }
  for (let i = 0; i < n - 1; i++) {
    V[(n - 1) * n + i] = V[i * n + i]
    V[i * n + i] = 1
    const h = d[i + 1]
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k * n + i + 1] / h
      for (let j = 0; j <= i; j++) {
        let g = 0
        for (let k = 0; k <= i; k++) g += V[k * n + i + 1] * V[k * n + j]
        for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k]
      }
    }
    for (let k = 0; k <= i; k++) V[k * n + i + 1] = 0
  }
  for (let j = 0; j < n; j++) {
    d[j] = V[(n - 1) * n + j]
    V[(n - 1) * n + j] = 0
  }
  V[(n - 1) * n + n - 1] = 1
  e[0] = 0
  for (let i = 1; i < n; i++) e[i - 1] = e[i]
  e[n - 1] = 0
  let f = 0
  let tst1 = 0
  const eps = Math.pow(2, -52)
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]))
    let m = l
    while (m < n) {
      if (Math.abs(e[m]) <= eps * tst1) break
      m++
    }
    if (m > l) {
      do {
        let g = d[l]
        let p = (d[l + 1] - g) / (2 * e[l])
        let r = Math.hypot(p, 1)
        if (p < 0) r = -r
        d[l] = e[l] / (p + r)
        d[l + 1] = e[l] * (p + r)
        const dl1 = d[l + 1]
        let h = g - d[l]
        for (let i = l + 2; i < n; i++) d[i] -= h
        f += h
        p = d[m]
        let c = 1
        let c2 = c
        let c3 = c
        const el1 = e[l + 1]
        let s = 0
        let s2 = 0
        for (let i = m - 1; i >= l; i--) {
          c3 = c2
          c2 = c
          s2 = s
          g = c * e[i]
          h = c * p
          r = Math.hypot(p, e[i])
          e[i + 1] = s * r
          s = e[i] / r
          c = p / r
          p = c * d[i] - s * g
          d[i + 1] = h + s * (c * g + s * d[i])
          for (let k = 0; k < n; k++) {
            h = V[k * n + i + 1]
            V[k * n + i + 1] = s * V[k * n + i] + c * h
            V[k * n + i] = c * V[k * n + i] - s * h
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1
        e[l] = s * p
        d[l] = c * p
      } while (Math.abs(e[l]) > eps * tst1)
    }
    d[l] += f
    e[l] = 0
  }
  for (let i = 0; i < n - 1; i++) {
    let k = i
    let p = d[i]
    for (let j = i + 1; j < n; j++) {
      if (d[j] < p) {
        k = j
        p = d[j]
      }
    }
    if (k !== i) {
      d[k] = d[i]
      d[i] = p
      for (let j = 0; j < n; j++) {
        const tmp = V[j * n + i]
        V[j * n + i] = V[j * n + k]
        V[j * n + k] = tmp
      }
    }
  }
  return { values: d, vectors: V }
}

function dot(X: Float64Array, i: number, Y: Float64Array, j: number, dim: number): number {
  let d = 0
  for (let c = 0; c < dim; c++) d += X[i * dim + c] * Y[j * dim + c]
  return d
}

function searchsortedLeft(a: Float64Array, v: number): number {
  let lo = 0
  let hi = a.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (a[mid] < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

function kmeansPlusPlus(X: Float64Array, norms: Float64Array, n: number, dim: number, k: number, rng: MT19937): Float64Array {
  const centers = new Float64Array(k * dim)
  const trials = 2 + Math.floor(Math.log(k))
  const cdf = new Float64Array(n)
  let acc = 0
  for (let i = 0; i < n; i++) cdf[i] = acc += 1 / n
  for (let i = 0; i < n; i++) cdf[i] /= acc
  const u = rng.random()
  let first = 0
  while (first < n && cdf[first] <= u) first++
  const sqDist = (id: number, out: Float64Array) => {
    for (let i = 0; i < n; i++) out[i] = Math.max(0, -2 * dot(X, id, X, i, dim) + norms[id] + norms[i])
  }
  let closest = new Float64Array(n)
  sqDist(first, closest)
  centers.set(X.subarray(first * dim, first * dim + dim), 0)
  let pot = 0
  for (let i = 0; i < n; i++) pot += closest[i]
  const cumulative = new Float64Array(n)
  let candidate = new Float64Array(n)
  let best = new Float64Array(n)
  for (let c = 1; c < k; c++) {
    const ids: number[] = []
    for (let t = 0; t < trials; t++) ids.push(rng.random() * pot)
    acc = 0
    for (let i = 0; i < n; i++) cumulative[i] = acc += closest[i]
    let bestPot = Infinity
    let bestId = 0
    for (const r of ids) {
      const id = Math.min(searchsortedLeft(cumulative, r), n - 1)
      sqDist(id, candidate)
      let candidatePot = 0
      for (let i = 0; i < n; i++) {
        candidate[i] = Math.min(closest[i], candidate[i])
        candidatePot += candidate[i]
      }
      if (candidatePot < bestPot) {
        bestPot = candidatePot
        bestId = id
        ;[best, candidate] = [candidate, best]
      }
    }
    pot = bestPot
    ;[closest, best] = [best, closest]
    centers.set(X.subarray(bestId * dim, bestId * dim + dim), c * dim)
  }
  return centers
}

function lloyd(X: Float64Array, n: number, dim: number, k: number, init: Float64Array, tol: number, maxIter: number): { labels: Int32Array; inertia: number } {
  let centers = Float64Array.from(init)
  let next = new Float64Array(k * dim)
  const labels = new Int32Array(n).fill(-1)
  const old = new Int32Array(n).fill(-1)
  const weights = new Float64Array(k)
  const cnorm = new Float64Array(k)
  const assign = () => {
    for (let c = 0; c < k; c++) cnorm[c] = dot(centers, c, centers, c, dim)
    for (let i = 0; i < n; i++) {
      let best = cnorm[0] - 2 * dot(X, i, centers, 0, dim)
      let bestC = 0
      for (let c = 1; c < k; c++) {
        const d = cnorm[c] - 2 * dot(X, i, centers, c, dim)
        if (d < best) {
          best = d
          bestC = c
        }
      }
      labels[i] = bestC
    }
  }
  let strict = false
  for (let iter = 0; iter < maxIter; iter++) {
    assign()
    next.fill(0)
    weights.fill(0)
    for (let i = 0; i < n; i++) {
      weights[labels[i]] += 1
      for (let j = 0; j < dim; j++) next[labels[i] * dim + j] += X[i * dim + j]
    }
    const empty: number[] = []
    for (let c = 0; c < k; c++) if (weights[c] === 0) empty.push(c)
    if (empty.length) {
      const far = Array.from({ length: n }, (_, i) => i)
      const dist = far.map((i) => {
        let d = 0
        for (let j = 0; j < dim; j++) d += (X[i * dim + j] - centers[labels[i] * dim + j]) ** 2
        return d
      })
      far.sort((a, b) => dist[b] - dist[a])
      empty.forEach((e, m) => {
        const i = far[m]
        for (let j = 0; j < dim; j++) {
          next[labels[i] * dim + j] -= X[i * dim + j]
          next[e * dim + j] = X[i * dim + j]
        }
        weights[e] = 1
        weights[labels[i]] -= 1
      })
    }
    let shift = 0
    for (let c = 0; c < k; c++) {
      let s = 0
      for (let j = 0; j < dim; j++) {
        if (weights[c] > 0) next[c * dim + j] /= weights[c]
        s += (next[c * dim + j] - centers[c * dim + j]) ** 2
      }
      shift += s
    }
    ;[centers, next] = [next, centers]
    if (labels.every((l, i) => l === old[i])) {
      strict = true
      break
    }
    if (shift <= tol) break
    old.set(labels)
  }
  if (!strict) assign()
  let inertia = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < dim; j++) inertia += (X[i * dim + j] - centers[labels[i] * dim + j]) ** 2
  return { labels, inertia }
}

function sameClustering(a: Int32Array, b: Int32Array, k: number): boolean {
  const map = new Int32Array(k).fill(-1)
  for (let i = 0; i < a.length; i++) {
    if (map[a[i]] === -1) map[a[i]] = b[i]
    else if (map[a[i]] !== b[i]) return false
  }
  return true
}

export function kmeans(data: Float64Array, n: number, dim: number, k: number, seed = 0, nInit = 10, maxIter = 300): Int32Array {
  const rng = new MT19937(seed)
  const X = Float64Array.from(data)
  let variance = 0
  for (let j = 0; j < dim; j++) {
    let mean = 0
    for (let i = 0; i < n; i++) mean += X[i * dim + j]
    mean /= n
    let v = 0
    for (let i = 0; i < n; i++) {
      X[i * dim + j] -= mean
      v += X[i * dim + j] ** 2
    }
    variance += v / n
  }
  const tol = (variance / dim) * 1e-4
  const norms = new Float64Array(n)
  for (let i = 0; i < n; i++) norms[i] = dot(X, i, X, i, dim)
  let best: Int32Array | null = null
  let bestInertia = Infinity
  for (let run = 0; run < nInit; run++) {
    const result = lloyd(X, n, dim, k, kmeansPlusPlus(X, norms, n, dim, k, rng), tol, maxIter)
    if (!best || (result.inertia < bestInertia && !sameClustering(result.labels, best, k))) {
      bestInertia = result.inertia
      best = result.labels
    }
  }
  return best ?? new Int32Array(n)
}

export function sections(stemMel: Float32Array, frames: number, beats: number[], downbeats: number[], minSections = 4): Section[] {
  const T = frames
  if (beats.length < 16) return [{ start: 0, end: T / FPS, label: 0 }]
  const mix = new Float64Array(N_MELS * T)
  const chroma = new Float64Array(N_CHROMA * T)
  const bands = chromaBands()
  for (let b = 0; b < N_MELS; b++) {
    for (let t = 0; t < T; t++) {
      const i = b * T + t
      const e0 = Math.exp(stemMel[i])
      const e1 = Math.exp(stemMel[N_MELS * T + i])
      const e2 = Math.exp(stemMel[2 * N_MELS * T + i])
      const e3 = Math.exp(stemMel[3 * N_MELS * T + i])
      mix[i] = Math.log(e0 + e1 + e2 + e3 + 1e-6)
      if (bands[b] >= 0) chroma[bands[b] * T + t] += e2 + e3
    }
  }
  for (let t = 0; t < T; t++) {
    let max = 0
    for (let c = 0; c < N_CHROMA; c++) max = Math.max(max, Math.abs(chroma[c * T + t]))
    if (max < TINY32) continue
    for (let c = 0; c < N_CHROMA; c++) chroma[c * T + t] /= max
  }
  const mfcc = new Float64Array(N_MFCC * T)
  const cosines = new Float64Array(N_MFCC * N_MELS)
  for (let c = 0; c < N_MFCC; c++) {
    const scale = c === 0 ? Math.sqrt(1 / N_MELS) : Math.sqrt(2 / N_MELS)
    for (let b = 0; b < N_MELS; b++) cosines[c * N_MELS + b] = scale * Math.cos((Math.PI * c * (2 * b + 1)) / (2 * N_MELS))
  }
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < N_MFCC; c++) {
      let acc = 0
      for (let b = 0; b < N_MELS; b++) acc += cosines[c * N_MELS + b] * mix[b * T + t]
      mfcc[c * T + t] = acc
    }
  }
  const beatFrames = beats.map((b) => Math.min(T - 1, Math.max(0, roundHalfEven(b * FPS))))
  const bounds = [...new Set([0, ...beatFrames, T])].sort((a, b) => a - b)
  const S = bounds.length - 1
  const timbre = syncMedian(mfcc, N_MFCC, T, bounds)
  const tonal = syncMedian(chroma, N_CHROMA, T, bounds)
  const R = recurrenceAffinity(tonal, N_CHROMA, S)
  const Rf = medianFilterAxis(R, S, S, 7, false)
  const pathDistance = new Float64Array(S - 1)
  for (let j = 0; j < S - 1; j++) {
    let d = 0
    for (let c = 0; c < N_MFCC; c++) d += (timbre[c * S + j + 1] - timbre[c * S + j]) ** 2
    pathDistance[j] = d
  }
  const sigma = median(Float64Array.from(pathDistance), S - 1) + 1e-9
  const pathSim = pathDistance.map((d) => Math.exp(-d / sigma))
  const degPath = new Float64Array(S)
  const degRec = new Float64Array(S)
  for (let i = 0; i < S; i++) {
    if (i > 0) degPath[i] += pathSim[i - 1]
    if (i < S - 1) degPath[i] += pathSim[i]
    for (let j = 0; j < S; j++) degRec[i] += Rf[i * S + j]
  }
  let num = 0
  let den = 0
  for (let i = 0; i < S; i++) {
    num += degPath[i] * (degPath[i] + degRec[i])
    den += (degPath[i] + degRec[i]) ** 2
  }
  const mu = num / (den + 1e-9)
  const A = new Float64Array(S * S)
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      let v = mu * Rf[i * S + j]
      if (j === i + 1) v += (1 - mu) * pathSim[i]
      if (j === i - 1) v += (1 - mu) * pathSim[j]
      A[i * S + j] = v
    }
    A[i * S + i] = 0
  }
  const w = new Float64Array(S)
  const isolated = new Uint8Array(S)
  for (let i = 0; i < S; i++) {
    let sum = 0
    for (let j = 0; j < S; j++) sum += A[j * S + i]
    isolated[i] = sum === 0 ? 1 : 0
    w[i] = sum === 0 ? 1 : Math.sqrt(sum)
  }
  const L = new Float64Array(S * S)
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < i; j++) L[i * S + j] = L[j * S + i] = -A[i * S + j] / w[j] / w[i]
    L[i * S + i] = 1 - isolated[i]
  }
  const { vectors } = symmetricEigen(L, S)
  const k = Math.min(10, Math.max(minSections, roundHalfEven(beats.length / 64) + 2))
  const lead = new Float64Array(S * k)
  for (let i = 0; i < S; i++) for (let c = 0; c < k; c++) lead[i * k + c] = vectors[i * S + c]
  const smoothed = medianFilterAxis(lead, S, k, 9, true)
  for (let i = 0; i < S; i++) {
    let norm = 0
    for (let c = 0; c < k; c++) norm += smoothed[i * k + c] ** 2
    norm = Math.sqrt(norm) + 1e-9
    for (let c = 0; c < k; c++) smoothed[i * k + c] /= norm
  }
  const labels = kmeans(smoothed, S, k, k)
  const starts = [0]
  for (let i = 1; i < S; i++) if (labels[i] !== labels[i - 1]) starts.push(i)
  const times = [...beats, T / FPS]
  const period = medianOf(beats.slice(1).map((b, i) => b - beats[i]))
  const out: Section[] = []
  for (let j = 0; j < starts.length; j++) {
    const b = starts[j]
    let start = j ? times[b] : 0
    if (j && downbeats.length) {
      let nearest = downbeats[0]
      for (const d of downbeats) if (Math.abs(d - start) < Math.abs(nearest - start)) nearest = d
      if (Math.abs(nearest - start) < 0.6 * period) start = nearest
    }
    const end = j + 1 < starts.length ? times[starts[j + 1]] : T / FPS
    out.push({ start: round3(start), end: round3(end), label: labels[b] })
  }
  const bar = 4 * period
  const merged: Section[] = []
  for (const s of out) {
    if (merged.length && s.end - s.start < 2 * bar) merged[merged.length - 1].end = s.end
    else merged.push(s)
  }
  return merged
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000
}

export function sectionIndex(sectionList: Section[], times: ArrayLike<number>): Int32Array {
  const out = new Int32Array(times.length)
  for (let i = 0; i < times.length; i++) {
    let idx = 0
    while (idx + 1 < sectionList.length && sectionList[idx + 1].start <= times[i]) idx++
    out[i] = idx
  }
  return out
}
