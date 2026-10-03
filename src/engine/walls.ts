import type { Difficulty, Wall } from './types'
import { Rng } from './rng'

type Phrase = { tempo: number; length: number; difficulty: Difficulty; walls: number[][] }

export type WallData = {
  placement: { w: number[]; mu: number[]; sd: number[] }
  amount: Record<Difficulty, { share_none: number; phrases_per_min: number[] }>
  library: Phrase[]
}

export type WallSong = {
  duration: number
  beats: number[]
  energy: Float32Array
  attacks: number[]
  syllables: number[]
  sections: { start: number; end: number }[]
  sustains: [number, number][]
  musicStart: number
  musicEnd: number
}

let data: Promise<WallData> | null = null

export function loadWallData(): Promise<WallData> {
  data ??= fetch(`${import.meta.env.BASE_URL}data/walls.json`).then((r) => r.json() as Promise<WallData>)
  return data
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function mean(xs: ArrayLike<number>, a: number, b: number): number {
  const lo = Math.max(0, a)
  const hi = Math.min(xs.length, b)
  if (hi <= lo) return 0
  let s = 0
  for (let i = lo; i < hi; i++) s += xs[i]
  return s / (hi - lo)
}

function countBetween(sorted: number[], lo: number, hi: number): number {
  let n = 0
  for (const v of sorted) if (v >= lo && v < hi) n++
  return n
}

function quantile(sorted: number[], q: number): number {
  const pos = q * (sorted.length - 1)
  const i = Math.floor(pos)
  const f = pos - i
  return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]
}

export function beatToTime(beats: number[], b: number): number {
  const gaps = beats.slice(1).map((v, i) => v - beats[i])
  const bi = median(gaps)
  if (b < 0) return beats[0] + b * bi
  if (b > beats.length - 1) return beats[beats.length - 1] + (b - beats.length + 1) * bi
  const i = Math.floor(b)
  const f = b - i
  return i + 1 < beats.length ? beats[i] * (1 - f) + beats[i + 1] * f : beats[i]
}

export function wallFeatures(song: WallSong, t: number[]): number[][] {
  const { duration, energy, attacks, syllables, sections, sustains } = song
  const eMed = median(Array.from(energy)) + 1e-6
  const secStarts = sections.map((s) => s.start)
  let srank: number[] = []
  if (sections.length >= 2) {
    const se = sections.map((s) => mean(energy, Math.floor(s.start * 50), Math.max(Math.floor(s.start * 50) + 1, Math.floor(s.end * 50))))
    const order = se.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
    srank = new Array(se.length).fill(0)
    order.forEach(([, i], rank) => (srank[i] = rank / Math.max(1, se.length - 1)))
  }
  const rate = attacks.length / Math.max(1, duration)
  return t.map((x) => {
    const i = Math.floor(x * 50)
    const before = i > 0 ? mean(energy, i - 100, i) : energy[0]
    const after = i < energy.length ? mean(energy, i, i + 100) : before
    const rise = Math.log((after + 1e-3) / (before + 1e-3))
    const loud = Math.log((after + 1e-3) / eMed)
    const local = countBetween(attacks, x - 2, x + 2) / 4 / Math.max(rate, 1e-3)
    const voc = syllables.length ? countBetween(syllables, x - 2, x + 2) / 4 : 0
    const secD = secStarts.length ? Math.min(...secStarts.map((s) => Math.abs(s - x))) : 9
    const held = sustains.some(([a0, a1]) => a0 - 0.3 <= x && x <= a1) ? 1 : 0
    let r = 0.5
    if (sections.length >= 2) {
      let k = 0
      while (k < secStarts.length && secStarts[k] <= x) k++
      r = srank[Math.max(0, k - 1)]
    }
    const pos = x / Math.max(1, duration)
    return [1, rise, Math.max(rise, 0), loud, local, Math.log1p(voc), secD < 0.5 ? 1 : 0, secD < 2 ? 1 : 0, held, r, pos, pos * pos, pos < 0.08 ? 1 : 0, pos > 0.9 ? 1 : 0]
  })
}

export type WallAmount = 'auto' | 'none' | 'light' | 'normal' | 'heavy'

export function wallRate(data: WallData, amount: WallAmount, seed: number): number {
  if (amount === 'none') return 0
  const rng = new Rng(seed)
  const am = data.amount.ExpertPlus
  if (amount === 'auto' && rng.random() < am.share_none) return 0
  const q = amount === 'auto' ? rng.uniform(0.2, 0.7) : amount === 'light' ? 0.25 : amount === 'normal' ? 0.5 : 0.85
  return quantile(am.phrases_per_min, q)
}

export function planWalls(data: WallData, song: WallSong, difficulty: Difficulty, seed: number, amount: WallAmount = 'auto', phrasesPerMin?: number): Wall[] {
  if (amount === 'none' || phrasesPerMin === 0) return []
  const rng = new Rng(seed)
  const { beats } = song
  const bi = median(beats.slice(1).map((v, i) => v - beats[i]))
  const tempo = 60 / bi
  const am = data.amount[difficulty]
  let perMin = phrasesPerMin ?? 0
  if (phrasesPerMin === undefined) {
    if (amount === 'auto' && rng.random() < am.share_none) return []
    const q = amount === 'auto' ? rng.uniform(0.2, 0.7) : amount === 'light' ? 0.25 : amount === 'normal' ? 0.5 : 0.9
    perMin = quantile(am.phrases_per_min, q)
  }
  const end = song.musicEnd
  const n = Math.max(1, Math.round((perMin * (end - song.musicStart)) / 60))
  const cand = beats.filter((b) => b >= song.musicStart + 2 && b <= end - 3)
  if (!cand.length) return []
  const X = wallFeatures(song, cand)
  const { w, mu, sd } = data.placement
  const s = X.map((row) => row.reduce((acc, v, j) => acc + ((v - mu[j]) / sd[j]) * w[j], 0))
  const sMax = Math.max(...s)
  const prob = s.map((v) => Math.exp(v - sMax))
  const total = prob.reduce((a, b) => a + b, 0)
  for (let i = 0; i < prob.length; i++) prob[i] /= total
  const nearTempo = data.library.map((p, i) => [p, i] as const).filter(([p]) => Math.abs(Math.log(p.tempo / tempo)) < Math.log(1.2))
  let pool = nearTempo.filter(([p]) => p.difficulty === difficulty).map(([, i]) => i)
  if (!pool.length) pool = nearTempo.map(([, i]) => i)
  if (!pool.length) pool = data.library.map((_, i) => i)
  const walls: Wall[] = []
  const used: [number, number][] = []
  for (let k = 0; k < n * 6 && used.length < n; k++) {
    const j = rng.choice(prob)
    let b0 = 0
    let best = Infinity
    beats.forEach((b, i) => {
      const d = Math.abs(b - cand[j])
      if (d < best) {
        best = d
        b0 = i
      }
    })
    const ph = data.library[pool[rng.integer(pool.length)]]
    const span: [number, number] = [b0, b0 + Math.max(ph.length, 1)]
    if (used.some(([a, b]) => a - 4 < span[1] && span[0] < b + 4)) continue
    used.push(span)
    for (const [off, x, wd, y, h, db] of ph.walls) {
      const t0 = beatToTime(beats, b0 + off)
      const t1 = beatToTime(beats, b0 + off + db)
      if (t1 > end) continue
      walls.push({ time: t0, duration: Math.max(t1 - t0, 0.02), x, y, width: Math.max(1, wd), height: h })
    }
  }
  return walls.sort((a, b) => a.time - b.time)
}
