import type { Difficulty, Note } from './types'
import * as ort from 'onnxruntime-web'

export type SongFacts = { duration: number; beats: number[]; attacks: number[]; musicStart: number; musicEnd: number }

export const FEATURES = [
  'diff', 'nps', 'log_nps', 'nps_music', 'eps', 'npb', 'notes_per_attack',
  'row0', 'row1', 'row2', 'col0', 'col1', 'col2', 'col3', 'grid_entropy', 'hand_side', 'left_share', 'vision_blocks',
  'dir_up', 'dir_down', 'dir_horiz', 'dir_diag', 'dir_dot', 'dir_entropy', 'angle_share',
  'double_share', 'stack_share', 'double_cross', 'double_same_dir',
  'gap_b0', 'gap_b1', 'gap_b2', 'gap_b3', 'gap_b4', 'rests_pm', 'long_rests_pm', 'longest_gap_beats', 'rest_time_share',
  'intro_beats', 'outro_beats', 'on_b1', 'on_b2', 'on_b4', 'off_grid', 'rhythm_entropy', 'hit_att', 'recall_att',
  'dens_cv', 'dens_max_ratio', 'empty_window_share', 'dens_attack_rho',
  'bottom_run_max', 'bottom_run_long', 'bottom_repeat', 'same_cell_repeat',
  'sps', 'notes_per_swing', 'parity_breaks', 'rest_resets', 'strained_rolls', 'extreme_rolls', 'sharp_redirects', 'crossovers',
  'paths_cross', 'speed_med', 'speed_p95', 'redirect_med', 'reversal_share', 'vertical_swing_share', 'fast_swing_share', 'swing_gap_med',
  'motif2', 'motif4', 'motif4_window', 'token_entropy',
] as const

type Vec = [number, number]
type Swing = {
  start: number
  end: number
  notes: Note[]
  dir: Vec | null
  parity: number
  cut: Vec
  roll: number
  entry: Vec
  exit: Vec
  mx: number
  gap?: number
  reset?: 'rest' | 'break' | null
  redirect?: number | null
  speed?: number
  crossover?: boolean
  paths?: boolean
}

const VECTORS: Vec[] = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]]
const DEG = 180 / Math.PI
const RAD = Math.PI / 180
const SWING_GAP = 0.08
const SWING_ANGLE = 45.0
const RESET_REST = 1.0
const SIMULTANEOUS = 0.05
const REACH = 0.75

const pymod = (a: number, b: number) => {
  const m = a % b
  return m !== 0 && b < 0 !== m < 0 ? m + b : m
}
const div = (a: number, b: number) => (b ? a / b : 0)
const count = <T>(xs: T[], pred: (x: T) => boolean) => xs.reduce((n, x) => n + (pred(x) ? 1 : 0), 0)

function mean(xs: number[]): number {
  let s = 0
  for (const x of xs) s += x
  return xs.length ? s / xs.length : 0
}

function std(xs: number[]): number {
  if (!xs.length) return 0
  const m = mean(xs)
  let s = 0
  for (const x of xs) s += (x - m) * (x - m)
  return Math.sqrt(s / xs.length)
}

function percentile(xs: number[], q: number): number {
  if (!xs.length) return 0
  const v = [...xs].sort((a, b) => a - b)
  const pos = ((v.length - 1) * q) / 100
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, v.length - 1)
  return v[lo] + (v[hi] - v[lo]) * (pos - lo)
}

const median = (xs: number[]) => percentile(xs, 50)

function entropy(keys: (string | number)[]): number {
  const counts = new Map<string | number, number>()
  for (const k of keys) counts.set(k, (counts.get(k) ?? 0) + 1)
  let h = 0
  for (const c of counts.values()) {
    const p = c / keys.length
    h -= p * Math.log2(p)
  }
  return h
}

function bisectRight(xs: number[], t: number): number {
  let lo = 0
  let hi = xs.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (t < xs[mid]) hi = mid
    else lo = mid + 1
  }
  return lo
}

function bisectLeft(xs: number[], t: number): number {
  let lo = 0
  let hi = xs.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (xs[mid] < t) lo = mid + 1
    else hi = mid
  }
  return lo
}

function near(xs: number[], t: number, tol: number): boolean {
  const i = bisectLeft(xs, t - tol)
  return i < xs.length && xs[i] <= t + tol
}

function beatOf(beats: number[], spb: number, t: number): number {
  if (beats.length < 2) return t / spb
  if (t <= beats[0]) return (t - beats[0]) / spb
  if (t >= beats[beats.length - 1]) return beats.length - 1 + (t - beats[beats.length - 1]) / spb
  const i = bisectRight(beats, t) - 1
  return i + (t - beats[i]) / (beats[i + 1] - beats[i])
}

function ranks(xs: number[]): number[] {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b] || a - b)
  const r = new Array<number>(xs.length).fill(0)
  let i = 0
  while (i < order.length) {
    let j = i
    while (j + 1 < order.length && xs[order[j + 1]] === xs[order[i]]) j++
    for (let k = i; k <= j; k++) r[order[k]] = (i + j) / 2 + 1
    i = j + 1
  }
  return r
}

function spearman(a: number[], b: number[]): number {
  if (a.length < 2 || std(a) === 0 || std(b) === 0) return 0
  const ra = ranks(a)
  const rb = ranks(b)
  const ma = mean(ra)
  const mb = mean(rb)
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < ra.length; i++) {
    sab += (ra[i] - ma) * (rb[i] - mb)
    saa += (ra[i] - ma) * (ra[i] - ma)
    sbb += (rb[i] - mb) * (rb[i] - mb)
  }
  return sab / Math.sqrt(saa * sbb)
}

function unit(v: Vec): Vec {
  const n = Math.sqrt(v[0] * v[0] + v[1] * v[1])
  return [v[0] / n, v[1] / n]
}

const ALL_CUTS = VECTORS.map(unit)

function noteVector(n: Note): Vec | null {
  if (n.direction === 8) return null
  const v = unit(VECTORS[n.direction])
  const a = (n.angle || 0) * RAD
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

const angleBetween = (a: Vec, b: Vec) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1]))) * DEG

function roll(hand: number, parity: number, cut: Vec): number {
  const x = hand === 1 ? cut[0] : -cut[0]
  return pymod(Math.atan2(cut[1], x) * DEG - (parity === 0 ? -90.0 : 90.0) + 180.0, 360.0) - 180.0
}

function rollCost(r: number): number {
  const m = Math.abs(r)
  if (m <= 90) return 0
  if (m <= 135) return (m - 90) / 45
  return 1.0 + (3.0 * (m - 135)) / 45
}

function resetCost(gap: number): number {
  if (gap >= 1.5) return 0.5
  if (gap <= 0.35) return 4.0
  return 4.0 - (3.5 * (gap - 0.35)) / (1.5 - 0.35)
}

type Cand = [number, Vec]

function transition(hand: number, a: Cand, b: Cand, gap: number): number {
  const rb = roll(hand, b[0], b[1])
  const cost = rollCost(rb)
  if (a[0] === b[0]) return cost + resetCost(gap)
  const delta = Math.abs(pymod(rb - roll(hand, a[0], a[1]) + 180.0, 360.0) - 180.0)
  return cost + 1.5 * (delta / 180.0) * (delta / 180.0)
}

function swingsOf(notes: Note[], hand: number): Swing[] {
  const out: Swing[] = []
  for (const n of notes) {
    if (n.hand !== hand) continue
    const v = noteVector(n)
    const last = out.length ? out[out.length - 1] : null
    if (last && n.time - last.end <= SWING_GAP && (last.dir === null || v === null || angleBetween(last.dir, v) <= SWING_ANGLE)) {
      last.notes.push(n)
      last.end = n.time
      if (last.dir === null) last.dir = v
    } else {
      out.push({ start: n.time, end: n.time, notes: [n], dir: v, parity: 0, cut: [0, 0], roll: 0, entry: [0, 0], exit: [0, 0], mx: 0 })
    }
  }
  if (!out.length) return out
  const cand: Cand[][] = out.map((s) => (s.dir !== null ? [s.dir] : ALL_CUTS).flatMap((c): Cand[] => [[0, c], [1, c]]))
  let fwd = cand[0].map(([p, c]) => rollCost(roll(hand, p, c)))
  const ptr: number[][] = [[]]
  for (let i = 1; i < out.length; i++) {
    const gap = out[i].start - out[i - 1].end
    const nxt: number[] = []
    const bp: number[] = []
    for (const b of cand[i]) {
      let best = Infinity
      let arg = 0
      for (let j = 0; j < cand[i - 1].length; j++) {
        const v = fwd[j] + transition(hand, cand[i - 1][j], b, gap)
        if (v < best) {
          best = v
          arg = j
        }
      }
      nxt.push(best)
      bp.push(arg)
    }
    fwd = nxt
    ptr.push(bp)
  }
  let k = 0
  for (let j = 0; j < fwd.length; j++) if (fwd[j] < fwd[k]) k = j
  for (let i = out.length - 1; i >= 0; i--) {
    const s = out[i]
    ;[s.parity, s.cut] = cand[i][k]
    s.roll = roll(hand, s.parity, s.cut)
    if (i) k = ptr[i][k]
  }
  for (const s of out) {
    const c = s.cut
    let lo = 0
    let hi = 0
    let plo: number | null = null
    let phi: number | null = null
    s.notes.forEach((n, j) => {
      const p = n.x * c[0] + n.y * c[1]
      if (plo === null || p < plo) {
        plo = p
        lo = j
      }
      if (phi === null || p > phi) {
        phi = p
        hi = j
      }
    })
    const a = s.notes[lo]
    const b = s.notes[hi]
    s.entry = [a.x - REACH * c[0], a.y - REACH * c[1]]
    s.exit = [b.x + REACH * c[0], b.y + REACH * c[1]]
    s.mx = mean(s.notes.map((n) => n.x))
  }
  for (let i = 1; i < out.length; i++) {
    const s = out[i]
    const prev = out[i - 1]
    const gap = s.start - prev.end
    s.gap = gap
    s.reset = s.parity !== prev.parity ? null : gap >= RESET_REST ? 'rest' : 'break'
    s.redirect = s.parity !== prev.parity ? angleBetween([-prev.cut[0], -prev.cut[1]], s.cut) : null
    const dx = s.entry[0] - prev.exit[0]
    const dy = s.entry[1] - prev.exit[1]
    s.speed = Math.sqrt(dx * dx + dy * dy) / Math.max(gap, 1e-3)
  }
  return out
}

function orient(a: Vec, b: Vec, c: Vec): number {
  const v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
  return Number(v > 0) - Number(v < 0)
}

function crossMarks(left: Swing[], right: Swing[]) {
  let j = 0
  for (const s of left) {
    while (j < right.length && right[j].end < s.start - SIMULTANEOUS) j++
    for (let k = j; k < right.length && right[k].start <= s.end + SIMULTANEOUS; k++) {
      const r = right[k]
      if (s.mx > r.mx + 0.5) s.crossover = r.crossover = true
      if (orient(s.entry, s.exit, r.entry) * orient(s.entry, s.exit, r.exit) < 0 && orient(r.entry, r.exit, s.entry) * orient(r.entry, r.exit, s.exit) < 0) s.paths = r.paths = true
    }
  }
}

const octant = (c: Vec) => Math.floor((Math.atan2(c[1], c[0]) * DEG + 360.0 + 22.5) / 45) % 8

function grams(seq: string[], n: number): string[] {
  const g: string[] = []
  for (let i = 0; i + n <= seq.length; i++) g.push(seq.slice(i, i + n).join('|'))
  return g
}

function ngramVariety(seq: string[], n: number): number | null {
  if (seq.length < n + 4) return null
  const g = grams(seq, n)
  return new Set(g).size / g.length
}

function windowVariety(seq: string[], n = 4, win = 64): number | null {
  if (seq.length < n + 4) return null
  const g = grams(seq, n)
  const vals: number[] = []
  for (let i = 0; i < Math.max(1, g.length - win + 1); i += win >> 1) {
    const w = g.slice(i, i + win)
    vals.push(new Set(w).size / w.length)
  }
  return mean(vals)
}

function avg(vals: (number | null)[]): number {
  const v = vals.filter((x): x is number => x !== null)
  return v.length ? mean(v) : 0
}

export function judgeFeatures(notes: Note[], facts: SongFacts, difficulty: Difficulty): Record<(typeof FEATURES)[number], number> {
  const ns = [...notes].sort((a, b) => a.time - b.time || a.hand - b.hand || a.y - b.y || a.x - b.x)
  const N = ns.length
  const F: Record<string, number> = {}
  const { duration: dur, beats, attacks: att, musicStart: m0, musicEnd: m1 } = facts
  const span = Math.max(1, m1 - m0)
  const spb = beats.length > 3 ? median(beats.slice(1).map((b, i) => b - beats[i])) : 0.5
  const attIn = att.filter((a) => m0 <= a && a <= m1)
  const groups: Note[][] = []
  for (const n of ns) {
    const g = groups[groups.length - 1]
    if (g && n.time - g[0].time <= 0.001) g.push(n)
    else groups.push([n])
  }
  const times = groups.map((g) => g[0].time)
  const G = groups.length
  F.diff = difficulty === 'ExpertPlus' ? 1 : 0
  F.nps = N / dur
  F.log_nps = Math.log(Math.max(F.nps, 0.01))
  F.nps_music = N / span
  F.eps = G / dur
  F.npb = (N * spb) / span
  F.notes_per_attack = N / Math.max(1, attIn.length)
  for (let y = 0; y < 3; y++) F[`row${y}`] = div(count(ns, (n) => n.y === y), N)
  for (let x = 0; x < 4; x++) F[`col${x}`] = div(count(ns, (n) => n.x === x), N)
  F.grid_entropy = N ? entropy(ns.map((n) => n.x * 3 + n.y)) / Math.log2(12) : 0
  F.hand_side = div(count(ns, (n) => n.x <= 1 === (n.hand === 0)), N)
  F.left_share = div(count(ns, (n) => n.hand === 0), N)
  const tl = ns.map((n) => n.time)
  let vb = 0
  for (const n of ns) {
    if ((n.x === 1 || n.x === 2) && n.y === 1) {
      const j = bisectLeft(tl, n.time + 0.02)
      if (j < N && tl[j] - n.time <= 0.5) vb++
    }
  }
  F.vision_blocks = div(vb, N)
  F.dir_up = div(count(ns, (n) => n.direction === 0), N)
  F.dir_down = div(count(ns, (n) => n.direction === 1), N)
  F.dir_horiz = div(count(ns, (n) => n.direction === 2 || n.direction === 3), N)
  F.dir_diag = div(count(ns, (n) => n.direction >= 4 && n.direction <= 7), N)
  F.dir_dot = div(count(ns, (n) => n.direction === 8), N)
  F.dir_entropy = N ? entropy(ns.map((n) => n.direction)) / Math.log2(9) : 0
  F.angle_share = div(count(ns, (n) => Math.abs(n.angle || 0) > 1), N)
  let dbl = 0
  let cross = 0
  let same = 0
  let stack = 0
  for (const g of groups) {
    const L = g.filter((n) => n.hand === 0)
    const R = g.filter((n) => n.hand === 1)
    stack += Number(L.length >= 2) + Number(R.length >= 2)
    if (L.length && R.length) {
      dbl++
      cross += Number(mean(L.map((n) => n.x)) > mean(R.map((n) => n.x)))
      same += Number(L[0].direction === R[0].direction)
    }
  }
  F.double_share = div(dbl, G)
  F.stack_share = div(stack, G)
  F.double_cross = div(cross, G)
  F.double_same_dir = div(same, dbl)
  const gaps = times.slice(1).map((t, i) => (t - times[i]) / spb)
  const edges = [0.375, 0.75, 1.5, 3.0]
  for (let b = 0; b < 5; b++) {
    const lo = b ? edges[b - 1] : -Infinity
    const hi = b < 4 ? edges[b] : Infinity
    F[`gap_b${b}`] = div(count(gaps, (g) => lo <= g && g < hi), gaps.length)
  }
  const mins = dur / 60
  F.rests_pm = count(gaps, (g) => g >= 2) / mins
  F.long_rests_pm = count(gaps, (g) => g >= 4) / mins
  F.longest_gap_beats = gaps.length ? Math.max(...gaps) : 0
  let rt = 0
  for (const g of gaps) if (g >= 2) rt += g * spb
  F.rest_time_share = G ? rt / Math.max(1, times[G - 1] - times[0]) : 0
  F.intro_beats = G ? (times[0] - m0) / spb : 0
  F.outro_beats = G ? (m1 - times[G - 1]) / spb : 0
  const cls = times.map((t) => {
    const b = beatOf(beats, spb, t)
    const f = b - Math.floor(b)
    const d1 = Math.min(f, 1 - f)
    const d2 = Math.abs(f - 0.5)
    const d4 = Math.min(Math.abs(f - 0.25), Math.abs(f - 0.75))
    return d1 < 0.06 ? 1 : d2 < 0.06 ? 2 : d4 < 0.06 ? 4 : 0
  })
  F.on_b1 = div(count(cls, (c) => c === 1), G)
  F.on_b2 = div(count(cls, (c) => c === 2), G)
  F.on_b4 = div(count(cls, (c) => c === 4), G)
  F.off_grid = div(count(cls, (c) => c === 0), G)
  F.rhythm_entropy = G ? entropy(cls) : 0
  F.hit_att = div(count(times, (t) => near(att, t, 0.05)), G)
  F.recall_att = div(count(attIn, (a) => near(times, a, 0.05)), attIn.length)
  const w = 4 * spb
  const cnt: number[] = []
  const acnt: number[] = []
  for (let k = 0; m0 + k * w < m1; k++) {
    const lo = m0 + k * w
    const hi = m0 + (k + 1) * w
    cnt.push(bisectLeft(tl, hi) - bisectLeft(tl, lo))
    acnt.push(bisectLeft(att, hi) - bisectLeft(att, lo))
  }
  const mc = mean(cnt)
  F.dens_cv = div(std(cnt), mc)
  F.dens_max_ratio = div(cnt.length ? Math.max(...cnt) : 0, mc)
  F.empty_window_share = div(count(cnt, (c) => c === 0), cnt.length)
  F.dens_attack_rho = spearman(cnt, acnt)
  let run = 0
  let best = 0
  let long = 0
  for (const n of [...ns, null]) {
    if (n !== null && n.y === 0) run++
    else {
      if (run >= 8) long += run
      run = 0
    }
    best = Math.max(best, run)
  }
  F.bottom_run_max = best
  F.bottom_run_long = div(long, N)
  let rep = 0
  let cell = 0
  let pairs = 0
  for (const h of [0, 1]) {
    const hn = ns.filter((n) => n.hand === h)
    for (let i = 1; i < hn.length; i++) {
      const a = hn[i - 1]
      const b = hn[i]
      pairs++
      rep += Number(a.y === 0 && b.y === 0)
      cell += Number(a.x === b.x && a.y === b.y)
    }
  }
  F.bottom_repeat = div(rep, pairs)
  F.same_cell_repeat = div(cell, pairs)
  const hands = [swingsOf(ns, 0), swingsOf(ns, 1)]
  crossMarks(hands[0], hands[1])
  const S = [...hands[0], ...hands[1]]
  const nS = Math.max(1, S.length)
  const T = S.filter((s) => s.gap !== undefined)
  F.sps = S.length / dur
  F.notes_per_swing = div(N, S.length)
  F.parity_breaks = (100 * count(T, (s) => s.reset === 'break')) / nS
  F.rest_resets = (100 * count(T, (s) => s.reset === 'rest')) / nS
  F.strained_rolls = (100 * count(S, (s) => Math.abs(s.roll) > 90 && Math.abs(s.roll) <= 135)) / nS
  F.extreme_rolls = (100 * count(S, (s) => Math.abs(s.roll) > 135)) / nS
  F.sharp_redirects = (100 * count(T, (s) => (s.redirect ?? 0) > 90)) / nS
  F.crossovers = (100 * count(S, (s) => !!s.crossover)) / nS
  F.paths_cross = (100 * count(S, (s) => !!s.paths)) / nS
  const sp = T.map((s) => s.speed ?? 0)
  const rd = T.flatMap((s) => (s.redirect === null || s.redirect === undefined ? [] : [s.redirect]))
  F.speed_med = median(sp)
  F.speed_p95 = percentile(sp, 95)
  F.redirect_med = median(rd)
  F.reversal_share = div(count(rd, (r) => r <= 1), rd.length)
  F.vertical_swing_share = div(count(S, (s) => Math.abs(s.cut[0]) < 0.01), S.length)
  F.fast_swing_share = div(count(T, (s) => (s.gap ?? 0) < 0.2), T.length)
  F.swing_gap_med = median(T.map((s) => s.gap ?? 0))
  const toks = hands.map((hs) => hs.map((s) => `${octant(s.cut)},${s.notes[0].x},${s.notes[0].y}`))
  F.motif2 = avg(toks.map((t) => ngramVariety(t, 2)))
  F.motif4 = avg(toks.map((t) => ngramVariety(t, 4)))
  F.motif4_window = avg(toks.map((t) => windowVariety(t)))
  F.token_entropy = avg(toks.map((t) => (t.length ? entropy(t) : null)))
  return Object.fromEntries(FEATURES.map((k) => [k, F[k]])) as Record<(typeof FEATURES)[number], number>
}

export async function judgeScores(session: ort.InferenceSession, charts: { notes: Note[]; difficulty: Difficulty }[], facts: SongFacts): Promise<number[]> {
  if (!charts.length) return []
  const data = new Float32Array(charts.length * FEATURES.length)
  charts.forEach((c, i) => {
    const f = judgeFeatures(c.notes, facts, c.difficulty)
    FEATURES.forEach((k, j) => (data[i * FEATURES.length + j] = f[k]))
  })
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, [charts.length, FEATURES.length]) })
  return Array.from(out[session.outputNames[0]].data as Float32Array)
}

// Shared swing geometry for the replay-trained strain model. This does not run
// the old composite judge or its learned scoring network.
export function analyzeSwings(notes: Note[]): [Swing[], Swing[]] {
  const ordered = [...notes].sort((a, b) => a.time - b.time || a.hand - b.hand || a.y - b.y || a.x - b.x)
  const left = swingsOf(ordered, 0), right = swingsOf(ordered, 1)
  crossMarks(left, right)
  return [left, right]
}
