import * as ort from 'onnxruntime-web'
import type { Arc, Note, Wall } from './types'
import { Rng } from './rng'
import { wallNoteMask } from './wallRules'
import { loadModel, type Backend } from './models'

export type NoteDifficulty = 'Easy' | 'Normal' | 'Hard' | 'Expert' | 'ExpertPlus'

export const NOTE_GRAPHS = ['encoder', 'cross_kv', 'candidates', 'step', 'follow', 'pointer', 'gesture'] as const
export type NoteGraph = (typeof NOTE_GRAPHS)[number]
export type NoteSessions = Record<NoteGraph, ort.InferenceSession>

export type RepeatSegment = { start: number; end: number; lag: number }

export type NoteSong = {
  features: Float32Array
  sectionLabel: Int32Array
  candTimes: ArrayLike<number>
  candFeats: Float32Array
  candTokens: Int32Array
  duration: number
  musicStart: number
  musicEnd: number | null
  introStem: number
  sustains: [number, number][]
  segments: RepeatSegment[]
  wallsPlan: Wall[]
}

export type Picker = (z: Float64Array, mask: boolean[] | null, topP: number) => number

export type NoteOptions = {
  difficulty: NoteDifficulty
  seed: number
  temperature?: number
  topP?: number
  cond?: [number, number]
  condScale?: number
  minEndFraction?: number
  maxSwings?: number
  literalness?: number
  focusStick?: number
  focusStickBoundary?: number
  copyP?: number
  mirrorShare?: number
  rng?: Rng
  pick?: Picker
  onProgress?: (fraction: number) => void
  signal?: AbortSignal
  encoding?: NoteEncoding
  flowCosts?: (state: NoteState) => Promise<Float32Array>
}

export type Placed = {
  hand: number
  gap: number
  parity: number
  cut: number
  dot: number
  cell: number
  stack: number
  chain: number
  chainSlices: number
  chainTail: number
  arcOut: number
  follow: number
  token: number
  cand: number
}

export type Gesture = {
  parity: number
  cut: number
  dot: number
  cell: number
  stack: number
  chain: number
  chainSlices: number
  chainTail: number
  arcOut: number
}

export type GestureLogits = Record<keyof Gesture, ArrayLike<number>>

export type Swing = {
  time: number
  hand: 0 | 1
  parity: number
  cut: number
  cell: number
  notes: Note[]
  arcOut: boolean
  token: number
  follow: number
  copied: boolean
}

type Copy = { src: number; cand: number; mirror: boolean }

export const DIFFICULTIES: NoteDifficulty[] = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']
export const WINDOW = 128
const GAP_BUCKETS = 32
const END = 2
const PREV_FIELDS: (keyof Placed)[] = ['hand', 'gap', 'parity', 'cut', 'dot', 'cell', 'stack', 'chain', 'chainSlices', 'chainTail', 'arcOut', 'follow']
const DEC_LAYERS = 8
const DEC_HEADS = 8
const HEAD_DIM = 64
const DC = 1024
const DK = 256

const VECTORS: [number, number][] = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]]
const OPPOSITE = [1, 0, 3, 2, 7, 6, 5, 4]
const MIRROR_CUT = [0, 1, 3, 2, 5, 4, 7, 6]
export const EXIT: number[][] = Array.from({ length: 12 }, (_, cell) =>
  VECTORS.map(([dx, dy]) => Math.min(3, Math.max(0, Math.floor(cell / 3) + dx)) * 3 + Math.min(2, Math.max(0, (cell % 3) + dy))),
)

const LITERALNESS = 1.0
const FOCUS_STICK = 0.5
const FOCUS_STICK_BOUNDARY = 0.0
const COPY_P = 0.7
const MIRROR_SHARE = 0.15
const COPY_SNAP = 0.035
const MAX_STACK_EXTRA = 2
const INTRO_FIRST: [number, number] = [-0.05, 0.75]
const INTRO_EASE = 6.0
const EASE_GAP = 1.5
const EASE_LAYER = 2.0
const EASE_DOUBLE = 6.0
const ARC_MIN = 1.0
const ARC_APART = 0.5
const END_MARGIN = 0.25
const RESET_SECONDS = 1.0
const MIN_SAME_HAND_GAP = 0.08
const CENTRE_PENALTY = 2.0
const CENTRE_CELLS = [4, 7]
const ROLL_LIMIT = 135.0
const EYE = 1.6
const NOTE_SIZE = 0.45
const CUT_Z = 0.6
const READ_WINDOW: [number, number] = [0.3, 0.8]
const DOUBLE_GAIN = 3.0
const DOUBLE_EMA = 0.03
const DOUBLE_TARGET: Record<NoteDifficulty, number> = { Easy: 0.06, Normal: 0.07, Hard: 0.08, Expert: 0.1, ExpertPlus: 0.14 }
const SPEED_CAP: Record<NoteDifficulty, number> = { Easy: 10.0, Normal: 11.0, Hard: 13.0, Expert: 17.4, ExpertPlus: 21.8 }
const NJS: Record<NoteDifficulty, number> = { Easy: 10, Normal: 12, Hard: 14, Expert: 16, ExpertPlus: 18 }

export function gapBucket(seconds: number): number {
  const s = Math.max(seconds, 0)
  if (s < 0.02) return 0
  return Math.min(GAP_BUCKETS - 1, Math.max(0, 1 + Math.floor(Math.log2(s / 0.02) * 4)))
}

export function roll(hand: number, parity: number, cut: number): number {
  const [vx, vy] = VECTORS[cut]
  const angle = Math.round((Math.atan2(vy, hand === 1 ? vx : -vx) * 180) / Math.PI)
  const neutral = parity === 0 ? -90 : 90
  return ((((angle - neutral + 180) % 360) + 360) % 360) - 180
}

export function sample(logits: ArrayLike<number>, rng: Rng, temperature: number, topP: number, mask: boolean[] | null): number {
  const n = logits.length
  const z = new Float64Array(n)
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    z[i] = mask && !mask[i] ? -Infinity : logits[i] / Math.max(temperature, 1e-4)
    if (z[i] > max) max = z[i]
  }
  let sum = 0
  for (let i = 0; i < n; i++) {
    z[i] = Math.exp(z[i] - max)
    sum += z[i]
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => z[b] - z[a])
  let cum = 0
  let kept = 0
  for (const i of order) {
    if (cum < topP) kept++
    cum += z[i] / sum
  }
  const keep = order.slice(0, kept)
  const total = keep.reduce((acc, i) => acc + z[i], 0)
  const u = rng.random()
  let cdf = 0
  for (const i of keep) {
    cdf += z[i] / total
    if (u < cdf) return i
  }
  return keep[keep.length - 1]
}

function clip(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

function searchSorted(xs: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = xs.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (xs[mid] < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

function stackCells(x: number, y: number, cut: number, extra: number): [number, number][] {
  const [dx, dy] = VECTORS[cut]
  const cells: [number, number][] = []
  let cx = x
  let cy = y
  for (let i = 0; i < extra; i++) {
    cx += dx
    cy += dy
    if (!(cx >= 0 && cx < 4 && cy >= 0 && cy < 3)) break
    cells.push([cx, cy])
  }
  return cells
}

function anyOf(a: boolean[]): boolean {
  return a.some(Boolean)
}

function and(a: boolean[], b: boolean[]): boolean[] {
  return a.map((v, i) => v && b[i])
}

export class NoteState {
  readonly candTimes: ArrayLike<number>
  readonly candTokens: Int32Array
  readonly sectionLabel: Int32Array
  readonly sound: boolean[]
  readonly stemOnset: Float32Array
  readonly duration: number
  readonly musicStart: number
  readonly musicEnd: number
  readonly introStem: number
  readonly sustains: [number, number][]
  readonly wallsPlan: Wall[]
  readonly segments: (RepeatSegment & { decided: boolean })[]
  readonly difficulty: NoteDifficulty
  readonly speedCap: number
  readonly njs: number
  readonly temperature: number
  readonly topP: number
  readonly minEndFraction: number
  readonly literalness: number
  readonly focusStick: number
  readonly focusStickBoundary: number
  readonly copyP: number
  readonly mirrorShare: number
  readonly rng: Rng
  readonly pick: Picker
  history: Placed[] = []
  swings: Swing[] = []
  last: [Swing | null, Swing | null] = [null, null]
  prevIdx = 0
  prevTime = 0
  follow: number | null = null
  hand: 0 | 1 = 0
  cand = 0
  time = 0
  token = 0
  queue: Copy[] = []
  forced: Copy | null = null
  copied = 0
  doubleRate: number

  constructor(song: NoteSong, options: NoteOptions) {
    this.candTimes = song.candTimes
    this.candTokens = song.candTokens
    this.sectionLabel = song.sectionLabel
    const C = song.candTimes.length
    const Fc = song.candFeats.length / C
    this.sound = Array.from({ length: C }, (_, i) => song.candFeats[i * Fc + Fc - 1] > 0.5)
    this.stemOnset = new Float32Array(C * 4)
    for (let i = 0; i < C; i++) for (let s = 0; s < 4; s++) this.stemOnset[i * 4 + s] = song.candFeats[i * Fc + 3 + s]
    this.duration = song.duration
    this.musicStart = song.musicStart
    this.musicEnd = song.musicEnd || song.duration
    this.introStem = song.introStem
    this.sustains = song.sustains
    this.wallsPlan = song.wallsPlan
    this.segments = song.segments.map((s) => ({ ...s, decided: false }))
    this.difficulty = options.difficulty
    this.speedCap = SPEED_CAP[options.difficulty]
    this.njs = NJS[options.difficulty]
    this.temperature = options.temperature ?? 1.0
    this.topP = options.topP ?? 0.95
    this.minEndFraction = options.minEndFraction ?? 0.5
    this.literalness = options.literalness ?? LITERALNESS
    this.focusStick = options.focusStick ?? FOCUS_STICK
    this.focusStickBoundary = options.focusStickBoundary ?? FOCUS_STICK_BOUNDARY
    this.copyP = options.copyP ?? COPY_P
    this.mirrorShare = options.mirrorShare ?? MIRROR_SHARE
    this.rng = options.rng ?? new Rng(options.seed)
    this.pick = options.pick ?? ((z, mask, topP) => sample(z, this.rng, this.temperature, topP, mask))
    this.doubleRate = DOUBLE_TARGET[options.difficulty]
  }

  s(logits: ArrayLike<number>, mask: boolean[] | null = null, topP: number = this.topP): number {
    return this.pick(Float64Array.from(logits), mask, topP)
  }

  rare(logits: ArrayLike<number>): number {
    return this.s(logits, null, 1.0)
  }

  planCopy() {
    if (this.queue.length || !this.swings.length) return
    const t0 = this.prevTime
    const C = this.candTimes.length
    for (const seg of this.segments) {
      if (seg.decided || !(seg.start - 0.05 <= t0 && t0 < seg.end - 0.5)) continue
      seg.decided = true
      if (this.rng.random() >= this.copyP) return
      const mirror = this.rng.random() < this.mirrorShare
      const lag = seg.lag
      let queue: Copy[] = []
      for (let i = 0; i < this.swings.length; i++) {
        const sw = this.swings[i]
        if (!(t0 - lag + 0.03 < sw.time && sw.time <= seg.end - lag)) continue
        const target = sw.time + lag
        const j = clip(searchSorted(this.candTimes, target), 1, C - 1)
        const c = Math.abs(this.candTimes[j - 1] - target) <= Math.abs(this.candTimes[j] - target) ? j - 1 : j
        if (Math.abs(this.candTimes[c] - target) <= COPY_SNAP) queue.push({ src: i, cand: c, mirror })
      }
      queue = this.alignParity(queue)
      if (queue.length >= 4) this.queue = queue
      return
    }
  }

  alignParity(queue: Copy[]): Copy[] {
    for (const h of [0, 1]) {
      for (let n = 0; n < queue.length; n++) {
        const q = queue[n]
        const src = this.swings[q.src]
        if ((q.mirror ? 1 - src.hand : src.hand) !== h) continue
        const last = this.last[h]
        if (last && this.candTimes[q.cand] - last.time < RESET_SECONDS && src.parity === last.parity) queue = [...queue.slice(0, n), ...queue.slice(n + 1)]
        break
      }
    }
    return queue
  }

  abortCopy() {
    this.queue = []
    this.forced = null
  }

  chooseHand(logits: ArrayLike<number>): number | null {
    this.planCopy()
    const endOk = this.prevTime >= this.minEndFraction * this.duration
    if (this.queue.length) {
      const q = this.queue[0]
      const h = this.swings[q.src].hand
      this.forced = q
      this.hand = q.mirror ? ((1 - h) as 0 | 1) : h
      return this.hand
    }
    this.forced = null
    const hand = this.s(logits, [true, true, endOk])
    if (hand === END) return null
    this.hand = hand as 0 | 1
    return hand
  }

  chooseFollow(logits: ArrayLike<number>): number {
    if (this.forced) {
      this.follow = this.history[this.forced.src].follow
      return this.follow
    }
    if (this.prevTime < this.musicStart + INTRO_EASE) {
      this.follow = this.introStem
      return this.follow
    }
    const z = Float64Array.from(logits)
    if (this.follow !== null) {
      const S = this.sectionLabel.length
      const C = this.candTokens.length
      const boundary =
        this.swings.length > 0 &&
        this.sectionLabel[Math.min(this.history[this.history.length - 1].token, S - 1)] !==
          this.sectionLabel[Math.min(this.candTokens[Math.min(this.prevIdx + 1, C - 1)], S - 1)]
      z[this.follow] += boundary ? this.focusStickBoundary : this.focusStick
    }
    this.follow = this.s(z)
    return this.follow
  }

  windowMask(window: ArrayLike<number>): boolean[] {
    const C = this.candTimes.length
    const t = Array.from(window, (w) => this.candTimes[clip(w, 0, C - 1)])
    const mine = this.last[this.hand]
    const earliest = mine ? mine.time + MIN_SAME_HAND_GAP : -1.0
    const lo = Math.max(earliest, this.prevTime)
    const hi = Math.min(this.duration - 0.05, this.musicEnd + END_MARGIN)
    let mask = Array.from(window, (w, k) => w < C && t[k] >= lo && t[k] <= hi)
    if (this.wallsPlan.length) mask = mask.map((m, k) => m && !wallNoteMask(this.wallsPlan, t[k]).all)
    if (!this.swings.length) {
      const first = this.musicStart + INTRO_FIRST[0]
      const until = this.musicStart + INTRO_FIRST[1]
      const start = mask.map((m, k) => m && t[k] >= first && t[k] <= until)
      if (!anyOf(start)) {
        const after = mask.findIndex((m, k) => m && t[k] >= first)
        if (after >= 0) start[after] = true
      }
      mask = start
    }
    return mask
  }

  chooseCandidate(logits: ArrayLike<number>, window: ArrayLike<number>): number | null {
    const mask = this.windowMask(window)
    if (this.forced) {
      const hit = Array.from(window).findIndex((w, k) => w === this.forced!.cand && mask[k])
      if (hit >= 0) return this.setCand(window[hit])
      this.abortCopy()
    }
    if (!anyOf(mask)) return null
    const C = this.candTimes.length
    const z = Float64Array.from(logits)
    const n = z.length
    const wc = Array.from(window, (w) => clip(w, 0, C - 1))
    const gap = wc.map((c) => this.candTimes[c] - this.prevTime)
    const ease = wc.map((c) => 1.0 - clip((this.candTimes[c] - this.musicStart) / INTRO_EASE, 0, 1))
    const otherHand = this.swings.length > 0 && this.swings[this.swings.length - 1].hand !== this.hand
    for (let k = 0; k < n; k++) if (!this.sound[wc[k]]) z[k] -= this.literalness
    if (anyOf(ease.map((e) => e !== 0))) {
      for (let k = 0; k < n; k++) {
        z[k] += ease[k] * (EASE_GAP * clip(Math.log(Math.max(gap[k], 0.05) / 0.25), -2, 1.5) + EASE_LAYER * this.stemOnset[wc[k] * 4 + this.introStem])
        if (otherHand && gap[k] <= 0.03) z[k] -= EASE_DOUBLE * ease[k]
      }
    }
    if (otherHand) {
      const target = DOUBLE_TARGET[this.difficulty]
      const bonus = clip((DOUBLE_GAIN * (target - this.doubleRate)) / target, -6.0, 1.0)
      for (let k = 0; k < n; k++) if (gap[k] <= 0.03) z[k] += bonus
    }
    return this.setCand(window[this.s(z, mask)])
  }

  setCand(c: number): number {
    this.cand = c
    this.time = this.candTimes[c]
    this.token = this.candTokens[c]
    return c
  }

  busyCells(time: number): Set<number> {
    const busy = new Set<number>()
    for (const sw of this.swings.slice(-4)) if (Math.abs(sw.time - time) < 0.03) for (const n of sw.notes) busy.add(n.x * 3 + n.y)
    return busy
  }

  cellMask(time: number): boolean[] {
    let allowed = new Array<boolean>(12).fill(true)
    const other = this.last[1 - this.hand]
    const together = other !== null && Math.abs(other.time - time) <= 0.03
    if (other && together) {
      const cols = other.notes.map((n) => n.x)
      const lo = Math.min(...cols)
      const hi = Math.max(...cols)
      for (let c = 0; c < 12; c++) {
        const x = Math.floor(c / 3)
        if (cols.includes(x) || (this.hand === 0 && x > lo) || (this.hand === 1 && x < hi)) allowed[c] = false
      }
    }
    const mine = this.last[this.hand]
    if (mine) {
      const gap = Math.max(time - mine.time, 1e-3)
      const p = mine.notes[mine.notes.length - 1]
      const fast = Array.from({ length: 12 }, (_, c) => Math.hypot(Math.floor(c / 3) - p.x, (c % 3) - p.y) / gap > this.speedCap)
      const slow = fast.map((f) => !f)
      if (anyOf(and(slow, allowed))) allowed = and(allowed, slow)
    }
    if (together) for (const c of CENTRE_CELLS) allowed[c] = false
    if (this.wallsPlan.length) {
      const open = wallNoteMask(this.wallsPlan, time).blocked.map((b) => !b)
      if (anyOf(and(allowed, open))) allowed = and(allowed, open)
    }
    const seen = this.hiddenCells(time).map((h) => !h)
    if (anyOf(and(allowed, seen))) allowed = and(allowed, seen)
    const busy = this.busyCells(time)
    const free = Array.from({ length: 12 }, (_, c) => !busy.has(c))
    const both = and(allowed, free)
    return anyOf(both) ? both : free
  }

  hiddenCells(time: number): boolean[] {
    const hidden = new Array<boolean>(12).fill(false)
    const earlier: Note[] = []
    for (const sw of this.swings.slice(-10)) for (const n of sw.notes) if (time - READ_WINDOW[1] - 0.05 < n.time && n.time < time - 0.02) earlier.push(n)
    if (!earlier.length) return hidden
    const at = earlier.map((n) => n.time)
    const ax = earlier.map((n) => (n.x - 1.5) * 0.6)
    const ay = earlier.map((n) => 0.8 + n.y * 0.55)
    const start = time - READ_WINDOW[1]
    const stop = time - READ_WINDOW[0]
    const step = (stop - start) / 5
    const samples = Array.from({ length: 6 }, (_, i) => (i === 5 ? stop : i * step + start))
    for (let c = 0; c < 12; c++) {
      const bx = (Math.floor(c / 3) - 1.5) * 0.6
      const by = 0.8 + (c % 3) * 0.55
      let covered = 0
      for (const sm of samples) {
        const dB = CUT_Z + (time - sm) * this.njs
        const rB = NOTE_SIZE / 2 / dB
        let any = false
        for (let i = 0; i < at.length && !any; i++) {
          if (!(at[i] > sm)) continue
          const dA = CUT_Z + (at[i] - sm) * this.njs
          const rA = NOTE_SIZE / 2 / dA
          any = Math.abs(ax[i] / dA - bx / dB) < rA + 0.3 * rB && Math.abs((ay[i] - EYE) / dA - (by - EYE) / dB) < rA + 0.3 * rB
        }
        if (any) covered++
      }
      hidden[c] = covered >= 3
    }
    return hidden
  }

  parityMask(time: number): boolean[] {
    const pmask = [true, true]
    const prev = this.last[this.hand]
    if (prev && time - prev.time < RESET_SECONDS) pmask[prev.parity] = false
    return pmask
  }

  forcedGesture(): Gesture | null {
    const q = this.forced!
    const h = this.history[q.src]
    const g: Gesture = { parity: h.parity, cut: h.cut, dot: h.dot, cell: h.cell, stack: h.stack, chain: 0, chainSlices: h.chainSlices, chainTail: h.chainTail, arcOut: h.arcOut }
    if (q.mirror) {
      g.cut = MIRROR_CUT[g.cut]
      g.cell = (3 - Math.floor(g.cell / 3)) * 3 + (g.cell % 3)
      g.chainTail = (3 - Math.floor(g.chainTail / 3)) * 3 + (g.chainTail % 3)
    }
    g.stack = Math.min(g.stack, MAX_STACK_EXTRA)
    if (!this.parityMask(this.time)[g.parity]) return null
    if (Math.abs(roll(this.hand, g.parity, g.cut)) > ROLL_LIMIT) return null
    if (!this.cellMask(this.time)[g.cell]) return null
    return g
  }

  chooseGesture(out: GestureLogits, flow?: ArrayLike<number>) {
    let g = this.forced ? this.forcedGesture() : null
    if (this.forced) {
      if (g === null) this.abortCopy()
      else {
        this.queue.shift()
        this.copied++
      }
    }
    if (g === null) {
      const parity = this.s(out.parity, this.parityMask(this.time))
      const cutOk = VECTORS.map((_, cut) => Math.abs(roll(this.hand, parity, cut)) <= ROLL_LIMIT)
      const cmask = this.cellMask(this.time)
      const cutLogits = Float64Array.from(out.cut)
      if (flow) for (let c = 0; c < 8; c++) {
        let best = Infinity
        for (let cell = 0; cell < 12; cell++) if (cmask[cell]) best = Math.min(best, flow[c * 12 + cell])
        cutLogits[c] -= 1.5 * Math.min(50, best)
      }
      const cut = this.s(cutLogits, cutOk)
      const cellLogits = Float64Array.from(out.cell)
      for (const c of CENTRE_CELLS) cellLogits[c] -= CENTRE_PENALTY
      if (flow) for (let c = 0; c < 12; c++) cellLogits[c] -= 1.5 * flow[cut * 12 + c]
      g = {
        parity,
        cut,
        dot: this.rare(out.dot),
        cell: this.s(cellLogits, cmask),
        stack: this.rare(out.stack),
        chain: this.rare(out.chain),
        chainSlices: 0,
        chainTail: 0,
        arcOut: this.rare(out.arcOut),
      }
      g.chain = 0
      g.stack = this.time >= this.musicStart + INTRO_EASE ? Math.min(g.stack, MAX_STACK_EXTRA) : 0
    }
    this.place(g)
    this.forced = null
  }

  place(g: Gesture) {
    const hand = this.hand
    const time = this.time
    const x = Math.floor(g.cell / 3)
    const y = g.cell % 3
    const direction = g.dot ? 8 : g.cut
    const busy = this.busyCells(time)
    const notes: Note[] = [{ time, x, y, hand, direction, angle: 0 }]
    const allowed = this.cellMask(time)
    let cells = stackCells(x, y, g.cut, g.stack)
    if (cells.length < g.stack) {
      const back = stackCells(x, y, OPPOSITE[g.cut], g.stack)
      if (back.length > cells.length) cells = back
    }
    for (const [cx, cy] of cells) {
      const c = cx * 3 + cy
      if (!busy.has(c) && allowed[c] && !CENTRE_CELLS.includes(c)) notes.push({ time: time + 0.001 * notes.length, x: cx, y: cy, hand, direction, angle: 0 })
    }
    const swing: Swing = { time, hand, parity: g.parity, cut: g.cut, cell: g.cell, notes, arcOut: g.arcOut !== 0, token: this.token, follow: this.follow ?? 0, copied: this.forced !== null }
    const prev = this.swings[this.swings.length - 1]
    const isDouble = prev !== undefined && prev.hand !== hand && time - prev.time <= 0.03
    this.doubleRate += DOUBLE_EMA * ((isDouble ? 1 : 0) - this.doubleRate)
    this.swings.push(swing)
    this.last[hand] = swing
    this.history.push({
      hand,
      gap: gapBucket(time - this.prevTime),
      parity: g.parity,
      cut: g.cut,
      dot: g.dot,
      cell: g.cell,
      stack: Math.min(3, notes.length - 1),
      chain: 0,
      chainSlices: 0,
      chainTail: 0,
      arcOut: g.arcOut,
      follow: this.follow ?? 0,
      token: this.token,
      cand: this.cand,
    })
    this.prevIdx = this.cand
    this.prevTime = time
  }

  held(t0: number, t1: number): boolean {
    if (!(ARC_MIN <= t1 - t0 && t1 - t0 <= 3.0)) return false
    let cover = 0
    for (const [a, b] of this.sustains) if (b > t0 && a < t1) cover += Math.max(0, Math.min(b, t1) - Math.max(a, t0))
    return cover >= 0.7 * (t1 - t0)
  }

  result(): { notes: Note[]; arcs: Arc[]; swings: Swing[] } {
    const notes = this.swings
      .flatMap((s) => s.notes)
      .sort((a, b) => a.time - b.time || a.hand - b.hand || a.x - b.x || a.y - b.y)
    const arcs: Arc[] = []
    for (const h of [0, 1] as const) {
      const seq = this.swings.filter((s) => s.hand === h)
      for (let i = 0; i + 1 < seq.length; i++) {
        const a = seq[i]
        const b = seq[i + 1]
        if (!this.held(a.time, b.time)) continue
        const n0 = a.notes[0]
        const n1 = b.notes[0]
        arcs.push({
          time: n0.time,
          tailTime: n1.time,
          hand: h,
          x: n0.x,
          y: n0.y,
          direction: n0.direction !== 8 ? n0.direction : a.cut,
          tailX: n1.x,
          tailY: n1.y,
          tailDirection: n1.direction !== 8 ? n1.direction : b.cut,
        })
      }
    }
    const paired: Arc[] = []
    for (const a of [...arcs].sort((p, q) => p.time - q.time)) {
      if (a.tailTime - a.time < ARC_MIN) continue
      if (paired.some((b) => b.hand !== a.hand && Math.abs(b.time - a.time) < ARC_APART)) continue
      if (paired.some((b) => b.hand === a.hand && b.tailTime > a.time - 0.01)) continue
      paired.push(a)
    }
    return { notes, arcs: paired, swings: this.swings }
  }
}

function f32(data: Float32Array, dims: number[]): ort.Tensor {
  return new ort.Tensor('float32', data, dims)
}

function i64(values: ArrayLike<number>, dims: number[]): ort.Tensor {
  return new ort.Tensor('int64', BigInt64Array.from(Array.from(values, (v) => BigInt(v))), dims)
}

function bool(values: ArrayLike<boolean>, dims: number[]): ort.Tensor {
  return new ort.Tensor('bool', Uint8Array.from(Array.from(values, (v) => (v ? 1 : 0))), dims)
}

export type NoteEncoding = { memory: ort.Tensor; memKv: Record<string, ort.Tensor>; cands: Float32Array; keys: Float32Array }

export function disposeNoteEncoding(encoding: NoteEncoding) {
  encoding.memory.dispose()
  for (const tensor of Object.values(encoding.memKv)) tensor.dispose()
}

export class NoteDecoder {
  readonly sessions: NoteSessions
  readonly C: number
  readonly candTimes32: Float32Array
  readonly difficulty: ort.Tensor
  readonly cond: ort.Tensor
  readonly condScale: ort.Tensor
  readonly songPad: ort.Tensor
  memory!: ort.Tensor
  memKv: Record<string, ort.Tensor> = {}
  cands!: Float32Array
  keys!: Float32Array
  past: Record<string, ort.Tensor> = {}
  position = 0
  prevIdx = 0
  prevTime = 0
  last: [Placed | null, Placed | null] = [null, null]
  lastTime: [number, number] = [0, 0]
  placed: Placed | null = null
  h!: ort.Tensor
  hand!: ort.Tensor
  follow!: ort.Tensor
  mineTokens!: ort.Tensor
  sameTime!: ort.Tensor
  sameOk!: ort.Tensor
  cand = 0

  constructor(sessions: NoteSessions, song: NoteSong, options: NoteOptions) {
    this.sessions = sessions
    this.C = song.candTimes.length
    this.candTimes32 = Float32Array.from(song.candTimes)
    this.difficulty = i64([DIFFICULTIES.indexOf(options.difficulty)], [1])
    this.cond = i64(options.cond ?? [0, 0], [1, 2])
    this.condScale = f32(Float32Array.of(options.condScale ?? 1.0), [1])
    this.songPad = bool(new Array<boolean>(song.sectionLabel.length).fill(false), [1, song.sectionLabel.length])
    for (const t of ['k', 'v']) for (let i = 0; i < DEC_LAYERS; i++) this.past[`past_${t}_${i}`] = f32(new Float32Array(0), [1, DEC_HEADS, 0, HEAD_DIM])
  }

  async prepare(song: NoteSong, encoding?: NoteEncoding) {
    if (encoding) { Object.assign(this, encoding); return }
    const T = song.sectionLabel.length
    const enc = await this.sessions.encoder.run({
      features: f32(song.features, [1, T, song.features.length / T]),
      section_label: i64(song.sectionLabel, [1, T]),
      difficulty: this.difficulty,
      song_pad: this.songPad,
    })
    this.memory = enc.memory
    this.memKv = await this.sessions.cross_kv.run({ memory: this.memory })
    const c = await this.sessions.candidates.run({
      memory: this.memory,
      cand_feats: f32(song.candFeats, [1, this.C, song.candFeats.length / this.C]),
      cand_tokens: i64(song.candTokens, [1, this.C]),
    })
    this.cands = Float32Array.from(c.cands.data as Float32Array)
    this.keys = Float32Array.from(c.keys.data as Float32Array)
    c.cands.dispose(); c.keys.dispose()
  }

  stateTokens(sw: Placed | null): number[] {
    if (!sw) return [0, 0, 0, 0]
    return [sw.cell + 1, sw.cut + 1, sw.parity + 1, EXIT[sw.cell][sw.cut] + 1]
  }

  async step(): Promise<{ h: Float32Array; handLogits: Float32Array }> {
    const prev = new Array<number>(PREV_FIELDS.length).fill(0)
    let rep = new Float32Array(DC)
    const state = new Array<number>(10).fill(0)
    if (this.placed) {
      PREV_FIELDS.forEach((k, i) => (prev[i] = this.placed![k]))
      rep = this.cands.slice(this.placed.cand * DC, (this.placed.cand + 1) * DC)
      for (const h of [0, 1]) {
        const sw = this.last[h]
        if (!sw) continue
        this.stateTokens(sw).forEach((v, i) => (state[h * 5 + i] = v))
        state[h * 5 + 4] = gapBucket(this.prevTime - this.lastTime[h]) + 1
      }
    }
    const out = await this.sessions.step.run({
      prev_tokens: i64(prev, [1, PREV_FIELDS.length]),
      prev_cand_repr: f32(rep, [1, DC]),
      state_tokens: i64(state, [1, 2, 5]),
      position: i64([this.position], [1]),
      difficulty: this.difficulty,
      cond: this.cond,
      cond_scale: this.condScale,
      song_pad: this.songPad,
      ...this.memKv,
      ...this.past,
    })
    for (const old of Object.values(this.past)) old.dispose()
    this.h?.dispose()
    for (const [k, v] of Object.entries(out)) if (k.startsWith('present_')) this.past[k.replace('present_', 'past_')] = v
    this.position++
    this.h = out.h
    const handLogits = Float32Array.from(out.hand_logits.data as Float32Array)
    out.hand_logits.dispose()
    return { h: out.h.data as Float32Array, handLogits }
  }

  async followLogits(hand: number): Promise<Float32Array> {
    this.hand?.dispose()
    this.hand = i64([hand], [1])
    const out = await this.sessions.follow.run({ h: this.h, hand: this.hand })
    const data = Float32Array.from(out.follow_logits.data as Float32Array)
    out.follow_logits.dispose()
    return data
  }

  window(): Int32Array {
    return Int32Array.from({ length: WINDOW }, (_, w) => this.prevIdx + w)
  }

  async pointerLogits(hand: number, follow: number): Promise<Float32Array> {
    for (const t of [this.follow, this.mineTokens, this.sameTime, this.sameOk]) t?.dispose()
    this.follow = i64([follow], [1])
    const mine = this.last[hand]
    this.mineTokens = i64(this.stateTokens(mine), [1, 4])
    this.sameTime = f32(Float32Array.of(mine ? this.lastTime[hand] : 0), [1])
    this.sameOk = bool([mine !== null], [1])
    const kw = new Float32Array(WINDOW * DK)
    const tw = new Float32Array(WINDOW).fill(this.candTimes32[this.C - 1])
    const n = Math.min(WINDOW, this.C - this.prevIdx)
    kw.set(this.keys.subarray(this.prevIdx * DK, (this.prevIdx + n) * DK))
    tw.set(this.candTimes32.subarray(this.prevIdx, this.prevIdx + n))
    const valid = Array.from({ length: WINDOW }, (_, w) => this.prevIdx + w < this.C)
    const out = await this.sessions.pointer.run({
      h: this.h,
      hand: this.hand,
      follow: this.follow,
      mine_tokens: this.mineTokens,
      keys_window: f32(kw, [1, WINDOW, DK]),
      times_window: f32(tw, [1, WINDOW]),
      valid: bool(valid, [1, WINDOW]),
      prev_time: f32(Float32Array.of(this.prevTime), [1]),
      same_time: this.sameTime,
      same_ok: this.sameOk,
    })
    const data = Float32Array.from(out.pointer_logits.data as Float32Array)
    out.pointer_logits.dispose()
    return data
  }

  async gestureLogits(cand: number): Promise<GestureLogits> {
    this.cand = cand
    const out = await this.sessions.gesture.run({
      h: this.h,
      hand: this.hand,
      follow: this.follow,
      mine_tokens: this.mineTokens,
      now: f32(Float32Array.of(this.candTimes32[cand]), [1]),
      same_time: this.sameTime,
      same_ok: this.sameOk,
      chosen_cand_repr: f32(this.cands.slice(cand * DC, (cand + 1) * DC), [1, DC]),
    })
    const get = (k: string) => { const t = out[`${k}_logits`]; const data = Float32Array.from(t.data as Float32Array); t.dispose(); return data }
    return {
      parity: get('parity'),
      cut: get('cut'),
      dot: get('dot'),
      cell: get('cell'),
      stack: get('stack'),
      chain: get('chain'),
      chainSlices: get('chain_slices'),
      chainTail: get('chain_tail'),
      arcOut: get('arc_out'),
    }
  }

  dispose() {
    for (const tensor of [...Object.values(this.past), this.h, this.hand, this.follow, this.mineTokens, this.sameTime, this.sameOk,
      this.difficulty, this.cond, this.condScale, this.songPad]) tensor?.dispose()
  }

  place(placed: Placed, time: number) {
    this.placed = placed
    this.last[placed.hand] = placed
    this.lastTime[placed.hand] = time
    this.prevIdx = placed.cand
    this.prevTime = time
  }
}

export function noteModelGroup(backend: Backend): string {
  return backend === 'webgpu' ? 'flow-1-v8-notes-fp16' : 'flow-1-v8-notes-int8'
}

export async function loadNoteSessions(backend: Backend, group = noteModelGroup(backend), onProgress?: (fraction: number) => void): Promise<NoteSessions> {
  const out = {} as NoteSessions
  for (const [i, name] of NOTE_GRAPHS.entries()) {
    out[name] = await loadModel(group, `${name}.onnx`, backend, (f) => onProgress?.((i + f) / NOTE_GRAPHS.length))
  }
  return out
}

export async function prepareNoteSong(sessions: NoteSessions, song: NoteSong, options: NoteOptions): Promise<NoteEncoding> {
  const dec = new NoteDecoder(sessions, song, options)
  try {
    await dec.prepare(song)
    return { memory: dec.memory, memKv: dec.memKv, cands: dec.cands, keys: dec.keys }
  } catch (error) {
    if (dec.memory) disposeNoteEncoding(dec)
    throw error
  } finally { dec.dispose() }
}

export async function generateNotes(sessions: NoteSessions, song: NoteSong, options: NoteOptions): Promise<{ notes: Note[]; arcs: Arc[]; swings: Swing[] }> {
  const check = () => {
    if (options.signal?.aborted) throw new DOMException('cancelled', 'AbortError')
  }
  const state = new NoteState(song, options)
  const dec = new NoteDecoder(sessions, song, options)
  check()
  try {
  await dec.prepare(song, options.encoding)
  const maxSwings = options.maxSwings ?? 4000
  for (let position = 0; position < maxSwings; position++) {
    check()
    const { handLogits } = await dec.step()
    const hand = state.chooseHand(handLogits)
    if (hand === null) break
    const follow = state.chooseFollow(await dec.followLogits(hand))
    const cand = state.chooseCandidate(await dec.pointerLogits(hand, follow), dec.window())
    if (cand === null) break
    const gesture = await dec.gestureLogits(cand)
    const flow = options.flowCosts && !(state.forced && state.forcedGesture()) ? await options.flowCosts(state) : undefined
    state.chooseGesture(gesture, flow)
    dec.place(state.history[state.history.length - 1], state.time)
    options.onProgress?.(Math.min(1, state.prevTime / song.duration))
  }
  return state.result()
  } finally {
    dec.dispose()
    if (!options.encoding && dec.memory) disposeNoteEncoding(dec)
  }
}
