import { brandedSongTitle } from '../branding'
import JSZip from 'jszip'
import { createOggEncoder } from 'wasm-media-encoders'
import type { Chart, Difficulty } from './types'

export const LEAD = 1.5
export const RANKS: Record<Difficulty, number> = { Expert: 7, ExpertPlus: 9 }
export const NJS: Record<Difficulty, number> = { Expert: 16, ExpertPlus: 18 }
const JUMP_DISTANCE: Record<Difficulty, number> = { Expert: 23.4, ExpertPlus: 21.8 }

type Segment = { beat: number; seconds: number; bpm: number }

export type Style = {
  environment: string
  colors: Record<'saberA' | 'saberB' | 'env0' | 'env1' | 'env0boost' | 'env1boost' | 'obstacles', { r: number; g: number; b: number; a?: number }>
}

export type LightEvent = { time: number; type: number; value: number; float: number }

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function tempoMap(beats: number[], toleranceMs = 4): Segment[] {
  const segments: Segment[] = []
  let start = 0
  while (start < beats.length - 1) {
    let end = start + 1
    while (end + 1 < beats.length) {
      const span = beats.slice(start, end + 2)
      const period = (span[span.length - 1] - span[0]) / (span.length - 1)
      let worst = 0
      span.forEach((v, i) => (worst = Math.max(worst, Math.abs(span[0] + period * i - v))))
      if (worst * 1000 > toleranceMs) break
      end++
    }
    const period = (beats[end] - beats[start]) / (end - start)
    segments.push({ beat: start, seconds: beats[start], bpm: 60 / period })
    start = end
  }
  return segments
}

export class BeatMap {
  segments: Segment[]
  baseBpm: number
  firstBeat: number

  constructor(grid: number[], lead: number) {
    const g = grid.map((b) => b + lead)
    this.segments = tempoMap(g)
    this.baseBpm = Math.round((60 / median(g.slice(1).map((v, i) => v - g[i]))) * 1000) / 1000
    this.firstBeat = (g[0] * this.baseBpm) / 60
  }

  songBeat(index: number): number {
    return this.firstBeat + index
  }

  toBeat(t: number): number {
    if (t < this.segments[0].seconds) return (t * this.baseBpm) / 60
    let seg = this.segments[0]
    for (const s of this.segments) {
      if (s.seconds <= t) seg = s
      else break
    }
    return this.songBeat(seg.beat) + ((t - seg.seconds) * seg.bpm) / 60
  }

  bpmEvents() {
    return this.segments.map((s) => ({ b: this.songBeat(s.beat), m: s.bpm }))
  }
}

export function jumpOffset(difficulty: Difficulty, bpm: number): number {
  const njs = NJS[difficulty]
  const beat = 60 / bpm
  let hj = 4
  while (njs * beat * hj > 17.999) hj /= 2
  const target = JUMP_DISTANCE[difficulty] / (2 * njs * beat)
  return Math.round((Math.max(0.25, target) - hj) * 1000) / 1000
}

export function toV3(chart: Chart, beats: BeatMap, lights: LightEvent[] = [], boosts: [number, boolean][] = []) {
  const b = (t: number) => beats.toBeat(t + LEAD)
  return {
    version: '3.3.0',
    bpmEvents: beats.bpmEvents(),
    rotationEvents: [],
    colorNotes: chart.notes.map((n) => ({ b: b(n.time), x: n.x, y: n.y, c: n.hand, d: n.direction, a: Math.round(n.angle ?? 0) })),
    bombNotes: [],
    obstacles: chart.walls.map((w) => ({ b: b(w.time), x: w.x, y: w.y, d: b(w.time + w.duration) - b(w.time), w: w.width, h: w.height })),
    sliders: chart.arcs.map((a) => ({
      b: b(a.time), c: a.hand, x: a.x, y: a.y, d: a.direction, mu: 1,
      tb: b(a.tailTime), tx: a.tailX, ty: a.tailY, tc: a.tailDirection, tmu: 1, m: 0,
    })),
    burstSliders: [],
    basicBeatmapEvents: lights.length
      ? lights.map((e) => ({ b: b(e.time), et: e.type, i: e.value, f: e.float }))
      : [0, 1, 2, 3, 4].map((et) => ({ b: 0, et, i: 1, f: 0.5 })),
    colorBoostBeatmapEvents: boosts.map(([t, o]) => ({ b: b(t), o })),
    lightColorEventBoxGroups: [],
    lightRotationEventBoxGroups: [],
    lightTranslationEventBoxGroups: [],
    basicEventTypesWithKeywords: { d: [] },
    useNormalEventsAsCompatibleEvents: true,
  }
}

export async function encodeOgg(stereo: [Float32Array, Float32Array], sampleRate: number, lead: number): Promise<Uint8Array> {
  const encoder = await createOggEncoder()
  encoder.configure({ sampleRate, channels: 2, vbrQuality: 4 })
  const pad = Math.round(lead * sampleRate)
  const left = new Float32Array(pad + stereo[0].length)
  const right = new Float32Array(pad + stereo[1].length)
  left.set(stereo[0], pad)
  right.set(stereo[1], pad)
  const parts: Uint8Array[] = []
  const step = sampleRate * 10
  for (let i = 0; i < left.length; i += step) parts.push(encoder.encode([left.subarray(i, i + step), right.subarray(i, i + step)]).slice())
  parts.push(encoder.finalize().slice())
  const total = parts.reduce((n, p) => n + p.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}

export async function defaultCover(title: string): Promise<Blob> {
  const size = 512
  const canvas = new OffscreenCanvas(size, size)
  const ctx = canvas.getContext('2d')!
  const hue = [...title].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7)
  const grad = ctx.createLinearGradient(0, 0, size, size)
  grad.addColorStop(0, `hsl(${hue} 70% 45%)`)
  grad.addColorStop(1, `hsl(${(hue + 60) % 360} 70% 25%)`)
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  ctx.fillStyle = 'rgba(255,255,255,0.92)'
  ctx.font = '600 44px system-ui, sans-serif'
  ctx.textBaseline = 'bottom'
  const words = title.split(' ')
  const lines: string[] = []
  let line = ''
  for (const w of words) {
    const next = line ? `${line} ${w}` : w
    if (ctx.measureText(next).width > size - 64 && line) {
      lines.push(line)
      line = w
    } else line = next
  }
  if (line) lines.push(line)
  lines.slice(-4).forEach((l, i, arr) => ctx.fillText(l, 32, size - 32 - (arr.length - 1 - i) * 52))
  return canvas.convertToBlob({ type: 'image/png' })
}

export type PackageInput = {
  title: string
  artist: string
  version: string
  duration: number
  grid: number[]
  charts: { chart: Chart; lights?: LightEvent[]; boosts?: [number, boolean][] }[]
  audio: [Float32Array, Float32Array]
  sampleRate: number
  cover?: Blob
  style?: Style
  report?: unknown
}

export async function buildPackage(input: PackageInput, onProgress?: (fraction: number) => void): Promise<Blob> {
  const beats = new BeatMap(input.grid, LEAD)
  const zip = new JSZip()
  const entries = input.charts
    .map(({ chart, lights, boosts }) => {
      zip.file(`${chart.difficulty}.dat`, JSON.stringify(toV3(chart, beats, lights, boosts)))
      const entry: Record<string, unknown> = {
        _difficulty: chart.difficulty,
        _difficultyRank: RANKS[chart.difficulty],
        _beatmapFilename: `${chart.difficulty}.dat`,
        _noteJumpMovementSpeed: NJS[chart.difficulty],
        _noteJumpStartBeatOffset: jumpOffset(chart.difficulty, beats.baseBpm),
      }
      if (input.style) {
        const c = input.style.colors
        const rgb = (x: { r: number; g: number; b: number }) => ({ r: x.r, g: x.g, b: x.b })
        entry._beatmapColorSchemeIdx = 0
        entry._environmentNameIdx = 0
        entry._customData = {
          _colorLeft: rgb(c.saberA), _colorRight: rgb(c.saberB), _envColorLeft: rgb(c.env0), _envColorRight: rgb(c.env1),
          _envColorLeftBoost: rgb(c.env0boost), _envColorRightBoost: rgb(c.env1boost), _obstacleColor: rgb(c.obstacles),
        }
      }
      return entry
    })
    .sort((a, b) => (a._difficultyRank as number) - (b._difficultyRank as number))
  onProgress?.(0.1)
  const cover = input.cover ?? (await defaultCover(input.title))
  const coverName = cover.type === 'image/jpeg' ? 'cover.jpg' : 'cover.png'
  const info: Record<string, unknown> = {
    _version: '2.1.0',
    _songName: brandedSongTitle(input.title),
    _songSubName: '',
    _songAuthorName: input.artist,
    _levelAuthorName: 'BeatFlow',
    _beatsPerMinute: beats.baseBpm,
    _songTimeOffset: 0,
    _shuffle: 0,
    _shufflePeriod: 0.5,
    _songFilename: 'song.egg',
    _coverImageFilename: coverName,
    _previewStartTime: Math.min(30, input.duration / 3),
    _previewDuration: 10,
    _environmentName: input.style?.environment ?? 'DefaultEnvironment',
    _allDirectionsEnvironmentName: 'GlassDesertEnvironment',
    _difficultyBeatmapSets: [{ _beatmapCharacteristicName: 'Standard', _difficultyBeatmaps: entries }],
  }
  if (input.style) {
    const c = input.style.colors
    info._environmentNames = [input.style.environment]
    info._colorSchemes = [{
      useOverride: true,
      colorScheme: {
        colorSchemeId: `BeatFlow ${input.title}`, saberAColor: c.saberA, saberBColor: c.saberB, environmentColor0: c.env0,
        environmentColor1: c.env1, obstaclesColor: c.obstacles, environmentColor0Boost: c.env0boost, environmentColor1Boost: c.env1boost,
      },
    }]
  }
  zip.file('Info.dat', JSON.stringify(info, null, 2))
  zip.file(coverName, new Uint8Array(await cover.arrayBuffer()))
  if (input.report) zip.file('generation.json', JSON.stringify(input.report, null, 2))
  onProgress?.(0.2)
  zip.file('song.egg', await encodeOgg(input.audio, input.sampleRate, LEAD))
  onProgress?.(0.9)
  const blob = await zip.generateAsync({ type: 'blob', compression: 'STORE' })
  onProgress?.(1)
  return blob
}
