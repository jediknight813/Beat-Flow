import type { Style } from './package'
import { Pcg64 } from './pcg64'

export type StyleAudio = {
  tempo: number
  stemRms: Float32Array[]
  vocalPeriodicity: Float32Array
}

export type CoverPixels = { rgb64: Uint8Array; rgb48: Uint8Array }

export type EnvironmentReference = {
  cover: number[][]
  audio: number[][]
  mapper_frequency: Record<string, number>
}

export type Category = 'moody' | 'energetic' | 'pop' | 'rock'
export type Colour = { r: number; g: number; b: number; a: number }
export type Cluster = { rgb: [number, number, number]; share: number; h: number; s: number; v: number; vivid: number }
export type ColourScheme = Style['colors'] & { source: Record<string, string | number> }
export type StyleReport = Style & {
  category: Category
  tags: Record<string, number>
  top3: string[]
  cover: boolean
  source: Record<string, string | number>
}

const MIN_HUE_GAP = 0.17
const FALLBACK: Record<Category, [number, number]> = { moody: [0.83, 0.52], energetic: [0.97, 0.52], pop: [0.92, 0.47], rock: [0.02, 0.58] }
const AFFINITY: [string, Record<string, number>][] = [
  ['BillieEnvironment', { dark: 1, calm: 1, cool: 1, mono: 1, vocal: 0.5 }],
  ['HalloweenEnvironment', { dark: 2, warm: 1, heavy: 1 }],
  ['NiceEnvironment', { cool: 1, pastel: 1, calm: 1 }],
  ['BigMirrorEnvironment', { bright: 1, vivid: 1, energetic: 0.5 }],
  ['KaleidoscopeEnvironment', { vivid: 2, pastel: 0.5, dark: 0.5 }],
  ['TriangleEnvironment', { vivid: 1, cool: 1, energetic: 0.5 }],
  ['OriginsEnvironment', { warm: 1, bright: 1, calm: 1 }],
  ['DefaultEnvironment', {}],
  ['PanicEnvironment', { energetic: 2, vivid: 1, warm: 0.5 }],
  ['RocketEnvironment', { energetic: 1, warm: 1, bright: 1 }],
  ['MonstercatEnvironment', { energetic: 2, cool: 1, vivid: 1 }],
  ['CrabRaveEnvironment', { energetic: 1, bright: 2, warm: 1 }],
  ['KDAEnvironment', { vivid: 1, warm: 1, energetic: 1, vocal: 0.5 }],
  ['BTSEnvironment', { pastel: 1, bright: 1, vocal: 2 }],
  ['GagaEnvironment', { vivid: 1, pastel: 1, vocal: 1, bright: 1 }],
  ['FitBeatEnvironment', { bright: 1, pastel: 1, energetic: 1 }],
  ['InterscopeEnvironment', { dark: 1, cool: 1, vocal: 1 }],
  ['TimbalandEnvironment', { dark: 1, energetic: 1, vocal: 1 }],
  ['SkrillexEnvironment', { energetic: 2, heavy: 1, vivid: 1, dark: 0.5 }],
  ['DragonsEnvironment', { heavy: 2, warm: 1, dark: 1 }],
  ['GreenDayEnvironment', { heavy: 1, energetic: 1 }],
  ['LinkinParkEnvironment', { heavy: 2, dark: 1, cool: 1 }],
]

export const ENVIRONMENTS = AFFINITY.map(([name]) => name)

let reference: Promise<EnvironmentReference> | null = null

export function loadEnvironmentReference(): Promise<EnvironmentReference> {
  reference ??= fetch(`${import.meta.env.BASE_URL}data/environment-reference.json`).then((r) => r.json() as Promise<EnvironmentReference>)
  return reference
}

function scaled(cover: ImageBitmap, size: number): Uint8Array {
  const canvas = new OffscreenCanvas(size, size)
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(cover, 0, 0, size, size)
  const data = ctx.getImageData(0, 0, size, size).data
  const out = new Uint8Array(size * size * 3)
  for (let i = 0; i < size * size; i++) {
    out[i * 3] = data[i * 4]
    out[i * 3 + 1] = data[i * 4 + 1]
    out[i * 3 + 2] = data[i * 4 + 2]
  }
  return out
}

export function coverPixels(cover: ImageBitmap): CoverPixels {
  return { rgb64: scaled(cover, 64), rgb48: scaled(cover, 48) }
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const range = max - min
  if (range === 0) return [0, 0, max]
  const rc = (max - r) / range
  const gc = (max - g) / range
  const bc = (max - b) / range
  const h = r === max ? bc - gc : g === max ? 2 + rc - bc : 4 + gc - rc
  return [(((h / 6) % 1) + 1) % 1, range / max, max]
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  if (s === 0) return [v, v, v]
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - s * f)
  const t = v * (1 - s * (1 - f))
  switch (i % 6) {
    case 0: return [v, t, p]
    case 1: return [q, v, p]
    case 2: return [p, v, t]
    case 3: return [p, q, v]
    case 4: return [t, p, v]
    default: return [v, p, q]
  }
}

export function coverPalette(rgb64: Uint8Array, k = 6, seed = 0): Cluster[] {
  const n = rgb64.length / 3
  const px = new Float32Array(rgb64.length)
  for (let i = 0; i < rgb64.length; i++) px[i] = rgb64[i] / 255
  const cent = new Pcg64(seed).choice(n, k).map((i) => [px[i * 3], px[i * 3 + 1], px[i * 3 + 2]])
  const label = new Int32Array(n)
  for (let iter = 0; iter < 12; iter++) {
    for (let i = 0; i < n; i++) {
      let best = 0
      let bestD = Infinity
      for (let j = 0; j < k; j++) {
        const d = (px[i * 3] - cent[j][0]) ** 2 + (px[i * 3 + 1] - cent[j][1]) ** 2 + (px[i * 3 + 2] - cent[j][2]) ** 2
        if (d < bestD) {
          bestD = d
          best = j
        }
      }
      label[i] = best
    }
    const sum = cent.map(() => [0, 0, 0])
    const cnt = new Array<number>(k).fill(0)
    for (let i = 0; i < n; i++) {
      const j = label[i]
      cnt[j]++
      for (let ch = 0; ch < 3; ch++) sum[j][ch] += px[i * 3 + ch]
    }
    for (let j = 0; j < k; j++) if (cnt[j]) cent[j] = sum[j].map((v) => Math.fround(v / cnt[j]))
  }
  const cnt = new Array<number>(k).fill(0)
  for (let i = 0; i < n; i++) cnt[label[i]]++
  return cent
    .map((c, j) => {
      const [h, s, v] = rgbToHsv(c[0], c[1], c[2])
      const share = cnt[j] / n
      return { rgb: [c[0], c[1], c[2]] as [number, number, number], share, h, s, v, vivid: share * s * Math.min(1, v * 1.5) }
    })
    .sort((a, b) => b.vivid - a.vivid)
}

function hueGap(a: number, b: number): number {
  const d = Math.abs(a - b) % 1
  return Math.min(d, 1 - d)
}

function warmth(h: number): number {
  return (Math.cos(2 * Math.PI * (h - 0.05)) + 1) / 2
}

function rgb(h: number, s: number, v: number): Colour {
  const [r, g, b] = hsvToRgb(((h % 1) + 1) % 1, Math.min(1, s), Math.min(1, v))
  const round = (x: number) => Math.round(x * 1e4) / 1e4
  return { r: round(r), g: round(g), b: round(b), a: 1 }
}

function scheme(ha: number, sa: number, hb: number, sb: number, vb: number, ho: number | null, rule: string): ColourScheme {
  const o = ho ?? (ha + hb) / 2 + 0.5
  return {
    saberA: rgb(ha, sa, 1),
    saberB: rgb(hb, sb, vb),
    env0: rgb(ha, Math.min(0.9, sa + 0.1), 0.9),
    env1: rgb(hb, Math.min(0.9, sb + 0.1), 0.9),
    env0boost: rgb(ha + 0.06, Math.min(1, sa + 0.15), 1),
    env1boost: rgb(hb - 0.06, Math.min(1, sb + 0.15), 1),
    obstacles: rgb(o, 0.6, 0.85),
    source: { left_hue: Math.round(ha * 360), left_sat: Math.round(sa * 100) / 100, right_hue: Math.round(hb * 360), right_sat: Math.round(sb * 100) / 100, rule },
  }
}

export function colorsFromCover(palette: Cluster[]): ColourScheme {
  const pal = palette.filter((c) => c.s > 0.15 && c.v > 0.15)
  if (!pal.length) return scheme(0.08, 0.08, 0.6, 0.45, 1, null, 'greyscale cover: paper and ink')
  const a = pal[0]
  const b = pal.slice(1).find((c) => hueGap(c.h, a.h) >= MIN_HUE_GAP)
  if (!b) return scheme(a.h, Math.max(0.75, a.s), a.h, 0.12, 1, null, 'monochrome cover: colour and its tint')
  const [ha, hb] = warmth(a.h) >= warmth(b.h) ? [a.h, b.h] : [b.h, a.h]
  const other = pal.find((c) => hueGap(c.h, ha) >= 0.08 && hueGap(c.h, hb) >= 0.08)
  return scheme(ha, Math.max(0.75, a.s), hb, Math.max(0.75, b.s), 1, other ? other.h : null, 'two cover hues')
}

export function fallbackColors(category: Category): ColourScheme {
  const [ha, hb] = FALLBACK[category]
  return {
    saberA: rgb(ha, 0.85, 1),
    saberB: rgb(hb, 0.85, 1),
    env0: rgb(ha, 0.85, 0.9),
    env1: rgb(hb, 0.85, 0.9),
    env0boost: rgb(ha + 0.06, 0.9, 1),
    env1boost: rgb(hb - 0.06, 0.9, 1),
    obstacles: rgb(ha + 0.5, 0.7, 0.85),
    source: { fallback: category },
  }
}

export function coverFeatures(rgb48: Uint8Array): number[] {
  const n = rgb48.length / 3
  const hist = new Array<number>(12).fill(0)
  let wSum = 0
  let sSum = 0
  let vSum = 0
  let grey = 0
  let dark = 0
  for (let i = 0; i < n; i++) {
    const r = Math.fround(rgb48[i * 3] / 255)
    const g = Math.fround(rgb48[i * 3 + 1] / 255)
    const b = Math.fround(rgb48[i * 3 + 2] / 255)
    const mx = Math.max(r, g, b)
    const mn = Math.min(r, g, b)
    const s = mx > 0 ? (mx - mn) / Math.max(mx, 1e-6) : 0
    const d = Math.max(mx - mn, 1e-6)
    const h = (mx === r ? ((((g - b) / d) % 6) + 6) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4) / 6
    const w = s * mx
    hist[Math.min(11, Math.max(0, Math.floor(h * 12)))] += w
    wSum += w
    sSum += s
    vSum += mx
    if (s < 0.12) grey++
    if (mx < 0.2) dark++
  }
  return [...hist.map((v) => v / Math.max(1e-6, wSum)), sSum / n, vSum / n, grey / n, dark / n, 1].map(Math.fround)
}

function percentile(sorted: Float32Array, q: number): number {
  const pos = (q / 100) * (sorted.length - 1)
  const i = Math.floor(pos)
  const f = pos - i
  return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i]
}

export function audioFeatures(audio: StyleAudio): number[] {
  const frames = audio.stemRms[0].length
  const e = new Float32Array(frames)
  const means = audio.stemRms.map((s) => {
    let sum = 0
    for (let i = 0; i < frames; i++) {
      sum += s[i]
      e[i] = Math.fround(e[i] + s[i])
    }
    return sum / frames
  })
  const sorted = e.slice().sort()
  const total = means.reduce((a, b) => a + b, 0) + 1e-6
  let vocal = 0
  for (let i = 0; i < frames; i++) if (audio.vocalPeriodicity[i] > 0.5 && audio.stemRms[3][i] > 0.02) vocal++
  const p10 = percentile(sorted, 10)
  const p75 = percentile(sorted, 75)
  const p90 = percentile(sorted, 90)
  return [audio.tempo / 150, p75, Math.log((p90 + 1e-4) / (p10 + 1e-4)), ...means.map((m) => m / total), vocal / frames].map(Math.fround)
}

function rank(ref: number[][], column: number, v: number): number {
  let n = 0
  for (const row of ref) if (row[column] < v) n++
  return n / ref.length
}

export function songTags(af: number[], cf: number[] | null, ref: EnvironmentReference): Record<string, number> {
  const A = ref.audio
  const energy = (rank(A, 0, af[0]) + 2 * rank(A, 1, af[1]) + rank(A, 3, af[3])) / 4
  const tags: Record<string, number> = {
    energetic: energy,
    calm: 1 - energy,
    heavy: ((rank(A, 3, af[3]) + rank(A, 2, af[2])) / 2) * (1 - rank(A, 6, af[6]) * 0.5),
    vocal: (rank(A, 6, af[6]) + rank(A, 7, af[7])) / 2,
  }
  if (cf) {
    const C = ref.cover
    const hist = cf.slice(0, 12)
    const val = rank(C, 13, cf[13])
    const sat = rank(C, 12, cf[12])
    Object.assign(tags, {
      dark: Math.max(rank(C, 15, cf[15]), 1 - val),
      bright: val,
      vivid: sat,
      pastel: val * (1 - sat) * 2,
      warm: hist[0] + hist[1] + hist[2] + hist[11],
      cool: hist[5] + hist[6] + hist[7] + hist[8],
      mono: cf[14] > 0.6 || Math.max(...hist) > 0.7 ? 1 : 0,
    })
  }
  return tags
}

async function titleHash(title: string): Promise<number> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(title))
  return new DataView(digest).getUint32(0)
}

export function chooseEnvironment(tags: Record<string, number>, ref: EnvironmentReference, pick: number): { environment: string; category: Category; top3: string[] } {
  const freq = ref.mapper_frequency
  const total = Object.values(freq).reduce((a, b) => a + b, 0)
  const score = new Map<string, number>()
  for (const [env, aff] of AFFINITY) {
    const entries = Object.entries(aff)
    const weight = entries.reduce((a, [, w]) => a + w, 0)
    const match = entries.reduce((a, [t, w]) => a + w * (tags[t] ?? 0), 0) / Math.max(1, weight)
    score.set(env, match + 0.05 * Math.log(((freq[env] ?? 0) + 1) / total))
  }
  const top3 = AFFINITY.map(([env]) => env).sort((a, b) => score.get(b)! - score.get(a)!).slice(0, 3)
  const environment = score.get(top3[0])! - score.get(top3[1])! > 0.15 ? top3[0] : top3[pick % 3]
  const category: Category = (tags.dark ?? 0) > 0.6 || tags.calm > 0.65 ? 'moody' : tags.energetic > 0.65 ? 'energetic' : tags.heavy > 0.6 ? 'rock' : 'pop'
  return { environment, category, top3 }
}

export async function styleFromPixels(pixels: CoverPixels | null, audio: StyleAudio, title: string, ref: EnvironmentReference): Promise<StyleReport> {
  const tags = songTags(audioFeatures(audio), pixels ? coverFeatures(pixels.rgb48) : null, ref)
  const { environment, category, top3 } = chooseEnvironment(tags, ref, await titleHash(title))
  const { source, ...colors } = pixels ? colorsFromCover(coverPalette(pixels.rgb64)) : fallbackColors(category)
  const rounded = Object.fromEntries(Object.entries(tags).map(([k, v]) => [k, Math.round(v * 100) / 100]))
  return { environment, colors, category, tags: rounded, top3, cover: pixels !== null, source }
}

export async function songStyle(cover: ImageBitmap | null, audio: StyleAudio, title: string): Promise<StyleReport> {
  return styleFromPixels(cover ? coverPixels(cover) : null, audio, title, await loadEnvironmentReference())
}
