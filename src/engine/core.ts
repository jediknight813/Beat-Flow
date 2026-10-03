import { brandedSongTitle } from '../branding'
import type { CandidateSummary, Chart, ChartSummary, Detected, Difficulty, Note, Palette, Progress, Settings, StageName } from './types'
import { ort, type Backend } from './models'
import { median, resample, roundDecimals, toFloat16 } from './dsp'
import { attacks as findAttacks, beatSpectrogram, frameTotal, SR, stemFeatures, storedFeatures, vocalSyllables } from './features'
import { beatThisProvider, chunkStarts, detectBeats } from './beats'
import { crepeProbs, decodePitch, maskPitch, vocalActive } from './pitch'
import { separate, DEMUCS_SR } from './stems'
import { songGrid } from './grid'
import { sections as findSections, type Section } from './sections'
import { songTokens, FEATURES, type FrameArrays } from './tokens'
import { candidates as findCandidates } from './candidates'
import { musicBounds, repeatSegments } from './structure'
import { flowCosts, flowMetrics, selectFlowCandidate, type StrainModel } from './flow'
import { generateNotes, prepareNoteSong, disposeNoteEncoding, noteModelGroup, NOTE_GRAPHS, type NoteSessions, type NoteSong } from './notes'
import { sampleLights, type Intensity } from './lights'
import { styleFromPixels, type CoverPixels, type EnvironmentReference } from './style'
import { buildPackage, LEAD, type LightEvent, type Style } from './package'
import { fileTags, nameTags } from './tags'
import type { Syllable } from './vocals'

export const VERSION = 'flow-1-v8'
const QUALITY = 0
const COND_SCALE = 1.0
const TOP_P = [0.95]
const DIFF_INDEX: Record<Difficulty, number> = { Expert: 0, ExpertPlus: 1 }

export type StereoAudio = { left: Float32Array; right: Float32Array; sampleRate: number }

export type Deps = {
  backend: Backend
  model(group: string, file?: string, onProgress?: (fraction: number) => void): Promise<ort.InferenceSession>
  release?(group: string): Promise<void>
  strainData(): Promise<StrainModel>
  environmentReference(): Promise<EnvironmentReference>
  coverPixels(cover: Blob): Promise<CoverPixels | null>
}

export type Analysis = {
  duration: number
  frames: number
  tempo: number
  beats: number[]
  downbeats: number[]
  attacks: number[]
  syllables: Syllable[]
  melF32: Float32Array
  arrays: FrameArrays
}

export type SongFile = { name: string; bytes: Uint8Array | null }

export type CoreResult = {
  charts: Chart[]
  zip: Blob
  fileName: string
  duration: number
  detected: Detected
  timings: Partial<Record<StageName, number>>
}

export type Candidate = CandidateSummary & { chart: Chart }

type Reporter = {
  (stage: StageName, fraction: number, detail?: string): void
  check(): void
  finish(): Partial<Record<StageName, number>>
}

export function reporter(onProgress: (p: Progress) => void, signal?: AbortSignal): Reporter {
  const timings: Partial<Record<StageName, number>> = {}
  let current: StageName | null = null
  let since = performance.now()
  const close = () => {
    const now = performance.now()
    if (current) timings[current] = (timings[current] ?? 0) + now - since
    since = now
  }
  const report = ((stage: StageName, fraction: number, detail?: string) => {
    if (signal?.aborted) throw new DOMException('cancelled', 'AbortError')
    if (stage !== current) {
      close()
      current = stage
    }
    onProgress({ stage, fraction: Math.max(0, Math.min(1, fraction)), detail })
  }) as Reporter
  report.check = () => {
    if (signal?.aborted) throw new DOMException('cancelled', 'AbortError')
  }
  report.finish = () => {
    close()
    current = null
    return timings
  }
  return report
}

export function hashSeed(seed: number, ...parts: (string | number)[]): number {
  let h = 0x811c9dc5
  for (const ch of `${seed}:${parts.join(':')}`) h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193) >>> 0
  return h
}

async function stage<T>(report: Reporter, name: StageName, detail: string | undefined, work: () => T | Promise<T>): Promise<T> {
  report(name, 0, detail)
  const out = await work()
  report(name, 1, detail)
  return out
}

export async function analyze(audio: StereoAudio, deps: Deps, report: Reporter, signal?: AbortSignal): Promise<Analysis> {
  if (audio.sampleRate !== DEMUCS_SR) throw new Error(`expected ${DEMUCS_SR} Hz audio, got ${audio.sampleRate}`)
  const { left, right } = audio
  report('stems', 0, 'Preparing the audio for analysis')
  const mono = new Float32Array(left.length)
  for (let i = 0; i < mono.length; i++) mono[i] = (left[i] + right[i]) / 2
  const mix16 = resample(mono, DEMUCS_SR, SR)
  const frames = frameTotal(mix16.length)
  const duration = mix16.length / SR
  const demucs = await deps.model('demucs', undefined, (f) => report('stems', 0.05 * f, 'Loading instrument separation'))
  const stems16 = await separate(demucs, [left, right], (f) => report('stems', 0.05 + 0.85 * f, 'Separating drums, bass, other and vocals'), signal)
  await deps.release?.('demucs')
  report('stems', 0.9, 'Reading each instrument’s rhythm and energy')
  const raw = stemFeatures(stems16, mix16)
  const stored = storedFeatures(raw)
  report('stems', 1)

  const beatModel = await deps.model('beatthis', undefined, (f) => report('beats', 0.1 * f, 'Loading beat detection'))
  report('beats', 0.1, 'Reading the rhythm')
  const spect = beatSpectrogram(mix16)
  const chunks = chunkStarts(spect.frames).length
  const provider = beatThisProvider(beatModel)
  let done = 0
  const tracked = await detectBeats(spect.data, spect.frames, async (chunk, n) => {
    const out = await provider(chunk, n)
    report('beats', 0.2 + (0.8 * ++done) / chunks)
    return out
  })
  const tempo = tracked.beats.length > 2 ? 60 / median(tracked.beats.slice(1).map((b, i) => b - tracked.beats[i])) : 120
  const beats = tracked.beats.map((b) => roundDecimals(b, 4))
  const downbeats = tracked.downbeats.map((b) => roundDecimals(b, 4))
  await deps.release?.('beatthis')
  if (beats.length < 4) throw new Error('No steady beat found in this track')
  report('beats', 1, `${Math.round(tempo)} BPM`)

  const crepe = await deps.model('crepe', undefined, (f) => report('vocals', 0.05 * f, 'Loading vocal analysis'))
  const vocal = stems16[3]
  const { probs, totalFrames } = await crepeProbs(crepe, vocal, (f) => report('vocals', 0.05 + 0.85 * f, 'Tracking vocal pitch'), signal)
  await deps.release?.('crepe')
  const track = decodePitch(probs, totalFrames, frames)
  const active = vocalActive(track.periodicity, raw.rms.subarray(3 * frames, 4 * frames), frames)
  const syllables = vocalSyllables(vocal, active)
  report('vocals', 1, `${syllables.length} syllables`)

  report('attacks', 0)
  const pcm = Float32Array.from(mix16, (v) => Math.max(-32768, Math.min(32767, Math.round(v * 32768))) / 32768)
  const attacks = findAttacks(pcm)
  report('attacks', 1, `${attacks.length} musical accents found`)

  return {
    duration,
    frames,
    tempo,
    beats,
    downbeats,
    attacks,
    syllables,
    melF32: raw.mel,
    arrays: {
      frames,
      mel: stored.mel,
      onset: stored.onset,
      rms: stored.rms,
      pitch: toFloat16(maskPitch(track.midi, active)),
      periodicity: toFloat16(track.periodicity),
    },
  }
}

export function selectCandidate(options: CandidateSummary[]): number {
  return selectFlowCandidate(options)
}

function dedupe(notes: Note[]): Note[] {
  const seen = new Set<string>()
  const out: Note[] = []
  for (const n of [...notes].sort((a, b) => a.time - b.time || a.hand - b.hand)) {
    const key = `${roundDecimals(n.time, 3)}|${n.x}|${n.y}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(n)
  }
  return out
}

function hex(c: { r: number; g: number; b: number }): string {
  return `#${[c.r, c.g, c.b].map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0')).join('')}`
}

function colour(value: string): { r: number; g: number; b: number; a: number } {
  const n = parseInt(value.replace('#', ''), 16)
  return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255, a: 1 }
}

function palette(style: Style): Palette {
  const c = style.colors
  return { left: hex(c.saberA), right: hex(c.saberB), lightA: hex(c.env0), lightB: hex(c.env1) }
}

function safeName(name: string): string {
  return name.replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').trim() || 'BeatFlow'
}

async function loadNotes(deps: Deps, onProgress: (fraction: number) => void): Promise<NoteSessions> {
  const group = noteModelGroup(deps.backend)
  const sessions = {} as NoteSessions
  for (const [i, name] of NOTE_GRAPHS.entries()) sessions[name] = await deps.model(group, `${name}.onnx`, (f) => onProgress((i + f) / NOTE_GRAPHS.length))
  return sessions
}

export async function compose(analysis: Analysis, audio: StereoAudio, file: SongFile, settings: Settings, deps: Deps, report: Reporter, signal?: AbortSignal): Promise<Omit<CoreResult, 'timings'>> {
  const { duration, frames, beats, downbeats, attacks, syllables, arrays } = analysis
  const difficulties = (['Expert', 'ExpertPlus'] as Difficulty[]).filter((d) => settings.difficulties.includes(d))
  if (!difficulties.length) throw new Error('Choose at least one difficulty')

  const grid = await stage(report, 'grid', undefined, () => songGrid(beats, attacks))
  const sectionList: Section[] = await stage(report, 'sections', undefined, () => findSections(analysis.melF32, frames, beats, downbeats))
  const info = { downbeats, duration, syllables, sections: sectionList }
  const tokens = await stage(report, 'tokens', undefined, () => songTokens(info, arrays, grid))
  const features = toFloat16(tokens.features)
  report('candidates', 0)
  const cands = findCandidates(tokens.grid, attacks, syllables, arrays.onset, arrays.mel, frames, duration, tokens.times)
  const bounds = musicBounds(arrays.rms, frames)
  const sustains: [number, number][] = []
  const segments = repeatSegments(features, tokens.count, FEATURES, tokens.times)
  report('candidates', 1, `${cands.count} possible note timings found`)

  const energy = new Float32Array(frames)
  for (let s = 0; s < 4; s++) for (let t = 0; t < frames; t++) energy[t] = Math.fround(energy[t] + arrays.rms[s * frames + t])
  const musicEnd = bounds.musicEnd ?? duration
  report('notes', 0, 'Loading flow guidance and strain checks')
  const strain = await deps.strainData()
  const critic = await deps.model('flow', 'critic.onnx')
  const facts = { spb: 60 / analysis.tempo, musicStart: bounds.musicStart, musicEnd, energy }

  const sessions = await loadNotes(deps, (f) => report('notes', 0.01 + 0.04 * f, 'Loading the flow model'))
  const song: Omit<NoteSong, 'wallsPlan'> = {
    features,
    sectionLabel: tokens.sectionLabel,
    candTimes: cands.times,
    candFeats: cands.features,
    candTokens: cands.tokens,
    duration,
    musicStart: bounds.musicStart,
    musicEnd: bounds.musicEnd,
    introStem: bounds.introStem,
    sustains,
    segments,
  }
  const count = Math.max(1, Math.round(settings.candidates))
  const jobs = difficulties.length * count
  const charts: Chart[] = []
  const summaries: ChartSummary[] = []
  let job = 0
  try {
  for (const d of difficulties) {
    const di = DIFF_INDEX[d]
    const options: Candidate[] = []
    const difficultyLabel = d === 'ExpertPlus' ? 'Expert+' : d
    const noteProgress = (completed: number) => 0.05 + (0.95 * completed) / jobs
    report('notes', noteProgress(job), `${difficultyLabel} · Preparing note patterns`)
    const encoding = await prepareNoteSong(sessions, { ...song, wallsPlan: [] }, { difficulty: d, seed: settings.seed })
    try {
      for (const [t, topP] of TOP_P.entries()) {
      const perTopP = Math.floor(count / TOP_P.length) + (t < count % TOP_P.length ? 1 : 0)
      for (let c = 0; c < perTopP; c++) {
        const styleBucket = 0
        const seed = settings.seed + 1000 * c + di + Math.round(topP * 100)
        const label = `${difficultyLabel} · Variation ${options.length + 1} of ${count}`
        const base = job
        const out = await generateNotes(sessions, { ...song, wallsPlan: [] }, {
          difficulty: d,
          encoding,
          seed,
          temperature: 1.0,
          topP,
          cond: [styleBucket, QUALITY],
          condScale: COND_SCALE,
          flowCosts: (state) => flowCosts(critic, state.swings, state.hand, state.time, d),
          signal,
          onProgress: (f) => report('notes', noteProgress(base + 0.9 * f), `${label} · Writing flowing notes`),
        })
        report('notes', noteProgress(base + 0.9), `${label} · Checking strain and pacing`)
        const notes = dedupe(out.notes.filter((n) => n.time <= duration - 0.05))
        const flow = flowMetrics(notes, d, strain, facts)
        options.push({ seed, styleBucket, topP, notes: notes.length, nps: notes.length / duration, ...flow, chart: { difficulty: d, notes, walls: [], arcs: [] } })
        job++
      }
    }
    } finally { disposeNoteEncoding(encoding) }
    report('notes', noteProgress(job - 0.05), `${difficultyLabel} · ${count === 1 ? 'Finalizing the variation' : `Choosing from ${count} variations`}`)
    const best = selectCandidate(options)
    const chosen = options[best].chart
    const end = duration - 0.05
    const chart: Chart = {
      difficulty: d,
      notes: dedupe(chosen.notes.filter((n) => n.time <= end)),
      walls: chosen.walls.filter((w) => w.time + w.duration <= end),
      arcs: [],
    }
    charts.push(chart)
    summaries.push({
      difficulty: d,
      notes: chart.notes.length,
      arcs: chart.arcs.length,
      walls: chart.walls.length,
      lightEvents: 0,
      nps: chart.notes.length / duration,
      picked: best,
      candidates: options.map(({ chart: _chart, ...summary }) => summary),
    })
  }
  } finally {
    await deps.release?.(noteModelGroup(deps.backend))
    await deps.release?.('flow')
  }
  report('notes', 1)

  const intensity: Intensity = settings.lighting === 'auto' ? 'normal' : settings.lighting
  const lightModel = await deps.model('lights', undefined, (f) => report('lights', 0.05 * f, 'Loading lighting'))
  const lights: { events: LightEvent[]; boosts: [number, boolean][] }[] = []
  for (const [i, chart] of charts.entries()) {
    const result = await sampleLights(
      lightModel,
      { features, times: tokens.times, notes: chart.notes, walls: chart.walls, difficulty: chart.difficulty, song: { sections: sectionList, energy } },
      hashSeed(settings.seed, chart.difficulty, 'lights') % 10000,
      intensity,
      (f) => report('lights', 0.05 + (0.95 * (i + f)) / charts.length, chart.difficulty === 'ExpertPlus' ? 'Expert+' : chart.difficulty),
    )
    lights.push(result)
    summaries[i].lightEvents = result.events.length
  }
  await deps.release?.('lights')
  report('lights', 1)

  report('style', 0)
  const tags = file.bytes ? fileTags(file.bytes) : { title: null, artist: null, cover: null }
  const fallback = nameTags(file.name)
  const title = settings.title.trim() || tags.title || fallback.title
  const artist = settings.artist.trim() || tags.artist || fallback.artist
  const cover = settings.cover !== 'auto' ? settings.cover : tags.cover ? new Blob([tags.cover.data.slice().buffer as ArrayBuffer], { type: tags.cover.mime }) : null
  const pixels = cover ? await deps.coverPixels(cover) : null
  const stemRms = [0, 1, 2, 3].map((s) => arrays.rms.subarray(s * frames, (s + 1) * frames))
  const auto = await styleFromPixels(pixels, { tempo: analysis.tempo, stemRms, vocalPeriodicity: arrays.periodicity }, title, await deps.environmentReference())
  const style: Style = { environment: settings.environment === 'auto' ? auto.environment : settings.environment, colors: { ...auto.colors } }
  if (settings.colors !== 'auto') {
    const p = settings.colors
    style.colors = { ...style.colors, saberA: colour(p.left), saberB: colour(p.right), env0: colour(p.lightA), env1: colour(p.lightB), env0boost: colour(p.lightA), env1boost: colour(p.lightB) }
  }
  report('style', 1, style.environment.replace(/Environment$/, ''))

  const detected: Detected = {
    title,
    artist,
    cover: cover !== null,
    bpm: analysis.tempo,
    duration,
    sections: sectionList.length,
    walls: { enabled: false, perMinute: 0 },
    lighting: intensity,
    environment: auto.environment,
    palette: palette({ environment: auto.environment, colors: auto.colors }),
    seed: settings.seed,
    charts: summaries,
  }
  report('package', 0, 'Saving notes and lighting')
  const zip = await buildPackage(
    {
      title,
      artist,
      version: VERSION,
      duration: duration + LEAD,
      grid: Array.from(tokens.grid),
      charts: charts.map((chart, i) => ({ chart, lights: lights[i].events, boosts: lights[i].boosts })),
      audio: [audio.left, audio.right],
      sampleRate: audio.sampleRate,
      cover: cover ?? undefined,
      style,
      report: { version: VERSION, settings: { ...settings, walls: 'off', arcs: false, cover: settings.cover === 'auto' ? 'auto' : 'custom' }, detected, style: { category: auto.category, tags: auto.tags, top3: auto.top3, source: auto.source } },
    },
    (f) => report('package', f, f < 0.2 ? 'Saving notes and lighting' : f < 0.9 ? 'Preparing the song audio' : 'Finishing the map download'),
  )
  report('package', 1)
  return { charts, zip, fileName: `${safeName(brandedSongTitle(title))}.zip`, duration, detected }
}

export async function runPipeline(audio: StereoAudio, file: SongFile, settings: Settings, deps: Deps, onProgress: (p: Progress) => void, signal?: AbortSignal): Promise<CoreResult> {
  const report = reporter(onProgress, signal)
  const analysis = await analyze(audio, deps, report, signal)
  const result = await compose(analysis, audio, file, settings, deps, report, signal)
  return { ...result, timings: report.finish() }
}
