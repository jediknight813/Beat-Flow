// Local-only stand-in for generate(): walks every pipeline stage on a timer and
// returns a small but well-formed map, so the whole UI can be exercised without
// models or audio. Loaded only when DEV_TOOLS is on (see ./flags).
import JSZip from 'jszip'
import type { Result } from '../engine/pipeline'
import type { Chart, Difficulty, Note, Progress, Settings, StageName } from '../engine/types'

const SONGS = [
  { title: 'Neon Horizon', artist: 'Midnight Arcade', bpm: 128, hues: [318, 196], colors: ['#f0359b', '#2cc3f2'] },
  { title: 'Glass Cathedral', artist: 'Lumen Drive', bpm: 140, hues: [262, 12], colors: ['#8a4dff', '#ff5a3d'] },
  { title: 'Pulse of the Static', artist: 'Kilowatt Bloom', bpm: 174, hues: [188, 344], colors: ['#21d4e8', '#ff2f6d'] },
  { title: 'Afterimage', artist: 'Velvet Circuit', bpm: 110, hues: [28, 286], colors: ['#ff8a2a', '#c04dff'] },
  { title: 'Satellite Hearts', artist: 'Nova Lane', bpm: 122, hues: [204, 330], colors: ['#2f9bff', '#ff3fa4'] },
]

type FakeStage = [StageName, number, (song: FakeSong) => string | undefined]

// Follow the real pipeline, including each difficulty's variations and checks.
function stages(settings: Settings): FakeStage[] {
  const count = Math.max(1, Math.round(settings.candidates))
  const notes: FakeStage[] = [['notes', 0.4, () => 'Loading flow guidance and strain checks']]
  for (const difficulty of settings.difficulties) {
    const label = difficulty === 'ExpertPlus' ? 'Expert+' : difficulty
    notes.push(['notes', 0.2, () => `${label} · Preparing note patterns`])
    for (let i = 1; i <= count; i++) {
      notes.push(['notes', 0.4, () => `${label} · Variation ${i} of ${count} · Writing flowing notes`])
      notes.push(['notes', 0.2, () => `${label} · Variation ${i} of ${count} · Checking strain and pacing`])
    }
    notes.push(['notes', 0.3, () => `${label} · ${count === 1 ? 'Finalizing the variation' : `Choosing from ${count} variations`}`])
  }
  return [
    ['decode', 0.4, (s) => `${s.duration.toFixed(1)} s at 44100 Hz`],
    ['stems', 2.2, () => 'drums · bass · vocals · other'],
    ['beats', 0.8, (s) => `${s.bpm} BPM`],
    ['vocals', 0.8, () => undefined],
    ['attacks', 0.5, () => undefined],
    ['grid', 0.3, () => undefined],
    ['sections', 0.4, () => '9 sections'],
    ['tokens', 0.4, () => undefined],
    ['candidates', 0.4, () => '1,184 possible note timings found'],
    ...notes,
    ['lights', 0.6, () => undefined],
    ['style', 0.3, () => undefined],
    ['package', 0.5, () => 'Saving notes, lighting, and audio'],
  ]
}
const STEP_MS = 120

type FakeSong = (typeof SONGS)[number] & { duration: number }

export function fakeSongFiles(count: number): File[] {
  const songs = [...SONGS].sort(() => Math.random() - 0.5).slice(0, count)
  return songs.map((song) => new File([new Uint8Array(1024)], `${song.artist} - ${song.title}.mp3`, { type: 'audio/mpeg' }))
}

function songFor(file: File): FakeSong {
  const song = SONGS.find((s) => file.name.includes(s.title)) ?? SONGS[0]
  return { ...song, duration: 180 + (song.bpm % 60) }
}

const wait = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  if (signal?.aborted) return reject(new DOMException('cancelled', 'AbortError'))
  const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve() }, ms)
  const onAbort = () => { clearTimeout(timer); reject(new DOMException('cancelled', 'AbortError')) }
  signal?.addEventListener('abort', onAbort, { once: true })
})

async function drawCover(song: FakeSong): Promise<Blob> {
  const size = 512
  const canvas = new OffscreenCanvas(size, size)
  const ctx = canvas.getContext('2d')!
  const [a, b] = song.hues
  const bg = ctx.createLinearGradient(0, 0, size, size)
  bg.addColorStop(0, `hsl(${a} 80% 18%)`)
  bg.addColorStop(1, `hsl(${b} 70% 8%)`)
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, size, size)
  const sun = ctx.createRadialGradient(size / 2, size * 0.42, 10, size / 2, size * 0.42, 190)
  sun.addColorStop(0, `hsl(${a} 95% 68%)`)
  sun.addColorStop(0.55, `hsl(${b} 90% 55% / 0.55)`)
  sun.addColorStop(1, 'transparent')
  ctx.fillStyle = sun
  ctx.fillRect(0, 0, size, size)
  ctx.strokeStyle = `hsl(${b} 90% 65% / 0.45)`
  ctx.lineWidth = 2
  for (let i = 1; i <= 8; i++) {
    const y = size * 0.62 + i * i * 2.6
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(size, y); ctx.stroke()
  }
  for (let i = -6; i <= 6; i++) {
    ctx.beginPath(); ctx.moveTo(size / 2 + i * 18, size * 0.62); ctx.lineTo(size / 2 + i * 120, size); ctx.stroke()
  }
  ctx.fillStyle = '#fff'
  ctx.font = '800 40px Inter, Arial, sans-serif'
  ctx.fillText(song.title.toUpperCase(), 32, 64, size - 64)
  ctx.fillStyle = 'rgba(255,255,255,0.7)'
  ctx.font = '400 22px Inter, Arial, sans-serif'
  ctx.fillText(song.artist, 32, 96, size - 64)
  return canvas.convertToBlob({ type: 'image/png' })
}

// Alternating down/up swings in the middle lanes: enough to read as a real chart.
function chart(difficulty: Difficulty, song: FakeSong): Chart {
  const step = 60 / song.bpm / (difficulty === 'ExpertPlus' ? 2 : 1.5)
  const notes: Note[] = []
  for (let time = 2, i = 0; time < song.duration - 4; time += step, i++) {
    const direction = i % 2 ? 0 : 1
    notes.push({ time, x: 1, y: 0, hand: 0, direction, angle: 0 }, { time, x: 2, y: 0, hand: 1, direction, angle: 0 })
  }
  return { difficulty, notes, walls: [], arcs: [] }
}

export async function fakeGenerate(file: File, settings: Settings, onProgress: (p: Progress) => void, signal?: AbortSignal): Promise<Result> {
  const song = songFor(file)
  const timings: Result['timings'] = {}
  const pipeline = stages(settings)
  const totals: Partial<Record<StageName, number>> = {}
  for (const [stage, seconds] of pipeline) totals[stage] = (totals[stage] ?? 0) + seconds
  const completed: Partial<Record<StageName, number>> = {}
  for (const [stage, seconds, detail] of pipeline) {
    const steps = Math.max(2, Math.round((seconds * 1000) / STEP_MS))
    for (let i = 0; i <= steps; i++) {
      onProgress({ stage, fraction: ((completed[stage] ?? 0) + seconds * i / steps) / totals[stage]!, detail: detail(song) })
      if (i < steps) await wait(STEP_MS, signal)
    }
    completed[stage] = (completed[stage] ?? 0) + seconds
    timings[stage] = completed[stage]! * 1000
  }

  const title = settings.title.trim() || song.title
  const artist = settings.artist.trim() || song.artist
  const charts = settings.difficulties.map((d) => chart(d, song))
  const palette = settings.colors === 'auto'
    ? { left: song.colors[0], right: song.colors[1], lightA: song.colors[0], lightB: song.colors[1] }
    : settings.colors
  const cover = settings.cover === 'auto' ? await drawCover(song) : settings.cover
  const coverName = cover.type === 'image/jpeg' ? 'cover.jpg' : 'cover.png'

  const zip = new JSZip()
  zip.file('Info.dat', JSON.stringify({
    _version: '2.1.0',
    _songName: `${title} [AI fake]`,
    _songSubName: 'Fake dev-tools map',
    _songAuthorName: artist,
    _levelAuthorName: 'BeatFlow dev tools',
    _beatsPerMinute: song.bpm,
    _coverImageFilename: coverName,
    _difficultyBeatmapSets: [],
  }, null, 2))
  zip.file(coverName, await cover.arrayBuffer())
  zip.file('generation.json', JSON.stringify({ fake: true, seed: settings.seed }, null, 2))

  return {
    charts,
    zip: await zip.generateAsync({ type: 'blob', compression: 'STORE' }),
    fileName: `${title} - BeatFlow.zip`,
    duration: song.duration,
    timings,
    detected: {
      title, artist, cover: settings.cover === 'auto', bpm: song.bpm, duration: song.duration, sections: 9,
      walls: { enabled: false, perMinute: 0 },
      lighting: settings.lighting === 'auto' ? 'normal' : settings.lighting,
      environment: settings.environment === 'auto' ? 'BillieEnvironment' : settings.environment,
      palette, seed: settings.seed,
      charts: charts.map((c) => ({
        difficulty: c.difficulty, notes: c.notes.length, arcs: 0, walls: 0, lightEvents: 0,
        nps: c.notes.length / song.duration, picked: 0, candidates: [],
      })),
    },
  }
}
