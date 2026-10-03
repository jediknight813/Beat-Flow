import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import * as ort from 'onnxruntime-web'
import { analyze, compose, reporter, type Analysis, type Deps, type StereoAudio } from '../src/engine/core'
import { attacks } from '../src/engine/features'
import type { CoverPixels, EnvironmentReference } from '../src/engine/style'
import type { WallData } from '../src/engine/walls'
import type { Progress, Settings } from '../src/engine/types'
import type { Syllable } from '../src/engine/vocals'

const ROOT = resolve(import.meta.dirname, '..')
const TRAINER = '/mnt/storage/BeatSaberModelTrainer'
const args = process.argv.slice(2)
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : fallback
}
const songId = flag('--song', 'bereal')
const fromParity = args.includes('--from-parity')
const out = flag('--out', '/tmp/claude-1000/-mnt-storage-BeatSaberModelTrainer/1adf4e49-e955-4180-a539-2ddfba56895a/scratchpad/bereal-beatflow.zip')
const modelsRoot = flag('--models', join(ROOT, 'public/models'))
const settings: Settings = {
  difficulties: ['Expert', 'ExpertPlus'],
  walls: flag('--walls', 'auto') as Settings['walls'],
  arcs: !args.includes('--no-arcs'),
  lighting: flag('--lighting', 'auto') as Settings['lighting'],
  environment: 'auto',
  colors: 'auto',
  title: '',
  artist: '',
  cover: 'auto',
  seed: Number(flag('--seed', '20261001')),
  candidates: Number(flag('--candidates', '4')),
}

ort.env.logLevel = 'error'
ort.env.wasm.numThreads = Number(process.argv.includes('--threads') ? process.argv[process.argv.indexOf('--threads') + 1] : '1')

const json = <T>(path: string): T => JSON.parse(readFileSync(path, 'utf8')) as T

function bytes(path: string): ArrayBuffer {
  const b = readFileSync(path)
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer
}

const f16 = (path: string) => Float32Array.from(new Float16Array(bytes(path)))

function wav16(path: string): Float32Array {
  const b = readFileSync(path)
  let o = 12
  while (b.toString('ascii', o, o + 4) !== 'data') o += 8 + b.readUInt32LE(o + 4)
  const n = b.readUInt32LE(o + 4) / 2
  const pcm = new Float32Array(n)
  for (let i = 0; i < n; i++) pcm[i] = b.readInt16LE(o + 8 + 2 * i) / 32768
  return pcm
}

function decode(path: string): StereoAudio {
  const raw = execFileSync('ffmpeg', ['-nostdin', '-v', 'error', '-i', path, '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100', '-f', 'f32le', 'pipe:1'], { maxBuffer: 1 << 30 })
  const all = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))
  const n = all.length / 2
  const left = new Float32Array(n)
  const right = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    left[i] = all[2 * i]
    right[i] = all[2 * i + 1]
  }
  return { left, right, sampleRate: 44100 }
}

function scaledRgb(path: string, size: number): Uint8Array {
  return new Uint8Array(execFileSync('ffmpeg', ['-nostdin', '-v', 'error', '-i', path, '-vf', `scale=${size}:${size}:flags=area`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']))
}

const parityDir = join(ROOT, 'parity', songId)
const parityCover = join(parityDir, 'lights/cover64.json')

const deps: Deps = {
  backend: 'wasm',
  async model(group, file) {
    const dir = join(modelsRoot, group)
    let name = file
    if (!name && existsSync(join(dir, 'spec.json'))) {
      const pick = json<Record<string, unknown>>(join(dir, 'spec.json')).wasm
      name = Array.isArray(pick) ? (pick[0] as string) : (pick as string | undefined)
    }
    const manifests = existsSync(join(dir, 'manifest.json')) ? json<Record<string, { files: string[] }>>(join(dir, 'manifest.json')) : {}
    const manifest = name ? manifests[name] : undefined
    if (!name || !manifest) throw new Error(`${group} model not available yet`)
    const data = new Uint8Array(Buffer.concat(manifest.files.map((f) => readFileSync(join(dir, f)))))
    return ort.InferenceSession.create(data, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
  },
  wallData: async () => json<WallData>(join(ROOT, 'public/data/walls.json')),
  environmentReference: async () => json<EnvironmentReference>(join(ROOT, 'public/data/environment-reference.json')),
  async coverPixels(cover): Promise<CoverPixels | null> {
    if (fromParity && existsSync(parityCover)) {
      const c = json<{ rgb64: number[]; rgb48: number[] }>(parityCover)
      return { rgb64: Uint8Array.from(c.rgb64), rgb48: Uint8Array.from(c.rgb48) }
    }
    const tmp = join(mkdtempSync(join(tmpdir(), 'beatflow-')), 'cover')
    writeFileSync(tmp, new Uint8Array(await cover.arrayBuffer()))
    return { rgb64: scaledRgb(tmp, 64), rgb48: scaledRgb(tmp, 48) }
  },
}

function parityAnalysis(report: ReturnType<typeof reporter>): Analysis {
  const info = json<{ duration: number; frames: number; tempo_bpm: number; beats: number[]; downbeats: number[]; vocal_syllables: Syllable[] }>(join(parityDir, 'analysis.json'))
  const mel = f16(join(parityDir, 'stem_mel.f16.bin'))
  report('attacks', 0)
  const found = attacks(wav16(join(parityDir, 'pcm16.wav')))
  report('attacks', 1)
  return {
    duration: info.duration,
    frames: info.frames,
    tempo: info.tempo_bpm,
    beats: info.beats,
    downbeats: info.downbeats,
    attacks: found,
    syllables: info.vocal_syllables,
    melF32: mel,
    arrays: {
      frames: info.frames,
      mel,
      onset: f16(join(parityDir, 'stem_onset.f16.bin')),
      rms: f16(join(parityDir, 'stem_rms.f16.bin')),
      pitch: f16(join(parityDir, 'vocal_pitch.f16.bin')),
      periodicity: f16(join(parityDir, 'vocal_periodicity.f16.bin')),
    },
  }
}

const song = json<{ songs: { id: string; audio: string }[] }>(join(TRAINER, 'configs/playtest_songs.json')).songs.find((s) => s.id === songId)
if (!song) throw new Error(`unknown song ${songId}`)
let last = ''
const onProgress = (p: Progress) => {
  const line = `${p.stage} ${Math.floor(p.fraction * 10) * 10}%${p.detail ? ` ${p.detail}` : ''}`
  if (line !== last) process.stderr.write(`\r${line.padEnd(80)}`)
  last = line
}
const t0 = performance.now()
const report = reporter(onProgress)
report('decode', 0)
const audio = decode(song.audio)
report('decode', 1)
const analysis = fromParity ? parityAnalysis(report) : await analyze(audio, deps, report)
const result = await compose(analysis, audio, { name: song.audio.split('/').pop()!, bytes: new Uint8Array(readFileSync(song.audio)) }, settings, deps, report)
const timings = report.finish()
process.stderr.write('\n')
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, new Uint8Array(await result.zip.arrayBuffer()))

const d = result.detected
console.log(`\n${d.title} by ${d.artist} (${fromParity ? 'parity analysis' : 'full analysis'}, wasm)`)
console.log(`  ${d.duration.toFixed(1)} s, ${d.bpm.toFixed(2)} BPM, ${d.sections} sections, walls ${d.walls.enabled ? `${d.walls.perMinute.toFixed(2)} phrases/min` : 'off'}, ${d.environment}, palette ${Object.values(d.palette).join(' ')}, cover ${d.cover}`)
console.log('  stage timings:')
for (const [stage, ms] of Object.entries(timings)) console.log(`    ${stage.padEnd(11)} ${(ms / 1000).toFixed(2)} s`)
console.log(`    total       ${((performance.now() - t0) / 1000).toFixed(2)} s`)
for (const c of d.charts) {
  console.log(`  ${c.difficulty}: ${c.notes} notes (${c.nps.toFixed(2)}/s), ${c.arcs} arcs, ${c.walls} walls, ${c.lightEvents} light events; picked #${c.picked}`)
  for (const [i, o] of c.candidates.entries()) console.log(`    #${i} seed ${o.seed} style ${o.styleBucket} top_p ${o.topP}: ${o.notes} notes, ${o.nps.toFixed(2)}/s`)
}
console.log(`  wrote ${out} (${(result.zip.size / 1e6).toFixed(1)} MB) as "${result.fileName}"`)
