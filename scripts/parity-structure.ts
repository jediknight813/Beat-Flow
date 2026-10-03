import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { chunkStarts, postprocessBeats, stitchChunks } from '../src/engine/beats'
import { decodePitch, PITCH_BINS } from '../src/engine/pitch'
import { sectionIndex, sections, symmetricEigen, type Section } from '../src/engine/sections'
import { musicBounds, repeatSegments, sustains } from '../src/engine/structure'
import { syllables, type Syllable } from '../src/engine/vocals'

const args = process.argv.slice(2)
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : fallback
}
const parityDir = flag('--parity', '/mnt/storage/BeatFlow/parity')
const songs = flag('--songs', 'bereal,cry-for-me,foggy-windows,lights-out,moody,my-dad-is-rich,pocket-locket,poseidon,prompting,tongue,tweety').split(',')

type Row = { song: string; check: string; value: string; pass: boolean }
const rows: Row[] = []
const report = (song: string, check: string, value: string, pass: boolean) => rows.push({ song, check, value, pass })

function raw(path: string): ArrayBuffer {
  const buf = readFileSync(path)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
}

function f32(path: string): Float32Array {
  return new Float32Array(raw(path))
}

function i32(path: string): Int32Array {
  return new Int32Array(raw(path))
}

function f64(path: string): Float64Array {
  return new Float64Array(raw(path))
}

function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const m = h & 0x3ff
  if (e === 0) return s * m * 2 ** -24
  if (e === 31) return m ? NaN : s * Infinity
  return s * (1 + m / 1024) * 2 ** (e - 15)
}

function f16(path: string): Float32Array {
  const u = new Uint16Array(raw(path))
  const out = new Float32Array(u.length)
  for (let i = 0; i < u.length; i++) out[i] = half(u[i])
  return out
}

function json<T>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf8')) as T
}

function matchShare(a: number[], b: number[], tol: number): number {
  if (!b.length) return a.length ? 0 : 1
  let hits = 0
  let j = 0
  for (const x of b) {
    while (j < a.length && a[j] < x - tol) j++
    if (j < a.length && Math.abs(a[j] - x) <= tol + 1e-9) {
      hits++
      j++
    }
  }
  return hits / Math.max(a.length, b.length)
}

function frameLabels(list: Section[], frames: number): Int32Array {
  const out = new Int32Array(frames)
  for (let t = 0; t < frames; t++) {
    let idx = 0
    while (idx + 1 < list.length && list[idx + 1].start <= t / 50) idx++
    out[t] = idx
  }
  return out
}

function ari(a: Int32Array, b: Int32Array): number {
  const n = a.length
  const table = new Map<string, number>()
  const ra = new Map<number, number>()
  const rb = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const key = `${a[i]},${b[i]}`
    table.set(key, (table.get(key) ?? 0) + 1)
    ra.set(a[i], (ra.get(a[i]) ?? 0) + 1)
    rb.set(b[i], (rb.get(b[i]) ?? 0) + 1)
  }
  const c2 = (x: number) => (x * (x - 1)) / 2
  let index = 0
  for (const v of table.values()) index += c2(v)
  let sa = 0
  let sb = 0
  for (const v of ra.values()) sa += c2(v)
  for (const v of rb.values()) sb += c2(v)
  const expected = (sa * sb) / c2(n)
  const max = (sa + sb) / 2
  return max === expected ? 1 : (index - expected) / (max - expected)
}

function labelSeq(list: Section[], frames: number): Int32Array {
  const idx = frameLabels(list, frames)
  const out = new Int32Array(frames)
  for (let t = 0; t < frames; t++) out[t] = list[idx[t]].label
  return out
}

const pct = (x: number) => `${(100 * x).toFixed(2)}%`
let eigenMax = { song: '', size: 0, ms: 0, sectionsMs: 0 }

for (const song of songs) {
  const dir = join(parityDir, song)
  const info = json<{ frames: number; beats: number[]; downbeats: number[]; vocal_syllables: Syllable[]; sections: Section[] }>(join(dir, 'analysis.json'))
  const grid = json<{ sustains: [number, number][]; segments: { start: number; end: number; lag: number; lag_tokens: number }[]; music_start: number; music_end: number | null; intro_stem: number }>(join(dir, 'grid.json'))
  const T = info.frames

  if (existsSync(join(dir, 'beats/meta.json'))) {
    const meta = json<{ frames: number; starts: number[]; chunks_shape: number[]; beats: number[]; downbeats: number[] }>(join(dir, 'beats/meta.json'))
    const starts = chunkStarts(meta.frames)
    report(song, 'beat chunk starts', starts.join(','), starts.join(',') === meta.starts.join(','))
    const chunkData = f32(join(dir, 'beats/chunks.f32.bin'))
    const [nc, , len] = meta.chunks_shape
    const chunks = Array.from({ length: nc }, (_, c) => [chunkData.subarray((2 * c) * len, (2 * c + 1) * len), chunkData.subarray((2 * c + 1) * len, (2 * c + 2) * len)])
    const stitched = stitchChunks(chunks, starts, meta.frames)
    const logits = f32(join(dir, 'beats/logits.f32.bin'))
    let diff = 0
    for (let t = 0; t < meta.frames; t++) diff = Math.max(diff, Math.abs(stitched.beat[t] - logits[t]), Math.abs(stitched.downbeat[t] - logits[meta.frames + t]))
    report(song, 'beat stitch max|d|', diff.toExponential(2), diff === 0)
    const out = postprocessBeats(stitched)
    const bShare = matchShare(out.beats, meta.beats, 0.01)
    const dShare = matchShare(out.downbeats, meta.downbeats, 0.01)
    report(song, 'beats vs dump 10ms', `${pct(bShare)} (${out.beats.length}/${meta.beats.length})`, bShare >= 0.98)
    report(song, 'downbeats vs dump 10ms', `${pct(dShare)} (${out.downbeats.length}/${meta.downbeats.length})`, dShare >= 0.98)
    const aShare = matchShare(out.beats, info.beats, 0.01)
    report(song, 'beats vs analysis 10ms', `${pct(aShare)} (${out.beats.length}/${info.beats.length})`, aShare >= 0.98)
  }

  if (existsSync(join(dir, 'crepe/meta.json'))) {
    const meta = json<{ total_frames: number; frames: number }>(join(dir, 'crepe/meta.json'))
    const probs = f32(join(dir, 'crepe/probs.f32.bin'))
    if (probs.length !== meta.total_frames * PITCH_BINS) throw new Error('crepe probs shape')
    const t0 = performance.now()
    const track = decodePitch(probs, meta.total_frames, meta.total_frames)
    const ms = performance.now() - t0
    const ref = i32(join(dir, 'crepe/bins.i32.bin'))
    let wrong = 0
    for (let t = 0; t < ref.length; t++) if (track.bins[t] !== ref[t]) wrong++
    report(song, 'crepe bins exact', `${wrong} / ${ref.length} differ (${ms.toFixed(0)} ms)`, wrong === 0)
    const per = f32(join(dir, 'crepe/periodicity.f32.bin'))
    const midi = f32(join(dir, 'crepe/midi.f32.bin'))
    let dp = 0
    let dm = 0
    for (let t = 0; t < per.length; t++) {
      dp = Math.max(dp, Math.abs(per[t] - track.periodicity[t]))
      dm = Math.max(dm, Math.abs(midi[t] - track.midi[t]))
    }
    report(song, 'crepe periodicity/midi max|d|', `${dp.toExponential(2)} / ${dm.toExponential(2)}`, dp <= 1e-6 && dm <= 1e-4)
  }

  const rms = f16(join(dir, 'stem_rms.f16.bin'))
  const onset = f16(join(dir, 'stem_onset.f16.bin'))
  const periodicity = f16(join(dir, 'vocal_periodicity.f16.bin'))
  const active = new Uint8Array(T)
  for (let t = 0; t < T; t++) active[t] = periodicity[t] > 0.5 && rms[3 * T + t] > 0.02 ? 1 : 0
  const envelope = f32(join(dir, 'vocal_onset.f32.bin'))
  const syl = syllables(envelope, active)
  const vref = json<{ syllables: Syllable[] }>(join(dir, 'vocals.json')).syllables
  let exact = syl.length === vref.length
  for (let i = 0; exact && i < syl.length; i++) exact = syl[i].time === vref[i].time && Math.abs(syl[i].strength - vref[i].strength) < 1.5e-3
  report(song, 'syllables vs dump exact', `${syl.length}/${vref.length}`, exact)
  const sShare = matchShare(syl.map((s) => s.time), info.vocal_syllables.map((s) => s.time), 0.02)
  report(song, 'syllables vs analysis 20ms', `${pct(sShare)} (${syl.length}/${info.vocal_syllables.length})`, sShare >= 0.95)

  const mel = f16(join(dir, 'stem_mel.f16.bin'))
  const ts = performance.now()
  const secs = sections(mel, T, info.beats, info.downbeats)
  const secMs = performance.now() - ts
  const f16Ref = json<{ sections: Section[] }>(join(dir, 'sections.json')).sections
  const same = secs.length === f16Ref.length && secs.every((x, i) => x.start === f16Ref[i].start && x.end === f16Ref[i].end && x.label === f16Ref[i].label)
  report(song, 'sections vs f16 dump exact', `${secs.length}/${f16Ref.length} (${secMs.toFixed(0)} ms)`, same)
  const boundaryAri = ari(frameLabels(secs, T), frameLabels(info.sections, T))
  const labelAri = ari(labelSeq(secs, T), labelSeq(info.sections, T))
  const floor = ari(labelSeq(f16Ref, T), labelSeq(info.sections, T))
  report(song, 'sections vs analysis label ARI', `${labelAri.toFixed(3)} (python f16 ${floor.toFixed(3)}, segment ARI ${boundaryAri.toFixed(3)}, ${secs.length}/${info.sections.length})`, labelAri >= 0.8 || labelAri >= floor - 1e-9)
  const tokTimes = f64(join(dir, 'token_times.f64.bin'))
  const idx = sectionIndex(info.sections, tokTimes)
  const refIdx = i32(join(dir, 'section_index.i32.bin'))
  const badIdx = refIdx.reduce((n, v, i) => n + (v !== idx[i] ? 1 : 0), 0)
  report(song, 'section_index exact', `${badIdx} / ${refIdx.length} differ`, badIdx === 0)
  const S = new Set([0, ...info.beats.map((b) => Math.min(T - 1, Math.max(0, Math.round(b * 50)))), T]).size - 1
  if (S > eigenMax.size) {
    const a = new Float64Array(S * S)
    for (let i = 0; i < S; i++) for (let j = 0; j <= i; j++) a[i * S + j] = a[j * S + i] = Math.sin(i * 7 + j * 13)
    const te = performance.now()
    symmetricEigen(a, S)
    eigenMax = { song, size: S, ms: performance.now() - te, sectionsMs: secMs }
  }

  const bounds = musicBounds(rms, T)
  report(song, 'music bounds', `${bounds.musicStart}/${grid.music_start} ${bounds.musicEnd}/${grid.music_end} ${bounds.introStem}/${grid.intro_stem}`,
    bounds.musicStart === grid.music_start && bounds.musicEnd === grid.music_end && bounds.introStem === grid.intro_stem)
  const sus = sustains(periodicity, rms, onset, info.vocal_syllables.map((s) => s.time), T)
  let susOk = Math.abs(sus.length - grid.sustains.length) <= 1
  if (sus.length === grid.sustains.length) for (let i = 0; i < sus.length; i++) susOk &&= Math.abs(sus[i][0] - grid.sustains[i][0]) <= 0.02 && Math.abs(sus[i][1] - grid.sustains[i][1]) <= 0.02
  report(song, 'sustains', `${sus.length}/${grid.sustains.length}`, susOk)

  const cached = join(dir, 'tokens_cached.f16.bin')
  const tokens = f16(existsSync(cached) ? cached : join(dir, 'tokens.f16.bin'))
  const times = f64(join(dir, 'token_times.f64.bin'))
  const segs = repeatSegments(tokens, times.length, 369, times)
  const tokenDur = (times[times.length - 1] - times[0]) / (times.length - 1)
  let segOk = Math.abs(segs.length - grid.segments.length) <= 1
  for (const g of grid.segments) {
    const m = segs.find((s) => Math.abs(s.start - g.start) <= 2 * tokenDur && Math.abs(s.end - g.end) <= 2 * tokenDur)
    if (m && (m.lagTokens !== g.lag_tokens || Math.abs(m.lag - g.lag) > 2 * tokenDur)) segOk = false
    if (!m && segs.length === grid.segments.length) segOk = false
  }
  report(song, 'segments', `${segs.length}/${grid.segments.length} lags ${segs.map((s) => s.lagTokens).join(',')} | ${grid.segments.map((s) => s.lag_tokens).join(',')}`, segOk)
}

const w = Math.max(...rows.map((r) => r.check.length))
let failed = 0
for (const r of rows) {
  if (!r.pass) failed++
  console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.song.padEnd(15)} ${r.check.padEnd(w)}  ${r.value}`)
}
console.log(`eigensolver: ${eigenMax.song} n=${eigenMax.size} ${eigenMax.ms.toFixed(0)} ms (sections total ${eigenMax.sectionsMs.toFixed(0)} ms)`)
console.log(`${rows.length - failed}/${rows.length} checks pass`)
process.exit(failed ? 1 : 0)
