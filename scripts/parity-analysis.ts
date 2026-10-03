import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { resample, toFloat16, zscore } from '../src/engine/dsp'
import { attackEnvelope, attacks, beatSpectrogram, crepeFrames, fit, frameRms, logMel, onsetEnvelope, MELS } from '../src/engine/features'
import { songGrid, extendGrid } from '../src/engine/grid'
import { songTokens, FEATURES } from '../src/engine/tokens'
import { candidates, CAND_FEATURES } from '../src/engine/candidates'
import { musicBounds } from '../src/engine/structure'
import { syllables, type Syllable } from '../src/engine/vocals'
import type { Section } from '../src/engine/sections'

const ROOT = resolve(import.meta.dirname, '..')
const PARITY = resolve(ROOT, 'parity')
const only = process.argv.slice(2).filter((a) => !a.startsWith('--'))

type Analysis = { duration: number; frames: number; beats: number[]; downbeats: number[]; vocal_syllables: Syllable[]; sections: Section[] }
type Check = { name: string; ok: boolean; detail: string }

function bytes(path: string): ArrayBuffer {
  const b = readFileSync(path)
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer
}
const f16 = (p: string) => Float32Array.from(new Float16Array(bytes(p)))
const f32 = (p: string) => new Float32Array(bytes(p))
const f64 = (p: string) => new Float64Array(bytes(p))
const i32 = (p: string) => new Int32Array(bytes(p))
const json = <T>(p: string): T => JSON.parse(readFileSync(p, 'utf8')) as T

function wav(path: string): Float32Array {
  const b = readFileSync(path)
  let o = 12
  while (b.toString('ascii', o, o + 4) !== 'data') o += 8 + b.readUInt32LE(o + 4)
  const n = b.readUInt32LE(o + 4) / 2
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = b.readInt16LE(o + 8 + 2 * i) / 32768
  return out
}

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>, pick?: (i: number) => boolean): number {
  let m = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (pick && !pick(i)) continue
    const e = Math.abs(a[i] - b[i])
    if (e > m || Number.isNaN(e)) m = Number.isNaN(e) ? Infinity : e
  }
  return m
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length)
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += a[i]
    mb += b[i]
  }
  ma /= n
  mb /= n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

function matched(ref: ArrayLike<number>, ours: ArrayLike<number>, tol: number): number {
  const s = Float64Array.from(ours).sort()
  let hits = 0
  let j = 0
  for (let i = 0; i < ref.length; i++) {
    while (j + 1 < s.length && s[j + 1] <= ref[i]) j++
    const d = Math.min(Math.abs(s[j] - ref[i]), j + 1 < s.length ? Math.abs(s[j + 1] - ref[i]) : Infinity)
    if (d <= tol + 1e-9) hits++
  }
  return ref.length ? hits / ref.length : 1
}

const fmt = (x: number) => (x === 0 ? '0' : Math.abs(x) >= 0.01 ? x.toFixed(4) : x.toExponential(1))

function dspChecks(dir: string, pcm: Float32Array, frames: number, checks: Check[]): void {
  const meta = json<{ shapes: Record<string, number[]>; attack_count: number }>(resolve(dir, 'dsp.json'))
  const mel = logMel(pcm, frames)
  const m1 = maxAbs(mel, f32(resolve(dir, 'mix_mel.f32.bin')))
  checks.push({ name: 'dsp mix_mel', ok: m1 <= 2e-3, detail: fmt(m1) })
  const onset = zscore(fit(onsetEnvelope(pcm), frames))
  const m2 = maxAbs(onset, f32(resolve(dir, 'mix_onset.f32.bin')))
  checks.push({ name: 'dsp mix_onset', ok: m2 <= 5e-3, detail: fmt(m2) })
  const r = frameRms(pcm, frames)
  const peak = r.reduce((a, b) => Math.max(a, b), 0) + 1e-8
  const m3 = maxAbs(r.map((v) => v / peak), f32(resolve(dir, 'mix_rms.f32.bin')))
  checks.push({ name: 'dsp mix_rms', ok: m3 <= 1e-3, detail: fmt(m3) })
  const env = attackEnvelope(pcm)
  const m4 = maxAbs(env, f32(resolve(dir, 'attack_env.f32.bin')))
  checks.push({ name: 'dsp attack_env', ok: m4 <= 5e-3 && attacks(pcm).length === meta.attack_count, detail: fmt(m4) })
  const beat = beatSpectrogram(pcm)
  const m5 = maxAbs(beat.data, f32(resolve(dir, 'beat_mel.f32.bin')))
  checks.push({ name: 'dsp beat_mel', ok: beat.frames === meta.shapes.beat_mel[0] && m5 <= 0.05, detail: fmt(m5) })
  const m6 = maxAbs(crepeFrames(pcm, 0, 8), f32(resolve(dir, 'crepe_frames.f32.bin')))
  checks.push({ name: 'dsp crepe_frames', ok: m6 <= 1e-4, detail: fmt(m6) })
  const m7 = maxAbs(resample(f32(resolve(dir, 'mix44_mono.f32.bin')), 44100, 16000), f32(resolve(dir, 'mix16_soxr.f32.bin')))
  checks.push({ name: 'dsp soxr 44.1k->16k', ok: m7 <= 1e-3, detail: fmt(m7) })
}

function leadOffset(ext: Float64Array, grid: number[]): number {
  let best = 0
  let bestErr = Infinity
  for (let k = 0; k + grid.length <= ext.length; k++) {
    const e = maxAbs(ext.subarray(k, k + grid.length), grid)
    if (e < bestErr) {
      bestErr = e
      best = k
    }
  }
  return best
}

function endToEnd(features: Float32Array, tokRef: Float32Array, cand: ReturnType<typeof candidates>, ctRef: Float32Array, cfRef: Float32Array, ckRef: Int32Array): string {
  const feats = toFloat16(features)
  let tokBad = 0
  if (feats.length === tokRef.length) for (let i = 0; i < feats.length; i++) if (Math.abs(feats[i] - tokRef[i]) > 1e-3) tokBad++
  const tokText = feats.length === tokRef.length ? `${tokBad} token values > 1e-3` : `token count ${feats.length / FEATURES}`
  if (cand.count !== ctRef.length) return `${tokText}, candidate count ${cand.count}/${ctRef.length}`
  const cf = toFloat16(cand.features)
  let rows = 0
  for (let c = 0; c < cand.count; c++) {
    let bad = Math.abs(cand.times[c] - ctRef[c]) > 1e-4 || cand.tokens[c] !== ckRef[c]
    for (let k = 0; k < CAND_FEATURES; k++) if (Math.abs(cf[c * CAND_FEATURES + k] - cfRef[c * CAND_FEATURES + k]) > (k === 16 ? 2e-2 : 1e-3)) bad = true
    if (bad) rows++
  }
  return `${tokText}, ${rows}/${cand.count} candidates differ`
}

function song(id: string): { checks: Check[]; ms: number } {
  const dir = resolve(PARITY, id)
  const checks: Check[] = []
  const info = json<Analysis>(resolve(dir, 'analysis.json'))
  const ref = json<{ grid: number[]; music_start: number; music_end: number | null; intro_stem: number }>(resolve(dir, 'grid.json'))
  const attackRef = json<{ attacks: number[] }>(resolve(dir, 'attacks.json')).attacks
  const frames = info.frames
  const pcm = wav(resolve(dir, 'pcm16.wav'))
  const ours = attacks(pcm)
  const hit = matched(attackRef, ours, 0.004)
  checks.push({ name: 'attacks', ok: hit >= 0.98, detail: `${(hit * 100).toFixed(1)}% (${ours.length}/${attackRef.length})` })

  const vocal = wav(resolve(dir, 'vocal_pcm.wav'))
  const env = onsetEnvelope(vocal)
  const envRef = f32(resolve(dir, 'vocal_onset.f32.bin'))
  const envErr = maxAbs(env, envRef) / envRef.reduce((a, b) => Math.max(a, b), 0)
  checks.push({ name: 'vocal onset', ok: env.length === envRef.length && envErr <= 1e-4, detail: `rel ${fmt(envErr)}` })
  const mel = f16(resolve(dir, 'stem_mel.f16.bin'))
  const onset = f16(resolve(dir, 'stem_onset.f16.bin'))
  const rms = f16(resolve(dir, 'stem_rms.f16.bin'))
  const pitch = f16(resolve(dir, 'vocal_pitch.f16.bin'))
  const periodicity = f16(resolve(dir, 'vocal_periodicity.f16.bin'))
  const active = new Uint8Array(frames)
  for (let t = 0; t < frames; t++) active[t] = periodicity[t] > 0.5 && rms[3 * frames + t] > 0.02 ? 1 : 0
  const syl = syllables(env, active)
  const sylRef = json<{ syllables: Syllable[] }>(resolve(dir, 'vocals.json')).syllables
  const sylHit = matched(sylRef.map((s) => s.time), syl.map((s) => s.time), 0.02)
  let strength = 0
  if (syl.length === sylRef.length) syl.forEach((s, i) => (strength = Math.max(strength, Math.abs(s.strength - sylRef[i].strength))))
  else strength = Infinity
  checks.push({ name: 'syllables', ok: sylHit >= 0.95 && strength <= 0.1, detail: `${(sylHit * 100).toFixed(1)}% (${syl.length}/${sylRef.length}) str ${fmt(strength)}` })
  const vmel = toFloat16(logMel(vocal, frames))
  const cm = correlation(vmel, mel.subarray(3 * MELS * frames, 4 * MELS * frames))
  const von = toFloat16(zscore(fit(env, frames)))
  const co = correlation(von, onset.subarray(3 * frames))
  checks.push({ name: 'vocal stem mel/onset', ok: cm >= 0.99 && co >= 0.95, detail: `r ${cm.toFixed(4)} / ${co.toFixed(4)}` })

  const t0 = performance.now()
  const grid = songGrid(info.beats, attackRef)
  const gErr = grid.length === ref.grid.length ? maxAbs(grid, ref.grid) : Infinity
  checks.push({ name: 'grid', ok: gErr <= 2e-3, detail: `${fmt(gErr)} (${grid.length}/${ref.grid.length})` })
  const extRef = f64(resolve(dir, 'grid_extended.f64.bin'))
  const ext = extendGrid(grid, info.duration).grid
  const eErr = ext.length === extRef.length ? maxAbs(ext, extRef) : Infinity
  checks.push({ name: 'grid_extended', ok: eErr <= 2e-3, detail: `${fmt(eErr)} (${ext.length}/${extRef.length})` })

  const songInfo = { downbeats: info.downbeats, duration: info.duration, syllables: info.vocal_syllables, sections: info.sections }
  const arrays = { frames, mel, onset, rms, pitch, periodicity }
  const e2e = songTokens(songInfo, arrays, grid)
  const e2eCand = candidates(e2e.grid, attackRef, info.vocal_syllables, onset, mel, frames, info.duration, e2e.times)
  const ms = performance.now() - t0
  const lead = leadOffset(extRef, ref.grid)
  const refBeats = Array.from(extRef.subarray(lead, lead + ref.grid.length))
  const tok = songTokens(songInfo, arrays, refBeats)
  const timesRef = f64(resolve(dir, 'token_times.f64.bin'))
  const cand = candidates(extRef, attackRef, info.vocal_syllables, onset, mel, frames, info.duration, timesRef)
  const tokRef = f16(resolve(dir, 'tokens.f16.bin'))
  const ctRef = f32(resolve(dir, 'cand_times.f32.bin'))
  const cfRef = f16(resolve(dir, 'cand_feats.f16.bin'))
  const ckRef = i32(resolve(dir, 'cand_tokens.i32.bin'))
  const T = tokRef.length / FEATURES
  if (tok.count !== T) checks.push({ name: 'tokens', ok: false, detail: `count ${tok.count}/${T}` })
  else {
    const feats = toFloat16(tok.features)
    const col = (i: number) => i % FEATURES
    const cont = maxAbs(feats, tokRef, (i) => col(i) < 352 || col(i) >= 364)
    const vocalCols = maxAbs(feats, tokRef, (i) => col(i) === 352 || col(i) === 353)
    let exact = 0
    for (let i = 0; i < feats.length; i++) if (col(i) >= 354 && col(i) < 364 && feats[i] !== tokRef[i]) exact++
    checks.push({ name: 'tokens', ok: cont <= 1e-3 && vocalCols <= 1e-3 && exact === 0, detail: `${fmt(cont)} / vocal ${fmt(vocalCols)} / exact-col mismatches ${exact}` })
  }
  const tErr = tok.times.length === timesRef.length ? maxAbs(tok.times, timesRef) : Infinity
  checks.push({ name: 'token_times', ok: tErr <= 2e-3, detail: fmt(tErr) })
  const siRef = i32(resolve(dir, 'section_index.i32.bin'))
  const slRef = i32(resolve(dir, 'section_label.i32.bin'))
  const siBad = tok.sectionIndex.reduce((n, v, i) => n + (v !== siRef[i] ? 1 : 0), 0)
  const slBad = tok.sectionLabel.reduce((n, v, i) => n + (v !== slRef[i] ? 1 : 0), 0)
  checks.push({ name: 'section index/label', ok: siBad === 0 && slBad === 0, detail: `${siBad}/${slBad} mismatches` })
  if (cand.count !== ctRef.length) checks.push({ name: 'candidates', ok: false, detail: `count ${cand.count}/${ctRef.length}` })
  else {
    const ct = maxAbs(Float32Array.from(cand.times), ctRef)
    const cf = toFloat16(cand.features)
    const main = maxAbs(cf, cfRef, (i) => i % CAND_FEATURES !== 16)
    const kick = maxAbs(cf, cfRef, (i) => i % CAND_FEATURES === 16)
    const kb = cand.tokens.reduce((n, v, i) => n + (v !== ckRef[i] ? 1 : 0), 0)
    checks.push({ name: 'cand_times', ok: ct <= 1e-4, detail: `${fmt(ct)} (${cand.count})` })
    checks.push({ name: 'cand_feats', ok: main <= 1e-3 && kick <= 2e-2, detail: `${fmt(main)} / kick ${fmt(kick)}` })
    checks.push({ name: 'cand_tokens', ok: kb === 0, detail: `${kb} mismatches` })
  }
  checks.push({ name: 'end-to-end (info)', ok: true, detail: endToEnd(e2e.features, tokRef, e2eCand, ctRef, cfRef, ckRef) })
  const mb = musicBounds(rms, frames)
  const endOk = mb.musicEnd === null ? ref.music_end === null : ref.music_end !== null && Math.abs(mb.musicEnd - ref.music_end) <= 0.02 + 1e-9
  const mOk = Math.abs(mb.musicStart - ref.music_start) <= 0.02 + 1e-9 && endOk && mb.introStem === ref.intro_stem
  checks.push({ name: 'music bounds', ok: mOk, detail: `${mb.musicStart}/${mb.musicEnd}/${mb.introStem} vs ${ref.music_start}/${ref.music_end}/${ref.intro_stem}` })

  const dsp = resolve(dir, 'dsp')
  if (existsSync(dsp)) dspChecks(dsp, pcm, frames, checks)
  return { checks, ms }
}

const index = json<{ songs: { id: string }[] }>(resolve(PARITY, 'index.json'))
let failures = 0
for (const { id } of index.songs) {
  if (only.length && !only.includes(id)) continue
  const { checks, ms } = song(id)
  const bad = checks.filter((c) => !c.ok)
  failures += bad.length
  console.log(`\n${id}  ${bad.length ? `FAIL ${bad.length}` : 'PASS'}  (grid+tokens+candidates ${ms.toFixed(0)} ms)`)
  for (const c of checks) console.log(`  ${c.ok ? 'ok  ' : 'FAIL'} ${c.name.padEnd(22)} ${c.detail}`)
}
console.log(failures ? `\n${failures} failing checks` : '\nall checks pass')
process.exitCode = failures ? 1 : 0
