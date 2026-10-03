import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'
import { judgeFeatures, FEATURES } from '../src/engine/judge'
ort.env.logLevel = 'error'
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'))
let worst = 0
let where = ''
for (const r of rows) {
  const f = judgeFeatures(r.notes, { duration: r.facts.duration, beats: r.facts.beats, attacks: r.facts.attacks, musicStart: r.facts.music_start, musicEnd: r.facts.music_end }, r.diff)
  for (const k of FEATURES) {
    const d = Math.abs(f[k] - r.py[k])
    if (!(d <= worst)) { worst = d; where = `${k} ts=${f[k]} py=${r.py[k]}` }
  }
}
console.log(rows.length, 'charts', FEATURES.length, 'features; max abs diff', worst, where)
