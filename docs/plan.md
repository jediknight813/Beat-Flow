# BeatFlow build plan

Everything runs in the browser (ONNX Runtime Web on WebGPU, WASM fallback), hosted on GitHub Pages. No file over 95 MB (`scripts/chunk_models.py` in the trainer repo splits weights; `src/engine/models.ts` reassembles them).

## Stages

| stage | source of truth (trainer repo) | browser module | status |
|---|---|---|---|
| decode audio | `scripts/prepare_playtest_songs.py` | `engine/audio.ts` | done |
| stems | Demucs htdemucs (`analysis/song.py`) | `engine/stems.ts` + `models/stems/` | pending export |
| beats | Beat This! + `analysis/grid.py` | `engine/beats.ts` + `models/beats/` | pending export |
| vocals | torchcrepe + syllables | `engine/vocals.ts` + `models/vocals/` | pending export |
| attacks, sections, tokens, candidates | `compute_attacks.py`, `v6/tokens.py`, `pack_v7.py` | `engine/tokens.ts` | pending spec |
| notes | `v10/notes.py`, `v7/generate.py` | `engine/notes.ts` + `models/notes/` | pending export |
| walls | `v91/walls.py`, `v8/walls.py` | `engine/walls.ts`, `engine/wallRules.ts`, `data/walls.json` | done |
| lights | `v9/canvas.py`, `v9/shape.py` | `engine/lights.ts` + `models/lights/`, `engine/pcg64.ts` | done; input `LightsInput` = `features` (T×369 float32 song tokens), `times` (T song seconds), `notes`, `walls`, `difficulty`, `song: LightsSong` = `sections` (`{start, end}` song seconds) + `energy` (`stem_rms` summed over the 4 stems, 50 fps, float32); times out in song seconds (`toV3` adds the lead); checked by `scripts/parity-lights.ts` |
| style | `v7/style.py` | `engine/style.ts`, `data/environment-reference.json` | done; input `StyleAudio` = `tempo` (bpm), `stemRms` (4 × frames at 50 fps: drums, bass, other, vocals), `vocalPeriodicity` (frames at 50 fps); cover as `ImageBitmap` (64×64 and 48×48 downscales via `OffscreenCanvas`); checked by `scripts/parity-lights.ts` |
| package | `v6/export.py` | `engine/package.ts` | done |

## Parity

`parity/<song>/` holds the Python intermediates for the 11 playtest songs. Each browser stage is checked against them before the next stage is built; tolerances are in `parity/index.json`.

## Order

1. notes model running on Python-dumped tokens and candidates (proves ONNX decoding).
2. analysis stages one by one, each against parity data.
3. walls, lights, style, package wired into `engine/pipeline.ts`.
4. candidate selection (distilled judge).
