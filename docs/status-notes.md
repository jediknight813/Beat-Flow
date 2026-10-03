# Notes decoder port: status (parity verified, 2026-10-02)

`src/engine/notes.ts` ports `V7SongState` (v10 knobs: flow critic, density push, style controller, exit_w, arc_windows and chains are dropped because they are off or empty) as `NoteState`. It also has the seven-graph ONNX driver `NoteDecoder`, `generateNotes(sessions, song, options)` and `loadNoteSessions(backend, group)`. Its default group is `notes-fp16` on webgpu and `notes-int8` on wasm.

## Parity (teacher-forced replay of the Python trace, 200 steps per song, onnxruntime-web wasm under Node)

Run: `npx tsx scripts/parity-notes.ts [--suffix .fp16 | --models public/models/notes-int8] [--tol X]`

| set | song | max logit diff (worst head) | h diff | ms/step (step graph / full loop) |
|---|---|---|---|---|
| fp32 (`runs/export/notes`) | bereal Expert s0 | 6.10e-5 (chain) | 1.32e-5 | 15.3 / 16.6 |
| fp32 | tweety ExpertPlus s3 | 3.15e-5 (pointer) | 9.00e-6 | 24.6 / 25.9 |
| fp16 (`--suffix .fp16` and `public/models/notes-fp16`, same graphs) | bereal | 9.16e-2 (chain) | 2.23e-2 | 18.0 / 19.4 |
| fp16 | tweety | 8.27e-2 (pointer) | 1.86e-2 | 24.7 / 25.9 |
| int8 chunked (`public/models/notes-int8`) | bereal | 3.91e-1 (chain) | 8.77e-2 | 13.3 / 14.5 |
| int8 chunked | tweety | 4.02e-1 (chain) | 8.26e-2 | 20.3 / 21.4 |

On all three sets, for both songs, these all match the Python trace: window, cell, parity and forced masks (200/200 each); every sampling mask, top_p value and adjusted-logit vector (1664 calls for bereal, 1865 for tweety); repeat-planner rng draws; forced gestures; placed swings; swing notes; and the final notes and arcs (bereal 202 notes, 1 arc, 32 copied; tweety 200 notes, 2 arcs, 12 copied). fp32 is within 1e-3. Choices are teacher-forced, so on fp16/int8 the sampled maps would still drift from Python. That is expected and not a port bug.

The first run passed with no changes needed. `npx tsc -b`, `npx oxlint src` and `npm run build` pass.

## Next
- `src/engine/pipeline.ts` still emits empty `notes`/`arcs`. Wire `generateNotes` in there along with the song inputs (features, section labels, candidates, segments, sustains). Python builds these in `bsmapper/v7/generate.py` and the trace dumper saves them, so each input producer needs its own parity check against `parity/<song>/notes/*.bin`.
- Re-dump traces with `.venv/bin/python scripts/dump_notes_trace.py --song tweety --difficulty ExpertPlus --seed 3` in the trainer repo if the Python side changes.
