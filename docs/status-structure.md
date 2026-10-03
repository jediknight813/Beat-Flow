# Structural stages: status (parity passing)

`npx tsx scripts/parity-structure.ts` (options `--songs a,b`, `--parity DIR`): **109/109 checks pass** on all 11 playtest songs. `npx tsc -b`, `npx oxlint src scripts`, `npm run build` pass.

| Stage | Check | Result |
|---|---|---|
| beats (`beats.ts`) | chunk starts, stitch of dumped chunk logits vs `beats/logits.f32.bin` | identical (max abs diff 0) on bereal, my-dad-is-rich, tweety |
| | postprocess vs `beats/meta.json` and `analysis.json`, 10 ms | 100 % beats and downbeats (289/289, 177/177, 251/251) |
| pitch (`pitch.ts`) | Viterbi bins vs `crepe/bins.i32.bin` | exact, 0 differing frames (6877, 5236, 11639 frames; 85-195 ms per song) |
| | periodicity vs `crepe/periodicity.f32.bin`, midi vs `crepe/midi.f32.bin` | max abs diff 0 / 7.6e-6 |
| vocals (`vocals.ts`) | syllables from `vocal_onset.f32.bin` vs `vocals.json` | exact (time and strength) on all 11 |
| | vs `analysis.json` syllables, 20 ms | 99.45-100 % |
| sections (`sections.ts`) | vs `sections.json` (Python `structure()` on the float16 `stem_mel` dump) | exact starts, ends and label ids on all 11 |
| | vs `analysis.json` (computed from float32 mel), frame-level label ARI | 1.000 on 7 songs; cry-for-me 0.933, my-dad-is-rich 0.900, poseidon 0.983, tongue 0.994, pocket-locket 0.622; each equals Python's own ARI on the float16 input, so the gap is float16 storage of the dumped mel, not the port |
| | `sectionIndex` vs `section_index.i32.bin` | exact |
| structure (`structure.ts`) | music_start / music_end / intro_stem vs `grid.json` | exact |
| | sustains vs `grid.json` (count +-1, 0.02 s) | identical counts and times on all 11 |
| | repeat segments vs `grid.json` (count +-1, start/end/lag within 2 tokens, lag_tokens exact) | identical counts and lag_tokens on all 11 (uses `tokens_cached.f16.bin` where present, since `load_song` read the token cache) |

Eigensolver (JAMA tred2 + tql2, `symmetricEigen`): 148-167 ms for n = 400 (cry-for-me, largest beat-sync matrix) in Node; whole `sections()` 45-350 ms per song.

Fixes made this round:
* `pitch.ts` `medianFilter3`: torchcrepe pads the values with reflect but the validity mask with zeros, so the first/last frame take the lower of their two real neighbours (`min(x0, x1)`, `min(x[n-2], x[n-1])`), not a reflected median. Fixed tweety's last-frame 0.135 periodicity error.
* `sections.ts` Laplacian: `scipy.sparse.csgraph.laplacian` normalises by **column** sums of `A` (axis 0), and `A` is not symmetric (`Rf` is median-filtered along rows only); `np.linalg.eigh` then reads only the lower triangle. The port now uses column-sum degrees and mirrors the lower triangle.
* `sections.ts` KMeans: exact sklearn 1.9.1 replication with a new `mt19937.ts` (`numpy.random.RandomState(0)`: init_genrand seeding, 53-bit `random_sample`): data centred first, k-means++ (`choice` via normalised cumsum + searchsorted right, `uniform * pot` trials, searchsorted left on the cumulative potential, `||x||^2 + ||c||^2 - 2 x.c` distances), Lloyd with sklearn's iteration order (assign + update, strict convergence on unchanged labels, else centre-shift tol, final re-assign), empty-cluster relocation without relabelling, best run replaced only if inertia is lower and the labelling is not a permutation of the best. Label ids now match Python, not just the partition.

Reference data: `parity/<song>/sections.json` added by `/mnt/storage/BeatSaberModelTrainer/scripts/dump_parity_beats.py` (`--sections-only` regenerates just those). Beat/CREPE model dumps exist only for bereal, my-dad-is-rich, tweety; the other songs test syllables, sections and structure only.
Note: recomputing CREPE from the int16 `vocal_pcm.wav` moves periodicity by 0.18-0.23 at p99 vs the cached float32-stem run (voiced agreement 0.99), so periodicity parity is checked against `crepe/` not `vocal_periodicity.f16.bin`.
Spec details confirmed by the exact matches: Viterbi per 256-frame batch; zero transitions cost log(float32 tiny); chroma from bands 100-5000 Hz of the Slaney 80-mel centres; kNN zero-distance links dropped before top-k; bandwidth = median over rows of the farthest surviving link; sync boundaries = unique({0} + beat_frames + {T}).
Not covered here: the vocal onset envelope itself (`dsp.onsetStrength` on the vocal stem, owned by the dsp/features port) — syllables are tested from the dumped envelope.
