# DSP / analysis port status (2026-10-02)

Done and parity-checked: `src/engine/dsp.ts`, `features.ts`, `grid.ts`, `tokens.ts`, `candidates.ts`, `scripts/parity-analysis.ts`.
Run `npx tsx scripts/parity-analysis.ts [song ...]` (exit code 1 on any failing check). All 11 songs pass every `parity/index.json` tolerance.

Modules:
- `dsp.ts`: FFT/STFT, Slaney mel banks, power_to_db, onset strength, rms, numpy float32 pairwise sums, float16 rounding, scipy `findPeaks`, librosa `peakPick` (exported; `vocals.ts` keeps its own copy), numpy-exact `interp`/`percentile`/`rint`, Kaiser polyphase `resample`.
- `features.ts`: per-stem log-mel/onset (z-scored)/rms relative to mix peak (`stemFeatures`, `storedFeatures` = float16 rounding), 4 ms `attacks`, Beat This! input `beatSpectrogram` (16 k -> 22.05 k, frame-major `(F,128)`), `crepeFrames` (ddof 1), `vocalSyllables` (uses `vocals.syllables`).
- `grid.ts`: repair, tricube smooth (weighted normal equations), `alignToAttacks`, doubling, `songGrid`, `extendGrid` (numpy `arange` fill semantics), `barPositions`.
- `tokens.ts`: `songTokens(info, arrays, songGrid)` -> 369 float32 features (float32 means via `meanF32`, bool means in float64); round with `toFloat16` before the model.
- `candidates.ts`: grid lines, merge, 18 features (kick column included), token assignment.
- `music_start/end/intro_stem` are checked through `structure.ts::musicBounds` (owned by the structure port).

Resampler: modelled on soxr HQ; constants tuned against the soxr dumps to passband 0.9137 (= soxr's `1 - 0.05/TO_3dB(120.4 dB)`) and 136 dB Kaiser. 44.1 k -> 16 k max abs 1.5e-4 vs soxr; Beat This! log-mel max abs 4.4e-3 (bereal) / 8.4e-3 (tweety), only in the 7.4-8 kHz bands (identical to 1e-5 below 7 kHz).

Parity method: tokens and candidates are checked stage-isolated, as the tolerances specify ("given the same grid / token_times"): tokens from the dumped float16 arrays and the reference song grid (sliced from `grid_extended.f64.bin`), candidates from `grid_extended` and `token_times`. Result: tokens, section index/label, cand_times/feats/tokens are bit-identical on all 11 songs (only pocket-locket kick column 4.9e-4). The script also prints an informational end-to-end line (our own `songGrid` feeding tokens and candidates).

Known end-to-end residual (not a port bug): Python's `smooth` uses `np.linalg.lstsq` (OpenBLAS dgelsd), whose result carries 1-ulp noise. When an attack time coincides exactly with a smoothed beat at the +-8-beat window edge of `align_to_attacks`, or a grid line coincides exactly with a token start, that noise decides the outcome. Ties exist in tweety (1), tongue (3), pocket-locket (3), poseidon (1); grid differences there are 2.1e-5 / 3.6e-5 / 1.3e-4 s (tolerance 2e-3). End-to-end effect: tweety 2 token values (one 1/48-beat slot edge), foggy-windows 145/2462 candidates (perfect 4-decimal beat lattice: candidates exactly on token starts land in the neighbouring token, gap 1e-16 s), poseidon 17, pocket-locket 14, tongue 1, all others 0. Matching this would need a bit-exact port of dgelsd plus OpenBLAS Haswell kernels (FMA); not attempted.

Node timings (bereal, 137.5 s, i7-13700K): attacks 0.4-0.5 s, 4x stem mel/onset/rms 1.2-1.6 s, Beat This! log-mel incl. resample 0.6-0.8 s, syllables 0.2-0.3 s, grid+tokens+candidates 40-75 ms; 44.1 k -> 16 k resample 0.9 s per mono signal (mix + 4 stems).

Parity table (all PASS): attacks 100 % within 4 ms on all 11 songs; syllables 100 % (strength <= 1e-3); vocal onset rel err <= 5e-7; grid max abs 5e-7 (rounding of grid.json) except tweety 2.1e-5, tongue 3.6e-5, pocket-locket 1.3e-4; tokens/cands exact; music bounds exact. bereal/tweety DSP dumps: mix_mel 9.5e-7, mix_onset <= 2.9e-6, mix_rms 1.2e-7, attack_env <= 5.7e-6, crepe_frames <= 4.8e-7.
