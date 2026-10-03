# BSMapper audio analysis → song tokens → candidates: port specification

Source of truth: `/mnt/storage/BeatSaberModelTrainer` (Python, librosa 1.0.0, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, torch 2.11, demucs 4.1.0, torchcrepe 0.0.24, beat_this, soxr 1.1.0, ffmpeg n9.0.1).
Every constant below is copied from the code. Files referenced:

| Stage | Python | Output on disk |
|---|---|---|
| decode, stems, mel/onset/rms, beats, pitch, syllables, sections | `src/bsmapper/analysis/song.py` (`Analyzer.analyze`) | `data/analysis/<key>.json`, `<key>.npz` |
| 4 ms attacks | `src/bsmapper/analysis/grid.py::attacks`, `scripts/compute_attacks.py`, `scripts/prepare_playtest_songs.py` | `data/analysis/<key>.attacks.json` |
| grid | `src/bsmapper/analysis/grid.py::song_grid` | inside tokens npz (`grid`, extended) |
| song tokens (369) | `src/bsmapper/v6/tokens.py::song_tokens`, `scripts/build_song_tokens.py` | `data/v6-song-tokens/<key>.npz` |
| candidates | `src/bsmapper/v6/candidates.py::build`, `src/bsmapper/v7/features.py::candidates`, `scripts/pack_v7.py::song_candidates` | computed at load time |
| music_start/end, intro stem, sustains, repeat segments | `scripts/evaluate_v6_pointer.py::music_start`, `src/bsmapper/v7/generate.py::{music_bounds,sustains,repeat_segments,load_song}` | computed at load time |

`<key>` = SHA-256 of the 16 kHz mono s16le PCM (section 1). Playtest keys: `data/playtest-keys.json`.
Parity data for the 11 playtest songs: `/mnt/storage/BeatFlow/parity/` (section 17).

Contents
1. Decoding, resampling, keys, playtest transcode
2. Stem separation (Demucs htdemucs)
3. Frame clock and `fit`
4. Per-stem log-mel, onset strength, RMS
5. Beats and downbeats (Beat This! final0)
6. Vocal pitch / periodicity (torchcrepe full), vocal_active
7. Syllables
8. Sections (Laplacian segmentation)
9. Attacks (4 ms)
10. `song_grid` (repair, smooth, attack alignment, doubling)
11. Song tokens (369 features)
12. Candidates (V7, 18 features)
13. `load_song` extras: music_start, music_end, intro_stem, sustains, repeat_segments
14. Third-party models and weights
15. Numerics that matter (librosa/numpy/torch specifics)
16. Heavy stages and fallbacks
17. Parity data
18. Out-of-scope consumers of the same analysis files

---

## 1. Decoding, resampling, keys, playtest transcode

Three different decodes of the same file exist. All are `ffmpeg -nostdin -v error -xerror -i <file> -map 0:a:0 -vn ...`:

| name | ffmpeg args | used by |
|---|---|---|
| `mix44` | `-ac 2 -ar 44100 -f f32le` → float32 (2, N44) | Demucs input; `mix44.mean(0)` → `mix16` |
| `mix16` | `librosa.resample(mix44.mean(0), 44100→16000, res_type='soxr_hq')` (python-soxr quality `HQ`; output length fixed to `ceil(N44*16000/44100)` by truncation/zero-pad) | duration, frame count, mix RMS reference, Beat This! input |
| `pcm16` (key) | `-ac 1 -ar 16000 -f s16le` → int16; `samples = int16/32768` float32 | `key = sha256(raw int16 bytes)`; attacks (section 9) |

So the *key* and the *attacks* come from ffmpeg's own downmix + swresample resampler + int16 quantisation, while every
other feature comes from soxr-HQ resampling of float32 stereo-averaged audio. The two 16 kHz signals differ by a few
LSB plus resampler phase/filter differences. In the browser the key cannot be reproduced bit-exactly (decoder and
resampler differ); treat the key as an opaque cache id (e.g. hash of the file bytes) and compare attacks by time matching.

`duration = len(mix16)/16000` seconds. `frames = ceil(len(mix16)/320)`.

`scripts/prepare_playtest_songs.py` additionally produces the *game* audio: `ffmpeg ... -af adelay=2500:all=1 -c:a libvorbis -q:a 6 -f ogg outputs/audio/<id>.egg`
(2.5 s silence prepended, Vorbis q6, original sample rate and channel count kept) and a 512 px cover from the embedded art.
Analysis runs on the *original* file, not the padded one; the 2.5 s `LEAD` is added when notes are exported.

## 2. Stem separation (Demucs htdemucs)

`Analyzer.separate(mix44)`:

```
wav = tensor(mix44)                    # (2, N44) float32 at 44.1 kHz
ref = wav.mean(0); mean = ref.mean(); std = ref.std() + 1e-8     # std: torch unbiased (ddof 1)
out = apply_model(htdemucs, ((wav - mean) / std)[None], shifts=1, split=True, overlap=0.25)[0]   # (4, 2, N44)
out = out * std + mean
stems16 = stack([librosa.resample(s.mean(0), 44100→16000, 'soxr_hq') for s in out])   # (4, N16) float32; drums, bass, other, vocals
```

`apply_model` details (demucs 4.1.0 `demucs/apply.py`):
* `shifts=1`: pads the mix by `max_shift = 22050` samples each side, picks `offset = random.randint(0, 22050)`
  (**unseeded Python RNG: the stems are not bit-reproducible even in Python**), runs the split pass on the shifted
  signal and un-shifts the output. A port may use `shifts=0`; expect stems to differ from the parity dump at
  the level of a different random shift (correlation ≥ 0.99 on loud stems).
* `split=True`: segment = `model.segment` = **7.8 s** (`Fraction(39,5)`, `use_train_segment=True`) → `segment_length = int(44100*7.8) = 343980`,
  `stride = int(0.75*343980) = 257985`, chunk offsets `range(0, N, stride)`; each chunk output is weighted by a
  triangle `cat(arange(1, L//2+1), arange(L - L//2, 0, -1)) / max` raised to `transition_power=1`, summed and divided by the weight sum.
  Chunks shorter than the segment are zero-padded to `training_length = 343980` inside `HTDemucs.forward` and cropped after.
* Model: `htdemucs` bag of **one** model, weights `[[1,1,1,1]]`, `sources = ['drums','bass','other','vocals']`,
  `samplerate 44100`, `audio_channels 2`, `nfft 4096`, `hop 1024`. Weight file:
  `/home/connor/.cache/huggingface/hub/models--adefossez--HTDemucs/snapshots/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003/955717e8.safetensors`
  (84,025,440 bytes; config `htdemucs.yaml` next to it = `models: ['955717e8']`; the demucs package also ships
  `.venv/lib/python3.12/site-packages/demucs/remote/htdemucs.yaml`).
* ONNX contract for the bare network: input `(B, 2, 343980)` float32 stereo at 44.1 kHz
  already standardised by the mix mean/std; output `(B, 4, 2, 343980)`. The STFT/iSTFT (nfft 4096, hop 1024, reflect
  padding `pad = 1536`) are inside `forward` and must be exported or re-implemented.

## 3. Frame clock and `fit`

All frame features run at **50 fps**: 16 kHz, hop **320**; frame `i` is centred at `i*0.02 s` (librosa `center=True`).
`fit(a, frames)` truncates the last axis to `frames` or edge-pads (repeat last value) to reach it. librosa produces
`1 + floor(N/320)` frames, which is ≥ `ceil(N/320)`, so in practice `fit` truncates.

## 4. Per-stem log-mel, onset strength, RMS

For each of the 4 stems `s` (16 kHz mono float32):

**log-mel** `(80, T)`: `librosa.feature.melspectrogram(y=s, sr=16000, n_fft=1024, win_length=400, hop_length=320, window='hann', n_mels=80, power=2.0, center=True)`:
* STFT: window = periodic Hann of length 400 (`scipy.signal.get_window('hann', 400, fftbins=True)`), zero-padded
  *centred* to 1024 (`pad_center`: 312 zeros before, 312 after); signal zero-padded (`pad_mode='constant'`) by 512
  samples both sides; frames every 320; `rfft` 1024 → 513 bins; `|X|^2`.
* Mel filterbank: `librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000, htk=False, norm='slaney')`
  — Slaney mel scale (linear below 1000 Hz with 200/3 Hz per mel, log above with `logstep = ln(6.4)/27`),
  triangular filters between `mel_frequencies(82, 0, 8000)`, each scaled by `2/(f[i+2]-f[i])`.
* `stem_mel = ln(mel + 1e-6)`, then `fit` to T. **Stored as float16** in the npz; the tokens are computed from the
  float16-rounded values (`astype(float32)` of float16).

**onset strength** `(T,)`: `librosa.onset.onset_strength(y=s, sr=16000, hop_length=320)` with all other defaults:
* `S = melspectrogram(y, sr=16000, n_fft=2048, hop_length=320, n_mels=128, fmax=8000, power=2, window='hann' (periodic, 2048), center=True, pad_mode='constant')`
* `S = power_to_db(S)` = `10*log10(max(S, 1e-10))`, then clipped from below to `(global max over the whole (128,T) matrix) − 80 dB` (`top_db=80`, `ref=1.0`).
* `lag=1`, `max_size=1` (ref = S itself): `env[t] = mean_over_128_bands(max(0, S[:, t+1] − S[:, t]))`.
* zero-pad **4** frames at the front (`lag + n_fft//(2*hop) = 1 + 2048//640 = 1 + 3`) and truncate to `S.shape[1]` = `1 + floor(N/320)`.
* `fit` to T, then **z-score per stem over the whole song**: `(onset − mean) / (std + 1e-6)` (numpy population std, ddof 0). Stored float16.

**RMS** `(T,)`: `librosa.feature.rms(y=s, frame_length=1024, hop_length=320)` = zero-pad 512 both sides, frames of 1024
every 320, `sqrt(mean(x^2))`; `fit` to T. Then divided by `max_t rms(mix16)[t] + 1e-8` where the mix RMS uses the same
parameters (`stem_rms` is relative to the mix peak). Stored float16.

## 5. Beats and downbeats (Beat This! final0)

`Audio2Beats(checkpoint_path='final0', device, float16=False, dbn=False)(mix16, 16000)`:

1. `soxr.resample(mix16, 16000, 22050)` (python-soxr default quality `HQ`) → float32. (The audio has been resampled twice: 44.1k → 16k (soxr) → 22.05k (soxr).)
2. `LogMelSpect`: `torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128, mel_scale='slaney', norm=None (torchaudio default: filters NOT area-normalised), normalized='frame_length', power=1, window=torch.hann_window (periodic), center=True, pad_mode='reflect')`.
   `normalized='frame_length'` → `torch.stft(..., normalized=True)` = magnitude divided by `sqrt(1024) = 32`. Output `(F, 128)` with `F = 1 + floor(N22050/441)` (50 fps); `log1p(1000 * S)`.
3. Model `BeatThis(spect_dim=128, transformer_dim=512, ff_mult=4, n_layers=6, head_dim=32, stem_dim=32, dropout={frontend 0.1, transformer 0.2}, sum_head=True, partial_transformers=True)`, eval, fp32. Run with `split_predict_aggregate(chunk_size=1500, border_size=6, overlap_mode='keep_first')`:
   chunk starts `arange(-6, F-6, 1488)`, last start moved to `F - 1494` when `F > 1488`; chunks zero-padded (left `max(0,-start)`, right up to 6); each chunk → `{beat, downbeat}` logits `(1500,)`; 6 border frames dropped on each side; written into an `(F,)` buffer initialised to −1000, **earlier chunks win** on overlap.
   ONNX contract: input `(1, 1500, 128)` float32 log-mel; outputs `beat (1,1500)`, `downbeat (1,1500)` logits.
4. `Postprocessor('minimal', fps=50)`: for each of beat/downbeat logits `x`: `peak = (x == max_pool1d(x, kernel 7, stride 1, pad 3)) & (x > 0)` (local max within ±3 frames and prob > 0.5); peak frame indices; `deduplicate_peaks(width=1)` (runs of indices each ≤ 1 apart are replaced by their running mean, may be non-integer); `/50` → seconds; every downbeat is moved to the nearest beat time; `np.unique` on downbeats.
5. Stored rounded to 4 decimals in `<key>.json`: `beats`, `downbeats`; `tempo_bpm = 60 / median(diff(beats))` (None if < 3 beats).

Weight file: `/home/connor/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` (81,058,141 bytes, Lightning checkpoint;
`state_dict` keys prefixed `model.`; `hyper_parameters` as above plus training-only fields).

## 6. Vocal pitch / periodicity (torchcrepe full), vocal_active

`Analyzer.vocal_pitch(stems16[3], frames)` with `torchcrepe.predict(audio (1,N16), 16000, hop_length=320, fmin=65.0, fmax=1100.0, model='full', return_periodicity=True, batch_size=256, pad=True, decoder=viterbi)`:

1. Framing: zero-pad 512 samples both sides; `total_frames = 1 + N16 // 320`; frames of **1024** every 320 (frame `i` centred at sample `320 i`); per frame subtract mean and divide by `max(1e-10, std)` (**torch unbiased std, ddof 1**).
2. Network CREPE `full` (`.venv/lib/python3.12/site-packages/torchcrepe/assets/full.pth`): input `(B, 1024)` → output `(B, 360)` **sigmoid** probabilities (one per 20-cent bin). Batched 256 frames at a time (batching does not change results).
3. Postprocess: probabilities `(1, 360, F)`; bins `< 62` and `≥ 308` set to −inf (`minidx = floor((1200*log2(65/10) − 1997.3794084376191)/20) = 62`, `maxidx = ceil(... 1100 ...) = 308`).
   Viterbi decoder: `softmax` over the 360 bins per frame, then `librosa.sequence.viterbi(probs, transition)` with
   `transition[i,j] = max(12 − |i−j|, 0)` row-normalised, uniform `p_init`, log domain with `tiny` epsilon → bin path.
   `cents = 20*bin + 1997.3794084376191 + dither`, where **dither is triangular random noise in [−20, +20] cents** (`scipy.stats.triang.rvs(c=0.5, loc=-20, scale=40)`, unseeded) → `f = 10 * 2^(cents/1200)` Hz.
   `periodicity[f] = probabilities[bin[f], f]` (the sigmoid value at the decoded bin, pre-softmax).
4. `periodicity = torchcrepe.filter.median(periodicity, 3)` (window 3, reflect padding, NaN-aware).
5. `midi = 69 + 12*log2(max(f, 1e-3)/440)`; both `fit` to T.
6. `vocal_active = (periodicity > 0.5) & (stem_rms[3] > 0.02)`; `vocal_pitch = midi where vocal_active else NaN`. Stored float16. `vocal_active_share = mean(vocal_active)` in JSON.

The dither makes `vocal_pitch` non-reproducible to ±20 cents (±0.167 MIDI); tokens use pitch only as `(mean − 60)/12`, so the effect on tokens is ≤ 0.014.

## 7. Syllables

`vocal_syllables(stems16[3], vocal_active)`:
* `envelope = librosa.onset.onset_strength(y=vocal, sr=16000, hop_length=320)` (exactly as section 4, **not** z-scored, not `fit`).
* `peaks = librosa.onset.onset_detect(onset_envelope=envelope, sr=16000, hop_length=320, backtrack=False, units='frames')`:
  envelope normalised to `[0,1]` (`− min`, `/ (max + tiny)`), then `peak_pick` greedy with
  `pre_max = floor(0.03*16000/320) = 1`, `post_max = 0*... + 1 = 1`, `pre_avg = floor(0.1*16000/320) = 5`, `post_avg = 5 + 1 = 6`, `wait = 1`, `delta = 0.07`.
  Greedy rule (`__peak_pick_greedy`): frame `n` is a peak iff `x[n] == max(x[n−1 .. n])` (window `[n−pre_max, n+post_max)`) and `x[n] ≥ mean(x[n−5 .. n+5]) + 0.07` (window `[n−pre_avg, n+post_avg)`), after which the next `wait = 1` frame is skipped. Frame 0 uses the truncated windows.
* keep `p` if `p < len(active)` and `active[max(0, p−2) : p+3].any()` (voice active within ±2 frames).
* output `[{time: round(p/50, 3), strength: round(z[p], 3)}]` with `z = (envelope − mean)/(std + 1e-6)` of the un-normalised envelope.

## 8. Sections (Laplacian segmentation)

`structure(stem_mel_float32, beats, downbeats, frames, min_sections=4)` — note it receives the **float32** log-mel (before float16 storage).
If `len(beats) < 16`: one section `{start 0, end frames/50, label 0}`. Otherwise:

1. `mix = ln(Σ_stems exp(stem_mel) + 1e-6)` `(80,T)`; `chroma_src = exp(stem_mel[2]) + exp(stem_mel[3])` (other + vocals power mel).
2. `beat_frames = clip(round(beats*50), 0, T−1)` (numpy round half-to-even).
3. `timbre = sync(mfcc(S=mix, n_mfcc=13), beat_frames, aggregate=median)`: `mfcc` = `scipy.fft.dct(mix, axis=0, type=2, norm='ortho')[:13]` (natural-log mel, no dB conversion, no lifter).
   `sync`: boundaries = `unique({0} ∪ beat_frames ∪ {T})`, segment `j` = `[b_j, b_{j+1})`, median over frames. Number of segments `S_n` = boundaries − 1 (normally `len(beats)+1`).
4. `tonal = sync(_mel_chroma(chroma_src), beat_frames, median)`: `_mel_chroma` folds mel bands whose centre frequency `mel_frequencies(80, 0, 8000)[i]` lies in `[100, 5000]` Hz (bands 3..67, 65 bands) into pitch class `round(12*log2(f/440) + 69) mod 12` (Python `round`, half-to-even), then `librosa.util.normalize(chroma, axis=0)` = each frame divided by its max-abs (frames with max < tiny are left unchanged).
5. `R = recurrence_matrix(tonal, width=3, mode='affinity', sym=True)` (feature axis = rows, time = columns):
   `t = S_n`; `k = 2*ceil(sqrt(t − 5))`; sklearn `NearestNeighbors(n_neighbors=min(t−1, k+6), metric='euclidean')`
   distance graph; zero out `|i−j| < 3`; keep the `k` nearest per row; diagonal 0; `sym`: elementwise `minimum(rec, rec.T)` on the sparse distances (→ mutual-kNN, zeros dropped); affinity `exp(−d / bw)` with `bw = nanmedian_i(k-th smallest distance in row i)` (`med_k_scalar`); returned dense `(t,t)`.
6. `Rf = scipy.ndimage.median_filter(R, size=(1,7))` (mode `reflect`).
7. `path_distance[j] = Σ_c (timbre[c, j+1] − timbre[c, j])^2`; `sigma = median(path_distance) + 1e-9`; `path_sim = exp(−path_distance/sigma)`; `R_path = diag(path_sim, +1) + diag(path_sim, −1)`.
8. `deg_path = R_path.sum(1)`, `deg_rec = Rf.sum(1)`; `mu = deg_path·(deg_path+deg_rec) / (Σ (deg_path+deg_rec)^2 + 1e-9)`; `A = mu*Rf + (1−mu)*R_path`.
9. `L = scipy.sparse.csgraph.laplacian(A, normed=True)` (symmetric normalised `I − D^{-1/2} A D^{-1/2}`); `evals, evecs = np.linalg.eigh(L)` (ascending); `evecs = median_filter(evecs, size=(9,1))` (along time, reflect).
10. `k = clip(round(len(beats)/64) + 2, 4, 10)`; `X = evecs[:, :k] / (‖row‖₂ + 1e-9)`; `labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)` (sklearn 1.9.1, k-means++ init, Lloyd). Labels in `[0, k)`, k ≤ 10; the model embeds labels with `nn.Embedding(32, …)`.
11. Boundaries `bounds = [0] + [i : labels[i] != labels[i−1]]`; `times = beats ++ [T/50]`; section `j` starts at `times[bounds[j]]` (0.0 for `j = 0`; **the index refers to `beats[b]` although sync segment `b` starts at `beats[b−1]`: replicate as is**), snapped to the nearest downbeat if `|d − start| < 0.6*median(diff(beats))`; ends at the next section's unsnapped `times[bounds[j+1]]` (last: `T/50`); `label = labels[bounds[j]]`; start/end rounded to 3 decimals.
12. Merge: walking in order, a section shorter than `2*bar` (`bar = 4*median(diff(beats))`) is absorbed into the previous one (its `end` extends; the previous label is kept). The first section is always kept.

Sections are consumed by the tokens only as `start`, `end`, `label`, and the section index (section 11).

## 9. Attacks (4 ms)

`attacks(samples, sr=16000)` on `pcm16` (`samples = int16/32768`):
* `env = librosa.onset.onset_strength(y=samples, sr=16000, hop_length=64, n_fft=512, lag=1, max_size=3)`:
  mel spectrogram `n_fft=512` (periodic Hann 512, zero-pad 256), hop 64, `n_mels=128`, `fmax=8000`, power 2 → `power_to_db` (global max − 80 dB floor) → `ref = maximum_filter1d(S, size=3, axis=mel)` (mode reflect) → `env = mean_mel(max(0, S[:, t+1] − ref[:, t]))` → front zero-pad `1 + 512//128 = 5` frames, truncate to `1 + floor(N/64)` frames.
* `env = (env − mean)/(std + 1e-6)`; `peaks = scipy.signal.find_peaks(env, height=2.0, distance=int(0.06*16000/64) = 15)` (strict local maxima with `env ≥ 2`, greedy removal of smaller peaks closer than 15 frames = 60 ms).
* `attacks = peaks * 64 / 16000` seconds, stored rounded to 4 decimals (`{key, hop_ms: 4, attacks: [...]}`).
* `MAPPER_CONVENTION = −0.0098 s` is **not** applied here; it is applied by `align_to_attacks` (grid) and by `candidates.build` (candidate time = attack − 0.0098).

## 10. `song_grid` (repair, smooth, attack alignment, doubling)

`song_grid(beats, attack_times)` (`attack_times` always provided for cached tokens; `attack_aligned=True` in the npz):

1. `repair`: `period = median(diff(beats))`; walk beats: drop a beat if `gap < 0.6*period`; if `missing = round(gap/period) − 1 ≥ 1` and `|gap/(missing+1) − period| < 0.15*period`, insert `missing` equally spaced beats; (`< 4` beats: unchanged).
2. `smooth` (`HALF_WIDTH = 8`, `OUTLIER_MS = 25`; `< 5` beats: unchanged): two passes; for each `i`, window `[i−8, i+8]`, weights `w = weights * (1 − (|x−i|/9)^3)^3` (tricube), weighted least squares of `y = a + b (x − i)` (`lstsq` on `sqrt(w)`-scaled rows); `refined[i] = a`. After pass 1 `weights = 0.05 where |beats − refined| > 25 ms else 1` (pass 2 fits the **original** beats with these weights).
3. `align_to_attacks(g, attacks, convention=−0.0098, half_width=8, div=2, window=0.06)`: if `< 4` beats or `< 6` attacks: `g + convention`. Else `pos = interp(attacks, g, 0..n−1)` (NaN outside `[g[0], g[-1]]`, dropped); `off = (pos*2 − round(pos*2))/2 * period` (numpy round half-even; signed seconds from the nearest eighth line); keep `|off| < 0.06`; `fallback = median(off)` (0 if empty); for each beat `i`: `m = |pos − i| ≤ 8`; `out[i] = g[i] + median(off[m])` if `m.sum() ≥ 6` else `+ fallback`; finally `+ convention`.
4. If `60/median(diff(g)) < 90` (`MIN_GRID_BPM`): `double_grid` = sorted union of the beats and their midpoints.

The `grid` stored in the tokens npz is the **extended** grid (section 11), not this output. `verified_grid`/`tempo_map`/`attack_periods`/`constant_grid` in grid.py are **not** used by the token/candidate path.

## 11. Song tokens (369 features)

`song_tokens(info, arrays, attack_times)`; inputs are the float16 arrays cast to float32. `FPS = 50`, `SLOTS = 12`, `FEATURES = 320 + 8 + 24 + 3 + 10 + 4 = 369`.

Grid: `beats = song_grid(info.beats, attacks)`; `grid, lead = extend_grid(beats, duration)`:
`first = beats[1] − beats[0]`, `last = beats[-1] − beats[-2]`; `before = arange(beats[0] − first, −first, −first)[::-1]` then `before = before[before > −first]`; `after = arange(beats[-1] + last, duration + last, last)`; `grid = before ++ beats ++ after`.

Token spans: `starts = (grid[:-1] + diff(grid) * k/4 for k in 0..3)` flattened (4 per grid interval); `ends = starts[1:] ++ [grid[-1]]`; keep `ends > 0 & starts < duration`; `index` = kept positions in the extended sixteenth list (`first_index = index[0]` stored). `T = len(starts)`.
Frame ranges: `lo = clip(floor(starts*50), 0, frames−1)`, `hi = clip(max(lo+1, ceil(ends*50)), 1, frames)`; slices `[lo, hi)`.

Precomputed:
* `melodic = max(onset[2], onset[3])`, `drums = onset[0]` (per frame).
* `syll` = syllable times; `down` = downbeats (from JSON, section 5, **not** grid-aligned).
* sections: `bounds = starts`, `ends_s = ends`; `energy = Σ_stems rms`; `sec_energy[s] = mean(energy[int(start*50) : max(int(start*50)+1, int(end*50))])` (Python `int` = truncation); `vocal_on = (periodicity > 0.5) & (rms[3] > 0.02)`; `sec_vocal[s] = mean(vocal_on[same slice])`.
* `beat_of = index // 4` (beat number in the **extended** grid); `beat_times = grid[clip(beat_of, 0, len(grid)−1)]`.
* `bar_pos`: if downbeats exist, `prev_down = searchsorted(down, beat_times + 0.03) − 1`; `bar_pos = round((beat_times − down[clip(prev_down, 0)]) / median(diff(beats))) if prev_down ≥ 0 else 0`, `astype(int) % 4` (numpy round half-even; `median(diff(beats))` uses the un-extended `song_grid` output). Else `beat_of % 4`.
* `section_index = clip(searchsorted(bounds, starts, 'right') − 1, 0, n_sections−1)`.

Feature layout per token `t` (`a = lo[t]`, `b = hi[t]`):

| index | count | value |
|---|---|---|
| 0–319 | 320 | `mel[stem, bin, a:b].mean(frames)` flattened stem-major (`stem*80 + bin`), stems drums, bass, other, vocals; natural-log mel (≈ −13.8 for silence) |
| 320–323 | 4 | `onset[stem, a:b].max()` (z-scored onset) |
| 324–327 | 4 | `rms[stem, a:b].mean()` |
| 328–339 | 12 | drums onset max in 12 sub-slots: `edges = linspace(starts, ends, 13)`, `f = clip(round(edges*50), 0, frames−1)`, slot `k` = `drums[f[k] : max(f[k]+1, f[k+1])].max()` |
| 340–351 | 12 | same with `melodic` |
| 352 | 1 | `(mean(pitch[a:b][voiced & finite]) − 60) / 12`, `0.0` if no voiced finite frame (`voiced = periodicity > 0.5`) |
| 353 | 1 | `voiced[a:b].mean()` |
| 354 | 1 | number of syllables with `starts[t] ≤ time < ends[t]` |
| 355–358 | 4 | one-hot `index[t] % 4` (sixteenth within beat) |
| 359–362 | 4 | one-hot `bar_pos[t]` |
| 363 | 1 | `1.0` if `min |down − starts[t]| < 0.03` else `0.0` |
| 364 | 1 | `ln( 60 / max(1e-3, (ends[t]−starts[t])*4) / 120 )` (log of local BPM relative to 120) |
| 365 | 1 | `(starts[t] − sec_start) / max(1e-3, sec_end − sec_start)` |
| 366 | 1 | `sec_energy[section]` |
| 367 | 1 | `sec_vocal[section]` |
| 368 | 1 | `min(8, sec_end − starts[t]) / 8` |

`features = nan_to_num(out)`; stored **float16** (`features`), plus `times = starts` (float64), `section_index`, `section_label = sections[section_index].label`, `grid` (extended, float64), `first_index`, `attack_aligned`.
The note model consumes `features` (float16 → float32), `section_label`, `times`.

## 12. Candidates (V7, 18 features)

`song_candidates(key, tok_times)` → `candidates(grid_extended, attacks, syllables, onset_f32, stem_mel_f16, duration)`:

`build` (`MERGE = 0.015`):
1. Events: `(a − 0.0098, kind 0)` for every attack; `(s.time, 1)` for every syllable; `(g, 2)` for every grid line where
   `grid_lines(grid) = unique(round(concat(grid, grid[:-1] + diff*k/div for div in (2,3,4), k in 1..div−1), 4))` — the 1/2, 1/3, 2/3, 1/4, 3/4 positions of the **extended** grid, rounded to 4 decimals and deduplicated. Keep `0 ≤ t < duration`; sort by `(time, kind)`.
2. Merge: walking in order, an event with `t − last.time ≤ 0.015` joins the last candidate: sets `flags[kind] = 1` and, if it is an attack, **replaces the candidate time** with the attack time (later grid/syllable events never move it). Otherwise it starts a new candidate with its own time.
3. `frames = clip(round(times*50), 1, T−2)`; `stems[c, s] = max(onset[s, f−1], onset[s, f], onset[s, f+1])`; `salience = stems.max over stems`.
4. `pos = grid_position(times, grid)` (fractional beat index: `interp` inside, linear extrapolation with the edge periods outside); `frac = pos − floor(pos)`.
   Subdivision one-hot (6): for `(div, j)` in `((1,0),(2,1),(4,2),(3,3),(6,4))` in that order, the first `div` with `|frac*div − round(frac*div)| / div * median(diff(grid)) ≤ 0.012` s wins; else index 5 (off-grid). (Python `round`, half-even.)
5. `phase = [sin(2π frac), cos(2π frac)]`.
6. `features = [flags(3), clip(stems, −3, 6)/3 (4), clip(salience, −3, 6)/3 (1), sub (6), phase (2)]` = 16.

V7 columns (`features.py`): `kick = max(0, diff(Σ_{bin<9} stem_mel[0, bin, :], prepend first))` from the **float16** mel cast to float32, `f = clip(round(times*50), 1, len−2)`, `k = max(kick[f−1], kick[f], kick[f+1]) / (percentile(kick, 95) + 1e-6)` (numpy linear percentile); column 16 = `clip(k, 0, 3)`; column 17 = `1 if flags[0] + flags[1] > 0 else 0` (sound flag). **`CAND_FEATURES = 18`**.
Token assignment: `cand_tokens = clip(searchsorted(tok_times, cand_times, 'right') − 1, 0, T_tok − 1)`.
Consumed as `cand_times` float32, `cand_feats` float16, `cand_tokens` int32.

## 13. `load_song` extras

`load_song(key)` returns `features, section_label, times, key, cand_times, cand_feats, cand_tokens, duration, music_start, segments, music_end, intro_stem, sustains`.

* `music_start`: `e = Σ_stems stem_rms` (float32 from float16); first index with `e > 0.1 * median(e[e > 0])`, `/50`; `0.0` if none.
* `music_bounds`: `music_end` = last index with `e > 0.1*median(e[e>0])`, `/50` (`None` if none); `a = int(music_start*50)`, `b = a + int(6.0*50)`; `intro_stem = argmax_stem(rms[:, a:b].mean(1))` (0 if `b ≤ a`).
* `sustains(min_vocal=0.6, min_other=0.8)` → sorted list of `(start, end)` seconds:
  * vocal: `vocal = (periodicity > 0.5) & (rms[3] > 0.02)`; syllable times sorted `syl`; for consecutive `(a, b)` (last `b = syl[-1] + 10`): `f0 = int(a*50)`, `f1 = int(min(b, a+6)*50)`, `run = vocal[f0:f1]`; `held = argmin(run)` if `run` non-empty and not all true, else `len(run)`; if `held/50 ≥ 0.6` → `(a, a + held/50)`.
  * other: `o = rms[2]`; `loud = o > 0.5*median(o[o>0])` (all false if no positive); `p = max(onset[2], 0)`; `pk = find_peaks(p, height = mean(p) + 1.5*std(p), distance=5)`; between consecutive edges `pk ++ [T]` starting at 0: a segment `[start, e)` with `mean(loud) > 0.9` and `(e−start)/50 ≥ 0.8` → `(start/50, e/50)`.
* `repeat_segments(features, times, min_seconds=4.0, threshold=0.88, hole=6)`:
  * `similarity_matrix(features, context=16)`: `mel = features[:, :320]` (float16 → float32) minus its column means; windowed descriptor `desc[i] = Σ mel[i−16 .. i+16] / 33` via cumulative sum over a zero-padded array (edge windows are **sums over fewer tokens divided by 33**, not means), L2-normalised (+1e-6); `local = mel / (‖mel‖ + 1e-6)`; `S = 0.7 desc·descᵀ + 0.3 local·localᵀ`.
  * `min_lag_tok = searchsorted(times, times[0] + 4.0)`; for every lag `L ≥ max(1, min_lag_tok)`: `d = diagonal(S, −L) ≥ 0.88` (`d[k]` compares token `k+L` with `k`), `idx = nonzero(d)`; skip if `< 8` hits; split runs where consecutive hits are more than `hole + 1 = 7` apart; run `[a, b]` covers destination tokens `k0 = idx[a]+L .. k1 = idx[b]+L`, length `times[k1] − times[k0]` (skip `< 4 s`); each destination token keeps the longest covering run (`>` with 1e-6 margin; earlier lags keep ties).
  * Emit maximal runs of equal `best_lag ≠ 0` with `times[j] − times[k] ≥ 4` as `{start: times[k], end: times[j], lag: times[k] − times[k−L], lag_tokens: L}`.
  * (`token_repeats` in `v6/repeats.py` with `THRESHOLD = 0.92` and `MIN_LAG = 4.0` is used only by `evaluate_v6_pointer.load_song`, not by v7+.)

## 14. Third-party models and weights

| Model | Variant | Weights on this machine | Input → output |
|---|---|---|---|
| Demucs | `htdemucs` (Hybrid Transformer Demucs, 4 stems, single-model bag `955717e8`) | `/home/connor/.cache/huggingface/hub/models--adefossez--HTDemucs/snapshots/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003/955717e8.safetensors` (84.0 MB) | `(B, 2, 343980)` standardised 44.1 kHz stereo → `(B, 4, 2, 343980)`; sources drums, bass, other, vocals; section 2 |
| Beat This! | `final0` (`BeatThis` transformer, 50 fps, no DBN) | `/home/connor/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` (81.1 MB, Lightning; strip `model.` prefix) | `(1, 1500, 128)` log1p(1000·mel) at 22.05 kHz → beat/downbeat logits `(1, 1500)` each; section 5 |
| CREPE | torchcrepe `full` (0.0.24) | `/mnt/storage/BeatSaberModelTrainer/.venv/lib/python3.12/site-packages/torchcrepe/assets/full.pth` | `(B, 1024)` normalised 16 kHz frames → `(B, 360)` sigmoid bin probabilities; section 6 |

Everything else (librosa, scipy, sklearn KMeans, numpy) is plain numerics.

## 15. Numerics that matter

* **Resampling**: `soxr` quality `HQ` (44.1k→16k for mix and stems; 16k→22.05k for Beat This!); `fix=True` crops/zero-pads to `ceil(N·ratio)`. The attacks path uses ffmpeg `swresample` to 16 kHz + int16 rounding. Browser resamplers will differ → compare downstream by tolerance, not bit-exactly.
* **STFT conventions**: librosa `center=True`, `pad_mode='constant'` (zeros), periodic Hann, window zero-padded centred to `n_fft`, `rfft`. torchaudio (Beat This!) `center=True`, `pad_mode='reflect'`, periodic Hann, `normalized` by `1/sqrt(n_fft)`.
* **Mel filterbanks**: librosa Slaney scale + Slaney area normalisation (`fmin 0`, `fmax 8000`, `n_mels 80` for stems; `n_mels 128` for onset envelopes); torchaudio Slaney scale **without** normalisation (`f_min 30`, `f_max 11000`, 128 mels).
* **`power_to_db` floor**: `max(x, 1e-10)` then clip at `global max − 80 dB` — the global max is over the whole song, so the onset envelope depends on the loudest frame.
* **Onset envelope padding**: 4 frames (stems/syllables) or 5 frames (attacks) of zeros at the front; frame count `1 + floor(N/hop)` then `fit`.
* **z-scores**: numpy population std (ddof 0) for onsets, attack envelope, syllable strength; torch unbiased std (ddof 1) for CREPE frame normalisation and the Demucs mix std.
* **Rounding**: numpy `round` and Python `round` are half-to-even (bar position, sub-slot edges, candidate frames, subdivision test, chroma pitch class). `int()` truncates (section slices, sustains).
* **float16 storage**: `stem_mel`, `stem_onset`, `stem_rms`, `vocal_pitch`, `vocal_periodicity` are float16 on disk and tokens/candidates are computed from the rounded values; `features` and `cand_feats` are stored float16. A port should round these intermediates to float16 before pooling to match within ~1e-3.
* **`find_peaks`** (attacks, sustains): strict local maxima with plateau handling, then greedy distance pruning keeping the higher peak.
* **`searchsorted`/`interp`** semantics as numpy (left/right as stated).
* **Random**: Demucs shift (`random.randint`, unseeded), CREPE dither (`scipy.stats.triang.rvs`, unseeded), KMeans (`random_state=0`, deterministic but implementation-specific k-means++). Section labels are reproducible only up to a permutation and small boundary moves.
* **`np.linalg.eigh`** of the normalised Laplacian: eigenvector signs are arbitrary but the clustering is sign-invariant per column; eigenvalue ties can rotate eigenvectors (rare).
* **Edge windows** in `similarity_matrix` are sums/33, not means (section 13).

## 16. Heavy stages and fallbacks

| Stage | Cost | Exact reproduction? | Fallback |
|---|---|---|---|
| Demucs htdemucs | by far the heaviest (transformer over 7.8 s chunks at 44.1 kHz; minutes on CPU/WASM, seconds on WebGPU) | no (random shift; different conv/STFT kernels) | compare stems by correlation ≥ 0.99 (vocal stem is dumped; others only via features); **recompute mel/onset/rms/tokens from the browser stems** with a Python harness using `song.py` feature code and compare those to the browser tokens at 1e-3 |
| Beat This! | moderate (6-layer transformer on 30 s chunks) | near-exact (deterministic; ONNX fp32 logits within ~1e-3) | beats compared as time lists: ≥ 98 % matched within 10 ms; downstream grid within 5 ms |
| torchcrepe full | moderate (one 1024-sample CNN per 20 ms frame, ~22 M params) | no (dither); Viterbi must be ported | periodicity max-abs ≤ 0.02 after median filter; pitch ±0.2 MIDI on voiced frames; token feature 352 tolerance 0.05 |
| Sections (kNN + eigh + KMeans) | light but numerically fragile | partition equal up to relabelling in most songs | boundaries within one beat, labels by adjusted Rand index ≥ 0.8; if the partition differs, token features 365–368 and `section_label` differ for the affected tokens only; acceptable if the note model is evaluated end-to-end on playtests |
| librosa mel/onset/rms, attacks, grid, tokens, candidates, sustains, repeats | light | yes (pure arithmetic) | none needed; tolerances in `parity/index.json` |

If stems cannot be matched, the cleanest parity test is two-stage: (1) browser stems → dump → Python `song.py`
feature code on those stems → compare tokens with the browser tokens at 1e-3; (2) separately compare browser stems
with the Python vocal stem by correlation only.

## 17. Parity data

`/mnt/storage/BeatFlow/parity/<song-id>/` (11 playtest songs; produced by `scripts/parity_dump.py` in the trainer repo from the cached analysis files, so they are exactly what the note model was run on). `parity/index.json` lists every file with shape, dtype and tolerance. Layout per song:

```
pcm16.wav                  16 kHz mono int16: the exact PCM whose SHA-256 is the key (attacks input)
vocal_pcm.wav              16 kHz mono int16 vocal stem (clip(stem,-1,1)*32767)
analysis.json              beats, downbeats, tempo, syllables, sections, duration, frames, key
attacks.json               4 ms attack times (s), hop_ms
stem_mel.f16.bin           (4, 80, T) float16, C order (as stored in Python)
stem_onset.f16.bin         (4, T) float16 (z-scored)
stem_rms.f16.bin           (4, T) float16
vocal_pitch.f16.bin        (T,) float16 MIDI, NaN unvoiced
vocal_periodicity.f16.bin  (T,) float16
grid.json                  song_grid output (un-extended), first_index, music_start/end, intro_stem, sustains, segments
grid_extended.f64.bin      extended grid (as in the tokens npz)
tokens.f16.bin             (T_tok, 369) float16 (as stored and consumed)
token_times.f64.bin        (T_tok,) float64
section_index.i32.bin, section_label.i32.bin
cand_times.f32.bin         (C,) float32
cand_feats.f16.bin         (C, 18) float16
cand_tokens.i32.bin        (C,) int32
shapes.json                shapes/dtypes of every .bin
```

The drums/bass/other stems are not on disk anywhere (only `vocal_pcm` was kept), so they are compared only through
derived features.

`tokens` in the dump are **recomputed** from the dumped arrays (self-consistent with this spec). For the 7 songs added
on 2026-09-29 (moody … tongue) the analysis files were regenerated on 2026-09-30 (`analyze_songs.py`, same models)
after `data/v6-song-tokens` was built, so the production token cache is stale for them; the cached tokens the note model
consumed in the v6–v10 playtests are kept as `tokens_cached.f16.bin` / `section_label_cached.i32.bin` and
`grid.json.cached_vs_recomputed` records the difference. Because only the Demucs random shift (and hence everything
downstream of the stems) changed between the two runs, this is a direct measurement of stem-level nondeterminism:
moody: token-mel correlation 0.998, median |Δ log-mel| 0.07, grid-derived columns 355–364 identical, syllable count
changed in 53/1416 tokens, section label agreement 9 % (20 vs 19 sections). Use these as the realistic floor for
cross-implementation tolerances on stem-derived features and sections. All `.bin` files are raw little-endian arrays; read with `new Float16Array/Float32Array/...` over the buffer.

## 18. Out-of-scope consumers of the same analysis files

The full playtest pipeline (`scripts/make_playtest_v10.py`) also reads the analysis for things outside this spec;
they need the same inputs but are owned by other ports: `bsmapper.v7.style.song_style` (`<key>.json`, `stem_rms`),
`bsmapper.v9.look_metrics.SongRef` (`<key>.json`, attacks + `MAPPER_CONVENTION`, `Σ stem_rms`), `v9.canvas.canvas_targets`,
`v91.walls`. All derive from `stem_rms`, attacks, beats and sections described above.
