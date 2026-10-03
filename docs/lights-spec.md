# Lights spec

Source of truth: `src/bsmapper/v9/canvas.py`, `src/bsmapper/v9/shape.py`, `scripts/make_playtest_v91.py` (`flash_cap`, `INTENSITY_BUCKET`, `FLASH_CAP`, `LEAD`), `scripts/make_playtest_v10.py` (call order). Model: `public/models/lights/canvas.onnx` (+ `canvas.int8.onnx`, `spec.json`). Fixtures: `parity/<song>/lights/`, index `parity/lights.json`.

## Contents

1. Pipeline order
2. Canvas model (inputs, outputs, graph)
3. Token vocabularies per channel
4. `note_context` features
5. Conditions, `INTENSITY_BUCKET`
6. Sampling procedure
7. Given tokens (walls)
8. `decode`: tokens to events
9. `shape_contrast`, `CONTRAST_FLOOR`, `SongRef` audio quantities
10. `flash_cap`
11. Packaging (`basicBeatmapEvents`, `colorBoostBeatmapEvents`)
12. Parity fixtures and tolerances
13. Not exported

## 1. Pipeline order

Per difficulty, after notes and walls exist (song seconds, no lead):

```
x       = concat(song_token_features (T,369), note_context(notes, times) (T,17))      float32 (T,386)
given   = canvas_targets(times, walls, [], [])                                        (T,11) int, channel 0 only
tokens  = sample(x, times, notes, diff, walls_c, lights_c, seed, channels=1..10, given, njs)
_, ev, bo = decode(tokens, times, lead=LEAD)                                          LEAD = 1.5 s
ev      = shape_contrast(ev, SongRef(key), CONTRAST_FLOOR[intensity], lead=LEAD, seed=shape_seed)
ev      = flash_cap(ev, FLASH_CAP[intensity])
```

`ev` and `bo` go to the packager as `(time_padded_seconds, et, value, f)` and `(time_padded_seconds, on)`; `engine/package.ts#toV3` already converts seconds to beats.

Seeds (trainer): `seed = crc32(repr((song_id, difficulty, 'lights'))) % 10000`, `shape_seed = crc32(repr((song_id, difficulty))) % 10000`. The browser may use any seed; see tolerances.

`times` = the song-token timeline (`data/v6-song-tokens/<key>.npz: times`, one per 1/4 beat, song seconds; may start slightly negative).

## 2. Canvas model

Masked discrete diffusion transformer, 14.53 M parameters: input LayerNorm(386) + Linear(386, 384); 11 token embeddings (VOCAB[c]+1, 384) summed; 6 condition embeddings summed and broadcast over T; sinusoidal position table (max 2048) added; 8 pre-norm TransformerEncoder layers (6 heads, FFN 1536, ReLU); final LayerNorm; 11 linear heads.

ONNX graph (opset 18, 452 nodes, dynamic `T` 2..2048, batch fixed at 1; standard ops only: MatMul, Gemm, LayerNormalization, Softmax, Gather, Reshape, Transpose, Slice):

| input | dtype, shape | meaning |
|---|---|---|
| `x` | float32 (1,T,386) | 369 song-token features (`docs/analysis-spec.md`) ++ 17 `note_context` |
| `tokens` | int64 (1,T,11) | current state; `VOCAB[c]` (= masked) for open entries |
| `diff` | int64 (1,) | index in `[Easy, Normal, Hard, Expert, ExpertPlus]` |
| `walls_c` | int64 (1,) | walls/min bucket (0 unknown, 1..5) |
| `lights_c` | int64 (1,) | light events/min bucket (0 unknown, 1..5) |
| `lights_q` | int64 (1,) | light quality quartile (0 unknown, 1..4) |
| `walls_q` | int64 (1,) | map-rating quartile (0 unknown, 1..4) |
| `notes_known` | int64 (1,) | 1 when `x` carries real notes |

Outputs `logits_0..logits_10`: float32 (1,T,VOCAB[c]). Sizes: fp32 61.97 MB, int8 (dynamic weight quantisation) 18.48 MB. Numerics on a real window (`scripts/check_onnx_lights.py`, bereal Expert): fp32 max abs 1e-5 vs PyTorch; int8 max abs 0.13 (768 tokens) / 0.22 (397 tokens) on logits of scale ~8, argmax agreement 100%. A WebGPU variant (`canvas.fp16.onnx`, onnxconverter-common float16 with fp32 I/O, 31.31 MB, no blocked nodes since the position table is a constant) differs by at most 0.008 with 100% argmax agreement; the int8 graph's DynamicQuantizeLinear/MatMulInteger nodes fall back to CPU under the WebGPU EP, so prefer fp16 there.

## 3. Token vocabularies

`VOCAB = [73, 26, 26, 26, 26, 26, 2, 2, 6, 6, 3]`, `MASK = 73` (shared id; each embedding has VOCAB[c]+1 rows and the mask is row VOCAB[c]). Token 0 is always "none / no event".

| channel | content | vocab |
|---|---|---|
| 0 | wall start | 0 none; `1 + (kind*6 + lane)*4 + dur`, kind in [side, crouch, dodge], lane index of [-1,0,1,2,3,4], dur bucket 0..3 |
| 1..5 | light group et 0 back lasers, 1 ring lights, 2 left lasers, 3 right lasers, 4 centre | 0 none; 1 off; `2 + (mode*2 + colour)*3 + bright`, mode in [on, flash, fade, transition], colour in [A, B], bright in [dim, normal, bright] |
| 6, 7 | ring spin (et 8), ring zoom (et 9) | 0 none, 1 event |
| 8, 9 | left (et 12) / right (et 13) laser speed | 0 none; 1..5 = value 0, 1, 3, 6, 8 |
| 10 | boost | 0 none, 1 on, 2 off |

Encoding of human data (for reference): light value v > 12 → 0, v = 0 → 1, colour = A if v >= 5 else B, mode = (v-1) mod 4, bright = 0 if f < 0.7, 1 if f < 1.15, else 2. Speed bucket = `1 + count(SPEED_BUCKETS (0,2,4,7) <= value)` via `searchsorted` left: value <= 0 → 1, <= 2 → 2, <= 4 → 3, <= 7 → 4, more → 5.

## 4. `note_context(notes, times)` (17 floats per token)

`notes` in song seconds with `x` 0..3, `y` 0..2, `hand` 0 left / 1 right. All zeros when there are no notes.

| columns | definition |
|---|---|
| 0..11 | cell `c = x*3 + y`; `nxt` = time of the first note in that cell with `t_note >= t_token` (searchsorted left), ∞ if none; value `clip(1 - nxt_delta / 1.5, 0, 1)` |
| 12..14 | for `w` in (0.5, 1.0, 2.0) s: `count(t_token <= t_note < t_token + w) / (w * 8)` (searchsorted left on both ends; not clipped) |
| 15, 16 | hand 0, hand 1: `x / 3` of the last note of that hand with `t_note <= t_token` (searchsorted right − 1); before that hand's first note: 0.0 for hand 0, 0.5 for hand 1; a hand with no notes at all stays 0 |

`x[:, 369:386]` = these columns; `x[:, :369]` = the song-token features cast to float32 (stored float16 in the trainer).

## 5. Conditions

| condition | value used |
|---|---|
| `diff` | `DIFFICULTIES.index(difficulty)` (Expert 3, ExpertPlus 4) |
| `walls_c` | `wall_style or 1`: 4 when the song has a wall plan (phrases_per_min > 0), else 1 |
| `lights_c` | `INTENSITY_BUCKET = {calm: 3, normal: 4, intense: 5}` |
| `lights_q` | 4 (ask for the best-lit quartile) |
| `walls_q` | 4 |
| `notes_known` | 1 if any notes |

Bucket edges, for completeness: `WALL_AMOUNT = (0, 2, 5, 11)` walls/min, `LIGHT_DENSITY = (100, 400, 900, 1600)` events/min, `bucket = 1 + count(edges <= value)`, 0 = unknown.

## 6. Sampling (`sample`, random-order ancestral unmasking)

Constants: `WINDOW = 768`, `OVERLAP = 128`, `STEPS = 16`, `temperature = 1.0`, generated channels `1..10`, channel 0 given.

```
tokens[:, c] = MASK for generated channels; tokens[:, 0] = given[:, 0]
start = 0
loop:
  end = min(T, start + 768)
  open = (tokens[start:end] == MASK) & generated        # (W, 11) bool
  total = open.sum()
  for s in 0..15:
    if nothing open: break
    logits = model(x[start:end], tokens[start:end], conds)
    target_open = floor(total * cos(pi/2 * (s+1)/16))   # entries still open after this step
    cands = []
    for c in 1..10:                                      # channel order matters for the RNG stream
      pos = indices where open[:, c]
      p = softmax(logits[c][pos] / temperature)
      u = rng.random((len(pos), 1)); choice = first index where cumsum(p) > u   # per row
      order = rng.random(len(pos))
      cands += (order[k], pos[k], c, choice[k]) for k
    sort cands by order descending (stable)
    n_commit = max(1, open.sum() - target_open)
    for the first n_commit: tokens[start+pos, c] = choice; open[pos, c] = false
  if end == T: break
  start = end - 128                                      # the overlap is already committed and acts as context
```

The trainer RNG is numpy `default_rng(seed)` (PCG64, `random()` = 53-bit doubles); draws happen exactly in the order above. The browser `Rng` is a different generator, so sampled tokens are compared statistically (section 12). Windows for the fixtures: T 1164 → [0,768], [640,1164]; T 1416 → [0,768], [640,1408], [1280,1416]; T 1262 → [0,768], [640,1262].

When walls are generated (channel 0 open; not the browser flow, walls come from `engine/walls.ts`) every committed wall token must pass `fair_wall` or becomes 0. With channel 0 given this never runs.

## 7. Given tokens: `canvas_targets(times, walls, [], [])`

For each wall `w` (song seconds): `tok = wall_token(w)`, placed at the nearest token: `i = clip(searchsorted(times, t - 1e-3), 0, T-1)`; if `i > 0` and `|times[i-1] - t| < |times[i] - t|` then `i - 1`. Later walls overwrite earlier ones on the same token.

`wall_token`: `kind(w)`: `y >= 2 and h <= 3` → crouch; `y == 0 and h >= 3` → dodge if lanes `[x, x+max(1,w))` meet {1,2} else side; otherwise "other" → token 0. `lane = index of clip(x, -1, 4) in [-1,0,1,2,3,4]`, `dur = count(DUR_BUCKETS (0.1, 0.5, 1.5) <= duration)` (searchsorted left: < 0.1 → 0, < 0.5 → 1, < 1.5 → 2, else 3). `tok = 1 + (kind*6 + lane)*4 + dur`, kind in [side 0, crouch 1, dodge 2].

Walls are never decoded from the canvas in the browser flow (the chart keeps the planned walls), but `wall_from_token` for completeness: `DUR_VALUES = (0.06, 0.25, 0.9, 2.2)`; side `x=lane, y=0, w=1, h=5, d=DUR_VALUES[dur]`; crouch `x = 0 if lane <= 0 else min(lane, 2), y=2, w = 4 if lane <= 0 else 2, h=3, d = max(DUR_VALUES[dur], 0.25)`; dodge `x = clip(lane, 1, 2), y=0, w=1, h=5, d = max(DUR_VALUES[dur], 0.25)`.

## 8. `decode(tokens, times, lead)`

Iterate tokens in index order; `t = times[i] + lead`. Per token, emit in this order (this order is the RNG order for `shape_contrast`):

| channel token | event `(t, et, value, f)` |
|---|---|
| light channel g (et 0..4), tok 1 | `(t, g, 0, 1.0)` |
| light channel g, tok >= 2 | `m = tok - 2; mc = m div 3; bright = m mod 3; mode = mc div 2; colour = mc mod 2; value = (5 if colour == 0 else 1) + mode; f = (0.5, 1.0, 1.4)[bright]` |
| channel 6 == 1 | `(t, 8, 0, 1.0)` |
| channel 7 == 1 | `(t, 9, 0, 1.0)` |
| channel 8 tok k >= 1 | `(t, 12, (0, 1, 3, 6, 8)[k-1], 1.0)` |
| channel 9 tok k >= 1 | `(t, 13, (0, 1, 3, 6, 8)[k-1], 1.0)` |
| channel 10 | boost `(t, tok == 1)` |

Beat Saber v3 values: 1..4 colour B on/flash/fade/transition, 5..8 colour A, 0 off; `f` is the brightness float.

## 9. `shape_contrast(events, ref, floor, lead, seed)`

`CONTRAST_FLOOR = {calm: 0.7, normal: 0.5, intense: 0.35}`.

Audio quantities (`SongRef(key)`): `sections` = `data/analysis/<key>.json: sections` (list of `{start, end, label}`, song seconds); `energy` = `data/analysis/<key>.npz: stem_rms` (4 stems × frames at 50 fps, float16) summed over stems, as float32. `downbeats` and `attacks` are loaded by `SongRef` but unused here.

```
if len(sections) < 3: return events unchanged
e[i]    = mean(energy[int(start_i*50) : max(int(start_i*50)+1, int(end_i*50))])
rank[i] = (position of e[i] in ascending order, 0-based) / max(1, n-1)      # argsort(argsort(e))
keep_p  = floor + (1 - floor) * rank
starts  = section starts + lead
rng = default_rng(seed)
for ev in events (decode order):
  if ev.et in 0..4 and ev.value in (1, 2, 5, 6, 9, 10):                   # on or flash
    i = max(0, searchsorted(starts, ev.t, 'right') - 1)
    if rng.random() > keep_p[i]: drop
  keep
```

One uniform draw per on/flash light event, in event order; offs, fades, transitions, rings, speeds and boosts are untouched. Per-song `section_energy`, `rank` and `keep_p` for every intensity are in `parity/<song>/lights/style.json: shaping`.

## 10. `flash_cap(events, per_second)`

`FLASH_CAP = {calm: 8, normal: 12, intense: 18}` flashes per rolling second across all groups.

```
recent = []
for e in sorted(events):                                  # tuple order: time, et, value, f
  if e.et in 0..4 and e.value in (2, 6, 10):              # flash
    recent = [t for t in recent if e.t - t < 1.0]
    if len(recent) >= per_second: drop e; continue
    recent.append(e.t)
  keep e
```

Output is time-sorted. Boosts are unchanged.

## 11. Packaging

`basicBeatmapEvents: {b: beat(t), et, i: value, f}`; `colorBoostBeatmapEvents: {b: beat(t), o}`; `useNormalEventsAsCompatibleEvents: true`; event box groups empty. When a chart has no lights the trainer writes `{b: 0, et: 0..4, i: 1, f: 0.5}`. The chosen `intensity` only enters through `lights_c`, `CONTRAST_FLOOR` and `FLASH_CAP`.

## 12. Parity fixtures (`parity/lights.json`)

Songs bereal, moody, lights-out; difficulties Expert, ExpertPlus; notes and walls = the v10.0-pre2 packages (all three songs drew wall_style 0, so `walls_c = 1` and `given` is all zeros; the given-walls path is specified above but not exercised by the fixtures). The CPU float32 resample with the trainer seeds reproduces the shipped `.dat` events and boosts exactly, so the fixtures equal what was playtested.

| file | contents |
|---|---|
| `canvas_<diff>.npz` | `x` (T,386) float32, `times`, `given` (T,11) int16, `tokens` (T,11) int16 sampled, `logits0_c` (768, VOCAB[c]) float16: step-0 logits of window 0 with `tokens = given walls + light channels masked` |
| `trace_<diff>.npz` | `step_k`: window-local token state before model call k (16 calls per window) |
| `events_<diff>.json` | notes, walls, seeds, conditions, `decoded`, `shaped`, `capped` as `[t+lead, et, value, f]`, `boosts` as `[t+lead, on]`, counts |
| `style.json` | see `docs/style-spec.md` |

Tolerances: `x` max abs 1e-3; `logits0` fp32 2e-3 (stored float16), int8 0.5 with argmax agreement >= 0.98; `tokens` per-channel non-zero counts within 20%, light events/min within 20%; `decoded` exact given the same tokens; `shaped` counts per section within 15%; `capped` exact given the same shaped events.

## 13. Not exported

- `fair_wall` wall checking inside `sample` (channel 0 is never generated in the browser; `engine/wallRules.ts` already has the rules).
- numpy PCG64 (replace with the app RNG; token parity is statistical).
- `snap_walls`, `music_marks` (walls stage, done elsewhere).
