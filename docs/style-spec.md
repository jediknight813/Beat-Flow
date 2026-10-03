# Style spec

Source of truth: `src/bsmapper/v7/style.py` (`song_style`, `colors_from_cover`, `choose_environment`), `scripts/train_environment_chooser.py` (`cover_features`, `audio_features`), `src/bsmapper/v6/export.py` (`package`). Reference data: `public/data/environment-reference.json` (corpus `audio` 1182×8, `cover` 1182×17, `mapper_frequency`). Fixtures: `parity/<song>/lights/style.json`.

## Contents

1. Output (`song_style`)
2. Cover palette (k-means)
3. Colour scheme (`scheme`, `colors_from_cover`, fallbacks)
4. Cover features (17) and audio features (8)
5. Tags (percentile ranks)
6. Environment choice (22 environments, affinities, rule)
7. Info.dat fields
8. Parity and tolerances

## 1. Output

```
song_style(song_id, key, cover) = {
  environment: <name>, category: moody | energetic | pop | rock,
  colors: {saberA, saberB, env0, env1, env0boost, env1boost, obstacles, source},   each {r, g, b, a: 1.0}
  character: {tags, cover: bool, top3}
}
```

`colors = colors_from_cover(cover)` when a cover exists, else `fallback_colors(category)`.

## 2. Cover palette

`cover_palette(path, k=6, seed=0)`: the cover scaled to 64×64 (ffmpeg `scale=64:64`, bicubic), rgb24, pixels in 0..1 (4096 × 3). k-means in RGB: 6 initial centroids = 6 distinct pixels chosen at random (`default_rng(0).choice(4096, 6, replace=False)`), then 12 Lloyd iterations (assign each pixel to the nearest centroid by squared distance, centroid = mean of its pixels, an empty cluster keeps its centroid). Per cluster: `rgb` (centroid), `share` (fraction of pixels), `h, s, v` (`colorsys.rgb_to_hsv`), `vivid = share * s * min(1, v * 1.5)`. Sorted by `vivid` descending.

The browser's scaler and RNG differ; the fixture lists the trainer's clusters so the browser can check that its two chosen hues land within tolerance rather than that the clusters match.

## 3. Colour scheme

Helpers: `rgb(h, s, v)` = `hsv_to_rgb(h mod 1, min(1, s), min(1, v))` rounded to 4 decimals, `a = 1.0`. `hue_gap(a, b) = min(d, 1 - d)` with `d = |a - b| mod 1`. `warmth(h) = (cos(2π (h - 0.05)) + 1) / 2`. `MIN_HUE_GAP = 0.17`.

```
scheme(ha, sa, hb, sb, vb=1.0, ho=None, note):
  ho = (ha + hb)/2 + 0.5 if ho is None
  saberA    = rgb(ha, sa, 1.0)            saberB    = rgb(hb, sb, vb)
  env0      = rgb(ha, min(0.9, sa+0.1), 0.9)   env1 = rgb(hb, min(0.9, sb+0.1), 0.9)
  env0boost = rgb(ha+0.06, min(1, sa+0.15), 1.0)   env1boost = rgb(hb-0.06, min(1, sb+0.15), 1.0)
  obstacles = rgb(ho, 0.6, 0.85)
  source    = {left_hue: round(ha*360), left_sat: round(sa, 2), right_hue: round(hb*360), right_sat: round(sb, 2), rule: note}
```

`colors_from_cover(path)`:

```
pal = [c for c in cover_palette(path) if c.s > 0.15 and c.v > 0.15]
if pal is empty:                                  greyscale cover
  return scheme(0.08, 0.08, 0.6, 0.45, note='greyscale cover: paper and ink')
a = pal[0]
b = first c in pal[1:] with hue_gap(c.h, a.h) >= 0.17
if b is None:                                     monochrome cover
  return scheme(a.h, max(0.75, a.s), a.h, 0.12, note='monochrome cover: colour and its tint')
(ha, hb) = (a.h, b.h) if warmth(a.h) >= warmth(b.h) else (b.h, a.h)      # warmer hue on the left
other = first c in pal with hue_gap(c.h, ha) >= 0.08 and hue_gap(c.h, hb) >= 0.08
return scheme(ha, max(0.75, a.s), hb, max(0.75, b.s), ho = other.h if other else None, note='two cover hues')
```

Note `sa` always comes from `a` (the most vivid cluster) and `sb` from `b`, even when the hues were swapped for warmth.

`fallback_colors(category)` (no cover): `FALLBACK = {moody: (0.83, 0.52), energetic: (0.97, 0.52), pop: (0.92, 0.47), rock: (0.02, 0.58)}` = (ha, hb); `saberA = rgb(ha, 0.85, 1)`, `saberB = rgb(hb, 0.85, 1)`, `env0 = rgb(ha, 0.85, 0.9)`, `env1 = rgb(hb, 0.85, 0.9)`, `env0boost = rgb(ha+0.06, 0.9, 1)`, `env1boost = rgb(hb-0.06, 0.9, 1)`, `obstacles = rgb(ha+0.5, 0.7, 0.85)`, `source = {fallback: category}`.

Fixture coverage: bereal = greyscale branch, moody = monochrome branch, lights-out = two hues.

## 4. Features

`cover_features(bytes)` → 17 floats. Cover scaled to 48×48 (ffmpeg, bicubic), pixels 0..1. Per pixel: `mx = max(r,g,b)`, `mn = min`, `v = mx`, `s = (mx - mn) / max(mx, 1e-6)` (0 when `mx == 0`), hue `h` in 0..1 by the standard sextant formula (`d = max(mx - mn, 1e-6)`; `mx == r` → `((g - b)/d) mod 6`, `mx == g` → `(b - r)/d + 2`, else `(r - g)/d + 4`; divided by 6), weight `w = s * v`.

| index | value |
|---|---|
| 0..11 | hue histogram: bin `clip(floor(h*12), 0, 11)`, weighted by `w`, normalised by `max(1e-6, sum w)` |
| 12 | mean `s` |
| 13 | mean `v` |
| 14 | fraction of pixels with `s < 0.12` (grey) |
| 15 | fraction with `v < 0.2` (dark) |
| 16 | 1.0 |

No cover → 17 zeros (and `has_cover = false`).

`audio_features(key)` → 8 floats, from `data/analysis/<key>.json: tempo_bpm` and `data/analysis/<key>.npz: stem_rms (4, frames), vocal_periodicity (frames)` at 50 fps (float16 → float32):

| index | value |
|---|---|
| 0 | `tempo_bpm / 150` |
| 1 | p75 of `e = stem_rms.sum(0)` |
| 2 | `log((p90(e) + 1e-4) / (p10(e) + 1e-4))` |
| 3..6 | `stem_rms.mean(1) / sum` per stem (drums, bass, other, vocals) |
| 7 | fraction of frames with `vocal_periodicity > 0.5 and stem_rms[3] > 0.02` |

Percentiles are numpy's default (linear interpolation).

## 5. Tags

`rank(col, v) = mean(col < v)` over the reference corpus columns `A = ref.audio`, `C = ref.cover`.

```
energy    = (rank(A0, a0) + 2*rank(A1, a1) + rank(A3, a3)) / 4
energetic = energy
calm      = 1 - energy
heavy     = (rank(A3, a3) + rank(A2, a2)) / 2 * (1 - rank(A6, a6) * 0.5)
vocal     = (rank(A6, a6) + rank(A7, a7)) / 2
with a cover (c = cover_features):
  warm   = c[0] + c[1] + c[2] + c[11]            cool = c[5] + c[6] + c[7] + c[8]
  dark   = max(rank(C15, c15), 1 - rank(C13, c13))
  bright = rank(C13, c13)
  vivid  = rank(C12, c12)
  pastel = rank(C13, c13) * (1 - rank(C12, c12)) * 2
  mono   = max(c14 > 0.6, max(hist) > 0.7)       hist = c[0..11]
```

Without a cover the cover tags are absent (treated as 0 in the affinity sum).

## 6. Environment choice

`AFFINITY` (22 classic environments; all light up from basic events):

| environment | tags |
|---|---|
| BillieEnvironment | dark 1, calm 1, cool 1, mono 1, vocal 0.5 |
| HalloweenEnvironment | dark 2, warm 1, heavy 1 |
| NiceEnvironment | cool 1, pastel 1, calm 1 |
| BigMirrorEnvironment | bright 1, vivid 1, energetic 0.5 |
| KaleidoscopeEnvironment | vivid 2, pastel 0.5, dark 0.5 |
| TriangleEnvironment | vivid 1, cool 1, energetic 0.5 |
| OriginsEnvironment | warm 1, bright 1, calm 1 |
| DefaultEnvironment | (none) |
| PanicEnvironment | energetic 2, vivid 1, warm 0.5 |
| RocketEnvironment | energetic 1, warm 1, bright 1 |
| MonstercatEnvironment | energetic 2, cool 1, vivid 1 |
| CrabRaveEnvironment | energetic 1, bright 2, warm 1 |
| KDAEnvironment | vivid 1, warm 1, energetic 1, vocal 0.5 |
| BTSEnvironment | pastel 1, bright 1, vocal 2 |
| GagaEnvironment | vivid 1, pastel 1, vocal 1, bright 1 |
| FitBeatEnvironment | bright 1, pastel 1, energetic 1 |
| InterscopeEnvironment | dark 1, cool 1, vocal 1 |
| TimbalandEnvironment | dark 1, energetic 1, vocal 1 |
| SkrillexEnvironment | energetic 2, heavy 1, vivid 1, dark 0.5 |
| DragonsEnvironment | heavy 2, warm 1, dark 1 |
| GreenDayEnvironment | heavy 1, energetic 1 |
| LinkinParkEnvironment | heavy 2, dark 1, cool 1 |

```
total = sum(mapper_frequency.values())
score[e] = sum(w * tags.get(t, 0) for t, w in AFFINITY[e]) / max(1, sum(w)) + 0.05 * ln((mapper_frequency.get(e, 0) + 1) / total)
top = the 3 highest scores (descending)                           # DefaultEnvironment scores only the frequency term
pick = int(sha256(song_id utf-8).hex[:8], 16)
environment = top[0] if score[top[0]] - score[top[1]] > 0.15 else top[pick mod 3]
category = moody if dark > 0.6 or calm > 0.65
           else energetic if energetic > 0.65
           else rock if heavy > 0.6
           else pop
```

Python sorts ties by insertion order of `AFFINITY` (stable sort on `-score`).

## 7. Info.dat fields (`export.package`, version 2.1.0)

Top level: `_environmentName = style.environment`, `_environmentNames = [style.environment]`, `_allDirectionsEnvironmentName = 'GlassDesertEnvironment'`, `_colorSchemes = [{useOverride: true, colorScheme: {colorSchemeId: 'BSMapper <title>', saberAColor: saberA, saberBColor: saberB, environmentColor0: env0, environmentColor1: env1, obstaclesColor: obstacles, environmentColor0Boost: env0boost, environmentColor1Boost: env1boost}}]` (colours with `a`).

Per difficulty entry: `_beatmapColorSchemeIdx: 0`, `_environmentNameIdx: 0`, `_customData: {_colorLeft: saberA, _colorRight: saberB, _envColorLeft: env0, _envColorRight: env1, _envColorLeftBoost: env0boost, _envColorRightBoost: env1boost, _obstacleColor: obstacles}` (SongCore; `r, g, b` only, no `a`). Other entry fields (`_difficulty`, `_difficultyRank`, `_beatmapFilename`, `_noteJumpMovementSpeed`, `_noteJumpStartBeatOffset`) come from the packager.

Without a style: `_environmentName = 'DefaultEnvironment'`, no `_colorSchemes`, no `_customData`.

## 8. Parity (`parity/<song>/lights/style.json`)

Fields: `style` (full `song_style` output incl. `character`), `tags`, `cover_palette` (6 clusters with rgb/share/h/s/v/vivid), `cover_features`, `audio_features`, `colors_from_cover`, `shaping` (lights). Reference covers: `outputs/covers/<song>.jpg` in the trainer (copy them beside the fixtures when the style port is built).

Tolerances: `audio_features` within 1e-3 given the same analysis arrays; `cover_features` within 0.03 per element (different image scaler); `tags` within 0.05; each colour channel within 0.02 (hue within 0.02 of the circle, saturation/value exact formula); environment identical when `score[top0] - score[top1] > 0.15`, otherwise one of `character.top3`; category identical.
