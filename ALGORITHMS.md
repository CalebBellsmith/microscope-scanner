# The Algorithms Behind the Scanner

This document explains every image-processing technique the scanner uses — what
each one does, the exact steps it runs, *why* it works, and *why* it was chosen
over the alternatives. It is the reference behind the code, not a substitute for
it: where a constant matters, the value is given, but the source of truth is the
file named in each section.

The system has four independent algorithmic subsystems:

| Subsystem | File | Job |
|---|---|---|
| **Legacy analysis** | `analysis_pipeline.py` (`_detect_legacy`) | Reproduce the old MATLAB scratch numbers exactly, for continuity |
| **Accurate analysis** | `analysis_pipeline.py` (`_detect_accurate`) | Measure genuine scratches as correctly as the optics allow |
| **Defect-aware analysis** | `analysis_pipeline.py` (`_detect_defect_aware`) | Legacy's measurement and scale, with defects excluded from the count |
| **Focus** | `capture_pipeline.py` (`_focus_score`, `_autofocus_search`) | Decide if a frame is sharp, and drive Z to make it sharp |
| **Defect / spec classifier** | `ml_inference.py` (`_rule_predict`) | Decide if a frame has avoidable debris worth nudging away from |

Per-image runtimes for all four are in [§6 Performance](#6-performance--how-fast-each-algorithm-runs).

A recurring idea ties them together, so it is worth stating once up front.

### The one trick used everywhere: *darkness below a local background*

Scratches and debris are **darker than the material around them**, but the
material itself is not uniformly bright — lighting falls off, the substrate has
texture, PET film is grainy. A fixed brightness threshold therefore fails: what
counts as "dark" in a bright corner is "background" in a dim one.

The fix, used by accurate mode, the focus metric, and the classifier alike:

1. Estimate the local bright background with a **morphological close** using a
   large kernel (31×31). A close fills in anything darker and smaller than the
   kernel, so thin scratches and small specs vanish and what remains is the
   "what the material would look like here with nothing on it" surface.
2. Subtract the real image from that background: `darkness = close(gray) − gray`.
   Now every pixel's value is *how much darker than its own neighbourhood it is* —
   immune to global lighting, and zero on clean material regardless of absolute
   brightness.

Thresholds are then set **relative to the frame's own darkness statistics**
(`mean + k·σ`), so on a grainy PET frame the bar automatically rises above the
grain, and on clean glass it drops low enough to catch faint marks. This single
choice is what lets one algorithm span "clean A-08 glass" and "filthy wet-
abrasion PET" without a mode switch.

---

## 1. Legacy analysis — a faithful MATLAB reproduction

**Goal:** produce the *same* scratch-area numbers the lab's original MATLAB
program produced, so historical data stays comparable. Accuracy here means
"agrees with MATLAB", not "is objectively correct" — the two are different jobs,
which is why accurate mode (§2) exists separately.

### 1.1 The pipeline, step by step

Every step is a deliberate translation of a specific MATLAB operation. The
subtle part was that MATLAB and Python/NumPy disagree on small conventions, and
each disagreement had to be neutralised.

1. **Greyscale, invert, contrast-stretch** (`_to_grey_adjusted`)
   `0.2989 R + 0.5870 G + 0.1140 B` (MATLAB's `rgb2gray` weights), then
   `1 − grey` to make scratches bright, then `imadjust` with 1% saturation at
   each end. Working on bright-on-dark matches the original.

2. **Per-column peak detection** (`_build_scratch_mask`)
   For each image column, `scipy.signal.find_peaks` finds dark ridges
   (`MaxPeakWidth = 100`, `MinPeakProminence = 0.1`). Each peak is painted into
   the mask as a vertical band `loc ± round(width/2)`. This is MATLAB's
   `findpeaks` loop reproduced exactly, including:
   - **1-based indexing**: scipy's 0-based `loc` gets `+1`, then `−1` again when
     written back, so the band lands on the same rows MATLAB used.
   - **Round-half-away-from-zero** (`_matlab_round`): MATLAB's `round()` rounds
     `2.5 → 3`; NumPy rounds half-to-even (`2.5 → 2`). A dedicated helper
     enforces MATLAB's rule so band widths match to the pixel.

3. **Gradient-based horizontal-particle removal** (`_remove_horizontal_particles`)
   `H_SIZE` iterations of: take the horizontal gradient, `imadjust` it (which
   clips negatives to zero), and subtract the shifted gradient from the mask.
   This erodes horizontally-oriented particles that aren't scratches. Faithful
   to the MATLAB loop including the one-column shift and the `round()` each pass.

4. **Bridge** (`_bwmorph_bridge`)
   MATLAB's `bwmorph(bw,'bridge')` sets a background pixel to foreground if its
   8 neighbours form **two or more disconnected groups** — i.e. the pixel bridges
   a one-pixel gap in a broken line. Implemented as a **256-entry lookup table**
   over the 8-neighbour bit pattern (`_bridge_lut`), applied in a single
   vectorised pass. This is byte-identical to the per-pixel definition but
   ~1000× faster than calling a connected-components routine at every pixel.

5. **Remove small objects** (`_bwareaopen`)
   `skimage.remove_small_objects` with `min_size = MIN_AREA_PIXELS`.
   **This is the single most important correction in the whole port.** skimage
   defaults to `connectivity=1` (4-connected); MATLAB `bwareaopen` defaults to
   **8-connected**. Under 4-connectivity a diagonal chain of scratch pixels is
   seen as many tiny separate objects, each below the size floor, so it gets
   *deleted*. That silently under-measured every leg by **5–9%**. Setting
   `connectivity=2` fixed it. This was found by diffing port output against the
   MATLAB numbers leg by leg — it is a genuine bug fix, not a tuning knob.

6. **Re-binarise against the original grey**, then **region properties**
   (`regionprops`). For each region compute the **roundness metric**
   `4π·area / perimeter²` (1.0 = perfect circle, →0 = long thin line) and the
   bounding-box row-span vs column-span.
   A region is accepted as a scratch iff **`roundness < THRESHOLD` AND
   `col_span > row_span`** — i.e. it is elongated *and* wider than tall (a
   horizontal scratch). Round specs and vertical features fail one of the two.

   The perimeter is traced with `cv2.findContours(CHAIN_APPROX_NONE)` — an
   8-connected closed pixel chain — because that matches MATLAB `bwboundaries`.
   skimage's `find_contours` returns sub-pixel marching-squares boundaries whose
   perimeter (and therefore roundness) would differ, quietly changing which
   regions pass the roundness gate.

### 1.2 Why a calibration is still needed

Even with every step faithful, the port cannot be *bit-identical* to MATLAB:
`findpeaks` uses interpolation whose last digits differ, the JPEG decoder is a
different implementation, and float rounding accumulates differently. The raw
port therefore sits a few percent off MATLAB, with a small systematic bias.

Rather than chase an unreachable bit-match, a **small linear calibration** maps
the port's per-image measurements onto the MATLAB numbers (`_legacy_calibrate`).

**The features** (per image, from the accepted scratch components):
- area in long scratches (`length ≥ 100 px`) and area in short ones,
- area in big components (`≥ 400 px²`) and area in small ones,
- count of long scratches,
- `min(count, 80)` and `max(count − 80, 0)`,
- small-component area (`< 200 px²`) *when the count exceeds 80*.

**Why these features and not a single scale factor?** Because the port-vs-MATLAB
disagreement is not uniform — it depends on the *regime* of the frame:
- Splitting area by **length** and by **component size** lets the map correct
  long clean scratches differently from short broken fragments (which is exactly
  where the findpeaks/rounding differences bite hardest).
- The **`80`-count hinge** (`min(count,80)` + `max(count−80,0)`) is a piecewise
  linear break. Below ~80 detected components a frame is normal; above it the
  frame is a **swarm** of noise squiggles where the port and MATLAB diverge in a
  different way. One linear term can't fit both sides; two hinged terms can. The
  extra swarm-only small-area term cleans up the high end.

**How it was fit:** leg-weighted least squares (each leg weighted by
`1/leg_mean`, so a big leg and a small leg count equally in *percentage* terms —
which is what "within X%" accuracy actually measures), with gentle
iteratively-reweighted minimax passes to pull in the worst legs without letting
the fit diverge. Fit on the full 30-set / 3,600-image archive (20 glass + 10 PET).

**Result:**
- Glass: **80/80** leg means within 4% of MATLAB (min 96.1%, mean 98.6%).
- PET: **38/40** within 4% (min 95.6%, mean 98.0%).

### 1.3 The proof it generalises (not just curve-fitting)

The obvious worry with any calibration is that it has simply memorised these
pictures. Two checks refute that:

- **Frozen cross-substrate test.** The calibration fit on *glass only* (the
  original 20 sets) was applied, unchanged, to the 10 PET sets it had never
  seen — a different substrate with texture, diagonals, and smears. It scored
  **38/40 legs ≥ 96%** out of the box. A memorised fit could not do that.
- **Held-out splits.** Repeated random 15-set-train / 15-set-test splits keep
  the test legs in range.

The takeaway stated in the code: this is an **algorithm-to-algorithm** map
(port behaviour → MATLAB behaviour), not a set-to-set lookup. That is also why
there is **one unified calibration** and no per-substrate mode — a PET-only refit
was tried and could not beat the unified fit even on its own data, so the extra
mode would add complexity for no gain.


### 1.4 Defect-aware mode — legacy's ruler, minus the defects

A third analysis mode sits between the other two: **the legacy measurement
with defects excluded.** It runs the exact MATLAB-faithful pipeline (§1.1) and
keeps legacy's per-scratch width/length characterisation and calibrated scale,
but applies one extra acceptance gate (`_defect_gate`) after the MATLAB gates:

- **dots / specs / bubbles / blob chains** — too short (< 18 px) or too square
  (aspect < 2.6; a bubble ring and a spec both sit near 1:1);
- **smears / chunky drag marks** — thick AND squat (row-span > 12 px with
  aspect < 4). Deliberately NO comet-smear exception here: accurate mode
  counts smears by design, this mode excludes them by design;
- **non-horizontal marks** — the component's principal axis must lie within
  25° of horizontal, rejecting diagonals, verticals and steep curve segments
  that MATLAB's row-span < column-span test lets by.

**The calibration subtlety.** The linear calibration (§1.2) must see the same
population it was fitted on — its count-hinge features encode the noise-
squiggle regime, and feeding them *filtered* counts extrapolates (in testing,
a noise-only frame's calibrated area went UP 11% after filtering). So the
calibration always runs on the full MATLAB-accepted population, and the
calibrated total is then scaled by the **raw-area fraction that survived the
defect filter**. Two properties follow by construction: the number stays on
legacy's scale, and defect-aware can never read higher than legacy on the
same frame — the gap between the two columns IS the defect contamination.

Validated against legacy on the sentinel panel: byte-identical legacy output
(the refactor changed nothing), clean scratch frames within 0.7% of legacy,
and the removals landing exactly where defects live — noise-only control
−53%, diagonal-contaminated frame −21%, textured control −15%. Runtime is
legacy's (~274 ms/frame): the same pipeline plus a per-component gate.

---

## 2. Accurate analysis — measuring real scratches (v6)

**Goal:** ignore what MATLAB would have said and measure the scratches that are
actually there, to their true extent, while refusing to count anything that
isn't a scratch (specs, dust chains, grey halos, diagonal rig artefacts,
substrate texture). This is the mode used when *correctness* matters more than
*continuity*.

The design principle is: **confirm conservatively, then measure generously, then
subtract contaminants.** Five stages.

### 2.1 Stage 1 — strong pass: confirm scratch *cores*

Threshold the darkness image high (`mean + 1.0σ`, floor 8) → `strong` mask. Open
it with a **horizontal** line element (15×1) to get `seed`: only pixels that sit
in a genuinely horizontal run survive.

Each connected component of `strong` is accepted as a **core** only if it looks
like a line, not a blob:
- area ≥ 30 and width ≥ 18 px (big enough, long enough),
- overlaps a horizontal seed run (it has a straight spine),
- aspect (w/h) ≥ 2.6 (wider than tall),
- if it is thick (`h > 12`) *and* not very elongated (`aspect < 4`), it must
  still contain a **long horizontal seed run** (≥ 40 px). This is the
  **comet-smear exception**: dense abrasion gouges are thick, but unlike round
  smudges they contain long straight streaks, so they survive while blobs don't.

These cores are the pixels we are *sure* are scratch. They are used only to
confirm — their extent is measured in the next stage.

**Why start conservative?** Because the next stage reconstructs full extent from
whatever a *weak* threshold connects to a core. If the cores themselves were
loose, that reconstruction would flood into noise. Precision first, recall second.

### 2.2 Stage 2 — hysteresis: measure to full faint extent

This is the fix for the original **undershoot** problem — the detector "picking
up half a scratch, or a handful of scratches in an area but not all of them."

Threshold the *same* darkness image low (`mean + 0.45σ`, floor 5) → `weak` mask.
A real scratch fades at its ends and has gaps, but those faint tails are
physically **connected** to its dark core. So: take each weak component, and
**accept it if it contains any confirmed-core pixel**. The scratch is then
measured end-to-end from its faint start to its faint finish, instead of only
its darkest fragments.

This is classic **hysteresis thresholding** (the same idea as the Canny edge
detector's two thresholds): a high threshold decides *what is real*, a low
threshold decides *how far it extends*. Using two thresholds beats any single
one — a single high threshold undershoots the faint parts; a single low
threshold floods into grain. Splitting the two jobs gets both right.

**Grain-flood guard.** If a weak component balloons into a blob (`h > 40` and
`aspect < 3`) — meaning the low threshold merged the scratch with surrounding
junk — it **falls back to just the confirmed core** rather than being dropped.
A scratch is never lost; at worst it is measured conservatively.

### 2.3 Stage 3 — faint-streak pass: catch whole faint lines

Hysteresis only extends scratches that already have a strong core. A scratch
that is faint along its **entire** length has no core to anchor to, so it would
be missed. A dedicated matched filter finds these.

**The matched filter.** Convolve the darkness image with a horizontal running
mean (a `1×L` box blur). A faint horizontal line is consistent along its length,
so averaging *L* pixels along it adds its signal coherently while averaging out
the random grain around it — the line's signal-to-noise ratio improves by
roughly `√L`. Dust dots and diagonals, being short in the horizontal direction,
get diluted instead of boosted. This is *why* a running mean specifically finds
faint horizontal lines and nothing else.

Two scales are run:
- **`L = 25`, standard route.** Components ≥ 45 px long, ≤ 10 px thin, aspect
  ≥ 6, that clear a **texture-adaptive darkness gate** `max(22, mean + 1.5σ)`.
  The gate is the discriminator that keeps chance dust-dot chains out: a real
  faint line is *consistently* darker than grain along its whole run (high
  median per-column darkness), whereas a random chain of dust dots only spikes
  at the dots (low median). Median, not mean, is what makes this robust.
- **`L = 51`, long-thin route.** A second, longer filter that doubles the SNR
  boost for the very faintest streaks. Here the acceptance rule changes: a
  **continuous run ≥ 120 px that stays ≤ 10 px thin (aspect ≥ 12)** cannot be
  chance texture — noise chains top out near 100 px — so **length itself is the
  evidence**, and the darkness bar softens to `max(25, mean + 1.0σ)`.

**Why the long-thin route was added (the dirty-frame fix).** On heavily
scratched frames, the scratches themselves inflate the frame's σ, which pushes
the standard adaptive gate up to ~65 — and real faint lines measuring 48–64 fell
just under it (one 304-px line missed by under 2 points). Length-as-evidence
sidesteps the σ-inflation trap: you don't need a frame-relative darkness argument
for something that is provably too long and too straight to be noise.

- **Ultra-faint route (third pass).** The `L = 51` filter is re-run at a *lower*
  pixel threshold (`mean + 0.6σ`), accepting only runs **≥ 150 px long, ≤ 8 px
  thin, aspect ≥ 18**, at a softer darkness gate `max(15, mean + 0.6σ)`. This
  recovers the very faintest full-length scratches on badly abraded frames.
  **The 150 px floor is a measured boundary, not a guess:** at this faintness,
  candidate darkness on real scratches (medians 40–54) overlaps chance texture
  ridges on *control-sample* PET (36–42) almost completely — darkness cannot
  separate them — but the longest chance texture ridge measured on control PET
  is **~127 px**. Length is therefore the only honest discriminator. Shorter
  faint wisps are deliberately **not** counted: adding them was tested and put
  ~10 false scratches on a control frame, and a false scratch on a control
  corrupts the control-vs-treatment comparison the instrument exists to make.
  This is a knowingly accepted floor (≲ 1% of frame area on the worst frames),
  not an oversight.

### 2.4 Stage 4 — remove dust specs that touch scratches

A spec sitting near or on a scratch is the hardest contaminant, because the weak
mask bridges it into the scratch and the hysteresis merge would count its area
as scratch area. Two mechanisms handle the two ways this happens.

**4a — Excision (spec *touching* a scratch).** Before the hysteresis merge, find
classic dust dots in the strong mask — compact (30–400 px², ≤ 25 px each way),
roundish (aspect < 2.2), solid (filled ≥ 45%) — and cut them out of the weak
mask so they detach from the line.

The test for "is this a dust dot or part of a scratch" is **horizontal context,
not shape alone.** For each candidate, look at its own rows a little to the left
and a little to the right (skipping the immediate halo). A dash *inside* a
stippled scratch, or a dot the line passes straight through, has dark structure
on **both flanks** and is left in place. An isolated spec — even one hanging off
a line's side — has **empty flanks** and is excised. This is what protects
stippled/wiper scratches (which *are* chains of dots) from being erased: the
distinction is not "is it a dot" but "does a line run through it."

**4b — Bulge-shave (spec *fused* with a scratch).** A spec dark enough to merge
with the line in the *strong* mask can't be separated as a component — it shows
up as a short fat bulge on an otherwise thin line. Walk each thin line's
thickness profile; where thickness spikes past `max(2.5×, +8 px)` its median
over a short span (≤ 34 columns) **with the line continuing ≥ 15 columns on both
sides**, shave the bulge back to the band spanned by the flanking columns'
*median* row-extent.

Guards keep this from ever cutting a real scratch:
- it only runs on genuinely thin lines (median thickness ≤ 15 px; thick smear
  bodies are exempt),
- the shaved region must be **one compact roundish blob** (a single
  8-connected component, plausible dot dimensions, ≥ 35% filled),
- **no column may be fully severed** — every bulge column must keep band pixels,
  so the line can never be cut in two.

Comet-smear heads escape naturally: they sit at the *end* of their scratch (no
line continuing on both sides) or are part of a thick body (median > 15).


### 2.5 Stage 4c — halo trim: count the line, not its blur skirt

Operator review flagged measured areas as a hair generous, and 5× zoom
confirmed it: a **dark** line carries a wide optical/JPEG blur skirt, and the
absolute weak threshold wades into it — on soft-optics PET frames the counted
mask ran well beyond the visible line body (total counted area was ~3× the
strict half-max (FWHM) line width).

The fix is **relative, per scratch, per column**: drop pixels fainter than a
**quarter of that column's own peak darkness**. Why quarter-max: half-max
(FWHM) is the strict metrology definition but visibly clips real line body on
these images; the full skirt extends to ~5% of peak; quarter-max sits at the
visible edge — verified at 5× zoom on glass swarm, faint PET lines, dirty
PET, and a deep-black glass line.

Three properties make the trim safe:
- **Faint lines are untouched** — their peak is near the threshold already, so
  the quarter-max floor sits below it. The faint-extent recovery (hysteresis,
  faint routes) cannot be undone by the trim.
- **No scratch loses length** — every column keeps its peak pixel, so the
  end-to-end extent is preserved; only thickness is tightened.
- **Severed halo pads are dropped by evidence** — trimming can separate a pad
  of halo that connected two structures; a final gate keeps a component only
  if it carries its own detection evidence (core or faint-route pixels) or is
  shaped like a line. Pads have neither and vanish.

Corpus impact: −15% mean leg area (−6 to −23%), counts stable — a pure
thickness correction, not a sensitivity change.


### 2.6 Stage 4d — horizontal-only: reject diagonals, verticals and curves

Only **horizontal** marks are abrasion on this rig; diagonal and curved marks
are handling damage. The per-component gates (aspect, seed) reject a diagonal
that stands alone — but a diagonal **attached to or crossing** a real
horizontal scratch rides into the count as part of that component. Operator
review caught exactly this (a curved handling scratch outlined on a PET frame).

Three layers remove non-horizontal structure without touching real lines:

1. **Pixel filter**: keep only pixels belonging to horizontal runs — close
   (11×1) first so stippled dashes fuse into their line, open (15×1) so only
   ≥ 15 px runs survive, then a small dilation so the trimmed line keeps its
   edges. Geometry does the work: a genuine scratch at a slight slope still
   forms long row-runs (a 4 px line at 5° has ~45 px runs); anything steeper
   than ~15° falls apart into short row-segments and is erased. Comet-smear
   heads are wide on every row, so they are untouched.
2. **Orientation gate**: small final pieces (< 400 px) must actually lie
   horizontal — principal-axis angle ≤ 25°. Remnants of curves tilt 30–70°.
3. **Squat/loop gate**: (< 150 px & aspect < 2.2) or (< 400 px & aspect < 1.8)
   drops loops and pads that have no meaningful axis at all.

Verified on the hardest case (an S-shaped curve crossing two counted lines):
body, fragments and loop all rejected, the crossing lines stay fully traced.
Known residue, accepted deliberately: the ~100 px knot where a curve crosses
a counted line (inseparable without damaging the line) and locally-horizontal
micro-segments of a curve (indistinguishable from a real tiny scratch without
global curve tracing) — < 0.5 % of frame area on the worst frame.

### 2.7 Stage 5 — recount

Union all accepted pixels and re-label with 8-connectivity, so touching pieces
merge into single scratches. Components below 30 px (crumbs left by the shave)
are dropped — every upstream gate already required ≥ 30 px, so nothing real is
lost. Each surviving component is one scratch; its pixel count is its area.

### 2.8 How it was validated

A fixed panel of 12 deliberately hard "sentinel" frames (dense swarm, noise-only,
sparse-faint, heavy-dust, stippled wiper, textured PET, diagonals+smears, comet
smears, a failing leg) is re-scored after every change and the overlays are
inspected by eye. The bar each change must clear:
- noise-only frame stays ≈ 0 (no false scratches),
- stippled/wiper scratches stay counted,
- diagonals stay rejected, comet smears stay counted,
- counts stay stable (a fix should change *extent*, not invent objects).

Across the full 120-leg corpus the v6 spec/faint fixes moved area **+0.8% on
glass, +1.0% on PET** on average, no leg by more than ±2.4%, with counts flat —
i.e. surgical corrections, not a recalibration.

---

## 3. Focus — deciding and achieving sharpness

**Goal:** two things. A **metric** that scores how sharp a frame is (used to
decide whether to skip, re-shoot, or exclude a frame), and a **search** that
drives the Z stage to bring a soft frame into focus.

The guiding principle, learned the hard way: **focus is objective.** A frame is
sharp or it isn't, independent of the slide — unlike defect dirtiness, which is
genuinely slide-dependent. So the focus *threshold* is a fixed dial, and the
part that adapts per-slide is elsewhere (§5).

### 3.1 The focus metric (`_focus_score`)

Higher = sharper. It measures the **steepness of edges** on the things that
should have crisp edges, normalised so that *how many* or *how dark* those things
are doesn't matter — only how sharp.

**Tier 1 — scratches.** When the frame has a substantial, genuinely dark
horizontal-scratch mask (≥ 1500 mask px, contrast ≥ 28), compute the
**vertical-gradient energy** on the scratch pixels (a Sobel `∂/∂y`, squared,
averaged) divided by the scratch darkness: **`E / c`**.

Why this form? A defocused line keeps its darkness `c` but loses edge steepness
`E` (blur spreads the edge over more pixels, lowering the gradient), so `E/c`
drops hard as the frame softens. And empirically, across sharp frames `E` grows
roughly *linearly* with darkness `c`, so `E/c` stays **flat** for sharp frames
regardless of how deep the scratches are. An earlier `E/c²` version
over-punished dark scratches (a genuinely sharp heavy frame scored as "soft"),
which is why the exponent is 1, established by measuring `E` vs `c` on sharp
frames from contrast 33 to 83.

**Tier 2 — dust specs (fallback).** When scratch evidence is weak — few mask
pixels, or low contrast meaning the "mask" is really JPEG-grain phantoms — judge
the **dust specs** instead. They exist on every frame, sit on the same slide
plane as the scratches, and blur out at exactly the same time. Same `E/c` form on
the spec pixels (using both gradient directions since specs aren't oriented),
scaled by a constant so the spec tier's numbers line up with the scratch tier's
range and **one threshold serves both**.

**Blank field → `+∞`** (treated as in-focus): if there are neither scratches nor
specs to judge, there is nothing to be blurry. This is a known blind spot (a
truly blank, truly blurry field would pass); it never occurred in 3,600 archive
frames but is documented as a caveat.

**Calibration.** In-focus frames read ≈ 4000–8000 in both tiers; soft frames
≲ 2100. The default threshold is **3000**, bench-confirmed — it sits in the wide
gap between the two bands. (Independent corroboration: on sharp glass the tour
in §5 suggests almost exactly 3000, i.e. half the sharp-frame median.)

### 3.2 The autofocus search (`_autofocus_search`)

The Z axis is a **continuous roller** — no hard travel limit — so the search is
bounded only by a runaway guard (default 10,000 half-steps), and the *peak*, not
the bound, is what normally stops it.

**Hill-climb with learned direction.** We don't know a-priori which way is "into
focus," so from the current height we probe **one step each way**. Whichever
improves the focus score sets the climb direction; then we keep stepping that way
until a step **stops improving** — meaning we just passed the peak. The winning
direction is **remembered** (`self._z_dir`) and tried *first* on the next field,
because adjacent fields on a slide focus at similar heights — so after the first
field, the search usually gets the direction right on probe one and saves a move.

This is why the old "Invert Z direction" checkbox was **removed**: the search
discovers direction by itself and adapts as it goes, so a manual polarity switch
was not just unnecessary but misleading (it could fight the learned direction).

**The key insight — a verified peak *is* focus, regardless of the number.**
This is the correction that made the metric dependable on dirty PET. When both
directions make the image *worse*, the field is at the sharpest it can physically
be — full stop. Grainy, low-contrast substrates simply score lower across the
board (a sharp dirty-PET frame might peak at 2700, under the 3000 threshold), and
excluding their sharp frames would throw away good data. So:

- The threshold decides only **when to search** (and when to escalate).
- The **peak verdict** decides whether the frame is as-good-as-it-gets.
- A frame is tagged soft (`NNN_soft.jpg`, excluded from analysis) **only if it is
  below threshold AND the search could not verify a peak** — i.e. the climb was
  cut short by the runaway bound, a stop, or dropped frames. That is the genuine
  "something is wrong with the optics" case.

**Escalation.** If the best score is below threshold, the search retries once
with **3× wider probes** over the full roller range. This does two jobs at once:
it reaches a focus peak that lies far from the start, and it **double-checks a
low peak** — if even ±3-step moves can't beat the spot, the peak is real, not
single-step score noise.

Simulated across sharp-peak, low-score-peak (accepted), off-start peak,
unreachable-peak (tagged soft), and flat-field (accepted) cases.

---

## 4. Defect / spec classifier — should we nudge?

**Goal:** during capture, decide whether a frame contains **avoidable** debris
(dust, fibre, smudge) worth nudging the stage away from — *without* misfiring on
the sample's own scratches or on substrate texture that a nudge can't escape.
This runs live per frame (`_rule_predict` in `ml_inference.py`), rules-only, no
neural net needed.

**Framing:** a frame is **good unless a check positively finds a defect.** Absence
of features means a clean slide, which is good. This asymmetry matters — we never
want to reject a clean or lightly-scratched frame.

### 4.1 Check 1 — blob / fibre detection (shape-based)

Work on darkness below the local background (§ the shared trick), thresholded two
ways and OR'd:
- **local** `> max(15, mean + 2.5σ)` — dark vs the local neighbourhood; catches
  thin fibres and edges under uneven light, with the bar riding above substrate
  grain automatically;
- **global** `gray < mean − 1.5σ` — absolutely dark. This is needed because a
  solid object *wider than the 31×31 close kernel* becomes its own "background"
  (local darkness reads ~0 in its interior), so big debris is only visible to an
  absolute threshold.

Then three passes hunt genuine debris while sparing scratch structure:

- **Pass A — curved fibres.** Judge each component by what fraction of it lies in
  horizontal **scratch runs** (a 15×1 opening). Scratches — however dense, even
  fused into 2-D clusters — are made of long runs (coverage ≈ 1). A curved fibre
  only touches a run where its tangent happens to be horizontal (coverage ≲ 0.3),
  so it fails the ≥ 0.5 coverage test and is flagged. Judging *coverage* (rather
  than erasing runs) keeps the fibre whole so its shape can be measured. A small
  9×1 close first fuses **stippled** scratches (dashed wiper marks) into runs so
  they read as scratches — this was the fix for stippled lines being flagged as
  fibres.
- **Pass B — solid blobs.** A 7×7 ellipse opening erases thin lines/fibres in any
  direction but keeps a compact chunk's core. A remnant is debris unless it is
  elongated (rotated-rect elongation ≥ 8 → a line at any angle) or its **structure
  continues past its ends** (`_line_continues`: dark rows persist 40 px to either
  side → it's a thick scratch *segment*, not a blob that happens to sit on a line).
- **Pass C — big debris riding on a scratch.** A hard 15×15 erosion with no
  re-dilation kills any line ≤ 14 px thin outright, leaving only a fat chunk's
  core standing wherever it sits — catching debris that Pass B's opening
  reconnected into the line.

Every candidate must also pass a **darkness gate** (interior ≥ 20–30% darker than
background, sensitivity-scaled). This exists specifically to ignore **grey focus
halos**, which measure only 17–21% darker — below the gate — while real debris is
30–63% darker.

### 4.2 Check 2 — FFT residual, localised

A complementary, thickness-agnostic check. Take the 2-D FFT and **zero the
low-`kx` band** (horizontal frequency content). Horizontal scratches — any
thickness — put nearly all their energy at `kx ≈ 0`, so after stripping it the
residual is near zero. A blob or watermark spreads energy across all `kx`, so its
residual survives. The **residual/original std ratio** is the signal.

**Localisation guard.** Uniform substrate texture (PET) *also* leaves a big global
residual — but a stage nudge can't escape texture that covers the whole frame, so
flagging it would be pointless. So the residual must additionally be **spatially
concentrated**: split into an 8×8 block grid, and require `peak/median block-std
≥ 3`. Texture scores ~1–2 (energy everywhere); a real localised defect ≳ 3. Only
a **diffuse-but-concentrated** signal is called a defect.

### 4.3 The decision

`bad` if a blob/fibre is found, **or** the FFT residual is both high *and*
localised. Otherwise `good`. The two checks are complementary: Check 1 catches
compact/curved dark objects by shape; Check 2 catches diffuse concentrated
anomalies (faint watermarks, smudges) that have no crisp contour.

**Why rules and not a neural net?** The physical invariants here are simple and
measurable (aspect, darkness %, horizontal-energy fraction), and a rule states
them transparently and runs instantly with no model file, no training drift, and
a clear reason for every decision. Across the corpus this rebuilt classifier cut
false-flag rates from 23–45% to 3–6% on glass and from ~90% to ~50% on genuinely
filthy PET — where the remaining flags are, on inspection, real debris.

---

## 5. Auto-calibrate — adapting to a slide in two moves

**Goal:** with one button, prepare the scanner for the slide in front of it. The
key design decision follows directly from §3: **focus is objective, dirtiness is
slide-dependent**, so calibration adapts the *defect sensitivity* but never the
focus threshold.

Two moves, in order (`_on_auto_calibrate`, `_autofocus_here`):

1. **Focus the frame first.** Run the same probe-and-climb Z search as capture
   (§3.2) at the current spot and leave the stage at the sharp height. This is
   not about *setting* a threshold — it's about giving the next step a sharp
   image to judge. If there's no motor or Z doesn't respond, it skips gracefully.
2. **Tour for sensitivity.** Visit a small ring of 9 nearby spots, grabbing one
   frame each (so a single funky spot can't skew the result), and pick the
   **strictest scratch-vs-background sensitivity at which this slide still reads
   clean.** Because step 1 made the frames sharp, this sensitivity sweep is
   judging real structure, not blur. The tour always returns to the start.

The focus threshold stays put at 3000. The status bar reports both outcomes, e.g.
`sensitivity 0.61, focused (score 6420)`.

---

## 6. Performance — how fast each algorithm runs

All four algorithms process a single **822 × 1024** frame. The numbers below are
median wall-clock times (steady-state, after warm-up) measured on an **Apple M1**
(Python 3.13, OpenCV 4.12, NumPy 2.2), across four representative frames — clean
glass, dense-swarm glass, textured PET, and a failing PET leg — to show how
runtime varies with frame content.

| Algorithm | Median / frame | Range across frames | ≈ frames/sec | Cost scaling |
|---|---:|---:|---:|---|
| **Focus detect** (`_focus_score`) | **≈ 32 ms** | 28–36 ms | ~31 | O(pixels) |
| **Spec detect** (`_rule_predict`) | **≈ 50 ms** | 48–52 ms | ~20 | O(pixels) |
| **Accurate analysis** (`_detect_accurate`) | **≈ 70 ms** | 52–76 ms | ~14 | O(pixels) + O(components) |
| **Legacy analysis** (`_detect_legacy`) | **≈ 274 ms** | 225–362 ms | ~4 | O(columns × peaks) |

Per-frame detail (median ms):

| Frame | Focus | Spec | Accurate | Legacy |
|---|---:|---:|---:|---:|
| glass clean | 36 | 50 | 47 | 362 |
| glass swarm | 31 | 49 | 68 | 225 |
| PET textured | 28 | 52 | 65 | 262 |
| PET failing | 33 | 48 | 71 | 247 |

(Accurate-mode times include the v6.1 ultra-faint scan and the v6.2 halo trim, ≈ +13 ms together.)

### What these numbers mean in practice

- **The capture-time algorithms are effectively free.** During a scan, each saved
  frame runs focus detect (~32 ms) and spec detect (~50 ms) — together ~80 ms,
  which is small next to the stage settle time (0.5 s) and camera exposure. An
  autofocus *search* costs one `_focus_score` per probe (~32 ms each), so even a
  10-probe hill-climb adds only ~0.3 s, and it only fires on the rare soft frame.
  Neither is a bottleneck; the scan is dominated by mechanical motion, not
  computation.

- **Accurate mode is fast enough for interactive analysis.** ~70 ms/frame means a
  full 30-frame leg analyses in under 2 seconds, and the whole 3,600-frame
  archive in ~4 minutes single-threaded (well under a minute across 6 cores).

- **Legacy mode is the outlier, by design.** At ~274 ms it is ~4× slower than
  accurate mode. The cost is almost entirely the **per-column peak detection**:
  it calls `scipy.signal.find_peaks` **1,024 times per image** (once per column),
  a sequential Python loop. That is inherent to faithfully reproducing the MATLAB
  `findpeaks` loop — the whole point of legacy mode is bit-level fidelity to the
  original, not speed, so the loop was kept literal rather than vectorised. It is
  still only a quarter-second per frame, so a 30-frame leg finishes in ~8 s — fine
  for its occasional "reconcile against historical numbers" use.

### Why the content-dependence differs by algorithm

- **Focus and spec detect are content-flat** (~±10%): they run a fixed set of
  whole-image operations (morphology, Sobel/FFT, one components pass) whose cost
  depends on pixel count, not on how many scratches are present.
- **Accurate mode rises with clutter** (47 ms clean → 71 ms failing-PET): its
  later stages iterate over *connected components*, and dirty/heavily-scratched
  frames have far more of them. Still bounded and modest.
- **Legacy mode is *fastest* on the busiest frames** (362 ms clean → 225 ms
  swarm), which is counterintuitive until you see why: `find_peaks` returns early
  on columns with strong clear peaks, whereas on a near-blank clean column it
  works harder scanning for prominence that isn't there. The cost tracks the
  peak-finding loop, not the eventual scratch count.

> Absolute times scale with CPU and image resolution; treat them as *relative*
> guidance. The ratios (legacy ≈ 4× accurate; focus/spec cheap) hold regardless
> of machine. To re-measure on the target hardware, time each entry point
> (`_detect_legacy`, `_detect_accurate`, `_rule_predict`, `_focus_score`) on a
> handful of real frames after one warm-up call.

---

## Design themes, in one place

- **Relative, not absolute.** Every "dark" decision is darkness-below-local-
  background with a frame-relative threshold. This is the one thing that makes a
  single set of algorithms work from clean glass to filthy PET.
- **Confirm, then extend.** Accurate mode and the focus search both use a strict
  test to decide *what is real* and a loose test to decide *how far it goes* —
  two thresholds beat one.
- **Objective vs slide-dependent.** Focus is objective (fixed threshold, verified
  by a peak); dirtiness is slide-dependent (adapted by calibration). Conflating
  them was the bug; separating them was the fix.
- **A peak is truth.** The most important focus insight: when both directions are
  worse, the frame is as sharp as physics allows — the absolute number is
  irrelevant. Grainy substrates score low and that's fine.
- **Fail safe, never silent.** A scratch is never dropped (grain-flood falls back
  to the core; the bulge-shave can't sever a line); a soft frame is kept but
  *tagged*, not discarded; a missing Z axis degrades gracefully instead of
  aborting a scan.
- **Validate by eye and by corpus.** Every change is checked on a fixed panel of
  hard frames *and* re-scored across all 120 legs, so a local fix can't cause a
  global regression unnoticed.
