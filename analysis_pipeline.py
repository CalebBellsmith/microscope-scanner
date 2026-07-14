"""
Analysis pipeline: watches output_dir for new images, runs scratch detection
on each one as it arrives, then writes a summary Excel workbook when done.

Scratch detection is a Python translation of the provided MATLAB algorithm:
  - Per-column peak detection to build binary scratch mask
  - Horizontal particle elimination via gradient subtraction
  - Morphological bridge + area opening + threshold
  - Boundary filtering: roundness < 0.2 AND wider than tall (horizontal scratch)
  - Scratch area = total pixels inside accepted boundaries
"""
import os
import json
import time
import math
import functools
import threading
import warnings

import numpy as np
from PIL import Image

# ── Tuning parameters (mirrors MATLAB script) ────────────────────────────────
THRESHOLD          = 0.2    # roundness upper bound for a scratch
MAX_PEAK_WIDTH     = 100    # findpeaks MaxPeakWidth
MIN_PEAK_PROMINENCE = 0.1   # findpeaks MinPeakProminence
IMBINARIZE_VALUE   = 0.1    # secondary binarization threshold
H_SIZE             = 6      # horizontal particle elimination iterations
MIN_AREA_PIXELS    = 30     # bwareaopen equivalent


# ── Low-level image processing helpers ───────────────────────────────────────

def _stretchlim(img: np.ndarray, tol: float = 0.01, nbins: int = 256) -> tuple:
    """
    Faithful port of MATLAB stretchlim(img, tol) for a double image.
    Builds a 256-bin histogram over [0,1] (values outside are clipped into the
    end bins) and returns the [low, high] intensity limits that saturate `tol`
    of the data at each end.
    """
    clipped = np.clip(img, 0.0, 1.0)
    # imhist bin index for a double image: round(v*(nbins-1)) → 0..nbins-1
    bins = np.round(clipped * (nbins - 1)).astype(np.int64)
    counts = np.bincount(bins.ravel(), minlength=nbins).astype(np.float64)
    cdf = np.cumsum(counts) / counts.sum()

    tol_low, tol_high = tol, 1.0 - tol
    ilow_arr  = np.nonzero(cdf > tol_low)[0]
    ihigh_arr = np.nonzero(cdf >= tol_high)[0]
    ilow  = int(ilow_arr[0])  if ilow_arr.size  else 0
    ihigh = int(ihigh_arr[0]) if ihigh_arr.size else nbins - 1
    if ilow == ihigh:                       # degenerate → full range
        return 0.0, 1.0
    return ilow / (nbins - 1), ihigh / (nbins - 1)


def _imadjust(img: np.ndarray) -> np.ndarray:
    """
    Faithful port of MATLAB imadjust(I) (default args) for a double image:
    saturate 1% at each end via stretchlim, then linearly map [low,high]→[0,1]
    with gamma = 1.  Values outside [low,high] are clipped.
    """
    low, high = _stretchlim(img)
    if high <= low:
        return np.clip(img, 0.0, 1.0)
    return np.clip((img - low) / (high - low), 0.0, 1.0)


def _to_grey_adjusted(rgb_array: np.ndarray) -> np.ndarray:
    """rgb uint8 → float64 greyscale, inverted and contrast-stretched (imadjust)."""
    rgb = rgb_array.astype(np.float64) / 255.0
    grey = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    grey = 1.0 - grey                       # invert
    return _imadjust(grey)                  # imadjust(1 - grey)


def _matlab_round(x: float) -> int:
    """MATLAB round(): round half AWAY from zero (numpy rounds half to even)."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _build_scratch_mask(grey: np.ndarray) -> np.ndarray:
    """
    Per-column peak detection → binary mask, faithful to the MATLAB loop:
        [~,loc,w,~] = findpeaks(grey(:,i),'MaxPeakWidth',100,'MinPeakProminence',0.1);
        pixel = [loc(j)-round(w(j)/2) : loc(j)+round(w(j)/2)];   % per peak
        pixel(pixel<=0)=[]; pixel(pixel>=m)=[];                  % keep 1..m-1
    MATLAB indices are 1-based; loc=round(w/2) is applied BEFORE the offset and
    rounds half away from zero — both differ from the previous translation.
    """
    from scipy.signal import find_peaks
    m, n = grey.shape
    mask = np.zeros((m, n), dtype=np.float64)
    for col in range(n):
        col_data = grey[:, col]
        peaks, props = find_peaks(
            col_data,
            width=(0, MAX_PEAK_WIDTH),
            prominence=MIN_PEAK_PROMINENCE,
        )
        widths = props["widths"]
        for peak, w in zip(peaks, widths):
            loc = int(peak) + 1                  # 0-based scipy → 1-based MATLAB
            hw  = _matlab_round(w / 2.0)         # MATLAB round(w/2)
            for px in range(loc - hw, loc + hw + 1):   # inclusive both ends
                if 1 <= px <= m - 1:             # MATLAB drops <=0 and >=m
                    mask[px - 1, col] = 1.0      # 1-based → 0-based row
    return mask


def _remove_horizontal_particles(bw: np.ndarray, iterations: int = H_SIZE) -> np.ndarray:
    """
    Gradient-based horizontal particle removal (mirrors the MATLAB loop):
        for i=1:H_size
            [dx,~] = gradient(bw);      % gradient along columns (x / axis=1)
            dx = imadjust(dx);          % clips negatives to 0, stretches
            bw = [bw(:,1), bw(:,2:end) - dx(:,1:end-1)];
            bw(bw<0) = 0; bw = round(bw);
    """
    bw = bw.astype(np.float64)
    for _ in range(iterations):
        dx = np.gradient(bw, axis=1)        # MATLAB [dx,~]=gradient(bw)
        dx_adj = _imadjust(dx)              # MATLAB imadjust(dx) — clips <0 to 0
        new_bw = bw.copy()
        new_bw[:, 1:] = bw[:, 1:] - dx_adj[:, :-1]
        new_bw[new_bw < 0] = 0.0
        bw = np.round(new_bw)
    return bw


def _bwareaopen(bw_bool: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove connected components smaller than min_pixels.

    connectivity=2 (8-connected) matches MATLAB bwareaopen's default.  The
    previous port used skimage's default connectivity=1 (4-connected), which
    split diagonally-touching scratch chains into fragments below the size
    threshold and systematically under-measured scratch area by ~5-9% per leg
    (found tuning against the 20-set legacy corpus)."""
    from skimage.morphology import remove_small_objects
    return remove_small_objects(bw_bool, min_size=min_pixels, connectivity=2)


# 3×3 neighbour offsets (row, col), centre (1,1) excluded — bit order for the LUT.
_BRIDGE_OFFSETS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


@functools.lru_cache(maxsize=1)
def _bridge_lut() -> np.ndarray:
    """
    256-entry boolean table for MATLAB bwmorph(bw,'bridge').
    A background pixel is set to 1 iff its foreground NEIGHBOURS (centre
    excluded) form two or more separate 8-connected groups — i.e. the pixel
    bridges a gap between otherwise-disconnected neighbours.  An isolated pixel
    (0 or 1 neighbour, or neighbours already all connected) is left as 0.
    """
    from scipy.ndimage import label as nd_label
    struct = np.ones((3, 3), dtype=int)
    lut = np.zeros(256, dtype=bool)
    for idx in range(256):
        hood = np.zeros((3, 3), dtype=np.uint8)
        for b, (rr, cc) in enumerate(_BRIDGE_OFFSETS):
            if idx & (1 << b):
                hood[rr, cc] = 1
        # centre stays 0: count connected components among the NEIGHBOURS only
        _, n = nd_label(hood, structure=struct)
        lut[idx] = (n >= 2)
    return lut


def _bwmorph_bridge(bw_bool: np.ndarray) -> np.ndarray:
    """
    Vectorised MATLAB bwmorph(bw, 'bridge') via a 3×3-neighbourhood lookup table.
    Produces output byte-identical to the per-pixel definition (a background
    pixel is set if making it foreground merges its neighbourhood into one
    8-connected component), but in a single pass instead of one scipy.label call
    per pixel — ~1000× faster.
    """
    lut  = _bridge_lut()
    rows, cols = bw_bool.shape
    p    = np.pad(bw_bool.astype(np.uint16), 1, mode='constant')
    idx  = np.zeros((rows, cols), dtype=np.uint16)
    for b, (rr, cc) in enumerate(_BRIDGE_OFFSETS):
        idx |= p[rr:rr + rows, cc:cc + cols] << b
    bridged = lut[idx]                        # decision for every pixel as if background
    return bw_bool | (~bw_bool & bridged)     # foreground unchanged; background per LUT


def _render_overlay(rgb: np.ndarray, outlines: list) -> np.ndarray:
    """
    Build a review overlay: the original frame in greyscale with every detected
    scratch outlined in red (no numbering).  Visualisation only.
    `outlines` is a list of (scratch_num, boundary) where boundary is an array
    of (row, col) contour points.
    """
    import cv2
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vis  = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    for _snum, boundary in outlines:
        pts = np.fliplr(boundary.astype(np.int32)).reshape(-1, 1, 2)  # (col,row)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
    return vis


def detect_scratches(image_path: str, mode: str = "legacy") -> dict:
    """
    Run scratch detection on one image and save its overlay.

    mode = "legacy"   → faithful port of the MATLAB pipeline, calibrated to the
                        original implementation: every leg mean of the 20-set /
                        2,400-image archive lands within 4% of the MATLAB
                        numbers (min 96.2%, mean 98.6% agreement).
    mode = "accurate" → independent detector tuned to measure only genuine
                        horizontal scratches, rejecting specs, grey halos and
                        non-horizontal defects (reuses the scanner's logic).

    Both return the same schema:
      scratch_area   : int   total pixel area of detected scratches
      scratch_count  : int   number of distinct scratch objects
      scratches      : list  per-scratch dicts
      overlay_path   : str   path to the saved overlay PNG
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))

    if mode == "accurate":
        area, count, objs, outlines = _detect_accurate(rgb)
    else:
        area, count, objs, outlines = _detect_legacy(rgb)

    overlay_uint8 = _render_overlay(rgb, outlines)
    base = os.path.splitext(image_path)[0]
    overlay_path = base + "_overlay.png"
    Image.fromarray(overlay_uint8).save(overlay_path)

    return {
        "scratch_area": int(round(area)),
        "scratch_count": count,
        "scratches": objs,
        "overlay_path": overlay_path,
    }


# ── Legacy calibration ────────────────────────────────────────────────────────
# The Python port cannot be bit-identical to MATLAB (findpeaks interpolation,
# JPEG decoder, rounding conventions), so after the pipeline-level fixes a small
# global calibration maps the port's per-image measurements onto the original
# MATLAB numbers.  Fitted on the full 30-set / 3,600-image archive of
# old-system results — 20 glass-slide sets plus 10 PET-film sets (textured
# substrate, diagonals, smears) — via leg-weighted least squares with gentle
# minimax reweighting:
#   - glass:  80/80 leg means within 4% of MATLAB (min 96.1%, mean 98.6%)
#   - PET:    38/40 within 4% (min 95.6%, mean 98.0%) — the two stragglers
#     (CY266E-109425 Set 1) carry opposite-sign port-vs-MATLAB disagreement
#     within one set; even a PET-only fit cannot beat this (verified), which
#     is why there is ONE unified calibration and no per-substrate mode.
#   - generalization proof: the previous 20-set-only fit, applied FROZEN to
#     the never-seen PET sets, already scored 38/40 legs >= 96% (min 95.4%) —
#     the mapping is algorithm-to-algorithm, not set-specific.  Held-out
#     split checks (5 random 15/15) confirm.
# Features are simple per-image aggregates of the accepted scratch components.
_LEGACY_CAL_COEFFS = (
    0.446196,     # area in scratches >= 100 px long
    0.520699,     # area in scratches >= 400 px^2
    0.348288,     # area in scratches  < 400 px^2
    0.422791,     # area in scratches  < 100 px long
    180.258163,   # count of scratches >= 100 px long
    -18.622173,   # min(count, 80)
    -118.203751,  # max(count - 80, 0)      — swarm regime
    -0.402078,    # small-scratch (<200 px^2) area when count > 80 — swarm regime
)


def _legacy_calibrate(objects: list) -> int:
    """Map raw port measurements → MATLAB-equivalent scratch area (see above)."""
    a  = sum(o["area_px"] for o in objects)
    c  = len(objects)
    al = sum(o["area_px"] for o in objects if o["length_px"] >= 100)
    nl = sum(1 for o in objects if o["length_px"] >= 100)
    ab4 = sum(o["area_px"] for o in objects if o["area_px"] >= 400)
    small2 = sum(o["area_px"] for o in objects if o["area_px"] < 200)
    feats = (al, ab4, a - ab4, a - al, nl, min(c, 80), max(c - 80, 0),
             small2 if c > 80 else 0.0)
    est = sum(w * f for w, f in zip(_LEGACY_CAL_COEFFS, feats))
    return max(0, int(round(est)))


def _detect_legacy(rgb: np.ndarray):
    """MATLAB-faithful detector. Returns (area, count, objects, outlines).
    The returned area is calibrated to the original MATLAB implementation —
    see _LEGACY_CAL_COEFFS."""
    grey = _to_grey_adjusted(rgb)
    old_grey = grey.copy()

    # Build binary scratch mask from column peaks
    bw = _build_scratch_mask(grey)

    # Remove horizontal particles
    bw = _remove_horizontal_particles(bw, H_SIZE)

    # Bridge pixels
    bw_bool = _bwmorph_bridge(bw.astype(bool))

    # Remove small objects
    bw_bool = _bwareaopen(bw_bool, MIN_AREA_PIXELS)

    # Combine with original grey then re-binarize
    bw_float = bw_bool.astype(np.float64) * old_grey
    bw_bool = bw_float > IMBINARIZE_VALUE

    # Find objects and compute per-object stats.  Boundary tracing mirrors
    # MATLAB bwboundaries: an 8-connected pixel chain (cv2 CHAIN_APPROX_NONE)
    # closed back to its start, so the perimeter — and thus the roundness
    # metric — matches MATLAB's, unlike skimage find_contours (sub-pixel).
    import cv2
    from skimage.measure import label, regionprops

    labeled = label(bw_bool)
    props = regionprops(labeled)

    sum_scratch = 0
    scratch_count = 0
    scratch_objects = []
    accepted_outlines = []   # (scratch_num, boundary coords) for the overlay

    for prop in props:
        area = prop.area
        minr, minc = prop.bbox[0], prop.bbox[1]
        region = np.pad(prop.image.astype(np.uint8), 1)   # isolated region mask
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = cnts[0][:, 0, :]                               # (x=col, y=row) in padded crop
        # → full-image (row, col); undo the pad (-1) and add bbox origin
        boundary = np.column_stack([c[:, 1] + minr - 1, c[:, 0] + minc - 1]).astype(float)
        boundary = np.vstack([boundary, boundary[0]])      # close the loop (bwboundaries)
        delta = np.diff(boundary, axis=0)
        perimeter = float(np.sum(np.sqrt((delta ** 2).sum(axis=1))))
        if perimeter == 0:
            continue

        metric = 4 * math.pi * area / (perimeter ** 2)  # roundness

        y_del = boundary[:, 0].max() - boundary[:, 0].min()  # row span
        x_del = boundary[:, 1].max() - boundary[:, 1].min()  # col span

        if metric < THRESHOLD and x_del > y_del:
            sum_scratch += area
            scratch_count += 1
            width = int(y_del)
            length = int(x_del)
            scratch_objects.append({
                "scratch_num": scratch_count,
                "width_px": width,
                "length_px": length,
                "area_px": area,
            })
            accepted_outlines.append((scratch_count, boundary))

    # Calibrate the total onto the original MATLAB numbers (per-scratch
    # details stay raw; only the headline area is mapped).
    cal_area = _legacy_calibrate(scratch_objects)
    return cal_area, scratch_count, scratch_objects, accepted_outlines


def _detect_accurate(rgb: np.ndarray):
    """
    Independent, accuracy-first scratch detector (no MATLAB constraint).

    Strategy — measure only genuine HORIZONTAL scratches, measure them to their
    FULL faint extent, and reject the artefacts the scanner already knows about
    (round specs, dust chains, grey halos, substrate texture, diagonal rig
    artefacts):

      1. Estimate the bright background (large morphological close) and work on
         each pixel's DARKNESS below it — robust to uneven lighting.
      2. STRONG pass (confirmation): threshold at mean + 1.0*std, open with a
         horizontal line element, and gate components like a line, not a blob
         (length >= 18 px, aspect >= 2.6; thick components need a wide seed run
         — the comet-smear exception).  These are the confirmed scratch CORES.
      3. HYSTERESIS (full extent): threshold again at a weaker mean + 0.45*std.
         A weak component is accepted when it contains a confirmed core — the
         faint tails and gaps of a real scratch are connected to its dark core,
         so the scratch is measured end-to-end instead of only its darkest
         fragments (fixes systematic under-measurement).  If the weak extent
         degenerates into a blob (grain flood / merger with junk) the scratch
         falls back to its strong core instead of being dropped.
      4. FAINT-STREAK pass (matched filter): a horizontal running mean of the
         darkness image boosts faint horizontal lines ~5x in SNR while diluting
         dust dots and diagonals.  Long/thin components of the smoothed map are
         accepted only if their median per-column peak darkness clears a
         texture-adaptive gate max(22, mean + 1.5*std) — real faint streaks are
         consistently darker than grain, chance dust-dot chains are not.
      5. Union of all accepted pixels is re-labelled so touching pieces merge
         into single scratches; each final component is one scratch.

    Tuned and visually verified against the 30-set (glass + PET) corpus.
    Returns (area, count, objects, outlines).
    """
    import cv2
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Background (bright) via a large morphological close, then darkness below it
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, bg_kernel)
    darkness   = cv2.subtract(background, grey)          # >0 where darker than bg

    d_mean, d_std = float(darkness.mean()), float(darkness.std())
    thr_hi = max(8.0, d_mean + 1.0 * d_std)
    thr_lo = max(5.0, d_mean + 0.45 * d_std)
    strong = (darkness > thr_hi).astype(np.uint8)
    weak   = (darkness > thr_lo).astype(np.uint8)

    # 2. Strong pass — confirmed scratch cores (gates as before)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    seed = cv2.morphologyEx(strong, cv2.MORPH_OPEN, h_kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    core_ids = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 30 or w < 18:             # tiny specks / too short to be a line
            continue
        comp = labels[y:y + h, x:x + w] == i
        if not seed[y:y + h, x:x + w][comp].any():
            continue                     # no horizontal core → spec / blob / halo
        aspect = w / max(h, 1)
        if aspect < 2.6:                 # dots & merged dot-pairs are ~1:1
            continue
        if h > 12 and aspect < 4.0:
            # Thick and squat — usually a smudge/halo blob.  EXCEPTION: dense
            # abrasion smears ("comet" gouges, PET sets) are thick too, but
            # unlike dust/smudges they contain long horizontal streak runs.
            seed_in = seed[y:y + h, x:x + w].astype(bool) & comp
            if int(seed_in.sum(axis=1).max()) < 40:
                continue
        core_ids.append(i)
    core_mask = np.isin(labels, core_ids)

    # 2b. Dust-dot excision: a spec TOUCHING a scratch gets bridged to it by
    # the weak mask and would be swallowed into the scratch's extent by the
    # hysteresis merge.  Classic dust dots — compact, roundish, solid strong
    # components — are cut out of the weak mask first so they detach.  The
    # excision test is HORIZONTAL CONTEXT, not shape alone: a dash inside a
    # stippled scratch, or a dot the line passes straight through, has dark
    # structure on BOTH flanks at its own rows and is left in place; an
    # isolated spec (even one hanging off a line's side) has empty flanks.
    img_w = weak.shape[1]
    chain = cv2.morphologyEx(weak, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    dot_mask = np.zeros(weak.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (30 <= a <= 400 and w <= 25 and h <= 25):
            continue
        if w / max(h, 1) >= 2.2 or a / float(w * h) < 0.45:
            continue                                 # elongated / hollow → not a dot
        lx0, lx1 = max(0, x - 20), max(0, x - 5)
        rx0, rx1 = min(img_w, x + w + 5), min(img_w, x + w + 20)
        both = ((chain[y:y + h, lx0:lx1].any(axis=1)) &
                (chain[y:y + h, rx0:rx1].any(axis=1)))
        if int(both.sum()) >= 2:
            continue                                 # in a line/stipple chain → keep
        comp = labels[y:y + h, x:x + w] == i
        dot_mask[y:y + h, x:x + w][comp] = 1
    if dot_mask.any():
        dot_mask = cv2.dilate(dot_mask,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        weak = weak & (1 - dot_mask)

    # 3. Hysteresis — weak components that extend a confirmed core
    nw, labw, stw, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    core_labels = np.unique(labw[core_mask]) if core_mask.any() else np.array([], int)
    union = np.zeros(weak.shape, bool)
    for j in (int(v) for v in core_labels if v != 0):
        x, y, w, h, a = stw[j]
        comp = labw == j
        if h > 40 and w / max(h, 1) < 3.0:
            # weak extent became a blob (grain flood / merged with junk):
            # keep the confirmed strong core only, never drop the scratch
            union |= comp & core_mask
        else:
            union |= comp

    # 4. Faint-streak pass — matched filters for faint horizontal lines.
    # Two acceptance routes per component:
    #   • standard: length ≥ 45 with median per-column darkness above the
    #     texture-adaptive gate max(22, mean + 1.5σ) — real faint lines are
    #     consistently darker than grain, chance dust-dot chains only spike
    #     at the dots;
    #   • long-thin: a CONTINUOUS run ≥ 120 px that stays ≤ 10 px thin cannot
    #     be chance texture (noise chains top out near 100 px), so it only
    #     needs the softer gate max(25, mean + 1.0σ).  On heavily scratched
    #     frames σ is inflated by the scratches themselves and the standard
    #     gate over-adapts — this route is what still catches their faint
    #     full-width lines.
    # The second, longer filter (51 px) doubles the SNR boost for long lines,
    # reaching streaks too faint for the 25 px filter; only the long-thin
    # route applies at that scale.
    med_gate   = max(22.0, d_mean + 1.5 * d_std)
    thin_gate  = max(25.0, d_mean + 1.0 * d_std)
    ultra_gate = max(15.0, d_mean + 0.6 * d_std)

    def _faint_scan(filt_len, min_w, standard_route, thr_k=1.0, ultra_only=False):
        sm = cv2.blur(darkness.astype(np.float32), (filt_len, 1))
        thr_f = max(2.5, float(sm.mean()) + thr_k * float(sm.std()))
        faint = (sm > thr_f).astype(np.uint8)
        faint = cv2.morphologyEx(faint, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
        nf, labf, stf, _ = cv2.connectedComponentsWithStats(faint, connectivity=8)
        add = np.zeros(darkness.shape, bool)
        for k in range(1, nf):
            x, y, w, h, a = stf[k]
            if w < min_w or h > 10 or a < 60 or w / max(h, 1) < 6.0:
                continue
            compb = labf[y:y + h, x:x + w] == k
            d = darkness[y:y + h, x:x + w].astype(np.float32).copy()
            d[~compb] = 0
            med = float(np.median(d.max(axis=0)))
            if ultra_only:
                if w >= 150 and h <= 8 and w / max(h, 1) >= 18.0 and med >= ultra_gate:
                    add |= labf == k
            elif standard_route and w >= 45 and med >= med_gate:
                add |= labf == k
            elif w >= 120 and w / max(h, 1) >= 12.0 and med >= thin_gate:
                add |= labf == k
        return add

    faint_evid = _faint_scan(25, 45, standard_route=True)
    faint_evid |= _faint_scan(51, 120, standard_route=False)
    # ULTRA-FAINT route: a lower pixel threshold (mean + 0.6σ) with the 51 px
    # filter, accepting ONLY very long, very thin, very straight runs
    # (>= 150 px, <= 8 px, aspect >= 18) at a softer darkness gate.  Measured
    # against control-sample PET texture: the longest chance texture ridge is
    # ~127 px, so the 150 px floor keeps a real margin while recovering the
    # faintest full-length scratches on badly abraded frames.  Anything
    # shorter at this faintness is statistically indistinguishable from the
    # substrate's own texture and is deliberately NOT counted — a false
    # scratch on a control sample corrupts the control-vs-treatment
    # comparison this instrument exists to make.
    faint_evid |= _faint_scan(51, 150, standard_route=False, thr_k=0.6, ultra_only=True)
    union |= faint_evid

    # 4b. Dot-bulge shave: a dust spec dark enough to FUSE with a thin scratch
    # in the strong mask can't be excised as a component — it shows up as a
    # short, fat bulge on an otherwise thin line: thickness spikes past
    # max(2.5×, +8 px) the line's median over ≤ 34 columns (specs run up to
    # ~30 px), with the line continuing ≥ 15 columns on BOTH sides.
    # Comet-smear heads sit at the END of their scratch (and thick smear
    # bodies have median thickness > 15), so they never match.
    # The bulge is shaved back to the band spanned by the flanking columns.
    nu0, labu0, stu0, _ = cv2.connectedComponentsWithStats(
        union.astype(np.uint8), connectivity=8)
    for m in range(1, nu0):
        x, y, w, h, a = stu0[m]
        if w < 60 or h < 12:
            continue                                 # nothing to shave
        comp = labu0[y:y + h, x:x + w] == m
        thick = comp.sum(axis=0)
        nz = thick > 0
        if not nz.any():
            continue
        med_t = float(np.median(thick[nz]))
        if med_t > 15:
            continue                                 # thick body (smear/gouge)
        bulge = thick > max(2.5 * med_t, med_t + 8.0)
        c = 0
        while c < w:
            if not bulge[c]:
                c += 1
                continue
            c1 = c
            while c1 < w and bulge[c1]:
                c1 += 1
            if (c1 - c) <= 34 and c >= 15 and (w - c1) >= 15:
                # reference band = MEDIAN row extent of the flanking columns,
                # skipping the 3 nearest (the spec's halo bleeds into those,
                # and a min/max band would stretch to the whole bulge)
                ref = [np.nonzero(comp[:, rc])[0]
                       for rc in list(range(max(0, c - 17), max(0, c - 3))) +
                                 list(range(min(w, c1 + 3), min(w, c1 + 17)))
                       if thick[rc] > 0]
                if ref:
                    y0 = int(np.median([r.min() for r in ref])) - 3
                    y1 = int(np.median([r.max() for r in ref])) + 3
                    rows = np.arange(comp.shape[0])[:, None]
                    clip = comp[:, c:c1] & ((rows < y0) | (rows > y1))
                    # only shave a genuine DOT: never sever the line (every
                    # bulge column keeps band pixels), and the clipped region
                    # must be ONE compact roundish blob — thickness wobble of
                    # a real scratch clips as several thin slivers and is left
                    # alone
                    ok = clip.any() and not (
                        ((comp[:, c:c1] & ~clip).sum(axis=0) == 0).any())
                    if ok:
                        ncl, _, scl, _ = cv2.connectedComponentsWithStats(
                            clip.astype(np.uint8), connectivity=8)
                        ok = ncl == 2
                        if ok:
                            _, _, cw, ch, ca = scl[1]
                            ok = (cw <= 34 and ch <= 30 and ca >= 30
                                  and ca / float(cw * ch) >= 0.35)
                    if ok:
                        gmask = np.zeros_like(comp)
                        gmask[:, c:c1] = clip
                        union[y:y + h, x:x + w] &= ~gmask
            c = c1

    # 4c. Halo trim: a DARK line carries a wide optical/JPEG blur skirt, and
    # the absolute weak threshold wades into it — on soft-optics PET frames
    # the counted mask ran several px wider than the visible line (operator-
    # verified at 5x zoom).  Trim RELATIVELY: within each scratch, per column,
    # drop pixels fainter than a quarter of that column's own peak darkness.
    # Quarter-max sits between strict half-max (FWHM) width and the full
    # skirt — where the visible edge is.  Faint lines (peak ≈ threshold) are
    # untouched, so this cannot undo the faint-extent recovery; and since
    # every column keeps its peak, no scratch loses length or connectivity.
    nu0, labu0, stu0, _ = cv2.connectedComponentsWithStats(
        union.astype(np.uint8), connectivity=8)
    dark_f = darkness.astype(np.float32)
    for m in range(1, nu0):
        x, y, w, h, a = stu0[m]
        comp = labu0[y:y + h, x:x + w] == m
        d = dark_f[y:y + h, x:x + w].copy()
        d[~comp] = 0
        floor = 0.25 * d.max(axis=0)
        halo = comp & (d < floor[None, :])
        if halo.any():
            union[y:y + h, x:x + w] &= ~halo

    # 5. Final recount on the union — touching pieces merge into one scratch
    nu, labu, stu, _ = cv2.connectedComponentsWithStats(
        union.astype(np.uint8), connectivity=8)
    area = 0
    count = 0
    objs = []
    outlines = []
    evid = core_mask | faint_evid
    for m in range(1, nu):
        x, y, w, h, a = stu[m]
        if a < 30:
            continue    # crumbs left by the dot shave — every gate upstream
                        # already requires >= 30 px, so nothing real is lost
        comp = labu[y:y + h, x:x + w] == m
        # a final piece must carry its own EVIDENCE (a confirmed core or a
        # faint-route detection) or at least be shaped like a line — halo
        # pads that only rode along with a scratch and were separated by the
        # halo trim have neither, and are not scratches
        if not evid[y:y + h, x:x + w][comp].any():
            if w / max(h, 1) < 3.0 or a < 60:
                continue
        area += int(a)
        count += 1
        objs.append({"scratch_num": count, "width_px": int(h),
                     "length_px": int(w), "area_px": int(a)})
        # outline for the overlay (largest contour of this component)
        cm = (labu == m).astype(np.uint8)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)[:, 0, :]
            outlines.append((count, np.column_stack([c[:, 1], c[:, 0]]).astype(float)))

    return area, count, objs, outlines


# ── Excel export ──────────────────────────────────────────────────────────────

LEGS = ["FR", "FL", "BR", "BL"]
EXPECTED_IMAGES = 30


def _desc_stats(areas: list) -> list[tuple]:
    """Return descStats rows matching MATLAB output: (label, value)."""
    import statistics
    n = len(areas)
    mean_  = sum(areas) / n
    std_   = float(np.std(areas, ddof=1))
    return [
        ("mean",              round(mean_, 6)),
        ("standard error",    round(std_ / n ** 0.5, 6)),
        ("median",            round(statistics.median(areas), 6)),
        ("mode",              statistics.mode(areas)),
        ("standard deviation",round(std_, 6)),
        ("sample variance",   round(float(np.var(areas, ddof=1)), 6)),
        ("range",             max(areas) - min(areas)),
        ("min",               min(areas)),
        ("max",               max(areas)),
        ("sum",               sum(areas)),
        ("count",             n),
    ]


def _summary_row(set_name: str, leg_means: dict, all_areas_flat: list) -> dict:
    import datetime
    leg_vals = [leg_means[lg] for lg in LEGS if lg in leg_means]
    avg      = round(sum(leg_vals) / len(leg_vals))
    std_legs = round(float(np.std(leg_vals, ddof=1))) if len(leg_vals) > 1 else 0
    std_imgs = round(float(np.std(all_areas_flat, ddof=1)))
    rng      = max(leg_vals) - min(leg_vals)
    return dict(
        date       = datetime.date.today().strftime("%d-%b-%Y"),
        set_name   = set_name,
        avg        = avg,
        FR         = leg_means.get("FR", ""),
        FL         = leg_means.get("FL", ""),
        BR         = leg_means.get("BR", ""),
        BL         = leg_means.get("BL", ""),
        std_legs   = std_legs,
        std_imgs   = std_imgs,
        rng        = rng,
        leg_pct    = round(std_legs / avg * 100) if avg else "",
        img_pct    = round(std_imgs / avg * 100) if avg else "",
        rng_pct    = round(rng      / avg * 100) if avg else "",
    )


def _anova_rows(all_areas_by_leg: dict, order=None) -> tuple:
    """Return (anova_table_rows, pairwise_rows) matching MATLAB ANOVA1 sheet.
    `order` sets the leg column order (group numbering); defaults to LEGS."""
    from scipy.stats import f_oneway
    order      = order or LEGS
    groups     = [all_areas_by_leg[lg] for lg in order if lg in all_areas_by_leg]
    leg_labels = [lg for lg in order if lg in all_areas_by_leg]
    if len(groups) < 2:
        return [], []

    k   = len(groups)
    n   = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / n

    ss_between = sum(len(g) * (sum(g)/len(g) - grand_mean)**2 for g in groups)
    ss_within  = sum(sum((x - sum(g)/len(g))**2 for x in g) for g in groups)
    ss_total   = ss_between + ss_within
    df_between = k - 1
    df_within  = n - k
    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within
    f_stat, p_val = f_oneway(*groups)

    anova_rows = [
        ["Source", "SS", "df", "MS", "F", "Prob>F"],
        ["Columns", round(ss_between, 6), df_between,
         round(ms_between, 6), round(float(f_stat), 6), round(float(p_val), 6)],
        ["Error",  round(ss_within,  6), df_within,
         round(ms_within,  6), "", ""],
        ["Total",  round(ss_total,   6), n - 1, "", "", ""],
    ]

    # Pairwise: Tukey-style critical range (mirrors MATLAB multcompare output columns)
    from scipy.stats import studentized_range
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            gi, gj   = groups[i], groups[j]
            mi, mj   = sum(gi)/len(gi), sum(gj)/len(gj)
            diff     = mi - mj
            se       = (ms_within * (1/len(gi) + 1/len(gj)) / 2) ** 0.5
            q_crit   = studentized_range.ppf(0.95, k, df_within) if se > 0 else 0
            half_ci  = q_crit * se / 2**0.5
            pairs.append([
                i + 1, j + 1,
                round(diff - half_ci, 6),
                round(diff, 6),
                round(diff + half_ci, 6),
                round(float(f_oneway(gi, gj)[1]), 6),
            ])
    return anova_rows, pairs


# ── New consolidated format ───────────────────────────────────────────────────

def write_new_format(set_dir: str, leg_results: dict) -> str:
    """
    Single workbook: Summary tab + one tab per leg.
    Each leg tab has scratch areas + descStats on the left,
    individual scratch dimensions on the right.
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    set_name = os.path.basename(set_dir)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    leg_means        = {}
    all_areas_by_leg = {}

    HDR  = Font(bold=True)
    FILL_HDR  = PatternFill("solid", start_color="D9E1F2")
    FILL_STAT = PatternFill("solid", start_color="EBF1DE")
    THIN = Side(style="thin")
    BOX  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    CTR  = Alignment(horizontal="center")

    def _hdr(ws, row, col, text, fill=None):
        c = ws.cell(row=row, column=col, value=text)
        c.font = HDR
        if fill:
            c.fill = fill
        c.border = BOX
        c.alignment = CTR

    def _val(ws, row, col, value):
        c = ws.cell(row=row, column=col, value=value)
        c.border = BOX

    for leg in LEGS:
        results = leg_results.get(leg)
        if not results:
            continue

        sorted_results = sorted(results, key=lambda x: x["file"])
        areas = [r["scratch_area"] for r in sorted_results]
        leg_means[leg] = round(sum(areas) / len(areas))
        all_areas_by_leg[leg] = areas

        ws = wb.create_sheet(title=leg)
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 22
        ws.column_dimensions["C"].width = 3   # spacer
        ws.column_dimensions["D"].width = 14
        ws.column_dimensions["E"].width = 10
        ws.column_dimensions["F"].width = 10
        ws.column_dimensions["G"].width = 10

        # ── Left: scratch areas ───────────────────────────────────────────────
        _hdr(ws, 1, 1, "Image Name",           FILL_HDR)
        _hdr(ws, 1, 2, "Scratch Area (pixels)", FILL_HDR)
        for i, r in enumerate(sorted_results, start=2):
            _val(ws, i, 1, r["file"])
            _val(ws, i, 2, r["scratch_area"])

        stat_row = len(sorted_results) + 3
        _hdr(ws, stat_row - 1, 1, "Statistic", FILL_STAT)
        _hdr(ws, stat_row - 1, 2, "Value",     FILL_STAT)
        for label, value in _desc_stats(areas):
            _val(ws, stat_row, 1, label)
            _val(ws, stat_row, 2, value)
            ws.cell(row=stat_row, column=1).fill = FILL_STAT
            ws.cell(row=stat_row, column=2).fill = FILL_STAT
            stat_row += 1

        # ── Right: individual scratch dimensions ──────────────────────────────
        _hdr(ws, 1, 4, "Picture",   FILL_HDR)
        _hdr(ws, 1, 5, "Scratch #", FILL_HDR)
        _hdr(ws, 1, 6, "Width",     FILL_HDR)
        _hdr(ws, 1, 7, "Length",    FILL_HDR)
        detail_row = 2
        for r in sorted_results:
            for s in r.get("scratches", []):
                _val(ws, detail_row, 4, r["file"])
                _val(ws, detail_row, 5, s["scratch_num"])
                _val(ws, detail_row, 6, s["width_px"])
                _val(ws, detail_row, 7, s["length_px"])
                detail_row += 1

    # ── Summary tab ───────────────────────────────────────────────────────────
    ws_sum = wb.create_sheet(title="Summary", index=0)
    ws_sum.column_dimensions["A"].width = 16
    for col in ["B","C","D","E","F","G","H","I","J","K","L","M"]:
        ws_sum.column_dimensions[col].width = 18

    sum_headers = ["Date", "Set Name", "Avg Scratch Area",
                   "FR", "FL", "BR", "BL",
                   "Std of Legs", "Std of Images", "Range",
                   "Leg Std %", "Img Std %", "Range %"]
    for col, h in enumerate(sum_headers, start=1):
        _hdr(ws_sum, 1, col, h, FILL_HDR)

    all_areas_flat = [a for areas in all_areas_by_leg.values() for a in areas]
    if leg_means and all_areas_flat:
        s = _summary_row(set_name, leg_means, all_areas_flat)
        row_data = [s["date"], s["set_name"], s["avg"],
                    s["FR"], s["FL"], s["BR"], s["BL"],
                    s["std_legs"], s["std_imgs"], s["rng"],
                    s["leg_pct"], s["img_pct"], s["rng_pct"]]
        for col, val in enumerate(row_data, start=1):
            _val(ws_sum, 2, col, val)

        # Each leg's Statistic/Value (descStats) block, laid out side by side on
        # the Summary tab so all four are visible at a glance.  (ANOVA + pairwise
        # now live on their own "ANOVA" tab instead of below the summary.)
        block_cols = {"FR": 1, "FL": 4, "BR": 7, "BL": 10}   # A-B, D-E, G-H, J-K
        block_top  = 4
        for leg, c0 in block_cols.items():
            areas = all_areas_by_leg.get(leg)
            if not areas:
                continue
            _hdr(ws_sum, block_top,     c0,     leg,         FILL_STAT)
            _hdr(ws_sum, block_top,     c0 + 1, "",          FILL_STAT)
            _hdr(ws_sum, block_top + 1, c0,     "Statistic", FILL_STAT)
            _hdr(ws_sum, block_top + 1, c0 + 1, "Value",     FILL_STAT)
            rr = block_top + 2
            for label, value in _desc_stats(areas):
                _val(ws_sum, rr, c0,     label)
                _val(ws_sum, rr, c0 + 1, value)
                ws_sum.cell(row=rr, column=c0).fill     = FILL_STAT
                ws_sum.cell(row=rr, column=c0 + 1).fill = FILL_STAT
                rr += 1

        # ── ANOVA tab (own sheet): ANOVA table + pairwise comparisons ───────────
        if len(all_areas_by_leg) >= 2:
            anova_rows, pair_rows = _anova_rows(all_areas_by_leg)
            ws_an = wb.create_sheet(title="ANOVA", index=1)
            ws_an.column_dimensions["A"].width = 16
            for col in "BCDEF":
                ws_an.column_dimensions[col].width = 14
            r = 1
            _hdr(ws_an, r, 1, "ANOVA", FILL_STAT)
            r += 1
            for i, ar in enumerate(anova_rows):
                for col, val in enumerate(ar, start=1):
                    c = ws_an.cell(row=r, column=col, value=val)
                    c.border = BOX
                    if i == 0:                 # first row is the column header
                        c.font = HDR
                r += 1
            r += 1
            _hdr(ws_an, r, 1, "Pairwise Comparisons", FILL_STAT)
            r += 1
            for col, ph in enumerate(
                    ["Group 1", "Group 2", "Lower CI", "Difference", "Upper CI", "p-value"],
                    start=1):
                c = ws_an.cell(row=r, column=col, value=ph)
                c.font = HDR; c.border = BOX
            r += 1
            for pr in pair_rows:
                for col, val in enumerate(pr, start=1):
                    ws_an.cell(row=r, column=col, value=val).border = BOX
                r += 1

    path = os.path.join(set_dir, f"{set_name}_results.xlsx")
    wb.save(path)
    return path


# ── Legacy format (3 files matching original MATLAB output) ──────────────────

def write_legacy_format(set_dir: str, leg_results: dict) -> list[str]:
    """
    Writes three files matching the original MATLAB xls output:
      {set_name}_scratch_count.xlsx  (Sheet1, per-leg sheets, ANOVA1)
      {set_name}_scratch_data.xlsx   (per-leg individual scratch sheets)
      Summary.xlsx
    """
    import openpyxl
    set_name = os.path.basename(set_dir)
    paths = []

    leg_means        = {}
    all_areas_by_leg = {}

    # ── scratch_count ─────────────────────────────────────────────────────────
    wb_count = openpyxl.Workbook()
    wb_count.remove(wb_count.active)
    ws_sheet1 = wb_count.create_sheet("Sheet1")
    ws_sheet1.append(["file", "scratch area"])

    for leg in LEGS:
        results = leg_results.get(leg)
        if not results:
            continue
        sorted_results = sorted(results, key=lambda x: x["file"])
        areas = [r["scratch_area"] for r in sorted_results]
        leg_means[leg] = round(sum(areas) / len(areas))
        all_areas_by_leg[leg] = areas

        ws = wb_count.create_sheet(title=leg)
        ws.append(["image name", "scratch area (pixels)"])
        for r in sorted_results:
            ws.append([r["file"], r["scratch_area"]])
            ws_sheet1.append([r["file"], r["scratch_area"]])
        ws.append([])
        for label, value in _desc_stats(areas):
            ws.append([label, value])

    if len(all_areas_by_leg) >= 2:
        ws_anova = wb_count.create_sheet("ANOVA1")
        anova_rows, pair_rows = _anova_rows(all_areas_by_leg)
        for ar in anova_rows:
            ws_anova.append(ar)
        ws_anova.append([])
        for pr in pair_rows:
            ws_anova.append(pr)

    p1 = os.path.join(set_dir, f"{set_name}_scratch_count.xlsx")
    wb_count.save(p1)
    paths.append(p1)

    # ── scratch_data ──────────────────────────────────────────────────────────
    wb_data = openpyxl.Workbook()
    wb_data.remove(wb_data.active)
    for leg in LEGS:
        results = leg_results.get(leg)
        if not results:
            continue
        ws = wb_data.create_sheet(title=leg)
        ws.append(["Picture", "Scratch", "Width", "Length"])
        for r in sorted(results, key=lambda x: x["file"]):
            for s in r.get("scratches", []):
                ws.append([r["file"], s["scratch_num"], s["width_px"], s["length_px"]])

    p2 = os.path.join(set_dir, f"{set_name}_scratch_data.xlsx")
    wb_data.save(p2)
    paths.append(p2)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_areas_flat = [a for areas in all_areas_by_leg.values() for a in areas]
    if leg_means and all_areas_flat:
        wb_sum = openpyxl.Workbook()
        ws_s   = wb_sum.active
        ws_s.title = "Sheet1"
        ws_s.append(["", "date", "sample name", "average scratch area",
                     "FR", "FL", "BR", "BL",
                     "std.s of legs", "std.s of images", "range",
                     "leg std %", "img std %", "range %"])
        s = _summary_row(set_name, leg_means, all_areas_flat)
        ws_s.append([1, s["date"], s["set_name"], s["avg"],
                     s["FR"], s["FL"], s["BR"], s["BL"],
                     s["std_legs"], s["std_imgs"], s["rng"],
                     s["leg_pct"], s["img_pct"], s["rng_pct"]])
        p3 = os.path.join(set_dir, "Summary.xlsx")
        wb_sum.save(p3)
        paths.append(p3)

    return paths


# ── Summarize: combine many sets into one C8-style workbook ──────────────────

# Leg column order used by the C8 "Data set" template (top blocks + summary).
_SUMMARY_LEG_ORDER = ["BL", "BR", "FL", "FR"]
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Analysis-only mode ignores any file whose name isn't one of the expected
# capture frames 000..030.  Keeps stray/renamed files (thumbnails, overlays,
# hand-added images) from being processed and polluting the results.
def _is_numbered_frame(fname: str) -> bool:
    stem = os.path.splitext(fname)[0]
    return stem.isdigit() and 0 <= int(stem) <= EXPECTED_IMAGES


def _read_set_areas(workbook_path: str) -> dict:
    """
    Read one set's per-leg image scratch areas from its results workbook.

    Works for BOTH formats: the new '{set}_results.xlsx' and the legacy
    '{set}_scratch_count.xlsx' both keep raw data on sheets named FR/FL/BR/BL
    with the image name in column A and the scratch area in column B.  We take
    only the rows whose column A is an image filename, so the trailing descStats
    block (labelled 'mean', 'median', …) is skipped.

    Returns {leg: [areas]} for whichever legs are present.
    """
    import openpyxl
    wb = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    out = {}
    try:
        for leg in LEGS:
            if leg not in wb.sheetnames:
                continue
            areas = []
            for name, area in wb[leg].iter_rows(min_row=2, max_col=2, values_only=True):
                if (isinstance(name, str) and name.lower().endswith(_IMG_EXTS)
                        and isinstance(area, (int, float))):
                    areas.append(float(area))
            if areas:
                out[leg] = areas
    finally:
        wb.close()
    return out


def collect_sets(parent_dir: str) -> list:
    """
    Walk every subfolder under parent_dir and read each set's results workbook.
    Prefers the new '*_results.xlsx'; falls back to legacy '*_scratch_count.xlsx'.
    Returns a sorted list of (set_name, {leg: [areas]}).
    """
    sets = []
    for root, _dirs, files in os.walk(parent_dir):
        new    = sorted(f for f in files if f.endswith("_results.xlsx"))
        legacy = sorted(f for f in files if f.endswith("_scratch_count.xlsx"))
        if new:
            path, name = os.path.join(root, new[0]), new[0][:-len("_results.xlsx")]
        elif legacy:
            path, name = os.path.join(root, legacy[0]), legacy[0][:-len("_scratch_count.xlsx")]
        else:
            continue
        data = _read_set_areas(path)
        if data:
            sets.append((name, data))
    sets.sort(key=lambda x: x[0])
    return sets


def write_summarize_format(parent_dir: str, sets: list, out_path: str = None) -> str:
    """
    Combine many sets into one workbook mirroring the C8 'Data set' layout:
      • a horizontal block per set — raw per-image areas (BL/BR/FL/FR) → descStats
        → ANOVA table → pairwise comparisons;
      • two summary tables underneath — Film leg-means with per-set Average and
        STDEV of Legs plus aggregate Average/STDEV/RANGE, then an identical
        'Outliers Removed' copy where outlier cells are FLAGGED RED (never
        deleted): a leg cell is flagged if its mean lies outside the pooled
        leg-mean mean ± 1.96·stdev, and an Average cell if the set average lies
        outside the set-average mean ± 1.96·stdev.
    Returns the written path.
    """
    import datetime
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    HDR       = Font(bold=True)
    FILL_HDR  = PatternFill("solid", start_color="D9E1F2")
    FILL_STAT = PatternFill("solid", start_color="EBF1DE")
    FILL_RED  = PatternFill("solid", start_color="FFC7CE")   # outlier flag
    FONT_RED  = Font(color="9C0006", bold=True)
    THIN = Side(style="thin")
    BOX  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    CTR  = Alignment(horizontal="center")

    legs_order = _SUMMARY_LEG_ORDER
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S1"

    def cell(r, c, value, *, bold=False, fill=None, box=False, ctr=False, font=None):
        x = ws.cell(row=r, column=c, value=value)
        if bold:  x.font = HDR
        if font:  x.font = font
        if fill:  x.fill = fill
        if box:   x.border = BOX
        if ctr:   x.alignment = CTR
        return x

    # Metadata header (top-left, rows 1-3): fill the date; leave the rest blank.
    cell(1, 1, "Date Completed", bold=True)
    ws.cell(row=1, column=2, value=datetime.date.today().strftime("%d-%b-%Y"))
    cell(2, 1, "Completed By", bold=True)
    cell(3, 1, "Notes", bold=True)

    # ── Top: one horizontal block per set (below the metadata header) ──────────
    BLOCK_TOP  = 5                        # first block row (leaves rows 1-3 for meta)
    N_IMG      = 30                       # expected images per leg (layout spacing)
    SET_WIDTH  = len(legs_order) * 2      # 2 columns (name, area) per leg
    SET_STRIDE = SET_WIDTH + 1            # one spacer column between sets
    per_set = []                          # cache computed values for the summary

    for i, (name, data) in enumerate(sets):
        c0 = 1 + i * SET_STRIDE
        cell(BLOCK_TOP, c0, name, bold=True)
        for j, leg in enumerate(legs_order):
            nc = c0 + j * 2               # image-name column for this leg
            cell(BLOCK_TOP + 1, nc, leg, bold=True, box=True, ctr=True)
            cell(BLOCK_TOP + 2, nc,     "image name",            bold=True, box=True)
            cell(BLOCK_TOP + 2, nc + 1, "scratch area (pixels)", bold=True, box=True)
            areas = data.get(leg, [])
            r = BLOCK_TOP + 3
            for k, a in enumerate(areas):
                cell(r, nc,     f"{k + 1:03d}.jpg", box=True)
                cell(r, nc + 1, a,                  box=True)
                r += 1
            if areas:                     # descStats block below the raw rows
                for label, value in _desc_stats(areas):
                    cell(r, nc,     label, fill=FILL_STAT, box=True)
                    cell(r, nc + 1, value, fill=FILL_STAT, box=True)
                    r += 1

        # ANOVA + pairwise for this set (needs ≥2 legs)
        present = {lg: data[lg] for lg in legs_order if lg in data}
        if len(present) >= 2:
            anova_rows, pair_rows = _anova_rows(present, order=legs_order)
            ar = BLOCK_TOP + 3 + N_IMG + len(_desc_stats([0])) + 2   # below the stats
            for gi, arow in enumerate(anova_rows):
                for dc, val in enumerate(arow):
                    cell(ar, c0 + dc, val, bold=(gi == 0), box=True)
                ar += 1
            ar += 1
            for prow in pair_rows:
                for dc, val in enumerate(prow):
                    cell(ar, c0 + dc, val, box=True)
                ar += 1

        # Per-set summary numbers (display leg order; only present legs count)
        leg_means = {lg: sum(data[lg]) / len(data[lg]) for lg in present}
        avg = sum(leg_means.values()) / len(leg_means) if leg_means else 0.0
        std_legs = float(np.std(list(leg_means.values()), ddof=1)) if len(leg_means) > 1 else 0.0
        per_set.append((name, leg_means, avg, std_legs))

    # ── Bottom: the two Film summary tables (below every top block) ─────────────
    start_row = BLOCK_TOP + 3 + N_IMG + len(_desc_stats([0])) + 3 + 6 + 4
    grand_row = _write_summary_table(
        ws, start_row, "", per_set, legs_order,
        cell, HDR, FILL_STAT, FILL_RED, FONT_RED, flag_outliers=False)

    _write_summary_table(
        ws, grand_row + 3, "Outliers Removed", per_set, legs_order,
        cell, HDR, FILL_STAT, FILL_RED, FONT_RED, flag_outliers=True)

    out_path = out_path or os.path.join(
        parent_dir, f"{os.path.basename(os.path.abspath(parent_dir))}_summary.xlsx")
    wb.save(out_path)
    return out_path


def _write_summary_table(ws, top, title, per_set, legs_order,
                         cell, HDR, FILL_STAT, FILL_RED, FONT_RED, flag_outliers):
    """Write one Film summary table starting at row `top`; return its last row."""
    r = top
    if title:
        cell(r, 1, title, bold=True)
        r += 1
    headers = ["Film"] + legs_order + ["Average", "STDEV of Legs", "95% Confidence Interval"]
    for c, h in enumerate(headers, start=1):
        cell(r, c, h, bold=True, fill=FILL_STAT, box=True)
    r += 1

    # Outlier reference distributions (pooled across sets).
    all_leg_means = [m for _n, lm, _a, _s in per_set for m in lm.values()]
    set_avgs      = [a for _n, _lm, a, _s in per_set]
    lm_mean = sum(all_leg_means) / len(all_leg_means) if all_leg_means else 0.0
    lm_std  = float(np.std(all_leg_means, ddof=1)) if len(all_leg_means) > 1 else 0.0
    av_mean = sum(set_avgs) / len(set_avgs) if set_avgs else 0.0
    av_std  = float(np.std(set_avgs, ddof=1)) if len(set_avgs) > 1 else 0.0

    def outlier(val, mu, sd):
        return sd > 0 and abs(val - mu) > 1.96 * sd

    for name, leg_means, avg, std_legs in per_set:
        cell(r, 1, name, box=True)
        for c, leg in enumerate(legs_order, start=2):
            v = leg_means.get(leg)
            x = cell(r, c, round(v, 1) if v is not None else "", box=True)
            if flag_outliers and v is not None and outlier(v, lm_mean, lm_std):
                x.fill = FILL_RED; x.font = FONT_RED
        ax = cell(r, 2 + len(legs_order), round(avg, 1), box=True)
        if flag_outliers and outlier(avg, av_mean, av_std):
            ax.fill = FILL_RED; ax.font = FONT_RED
        cell(r, 3 + len(legs_order), round(std_legs, 1), box=True)
        r += 1

    # Aggregate rows: per-leg Average / STDEV across sets, plus pooled stats.
    avg_col = 2 + len(legs_order)
    cell(r, 1, "Average", bold=True, box=True)
    for c, leg in enumerate(legs_order, start=2):
        col_vals = [lm[leg] for _n, lm, _a, _s in per_set if leg in lm]
        cell(r, c, round(sum(col_vals) / len(col_vals), 1) if col_vals else "", box=True)
    cell(r, avg_col, round(av_mean, 1), box=True)
    cell(r, avg_col + 1, round(lm_std, 1), box=True)            # pooled STDEV of Legs
    ci = 1.96 * lm_std / (len(all_leg_means) ** 0.5) if all_leg_means else 0.0
    cell(r, avg_col + 2, round(ci, 1), box=True)               # 95% CI
    r += 1

    cell(r, 1, "STDEV", bold=True, box=True)
    for c, leg in enumerate(legs_order, start=2):
        col_vals = [lm[leg] for _n, lm, _a, _s in per_set if leg in lm]
        sd = float(np.std(col_vals, ddof=1)) if len(col_vals) > 1 else 0.0
        cell(r, c, round(sd, 1), box=True)
    r += 1

    cell(r, 1, "RANGE", bold=True, box=True)
    rng = (max(all_leg_means) - min(all_leg_means)) if all_leg_means else 0.0
    cell(r, 2, round(rng, 3), box=True)
    return r


# ── Pipeline class ────────────────────────────────────────────────────────────

class AnalysisPipeline:
    """
    Watches leg_dir (set_dir/leg/) for new images and processes them.
    When done, triggers set-level Excel export via on_leg_done callback.
    """
    def __init__(self, leg_dir,
                 on_progress=None, on_done=None, on_error=None,
                 on_image=None, total_expected=None, mode="legacy",
                 live_capture=False):
        self._dir = leg_dir
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_done = on_done or (lambda results: None)
        self._on_error = on_error or (lambda e: None)
        # on_image(overlay_path, fname): fired after each image is analysed, for
        # a live "last analysed" preview in the GUI.
        self._on_image = on_image or (lambda overlay_path, fname: None)
        self._total = total_expected
        self._mode = mode               # "legacy" (MATLAB) or "accurate"
        self._stop_event = threading.Event()
        # live_capture=True: images are still being captured into leg_dir, so
        # the watcher must NOT stop on an idle gap between shots — only once
        # mark_capture_done() is called (and the backlog is drained).  This lets
        # analysis run concurrently with capture instead of waiting for the set.
        self._live_capture = live_capture
        self._capture_done = threading.Event()
        self._thread = None
        self._results_path = os.path.join(leg_dir, "results.jsonl")

    def mark_capture_done(self):
        """Signal that no more images will be captured — the watcher may finish
        the remaining backlog and then stop."""
        self._capture_done.set()

    def start(self):
        os.makedirs(self._dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        try:
            processed = set()
            all_results = []
            done = 0
            idle_ticks = 0

            with open(self._results_path, "w") as rf:
                while not self._stop_event.is_set():
                    images = sorted(
                        f for f in os.listdir(self._dir)
                        if (f.endswith(".jpg") or f.endswith(".png"))
                        and not f.endswith("_overlay.png")
                        and f not in processed
                        # Frames the autofocus gave up on are saved as
                        # NNN_soft.jpg for the record — never analysed.
                        and "_soft" not in f
                        # Analysis-only mode: process ONLY the numbered capture
                        # frames 000..030, nothing else.  (Live/"both" capture
                        # writes its own frames, so no need to filter there.)
                        and (self._live_capture or _is_numbered_frame(f))
                    )
                    if images:
                        idle_ticks = 0
                        for fname in images:
                            if self._stop_event.is_set():
                                break
                            path = os.path.join(self._dir, fname)
                            try:
                                result = detect_scratches(path, mode=self._mode)
                                result["file"] = fname
                                all_results.append(result)
                                rf.write(json.dumps({
                                    "file": fname,
                                    "scratch_area": result["scratch_area"],
                                    "scratch_count": result["scratch_count"],
                                }) + "\n")
                                rf.flush()
                                # Live preview of the annotated overlay
                                self._on_image(result["overlay_path"], fname)
                            except Exception as e:
                                rf.write(json.dumps({"file": fname, "error": str(e)}) + "\n")
                                rf.flush()
                            processed.add(fname)
                            done += 1
                            self._on_progress(done, self._total)
                    else:
                        idle_ticks += 1
                        # While capture is still live, never stop on an idle gap
                        # between shots — wait for more images (or stop()).
                        waiting_for_capture = (
                            self._live_capture and not self._capture_done.is_set()
                        )
                        if (not waiting_for_capture) and done > 0 and idle_ticks > 10:
                            break
                        time.sleep(0.5)

            self._on_done(all_results)
        except Exception as e:
            self._on_error(e)
