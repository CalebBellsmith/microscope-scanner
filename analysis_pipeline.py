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
    """Remove connected components smaller than min_pixels."""
    from skimage.morphology import remove_small_objects
    return remove_small_objects(bw_bool, min_size=min_pixels)


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

    mode = "legacy"   → faithful port of the MATLAB pipeline (~95% of MATLAB).
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


def _detect_legacy(rgb: np.ndarray):
    """MATLAB-faithful detector. Returns (area, count, objects, outlines)."""
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

    return sum_scratch, scratch_count, scratch_objects, accepted_outlines


def _detect_accurate(rgb: np.ndarray):
    """
    Independent, accuracy-first scratch detector (no MATLAB constraint).

    Strategy — measure only genuine HORIZONTAL scratches and reject the
    artefacts the scanner already knows about (round specs, grey dots/halos,
    non-horizontal fibres/blobs):

      1. Estimate the bright background and take each pixel's DARKNESS below it
         (scratches are darker than the slide).  Working on local darkness makes
         detection robust to uneven lighting.
      2. Threshold by CONTRAST (darkness vs. background noise) — soft grey
         halos are low-contrast and fall out here.
      3. Morphologically OPEN with a horizontal line element — only pixels that
         belong to a horizontal run survive, erasing round specs/dust and
         vertical/diagonal features.
      4. Keep dark components that overlap a horizontal seed AND are wider than
         tall (aspect gate) — the same shape logic the scanner uses to tell a
         scratch from a blob.  Their dark-pixel area is the scratch area.

    Returns (area, count, objects, outlines).
    """
    import cv2
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = grey.shape

    # 1. Background (bright) via a large morphological close, then darkness below it
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, bg_kernel)
    darkness   = cv2.subtract(background, grey)          # >0 where darker than bg

    # 2. Contrast threshold — grey halos / gradients are weak here and excluded
    d_mean, d_std = float(darkness.mean()), float(darkness.std())
    thr = max(8.0, d_mean + 1.0 * d_std)
    dark_mask = (darkness > thr).astype(np.uint8)

    # 3. Horizontal seeds — open with a wide-short element to keep only h-runs
    h_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horiz_seed = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, h_kernel)

    # 4. Keep dark components that are horizontal scratches
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    area = 0
    count = 0
    objs = []
    outlines = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 30:                       # ignore tiny dust specks
            continue
        comp = labels[y:y + h, x:x + w] == i
        if not horiz_seed[y:y + h, x:x + w][comp].any():
            continue                     # no horizontal core → spec / blob / halo
        if w <= h:                       # must be wider than tall (horizontal)
            continue
        area += int(a)
        count += 1
        objs.append({"scratch_num": count, "width_px": int(h),
                     "length_px": int(w), "area_px": int(a)})
        # outline for the overlay (contour of this component)
        cm = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = cnts[0][:, 0, :]
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


def _anova_rows(all_areas_by_leg: dict) -> tuple:
    """Return (anova_table_rows, pairwise_rows) matching MATLAB ANOVA1 sheet."""
    from scipy.stats import f_oneway
    groups     = [all_areas_by_leg[lg] for lg in LEGS if lg in all_areas_by_leg]
    leg_labels = [lg for lg in LEGS if lg in all_areas_by_leg]
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
