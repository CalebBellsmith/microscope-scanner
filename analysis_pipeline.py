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

def _to_grey_adjusted(rgb_array: np.ndarray) -> np.ndarray:
    """rgb uint8 → float64 greyscale, inverted and contrast-stretched (imadjust)."""
    rgb = rgb_array.astype(np.float64) / 255.0
    grey = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    grey = 1.0 - grey  # invert
    # imadjust: clip bottom/top 1 % and rescale to [0,1]
    lo, hi = np.percentile(grey, 1), np.percentile(grey, 99)
    if hi > lo:
        grey = np.clip((grey - lo) / (hi - lo), 0.0, 1.0)
    return grey


def _build_scratch_mask(grey: np.ndarray) -> np.ndarray:
    """Per-column peak detection → binary mask (mirrors MATLAB findpeaks loop)."""
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
        pixels = []
        for peak, w in zip(peaks, widths):
            lo_idx = int(round(peak - w / 2))
            hi_idx = int(round(peak + w / 2))
            pixels.extend(range(lo_idx, hi_idx + 1))
        pixels = sorted(set(pixels))
        for p in pixels:
            if 0 < p < m:
                mask[p, col] = 1.0
    return mask


def _remove_horizontal_particles(bw: np.ndarray, iterations: int = H_SIZE) -> np.ndarray:
    """Gradient-based horizontal particle removal (mirrors MATLAB loop)."""
    for _ in range(iterations):
        # gradient along columns (axis=1)
        dx = np.gradient(bw, axis=1)
        # imadjust on dx: clip 1-99 percentile
        lo, hi = np.percentile(dx, 1), np.percentile(dx, 99)
        if hi > lo:
            dx_adj = np.clip((dx - lo) / (hi - lo), 0.0, 1.0)
        else:
            dx_adj = dx.copy()
        # bw(:,2:end) - dx(:,1:end-1)  [MATLAB 1-indexed, shifted by 1]
        new_bw = bw.copy()
        new_bw[:, 1:] = bw[:, 1:] - dx_adj[:, :-1]
        new_bw[new_bw < 0] = 0.0
        bw = np.round(new_bw)
    return bw


def _bwareaopen(bw_bool: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove connected components smaller than min_pixels."""
    from skimage.morphology import remove_small_objects
    return remove_small_objects(bw_bool, min_size=min_pixels)


def _bwmorph_bridge(bw_bool: np.ndarray) -> np.ndarray:
    """
    Exact equivalent of MATLAB bwmorph(bw, 'bridge').
    Sets a background pixel to foreground if doing so connects two foreground
    pixels in its 3×3 neighbourhood that were not already 8-connected.
    Applied once (MATLAB default with no repeat count).
    """
    from scipy.ndimage import label as nd_label
    padded = np.pad(bw_bool.astype(np.uint8), 1, mode='constant')
    out = padded.copy()
    rows, cols = bw_bool.shape
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if padded[r, c]:
                continue  # already foreground
            hood = padded[r-1:r+2, c-1:c+2].copy()
            hood[1, 1] = 1  # pretend this pixel is set
            # count 8-connected components in the 3×3 hood
            _, n = nd_label(hood, structure=np.ones((3, 3), dtype=int))
            if n == 1:  # setting this pixel merges components → bridge
                out[r, c] = 1
    return out[1:-1, 1:-1].astype(bool)


def detect_scratches(image_path: str) -> dict:
    """
    Run the full scratch detection pipeline on one image.

    Returns a dict with:
      scratch_area   : int   total pixel area of detected scratches
      scratch_count  : int   number of distinct scratch objects
      overlay_path   : str   path to the saved overlay PNG
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))
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

    # Find contours and compute per-object stats
    from skimage.measure import label, regionprops, find_contours

    labeled = label(bw_bool)
    props = regionprops(labeled)

    scratch_matrix = np.zeros(bw_bool.shape, dtype=np.float64)
    sum_scratch = 0
    scratch_count = 0
    scratch_objects = []

    for prop in props:
        area = prop.area
        # Approximate perimeter from bounding box contour
        contour_coords = find_contours(labeled == prop.label, 0.5)
        if not contour_coords:
            continue
        boundary = contour_coords[0]  # largest contour
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
            # Mark boundary pixels in scratch matrix
            for r, c in boundary.astype(int):
                if 0 <= r < scratch_matrix.shape[0] and 0 <= c < scratch_matrix.shape[1]:
                    scratch_matrix[r, c] = 1.0

    # Fill scratch outlines
    from scipy.ndimage import binary_fill_holes
    scratch_matrix = binary_fill_holes(scratch_matrix.astype(bool)).astype(np.float64)

    # Save overlay: channel 0 = grey, channel 1 = scratch * 0.5
    overlay = np.zeros((*bw_bool.shape, 3), dtype=np.float64)
    overlay[:, :, 0] = old_grey
    overlay[:, :, 1] = scratch_matrix * 0.5
    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    base = os.path.splitext(image_path)[0]
    overlay_path = base + "_overlay.png"
    Image.fromarray(overlay_uint8).save(overlay_path)

    return {
        "scratch_area": int(round(sum_scratch)),
        "scratch_count": scratch_count,
        "scratches": scratch_objects,
        "overlay_path": overlay_path,
    }


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

        # ANOVA below summary
        if len(all_areas_by_leg) >= 2:
            anova_rows, pair_rows = _anova_rows(all_areas_by_leg)
            r = 4
            _hdr(ws_sum, r, 1, "ANOVA", FILL_STAT)
            r += 1
            for ar in anova_rows:
                for col, val in enumerate(ar, start=1):
                    c = ws_sum.cell(row=r, column=col, value=val)
                    c.border = BOX
                    if r == 5:
                        c.font = HDR
                r += 1
            r += 1
            _hdr(ws_sum, r, 1, "Pairwise Comparisons", FILL_STAT)
            r += 1
            for ph in ["Group 1", "Group 2", "Lower CI", "Difference", "Upper CI", "p-value"]:
                c = ws_sum.cell(row=r, column=["Group 1","Group 2","Lower CI","Difference","Upper CI","p-value"].index(ph)+1, value=ph)
                c.font = HDR; c.border = BOX
            r += 1
            for pr in pair_rows:
                for col, val in enumerate(pr, start=1):
                    ws_sum.cell(row=r, column=col, value=val).border = BOX
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
                 total_expected=None):
        self._dir = leg_dir
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_done = on_done or (lambda results: None)
        self._on_error = on_error or (lambda e: None)
        self._total = total_expected
        self._stop_event = threading.Event()
        self._thread = None
        self._results_path = os.path.join(leg_dir, "results.jsonl")

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
                                result = detect_scratches(path)
                                result["file"] = fname
                                all_results.append(result)
                                rf.write(json.dumps({
                                    "file": fname,
                                    "scratch_area": result["scratch_area"],
                                    "scratch_count": result["scratch_count"],
                                }) + "\n")
                                rf.flush()
                            except Exception as e:
                                rf.write(json.dumps({"file": fname, "error": str(e)}) + "\n")
                                rf.flush()
                            processed.add(fname)
                            done += 1
                            self._on_progress(done, self._total)
                    else:
                        idle_ticks += 1
                        if done > 0 and idle_ticks > 10:
                            break
                        time.sleep(0.5)

            self._on_done(all_results)
        except Exception as e:
            self._on_error(e)
