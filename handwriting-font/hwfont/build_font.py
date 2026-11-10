"""Turn extracted glyph bitmaps into an OpenType (CFF) handwriting font.

Normalisation (strip px -> font units)
    Every glyph comes from a strip whose rule spacing S (px) is a physical constant of the notebook,
    so S absorbs photo distance and perspective. Writing size still varies page to page, so each
    page gets rho = median(x-height letter height / S); the glyph is scaled so that page's x-height
    lands on X_HEIGHT units. Baseline = glyph bottom for letters that sit on the line; descender
    letters hang from the x-height line instead (top of main body + page x-height).

Pen weight
    Photos differ in focus and exposure, so raw traces range from hairline to marker. Each glyph is
    re-thresholded until its median stroke width matches STROKE_UNITS, giving one consistent pen.

Coverage
    Characters with no clean sample are filled, in order of preference, from (1) your own glyphs
    (missing capitals from scaled lowercase, p from a lowered P, q/g from your a + a stem), then
    (2) pen-drawn skeletons rendered at the same stroke width. `coverage.json` records which
    characters are real, derived or drawn, so it is auditable.

Variation
    Up to 3 forms per character: real samples first, then small affine perturbations. A `calt`
    lookup cycles default -> alt1 -> alt2 keyed on the previous glyph (spaces included), so repeated
    letters rarely look stamped. Alternates are also mapped to Private Use codepoints (see
    <font>.pua.json) so a renderer can pick forms explicitly.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import potrace
from fontTools.agl import UV2AGL
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.transformPen import TransformPen
from skimage.morphology import skeletonize

# ---- font metrics (units per em = 1000)
UPM = 1000
X_HEIGHT = 330
CAP = 1.75 * X_HEIGHT            # nominal; real capitals keep the size they were written at
ASCENT, DESCENT = 880, 320
SIDE_BEARING = 0.05 * X_HEIGHT   # tight: joined-hand writers leave little air between letters
SPACE_WIDTH = 0.95 * X_HEIGHT
STROKE_UNITS = 0.13 * X_HEIGHT   # target pen width
N_FORMS = 3
PUA_BASE = 0xE000
SLANT = math.tan(math.radians(6))  # forward lean applied to pen-drawn fallbacks

TRACE_UPSAMPLE = 2

X_LETTERS = set("acemnorsuvwxz")
DESCENDERS = set("gjpqy")
PUNCT_ANCHOR = {",": "comma", ";": "comma", "'": "cap_top", '"': "cap_top", "-": "mid", "=": "mid", "+": "mid"}
NEEDS_DOT = set("?!")

# Target glyph heights in x-heights. Samples come from pages written at different sizes and this
# hand has very tall ascender loops, so each glyph is pulled toward its class height, keeping
# HEIGHT_KEEP of its natural deviation (exponent) so the text still varies like real writing.
HEIGHT_KEEP = 0.2


def target_height(ch: str) -> float | None:
    if ch in X_LETTERS:
        return 1.0
    if ch == "t":
        return 1.6
    if ch in "bdfhkl":
        return 1.95
    if ch in "gjpqy":
        return 1.7
    if ch == "i":
        return 1.4
    if ch.isupper():
        return 1.75
    if ch.isdigit():
        return 1.6
    return {"(": 2.0, ")": 2.0, "/": 1.8, "!": 1.7, "?": 1.7, "$": 1.9}.get(ch)

REQUIRED = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?'\"-:;()&$%/"
)


def glyph_name(ch: str) -> str:
    return UV2AGL.get(ord(ch), f"uni{ord(ch):04X}")


# ---------------------------------------------------------------- helpers

def bounds(rec: RecordingPen):
    bp = BoundsPen(None)
    rec.replay(bp)
    return bp.bounds


def transformed(rec: RecordingPen, m) -> RecordingPen:
    out = RecordingPen()
    rec.replay(TransformPen(out, m))
    return out


def merged(*recs: RecordingPen) -> RecordingPen:
    out = RecordingPen()
    for r in recs:
        r.replay(out)
    return out


def trace_bitmap(ink: np.ndarray, px_to_units: float, baseline_px: float, left_px: float) -> RecordingPen:
    """potrace a boolean image (True = ink) into font units; baseline_px row -> y=0, left_px -> x=0."""
    path = potrace.Bitmap(~ink).trace(turdsize=int(10 * TRACE_UPSAMPLE ** 2), alphamax=1.15, opttolerance=0.5)

    def tf(p):
        return ((p.x - left_px) * px_to_units, (baseline_px - p.y) * px_to_units)

    rec = RecordingPen()
    # potrace (y down) gives outer contours positive, holes negative; flipping y inverts that,
    # so reverse every contour to get CFF's counter-clockwise outer contours.
    pen = ReverseContourPen(rec)
    for curve in path:
        pen.moveTo(tf(curve.start_point))
        for seg in curve.segments:
            if seg.is_corner:
                pen.lineTo(tf(seg.c))
                pen.lineTo(tf(seg.end_point))
            else:
                pen.curveTo(tf(seg.c1), tf(seg.c2), tf(seg.end_point))
        pen.closePath()
    return rec


def stroke_width_px(ink: np.ndarray) -> float:
    """Median stroke width: 2 x distance-to-edge sampled on the skeleton."""
    if ink.sum() < 20:
        return 0.0
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    vals = dt[sk]
    return float(2 * np.median(vals)) if len(vals) else 0.0


# ---------------------------------------------------------------- real glyphs

def page_rho(records) -> tuple[dict, float]:
    """x-height / rule-spacing per source image, and the global median as a fallback."""
    per = defaultdict(list)
    for r in records:
        if r["char"] in X_LETTERS:
            per[r["image"]].append((r["bottom"] - r["top"]) / r["spacing_px"])
    rho = {k: float(np.median(v)) for k, v in per.items() if len(v) >= 2}
    allv = [x for v in per.values() for x in v]
    return rho, (float(np.median(allv)) if allv else 0.42)


def main_body_top(mask: np.ndarray) -> int:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return int(st[i, cv2.CC_STAT_TOP])


def n_pieces(mask: np.ndarray) -> int:
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[0] - 1


def trace_record(record, scale: float, baseline_px: float) -> RecordingPen:
    """Vectorise one sample at the threshold that gives the target pen width."""
    soft = record["soft"]
    up = cv2.resize(soft, None, fx=TRACE_UPSAMPLE, fy=TRACE_UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.2 * TRACE_UPSAMPLE)
    m = cv2.resize(record["mask"].astype(np.uint8), (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST)
    region = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    px_to_units = scale / TRACE_UPSAMPLE
    target_px = STROKE_UNITS / px_to_units
    best, best_err = None, 1e9
    for thr in range(14, 96, 6):
        ink = (up > thr) & region
        w = stroke_width_px(ink)
        if w == 0:
            break
        err = abs(w - target_px)
        if err < best_err:
            best, best_err = ink, err
    ink = best
    # residual correction when thresholds alone cannot reach the target (very faint/very bold photo)
    w = stroke_width_px(ink)
    delta = int(round((target_px - w) / 2))
    if delta > 0:
        ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * delta + 1,) * 2)) > 0
    elif delta < -1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -delta + 1,) * 2)
        thinner = cv2.erode(ink.astype(np.uint8), k) > 0
        if n_pieces(thinner) <= n_pieces(ink):  # never break a stroke apart
            ink = thinner
    xs = np.where(ink.any(0))[0]
    left = xs.min() if len(xs) else 0
    return trace_bitmap(ink, px_to_units, baseline_px * TRACE_UPSAMPLE, left)


def place(records):
    """-> ({char: [RecordingPen]}, {char: [needs_dot]}) in font units, baseline at 0."""
    rho, rho_global = page_rho([r for r in records if r.get("source") != "template"])
    # deliberate template writing beats journal scrawl: if a char has template samples, use only those
    templated = {r["char"] for r in records if r.get("source") == "template"}
    records = [r for r in records if r.get("source") == "template" or r["char"] not in templated]
    out, dotless = defaultdict(list), defaultdict(list)
    for r in records:
        ch = r["char"]
        exact = "baseline_px" in r  # template boxes print the baseline and x-height
        xh_px = r["xh_px"] if exact else rho.get(r["image"], rho_global) * r["spacing_px"]
        scale = X_HEIGHT / xh_px
        top, bottom = r["top"], r["bottom"]
        tgt = target_height(ch)
        if tgt:
            measured = (bottom - top) / xh_px
            keep = 0.6 if exact else HEIGHT_KEEP  # template writing is consistent: normalise gently
            scale *= float(np.clip((tgt / measured) ** (1 - keep), 0.55, 1.7))
        anchor = PUNCT_ANCHOR.get(ch)
        missing_dot = ch in NEEDS_DOT and n_pieces(r["mask"]) == 1 and not exact
        if exact:
            baseline = r["baseline_px"]
        elif ch in DESCENDERS:
            baseline = main_body_top(r["mask"]) + xh_px
        elif anchor == "comma":
            baseline = top + 0.25 * xh_px
        elif anchor == "cap_top":
            baseline = top + 1.75 * xh_px
        elif anchor == "mid":
            baseline = (top + bottom) / 2 + 0.5 * xh_px
        elif missing_dot:
            baseline = bottom + 0.42 * xh_px  # leave room for the dot we add below
        else:
            baseline = bottom
        out[ch].append(trace_record(r, scale, baseline))
        dotless[ch].append(missing_dot)
    return out, dotless


# ---------------------------------------------------------------- pen-drawn fallbacks

def pen(strokes, seed: int = 0, res: float = 0.5) -> RecordingPen:
    """Render polylines (x-height units, baseline 0) as a pen stroke, then trace it.

    Points get Chaikin smoothing plus a little positional noise so fallbacks don't look geometric.
    """
    rng = np.random.default_rng(seed)
    pts_units = []
    for s in strokes:
        p = np.array(s, float) * X_HEIGHT
        p[:, 0] += p[:, 1] * SLANT
        if len(p) > 2:
            for _ in range(3):  # Chaikin corner cutting
                q = 0.75 * p[:-1] + 0.25 * p[1:]
                r = 0.25 * p[:-1] + 0.75 * p[1:]
                p = np.vstack([p[:1], np.column_stack([q, r]).reshape(-1, 2), p[-1:]])
        p += rng.normal(0, 0.012 * X_HEIGHT, p.shape)
        pts_units.append(p)
    allp = np.vstack(pts_units)
    pad = STROKE_UNITS * 2
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad
    W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    img = np.zeros((H, W), np.uint8)
    for p in pts_units:
        px = np.column_stack([(p[:, 0] - x0) / res, (y1 - p[:, 1]) / res])
        cv2.polylines(img, [np.round(px * 16).astype(np.int32)], False, 255,
                      int(round(STROKE_UNITS / res)), cv2.LINE_AA, shift=4)
        if len(p) == 1 or np.ptp(p, 0).max() < 1e-6:  # dot
            c = tuple(np.round(px[0]).astype(int))
            cv2.circle(img, c, int(STROKE_UNITS * 0.75 / res), 255, -1, cv2.LINE_AA)
    ink = img > 127
    return trace_bitmap(ink, res, y1 / res, -x0 / res)


def circle(cx, cy, r, n=14, start=0.0):
    t = np.linspace(start, start + 2 * np.pi, n)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


DOT = [[(0.0, 0.0), (0.02, 0.01)]]
SKELETONS = {
    "i": [[(0.08, 0.0), (0.12, 0.95)], [(0.16, 1.42), (0.18, 1.44)]],
    "j": [[(0.35, 0.95), (0.3, -0.45), (0.12, -0.72), (-0.15, -0.55)], [(0.4, 1.42), (0.42, 1.44)]],
    "x": [[(0.0, 0.0), (0.72, 1.0)], [(0.05, 1.0), (0.7, 0.0)]],
    "n": [[(0.02, 0.0), (0.05, 0.95), (0.06, 0.6), (0.3, 0.95), (0.55, 0.85), (0.58, 0.3), (0.62, 0.0)]],
    "y": [[(0.0, 1.0), (0.12, 0.3), (0.35, 0.1), (0.62, 0.45)], [(0.7, 1.0), (0.55, -0.4), (0.3, -0.72), (0.02, -0.55)]],
    "v": [[(0.0, 1.0), (0.32, 0.0), (0.7, 1.05)]],
    "o": [circle(0.35, 0.5, 0.42)],
    "H": [[(0.05, 0.0), (0.08, 1.75)], [(0.8, 0.0), (0.83, 1.75)], [(0.0, 0.92), (0.9, 0.98)]],
    "J": [[(0.0, 1.75), (0.95, 1.8)], [(0.55, 1.78), (0.5, 0.25), (0.28, 0.0), (0.0, 0.3)]],
    "K": [[(0.05, 0.0), (0.08, 1.75)], [(0.8, 1.7), (0.12, 0.85), (0.85, 0.0)]],
    "U": [[(0.0, 1.75), (0.02, 0.45), (0.38, 0.0), (0.75, 0.45), (0.78, 1.75)]],
    "Z": [[(0.0, 1.7), (0.88, 1.75), (0.0, 0.02), (0.92, 0.06)]],
    ",": [[(0.12, 0.08), (0.12, -0.05), (0.0, -0.3)]],
    "-": [[(0.0, 0.48), (0.5, 0.53)]],
    '"': [[(0.12, 1.75), (0.06, 1.38)], [(0.42, 1.75), (0.36, 1.38)]],
    "'": [[(0.12, 1.75), (0.06, 1.38)]],
    ".": DOT,
    ":": [[(0.05, 0.02), (0.07, 0.03)], [(0.1, 0.82), (0.12, 0.83)]],
    ";": [[(0.1, 0.82), (0.12, 0.83)], [(0.1, 0.08), (0.1, -0.05), (0.0, -0.3)]],
    "!": [[(0.12, 1.75), (0.08, 0.45)], [(0.05, 0.02), (0.07, 0.03)]],
    "?": [[(0.0, 1.4), (0.3, 1.75), (0.65, 1.5), (0.35, 0.9), (0.32, 0.5)], [(0.3, 0.02), (0.32, 0.03)]],
    "&": [[(0.95, 0.0), (0.15, 0.95), (0.2, 1.55), (0.5, 1.65), (0.6, 1.35), (0.0, 0.55), (0.12, 0.02),
           (0.55, 0.05), (0.95, 0.65)]],
    "%": [[(0.0, 0.0), (1.0, 1.75)], circle(0.2, 1.45, 0.2), circle(0.85, 0.3, 0.2)],
    "(": [[(0.45, 1.85), (0.08, 0.9), (0.4, -0.3)]],
    ")": [[(0.0, 1.85), (0.37, 0.9), (0.05, -0.3)]],
    "/": [[(0.0, -0.1), (0.75, 1.85)]],
    "$": [[(0.8, 1.45), (0.4, 1.6), (0.05, 1.3), (0.45, 0.85), (0.85, 0.4), (0.45, 0.05), (0.0, 0.25)],
          [(0.45, 1.9), (0.42, -0.2)]],
}
CAP_FROM_LOWER = set("CKOPSUVWXZ")


def fill_missing(placed: dict) -> dict:
    """Add every REQUIRED char that has no real sample. Returns {char: source} for the audit log."""
    source = {ch: "journal" for ch in placed}
    cap_scale = 1.6

    def first(ch):
        return placed[ch][0] if placed.get(ch) else None

    for ch in REQUIRED:
        if placed.get(ch):
            continue
        made, how = None, None
        if ch.isupper() and ch in CAP_FROM_LOWER and placed.get(ch.lower()):
            made = [transformed(r, (cap_scale, 0, 0, cap_scale, 0, 0)) for r in placed[ch.lower()]]
            how = f"scaled from your '{ch.lower()}'"
        elif ch == "p" and placed.get("P"):
            made = [transformed(r, (0.9, 0, 0, 0.9, 0, -0.62 * X_HEIGHT)) for r in placed["P"]]
            how = "your 'P', lowered onto the descender"
        elif ch in "qg" and placed.get("a"):
            a = first("a")
            x0, _, x1, _ = bounds(a)
            stem = [(x1 / X_HEIGHT - 0.08, 0.95), (x1 / X_HEIGHT - 0.12, -0.7)]
            if ch == "g":
                stem = stem[:1] + [(x1 / X_HEIGHT - 0.12, -0.45), (x1 / X_HEIGHT - 0.4, -0.75),
                                   (x1 / X_HEIGHT - 0.75, -0.6)]
            made = [merged(r, pen([[(s[0] - SLANT * s[1], s[1]) for s in stem]], seed=ord(ch) + i))
                    for i, r in enumerate(placed["a"][:2])]
            how = "your 'a' + drawn descender"
        elif ch in SKELETONS:
            made = [pen(SKELETONS[ch], seed=ord(ch) * 7 + k) for k in range(2)]
            how = "pen-drawn at your stroke width"
        if made:
            placed[ch] = made
            source[ch] = how
        else:
            source[ch] = "MISSING"
    return source


def add_dots(placed, dotless):
    """'?' and '!' whose dot fell outside the crop get one drawn under the stroke."""
    for ch in NEEDS_DOT:
        for i, need in enumerate(dotless.get(ch, [])):
            if not need:
                continue
            r = placed[ch][i]
            x0, y0, x1, _ = bounds(r)
            # stroke bottom sits ~0.42 x-height above the baseline; centre the dot under its lower end
            d = pen(DOT, seed=i)
            dx0, _, dx1, _ = bounds(d)
            cx = x0 + 0.35 * (x1 - x0)
            placed[ch][i] = merged(r, transformed(d, (1, 0, 0, 1, cx - (dx0 + dx1) / 2, 0)))


# ---------------------------------------------------------------- variation

def perturb(rec: RecordingPen, seed: int) -> RecordingPen:
    """Small affine wobble around the glyph's baseline centre: rotation, aspect, slant."""
    rng = np.random.default_rng(seed)
    x0, _, x1, _ = bounds(rec)
    cx = (x0 + x1) / 2
    th = math.radians(rng.uniform(-2.5, 2.5))
    sx, sy = rng.uniform(0.95, 1.05), rng.uniform(0.95, 1.05)
    sh = rng.uniform(-0.05, 0.05)
    a, b = sx * math.cos(th), sx * math.sin(th)
    c, d = -sy * math.sin(th) + sh * sy, sy * math.cos(th)
    return transformed(rec, (a, b, c, d, cx - a * cx, -b * cx))


def forms_for(samples: list[RecordingPen], ch: str) -> list[RecordingPen]:
    forms = list(samples[:N_FORMS])
    while len(forms) < N_FORMS:
        forms.append(perturb(samples[len(forms) % len(samples)], seed=ord(ch) * 31 + len(forms)))
    return forms


# ---------------------------------------------------------------- font assembly

def build(glyph_files: list[Path], out_path: Path, family: str = "My Hand"):
    records = [r for f in glyph_files for r in np.load(f, allow_pickle=True)]
    placed, dotless = place(records)
    add_dots(placed, dotless)
    source = fill_missing(placed)
    chars = sorted(ch for ch in placed if placed[ch])

    order = [".notdef", "space", "space.alt1", "space.alt2"]
    charstrings, metrics, cmap = {}, {}, {0x20: "space"}

    def add(name, rec: RecordingPen | None, advance: float):
        p = T2CharStringPen(round(advance), None)
        lsb = 0
        if rec is not None:
            rec.replay(p)
            lsb = bounds(rec)[0]
        charstrings[name] = p.getCharString()
        metrics[name] = (round(advance), round(lsb))

    add(".notdef", None, SPACE_WIDTH)
    for name, f in zip(["space", "space.alt1", "space.alt2"], [1.0, 0.88, 1.12]):
        add(name, None, SPACE_WIDTH * f)

    pua, pua_map = PUA_BASE, {}
    for ch in chars:
        base = glyph_name(ch)
        for i, rec in enumerate(forms_for(placed[ch], ch)):
            name = base if i == 0 else f"{base}.alt{i}"
            x0, _, x1, _ = bounds(rec)
            shifted = transformed(rec, (1, 0, 0, 1, SIDE_BEARING - x0, 0))
            add(name, shifted, (x1 - x0) + 2 * SIDE_BEARING)
            order.append(name)
            if i == 0:
                cmap[ord(ch)] = name
            cmap[pua] = name
            pua_map.setdefault(ch, []).append(pua)
            pua += 1

    fb = FontBuilder(UPM, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    ps = family.replace(" ", "") + "-Regular"
    fb.setupCFF(ps, {"FullName": family}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=-DESCENT)
    fb.setupNameTable({"familyName": family, "styleName": "Regular", "psName": ps,
                       "fullName": family, "uniqueFontIdentifier": ps + "-1.0", "version": "Version 1.000"})
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=-DESCENT, sTypoLineGap=200,
                usWinAscent=ASCENT, usWinDescent=DESCENT, sxHeight=X_HEIGHT, sCapHeight=int(CAP), fsType=0)
    fb.setupPost()

    # calt cycle: default -> alt1 -> alt2 -> default, keyed on the previous glyph's form
    bases = ["space"] + [glyph_name(c) for c in chars]
    fea = f"""
@BASE = [{' '.join(bases)}];
@ALT1 = [{' '.join(b + '.alt1' for b in bases)}];
@ALT2 = [{' '.join(b + '.alt2' for b in bases)}];
feature calt {{
  lookup cycle {{
    sub @ALT1 @BASE' by @ALT2;
    sub @BASE @BASE' by @ALT1;
  }} cycle;
}} calt;
"""
    addOpenTypeFeaturesFromString(fb.font, fea)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_path))
    out_path.with_suffix(".pua.json").write_text(json.dumps(pua_map, ensure_ascii=False))
    counts = defaultdict(int)
    for r in records:
        counts[r["char"]] += 1
    templ = {r["char"] for r in records if r.get("source") == "template"}
    audit = {ch: {"source": "template" if ch in templ else source.get(ch, "journal"),
                  "real_samples": counts.get(ch, 0)} for ch in REQUIRED}
    out_path.with_name("coverage.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    real = sum(1 for v in audit.values() if v["real_samples"])
    print(f"wrote {out_path}: {len(chars)} chars x {N_FORMS} forms; {real}/{len(REQUIRED)} from your handwriting")
    for ch, v in audit.items():
        if not v["real_samples"]:
            print(f"  {ch!r}: {v['source']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a handwriting font from extracted glyphs")
    ap.add_argument("glyphs", nargs="+", type=Path, help="glyph .npy files (journal and/or template)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .otf")
    ap.add_argument("--family", default="My Hand")
    a = ap.parse_args()
    build(a.glyphs, a.out, a.family)
