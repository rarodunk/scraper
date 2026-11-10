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

from . import legibility, tidy

# ---- font metrics (units per em = 1000)
UPM = 1000
X_HEIGHT = 330
CAP = 1.7 * X_HEIGHT             # nominal; real capitals keep the size they were written at
ASCENT, DESCENT = 880, 320
SIDE_BEARING = 0.05 * X_HEIGHT  # cards: 0.09 x-height median gap between separate letters   # tight: joined-hand writers leave little air between letters
SPACE_WIDTH = 0.68 * X_HEIGHT   # + side bearings = 0.78 x-height, the cards' median word gap
STROKE_UNITS = 0.175 * X_HEIGHT  # target pen width: cards measure 0.165-0.187 x-height (half-max width)
N_FORMS = 3
PUA_BASE = 0xE000
SLANT = math.tan(math.radians(6))  # forward lean applied to pen-drawn fallbacks

TRACE_UPSAMPLE = 2
TIERS = {"template": 0, "card": 1, "journal": 2}  # preference: deliberate writing beats journal scrawl
LEGIBLE_MAX = 4.2
SPLIT_PENALTY = 1.0
TWO_MARKS = set('":;')
# letters often misread as each other in this hand; a sample must look more like itself than these
CONFUSABLE = {"n": "ru", "r": "nv", "u": "nv", "a": "uo", "e": "c", "c": "e", "h": "nb", "o": "a",
              "v": "ru", "i": "l", "l": "i"}  # need both marks in the sample
USED_TIER: dict = {}  # char -> source tier of the samples actually used  # ranking handicap for letters cut out of joined words   # legibility.score above this = does not read as the letter (see legibility.py)
# Drawn when only journal samples pass the check: the journal m is a fast wave that reads as
# "w"/"vv" in running text. Card/template m samples are used normally.
FORCE_DRAWN = set("m")
# Always drawn: the card n carries an entry tick and a weak right leg; in running text it reads as
# r or "ʌ" ("and" -> "ard"). The drawn n is the same stem-and-arch shape without the tick.
ALWAYS_DRAWN = set("n")
ILLEGIBLE: set = set()  # chars whose every journal sample failed the legibility check
SCORES: dict = {}   # char -> legibility score of each placed sample, best first
SOURCES: dict = {}  # char -> label source of each placed sample, best first (written to forms.json)
TIDY = True        # legibility pass (hwfont/tidy.py); --raw turns it off for maximum fidelity
SLANT_KEEP = 0.35  # share of each glyph's own lean kept when evening out slant

X_LETTERS = set("acemnorsuvwxz")
# letters whose free horizontal ends are joining strokes, not part of the letter's shape
# (t/f/z crossbars, r arm and descender hooks are excluded on purpose)
CONNECTOR_TRIM = set("abcdehiklmnosuvwx")
DESCENDERS = set("gjpqy")
PUNCT_ANCHOR = {",": "comma", ";": "comma", "'": "cap_top", '"': "cap_top", "-": "mid", "=": "mid", "+": "mid"}
NEEDS_DOT = set("?!")

# Target glyph heights in x-heights. Samples come from pages written at different sizes and this
# hand has very tall ascender loops, so each glyph is pulled toward its class height, keeping
# HEIGHT_KEEP of its natural deviation (exponent) so the text still varies like real writing.
HEIGHT_KEEP = 0.2


def target_height(ch: str) -> float | None:
    """Total ink height in x-heights. Calibrated to the neat cards (hwfont.benchmark): ascenders and
    t 1.65, capitals 1.7, digits 1.5, descenders reaching 1.15 below the baseline."""
    if ch in X_LETTERS:
        return 1.0
    if ch == "t":
        return 1.65
    if ch in "bdfhkl":
        return 1.65
    if ch in "gjpqy":
        return 2.15
    if ch == "i":
        return 1.35
    if ch.isupper():
        return 1.7
    if ch.isdigit():
        return 1.5
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


def ink_bitmap(record, scale: float):
    """Threshold one sample (upsampled) at the level whose stroke width is closest to the target pen.

    Returns (ink, px_to_units, target_px)."""
    soft = record["soft"]
    up = cv2.resize(soft, None, fx=TRACE_UPSAMPLE, fy=TRACE_UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.2 * TRACE_UPSAMPLE)
    m = cv2.resize(record["mask"].astype(np.uint8), (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST)
    region = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    px_to_units = scale / TRACE_UPSAMPLE
    target_px = STROKE_UNITS / px_to_units
    if TIDY:
        # tidy mode re-renders the stroke, so only the centerline matters: take the thinnest ink that
        # does not break any stroke apart. Thin ink keeps loops open (filled e/a bowls skeletonise
        # to a point, which is the main source of illegible letters).
        def pieces(b):
            n, _, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
            return int((st[1:, cv2.CC_STAT_AREA] >= 30).sum())
        best = (up > 26) & region
        n0 = pieces(best)
        for thr in range(32, 150, 6):
            ink = (up > thr) & region
            if pieces(ink) > n0 or ink.sum() < 0.3 * best.sum():
                break
            best = ink
        return best, px_to_units, target_px
    best, best_err = (up > 40) & region, 1e9
    for thr in range(14, 96, 6):
        ink = (up > thr) & region
        w = stroke_width_px(ink)
        if w == 0:
            break
        err = abs(w - target_px)
        if err < best_err:
            best, best_err = ink, err
    return best, px_to_units, target_px


def match_width(ink: np.ndarray, target_px: float) -> np.ndarray:
    """Raw mode: dilate/erode toward the target pen width when thresholds alone could not reach it."""
    w = stroke_width_px(ink)
    delta = int(round((target_px - w) / 2))
    if delta > 0:
        ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * delta + 1,) * 2)) > 0
    elif delta < -1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -delta + 1,) * 2)
        thinner = cv2.erode(ink.astype(np.uint8), k) > 0
        if n_pieces(thinner) <= n_pieces(ink):  # never break a stroke apart
            ink = thinner
    return ink


def shape_signature(ink: np.ndarray, size: int = 40) -> np.ndarray:
    """Glyph normalised into a size x size box (aspect kept, centred), blurred for tolerant matching."""
    ys, xs = np.where(ink)
    if not len(ys):
        return np.zeros((size, size), np.float32)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    f = (size - 4) / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop / 255.0
    return cv2.GaussianBlur(out, (0, 0), 1.5)


def rank_typical(sigs: list[np.ndarray]) -> list[int]:
    """Order samples by typicality (mean distance to the other samples of the same character).

    The most typical instance is the one most likely to read as that letter; outliers are usually
    mis-segmentations or a letter written unusually fast."""
    n = len(sigs)
    if n <= 2:
        return list(range(n))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.abs(sigs[i] - sigs[j]).mean())
    return list(np.argsort(D.sum(1) / (n - 1)))


def place(records):
    """-> ({char: [RecordingPen]}, {char: [needs_dot]}) in font units, baseline at 0, best first."""
    rho, rho_global = page_rho([r for r in records if r.get("source") != "template"])
    # deliberate template writing beats journal scrawl: if a char has template samples, use only those
    templated = {r["char"] for r in records if r.get("source") == "template"}
    records = [r for r in records if r.get("source") == "template" or r["char"] not in templated]

    # pass 1: geometry, ink bitmap and (tidy mode) centerline + slant for every sample
    items = []
    for r in records:
        ch = r["char"]
        exact = "baseline_px" in r  # template boxes print the baseline and x-height
        xh_px = r["xh_px"] if exact else rho.get(r["image"], rho_global) * r["spacing_px"]
        scale = X_HEIGHT / xh_px
        top, bottom = r["top"], r["bottom"]
        tgt = target_height(ch)
        measured = (bottom - top) / xh_px
        # sanity: a sliver cut from a word can pass the shape check (a thin stroke "looks like" I);
        # drop samples whose height is implausible for the letter, and two-mark characters with one mark
        if tgt and not (0.5 * tgt <= measured <= 1.9 * tgt):
            continue
        if ch in TWO_MARKS and n_pieces(r["mask"]) < 2:
            continue
        if tgt:
            # deliberate writing (template, cards) is consistent: normalise gently
            keep = 0.6 if exact else (0.5 if r.get("source") == "card" else HEIGHT_KEEP)
            scale *= float(np.clip((tgt / measured) ** (1 - keep), 0.55, 1.7))
        anchor = PUNCT_ANCHOR.get(ch)
        missing_dot = ch in NEEDS_DOT and n_pieces(r["mask"]) == 1 and not exact
        line_base = r.get("source") == "card" and ch in DESCENDERS | set("f.,;:!?-'\"")
        if exact:
            baseline = r["baseline_px"]
        elif line_base:
            # cards have a fitted baseline that sits on the round letters (journal.detect_text_lines);
            # letters that hang below it (your f, g j p q y) keep their real depth
            baseline = r["v0"] - r["crop_y0"]
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
        ink, px_to_units, target_px = ink_bitmap(r, scale)
        it = dict(ch=ch, ink=ink, k=px_to_units, pen=target_px, base=baseline * TRACE_UPSAMPLE, dot=missing_dot,
                  tgt=tgt, exact=exact, line_base=line_base, keep=0.6 if exact else (0.5 if r.get("source") == "card" else HEIGHT_KEEP))
        it["tier"] = TIERS.get(r.get("source", "journal"), 2)
        # a letter cut out of a joined word (x-range label) is less reliable than a whole ink component
        it["split"] = not (exact or str(r.get("x0", "")).startswith("c"))
        it["pieces"] = n_pieces(r["mask"])
        it["src"] = f'{r["image"][:8]} R{r["rid"]} ' + (r["x0"] if isinstance(r["x0"], str) else f'{r["x0"]}-{r["x1"]}')
        if TIDY:
            it["paths"] = tidy.centerline(ink, target_px)
            if ch in CONNECTOR_TRIM:
                it["paths"] = tidy.trim_connectors(it["paths"], X_HEIGHT / px_to_units)  # x-height in bitmap px
            # sample the centerline every ~0.2 x-height so pixel staircases don't quantise the angle
            it["slant"], it["evidence"] = tidy.slant(it["paths"], step=max(6, int(0.2 * X_HEIGHT / px_to_units)))
            # only trust the slant of glyphs with real vertical strokes (not o, s, e, punctuation)
            it["evidence"] /= max(target_px, 1) * 8
        items.append(it)

    if TIDY:
        # slant of the most deliberate source available (cards/template), else everything
        top_tier = min(it["tier"] for it in items)
        votes = [it["slant"] for it in items if it["evidence"] >= 1 and it["ch"].isalnum() and it["tier"] == top_tier]
        if len(votes) < 20:
            votes = [it["slant"] for it in items if it["evidence"] >= 1 and it["ch"].isalnum()]
        target_slant = float(np.median(votes)) if votes else 0.0
        global SLANT
        SLANT = target_slant  # pen-drawn fallbacks lean like the writing
        print(f"tidy: writing slant {math.degrees(math.atan(target_slant)):+.1f} deg "
              f"(median of {len(votes)} glyphs), keeping {SLANT_KEEP:.0%} of each glyph's deviation")

    # pass 2: final bitmap per sample
    for it in items:
        if TIDY and it["paths"]:
            # re-measure on what will actually be drawn (connectors trimmed): baseline and size
            pts = np.vstack(it["paths"])
            xh_bmp = X_HEIGHT / it["k"]
            ch = it["ch"]
            if not it["exact"] and not it["line_base"] and ch.isalnum() and ch not in DESCENDERS:
                it["base"] = float(pts[:, 1].max() + it["pen"] / 2)
            if it["tgt"]:
                h = (pts[:, 1].max() - pts[:, 1].min() + it["pen"]) / xh_bmp
                f = float(np.clip((it["tgt"] / max(h, 1e-3)) ** (1 - it["keep"]), 0.6, 1.6))
                x0 = pts[:, 0].min()
                it["paths"] = [np.column_stack([x0 + (p[:, 0] - x0) * f, it["base"] + (p[:, 1] - it["base"]) * f])
                               for p in it["paths"]]
        if TIDY:
            shear = 0.0
            if it["evidence"] >= 1:
                shear = (target_slant - it["slant"]) * (1 - SLANT_KEEP)
            bmp, yoff = tidy.restroke(it["paths"], it["ink"].shape, it["pen"], it["base"], shear)
            it["final"], it["base_final"] = bmp, it["base"] + yoff
        else:
            it["final"], it["base_final"] = match_width(it["ink"], it["pen"]), it["base"]

    # pass 3: keep samples that read as their letter, best first, then trace
    by_char = defaultdict(list)
    for it in items:
        by_char[it["ch"]].append(it)
    out, dotless = defaultdict(list), defaultdict(list)
    SOURCES.clear(), SCORES.clear(), ILLEGIBLE.clear(), USED_TIER.clear()
    for ch, group in by_char.items():
        if ch.isalnum():
            dotted = ch in "ij" and any(it["pieces"] >= 2 for it in group)
            for it in group:
                paths = it["paths"] if TIDY else tidy.centerline(it["final"], it["pen"])
                it["score"] = legibility.score(paths, ch)
                # distinctiveness: an n whose right leg is weak scores fine as n but better as r
                rivals = [legibility.score(paths, o) for o in CONFUSABLE.get(ch, "")]
                it["margin"] = it["score"] - min(rivals) if rivals else -1.0
                if it["margin"] > 0.3:
                    it["score"] = max(it["score"], LEGIBLE_MAX + 0.01)  # reads as its rival: not legible
                # ranking only (the legibility cut-off still uses the raw score): prefer whole
                # components, and dotted i/j whenever a dotted sample exists
                it["rank"] = it["score"] + (SPLIT_PENALTY if it["split"] else 0.0) \
                    + 2.0 * max(0.0, it["margin"] + 0.3) \
                    + (2.0 if dotted and it["pieces"] < 2 else 0.0)
            group.sort(key=lambda it: it["rank"])
            legible = [it for it in group if it["score"] <= LEGIBLE_MAX]
            if dotted:  # an i or j without its dot is never used when a dotted sample exists
                legible = [it for it in legible if it["pieces"] >= 2] or legible
            if legible:  # use only the most deliberate source that has a legible sample
                best_tier = min(it["tier"] for it in legible)
                legible = [it for it in legible if it["tier"] == best_tier]
                if ch in ALWAYS_DRAWN or (ch in FORCE_DRAWN and best_tier == TIERS["journal"] and ch in SKELETONS):
                    legible = []  # journal form known to misread (see FORCE_DRAWN)
            if legible:
                group = legible
            elif ch in SKELETONS:
                ILLEGIBLE.add(ch)  # fill_missing draws it instead
                continue
            else:
                group = group[:1]  # nothing reads well and nothing to draw: vary the best one only
        else:
            group = [group[i] for i in rank_typical([shape_signature(it["final"]) for it in group])]
        USED_TIER[ch] = min(it["tier"] for it in group)
        for it in group:
            SOURCES.setdefault(ch, []).append(it["src"])
            SCORES.setdefault(ch, []).append(round(it.get("score", float("nan")), 2))
            xs = np.where(it["final"].any(0))[0]
            left = xs.min() if len(xs) else 0
            out[ch].append(trace_bitmap(it["final"], it["k"], it["base_final"], left))
            dotless[ch].append(it["dot"])
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
    "j": [[(0.35, 0.95), (0.3, -0.75), (0.12, -1.12), (-0.15, -0.9)], [(0.4, 1.42), (0.42, 1.44)]],
    "x": [[(0.0, 0.0), (0.72, 1.0)], [(0.05, 1.0), (0.7, 0.0)]],
    "a": [[(0.62, 0.8), (0.42, 1.0), (0.14, 0.9), (0.0, 0.5), (0.1, 0.08), (0.38, 0.02), (0.62, 0.35)],
          [(0.66, 1.0), (0.63, 0.2), (0.74, 0.0)]],
    "c": [[(0.6, 0.85), (0.38, 1.0), (0.1, 0.86), (0.0, 0.5), (0.1, 0.12), (0.35, 0.0), (0.62, 0.12)]],
    "e": [[(0.04, 0.48), (0.58, 0.58), (0.52, 0.88), (0.3, 1.0), (0.06, 0.82), (0.0, 0.4), (0.14, 0.08),
           (0.4, 0.0), (0.64, 0.15)]],
    "o": [[(0.38, 1.0), (0.1, 0.86), (0.0, 0.45), (0.14, 0.06), (0.4, 0.0), (0.62, 0.22), (0.66, 0.62),
           (0.52, 0.94), (0.3, 1.02)]],
    "u": [[(0.0, 1.0), (0.02, 0.3), (0.2, 0.0), (0.45, 0.05), (0.62, 0.45)], [(0.66, 1.0), (0.64, 0.15), (0.74, 0.0)]],
    "m": [[(0.0, 0.0), (0.02, 0.95)], [(0.03, 0.6), (0.2, 0.96), (0.37, 0.85), (0.39, 0.0)],
          [(0.4, 0.6), (0.57, 0.96), (0.75, 0.85), (0.78, 0.0)]],
    "r": [[(0.02, 0.0), (0.05, 0.95)], [(0.06, 0.55), (0.24, 0.92), (0.5, 0.96)]],
    "s": [[(0.55, 0.9), (0.3, 1.0), (0.06, 0.84), (0.14, 0.56), (0.44, 0.42), (0.6, 0.18), (0.36, 0.0), (0.0, 0.1)]],
    "Q": [circle(0.5, 0.88, 0.5, n=16, start=1.2), [(0.5, 0.35), (0.95, -0.05)]],
    "z": [[(0.02, 0.95), (0.66, 1.0), (0.02, 0.03), (0.7, 0.06)]],
    "w": [[(0.0, 1.0), (0.18, 0.0), (0.4, 0.78), (0.6, 0.0), (0.8, 1.0)]],
    "n": [[(0.02, 0.0), (0.05, 0.95), (0.06, 0.6), (0.3, 0.95), (0.55, 0.85), (0.58, 0.3), (0.62, 0.0)]],
    "y": [[(0.0, 1.0), (0.12, 0.3), (0.35, 0.1), (0.62, 0.45)], [(0.7, 1.0), (0.55, -0.7), (0.3, -1.12), (0.02, -0.9)]],
    "v": [[(0.0, 1.0), (0.32, 0.0), (0.7, 1.05)]],
    "o": [circle(0.35, 0.5, 0.42)],
    "I": [[(0.08, 1.7), (0.56, 1.72)], [(0.33, 1.71), (0.3, 0.02)], [(0.04, 0.0), (0.54, 0.02)]],
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
            stem = [(x1 / X_HEIGHT - 0.08, 0.95), (x1 / X_HEIGHT - 0.12, -1.12)]
            if ch == "g":
                stem = stem[:1] + [(x1 / X_HEIGHT - 0.12, -0.8), (x1 / X_HEIGHT - 0.4, -1.15),
                                   (x1 / X_HEIGHT - 0.75, -0.95)]
            made = [merged(r, pen([[(s[0] - SLANT * s[1], s[1]) for s in stem]], seed=ord(ch) + i))
                    for i, r in enumerate(placed["a"][:2])]
            how = "your 'a' + drawn descender"
        elif ch in SKELETONS:
            made = [pen(SKELETONS[ch], seed=ord(ch) * 7 + k) for k in range(2)]
            how = ("pen-drawn: samples misread in running text (ALWAYS_DRAWN)" if ch in ALWAYS_DRAWN
                   else "pen-drawn: no sample reads clearly" if ch in ILLEGIBLE
                   else "pen-drawn at your stroke width")
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
    tier_name = {v: k for k, v in TIERS.items()}
    audit = {ch: {"source": tier_name[USED_TIER[ch]] if source.get(ch) == "journal" and ch in USED_TIER
                  else ("template" if ch in templ else source.get(ch, "journal")),
                  "real_samples": counts.get(ch, 0)} for ch in REQUIRED}
    out_path.with_name("forms.json").write_text(json.dumps(
        {ch: (SOURCES.get(ch, []) + ["(perturbed copy)"] * N_FORMS)[:N_FORMS] for ch in chars}, indent=1, ensure_ascii=False))
    out_path.with_name("legibility.json").write_text(json.dumps(SCORES, indent=1, ensure_ascii=False))
    out_path.with_name("coverage.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    real = sum(1 for v in audit.values() if v["source"] in TIERS)
    by_src = defaultdict(int)
    for v in audit.values():
        by_src[v["source"] if v["source"] in TIERS else "derived/drawn"] += 1
    print(f"wrote {out_path}: {len(chars)} chars x {N_FORMS} forms; {real}/{len(REQUIRED)} from your handwriting "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_src.items()))})")
    for ch, v in audit.items():
        if v["source"] not in TIERS:
            print(f"  {ch!r}: {v['source']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a handwriting font from extracted glyphs")
    ap.add_argument("glyphs", nargs="+", type=Path, help="glyph .npy files (journal and/or template)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .otf")
    ap.add_argument("--family", default="My Hand")
    ap.add_argument("--raw", action="store_true", help="trace the ink as photographed (no tidy pass)")
    a = ap.parse_args()
    TIDY = not a.raw
    build(a.glyphs, a.out, a.family)
