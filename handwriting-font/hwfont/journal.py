"""Page analysis for ruled-notebook handwriting photos.

Pipeline per photo:
  1. darkness map  : background-normalised ink darkness (robust to uneven lighting)
  2. rule polylines: Hough segments of the printed ruling, clustered into one curve per rule
  3. strips        : each rule is unwarped into a straight horizontal strip (rule -> fixed row),
                     upsampled so glyph crops have enough pixels to vectorise cleanly
  4. components    : connected ink blobs inside each strip, with the rule pixels removed

Everything downstream (labelling sheets, glyph extraction) works in strip coordinates, where
the baseline rule is a known row and the local rule spacing gives a physical scale reference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold

UPSAMPLE = 4  # strip pixels per source pixel; source x-height is only ~15-25 px


# ---------------------------------------------------------------- darkness

def darkness_map(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (darkness, paper_mask). darkness = local paper brightness minus pixel brightness."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0]
    # dilate removes thin dark strokes, median smooths -> estimate of blank-paper brightness
    bg = cv2.medianBlur(cv2.dilate(L, np.ones((9, 9), np.uint8)), 41).astype(np.float32)
    dark = np.clip(bg - L.astype(np.float32), 0, 255)
    paper = cv2.erode((bg > 150).astype(np.uint8), np.ones((25, 25), np.uint8))
    return dark, paper


# ---------------------------------------------------------------- rules

@dataclass
class Rule:
    rid: int
    x0: float
    x1: float
    coef: np.ndarray  # polynomial y(x), highest power first

    def y(self, x):
        return np.polyval(self.coef, x)

    def dy(self, x):
        return np.polyval(np.polyder(self.coef), x)


def detect_rules(dark: np.ndarray, paper: np.ndarray) -> list[Rule]:
    binm = ((dark > 18) & (paper > 0)).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(binm, 1, np.pi / 720, threshold=250, minLineLength=350, maxLineGap=25)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    segs = segs[np.abs(ang) < 35]
    # normalise direction left->right
    flip = segs[:, 0] > segs[:, 2]
    segs[flip] = segs[flip][:, [2, 3, 0, 1]]
    n = len(segs)
    slope = (segs[:, 3] - segs[:, 1]) / np.maximum(segs[:, 2] - segs[:, 0], 1)

    def y_at(i, x):
        return segs[i, 1] + slope[i] * (x - segs[i, 0])

    # union-find: same rule if x-ranges overlap/abut, y agrees at the junction, slopes agree
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            lo = max(segs[i, 0], segs[j, 0])
            hi = min(segs[i, 2], segs[j, 2])
            if hi < lo - 60:  # more than 60 px apart horizontally
                continue
            xm = (lo + hi) / 2 if hi >= lo else (lo + hi) / 2
            if abs(y_at(i, xm) - y_at(j, xm)) < 5 and abs(np.degrees(np.arctan(slope[i]) - np.arctan(slope[j]))) < 2.5:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    rules = []
    for idx in groups.values():
        xs, ys = [], []
        for i in idx:
            t = np.linspace(segs[i, 0], segs[i, 2], max(2, int((segs[i, 2] - segs[i, 0]) / 10)))
            xs.append(t)
            ys.append(y_at(i, t))
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        span = xs.max() - xs.min()
        if span < 400:
            continue
        deg = 2 if span > 700 else 1
        coef = np.polyfit(xs, ys, deg)
        # reject fits that bend implausibly (gutter merges)
        if np.abs(np.polyval(coef, xs) - ys).max() > 6:
            coef = np.polyfit(xs, ys, 1)
        rules.append(Rule(len(rules), float(xs.min()), float(xs.max()), coef))
    rules.sort(key=lambda r: (r.y((r.x0 + r.x1) / 2)))
    for k, r in enumerate(rules):
        r.rid = k
    return rules


def local_spacing(rules: list[Rule], r: Rule) -> float:
    """Median vertical distance from rule r to its nearest neighbouring rules along its length."""
    xs = np.linspace(r.x0, r.x1, 15)
    d = []
    for x in xs:
        y0 = r.y(x)
        gaps = [abs(o.y(x) - y0) for o in rules if o is not r and o.x0 - 20 <= x <= o.x1 + 20]
        gaps = [g for g in gaps if g > 12]
        if gaps:
            d.append(min(gaps))
    return float(np.median(d)) if d else float("nan")


# ---------------------------------------------------------------- strips

@dataclass
class Strip:
    image_id: str
    rid: int
    spacing: float          # rule spacing in SOURCE px
    above: float            # strip extends this many spacings above the rule
    below: float
    dark: np.ndarray        # upsampled darkness, rule at row v0
    v0: int
    map_x: np.ndarray = field(repr=False)   # strip pixel -> source x (for provenance)
    map_y: np.ndarray = field(repr=False)


def make_strip(dark: np.ndarray, rule: Rule, spacing: float, image_id: str,
               above: float = 1.6, below: float = 0.8) -> Strip:
    """Unwarp the band around `rule` so the rule becomes row v0, sampling at UPSAMPLE x."""
    U = UPSAMPLE
    # arc-length parametrisation of the rule so letters are not stretched on sloped lines
    xs_fine = np.linspace(rule.x0, rule.x1, int((rule.x1 - rule.x0) * 2))
    ys_fine = rule.y(xs_fine)
    seg = np.hypot(np.diff(xs_fine), np.diff(ys_fine))
    arc = np.concatenate([[0], np.cumsum(seg)])
    W = int(arc[-1] * U)
    u = np.arange(W) / U
    px = np.interp(u, arc, xs_fine)
    py = rule.y(px)
    slope = rule.dy(px)
    norm = np.hypot(1, slope)
    nx, ny = -slope / norm, 1 / norm  # unit normal pointing "down" the page
    top, bot = above * spacing, below * spacing
    H = int((top + bot) * U)
    v = (np.arange(H) / U - top)[:, None]
    map_x = (px[None, :] + v * nx[None, :]).astype(np.float32)
    map_y = (py[None, :] + v * ny[None, :]).astype(np.float32)
    sd = cv2.remap(dark, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Strip(image_id, rule.rid, spacing, above, below, np.clip(sd, 0, 255), int(round(top * U)), map_x, map_y)


def rule_mask_source(shape, rules: list[Rule], thickness: int = 5) -> np.ndarray:
    """Draw every detected rule polyline into a source-resolution mask."""
    m = np.zeros(shape[:2], np.uint8)
    for r in rules:
        xs = np.arange(int(r.x0) - 30, int(r.x1) + 30, 4)
        pts = np.stack([xs, r.y(xs)], 1).round().astype(np.int32)
        cv2.polylines(m, [pts], False, 1, thickness)
    return m


def ink_mask(strip: Strip, rule_mask_src: np.ndarray) -> np.ndarray:
    """Binary ink with ruled lines removed (strokes crossing or sitting on a rule are preserved)."""
    d = cv2.GaussianBlur(strip.dark, (0, 0), 1.0)
    ink = apply_hysteresis_threshold(d, 20, 42).astype(np.uint8)
    lines = cv2.remap(rule_mask_src, strip.map_x, strip.map_y, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    off = ink & (1 - lines)
    # a rule pixel is kept only if ink continues just outside the rule band, above or below
    reach = int(2.0 * UPSAMPLE)
    band_up = cv2.dilate(lines, np.ones((reach, 1), np.uint8), anchor=(0, 0))       # band extended upward
    band_dn = cv2.dilate(lines, np.ones((reach, 1), np.uint8), anchor=(0, reach - 1))  # extended downward
    # propagate off-band ink into the band column-wise: dilate vertically by band height
    bh = int(7 * UPSAMPLE)
    sup_from_above = cv2.dilate(off & (band_up > 0), np.ones((bh, 1), np.uint8), anchor=(0, 0))
    sup_from_below = cv2.dilate(off & (band_dn > 0), np.ones((bh, 1), np.uint8), anchor=(0, bh - 1))
    keep = ink & lines & ((sup_from_above | sup_from_below) > 0)
    clean = (off | keep).astype(np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return clean


@dataclass
class Comp:
    cid: int
    x: int
    y: int
    w: int
    h: int
    area: int


def components(strip: Strip, ink: np.ndarray) -> tuple[np.ndarray, list[Comp]]:
    """Connected components that belong to this strip's text line."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    S = strip.spacing * UPSAMPLE
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 5 * UPSAMPLE * UPSAMPLE:  # below ~5 source px: speckle / rule residue
            continue
        # line assignment: the first rule below (top + 0.35*spacing) is this strip's rule
        t = y + 0.35 * S
        if not (strip.v0 - S < t <= strip.v0 + 0.05 * S):
            continue
        if y <= 1 or y + h >= ink.shape[0] - 1:
            continue  # clipped by strip border
        if h > 2.2 * S or w > 6 * S:
            continue  # page edges, underlines, scribbles
        out.append(Comp(i, int(x), int(y), int(w), int(h), int(a)))
    out.sort(key=lambda c: c.x)
    remap = {}
    for k, c in enumerate(out):
        remap[c.cid] = k + 1
        c.cid = k + 1
    lab2 = np.zeros_like(lab)
    for old, new in remap.items():
        lab2[lab == old] = new
    return lab2, out


# ---------------------------------------------------------------- page driver

def analyse_page(path: Path, image_id: str):
    img = cv2.imread(str(path))
    dark, paper = darkness_map(img)
    rules = detect_rules(dark, paper)
    rmask = rule_mask_source(dark.shape, rules)
    strips = []
    for r in rules:
        s = local_spacing(rules, r)
        if not np.isfinite(s) or s < 25 or s > 120:
            continue
        strips.append(make_strip(dark, r, s, image_id))
    return img, dark, rules, strips, rmask


# ---------------------------------------------------------------- over-segmentation

@dataclass
class Seg:
    sid: int
    cid: int
    x0: int
    x1: int
    y0: int
    y1: int


def segment(strip: Strip, lab: np.ndarray, comps: list[Comp]) -> tuple[np.ndarray, list[Seg]]:
    """Split joined handwriting at thin connector columns.

    For each component, the per-column ink count is a proxy for how many strokes pass through
    that column; cursive connectors are a single thin stroke, so cut at local minima whose count
    is close to one stroke width. Over-segmentation is intended: labels can merge segments.
    """
    S = strip.spacing * UPSAMPLE
    seglab = np.zeros(lab.shape, np.int32)
    segs: list[Seg] = []
    for c in comps:
        sub = (lab[c.y:c.y + c.h, c.x:c.x + c.w] == c.cid)
        cols = sub.sum(0).astype(float)
        # stroke width estimate: ink area / skeleton length ~ median vertical run of thin strokes
        runs = cols[cols > 0]
        stroke = max(2.0, np.percentile(runs, 25)) if len(runs) else 2.0
        cuts = [0]
        if c.w > 0.45 * S:
            sm = np.convolve(cols, np.ones(5) / 5, mode="same")
            min_w = int(0.16 * S)
            cand = [x for x in range(min_w, c.w - min_w)
                    if sm[x] <= 1.35 * stroke and sm[x] == sm[max(0, x - 6):x + 7].min()]
            for x in cand:
                if x - cuts[-1] >= min_w:
                    cuts.append(x)
            if c.w - cuts[-1] < min_w and len(cuts) > 1:
                cuts.pop()
        cuts.append(c.w)
        for a, b in zip(cuts[:-1], cuts[1:]):
            part = np.zeros_like(sub)
            part[:, a:b] = sub[:, a:b]
            if part.sum() < 20:
                continue
            ys = np.where(part.any(1))[0]
            sid = len(segs) + 1
            seglab[c.y:c.y + c.h, c.x:c.x + c.w][part] = sid
            segs.append(Seg(sid, c.cid, c.x + a, c.x + b, c.y + int(ys[0]), c.y + int(ys[-1]) + 1))
    return seglab, segs
