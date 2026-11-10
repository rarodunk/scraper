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


def detect_text_lines(dark: np.ndarray, paper: np.ndarray):
    """Baselines of handwritten lines on unruled paper (cards). Returns (rules, line_map).

    Lines are chained left to right: each letter-sized ink component joins the open line whose
    trend (fit through its recent members) passes closest to it, so lines may drift and curve.
    Fragments separated by wide gaps are merged when their trends meet. Small marks (dots,
    commas, quotes) join the line whose baseline is the first one at or below them.
    The baseline of each line is a robust fit through column-wise ink bottoms with descenders
    trimmed. line_map labels every member ink pixel with rid + 1, so strips can keep exactly
    the components chained to their line (a word that dips toward the next line stays put).
    """
    ink = ((dark > 45) & (paper > 0)).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 10:
        return [], np.zeros(dark.shape, np.uint16)
    X, Y = st[:, cv2.CC_STAT_LEFT].astype(float), st[:, cv2.CC_STAT_TOP].astype(float)
    W, H = st[:, cv2.CC_STAT_WIDTH].astype(float), st[:, cv2.CC_STAT_HEIGHT].astype(float)
    A = st[:, cv2.CC_STAT_AREA].astype(float)
    cx, cy = cen[:, 0], cen[:, 1]
    valid = np.arange(n) > 0
    xh = float(np.percentile(H[valid & (A >= 15) & (H >= 5)], 35))
    anchor = valid & (H >= 0.45 * xh) & (H <= 3.5 * xh) & (W <= 12 * xh) & (A >= 15)
    small = valid & ~anchor & (A >= 6) & (H < 0.9 * xh) & (W < 2 * xh)
    # each anchor's own baseline estimate: a low-ish percentile of its column-wise ink bottoms, so
    # ascenders don't lift it and descenders (a minority of columns) don't drag it down
    base = cy.copy()
    for i in np.where(anchor)[0]:
        x, y, w, h = st[i, :4]
        sub = lab[y:y + h, x:x + w] == i
        bottoms = h - 1 - np.argmax(sub[::-1], axis=0)
        bottoms = bottoms[sub.any(0)]
        base[i] = y + float(np.percentile(bottoms, 40))

    # seed lines: paint each anchor's x-height body band, smear sideways so neighbouring words on a
    # line fuse; lines are ~1.3 x-heights apart at the body, so the smeared bands stay separate
    body = np.zeros(dark.shape, np.uint8)
    for i in np.where(anchor)[0]:
        y0 = int(base[i] - 0.75 * xh)
        y1 = int(base[i] - 0.2 * xh)
        body[max(0, y0):max(0, y1), int(X[i]):int(X[i] + W[i])] = 1
    body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * xh) | 1, 1)))
    nb, blab = cv2.connectedComponents(body, connectivity=8)
    seeds: dict[int, list[int]] = {}
    for i in np.where(anchor)[0]:
        yb = int(np.clip(base[i] - 0.45 * xh, 0, dark.shape[0] - 1))
        xs = slice(int(X[i]), int(X[i] + W[i]))
        ids = blab[yb, xs]
        ids = ids[ids > 0]
        if len(ids):
            seeds.setdefault(int(np.bincount(ids).argmax()), []).append(i)
    lines = [sorted(L, key=lambda i: cx[i]) for L in seeds.values()]

    # refinement: chains that follow the same trend are one line (a capital or a descender word can
    # start a parallel chain); then every anchor is reassigned to the nearest fitted line
    anchors = np.where(anchor)[0]

    def fit(L):
        xs, ys = cx[L], base[L]
        deg = 2 if (len(L) >= 8 and np.ptp(xs) > 25 * xh) else (1 if len(L) >= 3 and np.ptp(xs) > 3 * xh else 0)
        keep = np.ones(len(L), bool)
        for _ in range(2):
            coef = np.polyfit(xs[keep], ys[keep], deg)
            keep = np.abs(ys - np.polyval(coef, xs)) < 0.6 * xh
            if keep.sum() <= deg + 1:
                break
        return coef, xs.min(), xs.max()

    for _ in range(4):
        lines = [L for L in lines if L]
        fits = [fit(L) for L in lines]
        changed = True
        while changed:
            changed = False
            for a_i in range(len(lines)):
                for b_i in range(len(lines)):
                    if a_i == b_i or not lines[a_i] or not lines[b_i]:
                        continue
                    (ca, a0, a1), (cb, b0, b1) = fits[a_i], fits[b_i]
                    lo_, hi_ = max(a0, b0), min(a1, b1)
                    if hi_ > lo_:
                        xs = np.linspace(lo_, hi_, 8)
                    elif lo_ - hi_ < 30 * xh:
                        xs = np.array([lo_, hi_])
                    else:
                        continue
                    if np.median(np.abs(np.polyval(ca, xs) - np.polyval(cb, xs))) < 0.6 * xh:
                        lines[a_i] = sorted(lines[a_i] + lines[b_i], key=lambda i: cx[i])
                        lines[b_i] = []
                        fits[a_i] = fit(lines[a_i])
                        changed = True
        keep_k = [k for k, L in enumerate(lines) if L]
        fits = [fits[k] for k in keep_k]
        new = [[] for _ in fits]
        for i in anchors:
            best = None
            for k, (c, x0, x1) in enumerate(fits):
                if x0 - 4 * xh <= cx[i] <= x1 + 4 * xh:
                    d = abs(base[i] - np.polyval(c, cx[i]))
                    if d < 1.0 * xh and (best is None or d < best[0]):
                        best = (d, k)
            if best:
                new[best[1]].append(i)
        lines = [sorted(L, key=lambda i: cx[i]) for L in new if len(L) >= 3]
    if not lines:
        return [], np.zeros(dark.shape, np.uint16)

    rules, members_of = [], []
    for L in lines:
        xs_pts, ys_pts = [], []
        step = max(1, int(xh / 3))
        for i in L:
            x, y, w, h = st[i, :4]
            sub = lab[y:y + h, x:x + w] == i
            for c in range(0, w, step):
                col = np.where(sub[:, c])[0]
                if len(col):
                    xs_pts.append(x + c)
                    ys_pts.append(y + col.max())
        xs_pts, ys_pts = np.array(xs_pts, float), np.array(ys_pts, float)
        deg = 2 if np.ptp(xs_pts) > 25 * xh else 1
        keep = np.ones(len(xs_pts), bool)
        for _ in range(4):
            coef = np.polyfit(xs_pts[keep], ys_pts[keep], deg)
            res = ys_pts - np.polyval(coef, xs_pts)
            med = np.median(res[keep])
            keep = (res < med + 0.25 * xh) & (res > med - 0.8 * xh)
        # column bottoms under arches and crossbars (n m h r T f) sit above the baseline and pull the
        # fit up; the true baseline is the most common bottom level at or below the fit (round
        # letters), while descender bottoms are spread out and don't form a peak
        res = ys_pts - np.polyval(coef, xs_pts)
        below = res[(res >= -0.1 * xh) & (res <= 0.7 * xh)]
        if len(below) >= 5:
            hist, edges = np.histogram(below, bins=np.arange(-0.1 * xh, 0.7 * xh + 1e-6, 0.08 * xh))
            k = int(np.argmax(hist))
            coef = coef.copy()
            coef[-1] += (edges[k] + edges[k + 1]) / 2
        r = Rule(0, float(xs_pts.min() - 0.5 * xh), float(xs_pts.max() + 0.5 * xh), coef)
        r.xh = xh
        rules.append(r)
        members_of.append(list(L))
    order = sorted(range(len(rules)), key=lambda k: rules[k].y((rules[k].x0 + rules[k].x1) / 2))
    rules = [rules[k] for k in order]
    members_of = [members_of[k] for k in order]
    for k, r in enumerate(rules):
        r.rid = k

    # small marks: first baseline at or below them, within reach
    for i in np.where(small)[0]:
        best = None
        for k, r in enumerate(rules):
            if not (r.x0 - xh <= cx[i] <= r.x1 + xh):
                continue
            d = r.y(cx[i]) - (cy[i] - 0.3 * xh)
            if 0 <= d < 2.5 * xh and (best is None or d < best[0]):
                best = (d, k)
        if best:
            members_of[best[1]].append(i)

    lut = np.zeros(n, np.uint16)
    for k, mem in enumerate(members_of):
        lut[mem] = k + 1
    return rules, lut[lab]


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
    unruled: bool = False   # True: the "rule" is a fitted text baseline, not printed ruling
    line_map: np.ndarray | None = field(default=None, repr=False)  # unruled: page line ids in strip space


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
        if strip.unruled:
            # keep components whose ink was chained to this line on the full page (see detect_text_lines)
            ids = strip.line_map[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]
            ids = ids[ids > 0]
            if not len(ids) or np.bincount(ids).argmax() != strip.rid + 1:
                continue
        else:
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
