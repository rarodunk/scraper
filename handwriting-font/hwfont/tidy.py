"""Legibility pass: redraw a glyph as one even pen stroke along its own centerline.

Raw traces carry everything the photo did: ink pooling at stroke starts, fuzzy edges, pressure
blobs that close counters, and stray tails from neighbouring letters. The tidy pass keeps the
*path* of the hand (which is what makes it yours) and replaces the ink:

  1. skeletonise the ink to a 1-px centerline
  2. split it into strokes at junctions and endpoints
  3. drop short spurs (pen-entry hooks and blob artefacts), smooth each stroke
  4. optionally shear toward a common slant (strokes keep their own shape, only the lean changes)
  5. re-render with a round pen of fixed width and trace that

All coordinates here are in the bitmap's pixel space; the caller converts to font units.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(sk: np.ndarray, y: int, x: int):
    h, w = sk.shape
    for dy, dx in _NB:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
            yield yy, xx


def strokes(ink: np.ndarray) -> list[np.ndarray]:
    """Centerline of `ink` as a list of (N, 2) float arrays of (x, y) points.

    Nodes are skeleton pixels whose neighbour count is not 2 (ends and junctions); a stroke is the
    pixel chain between two nodes. Closed loops with no node (an 'o') are walked separately.
    """
    sk = skeletonize(ink > 0)
    deg = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT).astype(int) - 1
    deg[~sk] = 0
    nodes = set(zip(*np.where(sk & (deg != 2))))
    seen_px = set()
    seen_edge = set()
    out = []
    for n in nodes:
        if deg[n] == 0:  # isolated pixel: a dot
            out.append(np.array([[n[1], n[0]]], float))
            seen_px.add(n)
            continue
        for q in _neighbours(sk, *n):
            if (n, q) in seen_edge:
                continue
            path = [n, q]
            seen_edge.add((n, q))
            prev, cur = n, q
            while cur not in nodes:
                seen_px.add(cur)
                nxt = [p for p in _neighbours(sk, *cur) if p != prev and p not in path[-3:]]
                if not nxt:
                    break
                # prefer the 4-connected step to avoid cutting staircase corners
                nxt.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
                prev, cur = cur, nxt[0]
                path.append(cur)
            seen_edge.add((cur, prev))
            out.append(np.array([[x, y] for y, x in path], float))
        seen_px.add(n)
    # remaining unvisited pixels form node-free loops
    rest = set(zip(*np.where(sk))) - seen_px
    while rest:
        start = rest.pop()
        path, prev, cur = [start], None, start
        while True:
            nxt = [p for p in _neighbours(sk, *cur) if p != prev and p in rest]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            rest.discard(cur)
            path.append(cur)
        if len(path) > 2:
            path.append(start)
        out.append(np.array([[x, y] for y, x in path], float))
    return out


def prune(paths: list[np.ndarray], ink: np.ndarray, min_len: float) -> list[np.ndarray]:
    """Drop short dangling strokes (one free end) that are spurs off a longer stroke."""
    if len(paths) <= 1:
        return paths
    ends = {}
    for p in paths:
        for e in (tuple(p[0].round().astype(int)), tuple(p[-1].round().astype(int))):
            ends[e] = ends.get(e, 0) + 1

    def near_other(pt, self_idx):
        for j, p in enumerate(paths):
            if j == self_idx:
                continue
            if np.min(np.hypot(*(p - pt).T)) <= 2.5:
                return True
        return False

    keep = []
    for i, p in enumerate(paths):
        length = float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0
        a_free = not near_other(p[0], i)
        b_free = not near_other(p[-1], i)
        attached = not (a_free and b_free)
        if attached and (a_free or b_free) and length < min_len:
            continue  # spur
        keep.append(p)
    return keep


def smooth(p: np.ndarray, win: int) -> np.ndarray:
    """Moving-average smoothing with fixed endpoints (so strokes still meet at junctions)."""
    if len(p) < 5 or win < 3:
        return p
    closed = len(p) > 3 and np.allclose(p[0], p[-1])
    k = np.ones(win) / win
    if closed:
        ext = np.vstack([p[-win:-1], p, p[1:win]])
        sm = np.column_stack([np.convolve(ext[:, i], k, mode="same") for i in range(2)])[win - 1:-(win - 1)]
        return sm
    pad = win // 2
    ext = np.vstack([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)])
    sm = np.column_stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)])
    sm[0], sm[-1] = p[0], p[-1]
    return sm


def centerline(ink: np.ndarray, pen_px: float) -> list[np.ndarray]:
    """Pruned centerline strokes of a glyph bitmap.

    Spurs are judged against the ink's own stroke width, not the output pen: skeleton spurs from
    blobby ink are about one stroke long, while real short strokes (t/f crossbars, I serifs) are
    longer than that even when the output pen is bold."""
    from .build_font import stroke_width_px
    own = stroke_width_px(ink > 0) or pen_px
    return prune(strokes(ink), ink, 1.2 * min(own, pen_px))


def slant(paths: list[np.ndarray], step: int = 6) -> tuple[float, float]:
    """Dominant lean as dx/dy (positive = leaning right) from near-vertical stroke segments,
    plus the total length of those segments (how much evidence the estimate rests on)."""
    tans, wts = [], []
    for p in paths:
        q = p[::step] if len(p) > step else p
        if len(q) < 2:
            continue
        d = np.diff(q, axis=0)
        dy = -d[:, 1]  # image y points down
        dx = d[:, 0]
        ln = np.hypot(dx, dy)
        ok = (ln > 0) & (np.abs(dx) < np.abs(dy) * math.tan(math.radians(35)))
        sign = np.sign(dy[ok])
        tans += list(dx[ok] * sign / (dy[ok] * sign))
        wts += list(ln[ok])
    if sum(wts) < 1:
        return 0.0, 0.0
    order = np.argsort(tans)
    cw = np.cumsum(np.array(wts)[order])
    return float(np.array(tans)[order][np.searchsorted(cw, cw[-1] / 2)]), float(cw[-1])


def restroke(paths: list[np.ndarray], shape, pen_px: float, baseline_px: float, shear: float = 0.0):
    """Redraw centerline `paths` with a round pen of width pen_px.

    Returns (bitmap, y_offset): original row y is row y + y_offset in the bitmap.
    shear: extra dx/dy applied about the baseline (positive leans further right).
    """
    ink = np.zeros(shape, bool)
    win = max(3, int(round(pen_px * 0.9)) | 1)
    paths = [smooth(p, win) for p in paths]
    if shear:
        paths = [np.column_stack([p[:, 0] + shear * (baseline_px - p[:, 1]), p[:, 1]]) for p in paths]
    allp = np.vstack(paths) if paths else np.zeros((1, 2))
    pad = int(pen_px * 2) + 4
    x0 = int(math.floor(min(allp[:, 0].min(), 0))) - pad
    W = int(math.ceil(max(allp[:, 0].max(), ink.shape[1]))) - x0 + pad
    H = ink.shape[0] + 2 * pad
    canvas = np.zeros((H, W), np.uint8)
    t = max(1, int(round(pen_px)))
    S = 16  # sub-pixel precision for cv2 drawing
    for p in paths:
        pts = np.round((p - [x0, -pad]) * S).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0]), int(pen_px * 0.65 * S), 255, -1, cv2.LINE_AA, shift=4)
            continue
        cv2.polylines(canvas, [pts], False, 255, t, cv2.LINE_AA, shift=4)
        for e in (pts[0], pts[-1]):  # round caps
            cv2.circle(canvas, tuple(e), int(pen_px / 2 * S), 255, -1, cv2.LINE_AA, shift=4)
    # original pixel (x, y) is at (x - x0, y + pad) in the returned bitmap
    return canvas > 127, pad


def _free(p_end: np.ndarray, paths: list[np.ndarray], self_idx: int) -> bool:
    for j, q in enumerate(paths):
        if j != self_idx and np.min(np.hypot(*(q - p_end).T)) <= 2.5:
            return False
    return True


def core_range(paths: list[np.ndarray], xh_px: float, bin_px: float):
    """x-range of the letter's body: columns where the centerline spans >= 0.3 x-height vertically.

    Joining strokes are one thin near-horizontal line, so their columns have almost no vertical
    extent; loops, stems and bowls do. Returns None when no column qualifies."""
    pts = np.vstack(paths)
    b = np.floor(pts[:, 0] / bin_px).astype(int)
    lo, hi = {}, {}
    for bi, y in zip(b, pts[:, 1]):
        lo[bi] = min(lo.get(bi, y), y)
        hi[bi] = max(hi.get(bi, y), y)
    core = [bi for bi in lo if hi[bi] - lo[bi] >= 0.3 * xh_px]
    if not core:
        return None
    return min(core) * bin_px, (max(core) + 1) * bin_px


def trim_connectors(paths: list[np.ndarray], xh_px: float, keep: float = 0.12,
                    max_angle: float = 50.0) -> list[np.ndarray]:
    """Shorten joining strokes: free stroke ends that run sideways out past the letter's body.

    Joined handwriting carries lead-in/lead-out strokes into the neighbouring letters; set as type
    they read as stray dashes ("s-e-e"). From each free end, the part of the stroke lying outside
    the body's x-range (plus keep * x-height) is removed if that part runs within max_angle of
    horizontal. A single stroke that is the whole letter is protected because its own columns
    define the body.
    """
    paths = [p for p in paths if len(p)]
    if not paths:
        return paths
    rng = core_range(paths, xh_px, max(2.0, 0.04 * xh_px))
    if rng is None:
        return paths
    lo, hi = rng[0] - keep * xh_px, rng[1] + keep * xh_px
    tan_max = math.tan(math.radians(max_angle))
    out = [p.copy() for p in paths]
    for i in range(len(out)):
        for side in (0, 1):
            p = out[i]
            if len(p) < 3 or (len(p) > 3 and np.allclose(p[0], p[-1])):
                continue
            q = p if side == 0 else p[::-1]
            if not _free(q[0], out, i):
                continue
            outside = (q[:, 0] < lo) | (q[:, 0] > hi)
            k = 0
            while k < len(q) and outside[k]:
                k += 1
            if k < 2 or len(q) - k < 2:
                continue
            chord = q[k - 1] - q[0]
            if abs(chord[1]) > tan_max * abs(chord[0]):
                continue  # steep: a descender/ascender flourish, not a connector
            q = q[k:]
            out[i] = q if side == 0 else q[::-1]
    return out
