"""Legibility score: how much a glyph's centerline looks like the letter it is labelled as.

Fast journal writing collapses letters (an 'e' becomes '<', an 'o' a tick), and those collapsed
forms are often the *most common* form, so picking the most typical sample picks the illegible
one. Instead each sample's centerline is compared with the same character rendered in a few
reference typefaces (sans, serif, italic: italic gives single-storey a/g like handwriting), each
sheared over a range of slants. Score = symmetric chamfer distance between the two centerlines
after normalising both into the same box; lower reads better. Only shape is compared, never ink.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

REFERENCE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]
BOX = 64
SHEARS = [math.tan(math.radians(a)) for a in (-10, 0, 10, 20)]


def _normalise(pts: np.ndarray) -> np.ndarray:
    """(x, y) points -> same shape scaled so the longer bbox side spans BOX-8, centred."""
    lo, hi = pts.min(0), pts.max(0)
    span = max(float((hi - lo).max()), 1e-6)
    return (pts - (lo + hi) / 2) * ((BOX - 8) / span) + BOX / 2


def _raster(pts: np.ndarray) -> np.ndarray:
    img = np.zeros((BOX, BOX), np.uint8)
    p = np.clip(np.round(pts).astype(int), 0, BOX - 1)
    img[p[:, 1], p[:, 0]] = 1
    return img


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    da = cv2.distanceTransform((1 - a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (float(db[a > 0].mean()) + float(da[b > 0].mean()))


def _densify(paths: list[np.ndarray], step: float = 1.0) -> np.ndarray:
    out = []
    for p in paths:
        if len(p) == 1:
            out.append(p)
            continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
        out.append(p[-1:])
    return np.vstack(out)


@functools.lru_cache(maxsize=None)
def _references(ch: str) -> tuple:
    refs = []
    for fp in REFERENCE_FONTS:
        if not Path(fp).exists():
            continue
        font = ImageFont.truetype(fp, 160)
        img = Image.new("L", (360, 360), 0)
        ImageDraw.Draw(img).text((100, 60), ch, font=font, fill=255)
        ink = np.array(img) > 127
        if ink.sum() < 20:
            continue
        sk = skeletonize(ink)
        ys, xs = np.where(sk)
        pts = np.column_stack([xs, ys]).astype(float)
        base = ys.max()
        for s in SHEARS:  # lean the reference instead of the sample
            q = pts.copy()
            q[:, 0] += s * (base - q[:, 1])
            refs.append(_raster(_normalise(q)))
    return tuple(refs)


def score(paths: list[np.ndarray], ch: str) -> float:
    """Lower = reads more like `ch`. inf when there is nothing to compare."""
    paths = [p for p in paths if len(p)]
    refs = _references(ch)
    if not paths or not refs:
        return math.inf
    g = _raster(_normalise(_densify(paths)))
    return min(_chamfer(g, r) for r in refs)
