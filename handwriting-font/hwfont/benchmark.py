"""Benchmark fonts against reference handwriting (the neat cards).

    python -m hwfont.benchmark build/cards/glyphs.npy old.otf new.otf -o build/benchmark

Everything is measured in x-heights from the baseline, so photos and fonts compare directly.

Style metrics (medians): stroke weight, slant, ascender / cap / t / digit / descender heights,
lowercase widths, and word spacing (from labels/card_spacing.json written by hwfont.align).

Per-letter likeness: centerlines of each font letter (form 0) and of every reference sample are
placed in the same x-height frame (left edge at 0, baseline at 0, so size and proportion count,
not just shape) and compared by symmetric chamfer distance; a letter's distance is the median
over its reference samples. Lower = closer to the reference hand. Writes benchmark.json and
benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from . import tidy
from .build_font import TWO_MARKS, n_pieces, page_rho, stroke_width_px, target_height

PX = 16  # frame pixels per x-height
FW, FH, BASE_ROW = 72, 72, 46  # frame size, baseline row (room for 2.8 xh above, 1.6 below)
ASC, DESC = set("bdhkl"), set("gjpqy")


def _frame(pts_xh: np.ndarray) -> np.ndarray:
    img = np.zeros((FH, FW), np.uint8)
    x = np.round(pts_xh[:, 0] * PX + 3).astype(int)
    y = np.round(BASE_ROW - pts_xh[:, 1] * PX).astype(int)
    ok = (x >= 0) & (x < FW) & (y >= 0) & (y < FH)
    img[y[ok], x[ok]] = 1
    return img


def _chamfer(a, b) -> float:
    if a.sum() == 0 or b.sum() == 0:
        return math.inf
    da = cv2.distanceTransform((1 - a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (float(db[a > 0].mean()) + float(da[b > 0].mean())) / PX  # in x-heights


def _densify(paths):
    pts = []
    for p in paths:
        if len(p) == 1:
            pts.append(p)
            continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.hypot(*(b - a))))
            pts.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
        pts.append(p[-1:])
    return np.vstack(pts)


# ---------------------------------------------------------------- reference (cards)

def measure_reference(glyphs_npy: Path):
    recs = list(np.load(glyphs_npy, allow_pickle=True))
    rho, g = page_rho(recs)
    per = defaultdict(list)  # char -> list of dicts
    for r in recs:
        xh = rho.get(r["image"], g) * r["spacing_px"]
        m = r["mask"]
        # same sanity rules as the font build: no slivers, no half-marks
        tgt = target_height(r["char"])
        if tgt and not (0.5 * tgt <= (r["bottom"] - r["top"]) / xh <= 1.9 * tgt):
            continue
        if r["char"] in TWO_MARKS and n_pieces(m) < 2:
            continue
        pen = max(2.0, stroke_width_px(m))
        paths = tidy.centerline(np.pad(m, 2), pen)
        if not paths:
            continue
        pts = _densify(paths) - 2
        base_row = r["v0"] - r["crop_y0"]  # baseline row inside the crop
        xs = np.where(m.any(0))[0]
        pts_xh = np.column_stack([(pts[:, 0] - xs.min()) / xh, (base_row - pts[:, 1]) / xh])
        ys = np.where(m.any(1))[0]
        sl, ev = tidy.slant(paths, step=max(6, int(0.2 * xh)))
        # stroke at half maximum of the photographed darkness (a binary mask's width depends on its
        # threshold; half-max does not)
        soft = r["soft"]
        peak = cv2.dilate(soft, np.ones((9, 9), np.uint8))
        hm = (soft > 0.5 * peak) & (peak > 60)
        sk_hm = skeletonize(hm)
        stroke = 2 * float(np.median(cv2.distanceTransform(hm.astype(np.uint8), cv2.DIST_L2, 5)[sk_hm])) / xh \
            if sk_hm.any() else pen / xh
        per[r["char"]].append(dict(
            frame=_frame(pts_xh), top=(base_row - ys.min()) / xh, bottom=(base_row - ys.max() - 1) / xh,
            width=(xs.max() - xs.min() + 1) / xh, stroke=stroke, slant=sl, evidence=ev / max(pen, 1)))
    return per


# ---------------------------------------------------------------- font

def measure_font(otf: Path):
    font = TTFont(str(otf))
    xh_units = font["OS/2"].sxHeight or 330
    upm = font["head"].unitsPerEm
    size = int(round(60 * upm / xh_units))  # 60 px x-height
    pil = ImageFont.truetype(str(otf), size)
    cmap = font.getBestCmap()
    per = {}
    for cp in cmap:
        ch = chr(cp)
        if not ch.isprintable() or ch.isspace() or cp >= 0xE000:
            continue
        img = Image.new("L", (size * 3, size * 3), 0)
        ImageDraw.Draw(img).text((size // 2, size * 2), ch, font=pil, fill=255, anchor="ls")
        ink = np.array(img) > 127
        if ink.sum() < 10:
            continue
        ys, xs = np.where(ink)
        sk = skeletonize(ink)
        sy, sx = np.where(sk)
        xh = 60.0
        pts_xh = np.column_stack([(sx - xs.min()) / xh, (size * 2 - sy) / xh])
        pen = stroke_width_px(ink)
        paths = tidy.centerline(ink, pen)
        sl, ev = tidy.slant(paths, step=12)
        per[ch] = dict(frame=_frame(pts_xh), top=(size * 2 - ys.min()) / xh, bottom=(size * 2 - ys.max() - 1) / xh,
                       width=(xs.max() - xs.min() + 1) / xh, stroke=pen / xh, slant=sl, evidence=ev / max(pen, 1))
    space = pil.getlength(" ") / 60.0
    return per, space


# ---------------------------------------------------------------- metrics

def style(per, multi: bool):
    """Median style metrics from per-char measurements (lists for reference, dicts for fonts)."""
    def vals(chars, key):
        out = []
        for c in chars:
            v = per.get(c)
            if not v:
                continue
            out += [d[key] for d in v] if multi else [v[key]]
        return out

    lower = "abcdefghijklmnopqrstuvwxyz"
    slants = [d["slant"] for c in lower + lower.upper() for d in (per.get(c, []) if multi else [per[c]] if c in per else [])
              if d["evidence"] >= 1]
    med = lambda v: round(float(np.median(v)), 3) if len(v) else None
    return {
        "stroke weight (x-heights)": med(vals(lower, "stroke")),
        "slant (deg, + leans right)": round(math.degrees(math.atan(np.median(slants))), 1) if slants else None,
        "ascender height b d h k l": med(vals(ASC, "top")),
        "t height": med(vals("t", "top")),
        "capital height": med(vals("ABCDEFGHIJKLMNOPRSTUVWXYZ", "top")),
        "digit height": med(vals("0123456789", "top")),
        "descender depth g j p q y": med(vals(DESC, "bottom")),
        "x-letter width a c e n o s u": med(vals("aceno su", "width")),
        "m / w width": med(vals("mw", "width")),
    }


def likeness(ref, font_per):
    out = {}
    for ch, samples in ref.items():
        if ch not in font_per or not ch.isalnum():
            continue
        d = [_chamfer(s["frame"], font_per[ch]["frame"]) for s in samples]
        d = [x for x in d if np.isfinite(x)]
        if d:
            out[ch] = round(float(np.median(d)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path, help="glyphs.npy of the reference handwriting (cards)")
    ap.add_argument("fonts", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--spacing", type=Path, default=Path("labels/card_spacing.json"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ref = measure_reference(a.reference)
    report = {"reference": style(ref, multi=True), "fonts": {}, "likeness": {}}
    if a.spacing.exists():
        report["reference"].update(json.loads(a.spacing.read_text()))
    for f in a.fonts:
        per, space = measure_font(f)
        st = style(per, multi=False)
        st["word space (x-heights)"] = round(space, 3)
        report["fonts"][f.name] = st
        report["likeness"][f.name] = likeness(ref, per)
    (a.out / "benchmark.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    names = [f.name for f in a.fonts]
    keys = list(report["reference"])
    for n in names:
        keys += [k for k in report["fonts"][n] if k not in keys]
    md = ["| metric | cards | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for k in keys:
        md.append(f"| {k} | {report['reference'].get(k, '')} | " +
                  " | ".join(str(report["fonts"][n].get(k, "")) for n in names) + " |")
    md.append("")
    common = sorted(set.intersection(*(set(report["likeness"][n]) for n in names)))
    md.append(f"Letter likeness to the cards (median centerline distance, x-heights, lower is closer), "
              f"{len(common)} letters present in every font and on the cards:")
    md.append("")
    md.append("| | " + " | ".join(names) + " |")
    md.append("|---|" + "---|" * len(names))
    md.append("| mean | " + " | ".join(f"{np.mean([report['likeness'][n][c] for c in common]):.3f}" for n in names) + " |")
    for c in common:
        md.append(f"| {c} | " + " | ".join(f"{report['likeness'][n][c]:.3f}" for n in names) + " |")
    (a.out / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:len(keys) + 2]))
    print(md[len(keys) + 3])
    print("mean likeness:", {n: round(float(np.mean([report['likeness'][n][c] for c in common])), 3) for n in names})


if __name__ == "__main__":
    main()
