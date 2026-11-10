"""Glyph extraction from labelled journal strips.

Label file format (one strip per line, items separated by '|'):
    <image_id> R<rid>: <char> <x0>-<x1> [y<y0>-<y1>] | <char> <x0>-<x1> | ...
x/y are *display* pixels from the labelling sheets (DISPLAY_SCALE px per source px; y measured
from the top of the strip image, below the ruler). <char> is the literal character, or a name
from NAMED (for characters that clash with the syntax).

Commands (run as `python -m hwfont.extract ...`, with HWFONT_PHOTOS pointing at the photos):
    index   <image_id>                                       -> index_<image_id>.png (every line + R-id)
    zoom    <image_id> <rid> <x0> <x1> [<rid> <x0> <x1> ...]  -> zoom_<image_id>.png (component x-ranges)
    extract <labels.txt> [<labels.txt> ...] <out_dir>         -> glyphs.npy + contact.png for review
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .journal import (UPSAMPLE, components, darkness_map, detect_rules, detect_text_lines, ink_mask, local_spacing,
                     make_strip, rule_mask_source, segment)

# Journal photos live outside the repo (they are private). Point HWFONT_PHOTOS at the folder;
# an image_id in the label file is the start of a photo's filename.
PHOTOS = Path(os.environ.get("HWFONT_PHOTOS", "photos"))


def photo_path(image_id: str) -> Path:
    hits = sorted(PHOTOS.glob(f"{image_id}*"))
    if not hits:
        raise FileNotFoundError(f"no photo starting with {image_id!r} in {PHOTOS} (set HWFONT_PHOTOS)")
    return hits[0]
DISPLAY_SCALE = 2.0
D2S = UPSAMPLE / DISPLAY_SCALE  # display px -> strip px

NAMED = {"comma": ",", "colon": ":", "pipe": "|", "hyphen": "-", "period": ".", "semicolon": ";",
         "dollar": "$", "percent": "%", "amp": "&", "apos": "'", "quote": '"', "excl": "!",
         "question": "?", "lparen": "(", "rparen": ")", "slash": "/", "plus": "+", "hash": "#",
         "at": "@", "equal": "=", "star": "*"}


# Per-photo options (rotation for photos taken sideways), keyed by image id prefix.
PHOTO_OPTS_PATH = Path(os.environ.get("HWFONT_PHOTO_OPTS", Path(__file__).parent.parent / "labels" / "photos.json"))
PHOTO_OPTS = json.loads(PHOTO_OPTS_PATH.read_text()) if PHOTO_OPTS_PATH.exists() else {}
_ROT = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def page(image_id: str):
    """-> (dark, rules, rule_mask, unruled). Ruled notebook pages use the printed rules; pages with
    fewer than 8 detected rules (cards, plain paper) use fitted text baselines and no rule removal."""
    img = cv2.imread(str(photo_path(image_id)))
    rot = PHOTO_OPTS.get(image_id, {}).get("rotate")
    if rot:
        img = cv2.rotate(img, _ROT[rot])
    dark, paper = darkness_map(img)
    rules = detect_rules(dark, paper)
    unruled = len(rules) < 8
    line_map = None
    if unruled:
        rules, line_map = detect_text_lines(dark, paper)
        rmask = np.zeros(dark.shape, np.uint8)
    else:
        rmask = rule_mask_source(dark.shape, rules)
    return dark, rules, rmask, line_map


def strip_for(dark, rules, r, image_id, line_map):
    st = make_strip(dark, r, local_spacing(rules, r), image_id)
    st.xh = getattr(r, "xh", None)  # unruled pages: x-height (source px) measured during line detection
    st.unruled = line_map is not None
    if st.unruled:
        st.line_map = cv2.remap(line_map.astype(np.float32), st.map_x, st.map_y, cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int64)
    return st


def load_strips(image_id: str, rids: set[int]):
    """Rebuild only the requested strips (deterministic: same code -> same rids)."""
    dark, rules, rmask, line_map = page(image_id)
    out = {}
    for r in rules:
        if r.rid not in rids:
            continue
        st = strip_for(dark, rules, r, image_id, line_map)
        ink = ink_mask(st, rmask)
        lab, comps = components(st, ink)
        assigned = np.isin(lab, [c.cid for c in comps])
        del st.map_x, st.map_y
        st.line_map = None
        st.cuts = sorted({sg.x0 for sg in segment(st, lab, comps)[1]} - {c.x for c in comps})
        out[r.rid] = (st, ink, assigned, lab, comps)
    return out


def parse_labels(path: Path):
    """-> {image_id: {rid: [(char, x0, x1, y0|None, y1|None), ...]}}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    for ln, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        m = re.match(r"(\w+)\s+R(\d+)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"line {ln}: cannot parse {raw!r}")
        image_id, rid, rest = m.group(1), int(m.group(2)), m.group(3)
        for item in rest.split("|"):
            item = item.strip()
            if not item:
                continue
            cm = re.match(r"(\S+)\s+(c\d+(?:\+c\d+)*)$", item)
            if cm:  # whole components, referenced by their displayed left edge
                ch = NAMED.get(cm.group(1), cm.group(1))
                refs = [int(t[1:]) for t in cm.group(2).split("+")]
                out[image_id][rid].append((ch, refs, None, None, None))
                continue
            im = re.match(r"(\S+)\s+(\d+)-(\d+)(?:\s+y(\d+)-(\d+))?$", item)
            if not im:
                raise ValueError(f"line {ln}: bad item {item!r}")
            ch = NAMED.get(im.group(1), im.group(1))
            y0 = int(im.group(4)) if im.group(4) else None
            y1 = int(im.group(5)) if im.group(5) else None
            out[image_id][rid].append((ch, int(im.group(2)), int(im.group(3)), y0, y1))
    return out


# ---------------------------------------------------------------- zoom view

def zoom(image_id: str, specs: list[tuple[int, int, int]], out: Path, mag: float = 2.0):
    strips = load_strips(image_id, {r for r, _, _ in specs})
    panels = []
    for rid, x0, x1 in specs:
        st, ink, assigned, lab, comps = strips[rid]
        a, b = int(x0 * D2S), int(x1 * D2S)
        vis = np.full(ink.shape + (3,), 255, np.uint8)
        vis[ink > 0] = (205, 205, 205)
        vis[assigned] = (30, 30, 30)
        vis = vis[:, a:b]
        rows = np.where(assigned[:, a:b].any(1))[0]
        ry0 = max(0, rows.min() - 12) if len(rows) else 0
        ry1 = min(vis.shape[0], rows.max() + 12) if len(rows) else vis.shape[0]
        vis = vis[ry0:ry1]
        f = DISPLAY_SCALE * mag / UPSAMPLE
        vis = cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        top = 24
        canvas = np.full((vis.shape[0] + top, vis.shape[1], 3), 255, np.uint8)
        canvas[top:] = vis
        for X in range((x0 // 5) * 5, x1 + 1, 5):
            x = int((X - x0) * mag)
            if x < 0 or x >= canvas.shape[1]:
                continue
            if X % 25 == 0:
                cv2.line(canvas, (x, top), (x, canvas.shape[0]), (255, 190, 190), 1)
                cv2.line(canvas, (x, 12), (x, top), (0, 0, 200), 1)
                cv2.putText(canvas, str(X), (x + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1, cv2.LINE_AA)
            else:
                cv2.line(canvas, (x, 18), (x, top), (0, 0, 200), 1)
        m = (vis < 128).all(2)
        canvas[top:][m] = vis[m]
        # component boxes with their x-range in display units, so labels can anchor on exact numbers
        cols = [(0, 0, 220), (0, 150, 0), (200, 0, 160), (200, 110, 0)]
        k = 0
        for c in comps:
            if c.x + c.w < a or c.x > b or c.area < 40 * UPSAMPLE:
                continue
            cx0, cx1 = c.x / D2S, (c.x + c.w) / D2S
            px0, px1 = int((cx0 - x0) * mag), int((cx1 - x0) * mag)
            py0, py1 = top + int((c.y - ry0) / D2S * mag), top + int((c.y + c.h - ry0) / D2S * mag)
            col = cols[k % 4]
            cv2.rectangle(canvas, (px0, py0), (px1, py1), col, 1)
            ty = py0 - 3 if k % 2 == 0 else py1 + 12
            cv2.putText(canvas, f"{int(cx0)}-{int(cx1)}", (max(0, px0), max(34, ty)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)
            k += 1
        # candidate cut points inside joined words (thin connector columns)
        for cx in st.cuts:
            X = cx / D2S
            if x0 <= X <= x1:
                px = int((X - x0) * mag)
                for yy in range(top, canvas.shape[0], 8):
                    cv2.line(canvas, (px, yy), (px, min(yy + 4, canvas.shape[0] - 1)), (0, 140, 255), 1)
                cv2.putText(canvas, str(int(X)), (px + 1, canvas.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (0, 110, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"R{rid}", (2, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    W = max(p.shape[1] for p in panels)
    sheet = np.vstack([np.pad(p, ((0, 6), (0, W - p.shape[1]), (0, 0)), constant_values=90) for p in panels])
    cv2.imwrite(str(out), sheet)


# ---------------------------------------------------------------- extraction

MULTIPART = set("ij!?;:\"%=\u00f7")


def drop_slivers(m: np.ndarray, ch: str) -> np.ndarray:
    """Remove fragments of neighbouring letters that a vertical cut drags in.

    Pieces under 12% of the largest piece are dropped, except for characters that are legitimately
    made of a big part plus a small one (dots, punctuation marks).
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    ib = int(np.argmax(areas))
    big = areas[ib]
    by0 = st[ib + 1, cv2.CC_STAT_TOP]
    by1 = by0 + st[ib + 1, cv2.CC_STAT_HEIGHT]
    thresh = 0.04 if ch in MULTIPART else 0.12
    # pieces that float well above/below the main body belong to the neighbouring text line
    max_gap = (0.45 if ch in MULTIPART else 0.15) * (by1 - by0)
    keep = []
    for i, a in enumerate(areas):
        t, h = st[i + 1, cv2.CC_STAT_TOP], st[i + 1, cv2.CC_STAT_HEIGHT]
        gap = max(by0 - (t + h), t - by1, 0)
        if a >= thresh * big and gap <= max_gap:
            keep.append(i + 1)
    return np.isin(lab, keep)


def extract(labels_paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: dict = defaultdict(lambda: defaultdict(list))
    for lp in ([labels_paths] if isinstance(labels_paths, (str, Path)) else labels_paths):
        for image_id, by_rid in parse_labels(lp).items():
            for rid, items in by_rid.items():
                labels[image_id][rid].extend(items)
    records = []
    for image_id, by_rid in labels.items():
        strips = load_strips(image_id, set(by_rid))
        for rid, items in by_rid.items():
            if rid not in strips:
                print(f"WARN {image_id} R{rid}: strip not found", file=sys.stderr)
                continue
            st, ink, assigned, lab, comps = strips[rid]
            for ch, x0, x1, y0, y1 in items:
                if isinstance(x0, list):
                    ids = []
                    for ref in x0:
                        hit = [c for c in comps if abs(c.x / D2S - ref) <= 3]
                        if not hit:
                            print(f"WARN {image_id} R{rid} {ch!r}: no component at {ref}", file=sys.stderr)
                        ids += [max(hit, key=lambda c: c.area).cid] if hit else []
                    m = np.isin(lab, ids) if ids else np.zeros_like(ink, bool)
                    x0, x1 = "c" + "+".join(map(str, x0)), ""
                else:
                    a, b = int(x0 * D2S), int(x1 * D2S)
                    m = np.zeros_like(ink, bool)
                    m[:, a:b] = (ink[:, a:b] > 0) & assigned[:, a:b]
                    if y0 is not None:
                        keep = np.zeros_like(m)
                        keep[int(y0 * D2S):int(y1 * D2S)] = True
                        m &= keep
                    m = drop_slivers(m, ch)
                if m.sum() < 10:
                    print(f"WARN {image_id} R{rid} {ch!r} {x0}-{x1}: empty", file=sys.stderr)
                    continue
                ys, xs = np.where(m)
                gy0, gy1, gx0, gx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                pad = 6
                Y0, Y1 = max(0, gy0 - pad), min(m.shape[0], gy1 + pad)
                X0, X1 = max(0, gx0 - pad), min(m.shape[1], gx1 + pad)
                # soft glyph: darkness restricted to a slightly dilated mask -> anti-aliased edges for tracing
                soft_mask = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                soft = np.where(soft_mask, st.dark, 0)[Y0:Y1, X0:X1].astype(np.float32)
                records.append(dict(
                    char=ch, image=image_id, rid=rid, x0=x0, x1=x1,
                    source=PHOTO_OPTS.get(image_id, {}).get("source", "journal"),
                    spacing_px=st.spacing * UPSAMPLE,           # rule spacing, strip px
                    top=int(gy0 - Y0), bottom=int(gy1 - Y0),     # glyph ink extent inside the crop
                    crop_y0=int(Y0), crop_x0=int(X0), v0=int(st.v0),
                    mask=m[Y0:Y1, X0:X1].copy(), soft=soft,
                ))
    np.save(out_dir / "glyphs.npy", np.array(records, dtype=object), allow_pickle=True)
    contact_sheet(records, out_dir / "contact.png")
    print(f"{len(records)} glyphs -> {out_dir}")
    counts = defaultdict(int)
    for r in records:
        counts[r["char"]] += 1
    print("coverage:", " ".join(f"{k}{v}" for k, v in sorted(counts.items())))
    return records


def contact_sheet(records, out: Path, cell=(96, 110), cols=16):
    cw, chh = cell
    rows = (len(records) + cols - 1) // cols
    sheet = np.full((rows * chh, cols * cw, 3), 255, np.uint8)
    for i, r in enumerate(records):
        g = r["mask"].astype(np.uint8)
        h, w = g.shape
        f = min((cw - 8) / w, (chh - 28) / h, 0.5)
        g = cv2.resize(g * 255, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * chh, (i % cols) * cw
        sub = sheet[y + 4:y + 4 + g.shape[0], x + 4:x + 4 + g.shape[1]]
        sub[g > 100] = (0, 0, 0)
        cv2.rectangle(sheet, (x, y), (x + cw - 1, y + chh - 1), (220, 220, 220), 1)
        cv2.putText(sheet, f"{i}:{r['char']}", (x + 3, y + chh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{r['image'][:4]}R{r['rid']}", (x + 3, y + chh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet)


# ---------------------------------------------------------------- whole-photo index

def index(image_id: str, out: Path, f: float = 0.3):
    """Every strip of a photo, assigned ink only, core band only, with R-id and a 100-unit scale."""
    dark, rules, rmask, line_map = page(image_id)
    rows = []
    for r in rules:
        s = local_spacing(rules, r)
        if not np.isfinite(s) or s < 25 or s > 120:
            continue
        st = strip_for(dark, rules, r, image_id, line_map)
        ink = ink_mask(st, rmask)
        lab, comps = components(st, ink)
        del st.map_x, st.map_y
        if len(comps) < 2:
            continue
        S = st.spacing * UPSAMPLE
        y0, y1 = int(max(0, st.v0 - 1.35 * S)), int(min(ink.shape[0], st.v0 + 0.45 * S))
        vis = np.where(np.isin(lab, [c.cid for c in comps]), 0, 255).astype(np.uint8)[y0:y1]
        k = f * DISPLAY_SCALE / UPSAMPLE
        vis = cv2.resize(vis, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        lab_w = 42
        row = np.full((vis.shape[0] + 10, vis.shape[1] + lab_w, 3), 255, np.uint8)
        row[10:, lab_w:] = vis
        for X in range(0, int(vis.shape[1] / f) + 1, 100):
            px = lab_w + int(X * f)
            cv2.line(row, (px, 0), (px, 6 if X % 500 else 10), (0, 0, 220), 1)
            if X % 500 == 0 and X:
                cv2.putText(row, str(X), (px + 2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 220), 1)
        cv2.putText(row, f"R{r.rid}", (2, row.shape[0] // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 130, 0), 1)
        rows.append(row)
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 2), (0, W - r.shape[1]), (0, 0)), constant_values=170) for r in rows])
    cv2.imwrite(str(out), sheet)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "index":
        index(sys.argv[2], Path(f"index_{sys.argv[2]}.png"))
    elif cmd == "zoom":
        nums = list(map(int, sys.argv[3:]))
        zoom(sys.argv[2], [tuple(nums[i:i + 3]) for i in range(0, len(nums), 3)], Path(f"zoom_{sys.argv[2]}.png"))
    elif cmd == "extract":  # extract <labels.txt> [<labels.txt> ...] <out_dir>
        extract([Path(p) for p in sys.argv[2:-1]], Path(sys.argv[-1]))
    else:
        sys.exit(__doc__)
