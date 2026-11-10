"""Auto-label a page from its transcription: match each line's ink components to the known text.

    python -m hwfont.align labels/card_text.txt labels/card_labels.txt

Transcription format, one detected line per row (R-ids from `python -m hwfont.extract index`):
    a426a31f R5: I pray that God would feel really near in this season! We'll be praying that

Per line, the components (sorted left to right) are aligned to the characters with dynamic
programming. A component can cover 1-6 characters of one word (joined letters), be skipped (dots,
stray marks, ink from another line) or a character can be skipped (ink that line detection lost).
Costs, in x-height units measured from the fitted baseline:
  height    the component's top/bottom vs where those characters sit (x-height, ascender, cap,
            descender, punctuation bands); 0.25 x-height of slack, squared beyond it
  width     log ratio of component width to the characters' expected widths, squared
  shape     single letters only: legibility.score against reference typefaces
  spacing   word boundaries in the text must fall on wide gaps and vice versa
Expected widths start from defaults and are re-estimated from the first pass (two passes).

Only single-character matches that are good on every term are written as labels; dots of i/j and
the second mark of : ; " ! ? are attached when a skipped small mark sits next to them. The output
is ordinary label syntax, reviewed through the usual extract contact sheet like hand labels.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import legibility, tidy
from .build_font import stroke_width_px
from .extract import D2S, NAMED, load_strips
from .journal import UPSAMPLE

INV_NAMED = {v: k for k, v in NAMED.items()}

# vertical band (top, bottom) of each character in x-heights above the baseline
def band(ch: str) -> tuple[float, float]:
    if ch in "acemnorsuvwxz":
        return 1.0, 0.0
    if ch in "bdhkl":
        return 1.9, 0.0
    if ch == "f":
        return 1.9, 0.0
    if ch == "t":
        return 1.5, 0.0
    if ch == "i":
        return 1.0, 0.0  # stem only; the dot is a separate mark
    if ch == "j":
        return 1.0, -0.7
    if ch in "gpqy":
        return 1.0, -0.7
    if ch.isupper():
        return (1.75, -0.4) if ch in "JQ" else (1.75, 0.0)
    if ch.isdigit():
        return 1.65, 0.0
    return {".": (0.3, 0.0), ",": (0.3, -0.45), ";": (0.3, -0.45), ":": (0.3, 0.0), "!": (1.75, 0.45),
            "?": (1.75, 0.45), "'": (1.9, 1.25), '"': (1.9, 1.25), "-": (0.7, 0.35), "(": (1.85, -0.3),
            ")": (1.85, -0.3), "&": (1.7, 0.0), "$": (1.9, -0.2), "%": (1.7, 0.0), "/": (1.8, -0.1)}.get(ch, (1.0, 0.0))


DEFAULT_WIDTH = {**{c: 0.75 for c in "abcdeghknopqsuvxyz"}, **{c: 0.3 for c in "ijl"}, "f": 0.55, "r": 0.55,
                 "t": 0.6, "m": 1.15, "w": 1.15, **{c: 1.0 for c in "ABCDEFGHKLNOPQRSTUVXYZ"},
                 "I": 0.45, "J": 0.7, "M": 1.3, "W": 1.4, **{c: 0.75 for c in "023456789"}, "1": 0.35,
                 ".": 0.2, ",": 0.2, ";": 0.2, ":": 0.2, "'": 0.2, '"': 0.25, "!": 0.2, "?": 0.6, "-": 0.5,
                 "(": 0.35, ")": 0.35, "&": 0.8}
MULTIMARK = set("ij!?;:\"")
SKIP_CHAR = 3.0
SKIP_CHAR_PUNCT = 0.8
MAX_K = 6


def parse_text(path: Path):
    out = defaultdict(dict)
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\w+)\s+R(\d+)\s*:\s?(.*)$", line)
        if not m:
            raise ValueError(f"cannot parse {raw!r}")
        out[m.group(1)][int(m.group(2))] = m.group(3)
    return out


class Line:
    """Components of one strip in x-height units relative to the fitted baseline."""

    def __init__(self, st, lab, comps):
        self.st = st
        self.xh = (st.xh or st.spacing * 0.42) * UPSAMPLE
        self.comps = sorted(comps, key=lambda c: c.x)
        self.top = [(st.v0 - c.y) / self.xh for c in self.comps]
        self.bot = [(st.v0 - (c.y + c.h)) / self.xh for c in self.comps]
        self.w = [c.w / self.xh for c in self.comps]
        self.small = [c.h < 0.6 * self.xh and c.w < 0.7 * self.xh for c in self.comps]
        right, last = [], None
        for c, sm in zip(self.comps, self.small):
            right.append(last)
            if not sm:
                last = c.x + c.w if last is None else max(last, c.x + c.w)
        self.prev_right = right
        self._lab = lab
        self._paths = {}
        self._score = {}

    def paths(self, i):
        if i not in self._paths:
            c = self.comps[i]
            m = self._lab[c.y:c.y + c.h, c.x:c.x + c.w] == c.cid
            pen = max(2.0, stroke_width_px(m))
            self._paths[i] = tidy.centerline(np.pad(m, 2), pen)
        return self._paths[i]

    def score(self, i, ch):
        if (i, ch) not in self._score:
            self._score[(i, ch)] = legibility.score(self.paths(i), ch)
        return self._score[(i, ch)]


BANDS: dict = {}  # learned per-character (top, bottom) overrides, filled after pass 1


def char_band(ch: str):
    return BANDS.get(ch) or band(ch)


def shape_cost(L: Line, i: int, s: str, widths: dict, detail: bool = False):
    tops, bots = zip(*(char_band(ch) for ch in s))
    et, eb = max(tops), min(bots)
    dt, db = abs(L.top[i] - et), abs(L.bot[i] - eb)
    cv = 1.5 * (max(0.0, dt - 0.25) ** 2 + max(0.0, db - 0.25) ** 2)
    ew = sum(widths.get(ch, 0.75) for ch in s)
    cw = 2.0 * math.log(max(L.w[i], 0.05) / ew) ** 2
    cr = 0.0
    if len(s) == 1 and s.isalnum():
        if cv + cw < 2.5:
            cr = 0.6 * (min(L.score(i, s), 8.0) - 2.5)
        else:
            cr = 3.0
    elif len(s) > 1:
        cr = 0.8 * (len(s) - 1)
    total = min(cv, 8) + min(cw, 8) + cr
    return (total, cv, cw, cr) if detail else total


def gap_cost(L: Line, i: int, boundary: bool) -> float:
    if L.prev_right[i] is None:
        return 0.0
    g = (L.comps[i].x - L.prev_right[i]) / L.xh
    return 3.0 * max(0.0, 0.35 - g) if boundary else 2.0 * max(0.0, g - 0.7)


def align_line(L: Line, text: str, widths: dict):
    """-> list of (comp index, chars) for assigned components, and set of skipped comp indices."""
    chars, wordend = [], []
    for w in text.split():
        for k, ch in enumerate(w):
            chars.append(ch)
            wordend.append(k == len(w) - 1)
    n, m = len(L.comps), len(chars)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    bp = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i, j]
            if cur >= INF:
                continue
            if i < n:  # skip component
                c = cur + (0.3 if L.small[i] else 4.0)
                if c < dp[i + 1, j]:
                    dp[i + 1, j] = c
                    bp[(i + 1, j)] = (i, j, None)
            if j < m:  # skip character
                c = cur + (SKIP_CHAR if chars[j].isalnum() else SKIP_CHAR_PUNCT)
                if c < dp[i, j + 1]:
                    dp[i, j + 1] = c
                    bp[(i, j + 1)] = (i, j, "skip")
            if i < n and not L.small[i] or (i < n and j < m and not chars[j].isalnum()):
                for k in range(1, MAX_K + 1):
                    if j + k > m:
                        break
                    if k > 1 and wordend[j + k - 2]:
                        break  # a component never spans a space
                    s = "".join(chars[j:j + k])
                    c = cur + shape_cost(L, i, s, widths) + gap_cost(L, i, j > 0 and wordend[j - 1])
                    if c < dp[i + 1, j + k]:
                        dp[i + 1, j + k] = c
                        bp[(i + 1, j + k)] = (i, j, s)
    # backtrack
    out, skipped = [], set()
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, s = bp[(i, j)]
        if s is None:
            skipped.add(pi)
        elif s != "skip":
            out.append((pi, s))
        i, j = pi, pj
    return out[::-1], skipped, float(dp[n, m])


def split_component(L: Line, i: int, s: str, widths: dict):
    """Cut a joined component into its characters: expected boundaries from character widths, each
    snapped to the thinnest ink column (a single connector stroke) within 0.3 x-height. Returns
    [(char, x0, x1)] in strip px, or None when a boundary has no clean connector to cut."""
    c = L.comps[i]
    m = L._lab[c.y:c.y + c.h, c.x:c.x + c.w] == c.cid
    cols = m.sum(0).astype(float)
    stroke = max(2.0, stroke_width_px(m))
    ew = np.array([widths.get(ch, 0.75) for ch in s])
    bounds = np.cumsum(ew)[:-1] / ew.sum() * c.w
    win = int(0.3 * L.xh)
    cuts = [0]
    for b in bounds:
        lo, hi = max(cuts[-1] + 3, int(b) - win), min(c.w - 3, int(b) + win)
        if hi <= lo:
            return None
        x = lo + int(np.argmin(cols[lo:hi]))
        if cols[x] > 1.8 * stroke:
            return None  # no thin connector here: the letters overlap, don't guess
        cuts.append(x)
    cuts.append(c.w)
    return [(ch, c.x + a, c.x + b) for ch, a, b in zip(s, cuts[:-1], cuts[1:])]


def load_rejects(path: Path) -> set:
    """Reviewed-and-rejected items, in label syntax: {(image_id, rid, 'char c123'), ...}."""
    out = set()
    if path.exists():
        for raw in path.read_text().splitlines():
            m = re.match(r"(\w+)\s+R(\d+)\s*:\s*(.*)$", raw.strip())
            if m and not raw.lstrip().startswith("#"):
                for item in m.group(3).split("|"):
                    out.add((m.group(1), int(m.group(2)), item.strip()))
    return out


def run(text_path: Path, out_path: Path):
    text = parse_text(text_path)
    rejects = load_rejects(out_path.with_name(out_path.stem.replace("_labels", "") + "_rejects.txt"))
    lines = {}
    for image_id, by_rid in text.items():
        strips = load_strips(image_id, set(by_rid))
        for rid, t in by_rid.items():
            if rid not in strips:
                print(f"WARN {image_id} R{rid}: no such line", file=sys.stderr)
                continue
            st, ink, assigned, lab, comps = strips[rid]
            lines[(image_id, rid)] = (Line(st, lab, comps), t)
    widths = dict(DEFAULT_WIDTH)
    BANDS.clear()
    for it in range(3):
        results = {key: align_line(L, t, widths) for key, (L, t) in lines.items()}
        if it < 2:  # re-estimate widths and vertical bands from confident single-letter matches
            seen_w, seen_b = defaultdict(list), defaultdict(list)
            for key, (pairs, _, _) in results.items():
                L = lines[key][0]
                for i, s in pairs:
                    if len(s) == 1:
                        tot, cv, cw, cr = shape_cost(L, i, s, widths, detail=True)
                        if s.isalnum() and L.score(i, s) < 3.5 and cw < 0.5:
                            seen_b[s].append((L.top[i], L.bot[i]))
                        if cv < 0.3 and (not s.isalnum() or cr < 0.6):
                            seen_w[s].append(L.w[i])
            for ch, ws in seen_w.items():
                if len(ws) >= 2:
                    widths[ch] = float(np.median(ws))
            # per class first (capitals, ascenders, descenders share a band), then per character
            classes = {"upper": [c for c in seen_b if c.isupper()], "asc": [c for c in seen_b if c in "bdhkl"],
                       "desc": [c for c in seen_b if c in "gjpqy"], "f": ["f"] if "f" in seen_b else []}
            for name, members in classes.items():
                vals = [v for c in members for v in seen_b[c]]
                if len(vals) >= 3:
                    t_, b_ = np.median(vals, axis=0)
                    group = {"upper": [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], "asc": list("bdhkl"),
                             "desc": list("gjpqy"), "f": ["f"]}[name]
                    for c in group:
                        BANDS[c] = (float(t_), float(b_))
            for ch, vals in seen_b.items():
                if len(vals) >= 3:
                    BANDS[ch] = tuple(float(v) for v in np.median(vals, axis=0))
    # spacing between consecutive matched components: across a space vs inside a word
    word_gaps, inner_gaps = [], []
    for (image_id, rid), (pairs, skipped, cost) in results.items():
        L = lines[(image_id, rid)][0]
        t = lines[(image_id, rid)][1]
        ends, pos = set(), 0
        for w in t.split():
            pos += len(w)
            ends.add(pos)
        consumed = 0
        prev = None
        for i, s_ in pairs:
            if prev is not None and not L.small[i] and not L.small[prev]:
                g = (L.comps[i].x - (L.comps[prev].x + L.comps[prev].w)) / L.xh
                (word_gaps if consumed in ends else inner_gaps).append(g)
            consumed += len(s_)
            prev = i
    spacing = {"word space (x-heights)": round(float(np.median(word_gaps)), 3) if word_gaps else None,
               "letter gap inside words (x-heights)": round(float(np.median(inner_gaps)), 3) if inner_gaps else None}
    out_path.with_name(out_path.stem.replace("_labels", "") + "_spacing.json").write_text(json.dumps(spacing, indent=1))
    rows, kept, total = [], 0, 0
    for (image_id, rid), (pairs, skipped, cost) in sorted(results.items()):
        L, t = lines[(image_id, rid)]
        items = []
        for i, s in pairs:
            total += 1
            if 2 <= len(s) <= 5 and s.isalpha():
                tot, cv, cw, cr = shape_cost(L, i, s, widths, detail=True)
                parts = split_component(L, i, s, widths) if (cv < 0.6 and cw < 0.5) else None
                for ch, a, b in parts or []:
                    item = f"{ch} {int(a / D2S)}-{int(b / D2S)}"
                    if (image_id, rid, item) not in rejects:
                        items.append(item)
                        kept += 1
                continue
            if len(s) != 1:
                continue
            tot, cv, cw, cr = shape_cost(L, i, s, widths, detail=True)
            if cv > 0.6 or cw > 0.8 or (s.isalnum() and L.score(i, s) > 4.5):
                continue
            c = L.comps[i]
            refs = [i]
            if s in MULTIMARK:  # attach a skipped neighbouring mark (i dot, colon dot, second quote tick)
                for j in skipped:
                    cj = L.comps[j]
                    if L.small[j] and cj.x < c.x + c.w + 0.5 * L.xh and cj.x + cj.w > c.x - 0.5 * L.xh \
                            and abs((cj.y + cj.h / 2) - (c.y + c.h / 2)) < 2.2 * L.xh:
                        refs.append(j)
            name = INV_NAMED.get(s, s)
            item = f"{name} " + "+".join(f"c{int(L.comps[r].x / D2S)}" for r in refs)
            if (image_id, rid, item) in rejects:
                continue
            items.append(item)
            kept += 1
        rows.append(f"{image_id} R{rid}: " + " | ".join(items) + f"   # align cost {cost:.1f}")
    out_path.write_text("# generated by hwfont.align from " + str(text_path) + "; review via contact sheet\n"
                        + "\n".join(rows) + "\n")
    print(f"{kept} single-character labels from {total} matched components -> {out_path}")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
