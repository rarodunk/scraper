"""Real card line above the same text set in one or more fonts (rendered by Chromium, so calt applies).

    HWFONT_PHOTOS=... python tools/side_by_side.py <image_id> <rid> "<text>" out.png font1.otf [font2.otf ...]
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hwfont.extract import load_strips  # noqa: E402

image_id, rid, text, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
fonts = [Path(f) for f in sys.argv[5:]]
st, ink, assigned, lab, comps = load_strips(image_id, {rid})[rid]
S = st.spacing * 4
y0, y1 = int(max(0, st.v0 - 1.25 * S)), int(min(st.dark.shape[0], st.v0 + 0.55 * S))
xs = np.where(assigned.any(0))[0]
crop = st.dark[y0:y1, max(0, xs.min() - 20):xs.max() + 20]
real = Image.fromarray(np.clip(255 - crop * 1.6, 0, 255).astype(np.uint8)).convert("RGB")
real = real.resize((int(real.width * 0.5), int(real.height * 0.5)))
rows = [("your card", real)]
tool = Path(__file__).with_name("preview.js")
for f in fonts:
    png = out.with_name(out.stem + f"_{f.stem}.png")
    xh_px = (st.xh or st.spacing * 0.32) * 4 * 0.5  # x-height of the real line as shown above
    subprocess.run(["node", str(tool), str(f), str(png), text.replace("&", "&amp;")], check=True,
                   env={**os.environ, "FS": str(int(xh_px / 0.33)), "VW": str(real.width + 200), "NODE_PATH": subprocess.check_output(["npm", "root", "-g"], text=True).strip()})
    im = Image.open(png).convert("RGB")
    rows.append((f.stem, im.crop((0, 0, min(im.width, real.width + 60), im.height))))
W = max(r.width for _, r in rows) + 90
H = sum(r.height + 8 for _, r in rows)
sheet = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(sheet)
y = 0
for name, im in rows:
    d.text((4, y + im.height // 2), name, fill=(200, 0, 0))
    sheet.paste(im, (90, y))
    y += im.height + 8
    d.line([(0, y - 4), (W, y - 4)], fill=(220, 220, 220))
sheet.save(out)
print(f"wrote {out}")
