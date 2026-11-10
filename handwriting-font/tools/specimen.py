"""Specimen sheet: every form of every character in a built font, labelled with its source sample.

    python tools/specimen.py build/MyHand.otf build/specimen.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

font_path, out = Path(sys.argv[1]), Path(sys.argv[2])
pua = json.loads(font_path.with_suffix(".pua.json").read_text())
forms = json.loads(font_path.with_name("forms.json").read_text())
cov = json.loads(font_path.with_name("coverage.json").read_text())
size, cw, ch_h, cols = 64, 128, 150, 12
f = ImageFont.truetype(str(font_path), size)
small = ImageFont.load_default()
chars = list(pua)
rows = (len(chars) + cols - 1) // cols
img = Image.new("RGB", (cols * cw, rows * ch_h), "white")
d = ImageDraw.Draw(img)
for k, c in enumerate(chars):
    x, y = (k % cols) * cw, (k // cols) * ch_h
    d.rectangle([x, y, x + cw - 1, y + ch_h - 1], outline=(225, 225, 225))
    d.line([x, y + 95, x + cw, y + 95], fill=(235, 210, 210))  # baseline
    src = cov.get(c, {}).get("source", "journal")
    tag = "" if src in ("journal", "template") else "*"
    d.text((x + 3, y + 2), f"{c}{tag}", fill=(200, 0, 0), font=small)
    for i, cp in enumerate(pua[c][:3]):
        d.text((x + 6 + i * 40, y + 95), chr(cp), font=f, fill=(20, 30, 80), anchor="ls")
    for i, s in enumerate(forms.get(c, [])[:3]):
        d.text((x + 3, y + 104 + i * 14), f"{i}:{s}"[:22], fill=(110, 110, 110), font=small)
img.save(out)
print(f"wrote {out}")
