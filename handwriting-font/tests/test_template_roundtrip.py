"""Round trip: fill a template raster with an existing font, fake a skewed phone photo, ingest it.

    python -m tests.test_template_roundtrip path/to/any_font.otf  [out_dir]

Passes when every box is recovered with the right character, which exercises marker detection,
perspective correction, guide-line rejection and box cutting.
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwfont import template as T


def main(font_path: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, int(0.55 * T.DPI))
    rng = np.random.default_rng(0)
    photos = []
    for page in range(T.N_PAGES):
        img = Image.fromarray(cv2.cvtColor(T.render_page_png(page), cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for k in range(T.PER_PAGE):
            i = page * T.PER_PAGE + k
            if i >= len(T.CHARSET):
                break
            x, y = T.box_origin(i)
            d.text(((x + 0.18) * T.DPI, (y + T.BASE_Y) * T.DPI), T.CHARSET[i], font=font, fill=(25, 30, 70), anchor="ls")
        page_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = page_img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (src * 0.55 + np.float32([250, 180]) + rng.uniform(-120, 120, (4, 2))).astype(np.float32)
        photo = cv2.warpPerspective(page_img, cv2.getPerspectiveTransform(src, dst),
                                    (int(w * 0.55) + 500, int(h * 0.55) + 400), borderValue=(40, 35, 30))
        grad = np.linspace(0.75, 1.0, photo.shape[1])[None, :, None]  # uneven lighting
        photo = np.clip(cv2.GaussianBlur(photo, (0, 0), 1.0) * grad + rng.normal(0, 4, photo.shape), 0, 255)
        p = outdir / f"fake_photo_p{page + 1}.jpg"
        cv2.imwrite(str(p), photo.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        photos.append(p)
    T.ingest(outdir / "glyphs_template.npy", photos)
    got = Counter(r["char"] for r in np.load(outdir / "glyphs_template.npy", allow_pickle=True))
    want = Counter(T.CHARSET)
    missing = {c: want[c] - got.get(c, 0) for c in want if got.get(c, 0) < want[c]}
    assert not missing, f"boxes not recovered: {missing}"
    print(f"PASS: {sum(got.values())}/{len(T.CHARSET)} boxes recovered")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "build/test_template"))
