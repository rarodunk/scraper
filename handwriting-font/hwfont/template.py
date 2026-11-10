"""Printable handwriting template and scan/photo ingestion.

    python -m hwfont.template make   out/template.pdf      # print this (2 US-letter pages)
    python -m hwfont.template ingest out/glyphs_template.npy photo_p1.jpg photo_p2.jpg

Each page has four ArUco corner markers; ingestion finds them, warps the photo back onto the
page grid, and cuts every box. Because the template prints the baseline and x-height in each box,
template glyphs carry their exact baseline and scale (no estimation), and they replace journal
samples for any character they cover.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- layout (inches, origin top-left)
PAGE_W, PAGE_H = 8.5, 11.0
MARKER = 0.55                       # marker side
MARKER_INSET = 0.35                 # marker distance from page edge
GRID_X0, GRID_Y0 = 0.5, 1.25
BOX_W, BOX_H = 0.75, 0.98
COLS, ROWS = 10, 9
# guide lines inside each box, measured from the box top
ASC_Y, XH_Y, BASE_Y, DESC_Y = 0.16, 0.40, 0.62, 0.88
DPI = 300

PUNCT = ".,!?'\"-:;()&$%/"
CHARSET = (
    [c for c in "abcdefghijklmnopqrstuvwxyz" for _ in range(3)]
    + [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for _ in range(2)]
    + [c for c in "0123456789" for _ in range(2)]
    + [c for c in PUNCT for _ in range(2)]
)
PER_PAGE = COLS * ROWS
N_PAGES = (len(CHARSET) + PER_PAGE - 1) // PER_PAGE
ARUCO = cv2.aruco.DICT_4X4_50


def marker_boxes(page: int):
    """(id, x, y) top-left of each corner marker, in inches. ids 4p..4p+3 encode the page."""
    a, b = MARKER_INSET, PAGE_W - MARKER_INSET - MARKER
    c, d = MARKER_INSET, PAGE_H - MARKER_INSET - MARKER
    return [(4 * page + 0, a, c), (4 * page + 1, b, c), (4 * page + 2, b, d), (4 * page + 3, a, d)]


def box_origin(i: int):
    r, c = divmod(i % PER_PAGE, COLS)
    return GRID_X0 + c * BOX_W, GRID_Y0 + r * BOX_H


# ---------------------------------------------------------------- make (PDF)

def make(out_pdf: Path):
    from reportlab.lib.colors import Color
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cv = canvas.Canvas(str(out_pdf), pagesize=(PAGE_W * 72, PAGE_H * 72))
    dic = cv2.aruco.getPredefinedDictionary(ARUCO)
    guide = Color(0.55, 0.78, 0.95)
    grey = Color(0.72, 0.72, 0.72)

    def Y(y_in):  # reportlab origin is bottom-left
        return (PAGE_H - y_in) * 72

    for page in range(N_PAGES):
        for mid, x, y in marker_boxes(page):
            img = cv2.aruco.generateImageMarker(dic, mid, 200, borderBits=1)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            from PIL import Image
            cv.drawImage(ImageReader(Image.fromarray(rgb)), x * 72, Y(y + MARKER), MARKER * 72, MARKER * 72)
        cv.setFillColor(Color(0.35, 0.35, 0.35))
        cv.setFont("Helvetica", 9)
        cv.drawString(1.1 * 72, Y(0.62), f"Handwriting template - page {page + 1} of {N_PAGES}")
        cv.setFont("Helvetica", 7.5)
        cv.drawString(1.1 * 72, Y(0.80), "Fine black or dark-blue pen. Write each character once per box, sitting on the solid line;")
        cv.drawString(1.1 * 72, Y(0.94), "lowercase body reaches the dashed line, capitals and tall letters the top line. Write naturally, not neatly.")
        for k in range(PER_PAGE):
            i = page * PER_PAGE + k
            if i >= len(CHARSET):
                break
            x, y = box_origin(i)
            cv.setStrokeColor(grey)
            cv.setLineWidth(0.6)
            cv.rect(x * 72, Y(y + BOX_H), BOX_W * 72, BOX_H * 72)
            cv.setStrokeColor(guide)
            cv.setLineWidth(0.5)
            for gy, dash in [(ASC_Y, (1, 2)), (XH_Y, (3, 2)), (BASE_Y, None), (DESC_Y, (1, 2))]:
                cv.setDash(*dash) if dash else cv.setDash()
                cv.line((x + 0.05) * 72, Y(y + gy), (x + BOX_W - 0.05) * 72, Y(y + gy))
            cv.setDash()
            cv.setFillColor(grey)
            cv.setFont("Helvetica", 7)
            cv.drawString((x + 0.04) * 72, Y(y + 0.11), CHARSET[i])
        cv.showPage()
    cv.save()
    print(f"wrote {out_pdf} ({N_PAGES} pages, {len(CHARSET)} boxes)")


def render_page_png(page: int, scale: int = DPI) -> np.ndarray:
    """Raster version of a template page (same layout), used by tests."""
    W, H = int(PAGE_W * scale), int(PAGE_H * scale)
    img = np.full((H, W, 3), 255, np.uint8)
    dic = cv2.aruco.getPredefinedDictionary(ARUCO)
    for mid, x, y in marker_boxes(page):
        m = cv2.aruco.generateImageMarker(dic, mid, int(MARKER * scale), borderBits=1)
        X, Y = int(x * scale), int(y * scale)
        img[Y:Y + m.shape[0], X:X + m.shape[1]] = m[..., None]
    for k in range(PER_PAGE):
        i = page * PER_PAGE + k
        if i >= len(CHARSET):
            break
        x, y = box_origin(i)
        p = lambda a, b: (int(a * scale), int(b * scale))
        cv2.rectangle(img, p(x, y), p(x + BOX_W, y + BOX_H), (184, 184, 184), 1)
        for gy in (ASC_Y, XH_Y, BASE_Y, DESC_Y):
            cv2.line(img, p(x + 0.05, y + gy), p(x + BOX_W - 0.05, y + gy), (242, 199, 140), 1)
    return img


# ---------------------------------------------------------------- ingest

def warp_to_page(photo: np.ndarray):
    """Detect corner markers and return (page_index, page image at DPI) or raise."""
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) < 3:
        raise ValueError("found fewer than 3 corner markers; retake the photo with all four corners visible")
    ids = ids.ravel()
    page = int(np.bincount(ids // 4).argmax())
    src, dst = [], []
    known = {mid: (x, y) for mid, x, y in marker_boxes(page)}
    for c, mid in zip(corners, ids):
        if mid not in known:
            continue
        x, y = known[mid]
        # detected corners are the outer corners of the black-framed square we printed (TL, TR, BR, BL)
        sq = [(x, y), (x + MARKER, y), (x + MARKER, y + MARKER), (x, y + MARKER)]
        src += list(c.reshape(4, 2))
        dst += [(u * DPI, v * DPI) for u, v in sq]
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 4.0)
    page_img = cv2.warpPerspective(photo, H, (int(PAGE_W * DPI), int(PAGE_H * DPI)), flags=cv2.INTER_CUBIC,
                                   borderValue=(255, 255, 255))
    return page, page_img


def ingest(out_npy: Path, photos: list[Path]):
    records = []
    for path in photos:
        photo = cv2.imread(str(path))
        if photo is None:
            raise FileNotFoundError(path)
        page, img = warp_to_page(photo)
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        bg = cv2.medianBlur(cv2.dilate(L.astype(np.uint8), np.ones((15, 15), np.uint8)), 61).astype(np.float32)
        dark = np.clip(bg - L, 0, 255)
        got = 0
        for k in range(PER_PAGE):
            i = page * PER_PAGE + k
            if i >= len(CHARSET):
                break
            x, y = box_origin(i)
            inset = 0.035
            X0, Y0 = int((x + inset) * DPI), int((y + 0.13) * DPI)  # skip the printed label row
            X1, Y1 = int((x + BOX_W - inset) * DPI), int((y + BOX_H - inset) * DPI)
            d = dark[Y0:Y1, X0:X1]
            # printed guides/border are light (darkness ~30-60); pen ink is far darker
            from skimage.filters import apply_hysteresis_threshold
            ink = apply_hysteresis_threshold(cv2.GaussianBlur(d, (0, 0), 0.8), 55, 95)
            n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            keep = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 25]
            if not keep:
                continue  # empty box: that sample was skipped
            m = np.isin(lab, keep)
            ys, xs = np.where(m)
            pad = 6
            cy0, cy1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
            cx0, cx1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)
            soft_mask = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            records.append(dict(
                char=CHARSET[i], image=f"template-p{page + 1}", rid=i, x0=0, x1=0,
                spacing_px=(BASE_Y - ASC_Y) * DPI,  # unused for template glyphs; kept for schema parity
                top=int(ys.min() - cy0), bottom=int(ys.max() + 1 - cy0),
                mask=m[cy0:cy1, cx0:cx1], soft=np.where(soft_mask, d, 0)[cy0:cy1, cx0:cx1].astype(np.float32),
                # exact geometry from the printed guides, relative to this crop
                baseline_px=float((y + BASE_Y) * DPI - Y0 - cy0),
                xh_px=float((BASE_Y - XH_Y) * DPI),
                source="template",
            ))
            got += 1
        print(f"{path}: page {page + 1}, {got} filled boxes")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, np.array(records, dtype=object), allow_pickle=True)
    print(f"{len(records)} template glyphs -> {out_npy}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]))
    elif sys.argv[1] == "ingest":
        ingest(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
