"""Bake a font into the standalone card tool: python -m hwfont.make_card_tool <font.otf> <out.html>"""
import base64
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("card_tool.html")


def make(font: Path, out: Path):
    pua = font.with_suffix(".pua.json")
    if not pua.exists():
        raise FileNotFoundError(f"{pua} missing; build the font with build_font first")
    html = TEMPLATE.read_text()
    html = html.replace("__FONT_B64__", base64.b64encode(font.read_bytes()).decode())
    html = html.replace("__PUA_JSON__", pua.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    make(Path(sys.argv[1]), Path(sys.argv[2]))
