"""拼合 figures/paper/ 下全部图的审阅用联系表（AI 辅助实现，2026-10-24）。

只读 PNG 预览，不改动任何图；输出 figures/paper/CONTACT_SHEET.png。
用法：python scripts/make_contact_sheet.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures" / "paper"
OUT = SRC / "CONTACT_SHEET.png"

TITLES = {
    "fig01": "Fig. 1  Study design",
    "fig02": "Fig. 2  Data quality and audit",
    "fig03": "Fig. 3  Latent structure",
    "fig04": "Fig. 4  Q1 models and fusion",
    "fig05": "Fig. 5  Q1 per-target performance",
    "fig06": "Fig. 6  Q1 chemical importance",
    "fig07": "Fig. 7  Q2 compound network",
    "fig08": "Fig. 8  Q2 evidence and stability",
    "fig09": "Fig. 9  Q3 fidelity and consistency",
    "fig10": "Fig. 10  Q3 constraint effects",
    "fig11": "Fig. 11  Q3 transfer Pareto",
    "fig12": "Fig. 12  Variance decomposition",
    "fig13": "Fig. 13  Robustness",
}

COL_W = 1500
PAD = 28
BAR = 54
COLS = 3


def main() -> None:
    files = sorted(SRC.glob("fig*.png"))
    if not files:
        raise SystemExit("no figure PNGs found")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    cells = []
    for f in files:
        im = Image.open(f).convert("RGB")
        scale = COL_W / im.width
        cells.append((f, im.resize((COL_W, round(im.height * scale)), Image.LANCZOS)))

    rows = (len(cells) + COLS - 1) // COLS
    widths = [COL_W] * COLS
    heights = []
    for r in range(rows):
        row = cells[r * COLS:(r + 1) * COLS]
        heights.append(max(im.height + BAR for _, im in row))
    W = PAD + sum(widths) + PAD * (COLS - 1)
    H = PAD + sum(heights) + PAD * (rows - 1)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    y = PAD
    for r in range(rows):
        x = PAD
        for c in range(COLS):
            i = r * COLS + c
            if i >= len(cells):
                break
            f, im = cells[i]
            key = f.stem.split("_")[0]
            d.text((x, y + 6), TITLES.get(key, f.stem), fill="#111111", font=font)
            d.line([(x, y + BAR - 10), (x + COL_W, y + BAR - 10)],
                   fill="#DDDDDD", width=2)
            sheet.paste(im, (x, y + BAR))
            x += COL_W + PAD
        y += heights[r] + PAD
    sheet.save(OUT, optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {sheet.width}×{sheet.height}")


if __name__ == "__main__":
    main()
