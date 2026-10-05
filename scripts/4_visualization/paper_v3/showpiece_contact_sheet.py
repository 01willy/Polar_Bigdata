"""showpiece · 산출 PNG 의 점검용 접촉 인쇄(contact sheet). 3열, 축소판마다 파일 이름·크기 글.
산출  deck/assets/paper_report/maps/showpiece_contact_sheet.png
실행  python3 scripts/4_visualization/paper_v3/showpiece_contact_sheet.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
MAPS = ROOT / "deck" / "assets" / "paper_report" / "maps"
PAPER = ROOT / "outputs" / "figures" / "paper" / "v3_restructure" / "maps"
ITEMS = [
    PAPER / "Lena_label_sequence_v1.png", PAPER / "Alaska_label_sequence_v1.png",
    MAPS / "Lena_label_sequence_slide.png", MAPS / "Alaska_label_sequence_slide.png",
    MAPS / "Lena_label_sequence_frames" / "frame_1_n10.png", MAPS / "Alaska_label_sequence_frames" / "frame_4_nall.png",
    PAPER / "Lena_ALT_hillshade_v1.png", PAPER / "Alaska_ALT_hillshade_v1.png",
    MAPS / "Lena_ALT_hillshade_slide.png", MAPS / "Alaska_ALT_hillshade_slide.png",
    ROOT / "deck" / "assets" / "paper_report" / "cover_bg_v1.png", MAPS / "Lena_ALT_3d_slide.png",
    PAPER / "Lena_transfer_methods_v1.png", MAPS / "Lena_transfer_methods_v1_slide.png",
    PAPER / "Alaska_obs_vs_pred_v1.png", MAPS / "Alaska_obs_vs_pred_v1_slide.png",
]
TW, PAD, CAP, NCOL = 1100, 30, 70, 3
FONT = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"


def main():
    font = ImageFont.truetype(FONT, 26)
    thumbs = []
    for p in ITEMS:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = TW / w
        th = im.resize((TW, int(round(h * s))), Image.LANCZOS)
        thumbs.append((p, th, (w, h), p.stat().st_size))
    rows = [thumbs[i:i + NCOL] for i in range(0, len(thumbs), NCOL)]
    row_h = [max(t[1].size[1] for t in r) + CAP for r in rows]
    W = NCOL * TW + (NCOL + 1) * PAD
    H = sum(row_h) + (len(rows) + 1) * PAD
    sheet = Image.new("RGB", (W, H), "#ffffff")
    dr = ImageDraw.Draw(sheet)
    y = PAD
    for r, rh in zip(rows, row_h):
        x = PAD
        for p, th, (w, h), nb in r:
            sheet.paste(th, (x, y))
            dr.rectangle([x - 1, y - 1, x + TW, y + th.size[1]], outline="#bbbbbb")
            rel = str(p.relative_to(ROOT))
            dr.text((x, y + th.size[1] + 8), f"{rel}", fill="#000000", font=font)
            dr.text((x, y + th.size[1] + 38), f"{w} x {h} px · {nb / 1e6:.2f} MB", fill="#555555", font=font)
            x += TW + PAD
        y += rh + PAD
    out = MAPS / "showpiece_contact_sheet.png"
    sheet.save(out, optimize=True)
    print(json.dumps(dict(out=str(out.relative_to(ROOT)), size=sheet.size, n=len(thumbs), bytes=out.stat().st_size)))


if __name__ == "__main__":
    main()
