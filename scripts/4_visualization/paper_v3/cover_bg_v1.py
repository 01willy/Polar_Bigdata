"""showpiece · 덱 표지 배경(글자 없음). 알래스카 음영기복 ALT 드레이프(hillshade_maps_v1 의 drape_rgb.npz, 250 m)를 13.333 × 7.5 in, 300 dpi
캔버스의 오른쪽 약 60 % 에 채우고, 왼쪽 40 % 는 수평 알파 경사로 흰색으로 잇는다(x ≤ 0.30 W 는 완전 흰색, 0.30–0.55 W smoothstep).
산출  deck/assets/paper_report/cover_bg_v1.png(+ _source_values.json, _legend.md)
실행  python3 scripts/4_visualization/paper_v3/cover_bg_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "2_evaluation"))
NPZ = ROOT / "data" / "processed" / "map_alaska" / "hillshade_v1" / "drape_rgb.npz"
OUT = ROOT / "deck" / "assets" / "paper_report" / "cover_bg_v1.png"
W_IN, H_IN, DPI = 13.333, 7.5, 300
X_WHITE, X_FULL = 0.30, 0.55                 # 알파 0 까지의 폭, 알파 1 이 되는 위치(캔버스 폭 비율)
MAP_H_FRAC = 1.12                            # 지도 높이 / 캔버스 높이(위·아래를 조금 잘라 오른쪽을 채운다)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    t0 = time.time()
    z = np.load(NPZ)
    rgb = z["rgb"]                                                     # (nrow, ncol, 3) uint8, 북쪽이 첫 행
    W, H = int(round(W_IN * DPI)), int(round(H_IN * DPI))
    mh = int(round(MAP_H_FRAC * H))
    mw = int(round(mh * rgb.shape[1] / rgb.shape[0]))
    im = Image.fromarray(rgb).resize((mw, mh), Image.LANCZOS)
    canvas = np.full((H, W, 3), 255, np.float32)
    x0 = W - mw                                                        # 오른쪽 끝에 맞춘다
    y0 = (H - mh) // 2
    arr = np.asarray(im, np.float32)
    src = arr[max(0, -y0):max(0, -y0) + H, max(0, -x0):max(0, -x0) + W]
    dst_y0, dst_x0 = max(0, y0), max(0, x0)
    sub = canvas[dst_y0:dst_y0 + src.shape[0], dst_x0:dst_x0 + src.shape[1]]
    xs = (np.arange(W) / W)
    alpha = smoothstep((xs - X_WHITE) / (X_FULL - X_WHITE))[dst_x0:dst_x0 + src.shape[1]]
    a = alpha[None, :, None]
    canvas[dst_y0:dst_y0 + src.shape[0], dst_x0:dst_x0 + src.shape[1]] = (1 - a) * sub + a * src
    out = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(OUT, dpi=(DPI, DPI), optimize=True)
    lum = out.astype(float).mean(axis=2) / 255.0
    band = lum[:, : int(0.30 * W)]
    band40 = lum[:, : int(0.40 * W)]
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/4_visualization/paper_v3/cover_bg_v1.py", source=str(NPZ.relative_to(ROOT)),
                size_px=[W, H], size_in=[W_IN, H_IN], dpi=DPI, map_px=[mw, mh], map_left_frac=round(x0 / W, 3), fade=dict(x_white=X_WHITE, x_full=X_FULL, kind="smoothstep"),
                check=dict(left30_mean_luminance=float(band.mean()), left30_min_luminance=float(band.min()), left40_mean_luminance=float(band40.mean()),
                           right60_mean_luminance=float(lum[:, int(0.40 * W):].mean())),
                bytes=int(OUT.stat().st_size), elapsed_s=round(time.time() - t0, 1))
    OUT.with_name("cover_bg_v1_source_values.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    OUT.with_name("cover_bg_v1_legend.md").write_text(
        "# Cover background v1\n\nBackground image for the deck title slide, 13.333 × 7.5 in at 300 dpi, no text. Right part: the Alaska 1 km "
        "residual-ML ALT map draped on a 250 m Copernicus DEM hillshade (same rendering as Alaska_ALT_hillshade_v1; ALT colour scale 35–75 cm, "
        "blue sequential; land outside the mapped domain grey with relief; sea white). The map is scaled to 112 % of the canvas height and anchored at the "
        f"right edge (left edge at {x0 / W:.2f} of the width). Left part: a horizontal smoothstep alpha ramp from fully white at 0.30 of the width to the full "
        f"map at 0.55, so the title zone stays white (mean luminance of the left 30 %: {band.mean():.3f}).\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
