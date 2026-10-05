"""showpiece · 레나 델타 ALT 의 3D 투시도(DEM 위에 드레이프). pyvista 오프스크린으로 그리고 matplotlib 로 색 막대·설명 줄을 붙인다.

입력  data/processed/map_lena/hillshade_v1/{dem_250m.tif, drape_rgb.npz}(hillshade_maps_v1 산출: 250 m DEM, 음영 곱 혼합된 ALT 색)
표면  z = DEM(결측·바다 0 m) × 수직 과장(VE, 기본 20). 색 = 드레이프 RGB 그대로(ALT 35–50 cm, oslo_r), 바다 흰색.
산출  deck/assets/paper_report/maps/Lena_ALT_3d_slide.png(12.0 × 5.2 in, 300 dpi), _source_values.json, _legend.md
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/lena_3d_v1.py [--ve 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import Normalize                                                                 # noqa: E402
import rasterio                                                                                         # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

HS = ROOT / "data" / "processed" / "map_lena" / "hillshade_v1"
OUT = MV.OUT_SLIDE / "Lena_ALT_3d_slide.png"
ALT_RANGE = (35.0, 50.0)                      # v3 레나 지도와 같은 범위(hillshade_v1_meta 의 drape.alt_range)


def render(ve: float, size=(3000, 1700)) -> tuple[np.ndarray, dict]:
    import pyvista as pv
    pv.OFF_SCREEN = True
    with rasterio.open(HS / "dem_250m.tif") as src:
        dem = src.read(1).astype(np.float64)
        tr = src.transform
    z = np.load(HS / "drape_rgb.npz")
    rgb = z["rgb"]
    nrow, ncol = dem.shape
    zz = np.where(np.isfinite(dem), dem, 0.0)
    zz = np.maximum(zz, 0.0) * ve
    xs = tr.c + (np.arange(ncol) + 0.5) * tr.a
    ys = tr.f + (np.arange(nrow) + 0.5) * tr.e
    X, Y = np.meshgrid(xs, ys)
    grid = pv.StructuredGrid(X, Y, zz)
    grid.point_data["rgb"] = rgb.reshape(-1, 3, order="F") if False else rgb.transpose(1, 0, 2).reshape(-1, 3)   # pyvista 는 F 순서(열 우선)
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("white")
    p.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, show_scalar_bar=False, ambient=0.35, diffuse=0.75, specular=0.05)
    cx, cy = float(xs.mean()), float(ys.mean())
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    p.camera_position = [(cx + 0.15 * span, cy - 1.35 * span, 0.95 * span), (cx, cy, 0.0), (0.0, 0.0, 1.0)]
    p.camera.view_angle = 30.0
    p.enable_anti_aliasing("ssaa")
    p.remove_all_lights()
    p.add_light(pv.Light(position=(cx - 1.5 * span, cy - 1.0 * span, 1.6 * span), focal_point=(cx, cy, 0.0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(cx + 1.0 * span, cy + 1.5 * span, 1.0 * span), focal_point=(cx, cy, 0.0), intensity=0.35, light_type="scene light"))
    img = p.screenshot(None, return_img=True)
    p.close()
    info = dict(dem_shape=[nrow, ncol], res_m=float(tr.a), ve=ve, camera=[list(map(float, c)) for c in p.camera_position], window=list(size),
                z_max_m=float(np.nanmax(dem)), extent_km=[round((xs.max() - xs.min()) / 1000, 1), round((ys.max() - ys.min()) / 1000, 1)])
    return img, info


def autocrop(img, pad=10):
    m = (img < 250).any(axis=2)
    r = np.where(m.any(axis=1))[0]
    c = np.where(m.any(axis=0))[0]
    return img[max(r[0] - pad, 0):r[-1] + pad, max(c[0] - pad, 0):c[-1] + pad]


def compose(img, ve, info):
    M = MV.Med("slide")
    MV.use_medium("slide")
    W, H = MV.SLIDE_W_MM, MV.SLIDE_H_MAX_MM
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    img = autocrop(img)
    ih, iw = img.shape[:2]
    TOP, BOT, LM, KEYR = 2.0, 10.0, 4.0, 26.0
    ph = H - TOP - BOT
    pw = ph * iw / ih
    if pw > W - LM - KEYR:
        pw = W - LM - KEYR
        ph = pw * ih / iw
    x0 = (W - KEYR - pw) / 2
    ax = S.axes_mm(fig, x0, TOP + (H - TOP - BOT - ph) / 2, pw, ph)
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    lo, hi = ALT_RANGE
    cax = S.axes_mm(fig, W - KEYR + 2.0, TOP + 0.28 * (H - TOP - BOT), M.cbar, 0.44 * (H - TOP - BOT))
    MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M, "max", orientation="vertical")
    note = f"Lena Delta · 1 km ALT on the 250 m DEM · vertical exaggeration ×{ve:g} · view from the south"
    t = fig.text(x0 / W, 1 - (H - 2.5) / H, note, fontsize=M.fs_small, color=MV.GRAY_TXT, ha="left", va="bottom")
    t.set_gid("note")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ve", type=float, default=20.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    img, info = render(a.ve)
    fig = compose(img, a.ve, info)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    info.update(alt_range=list(ALT_RANGE), out=str(OUT.relative_to(ROOT)), bytes=int(OUT.stat().st_size), elapsed_s=round(time.time() - t0, 1),
                script="scripts/4_visualization/paper_v3/lena_3d_v1.py", source=["data/processed/map_lena/hillshade_v1/dem_250m.tif",
                                                                                  "data/processed/map_lena/hillshade_v1/drape_rgb.npz"])
    OUT.with_name("Lena_ALT_3d_source_values.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    OUT.with_name("Lena_ALT_3d_legend.md").write_text(
        "# Lena Delta ALT, 3D perspective\n\nPerspective view of the Lena Delta: the 1 km residual-ML ALT map (same colours as the v3 map, "
        f"{ALT_RANGE[0]:.0f}–{ALT_RANGE[1]:.0f} cm, hillshade-blended) draped on the Copernicus DEM GLO-30 averaged to 250 m, elevation exaggerated "
        f"{a.ve:g} times (maximum elevation {info['z_max_m']:.0f} m in the south-west hills; the delta plain lies below about 30 m). Sea and water white. "
        f"View from the south-east, pyvista off-screen rendering with two scene lights. Extent about {info['extent_km'][0]:.0f} × {info['extent_km'][1]:.0f} km; "
        "no axes are drawn, so the figure gives shape and relative relief only.\n")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
