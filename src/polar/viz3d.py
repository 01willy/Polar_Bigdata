"""
세련된 3D 시각화 — baseline 체적을 직관적인 이미지로 렌더.

전략: PyVista로 3D 장면을 '텍스트/컬러바 없이' 고품질 렌더 → matplotlib(NanumGothic)으로
한글 제목·캡션·컬러바를 합성. (VTK는 한글을 못 그리므로 분리.)

hero_scene(): 지질 fence diagram 스타일
  - 지표 온도면(z=0)  + 수직 단면(fence) 2매 = 깊이별 온도 구조
  - borehole stick(온도 색) = 실측  + 0 °C 등온면(영구동토 경계)
  다크 그라디언트 배경 / 3점 조명 / SSAA / 고해상도.
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.colors import Normalize

from . import config as C
from . import geo

CLIM = (-10.0, 2.0)
CMAP = "RdBu_r"          # 따뜻=빨강, 차가움=파랑

for _fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _load():
    import pyvista as pv
    img = pv.read(C.PROCESSED / "volume_magt.vti")
    prof = pd.read_csv(C.PROFILES_CSV).dropna(subset=["magt", "lat", "lon", "depth"])
    summ = pd.read_csv(C.SUMMARY_CSV)
    return img, prof, summ


def _auto_ve(img):
    b = img.bounds
    horiz = max(b[1] - b[0], b[3] - b[2])
    return 0.32 * horiz / max(b[5] - b[4], 1.0)


def _scaled_image(img, ve):
    """z를 ve배 + 아래로(-) 향하게 한 표시용 ImageData."""
    import pyvista as pv
    sx, sy, sz = img.spacing
    di = pv.ImageData(dimensions=img.dimensions, spacing=(sx, sy, sz * ve),
                      origin=(img.origin[0], img.origin[1], 0.0))
    di.point_data["MAGT"] = img.point_data["MAGT"]
    di.points[:, 2] *= -1.0
    return di


def _borehole_tubes(prof, ve, radius=14000):
    import pyvista as pv
    x, y = geo.to_xy(prof["lon"].to_numpy(), prof["lat"].to_numpy())
    prof = prof.assign(x=x, y=y)
    tubes = []
    for _bid, g in prof.groupby("borehole_id"):
        g = g.sort_values("depth")
        if len(g) < 2:
            continue
        pts = np.column_stack([g["x"], g["y"], -g["depth"].to_numpy() * ve])
        line = pv.lines_from_points(pts)
        line["MAGT"] = np.clip(g["magt"].to_numpy(), *CLIM)
        tubes.append(line.tube(radius=radius))
    return tubes[0].merge(tubes[1:]) if len(tubes) > 1 else tubes[0]


def _render(prof, summ, di, ve, cam):
    """PyVista 3D 렌더 → numpy 이미지. 불투명 블록 단면(지질 block diagram 스타일)."""
    import pyvista as pv
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv.OFF_SCREEN = True
    b = di.bounds
    p = pv.Plotter(off_screen=True, window_size=(1920, 1280), lighting="three lights")
    p.set_background("#0a0e1a", top="#21405f")

    # 지표 온도면 (불투명 lid)
    top = di.slice(normal="z", origin=(0, 0, -0.01))
    if top.n_points:
        p.add_mesh(top, scalars="MAGT", cmap=CMAP, clim=CLIM, opacity=0.92,
                   show_scalar_bar=False, smooth_shading=True, ambient=0.3)
    # 수직 fence 단면 2매 (불투명) — 깊이별 온도 구조
    cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    for nrm, org in [("x", (cx, 0, 0)), ("y", (0, cy, 0))]:
        sl = di.slice(normal=nrm, origin=org)
        if sl.n_points:
            p.add_mesh(sl, scalars="MAGT", cmap=CMAP, clim=CLIM, opacity=1.0,
                       show_scalar_bar=False, smooth_shading=True, ambient=0.3)
    # borehole sticks (실측) — 굵게
    p.add_mesh(_borehole_tubes(prof, ve, radius=11000), scalars="MAGT", cmap=CMAP,
               clim=CLIM, smooth_shading=True, show_scalar_bar=False,
               specular=0.4, specular_power=15)
    # 지점 상단 흰 구
    xs, ys = geo.to_xy(summ["lon"].to_numpy(), summ["lat"].to_numpy())
    caps = pv.PolyData(np.column_stack([xs, ys, np.full_like(xs, 4000.0)]))
    p.add_mesh(caps.glyph(geom=pv.Sphere(radius=10000), scale=False, orient=False),
               color="white", smooth_shading=True)

    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    p.camera.azimuth = cam[0]
    p.camera.elevation = cam[1]
    p.camera.zoom(cam[2])
    arr = p.screenshot(return_img=True)
    p.close()
    return arr


def _compose(arr, title, subtitle, legend, ve, out):
    """3D 이미지에 한글 제목·부제·범례·세로 컬러바 합성 (겹침 없이)."""
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    fig.patch.set_facecolor("#0a0e1a")
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(arr); ax.axis("off")
    # 제목/부제 (상단)
    ax.text(0.022, 0.965, title, transform=ax.transAxes, fontsize=26, color="white",
            fontweight="bold", va="top", ha="left")
    ax.text(0.023, 0.908, subtitle, transform=ax.transAxes, fontsize=13.5,
            color="#9fc0e8", va="top", ha="left")
    # 범례 박스 (좌하단) — 각 요소가 무엇인지
    leg = "\n".join(legend)
    ax.text(0.022, 0.20, leg, transform=ax.transAxes, fontsize=12.5, color="#e8eef7",
            va="bottom", ha="left", linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.6", fc="#0d1726", ec="#3a557a", alpha=0.82))
    # 깊이 방향 화살표 (좌측, 범례 위)
    ax.annotate("", xy=(0.035, 0.52), xytext=(0.035, 0.78), transform=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#bcd", lw=2.2))
    ax.text(0.048, 0.77, "지표 0 m", transform=ax.transAxes, color="#bcd", fontsize=11,
            va="center", ha="left")
    ax.text(0.048, 0.53, f"깊이 ↓\n(수직과장 ×{ve:.0f})", transform=ax.transAxes,
            color="#bcd", fontsize=10.5, va="center", ha="left")
    # 세로 컬러바 (우측)
    cax = fig.add_axes([0.905, 0.40, 0.016, 0.34])
    sm = cm.ScalarMappable(norm=Normalize(*CLIM), cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical", extend="both")
    cb.set_label("평균 지중온도 MAGT (°C)", color="white", fontsize=12.5, labelpad=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("#88a")
    cax.text(0.5, 1.06, "따뜻함", transform=cax.transAxes, color="#e08", fontsize=10, ha="center")
    cax.text(0.5, -0.10, "차가움\n(영구동토)", transform=cax.transAxes, color="#39f",
             fontsize=10, ha="center", va="top")
    fig.savefig(out, dpi=150, facecolor="#0a0e1a")
    plt.close(fig)
    print(f"  saved {out}")


def hero_scene():
    img, prof, summ = _load()
    ve = _auto_ve(img)
    di = _scaled_image(img, ve)
    arr = _render(prof, summ, di, ve, cam=(42, 18, 1.0))
    legend = [
        "■ 윗면 = 지표 온도 지도 (위에서 본 Alaska)",
        "■ 수직 단면 2매 = 깊이별 온도 구조(크리깅 보간)",
        "▮ 막대 = 실측 borehole (색=깊이별 온도)",
        "● 흰 점 = borehole 지표 위치",
    ]
    _compose(arr,
             "Alaska 영구동토 3D 온도장 (크리깅 보간 결과)",
             f"목적: 흩어진 borehole {summ.shape[0]}개 → 공간 연속 3D 지중온도 예측. "
             "파란 영역(<0°C)이 영구동토.",
             legend, ve,
             C.FIGURES / "hero_3d_permafrost.png")


def cross_section():
    """위도-깊이 단면도 — 영구동토 열구조를 한 장으로(가장 직관적)."""
    from scipy.interpolate import griddata
    prof = pd.read_csv(C.PROFILES_CSV).dropna(subset=["magt", "lat", "depth"])
    lat, dep, magt = prof["lat"].to_numpy(), prof["depth"].to_numpy(), prof["magt"].to_numpy()

    gl = np.linspace(lat.min(), lat.max(), 240)
    gd = np.linspace(0, dep.max(), 140)
    GL, GD = np.meshgrid(gl, gd)
    # 위도-깊이 평면 보간 (convex hull 밖은 자동 NaN → 외삽 억제)
    T = griddata((lat, dep), magt, (GL, GD), method="linear")

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    levels = np.linspace(CLIM[0], CLIM[1], 25)
    cf = ax.contourf(GL, GD, np.clip(T, *CLIM), levels=levels, cmap=CMAP, extend="both")
    # 0 °C 경계선
    cs = ax.contour(GL, GD, T, levels=[0.0], colors="k", linewidths=2.2)
    ax.clabel(cs, fmt="0 °C", fontsize=11)
    # 실측 borehole 위치 표시(상단 삼각형)
    for la in np.unique(lat):
        ax.plot(la, 0, marker="v", color="#222", ms=7, clip_on=False)
    ax.scatter(lat, dep, c=np.clip(magt, *CLIM), cmap=CMAP, vmin=CLIM[0], vmax=CLIM[1],
               s=8, edgecolor="k", linewidth=0.2, zorder=3)

    ax.invert_yaxis()
    ax.set_xlabel("위도 (°N)  —  남(내륙) → 북(해안)", fontsize=13)
    ax.set_ylabel("깊이 (m)", fontsize=13)
    ax.set_title("Alaska 영구동토 위도–깊이 온도 단면  (▼ = borehole 위치)",
                 fontsize=16, fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, label="평균 지중온도 MAGT (°C)")
    cb.add_lines(cs)
    ax.text(0.015, 0.04, "검은 선 = 0 °C 등온면(영구동토 경계).  흰 영역 = 데이터 밖(미보간)",
            transform=ax.transAxes, fontsize=10, color="#444",
            bbox=dict(fc="white", ec="#bbb", alpha=0.8))
    fig.tight_layout()
    out = C.FIGURES / "cross_section_lat_depth.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def run():
    C.ensure_dirs()
    print("Rendering cross-section ...")
    cross_section()
    print("Rendering hero 3D scene ...")
    hero_scene()
    print("Done -> outputs/figures/")
