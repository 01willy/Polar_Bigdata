"""
③ 시각화 — 전처리 결과를 그림으로.

산출(outputs/figures, outputs/meshes):
  01_alaska_overview.png   지도/깊이분포/프로파일/지온경사 4-패널 (matplotlib)
  02_3d_boreholes.png      3D borehole stick (PyVista, 온도 색) + 예비 0 °C 등온면
  permafrost_base_preview.vtp  예비 0 °C 등온면 tri-mesh (있을 때)

헤드리스 서버 가정: matplotlib=Agg, PyVista off_screen=True.
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from . import config as C

# 한글 폰트 설정 (헤드리스 서버: NanumGothic 우선)
_KFONTS = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"]
for _fp in _KFONTS:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지

# 위/경도 -> 로컬 미터 (Alaska 국소 평면 근사; geopandas/pyproj 불필요)
_M_PER_DEG_LAT = 110_540.0


def _to_local_xy(lon, lat, lon0, lat0):
    x = (np.asarray(lon) - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (np.asarray(lat) - lat0) * _M_PER_DEG_LAT
    return x, y


def overview_figure(prof, summ):
    """4-패널 개요 그림."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (a) 지도: borehole 위치, 색=10 m MAGT(계절편향 적음), 크기=최대깊이
    ax = axes[0, 0]
    sc = ax.scatter(summ["lon"], summ["lat"], c=summ["magt_10m"],
                    s=20 + summ["max_depth"], cmap="coolwarm", vmin=-10, vmax=2,
                    edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="MAGT @10 m (°C)")
    ax.set(title=f"(a) Alaska boreholes (n={len(summ)})  size∝max depth",
           xlabel="longitude", ylabel="latitude")
    ax.grid(alpha=0.3)

    # (b) 최대깊이 분포
    ax = axes[0, 1]
    ax.hist(summ["max_depth"], bins=np.arange(0, summ["max_depth"].max() + 5, 5),
            color="#3a7", edgecolor="k")
    ax.axvline(15, color="r", ls="--", label="15 m (deep cutoff)")
    ax.set(title="(b) borehole 최대깊이 분포", xlabel="max depth (m)", ylabel="boreholes")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) 수직 프로파일: 가장 깊은 6개 borehole의 MAGT vs depth
    ax = axes[1, 0]
    deep = summ.nlargest(6, "max_depth")["borehole_id"]
    for bid in deep:
        g = prof[prof["borehole_id"] == bid].sort_values("depth")
        s = summ[summ["borehole_id"] == bid].iloc[0]
        ax.plot(g["magt"], g["depth"], "-o", ms=3, label=f"{s['site']} ({s['max_depth']:.0f}m)")
    ax.axvline(0, color="b", ls="--", lw=1)
    ax.invert_yaxis()
    ax.set(title="(c) 수직 MAGT 프로파일 (최심부 6개)", xlabel="MAGT (°C)", ylabel="depth (m)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) 위도 vs 10 m MAGT (+ 지온경사 색)
    ax = axes[1, 1]
    sc = ax.scatter(summ["lat"], summ["magt_10m"],
                    c=summ["geo_gradient_C_per_m"] * 1000, cmap="viridis",
                    s=40, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="지온경사 (°C/km)")
    ax.axhline(0, color="b", ls="--", lw=1)
    ax.set(title="(d) 위도 vs MAGT@10 m", xlabel="latitude", ylabel="MAGT @10 m (°C)")
    ax.grid(alpha=0.3)

    fig.suptitle("Alaska 영구동토 borehole — GTN-P 전처리 개요", fontsize=15, y=0.995)
    fig.tight_layout()
    out = C.FIGURES / "01_alaska_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def _rbf_isosurface(prof, ve, lon0, lat0):
    """프로파일 점으로 3D RBF 보간 → 0 °C 등온면 추출(예비). 실패 시 None."""
    try:
        import pyvista as pv
        from scipy.interpolate import RBFInterpolator
    except Exception as e:
        print("  isosurface skipped:", e)
        return None
    g = prof.dropna(subset=["magt", "lat", "lon", "depth"])
    if len(g) < 30:
        return None
    x, y = _to_local_xy(g["lon"], g["lat"], lon0, lat0)
    z = -g["depth"].to_numpy() * ve
    pts = np.column_stack([x, y, z])
    vals = g["magt"].to_numpy()

    rbf = RBFInterpolator(pts, vals, kernel="linear", smoothing=1.0)
    # 데이터 범위 내 coarse 격자
    nx = ny = 40; nz = 30
    gx = np.linspace(x.min(), x.max(), nx)
    gy = np.linspace(y.min(), y.max(), ny)
    gz = np.linspace(z.min(), z.max(), nz)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    T = rbf(grid_pts).reshape(nx, ny, nz)

    img = pv.ImageData(dimensions=(nx, ny, nz),
                       spacing=(gx[1] - gx[0], gy[1] - gy[0], gz[1] - gz[0]),
                       origin=(gx[0], gy[0], gz[0]))
    img.point_data["MAGT"] = T.flatten(order="F")
    try:
        surf = img.contour([0.0], scalars="MAGT", method="flying_edges")
    except Exception as e:
        print("  contour failed:", e)
        return None
    if surf.n_points == 0:
        return None
    surf.save(C.MESHES / "permafrost_base_preview.vtp")
    print(f"  saved {C.MESHES / 'permafrost_base_preview.vtp'}  ({surf.n_points} pts)")
    return surf


def scene_3d(prof, summ, ve=4000):
    """3D borehole stick (온도 색) + 예비 0 °C 등온면. PNG 저장."""
    try:
        import pyvista as pv
    except Exception as e:
        print("  3D scene skipped (no pyvista):", e)
        return None
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv.OFF_SCREEN = True

    lat0 = float(prof["lat"].mean()); lon0 = float(prof["lon"].mean())
    p = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    p.set_background("white")

    clim = (-12, 2)
    tubes = []
    for _bid, g in prof.groupby("borehole_id"):
        g = g.dropna(subset=["magt"]).sort_values("depth")
        if len(g) < 2:
            continue
        x, y = _to_local_xy(g["lon"].to_numpy(), g["lat"].to_numpy(), lon0, lat0)
        z = -g["depth"].to_numpy() * ve
        line = pv.lines_from_points(np.column_stack([x, y, z]))
        line["MAGT"] = np.clip(g["magt"].to_numpy(), *clim)
        tubes.append(line.tube(radius=8000))

    # 모든 borehole 막대를 하나로 병합 → 단일 컬러바
    merged = tubes[0].merge(tubes[1:]) if len(tubes) > 1 else tubes[0]
    p.add_mesh(merged, scalars="MAGT", cmap="coolwarm", clim=clim,
               show_scalar_bar=True,
               scalar_bar_args=dict(title="MAGT (degC)", n_labels=5,
                                    fmt="%.0f", title_font_size=18, label_font_size=14))

    # 표층 점(지점 위치)
    xs, ys = _to_local_xy(summ["lon"].to_numpy(), summ["lat"].to_numpy(), lon0, lat0)
    top = pv.PolyData(np.column_stack([xs, ys, np.zeros_like(xs)]))
    p.add_mesh(top, color="black", point_size=6, render_points_as_spheres=True)

    # 예비 0 °C 등온면
    surf = _rbf_isosurface(prof, ve, lon0, lat0)
    if surf is not None:
        p.add_mesh(surf, color="#66ccff", opacity=0.35, show_scalar_bar=False)
    p.add_text(f"Alaska permafrost boreholes 3D (vertical exaggeration x{ve})\n"
               f"sticks = MAGT;  cyan surface = preliminary 0 degC isotherm",
               font_size=11, color="black")
    p.add_axes()
    p.view_isometric()
    p.camera.azimuth = 25
    p.camera.elevation = 18
    p.camera.zoom(1.25)
    out = C.FIGURES / "02_3d_boreholes.png"
    p.screenshot(str(out))
    p.close()
    print(f"  saved {out}")
    return out


def run():
    C.ensure_dirs()
    prof = pd.read_csv(C.PROFILES_CSV)
    summ = pd.read_csv(C.SUMMARY_CSV)
    print("Rendering overview ...")
    overview_figure(prof, summ)
    print("Done. -> outputs/figures/")
    # 3D는 viz3d.hero_scene()이 더 세련된 버전을 생성(scene_3d는 보존하되 미호출)
