"""알래스카 전역 고해상도 활성층 두께(ALT) 예측 지도 — 조밀 격자 물리 예측장.

목적
    기존 ALT 지도는 관측점 산포라 공간적으로 성글다. 여기서는 알래스카 전역 0.02° 격자
    (약 100만 셀)에서 Stefan 해를 직접 계산해 연속 예측장을 만들고, 그 위에 실측 관측
    지점을 겹쳐 논문 수록용 단일 지도를 만든다. 새로운 학습은 하지 않는다.

방법 (모두 기존 자산·물리식만 사용)
    1) 기후 강제력: ERA5-Land 월 기후값(2015-2020, 0.1°)을 알래스카 창으로 잘라
       융해도일 TDD, 동결도일 FDD, 연평균기온, 최난·최한월 기온, 적설수당량을 산출한다.
       사이트 공변량(data/processed/alt_era5_covariates.csv)과 동일한 정의·기간이다.
    2) 격자 세분: 0.1° 원장을 0.02° 격자로 이중선형 세분한다. 세분은 렌더용 평활화이며
       정보량은 0.1° 원장에 한정된다(그림 각주에 명시).
    3) 토양 물성: SoilGrids(WCS namerica 창, 약 5 km, IGH 좌표계)에서 용적밀도·모래·자갈·
       유기탄소를 격자점에 이중선형 샘플링한다.
    4) 예측: polar.physics.physics_ensemble 로 열물성·동토상부온도(TTOP)를 계산하고,
       검증에서 최상위였던 Stefan 해 ALT = E x TDD^0.5 를 예측장으로 쓴다.
       E 는 재학습 없이 알래스카 실측에 최소제곱으로 적합한다(보고서의 다른 두 지도와 동일).
       영구동토 범위는 다년평균 연평균기온 < 0 °C 로 정의하며, 그 밖의 육지는 표시하지 않는다.

산출
    outputs/maps/alt_prediction_hires.png (dpi 300)
    outputs/maps/alt_prediction_hires.pdf

실행
    PYTHONPATH=/home/willy010313/Polar_Bigdata/src python scripts/4_visualization/report_alt_map_hires.py
    옵션: --res 0.05 (빠른 확인), --res 0.02 (기본, 논문용)
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from polar.plotstyle import use_polar  # noqa: E402
from polar.geomap import (ALASKA, make_ax, mask_ocean, add_scalebar,  # noqa: E402
                          add_inset_locator)
from polar.physics import physics_ensemble  # noqa: E402
from polar.fidelity import macro_region, fit_stefan_E  # noqa: E402
from polar.outputs import mappath  # noqa: E402


def fidelity_base_alaska():
    """계수 E 적합용 알래스카 실측(다른 보고서 지도와 동일 원본·동일 추정량)."""
    _d = pd.read_csv(os.path.join(ROOT, "data", "processed", "fidelity_base.csv"))
    _d["macro"] = macro_region(_d)
    return _d[_d.macro == "Alaska"]

plt = use_polar()

NC = os.path.join(ROOT, "data/raw/era5land/nh_monthly_2015-2020.nc")
SG_DIR = os.path.join(ROOT, "data/raw/soilgrids_wcs/namerica")
S2_META = os.path.join(ROOT, "data/processed/s2_physics_meta.json")
OBS_CSV = os.path.join(ROOT, "data/processed/dl_dataset_cell_v3_soil.csv")
AK_REGIONS = ("ABoVE_AK", "United States (Alaska)", "GTNPenv_US")

IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
# SoilGrids 정수 스케일 → 물리단위 (enrich_soilgrids_wcs.py SCALE 과 동일)
SG_LAYERS = {"sg_bdod_5_15": ("bdod_5-15cm_mean.tif", 100.0),   # cg/cm3 → g/cm3
             "sg_sand_5_15": ("sand_5-15cm_mean.tif", 10.0),    # g/kg → %
             "sg_cfvo_5_15": ("cfvo_5-15cm_mean.tif", 10.0),    # per-mille → vol%
             "sg_soc_5_15": ("soc_5-15cm_mean.tif", 10.0)}      # dg/kg → g/kg

# 저채도 냉색 규약(전역 그림 규범)
GREY_TXT = "#444444"
FOOT = "#6b6b6b"
LAND_BG = "#efece6"    # geomap 육지색 = 비영구동토(예측 제외) 표시색
# 지도 창(set_extent) 밖으로 계산 격자를 확장해 정사영 프레임 모서리까지 채운다.
PAD_LON, PAD_LAT = 9.0, 3.0


def alt_cmap():
    """ALT 순차형: cmcrameri oslo_r 유지.

    상단(근흑)은 과채도 방지로, 하단(근백)은 육지 배경(#efece6)·해양(#eaf1f4)과의
    혼동 방지로 잘라 낸다. 결과 범위 = 연청회(#b2bccc) → 중명도 진청(#1d4a78).
    """
    from cmcrameri import cm as cmc
    from matplotlib.colors import LinearSegmentedColormap
    cols = cmc.oslo_r(np.linspace(0.20, 0.68, 256))
    cm = LinearSegmentedColormap.from_list("alt_hires", cols)
    cm.set_bad(alpha=0.0)          # 결측·비영구동토 → 배경(육지·해양) 노출
    return cm


# ----------------------------------------------------------------------
# 1) 기후 강제력 격자
# ----------------------------------------------------------------------
def _nearest_fill(arr2d, valid):
    """육지 마스크 밖(해양) 값을 최근접 육지값으로 채운다(해안 세분 시 NaN 잠식 방지)."""
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr2d[tuple(idx)]


def climate_grid(glon, glat, extent, pad=0.6):
    """ERA5-Land 월 기후값 → 조밀 격자 도일·기온·적설. 반환 (dict, land_mask)."""
    import xarray as xr
    _, lo0, lo1, la0, la1 = extent
    ds = xr.open_dataset(NC)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    # 창을 먼저 자른 뒤 월 기후값을 낸다(전지구 배열 적재 회피 — 공유서버 메모리 보호).
    win = ds[["t2m", "stl1", "sd"]].sel(
        latitude=slice(la1 + pad, la0 - pad),
        longitude=slice(lo0 - pad, lo1 + pad)).load()
    clim = win.assign_coords(month=win[tname].dt.month).groupby("month").mean(tname)

    t0 = clim["t2m"].values.astype("float32")          # (12, LAT, LON) K
    land = np.isfinite(t0).all(axis=0)                 # ERA5-Land: 해양=NaN
    src = {}
    for k, v in (("t2m", clim["t2m"]), ("stl1", clim["stl1"]), ("sd", clim["sd"])):
        a = v.values.astype("float32")
        src[k] = np.stack([_nearest_fill(a[m], land) for m in range(a.shape[0])])

    filled = xr.Dataset(
        {k: (("month", "latitude", "longitude"), v) for k, v in src.items()},
        coords={"month": clim["month"].values,
                "latitude": clim["latitude"].values,
                "longitude": clim["longitude"].values})
    fine = filled.interp(latitude=glat, longitude=glon, method="linear")
    lmask = xr.DataArray(land.astype("float32"),
                         coords={"latitude": clim["latitude"].values,
                                 "longitude": clim["longitude"].values},
                         dims=("latitude", "longitude"))
    land_f = lmask.interp(latitude=glat, longitude=glon, method="nearest").values > 0.5

    days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)],
                    dtype="float32")[:, None, None]
    t = fine["t2m"].values - 273.15
    stl = fine["stl1"].values - 273.15
    sd = fine["sd"].values
    tdd = np.nansum(np.clip(t, 0, None) * days, axis=0)
    fdd = np.nansum(np.clip(-t, 0, None) * days, axis=0)
    G = {"e5_maat": np.nanmean(t, axis=0), "e5_tdd": tdd, "e5_fdd": fdd,
         "e5_sqrt_tdd": np.sqrt(np.clip(tdd, 0, None)),
         "e5_twarm": np.nanmax(t, axis=0), "e5_tcold": np.nanmin(t, axis=0),
         "e5_stl1": np.nanmean(stl, axis=0), "e5_swe": np.nanmean(sd, axis=0)}
    ds.close()
    return G, land_f


# ----------------------------------------------------------------------
# 2) 토양 물성 격자
# ----------------------------------------------------------------------
def soil_grid(lon_flat, lat_flat):
    """SoilGrids(IGH) 창에서 격자점 이중선형 샘플. nodata(해양·빙하)는 NaN."""
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt, map_coordinates

    tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
    x, y = tr.transform(lon_flat, lat_flat)
    out = {}
    for col, (fname, factor) in SG_LAYERS.items():
        path = os.path.join(SG_DIR, fname)
        with rasterio.open(path) as src:
            b = src.read(1).astype("float64")
            if src.nodata is not None:
                b[b == src.nodata] = np.nan
            b[b <= 0] = np.nan                     # 0/음수 = 해양·빙하 결측
            valid = np.isfinite(b)
            idx = distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
            fc, fr = (~src.transform) * (x, y)     # 픽셀 경계 좌표
            rr, cc = fr - 0.5, fc - 0.5            # 픽셀 중심 인덱스
            v = map_coordinates(b[tuple(idx)], [rr, cc], order=1, mode="nearest")
            keep = map_coordinates(valid.astype("float32"), [rr, cc],
                                   order=0, mode="nearest") > 0.5
            inb = (fr >= 0) & (fr < src.height) & (fc >= 0) & (fc < src.width)
        out[col] = np.where(keep & inb, v, np.nan) / factor
    return out


# ----------------------------------------------------------------------
# 3) 예측장
# ----------------------------------------------------------------------
def build_field(res, extent):
    """지도 창보다 넓은 계산 격자(정사영 모서리 충전용)에서 예측장을 만든다."""
    name, lo0, lo1, la0, la1 = extent
    wide = (name, lo0 - PAD_LON, lo1 + PAD_LON,
            max(la0 - PAD_LAT, 50.0), min(la1 + PAD_LAT, 82.0))
    _, wlo0, wlo1, wla0, wla1 = wide
    glon = np.arange(wlo0, wlo1 + res / 2, res)
    glat = np.arange(wla0, wla1 + res / 2, res)
    G, land = climate_grid(glon, glat, wide)
    ny, nx = land.shape
    print(f"[격자] {nx} x {ny} = {nx*ny:,} 셀 (해상도 {res}°), 육지 {land.sum():,}")

    lon2d, lat2d = np.meshgrid(glon, glat)
    sel = land.ravel()
    cols = {k: v.ravel()[sel].astype("float64") for k, v in G.items()}
    cols.update(soil_grid(lon2d.ravel()[sel], lat2d.ravel()[sel]))
    df = pd.DataFrame(cols)
    for c in SG_LAYERS:
        print(f"[토양] {c}: 유효 {np.isfinite(df[c]).mean()*100:.1f}%  "
              f"중앙 {np.nanmedian(df[c]):.3f}")

    # 계수와 영구동토 범위는 보고서의 다른 두 지도(연별 ALT 장, 결합 모델 지도)와 맞춘다.
    # 종전에는 전 지역 적합 E_global(1.5710)과 TTOP<0 마스크를 써서, 같은 알래스카 ALT를
    # 그린 세 지도의 값과 범위가 서로 달랐다. 알래스카 실측 최소제곱 E와 연평균기온 기준으로
    # 통일한다. 연평균기온 기준을 택한 것은 세 지도가 모두 산출할 수 있는 유일한 공통 기준이기
    # 때문이다(TTOP는 토양 격자가 필요해 연별 지도에서 산출할 수 없다).
    ak = fidelity_base_alaska()
    E = float(fit_stefan_E(ak["alt_cm"].values.astype(float),
                           ak["e5_sqrt_tdd"].values.astype(float)))
    phys = physics_ensemble(df, E)
    alt = np.full(ny * nx, np.nan)
    alt[sel] = phys["p1_stefan"]
    alt = alt.reshape(ny, nx)
    pf = G["e5_maat"] < 0.0                         # 영구동토 범위(다년평균 연평균기온 < 0 °C)
    alt_pf = np.where(pf & land, alt, np.nan)
    print(f"[예측] E={E:.4f}(알래스카 실측 최소제곱) | 영구동토 셀 "
          f"{int((pf & land).sum()):,} / 육지 {land.sum():,} "
          f"({(pf & land).sum()/max(land.sum(),1)*100:.1f}%)")
    print(f"[예측] ALT {np.nanpercentile(alt_pf,1):.0f}–{np.nanpercentile(alt_pf,99):.0f} cm "
          f"(중앙 {np.nanmedian(alt_pf):.0f})")
    return glon, glat, alt_pf, E


def load_obs(extent, dedup_deg=0.05):
    _, lo0, lo1, la0, la1 = extent
    d = pd.read_csv(OBS_CSV, low_memory=False,
                    usecols=["lat", "lon", "region", "alt_cm"])
    d = d[d.region.isin(AK_REGIONS)]
    d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)]
    g = (d.assign(_la=(d.lat / dedup_deg).round(), _lo=(d.lon / dedup_deg).round())
           .groupby(["_la", "_lo"], as_index=False)
           .agg(lat=("lat", "mean"), lon=("lon", "mean"), alt_cm=("alt_cm", "mean")))
    print(f"[관측] 원 셀 {len(d):,} → {dedup_deg}° 중복제거 {len(g):,} 지점")
    return g


def check_against_obs(glon, glat, alt, obs):
    """지도 격자값 vs 관측(정합성 점검용 표준출력, 그림에는 넣지 않음)."""
    iy = np.clip(np.searchsorted(glat, obs.lat.values) - 1, 0, len(glat) - 1)
    ix = np.clip(np.searchsorted(glon, obs.lon.values) - 1, 0, len(glon) - 1)
    p = alt[iy, ix]
    m = np.isfinite(p)
    if m.sum() < 10:
        return
    e = p[m] - obs.alt_cm.values[m]
    print(f"[점검] 관측 {m.sum():,}지점 대비 RMSE {np.sqrt((e**2).mean()):.1f} cm, "
          f"bias {e.mean():+.1f} cm (동토 격자에 든 지점만)")


# ----------------------------------------------------------------------
# 4) 렌더
# ----------------------------------------------------------------------
def render(glon, glat, alt, obs, E, res, extent=ALASKA, out="alt_prediction_hires"):
    import cartopy.crs as ccrs
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    # 색범위는 표시 창(알래스카) 안의 분위수로 잡는다 — 주변부 값에 끌려가지 않게.
    _, lo0, lo1, la0, la1 = extent
    iw = np.ix_((glat >= la0) & (glat <= la1), (glon >= lo0) & (glon <= lo1))
    win = alt[iw]
    v = win[np.isfinite(win)]
    print("[분포] " + " ".join(f"p{q}={np.percentile(v, q):.0f}"
                              for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)))
    vmin = float(np.floor(np.percentile(v, 5) / 5) * 5)
    vmax = float(np.ceil(np.percentile(v, 97) / 5) * 5)
    clipped = float(np.mean((v < vmin) | (v > vmax)) * 100)
    print(f"[색] vmin={vmin:.0f} vmax={vmax:.0f} cm, 범위 밖 {clipped:.1f}%")

    fig, ax = make_ax(extent, figsize=(8.8, 6.6), grid=True, grid_labels=True)
    gl = ax._gl
    if gl is not None:
        gl.rotate_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-170, -134, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(58, 74, 2))
        gl.xlabel_style = gl.ylabel_style = {"size": 8.5, "color": "0.35"}

    # 기후 구동장은 원자료가 0.1도라 연속 래스터로 그리면 뿌옇게 보인다. 등치대(filled
    # contour)로 그려 경계를 선명하게 하고, 5 cm 간격 등치선으로 구조를 드러낸다.
    cmap = alt_cmap()
    step = max(1, int(round(0.04 / res)))          # 등치선 계산용 간축(속도)
    lo, hi = int(np.floor(vmin / 5) * 5), int(np.ceil(vmax / 5) * 5)
    levels = np.arange(lo, hi + 5, 5)
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")
    A = np.ma.masked_invalid(alt)
    im = ax.contourf(glon, glat, A, levels=levels, cmap=cmap, norm=norm,
                     extend="both", transform=ccrs.PlateCarree(),
                     zorder=1.5, antialiased=True)
    for c in im.collections:                        # PDF 흰 실선 아티팩트 제거
        c.set_edgecolor("face")
        c.set_rasterized(True)
    ax.contour(glon[::step], glat[::step], A[::step, ::step], levels=levels,
               colors="#ffffff", linewidths=0.25, alpha=0.55,
               transform=ccrs.PlateCarree(), zorder=2.4)
    lv_bold = [x for x in levels if x % 15 == 0]
    if lv_bold:
        ax.contour(glon[::step], glat[::step], A[::step, ::step], levels=lv_bold,
                   colors="#2c3e50", linewidths=0.45, alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=2.5)

    mask_ocean(ax, zorder=3)
    ax.scatter(obs.lon.values, obs.lat.values, s=3.0, c="#1f1f1f",
               linewidths=0, alpha=0.8, transform=ccrs.PlateCarree(), zorder=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, shrink=0.84,
                      extend="both")
    cb.set_label("ALT (cm)", fontsize=10, color=GREY_TXT)
    cb.ax.tick_params(labelsize=9, length=2.5, color=GREY_TXT,
                      labelcolor=GREY_TXT)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#9a9a9a")

    handles = [plt.Line2D([], [], marker="o", ls="none", ms=3.2, mfc="#1f1f1f",
                          mec="none", label=f"관측 지점 (n={len(obs):,})"),
               mpatches.Patch(fc=LAND_BG, ec="#b8b3aa", lw=0.4,
                              label="영구동토 외 육지 (예측 제외)")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=9,
                    frameon=True, facecolor="white", framealpha=0.72,
                    edgecolor="none", handletextpad=0.6, borderpad=0.5)
    leg.set_zorder(6)
    for t in leg.get_texts():
        t.set_color(GREY_TXT)

    add_scalebar(ax, fontsize=8)
    add_inset_locator(fig, ax, extent, size=0.22)

    # 산출 조건(격자·계수·자료원)은 보고서 캡션이 담당하므로 그림 내부 각주는 두지 않는다.

    png, pdf = mappath(out), mappath(out, ext="pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("saved", png)
    print("saved", pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02, help="격자 해상도(도)")
    a = ap.parse_args()
    glon, glat, alt, E = build_field(a.res, ALASKA)
    obs = load_obs(ALASKA)
    check_against_obs(glon, glat, alt, obs)
    render(glon, glat, alt, obs, E, a.res, extent=ALASKA)


if __name__ == "__main__":
    main()
