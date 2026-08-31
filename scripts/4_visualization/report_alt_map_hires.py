"""알래스카 전역 고해상도 활성층 두께(ALT) 예측 지도 — 조밀 격자 물리 예측장.

목적
    기존 ALT 지도는 관측점 산포라 공간적으로 성글다. 여기서는 알래스카 전역 0.02° 격자
    (약 100만 셀)에서 학습 모델로 연속 예측장을 만들고, 그 위에 실측 셀 위치를 겹쳐
    논문 수록용 단일 지도를 만든다.
    실측 셀(13,606개)은 그대로 찍으면 서로 겹치므로 0.05° 격자로 집계해 표시한다.
    범례의 n은 관측 지점 수가 아니라 집계 위치 수다.

방법 (모두 기존 자산·물리식만 사용)
    1) 기후 강제력: ERA5-Land 월 기후값(2015-2020, 0.1°)을 알래스카 창으로 잘라
       융해도일 TDD, 동결도일 FDD, 연평균기온, 최난·최한월 기온, 적설수당량을 산출한다.
       사이트 공변량(data/processed/alt_era5_covariates.csv)과 동일한 정의·기간이다.
    2) 격자 세분: 0.1° 원장을 0.02° 격자로 이중선형 세분한다. 세분은 렌더용 평활화이며
       정보량은 0.1° 원장에 한정된다(그림 각주에 명시).
    3) 토양 물성: SoilGrids(WCS namerica 창, 약 5 km, IGH 좌표계) 9종을 격자점에
       이중선형 샘플링한다.
    4) 예측: 격자에서 전부 채울 수 있는 공변량 17종(기후 8 + 토양 9)으로 CatBoost를
       알래스카 실측 13,606셀에 학습해 격자에 적용한다(3-seed 앙상블 예측).
       이 입력 집합에서의 모델 비교는 scripts/2_evaluation/map_model_gate.py 이며,
       다층 퍼셉트론 13.86 · CatBoost 13.77 · Stefan 2모수 13.93 · Stefan 1모수 14.46 cm다
       (알래스카 0.5° 공간블록 6-fold). 지형 6종은 알래스카 전역 DEM 미확보로,
       SAR 8종은 관측 지점에만 존재해 격자 산출 불가로 제외한다.
       영구동토 범위는 다년평균 연평균기온 < 0 °C 로 정의하며, 그 밖의 육지는 표시하지 않는다.

산출
    outputs/maps/alt_prediction_hires.png (dpi 300)
    outputs/maps/alt_prediction_hires.pdf
    폭은 보고서 삽입 크기(0.54 x textwidth = 95.04 mm)와 같게 잡고 tight 크롭을 쓰지
    않으므로 축소 배율은 1.0이다. 즉 스크립트의 fontsize(pt)가 곧 인쇄 pt다.

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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "9"))

ROOT = "/home/willy010313/Polar_Bigdata"
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from polar.plotstyle import use_polar  # noqa: E402
from polar.geomap import (ALASKA, make_ax, mask_ocean, add_scalebar,  # noqa: E402
                          add_inset_locator, map_projection, projected_aspect)
from polar.tab_models import fit_predict, NAN_NATIVE, set_device  # noqa: E402
from polar.preprocessing import fold_prep  # noqa: E402
from polar.fidelity import macro_region, fit_stefan_E  # noqa: E402
from polar.outputs import mappath  # noqa: E402


def fidelity_base_alaska():
    """계수 E 적합용 알래스카 실측(다른 보고서 지도와 동일 원본·동일 추정량)."""
    _d = pd.read_csv(os.path.join(ROOT, "data", "processed", "fidelity_base.csv"))
    _d["macro"] = macro_region(_d)
    return _d[_d.macro == "Alaska"]

plt = use_polar()
# 축 사각형을 인치로 직접 배치하므로 tight 크롭을 끈다(최종 인쇄 크기를 확정하기 위함).
plt.rcParams["savefig.bbox"] = None

NC = os.path.join(ROOT, "data/raw/era5land/nh_monthly_2015-2020.nc")
SG_DIR = os.path.join(ROOT, "data/raw/soilgrids_wcs/namerica")
OBS_CSV = os.path.join(ROOT, "data/processed/dl_dataset_cell_v3_soil.csv")
# 보고서 본문·표의 알래스카 학습·평가 표본(13,606셀)과 동일한 지역 정의.
# GTN-P 알래스카 9셀은 그 표본에 들어 있지 않으므로 지도 관측점에서도 제외한다.
AK_REGIONS = ("ABoVE_AK", "United States (Alaska)")
# 관측 셀은 0.05° 격자로 집계해 표시한다(원 셀 13,606개는 겹쳐 그려져 분포를 가린다).
DEDUP_DEG = 0.05

IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
# SoilGrids 정수 스케일 → 물리단위 (enrich_soilgrids_wcs.py SCALE 과 동일)
SG_LAYERS = {"sg_bdod_5_15": ("bdod_5-15cm_mean.tif", 100.0),   # cg/cm3 → g/cm3
             "sg_sand_5_15": ("sand_5-15cm_mean.tif", 10.0),    # g/kg → %
             "sg_silt_5_15": ("silt_5-15cm_mean.tif", 10.0),    # g/kg → %
             "sg_clay_5_15": ("clay_5-15cm_mean.tif", 10.0),    # g/kg → %
             "sg_cfvo_5_15": ("cfvo_5-15cm_mean.tif", 10.0),    # per-mille → vol%
             "sg_phh2o_5_15": ("phh2o_5-15cm_mean.tif", 10.0),  # pH*10 → pH
             "sg_soc_0_5": ("soc_0-5cm_mean.tif", 10.0),        # dg/kg → g/kg
             "sg_soc_5_15": ("soc_5-15cm_mean.tif", 10.0),      # dg/kg → g/kg
             "sg_soc_15_30": ("soc_15-30cm_mean.tif", 10.0)}    # dg/kg → g/kg

# 격자에서 전부 채울 수 있는 공변량(= 예측 모델 입력). 지형 6종은 알래스카 전역 DEM
# 미확보로, SAR 8종은 관측 지점에만 존재해 격자 산출 불가로 제외한다.
# 이 집합에서 모델을 재검증한 결과가 data/processed/map_model_gate_meta.json 이다.
CLIM_COLS = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd",
             "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
GRIDDABLE = CLIM_COLS + ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15",
                         "sg_bdod_5_15", "sg_cfvo_5_15", "sg_phh2o_5_15",
                         "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
MAP_MODEL = "catboost"  # 게이트 최상위(13.77 cm). 트리 모델은 학습 라벨 범위 밖으로
                       # 외삽하지 않아 관측 없는 산지에서 비물리적 값이 생기지 않는다.
MAP_SEEDS = (0, 1, 2)  # 표 4와 같은 3-seed 앙상블 예측 규약

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

    # 예측 모델. 격자에서 전부 채울 수 있는 공변량 17종으로 재검증한 결과
    # (map_model_gate: 다층 퍼셉트론 13.86 · CatBoost 13.77 · Stefan 2모수 13.93 ·
    #  Stefan 1모수 14.46 cm, 알래스카 0.5° 공간블록 6-fold)에서 최상위권인
    # 다층 퍼셉트론을 쓴다. 학습 표본은 보고서 표본과 같은 알래스카 실측 13,606셀이다.
    # 영구동토 범위는 다년평균 연평균기온 < 0 °C 로 두어 연별 ALT 지도와 기준을 맞춘다
    # (TTOP 기준은 토양 격자가 필요해 연별 지도에서 산출할 수 없다).
    ak = fidelity_base_alaska()
    Xtr = ak[GRIDDABLE].to_numpy("float32")
    ytr = ak["alt_cm"].to_numpy(float)
    Xte = df[GRIDDABLE].to_numpy("float32")
    Xtr_p, Xte_p = fold_prep(Xtr, Xte, MAP_MODEL in NAN_NATIVE)
    preds = [fit_predict(MAP_MODEL, Xtr_p, ytr, Xte_p, seed=sd)["pred"]
             for sd in MAP_SEEDS]
    alt = np.full(ny * nx, np.nan)
    alt[sel] = np.clip(np.mean(preds, axis=0), 1.0, 400.0)
    alt = alt.reshape(ny, nx)
    E = float(fit_stefan_E(ak["alt_cm"].values.astype(float),
                           ak["e5_sqrt_tdd"].values.astype(float)))
    pf = G["e5_maat"] < 0.0                         # 영구동토 범위(다년평균 연평균기온 < 0 °C)
    alt_pf = np.where(pf & land, alt, np.nan)
    print(f"[예측] 모델 {MAP_MODEL} · 입력 {len(GRIDDABLE)}종 · "
          f"학습 {len(ak):,}셀 · seed {len(MAP_SEEDS)}회 앙상블")
    print(f"[예측] 영구동토 셀 {int((pf & land).sum()):,} / 육지 {land.sum():,} "
          f"({(pf & land).sum()/max(land.sum(),1)*100:.1f}%)")
    print(f"[예측] ALT {np.nanpercentile(alt_pf,1):.0f}–{np.nanpercentile(alt_pf,99):.0f} cm "
          f"(중앙 {np.nanmedian(alt_pf):.0f})")
    return glon, glat, alt_pf, E


def load_obs(extent, dedup_deg=DEDUP_DEG):
    _, lo0, lo1, la0, la1 = extent
    d = pd.read_csv(OBS_CSV, low_memory=False,
                    usecols=["lat", "lon", "region", "alt_cm"])
    d = d[d.region.isin(AK_REGIONS)]
    d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)]
    g = (d.assign(_la=(d.lat / dedup_deg).round(), _lo=(d.lon / dedup_deg).round())
           .groupby(["_la", "_lo"], as_index=False)
           .agg(lat=("lat", "mean"), lon=("lon", "mean"), alt_cm=("alt_cm", "mean")))
    print(f"[관측] 알래스카 원 셀 {len(d):,} → {dedup_deg}° 집계 {len(g):,} 위치")
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
# 인쇄 1:1 배치. 보고서 삽입 폭(0.54 x textwidth = 95.04 mm)과 같은 figsize로 만들고
# tight 크롭을 쓰지 않으므로, 아래 fontsize 값이 그대로 인쇄 pt가 된다(축소 배율 1.0).
FIG_W = 95.04 / 25.4                    # in — \includegraphics[width=0.54\textwidth]
M_L, M_R = 0.36, 0.03                   # 좌(위도 라벨)·우 여백
M_T, M_B = 0.08, 0.23                   # 상·하(경도 라벨) 여백
CB_GAP, CB_W, CB_LAB = 0.07, 0.085, 0.31   # 컬러바 블록(간격·폭·눈금/라벨 영역)
# 그림 안 글자 크기(pt). 인쇄 하한 6.5 pt 이상.
FS_GRID, FS_CB_TICK, FS_CB_LAB = 7.0, 7.0, 7.5
FS_LEGEND, FS_SCALE = 7.0, 7.0


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

    # 투영 후 지도 종횡비로 축 사각형을 인치 단위로 확정한다(축소 없이 저장).
    asp = projected_aspect(extent)
    map_w = FIG_W - M_L - M_R - CB_GAP - CB_W - CB_LAB
    map_h = map_w / asp
    fig_h = M_T + map_h + M_B
    x_cb = M_L + map_w + CB_GAP
    print(f"[배치] 종횡비 {asp:.4f} · 지도 {map_w*25.4:.1f} x {map_h*25.4:.1f} mm · "
          f"전체 {FIG_W*25.4:.2f} x {fig_h*25.4:.2f} mm (축소 배율 1.0)")

    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([M_L / FIG_W, M_B / fig_h, map_w / FIG_W, map_h / fig_h],
                      projection=map_projection(extent))
    make_ax(extent, ax=ax, fig=fig, grid=True, grid_labels=True)
    gl = ax._gl
    if gl is not None:
        gl.rotate_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-170, -134, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(58, 74, 2))
        gl.xlabel_style = gl.ylabel_style = {"size": FS_GRID, "color": "0.35"}

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
    # matplotlib 3.8은 ContourSet.collections를 폐기 예고하며 경고를 낸다. 대체 API
    # (im.set_edgecolor("face") + ax.set_rasterization_zorder)로 바꿔 대조 렌더한 결과,
    # 영구동토 마스크 경계에서 등치면 가장자리 반픽셀이 어긋나 PNG 37,500 px(전체 1.8%,
    # 최대 ΔRGB 197)가 달라졌다. 등치선 형상·색·마스크는 동일하고 차이는 경계
    # 안티에일리어싱뿐이지만, 산출이 바뀌므로 현행 구현을 유지한다(경고는 남는다).
    for c in im.collections:                        # PDF 흰 실선 아티팩트 제거
        c.set_edgecolor("face")
        c.set_rasterized(True)
    # 학습 모델 예측장은 물리식 단독보다 국소 변동이 커서 5 cm 간격 등치선을 전부 그리면
    # 선 잡음이 된다. 등치대 경계가 이미 구조를 나타내므로 굵은 등치선만 남긴다.
    lv_bold = [x for x in levels if x % 15 == 0]
    if lv_bold:
        ax.contour(glon[::step], glat[::step], A[::step, ::step], levels=lv_bold,
                   colors="#2c3e50", linewidths=0.45, alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=2.5)

    mask_ocean(ax, zorder=3)
    ax.scatter(obs.lon.values, obs.lat.values, s=3.0, c="#1f1f1f",
               linewidths=0, alpha=0.8, transform=ccrs.PlateCarree(), zorder=4)

    cax = fig.add_axes([x_cb / FIG_W, M_B / fig_h, CB_W / FIG_W, map_h / fig_h])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("ALT (cm)", fontsize=FS_CB_LAB, color=GREY_TXT, labelpad=3)
    cb.ax.tick_params(labelsize=FS_CB_TICK, length=2.0, pad=1.8, color=GREY_TXT,
                      labelcolor=GREY_TXT)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#9a9a9a")

    # 점 하나는 관측 지점이 아니라 0.05° 집계 위치다(그림 5와 같은 표기 규칙).
    # n은 집계 위치 수이지 실측 셀 수가 아니므로, 문구에서 이를 먼저 밝힌다.
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=3.2, mfc="#1f1f1f",
                          mec="none",
                          label=f"실측 셀 {DEDUP_DEG:g}° 집계 위치 (n={len(obs):,})"),
               mpatches.Patch(fc=LAND_BG, ec="#b8b3aa", lw=0.4,
                              label="영구동토 외 육지 (예측 제외)")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=FS_LEGEND,
                    frameon=True, facecolor="white", framealpha=0.72,
                    edgecolor="none", handletextpad=0.6, borderpad=0.5)
    leg.set_zorder(6)
    for t in leg.get_texts():
        t.set_color(GREY_TXT)

    add_scalebar(ax, fontsize=FS_SCALE)
    add_inset_locator(fig, ax, extent, size=0.22)

    # 산출 조건(격자·계수·자료원)은 보고서 캡션이 담당하므로 그림 내부 각주는 두지 않는다.

    png, pdf = mappath(out), mappath(out, ext="pdf")
    fig.savefig(png, dpi=300)          # bbox 크롭 없음 — figsize 그대로 저장
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[글자] 최소 {min(FS_GRID, FS_CB_TICK, FS_CB_LAB, FS_LEGEND, FS_SCALE):.1f} pt "
          f"(인쇄 크기 동일)")
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
