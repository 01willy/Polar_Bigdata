"""G4 입력 자료 (덱 전용 쪽, 12 × 4.65 in = 3600 × 1395 px, 300 dpi, 영문 FreeSans).

v10 개정(2026-10-16, 사용자 지시): 그림 안 글자는 모두 영어(논문 어휘). 띠 머리(Yearly inputs · Static covariates · Products)와 같은 띠의
썸네일 제목·단위를 모두 va='baseline' 으로 같은 기준선(y_base = 띠 위 − y_lab − 0.03 in)에 둔다(머리가 제목보다 높아 보이던 결함). 정적 공변량의
다섯째 칸은 지도 범위 기준(ESA CCI 영구동토 확률 ≥ 50 %, 모델 입력 아님)으로 'Map domain' 이라 적는다. 저장 전 한글 글자를 찾으면 멈춘다.

v9 개정(2026-10-15, 설계 지침 v9 항 10; 기본판에 적용, --literal 판은 v8 배치 그대로)
- 띠 머리 위 구분선과 머리 글자 사이 0.13 in 이상(머리 글자 가운데를 띠 위에서 0.24 in 아래로).
- 썸네일 아래 출처 줄을 지우고(출처·해상도는 띠 머리 아래 글줄에만), 기준 썸네일의 위도 눈금·점선을 지웠다. 지도 높이 1.02 → 1.06 in(띠 높이 1.55 in 에서 최대).
- '예시'·'2010–2024 중 2014' 같은 설명 대신 '2014년' 만 적는다. 공용 축척 막대(500 km)는 머리 열 아래, 띠 1 지도 바닥 높이.

명세(docs/DESIGN_BRIEF_v8_2026-10-15.md 2절 6쪽, docs/FIGURE_CONTRACT_v7.md)
- 메시지: 모형에 들어가는 입력과 비교 대상 제품을 실자료 썸네일(알래스카, 2014)로 한눈에 보인다. 글은 라벨(1–3 단어)과 출처·해상도 줄뿐이다.
- 기본판(G4_inputs): 3 띠 × 5 칸 격자. 왼쪽 열은 띠 머리(주황 굵게, M.head = G0·G2·G5 와 같은 토큰 title_orange)와 출처·해상도·기간(회색).
    띠 1 연도별 입력: ERA5-Land 융해 도일·동결 도일·적설 수당량(9 km, 2014) + MODIS 여름 NDVI·적설 종료일(라벨 셀 점, 2014).
    띠 2 정적 공변량: 고도·경사(Copernicus DEM 30 m), 토양 유기탄소·점토(SoilGrids 250 m, 5 km 추출창). 다섯째 칸 영구동토 확률(ESA CCI PFR 0.1° 평균)은
    모델 입력이 아니라 지도 범위 기준(≥ 50 %)이다(2026-10-15 정정: 주 입력 23개 = 기후 8 · 지형 6 · 토양 9).
    띠 3 비교 제품: Wei 2026 · Liu 2024 · ESA CCI v5 · Yi–Kimball 2019 (2014) · Aalto 2018 (2000–2014 평균), 같은 ALT 색 범위·색 막대 하나.
  (지시의 4 행 가운데 '그 해 강제력'과 '위성 입력'은 둘 다 연도별 입력이므로 한 띠에 두어 5 칸 격자를 채웠다. 제품은 지시의 3 종에
   Yi–Kimball·Aalto 를 더해 5 종, CCI 는 점 대신 래스터. 이 선택은 사용자 승인 대기이며 --literal 로 지시 그대로의 판을 따로 만든다.)
- 지시 그대로 판(--literal, G4_inputs_literal): 4 띠(그 해 강제력 3 · 정적 공변량 5 · 위성 입력 2 · 비교 제품 3: Wei·Liu·CCI 라벨 셀 점),
  격자 자료는 매 6번째 셀(ky, kx 모두 6 의 배수) 추출을 셀 크기 사각 점으로 그린다. 띠가 하나 늘어 지도 높이 0.66 in.
- 지도: PlateCarree, 가로/세로 비 = 1/cos 65°. 기본판의 격자 자료는 0.06° × 0.025°(≈ 2.9 × 2.8 km) 상자 평균으로 줄여 그린다.
  바탕은 Natural Earth 10m 육지(옅은 회색)·해안선. 자료 밖 육지는 바탕색 그대로. 15 썸네일의 범위가 같으므로 기준 썸네일(띠 1 칸 1)
  하나에만 위도 눈금(65°N·70°N, 틀 왼쪽 밖)과 점선 위도선을 두고(경도 라벨 '150°W' 는 12 pt 에서 0.54 in 라 썸네일 아래 출처 줄과 겹쳐 뺐다), 그 왼쪽(머리 열 아래)에 공용 축척 막대(500 km, 65°N 기준)를 둔다.
- 색: 순차 색표는 cmcrameri(무지개 없음), 변수마다 하나. 제품 ALT 는 지도 그림과 같은 oslo_r(MV.CM_ALT). 제품 이름은 토큰 제품 색(S.PRODUCT_HEX).
  색 범위는 백분위(2–98, 제품은 합쳐서 2–95)로 자르고, 범위 밖 값이 있는 쪽에 색 막대 삼각형(extend)을 붙여 포화를 표시한다.
- 글자: 띠 머리 15 pt 굵게 주황, 썸네일 라벨 13 pt, 출처·단위·색 막대·경위도 눈금 12 pt(회색은 text_aux).
- 자료(모두 파일에서 읽는다):
    data/processed/map_alaska/alaska_grid_x25_v1.csv.gz(정적 공변량, gray == 0 셀), data/processed/v5/maps/Alaska/pred_2014.csv.gz(e5_tdd 2014),
    data/processed/v5/grid/alaska_forcing_year.csv(2014 행의 e5_fdd·e5_swe; 3.6 GB 라 한 번 읽어 _cache 에 둔다),
    data/processed/v5/modis_cell_year_v5.csv × labels_cell_year_v5.csv(알래스카 라벨 셀 2014), products_cell_year_v5.csv(cci5y 2014, literal 판),
    data/raw/wei2026_alt_v2/extracted/ALT_2014.tif(EPSG:3995, cm, nodata −9999), data/raw/liu2024_alt/ALT_2014.tif(int16 cm × 100, ≤ 0 결측),
    data/raw/cci_alt_v5/...PP-2014-fv05.0.nc(m), data/raw/ornl_ds1760/Alaska_active_layer_thickness_1km_2001-2015.nc4(m, 2014),
    data/raw/aalto2018_alt/ALT_Rasters.zip/ALT_Baseline.tif(cm, 2000–2014).
- 산출: deck/assets/v8/G4_inputs.png(+pdf), G4_inputs_spec.json(배치·색 범위·출처·QA 기록), _cache/*.npz; --literal 은 G4_inputs_literal.*.
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g4_inputs.py [--literal]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU 사용 금지(CPU 전용)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
import cmcrameri.cm as cmc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402
import polar.style_v5 as S  # noqa: E402
import maps_alt_v3 as MV  # noqa: E402  (CM_ALT, ne_layers)

OUT = ROOT / "deck" / "assets" / "v8"
CACHE = OUT / "_cache"
M.OUT = OUT
PROC = ROOT / "data" / "processed"
V5 = PROC / "v5"
RAW = ROOT / "data" / "raw"
YEAR = 2014

# ================================================================ 배치(인치)
W, H = float(12.0), float(5.10)
EXT = (-167.5, -141.0, 59.8, 71.5)                     # lon0, lon1, lat0, lat1
LAT_REF = 65.0
K_ASPECT = 1.0 / np.cos(np.radians(LAT_REF))            # 위도 1° 가 경도 1° 보다 K 배 길다
DX, DY = 0.06, 0.025                                    # 표본 격자(°)
N_SLOT = 5
X_HEAD = 0.10                                           # 띠 머리 열
X0 = 2.50                                               # 첫 썸네일 왼쪽(머리 글 열 0.10–1.95 in, 그 오른쪽 0.45 in 은 기준 썸네일의 위도 눈금·축척 막대 열)
X1 = 11.96
PITCH = (X1 - X0) / N_SLOT
CB_GAP, CB_W, CB_INSET = 0.05, 0.07, 0.10                # 색 막대: 지도 오른쪽, 위아래 0.10 in 안쪽(끝 눈금 글자가 라벨·출처 줄과 겹치지 않게)
FS_HEAD, FS_LAB, FS_SMALL = M.FS["head"], 13.0, 12.0
LAND_FILL = "#E8E8E8"                                   # 자료 밖 육지(토큰 슬라이드 육지 #DEDEDE 보다 한 단계 옅게: 회색 고도 색표와 구분)
COAST = M.COAST                                         # #8A8A8A
GRAT = M.INK                                            # 기준 썸네일 격자선(점선, 투명도)
SCALE_KM, R_EARTH_KM = 500.0, 6371.0
GRAT_LON, GRAT_LAT = [], [65.0, 70.0]                   # 기준 썸네일 눈금: 위도만(12 pt '150°W' 는 0.54 in 라 0.96 in 썸네일 아래 출처 줄과 겹친다)
# 판별 배치: 띠 수, 라벨 중심(띠 위에서), 지도 위(띠 위에서), 지도 높이, 경도 눈금 줄(지도 아래에서), 출처 줄(지도 아래에서), 머리 보조줄 시작·간격
LAYOUT = {
    "main": dict(stem="G4_inputs", H=4.65, n_band=3, y_lab=0.22, y_map=0.36, hm=1.12, y_src=None, sub0=0.27, dsub=0.195, dot_s=6.0, unit_at="ticks",
                 x0=2.10, src_lines=False, ref_graticule=False, band_lines=False),
    "literal": dict(stem="G4_inputs_literal", H=5.10, n_band=4, y_lab=0.17, y_map=0.27, hm=0.66, y_src=0.265, sub0=0.24, dsub=0.20, dot_s=4.0, unit_at="after_ticks",
                    x0=2.50, src_lines=True, ref_graticule=True, band_lines=True),
}
SPEC: dict = {}


def lut(cm, lo, hi, n=250):
    c = ListedColormap(cm(np.linspace(lo, hi, n)))
    c.set_bad((0.0, 0.0, 0.0, 0.0))
    return c


def pct_range(v, lo, hi, step):
    """백분위 범위를 step 단위로 넓히고, 가운데 눈금이 step 의 배수가 되도록 구간 수를 짝수로 맞춘다."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    a = float(step * np.floor(np.percentile(v, lo) / step)); b = float(step * np.ceil(np.percentile(v, hi) / step))
    if int(round((b - a) / step)) % 2 == 1:
        b += step
    return [a, b]


def extend_of(v, lo, hi):
    """색 범위 밖 값이 있는 쪽: 색 막대 삼각형(extend)으로 포화를 표시한다."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    below = bool(v.min() < lo - 1e-9); above = bool(v.max() > hi + 1e-9)
    return "both" if below and above else "min" if below else "max" if above else "neither"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ================================================================ 표본 격자
NX = int(round((EXT[1] - EXT[0]) / DX)); NY = int(round((EXT[3] - EXT[2]) / DY))
LON_E = EXT[0] + DX * np.arange(NX + 1); LAT_E = EXT[2] + DY * np.arange(NY + 1)


def bin_mean(lon, lat, val):
    """점·화소 값을 표본 격자 상자 평균으로(행 0 = 남쪽). NaN 값은 뺀다."""
    lon = np.asarray(lon, float).ravel(); lat = np.asarray(lat, float).ravel(); val = np.asarray(val, float).ravel()
    ix = np.floor((lon - EXT[0]) / DX).astype(np.int64); iy = np.floor((lat - EXT[2]) / DY).astype(np.int64)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & np.isfinite(val)
    k = iy[ok] * NX + ix[ok]
    s = np.bincount(k, weights=val[ok], minlength=NX * NY); n = np.bincount(k, minlength=NX * NY)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(NY, NX).astype(np.float32)


def raster_centers(transform, win_row0, win_col0, h, w):
    """rasterio 창(행·열 시작, 크기)의 화소 중심 경위도(EPSG:4326 래스터)."""
    cols = win_col0 + np.arange(w) + 0.5; rows = win_row0 + np.arange(h) + 0.5
    lon = transform.c + cols * transform.a; lat = transform.f + rows * transform.e
    return np.meshgrid(lon, lat)


# ================================================================ 자료
def grid_static():
    """알래스카 1 km 격자 정적 공변량(gray == 0 셀) → 표본 격자."""
    f = PROC / "map_alaska" / "alaska_grid_x25_v1.csv.gz"
    cols = ["cell_id", "ky", "kx", "lat", "lon", "gray", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "pfr_mean"]
    g = pd.read_csv(f, usecols=cols, dtype={"cell_id": str})
    g = g[g.gray == 0]
    SPEC["sources"]["grid_static"] = dict(file=rel(f), n_cells=int(len(g)), rule="gray == 0",
                                         pfr_file="data/processed/cci_pfr_mean_1997_2021.nc (0.1°, ESA CCI PFR v4 1997–2021 평균, 최근접)")
    return g


def grid_forcing_2014(g):
    """2014 강제력: TDD 는 maps/Alaska/pred_2014(지시), FDD·SWE 는 grid/alaska_forcing_year.csv 의 2014 행(한 번 읽어 캐시)."""
    fp = V5 / "maps" / "Alaska" / f"pred_{YEAR}.csv.gz"
    p = pd.read_csv(fp, usecols=["cell_id", "e5_tdd"], dtype={"cell_id": str}).set_index("cell_id")
    fc = CACHE / f"G4_alaska_forcing_{YEAR}.csv.gz"
    big = V5 / "grid" / "alaska_forcing_year.csv"
    if not fc.exists():
        t = time.time(); out = []
        for ch in pd.read_csv(big, usecols=["cell_id", "year", "e5_tdd", "e5_fdd", "e5_swe"], chunksize=2_000_000,
                              dtype={"cell_id": str, "year": "int16", "e5_tdd": "float32", "e5_fdd": "float32", "e5_swe": "float32"}):
            out.append(ch[ch.year == YEAR])
        CACHE.mkdir(parents=True, exist_ok=True); pd.concat(out).to_csv(fc, index=False)
        print(f"  forcing {YEAR} cache: {time.time() - t:.0f} s")
    fy = pd.read_csv(fc, dtype={"cell_id": str}).set_index("cell_id")
    j = g.set_index("cell_id").join(p, how="inner").join(fy[["e5_tdd", "e5_fdd", "e5_swe"]].rename(columns={"e5_tdd": "tdd_chk"}), how="inner")
    SPEC["sources"]["forcing_2014"] = dict(tdd=rel(fp), fdd_swe=rel(big), cache=rel(fc), n_cells=int(len(j)),
                                          tdd_max_abs_diff_pred_vs_forcing=round(float((j.e5_tdd - j.tdd_chk).abs().max()), 3))
    return j.reset_index()


def label_cells():
    fl = V5 / "labels_cell_year_v5.csv"
    L = pd.read_csv(fl, usecols=["loc_id", "region", "lat", "lon", "in_forcing"])
    L = L[(L.in_forcing == 1) & (L.region == "Alaska")].drop_duplicates("loc_id")[["loc_id", "lat", "lon"]]
    return fl, L


def modis_points():
    """알래스카 라벨 셀의 2014 MODIS 값(여름 NDVI, 적설 종료일)."""
    fl, L = label_cells(); fm = V5 / "modis_cell_year_v5.csv"
    m = pd.read_csv(fm, usecols=["loc_id", "year", "ndvi_summer", "snow_off_doy"])
    m = m[m.year == YEAR]
    d = L.merge(m, on="loc_id", how="left")
    d.loc[d.ndvi_summer < -1, "ndvi_summer"] = np.nan          # 채움값 −3000
    SPEC["sources"]["modis"] = dict(labels=rel(fl), modis=rel(fm), n_label_cells=int(len(d)),
                                   n_ndvi=int(d.ndvi_summer.notna().sum()), n_snow_off=int(d.snow_off_doy.notna().sum()),
                                   products=dict(ndvi="MOD13Q1 250 m (6–8월)", snow="MOD10A2 500 m"))
    return d


def cci_points():
    """literal 판: 라벨 셀의 2014 ESA CCI v5 ALT(products_cell_year_v5.csv, cm)."""
    fl, L = label_cells(); fp = V5 / "products_cell_year_v5.csv"
    p = pd.read_csv(fp, usecols=["loc_id", "product", "year", "value_cm"])
    p = p[(p["product"] == "cci5y") & (p.year == YEAR)][["loc_id", "value_cm"]].rename(columns={"value_cm": "cci5y"})
    d = L.merge(p, on="loc_id", how="inner")
    SPEC["sources"]["cci_points"] = dict(labels=rel(fl), products=rel(fp), product="cci5y", year=YEAR, n=int(d.cci5y.notna().sum()))
    return d


def products_2014():
    """비교 제품 5종의 알래스카 창 → 표본 격자(cm). 캐시 npz."""
    fc = CACHE / f"G4_products_{YEAR}.npz"
    if fc.exists():
        z = np.load(fc, allow_pickle=True)
        SPEC["sources"]["products"] = json.loads(str(z["meta"])); SPEC["sources"]["products"]["cache"] = rel(fc)
        return {k: z[k] for k in ("wei", "liu", "cci", "yk", "aalto")}
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.windows import from_bounds
    from affine import Affine
    import xarray as xr
    meta = {}; out = {}
    lon0, lon1, lat0, lat1 = EXT
    # Wei 2026 (EPSG:3995, 1 km, cm): 창을 읽어 표본 격자로 평균 재투영
    fw = RAW / "wei2026_alt_v2" / "extracted" / f"ALT_{YEAR}.tif"
    with rasterio.open(fw) as src:
        b = transform_bounds("EPSG:4326", src.crs, lon0, lat0, lon1, lat1, densify_pts=41)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32); a[a == src.nodata] = np.nan
        dst = np.full((NY, NX), np.nan, np.float32)
        dst_tr = Affine(DX, 0.0, lon0, 0.0, -DY, lat1)
        reproject(a, dst, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan, dst_transform=dst_tr,
                  dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.average)
        out["wei"] = dst[::-1].copy()                       # 행 0 = 남쪽
        meta["wei"] = dict(file=rel(fw), crs=str(src.crs), units="cm", nodata=src.nodata, n_src_px_finite=int(np.isfinite(a).sum()))
    # Liu 2024 (EPSG:4326, 0.00833°, int16 cm × 100; ≤ 0 과 32767 은 결측)
    fl = RAW / "liu2024_alt" / f"ALT_{YEAR}.tif"
    with rasterio.open(fl) as src:
        win = from_bounds(lon0, lat0, lon1, lat1, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32)
        n_fill = int((a >= 32767).sum()); a[(a <= 0) | (a >= 32767)] = np.nan; a *= 0.01
        LON, LAT = raster_centers(src.transform, win.row_off, win.col_off, a.shape[0], a.shape[1])
        out["liu"] = bin_mean(LON, LAT, a)
        meta["liu"] = dict(file=rel(fl), crs=str(src.crs), units="cm (int16 × 0.01)", nodata="<= 0, 32767", n_32767=n_fill,
                           n_src_px_finite=int(np.isfinite(a).sum()))
    # ESA CCI v5 2014 (위경도 0.01°, m)
    fcci = RAW / "cci_alt_v5" / f"ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-{YEAR}-fv05.0.nc"
    d = xr.open_dataset(fcci)
    sub = d.ALT.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))[0]
    LON, LAT = np.meshgrid(sub.lon.values, sub.lat.values)
    out["cci"] = bin_mean(LON, LAT, sub.values * 100.0)
    meta["cci"] = dict(file=rel(fcci), units="m → cm", n_src_px_finite=int(np.isfinite(sub.values).sum()))
    d.close()
    # Yi–Kimball 2019 (ORNL ds1760, Albers, 2001–2015, m)
    fyk = RAW / "ornl_ds1760" / "Alaska_active_layer_thickness_1km_2001-2015.nc4"
    d = xr.open_dataset(fyk)
    ti = int(np.where(d.time.dt.year.values == YEAR)[0][0])
    v = d.ALT[ti].values.astype(np.float32); v[v <= -900] = np.nan
    out["yk"] = bin_mean(d.lon.values, d.lat.values, v * 100.0)
    meta["yk"] = dict(file=rel(fyk), time_index=ti, year=int(d.time.dt.year.values[ti]), units="m → cm", n_src_px_finite=int(np.isfinite(v).sum()))
    d.close()
    # Aalto 2018 기준기(2000–2014, EPSG:4326 0.00833°, cm) — zip 안의 GeoTIFF 를 창으로 읽는다
    fa = RAW / "aalto2018_alt" / "ALT_Rasters.zip"
    with rasterio.open(f"/vsizip/{fa}/ALT_Baseline.tif") as src:
        win = from_bounds(lon0, lat0, lon1, lat1, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype(np.float32); n_zero = int((a == 0).sum())
        a[(a < -1e30) | (a == src.nodata) | (a <= 0)] = np.nan            # 0 은 비영구동토 가림값으로 본다(ALT 0 cm 는 물리적으로 없다)
        LON, LAT = raster_centers(src.transform, win.row_off, win.col_off, a.shape[0], a.shape[1])
        out["aalto"] = bin_mean(LON, LAT, a)
        meta["aalto"] = dict(file=rel(fa) + "/ALT_Baseline.tif", crs=str(src.crs), units="cm", period="2000–2014", nodata="<= 0", n_zero_px=n_zero,
                             n_src_px_finite=int(np.isfinite(a).sum()))
    meta["sample_grid"] = dict(dx_deg=DX, dy_deg=DY, nx=NX, ny=NY, extent=list(EXT))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fc, meta=json.dumps(meta, ensure_ascii=False), **out)
    SPEC["sources"]["products"] = meta; SPEC["sources"]["products"]["cache"] = rel(fc)
    return out


# ================================================================ 지도 바탕
_NE = {}


def ne_paths():
    if not _NE:
        lay = MV.ne_layers("alaska")
        verts, codes = [], []
        for g in lay["land"]:
            polys = list(g.geoms) if hasattr(g, "geoms") else [g]
            for p in polys:
                for ring in [p.exterior] + list(p.interiors):
                    xy = np.asarray(ring.coords)[:, :2]
                    if len(xy) < 3:
                        continue
                    verts.append(xy); codes.append(np.r_[MPath.MOVETO, np.full(len(xy) - 2, MPath.LINETO), MPath.CLOSEPOLY])
        _NE["land"] = MPath(np.vstack(verts), np.concatenate(codes))
        segs = []
        for g in lay["coastline"]:
            lines = list(g.geoms) if hasattr(g, "geoms") else [g]
            segs += [np.asarray(l.coords)[:, :2] for l in lines if len(l.coords) >= 2]
        _NE["coast"] = segs
    return _NE


def thumb_axes(fig, x, y, wm, hm):
    """썸네일 지도 축(경위도 좌표, 세로 K 배). 바탕 육지·해안선."""
    ax = M.sub_axes(fig, x, y, wm, hm)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect(K_ASPECT, adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ne = ne_paths()
    ax.add_patch(PathPatch(ne["land"], facecolor=LAND_FILL, edgecolor="none", zorder=0))
    ax.add_collection(LineCollection(ne["coast"], colors=COAST, linewidths=0.45, zorder=3))
    for s in ax.spines.values():
        s.set_edgecolor(M.HAIR); s.set_linewidth(0.7)
    return ax


def draw_grid(ax, arr, cmap, norm):
    ax.imshow(np.ma.masked_invalid(arr), extent=(LON_E[0], LON_E[-1], LAT_E[0], LAT_E[-1]), origin="lower", cmap=cmap, norm=norm,
              interpolation="nearest", zorder=1)
    ax.set_aspect(K_ASPECT, adjustable="box")               # imshow 가 aspect 를 바꾸므로 되돌린다


def draw_dots(ax, d, col, cmap, norm, s):
    """라벨 셀 점: 무작위 순서, 얇은 어두운 테두리(옅은 바탕에서 보이게)."""
    o = d[d[col].notna()].sample(frac=1.0, random_state=0)
    ax.scatter(o.lon, o.lat, c=o[col], s=s, cmap=cmap, norm=norm, linewidths=0.2, edgecolors="#333333", zorder=2, rasterized=True)


def draw_cells(ax, d, col, cmap, norm, hm):
    """literal 판: 매 6번째 셀(ky·kx 모두 6 의 배수)을 셀 간격 크기의 사각 점으로 그린다."""
    side_pt = 6 * 0.009 * hm / (EXT[3] - EXT[2]) * 72.0 * 1.15
    o = d[d[col].notna()]
    ax.scatter(o.lon, o.lat, c=o[col], s=side_pt ** 2, marker="s", cmap=cmap, norm=norm, linewidths=0, zorder=1, rasterized=True)


def colorbar(fig, ax_c, x, y, hm, cmap, norm, ticks, unit, fmt="{:g}", extend="neither", y_unit=None, unit_at="ticks", x_slot=None, va="center"):
    cax = M.sub_axes(fig, x, y + CB_INSET, CB_W, hm - 2 * CB_INSET)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", extend=extend, extendfrac=0.07)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(M.HAIR)
    cb.set_ticks(list(ticks)); cb.set_ticklabels([fmt.format(t).replace("-", "−") for t in ticks])
    cax.tick_params(labelsize=FS_SMALL, length=3, width=0.8, pad=2, colors=M.INK)
    if unit and unit_at == "ticks":                         # 단위: 색 막대 눈금 글자 열에 맞춰(눈금 길이 3 pt + 간격 2 pt) 라벨 줄 높이에
        M.T(ax_c, x + CB_W + 5.0 / 72.0, y_unit, unit, size=FS_SMALL, color=M.AUX, ha="left", va=va)
    elif unit:                                              # 좁은 지도(literal 판): 눈금 글자 열(최대 폭 '4100' ≈ 0.39 in) 오른쪽, 13 pt 한글 라벨과 겹치지 않게
        M.T(ax_c, x_slot + 1.28, y_unit, unit, size=FS_SMALL, color=M.AUX, ha="left", va=va)
    return cb


def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    import re
    from matplotlib.text import Text
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")


def ticks3(lo, hi):
    return [lo, (lo + hi) / 2, hi]


def reference_panel(fig, ax, axm, x, y_map, wm, hm, L, graticule=True):
    """기준 썸네일: (graticule) 경위도 눈금(라벨은 틀 밖)·점선 격자선. 공용 축척 막대(500 km, 65°N 기준): graticule 판은 썸네일 왼쪽,
    아니면(v9 기본판) 머리 열 왼쪽 끝에서 시작해 막대 오른쪽에 글자."""
    if not graticule:
        deg = SCALE_KM / (2 * np.pi * R_EARTH_KM / 360.0 * np.cos(np.radians(LAT_REF)))
        Lin = deg / (EXT[1] - EXT[0]) * wm
        xb0 = X_HEAD + 0.02; xb1 = xb0 + Lin; yb = y_map + 0.06; tk = 0.035
        ax.plot([xb0, xb1], [yb, yb], color=M.INK, lw=1.0, solid_capstyle="butt", zorder=5)
        for xx in (xb0, xb1):
            ax.plot([xx, xx], [yb, yb + tk], color=M.INK, lw=0.8, solid_capstyle="butt", zorder=5)
        M.T(ax, xb1 + 0.08, yb + 0.01, f"{SCALE_KM:.0f} km", size=FS_SMALL, color=M.INK, ha="left")
        L["reference"] = dict(band=1, slot=1, graticule="none (v9)", scale_km=SCALE_KM, scale_lat_ref=LAT_REF, scale_bar_in=round(float(Lin), 3),
                              scale_bar_x=[round(xb0, 3), round(xb1, 3)], place="head column, at band-1 map bottom")
        return
    for lo in GRAT_LON:
        axm.plot([lo, lo], [EXT[2], EXT[3]], color=GRAT, lw=0.5, ls=(0, (2.0, 2.0)), alpha=0.45, zorder=2.5)
    for la in GRAT_LAT:
        axm.plot([EXT[0], EXT[1]], [la, la], color=GRAT, lw=0.5, ls=(0, (2.0, 2.0)), alpha=0.45, zorder=2.5)
    axm.set_xticks(GRAT_LON); axm.set_xticklabels([f"{abs(v):.0f}°W" for v in GRAT_LON])
    axm.set_yticks(GRAT_LAT); axm.set_yticklabels([f"{v:.0f}°N" for v in GRAT_LAT])
    axm.tick_params(axis="both", direction="out", length=2.5, width=0.6, pad=2.0, color=M.HAIR, labelcolor=M.INK, labelsize=FS_SMALL)
    # 축척 막대: 65°N 에서 500 km 에 해당하는 경도 폭을 지도 인치로 바꾼다(모든 썸네일의 축척이 같다)
    deg = SCALE_KM / (2 * np.pi * R_EARTH_KM / 360.0 * np.cos(np.radians(LAT_REF)))
    Lin = deg / (EXT[1] - EXT[0]) * wm
    xb1 = X0 - 0.06; xb0 = xb1 - Lin; yb = y_map; tk = 0.035
    ax.plot([xb0, xb1], [yb, yb], color=M.INK, lw=1.0, solid_capstyle="butt", zorder=5)
    for xx in (xb0, xb1):
        ax.plot([xx, xx], [yb, yb + tk], color=M.INK, lw=0.8, solid_capstyle="butt", zorder=5)
    M.T(ax, xb0 - 0.06, yb + 0.01, f"{SCALE_KM:.0f} km", size=FS_SMALL, color=M.INK, ha="right")
    L["reference"] = dict(band=1, slot=1, graticule_lon=GRAT_LON, graticule_lat=GRAT_LAT, scale_km=SCALE_KM, scale_lat_ref=LAT_REF,
                          scale_bar_in=round(float(Lin), 3), scale_bar_x=[round(xb0, 3), round(xb1, 3)])


# ================================================================ 그림
def g4(variant="main"):
    global H, X0, PITCH
    P = LAYOUT[variant]; stem = P["stem"]
    H = P["H"]; X0 = P["x0"]; PITCH = (X1 - X0) / N_SLOT
    SPEC.clear(); SPEC.update({"stem": stem, "variant": variant, "size_in": [W, H], "year": YEAR, "sources": {}, "layout": {}, "panels": [], "choices": {}})
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    t0 = time.time()
    g = grid_static(); gf = grid_forcing_2014(g); md = modis_points(); pr = products_2014()
    literal = variant == "literal"
    if literal:
        cp = cci_points()
        g6 = g[(g.ky % 6 == 0) & (g.kx % 6 == 0)]; gf6 = gf[(gf.ky % 6 == 0) & (gf.kx % 6 == 0)]
        SPEC["choices"]["subsample"] = dict(rule="ky % 6 == 0 and kx % 6 == 0", n_static=int(len(g6)), n_forcing=int(len(gf6)))
    print(f"  data: {time.time() - t0:.0f} s")
    n_band, hm = P["n_band"], P["hm"]
    band_h = H / n_band
    wm = hm * ((EXT[1] - EXT[0]) * np.cos(np.radians(LAT_REF))) / (EXT[3] - EXT[2])
    n_nd, n_so = SPEC["sources"]["modis"]["n_ndvi"], SPEC["sources"]["modis"]["n_snow_off"]

    # ---- 썸네일 정의: (라벨, 출처 줄, 배열 또는 (열, 점 표) 또는 (열, 셀 표, 'cells'), 색표, 범위, 단위, 서식)
    def A(col, src=g):
        return bin_mean(src.lon.values, src.lat.values, src[col].values)

    def G(col, src):                                        # literal 판: 6번째 셀 추출
        return (col, src, "cells")

    forcing = [
        ("TDD", "ERA5-Land · 9 km", G("e5_tdd", gf6) if literal else A("e5_tdd", gf), lut(cmc.lajolla_r, 0.05, 0.80), (2, 98, 100), "°C·d", "{:.0f}"),
        ("FDD", "ERA5-Land · 9 km", G("e5_fdd", gf6) if literal else A("e5_fdd", gf), lut(cmc.devon_r, 0.15, 0.95), (2, 98, 100), "°C·d", "{:.0f}"),
        ("SWE", "ERA5-Land · 9 km", G("e5_swe_mm", gf6.assign(e5_swe_mm=gf6.e5_swe * 1000.0)) if literal else A("e5_swe", gf) * 1000.0,
         lut(cmc.lapaz_r, 0.05, 0.90), (2, 95, 100), "mm", "{:.0f}"),
    ]
    satellite = [
        ("Summer NDVI", f"MOD13Q1 · {n_nd:,} cells", ("ndvi_summer", md), lut(cmc.navia_r, 0.10, 0.90), (2, 98, 0.1), "", "{:.1f}"),
        ("Snow-off date", f"MOD10A2 · {n_so:,} cells", ("snow_off_doy", md), lut(cmc.davos_r, 0.10, 0.90), (2, 98, 10), "DOY", "{:.0f}"),
    ]
    static = [
        ("Elevation", "Copernicus DEM · 30 m", G("dem_elev", g6) if literal else A("dem_elev"), lut(cmc.grayC_r, 0.20, 0.90), (2, 98, 100), "m", "{:.0f}"),
        ("Slope", "Copernicus DEM · 30 m", G("dem_slope", g6) if literal else A("dem_slope"), lut(cmc.bilbao_r, 0.15, 0.95), (2, 98, 5), "°", "{:.0f}"),
        ("Soil carbon", "SoilGrids · 5 km window", G("sg_soc_0_5", g6) if literal else A("sg_soc_0_5"), lut(cmc.bamako_r, 0.10, 0.90), (2, 98, 50), "g/kg", "{:.0f}"),
        ("Clay content", "SoilGrids · 5 km window", G("sg_clay_5_15", g6) if literal else A("sg_clay_5_15"), lut(cmc.acton_r, 0.10, 0.90), (2, 98, 5), "%", "{:.0f}"),
        ("Map domain", "ESA CCI PFR · 0.1°", G("pfr_mean", g6) if literal else A("pfr_mean"), lut(cmc.nuuk_r, 0.10, 0.90), ("fix", 50, 100), "%", "{:.0f}"),   # 지도 범위 기준(≥ 50 %), 모델 입력 아님
    ]
    # 제품 띠: 같은 ALT 색 범위(그리는 제품을 합친 2–95 백분위, 10 cm 단위), 색 막대 하나
    prod_keys = ("wei", "liu", "cci") if literal else ("wei", "liu", "cci", "yk", "aalto")
    pooled = np.concatenate([pr[k].ravel() for k in prod_keys] + ([cp.cci5y.values] if literal else []))
    alt_lo, alt_hi = pct_range(pooled, 2, 95, 10); alt_ext = extend_of(pooled, alt_lo, alt_hi)
    if literal:
        products = [("Wei 2026", f"1 km · {YEAR}", pr["wei"], "Wei"), ("Liu 2024", f"1 km · {YEAR}", pr["liu"], "Liu"),
                    ("ESA CCI v5", f"label cells {SPEC['sources']['cci_points']['n']:,} · {YEAR}", ("cci5y", cp), "CCI5")]
    else:
        products = [("Wei 2026", f"1 km · {YEAR}", pr["wei"], "Wei"), ("Liu 2024", f"1 km · {YEAR}", pr["liu"], "Liu"),
                    ("ESA CCI v5", f"1 km · {YEAR}", pr["cci"], "CCI5"), ("Yi–Kimball 2019", f"1 km · {YEAR}", pr["yk"], "YK"),
                    ("Aalto 2018", "1 km · 2000–2014", pr["aalto"], "Aalto")]
    SPEC["choices"]["alt_range_cm"] = dict(lo=alt_lo, hi=alt_hi, rule="pooled drawn products 2–95 pct, step 10; colorbar extend shows clipping", extend=alt_ext)
    SPEC["choices"]["product_hex"] = {k: S.PRODUCT_HEX[k] for k in ("Wei", "Liu", "CCI5", "YK", "Aalto")}
    for k in prod_keys:
        v = pr[k][np.isfinite(pr[k])]
        SPEC["sources"]["products"][k]["sample_grid_stats_cm"] = dict(n=int(v.size), p2=round(float(np.percentile(v, 2)), 1),
                                                                      p50=round(float(np.percentile(v, 50)), 1), p98=round(float(np.percentile(v, 98)), 1))

    if literal:
        bands = [
            ("Yearly climate", ["ERA5-Land 9 km", f"{YEAR}"], forcing),
            ("Static covariates", ["Copernicus DEM 30 m", "SoilGrids 250 m", "ESA CCI PFR 0.1°"], static),
            ("Satellite inputs", ["MODIS 250–500 m", f"{YEAR}", "Dots = label cells"], satellite),
            ("Products", ["1 km · 2014", "CCI: label-cell dots"], products),
        ]
    else:
        bands = [
            ("Yearly inputs", ["ERA5-Land · 9 km", "MODIS · 250–500 m", "Dots = label cells", f"{YEAR}"], forcing + satellite),
            ("Static covariates", ["Copernicus DEM · 30 m", "SoilGrids · 250 m", "Map domain: ESA CCI", "permafrost probability", "≥ 50 % (not an input)"], static),
            ("Products", [f"1 km · {YEAR}", "Aalto 2018:", "2000–2014 mean"], products),
        ]
    SPEC["layout"] = dict(bands=n_band, slots=N_SLOT, x0=X0, pitch=round(PITCH, 3), map_w=round(float(wm), 3), map_h=hm, band_h=round(band_h, 3),
                          extent=list(EXT), aspect=round(float(K_ASPECT), 3), head_x=X_HEAD, head_color=M.HEAD, **{k: v for k, v in P.items() if k not in ("stem", "x0")})

    for bi, (hd, subs, items) in enumerate(bands):
        top = H - bi * band_h
        y_lab = top - P["y_lab"]; y_map = top - P["y_map"] - hm; y_src = (y_map - P["y_src"]) if P["y_src"] is not None else None
        y_base = y_lab - 0.03                                       # 띠 머리·썸네일 제목·단위의 공통 기준선(va='baseline'; 지도 위 끝에서 0.11 in 위)
        y_unit = y_base                                             # 단위 글자는 같은 기준선, 색 막대 위
        M.T(ax, X_HEAD, y_base, hd, size=FS_HEAD, color=M.HEAD, weight="bold", va="baseline")   # 띠 머리: 열 머리 잉크색(M.HEAD), 굵게
        for k, s_ in enumerate(subs):
            M.T(ax, X_HEAD, y_lab - P["sub0"] - P["dsub"] * k, s_, size=FS_SMALL, color=M.AUX)
        if bi > 0 and P["band_lines"]:                              # v9 기본판: 띠 구분선 없음(머리 글자와 선이 붙어 보인다는 지적; 띠는 주황 머리로 나뉜다)
            M.hline(ax, X_HEAD, W - 0.06, top, color=M.SEP, lw=0.8)
        is_prod = items is products
        for si, it in enumerate(items):
            x = X0 + si * PITCH
            axm = thumb_axes(fig, x, y_map, wm, hm)
            if not is_prod:
                lab, src, data, cmap, rng, unit, fmt = it
                if isinstance(data, tuple):
                    vals = data[1][data[0]].values
                else:
                    vals = data
                lo, hi = (float(rng[1]), float(rng[2])) if rng[0] == "fix" else pct_range(vals, *rng)
                norm = Normalize(lo, hi, clip=False); ext = extend_of(vals, lo, hi)
                if isinstance(data, tuple) and len(data) == 3:
                    draw_cells(axm, data[1], data[0], cmap, norm, hm)
                elif isinstance(data, tuple):
                    draw_dots(axm, data[1], data[0], cmap, norm, P["dot_s"])
                else:
                    draw_grid(axm, data, cmap, norm)
                colorbar(fig, ax, x + wm + CB_GAP, y_map, hm, cmap, norm, ticks3(lo, hi), unit, fmt, extend=ext, y_unit=y_unit, unit_at=P["unit_at"], x_slot=x,
                         va="baseline")
                M.T(ax, x, y_base, lab, size=FS_LAB, va="baseline")
                SPEC["panels"].append(dict(band=bi + 1, slot=si + 1, label=lab, source=src, range=[lo, hi], ticks=ticks3(lo, hi), unit=unit, extend=ext,
                                           kind="dots" if (isinstance(data, tuple) and len(data) == 2) else "cells" if isinstance(data, tuple) else "box_mean"))
            else:
                lab, src, data, pk = it
                norm = Normalize(alt_lo, alt_hi, clip=False)
                if isinstance(data, tuple):
                    draw_dots(axm, data[1], data[0], MV.CM_ALT, norm, P["dot_s"])
                else:
                    draw_grid(axm, data, MV.CM_ALT, norm)
                M.T(ax, x, y_base, lab, size=FS_LAB, color=S.PRODUCT_HEX[pk], va="baseline")
                if si == len(items) - 1:
                    colorbar(fig, ax, x + wm + CB_GAP, y_map, hm, MV.CM_ALT, norm, ticks3(alt_lo, alt_hi), "cm", "{:.0f}", extend=alt_ext, y_unit=y_unit,
                             unit_at=P["unit_at"], x_slot=x, va="baseline")
                SPEC["panels"].append(dict(band=bi + 1, slot=si + 1, label=lab, source=src, range=[alt_lo, alt_hi], unit="cm", extend=alt_ext,
                                           color=S.PRODUCT_HEX[pk], kind="dots" if isinstance(data, tuple) else "box_mean"))
            if P["src_lines"]:
                M.T(ax, x, y_src, src, size=FS_SMALL, color=M.AUX)
            if bi == 0 and si == 0:
                reference_panel(fig, ax, axm, x, y_map, wm, hm, SPEC["layout"], graticule=P["ref_graticule"])

    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    head_texts = [t for t in ax.texts if t.get_position()[0] <= X_HEAD + 1e-6]
    SPEC["qa_head_col"] = dict(max_right_in=round(max(t.get_window_extent(rr).x1 / fig.dpi for t in head_texts), 3), first_thumb_x_in=X0,
                               fits=bool(max(t.get_window_extent(rr).x1 / fig.dpi for t in head_texts) <= X0 - 0.08))
    heads_top = [t.get_window_extent(rr).y1 / fig.dpi for t in ax.texts if t.get_text() in [b_[0] for b_ in bands]]
    band_tops = [H - bi * band_h for bi in range(n_band)]
    SPEC["qa_head_col"]["head_to_band_top_gap_in"] = [round(bt - ht, 3) for bt, ht in zip(band_tops[1:], sorted(heads_top, reverse=True)[1:])]
    SPEC["qa_head_col"]["band_lines"] = P["band_lines"]
    maps_ = [a_ for a_ in fig.axes[1:] if a_.get_xlim() == (EXT[0], EXT[1])]
    bottoms = sorted({round(a_.get_position().y0 * H, 3) for a_ in maps_}, reverse=True)
    SPEC["qa_head_col"]["map_bottom_to_next_head_top_in"] = [round(b_ - ht, 3) for b_, ht in zip(bottoms[:-1], sorted(heads_top, reverse=True)[1:])]
    print("머리 열 점검:", SPEC["qa_head_col"], flush=True)
    # 기준선 점검: 띠마다 머리·제목·단위 글자의 기준선(자료 좌표 y, va='baseline')이 모두 같은지
    base_chk = {}
    for bi, (hd, _, items) in enumerate(bands):
        row = [t for t in ax.texts if t.get_va() == "baseline" and abs(t.get_position()[1] - (H - bi * band_h - P["y_lab"] - 0.03)) < 0.2]
        ys = sorted({round(t.get_position()[1], 4) for t in row})
        base_chk[hd] = dict(n_texts=len(row), baselines_in=ys, same=bool(len(ys) == 1))
    SPEC["qa_head_col"]["row_baselines"] = base_chk
    print("기준선 점검:", {k: v["same"] for k, v in base_chk.items()}, flush=True)
    no_hangul(fig)
    rec = M.qa_and_save(fig, stem)
    for ext_ in ("png", "pdf"):
        p = OUT / f"{stem}_slide.{ext_}"
        if p.exists():
            p.replace(OUT / f"{stem}.{ext_}")
    from PIL import Image
    with Image.open(OUT / f"{stem}.png") as im:                 # 덱용 RGB(흰 바탕), 300 dpi 메타
        im.convert("RGB").save(OUT / f"{stem}.png", dpi=(M.DPI, M.DPI))
    SPEC["qa"] = rec
    SPEC["brief_deviation"] = ("main (v9 layout, 4.65 in): 3 bands (yearly ERA5-Land + MODIS merged), 5 products (adds Yi–Kimball, Aalto), CCI as raster, box-mean sampling, no per-thumbnail source lines; "
                               "literal: 4 bands, 3 products (CCI as label-cell dots), every-6th-cell subsample. User sign-off pending.")
    (OUT / f"{stem}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1, default=str))
    print(f"  saved {OUT / (stem + '.png')}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    g4("literal" if "--literal" in sys.argv else "main")
