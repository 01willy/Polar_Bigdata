"""XE 점 규모 공변량 표(data/processed/xe/xe_feat_v1.csv, 키 loc_id)와 메타·최종 해시(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절,
개정 1, T0 = git 2678100).

군과 원자료(계획 2.5 '입력 집합' 표, 열 목록은 scripts/3_deep_learning/x_hires_registry.py)
  h0   H0(10)  data/processed/covariates_ext_v1.csv 결합(H19 T·W 군, treecover_1km)
  t2   T2(6)   Copernicus DEM 30 m(data/raw/dem), JRC GSW occurrence 30 m(data/raw/gsw), Hansen treecover2000 30 m(data/raw/hansen)
  m    M(5)    ArcticDEM v4.1 10 m 창(data/raw/arcticdem_xe, m-fetch 가 /vsicurl 창 읽기로 받는다. 10 m 타일이 없으면 32 m 판)
  o    O(9)    SoilGrids 250 m 정수 화소 정렬 창(data/raw/soilgrids_xe250), NCSCDv2 0.012°(data/raw/ncscd_v2)
  v    V(12)   AppEEARS MOD13Q1 점 결과(data/raw/appeears_xe), ESA WorldCover 2021 v200(data/raw/worldcover_v200), CAVM(data/raw/cavm_raster)
  s    S(3)    AppEEARS MOD10A1 점 결과(주), 없으면 MOD10A2(대체, 등록 정의)
명령(하위 명령, 여러 개를 차례로 줄 수 있다)
  h0 t2 m-fetch m o v s   군별 조각 data/processed/xe/parts/xe_<군>_v1.csv 와 메타(원자료 경로, 크기, 수정 시각, 군 유한값 비율)
  assemble               조각 결합 → xe_feat_v1.csv 와 xe_feat_v1_meta.json(마감 판정, 90 % 규칙의 군 포함 여부, 군별 열 해시)
  qa                     등록 전에 허용한 확인(유한값 비율, 격자·위치 안 변동 비율, 새 열 사이 상관). ALT·잔차 통계는 계산하지 않는다
  finalize               최종 해시 단계: xe_feat_v1_final.json 과 개정 이력 문구(xe_feat_v1_revision_entry.md)를 쓴다. 커밋은 사용자가 한다
규칙
  - 라벨 표(fidelity_base_v3.csv)에서는 좌표 열(loc_id, lat, lon, region)만 읽는다. 라벨 열을 읽으려 하면 중단한다(계획 2.5 시험 (e)).
  - 30 m 이하 제품은 3 × 3 창 통계가 주 값이고 점 화소 값(*_px 열)은 민감도다. 250 m 이상 제품은 점 화소 값이다.
  - 마감(T0 + 72 h: O, M, WorldCover, CAVM / T0 + 96 h: MOD13Q1, S / 최종 T0 + 100 h)을 넘겨 확보한 원자료의 열은 결합 단계에서 결측으로 두고
    90 % 규칙으로 군을 뺀다. 결정은 메타에 남긴다.
  - 로컬 자원: 워커 4 이하, 워커당 스레드 1, nice 10, 프로세스 주소 공간 상한(--mem-gb, 기본 10 GB). 시작 전 가용 메모리 30 GB 를 확인한다.
  - 디스크(계획 1절): GDAL 임시 파일·캐시(CPL_TMPDIR 기본 data/raw/xe_tmp, GDAL_CACHEMAX 512 MB)는 /home 의 data/raw 아래에만 둔다.
  - 표준 출력에는 행 수, 군 유한값 비율, 시간, 해시만 쓴다.
실행(로컬 1c, ROOT 에서)
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/1_data_prep/xe_point_covariates.py h0 t2 assemble qa --workers 4
  python3 scripts/1_data_prep/xe_point_covariates.py m-fetch m o v s assemble qa --workers 4     # 2단계 자료가 들어온 뒤
  python3 scripts/1_data_prep/xe_point_covariates.py finalize                                    # 마감(T0 + 100 h) 전, 해시를 개정 이력에 커밋
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# GDAL 임시 파일·캐시는 /home 의 data/raw 아래에만 둔다(계획 1절 '디스크': 루트·/tmp 는 97 % 사용). 환경 변수가 이미 있으면 그대로 둔다
os.environ.setdefault("CPL_TMPDIR", str(ROOT / "data" / "raw" / "xe_tmp"))
os.environ.setdefault("GDAL_CACHEMAX", "512")                                # MB, 워커마다


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("x_hires_registry", ROOT / "scripts" / "3_deep_learning" / "x_hires_registry.py")
T = _load("xe_feature_tools", HERE / "xe_feature_tools.py")

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
FB = PROC / "fidelity_base_v3.csv"
COV_EXT = PROC / "covariates_ext_v1.csv"
XE_DIR = PROC / "xe"
FEAT_NAME = "xe_feat_v1"
D_DEM, D_GSW, D_HANSEN = RAW / "dem", RAW / "gsw", RAW / "hansen"
D_SG = RAW / "soilgrids_xe250"
D_NCSCD = RAW / "ncscd_v2" / "unzipped" / "NCSCDv2_Circumpolar_raster_0012deg"
D_WC = RAW / "worldcover_v200" / "tiles"
CAVM_TIF = RAW / "cavm_raster" / "v2" / "unzipped" / "raster_cavm_v1.tif"
D_AE = RAW / "appeears_xe"
D_AD = RAW / "arcticdem_xe"
AD_BASE = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
ALLOWED_OUT = (XE_DIR, D_AD, PROC / "xbatch")                              # 계획이 정한 쓰기 경로(XE 표, 새 원자료, xbatch)

CLUSTER_DEG = (0.1, 0.25)                                                   # 창 묶음(위도, 경도 °)
DEM_MARGIN_DEG = 0.05                                                       # 점 TWI 의 누적 창 여유(위도 °, 경도는 /cosφ)
SMALL_MARGIN_DEG = 0.002                                                    # GSW·Hansen 3 × 3 창 여유
WC_MARGIN_DEG = 0.0005
SG_LAYERS = {"sg250_soc_0_5": "soc_0-5cm_mean", "sg250_soc_5_15": "soc_5-15cm_mean", "sg250_soc_15_30": "soc_15-30cm_mean",
             "sg250_bdod_5_15": "bdod_5-15cm_mean", "sg250_clay_5_15": "clay_5-15cm_mean", "sg250_sand_5_15": "sand_5-15cm_mean",
             "sg250_silt_5_15": "silt_5-15cm_mean"}
NCSCD_LAYERS = {"ncscd_soc_0_30": "NCSCDv2_Circumpolar_WGS84_SOCC30_0012deg.tif", "ncscd_soc_0_100": "NCSCDv2_Circumpolar_WGS84_SOCC100_0012deg.tif"}
AD_WIN_BIN_M = 2000.0                                                       # ArcticDEM 창 묶음(EPSG:3413, m)
AD_WIN_MARGIN_M = 200.0
TRACE_READS: list = []                                                      # 시험 (e): 표 읽기 기록(경로, 열)


def rel(p):
    """저장소 상대 경로(저장소 밖이면 절대 경로)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def log(msg):
    print(f"[{datetime.now(R.KST).strftime('%H:%M:%S')}] {msg}", flush=True)


def now_kst():
    return datetime.now(R.KST).strftime("%Y-%m-%d %H:%M:%S %z")


# ================================================================ 경로·자원 보호
def check_out(path):
    p = Path(os.path.realpath(str(path)))
    roots = [Path(os.path.realpath(str(r))) for r in list(ALLOWED_OUT) + [v for v in os.environ.get("XE_OUT_ROOTS", "").split(os.pathsep) if v]]
    if not any(p == r or r in p.parents for r in roots):
        raise SystemExit(f"[거부] 쓰기 경로 {p} 는 허용 경로 밖이다(data/processed/xe, data/raw/arcticdem_xe, data/processed/xbatch)")
    return p


def set_local_limits(mem_gb, nice=10):
    try:
        os.nice(int(nice))
    except OSError:
        pass
    if mem_gb and float(mem_gb) > 0:
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            lim = int(float(mem_gb) * 2 ** 30)
            if soft == resource.RLIM_INFINITY or soft > lim:
                resource.setrlimit(resource.RLIMIT_AS, (lim, hard))
        except (ValueError, OSError):
            pass


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def max_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(r, c) / 1024.0, 1)


# ================================================================ 표 읽기(라벨 열 금지)
def read_csv_cols(path, cols):
    """열을 지정해 읽는다. 라벨 열(R.LABEL_COLS)은 거부한다(계획 2.5 시험 (e))."""
    cols = list(cols)
    bad = sorted(set(cols) & set(R.LABEL_COLS))
    if bad:
        raise RuntimeError(f"특징 추출 코드가 라벨 열 {bad} 을 읽으려 했다")
    TRACE_READS.append((str(path), tuple(cols)))
    return pd.read_csv(path, usecols=cols, low_memory=False)


def read_points(fb=None, targets=R.TARGETS):
    """XE 대상(알래스카, 레나, 캐나다) 행의 좌표. 좌표 열만 읽는다. 반환 loc_id 순 표(loc_id, lat, lon, region, target)."""
    d = read_csv_cols(FB if fb is None else fb, R.COORD_COLS)
    d["target"] = d.region.map(R.MACRO)
    d = d[d.target.isin(list(targets))].sort_values("loc_id").reset_index(drop=True)
    return d


def cluster_keys(lat, lon, deg=CLUSTER_DEG):
    return list(zip(np.floor(np.asarray(lat) / deg[0]).astype(int), np.floor(np.asarray(lon) / deg[1]).astype(int)))


def clusters_of(pts, limit=0):
    """창 묶음 [(키, 색인 배열)]. limit > 0 이면 앞 limit 개(시험 실행)."""
    keys = cluster_keys(pts.lat.values, pts.lon.values)
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    out = [(k, np.array(v, int)) for k, v in sorted(by.items())]
    return out[:int(limit)] if limit and limit > 0 else out


# ================================================================ 래스터 타일과 창
def dem_tile(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return D_DEM / f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"


def gsw_tile(top, left):
    return D_GSW / f"occurrence_{abs(left)}{'E' if left >= 0 else 'W'}_{abs(top)}{'N' if top >= 0 else 'S'}v1_4_2021.tif"


def hansen_tile(top, left):
    return D_HANSEN / f"Hansen_GFC-2023-v1.11_treecover2000_{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}.tif"


def wc_tile(lat0, lon0):
    name = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    return D_WC / f"ESA_WorldCover_10m_2021_v200_{name}_Map.tif"


def tiles_for(bounds, kind):
    """경계 상자가 닿는 타일 경로(있는 것만). kind = dem(1°), gsw·hansen(10°, 이름은 위 모서리), wc(3°, 이름은 아래 모서리)."""
    w, s, e, n = bounds
    out = []
    if kind == "dem":
        for la in range(int(math.floor(s)), int(math.floor(n)) + 1):
            for lo in range(int(math.floor(w)), int(math.floor(e)) + 1):
                out.append(dem_tile(la, lo))
    elif kind in ("gsw", "hansen"):
        for la in range(int(math.floor(s / 10)), int(math.floor(n / 10)) + 1):
            for lo in range(int(math.floor(w / 10)), int(math.floor(e / 10)) + 1):
                out.append((gsw_tile if kind == "gsw" else hansen_tile)(la * 10 + 10, lo * 10))
    elif kind == "wc":
        for la in range(int(math.floor(s / 3)), int(math.floor(n / 3)) + 1):
            for lo in range(int(math.floor(w / 3)), int(math.floor(e / 3)) + 1):
                out.append(wc_tile(la * 3, lo * 3))
    else:
        raise ValueError(kind)
    return [p for p in dict.fromkeys(out) if Path(p).exists()]


def read_mosaic(paths, bounds, ref=None):
    """여러 타일의 창을 기준 타일 격자에 맞춘 배열로 읽는다(rasterio.merge, 최근접). 반환 (float32 배열, 변환, 격자) 또는 (None, None, None).
    격자 = dict(c0, f0, a, ey, row_off, col_off): 기준 타일 원점과 화소 크기, 창의 정수 화소 위치. 점 화소 색인은 grid_rc 로 기준 격자에서
    정수로 구한다(창 변환의 부동소수 오차로 화소 경계의 점이 한 칸 어긋나지 않게 한다). 격자 밖과 원자료가 없는 칸은 NaN. ref = 기준 타일 경로."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    if not paths:
        return None, None, None
    srcs = [rasterio.open(p) for p in paths]
    try:
        rs = srcs[0] if ref is None else next((s_ for s_, p in zip(srcs, paths) if str(p) == str(ref)), srcs[0])
        T_ = rs.transform
        a, ey, c0, f0 = T_.a, abs(T_.e), T_.c, T_.f
        w, s, e, n = bounds
        col0, col1 = int(math.floor((w - c0) / a)), int(math.ceil((e - c0) / a))
        row0, row1 = int(math.floor((f0 - n) / ey)), int(math.ceil((f0 - s) / ey))
        left, right, top, bottom = c0 + col0 * a, c0 + col1 * a, f0 - row0 * ey, f0 - row1 * ey
        arr, _ = merge(srcs, bounds=(left, bottom, right, top), res=(a, ey), nodata=np.nan, dtype="float32", indexes=[1],
                       resampling=Resampling.nearest)
        arr = arr[0]
        if arr.shape != (row1 - row0, col1 - col0):
            raise RuntimeError(f"모자이크 크기 {arr.shape} 가 정수 창 {(row1 - row0, col1 - col0)} 과 다르다")
        tr = Affine(a, 0.0, left, 0.0, -ey, top)
        return arr, tr, dict(c0=c0, f0=f0, a=a, ey=ey, row_off=row0, col_off=col0)
    finally:
        for s_ in srcs:
            s_.close()


def grid_rc(grid, x, y):
    """기준 격자의 정수 화소 색인에서 창의 행·열(read_mosaic 의 격자)."""
    r = np.floor((grid["f0"] - np.asarray(y, float)) / grid["ey"]).astype(np.int64) - int(grid["row_off"])
    c = np.floor((np.asarray(x, float) - grid["c0"]) / grid["a"]).astype(np.int64) - int(grid["col_off"])
    return r, c


def cluster_bounds(lat, lon, margin_deg):
    la0, la1 = float(np.min(lat)), float(np.max(lat))
    lo0, lo1 = float(np.min(lon)), float(np.max(lon))
    mlo = margin_deg / max(math.cos(math.radians(0.5 * (la0 + la1))), 0.05)
    return (lo0 - mlo, la0 - margin_deg, lo1 + mlo, la1 + margin_deg)


def ref_dem_tile(lat, lon):
    p = dem_tile(int(math.floor(float(np.mean(lat)))), int(math.floor(float(np.mean(lon)))))
    return p if p.exists() else None


# ================================================================ 군 조각 계산(작업 단위 = 창 묶음)
def t2_cluster(task):
    """T2 군: 한 창 묶음의 점. DEM 창(여유 0.05°)에서 D8 누적과 TWI, 3 × 3·9 × 9 TPI, 3 × 3 경사, GSW·Hansen 3 × 3 평균과 점 화소."""
    key, loc, lat, lon = task
    t0 = time.time()
    out = {k: np.full(len(loc), np.nan) for k in ("twi_pt", "tpi_90", "tpi_270", "slope_90", "slope_px", "gsw_occ_pt", "gsw_occ_px",
                                                     "treecover_pt", "treecover_px")}
    note = {}
    b = cluster_bounds(lat, lon, DEM_MARGIN_DEG)
    paths = tiles_for(b, "dem")
    z, tr, grid = read_mosaic(paths, b, ref_dem_tile(lat, lon))
    if z is not None:
        z = z.astype(float)
        z[(z < -500) | ~np.isfinite(z)] = np.nan
        if np.isfinite(z).mean() >= 0.2:
            dy, dx = T.geo_spacing(tr, 0.5 * (b[1] + b[3]))
            try:
                acc = T.d8_accumulation(z, tr)
            except Exception as e:                                       # noqa: BLE001  누적 실패는 twi_pt 만 결측으로 둔다
                acc, note["acc_error"] = None, repr(e)[:160]
            r, c = grid_rc(grid, lon, lat)
            res = T.terrain_points(z, tr, r, c, dy, dx, acc)
            for k, v in res.items():
                out[k] = v
            note["dem_shape"] = list(z.shape)
        note["dem_tiles"] = [Path(p).name for p in paths]
    bs = cluster_bounds(lat, lon, SMALL_MARGIN_DEG)
    for kind, col in (("gsw", "gsw_occ"), ("hansen", "treecover")):
        a, _tr2, grid2 = read_mosaic(tiles_for(bs, kind), bs)
        if a is None:
            continue
        a = a.astype(float)
        if kind == "gsw":
            a[a == 255] = np.nan                                         # GSW 255 = 자료 없음(build_covariates_ext 와 같다)
        r, c = grid_rc(grid2, lon, lat)
        out[f"{col}_pt" if kind == "gsw" else "treecover_pt"] = np.array([T.win_mean(a, i, j, 1) for i, j in zip(r, c)])
        out[f"{col}_px" if kind == "gsw" else "treecover_px"] = np.array([T.px_value(a, i, j) for i, j in zip(r, c)])
    note["sec"] = round(time.time() - t0, 2)
    return key, loc, out, note


def wc_cluster(task):
    """V 군의 WorldCover 부분: 3 × 3 창(10 m 화소, 30 m) 등급 비율 7열과 점 화소 원핫(민감도)."""
    key, loc, lat, lon = task
    cols = list(R.WC_CODES) + [R.PX_MAP[k] for k in R.WC_CODES]
    out = {k: np.full(len(loc), np.nan) for k in cols}
    b = cluster_bounds(lat, lon, WC_MARGIN_DEG)
    a, _tr, grid = read_mosaic(tiles_for(b, "wc"), b)
    if a is not None:
        r, c = grid_rc(grid, lon, lat)
        for i, (ri, ci) in enumerate(zip(r, c)):
            fr = T.class_fractions(a, ri, ci, 1, R.WC_CODES, nodata=0)
            oh = T.class_onehot(a, ri, ci, R.WC_CODES, nodata=0)
            for k in R.WC_CODES:
                out[k][i] = fr[k]
                out[R.PX_MAP[k]][i] = oh[k]
    return key, loc, out, {}


def run_clusters(fn, pts, workers, limit=0, label=""):
    """창 묶음 작업을 프로세스 풀(spawn, 워커 ≤ 4)로 돌려 loc_id 순 결과 표를 만든다."""
    import multiprocessing
    tasks = [(k, pts.loc_id.values[idx], pts.lat.values[idx], pts.lon.values[idx]) for k, idx in clusters_of(pts, limit)]
    cols, rows, notes = None, {}, []
    t0 = time.time()
    done = 0

    def take(res):
        nonlocal cols, done
        k, loc, out, note = res
        cols = cols or list(out)
        for j, lid in enumerate(loc):
            rows[int(lid)] = [out[c][j] for c in cols]
        notes.append(dict(key=list(map(int, k)), n=int(len(loc)), **note))
        done += 1
        if done % 20 == 0 or done == len(tasks):
            log(f"[{label}] 창 묶음 {done}/{len(tasks)} · {time.time() - t0:.0f}s")
    w = max(0, min(int(workers), 4))
    if w <= 1:
        for t_ in tasks:
            take(fn(t_))
    else:
        with ProcessPoolExecutor(max_workers=w, mp_context=multiprocessing.get_context("spawn")) as ex:
            for f in as_completed([ex.submit(fn, t_) for t_ in tasks]):
                take(f.result())
    lids = sorted(rows)
    df = pd.DataFrame([[lid] + rows[lid] for lid in lids], columns=["loc_id"] + (cols or []))
    return df, notes


# ================================================================ 점 화소 표집(250 m 이상 제품)
def sample_points(path, x, y, nodata_extra=()):
    """한 래스터의 최근접 화소 값(격자 밖·자료 없음은 NaN). x, y 는 래스터 좌표계."""
    import rasterio
    with rasterio.open(path) as src:
        r, c = T.pixel_rc(src.transform, x, y)
        ok = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
        v = np.full(len(r), np.nan)
        if ok.any():
            r0, r1, c0, c1 = int(r[ok].min()), int(r[ok].max()) + 1, int(c[ok].min()), int(c[ok].max()) + 1
            from rasterio.windows import Window
            a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
            v[ok] = a[r[ok] - r0, c[ok] - c0]
            nd = [src.nodata] + list(nodata_extra)
            for q in nd:
                if q is not None and np.isfinite(q):
                    v[v == q] = np.nan
    return v


def to_crs(lon, lat, crs):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def soilgrids_points(pts, sg_dir=D_SG):
    """O 군의 SoilGrids 7층: 정렬 창(soilgrids_xe250)에서 층마다 이름 순 첫 창의 유효 값(> 0, 자료 없음 아님, xe_stage2_acquire 의 포함 규칙).
    단위는 SoilGrids 원값(soc dg/kg, bdod cg/cm³, 점토·모래·실트 g/kg)."""
    import rasterio
    x, y = to_crs(pts.lon.values, pts.lat.values, IGH)
    out = {k: np.full(len(pts), np.nan) for k in SG_LAYERS}
    used = {k: np.array([""] * len(pts), dtype=object) for k in SG_LAYERS}
    dirs = sorted(p for p in Path(sg_dir).iterdir() if p.is_dir()) if Path(sg_dir).exists() else []
    files = []
    for d in dirs:
        for col, lyr in SG_LAYERS.items():
            p = d / f"{lyr}.tif"
            if not p.exists():
                continue
            files.append(p)
            need = ~np.isfinite(out[col])
            if not need.any():
                continue
            with rasterio.open(p) as src:
                r, c = T.pixel_rc(src.transform, x[need], y[need])
                ok = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
                if not ok.any():
                    continue
                a = src.read(1).astype(float)
                v = np.full(len(r), np.nan)
                v[ok] = a[r[ok], c[ok]]
                good = np.isfinite(v) & (v > 0) & ((v != src.nodata) if src.nodata is not None else True)
                idx = np.where(need)[0][good]
                out[col][idx] = v[good]
                used[col][idx] = d.name
    return out, used, files


# ================================================================ AppEEARS 점 결과
def ae_results(product):
    """data/raw/appeears_xe 아래의 점 결과 CSV(시험 요청 폴더 test 는 뺀다)."""
    tag = {"MOD13Q1": "MOD13Q1-061-results.csv", "MOD10A1": "MOD10A1-061-results.csv", "MOD10A2": "MOD10A2-061-results.csv"}[product]
    fs = sorted(glob.glob(str(D_AE / "**" / f"*{tag}"), recursive=True))
    return [Path(f) for f in fs if "/test/" not in f.replace(os.sep, "/")]


def pixel_map(kind):
    """loc_id → MODIS 화소 번호(xe_stage2_acquire points 의 표). kind = 250 또는 500."""
    p = D_AE / "points" / "xe_labels_pixels.csv"
    if not p.exists():
        return None
    d = read_csv_cols(p, ["loc_id", f"pix{kind}_id"])
    return d.set_index("loc_id")[f"pix{kind}_id"]


def mod13q1_table(files):
    pre = "MOD13Q1_061__250m_16_days_"
    cols = ["ID", "Date", pre + "NDVI", pre + "EVI", pre + "NIR_reflectance", pre + "MIR_reflectance", pre + "pixel_reliability"]
    frames = [pd.read_csv(f, usecols=cols) for f in files]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["ID", "Date"])
    rows = []
    for pid, g in d.groupby("ID"):
        m = T.mod13q1_metrics(g.Date.values, g[pre + "NDVI"].values, g[pre + "EVI"].values, g[pre + "NIR_reflectance"].values,
                              g[pre + "MIR_reflectance"].values, g[pre + "pixel_reliability"].values)
        rows.append(dict(pix=str(pid), **m))
    return pd.DataFrame(rows)


def snow_table(files, product):
    if product == "MOD10A1":
        col = "MOD10A1_061_NDSI_Snow_Cover"
        fn = T.mod10a1_metrics
    else:
        col = "MOD10A2_061_Maximum_Snow_Extent"
        fn = T.mod10a2_metrics
    frames = []
    for f in files:
        for ch in pd.read_csv(f, usecols=["ID", "Date", col], chunksize=2_000_000):
            frames.append(ch)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["ID", "Date"])
    rows = []
    for pid, g in d.groupby("ID"):
        m = fn(g.Date.values, g[col].values)
        m.pop("by_year", None)
        rows.append(dict(pix=str(pid), **m))
    return pd.DataFrame(rows)


# ================================================================ 조각 입출력
def source_info(paths):
    """원자료 파일 목록의 수, 크기 합, 가장 늦은 수정 시각(KST). 마감 판정에 쓴다."""
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        return dict(n_files=0, bytes=0, latest_mtime=None)
    mt = max(p.stat().st_mtime for p in paths)
    return dict(n_files=len(paths), bytes=int(sum(p.stat().st_size for p in paths)),
                latest_mtime=datetime.fromtimestamp(mt, R.KST).strftime("%Y-%m-%d %H:%M:%S %z"))


def part_paths(out_dir, group):
    d = Path(out_dir) / "parts"
    return d / f"xe_{group}_v1.csv", d / f"xe_{group}_v1_meta.json"


def write_part(out_dir, group, df, pts, sources, extra=None):
    """조각 CSV(loc_id 순)와 메타. 메타에는 원자료 정보, 열별·대상별 유한값 비율, 최대 RSS 를 적는다(라벨 통계 없음)."""
    p, pm = part_paths(out_dir, group)
    check_out(p.parent)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pts[["loc_id"]].merge(df, on="loc_id", how="left").sort_values("loc_id")
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, p)
    tg = pts.set_index("loc_id").target
    fin = {}
    for c in df.columns:
        if c == "loc_id":
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        t_ = df.loc_id.map(tg).values
        fin[c] = {t: round(float(np.isfinite(v[t_ == t]).mean()), 4) if (t_ == t).any() else None for t in R.TARGETS}
    meta = dict(group=group, created=now_kst(), plan=f"{R.PLAN_DOC} {R.PLAN_SECTION}", n_rows=int(len(df)), targets=sorted(pts.target.unique().tolist()),
                columns=[c for c in df.columns if c != "loc_id"],
                finite_by_target=fin, sources=sources, sha256=R.sha256_file(p), max_rss_mb=max_rss_mb(), script="scripts/1_data_prep/xe_point_covariates.py")
    meta.update(extra or {})
    tmp = pm.with_name(pm.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, pm)
    log(f"[{group}] 조각 {rel(p)}: 행 {len(df)}, 열 {len(meta['columns'])}, sha256 {meta['sha256'][:16]}")
    return meta


# ================================================================ 하위 명령
def cmd_h0(a, pts):
    cols = R.GROUPS["H0"]
    d = read_csv_cols(COV_EXT, ["loc_id"] + cols)
    return write_part(a.out_dir, "H0", d, pts, dict(H0=dict(path=rel(COV_EXT), sha256=R.sha256_file(COV_EXT), **source_info([COV_EXT]))))


def cmd_t2(a, pts):
    df, notes = run_clusters(t2_cluster, pts, a.workers, a.limit, "t2")
    files = sorted(set(str(p) for k, idx in clusters_of(pts, a.limit)
                       for p in tiles_for(cluster_bounds(pts.lat.values[idx], pts.lon.values[idx], DEM_MARGIN_DEG), "dem")))
    src = dict(T2=dict(dem=rel(D_DEM), gsw=rel(D_GSW), hansen=rel(D_HANSEN), **source_info(files)))
    rule = dict(twi_pt="D8 누적(pysheds, 점 묶음 경계 상자 ± 0.05° 창), ln(a/tanβ), 30 m 점 화소", tpi_90="z − 3 × 3 평균", tpi_270="z − 9 × 9 평균",
                slope_90="3 × 3 경사(°) 평균", gsw_occ_pt="GSW occurrence 3 × 3 평균(255 결측)", treecover_pt="Hansen treecover2000 3 × 3 평균",
                px="slope_px, gsw_occ_px, treecover_px = 점 화소(민감도)")
    return write_part(a.out_dir, "T2", df, pts, src, dict(rule=rule, n_clusters=len(notes), cluster_sec=float(sum(n.get("sec", 0) for n in notes)),
                                                          acc_errors=sum(1 for n in notes if "acc_error" in n), limit=int(a.limit)))


def cmd_o(a, pts):
    out, used, files = soilgrids_points(pts)
    d = pd.DataFrame(dict(loc_id=pts.loc_id.values, **out))
    for col, f in NCSCD_LAYERS.items():
        p = D_NCSCD / f
        d[col] = sample_points(p, pts.lon.values, pts.lat.values) / 10.0 if p.exists() else np.nan   # hg C m-2 → kg C m-2
    d["sg250_window"] = used["sg250_soc_0_5"]
    ncs = [D_NCSCD / f for f in NCSCD_LAYERS.values() if (D_NCSCD / f).exists()]
    src = dict(O=dict(soilgrids=rel(D_SG), ncscd=rel(D_NCSCD), **source_info(files + ncs)))
    rule = dict(sg250="SoilGrids 250 m 정렬 창의 점 화소(층마다 이름 순 첫 유효 창, 값 > 0), 원 단위", ncscd="NCSCDv2 0.012° 점 화소, kg C m-2(원값 hg ÷ 10)")
    return write_part(a.out_dir, "O", d.drop(columns=["sg250_window"]), pts, src,
                      dict(rule=rule, sg_windows_used=pd.Series(used["sg250_soc_0_5"]).value_counts().to_dict()))


def cmd_v(a, pts):
    df, notes = run_clusters(wc_cluster, pts, a.workers, a.limit, "v-worldcover")
    d = pts[["loc_id"]].merge(df, on="loc_id", how="left")
    wc_files = sorted(set(str(p) for k, idx in clusters_of(pts, a.limit)
                          for p in tiles_for(cluster_bounds(pts.lat.values[idx], pts.lon.values[idx], WC_MARGIN_DEG), "wc")))
    src = dict(V_wc=dict(path=rel(D_WC), **source_info(wc_files)))
    if CAVM_TIF.exists():
        import rasterio
        with rasterio.open(CAVM_TIF) as s_:
            crs = s_.crs.to_wkt()
        x, y = to_crs(pts.lon.values, pts.lat.values, crs)
        d["cavm_class"] = sample_points(CAVM_TIF, x, y, nodata_extra=(127,))
        src["V_cavm"] = dict(path=rel(CAVM_TIF), sha256=R.sha256_file(CAVM_TIF), **source_info([CAVM_TIF]))
    else:
        d["cavm_class"] = np.nan
        src["V_cavm"] = dict(path=rel(CAVM_TIF), **source_info([]))
    files = ae_results("MOD13Q1")
    pm = pixel_map(250)
    vcols = ["ndvi250_jja", "evi250_jja", "ndwi250_jja", "ndvi250_max"]
    if files and pm is not None:
        tab = mod13q1_table(files).set_index("pix")
        pix = d.loc_id.map(pm)
        for c in vcols:
            d[c] = pix.map(tab[c]).astype(float).values
        cov = float(pix.isin(tab.index).mean())
    else:
        for c in vcols:
            d[c] = np.nan
        cov = 0.0
    src["V_mod13q1"] = dict(path=rel(D_AE), files=[rel(f) for f in files], pixel_coverage=round(cov, 4),
                            **source_info(files))
    rule = dict(worldcover="3 × 3 창(30 m) 등급 비율, 분모 = 자료가 있는 칸, 범례 R.WC_CODES", wc_px="점 화소 원핫(민감도)", cavm="1 km 점 화소 부호(127 결측)",
                mod13q1="합성 시작일 2015–2020 6–8월, pixel_reliability 0·1, 평균, ndvi250_max = 합성 일차별 기후값의 최댓값")
    return write_part(a.out_dir, "V", d, pts, src, dict(rule=rule))


def s_choose(pts, results_fn=None, table_fn=None, pm=None):
    """S 군 경로 고르기(계획 2.5 대체 순서: MOD10A1 → MOD10A2 → 군 제외). MOD10A1 결과가 라벨 화소의 90 % 이상을 덮으면 MOD10A1,
    아니면 MOD10A2 가 더 많이(같거나) 덮을 때 MOD10A2, 그 밖에 MOD10A1 결과가 있으면 MOD10A1, 없으면 none. 반환 (경로, 파일, 표, 덮음 비율)."""
    results_fn = results_fn or ae_results
    table_fn = table_fn or snow_table
    pm = pixel_map(500) if pm is None else pm
    tabs, cov, files = {}, {}, {}
    for prod in ("MOD10A1", "MOD10A2"):
        fs = results_fn(prod)
        if fs and pm is not None:
            t_ = table_fn(fs, prod)
            tabs[prod], files[prod] = t_, fs
            cov[prod] = float(pts.loc_id.map(pm).isin(set(t_.pix)).mean()) if len(t_) else 0.0
    if cov.get("MOD10A1", -1.0) >= R.FINITE_MIN:
        ch = "MOD10A1"
    elif "MOD10A2" in cov and cov["MOD10A2"] >= cov.get("MOD10A1", 0.0):
        ch = "MOD10A2"
    elif "MOD10A1" in cov:
        ch = "MOD10A1"
    else:
        return "none", [], pd.DataFrame(), cov
    return ch, files[ch], tabs[ch], cov


def cmd_s(a, pts):
    choice, files, tab, cov = s_choose(pts)
    pm = pixel_map(500)
    d = pts[["loc_id"]].copy()
    cols = R.GROUPS["S"]
    if len(tab):
        pix = d.loc_id.map(pm)
        t_ = tab.set_index("pix")
        for c in cols:
            d[c] = pix.map(t_[c]).astype(float).values
    else:
        for c in cols:
            d[c] = np.nan
    src = dict(S=dict(path=rel(D_AE), product=choice, files=[rel(f) for f in files], pixel_coverage=cov,
                      **source_info(files)))
    rule = dict(MOD10A1="일 NDSI 10–100 적설, 0–9 무적설, 그 밖은 가장 가까운 판정 날로 채움, 수문년(9–8월, 끝 해) 2015–2020 적설 일수 평균, "
                         "달력 연도 2015–2020 첫 소멸 전이·마지막 시작 전이 일차 평균",
                MOD10A2="등록 대체 정의: 적설 표시(200) 합성 수 × 8, 첫·마지막 무적설 전이 합성의 시작일(25 = 무적설, 그 밖 판정 없음)")
    return write_part(a.out_dir, "S", d, pts, src, dict(rule=rule, path_used=choice))


# ---------------------------------------------------------------- ArcticDEM(M 군)
def ad_tile_of(x, y):
    """EPSG:3413 좌표 → ArcticDEM v4.1 모자이크 타일 이름 'rr_cc'(100 km, 원점 (−4.1e6, −4.1e6)). 10m.json 의 bbox 로 확인한 규칙."""
    r = np.floor(np.asarray(y) / 1e5).astype(int) + 41
    c = np.floor(np.asarray(x) / 1e5).astype(int) + 41
    return [f"{i:02d}_{j:02d}" for i, j in zip(r, c)]


def ad_index(res):
    """ArcticDEM 색인 JSON(컬렉션의 자식 타일 목록)을 받아 data/raw/arcticdem_xe/index 에 둔다. 반환 타일 이름 집합."""
    import requests
    d = D_AD / "index"
    check_out(d)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{res}.json"
    if not p.exists():
        r = requests.get(f"{AD_BASE}/{res}.json", timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    js = json.loads(p.read_text())
    return {l_["title"].split()[-1] for l_ in js.get("links", []) if l_.get("rel") == "child"}, p


def _ad_fetch_one(job):
    """창 하나를 /vsicurl 로 읽어 GeoTIFF 로 둔다. 10 m 실패·전부 결측이면 32 m(대체 경로)."""
    import rasterio
    from rasterio.windows import from_bounds, Window
    name, tile, bounds, tiles10, tiles32 = job
    out = dict(window=name, tile=tile, status="", res=None, file="")
    for res, have in (("10m", tiles10), ("32m", tiles32)):
        if tile not in have:
            out["status"] += f"{res}:타일 없음;"
            continue
        url = f"/vsicurl/{AD_BASE}/{res}/{tile}/{tile}_{res}_v4.1_dem.tif"
        for k in range(3):
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3",
                                  GDAL_HTTP_RETRY_DELAY="2", CPL_TMPDIR=str(D_AD), GDAL_CACHEMAX=256):
                    with rasterio.open(url) as src:
                        w = from_bounds(*bounds, transform=src.transform)
                        w = Window(math.floor(w.col_off), math.floor(w.row_off), math.ceil(w.width) + 1, math.ceil(w.height) + 1)
                        w = w.intersection(Window(0, 0, src.width, src.height))
                        arr = src.read(1, window=w)
                        tr = src.window_transform(w)
                        nod = src.nodata
                        prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype, crs=src.crs, transform=tr,
                                    nodata=nod, compress="deflate")
                valid = np.isfinite(arr) & ((arr != nod) if nod is not None else True)
                if not valid.any():
                    out["status"] += f"{res}:전부 결측;"
                    break
                dst = D_AD / "windows" / f"{name}_{res}.tif"
                tmp = dst.with_name(dst.name + ".part")
                with rasterio.open(tmp, "w", **prof) as dd:
                    dd.write(arr, 1)
                os.replace(tmp, dst)
                out.update(status=out["status"] + f"{res}:ok", res=int(res[:-1]), file=dst.name, url=url[len("/vsicurl/"):],
                           sha256=R.sha256_file(dst), valid_frac=round(float(valid.mean()), 4), fetched=now_kst())
                return out
            except Exception as e:                                       # noqa: BLE001
                err = repr(e)[:160]
                time.sleep(2 * (k + 1))
        else:
            out["status"] += f"{res}:읽기 실패({err});"
    return out


def cmd_m_fetch(a, pts):
    """M 군 원자료: 점 묶음(EPSG:3413 2 km 칸)마다 경계 상자 ± 200 m 창을 /vsicurl 로 읽는다(pystac_client·boto3 없이, 계획 2.5 주 경로)."""
    tiles10, p10 = ad_index("10m")
    tiles32, p32 = ad_index("32m")
    key, x, y = ad_window_keys(pts)
    tl = ad_tile_of(x, y)
    jobs, member = [], []
    for nm, idx in key.groupby(key.values).indices.items():
        idx = np.asarray(idx)
        b = (float(x[idx].min()) - AD_WIN_MARGIN_M, float(y[idx].min()) - AD_WIN_MARGIN_M, float(x[idx].max()) + AD_WIN_MARGIN_M,
             float(y[idx].max()) + AD_WIN_MARGIN_M)
        jobs.append((nm, tl[idx[0]], b, tiles10, tiles32))
        member += [(nm, int(lid)) for lid in pts.loc_id.values[idx]]
    if a.limit and a.limit > 0:
        jobs = jobs[:int(a.limit)]
    (D_AD / "windows").mkdir(parents=True, exist_ok=True)
    have = {p.name for p in (D_AD / "windows").glob("*.tif")}
    todo = [j for j in jobs if f"{j[0]}_10m.tif" not in have and f"{j[0]}_32m.tif" not in have]
    log(f"[m-fetch] 창 {len(jobs)}개(이미 있음 {len(jobs) - len(todo)}), 타일 10 m {len(tiles10)}·32 m {len(tiles32)}")
    res_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(int(a.workers), 4))) as ex:
        for i, f in enumerate(as_completed([ex.submit(_ad_fetch_one, j) for j in todo]), 1):
            res_rows.append(f.result())
            if i % 25 == 0 or i == len(todo):
                log(f"[m-fetch] {i}/{len(todo)} · {time.time() - t0:.0f}s")
    idx_p = D_AD / "windows_index.csv"
    old = pd.read_csv(idx_p) if idx_p.exists() else pd.DataFrame()
    new = pd.DataFrame(res_rows)
    allw = pd.concat([old, new], ignore_index=True).drop_duplicates("window", keep="last") if len(new) else old
    allw.to_csv(idx_p, index=False)
    pd.DataFrame(member, columns=["window", "loc_id"]).to_csv(D_AD / "windows_members.csv", index=False)
    meta = dict(created=now_kst(), plan=f"{R.PLAN_DOC} {R.PLAN_SECTION} M 군(주 경로 10 m 색인 JSON + /vsicurl, 대체 32 m)", base=AD_BASE,
                index={"10m": dict(path=rel(p10), sha256=R.sha256_file(p10)), "32m": dict(path=rel(p32), sha256=R.sha256_file(p32))},
                licence="CC-BY-4.0(STAC 컬렉션 license 항목). 인용: Porter et al., ArcticDEM v4.1, Harvard Dataverse, https://doi.org/10.7910/DVN/3VDC4W",
                n_windows=int(len(allw)), status=allw.status.astype(str).str.split(";").str[-1].value_counts().to_dict() if len(allw) else {},
                window_rule=f"EPSG:3413 {AD_WIN_BIN_M:.0f} m 칸·타일별 점 경계 상자 ± {AD_WIN_MARGIN_M:.0f} m")
    (D_AD / "fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return meta


def ad_window_keys(pts):
    """점마다 ArcticDEM 창 이름(타일_2 km 칸 x_2 km 칸)과 EPSG:3413 좌표. m-fetch 와 m 이 같은 규칙을 쓴다."""
    x, y = to_crs(pts.lon.values, pts.lat.values, "EPSG:3413")
    tl = ad_tile_of(x, y)
    gx, gy = np.floor(x / AD_WIN_BIN_M).astype(int), np.floor(y / AD_WIN_BIN_M).astype(int)
    return pd.Series([f"{t}_{i}_{j}" for t, i, j in zip(tl, gx, gy)], index=pts.index), x, y


def cmd_m(a, pts):
    """M 군: 받은 창(windows_index.csv)에서 점의 창을 좌표로 다시 찾아 지표를 계산한다(창 이름 규칙은 m-fetch 와 같다)."""
    import rasterio
    cols = R.GROUPS["M"] + ["ad_slope_px", "ad_curv_px", "ad_res"]
    out = {k: np.full(len(pts), np.nan) for k in cols}
    idx_p = D_AD / "windows_index.csv"
    files = []
    if idx_p.exists():
        wi = pd.read_csv(idx_p)
        wi = wi[wi.file.astype(str).str.endswith(".tif")].set_index("window")
        key, x, y = ad_window_keys(pts)
        for nm, ii in key.groupby(key.values).indices.items():
            if nm not in wi.index:
                continue
            r_ = wi.loc[nm]
            p = D_AD / "windows" / str(r_.file)
            if not p.exists():
                continue
            files.append(p)
            with rasterio.open(p) as src:
                z = src.read(1).astype(float)
                if src.nodata is not None:
                    z[z == src.nodata] = np.nan
                z[z < -500] = np.nan
                rr, cc = T.pixel_rc(src.transform, x[ii], y[ii])
                res = T.arcticdem_points(z, float(abs(src.transform.a)), rr, cc)
            for k, v in res.items():
                out[k][ii] = v
            out["ad_res"][ii] = float(r_.res)
    d = pd.DataFrame(dict(loc_id=pts.loc_id.values, **out))
    src = dict(M=dict(path=rel(D_AD), **source_info(files)))
    rule = dict(ad_slope_30="3 × 3(10 m) 경사 평균(°)", ad_curv_30="3 × 3 라플라시안 평균(1/m)", ad_tpi_50="z − 5 × 5 평균", ad_tpi_150="z − 15 × 15 평균",
                ad_rough_50="5 × 5 표준편차", res32="32 m 대체 창은 같은 미터 규모를 반폭 max(1, round((L/res − 1)/2)) 화소로 근사", px="ad_slope_px, ad_curv_px = 점 화소")
    return write_part(a.out_dir, "M", d, pts, src, dict(rule=rule))


# ---------------------------------------------------------------- 결합·확인·최종
def source_late(src_meta):
    """조각 메타의 원자료별 가장 늦은 수정 시각이 마감을 넘었는가. 반환 {원자료 키: (늦음 여부, 시각, 마감)}."""
    out = {}
    for k, v in (src_meta or {}).items():
        if k not in R.DEADLINES:
            continue
        t_ = v.get("latest_mtime")
        if not t_:
            continue
        tt = datetime.strptime(t_, "%Y-%m-%d %H:%M:%S %z")
        out[k] = (bool(tt > R.DEADLINES[k]), t_, R.DEADLINES[k].strftime("%Y-%m-%d %H:%M:%S %z"))
    return out


LATE_COLS = {"O": R.GROUPS["O"], "M": R.GROUPS["M"] + ["ad_slope_px", "ad_curv_px"], "V_wc": list(R.WC_CODES) + [R.PX_MAP[k] for k in R.WC_CODES],
             "V_cavm": ["cavm_class"], "V_mod13q1": ["ndvi250_jja", "evi250_jja", "ndwi250_jja", "ndvi250_max"], "S": R.GROUPS["S"]}


def cmd_assemble(a, pts):
    out_dir = Path(a.out_dir)
    feat = pts[["loc_id", "target"]].copy()
    parts, late = {}, {}
    for g in R.H_GROUPS:
        p, pm = part_paths(out_dir, g)
        if not p.exists():
            continue
        m = json.loads(pm.read_text()) if pm.exists() else {}
        if m.get("sha256") and m["sha256"] != R.sha256_file(p):
            raise SystemExit(f"[assemble] 조각 {p.name} 의 sha256 이 메타와 다르다")
        miss_t = sorted(set(pts.target) - set(m.get("targets", pts.target.unique())))
        if miss_t and not a.force:                                       # 일부 대상만 만든 조각이 결합되면 그 대상의 군이 90 % 규칙으로 빠진다
            raise SystemExit(f"[assemble] 조각 {p.name} 에 대상 {miss_t} 의 행이 없다(그 대상을 포함해 다시 만든다. --force 면 결측으로 결합)")
        d = pd.read_csv(p)
        d = d[[c for c in d.columns if c == "loc_id" or c not in feat.columns]]
        feat = feat.merge(d, on="loc_id", how="left")
        parts[g] = dict(path=rel(p) if ROOT in p.parents else str(p), sha256=m.get("sha256", R.sha256_file(p)),
                        n_rows=int(len(d)), created=m.get("created"), sources=m.get("sources", {}), limit=m.get("limit", 0))
        for k, (is_late, t_, dl) in source_late(m.get("sources")).items():
            if is_late:
                late[k] = dict(latest_mtime=t_, deadline=dl, cols=LATE_COLS.get(k, []))
    for k, v in late.items():                                            # 마감을 넘겨 확보한 원자료의 열은 결측으로 둔다(계획 2.5)
        for c in v["cols"]:
            if c in feat.columns:
                feat[c] = np.nan
    for c in R.H_COLS + R.PX_COLS:                                       # 등록 열은 항상 있게 한다(없는 군은 모두 결측)
        if c not in feat.columns:
            feat[c] = np.nan
    cover_extra = {v: R.cover_cols(feat.columns, v) for v in R.COVER}
    cover_extra = {k: v for k, v in cover_extra.items() if v}
    dec = R.group_decisions(feat, feat.set_index("loc_id").target, extra=cover_extra)
    order = ["loc_id", "target"] + R.H_COLS + R.PX_COLS + [c for c in feat.columns if c not in ["loc_id", "target"] + R.H_COLS + R.PX_COLS]
    feat = feat[order].sort_values("loc_id").reset_index(drop=True)
    fp = out_dir / f"{FEAT_NAME}.csv"
    check_out(fp.parent)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + f".tmp{os.getpid()}")
    feat.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, fp)
    re_read = pd.read_csv(fp)                                            # 해시는 파일에서 다시 읽은 값으로 계산한다(판 사이 비교와 같은 경로)
    gsha = {g: R.cols_sha(re_read, R.GROUPS[g]) for g in R.H_GROUPS}
    stage = "stage2" if any(g in parts for g in R.STAGE2) else "stage1"
    meta = dict(created=now_kst(), plan=f"{R.PLAN_DOC} {R.PLAN_SECTION}", plan_commit=R.PLAN_COMMIT, file=rel(fp) if ROOT in fp.parents else str(fp),
                sha256=R.sha256_file(fp), n_rows=int(len(feat)), n_by_target=feat.target.value_counts().to_dict(), stage=stage, groups=R.GROUPS,
                px_cols=R.PX_MAP, cover_cols=cover_extra, parts=parts, late_sources=late, group_decisions=dec, group_cols_sha256=gsha,
                deadlines={k: v.strftime("%Y-%m-%d %H:%M:%S %z") for k, v in R.DEADLINES.items()}, finite_min=R.FINITE_MIN,
                finite_rule="군의 유한값 비율 = 대상 행 × 군 열 칸 가운데 유한한 칸의 비율. 90 % 미만이면 그 군을 그 대상에서 뺀다(계획 2.5)",
                limited=any(int(v.get("limit") or 0) > 0 for v in parts.values()), finalized=False, max_rss_mb=max_rss_mb())
    mp = out_dir / f"{FEAT_NAME}_meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"[assemble] {fp.name}: 행 {len(feat)}, 조각 {sorted(parts)}, 마감 초과 {sorted(late) or '없음'}, sha256 {meta['sha256'][:16]}")
    for tg in R.TARGETS:
        s_ = ", ".join(f"{g} {'포함' if v['included'] else '제외'}({v['finite_frac']})" for g, v in dec[tg].items())
        log(f"[assemble] {tg}: {s_}")
    return meta


def cmd_qa(a, pts):
    """등록 전에 허용한 확인(계획 2.5): 새 열의 유한값 비율, 격자·위치 안 변동 비율, 새 열 사이 상관. 라벨·잔차는 쓰지 않는다."""
    out_dir = Path(a.out_dir)
    fp = out_dir / f"{FEAT_NAME}.csv"
    feat = pd.read_csv(fp)
    cov = read_csv_cols(FB, ["loc_id", "e5_sqrt_tdd"])                     # 격자 묶음의 √TDD(공변량)
    d = pts[["loc_id", "lat", "lon", "target"]].merge(cov, on="loc_id", how="left").merge(feat.drop(columns=["target"]), on="loc_id", how="left")
    blk = (np.floor(d.lat / 0.5).astype(int) * 100000 + np.floor(d.lon / 0.5).astype(int)).astype(str)
    grid = blk + "|" + d.e5_sqrt_tdd.map(lambda v: f"{v:.12g}")
    ky = np.floor(d.lat / 0.009).astype(int)
    kx = np.floor(d.lon * np.cos(np.radians((ky + 0.5) * 0.009)) / 0.009).astype(int)
    loc = grid + "#" + ky.astype(str) + "_" + kx.astype(str)
    cols = [c for c in R.H_COLS + R.PX_COLS if c in d.columns]
    rows = []
    for tg in R.TARGETS:
        m = (d.target == tg).values
        for c in cols:
            v = pd.to_numeric(d.loc[m, c], errors="coerce")
            fin = float(np.isfinite(v).mean()) if m.any() else np.nan
            ok = np.isfinite(v.values)
            vg = v[ok].groupby(grid[m][ok].values).transform("nunique") if ok.any() else pd.Series(dtype=float)
            vl = v[ok].groupby(loc[m][ok].values).transform("nunique") if ok.any() else pd.Series(dtype=float)
            rows.append(dict(target=tg, col=c, finite_frac=round(fin, 4), frac_rows_grid_varying=round(float((vg > 1).mean()), 4) if len(vg) else np.nan,
                             frac_rows_loc_varying=round(float((vl > 1).mean()), 4) if len(vl) else np.nan, n_unique=int(v.nunique())))
    qa = pd.DataFrame(rows)
    corr = []
    for tg in R.TARGETS:
        m = (d.target == tg).values
        sub = d.loc[m, [c for c in R.H_COLS if c in d.columns]].apply(pd.to_numeric, errors="coerce")
        sub = sub.loc[:, sub.notna().mean() > 0]
        cm = sub.corr(method="spearman", min_periods=30)
        for i, c1 in enumerate(cm.columns):
            for c2 in cm.columns[i + 1:]:
                corr.append(dict(target=tg, col_a=c1, col_b=c2, spearman=cm.loc[c1, c2]))
    qp, cp = out_dir / f"{FEAT_NAME}_qa.csv", out_dir / f"{FEAT_NAME}_qa_corr.csv"
    check_out(qp.parent)
    qa.to_csv(qp, index=False)
    pd.DataFrame(corr).to_csv(cp, index=False, float_format="%.4g")
    log(f"[qa] 열 {len(cols)} × 대상 {len(R.TARGETS)} → {qp.name}, 상관 {len(corr)}쌍 → {cp.name}(ALT·잔차 통계 없음)")
    return dict(qa=str(qp), corr=str(cp))


def cmd_finalize(a, pts):
    """최종 해시 단계(계획 2.5 'xe_feat 최종판' 행, 마감 T0 + 100 h). xe_feat_v1_final.json 과 개정 이력 문구를 쓴다. git 커밋은 하지 않는다."""
    out_dir = Path(a.out_dir)
    fp, mp = out_dir / f"{FEAT_NAME}.csv", out_dir / f"{FEAT_NAME}_meta.json"
    if not fp.exists() or not mp.exists():
        raise SystemExit("[finalize] xe_feat_v1.csv 와 메타가 없다. assemble 을 먼저 한다")
    meta = json.loads(mp.read_text())
    sha = R.sha256_file(fp)
    if sha != meta.get("sha256"):
        raise SystemExit("[finalize] 표의 sha256 이 메타와 다르다. assemble 을 다시 한다")
    if meta.get("limited"):
        raise SystemExit("[finalize] 시험 실행(--limit) 조각으로 만든 표는 최종판이 될 수 없다")
    final_p = out_dir / f"{FEAT_NAME}_final.json"
    if final_p.exists() and not a.force:
        old = json.loads(final_p.read_text())
        if old.get("sha256") != sha:
            raise SystemExit("[finalize] 이미 다른 최종 해시가 있다(--force 로만 바꾼다. 개정 이력에 사유를 적는다)")
        log(f"[finalize] 같은 최종 해시가 이미 있다: {sha[:16]}")
        return old
    now = datetime.now(R.KST)
    after = now > R.DEADLINES["final"]
    dec = meta["group_decisions"]
    dropped = {tg: [g for g, v in gd.items() if not v["included"]] for tg, gd in dec.items()}
    final = dict(file=meta["file"], sha256=sha, finalized=now.strftime("%Y-%m-%d %H:%M:%S %z"), deadline=R.DEADLINES["final"].strftime("%Y-%m-%d %H:%M:%S %z"),
                 after_deadline=bool(after), stage=meta.get("stage"), n_rows=meta.get("n_rows"), group_decisions=dec, dropped_groups=dropped,
                 late_sources=meta.get("late_sources", {}), group_cols_sha256=meta.get("group_cols_sha256", {}), parts={g: v.get("sha256") for g, v in meta.get("parts", {}).items()},
                 s_path=meta.get("parts", {}).get("S", {}).get("sources", {}).get("S", {}).get("product", "none"),
                 commit_note="이 파일과 sha256 을 계획 개정 이력에 커밋한다(사용자 확인 뒤). 커밋 전에는 R3 를 제출하지 않고 xt2 표를 열지 않는다(계획 0.3)")
    if final_p.exists():
        final["previous"] = json.loads(final_p.read_text())
    final_p.write_text(json.dumps(final, ensure_ascii=False, indent=1, default=str))
    meta["finalized"] = True
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    drop_txt = "; ".join(f"{tg} {', '.join(v)}" for tg, v in dropped.items() if v) or "없음"
    late_txt = ", ".join(sorted(meta.get("late_sources", {}))) or "없음"
    entry = (f"- {final['finalized'][:16]} (XE xe_feat 최종판, 계획 2.5): `{meta['file']}` sha256 `{sha}`(행 {meta.get('n_rows')}). "
             f"2단계 적설 경로 {final['s_path']}. 90 % 규칙으로 뺀 군: {drop_txt}. 마감을 넘겨 결측으로 둔 원자료: {late_txt}."
             + (" 최종 마감(T0 + 100 h)을 넘긴 시점의 고정이다." if after else ""))
    ep = out_dir / f"{FEAT_NAME}_revision_entry.md"
    ep.write_text(entry + "\n")
    log(f"[finalize] 최종 sha256 {sha[:16]} → {final_p.name}. 개정 이력 문구 → {ep.name}(커밋은 사용자가 한다)")
    print(entry, flush=True)
    return final


CMDS = dict(h0=cmd_h0, t2=cmd_t2, o=cmd_o, v=cmd_v, s=cmd_s, m=cmd_m, assemble=cmd_assemble, qa=cmd_qa, finalize=cmd_finalize)
CMDS["m-fetch"] = cmd_m_fetch


def main(argv=None):
    ap = argparse.ArgumentParser(description="XE 점 규모 공변량 추출(계획 2.5)")
    ap.add_argument("cmds", nargs="+", choices=list(CMDS))
    ap.add_argument("--out-dir", default=str(XE_DIR), help="표·조각 경로(기본 data/processed/xe. 스모크는 data/processed/xbatch 아래)")
    ap.add_argument("--targets", default=",".join(R.TARGETS))
    ap.add_argument("--workers", type=int, default=4, help="프로세스 수(상한 4, 워커당 스레드 1)")
    ap.add_argument("--limit", type=int, default=0, help="창 묶음 수 상한(시험 실행. 0 = 전부). 이 조각으로 만든 표는 최종판이 될 수 없다")
    ap.add_argument("--mem-gb", type=float, default=10.0, help="프로세스 주소 공간 상한(GB, 0 = 두지 않음)")
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="시작 전 가용 메모리 하한(GB)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    g = mem_available_gb()
    if np.isfinite(g) and g < a.min_free_gb:
        raise SystemExit(f"[대기] 가용 메모리 {g:.1f} GB 가 하한 {a.min_free_gb} GB 아래다(계획 1절 로컬 자원). 나중에 다시 한다")
    set_local_limits(a.mem_gb)
    check_out(a.out_dir)
    tmpdir = Path(os.environ.get("CPL_TMPDIR", ""))
    if str(tmpdir).startswith(str(RAW)):                                    # 기본값(data/raw/xe_tmp)이면 만든다. 다른 경로는 사용자 책임
        tmpdir.mkdir(parents=True, exist_ok=True)
    pts = read_points(targets=[t for t in a.targets.split(",") if t])
    log(f"[points] XE 대상 행 {len(pts)}({pts.target.value_counts().to_dict()}), 창 묶음 {len(clusters_of(pts))}, GDAL 임시 {os.environ.get('CPL_TMPDIR')}")
    t0 = time.time()
    for c in a.cmds:
        CMDS[c](a, pts)
    log(f"[done] {' '.join(a.cmds)} · {time.time() - t0:.0f}s · 최대 RSS {max_rss_mb()} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
