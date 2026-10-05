"""XM 피복·식생 특징 표(1 km 라벨 셀): ESA WorldCover 2021 v200 10 m 등급 비율 11열, Sentinel-2 L2A 20 m 여름 식생 지수 3열.

등록: docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md(커밋 ff7c97f, 2026-10-05 13:23:43 +0900) 3절. 라벨 값은 읽지 않는다.

1 km 라벨 셀(등록 3절, LG 6B.3 셀, x_hires_covariates.loc_cell 과 xe_point_covariates.cmd_qa 의 위치 색인과 같다)
  ky = floor(lat/0.009), φ = (ky + 0.5)·0.009°, kx = floor(lon·cos φ/0.009).
  범위: 위도 [ky·0.009, (ky + 1)·0.009), 경도 [kx·0.009/cos φ, (kx + 1)·0.009/cos φ). 화소는 중심이 범위 안이면 셀에 넣는다.
WC 군: 셀 안 화소의 등급 비율(분모 = 자료가 있는 화소, 0 = 자료 없음). 자료 화소가 셀 화소의 10 % 미만이면 결측.
S2 군: Earth Search v1 STAC 의 sentinel-2-l2a. 2019–2023 각 해 7–8월, 장면 운량 < 30 %, COG 자산(red, nir, swir16, scl)의 HTTPS 주소가 있는 항목.
  같은 타일·같은 취득일(UTC)의 중복 항목은 s2:sequence 가 큰 것(같으면 운량이 작은 것, 그다음 항목 id). 셀마다 꼭짓점의 타일 경계 최소 여유가
  가장 큰 MGRS 타일에 배정. 타일마다 r(배정 셀 중심이 항목 geometry 안에 드는 비율) ≥ 0.9 먼저, 운량, 자료 없음 비율, 시각 순으로 6장면.
  반사도 = DN × 0.0001(모든 장면, 등록 정정 1). raster:bands 는 기준선 04.00 이후 항목에 offset −0.1 을 적지만 이 모음의 COG DN 에는 offset 이
  없다. 같은 취득의 원 기준선(02.13·02.14·03.01) 항목과 재처리(05.00) 항목의 red DN 이 같은 수준이었고(6WVB 2021-07-12 중앙값 476·466, 18XWR
  2019–2020 다섯 날짜, 3VWH 2019-07-09), 기준선 04.00·boa_offset_applied 거짓 항목의 육지 red DN 1 백분위수는 132–137 이었다. 장면별 red DN
  1 백분위수(SCL 4·5)는 진단으로만 적는다(밝은 지표·연무 장면에서 1,000 을 넘어 offset 판별에 쓸 수 없었다).
  red·nir 10 m → 같은 원점의 20 m 칸 2 × 2 평균(하나라도 자료 없음이면 결측). 유효 = SCL 4·5 이고 세 반사도 > 0.
  NDVI = (nir − red)/(nir + red), NDMI = (nir − swir16)/(nir + swir16). 화소별 유효 관측의 중앙값(시간 합성) → 셀 안 중앙값·표준편차(ddof 0).
  합성 값이 있는 화소가 셀 화소의 10 % 미만이거나 20개 미만이면 결측.
결합: 대상마다 군 유한값 비율(라벨 행 × 군 열) < 90 % 이면 그 군을 뺀다(x_hires_registry.group_finite_fraction, XE 2.5 와 같다).

하위 명령(차례로 줄 수 있다)
  cells       라벨 좌표 → 셀 표(inputs/parts/xm_cells_v1.csv). 좌표 열만 읽는다
  wc          WorldCover 셀 비율(inputs/parts/xm_wc_v1.csv). 타일은 data/raw/worldcover_v200/tiles 사본(sha256 대조), 없으면 /vsicurl 창 읽기
  s2-search   STAC 검색(해마다, 셀 중심 MultiPoint 교차) → data/raw/xm/s2/stac/, 타일 배정·장면 선택 → inputs/parts/xm_s2_selection_v1.csv
  s2-fetch    선택 장면의 셀 창 읽기(/vsicurl, 재시도) → data/raw/xm/s2/windows/<타일>/<항목>.npz. 내려받기 상한 --cap-gb(기본 25)
  s2          셀 지수 계산(inputs/parts/xm_s2_v1.csv)
  assemble    loc_id 결합 → inputs/xm_feat_v1.csv 와 xm_feat_v1_meta.json(유한값 비율, 90 % 규칙, 덮음 비율, 해시, 시간, 바이트)
자원: 스레드 8 이하(--threads, 입출력 스레드), nice 10, 주소 공간 상한 --mem-gb(기본 14), 시작 전 가용 메모리 확인(--min-free-gb).
표준 출력에는 행 수, 유한값 비율, 시간, 해시만 쓴다.
실행(ROOT 에서)
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/1_data_prep/xm_landcover_s2_features.py cells wc s2-search s2-fetch s2 assemble --threads 6
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RAW_XM = RAW / "xm"
# GDAL 임시 파일·캐시는 /home 의 data/raw/xm 아래(계획 1절 '디스크'). /vsicurl 읽기 설정(재시도, 범위 병합, HEAD 생략)
GDAL_ENV = dict(CPL_TMPDIR=str(RAW_XM / "tmp"), GDAL_CACHEMAX="1024", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="60",
                GDAL_HTTP_CONNECTTIMEOUT="30", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_INGESTED_BYTES_AT_OPEN="65536", CPL_VSIL_CURL_USE_HEAD="NO", GDAL_HTTP_VERSION="2", VSI_CACHE="TRUE")
for _k, _v in GDAL_ENV.items():
    os.environ.setdefault(_k, _v)
RIO_ENV = {k: (int(v) if str(v).isdigit() else v) for k, v in GDAL_ENV.items()}   # rasterio.Env 는 GDAL_CACHEMAX 등을 정수로 받는다

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("x_hires_registry", ROOT / "scripts" / "3_deep_learning" / "x_hires_registry.py")

KST = timezone(timedelta(hours=9))
ADDENDUM = "docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md"
ADDENDUM_COMMIT = "ff7c97f"
EXP_NAME = "XM_landcover_vegetation"
FB = PROC / "fidelity_base_v3.csv"
OUT = PROC / "xbatch" / EXP_NAME
INP = OUT / "inputs"
PARTS = INP / "parts"
FEAT_NAME = "xm_feat_v1"
D_WC = RAW / "worldcover_v200" / "tiles"
WC_MANIFEST = RAW / "worldcover_v200" / "manifest_tiles.csv"
WC_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
D_WC_XM = RAW_XM / "worldcover"
D_S2 = RAW_XM / "s2"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"
ALLOWED_OUT = (OUT, RAW_XM)

CELL_DEG = 0.009                                                            # LG 6B.3 셀 색인(약 1 km)
WC_CODES = {"wc1k_tree": 10, "wc1k_shrub": 20, "wc1k_grass": 30, "wc1k_crop": 40, "wc1k_built": 50, "wc1k_bare": 60, "wc1k_snow": 70,
            "wc1k_water": 80, "wc1k_wetland": 90, "wc1k_mangrove": 95, "wc1k_moss": 100}
WC_COLS = list(WC_CODES)
S2_COLS = ["s2_ndvi_med", "s2_ndmi_med", "s2_ndvi_sd"]
XM_GROUPS = {"WC": WC_COLS, "S2": S2_COLS}
AUX_COLS = ["wc1k_data_frac", "s2_valid_frac", "s2_n_scenes", "s2_tile"]
MIN_DATA_FRAC = 0.10                                                        # WC 자료 화소, S2 합성 화소 하한(셀 화소 대비)
S2_MIN_PX = 20
S2_YEARS = tuple(range(2019, 2024))
S2_MONTHS = ("07-01T00:00:00Z", "08-31T23:59:59Z")
S2_CLOUD_MAX = 30.0
S2_N_SCENES = 6
S2_R_MIN = 0.9
S2_SCL_VALID = (4, 5)
S2_ASSETS = ("red", "nir", "swir16", "scl")
S2_SCALE = 1e-4
S2_OFFSET = -0.1                                                            # 처리 기준선 04.00 이후 BOA_ADD_OFFSET(−1000 DN)
S2_RES = 20.0
CAP_GB_DEFAULT = 25.0
LICENCE = dict(
    worldcover="CC BY 4.0. '© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium'. "
               "Zanaga et al. 2022, ESA WorldCover 10 m 2021 v200, doi:10.5281/zenodo.7254221",
    sentinel2="Copernicus Sentinel 자료 무료·완전·공개 접근 약관(Legal Notice on the use of Copernicus Sentinel Data and Service Information). "
              "'Contains modified Copernicus Sentinel data [2019–2023]'. 접근: Earth Search v1 STAC(Element 84), AWS Open Data sentinel-cogs")
TRACE_READS: list = []                                                      # 시험: 표 읽기 기록(경로, 열)
_LOCK = threading.Lock()


def now_kst():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %z")


def log(msg):
    print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


# ================================================================ 자원·경로 보호
def check_out(path):
    p = Path(os.path.realpath(str(path)))
    roots = [Path(os.path.realpath(str(r))) for r in ALLOWED_OUT] + [Path(v) for v in os.environ.get("XM_OUT_ROOTS", "").split(os.pathsep) if v]
    if not any(p == r or r in p.parents for r in roots):
        raise SystemExit(f"[거부] 쓰기 경로 {p} 는 허용 경로 밖이다({', '.join(str(r) for r in ALLOWED_OUT)})")
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


# ================================================================ 좌표와 셀(라벨 열 금지)
def read_csv_cols(path, cols):
    """열을 지정해 읽는다. 라벨 열(R.LABEL_COLS)은 거부한다(등록 7절)."""
    cols = list(cols)
    bad = sorted(set(cols) & set(R.LABEL_COLS))
    if bad:
        raise RuntimeError(f"특징 추출 코드가 라벨 열 {bad} 을 읽으려 했다")
    TRACE_READS.append((str(path), tuple(cols)))
    return pd.read_csv(path, usecols=cols, low_memory=False)


def read_points(fb=None, targets=R.TARGETS):
    """XM 대상(알래스카, 레나, 캐나다) 행의 좌표(loc_id, lat, lon, region, target). 좌표 열만 읽는다."""
    d = read_csv_cols(FB if fb is None else fb, R.COORD_COLS)
    d["target"] = d.region.map(R.MACRO)
    return d[d.target.isin(list(targets))].sort_values("loc_id").reset_index(drop=True)


def cell_index(lat, lon):
    """LG 6B.3 셀 색인 (ky, kx). x_hires_covariates.loc_cell 과 같은 식이다."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / CELL_DEG).astype(np.int64)
    kx = np.floor(lon * np.cos(np.radians((ky + 0.5) * CELL_DEG)) / CELL_DEG).astype(np.int64)
    return ky, kx


def cell_bounds(ky, kx):
    """셀 범위 (w, s, e, n)(경도·위도 °). 위도 [ky·d, (ky + 1)·d), 경도 [kx·d/cos φ, (kx + 1)·d/cos φ), φ = (ky + 0.5)·d."""
    ky = np.asarray(ky, np.int64); kx = np.asarray(kx, np.int64)
    c = np.cos(np.radians((ky + 0.5) * CELL_DEG))
    return kx * CELL_DEG / c, ky * CELL_DEG, (kx + 1) * CELL_DEG / c, (ky + 1) * CELL_DEG


def in_cell(lon, lat, b):
    """중심 좌표가 셀 범위(반열림) 안인가."""
    w, s, e, n = b
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return (lon >= w) & (lon < e) & (lat >= s) & (lat < n)


def cell_table(pts):
    """점 표 → 셀 표(cell, ky, kx, target, w, s, e, n, clat, clon, n_labels). n_labels 는 라벨 행 수(라벨 값 아님)."""
    ky, kx = cell_index(pts.lat.values, pts.lon.values)
    p = pts.assign(ky=ky, kx=kx)
    p["cell"] = [f"{a}_{b}" for a, b in zip(ky, kx)]
    g = p.groupby("cell").agg(ky=("ky", "first"), kx=("kx", "first"), target=("target", "first"), n_targets=("target", "nunique"),
                              n_labels=("loc_id", "size")).reset_index()
    if (g.n_targets > 1).any():
        raise RuntimeError("한 셀에 대상이 둘 이상이다")
    w, s, e, n = cell_bounds(g.ky.values, g.kx.values)
    g["w"], g["s"], g["e"], g["n"] = w, s, e, n
    g["clat"], g["clon"] = 0.5 * (s + n), 0.5 * (w + e)
    return g.drop(columns=["n_targets"]).sort_values(["target", "ky", "kx"]).reset_index(drop=True), p[["loc_id", "target", "ky", "kx", "cell"]]


# ================================================================ WorldCover
def wc_tile_name(lat0, lon0):
    return f"{'N' if lat0 >= 0 else 'S'}{abs(int(lat0)):02d}{'E' if lon0 >= 0 else 'W'}{abs(int(lon0)):03d}"


def wc_tiles_for(b):
    """셀 범위가 닿는 3° 타일 이름(아래·왼쪽 모서리 이름)."""
    w, s, e, n = b
    out = []
    for la in range(int(math.floor(s / 3)), int(math.floor((n - 1e-12) / 3)) + 1):
        for lo in range(int(math.floor(w / 3)), int(math.floor((e - 1e-12) / 3)) + 1):
            out.append(wc_tile_name(la * 3, lo * 3))
    return out


def geo_window(transform, b, width, height, pad=2):
    """경위도 격자에서 범위 b 를 덮는 정수 창 (row0, col0, h, w)(여유 pad 화소, 격자 안으로 자른다). 비면 None."""
    a, c0, ey, f0 = transform.a, transform.c, abs(transform.e), transform.f
    w, s, e, n = b
    col0 = max(0, int(math.floor((w - c0) / a)) - pad)
    col1 = min(int(width), int(math.ceil((e - c0) / a)) + pad)
    row0 = max(0, int(math.floor((f0 - n) / ey)) - pad)
    row1 = min(int(height), int(math.ceil((f0 - s) / ey)) + pad)
    if col1 <= col0 or row1 <= row0:
        return None
    return row0, col0, row1 - row0, col1 - col0


def geo_centers(transform, row0, col0, h, w):
    """창 화소 중심의 (경도, 위도) 격자."""
    a, c0, ey, f0 = transform.a, transform.c, abs(transform.e), transform.f
    lon = c0 + (col0 + np.arange(w) + 0.5) * a
    lat = f0 - (row0 + np.arange(h) + 0.5) * ey
    return np.meshgrid(lon, lat)


def wc_counts(arr, inmask, nodata=0):
    """셀 안 화소의 등급 수(dict), 셀 화소 수, 자료 화소 수."""
    v = np.asarray(arr)[np.asarray(inmask, bool)]
    n_in = int(v.size)
    ok = v != nodata
    out = {k: int(np.sum(v[ok] == code)) for k, code in WC_CODES.items()}
    return out, n_in, int(ok.sum())


def wc_fractions(counts, n_in, n_data, min_frac=MIN_DATA_FRAC):
    """등급 비율(분모 = 자료 화소)과 자료 화소 비율. 자료 화소가 셀 화소의 min_frac 미만이면 비율은 결측."""
    frac = n_data / n_in if n_in > 0 else float("nan")
    if not n_in or n_data <= 0 or frac < min_frac:
        return {k: float("nan") for k in WC_CODES}, frac
    return {k: counts[k] / n_data for k in WC_CODES}, frac


def wc_manifest():
    if not WC_MANIFEST.exists():
        return {}
    m = pd.read_csv(WC_MANIFEST)
    return dict(zip(m.tile.astype(str), m.sha256.astype(str)))


def wc_source(tile, verified):
    """타일 원천: 사본(sha256 이 매니페스트와 같으면) 또는 /vsicurl 주소. verified = {타일: (경로, sha256, 방식)} 캐시."""
    if tile in verified:
        return verified[tile]
    p = D_WC / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    man = wc_manifest()
    if p.exists():
        sha = R.sha256_file(p)
        if man.get(tile) == sha:
            verified[tile] = (str(p), sha, "사본(sha256 대조 일치)")
            return verified[tile]
    verified[tile] = ("/vsicurl/" + WC_URL.format(tile=tile), "", "vsicurl")
    return verified[tile]


def cmd_wc(a, pts, cells):
    """WC 군: 셀마다 닿는 타일의 창을 읽어 셀 안 화소 등급 수를 더한다. /vsicurl 로 읽은 창은 data/raw/xm/worldcover 에 GeoTIFF 로 둔다."""
    import rasterio
    from rasterio.windows import Window
    t0 = time.time()
    verified, rows, used = {}, [], {}
    by_tile = {}
    for i, r in enumerate(cells.itertuples()):
        for t in wc_tiles_for((r.w, r.s, r.e, r.n)):
            by_tile.setdefault(t, []).append(i)
    acc = {i: [dict.fromkeys(WC_CODES, 0), 0, 0] for i in range(len(cells))}
    for t, idx in sorted(by_tile.items()):
        path, sha, how = wc_source(t, verified)
        used[t] = dict(path=rel(path) if not path.startswith("/vsicurl/") else path, sha256=sha, how=how, n_cells=len(idx))
        with rasterio.Env(**RIO_ENV), rasterio.open(path) as src:
            for i in idx:
                r = cells.iloc[i]
                b = (r.w, r.s, r.e, r.n)
                win = geo_window(src.transform, b, src.width, src.height)
                if win is None:
                    continue
                row0, col0, h, w = win
                arr = src.read(1, window=Window(col0, row0, w, h))
                if how == "vsicurl":                                    # 받은 창은 data/raw/xm/worldcover 에 둔다(재현용)
                    d = check_out(D_WC_XM)
                    d.mkdir(parents=True, exist_ok=True)
                    prof = dict(driver="GTiff", height=h, width=w, count=1, dtype=arr.dtype, crs=src.crs,
                                transform=src.window_transform(Window(col0, row0, w, h)), compress="deflate")
                    with rasterio.open(d / f"{t}_{r.cell}.tif", "w", **prof) as dd:
                        dd.write(arr, 1)
                lon, lat = geo_centers(src.transform, row0, col0, h, w)
                cnt, n_in, n_data = wc_counts(arr, in_cell(lon, lat, b))
                for k in WC_CODES:
                    acc[i][0][k] += cnt[k]
                acc[i][1] += n_in
                acc[i][2] += n_data
    for i, r in enumerate(cells.itertuples()):
        fr, dfrac = wc_fractions(*acc[i])
        rows.append(dict(cell=r.cell, wc1k_data_frac=dfrac, wc1k_n_px=acc[i][1], **fr))
    df = pd.DataFrame(rows)
    src = dict(tiles=used, n_tiles=len(used), n_copy=sum(1 for v in used.values() if v["how"].startswith("사본")),
               n_vsicurl=sum(1 for v in used.values() if v["how"] == "vsicurl"), manifest=rel(WC_MANIFEST))
    rule = dict(cell="LG 6B.3 셀, 화소 중심 포함", fraction="등급 수 / 자료 화소 수(0 = 자료 없음)", min_data_frac=MIN_DATA_FRAC, codes=WC_CODES)
    return write_part("wc", df, src, dict(rule=rule, seconds=round(time.time() - t0, 1)))


# ================================================================ Sentinel-2: STAC 검색·타일 배정·장면 선택
def _post(url, body, tries=5, timeout=120):
    import requests
    err = None
    for k in range(tries):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:                                           # noqa: BLE001
            err = repr(e)[:200]
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"STAC 요청 실패({tries}회): {err}")


def compact_item(f):
    """STAC 항목 → 필요한 항목만(속성, geometry, 자산 4개의 주소·raster:bands·proj)."""
    p = f.get("properties", {})
    keep = ("datetime", "platform", "grid:code", "eo:cloud_cover", "s2:nodata_pixel_percentage", "s2:sequence", "s2:datatake_id",
            "s2:processing_baseline", "earthsearch:boa_offset_applied", "proj:epsg", "s2:generation_time", "s2:product_uri")
    assets = {}
    for an in S2_ASSETS:
        x = (f.get("assets") or {}).get(an) or {}
        assets[an] = {k: x.get(k) for k in ("href", "raster:bands", "proj:transform", "proj:shape")}
    return dict(id=f.get("id"), properties={k: p.get(k) for k in keep}, geometry=f.get("geometry"), assets=assets)


def stac_search_year(year, points, limit=100):
    """한 해 7–8월, 운량 < 30 %, 셀 중심 MultiPoint 와 교차하는 항목(다음 쪽 연결을 따라간다). 반환 (항목 목록, 요청 수, 첫 요청 본문)."""
    body = {"collections": [STAC_COLLECTION], "intersects": {"type": "MultiPoint", "coordinates": points},
            "datetime": f"{year}-{S2_MONTHS[0]}/{year}-{S2_MONTHS[1]}", "limit": int(limit),
            "query": {"eo:cloud_cover": {"lt": S2_CLOUD_MAX}}}
    out, n_req, url, b = [], 0, STAC_URL, body
    while True:
        js = _post(url, b)
        n_req += 1
        out += [compact_item(f) for f in js.get("features", [])]
        nxt = [l_ for l_ in js.get("links", []) if l_.get("rel") == "next"]
        if not nxt or not js.get("features"):
            break
        url, b = nxt[0].get("href", STAC_URL), nxt[0].get("body") or {}
        if not b:
            break
    return out, n_req, body


def cog_ok(item):
    """네 자산이 모두 HTTPS COG(.tif) 주소인가(JP2·요청자 지불 버킷 항목은 후보가 아니다)."""
    for an in S2_ASSETS:
        h = str((item["assets"].get(an) or {}).get("href") or "")
        if not (h.startswith("https://") and h.endswith(".tif")):
            return False
    return True


def meta_offset_flag(props) -> bool:
    """메타데이터만 본 offset 표지(기록용): 처리 기준선 ≥ 04.00 이고 earthsearch:boa_offset_applied 가 참이 아니다. 반사도 계산에는 쓰지 않는다
    (등록 정정 1: 이 모음의 COG DN 에는 표지와 관계없이 offset 이 없었다)."""
    bl = str(props.get("s2:processing_baseline") or "00.00")
    try:
        major = float(bl)
    except ValueError:
        major = 0.0
    return bool(major >= 4.0 and props.get("earthsearch:boa_offset_applied") is not True)


DN_OFFSET = 1000.0                                                          # BOA_ADD_OFFSET(DN)
DN_P = 1.0                                                                  # 진단 백분위수


def dn_offset_check(red_dn20, scl20):
    """장면 진단(기록만, 등록 정정 1): SCL 4·5 화소에서 red DN(20 m 칸 평균)의 1 백분위수가 1,000 이상인가. offset 이 남은 장면이면 참이 되지만
    밝은 지표·연무 장면에서도 참이 되므로 반사도 계산에는 쓰지 않는다(같은 취득의 원 기준선 항목과 비교해 확인했다).
    red_dn20, scl20 = 같은 장면의 셀 창 배열 목록. 반환 (offset 여부, 1 백분위수, 화소 수). 유효 화소가 없으면 (False, NaN, 0)."""
    vals = []
    for r_, s_ in zip(red_dn20, scl20):
        r_ = np.asarray(r_, np.float32); s_ = np.asarray(s_)
        h = min(r_.shape[0], s_.shape[0]); w = min(r_.shape[1], s_.shape[1])
        m = np.isin(s_[:h, :w], S2_SCL_VALID) & np.isfinite(r_[:h, :w]) & (r_[:h, :w] > 0)
        if m.any():
            vals.append(r_[:h, :w][m])
    if not vals:
        return False, float("nan"), 0
    v = np.concatenate(vals)
    p = float(np.percentile(v, DN_P))
    return bool(p >= DN_OFFSET), p, int(v.size)


def reflectance(dn, offset=False):
    """DN → 반사도(DN 0 = 자료 없음 → NaN). offset 이 참이면(DN 에 +1000 이 남은 장면) −0.1 을 더한다."""
    v = np.asarray(dn, np.float32) * np.float32(S2_SCALE)
    if offset:
        v = v + np.float32(S2_OFFSET)
    v[np.asarray(dn) == 0] = np.nan
    return v


def item_date(item):
    return str(item["properties"].get("datetime") or "")[:10]


def dedupe(items):
    """같은 타일·같은 취득일(UTC)의 중복 항목은 s2:sequence 가 큰 것 하나(같으면 운량이 작은 것, 그다음 항목 id 순)."""
    best = {}
    for it in items:
        p = it["properties"]
        k = (p.get("grid:code"), item_date(it))
        key = (-int(p.get("s2:sequence") or 0), float(p.get("eo:cloud_cover") if p.get("eo:cloud_cover") is not None else 1e9), str(it["id"]))
        if k not in best or key < best[k][0]:
            best[k] = (key, it)
    return [v[1] for v in best.values()]


def tile_frames(items):
    """타일마다 (EPSG, 20 m 원점 X0·Y0, 폭·높이 m). scl 자산의 proj:transform·proj:shape(같은 타일은 같다)."""
    out = {}
    for it in items:
        t = it["properties"].get("grid:code")
        sc = it["assets"].get("scl") or {}
        tr, sh = sc.get("proj:transform"), sc.get("proj:shape")
        ep = it["properties"].get("proj:epsg")
        if t in out or not tr or not sh or not ep:
            continue
        out[t] = dict(epsg=int(ep), x0=float(tr[2]), y0=float(tr[5]), res=float(tr[0]), width_m=float(tr[0]) * int(sh[1]),
                      height_m=abs(float(tr[4])) * int(sh[0]))
    return out


def _transformer(epsg):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)


def cell_corners(cells):
    """셀 꼭짓점 (경도, 위도) 배열 (셀 수, 4)."""
    lon = np.stack([cells.w.values, cells.e.values, cells.e.values, cells.w.values], 1)
    lat = np.stack([cells.s.values, cells.s.values, cells.n.values, cells.n.values], 1)
    return lon, lat


def tile_margin(cells, fr):
    """셀 꼭짓점 네 개에서 타일 경계까지 최소 거리(m, 안쪽이 양수)."""
    lon, lat = cell_corners(cells)
    x, y = _transformer(fr["epsg"]).transform(lon.ravel(), lat.ravel())
    x = np.asarray(x).reshape(lon.shape); y = np.asarray(y).reshape(lon.shape)
    x0, y1 = fr["x0"], fr["y0"]
    x1, y0 = x0 + fr["width_m"], y1 - fr["height_m"]
    m = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
    return m.min(axis=1)


def assign_tiles(cells, frames, cand=None):
    """셀마다 꼭짓점 최소 여유가 가장 큰 타일. cand = {셀: 후보 타일 집합}(항목 geometry 가 셀 중심을 덮는 타일). 반환 (타일, 여유) 배열."""
    names = sorted(frames)
    M = np.full((len(cells), len(names)), -np.inf)
    for j, t in enumerate(names):
        M[:, j] = tile_margin(cells, frames[t])
    if cand is not None:
        for i, c in enumerate(cells.cell.values):
            allowed = cand.get(c, set())
            for j, t in enumerate(names):
                if t not in allowed:
                    M[i, j] = -np.inf
    jbest = np.argmax(M, axis=1)
    tiles = np.array([names[j] if np.isfinite(M[i, j]) else "" for i, j in enumerate(jbest)], dtype=object)
    return tiles, M[np.arange(len(cells)), jbest]


def coverage_r(item, lon, lat):
    """항목 geometry 안에 드는 점의 비율(셀 중심)."""
    import shapely
    from shapely.geometry import shape
    if not len(lon):
        return float("nan")
    g = shape(item["geometry"])
    return float(np.mean(shapely.contains_xy(g, np.asarray(lon, float), np.asarray(lat, float))))


def select_scenes(items, lon, lat, k=S2_N_SCENES, r_min=S2_R_MIN):
    """한 타일의 장면 선택(등록 3절 (b)): r ≥ 0.9 먼저, 운량, 자료 없음 비율, 시각 오름차순으로 앞의 k장면. 반환 [(항목, r)]."""
    sc = []
    for it in items:
        p = it["properties"]
        r = coverage_r(it, lon, lat)
        key = (0 if r >= r_min else 1, float(p.get("eo:cloud_cover") or 0.0), float(p.get("s2:nodata_pixel_percentage") or 0.0),
               str(p.get("datetime") or ""), str(it["id"]))
        sc.append((key, it, r))
    sc.sort(key=lambda v: v[0])
    return [(it, r) for _, it, r in sc[:int(k)]]


def cmd_s2_search(a, pts, cells):
    t0 = time.time()
    d = check_out(D_S2 / "stac")
    d.mkdir(parents=True, exist_ok=True)
    points = [[round(float(x), 6), round(float(y), 6)] for x, y in zip(cells.clon, cells.clat)]
    res, reqs, bodies = {}, {}, {}

    def one(y):
        p = d / f"items_{y}.jsonl.gz"
        mp = d / f"items_{y}_meta.json"
        if p.exists() and mp.exists() and not a.refresh:
            m = json.loads(mp.read_text())
            if m.get("points_sha256") == sha256_bytes(json.dumps(points).encode()):
                with gzip.open(p, "rt") as fh:
                    return y, [json.loads(l_) for l_ in fh], m.get("n_requests", 0), m.get("body")
        its, n_req, body = stac_search_year(y, points)
        with gzip.open(p, "wt") as fh:
            for it in its:
                fh.write(json.dumps(it) + "\n")
        mp.write_text(json.dumps(dict(year=y, url=STAC_URL, collection=STAC_COLLECTION, n_items=len(its), n_requests=n_req, fetched=now_kst(),
                                      points_sha256=sha256_bytes(json.dumps(points).encode()), file_sha256=R.sha256_file(p),
                                      body={k: v for k, v in body.items() if k != "intersects"}, n_points=len(points)), ensure_ascii=False, indent=1))
        return y, its, n_req, body
    with ThreadPoolExecutor(max_workers=min(len(S2_YEARS), max(1, int(a.threads)))) as ex:
        for y, its, n_req, body in ex.map(one, S2_YEARS):
            res[y], reqs[y], bodies[y] = its, n_req, body
    allit = [it for y in S2_YEARS for it in res[y]]
    n_all = len(allit)
    cog = [it for it in allit if cog_ok(it)]
    # 같은 항목 id 가 해 사이에 겹치지 않는다(7–8월). 그래도 id 로 한 번 거른다
    cog = list({it["id"]: it for it in cog}.values())
    dd = dedupe(cog)
    frames = tile_frames(dd)
    # 셀 중심을 덮는 항목이 있는 타일만 그 셀의 후보다
    import shapely
    from shapely.geometry import shape
    cand = {c: set() for c in cells.cell.values}
    geoms = {}
    for it in dd:
        t = it["properties"].get("grid:code")
        geoms.setdefault(t, []).append(shape(it["geometry"]))
    for t, gs in geoms.items():
        u = shapely.union_all(gs)
        inside = shapely.contains_xy(u, cells.clon.values, cells.clat.values)
        for c in cells.cell.values[inside]:
            cand[c].add(t)
    tiles, margin = assign_tiles(cells, frames, cand)
    cells = cells.assign(s2_tile=tiles, s2_tile_margin_m=margin)
    sel_rows = []
    for t in sorted(set(tiles) - {""}):
        m = cells.s2_tile.values == t
        cand_items = [it for it in dd if it["properties"].get("grid:code") == t]
        chosen = select_scenes(cand_items, cells.clon.values[m], cells.clat.values[m])
        for rank, (it, r) in enumerate(chosen, 1):
            p = it["properties"]
            sel_rows.append(dict(tile=t, rank=rank, item=it["id"], datetime=p.get("datetime"), platform=p.get("platform"),
                                 cloud=p.get("eo:cloud_cover"), nodata_pct=p.get("s2:nodata_pixel_percentage"), r=r, n_cells=int(m.sum()),
                                 n_candidates=len(cand_items), sequence=p.get("s2:sequence"), baseline=p.get("s2:processing_baseline"),
                                 boa_offset_applied=p.get("earthsearch:boa_offset_applied"), meta_offset_flag=meta_offset_flag(p), epsg=p.get("proj:epsg"),
                                 **{f"href_{an}": it["assets"][an]["href"] for an in S2_ASSETS}))
    sel = pd.DataFrame(sel_rows)
    check_out(PARTS)
    PARTS.mkdir(parents=True, exist_ok=True)
    sp = PARTS / "xm_s2_selection_v1.csv"
    sel.to_csv(sp, index=False)
    cp = PARTS / "xm_s2_cells_v1.csv"
    cells[["cell", "target", "ky", "kx", "s2_tile", "s2_tile_margin_m"]].to_csv(cp, index=False)
    meta = dict(created=now_kst(), addendum=ADDENDUM, addendum_commit=ADDENDUM_COMMIT, n_items_all=n_all, n_items_cog=len(cog), n_items_dedup=len(dd),
                n_requests=reqs, n_items_by_year={y: len(v) for y, v in res.items()}, n_tiles=len(frames), n_tiles_assigned=int(len(set(tiles) - {""})),
                n_cells_unassigned=int((tiles == "").sum()), n_cells_margin_neg=int((margin < 0).sum()), selection=rel(sp),
                selection_sha256=R.sha256_file(sp), cells_tile=rel(cp), cells_tile_sha256=R.sha256_file(cp),
                n_scenes_selected=int(len(sel)), n_tiles_lt6=int((sel.groupby("tile").size() < S2_N_SCENES).sum()) if len(sel) else 0,
                n_selected_r_lt_min=int((sel.r < S2_R_MIN).sum()) if len(sel) else 0, seconds=round(time.time() - t0, 1),
                rule=dict(years=list(S2_YEARS), months=S2_MONTHS, cloud_lt=S2_CLOUD_MAX, cog_only="네 자산이 HTTPS .tif",
                          dedupe="(grid:code, UTC 날짜)마다 s2:sequence 최대, 같으면 운량 최소, 그다음 id", assign="꼭짓점 최소 여유 최대 타일(셀 중심을 덮는 항목이 있는 타일 가운데)",
                          select=f"r ≥ {S2_R_MIN} 먼저, 운량, 자료 없음 비율, 시각 순 {S2_N_SCENES}장면"))
    (PARTS / "xm_s2_selection_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"[s2-search] 항목 {n_all}(COG {len(cog)}, 중복 제거 {len(dd)}) · 요청 {sum(reqs.values())} · 타일 {meta['n_tiles_assigned']} · "
        f"배정 없는 셀 {meta['n_cells_unassigned']} · 선택 장면 {len(sel)}(6장면 미만 타일 {meta['n_tiles_lt6']}, r < {S2_R_MIN} {meta['n_selected_r_lt_min']}) · "
        f"{meta['seconds']}s · sha256 {meta['selection_sha256'][:16]}")
    return meta


# ================================================================ Sentinel-2: 창 읽기
def utm_window(cells_sub, fr, pad=1):
    """셀마다 20 m 창 (row0, col0, h, w)(타일 격자, 여유 pad 화소, 격자 안으로 자른다)."""
    lon, lat = cell_corners(cells_sub)
    x, y = _transformer(fr["epsg"]).transform(lon.ravel(), lat.ravel())
    x = np.asarray(x).reshape(lon.shape); y = np.asarray(y).reshape(lon.shape)
    res = S2_RES
    nW, nH = int(round(fr["width_m"] / res)), int(round(fr["height_m"] / res))
    out = []
    for i in range(len(cells_sub)):
        c0 = max(0, int(math.floor((x[i].min() - fr["x0"]) / res)) - pad)
        c1 = min(nW, int(math.ceil((x[i].max() - fr["x0"]) / res)) + pad)
        r0 = max(0, int(math.floor((fr["y0"] - y[i].max()) / res)) - pad)
        r1 = min(nH, int(math.ceil((fr["y0"] - y[i].min()) / res)) + pad)
        out.append((r0, c0, max(0, r1 - r0), max(0, c1 - c0)))
    return out


def blocks_touched(windows, block_shape):
    """창 목록이 닿는 내부 블록 (i, j) 집합."""
    bh, bw = block_shape
    s = set()
    for r0, c0, h, w in windows:
        if h <= 0 or w <= 0:
            continue
        for i in range(r0 // bh, (r0 + h - 1) // bh + 1):
            for j in range(c0 // bw, (c0 + w - 1) // bw + 1):
                s.add((i, j))
    return s


class Budget:
    """내려받기 상한(바이트). 읽은 COG 내부 블록 바이트와 머리 읽기를 더한다."""

    def __init__(self, cap_bytes, used=0):
        self.cap, self.used, self.stopped = int(cap_bytes), int(used), False

    def add(self, b):
        with _LOCK:
            self.used += int(b)
            if self.used >= self.cap:
                self.stopped = True

    def ok(self):
        with _LOCK:
            return not self.stopped and self.used < self.cap


def window_cache_path(tile, item_id):
    return D_S2 / "windows" / str(tile).replace("MGRS-", "") / f"{item_id}.npz"


def fetch_task(task, budget, tries=3):
    """한 (타일, 장면): 네 자산을 /vsicurl 로 열어 셀 창을 읽고 npz 로 둔다. 반환 기록 dict."""
    import rasterio
    from rasterio.windows import Window
    tile, item_id, hrefs, fr, cells_sub = task
    p = window_cache_path(tile, item_id)
    cell_list = list(cells_sub.cell.values)
    if p.exists():
        try:
            with np.load(p, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
            if m.get("cells") == cell_list:
                return dict(tile=tile, item=item_id, status="cached", bytes=0, sec=0.0, file=rel(p))
        except Exception:                                               # noqa: BLE001
            pass
    if not budget.ok():
        return dict(tile=tile, item=item_id, status="상한 도달", bytes=0, sec=0.0)
    t0 = time.time()
    wins20 = utm_window(cells_sub, fr)
    err = ""
    for k in range(int(tries)):
        try:
            arrays, nbytes, info = {}, 0, {}
            with rasterio.Env(**RIO_ENV):
                for an in S2_ASSETS:
                    with rasterio.open("/vsicurl/" + hrefs[an]) as src:
                        f10 = an in ("red", "nir")
                        fac = 2 if f10 else 1
                        exp_res = 10.0 if f10 else 20.0
                        if abs(src.res[0] - exp_res) > 1e-6 or abs(src.transform.c - fr["x0"]) > 1e-6 or abs(src.transform.f - fr["y0"]) > 1e-6:
                            raise RuntimeError(f"{an} 격자가 타일 틀과 다르다({src.res}, {src.transform.c}, {src.transform.f})")
                        ws = [(r0 * fac, c0 * fac, h * fac, w * fac) for r0, c0, h, w in wins20]
                        bl = blocks_touched(ws, src.block_shapes[0])
                        nb = 0
                        for (i, j) in bl:
                            try:
                                nb += int(src.block_size(1, i, j))
                            except Exception:                           # noqa: BLE001
                                pass
                        nbytes += nb + int(GDAL_ENV["GDAL_INGESTED_BYTES_AT_OPEN"])
                        info[an] = dict(blocks=len(bl), block_bytes=nb, block_shape=list(src.block_shapes[0]))
                        for ci, (r0, c0, h, w) in zip(cell_list, ws):
                            arrays[f"{ci}|{an}"] = src.read(1, window=Window(c0, r0, w, h)) if h > 0 and w > 0 else np.zeros((0, 0), np.uint16)
            budget.add(nbytes)
            meta = dict(tile=tile, item=item_id, cells=cell_list, wins20=[list(map(int, w_)) for w_ in wins20], hrefs=hrefs, frame=fr,
                        bytes=nbytes, info=info, fetched=now_kst(), sec=round(time.time() - t0, 2))
            d = check_out(p.parent)
            d.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".tmp{os.getpid()}_{threading.get_ident()}.npz")
            np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **{k_.replace("|", "__"): v for k_, v in arrays.items()})
            os.replace(tmp, p)
            return dict(tile=tile, item=item_id, status="ok", bytes=nbytes, sec=meta["sec"], tries=k + 1, file=rel(p))
        except Exception as e:                                           # noqa: BLE001
            err = repr(e)[:300]
            time.sleep(5 * (k + 1))
    return dict(tile=tile, item=item_id, status=f"실패({err})", bytes=0, sec=round(time.time() - t0, 2), tries=int(tries))


def load_selection():
    sp, cp = PARTS / "xm_s2_selection_v1.csv", PARTS / "xm_s2_cells_v1.csv"
    if not sp.exists() or not cp.exists():
        raise SystemExit("[거부] s2-search 를 먼저 한다")
    return pd.read_csv(sp), pd.read_csv(cp)


def frames_from_cache():
    """STAC 캐시에서 타일 틀(scl 자산의 proj)."""
    its = []
    for y in S2_YEARS:
        p = D_S2 / "stac" / f"items_{y}.jsonl.gz"
        if p.exists():
            with gzip.open(p, "rt") as fh:
                its += [json.loads(l_) for l_ in fh]
    return tile_frames([it for it in its if cog_ok(it)])


def cmd_s2_fetch(a, pts, cells):
    t0 = time.time()
    sel, ct = load_selection()
    frames = frames_from_cache()
    cells = cells.merge(ct[["cell", "s2_tile"]], on="cell", how="left")
    log_p = PARTS / "xm_s2_fetch_log_v1.csv"
    prev = pd.read_csv(log_p) if log_p.exists() else pd.DataFrame(columns=["tile", "item", "status", "bytes"])
    used0 = int(pd.to_numeric(prev.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(prev) else 0
    budget = Budget(float(a.cap_gb) * 1e9, used0)
    tasks = []
    for r in sel.itertuples():
        sub = cells[cells.s2_tile == r.tile].sort_values("cell").reset_index(drop=True)
        hrefs = {an: getattr(r, f"href_{an}") for an in S2_ASSETS}
        tasks.append((r.tile, r.item, hrefs, frames[r.tile], sub))
    log(f"[s2-fetch] 장면 작업 {len(tasks)} · 타일 {sel.tile.nunique()} · 스레드 {a.threads} · 상한 {a.cap_gb} GB(이전 기록 {used0 / 1e9:.2f} GB)")
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(a.threads), 8))) as ex:
        futs = [ex.submit(fetch_task, t_, budget) for t_ in tasks]
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(tasks):
                st = pd.Series([r_["status"][:2] for r_ in rows]).value_counts().to_dict()
                log(f"[s2-fetch] {done}/{len(tasks)} · 상태 {st} · 누적 {budget.used / 1e9:.2f} GB · {time.time() - t0:.0f}s")
    new = pd.DataFrame(rows)
    allr = pd.concat([prev, new], ignore_index=True) if len(prev) else new
    allr = allr.drop_duplicates(["tile", "item"], keep="last") if len(allr) else allr
    check_out(PARTS)
    # 바이트 열은 실제로 받은 작업만(캐시 재사용은 0). 이전 기록의 바이트를 지우지 않도록 상태별로 합친다
    if len(prev):
        keep_b = prev.set_index(["tile", "item"]).bytes
        allr["bytes"] = [max(float(b or 0), float(keep_b.get((t, i), 0) or 0)) for t, i, b in zip(allr.tile, allr.item, allr.bytes)]
    allr.to_csv(log_p, index=False)
    n_fail = int(allr.status.astype(str).str.startswith("실패").sum())
    n_cap = int((allr.status == "상한 도달").sum())
    meta = dict(created=now_kst(), n_tasks=len(tasks), status=allr.status.astype(str).str[:6].value_counts().to_dict(), n_fail=n_fail, n_cap=n_cap,
                bytes_total=int(pd.to_numeric(allr.bytes, errors="coerce").fillna(0).sum()), cap_gb=float(a.cap_gb), seconds=round(time.time() - t0, 1),
                threads=int(a.threads), log=rel(log_p), gdal_env={k: v for k, v in GDAL_ENV.items() if k != "CPL_TMPDIR"},
                byte_rule="읽은 COG 내부 블록의 압축 바이트 합(src.block_size) + 자산마다 머리 읽기 65,536 B(근사, 범위 병합 여유 미포함)")
    (PARTS / "xm_s2_fetch_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"[s2-fetch] 완료 · 실패 {n_fail} · 상한 도달 {n_cap} · 내려받기 {meta['bytes_total'] / 1e9:.2f} GB · {meta['seconds']}s · 최대 RSS {max_rss_mb()} MB")
    return meta


# ================================================================ Sentinel-2: 셀 지수
def agg2x2(a10):
    """10 m 배열 → 20 m 2 × 2 평균(하나라도 NaN 이면 NaN). 모양은 (2h, 2w) → (h, w)."""
    a10 = np.asarray(a10, np.float32)
    h, w = a10.shape[0] // 2, a10.shape[1] // 2
    b = a10[:2 * h, :2 * w].reshape(h, 2, w, 2)
    return b.mean(axis=(1, 3))


def scene_indices(red10, nir10, swir20, scl20, offset=False):
    """한 장면의 20 m NDVI·NDMI(유효 관측만, 나머지 NaN). offset = 그 장면의 DN 에 BOA_ADD_OFFSET 이 남아 있는가(dn_offset_check)."""
    red = agg2x2(reflectance(red10, offset))
    nir = agg2x2(reflectance(nir10, offset))
    sw = reflectance(swir20, offset)
    h = min(red.shape[0], sw.shape[0], scl20.shape[0]); w = min(red.shape[1], sw.shape[1], scl20.shape[1])
    red, nir, sw, scl = red[:h, :w], nir[:h, :w], sw[:h, :w], np.asarray(scl20)[:h, :w]
    ok = np.isin(scl, S2_SCL_VALID) & np.isfinite(red) & np.isfinite(nir) & np.isfinite(sw) & (red > 0) & (nir > 0) & (sw > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(ok, (nir - red) / (nir + red), np.nan).astype(np.float32)
        ndmi = np.where(ok, (nir - sw) / (nir + sw), np.nan).astype(np.float32)
    return ndvi, ndmi


def composite(stack):
    """장면 축 중앙값(유효 관측만). 관측이 없는 화소는 NaN."""
    st = np.asarray(stack, np.float32)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(st, axis=0) if st.size else st


def cell_s2_values(ndvi_c, ndmi_c, inmask, min_frac=MIN_DATA_FRAC, min_px=S2_MIN_PX):
    """셀 값: 셀 안 합성 NDVI 중앙값·NDMI 중앙값·NDVI 표준편차(ddof 0). 합성 값이 있는 화소 비율 < min_frac 또는 수 < min_px 이면 결측."""
    m = np.asarray(inmask, bool)
    n_in = int(m.sum())
    v = np.asarray(ndvi_c)[m]
    q = np.asarray(ndmi_c)[m]
    ok = np.isfinite(v) & np.isfinite(q)
    n_ok = int(ok.sum())
    frac = n_ok / n_in if n_in else float("nan")
    out = dict(s2_ndvi_med=np.nan, s2_ndmi_med=np.nan, s2_ndvi_sd=np.nan, s2_valid_frac=frac, s2_n_px=n_in, s2_n_valid_px=n_ok)
    if n_in and n_ok >= min_px and frac >= min_frac:
        out.update(s2_ndvi_med=float(np.median(v[ok])), s2_ndmi_med=float(np.median(q[ok])), s2_ndvi_sd=float(np.std(v[ok], ddof=0)))
    return out


def utm_centers_lonlat(fr, r0, c0, h, w):
    """20 m 창 화소 중심의 (경도, 위도)."""
    from pyproj import Transformer
    xs = fr["x0"] + (c0 + np.arange(w) + 0.5) * S2_RES
    ys = fr["y0"] - (r0 + np.arange(h) + 0.5) * S2_RES
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(f"EPSG:{int(fr['epsg'])}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(X.ravel(), Y.ravel())
    return np.asarray(lon).reshape(h, w), np.asarray(lat).reshape(h, w)


def cmd_s2(a, pts, cells):
    t0 = time.time()
    sel, ct = load_selection()
    frames = frames_from_cache()
    props_by_item = {}
    for y in S2_YEARS:
        p = D_S2 / "stac" / f"items_{y}.jsonl.gz"
        if p.exists():
            with gzip.open(p, "rt") as fh:
                for l_ in fh:
                    it = json.loads(l_)
                    props_by_item[it["id"]] = it["properties"]
    cells = cells.merge(ct[["cell", "s2_tile"]], on="cell", how="left")
    rows, diag = [], []
    for t, sub in cells.groupby("s2_tile"):
        if not isinstance(t, str) or not t:
            continue
        fr = frames[t]
        items = sel[sel.tile == t].sort_values("rank").item.tolist()
        zs = []
        for it in items:
            p = window_cache_path(t, it)
            if p.exists():
                z = np.load(p, allow_pickle=False)
                m = json.loads(str(z["meta"]))
                reds = [agg2x2(np.where(z[f"{c_}__red"] == 0, np.nan, z[f"{c_}__red"].astype(np.float32))) for c_ in m["cells"]
                        if z[f"{c_}__red"].size]
                scls = [z[f"{c_}__scl"] for c_ in m["cells"] if z[f"{c_}__red"].size]
                p01_hi, p01, npx = dn_offset_check(reds, scls)
                diag.append(dict(tile=t, item=it, red_dn_p01=p01, n_px=npx, red_dn_p01_ge_1000=p01_hi, offset_applied=False,
                                 meta_offset_flag=meta_offset_flag(props_by_item.get(it, {})),
                                 baseline=props_by_item.get(it, {}).get("s2:processing_baseline"),
                                 boa_offset_applied=props_by_item.get(it, {}).get("earthsearch:boa_offset_applied")))
                zs.append((it, z, m, False))                              # 등록 정정 1: 모든 장면 offset 없음
        for c in sub.sort_values("cell").itertuples():
            stack_v, stack_q, n_sc, win = [], [], 0, None
            for it, z, m, off in zs:
                if c.cell not in m["cells"]:
                    continue
                k = m["cells"].index(c.cell)
                r0, c0, h, w = m["wins20"][k]
                if h <= 0 or w <= 0:
                    continue
                win = (r0, c0, h, w)
                nd, nm = scene_indices(z[f"{c.cell}__red"], z[f"{c.cell}__nir"], z[f"{c.cell}__swir16"], z[f"{c.cell}__scl"], off)
                stack_v.append(nd); stack_q.append(nm)
            if win is None:
                rows.append(dict(cell=c.cell, s2_tile=t, s2_n_scenes=0, **cell_s2_values(np.zeros(0), np.zeros(0), np.zeros(0, bool))))
                continue
            r0, c0, h, w = win
            hh = min(min(s_.shape[0] for s_ in stack_v), h); ww = min(min(s_.shape[1] for s_ in stack_v), w)
            sv = np.stack([s_[:hh, :ww] for s_ in stack_v]); sq = np.stack([s_[:hh, :ww] for s_ in stack_q])
            lon, lat = utm_centers_lonlat(fr, r0, c0, hh, ww)
            inm = in_cell(lon, lat, (c.w, c.s, c.e, c.n))
            n_sc = int(sum(np.isfinite(s_[inm]).any() for s_ in sv))
            vals = cell_s2_values(composite(sv), composite(sq), inm)
            rows.append(dict(cell=c.cell, s2_tile=t, s2_n_scenes=n_sc, **vals))
        for _, z, _, _ in zs:
            z.close()
    have = {r_["cell"] for r_ in rows}
    for c in cells.itertuples():
        if c.cell not in have:
            rows.append(dict(cell=c.cell, s2_tile="", s2_n_scenes=0, **cell_s2_values(np.zeros(0), np.zeros(0), np.zeros(0, bool))))
    df = pd.DataFrame(rows)
    dg = pd.DataFrame(diag)
    dp = PARTS / "xm_s2_offset_check_v1.csv"
    check_out(PARTS)
    dg.to_csv(dp, index=False)
    offset_summary = dict(file=rel(dp), sha256=R.sha256_file(dp), n_scenes=int(len(dg)), n_offset_applied=int(dg.offset_applied.sum()) if len(dg) else 0,
                          n_p01_ge_1000=int(dg.red_dn_p01_ge_1000.sum()) if len(dg) else 0, n_meta_flag=int(dg.meta_offset_flag.sum()) if len(dg) else 0,
                          red_dn_p01_max=float(np.nanmax(dg.red_dn_p01)) if len(dg) and np.isfinite(dg.red_dn_p01).any() else None,
                          red_dn_p01_median=float(np.nanmedian(dg.red_dn_p01)) if len(dg) and np.isfinite(dg.red_dn_p01).any() else None)
    log(f"[s2] offset 진단(기록만): 장면 {offset_summary['n_scenes']}, offset 적용 {offset_summary['n_offset_applied']}, "
        f"red DN 1 백분위수 ≥ 1,000 장면 {offset_summary['n_p01_ge_1000']}, 메타 표지 {offset_summary['n_meta_flag']}, "
        f"red DN 1 백분위수 최대 {offset_summary['red_dn_p01_max']}")
    rule = dict(reflectance="DN × 0.0001, 모든 장면(등록 정정 1: 이 모음의 COG DN 에는 BOA_ADD_OFFSET 이 없다). red DN 1 백분위수는 진단 기록", grid20="red·nir 2 × 2 평균",
                valid="SCL 4·5, 세 반사도 > 0", composite="화소별 유효 관측 중앙값", cell="셀 안 중앙값·표준편차(ddof 0)",
                min_frac=MIN_DATA_FRAC, min_px=S2_MIN_PX)
    src = dict(selection=rel(PARTS / "xm_s2_selection_v1.csv"), selection_sha256=R.sha256_file(PARTS / "xm_s2_selection_v1.csv"),
               n_items_selected=int(len(sel)), n_tiles=int(sel.tile.nunique()), windows=rel(D_S2 / "windows"))
    return write_part("s2", df, src, dict(rule=rule, offset_check=offset_summary, seconds=round(time.time() - t0, 1)))


# ================================================================ 조각·결합
def write_part(group, df, sources, extra=None):
    check_out(PARTS)
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"xm_{group}_v1.csv"
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df.sort_values("cell").to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, p)
    meta = dict(group=group, created=now_kst(), addendum=ADDENDUM, addendum_commit=ADDENDUM_COMMIT, n_cells=int(len(df)), columns=list(df.columns),
                sha256=R.sha256_file(p), sources=sources, max_rss_mb=max_rss_mb(), script=rel(Path(__file__)))
    meta.update(extra or {})
    (PARTS / f"xm_{group}_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    cols = [c for c in df.columns if c in WC_COLS + S2_COLS]
    fin = float(np.isfinite(df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1).mean()) if cols and len(df) else float("nan")
    log(f"[{group}] 조각 {rel(p)}: 셀 {len(df)}, 모든 열 유한 셀 비율 {fin:.4f}, sha256 {meta['sha256'][:16]}")
    return meta


def group_decisions(feat, min_frac=R.FINITE_MIN):
    """대상마다 군(WC, S2)의 유한값 비율(라벨 행 × 군 열 칸)과 포함 여부(XE 2.5 와 같은 90 % 규칙)."""
    out = {}
    for tg in R.TARGETS:
        sub = feat[feat.target == tg]
        out[tg] = {}
        for g, cols in XM_GROUPS.items():
            fr = R.group_finite_fraction(sub, cols)
            inc = bool(np.isfinite(fr) and fr >= min_frac)
            out[tg][g] = dict(finite_frac=None if not np.isfinite(fr) else round(float(fr), 6), included=inc, n_rows=int(len(sub)),
                              reason="" if inc else ("행 없음" if not len(sub) else f"유한값 비율 {fr:.3f} < {min_frac:.2f}"))
    return out


def coverage_table(feat, cells_df):
    """대상·군별 덮음 비율(군 열이 모두 유한한 라벨 행과 셀의 비율, %)."""
    out = {}
    for tg in R.TARGETS:
        f = feat[feat.target == tg]
        c = cells_df[cells_df.target == tg]
        out[tg] = {}
        for g, cols in XM_GROUPS.items():
            fr = np.isfinite(f[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1) if len(f) else np.zeros(0, bool)
            fc = np.isfinite(c[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1) if len(c) else np.zeros(0, bool)
            out[tg][g] = dict(rows_pct=round(100.0 * float(fr.mean()), 2) if len(fr) else None, n_rows=int(len(fr)),
                              cells_pct=round(100.0 * float(fc.mean()), 2) if len(fc) else None, n_cells=int(len(fc)))
    return out


def cmd_assemble(a, pts, cells):
    t0 = time.time()
    _, lab = cell_table(pts)
    parts, metas = {}, {}
    for g in ("wc", "s2"):
        p, pm = PARTS / f"xm_{g}_v1.csv", PARTS / f"xm_{g}_v1_meta.json"
        if not p.exists():
            continue
        m = json.loads(pm.read_text()) if pm.exists() else {}
        if m.get("sha256") and m["sha256"] != R.sha256_file(p):
            raise SystemExit(f"[assemble] 조각 {p.name} 의 sha256 이 메타와 다르다")
        parts[g], metas[g] = pd.read_csv(p), m
    cf = cells[["cell", "target", "ky", "kx"]].copy()
    for g, d in parts.items():
        cf = cf.merge(d[[c for c in d.columns if c not in ("target",)]], on="cell", how="left")
    for c in WC_COLS + S2_COLS + AUX_COLS:
        if c not in cf.columns:
            cf[c] = np.nan if c != "s2_tile" else ""
    feat = lab.merge(cf.drop(columns=["target", "ky", "kx"]), on="cell", how="left")
    feat = feat.rename(columns={"ky": "cell_ky", "kx": "cell_kx"})
    order = ["loc_id", "target", "cell", "cell_ky", "cell_kx"] + WC_COLS + S2_COLS + AUX_COLS
    feat = feat[order].sort_values("loc_id").reset_index(drop=True)
    check_out(INP)
    INP.mkdir(parents=True, exist_ok=True)
    fp = INP / f"{FEAT_NAME}.csv"
    tmp = fp.with_name(fp.name + f".tmp{os.getpid()}")
    feat.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, fp)
    re_read = pd.read_csv(fp)
    dec = group_decisions(re_read)
    cov = coverage_table(re_read, cf)
    gsha = {g: R.cols_sha(re_read, cols) for g, cols in XM_GROUPS.items()}
    sel_meta = json.loads((PARTS / "xm_s2_selection_v1_meta.json").read_text()) if (PARTS / "xm_s2_selection_v1_meta.json").exists() else {}
    fetch_meta = json.loads((PARTS / "xm_s2_fetch_v1_meta.json").read_text()) if (PARTS / "xm_s2_fetch_v1_meta.json").exists() else {}
    timing = {g: metas.get(g, {}).get("seconds") for g in metas}
    timing.update(s2_search=sel_meta.get("seconds"), s2_fetch=fetch_meta.get("seconds"))
    meta = dict(created=now_kst(), addendum=ADDENDUM, addendum_commit=ADDENDUM_COMMIT, file=rel(fp), sha256=R.sha256_file(fp), n_rows=int(len(feat)),
                n_by_target=feat.target.value_counts().to_dict(), n_cells_by_target=cf.target.value_counts().to_dict(), groups=XM_GROUPS,
                aux_cols=AUX_COLS, group_decisions=dec, coverage=cov, group_cols_sha256=gsha, finite_min=R.FINITE_MIN,
                finite_rule="군의 유한값 비율 = 대상 라벨 행 × 군 열 칸 가운데 유한한 칸의 비율. 90 % 미만이면 그 군을 그 대상에서 뺀다(등록 3절 (c))",
                parts={g: dict(path=rel(PARTS / f"xm_{g}_v1.csv"), sha256=m.get("sha256"), created=m.get("created"), sources=m.get("sources"),
                               rule=m.get("rule")) for g, m in metas.items()},
                s2_selection=sel_meta, s2_fetch=fetch_meta, timing_s=timing, licence=LICENCE, reads=TRACE_READS, max_rss_mb=max_rss_mb(),
                cell_rule="LG 6B.3 셀: ky = floor(lat/0.009), φ = (ky + 0.5)·0.009°, kx = floor(lon·cos φ/0.009). 화소 중심 포함",
                seconds=round(time.time() - t0, 1))
    mp = INP / f"{FEAT_NAME}_meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"[assemble] {fp.name}: 행 {len(feat)}, sha256 {meta['sha256'][:16]}")
    for tg in R.TARGETS:
        s_ = ", ".join(f"{g} {'포함' if v['included'] else '제외'}({v['finite_frac']})" for g, v in dec[tg].items())
        c_ = ", ".join(f"{g} 행 {v['rows_pct']} %·셀 {v['cells_pct']} %" for g, v in cov[tg].items())
        log(f"[assemble] {tg}: {s_} · 덮음 {c_}")
    return meta


def cmd_cells(a, pts, cells):
    check_out(PARTS)
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / "xm_cells_v1.csv"
    cells.to_csv(p, index=False, float_format="%.10g")
    log(f"[cells] 셀 {len(cells)}({cells.target.value_counts().to_dict()}) → {rel(p)}, sha256 {R.sha256_file(p)[:16]}")
    return dict(path=rel(p))


CMDS = {"cells": cmd_cells, "wc": cmd_wc, "s2-search": cmd_s2_search, "s2-fetch": cmd_s2_fetch, "s2": cmd_s2, "assemble": cmd_assemble}


def main(argv=None):
    ap = argparse.ArgumentParser(description="XM 피복·식생 특징(등록 3절)")
    ap.add_argument("cmds", nargs="+", choices=list(CMDS))
    ap.add_argument("--threads", type=int, default=6, help="입출력 스레드(상한 8)")
    ap.add_argument("--cap-gb", type=float, default=CAP_GB_DEFAULT, help="Sentinel-2 내려받기 상한(GB)")
    ap.add_argument("--mem-gb", type=float, default=14.0, help="프로세스 주소 공간 상한(GB, 0 = 두지 않음)")
    ap.add_argument("--min-free-gb", type=float, default=16.0, help="시작 전 가용 메모리 하한(GB)")
    ap.add_argument("--refresh", action="store_true", help="STAC 캐시를 다시 받는다")
    a = ap.parse_args(argv)
    a.threads = max(1, min(int(a.threads), 8))
    g = mem_available_gb()
    if np.isfinite(g) and g < a.min_free_gb:
        raise SystemExit(f"[대기] 가용 메모리 {g:.1f} GB 가 하한 {a.min_free_gb} GB 아래다. 나중에 다시 한다")
    set_local_limits(a.mem_gb)
    for d in (RAW_XM / "tmp", PARTS):
        check_out(d)
        d.mkdir(parents=True, exist_ok=True)
    pts = read_points()
    cells, _ = cell_table(pts)
    log(f"[points] 대상 행 {len(pts)}({pts.target.value_counts().to_dict()}), 셀 {len(cells)}, 가용 메모리 {g:.1f} GB, 스레드 {a.threads}")
    t0 = time.time()
    for c in a.cmds:
        CMDS[c](a, pts, cells)
    log(f"[done] {' '.join(a.cmds)} · {time.time() - t0:.0f}s · 최대 RSS {max_rss_mb()} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
