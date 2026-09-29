"""지도 과제(MAP) · 레나 x 1 km 격자 조립, 조립 QA, 마스크. 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.2–6.5, 6.10.

결과와 무관한 계산만 한다(6.11). 입력은 원자료, v3 입력 표, 토양 도일 표뿐이고 LG·LGX·LGT·LGU·LGF 의 조각과 판정 표를 읽지 않는다.

격자(6.2)
  셀 색인은 LG 6B.3 과 같다. ky = floor(lat / 0.009), kx = floor(lon·cos φ / 0.009), φ = (ky + 0.5)·0.009°.
  셀 중심은 lat_c = (ky + 0.5)·0.009, lon_c = (kx + 0.5)·0.009 / cos φ 이고, 중심이 영역(71.5–73.6°N, 123.3–130.1°E) 안인 셀만 둔다.
  색인 함수는 scripts/1_data_prep/build_ext_cells_v1.py 의 cell_index 를 importlib 로 불러 쓰고 중심의 왕복 일치를 단언한다.

단계(--stage, 쉼표로 여러 개. 기본 = grid,tiles,cov,qa,mask,merge)
  grid   격자 셀 목록(cell_id, ky, kx, lat, lon)
  tiles  영역에 필요한 Copernicus DEM 1° 타일 24개(N71–N73 × E123–E130)의 상태를 적고, 없는 타일은 기존 내려받기 스크립트
         scripts/0_download/copernicus_dem.py 로 받는다. 그 스크립트는 상대 경로(data/processed/dem_tiles_needed.csv, data/raw/dem)를
         쓰므로 임시 작업 폴더에 목록 파일(없는 타일만)과 data/raw/dem 으로의 기호 연결을 만들어 그 폴더에서 실행한다. 공유 목록 파일
         data/processed/dem_tiles_needed.csv 는 건드리지 않는다. 새 타일은 rasterio 로 열리는지 확인하고, 열리지 않는 새 파일만 지운다
         (기존 파일은 건드리지 않는다). 상태: exists(실행 전부터 있음), downloaded(이번에 받음), absent_public(공개 저장소 HTTP 404, 바다),
         error:…(그 밖의 실패).
  cov    ext_cells_covariates_v1 의 단계 함수(stage_dem, stage_e5, stage_cci, stage_sg)를 importlib 로 불러 격자 셀 중심에 적용한다.
         부작용 차단(6.3): 모듈 상수 PARTS 를 <out>/cov_parts 로 바꾸고 tag 에 map_lena_ 접두어를 쓴다. stage_sg 는 allow_download=False
         로만 부른다(기존 창만 읽고 SoilGrids 메타를 쓰지 않는다). stage_tiles, stage_elev, fetch_tile 은 부르지 않는다.
  qa     (6.4) 같은 단계 함수를 v3 레나 F4_direct 좌표 3,037개에서 실행해 x25 와 토양 도일을 v3 값과 열별로 대조한다.
         허용 차: ERA5-Land 8열·SoilGrids 9열·CCI 2열·토양 도일 절대 차 1e-6, DEM 6열 상대 차 1e-4(|v3| = 0 이면 절대 차 1e-9).
         양쪽 NaN 은 일치, 한쪽만 NaN 은 불일치. 통과 = 모든 열에서 불일치 비율 ≤ 1 %. 보조(판정 아님): (i) 격자 셀 값과 그 셀 안
         v3 행 값의 차(중심 표집과 점 표집의 차이), (ii) 고정 중앙값으로 계산한 p4_ku, p2_edaphic 이 load_base 값과 같은지.
  mask   (6.5) 영구동토: data/processed/cci_pfr_mean_1997_2021.nc(0.1°) 최근접 값 < 10 % 이면 회색. 값이 없으면(바다·큰 수체) 사유
         pfr_missing 으로 회색. 수체: data/raw/gsw 의 occurrence 타일 120E_80N, 130E_80N 에서 셀 안 30 m 화소(255 = 자료 없음 제외)의
         평균 발생 빈도 ≥ 50 % 이면 회색. 바다·입력 결측: DEM 타일이 공개 저장소에 없는 셀, eval_mask 의 라벨 외 조건(cci_alt 유효,
         e5_sqrt_tdd_soil 유효) 또는 물리식 입력 e5_sqrt_tdd 가 결측인 셀은 회색. 그 밖의 x25 결측은 셀을 빼지 않고 결측 열 수만 적는다.
         해석 두 가지는 부록이 정하지 않은 것이라 메타(masks.summary)에 수치로 남긴다. (i) 6.5 의 '발생 빈도 50 % 이상 셀'을 화소 평균으로
         읽었다. 대안(발생 빈도 ≥ 50 % 인 화소가 유효 화소의 절반 이상)과 다른 셀 수는 water_rule_alt 에 있다. (ii) PFR 결측 회색은 6.5 에
         없는 추가 규칙이다. 이 사유만으로 회색이 된 셀 수(n_only)는 pfr_missing_rule 에 있다. 두 해석의 확정은 예측(h49) 전에 한다.
  merge  x25, 토양 도일(v3 규칙: e5_tdd_soil ≤ 0 이면 e5_tdd_soil, e5_sqrt_tdd_soil 결측), p4_ku·p2_edaphic(결측 대체 통계를 v3 입력 표
         전체의 중앙값으로 고정해 physics_ensemble 로 계산, 6.7), 마스크 열 → lena_grid_x25_v1.csv.gz 와 메타(크기, SHA-256 포함).

산출(--out-dir, 기본 data/processed/map_lena. 부록 6.10 은 data/processed/map/ 이지만 워크플로 지시에 따라 map_lena 에 둔다)
  lena_grid_cells_v1.csv                     grid 단계의 셀 목록
  lena_grid_dem_tiles_v1.csv                 타일 24개의 상태, 크기, SHA-256
  cov_parts/map_lena_grid_{dem,e5,cci,sg}.csv, cov_parts/map_lena_qa_{dem,e5,cci,sg}.csv   단계 함수의 중간 산출
  lena_grid_mask_v1.csv                      마스크 열
  lena_grid_qa_v1.csv                        6.4 대조(열별 불일치 비율, 최대 차, 통과), 결측 셀 수, 값 범위, 보조 대조
  lena_grid_x25_v1.csv.gz, lena_grid_x25_v1_meta.json

실행(ROOT, 공유 서버 규칙: 스레드 1, nice 10, GPU 없음)
  nice -n 10 python3 scripts/1_data_prep/build_map_grid_lena_v1.py        (스크립트로 실행하면 OMP·OpenBLAS·MKL·NUMEXPR 스레드를 1 로 둔다)
  --count-only: 격자 셀 수와 타일 상태만 출력한다(내려받지 않는다).
  재조립(내려받기 없이): --stage cov,qa,mask,merge --note '<사유>'. 타일 상태는 기존 lena_grid_dem_tiles_v1.csv 를 읽는다(받은 타일의
  'downloaded' 기록을 유지한다). 이전 메타의 해시, 작성 시각, 단계별 시간은 새 메타의 previous_meta 에 남긴다.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "1"                                      # 스크립트 실행: 1 스레드로 명시한다(셸의 더 큰 값을 따르지 않는다)
    else:
        os.environ.setdefault(_v, "1")

import argparse                                                                                         # noqa: E402
import hashlib                                                                                          # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import re                                                                                               # noqa: E402
import resource                                                                                         # noqa: E402
import shutil                                                                                           # noqa: E402
import subprocess                                                                                       # noqa: E402
import sys                                                                                              # noqa: E402
import tempfile                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
DEM_DIR = RAW / "dem"
GSW_DIR = RAW / "gsw"
PFR_NC = PROC / "cci_pfr_mean_1997_2021.nc"
DL_SCRIPT = ROOT / "scripts" / "0_download" / "copernicus_dem.py"
for _p in (str(ROOT / "src"), str(ROOT / "scripts" / "1_data_prep")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- 고정 설계값(부록 6.2–6.5)
BOX = dict(lat0=71.5, lat1=73.6, lon0=123.3, lon1=130.1)          # 6B.4 의 레나 델타 영역
CELL_DEG = 0.009
TILE_LATS = (71, 72, 73)
TILE_LONS = tuple(range(123, 131))                                 # E123–E130
PFR_MIN = 10.0                                                     # 영구동토 마스크: PFR < 10 % 이면 회색(산발 영구동토 이상만 표시)
WATER_MIN = 50.0                                                   # 수체 마스크: 평균 발생 빈도 ≥ 50 % 이면 회색
GSW_TILES = ("occurrence_120E_80Nv1_4_2021.tif", "occurrence_130E_80Nv1_4_2021.tif")
GSW_NODATA = 255
TOL_ABS = 1e-6                                                     # ERA5-Land, SoilGrids, CCI, 토양 도일
TOL_REL_DEM = 1e-4                                                 # DEM 6열
TOL_ABS_DEM_ZERO = 1e-9                                            # DEM 의 v3 값이 0 일 때
PASS_FRAC = 0.01                                                   # 열별 불일치 비율의 통과 기준
TAG_GRID = "map_lena_grid"
TAG_QA = "map_lena_qa"
GRID_VERSION = "v1"
# 6.5 의 회색 사유(우선순위 순서. gray_reason 열은 처음 해당하는 사유다. 사유별 수는 겹침을 포함해 따로 센다)
GRAY_REASONS = ("dem_tile_absent", "water", "pfr_missing", "pfr_lt10", "cci_missing", "soil_tdd_missing", "s_missing")


def log(*a):
    print(*a, flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_info(path) -> dict:
    p = Path(path)
    if not p.exists():
        return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), exists=False)
    return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), bytes=int(p.stat().st_size), sha256=sha256_file(p))


def load_module(name: str, rel: str):
    """기존 스크립트를 파일 경로에서 읽는다(고치지 않는다). 이미 읽었으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cov_module(parts_dir: Path):
    """ext_cells_covariates_v1 을 불러 PARTS 를 이 과제의 폴더로 바꾼다(6.3 (1))."""
    m = load_module("ext_cells_covariates_v1", "scripts/1_data_prep/ext_cells_covariates_v1.py")
    parts_dir.mkdir(parents=True, exist_ok=True)
    m.PARTS = Path(parts_dir)
    assert m.PARTS == Path(parts_dir) and "ext_labels" not in str(m.PARTS), "PARTS 가 LGD 폴더를 가리킨다"
    return m


def cells_module():
    return load_module("build_ext_cells_v1", "scripts/1_data_prep/build_ext_cells_v1.py")


# ================================================================ grid
def build_grid(box=BOX) -> pd.DataFrame:
    """셀 중심이 영역 안인 1 km 셀. 색인은 build_ext_cells_v1.cell_index 와 같다(중심의 왕복 일치를 단언한다)."""
    ci = cells_module().cell_index
    assert abs(cells_module().CELL_DEG - CELL_DEG) < 1e-15
    ky0 = int(np.floor(box["lat0"] / CELL_DEG)) - 1
    ky1 = int(np.floor(box["lat1"] / CELL_DEG)) + 1
    rows = []
    for ky in range(ky0, ky1 + 1):
        latc = (ky + 0.5) * CELL_DEG
        if not (box["lat0"] <= latc <= box["lat1"]):
            continue
        c = np.cos(np.radians(latc))
        kx0 = int(np.floor(box["lon0"] * c / CELL_DEG)) - 1
        kx1 = int(np.floor(box["lon1"] * c / CELL_DEG)) + 1
        kx = np.arange(kx0, kx1 + 1)
        lonc = (kx + 0.5) * CELL_DEG / c
        ok = (lonc >= box["lon0"]) & (lonc <= box["lon1"])
        for k_, lo_ in zip(kx[ok], lonc[ok]):
            rows.append((ky, int(k_), latc, float(lo_)))
    g = pd.DataFrame(rows, columns=["ky", "kx", "lat", "lon"])
    kyc, kxc = ci(g.lat.values, g.lon.values)
    bad = int(((kyc != g.ky.values) | (kxc != g.kx.values)).sum())
    assert bad == 0, f"셀 중심의 색인 왕복이 {bad}셀에서 다르다"
    g.insert(0, "cell_id", [f"{a}_{b}" for a, b in zip(g.ky, g.kx)])
    assert g.cell_id.is_unique
    return g


# ================================================================ tiles
def tile_list():
    return [(la, lo) for la in TILE_LATS for lo in TILE_LONS]


def tile_status_now(tname):
    rows = []
    for la, lo in tile_list():
        nm = tname(la, lo)
        p = DEM_DIR / (nm + ".tif")
        rows.append(dict(tlat=la, tlon=lo, tile=nm, present=bool(p.exists() and p.stat().st_size > 1000)))
    return pd.DataFrame(rows)


def _tile_opens(path: Path) -> bool:
    try:
        import rasterio
        with rasterio.open(path) as ds:
            ds.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:                                                     # noqa: BLE001
        return False


def stage_tiles(out: Path, tname, count_only=False) -> pd.DataFrame:
    """24개 타일의 상태를 적고 없는 타일을 copernicus_dem.py 로 받는다(임시 작업 폴더, 공유 목록 파일 불변)."""
    before = tile_status_now(tname)
    need = before[~before.present]
    status = {r.tile: "exists" for r in before.itertuples() if r.present}
    dl_log = ""
    if len(need) and not count_only:
        wd = Path(tempfile.mkdtemp(prefix="map_lena_dem_", dir=str(out)))
        try:
            (wd / "data" / "processed").mkdir(parents=True)
            (wd / "data" / "raw").mkdir(parents=True)
            os.symlink(DEM_DIR, wd / "data" / "raw" / "dem")
            need[["tlat", "tlon"]].assign(n=0).to_csv(wd / "data" / "processed" / "dem_tiles_needed.csv", index=False)
            log(f"[tiles] 없는 타일 {len(need)}개를 {DL_SCRIPT.relative_to(ROOT)} 로 받는다(작업 폴더 {wd.name})")
            r = subprocess.run([sys.executable, str(DL_SCRIPT)], cwd=str(wd), capture_output=True, text=True, timeout=3600)
            dl_log = (r.stdout or "") + (r.stderr or "")
            log(dl_log.strip())
        finally:
            shutil.rmtree(wd, ignore_errors=True)                         # 기호 연결만 지운다(대상 폴더는 남는다)
        assert DEM_DIR.exists() and any(DEM_DIR.iterdir()), "DEM 폴더가 사라졌다"
    miss_code = {m.group(1): m.group(2) for m in re.finditer(r"MISS (\S+)\s+\(HTTP (\d+)", dl_log)}
    err_name = {m.group(1): m.group(2) for m in re.finditer(r"ERR (\S+)\s+(.*)", dl_log)}
    rows = []
    for r in before.itertuples():
        p = DEM_DIR / (r.tile + ".tif")
        if r.tile in status:
            st = status[r.tile]
        elif count_only:
            st = "not_checked"
        elif p.exists() and p.stat().st_size > 1000:
            if _tile_opens(p):
                st = "downloaded"
            else:
                p.unlink()                                                 # 이번에 받은 새 파일이고 열리지 않는다
                st = "error:unreadable_download_removed"
        elif r.tile in miss_code:
            st = "absent_public" if miss_code[r.tile] == "404" else f"error:http_{miss_code[r.tile]}"
        elif r.tile in err_name:
            st = f"error:{err_name[r.tile][:60]}"
        else:
            st = "error:unknown"
        info = dict(tlat=r.tlat, tlon=r.tlon, tile=r.tile, status=st)
        if p.exists():
            info.update(bytes=int(p.stat().st_size), sha256=sha256_file(p))
        rows.append(info)
    t = pd.DataFrame(rows)
    if not count_only:
        t.to_csv(out / "lena_grid_dem_tiles_v1.csv", index=False)
    log(f"[tiles] {t.status.value_counts().to_dict()}")
    return t


# ================================================================ cov
def stage_cov(M, cells: pd.DataFrame, tag: str, stages=("dem", "e5", "cci", "sg")) -> dict:
    cell = cells.rename(columns={"cell_id": "id"})[["id", "lat", "lon"]].copy() if "cell_id" in cells else cells[["id", "lat", "lon"]].copy()
    cell["macro"] = "Lena"
    cell = cell.reset_index(drop=True)
    timing = {}
    for st in stages:
        t0 = time.time()
        if st == "dem":
            M.stage_dem(cell, tag)
        elif st == "e5":
            M.stage_e5(cell, tag)
        elif st == "cci":
            M.stage_cci(cell, tag)
        elif st == "sg":
            M.stage_sg(cell, tag, allow_download=False)                     # 6.3 (2): 기존 창만, SoilGrids 메타를 쓰지 않는다
        else:
            raise ValueError(st)
        timing[st] = round(time.time() - t0, 1)
    return timing


def merge_parts(M, cells: pd.DataFrame, tag: str) -> pd.DataFrame:
    cell = cells.rename(columns={"cell_id": "id"}) if "cell_id" in cells else cells.copy()
    cell = cell.copy()
    cell["id"] = cell["id"].astype(str)
    o = cell[["id", "lat", "lon"]].copy()
    for st in ("dem", "e5", "cci", "sg"):
        p = M.PARTS / f"{tag}_{st}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        o = o.merge(pd.read_csv(p, dtype={"id": str}), on="id", how="left", validate="one_to_one")
    le0 = (o.e5_tdd_soil <= 0).fillna(False).values
    o.loc[le0, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan                  # v3 규칙(e5_soil_tdd_v3 와 같다)
    o["soil_tdd_le0"] = le0.astype(int)
    return o


# ================================================================ 물리식 열(6.7: 결측 대체 통계 고정)
def phys_medians() -> tuple[dict, dict]:
    """load_base(v3 F4_direct, 라벨 유효 행)의 PHYS_INPUT_COLS 중앙값. load_base 의 physics._fallback 이 쓴 값과 같다."""
    from polar.m1_core import load_base
    df = load_base(PROC)
    cols = phys_cols()
    med = {c: float(np.nanmedian(df[c].values.astype(float))) for c in cols}
    info = dict(n_rows=int(len(df)), source="polar.m1_core.load_base(data/processed, fidelity_base_v3.csv, e5_soil_tdd_v3.csv, F4_direct)")
    return med, info


def phys_cols():
    """h42 의 PHYS_INPUT_COLS(physics.load_physics_inputs 가 결측을 채우는 열)."""
    return ("e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_maat", "e5_twarm", "e5_tcold", "e5_swe", "sg_bdod_5_15", "sg_sand_5_15",
            "sg_cfvo_5_15", "sg_soc_5_15")


def phys_columns(df: pd.DataFrame, med: dict) -> pd.DataFrame:
    """결측을 고정 중앙값으로 채운 입력에서 physics_ensemble(E = 1)의 p4_kudryavtsev, p2_edaphic 을 계산한다.
    채운 뒤에는 physics._fallback 이 할 일이 없으므로 결과가 중앙값 고정 규칙과 같다."""
    from polar.physics import physics_ensemble
    X = pd.DataFrame({c: np.where(np.isfinite(df[c].values.astype(float)), df[c].values.astype(float), med[c]) for c in phys_cols()})
    pe = physics_ensemble(X, E=1.0)
    return pd.DataFrame(dict(p4_ku=np.asarray(pe["p4_kudryavtsev"], float), p2_edaphic=np.asarray(pe["p2_edaphic"], float)), index=df.index)


# ================================================================ qa (6.4)
def x25_groups():
    from polar.fidelity import TERRAIN, CLIMATE, SOIL, CCI
    return dict(dem=list(TERRAIN), e5=list(CLIMATE), sg=list(SOIL), cci=list(CCI))


def compare_cols(a: np.ndarray, b: np.ndarray, kind: str) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    nan_mis = fa != fb
    d = np.abs(a - b)
    if kind == "rel":
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(np.abs(b) > 0, d / np.abs(b), np.where(d <= TOL_ABS_DEM_ZERO, 0.0, np.inf))
        over = both & (rel > TOL_REL_DEM)
        mx_rel = float(np.nanmax(np.where(both, rel, np.nan))) if both.any() else np.nan
    else:
        over = both & (d > TOL_ABS)
        mx_rel = np.nan
    bad = over | nan_mis
    return dict(n=int(len(a)), n_both_nan=int((~fa & ~fb).sum()), n_nan_mismatch=int(nan_mis.sum()), n_over_tol=int(over.sum()),
                frac_mismatch=float(bad.mean()) if len(a) else np.nan, max_abs_diff=float(np.nanmax(np.where(both, d, np.nan))) if both.any() else np.nan,
                max_rel_diff=mx_rel, tol=f"rel {TOL_REL_DEM:g}" if kind == "rel" else f"abs {TOL_ABS:g}",
                pass_=bool(bad.mean() <= PASS_FRAC) if len(a) else False)


def v3_lena() -> pd.DataFrame:
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
    L = v3[(v3.region == "Lena_RU") & (v3.source_id == "F4_direct")].copy()
    L["id"] = L.loc_id.astype(str)
    return L.reset_index(drop=True)


def stage_qa(M, out: Path, grid_df: pd.DataFrame | None, med: dict | None) -> tuple[pd.DataFrame, dict]:
    """6.4 의 대조. 단계 함수를 v3 레나 좌표에서 다시 실행한다."""
    L = v3_lena()
    log(f"[qa] v3 레나 F4_direct {len(L)}행")
    timing = stage_cov(M, L[["id", "lat", "lon"]], TAG_QA)
    q = merge_parts(M, L[["id", "lat", "lon"]], TAG_QA)
    m = q.merge(L, on="id", suffixes=("", "_v3"), validate="one_to_one")
    assert len(m) == len(L)
    rows = []
    G = x25_groups()
    for grp, cols in G.items():
        for c in cols:
            r = compare_cols(m[c].values, m[c + "_v3"].values, "rel" if grp == "dem" else "abs")
            rows.append(dict(kind="v3_rerun", group=grp, column=c, **r))
    s3 = pd.read_csv(PROC / "e5_soil_tdd_v3.csv")
    s3 = s3[s3.loc_id >= 0].copy()
    s3["id"] = s3.loc_id.astype(str)
    ms = q.merge(s3, on="id", suffixes=("", "_s3"), how="inner", validate="one_to_one")
    for c in ("e5_tdd_soil", "e5_sqrt_tdd_soil", "e5_fdd_soil", "e5_stl1_twarm", "e5_stl1_tcold", "soil_fallback_deg"):
        r = compare_cols(ms[c].values, ms[c + "_s3"].values, "abs")
        rows.append(dict(kind="v3_rerun", group="soil_tdd" if c in ("e5_tdd_soil", "e5_sqrt_tdd_soil") else "soil_tdd_aux", column=c,
                         n_rows_joined=int(len(ms)), **r))
    extra = dict(n_v3=int(len(L)), n_soil_joined=int(len(ms)), timing_s=timing,
                 sg_all9_missing_rows=int(q[G["sg"]].isna().all(axis=1).sum()),
                 sg_any_missing_rows=int(q[G["sg"]].isna().any(axis=1).sum()),
                 v3_sg_all9_missing_rows=int(L[G["sg"]].isna().all(axis=1).sum()))
    # 보조 (ii): 고정 중앙값의 물리식 열이 load_base 값과 같은지(레나 행)
    if med is not None:
        from polar.m1_core import load_base
        lb = load_base(PROC)
        lb = lb[lb.macro == "Lena"][["loc_id", "p4_ku", "p2_edaphic"]].copy()
        lb["id"] = lb.loc_id.astype(str)
        pc = phys_columns(L, med)
        pc["id"] = L.id.values
        mm = pc.merge(lb, on="id", suffixes=("", "_lb"))
        for c in ("p4_ku", "p2_edaphic"):
            r = compare_cols(mm[c].values, mm[c + "_lb"].values, "abs")
            rows.append(dict(kind="phys_fixed_median_vs_load_base", group="phys", column=c, **r))
    # 보조 (i): 격자 셀 값 대 그 셀 안 v3 행 값(점 표집과 셀 중심 표집의 차이. 판정에 쓰지 않는다)
    if grid_df is not None:
        ci = cells_module().cell_index
        ky, kx = ci(L.lat.values, L.lon.values)
        key = pd.Series([f"{a}_{b}" for a, b in zip(ky, kx)], index=L.index)
        gi = grid_df.set_index("cell_id")
        inside = key.isin(gi.index).values
        extra.update(v3_rows_in_grid=int(inside.sum()), v3_cells_in_grid=int(key[inside].nunique()))
        for grp, cols in G.items():
            for c in cols:
                gv = gi.reindex(key[inside].values)[c].values.astype(float)
                vv = L.loc[inside, c].values.astype(float)
                ok = np.isfinite(gv) & np.isfinite(vv)
                d = gv[ok] - vv[ok]
                rows.append(dict(kind="grid_center_vs_v3_point", group=grp, column=c, n=int(inside.sum()), n_both_finite=int(ok.sum()),
                                 n_nan_mismatch=int((np.isfinite(gv) != np.isfinite(vv)).sum()),
                                 frac_equal=float(np.mean(np.abs(d) <= (np.abs(vv[ok]) * TOL_REL_DEM if grp == "dem" else TOL_ABS))) if ok.any() else np.nan,
                                 median_abs_diff=float(np.median(np.abs(d))) if ok.any() else np.nan,
                                 p90_abs_diff=float(np.percentile(np.abs(d), 90)) if ok.any() else np.nan,
                                 corr=float(np.corrcoef(gv[ok], vv[ok])[0, 1]) if ok.sum() > 2 and np.std(vv[ok]) > 0 and np.std(gv[ok]) > 0 else np.nan))
    qa = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    return qa, extra


def grid_summary_rows(g: pd.DataFrame, L: pd.DataFrame | None) -> pd.DataFrame:
    """열별 결측 셀 수와 값 범위(격자 전체, 회색이 아닌 셀), v3 레나 값 범위."""
    rows = []
    G = x25_groups()
    shown = g.gray.values == 0
    for grp, cols in list(G.items()) + [("soil_tdd", ["e5_tdd_soil", "e5_sqrt_tdd_soil"]), ("phys", ["p4_ku", "p2_edaphic"])]:
        for c in cols:
            v = g[c].values.astype(float)
            r = dict(kind="grid_range", group=grp, column=c, n=int(len(v)), n_missing=int((~np.isfinite(v)).sum()),
                     n_missing_shown=int((~np.isfinite(v[shown])).sum()))
            for tag, arr in (("grid", v), ("shown", v[shown]), ("v3", L[c].values.astype(float) if (L is not None and c in L) else np.array([]))):
                a = arr[np.isfinite(arr)]
                if len(a):
                    r.update({f"{tag}_min": float(a.min()), f"{tag}_p01": float(np.percentile(a, 1)), f"{tag}_p50": float(np.median(a)),
                              f"{tag}_p99": float(np.percentile(a, 99)), f"{tag}_max": float(a.max())})
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ mask (6.5)
def pfr_at(lat, lon) -> np.ndarray:
    import xarray as xr
    with xr.open_dataset(PFR_NC) as ds:
        la = ds["lat"].values.astype(float); lo = ds["lon"].values.astype(float)
        iy = np.clip(np.searchsorted(la, lat), 1, len(la) - 1)
        iy = np.where(np.abs(la[iy - 1] - lat) <= np.abs(la[iy] - lat), iy - 1, iy)
        ix = np.clip(np.searchsorted(lo, lon), 1, len(lo) - 1)
        ix = np.where(np.abs(lo[ix - 1] - lon) <= np.abs(lo[ix] - lon), ix - 1, ix)
        y0, y1, x0, x1 = int(iy.min()), int(iy.max()) + 1, int(ix.min()), int(ix.max()) + 1
        sub = ds["pfr_mean"].isel(lat=slice(y0, y1), lon=slice(x0, x1)).values.astype(float)
    return sub[iy - y0, ix - x0]


def gsw_cells(g: pd.DataFrame) -> pd.DataFrame:
    """셀 안 GSW 30 m 화소의 평균 발생 빈도(255 제외), 발생 빈도 ≥ 50 % 화소 비율, 유효 화소 비율, 중심 화소 값.
    화소 → 셀 배정은 화소 중심 좌표에 같은 셀 색인(그 위도 띠의 φ)을 적용한다. 타일마다 영역 창을 한 번 읽는다(uint8)."""
    import rasterio
    from rasterio.windows import Window
    lon_min, lon_max = float(g.lon.min()) - 0.02, float(g.lon.max()) + 0.02
    lat_min, lat_max = float(g.ky.min()) * CELL_DEG - 0.001, float(g.ky.max() + 1) * CELL_DEG + 0.001
    wins = []
    for f in GSW_TILES:
        with rasterio.open(GSW_DIR / f) as d:
            t = d.transform
            assert abs(t.a - 0.00025) < 1e-12 and abs(t.e + 0.00025) < 1e-12, "GSW 화소 크기가 예상과 다르다"
            c0 = max(0, int(np.floor((lon_min - t.c) / t.a)))
            c1 = min(d.width, int(np.ceil((lon_max - t.c) / t.a)) + 1)
            r0 = max(0, int(np.floor((t.f - lat_max) / (-t.e))))
            r1 = min(d.height, int(np.ceil((t.f - lat_min) / (-t.e))) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            a = d.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            wins.append(dict(a=a, plat=t.f + (np.arange(r0, r1) + 0.5) * t.e, plon=t.c + (np.arange(c0, c1) + 0.5) * t.a))
    out = {k: np.full(len(g), np.nan) for k in ("gsw_occ_mean", "gsw_water_frac50", "gsw_valid_frac", "gsw_occ_center")}
    pos = pd.Series(np.arange(len(g)), index=g.cell_id.values)
    for ky, sub in g.groupby("ky"):
        la0, la1 = ky * CELL_DEG, (ky + 1) * CELL_DEG
        cphi = np.cos(np.radians((ky + 0.5) * CELL_DEG))
        kmin, kmax = int(sub.kx.min()), int(sub.kx.max())
        nb = kmax - kmin + 1
        S = np.zeros(nb); C = np.zeros(nb); W = np.zeros(nb); T = np.zeros(nb)
        for w in wins:
            rin = (w["plat"] >= la0) & (w["plat"] < la1)
            if not rin.any():
                continue
            a = w["a"][rin]
            valid = a != GSW_NODATA
            kx_p = np.floor(w["plon"] * cphi / CELL_DEG).astype(int) - kmin
            ok = (kx_p >= 0) & (kx_p < nb)
            S += np.bincount(kx_p[ok], weights=np.where(valid, a, 0).sum(0)[ok].astype(float), minlength=nb)
            C += np.bincount(kx_p[ok], weights=valid.sum(0)[ok].astype(float), minlength=nb)
            W += np.bincount(kx_p[ok], weights=(valid & (a >= WATER_MIN)).sum(0)[ok].astype(float), minlength=nb)
            T += np.bincount(kx_p[ok], minlength=nb).astype(float) * float(rin.sum())
        j = sub.kx.values - kmin
        i = pos[sub.cell_id.values].values
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gsw_valid_frac"][i] = np.where(T[j] > 0, C[j] / T[j], np.nan)
            out["gsw_occ_mean"][i] = np.where(C[j] > 0, S[j] / C[j], np.nan)
            out["gsw_water_frac50"][i] = np.where(C[j] > 0, W[j] / C[j], np.nan)
    for w in wins:                                                      # 중심 화소 값(참고 열)
        m = (g.lat.values >= w["plat"].min() - 0.000125) & (g.lat.values <= w["plat"].max() + 0.000125) & \
            (g.lon.values >= w["plon"].min() - 0.000125) & (g.lon.values <= w["plon"].max() + 0.000125)
        if not m.any():
            continue
        r = np.clip(np.round((w["plat"][0] - g.lat.values[m]) / 0.00025).astype(int), 0, len(w["plat"]) - 1)
        c = np.clip(np.round((g.lon.values[m] - w["plon"][0]) / 0.00025).astype(int), 0, len(w["plon"]) - 1)
        v = w["a"][r, c].astype(float)
        v[v == GSW_NODATA] = np.nan
        out["gsw_occ_center"][np.where(m)[0]] = v
    return pd.DataFrame(out, index=g.index)


def build_masks(g: pd.DataFrame, tiles: pd.DataFrame, tname) -> pd.DataFrame:
    """6.5 의 마스크 열. 입력 g 는 x25·토양 도일이 붙은 격자."""
    m = pd.DataFrame(index=g.index)
    m["cell_id"] = g.cell_id.values
    m["pfr_mean"] = pfr_at(g.lat.values, g.lon.values)
    m = pd.concat([m, gsw_cells(g)], axis=1)
    tl = np.floor(g.lat.values).astype(int); tn = np.floor(g.lon.values).astype(int)
    tst = {(int(r.tlat), int(r.tlon)): r.status for r in tiles.itertuples()}
    m["dem_tile_status"] = [tst.get((a, b), "not_in_list") for a, b in zip(tl, tn)]
    m["dem_tile_absent"] = (m.dem_tile_status == "absent_public").astype(int)
    m["dem_tile_error"] = m.dem_tile_status.str.startswith("error").astype(int)
    m["water"] = (m.gsw_occ_mean >= WATER_MIN).fillna(False).astype(int)
    m["pfr_missing"] = (~np.isfinite(m.pfr_mean.values)).astype(int)
    m["pfr_lt10"] = (np.isfinite(m.pfr_mean.values) & (m.pfr_mean.values < PFR_MIN)).astype(int)
    m["cci_missing"] = (~np.isfinite(g.cci_alt.values.astype(float))).astype(int)
    m["soil_tdd_missing"] = (~np.isfinite(g.e5_sqrt_tdd_soil.values.astype(float))).astype(int)
    m["s_missing"] = (~np.isfinite(g.e5_sqrt_tdd.values.astype(float))).astype(int)
    reasons = np.array(["" for _ in range(len(m))], dtype=object)
    for r in GRAY_REASONS[::-1]:
        reasons = np.where(m[r].values == 1, r, reasons)
    m["gray_reason"] = reasons
    m["gray"] = (m.gray_reason != "").astype(int)
    # 육지 판정(보고용): DEM 타일이 있고 DEM 값이 유효하고 수체가 아닌 셀. GSW 자료가 없는 셀(바다 쪽 255)은 육지로 세지 않는다
    dem_ok = np.isfinite(g.dem_elev.values.astype(float)) & (m.dem_tile_absent.values == 0)
    m["land"] = (dem_ok & (m.water.values == 0) & np.isfinite(m.gsw_occ_mean.values)).astype(int)
    feats = [c for cols in x25_groups().values() for c in cols if c != "cci_valid"]
    m["n_x25_missing"] = g[feats].isna().sum(axis=1).astype(int).values
    return m


def mask_summary(m: pd.DataFrame) -> dict:
    land = m.land.values == 1
    out = dict(n_cells=int(len(m)), n_gray=int(m.gray.sum()), n_shown=int((m.gray == 0).sum()), n_land=int(land.sum()),
               n_land_gray=int((land & (m.gray.values == 1)).sum()),
               frac_land_gray=float((land & (m.gray.values == 1)).sum() / max(land.sum(), 1)),
               by_reason_any={r: int(m[r].sum()) for r in GRAY_REASONS},
               by_reason_first={r: int((m.gray_reason == r).sum()) for r in GRAY_REASONS},
               land_by_reason_any={r: int(m.loc[land, r].sum()) for r in GRAY_REASONS},
               land_by_reason_first={r: int((m.gray_reason[land] == r).sum()) for r in GRAY_REASONS},
               n_x25_missing_shown=m.loc[m.gray == 0, "n_x25_missing"].value_counts().sort_index().to_dict(),
               pfr_values_shown=dict(min=float(np.nanmin(m.pfr_mean[m.gray == 0])) if (m.gray == 0).any() else None,
                                     max=float(np.nanmax(m.pfr_mean[m.gray == 0])) if (m.gray == 0).any() else None))
    out["n_x25_missing_shown"] = {str(k): int(v) for k, v in out["n_x25_missing_shown"].items()}
    gray, water = m.gray.values == 1, m.water.values == 1
    if "gsw_water_frac50" in m.columns:                                  # 수체 규칙의 대안 해석(적용하지 않는다, 기록만)
        alt = np.nan_to_num(m.gsw_water_frac50.values.astype(float), nan=0.0) >= 0.5
        out["water_rule_alt"] = dict(
            applied="평균 발생 빈도 ≥ 50 %(gsw_occ_mean)", alternative="발생 빈도 ≥ 50 % 인 30 m 화소가 유효 화소의 절반 이상(gsw_water_frac50 ≥ 0.5)",
            n_water_applied=int(water.sum()), n_water_alt=int(alt.sum()), n_alt_not_applied=int((alt & ~water).sum()),
            n_alt_not_applied_shown=int((alt & ~water & ~gray).sum()), n_alt_not_applied_land=int((alt & ~water & land).sum()),
            n_applied_not_alt=int((water & ~alt).sum()),
            note="대안 해석이면 n_alt_not_applied_shown 셀이 표시에서 빠지고 n_applied_not_alt 셀은 수체가 아니게 된다. 확정은 예측 전에 한다")
    other = [r for r in GRAY_REASONS if r != "pfr_missing"]
    pfr_only = (m.pfr_missing.values == 1) & (m[other].sum(axis=1).values == 0)
    out["pfr_missing_rule"] = dict(
        note="PFR 결측 셀의 회색 처리는 6.5 에 없는 추가 규칙이다. n_only 는 이 사유만으로 회색이 된 셀 수(0 이면 규칙을 빼도 표시 셀이 같다)",
        n_any=int(m.pfr_missing.sum()), n_first=int((m.gray_reason == "pfr_missing").sum()),
        n_land_first=int(((m.gray_reason == "pfr_missing") & land).sum()), n_only=int(pfr_only.sum()), n_land_only=int((pfr_only & land).sum()))
    return out


# ================================================================ main
def to_csv_gz(df: pd.DataFrame, path: Path):
    """재현 가능한 gzip(헤더 시각 0). 같은 내용이면 같은 해시다."""
    df.to_csv(path, index=False, compression=dict(method="gzip", mtime=0), float_format="%.10g")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="지도 과제(MAP) 레나 x 1 km 격자 조립·QA·마스크")
    ap.add_argument("--stage", default="grid,tiles,cov,qa,mask,merge")
    ap.add_argument("--out-dir", default="data/processed/map_lena")
    ap.add_argument("--count-only", action="store_true", help="격자 셀 수와 타일 상태만 출력(내려받기·계산 없음)")
    ap.add_argument("--note", default="", help="메타에 적는 실행 사유(재조립 때)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()
    out = Path(a.out_dir) if os.path.isabs(a.out_dir) else ROOT / a.out_dir
    guard = [PROC / d for d in ("lg", "lgx", "lgt", "lgd", "lgu", "lgf", "ext_labels")]
    if any(out.resolve() == q.resolve() or q.resolve() in out.resolve().parents for q in guard):
        raise SystemExit(f"[거부] 출력 폴더 {out} 는 실험 산출 폴더(LG·LGX·LGT·LGD·LGU·LGF·ext_labels) 안이다")
    out.mkdir(parents=True, exist_ok=True)
    parts = out / "cov_parts"
    stages = [s for s in a.stage.split(",") if s]
    M = cov_module(parts)
    tname = M.tname
    timing = {}
    grid_path = out / "lena_grid_cells_v1.csv"

    g = build_grid()
    log(f"[grid] 셀 {len(g):,}개 · 행(ky) {g.ky.nunique()} · 위도 {g.lat.min():.4f}–{g.lat.max():.4f} · 경도 {g.lon.min():.4f}–{g.lon.max():.4f}")
    if a.count_only:
        stage_tiles(out, tname, count_only=True)
        return dict(n_cells=int(len(g)))
    if "grid" in stages:
        g.to_csv(grid_path, index=False, float_format="%.10g")
    g = pd.read_csv(grid_path, dtype={"cell_id": str})

    tiles_path = out / "lena_grid_dem_tiles_v1.csv"
    if "tiles" in stages:
        t0 = time.time()
        stage_tiles(out, tname)
        timing["tiles"] = round(time.time() - t0, 1)
    tiles = pd.read_csv(tiles_path) if tiles_path.exists() else tile_status_now(tname).assign(status="not_checked")

    if "cov" in stages:
        timing["cov"] = stage_cov(M, g, TAG_GRID)

    med, med_info = None, {}
    if any(s in stages for s in ("qa", "merge")):
        med, med_info = phys_medians()

    gx = None
    if any(s in stages for s in ("qa", "mask", "merge")):
        gx = merge_parts(M, g, TAG_GRID).rename(columns={"id": "cell_id"})
        gx = g[["cell_id", "ky", "kx"]].merge(gx, on="cell_id", validate="one_to_one")

    qa_extra = {}
    if "qa" in stages:
        t0 = time.time()
        qa, qa_extra = stage_qa(M, out, gx, med)
        qa.to_csv(out / "lena_grid_qa_v1.csv", index=False)
        timing["qa"] = round(time.time() - t0, 1)
        prim = qa[qa.kind == "v3_rerun"]
        prim = prim[prim.group.isin(["dem", "e5", "sg", "cci", "soil_tdd"])]
        log(f"[qa] 6.4 대조 열 {len(prim)} · 통과 {int(prim['pass'].sum())} · 최대 불일치 비율 {prim.frac_mismatch.max():.4f}")
        if not prim["pass"].all():
            log("[qa] 통과하지 못한 열:\n" + prim[~prim["pass"]][["group", "column", "frac_mismatch", "n_nan_mismatch", "n_over_tol", "max_abs_diff"]].to_string())

    mpath = out / "lena_grid_mask_v1.csv"
    if "mask" in stages:
        t0 = time.time()
        mk = build_masks(gx, tiles, tname)
        mk.to_csv(mpath, index=False, float_format="%.6g")
        timing["mask"] = round(time.time() - t0, 1)
        log(f"[mask] {json.dumps(mask_summary(mk), ensure_ascii=False)}")

    if "merge" in stages:
        mk = pd.read_csv(mpath, dtype={"cell_id": str})
        pc = phys_columns(gx, med)
        full = pd.concat([gx.reset_index(drop=True), pc.reset_index(drop=True)], axis=1)
        full = full.merge(mk, on="cell_id", validate="one_to_one")
        x25 = [c for cols in x25_groups().values() for c in cols]
        keep = (["cell_id", "ky", "kx", "lat", "lon"] + x25 + ["e5_tdd_soil", "e5_sqrt_tdd_soil", "soil_tdd_le0", "e5_fallback_deg", "soil_fallback_deg",
                "cci_n_years", "sg_windows", "dem_tile_missing", "p4_ku", "p2_edaphic"] + [c for c in mk.columns if c != "cell_id"])
        full = full[keep]
        gpath = out / f"lena_grid_x25_{GRID_VERSION}.csv.gz"
        to_csv_gz(full, gpath)
        L = v3_lena()
        qa_path = out / "lena_grid_qa_v1.csv"
        qa_old = pd.read_csv(qa_path) if qa_path.exists() else pd.DataFrame()
        rng = grid_summary_rows(full, L)
        qa_all = pd.concat([qa_old[qa_old.kind != "grid_range"] if len(qa_old) else qa_old, rng], ignore_index=True)
        qa_all.to_csv(qa_path, index=False)
        prim = qa_all[(qa_all.kind == "v3_rerun") & qa_all.group.isin(["dem", "e5", "sg", "cci", "soil_tdd"])]
        x25_valid = full[x25].notna().all(axis=1)
        ms = mask_summary(full)
        sg_used = sorted({w for x in full.sg_windows.fillna("") for w in str(x).split(";") if w})
        meta = dict(
            created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/1_data_prep/build_map_grid_lena_v1.py",
            script_sha256=sha256_file(Path(__file__)), plan="docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.2–6.5, 6.10 (커밋 316714c)",
            out_dir_note="부록 6.10 의 data/processed/map/ 대신 워크플로 지시의 data/processed/map_lena/ 에 둔다",
            grid=dict(box=BOX, cell_deg=CELL_DEG, rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009; "
                      "셀 중심 lat_c = (ky+0.5)*0.009, lon_c = (kx+0.5)*0.009/cos(phi); 중심이 영역 안인 셀", n_cells=int(len(full)),
                      n_rows_ky=int(full.ky.nunique()), cell_index_fn="scripts/1_data_prep/build_ext_cells_v1.py:cell_index"),
            covariates=dict(module="scripts/1_data_prep/ext_cells_covariates_v1.py", module_sha256=sha256_file(ROOT / "scripts/1_data_prep/ext_cells_covariates_v1.py"),
                            stages=["stage_dem", "stage_e5", "stage_cci", "stage_sg(allow_download=False)"], parts_dir=str(parts.relative_to(ROOT)),
                            tags=[TAG_GRID, TAG_QA], era5land=file_info(M.NC),
                            cci_alt_files={Path(f).name: sha256_file(f) for f in sorted(M.CCI_DIR.glob("*.nc"))},
                            soilgrids_windows=sg_used,
                            soilgrids_files={f"{w}/{f.name}": sha256_file(f) for w in sg_used for f in sorted((M.SG_RAW / w).glob("*.tif"))},
                            soil_tdd_le0_set_nan=int(full.soil_tdd_le0.sum()),
                            e5_fallback=full.e5_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict(),
                            soil_fallback=full.soil_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict()),
            dem_tiles=tiles.to_dict("records"), dem_tiles_status_counts=tiles.status.value_counts().to_dict(),
            dem_download_script=file_info(DL_SCRIPT),
            phys_fallback_medians=dict(values=med, **med_info, note="p4_ku, p2_edaphic 의 결측 대체 통계(6.7). 격자 셀로 다시 계산하지 않는다"),
            masks=dict(pfr=dict(file=file_info(PFR_NC), rule=f"최근접 0.1° 값 < {PFR_MIN:g} % 이면 회색(pfr_lt10). 값이 없으면 회색(pfr_missing)",
                                interpretation="pfr_missing 회색은 6.5 에 없는 추가 규칙이다. 영향은 summary.pfr_missing_rule(확정 전)"),
                       water=dict(files={f: file_info(GSW_DIR / f) for f in GSW_TILES},
                                  rule=f"셀 안 30 m 화소(255 제외)의 평균 발생 빈도 ≥ {WATER_MIN:g} % 이면 회색(water). 중심 화소 값은 참고 열",
                                  interpretation="6.5 의 '발생 빈도 50 % 이상 셀'을 화소 평균으로 읽었다. 대안 해석과의 차이는 summary.water_rule_alt(확정 전)"),
                       sea_input=dict(rule="DEM 타일이 공개 저장소에 없음(dem_tile_absent), cci_alt 결측, e5_sqrt_tdd_soil 결측, e5_sqrt_tdd 결측 이면 회색. "
                                           "그 밖의 x25 결측은 n_x25_missing 에 적고 셀을 빼지 않는다"),
                       land_rule="land = DEM 타일 있음 · dem_elev 유효 · GSW 유효 화소 있음 · 수체 아님(보고용)", gray_reason_order=list(GRAY_REASONS),
                       summary=ms),
            validity=dict(x25_all_valid_frac=float(x25_valid.mean()), x25_all_valid_frac_shown=float(x25_valid[full.gray == 0].mean()) if (full.gray == 0).any() else None,
                          missing_by_col={c: int(full[c].isna().sum()) for c in x25 + ["e5_sqrt_tdd_soil", "p4_ku", "p2_edaphic"]}),
            qa=dict(file=str((out / "lena_grid_qa_v1.csv").relative_to(ROOT)), tolerance=dict(abs=TOL_ABS, rel_dem=TOL_REL_DEM, pass_frac=PASS_FRAC),
                    n_cols=int(len(prim)), n_pass=int(prim["pass"].astype(bool).sum()) if len(prim) else 0,
                    passed=bool(len(prim) and prim["pass"].astype(bool).all()), max_frac_mismatch=float(prim.frac_mismatch.max()) if len(prim) else None,
                    **qa_extra),
            inputs=dict(fidelity_base_v3=file_info(PROC / "fidelity_base_v3.csv"), e5_soil_tdd_v3=file_info(PROC / "e5_soil_tdd_v3.csv")),
            outputs={k: file_info(p) for k, p in dict(grid=gpath, cells=grid_path, tiles=tiles_path, mask=mpath, qa=qa_path).items()},
            cov_parts={p.name: file_info(p) for p in sorted(parts.glob("map_lena_*.csv"))},
            timing_s=timing, max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), nice=os.nice(0)),
            status_note="6.3–6.5 의 결과 전 산출이다. 예측(h49)과 그림은 LG 집계, LGX L29, LGU-B2 판정 뒤에 한다(6.11)")
        mpath_meta = out / f"lena_grid_x25_{GRID_VERSION}_meta.json"
        meta["stages_run"] = stages
        meta["run_note"] = a.note
        if mpath_meta.exists():                                           # 재조립: 이전 메타의 기록을 남긴다
            try:
                old = json.loads(mpath_meta.read_text())
                meta["previous_meta"] = dict(sha256=sha256_file(mpath_meta), created=old.get("created"), timing_s=old.get("timing_s"),
                                             max_rss_mb=old.get("max_rss_mb"), stages_run=old.get("stages_run"),
                                             grid_sha256=old.get("outputs", {}).get("grid", {}).get("sha256"),
                                             grid_unchanged=old.get("outputs", {}).get("grid", {}).get("sha256") == meta["outputs"]["grid"]["sha256"])
            except (ValueError, OSError):
                meta["previous_meta"] = dict(note="이전 메타를 읽지 못했다")
        mpath_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        log(f"[merge] {gpath.relative_to(ROOT)} {full.shape} · {meta['outputs']['grid']['bytes']:,} B · sha256 {meta['outputs']['grid']['sha256'][:16]}…")
        log(f"[merge] QA 통과 {meta['qa']['passed']} · x25 모두 유효 {meta['validity']['x25_all_valid_frac']:.3f} · 회색 {ms['n_gray']:,}/{ms['n_cells']:,}")
    log(f"[done] {timing} · 합계 {time.time() - t_start:.0f}s · 최대 RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    return dict(n_cells=int(len(g)))


if __name__ == "__main__":
    main()
