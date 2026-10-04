"""지도 과제(MAP) · 알래스카 1 km 격자 조립, 조립 QA, 마스크. 레나 격자 스크립트(build_map_grid_lena_v1.py)의 함수를 importlib 로 불러 쓴다.
레나 스크립트와 ext_cells_covariates_v1.py 는 고치지 않는다. 이 스크립트는 두 모듈의 상수 몇 개를 실행 중에만 알래스카 값으로 바꾼다(아래 '재사용').

영역(지시 2026-10-05)
  위도 59–71.5°N, 경도 168–141°W 상자 안의 셀 가운데 (i) 셀 중심이 Natural Earth 10m 육지 다각형(ne_10m_land, cartopy 캐시) 안이고
  (ii) ESA CCI PFR(data/processed/cci_pfr_mean_1997_2021.nc, 1997–2021 평균, 0.1°) 최근접 값이 50 % 이상인 셀만 격자에 둔다.
  지시의 'PFR ≥ 0.5' 를 비율 0.5(= 50 %, 연속·불연속 영구동토대)로 읽었다. 파일 단위는 % 이다. 다른 문턱(10 %, 0.5 %)의 셀 수는 메타
  grid.domain_counts 에 적는다. 셀 규칙은 레나와 같다(build_ext_cells_v1.cell_index, 0.009°, 레나 build_grid(box=…)).

단계(--stage, 쉼표로 여러 개. 기본 = grid,tiles,cov,qa,mask,merge)
  grid   레나 build_grid(box=알래스카 상자) → 레나 pfr_at → 육지 판정 → 영역 셀 목록(alaska_grid_cells_v1.csv)
  tiles  영역 셀이 걸친 Copernicus DEM GLO-30 1° 타일. 없는 타일은 AWS 공개 버킷(https://copernicus-dem-30m.s3.amazonaws.com, 계정 불필요,
         scripts/0_download/copernicus_dem.py 와 같은 주소와 파일 이름)에서 data/raw/dem 에 받는다. 받는 동안은 <이름>.tif.part 에 쓰고
         끝나면 이름을 바꾼다. 기존 파일은 덮어쓰지 않는다. 새 파일이 rasterio 로 열리지 않으면 그 새 파일만 지운다. 동시 내려받기 4개.
         상태: exists, downloaded, absent_public(HTTP 404, 바다), error:…
  cov    ext_cells_covariates_v1 의 stage_dem, stage_e5, stage_cci, stage_sg(allow_download=False) 를 격자 셀 중심에 적용한다(레나 cov_module 로
         PARTS 를 <out>/cov_parts 로 바꾼다, tag map_alaska_grid). 기존 SoilGrids 창(enrich_soilgrids_wcs.WINDOWS 의 namerica 등, E3 창) 밖이라
         9층이 비는 셀은 같은 WCS 경로(maps.isric.org mapserv WCS 2.0.1, IGH, GEOTIFF_INT16, enrich_soilgrids_wcs.igh_bbox·scalesize 의 5 km
         규칙, 9층)로 새 창 map_alaska_v1_<lobe> 를 data/raw/soilgrids_wcs 에 받아 같은 표집 함수(_sample, SCALE 나눔)로 채운다. 기존 창 메타
         (windows_wcs_meta*.json)는 쓰지 않고 새 창 메타는 data/raw/soilgrids_wcs/windows_wcs_meta_map_alaska_v1.json 에 따로 쓴다.
  qa     같은 단계 함수를 v3 알래스카 F4_direct 좌표(region ABoVE_AK, United States (Alaska))에서 다시 실행해 x25 와 토양 도일을 v3 값과
         열별로 대조한다(레나 stage_qa, 같은 허용 차: ERA5-Land·SoilGrids·CCI·토양 도일 절대 차 1e-6, DEM 상대 차 1e-4, 열별 불일치 ≤ 1 %).
  mask   레나 build_masks 와 같은 사유 순서로 회색을 정한다. 수체 = GSW occurrence(data/raw/gsw 의 170W·160W·150W × 70N·80N 타일) 셀 평균
         ≥ 50 %. 59–60°N 띠는 GSW 타일(60N)이 없어 수체를 판정하지 않는다(gsw_occ_mean 결측, 셀 수는 메타). PFR 문턱은 50 %(열 이름 pfr_lt50).
  merge  x25, 토양 도일(v3 규칙: e5_tdd_soil ≤ 0 이면 결측), 마스크 열 → alaska_grid_x25_v1.csv.gz 와 메타(크기, SHA-256, 셀 수, 시간).
         p4_ku, p2_edaphic 은 알래스카 지도 방법(P1, R1)이 쓰지 않아 계산하지 않는다.

재사용(레나 스크립트 함수와 실행 중 상수 교체)
  build_grid(box), pfr_at, cov_module, merge_parts, x25_groups, compare_cols, stage_qa, gsw_cells, build_masks, mask_summary, to_csv_gz,
  sha256_file, file_info, _tile_opens 를 그대로 부른다. 실행 중 교체: LB.v3_lena → 알래스카 v3 행, LB.TAG_QA → map_alaska_qa,
  LB.stage_cov → macro 를 'Alaska' 로 두는 같은 순서의 함수, LB.GSW_TILES → 알래스카 GSW 타일, LB.PFR_MIN → 50.

산출(--out-dir, 기본 data/processed/map_alaska)
  alaska_grid_cells_v1.csv, alaska_grid_dem_tiles_v1.csv, cov_parts/map_alaska_{grid,qa}_{dem,e5,cci,sg}.csv, alaska_grid_mask_v1.csv,
  alaska_grid_qa_v1.csv, alaska_grid_x25_v1.csv.gz, alaska_grid_x25_v1_meta.json

실행(ROOT, 공유 서버 규칙: 계산 스레드 1, nice 10)
  nice -n 10 python3 scripts/1_data_prep/build_map_grid_alaska_v1.py
  --limit N: 영역 셀 가운데 N개만(시험용, --out-dir 를 따로 준다). --count-only: 격자 셀 수와 필요한 타일 수만 출력한다.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "1"
    else:
        os.environ.setdefault(_v, "1")

import argparse                                                                                         # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import subprocess                                                                                       # noqa: E402
import sys                                                                                              # noqa: E402
import time                                                                                             # noqa: E402
import urllib.error                                                                                     # noqa: E402
import urllib.request                                                                                   # noqa: E402
from concurrent.futures import ThreadPoolExecutor                                                       # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
for _p in (str(ROOT / "src"), str(ROOT / "scripts" / "1_data_prep")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LB = _load("build_map_grid_lena_v1", "scripts/1_data_prep/build_map_grid_lena_v1.py")

# ---------------------------------------------------------------- 고정 설계값(지시 2026-10-05)
BOX = dict(lat0=59.0, lat1=71.5, lon0=-168.0, lon1=-141.0)
CELL_DEG = LB.CELL_DEG
PFR_MIN = 50.0                                                     # 'PFR ≥ 0.5' = 50 %(파일 단위 %)
PFR_ALT = (10.0, 0.5)                                              # 기록용 대안 문턱(셀 수만)
MACRO = "Alaska"
V3_REGIONS = ("ABoVE_AK", "United States (Alaska)")
GSW_TILES_AK = ("occurrence_170W_80Nv1_4_2021.tif", "occurrence_160W_80Nv1_4_2021.tif", "occurrence_150W_80Nv1_4_2021.tif",
                "occurrence_170W_70Nv1_4_2021.tif", "occurrence_160W_70Nv1_4_2021.tif", "occurrence_150W_70Nv1_4_2021.tif")
TAG_GRID = "map_alaska_grid"
TAG_QA = "map_alaska_qa"
GRID_VERSION = "v1"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{nm}/{nm}.tif"
DL_WORKERS = 4
SG_NEW_PREFIX = "map_alaska_v1"
SG_META = RAW / "soilgrids_wcs" / "windows_wcs_meta_map_alaska_v1.json"
log = LB.log


# ================================================================ 재사용 함수의 실행 중 교체
def v3_alaska() -> pd.DataFrame:
    """v3 의 알래스카 F4_direct 행(레나 v3_lena 와 같은 형식, id = loc_id)."""
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
    A = v3[v3.region.isin(V3_REGIONS) & (v3.source_id == "F4_direct")].copy()
    A["id"] = A.loc_id.astype(str)
    return A.reset_index(drop=True)


def stage_cov(M, cells: pd.DataFrame, tag: str, stages=("dem", "e5", "cci", "sg")) -> dict:
    """레나 stage_cov 와 같은 순서·같은 단계 함수. macro 만 'Alaska' 로 둔다(stage_sg 의 기록 줄에만 쓰인다)."""
    cell = cells.rename(columns={"cell_id": "id"})[["id", "lat", "lon"]].copy() if "cell_id" in cells else cells[["id", "lat", "lon"]].copy()
    cell["macro"] = MACRO
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
            M.stage_sg(cell, tag, allow_download=False)
            timing["sg_new_windows"] = sg_fill_new_windows(M, cell, tag)
        else:
            raise ValueError(st)
        timing[st] = round(time.time() - t0, 1)
    return timing


def patch_lena_module():
    LB.v3_lena = v3_alaska
    LB.TAG_QA = TAG_QA
    LB.stage_cov = stage_cov
    LB.GSW_TILES = GSW_TILES_AK
    LB.PFR_MIN = PFR_MIN


# ================================================================ grid
def ne_land(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """셀 중심이 Natural Earth 10m 육지 다각형 안인가(shapely contains_xy, 상자로 자른 합집합)."""
    import cartopy.io.shapereader as shpreader
    import shapely
    path = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    clip = shapely.box(BOX["lon0"] - 1, BOX["lat0"] - 1, BOX["lon1"] + 1, BOX["lat1"] + 1)
    geoms = [shapely.intersection(g, clip) for g in shpreader.Reader(path).geometries() if g.intersects(clip)]
    land = shapely.union_all([g for g in geoms if not g.is_empty])
    shapely.prepare(land)
    return shapely.contains_xy(land, np.asarray(lon, float), np.asarray(lat, float)), path


def build_domain(limit: int = 0, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    g = LB.build_grid(box=BOX)
    n_box = len(g)
    pfr = LB.pfr_at(g.lat.values, g.lon.values)
    land, ne_path = ne_land(g.lat.values, g.lon.values)
    fin = np.isfinite(pfr)
    counts = dict(n_box=int(n_box), n_ne_land=int(land.sum()), n_pfr_finite=int(fin.sum()),
                  n_land_pfr_ge50=int((land & fin & (pfr >= PFR_MIN)).sum()),
                  **{f"n_land_pfr_ge{str(t).replace('.', 'p')}": int((land & fin & (pfr >= t)).sum()) for t in PFR_ALT},
                  n_land_pfr_missing=int((land & ~fin).sum()))
    keep = land & fin & (pfr >= PFR_MIN)
    d = g[keep].copy()
    d["pfr_mean"] = pfr[keep]
    d["ne_land"] = 1
    d = d.reset_index(drop=True)
    if limit and limit < len(d):
        d = d.sample(n=int(limit), random_state=seed).sort_values(["ky", "kx"]).reset_index(drop=True)
        counts["limit"] = int(limit)
    counts["n_domain"] = int(len(d))
    return d, dict(counts=counts, natural_earth_land=LB.file_info(ne_path))


# ================================================================ tiles
def _download(nm: str, attempts: int = 3) -> tuple[str, str]:
    """시간 초과 같은 일시 오류는 3회까지 다시 받는다(404 는 바로 absent_public)."""
    st = "error:unknown"
    for k in range(attempts):
        nm_, st = _download_once(nm)
        if not st.startswith("error"):
            return nm_, st
        log(f"  [tiles] {nm}: {st} (시도 {k + 1}/{attempts})")
        time.sleep(5)
    return nm, st


def _download_once(nm: str) -> tuple[str, str]:
    dst = LB.DEM_DIR / (nm + ".tif")
    if dst.exists() and dst.stat().st_size > 1000:
        return nm, "exists"
    tmp = LB.DEM_DIR / (nm + ".tif.part")
    try:
        with urllib.request.urlopen(DEM_URL.format(nm=nm), timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        if dst.exists():                                             # 그 사이 다른 실행이 받았으면 기존 파일을 둔다
            tmp.unlink()
            return nm, "exists"
        os.replace(tmp, dst)
        if not LB._tile_opens(dst):
            dst.unlink()
            return nm, "error:unreadable_download_removed"
        return nm, "downloaded"
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        return nm, "absent_public" if e.code == 404 else f"error:http_{e.code}"
    except Exception as e:                                           # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        return nm, f"error:{str(e)[:60]}"


def stage_tiles(out: Path, cells: pd.DataFrame, tname, count_only=False) -> pd.DataFrame:
    tl = np.floor(cells.lat.values).astype(int); tn = np.floor(cells.lon.values).astype(int)
    key = pd.Series(list(zip(tl, tn)))
    ncell = key.value_counts()
    tiles = sorted(ncell.index)
    rows = [dict(tlat=a, tlon=b, tile=tname(a, b), n_cells=int(ncell[(a, b)])) for a, b in tiles]
    t = pd.DataFrame(rows)
    pres = np.array([(LB.DEM_DIR / (nm + ".tif")).exists() and (LB.DEM_DIR / (nm + ".tif")).stat().st_size > 1000 for nm in t.tile])
    log(f"[tiles] 필요 {len(t)} · 보유 {int(pres.sum())} · 없음 {int((~pres).sum())}")
    if count_only:
        t["status"] = np.where(pres, "exists", "not_checked")
        return t
    status = {nm: "exists" for nm in t.tile[pres]}
    need = list(t.tile[~pres])
    if need:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
            for k, (nm, st) in enumerate(ex.map(_download, need), 1):
                status[nm] = st
                if k % 10 == 0 or k == len(need):
                    log(f"  [tiles] {k}/{len(need)} · {time.time() - t0:.0f}s · 마지막 {nm}: {st}")
    t["status"] = [status[nm] for nm in t.tile]
    t["bytes"] = [int((LB.DEM_DIR / (nm + ".tif")).stat().st_size) if (LB.DEM_DIR / (nm + ".tif")).exists() else 0 for nm in t.tile]
    t["sha256"] = [LB.sha256_file(LB.DEM_DIR / (nm + ".tif")) if (LB.DEM_DIR / (nm + ".tif")).exists() else "" for nm in t.tile]
    t.to_csv(out / "alaska_grid_dem_tiles_v1.csv", index=False)
    log(f"[tiles] {t.status.value_counts().to_dict()}")
    return t


# ================================================================ SoilGrids 새 창(같은 WCS 경로)
def sg_fill_new_windows(M, cell: pd.DataFrame, tag: str) -> dict:
    """기존 창으로 9층이 모두 빈 셀 가운데 기존 창 상자 밖인 셀에 새 창을 받아 채운다. stage_sg 의 새 창 규칙(상자 ±0.3°, lobe 경계,
    igh_bbox, scalesize 5 km, GEOTIFF_INT16 URL, curl 3회)과 같다. 결과는 <tag>_sg.csv 를 고쳐 쓴다(이 과제의 부분 산출)."""
    import enrich_soilgrids_wcs as sgw
    path = M.PARTS / f"{tag}_sg.csv"
    o = pd.read_csv(path, dtype={"id": str})
    assert (o.id.values == cell.id.astype(str).values).all()
    sg_cols = [sgw.col_name(p, d) for p, d in sgw.LAYERS]
    wins = [(w, b) for w, b in sgw.WINDOWS.items()]
    e3m = json.loads((M.SG_RAW / "windows_wcs_meta_e3.json").read_text())
    wins += [(w, tuple(m["wgs84_bbox"])) for w, m in e3m.items()]
    inside = np.zeros(len(cell), bool)
    for _, (lo0, lo1, la0, la1) in wins:
        inside |= ((cell.lon >= lo0) & (cell.lon <= lo1) & (cell.lat >= la0) & (cell.lat <= la1)).values
    miss = o[sg_cols].isna().all(axis=1).values & ~inside
    info = dict(n_outside_existing=int((~inside).sum()), n_fill_candidates=int(miss.sum()), windows={})
    if not miss.any():
        return info
    meta = json.loads(SG_META.read_text()) if SG_META.exists() else {}
    lobe = np.digitize(cell.lon.values, [-40, 20, 100])
    ix_x, ix_y = sgw.TR.transform(cell.lon.values, cell.lat.values)
    ix_x, ix_y = np.asarray(ix_x), np.asarray(ix_y)
    src = o["sg_windows"].fillna("").astype(str).values.copy()
    for lb_ in np.unique(lobe[miss]):
        idx = np.where(miss & (lobe == lb_))[0]
        g = cell.iloc[idx]
        wname = f"{SG_NEW_PREFIX}_{lb_}"
        lo0, lo1 = float(g.lon.min() - 0.3), float(g.lon.max() + 0.3)
        la0, la1 = float(g.lat.min() - 0.3), float(g.lat.max() + 0.3)
        bounds = {0: (-180, -40), 1: (-40, 20), 2: (20, 100), 3: (100, 180)}[int(lb_)]
        lo0, lo1 = max(lo0, bounds[0] + 0.01), min(lo1, bounds[1] - 0.01)
        x0, x1, y0, y1 = sgw.igh_bbox(lo0, lo1, la0, la1)
        nx, ny = sgw.scalesize(x0, x1, y0, y1)
        wdir = M.SG_RAW / wname
        wdir.mkdir(parents=True, exist_ok=True)
        mm = meta.setdefault(wname, dict(wgs84_bbox=[lo0, lo1, la0, la1], igh_bbox=[round(x0), round(x1), round(y0), round(y1)],
                                         scalesize=[nx, ny], n_cells=int(len(g)), layers={}))
        log(f"[sg] 새 창 {wname}: lon[{lo0:.2f},{lo1:.2f}] lat[{la0:.2f},{la1:.2f}] size {nx}x{ny} · 셀 {len(g)}")
        for prop, dpt in sgw.LAYERS:
            p = wdir / f"{prop}_{dpt}_mean.tif"
            if p.exists() and sgw._valid_tif(str(p)):
                mm["layers"].setdefault(f"{prop}_{dpt}", "cached")
                continue
            url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1"
                   f"&REQUEST=GetCoverage&COVERAGEID={prop}_{dpt}_mean&FORMAT=GEOTIFF_INT16"
                   f"&SUBSET=X({x0:.0f},{x1:.0f})&SUBSET=Y({y0:.0f},{y1:.0f})&SCALESIZE=X({nx}),Y({ny})")
            ok = False
            for _ in range(3):
                rc = subprocess.run(["curl", "-s", "--max-time", "300", "-o", str(p), url]).returncode
                if rc == 0 and sgw._valid_tif(str(p)):
                    ok = True
                    break
                time.sleep(3)
            mm["layers"][f"{prop}_{dpt}"] = ("ok " + time.strftime("%Y-%m-%d")) if ok else "failed"
            if not ok and p.exists():
                p.unlink()
            log(f"      {prop}_{dpt}: {'OK' if ok else 'FAILED'}")
        got = 0
        for prop, dpt in sgw.LAYERS:
            p = wdir / f"{prop}_{dpt}_mean.tif"
            if not p.exists():
                continue
            s = sgw._sample(str(p), ix_x[idx], ix_y[idx]) / sgw.SCALE.get(prop, (1.0, ""))[0]
            col = sgw.col_name(prop, dpt)
            cur = o[col].values.astype(float)
            fill = np.isfinite(s) & np.isnan(cur[idx])
            cur[idx[fill]] = s[fill]
            o[col] = cur
            for k in idx[fill]:
                if wname not in src[k].split(";"):
                    src[k] = (src[k] + ";" + wname).strip(";")
            got += int(fill.sum())
        info["windows"][wname] = dict(n_cells=int(len(g)), n_values_filled=int(got), bbox=[lo0, lo1, la0, la1], scalesize=[nx, ny])
        log(f"      채움 {got}")
    o["sg_windows"] = src
    o.to_csv(path, index=False)
    SG_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return info


# ================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="지도 과제(MAP) 알래스카 1 km 격자 조립·QA·마스크")
    ap.add_argument("--stage", default="grid,tiles,cov,qa,mask,merge")
    ap.add_argument("--out-dir", default="data/processed/map_alaska")
    ap.add_argument("--limit", type=int, default=0, help="영역 셀 가운데 N개만(시험용)")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--note", default="")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()
    out = Path(a.out_dir) if os.path.isabs(a.out_dir) else ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    parts = out / "cov_parts"
    stages = [s for s in a.stage.split(",") if s]
    patch_lena_module()
    M = LB.cov_module(parts)
    tname = M.tname
    timing = {}
    cells_path = out / "alaska_grid_cells_v1.csv"
    dom_path = out / "alaska_grid_domain_v1.json"

    if "grid" in stages or a.count_only:
        t0 = time.time()
        g, dom = build_domain(a.limit)
        timing["grid"] = round(time.time() - t0, 1)
        log(f"[grid] {json.dumps(dom['counts'], ensure_ascii=False)} · {timing['grid']}s")
        if a.count_only:
            stage_tiles(out, g, tname, count_only=True)
            return dict(n_cells=int(len(g)))
        g.to_csv(cells_path, index=False, float_format="%.10g")
        dom_path.write_text(json.dumps(dom, ensure_ascii=False, indent=1))
    g = pd.read_csv(cells_path, dtype={"cell_id": str})
    dom = json.loads(dom_path.read_text())

    tiles_path = out / "alaska_grid_dem_tiles_v1.csv"
    if "tiles" in stages:
        t0 = time.time()
        stage_tiles(out, g, tname)
        timing["tiles"] = round(time.time() - t0, 1)
    tiles = pd.read_csv(tiles_path)

    if "cov" in stages:
        timing["cov"] = stage_cov(M, g, TAG_GRID)
        log(f"[cov] {timing['cov']}")

    gx = None
    if any(s in stages for s in ("qa", "mask", "merge")):
        gx = LB.merge_parts(M, g, TAG_GRID).rename(columns={"id": "cell_id"})
        gx = g[["cell_id", "ky", "kx", "pfr_mean", "ne_land"]].merge(gx, on="cell_id", validate="one_to_one")

    qa_extra = {}
    qa_path = out / "alaska_grid_qa_v1.csv"
    if "qa" in stages:
        t0 = time.time()
        qa, qa_extra = LB.stage_qa(M, out, gx, None)
        qa.to_csv(qa_path, index=False)
        (out / "alaska_grid_qa_extra_v1.json").write_text(json.dumps(qa_extra, ensure_ascii=False, indent=1, default=str))
        timing["qa"] = round(time.time() - t0, 1)
        prim = qa[(qa.kind == "v3_rerun") & qa.group.isin(["dem", "e5", "sg", "cci", "soil_tdd"])]
        log(f"[qa] 대조 열 {len(prim)} · 통과 {int(prim['pass'].sum())} · 최대 불일치 비율 {prim.frac_mismatch.max():.4f}")
        ok_ = prim["pass"].astype(bool)
        if not ok_.all():
            log("[qa] 통과하지 못한 열:\n" + prim[~ok_][["group", "column", "frac_mismatch", "n_nan_mismatch", "n_over_tol", "max_abs_diff"]].to_string())
    elif (out / "alaska_grid_qa_extra_v1.json").exists():
        qa_extra = json.loads((out / "alaska_grid_qa_extra_v1.json").read_text())

    mpath = out / "alaska_grid_mask_v1.csv"
    if "mask" in stages:
        t0 = time.time()
        mk = LB.build_masks(gx.drop(columns=["pfr_mean"]), tiles, tname)
        mk.to_csv(mpath, index=False, float_format="%.6g")
        timing["mask"] = round(time.time() - t0, 1)
        log(f"[mask] {json.dumps(LB.mask_summary(mk), ensure_ascii=False)} · {timing['mask']}s")

    if "merge" in stages:
        t0 = time.time()
        mk = pd.read_csv(mpath, dtype={"cell_id": str})
        assert np.allclose(mk.pfr_mean.values, gx.pfr_mean.values, equal_nan=True), "PFR 값이 격자 단계와 다르다"
        full = gx.drop(columns=["pfr_mean"]).reset_index(drop=True).merge(mk, on="cell_id", validate="one_to_one")
        x25 = [c for cols in LB.x25_groups().values() for c in cols]
        keep = (["cell_id", "ky", "kx", "lat", "lon"] + x25 + ["e5_tdd_soil", "e5_sqrt_tdd_soil", "soil_tdd_le0", "e5_fallback_deg", "soil_fallback_deg",
                "cci_n_years", "sg_windows", "dem_tile_missing", "ne_land"] + [c for c in mk.columns if c != "cell_id"])
        full = full[keep]
        ms = LB.mask_summary(full)
        full = full.rename(columns={"pfr_lt10": "pfr_lt50"})
        full["gray_reason"] = full.gray_reason.fillna("").replace({"pfr_lt10": "pfr_lt50"})
        gpath = out / f"alaska_grid_x25_{GRID_VERSION}.csv.gz"
        LB.to_csv_gz(full, gpath)
        L = v3_alaska()
        qa_old = pd.read_csv(qa_path) if qa_path.exists() else pd.DataFrame()
        rng = LB.grid_summary_rows(full.assign(p4_ku=np.nan, p2_edaphic=np.nan), L)
        rng = rng[rng.group != "phys"]
        qa_all = pd.concat([qa_old[qa_old.kind != "grid_range"] if len(qa_old) else qa_old, rng], ignore_index=True)
        qa_all.to_csv(qa_path, index=False)
        prim = qa_all[(qa_all.kind == "v3_rerun") & qa_all.group.isin(["dem", "e5", "sg", "cci", "soil_tdd"])] if len(qa_all) else qa_all
        x25_valid = full[x25].notna().all(axis=1)
        sg_used = sorted({w for x in full.sg_windows.fillna("") for w in str(x).split(";") if w})
        timing["merge"] = round(time.time() - t0, 1)
        no_gsw = ~np.isfinite(full.gsw_occ_mean.values.astype(float))
        meta = dict(
            created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/1_data_prep/build_map_grid_alaska_v1.py",
            script_sha256=LB.sha256_file(Path(__file__)), reused_script=LB.file_info(ROOT / "scripts/1_data_prep/build_map_grid_lena_v1.py"),
            instruction="2026-10-05 지시: 알래스카 1 km 지도(육지, PFR ≥ 0.5, 59–71.5°N, 168–141°W, 0.009° 셀 규칙, x25 는 라벨 표와 같은 추출)",
            grid=dict(box=BOX, cell_deg=CELL_DEG, rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009; 셀 중심이 상자 안 · "
                      "Natural Earth 10m 육지 안 · PFR(0.1° 최근접) ≥ 50 %", pfr_threshold_pct=PFR_MIN,
                      pfr_interpretation="지시의 PFR ≥ 0.5 를 비율 0.5 = 50 % 로 읽었다(파일 단위 %). 대안 문턱의 셀 수는 domain_counts",
                      domain_counts=dom["counts"], natural_earth_land=dom["natural_earth_land"], n_cells=int(len(full)),
                      n_rows_ky=int(full.ky.nunique()), cell_index_fn="scripts/1_data_prep/build_ext_cells_v1.py:cell_index"),
            covariates=dict(module="scripts/1_data_prep/ext_cells_covariates_v1.py", module_sha256=LB.sha256_file(ROOT / "scripts/1_data_prep/ext_cells_covariates_v1.py"),
                            stages=["stage_dem", "stage_e5", "stage_cci", "stage_sg(allow_download=False)", "새 SoilGrids 창(같은 WCS 경로)"],
                            parts_dir=str(parts.relative_to(ROOT)), tags=[TAG_GRID, TAG_QA], era5land=LB.file_info(M.NC),
                            cci_alt_files={Path(f).name: LB.sha256_file(f) for f in sorted(M.CCI_DIR.glob("*.nc"))},
                            soilgrids_windows=sg_used,
                            soilgrids_files={f"{w}/{f.name}": LB.sha256_file(f) for w in sg_used for f in sorted((M.SG_RAW / w).glob("*.tif"))},
                            soilgrids_new_window_meta=LB.file_info(SG_META),
                            sg_new_windows=(timing.get("cov") or {}).get("sg_new_windows"),
                            soil_tdd_le0_set_nan=int(full.soil_tdd_le0.sum()),
                            e5_fallback=full.e5_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict(),
                            soil_fallback=full.soil_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict(),
                            effective_resolution="ERA5-Land 0.1°(기후 8열·토양 도일·s), SoilGrids 창 약 5 km(9열), CCI ALT 약 1 km(2열), DEM 30 m 창(6열)"),
            dem_tiles=tiles.drop(columns=["sha256"], errors="ignore").to_dict("records"), dem_tiles_sha256=dict(zip(tiles.tile, tiles.get("sha256", [""] * len(tiles)))),
            dem_tiles_status_counts=tiles.status.value_counts().to_dict(), dem_url=DEM_URL,
            masks=dict(pfr=dict(file=LB.file_info(LB.PFR_NC), rule="영역 정의에서 PFR ≥ 50 % 만 둔다. 열 pfr_lt50 은 항상 0"),
                       water=dict(files={f: LB.file_info(LB.GSW_DIR / f) for f in GSW_TILES_AK},
                                  rule="셀 안 30 m 화소(255 제외)의 평균 발생 빈도 ≥ 50 % 이면 회색(water). 레나와 같은 해석",
                                  n_no_gsw_coverage=int(no_gsw.sum()),
                                  note="59–60°N 띠와 GSW 유효 화소가 없는 셀은 수체를 판정하지 않는다(회색 아님)"),
                       sea_input="DEM 타일 없음(dem_tile_absent), cci_alt 결측, e5_sqrt_tdd_soil 결측, e5_sqrt_tdd 결측이면 회색(레나와 같다)",
                       land_rule="격자 = Natural Earth 10m 육지 안. land 열(보고용)은 레나 정의(DEM 유효 · GSW 유효 · 수체 아님)",
                       gray_reason_order=[r.replace("pfr_lt10", "pfr_lt50") for r in LB.GRAY_REASONS], summary=ms),
            validity=dict(x25_all_valid_frac=float(x25_valid.mean()),
                          x25_all_valid_frac_shown=float(x25_valid[full.gray == 0].mean()) if (full.gray == 0).any() else None,
                          missing_by_col={c: int(full[c].isna().sum()) for c in x25 + ["e5_sqrt_tdd_soil"]}),
            qa=dict(file=str(qa_path.relative_to(ROOT)), tolerance=dict(abs=LB.TOL_ABS, rel_dem=LB.TOL_REL_DEM, pass_frac=LB.PASS_FRAC),
                    n_cols=int(len(prim)), n_pass=int(prim["pass"].astype(bool).sum()) if len(prim) else 0,
                    passed=bool(len(prim) and prim["pass"].astype(bool).all()), max_frac_mismatch=float(prim.frac_mismatch.max()) if len(prim) else None,
                    **qa_extra),
            inputs=dict(fidelity_base_v3=LB.file_info(PROC / "fidelity_base_v3.csv"), e5_soil_tdd_v3=LB.file_info(PROC / "e5_soil_tdd_v3.csv")),
            outputs={k: LB.file_info(p) for k, p in dict(grid=gpath, cells=cells_path, tiles=tiles_path, mask=mpath, qa=qa_path).items()},
            cov_parts={p.name: LB.file_info(p) for p in sorted(parts.glob("map_alaska_*.csv"))},
            timing_s=timing, elapsed_s=round(time.time() - t_start, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), nice=os.nice(0), dem_download_workers=DL_WORKERS),
            stages_run=stages, run_note=a.note)
        mpath_meta = out / f"alaska_grid_x25_{GRID_VERSION}_meta.json"
        if mpath_meta.exists():
            try:
                old = json.loads(mpath_meta.read_text())
                meta["previous_meta"] = dict(created=old.get("created"), timing_s=old.get("timing_s"),
                                             grid_sha256=old.get("outputs", {}).get("grid", {}).get("sha256"))
            except (ValueError, OSError):
                meta["previous_meta"] = dict(note="이전 메타를 읽지 못했다")
        mpath_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        log(f"[merge] {gpath} {full.shape} · {meta['outputs']['grid']['bytes']:,} B · sha256 {meta['outputs']['grid']['sha256'][:16]}…")
        log(f"[merge] QA 통과 {meta['qa']['passed']} · x25 모두 유효 {meta['validity']['x25_all_valid_frac']:.3f} · 회색 {ms['n_gray']:,}/{ms['n_cells']:,}")
    log(f"[done] {timing} · 합계 {time.time() - t_start:.0f}s · 최대 RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    return dict(n_cells=int(len(g)))


if __name__ == "__main__":
    main()
