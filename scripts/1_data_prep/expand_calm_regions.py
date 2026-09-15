"""E3: 비북미 CALM 사이트 확충 — 타일 목록(1단계) + 공변량 부착·셀 집계·fidelity_base_v2 조립(2단계).

배경
----
`assemble_dl_dataset.py:45`의 필터 `(lat > 50) & (lon < -100)`로 CALM 비북미 157좌표가 셀 데이터셋에
편입되지 않았다. 계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §3.

1단계 (--tiles-only)
  alt_global.csv에서 비북미 좌표를 뽑아 매크로 지역을 부여하고 alt_calm_new_sites.csv 저장,
  Copernicus DEM 1° 타일을 dem_tiles_needed.csv에 추가(기존 행 보존). 이후 copernicus_dem.py 실행.

2단계 (--attach [--skip-soil])
  a) site-year → 좌표별 셀 집계(alt_cm 평균·alt_sd·min·max·n_obs·year_min/max·n_years).  region = "CALM_"+macro
  b) 기존 fidelity_base 셀과 0.01°(체비쇼프) 이내 근접 좌표 제거(기존 우선)
  c) 지형 6종(DEM 33×33 창, enrich_new_regions.py 로직) · d) ERA5-Land 8종(0.2°→0.5° 육지 폴백)
  e) CCI 2종(25개 연도 파일 최근접 픽셀 다년평균) · f) SoilGrids 9종(기존 WCS 창 → 부족분은 (macro, lobe) 창 신규 다운로드)
  g) fidelity_base_v2.csv = 기존 17,423행 불변 + 신규 행(loc_id 연속, source_id F4_direct, SAR NaN·insar_miss 1)
산출: data/processed/alt_calm_global_cell.csv, fidelity_base_v2.csv, fidelity_base_v2_meta.json,
      data/raw/soilgrids_wcs/e3_*/ (신규 창), windows_wcs_meta_e3.json

실행(ROOT): python3 scripts/1_data_prep/expand_calm_regions.py --tiles-only
            python3 scripts/1_data_prep/expand_calm_regions.py --attach
"""
from __future__ import annotations
import argparse
import calendar
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
DEM = ROOT / "data" / "raw" / "dem"
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2015-2020.nc"
SG_RAW = ROOT / "data" / "raw" / "soilgrids_wcs"
TILES = PROC / "dem_tiles_needed.csv"
SITES = PROC / "alt_calm_new_sites.csv"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "1_data_prep"))

ap = argparse.ArgumentParser()
ap.add_argument("--tiles-only", action="store_true")
ap.add_argument("--attach", action="store_true")
ap.add_argument("--skip-soil", action="store_true")
args = ap.parse_args()


def macro_region(country: str, lat: float, lon: float) -> str:
    c = str(country)
    if "Alaska" in c:
        return "Alaska"
    if c == "Canada":
        return "Canada"
    if "Greenland" in c:
        return "Greenland"
    if "Svalbard" in c:
        return "Svalbard"
    if c in ("Switzerland", "Italy"):
        return "Alps"
    if c in ("Mongolia", "Kazakstan"):
        return "Mongolia_CAsia"
    if c == "China":
        return "QTP_China"
    if c == "Russia":
        if lon < 90:
            return "Russia_W"
        if lon < 140:
            return "Russia_C"
        return "Russia_E"
    return c


d = pd.read_csv(PROC / "alt_global.csv", low_memory=False).dropna(subset=["alt_cm"])
north_america = (d.lat > 50) & (d.lon < -100)          # 기존 조립 필터(assemble_dl_dataset.py:45)
new = d[~north_america].copy()
new["macro"] = [macro_region(c, la, lo) for c, la, lo in zip(new.country, new.lat, new.lon)]

# ---------------------------------------------------------------- 1단계: 사이트·타일
sites = (new.groupby(["lat", "lon"], as_index=False)
            .agg(site=("site", "first"), name=("name", "first"), country=("country", "first"),
                 macro=("macro", "first"), n_siteyear=("alt_cm", "size"),
                 year_min=("year", "min"), year_max=("year", "max"),
                 alt_mean=("alt_cm", "mean"), alt_sd=("alt_cm", "std")))
sites.to_csv(SITES, index=False)
print(f"[sites] 비북미 CALM 좌표 {len(sites)}개 → {SITES.name}")
print(sites.groupby("macro").agg(n=("site", "size"), alt=("alt_mean", "mean")).round(1).to_string())

sites["tlat"] = np.floor(sites.lat).astype(int)
sites["tlon"] = np.floor(sites.lon).astype(int)
need = sites.groupby(["tlat", "tlon"]).size().reset_index(name="n")
old = pd.read_csv(TILES)
oldkey = set(zip(old.tlat.astype(int), old.tlon.astype(int)))
add = need[[(a, b) not in oldkey for a, b in zip(need.tlat, need.tlon)]]
if len(add):
    pd.concat([old, add[["tlat", "tlon", "n"]]], ignore_index=True).to_csv(TILES, index=False)
print(f"[tiles] 필요 {len(need)} · 기존 목록 {len(old)} · 신규 추가 {len(add)}")

if args.tiles_only or not args.attach:
    print("[done] 1단계. 다음: python3 scripts/0_download/copernicus_dem.py → --attach")
    sys.exit(0)

# ================================================================ 2단계
import rasterio            # noqa: E402
import xarray as xr        # noqa: E402
import enrich_soilgrids_wcs as sgw   # noqa: E402  (LAYERS·SCALE·WINDOWS·igh_bbox·scalesize·_sample·_valid_tif·col_name)
from polar.fidelity import add_group_keys, COVARIATE_CORE, FIDELITY, TARGET  # noqa: E402

t0 = time.time()

# --- a) 셀 집계 ---
cell = (new.groupby(["lat", "lon"], as_index=False)
           .agg(alt_cm=("alt_cm", "mean"), alt_sd=("alt_cm", "std"), alt_min=("alt_cm", "min"),
                alt_max=("alt_cm", "max"), n_obs=("alt_cm", "size"), year_min=("year", "min"),
                year_max=("year", "max"), n_years=("year", "nunique"),
                site=("site", "first"), name=("name", "first"), country=("country", "first"),
                macro=("macro", "first")))
cell["region"] = "CALM_" + cell.macro
cell["qc"] = np.nan
cell["qc_flag"] = np.nan
cell["borehole_id"] = np.nan
cell["censor_flag"] = 0
# CALM 탐침 상한(활동층 깊이 150 cm 부근이 반복되면 우측검열 의심)만 플래그. 심부 지역은 융해관·온도 기반이라 예외.
print(f"[cells] {len(cell)} 셀 · n_years 중앙값 {cell.n_years.median():.0f}")

# --- b) 중복 제거(기존 우선) ---
fb = pd.read_csv(PROC / "fidelity_base.csv", low_memory=False)
from scipy.spatial import cKDTree  # noqa: E402
tree = cKDTree(fb[["lat", "lon"]].values)
dist, j = tree.query(cell[["lat", "lon"]].values, p=np.inf)   # 체비쇼프
dup = dist < 0.01
dropped = cell[dup].assign(near_region=fb.region.values[j[dup]], near_dist=dist[dup])
cell = cell[~dup].reset_index(drop=True)
print(f"[dedup] 기존 셀 0.01° 이내 {int(dup.sum())}개 제거 → {len(cell)}셀 유지")
if len(dropped):
    print(dropped[["site", "country", "macro", "near_region", "near_dist"]].round(4).to_string(index=False))

# --- c) 지형 6종 ---
W, H, MPD = 33, 16, 111320.0


def tname(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


cell["tlat"] = np.floor(cell.lat).astype(int)
cell["tlon"] = np.floor(cell.lon).astype(int)
feats = np.full((len(cell), 6), np.nan)
missing_tiles = []
for (tlat, tlon), g in cell.groupby(["tlat", "tlon"]):
    path = DEM / (tname(tlat, tlon) + ".tif")
    if not path.exists():
        missing_tiles.append((int(tlat), int(tlon), len(g)))
        continue
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        arr[arr == ds.nodata] = np.nan
        ny, nx = arr.shape
        dy = MPD * abs(ds.transform.e)
        for idx, r in g.iterrows():
            row, col = ds.index(r.lon, r.lat)
            if -1 <= row <= ny and -1 <= col <= nx:
                row = min(max(row, 0), ny - 1)
                col = min(max(col, 0), nx - 1)
            else:
                continue
            dx = MPD * abs(ds.transform.a) * np.cos(np.radians(r.lat))
            r0, r1 = max(0, row - H), min(ny, row + H + 1)
            c0, c1 = max(0, col - H), min(nx, col + H + 1)
            win = arr[r0:r1, c0:c1]
            if win.size == 0 or np.isnan(win).all():
                continue
            i, jj = row - r0, col - c0
            if np.isnan(win[i, jj]):
                yy, xx = np.where(np.isfinite(win))
                k = np.argmin((yy - i) ** 2 + (xx - jj) ** 2)
                i, jj = int(yy[k]), int(xx[k])
            gy, gx = np.gradient(win, dy, dx)
            sl = np.degrees(np.arctan(np.hypot(gy[i, jj], gx[i, jj])))
            asp = np.arctan2(-gy[i, jj], gx[i, jj])
            cen = win[i, jj]
            feats[idx] = [cen, sl, np.sin(asp), np.cos(asp), cen - np.nanmean(win), float(np.nanstd(win))]
for k, nm in enumerate(["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]):
    cell[nm] = feats[:, k]
print(f"[dem] 유효 {int(np.isfinite(feats[:, 0]).sum())}/{len(cell)} · 미보유 타일 {len(missing_tiles)} {missing_tiles[:8]}")

# --- d) ERA5-Land 8종 ---
ds = xr.open_dataset(NC)
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
t2m = clim["t2m"].values - 273.15
stl = clim["stl1"].values - 273.15
swe = clim["sd"].values
glat, glon = clim["latitude"].values, clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])
land = np.isfinite(t2m).all(axis=0)


def fallback(iy, ix, la_, lo_, nb):
    best = None
    for dy_ in range(-nb, nb + 1):
        for dx_ in range(-nb, nb + 1):
            y2, x2 = iy + dy_, (ix + dx_) % len(glon)
            if 0 <= y2 < len(glat) and land[y2, x2]:
                d2 = (glat[y2] - la_) ** 2 + (glon[x2] - lo_) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, y2, x2)
    return best


e5 = {k: np.full(len(cell), np.nan) for k in
      ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]}
e5_fb = np.zeros(len(cell))
n_fb1 = n_fb2 = n_out = 0
for pos, r in enumerate(cell.itertuples(index=False)):
    if not (glat.min() <= r.lat <= glat.max()):
        n_out += 1
        continue
    iy, ix = int(np.abs(glat - r.lat).argmin()), int(np.abs(glon - r.lon).argmin())
    if not land[iy, ix]:
        best = fallback(iy, ix, r.lat, r.lon, 2)
        if best is not None:
            n_fb1 += 1; e5_fb[pos] = 0.2
        else:
            best = fallback(iy, ix, r.lat, r.lon, 5)
            if best is not None:
                n_fb2 += 1; e5_fb[pos] = 0.5
        if best is None:
            n_out += 1
            continue
        iy, ix = best[1], best[2]
    tm = t2m[:, iy, ix]
    tdd = float(np.nansum(np.clip(tm, 0, None) * days))
    e5["e5_maat"][pos] = float(np.nanmean(tm)); e5["e5_tdd"][pos] = tdd
    e5["e5_fdd"][pos] = float(np.nansum(np.clip(-tm, 0, None) * days))
    e5["e5_sqrt_tdd"][pos] = float(np.sqrt(tdd))
    e5["e5_twarm"][pos] = float(np.nanmax(tm)); e5["e5_tcold"][pos] = float(np.nanmin(tm))
    e5["e5_stl1"][pos] = float(np.nanmean(stl[:, iy, ix])); e5["e5_swe"][pos] = float(np.nanmean(swe[:, iy, ix]))
for k, v in e5.items():
    cell[k] = v
cell["e5_fallback_deg"] = e5_fb
print(f"[e5] 유효 {int(np.isfinite(e5['e5_maat']).sum())}/{len(cell)} (폴백 0.2° {n_fb1} · 0.5° {n_fb2} · 실패 {n_out})")

# --- e) CCI 2종 ---
files = sorted(glob.glob(str(ROOT / "data" / "raw" / "cci_alt" / "*.nc")))
acc = np.zeros(len(cell)); cnt = np.zeros(len(cell))
for f in files:
    dsc = xr.open_dataset(f)
    a = dsc["ALT"]
    if "time" in a.dims:
        a = a.isel(time=0)
    v = a.sel(lat=xr.DataArray(cell.lat.values, dims="p"), lon=xr.DataArray(cell.lon.values, dims="p"),
              method="nearest").values.astype(float)
    ok = np.isfinite(v)
    acc[ok] += v[ok]; cnt[ok] += 1
    dsc.close()
cci = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) * 100.0
cell["cci_alt"] = np.round(cci, 2)
cell["cci_valid"] = np.isfinite(cci).astype(int)
print(f"[cci] 유효 {int(cell.cci_valid.sum())}/{len(cell)} · 연도파일 {len(files)}")

# --- f) SoilGrids 9종 ---
sg_cols = [sgw.col_name(p, dpt) for p, dpt in sgw.LAYERS]
for c in sg_cols:
    cell[c] = np.nan
if not args.skip_soil:
    ix_x, ix_y = sgw.TR.transform(cell.lon.values, cell.lat.values)
    ix_x, ix_y = np.asarray(ix_x), np.asarray(ix_y)

    def sample_window(wname, idx):
        got = 0
        for (prop, dpt) in sgw.LAYERS:
            path = SG_RAW / wname / f"{prop}_{dpt}_mean.tif"
            if not path.exists():
                return 0
            s = sgw._sample(str(path), ix_x[idx], ix_y[idx]) / sgw.SCALE.get(prop, (1.0, ""))[0]
            col = sgw.col_name(prop, dpt)
            cur = cell[col].values
            fill = np.isfinite(s) & np.isnan(cur[idx])
            cur[idx[fill]] = s[fill]
            cell[col] = cur
            got += int(fill.sum())
        return got

    # 기존 창에서 먼저 채움(bbox 내부 셀만)
    for wname, (lo0, lo1, la0, la1) in sgw.WINDOWS.items():
        inb = np.where((cell.lon >= lo0) & (cell.lon <= lo1) & (cell.lat >= la0) & (cell.lat <= la1))[0]
        if len(inb):
            g = sample_window(wname, inb)
            print(f"[sg] 기존 창 {wname}: 후보 {len(inb)} · 채움 {g}")
    # 부족분: (macro, lobe) 창 신규 다운로드
    lobe = np.digitize(cell.lon.values, [-40, 20, 100])
    miss = cell[sg_cols].isna().any(axis=1).values
    meta_e3 = {}
    for (mac, lb), g in cell[miss].groupby([cell.macro[miss], lobe[miss]]):
        wname = f"e3_{mac}_{lb}"
        lo0, lo1 = float(g.lon.min() - 0.3), float(g.lon.max() + 0.3)
        la0, la1 = float(g.lat.min() - 0.3), float(g.lat.max() + 0.3)
        bounds = {0: (-180, -40), 1: (-40, 20), 2: (20, 100), 3: (100, 180)}[lb]
        lo0, lo1 = max(lo0, bounds[0] + 0.01), min(lo1, bounds[1] - 0.01)
        x0, x1, y0, y1 = sgw.igh_bbox(lo0, lo1, la0, la1)
        nx, ny = sgw.scalesize(x0, x1, y0, y1)
        wdir = SG_RAW / wname
        wdir.mkdir(parents=True, exist_ok=True)
        meta_e3[wname] = dict(wgs84_bbox=[lo0, lo1, la0, la1], scalesize=[nx, ny], n_cells=int(len(g)), layers={})
        print(f"[sg] 신규 창 {wname}: lon[{lo0:.1f},{lo1:.1f}] lat[{la0:.1f},{la1:.1f}] size {nx}x{ny} · 셀 {len(g)}")
        for prop, dpt in sgw.LAYERS:
            path = wdir / f"{prop}_{dpt}_mean.tif"
            if path.exists() and sgw._valid_tif(str(path)):
                meta_e3[wname]["layers"][f"{prop}_{dpt}"] = "cached"
                continue
            url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1"
                   f"&REQUEST=GetCoverage&COVERAGEID={prop}_{dpt}_mean&FORMAT=GEOTIFF_INT16"
                   f"&SUBSET=X({x0:.0f},{x1:.0f})&SUBSET=Y({y0:.0f},{y1:.0f})&SCALESIZE=X({nx}),Y({ny})")
            ok = False
            for attempt in range(3):
                rc = subprocess.run(["curl", "-s", "--max-time", "180", "-o", str(path), url]).returncode
                if rc == 0 and sgw._valid_tif(str(path)):
                    ok = True
                    break
                time.sleep(3)
            meta_e3[wname]["layers"][f"{prop}_{dpt}"] = "ok" if ok else "failed"
            if not ok and path.exists():
                path.unlink()
            print(f"      {prop}_{dpt}: {'OK' if ok else 'FAILED'}")
        got = sample_window(wname, g.index.values)
        print(f"      채움 {got}")
    (SG_RAW / "windows_wcs_meta_e3.json").write_text(json.dumps(meta_e3, ensure_ascii=False, indent=1))
valid_sg = cell[sg_cols].notna().all(axis=1)
print(f"[sg] 9종 전부 유효 {int(valid_sg.sum())}/{len(cell)} · 지역별 " +
      str(cell.groupby('macro').apply(lambda g: int(g[sg_cols].notna().all(axis=1).sum())).to_dict()))

# --- g) 조립 ---
cell = add_group_keys(cell)
cell["loc_id"] = np.arange(len(cell)) + int(fb.loc_id.max()) + 1
cell["source_id"] = "F4_direct"
cell["fidelity_level"] = FIDELITY["F4_direct"]
cell["spatial_support_m"] = 100.0
cell["sigma_prior_cm"] = pd.to_numeric(cell.alt_sd, errors="coerce").fillna(12.0).clip(3, 40)
cell["right_censored"] = 0
for c in ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n", "polsar_alt", "polsar_std"]:
    cell[c] = np.nan
cell["polsar_valid"] = 0.0
cell["insar_miss"] = 1

qc_out = PROC / "alt_calm_global_cell.csv"
cell.drop(columns=["tlat", "tlon"]).to_csv(qc_out, index=False)

newrows = pd.DataFrame({c: (cell[c].values if c in cell.columns else np.nan) for c in fb.columns})
v2 = pd.concat([fb, newrows], ignore_index=True)
assert len(v2) == len(fb) + len(cell)
assert v2.loc_id.is_unique
core_missing = [c for c in COVARIATE_CORE if c not in v2.columns]
assert not core_missing, core_missing
v2.to_csv(PROC / "fidelity_base_v2.csv", index=False)

old_hash = hashlib.md5(pd.util.hash_pandas_object(fb, index=False).values).hexdigest()[:12]
v2_head_hash = hashlib.md5(pd.util.hash_pandas_object(v2.iloc[:len(fb)], index=False).values).hexdigest()[:12]
meta = dict(stage="E3-2", created="2026-09-08", n_old=int(len(fb)), n_new=int(len(cell)), n_total=int(len(v2)),
            old_rows_unchanged=(old_hash == v2_head_hash), old_hash=old_hash,
            dedup_dropped=int(dup.sum()), missing_dem_tiles=missing_tiles,
            e5_fallback={"0.2": n_fb1, "0.5": n_fb2, "fail": n_out},
            coverage_by_region={
                reg: dict(n=int(len(g)), dem=int(g.dem_elev.notna().sum()), e5=int(g.e5_maat.notna().sum()),
                          cci=int(g.cci_valid.sum()), sg9=int(g[sg_cols].notna().all(axis=1).sum()),
                          alt_mean=round(float(g.alt_cm.mean()), 1), n_blocks=int(g.block.nunique()))
                for reg, g in cell.groupby("region")},
            note="fidelity_base.csv 불변. 신규 region은 'CALM_'+macro, MACRO_REGION(fidelity.py)에 매핑. "
                 "SAR 공변량 NaN·insar_miss=1(신규 지역 규약 동일). 심부 레짐(몽골·알프스·QTP)은 TRANSFER_DEEP.",
            plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §3", elapsed_s=round(time.time() - t0, 1))
(PROC / "fidelity_base_v2_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(json.dumps(meta["coverage_by_region"], ensure_ascii=False, indent=1))
print(f"[save] fidelity_base_v2.csv {v2.shape} · 기존 행 불변 {meta['old_rows_unchanged']} · {time.time()-t0:.0f}s")
