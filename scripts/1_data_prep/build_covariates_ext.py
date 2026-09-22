"""H18·H19 공변량 확장표 조립 → data/processed/covariates_ext_v1.csv (키 loc_id; 17,572 셀 전부).

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §1.1·§2.1. 원자료는 scripts/1_data_prep/fetch_ext_covariates.py.
군별 산출 열(단위)
  LST(H18, MOD11C3 Terra 월별 0.05°, 2015–2020 월 기후값, 주·야 평균)
    lst_tdd(°C·day) lst_fdd(°C·day) lst_sqrt_tdd lst_nfactor(=lst_tdd/e5_tdd) lst_amp(°C, 최난월−최한월) lst_nclear_summer(일/월, 6–8월 청천 일수 평균)
    lst_day_tdd lst_night_tdd(보조) lst_fallback(0 직접, 1 = 3×3, 2 = 5×5 창 평균 폴백)
  T(지형 수문, Copernicus DEM 30 m → 90 m 평균 다운샘플, pysheds D8): twi(1 km 평균) twi_sd flowacc_log(1 km 평균 log10 누적 셀 수) curv_plan curv_prof(×1e3 /m, 90 m 평활) rel_1km(m)
  W(수체, JRC GSW occurrence 2021): water_occ_1km water_occ_5km(%, 창 평균) water_dist_km(occurrence>50% 픽셀까지, 5 km 상한)
  V(식생): treecover_1km(%, Hansen 2000) ndvi_summer ndvi_max evi_summer ndwi_summer(MOD13C2 월 기후값; NDWI = (NIR−MIR)/(NIR+MIR))
  H(ERA5-Land 확장 월평균 2015–2020): e5_swvl1_summer e5_swvl2_summer(m³/m³) e5_skt_tdd e5_skt_fdd(°C·day) e5_skt_nfactor e5_lai_summer e5_rsn_winter(kg/m³) e5_sde_winter(m) e5_tp_summer e5_tp_annual(mm)
실행(ROOT): python3 scripts/1_data_prep/build_covariates_ext.py --lst --ndvi --terrain --water --trees --era5 --merge [--workers 4]
"""
from __future__ import annotations
import argparse
import calendar
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
PART = PROC / "covariates_ext_parts"; PART.mkdir(exist_ok=True)
ap = argparse.ArgumentParser()
for k in ("lst", "ndvi", "terrain", "water", "trees", "era5", "merge"):
    ap.add_argument(f"--{k}", action="store_true")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--years", default="2015-2020")
args = ap.parse_args()
Y0, Y1 = [int(v) for v in args.years.split("-")]
DAYS = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)], float)

CELLS = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "e5_tdd"])
LAT, LON = CELLS.lat.values, CELLS.lon.values
N = len(CELLS)
print(f"[cells] {N:,}", flush=True)


# ---------------------------------------------------------------- 공통: CMG 0.05° 샘플(폴백 창)
def cmg_rc(lat, lon, res=0.05):
    r = np.clip(np.floor((90.0 - lat) / res).astype(int), 0, int(180 / res) - 1)
    c = np.clip(np.floor((lon + 180.0) / res).astype(int), 0, int(360 / res) - 1)
    return r, c


def sample_window(arr, r, c, half):
    """arr(float, NaN=결측)에서 (r,c) 중심 (2·half+1)² 창 nanmean. half=0 이면 중심 픽셀."""
    if half == 0:
        return arr[r, c]
    out = np.full(len(r), np.nan)
    H, W = arr.shape
    for i in range(len(r)):
        w = arr[max(0, r[i] - half):min(H, r[i] + half + 1), max(0, c[i] - half):min(W, c[i] + half + 1)]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out


def sample_fallback(arr, r, c):
    v = sample_window(arr, r, c, 0); fb = np.zeros(len(r), int)
    for half, code in ((1, 1), (2, 2)):
        m = ~np.isfinite(v)
        if not m.any():
            break
        v[m] = sample_window(arr, r[m], c[m], half); fb[m & np.isfinite(v)] = code
    return v, fb


def monthly_clim(files_by_month, reader):
    """{month: [files]} → 셀별 12개월 기후값 (N,12) + 폴백 코드 최대."""
    r, c = cmg_rc(LAT, LON)
    clim = np.full((N, 12), np.nan); fbmax = np.zeros(N, int); nclr = np.full((N, 12), np.nan)
    for m in range(1, 13):
        vals, clrs = [], []
        for f in files_by_month.get(m, []):
            a, clr = reader(f)
            v, fb = sample_fallback(a, r, c); fbmax = np.maximum(fbmax, fb); vals.append(v)
            if clr is not None:
                clrs.append(sample_window(clr, r, c, 0))
        if vals:
            clim[:, m - 1] = np.nanmean(np.stack(vals), 0)
        if clrs:
            nclr[:, m - 1] = np.nanmean(np.stack(clrs), 0)
    return clim, fbmax, nclr


def degree_days(clim):
    tdd = np.nansum(np.clip(clim, 0, None) * DAYS, 1); fdd = np.nansum(np.clip(-clim, 0, None) * DAYS, 1)
    ok = np.isfinite(clim).sum(1) >= 10                     # 10개월 이상 있어야 도일 산출
    tdd[~ok] = np.nan; fdd[~ok] = np.nan
    return tdd, fdd


def modis_files(prefix):
    out = {}
    for f in sorted((RAW / "modis_cmg").glob(f"{prefix}.A*.hdf")):
        doy = int(f.name.split(".")[1][5:8]); yr = int(f.name.split(".")[1][1:5])
        if not (Y0 <= yr <= Y1):
            continue
        mth = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=doy - 1)
        out.setdefault(mth.month, []).append(f)
    return out


# ---------------------------------------------------------------- LST (MOD11C3)
def read_lst(f):
    from pyhdf.SD import SD, SDC
    s = SD(str(f), SDC.READ)
    def get(name):
        d = s.select(name); a = d.get().astype(float); att = d.attributes()
        a[a == att.get("_FillValue", 0)] = np.nan
        a = a * att.get("scale_factor", 1.0) + att.get("add_offset", 0.0)
        return a - 273.15
    day, night = get("LST_Day_CMG"), get("LST_Night_CMG")
    both = np.where(np.isfinite(day) & np.isfinite(night), 0.5 * (day + night), np.nan)   # 주·야 둘 다 있을 때만 일평균 대리
    clr = s.select("Clear_sky_days").get().astype(np.uint32)
    nclr = np.unpackbits(clr.view(np.uint8).reshape(clr.shape + (4,)), axis=-1).sum(-1).astype(float)
    return both, nclr, day, night


if args.lst:
    t0 = time.time(); fm = modis_files("MOD11C3")
    print(f"[lst] MOD11C3 파일 {sum(len(v) for v in fm.values())} · 월 {sorted(fm)}", flush=True)
    cache = {}
    def rd(f):
        if f not in cache:
            cache.clear(); cache[f] = read_lst(f)
        return cache[f]
    clim, fb, nclr = monthly_clim(fm, lambda f: rd(f)[:2])
    clim_d, _, _ = monthly_clim(fm, lambda f: (rd(f)[2], None))
    clim_n, _, _ = monthly_clim(fm, lambda f: (rd(f)[3], None))
    tdd, fdd = degree_days(clim); tdd_d, _ = degree_days(clim_d); tdd_n, _ = degree_days(clim_n)
    out = pd.DataFrame(dict(loc_id=CELLS.loc_id, lst_tdd=tdd, lst_fdd=fdd, lst_sqrt_tdd=np.sqrt(np.clip(tdd, 0, None)),
                            lst_nfactor=tdd / np.where(CELLS.e5_tdd.values > 0, CELLS.e5_tdd.values, np.nan),
                            lst_amp=np.nanmax(clim, 1) - np.nanmin(clim, 1), lst_nclear_summer=np.nanmean(nclr[:, 5:8], 1),
                            lst_day_tdd=tdd_d, lst_night_tdd=tdd_n, lst_fallback=fb))
    for m in range(12):
        out[f"lst_m{m+1:02d}"] = clim[:, m]
    out.to_csv(PART / "lst.csv", index=False)
    print(f"[lst] 결측 tdd {np.isnan(tdd).mean():.3f} · 폴백 {np.bincount(fb, minlength=3)} · {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- NDVI/EVI/NDWI (MOD13C2)
def read_vi(f):
    from pyhdf.SD import SD, SDC
    s = SD(str(f), SDC.READ)
    def get(name):
        d = s.select(name); a = d.get().astype(float); att = d.attributes()
        a[a == att.get("_FillValue")] = np.nan
        lo, hi = att.get("valid_range", (-np.inf, np.inf)); a[(a < lo) | (a > hi)] = np.nan
        return a / att.get("scale_factor", 1.0)
    ndvi, evi = get("CMG 0.05 Deg Monthly NDVI"), get("CMG 0.05 Deg Monthly EVI")
    nir, mir = get("CMG 0.05 Deg Monthly NIR reflectance"), get("CMG 0.05 Deg Monthly MIR reflectance")
    ndwi = (nir - mir) / np.where((nir + mir) > 0, nir + mir, np.nan)
    return ndvi, evi, ndwi


if args.ndvi:
    t0 = time.time(); fm = modis_files("MOD13C2")
    print(f"[ndvi] MOD13C2 파일 {sum(len(v) for v in fm.values())}", flush=True)
    cache = {}
    def rd(f):
        if f not in cache:
            cache.clear(); cache[f] = read_vi(f)
        return cache[f]
    c_ndvi, fb, _ = monthly_clim(fm, lambda f: (rd(f)[0], None))
    c_evi, _, _ = monthly_clim(fm, lambda f: (rd(f)[1], None))
    c_ndwi, _, _ = monthly_clim(fm, lambda f: (rd(f)[2], None))
    out = pd.DataFrame(dict(loc_id=CELLS.loc_id, ndvi_summer=np.nanmean(c_ndvi[:, 5:8], 1), ndvi_max=np.nanmax(c_ndvi, 1),
                            evi_summer=np.nanmean(c_evi[:, 5:8], 1), ndwi_summer=np.nanmean(c_ndwi[:, 5:8], 1), vi_fallback=fb))
    out.to_csv(PART / "ndvi.csv", index=False)
    print(f"[ndvi] 결측 {np.isnan(out.ndvi_summer).mean():.3f} · {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 지형 수문 (Copernicus DEM 30 m, pysheds)
def tname(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return RAW / "dem" / f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"


def terrain_tile(task):
    """1° 타일: DEM 을 약 90 m 로 평균 다운샘플(30 m 함몰 채움은 타일당 수십 분 → 90 m 5 s) 후 pysheds D8 로 TWI·유량 누적,
    numpy 로 곡률·기복. 셀별 1 km 창 평균. 반환 dict 또는 None."""
    import pyproj
    import rasterio
    from affine import Affine
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder
    from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
    (tlat, tlon), idx = task
    path = tname(tlat, tlon)
    if not path.exists():
        return idx, None
    with rasterio.open(path) as ds:
        z = ds.read(1).astype(float); tr = ds.transform
    z[(z < -500) | ~np.isfinite(z)] = np.nan
    if np.isfinite(z).mean() < 0.2:
        return idx, None
    coslat = np.cos(np.radians(tlat + 0.5))
    fy = max(1, int(round(TARGET_RES_M / (abs(tr.e) * 111320.0)))); fx = max(1, int(round(TARGET_RES_M / (abs(tr.a) * 111320.0 * coslat))))
    H, W = z.shape[0] // fy * fy, z.shape[1] // fx * fx
    z = np.nanmean(z[:H, :W].reshape(H // fy, fy, W // fx, fx), axis=(1, 3))
    tr = Affine(tr.a * fx, tr.b, tr.c, tr.d, tr.e * fy, tr.f)
    H, W = z.shape
    dy = abs(tr.e) * 111320.0; dx = abs(tr.a) * 111320.0 * coslat                     # m
    nod = ~np.isfinite(z)
    zf = np.where(nod, np.nanmedian(z), z)
    vf = ViewFinder(affine=tr, shape=zf.shape, crs=pyproj.Proj("epsg:4326"), nodata=np.float64(np.nan))
    grid = Grid(viewfinder=vf); dem = Raster(zf, viewfinder=vf)
    pit = grid.fill_pits(dem); dep = grid.fill_depressions(pit); infl = grid.resolve_flats(dep)
    fdir = grid.flowdir(infl); acc = np.asarray(grid.accumulation(fdir), float)
    gy, gx = np.gradient(zf, dy, dx)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    cell_area = dx * dy; contour = 0.5 * (dx + dy)
    a_spec = (acc + 1.0) * cell_area / contour
    twi = np.log(a_spec / np.maximum(slope, 1e-3))
    zs = uniform_filter(zf, 3)
    zx, zy = np.gradient(zs, dx, axis=1), np.gradient(zs, dy, axis=0)
    zxx, zyy = np.gradient(zx, dx, axis=1), np.gradient(zy, dy, axis=0)
    zxy = np.gradient(zx, dy, axis=0)
    p, q = zx ** 2 + zy ** 2, zx ** 2 + zy ** 2 + 1e-9
    curv_prof = -(zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / (q * (1 + p) ** 1.5 + 1e-12)
    curv_plan = -(zxx * zy ** 2 - 2 * zxy * zx * zy + zyy * zx ** 2) / (q ** 1.5 + 1e-12)
    ky, kx = max(1, int(round(1000 / dy))), max(1, int(round(1000 / dx)))
    ky += (ky + 1) % 2; kx += (kx + 1) % 2
    win = (ky, kx)
    rel = maximum_filter(zf, size=win) - minimum_filter(zf, size=win)
    def m1k(a):
        return uniform_filter(np.where(np.isfinite(a), a, np.nanmedian(a)), size=win)
    twi_m, twi_sd = m1k(twi), np.sqrt(np.maximum(m1k(twi ** 2) - m1k(twi) ** 2, 0))
    acc_m = m1k(np.log10(acc + 1.0)); cp_m, cq_m = m1k(curv_plan) * 1e3, m1k(curv_prof) * 1e3
    rows, cols = rasterio.transform.rowcol(tr, LON[idx], LAT[idx])
    rows = np.clip(np.asarray(rows), 0, H - 1); cols = np.clip(np.asarray(cols), 0, W - 1)
    out = dict(twi=twi_m[rows, cols], twi_sd=twi_sd[rows, cols], flowacc_log=acc_m[rows, cols], curv_plan=cp_m[rows, cols],
               curv_prof=cq_m[rows, cols], rel_1km=rel[rows, cols])
    bad = nod[rows, cols]
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    return idx, out


TARGET_RES_M = 90.0


if args.terrain:
    t0 = time.time()
    tl, tn = np.floor(LAT).astype(int), np.floor(LON).astype(int)
    tasks = [((a, b), np.where((tl == a) & (tn == b))[0]) for a, b in sorted(set(zip(tl, tn)))]
    res = {k: np.full(N, np.nan) for k in ("twi", "twi_sd", "flowacc_log", "curv_plan", "curv_prof", "rel_1km")}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(terrain_tile, t) for t in tasks]
        for f in as_completed(futs):
            idx, out = f.result(); done += 1
            if out is not None:
                for k in res:
                    res[k][idx] = out[k]
            if done % 20 == 0:
                print(f"[terrain] {done}/{len(tasks)} 타일 · {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(dict(loc_id=CELLS.loc_id, **res)); out.to_csv(PART / "terrain.csv", index=False)
    print(f"[terrain] 결측 twi {np.isnan(res['twi']).mean():.3f} · {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 30 m 10° 타일 창 샘플(GSW·Hansen)
def tile10(lat, lon):
    return int(np.floor(lat / 10) * 10 + 10), int(np.floor(lon / 10) * 10)


def window_stats(path, idx, fn):
    import rasterio
    out = {}
    with rasterio.open(path) as ds:
        tr = ds.transform; H, W = ds.shape
        px = abs(tr.a)                                       # deg/px (30 m ≈ 0.00027°)
        for i in idx:
            r, c = rasterio.transform.rowcol(tr, LON[i], LAT[i])
            out[i] = fn(ds, r, c, H, W, px, LAT[i])
    return out


def gsw_fn(ds, r, c, H, W, px, lat):
    import rasterio
    res = {}
    for name, km in (("water_occ_1km", 0.5), ("water_occ_5km", 2.5)):
        hy = int(round(km * 1000 / (px * 111320))); hx = int(round(km * 1000 / (px * 111320 * np.cos(np.radians(lat)))))
        win = rasterio.windows.Window(max(0, c - hx), max(0, r - hy), min(W, c + hx + 1) - max(0, c - hx), min(H, r + hy + 1) - max(0, r - hy))
        a = ds.read(1, window=win).astype(float)
        a[a == 255] = np.nan                                 # 255 = no data
        res[name] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        if km == 2.5:
            wet = a > 50
            if wet.any():
                yy, xx = np.where(wet)
                cy, cx = r - max(0, r - hy), c - max(0, c - hx)
                d = np.sqrt(((yy - cy) * px * 111320) ** 2 + ((xx - cx) * px * 111320 * np.cos(np.radians(lat))) ** 2) / 1000
                res["water_dist_km"] = float(min(d.min(), 5.0))
            else:
                res["water_dist_km"] = 5.0
    return res


def tree_fn(ds, r, c, H, W, px, lat):
    import rasterio
    hy = int(round(500 / (px * 111320))); hx = int(round(500 / (px * 111320 * np.cos(np.radians(lat)))))
    win = rasterio.windows.Window(max(0, c - hx), max(0, r - hy), min(W, c + hx + 1) - max(0, c - hx), min(H, r + hy + 1) - max(0, r - hy))
    a = ds.read(1, window=win).astype(float)
    return dict(treecover_1km=float(a.mean()) if a.size else np.nan)


def run_tiles(kind):
    t0 = time.time()
    groups = {}
    for i in range(N):
        groups.setdefault(tile10(LAT[i], LON[i]), []).append(i)
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for (top, left), idx in groups.items():
            ns = f"{abs(top)}{'N' if top >= 0 else 'S'}"
            if kind == "water":
                ew = f"{abs(left)}{'E' if left >= 0 else 'W'}"; path = RAW / "gsw" / f"occurrence_{ew}_{ns}v1_4_2021.tif"; fn = gsw_fn
            else:
                ew = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"; path = RAW / "hansen" / f"Hansen_GFC-2023-v1.11_treecover2000_{ns}_{ew}.tif"; fn = tree_fn
            if path.exists():
                futs.append(ex.submit(window_stats, path, np.array(idx), fn))
        for f in as_completed(futs):
            rows.update(f.result())
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(range(N))
    df.insert(0, "loc_id", CELLS.loc_id.values)
    df.to_csv(PART / f"{kind}.csv", index=False)
    print(f"[{kind}] 결측 {df.iloc[:, 1].isna().mean():.3f} · {time.time()-t0:.0f}s", flush=True)


if args.water:
    run_tiles("water")
if args.trees:
    run_tiles("trees")


# ---------------------------------------------------------------- ERA5-Land 확장
if args.era5:
    import xarray as xr
    t0 = time.time()
    ds = xr.open_dataset(RAW / "era5land" / "nh_monthly_ext_2015-2020.nc")
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    ds = ds.assign_coords(month=ds[tn].dt.month)
    clim = ds.groupby("month").mean(tn)
    lat_g, lon_g = clim.latitude.values, clim.longitude.values
    lon_q = np.where(LON > lon_g.max(), LON - 360, LON) if lon_g.max() <= 180 else np.where(LON < 0, LON + 360, LON)
    ri = np.abs(lat_g[None, :] - LAT[:, None]).argmin(1); ci = np.abs(lon_g[None, :] - lon_q[:, None]).argmin(1)
    def samp(var):
        a = clim[var].values                                  # (12, lat, lon)
        v = a[:, ri, ci].T                                    # (N,12)
        miss = ~np.isfinite(v).all(1)
        for half in (2, 5):                                   # 육지 전용 격자 폴백(era5land_soil_tdd.py 규약)
            if not miss.any():
                break
            for i in np.where(miss)[0]:
                w = a[:, max(0, ri[i] - half):ri[i] + half + 1, max(0, ci[i] - half):ci[i] + half + 1]
                if np.isfinite(w).any():
                    v[i] = np.nanmean(w.reshape(12, -1), 1); miss[i] = False
        return v
    swvl1, swvl2 = samp("swvl1"), samp("swvl2")
    skt = samp("skt") - 273.15
    lai = samp("lai_hv") + samp("lai_lv")
    rsn, sde, tp = samp("rsn"), samp("sde"), samp("tp") * 1000.0 * DAYS[None, :]   # m/day → mm/month
    tdd_s, fdd_s = degree_days(skt)
    out = pd.DataFrame(dict(loc_id=CELLS.loc_id, e5_swvl1_summer=swvl1[:, 5:8].mean(1), e5_swvl2_summer=swvl2[:, 5:8].mean(1),
                            e5_skt_tdd=tdd_s, e5_skt_fdd=fdd_s, e5_skt_nfactor=tdd_s / np.where(CELLS.e5_tdd.values > 0, CELLS.e5_tdd.values, np.nan),
                            e5_lai_summer=lai[:, 5:8].mean(1), e5_rsn_winter=rsn[:, [0, 1, 11]].mean(1), e5_sde_winter=sde[:, [0, 1, 2, 11]].mean(1),
                            e5_tp_summer=tp[:, 5:8].sum(1), e5_tp_annual=tp.sum(1)))
    out.to_csv(PART / "era5ext.csv", index=False)
    print(f"[era5] 결측 {out.e5_skt_tdd.isna().mean():.3f} · {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 병합
GROUPS = {"LST": ["lst_tdd", "lst_fdd", "lst_sqrt_tdd", "lst_nfactor", "lst_amp", "lst_nclear_summer"],
          "T": ["twi", "twi_sd", "flowacc_log", "curv_plan", "curv_prof", "rel_1km"],
          "W": ["water_occ_1km", "water_occ_5km", "water_dist_km"],
          "V": ["treecover_1km", "ndvi_summer", "ndvi_max", "evi_summer", "ndwi_summer"],
          "H": ["e5_swvl1_summer", "e5_swvl2_summer", "e5_skt_tdd", "e5_skt_fdd", "e5_skt_nfactor", "e5_lai_summer", "e5_rsn_winter",
                "e5_sde_winter", "e5_tp_summer", "e5_tp_annual"]}
if args.merge:
    out = CELLS[["loc_id", "lat", "lon"]].copy()
    for p in ("lst", "ndvi", "terrain", "water", "trees", "era5ext"):
        f = PART / f"{p}.csv"
        if f.exists():
            out = out.merge(pd.read_csv(f), on="loc_id", how="left")
    out.to_csv(PROC / "covariates_ext_v1.csv", index=False)
    base = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "region"])
    mm = out.merge(base, on="loc_id")
    miss = {g: {r: float(mm.loc[mm.region == r, [c for c in cols if c in mm]].isna().any(axis=1).mean()) for r in mm.region.unique()}
            for g, cols in GROUPS.items() if any(c in mm for c in cols)}
    meta = dict(groups={g: [c for c in cols if c in out] for g, cols in GROUPS.items()}, n=int(len(out)), years=args.years,
                missing_rate_by_region=miss, sources=dict(LST="MOD11C3 v061 Terra", V="MOD13C2 v061 + Hansen GFC-2023 v1.11",
                W="JRC GSW occurrence v1.4 2021", T="Copernicus DEM GLO-30 + pysheds 0.5", H="ERA5-Land monthly means (CDS)"),
                design="docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md")
    (PROC / "covariates_ext_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("[merge] 열", list(out.columns)); print(json.dumps(miss, indent=1))
