"""LGD 새 셀(ext_cells_v1 의 in_v4 = 1)에 공변량을 붙인다. v3 의 E3 규약(expand_calm_regions.py 2단계,
era5land_soil_tdd.py)을 그대로 따르되, 큰 격자는 필요한 영역만 읽는다(공유 서버 규칙).

단계(--stage, 쉼표로 여러 개)
  tiles  필요한 Copernicus DEM GLO-30 1° 타일 목록을 만들고 없는 타일을 내려받는다(AWS 공개 버킷, 계정 불필요)
  dem    지형 6종(33 × 33 창, 타일 경계에서 잘림, 중심 결측이면 창 안 최근접 유효 화소). expand_calm_regions.py c) 와 같다
  e5     ERA5-Land 8종(nh_monthly_2015-2020.nc 의 2015–2020 월 기후값, 최근접 격자, 해양이면 ±2 격자(0.2°) → ±5 격자(0.5°)
         육지 폴백, 폴백 선택은 위경도 유클리드 거리). expand_calm_regions.py d) 와 같다.
         토양 도일 5종(stl1 기후값, 육지 판정은 stl1, 폴백은 체비쇼프 격자 거리). era5land_soil_tdd.py 와 같다.
         기후값은 셀이 모인 청크 영역(±6 격자)만 읽어 계산한다(격자점별 계산이라 전 격자 계산과 값이 같다)
  cci    CCI ALT 25개 연도 파일(1997–2021)의 최근접 화소(xarray sel nearest 와 같은 색인) 다년 평균 × 100, 소수 2자리.
         파일마다 필요한 1000 × 1000 청크만 읽는다. expand_calm_regions.py e) 와 같다
  sg     SoilGrids WCS 9층. 기존 창(enrich_soilgrids_wcs.WINDOWS) → E3 창(windows_wcs_meta_e3.json) → 새 창
         v4_<macro>_<lobe>(E3 와 같은 bbox·lobe·5 km 규칙) 순으로 채운다. 새 창 메타는 data/raw/soilgrids_wcs/windows_wcs_meta_v4.json
  elev   (개정 13) ERA5-Land 사용 격자(e5 단계의 기온 격자) 중심의 0.1° 상자에서 Copernicus DEM 평균 고도를 계산한다.
         ERA5-Land 지형 고도의 대리값이다(ERA5-Land geopotential 은 내려받지 않았다). 상자가 걸친 타일이 없으면 내려받고,
         바다 타일(404)은 뺀 면적 비율을 e5_grid_dem_cov 에 적는다
  merge  부분 산출을 합쳐 ext_cells_v1_cov.csv 와 ext_cells_v1_cov_meta.json 을 쓴다. 토양 도일이 0 이하이면
         e5_tdd_soil, e5_sqrt_tdd_soil 을 결측으로 둔다(v3 규칙, e5_soil_tdd_v4 와 같다. 개정 13)
--validate: 입력을 v3 의 CALM_* 행(149행)으로 바꾸어 dem, e5, cci, sg 를 다시 계산하고 v3 값과 대조한다(새 창은 내려받지 않는다).
--input v3all: (개정 13) 입력을 v3 의 전체 행(17,572)으로 바꾸어 e5 단계만 계산한다. 사용 격자의 월 적설 최솟값
  (e5_grid_sd_min_m)을 라벨 부가 표에 붙이고, 같은 격자 규칙으로 다시 계산한 e5_tdd 가 v3 값과 같은지 대조한다.
e5 단계 부가 열(개정 13): e5_grid_lat, e5_grid_lon(기온·적설에 쓴 격자 중심), e5_grid_sd_min_m(그 격자의 2015–2020년
  72개월 월 적설 수당량 최솟값, m). 1 m 이상이면 연중 눈이 남는 격자(빙하 또는 빙상 가장자리)로 본다.

산출: data/processed/ext_labels/cov_parts/<입력>_<단계>.csv, data/processed/ext_labels/ext_cells_v1_cov.csv,
      data/processed/ext_labels/ext_cells_v1_cov_meta.json (검증 모드는 cov_validate_meta.json)
실행(ROOT): OMP_NUM_THREADS=1 python3 scripts/1_data_prep/ext_cells_covariates_v1.py --stage tiles,dem,e5,cci,sg,merge
"""
from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
PARTS = EXT / "cov_parts"
DEM = ROOT / "data" / "raw" / "dem"
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2015-2020.nc"
CCI_DIR = ROOT / "data" / "raw" / "cci_alt"
SG_RAW = ROOT / "data" / "raw" / "soilgrids_wcs"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "1_data_prep"))

TERR = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
E5 = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
SOILT = ["e5_tdd_soil", "e5_fdd_soil", "e5_sqrt_tdd_soil", "e5_stl1_twarm", "e5_stl1_tcold", "soil_fallback_deg"]


def log(*a):
    print(*a, flush=True)


def load_input(validate: bool, inp: str = "") -> pd.DataFrame:
    if inp == "v3all":
        v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False, usecols=["loc_id", "lat", "lon", "region"])
        v3["id"] = v3.loc_id.astype(str)
        v3["macro"] = v3.region
        return v3[["id", "lat", "lon", "macro"]].reset_index(drop=True)
    if validate:
        from polar.fidelity import macro_region
        v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
        d = v3[v3.region.str.startswith("CALM_")].copy()
        d["id"] = d.loc_id.astype(str)
        d["macro"] = macro_region(d)
        return d[["id", "lat", "lon", "macro"]].reset_index(drop=True)
    c = pd.read_csv(EXT / "ext_cells_v1.csv")
    c = c[c.in_v4 == 1].copy()
    c["id"] = c.cell_uid
    return c[["id", "lat", "lon", "macro"]].reset_index(drop=True)


def tname(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


# ---------------------------------------------------------------- tiles
def stage_tiles(cell, tag):
    t = sorted(set(zip(np.floor(cell.lat).astype(int), np.floor(cell.lon).astype(int))))
    rows = []
    for tlat, tlon in t:
        nm = tname(tlat, tlon)
        dst = DEM / (nm + ".tif")
        st = "exists"
        if not (dst.exists() and dst.stat().st_size > 1000):
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{nm}/{nm}.tif"
            tmp = dst.with_suffix(".tif.part")
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, dst)
                st = "downloaded"
            except urllib.error.HTTPError as e:
                st = f"http_{e.code}"
                if tmp.exists():
                    tmp.unlink()
            except Exception as e:  # noqa: BLE001
                st = f"error:{str(e)[:60]}"
                if tmp.exists():
                    tmp.unlink()
            log(f"  {nm}: {st}")
        rows.append(dict(tlat=tlat, tlon=tlon, tile=nm, status=st, bytes=int(dst.stat().st_size) if dst.exists() else 0,
                         n_cells=int(((np.floor(cell.lat) == tlat) & (np.floor(cell.lon) == tlon)).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(PARTS / f"{tag}_tiles.csv", index=False)
    log(f"[tiles] {len(out)} 타일 · " + str(out.status.value_counts().to_dict()))


# ---------------------------------------------------------------- dem (expand_calm_regions.py c)
def stage_dem(cell, tag):
    import rasterio
    W, H, MPD = 33, 16, 111320.0
    cell = cell.copy()
    cell["tlat"] = np.floor(cell.lat).astype(int)
    cell["tlon"] = np.floor(cell.lon).astype(int)
    feats = np.full((len(cell), 6), np.nan)
    missing = []
    for (tlat, tlon), g in cell.groupby(["tlat", "tlon"]):
        path = DEM / (tname(tlat, tlon) + ".tif")
        if not path.exists():
            missing.append((int(tlat), int(tlon), len(g)))
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
        del arr
    out = pd.DataFrame(feats, columns=TERR)
    out.insert(0, "id", cell.id.values)
    out["dem_tile_missing"] = [int((int(a), int(b)) in {(m[0], m[1]) for m in missing}) for a, b in zip(cell.tlat, cell.tlon)]
    out.to_csv(PARTS / f"{tag}_dem.csv", index=False)
    log(f"[dem] 유효 {int(np.isfinite(feats[:, 0]).sum())}/{len(cell)} · 미보유 타일 {missing}")


# ---------------------------------------------------------------- e5 + soil tdd
def stage_e5(cell, tag):
    import xarray as xr
    ds = xr.open_dataset(NC)
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    glat, glon = ds["latitude"].values, ds["longitude"].values
    NY, NX = len(glat), len(glon)
    days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])
    iy = np.array([int(np.abs(glat - la).argmin()) for la in cell.lat])
    ix = np.array([int(np.abs(glon - lo).argmin()) for lo in cell.lon])
    out = {k: np.full(len(cell), np.nan) for k in E5 + SOILT + ["e5_grid_lat", "e5_grid_lon", "e5_grid_sd_min_m"]}
    e5_fb = np.full(len(cell), np.nan)
    cnt = dict(fb02=0, fb05=0, fail=0, s_direct=0, s02=0, s05=0, s_fail=0)
    grp = pd.Series(list(zip(iy // 148, ix // 900)))
    M = 6
    for key, gidx in grp.groupby(grp).groups.items():
        gidx = np.asarray(list(gidx))
        y0, y1 = max(int(iy[gidx].min()) - M, 0), min(int(iy[gidx].max()) + M + 1, NY)
        x0, x1 = int(ix[gidx].min()) - M, int(ix[gidx].max()) + M + 1
        if x0 < 0 or x1 > NX:                                      # 날짜 변경선 근처: 경도 전체
            x0, x1 = 0, NX
        sub = ds[["t2m", "sd", "stl1"]].isel(latitude=slice(y0, y1), longitude=slice(x0, x1)).load()
        clim = sub.assign_coords(month=sub[tn].dt.month).groupby("month").mean(tn)
        t2m = clim["t2m"].values - 273.15
        stl = clim["stl1"].values - 273.15
        swe = clim["sd"].values
        sd_raw = sub["sd"].values                                  # 72개월 월값(개정 13: 사용 격자의 최솟값)
        land = np.isfinite(t2m).all(axis=0)
        valid_s = np.isfinite(stl).all(axis=0)
        sy, sx = y1 - y0, x1 - x0
        full_lon = (x0 == 0 and x1 == NX)

        def loc(yg, xg):                                           # 전역 색인 → 부분 색인(없으면 None)
            yl = yg - y0
            xl = (xg % NX) - x0 if full_lon else xg - x0
            if 0 <= yl < sy and 0 <= xl < sx:
                return yl, xl
            return None

        for p in gidx:
            la_, lo_ = float(cell.lat.iloc[p]), float(cell.lon.iloc[p])
            if not (glat.min() <= la_ <= glat.max()):
                cnt["fail"] += 1
                continue
            # --- 기온·적설(E3 fallback: 유클리드 거리, 엄격한 < 로 먼저 찾은 것 유지)
            by, bx = iy[p], ix[p]
            q = loc(by, bx)
            ok = True
            if not land[q]:
                def fallback(nb):
                    best = None
                    for dy_ in range(-nb, nb + 1):
                        for dx_ in range(-nb, nb + 1):
                            y2, x2 = by + dy_, (bx + dx_) % NX
                            qq = loc(y2, x2) if 0 <= y2 < NY else None
                            if qq is not None and land[qq]:
                                d2 = (glat[y2] - la_) ** 2 + (glon[x2] - lo_) ** 2
                                if best is None or d2 < best[0]:
                                    best = (d2, y2, x2)
                    return best
                best = fallback(2)
                if best is not None:
                    cnt["fb02"] += 1; e5_fb[p] = 0.2
                else:
                    best = fallback(5)
                    if best is not None:
                        cnt["fb05"] += 1; e5_fb[p] = 0.5
                if best is None:
                    cnt["fail"] += 1; ok = False
                else:
                    q = loc(best[1], best[2])
            else:
                e5_fb[p] = 0.0
            if ok:
                tm = t2m[:, q[0], q[1]]
                tdd = float(np.nansum(np.clip(tm, 0, None) * days))
                out["e5_maat"][p] = float(np.nanmean(tm)); out["e5_tdd"][p] = tdd
                out["e5_fdd"][p] = float(np.nansum(np.clip(-tm, 0, None) * days))
                out["e5_sqrt_tdd"][p] = float(np.sqrt(tdd))
                out["e5_twarm"][p] = float(np.nanmax(tm)); out["e5_tcold"][p] = float(np.nanmin(tm))
                out["e5_stl1"][p] = float(np.nanmean(stl[:, q[0], q[1]])); out["e5_swe"][p] = float(np.nanmean(swe[:, q[0], q[1]]))
                out["e5_grid_lat"][p] = float(glat[y0 + q[0]])
                out["e5_grid_lon"][p] = float(glon[(x0 + q[1]) % NX])
                out["e5_grid_sd_min_m"][p] = float(np.nanmin(sd_raw[:, q[0], q[1]]))
            # --- 토양 도일(era5land_soil_tdd.py: stl1 육지 판정, 체비쇼프 격자 거리, 먼저 찾은 것 유지)
            by, bx = iy[p], ix[p]

            def nearest_valid(max_off):
                best, bestd = None, None
                for dy_ in range(-max_off, max_off + 1):
                    for dx_ in range(-max_off, max_off + 1):
                        y2, x2 = by + dy_, (bx + dx_) % NX
                        qq = loc(y2, x2) if 0 <= y2 < NY else None
                        if qq is not None and valid_s[qq]:
                            d = max(abs(dy_), abs(dx_))
                            if bestd is None or d < bestd:
                                best, bestd = qq, d
                return best, bestd
            qs = loc(by, bx)
            if not valid_s[qs]:
                c2, d = nearest_valid(2)
                if c2 is None:
                    c2, d = nearest_valid(5)
                    off = 5 if c2 is not None else None
                else:
                    off = 2
                if c2 is None:
                    cnt["s_fail"] += 1
                    continue
                cnt["s02" if off == 2 else "s05"] += 1
                qs, fb = c2, d * 0.1
            else:
                cnt["s_direct"] += 1
                fb = 0.0
            tm = stl[:, qs[0], qs[1]]
            tdd = float(np.sum(np.clip(tm, 0, None) * days))
            out["e5_tdd_soil"][p] = tdd
            out["e5_fdd_soil"][p] = float(np.sum(np.clip(-tm, 0, None) * days))
            out["e5_sqrt_tdd_soil"][p] = float(np.sqrt(tdd))
            out["e5_stl1_twarm"][p] = float(tm.max()); out["e5_stl1_tcold"][p] = float(tm.min())
            out["soil_fallback_deg"][p] = fb
        log(f"  [e5] 청크 {key}: 셀 {len(gidx)} · 영역 {sy}×{sx}")
        del sub, clim, t2m, stl, swe, sd_raw
    ds.close()
    o = pd.DataFrame(out)
    o.insert(0, "id", cell.id.values)
    o["e5_fallback_deg"] = e5_fb
    o.to_csv(PARTS / f"{tag}_e5.csv", index=False)
    log(f"[e5] 유효 {int(np.isfinite(o.e5_maat).sum())}/{len(o)} · 토양 {int(np.isfinite(o.e5_tdd_soil).sum())} · {cnt}")
    return cnt


# ---------------------------------------------------------------- elev (개정 13)
def fetch_tile(tlat, tlon):
    nm = tname(tlat, tlon)
    dst = DEM / (nm + ".tif")
    if dst.exists() and dst.stat().st_size > 1000:
        return "exists"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{nm}/{nm}.tif"
    tmp = dst.with_suffix(".tif.part")
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        return "downloaded"
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        return f"http_{e.code}"
    except Exception as e:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        return f"error:{str(e)[:60]}"


def stage_elev(cell, tag, half=0.05):
    """ERA5-Land 사용 격자 중심 ±0.05° 상자의 Copernicus DEM 평균(타일 조각별 평균을 조각 면적으로 가중)."""
    import rasterio
    from rasterio.windows import from_bounds
    e5 = pd.read_csv(PARTS / f"{tag}_e5.csv", dtype={"id": str}).set_index("id")
    glat = e5.loc[cell.id, "e5_grid_lat"].values
    glon = e5.loc[cell.id, "e5_grid_lon"].values
    need = set()
    for la, lo in zip(glat, glon):
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        for tl in range(int(np.floor(la - half)), int(np.floor(la + half)) + 1):
            for tn in range(int(np.floor(lo - half)), int(np.floor(lo + half)) + 1):
                need.add((tl, tn))
    status = {t: fetch_tile(*t) for t in sorted(need)}
    got = {k: v for k, v in status.items() if v not in ("exists", "downloaded")}
    log(f"[elev] 필요 타일 {len(need)} · 새로 받음 {sum(v == 'downloaded' for v in status.values())} · 없음 {len(got)} {got}")
    mean = np.full(len(cell), np.nan)
    cov = np.full(len(cell), np.nan)
    npx = np.zeros(len(cell), int)
    for i, (la, lo) in enumerate(zip(glat, glon)):
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        b0, b1, c0, c1 = la - half, la + half, lo - half, lo + half
        acc, area, area_all = 0.0, 0.0, 0.0
        for tl in range(int(np.floor(b0)), int(np.floor(b1)) + 1):
            for tn in range(int(np.floor(c0)), int(np.floor(c1)) + 1):
                pb0, pb1 = max(b0, tl), min(b1, tl + 1)
                pc0, pc1 = max(c0, tn), min(c1, tn + 1)
                if pb1 <= pb0 or pc1 <= pc0:
                    continue
                a = (pb1 - pb0) * (pc1 - pc0)
                area_all += a
                path = DEM / (tname(tl, tn) + ".tif")
                if not path.exists():
                    continue
                with rasterio.open(path) as ds:
                    w = from_bounds(pc0, pb0, pc1, pb1, ds.transform).round_offsets().round_lengths()
                    arr = ds.read(1, window=w, boundless=False).astype(np.float64)
                    if ds.nodata is not None:
                        arr[arr == ds.nodata] = np.nan
                ok = np.isfinite(arr)
                if ok.sum() == 0:
                    continue
                acc += float(np.nanmean(arr)) * a
                area += a
                npx[i] += int(ok.sum())
        if area > 0:
            mean[i] = acc / area
            cov[i] = area / area_all
    o = pd.DataFrame(dict(id=cell.id.values, e5_grid_dem_mean_m=np.round(mean, 2), e5_grid_dem_cov=np.round(cov, 4), e5_grid_dem_npx=npx))
    o.to_csv(PARTS / f"{tag}_elev.csv", index=False)
    pd.DataFrame([dict(tlat=a, tlon=b, tile=tname(a, b), status=v) for (a, b), v in status.items()]).to_csv(
        PARTS / f"{tag}_elev_tiles.csv", index=False)
    log(f"[elev] 유효 {int(np.isfinite(mean).sum())}/{len(cell)} · 면적 비율 < 1 셀 {int((cov < 1).sum())}")


# ---------------------------------------------------------------- cci (expand_calm_regions.py e)
def stage_cci(cell, tag):
    import xarray as xr
    files = sorted(glob.glob(str(CCI_DIR / "*.nc")))
    acc = np.zeros(len(cell)); cnt = np.zeros(len(cell))
    per_file = []
    for f in files:
        ds = xr.open_dataset(f)
        a = ds["ALT"]
        if "time" in a.dims:
            a = a.isel(time=0)
        # xarray 의 sel(method="nearest") 는 질의 값을 좌표 dtype(float32)으로 바꾼 뒤 찾는다. 같은 규칙을 쓴다
        jy = ds.indexes["lat"].get_indexer(cell.lat.values.astype(ds["lat"].dtype), method="nearest")
        jx = ds.indexes["lon"].get_indexer(cell.lon.values.astype(ds["lon"].dtype), method="nearest")
        v = np.full(len(cell), np.nan)
        key = pd.Series(list(zip(jy // 1000, jx // 1000)))
        for (cy, cx), gi in key.groupby(key).groups.items():
            gi = np.asarray(list(gi))
            ys, xs = cy * 1000, cx * 1000
            blk = a.isel(lat=slice(ys, ys + 1000), lon=slice(xs, xs + 1000)).values
            v[gi] = blk[jy[gi] - ys, jx[gi] - xs].astype(float)
        ok = np.isfinite(v)
        acc[ok] += v[ok]; cnt[ok] += 1
        per_file.append(dict(file=Path(f).name, n_valid=int(ok.sum())))
        ds.close()
    cci = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) * 100.0
    o = pd.DataFrame(dict(id=cell.id.values, cci_alt=np.round(cci, 2), cci_valid=np.isfinite(cci).astype(int), cci_n_years=cnt.astype(int)))
    o.to_csv(PARTS / f"{tag}_cci.csv", index=False)
    log(f"[cci] 유효 {int(o.cci_valid.sum())}/{len(o)} · 연도 파일 {len(files)}")


# ---------------------------------------------------------------- sg
def stage_sg(cell, tag, allow_download=True):
    import enrich_soilgrids_wcs as sgw
    sg_cols = [sgw.col_name(p, d) for p, d in sgw.LAYERS]
    vals = pd.DataFrame(np.nan, index=range(len(cell)), columns=sg_cols)
    src = np.array([""] * len(cell), dtype=object)
    ix_x, ix_y = sgw.TR.transform(cell.lon.values, cell.lat.values)
    ix_x, ix_y = np.asarray(ix_x), np.asarray(ix_y)

    def sample_window(wname, idx):
        got = 0
        for (prop, dpt) in sgw.LAYERS:
            path = SG_RAW / wname / f"{prop}_{dpt}_mean.tif"
            if not path.exists():
                return got
            s = sgw._sample(str(path), ix_x[idx], ix_y[idx]) / sgw.SCALE.get(prop, (1.0, ""))[0]
            col = sgw.col_name(prop, dpt)
            cur = vals[col].values
            fill = np.isfinite(s) & np.isnan(cur[idx])
            cur[idx[fill]] = s[fill]
            vals[col] = cur
            for k in idx[fill]:
                if wname not in src[k].split(";"):
                    src[k] = (src[k] + ";" + wname).strip(";")
            got += int(fill.sum())
        return got

    wins = [(w, b) for w, b in sgw.WINDOWS.items()]
    e3m = json.loads((SG_RAW / "windows_wcs_meta_e3.json").read_text())
    wins += [(w, tuple(m["wgs84_bbox"])) for w, m in e3m.items()]
    v4m_path = SG_RAW / "windows_wcs_meta_v4.json"
    v4m = json.loads(v4m_path.read_text()) if v4m_path.exists() else {}
    for wname, (lo0, lo1, la0, la1) in wins:
        inb = np.where((cell.lon >= lo0) & (cell.lon <= lo1) & (cell.lat >= la0) & (cell.lat <= la1))[0]
        if len(inb):
            g = sample_window(wname, inb)
            log(f"[sg] 기존 창 {wname}: 후보 {len(inb)} · 채움 {g}")
    if allow_download:
        lobe = np.digitize(cell.lon.values, [-40, 20, 100])
        miss = vals.isna().any(axis=1).values
        for (mac, lb), g in cell[miss].groupby([cell.macro[miss], lobe[miss]]):
            wname = f"v4_{mac}_{lb}"
            lo0, lo1 = float(g.lon.min() - 0.3), float(g.lon.max() + 0.3)
            la0, la1 = float(g.lat.min() - 0.3), float(g.lat.max() + 0.3)
            bounds = {0: (-180, -40), 1: (-40, 20), 2: (20, 100), 3: (100, 180)}[lb]
            lo0, lo1 = max(lo0, bounds[0] + 0.01), min(lo1, bounds[1] - 0.01)
            x0, x1, y0, y1 = sgw.igh_bbox(lo0, lo1, la0, la1)
            nx, ny = sgw.scalesize(x0, x1, y0, y1)
            wdir = SG_RAW / wname
            wdir.mkdir(parents=True, exist_ok=True)
            mm = v4m.setdefault(wname, dict(wgs84_bbox=[lo0, lo1, la0, la1], igh_bbox=[round(x0), round(x1), round(y0), round(y1)],
                                            scalesize=[nx, ny], n_cells=int(len(g)), layers={}))
            log(f"[sg] 새 창 {wname}: lon[{lo0:.2f},{lo1:.2f}] lat[{la0:.2f},{la1:.2f}] size {nx}x{ny} · 셀 {len(g)}")
            for prop, dpt in sgw.LAYERS:
                path = wdir / f"{prop}_{dpt}_mean.tif"
                if path.exists() and sgw._valid_tif(str(path)):
                    mm["layers"].setdefault(f"{prop}_{dpt}", "cached")
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
                mm["layers"][f"{prop}_{dpt}"] = ("ok " + time.strftime("%Y-%m-%d")) if ok else "failed"
                if not ok and path.exists():
                    path.unlink()
                log(f"      {prop}_{dpt}: {'OK' if ok else 'FAILED'}")
            got = sample_window(wname, g.index.values)
            log(f"      채움 {got}")
        v4m_path.write_text(json.dumps(v4m, ensure_ascii=False, indent=1))
    o = vals.copy()
    o.insert(0, "id", cell.id.values)
    o["sg_windows"] = src
    o.to_csv(PARTS / f"{tag}_sg.csv", index=False)
    log(f"[sg] 9층 모두 유효 {int(vals.notna().all(axis=1).sum())}/{len(o)} · 지역별 " +
        str(pd.Series(vals.notna().all(axis=1).values).groupby(cell.macro.values).sum().astype(int).to_dict()))


# ---------------------------------------------------------------- merge / validate
def merge(cell, tag):
    o = cell[["id", "lat", "lon", "macro"]].copy()
    for st in ["dem", "e5", "cci", "sg", "elev"]:
        p = PARTS / f"{tag}_{st}.csv"
        if p.exists():
            o = o.merge(pd.read_csv(p, dtype={"id": str}), on="id", how="left")
        else:
            log(f"[merge] {st} 없음")
    return o


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cov_meta(o, tag, timing):
    """공변량 단계의 입력 해시, 폴백 수, DEM 타일 목록, 사용한 SG 창(개정 13)."""
    tiles = sorted({tname(int(np.floor(a)), int(np.floor(b))) for a, b in zip(o.lat, o.lon)})
    et = PARTS / f"{tag}_elev_tiles.csv"
    elev_tiles = pd.read_csv(et) if et.exists() else pd.DataFrame(columns=["tile", "status"])
    all_tiles = sorted(set(tiles) | set(elev_tiles.tile[elev_tiles.status.isin(["exists", "downloaded"])]))
    sg_used = sorted({w for x in o.sg_windows.fillna("") for w in str(x).split(";") if w})
    sg_files = {}
    for w in sg_used:
        for f in sorted((SG_RAW / w).glob("*.tif")):
            sg_files[f"{w}/{f.name}"] = sha256_file(f)
    return dict(
        created=time.strftime("%Y-%m-%d %H:%M"), script="scripts/1_data_prep/ext_cells_covariates_v1.py",
        script_sha256=sha256_file(Path(__file__)), tag=tag, n_cells=int(len(o)),
        inputs=dict(era5land=dict(path=str(NC.relative_to(ROOT)), sha256=sha256_file(NC)),
                    cci_alt={Path(f).name: sha256_file(f) for f in sorted(glob.glob(str(CCI_DIR / "*.nc")))},
                    dem_tiles={t: sha256_file(DEM / (t + ".tif")) for t in all_tiles if (DEM / (t + ".tif")).exists()},
                    dem_tiles_missing=[t for t in all_tiles if not (DEM / (t + ".tif")).exists()],
                    elev_tiles_status=elev_tiles.status.value_counts().to_dict(),
                    soilgrids_windows=sg_used, soilgrids_files=sg_files,
                    ext_cells=dict(path="data/processed/ext_labels/ext_cells_v1.csv", sha256=sha256_file(EXT / "ext_cells_v1.csv"))),
        fallback=dict(e5=o.e5_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict(),
                      soil=o.soil_fallback_deg.value_counts(dropna=False).rename(index=str).to_dict()),
        soil_tdd_le0_set_nan=int(o["_soil_le0"].sum()),
        e5_glacier_grid=int((o.e5_grid_sd_min_m >= 1.0).sum()),
        elev_cov_lt1=int((o.e5_grid_dem_cov < 1).sum()) if "e5_grid_dem_cov" in o else None,
        valid=dict(dem=int(o.dem_elev.notna().sum()), e5=int(o.e5_maat.notna().sum()),
                   soil_tdd=int(o.e5_sqrt_tdd_soil.notna().sum()), cci=int(o.cci_valid.sum()),
                   sg9=int(o[[c for c in o.columns if c.startswith("sg_")]].notna().all(axis=1).sum())),
        rules="dem·e5·cci·sg 는 v3 E3 규약(expand_calm_regions.py, era5land_soil_tdd.py)과 같다. e5_tdd_soil <= 0 이면 "
              "e5_tdd_soil, e5_sqrt_tdd_soil 결측(v3 규칙). elev 는 개정 13 에 더한 진단 열이고 입력 공변량이 아니다",
        timing_s=timing)


def validate(o):
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
    v3["id"] = v3.loc_id.astype(str)
    m = o.merge(v3, on="id", suffixes=("", "_v3"))
    import enrich_soilgrids_wcs as sgw
    cols = TERR + E5 + ["cci_alt", "cci_valid"] + [sgw.col_name(p, d) for p, d in sgw.LAYERS]
    rep = {}
    for c in cols:
        a, b = m[c].values.astype(float), m[c + "_v3"].values.astype(float)
        both = np.isfinite(a) & np.isfinite(b)
        rep[c] = dict(n=int(len(m)), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                      max_abs_diff=float(np.max(np.abs(a[both] - b[both]))) if both.any() else None)
    s3 = pd.read_csv(PROC / "e5_soil_tdd_v3.csv")
    s3 = s3[s3.loc_id < 0].copy()
    s3["klat"], s3["klon"] = s3.lat.round(4), s3.lon.round(4)
    m["klat"], m["klon"] = m.lat.round(4), m.lon.round(4)
    ms = m.merge(s3.drop_duplicates(["klat", "klon"]), on=["klat", "klon"], suffixes=("", "_s3"))
    for c in ["e5_tdd_soil", "e5_fdd_soil", "e5_sqrt_tdd_soil", "e5_stl1_twarm", "e5_stl1_tcold", "soil_fallback_deg"]:
        a, b = ms[c].values.astype(float), ms[c + "_s3"].values.astype(float)
        if c in ("e5_tdd_soil", "e5_sqrt_tdd_soil"):                 # v3 규칙: tdd_soil <= 0 이면 NaN
            a = np.where(ms["e5_tdd_soil"].values <= 0, np.nan, a)
        both = np.isfinite(a) & np.isfinite(b)
        rep[c] = dict(n=int(len(ms)), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                      max_abs_diff=float(np.max(np.abs(a[both] - b[both]))) if both.any() else None)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="tiles,dem,e5,cci,sg,merge")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--input", default="", help="v3all: v3 전체 행(e5 단계만)")
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    tag = "validate" if a.validate else ("v3all" if a.input == "v3all" else "ext_v1")
    cell = load_input(a.validate, a.input)
    log(f"[in] {tag}: {len(cell)} 셀")
    stages = [s for s in a.stage.split(",") if s]
    t0 = time.time()
    timing = {}
    for st in stages:
        t1 = time.time()
        if st == "tiles" and not a.validate:
            stage_tiles(cell, tag)
        elif st == "dem":
            stage_dem(cell, tag)
        elif st == "e5":
            stage_e5(cell, tag)
        elif st == "cci":
            stage_cci(cell, tag)
        elif st == "sg":
            stage_sg(cell, tag, allow_download=not a.validate)
        elif st == "elev":
            stage_elev(cell, tag)
        elif st == "merge" and a.input == "v3all":
            log("[merge] v3all 은 e5 부분 산출만 쓴다(build_fidelity_base_v4 가 읽는다)")
        elif st == "merge":
            o = merge(cell, tag)
            if a.validate:
                rep = validate(o)
                (EXT / "cov_validate_meta.json").write_text(json.dumps(dict(
                    created=time.strftime("%Y-%m-%d %H:%M"), input="v3 CALM_* 행(E3 규약으로 공변량을 붙인 행)",
                    n=int(len(o)), compare=rep), ensure_ascii=False, indent=1))
                log(json.dumps(rep, ensure_ascii=False, indent=0))
            else:
                le0 = (o.e5_tdd_soil <= 0).fillna(False)
                o.loc[le0, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan          # v3 규칙(개정 13)
                o["_soil_le0"] = le0.astype(int)
                meta = cov_meta(o, tag, timing)
                o = o.drop(columns=["_soil_le0"])
                o.to_csv(EXT / "ext_cells_v1_cov.csv", index=False)
                (EXT / "ext_cells_v1_cov_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
                log(f"[merge] ext_cells_v1_cov.csv {o.shape} · 토양 도일 0 이하 결측 {meta['soil_tdd_le0_set_nan']} · "
                    f"빙하 격자 {meta['e5_glacier_grid']}")
        timing[st] = round(time.time() - t1, 1)
    log(f"[done] {timing} · 합계 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
