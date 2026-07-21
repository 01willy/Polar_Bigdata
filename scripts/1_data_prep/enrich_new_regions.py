"""신규 지역(Lena·QTP·GTNP envelope) ALT 위치의 공변량 부착.

단계
1) 신규 위치의 1° floor DEM 타일 목록 -> data/processed/dem_tiles_needed.csv
   (기존 타일 행은 보존하고 신규 타일만 추가; copernicus_dem.py 입력 형식 tlat,tlon,n)
   --tiles-only 로 실행하면 여기서 종료. 이후 scripts/0_download/copernicus_dem.py 실행.
2) 지형 6종: terrain_features_dem.py와 동일 로직(33x33 창, gradient slope,
   tpi=중심-창평균, rough=창 표준편차). 패치 npy는 만들지 않는다.
3) e5_* 8종: era5land_covariates.py 파생 로직 재사용(2015-2020 월기후,
   maat/tdd/fdd/sqrt_tdd/twarm/tcold/stl1/swe). ERA5-Land는 land-only이므로
   최근접 격자가 NaN이면 0.2°(±2셀) 내 최근접 유효 격자로 폴백하고 사용 수를 로그.
   레나 삼각주 내부(예: Samoylov 72.37N 126.48E)는 0.1° 격자에서 수로·습지가
   수역으로 마스크되어 최근접 land까지 최대 0.45°가 필요하다. 이에 0.5°(±5셀)
   확장 폴백을 2단으로 두고 사용 수를 별도 로그한다(기후값은 평탄 저지대에서
   0.5°≈15km(경도, 72N 기준) 거리의 공간변동이 작아 근사 타당).
4) cci_* 2종: enrich_cci_cell.py 로직(25개 연도파일 다년평균, 최근접 샘플)을
   신규 위치에 적용. 10° 블록 단위 bbox 서브셋으로 메모리 절약.

라벨 QC 컬럼 보존
- qc(GTNPenv: high/approx, 비계절 샘플링 식별), qc_flag(ALLena: 1=그림추출),
  borehole_id·site(GTNPenv·QTEC 추적용), censor_flag(QTEC: 1=censored 연도 제외로
  라벨 하한 성격)를 탈락시키지 않고 그대로 통과시킨다.
- 소스에 없는 컬럼은 NaN, censor_flag만 0으로 채운다(비해당=비censored).

산출: data/processed/new_regions_covariates.csv (라벨 + QC + 공변량)
실행(ROOT): /home/anaconda3/bin/python scripts/1_data_prep/enrich_new_regions.py [--tiles-only]
"""
import calendar
import glob
import os
import sys

import numpy as np
import pandas as pd

TILES = "data/processed/dem_tiles_needed.csv"
DEM = "data/raw/dem"
NC = "data/raw/era5land/nh_monthly_2015-2020.nc"
OUT = "data/processed/new_regions_covariates.csv"
LABEL_COLS = ["lat", "lon", "region", "alt_cm", "alt_sd", "alt_min", "alt_max",
              "n_obs", "year_min", "year_max", "n_years",
              "qc", "qc_flag", "borehole_id", "site", "censor_flag"]

# ---------- 신규 위치 로드 ----------
parts = []
for f in ["data/processed/alt_allena_cell.csv",
          "data/processed/alt_qtec_cell.csv",
          "data/processed/alt_gtnp_envelope_cell.csv"]:
    d = pd.read_csv(f).reindex(columns=LABEL_COLS)   # 소스에 없는 컬럼은 NaN
    parts.append(d)
loc = pd.concat(parts, ignore_index=True)
loc["censor_flag"] = loc["censor_flag"].fillna(0).astype(int)
print(f"[locs] 신규 위치 {len(loc):,}개: {loc.region.str.split('_').str[0].value_counts().to_dict()}")

# ---------- 1) DEM 타일 목록 ----------
loc["tlat"] = np.floor(loc.lat).astype(int)
loc["tlon"] = np.floor(loc.lon).astype(int)
new_tiles = loc.groupby(["tlat", "tlon"]).size().reset_index(name="n")
old = pd.read_csv(TILES)
oldkey = set(zip(old.tlat, old.tlon))
add = new_tiles[[(a, b) not in oldkey for a, b in zip(new_tiles.tlat, new_tiles.tlon)]]
tiles = pd.concat([old, add], ignore_index=True)
tiles.to_csv(TILES, index=False)
print(f"[tiles] 기존 {len(old)} + 신규 {len(add)} = {len(tiles)}개 -> {TILES}")
if "--tiles-only" in sys.argv:
    sys.exit(0)

import rasterio  # noqa: E402  (tiles-only 경로에서는 불필요)
import xarray as xr  # noqa: E402

# ---------- 2) 지형 6종 (terrain_features_dem.py 로직) ----------
W = 33
H = W // 2
MPD = 111320.0

def tname(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"

feats = np.full((len(loc), 6), np.nan)
missing_tiles = []
for (tlat, tlon), g in loc.groupby(["tlat", "tlon"]):
    path = os.path.join(DEM, tname(tlat, tlon) + ".tif")
    if not os.path.exists(path):
        missing_tiles.append((tlat, tlon, len(g)))
        continue
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        arr[arr == ds.nodata] = np.nan
        ny, nx = arr.shape
        dy = MPD * abs(ds.transform.e)
        for idx, r in g.iterrows():
            row, col = ds.index(r.lon, r.lat)
            # 타일 경계(위도 정수 등) 1px 이내 이탈은 가장자리 픽셀로 클램프
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
            i, j = row - r0, col - c0
            if np.isnan(win[i, j]):
                # 중심 픽셀이 nodata(해안 수역)이면 창 내 최근접 유효 픽셀로 대체
                yy, xx = np.where(np.isfinite(win))
                k = np.argmin((yy - i) ** 2 + (xx - j) ** 2)
                i, j = int(yy[k]), int(xx[k])
            gy, gx = np.gradient(win, dy, dx)
            sl = np.degrees(np.arctan(np.hypot(gy[i, j], gx[i, j])))
            asp = np.arctan2(-gy[i, j], gx[i, j])
            cen = win[i, j]
            feats[idx] = [cen, sl, np.sin(asp), np.cos(asp),
                          cen - np.nanmean(win), float(np.nanstd(win))]
for k, nm in enumerate(["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos",
                        "dem_tpi", "dem_rough"]):
    loc[nm] = feats[:, k]
nd = int(np.isfinite(feats[:, 0]).sum())
print(f"[dem] 지형특징 {nd:,}/{len(loc):,} 위치 유효, 미보유 타일 {len(missing_tiles)}개 {missing_tiles}")

# ---------- 3) e5_* 8종 (era5land_covariates.py 로직 + 0.2° 폴백) ----------
ds = xr.open_dataset(NC)
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
tvar = "t2m" if "t2m" in clim else [v for v in clim.data_vars if "t2m" in v][0]
svar = "sd" if "sd" in clim else [v for v in clim.data_vars if v == "sd" or "snow" in v][0]
gvar = "stl1" if "stl1" in clim else [v for v in clim.data_vars if "stl1" in v][0]
glat = clim["latitude"].values
glon = clim["longitude"].values
t2m = clim[tvar].values - 273.15
stl = clim[gvar].values - 273.15
swe = clim[svar].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])
land = np.isfinite(t2m).all(axis=0)   # 12개월 모두 유효 = land 격자
res = abs(float(glat[1] - glat[0]))   # 0.1°
NB1 = int(round(0.2 / res))           # 1단 폴백 반경 ±2셀(0.2°)
NB2 = int(round(0.5 / res))           # 2단 확장 폴백 ±5셀(0.5°)

def nearest(arr, v):
    return int(np.abs(arr - v).argmin())

def fallback(iy, ix, la_, lo_, nb):
    """±nb셀 내 최근접 유효(land) 격자. 없으면 None."""
    best = None
    for dy_ in range(-nb, nb + 1):
        for dx_ in range(-nb, nb + 1):
            y2, x2 = iy + dy_, (ix + dx_) % len(glon)
            if 0 <= y2 < len(glat) and land[y2, x2]:
                d2 = (glat[y2] - la_) ** 2 + (glon[x2] - lo_) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, y2, x2)
    return best

e5 = {k: np.full(len(loc), np.nan) for k in
      ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]}
n_fb1 = n_fb2 = n_out = 0
for pos, (idx, r) in enumerate(loc.iterrows()):
    if not (glat.min() <= r.lat <= glat.max()):
        n_out += 1
        continue
    iy, ix = nearest(glat, r.lat), nearest(glon, r.lon)
    if not land[iy, ix]:
        best = fallback(iy, ix, r.lat, r.lon, NB1)
        if best is not None:
            n_fb1 += 1
        else:
            best = fallback(iy, ix, r.lat, r.lon, NB2)
            if best is not None:
                n_fb2 += 1
        if best is None:
            n_out += 1
            continue
        iy, ix = best[1], best[2]
    tm = t2m[:, iy, ix]
    tdd = float(np.nansum(np.clip(tm, 0, None) * days))
    fdd = float(np.nansum(np.clip(-tm, 0, None) * days))
    e5["e5_maat"][pos] = float(np.nanmean(tm))
    e5["e5_tdd"][pos] = tdd
    e5["e5_fdd"][pos] = fdd
    e5["e5_sqrt_tdd"][pos] = float(np.sqrt(tdd))
    e5["e5_twarm"][pos] = float(np.nanmax(tm))
    e5["e5_tcold"][pos] = float(np.nanmin(tm))
    e5["e5_stl1"][pos] = float(np.nanmean(stl[:, iy, ix]))
    e5["e5_swe"][pos] = float(np.nanmean(swe[:, iy, ix]))
for k, v in e5.items():
    loc[k] = v
print(f"[e5] 유효 {int(np.isfinite(e5['e5_maat']).sum()):,}/{len(loc):,} "
      f"(폴백 0.2° {n_fb1}건, 확장 0.5° {n_fb2}건, 커버리지 밖/폴백 실패 {n_out}건)")

# ---------- 4) cci_* 2종 (enrich_cci_cell.py 로직, 10° 블록 bbox) ----------
files = sorted(glob.glob("data/raw/cci_alt/*.nc"))
cci_vals = np.full(len(loc), np.nan)
loc["_blk"] = list(zip((loc.lat // 10).astype(int), (loc.lon // 10).astype(int)))

def nearest_idx(coords, vals):
    order = np.argsort(coords)
    cs = coords[order]
    j = np.searchsorted(cs, vals)
    j = np.clip(j, 1, len(cs) - 1)
    left, right = cs[j - 1], cs[j]
    j = np.where(np.abs(vals - left) <= np.abs(vals - right), j - 1, j)
    return order[j]

for blk, g in loc.groupby("_blk"):
    la, lo = g.lat.values, g.lon.values
    mla, Mla = la.min() - 0.2, la.max() + 0.2
    mlo, Mlo = lo.min() - 0.2, lo.max() + 0.2
    acc = cnt = latc = lonc = None
    for f in files:
        d = xr.open_dataset(f)
        if not (float(d.lat.min()) <= Mla and float(d.lat.max()) >= mla):
            d.close()
            break                      # CCI 커버리지(25-85N) 밖 블록
        sub = d["ALT"].sel(lat=slice(mla, Mla), lon=slice(mlo, Mlo))
        if sub.ndim == 3:
            sub = sub.isel(time=0)
        a = sub.values.astype("float32")
        if acc is None:
            acc = np.zeros_like(a)
            cnt = np.zeros_like(a)
            latc, lonc = sub.lat.values, sub.lon.values
        v = np.isfinite(a)
        acc[v] += a[v]
        cnt[v] += 1
        d.close()
    if acc is None or acc.size == 0:
        continue
    mean_cm = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) * 100.0
    iy = nearest_idx(latc, la)
    ix = nearest_idx(lonc, lo)
    cci_vals[loc.index.get_indexer(g.index)] = mean_cm[iy, ix]
loc["cci_alt"] = np.round(cci_vals, 2)
loc["cci_valid"] = np.isfinite(cci_vals).astype(int)
print(f"[cci] 유효 {int(loc.cci_valid.sum()):,}/{len(loc):,} "
      f"(중앙 {np.nanmedian(cci_vals):.1f}cm)")

loc = loc.drop(columns=["tlat", "tlon", "_blk"])
loc.to_csv(OUT, index=False)
print(f"[saved] {OUT}: {len(loc):,}행 x {loc.shape[1]}열")
for reg, g in loc.groupby(loc.region.str.replace(r"GTNPenv_.*", "GTNPenv", regex=True)):
    print(f"  {reg:10s} n={len(g):5,}  e5_maat {np.nanmedian(g.e5_maat):6.1f}°C  "
          f"dem_elev {np.nanmedian(g.dem_elev):7.1f}m  cci {np.nanmedian(g.cci_alt):6.1f}cm  "
          f"alt {g.alt_cm.median():6.1f}cm")
