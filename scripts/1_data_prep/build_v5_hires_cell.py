"""v5 고해상 정적 입력(셀 단위): NCSCD v2 토양 유기탄소(0–200 cm, 0.012°), CAVM 식생 단위(1 km, LAEA), ESA WorldCover 2021 10 m 피복 비율(셀 중심 ±500 m 창).
라벨 값은 읽지 않는다. 출력: data/processed/v5/hires_cell_v5.csv (loc_id, ncscd_soc, ncscd_gelisol, cavm_class, wc_tree, wc_shrub, wc_grass, wc_wet, wc_water, wc_bare, wc_moss, wc_n)
WorldCover 클래스: 10 나무, 20 관목, 30 초지, 40 경작, 50 건물, 60 나지, 70 눈·얼음, 80 수체, 90 습지, 95 맹그로브, 100 이끼·지의. 타일이 없는 셀은 결측.
실행: python3 scripts/1_data_prep/build_v5_hires_cell.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
cells = pd.read_csv(ROOT / "data" / "processed" / "v5" / "static_v5.csv", usecols=["loc_id", "lat", "lon"])
t0 = time.time(); out = cells[["loc_id"]].copy(); meta = {}

# NCSCD
nd = RAW / "ncscd_v2" / "unzipped" / "NCSCDv2_Circumpolar_raster_0012deg"
for key, fn in (("ncscd_soc", "NCSCDv2_Circumpolar_WGS84_SOCC200_0012deg.tif"), ("ncscd_gelisol", "NCSCDv2_Circumpolar_WGS84_gelisol_pct_0012deg.tif")):
    with rasterio.open(nd / fn) as r:
        cf, rf = ~r.transform * (cells.lon.values, cells.lat.values)
        rows, cols = np.floor(rf).astype(int), np.floor(cf).astype(int)
        a = r.read(1)
        rows = np.clip(rows, 0, r.height - 1); cols = np.clip(cols, 0, r.width - 1)
        v = a[rows, cols].astype(float); v[v == r.nodata] = np.nan
        out[key] = v; meta[key] = dict(file=fn, nodata=r.nodata, frac_finite=float(np.isfinite(v).mean()))
print(f"[ncscd] {time.time()-t0:.0f}s finite {np.isfinite(out.ncscd_soc).mean():.3f}", flush=True)

# CAVM
cv = RAW / "cavm_raster" / "v1" / "unzipped" / "raster_cavm_v1.tif"
with rasterio.open(cv) as r:
    tf = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True)
    x, y = tf.transform(cells.lon.values, cells.lat.values)
    cf, rf = ~r.transform * (np.asarray(x), np.asarray(y))
    rows, cols = np.floor(rf).astype(int), np.floor(cf).astype(int)
    a = r.read(1); rows = np.clip(rows, 0, r.height - 1); cols = np.clip(cols, 0, r.width - 1)
    v = a[rows, cols].astype(float); v[(v == r.nodata) | (v >= 90)] = np.nan      # 90 이상은 비식생(바다·빙하 등) 코드
    out["cavm_class"] = v; meta["cavm_class"] = dict(file=str(cv.relative_to(ROOT)), frac_finite=float(np.isfinite(v).mean()), classes=sorted(int(c) for c in np.unique(v[np.isfinite(v)])))
print(f"[cavm] {time.time()-t0:.0f}s finite {np.isfinite(out.cavm_class).mean():.3f}", flush=True)

# WorldCover: 셀 중심 ±0.0045°(약 500 m) 창의 클래스 비율
wdir = RAW / "worldcover_v200" / "tiles"
def tile_name(lat, lon):
    la = int(np.floor(lat / 3) * 3); lo = int(np.floor(lon / 3) * 3)
    return f"ESA_WorldCover_10m_2021_v200_{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}_Map.tif"
cells["tile"] = [tile_name(a, b) for a, b in zip(cells.lat, cells.lon)]
CL = {"wc_tree": 10, "wc_shrub": 20, "wc_grass": 30, "wc_wet": 90, "wc_water": 80, "wc_bare": 60, "wc_moss": 100}
res = {k: np.full(len(cells), np.nan) for k in CL}; res["wc_n"] = np.zeros(len(cells))
missing = 0
for tname, g in cells.groupby("tile"):
    p = wdir / tname
    if not p.exists():
        missing += len(g); continue
    with rasterio.open(p) as r:
        for i, (la, lo) in zip(g.index, zip(g.lat, g.lon)):
            c0f, r0f = ~r.transform * (lo - 0.0045, la + 0.0045); c1f, r1f = ~r.transform * (lo + 0.0045, la - 0.0045)
            r0, c0, r1, c1 = int(np.floor(r0f)), int(np.floor(c0f)), int(np.floor(r1f)), int(np.floor(c1f))
            r0, r1 = max(0, min(r0, r1)), min(r.height, max(r0, r1) + 1); c0, c1 = max(0, min(c0, c1)), min(r.width, max(c0, c1) + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            a = r.read(1, window=((r0, r1), (c0, c1)))
            n = int((a > 0).sum()); res["wc_n"][i] = n
            if n:
                for k, c in CL.items():
                    res[k][i] = float((a == c).sum()) / n
for k in res:
    out[k] = res[k]
meta["worldcover"] = dict(tiles_dir=str(wdir.relative_to(ROOT)), window_deg=0.009, cells_without_tile=int(missing), frac_finite=float(np.isfinite(out.wc_tree).mean()))
print(f"[worldcover] {time.time()-t0:.0f}s finite {np.isfinite(out.wc_tree).mean():.3f} missing tiles cells {missing}", flush=True)
out.to_csv(ROOT / "data" / "processed" / "v5" / "hires_cell_v5.csv", index=False)
meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S"); meta["elapsed_s"] = round(time.time() - t0, 1)
json.dump(meta, open(ROOT / "data" / "processed" / "v5" / "hires_cell_v5_meta.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "created"}, ensure_ascii=False)[:800])
