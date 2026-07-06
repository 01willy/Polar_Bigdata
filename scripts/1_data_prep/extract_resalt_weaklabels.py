"""A2: InSAR ReSALT 51 granule → DL 사전학습용 weak label 서브샘플.
granule당 최대 80k 픽셀 무작위 추출(유효 ALT만) → ~4M 점.
좌표: ABoVE Albers(NAD83) → WGS84 lat/lon. 산출: data/processed/resalt_weaklabels.parquet
"""
import glob, os
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

PER = 80_000
rng = np.random.default_rng(0)
albers = ("+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 "
          "+x_0=0 +y_0=0 +datum=NAD83 +units=m")
tf = Transformer.from_crs(albers, "EPSG:4326", always_xy=True)

parts = []
for f in sorted(glob.glob("data/raw/resalt/*.nc4")):
    site = os.path.basename(f).split("_")[2]
    ds = xr.open_dataset(f)
    alt = ds["alt"].values
    ok = np.isfinite(alt) & (alt > 0.05) & (alt < 3.0)
    iy, ix = np.where(ok)
    if len(iy) == 0:
        ds.close(); continue
    if len(iy) > PER:
        sel = rng.choice(len(iy), PER, replace=False)
        iy, ix = iy[sel], ix[sel]
    x = ds.x.values[ix]; y = ds.y.values[iy]
    lon, lat = tf.transform(x, y)

    def pick(name):
        if name in ds.data_vars:
            return ds[name].values[iy, ix]
        return np.full(len(iy), np.nan, dtype=np.float32)

    part = pd.DataFrame(dict(
        site=site, lat=lat.round(5), lon=lon.round(5),
        alt_cm=(alt[iy, ix] * 100).round(1),
        alt_unc_cm=(pick("alt_unc") * 100).round(1),
        sub_cm=(pick("sub") * 100).round(2),
        sw0=pick("Sw0").round(3),
    ))
    parts.append(part)
    ds.close()
    print(f"{site:8s} 유효 {ok.sum():>9,} → 추출 {len(part):>6,}")

wl = pd.concat(parts, ignore_index=True)
wl.to_parquet("data/processed/resalt_weaklabels.parquet", index=False)
print(f"\n[weak label] {len(wl):,} 점 / {wl.site.nunique()} 사이트 "
      f"/ ALT 5/50/95% = {np.percentile(wl.alt_cm,[5,50,95]).round(0)} cm")
print(f"lat {wl.lat.min():.1f}~{wl.lat.max():.1f}, lon {wl.lon.min():.1f}~{wl.lon.max():.1f}")
print("→ data/processed/resalt_weaklabels.parquet")
