"""SoilGrids 250m 알래스카 창 다운로드(무계정, ISRIC WebDAV VRT). ALT 결측 공변량=토양.
IGH(Interrupted Goode Homolosine) 원좌표에서 알래스카 bbox 창만 windowed-read → 로컬 GeoTIFF.
층: soc(유기탄소) 3깊이, bdod(용적밀도), clay, sand — 단열·수분보유로 ALT 좌우.
산출: data/raw/soilgrids/<layer>.tif  (값 ×10 스케일: soc dg/kg, bdod cg/cm³, clay/sand g/kg)
"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

OUT = "data/raw/soilgrids"; os.makedirs(OUT, exist_ok=True)
BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
LAYERS = ["soc/soc_0-5cm_mean", "soc/soc_5-15cm_mean", "soc/soc_15-30cm_mean",
          "bdod/bdod_5-15cm_mean", "clay/clay_5-15cm_mean", "sand/sand_5-15cm_mean"]

# 알래스카 bbox(lon -168..-140, lat 52..72) → IGH 좌표
tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
lons = [-168, -140, -168, -140, -154]; lats = [52, 52, 72, 72, 62]
xs, ys = tr.transform(lons, lats)
minx, maxx, miny, maxy = min(xs) - 3e4, max(xs) + 3e4, min(ys) - 3e4, max(ys) + 3e4
print(f"IGH 창: x {minx:.0f}~{maxx:.0f}, y {miny:.0f}~{maxy:.0f}")

for lyr in LAYERS:
    name = lyr.split("/")[-1]
    dst = f"{OUT}/{name}.tif"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print("skip", name); continue
    try:
        with rasterio.open(f"{BASE}/{lyr}.vrt") as src:
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            arr = src.read(1, window=win)
            wt = src.window_transform(win)
            prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                        dtype=arr.dtype, crs=IGH, transform=wt, nodata=src.nodata,
                        compress="deflate")
            with rasterio.open(dst, "w", **prof) as d:
                d.write(arr, 1)
        v = arr[arr != src.nodata].astype(float)
        print(f"OK {name}: {arr.shape}, 유효 {len(v):,}, 값(×10) 5/50/95%={np.percentile(v,[5,50,95]).round(0) if len(v) else 'NA'}")
    except Exception as e:
        print(f"ERR {name}: {str(e)[:120]}")
print("완료 →", OUT)
