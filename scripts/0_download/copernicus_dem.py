"""Copernicus DEM GLO-30 (30m 고해상 지형) — ALT 관측점이 있는 1°타일만 다운로드(무계정 AWS).
지형(slope/aspect/TWI)으로 공간 DL의 해상도 병목 해소용.
"""
import os, urllib.request, urllib.error
import pandas as pd

OUT = "data/raw/dem"
os.makedirs(OUT, exist_ok=True)
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
tiles = pd.read_csv("data/processed/dem_tiles_needed.csv").sort_values("n", ascending=False)

def tile_name(tlat, tlon):
    ns = f"N{abs(int(tlat)):02d}" if tlat >= 0 else f"S{abs(int(tlat)):02d}"
    ew = f"E{abs(int(tlon)):03d}" if tlon >= 0 else f"W{abs(int(tlon)):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"

ok = miss = 0; total = 0
for _, r in tiles.iterrows():
    name = tile_name(r.tlat, r.tlon)
    dst = os.path.join(OUT, name + ".tif")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        ok += 1; total += os.path.getsize(dst); continue
    url = f"{BASE}/{name}/{name}.tif"
    try:
        urllib.request.urlretrieve(url, dst)
        sz = os.path.getsize(dst); total += sz; ok += 1
        print(f"OK  {name}  ({sz/1e6:.1f}MB, {int(r.n)}점)")
    except urllib.error.HTTPError as e:
        miss += 1
        if os.path.exists(dst):
            os.remove(dst)
        print(f"MISS {name}  (HTTP {e.code}, 해양/비존재, {int(r.n)}점)")
    except Exception as e:
        miss += 1
        print(f"ERR {name}  {str(e)[:60]}")
print(f"\n완료: {ok}개 다운({total/1e9:.2f}GB), {miss}개 없음. → {OUT}")
