"""SoilGrids 250m 다지역 창 다운로드(무계정, ISRIC WebDAV VRT).

soilgrids_alaska.py 확장: 알래스카뿐 아니라 v2 셀이 존재하는 신규 지역도 다운로드.
IGH(Interrupted Goode Homolosine) 원좌표에서 창만 windowed-read → 로컬 GeoTIFF.

취득 전략(ISRIC VRT windowed read 는 폭넓은 창에서 매우 느림 → 실측: 2°×2° 한 층 ~70s):
- 고정 대륙 창 대신 v2 셀이 실제로 존재하는 2° 격자 bin 만 골라 tight 창을 다운로드한다.
- 셀이 없는 격자는 건너뛰어 취득 데이터·시간을 최소화한다. 전 셀은 어느 창엔가 반드시 포함된다.
- 창별 tif 를 따로 저장(창마다 IGH transform 상이). 부착 단계에서 셀이 든 창 tif 에서 샘플.

층(값 스케일은 ISRIC 관례, 원 정수 ×scale 유지. 부착 단계에서 물리단위로 변환):
  soc  0-5·5-15·15-30cm  유기탄소  dg/kg   (÷10 → g/kg)   단열·수분보유
  bdod 5-15cm            용적밀도  cg/cm3  (÷100 → kg/dm3) 열용량·밀도
  clay 5-15cm            점토      g/kg    (÷10 → %)        수분보유
  sand 5-15cm            모래      g/kg    (÷10 → %)        배수·열전도
  silt 5-15cm            실트      g/kg    (÷10 → %)        입도
토양 유기탄소·수분보유·밀도는 활성층 단열/열특성의 직접 물리 인자이며 Stefan 방정식의
땅 열특성 E 와 직결된다. 전지구 커버라 다지역 셀에서 결측이 적어야 정상이다.

산출: data/raw/soilgrids_multi/<bin>/<layer>.tif  + windows_meta.json
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import rasterio
from rasterio.env import Env
from rasterio.windows import from_bounds
from pyproj import Transformer

ROOT = "/home/willy010313/Polar_Bigdata"
OUT = os.path.join(ROOT, "data/raw/soilgrids_multi")
os.makedirs(OUT, exist_ok=True)
BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

# 필수 물리층 7. 실패 층은 로그하고 계속. (cfvo·phh2o·nitrogen 은 취득시간 대비 물리 우선순위 낮아 제외)
LAYERS = [
    "soc/soc_0-5cm_mean", "soc/soc_5-15cm_mean", "soc/soc_15-30cm_mean",
    "bdod/bdod_5-15cm_mean",
    "clay/clay_5-15cm_mean",
    "sand/sand_5-15cm_mean",
    "silt/silt_5-15cm_mean",
]

BIN_DEG = 2.0        # 격자 크기(도)
MARGIN_M = 1.2e4     # IGH 창 여유(m)
TR = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)


def occupied_bins():
    """v2 셀에서 셀이 존재하는 2° bin 목록(bin_lon0, bin_lat0, n_cells)."""
    df = pd.read_csv(os.path.join(ROOT, "data/processed/dl_dataset_cell_v2.csv"), low_memory=False)
    bl = (np.floor(df.lon / BIN_DEG) * BIN_DEG).astype(int)
    bt = (np.floor(df.lat / BIN_DEG) * BIN_DEG).astype(int)
    g = pd.DataFrame({"bl": bl, "bt": bt}).value_counts().reset_index(name="n")
    g = g.sort_values("n", ascending=False).reset_index(drop=True)
    return [(int(r.bl), int(r.bt), int(r.n)) for r in g.itertuples()]


def igh_bounds(lo0, lo1, la0, la1):
    """WGS84 bbox → IGH bbox(가장자리·중앙 변환 min/max + 여유)."""
    lons = [lo0, lo1, lo0, lo1, (lo0 + lo1) / 2, (lo0 + lo1) / 2]
    lats = [la0, la0, la1, la1, la0, la1]
    xs, ys = TR.transform(lons, lats)
    xs = np.asarray(xs); ys = np.asarray(ys)
    return xs.min() - MARGIN_M, xs.max() + MARGIN_M, ys.min() - MARGIN_M, ys.max() + MARGIN_M


def fetch_layer(lyr, bdir, minx, miny, maxx, maxy):
    """단일 (bin, layer) 다운로드. dict 상태 반환. 스레드에서 개별 dataset 핸들 사용."""
    name = lyr.split("/")[-1]
    dst = os.path.join(bdir, f"{name}.tif")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return name, "cached", None
    t0 = time.time()
    try:
        with rasterio.open(f"{BASE}/{lyr}.vrt") as src:
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            if win.width <= 0 or win.height <= 0:
                return name, "empty", None
            arr = src.read(1, window=win)
            wt = src.window_transform(win)
            prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                        count=1, dtype=arr.dtype, crs=IGH, transform=wt,
                        nodata=src.nodata, compress="deflate")
            with rasterio.open(dst, "w", **prof) as d:
                d.write(arr, 1)
            v = arr[arr != src.nodata].astype(float)
            pct = np.percentile(v, [5, 50, 95]).round(0) if v.size else np.array([np.nan]*3)
            st = {"shape": list(arr.shape), "valid": int(v.size), "p5_50_95": pct.tolist(),
                  "sec": round(time.time()-t0, 1)}
            return name, st, f"OK {name}: {arr.shape} 유효 {v.size:,} 5/50/95%={pct} ({st['sec']}s)"
    except Exception as e:
        return name, f"ERR: {str(e)[:140]}", f"ERR {name}: {str(e)[:140]}"


def main():
    bins = occupied_bins()
    print(f"[bins] {len(bins)} occupied 2° bins, total cells={sum(b[2] for b in bins)}, "
          f"{len(LAYERS)} layers", flush=True)
    summary = {}
    env = dict(GDAL_HTTP_TIMEOUT="90", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".vrt,.tif", VSI_CACHE="TRUE",
               GDAL_HTTP_MULTIPLEX="YES", CPL_VSIL_CURL_CACHE_SIZE="200000000")
    with Env(**env):
        for bi, (bl, bt, ncell) in enumerate(bins):
            lo0, lo1, la0, la1 = bl, bl + BIN_DEG, bt, bt + BIN_DEG
            bname = f"lon{bl:+04d}_lat{bt:+03d}"
            bdir = os.path.join(OUT, bname)
            os.makedirs(bdir, exist_ok=True)
            minx, maxx, miny, maxy = igh_bounds(lo0, lo1, la0, la1)
            summary[bname] = {"wgs84_bbox": [lo0, lo1, la0, la1], "n_cells": ncell, "layers": {}}
            print(f"\n=== [{bi+1}/{len(bins)}] {bname}  cells={ncell} ===", flush=True)
            # 층별 병렬 다운로드(각 스레드 개별 dataset 핸들)
            with ThreadPoolExecutor(max_workers=len(LAYERS)) as ex:
                futs = {ex.submit(fetch_layer, lyr, bdir, minx, miny, maxx, maxy): lyr for lyr in LAYERS}
                for fu in as_completed(futs):
                    name, st, msg = fu.result()
                    summary[bname]["layers"][name] = st
                    if msg:
                        print("  " + msg, flush=True)
            with open(os.path.join(OUT, "windows_meta.json"), "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n완료 →", OUT, flush=True)


if __name__ == "__main__":
    main()
