"""SoilGrids 토양 공변량을 ISRIC WCS(2.0.1)로 취득해 다지역 셀에 부착한다.

배경: VRT(vsicurl) 전지구 접근은 지연이 커 실패했다. 대신 지역별 창을 IGH 좌표계
bbox 로 잘라 GetCoverage(GEOTIFF_INT16, SCALESIZE) 로 내려받는 검증된 WCS 레시피를 쓴다.

레시피(검증됨):
  URL = https://maps.isric.org/mapserv?map=/map/{PROP}.map&SERVICE=WCS&VERSION=2.0.1
        &REQUEST=GetCoverage&COVERAGEID={PROP}_{DEPTH}_mean&FORMAT=GEOTIFF_INT16
        &SUBSET=X(x0,x1)&SUBSET=Y(y0,y1)&SCALESIZE=X(nx),Y(ny)
  좌표계 = Interrupted Goode Homolosine(IGH). 셀 lat/lon 을 pyproj(EPSG:4326→IGH,
  always_xy=True) 로 변환해 bbox 를 만든다. 반환 tif 는 CRS 태그가 None 이나 embedded
  transform 이 요청 bbox 와 정확히 일치하므로 그 transform 으로 row/col 을 계산해 샘플한다.

층(9): clay·sand·silt·soc·bdod·cfvo·phh2o 를 5-15cm, 추가로 soc 를 0-5·15-30cm(깊이 정보).

창 규약: IGH 북반구 lobe 경계는 lon=-40(북미 lobe -180..-40), 이후 lon 20·100·180 에서
분리된다. 어떤 창의 bbox 도 lobe 경계를 넘지 않도록 지역 클러스터별로 창을 나눈다.

산출:
  data/raw/soilgrids_wcs/<window>/<PROP>_<DEPTH>_mean.tif   (원 정수 int16, IGH)
  data/raw/soilgrids_wcs/windows_wcs_meta.json              (창·층 다운로드 메타)
실행: /home/anaconda3/bin/python scripts/1_data_prep/enrich_soilgrids_wcs.py --download
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = "/home/willy010313/Polar_Bigdata"
PROC = os.path.join(ROOT, "data/processed")
RAW = os.path.join(ROOT, "data/raw/soilgrids_wcs")
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
TR = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)

# 층: (PROP, DEPTH). soc 는 3깊이, 나머지는 5-15cm 단일.
LAYERS = [
    ("clay", "5-15cm"), ("sand", "5-15cm"), ("silt", "5-15cm"),
    ("bdod", "5-15cm"), ("cfvo", "5-15cm"), ("phh2o", "5-15cm"),
    ("soc", "0-5cm"), ("soc", "5-15cm"), ("soc", "15-30cm"),
]

# 값 스케일(÷factor → 물리단위) + 컬럼 접미 단위 주석
SCALE = {
    "clay": (10.0, "%"), "sand": (10.0, "%"), "silt": (10.0, "%"),
    "soc": (10.0, "dg/kg->g/kg"), "bdod": (100.0, "cg/cm3->kg/dm3"),
    "cfvo": (10.0, "vol%"), "phh2o": (10.0, "pH"), "nitrogen": (100.0, "g/kg"),
}

# 지역 창(WGS84 bbox: lon0,lon1,lat0,lat1). 여유 0.3° 포함. SCALESIZE 로 ~5km.
# 창은 lobe 경계(lon=-40,20,100,180)를 넘지 않는다.
WINDOWS = {
    # 북미 lobe(-180..-40): 알래스카+서캐나다+US_Alaska+Canada 를 한 창으로.
    "namerica": (-166.3, -112.7, 52.5, 71.7),
    # 레나(아시아, 100..180 lobe): Lena_RU + GTNPenv_RU(147E) 인접.
    "lena": (122.9, 148.0, 71.2, 73.9),
    # 레나 남부 확장은 불필요(레나는 71.5N 이북). GTNPenv_RU 147E(70.8N) 포함됨.
    # 서러시아(유럽 lobe -40..20 아님 → 20..100 lobe): 53..77E.
    "wrussia": (53.0, 77.0, 55.4, 71.2),
    # 추코트카(GTNPenv_RU 172E, 100..180 lobe).
    "chukotka": (171.5, 172.4, 67.1, 67.9),
    # 스발바르(GTNPenv_SJ, 유럽 lobe -40..20).
    "svalbard": (11.4, 16.5, 77.7, 79.3),
    # 알프스(GTNPenv_CH, 유럽 lobe).
    "alps": (7.4, 10.3, 46.1, 46.9),
    # 티베트(QTP_CN, 20..100 lobe).
    "qtp": (92.7, 93.5, 34.9, 35.6),
    # 남극(GTNPenv_AQ) 시도. SoilGrids 남극 커버 없을 수 있음.
    "antarctica": (-61.2, -60.2, -63.4, -62.6),
}

# 지역→창 매핑(부착 단계 사용; 한 셀이 여러 창에 들면 유효한 첫 창 채택).
REGION_WINDOWS = {
    "ABoVE_AK": ["namerica"], "ABoVE_CA": ["namerica"], "Canada": ["namerica"],
    "United States (Alaska)": ["namerica"], "GTNPenv_US": ["namerica"],
    "Lena_RU": ["lena"], "GTNPenv_RU": ["wrussia", "lena", "chukotka"],
    "GTNPenv_SJ": ["svalbard"], "GTNPenv_CH": ["alps"], "QTP_CN": ["qtp"],
    "GTNPenv_AQ": ["antarctica"],
}


def igh_bbox(lon0, lon1, lat0, lat1):
    """WGS84 bbox 모서리+변 중점을 IGH 로 변환해 안전한 X/Y 범위 산출."""
    lons = np.linspace(lon0, lon1, 9)
    lats = np.linspace(lat0, lat1, 9)
    xx, yy = [], []
    for lo in lons:
        for la in lats:
            x, y = TR.transform(lo, la)
            xx.append(x)
            yy.append(y)
    return min(xx), max(xx), min(yy), max(yy)


def scalesize(x0, x1, y0, y1, res_m=5000.0, cap=4000):
    """~res_m(기본 5km) 해상도 픽셀 수. 상한 cap 로 과대 요청 방지."""
    nx = int(round((x1 - x0) / res_m))
    ny = int(round((y1 - y0) / res_m))
    nx = max(2, min(nx, cap))
    ny = max(2, min(ny, cap))
    return nx, ny


def download(retries=3, force=False):
    os.makedirs(RAW, exist_ok=True)
    meta = {}
    for wname, (lon0, lon1, lat0, lat1) in WINDOWS.items():
        wdir = os.path.join(RAW, wname)
        os.makedirs(wdir, exist_ok=True)
        x0, x1, y0, y1 = igh_bbox(lon0, lon1, lat0, lat1)
        nx, ny = scalesize(x0, x1, y0, y1)
        meta[wname] = {"wgs84_bbox": [lon0, lon1, lat0, lat1],
                       "igh_bbox": [round(x0), round(x1), round(y0), round(y1)],
                       "scalesize": [nx, ny], "layers": {}}
        print(f"[window] {wname}: WGS84 lon[{lon0},{lon1}] lat[{lat0},{lat1}] "
              f"IGH X[{x0:.0f},{x1:.0f}] Y[{y0:.0f},{y1:.0f}] size {nx}x{ny}")
        for prop, depth in LAYERS:
            base = f"{prop}_{depth}_mean"
            path = os.path.join(wdir, base + ".tif")
            if os.path.exists(path) and not force and _valid_tif(path):
                st = _summ(path)
                meta[wname]["layers"][base] = {**st, "cached": True}
                print(f"   {base:18s} cached  {st}")
                continue
            url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
                   f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
                   f"&COVERAGEID={prop}_{depth}_mean&FORMAT=GEOTIFF_INT16"
                   f"&SUBSET=X({x0:.0f},{x1:.0f})&SUBSET=Y({y0:.0f},{y1:.0f})"
                   f"&SCALESIZE=X({nx}),Y({ny})")
            ok = False
            for attempt in range(1, retries + 1):
                t0 = time.time()
                r = subprocess.run(["curl", "-s", "--max-time", "120", "-o", path, url])
                if r.returncode == 0 and _valid_tif(path):
                    ok = True
                    break
                print(f"      retry {attempt}/{retries} ({prop}_{depth}) rc={r.returncode}")
                time.sleep(3)
            if ok:
                st = _summ(path)
                meta[wname]["layers"][base] = {**st, "sec": round(time.time() - t0, 1)}
                print(f"   {base:18s} OK  {st}  {time.time()-t0:.0f}s")
            else:
                meta[wname]["layers"][base] = {"error": "download_failed"}
                if os.path.exists(path):
                    os.remove(path)
                print(f"   {base:18s} FAILED")
    with open(os.path.join(RAW, "windows_wcs_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[download] meta -> {os.path.join(RAW, 'windows_wcs_meta.json')}")
    return meta


def _valid_tif(path):
    try:
        with rasterio.open(path) as src:
            _ = src.read(1)
        return os.path.getsize(path) > 200
    except Exception:
        return False


def _mask_nodata(band, nod):
    """SoilGrids nodata: src.nodata 우선, 없으면 <=-32768 또는 0(음수/비토지)마스킹.
    clay/sand/silt/soc/bdod/cfvo/phh2o 는 물리적으로 0 이 결측(해양·빙하) 신호다."""
    b = band.astype(float)
    if nod is not None:
        b = np.where(b == nod, np.nan, b)
    b = np.where(b <= -32768, np.nan, b)
    b = np.where(b <= 0, np.nan, b)  # 0/음수 = nodata(해양/빙하)
    return b


def _summ(path):
    with rasterio.open(path) as src:
        b = _mask_nodata(src.read(1), src.nodata)
        v = np.isfinite(b).mean()
        med = float(np.nanmedian(b)) if np.isfinite(b).any() else float("nan")
    return {"shape": list(src.shape), "valid_frac": round(float(v), 3),
            "median_raw": round(med, 1) if med == med else None}


def _sample(path, ix_x, ix_y):
    """IGH 좌표 배열에서 최근접 픽셀 샘플. 창 밖·nodata → NaN."""
    with rasterio.open(path) as src:
        b = _mask_nodata(src.read(1), src.nodata)
        fc, fr = (~src.transform) * (np.asarray(ix_x), np.asarray(ix_y))
        r = np.floor(fr).astype(int)
        c = np.floor(fc).astype(int)
        inb = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
        out = np.full(len(ix_x), np.nan)
        rr = np.clip(r, 0, src.height - 1)
        cc = np.clip(c, 0, src.width - 1)
        vals = b[rr, cc]
        out[inb] = vals[inb]
        return out


def col_name(prop, depth):
    """clay_5-15cm -> sg_clay_5_15 ; soc_0-5cm -> sg_soc_0_5."""
    return "sg_" + prop + "_" + depth.replace("cm", "").replace("-", "_")


def attach():
    df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v2.csv"), low_memory=False)
    n = len(df)
    ix_x, ix_y = TR.transform(df.lon.values, df.lat.values)
    ix_x = np.asarray(ix_x)
    ix_y = np.asarray(ix_y)

    sg_cols = []
    for prop, depth in LAYERS:
        col = col_name(prop, depth)
        sg_cols.append(col)
        factor = SCALE.get(prop, (1.0, ""))[0]
        acc = np.full(n, np.nan)
        base = f"{prop}_{depth}_mean.tif"
        # 각 셀을 그 지역의 창들에서 샘플(유효한 첫 창 채택).
        for wname in WINDOWS:
            path = os.path.join(RAW, wname, base)
            if not os.path.exists(path):
                continue
            # 이 창을 쓰는 지역의 셀만 대상.
            rmask = df.region.map(lambda rg: wname in REGION_WINDOWS.get(rg, [])).values
            if not rmask.any():
                continue
            idx = np.where(rmask & np.isnan(acc))[0]
            if idx.size == 0:
                continue
            s = _sample(path, ix_x[idx], ix_y[idx])
            fill = np.isfinite(s)
            acc[idx[fill]] = s[fill]
        acc = acc / factor
        df[col] = acc
        miss = float(np.isnan(acc).mean() * 100)
        med = float(np.nanmedian(acc)) if np.isfinite(acc).any() else float("nan")
        print(f"  {col:16s} 결측 {miss:5.1f}%  중앙 {med:.3g}")

    # 지역별 유효율.
    rows = []
    for reg, g in df.groupby("region"):
        valid = {c: round(100 * float(np.isfinite(g[c]).mean()), 1) for c in sg_cols}
        avg = round(float(np.mean(list(valid.values()))), 1)
        rows.append({"region": reg, "n": len(g), "sg_valid_avg_pct": avg, **valid})
    vdf = pd.DataFrame(rows).sort_values("n", ascending=False)
    vdf.to_csv(os.path.join(PROC, "soilgrids_wcs_valid_by_region.csv"), index=False)
    print("\n[지역별 sg_ 평균 유효율]")
    print(vdf[["region", "n", "sg_valid_avg_pct"]].to_string(index=False))

    out = os.path.join(PROC, "dl_dataset_cell_v3_soil.csv")
    df.to_csv(out, index=False)
    meta = {"created": "2026-07-20", "source": "ISRIC SoilGrids WCS 2.0.1 (IGH)",
            "base": "dl_dataset_cell_v2.csv", "n_cells": n,
            "sg_cols": sg_cols, "n_sg": len(sg_cols),
            "layers": [f"{p}_{d}" for p, d in LAYERS],
            "windows": list(WINDOWS.keys()),
            "overall_valid_pct": {c: round(100 * float(np.isfinite(df[c]).mean()), 1) for c in sg_cols},
            "scale_note": ("부착값=물리단위(원 정수 ÷ factor). clay/sand/silt/cfvo %, "
                           "soc g/kg, bdod kg/dm3, phh2o pH."),
            "nodata_note": "src.nodata 우선, 없으면 값<=0(해양/빙하) 또는 <=-32768 마스킹.",
            "note": "GTNPenv_AQ(남극)는 SoilGrids 육지 커버 밖일 수 있어 NaN 가능."}
    with open(os.path.join(PROC, "dl_dataset_cell_v3_soil_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}  ({df.shape[1]}컬럼, sg_ {len(sg_cols)}개)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="WCS 다운로드 수행")
    ap.add_argument("--force", action="store_true", help="캐시 무시 재다운로드")
    ap.add_argument("--attach-only", action="store_true", help="다운로드 생략, 부착만")
    args = ap.parse_args()
    if args.download or args.force:
        download(force=args.force)
    if not args.attach_only or args.download or args.force:
        attach()


if __name__ == "__main__":
    main()
