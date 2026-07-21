"""SoilGrids sg_* 공변량을 v2 전 셀에 부착 → dl_dataset_cell_v3_soil.csv.

각 셀 좌표를 IGH 로 변환해, 해당 셀이 속한 지역 창(WINDOWS)의 tif 에서 windowed/최근접 샘플.
한 셀은 여러 창에 들 수 있으므로(창이 겹치는 경계), 값이 유효한 첫 창을 채택한다.
nodata → NaN. 지역별 sg_ 유효율 보고(전지구 커버라 결측 적어야 정상).

산출: data/processed/dl_dataset_cell_v3_soil.csv  (v2 + sg_* 컬럼, 기존 컬럼 불변)
      data/processed/soilgrids_valid_by_region.csv  (지역별 sg_ 유효율)
      data/processed/dl_dataset_cell_v3_soil_meta.json
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.env import Env
from pyproj import Transformer

ROOT = "/home/willy010313/Polar_Bigdata"
PROC = os.path.join(ROOT, "data/processed")
RAW = os.path.join(ROOT, "data/raw/soilgrids_multi")
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
TR = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)

# 값 스케일(÷factor → 물리단위). 부착값은 물리단위로 변환해 저장(해석·물리연결 용이).
SCALE = {
    "soc": (10.0, "g/kg"), "bdod": (100.0, "kg/dm3"), "clay": (10.0, "%"),
    "sand": (10.0, "%"), "silt": (10.0, "%"), "cfvo": (10.0, "vol%"),
    "phh2o": (10.0, "pH"), "nitrogen": (100.0, "g/kg"),
}


def col_name(tif_base):
    """clay_5-15cm_mean -> sg_clay_5_15."""
    b = tif_base.replace("_mean", "").replace("cm", "").replace("-", "_")
    return "sg_" + b


def var_of(tif_base):
    return tif_base.split("_")[0]


def sample_tif(path, ix_x, ix_y):
    """IGH 좌표 배열에서 최근접 픽셀 샘플. 창 밖·nodata → NaN."""
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        nod = src.nodata
        fc, fr = (~src.transform) * (np.asarray(ix_x), np.asarray(ix_y))
        r = np.floor(fr).astype(int)
        c = np.floor(fc).astype(int)
        inb = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
        out = np.full(len(ix_x), np.nan)
        rr = np.clip(r, 0, src.height - 1)
        cc = np.clip(c, 0, src.width - 1)
        vals = band[rr, cc]
        if nod is not None:
            vals = np.where(vals == nod, np.nan, vals)
        out[inb] = vals[inb]
        return out


def main():
    df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v2.csv"), low_memory=False)
    n = len(df)
    ix_x, ix_y = TR.transform(df.lon.values, df.lat.values)
    ix_x = np.asarray(ix_x); ix_y = np.asarray(ix_y)

    windows = sorted([d for d in glob.glob(os.path.join(RAW, "*")) if os.path.isdir(d)])
    # 층 이름 집합(창 간 공통). 창별로 존재하는 tif 만 사용.
    layer_bases = set()
    for w in windows:
        for t in glob.glob(os.path.join(w, "*.tif")):
            layer_bases.add(os.path.basename(t).replace(".tif", ""))
    layer_bases = sorted(layer_bases)
    print(f"[layers] {len(layer_bases)}: {[col_name(b) for b in layer_bases]}")

    env = dict(GDAL_NUM_THREADS="ALL_CPUS")
    with Env(**env):
        for base in layer_bases:
            col = col_name(base)
            var = var_of(base)
            factor, unit = SCALE.get(var, (1.0, ""))
            acc = np.full(n, np.nan)
            for w in windows:
                path = os.path.join(w, base + ".tif")
                if not os.path.exists(path):
                    continue
                s = sample_tif(path, ix_x, ix_y)
                # 유효한 첫 창 채택(기존 NaN 만 채움)
                fill = np.isnan(acc) & np.isfinite(s)
                acc[fill] = s[fill]
            acc = acc / factor  # 물리단위
            df[col] = acc
            miss = np.isnan(acc).mean() * 100
            med = np.nanmedian(acc) if np.isfinite(acc).any() else np.nan
            print(f"  {col:16s} 결측 {miss:5.1f}%  중앙 {med:.3g} {unit}")

    sg_cols = [col_name(b) for b in layer_bases]
    # 지역별 유효율(sg_ 컬럼 전체 평균 유효율)
    rows = []
    for reg, g in df.groupby("region"):
        valid = {c: round(100 * np.isfinite(g[c]).mean(), 1) for c in sg_cols}
        avg = round(np.mean(list(valid.values())), 1)
        rows.append({"region": reg, "n": len(g), "sg_valid_avg_pct": avg, **valid})
    vdf = pd.DataFrame(rows).sort_values("n", ascending=False)
    vdf.to_csv(os.path.join(PROC, "soilgrids_valid_by_region.csv"), index=False)
    print("\n[지역별 sg_ 평균 유효율]")
    print(vdf[["region", "n", "sg_valid_avg_pct"]].to_string(index=False))

    out = os.path.join(PROC, "dl_dataset_cell_v3_soil.csv")
    df.to_csv(out, index=False)
    meta = {"n_cells": n, "sg_cols": sg_cols, "n_sg": len(sg_cols),
            "overall_valid_pct": {c: round(100 * np.isfinite(df[c]).mean(), 1) for c in sg_cols},
            "windows": [os.path.basename(w) for w in windows],
            "scale_note": "부착값=물리단위(원 정수 ÷ factor). soc g/kg, bdod kg/dm3, clay/sand/silt %, cfvo vol%, phh2o pH, nitrogen g/kg"}
    with open(os.path.join(PROC, "dl_dataset_cell_v3_soil_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}  ({df.shape[1]}컬럼, sg_ {len(sg_cols)}개)")


if __name__ == "__main__":
    main()
