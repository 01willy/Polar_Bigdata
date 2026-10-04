"""XL_map_products(지역별 고해상 ALT 지도와 제품 비교). 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md(서술, 판정 없음).

단계
  extract  알래스카·레나 1 km 격자 셀 중심에서 제품 값을 읽는다. 읽기 함수는 XG(scripts/3_deep_learning/x_product_comparison.py)의
           extract_cci(CCI v5), extract_tif(Wei 2026 v2, Aalto 2018), extract_nc_latlon(Yi·Kimball, 알래스카)를 그대로 쓴다(최근접 화소,
           화소 크기 1.5배 밖이면 결측, 단위 확인 기록). 값은 기간 평균(year_rule 'static')이다: CCI v5 1997–2023, Wei 2000–2024,
           Aalto 2018 기준기 2000–2014(단일 래스터), Yi·Kimball 2001–2015. zip 의 md5 는 XG 추출 때 대조했으므로 다시 계산하지 않는다(check_md5=False).
  summary  표시 셀(회색 아님)에서 우리 지도(라벨 전량 재보정 Stefan + 잔차 ML), 재보정 Stefan, 각 제품의 영역 평균·사분위, 우리 − 제품 차의
           평균·사분위, 공간 상관(피어슨 r, 스피어먼 ρ, 두 값이 모두 있는 셀)을 표로 쓴다. 지도에 정확도 주장을 붙이지 않는다(정확도는 XG, XK).
           0.1° 표시판: 우리 지도를 ERA5-Land 격자 셀(색인 floor((좌표 + 0.05)/0.1))의 셀 평균으로 바꾼 열(ours_e5mean)을 함께 쓴다.
산출 data/processed/xbatch/XL_map_products/{xl_alaska_products_v1.csv.gz, xl_lena_products_v1.csv.gz, xl_summary_v1.csv, xl_meta.json}
실행 nice -n 10 python3 scripts/2_evaluation/xl_map_products_v1.py --stage extract,summary
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2" if __name__ == "__main__" else os.environ.get(_v, "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
import warnings                                                                                         # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "scripts" / "2_evaluation"), str(ROOT / "scripts" / "3_deep_learning"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import map_alaska_v1 as MA                                                                              # noqa: E402

OUT = ROOT / "data" / "processed" / "xbatch" / "XL_map_products"
PROC = ROOT / "data" / "processed"
PRODUCTS = ("cci5", "wei", "aalto", "yk")
LABEL = dict(ours="Anchor + residual ML", stefan="Recalibrated Stefan", cci5="CCI v5 (1997–2023 mean)", wei="Wei 2026 (2000–2024 mean)",
             aalto="Aalto 2018 (2000–2014)", yk="Yi-Kimball (2001–2015 mean)")


def region_cells(key):
    if key == "alaska":
        pp = PROC / "map_alaska" / "alaska_pred_v1.csv.gz"
        if not pp.exists():                                  # 추출만: 격자 셀 목록(격자·예측 표와 같은 순서)
            g = pd.read_csv(PROC / "map_alaska" / "alaska_grid_cells_v1.csv", dtype={"cell_id": str})
            return g.assign(gray=np.nan, ours=np.nan, stefan=np.nan)[["cell_id", "lat", "lon", "gray", "ours", "stefan"]]
        p = pd.read_csv(pp, dtype={"cell_id": str})
        return p.rename(columns={"pred_r1": "ours", "pred_p1": "stefan"})[["cell_id", "lat", "lon", "gray", "ours", "stefan"]]
    p = pd.read_csv(PROC / "map_lena" / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
    return p.rename(columns={"pred_c": "ours", "m_P1_all": "stefan"})[["cell_id", "lat", "lon", "gray", "ours", "stefan"]]


def extract(key, XP, cells):
    c = cells[["lat", "lon"]].copy()
    c["year_min"] = np.nan; c["year_max"] = np.nan
    out = pd.DataFrame(dict(cell_id=cells.cell_id.values))
    rec = {}
    jobs = [("cci5", "cci5y"), ("wei", "wei"), ("aalto", "aalto")] + ([("yk", "yk")] if key == "alaska" else [])
    for name, base in jobs:
        spec = XP.PRODUCTS[base]
        t0 = time.time()
        if spec["kind"] == "cci_nc":
            V, V3, years, dist, r = XP.extract_cci(spec, c)
        elif spec["kind"] == "zip_tif":
            V, V3, years, dist, r = XP.extract_tif(spec, c, check_md5=False)
        else:
            V, V3, years, dist, r = XP.extract_nc_latlon(spec, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = np.nanmean(V, axis=1) if V.shape[1] > 1 else V[:, 0]
        out[name] = val
        out[f"{name}_n_years"] = np.isfinite(V).sum(1)
        out[f"{name}_pix_km"] = np.round(np.asarray(dist, float), 3)
        rec[name] = dict(base=base, period=list(spec["period"]), years=[y for y in years if y is not None], n_valid=int(np.isfinite(val).sum()),
                         seconds=round(time.time() - t0, 1), **{k: v for k, v in r.items() if k in ("crs", "res_deg", "res", "units", "shape", "nodata",
                                                                                                      "fill_value", "files")})
        print(f"[xl] {key} {name}: 유효 {rec[name]['n_valid']:,}/{len(c):,} · {rec[name]['seconds']}s", flush=True)
        del V, V3
    return out, rec


def e5_mean(cells, col):
    iy = np.floor((cells.lat.values + 0.05) / 0.1).astype(np.int64)
    ix = np.floor((cells.lon.values + 0.05) / 0.1).astype(np.int64)
    k = pd.Series(iy).astype(str) + "_" + pd.Series(ix).astype(str)
    v = pd.Series(cells[col].values)
    m = v.groupby(k.values).transform("mean")
    return m.values


def summary_rows(key, d):
    from scipy.stats import pearsonr, spearmanr
    sh = (d.gray.values == 0) & np.isfinite(d.ours.values)
    rows = []

    def q(v):
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), mean=float(v.mean()), p25=float(np.percentile(v, 25)), p50=float(np.median(v)), p75=float(np.percentile(v, 75))) if len(v) else dict(n=0)
    for col in ["ours", "stefan", "ours_e5mean"] + [p for p in PRODUCTS if p in d]:
        v = d[col].values.astype(float)[sh]
        r = dict(region=key, item=col, label=LABEL.get(col, "Anchor + residual ML, 0.1° cell mean"), kind="value", **q(v))
        if col != "ours":
            o = d.ours.values.astype(float)[sh]
            ok = np.isfinite(v) & np.isfinite(o)
            dd = o[ok] - v[ok]
            r.update(diff_mean=float(dd.mean()) if ok.any() else np.nan, diff_p25=float(np.percentile(dd, 25)) if ok.any() else np.nan,
                     diff_p50=float(np.median(dd)) if ok.any() else np.nan, diff_p75=float(np.percentile(dd, 75)) if ok.any() else np.nan,
                     diff_abs_mean=float(np.abs(dd).mean()) if ok.any() else np.nan, n_pair=int(ok.sum()),
                     pearson_r=float(pearsonr(o[ok], v[ok])[0]) if ok.sum() > 2 else np.nan,
                     spearman_rho=float(spearmanr(o[ok], v[ok])[0]) if ok.sum() > 2 else np.nan,
                     frac_shown_with_value=float(ok.mean()))
        rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="extract,summary")
    ap.add_argument("--regions", default="lena,alaska")
    a = ap.parse_args(argv)
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = a.stage.split(",")
    meta_path = OUT / "xl_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("extract", {})
    if "extract" in stages:
        XP = MA._load("x_product_comparison", "scripts/3_deep_learning/x_product_comparison.py")
        genv = XP.gdal_env()
        for key in a.regions.split(","):
            cells = region_cells(key)
            t0 = time.time()
            vals, rec = extract(key, XP, cells)
            path = OUT / f"xl_{key}_products_v1.csv.gz"
            vals.to_csv(path, index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")
            meta["extract"][key] = dict(n_cells=int(len(cells)), products=rec, output=MA.file_info(path), seconds=round(time.time() - t0, 1),
                                        gdal=genv, rule="최근접 화소, 기간 평균(XG extract_* 함수, check_md5=False)")
    if "summary" in stages:
        rows = []
        for key in a.regions.split(","):
            cells = region_cells(key)
            vals = pd.read_csv(OUT / f"xl_{key}_products_v1.csv.gz", dtype={"cell_id": str})
            assert (vals.cell_id.values == cells.cell_id.values).all()
            d = pd.concat([cells.reset_index(drop=True), vals.drop(columns=["cell_id"]).reset_index(drop=True)], axis=1)
            d["ours_e5mean"] = e5_mean(d.assign(ours=np.where(d.gray.values == 0, d.ours.values, np.nan)), "ours")
            rows += summary_rows(key, d)
        tab = pd.DataFrame(rows)
        tab.to_csv(OUT / "xl_summary_v1.csv", index=False, float_format="%.6g")
        meta["summary"] = dict(file=MA.file_info(OUT / "xl_summary_v1.csv"), shown_rule="회색 아님(gray = 0)·우리 예측 유한",
                               diff="우리(라벨 전량 재보정 Stefan + 잔차 ML) − 항목", corr="피어슨 r, 스피어먼 ρ(두 값 유한한 표시 셀)",
                               e5mean="0.1° 표시판: ERA5-Land 셀(floor((좌표 + 0.05)/0.1)) 안 표시 셀 평균")
        print(tab[["region", "item", "n", "mean", "p50", "diff_mean", "pearson_r", "n_pair"]].round(2).to_string(index=False), flush=True)
    meta.update(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/xl_map_products_v1.py", script_sha256=MA.sha256_file(Path(__file__)),
                registration="docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md(서술, 판정 없음)",
                inputs=dict(alaska_pred=MA.file_info(PROC / "map_alaska" / "alaska_pred_v1.csv.gz"), lena_pred=MA.file_info(PROC / "map_lena" / "lena_pred_v1.csv.gz")),
                elapsed_s_last=round(time.time() - t_start, 1), max_rss_mb_last=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
