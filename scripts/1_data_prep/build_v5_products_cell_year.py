"""v5 기존 ALT 제품의 셀·연도 값 추출(재설정 문서 5.1 단계 A4). 라벨 값은 읽지 않는다(좌표만).

제품(연별): CCI v4(1997–2021, 0.01°), CCI v5(1997–2023, 1 km), Wei 2026 v2(2000–2024, 1 km), Yi·Kimball ds1760(2001–2015, 알래스카 1 km).
제품(대표값): Aalto 2018(2000–2014 기준기, 30 초각). Li 2022·Liu 2024 는 보유하지 않아 이 판에 없다(등록부에 적는다).
추출 함수는 scripts/3_deep_learning/x_product_comparison.py(XG) 의 것을 그대로 쓴다(최근접 화소 + 3×3 평균, 단위 확인, zip md5 대조).

출력: data/processed/v5/products_cell_year_v5.csv  (loc_id, product, year, value_cm, value3_cm, pix_dist_km; 대표값 제품은 year = -1)
      data/processed/v5/products_cell_year_v5_meta.json
실행: python3 scripts/1_data_prep/build_v5_products_cell_year.py [--products cci4y,cci5y,wei,yk,aalto]
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))
sys.path.insert(0, str(ROOT / "src"))
import x_product_comparison as XP  # noqa: E402

SPECS = {
    "cci4y": dict(kind="cci_nc", folder="cci_alt", tag="fv04.0", period=(1997, 2021), label="CCI v4 (yearly)"),
    "cci5y": dict(kind="cci_nc", folder="cci_alt_v5", tag="fv05.0", period=(1997, 2023), label="CCI v5 (yearly)"),
    "wei": dict(XP.PRODUCTS["wei"]),
    "yk": dict(XP.PRODUCTS["yk"]),
    "aalto": dict(XP.PRODUCTS["aalto"]),
    "liu": dict(kind="tif_dir", folder="liu2024_alt", period=(2010, 2020), member="ALT_{year}.tif", label="Liu 2024 (yearly)", scale=0.01,
                nodata_le=0, note="int16, 값은 cm × 100 (중앙값 5,266, Wei 와의 비 약 100). 0 이하는 결측"),
}


def extract_tif_dir(spec, cells):
    """폴더 안 연별 GeoTIFF(Liu 2024). 파일마다 최근접 화소와 3×3 평균."""
    folder = ROOT / "data" / "raw" / spec["folder"]
    years = [y for y in range(spec["period"][0], spec["period"][1] + 1) if (folder / spec["member"].format(year=y)).exists()]
    V = np.full((len(cells), len(years)), np.nan); V3 = V.copy(); rec, dist = {}, None
    for j, y in enumerate(years):
        v, v3, dist, rec = XP._tif_points(str(folder / spec["member"].format(year=y)), cells)
        V[:, j], V3[:, j] = v, v3
    if "nodata_le" in spec:
        V[V <= spec["nodata_le"]] = np.nan; V3[V3 <= spec["nodata_le"]] = np.nan
    mult = spec.get("scale", None)
    if mult is None:
        mult, urec = XP.infer_units(V)
    else:
        urec = dict(units="cm", how=f"명시 배율 {mult}", note=spec.get("note", ""))
    rec = dict(rec, units=urec, files=[spec["member"].format(year=y) for y in years])
    return V * mult, V3 * mult, years, dist, rec


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="data/processed/v5/static_v5.csv")
    ap.add_argument("--products", default="cci4y,cci5y,wei,yk,aalto")
    ap.add_argument("--out", default="data/processed/v5/products_cell_year_v5.csv")
    ap.add_argument("--append", action="store_true", help="기존 출력에 지정 제품만 바꿔 넣는다")
    a = ap.parse_args()
    t0 = time.time()
    os_env = XP.gdal_env()
    import os
    os.environ.update({k: str(v) for k, v in os_env.items()})
    cells = pd.read_csv(ROOT / a.cells, usecols=["loc_id", "lat", "lon", "region"], low_memory=False).drop_duplicates("loc_id").reset_index(drop=True)
    print(f"[cells] {len(cells):,}", flush=True)
    parts, meta = [], dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), cells=a.cells, products={})
    for p in a.products.split(","):
        spec = SPECS[p]
        sub = cells if "regions" not in spec else cells[cells.region.isin(spec["regions"])].reset_index(drop=True)
        t1 = time.time()
        if spec["kind"] == "cci_nc":
            V, V3, years, dist, rec = XP.extract_cci(spec, sub)
        elif spec["kind"] == "zip_tif":
            V, V3, years, dist, rec = XP.extract_tif(spec, sub)
        elif spec["kind"] == "nc_latlon":
            V, V3, years, dist, rec = XP.extract_nc_latlon(spec, sub)
        elif spec["kind"] == "tif_dir":
            V, V3, years, dist, rec = extract_tif_dir(spec, sub)
        else:
            raise SystemExit(p)
        for j, y in enumerate(years):
            parts.append(pd.DataFrame({"loc_id": sub.loc_id.values, "product": p, "year": -1 if y is None else int(y),
                                       "value_cm": V[:, j], "value3_cm": V3[:, j], "pix_dist_km": dist}))
        fin = np.isfinite(V)
        meta["products"][p] = dict(label=spec.get("label"), period=list(spec["period"]), n_cells=int(len(sub)), years=[int(y) for y in years if y is not None],
                                   frac_finite=float(fin.mean()), median_cm=float(np.nanmedian(V)) if fin.any() else None,
                                   record=json.loads(json.dumps(rec, default=str)), elapsed_s=round(time.time() - t1, 1))
        print(f"[{p}] years {len(years)} finite {fin.mean():.3f} median {np.nanmedian(V) if fin.any() else float('nan'):.1f} cm {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    if a.append and (ROOT / a.out).exists():
        old = pd.read_csv(ROOT / a.out)
        old = old[~old["product"].isin(out["product"].unique())]
        out = pd.concat([old, out], ignore_index=True)
        mpath = ROOT / a.out.replace(".csv", "_meta.json")
        if mpath.exists():
            oldm = json.load(open(mpath)); oldm["products"].update(meta["products"]); meta["products"] = oldm["products"]
    out.to_csv(ROOT / a.out, index=False)
    meta["output"] = dict(path=a.out, rows=int(len(out)), sha256=sha(ROOT / a.out))
    meta["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(ROOT / a.out.replace(".csv", "_meta.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: dict(frac=v["frac_finite"], med=v["median_cm"], s=v["elapsed_s"]) for k, v in meta["products"].items()}))


if __name__ == "__main__":
    main()
