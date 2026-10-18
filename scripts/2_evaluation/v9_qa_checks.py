# -*- coding: utf-8 -*-
"""v9_qa_checks.py: 사용자 10-15 연구 질문 6건의 근거 수치(기존 예측·자료 집계, 학습 없음).
1 남서 알래스카(60°N 160°W 부근) 지도 범위: ESA CCI 영구동토 확률과 지도 격자 규칙(확률 ≥ 50 %)
2 위도대별 편향(관측 셀, 공간 블록 OOF): 물리 + 잔차 ML · Stefan(재보정) · Wei · Liu · ESA CCI v4
3 지도 수준 제품 − 우리 지도 차이(알래스카 2014, 위도대별, 표본 1/10 셀)
4 제품 학습 지점(Wei)까지 거리: 지역별 분포
5 새 지역 전이(대상 라벨 0·25개) 대 제품, 같은 셀-연도, 블록 부트스트랩 ΔRMSE
6 외부 지역 재고(CALM 2010–2024)와 v4 기타 자료원
출력 data/processed/v9_qa/*.csv, qa_summary.json. 실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/2_evaluation/v9_qa_checks.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402
P = ROOT / "data" / "processed"; OUT = P / "v9_qa"; OUT.mkdir(exist_ok=True); RUN = P / "v5" / "runs"
S = {}

# 1 ------------------------------------------------------------ 남서 알래스카
import xarray as xr
pfr = xr.open_dataset(P / "cci_pfr_mean_1997_2021.nc").pfr_mean
box = pfr.sel(lat=slice(59.0, 61.5), lon=slice(-163.0, -157.0)).values
grid = pd.read_csv(P / "map_alaska" / "alaska_grid_x25_v1.csv.gz", usecols=["lat", "lon", "pfr_mean", "gray"])
mp = pd.read_csv(P / "v5" / "maps" / "Alaska" / "mean_2011_2024.csv.gz", usecols=["lat", "lon"])
S["sw_alaska"] = dict(box="59.0–61.5°N, 163–157°W", pfr_median=float(np.nanmedian(box)), pfr_p90=float(np.nanpercentile(box, 90)),
                      share_pfr_ge50=float(np.nanmean(box >= 50)), grid_pfr_min=float(grid.pfr_mean.min()), map_lat_min=float(mp.lat.min()),
                      map_cells_in_box=int(((mp.lat.between(59, 61.5)) & (mp.lon.between(-163, -157))).sum()))

# 2 ------------------------------------------------------------ 위도대별 편향
lab = pd.read_csv(P / "v5" / "labels_cell_year_v5.csv", usecols=["loc_id", "year", "lat", "source", "obs_doy"]).drop_duplicates(["loc_id", "year"])
pr = pd.read_csv(P / "v5" / "products_cell_year_v5.csv")


def oof(region, meth, learner, lam, run="core_v6", forcing="year_obs"):
    d = pd.read_csv(RUN / run / "preds" / f"{region}__block__{forcing}__F8TS__{meth}__{learner}__lam{lam}.csv.gz")
    return d.groupby(["cell_year_id"]).agg(loc_id=("loc_id", "first"), year=("year", "first"), block=("block", "first"), y=("y", "first"), pred=("pred", "mean"))


rows = []; per = {}
for reg, bins in (("Alaska", [55, 64, 66, 68, 70, 72]), ("Lena", [69, 72, 74]), ("Canada", [60, 64, 66, 68, 70, 72])):
    d = oof(reg, "R1", "catboost", "0.5").rename(columns={"pred": "R1"}); d["P1"] = oof(reg, "P1", "-", "-1").pred.reindex(d.index).values
    for p in ("wei", "liu", "cci4y"):
        q = pr[pr["product"] == p].set_index(["loc_id", "year"]).value_cm
        d[p] = pd.MultiIndex.from_frame(d[["loc_id", "year"]]).map(q).values
    d = d.merge(lab, on=["loc_id", "year"], how="left"); per[reg] = d
    d["band"] = pd.cut(d.lat, bins)
    for b, g in d.groupby("band", observed=True):
        row = dict(region=reg, band=str(b), n=len(g), obs_mean=g.y.mean(), obs_doy_median=g.obs_doy.median())
        for m in ("R1", "P1", "wei", "liu", "cci4y"):
            gg = g[g[m].notna()]; row[f"{m}_bias"] = (gg[m] - gg.y).mean() if len(gg) else np.nan; row[f"{m}_n"] = len(gg)
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "lat_band_bias.csv", index=False)

# 3 ------------------------------------------------------------ 지도 수준 제품 − 우리(알래스카 2014)
import rasterio
from pyproj import Transformer
m14 = pd.read_csv(P / "v5" / "maps" / "Alaska" / "pred_2014.csv.gz", usecols=["lat", "lon", "r1"]).iloc[::10].reset_index(drop=True)
wei_tif = ROOT / "data" / "raw" / "wei2026_alt_v2" / "extracted" / "ALT_2014.tif"; liu_tif = ROOT / "data" / "raw" / "liu2024_alt" / "ALT_2014.tif"
with rasterio.open(wei_tif) as s:
    tr = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True); x, y = tr.transform(m14.lon.values, m14.lat.values)
    v = np.array([z[0] for z in s.sample(zip(x, y))], float); v[(v == s.nodata) | (v <= -9000)] = np.nan; m14["wei"] = v
with rasterio.open(liu_tif) as s:
    v = np.array([z[0] for z in s.sample(zip(m14.lon.values, m14.lat.values))], float); v[v <= 0] = np.nan; m14["liu"] = v * 0.01
m14["band"] = pd.cut(m14.lat, [59, 64, 66, 68, 70, 72])
mb = m14.groupby("band", observed=True).agg(n=("r1", "size"), ours=("r1", "mean"), wei=("wei", "mean"), liu=("liu", "mean")).reset_index()
mb["wei_minus_ours"] = mb.wei - mb.ours; mb["liu_minus_ours"] = mb.liu - mb.ours; mb["band"] = mb.band.astype(str); mb.to_csv(OUT / "map_band_alaska2014.csv", index=False)

# 4 ------------------------------------------------------------ Wei 학습 지점까지 거리
lk = pd.read_csv(P / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv"); lk = lk[(lk["product"] == "wei") & lk.target.str.endswith("|r")]
lk["region"] = lk.target.str.split("|").str[0]
dist = {}
for reg, g in lk.groupby("region"):
    if reg in T.REGIONS:
        dist[reg] = dict(cells=int(len(g)), median_km=float(g.dist_km.median()), min_km=float(g.dist_km.min()), share_le5=float((g.dist_km <= 5).mean()),
                         share_le25=float((g.dist_km <= 25).mean()), share_le100=float((g.dist_km <= 100).mean()))
S["dist_wei_train"] = dist

# 5 ------------------------------------------------------------ 전이(대상 라벨 0·25) 대 제품
tl_rows = []
for reg in T.REGIONS:
    p = pd.read_csv(RUN / "ladder_v6" / f"preds_TL_{reg}.csv.gz"); p["n_label"] = p.n_label.astype(int)
    for n in (0, 25):
        w = p[p.n_label == n].groupby(["cell_year_id", "method"]).pred.mean().unstack()
        first = p[p.n_label == n].drop_duplicates("cell_year_id").set_index("cell_year_id")[["loc_id", "year", "block", "y"]]
        w = w.join(first)
        for prod in ("wei", "liu"):
            q = pr[pr["product"] == prod].set_index(["loc_id", "year"]).value_cm
            w[prod] = pd.MultiIndex.from_frame(w[["loc_id", "year"]]).map(q).values
            ok = w[prod].notna()
            for m in ("P0", "P1n", "D0S", "R1S"):
                if m not in w:
                    continue
                r = T.boot_delta(w.y[ok].values, w[m][ok].values, w[prod][ok].values, w.block[ok].values, nboot=2000)
                tl_rows.append(dict(region=reg, n_label=n, method=m, product=prod, rmse_method=T.rmse(w.y[ok].values, w[m][ok].values),
                                    rmse_product=T.rmse(w.y[ok].values, w[prod][ok].values), **r))
pd.DataFrame(tl_rows).to_csv(OUT / "transfer_vs_products.csv", index=False)

# 6 ------------------------------------------------------------ 외부 지역 재고
em = json.load(open(P / "v9_ext" / "build_meta.json")); S["external_calm"] = em["labels"]; S["external_excluded"] = em["excluded_near_study_regions"]
v4 = pd.read_csv(P / "fidelity_base_v4_labels.csv", usecols=["loc_id", "region_v4", "sources", "year_min", "year_max", "n_values"], low_memory=False)
oth = v4[~v4.region_v4.isin(["ABoVE_AK", "Lena_RU", "ABoVE_CA", "United States (Alaska)", "Canada"])]
inv = oth.groupby("region_v4").agg(cells=("loc_id", "size"), year_min=("year_min", "min"), year_max=("year_max", "max"), values=("n_values", "sum")).reset_index()
inv.sort_values("cells", ascending=False).to_csv(OUT / "inventory_other_v4.csv", index=False)
json.dump(S, open(OUT / "qa_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(S, ensure_ascii=False, indent=1, default=float)[:3000])
print(pd.DataFrame(rows).round(1).to_string(index=False)); print(mb.round(1).to_string(index=False))
t = pd.DataFrame(tl_rows); t["txt"] = t.apply(lambda r: f"{r.rmse_method:.1f} vs {r.rmse_product:.1f}  Δ {r.delta:+.1f} [{r.lo:+.1f},{r.hi:+.1f}] n{r.n}", axis=1)
print(t[["region", "n_label", "method", "product", "txt"]].to_string(index=False))
