"""A1: GTN-P 전지구 파싱 — 546개 PT CSV 중 'Ground Temperature'만 → borehole×깊이 MAGT 프로파일.
기존 알래스카 파서(preprocess.compute_magt)와 동일한 월별 기후값 방식(혼합 샘플링 대응).
메모리 절약: 파일별로 (borehole,depth,year,month) 월평균으로 축약 후 전역 결합.
산출: data/processed/ground_temp_gtnp_global.csv  (+ 통합본 ground_temp_all.csv)
"""
import os, json, glob
import numpy as np
import pandas as pd

RAW = "data/raw/gtnp"
manifest = json.load(open(f"{RAW}/pt_manifest.json"))
gt_ids = {m["id"] for m in manifest if m.get("variable") == "Ground Temperature"}
print(f"manifest {len(manifest)} 중 Ground Temperature 데이터셋 {len(gt_ids)}개")

bh = pd.read_csv(f"{RAW}/boreholes.csv")
print(f"borehole 메타 {len(bh)}개, 국가 {bh.country_name.nunique()}개")

mon_parts = []
files = sorted(glob.glob(f"{RAW}/pt_csv/pt_dataset_*.csv"))
used = skipped = 0
for f in files:
    ds_id = int(os.path.basename(f).split("_")[2])
    if ds_id not in gt_ids:
        skipped += 1
        continue
    d = pd.read_csv(f, usecols=["date", "depth", "temperature", "borehole_id"],
                    parse_dates=["date"], low_memory=False)
    d = d.dropna(subset=["temperature", "date", "depth"])
    d = d[(d.temperature > -40) & (d.temperature < 40) & (d.depth >= 0) & (d.depth <= 2000)]
    if not len(d):
        continue
    d["year"] = d.date.dt.year
    d["month"] = d.date.dt.month
    m = (d.groupby(["borehole_id", "depth", "year", "month"], as_index=False)
           ["temperature"].mean())
    mon_parts.append(m)
    used += 1
print(f"파일 {used}개 파싱, {skipped}개 제외(비-지중온도)")

mon = pd.concat(mon_parts, ignore_index=True)
# 같은 (bh,depth,year,month)가 여러 dataset에 있으면 평균
mon = mon.groupby(["borehole_id", "depth", "year", "month"], as_index=False)["temperature"].mean()
print(f"월평균 레코드 {len(mon):,}")

# 월별 기후값 → MAGT (혼합 빈도 공정 처리)
clim = mon.groupby(["borehole_id", "depth", "month"], as_index=False)["temperature"].mean()
prof = (clim.groupby(["borehole_id", "depth"])
            .agg(magt=("temperature", "mean"),
                 t_min=("temperature", "min"), t_max=("temperature", "max"),
                 n_months=("month", "nunique")).reset_index())
nobs = mon.groupby(["borehole_id", "depth"]).size().rename("n_obs").reset_index()
yrs = mon.groupby(["borehole_id", "depth"])["year"].agg(["min", "max"]).reset_index()
yrs.columns = ["borehole_id", "depth", "year_min", "year_max"]
prof = prof.merge(nobs, on=["borehole_id", "depth"]).merge(yrs, on=["borehole_id", "depth"])
prof["amp"] = np.where(prof.n_months >= 8, (prof.t_max - prof.t_min) / 2, np.nan)
prof["sampling"] = np.where(prof.n_months >= 8, "seasonal", "sparse")

prof = prof.merge(bh[["borehole_id", "lat", "lon", "elev", "country_name", "site"]],
                  on="borehole_id", how="left")
prof = prof.dropna(subset=["lat", "lon"])
prof.to_csv("data/processed/ground_temp_gtnp_global.csv", index=False)
print(f"\n[GTN-P 전지구] borehole {prof.borehole_id.nunique()}개 / 프로파일 점 {len(prof):,} "
      f"/ 국가 {prof.country_name.nunique()}개")
print("국가별 borehole:", prof.groupby("country_name").borehole_id.nunique().sort_values(ascending=False).to_dict())
print(f"깊이 범위 {prof.depth.min():.1f}~{prof.depth.max():.1f}m, "
      f"0~20m 점 {len(prof[prof.depth<=20]):,}")

# ---- 통합: GTN-P 전지구 + GGD200 + PERMOS ----
gt = pd.read_csv("data/processed/ground_temp.csv")   # GGD200+PERMOS (source,site,lat,lon,depth_m,temp_c,...)
uni_a = prof.rename(columns={"depth": "depth_m", "magt": "temp_c", "country_name": "region"})[
    ["site", "lat", "lon", "depth_m", "temp_c", "region"]].assign(source="GTNP")
uni_b = gt.rename(columns={})[["site", "lat", "lon", "depth_m", "temp_c", "source"]]
uni_b["region"] = np.where(uni_b.source == "GGD200", "Russia(W.Siberia)", "Switzerland")
allgt = pd.concat([uni_a, uni_b], ignore_index=True)
allgt.to_csv("data/processed/ground_temp_all.csv", index=False)
print(f"\n[통합 3D 라벨] {len(allgt):,} 점 (GTNP {len(uni_a):,} + GGD200/PERMOS {len(uni_b):,}) "
      f"→ ground_temp_all.csv")
