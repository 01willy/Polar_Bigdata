"""스레드 R ㉡㉢ — ALT 셀단위 집계 (pseudo-replication 제거 + 척도정합).

문제: dl_dataset.csv는 같은 위치(공변량 동일)에 여러 점·여러 연도를 별도 행으로 쌓아
  "같은 X, 다른 y"(셀내 34~96cm)를 만듦 → 모델 평균회귀 강제 + 조밀셀 가중편향.
해결: loc_id(고유 위치=공변량 동일 단위)로 집계:
  - 정답 = 셀 평균 ALT (입력 해상도에 맞춘 예측대상)
  - alt_sd = 셀내 표준편차 → '9km로 못 보는 미세지형 불확실성' 라벨(모델 실패 아님)
  - 각 위치 1행 = 모든 위치가 동등 가중(㉢)

입력: dl_dataset.csv (+ polsar/insar 부착본, 나중 다중모달용)
출력: data/processed/dl_dataset_cell.csv  (위치당 1행)
실행: python3 scripts/1_data_prep/aggregate_alt_cell.py
"""
import os
import numpy as np
import pandas as pd

PROC = "data/processed"
base = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
pol = pd.read_csv(os.path.join(PROC, "dl_dataset_polsar.csv"), usecols=["polsar_alt", "polsar_std", "polsar_valid"])
ins = pd.read_csv(os.path.join(PROC, "dl_dataset_insar.csv"),
                  usecols=["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"])
df = pd.concat([base, pol, ins], axis=1)

FEATS = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
         "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
PHYS = ["polsar_alt", "polsar_std", "polsar_valid", "insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]

agg = {
    "lat": "mean", "lon": "mean", "region": "first",
    "alt_cm": ["mean", "std", "min", "max", "count"], "year": ["min", "max", "nunique"],
}
for f in FEATS:
    agg[f] = "first"     # loc 내 상수(검증됨)
for f in PHYS:
    agg[f] = "mean"      # 물리관측은 loc 내 평균

g = df.groupby("loc_id").agg(agg)
g.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in g.columns]
g = g.rename(columns={
    "lat_mean": "lat", "lon_mean": "lon", "region_first": "region",
    "alt_cm_mean": "alt_cm", "alt_cm_std": "alt_sd", "alt_cm_min": "alt_min",
    "alt_cm_max": "alt_max", "alt_cm_count": "n_obs",
    "year_min": "year_min", "year_max": "year_max", "year_nunique": "n_years",
})
for f in FEATS:
    g = g.rename(columns={f + "_first": f})
for f in PHYS:
    g = g.rename(columns={f + "_mean": f})
g["alt_sd"] = g["alt_sd"].fillna(0.0)   # 단일관측 loc은 SD=0
g = g.reset_index()

out = os.path.join(PROC, "dl_dataset_cell.csv")
g.to_csv(out, index=False)
print(f"집계 완료: {len(df):,}행(pooled) → {len(g):,}행(위치당 1행)")
print(f"저장: {out}")
print("\n셀내 SD(불확실성 라벨) 요약:")
print(g["alt_sd"].describe().round(2).to_string())
print(f"\n다관측 위치(n_obs≥2): {(g.n_obs>=2).sum():,}개, 평균 관측수 {g.n_obs.mean():.1f}")
print(f"셀 평균 ALT SD(위치간): {g.alt_cm.std():.1f}cm  (pooled 점 ALT SD: {df.alt_cm.std():.1f}cm)")
