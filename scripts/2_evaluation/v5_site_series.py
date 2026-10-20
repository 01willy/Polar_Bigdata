"""v5 지점 시계열(사용자 질문 3). 블록 교차검증 구조로 모든 라벨 셀의 2010–2024 전 연도 예측을 만든다(각 블록 겹의 셀은 그 겹을 뺀 모델로 예측).
관측이 있는 해에는 OOF 평가와 같고, 관측이 없는 해에도 같은 모델로 예측해 연도별 추론의 안정성을 본다.
방법: P1(재보정 Stefan), R1(CatBoost 잔차, λ), D0. 강제력: --forcing year|year_obs(관측 없는 해는 연간 TDD 로 끝-계절값, 관측 있는 해는 관측일 TDD 로도 낸다).
출력: <out>/site_series_<forcing>.csv.gz (loc_id, region, block, year, observed(0/1), y, tdd, p1, r1, d0, p1_eos, r1_eos), 다년 지점 목록 site_list.csv
실행: python3 scripts/2_evaluation/v5_site_series.py --forcing year_obs --threads 8
"""
from __future__ import annotations
import os, argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--forcing", default="year_obs"); ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", default="data/processed/v5/site_series")
ap.add_argument("--ref-doy", type=int, default=0, help="end-of-season reference day for the *_eos series (0 = full-season TDD; 244 = 1 September, as the yearly maps; 2026-10-20)")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.nice(10)
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T
from catboost import CatBoostRegressor
PROC = ROOT / "data" / "processed" / "v5"; OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
df = T.load_v5(PROC, products=False)
fy = pd.read_csv(PROC / "forcing_year.csv"); st = pd.read_csv(PROC / "static_v5.csv", low_memory=False)
feats = T.FEATSETS["F8TS"]; fcols = [c for c in feats if c.startswith("e5_")]; scols = [c for c in feats if not c.startswith("e5_")]
rows = []
for region in T.REGIONS:
    d = T.with_forcing(df[df.region == region], a.forcing); d = d[T.usable_mask(d, feats)].reset_index(drop=True)
    folds = T.folds_block(d, a.k, 0)
    cells = d.drop_duplicates("loc_id")[["loc_id", "block"]]
    # 전 연도 격자: 셀 × 2010–2024(끝-계절 강제력 = 연간 TDD)
    grid = fy[fy.loc_id.isin(cells.loc_id)].merge(st[["loc_id"] + scols], on="loc_id", how="left").merge(cells, on="loc_id")
    grid = grid[np.isfinite(grid.e5_sqrt_tdd) & (grid.e5_sqrt_tdd > 0)].reset_index(drop=True)
    if a.ref_doy > 0:                                   # TDD accumulated to the reference day (linear within the month, non-leap calendar as v5_map_predict)
        starts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); k = int(np.searchsorted(starts, a.ref_doy, side="right"))   # month containing the day (1-12)
        c_prev = grid[f"tdd_cum_{k - 1}"].values if k > 1 else 0.0; c_k = grid[f"tdd_cum_{k}"].values
        frac = (a.ref_doy - starts[k - 1]) / (starts[k] - starts[k - 1])
        grid["e5_tdd"] = c_prev + frac * (c_k - c_prev); grid["e5_sqrt_tdd"] = np.sqrt(np.clip(grid.e5_tdd, 0, None))
        grid = grid[grid.e5_sqrt_tdd > 0].reset_index(drop=True)
    obs = d.set_index(["loc_id", "year"])
    for f in folds:
        tr = d.iloc[f.tr]; te_blocks = set(d.block.values[f.te])
        k = T.fit_phys(tr); E = k["E"]
        Xtr = tr[feats].values.astype(float); ytr = tr.alt_cm.values.astype(float); atr = E * tr.e5_sqrt_tdd.values
        g = grid[grid.block.isin(te_blocks)].reset_index(drop=True)
        Xg = g[feats].values.astype(float)
        p1_eos = E * g.e5_sqrt_tdd.values
        r_res = np.zeros(len(g)); d0 = np.zeros(len(g))
        for s in [int(x) for x in a.seeds.split(",")]:
            m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False, thread_count=a.threads)
            m.fit(Xtr, ytr - atr); r_res += m.predict(Xg)
            m2 = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False, thread_count=a.threads)
            m2.fit(Xtr, ytr); d0 += m2.predict(Xg)
        ns = len(a.seeds.split(",")); r_res /= ns; d0 /= ns
        r1_eos = p1_eos + a.lam * r_res
        # 관측 있는 해: 관측일 TDD 판(강제력이 year_obs 일 때)으로도 예측
        key = list(zip(g.loc_id, g.year)); has = [kk in obs.index for kk in key]
        tdd_obs = np.array([obs.loc[kk, "e5_tdd"] if h else np.nan for kk, h in zip(key, has)])
        y = np.array([obs.loc[kk, "alt_cm"] if h else np.nan for kk, h in zip(key, has)])
        p1_obs = E * np.sqrt(np.clip(tdd_obs, 0, None)); 
        # 잔차 모델은 정적+강제력 특징으로 예측하므로 관측일 강제력에서 다시 예측
        if a.forcing == "year_obs":
            go = g.copy(); go["e5_tdd"] = np.where(np.isfinite(tdd_obs), tdd_obs, go.e5_tdd); go["e5_sqrt_tdd"] = np.sqrt(np.clip(go.e5_tdd, 0, None))
            Xo = go[feats].values.astype(float); ro = np.zeros(len(g)); do = np.zeros(len(g))
            for s in [int(x) for x in a.seeds.split(",")]:
                m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False, thread_count=a.threads)
                m.fit(Xtr, ytr - atr); ro += m.predict(Xo)
                m2 = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False, thread_count=a.threads)
                m2.fit(Xtr, ytr); do += m2.predict(Xo)
            r1_obs = p1_obs + a.lam * ro / ns; d0_obs = do / ns
        else:
            r1_obs = r1_eos; d0_obs = d0; p1_obs = p1_eos
        rows.append(pd.DataFrame(dict(loc_id=g.loc_id.values, region=region, block=g.block.values, year=g.year.values, observed=np.array(has, int), y=y,
                                      tdd_eos=g.e5_tdd.values, tdd_obs=tdd_obs, p1_eos=p1_eos, r1_eos=r1_eos, d0_eos=d0, p1_obs=p1_obs, r1_obs=r1_obs, d0_obs=d0_obs)))
        print(f"[{region} {f.name}] cells {g.loc_id.nunique():,} rows {len(g):,} {time.time()-t0:.0f}s", flush=True)
out = pd.concat(rows, ignore_index=True)
out.to_csv(OUT / (f"site_series_{a.forcing}.csv.gz" if a.ref_doy == 0 else f"site_series_{a.forcing}_d{a.ref_doy}.csv.gz"), index=False)
multi = out[out.observed == 1].groupby(["region", "loc_id"]).year.nunique().reset_index().rename(columns={"year": "n_years"})
multi = multi[multi.n_years >= 4].sort_values(["region", "n_years"], ascending=[True, False])
multi.to_csv(OUT / "site_list.csv", index=False)
print(f"[done] rows {len(out):,}, multi-year sites (>=4 yrs) {len(multi)} {time.time()-t0:.0f}s")
