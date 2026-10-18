"""v5 방법 비교 하네스(재설정 문서 6절 단계 C·D). 시험지는 src/polar/testbed_v5.py.

출력: <out>/<tag>/preds/<region>__<scheme>__<forcing>__<featset>__<method>__<learner>__lam<λ>.csv.gz
      (열: region, scheme, fold, forcing, featset, method, learner, lam, seed, cell_year_id, loc_id, year, block, y, pred)
      <out>/<tag>/meta/<같은 이름>.json (물리 계수, E0, 시간). 있는 파일은 건너뛴다(재개).
집계는 scripts/2_evaluation/v5_aggregate.py.
실행 예: python3 scripts/3_deep_learning/v5_compare.py --tag smoke --regions Lena --schemes block --methods P0,P1,Ku,D0,R1 --learners catboost_lo --seeds 0
"""
from __future__ import annotations
import os
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--proc", default="data/processed/v5")
ap.add_argument("--out", default="data/processed/v5/runs")
ap.add_argument("--tag", required=True)
ap.add_argument("--regions", default="Alaska,Lena,Canada")
ap.add_argument("--schemes", default="block,year,block_year,random,region")
ap.add_argument("--forcings", default="year,clim")
ap.add_argument("--featsets", default="F8TS")
ap.add_argument("--methods", default="P0,P1,P1a,Ku,D0,F1,D1,R1,S")
ap.add_argument("--learners", default="catboost")
ap.add_argument("--lams", default="0.5")
ap.add_argument("--seeds", default="0")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--fold-seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--gpu", default="", help="CUDA 장치 번호. 비우면 CPU")
ap.add_argument("--nice", type=int, default=10)
ap.add_argument("--exclude-years", default="", help="학습·평가에서 뺄 연도(쉼표). 진단용")
ap.add_argument("--exclude-source", default="", help="학습·평가에서 뺄 라벨 출처(쉼표, 예: ABoVE_ds2369)")
ap.add_argument("--block-deg", type=float, default=0.5, help="블록 크기(°). 0.5 가 기본")
ap.add_argument("--year-weight", action="store_true", help="학습 행을 연도 역빈도로 가중")
ap.add_argument("--no-products", action="store_true", help="제품 표 없이 로드(S 방법 제외)")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(a.threads))
os.nice(a.nice)

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

OUT = ROOT / a.out / a.tag; (OUT / "preds").mkdir(parents=True, exist_ok=True); (OUT / "meta").mkdir(exist_ok=True)
t0 = time.time()
df = T.load_v5(ROOT / a.proc, products=not a.no_products)
if a.exclude_years:
    df = df[~df.year.isin([int(y) for y in a.exclude_years.split(",")])].reset_index(drop=True)
if a.exclude_source:
    df = df[~df.source.isin(a.exclude_source.split(","))].reset_index(drop=True)
if abs(a.block_deg - 0.5) > 1e-9:
    df["block"] = (np.floor(df.lat / a.block_deg).astype(int) * 100000 + np.floor(df.lon / a.block_deg).astype(int))
print(f"[load] rows {len(df):,} cells {df.loc_id.nunique():,} blocks {df.block.nunique()} {time.time()-t0:.0f}s", flush=True)
regions = a.regions.split(","); schemes = a.schemes.split(","); forcings = a.forcings.split(","); featsets = a.featsets.split(",")
methods = a.methods.split(","); learners = a.learners.split(","); lams = [float(x) for x in a.lams.split(",")]; seeds = [int(s) for s in a.seeds.split(",")]
run_meta = dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), rows=int(len(df)))
json.dump(run_meta, open(OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), ensure_ascii=False, indent=1)

for forcing in forcings:
    d_all = T.with_forcing(df, forcing)
    use = T.usable_mask(d_all, None)
    d_all = d_all[use].reset_index(drop=True)
    # P0 원천 계수: 다른 두 지역 전량(같은 강제력 판)
    E0 = {r: T.ls_scale(d_all[d_all.region != r].e5_sqrt_tdd.values, d_all[d_all.region != r].alt_cm.values) for r in T.REGIONS}
    print(f"[{forcing}] usable {len(d_all):,} E0 {json.dumps({k: round(v, 3) for k, v in E0.items()})}", flush=True)
    for scheme in schemes:
        targets = ["ALL"] if scheme == "region" else regions
        for region in targets:
            dr = d_all if region == "ALL" else d_all[d_all.region == region].reset_index(drop=True)
            folds = T.make_folds(dr, scheme, a.k, a.fold_seed)
            for featset in featsets:
                feats = [f for f in T.FEATSETS[featset] if f in dr]
                if featset.endswith("_SAR") and region != "Alaska":
                    continue
                for method in methods:
                    L = ["-"] if method in T.PHYS_METHODS or method == "S" else learners
                    LAMS = lams if method == "R1" else [-1.0]
                    SEEDS = [0] if method in T.PHYS_METHODS or method == "S" else seeds
                    for learner in L:
                        for lam in LAMS:
                            name = f"{region}__{scheme}__{forcing}__{featset}__{method}__{learner}__lam{lam:g}"
                            fp = OUT / "preds" / f"{name}.csv.gz"
                            if fp.exists():
                                continue
                            t1 = time.time(); rows = []; coefs = {}
                            for seed in SEEDS:
                                for f in folds:
                                    tr, te = dr.iloc[f.tr], dr.iloc[f.te]
                                    e0 = E0[te.region.iloc[0]] if region != "ALL" else T.ls_scale(tr.e5_sqrt_tdd.values, tr.alt_cm.values)
                                    pred, k = T.run_method(method, tr, te, feats, learner=learner, lam=lam, seed=seed, threads=a.threads, E0=e0,
                                                           epochs=a.epochs, year_weight=a.year_weight)
                                    coefs[f"{f.name}|s{seed}"] = {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None) for kk, vv in k.items()}
                                    rows.append(pd.DataFrame({"region": te.region.values, "scheme": scheme, "fold": f.name, "forcing": forcing,
                                                              "featset": featset, "method": method, "learner": learner, "lam": lam, "seed": seed,
                                                              "cell_year_id": te.cell_year_id.values, "loc_id": te.loc_id.values, "year": te.year.values,
                                                              "block": te.block.values, "y": te.alt_cm.values, "pred": pred}))
                            out = pd.concat(rows, ignore_index=True)
                            out.to_csv(fp, index=False)
                            r = T.rmse(out.y.values, out.pred.values)
                            json.dump(dict(name=name, n=int(len(out)), rmse_pooled_seeds=r, E0=E0 if region != "ALL" else None, coefs=coefs,
                                           n_folds=len(folds), feats=feats, elapsed_s=round(time.time() - t1, 1)),
                                      open(OUT / "meta" / f"{name}.json", "w"), ensure_ascii=False, indent=1)
                            print(f"[{name}] n {len(out):,} rmse {r:.2f} {time.time()-t1:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
print(f"[done] {time.time()-t0:.0f}s")
