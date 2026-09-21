"""A2-2 ML 붕괴의 층화 진단(C2-C) — 정보 없음 조건에서 ML 직접·잔차 예측 대 Stefan 단독의 ΔRMSE를 셀 특성으로 층화.

모델: ridge·catboost_lo(알래스카 학습 재적합, seed 0·1·2 평균) + mlp(E1-e3 v2 예측 재사용, loc_id 대응, 같은 알래스카 학습·같은 E).
모드: 직접(anchor=none, λ=1) / Stefan 앵커 + λ=0.25 잔차. 기준 = Stefan 단독(알래스카 E).
층: (a) SoilGrids 결측 유무(평가 셀은 CCI 유효가 전제; --eval-all 이면 CCI 결측 셀도 포함해 cci_missing 층 추가)
    (b) AOA x25 안/밖(DI 임계 = 알래스카 OOF Q3+1.5IQR)
    (c) SWE 외삽(알래스카 학습 범위 밖) 유무, x25 중 하나라도 외삽 유무.
감사 주장 검증: "결측 없는 셀에서는 레나 ridge 단독이 Stefan보다 낫다" → complete 층의 Δ와 블록 CI.
산출: data/processed/m1/a2_ml_collapse_strata.csv (지역×모델×모드×층), a2_ml_collapse_summary.csv (표 1: complete/missing/AOA/SWE Δ)
실행: python3 scripts/2_evaluation/a2_ml_collapse_strata.py [--eval-all] [--lam 0.25]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import (Base, NoinfoPredictor, E1Preds, OUT, INPUT_SETS, SOIL, aoa_di, out_of_range, paired_boot,  # noqa: E402
                       rmse, bias, repro_check, flag_small)

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--eval-all", action="store_true", help="평가 셀 = y 유효만(CCI·토양 도일 결측 셀 포함)")
ap.add_argument("--nboot", type=int, default=400)
args = ap.parse_args()
t0 = time.time()
base = Base(sources=tuple(args.sources.split(",")), eval_all=args.eval_all)
print(base.describe(), flush=True)
pred = NoinfoPredictor(base)
e1 = E1Preds()
if not args.eval_all:
    repro_check(base, pred)

feats25 = INPUT_SETS["x25"]
di25, thr25, _ = aoa_di(base, feats25)
oor25 = out_of_range(base, feats25)

MODELS = ["ridge", "catboost_lo", "mlp"]
MODES = [("direct", "none", 1.0), (f"stefan+lam{args.lam:g}", "stefan", args.lam)]


def model_pred(tg, model, anchor, lam):
    """seed 평균 예측(평가 셀 순서). mlp는 E1-e3 재사용, 없으면 None."""
    if model in ("ridge", "catboost_lo"):
        return pred.predict(tg, model, anchor, lam)
    loc = base.te(tg).loc_id.values
    g = e1.g(tg, "mlp", anchor, loc)
    if g is None or np.isnan(g).all():
        return None
    return g if anchor == "none" else base.anchor(anchor, tg) + lam * g


rows = []
for tg in base.regions:
    idx = base.eval_idx(tg)
    te = base.df.iloc[idx]
    y, blocks = base.y(tg), base.blocks(tg)
    st = base.anchor("stefan", tg)
    d = di25[idx]
    o = oor25.iloc[idx]
    soil_miss = te[list(SOIL)].isna().any(axis=1).values
    cci_miss = te.cci_alt.isna().values
    strata = {"all": np.ones(len(idx), bool),
              "complete": ~soil_miss & ~cci_miss, "sg_missing": soil_miss,
              "aoa_in": d <= thr25, "aoa_out": d > thr25,
              "swe_in_range": ~o["e5_swe"].values, "swe_out_of_range": o["e5_swe"].values,
              "x25_all_in_range": o.sum(1).values == 0, "x25_any_out_of_range": o.sum(1).values > 0}
    if args.eval_all:
        strata["cci_missing"] = cci_miss
    for model in MODELS:
        for mode, anchor, lam in MODES:
            p = model_pred(tg, model, anchor, lam)
            if p is None:
                continue
            for sname, m in strata.items():
                m = m & np.isfinite(p) & np.isfinite(st)
                if m.sum() == 0:
                    continue
                d0, lo, hi, n, nb = paired_boot(y[m], p[m], st[m], blocks[m], nboot=args.nboot)
                rows.append(dict(target=tg, model=model, mode=mode, stratum=sname, n=n, n_blocks=nb, frac=round(m.mean(), 3),
                                 rmse_model=round(rmse(y[m], p[m]), 3), rmse_stefan=round(rmse(y[m], st[m]), 3),
                                 delta_rmse=round(d0, 3), ci_lo=round(lo, 3) if np.isfinite(lo) else np.nan,
                                 ci_hi=round(hi, 3) if np.isfinite(hi) else np.nan,
                                 bias_model=round(bias(y[m], p[m]), 3), bias_stefan=round(bias(y[m], st[m]), 3),
                                 flag=flag_small(n)))
    print(f"  [{tg}] n={len(idx)} sg_missing={soil_miss.mean():.2f} aoa_out={(d > thr25).mean():.2f} "
          f"swe_oor={o['e5_swe'].mean():.2f} · {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
tag = "_evalall" if args.eval_all else ""
res.to_csv(OUT / f"a2_ml_collapse_strata{tag}.csv", index=False)

# ---------------------------------------------------------------- 표 1: 층별 Δ(모델 − Stefan) 피벗
keep = ["all", "complete", "sg_missing", "aoa_in", "aoa_out", "swe_in_range", "swe_out_of_range"] + (["cci_missing"] if args.eval_all else [])
piv = res[res.stratum.isin(keep)].copy()
piv["cfg"] = piv["model"] + "|" + piv["mode"]
tab = piv.pivot_table(index=["target", "stratum"], columns="cfg", values="delta_rmse", aggfunc="first")
nn = piv.pivot_table(index=["target", "stratum"], columns="cfg", values="n", aggfunc="first").iloc[:, 0].rename("n")
tab = pd.concat([nn, tab], axis=1).reset_index()
tab["stratum"] = pd.Categorical(tab.stratum, keep, ordered=True)
tab = tab.sort_values(["target", "stratum"])
tab.to_csv(OUT / f"a2_ml_collapse_summary{tag}.csv", index=False)
(OUT / f"a2_ml_collapse_strata{tag}_meta.json").write_text(json.dumps(dict(
    sources=args.sources, lam=args.lam, eval_all=args.eval_all, nboot=args.nboot, di_threshold_x25=thr25,
    models=MODELS, modes=[m[0] for m in MODES], mlp_source="e1_factorial_e3 (v2) noinfo 예측 loc_id 대응",
    regions=base.regions, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))

print("\n=== A2-2 ΔRMSE(모델 − Stefan 단독, cm; 음수 = 모델 우세) 층별 ===")
print(tab.to_string(index=False))
print("\n=== 감사 주장: 결측 없는(complete) 셀에서 ridge 직접 대 Stefan ===")
q = res[(res.stratum.isin(["all", "complete", "sg_missing", "aoa_in", "aoa_out"])) & (res["model"] == "ridge") & (res["mode"] == "direct")]
print(q[["target", "stratum", "n", "n_blocks", "rmse_model", "rmse_stefan", "delta_rmse", "ci_lo", "ci_hi", "bias_model", "flag"]].to_string(index=False))
print(f"\n저장: {OUT/('a2_ml_collapse_strata'+tag+'.csv')} · {time.time()-t0:.0f}s")
