"""H19a 해석적 사전 검정: 신규 공변량 군이 0.5° 블록 Stefan 계수 E 의 변동을 설명하는가 (CPU, 수 분).

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §2.2.
  블록 E = 블록 셀(≥3)의 최소제곱 E. 입력 = 블록 평균 공변량.
  기준 17 = 토양 9 + 기후 8(M1 EMAP_FEATS). 수준 = 17 + {LST, T, W, V, H, A(=T∪W∪V∪H), A+LST}.
  LOBO R²(알래스카 74블록, 10-fold 블록 CV, ridge α=10 표준화·√n 가중 + catboost_lo) ·
  LORO R²(6 라벨 지역 블록: 알래스카·레나·캐나다·러시아 W·C·E; 지역 hold-out, 블록 풀링 R²) ·
  계수 지도 앵커(E_hat(block)·√TDD)의 셀 RMSE 대 Stefan(E_AK) — 정보 없음 6지역(지역별 표, 부트스트랩은 h_analysis 에서).
산출 data/processed/h2/h19_blockE.csv, h19_blockE_cells.csv, h19_blockE_meta.json
실행: python3 scripts/3_deep_learning/h19_blockE.py --ext covariates_ext_v1.csv
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TRANSFER_MAIN                                                          # noqa: E402
from polar.m1_core import EMAP_FEATS, EXT_GROUPS, load_base, eval_mask, fit_coefs                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ext", default="covariates_ext_v1.csv")
ap.add_argument("--min-cells", type=int, default=2)
args = ap.parse_args()
PROC = ROOT / "data" / "processed"; OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
t0 = time.time()
DF = load_base(PROC, ext=args.ext)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
E_AK = float(fit_coefs(DF[DF.macro == "Alaska"])["E"])
LABEL = ["Alaska"] + [r for r in TRANSFER_MAIN if r != "Greenland"]
G = dict(EXT_GROUPS)
A = sum([G.get(k, []) for k in ("T", "W", "V", "H")], [])
SETS = {"base17": list(EMAP_FEATS)}
for k in ("LST", "T", "W", "V", "H"):
    if k in G:
        SETS[f"base17+{k}"] = list(EMAP_FEATS) + G[k]
SETS["base17+A"] = list(EMAP_FEATS) + A
SETS["base17+A+LST"] = list(EMAP_FEATS) + A + G.get("LST", [])
SETS["A_only"] = list(A)
ALLF = sorted(set(sum(SETS.values(), [])))

# ---------------------------------------------------------------- 블록 표
rows = []
for r in LABEL:
    d = DF[(DF.macro == r) & np.isfinite(DF.y.values) & (DF.s.values > 0)]
    for b, sub in d.groupby("block"):
        if len(sub) < args.min_cells:
            continue
        y, s = sub.y.values, sub.s.values
        rows.append(dict(region=r, block=b, n=len(sub), E=float((s @ y) / (s @ s)), **{f: float(np.nanmean(sub[f].values.astype(float))) for f in ALLF}))
BT = pd.DataFrame(rows)
print(f"[blocks] {len(BT)} 블록 · {BT.region.value_counts().to_dict()} · 특징 {len(ALLF)}", flush=True)


def fit_predict(kind, Xtr, ytr, wtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if kind == "ridge":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, ytr, sample_weight=np.sqrt(wtr))
        return m.predict((Xte - mu) / sd)
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=0, verbose=0, allow_writing_files=False, thread_count=4)
    m.fit(Xtr, ytr, sample_weight=np.sqrt(wtr))
    return m.predict(Xte)


def r2(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    return float(1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2))


res, cell_rows = [], []
ak = BT[BT.region == "Alaska"].reset_index(drop=True)
rng = np.random.RandomState(0); fold = rng.permutation(len(ak)) % 10
for name, feats in SETS.items():
    for kind in ("ridge", "catboost"):
        # LOBO 알래스카
        p = np.zeros(len(ak))
        for f in range(10):
            tr, te = fold != f, fold == f
            p[te] = fit_predict(kind, ak.loc[tr, feats].values.astype(float), ak.E.values[tr], ak.n.values[tr], ak.loc[te, feats].values.astype(float))
        r2_lobo = r2(ak.E.values, p); r2_lobo_w = r2(ak.E.values, p, ak.n.values)
        # LORO 6지역(블록 풀링 + 지역별)
        pl = np.zeros(len(BT)); reg_r2 = {}
        for r in LABEL:
            tr, te = (BT.region != r).values, (BT.region == r).values
            if te.sum() == 0:
                continue
            pl[te] = fit_predict(kind, BT.loc[tr, feats].values.astype(float), BT.E.values[tr], BT.n.values[tr], BT.loc[te, feats].values.astype(float))
            E_lo, E_hi = np.percentile(BT.E.values[tr], 2), np.percentile(BT.E.values[tr], 98)
            pl[te] = np.clip(pl[te], E_lo, E_hi)
            reg_r2[r] = dict(r2=r2(BT.E.values[te], pl[te]) if te.sum() >= 3 else np.nan, r2_w=r2(BT.E.values[te], pl[te], BT.n.values[te]) if te.sum() >= 3 else np.nan,
                             rmse_E=float(np.sqrt(np.mean((BT.E.values[te] - pl[te]) ** 2))), bias_E=float(np.mean(pl[te] - BT.E.values[te])),
                             rmse_E_AK=float(np.sqrt(np.mean((BT.E.values[te] - E_AK) ** 2))))
            # 셀 수준 앵커 RMSE(정보 없음): E_hat(block)·√TDD vs E_AK·√TDD. 블록 E 가 없는 셀(블록 < min_cells)은 학습 E 의 중앙값
            if r != "Alaska":
                d = DF[DF.macro == r]; m = eval_mask(d); d = d[m]
                emap = dict(zip(BT.block.values[te], pl[te]))
                Eh = np.array([emap.get(b, float(np.median(pl[te]))) for b in d.block.values])
                cell_rows.append(dict(set=name, model=kind, target=r, n=len(d), rmse_emap=float(np.sqrt(np.mean((Eh * d.s.values - d.y.values) ** 2))),
                                      rmse_stefan=float(np.sqrt(np.mean((E_AK * d.s.values - d.y.values) ** 2))), E_hat_mean=float(Eh.mean()), E_own=float((d.s.values @ d.y.values) / (d.s.values @ d.s.values))))
        nonak = (BT.region != "Alaska").values
        r2_loro = r2(BT.E.values, pl); r2_loro_nonak = r2(BT.E.values[nonak], pl[nonak])
        r2_loro_w = r2(BT.E.values, pl, BT.n.values); r2_loro_nonak_w = r2(BT.E.values[nonak], pl[nonak], BT.n.values[nonak])
        res.append(dict(set=name, model=kind, n_feat=len(feats), r2_lobo_AK=r2_lobo, r2_lobo_AK_w=r2_lobo_w, r2_loro_all=r2_loro, r2_loro_nonAK=r2_loro_nonak,
                        r2_loro_all_w=r2_loro_w, r2_loro_nonAK_w=r2_loro_nonak_w,
                        **{f"r2_{r}": v["r2"] for r, v in reg_r2.items()}, **{f"r2w_{r}": v["r2_w"] for r, v in reg_r2.items()}, **{f"rmseE_{r}": v["rmse_E"] for r, v in reg_r2.items()},
                        **{f"biasE_{r}": v["bias_E"] for r, v in reg_r2.items()}, rmseE_AK_ref=float(np.mean([v["rmse_E_AK"] for r, v in reg_r2.items() if r != "Alaska"]))))
R = pd.DataFrame(res); R.to_csv(OUT / "h19_blockE.csv", index=False)
C = pd.DataFrame(cell_rows); C.to_csv(OUT / "h19_blockE_cells.csv", index=False)
(OUT / "h19_blockE_meta.json").write_text(json.dumps(dict(stage="H19a", ext=args.ext, sets={k: len(v) for k, v in SETS.items()}, groups=G, n_blocks=int(len(BT)),
    blocks_by_region=BT.region.value_counts().to_dict(), E_AK=E_AK, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1, default=float))
print(R[["set", "model", "n_feat", "r2_lobo_AK", "r2_lobo_AK_w", "r2_loro_all_w", "r2_loro_nonAK_w", "r2w_Lena", "r2w_Canada", "rmseE_Lena", "rmseE_Canada", "rmseE_Russia_W", "rmseE_AK_ref"]].round(3).to_string())
if len(C):
    print(C.pivot_table(index=["set", "model"], columns="target", values="rmse_emap").round(2).to_string())
    print("stefan:", C.groupby("target").rmse_stefan.first().round(2).to_dict())
print(f"done {time.time()-t0:.0f}s")
