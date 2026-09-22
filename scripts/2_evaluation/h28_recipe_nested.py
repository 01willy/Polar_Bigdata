"""H28 라벨 있음 조건 레시피 확정 — M1 예측(npz) 재사용, 중첩 선택으로 선택 효과 제거. CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §3 H28, §4 B1·B2.
후보 = 앵커 축(14 앵커 × catboost_lo × λ 5) ∪ 모델 축(앵커 {none, stefan, stefan_cci} × 모델 10 × λ) — 라벨 있음 조건(대상 A블록 실측 학습, B 채점, 분할 3, seed 3).
중첩 선택: 대상 t 의 레시피는 다른 3 AB4 지역의 지역 등가중 RMSE 가 최소인 후보. 가족별(전체·CatBoost 계열·신경망 계열·해석적 앵커) 선택도 산출.
알래스카 지역 내: fold 중첩(다른 5 fold 에서 선택 → 해당 fold 채점).
검정: nested − Stefan, nested_cb − Stefan, nested_nn − Stefan, nested_cb − nested_nn, 사전 지정(stefan+catboost_lo λ.25/.5) − Stefan, oracle(자기 지역 최선, 참고) − Stefan. seed 짝지음·층화 블록 부트스트랩.
산출 data/processed/h3/h28_{scores,nested,tests}.csv
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.m1_stats import boot_delta, summarize_delta, seed_of                                     # noqa: E402

PROC = ROOT / "data" / "processed"; OUT = PROC / "h3"; OUT.mkdir(exist_ok=True)
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
NN = {"mlp", "ftt", "tabm", "realmlp", "cfm", "ddpm", "nflow"}
CB = {"catboost_lo", "catboost"}
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]

P, EV = {}, {}
for f in sorted(glob.glob(str(PROC / "m1" / "m1_*_shard*_preds.npz")) + glob.glob(str(PROC / "m1" / "h28model_shard*_preds.npz"))):
    if "smoke" in f or "gate" in f:
        continue
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        if (k.startswith("g::labels") or k.startswith("anchor::labels")):
            P[k] = z[k]
        elif k.startswith("eval::labels"):
            _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
print(f"[load] 예측 {len(P)} · 평가 {len(EV)}")

# 후보 열거(키에서 직접)
cands = set()
for k in P:
    if k.startswith("g::"):
        p = k.split("|"); cands.add((p[4], p[5], p[6], p[7], p[8], p[9], p[10]))   # dset,xset,anchor,pseudo,r,resid,iw
cands = sorted(c for c in cands if c[1] == "x25" and c[3] == "none" and c[6] == "0")
print(f"[cands] {len(cands)} 구성")


def tks_of(tg):
    return sorted([k for k in EV if k.startswith(f"labels|{tg}|")], key=lambda x: int(x.split("|")[2]))


def pred(tg, c, lam, tks=None):
    dset, xset, anchor, pseudo, r, resid, iw = c
    outs, ys, bs = [], [], []
    for tk in (tks or tks_of(tg)):
        seeds = sorted({int(k.split("|")[3]) for k in P if k.startswith(f"g::{tk}|") and k.endswith(f"|{dset}|{xset}|{anchor}|{pseudo}|{r}|{resid}|{iw}")})
        if not seeds:
            return None, None, None
        G = np.stack([P[f"g::{tk}|{sd}|{dset}|{xset}|{anchor}|{pseudo}|{r}|{resid}|{iw}"] for sd in seeds])
        if anchor == "none":
            if lam != 1.0:
                return None, None, None
            outs.append(G)
        else:
            ak = f"anchor::{tk}|{dset}|{anchor}"
            if ak not in P:
                return None, None, None
            outs.append(P[ak][None, :] + lam * G)
        ys.append(EV[tk]["y"]); bs.append(EV[tk]["block"])
    S = min(o.shape[0] for o in outs)
    return np.concatenate([o[:S] for o in outs], 1), np.concatenate(ys), np.concatenate(bs)


def stefan(tg, tks=None):
    tks = tks or tks_of(tg)
    return np.concatenate([P[f"anchor::{tk}|alaska|stefan"] for tk in tks])[None, :]


def rmse(y, p):
    return float(np.sqrt(np.nanmean((y - p) ** 2)))


# ---------------------------------------------------------------- 점수표(대상 × 후보 × λ)
rows = []
SPECS = []
for c in cands:
    for lam in (LAMS if c[2] != "none" else [1.0]):
        SPECS.append((c, lam))
MISSING = []
for tg in AB4:
    for c, lam in SPECS:
        PA, y, _ = pred(tg, c, lam)
        if PA is None:
            MISSING.append((tg, c[2], c[5], lam)); continue
        rows.append(dict(target=tg, anchor=c[2], resid=c[5], lam=lam, family=("analytic" if lam == 0.0 else ("cb" if c[5] in CB else ("nn" if c[5] in NN else "lin"))),
                         rmse=float(np.mean([rmse(y, PA[s]) for s in range(PA.shape[0])])), n_seed=int(PA.shape[0]), n=int(len(y))))
S = pd.DataFrame(rows); S.to_csv(OUT / "h28_scores.csv", index=False)
import collections
print(f"[scores] {len(S)}행 · 누락 {len(MISSING)}: {collections.Counter((m[1], m[2]) for m in MISSING).most_common(8)}")


def nested_pick(tg, fam=None):
    others = [r for r in AB4 if r != tg]
    q = S[S.target.isin(others)]
    if fam:
        q = q[q.family.isin(fam)]
    m = q.groupby(["anchor", "resid", "lam"]).agg(rmse=("rmse", "mean"), k=("rmse", "size")).reset_index()
    m = m[m.k == len(others)]
    best = m.sort_values("rmse").iloc[0]
    return (best.anchor, best.resid, float(best.lam)), float(best.rmse)


def spec_of(anchor, resid, lam):
    c = next(c for c in cands if c[2] == anchor and c[5] == resid)
    return c, lam


picks, tests = [], []
PRE = {"prespec25": ("stefan", "catboost_lo", 0.25), "prespec50": ("stefan", "catboost_lo", 0.5)}
for tg in AB4:
    ch = {"nested_all": nested_pick(tg), "nested_cb": nested_pick(tg, {"cb"}), "nested_nn": nested_pick(tg, {"nn"}), "nested_analytic": nested_pick(tg, {"analytic"})}
    own = S[S.target == tg].sort_values("rmse").iloc[0]
    ch["oracle"] = ((own.anchor, own.resid, float(own.lam)), float(own.rmse))
    for k, v in PRE.items():
        ch[k] = (v, float(S[(S.target == tg) & (S.anchor == v[0]) & (S.resid == v[1]) & (S.lam == v[2])].rmse.iloc[0]))
    for k, (rc, sc) in ch.items():
        picks.append(dict(target=tg, choice=k, anchor=rc[0], resid=rc[1], lam=rc[2], selection_score=sc,
                          rmse_target=float(S[(S.target == tg) & (S.anchor == rc[0]) & (S.resid == rc[1]) & (S.lam == rc[2])].rmse.iloc[0])))
PK = pd.DataFrame(picks); PK.to_csv(OUT / "h28_nested.csv", index=False)


def preds_choice(tg, choice):
    r = PK[(PK.target == tg) & (PK.choice == choice)].iloc[0]
    return pred(tg, *spec_of(r.anchor, r.resid, float(r.lam)))


CONTRASTS = [("H28_nested_all_vs_stefan", "nested_all", "stefan"), ("H28_nested_cb_vs_stefan", "nested_cb", "stefan"), ("H28x_nested_nn_vs_stefan", "nested_nn", "stefan"),
             ("H28x_nested_analytic_vs_stefan", "nested_analytic", "stefan"), ("H28b_nested_cb_vs_nested_nn", "nested_cb", "nested_nn"),
             ("H28ref_prespec25_vs_stefan", "prespec25", "stefan"), ("H28ref_prespec50_vs_stefan", "prespec50", "stefan"), ("H28ref_oracle_vs_stefan", "oracle", "stefan")]
for name, a, b in CONTRASTS:
    per = {}
    for tg in AB4:
        PA, y, blk = preds_choice(tg, a)
        PB = stefan(tg) if b == "stefan" else preds_choice(tg, b)[0]
        if PA is None or PB is None:
            continue
        Sn = min(PA.shape[0], PB.shape[0]) if PB.shape[0] > 1 else PA.shape[0]
        PBm = np.repeat(PB, PA.shape[0], 0) if PB.shape[0] == 1 else PB[:Sn]
        per[tg] = boot_delta(y, blk, PA[:PBm.shape[0]], PBm, 1000, seed_of(name, tg))
    tests += [dict(family="recipe", cond="labels", **r) for r in summarize_delta(name, per, AB4, 0)]

# ---------------------------------------------------------------- 알래스카 지역 내 fold 중첩
ak_tks = tks_of("Alaska")
if ak_tks:
    ak_rows = []
    for c, lam in SPECS:
        per_fold = []
        for tk in ak_tks:
            PA, y, _ = pred("Alaska", c, lam, tks=[tk])
            if PA is None:
                per_fold = None; break
            per_fold.append(float(np.mean([rmse(y, PA[s]) for s in range(PA.shape[0])])))
        if per_fold:
            ak_rows.append(dict(anchor=c[2], resid=c[5], lam=lam, family=("analytic" if lam == 0.0 else ("cb" if c[5] in CB else ("nn" if c[5] in NN else "lin"))), **{f"f{i}": v for i, v in enumerate(per_fold)}))
    AK = pd.DataFrame(ak_rows); AK.to_csv(OUT / "h28_alaska_scores.csv", index=False)
    fcols = [c for c in AK.columns if c.startswith("f") and c[1:].isdigit()]
    # fold 중첩: fold i 의 레시피는 다른 fold 평균 최소
    nested_preds, ys, bs, picks_ak = [], [], [], []
    for i, tk in enumerate(ak_tks):
        others = [c for c in fcols if c != f"f{i}"]
        best = AK.assign(sc=AK[others].mean(1)).sort_values("sc").iloc[0]
        picks_ak.append(dict(fold=i, anchor=best.anchor, resid=best.resid, lam=float(best.lam), sel_score=float(best.sc), fold_rmse=float(best[f"f{i}"])))
        PA, y, blk = pred("Alaska", *spec_of(best.anchor, best.resid, float(best.lam)), tks=[tk])
        nested_preds.append(PA); ys.append(y); bs.append(blk)
    Smin = min(p.shape[0] for p in nested_preds)
    PA = np.concatenate([p[:Smin] for p in nested_preds], 1); y = np.concatenate(ys); blk = np.concatenate(bs)
    PB = np.repeat(stefan("Alaska", ak_tks), Smin, 0)
    d = boot_delta(y, blk, PA, PB, 1000, seed_of("H28_alaska"))
    tests += [dict(family="recipe", cond="labels", **r) for r in summarize_delta("H28_alaska_fold_nested_vs_stefan", {"Alaska": d}, ["Alaska"], 0)]
    pd.DataFrame(picks_ak).to_csv(OUT / "h28_alaska_nested.csv", index=False)
    print(pd.DataFrame(picks_ak).to_string(index=False))

T = pd.DataFrame(tests); T.to_csv(OUT / "h28_tests.csv", index=False)
pd.set_option("display.width", 250)
print(PK.pivot_table(index="target", columns="choice", values="rmse_target").round(2).to_string())
print(PK[PK.choice.isin(["nested_all", "nested_cb", "nested_nn"])][["target", "choice", "anchor", "resid", "lam"]].to_string(index=False))
print(T[T.target.astype(str).str.startswith("MEAN") | (T.target == "Alaska")][["test", "target", "rmse_A", "rmse_B", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_flag"]].round(2).to_string(index=False))
