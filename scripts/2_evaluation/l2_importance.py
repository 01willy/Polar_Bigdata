"""L2-1 변수 중요도 — 순열 중요도(ΔRMSE, cm 단위).

(a) 알래스카 지역 내 6-fold: catboost_lo 직접(x25)과 Stefan 앵커 + catboost_lo 잔차(λ=1). fold 외 평가 셀에서 공변량 한 열을
    섞어(10회 반복, fold별 독립 순열) 풀링 RMSE 증가분 ΔRMSE = RMSE(순열) − RMSE(기준). 앵커 모델은 e5_sqrt_tdd 순열이 앵커에도
    전파되는 합성 예측기로 평가한다. 공변량 그룹(지형/기후/토양/CCI)은 변수 ΔRMSE 합산과 그룹 동시 순열 두 가지.
(b) 정보 없음 전이: 알래스카 전 셀 학습 catboost_lo 직접 모델(3 seed 앙상블)을 레나·캐나다·러시아 W/E 평가 셀에 적용해 지역별
    순열 중요도, 상위 10, 알래스카(a) 대비 Spearman 순위 상관(25 변수).
산출 data/processed/m1/l2_importance.csv · l2_importance_folds.csv · l2_importance_groups.csv · l2_importance_transfer.csv ·
     l2_importance_rankcorr.csv · l2_importance_meta.json  그림 outputs/figures/m1/l2_importance.{png,pdf}
실행 python3 scripts/2_evaluation/l2_importance.py [--nrep 10]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET                                        # noqa: E402
from polar.m1_core import eval_mask                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nrep", type=int, default=10)
ap.add_argument("--regions", default="Lena,Canada,Russia_W,Russia_E")
args = ap.parse_args()
t0 = time.time()
METHODS_A = ["catboost_lo", "stefan_cb1"]
REGIONS = args.regions.split(",")

df = L.load()
folds = L.alaska_folds(df)
print(f"[data] 셀 {len(df):,} · 알래스카 fold {len(folds)} · 반복 {args.nrep}", flush=True)

# ---------------------------------------------------------------- (a) 알래스카 6-fold 적합
FOLD = []                                                                # (X, y, blocks, comps)
for fi, (tr, te) in enumerate(folds):
    tev = te[eval_mask(df.iloc[te])]
    trn, tst = df.iloc[tr], df.iloc[tev]
    comps = L.fit_methods(trn, tst, METHODS_A)
    FOLD.append(dict(X=tst[L.FEATS].values.astype(np.float32), y=tst[TARGET].values.astype(float),
                     blocks=tst.block.values, comps=comps, fold=fi))
    print(f"  fold {fi} 적합 완료 ({time.time()-t0:.0f}s)", flush=True)


def pooled_rmse(method, perm_cols=None, rng=None):
    """perm_cols: 순열할 열 인덱스 목록(동시 순열: 같은 행 순열 적용). fold별 독립 순열."""
    ys, ps, per_fold = [], [], []
    for f in FOLD:
        X = f["X"]
        if perm_cols is not None:
            X = X.copy(); pi = rng.permutation(len(X)); X[:, perm_cols] = f["X"][pi][:, perm_cols]
        p = f["comps"][method](X)
        ys.append(f["y"]); ps.append(p); per_fold.append(L.rmse(f["y"], p))
    return L.rmse(np.concatenate(ys), np.concatenate(ps)), per_fold


base = {m: pooled_rmse(m) for m in METHODS_A}
print("[base] " + " · ".join(f"{m} {base[m][0]:.3f}" for m in METHODS_A), flush=True)

rows, frows, grows = [], [], []
for m in METHODS_A:
    D = np.zeros((args.nrep, len(L.FEATS))); DF = np.zeros((args.nrep, len(L.FEATS), len(FOLD)))
    for r in range(args.nrep):
        for j in range(len(L.FEATS)):
            rng = np.random.RandomState(10_000 * r + j)
            v, pf = pooled_rmse(m, [j], rng)
            D[r, j] = v - base[m][0]; DF[r, j] = np.array(pf) - np.array(base[m][1])
    for j, f in enumerate(L.FEATS):
        rows.append(dict(scope="alaska_oof", method=m, feature=f, group=L.GROUP_OF[f], delta_rmse_cm=D[:, j].mean(),
                         delta_rmse_sd=D[:, j].std(ddof=1), rmse_base=base[m][0], n_rep=args.nrep))
        for fi in range(len(FOLD)):
            frows.append(dict(method=m, feature=f, group=L.GROUP_OF[f], fold=fi, delta_rmse_cm=DF[:, j, fi].mean(),
                              rmse_base_fold=base[m][1][fi], n_eval=len(FOLD[fi]["y"])))
    # 그룹: 합산 + 동시 순열
    for g in L.GROUPS:
        cols = [i for i, f in enumerate(L.FEATS) if L.GROUP_OF[f] == g]
        joint = np.array([pooled_rmse(m, cols, np.random.RandomState(777 + r))[0] - base[m][0] for r in range(args.nrep)])
        grows.append(dict(scope="alaska_oof", method=m, group=g, n_features=len(cols), delta_sum_cm=D[:, cols].mean(0).sum(),
                          delta_joint_cm=joint.mean(), delta_joint_sd=joint.std(ddof=1), rmse_base=base[m][0]))
    print(f"[perm] {m} 완료 ({time.time()-t0:.0f}s)", flush=True)
imp = pd.DataFrame(rows)
imp["rank"] = imp.groupby("method").delta_rmse_cm.rank(ascending=False).astype(int)

# ---------------------------------------------------------------- (b) 정보 없음 전이
trn = df.iloc[L.region_idx(df, "Alaska")]
comps_t = L.fit_methods(trn, None, ["catboost_lo"])
trows, base_t = [], {}
for reg in REGIONS:
    idx = L.region_idx(df, reg); tst = df.iloc[idx[eval_mask(df.iloc[idx])]]
    X, y = tst[L.FEATS].values.astype(np.float32), tst[TARGET].values.astype(float)
    b0 = L.rmse(y, comps_t["catboost_lo"](X)); base_t[reg] = b0
    D = np.zeros((args.nrep, len(L.FEATS)))
    for r in range(args.nrep):
        for j in range(len(L.FEATS)):
            Xp = X.copy(); Xp[:, j] = np.random.RandomState(10_000 * r + j).permutation(Xp[:, j])
            D[r, j] = L.rmse(y, comps_t["catboost_lo"](Xp)) - b0
    for j, f in enumerate(L.FEATS):
        trows.append(dict(scope="noinfo", region=reg, method="catboost_lo", feature=f, group=L.GROUP_OF[f], delta_rmse_cm=D[:, j].mean(),
                          delta_rmse_sd=D[:, j].std(ddof=1), rmse_base=b0, n_eval=len(y), n_blocks=tst.block.nunique(), n_rep=args.nrep))
    print(f"[transfer] {reg}: n {len(y)} · 기준 RMSE {b0:.2f} ({time.time()-t0:.0f}s)", flush=True)
timp = pd.DataFrame(trows)
timp["rank"] = timp.groupby("region").delta_rmse_cm.rank(ascending=False).astype(int)

# 순위 상관(알래스카 OOF catboost_lo 대비, 25 변수 Spearman) + 지역 간
ref = imp[imp.method == "catboost_lo"].set_index("feature").delta_rmse_cm.reindex(L.FEATS).values
crows = []
vec = {reg: timp[timp.region == reg].set_index("feature").delta_rmse_cm.reindex(L.FEATS).values for reg in REGIONS}
for reg in REGIONS:
    rho, p = spearmanr(ref, vec[reg])
    top5_ref = set(imp[(imp.method == "catboost_lo") & (imp["rank"] <= 5)].feature)
    top5 = set(timp[(timp.region == reg) & (timp["rank"] <= 5)].feature)
    crows.append(dict(a="Alaska_oof", b=reg, spearman_rho=rho, p=p, top5_overlap=len(top5_ref & top5), n_features=len(L.FEATS)))
for i, a in enumerate(REGIONS):
    for b in REGIONS[i + 1:]:
        rho, p = spearmanr(vec[a], vec[b])
        crows.append(dict(a=a, b=b, spearman_rho=rho, p=p, top5_overlap=len(set(timp[(timp.region == a) & (timp["rank"] <= 5)].feature)
                                                                              & set(timp[(timp.region == b) & (timp["rank"] <= 5)].feature)),
                          n_features=len(L.FEATS)))
rank = pd.DataFrame(crows)

imp.to_csv(L.OUT / "l2_importance.csv", index=False)
pd.DataFrame(frows).to_csv(L.OUT / "l2_importance_folds.csv", index=False)
pd.DataFrame(grows).to_csv(L.OUT / "l2_importance_groups.csv", index=False)
timp.to_csv(L.OUT / "l2_importance_transfer.csv", index=False)
rank.to_csv(L.OUT / "l2_importance_rankcorr.csv", index=False)

# ---------------------------------------------------------------- 그림
from polar.plotstyle import use_polar, despine                            # noqa: E402
plt = use_polar()
fig = plt.figure(figsize=(13.0, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.75, 1, 1], wspace=0.55, hspace=0.55, left=0.13, right=0.98, top=0.90, bottom=0.10)
ax = fig.add_subplot(gs[:, 0])
top = imp[imp.method == "catboost_lo"].nlargest(15, "delta_rmse_cm").sort_values("delta_rmse_cm")
alt = imp[imp.method == "stefan_cb1"].set_index("feature").reindex(top.feature)
yy = np.arange(len(top)); h = 0.38
ax.barh(yy + h / 2, top.delta_rmse_cm, h, xerr=top.delta_rmse_sd, color=[L.GROUP_COLOR[g] for g in top.group], edgecolor="#333333",
        linewidth=0.5, error_kw=dict(lw=0.7, capsize=2), label="CatBoost 직접")
ax.barh(yy - h / 2, alt.delta_rmse_cm, h, xerr=alt.delta_rmse_sd, color="white", edgecolor=[L.GROUP_COLOR[g] for g in top.group],
        hatch="////", linewidth=1.0, error_kw=dict(lw=0.7, capsize=2), label="Stefan+CatBoost 잔차 (λ=1)")
ax.set_yticks(yy); ax.set_yticklabels([f"{f} ({g})" for f, g in zip(top.feature, top.group)], fontsize=9)
ax.set_xlabel("순열 ΔRMSE (cm)"); ax.set_title("(a) 알래스카 지역 내 6-fold, 상위 15", loc="left")
ax.axvline(0, color="#444444", lw=0.6); despine(ax)
from matplotlib.patches import Patch                                     # noqa: E402
hs = [Patch(facecolor="#777777", edgecolor="#333333", label="CatBoost 직접"),
      Patch(facecolor="white", edgecolor="#777777", hatch="////", label="Stefan+CatBoost 잔차 (λ=1)")]
hs += [Patch(facecolor=L.GROUP_COLOR[g], label=g) for g in L.GROUPS]
ax.legend(handles=hs, loc="lower right", fontsize=8.5, ncol=1)
for k, reg in enumerate(REGIONS):
    axb = fig.add_subplot(gs[k // 2, 1 + k % 2])
    t5 = timp[timp.region == reg].nlargest(5, "delta_rmse_cm").sort_values("delta_rmse_cm")
    axb.barh(np.arange(5), t5.delta_rmse_cm, 0.6, xerr=t5.delta_rmse_sd, color=[L.GROUP_COLOR[g] for g in t5.group],
             edgecolor="#333333", linewidth=0.5, error_kw=dict(lw=0.7, capsize=2))
    axb.set_yticks(np.arange(5)); axb.set_yticklabels([f"{f} ({g})" for f, g in zip(t5.feature, t5.group)], fontsize=8.2)
    rho = rank[(rank.a == "Alaska_oof") & (rank.b == reg)].spearman_rho.values[0]
    n_ev = int(timp[timp.region == reg].n_eval.iloc[0])
    axb.set_title(f"{reg} (n={n_ev:,}, ρ={rho:.2f})", loc="left", fontsize=10.5)
    axb.set_xlabel("순열 ΔRMSE (cm)", fontsize=9.5); axb.tick_params(axis="x", labelsize=8.5); despine(axb)
    if k == 0:
        axb.text(-0.05, 1.28, "(b) 정보 없음 전이, CatBoost 직접(알래스카 학습), 지역별 상위 5 · ρ = 알래스카 대비 Spearman",
                 transform=axb.transAxes, fontsize=10.5, fontweight="bold", ha="left")
exports = L.savefig(fig, "l2_importance")

L.write_meta(L.OUT / "l2_importance_meta.json", script="scripts/2_evaluation/l2_importance.py", n_rep=args.nrep,
             methods_alaska=METHODS_A, transfer_regions=REGIONS, rmse_base_alaska={m: base[m][0] for m in METHODS_A},
             rmse_base_transfer=base_t, seeds=L.SEEDS, note="ΔRMSE = 순열 후 풀링 RMSE − 기준 풀링 RMSE(cm). fold별 독립 순열. "
             "앵커 모델은 e5_sqrt_tdd 순열이 Stefan 앵커에도 전파. 그룹 동시 순열은 그룹 열에 같은 행 순열 적용.",
             elapsed_s=round(time.time() - t0, 1), exports=exports)
print(imp.sort_values(["method", "rank"]).groupby("method").head(8)[["method", "feature", "group", "delta_rmse_cm", "rank"]].to_string(index=False))
print(pd.DataFrame(grows)[["method", "group", "delta_sum_cm", "delta_joint_cm"]].round(3).to_string(index=False))
print(rank.round(3).to_string(index=False))
print(f"done {time.time()-t0:.0f}s")
