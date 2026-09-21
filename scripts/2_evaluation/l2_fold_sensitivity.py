"""L2-6 알래스카 fold 배정 민감도.

현행 GroupKFold(sklearn 1.3: 블록 크기 내림차순 탐욕 배정, fold 0 = 단일 블록 4,716셀 35%)와 무작위 블록 배정 10회
(블록 순서를 seed로 순열한 뒤 같은 '셀 수 최소 fold' 탐욕 배정; 블록 id만 섞어 GroupKFold에 넘기면 크기순 배정이라 분할이 바뀌지
않음을 함께 확인)에서 Stefan(λ=0)·Stefan+ridge(λ=0.75)·ridge 직접·catboost_lo 직접(3 seed 앙상블)의 풀링 RMSE 분포
(중앙값·5–95%), fold별 값, 대블록(4,716셀) 제외 풀링 값, fold 평균 RMSE.
산출 data/processed/m1/l2_fold_sensitivity.csv(배정×방법) · l2_fold_sensitivity_folds.csv(fold별) · l2_fold_sensitivity_summary.csv ·
     l2_fold_sensitivity_meta.json  그림 outputs/figures/m1/l2_fold_sensitivity.{png,pdf}
실행 python3 scripts/2_evaluation/l2_fold_sensitivity.py [--nrand 10]
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nrand", type=int, default=10)
args = ap.parse_args()
t0 = time.time()
METHODS = ["stefan", "stefan_ridge075", "ridge", "catboost_lo"]

df = L.load()
ak = L.region_idx(df, "Alaska")
blocks = df.block.values[ak]
ub, cnt = np.unique(blocks, return_counts=True)
big_block = ub[np.argmax(cnt)]
print(f"[data] 알래스카 {len(ak):,}셀 · 블록 {len(ub)} · 대블록 {big_block} {cnt.max():,}셀 ({cnt.max()/len(ak):.1%})", flush=True)


def partition_key(folds):
    """블록→fold 배정을 fold 재라벨과 무관한 키(블록 집합의 집합)로."""
    return frozenset(frozenset(np.unique(df.block.values[te]).tolist()) for _, te in folds)


current = L.alaska_folds(df)
key_cur = partition_key(current)
# 문자 그대로 '블록 id 순열 → GroupKFold' 는 크기순 배정이라 분할이 바뀌는지 확인
literal_same = 0
for s in range(1, args.nrand + 1):
    perm = np.random.RandomState(s).permutation(len(ub)); remap = dict(zip(ub, perm))
    g = np.array([remap[b] for b in blocks])
    lit = [(ak[tr], ak[te]) for tr, te in GroupKFold(n_splits=6).split(ak, groups=g)]
    literal_same += int(partition_key(lit) == key_cur)
print(f"[check] 블록 id 순열→GroupKFold 가 현행과 동일한 분할: {literal_same}/{args.nrand}", flush=True)

ASSIGN = [("current", current)] + [(f"random_{s}", L.random_block_folds(df, ak, 6, seed=s)) for s in range(1, args.nrand + 1)]
rows, frows = [], []
for name, folds in ASSIGN:
    print(f"[{name}] fold 크기 {[len(te) for _, te in folds]}", flush=True)
    oof = L.run_oof(df, folds, METHODS, verbose=False)
    y = oof[TARGET].values; nb = oof.block.values != big_block
    sizes = oof.groupby("fold").size()
    for m in METHODS:
        p = oof[f"pred_{m}"].values
        per = [L.rmse(y[oof.fold.values == f], p[oof.fold.values == f]) for f in range(6)]
        rows.append(dict(assignment=name, method=m, rmse_pooled=L.rmse(y, p), rmse_pooled_excl_bigblock=L.rmse(y[nb], p[nb]),
                         rmse_fold_mean=float(np.mean(per)), rmse_fold_sd=float(np.std(per, ddof=1)), rmse_fold_min=float(np.min(per)),
                         rmse_fold_max=float(np.max(per)), max_fold_share=float(sizes.max() / len(oof)), bigblock_fold=int(oof.fold.values[~nb][0]),
                         bigblock_fold_size=int(sizes.loc[int(oof.fold.values[~nb][0])])))
        for f in range(6):
            mf = oof.fold.values == f
            frows.append(dict(assignment=name, method=m, fold=f, n=int(mf.sum()), n_blocks=int(oof.block.values[mf].__len__() and len(np.unique(oof.block.values[mf]))),
                              has_bigblock=bool((~nb & mf).any()), rmse=per[f], bias=L.bias(y[mf], p[mf]), E_fold=float(oof.E_fold.values[mf][0])))
    print("   " + " · ".join(f"{m} {rows[-len(METHODS)+i]['rmse_pooled']:.3f}" for i, m in enumerate(METHODS)) + f" · {time.time()-t0:.0f}s", flush=True)
res, fr = pd.DataFrame(rows), pd.DataFrame(frows)
res.to_csv(L.OUT / "l2_fold_sensitivity.csv", index=False)
fr.to_csv(L.OUT / "l2_fold_sensitivity_folds.csv", index=False)

srows = []
for m in METHODS:
    c = res[(res.assignment == "current") & (res.method == m)].iloc[0]
    r = res[(res.assignment != "current") & (res.method == m)]
    srows.append(dict(method=m, current_pooled=c.rmse_pooled, current_excl_bigblock=c.rmse_pooled_excl_bigblock, current_fold_mean=c.rmse_fold_mean,
                      random_median=r.rmse_pooled.median(), random_p05=r.rmse_pooled.quantile(.05), random_p95=r.rmse_pooled.quantile(.95),
                      random_min=r.rmse_pooled.min(), random_max=r.rmse_pooled.max(), random_excl_bigblock_median=r.rmse_pooled_excl_bigblock.median(),
                      random_fold_mean_median=r.rmse_fold_mean.median(), n_random=len(r)))
summ = pd.DataFrame(srows)
# 방법 간 차(현행 vs 무작위): Stefan 대비 Δ의 분포
for m in METHODS:
    if m == "stefan":
        continue
    d = (res[res.method == m].set_index("assignment").rmse_pooled - res[res.method == "stefan"].set_index("assignment").rmse_pooled)
    summ.loc[summ.method == m, "delta_vs_stefan_current"] = d.loc["current"]
    summ.loc[summ.method == m, "delta_vs_stefan_random_median"] = d.drop("current").median()
    summ.loc[summ.method == m, "delta_vs_stefan_random_p05"] = d.drop("current").quantile(.05)
    summ.loc[summ.method == m, "delta_vs_stefan_random_p95"] = d.drop("current").quantile(.95)
    summ.loc[summ.method == m, "n_random_delta_negative"] = int((d.drop("current") < 0).sum())
summ.to_csv(L.OUT / "l2_fold_sensitivity_summary.csv", index=False)
print(summ.round(3).to_string(index=False))

# ---------------------------------------------------------------- 그림
from polar.plotstyle import use_polar, despine                            # noqa: E402
plt = use_polar()
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
ax = axes[0]
rng = np.random.RandomState(0)
for i, m in enumerate(METHODS):
    r = res[(res.assignment != "current") & (res.method == m)].rmse_pooled.values
    ax.boxplot(r, positions=[i], widths=0.45, showfliers=False, medianprops=dict(color="#333333"), boxprops=dict(color="#666666"),
               whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(r)), r, s=16, color=L.METHOD_COLOR[m], alpha=0.8, zorder=3, marker=L.METHOD_MARKER[m])
    c = res[(res.assignment == "current") & (res.method == m)].iloc[0]
    ax.scatter([i], [c.rmse_pooled], s=95, marker="*", color="#111111", zorder=4, label="현행 GroupKFold" if i == 0 else None)
    ax.scatter([i], [c.rmse_pooled_excl_bigblock], s=60, marker="*", facecolor="white", edgecolor="#111111", zorder=4,
               label="현행, 대블록 제외" if i == 0 else None)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([L.METHOD_LABEL[m].replace(" (", "\n(") for m in METHODS], fontsize=9)
ax.set_ylabel("풀링 RMSE (cm)"); ax.set_title(f"(a) 무작위 블록 fold 배정 {args.nrand}회 분포 (점·상자)", loc="left"); ax.legend(fontsize=9); despine(ax)
ax = axes[1]
fc = fr[fr.assignment == "current"]
for i, m in enumerate(METHODS):
    s = fc[fc.method == m].sort_values("fold")
    ax.plot(s.fold + (i - 1.5) * 0.08, s.rmse, marker=L.METHOD_MARKER[m], color=L.METHOD_COLOR[m], lw=0, ms=6, label=L.METHOD_LABEL[m])
for f, n in fc[fc.method == "stefan"].sort_values("fold")[["fold", "n"]].values:
    ax.text(f, ax.get_ylim()[0] + 0.3, f"n={int(n):,}", ha="center", fontsize=8, color="#555555")
ax.set_xticks(range(6)); ax.set_xticklabels([f"fold {f}" + ("\n(대블록)" if f == int(fc.bigblock_fold.iloc[0] if 'bigblock_fold' in fc else 0) else "") for f in range(6)], fontsize=9)
ax.set_ylabel("fold RMSE (cm)"); ax.set_title("(b) 현행 배정의 fold별 RMSE", loc="left"); ax.legend(fontsize=8.5); despine(ax)
exports = L.savefig(fig, "l2_fold_sensitivity")

L.write_meta(L.OUT / "l2_fold_sensitivity_meta.json", script="scripts/2_evaluation/l2_fold_sensitivity.py", n_random=args.nrand, methods=METHODS,
             seeds=L.SEEDS, big_block=int(big_block), big_block_cells=int(cnt.max()), n_blocks=int(len(ub)),
             literal_shuffle_identical=f"{literal_same}/{args.nrand}",
             note="무작위 배정 = 블록 순서 seed 순열 후 셀 수 최소 fold 탐욕 배정(6 fold). sklearn 1.3 GroupKFold 는 크기순 결정적 배정이라 id 순열로는 분할 불변.",
             elapsed_s=round(time.time() - t0, 1), exports=exports)
print(f"done {time.time()-t0:.0f}s")
