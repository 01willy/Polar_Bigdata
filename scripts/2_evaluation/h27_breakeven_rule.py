"""H27 필요 라벨 수 규칙 — 13 대상의 손익분기 n(h25)을 라벨 없이 계산한 이질성 지표로 예측한다(leave-one-target-out). CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §3 H27, §4 A4.
입력 data/processed/h3/h25_{curve,breakeven,targets}.csv. 손익분기 정의 = 회복률 ≥ 0.5 이고 승률 ≥ 0.75 인 최소 n(규칙 kmedoid, 단계 S3 λ=.25 α=1; S1 도 병기).
지표(라벨 미사용): smd_x25(원천 대비 표준화 평균차), var_ratio_x25, n_blocks, n_cells. 진단(라벨 기반): E_own/E0 비, E_block_cv, within_block_sd_ratio.
모델: log2(손익분기 n) ~ 지표(ridge, 표준화), LOO R²; 손익분기가 격자 최댓값을 넘는 대상(-1)은 n_max·2 로 우측 절단 처리해 순위 상관(Spearman)도 병기.
산출 data/processed/h3/h27_rule.csv, h27_loo.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "h3"
be = pd.read_csv(OUT / "h25_breakeven.csv"); tg = pd.read_csv(OUT / "h25_targets.csv")
T = tg.groupby("target").agg(parent=("parent", "first"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first"), smd_x25=("smd_x25", "mean"),
                             var_ratio_x25=("var_ratio_x25", "mean"), E_own=("E_own", "first"), E0=("E0", "mean"), E_block_cv=("E_block_cv", "first"),
                             within_block_sd_ratio=("within_block_sd_ratio", "first"), y_sd=("y_sd", "first")).reset_index()
T["E_ratio"] = T.E_own / T.E0; T["abs_logE"] = np.abs(np.log(T.E_ratio))
rows = []
for stage, lam in (("S1", 0.0), ("S3", 0.25)):
    b = be[(be.rule == "kmedoid") & (be.stage == stage) & (be.lam == lam) & (be.alpha == 1.0)][["target", "breakeven_n_rec50", "breakeven_n_ci", "n_max", "best_n", "best_d_phys"]]
    M = T.merge(b, on="target")
    M["be_n"] = np.where(M.breakeven_n_rec50 > 0, M.breakeven_n_rec50, 2 * M.n_max)          # 미달성 = 우측 절단
    M["censored"] = M.breakeven_n_rec50 <= 0
    y = np.log2(M.be_n.values.astype(float))
    for name, feats in (("labelfree", ["smd_x25", "var_ratio_x25", "n_blocks", "n_cells"]), ("labelfree+E", ["smd_x25", "var_ratio_x25", "n_blocks", "abs_logE"]), ("E_only", ["abs_logE"])):
        X = M[feats].values.astype(float); X = np.log(X + 1e-9) if name == "labelfree" else X
        pred = np.zeros(len(M))
        for i in range(len(M)):
            tr = np.arange(len(M)) != i
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr]); pred[i] = m.predict(((X[i] - mu) / sd)[None, :])[0]
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rho, p = spearmanr(y, pred)
        rho_E, p_E = spearmanr(M.abs_logE, M.be_n)
        rows.append(dict(stage=stage, lam=lam, features=name, n_targets=len(M), n_censored=int(M.censored.sum()), loo_r2=float(r2), spearman_pred=float(rho), p_pred=float(p),
                         spearman_Eratio_vs_be=float(rho_E), p_Eratio=float(p_E)))
        M[f"pred_{name}"] = 2 ** pred
    M.to_csv(OUT / f"h27_loo_{stage}.csv", index=False)
R = pd.DataFrame(rows); R.to_csv(OUT / "h27_rule.csv", index=False)
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))
print(M[["target", "parent", "n_blocks", "smd_x25", "var_ratio_x25", "E_ratio", "abs_logE", "E_block_cv", "breakeven_n_rec50", "breakeven_n_ci", "best_n", "best_d_phys"]].round(3).to_string(index=False))
