"""스레드 R — R3 게이트: 데이터 재구조화(집계·가중)가 예측을 개선하나?

공정 비교(핵심): 세 방법 모두 **held-out 셀의 평균 ALT 예측**으로 채점(같은 대상, apples-to-apples).
 (1) pooled           : 225k 점 학습 → 점 예측을 셀평균으로 집계  (현재 방식)
 (2) pooled + 1/n 가중 : 조밀셀 편향 제거(㉢)                    (가중만)
 (3) cell-trained     : 14k 셀평균 직접 학습(㉡)                  (집계)
평가: 공간블록(보간) + LORO(전이). 전이 개선이 진짜 이득(정적 데이터 한계 내).
※ 시간정합(㉠)은 ERA5 다년 다운로드 후 별도. 이번은 집계·가중 효과만.

실행: python3 scripts/2_evaluation/restructure_gate.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics
from polar.outputs import figpath
from polar.plotstyle import use_polar, CMAP, despine

plt = use_polar()
PROC = "data/processed"
CLIP = (np.log1p(1), np.log1p(600))
FEATS = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
         "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

cell = pd.read_csv(os.path.join(PROC, "dl_dataset_cell.csv"))
pool = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
cell["block"] = (np.floor(cell.lat / 0.5).astype(int) * 100000 + np.floor(cell.lon / 0.5).astype(int))
loc2block = dict(zip(cell.loc_id, cell.block))
loc2n = dict(zip(cell.loc_id, cell.n_obs))
pool["block"] = pool.loc_id.map(loc2block)
pool["w"] = 1.0 / pool.loc_id.map(loc2n)     # 1/n 가중(위치당 총합 1)
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                            l2_regularization=1.0, early_stopping=True, random_state=0)


def cell_truth(cell_df):
    return dict(zip(cell_df.loc_id, cell_df.alt_cm))


def eval_fold(cell_tr, cell_te, weighted):
    """반환: (loc_id, truth, pred_pooled, pred_cell) for held-out cells."""
    tr_locs = set(cell_tr.loc_id); te_locs = set(cell_te.loc_id)
    p_tr = pool[pool.loc_id.isin(tr_locs)]; p_te = pool[pool.loc_id.isin(te_locs)]
    # (1/2) pooled (가중 옵션)
    m = gbm()
    if weighted:
        m.fit(p_tr[FEATS].values, np.log1p(p_tr.alt_cm.values), sample_weight=p_tr.w.values)
    else:
        m.fit(p_tr[FEATS].values, np.log1p(p_tr.alt_cm.values))
    p_te = p_te.assign(pred=to_cm(m.predict(p_te[FEATS].values)))
    pooled_cellpred = p_te.groupby("loc_id").pred.mean()
    # (3) cell-trained
    mc = gbm(); mc.fit(cell_tr[FEATS].values, np.log1p(cell_tr.alt_cm.values))
    cellpred = pd.Series(to_cm(mc.predict(cell_te[FEATS].values)), index=cell_te.loc_id.values)
    out = cell_te[["loc_id", "alt_cm"]].copy()
    out["pred_pooled"] = out.loc_id.map(pooled_cellpred)
    out["pred_cell"] = out.loc_id.map(cellpred)
    return out


def run(cv_name, splits):
    rows_pool, rows_poolw, rows_cell = [], [], []
    for tr_idx, te_idx in splits:
        cell_tr, cell_te = cell.iloc[tr_idx], cell.iloc[te_idx]
        r = eval_fold(cell_tr, cell_te, weighted=False); rows_pool.append(r)
        rw = eval_fold(cell_tr, cell_te, weighted=True); rows_poolw.append(rw[["loc_id", "pred_pooled"]])
    P = pd.concat(rows_pool); Pw = pd.concat(rows_poolw)
    P = P.merge(Pw.rename(columns={"pred_pooled": "pred_poolw"}), on="loc_id")
    res = []
    for name, col in [("(1) pooled(현재)", "pred_pooled"), ("(2) pooled+1/n가중", "pred_poolw"), ("(3) cell-trained", "pred_cell")]:
        m = all_metrics(P.alt_cm.values, P[col].values); m["method"] = name; m["cv_type"] = cv_name
        res.append(m)
    return pd.DataFrame(res)


gkf = GroupKFold(6)
sp = list(gkf.split(cell, cell.alt_cm, groups=cell.block.values))
loro = []
for r in pd.unique(cell.region):
    te = np.where(cell.region.values == r)[0]; tr = np.where(cell.region.values != r)[0]
    if len(te) >= 30:
        loro.append((tr, te))

allres = pd.concat([run("spatial_block", sp), run("LORO", loro)], ignore_index=True)
allres = allres[["cv_type", "method", "n", "rmse_cm", "mae_cm", "r2", "target_sd_cm", "skill_over_mean"]]
allres.to_csv(os.path.join(PROC, "restructure_gate_results.csv"), index=False)
print("=== R3 게이트: 셀평균 ALT 예측 (모두 동일 대상 채점) ===")
print(allres.to_string(index=False))

# 참고: pooled 점ALT 직접 RMSE (대상 다름 — 맥락용)
within_sd = cell.alt_sd.mean()
print(f"\n[맥락] 셀내 SD 평균(9km로 못 보는 부분) = {within_sd:.2f}cm · 셀평균 대상은 이 잡음이 제거됨")

# ---------- 시각화 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
for ax, cv in zip(axes, ["spatial_block", "LORO"]):
    sub = allres[allres.cv_type == cv].reset_index(drop=True)
    x = np.arange(len(sub))
    best = int(sub.rmse_cm.idxmin())
    colors = ["#0b7285" if i == best else "#8a8f98" if s < 0 else "#1f4e79"
              for i, s in enumerate(sub.skill_over_mean)]
    ax.bar(x, sub.rmse_cm, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(sub.method, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("셀평균 ALT RMSE (cm)  ↓좋음", fontsize=11)
    ax.set_title(f"{cv} 검증", fontsize=12)
    for xi, (rm, s) in enumerate(zip(sub.rmse_cm, sub.skill_over_mean)):
        ax.text(xi, rm + 0.15, f"{rm:.1f}\nskill {s*100:.0f}%", ha="center", fontsize=8.5)
    despine(ax)
fig.suptitle("스레드 R 게이트 — 데이터 재구조화(집계·가중)가 셀평균 ALT 예측을 개선하나\n(전이=LORO 개선이 진짜 이득; 청록=최고)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(figpath("eval", "restructure_gate", ext=ext), dpi=300 if ext == "png" else None, bbox_inches="tight")
print("저장:", figpath("eval", "restructure_gate"))
