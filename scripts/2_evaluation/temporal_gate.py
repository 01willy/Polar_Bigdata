"""스레드 R ㉠ 게이트 — 그 해 기후(시간정합)가 climatology(정적)보다 나은가?

단위: (위치, 연도) 평균 ALT. 같은 위치라도 연도마다 그 해 기후가 다름 → 연도차를 맞히나?
비교(동일 CV, 1/n_year 아님 — 연도는 신호):
  static   : 지형6 + 정적기후8 (2015-2020 평균, e5_*)
  temporal : 지형6 + 그해기후9 (e5t_*, +전년겨울적설)
평가: 공간블록(보간) + LORO(전이) + per-year holdout(그 해 완전 미학습).

출력: dl_dataset_temporal.csv, temporal_gate_results.csv, figures/02_evaluation/temporal_gate.png
실행: python3 scripts/2_evaluation/temporal_gate.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics
from polar.outputs import figpath
from polar.plotstyle import use_polar, despine

plt = use_polar()
PROC = "data/processed"
CLIP = (np.log1p(1), np.log1p(600))
TER = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
STAT = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TEMP = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm", "e5t_tcold", "e5t_stl1", "e5t_swe", "e5t_swe_prevwinter"]

lab = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
lab = lab[(lab.year >= 2010) & (lab.year <= 2024)].copy()
tcov = pd.read_csv(os.path.join(PROC, "alt_era5_temporal.csv"))
merged = lab.merge(tcov, on=["lat", "lon", "year"], how="left")
cov_ok = merged["e5t_tdd"].notna().mean()
print(f"시간정합 조인 커버리지: {cov_ok*100:.1f}% ({merged['e5t_tdd'].notna().sum():,}/{len(merged):,})")
merged = merged[merged["e5t_tdd"].notna()].copy()

# (위치,연도) 단위 집계 — 셀내 반복 제거, 연도는 유지
agg = {c: "first" for c in TER + STAT + TEMP + ["lat", "lon", "region"]}
agg["alt_cm"] = "mean"
u = merged.groupby(["loc_id", "year"]).agg(agg).reset_index()
u["block"] = (np.floor(u.lat / 0.5).astype(int) * 100000 + np.floor(u.lon / 0.5).astype(int))
merged.to_csv(os.path.join(PROC, "dl_dataset_temporal.csv"), index=False)
print(f"(위치,연도) 단위: {len(u):,}행 · 고유위치 {u.loc_id.nunique():,} · 연도 {int(u.year.min())}-{int(u.year.max())}")

y = u.alt_cm.values; ylog = np.log1p(y)
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                            l2_regularization=1.0, early_stopping=True, random_state=0)


def oof(feats, splits):
    p = np.full(len(u), np.nan)
    for tr, te in splits:
        m = gbm(); m.fit(u[feats].values[tr], ylog[tr]); p[te] = to_cm(m.predict(u[feats].values[te]))
    return p


def splits_for(kind):
    if kind == "spatial_block":
        return list(GroupKFold(6).split(u, ylog, groups=u.block.values))
    if kind == "LORO":
        out = []
        for r in pd.unique(u.region):
            te = np.where(u.region.values == r)[0]; tr = np.where(u.region.values != r)[0]
            if len(te) >= 30:
                out.append((tr, te))
        return out
    if kind == "per_year":
        out = []
        for yr in pd.unique(u.year):
            te = np.where(u.year.values == yr)[0]; tr = np.where(u.year.values != yr)[0]
            if len(te) >= 100:
                out.append((tr, te))
        return out


rows = []
for cv in ["spatial_block", "LORO", "per_year"]:
    sp = splits_for(cv)
    for name, feats in [("static(정적기후)", TER + STAT), ("temporal(그해기후)", TER + TEMP)]:
        m = all_metrics(y, oof(feats, sp)); m["method"] = name; m["cv_type"] = cv; m["nfold"] = len(sp)
        rows.append(m)
res = pd.DataFrame(rows)[["cv_type", "method", "nfold", "n", "rmse_cm", "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "temporal_gate_results.csv"), index=False)
print("\n=== ㉠ 시간정합 게이트 ((위치,연도) 평균 ALT) ===")
print(res.to_string(index=False))

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, cv in zip(axes, ["spatial_block", "LORO", "per_year"]):
    sub = res[res.cv_type == cv].reset_index(drop=True)
    x = np.arange(len(sub))
    colors = ["#8a8f98", "#0b7285"]  # static grey, temporal teal
    ax.bar(x, sub.rmse_cm, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(["static\n(정적)", "temporal\n(그해)"], fontsize=9)
    ax.set_ylabel("(위치,연도) ALT RMSE (cm)  ↓좋음", fontsize=10)
    ax.set_title(f"{cv} 검증", fontsize=12)
    for xi, (rm, s) in enumerate(zip(sub.rmse_cm, sub.skill_over_mean)):
        ax.text(xi, rm + 0.15, f"{rm:.1f}\nskill {s*100:.0f}%", ha="center", fontsize=9)
    despine(ax)
fig.suptitle("스레드 R ㉠ — '그 해 기후'가 climatology보다 ALT를 잘 맞히나 (temporal 우세=시간축 이득)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(figpath("eval", "temporal_gate", ext=ext), dpi=300 if ext == "png" else None, bbox_inches="tight")
print("저장:", figpath("eval", "temporal_gate"))
