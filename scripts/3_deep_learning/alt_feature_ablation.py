"""스레드 A — ALT 다중모달 feature-group ablation (GBM 고정, 공간블록+LORO).

질문: "무엇이 ALT를 지배하는가?" 를 누설통제 하에 정직하게 분해.
- 모델 고정 = GBM(HistGradientBoosting). NaN 네이티브 처리 → 부분커버리지(PolSAR 64%) 그대로.
- feature 그룹: TERRAIN(6) · CLIMATE(8) · INSAR(5) · POLSAR(3). (SoilGrids/식생/CCI = 미취득, pending)
- 평가: 공간블록(0.5° GroupKFold) + LORO(leave-one-region-out). 표준지표(rmse/mae/bias/r2/target_sd/skill).

실행: python3 scripts/3_deep_learning/alt_feature_ablation.py
"""
import sys, os, time
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

# ---------- 데이터 결합 (행 정렬 검증됨) ----------
base = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
pol = pd.read_csv(os.path.join(PROC, "dl_dataset_polsar.csv"),
                  usecols=["polsar_alt", "polsar_std", "polsar_valid"])
ins = pd.read_csv(os.path.join(PROC, "dl_dataset_insar.csv"),
                  usecols=["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"])
df = pd.concat([base, pol, ins], axis=1)

TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]

CONFIGS = [
    ("M0 지역평균(baseline)", None),
    ("M1 지형(DEM)", TERRAIN),
    ("M2 기후(ERA5)", CLIMATE),
    ("M3 기후+지형 [현재]", CLIMATE + TERRAIN),
    ("M4 +InSAR", CLIMATE + TERRAIN + INSAR),
    ("M5 +PolSAR", CLIMATE + TERRAIN + POLSAR),
    ("M6 +InSAR+PolSAR", CLIMATE + TERRAIN + INSAR + POLSAR),
]
PENDING = ["M7 +Sentinel/식생 (미취득)", "M8 +SoilGrids 토양 (서버복구 대기)", "M9 +CCI prior (추출 예정)"]

# 공간블록 group (0.5°)
df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
y_cm = df["alt_cm"].values
y_log = np.log1p(y_cm)


def to_cm(p):
    return np.expm1(np.clip(p, *CLIP))


def fit_predict_oof(feats, groups, splitter):
    """OOF 예측 배열 반환(cm). feats=None이면 그룹평균 baseline."""
    oof = np.full(len(df), np.nan)
    if feats is None:
        # 지역평균 baseline: train fold의 region 평균
        for tr, te in splitter:
            reg = df.iloc[tr].groupby("region")["alt_cm"].mean()
            gm = df.iloc[tr]["alt_cm"].mean()
            oof[te] = df.iloc[te]["region"].map(reg).fillna(gm).values
        return oof
    X = df[feats].values
    for tr, te in splitter:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          early_stopping=True, random_state=0)
        m.fit(X[tr], y_log[tr])
        oof[te] = to_cm(m.predict(X[te]))
    return oof


def spatial_splits():
    gkf = GroupKFold(n_splits=6)
    return list(gkf.split(df, y_log, groups=df["block"].values))


def loro_splits():
    reg = df["region"].values
    out = []
    for r in pd.unique(reg):
        te = np.where(reg == r)[0]
        tr = np.where(reg != r)[0]
        if len(te) >= 200:  # 너무 작은 region 제외
            out.append((tr, te))
    return out


rows = []
best_oof = None
for cv_name, splits in [("spatial_block", spatial_splits()), ("LORO", loro_splits())]:
    print(f"\n===== CV: {cv_name} ({len(splits)} folds) =====")
    for name, feats in CONFIGS:
        t0 = time.time()
        oof = fit_predict_oof(feats, None, splits)
        m = all_metrics(y_cm, oof)
        m.update({"config": name, "cv_type": cv_name, "nfeat": (0 if feats is None else len(feats))})
        rows.append(m)
        if cv_name == "spatial_block" and name.startswith("M6"):
            best_oof = oof.copy()
        print(f"  {name:24s} rmse={m['rmse_cm']:.2f}  r2={m['r2']:.3f}  skill={m['skill_over_mean']*100:.1f}%  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)[["cv_type", "config", "nfeat", "n", "rmse_cm", "mae_cm", "bias_cm",
                          "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "alt_feature_ablation_results.csv"), index=False)
print("\n저장:", os.path.join(PROC, "alt_feature_ablation_results.csv"))

# M6 OOF 저장(AOA/UQ 스레드용)
if best_oof is not None:
    pd.DataFrame({"lat": df.lat, "lon": df.lon, "region": df.region, "alt_cm": y_cm,
                  "pred_cm": best_oof}).to_csv(os.path.join(PROC, "alt_ablation_M6_oof.csv"), index=False)

# ---------- waterfall 시각화 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
ymax = res.rmse_cm.max() * 1.15  # 두 패널 공유 y축(정직한 교차비교)
for ax, cv in zip(axes, ["spatial_block", "LORO"]):
    sub = res[res.cv_type == cv].reset_index(drop=True)
    x = np.arange(len(sub))
    best = int(sub.rmse_cm.idxmin())
    colors = ["#0b7285" if i == best else ("#8a8f98" if s < 0 else "#1f4e79")
              for i, s in enumerate(sub.skill_over_mean)]
    ax.bar(x, sub.rmse_cm, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(sub.config, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (cm)  ↓좋음", fontsize=11)
    ax.set_title(f"{cv} 검증: 모달리티별 ALT 예측 기여", fontsize=12)
    for xi, (r, s) in enumerate(zip(sub.rmse_cm, sub.skill_over_mean)):
        neg = s < 0
        ax.text(xi, r + 0.4, f"{r:.1f}\n{s*100:.0f}%", ha="center", fontsize=8,
                color=("#8a8f98" if neg else "#333333"), fontweight=("bold" if neg else "normal"))
    ax.set_ylim(0, ymax)
    despine(ax)
axes[0].text(0.02, 0.97, "파랑=양(+)skill · 청록=최고 · 회색=음(−)skill(평균보다 나쁨)",
             transform=axes[0].transAxes, fontsize=8, va="top", color="#555")
fig.suptitle('스레드 A — "무엇이 ALT를 지배하나": 다중모달 feature ablation (GBM 고정, 하단 %=skill-over-mean)',
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(figpath("spatial_dl", "alt_feature_ablation", ext=ext),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("저장:", figpath("spatial_dl", "alt_feature_ablation"))
print("\n미취득(다음 데이터 확장):", ", ".join(PENDING))
