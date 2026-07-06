"""areal(격자평균) 정확도 — 깨끗한 검증(표본손실 confound 제거).
같은 모델·같은 공간블록 CV의 out-of-fold 예측을 두고, '평가 지지'만 점→격자평균으로 바꿔
대표성 잡음(사이트내 SD)이 상쇄되며 RMSE가 실제로 낮아지는지 정량화.
 point RMSE² ≈ (격자평균 오차)² + (셀내 잡음)²  를 실측 분해.
입력: data/processed/model_tournament_predictions.csv (out-of-fold 예측/실측)
산출: data/processed/areal_eval_results.csv, figures/06_deep_learning/areal_accuracy.png
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, FROZEN, THAWED
from polar.outputs import figpath
plt = use_polar()

pred = pd.read_csv("data/processed/model_tournament_predictions.csv")
MODELS = [c[5:] for c in pred.columns if c.startswith("pred_")]
best = "앙상블(GBM+FT-T)" if "pred_앙상블(GBM+FT-T)" in pred.columns else MODELS[0]
print("모델:", MODELS, "| 대표:", best)

rows = []
for s_km in [0, 0.5, 1, 2.5, 5, 10]:
    if s_km == 0:
        e = pred[f"pred_{best}"].values - pred.alt_cm.values
        rmse = np.sqrt(np.mean(e ** 2)); n = len(pred); wsd = np.nan
    else:
        r = s_km / 111.0
        cy = np.round(pred.lat / r).astype(int); cx = np.round(pred.lon / r).astype(int)
        g = pred.assign(_cy=cy, _cx=cx).groupby(["_cy", "_cx"]).agg(
            pm=(f"pred_{best}", "mean"), om=("alt_cm", "mean"),
            osd=("alt_cm", "std"), n=("alt_cm", "size"))
        e = g.pm.values - g.om.values
        rmse = np.sqrt(np.mean(e ** 2)); n = len(g)
        wsd = float(np.nanmean(g.osd[g.n >= 3]))
    rows.append(dict(support=("점" if s_km == 0 else f"{s_km}km"), n_units=n,
                     rmse_cm=round(rmse, 2), cell_obs_sd=round(wsd, 1) if wsd == wsd else None))
    print(f"support {s_km}km: n={n:,}  격자평균 RMSE={rmse:.2f}cm  셀내SD={wsd if wsd==wsd else 'NA'}")

out = pd.DataFrame(rows)
out.to_csv("data/processed/areal_eval_results.csv", index=False)
print("\n=== areal 평가지지별 정확도(동일 모델·동일 CV) ===")
print(out.to_string(index=False))

# 시각화
xs = [0.35 if r["support"] == "점" else float(r["support"][:-2]) for r in rows]
rm = [r["rmse_cm"] for r in rows]
sd = [r["cell_obs_sd"] for r in rows]
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.plot(xs, rm, "-o", color=FROZEN, lw=2.2, ms=9, label="격자평균 예측 RMSE (동일 모델·CV)")
xsd = [x for x, s in zip(xs, sd) if s]; ysd = [s for s in sd if s]
ax.plot(xsd, ysd, "--s", color=THAWED, lw=1.6, ms=6, label="셀 내부 ALT SD (대표성 잡음)")
for x, v in zip(xs, rm):
    ax.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=9, color=FROZEN)
ax.set_xscale("symlog", linthresh=0.4)
ax.set_xticks([0.35, 0.5, 1, 2.5, 5, 10]); ax.set_xticklabels(["점", "0.5", "1", "2.5", "5", "10"])
ax.set_xlabel("평가 지지(support) 크기 (km)"); ax.set_ylabel("RMSE / SD (cm)")
ax.set_title("깨끗한 areal 검증: 격자평균으로 평가해도 RMSE 안 줄어듦\n"
             "(오차가 랜덤 잡음이 아니라 계통적 — 셀간 ALT를 못 맞힘. areal 재프레이밍 무익 확인)", weight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(figpath("deep_learning", "areal_accuracy")); plt.close(fig)
print("saved", figpath("deep_learning", "areal_accuracy"))
