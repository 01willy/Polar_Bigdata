"""기존 결과 CSV 재채점 — RMSE 옆에 R²·target_SD·skill-over-mean 병기 (P1).

정정 근거 생성:
- model_tournament: 예측파일에서 표준지표 재계산 → *_rescored.csv
- curated_scope: alt_sd로 skill-over-mean 계산 → 큐레이션이 skill을 낮춘다는 것 시각화
  (절대 RMSE↓는 범위축소 아티팩트 = "12.97 SOTA 돌파" 반증)

실행: python3 scripts/2_evaluation/rescore_results.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from polar.eval_metrics import all_metrics, skill_over_mean
from polar.plotstyle import use_polar, CMAP
from polar.outputs import figpath

plt = use_polar()
PROC = "data/processed"

# ---------- 1. 토너먼트 재채점 (예측 기반) ----------
tour_pred = os.path.join(PROC, "model_tournament_predictions.csv")
rows = []
if os.path.exists(tour_pred):
    dp = pd.read_csv(tour_pred)
    pred_cols = [c for c in dp.columns if c.startswith("pred_")]
    for c in pred_cols:
        model = c.replace("pred_", "")
        m = all_metrics(dp["alt_cm"].values, dp[c].values)
        m["model"] = model
        rows.append(m)
    tour = pd.DataFrame(rows).sort_values("rmse_cm")
    tour = tour[["model"] + [c for c in tour.columns if c != "model"]]
    tour.to_csv(os.path.join(PROC, "model_tournament_results_rescored.csv"), index=False)
    print("=== 토너먼트 재채점 (전체 예측 기반) ===")
    print(tour[["model", "rmse_cm", "mae_cm", "r2", "target_sd_cm", "skill_over_mean"]].to_string(index=False))
else:
    tour = None
    print("model_tournament_predictions.csv 없음 — 토너먼트 재채점 건너뜀")

# ---------- 2. 큐레이션 재채점 (alt_sd 기반 skill) ----------
cur = pd.read_csv(os.path.join(PROC, "curated_scope_results.csv"))
model_cols = ["gbm14", "gbm18", "diff18", "ens18"]
out = cur[["scope", "n", "alt_sd", "polsar_r"]].copy()
for mc in model_cols:
    out[mc + "_rmse"] = cur[mc]
    out[mc + "_skill"] = 1.0 - cur[mc] / cur["alt_sd"]
    out[mc + "_r2"] = 1.0 - (cur[mc] / cur["alt_sd"]) ** 2  # R² = 1-(RMSE/SD)^2 (mean-model 기준)
out.to_csv(os.path.join(PROC, "curated_scope_results_rescored.csv"), index=False)
print("\n=== 큐레이션 재채점 (best=ens18) ===")
print(out[["scope", "n", "alt_sd", "ens18_rmse", "ens18_skill", "ens18_r2"]].to_string(index=False))
print("→ 절대 RMSE는 평탄지에서↓지만 skill-over-mean은 전역이 더 높음 = '돌파'가 아니라 범위축소 아티팩트")

# ---------- 3. 정정 시각화 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
# 좌: 큐레이션 — RMSE vs skill (이중축)
sc = out.iloc[::-1].reset_index(drop=True)  # 전역→평탄 순
x = np.arange(len(sc))
ax = axes[0]
b = ax.bar(x - 0.2, sc["ens18_rmse"], width=0.4, color=CMAP.alt(0.55), label="RMSE (cm)")
ax.set_ylabel("RMSE (cm)  ↓좋음", fontsize=11)
ax.set_ylim(0, 22)
ax2 = ax.twinx()
ax2.plot(x + 0.0, sc["ens18_skill"] * 100, "o-", color="#0b7285", lw=1.8, ms=6,
         label="skill-over-mean (%)")
ax2.set_ylabel("skill-over-mean (%)  ↑좋음", color="#0b7285", fontsize=11)
ax2.set_ylim(0, 18)
ax2.tick_params(axis="y", colors="#0b7285")
ax.set_xticks(x)
ax.set_xticklabels([s.split(" ")[0].replace("(다양지형)", "") for s in sc["scope"]], fontsize=9)
ax.set_title("큐레이션은 RMSE를 낮추지만 설명력(skill)은 낮춘다\n= '12.97cm SOTA 돌파'는 범위축소 아티팩트", fontsize=11.5)
for xi, (r, s) in enumerate(zip(sc["ens18_rmse"], sc["ens18_skill"])):
    ax.text(xi - 0.2, r + 0.3, f"{r:.1f}", ha="center", fontsize=9)
    ax2.text(xi, s * 100 + 0.5, f"{s*100:.1f}%", ha="center", fontsize=9, color="#0b7285")
# 범례 통합
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

# 우: 토너먼트 — RMSE with R²/skill 주석 (전부 동률)
ax = axes[1]
if tour is not None:
    t = tour.sort_values("rmse_cm")
    y = np.arange(len(t))
    bar_c = [("#8a8f98" if s < 0 else CMAP.count(0.5)) for s in t["skill_over_mean"]]
    ax.barh(y, t["rmse_cm"], color=bar_c)
    ax.set_yticks(y); ax.set_yticklabels(t["model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (cm)", fontsize=11)
    ax.set_xlim(0, max(t["rmse_cm"]) * 1.3)
    for yi, (r, s) in enumerate(zip(t["rmse_cm"], t["skill_over_mean"])):
        neg = s < 0
        ax.text(r + 0.2, yi, f"{r:.1f} (skill {s*100:.0f}%)", va="center", fontsize=9,
                color=("#8a8f98" if neg else "#333333"), fontweight=("bold" if neg else "normal"))
    ax.set_title("모델 토너먼트: RMSE 옆 skill 병기\n최고 모델도 평균 대비 ~12%만 개선(정보병목)", fontsize=11.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = figpath("eval", "skill_reframing", ext=ext)
    fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
print("\n저장:", figpath("eval", "skill_reframing"))
