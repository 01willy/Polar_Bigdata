"""스레드 C — "17cm는 물리벽이 아니다" apparent-floor 진단 (hero figure).

메시지: 현재 ALT RMSE 17cm의 상당부분은 물리하한이 아니라
  (1) pseudo-replication: 같은 공변량 셀에 ALT가 34~96cm로 공존(같은 X→다른 y)
  (2) 척도불일치: 라벨=30m 점, 기후공변량=9km → 셀내 미세지형 변동을 공변량이 못 봄
  → 셀 단위 covariate로는 셀내 분산이 비가역. 하지만 그 비가역 하한(~within SD)은
     현재 17cm보다 훨씬 낮음 = 아직 covariate 정보 헤드룸이 있다.

실행: python3 scripts/2_evaluation/diagnose_apparent_floor.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from polar.plotstyle import use_polar, CMAP
from polar.outputs import figpath

plt = use_polar()
PROC = "data/processed"

df = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"),
                 usecols=["loc_id", "year", "alt_cm", "e5_maat", "dem_elev"])
N = len(df)
nloc = df.loc_id.nunique()

# ---------- 분산분해 (within-loc = 현재 공변량으로 비가역) ----------
g = df.groupby("loc_id")["alt_cm"]
n_i = g.size().values
mean_i = g.mean().values
var_i = g.var(ddof=0).values  # population
grand = df.alt_cm.mean()
within_ss = np.nansum(n_i * var_i)
between_ss = np.nansum(n_i * (mean_i - grand) ** 2)
total_ss = within_ss + between_ss
within_frac = within_ss / total_ss
irreducible_rmse = np.sqrt(within_ss / N)          # 셀 covariate 완벽 예측시 하한
total_sd = np.sqrt(total_ss / N)
current_best = 16.95                                 # 토너먼트 최고(앙상블)
print(f"N={N}, 고유 loc={nloc}")
print(f"분산분해: within(셀내) {within_frac*100:.1f}%  /  between(셀간) {(1-within_frac)*100:.1f}%")
print(f"비가역 RMSE 하한(셀내 SD) = {irreducible_rmse:.1f}cm  |  전체 SD = {total_sd:.1f}cm  |  현재 최고 = {current_best:.1f}cm")
print(f"→ 현재 17cm는 비가역하한 {irreducible_rmse:.0f}cm과 전체SD {total_sd:.0f}cm 사이 = covariate 병목(헤드룸 존재)")

# 셀내 SD 분포(관측 많은 셀)
big = g.agg(["size", "std"]).reset_index()
big = big[big["size"] >= 20]
within_sd_med = big["std"].median()

# 예시 셀(관측 최다)
ex_id = g.size().idxmax()
ex = df[df.loc_id == ex_id]
ex_cov = ex.iloc[0]

# support→RMSE
gs = pd.read_csv(os.path.join(PROC, "grid_support_results.csv"))

# ---------- 4패널 ----------
fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3))

# A: 예시 셀 — 같은 공변량, 다른 ALT
ax = axes[0]
ax.hist(ex.alt_cm, bins=24, color=CMAP.alt(0.55), edgecolor="white", linewidth=0.4)
ax.axvline(ex.alt_cm.mean(), color="#b5651d", lw=2, label=f"모델이 낼 수 있는 값\n(평균 {ex.alt_cm.mean():.0f}cm)")
ax.set_xlabel("ALT (cm)", fontsize=11)
ax.set_ylabel("관측 점 수", fontsize=11)
ax.set_title(f"① 한 셀 안: 같은 입력 X, 다른 정답 y\n동일 공변량(고도 {ex_cov.dem_elev:.0f}m, MAAT {ex_cov.e5_maat:.1f}°C)"
             f"\n인데 ALT {ex.alt_cm.min():.0f}~{ex.alt_cm.max():.0f}cm (n={len(ex)})", fontsize=11)
ax.legend(fontsize=8, loc="upper left")

# B: 셀내 ALT SD 분포
ax = axes[1]
ax.hist(big["std"].dropna(), bins=40, color=CMAP.err(0.5), edgecolor="white", linewidth=0.3)
ax.axvline(within_sd_med, color="#0b7285", lw=2, label=f"중앙값 {within_sd_med:.1f}cm")
ax.set_xlabel("셀내 ALT 표준편차 (cm)", fontsize=11)
ax.set_ylabel("셀 수 (n≥20 관측)", fontsize=11)
ax.set_title("② 셀내 ALT 변동 = 공변량이 못 보는 잡음\n(라벨 30m vs 기후 9km 척도불일치)", fontsize=10)
ax.legend(fontsize=8)

# C: 분산분해
ax = axes[2]
ax.bar([0], [within_frac * 100], color=CMAP.err(0.55), hatch="///", edgecolor="white",
       label=f"셀내(현 공변량 설명불가) {within_frac*100:.0f}%")
ax.bar([0], [(1 - within_frac) * 100], bottom=[within_frac * 100], color=CMAP.alt(0.55), edgecolor="white",
       label=f"셀간(공변량이 설명가능) {(1-within_frac)*100:.0f}%")
ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_ylim(0, 100)
ax.set_ylabel("ALT 분산 분해 (%)", fontsize=11)
ax.set_title("③ 분산분해\n비가역은 일부, 대부분은 셀간(=더 나은 공변량으로 설명 가능)", fontsize=10)
ax.legend(fontsize=8, loc="center right")

# D: 비가역하한 vs 현재 vs 집계
ax = axes[3]
labels = ["비가역\n하한", "현재\n최고", "1km\n집계", "5km\n집계"]
gs_map = dict(zip(gs["support_km"].astype(str), gs["rmse_cm"]))
vals = [irreducible_rmse, current_best, gs_map.get("1", np.nan), gs_map.get("5", np.nan)]
colors = ["#2e8b9e", "#1f4e79", "#8a8f98", "#8a8f98"]
ax.bar(range(4), vals, color=colors)
for i, v in enumerate(vals):
    if np.isfinite(v):
        ax.text(i, v + 0.3, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("RMSE (cm)", fontsize=11)
ax.set_title("④ 17cm ≠ 물리벽\n비가역하한↔현재 사이 헤드룸(집계는 오히려 악화)", fontsize=10)

fig.suptitle('"apparent floor" 진단 — 17cm는 물리하한이 아니라 공변량 정보병목 + 척도불일치',
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(figpath("eval", "apparent_floor_diagnosis", ext=ext),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("저장:", figpath("eval", "apparent_floor_diagnosis"))

# 근거 CSV
pd.DataFrame([{
    "n_points": N, "n_unique_loc": nloc, "within_frac_pct": round(within_frac * 100, 1),
    "between_frac_pct": round((1 - within_frac) * 100, 1),
    "irreducible_rmse_cm": round(irreducible_rmse, 2), "total_sd_cm": round(total_sd, 2),
    "current_best_rmse_cm": current_best, "within_cell_sd_median_cm": round(within_sd_med, 2),
}]).to_csv(os.path.join(PROC, "apparent_floor_diagnosis.csv"), index=False)
print("근거:", os.path.join(PROC, "apparent_floor_diagnosis.csv"))
