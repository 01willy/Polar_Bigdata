"""정보 병목 진단 그림 (발표 서사축 5).

메시지 하나: ALT 예측 skill은 공변량이 아니라 공간좌표에서 포화한다.
위경도 2개만 쓴 대조모델(Mloc)이 물리 공변량 24개(M9)와 동률 이상이다.
곧 병목은 모델·공변량 개수가 아니라 셀 단위 정보량(공변량이 담지 못한 국소 편차)에 있다.

재료(시각화 전용, 재계산 없음): data/processed/alt_ablation_cell_results.csv
  셀 프로토콜 n=14,348 위치. GBM 고정. spatial_block(0.5°) + LORO(지역제외) 두 검증.
  config: M0 지역평균 · M1 기후 · M2 지형 · M3 기후+지형 · M4 +InSAR · M5 +PolSAR
          · M8 +CCI · M9 전체(24) · Mloc 위경도만(2, 대조군).

출력: outputs/figures/06_deep_learning/alt_info_bottleneck.png/.pdf

실행: python3 scripts/4_visualization/info_bottleneck_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar.plotstyle import use_polar, despine
from polar.outputs import figpath
from cmcrameri import cm as cmc

plt = use_polar()
plt.rcParams["pdf.fonttype"] = 42

res = pd.read_csv("data/processed/alt_ablation_cell_results.csv")

# 표시 순서(정보 추가 순 + 대조군 Mloc). 라벨은 심사자용으로 간결화(내부 코드 병기).
ORDER = [
    ("M0 지역평균", "지역평균\n(baseline)", 0),
    ("M2 지형", "지형\n(DEM 6)", 6),
    ("M3 기후+지형", "기후+지형\n(14)", 14),
    ("M4 +InSAR", "+InSAR\n(19)", 19),
    ("M9 전체", "전체 공변량\n(24)", 24),
    ("Mloc 위치만(대조)", "위경도만\n(2, 대조)", 2),
]
CV = [("spatial_block", "공간블록 검증 (0.5° 이웃 분리)"),
      ("LORO", "지역제외 전이 검증 (LORO)")]

# 냉색 이산 팔레트(붉은계열 회피). 대조군 Mloc은 강조색(짙은 청록)으로 구분.
base = cmc.davos_r
pal = [base(t) for t in np.linspace(0.16, 0.72, len(ORDER) - 1)]
HILITE = "#0b7285"   # Mloc 위경도만: 강조

sd_ref = float(res.target_sd_cm.iloc[0])   # 관측 표준편차(skill 기준선)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
ymax = 0
for ax, (cv, cv_title) in zip(axes, CV):
    sub = res[res.cv_type == cv].set_index("config")
    rmse = [float(sub.loc[cfg, "rmse_cm"]) for cfg, _, _ in ORDER]
    ymax = max(ymax, max(rmse))
    colors = [(HILITE if cfg.startswith("Mloc") else pal[i]) for i, (cfg, _, _) in enumerate(ORDER)]
    x = np.arange(len(ORDER))
    bars = ax.bar(x, rmse, color=colors, edgecolor="0.3", linewidth=0.5, width=0.72,
                  zorder=3)
    # Mloc 기준 수평선: 물리 공변량이 넘지 못하는 하한
    mloc_r = float(sub.loc["Mloc 위치만(대조)", "rmse_cm"])
    ax.axhline(mloc_r, color=HILITE, ls="--", lw=1.2, zorder=2)
    ax.text(-0.35, mloc_r - 0.4, f"위경도만 {mloc_r:.1f} cm", color=HILITE,
            fontsize=9, va="top", ha="left", fontweight="bold")
    for xi, r in zip(x, rmse):
        ax.text(xi, r + 0.25, f"{r:.1f}", ha="center", va="bottom", fontsize=9.5,
                color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in ORDER], fontsize=8.5)
    ax.set_title(cv_title, fontsize=11)
    ax.grid(axis="y", alpha=0.35, lw=0.5)
    ax.grid(axis="x", visible=False)
    despine(ax)

axes[0].set_ylabel("셀 단위 RMSE (cm, 낮을수록 우수)", fontsize=11)
for ax in axes:
    ax.set_ylim(0, ymax * 1.16)

# skill 보조축(우측 패널 오른쪽): skill = 1 - RMSE/σ, σ=관측 표준편차
axr = axes[1].twinx()
axr.set_ylim(0, ymax * 1.16)
tick_r = np.array([12, 15, 18, 21])
axr.set_yticks(tick_r)
axr.set_yticklabels([f"{1 - r / sd_ref:+.2f}" for r in tick_r], fontsize=11)
axr.set_ylabel(f"skill = 1 - RMSE/σ  (σ = {sd_ref:.1f} cm)", fontsize=11.5, color="#555")
axr.grid(False)

fig.suptitle("정보 병목 진단: 예측 skill이 공변량이 아니라 공간좌표에서 포화한다",
             fontsize=14, fontweight="bold", y=1.005)
fig.text(0.5, 0.955,
         "위경도 2개만 쓴 대조모델(Mloc)이 물리 공변량 24개(전체)와 동률 이상. "
         "모델·공변량 추가로는 셀 단위 정보 하한을 넘지 못한다.",
         ha="center", va="top", fontsize=10, color="#333333")
fig.text(0.5, 0.012,
         "GBM 고정, 셀 프로토콜 n=14,348 위치. 두 검증 모두 위경도만(점선)이 전체 공변량 이상. "
         "병목은 모델 용량이 아니라 공변량이 담지 못한 국소 편차(정보량)에 있다.",
         ha="center", va="bottom", fontsize=8.5, color="#666666")
fig.subplots_adjust(left=0.075, right=0.905, top=0.86, bottom=0.13, wspace=0.08)

for ext in ("png", "pdf"):
    fig.savefig(figpath("spatial_dl", "alt_info_bottleneck", ext=ext),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("[fig] outputs/figures/06_deep_learning/alt_info_bottleneck.png+pdf")
print(f"[check] σ(obs)={sd_ref:.2f} cm")
for cv, _ in CV:
    sub = res[res.cv_type == cv].set_index("config")
    print(f"  {cv}: Mloc={sub.loc['Mloc 위치만(대조)','rmse_cm']:.1f}  "
          f"M9전체={sub.loc['M9 전체','rmse_cm']:.1f}  "
          f"M3현재={sub.loc['M3 기후+지형','rmse_cm']:.1f} cm")
