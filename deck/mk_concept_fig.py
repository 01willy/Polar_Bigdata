# -*- coding: utf-8 -*-
"""연구 개념 모식도 — 입력(다중모달) → 모델 → 출력(ALT·3D·불확실성·전이).
깨끗한 냉색, 겹침 없는 배치, 한글 가독. 출력 deck/assets/mid/concept.png
"""
import os, sys
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
from polar.plotstyle import use_polar
plt = use_polar()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)
TEAL="#0E5A61"; TEAL2="#2C7D83"; NAVY="#1F3A52"; SLATE="#5C666E"; MUTE="#8A9096"
INK="#1B1E23"; PAPER="#FBFAF6"; CARD="#EEF2F1"; RULE="#D9D6CC"

fig, ax = plt.subplots(figsize=(11, 4.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)

def box(x, y, w, h, title, items, head=TEAL, fc=CARD):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=fc, ec=RULE, lw=1.0, zorder=2))
    ax.text(x+w/2, y+h-2.0, title, ha="center", va="center", fontsize=11.5,
            color=head, fontweight="bold", zorder=3)
    for i, it in enumerate(items):
        ax.text(x+1.6, y+h-4.6-i*3.0, it, ha="left", va="center", fontsize=9.3,
                color=INK, zorder=3)

# 입력(좌)
box(2, 6, 26, 34, "입력: 다중모달 관측", [
    "· 위성 SAR / InSAR (P·L·C밴드)",
    "· ERA5-Land 재분석 기후",
    "· 고해상 지형 (ArcticDEM)",
    "· 전지구 시추공 지중온도",
    "· CALM / ABoVE 활동층 관측",
    "· KPDC 현장관측(Council 등)",
], head=NAVY)

# 모델(중)
box(37, 13, 24, 20, "모델", [
    "· GBM 조건장 (주력)",
    "· 6모델 토너먼트 비교",
    "· 물리 결합 (Stefan 잔차)",
    "· 셀 단위·1/n 가중",
], head=TEAL)
ax.text(49, 9.5, "누설 통제 평가\n공간블록 · LORO", ha="center", va="center",
        fontsize=9.0, color=SLATE, style="italic")

# 출력(우)
box(70, 6, 28, 34, "출력", [
    "· 활동층 두께(ALT) 2D 지도",
    "· 얕은 3D 지중 열구조(0~20m)",
    "· 0°C 등온면(영구동토 상단)",
    "· 셀별 보정 불확실성(conformal)",
    "· 적용범위(AOA) 마스크",
    "· 알래스카→타 지대 전이 검증",
], head=TEAL2)

# 화살표
for x0, x1 in [(28.5, 36.5), (61.5, 69.5)]:
    ax.add_patch(FancyArrowPatch((x0, 23), (x1, 23), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.0, color=SLATE, zorder=1))

ax.text(50, 43.5, "관측기반 GeoAI: 무엇이 활동층 두께를 지배하는지 분해하고, 어디까지 믿을 수 있는지까지 제시한다",
        ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/concept.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)
print("saved concept.png")
