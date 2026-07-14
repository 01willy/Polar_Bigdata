# -*- coding: utf-8 -*-
"""중간보고 PPT 전용 그래프 (EMP·Digital Rock 논문 톤).
- Pretendard 폰트, 큰 굵은 글씨, 뚜렷한 색 구분, 막대 위 값 라벨, 최소 격자, 겹침 없음.
출력: deck/assets/mid/*.png
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- Pretendard 폰트 ----------
for _f in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    try: fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception: pass
FP  = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none"})

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)

# ---------- 색 (뚜렷·선명; 카테고리 대비 강조) ----------
GRAY   = "#9AA0A6"   # 기준선/약함
BLUE   = "#2E86AB"   # 기후·물리
GREEN  = "#2E8B57"   # 양호
NAVY   = "#12324B"   # 강조(본 연구/최고)
AMBER  = "#E08A1E"   # 경고/주목
TEAL   = "#0E7C86"   # 대조군 강조
RED    = "#C0392B"   # 임계/음성
INK    = "#18181B"; SLATE = "#4A5158"; MUTE = "#8A9096"
PAPER  = "#FFFFFF"; GRID = "#E6E8EB"

def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9CDD2"); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(length=0, labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FPM)

# ================================================================= 1. 정보병목
df = pd.read_csv("data/processed/alt_ablation_cell_results.csv")
loro = df[df.cv_type == "LORO"].set_index("config")
order  = ["M2 지형", "M8 +CCI prior", "M5 +PolSAR", "M3 기후+지형", "M1 기후", "M9 전체", "M4 +InSAR", "Mloc 위치만(대조)"]
labels = ["지형", "+CCI prior", "+PolSAR", "기후+지형", "기후", "전체 물리", "+InSAR", "위경도(대조군)"]
skill  = [loro.loc[c, "skill_over_mean"] * 100 for c in order]
fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
colors = [GRAY if s < 0 else (TEAL if "위치만" in c else BLUE) for c, s in zip(order, skill)]
ax.barh(range(len(order)), skill, color=colors, height=0.68, zorder=3)
ax.axvline(0, color=INK, lw=1.1, zorder=2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=13.5)
for t, c in zip(ax.get_yticklabels(), order):
    t.set_fontproperties(FPB if "위치만" in c else FPM)
ax.set_xlim(-8, 18)
for i, v in enumerate(skill):
    ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=13,
            fontproperties=FPB if order[i].startswith("Mloc") else FP,
            color=TEAL if order[i].startswith("Mloc") else INK, zorder=4)
ax.set_xlabel("전이(LORO) skill-over-mean (%)", fontsize=13.5, fontproperties=FP)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
_style(ax)
ax.set_title("위경도 대조군이 물리 공변량 조합을 능가한다", fontsize=16.5, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/bottleneck.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved bottleneck.png")

# ================================================================= 2. SOTA 비교
rows = [
    ("QTP SCE 2025 (청장고원)",      32.68, "무작위 CV",       GRAY),
    ("Gautam 2025 RF (알래스카)",    22.00, "무작위 CV",       GRAY),
    ("Liu 2024 격자 (범북극)",       21.60, "무작위 CV",       GRAY),
    ("ESA CCI 제품 (우리 셀)",       20.55, "제품 직접평가",   GRAY),
    ("ASM 2026 물리 (전지구 55점)",  16.00, "사이트 직접검증", GREEN),
    ("본 연구 앙상블 (알래스카)",    16.95, "공간블록·LORO",   NAVY),
]
fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
ys = np.arange(len(rows))[::-1]
for y, (lab, rmse, proto, col) in zip(ys, rows):
    ax.barh(y, rmse, color=col, height=0.66, zorder=3)
    ax.text(rmse + 0.5, y, f"{rmse:.1f}", va="center", ha="left", fontsize=13.5,
            fontproperties=FPB if "본 연구" in lab else FP, color=INK, zorder=4)
    ax.text(0.6, y, proto, va="center", ha="left", fontsize=10.5, color="white",
            fontproperties=FPM, zorder=5)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
for t, r in zip(ax.get_yticklabels(), rows):
    t.set_fontproperties(FPB if "본 연구" in r[0] else FPM)
ax.set_xlim(0, 37); ax.set_xlabel("보고 RMSE (cm)  ·  낮을수록 정확", fontsize=13.5, fontproperties=FP)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.axvspan(14, 18, color="#EAF3EE", zorder=0)
ax.text(16, len(rows)-0.35, "물리기반 정직 검증군 대역", fontsize=10.5, color=GREEN,
        ha="center", fontproperties=FP)
ax.set_title("프로토콜을 통제하면 정직 검증군과 동일 대역", fontsize=16.5, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/sota.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved sota.png")

# ================================================================= 3. 전이 개선 (ERA5)
e = pd.read_csv("data/processed/stage2_era5_rescore.csv")
lo = e[e.cv == "LORO(전이)"].reset_index(drop=True)
labels3 = ["WorldClim\n(기존)", "+사인 도일", "ERA5-Land\n실측(+고도)", "전체"]
vals3 = list(lo.rmse)
fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=200)
cols = [GRAY, BLUE, NAVY, BLUE]
ax.bar(range(4), vals3, color=cols, width=0.64, zorder=3)
for i, v in enumerate(vals3):
    ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=14,
            fontproperties=FPB if i == 2 else FP, color=INK, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(labels3, fontsize=12)
ax.set_ylabel("전이(LORO) RMSE (cm)", fontsize=13.5, fontproperties=FP)
ax.set_ylim(0, 126)
ax.annotate("", xy=(2, 92), xytext=(0, 112),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4,
                            connectionstyle="arc3,rad=0.15"))
ax.text(1.5, 118, "실측 공변량으로 20% 개선", fontsize=12, color=NAVY,
        ha="center", fontproperties=FPB)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.set_title("실측 공변량(ERA5-Land)이 전이를 개선한다", fontsize=16, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/era5_transfer.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved era5_transfer.png")

# ================================================================= 4. CV 개념도
from matplotlib.patches import Rectangle
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=200)
fig.patch.set_facecolor(PAPER)
rng = np.random.RandomState(3)
base = rng.rand(44, 2)
for ax, (ttl, mode) in zip(axs, [("무작위 CV", "random"), ("공간블록 · LORO", "block")]):
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.12); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="#C9CDD2", lw=1.4))
    for i, (px, py) in enumerate(base):
        if mode == "random":
            c = AMBER if i % 3 == 0 else NAVY
        else:
            c = AMBER if px > 0.6 else NAVY
        ax.plot(px, py, "o", ms=9, color=c, alpha=0.9, mec="white", mew=0.6)
    if mode == "block":
        ax.axvline(0.6, 0.0, 0.893, color=RED, lw=2.2, ls="--")
    ax.text(0.5, 1.06, ttl, ha="center", fontsize=14.5, color=INK, fontproperties=FPB)
axs[0].text(0.5, -0.02, "이웃한 학습·시험이 섞여 낙관", ha="center", va="top",
            fontsize=11.5, color=SLATE, fontproperties=FP)
axs[1].text(0.5, -0.02, "지역째 분리 → 실제 전이 성능", ha="center", va="top",
            fontsize=11.5, color=SLATE, fontproperties=FP)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=NAVY, ms=11, label="학습"),
                    plt.Line2D([], [], marker="o", ls="", color=AMBER, ms=11, label="시험")],
           loc="lower center", ncol=2, frameon=False, prop=FP, fontsize=12, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("무작위 CV는 공간 자기상관으로 약 4배 낙관한다", fontsize=15.5,
             color=INK, y=1.04, fontproperties=FPB)
fig.tight_layout()
fig.savefig(f"{OUT}/cv_concept.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved cv_concept.png")

# ================================================================= 5. 타임라인
fig, ax = plt.subplots(figsize=(11.2, 3.4), dpi=200)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
cols5 = [("완료", NAVY, ["셀 단위 정직 재평가", "정보병목 진단", "6모델 토너먼트", "conformal·AOA·전이", "음성결과 게이트"]),
         ("진행", BLUE, ["KPDC 현장관측 편입", "다지역 데이터(시베리아·티베트)", "외삽 완화 실험(E2·E4)", "예선 보고서"]),
         ("예정", GREEN, ["Stefan 물리결합(E3)", "얕은 3D 열큐브 고도화", "진짜 다지역 전이(E1)", "발표덱·시각화"])]
w = 30
for i, (h, c, items) in enumerate(cols5):
    x = 3 + i*33
    ax.add_patch(Rectangle((x, 24), w, 4.4, fc=c, ec="none"))
    ax.text(x+w/2, 26.2, h, ha="center", va="center", fontsize=14, color="white", fontproperties=FPB)
    for j, it in enumerate(items):
        ax.text(x+0.6, 21.3-j*3.8, "·  "+it, ha="left", va="center", fontsize=10.5, color=INK, fontproperties=FP)
ax.plot([2, 98], [1.5, 1.5], color="#C9CDD2", lw=1.4)
for xx, lab in [(10, "2026-06"), (42, "07-31 예선"), (72, "08~31 본선자료"), (95, "09 본선")]:
    ax.plot(xx, 1.5, "o", ms=8, color=AMBER)
    ax.text(xx, -1.4, lab, ha="center", fontsize=10, color=SLATE, fontproperties=FP)
fig.tight_layout()
fig.savefig(f"{OUT}/timeline.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved timeline.png")

# ================================================================= 6. 모델 토너먼트 (신규 · DL 설명용)
try:
    mt = pd.read_csv("data/processed/model_tournament_results.csv")
except Exception:
    mt = None
mdl = [("지역평균", 20.9, GRAY), ("GBM", 17.24, BLUE), ("MLP", 17.4, BLUE),
       ("FT-Transformer", 17.3, BLUE), ("TabM", 17.2, BLUE), ("Flow", 17.5, BLUE),
       ("Diffusion", 17.09, GREEN), ("앙상블", 16.95, NAVY)]
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
xs = range(len(mdl))
ax.bar(xs, [m[1] for m in mdl], color=[m[2] for m in mdl], width=0.66, zorder=3)
for i, (lab, v, c) in enumerate(mdl):
    ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=12,
            fontproperties=FPB if lab in ("앙상블",) else FP, color=INK, zorder=4)
ax.set_xticks(list(xs)); ax.set_xticklabels([m[0] for m in mdl], rotation=20, ha="right", fontsize=11)
ax.set_ylabel("RMSE (cm)  ·  낮을수록 정확", fontsize=13, fontproperties=FP)
ax.set_ylim(15, 22)
ax.axhspan(16.9, 17.5, color="#EAF0F5", zorder=0)
ax.text(6.5, 17.7, "6모델 전부 동률(부트스트랩)", fontsize=11, color=SLATE, ha="center", fontproperties=FP)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.set_title("정적 활동층 두께에서 6모델이 동률 · 병목은 모델이 아니다", fontsize=15,
             fontproperties=FPB, color=INK, pad=12, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/tournament.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved tournament.png")

# ================================================================= 7. conformal 커버리지 (신규)
fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
cats = ["보정 전\n(raw)", "보정 후\n(conformal)"]
cov = [56.1, 85.9]
bars = ax.bar(cats, cov, color=[AMBER, NAVY], width=0.5, zorder=3)
ax.axhline(90, color=RED, lw=2.0, ls="--", zorder=2)
ax.text(1.45, 90.8, "목표 90%", fontsize=12, color=RED, ha="right", fontproperties=FP)
for i, v in enumerate(cov):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=15, fontproperties=FPB, color=INK, zorder=4)
ax.set_ylim(0, 100); ax.set_ylabel("90% 예측구간 실제 커버리지 (%)", fontsize=13, fontproperties=FP)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
for t in ax.get_xticklabels(): t.set_fontsize(12.5)
ax.set_title("과신하는 예측구간을 목표 수준으로 보정", fontsize=15.5, fontproperties=FPB,
             color=INK, pad=12, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/conformal.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved conformal.png")
