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

def _style(ax, ts=12):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9CDD2"); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(length=0, labelsize=ts)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FPM); t.set_fontsize(ts)

def _src(fig, txt):
    """데이터 출처 각주(좌하단, 회색 소자)."""
    fig.text(0.01, -0.015, txt, fontsize=8.5, color=MUTE, fontproperties=FPM,
             ha="left", va="top")

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
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
ax.set_xlim(-8, 18)
for i, v in enumerate(skill):
    ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=13.5,
            fontproperties=FPB if order[i].startswith("Mloc") else FP,
            color=TEAL if order[i].startswith("Mloc") else INK, zorder=4)
ax.set_xlabel("전이(LORO) skill (%)  ·  평균 예측 대비 RMSE 개선율", fontsize=15, fontproperties=FP)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
_style(ax, ts=14)
for t, c in zip(ax.get_yticklabels(), order):
    if "위치만" in c:
        t.set_fontproperties(FPB); t.set_fontsize(14)
ax.set_title("위경도 대조군이 물리 공변량 조합을 능가한다", fontsize=16.5, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
_src(fig, "자료: alt_ablation_cell_results.csv · 셀 단위 LORO 평가(n=14,268)")
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
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 40); ax.set_xlabel("보고 RMSE (cm)  ·  낮을수록 정확", fontsize=14, fontproperties=FP)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax, ts=12.5)
for t, r in zip(ax.get_yticklabels(), rows):
    if "본 연구" in r[0]:
        t.set_fontproperties(FPB); t.set_fontsize(12.5)
ax.axvspan(14, 18, color="#EAF3EE", zorder=0)
ax.text(16, len(rows)-0.35, "물리기반 정직 검증군 대역", fontsize=10.5, color=GREEN,
        ha="center", fontproperties=FP)
ax.set_ylim(-1.15, len(rows)-0.45)
ax.text(39.4, -0.82, "LORO = 지역 제외 평가(한 지역을 통째로 제외하고 학습해 그 지역에서 평가)",
        fontsize=10.5, color=SLATE, ha="right", va="center", fontproperties=FPM, zorder=4)
ax.set_title("프로토콜을 통제하면 정직 검증군과 동일 대역", fontsize=16.5, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
_src(fig, "자료: 각 연구 보고값 · ESA CCI는 본 연구 평가 셀에서 직접 채점 · 본 연구는 공간블록·LORO 평가")
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
ax.axhline(vals3[0], color=SLATE, lw=1.3, ls="--", zorder=2)
ax.text(3.42, vals3[0] + 4.5, f"기준 {vals3[0]:.1f}", fontsize=11, color=SLATE,
        ha="right", va="bottom", fontproperties=FPM, zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
for i, v in enumerate(vals3):
    ax.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=13.5,
            fontproperties=FPB if i == 2 else FP,
            color=NAVY if i == 2 else INK, zorder=4)
# −20% 라벨은 기준선과 겹치지 않도록 화살표 주석으로 분리(기준선→개선 막대 상단)
ax.annotate("", xy=(2.42, vals3[2]), xytext=(2.42, vals3[0]),
            arrowprops=dict(arrowstyle="<|-|>", color=NAVY, lw=1.8), zorder=4)
ax.text(2.34, (vals3[0] + vals3[2]) / 2, "−20%", fontsize=14, color=NAVY,
        ha="right", va="center", fontproperties=FPB, zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
ax.set_xticks(range(4)); ax.set_xticklabels(labels3, fontsize=12)
ax.set_ylabel("전이(LORO) RMSE (cm)", fontsize=13.5, fontproperties=FP)
ax.set_ylim(0, 125)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.set_title("실측 공변량(ERA5-Land)이 전이 오차를 20% 낮춘다", fontsize=16, fontproperties=FPB,
             color=INK, pad=14, loc="left")
fig.tight_layout()
_src(fig, "자료: stage2_era5_rescore.csv · LORO(지역 전이) 평가")
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

# ================================================================= 4b. CV 방식별 누설 정량화 (슬라이드 12 전용)
lk = pd.read_csv("data/processed/cv_leakage_table.csv")
schemes = ["무작위 K-fold", "site-disjoint", "공간블록(500km)", "LORO(지역전이)"]
xlab4 = ["무작위 K-fold", "사이트 분리", "공간블록(500km)", "LORO(지역 전이)"]
idw = [float(lk[(lk.cv == s) & (lk.method == "IDW")].rmse.iloc[0]) for s in schemes]
gbm = [float(lk[(lk.cv == s) & (lk.method == "GBM(공변량)")].rmse.iloc[0]) for s in schemes]
x4 = np.arange(4); wbar = 0.36
fig, ax = plt.subplots(figsize=(10.8, 3.9), dpi=200)
# IDW 계열은 색 + 빗금(hatch)으로 이중 부호화 — 저해상 투사에서도 GBM과 구분
ax.bar(x4 - wbar/2, idw, wbar, color=TEAL, label="IDW(공간보간)", zorder=3,
       hatch="//", edgecolor="white", linewidth=0.4)
ax.bar(x4 + wbar/2, gbm, wbar, color=NAVY, label="GBM(공변량)", zorder=3)
for xx, v in zip(x4 - wbar/2, idw):
    ax.text(xx, v + 3, f"{v:.0f}", ha="center", fontsize=12.5, fontproperties=FPB, color=TEAL, zorder=4)
for xx, v in zip(x4 + wbar/2, gbm):
    ax.text(xx, v + 3, f"{v:.0f}", ha="center", fontsize=12.5, fontproperties=FP, color=INK, zorder=4)
ax.annotate("", xy=(3 - wbar/2, idw[3] + 13), xytext=(0 - wbar/2, idw[0] + 13),
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2,
                            connectionstyle="arc3,rad=-0.12"))
ax.text(1.5, 132, "무작위 28.0 → LORO 111.4 cm  (약 4배)", fontsize=12.5, color=AMBER,
        ha="center", fontproperties=FPB, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.2))
ax.set_xticks(x4); ax.set_xticklabels(xlab4, fontsize=12.5)
for t in ax.get_xticklabels(): t.set_fontproperties(FPM)
ax.set_ylabel("RMSE (cm)", fontsize=13, fontproperties=FP)
ax.set_ylim(0, 145)
ax.legend(loc="upper left", frameon=False, prop=FP, fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.set_title("같은 데이터·같은 모델에서 CV 분할 방식만 바꾼 결과", fontsize=15.5,
             fontproperties=FPB, color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "자료: cv_leakage_table.csv · 점 단위 전 지구 평가(n=3,604) · 셀 단위 수치(16.95cm)와 스케일이 다름")
fig.savefig(f"{OUT}/cv_leakage.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved cv_leakage.png")

# ================================================================= 5. 타임라인
fig, ax = plt.subplots(figsize=(11.2, 3.4), dpi=200)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
cols5 = [("완료", NAVY, ["셀 단위 정직 재평가", "정보병목 진단", "6모델 토너먼트", "다지역 데이터 확충(시베리아·티베트)"]),
         ("진행", BLUE, ["3D 지중온도장 라벨 증강(main)", "3D 기질추정·Stefan 잔차(병렬)", "불확실성·전이 검증", "예선 보고서"]),
         ("예정", GREEN, ["개선 물리 결합", "3D 지중온도장 고도화", "진짜 다지역 전이", "본선 자료·발표"])]
w = 30
for i, (h, c, items) in enumerate(cols5):
    x = 3 + i*33
    ax.add_patch(Rectangle((x, 24), w, 4.4, fc=c, ec="none"))
    ax.text(x+w/2, 26.2, h, ha="center", va="center", fontsize=14, color="white", fontproperties=FPB)
    for j, it in enumerate(items):
        ax.text(x+0.6, 21.3-j*3.8, "·  "+it, ha="left", va="center", fontsize=10.5, color=INK, fontproperties=FP)
ax.plot([2, 98], [1.5, 1.5], color="#C9CDD2", lw=1.4)
for xx, lab in [(10, "2026.06"), (42, "07.31 예선"), (72, "08~09 본선자료"), (95, "09 본선")]:
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
cf = pd.read_csv("data/processed/alt_conformal_cell_results.csv")
cov = list(cf.coverage_pct)          # [56.1, 85.9]
wid = list(cf.width_cm)              # [29.6, 50.6]
fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
cats = ["보정 전\n(raw quantile-GBM)", "보정 후\n(CQR conformal)"]
bars = ax.bar(cats, cov, color=[AMBER, NAVY], width=0.5, zorder=3)
ax.axhline(90, color=RED, lw=2.0, ls="--", zorder=2)
ax.text(1.45, 91.2, "목표 90%", fontsize=12, color=RED, ha="right", fontproperties=FP)
for i, v in enumerate(cov):
    ax.text(i, v + 1.8, f"{v:.1f}%", ha="center", fontsize=15, fontproperties=FPB, color=INK, zorder=4)
    ax.text(i, v/2, f"구간 폭\n{wid[i]:.1f} cm", ha="center", va="center", fontsize=11,
            fontproperties=FPM, color="white", zorder=4)
ax.set_ylim(0, 100); ax.set_ylabel("90% 예측구간 실제 커버리지 (%)", fontsize=13, fontproperties=FP)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
for t in ax.get_xticklabels(): t.set_fontsize(12)
ax.set_title("과신하는 예측구간을 목표 수준으로 보정", fontsize=15.5, fontproperties=FPB,
             color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "자료: alt_conformal_cell_results.csv · 셀 단위 평가")
fig.savefig(f"{OUT}/conformal.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved conformal.png")

# ================================================================= 8. 개념도: 활동층·영구동토 (슬라이드 2)
# 기존 outputs/figures/00_concept 그림의 소자 문제 → 큰 폰트로 재작성(대표값 개념도)
z = np.linspace(0, 20, 600)
magt0, grad, A0, damp = -6.0, 0.04, 9.0, 4.3
MAGT = magt0 + grad * z
Aenv = A0 * np.exp(-z / damp)
Tmax, Tmin = MAGT + Aenv, MAGT - Aenv
alt_z = z[np.where(Tmax <= 0)[0][0]]
fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
ax.fill_betweenx([0, alt_z], -16, 8, color="#EAD9B8", alpha=0.55, zorder=0)
ax.fill_betweenx([alt_z, 20], -16, 8, color="#D3E5F0", alpha=0.60, zorder=0)
ax.plot(Tmin, z, color=BLUE, lw=2.6, label="연중 최저 지중온도", zorder=3)
ax.plot(Tmax, z, color=RED, lw=2.6, label="연중 최고 지중온도", zorder=3)
ax.plot(MAGT, z, color=INK, lw=2.2, ls="--", label="연평균 지중온도(MAGT)", zorder=3)
# 0°C 경계선: 진한 남색 실선 + 라인 직접 라벨(범례와 분리, 흰 배경으로 판독 보장)
ax.axvline(0, color=NAVY, lw=2.4, zorder=2)
ax.text(0.4, 0.35, "0°C", fontsize=15, color=NAVY, fontproperties=FPB, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5), zorder=5)
ax.axhline(alt_z, color="#8A6A1F", lw=1.8, ls=":", zorder=2)
ax.annotate("활동층 두께(ALT)\n= 연중 최고온도가 0°C인 깊이",
            xy=(0, alt_z), xytext=(2.4, 6.2), fontsize=12.5, color="#6B4F1D", fontproperties=FP,
            arrowprops=dict(arrowstyle="->", color="#6B4F1D", lw=1.4), zorder=5)
# 활동층 라벨: 좌상단은 최저온도 곡선이 지나므로, 곡선이 없는 밴드 중앙 여백(-5.6~-1)에 배치
ax.text(-5.55, alt_z / 2 - 0.05, "활동층(여름 융해)", fontsize=12.5, color="#7A5A1E",
        va="center", ha="left", fontproperties=FP, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))
ax.text(-15.3, (alt_z + 20) / 2, "영구동토\n(연중 0°C 이하)", fontsize=12.5, color="#1F6699",
        va="center", fontproperties=FP, zorder=5)
ax.set_xlim(-16, 8); ax.set_ylim(0, 20); ax.invert_yaxis()
ax.set_xlabel("지중온도 (°C)", fontsize=14, fontproperties=FP)
ax.set_ylabel("깊이 (m)", fontsize=14, fontproperties=FP)
ax.legend(loc="lower left", prop=FP, fontsize=12, framealpha=0.95, edgecolor="#C9CDD2")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); _style(ax, ts=12.5)
ax.set_title("활동층과 영구동토: 지중온도 포락선과 0°C 경계", fontsize=15.5,
             fontproperties=FPB, color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "대표값으로 그린 개념도 · 포락선 = 깊이별 연중 최고·최저 지중온도")
fig.savefig(f"{OUT}/concept_alt.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved concept_alt.png")

# ================================================================= 9. AOA·DI 진단 (슬라이드 9)
# 기존 outputs/maps/alt_aoa_mask.png(3열 축소로 판독 불가) → 확정 CSV에서 큰 폰트로 재작성
aoa = pd.read_csv("data/processed/alt_aoa_transfer_results.csv")
xd = aoa.di_decile.values
fig, ax = plt.subplots(figsize=(9.0, 4.05), dpi=200)
ax.plot(xd, aoa.rmse_cm, "o-", color=NAVY, lw=2.4, ms=7, label="전이 RMSE (cm)", zorder=3)
ax.set_ylim(13, 29)
ax.set_xlabel("학습환경 비유사도 DI 분위수 (1=유사 → 10=상이)", fontsize=15, fontproperties=FP)
ax.set_ylabel("전이 RMSE (cm)", fontsize=15, color=NAVY, fontproperties=FP)
ax.set_xticks(list(xd))
ax2 = ax.twinx()
ax2.plot(xd, aoa.coverage_pct, "s--", color=TEAL, lw=2.0, ms=7, label="90% 구간 커버리지 (%)", zorder=3)
ax2.axhline(90, color=RED, ls=":", lw=1.6, zorder=2)
ax2.text(5.3, 92.5, "목표 90%", fontsize=11.5, color=RED, ha="left", fontproperties=FP)
ax2.set_ylim(0, 100)
ax2.set_ylabel("90% 구간 커버리지 (%)", fontsize=15, color=TEAL, fontproperties=FP)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color("#C9CDD2")
ax2.tick_params(length=0, labelsize=13)
for t in ax2.get_yticklabels():
    t.set_fontproperties(FPM); t.set_fontsize(13)
# 양 끝 값 직접 라벨(그래프에서 수치 근거를 바로 읽도록)
ax.text(xd[0], aoa.rmse_cm.iloc[0] - 1.1, f"{aoa.rmse_cm.iloc[0]:.1f}", fontsize=12.5,
        color=NAVY, ha="center", va="top", fontproperties=FPB)
ax.text(xd[-1], aoa.rmse_cm.iloc[-1] + 0.6, f"{aoa.rmse_cm.iloc[-1]:.1f}", fontsize=12.5,
        color=NAVY, ha="right", va="bottom", fontproperties=FPB)
ax2.text(xd[0], aoa.coverage_pct.iloc[0] + 3.5, f"{aoa.coverage_pct.iloc[0]:.1f}%", fontsize=12.5,
         color=TEAL, ha="center", va="bottom", fontproperties=FPB)
ax2.text(xd[-1], aoa.coverage_pct.iloc[-1] - 4, f"{aoa.coverage_pct.iloc[-1]:.1f}%", fontsize=12.5,
         color=TEAL, ha="center", va="top", fontproperties=FPB)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", prop=FP, fontsize=12, frameon=False)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); _style(ax, ts=13)
ax.set_title("학습환경에서 멀어질수록(DI↑) 오차가 늘고 커버리지가 떨어진다", fontsize=16,
             fontproperties=FPB, color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "자료: alt_aoa_transfer_results.csv · LORO held-out 점 225,421 · DI = 학습 공변량과의 비유사도")
fig.savefig(f"{OUT}/aoa_di.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved aoa_di.png")

# ================================================================= 10. 국소 데모 재조판 (슬라이드 8)
# 원본(maps/local_demo_alt_field.png)은 GPU diffusion 산출이라 재계산 불가 →
# 지도 래스터(데이터)는 그대로 두고 제목·축·컬러바만 큰 폰트로 다시 조판한다.
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from polar.plotstyle import CMAP as PCMAP

srcA = np.asarray(Image.open("outputs/maps/local_demo_alt_field.png").convert("RGB"))
# 축 스파인 자동탐지로 얻은 패널 내부 픽셀 좌표(원본은 고정 산출물)
PBOX = [(178, 1295), (1725, 2841), (3271, 4388)]; PY0, PY1 = 184, 1453
WL, EL, SL, NL = -153.6, -150.8, 69.7, 70.9      # 스크립트의 지리 범위와 동일
UNC_VMAX = 59.7                                   # 원본 컬러바 눈금 실측(0~50 눈금, 상단 59.7)
specs = [("PolSAR 원자료 (P-band 물리, 30 m)", PCMAP.alt, (20, 70), "ALT (cm)", None),
         ("본 연구 예측 (PolSAR+공변량 앙상블)", PCMAP.alt, (20, 70), "ALT (cm)", None),
         ("Diffusion 90% 예측구간 폭", PCMAP.err, (0, UNC_VMAX), "구간 폭 (cm)", [0, 10, 20, 30, 40, 50])]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=200)
lonfmt = FuncFormatter(lambda v, p: f"{abs(v):.0f}°W" if v < 0 else f"{v:.0f}°E")
latfmt = FuncFormatter(lambda v, p: f"{abs(v):.1f}°N")
for ax, (x0, x1), (ttl, cmap, vlim, lab, cbt) in zip(axes, PBOX, specs):
    ax.imshow(srcA[PY0:PY1, x0:x1], extent=[WL, EL, SL, NL], aspect="auto",
              interpolation="bilinear", zorder=1)
    ax.set_title(ttl, fontsize=17, fontproperties=FPB, color=INK, pad=9)
    ax.set_xlabel("경도", fontsize=14.5, fontproperties=FP)
    ax.set_ylabel("위도", fontsize=14.5, fontproperties=FP)
    ax.set_xticks([-153, -152, -151]); ax.set_yticks([69.8, 70.2, 70.6])
    ax.xaxis.set_major_formatter(lonfmt); ax.yaxis.set_major_formatter(latfmt)
    ax.tick_params(length=0, labelsize=13)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FPM); t.set_fontsize(13)
    for sp in ax.spines.values():
        sp.set_color("#C9CDD2"); sp.set_linewidth(1.0)
    ax.grid(False)
    sm = ScalarMappable(norm=Normalize(*vlim), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
    if cbt is not None:
        cb.set_ticks(cbt)
    cb.set_label(lab, fontsize=16, fontproperties=FP)
    cb.ax.tick_params(labelsize=14, length=2.5)
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(FPM); t.set_fontsize(14)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#444444")
fig.tight_layout()
_src(fig, "자료: maps/local_demo_alt_field.png 동일 데이터 재조판 · 회색 = 모델 유효범위(평탄 툰드라) 밖 · 테두리 점 = 실측")
fig.savefig(f"{OUT}/local_demo.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved local_demo.png")

# ================================================================= 11. MAGT 2m/20m 우측 여백 (슬라이드 11)
# 컬러바 라벨이 그림 우측 가장자리에 붙어 투사 시 잘릴 위험 → 우측 10% 흰 여백 패딩
imm = Image.open("outputs/maps/magt_alaska_2m_20m.png").convert("RGB")
wp, hp = imm.size
canvas = Image.new("RGB", (int(wp * 1.10), hp), (255, 255, 255))
canvas.paste(imm, (0, 0))
canvas.save(f"{OUT}/magt_2m_20m.png")
print("saved magt_2m_20m.png (우측 여백 패딩)")

# ================================================================= 12. 알래스카 ALT 지도 재렌더 (슬라이드 7)
# 동일 데이터·동일 모델(HistGB random_state=0, 결정적)로 폰트만 확대해 재렌더.
# 프로젝트 산출물(outputs/maps)은 건드리지 않고 deck/assets/mid/alt_map.png 로 저장.
import runpy
import matplotlib.axes as _maxes
from polar import plotstyle as _ps
from polar import outputs as _po

_use0, _cb0, _sg0, _mp0, _lg0 = _ps.use_polar, _ps.add_cbar, _ps.style_geo, _po.mappath, _maxes.Axes.legend

def _sanitize(txt):
    return (txt.replace(" — ", " · ").replace("—", "·").replace("활성층", "활동층")) if txt else txt

def _use_big():
    p = _use0()
    p.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 19,
                       "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16})
    return p

def _cb_big(fig, m, ax, label, **kw):
    kw.setdefault("shrink", 0.85); kw.setdefault("pad", 0.02)
    cb = fig.colorbar(m, ax=ax, **kw)
    cb.set_label(_sanitize(label), fontsize=18)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#444444")
    cb.ax.tick_params(labelsize=16, length=3)
    return cb

def _sg_big(ax, title=None, xlabel="경도 (°E)", ylabel="위도 (°N)"):
    return _sg0(ax, title=_sanitize(title), xlabel=xlabel, ylabel=ylabel)

def _mp_big(name, ext="png"):
    return f"{OUT}/alt_map.{ext}"

def _lg_big(self, *a, **k):
    k["fontsize"] = 16
    leg = _lg0(self, *a, **k)
    for t in leg.get_texts():
        t.set_text(_sanitize(t.get_text()))
    return leg

_ps.use_polar, _ps.add_cbar, _ps.style_geo, _po.mappath = _use_big, _cb_big, _sg_big, _mp_big
_maxes.Axes.legend = _lg_big
try:
    runpy.run_path("scripts/4_visualization/map_alt_alaska.py", run_name="__main__")
    print("saved alt_map.png (대형 폰트 재렌더)")
finally:
    _ps.use_polar, _ps.add_cbar, _ps.style_geo, _po.mappath = _use0, _cb0, _sg0, _mp0
    _maxes.Axes.legend = _lg0
