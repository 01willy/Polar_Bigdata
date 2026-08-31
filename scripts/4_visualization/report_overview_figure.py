"""보고서 1쪽 개요 도판 — 기존 연구의 한계 · 개선 방안 · 기대 효과.

구조
    목표 띠(한 줄) 아래 3열. 각 열은 머리 띠 + 실제 결과 자료로 그린 미니 패널 2장 +
    한 줄 요약이다. 그림 안에는 문장을 넣지 않고 명사형 문구만 쓴다.
        1열 기존 연구의 한계 : (1) 라벨 희소         (2) 지역 간 전이 오차
        2열 개선 방안        : (3) 물리 유사라벨 증강 (4) 물리 잔차·독립 관측 결합
        3열 기대 효과        : (5) 광역 연속장 산출   (6) 예측 신뢰 범위

미니 패널의 자료 출처 (전부 보고서 본문·표와 동일)
    (1) 예측 대상 격자 892,865셀 대 실측 라벨 13,606셀 (0.02도 격자, 4.1절·표 2)
    (2) 표 7 전이 열 — 물리식 21.3 · 크리깅 29.4 · 신경망 34.2 · 부스팅 38.4 · 역거리 50.9
    (3) data/processed/s3_aug_curve_results.csv — 증강 비율 r 에 따른 RMSE(레나델타)
    (4) 표 8 정보 없음 열 — 물리식 24.11 → 물리식+위성 23.27 → +잔차 22.92
    (5) outputs/maps/alt_prediction_hires.png — 산출된 광역 ALT 예측장
    (6) outputs/maps/alt_uncertainty_aoa_map.png (a) — 보정 예측구간 폭·적용가능 영역

산출
    outputs/figures/00_overview/report_overview.{png,pdf}
    outputs/figures/00_overview/panel_{1..6}.png    — 편집용 PPTX가 재사용하는 개별 패널
    폭은 보고서 삽입 크기(1.0 x textwidth = 176 mm)와 같게 잡고 tight 크롭을 쓰지
    않으므로 인쇄 배율이 1.0이다. 즉 스크립트의 fontsize(pt)가 곧 인쇄 pt다.

실행
    PYTHONPATH=src python scripts/4_visualization/report_overview_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, Circle       # noqa: E402
import matplotlib.image as mpimg                            # noqa: E402

plt = use_polar()
plt.rcParams["savefig.bbox"] = None        # 인쇄 배율 1.0 확정

# 편집용 PPTX와 같은 글씨체를 쓴다(Pretendard, 전부 굵게 · 목표 문구만 ExtraBold).
import matplotlib.font_manager as _fm                        # noqa: E402
for _f in sorted(Path.home().joinpath(".fonts").glob("Pretendard-*.otf")):
    _fm.fontManager.addfont(str(_f))
FONT = "Pretendard"
FONT_XB = "Pretendard ExtraBold"
_avail = {f.name for f in _fm.fontManager.ttflist}
if FONT in _avail:
    plt.rcParams["font.family"] = FONT
if FONT_XB not in _avail:
    FONT_XB = FONT
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

OUT = ROOT / "outputs" / "figures" / "00_overview"
OUT.mkdir(parents=True, exist_ok=True)
NAME = "report_overview"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"

# ---------------------------------------------------------------- 치수(mm)
W = 176.0
M = 0.9
GOAL_H = 8.6
HEAD_H = 6.6
CARD_H = 79.0
GAP = 2.6
COL_GAP = 2.2
H = M * 2 + GOAL_H + GAP + HEAD_H + 1.6 + CARD_H
COL_W = (W - 2 * M - 2 * COL_GAP) / 3.0

PAD_X = 3.0
LAB_H = 5.2                # 소제목 줄
CAP_H = 4.6                # 아래 문구(1행)
CAP_PAD = 2.4              # 문구 좌측 여백(폭 확보용)
LAB_GAP = 1.6              # 소제목과 패널 사이
CAP_GAP = 1.8              # 패널과 문구 사이
PANEL_H = 19.5             # 기본 패널 높이(그래프)
TICK_H = 5.0               # 기본 축 라벨 자리
MAP_H = 24.6               # 지도 썸네일 높이
MAP_TICK = 0.6

# ---------------------------------------------------------------- 색
NAVY, MID, STEEL = "#12365c", "#2f6f9f", "#5f7d95"
INK, SUB, MUTE = "#141a20", "#2b3a46", "#66727e"
HEAD_BG = ["#5d666f", NAVY, MID]
CARD_BG = ["#f5f6f7", "#eef4f9", "#eff5f9"]
CARD_EC = ["#c8cdd2", "#b9cbdb", "#bdd0dd"]
RULE = ["#d5d9dd", "#c8d8e5", "#cbd9e5"]
GOAL_BG, GOAL_EC = "#e9eff5", "#b9c4ce"
GREY_LT, GREY = "#dde2e7", "#a4b2bf"
WARM = "#8a7f5c"           # 부정확 물리(대조) — 붉은 계열 회피

# ---------------------------------------------------------------- 글자(= 인쇄 pt)
FS_GOAL_LAB, FS_GOAL = 8.6, 10.2
FS_HEAD = 10.4
FS_PLAB, FS_CAP, FS_TICK, FS_BADGE = 8.8, 6.4, 6.9, 7.4

GOAL = "희소 라벨 조건에서의 광역 활동층 두께 예측 및 미관측 영역으로의 확장"
HEADS = ["기존 연구 한계점", "개선 방안", "기대 효과"]


# ---------------------------------------------------------------- 미니 패널
def _bare(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s, on in (("left", left), ("bottom", bottom)):
        ax.spines[s].set_visible(on)
        if on:
            ax.spines[s].set_linewidth(0.5); ax.spines[s].set_color("#b3bcc5")
    ax.tick_params(length=1.4, pad=1.2, labelsize=FS_TICK, colors=MUTE)


def _frame(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#b3bcc5")


def panel_1(ax):
    """라벨 희소 — 예측 대상 격자 대비 실측 라벨이 1.5%에 그친다."""
    n_grid, n_lab = 892_865, 13_606
    ax.barh([1], [n_grid], height=0.5, color=GREY_LT, edgecolor="none", zorder=3)
    ax.barh([0], [n_lab], height=0.5, color=NAVY, edgecolor="none", zorder=4)
    ax.set_xscale("log"); ax.set_xlim(3e3, 6e6)
    ax.set_ylim(-0.75, 1.6)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["예측 대상 격자", "실측 라벨"], fontsize=FS_TICK)
    ax.set_xticks([])
    ax.text(n_grid * 1.4, 1, f"{n_grid:,}", va="center", ha="left",
            fontsize=FS_TICK, color=MUTE)
    ax.text(n_lab * 1.4, 0, f"{n_lab:,}", va="center", ha="left",
            fontsize=FS_TICK, color=NAVY, weight="bold")
    ax.text(3.4e3, -0.62, "라벨 = 대상 격자의 1.5%", va="center", ha="left",
            fontsize=FS_TICK, color=NAVY, weight="bold")
    _bare(ax, left=False, bottom=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()


panel_1.left_pad = 15.0


def panel_2(ax):
    """지역 간 전이 오차 — 물리식만 낮고 순수 데이터 방법은 무너진다(표 7)."""
    lab = ["역거리", "부스팅", "신경망", "크리깅", "물리식"]
    val = [50.9, 38.4, 34.2, 29.4, 21.3]
    col = [GREY, GREY, GREY, GREY, NAVY]
    ax.barh(range(5), val, height=0.58, color=col, edgecolor="none", zorder=3)
    ax.axvline(21.3, color=NAVY, lw=0.7, ls=(0, (3, 2)), zorder=4)
    for i, v in enumerate(val):
        ax.text(v + 1.4, i, f"{v:.0f}", va="center", ha="left", fontsize=FS_TICK,
                color=NAVY if i == 4 else MUTE, weight="bold" if i == 4 else "normal")
    ax.set_yticks(range(5)); ax.set_yticklabels(lab, fontsize=FS_TICK, color=MUTE)
    ax.set_xticks([]); ax.set_xlim(0, 63)
    ax.set_xlabel("전이 RMSE (cm)", fontsize=FS_TICK, color=MUTE, labelpad=1.0)
    _bare(ax, left=False, bottom=False)
    ax.tick_params(axis="y", length=0)


panel_2.left_pad = 9.6


def panel_3(ax):
    """물리 유사라벨 증강 — 정확한 물리식만 오차를 낮춘다."""
    d = pd.read_csv(PROC / "s3_aug_curve_results.csv")
    d = d[(d.target == "Lena") & (d.model == "catboost")]
    n = 0
    for phys, col, lab in (("stefan", NAVY, "정확한 물리식"), ("kudryavtsev", WARM, "부정확한 물리식")):
        g = d[d.phys == phys].groupby("r")["rmse_cm"].mean()
        if g.empty:
            continue
        n = len(g)
        ax.plot(np.arange(n), g.values, "-o", ms=2.2, lw=1.3, color=col,
                mfc="white", mew=1.0, zorder=3, label=lab)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - (y1 - y0) * 0.04, y1 + (y1 - y0) * 0.42)   # 범례 자리
    leg = ax.legend(fontsize=FS_TICK, frameon=False, loc="upper left",
                    handlelength=1.2, handletextpad=0.4, borderpad=0.0,
                    labelspacing=0.15, ncol=2, columnspacing=0.9,
                    bbox_to_anchor=(-0.02, 1.06))
    for t, c in zip(leg.get_texts(), [NAVY, WARM]):
        t.set_color(c)
    ax.set_xticks([0, n - 1])
    ax.set_xticklabels(["증강 없음", "r = 10"], fontsize=FS_TICK)
    ax.set_ylabel("RMSE (cm)", fontsize=FS_TICK, color=MUTE, labelpad=2.0)
    ax.set_xlim(-0.5, n - 0.2)
    _bare(ax)


panel_3.left_pad = 7.4


def panel_4(ax):
    """물리 잔차·독립 관측 결합 — 전이 오차가 단계적으로 낮아진다(표 8, 정보 없음)."""
    lab = ["물리식", "＋위성", "＋잔차"]
    val = [24.11, 23.27, 22.92]
    col = [STEEL, MID, NAVY]
    ax.bar(range(3), val, width=0.5, color=col, edgecolor="none", zorder=3)
    for i, v in enumerate(val):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_TICK,
                color=col[i], weight="bold" if i == 2 else "normal")
    ax.annotate("", xy=(2.62, 22.92), xytext=(2.62, 24.11),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(2.74, 23.52, "-1.19", ha="left", va="center", fontsize=FS_TICK,
            color=NAVY, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab, fontsize=FS_TICK, color=MUTE)
    ax.set_ylim(22.45, 24.62); ax.set_xlim(-0.62, 3.42)
    ax.set_yticks([23, 24]); ax.set_yticklabels(["23", "24"], fontsize=FS_TICK)
    ax.set_ylabel("전이 RMSE (cm)", fontsize=FS_TICK, color=MUTE, labelpad=2.0)
    _bare(ax)
    ax.tick_params(axis="x", length=0)


panel_4.left_pad = 7.8


def _thumb(ax, src, box):
    im = mpimg.imread(src)
    h, w = im.shape[:2]
    x0, x1, y0, y1 = box
    ax.imshow(im[int(h * y0):int(h * y1), int(w * x0):int(w * x1)],
              interpolation="bilinear", zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    _frame(ax)


def panel_5(ax):
    """광역 연속장 산출 — 미관측 영역까지 확장된 ALT 예측장(그림 5)."""
    _thumb(ax, MAPS / "alt_prediction_hires.png", (0.115, 0.995, 0.015, 0.945))


panel_5.panel_h = MAP_H
panel_5.tick_h = MAP_TICK


def panel_6(ax):
    """예측 신뢰 범위 — 보정 예측구간 폭과 적용가능 영역 표기(그림 13(a))."""
    # 잘라낼 영역은 그림 13 의 패널 (a)와 그 컬러바이다. 원 그림을 인쇄 1:1 로 다시
    # 만들면서 축 배치가 바뀌었으므로 좌표도 그에 맞춰 잡았다.
    _thumb(ax, MAPS / "alt_uncertainty_aoa_map.png", (0.010, 0.502, 0.004, 0.508))


panel_6.panel_h = MAP_H
panel_6.tick_h = MAP_TICK


COLS = [
    [(panel_1, "라벨 희소", "· 예측 대상 격자의 1.5%만 실측 · 74개 0.5° 블록에 집중"),
     (panel_2, "지역 간 전이 오차", "· 미관측 지역에서 데이터 기반 방법론 오차 급증")],
    [(panel_3, "물리경험식 기반 라벨 증강", "· 물리식 정확도별 증강 효과 비교 · 증강 비율 반응 분석"),
     (panel_4, "물리 잔차 · 독립 관측 결합", "· 물리식 단독 대비 전이 오차 단계적 감소안 개발")],
    [(panel_5, "광역 연속장 산출", "· 미관측 영역까지 활동층 두께 예측장 확장 가능"),
     (panel_6, "예측 신뢰 범위 제시", "· 보정 예측구간 폭과 전이 적용가능 영역 표기")],
]
KEYS = ["1", "2", "3", "4", "5", "6"]


def build(fig, bg):
    def box(x, y, w, h, fc, ec, lw=0.7, r=1.2, z=2):
        bg.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
                     edgecolor=ec, linewidth=lw, zorder=z))

    y = H - M - GOAL_H
    box(M, y, W - 2 * M, GOAL_H, GOAL_BG, GOAL_EC)
    box(M + 1.6, y + GOAL_H / 2 - 2.9, 15.6, 5.8, NAVY, "none", r=1.0, z=3)
    bg.text(M + 9.4, y + GOAL_H / 2, "목표", ha="center", va="center",
            fontsize=FS_GOAL_LAB, color="white", weight="bold", zorder=4)
    bg.text(M + 20.6, y + GOAL_H / 2, GOAL, ha="left", va="center",
            fontsize=FS_GOAL, color=INK, fontfamily=FONT_XB, weight="bold", zorder=4)

    y_head = y - GAP - HEAD_H
    axes, k = [], 0
    for c in range(3):
        x0 = M + c * (COL_W + COL_GAP)
        box(x0, y_head, COL_W, HEAD_H, HEAD_BG[c], "none", z=2)
        bg.text(x0 + COL_W / 2, y_head + HEAD_H / 2, HEADS[c], ha="center",
                va="center", fontsize=FS_HEAD, color="white", weight="bold", zorder=3)

        y_card = y_head - 1.6 - CARD_H
        box(x0, y_card, COL_W, CARD_H, CARD_BG[c], CARD_EC[c], z=2)
        slot = CARD_H / 2.0
        for r_, (fn, plab, cap) in enumerate(COLS[c]):
            k += 1
            ph = getattr(fn, "panel_h", PANEL_H)
            th = getattr(fn, "tick_h", TICK_H)
            top = y_card + CARD_H - r_ * slot - 0.9
            bg.text(x0 + PAD_X, top - LAB_H / 2, plab, ha="left", va="center",
                    fontsize=FS_PLAB, color=INK, weight="bold", zorder=4)

            lp = getattr(fn, "left_pad", 0.0)
            ay = top - LAB_H - LAB_GAP - ph
            ax = fig.add_axes([(x0 + PAD_X + lp) / W, ay / H,
                               (COL_W - 2 * PAD_X - lp) / W, ph / H], zorder=4)
            ax.set_facecolor("none")
            fn(ax)
            axes.append(ax)
            bg.text(x0 + CAP_PAD, ay - th - CAP_GAP, cap, ha="left", va="top",
                    fontsize=FS_CAP, color=SUB, weight="bold", zorder=4)
    return axes


def main():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
    axes = build(fig, bg)

    allt = list(bg.texts)
    for a in axes:
        allt += list(a.texts) + a.get_xticklabels() + a.get_yticklabels()
    mn = min(t.get_fontsize() for t in allt if t.get_text())
    print(f"[배치] {W:.1f} x {H:.1f} mm (인쇄 1:1) · 열 {COL_W:.1f} mm · 카드 {CARD_H:.1f} mm")
    print(f"[글자] 최소 {mn:.1f} pt · 목표 {FS_GOAL} · 열머리 {FS_HEAD} · "
          f"소제목 {FS_PLAB} · 문구 {FS_CAP}")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=300 if ext == "png" else None)
        print("saved", (OUT / f"{NAME}.{ext}").relative_to(ROOT))
    plt.close(fig)

    fns = [fn for col in COLS for fn, _, _ in col]
    for key, fn in zip(KEYS, fns):
        lp = getattr(fn, "left_pad", 0.0)
        f = plt.figure(figsize=((COL_W - 2 * PAD_X) / 25.4, PANEL_H / 25.4))
        l0 = 0.03 + lp / (COL_W - 2 * PAD_X)
        a = f.add_axes([l0, 0.24, 0.965 - l0, 0.72]); a.set_facecolor("none")
        fn(a)
        f.savefig(OUT / f"panel_{key}.png", dpi=300, transparent=True)
        plt.close(f)
    print("saved", ", ".join(f"panel_{k}.png" for k in KEYS))


if __name__ == "__main__":
    main()
