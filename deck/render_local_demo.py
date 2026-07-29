# -*- coding: utf-8 -*-
"""국소 데모(슬라이드 8) 재조판 그림 단일 렌더러.

- deck/mk_midreport_figs.py 의 section 10을 분리해 이 함수만 독립 호출 가능하게 한다.
- 원본(outputs/maps/local_demo_alt_field.png)은 GPU diffusion 산출이라 재계산 불가 →
  지도 래스터(데이터)는 그대로 두고 축·컬러바만 큰 폰트로 재조판한다.
- 산출:
    deck/assets/mid/local_demo.png            (덱 슬라이드용, save_deck=True 일 때만)
    outputs/maps/local_demo_restyled.{png,pdf} (보고서용: 내부 제목·출처각주 제거 + km 스케일바)

단독 실행 시(python3 deck/render_local_demo.py) 보고서용 재조판본만 재생성한다.
프로젝트 루트에서 실행할 것. 필요 시 PYTHONPATH=src.
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from PIL import Image
from polar.plotstyle import CMAP as PCMAP

# ---------- Pretendard 폰트 ----------
for _f in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    try:
        fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception:
        pass
FP = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none"})

INK = "#18181B"
MUTE = "#8A9096"
PAPER = "#FFFFFF"

# 패널 내부 픽셀 좌표 및 지리 범위(원본은 고정 산출물)
PBOX = [(178, 1295), (1725, 2841), (3271, 4388)]
PY0, PY1 = 184, 1453
WL, EL, SL, NL = -153.6, -150.8, 69.7, 70.9   # 스크립트의 지리 범위와 동일
UNC_VMAX = 59.7                               # 원본 컬러바 눈금 실측(0~50 눈금, 상단 59.7)


def _add_scalebar(ax, km=20):
    """위도 기반 km 환산 스케일바(패널 좌하단). 배경 무관 판독 위해 흰색 스트로크."""
    latc = 0.5 * (SL + NL)
    km_per_deg = 111.320 * np.cos(np.deg2rad(latc))   # 경도 1도당 동서 거리(km)
    dlon = km / km_per_deg
    x0 = WL + 0.06 * (EL - WL)
    y0 = SL + 0.09 * (NL - SL)
    tick = 0.02 * (NL - SL)
    halo = [pe.withStroke(linewidth=4.0, foreground="white")]
    ax.plot([x0, x0 + dlon], [y0, y0], color=INK, lw=3.2,
            solid_capstyle="butt", zorder=7, path_effects=halo)
    for xx in (x0, x0 + dlon):
        ax.plot([xx, xx], [y0, y0 + tick], color=INK, lw=3.2, zorder=7,
                path_effects=halo)
    ax.text(x0 + dlon / 2, y0 + 1.6 * tick, f"{km} km", ha="center", va="bottom",
            fontsize=16, fontproperties=FP, color=INK, zorder=7, path_effects=halo)


def render_local_demo(out_dir="deck/assets/mid", maps_dir="outputs/maps",
                      save_deck=False):
    os.makedirs(maps_dir, exist_ok=True)
    srcA = np.asarray(
        Image.open(os.path.join(maps_dir, "local_demo_alt_field.png")).convert("RGB"))

    # 패널 제목은 중립 기술 표현만 사용(주장형 서술 금지). obs=True 인 패널에만
    # 실측 산점도(테두리 점)가 래스터에 포함되어 있어 프록시 범례를 단다.
    specs = [
        ("(a) PolSAR 원자료 (30 m)", PCMAP.alt, (20, 70), "ALT (cm)", None, True),
        ("(b) 앙상블 예측 ALT", PCMAP.alt, (20, 70), "ALT (cm)", None, True),
        ("(c) 90% 예측구간 폭", PCMAP.err, (0, UNC_VMAX), "구간 폭 (cm)",
         [0, 10, 20, 30, 40, 50], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), dpi=200)
    lonfmt = FuncFormatter(lambda v, p: f"{abs(v):.0f}°W" if v < 0 else f"{v:.0f}°E")
    latfmt = FuncFormatter(lambda v, p: f"{abs(v):.1f}°N")
    for ax, (x0, x1), (ttl, cmap, vlim, lab, cbt, obs) in zip(axes, PBOX, specs):
        ax.imshow(srcA[PY0:PY1, x0:x1], extent=[WL, EL, SL, NL], aspect="auto",
                  interpolation="bilinear", zorder=1)
        ax.set_title(ttl, fontsize=19, fontproperties=FPB, color=INK, pad=10)
        ax.set_xlabel("경도 (°)", fontsize=18.5, fontproperties=FP)
        ax.set_ylabel("위도 (°)", fontsize=18.5, fontproperties=FP)
        ax.set_xticks([-153, -152, -151])
        ax.set_yticks([69.8, 70.2, 70.6])
        ax.xaxis.set_major_formatter(lonfmt)
        ax.yaxis.set_major_formatter(latfmt)
        ax.tick_params(length=0, labelsize=18)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontproperties(FPM)
            t.set_fontsize(18)
        for sp in ax.spines.values():
            sp.set_color("#C9CDD2")
            sp.set_linewidth(1.0)
        ax.grid(False)
        if obs:
            hdl = Line2D([], [], linestyle="none", marker="o",
                         markerfacecolor="none", markeredgecolor=INK,
                         markeredgewidth=1.8, markersize=10, label="실측 지점")
            leg = ax.legend(handles=[hdl], loc="upper right", fontsize=17,
                            frameon=True, framealpha=0.95, edgecolor="#C9CDD2",
                            borderpad=0.5, handletextpad=0.4, borderaxespad=0.4)
            for t in leg.get_texts():
                t.set_fontproperties(FP)
        sm = ScalarMappable(norm=Normalize(*vlim), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
        if cbt is not None:
            cb.set_ticks(cbt)
        cb.set_label(lab, fontsize=19, fontproperties=FP)
        cb.ax.tick_params(labelsize=18, length=2.5)
        for t in cb.ax.get_yticklabels():
            t.set_fontproperties(FPM)
            t.set_fontsize(18)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#444444")

    # 덱 슬라이드용(local_demo.png): 상단 제목 + 좌하단 출처각주 유지.
    if save_deck:
        os.makedirs(out_dir, exist_ok=True)
        sup = fig.suptitle("고정밀 국소 데모: 알래스카 북사면 평탄 툰드라",
                           fontsize=19, fontproperties=FPB, color=INK, y=1.06)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        foot = fig.text(
            0.01, -0.015,
            "자료: maps/local_demo_alt_field.png 동일 데이터 재조판 · "
            "회색 = 모델 유효범위(평탄 툰드라) 밖 · 테두리 점 = 실측",
            fontsize=8.5, color=MUTE, fontproperties=FPM, ha="left", va="top")
        fig.savefig(os.path.join(out_dir, "local_demo.png"), dpi=200,
                    facecolor=PAPER, bbox_inches="tight")
        sup.remove()
        foot.remove()

    # 보고서용 재조판본: 내부 suptitle·출처각주 제거(캡션이 대신) + km 스케일바.
    for ax in axes:
        _add_scalebar(ax, km=20)
    fig.tight_layout()
    fig.savefig(os.path.join(maps_dir, "local_demo_restyled.png"), dpi=300,
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(maps_dir, "local_demo_restyled.pdf"),
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("saved outputs/maps/local_demo_restyled.{png,pdf}"
          + (" + deck/assets/mid/local_demo.png" if save_deck else ""))


if __name__ == "__main__":
    # 단독 실행: 보고서용 재조판본만 재생성(다른 덱 그림 불변).
    render_local_demo(save_deck=False)
