"""XK 격자 크기별 오차 그림(개정 2, 2026-10-05 검토 반영). 입력 data/processed/xbatch/XK_support_scale/xk_support_scale_v1.csv.

두 행 × 세 지역(알래스카, 레나 델타, 캐나다). 가로축은 같은 간격의 범주 위치(1 km 셀, 0.05°, 0.1°, 0.25°)이고 눈금 아래 회색 수는 격자 수다.
  위 행: 셀 가중 RMSE(cm, 로그 축 8–100, 행 안 공유). 재보정 Stefan 과 잔차 ML 은 블록 재표집 95 % CI 띠(제 선 색, 낮은 불투명도), 제품은 선과 점.
         셀 집합은 그 지역의 모든 표시 방법 값이 있는 셀(알래스카 Yi-Kimball 포함 'with_yk', 레나·캐나다 'common').
  아래 행: RMSE 차(재보정 Stefan − 잔차 ML, cm)와 95 % CI. 셀 가중(채운 점, 굵은 막대)과 블록 등가중(빈 점, 가는 막대), 0 선. 셀 집합은
         두 모형 값이 있는 모든 라벨 셀('models'). 행 안 y 공유.
  마스크 없음 판이다. Wei 학습 지점 5 km 마스크 판은 원천 표에 있다.
논문판 170 mm, Liberation Sans 7 pt, 패널 문자와 지역 이름. 슬라이드판 12.0 in 폭, 5.2 in 높이 이하, Pretendard.
산출 outputs/figures/paper/v3_restructure/maps/XK_support_scale.{pdf,svg,png}, XK_support_scale_source.csv, deck/assets/paper_report/maps/XK_support_scale_slide.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
import matplotlib.ticker as mticker                                                                     # noqa: E402
from matplotlib.lines import Line2D                                                                     # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

TAB = ROOT / "data" / "processed" / "xbatch" / "XK_support_scale" / "xk_support_scale_v1.csv"
OUT_PAPER, OUT_SLIDE = MV.OUT_PAPER, MV.OUT_SLIDE
REGIONS = (("Alaska", "with_yk"), ("Lena", "common"), ("Canada", "common"))
SUPS = ("1km", "0.05deg", "0.1deg", "0.25deg")
SUP_LAB = {"1km": "1 km", "0.05deg": "0.05°", "0.1deg": "0.1°", "0.25deg": "0.25°"}
STY = {
    "p1": dict(label="Recalibrated Stefan", color=S.METHOD["recalibrated_stefan"]["color"], ls="-", band=True),
    "r1": dict(label="Anchor + residual ML", color=S.METHOD["anchor_residual"]["color"], ls="-", band=True),
    "cci4": dict(label="CCI v4", color=S.PRODUCT["CCI"]["color"], ls=(0, (4, 2)), band=False),      # v4 토큰 color.products
    "cci5": dict(label="CCI v5", color=S.PRODUCT["CCI"]["color"], ls="-", band=False),
    "wei": dict(label="Wei 2026", color=S.PRODUCT["Wei"]["color"], ls="-", band=False),
    "yk": dict(label="Yi-Kimball", color=S.PRODUCT["YK"]["color"], ls=(0, (5, 1.5, 1.5, 1.5)), band=False),
}
D_COLOR = S.METHOD["anchor_residual"]["color"]      # d–f 의 차(Stefan − 잔차 ML)는 잔차 ML 의 이득이라 제안 방법 색(v4: 검정만 쓰는 계열 없음)
BAND_ALPHA = 0.15
GRAY = "#6b6b6b"


def n_under_ticks(ax, ns, M):
    """눈금 라벨 아래 회색 격자 수."""
    off = -(M.tick_len + 2.0 + M.fs + 1.6) if M.paper else -(M.tick_len + 3.0 + M.fs + 3.0)
    for x, n in enumerate(ns):
        t = ax.annotate(S.fmt_int(n), xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, off), textcoords="offset points",
                        ha="center", va="top", fontsize=M.fs if M.paper else 12.0, color=GRAY)
        t.set_gid("sizekey")


def setup_x(ax, M):
    ax.set_xticks(range(len(SUPS)))
    ax.set_xticklabels([SUP_LAB[s] for s in SUPS])
    ax.set_xlim(-0.45, len(SUPS) - 0.55)
    ax.tick_params(axis="both", labelsize=M.fs, width=M.lw, length=M.tick_len)


def top_panel(ax, d, M):
    for m, st in STY.items():
        r = d[d.method == m].set_index("support").reindex(SUPS)
        if r.rmse_cw.isna().all():
            continue
        x = np.arange(len(SUPS))
        lw = (S.LW["main"] if M.paper else 3.0) if st["band"] else (S.LW["aux"] if M.paper else 2.0)
        ms = (S.MS["main"] if M.paper else 8.0) if st["band"] else (2.5 if M.paper else 6.0)
        if st["band"]:
            ax.fill_between(x, r.rmse_cw_lo.values, r.rmse_cw_hi.values, color=st["color"], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(x, r.rmse_cw.values, color=st["color"], ls=st["ls"], lw=lw, marker="o", ms=ms, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(8, 100)
    ax.yaxis.set_major_locator(mticker.FixedLocator([10, 20, 40, 80]))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(["10", "20", "40", "80"]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    setup_x(ax, M)


def bottom_panel(ax, d, M, ylim):
    r = d[d.method == "p1"].set_index("support").reindex(SUPS)
    x = np.arange(len(SUPS))
    ax.axhline(0.0, color=S.INK_AUX, lw=M.lw, zorder=1)
    off = 0.12
    ax.vlines(x - off, r.d_rmse_vs_r1_cw_lo.values, r.d_rmse_vs_r1_cw_hi.values, color=D_COLOR, lw=2.0 if M.paper else 4.0, zorder=3)
    ax.plot(x - off, r.d_rmse_vs_r1_cw.values, ls="none", marker="o", ms=S.MS["main"] if M.paper else 8.0, color=D_COLOR, zorder=4)
    ax.vlines(x + off, r.d_rmse_vs_r1_be_lo.values, r.d_rmse_vs_r1_be_hi.values, color=D_COLOR, lw=1.0 if M.paper else 2.0, zorder=3)
    ax.plot(x + off, r.d_rmse_vs_r1_be.values, ls="none", marker="o", ms=S.MS["main"] if M.paper else 8.0, mfc="#ffffff", mec=D_COLOR,
            mew=1.0 if M.paper else 2.0, zorder=4)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: S.fmt_num(v, 0)))
    setup_x(ax, M)


def build(medium):
    t = pd.read_csv(TAB)
    M = MV.Med(medium)
    MV.use_medium(medium)
    top = {r: t[(t.region == r) & (t.cell_set == cs) & (t["mask"] == "none")] for r, cs in REGIONS}
    bot = {r: t[(t.region == r) & (t.cell_set == "models") & (t["mask"] == "none")] for r, _ in REGIONS}
    b = pd.concat([bot[r][bot[r].method == "p1"] for r, _ in REGIONS])
    lo = float(np.floor(min(b.d_rmse_vs_r1_cw_lo.min(), b.d_rmse_vs_r1_be_lo.min()) - 0.4))
    hi = float(np.ceil(max(b.d_rmse_vs_r1_cw_hi.max(), b.d_rmse_vs_r1_be_hi.max()) + 0.4))
    if M.paper:
        W, LM, GAP, RM = 170.0, 13.0, 3.0, 1.0
        LEG, HEAD, PH1, TCK, ROWG, PH2 = 5.0, 5.0, 40.0, 8.0, 5.0, 32.0
        fig = S.fig_mm(W, LEG + HEAD + PH1 + TCK + ROWG + PH2 + TCK + 0.5)
    else:
        W, LM, GAP, RM = MV.SLIDE_W_MM, 24.0, 7.0, 1.5
        LEG, HEAD, PH1, TCK, ROWG, PH2 = 9.0, 9.0, 36.0, 14.0, 8.0, 40.0
        H = LEG + HEAD + PH1 + TCK + ROWG + PH2 + TCK + 1.0
        assert H <= MV.SLIDE_H_MAX_MM, H
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    pw = (W - LM - 2 * GAP - RM) / 3
    xs = [LM + j * (pw + GAP) for j in range(3)]
    y1 = LEG + HEAD
    y2 = y1 + PH1 + TCK + ROWG
    for j, (r, _) in enumerate(REGIONS):
        ax = S.axes_mm(fig, xs[j], y1, pw, PH1)
        top_panel(ax, top[r], M)
        nt = top[r][top[r].method == "p1"].set_index("support").reindex(SUPS).n_units.values
        n_under_ticks(ax, nt, M)
        if j == 0:
            ax.set_ylabel("RMSE (cm)", fontsize=M.fs_lab)
        else:
            ax.set_yticklabels([])
        MV.letter(fig, xs[j], y1 - (1.2 if M.paper else 2.0), "abc"[j], M, S.REGION_NAME.get(r, r))
        ax2 = S.axes_mm(fig, xs[j], y2, pw, PH2)
        bottom_panel(ax2, bot[r], M, (lo, hi))
        nb = bot[r][bot[r].method == "p1"].set_index("support").reindex(SUPS).n_units.values
        n_under_ticks(ax2, nb, M)
        if j == 0:
            ax2.set_ylabel(f"Stefan {MV.MINUS} residual ML (cm)" if M.paper else "\u0394RMSE (cm)", fontsize=M.fs_lab)
        else:
            ax2.set_yticklabels([])
        MV.letter(fig, xs[j], y2 - (1.2 if M.paper else 2.0), "def"[j], M, None if (M.paper or j > 0) else f"Stefan {MV.MINUS} residual ML")
        if j == 0:
            hw = [Line2D([], [], color=D_COLOR, lw=2.0 if M.paper else 4.0, marker="o", ms=S.MS["main"] if M.paper else 8.0, label="Cell-weighted"),
                  Line2D([], [], color=D_COLOR, lw=1.0 if M.paper else 2.0, marker="o", ms=S.MS["main"] if M.paper else 8.0, mfc="#ffffff",
                         mec=D_COLOR, mew=1.0 if M.paper else 2.0, label="Block-equal")]
            ax2.legend(handles=hw, loc="lower right", frameon=False, fontsize=M.fs if M.paper else 14.0, handlelength=1.4, borderaxespad=0.3,
                       labelspacing=0.3)
    hs = [Line2D([], [], color=st["color"], ls=st["ls"], lw=(S.LW["main"] if M.paper else 3.0) if st["band"] else (S.LW["aux"] if M.paper else 2.0),
                 marker="o", ms=(S.MS["main"] if M.paper else 8.0) if st["band"] else (2.5 if M.paper else 6.0), label=st["label"]) for st in STY.values()]
    Wf, Hf = fig.get_size_inches() * 25.4
    fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False, fontsize=M.fs if M.paper else 14.0, bbox_to_anchor=(0.5, 1.0 - 0.3 / Hf),
               handlelength=2.6 if M.paper else 2.4, columnspacing=1.4 if M.paper else 1.2, handletextpad=0.5)
    return fig, dict(ylim_bottom=[lo, hi], size_mm=[round(Wf, 1), round(Hf, 1)])


def main():
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    OUT_SLIDE.mkdir(parents=True, exist_ok=True)
    fig, rec = build("paper")
    S.save_fig(fig, "XK_support_scale", OUT_PAPER, formats=("pdf", "svg", "png"))
    au = S.audit_v3(fig)
    print(f"[audit] sizes {sorted(au['sizes'])} chars {au['chars']} thin {au['thin_lines'][:4]} long {au['long_labels'][:4]} numbers {au['loose_numbers'][:6]}")
    plt.close(fig)
    fig, rec_s = build("slide")
    fig.savefig(OUT_SLIDE / "XK_support_scale_slide.png", dpi=300)
    plt.close(fig)
    t = pd.read_csv(TAB)
    t["in_figure_top"] = [any(r == rr and cs == cc for rr, cc in REGIONS) and m == "none" for r, cs, m in zip(t.region, t.cell_set, t["mask"])]
    t["in_figure_bottom"] = (t.cell_set == "models") & (t["mask"] == "none") & (t.method == "p1")
    t.to_csv(OUT_PAPER / "XK_support_scale_source.csv", index=False, float_format="%.4g")
    print("[xk-fig]", json.dumps(dict(paper=rec, slide=rec_s)))


if __name__ == "__main__":
    main()
