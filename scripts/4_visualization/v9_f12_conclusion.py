# -*- coding: utf-8 -*-
"""v9_f12_conclusion.py: 결론 쪽 근거 그림(2026-10-16 사용자 지시 12: 결론을 한 상자에 몰지 말고 주장별로 근거와 함께).
덱 전용 PNG(영문 라벨, 300 dpi): deck/assets/v9/C01_evidence.png(5.9 × 5.56 in, 패널 셋이 덱의 주장 카드 세 줄과 같은 높이에 정렬).
  1 지역 안 이득: 물리 + 잔차 ML − Stefan(재보정) ΔRMSE, 공간 블록, 95 % 블록 구간(runs/core_v6/deltas_vs_ref.csv).
  2 새 지역 소수 라벨(레나 델타): 대상 라벨 n 에 따른 RMSE, Wei 2026 값이 있는 같은 셀-연도(runs/ladder_v6/preds_TL_Lena.csv.gz + 제품 표).
  3 같은 셀 · 같은 해 제품 비교: 공통 셀-연도 RMSE(runs/core_v6/common_cells_eval.csv, mask 0).
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v9_f12_conclusion.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S  # noqa: E402
OUT = ROOT / "deck" / "assets" / "v9"; OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT / "data" / "processed" / "v5" / "runs"
REG = ["Alaska", "Lena", "Canada"]
MH = S.METHOD_HEX; PH = S.PRODUCT_HEX; ZERO = S.NEUTRAL.get("zero_line", "#9a9a9a")
W, H = 5.9, 5.56
ROWS = [(3.76, 1.80), (1.88, 1.80), (0.0, 1.80)]          # 주장 카드 세 줄의 (아래 y, 높이), 그림 좌표(in, 아래 기준)
FS_K, FS_L, FS_T = 12, 13, 12


def tl_on_wei_rows():
    p = pd.read_csv(RUN / "ladder_v6" / "preds_TL_Lena.csv.gz")
    pr = pd.read_csv(ROOT / "data" / "processed" / "v5" / "products_cell_year_v5.csv")
    wei = pr[pr["product"] == "wei"].set_index(["loc_id", "year"]).value_cm
    p["wei"] = pd.MultiIndex.from_frame(p[["loc_id", "year"]]).map(wei).values
    p = p[p.wei.notna()]
    w = p.groupby(["n_label", "method", "cell_year_id"]).agg(pred=("pred", "mean"), y=("y", "first"), wei=("wei", "first")).reset_index()
    rm = w.groupby(["n_label", "method"]).apply(lambda q: float(np.sqrt(np.mean((q.pred - q.y) ** 2)))).unstack()
    wq = w[w.n_label == 0].drop_duplicates("cell_year_id"); rw = float(np.sqrt(np.mean((wq.wei - wq.y) ** 2)))
    return rm, rw, int(wq.cell_year_id.nunique())


def main():
    S.setup(scale=1.69); plt.rcParams.update({"font.size": FS_K})
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    L0, AW = 1.55, W - 1.55 - 0.15
    # 1 지역 안 이득
    d = pd.read_csv(RUN / "core_v6" / "deltas_vs_ref.csv")
    d = d[(d.scheme == "block") & (d.forcing == "year_obs") & (d.featset == "F8TS") & (d.method == "R1") & (d.learner == "catboost") & (d.lam == 0.5) & (d.ref == "P1")].set_index("region")
    y0, h0 = ROWS[0]; ax = fig.add_axes([L0 / W, (y0 + 0.62) / H, AW / W, (h0 - 0.80) / H])
    for i, r in enumerate(REG):
        v = d.loc[r]; y = 2 - i
        ax.plot([v.lo, v.hi], [y, y], color=MH["R1"], lw=2.0, solid_capstyle="butt"); ax.plot([v.delta], [y], "o", ms=8.5, color=MH["R1"])
    ax.axvline(0, color=ZERO, lw=0.9); ax.set_yticks([2, 1, 0]); ax.set_yticklabels([S.REGION_EN[r] for r in REG], fontsize=FS_L)
    for t, r in zip(ax.get_yticklabels(), REG):
        t.set_color(S.REGION_HEX[r])
    ax.set_ylim(-0.5, 2.5); ax.set_xlim(-9, 2); ax.set_xticks([-8, -6, -4, -2, 0, 2]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ΔRMSE vs Stefan (recalibrated), cm", fontsize=FS_K); ax.spines["left"].set_visible(False)
    # 2 새 지역 소수 라벨(레나 델타, Wei 행)
    rm, rw, nrow = tl_on_wei_rows(); ns = [0, 25, 50, 100, 250, 500]; xs = np.arange(len(ns))
    y1, h1 = ROWS[1]; ax2 = fig.add_axes([L0 / W, (y1 + 0.62) / H, AW / W, (h1 - 0.74) / H])
    for m, nm, col, ls, mk in (("R1S", "Physics + residual ML", MH["R1"], "-", "o"), ("P1n", "Stefan (recalibrated on n)", MH["P1"], "-", "s"),
                               ("D0S", "Direct ML", MH["D0"], (0, (3.5, 1.8)), "o")):
        v = [rm.loc[n, m] if m in rm.columns and n in rm.index else np.nan for n in ns]
        ax2.plot(xs, v, ls=ls, color=col, lw=2.0, marker=mk, ms=5.5, mfc=col if mk == "o" and m != "D0S" else "white", mec=col, label=nm)
    ax2.axhline(rw, color=PH["wei"], lw=1.6, ls=(0, (5, 2)))
    ax2.text(2.0, rw + 0.8, "Wei 2026", color=PH["wei"], fontsize=FS_T, va="bottom", ha="center")
    ax2.set_xticks(xs); ax2.set_xticklabels([str(n) for n in ns]); ax2.set_ylim(15, 45); ax2.set_yticks([20, 30, 40])
    ax2.set_xlim(-0.3, len(ns) - 0.7)
    ax2.set_xlabel("Lena Delta labels added to the other regions", fontsize=FS_K); ax2.set_ylabel("RMSE (cm)", fontsize=FS_K)
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(h, ["Physics + residual ML", "Stefan, recalibrated", "Direct ML"], loc="lower left", bbox_to_anchor=(0.03, (y1 + 0.62 + h1 - 0.74 + 0.03) / H), bbox_transform=fig.transFigure,
               ncol=3, fontsize=FS_T - 1, frameon=False, handlelength=1.4, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    # 3 같은 셀 · 같은 해 제품 비교
    c = pd.read_csv(RUN / "core_v6" / "common_cells_eval.csv"); c = c[c.mask_km == 0]
    y2, h2 = ROWS[2]; ax3 = fig.add_axes([L0 / W, (y2 + 0.62) / H, AW / W, (h2 - 0.82) / H])
    ser = [("R1 (CatBoost)", "Physics + residual ML", MH["R1"], "o", 9), ("Wei 2026", "Wei 2026", PH["wei"], "D", 7.5), ("Liu 2024", "Liu 2024", PH["liu"], "D", 7.5),
           ("ESA CCI v4", "ESA CCI v4", PH["cci4y"], "s", 7.0)]
    for i, r in enumerate(REG):
        q = c[c.region == r].set_index("series"); y = 2 - i
        vals = [q.loc[k, "rmse"] for k, *_ in ser if k in q.index]
        ax3.plot([min(vals), max(vals)], [y, y], color="#d9d9d9", lw=1.2, zorder=1)
        for k, nm, col, mk, ms in ser:
            if k in q.index:
                ax3.plot([q.loc[k, "rmse"]], [y], ls="none", marker=mk, ms=ms, color=col, zorder=3 if k.startswith("R1") else 2)
    ax3.set_yticks([2, 1, 0]); ax3.set_yticklabels([S.REGION_EN[r] for r in REG], fontsize=FS_L)
    for t, r in zip(ax3.get_yticklabels(), REG):
        t.set_color(S.REGION_HEX[r])
    ax3.set_ylim(-0.5, 2.5); ax3.set_xlim(14, 37); ax3.set_xticks([15, 20, 25, 30, 35]); ax3.tick_params(axis="y", length=0); ax3.spines["left"].set_visible(False)
    ax3.set_xlabel("RMSE, same cell and year (cm)", fontsize=FS_K)
    hs = [Line2D([], [], ls="none", marker=mk, ms=ms * 0.8, color=col, label=nm) for _, nm, col, mk, ms in ser]
    ax3.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=4, fontsize=FS_T - 1.5, frameon=False, handletextpad=0.2, columnspacing=0.7, borderaxespad=0.0)
    for a_ in (ax, ax2, ax3):
        a_.tick_params(labelsize=FS_K); a_.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    p = OUT / "C01_evidence.png"; fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    from PIL import Image
    with Image.open(p) as im:
        im.convert("RGB").save(p, dpi=(300, 300)); print(p, im.size, "TL rows", nrow, "Wei RMSE", round(rw, 1))
    print(rm.round(1).to_string())


def c02(bottom="ft"):
    """의의 쪽 근거 그림(6.9 × 5.56 in): 위 = 해마다 다른 지도(G6 지도 영역을 같은 배율로 잘라 씀), 아래 = 외부 지역 소수 라벨 적응(전체) 또는 지역 홀드아웃 전이."""
    from PIL import Image
    S.setup(scale=1.69); plt.rcParams.update({"font.size": FS_K})
    Wc, Hc = 6.9, 5.56
    fig = plt.figure(figsize=(Wc, Hc)); fig.patch.set_facecolor("white")
    g6 = Image.open(ROOT / "deck" / "assets" / "v8" / "G6_conclusion.png").convert("RGB")
    crop = np.asarray(g6.crop((110, 0, 2190, 1110))); ch, cw = crop.shape[0] / 300.0, crop.shape[1] / 300.0
    axm = fig.add_axes([(Wc - cw) / 2 / Wc, (Hc - ch) / Hc, cw / Wc, ch / Hc]); axm.imshow(crop, interpolation="none"); axm.axis("off")
    ax = fig.add_axes([1.05 / Wc, 0.55 / Hc, (Wc - 1.25) / Wc, 1.05 / Hc])
    if bottom == "ft":
        import glob
        pp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RUN / "extra_v9" / "preds_EXTFT_*.csv.gz")))], ignore_index=True); pp["n_label"] = pp.n_label.astype(str)
        first = pp.drop_duplicates("cell_year_id").set_index("cell_year_id")[["y", "prod_wei", "prod_liu", "prod_cci5y"]]
        ns = ["0", "5", "10", "25", "50", "all"]; xs = np.arange(len(ns)); vals = {}
        for nl in ns:
            w = pp[pp.n_label == nl].groupby(["cell_year_id", "method"]).pred.mean().unstack().join(first)
            w = w[w.prod_wei.notna() & w.prod_liu.notna() & w.prod_cci5y.notna()]          # F10 FT 그림과 같은 행(세 제품 값이 모두 있는 셀-연도)
            for m in ("R1S", "D0S"):
                vals.setdefault(m, []).append(float(np.sqrt(np.mean((w[m] - w.y) ** 2))))
            vals["cci"] = float(np.sqrt(np.mean((w.prod_cci5y - w.y) ** 2)))
        for m, nm, col, ls, mk in (("R1S", "Physics + residual ML", MH["R1"], "-", "o"), ("D0S", "Direct ML", MH["D0"], (0, (3.5, 1.8)), "o")):
            ax.plot(xs, vals[m], ls=ls, color=col, lw=2.0, marker=mk, ms=5.5, mfc=col if m == "R1S" else "white", mec=col, label=nm)
        ax.axhline(vals["cci"], color=PH["cci5y"], lw=1.6, ls=(0, (5, 2)), label="ESA CCI v5 (independent)")
        ax.set_xticks(xs); ax.set_xticklabels(["0", "5", "10", "25", "50", "all"]); ax.set_xlabel("Local labels in external CALM regions", fontsize=FS_K)
        ax.set_ylabel("RMSE (cm)", fontsize=FS_K); ax.set_ylim(30, 65); ax.set_yticks([30, 40, 50, 60])
        ax.legend(loc="lower left", bbox_to_anchor=(0.02, (0.55 + 1.05 + 0.04) / Hc), bbox_transform=fig.transFigure, ncol=3, fontsize=FS_T - 2, frameon=False,
                  handlelength=1.3, columnspacing=0.7, handletextpad=0.3)
    else:
        d = pd.read_csv(RUN / "core_v6" / "deltas_vs_ref.csv")
        d = d[(d.scheme == "region") & (d.forcing == "year_obs") & (d.featset == "F8TS") & (d.learner == "catboost") & (d.ref == "P1")]
        for j, (m, lam, nm, col, filled) in enumerate((("D0", -1.0, "Direct ML", MH["D0"], False), ("R1", 0.5, "Physics + residual ML", MH["R1"], True))):
            q = d[(d.method == m) & (d.lam == lam)].set_index("region")
            for i, r in enumerate(REG):
                v = q.loc[r]; x = i + (j - 0.5) * 0.25
                ax.plot([x, x], [v.lo, v.hi], color=col, lw=1.8); ax.plot([x], [v.delta], "o", ms=8, mfc=col if filled else "white", mec=col, mew=1.6, label=nm if i == 0 else None)
        ax.axhline(0, color=ZERO, lw=0.9); ax.set_xticks(range(3)); ax.set_xticklabels([S.REGION_EN[r] for r in REG])
        ax.set_ylabel("ΔRMSE vs Stefan,\nnew region (cm)", fontsize=FS_K); ax.legend(loc="upper right", fontsize=FS_T - 1, frameon=False)
    ax.tick_params(labelsize=FS_K); ax.grid(axis="y", color="#eeeeee", lw=0.6)
    p = OUT / "C02_evidence.png"; fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        im.convert("RGB").save(p, dpi=(300, 300)); print(p, im.size, bottom)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--c02", default=""); a = ap.parse_args()
    if a.c02:
        c02(a.c02)
    else:
        main()
