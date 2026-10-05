"""showpiece · 레나 델타 전이(라벨 0개) 방법별 지도와 실측 대비 잔차. 모형 산출은 scripts/2_evaluation/lena_transfer_methods_v1.py 가 쓴 것을 읽는다.

구성  1행 a–c: 원천 계수 Stefan(P0), 직접 ML(D0), 물리 유사라벨 증강(D1)의 1 km ALT(한 색 척도), d: 라벨 셀의 실측 대 예측 산점도(1:1 선).
      2행 e–g: 라벨 셀의 잔차(예측 − 실측)를 지도 위 점으로(0 중심 broc, 공통 범위), h: 잔차 분포(방법별 계단 히스토그램).
      RMSE 는 그림에 적지 않고 설명문(_legend.md)에만 둔다.
산출  outputs/figures/paper/v3_restructure/maps/Lena_transfer_methods_v1.{pdf,png}, _legend.md, _source_values.json
      deck/assets/paper_report/maps/Lena_transfer_methods_v1_slide.png
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/lena_transfer_fig.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm                                                   # noqa: E402
from matplotlib.lines import Line2D                                                                     # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402
import label_sequence_fig as LS                                                                         # noqa: E402

PROC = ROOT / "data" / "processed"
TM = PROC / "map_lena" / "transfer_methods_v1"
STEM = "Lena_transfer_methods_v1"
METHODS = ("P0", "D0", "D1")
NAME = {"P0": S.METHOD["source_stefan"]["short"], "D0": S.METHOD["direct_ml"]["short"], "D1": S.METHOD["physics_pseudo"]["short"]}
COLOR = {"P0": S.METHOD["source_stefan"]["color"], "D0": S.METHOD["direct_ml"]["color"], "D1": S.METHOD["physics_pseudo"]["color"]}


def load() -> dict:
    d = MV.prepare(MV.load_region("lena"))
    meta = json.loads((TM / "transfer_methods_v1_meta.json").read_text())
    sp = int(meta["fig_split"])
    g = pd.read_csv(TM / f"pred_grid_split{sp}.csv.gz", dtype={"cell_id": str})
    assert (g.cell_id.values == d["cells"].cell_id.values).all()
    lab = pd.read_csv(TM / f"pred_labels_split{sp}.csv")
    preds = {m: g[m].values.astype(float) for m in METHODS}
    shown = d["shown"].copy()
    for v in preds.values():
        shown &= np.isfinite(v)
    d["shown"] = shown
    res = {m: lab[m].values - lab.y.values for m in METHODS}
    pooled = np.concatenate([preds[m][shown] for m in METHODS])
    d.update(preds=preds, lab=lab, res=res, split=sp, tm_meta=meta, xy_lab=d["proj"].transform_points(MV.PC, lab.lon.values, lab.lat.values)[:, :2],
             rng_tm=dict(alt=MV.pct_range(pooled), res_max=MV.sym_max(np.concatenate([res[m] for m in METHODS]))))
    return d


def rmse(v):
    return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))


# ================================================================ 패널
def map_panel(ax, d, m, M, kind, norm):
    if kind == "alt":
        MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["preds"][m], MV.CM_ALT, norm))
    else:
        MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, np.ones(len(d["shown"])), None, None))            # 표시 셀 바탕(#e6e6e6)
        xy = d["xy_lab"]
        ax.scatter(xy[:, 0], xy[:, 1], c=d["res"][m], cmap=MV.CM_DIFF, norm=norm, s=2.2 if M.paper else 6.0, linewidths=0, transform=d["proj"], zorder=5)


def scatter_panel(ax, d, M):
    lab = d["lab"]
    y = lab.y.values
    hi = float(np.ceil(max(y.max(), max(lab[m].values.max() for m in METHODS)) / 20) * 20)
    ax.plot([0, hi], [0, hi], color=S.REF_COLOR, lw=M.lw, zorder=1, gid="one_to_one")
    for m in METHODS:
        ax.scatter(y, lab[m].values, s=1.2 if M.paper else 4.0, c=COLOR[m], alpha=0.45, linewidths=0, zorder=2 + METHODS.index(m), rasterized=True)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed ALT (cm)", fontsize=M.fs_lab)
    ax.set_ylabel("Predicted ALT (cm)", fontsize=M.fs_lab)
    ax.tick_params(labelsize=M.fs, width=M.lw, length=M.tick_len)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_linewidth(M.lw)
    return hi


def hist_panel(ax, d, M):
    v = d["rng_tm"]["res_max"]
    lim = float(max(v, 10.0))
    bins = np.arange(-lim, lim + 1e-9, 5.0)
    for m in METHODS:
        r = np.clip(d["res"][m], -lim + 1e-6, lim - 1e-6)
        ax.hist(r, bins=bins, histtype="step", color=COLOR[m], lw=M.lw_scale * 0.8, zorder=2 + METHODS.index(m))
    ax.axvline(0, color=S.REF_COLOR, lw=M.lw, zorder=1)
    ax.set_xlim(-lim, lim)

    ax.set_xlabel("Prediction − observed (cm)", fontsize=M.fs_lab)
    ax.set_ylabel("Label cells", fontsize=M.fs_lab)
    ax.tick_params(labelsize=M.fs, width=M.lw, length=M.tick_len)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_linewidth(M.lw)


def figure(d, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    paper = medium == "paper"
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    if paper:
        W, LM, GAP, KEY, PW, PL, RM = 170.0, 8.5, 1.8, 15.0, 33.0, 11.5, 0.5
        TOP, CNT, ROWGAP, BOT = 7.0, 0.0, 9.5, 10.0
    else:
        W, LM, GAP, KEY, PW, PL, RM = MV.SLIDE_W_MM, 15.0, 4.0, 27.0, 54.0, 19.0, 1.5
        TOP, CNT, ROWGAP, BOT = 17.0, 0.0, 15.0, 14.0
    mw = (W - LM - 2 * GAP - KEY - PL - PW - RM) / 3
    mh = mw * asp
    H = TOP + mh + ROWGAP + mh + BOT
    if not paper and H > MV.SLIDE_H_MAX_MM:
        mh = (MV.SLIDE_H_MAX_MM - TOP - ROWGAP - BOT) / 2
        mw = mh / asp
        H = MV.SLIDE_H_MAX_MM
    fig = S.fig_mm(W, H) if paper else plt.figure(figsize=(W / 25.4, H / 25.4))
    content_w = LM + 3 * mw + 2 * GAP + KEY + PL + PW + RM
    LM = LM + max(0.0, (W - content_w) / 2)                                # 슬라이드: 내용이 좁으면 가운데에 둔다
    xs = [LM + j * (mw + GAP) for j in range(3)]
    ys = [TOP, TOP + mh + ROWGAP]
    lo, hi = d["rng_tm"]["alt"]
    v = d["rng_tm"]["res_max"]
    n_alt, n_res = Normalize(lo, hi), TwoSlopeNorm(0.0, -v, v)
    letters = "abcdefgh"
    for i, kind in enumerate(("alt", "res")):
        for j, m in enumerate(METHODS):
            ax = S.axes_mm(fig, xs[j], ys[i], mw, mh, projection=proj)
            MV.base_map(ax, d, ext, M)
            map_panel(ax, d, m, M, kind, n_alt if kind == "alt" else n_res)
            LS.graticule_tl(ax, d, M, left=(j == 0), top=(paper and i == 0 and j == 0))
            k = letters[i * 4 + j]
            MV.letter(fig, xs[j], ys[i] - (0.5 if paper else 1.5), k, M, None if paper else (NAME[m] if i == 0 else None))
    cx = xs[2] + mw + (1.5 if paper else 2.5)
    cax = S.axes_mm(fig, cx, ys[0] + 0.06 * mh, M.cbar, 0.76 * mh)
    sh = d["shown"]
    MV.colorbar(fig, cax, MV.CM_ALT, n_alt, "ALT (cm)", MV.ticks_between(lo, hi), M,
                MV.extend_of(np.concatenate([d["preds"][m][sh] for m in METHODS]), lo, hi), orientation="vertical")
    cax = S.axes_mm(fig, cx, ys[1] + 0.06 * mh, M.cbar, 0.76 * mh)
    MV.colorbar(fig, cax, MV.CM_DIFF, n_res, "Prediction − observed (cm)", MV.ticks_between(-v, v), M,
                MV.extend_of(np.concatenate([d["res"][m] for m in METHODS]), -v, v), orientation="vertical")
    L_mm = MV.bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    LS.scale_bar_fig(fig, cx, ys[1] + 0.85 * mh, L_mm, d["scale_km"], M, 3.3 if paper else 5.5)
    px = xs[2] + mw + KEY + PL
    ph = mh
    ax = S.axes_mm(fig, px, ys[0], PW, ph)
    scatter_panel(ax, d, M)
    MV.letter(fig, px - PL + 1.0, ys[0] - (0.5 if paper else 1.5), letters[3], M)
    ax = S.axes_mm(fig, px, ys[1], PW, ph)
    hist_panel(ax, d, M)
    MV.letter(fig, px - PL + 1.0, ys[1] - (0.5 if paper else 1.5), letters[7], M)
    hs = [Line2D([], [], marker="o", ls="none", color=COLOR[m], markersize=3.0 if M.paper else 6.0, label=NAME[m]) for m in METHODS]
    fig.legend(handles=hs, loc="upper right", bbox_to_anchor=((px + PW) / W, 1 - (0.3 if paper else 1.0) / H), ncol=3, fontsize=M.fs_small,
               frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)                     # 방법 열쇠: 오른쪽 열 위 한 줄
    lay = dict(size_mm=[W, round(H, 1)], map_mm=[round(mw, 1), round(mh, 1)], scatter_mm=[PW, round(ph, 1)])
    return fig, lay


def legend_md(d, sv) -> str:
    m = d["tm_meta"]
    lo, hi = d["rng_tm"]["alt"]
    v = d["rng_tm"]["res_max"]
    r_all = sv["rmse_all_label_cells"]
    r_reg = sv["rmse_registered_scoring_half_mean_over_splits"]
    reg = m.get("registered_lg_curve_n0", {})
    ps = m["per_split"][m["splits"].index(m["fig_split"])]
    rg = lambda k: reg.get(k, {}).get("rmse", float("nan"))                                            # noqa: E731
    return (f"# Lena Delta transfer maps without target labels\n\n"
            f"**1 km active-layer thickness (ALT) of the Lena Delta predicted without Lena labels by three methods, and their errors at the "
            f"{S.fmt_int(len(d['lab']))} Lena label cells.** All fits use only the {S.fmt_int(ps['n_src'])} source label cells of the other regions "
            f"(Lena and a 100 km buffer removed), so every Lena label cell is a test point. a, source-coefficient Stefan E0·√TDD, E0 = {ps['E0']:.3f}. "
            f"b, direct ML: CatBoost (200 iterations, depth 3, two seeds averaged) on the 25 covariates of the source rows. "
            f"c, physics pseudo-label augmentation: the same learner on the source rows plus {S.fmt_int(ps['n_pseudo'])} pseudo rows (ten per source row) "
            f"drawn with replacement from the {S.fmt_int(ps['n_A'])} cells of the label half of registered split {m['fig_split']}, valued E0·√TDD. "
            f"d, observed against predicted ALT (grey line, 1:1). e–g, prediction minus observed per label cell. h, residual distribution (5 cm bins). "
            f"Colour scales: ALT {lo:.0f}–{hi:.0f} cm (1st–99th percentile of a–c), residual ±{v:.0f} cm (99th percentile). "
            f"RMSE at all label cells: {r_all['P0']:.2f}, {r_all['D0']:.2f} and {r_all['D1']:.2f} cm (a, b, c); "
            f"on the registered scoring halves, mean of five splits: {r_reg['P0']:.2f}, {r_reg['D0']:.2f}, {r_reg['D1']:.2f} cm "
            f"(registered record {rg('P0|none|lam0.0'):.2f}, {rg('D0|catboost_lo|lam1.0'):.2f}, {rg('D1|catboost_lo|lam1.0'):.2f} cm). "
            f"White, water; grey, no data.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", default="both")
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load()
    m = d["tm_meta"]
    sv = dict(region="Lena Delta", split=d["split"], n_label_cells=int(len(d["lab"])), n_shown=int(d["shown"].sum()), E0=m["per_split"][0]["E0"],
              n_src=m["per_split"][0]["n_src"], n_pseudo=m["per_split"][0]["n_pseudo"], ranges=dict(alt=d["rng_tm"]["alt"], residual=[-d["rng_tm"]["res_max"], d["rng_tm"]["res_max"]]),
              rmse_all_label_cells={k: rmse(v) for k, v in d["res"].items()}, bias_all_label_cells={k: float(np.mean(v)) for k, v in d["res"].items()},
              rmse_registered_scoring_half_mean_over_splits={k: float(np.mean([ps["rmse_B_seedmean"][k] for ps in m["per_split"] if ps["valid"]])) for k in METHODS},
              rmse_registered_scoring_half_per_split={str(ps["split"]): ps["rmse_B_seedmean"] for ps in m["per_split"]},
              rmse_all_label_cells_per_split={str(ps["split"]): ps["rmse_all_labels_seedmean"] for ps in m["per_split"]},
              registered_lg_curve_n0=m.get("registered_lg_curve_n0", {}), map_stats={k: dict(p50=float(np.median(v[d["shown"]])), mean=float(np.mean(v[d["shown"]])))
                                                                                   for k, v in d["preds"].items()}, layout={})
    media = ["paper", "slide"] if a.medium == "both" else [a.medium]
    for medium in media:
        fig, lay = figure(d, medium)
        if medium == "paper":
            S.save_fig(fig, STEM, MV.OUT_PAPER, formats=("pdf", "png"))
            au = S.audit_v3(fig)
            ov = LS.overlaps_visible(fig)
            sv["layout"]["paper"] = dict(lay, audit=dict(sizes=sorted(au["sizes"]), chars=au["chars"], thin=au["thin_lines"][:4], overlaps=ov["overlap_pairs"][:6],
                                                        outside=ov["outside"][:6]))
        else:
            fig.savefig(MV.OUT_SLIDE / f"{STEM}_slide.png", dpi=300)
            ov = LS.overlaps_visible(fig)
            sv["layout"]["slide"] = dict(lay, overlaps=ov["overlap_pairs"][:6], outside=ov["outside"][:6])
        plt.close(fig)
    (MV.OUT_PAPER / f"{STEM}_source_values.json").write_text(json.dumps(sv, ensure_ascii=False, indent=1, default=str))
    (MV.OUT_PAPER / f"{STEM}_legend.md").write_text(legend_md(d, sv))
    print(json.dumps(dict(rmse=sv["rmse_all_label_cells"], reg_half=sv["rmse_registered_scoring_half_mean_over_splits"], ranges=sv["ranges"], layout=sv["layout"]),
                     ensure_ascii=False, default=str)[:1500], f"· {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
