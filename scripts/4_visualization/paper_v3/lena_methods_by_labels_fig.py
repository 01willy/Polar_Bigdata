"""showpiece · 레나 델타, 라벨 n = 10, 40, 160(행) × 방법 4종(열)의 1 km ALT 지도. 지도 아래 보류 라벨 RMSE(지시에 따른 그림 안 수치),
행마다 RMSE 가 가장 작은 지도에 가는 주홍 테두리. 오른쪽에 교차검증 방법 선택(W)의 선택과 보류 RMSE.
입력  data/processed/map_lena/methods_by_labels_v1/(scripts/2_evaluation/lena_methods_by_labels_v1.py)
산출  deck/assets/paper_report/maps/Lena_methods_by_labels_slide.png, outputs/figures/paper/v3_restructure/maps/Lena_methods_by_labels_v1.{pdf,png},
      _legend.md, _source_values.json
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json                                                                                             # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import Normalize                                                                 # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402
import label_sequence_fig as LS                                                                         # noqa: E402

DIR = ROOT / "data" / "processed" / "map_lena" / "methods_by_labels_v1"
STEM = "Lena_methods_by_labels"
NS = (10, 40, 160)
METH = ("P1", "R1", "D0", "D1")
NAME = {"P1": S.METHOD["recalibrated_stefan"]["short"], "R1": S.METHOD["anchor_residual"]["short"], "D0": S.METHOD["direct_ml"]["short"],
        "D1": S.METHOD["physics_pseudo"]["short"]}
W_NAME = {"P0": "Source Stefan", "P1": "Recalibrated Stefan", "P2": "Local Stefan", "R1@0.25": "Anchor + residual ML, λ 0.25",
          "R1@1.0": "Anchor + residual ML, λ 1.0", "R2@0.25": "Augmented anchor + residual", "D1": "Physics pseudo-labels"}
ACCENT = S.ACCENT                      # 주홍 #D55E00
TITLE2 = {"P1": "Recalibrated\nStefan", "R1": "Anchor +\nresidual ML", "D0": "Direct\nML", "D1": "Physics\npseudo-labels"}
W_SHORT = {"P0": "Source Stefan", "P1": "Recalibrated Stefan", "P2": "Local Stefan", "R1@0.25": "Residual ML, λ 0.25", "R1@1.0": "Residual ML, λ 1.0",
           "R2@0.25": "Augmented residual", "D1": "Physics pseudo-labels"}


def load():
    d = MV.prepare(MV.load_region("lena"))
    meta = json.loads((DIR / "methods_by_labels_v1_meta.json").read_text())
    G, L = {}, {}
    for n in NS:
        g = pd.read_csv(DIR / f"pred_grid_n{n}.csv.gz", dtype={"cell_id": str})
        assert (g.cell_id.values == d["cells"].cell_id.values).all()
        G[n] = {m: g[m].values.astype(float) for m in METH}
        L[n] = pd.read_csv(DIR / f"pred_labels_n{n}.csv")
    sh = d["shown"].copy()
    for n in NS:
        for m in METH:
            sh &= np.isfinite(G[n][m])
    d["shown"] = sh
    pooled = np.concatenate([G[n][m][sh] for n in NS for m in METH])
    st = {s["n"]: s for s in meta["steps"]}
    d.update(G=G, L=L, meta=meta, st=st, rng=dict(alt=MV.pct_range(pooled)))
    return d


def figure(d, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    paper = medium == "paper"
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    if paper:
        W, H, TOP, RM_TXT, GAP, ROWL, KEY, WCOL = 170.0, 100.0, 7.5, 4.0, 1.8, 15.0, 14.0, 30.0
        RG = 1.2
    else:
        W, H, TOP, RM_TXT, GAP, ROWL, KEY, WCOL = MV.SLIDE_W_MM, MV.SLIDE_H_MAX_MM, 13.5, 6.0, 6.0, 24.0, 24.0, 62.0
        RG = 2.0
    mh = (H - TOP - 3 * (RM_TXT + RG) - 1.0) / 3
    mw = mh / asp
    tot = ROWL + 4 * mw + 3 * GAP + KEY + WCOL
    x0 = max((W - tot) / 2, 0.5) + ROWL
    fig = S.fig_mm(W, H) if paper else plt.figure(figsize=(W / 25.4, H / 25.4))
    xs = [x0 + j * (mw + GAP) for j in range(4)]
    ys = [TOP + i * (mh + RM_TXT + RG) for i in range(3)]
    lo, hi = d["rng"]["alt"]
    norm = Normalize(lo, hi)
    Wd, Hd = W, H
    for j, m in enumerate(METH):
        fig.text((xs[j] + mw / 2) / Wd, 1 - (TOP - 1.2) / Hd, TITLE2[m], ha="center", va="bottom", fontsize=M.fs_head, linespacing=1.05)
    letters = "abcdefghijkl"
    for i, n in enumerate(NS):
        st = d["st"][n]
        r = st["rmse_held"]
        best = min(METH, key=lambda m: r[m])
        t = fig.text((x0 - 3.0) / Wd, 1 - (ys[i] + mh / 2) / Hd, f"{n} labels", ha="right", va="center", fontsize=M.fs_head)
        t.set_gid("category")
        lab = d["L"][n]
        drawn = lab[lab.drawn == 1]
        xy = proj.transform_points(MV.PC, drawn.lon.values, drawn.lat.values)[:, :2]
        for j, m in enumerate(METH):
            ax = S.axes_mm(fig, xs[j], ys[i], mw, mh, projection=proj)
            MV.base_map(ax, d, ext, M)
            MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["G"][n][m], MV.CM_ALT, norm))
            ax.scatter(xy[:, 0], xy[:, 1], s=1.6 if paper else 5.0, c="#000000", edgecolors="#ffffff", linewidths=0.3 if paper else 0.5,
                       transform=proj, zorder=5)
            if m == best:
                ax.spines["geo"].set_edgecolor(ACCENT)
                ax.spines["geo"].set_linewidth(1.6 if paper else 2.6)
            if paper:
                MV.letter(fig, xs[j] + 0.6, ys[i] + 0.6 + 2.6, letters[i * 4 + j], M)
            t = fig.text((xs[j] + mw / 2) / Wd, 1 - (ys[i] + mh + 0.8) / Hd, f"RMSE {r[m]:.1f} cm", ha="center", va="top", fontsize=M.fs_small,
                         color=ACCENT if m == best else MV.GRAY_TXT)
            t.set_gid("sizekey")
        # 교차검증 방법 선택(W)
        xw = xs[3] + mw + KEY
        w = st["W"]
        t1 = fig.text(xw / Wd, 1 - (ys[i] + mh / 2 - (2.2 if paper else 3.6)) / Hd, "CV rule: " + (W_SHORT if paper else W_NAME).get(w["choice"], w["choice"]), ha="left", va="center",
                      fontsize=M.fs_small)
        t2 = fig.text(xw / Wd, 1 - (ys[i] + mh / 2 + (2.2 if paper else 3.6)) / Hd, f"RMSE {r['W']:.1f} cm", ha="left", va="center", fontsize=M.fs_small,
                      color=MV.GRAY_TXT)
        t1.set_gid("category"); t2.set_gid("sizekey")
    cax = S.axes_mm(fig, xs[3] + mw + (2.0 if paper else 3.0), ys[0] + 0.2 * mh, M.cbar, ys[2] - ys[0] + 0.6 * mh)
    sh = d["shown"]
    MV.colorbar(fig, cax, MV.CM_ALT, norm, "ALT (cm)", MV.ticks_between(lo, hi), M,
                MV.extend_of(np.concatenate([d["G"][n][m][sh] for n in NS for m in METH]), lo, hi), orientation="vertical")
    return fig, dict(size_mm=[W, round(H, 1)], map_mm=[round(mw, 1), round(mh, 1)])


def main():
    t0 = time.time()
    d = load()
    out = {}
    fig, lay = figure(d, "slide")
    fig.savefig(MV.OUT_SLIDE / f"{STEM}_slide.png", dpi=300)
    out["slide"] = dict(lay, overlaps=LS.overlaps_visible(fig)["overlap_pairs"][:6], outside=LS.overlaps_visible(fig)["outside"][:6])
    plt.close(fig)
    fig, lay = figure(d, "paper")
    S.save_fig(fig, f"{STEM}_v1", MV.OUT_PAPER, formats=("pdf", "png"))
    out["paper"] = dict(lay, overlaps=LS.overlaps_visible(fig)["overlap_pairs"][:6], outside=LS.overlaps_visible(fig)["outside"][:6])
    plt.close(fig)
    m = d["meta"]
    sv = dict(region="Lena Delta", split=m["split"], n_src=m["n_src"], n_pseudo=m["n_pseudo"], E0=m["E0"], alt_range=d["rng"]["alt"],
              steps={str(s["n"]): dict(E_n=s["E_n"], E_ls=s["E_ls"], n_held=s["n_held"], rmse_held=s["rmse_held"], bias_held=s["bias_held"], W=s["W"])
                     for s in m["steps"]}, held_out=m["held_out"], layout=out)
    (MV.OUT_PAPER / f"{STEM}_v1_source_values.json").write_text(json.dumps(sv, ensure_ascii=False, indent=1, default=str))
    st = d["st"]
    rows = "; ".join(f"{n} labels: " + ", ".join(f"{NAME[k]} {st[n]['rmse_held'][k]:.2f}" for k in METH)
                     + f", CV rule ({W_NAME.get(st[n]['W']['choice'])}) {st[n]['rmse_held']['W']:.2f}" for n in NS)
    (MV.OUT_PAPER / f"{STEM}_v1_legend.md").write_text(
        "# Lena Delta methods by label count\n\n"
        f"**1 km ALT of the Lena Delta from four methods fitted on the source pool ({S.fmt_int(m['n_src'])} cells of other regions, Lena and a 100 km "
        f"buffer removed) plus n Lena labels.** Rows, n = 10, 40, 160 labels: the first n cells of the seed-0 block-dispersed order used in the "
        f"label-count map sequence (black points). Columns: recalibrated Stefan (E_n, κ = 10); anchor plus residual ML (λ 0.25); direct ML; "
        f"physics pseudo-labels (ten pseudo rows per source row from the label half of split 1, value E_n·√TDD) plus the n labels. CatBoost, two seeds "
        f"averaged. Text under each map: RMSE at all Lena label cells not drawn (held-out cells include cells in blocks that also hold drawn labels); "
        f"vermillion frame, lowest of the four in the row. Right: choice of the cross-validation rule (seven candidates, five-fold block cross-validation "
        f"within the drawn labels) and its held-out RMSE. One draw and one split only. Held-out RMSE (cm): {rows}. Colour scale "
        f"{d['rng']['alt'][0]:.0f}–{d['rng']['alt'][1]:.0f} cm.\n")
    print(json.dumps(out, ensure_ascii=False)[:800], f"· {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
