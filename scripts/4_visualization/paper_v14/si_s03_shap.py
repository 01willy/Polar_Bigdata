"""Supplementary Fig. S3 (v11): SHAP attribution of the residual learner (content of v9_f10_extra.py SHAP).

Copied from paper_v10/si_s03_shap.py (data unchanged) and restyled for v11 (si13_common.py, POLAR_PALETTE=paper13). Input groups take the
muted colours of the deck SHAP figure (v9_f10_extra.GROUP_HEX: climate blue, terrain olive, soil brown) instead of the v10 greys; numbers inside
the bars use dark grey or white, whichever has the higher contrast on the bar.
a  Share of mean |SHAP| by input group (climate, terrain, soil) for the residual learner of Physics + residual ML and for Direct ML, per region.
b  The four inputs with the largest mean |SHAP| of the residual learner (cm), per region; bar colour = input group.
Inputs: data/processed/v5/runs/extra_v9/shap_groups.csv, shap_summary.csv (CatBoost, all label cell-years of each region).
Output: outputs/figures/paper/v14_si/FigS3_shap.{pdf,png}, entry S3 of outputs/figures/paper/v14_si/SI_figures.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v14/si_s03_shap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si13_common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v9_f10_extra as F10  # noqa: E402

STEM = "FigS3_shap"
RUN = C.ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9"
REG = C.REGIONS
GROUP_HEX = {"climate": "#4D78A7", "terrain": "#919161", "soil": "#9B6B49"}   # deck SHAP colours (v9_f10_extra.GROUP_HEX #4F7CAC, #8C8C5A, #A8744F)
                                                                             # shifted by 3-8 % so the 6 pt numbers reach >= 4.5:1 contrast
GROUP = {k: (k.capitalize(), GROUP_HEX[k]) for k in ("climate", "terrain", "soil")}


def main():
    C.setup()
    g = pd.read_csv(RUN / "shap_groups.csv"); s = pd.read_csv(RUN / "shap_summary.csv")
    grp = {f: ("climate" if f.startswith("e5_") else "terrain" if f.startswith("dem_") else "soil") for f in s.feature.unique()}
    W, H = 180.0, 64.0
    fig = C.figure(W, H)
    axa = C.ax_mm(fig, 33.0, 8.0, 52.0, 46.0)
    vals = {"a": {}, "b": {}}
    yt, yl, heads = [], [], []
    for i, r in enumerate(REG):
        for j, kind in enumerate(("residual", "direct")):
            y = i * 2.7 + j * 1.0; left = 0.0
            q = g[(g.region == r) & (g.kind == kind)].set_index("group").share
            for gk in ("climate", "terrain", "soil"):
                v = float(q.get(gk, 0.0)); col = GROUP[gk][1]
                axa.barh(y, v, left=left, color=col, height=0.78, edgecolor="white", lw=0.6)
                if v >= 0.10:
                    axa.text(left + v / 2, y, f"{v * 100:.0f}", ha="center", va="center", fontsize=C.FS_SMALL, color=C.on_colour(col))
                left += v
                vals["a"].setdefault(r, {}).setdefault(kind, {})[gk] = round(v, 4)
            yt.append(y); yl.append("Residual learner" if kind == "residual" else "Direct ML")
        heads.append((i * 2.7 + 0.5, C.REGION[r][0]))
    axa.set_yticks(yt); axa.set_yticklabels(yl); axa.set_ylim(yt[-1] + 0.65, -0.65)
    axa.tick_params(axis="y", length=0, pad=2)
    for y, nm in heads:
        axa.text(-0.53, y, nm, transform=axa.get_yaxis_transform(), ha="left", va="center", fontsize=C.FS, color=C.TEXT)
    axa.set_xlim(0, 1); axa.set_xticks([0, 0.25, 0.5, 0.75, 1]); axa.set_xticklabels(["0", "25", "50", "75", "100"])
    axa.set_xlabel("Share of mean |SHAP| (%)")
    C.clean_axes(axa, left=False); axa.tick_params(axis="y", length=0)
    # b
    xmax = float(np.ceil(s[s.kind == "residual"].mean_abs.max()))
    axes_b = [C.ax_mm(fig, 127.0, 8.0 + k * 15.6, 50.0, 10.8) for k in range(3)]
    for k, (ax, r) in enumerate(zip(axes_b, REG)):
        q = s[(s.region == r) & (s.kind == "residual")].sort_values("mean_abs", ascending=False).head(4)
        C.grid(ax, "x")
        ax.barh(range(len(q)), q.mean_abs, color=[GROUP[grp[f]][1] for f in q.feature], height=0.68, edgecolor="none", zorder=2)
        ax.set_yticks(range(len(q))); ax.set_yticklabels([F10.FEAT.get(f, f) for f in q.feature]); ax.set_ylim(len(q) - 0.45, -0.55)
        ax.set_xlim(0, xmax); C.clean_axes(ax, left=False); ax.tick_params(axis="y", length=0, pad=2)
        hb = ax.get_position().height * H
        ax.text(0.0, 1.0 + 0.9 / hb, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        if k < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Mean |SHAP| of the residual learner (cm)")
        vals["b"][r] = {F10.FEAT.get(f, f): round(float(v), 3) for f, v in zip(q.feature, q.mean_abs)}
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 90.0, 0.5, "b")
    hs = [Patch(facecolor=GROUP[k][1], edgecolor="none") for k in GROUP]
    fig.legend(hs, [GROUP[k][0] for k in GROUP], loc="upper left", bbox_to_anchor=(33.0 / W, 1 - 1.0 / H), ncol=3, handlelength=1.2,
               handleheight=0.9, columnspacing=1.4, fontsize=C.FS_TICK)
    rep = C.qa_figure(fig, ignore_ink_for=[str(v) for v in range(101)]); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    n = s.groupby("region").n.first().to_dict()
    C.update_si_json("S3", dict(
        title="SHAP attribution of the residual learner", stem=STEM, script="scripts/4_visualization/paper_v14/si_s03_shap.py",
        version="v11 (2026-10-16): v11 restyle of v10 FigS3_shap (input-group colours from the deck SHAP figure); numbers unchanged",
        inputs=["data/processed/v5/runs/extra_v9/shap_groups.csv", "data/processed/v5/runs/extra_v9/shap_summary.csv"],
        years="all label cell-years of each region: Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024; climate input to the observation day",
        n_cell_years={k: int(v) for k, v in n.items()}, groups="climate = ERA5-Land inputs (e5_*), terrain = DEM inputs (dem_*), soil = SoilGrids (sg_*)",
        group_colours={GROUP[k][0]: GROUP[k][1] for k in GROUP},
        values=vals, numbers_printed="a: group shares in percent (shares >= 10 % printed inside the bars)", outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
