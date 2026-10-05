"""showpiece · 알래스카 실측 대비 예측(0.5° 블록 5겹 교차검증의 블록 밖 예측, data/processed/map_alaska/alaska_cv_oof_v1.csv).

구성  a 실측 대 재보정 Stefan(블록 밖), b 실측 대 잔차 ML(블록 밖): 점은 2 × 2 cm 칸의 점 수로 색을 입힌다(밀도, 로그 눈금, 두 패널 공통), 1:1 선.
      c 블록별 |잔차| 평균의 변화(잔차 ML − 재보정 Stefan, 음수 = 잔차 ML 이 작다)를 0.5° 블록 사각형으로(0 중심 broc, 대칭 범위).
      RMSE 는 설명문(_legend.md)에만 적는다.
산출  outputs/figures/paper/v3_restructure/maps/Alaska_obs_vs_pred_v1.{pdf,png}, _legend.md, _source_values.json, _blocks.csv
      deck/assets/paper_report/maps/Alaska_obs_vs_pred_v1_slide.png
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/alaska_obs_vs_pred_fig.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import math                                                                                             # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm                                                     # noqa: E402
import cmcrameri.cm as cmc                                                                              # noqa: E402
import shapely                                                                                          # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402
import label_sequence_fig as LS                                                                         # noqa: E402

OOF = ROOT / "data" / "processed" / "map_alaska" / "alaska_cv_oof_v1.csv"
STEM = "Alaska_obs_vs_pred_v1"
CM_DENS = MV.lut(cmc.davos_r, 0.10, 0.95)
BLOCK_DEG = 0.5
BIN_CM = 2.0
NAME = {"p1": S.METHOD["recalibrated_stefan"]["short"], "r1": S.METHOD["anchor_residual"]["short"]}


def block_bounds(b: int):
    ilat = (int(b) + 50000) // 100000
    ilon = int(b) - ilat * 100000
    return ilon * BLOCK_DEG, ilat * BLOCK_DEG, (ilon + 1) * BLOCK_DEG, (ilat + 1) * BLOCK_DEG


def load() -> dict:
    o = pd.read_csv(OOF)
    y = o.y.values.astype(float)
    res = {"p1": o.oof_p1.values - y, "r1": o.oof_r1.values - y}
    g = pd.DataFrame(dict(block=o.block.values, a_p1=np.abs(res["p1"]), a_r1=np.abs(res["r1"]), lat=o.lat.values, lon=o.lon.values))
    blk = g.groupby("block").agg(n=("a_p1", "size"), mean_abs_p1=("a_p1", "mean"), mean_abs_r1=("a_r1", "mean"), lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    blk["delta"] = blk.mean_abs_r1 - blk.mean_abs_p1
    bb = np.array([block_bounds(b) for b in blk.block])
    blk["lon0"], blk["lat0"], blk["lon1"], blk["lat1"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    hi = float(math.ceil(max(y.max(), o.oof_p1.max(), o.oof_r1.max()) / 10) * 10)
    dens = {}
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    for k, p in (("p1", o.oof_p1.values), ("r1", o.oof_r1.values)):
        Hc, _, _ = np.histogram2d(y, p, bins=[edges, edges])
        ix = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
        iy = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        dens[k] = Hc[ix, iy]
    d = dict(o=o, y=y, res=res, blk=blk, hi=hi, dens=dens, dens_max=float(max(v.max() for v in dens.values())),
             delta_max=float(math.ceil(np.percentile(np.abs(blk.delta), 99))), rmse={k: float(np.sqrt(np.mean(v ** 2))) for k, v in res.items()},
             bias={k: float(np.mean(v)) for k, v in res.items()}, mean_abs={k: float(np.mean(np.abs(v))) for k, v in res.items()})
    d["proj"] = MV.projection("alaska")
    d["ext"] = MV.box_extent(d["proj"], MV.REG["alaska"]["box"], MV.REG["alaska"]["pad_m"])
    d.update(key="alaska", **{k: v for k, v in MV.REG["alaska"].items()})
    return d


def scatter_panel(ax, d, k, M, norm):
    o = d["o"]
    p = o[f"oof_{k}"].values
    order = np.argsort(d["dens"][k])
    hi = d["hi"]
    ax.plot([0, hi], [0, hi], color=S.REF_COLOR, lw=M.lw, zorder=1, gid="one_to_one")
    sc = ax.scatter(d["y"][order], p[order], c=d["dens"][k][order], cmap=CM_DENS, norm=norm, s=1.4 if M.paper else 4.5, linewidths=0, zorder=2, rasterized=True)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed ALT (cm)", fontsize=M.fs_lab)
    ax.set_ylabel("Predicted ALT (cm)", fontsize=M.fs_lab)
    ax.tick_params(labelsize=M.fs, width=M.lw, length=M.tick_len)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_linewidth(M.lw)
    return sc


def block_map(ax, d, M, norm):
    MV.base_map(ax, d, d["ext"], M)
    blk = d["blk"]
    for r in blk.itertuples():
        geom = shapely.box(r.lon0, r.lat0, r.lon1, r.lat1)
        ax.add_geometries([geom], crs=MV.PC, facecolor=MV.CM_DIFF(norm(r.delta)), edgecolor="#4d4d4d", linewidth=M.lw_grat, zorder=3)
    MV.graticule(ax, d["xlocs"], d["ylocs"], M, left=True, bottom=True)


def figure(d, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    paper = medium == "paper"
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    norm_d = LogNorm(1, max(d["dens_max"], 10))
    v = max(d["delta_max"], 1.0)
    norm_b = TwoSlopeNorm(0.0, -v, v)
    if paper:                                                            # 논문판: a·b 를 왼쪽에 세로로, c 를 오른쪽에 크게
        W, LM, PW, KEY, PL, RM, TOP, ROWGAP, BOT = 170.0, 12.0, 36.0, 13.0, 11.0, 0.5, 3.2, 9.5, 9.5
        mw = W - LM - PW - KEY - PL - KEY - RM
        mh = mw * asp
        H = max(TOP + 2 * PW + ROWGAP + BOT, TOP + mh + BOT)
        fig = S.fig_mm(W, H)
        xs, ys = [LM, LM], [TOP, TOP + PW + ROWGAP]
        xm, ym = LM + PW + KEY + PL, TOP
    else:                                                                # 슬라이드판: 한 줄
        W, LM, PW, GAP, PL, KEY, RM, TOP, BOT = MV.SLIDE_W_MM, 17.0, 58.0, 6.0, 17.0, 26.0, 1.5, 9.0, 14.0
        mw = W - LM - 2 * PW - GAP - 2 * PL - 2 * KEY - RM
        mh = mw * asp
        if TOP + mh + BOT > MV.SLIDE_H_MAX_MM:
            mh = MV.SLIDE_H_MAX_MM - TOP - BOT
            mw = mh / asp
        H = TOP + max(PW, mh) + BOT
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        yoff = max(PW, mh) - PW
        xs, ys = [LM, LM + PW + GAP + PL], [TOP + yoff, TOP + yoff]
        xm, ym = LM + 2 * PW + GAP + PL + KEY + PL, TOP
    for j, k in enumerate(("p1", "r1")):
        ax = S.axes_mm(fig, xs[j], ys[j], PW, PW)
        sc = scatter_panel(ax, d, k, M, norm_d)
        MV.letter(fig, xs[j] - PL + 1.0, ys[j] - (0.5 if paper else 1.5), "ab"[j], M, None if paper else NAME[k])
    if paper:
        cax = S.axes_mm(fig, xs[0] + PW + 1.5, ys[0] + 0.1 * PW, M.cbar, 0.8 * PW)
    else:
        cax = S.axes_mm(fig, xs[1] + PW + 2.5, ys[1] + 0.1 * PW, M.cbar, 0.8 * PW)
    cb = fig.colorbar(sc, cax=cax, orientation="vertical", ticks=[1, 10, 100])
    cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.ax.set_yticklabels(["1", "10", "100"])                                                       # 지수 표기 대신 정수(5 pt 미만 글자 방지)
    MV.style_cbar(cb, "Cells per 2 × 2 cm bin", M)
    ax = S.axes_mm(fig, xm, ym, mw, mh, projection=proj)
    block_map(ax, d, M, norm_b)
    MV.letter(fig, xm, ym - (0.5 if paper else 1.5), "c", M, None)
    cx = xm + mw + (1.5 if paper else 2.5)
    cax = S.axes_mm(fig, cx, ym + 0.08 * mh, M.cbar, 0.70 * mh)
    MV.colorbar(fig, cax, MV.CM_DIFF, norm_b, "Change in mean |residual| (cm)", MV.ticks_between(-v, v), M, MV.extend_of(d["blk"].delta.values, -v, v),
                orientation="vertical")
    L_mm = MV.bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    LS.scale_bar_fig(fig, cx, ym + 0.84 * mh, L_mm, d["scale_km"], M, 3.3 if paper else 5.5)
    return fig, dict(size_mm=[W, round(H, 1)], scatter_mm=[PW, PW], map_mm=[round(mw, 1), round(mh, 1)])


def legend_md(d) -> str:
    blk = d["blk"]
    n_small = int((blk.n < 5).sum())
    better = int((blk.delta < 0).sum())
    v = max(d["delta_max"], 1.0)
    return (f"# Alaska observed versus predicted ALT (block cross-validation)\n\n"
            f"**Out-of-block predictions of active-layer thickness (ALT) at the {S.fmt_int(len(d['y']))} Alaska label cells.** Each cell is predicted "
            f"by a model fitted without its 0.5° block (five-fold block cross-validation of the Alaska map model, {len(blk)} blocks; in each fold the "
            f"Stefan coefficient and the residual weight λ are re-estimated within the training folds). a, observed against the recalibrated Stefan "
            f"model; b, observed against the recalibrated anchor plus residual ML (the map model). Points are coloured by the number of cells in their "
            f"2 × 2 cm bin (log scale); grey line, 1:1. c, change in the mean absolute residual per block, residual ML minus recalibrated Stefan "
            f"(negative, residual ML residuals smaller), drawn as 0.5° blocks; {better} of {len(blk)} blocks are negative; {n_small} blocks hold fewer "
            f"than five cells. Colour range ±{v:.0f} cm (99th percentile of |change|). Cell-weighted RMSE: recalibrated Stefan {d['rmse']['p1']:.2f} cm, "
            f"residual ML {d['rmse']['r1']:.2f} cm (registered map record 14.30 and 13.80 cm); mean absolute residual {d['mean_abs']['p1']:.2f} and "
            f"{d['mean_abs']['r1']:.2f} cm; bias {S.fmt_num(d['bias']['p1'], 2)} and {S.fmt_num(d['bias']['r1'], 2)} cm. Observed values above about 100 cm are few and are "
            f"under-predicted by both models. Projection NAD83 / Alaska Albers; land grey, Natural Earth 10 m.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", default="both")
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load()
    sv = dict(n_cells=int(len(d["y"])), n_blocks=int(len(d["blk"])), rmse=d["rmse"], bias=d["bias"], mean_abs=d["mean_abs"],
              blocks_delta_negative=int((d["blk"].delta < 0).sum()), blocks_lt5_cells=int((d["blk"].n < 5).sum()),
              delta_range=[-max(d["delta_max"], 1.0), max(d["delta_max"], 1.0)], dens_max=d["dens_max"], axis_max=d["hi"], source=str(OOF.relative_to(ROOT)),
              layout={})
    d["blk"].to_csv(MV.OUT_PAPER / f"{STEM}_blocks.csv", index=False, float_format="%.6g")
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
    (MV.OUT_PAPER / f"{STEM}_legend.md").write_text(legend_md(d))
    print(json.dumps(sv, ensure_ascii=False, default=str)[:1200], f"· {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
