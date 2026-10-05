"""showpiece · 라벨 수별 지도 수열 그림(레나 델타, 알래스카). 모형 산출은 scripts/2_evaluation/label_sequence_v1.py 가 쓴 것을 읽기만 한다.

구성  1행 n = 0, 10, 40, 160, 전량의 ALT 지도(한 색 척도: 다섯 지도 합동 1–99 백분위를 5 cm 단위로 바깥 반올림, 승인된 oslo_r),
      2행 '전량 지도와의 차'(0 중심 broc, 전량이 아닌 네 지도의 |차| 99 백분위를 5 cm 단위로 올린 대칭 범위), 모든 지도에 그 n 의 라벨 셀을 검은 점으로,
      지도마다 아래에 라벨 수. RMSE 등 수치는 적지 않는다.
매체  논문판 170 mm(패널 문자 a–j, 7 pt), 슬라이드판 12.0 × 5.2 in(글자 12 pt 이상, 패널 제목에 방법 이름), GIF(프레임 n 마다 ALT + 차, 색 범위 고정,
      1.2 s/프레임, ffmpeg palettegen/paletteuse, 8 MB 이하)와 프레임 PNG.
그리기  maps_alt_v3 의 함수(원 격자 메시, 바탕 지도, 색표, 축척 막대)를 가져다 쓴다(그 모듈은 고치지 않는다).
산출  outputs/figures/paper/v3_restructure/maps/<Region>_label_sequence_v1.{pdf,png}, _legend.md, _source_values.json
      deck/assets/paper_report/maps/<Region>_label_sequence_slide.png, <Region>_label_sequence.gif, <Region>_label_sequence_frames/
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/label_sequence_fig.py --region lena|alaska|both
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import shutil                                                                                           # noqa: E402
import subprocess                                                                                       # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
import matplotlib.ticker as mticker                                                                     # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm                                                   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

PROC = ROOT / "data" / "processed"
OUT_PAPER = MV.OUT_PAPER
OUT_SLIDE = MV.OUT_SLIDE
TAGS = ("0", "10", "40", "160", "all")
METHOD_NAME = {"P0": "Source Stefan", "P1": "Recalibrated Stefan", "R1": "Anchor + residual ML"}
GRAT = {"lena": dict(x=[125, 129], y=[72, 73]), "alaska": dict(x=[-160, -150], y=[60, 65, 70])}
SEQ_DIR = {"lena": PROC / "map_lena" / "label_sequence_v1", "alaska": PROC / "map_alaska" / "label_sequence_v1"}
STEM = {"lena": "Lena_label_sequence", "alaska": "Alaska_label_sequence"}


# ================================================================ 자료
def load_sequence(key: str) -> dict:
    d = MV.prepare(MV.load_region(key))
    meta = json.loads((SEQ_DIR[key] / "label_sequence_v1_meta.json").read_text())
    from polar.m1_core import load_base
    lab = load_base(PROC).set_index("loc_id")
    preds, pts, steps = {}, {}, {}
    for st in meta["steps"]:
        tag = st["n"]
        p = pd.read_csv(SEQ_DIR[key] / f"pred_n{tag}.csv.gz", dtype={"cell_id": str})
        assert (p.cell_id.values == d["cells"].cell_id.values).all(), "셀 순서 불일치"
        preds[tag] = p.pred.values.astype(float)
        ids = [int(v) for v in st["cells_loc_id"]]
        sub = lab.loc[ids]
        pts[tag] = d["proj"].transform_points(MV.PC, sub.lon.values.astype(float), sub.lat.values.astype(float))[:, :2]
        steps[tag] = st
    shown = d["shown"].copy()
    for v in preds.values():
        shown &= np.isfinite(v)
    d["shown"] = shown
    ref = preds["all"]
    changes = {t: preds[t] - ref for t in TAGS}
    pooled_ch = np.concatenate([changes[t][shown] for t in TAGS if t != "all"])
    d.update(seq=preds, change=changes, pts=pts, steps=steps, seq_meta=meta,
             rng_seq=dict(alt=MV.pct_range(ref[shown]), ch_max=MV.sym_max(pooled_ch)))   # ALT 범위 = 전량 지도(e)의 1–99 백분위
    return d


def count_text(d, tag: str) -> str:
    st = d["steps"][tag]
    n = st["n_used"]
    return f"All {S.fmt_int(n)} labels" if tag == "all" else f"{n} labels"


# ================================================================ 그리기 도우미
def graticule_tl(ax, d, M, left=False, top=False):
    """경위선. 라벨은 왼쪽(위도)·위(경도)만: 지도 아래 띠를 라벨 수 글에 쓴다."""
    lab = {}
    if left:
        lab["left"] = "y"
    if top:
        lab["top"] = "x"
    g = GRAT[d["key"]]
    gl = ax.gridlines(crs=MV.PC, draw_labels=lab if lab else False, linewidth=M.lw_grat, color=S.BASEMAP["graticule"], xlocs=mticker.FixedLocator(g["x"]),
                      ylocs=mticker.FixedLocator(g["y"]), x_inline=False, y_inline=False, zorder=3)
    if lab:
        gl.rotate_labels = False
        gl.xlabel_style = dict(size=M.fs, rotation=0)
        gl.ylabel_style = dict(size=M.fs, rotation=0)
        gl.xpadding = 2 if M.paper else 4
        gl.ypadding = 2 if M.paper else 4
    return gl


def draw_alt(ax, d, tag, norm):
    MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["seq"][tag], MV.CM_ALT, norm))


def draw_change(ax, d, tag, norm):
    MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["change"][tag], MV.CM_DIFF, norm))


def draw_points(ax, d, tag, M, size=None):
    xy = d["pts"][tag]
    if len(xy) == 0:
        return
    s = size if size is not None else (2.0 if M.paper else 5.0)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c="#000000", edgecolors="#ffffff", linewidths=0.4 if M.paper else 0.6, transform=d["proj"], zorder=5)


def scale_bar_fig(fig, x_mm, y_top_mm, L_mm, km, M, gap_mm):
    """그림 좌표의 축척 막대: 길이 글을 위에, 막대를 그 아래(gap_mm)에 둔다(열쇠 칸 아래 빈 곳용)."""
    from matplotlib.lines import Line2D
    W, H = fig.get_size_inches() * 25.4
    t = fig.text((x_mm + L_mm / 2) / W, 1 - y_top_mm / H, f"{km:g} km", ha="center", va="top", fontsize=M.fs)
    t.set_gid("scale")
    yb = y_top_mm + gap_mm
    fig.add_artist(Line2D([x_mm / W, (x_mm + L_mm) / W], [1 - yb / H] * 2, color="#000000", lw=M.lw_scale, solid_capstyle="butt",
                          transform=fig.transFigure, gid="scale"))


def norms(d):
    lo, hi = d["rng_seq"]["alt"]
    v = d["rng_seq"]["ch_max"]
    return Normalize(lo, hi), TwoSlopeNorm(0.0, -v, v)


def alt_extend(d):
    lo, hi = d["rng_seq"]["alt"]
    sh = d["shown"]
    return MV.extend_of(np.concatenate([d["seq"][t][sh] for t in TAGS]), lo, hi)


def ch_extend(d):
    v = d["rng_seq"]["ch_max"]
    sh = d["shown"]
    return MV.extend_of(np.concatenate([d["change"][t][sh] for t in TAGS if t != "all"]), -v, v)


# ================================================================ 논문판
def fig_paper(d):
    M = MV.Med("paper")
    MV.use_medium("paper")
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    W, LM, GAP, KEYR, RM = 170.0, 8.5, 1.6, 12.5, 0.3
    mw = (W - LM - 4 * GAP - KEYR - RM) / 5
    mh = mw * asp
    TOP, CNT, ROWGAP, BOT = 6.5, 4.0, 4.2, 5.0
    H = TOP + mh + CNT + ROWGAP + mh + BOT
    fig = S.fig_mm(W, H)
    xs = [LM + j * (mw + GAP) for j in range(5)]
    ys = [TOP, TOP + mh + CNT + ROWGAP]
    n_alt, n_ch = norms(d)
    axes = {}
    letters = "abcdefghij"
    for i in range(2):
        for j, tag in enumerate(TAGS):
            ax = S.axes_mm(fig, xs[j], ys[i], mw, mh, projection=proj)
            MV.base_map(ax, d, ext, M)
            (draw_alt if i == 0 else draw_change)(ax, d, tag, n_alt if i == 0 else n_ch)
            draw_points(ax, d, tag, M)
            graticule_tl(ax, d, M, left=(j == 0), top=(i == 0 and j == 0))
            MV.letter(fig, xs[j], ys[i] - 0.5, letters[i * 5 + j], M)
            t = fig.text((xs[j] + mw / 2) / W, 1 - (ys[i] + mh + 1.0) / H, count_text(d, tag), ha="center", va="top", fontsize=M.fs)
            t.set_gid("category")
            axes[(i, j)] = ax
    lo, hi = d["rng_seq"]["alt"]
    v = d["rng_seq"]["ch_max"]
    cax = S.axes_mm(fig, xs[4] + mw + 1.5, ys[0] + 0.06 * mh, M.cbar, 0.74 * mh)
    MV.colorbar(fig, cax, MV.CM_ALT, n_alt, "ALT (cm)", MV.ticks_between(lo, hi), M, alt_extend(d), orientation="vertical")
    cax = S.axes_mm(fig, xs[4] + mw + 1.5, ys[1] + 0.06 * mh, M.cbar, 0.74 * mh)
    MV.colorbar(fig, cax, MV.CM_DIFF, n_ch, "Change vs all labels (cm)", MV.ticks_between(-v, v), M, ch_extend(d), orientation="vertical")
    L_mm = MV.bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    scale_bar_fig(fig, xs[4] + mw + 1.5, ys[1] + 0.86 * mh, L_mm, d["scale_km"], M, 3.3)     # 열쇠 칸, 차 색 막대 아래
    d.setdefault("layout", {})["paper"] = dict(size_mm=[W, round(H, 1)], map_mm=[round(mw, 1), round(mh, 1)],
                                              scale_bar=dict(place="key column, below change bar", length_mm=round(L_mm, 2)))
    return fig


# ================================================================ 슬라이드판
def fig_slide(d):
    M = MV.Med("slide")
    MV.use_medium("slide")
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    W, LM, GAP, KEYR, RM = MV.SLIDE_W_MM, 15.0, 4.0, 27.0, 1.0
    mw = (W - LM - 4 * GAP - KEYR - RM) / 5
    mh = mw * asp
    TOP, CNT, ROWGAP, BOT = 12.0, 7.0, 6.0, 7.0
    H = TOP + mh + CNT + ROWGAP + mh + BOT
    if H > MV.SLIDE_H_MAX_MM:                                           # 높이가 넘치면 지도를 줄인다
        mh = (MV.SLIDE_H_MAX_MM - TOP - CNT - ROWGAP - BOT) / 2
        mw = mh / asp
        H = MV.SLIDE_H_MAX_MM
        xs0 = LM + ((W - LM - KEYR - RM) - (5 * mw + 4 * GAP)) / 2
    else:
        xs0 = LM
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    xs = [xs0 + j * (mw + GAP) for j in range(5)]
    ys = [TOP, TOP + mh + CNT + ROWGAP]
    n_alt, n_ch = norms(d)
    axes = {}
    letters = "abcdefghij"
    for i in range(2):
        for j, tag in enumerate(TAGS):
            ax = S.axes_mm(fig, xs[j], ys[i], mw, mh, projection=proj)
            MV.base_map(ax, d, ext, M)
            (draw_alt if i == 0 else draw_change)(ax, d, tag, n_alt if i == 0 else n_ch)
            draw_points(ax, d, tag, M)
            graticule_tl(ax, d, M, left=(j == 0), top=False)
            title = METHOD_NAME[d["steps"][tag]["method"]] if i == 0 else None
            MV.letter(fig, xs[j], ys[i] - 1.5, letters[i * 5 + j], M, title)
            t = fig.text((xs[j] + mw / 2) / W, 1 - (ys[i] + mh + 1.2) / H, count_text(d, tag), ha="center", va="top", fontsize=M.fs)
            t.set_gid("category")
            axes[(i, j)] = ax
    lo, hi = d["rng_seq"]["alt"]
    v = d["rng_seq"]["ch_max"]
    cax = S.axes_mm(fig, xs[4] + mw + 2.5, ys[0] + 0.04 * mh, M.cbar, 0.76 * mh)
    MV.colorbar(fig, cax, MV.CM_ALT, n_alt, "ALT (cm)", MV.ticks_between(lo, hi), M, alt_extend(d), orientation="vertical")
    cax = S.axes_mm(fig, xs[4] + mw + 2.5, ys[1] + 0.04 * mh, M.cbar, 0.76 * mh)
    MV.colorbar(fig, cax, MV.CM_DIFF, n_ch, "Change vs all labels (cm)", MV.ticks_between(-v, v), M, ch_extend(d), orientation="vertical")
    L_mm = MV.bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    scale_bar_fig(fig, xs[4] + mw + 2.5, ys[1] + 0.85 * mh, L_mm, d["scale_km"], M, 5.5)     # 열쇠 칸, 차 색 막대 아래
    d.setdefault("layout", {})["slide"] = dict(size_in=[round(W / 25.4, 2), round(H / 25.4, 2)], map_mm=[round(mw, 1), round(mh, 1)],
                                              scale_bar=dict(place="key column, below change bar", length_mm=round(L_mm, 2)))
    return fig


# ================================================================ GIF 프레임
def fig_frame(d, tag, k):
    M = MV.Med("slide")
    MV.use_medium("slide")
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    W, H = 10.0 * 25.4, 5.6 * 25.4
    TOP, LM, GAP, CB, BOT = 19.0, 16.0, 12.0, 24.0, 2.0
    mh = H - TOP - CB - BOT
    mw = mh / asp
    if LM + 2 * mw + GAP + 4 > W:
        mw = (W - LM - GAP - 4) / 2
        mh = mw * asp
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    xs = [LM, LM + mw + GAP]
    n_alt, n_ch = norms(d)
    st = d["steps"][tag]
    head = f"{d['name']} · {count_text(d, tag)} · {METHOD_NAME[st['method']]}"
    fig.text(xs[0] / W, 1 - 4.0 / H, head, fontsize=M.fs_lab, ha="left", va="top")
    fig.text((xs[1] + mw) / W, 1 - 4.0 / H, f"{k + 1}/{len(TAGS)}", fontsize=M.fs, ha="right", va="top", color=MV.GRAY_TXT)
    for j, (title, fn, nm) in enumerate((("ALT", draw_alt, n_alt), ("Change vs all labels", draw_change, n_ch))):
        ax = S.axes_mm(fig, xs[j], TOP, mw, mh, projection=proj)
        MV.base_map(ax, d, ext, M)
        fn(ax, d, tag, nm)
        draw_points(ax, d, tag, M, size=9.0)
        graticule_tl(ax, d, M, left=(j == 0), top=False)
        MV.letter(fig, xs[j], TOP - 1.5, "ab"[j], M, title)
        cax = S.axes_mm(fig, xs[j] + 0.1 * mw, TOP + mh + 8.0, 0.8 * mw, M.cbar)
        if j == 0:
            lo, hi = d["rng_seq"]["alt"]
            MV.colorbar(fig, cax, MV.CM_ALT, n_alt, "ALT (cm)", MV.ticks_between(lo, hi), M, alt_extend(d))
        else:
            v = d["rng_seq"]["ch_max"]
            MV.colorbar(fig, cax, MV.CM_DIFF, n_ch, "Change vs all labels (cm)", MV.ticks_between(-v, v), M, ch_extend(d))
    L_mm = MV.bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    MV.scale_bar_below(fig, xs[0], TOP + mh + 3.5, L_mm, d["scale_km"], M)
    return fig


def make_gif(frames: list[Path], gif: Path, frame_s: float = 1.2) -> dict:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = gif.with_suffix(".frames.txt")
    tmp.write_text("".join(f"file '{p.resolve()}'\nduration {frame_s}\n" for p in frames))
    vf = "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp), "-vf", vf, "-r", f"{1.0 / frame_s:.6f}", "-loop", "0", str(gif)]
    subprocess.run(cmd, check=True)
    tmp.unlink()
    from PIL import Image
    im = Image.open(gif)
    durs = []
    try:
        while True:
            durs.append(im.info.get("duration"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return dict(path=str(gif.relative_to(ROOT)), bytes=int(gif.stat().st_size), n_frames=len(durs), durations_ms=durs, size_px=list(im.size))


# ================================================================ 기록
def overlaps_visible(fig) -> dict:
    """style.text_overlaps 에서 GeoAxes 의 숨은 축 눈금 글(축이 꺼져 있어 그려지지 않는다)을 뺀다."""
    import itertools
    from matplotlib.text import Text
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                hidden.update(axis.get_ticklabels(which="both"))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]
    bbs = [(t.get_text().strip(), t.get_window_extent(rend)) for t in texts]
    pairs = [(a[:25], b[:25]) for (a, ba), (b, bb) in itertools.combinations(bbs, 2) if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    outside = [s[:25] for s, b in bbs if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    return dict(n_texts=len(bbs), overlap_pairs=pairs, outside=outside)


def legend_md(d) -> str:
    m = d["seq_meta"]
    st = d["steps"]
    lo, hi = d["rng_seq"]["alt"]
    v = d["rng_seq"]["ch_max"]
    n_all = st["all"]["n_used"]
    reg = d["name"]
    lam = {t: st[t]["lam"] for t in ("40", "160", "all")}
    of = "the Lena Delta" if d["key"] == "lena" else "Alaska"
    txt = (f"# {reg} label-count map sequence\n\n"
           f"**1 km active-layer thickness (ALT) of {of} as the number of label cells used by the workflow grows.** "
           f"a–e, ALT with 0, 10, 40, 160 and all {S.fmt_int(n_all)} label cells ({m['labels']['n_blocks']} blocks of 0.5°); "
           f"f–j, each map minus the all-label map (e). Label cells (black points) are the first n cells of one block-dispersed order "
           f"(blocks and cells permuted with seed {m['seed']}, one cell per block in turn), so each step keeps the previous cells. "
           f"0 labels: source-coefficient Stefan E0·√TDD, E0 = {st['0']['E0']:.3f} from {S.fmt_int(m['model']['E0_source']['n_src'])} source cells "
           f"(other regions). 10 labels: recalibrated Stefan, E = {st['10']['E_n']:.3f} (least squares {st['10']['E_ls']:.3f} shrunk towards E0, κ = 10). "
           f"40, 160 and all labels: recalibrated Stefan anchor plus λ·g, g a CatBoost residual model on the 25 covariates (two seeds), "
           f"λ from 0.25, 0.5 and 1.0 by 0.5° block five-fold cross-validation within the labels "
           f"(E = {st['40']['E_n']:.3f}, {st['160']['E_n']:.3f}, {st['all']['E_n']:.3f}; λ = {lam['40']:g}, {lam['160']:g}, {lam['all']:g}). "
           f"Colour scales: ALT {lo:.0f}–{hi:.0f} cm (1st–99th percentile of e; arrows mark values beyond the ends), change ±{v:.0f} cm "
           f"(99th percentile of |change| over a–d). White, water; grey, no data. The 1 km grid is a display resolution; climate inputs are ERA5-Land 0.1°.\n")
    return txt


def source_values(d) -> dict:
    sh = d["shown"]
    m = d["seq_meta"]
    out = dict(region=d["name"], seed=m["seed"], draw_rule=m["draw_rule"], n_cells=int(len(sh)), n_shown=int(sh.sum()),
               n_labels_total=m["labels"]["n"], n_blocks_total=m["labels"]["n_blocks"], E0=m["model"]["E0"], kappa=m["model"]["kappa"],
               ranges=dict(alt=d["rng_seq"]["alt"], change=[-d["rng_seq"]["ch_max"], d["rng_seq"]["ch_max"]]), steps={}, layout=d.get("layout", {}))
    for t in TAGS:
        st = d["steps"][t]
        v = d["seq"][t][sh]
        c = d["change"][t][sh]
        out["steps"][t] = dict(n_used=st["n_used"], n_blocks=st["n_blocks"], method=st["method"], E_ls=st["E_ls"], E_n=st["E_n"], lam=st["lam"],
                               lam_cv_folds=st["lam_cv_folds"], lam_cv_flag=st["lam_cv_flag"], fit_s=st["fit_s"],
                               alt=dict(p01=float(np.percentile(v, 1)), p50=float(np.median(v)), p99=float(np.percentile(v, 99)), mean=float(v.mean())),
                               change=dict(mean=float(c.mean()), mean_abs=float(np.abs(c).mean()), p01=float(np.percentile(c, 1)),
                                           p99=float(np.percentile(c, 99))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both")
    ap.add_argument("--what", default="paper,slide,gif")
    a = ap.parse_args(argv)
    regions = ["lena", "alaska"] if a.region == "both" else [a.region]
    what = set(a.what.split(","))
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    OUT_SLIDE.mkdir(parents=True, exist_ok=True)
    for key in regions:
        t0 = time.time()
        d = load_sequence(key)
        stem = STEM[key]
        rec = dict(ranges=d["rng_seq"])
        if "paper" in what:
            fig = fig_paper(d)
            S.save_fig(fig, f"{stem}_v1", OUT_PAPER, formats=("pdf", "png"))
            au = S.audit_v3(fig)
            ov = overlaps_visible(fig)
            rec["paper_audit"] = dict(sizes=sorted(au["sizes"]), chars=au["chars"], thin=au["thin_lines"][:4], overlaps=ov["overlap_pairs"][:6],
                                      outside=ov["outside"][:6])
            plt.close(fig)
        if "slide" in what:
            fig = fig_slide(d)
            fig.savefig(OUT_SLIDE / f"{stem}_slide.png", dpi=300)
            ov = overlaps_visible(fig)
            rec["slide_audit"] = dict(overlaps=ov["overlap_pairs"][:6], outside=ov["outside"][:6], size_in=d["layout"]["slide"]["size_in"])
            plt.close(fig)
        if "gif" in what:
            fdir = OUT_SLIDE / f"{stem}_frames"
            if fdir.exists():
                shutil.rmtree(fdir)
            fdir.mkdir(parents=True)
            frames = []
            for k, tag in enumerate(TAGS):
                fig = fig_frame(d, tag, k)
                p = fdir / f"frame_{k}_n{tag}.png"
                fig.savefig(p, dpi=160)
                plt.close(fig)
                frames.append(p)
            rec["gif"] = make_gif(frames, OUT_SLIDE / f"{stem}.gif")
        if "paper" in what:
            sv = source_values(d)
            sv.update(audit=rec)
            (OUT_PAPER / f"{stem}_v1_source_values.json").write_text(json.dumps(sv, ensure_ascii=False, indent=1, default=str))
            (OUT_PAPER / f"{stem}_v1_legend.md").write_text(legend_md(d))
        print(f"[{key}] {json.dumps(rec, ensure_ascii=False, default=str)[:1500]} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
