"""덱 슬라이드용 패널 재렌더(2026-10-05). 보고 덱이 논문 그림을 잘라 쓰던 자리(5, 11, 21, 24, 36쪽)와 표지 그림.

원칙
  같은 그리기 함수와 같은 자료를 쓴다(fig1.py, fig5.py, fig6.py, fig7.py, maps_alt_v3.py). 이 프로세스 안에서만 글자 크기(S.FONT_PT),
  선 굵기(S.LW), 표지 크기(S.MS, 원 면적 상수)와 배치 상수를 바꾼다. 그림 스크립트 파일은 고치지 않는다. 새 자료와 새 주장은 없다.
  글자는 슬라이드 크기에서 12 pt 이상(Pretendard), 선은 1.5 pt 이상, 표지는 배율만큼 크게 한다.
두 방식
  (S) 단독 배치: 패널 함수를 슬라이드 배치 상자 크기의 그림에 직접 그린다(Fig 1a, 1b, 1e, 6a, 6b, 6 컬러바, 7b, 7c, 7e, 표지 지도).
  (P) 논문 배치 + 같은 자르기: 그림 전체를 논문 배치로 다시 그리고 덱의 자르기 상자(deck/paper_report_lib.py 의 좌표를 옮겨 적음)로
      잘라 배치 상자 크기(300 dpi)로 맞춘다(Fig 5c: 자르기 상자 안에 글자가 없다). Fig 6 은 P 로 시작했으나 큰 글자가 상자 밖으로 나가 S 로 바꿨다.
산출  deck/assets/paper_report/slide_panels/<figure>_<panel>_slide.png, README.md(쪽, 대체하는 자르기, 새 파일, 크기)
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/slide_panels.py [--only title,fig1a,fig1b,fig1e,fig7b,fig7c,fig7e,fig5c,fig6]
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
from matplotlib.colors import Normalize                                                                 # noqa: E402
from matplotlib.lines import Line2D                                                                     # noqa: E402
from matplotlib.patches import Rectangle                                                                # noqa: E402
from PIL import Image                                                                                   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

OUT = ROOT / "deck" / "assets" / "paper_report" / "slide_panels"
DPI = 300
FONT = 14.0
MM = 25.4
SLIDE_BLOCK_OFFSET_MM = 1.6     # 포레스트 블록 등가중 막대의 어긋남(슬라이드, 논문 0.8 mm): 3 pt 셀 막대와 1.5 pt 막대 사이 약 0.8 mm
_LW0, _MS0, _FONT0, _TICK0 = dict(S.LW), dict(S.MS), S.FONT_PT, S.TICK_LEN_PT
_USE_V3 = S.use_v3                                                             # 원래 함수(P 방식에서 S.use_v3 를 잠시 바꾼다)
RECORD: list[dict] = []

# 덱의 자르기 상자(2000 px 폭 기준, deck/paper_report_lib.py CROPS·FIG6_CROPS 의 값을 옮겨 적음)
CROP = {"f6a": (0, 40, 965, 927), "f6b": (1025, 40, 1995, 927), "f6_cbar": (105, 930, 1672, 1054), "f5c": (1018, 44, 1492, 520)}


# ================================================================ 공용
def slide_rc(font=FONT, paper_geometry=False):
    """Pretendard, 글자 font pt, 축선 1.5 pt. paper_geometry=True 는 (P) 방식(논문 배치에 키운 글자, 선은 S.LW 로)."""
    MV.register_pretendard()
    _USE_V3("slide")
    lw_ax = 1.5 if not paper_geometry else max(1.0, 1.5 / 1.3)
    matplotlib.rcParams.update({
        "font.size": font, "axes.labelsize": font, "xtick.labelsize": font, "ytick.labelsize": font, "legend.fontsize": font,
        "axes.titlesize": font, "font.weight": "medium", "axes.linewidth": lw_ax, "xtick.major.width": lw_ax, "ytick.major.width": lw_ax,
        "xtick.major.size": 5.0 if not paper_geometry else 3.0, "ytick.major.size": 5.0 if not paper_geometry else 3.0,
        "mathtext.fontset": "custom", "mathtext.rm": "Pretendard", "mathtext.it": "Pretendard", "mathtext.bf": "Pretendard:bold",
        "mathtext.sf": "Pretendard", "axes.unicode_minus": True, "hatch.linewidth": 1.5 if not paper_geometry else 1.2,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3.0, "lines.solid_capstyle": "butt",
    })
    S.FONT_PT = font


def slide_scales(lw_mult=1.5, lw_min=1.5, ms_mult=1.6):
    S.LW.clear(); S.LW.update({k: max(lw_min, v * lw_mult) for k, v in _LW0.items()})
    S.MS.clear(); S.MS.update({k: v * ms_mult for k, v in _MS0.items()})
    S.TICK_LEN_PT = 5.0


def record(name, pages, replaces, size_in, how, note=""):
    RECORD.append(dict(file=name, pages=pages, replaces=replaces, size_in=[round(size_in[0], 2), round(size_in[1], 2)], how=how, note=note))


def check_text(fig, name):
    """글자끼리 겹침과 캔버스 밖 글자(style.text_overlaps)와 가장 작은 글자 크기(pt, 그림 크기 기준)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    sizes = sorted({round(t.get_fontsize(), 1) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()})
    ov = S.text_overlaps(fig) if hasattr(S, "text_overlaps") else {}
    print(f"[check] {name}: 글자 크기 {sizes} · 겹침 {ov.get('overlap_pairs', '?')} · 캔버스 밖 {ov.get('outside', '?')}", flush=True)
    return sizes, ov


def save_s(fig, name, size_in, pages, replaces, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    sizes, ov = check_text(fig, name)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    record(name, pages, replaces, size_in, "S", note + (f"; text sizes {sizes} pt" if sizes else "; no text in this panel (as in the paper)"))


def crop_resize(src_png, box2000, size_in, name, pages, replaces, eff_pt, note=""):
    """(P) 방식: 논문 배치로 그린 큰 글자 PNG 를 덱과 같은 상자로 잘라 배치 상자 크기(300 dpi)로 맞춘다."""
    with Image.open(src_png) as im0:
        im = im0.convert("RGB")
    f = im.size[0] / 2000.0
    b = tuple(int(round(v * f)) for v in box2000)
    b = (max(0, b[0]), max(0, b[1]), min(im.size[0], b[2]), min(im.size[1], b[3]))
    c = im.crop(b)
    w_px = int(round(size_in[0] * DPI))
    h_px = int(round(w_px * c.size[1] / c.size[0]))
    c = c.resize((w_px, h_px), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    c.save(OUT / name, dpi=(DPI, DPI))
    record(name, pages, replaces, (w_px / DPI, h_px / DPI), "P", note + (f"; text about {eff_pt:.1f} pt at slide size" if eff_pt else ""))


def restore():
    S.LW.clear(); S.LW.update(_LW0); S.MS.clear(); S.MS.update(_MS0); S.FONT_PT = _FONT0; S.TICK_LEN_PT = _TICK0


# ================================================================ Fig 1a, 1b (S)
def _fig1():
    import fig1 as F1
    return F1


def fig1_key(fig, F1, x_mm, y_mm, font):
    """Fig 1a 열쇠(같은 부호화): 줄 1 = 원 면적 열쇠 1, 10, 50 과 새 지역(흰 원), 줄 2 = 영구동토 견본 2개."""
    W, H = fig.get_size_inches() * MM
    ax = S.axes_mm(fig, 0, 0, W, H)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off(); ax.patch.set_visible(False)
    rend = fig.canvas.get_renderer()

    def put(x, y, s_):
        t = ax.text(x, y, s_, ha="left", va="center", fontsize=font)
        fig.canvas.draw()
        bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
        return max(bb.x0, bb.x1)
    base = y_mm
    x = x_mm
    for v in F1.SIZE_KEY:
        r = F1.pt2mm(np.sqrt(F1.size_pt2(v)) / 2)
        ax.scatter([x + r], [base - r], s=F1.size_pt2(v), facecolors=[F1.FILL_OBS], edgecolors=S.INK, linewidths=F1.EDGE_LW, zorder=3)
        x = put(x + 2 * r + 1.0, base - 1.8, f"{v}") + 2.6
    r = F1.pt2mm(np.sqrt(F1.size_pt2(F1.SIZE_KEY[1])) / 2)
    x += 1.0
    ax.scatter([x + r], [base - r], s=F1.size_pt2(F1.SIZE_KEY[1]), facecolors="white", edgecolors=S.INK, linewidths=F1.EDGE_LW, zorder=3)
    put(x + 2 * r + 1.0, base - 1.8, "New regions")
    y2, sw = base + 6.0, 3.6
    x = x_mm
    for lab, col in (("Continuous", S.BASEMAP["continuous"]), ("Discontinuous", S.BASEMAP["discontinuous"])):
        ax.add_patch(Rectangle((x, y2 - sw / 2), sw, sw, facecolor=col, edgecolor="none", linewidth=0, zorder=2))
        x = put(x + sw + 1.2, y2, lab) + 4.0
    return ax


def _circles_px(ax):
    """지도 축의 블록 원(PathCollection)의 화면 중심과 반지름(px)."""
    from matplotlib.collections import PathCollection
    cs, rs = [], []
    dpi = ax.figure.dpi
    for c in ax.collections:
        if isinstance(c, PathCollection) and len(c.get_offsets()):
            xy = c.get_offset_transform().transform(c.get_offsets())
            sz = np.asarray(c.get_sizes(), float)
            r = np.sqrt(sz if len(sz) == len(xy) else np.full(len(xy), sz[0])) / 2 * dpi / 72
            cs.append(xy); rs.append(r)
    return (np.vstack(cs), np.concatenate(rs)) if cs else (np.zeros((0, 2)), np.zeros(0))


def _box_hits(bb, C, R, pad=5.0):
    return int(((C[:, 0] + R > bb.x0 - pad) & (C[:, 0] - R < bb.x1 + pad) & (C[:, 1] + R > bb.y0 - pad) & (C[:, 1] - R < bb.y1 + pad)).sum())


def free_lat_labels_and_scale(fig, ax, proj, F1, vals):
    """위도 라벨(경선 후보)과 1000 km 축척 막대(자리 후보)를 블록 원·다른 글자와 겹치지 않는 첫 자리로 옮긴다(14 pt 글자)."""
    import cartopy.crs as ccrs
    from matplotlib.transforms import offset_copy
    pc = ccrs.PlateCarree()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    C, R = _circles_px(ax)
    labs = [t for t in ax.texts if t.get_gid() == "graticule"]
    fixed = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and t not in labs
             and t.get_gid() != "scale"]

    def bbs(ts):
        fig.canvas.draw()
        return [t.get_window_extent(rend) for t in ts]
    fixed_bb = bbs(fixed)
    chosen_lon = {}
    for t, la in zip(labs, (60, 70, 80)):                  # 라벨마다 따로: 원·다른 글자와 겹치지 않는 첫 경선
        best_l = None
        for lon in (-40, -30, -50, -20, -60, -10, -70, 0, -80, 10, 20):
            t.xy = proj.transform_point(lon, la, pc)
            b = bbs([t])[0]
            hits = _box_hits(b, C, R) + sum(b.overlaps(f) for f in fixed_bb)
            if best_l is None or hits < best_l[0]:
                best_l = (hits, lon)
            if hits == 0:
                break
        t.xy = proj.transform_point(best_l[1], la, pc)
        fixed_bb.append(bbs([t])[0])
        chosen_lon[la] = best_l[1] if best_l[0] == 0 else f"{best_l[1]} (hits {best_l[0]})"
    vals["slide_lat_label_lon"] = chosen_lon
    bar0 = ax.lines[-1]                                    # panel_a 의 마지막 선 = 축척 막대(scale_bar 가 마지막에 그린다)
    assert len(bar0.get_xdata()) == 2 and abs(bar0.get_linewidth() - S.LW["scale_bar"]) < 1e-9
    bar0.remove()
    for t in [t for t in ax.texts if t.get_gid() == "scale"]:
        t.remove()
    best = None
    for la in (70, 65, 60, 75, 55):
        for lon in range(-80, 61, 10):
            F1.scale_bar(ax, proj, lon, la, 1000)
            t = [t_ for t_ in ax.texts if t_.get_gid() == "scale"][-1]
            t.set_transform(offset_copy(t.get_transform(), fig=fig, y=3.0, units="points"))
            ln = ax.lines[-1]
            fig.canvas.draw()
            bt, bl_ = t.get_window_extent(rend), ln.get_window_extent(rend)
            axbb = ax.get_window_extent(rend)
            inside = bt.x0 > axbb.x0 and bt.x1 < axbb.x1 and bt.y1 < axbb.y1
            hits = _box_hits(bt, C, R) + _box_hits(bl_, C, R) + sum(bt.overlaps(f) or bl_.overlaps(f) for f in fixed_bb)
            if hits == 0 and inside:
                best = (lon, la); break
            t.remove(); ln.remove()
        if best:
            break
    vals["slide_scale_lonlat"] = best
    print(f"  [fig1a] 위도 라벨 경선 {chosen_lon}, 축척 막대 자리 {best}", flush=True)


def fig1a():
    F1 = _fig1()
    slide_rc(FONT); slide_scales()
    W_in, H_in = 5.02, 5.20
    Wm, Hm = W_in * MM, H_in * MM
    dia = 100.0
    F1.A_MAP = ((Wm - dia) / 2, 7.0, dia)
    F1.SIZE_K = 5.0 * (dia / 84.0) ** 2
    F1.EDGE_LW = 1.5
    F1.TIBET_AX = (Wm - 27.5 - 0.6, Hm - 20.5 - 3.6, 27.5, 20.5)            # 오른쪽 아래 빈 구석(원 밖, 열쇠 오른쪽)
    F1.LAT_LABEL_LON = -40.0                                                   # 위도 라벨: 그린란드 빙상 위(라벨 셀 없음)
    F1.SCALE_A = dict(lon=-28.0, lat=61.0, km=1000)                            # 축척 막대: 아이슬란드 남서 바다(라벨 셀 없음)
    F1.REGION_LABELS = dict(F1.REGION_LABELS)
    F1.REGION_LABELS["Alaska"] = ("Alaska", (0.99, 0.10))
    vals = {}
    D = F1.load_all(); P = F1.pfr_classes()
    L = F1.lena_design(vals)
    _, _, lon_c, _ = F1.zoom_projection(L)
    fig = plt.figure(figsize=(W_in, H_in))
    ax_a, proj_a = F1.panel_a(fig, D, vals, P, lon_c if F1.LON0_A is None else F1.LON0_A)
    ax_t = F1.panel_tibet(fig, D, vals, P, ax_a, proj_a)
    for t in ax_t.texts:                                   # 이름은 삽도 오른쪽 끝에 맞춘다(캔버스 오른쪽 밖으로 나가지 않게)
        if t.get_text() == "Tibetan Plateau":
            t.set_position((1.0, 1.0)); t.set_ha("right")
    from matplotlib.transforms import offset_copy
    for t in ax_t.texts:                                   # 삽도 축척 길이 글자: 막대 왼쪽 끝에 맞추고(14 pt 글자가 틀 왼쪽 밖으로 나가지 않게) 3 pt 띄운다
        if t.get_gid() == "scale":
            bar = [l_ for l_ in ax_t.lines if len(l_.get_xdata()) == 2 and abs(l_.get_linewidth() - S.LW["scale_bar"]) < 1e-9][-1]
            t.set_position((float(min(bar.get_xdata())), float(bar.get_ydata()[0])))
            t.set_ha("left")
            t.set_transform(offset_copy(t.get_transform(), fig=fig, y=3.0, units="points"))
    free_lat_labels_and_scale(fig, ax_a, proj_a, F1, vals)
    fig1_key(fig, F1, x_mm=(Wm - dia) / 2 + 4.0, y_mm=7.0 + dia + 8.6, font=FONT)
    save_s(fig, "Fig1_a_slide.png", (W_in, H_in), "5, 24 (also 1, 2, 37 if wanted)", "crops/f1a.png",
           "Lena Delta zoom rectangle and connector to c and d omitted (c, d are not on these slides)")
    restore()


def fig1b():
    F1 = _fig1()
    slide_rc(FONT); slide_scales()
    W_in, H_in = 4.05, 5.20
    Wm, Hm = W_in * MM, H_in * MM
    F1.B_AX = (39.0, 5.0, Wm - 39.0 - 3.5, Hm - 5.0 - 23.0)
    F1.E_TICKS = (0.1, 1, 10)
    name0 = dict(S.REGION_NAME)
    S.REGION_NAME.update({"Russia_C_LGD": "Central Russia\n(expanded)"})
    vals = {}
    D = F1.load_all()
    fig = plt.figure(figsize=(W_in, H_in))
    ax = F1.panel_b(fig, D, vals)
    ax.set_xlabel("Stefan coefficient, E\n(cm per √(°C d))", linespacing=1.1)
    for t in ax.get_yticklabels():
        t.set_linespacing(1.0)
    S.REGION_NAME.clear(); S.REGION_NAME.update(name0)
    save_s(fig, "Fig1_b_slide.png", (W_in, H_in), "5", "crops/f1b.png", "x ticks 0.1, 1, 10 (paper also 0.3, 3, 30)")
    restore()


def fig1e():
    """8쪽 '검증 사다리': Fig 1e(XH + 지역 홀드아웃)를 7.5 × 4.6 in 에 단독으로. panel_e 가 그린 자료 표지는 그대로 두고
    직접 라벨(단 이름 5개, 방법 이름 2개)만 슬라이드 크기에 맞춰 다시 둔다. 지역 점은 논문처럼 옅게(alpha 0.5)."""
    F1 = _fig1()
    slide_rc(FONT); slide_scales(lw_mult=2.0, lw_min=1.5, ms_mult=2.0)
    W_in, H_in = 7.5, 4.6
    Wm, Hm = W_in * MM, H_in * MM
    F1.E_AX = (21.0, 2.5, Wm - 21.0 - 4.0, Hm - 2.5 - 22.5)
    vals = {}
    fig = plt.figure(figsize=(W_in, H_in))
    ax = F1.panel_e(fig, vals)
    for t in [t for t in ax.texts if t.get_gid() == "direct_label"]:
        t.remove()
    Mn = vals["xh"]["means"]
    xg = Mn[Mn.method == "D0w"].set_index("stage").x_geo
    y_lab = {"W1R": 7.4, "W1S": 7.4, "W1B": 7.4, "W1K": 10.6, "V-G": 8.6}     # kNNDM 은 Block 과 가로로 붙어 한 단 위로
    for stg, lab in F1.XH_STAGES:
        ax.text(xg[stg], y_lab[stg], "Region\nholdout" if stg == "V-G" else lab, ha="center", va="center", fontsize=FONT,
                linespacing=1.0, gid="direct_label", zorder=6)
    ax.text(0.15, 12.2, "Direct ML", ha="center", va="center", fontsize=FONT, gid="direct_label", zorder=6)
    ax.text(0.15, 25.6, "Recalibrated Stefan", ha="center", va="center", fontsize=FONT, gid="direct_label", zorder=6)
    W, H = fig.get_size_inches() * MM
    x_ax = F1.E_AX[0]
    fig.text(x_ax / W, 0.8 / H, "Open symbol: Stefan with source coefficient (region holdout)", ha="left", va="bottom",
             fontsize=13.0, color=S.INK)
    # 지역 기호 열쇠(옅은 기호의 모양 = 지역, 논문 S.REGION_MARKER): 축 왼쪽 위 빈 곳, 13 pt
    rend = fig.canvas.get_renderer()
    aw = F1.E_AX[2]
    x_mm, y_fr = 3.0, 0.94
    for reg in F1.XH_REGIONS:
        ax.plot([(x_mm + 1.2) / aw], [y_fr], ls="none", marker=S.REGION_MARKER[reg], ms=S.MS["region_point"], color=S.INK_AUX,
                alpha=0.6, mew=0, transform=ax.transAxes, clip_on=False, zorder=6)
        t = ax.text((x_mm + 3.4) / aw, y_fr, S.REGION_NAME.get(reg, reg), transform=ax.transAxes, ha="left", va="center",
                    fontsize=13.0, zorder=6, gid="region_key")
        fig.canvas.draw()
        bb = t.get_window_extent(rend)
        x_mm = (bb.x1 / fig.dpi * MM - F1.E_AX[0]) + 4.0
    _thicken(fig)
    for ln in ax.lines:                                                      # 빈 기호(원천 계수 Stefan) 테두리: 평균 2.0 pt, 지역 1.5 pt
        if ln.get_markerfacecolor() == "white":
            ln.set_markeredgewidth(2.0 if ln.get_markersize() >= S.MS["main"] - 1e-9 else 1.5)
    save_s(fig, "Fig1_e_slide.png", (W_in, H_in), "8", "deck/assets/paper_report/S08_validation_ladder.png (old five-region chart)",
           "Fig 1e data and encodings (XH 24 rows + region holdout 6 rows); stage and method names as direct labels; region symbols "
           "alpha 0.5; one-line note on the open symbol")
    restore()


# ================================================================ Fig 7b, 7c (S)
def _fig7():
    import fig7 as F7
    return F7


def fig7b(W_in, H_in, name, pages, font):
    F7 = _fig7()
    slide_rc(font); slide_scales()
    F7.FS = font
    matplotlib.rcParams.update({"axes.facecolor": "none", "savefig.pad_inches": 0.0})
    Wm, Hm = W_in * MM, H_in * MM
    top = 7.5 if font >= 14 else 6.5                                         # 위 틀 바깥 경도 눈금 라벨
    cb_block = 21.0 if font >= 14 else 18.0
    mh = Hm - top - cb_block - 1.0
    F7.L["b_map"] = (1.0, top, Wm - 1.0 - (18.0 if font >= 14 else 15.5), mh)  # 오른쪽 틀 바깥 위도 눈금 라벨 칸
    F7.L["b_cbar"] = (Wm * 0.18, top + mh + 3.0, Wm * 0.62, 3.5)
    B, _ = F7.load_b()
    obs, _ = F7.load_obs()
    fig = plt.figure(figsize=(W_in, H_in))
    ax, info = F7.draw_b(fig, B, obs)
    for ln in ax.lines:                                                      # 관측 위치 점(논문 1.5 pt)을 배율만큼
        if ln.get_marker() == "o":
            ln.set_markersize(2.4)
    # 축척 막대: 논문 자리(틀 위끝에서 5 mm 아래)는 큰 글자가 틀 선에 걸린다. 막대를 틀 아래 8 mm(14 pt) 또는 7 mm(12 pt)로, 글자는 막대 위 1 mm
    x0_, x1_ = ax.get_xlim(); y0_, y1_ = ax.get_ylim()
    mpm = (x1_ - x0_) / (ax.get_position().width * W_in * MM)
    bar = [l_ for l_ in ax.lines if len(l_.get_xdata()) == 2 and abs(l_.get_linewidth() - S.LW["scale_bar"]) < 1e-9][-1]
    t_s = [t_ for t_ in ax.texts if t_.get_gid() == "scale"][0]
    sy = y1_ - (8.0 if font >= 14 else 7.0) * mpm
    bar.set_ydata([sy, sy])
    t_s.set_position((float(min(bar.get_xdata())), sy + 1.0 * mpm))         # 글자는 막대 왼쪽 끝에 맞춤(틀 왼쪽 선과 띄움)
    t_s.set_ha("left")
    save_s(fig, name, (W_in, H_in), pages, "crops/f7b.png", f"same blocks, colour scale ±{info['vmax']:g} cm")
    restore()


def fig7c():
    F7 = _fig7()
    slide_rc(FONT); slide_scales()
    off0 = S.FOREST_BLOCK_OFFSET_MM
    S.FOREST_BLOCK_OFFSET_MM = SLIDE_BLOCK_OFFSET_MM                         # 3 pt 셀 막대와 1.5 pt 블록 막대가 붙지 않게
    F7.FS = FONT
    matplotlib.rcParams.update({"axes.facecolor": "none"})
    W_in, H_in = 4.92, 4.80
    Wm, Hm = W_in * MM, H_in * MM
    lab_r = 44.0
    gap = 4.0
    wc = (Wm - lab_r - 2.5 - 2 * gap - 2.0) / 3
    F7.L.update(c_label_right=lab_r, c_x0=lab_r + 2.5, c_wc=wc, c_gap=gap, c_top=19.0, c_h=Hm - 19.0 - 34.0)
    C = F7.load_c()
    fig = plt.figure(figsize=(W_in, H_in))
    axes = F7.draw_c(fig, C)
    axes[1].set_xlabel("Error change vs\nrecalibrated Stefan (cm)", linespacing=1.1)
    for ax_ in axes:                                                         # 14 pt 에서 이웃 열 눈금 글자가 붙지 않게 −1, 0, 1 만
        F7.set_fixed_ticks(ax_.xaxis, [-1, 0, 1], [F7.num_tick(v) for v in (-1, 0, 1)])
    for t in fig.texts:                                                      # 'Three-region mean' 은 두 줄로(이름 열 폭)
        if t.get_text() == "Three-region mean":
            t.set_text("Three-region\nmean"); t.set_linespacing(1.0)
        if t.get_text() == "climate cells":                                  # 묶음 머리를 열 머리(14 pt) 위로
            x_, y_ = t.get_position(); t.set_position((x_, y_ + 5.5 / Hm))
    F7.key_line(fig, 3.0, Hm - 4.0, gap_mm=3.6, pad_mm=1.4, half_h=1.8)
    S.FOREST_BLOCK_OFFSET_MM = off0
    save_s(fig, "Fig7_c_slide.png", (W_in, H_in), "21", "crops/f7c.png",
           f"weighting key under the panel; block-equal bar {SLIDE_BLOCK_OFFSET_MM} mm below the cell-weighted bar (paper 0.8 mm)")
    restore()


def fig7e():
    """Fig 7e(XC 워크플로 대비) 슬라이드판 6.0 × 4.0 in, 14 pt. fig7.draw_e(같은 자료·부호화)를 슬라이드 배치 상수로 부른다.
    위에 열쇠 두 줄(두 가중 막대와 ±0.5 cm 띠, 지역 모양), 아래에 축 이름. 지역 기호는 논문과 같은 alpha 0.5, 크기만 6 pt."""
    F7 = _fig7()
    slide_rc(FONT); slide_scales()
    F7.FS = FONT
    matplotlib.rcParams.update({"axes.facecolor": "none"})
    W_in, H_in = 6.0, 4.0
    Wm, Hm = W_in * MM, H_in * MM
    E, ER = F7.load_e()
    L0 = dict(F7.L)
    name_right, ax_x0, top, bot = 31.0, 33.0, 19.0, 21.0
    F7.L.update(e_ax=(ax_x0, top, Wm - ax_x0 - 3.0, Hm - top - bot))
    off0 = S.FOREST_BLOCK_OFFSET_MM
    S.FOREST_BLOCK_OFFSET_MM = SLIDE_BLOCK_OFFSET_MM                         # 3 pt 셀 막대와 1.5 pt 블록 막대가 붙지 않게
    try:
        fig = plt.figure(figsize=(W_in, H_in))
        F7.draw_e(fig, E, ER, fs=FONT, key_xy=(1.5, 11.6), ms_region=6.0, name_right=name_right, head_x=1.5, key_gap_mm=1.2, key_sep_mm=4.0)
        F7.key_line(fig, 1.5, 4.2, gap_mm=4.0, pad_mm=1.5, half_h=1.9)
    finally:
        F7.L.clear(); F7.L.update(L0)
        S.FOREST_BLOCK_OFFSET_MM = off0
    _thicken(fig)
    save_s(fig, "Fig7_e_slide.png", (W_in, H_in), "new slide (XC workflow contrast; no deck page yet)", "none (Fig 7e was a placeholder)",
           "Fig 7e data and encodings (XC-1 n 40, 160; XC-2 n 10, 40, 160; regions worse than recalibrated Stefan by > 0.5 cm as faint "
           "region symbols); weighting key and region key above the panel")
    restore()


# ================================================================ Fig 6a, 6b, 공유 컬러바 (S)
# (P) 방식은 큰 글자가 덱의 자르기 상자 밖으로 나가 잘렸다(b 의 열 머리 조각, a 의 삽도 이름, 컬러바 이름·화살표). 그래서 지도 두 장과
# 컬러바를 fig6.py 의 그리기 함수(polar_axes, draw_pfr, graticule, draw_targets, tibet_inset, alaska_inset, zoom_frame, lat_labels,
# scale_bar)로 배치 상자 크기에 단독으로 그린다. 기하 배율 F6_K = 덱의 지도 자르기 배율(1.32)과 같다(원 지름, 원 간격, 삽도 크기).
F6_FONT = 13.0
F6_K = 1.32
F6_MAP_D = 86.0        # 지도 원 지름(mm, 논문 70 mm): 왼쪽 위 구석의 13 pt 영구동토 열쇠 2줄이 원 테두리와 겹치지 않게 [판단]
F6_MAP_TOP = 3.5       # 원 위끝(mm)
F6_BOTTOM = 5.8        # 삽도 이름 줄(13 pt) 높이(mm): 삽도 아래끝 = 캔버스 높이 − 이 값
F6_TIBET_W = 25.0     # 티베트 삽도 폭(mm, 논문 17 mm × 1.32 = 22.4 mm 보다 넓게). 범위 반폭 520 km 는 같고 세로 범위가 그만큼 준다
F6_MAP_BOX = (4.26, 3.91)
F6_CBAR_BOX = (9.95, 0.79)
F6_PATCH = ("W_MM", "H_MM", "MAP_D_MM", "CIRCLE_D_MM", "CIRCLE_GAP_MM", "LEADER_MIN_MM", "AK_PAD_MM",
            "INSET_W_MM", "INSET_H_MM", "AK_INSET_W_MM", "AK_INSET_H_MM", "resolve_overlaps")


def _f6_patch(F6):
    """fig6 모듈 상수를 슬라이드 기하로(이 프로세스 안에서만). resolve_overlaps 의 기본 간격(정의 때 묶임)도 감싸서 바꾼다."""
    saved = {n: getattr(F6, n) for n in F6_PATCH}
    k = F6_K
    gap = saved["CIRCLE_GAP_MM"] * k
    ro = saved["resolve_overlaps"]
    F6.MAP_D_MM = F6_MAP_D
    F6.CIRCLE_D_MM = saved["CIRCLE_D_MM"] * k
    F6.CIRCLE_GAP_MM = gap
    F6.LEADER_MIN_MM = saved["LEADER_MIN_MM"] * k
    F6.AK_PAD_MM = saved["AK_PAD_MM"] * k
    F6.INSET_W_MM, F6.INSET_H_MM = F6_TIBET_W, saved["INSET_H_MM"] * k       # 티베트 삽도 폭: 막대 오른쪽 13 pt '200 km' 이 틀 안에 들도록
    F6.AK_INSET_W_MM, F6.AK_INSET_H_MM = saved["AK_INSET_W_MM"] * k, saved["AK_INSET_H_MM"] * k
    F6.resolve_overlaps = lambda P_mm, center_mm, r_map_mm, gap_=None, **kw: ro(P_mm, center_mm, r_map_mm, gap=gap)
    return saved


def _f6_unpatch(F6, saved):
    for n, v in saved.items():
        setattr(F6, n, v)


def _f6_data(F6):
    """build() 와 같은 자료와 같은 공유 정규화(티베트를 뺀 |Δ| 99 백분위를 5 cm 단위로 올림)."""
    from cmcrameri import cm as cmc
    from matplotlib.colors import TwoSlopeNorm
    _, M = F6.load_target_deltas()
    M = M.merge(F6.target_centroids(), on="name", how="left", validate="one_to_one")
    arc = M[~M.name.isin(F6.INSET_TARGETS)]
    v = np.abs(np.r_[arc["D0_1.0"].values, arc["D1_1.0"].values])
    vmax = 5.0 * math.ceil(float(np.percentile(v, 99)) / 5.0)
    lo = bool((M["D0_1.0"].min() < -vmax) or (M["D1_1.0"].min() < -vmax))
    hi = bool((M["D0_1.0"].max() > vmax) or (M["D1_1.0"].max() > vmax))
    ext = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[(lo, hi)]
    return dict(M=M, arc=arc, P=F6.pfr_classes(), norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), cmap=cmc.broc,
                vmax=vmax, ext=ext)


def _thicken(fig, lw_min=1.5):
    """선(Line2D), 선 모음(테두리 포함), 축 테두리 가운데 0 보다 크고 lw_min 보다 가는 것을 lw_min 으로(슬라이드 1.5 pt 규칙)."""
    from matplotlib.collections import Collection
    for ln in fig.findobj(Line2D):
        if 0 < ln.get_linewidth() < lw_min:
            ln.set_linewidth(lw_min)
        if ln.get_marker() not in (None, "None", "", " ") and 0 < ln.get_markeredgewidth() < lw_min and ln.get_markerfacecolor() == "white":
            ln.set_markeredgewidth(lw_min)
    for c in fig.findobj(Collection):
        lws = np.atleast_1d(np.asarray(c.get_linewidths(), float))
        if len(lws) and ((lws > 0) & (lws < lw_min)).any():
            c.set_linewidths(np.where((lws > 0) & (lws < lw_min), lw_min, lws))
    for ax in fig.axes:
        for sp in ax.spines.values():
            if sp.get_visible() and 0 < sp.get_linewidth() < lw_min:
                sp.set_linewidth(lw_min)


def _f6_free_labels(fig, ax, F6, vals):
    """위도 라벨(경선 후보)과 1000 km 축척 막대(자리 후보)를 대상 원·다른 글자와 겹치지 않는 첫 자리로(논문 자리 −10°, 12° E 70° N 부터)."""
    pc = F6.PC
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    C, R = _circles_px(ax)
    labs = [t for t in ax.texts if t.get_gid() == "graticule"]
    others = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and t not in labs
              and t.get_gid() != "scale"]
    fig.canvas.draw()
    fixed = [t.get_window_extent(rend) for t in others]
    chosen = {}
    for t, la in zip(labs, (60, 70, 80)):
        best = None
        for lon in (F6.LAT_LABEL_LON, -20, 0, -30, 10, -40, -50):
            t.xy = F6.PROJ.transform_point(lon, la, pc)
            fig.canvas.draw()
            b = t.get_window_extent(rend)
            hits = _box_hits(b, C, R, pad=4.0) + sum(b.overlaps(f) for f in fixed)
            if best is None or hits < best[0]:
                best = (hits, lon)
            if hits == 0:
                break
        t.xy = F6.PROJ.transform_point(best[1], la, pc)
        fig.canvas.draw()
        fixed.append(t.get_window_extent(rend))
        chosen[la] = best[1] if best[0] == 0 else f"{best[1]} (hits {best[0]})"
    old = [ln for ln in ax.lines if ln.get_gid() == "scale_bar"]
    for ln in old:
        ln.remove()
    for t in [t for t in ax.texts if t.get_gid() == "scale"]:
        t.remove()
    best = None
    from matplotlib.transforms import offset_copy
    for lon, la in [(F6.SCALE["lon"], F6.SCALE["lat"])] + [(lo, la_) for la_ in (70, 65, 75, 60) for lo in range(-60, 61, 6)]:
        info = F6.scale_bar(ax, F6.PROJ, lon, la, F6.SCALE["km"])
        t = info["text"]
        t.set_transform(offset_copy(t.get_transform(), fig=fig, y=2.5, units="points"))   # 13 pt 글자를 막대에서 2.5 pt 띄운다
        ln = [l_ for l_ in ax.lines if l_.get_gid() == "scale_bar"][-1]
        fig.canvas.draw()
        bt, bl = t.get_window_extent(rend), ln.get_window_extent(rend)
        hits = _box_hits(bt, C, R, pad=4.0) + _box_hits(bl, C, R, pad=4.0) + sum(bt.overlaps(f) or bl.overlaps(f) for f in fixed)
        if hits == 0:
            best = (lon, la)
            break
        t.remove(); ln.remove()
    vals.update(lat_label_lon=chosen, scale_lonlat=best)
    print(f"  [fig6a] 위도 라벨 경선 {chosen}, 축척 막대 자리 {best}", flush=True)


def _inset_scale_beside(fig, bar_gid, lift_mm=3.0, gap_mm=1.0):
    """삽도 축척 막대를 아래 틀에서 lift_mm 위로 올리고 길이 글자를 막대 위가 아니라 오른쪽(같은 높이)에 둔다."""
    for ax_ in fig.axes:
        lns = [l_ for l_ in ax_.lines if l_.get_gid() == bar_gid]
        if not lns:
            continue
        ln = lns[0]
        t = [t_ for t_ in ax_.texts if t_.get_gid() == "scale"][0]
        x0, x1 = ax_.get_xlim()
        y0, _ = ax_.get_ylim()
        mpm = (x1 - x0) / (ax_.get_position().width * fig.get_size_inches()[0] * MM)
        yb = y0 + lift_mm * mpm
        ln.set_ydata([yb, yb])
        t.set_position((max(ln.get_xdata()) + gap_mm * mpm, yb))
        t.set_ha("left"); t.set_va("center_baseline")
        return


def fig6_map(F6, Dd, letter):
    W_in, H_in = F6_MAP_BOX
    Wm, Hm = W_in * MM, H_in * MM
    F6.W_MM, F6.H_MM = Wm, Hm                                                # overlay_axes, zoom_frame 의 mm 좌표
    fig = plt.figure(figsize=(W_in, H_in))
    ov = F6.overlay_axes(fig)
    M, arc, P, norm, cmap = Dd["M"], Dd["arc"], Dd["P"], Dd["norm"], Dd["cmap"]
    col = F6.MAP_COLS[letter][0]
    main = arc[~arc.name.isin(F6.AK_TARGETS)].reset_index(drop=True)
    ak = M[M.name.isin(F6.AK_TARGETS)].set_index("name").loc[F6.AK_TARGETS].reset_index()
    D = F6.MAP_D_MM
    rho = abs(F6.PROJ.transform_point(F6.LON0, F6.LAT_MIN, F6.PC)[1])
    Pm = F6.PROJ.transform_points(F6.PC, main.lon.values, main.lat.values)[:, :2]
    P_mm = (Pm + rho) / (2 * rho / D)
    Q_mm, disp, _ = F6.resolve_overlaps(P_mm, np.array([D / 2, D / 2]), D / 2)
    GAK = F6.alaska_geometry(ak, F6.AK_INSET_W_MM, F6.AK_INSET_H_MM)
    rect = (Wm - 0.75 - D, F6_MAP_TOP, D, D)
    ax, _ = F6.polar_axes(fig, rect)
    F6.draw_pfr(ax, F6.PROJ, rect[2], rect[3], P)
    F6.graticule(ax)
    F6.draw_targets(ax, rho, main, col, P_mm, Q_mm, disp, norm, cmap, letter)
    y_bot = Hm - F6_BOTTOM
    trow = M[M.name == "Tibet_LGD"].iloc[0]
    irect = (0.6, y_bot - F6.INSET_H_MM, F6.INSET_W_MM, F6.INSET_H_MM)
    F6.tibet_inset(fig, irect, trow, col, norm, cmap, P, letter)
    arect = (Wm - 0.6 - F6.AK_INSET_W_MM, y_bot - F6.AK_INSET_H_MM, F6.AK_INSET_W_MM, F6.AK_INSET_H_MM)
    ax_ak, _ = F6.alaska_inset(fig, arect, ak, col, norm, cmap, P, GAK, letter)
    F6.zoom_frame(ax, GAK, ov, ax_ak, letter)
    for ln in fig.findobj(Line2D):                                           # 실제 위치 점(논문 1.2 mm)도 같은 배율
        if str(ln.get_gid()).startswith("truepos"):
            ln.set_markersize(1.2 * F6_K / MM * 72.0)
    vals = dict(min_pair_gap_mm=float(min(np.hypot(*(Q_mm[i] - Q_mm[j])) for i in range(len(Q_mm)) for j in range(i + 1, len(Q_mm)))),
                circle_d_mm=F6.CIRCLE_D_MM, n_leaders=int((disp > F6.LEADER_MIN_MM).sum()))
    if letter == "a":                                                        # 논문처럼 a 에만: 삽도 이름, 위도 라벨, 축척 막대, 영구동토 열쇠
        ov.text(irect[0] + 0.3, y_bot + 0.4, "Tibetan Plateau", ha="left", va="top", fontsize=F6_FONT)
        ov.text(arect[0], y_bot + 0.4, "Alaska", ha="left", va="top", fontsize=F6_FONT)
        sw_w, sw_h = 4.0, 2.6
        for k_, (lab, colr) in enumerate((("Continuous", S.BASEMAP["continuous"]), ("Discontinuous", S.BASEMAP["discontinuous"]))):
            yy = 3.0 + k_ * 5.6
            ov.add_patch(Rectangle((0.6, yy - sw_h / 2), sw_w, sw_h, facecolor=colr, edgecolor="none", zorder=6))
            ov.text(0.6 + sw_w + 1.2, yy, lab, ha="left", va="center", fontsize=F6_FONT, zorder=6)
        _inset_scale_beside(fig, "scale_bar_inset_tibet", lift_mm=3.0)      # 13 pt '200 km' 이 삽도 가운데 원에 닿지 않게 막대 오른쪽으로
        F6.lat_labels(ax)
        F6.scale_bar(ax, F6.PROJ, F6.SCALE["lon"], F6.SCALE["lat"], F6.SCALE["km"])
        _f6_free_labels(fig, ax, F6, vals)
    _thicken(fig)
    print(f"  [fig6{letter}] {vals}", flush=True)
    save_s(fig, f"Fig6_{letter}_slide.png", (W_in, H_in), "11", f"crops/f6{letter}.png",
           f"circle {F6.CIRCLE_D_MM:.2f} mm (paper 3.0), map circle {D:.0f} mm; column head left to the deck"
           + ("; key, inset names, latitude labels and scale bars only in a (as in the paper)" if letter == "a" else ""))


def fig6_cbar(F6, Dd, font=FONT):
    """공유 컬러바(가로)와 방향 표지 'Lower error' + 왼쪽 화살표. 막대 가운데 = 두 지도 그림 사이 가운데(덱 11쪽에서 이 띠를
    COL[0] = 0.667 in 에 두면 5.87 in), 논문(두 슬롯 가운데)과 같은 규칙."""
    W_in, H_in = F6_CBAR_BOX
    Wm, Hm = W_in * MM, H_in * MM
    fig = plt.figure(figsize=(W_in, H_in))
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_axis_off(); ov.set_xlim(0, Wm); ov.set_ylim(Hm, 0)
    mid = ((0.667 + 6.817 + F6_MAP_BOX[0]) / 2 - 0.667) * MM
    L_bar, bar_h, y0 = 180.0, 3.5, 1.2
    x0 = mid - L_bar / 2
    cax = S.axes_mm(fig, x0, y0, L_bar, bar_h)
    sm = matplotlib.cm.ScalarMappable(norm=Dd["norm"], cmap=Dd["cmap"])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=Dd["ext"], extendfrac=0.03)
    ticks = np.linspace(-Dd["vmax"], Dd["vmax"], 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([S.fmt_int(v) for v in ticks])
    cb.outline.set_linewidth(1.5); cb.outline.set_edgecolor(S.INK)
    cb.dividers.set_linewidth(1.5)
    cax.tick_params(width=1.5, length=5.0, pad=2.5, labelsize=font)
    from matplotlib.ticker import NullLocator
    cax.xaxis.set_minor_locator(NullLocator())
    cb.set_label("Error change vs source Stefan (cm)", labelpad=2.0, fontsize=font)
    t = ov.text(x0 - 2.0, y0 + bar_h / 2, "Lower error", ha="right", va="center", fontsize=font, color=S.INK_AUX)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ov.transData.inverted())
    xl = min(bb.x0, bb.x1)
    ov.annotate("", xy=(xl - 8.0, y0 + bar_h / 2), xytext=(xl - 1.2, y0 + bar_h / 2),
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=1.5, color=S.INK_AUX, shrinkA=0, shrinkB=0))
    save_s(fig, "Fig6_cbar_slide.png", (W_in, H_in), "11", "crops/f6_cbar.png",
           f"bar {L_bar:.0f} mm centred at {mid / MM:.2f} in from the left edge (midpoint of the two map images at COL[0] and COL[3]); "
           f"extend {Dd['ext']}, ±{Dd['vmax']:g} cm as in the paper")


def fig6():
    import fig6 as F6
    slide_rc(F6_FONT); slide_scales()
    saved = _f6_patch(F6)
    try:
        Dd = _f6_data(F6)
        fig6_map(F6, Dd, "a")
        fig6_map(F6, Dd, "b")
        slide_rc(FONT)
        fig6_cbar(F6, Dd)
    finally:
        _f6_unpatch(F6, saved)
        restore()


# ================================================================ Fig 5c (P)


def fig5c():
    import fig5 as F5
    m = (3.39 * MM) / ((1492 - 1018) / 2000 * F5.L["W"])
    font = max(S.FONT_PT, 12.5 / m)
    orig_use = S.use_v3
    S.use_v3 = lambda medium="paper": slide_rc(font, paper_geometry=True)
    slide_scales(lw_mult=1.0, lw_min=max(1.0, 1.5 / m), ms_mult=1.0)
    try:
        T = F5.prepare([])
        fig, _, _ = F5.draw(T, "paper")
    finally:
        S.use_v3 = orig_use
    tmp = OUT / "_tmp_fig5_full.png"
    check_text(fig, "Fig5(full, paper geometry)")
    fig.savefig(tmp, dpi=600)
    plt.close(fig)
    crop_resize(tmp, CROP["f5c"], (3.39, 3.40), "Fig5_c_slide.png", "36", "crops/f5c.png", None,
                f"map only, no text inside the deck crop box; same framing as the crop, lines at least {max(1.0, 1.5 / m) * m:.1f} pt "
                f"at slide size (deck magnification {m:.2f})")
    tmp.unlink()
    restore()


# ================================================================ 표지 지도(S)
def title_map():
    d = MV.prepare(MV.load_region("alaska"))
    MV.use_medium("slide")
    M = MV.Med("slide")
    W_in, H_in = 5.5, 5.0
    Wm, Hm = W_in * MM, H_in * MM
    ext = d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    cb_block = 21.0
    mh = Hm - 1.5 - cb_block
    mw = mh / asp
    if mw > Wm - 2.0:
        mw = Wm - 2.0; mh = mw * asp
    fig = plt.figure(figsize=(W_in, H_in))
    ax = S.axes_mm(fig, (Wm - mw) / 2, 1.5, mw, mh, projection=d["proj"])
    Mt = MV.Med("slide")
    Mt.lw = 1.0                                                              # 얇은 해안선(표지)
    MV.base_map(ax, d, ext, Mt)
    lo, hi = d["rng"]["alt"]
    MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["r1"], MV.CM_ALT, Normalize(lo, hi)))
    ax.spines["geo"].set_visible(False)
    M.fs_lab = 14.0; M.fs = 14.0
    cax = S.axes_mm(fig, Wm * 0.2, 1.5 + mh + 3.0, Wm * 0.6, 2.6)
    MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M,
                MV.extend_of(d["r1"][d["shown"]], lo, hi))
    save_s(fig, "Alaska_ALT_map_v3_c_slide.png", (W_in, H_in), "1 (title)", "crops/f1a.png on the title slide",
           "Alaska residual ML ALT (panel c of Alaska_ALT_map_v3), no graticule, thin coastline")


# ================================================================ README
def write_readme():
    old = {}
    p = OUT / "slide_panels_record.json"
    if p.exists():
        try:
            old = {r["file"]: r for r in json.loads(p.read_text())}
        except (ValueError, KeyError):
            old = {}
    for r in RECORD:
        old[r["file"]] = r
    recs = sorted(old.values(), key=lambda r: r["file"])
    p.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    lines = ["# 슬라이드용 패널(2026-10-05)", "",
             "논문 그림의 그리기 함수와 자료를 그대로 쓰고 글자(Pretendard, 슬라이드 크기에서 12 pt 이상), 선(1.5 pt 이상), 표지 크기만 슬라이드 배치에 "
             "맞춘 재렌더다. 새 자료와 새 주장은 없다. 만든 스크립트: `scripts/4_visualization/paper_v3/slide_panels.py`.",
             "방식 S = 슬라이드 배치 상자 크기로 패널을 단독으로 다시 그림, P = 논문 배치에 큰 글자로 다시 그린 뒤 덱과 같은 상자로 자름.", "",
             "| 덱 쪽 | 대체하는 자르기 | 새 파일 | 크기(in, 300 dpi) | 방식 | 비고 |", "|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['pages']} | `{r['replaces']}` | `{r['file']}` | {r['size_in'][0]:.2f} × {r['size_in'][1]:.2f} | {r['how']} | {r['note']} |")
    lines += ["", "배치 상자(deck/build_paper_report.py): 1쪽 표지 그림 3.80 × 3.80 안(새 표지 지도는 요청 크기 5.5 × 5.0); 5쪽 f1a 높이 5.20, "
              "f1b 높이 5.20; 8쪽 검증 사다리는 요청 크기 7.5 × 4.6(현재 S08 차트는 12.0 × 5.1 전폭); 11쪽 f6a·f6b 높이 약 3.91, f6_cbar 폭 9.95 "
              "(띠를 COL[0] 에 두면 막대 가운데가 두 지도 그림 사이 가운데); 21쪽 f7b 폭 5.40, f7c 높이 4.80; 24쪽 f1a 높이 5.20; "
              "36쪽 f7b·f5c 3.80 × 3.40 안. 파일 크기는 그 상자에 맞췄다.",
              "", "Fig 6 지도는 처음에 P 방식으로 만들었으나 큰 글자가 덱 자르기 상자 밖으로 나가 잘려(b 위 열 머리 조각, 삽도 이름, 컬러바 이름) "
              "S 방식으로 바꿨다. 지도 원 지름 86 mm(덱 자르기의 92 mm 보다 7 % 작음), 대상 원 지름 3.96 mm(논문 3.0 mm × 1.32)."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="title,fig1a,fig1b,fig1e,fig7b,fig7c,fig7e,fig5c,fig6")
    a = ap.parse_args(argv)
    todo = a.only.split(",")
    t0 = time.time()
    for k in todo:
        t1 = time.time()
        if k == "title":
            title_map()
        elif k == "fig1a":
            fig1a()
        elif k == "fig1b":
            fig1b()
        elif k == "fig1e":
            fig1e()
        elif k == "fig7b":
            fig7b(5.40, 4.29, "Fig7_b_slide.png", "21", FONT)
            fig7b(3.80, 3.02, "Fig7_b_small_slide.png", "36", 12.0)
        elif k == "fig7c":
            fig7c()
        elif k == "fig7e":
            fig7e()
        elif k == "fig5c":
            fig5c()
        elif k == "fig6":
            fig6()
        print(f"[done] {k} {time.time() - t1:.0f}s", flush=True)
    write_readme()
    print(f"[all] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
