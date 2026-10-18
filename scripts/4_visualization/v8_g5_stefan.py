"""G5 Stefan 물리식 (덱 전용 쪽, 12 × 4.65 in = 3600 × 1395 px, 300 dpi, 영문 FreeSans). docs/DESIGN_BRIEF_v9_2026-10-15.md 2.1(항 8·13).

v10(2026-10-16, 사용자 지시: 그림 안 글자는 모두 영어, 논문 어휘): 배치는 그대로, 글자만 영어로 바꿨다. 열 1 열쇠는 한 줄(열 왼쪽 끝에서 시작),
지역 이름 눈금 폭은 글자 치수로 잰다. 저장 전 모든 글자 객체에서 한글을 찾으면 멈춘다(no_hangul).

명세(v9: 라벨만 두고 설명 글줄은 모두 덱으로 보낸다. v7 F1 처럼 깨끗하게)
- 메시지: ALT = E√TDD. 한 지점의 한 해에서 누적 융해 도일이 쌓이는 만큼 융해 깊이가 제곱근 비례로 깊어지고, 계수 E 는 지역에 따라 다르다.
- 제곱근은 모두 mathtext(\sqrt{\mathrm{TDD}}, 윗줄 포함). 글리프 '√' 단독 표기 없음(식·축 라벨·단위·범례·열 머리 모두).
- 왼쪽 열 '식과 계수': 큰 식 + 지역별 계수 E 두 가지(원천 계수 = 다른 두 지역 라벨의 최소제곱, 재보정 = 학습 겹 라벨의 최소제곱, 5겹 평균)의 점·선 차트.
  열쇠는 차트 위 한 줄(원천 계수 · 재보정). 값 글자 없음(눈금으로 읽는다). 수치는 runs/core_v6/meta/<Region>__block__year_obs__F8TS__P1__-__lam-1.json.
- 가운데 열 '사모일로프 2015': 같은 시간축 위에 누적 TDD(ERA5-Land 월평균의 일 보간, v7_f2_problem_time._daily_tdd)와 탐침 관측 융해 깊이
  (ALLena, PANGAEA 973813, 일 평균 ± SD; 깊이 축은 아래 방향)와 Stefan 곡선 E√TDD(t)(E 는 그 계열의 원점 통과 최소제곱). 범례 둘(탐침 관측 · E√TDD).
- 오른쪽 열 '융해 깊이 대 √TDD': 사모일로프 2015 · 쿠룽나흐 2020 의 관측일까지 √TDD 대 융해 깊이, 원점 통과 직선. 범례 둘(지점 · 연도).
- 글자: 한글 ≥ 13 pt, 눈금 12 pt, 축 제목 14 pt, 열 머리 15 pt 굵게(주황). 색: 지점 = 레나 지역 색의 두 명도(토큰 region_tints), 계수 = 방법 색(P0 원천, P1 재보정).
- 산출: deck/assets/v8/G5_stefan.png(RGB, 흰 바탕, 3600 × 1395 px; +pdf), G5_stefan_spec.json(배치·수치 출처·QA 기록).
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g5_stefan.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU 사용 금지(CPU 전용)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (M.canvas, M.T, M.head, M.hline, M.vline, M.sub_axes, M.qa_and_save, M.setup_fonts)
import polar.style_v5 as S  # noqa: E402
import v7_f2_problem_time as F2  # noqa: E402  (_allena_series, _daily_tdd, SITE_HEX)

M.OUT = ROOT / "deck" / "assets" / "v8"
PROC = ROOT / "data" / "processed" / "v5"
META = PROC / "runs" / "core_v6" / "meta"
STEM = "G5_stefan"

# ================================================================ 배치(인치)
W, H = 12.0, 4.65
COLS = [(0.10, 3.55), (4.15, 7.55), (8.15, 11.90)]      # 열 안쪽 x 범위(열 사이 0.60 in)
SEPX = [3.85, 7.85]
Y_HEAD, Y_HEADRULE = H - 0.17, H - 0.31                 # 열 머리 · 머리 밑줄
AX_B, AX_T = 0.62, H - 0.55                             # 차트 축 아래·위(가운데·오른쪽 열)
X_OFF = 0.74                                            # 축 라벨·눈금 글자 폭
FS_T = 13.0                                             # 범례·한글 눈금·직접 라벨 13 pt
FS_AX = 14.0                                            # 축 제목 14 pt(토큰 slide.axis_label_pt)
FS_K = 12.0                                             # 숫자 눈금 12 pt
FS_HEAD = M.FS["head"]                                  # 열 머리 15 pt
FS_EQ = 34.0                                            # 식
E_UNIT = r"cm/$\sqrt{\mathrm{°C·d}}$"                    # E 단위(mathtext 제곱근)
SQRT_TDD = r"$\sqrt{\mathrm{TDD}}$"
REG = ["Alaska", "Lena", "Canada"]
REG_KO = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}      # 그림 지역 이름(영문, S.REGION_EN 과 같다)
SITE_KO = {"Samoylov": "Samoylov", "Kurungnakh": "Kurungnakh"}
SITE_HEX = F2.SITE_HEX                                  # 레나 지역 색의 두 명도(토큰 region_tints)
P0, P1 = M.COL["P0"], M.COL["P1"]                       # 원천 계수 Stefan · 재보정 Stefan(토큰 methods)
MONTH_DOY = {152: "Jun", 182: "Jul", 213: "Aug", 244: "Sep", 274: "Oct"}   # 2015(평년) 월초 연중 일수
X_DOY = (140.0, 278.0)
SPEC: dict = {"stem": STEM, "size_in": [W, H], "sources": {}, "choices": {}}


# ================================================================ 자료
def e_table():
    """지역별 Stefan 계수: 원천 E0(다른 두 지역 라벨의 최소제곱, v5_compare.py) · 재보정 E(학습 겹 라벨의 최소제곱, 블록 5겹 평균)."""
    rows = []
    for r in REG:
        f = META / f"{r}__block__year_obs__F8TS__P1__-__lam-1.json"
        m = json.loads(f.read_text())
        folds = {k: float(v["E"]) for k, v in m["coefs"].items()}
        rows.append(dict(region=r, E_source=float(m["E0"][r]), E_recal=float(np.mean(list(folds.values()))), E_recal_folds=folds,
                         n_folds=int(m["n_folds"]), n=int(m["n"])))
        SPEC["sources"][f"meta_{r}"] = str(f.relative_to(ROOT))
    SPEC["choices"]["coefficients"] = rows
    return rows


def site_series():
    """ALLena 두 계열(사모일로프 2015 · 쿠룽나흐 2020): 일별 누적 TDD, 탐침 일 평균 ± SD, 원점 통과 최소제곱 E(모든 탐침값; F2c 와 같다)."""
    out = []
    for name, lat, lon, yr, doy_s, z in F2._allena_series():
        dd, cum = F2._daily_tdd(lat, lon, yr)
        tdd = np.interp(doy_s, dd, cum)
        g = pd.DataFrame({"doy": doy_s, "tdd": tdd, "z": z}).groupby("doy").agg(tdd=("tdd", "first"), z=("z", "mean"), sd=("z", "std"), n=("z", "size"))
        E = float((np.sqrt(tdd) @ z) / (np.sqrt(tdd) @ np.sqrt(tdd)))
        out.append(dict(name=name, lat=lat, lon=lon, year=yr, doy=dd, cum=cum, g=g, E=E, n_probe=int(len(z))))
        SPEC["choices"][f"series_{name}_{yr}"] = dict(lat=lat, lon=lon, E=round(E, 4), E_unit=E_UNIT, n_probe=int(len(z)), n_days=int(len(g)),
                                                       doy_first=int(g.index.min()), doy_last=int(g.index.max()),
                                                       tdd_at_last_day=round(float(g.tdd.iloc[-1]), 1), color=SITE_HEX[name])
    SPEC["sources"]["allena"] = "data/raw/allena/PANGAEA_973813_ALLena_main.txt"
    SPEC["sources"]["era5land"] = "data/raw/era5land/nh_monthly_2010-2024.nc"
    return out


# ================================================================ 축 양식
def chart_axes(fig, x, y, w, h):
    ax = M.sub_axes(fig, x, y, w, h)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.8); ax.spines[k].set_edgecolor(M.ZERO)
    ax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2, colors=M.INK)
    return ax


def head_mixed(ax, x, y, segs, size=None):
    """열 머리(주황, 굵게)에 한글과 수식(mathtext)을 나란히 놓는다. segs 는 문자열 목록, '$..$' 는 수식. y 는 글줄 가운데."""
    size = size or FS_HEAD; fig = ax.figure; r = fig.canvas.get_renderer()
    yb = y - 0.33 * size / 72.0; cx = x; arts = []
    for s_ in segs:
        t = ax.text(cx, yb, s_, fontsize=size, color=M.HEAD, fontweight="bold", ha="left", va="baseline", zorder=6)
        cx += t.get_window_extent(r).width / fig.dpi; arts.append(t)
    return arts


def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    import re
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")


def tick_label_artists(fig):
    ids = set()
    for a in fig.axes:
        if not a.axison:
            continue
        for t in a.get_xticklabels(minor=False) + a.get_yticklabels(minor=False):
            ids.add(id(t))
    return ids


def site_note(fig, a3, lg, E_steep, lines=("지점 E: 한 지점·한 해의", "원점 통과 적합")):
    """범례 바로 밑에 지점 E 의 뜻을 두 글줄로 놓는다(지역 E 와 구분). 각 글줄 상자가 가장 가파른 적합 직선 y = E·x 위에 있어야 한다."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = lg.get_window_extent(r); tx = [t.get_window_extent(r) for t in lg.get_texts()]
    x_px = min(t.x0 for t in tx); y_px = bb.y0 - 0.10 * fig.dpi
    inv = a3.transData.inverted(); arts = []
    for s in lines:
        t = a3.text(*inv.transform((x_px, y_px)), s, ha="left", va="top", fontsize=FS_T, color=M.AUX, zorder=6)
        tb = t.get_window_extent(r); (x1, y0) = inv.transform((tb.x1, tb.y0))
        ok = y0 > E_steep * x1; arts.append(dict(text=s, right_x=round(float(x1), 2), bottom_y=round(float(y0), 2), above_line=bool(ok)))
        assert ok, f"지점 E 글줄이 적합 직선과 겹침: {s} ({x1:.1f}, {y0:.1f})"
        y_px = tb.y0 - 0.05 * fig.dpi
    SPEC["choices"]["site_note"] = arts


# ================================================================ 그림
def g5():
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    rows = e_table(); ser = site_series()
    samo = [s for s in ser if s["name"] == "Samoylov"][0]

    # ---- 열 머리 · 헤어라인 · 열 구분
    M.head(ax, COLS[0][0], Y_HEAD, "Stefan equation", size=FS_HEAD)
    M.head(ax, COLS[1][0], Y_HEAD, "Samoylov 2015", size=FS_HEAD)
    head_mixed(ax, COLS[2][0], Y_HEAD, ["Thaw depth vs ", r"$\mathbf{\sqrt{TDD}}$"])
    for (x0, x1) in COLS:
        M.hline(ax, x0, x1, Y_HEADRULE, color=M.HAIR, lw=0.8)
    for x in SEPX:
        M.vline(ax, x, 0.12, Y_HEADRULE - 0.06, color=M.SEP, lw=0.9, ls=(0, (3, 3)))

    # ---- 열 1 · 식(mathtext, 제곱근 윗줄)
    xl0, xl1 = COLS[0]
    t_eq = M.T(ax, (xl0 + xl1) / 2, 3.66, r"$\mathrm{ALT} = E\,\sqrt{\mathrm{TDD}}$", size=FS_EQ, ha="center", va="center")

    # ---- 열 1 · 아래: 지역별 계수(원천 · 재보정) 점·선 차트, 열쇠는 차트 위 한 줄
    _f = plt.figure(figsize=(3, 1), dpi=M.DPI); _t = _f.text(0, 0, "Canada (NW)", fontsize=FS_T); _f.canvas.draw()
    x_off1 = _t.get_window_extent(_f.canvas.get_renderer()).width / M.DPI + (6 + 2) / 72.0 + 0.04; plt.close(_f)   # 가장 긴 지역 이름 + 눈금 간격
    a1 = chart_axes(fig, xl0 + x_off1, AX_B, xl1 - 0.12 - (xl0 + x_off1), 2.36 - AX_B)
    ys = [2, 1, 0]; vals = []
    for yy, r in zip(ys, rows):
        a1.plot([r["E_source"], r["E_recal"]], [yy, yy], color=M.HAIR, lw=1.6, solid_capstyle="butt", zorder=2)
        a1.plot([r["E_source"]], [yy], "o", ms=9.0, color=P0, mec="white", mew=0.8, zorder=4)
        a1.plot([r["E_recal"]], [yy], "o", ms=9.0, color=P1, mec="white", mew=0.8, zorder=4)
        vals += [r["E_source"], r["E_recal"]]
    a1.set_yticks(ys); a1.set_yticklabels([REG_KO[r["region"]] for r in rows], fontsize=FS_T)
    a1.tick_params(axis="y", length=0, pad=6)
    a1.set_ylim(-0.55, 2.55); a1.set_xlim(min(vals) - 0.10, max(vals) + 0.10)
    a1.set_xticks([1.6, 1.8, 2.0]); a1.spines["left"].set_visible(False)
    a1.set_xlabel(r"$E$ (" + E_UNIT + ")", fontsize=FS_AX, labelpad=2)
    hs = [Line2D([], [], marker="o", ls="none", ms=9.0, color=P0, mec="white", mew=0.8), Line2D([], [], marker="o", ls="none", ms=9.0, color=P1, mec="white", mew=0.8)]
    lg1 = ax.legend(hs, ["Source coeff.", "Recalibrated"], loc="center left", bbox_to_anchor=(xl0 + 0.02, 2.76), bbox_transform=ax.transData, ncol=2,
                    fontsize=FS_T, frameon=False, handlelength=0.8, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0, borderpad=0.0)

    # ---- 열 2 · 같은 시간축: 위 누적 TDD, 아래 융해 깊이(아래 방향) + Stefan 곡선
    xm0, xm1 = COLS[1]
    c = SITE_HEX["Samoylov"]; g = samo["g"]; doy, cum = samo["doy"], samo["cum"]; E = samo["E"]
    y_mid = 2.38; gap_mid = 0.20
    au = chart_axes(fig, xm0 + X_OFF, y_mid + gap_mid, xm1 - 0.15 - (xm0 + X_OFF), AX_T - (y_mid + gap_mid))
    al = chart_axes(fig, xm0 + X_OFF, AX_B, xm1 - 0.15 - (xm0 + X_OFF), y_mid - AX_B)
    sel = (doy >= X_DOY[0]) & (doy <= X_DOY[1])
    au.plot(doy[sel], cum[sel], color=M.INK, lw=1.8, zorder=3)
    cmax = float(cum[sel].max()); step = 250.0 if cmax < 1100 else 500.0
    au.set_ylim(0, cmax * 1.10); au.set_yticks(np.arange(0, cmax * 1.10 + 1e-9, step))
    au.set_ylabel("Cumulative TDD\n(°C·d)", fontsize=FS_AX, labelpad=4, linespacing=1.1)
    sd = g.sd.fillna(0)
    al.fill_between(g.index.values, g.z - sd, g.z + sd, color=c, alpha=0.28, lw=0, zorder=2)
    al.plot(doy[sel], E * np.sqrt(cum[sel]), color=c, lw=1.8, zorder=3)
    al.plot(g.index.values, g.z, "o", ms=5.0, color=c, mec="white", mew=0.5, zorder=5)
    zmax = max(float((g.z + sd).max()), float(E * np.sqrt(cmax)))
    al.set_ylim(zmax * 1.12, 0); al.set_yticks(np.arange(0, zmax * 1.12 + 1e-9, 10))
    al.set_ylabel("Thaw depth (cm)", fontsize=FS_AX, labelpad=4)
    for a_ in (au, al):
        a_.set_xlim(*X_DOY); a_.set_xticks(list(MONTH_DOY))
    au.set_xticklabels([]); au.tick_params(axis="x", length=3)
    al.set_xticklabels(list(MONTH_DOY.values()), fontsize=FS_T)
    lg2 = al.legend([Line2D([], [], marker="o", ms=5.0, color=c, mec="white", mew=0.5, ls="none"), Line2D([], [], color=c, lw=1.8)],
                    ["Observed", r"$E\sqrt{\mathrm{TDD}}$"], loc="upper right", fontsize=FS_T, frameon=False, handlelength=1.6,
                    handletextpad=0.6, borderaxespad=0.1, borderpad=0.1, labelspacing=0.4)
    SPEC["choices"]["middle"] = dict(site="Samoylov", year=2015, x_doy=list(X_DOY), month_ticks=MONTH_DOY, cum_tdd_max_in_window=round(cmax, 1),
                                     E=round(E, 4), depth_axis="downward", tdd_source="ERA5-Land monthly means, daily linear interpolation (v7_f2_problem_time._daily_tdd)")

    # ---- 열 3 · 융해 깊이 대 √TDD(관측일까지), 두 계열 원점 통과 직선
    xr0, xr1 = COLS[2]
    a3 = chart_axes(fig, xr0 + X_OFF, AX_B, xr1 - 0.05 - (xr0 + X_OFF), AX_T - AX_B)
    xmax = 0.0; hs, labs = [], []
    for s_ in ser:
        g_ = s_["g"]; c_ = SITE_HEX[s_["name"]]; zo = 3 + (s_["name"] == "Kurungnakh"); sd_ = g_.sd.fillna(0)
        a3.fill_between(np.sqrt(g_.tdd), g_.z - sd_, g_.z + sd_, color=c_, alpha=0.28, lw=0, zorder=zo - 1)
        a3.plot(np.sqrt(g_.tdd), g_.z, "o", ms=5.0, color=c_, mec="white", mew=0.5, zorder=zo + 2)
        xmax = max(xmax, float(np.sqrt(g_.tdd.max())))
    xe_ = xmax * 1.12; xx = np.linspace(0, xe_, 20)
    for s_ in ser:
        c_ = SITE_HEX[s_["name"]]
        a3.plot(xx, s_["E"] * xx, color=c_, lw=1.8, zorder=2)
        hs.append(Line2D([], [], color=c_, lw=1.8, marker="o", ms=5.0, mec="white", mew=0.5))
        labs.append(f"{SITE_KO[s_['name']]} {s_['year']}")
    xt_end = np.ceil(xe_ / 10) * 10
    a3.set_xticks(np.arange(0, xt_end + 1e-9, 10)); a3.spines["bottom"].set_bounds(0, xt_end)
    y3_top = max(s_["E"] for s_ in ser) * xe_ * 1.10
    a3.set_xlim(0, xt_end * 1.03); a3.set_ylim(0, y3_top); a3.set_yticks(np.arange(0, y3_top + 1e-9, 10))   # 눈금은 축 범위 안만(밖 눈금 글자 없음)
    a3.set_xlabel(SQRT_TDD + r" ($\sqrt{\mathrm{°C·d}}$)", fontsize=FS_AX, labelpad=2); a3.set_ylabel("Thaw depth (cm)", fontsize=FS_AX, labelpad=4)
    lg3 = a3.legend(hs, labs, loc="upper left", fontsize=FS_T, frameon=False, handlelength=1.6, handletextpad=0.6, borderaxespad=0.1,
                    borderpad=0.1, labelspacing=0.5)
    SPEC["choices"]["right"] = dict(x="√TDD to the observation day", lines="origin-through least squares per series", E={f"{s_['name']}_{s_['year']}": round(s_["E"], 4) for s_ in ser},
                                    note="E 값은 그림에 적지 않는다(v9 글자 최소화)")

    # ---- 점검: 글자 크기(눈금 12 pt · 그 밖 13 pt), 범례 상자 안 자료점, 캔버스 사용률, 저장
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    ticks = tick_label_artists(fig); below13 = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FS_T - 1e-6:
            if id(t) in ticks and t.get_fontsize() >= FS_K - 1e-6:
                continue
            below13.append([t.get_text()[:30], float(t.get_fontsize())])

    def pts_in_legend(a_, lg_, xs, ys_):
        bb = lg_.get_window_extent(rr); inv = a_.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return int(sum(1 for xv, yv in zip(xs, ys_) if x0 <= xv <= x1 and lo_y <= yv <= hi_y))
    xs3 = np.concatenate([np.sqrt(s_["g"].tdd.values) for s_ in ser]); ys3 = np.concatenate([s_["g"].z.values for s_ in ser])
    lines3 = [(xv, s_["E"] * xv) for s_ in ser for xv in np.linspace(0, xe_, 200)]
    gz = g.z.values; gx = g.index.values.astype(float)
    curve2 = [(xv, E * np.sqrt(np.interp(xv, doy, cum))) for xv in np.linspace(*X_DOY, 200)]
    SPEC["qa_extra"] = dict(legend3_points_inside=pts_in_legend(a3, lg3, xs3, ys3), legend3_line_inside=pts_in_legend(a3, lg3, *zip(*lines3)),
                            legend2_points_inside=pts_in_legend(al, lg2, gx, gz), legend2_curve_inside=pts_in_legend(al, lg2, *zip(*curve2)),
                            legend2_band_inside=pts_in_legend(al, lg2, np.r_[gx, gx], np.r_[gz - sd.values, gz + sd.values]),
                            equation_box_in=[round(v / fig.dpi, 3) for v in t_eq.get_window_extent(rr).extents])
    tb = fig.get_tightbbox(rr)
    SPEC["choices"]["canvas_fill"] = dict(x0_in=round(tb.x0, 2), x1_in=round(tb.x1, 2), y0_in=round(tb.y0, 2), y1_in=round(tb.y1, 2))
    no_hangul(fig)
    rec = M.qa_and_save(fig, STEM, allow_overlap=())
    from PIL import Image
    src_png, src_pdf = M.OUT / f"{STEM}_slide.png", M.OUT / f"{STEM}_slide.pdf"
    png, pdf = M.OUT / f"{STEM}.png", M.OUT / f"{STEM}.pdf"
    with Image.open(src_png) as im:                  # RGBA → RGB(흰 바탕), 화소 크기 확인
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        px = list(im.size)
    bg.save(png, dpi=(M.DPI, M.DPI)); src_png.unlink()
    if src_pdf.exists():
        src_pdf.replace(pdf)
    SPEC["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    SPEC["qa"]["png_px"] = px; SPEC["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    SPEC["qa"]["non_tick_text_below_13pt"] = below13
    SPEC["qa"]["bare_sqrt_glyph_texts"] = [t.get_text()[:40] for t in fig.findobj(Text) if "√" in t.get_text()]
    print("글자 크기 점검(눈금 외 13 pt 미만):", "OK" if not below13 else f"CHECK {below13}", "· 화소", px, flush=True)
    print("범례 안 자료:", SPEC["qa_extra"], "· 글리프 √:", SPEC["qa"]["bare_sqrt_glyph_texts"] or "없음", flush=True)
    (M.OUT / f"{STEM}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1, default=float))
    return png, pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", action="store_true", help="호환용. 이 그림은 슬라이드판 하나만 만든다")
    ap.add_argument("--fig", default="G5", help="호환용")
    ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    for p in g5():
        print("출력:", p)
