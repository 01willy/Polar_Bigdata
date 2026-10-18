"""G0 활동층 두께 개념(덱 전용 쪽, 12 × 4.65 in = 3600 × 1395 px, 300 dpi, 영문 FreeSans).

v10(2026-10-16, 사용자 지시: 그림 안 글자는 모두 영어, 논문 어휘): 배치는 그대로, 글자만 영어로 바꿨다. 영어 글줄이 길어 (a) 의 층 이름 '활동층' 자리를
치수선 라벨('Active layer' / 'thickness (ALT)')로 합치고 기둥 폭을 0.54 → 0.50 in 로 줄여 경계 라벨('Ground surface', 'Permafrost table')이 열 안에 든다.
저장 전 모든 글자 객체에서 한글을 찾으면 멈춘다(no_hangul).

v9(2026-10-15, 설계 지침 v9 1.4·2.1): 덱 규격 12 × 4.65 in. 어휘 통일('녹는 깊이' → '융해 깊이', 누적 융해 도일과 같은 계열),
열 머리 괄호 앞 띄어쓰기, ALT 라벨 두 줄('ALT' / '최대 융해 깊이'; 관계 기호로 시작하는 줄 없음), 각주 문구 정리. 구조는 그대로.

명세
- 메시지: 활동층 두께(ALT)는 여름 최대 융해 깊이이며, 한 해 안에서 융해 깊이는 누적 융해 도일과 함께 커진다.
- 세 열(왼쪽부터):
  (a) 토양 기둥 개략. 겨울·여름 두 기둥(같은 폭, 0–2 m, 공통 깊이 축), 활동층 = 연갈색(토큰 status.terrain 의 30 % 틴트),
      영구동토 = 청회색(토큰 map_base.slide.pf_continuous), 빗금 = 동결 상태. 융해 전선(영구동토 상단면)의 깊이는 (b)의 최대값(m)으로 둔다.
      층 이름('활동층', '영구동토')은 기둥 오른쪽에 붙인 글자, 경계('지표', '영구동토 상단면')는 이음선 라벨, ALT 는 화살촉 없는 치수선
      (세로선 + 양 끝 가로 눈금; style_tokens_v4 symbols.forbidden '화살촉 주석' 준수)과 두 줄 라벨 'ALT / 최대 융해 깊이'.
      열쇠 세 항목(활동층·동결, 활동층·융해, 영구동토·동결): 색 = 층, 빗금 = 동결 상태.
  (b) 사모일로프 2015 계절 안 융해 깊이. ALLena 원자료(PANGAEA 973813, v7_f2_problem_time._allena_series)의 날짜별 탐침 평균 ± SD,
      깊이 축 아래 방향(0 이 위), 계절 최대(일 평균)에 파선 'ALT <값> cm', 최대 날에 점선 세로선, 첫 관측일 표기('관측 시작 <날짜>').
  (c) 같은 셀·같은 해의 일 누적 융해 도일(ERA5-Land 월평균 보간, v7_f2_problem_time._daily_tdd). (b)와 같은 x 범위·눈금, 최대 날의 값을 표식.
      달 눈금은 매달 1일(F2 와 같은 규약)이라 점선(9/16)이 달 이름과 떨어진다.
- 수치는 모두 자료에서 계산한다(ALT 최대값·날짜·그날의 누적 TDD·첫 관측일). 손으로 적은 수치 없음.
- 글자: 한글 라벨 ≥ 13 pt, 눈금 ≥ 12 pt, 열 머리 15 pt 굵게 주황(M.head). 자료 출처 각주는 12 pt 회색(눈금 크기, 라벨 13 pt 보다 작게 둔 의도).
  상자는 토양 기둥·열쇠 견본만, 화살표·화살촉 없음, 그라데이션·아이콘 없음.
- 산출: deck/assets/v8/G0_alt_concept.png(RGB, 흰 바탕; +pdf), G0_alt_concept_spec.json(배치·자료 출처·수치·QA 기록).
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g0_alt_concept.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (M.canvas, M.T, M.head, M.hline, M.vline, M.sub_axes, M.qa_and_save, M.setup_fonts; 화살표 없음)
import polar.style_v5 as S  # noqa: E402  (토큰 색: status.terrain, 지역 틴트)
import v7_f2_problem_time as F2  # noqa: E402  (_allena_series, _daily_tdd, SITE_HEX)

M.OUT = ROOT / "deck" / "assets" / "v8"
STEM = "G0_alt_concept"

# ================================================================ 배치(인치)
W, H = 12.0, 4.65
COLS = [(0.10, 3.80), (4.10, 7.80), (8.10, 11.90)]      # 열 안쪽 x 범위
SEPX = [3.95, 7.95]
Y_HEAD, Y_HEADRULE = H - 0.17, H - 0.31                 # 열 머리 · 머리 밑줄
AX_B, AX_T = 0.90, H - 0.62                             # (b)(c) 차트 축 아래/위
X_OFF = 0.64                                            # y축 라벨·눈금 글자 폭
FS_T, FS_K, FS_HEAD = 13.0, 12.0, M.FS["head"]          # 라벨 13 pt · 눈금 12 pt · 열 머리 15 pt
DASH = (0, (3.5, 1.8))                                  # style_v5 METHOD_LS 'dashed'
DOT = (0, (1.2, 1.8))
SITE, YEAR = "Samoylov", 2015
SAM = F2.SITE_HEX["Samoylov"]                           # 레나 지역 색의 진한 명도(F2c 와 같다)
HEADS = ("Active-layer thickness", "Thaw depth, Samoylov 2015", "Cumulative TDD, same year")
NOTES = ("Probe mean ± SD (ALLena)", "ERA5-Land monthly, interpolated")   # (b)(c) 자료 출처 각주(12 pt 회색)
MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
Y_NOTE = 0.21                                           # 각주 높이(캔버스 인치)
SPEC: dict = {"stem": STEM, "size_in": [W, H], "sources": {}, "choices": {}, "numbers": {}}
SMALL_OK: set = set()                                   # 12 pt 허용 글자 id: (a) 깊이 축 눈금, 자료 출처 각주


def _mix(hex_a, hex_b, t):
    """hex_a 와 hex_b 를 t : 1−t 로 섞는다."""
    a, b = np.array(to_rgb(hex_a)), np.array(to_rgb(hex_b))
    return to_hex(t * a + (1 - t) * b)


AL_FILL = _mix(S.STATUS["terrain"], "#FFFFFF", 0.30)    # 활동층: 지형 갈색의 30 % 틴트
AL_HATCH = S.STATUS["terrain"]                          # 겨울 활동층 빗금(동결)
PF_FILL = M.PF_CONT                                     # 영구동토: 지도 연속 영구동토 색
PF_HATCH = _mix(M.PF_CONT, M.INK, 0.55)                 # 영구동토 빗금
SPEC["choices"]["colors"] = dict(active_layer=AL_FILL, active_layer_hatch=AL_HATCH, permafrost=PF_FILL, permafrost_hatch=PF_HATCH, site=SAM)


# ================================================================ 자료
def samoylov_daily():
    """ALLena 사모일로프 2015: 날짜(연중 일수)별 탐침 평균·SD·개수. 반환 (lat, lon, 표)."""
    for name, lat, lon, yr, doy, z in F2._allena_series():
        if name == SITE and yr == YEAR:
            g = (pd.DataFrame({"doy": doy.astype(int), "z": z}).groupby("doy")
                 .agg(z=("z", "mean"), sd=("z", "std"), n=("z", "size")).reset_index())
            SPEC["sources"]["thaw_depth"] = "data/raw/allena/PANGAEA_973813_ALLena_main.txt (v7_f2_problem_time._allena_series)"
            SPEC["choices"]["site"] = dict(name=name, lat=lat, lon=lon, year=yr, n_probe=int(len(z)), n_days=int(len(g)))
            return lat, lon, g
    raise KeyError(f"{SITE} {YEAR} 계열 없음")


def month_axis(year):
    """6–10월 1일(연중 일수)과 달 이름. 눈금·글자 모두 매달 1일에 둔다(F2 의 Jan/Apr/Jul/Oct 규약과 같다)."""
    ms = (6, 7, 8, 9, 10)
    b = [dt.date(year, m, 1).timetuple().tm_yday for m in ms]
    return b, [MONTHS_EN[m - 1] for m in ms]


# ================================================================ 축 양식
def chart_axes(fig, col):
    x0, x1 = COLS[col]
    ax = M.sub_axes(fig, x0 + X_OFF, AX_B, x1 - 0.15 - (x0 + X_OFF), AX_T - AX_B)   # 오른쪽 0.15 in: '10월' 글자 반폭
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.8); ax.spines[k].set_edgecolor(M.ZERO)
    ax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2, colors=M.INK)
    return ax


def months(ax, bounds, names, xlo, xhi):
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(bounds); ax.set_xticklabels(names)
    ax.tick_params(axis="x", labelsize=FS_T, pad=3, colors=M.INK)   # 달 이름 13 pt
    ax.set_xlabel(f"{YEAR}", fontsize=FS_T, labelpad=2)


def note(ax, x_right, text):
    """차트 아래 자료 출처 각주: 12 pt 회색, 축 오른쪽 끝에 오른맞춤."""
    t = M.T(ax, x_right, Y_NOTE, text, size=FS_K, color=M.AUX, ha="right")
    SMALL_OK.add(id(t))
    return t


# ================================================================ 점검(G1 과 같은 방식)
def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    import re
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")



def _glyph_bitmap(s, rot):
    fig = plt.figure(figsize=(2.5, 2.5), dpi=M.DPI); ax_ = fig.add_axes([0, 0, 1, 1]); ax_.axis("off")
    ax_.text(0.5, 0.5, s, rotation=rot, ha="center", va="center", fontsize=FS_T)
    fig.canvas.draw(); a = np.asarray(fig.canvas.buffer_rgba())[:, :, 0].copy(); plt.close(fig)
    ys, xs = np.where(a < 128)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def ylabel_rotation_check(labels):
    """90° 회전 글줄을 되돌린 비트맵과 정립 비트맵을 비교한다(한글·라틴 혼합 글줄이 한 덩어리로 회전되는지)."""
    out = {}
    for s in labels:
        a0 = _glyph_bitmap(s, 0); a1 = np.rot90(_glyph_bitmap(s, 90), -1)
        h = min(a0.shape[0], a1.shape[0]); w = min(a0.shape[1], a1.shape[1])
        d = np.abs(a0[:h, :w].astype(int) - a1[:h, :w].astype(int))
        out[s] = dict(shape_upright=list(a0.shape), shape_rotated_back=list(a1.shape), mean_abs_diff=round(float(d.mean()), 2),
                      frac_pixels_differ=round(float((d > 128).mean()), 4), ok=bool(a0.shape == a1.shape and (d > 128).mean() < 0.01))
    return out


def tick_label_artists(fig):
    ids = set()
    for a in fig.axes:
        if not a.axison:
            continue
        for t in a.get_xticklabels(minor=False) + a.get_yticklabels(minor=False) + a.get_xticklabels(minor=True) + a.get_yticklabels(minor=True):
            ids.add(id(t))
    return ids


# ================================================================ (a) 토양 기둥 개략
def dim_bracket(ax, x, y0, y1, cap=0.06, color=M.INK, lw=1.1):
    """화살촉 없는 치수선: 세로선과 양 끝의 짧은 가로 눈금(style_tokens_v4 symbols.forbidden '화살촉 주석')."""
    M.vline(ax, x, y0, y1, color=color, lw=lw, z=5)
    for yy in (y0, y1):
        M.hline(ax, x - cap, x + cap, yy, color=color, lw=lw, z=5)


def legend_column(ax, x, y_top, pitch=0.26, kw=0.26, kh=0.15):
    """열쇠(세로 배열): 색 = 층, 빗금 = 동결 상태. 항목 라벨 = '층 · 상태'."""
    entries = (("Active layer, frozen", AL_FILL, AL_HATCH), ("Active layer, thawed", AL_FILL, None), ("Permafrost, frozen", PF_FILL, PF_HATCH))
    for i, (lab, fill, hatch) in enumerate(entries):
        yy = y_top - i * pitch
        ax.add_patch(Rectangle((x, yy - kh / 2), kw, kh, facecolor=fill, edgecolor="none", zorder=2))
        if hatch:
            ax.add_patch(Rectangle((x, yy - kh / 2), kw, kh, facecolor="none", edgecolor=hatch, hatch="//", lw=0, zorder=3))
        ax.add_patch(Rectangle((x, yy - kh / 2), kw, kh, facecolor="none", edgecolor=M.ZERO, lw=0.7, zorder=4))
        M.T(ax, x + kw + 0.08, yy, lab, size=FS_T, ha="left")
    return [e[0] for e in entries]


def draw_column_schematic(fig, ax, alt_m):
    x0, x1 = COLS[0]
    XAX = x0 + 0.36                                      # 깊이 축
    WW, GAP, LAB, BR, PITCH = 0.50, 0.22, 0.12, 0.14, 0.21   # 기둥 폭(겨울 = 여름) · 기둥 간격(영어 'Winter'·'Summer' 가 떨어지게) · 기둥–층 라벨 · 치수선 간격 · ALT 줄 간격
    xr = x1 + 0.06                                       # 오른쪽 라벨 끝(열 안쪽 가장자리)
    ALT_LINES = ("Active layer", "thickness (ALT)")          # 두 줄: 층 이름과 약어(치수선 라벨이 층 이름을 겸한다)
    xw0 = XAX + 0.12; xw1 = xw0 + WW; xs0 = xw1 + GAP; xs1 = xs0 + WW
    y_surf, y_bot = H - 0.66, 0.45; ppm = (y_surf - y_bot) / 2.0  # 2 m 기둥
    y_front = y_surf - alt_m * ppm
    ym = (y_surf + y_front) / 2
    xl = xs1 + LAB                                       # 층 이름 x(활동층·영구동토, 왼맞춤)
    r = fig.canvas.get_renderer()
    xa = xs1 + BR                                        # 치수선 x(여름 기둥 바로 오른쪽; 층 이름은 치수선 라벨이 겸한다)
    xlab = xa + 0.12                                     # ALT 라벨 x
    plt.rcParams["hatch.linewidth"] = 0.6

    # 깊이 축(왼쪽): 0·1·2 m
    M.vline(ax, XAX, y_bot, y_surf, color=M.ZERO, lw=0.8)
    for d in (0, 1, 2):
        yy = y_surf - d * ppm
        M.hline(ax, XAX - 0.05, XAX, yy, color=M.ZERO, lw=0.8)
        SMALL_OK.add(id(M.T(ax, XAX - 0.09, yy, f"{d} m", size=FS_K, ha="right")))   # 깊이 축 눈금(12 pt 허용)

    # 두 기둥(같은 폭): 겨울(활동층 동결 빗금) · 여름(활동층 융해)
    for cx0, cx1, frozen_top in ((xw0, xw1, True), (xs0, xs1, False)):
        w = cx1 - cx0
        ax.add_patch(Rectangle((cx0, y_front), w, y_surf - y_front, facecolor=AL_FILL, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((cx0, y_bot), w, y_front - y_bot, facecolor=PF_FILL, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((cx0, y_bot), w, y_front - y_bot, facecolor="none", edgecolor=PF_HATCH, hatch="//", lw=0, zorder=3))
        if frozen_top:
            ax.add_patch(Rectangle((cx0, y_front), w, y_surf - y_front, facecolor="none", edgecolor=AL_HATCH, hatch="//", lw=0, zorder=3))
        ax.add_patch(Rectangle((cx0, y_bot), w, y_surf - y_bot, facecolor="none", edgecolor=M.ZERO, lw=0.9, zorder=4))
        M.hline(ax, cx0, cx1, y_front, color=M.ZERO, lw=0.9, z=4)          # 영구동토 상단면
        M.hline(ax, cx0, cx1, y_surf, color=M.INK, lw=1.8, z=5)            # 지표(굵은 선)
    M.T(ax, (xw0 + xw1) / 2, y_surf + 0.20, "Winter", size=FS_T, ha="center")
    M.T(ax, (xs0 + xs1) / 2, y_surf + 0.20, "Summer", size=FS_T, ha="center")

    # 층 이름: 기둥 오른쪽에 붙인 글자(활동층은 위에서 만들었다) · 경계 이름: 이음선 라벨, ALT 라벨과 같은 x 에 왼맞춤,
    # 지표는 선 위에, 영구동토 상단면은 선 아래에 두어 치수선 바깥에 놓는다
    M.T(ax, xl, (y_front + y_bot) / 2, "Permafrost", size=FS_T, ha="left")
    t_gs = M.T(ax, xlab, y_surf + 0.035, "Ground surface", size=FS_T, ha="left", va="bottom")
    t_tab = M.T(ax, xlab, y_front - 0.035, "Permafrost table", size=FS_T, ha="left", va="top")
    assert max(t_tab.get_window_extent(r).x1, t_gs.get_window_extent(r).x1) / fig.dpi <= xr, "경계 라벨이 열 밖으로 나간다"
    M.hline(ax, xs1, xlab - 0.08, y_surf, color=M.ZERO, lw=0.8, z=1)
    M.hline(ax, xs1, xlab - 0.08, y_front, color=M.ZERO, lw=0.8, z=1)

    # ALT 치수선(화살촉 없음): 지표 ↔ 영구동토 상단면, 세 줄 라벨
    dim_bracket(ax, xa, y_front, y_surf)
    for k, line in enumerate(ALT_LINES):
        M.T(ax, xlab, ym + ((len(ALT_LINES) - 1) / 2 - k) * PITCH, line, size=FS_T, ha="left")

    # 열쇠(오른쪽 아래, 세로): 색 = 층, 빗금 = 동결 상태
    leg = legend_column(ax, xl, 1.15)
    SPEC["choices"]["schematic_in"] = dict(axis_x=XAX, winter=[xw0, xw1], summer=[xs0, xs1], column_width=WW, y_surface=y_surf, y_bottom=y_bot,
                                           y_front=round(float(y_front), 3), alt_m=round(float(alt_m), 4), layer_label_x=round(xl, 3),
                                           bracket_x=round(float(xa), 3), bracket_style="vertical line + end ticks, no arrowheads",
                                           alt_label_x=round(float(xlab), 3), alt_label_lines=list(ALT_LINES), alt_line_pitch_in=PITCH,
                                           legend=dict(entries=leg, x=round(xl, 3), y_top=1.15, encoding="색 = 층, 빗금 = 동결 상태"))


# ================================================================ 그림
def g0():
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    for (x0, x1), s in zip(COLS, HEADS):
        M.head(ax, x0, Y_HEAD, s)
        M.hline(ax, x0, x1, Y_HEADRULE, color=M.HAIR, lw=0.8)
    for x in SEPX:
        M.vline(ax, x, 0.12, Y_HEADRULE - 0.06, color=M.SEP, lw=0.9, ls=(0, (3, 3)))

    # ---- 자료
    lat, lon, g = samoylov_daily()
    i = int(g.z.idxmax()); doy_max = int(g.doy[i]); alt_cm = float(g.z[i])
    date_max = dt.date(YEAR, 1, 1) + dt.timedelta(days=doy_max - 1)
    bounds, names = month_axis(YEAR)
    xlo, xhi = bounds[0] - 7, bounds[-1] + 1                # 왼쪽 7일: '6월' 글자가 y 눈금 글자와 닿지 않게
    dd, cum = F2._daily_tdd(lat, lon, YEAR)
    SPEC["sources"]["tdd"] = "data/raw/era5land/nh_monthly_2010-2024.nc (v7_f2_problem_time._daily_tdd, forcing_clim.csv e5_grid)"
    tdd_at = float(np.interp(doy_max, dd, cum)); tdd_end = float(np.interp(xhi, dd, cum))
    SPEC["numbers"] = dict(alt_cm=round(alt_cm, 2), alt_doy=doy_max, alt_date=date_max.isoformat(), n_probe_on_alt_day=int(g.n[i]),
                           sd_on_alt_day=round(float(g.sd[i]), 2), tdd_on_alt_day=round(tdd_at, 1), tdd_at_axis_end=round(tdd_end, 1),
                           daily=dict(doy=g.doy.tolist(), mean_cm=g.z.round(2).tolist(), sd_cm=g.sd.round(2).tolist(), n=g.n.tolist()))

    # ---- (a) 토양 기둥
    draw_column_schematic(fig, ax, alt_cm / 100.0)

    # ---- (b) 계절 안 융해 깊이(깊이 축 아래 방향)
    ab = chart_axes(fig, 1)
    sd = g.sd.fillna(0.0)
    ab.fill_between(g.doy, g.z - sd, g.z + sd, color=SAM, alpha=0.25, lw=0, zorder=2)
    ab.plot(g.doy, g.z, color=SAM, lw=1.2, zorder=3)
    ab.plot(g.doy, g.z, "o", ms=5.0, mfc=SAM, mec="white", mew=0.7, ls="none", zorder=4)
    ab.axhline(alt_cm, color=M.INK, lw=1.0, ls=DASH, zorder=2)
    ab.text(xlo + 3, alt_cm - 1.0, f"ALT {alt_cm:.1f} cm", ha="left", va="bottom", fontsize=FS_T, color=M.INK)
    ab.axvline(doy_max, color=M.AUX, lw=0.9, ls=DOT, zorder=1)
    date_lab = f"{date_max.day} {MONTHS_EN[date_max.month - 1]}"
    ab.text(doy_max - 2.5, 0.8, date_lab, ha="right", va="top", fontsize=FS_T, color=M.INK)
    months(ab, bounds, names, xlo, xhi)
    ytop = float(np.ceil((g.z + sd).max() / 5) * 5 + 2)
    ab.set_yticks([0, 10, 20, 30, 40]); ab.set_ylim(ytop, 0)
    ab.set_ylabel("Thaw depth (cm)", fontsize=FS_T, labelpad=3)
    # 첫 관측일: 6월 구간이 비어 있는 까닭(관측 시작) 을 첫 점 왼쪽에 두 줄로 적는다
    doy_first = int(g.doy.iloc[0]); z_first = float(g.z.iloc[0])
    date_first = dt.date(YEAR, 1, 1) + dt.timedelta(days=doy_first - 1)
    cm_per_line = 0.21 * ytop / (AX_T - AX_B)            # 13 pt 줄 간격(0.21 in)을 깊이(cm)로
    ab.text(doy_first - 2.5, z_first - cm_per_line / 2, "First survey", ha="right", va="center", fontsize=FS_T, color=M.AUX)
    ab.text(doy_first - 2.5, z_first + cm_per_line / 2, f"{date_first.day} {MONTHS_EN[date_first.month - 1]}", ha="right", va="center", fontsize=FS_T, color=M.AUX)
    note(ax, COLS[1][1] - 0.15, NOTES[0])
    SPEC["numbers"]["first_obs"] = dict(doy=doy_first, date=date_first.isoformat(), mean_cm=round(z_first, 2))

    # ---- (c) 같은 해 누적 융해 도일
    ac = chart_axes(fig, 2)
    ac.plot(dd, cum, color=M.INK, lw=1.8, zorder=3)
    ac.axvline(doy_max, color=M.AUX, lw=0.9, ls=DOT, zorder=1)
    ac.plot([doy_max], [tdd_at], "o", ms=7.5, mfc=SAM, mec="white", mew=0.8, zorder=5)
    ymax = tdd_end * 1.15
    ac.text(doy_max - 3, tdd_at + 0.02 * ymax, f"{tdd_at:.0f} °C·d", ha="right", va="bottom", fontsize=FS_T, color=M.INK)
    ac.text(doy_max - 2.5, 0.985 * ymax, date_lab, ha="right", va="top", fontsize=FS_T, color=M.INK)
    months(ac, bounds, names, xlo, xhi)
    step = 100.0 if ymax < 450 else (200.0 if ymax < 1100 else 500.0)
    ac.set_yticks(np.arange(0.0, ymax + 1e-9, step)); ac.set_ylim(0, ymax)
    ac.set_ylabel("Cumulative TDD (°C·d)", fontsize=FS_T, labelpad=3)
    note(ax, COLS[2][1] - 0.15, NOTES[1])
    SPEC["choices"]["x_axis_doy"] = dict(lo=xlo, hi=xhi, month_ticks=bounds, tick_rule="1st of month (F2 convention)", labels=names)
    SPEC["choices"]["notes"] = dict(text=list(NOTES), size_pt=FS_K, color=M.AUX, y_in=Y_NOTE, why="자료 출처 각주; 눈금 크기(12 pt) 로 라벨(13 pt) 보다 작게")

    # ---- 점검: 글자 크기(눈금 외 13 pt 미만), y축 라벨 회전, 겹침(M.qa_and_save)
    fig.canvas.draw()
    ticks = tick_label_artists(fig) | SMALL_OK
    below13 = [[t.get_text()[:30], float(t.get_fontsize())] for t in fig.findobj(Text)
               if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FS_T - 1e-6
               and not (id(t) in ticks and t.get_fontsize() >= FS_K - 1e-6)]
    ylabels = [ab.yaxis.get_label().get_text(), ac.yaxis.get_label().get_text()]
    no_hangul(fig)
    rec = M.qa_and_save(fig, STEM, allow_overlap=())
    from PIL import Image
    src_png, src_pdf = M.OUT / f"{STEM}_slide.png", M.OUT / f"{STEM}_slide.pdf"
    png, pdf = M.OUT / f"{STEM}.png", M.OUT / f"{STEM}.pdf"
    with Image.open(src_png) as im:                   # RGBA → RGB(흰 바탕)
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        px = list(im.size)
    bg.save(png, dpi=(M.DPI, M.DPI)); src_png.unlink()
    if src_pdf.exists():
        src_pdf.replace(pdf)
    SPEC["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    SPEC["qa"]["png_px"] = px; SPEC["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    SPEC["qa"]["non_tick_text_below_13pt"] = below13
    SPEC["qa"]["ylabel_rotation"] = ylabel_rotation_check(ylabels)
    bad = [k for k, v in SPEC["qa"]["ylabel_rotation"].items() if not v["ok"]]
    print("y축 라벨 회전 점검:", "OK" if not bad else f"CHECK {bad}", flush=True)
    print("글자 크기 점검(눈금 외 13 pt 미만):", "OK" if not below13 else f"CHECK {below13}", "· 화소", px, flush=True)
    (M.OUT / f"{STEM}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
    print("출력:", png, pdf, M.OUT / f"{STEM}_spec.json")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", action="store_true", help="호환용. 이 그림은 슬라이드판 하나만 만든다")
    ap.add_argument("--fig", default="G0", help="호환용")
    ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    g0()
