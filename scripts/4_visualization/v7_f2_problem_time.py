"""v7 F2: 문제(입력의 시간 정의)와 물리 근거(Stefan √TDD). figures/figure_spec_v7_F2.json.
실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f2_problem_time.py [--slide | --deck] [--fig 2]
CPU 만. 수치는 전부 data/processed/v5 의 CSV 와 원자료에서 읽는다.
패널: a1 한 알래스카 셀의 2014 일별 누적 TDD 곡선, 관측일 표식, '관측일까지'·'연간' 기준선(삽도가 아니라 독립 패널)
      a2 같은 셀의 연도별 TDD 막대(관측 해 = 지역 색)·기후값 파선·관측 ALT 원(오른쪽 전체 축, 진회색 spine·눈금)
      b  레나 장기 지점(연도별 제품 3종이 모두 값을 가진 셀 중 관측 연도 최다)의 관측 ALT(검은 빈 원, 열쇠에 관측 연도 수)와
         Wei 2026(실선)·Liu 2024(파선)·ESA CCI v5(점선)의 연도 값. 각 계열의 마지막 자료 연도에 제품 색 점(끝점 표식),
         직접 라벨은 그 끝점 옆(오른쪽 우선, 다른 선·원·열쇠·라벨과 ≥ 5.5 pt 떨어진 첫 자리를 자동 탐색)
      c  계절 안 융해 깊이 대 √(관측일 누적 TDD): 일별 평균 = 작은 반투명 점, ± SD = 지점 색 띠(alpha 0.2), Stefan 적합(ALLena 원자료)
      d  다년 셀(관측 4년 이상)의 셀 안 편차: (ALT − 셀 평균) 대 (√TDD − 셀 평균 √TDD). 알래스카 = 옅은 밀도 육각 격자, 레나 델타·캐나다 = 지역 색 점,
         지역별 원점 통과 회귀선, 열쇠에 기울기(cm per √(°C·d), 그림에서는 mathtext)
판: 논문(기본, pdf·svg·png) · 슬라이드 크롭(--slide, slide/<stem>__<panels>.png) · 덱(--deck, deck/<stem>__deck_<SLIDE>.png)
덱: 슬라이드 본문 한 장 = 12.0 × 4.65 in, 300 dpi(3600 × 1395 px, RGB 흰 바탕). N02 = a1 | a2 | b(한 줄), N04 = c | d.
    글자는 토큰 font.slide(축 라벨 14·눈금 12·직접 라벨 13·패널 문자 14 굵게), 선·표식은 lines.slide 비율. 자료 처리·그리기 함수는 논문판과 같다.
개정(10-15, 설계 지침 v8 항 6): a 삽도 → a1·a2 분리(겹침 0), a2 오른쪽 축 전체 길이(진회색), b 관측 열쇠에 연도 수, CCI 점선,
      c 오차 막대 → 작은 점 + ± SD 띠(alpha 0.2), d 셀별 연결선 → 셀 안 편차 밀도 + 지역 회귀선(기울기 라벨). 1:1 기준선 없음.
개정 2(10-15 검토 반영): b 계열 끝점 표식 + 끝점 옆 라벨(Liu 2020·CCI 2023 에서 끝남이 보이도록), a1 라벨 간격 5.5 pt(점 단위 변위),
      a2 막대 = 지역 색 원색(F1 e·F7 과 같은 채움, alpha 없음)·기후값 파선에 흰 테두리, 논문 PDF 쪽 크기를 qa 에 기록.
개정 3(10-15, 설계 지침 v9 항 7·8·12·14): 덱 12.0 × 4.65 in(3600 × 1395 px). N02 패널 간격·축 안 글자와 y 축 사이 여유 확대(축 안 글자는 축선에서
      K.in_pad_pt 떨어짐), a2 오른쪽 그래프에 패널 문자 a′(덱 설명 줄의 a′). 'climatological forcing' → 'mean climate input'(2011–2024 평균 TDD).
      제곱근은 mathtext($\sqrt{\mathrm{TDD}}$, 윗줄)만 쓴다(글리프 '√' 없음). d: 알래스카(셀-연도 1,808)는 알래스카 색의 옅은 밀도 육각 격자,
      레나 델타(원)·캐나다(사각)는 토큰 지역 색 점(흰 테두리)으로 위에 그리고, 지역 회귀선은 흰 테두리. 색 막대 = 알래스카 셀-연도 수.
"""
from __future__ import annotations
import argparse, calendar, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import polar.testbed_v5 as T
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.transforms import offset_copy

PROC = ROOT / "data" / "processed" / "v5"; RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK = OUT / "deck"
STEM = "F2_problem_time"
REG = ["Alaska", "Lena", "Canada"]
YEARS = np.arange(2010, 2025)
EOS_DOY = 244                                   # CALM 'eos'(날짜 없음)의 관측일 규약 = 9월 1일(testbed_v5.with_forcing 과 같다)
DASH = (0, (3.5, 1.8)); DOT = (0, (1.0, 1.5))
_TINT = S.TOK["color"].get("region_tints", {}).get("Lena Delta", {})
SITE_HEX = {"Samoylov": _TINT.get("dark", "#1B86BB"), "Kurungnakh": _TINT.get("light", S.REGION_HEX["Lena"])}   # 레나 지역 색의 두 명도
PROD = ["wei", "liu", "cci5y"]
PROD_HEX = {p: S.PRODUCT_HEX[p] for p in PROD}                                   # 토큰 color.products 와 같은 값(style_v5)
_LS = {"solid": "-", "dashed": DASH, "dotted": DOT, "dashdot": "-."}
PROD_LS = {"wei": "-", "liu": _LS[S.TOK["color"]["products"].get("Liu", {}).get("line", "dashed")], "cci5y": DOT}   # Liu 토큰 파선, CCI v5 점선
PROD_LS_NAME = {"wei": "solid", "liu": "dashed", "cci5y": "dotted"}
SQ_UNIT = r"$\sqrt{{}^{\circ}\mathrm{C{\cdot}d}}$"                     # √(°C·d): 제곱근은 mathtext 로만(윗줄 포함, 글리프 '√' 금지; 설계 지침 v9 항 8)
SQ_TDD = r"$\sqrt{\mathrm{TDD}}$"
E_UNIT = "cm/" + SQ_UNIT                                     # Stefan 계수 E 의 단위(ALT = E √TDD)
SLOPE_UNIT = "cm per " + SQ_UNIT                             # d 편차 회귀 기울기의 단위
E_UNIT_TXT = "cm/√(°C·d)"; SLOPE_UNIT_TXT = "cm per √(°C·d)"   # source_data JSON 용(글자 그대로)
AUX = S.NEUTRAL["text_aux"]; GREY = S.NEUTRAL["zero_line"]
BAND_ALPHA = float(S.TOK["color"].get("ci_band_alpha", 0.2))
def _tint(hex_, f):
    """같은 색상의 밝은 명도(f = 흰색 쪽 비율)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hex_)); c = c + (1 - c) * f
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


AK_DENS = [_tint(S.REGION_HEX["Alaska"], 0.90), _tint(S.REGION_HEX["Alaska"], 0.55)]   # 알래스카 밀도: 지역 색(토큰)의 옅은 두 명도(위에 그리는 레나·캐나다 점보다 밝다)
DENS_CMAP = LinearSegmentedColormap.from_list("dens_ak", AK_DENS)
D_MARK = {"Lena": "o", "Canada": "s"}                         # d 의 점 모양(색 외의 구분)
A1_PAD_DAYS = 20.0                                            # a1 x 축 왼쪽 여유(일): 축 모서리의 눈금 라벨 'Jan' 과 '0' 사이 간격
YEAR_XLIM = (2008.9, 2024.6)                                  # a2·b 연도 축: 2010 눈금 라벨이 y 눈금 '0' 과 붙지 않게 왼쪽 여유 1.1 년
CLIM_LABEL = "mean climate input"                             # a2 파선 = 2011–2024 평균 TDD(평년 기후 입력). 'forcing' 표현 금지(지침 v9 항 7)
D_XLIM = (-8.0, 6.0); D_YLIM = 45.0                           # d 축 범위(편차, 대칭 y). 밖의 점은 세지 않고 source_data 에 수를 적는다

# ---------------------------------------------------------------- 판별 크기 표. 자료·배치 규칙은 같고 숫자로 넘기는 크기만 다르다
_K_PAPER = dict(mode="paper", fs_direct=6.5, fs_aux=6.0, fs_key=6.5, lw=1.0, ms=1.0, a_ytop=1.55, a_band=(0.56, 0.92),
                lab_gap_pt=5.5, cb_x=(0.66, 0.26), in_pad_pt=4.0, d_s=5.0)        # lab_gap_pt: 라벨–선·라벨–라벨 최소 간격(pt), in_pad_pt: 축 안 글자–축선(pt)
_K_DECK = dict(mode="deck", fs_direct=13.0, fs_aux=12.0, fs_key=12.0, lw=1.8, ms=1.8, a_ytop=1.55, a_band=(0.56, 0.92),
               lab_gap_pt=7.0, cb_x=(0.66, 0.26), in_pad_pt=9.0, d_s=26.0)        # d_s: d 의 레나·캐나다 점 면적(pt²)
END_MS = 2.2                                                  # b 계열 끝점 표식 지름(pt × K.ms). 관측 원(3.0)보다 작고 속이 찬 제품 색
_CAND = [("left", "center", 1.0, 0.0), ("right", "bottom", -0.7, 0.7), ("left", "bottom", 0.7, 0.7), ("center", "bottom", 0.0, 1.0),
         ("right", "top", -0.7, -0.7), ("left", "top", 0.7, -0.7), ("center", "top", 0.0, -1.0)]
# b 직접 라벨 후보(ha, va, 단위 변위 x, y): 끝점 오른쪽 → 왼쪽 위 → 오른쪽 위 → 위 → 왼쪽 아래 → 오른쪽 아래 → 아래. 변위 크기는 4·6·8·11·15 pt 순
K = _K_PAPER


def set_mode(mode):
    """paper / deck. 덱은 S.setup(scale=1.69) 뒤 rcParams 를 토큰 font.slide·lines.slide 로 바꾼다(style_v5 의 scale 은 저장 dpi 만 바꾼다)."""
    global K
    K = _K_DECK if mode == "deck" else _K_PAPER
    if mode == "deck":
        fs = S.TOK["font"]["slide"]; ln = S.TOK["lines"]["slide"]
        plt.rcParams.update({
            "font.size": fs["direct_label_pt"], "axes.labelsize": fs["axis_label_pt"], "xtick.labelsize": fs["tick_pt"], "ytick.labelsize": fs["tick_pt"],
            "legend.fontsize": fs["tick_pt"], "axes.linewidth": ln["axis_pt"], "xtick.major.width": ln["tick_pt"], "ytick.major.width": ln["tick_pt"],
            "xtick.major.size": ln["tick_len_pt"], "ytick.major.size": ln["tick_len_pt"], "lines.linewidth": ln["data_pt"], "lines.markersize": ln["marker_pt"],
            "axes.labelpad": 6.0, "xtick.major.pad": 4.0, "ytick.major.pad": 4.0})


# ================================================================ 자료 도우미(v6_figs.py 에서 복사)
_GRID = None


def _grid_of(lat, lon):
    """라벨 셀 표(forcing_clim)에서 가장 가까운 셀의 ERA5-Land 사용 격자(육지 폴백 포함) 좌표."""
    global _GRID
    if _GRID is None:
        _GRID = pd.read_csv(PROC / "forcing_clim.csv", usecols=["loc_id", "e5_grid_lat", "e5_grid_lon"]).merge(
            pd.read_csv(PROC / "static_v5.csv", usecols=["loc_id", "lat", "lon"]), on="loc_id")
    j = ((_GRID.lat - lat) ** 2 + (_GRID.lon - lon) ** 2).idxmin()
    return float(_GRID.e5_grid_lat[j]), float(_GRID.e5_grid_lon[j])


def _daily_tdd(lat, lon, year):
    """ERA5-Land 월평균에서 일별 누적 TDD(월 누적값의 선형 보간). 반환 (doy, 누적 TDD)."""
    import xarray as xr
    ds = xr.open_dataset(RAW / "era5land" / "nh_monthly_2010-2024.nc")
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    glat, glon = _grid_of(lat, lon)
    t = ds["t2m"].sel(latitude=glat, longitude=glon, method="nearest")
    yrs = t[tn].dt.year.values
    Tm = t.values[yrs == year] - 273.15
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    cum = np.cumsum(np.clip(Tm, 0, None) * days); starts = np.r_[0, np.cumsum(days)]
    doy = np.arange(1, starts[-1] + 1); out = np.interp(doy, np.r_[0, starts[1:]], np.r_[0, cum])
    return doy, out


def _allena_series():
    """ALLena 원자료(PANGAEA 973813)에서 사모일로프 2015·쿠룽나흐 2020 계절 안 융해 깊이. (이름, lat, lon, year, doy, z)."""
    al = pd.read_csv(RAW / "allena" / "PANGAEA_973813_ALLena_main.txt", sep="\t", skiprows=437, low_memory=False)
    col = lambda p: [c for c in al.columns if c.startswith(p)][0]
    al["dt"] = pd.to_datetime(al[col("Date/Time (")]); al = al[(al[col("QF TD")] != 2) & al[col("Thaw depth mean [cm]")].notna()]
    al["year"] = al.dt.dt.year; al["doy"] = al.dt.dt.dayofyear
    out = []
    for name, lat, lon, yr in (("Samoylov", 72.374, 126.497, 2015), ("Kurungnakh", 72.373, 126.495, 2020)):
        s = al[(al.year == yr) & (al[col("Latitude (")].round(3) == lat) & (al[col("Longitude (")].round(3) == lon)]
        out.append((name, lat, lon, yr, s.doy.values.astype(float), s[col("Thaw depth mean [cm]")].values.astype(float)))
    return out


def _box_pts_dist(bb, pts):
    """상자(px)와 점 집합(px)의 최소 거리(px). 상자 안의 점은 0."""
    if len(pts) == 0:
        return float("inf")
    dx = np.maximum(np.maximum(bb.x0 - pts[:, 0], pts[:, 0] - bb.x1), 0.0); dy = np.maximum(np.maximum(bb.y0 - pts[:, 1], pts[:, 1] - bb.y1), 0.0)
    return float(np.hypot(dx, dy).min())


def _box_box_dist(a, b):
    """두 상자(px)의 최소 거리(px). 겹치면 0."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0); dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return float(np.hypot(dx, dy))


def _place_end_labels(ax, ser, obs_xy, r_obs_pt, r_end_pt, fixed, margin_pt):
    """b 직접 라벨: 각 계열의 마지막 자료점 옆에 둔다. 후보 = _CAND × 변위(4·6·8·11·15 pt). 채택 조건: 다른 계열의 선·끝점 표식, 관측 원,
    고정 글자(좌표 라벨·열쇠), 먼저 놓은 라벨과 margin_pt 이상 떨어지고, 그림 오른쪽 가장자리 안·축 세로 범위 안. 자기 선은 끝점 근방
    (변위 + margin)만 빼고 겹침(여백 1.5 pt)만 검사한다. 끝 연도가 늦은 계열부터 놓는다. 반환: 이름 → 자리·간격(pt) 표(qa)."""
    fig = ax.figure; fig.canvas.draw(); rr = fig.canvas.get_renderer(); ppt = fig.dpi / 72.0; m = margin_pt * ppt
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False); ring = np.c_[np.cos(ang), np.sin(ang)]
    ring_px = lambda c, r_pt: c + r_pt * ppt * ring
    def poly(x, y):
        seg = [ax.transData.transform(np.c_[np.linspace(x0, x1, 60), np.linspace(y0, y1, 60)]) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
        return np.concatenate(seg) if seg else ax.transData.transform(np.c_[x, y])
    samp = {s["key"]: poly(s["x"], s["y"]) for s in ser}
    ends = {s["key"]: ax.transData.transform((s["x"][-1], s["y"][-1])) for s in ser}
    obs_px = ax.transData.transform(np.asarray(obs_xy, float).reshape(-1, 2))
    obs_pts = np.concatenate([ring_px(c, r_obs_pt) for c in obs_px]) if len(obs_px) else np.empty((0, 2))
    fixed_bb = [f.get_window_extent(rr) for f in fixed]
    axbb = ax.get_window_extent(rr); W = fig.get_figwidth() * fig.dpi
    m_own = min(1.5 * ppt, m)                                                   # 자기 선: 글자를 덮지만 않으면 된다(같은 색·같은 계열)
    hit_pts = lambda bb, pts, mm=m: bool(((pts[:, 0] >= bb.x0 - mm) & (pts[:, 0] <= bb.x1 + mm) & (pts[:, 1] >= bb.y0 - mm) & (pts[:, 1] <= bb.y1 + mm)).any())
    hit_box = lambda bb, ob: not (bb.x1 + m < ob.x0 or bb.x0 - m > ob.x1 or bb.y1 + m < ob.y0 or bb.y0 - m > ob.y1)
    out = {}; boxes = []
    for s in sorted(ser, key=lambda s: -s["x"][-1]):
        k = s["key"]; xe, ye = float(s["x"][-1]), float(s["y"][-1])
        others = np.concatenate([samp[j] for j in samp if j != k] + [ring_px(ends[j], r_end_pt) for j in ends if j != k] + [obs_pts])
        txt = None; found = False
        for k_pt in (4.0, 6.0, 8.0, 11.0, 15.0):
            for ha, va, ux, uy in _CAND:
                dx, dy = ux * k_pt, uy * k_pt
                if txt is not None:
                    txt.remove()
                txt = ax.text(xe, ye, s["name"], ha=ha, va=va, fontsize=K["fs_direct"], color=s["color"], clip_on=False, zorder=6,
                              transform=offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points"))
                bb = txt.get_window_extent(rr)
                near = (np.hypot(dx, dy) + margin_pt + 1.0) * ppt
                own = samp[k]; own = own[np.hypot(own[:, 0] - ends[k][0], own[:, 1] - ends[k][1]) > near]
                found = (bb.x1 + 2.0 * ppt <= W and bb.x0 >= axbb.x0 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1
                         and not hit_pts(bb, others) and not hit_pts(bb, own, m_own) and not any(hit_box(bb, ob) for ob in fixed_bb + boxes))
                if found:
                    break
            if found:
                break
        clear = min([_box_pts_dist(bb, others)] + [_box_box_dist(bb, ob) for ob in fixed_bb + boxes]) / ppt
        out[s["name"]] = dict(end_year=int(round(xe)), end_value_cm=ye, ha=ha, va=va, dx_pt=dx, dy_pt=dy, found=bool(found),
                              clearance_pt=round(clear, 2), own_line_clearance_pt=round(_box_pts_dist(bb, own) / ppt, 2),
                              bbox_in=[round(v / fig.dpi, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)])
        boxes.append(bb)
    gaps = [_box_box_dist(a, b) / ppt for i, a in enumerate(boxes) for b in boxes[i + 1:]]
    out["_min_label_gap_pt"] = round(min(gaps), 2) if gaps else float("inf")
    out["_min_clearance_pt"] = round(min(v["clearance_pt"] for v in out.values() if isinstance(v, dict)), 2)
    out["_right_in"] = round(max(b.x1 for b in boxes) / fig.dpi, 3)
    return out


def _doy_cum(row, year):
    """forcing_year 한 행의 tdd_cum_1..12 → 일별 누적 TDD(월말 값 선형 보간)."""
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)]); ends = np.cumsum(days)
    cum = np.array([row[f"tdd_cum_{k}"] for k in range(1, 13)], float)
    doy = np.arange(1, ends[-1] + 1)
    return doy, np.interp(doy, np.r_[0, ends], np.r_[0, cum])


def _clearance_pt(ax, txt, x, y, side="below"):
    """글자 상자와 곡선(x, y) 사이의 세로 간격(pt). side='below' 면 글자가 곡선 위에 있어야 하는 구간, 'above' 는 그 반대. 음수 = 겹침."""
    fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = txt.get_window_extent(r); inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    if len(x) < 50:                                       # 두 점짜리 기준선은 촘촘히 보간한다
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 400); y = np.interp(xs, x, y); x = xs
    sel = (x >= x0) & (x <= x1)
    if not sel.any():
        return float("inf")
    px_per_unit = (bb.y1 - bb.y0) / (y1 - y0)
    d = (y0 - float(np.max(y[sel]))) if side == "below" else (float(np.min(y[sel])) - y1)
    return d * px_per_unit * 72.0 / fig.dpi


def _bbox_data(ax, art):
    """글자·표식의 화면 상자를 축 자료 좌표로 (x0, y0, x1, y1)."""
    fig = ax.figure; fig.canvas.draw(); bb = art.get_window_extent(fig.canvas.get_renderer()); inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, y0, x1, y1


def _key(ax, handles, labels, loc="upper left", **kw):
    """테두리 없는 열쇠. 적합 계수·기울기를 단위와 함께 적는 패널(c, d)과 b 의 관측 항목에 쓴다."""
    o = dict(frameon=False, fontsize=K["fs_key"], handlelength=1.8, handletextpad=0.6, borderaxespad=0.1, borderpad=0.1, labelspacing=0.45); o.update(kw)
    return ax.legend(handles, labels, loc=loc, **o)


# ================================================================ 자료 선택
def pick_alaska_cell(d):
    """관측 연도 수가 가장 많은 알래스카 셀. 동률이면 관측일까지 누적 TDD 비율(중앙값)이 가장 작은 셀(그 다음 loc_id 오름차순)."""
    ak = d[d.region == "Alaska"]; n = ak.groupby("loc_id").year.nunique(); ids = n[n == n.max()].index
    f = ak[ak.loc_id.isin(ids)].groupby("loc_id").tdd_obs_frac.median().sort_index()
    return int(f.sort_values(kind="stable").index[0]), int(n.max())


def pick_lena_cell(ss, pr):
    """연도별 제품 3종(wei, liu, cci5y)이 모두 값을 가진 레나 셀 중 관측 연도 수 최다(동률은 loc_id 오름차순).
    관측 연도가 가장 긴 셀(loc_id 14813, 8년)은 Wei·Liu 화소가 결측(물 마스크)이라 제외된다(source_data 에 기록)."""
    obs = ss[(ss.region == "Lena") & (ss.observed == 1)]; n_all = obs.groupby("loc_id").year.nunique()
    p = pr[(pr.year >= 2010) & (pr.year <= 2024) & pr["product"].isin(PROD)]
    ok = p.groupby(["loc_id", "product"]).value_cm.apply(lambda s: s.notna().any()).unstack().fillna(False)
    ok = ok.index[ok.all(axis=1)]
    n = n_all[n_all.index.isin(ok)].sort_index()
    lid = int(n.sort_values(ascending=False, kind="stable").index[0])
    longest = int(n_all.sort_values(ascending=False, kind="stable").index[0])
    return lid, int(n[lid]), dict(loc_id=longest, n_years=int(n_all[longest]), products_available=sorted(
        p[(p.loc_id == longest)].dropna(subset=["value_cm"])["product"].unique().tolist()))


def load_data():
    """패널이 함께 쓰는 표. 논문판·슬라이드판·덱이 같은 함수를 쓴다."""
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    fy = pd.read_csv(PROC / "forcing_year.csv"); fc = pd.read_csv(PROC / "forcing_clim.csv")
    pr = pd.read_csv(PROC / "products_cell_year_v5.csv")
    ss = pd.read_csv(PROC / "site_series" / "site_series_year_obs.csv.gz")
    d = T.with_forcing(T.load_v5(PROC, products=False), "year_obs")          # e5_sqrt_tdd = 관측일까지, tdd_obs_frac
    return dict(lab=lab, fy=fy, fc=fc, pr=pr, ss=ss, d=d)


# ================================================================ 패널 그리기(축 하나를 받는다)
def draw_a1(ax, D, src, qa):
    """a1 선택 셀의 2014 일별 누적 TDD 곡선, 관측일(CALM 은 9월 1일 규약) 표식, '관측일까지'·'연간' 기준선. x 축 라벨 = 연도.
    기준선 라벨은 선에서 K.lab_gap_pt(논문 5.5 pt) 만큼 점 단위로 띄운다(판·배율에 무관)."""
    fig = ax.figure; lab, fy, d = D["lab"], D["fy"], D["d"]
    cell, nyr = pick_alaska_cell(d)
    yi = 2014; row = fy[(fy.loc_id == cell) & (fy.year == yi)].iloc[0]; doy, cum = _doy_cum(row, yi)
    od = lab[(lab.loc_id == cell) & (lab.year == yi)].obs_doy; od = float(od.iloc[0]) if len(od) and np.isfinite(od.iloc[0]) else float(EOS_DOY)
    c_obs = float(np.interp(od, doy, cum)); c_full = float(cum[-1]); ytop = c_full * 1.42
    ax.plot(doy, cum, color="black", lw=1.0 * K["lw"], zorder=3)
    ax.plot([od, od], [0, c_obs], color=AUX, lw=0.6 * K["lw"], ls=DOT, zorder=2)
    x0d = -A1_PAD_DAYS                                                                             # 축 왼쪽 여유: 'Jan' 눈금 라벨이 y 눈금 '0' 과 붙지 않게
    ax.plot([x0d, od], [c_obs, c_obs], color=AUX, lw=0.6 * K["lw"], ls=DOT, zorder=2)
    ax.plot([x0d, doy[-1]], [c_full, c_full], color=AUX, lw=0.6 * K["lw"], ls=DOT, zorder=2)
    ax.plot([od], [c_obs], "o", ms=3.0 * K["ms"], mfc="white", mec="black", mew=0.7 * K["lw"], zorder=4)
    up = offset_copy(ax.transData, fig=fig, x=K["in_pad_pt"], y=K["lab_gap_pt"], units="points")
    dn = offset_copy(ax.transData, fig=fig, x=K["in_pad_pt"], y=-K["lab_gap_pt"], units="points")                       # 글자는 y 축선에서 in_pad_pt 오른쪽
    t_full = ax.text(x0d, c_full, "full year", ha="left", va="bottom", fontsize=K["fs_aux"], color=AUX, transform=up)          # 선에서 정확히 lab_gap_pt 위
    t_obs = ax.text(x0d, c_obs, "observation\nday", ha="left", va="top", fontsize=K["fs_aux"], color=AUX, linespacing=1.05, transform=dn)
    ax.text(od + 7, c_obs * 0.42, "1 Sep" if od == EOS_DOY else f"day {od:.0f}", ha="left", va="center", fontsize=K["fs_aux"], color=AUX)
    ax.set_xlim(x0d, doy[-1]); ax.set_ylim(0, ytop)
    ax.set_xticks([1, 91, 182, 274]); ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    ystep = float([s for s in (50, 100, 200, 250, 500, 1000) if c_full / s <= 3.2][0])      # 눈금 2–3개, 고른 간격
    yt = np.arange(0, c_full + 1e-9, ystep); ax.set_yticks(yt); ax.spines["left"].set_bounds(0, yt[-1]); ax.spines["bottom"].set_bounds(x0d, doy[-1])
    ax.set_xlabel(f"{yi}"); ax.set_ylabel("Cumulative TDD (°C·d)")
    qa["a1_clearance_pt"] = dict(to_obs_vs_curve=_clearance_pt(ax, t_obs, doy, cum, "below"),
                                 to_obs_vs_line=_clearance_pt(ax, t_obs, np.array([0.0, od]), np.array([c_obs, c_obs]), "above"),
                                 full_vs_line=_clearance_pt(ax, t_full, np.array([0.0, doy[-1]]), np.array([c_full, c_full]), "below"),
                                 full_vs_curve=_clearance_pt(ax, t_full, doy, cum, "below"))
    lat0, lon0 = float(lab[lab.loc_id == cell].lat.iloc[0]), float(lab[lab.loc_id == cell].lon.iloc[0])
    src["a1"] = dict(loc_id=cell, n_years=nyr, lat=lat0, lon=lon0, year=yi, obs_day=od, obs_day_rule="CALM eos: pipeline convention 1 Sep (doy 244)",
                     tdd_to_obs_day=c_obs, tdd_full=c_full, cum_month_end=[float(row[f"tdd_cum_{k}"]) for k in range(1, 13)])
    return cell


def draw_a2(ax, D, src, qa):
    """a2 같은 셀의 연도별 TDD 막대(관측 해 = 지역 색 원색, F1 e·F7 의 지역 막대와 같은 채움; 그 밖 = 무자료 회색), 기후값 파선(흰 테두리로
    막대 위에서도 읽힌다), 관측 ALT 원(오른쪽 전체 축, 진회색). 반환 ax2."""
    fig = ax.figure; lab, fy, fc, d = D["lab"], D["fy"], D["fc"], D["d"]
    cell, nyr = pick_alaska_cell(d)
    f = fy[fy.loc_id == cell].set_index("year").reindex(YEARS)
    o = lab[lab.loc_id == cell].groupby("year").alt_cm.mean().reindex(YEARS)
    clim = float(fc.loc[fc.loc_id == cell, "clim1124_tdd"].iloc[0]) if "clim1124_tdd" in fc else float(f.loc[2011:2024, "e5_tdd"].mean())
    obs_yrs = o.index[o.notna()]
    cols = [S.REGION_HEX["Alaska"] if y in obs_yrs else S.NEUTRAL["no_data"] for y in YEARS]
    ax.bar(YEARS, f.e5_tdd.values, width=0.72, color=cols, lw=0, zorder=1)
    ax.axhline(clim, color=GREY, lw=0.8 * K["lw"], ls=DASH, zorder=2, path_effects=[pe.withStroke(linewidth=2.0 * K["lw"], foreground="white")])
    tmax = float(np.nanmax(f.e5_tdd)); ytop = tmax * K["a_ytop"]; ax.set_ylim(0, ytop)      # 위쪽 1/3 은 관측 ALT 원의 띠
    ax.set_xlim(*YEAR_XLIM); ax.set_xticks([2010, 2014, 2018, 2022]); ax.set_xlabel("Year"); ax.set_ylabel("Thawing degree-days (°C·d)")
    yt = ax.get_yticks(); yt = [t for t in yt if t <= tmax] + [t for t in yt if t > tmax][:1]   # 막대 범위 + 한 눈금
    ax.set_yticks(yt); ax.spines["left"].set_bounds(0, yt[-1])
    # 오른쪽 축: 관측 ALT(검은 빈 원). 원 띠가 패널 높이의 a_band(0.42–0.86)에 들도록 범위를 잡는다. spine·눈금·라벨은 진회색, 전체 길이
    ax2 = ax.twinx()
    amin, amax = float(o.min()), float(o.max()); b0, b1 = K["a_band"]; span = (amax - amin) / (b1 - b0)
    lo = amin - b0 * span; hi = lo + span
    pts = ax2.plot(obs_yrs, o[obs_yrs].values, "o", ms=3.0 * K["ms"], mfc="white", mec="black", mew=0.7 * K["lw"], zorder=4, clip_on=False)[0]
    ax2.set_ylim(lo, hi); ax2.set_ylabel("Observed ALT (cm)", color=GREY)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(GREY); ax2.spines["left"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=GREY, length=plt.rcParams["ytick.major.size"], width=plt.rcParams["ytick.major.width"])
    step = 10 if span <= 120 else 20
    ax2.set_yticks(np.arange(np.ceil(lo / step) * step, hi + 1e-9, step))
    # 기후값 라벨: 글자 폭을 재서 막대·원과 겹치지 않는 가장 낮은 창을 고르고 선 바로 위에 둔다
    txt = ax.text(2015, clim * 1.03, CLIM_LABEL, ha="center", va="bottom", fontsize=K["fs_aux"], color=GREY, zorder=3)
    x0, y0, x1, y1 = _bbox_data(ax, txt); hw = (x1 - x0) / 2 * 1.05; hh = (y1 - y0)
    r_px = 1.5 * K["ms"] * 1.3 * fig.dpi / 72.0                                                   # 원 반지름(px, 여유 포함)
    r_x = r_px / (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]); r_y = r_px / (ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    circ = [(float(y), (float(o[y]) - lo) / (hi - lo) * ytop) for y in obs_yrs]                     # 원 중심을 왼쪽 축 단위로
    best = None
    xl0, xl1 = ax.get_xlim()
    for xc in np.arange(xl0 + hw + 0.1, xl1 - hw - 0.1, 0.05):                                    # 라벨 전체가 축 안에 든다
        tops = [h for y, h in zip(YEARS, f.e5_tdd.values) if (y + 0.36 > xc - hw) and (y - 0.36 < xc + hw) and np.isfinite(h)]
        yb = max([clim * 1.03] + [t * 1.02 for t in tops])
        hit = any((cx + r_x > xc - hw) and (cx - r_x < xc + hw) and (cy + r_y > yb) and (cy - r_y < yb + hh) for cx, cy in circ)
        cand = (hit, yb, xc)
        if best is None or cand < best:
            best = cand
    hit, yb, xc = best; txt.set_x(xc); txt.set_y(yb)
    qa["a2_clim_label"] = dict(x=float(xc), y=float(yb), overlaps_circle=bool(hit))
    lat0, lon0 = float(lab[lab.loc_id == cell].lat.iloc[0]), float(lab[lab.loc_id == cell].lon.iloc[0])
    ax.text(0, 1, f"Alaska, {lat0:.1f}°N {abs(lon0):.1f}°W", transform=offset_copy(ax.transAxes, fig=fig, x=K["in_pad_pt"], y=-1.0, units="points"),
            ha="left", va="top", fontsize=K["fs_aux"], color=AUX)
    src["a2"] = dict(clim_label=CLIM_LABEL, loc_id=cell, n_years=nyr, lat=lat0, lon=lon0, clim_tdd_2011_2024=clim, years=YEARS.tolist(),
                     e5_tdd=[None if not np.isfinite(v) else float(v) for v in f.e5_tdd.values],
                     alt_cm=[None if not np.isfinite(v) else float(v) for v in o.values], observed_years=[int(y) for y in obs_yrs],
                     right_axis=dict(lo=lo, hi=hi, band=list(K["a_band"])), bar_fill="region colour at full saturation (same as F1 e, F7)",
                     clim_line="dashed, zero_line grey, white halo 2.0 x lw")
    return ax2


def draw_b(ax, D, src, qa):
    """b 레나 장기 지점: 관측(검은 빈 원, 열쇠에 연도 수) 대 제품 3종의 연도 값(실선·파선·점선). 각 계열의 마지막 자료점에 제품 색 점(끝점 표식),
    직접 라벨은 그 끝점 옆(_place_end_labels: 다른 선·원·열쇠·라벨과 ≥ K.lab_gap_pt). 끝 연도가 다른 계열(Liu 2020, CCI 2023)이 그대로 보인다."""
    fig = ax.figure; lab, pr, ss = D["lab"], D["pr"], D["ss"]
    lid, nl, longest = pick_lena_cell(ss, pr)
    obs = ss[(ss.loc_id == lid) & (ss.observed == 1)].sort_values("year")
    ser = []
    for p in PROD:
        s = pr[(pr.loc_id == lid) & (pr["product"] == p) & pr.year.between(2010, 2024)].dropna(subset=["value_cm"]).sort_values("year")
        c = PROD_HEX[p]
        ax.plot(s.year, s.value_cm, ls=PROD_LS[p], color=c, lw=1.0 * K["lw"], zorder=2, solid_capstyle="round", dash_capstyle="round")
        ax.plot([s.year.iloc[-1]], [s.value_cm.iloc[-1]], "o", ms=END_MS * K["ms"], color=c, mec="none", ls="none", zorder=3)   # 끝점 표식
        ser.append(dict(key=p, name=S.PRODUCT_EN[p], x=s.year.values.astype(float), y=s.value_cm.values.astype(float), color=c))
        src[f"b_{p}"] = dict(zip(s.year.astype(int).tolist(), s.value_cm.astype(float).tolist()))
    h_obs = ax.plot(obs.year, obs.y, "o", ms=3.0 * K["ms"], mfc="white", mec="black", mew=0.7 * K["lw"], ls="none", zorder=4)[0]
    ax.set_xlim(*YEAR_XLIM); ax.set_xticks([2010, 2014, 2018, 2022]); ax.spines["bottom"].set_bounds(2010, 2024)
    ymax = max(float(obs.y.max()), float(pr[(pr.loc_id == lid) & pr["product"].isin(PROD) & pr.year.between(2010, 2024)].value_cm.max()))
    ax.set_ylim(0, ymax * 1.12); ax.set_xlabel("Year"); ax.set_ylabel("ALT (cm)")
    lat1, lon1 = float(lab[lab.loc_id == lid].lat.iloc[0]), float(lab[lab.loc_id == lid].lon.iloc[0])
    t_coord = ax.text(0, 1, f"Lena Delta, {lat1:.1f}°N {lon1:.1f}°E", transform=offset_copy(ax.transAxes, fig=fig, x=K["in_pad_pt"], y=-1.0, units="points"),
                      ha="left", va="top", fontsize=K["fs_aux"], color=AUX)
    fig.canvas.draw(); _bb = t_coord.get_window_extent(fig.canvas.get_renderer()); _ab = ax.get_window_extent(fig.canvas.get_renderer())
    leg = _key(ax, [h_obs], [f"observed ALT ({nl} years)"], loc="upper left", handletextpad=0.3,
               bbox_to_anchor=((K["in_pad_pt"] * fig.dpi / 72.0 - 0.1 * K["fs_key"] * fig.dpi / 72.0) / _ab.width, (_bb.y0 - _ab.y0) / _ab.height - 0.01))
    # 열쇠·좌표 라벨 아래의 자료(Wei 2023 봉우리)가 글자와 겹치지 않도록 y 상한을 잰 값으로 올린다: 열쇠 x 범위 안 자료 최고점이 열쇠 바닥에서 lab_gap_pt 아래
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); axbb = ax.get_window_extent(rr); ppt = fig.dpi / 72.0; m = K["lab_gap_pt"] * ppt
    pts = np.concatenate([np.c_[e["x"], e["y"]] for e in ser] + [np.c_[obs.year.values, obs.y.values]])
    need = ax.get_ylim()[1]
    for art in (leg, t_coord):
        bb = art.get_window_extent(rr); x0, x1 = ax.transData.inverted().transform([(bb.x0 - m, 0), (bb.x1 + m, 0)])[:, 0]
        sel = (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
        if sel.any():
            f_bot = (bb.y0 - m - axbb.y0) / axbb.height                       # 글자 바닥(여백 포함)의 축 높이 비율(글자는 축 좌표에 고정)
            need = max(need, float(pts[sel, 1].max()) / f_bot)
    ax.set_ylim(0, need); yt = [t for t in ax.get_yticks() if 0 <= t <= need]; ax.set_yticks(yt); ax.spines["left"].set_bounds(0, yt[-1])
    qa["b_ylim_top"] = dict(ylim=round(need, 1), ymax_data=round(ymax, 1), key_clear_pt=K["lab_gap_pt"])
    placed = _place_end_labels(ax, ser, obs_xy=np.c_[obs.year.values, obs.y.values], r_obs_pt=(3.0 * K["ms"] + 0.7 * K["lw"]) / 2,
                               r_end_pt=END_MS * K["ms"] / 2, fixed=[t_coord, leg], margin_pt=K["lab_gap_pt"])
    qa["b_labels"] = placed; qa["b_label_min_gap_pt"] = placed["_min_label_gap_pt"]; qa["b_label_min_clearance_pt"] = placed["_min_clearance_pt"]
    qa["b_label_right_in"] = placed["_right_in"]                                   # 가장 오른쪽 라벨의 끝(인치, 그림 왼쪽 기준)
    src["b"] = dict(loc_id=lid, n_years=nl, lat=lat1, lon=lon1, observed=dict(zip(obs.year.astype(int).tolist(), obs.y.astype(float).tolist())),
                    series_end={e["name"]: dict(year=int(e["x"][-1]), value_cm=float(e["y"][-1])) for e in ser}, colors=PROD_HEX, linestyle=PROD_LS_NAME,
                    end_marker=f"filled dot in product colour at the last data year, {END_MS} pt x K.ms",
                    label_placement={n: dict(ha=v["ha"], va=v["va"], dx_pt=v["dx_pt"], dy_pt=v["dy_pt"]) for n, v in placed.items() if isinstance(v, dict)},
                    longest_record_cell_excluded=longest)


def draw_c(ax, D, src):
    """c 계절 안 융해 깊이 대 √TDD(관측일), Stefan 적합. 일별 평균 = 작은 반투명 점, ± SD = 지점 색 띠(alpha 0.2)."""
    series = _allena_series(); xmax = 0.0; fits = []
    for name, lat, lon, yr, doy_s, z in series:
        dd, cum = _daily_tdd(lat, lon, yr); tdd = np.interp(doy_s, dd, cum)
        g = pd.DataFrame({"doy": doy_s, "tdd": tdd, "z": z}).groupby("doy").agg(tdd=("tdd", "first"), z=("z", "mean"), sd=("z", "std"), n=("z", "size"))
        E = float((np.sqrt(tdd) @ z) / (np.sqrt(tdd) @ np.sqrt(tdd)))               # 원점 통과 최소제곱(모든 탐침값)
        c = SITE_HEX[name]; zo = 3 + (name == "Kurungnakh"); sd = g.sd.fillna(0)
        ax.fill_between(np.sqrt(g.tdd), g.z - sd, g.z + sd, color=c, alpha=BAND_ALPHA, lw=0, zorder=zo - 1)
        ax.plot(np.sqrt(g.tdd), g.z, "o", ms=2.1 * K["ms"], color=c, mew=0, alpha=0.95, ls="none", zorder=zo + 2)   # 밝은 명도(쿠룽나흐)도 흰 바탕에서 보이게 불투명에 가깝게
        xmax = max(xmax, float(np.sqrt(tdd.max()))); fits.append((name, yr, E, c))
        src[f"c_{name}_{yr}"] = dict(lat=lat, lon=lon, E=E, E_unit=E_UNIT_TXT, color=c, n_probe=int(len(z)), n_days=int(len(g)), band="mean ± 1 sd per day",
                                     sqrt_tdd=np.sqrt(g.tdd).round(2).tolist(), thaw_depth_cm=g.z.round(2).tolist(), sd_cm=sd.round(2).tolist())
    xe = xmax * 1.12; xx = np.linspace(0, xe, 20)
    for name, yr, E, c in fits:
        ax.plot(xx, E * xx, color=c, lw=0.9 * K["lw"], zorder=2)
    xt_end = np.ceil(xe / 10) * 10; ax.set_xticks(np.arange(0, xt_end + 1e-9, 10)); ax.spines["bottom"].set_bounds(0, xt_end)
    ax.set_xlim(0, xt_end * 1.03); ax.set_ylim(0, max(E for _, _, E, _ in fits) * xe * 1.12)
    ax.set_xlabel(f"{SQ_TDD} to observation day ({SQ_UNIT})"); ax.set_ylabel("Thaw depth (cm)")
    _key(ax, [Line2D([], [], color=c, lw=0.9 * K["lw"], marker="o", ms=1.9 * K["ms"], mec=c) for _, _, _, c in fits],
         [f"{name} {yr}, $E$ = {E:.2f} {E_UNIT}" for name, yr, E, _ in fits], bbox_to_anchor=(K["in_pad_pt"] / 72.0 * ax.figure.dpi / ax.get_window_extent().width, 1.0))


def draw_d(ax, D, src):
    """d 다년 셀(관측 4년 이상)의 셀 안 편차: (ALT − 셀 평균) 대 (√TDD − 셀 평균). 알래스카(셀-연도 최다) = 지역 색의 옅은 밀도 육각 격자(아래 층),
    레나 델타(원)·캐나다(사각) = 토큰 지역 색 점(흰 테두리, 위 층), 지역별 원점 통과 회귀선(흰 테두리). 열쇠·색 막대는 자료 위 띠. 반환 cax."""
    fig = ax.figure; d = D["d"]
    n = d.groupby("loc_id").year.nunique(); m = d[d.loc_id.isin(n[n >= 4].index)].copy()
    g = m.groupby("loc_id"); m["dx"] = m.e5_sqrt_tdd - g.e5_sqrt_tdd.transform("mean"); m["dy"] = m.alt_cm - g.alt_cm.transform("mean")
    inside = m.dx.between(*D_XLIM) & m.dy.between(-D_YLIM, D_YLIM)
    ak = m[(m.region == "Alaska") & inside]
    hb = ax.hexbin(ak.dx, ak.dy, gridsize=(24, 14), extent=(*D_XLIM, -D_YLIM, D_YLIM), cmap=DENS_CMAP, norm=LogNorm(vmin=1),
                   mincnt=1, linewidths=0.25 * K["lw"], edgecolors="face", zorder=1)
    ax.axhline(0, color=S.NEUTRAL["coast"], lw=0.5 * K["lw"], zorder=2)
    ax.plot([0, 0], [-D_YLIM, D_YLIM], color=S.NEUTRAL["coast"], lw=0.5 * K["lw"], zorder=2)      # 자료 범위까지만(위쪽 띠의 열쇠를 가로지르지 않는다)
    for r in ("Canada", "Lena"):                                  # 셀-연도가 많은 캐나다(219)를 먼저, 레나(134)를 위에
        sr = m[(m.region == r) & inside]
        ax.scatter(sr.dx, sr.dy, s=K["d_s"], marker=D_MARK[r], facecolor=S.REGION_HEX[r], edgecolor="white", linewidths=0.35 * K["lw"],
                   alpha=0.9, zorder=4 + (r == "Lena"))
    fits = {}
    for r in REG:
        sr = m[m.region == r]; b = float((sr.dx @ sr.dy) / (sr.dx @ sr.dx))                         # 편차는 셀마다 평균 0 → 원점 통과 = OLS
        x0, x1 = float(np.percentile(sr.dx, 1)), float(np.percentile(sr.dx, 99))
        fits[r] = (b, x0, x1)
        src[f"d_{r}"] = dict(n_cells=int(sr.loc_id.nunique()), n_cell_years=int(len(sr)), n_shown=int(inside[sr.index].sum()), slope=b, slope_unit=SLOPE_UNIT_TXT,
                             r=float(np.corrcoef(sr.dx, sr.dy)[0, 1]), dx_range_1_99=[x0, x1],
                             drawn_as="light density hexbin (region tint)" if r == "Alaska" else f"points ({D_MARK[r]}), region colour")
    halo = [pe.withStroke(linewidth=2.4 * K["lw"], foreground="white")]
    for r in ("Canada", "Lena", "Alaska"):                        # 알래스카(셀-연도 최다)를 맨 위에. 흰 테두리로 점·밀도 위에서 읽힌다
        b, x0, x1 = fits[r]
        ax.plot([x0, x1], [b * x0, b * x1], color=S.REGION_HEX[r], lw=1.1 * K["lw"], zorder=8, solid_capstyle="round", path_effects=halo)
    ax.set_xlim(*D_XLIM); ax.set_ylim(-D_YLIM, D_YLIM)
    ax.set_xticks(np.arange(-8, 6.1, 2)); ax.set_yticks(np.arange(-40, 40.1, 20))
    ax.set_xlabel(f"{SQ_TDD} − cell mean ({SQ_UNIT})"); ax.set_ylabel("ALT − cell mean (cm)")
    order = ["Alaska", "Lena", "Canada"]; ms_pt = np.sqrt(K["d_s"])
    hs = [Line2D([], [], color=S.REGION_HEX["Alaska"], lw=1.1 * K["lw"], marker="h", ms=ms_pt * 1.25, mfc=AK_DENS[1], mec=AK_DENS[1])]
    hs += [Line2D([], [], color=S.REGION_HEX[r], lw=1.1 * K["lw"], marker=D_MARK[r], ms=ms_pt, mfc=S.REGION_HEX[r], mec="white", mew=0.35 * K["lw"]) for r in order[1:]]
    leg = _key(ax, hs, [f"{S.REGION_EN[r]}, {fits[r][0]:.2f} {SLOPE_UNIT}".replace("-", "−") for r in order], loc="upper left", handlelength=2.2,
               bbox_to_anchor=(K["in_pad_pt"] / 72.0 * fig.dpi / ax.get_window_extent().width, 1.0))
    # 열쇠·색 막대는 자료 위가 아니라 자료 없는 위쪽 띠(y > D_YLIM)에 둔다: 열쇠 높이를 재서 y 상한을 그만큼 늘리고 spine 은 ±D_YLIM 까지만
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    f_key = leg.get_window_extent(rr).height / ax.get_window_extent(rr).height; band = f_key + 0.05
    ytop = D_YLIM + 2 * D_YLIM * band / (1 - band); ax.set_ylim(-D_YLIM, ytop); ax.spines["left"].set_bounds(-D_YLIM, D_YLIM)
    cx0, cw = K["cb_x"]
    fig.canvas.draw(); bl0 = leg.get_window_extent(rr); axb0 = ax.get_window_extent(rr)
    cx0 = max(cx0, (bl0.x1 - axb0.x0) / axb0.width + 0.06)                                        # 색 막대는 열쇠 오른쪽 끝에서 ≥ 6 % 축 폭 떨어진다
    cw = min(cw, 0.985 - cx0)
    cax = ax.inset_axes([cx0, 1 - 0.56 * band, cw, 0.17 * band])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4 * K["lw"]); cb.minorticks_off()
    cmax = int(hb.get_array().max()); ticks = [t for t in (1, 3, 10, 30, 100) if t <= cmax]
    cb.set_ticks(ticks); cb.set_ticklabels([str(t) for t in ticks]); cb.ax.tick_params(labelsize=K["fs_aux"], length=1.5 * K["lw"], width=0.4 * K["lw"], pad=1.5 * K["lw"])
    t_cb = ax.text(cx0, 1 - 0.04 * band, "Alaska cell-years", transform=ax.transAxes, ha="left", va="top", fontsize=K["fs_aux"], color=S.NEUTRAL["text_paper"])
    fig.canvas.draw(); bl = leg.get_window_extent(rr); bc = cax.get_window_extent(rr); bt = t_cb.get_window_extent(rr)
    src["d"] = dict(n_cells=int(m.loc_id.nunique()), n_cell_years=int(len(m)), n_shown=int(inside.sum()), n_outside_axes=int((~inside).sum()),
                    xlim=list(D_XLIM), ylim=[-D_YLIM, D_YLIM], band_top=ytop, gridsize=[24, 14], max_count_alaska=cmax,
                    colormap=f"Alaska density, region tint {AK_DENS[0]}–{AK_DENS[1]}, log counts", lena_canada="points in region colour (tokens), white edge",
                    qa=dict(key_frac=round(f_key, 3), band_frac=round(band, 3), key_to_colorbar_gap_pt=round((min(bc.x0, bt.x0) - bl.x1) * 72.0 / fig.dpi, 1),
                            key_bottom_above_data_pt=round((bl.y0 - ax.transData.transform((0, D_YLIM))[1]) * 72.0 / fig.dpi, 1)))
    return cax


def _write_source(src, qa, name=STEM):
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    src["qa"] = qa
    (OUT / "source_data" / f"{name}.json").write_text(json.dumps(src, ensure_ascii=False, indent=1, default=float))
    print("QA a1 clearance (pt):", {k: round(v, 2) for k, v in qa["a1_clearance_pt"].items()}, "| a2 clim label:", qa["a2_clim_label"])
    print("QA b labels:", {n: (v["ha"], v["va"], v["dx_pt"], v["dy_pt"], "found" if v["found"] else "NOT FOUND", v["clearance_pt"]) for n, v in qa["b_labels"].items() if isinstance(v, dict)},
          "| min label gap (pt):", qa["b_label_min_gap_pt"], "| min clearance (pt):", qa["b_label_min_clearance_pt"], "| right (in):", qa["b_label_right_in"])
    if "pdf_page_mm" in qa:
        print("QA pdf page (mm):", qa["pdf_page_mm"])


def _pdf_page_mm(p):
    """pdfinfo 로 논문 PDF 의 쪽 크기(mm). bbox_inches='tight' 저장이라 그림 캔버스(180 × 125 mm)보다 작을 수 있다(저널은 폭 ≤ 180 mm 만 요구)."""
    import subprocess, re
    try:
        o = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, timeout=30).stdout
        w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", o).groups()
        return [round(float(w) * 25.4 / 72.0, 1), round(float(h) * 25.4 / 72.0, 1)]
    except Exception:
        return None


# ================================================================ 배치 도우미(인치 또는 mm 로 축 위치를 준다)
def _ax(fig, x0, y0, w, h, unit="mm"):
    k = S.MM if unit == "mm" else 1.0
    W, H = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([x0 * k / W, y0 * k / H, w * k / W, h * k / H])


def _letter(fig, ax, s, x, y=None, unit="mm", size=None):
    """패널 문자: 그림 좌표(x 는 그림 왼쪽에서 unit 단위, y 는 축 위 여백 또는 지정값)."""
    k = S.MM if unit == "mm" else 1.0
    bb = ax.get_position(); W, H = fig.get_figwidth(), fig.get_figheight()
    yy = bb.y1 + (1.5 * S.MM / H) if y is None else y * k / H
    fig.text(x * k / W, yy, s, fontsize=size or S.F["paper"]["panel_letter_pt"], fontweight="bold", ha="left", va="bottom")
    S.PANELS.setdefault(s, []).append(ax)


# ================================================================ 논문판·슬라이드판
def f2():
    D = load_data(); src = {}; qa = {}
    H = 125.0; fig = S.figure(S.W2, H)
    yT, yB, hA = 73.0, 11.0, 45.0
    a1 = _ax(fig, 11.0, yT, 24.0, hA); draw_a1(a1, D, src, qa); _letter(fig, a1, "a", 1.0)
    a2 = _ax(fig, 49.0, yT, 54.0, hA); ax2 = draw_a2(a2, D, src, qa); S.register("a", a2, ax2)
    b = _ax(fig, 127.0, yT, 40.0, hA); draw_b(b, D, src, qa); _letter(fig, b, "b", 114.0)
    c = _ax(fig, 11.0, yB, 71.0, hA); draw_c(c, D, src); _letter(fig, c, "c", 1.0)
    d = _ax(fig, 107.0, yB, 71.0, hA); cax = draw_d(d, D, src); _letter(fig, d, "d", 95.0); S.register("d", cax)
    S.register("ab", *S.PANELS["a"], *S.PANELS["b"]); S.register("cd", *S.PANELS["c"], *S.PANELS["d"])
    paths = S.export(fig, STEM, outdir=OUT)
    pdf = [p for p in paths if str(p).endswith(".pdf")]
    if pdf:
        qa["pdf_page_mm"] = _pdf_page_mm(pdf[0]); qa["canvas_mm"] = [S.W2, H]
    _write_source(src, qa, name=STEM if S.SCALE == 1.0 else f"{STEM}__slide")       # 슬라이드판 qa 는 따로 적는다(논문판 기록을 덮지 않음)
    return paths


# ================================================================ 덱(슬라이드 본문 한 장 = 캔버스 하나)
def _deck_save(fig, slide, H, qa):
    """deck/<stem>__deck_<slide>.png, 300 dpi, bbox 자르기 없음(정확히 3600 × H·300 px), RGB 흰 바탕. 캔버스 사용률을 qa 에 적는다."""
    from PIL import Image
    DECK.mkdir(parents=True, exist_ok=True); p = DECK / f"{STEM}__deck_{slide}.png"
    fig.canvas.draw(); tb = fig.get_tightbbox(fig.canvas.get_renderer())                       # 인치
    qa[f"{slide}_canvas_fill"] = dict(width=round(tb.width / fig.get_figwidth(), 3), height=round(tb.height / fig.get_figheight(), 3),
                                      x0_in=round(tb.x0, 2), x1_in=round(tb.x1, 2), y0_in=round(tb.y0, 2), y1_in=round(tb.y1, 2))
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    im = Image.open(p); im.load(); rgb = im.convert("RGB"); im.close(); rgb.save(p, dpi=(300, 300))
    want = (3600, int(round(H * 300)))
    if rgb.size != want:
        raise RuntimeError(f"deck png size {rgb.size} != {want}: {p}")
    qa[f"{slide}_px"] = list(rgb.size)
    return p


def _deck_fig(H):
    """12.0 × H in 캔버스. mm 왕복의 부동소수 오차(3599 px)를 막으려고 인치로 다시 맞춘다."""
    fig = S.figure(12.0 * 25.4, H * 25.4); fig.set_size_inches(12.0, H)
    return fig


def deck():
    """덱 N02·N04(12.0 × 4.65 in). 가로 배치(in): 축 왼쪽 여백 = y 라벨·눈금 폭, 패널 사이 여백 ≥ 0.40 in(이웃 패널의 눈금·축 라벨 끝 사이)."""
    D = load_data(); src = {}; qa = {}; out = []
    LP = S.TOK["font"]["slide"]["panel_letter_pt"]; H = 4.65; y0, hA = 0.62, 3.58; yL = 4.43
    # N02: a1 | a2(오른쪽 축 = 관측 ALT) | b. b 오른쪽은 계열 끝 라벨 자리(≈ 0.75 in)
    fig = _deck_fig(H)
    a1 = _ax(fig, 0.86, y0, 2.20, hA, "in"); draw_a1(a1, D, src, qa); _letter(fig, a1, "a", 0.04, yL, "in", LP)
    a2 = _ax(fig, 4.40, y0, 2.40, hA, "in"); draw_a2(a2, D, src, qa); _letter(fig, a2, "a′", 3.52, yL, "in", LP)
    b = _ax(fig, 8.66, y0, 2.50, hA, "in"); draw_b(b, D, src, qa); _letter(fig, b, "b", 7.80, yL, "in", LP)
    out.append(_deck_save(fig, "N02", H, qa))
    # N04: c | d. 아래 여백 0.74 in: mathtext 제곱근 x 라벨(윗줄 포함)이 캔버스 안에 든다
    fig = _deck_fig(H); y4, h4 = 0.74, 3.46
    c = _ax(fig, 0.72, y4, 4.70, h4, "in"); draw_c(c, D, src); _letter(fig, c, "c", 0.04, yL, "in", LP)
    d = _ax(fig, 6.82, y4, 5.02, h4, "in"); draw_d(d, D, src); _letter(fig, d, "d", 5.92, yL, "in", LP)
    out.append(_deck_save(fig, "N04", H, qa))
    _write_source(src, qa, name=f"{STEM}__deck")
    print("QA deck canvas fill:", {k: v for k, v in qa.items() if k.endswith("_canvas_fill")}, "| px:", {k: v for k, v in qa.items() if k.endswith("_px")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, default=2); ap.add_argument("--slide", action="store_true")
    ap.add_argument("--deck", action="store_true", help="슬라이드 본문용 PNG(12 × 4.65 in, 300 dpi): deck/F2_problem_time__deck_N02.png, _N04.png")
    a = ap.parse_args()
    if a.deck:
        S.setup(scale=1.69); set_mode("deck"); paths = deck()
    else:
        S.setup(scale=1.69 if a.slide else 1.0); set_mode("paper"); paths = f2()
    for p in paths:
        print(p)
