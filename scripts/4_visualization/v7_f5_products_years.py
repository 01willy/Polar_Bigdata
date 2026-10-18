"""v7 F5: 제품 비교와 연도별 거동(a 공통 셀 RMSE, b 짝지은 차이, c 연도 간 변화 밀도, d 연도별 편향, e–f 장기 지점).
실행(CPU 만): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f5_products_years.py [--slide] [--deck] [--report]
계약: docs/FIGURE_CONTRACT_v7.md, 설계 지침 v9(docs/DESIGN_BRIEF_v9_2026-10-15.md 2.3 항 20·21). 수치는 모두 CSV 에서 읽는다.
격자(mm): 왼쪽 열 축 시작 XL, 오른쪽 열 축 XB–XE 공통, XE–XR 은 오른쪽 열 전체의 직접 라벨 홈(b·d·f 축 오른끝이 같다).
2026-10-14 검토 반영(2차):
  c  runs/core_v6/preds 의 블록 CV OOF(씨앗 평균) 로 yearwise/multiyear_changes.csv 와 같은 정의(관측 연도의 연속 쌍, 간격 무관) 로 계산하고 r·n 을 전체 쌍에서 적는다.
  a  범위 밖(> 45 cm) 표식은 RMSE 값(cm) 이다(둘 다 RMSE: 25 km 밖 부분집합의 RMSE 는 전체보다 클 수 있다). 단위를 붙여 적고 셀-연도 수는 별도 줄에 둔다.
  b  끊긴 축(아래 토막 −132…−42 cm) 으로 CCI v5 의 −102·−63 cm 를 그린다. 값을 바닥에 붙이지 않는다.
  e·f Liu 2024 는 토큰 color.products.Liu 의 색·파선.
덱 판(--deck, v9 2026-10-15): 덱 본문 12.0 × 4.65 in(3600 × 1395 px), 300 dpi, outputs/figures/paper/v7/deck/F5_products_years__deck_R04..R06.png.
  R04 같은 셀 · 같은 해 비교: 지역별 패널의 행별 점 그림(행 = 우리 방법 셋, 옅은 파선, 제품 다섯; x = RMSE cm). 채운 원 = 공통 셀-연도 전체,
      빈 원 = 제품 학습 지점 25 km 밖(common_cells_eval.csv mask 0 · 25). 45 cm 를 넘는 값은 축 끝에 마커와 숫자로 적는다. 그림 안 글자는 범례·축 라벨뿐.
  R05 연도별 평가: 지역별 패널, 관측 연도마다 같은 셀-연도에서 Physics + residual ML · Stefan (recalibrated) · Wei 2026 · Liu 2024 의 RMSE.
      셀-연도 집합: 그 해 R1 · P1 · Wei 가 모두 값을 가진 라벨 셀-연도, Liu(2010–2020)도 값을 가진 것이 30 이상이면 Liu 를 넣은 집합.
      30 미만인 해는 뺀다. 아래 띠 = 그 해 셀-연도 수(로그 축). 표: outputs/figures/paper/v7/source_data/F5_yearwise_rmse.csv(mask 0 · 25 km).
  R06 장기 관측 지점: 위쪽 범례(관측 · 관측일 예측 · 계절 말 예측 · Stefan 계절 말 · Wei 2026 · Liu 2024), 패널 머리에 지역 · 좌표.
그리기 함수(draw_a·draw_b·draw_c·draw_d·draw_site) 는 축과 크기 묶음(Sty) 을 받아 논문·슬라이드가 같은 자료·같은 논리를 쓴다.
"""
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

PROC = ROOT / "data" / "processed" / "v5"; RUN = PROC / "runs" / "core_v6"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK = OUT / "deck"; STEM = "F5_products_years"
REG = ["Alaska", "Lena", "Canada"]
OFF = {"Alaska": -0.24, "Lena": 0.0, "Canada": 0.24}
W, H = S.W2, 142.0                       # 논문 캔버스(mm)
XL, LX = 13.0, 3.0                       # 왼쪽 열: 축 시작·패널 문자 열(mm)
XB, XE, XR, LXR = 113.0, 165.0, 177.0, 100.0   # 오른쪽 열: 축 시작·축 끝·라벨 홈 끝·패널 문자 열(mm)
DW, DH = 12.0 * 25.4, 4.65 * 25.4        # 덱 캔버스(mm): 덱 본문 12.0 × 4.65 in(v9 지침 1.4)
AUX = S.NEUTRAL["text_aux"]; ZERO = S.NEUTRAL["zero_line"]; HAIR = "#BDBDBD"
PTOK = S.C["products"]                   # 토큰 color.products 가 정본
PHEX = {"wei": PTOK["Wei"]["hex"], "liu": PTOK["Liu"]["hex"]}
DASH = (0, (3.5, 1.8))
PLS = {p: (DASH if PTOK[k].get("line") == "dashed" else "-") for p, k in (("wei", "Wei"), ("liu", "Liu"))}
PRED_FILE = "{r}__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz"   # 블록 CV OOF, 물리 + 잔차 ML(주 설정)
CAP, Y0A = 45.0, 10.0                    # a: 표시 상한·하한(cm)
B_HI, B_LO = (-30.0, 12.0), (-132.0, -42.0)   # b: 끊긴 축 위·아래 토막 범위(cm)
LIM = 50                                 # c: ±창(cm)
YR = (2010, 2024)                        # e·f: 연도 범위
ORDER_A = ["P1 (Stefan, recal.)", "R1 (CatBoost)", "D0 (CatBoost)", "Wei 2026", "Liu 2024", "ESA CCI v4", "ESA CCI v5", "Aalto 2018"]
SHORT_A = ["Stefan\n(recalibrated)", "Physics +\nresidual\nML", "Direct\nML", "Wei 2026", "Liu 2024", "CCI v4", "CCI v5", "Aalto 2018"]
PRODS_B = ["wei", "liu", "cci4y", "cci5y"]; SHORT_B = ["Wei 2026", "Liu 2024", "CCI v4", "CCI v5"]


@dataclass(frozen=True)
class Sty:
    """크기 묶음(pt·mm). 논문(7 pt 본문) 과 덱(축 라벨 14·눈금 12·직접 라벨 13·패널 문자 14 pt) 의 두 벌."""
    deck: bool
    text: float; small: float; tiny: float; tick: float; letter: float
    ms: float; ms_obs: float; ms_sq: float; mew: float
    lw: float; lw_ref: float; lw_main: float; lw_p1: float; lw_prod: float; hex_lw: float; cb_lw: float
    tick_half_mm: float; break_mm: float; b_gap_mm: float; b_lo_mm: float


PAPER = Sty(False, 7.0, 6.0, 5.0, 6.0, 8.0, 3.0, 3.2, 2.2, 0.8, 0.8, 0.6, 1.1, 0.9, 0.7, 0.15, 0.5, 0.8, 1.0, 1.5, 10.0)
DECKS = Sty(True, 14.0, 13.0, 12.0, 12.0, 14.0, 5.5, 6.0, 4.2, 1.3, 1.4, 1.0, 2.0, 1.6, 1.3, 0.3, 0.8, 1.6, 2.0, 3.0, 27.0)


# ---------------------------------------------------------------- mm 격자 배치 도우미
def fig_mm(fig):
    w, h = fig.get_size_inches(); return w * 25.4, h * 25.4


def ax_mm(fig, x, y_top, w, h, **kw):
    """왼쪽 위 원점(mm) 으로 축을 놓는다(캔버스 크기는 fig 에서 읽는다)."""
    Wf, Hf = fig_mm(fig)
    return fig.add_axes([x / Wf, 1 - (y_top + h) / Hf, w / Wf, h / Hf], **kw)


def ax_h_mm(fig, ax):
    return ax.get_position().height * fig_mm(fig)[1]


def letter(fig, ax, s, x_mm, sty, dy_mm=1.5):
    """패널 문자를 그림 x(mm) 열에 맞춰 축 위쪽에 놓는다(논문은 S.panel_letter, 덱은 14 pt 굵게)."""
    Wf, Hf = fig_mm(fig); b = ax.get_position()
    x = (x_mm / Wf - b.x0) / b.width; y = 1 + (dy_mm / Hf) / b.height
    if sty.deck:
        ax.text(x, y, s, transform=ax.transAxes, fontsize=sty.letter, fontweight="bold", va="bottom", ha="left"); S.register(s, ax)
    else:
        S.panel_letter(ax, s, x=x, y=y)


def text_mm(fig, x, y_top, s, **kw):
    """그림 좌표(mm, 왼쪽 위 원점) 글자."""
    Wf, Hf = fig_mm(fig)
    return fig.text(x / Wf, 1 - y_top / Hf, s, **kw)


def spread(vals, half, gap, lo=None, hi=None):
    """직접 라벨 세로 겹침 제거: 값 순으로 라벨 반높이(half)·간격(gap) 을 지키고, 위쪽 한계(hi) 를 넘으면 전체를 내린다."""
    vals = np.asarray(vals, dtype=float); half = np.asarray(half, dtype=float)
    order = np.argsort(vals); out = vals.copy()
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        out[j] = max(out[j], out[i] + half[i] + half[j] + gap)
    if hi is not None and len(out):
        top = order[-1]; over = out[top] + half[top] - hi
        if over > 0:
            out -= over
    if lo is not None and len(out):
        bot = order[0]; under = lo - (out[bot] - half[bot])
        if under > 0:
            out += under
    return out


def fmt_coord(lat, lon):
    return f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"


def pt_mm(pt):
    return pt / 72.0 * 25.4


# ---------------------------------------------------------------- 자료
def year_changes_oof(r):
    """관측 연도의 연속 쌍(간격 무관) Δ관측·Δ예측. 예측 = 블록 CV OOF(물리 + 잔차 ML, 씨앗 평균). yearwise/multiyear_changes.csv 와 같은 정의."""
    d = pd.read_csv(RUN / "preds" / PRED_FILE.format(r=r), usecols=["cell_year_id", "loc_id", "year", "y", "pred"])
    g = d.groupby("cell_year_id").agg(loc_id=("loc_id", "first"), year=("year", "first"), y=("y", "first"), pred=("pred", "mean")).reset_index()
    g = g.dropna(subset=["y", "pred"]).sort_values(["loc_id", "year"])
    do, dp = [], []
    for _, q in g.groupby("loc_id"):
        if len(q) < 2:
            continue
        do.extend(np.diff(q.y.values)); dp.extend(np.diff(q.pred.values))
    return np.array(do), np.array(dp)


def load():
    cc = pd.read_csv(RUN / "common_cells_eval.csv")
    pe = pd.read_csv(RUN / "products_eval_mask25.csv")
    ym = pd.read_csv(RUN / "yearwise" / "yearwise_metrics.csv")
    st = pd.read_csv(PROC / "site_series" / "site_series_year_obs.csv.gz")
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv", usecols=["loc_id", "region", "lat", "lon", "year", "in_forcing", "n_obs"])
    lab = lab[lab.in_forcing == 1]
    pr = pd.read_csv(PROC / "products_cell_year_v5.csv", usecols=["loc_id", "product", "year", "value_cm"])
    ch = {r: year_changes_oof(r) for r in REG}
    return dict(cc=cc, pe=pe, ym=ym, st=st, lab=lab, pr=pr, ch=ch)


def long_sites(st, lab):
    """지역별 관측 연도 수 최대 셀(동률: 관측 기록 수 최대 → loc_id 최소)."""
    o = st[st.observed == 1]
    ny = o.groupby(["region", "loc_id"]).year.nunique().rename("n_years").reset_index()
    nobs = lab.groupby("loc_id").n_obs.sum().rename("n_obs")
    ny = ny.merge(nobs, left_on="loc_id", right_index=True, how="left").fillna({"n_obs": 0})
    picks = {}
    for r in ("Alaska", "Lena"):
        q = ny[ny.region == r].sort_values(["n_years", "n_obs", "loc_id"], ascending=[False, False, True])
        picks[r] = int(q.loc_id.iloc[0])
    return picks


# ---------------------------------------------------------------- 패널 그리기(축 + 크기 묶음)
def draw_a(fig, ax, cc, sty):
    """a 공통 셀-연도 RMSE. 범위 밖(> CAP) 은 세로 눈금 + 'all / > 25 km' RMSE 값(cm). 반환: 적은 값 목록(검증용)."""
    ax.set_xlim(-0.5, len(ORDER_A) - 0.5); ax.set_ylim(Y0A, CAP + 9)
    per_cm = ax_h_mm(fig, ax) / (CAP + 9 - Y0A)              # mm / cm
    tick = sty.tick_half_mm / per_cm; step = pt_mm(sty.tiny) * 1.65 / per_cm; y_txt0 = CAP + tick + 0.6 / per_cm
    clip_txt, printed = {}, []
    for r in REG:
        t = {m: cc[(cc.region == r) & (cc.mask_km == m)].set_index("series").reindex(ORDER_A) for m in (0, 25)}
        v = {m: t[m].rmse.values for m in (0, 25)}; n = {m: t[m].n.values for m in (0, 25)}
        bad = np.isfinite(n[0]) & np.isfinite(n[25]) & (n[25] > n[0])
        assert not bad.any(), f"{r}: '> 25 km' count exceeds 'all' count for {np.array(ORDER_A)[bad]}"
        x = np.arange(len(ORDER_A)) + OFF[r]
        for m, mfc in ((0, S.REGION_HEX[r]), (25, "white")):
            ok = v[m] <= CAP
            ax.plot(x[ok], v[m][ok], "o", ms=sty.ms, color=S.REGION_HEX[r], mfc=mfc, mew=sty.mew, lw=0, clip_on=False, zorder=4)
        over = np.fmax(v[0], v[25]) > CAP                   # 범위 밖: 채움/빈 원 어휘와 무관한 세로 눈금 + 두 RMSE 값(cm)
        for xi in np.flatnonzero(over):
            ax.plot([x[xi], x[xi]], [CAP - tick, CAP + tick], "-", lw=sty.mew, color=S.REGION_HEX[r], solid_capstyle="butt", clip_on=False, zorder=4)
            k = clip_txt.get(xi, 0); clip_txt[xi] = k + 1
            ax.text(xi, y_txt0 + step * k, f"{v[0][xi]:.0f} / {v[25][xi]:.0f}", ha="center", va="bottom", fontsize=sty.tiny, color=S.REGION_HEX[r])
            printed.append((r, ORDER_A[xi], float(v[0][xi]), float(v[25][xi]), int(n[0][xi]), int(n[25][xi])))
    ax.set_yticks([10, 20, 30, 40]); ax.spines["left"].set_bounds(Y0A, CAP)
    ax.set_xticks(range(len(ORDER_A))); ax.set_xticklabels(SHORT_A, fontsize=sty.tick - 0.5)   # 논문 5.5 pt: "(recalibrated)" 과 이웃 라벨 사이 여백
    ax.set_ylabel("RMSE on common cell-years (cm)")
    hs = [Line2D([], [], marker="o", ms=sty.ms, lw=0, color=S.REGION_HEX[r], label=S.REGION_EN[r]) for r in REG]
    hs += [Line2D([], [], marker="o", ms=sty.ms, lw=0, color=AUX, label="all common cells"),
           Line2D([], [], marker="o", ms=sty.ms, lw=0, color=AUX, mfc="white", mew=sty.mew, label="> 25 km from product training sites"),
           Line2D([], [], marker="|", ms=sty.ms * 1.3, mew=sty.mew, lw=0, color=AUX, label=f"> {CAP:.0f} cm: RMSE printed, all / > 25 km (cm)")]
    leg = ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=sty.tick, handlelength=1.0, handletextpad=0.5,
                    columnspacing=1.2, borderaxespad=0.0, labelspacing=0.35)
    n0 = {r: int(cc[(cc.region == r) & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")].n.iloc[0]) for r in REG}
    n25 = {r: int(cc[(cc.region == r) & (cc.mask_km == 25) & (cc.series == "R1 (CatBoost)")].n.iloc[0]) for r in REG}
    fig.canvas.draw(); rd = fig.canvas.get_renderer()                 # 범례 바로 아래에 셀-연도 수 줄
    lb = leg.get_window_extent(rd); ab = ax.get_window_extent(rd)
    y_n = (lb.y0 - ab.y0) / ab.height - 0.5 * pt_mm(sty.tiny) / ax_h_mm(fig, ax)
    y_n = min(y_n, (CAP - tick - 1.0 / per_cm - Y0A) / (CAP + 9 - Y0A))          # 범위 밖 눈금 글리프 아래 ≥ 1 mm
    ax.text(0.01, y_n, "n cell-years (all / > 25 km):\n" + " · ".join(f"{S.REGION_EN[r]} {n0[r]:,} / {n25[r]:,}" for r in REG),
            transform=ax.transAxes, ha="left", va="top", fontsize=sty.tiny, color=AUX, linespacing=1.1)
    y_sep = y_n - (2 * 1.1 * pt_mm(sty.tiny) + 1.5) / ax_h_mm(fig, ax)          # 우리 방법 | 제품 구분선: 범례·셀-연도 줄 아래에서 멈춘다(글자와 겹치지 않게)
    ax.plot([2.5, 2.5], [Y0A, Y0A + y_sep * (CAP + 9 - Y0A)], color=HAIR, lw=sty.lw_ref, zorder=0)
    return printed


def axes_b(fig, x, y_top, w, h, sty):
    """b 의 끊긴 축: 위 토막(B_HI) 과 아래 토막(B_LO) 두 축. 반환 (ax_hi, ax_lo)."""
    h_hi = h - sty.b_gap_mm - sty.b_lo_mm
    return ax_mm(fig, x, y_top, w, h_hi), ax_mm(fig, x, y_top + h_hi + sty.b_gap_mm, w, sty.b_lo_mm)


def draw_b(fig, ax_hi, ax_lo, pe, sty):
    """b 짝지은 ΔRMSE(마스크 25 km, 연도 정합, 95 % CI). 두 토막에 같은 자료를 그리고 각 축 범위로 자른다."""
    q = pe[(pe.series == "ours:R1:0.5") & pe.definition.str.contains("year-matched")].copy()
    q["prod"] = q.definition.str.extract(r"vs (\w+) \(")[0]
    for r in REG:
        s = q[q.region == r].set_index("prod").reindex(PRODS_B)
        x = np.arange(len(PRODS_B)) + OFF[r]
        d, lo, hi = s.delta_vs_product.values, s.delta_lo.values, s.delta_hi.values
        for ax in (ax_hi, ax_lo):
            ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o", ms=sty.ms, color=S.REGION_HEX[r], mfc="white", mew=sty.mew, lw=sty.lw, capsize=0, clip_on=True, zorder=4)
    ax_hi.axhline(0, color=ZERO, lw=sty.lw_ref, zorder=0)
    for ax in (ax_hi, ax_lo):
        ax.set_xlim(-0.6, len(PRODS_B) - 0.4)
    ax_hi.set_ylim(*B_HI); ax_hi.set_yticks([-30, -20, -10, 0, 10]); ax_hi.spines["left"].set_bounds(*B_HI)
    ax_hi.spines["bottom"].set_visible(False); ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_lo.set_ylim(*B_LO); ax_lo.set_yticks([-120, -80]); ax_lo.spines["left"].set_bounds(*B_LO)
    ax_lo.set_xticks(range(len(PRODS_B))); ax_lo.set_xticklabels(SHORT_B, fontsize=sty.tick)
    col = ax_hi.spines["left"].get_edgecolor(); kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=sty.break_mm / 25.4 * 72, linestyle="none",
                                                          color=col, mec=col, mew=plt.rcParams["axes.linewidth"], clip_on=False, zorder=5)
    ax_hi.plot([0], [0], transform=ax_hi.transAxes, **kw); ax_lo.plot([0], [1], transform=ax_lo.transAxes, **kw)   # 끊김 표시(왼쪽 축)
    ax_hi.set_ylabel("ΔRMSE (cm)\nPhysics + residual ML − product", fontsize=sty.small)
    ax_hi.text(0.98, 0.97, "> 25 km, year-matched, 95 % CI", transform=ax_hi.transAxes, ha="right", va="top", fontsize=sty.tiny, color=AUX)


def draw_c(axes, ch, sty, ylab, xlab):
    """c 연도 간 변화 밀도(지역별 한 축). r·n 은 전체 쌍(창 밖 포함) 에서 계산. 반환 (hexbin 목록, vmax, 통계)."""
    from cmcrameri import cm as cmc
    cmap = LinearSegmentedColormap.from_list("davos_trunc", cmc.davos_r(np.linspace(0.18, 1.0, 256)))
    colls, stats = [], []
    for ax, r, yl, xl in zip(axes, REG, ylab, xlab):
        do, dp = ch[r]
        hb = ax.hexbin(do, dp, gridsize=16, extent=(-LIM, LIM, -LIM, LIM), mincnt=1, cmap=cmap, linewidths=sty.hex_lw, edgecolors="face", rasterized=True)
        colls.append(hb)
        ax.plot([-LIM, LIM], [-LIM, LIM], color=ZERO, lw=sty.lw_ref, zorder=3)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
        ax.set_xticks([-50, -25, 0, 25, 50]); ax.set_yticks([-50, -25, 0, 25, 50])
        rr = float(np.corrcoef(do, dp)[0, 1]); n_out = int(((np.abs(do) > LIM) | (np.abs(dp) > LIM)).sum())
        ax.text(0.04, 0.97, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="top", fontsize=sty.small, color=S.REGION_HEX[r])
        ax.text(0.96, 0.04, f"r = {rr:.2f}, n = {len(do):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=sty.small, color=AUX)
        if not yl:
            ax.set_yticklabels([])
        if not xl:
            ax.set_xticklabels([])
        stats.append((r, rr, int(len(do)), n_out))
    vmax = max(c.get_array().max() for c in colls)
    for c in colls:
        c.set_norm(LogNorm(1, vmax))
    return colls, vmax, stats


def add_cbar(fig, cax, coll, vmax, sty):
    cb = fig.colorbar(coll, cax=cax); k = 2.0 if sty.deck else 1.0
    cb.set_label("Cell-year pairs (count)", fontsize=sty.small, labelpad=1.5 * k); cb.ax.tick_params(labelsize=sty.tick, length=2 * k, which="both", pad=1.5 * k)
    tk = [v for v in (1, 10, 100, 1000) if v <= vmax]; cb.set_ticks(tk); cb.set_ticklabels([str(v) for v in tk]); cb.ax.minorticks_off()
    cb.outline.set_linewidth(sty.cb_lw)
    return cb


def draw_d(ax, ym, sty):
    """d 연도별 편향(블록 CV, n ≥ 50 인 연도)."""
    y = ym[(ym.series == "ours:R1:0.5:blockCV") & (ym.n >= 50)]
    for r in REG:
        s = y[y.region == r].sort_values("year")
        ax.plot(s.year, s.bias, "-o", ms=sty.ms_sq, lw=sty.lw, color=S.REGION_HEX[r], label=S.REGION_EN[r])
    ax.axhline(0, color=ZERO, lw=sty.lw_ref, zorder=0)
    ax.set_xlim(2009.5, 2024.5); ax.set_xticks([2010, 2014, 2018, 2022])
    lo_, hi_ = float(y.bias.min()), float(y.bias.max()); ax.set_ylim(np.floor(lo_ / 10) * 10 - 2, np.ceil(hi_ / 10) * 10 + 6)
    ax.set_xlabel("Observation year"); ax.set_ylabel("Bias, block CV (cm)")
    ax.legend(loc="lower left", fontsize=sty.tick, handlelength=1.2, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.25, bbox_to_anchor=(0.0, 0.0))
    ax.text(0.98, 0.97, "Physics + residual ML, years with n ≥ 50", transform=ax.transAxes, ha="right", va="top", fontsize=sty.tiny, color=AUX)


def draw_site(fig, ax, r, lid, D, sty, legend, site_lines=1):
    """e·f 장기 지점 시계열: 관측(원)·관측일 예측(사각)·계절 말 예측 선·제품 선, 오른쪽 직접 라벨."""
    st, pr, lab = D["st"], D["pr"], D["lab"]
    g = st[(st.region == r) & (st.loc_id == lid) & st.year.between(*YR)].sort_values("year")
    o = g[g.observed == 1]
    lines = [("p1_eos", "Stefan\n(recalibrated)", S.METHOD_HEX["P1"], sty.lw_p1, S.METHOD_LS["P1"]),
             ("r1_eos", "Physics +\nresidual ML", S.METHOD_HEX["R1"], sty.lw_main, S.METHOD_LS["R1"])]
    ends, missing, drawn = [], [], []                        # (라벨, 색, 끝 y, 끝 연도); drawn = 그린 모든 값(y 범위용)
    for col, name, colr, lw, ls in lines:
        ax.plot(g.year, g[col], ls=ls, lw=lw, color=colr, zorder=3); ends.append((name, colr, float(g[col].iloc[-1]), int(g.year.iloc[-1])))
    for p, name in (("wei", "Wei 2026"), ("liu", "Liu 2024")):
        s = pr[(pr.loc_id == lid) & (pr["product"] == p) & pr.year.between(*YR)].dropna(subset=["value_cm"]).sort_values("year")
        if len(s):
            ax.plot(s.year, s.value_cm, ls=PLS[p], lw=sty.lw_prod, color=PHEX[p], zorder=2); drawn.append(s.value_cm.values)
            ends.append((name, PHEX[p], float(s.value_cm.iloc[-1]), int(s.year.iloc[-1])))
        else:
            missing.append(name)
    ax.plot(o.year, o.y, "o", ms=sty.ms_obs, color="black", mfc="white", mew=sty.mew, lw=0, zorder=5, label="Observed")
    ax.plot(o.year, o.r1_obs, "s", ms=sty.ms_sq, color=S.METHOD_HEX["R1"], lw=0, zorder=6, label="At observation day")
    ax.set_xlim(YR[0] - 0.5, YR[1] + 0.5); ax.set_xticks([2010, 2014, 2018, 2022])
    ax.set_xlabel("Year"); ax.set_ylabel("ALT (cm)")
    y_all = np.concatenate([o.y.values, o.r1_obs.values, g.p1_eos.values, g.r1_eos.values, np.array([e[2] for e in ends], dtype=float)] + drawn)
    lo_, hi_ = np.nanmin(y_all), np.nanmax(y_all); pad = 0.10 * (hi_ - lo_)
    ax.set_ylim(lo_ - pad, hi_ + pad * 4.2)                    # 위쪽 띠: 지점 라벨·범례
    EH = ax_h_mm(fig, ax); line_h_mm = pt_mm(sty.small) * 1.2 * 0.95
    yl = ax.get_ylim(); span = yl[1] - yl[0]; per_mm = span / EH
    names = [n if yr >= YR[1] else f"{n}\n(to {yr})" for n, _, _, yr in ends]
    half = [0.5 * n.count("\n") * line_h_mm * per_mm + 0.5 * line_h_mm * per_mm for n in names]
    ys = spread([e[2] for e in ends], half, 0.8 * per_mm, lo=yl[0], hi=yl[1])
    for (nm, colr, _, _), n2, yy in zip(ends, names, ys):
        ax.text(YR[1] + 0.8, yy, n2, color=colr, fontsize=sty.small, va="center", ha="left", clip_on=False, linespacing=0.95)
    if missing:
        ax.text(YR[1] + 0.8, yl[0] + 0.02 * span, "no value:\n" + ",\n".join(missing), color=AUX, fontsize=sty.tiny, va="bottom", ha="left", clip_on=False, linespacing=0.95)
    la = float(lab[lab.loc_id == lid].lat.iloc[0]); lo = float(lab[lab.loc_id == lid].lon.iloc[0])
    site = f"{S.REGION_EN[r]}, {fmt_coord(la, lo)}, {len(o)} observed years" if site_lines == 1 else f"{S.REGION_EN[r]}, {fmt_coord(la, lo)}\n{len(o)} observed years"
    ax.text(0.02, 0.97, site, transform=ax.transAxes, ha="left", va="top", fontsize=sty.small, color=AUX, linespacing=1.1)
    if legend:
        y_leg = 0.97 - (site_lines * pt_mm(sty.small) * 1.25 + 1.2) / EH
        ax.legend(loc="upper left", fontsize=sty.tick, handlelength=1.0, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.3, ncol=2, columnspacing=1.0, bbox_to_anchor=(0.02, y_leg))


# ---------------------------------------------------------------- 배치 점검(패널별 꽉 찬 상자, mm)
def panel_boxes(fig):
    """S._crop_panels 와 같은 요소(축 틀·축 라벨·범례·축 안 글자) 로 패널 문자별 상자를 mm 로 돌려준다."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); k = 25.4 / fig.dpi; out = {}
    for let, axes in S.PANELS.items():
        bbs = []
        for a in axes:
            bbs.append(a.get_window_extent(r))
            for axis in (a.xaxis, a.yaxis):
                b = axis.get_tightbbox(r)
                if b is not None:
                    bbs.append(b)
            leg = a.get_legend()
            if leg is not None:
                bbs.append(leg.get_window_extent(r))
            for t in a.texts:
                bbs.append(t.get_window_extent(r))
        bbs = [b for b in bbs if b is not None and b.width > 0]
        out[let] = (min(b.x0 for b in bbs) * k, max(b.x1 for b in bbs) * k, (fig.bbox.height - max(b.y1 for b in bbs)) * k,
                    (fig.bbox.height - min(b.y0 for b in bbs)) * k)
    return out


def gap_report(fig):
    bx = panel_boxes(fig); Wf, Hf = fig_mm(fig)
    for let in ("a", "b", "c", "d", "e", "f"):
        x0, x1, y0, y1 = bx[let]; print(f"  panel {let}: x {x0:6.1f}–{x1:6.1f} mm  y {y0:6.1f}–{y1:6.1f} mm")
    m = 0.03 * 25.4 * 2                                     # 크롭 여백(양쪽)
    for l, rgt in (("a", "b"), ("c", "d"), ("e", "f")):
        g = bx[rgt][0] - bx[l][1]; print(f"  gap {l}|{rgt}: {g:5.1f} mm  crop {'OK' if g > m else 'OVERLAP'} (need > {m:.1f})")
    for up, dn in (("a", "c"), ("b", "d"), ("c", "e"), ("d", "f")):
        g = bx[dn][2] - bx[up][3]; print(f"  gap {up}/{dn}: {g:5.1f} mm  {'OK' if g > m else 'OVERLAP'}")
    print(f"  figure right edge used: {max(b[1] for b in bx.values()):.1f} mm of {Wf:.0f}; bottom {max(b[3] for b in bx.values()):.1f} of {Hf:.0f}")


def print_checks(printed, stats):
    print("  a (> 45 cm): region, series, RMSE all / > 25 km (cm), n all / > 25 km")
    for r, s, v0, v25, n0, n25 in printed:
        print(f"    {r:7s} {s:12s} {v0:6.1f} / {v25:6.1f} cm   n {n0:,} / {n25:,}")
    print("  c (OOF, successive observed years): region, r, n pairs, pairs outside ±50 cm window (not drawn, counted)")
    for r, rr, n, n_out in stats:
        print(f"    {r:7s} r = {rr:.2f}  n = {n:,}  outside window {n_out}")


# ---------------------------------------------------------------- 논문 그림
def f5(report=False):
    D = load(); sty = PAPER
    fig = S.figure(W, H)
    # a
    ax = ax_mm(fig, XL, 4, 97.0 - XL, 36.5); printed = draw_a(fig, ax, D["cc"], sty); letter(fig, ax, "a", LX, sty)   # 36.5 mm: 3줄 눈금 라벨 아래 c 와 > 1.5 mm
    # b(끊긴 축)
    ax_hi, ax_lo = axes_b(fig, XB, 4, XE - XB, 38, sty); draw_b(fig, ax_hi, ax_lo, D["pe"], sty); letter(fig, ax_hi, "b", LXR, sty); S.register("b", ax_lo)
    # c
    SQ, GAP = 22.0, 3.5; cx = [XL + j * (SQ + GAP) for j in range(3)]; cy = 56.0
    axes_c = [ax_mm(fig, cx[j], cy, SQ, SQ) for j in range(3)]
    colls, vmax, stats = draw_c(axes_c, D["ch"], sty, ylab=[True, False, False], xlab=[True, True, True])
    axes_c[0].set_ylabel("Predicted Δ ALT (cm)\nPhysics + residual ML"); axes_c[1].set_xlabel("Observed Δ ALT between successive observed years (cm)")
    cax = ax_mm(fig, cx[2] + SQ + 2.0, cy, 2.0, SQ); add_cbar(fig, cax, colls[0], vmax, sty)
    letter(fig, axes_c[0], "c", LX, sty); S.register("c", axes_c[1], axes_c[2], cax)
    # d
    ax = ax_mm(fig, XB, cy, XE - XB, SQ); draw_d(ax, D["ym"], sty); letter(fig, ax, "d", LXR, sty)
    # e, f
    picks = long_sites(D["st"], D["lab"]); ey, EW, EH = 93.0, XE - XB, 37.0; ex = {"Alaska": XL, "Lena": XB}
    for k, r in enumerate(("Alaska", "Lena")):
        ax = ax_mm(fig, ex[r], ey, EW, EH); draw_site(fig, ax, r, picks[r], D, sty, legend=(k == 0)); letter(fig, ax, "ef"[k], LX if k == 0 else LXR, sty)
    S.register("ab", *S.PANELS["a"], *S.PANELS["b"])
    S.register("cd", *S.PANELS["c"], *S.PANELS["d"])
    S.register("ef", *S.PANELS["e"], *S.PANELS["f"])
    print_checks(printed, stats)
    if report:
        gap_report(fig)
    return S.export(fig, STEM, outdir=OUT)


# ---------------------------------------------------------------- 덱 판(덱 본문 12.0 × 4.65 in, 300 dpi)
SRC = OUT / "source_data"; YW_CSV = SRC / "F5_yearwise_rmse.csv"
ORDER_R04 = [("P1 (Stefan, recal.)", "Stefan (recalibrated)", S.METHOD_HEX["P1"]), ("R1 (CatBoost)", "Physics + residual ML", S.METHOD_HEX["R1"]),
             ("D0 (CatBoost)", "Direct ML", S.METHOD_HEX["D0"]), ("Wei 2026", "Wei 2026", S.PRODUCT_HEX["wei"]), ("Liu 2024", "Liu 2024", S.PRODUCT_HEX["liu"]),
             ("ESA CCI v4", "ESA CCI v4", S.PRODUCT_HEX["cci4y"]), ("ESA CCI v5", "ESA CCI v5", S.PRODUCT_HEX["cci5y"]), ("Aalto 2018", "Aalto 2018", S.PRODUCT_HEX["aalto"])]
N_OURS = 3                                # R04: 우리 방법 행 수(그 아래 옅은 파선)
X4 = (10.0, 45.0)                         # R04: x 범위(cm). 넘는 값은 축 끝에 마커 + 숫자
MIN_N_YEAR = 30                           # R05: 해마다 최소 공통 셀-연도 수
SER_R05 = [("r1", "Physics + residual ML", S.METHOD_HEX["R1"], "o", "-"), ("p1", "Stefan (recalibrated)", S.METHOD_HEX["P1"], "s", "-"),
           ("wei", "Wei 2026", PHEX["wei"], "^", PLS["wei"]), ("liu", "Liu 2024", PHEX["liu"], "D", PLS["liu"])]
LIU_LAST = 2020
MS_R05 = {"r1": 8.6, "p1": 5.6, "wei": 6.4, "liu": 5.6}   # R05 마커 지름(pt): 주 방법 원을 크게


def deck_canvas():
    """덱 캔버스: S.figure 로 만들고 크기를 12.0 × 4.65 in 으로 정확히 맞춘다(mm 왕복은 3600 px 가 3599 로 내림된다)."""
    fig = S.figure(DW, DH); fig.set_size_inches(12.0 + 1e-9, DH / 25.4 + 1e-9)
    return fig


def setup_deck():
    S.setup(scale=1.69)
    Ls = S.TOK["lines"]["slide"]; Fs = S.F["slide"]
    plt.rcParams.update({"font.size": Fs["direct_label_pt"], "axes.labelsize": Fs["axis_label_pt"], "xtick.labelsize": Fs["tick_pt"], "ytick.labelsize": Fs["tick_pt"],
                         "legend.fontsize": Fs["tick_pt"], "axes.linewidth": Ls["axis_pt"], "xtick.major.width": Ls["tick_pt"], "ytick.major.width": Ls["tick_pt"],
                         "xtick.major.size": Ls["tick_len_pt"], "ytick.major.size": Ls["tick_len_pt"], "lines.linewidth": Ls["data_pt"], "lines.markersize": Ls["marker_pt"],
                         "xtick.major.pad": 4.0, "ytick.major.pad": 4.0, "axes.labelpad": 6.0})


def save_deck(fig, slide):
    """캔버스 그대로 저장(bbox tight 없음) → 3600 × 1395 px, RGB 흰 배경. 잉크 상자의 캔버스 채움 비율을 적는다."""
    from PIL import Image
    DECK.mkdir(parents=True, exist_ok=True); p = DECK / f"{STEM}__deck_{slide}.png"
    fig.canvas.draw(); tb = fig.get_tightbbox(fig.canvas.get_renderer()); Wi, Hi = fig.get_size_inches()
    print(f"  {slide}: ink x {tb.x0 / Wi * 100:5.1f}–{tb.x1 / Wi * 100:5.1f} %, y {tb.y0 / Hi * 100:5.1f}–{tb.y1 / Hi * 100:5.1f} %  →  fill {tb.width / Wi * 100:.0f} % × {tb.height / Hi * 100:.0f} %"
          + ("  OVERFLOW" if tb.x0 < -1e-3 or tb.y0 < -1e-3 or tb.x1 > Wi + 1e-3 or tb.y1 > Hi + 1e-3 else ""))
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    rgb.save(p, dpi=(300, 300))
    want = (int(round(DW / 25.4 * 300)), int(round(DH / 25.4 * 300)))
    print(f"  {p.name}: {rgb.size[0]} × {rgb.size[1]} px {rgb.mode}  {'OK' if rgb.size == want else f'MISMATCH want {want}'}")
    assert rgb.size == want
    return p


def ax_in(fig, x, y, w, h):
    """inch 좌표(왼쪽 아래 원점) 축."""
    fw, fh = fig.get_size_inches(); return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def text_w_in(fig, txt, fs):
    t = fig.text(0, 0, txt, fontsize=fs); fig.canvas.draw(); w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi; t.remove()
    return w


def top_legend(fig, handles, labels, x_in, fs, ncol=None, y_in=0.04, colsp=1.6):
    fw, fh = fig.get_size_inches()
    return fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x_in / fw, 1 - y_in / fh), ncol=ncol or len(handles), frameon=False, fontsize=fs,
                      handlelength=1.8, handletextpad=0.5, columnspacing=colsp, borderaxespad=0.0, borderpad=0.0, labelspacing=0.45)


def deck_r04(D):
    """R04: 지역별 패널의 행별 점 그림(같은 셀 · 같은 해 RMSE). 채운 원 = 공통 셀-연도 전체(행 위쪽), 빈 원 = 25 km 밖(행 아래쪽).
    행 = 우리 방법 셋 → 옅은 파선 → 제품 다섯. X4 를 넘는 값은 그 행의 축 끝에 마커와 숫자(cm)로 적는다."""
    cc = D["cc"]; fig = deck_canvas(); sty = DECKS; fw, fh = fig.get_size_inches()
    lab_w = max(text_w_in(fig, lab, sty.tick) for _, lab, _ in ORDER_R04)
    x0 = 0.10 + lab_w + 0.14; gap, xr, yb, top = 0.88, 0.42, 0.80, 0.78
    w = (fw - x0 - xr - 2 * gap) / len(REG); h = fh - yb - top
    MS4, MEW4, DY = 8.5, 1.7, 0.24; nr = len(ORDER_R04); printed = []
    for j, r in enumerate(REG):
        ax = ax_in(fig, x0 + j * (w + gap), yb, w, h)
        ax.set_xlim(*X4); ax.set_ylim(nr - 0.45, -0.55)
        for i in range(nr):
            ax.axhline(i, color=S.NEUTRAL["equiv_band"], lw=1.0, zorder=0)
        ax.axhline(N_OURS - 0.5, color=S.NEUTRAL["coast"], lw=1.2, ls=(0, (4, 3)), zorder=0)
        for i, (ser, lab, col) in enumerate(ORDER_R04):
            v = {m: cc[(cc.region == r) & (cc.mask_km == m) & (cc.series == ser)].rmse for m in (0, 25)}
            v = {m: (float(q.iloc[0]) if len(q) else np.nan) for m, q in v.items()}
            for m, dy, mfc in ((0, -DY, col), (25, DY, "white")):    # 채운 원 = 전체(행 위쪽), 빈 원 = 25 km 밖(행 아래쪽)
                if not np.isfinite(v[m]):
                    continue
                if v[m] <= X4[1]:
                    ax.plot([v[m]], [i + dy], "o", ms=MS4, mfc=mfc, mec=col, mew=MEW4, lw=0, zorder=3, clip_on=False)
                else:                                   # 범위 밖: 축 끝 위 마커 + 축 밖 숫자(cm)
                    ax.plot([X4[1]], [i + dy], "o", ms=MS4, mfc=mfc, mec=col, mew=MEW4, lw=0, zorder=3, clip_on=False)
                    ax.annotate(f"{v[m]:.0f}", (X4[1], i + dy), xytext=(MS4 / 2 + 3, 0), textcoords="offset points", ha="left", va="center",
                                fontsize=sty.tick, color=col, annotation_clip=False)
                    printed.append((r, lab, m, v[m]))
        ax.set_yticks(range(nr)); ax.set_yticklabels([lab for _, lab, _ in ORDER_R04] if j == 0 else [], fontsize=sty.tick)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(j == 0)
        ax.set_xticks([10, 20, 30, 40])
        ax.text(0.0, 1.0 + 0.08 / h, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="bottom", fontsize=sty.text, color=S.REGION_HEX[r])
        if j == 1:
            ax.set_xlabel("RMSE, same cell and year (cm)", fontsize=sty.text)
    hs = [Line2D([], [], marker="o", ms=MS4, mfc=AUX, mec=AUX, mew=MEW4, lw=0), Line2D([], [], marker="o", ms=MS4, mfc="white", mec=AUX, mew=MEW4, lw=0)]
    top_legend(fig, hs, ["All common cell-years", "> 25 km from product training sites"], x0 - 0.08, sty.small)
    return save_deck(fig, "R04"), printed


def yearwise_rmse():
    """R05 표: 지역 × 연도 × 마스크(0 · 25 km)의 같은 셀-연도 RMSE. 셀-연도 집합은 docstring(덱 판 R05) 규칙. 표를 저장하고 돌려준다."""
    import polar.testbed_v5 as T
    base = T.load_v5(PROC).set_index("cell_year_id")
    lk = pd.read_csv(PROC.parent / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv")
    lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min(); base["dist_km"] = base.loc_id.map(lk)
    rows = []
    for r in REG:
        b = base[base.region == r].copy()
        for key, f in (("r1", PRED_FILE.format(r=r)), ("p1", f"{r}__block__year_obs__F8TS__P1__-__lam-1.csv.gz")):
            q = pd.read_csv(RUN / "preds" / f, usecols=["cell_year_id", "pred"]).groupby("cell_year_id").pred.mean(); b[key] = b.index.map(q)
        for mask in (0, 25):
            q = b if mask == 0 else b[~(b.dist_km <= mask)]
            for y, g in q.groupby("year"):
                s3 = g[np.isfinite(g[["alt_cm", "r1", "p1", "prod_wei"]]).all(axis=1)]
                s4 = s3[np.isfinite(s3.prod_liu)] if y <= LIU_LAST else s3.iloc[0:0]
                use, with_liu = (s4, True) if len(s4) >= MIN_N_YEAR else (s3, False)
                row = dict(region=r, mask_km=mask, year=int(y), n=len(use), n_blocks=int(use.block.nunique()), n_without_liu=len(s3), n_with_liu=len(s4),
                           liu_included=with_liu, shown=len(use) >= MIN_N_YEAR)
                for key, col in (("r1", "r1"), ("p1", "p1"), ("wei", "prod_wei"), ("liu", "prod_liu")):
                    ok = len(use) > 0 and (key != "liu" or with_liu)
                    row[f"rmse_{key}"] = T.rmse(use.alt_cm.values, use[col].values) if ok else np.nan
                    row[f"bias_{key}"] = float((use[col] - use.alt_cm).mean()) if ok else np.nan
                rows.append(row)
    out = pd.DataFrame(rows); SRC.mkdir(parents=True, exist_ok=True); out.to_csv(YW_CSV, index=False, float_format="%.3f")
    (YW_CSV.with_suffix(".json")).write_text(__import__("json").dumps(dict(
        script="scripts/4_visualization/v7_f5_products_years.py (yearwise_rmse)", csv=str(YW_CSV.relative_to(ROOT)),
        predictions="core_v6 preds, block CV out-of-fold, year_obs, F8TS; R1 catboost lam 0.5 (3 seeds averaged), P1 recalibrated Stefan",
        products="products_cell_year_v5.csv via polar.testbed_v5.load_v5: value of the observation year (prod_wei, prod_liu)",
        cell_years=f"per year: labelled cell-years where R1, P1 and Wei 2026 all have values; Liu 2024 (to {LIU_LAST}) is included when it leaves >= {MIN_N_YEAR} "
                   f"cell-years, otherwise the year is shown without Liu; years with < {MIN_N_YEAR} cell-years are not shown",
        mask_km="0 = all cell-years; 25 = cells more than 25 km from Wei 2026 training sites (xg_leak_cells_v1.csv, as v5_common_cells.py)"), indent=1))
    return out


def yearwise_summary(t):
    for mask in (0, 25):
        print(f"  R05 mask {mask} km: years shown and Physics + residual ML lower RMSE than ...")
        for r in REG:
            q = t[(t.region == r) & (t.mask_km == mask) & t.shown]
            liu = q[q.liu_included]
            print(f"    {r:7s} years {len(q):2d} ({q.year.min()}–{q.year.max()}): < Wei {int((q.rmse_r1 < q.rmse_wei).sum())}/{len(q)}, "
                  f"< Liu {int((liu.rmse_r1 < liu.rmse_liu).sum())}/{len(liu)}, < Stefan {int((q.rmse_r1 < q.rmse_p1).sum())}/{len(q)}")


def deck_r05(D):
    """R05: 지역별 패널, 연도마다 같은 셀-연도에서 네 계열 RMSE(공통 y). 아래 띠 = 셀-연도 수(로그 막대)."""
    t = yearwise_rmse(); yearwise_summary(t); q0 = t[(t.mask_km == 0) & t.shown]
    fig = deck_canvas(); sty = DECKS; fw, fh = fig.get_size_inches()
    x0, gap, xr, yb, top, hs_, gs = 1.00, 0.72, 0.12, 0.66, 0.78, 0.60, 0.10
    w = (fw - x0 - xr - 2 * gap) / len(REG); h = fh - yb - top - hs_ - gs
    NL = (20, 1.0e4)                                    # 띠: 셀-연도 수 로그 축(눈금 100 · 1,000)
    for j, r in enumerate(REG):
        ax = ax_in(fig, x0 + j * (w + gap), yb + hs_ + gs, w, h); axn = ax_in(fig, x0 + j * (w + gap), yb, w, hs_)
        q = q0[q0.region == r].sort_values("year")
        for k, lab, col, mk, ls in SER_R05[::-1]:     # Physics + residual ML 을 맨 위에
            z = q[np.isfinite(q[f"rmse_{k}"])]
            yrs, v = z.year.values, z[f"rmse_{k}"].values
            seg = np.split(np.arange(len(yrs)), np.where(np.diff(yrs) > 1)[0] + 1) if len(yrs) else []
            main = k == "r1"                          # 겹침: 큰 주황 원을 아래, 작은 마커(Stefan · 제품)를 위에 그려 같은 값도 둘 다 보이게
            for sgi in seg:
                if len(sgi) > 1:
                    ax.plot(yrs[sgi], v[sgi], ls=ls, color=col, lw=sty.lw_main if main else sty.lw_prod, zorder=3.5 if main else 3)
            ax.plot(yrs, v, mk, ms=MS_R05[k], color=col, mec="white" if main else col, mew=0.8 if main else 0, lw=0, zorder=4 if main else 5)
        vmax = np.nanmax(q[[f"rmse_{k}" for k, *_ in SER_R05]].values)
        ax.set_xlim(YR[0] - 0.6, YR[1] + 0.6); ax.set_ylim(0, np.ceil(1.06 * vmax / 10) * 10); ax.set_xticks([2010, 2014, 2018, 2022])
        ax.tick_params(axis="x", labelbottom=False); ax.yaxis.set_major_locator(plt.MultipleLocator(20 if vmax > 60 else 10))
        ax.text(0.0, 1.0 + 0.08 / h, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="bottom", fontsize=sty.text, color=S.REGION_HEX[r])
        axn.bar(q.year, q.n, width=0.62, color=S.NEUTRAL["no_data"], lw=0, zorder=2)
        axn.set_yscale("log"); axn.set_ylim(*NL); axn.set_xlim(*ax.get_xlim()); axn.set_xticks([2010, 2014, 2018, 2022])
        axn.set_yticks([100, 1000]); axn.yaxis.set_minor_locator(plt.NullLocator())
        axn.set_yticklabels(["100", "1,000"] if j == 0 else [])   # 띠는 세 패널이 같은 로그 축
        if j == 0:
            ax.set_ylabel("RMSE, same cell-years (cm)", fontsize=sty.text)
            axn.set_ylabel("n", fontsize=sty.text, rotation=0, ha="right", va="center", labelpad=8)
        if j == 1:
            axn.set_xlabel("Observation year", fontsize=sty.text)
    hs = [Line2D([], [], color=col, ls=ls, marker=mk, ms=MS_R05[k], mec="white" if k == "r1" else col, mew=0.8 if k == "r1" else 0,
                 lw=sty.lw_main if k == "r1" else sty.lw_prod) for k, lab, col, mk, ls in SER_R05]
    top_legend(fig, hs, [lab for _, lab, *_ in SER_R05], x0 - 0.08, sty.small)
    return save_deck(fig, "R05"), t


def deck_r06(D):
    """R06: 두 지점 패널 + 위쪽 범례 두 줄(처음 보는 사람도 읽게 기호마다 이름). 패널 머리 = 지역 · 좌표, 제품 값이 없으면 오른쪽에 'No product value'."""
    fig = deck_canvas(); sty = DECKS; fw, fh = fig.get_size_inches()
    st, pr, lab = D["st"], D["pr"], D["lab"]; picks = long_sites(st, lab)
    x0, gap, xr, yb, top = 0.95, 0.95, 0.15, 0.80, 1.05
    w = (fw - x0 - xr - gap) / 2; h = fh - yb - top
    for k, r in enumerate(("Alaska", "Lena")):
        ax = ax_in(fig, x0 + k * (w + gap), yb, w, h); lid = picks[r]
        g = st[(st.region == r) & (st.loc_id == lid) & st.year.between(*YR)].sort_values("year"); o = g[g.observed == 1]
        ax.plot(g.year, g.p1_eos, ls=S.METHOD_LS["P1"], lw=sty.lw_p1, color=S.METHOD_HEX["P1"], zorder=3)
        ax.plot(g.year, g.r1_eos, ls=S.METHOD_LS["R1"], lw=sty.lw_main, color=S.METHOD_HEX["R1"], zorder=4)
        drawn = [o.y.values, o.r1_obs.values, g.p1_eos.values, g.r1_eos.values]; have = []
        for p in ("wei", "liu"):
            sp = pr[(pr.loc_id == lid) & (pr["product"] == p) & pr.year.between(*YR)].dropna(subset=["value_cm"]).sort_values("year")
            if len(sp):
                ax.plot(sp.year, sp.value_cm, ls=PLS[p], lw=sty.lw_prod, color=PHEX[p], zorder=2); drawn.append(sp.value_cm.values); have.append(p)
        ax.plot(o.year, o.y, "o", ms=sty.ms_obs + 1.0, color="black", mfc="white", mew=sty.mew, lw=0, zorder=6)
        ax.plot(o.year, o.r1_obs, "s", ms=sty.ms_sq + 0.8, color=S.METHOD_HEX["R1"], lw=0, zorder=5)
        ya = np.concatenate(drawn); lo_, hi_ = np.nanmin(ya), np.nanmax(ya); pad = 0.08 * (hi_ - lo_)
        ax.set_ylim(np.floor((lo_ - pad) / 5) * 5, np.ceil((hi_ + pad) / 5) * 5)
        ax.set_xlim(YR[0] - 0.5, YR[1] + 0.5); ax.set_xticks([2010, 2014, 2018, 2022]); ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.set_xlabel("Year", fontsize=sty.text)
        if k == 0:
            ax.set_ylabel("ALT (cm)", fontsize=sty.text)
        la = float(lab[lab.loc_id == lid].lat.iloc[0]); lo = float(lab[lab.loc_id == lid].lon.iloc[0])
        ax.text(0.0, 1.0 + 0.08 / h, f"{S.REGION_EN[r]} · {fmt_coord(la, lo)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=sty.text, color=S.REGION_HEX[r])
        if not have:
            ax.text(1.0, 1.0 + 0.08 / h, "No product value", transform=ax.transAxes, ha="right", va="bottom", fontsize=sty.tick, color=AUX)
        print(f"  R06 {r}: loc_id {lid}, observed years {len(o)}, products drawn {have or 'none'}")
    hd = {"obs": Line2D([], [], marker="o", ms=sty.ms_obs + 1.0, color="black", mfc="white", mew=sty.mew, lw=0),
          "r1o": Line2D([], [], marker="s", ms=sty.ms_sq + 0.8, color=S.METHOD_HEX["R1"], lw=0),
          "r1e": Line2D([], [], color=S.METHOD_HEX["R1"], lw=sty.lw_main, ls=S.METHOD_LS["R1"]),
          "p1e": Line2D([], [], color=S.METHOD_HEX["P1"], lw=sty.lw_p1, ls=S.METHOD_LS["P1"]),
          "wei": Line2D([], [], color=PHEX["wei"], lw=sty.lw_prod, ls=PLS["wei"]), "liu": Line2D([], [], color=PHEX["liu"], lw=sty.lw_prod, ls=PLS["liu"])}
    lb = {"obs": "Observed", "r1o": "Physics + residual ML, observation day", "r1e": "Physics + residual ML, end of season",
          "p1e": "Stefan (recalibrated), end of season", "wei": "Wei 2026", "liu": "Liu 2024"}
    order = ["obs", "p1e", "r1o", "wei", "r1e", "liu"]          # 열 우선 채움 → 줄 1: 관측 · 관측일 · 계절 말, 줄 2: Stefan · Wei · Liu
    top_legend(fig, [hd[q] for q in order], [lb[q] for q in order], x0 - 0.08, sty.small, ncol=3, colsp=2.0)
    return save_deck(fig, "R06")


def deck():
    setup_deck(); D = load()
    p4, printed = deck_r04(D); p5, _ = deck_r05(D); p6 = deck_r06(D)
    print("  R04 values beyond the axis (region, series, mask km, RMSE cm):", "; ".join(f"{r} {s_} {m} {v:.1f}" for r, s_, m, v in printed))
    return [p4, p5, p6]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, default=5); ap.add_argument("--slide", action="store_true")
    ap.add_argument("--deck", action="store_true", help="덱 판 3장(R04–R06, 12.0 × 4.65 in) 을 outputs/figures/paper/v7/deck/ 에 낸다")
    ap.add_argument("--report", action="store_true", help="패널 상자·간격(mm) 보고")
    a = ap.parse_args()
    if a.deck:
        for p in deck():
            print(p)
        sys.exit(0)
    S.setup(scale=1.69 if a.slide else 1.0)
    for p in f5(report=a.report):
        print(p)
    if a.slide:
        from PIL import Image
        sd = OUT / "slide"; wd = {k: Image.open(sd / f"{STEM}__{k}.png").size[0] for k in ("a", "b", "ab", "c", "d", "cd", "e", "f", "ef")}
        for l, r_, lr in (("a", "b", "ab"), ("c", "d", "cd"), ("e", "f", "ef")):
            print(f"crop widths {l}+{r_} = {wd[l] + wd[r_]} vs {lr} = {wd[lr]} -> {'OK' if wd[l] + wd[r_] <= wd[lr] else 'OVERLAP'}")
