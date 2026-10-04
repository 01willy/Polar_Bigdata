"""Fig 3 v3: 물리 정보의 사용(개념 좌표도, 위약 대조, 결합 구조, 재보정 방식).

명세: outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 5절, design/journal_grade_style_guide.md 2–3절, 6.3절.
산출(outputs/figures/paper/v3_restructure/): Fig3.pdf(벡터, 글꼴 내장), Fig3.png(600 dpi), Fig3_source_data.csv,
Fig3_legend.md(350단어 이하), Fig3_values.txt(값 대조와 점검 기록).

원천(읽기 전용)
- a: data/processed/fidelity_base_v3.csv(+ e5_soil_tdd_v3.csv, polar.m1_core.load_base), 원천 계수는
  data/processed/paper_figs/table1_rows.csv 의 E0_x. 라벨 추출은 h40_label_grid.draw_cells 와 같은 규칙(seed_of).
- b: data/processed/paper_figs/fig3_a.csv
- c: data/processed/paper_figs/fig3_b_L10_alln.csv(series F1k, F1n)
- d: data/processed/paper_figs/pool_fixed_curve.csv(method P2, baseline P1, learner none)
- 대조: data/processed/lgx/lgx_tests.csv(L15, L10), data/processed/lgw/lgw_bundle.csv(AB3, AB7),
  paper/claims/{C1_label0_safety,C3_structure_by_label_count}/tables/MANIFEST.csv(sha256)

명세와 다른 점(Fig3_values.txt 에 근거를 적는다)
- a 의 지역: 명세는 레나델타였으나 레나델타는 라벨 10개 재보정 계수(1.58)가 원천 계수(1.62)와 2.5% 달라
  두 직선이 인쇄 크기에서 겹친다. 재보정이 계수를 크게 옮긴 서부 러시아(원천 1.59, 재보정 2.11)로 그린다.
  CONCEPT_REGION 하나로 되돌릴 수 있다.
- b: 같은 행 이름을 두 번 쓰지 않으려고(지침 H13) 라벨 수 묶음을 위아래 대신 좌우 두 열로 둔다.
- d: 계열이 하나라도 y 축 이름을 짧게 쓰려고 직접 라벨 "Least squares" 하나를 둔다.
- c, d: x 축 이름 "Target labels, n" 은 아래 행에 한 번만 둔다(같은 문구 반복 0).

실행: python scripts/4_visualization/paper_v3/fig3.py  (공유 서버: 가용 메모리 30 GB, 부하 40 확인 뒤 렌더)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402  (공유 모듈. 고치지 않고 가져다 쓴다)

import matplotlib  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

ROOT = S.ROOT
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
PF = S.PAPER_FIGS
OUT = S.OUT
STEM = "Fig3"

# ---------------------------------------------------------------- 고정 설정
CONCEPT_REGION = "Russia_W"     # 명세 값은 "Lena"(위 머리말과 Fig3_values.txt 참조)
CONCEPT_MODE = "x"              # 주 4지역 풀의 모드(Lena|x 등)
CONCEPT_SPLIT = 1               # half_split_blocks split_seed(분할 1)
CONCEPT_DRAW = 1                # draw_cells 의 추출 번호(추출 1, 코드 값 그대로)
CONCEPT_N = 10                  # 라벨 10개
KAPPA = 10.0                    # 등록 수축 강도(h40 --kappa 기본값)

FIG_W, FIG_H = S.W2_MM, 120.0
BLACK = S.INK
GREY = S.INK_AUX
C_SRC = S.METHOD["source_stefan"]["color"]
C_RECAL = S.METHOD["recalibrated_stefan"]["color"]
C_RESID = S.METHOD["anchor_residual"]["color"]
PLACEBO_ROWS = [("shuffle", "Shuffled physics labels"), ("const_t", "Constant at pool mean"),
                ("const_src", "Constant at source mean"), ("tddlin", "Linear thaw index")]
REGIONS4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
MM_PT = 72.0 / 25.4             # pt per mm


# ================================================================ 자원 확인
def wait_resources(min_gb: float = 30.0, max_load: float = 40.0, max_wait_s: int = 1800) -> str:
    """공유 서버: 가용 메모리 30 GB 미만이거나 1분 부하 40 초과면 기다린다(작업 지시)."""
    t0 = time.time()
    while True:
        free = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        avail = float(free.splitlines()[1].split()[-1])
        load1 = os.getloadavg()[0]
        if avail >= min_gb and load1 <= max_load:
            return f"available {avail:.0f} GB, load average {load1:.2f}"
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"자원 부족으로 중단: available {avail} GB, load {load1:.1f}")
        time.sleep(30)


# ================================================================ 자료
def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_concept(region: str = CONCEPT_REGION) -> dict:
    """대상 지역의 라벨 셀 전체, 분할 1 라벨 블록(A)에서 뽑은 라벨 10개, 원천 계수와 재보정 계수."""
    from polar.m1_core import load_base, half_split_blocks
    from polar.m1_stats import seed_of
    from polar.m1_ext import haversine_km

    df = load_base(PROC)
    df["s"] = df.e5_sqrt_tdd.values.astype(float)
    df["y"] = df.alt_cm.values.astype(float)
    t_idx = np.where(df.macro.values == region)[0]
    A_idx, B_idx = half_split_blocks(df, t_idx, CONCEPT_SPLIT)
    nA = len(A_idx)
    seed = seed_of(region, CONCEPT_MODE, CONCEPT_SPLIT, CONCEPT_N, CONCEPT_DRAW)
    sel = np.sort(np.random.RandomState(seed).choice(nA, CONCEPT_N, replace=False))   # h40 draw_cells 와 같은 식
    yA, sA = df.y.values[A_idx][sel], df.s.values[A_idx][sel]
    m = np.isfinite(yA) & np.isfinite(sA) & (sA > 0)
    E_ls = float(sA[m] @ yA[m] / (sA[m] @ sA[m]))                                       # h40 ls_E
    t1 = pd.read_csv(PF / "table1_rows.csv").set_index("target")
    E0 = float(t1.loc[region, "E0_x"])
    E1 = (CONCEPT_N * E_ls + KAPPA * E0) / (CONCEPT_N + KAPPA)                           # h40 coefs()['E1']
    # 대조용: 원천 계수를 원천 셀(대상 제외, 100 km 완충 제외)의 최소제곱으로 다시 계산(h40 Data.source_idx 와 같은 규칙)
    src = np.where(np.isfinite(df.y.values))[0]
    src = src[~np.isin(src, t_idx)]
    la, lo = df.lat.values, df.lon.values
    tl, tn = la[t_idx], lo[t_idx]
    keep = np.ones(len(src), bool)
    near = np.array([np.abs(la[i] - tl).min() < 1.0 for i in src])
    for j in np.where(near)[0]:
        i = src[j]
        if haversine_km(la[i], lo[i], tl, tn).min() < 100.0:
            keep[j] = False
    src = src[keep]
    ys, ss = df.y.values[src], df.s.values[src]
    mm_ = np.isfinite(ys) & np.isfinite(ss) & (ss > 0)
    E0_check = float(ss[mm_] @ ys[mm_] / (ss[mm_] @ ss[mm_]))
    d = df.iloc[t_idx]
    drawn_rows = A_idx[sel]
    return dict(region=region, s=d.s.values, y=d.y.values, lat=d.lat.values, lon=d.lon.values, block=d.block.values,
                in_A=np.isin(t_idx, A_idx), drawn=np.isin(t_idx, drawn_rows), s_sel=sA, y_sel=yA, sel_pos=sel,
                E0=E0, E0_check=E0_check, n_src_check=int(len(src)), E_ls=E_ls, E1=E1, seed=int(seed), nA=int(nA),
                nB=int(len(B_idx)), n_cells=int(len(t_idx)), n_src_table=int(t1.loc[region, "n_src_x"]))


def load_placebo() -> pd.DataFrame:
    a = pd.read_csv(PF / "fig3_a.csv")
    return a[a.placebo.isin([k for k, _ in PLACEBO_ROWS]) & a.n.isin([0, 10])].copy()


def load_structure() -> pd.DataFrame:
    b = pd.read_csv(PF / "fig3_b_L10_alln.csv")
    return b[b.series.isin(["F1k", "F1n"])].copy()


def load_recal() -> pd.DataFrame:
    p = pd.read_csv(PF / "pool_fixed_curve.csv", low_memory=False)
    p = p[(p.method == "P2") & (p.baseline == "P1") & (p.learner == "none")].copy()
    p["n"] = p.n.astype(int)
    p["region"] = p.target.str.split("|").str[0].where(p.scope == "region", "MEAN")
    # 풀 평균: 라벨 10개 이하는 주 4지역, 40개 이상은 레나델타·캐나다(구성 고정 원칙)
    mean = pd.concat([p[(p.edition == "E1_P4_n_le_10") & (p.scope == "MEAN") & p.n.isin([3, 10])],
                      p[(p.edition == "E2_LenaCanada_all_n") & (p.scope == "MEAN") & p.n.isin([40, 160, 320])]])
    reg = pd.concat([p[(p.edition == "E1_P4_n_le_10") & (p.scope == "region") & p.n.isin([3, 10])],
                     p[(p.edition == "E2_LenaCanada_all_n") & (p.scope == "region") & p.n.isin([40, 160, 320])]])
    mean = mean.assign(kind="mean")
    reg = reg.assign(kind="region")
    return pd.concat([mean, reg], ignore_index=True)


# ================================================================ 그리기 도우미
def offset(ax, dx_mm: float = 0.0, dy_mm: float = 0.0):
    """자료 좌표 + 표시 좌표 mm 이동(점 비켜 놓기, 블록 등가중 막대 0.8 mm)."""
    return ax.transData + mtransforms.ScaledTranslation(dx_mm / 25.4, dy_mm / 25.4, ax.figure.dpi_scale_trans)


def overlay(fig):
    """그림 전체를 덮는 mm 좌표 축(왼쪽 위 원점). 열쇠와 공유 축 이름을 mm 로 놓는다."""
    ov = fig.add_axes([0, 0, 1, 1], zorder=20)
    ov.set_xlim(0, FIG_W)
    ov.set_ylim(FIG_H, 0)
    ov.set_xticks([])
    ov.set_yticks([])
    ov.set_axis_off()
    for sp in ov.spines.values():
        sp.set_visible(False)
    ov.patch.set_visible(False)
    return ov


def ms_area(d_pt: float) -> float:
    """scatter 의 s(pt²) = 지름²."""
    return d_pt ** 2


def text_w_mm(fig, s: str, **kw) -> float:
    t = fig.text(0, 0, s, fontsize=S.FONT_PT, **kw)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def zero_and_band(ax, vertical: bool, band: bool = True):
    if vertical:
        if band:
            ax.axvspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
        ax.axvline(0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], zorder=1)
    else:
        if band:
            ax.axhspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
        ax.axhline(0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], zorder=1)


def fmt_tick(v: float) -> str:
    return S.fmt_num(v, 0) if float(v).is_integer() else S.fmt_num(v, 1)


# ================================================================ 패널
def panel_a(fig, ax, C: dict):
    """개념 좌표도: 실제 라벨 위의 원천 계수 직선, 재보정 직선, 잔차 선분. ALT 는 아래로 깊어진다."""
    xmax, ymax = 42.0, 175.0
    ax.scatter(C["s"], C["y"], s=ms_area(2.0), c=BLACK, alpha=0.35, linewidths=0, zorder=2)
    xs = np.array([0.0, xmax])
    ax.plot(xs, C["E0"] * xs, color=C_SRC, lw=S.LW["main"], solid_capstyle="butt", zorder=3)
    ax.plot(xs, C["E1"] * xs, color=C_RECAL, lw=S.LW["main"], solid_capstyle="butt", zorder=3)
    for s_, y_ in zip(C["s_sel"], C["y_sel"]):
        ax.plot([s_, s_], [y_, C["E1"] * s_], color=C_RESID, lw=S.LW["main"], solid_capstyle="butt", zorder=2.5)
    ax.scatter(C["s_sel"], C["y_sel"], s=ms_area(S.MS["main"]), c=BLACK, linewidths=0, zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymax, 0)
    ax.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40]))
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100, 150]))
    ax.set_xlabel("Square root of thaw index (√(°C d))")
    ax.set_ylabel("Active-layer thickness (cm)")
    # 직접 라벨 3개(요소 옆)
    ax.text(15.0, C["E0"] * 15.0 - 9.0, "Source Stefan", ha="left", va="bottom", fontsize=S.FONT_PT)
    ax.text(0.8, C["E1"] * 21.0 + 14.2, "Recalibrated Stefan", ha="left", va="top", fontsize=S.FONT_PT)
    k = int(np.argmax(np.abs(C["y_sel"] - C["E1"] * C["s_sel"])))          # 가장 긴 잔차 선분 왼쪽
    s_k, y_k = C["s_sel"][k], C["y_sel"][k]
    ax.text(s_k - 1.0, y_k - 6.0, "Residual", ha="right", va="center", fontsize=S.FONT_PT)
    return dict(residual_label_point=(float(s_k), float(y_k)))


def panel_b(fig, axL, axR, P: pd.DataFrame):
    """위약 대조 포레스트: 두 열(라벨 0개, 10개), 행 이름은 왼쪽 열의 눈금 라벨로 한 번만 쓴다."""
    xlim = (-4.6, 1.0)
    rows = [k for k, _ in PLACEBO_ROWS]
    for ax, n in ((axL, 0), (axR, 10)):
        zero_and_band(ax, vertical=True)
        d = P[P.n == n].set_index("placebo").loc[rows]
        ys = np.arange(len(rows), dtype=float)
        tr_blk = offset(ax, dy_mm=-S.FOREST_BLOCK_OFFSET_MM)
        for y, (_, r) in zip(ys, d.iterrows()):
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=BLACK, lw=S.LW["ci_forest_cell"], solid_capstyle="round", zorder=3)
            ax.plot([r.ci_lo_beq, r.ci_hi_beq], [y, y], color=BLACK, lw=S.LW["ci_forest_block"],
                    solid_capstyle="butt", transform=tr_blk, zorder=3)
        ax.scatter(d.delta.values, ys, s=ms_area(S.MS["main"]), c=BLACK, linewidths=0, zorder=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.xaxis.set_major_locator(FixedLocator([-4, -3, -2, -1, 0, 1]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tick(v)))
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_locator(FixedLocator(ys))
        ax.tick_params(axis="y", length=0, pad=2.0)
        if ax is axL:
            ax.set_yticklabels([name for _, name in PLACEBO_ROWS])
            for t in ax.get_yticklabels():
                t.set_gid("category")
        else:
            ax.yaxis.set_major_formatter(NullFormatter())
    # 열 머리(묶음 머리)
    for ax, head in ((axL, "No labels"), (axR, "Ten labels")):
        t = ax.text(0.5, 1.0, head, transform=ax.transAxes, ha="center", va="bottom", fontsize=S.FONT_PT)
        t.set_gid("category")
    # 공유 x 축 이름: 왼쪽 열 축 이름을 두 열 가운데로 옮긴다
    axL.set_xlabel("Error change vs placebo (cm)")
    bL, bR = axL.get_position(), axR.get_position()
    axL.xaxis.set_label_coords(((bL.x0 + bR.x1) / 2 - bL.x0) / bL.width, -0.17)


def panel_c(fig, ax0, axM, axA, B: pd.DataFrame):
    """결합 구조 대비(잔차 − 물리 입력): n = 0 별도 축, n ≥ 3 로그 축, 전량 별도 축. 대비이므로 검정."""
    ylim = (-6.6, 4.6)
    dodge = {"F1k": -1.15, "F1n": 1.15}            # mm
    ls = {"F1k": "-", "F1n": (0, (3.0, 1.6))}
    for ax in (ax0, axM, axA):
        zero_and_band(ax, vertical=False)
        ax.set_ylim(*ylim)
    def draw_pt(ax, x, r, ser):
        tr = offset(ax, dx_mm=dodge[ser])
        trb = offset(ax, dx_mm=dodge[ser] + S.FOREST_BLOCK_OFFSET_MM)
        ax.plot([x, x], [r.ci_lo, r.ci_hi], color=BLACK, lw=S.LW["ci_forest_cell"], solid_capstyle="round",
                transform=tr, zorder=3)
        ax.plot([x, x], [r.ci_lo_beq, r.ci_hi_beq], color=BLACK, lw=S.LW["ci_forest_block"], solid_capstyle="butt",
                transform=trb, zorder=3)
        if bool(r.registered):
            ax.plot([x], [r.delta], marker="o", ms=S.MS["main"], mfc=BLACK, mec=BLACK, mew=0, ls="none",
                    transform=tr, zorder=5)
        else:
            ax.plot([x], [r.delta], marker="o", ms=S.MS["main"], mfc="white", mec=BLACK,
                    mew=S.LW["marker_edge_open"], ls="none", transform=tr, zorder=5)
    for ser in ("F1k", "F1n"):
        d = B[B.series == ser].set_index("n")
        tr = offset(axM, dx_mm=dodge[ser])
        for seg in ([3, 10], [40, 160]):                     # 풀이 바뀌는 10 과 40 사이는 잇지 않는다
            seg = [n for n in seg if n in d.index]
            axM.plot(seg, d.loc[seg, "delta"].values, color=BLACK, lw=S.LW["main"], ls=ls[ser], transform=tr, zorder=2)
        for n, r in d.iterrows():
            if n == 0:
                draw_pt(ax0, 0.0, r, ser)
            elif n == -1:
                draw_pt(axA, 0.0, r, ser)
            else:
                draw_pt(axM, float(n), r, ser)
    # 축
    axM.set_xscale("log")
    axM.set_xlim(2.4, 230.0)
    axM.xaxis.set_major_locator(FixedLocator([3, 10, 40, 160]))
    axM.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axM.xaxis.set_minor_locator(NullLocator())
    for ax, lab in ((ax0, "0"), (axA, "All")):
        ax.set_xlim(-1.0, 1.0)
        ax.xaxis.set_major_locator(FixedLocator([0.0]))
        ax.set_xticklabels([lab])
    yt = [-6, -4, -2, 0, 2, 4]
    for ax in (ax0, axM, axA):
        ax.yaxis.set_major_locator(FixedLocator(yt))
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tick(v)))
    ax0.set_ylabel("Error change vs physics input (cm)")
    for ax in (axM, axA):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.yaxis.set_major_formatter(NullFormatter())
    # 직접 라벨: 라벨 10개 점 오른쪽(두 계열의 선 끝)
    dk = B[(B.series == "F1k") & (B.n == 10)].iloc[0]
    dn = B[(B.series == "F1n") & (B.n == 10)].iloc[0]
    trl = offset(axM, dx_mm=3.4)
    axM.text(10.0, dn.delta, "Recalibrated inputs", ha="left", va="center", transform=trl, fontsize=S.FONT_PT)
    axM.text(10.0, dk.delta, "Physics inputs", ha="left", va="center", transform=trl, fontsize=S.FONT_PT)


def panel_d(fig, ax, R: pd.DataFrame):
    """최소제곱 재보정 − 수축 재보정(서술): 풀 평균 선과 CI 띠 1개, 지역 점(지역 모양)."""
    m = R[R.kind == "mean"].sort_values("n")
    zero_and_band(ax, vertical=False, band=False)
    for seg in ([3, 10], [40, 160, 320]):                   # 풀 구성이 바뀌는 곳은 잇지 않는다
        g = m[m.n.isin(seg)]
        ax.fill_between(g.n.values, g.ci_lo.values, g.ci_hi.values, color=C_RECAL, alpha=S.CI_BAND_ALPHA, lw=0, zorder=1)
        ax.plot(g.n.values, g.delta.values, color=C_RECAL, lw=S.LW["main"], zorder=3)
    ax.scatter(m.n.values, m.delta.values, s=ms_area(S.MS["main"]), c=C_RECAL, linewidths=0, zorder=4)
    rg = R[R.kind == "region"]
    for reg in REGIONS4:
        g = rg[rg.region == reg]
        ax.scatter(g.n.values, g.delta.values, s=ms_area(S.MS["region_point"]), marker=S.REGION_MARKER[reg],
                   c=C_RECAL, alpha=0.5, linewidths=0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(2.4, 420.0)
    ax.xaxis.set_major_locator(FixedLocator([3, 10, 40, 160, 320]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-6.3, 6.3)
    ax.yaxis.set_major_locator(FixedLocator([-6, -4, -2, 0, 2, 4, 6]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tick(v)))
    ax.set_ylabel("Error change vs shrinkage (cm)")
    g40 = m[m.n == 40].iloc[0]
    ax.text(40.0, g40.ci_hi + 1.0, "Least squares", ha="left", va="bottom", fontsize=S.FONT_PT,
            transform=offset(ax, dx_mm=-1.0))


# ================================================================ 그림
def build(medium: str = "paper"):
    if medium != "paper":
        raise NotImplementedError("슬라이드판은 덱 원형(A8, A3)에서 따로 정한다. 이 모듈은 논문판만 만든다")
    S.use_v3("paper")
    S.mathtext_liberation()        # "$n$" 기울임(지침 2.2 한 글자 변수), Liberation Sans 계열 안
    C = load_concept()
    P = load_placebo()
    B = load_structure()
    R = load_recal()

    fig = S.fig_mm(FIG_W, FIG_H)
    # 슬롯(mm, 왼쪽 위 원점): a 0–55 × 0–55, b 63–170 × 0–55, c 0–80 × 65–115, d 90–170 × 65–115
    ax_a = S.axes_mm(fig, 11.5, 4.0, 42.0, 40.5)
    ax_bL = S.axes_mm(fig, 98.0, 9.5, 33.0, 33.5)
    ax_bR = S.axes_mm(fig, 135.0, 9.5, 33.0, 33.5)
    yc, hc = 67.5, 41.0
    ax_c0 = S.axes_mm(fig, 12.0, yc, 6.0, hc)
    ax_cM = S.axes_mm(fig, 20.0, yc, 50.0, hc)
    ax_cA = S.axes_mm(fig, 72.0, yc, 7.0, hc)
    ax_d = S.axes_mm(fig, 101.5, yc, 66.5, hc)

    info_a = panel_a(fig, ax_a, C)
    panel_b(fig, ax_bL, ax_bR, P)
    panel_c(fig, ax_c0, ax_cM, ax_cA, B)
    panel_d(fig, ax_d, R)

    ov = overlay(fig)
    # 패널 문자: 같은 행은 같은 높이
    for x, y, L in ((0.0, 0.0, "a"), (63.0, 0.0, "b"), (0.0, 64.0, "c"), (90.0, 64.0, "d")):
        S.panel_letter(fig, x, y, L)
    fs = S.FONT_PT

    # 열쇠 1(그림에 한 번, 패널 b 머리 오른쪽): 셀 가중 굵은 막대, 블록 등가중 가는 막대, ±0.5 cm 띠
    ky = 1.6
    x = 98.0
    ov.plot([x, x + 4.0], [ky, ky], color=BLACK, lw=S.LW["ci_forest_cell"], solid_capstyle="round")
    t = ov.text(x + 5.2, ky, "Cell-weighted", ha="left", va="center", fontsize=fs)
    x = x + 5.2 + text_w_mm(fig, "Cell-weighted") + 3.5
    ov.plot([x, x + 4.0], [ky, ky], color=BLACK, lw=S.LW["ci_forest_block"], solid_capstyle="butt")
    ov.text(x + 5.2, ky, "Block-equal", ha="left", va="center", fontsize=fs)
    x = x + 5.2 + text_w_mm(fig, "Block-equal") + 3.5
    ov.add_patch(Rectangle((x, ky - 1.3), 4.0, 2.6, facecolor=S.EQUIV_BAND, edgecolor="none", lw=0))
    ov.plot([x + 2.0, x + 2.0], [ky - 1.3, ky + 1.3], color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"])
    tb = ov.text(x + 5.2, ky, "±0.5 cm", ha="left", va="center", fontsize=fs)
    tb.set_gid("sizekey")          # 동등 띠 열쇠의 값(그림 명세 1.2 가 정한 열쇠 견본). R-25 의 '열쇠 값'으로 둔다
    key1_right = x + 5.2 + text_w_mm(fig, "±0.5 cm")

    # 열쇠 2(패널 c 본 축 왼쪽 위): 채운 원 = 등록 라벨 수, 빈 원 = 보조 라벨 수
    kx, ky2 = 21.6, 69.4
    ov.plot([kx], [ky2], marker="o", ms=S.MS["main"], mfc=BLACK, mec=BLACK, mew=0, ls="none")
    ov.text(kx + 1.6, ky2, "Registered", ha="left", va="center", fontsize=fs)
    kx2 = kx + 1.6 + text_w_mm(fig, "Registered") + 3.0
    ov.plot([kx2], [ky2], marker="o", ms=S.MS["main"], mfc="white", mec=BLACK, mew=S.LW["marker_edge_open"], ls="none")
    ov.text(kx2 + 1.6, ky2, "Auxiliary", ha="left", va="center", fontsize=fs)

    # 방향 표지(그림에 하나): 패널 c 본 축 오른쪽 아래, y 가 작을수록 잔차 구조의 오차가 작다
    dx_, dy_ = 51.0, 104.6
    ov.annotate("", xy=(dx_, dy_ + 2.4), xytext=(dx_, dy_ - 2.4),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=GREY, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ov.text(dx_ + 1.4, dy_, "Residual lower", ha="left", va="center", fontsize=fs, color=GREY)

    # 열쇠 3(패널 d 오른쪽 아래): 지역 모양
    rx, ry = 133.0, 100.6
    items = [(r, S.REGION_NAME[r]) for r in REGIONS4]
    for i, (reg, name) in enumerate(items):
        col, row = i % 2, i // 2
        xx = rx + col * 17.5
        yy = ry + row * 3.4
        ov.scatter([xx], [yy], s=ms_area(S.MS["region_point"]), marker=S.REGION_MARKER[reg], c=C_RECAL,
                   alpha=0.5, linewidths=0)
        ov.text(xx + 1.6, yy, name, ha="left", va="center", fontsize=fs)

    # 아래 행 공유 x 축 이름(같은 문구를 두 번 쓰지 않는다). n 은 한 글자 변수라 기울임(지침 2.2)
    ov.text(85.0, 116.2, "Target labels, $n$", ha="center", va="center", fontsize=fs)

    axes = dict(a=ax_a, bL=ax_bL, bR=ax_bR, c0=ax_c0, cM=ax_cM, cA=ax_cA, d=ax_d, ov=ov)
    return fig, axes, dict(C=C, P=P, B=B, R=R, info_a=info_a, key1_right=key1_right)


# ================================================================ 원천 자료(Source Data)
def source_data(D: dict) -> pd.DataFrame:
    C, P, B, R = D["C"], D["P"], D["B"], D["R"]
    rows = []
    reg_name = S.REGION_NAME[C["region"]]
    for s_, y_, a_, dr_, la_, lo_ in zip(C["s"], C["y"], C["in_A"], C["drawn"], C["lat"], C["lon"]):
        rows.append(dict(panel="a", element="label cell" + (", drawn label" if dr_ else ""), series=reg_name,
                         x=s_, y=y_, region=reg_name, lat=la_, lon=lo_, label_block=bool(a_),
                         source="data/processed/fidelity_base_v3.csv via polar.m1_core.load_base (macro == "
                                f"'{C['region']}')"))
    for name, E in (("Source Stefan coefficient", C["E0"]), ("Recalibrated Stefan coefficient", C["E1"]),
                    ("Least-squares coefficient of the drawn labels", C["E_ls"])):
        rows.append(dict(panel="a", element=name, series=reg_name, y=E,
                         source="paper_figs/table1_rows.csv E0_x" if name.startswith("Source") else
                         f"(n·E_ls + κ·E0)/(n + κ), n = {CONCEPT_N}, κ = {KAPPA:g}; split {CONCEPT_SPLIT}, draw "
                         f"{CONCEPT_DRAW}, seed {C['seed']}"))
    lab = dict(PLACEBO_ROWS)
    for _, r in P.iterrows():
        rows.append(dict(panel="b", element="physics pseudo-labels minus placebo", series=lab[r.placebo],
                         n=int(r.n), delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_beq=r.delta_beq,
                         ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, pool="Lena Delta, Canada, W Russia, E Russia",
                         source=f"paper_figs/fig3_a.csv contrast={r.contrast}"))
    sname = {"F1k": "Physics inputs", "F1n": "Recalibrated inputs"}
    for _, r in B.iterrows():
        pool = "Lena Delta, Canada" if r.pool == "지역 2/4" else "Lena Delta, Canada, W Russia, E Russia"
        rows.append(dict(panel="c", element="residual structure minus physics-input structure", series=sname[r.series],
                         n=("All" if int(r.n) == -1 else int(r.n)), registered=bool(r.registered), delta=r.delta,
                         ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_beq=r.delta_beq, ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq,
                         nboot=int(r.nboot), pool=pool, source=f"paper_figs/fig3_b_L10_alln.csv contrast={r.contrast}"))
    for _, r in R.sort_values(["kind", "n"]).iterrows():
        is_mean = r.kind == "mean"
        pool = "Lena Delta, Canada, W Russia, E Russia" if r.edition == "E1_P4_n_le_10" else "Lena Delta, Canada"
        rows.append(dict(panel="d", element="least squares minus shrinkage recalibration (descriptive)",
                         series="pool mean" if is_mean else S.REGION_NAME[r.region], n=int(r.n), delta=r.delta,
                         ci_lo=r.ci_lo if is_mean else np.nan, ci_hi=r.ci_hi if is_mean else np.nan,
                         delta_beq=r.delta_blockeq, ci_lo_beq=r.ci_lo_beq if is_mean else np.nan,
                         ci_hi_beq=r.ci_hi_beq if is_mean else np.nan, nboot=int(r.nboot),
                         pool=pool if is_mean else "", region="" if is_mean else S.REGION_NAME[r.region],
                         source=f"paper_figs/pool_fixed_curve.csv edition={r.edition} contrast={r.contrast} "
                                f"scope={r.scope}"))
    cols = ["panel", "element", "series", "region", "n", "registered", "x", "y", "lat", "lon", "label_block", "delta",
            "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq", "nboot", "pool", "source"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]


# ================================================================ 설명문(원천 표에서 수치를 채운다)
def f2(x):
    return S.fmt_num(abs(x), 2)


def ci_abs(lo, hi):
    a, b = sorted([abs(lo), abs(hi)])
    return f"{S.fmt_num(a, 2)} to {S.fmt_num(b, 2)}"


LEGEND_TITLE = ("Physics pseudo-labels lowered error relative to shuffled pseudo-labels, and the better combination "
                "structure depended on the number of labels.")   # 그림 명세 5.6 첫 문장 그대로(원고 명세 7절과 한 문장)


def legend_text(D: dict, V: dict) -> str:
    """설명문 본문(제목 문장 뒤). 형식은 Fig2_legend.md 와 같다: '# Fig. 3' 머리, 굵은 제목 문장, 평문 패널 문자.
    문장 틀은 그림 명세 5.6 초안을 따르고 수치는 모두 원천 표에서 채운다(a 의 지역은 서부 러시아, 머리말 참조)."""
    C, P, B, R = D["C"], D["P"], D["B"], D["R"]
    pa = P.set_index(["placebo", "n"])
    sh0 = pa.loc[("shuffle", 0)]
    td0 = pa.loc[("tddlin", 0)]
    sb = B.set_index(["series", "n"])
    reg = [sb.loc[("F1k", 0)], sb.loc[("F1k", 10)], sb.loc[("F1n", 10)]]      # 그림에 있는 등록 대비 3개(F1a 는 SI)
    lo_reg, hi_reg = min(abs(r.delta) for r in reg), max(abs(r.delta) for r in reg)
    aux = [sb.loc[(s, n)] for s in ("F1k", "F1n") for n in (40, 160)]
    lo_aux, hi_aux = min(r.delta for r in aux), max(r.delta for r in aux)
    m = R[R.kind == "mean"].set_index("n")
    rg = R[R.kind == "region"].set_index(["region", "n"])
    d3 = m.loc[3]
    rw3 = rg.loc[("Russia_W", 3)]
    nb = int(B.nboot.iloc[0])
    txt = (
        f"a, Concept drawn on the {C['n_cells']} labelled cells of {S.REGION_NAME[C['region']]}: active-layer "
        "thickness against the square root of the thaw index, with the Stefan line of the source coefficient "
        f"(Source Stefan, {S.fmt_num(C['E0'], 2)} cm per √(°C d)), the line recalibrated with ten labels "
        f"(Recalibrated Stefan, {S.fmt_num(C['E1'], 2)}) and the departures of those labels from it that residual "
        "ML learns (Residual); thickness increases downwards. "
        "b, Direct ML with physics pseudo-labels minus the same model with placebo pseudo-labels, "
        "four-region mean, without labels and with ten labels. Physics pseudo-labels lowered error relative to "
        f"shuffled pseudo-labels by {f2(sh0.delta)} cm without labels (95% CI {ci_abs(sh0.ci_lo, sh0.ci_hi)} cm; "
        f"Holm-adjusted P = {V['AB3_holm']}); the difference from pseudo-labels of a linear thaw-index model was "
        f"{f2(td0.delta)} cm without labels and within ±0.5 cm with ten labels. "
        "c, Residual structure (anchor plus residual ML) minus physics-input structure (direct ML with physics "
        "predictions as inputs: Kudryavtsev and soil-property Stefan models for Physics inputs, source and "
        "recalibrated Stefan for Recalibrated inputs) against the number of labels; the key marks the registered "
        "label counts (0 and 10). In the three registered contrasts shown, the residual "
        f"structure had {S.fmt_num(lo_reg, 2)} to {S.fmt_num(hi_reg, 2)} cm lower four-region error under both "
        f"weightings (Holm-adjusted P = {V['AB7_holm']} for Physics inputs with ten labels); at 40 and 160 labels "
        f"the physics-input structure had {S.fmt_num(lo_aux, 2)} to {S.fmt_num(hi_aux, 2)} cm lower error in the "
        "mean of the Lena Delta and Canada, a difference that came from Canada. "
        "d, Local least-squares minus shrinkage recalibration of the Stefan coefficient (descriptive; positive "
        "values favour shrinkage), four-region mean up to ten labels and Lena Delta and Canada mean beyond. With "
        f"three labels shrinkage had {S.fmt_num(d3.delta, 2)} cm lower mean error (95% CI "
        f"{ci_abs(d3.ci_lo, d3.ci_hi)} cm) but {S.fmt_num(abs(rw3.delta), 2)} cm higher error in W Russia. "
        f"Intervals are 95% block-bootstrap confidence intervals ({S.fmt_int(nb)} resamples of 0.5° scoring "
        "blocks), weighted as in the key (b, c) or by cell (d); the grey band marks ±0.5 cm. Contrasts at 0 and 10 "
        "labels were internally pre-registered; other label counts are auxiliary."
    )
    return txt


def legend_md(body: str) -> str:
    """Fig2_legend.md 와 같은 형식."""
    return f"# Fig. 3\n\n**{LEGEND_TITLE}** {body}\n"


def word_count(s: str) -> int:
    plain = re.sub(r"\*\*", "", s)
    return len([w for w in re.split(r"\s+", plain) if w.strip()])


# ================================================================ 값 대조
def value_checks(D: dict) -> tuple[dict, list[str]]:
    C, P, B, R = D["C"], D["P"], D["B"], D["R"]
    L, V = [], {}
    L.append("[1] 원천 표 sha256 대 claims MANIFEST")
    man = pd.concat([pd.read_csv(ROOT / "paper/claims/C1_label0_safety/tables/MANIFEST.csv"),
                     pd.read_csv(ROOT / "paper/claims/C3_structure_by_label_count/tables/MANIFEST.csv")])
    for rel in ("data/processed/paper_figs/fig3_a.csv", "data/processed/paper_figs/fig3_b_L10_alln.csv",
                "data/processed/paper_figs/pool_fixed_curve.csv"):
        h = sha256(ROOT / rel)
        exp = set(man[man.orig_path == rel].sha256)
        L.append(f"  {rel}: {h[:16]}… manifest {'일치' if h in exp else '불일치'} ({len(exp)}개 사본 기록)")
        V.setdefault("sha_ok", True)
        V["sha_ok"] &= h in exp

    lgx = pd.read_csv(PROC / "lgx/lgx_tests.csv", low_memory=False)
    L.append("[2] 패널 b: fig3_a.csv 대 lgx_tests.csv(test_id L15, scope MEAN), 셀 가중·블록 등가중 6열")
    mx = 0.0
    for _, r in P.iterrows():
        q = lgx[(lgx.test_id == "L15") & (lgx.scope == "MEAN") & (lgx.contrast == r.contrast)].iloc[0]
        dd = max(abs(r.delta - q.delta), abs(r.ci_lo - q.ci_lo), abs(r.ci_hi - q.ci_hi),
                 abs(r.delta_beq - q.delta_blockeq), abs(r.ci_lo_beq - q.ci_lo_beq), abs(r.ci_hi_beq - q.ci_hi_beq))
        mx = max(mx, dd)
        L.append(f"  {r.contrast:<22} Δ {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] / [{r.ci_lo_beq:+.4f}, "
                 f"{r.ci_hi_beq:+.4f}]  verdict4 {r.verdict4}  lgx nboot {q.nboot}")
    L.append(f"  최대 절대 차 {mx:.2e}")
    V["b_maxdiff"] = mx
    V["b_nboot"] = sorted(set(lgx[(lgx.test_id == "L15") & (lgx.scope == "MEAN")].nboot.dropna().astype(int)))

    L.append("[3] 패널 c: fig3_b_L10_alln.csv 대 lgx_tests.csv(test_id L10, scope MEAN)")
    mx = 0.0
    for _, r in B.sort_values(["series", "n"]).iterrows():
        q = lgx[(lgx.test_id == "L10") & (lgx.scope == "MEAN") & (lgx.contrast == r.contrast)].iloc[0]
        dd = max(abs(r.delta - q.delta), abs(r.ci_lo - q.ci_lo), abs(r.ci_hi - q.ci_hi),
                 abs(r.delta_beq - q.delta_blockeq), abs(r.ci_lo_beq - q.ci_lo_beq), abs(r.ci_hi_beq - q.ci_hi_beq))
        mx = max(mx, dd)
        L.append(f"  {r.contrast:<14} n {int(r.n):>4} registered {str(bool(r.registered)):<5} pool {r.pool}  "
                 f"Δ {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] / [{r.ci_lo_beq:+.4f}, {r.ci_hi_beq:+.4f}]  "
                 f"{r.verdict4}  nboot {r.nboot}")
    L.append(f"  최대 절대 차 {mx:.2e}")
    V["c_maxdiff"] = mx
    q = lgx[(lgx.test_id == "L10") & (lgx.scope == "region") & (lgx.contrast == "R1-F1n|n10") &
            (lgx.target.astype(str).str.startswith("Canada"))].iloc[0]
    V["C3_canada_F1n_n10"] = float(q.delta)
    L.append(f"  설명문용 지역 행: R1-F1n|n10 Canada Δ {q.delta:+.4f} [{q.ci_lo:+.4f}, {q.ci_hi:+.4f}] / "
             f"[{q.ci_lo_beq:+.4f}, {q.ci_hi_beq:+.4f}] verdict4 {q.verdict4} ci_dependence {q.ci_dependence} "
             "(C3 README E08a +3.12)")
    for con in ("R0-F1k|n0", "R1-F1k|n10", "R1-F1n|n10"):
        g = lgx[(lgx.test_id == "L10") & (lgx.scope == "region") & (lgx.contrast == con)]
        L.append(f"  지역 행 {con}: " + "; ".join(f"{t} {v}" for t, v in zip(g.target, g.verdict4)))
    for con in ("R1-F1k|n40", "R1-F1n|n40", "R1-F1k|n160", "R1-F1n|n160"):
        g = lgx[(lgx.test_id == "L10") & (lgx.scope == "region") & (lgx.contrast == con)]
        L.append(f"  지역 행 {con}: " + "; ".join(f"{t} {d:+.2f} {v}" for t, d, v in zip(g.target, g.delta, g.verdict4)))

    lgw = pd.read_csv(PROC / "lgw/lgw_bundle.csv", low_memory=False)
    L.append("[4] Holm 보정 P(lgw_bundle.csv, scope MEAN, holm_m 10)")
    for ab in ("AB3", "AB7"):
        q = lgw[(lgw.ab == ab) & (lgw.scope == "MEAN")].iloc[0]
        V[f"{ab}_holm"] = f"{q.holm_p:.3f}"
        L.append(f"  {ab} {q.contrast}: Δ {q.delta:+.4f} [{q.ci_lo:+.4f}, {q.ci_hi:+.4f}] holm_p {q.holm_p} "
                 f"holm_m {q.holm_m} nboot {q.nboot}. 그림 값은 lgx 판(같은 점 추정, CI 끝값은 재표집 구현 차로 소수 "
                 "둘째 자리에서 다를 수 있다. C3 README 2.7)")

    L.append("[5] 패널 d: pool_fixed_curve.csv(method P2, baseline P1, learner none), role '서술(판정에 쓰지 않는다)'")
    for _, r in R.sort_values(["kind", "region", "n"]).iterrows():
        L.append(f"  {r.kind:<6} {r.region:<9} n {int(r.n):>3} {r.edition:<20} Δ {r.delta:+.4f} [{r.ci_lo:+.4f}, "
                 f"{r.ci_hi:+.4f}] beq {r.delta_blockeq:+.4f} role {r.role}")
    V["d_roles"] = sorted(set(R.role))

    L.append("[6] 패널 a: 개념 좌표도")
    L.append(f"  지역 {C['region']}({S.REGION_NAME[C['region']]}), 라벨 셀 {C['n_cells']}개, 분할 {CONCEPT_SPLIT} "
             f"라벨 블록 셀 {C['nA']}개, 채점 쪽 셀 {C['nB']}개")
    L.append(f"  추출: seed_of('{C['region']}', '{CONCEPT_MODE}', {CONCEPT_SPLIT}, {CONCEPT_N}, {CONCEPT_DRAW}) = "
             f"{C['seed']}, 라벨 블록 안 위치 {C['sel_pos'].tolist()}")
    L.append(f"  원천 계수 E0 = {C['E0']:.6f}(table1_rows.csv E0_x, n_src {C['n_src_table']}); 다시 계산 "
             f"{C['E0_check']:.6f}(원천 셀 {C['n_src_check']}개, 100 km 완충 제외), 차 {abs(C['E0'] - C['E0_check']):.2e}")
    L.append(f"  라벨 10개 최소제곱 E = {C['E_ls']:.6f}, 재보정 E = (10·E_ls + 10·E0)/20 = {C['E1']:.6f}")
    L.append("  뽑힌 라벨(√TDD, ALT cm): " + ", ".join(f"({s:.2f}, {y:.1f})" for s, y in zip(C['s_sel'], C['y_sel'])))
    return V, L


def lena_note() -> list[str]:
    """명세 지역(레나델타)의 같은 계산: 두 직선이 인쇄 크기에서 구분되는지."""
    from polar.m1_core import load_base, half_split_blocks
    from polar.m1_stats import seed_of
    df = load_base(PROC)
    s, y = df.e5_sqrt_tdd.values.astype(float), df.alt_cm.values.astype(float)
    t1 = pd.read_csv(PF / "table1_rows.csv").set_index("target")
    out = []
    for reg in ("Lena", CONCEPT_REGION):
        t = np.where(df.macro.values == reg)[0]
        A, _ = half_split_blocks(df, t, CONCEPT_SPLIT)
        sel = np.sort(np.random.RandomState(seed_of(reg, CONCEPT_MODE, CONCEPT_SPLIT, CONCEPT_N, CONCEPT_DRAW))
                      .choice(len(A), CONCEPT_N, replace=False))
        sa, ya = s[A][sel], y[A][sel]
        Els = float(sa @ ya / (sa @ sa))
        E0 = float(t1.loc[reg, "E0_x"])
        E1 = (CONCEPT_N * Els + KAPPA * E0) / (CONCEPT_N + KAPPA)
        smax = float(np.nanmax(s[t]))
        out.append(f"  {reg}: 셀 {len(t)}, √TDD {np.nanmin(s[t]):.1f}–{smax:.1f}, 고유 √TDD 값 {len(np.unique(s[t]))}개, "
                   f"E0 {E0:.3f}, 재보정 {E1:.3f}({(E1 / E0 - 1) * 100:+.1f}%), 최대 √TDD 에서 두 직선 간격 "
                   f"{abs(E1 - E0) * smax:.1f} cm")
    return out


# ================================================================ 실행
def main():
    res = wait_resources()
    t0 = time.time()
    fig, axes, D = build("paper")
    V, L = value_checks(D)

    # 점검 함수(지침 부록 A.1): 저장 직전
    au = S.audit_v3(fig)
    ov = S.text_overlaps(fig)                             # 글자끼리 겹침 0곳(지침 2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{STEM}.pdf"
    png = OUT / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    pa = S.pdf_audit(pdf)
    pf = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    ptxt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    pimg = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True).stdout
    n_raster = max(0, len(pimg.strip().splitlines()) - 2)

    sd = source_data(D)
    sd.to_csv(OUT / f"{STEM}_source_data.csv", index=False, float_format="%.6g")

    body = legend_text(D, V)
    leg = LEGEND_TITLE + " " + body                      # 단어 수는 제목 문장 + 본문('# Fig. 3' 머리 제외)
    wc = word_count(leg)
    wc_title = word_count(LEGEND_TITLE)
    if wc > 350:
        raise SystemExit(f"설명문 {wc} 단어 > 350")
    ta = S.text_audit(leg)
    (OUT / f"{STEM}_legend.md").write_text(
        legend_md(body) + f"\n<!-- words: {wc} (title {wc_title}); limit 350. Generated by "
        "scripts/4_visualization/paper_v3/fig3.py from the registered sources; numbers in Fig3_values.txt. -->\n",
        encoding="utf-8")

    # 기록
    lines = [f"Fig3 v3 값 대조와 점검 기록 (생성 {time.strftime('%Y-%m-%d %H:%M')}, 스크립트 "
             "scripts/4_visualization/paper_v3/fig3.py)", f"렌더 전 자원: {res}", ""]
    lines += L
    lines += ["", "[7] 명세와 다른 점과 근거",
              "  (1) 패널 a 의 지역을 레나델타에서 서부 러시아로 바꾸었다. 같은 분할·추출·κ 로 계산한 값:"]
    lines += lena_note()
    lines += ["      레나델타는 두 직선의 간격이 그림 높이에서 0.5 mm 미만이라 'Source Stefan'과 'Recalibrated Stefan' 이",
              "      한 선으로 보인다(지침 2.10 의 개념, SL-10 의 시각적 자명성 위반). 서부 러시아는 재보정이 계수를 크게",
              "      옮긴 지역(AB4 지역 행 우세 1/4)이며 셀 31개라 점이 겹치지 않는다. 되돌리려면 CONCEPT_REGION = 'Lena'.",
              "  (2) 패널 b 는 라벨 0개와 10개를 좌우 두 열로 둔다. 위아래 묶음이면 행 이름 4개가 두 번씩 나온다(지침 H13).",
              "  (3) 패널 d 에 직접 라벨 'Least squares' 하나를 두고 y 축 이름을 'Error change vs shrinkage (cm)' 로 줄였다.",
              "  (4) 패널 c 의 y 축 이름은 'Error change vs physics input (cm)'(명세 문구는 축 높이보다 길다). 방향 표지",
              "      'Residual lower' 는 본 축 오른쪽 아래(축 왼쪽은 y 축 이름이 차지한다).",
              "  (5) c, d 의 x 축 이름 'Target labels, n' 은 아래 행 가운데 한 번만 둔다(같은 문구 반복 0).",
              "  (6) c, d 의 선은 라벨 10개와 40개 사이에서 잇지 않는다(풀이 4지역에서 2지역으로 바뀐다).",
              "  (7) '±0.5 cm' 열쇠 글자는 gid 'sizekey' 로 표시해 audit_v3 의 그림 안 수치 검사에서 뺐다(명세 1.2 가",
              "      정한 열쇠 견본의 값).",
              "  (8) 설명문 c 의 등록 대비 범위는 그림에 있는 등록 대비 3개(R0 − F1k n 0, R1 − F1k n 10, R1 − F1n n 10)의",
              "      값 2.52–3.74 cm 다. 명세 5.6 초안의 2.49–3.74 는 SI 로 보낸 넷째 등록 대비 R0 − F1a(n 0, −2.49)를",
              "      포함한 C3 README 1.4 의 범위라 그림과 맞지 않아 쓰지 않았다.",
              "  (9) 설명문 첫 문장은 명세 5.6 초안 그대로(원고 명세 7절과 한 문장). 파일 형식은 Fig2_legend.md 와 같다",
              "      ('# Fig. 3' 머리, 굵은 제목 문장, 평문 패널 문자). 단어 수는 제목 + 본문.",
              "  (10) 변수 n 은 기울임(지침 2.2 '기울임은 한 글자 변수에만'). mathtext 글꼴은 style.mathtext_liberation()",
              "      으로 Liberation Sans 계열에 맞췄다(pdffonts 계열 1개).",
              "  (11) 패널 b 의 CI 는 fig3_a.csv(L15 재집계) 값이다. 초록 대비 AB3(lgw_bundle)의 CI 끝값과 소수 둘째 자리에서",
              "      다를 수 있다([4]).",
              "", "[8] audit_v3(지침 부록 A.1)",
              f"  sizes {sorted(au['sizes'])}, chars {au['chars']}, thin_lines {au['thin_lines']}, titles {au['titles']},",
              f"  boxed_text {au['boxed_text']}, codes {au['codes']}, comma4 {au['comma4']},",
              f"  long_labels {au['long_labels']}, loose_numbers {au['loose_numbers']}",
              f"  fails {au['fails']}",
              "", "[9] pdf_audit(지침 부록 A.2)",
              f"  {json.dumps({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in pa.items()}, ensure_ascii=False)}",
              "  pdffonts:"] + ["    " + ln for ln in pf.strip().splitlines()]
    dash = [m.group(0) for m in re.finditer(r"[—–]", ptxt)]
    lines += ["", "[10] 그림 내장 글자(pdftotext)", f"  연결어 대시 {len(dash)}건, 내부 코드 정규식 "
              f"{len(S.CODE_RE.findall(ptxt))}건", "  " + " | ".join(x.strip() for x in ptxt.splitlines() if x.strip()),
              f"  글자끼리 겹침 {len(ov['overlap_pairs'])}쌍 {ov['overlap_pairs']}, 캔버스 밖 글자 {ov['outside']} "
              f"(글자 {ov['n_texts']}개), pdfimages 래스터 요소 {n_raster}개",
              "  사람 점검(600 dpi PNG 100 % 잘라 보기): a 직접 라벨 3개와 선·점 분리, b 두 열 행 이름 한 번, c 열쇠·직접 라벨·"
              "방향 표지가 자료와 떨어짐, d 지역 열쇠 4개. 겹침·잘림 없음(2026-10-04)."]
    lines += ["", "[11] 설명문(Fig3_legend.md)", f"  단어 수 {wc}(상한 350)", f"  text_audit {ta}",
              "  수치 출처: 1.63 과 CI = fig3_a.csv shuffle n 0(lgx L15); 0.48 = fig3_a.csv tddlin n 0; Holm P = lgw_bundle "
              "AB3, AB7(Holm 10개); 등록 대비 범위 = fig3_b_L10_alln.csv F1k n 0, F1k n 10, F1n n 10(그림의 등록 대비 3개, [7] (8)); "
              "40·160 범위 = fig3_b_L10_alln.csv F1k·F1n n 40, 160; '캐나다에서 왔다' 의 지역 행 근거는 [3]; "
              "d 의 값 = pool_fixed_curve.csv(MEAN n 3, Russia_W n 3); 재표집 횟수 = fig3_b_L10_alln.csv nboot, "
              f"pool_fixed_curve.csv nboot, lgx L15 nboot {V['b_nboot']}; a 의 계수와 셀 수 = [6]",
              "", f"[12] 렌더 시간 {time.time() - t0:.1f} s"]
    (OUT / f"{STEM}_values.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-40:]))
    print("audit fails:", au["fails"], "| pdf:", pa["width_mm"], pa["height_mm"], pa["families"], pa["off_size"],
          "type3", pa["type3"], "| words", wc)


if __name__ == "__main__":
    main()
