"""Fig 3. 라벨 예산 계단(H25 블록 CI 재실행, a–d)과 부분 풀링(B2, e). 정본 스펙: figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.3.

주장: 라벨 수가 늘 때 E 재적합·증강·잔차 ML 의 이득은 지역 오차 유형과 라벨 배치 규칙에 따라 다르게 변한다(H25).
고정 κ=10 수축은 관측 범위(n 3–320)의 14 대상 평균에서 가장 견고하며, 적응형 풀링 추정기는 이를 넘지 못한다(F5 기각).

자료
  a–d  h3/h25b_curve.csv(블록 CI 재실행, 스펙 §2.3 전환 경로). 계열:
       S1 κ=10 E 재적합(k-중심 실선·무작위 파선), S2 의사라벨 증강(k-중심), S3 잔차 ML λ 0.25(k-중심 실선·무작위 파선).
       S3 무작위는 레나에서 k-중심과 부호가 갈리는 차이(n ≥ 20 개선 대 n 10–80 악화)를 보이기 위해 추가했다.
       잔차 ML CI = 95 % 블록 부트스트랩 CI(blk_d_phys_lo/hi): k-중심 = 채운 띠, 무작위 = 파선 테두리(범례 두 항목).
       참조선 = scope allA, S3, λ 0.25, α 1 의 분할 평균(파선) + 분할 범위 수염(d_phys_splits JSON 의 분할별 값,
       축 상자 우측 바깥 1.5 mm). 물리식 0선은 범례에 넣지 않고 캡션에 적는다(범례 8 항목 상한).
  e    h4/b2_regional.csv: vs = phys, region_set = ALL, variant = E(앵커만, 잔차 없음), rule = random, scope = n.
       대상 층화 평균(대상별 블록 부트스트랩 분포를 같은 번호끼리 평균, h33_partial_pooling.regional_table).
       n 3–10 은 14 대상(CA-1 제외), n 20–160 은 12(러시아 W·E 라벨 풀 소진), n 320 은 9. 대상 집합이 바뀌는 곳에서
       선을 끊고 세로 점선과 대상 수 표지로 구분한다. 혼합효과 부스팅은 전 구간 축 위(+5.3 ~ +10.4 cm)이므로
       축 위 끝의 위쪽 화살표(열린 머리)와 수치로 표기한다(off-scale). ▲ 는 이 연재에서 '최소 n 미달성' 전용이므로 쓰지 않는다.
       네 추정기는 모두 E 추정 방법(앵커만)이므로 METHOD 의 refit 파랑 하나로 그리고, 선종 + 마커로 구분한다
       (κ=10 파선 o = a–d 'E shrinkage, random labels' 와 같은 부호, 오프셋 MLE 일점쇄선 (0,(4,2,1,2)) s = Fig 4 와 같은 선종,
       PPI++ 점선 P, 혼합효과 부스팅 ↑). 새 색을 만들지 않는다(METHOD 색 고정).
오차 유형 분류(감사 조치 9): 구조 |log E비| < 0.15, 수준 ≥ 0.2, 그 사이 = 중간.
  표시: 축 상자 위 3 mm 머리 띠(구조 = 채움 없음, 수준 = #f2f2f2, 중간 = Fig 4 와 같은 방향의 옅은 회색 빗금) + 우측 범주 글자.
  축 배경 전체를 칠하지 않는 이유: 같은 패널의 n > 라벨 풀 음영이 #f2f2f2 이므로(스펙 §2.3) 수준 패널(러시아 W)에서 겹친다.
배치(mm, 좌하 원점, 180 × 118): 행 1 축 상자 y 73–113(a 14+36, b–d 4+36, 우 10 = 참조 수수염 여백).
  행 2 축 상자 y 11–59(e 14+100), 범례(그림당 1개, 8 항목) x 118–180 y 11–59, 높이 약 44 mm 로 세로 채움(QA 지적: 빈 띠).
  e 추정기 표지는 e 축 안 위 가운데 4줄, κ=10 항목 이름은 a–d 범례와 같게 'E shrinkage (κ = 10), random labels'.
  끝 틱 = T 모양(0.8 pt + 1.2 mm 캡). --qa 때 _qa/<name>_crop_{ab,e}_300dpi.png(인쇄 크기 잘라내기)를 만든다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from _common import (ps, rd, H4, load_h25_curve, load_h25_targets, h25_series, h25_allA_ref, save_paper,
                     MissingData, QA_OPTS, QA_DIR)  # noqa: F401

MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]            # 구조 2 → 수준 → 중간 순(스펙 §2.3 a–d)
Y_LIM = (-14.0, 6.0)
Y_TICKS = (-10, -5, -2, -1, 0, 1, 2, 5)
REF_WHISKER_MM = 1.5                                          # 참조선 분할 범위 수염: 축 상자 우측 바깥 거리(mm)
LINSCALE = 0.45                                                # 선형 ±2 cm 구간 폭(10 진 단위)
STRUCT_MAX, LEVEL_MIN = 0.15, 0.20

# a–d 계열: (단계, 규칙, λ, 방법 키, 선종 변형, 범례 라벨)
SERIES = [
    ("S1", "kmedoid", 0.0, "refit", None, "E shrinkage (κ = 10), k-medoid labels"),
    ("S1", "random", 0.0, "refit", "dashed", "E shrinkage (κ = 10), random labels"),
    ("S2", "kmedoid", 1.0, "augment", None, "+ pseudo-label augmentation, k-medoid"),
    ("S3", "kmedoid", 0.25, "residual", None, "+ residual ML (λ = 0.25), k-medoid"),
    ("S3", "random", 0.25, "residual", "dashed", "+ residual ML (λ = 0.25), random"),
]

# e 추정기: (키, 선종, 마커, 표지 글). 색은 전부 METHOD refit(E 추정 방법 계열). 'up' = 축 위 화살표(off-scale)
LS_OFFSET = (0, (4.0, 2.0, 1.0, 2.0))                          # Fig 4 오프셋 MLE 와 같은 일점쇄선
E_EST = [
    ("shrink_k10", ps.VARIANT_LS["dashed"], "o", "E shrinkage (κ = 10), random"),   # e 안 범례: 긴 이름이 표적 집합 경계선(n≈14)에 닿아 줄임   # a–d 범례와 같은 이름(= Fig 4 shrinkage)
    ("offset_mle", LS_OFFSET, "s", "Offset MLE"),
    ("ppi", (0, (1.0, 1.3)), "P", "PPI++"),
    ("mixed_boost", None, "up", "Mixed-effects boosting"),
]
E_COL = ps.COLOR["refit"]
E_LW = 0.8
OFF_ARROW_MM = 2.6                                              # off-scale 화살표 길이(mm)
RAND_CI_LS = (0, (2.0, 1.5))                                    # 무작위 라벨 잔차 ML CI 테두리(파선)
HEAD_MM = 3.0                                                   # a–d 머리 띠 높이(mm)
HATCH_INTER = dict(facecolor="none", edgecolor="#d9d9d9", hatch="////", lw=0)     # Fig 4 중간 빗금과 같은 방향, 3 mm 머리 띠라 밀도·명도 낮춤
BAND_STAGES = ("S3",)
E_VARIANT, E_RULE = "E", "random"
E_YLIM = (-2.0, 3.0)
E_XLIM = (2.5, 420)
E_DODGE = 1.13                                                 # 로그 x 추정기 간 곱 간격
E_W_MM = 100.0                                                 # 행 2: e 축 폭(x 14–114)
LEG_RECT = (118.0, 11.0, 62.0, 48.0)                           # 행 2 범례 사각형(x, y, w, h mm)
LEG_FILL_MM = 44.0                                             # 범례 목표 높이(e 축 높이 48 mm 의 대부분을 채움)
END_TICK_CAP_MM = 1.2                                          # 끝 틱 T 캡 폭(mm)


def err_class(v: float) -> str:
    """감사 조치 9 의 통일 분류(오라클 E_own 기반 사후 지표)."""
    if not np.isfinite(v):
        return ""
    return "structure" if v < STRUCT_MAX else ("level" if v >= LEVEL_MIN else "intermediate")


def _split_values(C: pd.DataFrame, target: str, stage: str = "S3", lam: float = 0.25, alpha: float = 1.0) -> np.ndarray:
    """전량 A 라벨 참조의 분할별 값(감사 조치 8: 분할 간 편차 공개용)."""
    q = C[(C.target == target) & (C.scope == "allA") & (C.stage == stage) & np.isclose(C.lam, lam) & np.isclose(C.alpha, alpha)]
    if not len(q):
        return np.array([])
    # allA 는 대상당 1행(분할 평균)이고 분할별 값은 d_phys_splits JSON 열에 있다.
    raw = q.d_phys_splits.iloc[0] if "d_phys_splits" in q.columns else None
    if isinstance(raw, str) and raw.strip().startswith("{"):
        v = np.array([float(x) for x in json.loads(raw).values()], float)
        return v[np.isfinite(v)]
    return q.d_phys_mean.to_numpy(float)


def _symlog_y(ax, linscale: float = LINSCALE):
    """스펙 §2.3 공유 symlog(linthresh 2 cm), linscale 만 줄여 수준 지역(−5 ~ −12 cm)이 눌리지 않게 한다."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_yscale("symlog", linthresh=2.0, linscale=linscale)
    ax.set_ylim(*Y_LIM)
    ax.yaxis.set_major_locator(FixedLocator(list(Y_TICKS))); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))


def _end_tick(ax, n_last: float, yv, ref: float, h_mm: float = 40.0, w_mm: float = 36.0):
    """곡선 끝 3 mm 세로 틱 + 데이터 반대쪽 끝의 가로 캡(T 모양, 0.8 pt): 끝점 무리 아래 1 mm(공간 부족이면 위).
    캡 없는 가는 선은 축 눈금이나 캡 없는 CI 로 오인되므로(QA 지적) 캡과 굵기로 구분한다. 축 좌표로 그린다."""
    if not len(yv):
        return None
    to_ax = lambda v: ax.transLimits.transform(ax.transScale.transform((1.0, v)))[1]
    lo_a, hi_a = min(to_ax(v) for v in yv), max(to_ax(v) for v in yv)
    gap, L = 1.0 / h_mm, 3.0 / h_mm
    floor = max(0.0, to_ax(ref) if np.isfinite(ref) and to_ax(ref) < lo_a else 0.0)
    if lo_a - floor >= 4.5 / h_mm:
        y0, y1, yc = lo_a - gap - L, lo_a - gap, lo_a - gap - L      # 아래: 캡은 아래 끝
    else:
        y0, y1, yc = hi_a + gap, hi_a + gap + L, hi_a + gap + L      # 위: 캡은 위 끝
    xa = ax.transLimits.transform(ax.transScale.transform((n_last, 1.0)))[0]
    cw = END_TICK_CAP_MM / 2 / w_mm
    kw = dict(color=ps.GREY["text2"], lw=0.8, transform=ax.transAxes, solid_capstyle="butt", zorder=4, clip_on=False)
    ln, = ax.plot([xa, xa], [y0, y1], **kw)
    cap, = ax.plot([xa - cw, xa + cw], [yc, yc], **kw)
    ln.set_gid("end_tick"); cap.set_gid("end_tick_cap")
    return ln


def _load_e() -> pd.DataFrame:
    R = rd(H4 / "b2_regional.csv")
    q = R[(R.vs == "phys") & (R.region_set == "ALL") & (R.variant == E_VARIANT) & (R.rule == E_RULE) & (R.scope == "n")
          & R.est.isin([k for k, *_ in E_EST])].copy()
    if not len(q) or not (q.n_targets == 14).any():
        raise MissingData("b2_regional: ALL · variant E · random · 14 대상 행 없음")
    q.attrs.update(R.attrs)
    return q.sort_values(["est", "n"])


def _up_arrow(ax, x, y_top_ax: float, length_ax: float, transform, zorder: float = 4.0):
    """off-scale 표지: 열린 머리 위쪽 화살표(▲ 와 다른 기호). x 는 transform 의 x 좌표, y 는 축 좌표."""
    a = ax.annotate("", xy=(x, y_top_ax), xycoords=transform, xytext=(x, y_top_ax - length_ax), textcoords=transform,
                    arrowprops=dict(arrowstyle="->,head_length=0.28,head_width=0.16", color=E_COL, lw=E_LW,
                                    shrinkA=0, shrinkB=0), annotation_clip=False, zorder=zorder)
    a.set_gid("offscale_arrow")
    return a


def _e_key(ax, w_mm: float = 84.0, h_mm: float = 48.0, x0: float = 0.475, top_mm: float = 8.2, pitch_mm: float = 3.0):
    """e 축 안 추정기 표지(4줄, 축 좌표). 그림 범례와 따로 세지 않도록 범례 객체를 만들지 않는다."""
    hl = 6.0 / w_mm
    for j, (key, ls_, mk, s_) in enumerate(E_EST):
        y = 1.0 - (top_mm + j * pitch_mm) / h_mm
        if mk == "up":
            xm = x0 + hl / 2
            _up_arrow(ax, xm, y + 1.1 / h_mm, 2.2 / h_mm, ax.transAxes, zorder=5)
        else:
            ax.plot([x0, x0 + hl], [y, y], color=E_COL, lw=E_LW, ls=ls_, transform=ax.transAxes, zorder=5)
            ax.plot([x0 + hl / 2], [y], ls="none", marker=mk, ms=ps.MS["main"] + (0.4 if mk == "P" else 0.0), mfc=E_COL,
                    mec=E_COL, mew=ps.LW["marker_edge"], transform=ax.transAxes, zorder=5)
        ax.text(x0 + hl + 1.6 / w_mm, y, s_, transform=ax.transAxes, ha="left", va="center", fontsize=ps.FS["legend"],
                zorder=5).set_gid("e_key")


def _header(fig, ax, name: str, v: float, cls: str):
    """축 상자 위 머리 띠: 오차 유형 채움(구조 없음, 수준 #f2f2f2, 중간 빗금) + 좌 지역 라벨 + 우 범주 글자."""
    x0, y0, w, h = ax._box_mm
    yb = y0 + h + 0.6
    W, H = ps.fig_mm(fig)
    kw = dict(transform=fig.transFigure, zorder=0.5)
    if cls == "level":
        fig.add_artist(Rectangle((x0 / W, yb / H), w / W, HEAD_MM / H, facecolor=ps.GREY["band"], edgecolor="none", **kw))
    elif cls == "intermediate":
        fig.add_artist(Rectangle((x0 / W, yb / H), w / W, HEAD_MM / H, **HATCH_INTER, **kw))
    ym = (yb + HEAD_MM / 2) / H
    box = dict(boxstyle="square,pad=0.12", fc="white", ec="none") if cls == "intermediate" else None
    fig.text((x0 + 0.8) / W, ym, ps.region_label(name, v), ha="left", va="center", fontsize=ps.FS["annot"],
             color="#000000", bbox=box).set_gid("region_label")
    fig.text((x0 + w - 0.8) / W, ym, cls, ha="right", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"],
             bbox=box).set_gid("err_class")


def _fill_legend(fig, handles):
    """행 2 범례(8 항목, 1열): labelspacing 을 조정해 높이를 LEG_FILL_MM 에 맞추고 사각형 안 세로 가운데에 둔다."""
    lg = None
    sp = 0.5
    for _ in range(6):
        if lg is not None:
            lg.remove()
        lg = ps.legend_below(fig, handles, LEG_RECT, ncol=1, labelspacing=sp, handlelength=2.6, borderaxespad=0.0,
                             borderpad=0.0)
        lg._legend_box.align = "left"
        fig.canvas.draw()
        h_mm = lg.get_window_extent().height / fig.dpi * 25.4
        # 높이 ≈ 8 줄 글자 + 7 간격 × sp × 글자 크기 → sp 를 선형 보정
        fs_mm = ps.FS["legend"] * 25.4 / 72
        sp = max(0.3, sp + (LEG_FILL_MM - h_mm) / (7 * fs_mm))
        if abs(LEG_FILL_MM - h_mm) < 0.3:
            break
    return lg


def build(draft: bool = False):
    ps.use_paper()
    C = load_h25_curve()
    if C.attrs.get("ci_kind") != "block":
        raise MissingData("h3/h25b_curve.csv(블록 CI) 없음")
    T = load_h25_targets()
    alr = dict(zip(T.target, T.abslogE))
    pool_n = dict(zip(T.target, T.n_A))
    E = _load_e()

    fig = ps.paper_figure(180, 118)

    # ------------------------------------------------------------ a–d 계단
    axes, rows_ad, whisk = [], [], {}
    slots = [(0, 14), (50, 4), (90, 4), (130, 4)]
    for i, (t, (x0, lab)) in enumerate(zip(MAIN4, slots)):
        ax = ps.slot_mm(fig, x0, 73, lab, 36, 40)
        ps.zero_line(ax, "y", better_text=(i == 3))
        _symlog_y(ax)
        ps.log_n_axis(ax, ticks=(3, 10, 30, 100, 300), lim=(2.5, 380))
        n_last, ends = 0, []
        for st, rule, lam, m, var, lbl in SERIES:
            q = h25_series(C, t, st, rule, lam, 1.0)
            if not len(q):
                continue
            # CI 는 잔차 ML 두 규칙만 그린다(5 계열 띠 중첩 방지, 나머지 CI 는 원자료 표). k-중심 = 채운 띠,
            # 무작위 = 파선 테두리(채움 없음): 같은 색 두 띠가 한 덩어리로 보이지 않게 한다.
            band = st in BAND_STAGES and rule == "kmedoid"
            ps.plot_curve(ax, q.n, q.blk_d_phys, q.blk_d_phys_lo if band else None, q.blk_d_phys_hi if band else None,
                          method=m, variant=var, band=band, label=lbl, delta=False)
            if st in BAND_STAGES and rule == "random":
                for col_ in ("blk_d_phys_lo", "blk_d_phys_hi"):
                    ax.plot(q.n, q[col_], color=ps.COLOR[m], lw=0.5, ls=RAND_CI_LS, zorder=2.2)[0].set_gid("ci_edge_random")
            n_last = max(n_last, int(q.n.max()))
            ends.append((int(q.n.max()), float(q.sort_values("n").blk_d_phys.iloc[-1])))
            for _, r in q.iterrows():
                rows_ad.append(dict(panel="abcd"[i], target=t, stage=st, rule=rule, lam=lam, alpha=1.0, n=int(r.n),
                                    d_phys=float(r.blk_d_phys), ci_lo=float(r.blk_d_phys_lo), ci_hi=float(r.blk_d_phys_hi),
                                    split_win=float(r.split_win), rep_win=float(r.rep_win), n_runs=int(r.n_runs),
                                    n_splits=int(r.n_splits), ci_kind="block"))
        ref = h25_allA_ref(C, t)
        if np.isfinite(ref):
            ps.ref_line(ax, ref, "ref_allA")
            sv = _split_values(C, t)
            if len(sv) > 1:   # 분할 간 범위: 축 상자 우측 바깥 1.5 mm 세로 수염(감사 조치 8)
                trw = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
                xw, cw = 1.0 + REF_WHISKER_MM / 36.0, 0.6 / 36.0
                ax.plot([xw, xw], [sv.min(), sv.max()], color=ps.COLOR["ref_allA"], lw=ps.LW["ci"], transform=trw,
                        solid_capstyle="butt", zorder=3.5, clip_on=False)
                for v_ in (sv.min(), sv.max()):   # 캡(끝 가로 0.6 mm × 2)
                    ax.plot([xw - cw, xw + cw], [v_, v_], color=ps.COLOR["ref_allA"], lw=ps.LW["ci"], transform=trw,
                            solid_capstyle="butt", zorder=3.5, clip_on=False)
            if ref < Y_TICKS[0]:          # 눈금 밖 구간의 참조값은 수치로 적는다
                ax.text(60, ref, ps.fmt_num(ref, 1), ha="center", va="bottom", fontsize=ps.FS["annot"],
                        color=ps.COLOR["ref_allA"], zorder=4)
            rows_ad.append(dict(panel="abcd"[i], target=t, stage="S3", rule="all", lam=0.25, alpha=1.0, n=-1, d_phys=ref,
                                ci_lo=float(sv.min()) if len(sv) else np.nan, ci_hi=float(sv.max()) if len(sv) else np.nan,
                                n_runs=np.nan, n_splits=len(sv), ci_kind="split_range",
                                split_values=";".join(f"{v_:.4f}" for v_ in sv)))
            whisk[t] = sv
        npool = pool_n.get(t, np.inf)
        if n_last and n_last < 320:
            ax.axvspan(max(npool, n_last), 380, color=ps.GREY["band"], lw=0, zorder=0.2)
            _end_tick(ax, n_last, [v for nn, v in ends if nn == n_last], ref)
        _header(fig, ax, t, alr.get(t), err_class(alr.get(t, np.nan)))
        if i == 0:
            ax.set_ylabel("ΔRMSE vs physics (cm)")
        else:
            ps.hide_ylabels(ax)
        axes.append(ax)
    fx, fy = ps.mm_to_fig(fig, (14 + 170) / 2, 64.2)
    fig.text(fx, fy, "Target labels n", ha="center", va="bottom", fontsize=ps.FS["label"]).set_gid("shared_xlabel")

    # ------------------------------------------------------------ e 부분 풀링(14 대상 평균 Δ 대 n)
    ax_e = ps.slot_mm(fig, 0, 11, 14, E_W_MM, 48)
    ps.zero_line(ax_e, "y", better_text=False)
    ax_e.set_yscale("linear"); ax_e.set_ylim(*E_YLIM)
    ax_e.set_yticks([-2, -1, 0, 1, 2, 3])
    ax_e.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ps.log_n_axis(ax_e, ticks=(3, 10, 30, 100, 300), lim=E_XLIM)
    ax_e.set_ylabel("ΔRMSE vs physics (cm)")
    ax_e.set_xlabel("Target labels n")
    # 대상 집합 구간(선 끊김 경계)
    sets = E.groupby("n").n_targets.first().sort_index()
    seg_id = (sets != sets.shift()).cumsum()
    segs = [list(g.index) for _, g in sets.groupby(seg_id)]
    for a_, b_ in zip(segs[:-1], segs[1:]):
        xb = float(np.sqrt(a_[-1] * b_[0]))
        ax_e.axvline(xb, color=ps.GREY["light"], lw=0.5, ls=(0, (1.5, 1.5)), zorder=0.3)
    tr_ax = matplotlib.transforms.blended_transform_factory(ax_e.transData, ax_e.transAxes)
    bounds = [E_XLIM[0]] + [float(np.sqrt(a_[-1] * b_[0])) for a_, b_ in zip(segs[:-1], segs[1:])] + [E_XLIM[1]]
    for k_, sg in enumerate(segs):
        xm = float(np.sqrt(bounds[k_] * bounds[k_ + 1]))
        nt = int(sets.loc[sg[0]])
        ax_e.text(xm, 0.025, f"{nt} targets", transform=tr_ax, ha="center", va="bottom", fontsize=ps.FS["annot"],
                  color=ps.GREY["text2"]).set_gid("target_set")
    rows_e, off_vals = [], []
    ne = len(E_EST)
    trb = matplotlib.transforms.blended_transform_factory(ax_e.transData, ax_e.transAxes)
    for j, (key, ls_, mk, _) in enumerate(E_EST):
        q = E[E.est == key].sort_values("n")
        # 가로 이동은 축 안에 그리는 3 추정기에만(가운데 = n). 축 위 화살표(혼합효과 부스팅)는 n 에 둔다.
        fac = 1.0 if mk == "up" else E_DODGE ** (j - (ne - 2) / 2)
        for _, r in q.iterrows():
            rows_e.append(dict(panel="e", est=key, variant=E_VARIANT, rule=E_RULE, n=int(r.n), x_plot=float(r.n) * fac,
                               n_targets=int(r.n_targets), targets=r.targets, delta=float(r.delta), ci_lo=float(r.ci_lo),
                               ci_hi=float(r.ci_hi), n_targets_improved=int(r.n_targets_improved), ci_kind="block_target_mean",
                               off_scale=bool(r.delta > E_YLIM[1] or r.delta < E_YLIM[0])))
        if mk == "up":                # 전 구간 축 위: 위쪽 화살표(열린 머리) + 수치
            L = OFF_ARROW_MM / 48.0
            for _, r in q.iterrows():
                x = float(r.n) * fac
                if r.delta > E_YLIM[1]:
                    _up_arrow(ax_e, x, 1.0, L, trb)
                    ax_e.text(x, 1.0 - L - 0.5 / 48.0, ps.fmt_num(r.delta, 1), transform=trb, ha="center", va="top",
                              fontsize=ps.FS["annot"], color=E_COL, zorder=5).set_gid("offscale_value")
                    off_vals.append((int(r.n), float(r.delta)))
            continue
        for sg in segs:
            s_ = q[q.n.isin(sg)]
            if not len(s_):
                continue
            x = s_.n.to_numpy(float) * fac
            ax_e.plot(x, s_.delta, color=E_COL, lw=E_LW, ls=ls_, zorder=2.8)
            for xx, a, b in zip(x, s_.ci_lo, s_.ci_hi):
                ax_e.plot([xx, xx], [a, b], color=E_COL, lw=ps.LW["ci"], solid_capstyle="round", zorder=2.9)
            ax_e.plot(x, s_.delta, ls="none", marker=mk, ms=ps.MS["main"] + (0.4 if mk == "P" else 0.0), mfc=E_COL,
                      mec=E_COL, mew=ps.LW["marker_edge"], zorder=3.2)
    _e_key(ax_e, w_mm=E_W_MM, x0=0.47)

    # ------------------------------------------------------------ 범례(그림당 1개, a–d 문법) + e 기호 설명
    handles = [ps.method_handle(m, lbl, var) for _, _, _, m, var, lbl in SERIES]
    band_rgba = matplotlib.colors.to_rgba(ps.COLOR["residual"], ps.BAND_ALPHA)
    handles += [Line2D([], [], color=ps.COLOR["ref_allA"], ls=ps.METHOD["ref_allA"]["ls"], lw=ps.METHOD["ref_allA"]["lw"],
                       label="All A-block labels (reference)"),
                Patch(facecolor=band_rgba, edgecolor="none", label="95 % CI, residual ML, k-medoid (band)"),
                Patch(facecolor="none", edgecolor=ps.COLOR["residual"], lw=0.5, ls=RAND_CI_LS,
                      label="95 % CI, residual ML, random (outline)")]
    lg = _fill_legend(fig, handles)
    ps.label_panels(axes + [ax_e], "abcde")

    # ------------------------------------------------------------ 캡션 수치(CSV 에서 읽은 값만)
    def g(est, n):
        r = E[(E.est == est) & (E.n == n)].iloc[0]
        return r.delta, r.ci_lo, r.ci_hi

    def sg(v, nd=2):                  # 양수는 '+' 를 붙인다(악화 방향을 글로도 드러냄)
        return ("+" if v > 0 else "") + ps.fmt_num(v, nd)

    def fci(v):
        return f"{sg(v[0])} [{ps.fmt_num(v[1], 2)}, {ps.fmt_num(v[2], 2)}]"

    k3, k10, m3 = g("shrink_k10", 3), g("shrink_k10", 10), g("offset_mle", 3)
    p3 = g("ppi", 3)
    mb = E[E.est == "mixed_boost"].delta
    best = E[E.est != "mixed_boost"].loc[lambda d: d.groupby("n").delta.idxmin()].est
    k_best_all = bool((best == "shrink_k10").all())
    lena_rand = h25_series(C, "Lena", "S3", "random", 0.25, 1.0)
    lena_km = h25_series(C, "Lena", "S3", "kmedoid", 0.25, 1.0)
    lr = lena_rand[lena_rand.n >= 20].blk_d_phys
    lk = lena_km[lena_km.n.between(10, 80)].blk_d_phys
    # κ=10 우위의 범위 한정: 모든 검사 n 을 훑어 κ=10 CI 가 오프셋 MLE·PPI++ CI 와 겹치기 시작하는 n(이후 전 n 겹침)
    P = E.pivot_table(index="n", columns="est", values=["delta", "ci_lo", "ci_hi"])
    allns = sorted(int(v) for v in P.index)
    others = [k for k, *_ in E_EST if k not in ("shrink_k10", "mixed_boost")]
    ovl = {n_: bool(all(P.loc[n_, ("ci_hi", "shrink_k10")] >= P.loc[n_, ("ci_lo", o)] for o in others)) for n_ in allns}
    onset = next((n_ for n_ in allns if all(ovl[m_] for m_ in allns if m_ >= n_)), None)
    no_ovl = [n_ for n_ in allns if not ovl[n_]]
    marg = pd.Series({n_: min(P.loc[n_, ("delta", o)] for o in others) - P.loc[n_, ("delta", "shrink_k10")]
                      for n_ in allns if onset is not None and n_ >= onset})
    src_ad = C.attrs.get("source", "data/processed/h3/h25b_curve.csv")
    src_e = E.attrs.get("source", "data/processed/h4/b2_regional.csv")
    caption = dict(
        definition=("Fig. 3 | Label-budget staircase and partial pooling of the scaling coefficient E. ΔRMSE, method minus "
                    "physics-anchor RMSE (E0 from source regions, n = 0) on held-out B blocks, cm; negative is better (0 line, physics anchor). "
                    "E shrinkage: target least-squares E shrunk toward E0 with κ = 10; augmentation and residual ML use this E as anchor. "
                    "All A-block labels is a reference, not an upper bound. F5 not supported: no adaptive pooling estimator improves on fixed κ = 10."),
        panels=("a–d, Main regions, |log(E_own/E0)|; header fill, error type (none, structure < 0.15; grey, "
                "level ≥ 0.20; hatched, intermediate). y symmetric-log, linear within ±2 cm. "
                "Right whiskers, reference split range; T-ticks, largest tested n; grey shading in axes, n above "
                "label pool. Residual ML CIs: filled band, k-medoid; dashed outline, random. "
                "In Lena, residual ML improves at n ≥ 20 with random labels "
                f"({ps.fmt_num(lr.min(), 2)} to {ps.fmt_num(lr.max(), 2)} cm) but worsens at n = 10–80 with k-medoid "
                f"({sg(lk.min(), 1)} to {sg(lk.max(), 1)} cm). "
                "e, Target-stratified mean of four E estimators (anchor only, random labels), broken at target-set changes. "
                f"κ = 10: {fci(k3)} cm at n = 3, {fci(k10)} cm at n = 10"
                + (", lowest at every n" if k_best_all else "")
                + (f", but CIs overlap from n = {onset} (margin {ps.fmt_num(marg.min(), 2)} cm at "
                   f"n = {int(marg.idxmin())})" if onset is not None else "")
                + f". Offset MLE, {fci(m3)}; PPI++, {fci(p3)} cm at n = 3. "
                f"Arrows, mixed-effects boosting ({sg(mb.min(), 1)} to {sg(mb.max(), 1)} cm, off scale)."),
        statistics=("a–d: means over 3 splits × 20 label draws (E shrinkage) or 3 splits × 5 draws × 2 seeds (augmentation, residual ML); "
                    "e: 3 splits × 10 draws per target, averaged over targets at n = 3, 5, 10 (14, CA-1 excluded), 20–160 (12) "
                    "and 320 (9). All intervals are 95 % block-bootstrap CIs (1,000 resamples of scoring blocks "
                    "within each split, paired across methods and repeats); in e, averaged index-wise across targets."),
        data=f"Source data: {src_ad} (a–d); {src_e} (e; vs = phys, region_set = ALL, variant = E, rule = random).",
    )
    spec = dict(
        intent=("라벨 예산 계단(H25b 주 4지역, 블록 CI 띠, S3 무작위 추가로 레나 규칙 차이 표시)과 B2 부분 풀링 "
                "14 대상 평균 Δ 대 n(추정기 4종, 앵커만, 무작위 라벨, 블록 CI)."),
        claim=("H25 이득은 오차 유형·라벨 배치 규칙에 따라 다르게 변한다. 고정 κ=10 수축이 관측 범위 전 n 에서 14(12·9) 대상 평균 "
               "최선이고, 오프셋 MLE·PPI++·혼합효과 부스팅은 이를 넘지 못한다(F5 기각)."),
        panels=5, assets=[src_ad, src_e], ci_kinds=dict(a_d="block", e="block (target-stratified mean)"),
        layout_mm=dict(row1="axes y 73–113; slots a 14+36, b–d 4+36, right 10 (reference whisker margin); "
                            "header band y 113.6–116.6 over each axes box (error type fill + region label + class text)",
                       row2=(f"axes y 11–59; e 14+{E_W_MM:g} (x 14–{14 + E_W_MM:g}); legend x {LEG_RECT[0]:g}–180 y 11–59 "
                             f"(single legend, 8 items, labelspacing tuned to ~{LEG_FILL_MM:g} mm tall, vertically centred); "
                             "e estimator key inside e (upper middle, 4 rows; κ=10 entry named as in the a–d legend)")),
        error_type_encoding=("머리 띠 채움: 구조 = 없음, 수준 = #f2f2f2, 중간 = 회색 빗금(Fig 4 HATCH_INTER) + 우측 범주 글자. "
                             "축 배경 전체 채색은 n > 라벨 풀 음영(#f2f2f2)과 겹쳐 쓰지 않는다"),
        ci_ad=dict(k_medoid="잔차 ML 채운 띠(alpha 0.18)", random="잔차 ML 파선 테두리 (0,(2,1.5)) 0.5 pt, 채움 없음",
                   legend="두 항목으로 구분"),
        panel_e=dict(filter="vs=phys, region_set=ALL, variant=E, rule=random, scope=n", ylim=list(E_YLIM),
                     target_sets={str(int(s[0])) + "-" + str(int(s[-1])): int(sets.loc[s[0]]) for s in segs},
                     offscale_mixed_boost=[f"n={n}:{v:.2f}" for n, v in off_vals], k10_best_every_n=k_best_all,
                     k10_ci_overlap=dict(onset_n=onset, no_overlap_n=no_ovl, overlap_by_n={str(k): v for k, v in ovl.items()},
                                         vs=others, min_margin_cm=round(float(marg.min()), 4), at_n=int(marg.idxmin()),
                                         rule="κ=10 ci_hi ≥ 비교 추정기 ci_lo, 모든 검사 n 스캔(onset = 이후 전 n 겹침)"),
                     encoding=("METHOD refit 파랑 단일색 + 선종·마커(κ10 파선 o, 오프셋 MLE 일점쇄선 (0,(4,2,1,2)) s = Fig 4 선종, "
                               "PPI++ 점선 P, 혼합효과 부스팅 = 축 위 열린 머리 ↑ + 수치). ▲ 는 최소 n 미달성 전용이라 쓰지 않음"),
                     key="e 축 안 우상단 4줄 표지(그림 범례와 별개, 범례 항목 상한 8 은 그림 범례에 적용)"),
        end_tick="검사 n 이 끝나는 패널(c·d)에서 곡선 끝점 무리 아래 1 mm, 3 mm 길이 0.8 pt + 먼 끝 1.2 mm 가로 캡(T 모양; 공간 부족 시 위)",
        ref_allA=("분할 평균 파선 + 분할 범위 수염(축 상자 우측 바깥 1.5 mm, 분할별 값 = h25b_curve d_phys_splits JSON): "
                  + "; ".join(f"{k} {', '.join(f'{v:.2f}' for v in vv)}" for k, vv in whisk.items())),
    )
    qa = save_paper(fig, "Fig3_label_budget", caption=caption, spec=spec, draft=False,
                    sources={"a-d": pd.DataFrame(rows_ad), "e": pd.DataFrame(rows_e)})
    if QA_OPTS.get("sheets"):
        _qa_crops("Fig3_label_budget")
    return qa


def _qa_crops(name: str, W: float = 180.0, H: float = 118.0):
    """인쇄 크기(180 mm 폭, 300 dpi) 확인 잘라내기: a–b 영역, e + 공유 범례 영역. 캔버스 = 설계 mm(bbox tight 미사용)."""
    from PIL import Image
    src = QA_DIR / f"{name}_100pct.png"
    im = Image.open(src)
    px = im.size[0] / W
    boxes = dict(ab=(0, 60, 90, 118), e=(0, 0, 180, 64))      # (x0, y0, x1, y1) mm, 좌하 원점
    for k, (x0, y0, x1, y1) in boxes.items():
        c = im.crop((round(x0 * px), round((H - y1) * px), round(x1 * px), round((H - y0) * px)))
        c.save(QA_DIR / f"{name}_crop_{k}_300dpi.png", dpi=(300, 300))
