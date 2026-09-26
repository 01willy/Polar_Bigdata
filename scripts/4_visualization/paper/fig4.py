"""Fig 4. 교락 제거 최소 라벨 수(A2). 정본 스펙: figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.4·§1.6·§5,
확정 사실 목록(2026-09-26 20:45)의 F1–F3 과 단일 분류(오라클 |log(E_own/E0)|, h3/h25b_targets.csv 최소제곱 E).

패널
  a–d  대표 4대상 n 곡선: a Russia W(수준, n_max 10), b Lena(구조), c Canada(구조, n = 0 효과), d AL-2(구조, 미달성).
       라벨 배치 all_blocks, 잔차 ML λ 0.25. E0 고정(실선, 블록 CI 띠), κ=10 수축(파선), 오프셋 MLE(일점쇄선), 모두 residual 보라 D.
       E_own 고정 + 잔차 = 검정 점선(참조, 마커 없음). x 축 왼쪽 '0' 자리(축 끊김) = n = 0(원천 잔차만, 세 E 처리 공통) E0 고정 점.
       n별 채움 = 최소 n 세 조건 충족, 빈 = 불충족(이 그림의 유일한 빈 마커 의미). 음영 = n > n_max(최대 검사 n).
       κ 수축·오프셋 MLE 표지 = E0 고정의 80 %. E_own 점선이 표지에 가리고 음영 자리가 있으면(a 만) n = 3·n_max 값 표.
       헤더 |log E비| = 유효숫자 2자리. y 축 a–d 공유(캡션 명시).
  블록 CI 표시 규칙(Fig 1·2·Table 1 과 같은 문턱): 분할 합계 채점 블록 < 8 인 대상은 CI 를 그리지 않는다.
       어느 분할이든 채점 블록 < 5(h4_common ci_flag 'blocks<5')인 대상은 이름 뒤 † 를 붙이고 캡션에 블록 수를 밝힌다.
  e    14 대상 n*(E0 고정 + 잔차, 블록 CI 주 정의) 대 오라클 |log(E_own/E0)|(제곱근 축, 범주 경계 0.15·0.20).
       위 영역 = 달성 n*(로그), 축 끊김 아래 'not reached' 띠 = 미달성 ▲ + '> n_max'(대입값 금지).
       n = 0(원천 잔차만)에서 이미 세 조건을 충족한 대상은 회색 고리로 둘러 n* = 3 이 대상 적응이 아님을 표시한다.
  f    라벨 배치 효과 Δ(all_blocks) − Δ(concentrated), E 처리 4종 × n = 10(실선 CI)·40(파선 CI).
       큰 마커 = 대상 평균(블록 CI), 회색 점 = 개별 대상. 행 묶음: E 고정(E0·E_own) 대 라벨로 E 추정(κ 수축·오프셋 MLE).

자료: h4/a2_curve.csv, a2_minn.csv, a2_spread.csv(본 실행, 블록 CI), |log(E_own/E0)|: h3/h25b_targets.csv.

검토 전용 환경 변수(논문 산출물·CAPTIONS·figure_spec 을 건드리지 않고 scratchpad 로만 저장)
  FIG4_A2_DIR=<dir>   : <dir>/a2_{curve,minn,spread}.csv 를 쓴다
  FIG4_STRESS_E=1     : e 패널을 h3/h27b_targets(S3 λ 0.25 k-중심, CI 규칙)로 채워 라벨 배치를 시험한다
  FIG4_OUT=<dir>      : 위 두 모드의 저장 위치(기본 = 현재 디렉터리)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D

from _common import (ps, H3, H4, ROOT, TARGETS14, MissingData, abslogE_map, order_by_abslogE,  # noqa: F401
                     load_a2_curve, load_a2_minn, load_a2_spread, minn_pass, save_paper, qa_check, _std_cols)

NAME = "Fig4_min_labels"
PANEL_TARGETS = ["Russia_W", "Lena", "Canada", "AL-2"]      # a = 수준 오차; b–d = 구조 오차(사실 목록 단일 분류)
E_SERIES = [                                                # (e_treat, 선종 변형, x 이동 배율, 범례 이름)
    ("E0_fixed", None, 1.0, "E$_0$ fixed + residual ML"),
    ("shrink_k10", "dashed", 1 / 1.18, "Shrinkage (κ = 10) + residual ML"),
    ("offset_mle", "dashdot", 1.18, "Offset MLE + residual ML"),
]
LAM, SPREAD, STAGE = 0.25, "all_blocks", "resid"
N_F = (10, 40)                                              # f 패널 n(스펙)
F_ROWS = [("E0_fixed", "E$_0$ fixed"), ("E_own_fixed", "E$_\\mathrm{own}$ known"),
          ("shrink_k10", "Shrinkage (κ = 10)"), ("offset_mle", "Offset MLE")]
F_XLIM = (-10, 15)                                          # f symlog x(선형 ±1 cm)
YLIM = (-14, 8)                                             # symlog Δ 축(Russia W −11.8, Lena 오프셋 MLE +6.8 포함)
XLIM_N = (1.12, 400)
N0_X = 1.55                                                 # n = 0 자리(로그 축 위 가짜 좌표, 눈금 글자 "0")
XBREAK = 2.15                                               # n = 0 자리와 n = 3 사이 축 끊김
YLIM_E = (2.4, 420)                                         # e 위 영역 n* 로그 축
RING_MS = 5.6                                               # e: n = 0 충족 고리 지름(pt)
XLIM_E = (0.0, 0.56)                                        # e 패널 |log E비| 제곱근 축
STRUCT_MAX, LEVEL_MIN = 0.15, 0.20                          # 감사 조치 9 세 범주(Fig 3·Table 1 모듈과 같은 값)
# 오프셋 MLE 일점쇄선: 기본 '-.'(6.4, 1.6, 1, 1.6)은 짧은 곡선(AL-3 n 3–20, 약 8 mm)과 범례 견본에서 실선과 구분되지 않는다.
# 주기 9 pt(약 3.2 mm, lw 1.0 기준)로 범례 견본(약 6 mm)에 두 주기, 가장 짧은 곡선에 두 주기 이상이 들어가게 한다.
LS_OFFSET = (0, (4.0, 2.0, 1.0, 2.0))
HATCH_INTER = dict(facecolor="none", edgecolor="#c8c8c8", hatch="//////", lw=0)   # 중간 범주 빗금(e 영역·f 행 공통)
PEND_COL = ps.GREY["text2"]                                 # 'pending'·'not reached' 등 부가 글자색 하나로 통일(구분은 글로)
EOWN_STYLE = dict(color="#000000", ls=(0, (1.0, 1.2)), lw=0.9)   # E_own 고정 참조(검정 점선, ref_allA 와 구분; 표지 위에 그려도 보이게 0.9 pt)
MIN_BLK_CI = 8                                              # 분할 합계 채점 블록 < 8 이면 CI 생략(Fig 1·2·Table 1 규칙)
FLAG_BLK = 5                                                # 분할 채점 블록 < 5(ci_flag) 이면 † 표기
DAGGER = "\u2020"
MS_SEC = 0.8 * ps.MS["main"]                                # 라벨로 E 를 추정하는 두 계열(κ 수축·오프셋 MLE) 표지 크기(E0 고정 대비 80 %)
VT_ROWS = [("shrink_k10", "κ = 10"), ("offset_mle", "Offset"), ("E_own_fixed", "E$_\\mathrm{own}$")]   # a: 값 표(n = 3·n_max)


def err_class(v) -> str:
    """|log(E_own/E0)| → 'structure'(< 0.15) · 'intermediate'(0.15–0.20) · 'level'(≥ 0.20) · 'unknown'."""
    if v is None or not np.isfinite(v):
        return "unknown"
    return "structure" if v < STRUCT_MAX else ("level" if v >= LEVEL_MIN else "intermediate")


def _series_style(var):
    """잔차 ML 계열 plot kwargs(오프셋 MLE 는 LS_OFFSET)."""
    st = ps.style_of("residual", None if var == "dashdot" else var)
    if var == "dashdot":
        st["ls"] = LS_OFFSET
    return st


# ---------------------------------------------------------------- 자료
def _preview_frames():
    """검토 전용: 체크포인트 집계 사본 + 스모크 자료."""
    d = Path(os.environ["FIG4_A2_DIR"])
    out = {}
    for kind in ("curve", "minn", "spread"):
        parts = [pd.read_csv(d / f"a2_{kind}.csv")]
        sm = H4 / f"a2_smoke_{kind}.csv"
        if sm.exists():
            s = pd.read_csv(sm)
            parts.append(s[~s.target.isin(parts[0].target.unique())])
        df = pd.concat(parts, ignore_index=True)
        df.attrs.update(source=f"preview:{d.name}/a2_{kind}.csv", is_smoke=True)
        out[kind] = df
    C = _std_cols(out["curve"], "d_phys_mean", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep")
    M = out["minn"]
    M["censored"] = M.censored.astype(str).str.lower().isin(["true", "1"])
    M.loc[M.censored, "n_star"] = np.nan
    M["range_limited"] = M.censored & (M.n_max < 40)
    S = out["spread"]; S["ci_kind"] = "block"
    return C, M, S


def oracle_abslogE() -> dict:
    """대상 → 오라클 |log(E_own/E0)|(사실 목록 단일 정의: h3/h25b_targets.csv 최소제곱 E_own, 분할 평균 E0)."""
    T = pd.read_csv(H3 / "h25b_targets.csv")
    g = T.groupby("target").agg(E_own=("E_own", "first"), E0=("E0", "mean"))
    return dict(zip(g.index, np.abs(np.log(g.E_own / g.E0))))


def load_data(draft: bool):
    if os.environ.get("FIG4_A2_DIR"):
        C, M, S = _preview_frames()
    else:
        C, M, S = load_a2_curve(draft), load_a2_minn(draft), load_a2_spread(draft)
    m = oracle_abslogE()
    # a–d: all_blocks · scope n · resid λ 0.25, + n = 0(scope n0, spread none) E0 고정 점
    cq = C[(C.spread == SPREAD) & (C.scope == "n") & (C.stage == STAGE) & np.isclose(C.lam, LAM)].copy()
    cq = cq[cq.n_splits == cq.groupby(["target", "e_treat"]).n_splits.transform("max")]
    cq["pass3"] = minn_pass(cq)
    cq["abs_logE"] = cq.target.map(m)
    c0 = C[(C.scope == "n0") & (C.stage == STAGE) & np.isclose(C.lam, LAM) & (C.e_treat == "E0_fixed")].copy()
    c0["pass3"] = minn_pass(c0)
    c0["abs_logE"] = c0.target.map(m)
    # e: E0 고정 · all_blocks · resid λ 0.25 · 14 대상. pass0 = n = 0 에서 이미 세 조건 충족
    mq = M[(M.e_treat == "E0_fixed") & (M.spread == SPREAD) & (M.stage == STAGE) & np.isclose(M.lam, LAM)
           & M.target.isin(TARGETS14)].copy()
    if os.environ.get("FIG4_STRESS_E"):
        mq = _stress_e()
    mq["abs_logE"] = mq.target.map(m)
    mq["pass0"] = mq.target.map(dict(zip(c0.target, c0.pass3))).fillna(False).astype(bool)
    # f: all_blocks − concentrated · E 처리 4종 · resid λ 0.25 · n = 10·40 (14 대상 + 대상 평균)
    sq = S[(S.contrast == f"{SPREAD}-concentrated") & S.e_treat.isin([e for e, _ in F_ROWS]) & (S.stage == STAGE)
           & np.isclose(S.lam, LAM) & S.target.isin(TARGETS14 + ["MEAN_ANALYSIS"])].copy()
    n_f = tuple(n for n in N_F if n in set(sq.n))
    sq = sq[sq.n.isin(n_f)]
    sq["abs_logE"] = sq.target.map(m)
    cq = pd.concat([cq, c0.assign(n=0)], ignore_index=True)
    src = dict(curve=C.attrs.get("source", ""), minn=M.attrs.get("source", ""), spread=S.attrs.get("source", ""))
    blk = block_info(C, src)
    return cq, mq, sq, n_f, m, src, blk


def block_info(C, src) -> pd.DataFrame:
    """대상별 채점 블록 수: 분할 합계(tot)·분할 최솟값(min). 곡선과 같은 출처의 *_targets.csv(분할별 n_blocks_eval)를 읽고,
    없으면 곡선의 n_blocks_eval_min 만 쓴다. show_ci = tot ≥ 8, dagger = min < 5 또는 곡선 ci_flag 'blocks<5'."""
    cur = str(src.get("curve", ""))
    p = (Path(os.environ.get("FIG4_A2_DIR", ".")) / "a2_targets.csv") if cur.startswith("preview:") else (ROOT / cur)
    p = p.with_name(p.name.replace("_curve.csv", "_targets.csv"))
    rows = {}
    if p.exists():
        T = pd.read_csv(p).drop_duplicates(["target", "split"])
        if os.environ.get("FIG4_A2_DIR") and (H4 / "a2_smoke_targets.csv").exists():   # 미리보기: 스모크 대상 보충
            sm = pd.read_csv(H4 / "a2_smoke_targets.csv")
            T = pd.concat([T, sm[~sm.target.isin(T.target)]], ignore_index=True).drop_duplicates(["target", "split"])
        g = T.groupby("target").n_blocks_eval
        rows = {t: dict(tot=float(g.sum()[t]), min=float(g.min()[t]), n_splits=int(g.size()[t])) for t in g.groups}
    fl = C[C.get("ci_flag", pd.Series("", index=C.index)).astype(str).str.startswith("blocks<")]
    for t in C.target.unique():
        r = rows.setdefault(t, dict(tot=np.nan, min=np.nan, n_splits=np.nan))
        if not np.isfinite(r["min"]) and "n_blocks_eval_min" in C:
            r["min"] = float(C.loc[C.target == t, "n_blocks_eval_min"].min())
        r["flag"] = bool((t in set(fl.target)) or (np.isfinite(r["min"]) and r["min"] < FLAG_BLK))
        r["show_ci"] = bool(not np.isfinite(r["tot"]) or r["tot"] >= MIN_BLK_CI)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "target"
    for c in ("flag", "show_ci"):
        out[c] = out[c].fillna(False if c == "flag" else True).astype(bool)
    return out


def fmt_sig2(v) -> str:
    """|log E비| 표기 규칙 하나: 유효숫자 2자리(0.51, 0.12, 0.0035, 0.059). 끝 0 유지('#')."""
    s = f"{v:#.2g}"
    return s[:-1] if s.endswith(".") else s


def tname(t, blk, abslogE=None, with_e=False):
    """대상 표기: 이름(+ |log E비|, 소수 둘째 자리: Fig 2·3·Table 1 과 같은 규칙) + 채점 블록 < 5 분할이 있으면 †."""
    s = ps.region_name(t)
    if with_e and abslogE is not None and np.isfinite(abslogE):
        s = f"{s} ({abslogE:.2f})"
    if blk is not None and t in blk.index and bool(blk.loc[t, "flag"]):
        s = s.replace(" (", DAGGER + " (", 1) if " (" in s else s + DAGGER
    return s


def _stress_e():
    """검토 전용: e 패널 라벨 배치 시험(h27b CI 규칙 결과를 대용 자료로)."""
    T = pd.read_csv(H3 / "h27b_targets.csv")
    T = T[(T.stage == "S3") & np.isclose(T.lam, 0.25) & (T.rule == "kmedoid") & (T.definition == "ci")].copy()
    T["censored"] = T.censored.astype(str).str.lower().isin(["true", "1"])
    T["n_star"] = np.where(T.censored, np.nan, T.be_n)
    T.loc[T.target == "Russia_E", "n_max"] = 10
    T["range_limited"] = T.censored & (T.n_max < 40)
    return T[["target", "n_star", "censored", "n_max", "range_limited"]]


# ---------------------------------------------------------------- 라벨 배치(렌더러 기반 탐욕 배치)
def _overlap(a, b, pad=0.8):
    return a.x0 - pad < b.x1 and b.x0 - pad < a.x1 and a.y0 - pad < b.y1 and b.y0 - pad < a.y1


def _seg_hits(p0, p1, boxes, pad=1.0):
    """선분 p0→p1(픽셀)이 상자 중 하나를 지나는지(표본 24점)."""
    for u in np.linspace(0.08, 0.92, 24):
        x, y = p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1])
        if any(b.x0 - pad < x < b.x1 + pad and b.y0 - pad < y < b.y1 + pad for b in boxes):
            return True
    return False


def _segs_cross(p, q, r_, s_):
    """두 선분 p–q, r–s 의 교차(끝점 공유 제외)."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return orient(p, q, r_) * orient(p, q, s_) < 0 and orient(r_, s_, p) * orient(r_, s_, q) < 0


def _plan_labels(ax, anchors, texts, cands, order, obstacles=(), pad_pt=2.0, dy_direct=5.0, marker_r_pt=2.4):
    """한 순서(order)로 탐욕 배치 계획을 세운다(그리지 않음). 반환 (계획 목록, 대체 배치 수, 총비용).
    조건: 축 안, 다른 라벨·표지·장애물과 비겹침, 지시선이 다른 라벨·표지를 지나지 않고 기존 지시선이 새 라벨을 지나지 않음."""
    fig = ax.figure; r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    pad = pad_pt * ppt
    inv = fig.transFigure.inverted()
    mk = [matplotlib.transforms.Bbox([[x - marker_r_pt * ppt, y - marker_r_pt * ppt], [x + marker_r_pt * ppt, y + marker_r_pt * ppt]])
          for x, y in anchors]
    obst = mk + list(obstacles)
    axbb = ax.bbox
    placed, leaders, plan = [], [], [None] * len(anchors)
    nfb, tot = 0, 0.0
    probe = fig.text(0, 0, "", fontsize=ps.FS["annot"])
    for i in order:
        ax_, ay_ = anchors[i]
        best = None
        for dx, dy, ha, va, cost in cands:
            fx, fy = inv.transform((ax_ + dx * ppt, ay_ + dy * ppt))
            probe.set_text(texts[i]); probe.set_position((fx, fy)); probe.set_ha(ha); probe.set_va(va)
            bb = probe.get_window_extent(r)
            if not (bb.x0 >= axbb.x0 + 1 and bb.x1 <= axbb.x1 - 1 and bb.y0 >= axbb.y0 + 1 and bb.y1 <= axbb.y1 - 1):
                continue
            if any(_overlap(bb, o, pad) for o in placed) or any(_overlap(bb, o, 0.5 * ppt) for j, o in enumerate(obst) if j != i):
                continue
            if any(_seg_hits(q0, q1, [bb], pad=0.8 * ppt) for q0, q1 in leaders):
                continue
            far = abs(dy) > dy_direct or (abs(dx) > 3.0 and dy != 0 and ha == "center")   # 옆 자리(ha left/right)는 지시선 없음
            seg = None
            if far:
                ex = min(max(ax_, bb.x0 + 0.5 * ppt), bb.x1 - 0.5 * ppt)
                ey = bb.y0 - 0.3 * ppt if bb.y0 > ay_ else (bb.y1 + 0.3 * ppt if bb.y1 < ay_ else (bb.y0 + bb.y1) / 2)
                p0 = np.array([ax_, ay_]); p1 = np.array([ex, ey]); v = p1 - p0; L = np.hypot(*v)
                p0 = p0 + v / L * marker_r_pt * 0.9 * ppt if L > 0 else p0
                if _seg_hits(tuple(p0), tuple(p1), placed + [o for j, o in enumerate(mk) if j != i]):
                    continue
                if any(_segs_cross(tuple(p0), tuple(p1), q0, q1) for q0, q1 in leaders):
                    continue
                seg = (tuple(p0), tuple(p1))
            best = (fx, fy, ha, va, bb, seg, cost)
            break
        if best is None:                                          # 자리 없음: 오른쪽 제자리(QA 가 겹침으로 잡는다)
            fx, fy = inv.transform((ax_ + 4 * ppt, ay_))
            probe.set_text(texts[i]); probe.set_position((fx, fy)); probe.set_ha("left"); probe.set_va("center")
            best = (fx, fy, "left", "center", probe.get_window_extent(r), None, 1e3)
            nfb += 1
        plan[i] = best; placed.append(best[4]); tot += best[6]
        if best[5] is not None:
            leaders.append(best[5])
    probe.remove()
    return plan, nfb, tot


def _greedy_labels(ax, anchors, texts, colors, cands, obstacles=(), pad_pt=2.0, dy_direct=5.0, marker_r_pt=2.4):
    """렌더러 기반 탐욕 라벨 배치. 순서 3가지(x 오름차순, 내림차순, 이웃 밀집 우선)로 계획을 세워
    대체 배치가 가장 적고 총비용이 가장 작은 계획을 그린다. 떨어진 라벨은 0.3 pt 회색 지시선으로 표지에 잇는다."""
    fig = ax.figure; fig.canvas.draw()
    cands = sorted(cands, key=lambda c: c[4])
    xs = np.array([a_[0] for a_ in anchors])
    crowd = np.array([np.sum(np.abs(xs - x) < 40 * fig.dpi / 72) for x in xs])
    orders = [np.argsort(xs), np.argsort(-xs), np.lexsort((xs, -crowd))]
    best = None
    for od in orders:
        plan, nfb, tot = _plan_labels(ax, anchors, texts, cands, od, obstacles, pad_pt, dy_direct, marker_r_pt)
        if best is None or (nfb, tot) < (best[1], best[2]):
            best = (plan, nfb, tot)
    inv = fig.transFigure.inverted()
    arts = []
    for i, (fx, fy, ha, va, bb, seg, _) in enumerate(best[0]):
        t = fig.text(fx, fy, texts[i], ha=ha, va=va, fontsize=ps.FS["annot"], color=colors[i], zorder=5,
                     bbox=dict(boxstyle="square,pad=0.15", fc=ax.get_facecolor(), ec="none"))   # 빗금(중간 범주) 위에서도 글자가 깨끗하게
        t.set_gid("point_label"); arts.append(t)
        if seg is not None:
            (a0, a1), (b0, b1) = inv.transform(seg[0]), inv.transform(seg[1])
            fig.add_artist(Line2D([a0, b0], [a1, b1], transform=fig.transFigure, lw=0.3, color=ps.GREY["mid"], zorder=4))
    return arts


def _shift_cands(tiers, step=1.5, mx=60.0, base=0.0, per_pt=0.25):
    """층(dy_pt, va, 층 비용)마다 가운데 정렬 좌우 이동 후보."""
    out = []
    for dy, va, tc in tiers:
        s = 0.0
        while s <= mx:
            for sg in ((1,) if s == 0 else (1, -1)):
                out.append((sg * s, dy, "center", va, base + tc + per_pt * s))
            s += step
    return out


def place_point_labels(ax, xs, ys, texts, color="#000000"):
    """e 위 영역: 달성 점 라벨. 규칙 하나: 점 위 가운데 정렬. 겹치면 좌우로 밀고, 그래도 안 되면 위층으로 올리며
    (둘 다 0.3 pt 지시선), 위쪽 자리가 없을 때만 아래층을 쓴다(점이 축 위 끝에 붙은 경우)."""
    anchors = [tuple(ax.transData.transform((x, y))) for x, y in zip(xs, ys)]
    cands = _shift_cands([(4.0, "bottom", 0.0), (11.0, "bottom", 4.0), (18.0, "bottom", 8.0), (25.0, "bottom", 11.0), (32.0, "bottom", 14.0),
                          (39.0, "bottom", 17.0), (-4.0, "top", 40.0), (-11.0, "top", 44.0)], per_pt=1.0)
    # 표지 바로 오른쪽·왼쪽(같은 높이, 지시선 없음): 위 가운데 자리가 막히면 옆으로 7 pt 넘게 밀어 비스듬한 지시선을 쓰는 것보다 우선한다
    # (AL-3·CA-2 처럼 표지가 붙은 묶음에서 지시선이 이웃 표지를 가리키는 것처럼 읽히는 문제 방지). 고리 반지름 2.8 pt 바깥 5.2 pt
    # 옆 자리는 표지 중심보다 1.5 pt 위(아래 축 끝 2.4 에 붙은 n* = 3 표지에서도 글 상자가 축 안에 들게). 왼쪽 우선(Lena 가 빗금 위로 가지 않게)
    cands += [(dx, dy, ha, "center", c) for dy in (1.5, 3.0) for dx, ha, c in ((-5.2, "right", 3.9), (5.2, "left", 4.1))]
    return _greedy_labels(ax, anchors, texts, [color] * len(texts), cands)


def dodge_px(px, py, sep):
    """표지 겹침 해소(결정적): 같은 높이(|Δy| < sep)에서 x 간격이 sep 보다 좁은 이웃을 묶어 묶음 평균을 중심으로 sep 간격으로 편다.
    묶음끼리 다시 닿으면 합쳐 반복한다. 입력·출력은 픽셀 좌표."""
    px = np.asarray(px, float).copy(); py = np.asarray(py, float)
    out = px.copy()
    for yv in np.unique(np.round(py / max(sep, 1e-9))):
        idx = np.where(np.round(py / max(sep, 1e-9)) == yv)[0]
        idx = idx[np.argsort(px[idx])]
        groups = [[i] for i in idx]
        for _ in range(len(idx) + 1):
            pos = {}
            for g in groups:
                c = np.mean(px[g]); k = len(g)
                for j, i in enumerate(g):
                    pos[i] = c + (j - (k - 1) / 2) * sep
            merged, changed = [groups[0]], False
            for g in groups[1:]:
                if pos[g[0]] - pos[merged[-1][-1]] < sep - 1e-6:
                    merged[-1] = merged[-1] + g; changed = True
                else:
                    merged.append(g)
            groups = merged
            if not changed:
                break
        for i in idx:
            out[i] = pos[i]
    return out


def marker_overlaps(fig, axes, frac=0.9):
    """QA 보충(_common.qa_check 는 글자 상자만 본다): 축마다 선 없는 표지 점들의 픽셀 상자(크기 × frac)가 겹치는 쌍을 센다.
    같은 Line2D 안의 점끼리도 검사한다. 반환 [(축 번호, (x0, y0), (x1, y1))]."""
    fig.canvas.draw()
    ppt = fig.dpi / 72
    bad = []
    for k, ax in enumerate(axes):
        pts = []
        for ln in ax.get_lines():
            if ln.get_gid() == "xbreak" or ln.get_marker() in (None, "", "None", " ") or ln.get_linestyle() not in ("None", "none", " ", ""):
                continue
            xy = ax.transData.transform(np.c_[ln.get_xdata(orig=False), ln.get_ydata(orig=False)]) if ln.get_transform() == ax.transData \
                else ln.get_transform().transform(np.c_[ln.get_xdata(orig=False), ln.get_ydata(orig=False)])
            r = ln.get_markersize() * ppt * frac
            for x, y in xy:
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y, r))
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (x0, y0, r0), (x1, y1, r1) = pts[i], pts[j]
                if abs(x0 - x1) < (r0 + r1) / 2 and abs(y0 - y1) < (r0 + r1) / 2:
                    bad.append((k, (round(x0), round(y0)), (round(x1), round(y1))))
    return bad


def _spread_1d(c, w, gap, lo, hi):
    """순서 보존 1차원 배치: x_{i+1} − x_i ≥ (w_i + w_{i+1})/2 + gap, lo ≤ x_i − w_i/2, x_i + w_i/2 ≤ hi 에서
    Σ(x_i − c_i)² 최소(누적 오프셋 변환 후 PAVA 등위 회귀, 범위는 평행 이동)."""
    c = np.asarray(c, float); w = np.asarray(w, float); n = len(c)
    off = np.r_[0.0, np.cumsum((w[:-1] + w[1:]) / 2 + gap)]
    z = c - off
    blocks = [[z[i], 1.0, i, i] for i in range(n)]            # (평균, 가중, 시작, 끝)
    st = []
    for bl in blocks:
        st.append(bl)
        while len(st) > 1 and st[-2][0] > st[-1][0]:
            m2, w2, s2, _ = st.pop(); m1, w1, s1, _ = st.pop()
            st.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, s1, bl[3]])
    y = np.empty(n)
    for m, _, s0, e0 in st:
        y[s0:e0 + 1] = m
    x = y + off
    if n:
        x += max(0.0, (lo + w[0] / 2) - x[0])
        x -= max(0.0, x[-1] + w[-1] / 2 - hi)
    return x


def place_band_labels(ax, xs, texts, colors, y_marker=0.86, y_label=0.30, gap_pt=4.5):
    """'not reached' 띠: 두 줄 라벨('이름' / '> n_max')을 ▲ 아래 한 줄에 x 순서대로 놓는다.
    라벨 가로 위치는 순서 보존 1차원 최소제곱(_spread_1d)으로 겹침 없이 ▲ 에 가깝게 정하고,
    ▲ 와 라벨은 꺾인 0.3 pt 회색 지시선(▲ 아래 수직 → 공통 꺾임 높이 → 라벨 위)으로 잇는다. 순서가 보존되므로 지시선은 교차하지 않는다."""
    fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    inv = fig.transFigure.inverted()
    o = np.argsort(xs)
    px = np.array([ax.transData.transform((x, 0))[0] for x in xs])
    y_lab = ax.transAxes.transform((0, y_label))[1]
    y_mk = ax.transAxes.transform((0, y_marker))[1] - 2.4 * ppt
    arts = []
    for i in o:
        fx, fy = inv.transform((px[i], y_lab))
        t = fig.text(fx, fy, texts[i], ha="center", va="center", fontsize=ps.FS["annot"], color=colors[i], linespacing=0.95, zorder=5,
                     bbox=dict(boxstyle="square,pad=0.22", fc=ps.GREY["band"], ec="none"))   # 중간 범주 빗금이 글자를 지나지 않게(불투명)
        t.set_gid("point_label"); arts.append(t)
    w = np.array([t.get_window_extent(r).width for t in arts]) + 2 * 0.22 * ps.FS["annot"] * ppt   # 배경 상자 여백 포함
    top = max(t.get_window_extent(r).y1 for t in arts) + 0.22 * ps.FS["annot"] * ppt   # 배경 상자(pad 0.22) 윗변
    newx = _spread_1d(px[o], w, gap_pt * ppt, ax.bbox.x0 + 1, ax.bbox.x1 - 1)

    for t, i, nx in zip(arts, o, newx):
        t.set_x(inv.transform((nx, y_lab))[0])
        pts = [(px[i], y_mk), (nx, top + 0.6 * ppt)]           # 직선 지시선(순서 보존이라 교차 없음; 꺾인 선은 수평 구간이 이어져 막대로 읽혔다)
        f = np.array([inv.transform(q) for q in pts])
        fig.add_artist(Line2D(f[:, 0], f[:, 1], transform=fig.transFigure, lw=0.3, color=ps.GREY["mid"], zorder=4))
    return arts


# ---------------------------------------------------------------- 패널
def _halo(lw):
    """근접 곡선 구분용 흰 테두리(선 굵기 + 1.4 pt)."""
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw + 1.4, foreground="white"), pe.Normal()]


def _n_axis(ax):
    """n 축: 로그 3–400 + 왼쪽 n = 0 자리(N0_X, 눈금 '0'), 사이에 축 끊김."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_xscale("log"); ax.set_xlim(*XLIM_N)
    ticks = [N0_X, 3, 10, 30, 100, 300]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "0" if abs(v - N0_X) < 1e-6 else f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([v for v in (4, 5, 6, 7, 8, 9, 20, 40, 50, 60, 70, 80, 90, 200) if v < XLIM_N[1]]))
    _xbreak(ax)


def _xbreak(ax):
    """아래 축선 끊김: XBREAK 에 흰 틈 + 사선 두 개('//')."""
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([XBREAK / 1.07, XBREAK * 1.07], [0, 0], transform=tr, color="white", lw=2.0, clip_on=False, zorder=6,
            solid_capstyle="butt")
    for k in (1 / 1.07, 1.07):
        ln, = ax.plot([XBREAK * k], [0], transform=tr, marker=[(-0.45, -1.0), (0.45, 1.0)], ms=4.0, mew=0.6, color="#000000",
                      ls="none", clip_on=False, zorder=7)
        ln.set_gid("xbreak")


def _curve_panel(ax, cq, t, first: bool, draft: bool = False, show_ci: bool = True):
    q_all = cq[cq.target == t]
    q = q_all[q_all.n > 0]
    q0 = q_all[q_all.n == 0]
    ps.zero_line(ax, "y", better_text=first)
    ps.symlog_delta_axis(ax, lim=YLIM)
    _n_axis(ax)
    if not len(q):                                          # 자료 없음: 빈 축이 '모든 n 에서 0' 으로 읽히지 않게 표지
        tx = ax.text(0.5, 0.78, "no data", transform=ax.transAxes, ha="center", va="center",
                     fontsize=ps.FS["annot"], color=PEND_COL, linespacing=1.0)
        tx.set_gid("pending")
        return pd.DataFrame()
    n_max = int(q.n.max())
    # 최대 검사 n(n_max) 너머 음영(경계 = n_max). n_max 는 검사 격자의 최댓값(Russia W 는 라벨 풀 16 개라 10)
    if n_max < 320:
        ax.axvspan(n_max * 1.12, XLIM_N[1], color=ps.GREY["band"], lw=0, zorder=0)
    rows = []
    ref = q[q.e_treat == "E_own_fixed"].sort_values("n")
    if len(ref):
        # 흰 테두리는 곡선 아래(2.6), 검정 점선은 표지 위(3.4): 표지 묶음에 가려도 점선이 보인다(Russia W −10.5 ~ −11.4 cm)
        ax.plot(ref.n, ref.blk_d_phys, color="white", lw=EOWN_STYLE["lw"] + 1.4, zorder=2.6, solid_capstyle="round")
        ax.plot(ref.n, ref.blk_d_phys, zorder=3.4, **EOWN_STYLE)
        rows.append(ref.assign(series="E_own_fixed", x_plot=ref.n))
    for e_treat, var, fx, _ in E_SERIES:
        s_ = q[q.e_treat == e_treat].sort_values("n")
        if not len(s_):
            continue
        x = s_.n.to_numpy(float) * fx
        y = s_.blk_d_phys.to_numpy(float)
        st = _series_style(var)
        if e_treat == "E0_fixed" and show_ci and s_.blk_d_phys_lo.notna().any():
            ax.fill_between(x, np.clip(s_.blk_d_phys_lo, *YLIM), np.clip(s_.blk_d_phys_hi, *YLIM), color=st["color"],
                            alpha=ps.BAND_ALPHA, lw=0, zorder=1.8)
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3, path_effects=_halo(st["lw"]))
        inr = (y >= YLIM[0]) & (y <= YLIM[1])
        fm = s_.pass3.to_numpy(bool)
        for filled in (True, False):
            k = inr & (fm == filled)
            ax.plot(x[k], y[k], ls="none", marker="D", ms=ps.MS["main"] if e_treat == "E0_fixed" else MS_SEC, mec=st["color"],
                    mew=ps.LW["marker_edge"], mfc=st["color"] if filled else "white", zorder=3.2)
        for xi, yi in zip(x[~inr], y[~inr]):
            ps.offscale_marker(ax, yi, xi, YLIM, orient="v", method="residual")
        rows.append(s_.assign(series=e_treat, x_plot=x))
    vt = None
    if len(ref) and n_max < 320 and _eown_occluded(ax, ref, rows):
        # 규칙: E_own 점선이 표지에 가리고(symlog 로그 구간 압축) n > n_max 음영에 자리가 있으면, 그 자리에
        # 압축된 세 계열(κ 수축·오프셋 MLE·E_own)의 n = 3·n_max 값 표를 적는다(이 자료에서는 a 만 해당; 캡션에 명시)
        vt = _value_table(ax, q, n_max)
    # n = 0: 원천 잔차만(대상 라벨 없음, 세 E 처리 공통 = E0 고정). 블록 CI 막대 + 채움 규칙 동일
    if len(q0):
        r0 = q0.iloc[0]; c = ps.COLOR["residual"]
        if show_ci and np.isfinite(r0.blk_d_phys_lo):
            ax.plot([N0_X, N0_X], [np.clip(r0.blk_d_phys_lo, *YLIM), np.clip(r0.blk_d_phys_hi, *YLIM)], color=c,
                    lw=ps.LW["ci"], alpha=0.55, zorder=3.0, solid_capstyle="butt")
        ax.plot([N0_X], [r0.blk_d_phys], ls="none", marker="D", ms=ps.MS["main"], mec=c, mew=ps.LW["marker_edge"],
                mfc=c if bool(r0.pass3) else "white", zorder=3.2)
        rows.append(q0.assign(series="E0_fixed_n0", x_plot=N0_X))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _value_table(ax, q, n_max):
    """n > n_max 음영 자리에 값 표: 머리 'n | 3 | n_max', 행 = VT_ROWS(블록 점 추정 Δ, cm, 소수 1자리).
    열 위치는 렌더된 글 폭으로 정한다(이름 열 왼쪽 = 음영 왼끝 + 2 pt, 값 열 오른쪽 정렬, 열 간격 3 pt, 끝 = 축 오른쪽 − 1.5 pt)."""
    n_lo = int(q.n.min())
    fig = ax.figure; ppt = fig.dpi / 72; r = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    x_sh = inv.transform(ax.transData.transform((n_max * 1.12, 0)))[0]
    w_ax = ax.bbox.width
    rows = [("n", str(n_lo), str(n_max), ps.GREY["text2"])]
    for e, nm in VT_ROWS:
        d = q[q.e_treat == e].set_index("n").blk_d_phys
        if n_lo in d.index and n_max in d.index:
            rows.append((nm, ps.fmt_num(d[n_lo], 1), ps.fmt_num(d[n_max], 1), "#000000"))
    lh = 1.25 * ps.FS["annot"] * ppt / ax.bbox.height
    y0 = 0.07
    arts = []
    for k, (nm, v1, v2, c) in enumerate(rows):
        yy = y0 + (len(rows) - 1 - k) * lh
        kw = dict(transform=ax.transAxes, fontsize=ps.FS["annot"], color=c, va="bottom", zorder=5)
        arts.append([ax.text(0, yy, nm, ha="left", **kw), ax.text(0, yy, v1, ha="right", **kw), ax.text(0, yy, v2, ha="right", **kw)])
    fig.canvas.draw()
    wid = [[a_.get_window_extent(r).width / w_ax for a_ in row] for row in arts]
    gap = 3.0 * ppt / w_ax
    x_nm = x_sh + 2.0 * ppt / w_ax
    x_v1 = x_nm + max(w[0] for w in wid) + gap + max(w[1] for w in wid)
    x_v2 = max(x_v1 + gap + max(w[2] for w in wid), 1.0 - 1.5 * ppt / w_ax)
    for row in arts:
        row[0].set_x(x_nm); row[1].set_x(x_v1); row[2].set_x(x_v2)
        for a_ in row:
            a_.set_gid("value_table")
    yl = y0 + (len(rows) - 1) * lh - 0.12 * lh                 # 머리 아래 가는 선
    ax.plot([x_nm, x_v2], [yl, yl], transform=ax.transAxes, color=ps.GREY["mid"], lw=0.3, zorder=5)
    if x_v2 > 1.0:
        print(f"[fig4] WARNING value table exceeds axis: right = {x_v2:.3f}")
    return [a_ for row in arts for a_ in row]


def _eown_occluded(ax, ref, rows, frac=1.0) -> bool:
    """E_own 점선의 점 하나라도 잔차 계열 표지(지름 × frac) 안에 들면 True(픽셀 거리)."""
    ppt = ax.figure.dpi / 72
    r_mk = ps.MS["main"] * ppt * frac
    a = ax.transData.transform(np.c_[ref.n.to_numpy(float), ref.blk_d_phys.to_numpy(float)])
    pts = [ax.transData.transform(np.c_[d.x_plot.to_numpy(float), d.blk_d_phys.to_numpy(float)]) for d in rows if d.series.iloc[0] != "E_own_fixed"]
    if not pts:
        return False
    b = np.vstack(pts)
    hit = [np.min(np.hypot(*(b - p).T)) < r_mk for p in a]
    return bool(np.any(hit))


def _cond_label(ax, t, m, blk=None):
    v = m.get(t, np.nan)
    ax.text(0.03, 1.035, tname(t, blk, v, with_e=True), transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["tick"])


def _e_panel(ax_up, ax_band, mq, fig, blk=None):
    for a in (ax_up, ax_band):                            # 제곱근 x 축: 구조(0.003–0.13)·수준(0.17–0.51) 대상에 폭을 고르게
        a.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: np.square(v)))
        a.set_xlim(*XLIM_E)
    for a in (ax_up, ax_band):                            # 중간 범주 영역(빗금) + 두 경계 점선
        a.axvspan(STRUCT_MAX, LEVEL_MIN, zorder=0.4, **HATCH_INTER)
        for v in (STRUCT_MAX, LEVEL_MIN):
            a.axvline(v, color=ps.GREY["light"], lw=0.5, ls=(0, (1.5, 1.5)), zorder=0.5)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax_band.xaxis.set_major_locator(FixedLocator([0, 0.01, 0.05, 0.1, STRUCT_MAX, LEVEL_MIN, 0.3, 0.4, 0.5]))   # 범주 경계 0.15·0.20 표기
    ax_band.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax_band.xaxis.set_minor_locator(NullLocator())
    ax_band.set_xlabel("|log(E$_\\mathrm{own}$/E$_0$)|")
    # 위 영역: 달성 n*
    ps.log_n_axis(ax_up, lim=YLIM_E, axis="y")
    ax_up.set_ylabel("n* (E$_0$ fixed + residual ML)")
    ax_up.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_up.spines["bottom"].set_visible(False)
    tr = matplotlib.transforms.blended_transform_factory(ax_up.transData, ax_up.transAxes)
    # 범주 이름 한 줄: 'intermediate' 는 중간 영역 가운데, 'structure'·'level' 은 그 양옆(글자 폭 기준 간격 3 pt)
    xc = ((np.sqrt(STRUCT_MAX) + np.sqrt(LEVEL_MIN)) / 2) ** 2
    kw = dict(fontsize=ps.FS["annot"], color=ps.GREY["text2"], va="bottom", xycoords=tr, textcoords="offset points")
    t_mid = ax_up.annotate("intermediate", (xc, 1.01), xytext=(0, 0), ha="center", **kw)
    fig.canvas.draw()
    half = t_mid.get_window_extent(fig.canvas.get_renderer()).width / 2 / (fig.dpi / 72)
    ax_up.annotate("structure", (xc, 1.01), xytext=(-(half + 3.0), 0), ha="right", **kw)
    ax_up.annotate("level", (xc, 1.01), xytext=(half + 3.0, 0), ha="left", **kw)
    reached = mq[~mq.censored & mq.abs_logE.notna()].assign(error_type=lambda d: d.abs_logE.map(err_class))
    col = ps.COLOR["residual"]
    ppt = fig.dpi / 72
    sep = (ps.MS["main"] + 1.4) * ppt                           # 표지 중심 간격 하한(표지 3 pt + 여백 1.4 pt)
    if len(reached):
        fig.canvas.draw()
        xy = ax_up.transData.transform(np.c_[reached.abs_logE.to_numpy(float), reached.n_star.to_numpy(float)])
        xd = ax_up.transData.inverted().transform(np.c_[dodge_px(xy[:, 0], xy[:, 1], sep), xy[:, 1]])[:, 0]
        reached = reached.assign(x_plot=xd)
        ax_up.plot(reached.x_plot, reached.n_star, ls="none", marker="D", ms=ps.MS["main"], mfc=col, mec=col, mew=ps.LW["marker_edge"], zorder=3)
        r0 = reached[reached.pass0]                             # n = 0 에서 이미 충족: 회색 고리(scatter; 표지 겹침 검사 대상 아님)
        if len(r0):
            ax_up.scatter(r0.x_plot, r0.n_star, s=RING_MS ** 2, facecolors="none", edgecolors=ps.GREY["mid"], linewidths=0.6, zorder=2.9)
        place_point_labels(ax_up, reached.x_plot.to_numpy(float), reached.n_star.to_numpy(float),
                           [tname(t, blk) for t in reached.target])
    # 아래 띠: 미달성
    ax_band.set_facecolor(ps.GREY["band"]); ax_band.set_ylim(0, 1)
    ax_band.set_yticks([0.5]); ax_band.set_yticklabels(["not\nreached"], fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=0.95)
    ax_band.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right", "left"):
        ax_band.spines[sp].set_visible(False)
    cens = (mq[mq.censored & mq.abs_logE.notna()].sort_values("abs_logE").reset_index(drop=True)
            .assign(error_type=lambda d: d.abs_logE.map(err_class)))
    if len(cens):
        xs = cens.abs_logE.to_numpy(float).copy()
        # 가까운 ▲ 는 묶음 중심 기준으로 좌우로 편다(중심 간격 ≥ 표지 + 1.4 pt; Russia E·CA-3 는 |log E비| 차 0.001)
        fig.canvas.draw()
        px = ax_band.transData.transform(np.c_[xs, np.zeros_like(xs)])[:, 0]
        px = dodge_px(px, np.zeros_like(px), (ps.MS["main"] + 0.4 + 1.4) * ppt)
        xs_draw = ax_band.transData.inverted().transform(np.c_[px, np.zeros_like(px)])[:, 0]
        # 검열 = 채운 ▲(#4d4d4d), 범위 제한(n_max < 40) = 빈 △(같은 테두리색, 흰 채움): 밝기가 아니라 모양으로 구분(대비 8.5:1, 흑백 인쇄 대비)
        cc = ps.CENSOR["censored"]
        tcols = [PEND_COL] * len(cens)                      # 글자색 하나(구분은 글 '(range-limited)' 과 ▲/△ 모양)
        for x, rl in zip(xs_draw, cens.range_limited):
            ax_band.plot([x], [0.86], ls="none", marker="^", ms=ps.MS["main"] + 0.4, mfc="white" if rl else cc, mec=cc,
                         mew=0.7, zorder=3, clip_on=False)
        # 스펙 §2.4: n_max < 40 이면 '(range-limited)' 를 글로도 붙인다(회색 ▲ 와 이중 부호화, 흑백·색각 이상 대비)
        texts = [f"{tname(t, blk)}\n> {int(nm)}" + ("\n(range-limited)" if rl else "")
                 for t, nm, rl in zip(cens.target, cens.n_max, cens.range_limited)]
        place_band_labels(ax_band, xs_draw, texts, tcols)
        cens = cens.assign(x_plot=xs_draw)
    # 축 끊김 표시(왼쪽 축선, 위 영역 아래 끝과 띠 위 끝)
    for a, yy in ((ax_up, 0.0), (ax_band, 1.0)):
        tr2 = matplotlib.transforms.blended_transform_factory(a.transAxes, a.transAxes)
        d = 1.2 / 25.4 * fig.dpi / a.bbox.height if a is ax_up else 1.2 / 25.4 * fig.dpi / a.bbox.height
        w = 1.2 / 25.4 * fig.dpi / a.bbox.width
        a.plot([-w, w], [yy - d, yy + d], transform=tr2, color="#000000", lw=0.6, clip_on=False, zorder=5)
    ax_band.plot([0, 0], [0, 1], transform=ax_band.transAxes, color="#000000", lw=ps.LW["axis"], clip_on=False, zorder=4)
    return reached, cens


def _mean_k(sq, e_treat, n):
    """대상 평균 행의 기여 대상 수(a2_spread n_targets; 없으면 그 n 에서 Δ 가 있는 대상 수)."""
    d = sq[(sq.n == n) & (sq.e_treat == e_treat) & (sq.target == "MEAN_ANALYSIS")]
    if len(d) and "n_targets" in d and d.n_targets.notna().any():
        return int(d.n_targets.iloc[0])
    return int(sq[(sq.n == n) & (sq.e_treat == e_treat) & (sq.target != "MEAN_ANALYSIS")].dropna(subset=["delta"]).target.nunique())


F_Y = [3.7, 2.7, 1.0, 0.0]                                  # 행 y(E 고정 두 행 · 간격 · 라벨로 E 추정 두 행)
F_OFF = {0: 0.2, 1: -0.2}                                   # n = 10 위(실선 CI), n = 40 아래(파선 CI)


def _f_panel(ax, sq, n_f, blk=None):
    """배치 효과(분산 − 집중) E 처리별 포레스트. 큰 마커 = 대상 평균 + 블록 CI, 회색 점 = 개별 대상 Δ(CI 없음)."""
    col = ps.COLOR["residual"]
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_xlim(*F_XLIM)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-10, -1, 0, 1, 10]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ps.zero_line(ax, "x", better_text=True)
    rng = np.random.default_rng(7)
    out = []
    for (e_treat, _), y in zip(F_ROWS, F_Y):
        for j, n in enumerate(n_f):
            yy = y + F_OFF[j]
            ind = sq[(sq.e_treat == e_treat) & (sq.n == n) & (sq.target != "MEAN_ANALYSIS")].dropna(subset=["delta"])
            jit = rng.uniform(-0.07, 0.07, len(ind))
            v = ind.delta.to_numpy(float)
            vin = np.clip(v, *F_XLIM)
            ax.scatter(vin, yy + jit, s=ps.MS["point"] ** 2 * 1.2, color=ps.GREY["mid"], alpha=0.55, linewidths=0, zorder=2.2)
            out.append(ind.assign(y_plot=yy + jit, kind="target", n_plot=n))
            mrow = sq[(sq.e_treat == e_treat) & (sq.n == n) & (sq.target == "MEAN_ANALYSIS")]
            if len(mrow):
                r = mrow.iloc[0]
                ax.plot([max(r.ci_lo, F_XLIM[0]), min(r.ci_hi, F_XLIM[1])], [yy, yy], color=col,
                        ls="-" if j == 0 else ps.VARIANT_LS["dashed"], lw=ps.LW["ci_forest"], zorder=3.0,
                        solid_capstyle="butt", dash_capstyle="butt")
                ax.plot([r.delta], [yy], ls="none", marker="D", ms=ps.MS["mean"], mfc=col, mec="#000000", mew=0.5, zorder=3.5)
                out.append(mrow.assign(y_plot=yy, kind="mean", n_plot=n, mean_k=_mean_k(sq, e_treat, n)))
    ax.set_yticks(F_Y)
    ax.set_yticklabels([lab for _, lab in F_ROWS])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, F_Y[0] + 0.6)
    ax.set_xlabel("ΔRMSE, spread minus concentrated (cm)")
    # n 표지(직접 표기): 맨 위 행 오른쪽 끝
    trb = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for j, n in enumerate(n_f):
        tx = ax.text(0.99, F_Y[0] + F_OFF[j], f"n = {n}", transform=trb, ha="right", va="center", fontsize=ps.FS["annot"],
                     color=ps.GREY["text2"])
        tx.set_gid("n_label")
        ax.plot([0.78, 0.84], [F_Y[0] + F_OFF[j]] * 2, transform=trb, color=col, lw=ps.LW["ci_forest"],
                ls="-" if j == 0 else ps.VARIANT_LS["dashed"], solid_capstyle="butt", dash_capstyle="butt")
    # 오른쪽 묶음 괄호: E 고정 대 라벨로 E 추정
    for (y0, y1), txt in (((F_Y[1] - 0.4, F_Y[0] + 0.4), "E fixed"), ((F_Y[3] - 0.4, F_Y[2] + 0.4), "E from\nlabels")):
        ax.plot([1.015, 1.03, 1.03, 1.015], [y0, y0, y1, y1], transform=trb, color=ps.GREY["mid"], lw=0.5, clip_on=False)
        tb = ax.text(1.045, (y0 + y1) / 2, txt, transform=trb, fontsize=ps.FS["annot"], rotation=90, ha="left", va="center",
                     color=ps.GREY["text2"], clip_on=False, linespacing=0.95, multialignment="center")
        tb.set_gid("bracket")
    ax.axhline((F_Y[1] + F_Y[2]) / 2, color=ps.GREY["light"], lw=0.4, zorder=0.5)
    res = pd.concat(out, ignore_index=True)
    return res


# ---------------------------------------------------------------- 캡션
def _rep_sentence(cq, src) -> str:
    """반복 수 문장을 그린 자료에서 계산한다(분할 수 = 곡선 n_splits 최댓값, 라벨 추출·seed = 같은 출처의 meta)."""
    import json
    p = ROOT / src["curve"] if not str(src["curve"]).startswith("preview:") else Path(os.environ.get("FIG4_A2_DIR", ".")) / "a2_curve.csv"
    mp = p.with_name(p.name.replace("_curve.csv", "_meta.json"))
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    ns = int(cq.n_splits.max()) if "n_splits" in cq and cq.n_splits.notna().any() else len(meta.get("splits", [])) or None
    nr = meta.get("reps_cb"); nseed = len(meta.get("seeds", [])) or None
    f = lambda v: str(v) if v else "?"                      # noqa: E731
    return f"{f(ns)} splits × {f(nr)} label draws × {f(nseed)} seeds"


def _stats_sentence(mq, draft: bool) -> str:
    if draft or os.environ.get("FIG4_A2_DIR") or os.environ.get("FIG4_STRESS_E"):
        return "Rank tests of n* against |log(E_own/E0)|: review mode."
    from polar.h4_common import achieved_test, kendall_censored
    q = mq[mq.abs_logE.notna()]
    ach = ~q.censored.to_numpy(bool)
    if ach.sum() < 3:
        return f"{int(ach.sum())} of {len(q)} targets reached n* within the tested range (too few for a rank test)."
    a = achieved_test(ach, q.abs_logE.to_numpy(float))
    tau, p_tau, _ = kendall_censored(q.abs_logE.to_numpy(float), q.n_star.fillna(np.inf).to_numpy(float), censored_y=q.censored.to_numpy(bool))
    sep = "; complete separation" if a["separation"] else ""
    return (f"Reached ({a['n_achieved']} of {a['n']}) versus not: Mann-Whitney U {a['mw_method']} p = {a['p_mw']:.2f}{sep}; "
            f"censoring-aware Kendall τ_b(|log(E_own/E0)|, n*) = {ps.fmt_num(tau, 2)} (p = {p_tau:.2f}).")


def _f1_sentence(mq) -> str:
    """F1 판정 문장(사실 목록 정정본: 단일 분류 |log E비| < 0.15 구조 대상 중 n* ≤ 40 미달 수). e 패널 자료에서 계산."""
    q = mq[mq.abs_logE.notna() & (mq.abs_logE < STRUCT_MAX)]
    if not len(q):
        return ""
    slow = int((q.censored.to_numpy(bool) | (q.n_star.fillna(np.inf).to_numpy(float) > 40)).sum())
    verdict = "supported" if 2 * slow > len(q) else "not supported"
    return f" F1 {verdict}: {slow} of {len(q)} structure targets lack n* ≤ 40."


def _block_sentence(blk, targets) -> str:
    """CI 표시 규칙 + † 규칙(대상별 최소 채점 블록 수는 Supplementary Table 3)."""
    b = blk.loc[[t for t in targets if t in blk.index]] if blk is not None else pd.DataFrame()
    s = "CIs shown for targets with at least 8 scoring blocks across splits."
    if len(b) and b.flag.any():
        s += " †, a split with fewer than 5 scoring blocks (Supplementary Table 3)."
    return s


def _caption(mq, cq, sq, n_f, draft, src, blk=None):
    rep_s = _rep_sentence(cq, src)
    shown = sorted((set(cq.target) & set(PANEL_TARGETS)) | set(mq.target))
    kk = {n: _mean_k(sq, "E0_fixed", n) for n in n_f}
    nf = " and ".join(str(n) for n in n_f)
    kf = " and ".join(str(kk[n]) for n in n_f)
    return dict(
        definition=("Fig. 4 | Minimum target labels for residual learning, E treatment separated from label placement (A2). "
                    "ΔRMSE, method minus physics-anchor RMSE (E0 from source regions, n = 0) on held-out B blocks (cm); "
                    "negative is better. n* is the smallest n at which the block-bootstrap 95 % CI upper bound of ΔRMSE is below 0, "
                    "at least two of three splits improve, and at least 75 % of repeats improve (pre-specified). "
                    "Residual ML: CatBoost, λ = 0.25. Targets: 14 (CA-1 excluded, |A| = 3)."),
        statistics=("95 % block bootstrap CIs: 1,000 resamples of scoring blocks within each split, paired; "
                    f"{rep_s}. {_block_sentence(blk, shown)} " + _stats_sentence(mq, draft) + _f1_sentence(mq)),
        panels=("a–d, ΔRMSE against n, all-block placement: a, Russia W (level error); b, Lena; c, Canada; d, AL-2 (structure error). "
                "Parentheses: |log(E_own/E0)|, two decimals. Shared symmetric-log y axis, linear within ±2 cm. "
                "Filled or open: n* criteria met or not at that n. n = 0: source residual only. Band, bar: E0-fixed CI. "
                "Dotted: E_own known. Smaller, x-offset markers: E fitted to labels. Grey: n untested. "
                "a, table: ΔRMSE (cm) at n = 3 and 10, where markers hide the E_own line. "
                "e, E0-fixed n* against oracle |log(E_own/E0)| (square-root axis); ring, met at n = 0; band, "
                "not reached by the largest tested n, n_max (open triangle, n_max < 40). Structure < 0.15 ≤ intermediate (hatched) < 0.20 ≤ level. "
                f"f, spread minus concentrated at n = {nf}: diamonds, target mean with CI ({kf} targets); dots, "
                "targets. Symmetric-log x axis, linear within ±1 cm."),
        data=(f"Source: h4/{Path(src['curve']).name}, {Path(src['minn']).name}, {Path(src['spread']).name}; "
              "|log(E_own/E0)|: h3/h25b_targets.csv (least-squares E). n* for other E treatments: Supplementary Table 5, "
              "Supplementary Fig. 12. n = 40 excludes Russia W and E (n_max 10)."),
    )


# ---------------------------------------------------------------- 조립
def build(draft: bool = False):
    ps.use_paper()
    cq, mq, sq, n_f, m, src, blk = load_data(draft)
    H = 132.0
    fig = ps.paper_figure(180, H)
    # 행 1(a–d): 축 40 mm, 상자 y 84–124, 조건 라벨 124–128, 수준·구조 괄호 129–132
    Y1, H1 = 84.0, 40.0
    axes1 = []
    for i, t in enumerate(PANEL_TARGETS):
        x0, lab = (0.0, 14.0) if i == 0 else (50.0 + 40.0 * (i - 1), 4.0)
        axes1.append(ps.slot_mm(fig, x0, Y1, lab, 36.0, H1))
    rows_ad = []
    for i, (ax, t) in enumerate(zip(axes1, PANEL_TARGETS)):
        show_ci = bool(blk.loc[t, "show_ci"]) if t in blk.index else True
        r = _curve_panel(ax, cq, t, first=(i == 0), draft=draft, show_ci=show_ci)
        if len(r):
            rows_ad.append(r)
        _cond_label(ax, t, m, blk)
        if i > 0:
            ps.hide_ylabels(ax)
    axes1[0].set_ylabel("ΔRMSE vs physics (cm)")
    fx, fy = ps.mm_to_fig(fig, (14 + 170) / 2, 75.6)
    fig.text(fx, fy, "Target labels n", ha="center", va="bottom", fontsize=ps.FS["label"])
    # 수준·구조 괄호(행 1 위): a = 수준, b–d = 구조
    for (xa, xb), txt in (((14.0, 50.0), "level error"), ((54.0, 170.0), "structure error")):
        yb = 129.4
        (a0, b0), (a1, _) = ps.mm_to_fig(fig, xa, yb), ps.mm_to_fig(fig, xb, yb)
        _, tick = ps.mm_to_fig(fig, 0, yb - 0.9)
        fig.add_artist(Line2D([a0, a0, a1, a1], [tick, b0, b0, tick], transform=fig.transFigure, color=ps.GREY["mid"], lw=0.5))
        tx, ty = ps.mm_to_fig(fig, (xa + xb) / 2, yb + 0.35)
        fig.text(tx, ty, txt, ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    # 범례(1개, 7항목)
    col = ps.COLOR["residual"]
    hd = [Line2D([], [], ls="none", marker="D", ms=ps.MS["main"], mfc=col, mec=col, label="n* criteria met at this n"),
          Line2D([], [], ls="none", marker="D", ms=ps.MS["main"], mfc="white", mec=col, label="n* criteria not met")]
    for e_, var, _, lab in E_SERIES:
        st_ = _series_style(var)
        if e_ != "E0_fixed":
            st_["ms"] = MS_SEC
        hd.append(Line2D([], [], label=lab, **st_))
    hd.append(Line2D([], [], label="E$_\\mathrm{own}$ known + residual ML (reference)", **EOWN_STYLE))
    hd.append(Line2D([], [], ls="none", marker="o", ms=RING_MS, mfc="none", mec=ps.GREY["mid"], mew=0.6,
                     label="e: criteria already met at n = 0"))
    ps.legend_below(fig, hd, (4.0, 65.6, 172.0, 7.4), ncol=4, handlelength=3.0, columnspacing=1.4)
    # 행 2: e(위 영역 25–62, 끊김 2, 띠 12–23) + f
    ax_up = ps.slot_mm(fig, 0.0, 31.0, 14.0, 80.0, 31.0)
    ax_band = ps.slot_mm(fig, 0.0, 12.0, 14.0, 80.0, 17.0, sharex=ax_up)
    reached, cens = _e_panel(ax_up, ax_band, mq, fig, blk)
    ax_f = ps.slot_mm(fig, 96.0, 12.0, 25.0, 51.0, 50.0)          # 우 8 mm: 세로 괄호 글('E fixed'·'E from labels')
    fq = _f_panel(ax_f, sq, n_f, blk=blk)
    ps.label_panels(axes1 + [ax_up, ax_f], "abcdef")
    src_e = pd.concat([reached.assign(status="reached"), cens.assign(status="not reached")], ignore_index=True)
    cap = _caption(mq, cq, sq, n_f, draft, src, blk)
    mo = marker_overlaps(fig, [ax_up, ax_band] + axes1)
    if mo:
        print(f"[fig4] WARNING marker overlaps: {len(mo)} {mo[:6]}")
    if not (os.environ.get("FIG4_A2_DIR") or os.environ.get("FIG4_STRESS_E")):
        import json
        qd = ROOT / "outputs/figures/paper/_qa"; qd.mkdir(parents=True, exist_ok=True)
        (qd / f"{NAME}_markers.json").write_text(
            json.dumps(dict(n_marker_overlaps=len(mo), pairs=[list(map(str, x)) for x in mo], rule="표지 상자(크기 × 0.9) 겹침, 축별"), indent=1, ensure_ascii=False))
    if os.environ.get("FIG4_A2_DIR") or os.environ.get("FIG4_STRESS_E"):
        out = Path(os.environ.get("FIG4_OUT", ".")); out.mkdir(parents=True, exist_ok=True)
        tag = ("preview" if os.environ.get("FIG4_A2_DIR") else "") + ("_stressE" if os.environ.get("FIG4_STRESS_E") else "")
        paths = ps.save_figure(fig, out / f"{NAME}_{tag}")
        qa = qa_check(fig, paths["pdf"], cap)
        print(f"[fig4 {tag}] ok={qa['ok']} fails={qa['fails']} min={qa['min_font_pt']} overlaps={qa['n_text_overlaps']} marker_overlaps={len(mo)}")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None
    keep = ["target", "series", "n", "x_plot", "blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "split_win", "rep_win", "pass3", "n_runs",
            "n_splits", "ci_kind"]
    src_ad = pd.concat(rows_ad, ignore_index=True)[lambda d: [c for c in keep if c in d.columns]] if rows_ad else pd.DataFrame()
    keep_e = ["target", "abs_logE", "x_plot", "n_star", "censored", "n_max", "range_limited", "pass0", "status"]
    src_e = src_e[[c for c in keep_e if c in src_e.columns]]
    keep_f = ["target", "e_treat", "n_plot", "kind", "delta", "ci_lo", "ci_hi", "split_win", "target_win", "n_targets", "mean_k", "y_plot", "ci_kind"]
    fq = fq[[c for c in keep_f if c in fq.columns]]
    spec = dict(
        intent="A2 교락 제거 최소 라벨 수: E 처리와 라벨 배치를 분리한 잔차 학습의 n*(주 정의 세 조건), 달성/미달성 이진 표시, E 처리별 라벨 배치 효과",
        claim=("대상 라벨 수십 개로는 원천 잔차가 대상에 적응하지 못한다(구조 지역 8곳 중 Lena n* = 320, AL-2·AL-4·AL-6·LE-1·LE-2 미달성, "
               "Canada·AL-5 는 n = 0 원천 잔차만으로 이미 충족; n* = 3 달성 6곳 중 5곳이 n = 0 충족). "
               "소수 라벨의 가치는 E 추정에서 나오며(Russia W: E0 고정 −0.6 대 κ 수축 −11.4 cm, n = 10), 배치 효과도 E 를 라벨로 추정할 때만 생긴다"
               "(n = 40 분산 − 집중: κ −1.14, 오프셋 MLE −1.65, E0 고정 +0.04 cm)."),
        facts="scratchpad results_facts.md F1–F3(2026-09-26 20:45), 단일 분류 = h3/h25b_targets 최소제곱 |log E비|",
        panels=6, ci_kinds=dict(a_d="block", e="none (판정 결과)", f="block (대상 평균), 개별 대상 점은 CI 없음"),
        ci_rule=f"분할 합계 채점 블록 ≥ {MIN_BLK_CI} 일 때만 CI, 분할 채점 블록 < {FLAG_BLK} 대상은 †",
        block_info={t: dict(tot=(None if not np.isfinite(r["tot"]) else int(r["tot"])), min=(None if not np.isfinite(r["min"]) else int(r["min"])),
                            flag=bool(r["flag"]), show_ci=bool(r["show_ci"])) for t, r in blk.iterrows()},
        marker_overlaps=len(mo),
        assets=[src["curve"], src["minn"], src["spread"], "data/processed/h3/h25b_targets.csv"],
        layout=dict(figsize_mm=[180, H], row1="a 14+36, b–d 4+36, 우 10; 축 40 mm, 조건 라벨·수준/구조 괄호 8 mm",
                    legend="1개 7항목 4열 2줄(66–73 mm)",
                    row2="e 14+80(위 31 mm, 끊김 2, 띠 17) · 간격 2 · f 25+51(우 8, E 고정/라벨 추정 괄호)",
                    deviation=("스펙 §2.4 대비: a–d 대상을 Russia W·Lena·Canada·AL-2 로 교체(사실 목록 메시지), n = 0 자리 추가(축 끊김), "
                               "f 를 대상 행에서 E 처리 행(평균 + 개별 대상 점)으로 교체, |log E비| = h25b_targets 최소제곱(사실 목록 단일 정의)")),
        encoding=dict(filled="n* 세 조건 충족(이 그림 유일한 빈 마커 의미)", E0_fixed="실선 + 블록 CI 띠", shrink_k10="파선",
                      offset_mle="일점쇄선 (0,(4,2,1,2))", E_own_fixed="검정 점선 0.9 pt(참조, 상한 아님), 표지 위 zorder 3.4",
                      secondary_markers=f"κ 수축·오프셋 MLE 표지 {MS_SEC:.1f} pt(E0 고정 {ps.MS['main']} pt 의 80 %)",
                      value_table="E_own 점선이 표지에 가리고 n > n_max 음영에 자리가 있을 때(이 자료에서 a 만) n = 3·n_max 값 표(κ·Offset·E_own, cm)",
                      abslogE_header="|log E비| 소수 둘째 자리(0.51, 0.12, 0.00, 0.06; Fig 2·3·Table 1 과 통일)",
                      n0="x 축 '0' 자리(축 끊김), E0 고정 점 + 블록 CI 막대", ring="e: n = 0 에서 이미 세 조건 충족(회색 고리)",
                      untested="a–d 음영 = n > n_max", censored="띠 채운 ▲ #4d4d4d, range-limited 빈 △(흰 채움, #4d4d4d 테두리) + 글 '(range-limited)'",
                      error_type="수준 ≥ 0.20, 구조 < 0.15, 중간 0.15–0.20(e 빗금)",
                      f_n=f"n={n_f[0]} CI 실선(+0.2), n={n_f[-1]} CI 파선(−0.2), 직접 표기"),
    )
    return save_paper(fig, NAME, caption=cap, spec=spec, draft=draft,
                      sources={"a-d": src_ad, "e": src_e, "f": fq})
