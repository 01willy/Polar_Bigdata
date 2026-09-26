"""Fig 2. 라벨 있는 지역: 대조군 사다리·레시피 지역별 Δ·이득 분해·선택 방식(스펙 §2.2, 감사 조치 5 반영).

주장(한 문장): 물리 유사라벨 대조군 사다리의 매 단이 0 을 제외하고, 사전 지정 레시피(Stefan 앵커 + CatBoost 잔차 λ 0.25)의
이득은 E 재적합(수준)과 잔차(구조) 두 성분으로 나뉜다. 중첩 선택(H28 등록 가설)은 0 을 제외하지 못한다.

패널
  a  대조군 사다리 5단(AB4 평균, strat_AB4 CI) + 지역별 점 추정(작은 회색 점). 기준 = 대조군(행)이므로 x 라벨 "ΔRMSE vs control".
  b  M1 H13 레시피 지역별 Δ(λ 0.25 실선 CI, λ 0.5 파선 CI, 세로 오프셋 ±0.18) + 4 지역 평균(strat_AB4).
  c  이득 분해 덤벨(15 대상, |log E비| 오름차순, 수준·구조 띠). 수준 = E 재적합(refit 'o'), 총 = 재적합 + 잔차(residual 'D'),
     선분 = 구조 성분(개선 진회색, 악화 연회색). 러시아 W(−12 cm)는 끊긴 축 왼쪽 조각에 그린다. CI 없음(분할 1–3).
  d  선택 방식 4 행: 중첩 선택(CatBoost 계열), 사전 지정 λ 0.25·0.5, 사후 최선 레시피(참조). 채움 = 셀 가중, 빈 = 블록 등가중.

스펙 대비 변경(근거는 notes·figure_spec 기록)
  - d 를 세로 Δ 축에서 가로 포레스트로 바꿨다(§1.6 '한 그림 안에서 포레스트 가로·곡선 세로를 섞지 않는다').
  - H28ref_oracle 은 코드상 '대상 지역 B 점수로 고른 최선 레시피'(h28_recipe_nested.py 126행)이므로 "all A-block labels"가 아니라
    "post hoc best recipe (reference)"로 표기하고 oracle_branch(검정 ★) 부호를 쓴다.
  - a 행 라벨은 m1_core.pseudo_label 정의를 따른다(const_t = 대상 Stefan 평균 상수). _common.LADDER_LABEL 의 "constant + TDD"는 쓰지 않는다.
  - 범례를 d 아래 빈 영역에 둔다(별도 범례 행 7 mm 대신).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (ps, AB4, LEVEL_THRESH, load_ladder, load_m1_tests, load_decomp, load_h28_tests, abslogE_map,
                     order_by_abslogE, save_paper, MissingData)  # noqa: F401

NAME = "Fig2_recipe_decomp"

# a: 대조군 행(위 → 아래 = 대조군이 Stefan 에 가까워지는 순서). 원문 label 의 대조군 키 → 영문 행 라벨
LADDER_ROWS = [("none", "No pseudo-labels"), ("const", "Source-mean constant"), ("const_t", "Target-mean constant"),
               ("shuffle", "Shuffled Stefan"), ("tddlin", "Linear TDD")]
# d: 선택 방식 행(h3/h28_tests.csv test 이름)
D_ROWS = [("H28_nested_cb_vs_stefan", "Nested selection", "residual", None),
          ("H28ref_prespec25_vs_stefan", "Pre-specified, λ = 0.25", "residual", None),
          ("H28ref_prespec50_vs_stefan", "Pre-specified, λ = 0.5", "residual", "dashed"),
          ("H28ref_oracle_vs_stefan", "Post hoc best recipe", "oracle_branch", None)]
H28_FAMILY_MEAN = "MEAN[Lena,Canada,Russia_W,Russia_E]"

# 레이아웃(mm, 좌하 원점). 폭 합계: 행 1 = 27 + 50 + 3 + 24 + 74 + 2 = 180, 행 2 = 24 + 7 + 1.5 + 63.5 + 10 + 30 + 42 + 2 = 180
# 높이 136 mm = §2.2 표 합계(46 + 7 + 79 + 4). 범례 행 7 mm 를 d 아래로 옮긴 만큼 행 2 축을 70 mm(15행 × 4.67)로 늘렸다.
H_MM = 136.0
ROW1 = dict(y=94.0, h=36.0)
ROW2 = dict(y=10.0, h=70.0)
D_BOX = dict(y=40.0, h=40.0)       # d 축 상자: 행 2 상단 정렬, 높이 40(스펙 §2.2)
LEG_TOP_MM = 31.0                  # 범례 상단 = d x 라벨 바로 아래(아래 여백 ≤ 6 mm)
BAND_TAG = dict(level="|log(E$_\\mathrm{own}$/E$_0$)| ≥ 0.15", structure="|log(E$_\\mathrm{own}$/E$_0$)| < 0.15")   # Fig 4·Table 1 과 같은 표기   # 띠 = 사전 분류(성분 우위 아님)
C_BREAK = (-13.0, -11.3)          # c 끊긴 축 왼쪽 조각(러시아 W)
C_MAIN = (-8.0, 6.5)
# c 선분: 구조 성분 개선 = 진회색 실선, 악화 = 중회색 짧은 파선(스펙 #b0b0b0 은 흰 바탕 대비 2.2:1 로 기준 3:1 미달 → #8c8c8c 3.4:1 + 선종 중복 부호)
SEG_GAIN, SEG_LOSS, SEG_LOSS_LS = "#4d4d4d", "#8c8c8c", (0, (2.0, 1.0))
# c 부호 표지: 구조 성분 |struct| < SIGN_MIN_CM 인 행은 선분이 인쇄 크기에서 판독 불가(0.5 cm ≈ 2.2 mm) →
# ◆(총) 바깥, 구조 성분 방향 쪽 SIGN_GAP_MM 에 열린 꺾쇠(획만, 회색)를 둔다. 범위 밖 표지(채운 계열색 삼각형, 축 끝)와 구별된다.
SIGN_MIN_CM, SIGN_GAP_MM, SIGN_MS = 0.5, 1.9, 2.6
PANEL_DX_MM = 1.2                 # 패널 문자 좌측 여백: 'a'·'c' 잉크가 캔버스 왼쪽 끝에서 ≥ 1 mm(조판 재크롭 대비)
A_LIM, B_LIM, D_LIM = (-8.0, 1.0), (-7.0, 3.0), (-9.0, 3.5)


def _ctrl_key(label: str) -> str:
    """'covonly: Stefan 유사라벨 − const_t (직접 …)' → 'const_t'."""
    return label.split("−")[1].strip().split(" ")[0]


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj


# ---------------------------------------------------------------- 자료
def data_a():
    L = load_ladder()
    L["ctrl"] = L.label.map(_ctrl_key)
    T = load_m1_tests(valid_only=False)
    R = T[(T.H == "H3") & T.label.isin(L.label) & T.target.isin(AB4)].copy()
    R["ctrl"] = R.label.map(_ctrl_key)
    order = [k for k, _ in LADDER_ROWS]
    miss = set(order) - set(L.ctrl)
    if miss:
        raise KeyError(f"사다리 대조군 누락: {miss}")
    L = L.set_index("ctrl").loc[order].reset_index()
    L["row_label"] = [dict(LADDER_ROWS)[k] for k in L.ctrl]
    return L, R


def data_b():
    T = load_m1_tests(valid_only=False)
    q = T[(T.H == "H13") & (T.cond == "labels")].copy()
    q["lam"] = np.where(q.label.str.contains("0.25"), 0.25, np.where(q.label.str.contains("0.5"), 0.5, np.nan))
    if q.lam.isna().any():
        raise ValueError("H13 λ 파싱 실패")
    m = abslogE_map()
    regions = [t for t in q.target.unique() if not str(t).startswith("REGION_SUMMARY")]
    with_e = order_by_abslogE([t for t in regions if t in m], m)
    no_e = [t for t in ["Russia_C", "Greenland"] if t in regions]
    rows = with_e + no_e + ["REGION_SUMMARY_AB4"]
    labels = [ps.region_label(t, m.get(t)) for t in with_e + no_e] + ["Mean (4 regions)"]
    out = []
    for lam in (0.25, 0.5):
        s = q[q.lam == lam].set_index("target").loc[rows]
        ok = s.ci_valid.astype(str).str.lower().isin(["true", "1"]).to_numpy() if "ci_valid" in s else np.ones(len(s), bool)
        lo = np.where(ok, s.ci_lo, np.nan); hi = np.where(ok, s.ci_hi, np.nan)
        out.append(pd.DataFrame(dict(target=rows, row_label=labels, lam=lam, delta=s.delta_rmse.to_numpy(), ci_lo=lo, ci_hi=hi,
                                     ci_kind=s.ci_kind.to_numpy(), n_cells=s.n_cells.to_numpy(), n_blocks=s.n_blocks.to_numpy(),
                                     p_holm=s.p_holm.to_numpy())))
    return pd.concat(out, ignore_index=True), labels


def data_c():
    D = load_decomp()
    if "level_shrunk" in D:                       # h25 전량 A 라벨 판(κ 수축 재적합 + 잔차 λ 0.25)
        lev, st, tot = "level_shrunk", "struct_on_Eshrunk", "total_shrunk_resid"
    else:                                         # A2 판(E_own 고정 + 잔차 λ 0.25)
        lev, st, tot = "level", "struct_on_Eown", "total_Eown_resid"
    D = D.rename(columns={lev: "level_c", st: "struct_c", tot: "total_c"})
    if not np.allclose(D.level_c + D.struct_c, D.total_c, atol=1e-6):
        raise ValueError("분해 불일치: 수준 + 구조 ≠ 총")
    m = dict(zip(D.target, D.abslogE))
    order = order_by_abslogE(D.target, m)
    D = D.set_index("target").loc[order].reset_index()
    D["row_label"] = [ps.region_label(t, m[t]) for t in D.target]
    D["error_type"] = np.where(D.abslogE >= LEVEL_THRESH, "level", "structure")
    D.attrs["variant"] = lev
    return D


def data_d():
    T = load_h28_tests()
    M = T[T.target == H28_FAMILY_MEAN].copy()
    fam = M[M.test.str.startswith("H28") & ~M.test.str.contains("alaska")]
    if len(fam) != 8:
        raise ValueError(f"H28 가족 대조 수 {len(fam)} ≠ 8")
    fam = fam.assign(p_holm8=_holm(fam.p_boot.to_numpy()))
    rows = []
    for test, lab, meth, var in D_ROWS:
        r = fam[fam.test == test]
        if len(r) != 1:
            raise KeyError(test)
        r = r.iloc[0]
        rows.append(dict(test=test, row_label=lab, method=meth, variant=var, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         delta_blockeq=r.delta_blockeq, ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, role=r.role,
                         p_boot=r.p_boot, p_holm8=r.p_holm8, ci_kind="strat_AB4"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그리기 도우미
def _break_marks(fig, ax_l, ax_r, size_mm=1.2):
    """끊긴 축 표시(아래 spine 두 곳에 사선)."""
    W, H = ps.fig_mm(fig)
    for ax, side in ((ax_l, 1.0), (ax_r, 0.0)):
        x0, y0, w, h = ax._box_mm
        xm = x0 + side * w
        for dx in (0.0,):
            fig.add_artist(__import__("matplotlib").lines.Line2D(
                [(xm + dx - size_mm / 3) / W, (xm + dx + size_mm / 3) / W], [(y0 - size_mm / 2) / H, (y0 + size_mm / 2) / H],
                color="#000000", lw=ps.LW["axis"], transform=fig.transFigure, clip_on=False))


def _chevron(sign: float):
    """열린 꺾쇠 마커 경로(획만). sign < 0 → '<'(구조 성분 개선), > 0 → '>'(악화)."""
    from matplotlib.path import Path
    v = [(0.5, 1.0), (-0.5, 0.0), (0.5, -1.0)] if sign < 0 else [(-0.5, 1.0), (0.5, 0.0), (-0.5, -1.0)]
    return Path(v, [Path.MOVETO, Path.LINETO, Path.LINETO])


def _row_sep(ax, y, lim):
    ax.plot(lim, [y, y], color=ps.GREY["light"], lw=0.4, zorder=0.5, clip_on=False)


# ---------------------------------------------------------------- 그림
def build(draft: bool = False):
    ps.use_paper()
    import matplotlib.pyplot as plt                  # noqa: F401
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    La, Ra = data_a()
    Bdf, b_labels = data_b()
    Dc = data_c()
    Dd = data_d()

    fig = ps.paper_figure(180, H_MM)

    # ---- a: 대조군 사다리
    ax_a = ps.slot_mm(fig, 0, ROW1["y"], 27, 50, ROW1["h"])
    ya = ps.forest(ax_a, list(La.row_label), La.delta_rmse, La.ci_lo, La.ci_hi, method="augment", offscale=A_LIM)
    rng = np.random.default_rng(0)
    for yi, k in zip(ya, La.ctrl):
        v = Ra[Ra.ctrl == k].delta_rmse.to_numpy()
        jit = np.linspace(-0.22, 0.22, len(v)) if len(v) > 1 else np.zeros(1)
        ax_a.plot(np.clip(v, *A_LIM), yi + jit + rng.normal(0, 0.0, len(v)), ls="none", marker="o", ms=ps.MS["point"] + 0.4,
                  mfc=ps.GREY["mid"], mec="none", alpha=0.55, zorder=2)
    ax_a.set_xlabel("ΔRMSE vs control (cm)")
    ax_a.set_xticks([-8, -6, -4, -2, 0])

    # ---- b: 레시피 지역별 Δ(λ 0.25 / 0.5)
    ax_b = ps.slot_mm(fig, 80, ROW1["y"], 24, 74, ROW1["h"])
    b25, b50 = Bdf[Bdf.lam == 0.25], Bdf[Bdf.lam == 0.5]
    yb = ps.forest(ax_b, b_labels, b25.delta, b25.ci_lo, b25.ci_hi, method="residual", offset=0.18, ls="-", offscale=B_LIM)
    ps.forest(ax_b, b_labels, b50.delta, b50.ci_lo, b50.ci_hi, method="residual", offset=-0.18, ls="dashed", offscale=B_LIM,
              set_labels=False)
    _row_sep(ax_b, yb[-1] + 0.5, B_LIM)
    ax_b.set_xlabel("ΔRMSE vs physics (cm)")
    ax_b.set_xticks([-6, -4, -2, 0, 2])

    # ---- c: 이득 분해 덤벨(끊긴 축)
    xL, lab_c = 0.0, 24.0
    wL, gap = 7.0, 1.5
    ax_c0 = ps.slot_mm(fig, xL, ROW2["y"], lab_c, wL, ROW2["h"])
    ax_c = ps.axes_mm(fig, xL + lab_c + wL + gap, ROW2["y"], 63.5, ROW2["h"])
    ax_c._slot_mm = ax_c0._slot_mm
    labels_c = list(Dc.row_label)
    yc = np.arange(len(Dc))[::-1].astype(float)                                  # 첫 행이 맨 위(forest 규약)
    for ax, lim, main in ((ax_c0, C_BREAK, False), (ax_c, C_MAIN, True)):
        lv_rows = [y for y, e in zip(yc, Dc.error_type) if e == "level"]
        st_rows = [y for y, e in zip(yc, Dc.error_type) if e == "structure"]
        ps.group_bands(ax, lv_rows, text=BAND_TAG["level"] if main else None)
        ps.group_bands(ax, st_rows, color="white", text=BAND_TAG["structure"] if main else None)
        for yi, a, b, s in zip(yc, Dc.level_c, Dc.total_c, Dc.struct_c):
            ax.plot([a, b], [yi, yi], color=SEG_GAIN if s < 0 else SEG_LOSS, ls="-" if s < 0 else SEG_LOSS_LS, lw=ps.LW["ci"] + 0.2,
                    solid_capstyle="butt", dash_capstyle="butt", zorder=2)
        # 총(◆, 크게)을 먼저, 수준(●, 작게)을 위에 그려 두 값이 겹쳐도 둘 다 보이게 한다
        ax.plot(Dc.total_c, yc, ls="none", **{k: v for k, v in ps.style_of("residual", ms=ps.MS["main"] + 0.8).items() if k not in ("ls", "lw")}, zorder=3)
        # 수준 ● 에 흰 테두리(0.5 pt)를 둘러 ◆ 와 겹칠 때도 두 모양이 인쇄 크기에서 갈라져 보이게 한다
        st_lv = {k: v for k, v in ps.style_of("refit", ms=ps.MS["main"] - 0.2).items() if k not in ("ls", "lw")}
        st_lv.update(mec="white", mew=0.5)
        ax.plot(Dc.level_c, yc, ls="none", **st_lv, zorder=3.1)
        # 짧은 구조 성분의 부호 꺾쇠(길이는 늘리지 않는다: 크기 왜곡 방지)
        cm_per_mm = (lim[1] - lim[0]) / ax._box_mm[2]
        for yi, b, s in zip(yc, Dc.total_c, Dc.struct_c):
            if abs(s) < SIGN_MIN_CM and s != 0 and lim[0] <= b <= lim[1]:
                sg = np.sign(s)
                ax.plot([b + sg * SIGN_GAP_MM * cm_per_mm], [yi], ls="none", marker=_chevron(sg), ms=SIGN_MS, mfc="none",
                        mec=SEG_GAIN if s < 0 else SEG_LOSS, mew=0.6, zorder=3.2, gid="sign_chevron")
        ax.set_xlim(*lim); ax.set_ylim(-0.6, len(yc) - 0.4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=2)
    ax_c0.set_yticks(yc); ax_c0.set_yticklabels(labels_c)
    ax_c.set_yticks(yc); ax_c.set_yticklabels([])
    ax_c0.set_xticks([-12])
    ax_c.set_xticks([-8, -4, 0, 4])
    ps.zero_line(ax_c, "x")
    _break_marks(fig, ax_c0, ax_c)
    ax_c.set_xlabel("ΔRMSE vs physics (cm)")

    # ---- d: 선택 방식(셀 가중 채움 / 블록 등가중 빈)
    ax_d = ps.slot_mm(fig, 106, D_BOX["y"], 30, 42, D_BOX["h"])
    yd = np.arange(len(Dd))[::-1].astype(float)
    for yi, r in zip(yd, Dd.itertuples()):
        for off, est, lo, hi, filled in ((0.2, r.delta, r.ci_lo, r.ci_hi, True), (-0.2, r.delta_blockeq, r.ci_lo_beq, r.ci_hi_beq, False)):
            ps.ci_errorbar(ax_d, yi + off, max(lo, D_LIM[0]), min(hi, D_LIM[1]), orient="h", method=r.method, variant=r.variant,
                           lw=ps.LW["ci_forest"])
            st = ps.style_of(r.method, filled=filled, ms=ps.MS["main"] + (1.6 if r.method == "oracle_branch" else 0))
            ax_d.plot([est], [yi + off], ls="none", marker=st["marker"], ms=st["ms"], mec=st["mec"], mfc=st["mfc"], mew=st["mew"], zorder=3)
    ax_d.set_yticks(yd); ax_d.set_yticklabels(list(Dd.row_label))
    ax_d.set_ylim(-0.6, len(yd) - 0.4); ax_d.set_xlim(*D_LIM)
    ax_d.tick_params(axis="y", length=0, pad=2); ax_d.spines["left"].set_visible(False)
    ps.zero_line(ax_d, "x")
    ax_d.set_xlabel("ΔRMSE vs physics (cm)")
    ax_d.set_xticks([-8, -4, 0])

    # ---- 범례(그림당 1개, d 아래)
    # d 의 빈 마커는 대부분 ◇(residual)·☆(사후 최선) → 범례도 빈 ◇(중립 회색). a 의 채운 회색 점과 짝으로 오독되지 않게 원을 쓰지 않는다
    grey_open = Line2D([], [], ls="none", marker="D", ms=ps.MS["main"], mec=ps.GREY["text2"], mfc="white", mew=ps.LW["marker_edge"])
    seg_gain = Line2D([], [], color=SEG_GAIN, lw=ps.LW["ci"] + 0.2)
    seg_loss = Line2D([], [], color=SEG_LOSS, ls=SEG_LOSS_LS, lw=ps.LW["ci"] + 0.2)
    handles = [
        ps.method_handle("augment", "Stefan pseudo-labels", line=False),
        Line2D([], [], ls="none", marker="o", ms=ps.MS["point"] + 0.4, mfc=ps.GREY["mid"], mec="none", alpha=0.55, label="Single region"),
        ps.method_handle("residual", "Anchor + residual CatBoost, λ = 0.25"),
        ps.method_handle("residual", "Same, λ = 0.5", variant="dashed"),
        ps.method_handle("refit", "E shrinkage only (level part)", line=False),
        (seg_gain, seg_loss),
        grey_open,
        ps.method_handle("oracle_branch", "Post hoc best recipe (reference)", line=False),
    ]
    labels = [h.get_label() if not isinstance(h, tuple) else "" for h in handles]
    labels[5] = "Structure part (gain, loss)"
    labels[6] = "Open: block-equal scoring"
    handles[7].set_markersize(ps.MS["main"] + 1.6)   # d 의 ★ 크기와 맞춘다
    for h in handles[2:4]:                           # CI 선종이 보이도록 굵은 선 + 마커
        h.set_linewidth(ps.LW["ci_forest"])
    assert len(handles) <= 8
    fx, fy = ps.mm_to_fig(fig, 106.0, 1.0); fw, fh = ps.mm_to_fig(fig, 74.0, LEG_TOP_MM - 1.0)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(fx, fy, fw, fh), bbox_transform=fig.transFigure, ncol=1,
               fontsize=ps.FS["legend"], frameon=False, handlelength=2.8, labelspacing=0.75, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})

    ps.label_panels([ax_a, ax_b, ax_c0, ax_d], "abcd", dx_mm=PANEL_DX_MM)

    # ---- 캡션
    p25 = Dd.set_index("test").p_holm8
    sig25, sig50 = p25["H28ref_prespec25_vs_stefan"] < 0.05, p25["H28ref_prespec50_vs_stefan"] < 0.05
    nest_ci0 = bool(Dd.set_index("test").loc["H28_nested_cb_vs_stefan", "ci_hi"] > 0)
    if not (sig25 and not sig50 and nest_ci0):
        raise AssertionError("H28 Holm 판정이 캡션 문구와 다르다(λ 0.25 유의, λ 0.5 비유의, 중첩 CI 0 포함)")
    splits = sorted(set(int(v) for v in Dc.n_splits.dropna())) if "n_splits" in Dc else []
    split_txt = f"{splits[0]}–{splits[-1]}" if len(splits) > 1 else (str(splits[0]) if splits else "1–3")
    caption = dict(
        definition=("Fig. 2 | Recipe and gain decomposition in labelled regions. ΔRMSE is method RMSE minus comparator RMSE on held-out "
                    "B blocks (cm); negative is better. The comparator is the physics anchor (Stefan, E0) unless the axis names a "
                    "control. Recipe: Stefan anchor plus CatBoost residual with weight λ. Parentheses give |log(E_own/E0)|; AL, CA and LE "
                    "denote subregions of Alaska, Canada and Lena."),
        statistics=("Error bars are unadjusted 95 % bootstrap CIs from 1,000 resamples of scoring blocks within a region, paired by "
                    "seed, with splits pooled; regions with fewer than 8 blocks have no CI. Means (a, b, d) combine the regional "
                    "distributions of Lena, Canada, Russia W and Russia E, stratified by region. Significance follows Holm-adjusted "
                    f"p (Supplementary Table 2), not CI overlap with zero. Panel c shows point estimates averaged over {split_txt} "
                    "splits, without CI."),
        panels=("a, Stefan pseudo-label augmentation versus five controls (covariates-only, direct CatBoost); grey dots, single "
                "regions. b, Recipe per region (M1 H13); upper points λ = 0.25, lower λ = 0.5; Russia C and Greenland have fewer "
                "than 8 blocks. c, Gain with all A-block labels split into level (E shrinkage, κ = 10) and structure "
                "(residual on the shrunk-E anchor) parts; segment, structure part (solid, gain; dashed, loss); open chevron, its "
                "sign where shorter than 0.5 cm. Grey/white marks the pre-classified |log(E_own/E0)| ≥ 0.15 / < 0.15 threshold "
                "only; it does not indicate which part is larger. Russia W is on the broken axis. d, "
                "Filled, cell-weighted; open, block-equal. Registered nested selection (H28) includes zero (H28 rejected). "
                "Pre-specified λ = 0.25 remains significant after Holm adjustment over eight H28 contrasts; λ = 0.5 does not. The "
                "post hoc best recipe is chosen on target scores (reference only)."),
        data=("Source: data/processed/h4/a1_ladder.csv, m1/m1_sc_tests.csv, h4/a1_decomp.csv, h3/h28_tests.csv. Scoring cells "
              "per region: Supplementary Table 1; cell-weighted and block-equal values: Supplementary Table 4."),
    )
    src_a = La[["ctrl", "row_label", "delta_rmse", "ci_lo", "ci_hi", "ci_kind"]].merge(
        Ra[["ctrl", "target", "delta_rmse"]].rename(columns={"delta_rmse": "region_delta"}), on="ctrl", how="left")
    spec = dict(intent="라벨 있는 지역: 대조군 사다리 전 단 0 제외, 사전 지정 레시피 이득의 수준·구조 분해, 중첩 선택(H28 등록) 기각",
                panels=4, layout_mm=dict(width=180, height=H_MM, row1=ROW1, row2=ROW2,
                                         slots=dict(a=[0, 27, 50], b=[80, 24, 74], c=[0, 24, "7 | 1.5 | 63.5"], d=[106, 30, 42]),
                                         d_box=D_BOX, legend_top=LEG_TOP_MM,
                                         height_note=("136 = §2.2 합계(46 + 7 + 79 + 4). 범례 행 7 mm 는 d 아래로 옮기고 행 2 축을 70 mm 로 늘림. "
                                              "행 1 축 하단(94)과 행 2 축 상단(80) 사이 14 mm 는 빈 띠가 아니다: 행 1 x 눈금·x 라벨(약 8 mm) + "
                                              "행 2 패널 문자·'better ←' 표지(약 5 mm) + 여유 1 mm")),
                assets=["data/processed/h4/a1_ladder.csv", "data/processed/m1/m1_sc_tests.csv", "data/processed/h4/a1_decomp.csv",
                        "data/processed/h3/h28_tests.csv"],
                ci_kinds=dict(a="strat_AB4", b="지역 = m1_stats.boot_delta 블록 부트스트랩(분할 합침, seed 짝지음, 블록 ≥ 8), 평균 = strat_AB4",
                              c="none (point estimates)", d="strat_AB4(h28, 같은 블록 부트스트랩)"),
                deviations=["d: 가로 포레스트(§1.6 방향 규칙)", "H28ref_oracle = 사후 최선 레시피(oracle_branch ★), all A-block labels 아님",
                            "a 행 라벨: const_t = target-mean constant(m1_core 정의)", "범례를 d 아래 빈 영역에 배치(d x 라벨 바로 아래, 1열 8항목)",
                            "c 띠 태그: level/structure 대신 '|log E ratio| ≥ 0.15 / < 0.15'(성분 우위로 오독 방지)",
                            "c: 러시아 W 끊긴 축 조각(off-scale 마커 대신 실제 값)",
                            "c: |구조 성분| < 0.5 cm 행에 열린 회색 부호 꺾쇠(◆ 바깥 1.9 mm, 선분 길이 불변)",
                            "패널 문자 dx = +1.2 mm(좌측 열 잉크-캔버스 여백 ≥ 1 mm)"],
                decomp_variant=Dc.attrs.get("variant"))
    return save_paper(fig, NAME, caption=caption, spec=spec, draft=draft,
                      sources={"a": src_a, "b": Bdf, "c": Dc[["target", "row_label", "abslogE", "error_type", "level_c", "struct_c",
                                                               "total_c", "rmse_phys"] + (["n_splits"] if "n_splits" in Dc else [])],
                               "d": Dd})
