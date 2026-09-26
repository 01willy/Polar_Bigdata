"""Fig 6. 라벨 0 지역(스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.6).

주장: 라벨 없는 지역에서 H18–H23 우회 기법은 물리식을 유의하게 넘지 못했고, 배포 규칙(H30)은 물리식과 동급이었다(확정).
CCI 결합(C1)과 계층 conformal 구간(C2)의 판정 문장은 F9·F10 확정 전까지 캡션 자리표로 둔다.

패널
  a  우회 기법 포레스트(10행, 3가족 띠). 조건 = CI 선종(정보 없음 실선, 공변량만 파선), CI = strat_AB4(h2 기존 자료).
     H23 MAML 은 h23_summary 의 라벨 0(maml0, λ 0.25) AB4 평균 d_phys 점 추정(구간 없음, 채점 블록이 H23 평가 집합).
  b  배포 규칙 7종 AB4 평균(채운 마커)·최악 지역(세로 틱) 덤벨, 절대 RMSE [25, 50] cm, 물리식 평균·최악 점선.
     연결선은 모두 실선이다(검토 2회차: 파선은 공용 범례에서 a 의 '공변량만' 조건을 뜻하므로 b 에서 재사용하지 않는다).
     게이트 변형은 행 이름(+AOA, CCI gate)으로만 구분한다.
  c  C1 CCI 4수준 대 물리식. 채운 = 셀 가중 AB4 평균, 빈 = 블록 등가중 AB4 평균(그림 안 빈 마커 의미는 이것 하나),
     연한 점 = AB4 지역별 셀 가중 Δ(채운 마커 줄 위 별도 띠). CI = 블록 부트스트랩(지역 분포를 같은 번호끼리 평균).
     채운·빈 의미는 두 표지 모두 자기 CI 오른쪽 끝 바깥 3 pt 에 둔다(같은 방향 규칙). 두 표지가 축 안에 들어가는
     첫 행을 렌더러로 골라 표기한다(행 1 은 block-equal 이 축 밖으로 나가 행 2 가 선택됨).
  d  C2 LORO 라벨 0 커버리지 대 폭(AB4 지역 비가중 평균). 계층 = refit 파랑 채운 원, 참조(CQR·앵커 분위수·비계층 풀링) = 빈 회색 원(범례 'Reference interval (d)').
     참조(H17 복사 값)는 AB4 평균을 대상별 행에서 다시 계산한 점 추정(구간 없음). hier2_exact(폭 ∞)는 제외하고 캡션에 쓴다.
     wconf(라벨 있음 n = 3·10, Alaska 대상)는 라벨 0 비교 밖이므로 넣지 않고 캡션에 명시한다.
     방법 약칭 8개는 한 규칙으로 둔다: 점의 가장 왼쪽 요소(가로 CI 왼쪽 끝, 없으면 마커 가장자리)에서 왼쪽 3 pt, 세로 가운데.
     blk·cdf 는 세로 간격(약 2 mm)이 글자 상자 높이보다 작아, 렌더러 기반으로 겹친 쌍만 위아래로 벌린다(_repel_y).

색 의미(모델 계열, 전 그림 METHOD 와 같음): 물리식만 = phys 회흑(o), 직접 ML = direct 회색(x), 물리 앵커 + 잔차 ML = residual 보라(D),
CCI 결합 = cci 녹색(v), 계층 conformal = refit 파랑(o). 스펙 표의 '가족별 색'(H19 잔차 모델을 회색으로 칠하는 안)은 다른 그림의
보라 = 잔차 ML 의미와 충돌하므로 모델 계열 색으로 바꾸고, 가족은 배경 띠와 우측 텍스트로 구분한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _common import (ps, H2, H3, H4, AB4, rd, load_tests_mean, load_deploy, load_c1_tests, load_c2_coverage,
                     save_paper, MissingData, fmt_ci)  # noqa: F401

FIG_H = 140.0

# ---------------------------------------------------------------- 패널 a 행 정의(첫 행이 맨 위)
# (약칭, 검정 이름 또는 None(H23), 조건 목록, 모델 계열, 가족)
A_ROWS = [
    ("H18 LST", "H18a_stefan_lst_vs_stefan", ("noinfo", "covonly"), "phys", "covariates"),
    ("H18x soil", "H18x_stefan_soil_vs_stefan", ("noinfo", "covonly"), "phys", "covariates"),
    ("H19 x14", "H19r_x14_A_resid25_vs_stefan", ("noinfo", "covonly"), "residual", "covariates"),
    ("H19 x16", "H19r_x16_resid25_vs_stefan", ("noinfo", "covonly"), "residual", "covariates"),
    ("H21 E borrowing", "H21_nested_vs_EAK", ("noinfo",), "phys", "coefficient"),
    ("H20 CCI block E", "H20_cci_blockE_vs_stefan", ("noinfo",), "cci", "coefficient"),
    ("H22 IRM", "H22_irm_resid_lam0.25_vs_stefan", ("noinfo", "covonly"), "residual", "objective"),
    ("H22 V-REx", "H22_vrex_resid_lam0.25_vs_stefan", ("noinfo", "covonly"), "residual", "objective"),
    ("H22 DANN", "H22_dann_resid_lam0.25_vs_stefan", ("covonly",), "residual", "objective"),
    ("H23 MAML", None, ("noinfo",), "residual", "objective"),
]
# 공변량만 조건 파선 = 공용 VARIANT_LS['dashed'](3.5, 1.8). 선마다 무늬를 ±25 % 안에서 늘이거나 줄이고 위상을 맞춰
# 양 끝이 대시로 끝나고 마커 밖에 틈이 보이게 한다(조판 도구의 '대시를 경로 끝에 맞춤'과 같은 처리, 검토 2회차 DANN 결함).
COV_DASH = ps.VARIANT_LS["dashed"]
DASH_SCALE = np.round(np.arange(0.75, 1.2501, 0.025), 3)
DASH_END_MIN_PT = 1.2    # 양 끝 대시 최소 길이(pt, 선 굵기 1.2 pt 이상이어야 점이 아니라 대시로 보임)
COND_LS = {"noinfo": "solid", "covonly": COV_DASH}
COND_OFF = {"noinfo": -0.17, "covonly": 0.17}          # 두 조건이 있는 행의 세로 오프셋(행 좌표, 위 = +)
A_XLIM = (-3.0, 4.0)

# ---------------------------------------------------------------- 패널 b 규칙(첫 행이 맨 위)
# (규칙 키, 약칭, 모델 계열, 게이트 변형 여부(그리기에는 쓰지 않고 source data 기록용))
B_RULES = [
    ("stefan", "Physics", "phys", None),
    ("ml_direct", "Direct", "direct", None),
    ("aoa_direct", "Direct+AOA", "direct", "gated"),
    ("ml_resid", "Residual", "residual", None),
    ("aoa_resid", "Residual+AOA", "residual", "gated"),
    ("stefan_cci", "Physics+CCI", "cci", None),
    ("cci_agree20", "CCI gate", "cci", "gated"),
]
B_XLIM = (25.0, 50.0)

# ---------------------------------------------------------------- 패널 c 수준
C_LEVELS = [("stefan_cci", "Equal weight"), ("cci_uncw", "Unc. weight"), ("cci_blocksmooth", "Block smooth"),
            ("gate_gtd_pfr", "GTD·PFR gate")]
C_XLIM = (-3.5, 3.5)

# ---------------------------------------------------------------- 패널 d 방법·약칭·라벨 오프셋(pt, 수동 고정: 스펙 §2.6 함정 4)
D_METHODS = [  # (method, test, 약칭, 계열)
    ("ref_cqr_ak", "ref_h17", "CQR-ak", "direct"),
    ("ref_cqr_iw", "ref_h17", "CQR-iw", "direct"),
    ("ref_anchor_ak", "ref_h17", "anc-ak", "direct"),
    ("ref_anchor_iw", "ref_h17", "anc-iw", "direct"),
    ("pooled", "label0", "pool", "direct"),
    ("hier2_sub", "label0", "sub", "refit"),
    ("hier2_cdf", "label0", "cdf", "refit"),
    ("hier2_blk", "label0", "blk", "refit"),
]
C2_LABEL_GAP_PT = 3.0   # d 약칭: 가장 왼쪽 요소에서 왼쪽 간격(pt), 세로 가운데(모든 약칭 공통 규칙)
# d 계층 conformal = 물리식 앵커 구간이므로 물리식 색(교차 점검 2026-09-26: 파랑은 Fig 2·3·7 에서 E 수축 전용)
CONF_COLOR = ps.COLOR["phys"]
D_XLIM, D_YLIM = (0.30, 1.00), (25.0, 95.0)   # 0.3 부터: CQR-iw 약칭이 마커 왼쪽에 들어갈 자리
# d 참조 구간(H17·비계층 풀링) 회색. METHOD['direct'](#6b7280, L* 약 48)보다 L* 약 9 밝게 두어 그림 전체에서 같은 색값을
# 공유하지 않는다(검토 2회차). 흰 배경 대비 3.48(≥ 3).
REF_COLOR = "#838a96"


def _truthy(s):
    return s.astype(str).str.lower().isin(["true", "1"])


# ================================================================ 자료
def data_a() -> pd.DataFrame:
    """h2 평균 행(is_mean) + H23 MAML 라벨 0 점 추정 → 행·조건별 Δ, CI(strat_AB4)."""
    T = rd(H2 / "h_tests_all.csv")
    T = T[_truthy(T.is_mean)]
    rows = []
    for i, (lab, test, conds, meth, fam) in enumerate(A_ROWS):
        for cond in conds:
            if test is None:
                S = rd(H2 / "h23_summary.csv")
                q = S[(S.method == "maml0") & np.isclose(S.lam, 0.25) & S.target.isin(AB4)]
                if q.target.nunique() != len(AB4):
                    raise MissingData("h2/h23_summary.csv maml0 λ0.25 AB4")
                rows.append(dict(row=i, label=lab, test="H23_maml0_lam0.25_d_phys(AB4 mean)", cond=cond, method=meth, family=fam,
                                 delta=float(q.d_phys.mean()), ci_lo=np.nan, ci_hi=np.nan, ci_kind="none",
                                 source="h2/h23_summary.csv"))
                continue
            q = T[(T.test == test) & (T.cond == cond)]
            if len(q) != 1:
                raise MissingData(f"h2/h_tests_all.csv: {test} {cond} 행 {len(q)}개")
            r = q.iloc[0]
            rows.append(dict(row=i, label=lab, test=test, cond=cond, method=meth, family=fam, delta=r.delta, ci_lo=r.ci_lo,
                             ci_hi=r.ci_hi, ci_kind="strat_AB4", p_holm=r.p_holm, source="h2/h_tests_all.csv"))
    return pd.DataFrame(rows)


def data_b() -> pd.DataFrame:
    G, W = load_deploy()
    G = G.set_index("rule")
    out = []
    for key, lab, meth, var in B_RULES:
        r = G.loc[key]
        worst_reg = max(AB4, key=lambda t: r[f"rmse_{t}"])
        if key not in set(W.rule):
            raise MissingData(f"h3/h30_deploy_worst.csv: rule {key}")
        out.append(dict(rule=key, label=lab, method=meth, mean_AB4=r.mean_AB4, worst_AB4=r.worst_AB4, worst_region=worst_reg,
                        ml_frac_mean=r.ml_frac_mean))
    return pd.DataFrame(out)


def data_c(draft: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    D = load_c1_tests(draft)
    D = D[(D.ref == "stefan") & (D.cond == "noinfo")]
    M, R = [], []
    for m, lab in C_LEVELS:
        q = D[(D.method == m) & D.target.astype(str).str.startswith("MEAN[AB4")]
        if len(q) != 1:
            raise MissingData(f"c1_tests MEAN[AB4] {m}")
        r = q.iloc[0]
        M.append(dict(method=m, label=lab, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_beq=r.delta_beq,
                      ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, p_holm=r.p_holm, ci_kind="block"))
        for t in AB4:
            rt = D[(D.method == m) & (D.target == t)]
            if len(rt) == 1:
                R.append(dict(method=m, label=lab, target=t, delta=rt.iloc[0].delta, ci_scope=rt.iloc[0].ci_scope))
    M, R = pd.DataFrame(M), pd.DataFrame(R)
    M.attrs["source"] = D.attrs.get("source")
    return M, R


def data_d(draft: bool) -> pd.DataFrame:
    """AB4 지역 비가중 평균 커버리지·폭. label0 은 저장된 MEAN6[AB4 4지역] 행(블록 CI), 참조는 대상별 행 평균(점 추정)."""
    C = load_c2_coverage(draft)
    ab4 = set(AB4)
    out = []
    for m, test, lab, fam in D_METHODS:
        if test == "label0":
            q = C[(C.test == "label0") & (C.method == m) & C.target.astype(str).str.startswith("MEAN")]
            q = q[q.target.apply(lambda s: set(s[s.index("[") + 1:-1].split(",")) == ab4)]
            if len(q) != 1:
                raise MissingData(f"c2_coverage label0 MEAN[AB4] {m}")
            r = q.iloc[0]
            out.append(dict(method=m, label=lab, family=fam, coverage=r.coverage, coverage_lo=r.coverage_lo, coverage_hi=r.coverage_hi,
                            width_cm=r.width_cm, width_cm_lo=r.width_cm_lo, width_cm_hi=r.width_cm_hi, n_regions=4, ci_kind="block"))
        else:
            q = C[(C.test == test) & (C.method == m) & C.target.isin(AB4)]
            if q.target.nunique() != 4:
                raise MissingData(f"c2_coverage {test} {m} AB4")
            out.append(dict(method=m, label=lab, family=fam, coverage=q.coverage.mean(), coverage_lo=np.nan, coverage_hi=np.nan,
                            width_cm=q.width_cm.mean(), width_cm_lo=np.nan, width_cm_hi=np.nan, n_regions=4, ci_kind="none"))
    ex = C[(C.test == "label0") & (C.method == "hier2_exact")]
    df = pd.DataFrame(out)
    df.attrs["exact_inf"] = bool(len(ex) and np.isinf(ex.width_cm.astype(float)).all())
    return df


# ================================================================ 그리기 보조
DASH_CHECK: list = []   # 파선 CI 의 보이는 틈 검사 기록(build 에서 검사)


def _pt_per_data(ax):
    """가로축 1 단위의 길이(pt). 축 위치·범위가 정해진 뒤 호출한다."""
    x0, x1 = ax.get_xlim()
    p = ax.transData.transform([(x0, 0), (x1, 0)])
    return (p[1, 0] - p[0, 0]) / (x1 - x0) * 72.0 / ax.figure.dpi


def _dash_phased(ax, lo, hi, x, lw, ms, pattern):
    """짧은 파선 CI 에서 틈이 마커 밑에 숨지 않도록 무늬 배율(DASH_SCALE)과 위상을 고른다.
    조건: (1) 양 끝이 대시 안에 있고 그 대시가 DASH_END_MIN_PT 이상(CI 끝이 보임) (2) 마커 밖에 보이는 틈이 있음.
    조건을 만족하는 (배율, 위상) 중 배율이 1 에 가장 가까운 것, 같으면 min(끝 대시, 보이는 틈)이 큰 것을 고른다.
    반환 = (matplotlib ls, 검사 기록)."""
    _, (d, g) = pattern
    k = _pt_per_data(ax)
    L = (hi - lo) * k                      # 선 길이(pt)
    sm = (x - lo) * k                      # 마커 중심 위치(pt, 왼쪽 끝 기준)
    rm = ms / 2 + ps.LW["marker_edge"] / 2 + 0.3
    stretches = [(0.0, max(0.0, sm - rm)), (min(L, sm + rm), L)]

    def score(f, o):
        D, G = d * lw * f, g * lw * f      # lines.scale_dashes: 무늬는 선 굵기로 곱해진다
        P = D + G
        u0, uL = o % P, (L + o) % P        # 위치 t 의 무늬 좌표 = (t + o) mod P, 대시 = [0, D)
        if u0 >= D or uL >= D:
            return None
        end0, endL = min(D - u0, L), min(uL, L)
        vis = 0.0
        for n in range(-2, int(L / P) + 3):
            g0, g1 = n * P + D - o, (n + 1) * P - o
            for s0, s1 in stretches:
                vis += max(0.0, min(g1, s1) - max(g0, s0))
        return min(end0, endL), vis, G

    cands = []
    for f in DASH_SCALE:
        P = (d + g) * lw * f
        for o in np.arange(0, P, 0.02):
            r = score(f, o)
            if r and r[0] >= DASH_END_MIN_PT and r[1] >= 0.6 * r[2]:
                cands.append((abs(f - 1), -min(r[0], r[1]), f, o, r))
    if not cands:
        raise RuntimeError(f"파선 CI(길이 {L:.1f} pt)에 맞는 무늬 배율·위상이 없음")
    _, _, f, o, (end, vis, G) = min(cands, key=lambda t: (round(t[0], 3), t[1]))
    return (o / lw, (d * f, g * f)), dict(len_pt=round(L, 2), scale=float(f), visible_gap_pt=round(vis, 2),
                                                   end_dash_pt=round(end, 2), gap_pt=round(G, 2))


def _point(ax, x, y, lo, hi, method, ls="solid", filled=True, ms=None, lw=None, alpha=1.0, zorder=3, tag=None):
    """한 점 + 가로 CI(캡 없음). phys 는 마커 'o'. 파선이면 틈이 마커 밖에 보이도록 위상을 맞춘다(축 범위 설정 후 호출)."""
    c = ps.COLOR[method]
    mk = ps.METHOD[method]["marker"] or "o"
    ms = ms or ps.MS["main"] + 0.4
    lw = lw or ps.LW["ci_forest"]
    if np.isfinite(lo) and np.isfinite(hi):
        lsv = ps.VARIANT_LS.get(ls, ls)
        if isinstance(lsv, tuple):
            lsv, rec = _dash_phased(ax, lo, hi, x, lw, ms, lsv)
            DASH_CHECK.append(dict(tag=tag, **rec))
        ax.plot([lo, hi], [y, y], color=c, ls=lsv, lw=lw, solid_capstyle="round",
                dash_capstyle="butt", alpha=alpha, zorder=zorder - 0.5)
    ax.plot([x], [y], ls="none", marker=mk, ms=ms, mec=c, mew=ps.LW["marker_edge"] + (0.3 if mk == "x" else 0),
            mfc=(c if filled else "white"), alpha=alpha, zorder=zorder)


def _rows_axis(ax, labels, fontsize=None, pad=0.6):
    """pad = 첫·끝 행 중심에서 축 경계까지 여백(행 단위). 오프셋 쌍이 있는 패널 c 는 0.7."""
    n = len(labels)
    y = np.arange(n)[::-1].astype(float)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fontsize or ps.FS["tick"])
    ax.set_ylim(-pad, n - 1 + pad)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    return y


# ================================================================ 패널
def panel_a(ax, A):
    y = _rows_axis(ax, [r[0] for r in A_ROWS])
    # 가족 띠: covariates·objective 음영, coefficient 는 띠 없이 텍스트만
    fam_rows = {}
    for i, r in enumerate(A_ROWS):
        fam_rows.setdefault(r[4], []).append(y[i])
    for fam, ys in fam_rows.items():
        ps.group_bands(ax, ys, color=ps.GREY["band"] if fam != "coefficient" else "none", text=fam)
    ax.set_xlim(*A_XLIM)                      # 파선 위상 계산이 축 범위를 쓰므로 먼저 정한다
    for _, r in A.iterrows():
        yy = y[int(r.row)] + (COND_OFF[r.cond] if len(A_ROWS[int(r.row)][2]) == 2 else 0.0)
        _point(ax, r.delta, yy, r.ci_lo, r.ci_hi, r.method, ls=COND_LS[r.cond], tag=f"{r.label}|{r.cond}")
    ax.set_xticks(np.arange(-3, 5, 1))
    ax.xaxis.set_major_formatter(_minus_fmt(0))
    ax.set_xlabel("ΔRMSE vs physics (cm)")
    ps.zero_line(ax, "x")


def panel_b(ax, B):
    y = _rows_axis(ax, list(B.label))
    phys = B[B.rule == "stefan"].iloc[0]
    for v in (phys.mean_AB4, phys.worst_AB4):
        ps.ref_line(ax, v, "phys", axis="x")
    for yi, (_, r) in zip(y, B.iterrows()):
        c = ps.COLOR[r.method]
        # 연결선은 모두 실선(파선 = a 의 '공변량만' 조건으로 범례가 정의하므로 재사용하지 않는다)
        ax.plot([r.mean_AB4, r.worst_AB4], [yi, yi], color=c, lw=ps.LW["main"], ls="-", solid_capstyle="butt", zorder=2)
        _point(ax, r.mean_AB4, yi, np.nan, np.nan, r.method, zorder=3.2)
        ax.plot([r.worst_AB4], [yi], ls="none", marker="|", ms=5.0, mew=1.1, color=c, zorder=3.2)
    ax.set_xlim(*B_XLIM)
    ax.set_xticks(np.arange(25, 51, 5))
    ax.set_xlabel("RMSE (cm)")
    top = len(B) - 0.4
    for v, t in ((phys.mean_AB4, "AB4 mean"), (phys.worst_AB4, "worst region")):
        ax.text(v, top + 0.05, t, ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False)


B_RULES_VAR = [(k, v) for k, _, _, v in B_RULES]


C_OFF = 0.2          # 채운(셀 가중)·빈(블록 등가중) 쌍의 세로 오프셋(행 단위)
C_REG = 0.42         # 지역별 점 띠의 세로 오프셋(채운 마커 줄 바로 위)


def _tint(hex_, f=0.5):
    """색을 흰색과 섞은 불투명 연한 색(반투명 겹침 농도 차이로 점마다 진하기가 달라 보이는 문제 방지)."""
    from matplotlib.colors import to_rgb
    return tuple(1 - f * (1 - c) for c in to_rgb(hex_))


def panel_c(ax, M, R):
    # 첫 행 채운 마커(yi + C_OFF) 위에도 다른 행 간격과 같은 여백을 두도록 pad 를 넓힌다(검토 1회차 지적 1)
    y = _rows_axis(ax, list(M.label), pad=0.7)
    for yi, (_, r) in zip(y, M.iterrows()):
        _point(ax, r.delta, yi + C_OFF, r.ci_lo, r.ci_hi, "cci", ms=ps.MS["mean"] + 0.4)
        _point(ax, r.delta_beq, yi - C_OFF, r.ci_lo_beq, r.ci_hi_beq, "cci", filled=False, lw=ps.LW["ci"] + 0.2)
        rr = R[R.method == r.method]
        # 지역 점: 채운 마커 줄 위 별도 띠(y 오프셋, 스펙 §2.6 c), 불투명 연한 색(떨어진 점도 같은 진하기, CI 선을 끊지 않음)
        ax.plot(rr.delta, np.full(len(rr), yi + C_REG), ls="none", marker="o", ms=ps.MS["point"] + 0.9,
                mfc=_tint(ps.COLOR["cci"], 0.6), mec="none", zorder=2.6)
    ax.set_xlim(*C_XLIM)
    _mark_c_pair(ax, M, y)
    lo_all = np.nanmin(np.r_[R.delta.values, M.ci_lo.values, M.ci_lo_beq.values])
    hi_all = np.nanmax(np.r_[R.delta.values, M.ci_hi.values, M.ci_hi_beq.values])
    if lo_all < C_XLIM[0] + 0.3 or hi_all > C_XLIM[1] - 0.3:   # 모든 값이 축 안쪽 0.3 cm 이상 여백을 갖는지 확인
        raise ValueError(f"panel c 범위 {lo_all:.2f}..{hi_all:.2f} 가 C_XLIM {C_XLIM} 여백 밖")
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.xaxis.set_major_formatter(_minus_fmt(0))
    ax.set_xlabel("ΔRMSE vs physics (cm)")
    ps.zero_line(ax, "x")


def _mark_c_pair(ax, M, y):
    """채운·빈 의미 표지: 두 표지 모두 자기 CI 오른쪽 끝 바깥 C_TAG_PT 에 세로 가운데로 둔다(같은 방향 규칙).
    두 표지가 축 오른쪽 경계 안에 모두 들어가는 첫 행을 렌더러로 고른다."""
    fig = ax.figure
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    bb_ax = ax.get_window_extent(R)
    kw = dict(textcoords="offset points", xytext=(C_TAG_PT, 0), ha="left", va="center", fontsize=ps.FS["annot"],
              color=ps.GREY["text2"], annotation_clip=False)
    for i in range(len(M)):
        r = M.iloc[i]
        t1 = ax.annotate("cell-weighted", (r.ci_hi, y[i] + C_OFF), **kw)
        t2 = ax.annotate("block-equal", (r.ci_hi_beq, y[i] - C_OFF), **kw)
        if max(t.get_window_extent(R).x1 for t in (t1, t2)) <= bb_ax.x1 - 1:
            return i
        t1.remove(); t2.remove()
    raise ValueError("panel c: cell-weighted·block-equal 표지를 둘 행이 없음")


C_TAG_PT = 3.0


def _repel_y(texts, pad_pt=0.6, iters=30):
    """렌더러 상자 기준으로 가로가 겹치고 세로가 겹치는 약칭 쌍만 위아래로 반씩 벌린다(offset points 수정)."""
    if not texts:
        return
    fig = texts[0].figure
    k = 72.0 / fig.dpi
    for _ in range(iters):
        fig.canvas.draw()
        R = fig.canvas.get_renderer()
        bbs = [t.get_window_extent(R) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = bbs[i], bbs[j]
                if a.x1 <= b.x0 or b.x1 <= a.x0:
                    continue
                ov = min(a.y1, b.y1) - max(a.y0, b.y0) + pad_pt / k
                if ov <= 0:
                    continue
                up, dn = (i, j) if (a.y0 + a.y1) >= (b.y0 + b.y1) else (j, i)
                for idx, sgn in ((up, 1), (dn, -1)):
                    dx, dy = texts[idx].xyann
                    texts[idx].xyann = (dx, dy + sgn * ov * k / 2)
                moved = True
        if not moved:
            return


def panel_d(ax, D):
    ax.axvspan(0.85, 0.95, color=ps.GREY["band"], lw=0, zorder=0)
    ax.axvline(0.90, color=ps.COLOR["phys"], lw=ps.LW["ref"], ls=(0, (4, 2)), zorder=1)
    ax.text(0.90, D_YLIM[1], "target", ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False)
    texts = []
    for _, r in D.iterrows():
        ref = r.family != "refit"
        c = REF_COLOR if ref else CONF_COLOR
        if np.isfinite(r.coverage_lo):
            ax.plot([r.coverage_lo, r.coverage_hi], [r.width_cm] * 2, color=c, lw=ps.LW["ci"], solid_capstyle="round", zorder=2)
            ax.plot([r.coverage] * 2, [r.width_cm_lo, r.width_cm_hi], color=c, lw=ps.LW["ci"], solid_capstyle="round", zorder=2)
        if ref:   # 참조 구간 = 빈 회색 원(물리식만 채운 회흑 원, 직접 ML × 와 구분)
            ax.plot([r.coverage], [r.width_cm], ls="none", marker="o", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, zorder=3)
        else:
            ax.plot([r.coverage], [r.width_cm], ls="none", marker="o", ms=ps.MS["main"] + 0.6, mfc=c, mec="white", mew=0.5, zorder=3)
        # 단일 규칙: 가장 왼쪽 요소(가로 CI 왼쪽 끝, 없으면 마커 가장자리)에서 왼쪽 C2_LABEL_GAP_PT, 세로 가운데
        has_ci = np.isfinite(r.coverage_lo)
        x0 = r.coverage_lo if has_ci else r.coverage
        gap = C2_LABEL_GAP_PT + (0 if has_ci else (ps.MS["main"] + 0.6) / 2)
        texts.append(ax.annotate(r.label, (x0, r.width_cm), xytext=(-gap, 0), textcoords="offset points", ha="right", va="center",
                                 fontsize=ps.FS["annot"], color=c if r.family == "refit" else ps.GREY["text2"]))
    ax.set_xlim(*D_XLIM); ax.set_ylim(*D_YLIM)
    ax.set_yticks(np.arange(30, 91, 10))
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    _repel_y(texts)
    ax.xaxis.set_major_formatter(_minus_fmt(1))
    ax.set_xlabel("Empirical coverage of 90 % intervals")
    ax.set_ylabel("Interval width (cm)")


def _minus_fmt(nd):
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, p: ps.fmt_num(v, nd))


def legend_handles():
    ms = ps.MS["main"] + 0.4
    H = [Line2D([], [], ls="none", marker="o", ms=ms, mfc=ps.COLOR["phys"], mec=ps.COLOR["phys"], label="Physics only"),
         Line2D([], [], ls="none", marker="x", ms=ms, mec=ps.COLOR["direct"], mew=0.9, label="Direct ML"),
         Line2D([], [], ls="none", marker="D", ms=ms, mfc=ps.COLOR["residual"], mec=ps.COLOR["residual"], label="Physics + residual ML"),
         Line2D([], [], ls="none", marker="v", ms=ms, mfc=ps.COLOR["cci"], mec=ps.COLOR["cci"], label="CCI combination"),
         Line2D([], [], ls="none", marker="o", ms=ms + 0.6, mfc=CONF_COLOR, mec="white", mew=0.5, label="Physics + hierarchical conformal (d)"),
         Line2D([], [], color="#000000", lw=ps.LW["ci_forest"], label="No target covariates (a)"),
         Line2D([], [], color="#000000", lw=ps.LW["ci_forest"], ls=COV_DASH, label="Target covariates only (a)"),
         Line2D([], [], ls="none", marker="o", ms=ms, mfc="white", mec=REF_COLOR, mew=0.9, label="Reference interval (d)")]
    return H


# ================================================================ 마커 잘림 검사(공용 qa_check 는 텍스트만 본다)
def marker_clip_check(fig, tol_px=0.5):
    """축 안 Line2D 마커의 화면 상자(반지름 = ms/2 pt)가 축 상자를 벗어나는지 본다. 반환 = 잘린 항목 목록."""
    fig.canvas.draw()
    k = fig.dpi / 72.0
    bad = []
    for ai, ax in enumerate(fig.axes):
        bb = ax.get_window_extent()
        for ln in ax.get_lines():
            mk = ln.get_marker()
            if not ln.get_visible() or mk in (None, "None", "none", "", " ") or not ln.get_clip_on():
                continue
            xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
            xy = xy[np.isfinite(xy).all(1)]
            if not len(xy):
                continue
            P = ax.transData.transform(xy)
            r = ln.get_markersize() / 2 * k + ln.get_markeredgewidth() / 2 * k
            for (px, py), (dx, dy) in zip(P, xy):
                if px - r < bb.x0 - tol_px or px + r > bb.x1 + tol_px or py - r < bb.y0 - tol_px or py + r > bb.y1 + tol_px:
                    bad.append(dict(axes=ai, marker=str(mk), x=float(dx), y=float(dy)))
    return bad


# ================================================================ 캡션
def ab4_counts(draft: bool) -> tuple[int, int]:
    """패널 c·d 의 채점 셀·블록 수. c1_tests 의 MEAN[AB4] 행(n_cells·n_blocks)과 c2_coverage 의 label0 MEAN6[AB4] 행
    (n_eval·n_blocks)이 모두 같아야 한다(두 하네스가 같은 채점 블록을 쓰는지 확인). 다르면 예외."""
    D = load_c1_tests(draft)
    q = D[(D.ref == "stefan") & (D.cond == "noinfo") & D.target.astype(str).str.startswith("MEAN[AB4")]
    cells, blocks = set(q.n_cells.astype(int)), set(q.n_blocks.astype(int))
    C = load_c2_coverage(draft)
    ab4 = set(AB4)
    c = C[(C.test == "label0") & C.target.astype(str).str.startswith("MEAN")]
    c = c[c.target.apply(lambda s: set(s[s.index("[") + 1:-1].split(",")) == ab4)]
    cells |= set(c.n_eval.astype(int)); blocks |= set(c.n_blocks.astype(int))
    if len(cells) != 1 or len(blocks) != 1:
        raise ValueError(f"AB4 셀·블록 수 불일치: cells {cells}, blocks {blocks}")
    return cells.pop(), blocks.pop()


def verdicts(CM: pd.DataFrame, D: pd.DataFrame, draft: bool) -> dict:
    """F9·F10 판정 문장에 쓰는 수치(c1_tests MEAN[AB4] Holm p, c2_coverage label0 hier2_cdf 지역별·평균)."""
    uw = CM[CM.method == "cci_uncw"]
    if len(uw) != 1:
        raise MissingData("c1_tests MEAN[AB4] cci_uncw")
    uw = uw.iloc[0]
    C = load_c2_coverage(draft)
    # 원천(알래스카) 행은 제외: F10 은 LORO 6 대상(AB4 + Russia C·Greenland) 기준(c2_coverage test F10 행과 같은 집합)
    reg = C[(C.test == "label0") & (C.method == "hier2_cdf") & ~C.target.astype(str).str.startswith("MEAN")
            & (C.target != "Alaska")].set_index("target")
    inb = reg[(reg.coverage >= 0.85) & (reg.coverage <= 0.95)]
    cdf = D[D.method == "hier2_cdf"].iloc[0]
    pool = D[D.method == "pooled"].iloc[0]
    return dict(n_sig=int((CM.p_holm < 0.05).sum()), n_c1=len(CM), uw=uw, n_in=len(inb), n_reg=len(reg),
                rw=float(reg.loc["Russia_W", "coverage"]), cdf=cdf, pool=pool)


def caption(n_cells: int, n_blocks: int, V: dict):
    uw, cdf = V["uw"], V["cdf"]
    f9 = (f"F9 {'not supported' if V['n_sig'] == 0 else 'supported'}: {V['n_sig']} of {V['n_c1']} CCI combinations "
          f"Holm-significant (uncertainty weight {fmt_ci(uw.delta, uw.ci_lo, uw.ci_hi, 2)} cm). ")
    f10 = (f"F10 partly supported: cdf coverage {cdf.coverage:.2f} [{cdf.coverage_lo:.2f}, {cdf.coverage_hi:.2f}], "
           f"width {cdf.width_cm:.0f} cm; {V['n_in']} of {V['n_reg']} regions within 0.85–0.95 (Russia W {V['rw']:.2f}); "
           f"pool {V['pool'].coverage:.2f}.")
    return dict(
        definition=("Fig. 6 | Zero-label regions. ΔRMSE, method RMSE minus physics-anchor RMSE (Stefan model, Alaska E0), cm; "
                    "negative is better. AB4, unweighted mean over Lena, Canada, Russia W and Russia E. "
                    "H18, Stefan forcing from MODIS LST or ERA5-Land soil temperature; "
                    "H19, residual ML on reduced covariate sets (x14, terrain and climate plus new groups; x16, without SoilGrids); "
                    "H20, E from CCI thaw depth; H21, E borrowed from labelled regions; "
                    "H22, invariance or domain-adversarial objectives; H23, meta-learning."),
        statistics=(f"c, d score {n_cells:,} cells in {n_blocks} blocks (4 AB4 regions). All intervals: 95 %, 1,000 bootstrap resamples; "
                    "a, stratified over regions (H18 to H22); c, d, scoring blocks within regions (at least 8 blocks), "
                    "averaged across regions by resample index. Holm-adjusted p: Supplementary Table 2. " + f10),
        panels=("a, Solid and dashed intervals: no target covariates and target covariates only; residual ML uses λ = 0.25; "
                "H23 is a point estimate on its own evaluation blocks. "
                "b, Filled marker, AB4 mean RMSE; tick, worst AB4 region (Russia W for every rule); dotted lines, physics values; "
                "AOA, area of applicability; CCI gate, CCI where within 20 cm of physics. "
                "c, Filled and open markers, cell-weighted and block-equal AB4 means; light points, single AB4 regions. " + f9 +
                "d, Leave-one-region-out coverage versus mean width; grey band, 0.85–0.95. Filled, two-level hierarchical conformal: "
                "cdf, region CDF pooling (pre-specified primary); sub, one-cell-per-region subsampling; blk, 0.5° blocks as groups. "
                "Open grey, references without CI: pool, cell pooling; H17 CQR and anchor-residual (anc) intervals, "
                "Alaska-calibrated (ak) or density-ratio weighted (iw). Omitted: exact hierarchical (infinite width), "
                "weighted conformal (uses target labels)."),
        data=("Source data: data/processed/h2/h_tests_all.csv, h2/h23_summary.csv, h2/h_deploy_gating.csv, "
              "h3/h30_deploy_worst.csv, h4/c1_tests.csv and h4/c2_coverage.csv."),
    )


# ================================================================ 진입점
def build(draft: bool = False):
    ps.use_paper()
    A = data_a(); B = data_b(); CM, CR = data_c(draft); D = data_d(draft)
    n_cells, n_blocks = ab4_counts(draft)
    DASH_CHECK.clear()

    fig = ps.paper_figure(180, FIG_H)
    # 행 1(축 84–134 mm), 범례 64–72 mm, 행 2(축 12–59 mm), 패널 문자 상단 여백
    ax_a = ps.slot_mm(fig, 0, 84, 28, 60, 50)
    ax_b = ps.slot_mm(fig, 90, 84, 24, 64, 50)
    ax_c = ps.slot_mm(fig, 0, 12, 18, 66, 47)
    ax_d = ps.slot_mm(fig, 88, 12, 12, 78, 47)
    panel_a(ax_a, A); panel_b(ax_b, B); panel_c(ax_c, CM, CR); panel_d(ax_d, D)
    ps.legend_below(fig, legend_handles(), (4, 64.5, 172, 7.5), ncol=4)
    ps.label_panels([ax_a, ax_b, ax_c, ax_d])
    clip = marker_clip_check(fig)
    if clip:
        raise RuntimeError(f"Fig6 마커가 축 경계에서 잘림: {clip}")
    # 파선 CI 는 마커 밖에 틈이 무늬 틈의 60 % 이상 보이고 양 끝 대시가 1 pt 이상이어야 한다
    # (검토 2회차: H22 DANN 파선이 실선으로 보인 결함, 끝 대시가 점처럼 짧으면 CI 끝이 흐려짐)
    short = [d for d in DASH_CHECK if d["visible_gap_pt"] < 0.6 * d["gap_pt"] or d["end_dash_pt"] < DASH_END_MIN_PT]
    if short:
        raise RuntimeError(f"Fig6 파선 CI 틈이 마커에 가려짐: {short}")

    src = {"a": A, "b": B, "c": CM, "c_regions": CR, "d": D}
    return save_paper(fig, "Fig6_label0", caption=caption(n_cells, n_blocks, verdicts(CM, D, draft)), draft=draft, sources=src, spec=dict(
        intent="라벨 0 지역: 우회 기법(H18–H23) 기각, 배포 규칙(H30) 물리식 동급, CCI 결합(C1) 기각(F9), 계층 conformal(C2) 부분 지지(F10)",
        panels=4,
        assets=["data/processed/h2/h_tests_all.csv", "data/processed/h2/h23_summary.csv", "data/processed/h2/h_deploy_gating.csv",
                "data/processed/h3/h30_deploy_worst.csv", "data/processed/h4/c1_tests.csv", "data/processed/h4/c2_coverage.csv"],
        ci_kinds=dict(a="strat_AB4 (H23 MAML: none)", b="none", c="block", d="block (label0), none (H17 references)"),
        layout_mm="180 x 140: row1 axes 84-134 (a 28+60, b 24+64), legend 64-72 (4 cols x 2 rows), row2 axes 12-59 (c 18+66, d 12+78)",
        deviations=["색 = 모델 계열(스펙 가족별 색 대신, 보라 = 잔차 ML 전역 의미 유지)", "b 축 높이 50 mm(스펙 35, 행 1 하단 정렬)",
                    "그림 높이 140 mm(스펙 132, 행 2 패널 문자와 범례 분리; 스펙 문서 §2.6 표는 담당 밖이라 미수정)",
                    "d 에서 wconf 제외(라벨 있음 n = 3·10, 캡션 명시)",
                    "c 에 블록 등가중 AB4 평균(빈 마커) 추가, 채운·빈 표지는 두 표지 모두 CI 오른쪽 끝 바깥(렌더러로 행 선택)",
                    "d 참조 구간 = 빈 회색 원 #838a96(범례 'Reference interval (d)'), direct #6b7280 과 색값 분리",
                    "b 연결선 모두 실선(파선은 a 조건 전용), 게이트 변형은 행 이름으로 구분",
                    "a 파선 = 공용 무늬(3.5, 1.8), 선마다 배율 0.75–1.25·위상을 맞춰 양 끝 대시 ≥ 1.2 pt·마커 밖 틈 보장",
                    "d 계층 conformal = 물리식 색 #4d4d4d(교차 점검: 파랑은 Fig 2·3·7 의 E 수축 전용), 약칭 blk·cdf·sub·pool·anc·CQR·ak·iw 를 캡션에 정의", "d 약칭 단일 규칙(가장 왼쪽 요소 왼쪽 3 pt, 세로 가운데) + 겹친 쌍만 세로 벌림, x 축 0.3 부터"],
        qa_extra=dict(marker_clip="fig6.marker_clip_check: 0 clipped markers (build 에서 실패 시 예외)",
                      dash_gaps=[dict(d) for d in DASH_CHECK],
                      ab4_counts=dict(cells=n_cells, blocks=n_blocks, source="h4/c1_tests.csv MEAN[AB4] = h4/c2_coverage.csv label0 MEAN6[AB4]")),
        placeholders=[]))
