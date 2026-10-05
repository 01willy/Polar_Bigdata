"""Fig 4 v3(재구성 비교판): ML 이득의 조건.

근거 문서
- outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 6절(역할, 배치, 패널 명세, 설명문 초안), 1절(공통 규격), 12절(XA 자리표시)
- design/journal_grade_style_guide.md 2절(그림), 3절(표), 6.4절(Fig 4), 부록 A(audit_v3, pdf_audit)
- 공용 양식은 같은 폴더의 style.py 를 그대로 쓴다(다른 그림 모듈과 공유하므로 고치지 않는다).

패널(170 mm 폭, 배치는 명세 6.2의 슬롯 좌표)
  a  대상 15개의 최소 라벨 수 n*. 재보정(재보정 Stefan − 원천 계수 Stefan)과 재보정을 넘어선 잔차 ML
     (앵커 + 잔차 − 재보정 Stefan). 원천 data/processed/paper_figs/fig4_a.csv. 행은 라벨 10개 편향(WF4-c 진단값) 내림차순.
  b  라벨 10개 편향(x, log10) 대 대상 라벨 교차검증 선정의 원천 계수 Stefan 대비 오차 변화(y, 라벨 전량), 28대상.
     원천 results/rescale_wf/data/processed/wf/wf_tests.csv(WF4-c 지역 행)와 wf_curve.csv(exp wf4, method W, n −1).
  c  대상 라벨 교차검증 선정 − 고정 잔차 레시피(λ 0.25). 원천 wf_tests.csv(WF4-a 풀 평균과 지역 행).
  d  이득 분해(XA, 사후 분석, 2026-10-04 실행): 왼쪽 하위 축 재보정 몫(재보정 Stefan − 원천 계수 Stefan = −G_recal),
     오른쪽 하위 축 재보정을 넘어선 ML 몫(앵커 + 잔차 − 두 재보정 가운데 나은 쪽 = −G_ML2), x 는 b 와 같은 편향.
     원천 data/processed/xbatch/XA_c2_gain_decomposition/sealed/(xa_targets.csv in_main, xa_gains.csv n −1 의 g_cell,
     ρ 와 CI 는 xa_spearman.csv·xa_hyp.csv). 부호는 b 와 같은 오차 변화 규약(음수 = 오차 감소)으로 바꿔 그린다.
     --preview(사후 미리보기, derived_c2_wf4c_split.csv)는 XA 결과 전 비교용으로 남긴 선택지이며 정본에 쓰지 않는다.

명세와 다르게 둔 것([판단], 결과 보고에 적는다)
  1. 북대서양 대상의 모양: 지침 목록에 없다(명세 D-15). 이 그림은 빈 오각형으로 그리고 직접 라벨을 단다.
  2. 중앙 러시아와 북대서양은 직접 라벨, 나머지 지역 모양은 b 안 열쇠(6항목)로 보인다. 명세 b 는 열쇠를 적지 않았으나
     지침 2.5('모양의 뜻은 그림 안 열쇠 한 줄')와 범례 6항목 상한을 함께 지키는 배치다.
  3. d 의 두 하위 축은 y 를 공유하므로(간격 4 mm) y 축 이름은 하나("Error change (cm)")이고, 두 몫의 이름은 하위 축 머리
     "Recalibration", "Beyond recalibration"이 맡는다. 하위 축 폭은 y 눈금 자리를 두려고 33 mm 로 줄였다.
  4. d 의 y 범위는 두 하위 축 공통이며 티베트를 뺀 d 의 값(−13.50 – +4.64 cm)으로 정했다(명세 6.3 d [판단]).
     대상별 CI 는 그리지 않고 Source Data 에 둔다(명세 6.3 d). b 의 직접 라벨(중앙 러시아, 북대서양)은 d 에서 되풀이하지
     않는다(지침 H13, 모양 열쇠는 b 에 한 번).
  5. 그림 높이는 122 mm 이다(명세 표제의 '약 150 mm' 보다 작다). 아래 행을 명세 슬롯(y 82)보다 7.5 mm 올려 두 행 사이
     빈 띠(약 11 mm)를 없앴다. 위 행 슬롯(0–70 mm)과 패널 폭은 명세 그대로다.

실행(저장소 뿌리에서): python scripts/4_visualization/paper_v3/fig4.py
  산출: outputs/figures/paper/v3_restructure/Fig4.{pdf,png}, Fig4_source_data.csv, Fig4_legend.md, Fig4_values.txt
  선택: --medium slide --panel b --out <폴더>(슬라이드판 한 패널, 300 dpi PNG), --preview --out <폴더>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402  (matplotlib Agg 설정 포함)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms as mtrans  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator  # noqa: E402

# ---------------------------------------------------------------- 원천(명세 6.3)
ROOT = S.ROOT
SRC = {
    "fig4_a": ROOT / "data/processed/paper_figs/fig4_a.csv",
    "wf_tests": ROOT / "results/rescale_wf/data/processed/wf/wf_tests.csv",
    "wf_curve": ROOT / "results/rescale_wf/data/processed/wf/wf_curve.csv",
    "wf_meta": ROOT / "results/rescale_wf/data/processed/wf/wf_meta.json",
}
MANIFEST = ROOT / "paper/claims/C2_bias_diagnosis/tables/MANIFEST.csv"
XA_DIR = ROOT / "data/processed/xbatch/XA_c2_gain_decomposition/sealed"      # XA 봉인 표(2026-10-04 21:07 첫 열람)
XA_SRC = {k: XA_DIR / f"{k}.csv" for k in ("xa_targets", "xa_gains", "xa_spearman", "xa_hyp")}
XA_MANIFEST = XA_DIR / "sealed_manifest.json"
PREVIEW_SRC = ROOT / "paper/claims/C2_bias_diagnosis/tables/derived_c2_wf4c_split.csv"
STEM = "Fig4"
INK, AUX = S.INK, S.INK_AUX
MINUS = S.MINUS

# XA 자리표시(명세 6.3 d, 12절, 원고 명세 11절과 글자 하나까지 같음)
XA_PLACEHOLDER = "[XA: Fig 4d | 사후 분석(재현(비맹검)) | 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]"

# ---------------------------------------------------------------- 대상 이름과 모양(지침 2.5, 명세 D-14, D-15)
SUB_PREFIX = {"AL-": "Alaska", "CA-": "Canada", "LE-": "Lena"}
TARGET_NAME = {"Alaska": "Alaska", "Canada": "Canada", "Lena": "Lena Delta", "Russia_W": "W Russia",
               "Russia_E": "E Russia", "Russia_C~lgd": "Central Russia", "NAtlantic~lic": "North Atlantic",
               "Tibet_LGD": "Tibetan Plateau"}
TARGET_REGION = {"Russia_C~lgd": "Russia_C", "NAtlantic~lic": "NAtlantic", "Tibet_LGD": "Tibet"}
MARKER = dict(S.REGION_MARKER)
MARKER["NAtlantic"] = "p"          # [판단] D-15 미정. 이 그림 안에서만 쓰고 직접 라벨로 이름을 보인다
POINT_ESTIMATE = {"NAtlantic~lic"}  # 채점 블록 합집합 6 < 8, 분포 없음(wf_tests ci_flag '점 추정(분포 없음)')
DIRECT_LABEL = ("Russia_C~lgd", "NAtlantic~lic")   # b 에서 직접 라벨을 다는 대상(열쇠 밖 지역)
KEY_REGIONS = ("Alaska", "Canada", "Lena", "Russia_W", "Russia_E")
# b 의 대상별 CI 막대: 27개 막대가 x 5–15 cm 에 몰려 검정이면 점을 가린다. 굵기 대신 색을 옅게 한다(지침 2.3, R-07).
CI_GREY_B = "#a0a0a0"
COL_W = S.METHOD["target_cv_selection"]["color"]   # v4: b·c 의 대상 라벨 교차검증 선정 색(#A33E0B)
# 모양별 크기 보정: 같은 ms 에서 삼각형·십자는 원보다 작게, 사각·마름모는 크게 보인다(면적을 원에 맞춘 [판단] 값)
MS_SCALE = {"o": 1.0, "s": 0.9, "D": 0.85, "^": 1.15, "v": 1.15, "P": 1.25, "p": 1.05}
DODGE_CM, DODGE_MM = 0.15, 1.6        # c: 같은 행 지역 점이 0.15 cm 안이면 1.6 mm 위 단으로 올린다(1.1 mm 는 삼각형끼리 닿았다)


def parent(t: str) -> str:
    for p, r in SUB_PREFIX.items():
        if t.startswith(p):
            return r
    return TARGET_REGION.get(t, t)


def is_sub(t: str) -> bool:
    return any(t.startswith(p) for p in SUB_PREFIX)


def disp(t: str) -> str:
    """그림 글자. 하위 지역 ID 는 Supplementary Table S1 에서 정의한다(지침 2.5)."""
    return t if is_sub(t) else TARGET_NAME[t]


# ---------------------------------------------------------------- 매체별 크기(지침 2.3, 5.3, slide_archetypes 1.3)
TOK = {
    # v4 토큰(design/style_tokens_v4.json lines.paper / lines.slide)
    "paper": dict(lw_main=S.LW["main"], lw_aux=S.LW["aux"], lw_cell=S.LW["ci_forest_cell"], lw_block=S.LW["ci_forest_block"], lw_ref=S.LW["ref"],
                  ms=S.MS["main"], ms_reg=S.MS["region_point"], ms_tri=3.0,
                  mew_open=S.LW["aux"], off_ci=0.8, off_series=0.8, off_region=1.3, arrow_ms=6.0, key_len=3.0, scale=1.0),
    "slide": dict(lw_main=S.LW_SLIDE["main"], lw_aux=S.LW_SLIDE["aux"], lw_cell=S.LW_SLIDE["ci_forest_cell"], lw_block=S.LW_SLIDE["ci_forest_block"],
                  lw_ref=S.LW_SLIDE["ref"], ms=S.MS_SLIDE["main"], ms_reg=S.MS_SLIDE["region_point"], ms_tri=7.0,
                  mew_open=S.LW_SLIDE["aux"], off_ci=1.8, off_series=1.8, off_region=3.0, arrow_ms=13.0, key_len=7.0, scale=16 / 7),
}

# ---------------------------------------------------------------- 배치(mm, 그림 왼쪽 위 원점, 명세 6.2 슬롯 안)
FIG_H = 122.0
GEO = {
    "a": (14.5, 3.0, 58.5, 59.0),     # 슬롯 x 0–75, y 0–70(행 이름 14 mm 포함)
    "b": (99.5, 3.0, 60.5, 59.0),     # 슬롯 x 90–160, y 0–70
    "c": (13.5, 80.0, 64.5, 33.5),    # 슬롯 x 0–80, 아래 행(행 이름과 묶음 머리 포함)
    "dL": (99.5, 80.0, 33.0, 33.5),   # 슬롯 x 90–170, 두 하위 축(간격 4 mm)
    "dR": (136.5, 80.0, 33.0, 33.5),
}
LETTER = {"a": (0.0, 0.0), "b": (90.0, 0.0), "c": (0.0, 74.5), "d": (90.0, 74.5)}
HEAD_Y = 79.2                         # c 열쇠 줄과 d 하위 축 머리의 아래끝(mm)

N_GRID = [3, 10, 40, 160, 320, 1000]  # 명세 1.5 전이 격자
XLIM_A = (2.2, 2000.0)                # 오른쪽 끝 = 미도달 화살표 머리가 닿는 축 끝
XLIM_B, XT_B = (2.5, 50.0), [3, 10, 30]
YLIM_B, YT_B = (-20.0, 13.0), [-20, -10, 0, 10]
YLIM_D, YT_D = (-16.0, 6.5), [-15, -10, -5, 0, 5]   # d: 두 하위 축 공통, 티베트를 뺀 d 값 −13.50 – +4.64 cm
XLIM_C, XT_C = (-3.5, 1.5), [-3, -2, -1, 0, 1]
C_ROWS = [("head", "Four main regions"), (10, "10 labels"), (-1, "All labels"),
          ("head", "Lena Delta and Canada"), (40, "40 labels"), (160, "160 labels")]
C_POOL = {10: "Lena|x,Canada|x,Russia_W|x,Russia_E|x", -1: "Lena|x,Canada|x,Russia_W|x,Russia_E|x",
          40: "Lena|x,Canada|x", 160: "Lena|x,Canada|x"}


def fmt_tick(v) -> str:
    return S.fmt_int(v)


# ---------------------------------------------------------------- 자료
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    tests = pd.read_csv(SRC["wf_tests"])
    curve = pd.read_csv(SRC["wf_curve"])
    meta = json.loads(SRC["wf_meta"].read_text())

    # WF4-c 진단 행(28대상)과 풀 행
    c4 = tests[tests.test_id == "WF4-c"]
    reg = c4[c4.scope == "region"].copy()
    reg[["tname", "mode"]] = reg.target.str.split("|", expand=True)
    mean_c = c4[c4.scope == "MEAN"].iloc[0]
    verdict_c = c4[c4.scope == "verdict"].iloc[0]

    # a: 두 대비, 15대상
    A = pd.read_csv(SRC["fig4_a"])
    A = A[A.contrast.isin(["P1 − P0", "R1 − P1"])].copy()
    A["key"] = A.target + "|" + A["mode"]
    A["censored"] = A.censored.astype(str).str.lower().isin(["true", "1"])
    A["diag_abs"] = A.key.map(reg.set_index("target")["diag_abs"])
    order = (A.drop_duplicates("key").sort_values("diag_abs", ascending=False).key.tolist())
    A["row"] = A.key.map({k: i for i, k in enumerate(order)})
    A = A.sort_values(["row", "contrast"]).reset_index(drop=True)

    # b: 규칙 W, 라벨 전량, 원천 계수 Stefan 대비
    w = curve[(curve.exp == "wf4") & (curve.method == "W") & (curve.n == -1)]
    cols = ["target", "mode", "d_p0", "d_p0_lo", "d_p0_hi", "d_p0_beq", "d_p0_beq_lo", "d_p0_beq_hi",
            "verdict4_p0", "n_splits_valid", "ci_flag"]
    B = reg.drop(columns=["ci_flag"]).merge(w[cols].rename(columns={"target": "tname"}), on=["tname", "mode"],
                                             how="left")
    B["ci_flag"] = B["ci_flag"].fillna(pd.Series(reg.set_index("target").loc[B.target, "ci_flag"].to_numpy(),
                                                  index=B.index))

    # c: WF4-a 풀 평균과 지역 행
    a4 = tests[tests.test_id == "WF4-a"]
    Cm = a4[a4.scope == "MEAN"].copy()
    Cr = a4[a4.scope == "region"].copy()
    Cr[["tname", "mode"]] = Cr.target.str.split("|", expand=True)
    verdict_a = a4[a4.scope == "verdict"].iloc[0]
    X, XS = load_xa()
    return dict(A=A, order=order, B=B, reg=reg, mean_c=mean_c, verdict_c=verdict_c, Cm=Cm, Cr=Cr,
                verdict_a=verdict_a, meta=meta, X=X, XS=XS)


XA_GAINS = {"G_recal": "y_recal", "G_ML2": "y_ml"}     # d 의 두 하위 축(명세 6.3 d, XA 의 G_recal, G_ML2)
XA_HYP = {"XA-1": "G_recal", "XA-2": "G_W", "XA-3": "G_ML2"}   # 등록 가설(Holm 가족 m 3)


def load_xa():
    """XA 봉인 표에서 d 의 점(주 묶음 28대상, 라벨 전량, 셀 가중 점 추정)과 설명문 통계(ρ, 두 CI, 범주)를 읽는다.
    y 는 b 와 같은 오차 변화 규약: y_recal = −G_recal = RMSE(P1) − RMSE(P0), y_ml = −G_ML2 = RMSE(R1) − min(RMSE(P1), RMSE(P2))."""
    t = pd.read_csv(XA_SRC["xa_targets"])
    t = t[t.in_main].copy()
    g = pd.read_csv(XA_SRC["xa_gains"])
    g = g[(g.n == -1) & g.gain.isin(list(XA_GAINS)) & g.target.isin(t.target)]
    X = t[["target", "family", "mode", "point_only", "CE2"]].copy()
    X["tname"] = X.target.str.split("|").str[0]
    for gain, col in XA_GAINS.items():
        r = g[g.gain == gain].set_index("target")
        X[col] = -X.target.map(r.g_cell)                       # 셀 가중 점 추정, 부호 반전
        X[col + "_beq"] = -X.target.map(r.g_beq)
        X[col + "_lo"] = -X.target.map(r.ci_hi_cell)           # 부호 반전이므로 끝값을 바꾼다
        X[col + "_hi"] = -X.target.map(r.ci_lo_cell)
        X[col + "_beq_lo"] = -X.target.map(r.ci_hi_beq)
        X[col + "_beq_hi"] = -X.target.map(r.ci_lo_beq)
        X[col + "_has_dist"] = X.target.map(r.has_dist)
    X = X.rename(columns={"CE2": "diag_abs"}).sort_values("diag_abs").reset_index(drop=True)
    sp = pd.read_csv(XA_SRC["xa_spearman"])
    hy = pd.read_csv(XA_SRC["xa_hyp"])
    XS = {}
    for h, gain in XA_HYP.items():
        r = sp[(sp.hyp == h) & (sp.gain == gain) & (sp.x == "CE2") & (sp.scale == "cm") & (sp.n == -1)
               & (sp.subset == "main")]
        q = hy[hy.hyp == h]
        assert len(r) == 1 and len(q) == 1, f"XA: {h} 행"
        XS[h] = dict(r.iloc[0]) | {k: q.iloc[0][k] for k in ("category", "cat_cluster", "cat_fixed", "holm_p",
                                                              "sentence", "definition_robust", "scale_robust")}
    return X, XS


def checks(D) -> list[str]:
    """그림 값과 원천의 대조(F-13). 실패하면 멈춘다."""
    A, B, Cm, Cr, reg = D["A"], D["B"], D["Cm"], D["Cr"], D["reg"]
    out = []
    assert len(A) == 30 and A.key.nunique() == 15, "a: 15대상 × 2대비"
    assert A.diag_abs.notna().all(), "a: 모든 행에 WF4-c 진단값"
    assert ((~A.censored) == A.n_star.notna()).all(), "a: n* 와 미도달 표시 일치"
    assert len(B) == 28 and B.target.is_unique, "b: WF4-c 지역 행 28"
    assert B.d_p0.notna().all(), "b: 모든 대상에 규칙 W 곡선 행"
    dmax = float((B.d_p0 + B.gain_w).abs().max())
    assert dmax < 1e-9, f"b: d_p0 = −gain_w 불일치 {dmax}"
    out.append(f"b: d_p0 = −gain_w(WF4-c) 28대상 최대 절대 차 {dmax:.2e} cm")
    rho = B[["diag_abs", "gain_w"]].rank().corr().iloc[0, 1]
    assert abs(rho - D["mean_c"].rho) < 1e-9, "b: ρ 재계산 불일치"
    out.append(f"b: ρ(diag_abs, gain_w) 재계산 {rho:.6f} = 파일 {D['mean_c'].rho:.6f}")
    for n, pool in C_POOL.items():
        r = Cm[Cm.n == n]
        assert len(r) == 1 and r.iloc[0].pool_regions == pool, f"c: n {n} 풀 구성"
        regs = Cr[Cr.n == n]
        assert sorted(regs.target) == sorted(pool.split(",")), f"c: n {n} 지역 행"
    out.append("c: 풀 구성(n 10, 전량 = 주 4지역, n 40, 160 = 레나델타·캐나다)과 지역 행 일치")

    # d: XA 봉인 표
    X, XS = D["X"], D["XS"]
    man = {e["file"]: e["sha256"] for e in json.loads(XA_MANIFEST.read_text())["entries"]}
    for k, p in XA_SRC.items():
        assert sha256(p) == man[p.name], f"d: {p.name} sha256 이 봉인 기록과 다르다"
    out.append(f"d: 봉인 표 {len(XA_SRC)}개 sha256 = sealed_manifest.json(기록 2026-10-04 21:06:57)")
    assert len(X) == 28 and set(X.target) == set(B.target), "d: 주 묶음 28대상 = b 의 WF4-c 28대상"
    dx = float((X.set_index("target").diag_abs - B.set_index("target").diag_abs).abs().max())
    assert dx < 1e-9, f"d: CE2 ≠ WF4-c diag_abs {dx}"
    out.append(f"d: x(CE2, xa_targets.csv) = b 의 x(WF4-c diag_abs) 28대상 최대 절대 차 {dx:.2e} cm")
    assert X[[c for c in XA_GAINS.values()]].notna().all().all(), "d: 모든 대상에 두 이득"
    for h, col in (("XA-1", "y_recal"), ("XA-3", "y_ml")):
        rho = pd.concat([X.diag_abs, -X[col]], axis=1).rank().corr().iloc[0, 1]
        assert abs(rho - XS[h]["rho_cell"]) < 1e-9, f"d: {h} ρ 재계산 불일치"
        out.append(f"d: {h} ρ(CE2, 셀 가중 이득) 재계산 {rho:.6f} = xa_spearman.csv {XS[h]['rho_cell']:.6f}")
    for h in XA_HYP:
        assert XS[h]["category"] == "확인하지 못함", f"d: {h} 범주가 바뀌었다. 설명문 문장을 다시 정한다"
    out.append("d: 등록 가설 XA-1, XA-2, XA-3 의 최종 범주 = '확인하지 못함'(xa_hyp.csv category). XA-3 은 주 CI "
               f"'{XS['XA-3']['cat_cluster']}', 보조 CI(대상 고정) '{XS['XA-3']['cat_fixed']}'")
    return out


# ---------------------------------------------------------------- 그리기 도우미
def set_ticks(axis, vals, labels=None):
    axis.set_major_locator(FixedLocator(vals))
    axis.set_major_formatter(FixedFormatter(labels if labels is not None else [fmt_tick(v) for v in vals]))
    axis.set_minor_locator(NullLocator())


def off(ax, dx_mm=0.0, dy_mm=0.0):
    """자료 좌표에 화면 mm 어긋남을 더한 변환(두 가중 막대, 두 계열의 ±0.8 mm)."""
    return mtrans.offset_copy(ax.transData, fig=ax.figure, x=dx_mm / 25.4, y=dy_mm / 25.4, units="inches")


def ax_frac(ax, dx_mm, dy_mm):
    """축 왼쪽 아래 기준 mm → 축 분수 좌표."""
    bb = ax.get_position()
    W, H = ax.figure.get_size_inches() * 25.4
    return dx_mm / (bb.width * W), dy_mm / (bb.height * H)


def text_w_mm(fig, t) -> float:
    r = fig.canvas.get_renderer()
    return t.get_window_extent(r).width / fig.dpi * 25.4


def region_marker(t):
    return MARKER[parent(t)]


def msz(marker, base):
    return base * MS_SCALE.get(marker, 1.0)


def hollow(t):
    return is_sub(t) or t in POINT_ESTIMATE


# ---------------------------------------------------------------- a
def draw_a(ax, D, T):
    A, order = D["A"], D["order"]
    s = T["scale"]
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_A)
    set_ticks(ax.xaxis, N_GRID, [str(v) for v in N_GRID])
    nrow = len(order)
    ax.set_ylim(nrow - 0.45, -1.2)            # 위 여백 = 직접 라벨 자리
    ax.yaxis.set_major_locator(FixedLocator(list(range(nrow))))
    ax.yaxis.set_major_formatter(FixedFormatter([disp(k.split("|")[0]) for k in order]))
    ax.yaxis.set_minor_locator(NullLocator())
    for lab in ax.get_yticklabels():
        lab.set_gid("category")
    ax.tick_params(axis="y", left=False, pad=1.5 * s)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Labels, $n$")

    series = {"P1 − P0": ("recalibrated_stefan", +T["off_series"]),
              "R1 − P1": ("anchor_residual", -T["off_series"])}
    for _, r in A.iterrows():
        mkey, dy = series[r.contrast]
        col = S.METHOD[mkey]["color"]
        tr = off(ax, 0.0, dy)
        y = float(r.row)
        if not r.censored:
            x0 = XLIM_A[0] if r.n_lo == 0 else float(r.n_lo)   # 이전 격자 값(0 이면 축 왼쪽 끝)
            ax.plot([x0, r.n_star], [y, y], color=col, lw=T["lw_main"], solid_capstyle="butt", transform=tr,
                    zorder=3, gid="nstar_interval")
            ax.plot([r.n_star], [y], ls="none", marker="o", ms=T["ms"], mfc=col, mec=col, mew=0, transform=tr,
                    zorder=4)
        else:
            # v4 토큰: 화살촉 없이, 시험한 최대 라벨 수에서 축 끝까지 파선(도달하지 않음)
            ax.plot([float(r.n_max), XLIM_A[1]], [y, y], color=col, lw=T["lw_aux"], ls=(0, (2.0, 1.4)), transform=tr,
                    zorder=3, clip_on=False, gid="nstar_censored")

    # 직접 라벨 3개(각 1회). 위치는 자료에서 고른다.
    rec = A[A.contrast == "P1 − P0"].set_index("row")
    res = A[A.contrast == "R1 − P1"].set_index("row")
    r_top = 0
    assert not rec.loc[r_top].censored and res.loc[r_top].censored      # 첫 행: 재보정 n*, 잔차 미도달
    up = T["off_series"] + 0.35 * T["ms"] / 72 * 25.4 + 0.45 * s           # 점 위끝 위
    ax.text(XLIM_A[0] * 1.12, r_top, "Recalibration", transform=off(ax, 0.0, up), ha="left", va="bottom")
    ax.text(XLIM_A[1], r_top, "Not reached", transform=off(ax, 0.0, 0.35 * s), ha="right", va="bottom")
    r_res = int(res[(~res.censored) & (res.n_star == 3)].index.min())       # 잔차 n* = 3 인 첫 행
    assert r_res > r_top
    ax.text(3.0, r_res, "Residual ML", transform=off(ax, 1.05 * s, -T["off_series"]), ha="left", va="center")


# ---------------------------------------------------------------- b
def draw_b(ax, D, T):
    B = D["B"]
    s = T["scale"]
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_B)
    set_ticks(ax.xaxis, XT_B, [str(v) for v in XT_B])
    ax.set_ylim(*YLIM_B)
    set_ticks(ax.yaxis, YT_B)
    ax.axhspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
    ax.axhline(0.0, color=S.ZERO_LINE["color"], lw=T["lw_ref"], ls="-", zorder=1)
    ax.set_xlabel("Bias at ten labels (cm)")
    ax.set_ylabel("Error change vs source Stefan (cm)")

    inx = B.diag_abs.between(*XLIM_B) & B.d_p0.between(*YLIM_B)
    for _, r in B[inx].iterrows():
        x = float(r.diag_abs)
        if pd.notna(r.d_p0_lo):
            ci = [r.d_p0_lo, r.d_p0_hi, r.d_p0_beq_lo, r.d_p0_beq_hi]
            assert all(YLIM_B[0] < v < YLIM_B[1] for v in ci), f"b: CI 가 축 밖 {r.target}"
            ax.plot([x, x], [r.d_p0_lo, r.d_p0_hi], color=COL_W, alpha=0.35, lw=T["lw_cell"], solid_capstyle="round",
                    zorder=2)
            ax.plot([x, x], [r.d_p0_beq_lo, r.d_p0_beq_hi], color=COL_W, alpha=0.35, lw=T["lw_block"],
                    solid_capstyle="butt", transform=off(ax, T["off_ci"], 0.0), zorder=2)
        h = hollow(r.tname)
        mk = region_marker(r.tname)
        ax.plot([x], [r.d_p0], ls="none", marker=mk, ms=msz(mk, T["ms"]), mfc="white" if h else COL_W,
                mec=COL_W, mew=T["mew_open"] if h else 0.0, zorder=4)

    # 축 밖 대상(티베트): 오른쪽 아래 모서리 삼각 표지 1개(명세 6.3 b). 값은 설명문.
    out = B[~inx]
    assert len(out) == 1 and out.iloc[0].tname == "Tibet_LGD", "b: 축 밖 대상은 티베트 하나"
    o = out.iloc[0]
    assert o.diag_abs > XLIM_B[1] and o.d_p0 < YLIM_B[0]
    fx, fy = ax_frac(ax, 1.0 * s, 1.0 * s)                                   # v4: 삼각 표지 대신 축 끝에 y 값을 숫자로
    x_end = 1 - fx
    near = B[inx & (B.diag_abs > XLIM_B[1] / 1.6)]                         # 오른쪽 아래 구석 가까운 CI 막대(서부 러시아)와 겹치지 않게 그 왼쪽에
    if len(near):
        x_ci = ax.transAxes.inverted().transform(ax.transData.transform((float(near.diag_abs.min()), 0.0)))[0]
        x_end = min(x_end, x_ci - ax_frac(ax, 1.6 * s, 0.0)[0])
    ax.text(x_end, fy, S.fmt_num(o.d_p0, 1), transform=ax.transAxes, ha="right", va="bottom", color=COL_W,
            clip_on=False, zorder=5, gid="offscale")

    # 직접 라벨: 열쇠 밖 지역 두 곳
    for t in DIRECT_LABEL:
        r = B[B.tname == t].iloc[0]
        gap = 1.5 * s
        if t == "Russia_C~lgd":    # 왼쪽(오른쪽은 다른 대상의 막대가 가깝다)
            ax.text(r.diag_abs, r.d_p0, disp(t), transform=off(ax, -gap, 0.0), ha="right", va="center")
        else:
            ax.text(r.diag_abs, r.d_p0, disp(t), transform=off(ax, gap, 0.0), ha="left", va="center")

    # 모양 열쇠(범례 1개, 6항목): 왼쪽 아래 빈 곳, 3행 × 2열
    items = [(MARKER[k], False, disp(k) if k != "Lena" else "Lena Delta") for k in KEY_REGIONS]
    items.append(("o", True, "Sub-region"))
    col_x = [1.9 * s, 20.5 * s]
    row_y = [7.9 * s, 4.9 * s, 1.9 * s]
    for i, (m, h, name) in enumerate(items):
        cx, ry = col_x[i // 3], row_y[i % 3]
        mx, my = ax_frac(ax, cx, ry)
        tx, _ = ax_frac(ax, cx + 1.7 * s, ry)
        ax.plot([mx], [my], transform=ax.transAxes, ls="none", marker=m, ms=msz(m, T["ms"]), mfc="white" if h else INK,
                mec=INK, mew=T["mew_open"] if h else 0.0, clip_on=False, zorder=5)
        ax.text(tx, my, name, transform=ax.transAxes, ha="left", va="center")

    # 방향 표지: v4 토큰에서 화살촉 주석을 쓰지 않는다


# ---------------------------------------------------------------- c
def c_positions():
    pos, y = [], 0.0
    for i, (n, lab) in enumerate(C_ROWS):
        if n == "head" and i > 0:
            y += 0.35
        pos.append(y)
        y += 1.0
    return pos


def draw_c(ax, D, T):
    Cm, Cr = D["Cm"], D["Cr"]
    s = T["scale"]
    pos = c_positions()
    ax.set_xlim(*XLIM_C)
    set_ticks(ax.xaxis, XT_C)
    ax.set_ylim(pos[-1] + 0.55, pos[0] - 0.55)
    ax.axvspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
    ax.axvline(0.0, color=S.ZERO_LINE["color"], lw=T["lw_ref"], ls="-", zorder=1)
    ax.set_xlabel("Error change vs fixed recipe (cm)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, pad=1.5 * s)

    yt, yl = [], []
    head_tr = mtrans.blended_transform_factory(ax.figure.transFigure, ax.transData)
    for (n, lab), y in zip(C_ROWS, pos):
        if n == "head":
            ax.text(0.0, y, lab, transform=head_tr, ha="left", va="center", gid="category", clip_on=False)
            continue
        yt.append(y)
        yl.append(lab)
        m = Cm[Cm.n == n].iloc[0]
        ax.plot([m.ci_lo, m.ci_hi], [y, y], color=COL_W, lw=T["lw_cell"], solid_capstyle="round", zorder=3)
        ax.plot([m.ci_lo_beq, m.ci_hi_beq], [y, y], color=COL_W, lw=T["lw_block"], solid_capstyle="butt",
                transform=off(ax, 0.0, -T["off_ci"]), zorder=3)
        ax.plot([m.delta], [y], ls="none", marker="o", ms=T["ms"], mfc=COL_W, mec=COL_W, mew=0, zorder=4)
        placed = []                                    # (x, 단) 겹침 피하기
        for _, r in Cr[Cr.n == n].sort_values("delta").iterrows():
            tier = 0
            while any(abs(r.delta - px) < DODGE_CM and pt == tier for px, pt in placed):
                tier += 1
            placed.append((r.delta, tier))
            mk = region_marker(r.tname)
            ax.plot([r.delta], [y], ls="none", marker=mk, ms=msz(mk, T["ms_reg"]), mfc=COL_W, mec=COL_W,
                    mew=0, alpha=0.5, transform=off(ax, 0.0, T["off_region"] + tier * DODGE_MM * T["scale"]),
                    zorder=4)
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(yl))
    ax.yaxis.set_minor_locator(NullLocator())
    for lab in ax.get_yticklabels():
        lab.set_gid("category")


def ci_key(fig, x0_mm, y_bottom_mm, T):
    """두 가중 막대와 동등 띠의 열쇠 한 줄(명세 1.2). 그림 mm 좌표."""
    s = T["scale"]
    W, H = fig.get_size_inches() * 25.4
    L = T["key_len"]
    yc = y_bottom_mm - 1.15 * s

    def fx(v):
        return v / W

    def fy(v):
        return 1 - v / H

    x = x0_mm
    fig.add_artist(Line2D([fx(x), fx(x + L)], [fy(yc)] * 2, color=INK, lw=T["lw_cell"], solid_capstyle="round",
                          transform=fig.transFigure))
    x += L + 1.0 * s
    t = fig.text(fx(x), fy(yc), "Cell-weighted", ha="left", va="center")
    x += text_w_mm(fig, t) + 2.8 * s
    fig.add_artist(Line2D([fx(x), fx(x + L)], [fy(yc)] * 2, color=INK, lw=T["lw_block"], solid_capstyle="butt",
                          transform=fig.transFigure))
    x += L + 1.0 * s
    t = fig.text(fx(x), fy(yc), "Block-equal", ha="left", va="center")
    x += text_w_mm(fig, t) + 2.8 * s
    hgt = 2.0 * s
    fig.add_artist(Rectangle((fx(x), fy(yc + hgt / 2)), L / W, hgt / H, facecolor=S.EQUIV_BAND, edgecolor="none",
                             lw=0, transform=fig.transFigure, gid="key_swatch"))
    fig.add_artist(Line2D([fx(x + L / 2)] * 2, [fy(yc + hgt / 2), fy(yc - hgt / 2)], color=S.ZERO_LINE["color"],
                          lw=T["lw_ref"], transform=fig.transFigure))   # 띠 견본 안의 0선(Fig 3 열쇠와 같은 모양)
    x += L + 1.0 * s
    t = fig.text(fx(x), fy(yc), f"±{S.fmt_num(S.EQUIV_HALF_WIDTH_CM, 1)} cm", ha="left", va="center",
                 gid="sizekey")   # 열쇠 값(명세 1.2 의 "±0.5 cm" 견본)
    return x + text_w_mm(fig, t)


# ---------------------------------------------------------------- d(자리표시)
def draw_d(axL, axR, D, T, preview=None):
    """이득 분해(XA): 대상마다 점 하나, b 와 같은 모양·채움 규칙, 왼쪽 재보정 색, 오른쪽 잔차 ML 색(명세 6.3 d).
    맞춤선·대상별 CI 없음(CI 는 Source Data). 축 밖 대상(티베트)은 하위 축마다 검정 삼각 표지 1개(명세 1.5)."""
    s = T["scale"]
    P = D["X"] if preview is None else preview
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlim(*XLIM_B)
        set_ticks(ax.xaxis, XT_B, [str(v) for v in XT_B])
        ax.set_ylim(*YLIM_D)
        set_ticks(ax.yaxis, YT_D)
        ax.axhspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
        ax.axhline(0.0, color=S.ZERO_LINE["color"], lw=T["lw_ref"], ls="-", zorder=1)
    axR.yaxis.set_major_formatter(NullFormatter())     # y 공유: 눈금은 두고 숫자는 왼쪽 하위 축에만
    axL.set_ylabel("Error change (cm)")
    wL = axL.get_position().width
    span = (axR.get_position().x1 - axL.get_position().x0) / wL
    axL.set_xlabel("Bias at ten labels (cm)", x=span / 2)
    for ax, head in ((axL, "Recalibration"), (axR, "Beyond recalibration")):
        _, fy = ax_frac(ax, 0, 0.6 * s)
        ax.text(0.5, 1.0 + fy, head, transform=ax.transAxes, ha="center", va="bottom", gid="category")
    for ax, col, ycol in ((axL, S.METHOD["recalibrated_stefan"]["color"], "y_recal"),
                          (axR, S.METHOD["anchor_residual"]["color"], "y_ml")):
        inx = P.diag_abs.between(*XLIM_B) & P[ycol].between(*YLIM_D)
        # 채운 점을 먼저, 빈 점을 위에 그려 빈 모양의 테두리가 가려지지 않게 한다(같은 자료, 그리는 순서만)
        for _, r in P[inx].assign(_h=P[inx].tname.map(hollow)).sort_values("_h").iterrows():
            h = bool(r._h)
            mk = region_marker(r.tname)
            ax.plot([r.diag_abs], [r[ycol]], ls="none", marker=mk, ms=msz(mk, T["ms"]),
                    mfc="white" if h else col, mec=col, mew=T["mew_open"] if h else 0.0, zorder=4 + h,
                    gid="d_point")
        out = P[~inx]
        assert len(out) == 1 and out.iloc[0].tname == "Tibet_LGD", f"d: 축 밖 대상은 티베트 하나({ycol})"
        o = out.iloc[0]
        fx, fy = ax_frac(ax, 1.0 * s, 1.0 * s)
        if o.diag_abs > XLIM_B[1] and o[ycol] < YLIM_D[0]:          # x, y 모두 밖: 오른쪽 아래 모서리에 y 값(v4: 삼각 표지 대신 숫자)
            x_end = 1 - fx
            near = P[inx & (P.diag_abs > XLIM_B[1] / 1.6) & (P[ycol] < YLIM_D[0] + 0.35 * (YLIM_D[1] - YLIM_D[0]))]
            if len(near):                                                   # 구석 가까운 점(서부 러시아)과 겹치지 않게 그 왼쪽에
                x_pt = ax.transAxes.inverted().transform(ax.transData.transform((float(near.diag_abs.min()), 0.0)))[0]
                x_end = min(x_end, x_pt - ax_frac(ax, 2.2 * s, 0.0)[0])
            ax.text(x_end, fy, S.fmt_num(o[ycol], 1), transform=ax.transAxes, ha="right", va="bottom", color=col,
                    clip_on=False, zorder=6, gid="offscale")
        elif o.diag_abs > XLIM_B[1] and YLIM_D[0] <= o[ycol] <= YLIM_D[1]:   # x 만 밖: 오른쪽 끝, 그 y 에 x 값
            tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1 - fx, o[ycol], S.fmt_num(o.diag_abs, 0), transform=tr, ha="right", va="center", color=col,
                    clip_on=False, zorder=6, gid="offscale")
        else:
            raise AssertionError(f"d: 예상하지 않은 축 밖 위치 {o.diag_abs}, {o[ycol]}")


def load_preview():
    P = pd.read_csv(PREVIEW_SRC)
    P["tname"] = P.target.str.split("|").str[0]
    P["y_recal"] = -P.recal_share          # Δ 규약: 재보정 Stefan − 원천 계수 Stefan
    P["y_ml"] = -P.ml_beyond_R1            # Δ 규약: 앵커 + 잔차(λ 0.25) − 재보정 Stefan(XA 의 G_ML2 와 다름)
    return P


# ---------------------------------------------------------------- 그림 조립
def build_paper(D, preview=None):
    S.use_v3("paper")
    S.mathtext_liberation()        # "$n$" 기울임, Liberation Sans 계열 안(findfont 경고 없음)
    T = TOK["paper"]
    fig = S.fig_mm(S.W2_MM, FIG_H)
    axa = S.axes_mm(fig, *GEO["a"])
    axb = S.axes_mm(fig, *GEO["b"])
    axc = S.axes_mm(fig, *GEO["c"])
    axdL = S.axes_mm(fig, *GEO["dL"])
    axdR = S.axes_mm(fig, *GEO["dR"])
    draw_a(axa, D, T)
    draw_b(axb, D, T)
    draw_c(axc, D, T)
    ci_key(fig, GEO["c"][0], HEAD_Y, T)
    draw_d(axdL, axdR, D, T, preview=preview)
    for k, (x, y) in LETTER.items():
        S.panel_letter(fig, x, y, k)
    return fig


def register_pretendard() -> str:
    """matplotlib 글꼴 목록에 Pretendard 가 없으면 ~/.fonts 의 OTF 를 더한다(style.use_v3('slide') 는 이름만 지정한다)."""
    from matplotlib import font_manager as fm
    if not any(f.name == "Pretendard" for f in fm.fontManager.ttflist):
        for p in sorted((Path.home() / ".fonts").glob("Pretendard-*.otf")):
            fm.fontManager.addfont(str(p))
    hit = fm.findfont(fm.FontProperties(family=["Pretendard"]), fallback_to_default=False)
    if "Pretendard" not in hit:
        raise SystemExit(f"Pretendard 를 찾지 못했다: {hit}")
    return hit


def build_slide(D, panel: str, w_in: float = 7.90, h_in: float = 5.20):
    """슬라이드판 한 패널(배치 크기 그대로, 300 dpi PNG, 지침 5.6 경로 B). 같은 자료·색·범위.
    기본 크기 7.90 × 5.20 in 는 4열 묶음 폭과 본문 영역 높이(지침 5.2)다."""
    register_pretendard()
    S.use_v3("slide")
    T = TOK["slide"]
    fig = plt.figure(figsize=(w_in, h_in))
    W, H = w_in * 25.4, h_in * 25.4
    if panel == "d":
        raise ValueError("d 는 XA 결과 전 자리표시라 슬라이드판을 만들지 않는다")
    left = {"a": 42.0, "b": 26.0, "c": 50.0}[panel]
    top = 4.0 if panel != "c" else 22.0
    ax = S.axes_mm(fig, left, top, W - left - 6.0, H - top - 22.0)
    {"a": draw_a, "b": draw_b, "c": draw_c}[panel](ax, D, T)
    if panel == "c":
        ci_key(fig, left, 12.0, T)
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontsize() < 16:
            t.set_fontsize(16)
    return fig


# ---------------------------------------------------------------- 산출물
def source_table(D) -> pd.DataFrame:
    A, B, Cm, Cr = D["A"], D["B"], D["Cm"], D["Cr"]
    rows = []
    sa = "data/processed/paper_figs/fig4_a.csv; row order: wf_tests.csv WF4-c region diag_abs"
    for _, r in A.iterrows():
        rows.append(dict(panel="a", element="Recalibration" if r.contrast == "P1 − P0" else "Residual ML",
                         target=r.target, mode=r["mode"], label=disp(r.target), row=int(r.row),
                         x_var="labels n*", x=r.n_star, n_lo=r.n_lo, n_max=r.n_max, censored=bool(r.censored),
                         y_var="bias at ten labels (cm), row order", y=r.diag_abs, flag=r.flag, source=sa,
                         note="not reached up to n_max" if r.censored else "interval n_lo < n* <= n*"))
    sb = "results/rescale_wf/data/processed/wf/wf_tests.csv WF4-c region; wf_curve.csv exp wf4 method W n -1"
    for _, r in B.iterrows():
        inside = XLIM_B[0] <= r.diag_abs <= XLIM_B[1] and YLIM_B[0] <= r.d_p0 <= YLIM_B[1]
        rows.append(dict(panel="b", element="target", target=r.tname, mode=r["mode"], label=disp(r.tname),
                         region=parent(r.tname), sub_region=is_sub(r.tname), point_estimate=r.tname in POINT_ESTIMATE,
                         x_var="bias at ten labels (cm)", x=r.diag_abs,
                         y_var="error change vs source Stefan, all labels (cm)", y=r.d_p0,
                         ci_cell_lo=r.d_p0_lo, ci_cell_hi=r.d_p0_hi, y_block=r.d_p0_beq,
                         ci_block_lo=r.d_p0_beq_lo, ci_block_hi=r.d_p0_beq_hi, drawn="in axes" if inside else
                         "outside axes (corner marker)", source=sb))
    sc = "results/rescale_wf/data/processed/wf/wf_tests.csv WF4-a"
    for n, lab in C_ROWS:
        if n == "head":
            continue
        m = Cm[Cm.n == n].iloc[0]
        rows.append(dict(panel="c", element="pool mean", target=m.pool_regions, label=lab, x_var="labels n",
                         x=n, y_var="error change vs fixed recipe (cm)", y=m.delta, ci_cell_lo=m.ci_lo,
                         ci_cell_hi=m.ci_hi, y_block=m.delta_blockeq, ci_block_lo=m.ci_lo_beq,
                         ci_block_hi=m.ci_hi_beq, ci_cell_lo_common=m.ci_lo_c, ci_cell_hi_common=m.ci_hi_c,
                         ci_block_lo_common=m.ci_lo_beq_c, ci_block_hi_common=m.ci_hi_beq_c,
                         non_inferior=bool(m.ni), ci_dependence=m.ci_dependence, source=sc))
        for _, r in Cr[Cr.n == n].iterrows():
            rows.append(dict(panel="c", element="region", target=r.tname, mode=r["mode"], label=disp(r.tname),
                             region=parent(r.tname), x_var="labels n", x=n,
                             y_var="error change vs fixed recipe (cm)", y=r.delta, ci_cell_lo=r.ci_lo,
                             ci_cell_hi=r.ci_hi, y_block=r.delta_blockeq, ci_block_lo=r.ci_lo_beq,
                             ci_block_hi=r.ci_hi_beq, drawn="point only", source=sc))
    X, XS = D["X"], D["XS"]
    xa = "data/processed/xbatch/XA_c2_gain_decomposition/sealed/"
    parts = (("Recalibration", "y_recal", "G_recal", "error change, recalibrated minus source-coefficient Stefan (cm)"),
             ("Beyond recalibration", "y_ml", "G_ML2",
              "error change, anchor plus residual ML minus the better recalibrated Stefan (cm)"))
    for part, col, gain, yvar in parts:
        for _, r in X.iterrows():
            inside = XLIM_B[0] <= r.diag_abs <= XLIM_B[1] and YLIM_D[0] <= r[col] <= YLIM_D[1]
            rows.append(dict(panel="d", element=part, target=r.tname, mode=r["mode"], label=disp(r.tname),
                             region=parent(r.tname), sub_region=is_sub(r.tname),
                             point_estimate=bool(r.point_only), x_var="bias at ten labels (cm)", x=r.diag_abs,
                             y_var=yvar, y=r[col], ci_cell_lo=r[col + "_lo"], ci_cell_hi=r[col + "_hi"],
                             y_block=r[col + "_beq"], ci_block_lo=r[col + "_beq_lo"], ci_block_hi=r[col + "_beq_hi"],
                             drawn="in axes, no interval drawn" if inside else "outside axes (edge or corner marker)",
                             note=f"y = -{gain}; target-level CI not drawn (Source Data only)",
                             source=f"{xa}xa_gains.csv n == -1 & gain == '{gain}' & target == '{r.target}' "
                                    f"(y = -g_cell); x = {xa}xa_targets.csv CE2 (in_main)"))
    for h, part in (("XA-1", "Recalibration"), ("XA-3", "Beyond recalibration")):
        q = XS[h]
        rows.append(dict(panel="d", element=f"Spearman rho, bias vs gain ({part}); legend only", x_var="statistic",
                         y=q["rho_cell"], ci_cell_lo=q["cluster_lo_cell"], ci_cell_hi=q["cluster_hi_cell"],
                         y_block=q["rho_beq"], ci_block_lo=q["cluster_lo_beq"], ci_block_hi=q["cluster_hi_beq"],
                         ci_fixed_cell_lo=q["fixed_lo_cell"], ci_fixed_cell_hi=q["fixed_hi_cell"],
                         ci_fixed_block_lo=q["fixed_lo_beq"], ci_fixed_block_hi=q["fixed_hi_beq"],
                         drawn="legend only",
                         note=("ci_cell/ci_block = family-cluster joint resampling (10,000, primary); ci_fixed_* = "
                               f"target-fixed joint resampling (auxiliary); category '{q['category']}' (cluster "
                               f"'{q['cat_cluster']}', fixed '{q['cat_fixed']}'); post hoc"),
                         source=f"{xa}xa_spearman.csv hyp == '{h}' & x == 'CE2' & scale == 'cm' & n == -1 & "
                                f"subset == 'main'; category from {xa}xa_hyp.csv"))
    T = pd.DataFrame(rows)
    T.loc[T.panel == "a", "x"] = T.loc[T.panel == "a", "x"]
    return T


LEGEND_TITLE = ("The error reduction of label-based correction was rank-correlated with the source-coefficient bias "
                "measured with ten labels.")      # 명세 6.6 첫 문장 그대로(C2 README 1.3 의 좁힌 문구)


def legend_text(D) -> str:
    """설명문 본문(제목 문장 뒤). 문장 틀은 명세 6.6 초안이고 수치는 모두 원천에서 채운다.
    d 는 XA 등록 조각(계획 2.1, '확인하지 못함' 칸)의 영문 수치 문장이다(명세 16절: 설명문에는 수치 문장만, 해석 조각
    전문은 결과 절 R3). 그림에 있는 두 몫(XA-1 재보정, XA-3 재보정을 넘어선 ML)만 쓴다. XA-3 은 주 CI 와 보조 CI(대상 고정)의
    범주가 달라 두 구간을 모두 적고 약한 범주('확인하지 못함')로 쓴다. 모든 XA 문장에 'post hoc analysis'.
    350 단어를 맞추려고 고정 레시피의 정의(재보정 앵커 + 잔차 ML)는 Methods 로 넘겼다(명세 16절 규칙)."""
    mc, Cm, Cr, B, X, XS = D["mean_c"], D["Cm"], D["Cr"], D["B"], D["X"], D["XS"]
    f2 = lambda v: S.fmt_num(v, 2)  # noqa: E731
    tib = B[B.tname == "Tibet_LGD"].iloc[0]
    tx = X[X.tname == "Tibet_LGD"].iloc[0]
    m40, m160 = Cm[Cm.n == 40].iloc[0], Cm[Cm.n == 160].iloc[0]
    ca = Cr[Cr.tname == "Canada"].set_index("n").delta
    le = Cr[Cr.tname == "Lena"].set_index("n").delta
    margin = D["meta"]["ni_margin"]
    nboot = int(mc.nboot)
    assert all(int(XS[h]["nboot"]) == nboot for h in XS), "재표집 횟수가 b·c 와 d 에서 다르다"
    lam = re.search(r"R1\(([\d.]+)\)", Cm.iloc[0].contrast).group(1)
    sgn = lambda v: ("+" if v > 0 else "") + f2(v)  # noqa: E731
    x1, x3 = XS["XA-1"], XS["XA-3"]
    a = ("a, Smallest tested number of labels n* (interval from the previous grid value) at which recalibration "
         "lowered error relative to the source-coefficient Stefan model (Recalibration) and residual ML lowered error "
         "relative to the recalibrated model (Residual ML), under three registered conditions (Methods); Not reached "
         "marks targets without n* up to the largest tested count. Rows follow the ten-label bias of b; sub-regions "
         "(Supplementary Table S1) are not independent.")
    b = (f"b, Error change of the method chosen by cross-validation within the target labels, relative to the "
         f"source-coefficient Stefan model with all labels, against the absolute mean bias of that model at the "
         f"first ten labels (Spearman ρ = {f2(mc.rho)}, 95% CI {f2(mc.ci_lo)} to {f2(mc.ci_hi)}, "
         f"{int(mc.n_targets)} targets; signed bias ρ = {f2(mc.rho_signed)}). The Tibetan Plateau lies outside the "
         f"axes (bias {f2(tib.diag_abs)} cm; error change {f2(tib.d_p0)} cm in b, {f2(tx.y_recal)} and "
         f"{f2(tx.y_ml)} cm in d); the North Atlantic is a point estimate.")
    c = (f"c, Cross-validation selection minus a fixed residual recipe (λ = {lam}). The selection was non-inferior "
         f"(margin {S.fmt_num(margin, 1)} cm) at 40 and 160 labels in the mean of the Lena Delta and Canada, with "
         f"{f2(-m160.delta)} cm lower error at 160 labels; the {f2(-m40.delta)} cm reduction at 40 labels and "
         f"non-inferiority with all labels depend on the split-independence assumption, and non-inferiority did not "
         f"hold at ten labels. The gain came from Canada ({f2(-ca[40])} and {f2(-ca[160])} cm); the Lena Delta "
         f"changed by {sgn(le[40])} and {sgn(le[160])} cm.")
    d = ("d, Error change from recalibration and from residual ML beyond the better recalibrated Stefan model with all "
         "labels, against the same bias. Relations of the bias with the error reduction from recalibration "
         f"(ρ = {f2(x1['rho_cell'])}, 95% CI {f2(x1['cluster_lo_cell'])} to {f2(x1['cluster_hi_cell'])}) and beyond "
         f"recalibration (ρ = {f2(x3['rho_cell'])}, {f2(x3['cluster_lo_cell'])} to {f2(x3['cluster_hi_cell'])}; "
         f"{f2(x3['fixed_lo_cell'])} to {f2(x3['fixed_hi_cell'])} with targets fixed) were not established "
         "(post hoc analysis).")
    tail = (f"Intervals are 95% confidence intervals from {S.fmt_int(nboot)} resamples of 0.5° scoring blocks (plus "
            f"target families in d), weighted as in the key; the grey band marks ±{S.fmt_num(margin, 1)} cm. "
            "Analyses in b and c were designed after earlier results had been inspected.")
    return " ".join([a, b, c, d, tail])


def legend_md(body: str) -> str:
    """Fig2_legend.md 와 같은 형식: '# Fig. 4' 머리, 굵은 제목 문장, 평문 패널 문자."""
    return f"# Fig. 4\n\n**{LEGEND_TITLE}** {body}\n"


def resource_guard(min_gb=None, max_load=40.0, wait_s=60, max_wait_s=1800) -> str:
    min_gb = float(os.environ.get("PAPER_MIN_AVAIL_GB", "30")) if min_gb is None else min_gb   # 기본 30 GB(작업 지시), 조정 담당이 정한 값은 환경 변수로
    """공유 서버 규칙: 가용 메모리 30 GB 미만이거나 부하 40 초과면 기다린다."""
    t0 = time.time()
    while True:
        avail = None
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    avail = int(ln.split()[1]) / 1024 ** 2
        load1 = os.getloadavg()[0]
        if avail is not None and avail >= min_gb and load1 <= max_load:
            return f"자원 확인: 가용 메모리 {avail:.1f} GB, 부하(1분) {load1:.2f}"
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"자원 부족으로 중단: 가용 {avail} GB, 부하 {load1}")
        time.sleep(wait_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medium", choices=["paper", "slide"], default="paper")
    ap.add_argument("--panel", choices=list("abc"), default="b", help="슬라이드판 패널")
    ap.add_argument("--preview", action="store_true", help="d 에 사후 미리보기를 그린다(제출 불가, --out 필수)")
    ap.add_argument("--out", default=None, help="미리보기·슬라이드판 저장 폴더(v3 비교 폴더가 아닌 곳)")
    args = ap.parse_args()

    log = [resource_guard()]
    D = load()
    log += checks(D)

    if args.medium == "slide" or args.preview:
        if not args.out:
            raise SystemExit("--out 이 필요하다(정본 폴더에 미리보기·슬라이드판을 쓰지 않는다)")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.medium == "slide":
            fig = build_slide(D, args.panel)
            p = out / f"{STEM}_{args.panel}_slide.png"
            fig.savefig(p, dpi=300)
        else:
            fig = build_paper(D, preview=load_preview())
            S.save_fig(fig, f"{STEM}_preview", out, formats=("pdf", "png"))
        print("\n".join(log))
        return

    fig = build_paper(D)
    au = S.audit_v3(fig, allowed_num_gids=("scale", "sizekey", "offscale"))
    ov = S.text_overlaps(fig)                           # 글자끼리 겹침 0곳(지침 2.2)
    pdf, png = S.save_fig(fig, STEM, S.OUT, formats=("pdf", "png"))
    pa = S.pdf_audit(pdf)
    ptxt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    pimg = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True).stdout
    extra = dict(overlaps=ov, pdf_codes=S.CODE_RE.findall(ptxt), pdf_dashes=re.findall(r"[—–]", ptxt),
                 n_raster=max(0, len(pimg.strip().splitlines()) - 2))

    # Source Data, 설명문, 값 기록
    ST = source_table(D)
    ST.to_csv(S.OUT / f"{STEM}_source_data.csv", index=False)
    body = legend_text(D)
    leg = LEGEND_TITLE + " " + body                      # 단어 수 = 제목 문장 + 본문('# Fig. 4' 머리 제외)
    nwords = len(leg.split())
    nwords_ph = nwords                                   # XA 결과로 채워 자리표시가 없다
    assert XA_PLACEHOLDER not in leg and "[XA" not in leg, "설명문에 XA 자리표시가 남았다"
    if nwords > 350:
        raise SystemExit(f"설명문 {nwords} 단어 > 350")
    ta = S.text_audit(leg)
    (S.OUT / f"{STEM}_legend.md").write_text(
        legend_md(body) + "\n"
        f"<!-- words: {nwords}; limit 350. Generated by scripts/4_visualization/paper_v3/fig4.py from the registered "
        "sources. Panel d sentence = registered XA fragments (plan 2.1, category 'not established') rendered as "
        "numbers from data/processed/xbatch/XA_c2_gain_decomposition/sealed/ (FIGURE_SPEC_v3 6.3 d, 12, 16). -->\n",
        encoding="utf-8")
    write_values(D, ST, au, pa, nwords, nwords_ph, ta, log, extra)
    print("\n".join(log))
    print("audit_v3 fails:", au["fails"], "chars", au["chars"])
    print("pdf_audit:", {k: pa[k] for k in ("width_mm", "height_mm", "chars", "off_size", "families", "type3")})
    print("legend words:", nwords, nwords_ph, "text_audit:", ta)


def write_values(D, ST, au, pa, nwords, nwords_ph, ta, log, extra=None):
    A, B, Cm, Cr, mc = D["A"], D["B"], D["Cm"], D["Cr"], D["mean_c"]
    man = pd.read_csv(MANIFEST)
    L = ["Fig 4 v3 값 대조 기록", f"생성: {time.strftime('%Y-%m-%d %H:%M')}, scripts/4_visualization/paper_v3/fig4.py",
         "", "## 1. 원천 파일(sha256)"]
    for k, p in SRC.items():
        h = sha256(p)
        rel = str(p.relative_to(ROOT))
        mm_ = man[man.orig_path == rel]
        tag = ("claims MANIFEST 사본과 같음" if len(mm_) and mm_.iloc[0].sha256 == h
               else ("claims MANIFEST 에 없음" if not len(mm_) else "claims MANIFEST 사본과 다름"))
        L.append(f"- {rel}: {h} ({tag})")
    xman = {e["file"]: e["sha256"] for e in json.loads(XA_MANIFEST.read_text())["entries"]}
    for k, p in XA_SRC.items():
        h = sha256(p)
        L.append(f"- {p.relative_to(ROOT)}: {h} (sealed_manifest.json {'과 같음' if xman.get(p.name) == h else '과 다름'})")
    L += ["", "## 2. 자동 대조(스크립트 assert 통과)"] + [f"- {x}" for x in log[1:]]
    L += ["", "## 3. 패널 a(n*, 행 순서 = 라벨 10개 편향 내림차순)"]
    for k in D["order"]:
        rr = A[A.key == k].set_index("contrast")
        def nst(c):
            r = rr.loc[c]
            return f"미도달(n_max {int(r.n_max)})" if r.censored else f"{int(r.n_star)}(구간 {int(r.n_lo)}–{int(r.n_star)})"
        L.append(f"- {disp(k.split('|')[0])} [{k}] 편향 {rr.iloc[0].diag_abs:.2f} cm: 재보정 {nst('P1 − P0')}, "
                 f"잔차 ML {nst('R1 − P1')}")
    L += ["", "## 4. 패널 b(규칙 W − 원천 계수 Stefan, 라벨 전량; 셀 가중 CI / 블록 등가중 CI)"]
    for _, r in B.sort_values("diag_abs", ascending=False).iterrows():
        ci = ("CI 없음(점 추정)" if pd.isna(r.d_p0_lo) else
              f"[{r.d_p0_lo:.2f}, {r.d_p0_hi:.2f}] / [{r.d_p0_beq_lo:.2f}, {r.d_p0_beq_hi:.2f}]")
        L.append(f"- {r.target}: x {r.diag_abs:.2f}, y {r.d_p0:.2f} {ci}")
    L += [f"- 축 범위 x {XLIM_B}, y {YLIM_B}. 축 밖: Tibet_LGD|x 하나(오른쪽 아래 모서리 표지).",
          f"- WF4-c 풀 행: ρ {mc.rho:.6f}, CI [{mc.ci_lo:.6f}, {mc.ci_hi:.6f}], 대상 {int(mc.n_targets)}, "
          f"부호 있는 ρ {mc.rho_signed:.6f}, 재표집 {int(mc.nboot)}, 판정 '{D['verdict_c'].verdict}', "
          f"design_note '{mc.design_note}'"]
    L += ["", "## 5. 패널 c(규칙 W − R1(λ 0.25); 셀 / 블록 등가중 / 공통 재표집 셀·블록)"]
    for n, lab in C_ROWS:
        if n == "head":
            continue
        m = Cm[Cm.n == n].iloc[0]
        L.append(f"- {lab} [{m.pool_regions}]: Δ {m.delta:.4f}, 셀 [{m.ci_lo:.4f}, {m.ci_hi:.4f}], 블록 "
                 f"[{m.ci_lo_beq:.4f}, {m.ci_hi_beq:.4f}], 공통 셀 [{m.ci_lo_c:.4f}, {m.ci_hi_c:.4f}], 공통 블록 "
                 f"[{m.ci_lo_beq_c:.4f}, {m.ci_hi_beq_c:.4f}], ni {m.ni}, verdict4 {m.verdict4}, "
                 f"common {m.verdict4_common}, ci_dependence {m.ci_dependence}")
        for _, r in Cr[Cr.n == n].iterrows():
            L.append(f"    지역 {r.tname}: Δ {r.delta:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}] / "
                     f"[{r.ci_lo_beq:.4f}, {r.ci_hi_beq:.4f}] {r.verdict4}")
    L.append(f"- 판정 행: '{D['verdict_a'].verdict}'")
    L += ["", "## 6. 설명문 수치와 원천(모두 스크립트가 원천에서 채움)"]
    tib = B[B.tname == "Tibet_LGD"].iloc[0]
    L += [f"- ρ 0.38, CI 0.17 to 0.55, 28 targets, signed ρ −0.01: WF4-c MEAN 행 rho {mc.rho:.4f}, "
          f"ci {mc.ci_lo:.4f}–{mc.ci_hi:.4f}, n_targets {int(mc.n_targets)}, rho_signed {mc.rho_signed:.4f} "
          "(C2 README A3, A4 와 같음)",
          f"- 티베트 편향 227.71 cm, 오차 변화 −159.55 cm: diag_abs {tib.diag_abs:.4f}, d_p0 {tib.d_p0:.4f} "
          "(C2 README 2.2 TB9: 진단 227.71, 이득 159.55)",
          f"- 1.32 cm(160), 0.87 cm(40): WF4-a MEAN delta {Cm[Cm.n == 160].delta.iloc[0]:.4f}, "
          f"{Cm[Cm.n == 40].delta.iloc[0]:.4f} (C5 README E2, E3)",
          "- 비열등 40·160(ni True), 전량 ni True 이나 공통 재표집 셀 가중 상한 "
          f"{Cm[Cm.n == -1].ci_hi_c.iloc[0]:.4f} > 0.5(분할 독립 가정 의존, C5 README N4), 10 ni False",
          f"- 캐나다 2.10, 2.72; 레나델타 +0.35, +0.07: WF4-a 지역 행 "
          f"{Cr[(Cr.tname == 'Canada') & (Cr.n == 40)].delta.iloc[0]:.4f}, "
          f"{Cr[(Cr.tname == 'Canada') & (Cr.n == 160)].delta.iloc[0]:.4f}, "
          f"{Cr[(Cr.tname == 'Lena') & (Cr.n == 40)].delta.iloc[0]:.4f}, "
          f"{Cr[(Cr.tname == 'Lena') & (Cr.n == 160)].delta.iloc[0]:.4f} (C5 README E6, E7, E10, E11)",
          f"- 한계 0.5 cm: wf_meta.json ni_margin {D['meta']['ni_margin']}; 재표집 10,000: WF4-c·WF4-a nboot, "
          "XA xa_spearman.csv nboot(같음, assert)",
          f"- 티베트 d 값 {D['X'].set_index('tname').loc['Tibet_LGD', 'y_recal']:.4f}, "
          f"{D['X'].set_index('tname').loc['Tibet_LGD', 'y_ml']:.4f} cm: xa_gains.csv target 'Tibet_LGD|x' n −1 "
          "gain G_recal, G_ML2 의 −g_cell(XA 요약 6절 표: 135.65, 5.89)",
          f"- d 의 ρ 0.37 [−0.11, 0.93]: xa_spearman.csv hyp 'XA-1' CE2 cm n −1 main, rho_cell "
          f"{D['XS']['XA-1']['rho_cell']:.6f}, cluster_lo_cell {D['XS']['XA-1']['cluster_lo_cell']:.6f}, cluster_hi_cell "
          f"{D['XS']['XA-1']['cluster_hi_cell']:.6f} (XA 요약 4절 표, 계획 8.1)",
          f"- d 의 ρ 0.28 [−0.19, 0.80], 대상 고정 0.18 to 0.54: xa_spearman.csv hyp 'XA-3' 같은 필터, rho_cell "
          f"{D['XS']['XA-3']['rho_cell']:.6f}, cluster [{D['XS']['XA-3']['cluster_lo_cell']:.6f}, "
          f"{D['XS']['XA-3']['cluster_hi_cell']:.6f}], fixed [{D['XS']['XA-3']['fixed_lo_cell']:.6f}, "
          f"{D['XS']['XA-3']['fixed_hi_cell']:.6f}]",
          "- 'not established' = xa_hyp.csv category '확인하지 못함'(XA-1, XA-3; XA-2 도 같음. XA-2 의 G_W 는 d 에 그리지 않아 "
          "설명문에서 빼고 결과 절 R3 자리표시로 둔다). XA-3 은 cat_cluster '확인하지 못함', cat_fixed '양(벗어남)' 이라 "
          "두 구간을 모두 적고 약한 범주로 썼다(계획 2.1 공통 규칙). 블록 등가중 ρ 와 CI 는 Source Data 와 12절",
          f"- 설명문 단어 수 {nwords}, 한도 350. text_audit: {ta}"]
    L += ["", "## 7. audit_v3(저장 직전 그림)"]
    for k in ("fails", "sizes", "chars", "thin_lines", "titles", "boxed_text", "codes", "comma4", "long_labels",
              "loose_numbers"):
        L.append(f"- {k}: {au[k]}")
    L += ["", "## 8. pdf_audit(Fig4.pdf)"]
    for k in ("width_mm", "height_mm", "chars", "sizes", "off_size", "lt5", "families", "type3"):
        L.append(f"- {k}: {pa[k]}")
    L += ["", "## 9. Source Data 행 수", f"- {ST.groupby('panel').size().to_dict()}"]
    L += ["", "## 10. 명세와 다르게 둔 것([판단], 모듈 머리말과 같음)",
          "- 북대서양 대상의 모양: 지침 2.5 목록에 없다(명세 D-15). 빈 오각형 + 직접 라벨. 빈 마커의 뜻은 '하위 지역 또는 점 추정"
          " 대상' 하나(명세 6.5).",
          "- 중앙 러시아와 북대서양은 직접 라벨, 나머지 5지역 모양은 b 안 열쇠(6항목, 지침 2.8 상한)로 보인다.",
          "- d 의 두 하위 축은 y 를 공유하므로 y 축 이름은 하나('Error change (cm)'), 두 몫의 이름은 하위 축 머리가 맡는다."
          " 하위 축 폭 33 mm(명세 36 mm, y 눈금 자리).",
          f"- d 의 y 범위 {YLIM_D}(두 하위 축 공통)는 티베트를 뺀 d 의 값 범위(재보정 −13.50 – +4.64 cm, 재보정 너머"
          " −1.26 – +0.93 cm)로 정했다(명세 6.3 d [판단]). b 의 범위(−20, 13)보다 좁혀 오른쪽 하위 축의 0 근처 점을 펼쳤다.",
          "- d 의 부호: 봉인 표의 이득 G(양수 = 개선)를 −G 로 바꿔 b 와 같은 오차 변화 규약(음수 = 오차 감소)으로 그렸다"
          "(명세 6.3 d 의 축 이름 'Error change …'). 설명문의 ρ 는 봉인 표 그대로 편향과 이득 G 의 순위상관이다.",
          "- d 의 점: b 와 같은 모양·채움 규칙(빈 모양 = 하위 지역 또는 점 추정), 색은 왼쪽 #2b5c8f, 오른쪽 #9a7bc9(명세)."
          " 맞춤선 없음, 대상별 CI 는 Source Data 만. 티베트 표지: 왼쪽은 x·y 모두 밖이라 b 와 같은 모서리 표지, 오른쪽은"
          " x 만 밖(y −5.89 cm)이라 그 높이의 오른쪽 끝 표지. b 의 직접 라벨 두 개는 d 에서 되풀이하지 않는다(H13).",
          "- d 의 설명문은 등록 조각(계획 2.1 '확인하지 못함' 칸)의 영문 수치 문장이다. 등록 문서에 영문 조각이 없어 국문 조각을"
          " 옮겼다('…의 관계는 확인하지 못했다(사후 분석)' → 'Relations … were not established (post hoc analysis)'). 해석 조각"
          " 전문과 XA-2 문장은 결과 절 R3 에 둔다(명세 16절).",
          f"- 그림 높이 {FIG_H:g} mm(명세 '약 150 mm'). 아래 행을 명세 슬롯(y 82)보다 위로 올려 행 사이 빈 띠를 없앴다."
          " 위 행 슬롯과 패널 폭은 명세 그대로.",
          "- b 의 대상별 CI 막대는 검정 대신 #a0a0a0(27개 막대가 x 5–15 cm 에 몰려 검정이면 점을 가린다. 지침 2.3 '굵기 대신"
          " 색을 옅게').",
          "- a 의 하위 지역 ID(AL-1 등)는 Supplementary Table S1 정의를 전제로 쓴다(지침 2.5). 설명문이 그 표를 가리킨다.",
          "- a 의 x 축 이름 'Labels, n' 의 n 은 기울임(지침 2.2 한 글자 변수). mathtext 글꼴은 style.mathtext_liberation().",
          "- n* 의 등록 조건 3개(블록 CI 상한 < 0, 분할 승률 ≥ 2/3, 반복 승률 ≥ 0.75)와 구간 보고 규칙은 CAPTIONS.md"
          " 'v2/Fig4_min_labels' 에서 옮겼다. fig4_a.csv 는 v2 등록 표(scripts/4_visualization/paper/v2_data.py)로"
          " claims MANIFEST 에는 없다.",
          "- 설명문은 Fig2_legend.md 형식('# Fig. 4' 머리, 굵은 제목 문장, 평문 패널 문자). 명세 6.6 초안에 더한 것은"
          " 하위 지역 표 참조, 북대서양 점 추정 표지, 고정 레시피의 λ 뿐이다. XA 치환 뒤 350 단어를 맞추려고 고정 레시피의"
          " 정의(재보정 앵커 + 잔차 ML)를 Methods 로 넘기고, 티베트의 d 값을 b 문장에 합치고, 구간 문장을 b·c·d 하나로"
          " 묶었다(명세 16절 규칙)."]
    if extra:
        L += ["", "## 11. 추가 점검(저장 뒤)",
              f"- 글자끼리 겹침 {len(extra['overlaps']['overlap_pairs'])}쌍 {extra['overlaps']['overlap_pairs']}, 캔버스 밖"
              f" 글자 {extra['overlaps']['outside']} (글자 {extra['overlaps']['n_texts']}개)",
              f"- pdftotext 내부 코드 정규식 {len(extra['pdf_codes'])}건 {extra['pdf_codes']}, 연결어 대시"
              f" {len(extra['pdf_dashes'])}건",
              f"- pdfimages 래스터 요소 {extra['n_raster']}개",
              "- 사람 점검(600 dpi PNG 100 % 잘라 보기): a 행 이름·직접 라벨 3개·화살표 머리 축 끝 닿음, b 직접 라벨 2개와 막대"
              " 분리, c 겹친 지역 점의 단 간격 1.6 mm(2026-10-04). d(XA, 2026-10-05): 왼쪽 W Russia 점과 티베트 모서리 표지"
              " 분리, 오른쪽 0 근처 점 겹침은 자료 그대로(빈 모양을 위에 그려 테두리가 보임), 티베트 오른쪽 끝 표지 y −5.89 cm."
              " 글자 겹침·잘림 없음."]
    X, XS = D["X"], D["XS"]
    L += ["", "## 12. 패널 d(XA, 사후 분석; 원천 data/processed/xbatch/XA_c2_gain_decomposition/sealed/)",
          "- 필터: x = xa_targets.csv in_main 의 CE2(= WF4-c diag_abs, 2절 대조), y = xa_gains.csv n == -1 & gain ∈"
          " {G_recal, G_ML2} & target 의 g_cell 에 −1 을 곱한 값. 대상 28(계열 8). 아래 값은 그림 값(x, y_recal, y_ml) 그대로.",
          "- 형식: 대상: x | 재보정 y(셀) [대상 CI] / 블록 등가중 y | 재보정 너머 y(셀) [대상 CI] / 블록 등가중 y | 그림 상태"]
    for _, r in X.sort_values("diag_abs", ascending=False).iterrows():
        def part(c):
            ci = (f"[{r[c + '_lo']:.2f}, {r[c + '_hi']:.2f}]" if bool(r[c + "_has_dist"]) else "CI 없음(점 추정)")
            return f"{r[c]:.4f} {ci} / {r[c + '_beq']:.4f}"
        inside = all(XLIM_B[0] <= r.diag_abs <= XLIM_B[1] and YLIM_D[0] <= r[c] <= YLIM_D[1] for c in ("y_recal", "y_ml"))
        L.append(f"- {r.target}: x {r.diag_abs:.4f} | {part('y_recal')} | {part('y_ml')} | "
                 f"{'축 안' if inside else '축 밖(삼각 표지)'}")
    for h in ("XA-1", "XA-2", "XA-3"):
        q = XS[h]
        L.append(f"- {h} ({XA_HYP[h]}): ρ 셀 {q['rho_cell']:.6f} / 블록 {q['rho_beq']:.6f}; 계열 군집 CI 셀 "
                 f"[{q['cluster_lo_cell']:.4f}, {q['cluster_hi_cell']:.4f}] / 블록 [{q['cluster_lo_beq']:.4f}, "
                 f"{q['cluster_hi_beq']:.4f}]; 대상 고정 CI 셀 [{q['fixed_lo_cell']:.4f}, {q['fixed_hi_cell']:.4f}] / 블록 "
                 f"[{q['fixed_lo_beq']:.4f}, {q['fixed_hi_beq']:.4f}]; 대상 {int(q['n_targets'])}, 계열 {int(q['n_families'])},"
                 f" 재표집 {int(q['nboot'])}; 범주 '{q['category']}'(주 '{q['cat_cluster']}', 보조 '{q['cat_fixed']}'), "
                 f"Holm p {q['holm_p']}; 정의 강건 {q['definition_robust']}, 척도 강건 {q['scale_robust']}")
    L.append("- 등록 조각(xa_hyp.csv sentence): " + " / ".join(str(XS[h]["sentence"]) for h in ("XA-1", "XA-2", "XA-3")))
    (S.OUT / f"{STEM}_values.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
