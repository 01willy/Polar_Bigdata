"""논문 그림 공용 로더·기록기·QA(스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §1.6·§5·§6 단계 3).

그림 모듈(fig1.py … fig7.py, table1.py, supp.py)은 다음만 쓴다.
    from _common import *            # 경로 상수, ps(paperstyle), load_*, order_by_abslogE, save_paper, MissingData
    def build(draft: bool = False):
        ps.use_paper()
        C = load_h25_curve()                      # h25b 가 있으면 블록 CI 판, 없으면 h25(행 재표집 CI → ci_rep_*)
        fig = ps.paper_figure(180, 118)
        ...
        return save_paper(fig, "Fig3_label_budget", caption=dict(definition=..., statistics=..., panels=..., data=...),
                          spec=dict(intent=..., panels=5, assets=[...]), draft=draft, sources={"a": df_a})

자료 규약
  - 모든 로더는 표준 열 이름을 덧붙인다: blk_d_phys, blk_d_phys_lo, blk_d_phys_hi, split_win, rep_win, ci_rep_lo, ci_rep_hi, ci_kind.
    ci_kind ∈ {'block', 'rep_row', 'strat_AB4', 'none'}(§1.6). df.attrs['source'] = 읽은 파일, df.attrs['is_smoke'] = 스모크 여부.
  - h4 본 실행 파일이 없으면 draft=True 일 때만 *_smoke_* 로 대체한다. draft=False 에서 없으면 MissingData 예외.
  - 절단(미달성) n 은 NaN + censored=True + n_max. 대입값을 만들지 않는다.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from polar import paperstyle as ps          # noqa: E402

PROC = ROOT / "data" / "processed"
H2, H3, H4, M1 = PROC / "h2", PROC / "h3", PROC / "h4", PROC / "m1"
OUT = ROOT / "outputs" / "figures" / "paper"
QA_DIR = OUT / "_qa"
SRC_DIR = OUT / "source_data"
CAPTIONS = OUT / "CAPTIONS.md"
FIG_SPEC = ROOT / "figures" / "figure_spec.json"
for _d in (OUT, QA_DIR, SRC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]
TARGETS14 = ["Lena", "Canada", "Russia_W", "Russia_E", "AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6", "CA-2", "CA-3", "LE-1", "LE-2"]
LEVEL_THRESH = 0.15                            # |log E비| ≥ 0.15 = 수준 오차 지역(스펙 §2.2 c)
QA_OPTS = dict(sheets=False)                   # paper_figs.py --qa 가 True 로 바꾼다(100 % 확인 시트·pdffonts 상세)
CI_STD = ["blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "split_win", "rep_win", "ci_rep_lo", "ci_rep_hi", "ci_kind"]

__all__ = [n for n in dir() if not n.startswith("_")]


class MissingData(FileNotFoundError):
    """필요한 자료 파일이 없다(그림 모듈은 잡아서 건너뛰거나 _draft 로 저장)."""


# ================================================================ 0) 파일 도우미
def rd(path, required: bool = True) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        if required:
            raise MissingData(str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p))
        return None
    df = pd.read_csv(p)
    df.attrs["source"] = str(p.relative_to(ROOT))
    df.attrs["is_smoke"] = "_smoke" in p.name
    return df


def h4_path(tag: str, kind: str, draft: bool = False, ext: str = "csv") -> Path:
    """h4 본 실행 파일 <tag>_<kind>.<ext>. 없고 draft=True 이면 <tag>_smoke_<kind>.<ext>. 둘 다 없으면 MissingData."""
    main = H4 / f"{tag}_{kind}.{ext}"
    if main.exists():
        return main
    smoke = H4 / f"{tag}_smoke_{kind}.{ext}"
    if draft and smoke.exists():
        return smoke
    raise MissingData(f"{main.relative_to(ROOT)}" + ("" if draft else " (스모크 대체는 --draft 에서만)"))


def load_h4(tag: str, kind: str, draft: bool = False) -> pd.DataFrame:
    """h4 요약 CSV(본 실행 우선, draft 면 스모크 대체). 표준 CI 열은 load_a2_curve 등 전용 로더가 붙인다."""
    return rd(h4_path(tag, kind, draft))


def _std_cols(df: pd.DataFrame, est=None, lo=None, hi=None, sw=None, rw=None, rlo=None, rhi=None, kind="block") -> pd.DataFrame:
    """원천 열 이름 → 표준 CI 열. 원천 열이 없으면 NaN, 값이 모두 NaN 이면 ci_kind='none'."""
    def g(c):
        return df[c] if c is not None and c in df.columns else np.nan
    df["blk_d_phys"], df["blk_d_phys_lo"], df["blk_d_phys_hi"] = g(est), g(lo), g(hi)
    df["split_win"], df["rep_win"] = g(sw), g(rw)
    df["ci_rep_lo"], df["ci_rep_hi"] = g(rlo), g(rhi)
    df["ci_kind"] = np.where(pd.notna(df["blk_d_phys_lo"]), kind, "none") if kind == "block" else kind
    return df


# ================================================================ 1) 완료 실험(h2·h3·m1) 로더
def load_h25_curve(prefer: str = "h25b") -> pd.DataFrame:
    """H25 라벨 예산 곡선. h3/h25b_curve.csv(블록 CI 재실행)가 있으면 우선, 없으면 h3/h25_curve.csv.
    표준 열: blk_d_phys(_lo/_hi) = 블록 CI(h25b) 또는 NaN(h25), ci_rep_lo/hi = 행 재표집 CI, ci_kind.
    attrs['ci_kind'] = 'block' | 'none'(h25: 그림에 CI 를 그리지 않는다, 스펙 §2.3)."""
    pb = H3 / f"{prefer}_curve.csv"
    if pb.exists():
        df = rd(pb)
        df = _std_cols(df, "d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep")
        df.attrs["ci_kind"] = "block"
    else:
        df = rd(H3 / "h25_curve.csv")
        df = _std_cols(df, None, None, None, None, None, "d_phys_lo", "d_phys_hi", kind="none")
        df.attrs["ci_kind"] = "none"
    df.attrs["source"] = str((pb if pb.exists() else H3 / "h25_curve.csv").relative_to(ROOT))
    return df


def h25_series(C: pd.DataFrame, target: str, stage: str, rule: str = "kmedoid", lam: float = 0.0, alpha: float = 1.0,
               scope: str = "n") -> pd.DataFrame:
    """곡선 한 계열(n 오름차순). 분할 수가 다른 중복 행이 있으면 최대 분할 행만."""
    q = C[(C.target == target) & (C.stage == stage) & (C.rule == rule) & (C.scope == scope)
          & np.isclose(C.lam, lam) & np.isclose(C.alpha, alpha)]
    if "n_splits" in q and len(q):
        q = q[q.n_splits == q.n_splits.max()]
    return q.sort_values("n")


def h25_allA_ref(C: pd.DataFrame, target: str, stage: str = "S3", lam: float = 0.25, alpha: float = 1.0) -> float:
    """전량 A 라벨 참조 Δ(분할 평균). 참조이며 상한이 아니다."""
    q = C[(C.target == target) & (C.scope == "allA") & (C.stage == stage) & np.isclose(C.lam, lam) & np.isclose(C.alpha, alpha)]
    return float(q.d_phys_mean.mean()) if len(q) else np.nan


def load_h25_breakeven(prefer: str = "h25b") -> pd.DataFrame:
    p = H3 / f"{prefer}_breakeven.csv"
    return rd(p) if p.exists() else rd(H3 / "h25_breakeven.csv")


def load_h25_targets() -> pd.DataFrame:
    """대상별 E_own·E0(분할 평균)·|log E비|·셀·블록 수(분할 평균)."""
    T = rd(H3 / "h25_targets.csv")
    g = T.groupby("target").agg(parent=("parent", "first"), E_own=("E_own", "first"), E0=("E0", "mean"), n_A=("n_A", "min"),
                                n_eval=("n_eval", "mean"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first")).reset_index()
    g["E_ratio"] = g.E_own / g.E0
    g["abslogE"] = np.abs(np.log(g.E_ratio))
    g.attrs["source"] = T.attrs["source"]
    return g


def abslogE_map() -> dict:
    """대상 → |log(E_own/E0)|. 우선 h3/h27_loo_S3.csv(abs_logE), 없는 대상은 h25_targets 로 보충."""
    out = {}
    t = load_h25_targets()
    out.update(dict(zip(t.target, t.abslogE)))
    p = H3 / "h27_loo_S3.csv"
    if p.exists():
        L = pd.read_csv(p)
        out.update(dict(zip(L.target, L.abs_logE)))
    return out


def order_by_abslogE(targets, abslogE: dict | pd.Series | None = None, descending: bool = False) -> list:
    """|log E비| 오름차순(스펙 §1.2 포레스트 정렬). 값이 없는 대상은 끝으로. forest() 는 첫 원소를 맨 위에 그린다."""
    m = dict(abslogE) if abslogE is not None else abslogE_map()
    key = lambda t: (not np.isfinite(m.get(t, np.nan)), m.get(t, np.inf) * (-1 if descending else 1))   # noqa: E731
    return sorted(dict.fromkeys(targets), key=key)


def error_type(target: str, abslogE: dict | None = None) -> str:
    """'level'(|log E비| ≥ 0.15) 또는 'structure'."""
    m = abslogE or abslogE_map()
    v = m.get(target, np.nan)
    return "level" if np.isfinite(v) and v >= LEVEL_THRESH else "structure"


def load_recipe_table(H: str = "H13", cond: str | None = "labels") -> pd.DataFrame:
    """h4/a1_recipe_table.csv 한 가설. 표준 CI 열: ci_kind = 평균 행 strat_AB4, 지역 행 rep_row."""
    R = rd(H4 / "a1_recipe_table.csv")
    q = R[R.H == H].copy()
    if cond is not None:
        q = q[q.cond == cond]
    q["ci_kind"] = np.where(q.target.astype(str).str.startswith(("REGION_SUMMARY", "MEAN")), "strat_AB4", "rep_row")
    q.attrs.update(R.attrs)
    return q


LADDER_LABEL = {
    "covonly: Stefan 유사라벨 − none (직접 catboost_lo, r=10)": "vs no pseudo-labels",
    "covonly: Stefan 유사라벨 − const (직접 catboost_lo, r=10)": "vs constant",
    "covonly: Stefan 유사라벨 − const_t (직접 catboost_lo, r=10)": "vs constant + TDD",
    "covonly: Stefan 유사라벨 − shuffle (직접 catboost_lo, r=10)": "vs shuffled",
    "covonly: Stefan 유사라벨 − tddlin (직접 catboost_lo, r=10)": "vs linear TDD",
}


def load_ladder() -> pd.DataFrame:
    """h4/a1_ladder.csv + 영문 행 라벨(row_label). 사전에 없는 label 이 있으면 실패(한국어 누출 방지)."""
    L = rd(H4 / "a1_ladder.csv")
    bad = [l for l in L.label if l not in LADDER_LABEL]
    if bad:
        raise KeyError(f"LADDER_LABEL 에 없는 사다리 라벨: {bad}")
    L["row_label"] = L.label.map(LADDER_LABEL); L["ci_kind"] = "strat_AB4"
    return L


def load_decomp() -> pd.DataFrame:
    D = rd(H4 / "a1_decomp.csv")
    D["error_type"] = np.where(D.abslogE >= LEVEL_THRESH, "level", "structure")
    return D


def load_m1_tests(valid_only: bool = True) -> pd.DataFrame:
    """정정본 m1/m1_sc_tests.csv. valid_only=True 면 ci_valid 행만(점 추정이 자기 CI 집합과 같은 지역 기준)."""
    T = rd(M1 / "m1_sc_tests.csv")
    if valid_only and "ci_valid" in T:
        T = T[T.ci_valid.astype(str).str.lower().isin(["true", "1"])].copy()
    T["ci_kind"] = np.where(T.target.astype(str).str.startswith(("REGION_SUMMARY", "MEAN")), "strat_AB4", "rep_row")
    return T


def load_tests_mean(confirmatory: bool | None = True, source: str = "h_tests_all") -> pd.DataFrame:
    """h2/h_tests_all.csv(또는 h_tests_main) 평균 행(is_mean). ci_kind = strat_AB4."""
    T = rd(H2 / f"{source}.csv")
    q = T[T.is_mean.astype(str).str.lower().isin(["true", "1"])].copy()
    if confirmatory is not None and "confirmatory" in q:
        q = q[q.confirmatory.astype(str).str.lower().isin(["true", "1"]) == confirmatory]
    q["ci_kind"] = "strat_AB4"
    q.attrs.update(T.attrs)
    return q


def load_h28_tests() -> pd.DataFrame:
    T = rd(H3 / "h28_tests.csv")
    T["ci_kind"] = np.where(T.target.astype(str).str.startswith("MEAN"), "strat_AB4", "rep_row")
    return T


def load_subregions() -> pd.DataFrame:
    return rd(H3 / "subregions.csv")


def load_h27b() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(규칙 통계 h27b_rule, 대상별 h27b_targets). be_n 은 미달성 시 대입값이므로 censored 행에서 NaN 으로 바꾼다."""
    R = rd(H3 / "h27b_rule.csv"); T = rd(H3 / "h27b_targets.csv")
    if "censored" in T:
        c = T.censored.astype(str).str.lower().isin(["true", "1"])
        T["censored"] = c
        T.loc[c, "be_n"] = np.nan
    return R, T


def load_h29_blocks(prefer_b: bool = True) -> pd.DataFrame:
    """블록 라벨 가치(대상 × 블록, 분할 평균). h3/h29b_blocks.csv 우선, 없으면 h29_block_value.csv 를 (target, block) 평균."""
    p = H3 / "h29b_blocks.csv"
    if prefer_b and p.exists():
        return rd(p)
    V = rd(H3 / "h29_block_value.csv")
    g = V.groupby(["target", "block"], as_index=False).agg(n_splits=("split", "nunique"), n_cells=("n_cells", "mean"), lat=("lat", "first"),
                                                           lon=("lon", "first"), value_S1=("value_S1", "mean"), value_S3=("value_S3", "mean"),
                                                           E_block_minus_E0=("E_block_minus_E0", "mean"), kmedoid_freq=("kmedoid_freq", "mean"))
    g.attrs["source"] = V.attrs["source"]
    return g


def load_h29_summary(prefer_b: bool = True) -> pd.DataFrame:
    p = H3 / "h29b_summary.csv"
    return rd(p) if prefer_b and p.exists() else rd(H3 / "h29_summary.csv")


def load_deploy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(h2/h_deploy_gating 규칙 요약, h3/h30_deploy_worst 대상별)."""
    return rd(H2 / "h_deploy_gating.csv"), rd(H3 / "h30_deploy_worst.csv")


# ================================================================ 2) 진행 중 실험(h4) 로더
def _merge_blk(df: pd.DataFrame, tag: str, on: list, draft: bool) -> pd.DataFrame:
    """<tag>_blk.csv(h4_analysis 재집계)의 표준 열로 덮어쓴다(있을 때만)."""
    name = f"{tag}_smoke_blk.csv" if df.attrs.get("is_smoke") else f"{tag}_blk.csv"
    p = H4 / name
    if not p.exists():
        return df
    B = pd.read_csv(p)
    on = [c for c in on if c in B.columns and c in df.columns]
    cols = [c for c in CI_STD if c in B.columns]
    keep = df.drop(columns=[c for c in cols if c in df.columns])
    out = keep.merge(B[on + cols].drop_duplicates(on), on=on, how="left")
    out.attrs.update(df.attrs); out.attrs["blk_source"] = str(p.relative_to(ROOT))
    return out


def load_a2_curve(draft: bool = False) -> pd.DataFrame:
    """A2 곡선(h4/a2_curve.csv). 표준 CI: d_phys_mean/lo/hi = 블록 부트스트랩(h31 규약), *_rep = 행 재표집."""
    df = load_h4("a2", "curve", draft)
    df = _std_cols(df, "d_phys_mean", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep")
    return _merge_blk(df, "a2", ["target", "e_treat", "spread", "scope", "n", "lam", "stage"], draft)


def minn_pass(df: pd.DataFrame, hi="blk_d_phys_hi", sw="split_win", rw="rep_win") -> np.ndarray:
    """점별 최소 n 세 조건(블록 CI 상한 < 0, 분할 승률 ≥ 2/3, 반복 승률 ≥ 0.75) 충족 여부(Fig 4a–d 채움)."""
    from polar.h4_common import SPLIT_WIN_MIN, REP_WIN_MIN
    return ((df[hi] < 0) & (df[sw] >= SPLIT_WIN_MIN - 1e-12) & (df[rw] >= REP_WIN_MIN - 1e-12)).to_numpy()


def load_a2_minn(draft: bool = False) -> pd.DataFrame:
    """A2 최소 n(주 정의 n_star, censored, n_max). 구형(minn_ci 센티널 −1) 파일은 censored = minn_ci < 0 으로 변환하고
    attrs['legacy'] = True(rep_row 단일 조건이므로 _draft 전용). range_limited = censored & n_max < 40."""
    df = load_h4("a2", "minn", draft)
    if "n_star" not in df and "minn_ci" in df:
        df["censored"] = df.minn_ci < 0
        df["n_star"] = np.where(df.censored, np.nan, df.minn_ci)
        df.attrs["legacy"] = True
    df["censored"] = df.censored.astype(str).str.lower().isin(["true", "1"])
    df.loc[df.censored, "n_star"] = np.nan
    df["range_limited"] = df.censored & (df.n_max < 40)
    if "abs_logE_ratio" not in df:
        m = abslogE_map(); df["abs_logE_ratio"] = df.target.map(m)
    return df


def load_a2_spread(draft: bool = False) -> pd.DataFrame:
    """배치 효과 Δ(분산 − 집중), 블록 CI(h31 산출 a2_spread 또는 h4_analysis 의 a2_spread_tests)."""
    for kind in ("spread_tests", "spread"):
        try:
            df = load_h4("a2", kind, draft)
            df["ci_kind"] = "block"
            return df
        except MissingData:
            continue
    raise MissingData("h4/a2_spread_tests.csv")


def load_b2_pooling(draft: bool = False) -> pd.DataFrame:
    """B2 부분 풀링. 표준 CI ← d_phys_blk/lo/hi(블록), ci_rep_phys_* (행 재표집)."""
    df = load_h4("b2", "pooling", draft)
    df = _std_cols(df, "d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win_phys", "rep_win_phys", "ci_rep_phys_lo", "ci_rep_phys_hi")
    return _merge_blk(df, "b2", ["target", "est", "variant", "rule", "lam", "n"], draft)


def load_b3(kind: str = "summary", draft: bool = False) -> pd.DataFrame:
    """B3 라벨 설계. kind ∈ {summary, evar, f7, region, selection}. summary 는 blk_d_phys*·blk_d_rand* 블록 열을 그대로 쓴다."""
    df = load_h4("b3", kind, draft)
    if kind == "summary":
        df = _std_cols(df, "blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "blk_d_phys_split_win", "blk_d_phys_rep_win",
                       "d_phys_ci_rep_lo", "d_phys_ci_rep_hi")
    return df


def load_b4(kind: str = "summary", draft: bool = False) -> pd.DataFrame:
    df = load_h4("b4", kind, draft)
    if kind == "summary":
        df = _std_cols(df, "d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "ci_rep_lo", "ci_rep_hi")
    return df


def load_b1(draft: bool = False) -> dict:
    """B1 2단계 프로토콜. {'summary', 'protocol', 'meta', 'tau_sel'}.
    tau_sel: meta 의 tau_sel_loto 키, 없으면 protocol@loto 행 존재 시 'loto'(대상별 LOTO 선택), 둘 다 없으면 None(대체 금지)."""
    S = load_h4("b1", "summary", draft)
    S = _std_cols(S, "d_phys_block", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep")
    P = load_h4("b1", "protocol", draft)
    meta = json.loads(h4_path("b1", "meta", draft, "json").read_text())
    tau = meta.get("tau_sel_loto")
    if tau is None and (S.method == "protocol@loto").any():
        tau = "loto"
    return dict(summary=S, protocol=P, meta=meta, tau_sel=tau)


D1_EXPLORATORY = {"offset_only"}              # 결과 열람 후 추가한 탐색 행(사전 등록 F11 밖): 회색·캡션 표기


def load_d1(draft: bool = False) -> pd.DataFrame:
    """D1 외부 홀드아웃 요약. 표준 CI ← d_blk/_lo/_hi(블록). 상대 Δ(% of physics RMSE) 열 rel_* 추가.
    exploratory = method ∈ D1_EXPLORATORY. 라벨 정의 교락(F4_calm_temp vs F4_direct)은 캡션에 명시할 것."""
    df = load_h4("d1", "summary", draft)
    est = "d_blk" if "d_blk" in df else "d_phys_mean"
    lo = "d_blk_lo" if "d_blk_lo" in df else None
    hi = "d_blk_hi" if "d_blk_hi" in df else None
    rlo = "d_phys_lo_rep" if "d_phys_lo_rep" in df else "d_phys_lo"
    rhi = "d_phys_hi_rep" if "d_phys_hi_rep" in df else "d_phys_hi"
    df = _std_cols(df, est, lo, hi, "split_win" if "split_win" in df else None, "rep_win" if "rep_win" in df else None, rlo, rhi)
    for c, s in (("rel_d", "blk_d_phys"), ("rel_lo", "blk_d_phys_lo"), ("rel_hi", "blk_d_phys_hi")):
        df[c] = df[s] / df.rmse_phys * 100.0
    df["exploratory"] = df.method.isin(D1_EXPLORATORY)
    try:
        df.attrs["prediction"] = json.loads(h4_path("d1", "prediction", draft, "json").read_text())
    except MissingData:
        df.attrs["prediction"] = None
    return df


def load_c1_tests(draft: bool = False) -> pd.DataFrame:
    """C1 CCI 결합 검정. ci_lo/hi = 블록(ci_scope 에 따름), ci_rep_* = 보조."""
    df = load_h4("c1", "tests", draft)
    df["ci_kind"] = np.where(df.target.astype(str).str.startswith("MEAN"), "strat_AB4", "block")
    return df


def load_c2_coverage(draft: bool = False) -> pd.DataFrame:
    df = load_h4("c2", "coverage", draft)
    df["is_inf"] = df.is_inf.astype(str).str.lower().isin(["true", "1"]) if "is_inf" in df else False
    return df


# ================================================================ 3) 라벨 형식
def fmt_num(v, nd: int = 1) -> str:
    return ps.fmt_num(v, nd)


def fmt_ci(est, lo, hi, nd: int = 1) -> str:
    """'−0.7 [−1.0, −0.3]'(U+2212)."""
    return f"{fmt_num(est, nd)} [{fmt_num(lo, nd)}, {fmt_num(hi, nd)}]"


def fmt_nstar(n_star, censored: bool, n_max, range_limited: bool | None = None) -> str:
    """'10', '> 320 (max tested)', '> 10 (max tested, range-limited)'."""
    if not censored and np.isfinite(n_star):
        return f"{int(n_star)}"
    rl = (n_max < 40) if range_limited is None else range_limited
    return f"> {int(n_max)} (max tested{', range-limited' if rl else ''})"


def region_label(name: str, abslogE: float | None = None) -> str:
    return ps.region_label(name, abslogE)


def region_labels(targets, abslogE: dict | None = None) -> list:
    m = abslogE if abslogE is not None else abslogE_map()
    return [ps.region_label(t, m.get(t)) for t in targets]


# ================================================================ 4) 캡션 기록기
CAPTION_BUDGET = dict(definition=80, statistics=80, panels=150, data=40)
CAPTION_MAX = 350
_WORD = re.compile(r"[A-Za-z0-9Ͱ-Ͽ−%]+(?:[.'’\-][A-Za-z0-9]+)*")


def word_count(s: str) -> int:
    return len(_WORD.findall(s or ""))


def caption_text(caption: str | dict) -> tuple[str, dict]:
    """dict(definition, statistics, panels, data) 또는 문자열 → (본문, 블록별 단어 수)."""
    if isinstance(caption, str):
        return caption.strip(), dict(total=word_count(caption))
    parts = [caption.get(k, "").strip() for k in ("definition", "statistics", "panels", "data")]
    wc = {k: word_count(caption.get(k, "")) for k in CAPTION_BUDGET}
    txt = " ".join(p for p in parts if p)
    wc["total"] = word_count(txt)
    return txt, wc


def caption_problems(caption: str | dict) -> list:
    """단어 예산 초과·em dash·한글·하이픈 음수 등 캡션 규칙 위반 목록."""
    txt, wc = caption_text(caption)
    bad = []
    if wc["total"] > CAPTION_MAX:
        bad.append(f"caption words {wc['total']} > {CAPTION_MAX}")
    for k, b in CAPTION_BUDGET.items():
        if wc.get(k, 0) > b:
            bad.append(f"caption block '{k}' {wc[k]} > {b}")
    if "\u2014" in txt:
        bad.append("em dash in caption")
    for m in re.finditer("–", txt):                           # en dash 는 숫자 사이만
        a, b = txt[max(0, m.start() - 1):m.start()], txt[m.end():m.end() + 1]
        num = a.isdigit() and (b.isdigit() or b == "\u2212")
        pre, post = txt[m.start() - 2:m.start() - 1], txt[m.end() + 1:m.end() + 2]
        panel = bool(re.fullmatch(r"[a-h]", a) and re.fullmatch(r"[a-h]", b) and not pre.isalpha() and not post.isalpha())
        if not (num or panel):                               # 숫자 범위(0–20)와 패널 범위(a–d, 3a–d)만 허용
            bad.append(f"en dash outside numeric range: '{txt[max(0, m.start() - 8):m.end() + 8]}'")
    if re.search(r"[가-힣]", txt):
        bad.append("Hangul in caption")
    return bad


@contextlib.contextmanager
def _locked(path: Path):
    lock = QA_DIR / f".{path.name}.lock"                    # 잠금 파일은 _qa 아래에 둔다
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_caption(name: str, caption: str | dict, title: str | None = None) -> dict:
    """CAPTIONS.md 의 '## <name>' 절을 덮어쓴다(없으면 추가, 파일 잠금). 절 끝에 단어 수 주석. 반환 단어 수 dict."""
    txt, wc = caption_text(caption)
    wtxt = ", ".join(f"{k} {v}" for k, v in wc.items())
    body = f"## {name}\n\n{('**' + title + '** ') if title else ''}{txt}\n\n<!-- words: {wtxt} -->\n"
    with _locked(CAPTIONS):
        cur = CAPTIONS.read_text() if CAPTIONS.exists() else "# Figure captions\n"
        if not cur.startswith("# "):
            cur = "# Figure captions\n\n" + cur          # 구형(절 없는) 파일은 머리말만 붙이고 유지
        pat = re.compile(rf"^## {re.escape(name)}\n.*?(?=^## |\Z)", re.S | re.M)
        new = pat.sub(body + "\n", cur) if pat.search(cur) else cur.rstrip("\n") + "\n\n" + body
        CAPTIONS.write_text(new)
    return wc


def update_figure_spec(entry: dict) -> None:
    """figures/figure_spec.json 의 figures[] 에서 id 가 같은 항목을 교체(없으면 추가). 파일 잠금."""
    with _locked(FIG_SPEC):
        d = json.loads(FIG_SPEC.read_text())
        figs = d.setdefault("figures", [])
        for i, f in enumerate(figs):
            if f.get("id") == entry["id"]:
                figs[i] = {**f, **entry}
                break
        else:
            figs.append(entry)
        FIG_SPEC.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")


def save_source_data(fig_name: str, panel: str, df: pd.DataFrame) -> Path:
    """그림에 찍힌 값 그대로 outputs/figures/paper/source_data/<Fig>_<panel>.csv."""
    p = SRC_DIR / f"{fig_name.split('_')[0]}_{panel}.csv"
    df.to_csv(p, index=False)
    return p


# ================================================================ 5) QA
def _texts(fig):
    """보이는 비어 있지 않은 Text 목록(그림·축·눈금·범례·주석)."""
    from matplotlib.text import Text
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        out.append(t)
    return out


def _is_title(t, ax) -> bool:
    return t in (ax.title, ax._left_title, ax._right_title)


def pdf_fonts(pdf_path) -> list:
    """pdffonts 출력 → [{name, type, emb}]."""
    try:
        out = subprocess.run(["pdffonts", str(pdf_path)], capture_output=True, text=True, timeout=30).stdout.splitlines()[2:]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = []
    for ln in out:
        f = ln.split()
        if len(f) >= 5:
            emb_i = next((i for i, v in enumerate(f) if v in ("yes", "no")), None)
            rows.append(dict(name=f[0], type=" ".join(f[1:emb_i - 1]) if emb_i else f[1], emb=f[emb_i] if emb_i else "?"))
    return rows


def pdf_text(pdf_path) -> str:
    try:
        return subprocess.run(["pdftotext", "-q", str(pdf_path), "-"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def qa_check(fig, pdf_path=None, caption: str | dict | None = None, expect_width_mm: float = ps.W2_MM,
             overlap_tol_px: float = 0.5, ignore_overlap_gids: tuple = ("inset",)) -> dict:
    """그림 규격·충돌·문법 검사. 반환 dict(ok, fails, …).
    - width_mm/height_mm: 저장 PDF MediaBox(있으면) 또는 캔버스, 폭 ±1 mm, 높이 ≤ 200 mm
    - min_font_pt: 보이는 글자 최소 크기(≥ 6.5)
    - overlaps: 텍스트 상자 교차 쌍(허용 오차 overlap_tol_px, 같은 축 눈금 라벨끼리도 검사)
    - clipped: 캔버스 밖으로 나간 글자·축 장식
    - has_title_in_axes: 축 제목(set_title) 사용 여부(금지)
    - delta_axes_have_zero: gid 'delta_x'/'delta_y' 축에 0선(gid 'delta_zero') 존재
    - fonts_ok / unicode_minus_ok: pdffonts(Type 42·Liberation/Arial 만, DejaVu 금지), pdftotext 에서 '-숫자' 부재
    - caption_words / caption_problems: 단어 예산(≤ 350)·em dash·한글"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = ps.fig_mm(fig)
    res = dict(width_mm=round(W, 2), height_mm=round(H, 2))
    if pdf_path and Path(pdf_path).exists():
        try:
            head = Path(pdf_path).read_bytes()
            m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", head)
            if m:
                res["width_mm"] = round(float(m.group(1)) / 72 * 25.4, 2); res["height_mm"] = round(float(m.group(2)) / 72 * 25.4, 2)
        except OSError:
            pass
    fails = []
    if abs(res["width_mm"] - expect_width_mm) > 1.0:
        fails.append(f"width {res['width_mm']} mm")
    if res["height_mm"] > ps.HMAX_MM + 0.5:
        fails.append(f"height {res['height_mm']} mm")
    texts = _texts(fig)
    sizes = [float(t.get_fontsize()) for t in texts]
    res["min_font_pt"] = round(min(sizes), 2) if sizes else np.nan
    small = [(t.get_text()[:20], t.get_fontsize()) for t in texts if t.get_fontsize() < ps.MIN_FONT_PT - 1e-6]
    if small:
        fails.append(f"font < {ps.MIN_FONT_PT} pt: {small[:5]}")
    # 텍스트 상자
    fb = fig.bbox
    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(r)
        except Exception:                                   # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        ax = t.axes
        skip = ax is not None and ax.get_gid() in ignore_overlap_gids
        boxes.append((t, bb, skip))
    clipped = []
    for t, bb, _ in boxes:
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            clipped.append(t.get_text()[:30])
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        tb = ax.get_tightbbox(r)
        if tb is not None and (tb.x0 < fb.x0 - 1 or tb.y0 < fb.y0 - 1 or tb.x1 > fb.x1 + 1 or tb.y1 > fb.y1 + 1):
            clipped.append(f"axes@{ax.get_position().bounds[0]:.2f},{ax.get_position().bounds[1]:.2f}")
    res["clipped"] = clipped
    if clipped:
        fails.append(f"clipped {len(clipped)}: {clipped[:4]}")
    ov = []
    tol = overlap_tol_px
    for (t1, b1, s1), (t2, b2, s2) in combinations(boxes, 2):
        if s1 or s2:
            continue
        if b1.x0 + tol < b2.x1 - tol and b2.x0 + tol < b1.x1 - tol and b1.y0 + tol < b2.y1 - tol and b2.y0 + tol < b1.y1 - tol:
            ov.append((t1.get_text()[:24], t2.get_text()[:24]))
    res["overlaps"] = ov; res["n_text_overlaps"] = len(ov)
    if ov:
        fails.append(f"text overlaps {len(ov)}: {ov[:4]}")
    # 축 제목·Δ 0선
    res["has_title_in_axes"] = any(_is_title(t, t.axes) for t in texts if t.axes is not None)
    if res["has_title_in_axes"]:
        fails.append("axes title present (move to caption or use condition label via ax.text)")
    miss0 = []
    for ax in fig.axes:
        gid = ax.get_gid() or ""
        if gid.startswith("delta_") and not any((l.get_gid() == "delta_zero") for l in ax.get_lines()):
            miss0.append(gid)
    res["delta_axes_have_zero"] = not miss0
    res["n_delta_axes"] = sum(1 for ax in fig.axes if (ax.get_gid() or "").startswith("delta_"))
    if miss0:
        fails.append(f"delta axes without zero line: {len(miss0)}")
    # 범례 수
    n_leg = len(fig.legends) + sum(1 for ax in fig.axes if ax.get_legend() is not None)
    res["n_legends"] = n_leg
    if n_leg > 1:
        fails.append(f"legends {n_leg} > 1")
    # PDF 글꼴·마이너스
    if pdf_path and Path(pdf_path).exists():
        fonts = pdf_fonts(pdf_path)
        res["fonts"] = sorted({f"{f['name']}|{f['type']}|{f['emb']}" for f in fonts})
        badf = [f for f in fonts if ("DejaVu" in f["name"]) or not re.search(r"Liberation|Arial", f["name"])
                or "TrueType" not in f["type"] or f["emb"] != "yes"]
        res["fonts_ok"] = bool(fonts) and not badf
        if not res["fonts_ok"]:
            fails.append(f"fonts: {[f['name'] + '/' + f['type'] for f in badf][:4] or 'none found'}")
        txt = pdf_text(pdf_path)
        hy = re.findall(r"(?<![A-Za-z0-9])-\d", txt)
        res["unicode_minus_ok"] = not hy
        if hy:
            fails.append(f"hyphen-minus numbers in PDF text: {hy[:5]}")
    if caption is not None:
        _, wc = caption_text(caption)
        res["caption_words"] = wc
        cp = caption_problems(caption)
        res["caption_problems"] = cp
        fails += cp
    res["fails"] = fails; res["ok"] = not fails
    return res


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def save_paper(fig, name: str, caption: str | dict, spec: dict | None = None, draft: bool = False,
               sources: dict | None = None, title: str | None = None, width_mm: float = ps.W2_MM) -> dict:
    """저장 + QA + 기록을 한 번에.
    1) name(+'_draft') 으로 PDF·600 dpi PNG 저장(paperstyle.save_figure)
    2) qa_check(저장 PDF, 캡션) → _qa/<name>_qa.json. 실패하면 파일명에 '_FAIL' 을 붙이고 CAPTIONS·figure_spec 갱신을 건너뛴다
    3) 통과하면 CAPTIONS.md '## <name>' 절과 figure_spec.json 'paper_<name 소문자>' 항목 갱신
    4) sources = {panel: DataFrame} → source_data/<Fig>_<panel>.csv
    --qa 모드(QA_OPTS['sheets'])면 _qa/<name>_100pct.png(300 dpi) 확인 시트도 만든다. 반환 = QA dict + 경로."""
    import matplotlib.pyplot as plt
    stem_name = f"{name}_draft" if draft else name
    for old in OUT.glob(f"{stem_name}_FAIL.*"):
        old.unlink()
    paths = ps.save_figure(fig, OUT / stem_name, expect_width_mm=width_mm)
    qa = qa_check(fig, paths["pdf"], caption, expect_width_mm=width_mm)
    qa.update(name=stem_name, draft=draft, time=_dt.datetime.now().isoformat(timespec="seconds"))
    if QA_OPTS.get("sheets"):
        fig.savefig(QA_DIR / f"{stem_name}_100pct.png", dpi=300, facecolor="white")
    if not qa["ok"]:
        for k in ("pdf", "png"):
            p = Path(paths[k]); q = p.with_name(f"{stem_name}_FAIL{p.suffix}")
            p.replace(q); paths[k] = str(q)
    qa["paths"] = paths
    (QA_DIR / f"{stem_name}_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=_jsonable))
    if sources:
        qa["source_data"] = [str(save_source_data(name, k, v).relative_to(ROOT)) for k, v in sources.items()]
    if qa["ok"]:
        write_caption(stem_name, caption, title)
        entry = dict(id=f"paper_{stem_name.lower()}", script=f"scripts/4_visualization/paper_figs.py --only {name.split('_')[0].lower()}",
                     exports=[str(Path(paths['pdf']).relative_to(ROOT)), str(Path(paths['png']).relative_to(ROOT))],
                     layout=dict(figsize_mm=[qa["width_mm"], qa["height_mm"]]), units="cm", medium="paper (Sci Rep 2-column 180 mm)",
                     style="polar.paperstyle(Liberation Sans 7–8 pt, METHOD 색 고정, mm 절대 배치, 굵은 소문자 패널 문자)",
                     generated=_dt.date.today().isoformat(), draft=draft, qa="pass")
        entry.update(spec or {})
        update_figure_spec(entry)
    plt.close(fig)
    status = "ok" if qa["ok"] else "FAIL " + "; ".join(qa["fails"][:3])
    print(f"[paper] {stem_name}: {qa['width_mm']}×{qa['height_mm']} mm, min {qa['min_font_pt']} pt, {status}")
    return qa


__all__ = [n for n in dir() if not n.startswith("_")]
