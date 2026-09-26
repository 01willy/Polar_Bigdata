"""Table 1. 대상별 요약(CSV + LaTeX booktabs + 180 mm 렌더 미리보기). 정본 스펙: figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §3·§1.6.
확정 사실 목록(2026-09-26 20:45)의 정의를 따른다. 행 = H25b 분석 대상 14(오차 유형 묶음, |ln E비| 내림차순) + 라벨 0 평가만 있는
주 지역 2(Russia C·Greenland) + 요약 2행(주 4지역 평균, n* 달성 수).

열(원고 절 순서: 지역 기술 → 라벨 0 → 라벨 희소 → 라벨 전량)
  Cells / blocks   h3/h25b_targets.csv n_cells, n_blocks(대상 전체 라벨 셀·블록; 분석별 채점 부분 집합은 보충표 1).
                   Russia C·Greenland 는 fidelity_base_v3.csv F4_direct 매크로 지역에서 같은 정의로 센다(Lena·Canada 대조).
  E_own/E0 (|ln|)  h3/h25b_targets.csv E_own, E0(원점 통과 최소제곱, 대상 + 100 km 버퍼 제외 원천). 오차 유형(단일 정의):
                   수준 |ln| ≥ 0.20, 중간 0.15–0.20, 구조 < 0.15. CA-1 은 A 블록 라벨 3개라 분석 제외(사실 목록).
  라벨 0 잔차 Δ    h4/a2_curve.csv scope n0, E0_fixed, resid, λ 0.25(원천 잔차만, 대상 라벨 0): d_phys_mean [d_phys_lo, d_phys_hi]
                   = 블록 부트스트랩(분할 안 채점 블록 재표집, 분할 평균). 세 조건(CI 상한 < 0, 분할 승률 ≥ 2/3, 반복 승률 ≥ 0.75)을
                   n = 0 에서 이미 만족하면 각주.
  라벨 0 커버리지  h4/c2_coverage.csv test label0, scope all, method = c2_meta primary_method(hier2_cdf): MAIN6 만.
                   평균 행 = MEAN6[Russia_W,Russia_E,Canada,Lena](주 4지역, 층화 블록 부트스트랩).
  n* H25b          h3/h25b_breakeven.csv S3(κ 10 수축 E + CatBoost 잔차), λ 0.25, α 1, rule kmedoid·random: breakeven_n_ci
                   (주 정의 세 조건), 미달성 → "> n_max". h3/h27c_targets.csv definition main 과 대조.
  n* A2            h4/a2_minn.csv E0_fixed, all_blocks, λ 0.25, resid: n_star, censored, n_max. 미달성 "> n_max".
  레시피 Δ         h4/a1_recipe_table.csv H13 labels λ 0.25(M1 사전 지정 Stefan + CatBoost 잔차): AB4 지역 행(블록 CI,
                   m1/m1_sc_tests.csv 대조) + REGION_SUMMARY_AB4(층화) 평균 행.
  구간 폭(cm)      h4/c2_coverage.csv width_cm(같은 행). 정수 cm. 평균 행 = MEAN6 폭(사실 목록 width_ab4 81 과 대조).
  대상 (No.)       Fig 1a 지역 번호(fig1.REGION_NO). 하위 지역은 상위 지역 번호(AL → 1, LE → 2, CA → 3).

스펙 §3 초안(주 지역 7 + 외부 홀드아웃 + 평균 행, 하위 지역 14는 ST3)과 다른 점과 근거(2026-09-26 검토 반영):
  - 행 = H25b 분석 대상 14. F1–F11 의 분석 단위가 이 14 대상이고(n*·잔차 Δ 모두 대상별), 주 지역 행만 두면 n* 열이
    Alaska(하위 지역 6개로만 채점)에서 비게 된다. 라벨 0 평가만 있는 Russia C·Greenland 는 별도 묶음으로 둔다.
  - 외부 홀드아웃(8, Mongolia·Central Asia)은 라벨 정의가 달라(지중온도 유도 융해 깊이) 같은 열 체계로 비교할 수 없어
    Fig 7c·Supplementary Table 7 에 둔다.
  - 'Worst deploy RMSE' 열 → Fig 7b(최악 대상 틱)·Supplementary Table 4a(B1 전 방법·τ 의 최악 대상).
  - 'Workflow branch' 열 → Fig 7a 잎 태그(outputs/figures/paper/workflow_tags.json). 표에 다시 적으면 태그 근거와
    이중 관리가 되므로 넣지 않는다.
  - H27 손익분기(회복률 규칙)는 Supplementary Table 6a(스펙과 같음).
산출: outputs/figures/paper/Table1_region_summary.{csv,tex,pdf,png}, _qa/Table1_region_summary_values.txt,
      _qa/table1_tex/(LaTeX 컴파일 점검), source_data/Table1_table.csv
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from _common import (ps, ROOT, PROC, H3, H4, OUT, QA_DIR, CAPTIONS, AB4, MAIN6, TARGETS14, MissingData, rd, h4_path,
                     load_a2_minn, load_recipe_table, save_paper, _locked)
from fig1 import REGION_NO                        # Fig 1a 지역 번호(두 표시 항목의 번호를 한 곳에서 관리)

NAME = "Table1_region_summary"
STRUCT_MAX, LEVEL_MIN = 0.15, 0.20              # 사실 목록 단일 정의
MIN_BLOCKS_CI = 8
SPLIT_WIN_MIN, REP_WIN_MIN = 2 / 3, 0.75
DASH = "–"
NV, NSV = "$\\mathit{n}$", "$\\mathit{n}^*$"          # 변수 이탤릭(PNG mathtext·LaTeX 공통 표기)
EXTRA = ["Russia_C", "Greenland"]               # MAIN6 중 H25b 대상이 아닌 지역(라벨 0 평가만)
MEAN_KEY, COUNT_KEY = "MEAN_AB4", "COUNT"
N0_NOTE_CM = 3.0                                # 라벨 0 잔차 Δ 가 이보다 크면(열 안 극단값) 크기 근거 각주
GROUP_TXT = {"level": "Level error (|log| ≥ 0.20)", "intermediate": "Intermediate (0.15 ≤ |log| < 0.20)",
             "structure": "Structure error (|log| < 0.15)", "extra": "Label-free evaluation only"}

# 각주(의미 키 → 문장). 기호는 첫 등장 순서(머리 각주 먼저, 다음 본문 행 우선·열 순서)로 cells() 가 부여한다.
FOOT = {
    "total": "All labelled cells and blocks of the target. Scored cell sets differ by analysis (Supplementary Table 1).",
    "fewblk": "Fewer than 5 scoring blocks in at least one split; block CIs are approximate.",
    "n0met": f"All three {NSV} criteria are already met at {NV} = 0 (source residual only).",
    "n0lost": f"Criteria met at {NV} = 0 fail at every tested {NV} ≥ 3 under this rule (split or repeat win rate, or CI upper bound).",
    "range": f"Range-limited: largest tested {NV} < 40 (about 30 labelled cells).",
    "onesplit": f"Fewer than two valid splits in H25; {NSV} cannot be reached by rule.",
    "few": "Fewer than 8 scoring blocks (Russia C: 7 scored cells; Greenland: 1): point estimate without CI.",
    "m1set": "Scored on M1 B-block cells summed over three splits ({counts}), hence more than Cells.",
    "mean": "Equal-weight mean over Lena, Canada, Russia W and Russia E; stratified block-bootstrap CI.",
    "n0big": "{text}",
}


# ================================================================ 1) 자료 수집
def err_class(v: float) -> str:
    if not np.isfinite(v):
        return "extra"
    return "level" if v >= LEVEL_MIN else ("structure" if v < STRUCT_MAX else "intermediate")


def load_targets() -> pd.DataFrame:
    T = rd(H3 / "h25b_targets.csv")
    g = T.groupby("target").agg(parent=("parent", "first"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first"),
                                E_own=("E_own", "first"), E0=("E0", "first"), E0_n=("E0", "nunique")).reset_index()
    if (g.E0_n > 1).any():
        raise ValueError("h25b_targets E0 가 분할마다 다르다: 대상별 단일 E0 가정 위반")
    g = g[g.target.isin(TARGETS14)].drop(columns="E0_n")
    if set(g.target) != set(TARGETS14):
        raise MissingData(f"h25b_targets 대상 누락: {set(TARGETS14) - set(g.target)}")
    g["E_ratio"] = g.E_own / g.E0
    g["abs_logE"] = np.abs(np.log(g.E_ratio))
    g["error_type"] = g.abs_logE.map(err_class)
    g.attrs["source"] = T.attrs["source"]
    return g


def extra_counts(log: list) -> dict:
    """Russia C·Greenland 라벨 셀·블록 수(F4_direct 매크로). 같은 계산으로 Lena·Canada 가 h25b_targets 와 일치하는지 대조."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from polar.m1_core import load_base
    df = load_base(PROC, sources=("F4_direct",))
    ref = rd(H3 / "h25b_targets.csv").groupby("target").first()
    out = {}
    for key in EXTRA + ["Lena", "Canada"]:
        s = df[df.macro == key]
        out[key] = (int(len(s)), int(s.block.nunique()))
    for key in ("Lena", "Canada"):
        if out[key] != (int(ref.loc[key, "n_cells"]), int(ref.loc[key, "n_blocks"])):
            raise ValueError(f"{key} 셀·블록 계산 {out[key]} ≠ h25b_targets")
    meta = json.loads(h4_path("c2", "meta", False, "json").read_text())
    for p in meta.get("per_target", []):
        if p["target"] in EXTRA and int(p["n_cells"]) != out[p["target"]][0]:
            raise ValueError(f"{p['target']} 셀 수 {out[p['target']][0]} ≠ c2_meta {p['n_cells']}")
    log.append("cells/blocks Russia C, Greenland from fidelity_base_v3.csv (F4_direct, macro): "
               + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in out.items()) + "; Lena, Canada = h25b_targets; cells = c2_meta")
    return {k: out[k] for k in EXTRA}


def n0_rows(log: list) -> tuple[dict, str]:
    """{대상: (est, lo, hi, split_win, rep_win, ci_flag)}: A2 라벨 0(원천 잔차만) 행."""
    A = rd(h4_path("a2", "curve"))
    q = A[(A.scope == "n0") & (A.e_treat == "E0_fixed") & (A.stage == "resid") & np.isclose(A.lam, 0.25)]
    out = {}
    for r in q.itertuples():
        if r.target in TARGETS14:
            out[r.target] = (float(r.d_phys_mean), float(r.d_phys_lo), float(r.d_phys_hi), float(r.split_win), float(r.rep_win),
                             str(r.ci_flag) if isinstance(r.ci_flag, str) else "",
                             float(r.rmse_phys), float(r.rmse_mean), list(json.loads(r.d_phys_split).values()))
    miss = set(TARGETS14) - set(out)
    if miss:
        raise MissingData(f"a2_curve n0 행 누락: {miss}")
    log.append(f"n = 0 source residual from {A.attrs['source']} (scope n0, E0_fixed, resid, λ 0.25)")
    return out, A.attrs["source"]


def h25_rows(log: list) -> tuple[dict, str]:
    """{(대상, rule): dict(n_star, censored, n_max, flag)}: H25b 주 정의(S3, λ 0.25, α 1). h27c main 과 대조."""
    B = rd(H3 / "h25b_breakeven.csv")
    q = B[(B.stage == "S3") & np.isclose(B.lam, 0.25) & np.isclose(B.alpha, 1.0) & B.rule.isin(["kmedoid", "random"])]
    X = pd.read_csv(H3 / "h27c_targets.csv")
    X = X[(X.stage == "S3") & np.isclose(X.lam, 0.25) & (X.definition == "main")].set_index(["target", "rule"])
    out = {}
    for r in q.itertuples():
        if r.target not in TARGETS14:
            continue
        cens = bool(r.censored)
        ns = np.nan if cens else float(r.breakeven_n_ci)
        x = X.loc[(r.target, r.rule)]
        if bool(x.censored) != cens or int(x.n_max) != int(r.n_max) or (not cens and float(x.be_n) != ns):
            raise ValueError(f"H25b n* {r.target} {r.rule}: breakeven {ns}/{cens} ≠ h27c main {x.be_n}/{x.censored}")
        out[(r.target, r.rule)] = dict(n_star=ns, censored=cens, n_max=int(r.n_max),
                                       flag=str(r.be_flag) if isinstance(r.be_flag, str) else "",
                                       n_valid_splits=int(r.n_valid_splits))
    if len(out) != 2 * len(TARGETS14):
        raise MissingData("h25b_breakeven S3 λ 0.25 행 누락")
    log.append(f"H25b n* from {B.attrs['source']} (S3, λ 0.25, α 1, breakeven_n_ci) = h3/h27c_targets.csv main (28 of 28)")
    return out, B.attrs["source"]


def a2_rows(log: list) -> tuple[dict, str]:
    M = load_a2_minn(False)
    if M.attrs.get("is_smoke") or M.attrs.get("legacy"):
        raise MissingData("h4/a2_minn.csv 본 실행(세 조건 n_star) 필요")
    q = M[(M.e_treat == "E0_fixed") & (M.spread == "all_blocks") & np.isclose(M.lam, 0.25) & (M.stage == "resid")]
    out = {r.target: dict(n_star=float(r.n_star), censored=bool(r.censored), n_max=int(r.n_max),
                          range_limited=bool(r.range_limited)) for r in q.itertuples() if r.target in TARGETS14}
    if set(out) != set(TARGETS14):
        raise MissingData(f"a2_minn 대상 누락: {set(TARGETS14) - set(out)}")
    log.append(f"A2 n* from {M.attrs['source']} (E0_fixed, all_blocks, λ 0.25, resid)")
    return out, M.attrs["source"]


def recipe_rows(log: list) -> tuple[dict, str]:
    Q = load_recipe_table("H13", "labels")
    Q = Q[Q.label.astype(str).str.contains("λ=0.25", regex=False)]
    out = {}
    for r in Q.itertuples():
        key = MEAN_KEY if r.target == "REGION_SUMMARY_AB4" else r.target
        if key == MEAN_KEY or key in AB4:
            out[key] = (float(r.delta), float(r.ci_lo), float(r.ci_hi), float(r.p_holm) if np.isfinite(r.p_holm) else np.nan,
                        int(r.n_cells), int(r.n_blocks))
    if set(out) != set(AB4) | {MEAN_KEY}:
        raise MissingData("a1_recipe_table H13 λ 0.25 AB4 행 누락")
    log.append(f"recipe from {Q.attrs.get('source')} (H13 labels λ=0.25; regions block CI, mean REGION_SUMMARY_AB4)")
    return out, Q.attrs.get("source", "data/processed/h4/a1_recipe_table.csv")


def cov_rows(log: list) -> tuple[dict, str, str]:
    meta = json.loads(h4_path("c2", "meta", False, "json").read_text())
    method = meta.get("primary_method", "hier2_cdf")
    C = rd(h4_path("c2", "coverage"))
    q = C[(C.test == "label0") & (C.method == method) & (C.scope == "all")]
    out = {}
    for r in q.itertuples():
        key = r.target
        if key.startswith("MEAN"):
            inner = re.findall(r"\[(.*)\]", key)
            if not inner or set(inner[0].split(",")) != set(AB4):
                continue
            key = MEAN_KEY
        if key in MAIN6 or key == MEAN_KEY:
            nb = float(r.n_blocks) if np.isfinite(r.n_blocks) else np.nan
            out[key] = (float(r.coverage), float(r.coverage_lo), float(r.coverage_hi), float(r.width_cm), nb, float(r.n_eval))
    if set(out) != set(MAIN6) | {MEAN_KEY}:
        raise MissingData(f"c2_coverage label0 {method} 행 누락: {set(MAIN6) | {MEAN_KEY} - set(out)}")
    log.append(f"coverage from {C.attrs['source']} (label0, scope all, {method} = c2_meta primary_method)")
    return out, method, C.attrs["source"]


def collect() -> tuple[pd.DataFrame, dict]:
    """표 한 행 = 대상(kind data), 묶음 머리(kind group), 요약(kind mean·count). 원시 수치 열 + marks."""
    log = []
    G = load_targets()
    ex = extra_counts(log)
    n0, n0_src = n0_rows(log)
    h25, h25_src = h25_rows(log)
    a2, a2_src = a2_rows(log)
    rec, rec_src = recipe_rows(log)
    cov, cov_method, cov_src = cov_rows(log)
    Te = rd(H3 / "h25b_targets.csv").groupby("target").n_eval.sum()
    for k in AB4:
        if int(Te[k]) != rec[k][4]:
            raise ValueError(f"레시피 셀 수 {k} {rec[k][4]} ≠ h25b_targets n_eval 3분할 합 {int(Te[k])}")
    FOOT["m1set"] = FOOT["m1set"].replace("{counts}", "; ".join(f"{ps.region_name(k)} {rec[k][4]:,}"
                                                                 for k in ("Lena", "Canada", "Russia_W", "Russia_E")))
    log.append("recipe n_cells (a1_recipe_table) = sum over splits of h25b_targets n_eval (B-block scored cells) for AB4")
    FOOT["few"] = ("Fewer than 8 scoring blocks (" + "; ".join(f"{ps.region_name(k)}: {int(cov[k][5])} scored cell"
                   + ("s" if cov[k][5] != 1 else "") for k in EXTRA) + "): point estimate without CI.")

    def empty_marks():
        return dict(name=[], n0=[], cov=[], km=[], rd=[], a2=[], rec=[])

    rows, big = [], []
    order = ["level", "intermediate", "structure"]
    for grp in order:
        sub = G[G.error_type == grp].sort_values("abs_logE", ascending=False)
        if sub.empty:
            continue
        rows.append(dict(kind="group", region=f"GROUP_{grp}", name=GROUP_TXT[grp], error_type=grp, marks=empty_marks()))
        for r in sub.itertuples():
            k = r.target
            mk = empty_marks()
            d = dict(kind="data", region=k, name=f"{ps.region_name(k)} ({REGION_NO[r.parent]})", parent=r.parent,
                     error_type=grp, n_cells=int(r.n_cells),
                     n_blocks=int(r.n_blocks), E_own=r.E_own, E0=r.E0, E_ratio=r.E_ratio, abs_logE=r.abs_logE)
            e, lo, hi, sw, rw, flag, rp, rm, dsp = n0[k]
            met = (hi < 0) and (sw >= SPLIT_WIN_MIN - 1e-12) and (rw >= REP_WIN_MIN - 1e-12)
            d.update(n0_d=e, n0_lo=lo, n0_hi=hi, n0_split_win=sw, n0_rep_win=rw, n0_criteria_met=met, n0_ci_flag=flag,
                     n0_rmse_phys=rp, n0_rmse=rm)
            if met:
                mk["n0"].append("n0met")
            if e > N0_NOTE_CM:
                big.append((k, rp, rm, dsp))
                mk["n0"].append("n0big")
            h_flags = {h25[(k, ru)]["flag"] for ru in ("kmedoid", "random")}
            if "blocks<5" in flag or any("blocks<5" in f for f in h_flags):
                mk["name"].append("fewblk")
            for ru, col in (("kmedoid", "km"), ("random", "rd")):
                v = h25[(k, ru)]
                d.update({f"nstar_h25_{ru}": v["n_star"], f"nstar_h25_{ru}_censored": v["censored"],
                          f"nstar_h25_{ru}_nmax": v["n_max"]})
                if v["censored"] and met:
                    mk[col].append("n0lost")
                if v["censored"] and v["n_valid_splits"] < 2:
                    mk[col].append("onesplit")
                elif v["censored"] and v["n_max"] < 40:
                    mk[col].append("range")
            v = a2[k]
            d.update(nstar_a2=v["n_star"], nstar_a2_censored=v["censored"], nstar_a2_nmax=v["n_max"],
                     nstar_a2_range_limited=v["range_limited"])
            if v["censored"] and met:
                mk["a2"].append("n0lost")
            if v["censored"] and v["range_limited"]:
                mk["a2"].append("range")
            if k in cov:
                c = cov[k]
                d.update(cov=c[0], cov_lo=c[1], cov_hi=c[2], cov_width_cm=c[3], cov_n_blocks=c[4], cov_n_eval=c[5])
                if not (np.isfinite(c[1]) and c[4] >= MIN_BLOCKS_CI):
                    mk["cov"].append("few")
            if k in rec:
                e, lo, hi, p, nc, nb = rec[k]
                d.update(recipe_d=e, recipe_lo=lo, recipe_hi=hi, recipe_n_cells=nc, recipe_n_blocks=nb)
            d["marks"] = mk
            rows.append(d)
    if len(big) != 1:
        raise ValueError(f"라벨 0 잔차 Δ > {N0_NOTE_CM} cm 대상이 1개가 아니다: {[b[0] for b in big]} (각주 문장 재검토)")
    k, rp, rm, dsp = big[0]
    lowest = rp <= min(v[6] for v in n0.values()) + 1e-12
    FOOT["n0big"] = (f"Physics-anchor RMSE {rp:.1f} cm" + (f", lowest of the {len(TARGETS14)} targets" if lowest else "")
                     + f"; the source-trained residual gives {rm:.1f} cm and is worse in all {len(dsp)} splits "
                     f"(+{min(dsp):.1f} to +{max(dsp):.1f} cm)." if min(dsp) > 0 else
                     f"Physics-anchor RMSE {rp:.1f} cm; the source-trained residual gives {rm:.1f} cm.")
    log.append(f"n = 0 extreme {k}: rmse_phys {rp:.2f}, rmse {rm:.2f}, splits {dsp}, lowest phys RMSE {lowest}")
    rows.append(dict(kind="group", region="GROUP_extra", name=GROUP_TXT["extra"], error_type="extra", marks=empty_marks()))
    for k in EXTRA:
        mk = empty_marks()
        c = cov[k]
        d = dict(kind="data", region=k, name=f"{ps.region_name(k)} ({REGION_NO[k]})", parent=k, error_type="not classified",
                 n_cells=ex[k][0], n_blocks=ex[k][1], E_own=np.nan, E0=np.nan, E_ratio=np.nan, abs_logE=np.nan,
                 cov=c[0], cov_lo=c[1], cov_hi=c[2], cov_width_cm=c[3], cov_n_blocks=c[4], cov_n_eval=c[5])
        if not (np.isfinite(c[1]) and c[4] >= MIN_BLOCKS_CI):
            mk["cov"].append("few")
        d["marks"] = mk
        rows.append(d)
    # 요약 2행
    e, lo, hi, p, nc, nb = rec[MEAN_KEY]
    c = cov[MEAN_KEY]
    rows.append(dict(kind="mean", region=MEAN_KEY, name="Mean of 4 regions", recipe_d=e, recipe_lo=lo, recipe_hi=hi,
                     recipe_p_holm=p, cov=c[0], cov_lo=c[1], cov_hi=c[2], cov_width_cm=c[3],
                     marks=dict(empty_marks(), name=["mean"])))
    cnt = dict(kind="count", region=COUNT_KEY, name=f"Targets reaching {NSV}", marks=empty_marks())
    for ru in ("kmedoid", "random"):
        cnt[f"reach_h25_{ru}"] = int(sum(not h25[(k, ru)]["censored"] for k in TARGETS14))
    cnt["reach_a2"] = int(sum(not a2[k]["censored"] for k in TARGETS14))
    cnt["n_targets"] = len(TARGETS14)
    rows.append(cnt)
    T = pd.DataFrame(rows)
    info = dict(log=log, cov_method=cov_method,
                sources=[G.attrs["source"], "data/processed/fidelity_base_v3.csv", n0_src, cov_src, h25_src,
                         "data/processed/h3/h27c_targets.csv", a2_src, rec_src, "data/processed/m1/m1_sc_tests.csv"])
    return T, info


# ================================================================ 2) 표시 문자열
def plain(s: str) -> str:
    """mathtext 표기 → CSV 평문('$\\mathit{n}^*$' → 'n*')."""
    s = re.sub(r"\\math(?:it|rm)\{([^}]*)\}", r"\1", str(s))
    return s.replace("^*", "*").replace("$", "")


def _f(v, nd):
    return ps.fmt_num(v, nd) if v is not None and np.isfinite(v) else DASH


def _nstar(ns, cens, nmax) -> str:
    """주 정의 n*: 달성 = 정수, 미달성 = '> n_max'(대입값 없음)."""
    if cens or not np.isfinite(ns):
        return f"> {int(nmax)}"
    return f"{int(ns)}"


def _pair(est, lo, hi, nd, marks):
    """(추정 셀, CI 셀). CI 없음: CI 자리에 "[–]" + 각주. 각주 기호는 CI 셀(왼쪽 정렬) 끝에 붙인다."""
    if est is None or not np.isfinite(est):
        return (DASH, []), ("", [])
    if lo is not None and np.isfinite(lo):
        return (_f(est, nd), []), (f"[{_f(lo, nd)}, {_f(hi, nd)}]", marks)
    return (_f(est, nd), []), (f"[{DASH}]", marks)


def _cells_raw(T: pd.DataFrame) -> list[dict]:
    out = []
    for r in T.itertuples():
        mk = r.marks
        c = {k: ("", []) for k, *_ in COLS}
        c["name"] = (r.name, mk["name"])
        if r.kind == "group":
            out.append(c)
            continue
        g = lambda a: getattr(r, a, np.nan)                                                   # noqa: E731
        if r.kind == "data":
            c["cells"] = (f"{int(r.n_cells):,}", [])
            c["blocks"] = (f"/ {int(r.n_blocks)}", [])
            c["eratio"] = (f"{_f(r.E_ratio, 2)} ({_f(r.abs_logE, 2)})" if np.isfinite(r.E_ratio) else DASH, [])
            c["n0_est"], c["n0_ci"] = _pair(g("n0_d"), g("n0_lo"), g("n0_hi"), 2, mk["n0"])
            if isinstance(g("nstar_h25_kmedoid_nmax"), (int, float)) and np.isfinite(g("nstar_h25_kmedoid_nmax")):
                c["km"] = (_nstar(g("nstar_h25_kmedoid"), bool(g("nstar_h25_kmedoid_censored")), g("nstar_h25_kmedoid_nmax")), mk["km"])
                c["rd"] = (_nstar(g("nstar_h25_random"), bool(g("nstar_h25_random_censored")), g("nstar_h25_random_nmax")), mk["rd"])
                c["a2"] = (_nstar(g("nstar_a2"), bool(g("nstar_a2_censored")), g("nstar_a2_nmax")), mk["a2"])
            else:
                c["km"] = c["rd"] = c["a2"] = (DASH, [])
            if not np.isfinite(g("E_ratio")):
                c["n0_est"], c["n0_ci"] = (DASH, []), ("", [])
        if r.kind == "count":
            n = int(r.n_targets)
            c["km"] = (f"{int(r.reach_h25_kmedoid)} of {n}", [])
            c["rd"] = (f"{int(r.reach_h25_random)} of {n}", [])
            c["a2"] = (f"{int(r.reach_a2)} of {n}", [])
            out.append(c)
            continue
        c["cov_est"], c["cov_ci"] = _pair(g("cov"), g("cov_lo"), g("cov_hi"), 2, mk["cov"])
        c["cov_w"] = (_f(g("cov_width_cm"), 0), [])
        c["rec_est"], c["rec_ci"] = _pair(g("recipe_d"), g("recipe_lo"), g("recipe_hi"), 2, mk["rec"])
        out.append(c)
    return out


def cells(T: pd.DataFrame) -> tuple[list[dict], list, dict]:
    """(행별 표시 셀 {열 키: (본문, [각주 기호])}, [(기호, 각주 문장)], 머리 각주 {열 키: [기호]})."""
    raw = _cells_raw(T)
    order = []
    for k, *_ in COLS:
        for m in HEAD_MARKS.get(k, []):
            if m not in order:
                order.append(m)
    for c in raw:
        for k, *_ in COLS:
            for m in c[k][1]:
                if m not in order:
                    order.append(m)
    letter = {m: "abcdefghij"[i] for i, m in enumerate(order)}
    out = [{k: (v[0], [letter[m] for m in v[1]]) for k, v in c.items()} for c in raw]
    head = {k: [letter[m] for m in v] for k, v in HEAD_MARKS.items()}
    return out, [(letter[m], FOOT[m]) for m in order], head


# 열 정의: (키, 머리 2행(L2, L3), 최소 폭 mm, 정렬). 실제 폭 = max(최소 폭, 렌더 측정 폭).
COLS = [
    ("name", ("", "Target (No.)"), 16.0, "l"),
    ("cells", ("", "Cells / blocks"), 7.0, "r"),
    ("blocks", ("", ""), 4.5, "l"),
    ("eratio", ("$\\mathit{E}_\\mathrm{own}/\\mathit{E}_0$", "(|log|)"), 12.0, "c"),
    ("n0_est", ("Residual ΔRMSE", "(cm) [95 % CI]"), 6.0, "r"),
    ("n0_ci", ("", ""), 12.0, "l"),
    ("cov_est", ("90 % interval", "coverage [95 % CI]"), 5.0, "r"),
    ("cov_ci", ("", ""), 11.0, "l"),
    ("cov_w", ("Width", "(cm)"), 6.0, "r"),
    ("km", ("", "k-medoid"), 10.0, "l"),
    ("rd", ("", "random"), 9.0, "l"),
    ("a2", ("$\\mathit{E}_0$ fixed", "(A2)"), 9.0, "l"),
    ("rec_est", ("Recipe ΔRMSE", "(cm) [95 % CI]"), 6.0, "r"),
    ("rec_ci", ("", ""), 12.0, "l"),
]
PAIRS = {"cells": ("blocks", 0.8), "n0_est": ("n0_ci", 1.0), "cov_est": ("cov_ci", 1.0), "rec_est": ("rec_ci", 1.0)}
HEAD_MARKS = {"cells": ["total"], "rec_est": ["m1set"]}
GROUPS = [(f"No target labels ({NV} = 0)", ["n0_est", "n0_ci", "cov_est", "cov_ci", "cov_w"]),
          (f"Few labels: {NSV}", ["km", "rd", "a2"]), ("All labels", ["rec_est", "rec_ci"])]
SUBGROUPS = [("Shrunk $\\mathit{E}$ (H25)", ["km", "rd"])]
TABLE_RIGHT_MM = 178.5
GAP_RANGE = (1.2, 3.2)
PAIR_SPILL_MM = 2.5
FOOT_PITCH = 3.4                                   # 각주 줄 간격(mm): 윗첨자 기호와 윗줄 내림획 간섭 방지


# ================================================================ 3) 렌더 미리보기(matplotlib, mm 절대 배치)
def _sup(text: str, marks: list) -> str:
    return text + (f"$^{{{','.join(marks)}}}$" if marks else "")


def _text_mm(fig, s: str, size: float) -> float:
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=size)
    w = t.get_window_extent(r).width / fig.dpi * 25.4
    t.remove()
    return w


def _layout(fig, C: list, kinds: list, head: dict, fs: float, x0: float = 1.5) -> tuple[dict, dict]:
    """열 키 → (x_left, x_right) mm. 폭 = max(최소 폭, 본문 최장 + 0.2 mm). 묶음 머리 행(group)의 이름은 표 전체 폭을
    쓰므로 측정에서 뺀다. 남는 폭은 열 간격으로 고르게 나누되 GAP_RANGE 로 제한(하한 미만이면 ValueError)."""
    need = {}
    seconds = {v[0] for v in PAIRS.values()}
    for k, (l2, l3), wmin, al in COLS:
        ws = [_text_mm(fig, _sup(s, mk), fs) for (s, mk), kd in ((c[k], kd) for c, kd in zip(C, kinds))
              if (s or mk) and not (kd == "group" and k == "name")]
        if k not in PAIRS and k not in seconds:
            ws += [_text_mm(fig, h, fs) for h in (l2, _sup(l3, head.get(k, []))) if h]
        need[k] = max([wmin] + [w + 0.2 for w in ws])
    for a, (b, g) in PAIRS.items():
        l2, l3 = next(h for k, h, *_ in COLS if k == a)
        hw = max(_text_mm(fig, h, fs) for h in (l2, _sup(l3, head.get(a, []))) if h)
        short = hw - 2 * PAIR_SPILL_MM - (need[a] + g + need[b])
        if short > 0:
            need[b] += short
    for label, keys in SUBGROUPS:                                         # 소묶음 머리가 묶음 폭보다 넓으면 마지막 열을 넓힌다
        span = sum(need[k] for k in keys)
        short = _text_mm(fig, label, fs) + 0.4 - span
        if short > 0:
            need[keys[-1]] += short
    keys = [k for k, *_ in COLS]
    joined = {a: g for a, (b, g) in PAIRS.items()}
    n_gap = sum(1 for k in keys[:-1] if k not in joined)
    fixed = sum(need.values()) + sum(joined.values())
    gap = (TABLE_RIGHT_MM - x0 - fixed) / n_gap
    if gap < GAP_RANGE[0]:
        raise ValueError(f"Table 1 열 폭 합계 {fixed:.1f} mm: 간격 {gap:.2f} mm < {GAP_RANGE[0]} mm")
    gap = min(gap, GAP_RANGE[1])
    x0 = max(x0, (180.0 - (fixed + n_gap * gap)) / 2)
    pos, x = {}, x0
    for k in keys:
        pos[k] = (x, x + need[k])
        x += need[k] + (joined[k] if k in joined else gap)
    return pos, dict(gap_mm=round(gap, 2), left_mm=round(x0, 2), right_mm=round(pos[keys[-1]][1], 2),
                     width_mm={k: round(v, 2) for k, v in need.items()})


def render(T: pd.DataFrame):
    from matplotlib.lines import Line2D
    ps.use_paper()
    C, feet, head = cells(T)
    kinds = list(T.kind)
    fs, fs_note = 7.0, ps.FS["annot"]
    pitch, hdr, grp_pitch = 3.7, 3.4, 3.9
    n_data = sum(1 for k in kinds if k != "group")
    n_grp = sum(1 for k in kinds if k == "group")
    H = 2.0 + 3 * hdr + 2.6 + n_data * pitch + n_grp * grp_pitch + 1.6 + 1.8 + len(feet) * FOOT_PITCH + 1.2
    fig = ps.paper_figure(180, round(H, 1))
    W = 180.0
    pos, lay = _layout(fig, C, kinds, head, fs)
    fig._table_layout = lay
    right = max(v[1] for v in pos.values())

    def tx(x, y, s, ha="left", size=fs, **kw):
        return fig.text(x / W, y / H, s, ha=ha, va="baseline", fontsize=size, **kw)

    def hline(x_a, x_b, y, lw):
        fig.add_artist(Line2D([x_a / W, x_b / W], [y / H, y / H], lw=lw, color="#000000", solid_capstyle="butt"))

    def anchor(k, align):
        a, b = pos[k]
        return (a, "left") if align == "l" else ((b, "right") if align == "r" else ((a + b) / 2, "center"))

    x_l, x_r = pos["name"][0], right
    y = H - 2.0
    hline(x_l, x_r, y, 0.8)                                      # toprule
    y1 = y - hdr + 0.6                                           # L1: 묶음 머리 + cmidrule
    for label, keys in GROUPS:
        a, b = pos[keys[0]][0], pos[keys[-1]][1]
        tx((a + b) / 2, y1, label, ha="center")
        hline(a + 0.4, b - 0.4, y1 - 1.1, 0.4)
    y2 = y1 - hdr                                                # L2: 소묶음 머리·열 머리 첫 줄
    for label, keys in SUBGROUPS:
        a, b = pos[keys[0]][0], pos[keys[-1]][1]
        tx((a + b) / 2, y2, label, ha="center")
        hline(a + 0.4, b - 0.4, y2 - 1.1, 0.4)
    y3 = y2 - hdr
    seconds = {v[0] for v in PAIRS.values()}
    for k, (l2, l3), _, al in COLS:
        if k in seconds:
            continue
        if k in PAIRS:                                           # 쌍 머리: 두 열 전체 가운데
            a = (pos[k][0] + pos[PAIRS[k][0]][1]) / 2
            if l2:
                tx(a, y2, l2, ha="center")
            tx(a, y3, _sup(l3, head.get(k, [])), ha="center")
            continue
        x, ha = anchor(k, "c" if k in ("eratio", "km", "rd", "a2") else al)
        if l2:
            tx(x, y2, l2, ha=ha)
        tx(x, y3, _sup(l3, head.get(k, [])), ha=ha)
    y = y3 - 1.5
    hline(x_l, x_r, y, 0.4)                                      # midrule
    y -= 0.9
    for i, c in enumerate(C):
        kind = kinds[i]
        if kind == "group":
            yb = y - grp_pitch + 1.0
            tx(x_l, yb, c["name"][0], style="italic")
            y -= grp_pitch
            continue
        if kind == "mean":
            y -= 0.8
            hline(x_l, x_r, y, 0.4)
            y -= 0.6
        yb = y - pitch + 1.0
        for k, _, _, al in COLS:
            s, mk = c[k]
            if not s and not mk:
                continue
            if k in PAIRS and s == DASH and not c[PAIRS[k][0]][0]:     # 쌍 전체가 빈칸이면 "–" 를 쌍 가운데
                tx((pos[k][0] + pos[PAIRS[k][0]][1]) / 2, yb, s, ha="center")
                continue
            if k in ("km", "rd", "a2"):                          # n* 열: 가운데 정렬(각주 기호는 폭에서 빼고 오른쪽에)
                x = (pos[k][0] + pos[k][1]) / 2
                tx(x, yb, s, ha="center")
                if mk:
                    w = _text_mm(fig, s, fs)
                    tx(x + w / 2 + 0.05, yb, _sup("", mk), ha="left")
                continue
            x, ha = anchor(k, "c") if (al == "r" and s == DASH) else anchor(k, al)
            if al == "r":
                tx(x, yb, s, ha=ha)
                if mk:
                    tx(pos[k][1] + 0.1, yb, _sup("", mk), ha="left")
            else:
                tx(x, yb, _sup(s, mk), ha=ha)
        y -= pitch
    y -= 0.5
    hline(x_l, x_r, y, 0.8)                                      # bottomrule
    y -= FOOT_PITCH
    for m, txt in feet:
        tx(x_l, y, f"$^{{{m}}}$" + txt, size=fs_note)
        y -= FOOT_PITCH
    return fig


# ================================================================ 4) LaTeX(booktabs)
_TEX = [("\\", "\\textbackslash{}"), ("%", "\\%"), ("&", "\\&"), ("_", "\\_"), ("#", "\\#"),
        ("−", "$-$"), ("–", "--"), ("Δ", "$\\Delta$"), ("≥", "$\\geq$"), ("≤", "$\\leq$"), ("<", "$<$"), (">", "$>$"),
        ("λ", "$\\lambda$"), ("|", "$|$"), ("*", "$^{*}$")]


def tex_escape(s: str) -> str:
    if "$" in s:
        parts = s.split("$")
        return "$".join(tex_escape(p) if i % 2 == 0 else p for i, p in enumerate(parts))
    for a, b in _TEX:
        s = s.replace(a, b)
    return s.replace("$$", "")


def _tex_sup(mk: list) -> str:
    return f"\\textsuperscript{{{','.join(mk)}}}" if mk else ""


def to_latex(T: pd.DataFrame) -> str:
    C, feet, head = cells(T)
    kinds = list(T.kind)
    seconds = {v[0] for v in PAIRS.values()}
    spec = []
    for k, _, _, al in COLS:
        al = "c" if k in ("km", "rd", "a2") else al
        spec.append(al + ("@{\\hspace{2pt}}" if k in PAIRS else ""))
    spec = "@{}" + " ".join(spec) + "@{}"
    ncol = len(COLS)
    idx = {k: i + 1 for i, (k, *_) in enumerate(COLS)}
    L = ["% Table 1 (자동 생성: scripts/4_visualization/paper/table1.py; 수정은 생성기에서)",
         "% requires \\usepackage{booktabs}",
         "\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]

    def spanned(levels):
        row, cm = [""] * ncol, []
        for label, keys in levels:
            a, b = idx[keys[0]], idx[keys[-1]]
            row[a - 1] = f"\\multicolumn{{{b - a + 1}}}{{c}}{{{tex_escape(label)}}}"
            for j in range(a, b):
                row[j] = None
            cm.append(f"\\cmidrule(lr){{{a}-{b}}}")
        return row, cm

    l1, cm1 = spanned(GROUPS)
    L.append(" & ".join(x for x in l1 if x is not None) + " \\\\")
    L.append("".join(cm1))
    l2, cm2 = spanned(SUBGROUPS)
    in_sub = {k for _, ks in SUBGROUPS for k in ks}
    for k, (h2, h3), _, _ in COLS:
        if k in in_sub or k in seconds:
            continue
        if k in PAIRS:
            l2[idx[k] - 1] = f"\\multicolumn{{2}}{{c}}{{{tex_escape(h2)}}}" if h2 else "\\multicolumn{2}{c}{}"
            l2[idx[PAIRS[k][0]] - 1] = None
        else:
            l2[idx[k] - 1] = tex_escape(h2)
    L.append(" & ".join(x for x in l2 if x is not None) + " \\\\")
    L.append("".join(cm2))
    l3 = []
    for k, (h2, h3), _, _ in COLS:
        if k in seconds:
            continue
        txt = tex_escape(h3) + _tex_sup(head.get(k, []))
        l3.append(f"\\multicolumn{{2}}{{c}}{{{txt}}}" if k in PAIRS else txt)
    L.append(" & ".join(l3) + " \\\\")
    L.append("\\midrule")
    for i, c in enumerate(C):
        if kinds[i] == "group":
            if i > 0:
                L.append("\\addlinespace")
            L.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{tex_escape(c['name'][0])}}}}} \\\\")
            continue
        if kinds[i] == "mean":
            L.append("\\midrule")
        vals, skip = [], set()
        for k, _, _, al in COLS:
            if k in skip:
                continue
            s, mk = c[k]
            sup = _tex_sup(mk)
            if k in PAIRS and s == DASH and not c[PAIRS[k][0]][0]:      # 쌍 전체 빈칸: "–" 를 두 열 가운데
                vals.append("\\multicolumn{2}{c}{--}")
                skip.add(PAIRS[k][0])
                continue
            if al == "r" and s == DASH:
                v = "\\multicolumn{1}{c}{--}"
            elif al == "r":
                v = tex_escape(s) + (f"\\rlap{{{sup}}}" if sup else "")
            elif k in ("km", "rd", "a2"):
                v = tex_escape(s) + (f"\\rlap{{{sup}}}" if sup else "")
            else:
                v = tex_escape(s) + sup
            vals.append(v)
        L.append(" & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}", "\\begin{minipage}{\\linewidth}\\scriptsize"]
    L += [f"\\textsuperscript{{{m}}}{tex_escape(txt)}\\\\" for m, txt in feet]
    L += ["\\end{minipage}", "\\end{table*}", ""]
    return "\n".join(L)


def latex_check(tex: str, stem: str) -> dict:
    """최소 문서로 pdflatex 컴파일 점검(_qa/table1_tex/). pdflatex 가 없으면 skipped."""
    exe = shutil.which("pdflatex")
    d = QA_DIR / "table1_tex"
    d.mkdir(parents=True, exist_ok=True)
    doc = ("\\documentclass[10pt]{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{booktabs}\n"
           "\\usepackage{helvet}\n\\renewcommand{\\familydefault}{\\sfdefault}\n"
           "\\usepackage[paperwidth=196mm,paperheight=150mm,margin=8mm]{geometry}\n\\pagestyle{empty}\n\\begin{document}\n"
           + tex + "\n\\end{document}\n")
    (d / f"{stem}_check.tex").write_text(doc)
    if not exe:
        return dict(ok=None, note="pdflatex not found")
    p = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", f"{stem}_check.tex"], cwd=d,
                       capture_output=True, text=True, timeout=120)
    logtxt = (d / f"{stem}_check.log").read_text(errors="ignore") if (d / f"{stem}_check.log").exists() else ""
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", logtxt)
    return dict(ok=p.returncode == 0, pdf=str((d / f"{stem}_check.pdf").relative_to(ROOT)), overfull_pt=[float(v) for v in over])


# ================================================================ 5) 값 대조(원천 CSV 독립 재독 + 사실 목록)
FACTS = dict(level={"Russia_W", "AL-3", "CA-2", "AL-1"}, intermediate={"CA-3", "Russia_E"},
             reach_kmedoid={"Russia_W", "AL-1", "AL-3", "CA-2", "Canada", "Lena"}, n_reach_random=5,
             n0_met={"Canada", "AL-1", "CA-2", "AL-5", "AL-3"},
             canada_n0=(-0.95, -1.18, -0.57), recipe_ab4=(-0.68, -1.01, -0.27), cov_ab4=(0.86, 0.80, 0.92), width_ab4=81,
             cov_russia_w=0.75, lena_a2_nstar=320)


def value_check(T: pd.DataFrame, info: dict) -> list:
    out = []
    t = T.set_index("region")
    d = T[T.kind == "data"].set_index("region")
    # 1) 오차 유형 = 사실 목록
    for grp in ("level", "intermediate"):
        got = set(d.index[d.error_type == grp])
        assert got == FACTS[grp], f"{grp}: {got} ≠ facts {FACTS[grp]}"
    out.append("error types = results_facts.md (level: Russia W, AL-3, CA-2, AL-1; intermediate: CA-3, Russia E; rest structure)")
    L = pd.read_csv(H3 / "h27c_targets.csv").groupby("target").first()
    for k in ("Russia_W", "Canada", "AL-3"):
        assert abs(d.loc[k, "abs_logE"] - L.loc[k, "abs_logE_oracle"]) < 1e-9
        out.append(f"|ln E ratio| {k} {d.loc[k, 'abs_logE']:.4f} = h27c_targets abs_logE_oracle")
    # 2) n* 달성 수·집합
    reach_k = {k for k in d.index if np.isfinite(d.loc[k].get("nstar_h25_kmedoid", np.nan))}
    assert reach_k == FACTS["reach_kmedoid"], reach_k
    assert int(t.loc[COUNT_KEY, "reach_h25_random"]) == FACTS["n_reach_random"]
    out.append(f"H25b reach: kmedoid {sorted(reach_k)} (6 of 14), random {int(t.loc[COUNT_KEY, 'reach_h25_random'])} of 14 = facts")
    assert int(d.loc["Lena", "nstar_a2"]) == FACTS["lena_a2_nstar"]
    M = pd.read_csv(H4 / "a2_minn.csv")
    m = M[(M.target == "AL-2") & (M.e_treat == "E0_fixed") & (M.spread == "all_blocks") & np.isclose(M.lam, 0.25) & (M.stage == "resid")].iloc[0]
    assert bool(d.loc["AL-2", "nstar_a2_censored"]) and int(m.n_max) == int(d.loc["AL-2", "nstar_a2_nmax"])
    out.append(f"A2 n* Lena 320, AL-2 > {int(m.n_max)} = h4/a2_minn.csv; reach {int(t.loc[COUNT_KEY, 'reach_a2'])} of 14")
    # 3) n = 0 잔차 Δ
    A = pd.read_csv(H4 / "a2_curve.csv")
    a = A[(A.target == "Canada") & (A.scope == "n0") & (A.e_treat == "E0_fixed") & (A.stage == "resid") & np.isclose(A.lam, 0.25)].iloc[0]
    assert abs(d.loc["Canada", "n0_d"] - a.d_phys_mean) < 1e-12
    for v, f in zip((a.d_phys_mean, a.d_phys_lo, a.d_phys_hi), FACTS["canada_n0"]):
        assert abs(round(v, 2) - f) < 1e-9, (v, f)
    met = set(d.index[d.n0_criteria_met.fillna(False).astype(bool)])
    assert met == FACTS["n0_met"], met
    out.append(f"n = 0 Canada {a.d_phys_mean:.3f} [{a.d_phys_lo:.3f}, {a.d_phys_hi:.3f}] = a2_curve = facts; criteria met at n = 0: {sorted(met)} = facts")
    # 4) 레시피
    R = pd.read_csv(H4 / "a1_recipe_table.csv")
    r = R[(R.H == "H13") & (R.cond == "labels") & (R.target == "REGION_SUMMARY_AB4") & R.label.str.contains("λ=0.25", regex=False)].iloc[0]
    assert abs(t.loc[MEAN_KEY, "recipe_d"] - r.delta) < 1e-12
    assert tuple(round(v, 2) for v in (r.delta, r.ci_lo, r.ci_hi)) == FACTS["recipe_ab4"]
    Mt = pd.read_csv(PROC / "m1" / "m1_sc_tests.csv")
    Mt = Mt[(Mt.H == "H13") & (Mt.cond == "labels") & Mt.label.astype(str).str.contains("λ=0.25", regex=False)].set_index("target")
    for k in AB4:
        assert bool(Mt.loc[k, "ci_valid"]) and abs(d.loc[k, "recipe_lo"] - Mt.loc[k, "ci_lo"]) < 1e-9 \
            and abs(d.loc[k, "recipe_hi"] - Mt.loc[k, "ci_hi"]) < 1e-9
    out.append(f"recipe mean {r.delta:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] = a1_recipe_table = facts; AB4 region CIs = m1_sc_tests (block, ci_valid)")
    # 5) 커버리지
    C = pd.read_csv(H4 / "c2_coverage.csv")
    c = C[(C.test == "label0") & (C.method == info["cov_method"]) & (C.scope == "all") & C.target.str.startswith("MEAN6")].iloc[0]
    assert abs(t.loc[MEAN_KEY, "cov"] - c.coverage) < 1e-12
    assert tuple(round(v, 2) for v in (c.coverage, c.coverage_lo, c.coverage_hi)) == FACTS["cov_ab4"]
    assert round(c.width_cm) == FACTS["width_ab4"]
    assert round(d.loc["Russia_W", "cov"], 2) == FACTS["cov_russia_w"]
    Cd, _, _ = cells(T)
    disp = dict(zip(T.region, (c["cov_w"][0] for c in Cd)))
    assert disp[MEAN_KEY] == str(FACTS["width_ab4"]), disp[MEAN_KEY]
    for k in MAIN6:
        w = C[(C.test == "label0") & (C.method == info["cov_method"]) & (C.scope == "all") & (C.target == k)].width_cm.iloc[0]
        assert disp[k] == ps.fmt_num(w, 0), (k, disp[k], w)
    out.append("width column = c2_coverage width_cm (0 dp): " + ", ".join(f"{k} {disp[k]}" for k in MAIN6)
               + f"; mean {disp[MEAN_KEY]} = facts width_ab4")
    al6 = A[(A.target == "AL-6") & (A.scope == "n0") & (A.e_treat == "E0_fixed") & (A.stage == "resid") & np.isclose(A.lam, 0.25)].iloc[0]
    assert abs(d.loc["AL-6", "n0_d"] - al6.d_phys_mean) < 1e-12 and al6.d_phys_lo <= al6.d_phys_mean <= al6.d_phys_hi
    out.append(f"AL-6 n = 0 {al6.d_phys_mean:.3f} [{al6.d_phys_lo:.3f}, {al6.d_phys_hi:.3f}] inside CI; phys {al6.rmse_phys:.2f} cm, "
               f"resid {al6.rmse_mean:.2f} cm, splits {al6.d_phys_split}")
    out.append(f"coverage mean {c.coverage:.3f} [{c.coverage_lo:.3f}, {c.coverage_hi:.3f}], width {c.width_cm:.1f} cm = c2_coverage {c.target} = facts; Russia W 0.75")
    return out


# ================================================================ 6) 진입점
def caption(T: pd.DataFrame) -> dict:
    cnt = T.set_index("region").loc[COUNT_KEY]
    return dict(
        definition=(
            "Table 1 | Per-target summary by label regime. E_own is the least-squares Stefan coefficient (ALT on √TDD, zero "
            "intercept) of a target's labelled cells; E0, the same fit on source cells over 100 km from the target. Error type uses |log(E_own/E0)|: level ≥ 0.20, structure < 0.15, intermediate between (post hoc). ΔRMSE: method minus physics anchor (E0), cm; negative better. AL, CA and LE: Alaskan, "
            "Canadian and Lena subregions (Supplementary Fig. 3), numbered (No., Fig. 1a) by parent region."),
        statistics=(
            "Brackets: 95 % CIs from 1,000 bootstrap resamples of scoring blocks within each split, paired across "
            "methods, repeats and seeds, split-averaged; the recipe pools splits within region. n* (pre-specified) is the "
            "smallest tested n (3 to 320) at which the ΔRMSE CI upper bound is below 0, at least two of three splits "
            "improve and at least 75 % of repeats improve; '> n': not reached by the largest tested n (no imputed value)."),
        panels=(
            "Residual ΔRMSE: CatBoost residual (λ = 0.25) trained on source labels only, anchored on E0. Coverage: "
            "two-level hierarchical conformal 90 % interval (CDF pooling, pre-specified primary method), leave-one-region-out, "
            "shown for six regions (Russia W, Russia E, Lena, Canada, plus label-free-only Russia C and Greenland); "
            "Width, mean interval width. n* (H25): E shrunk toward E0 "
            "(κ = 10) plus residual ML, labels placed by k-medoid or at random, 3 splits × 5 draws × 2 seeds. n* (A2): E0 fixed plus "
            "residual ML on source and target labels, labels spread at random over all A blocks, 3 splits × 10 draws × "
            f"2 seeds. Reached: {int(cnt.reach_h25_kmedoid)}, {int(cnt.reach_h25_random)} and {int(cnt.reach_a2)} of "
            f"{int(cnt.n_targets)} targets (CA-1, 3 labels, excluded); criteria are re-tested at each n and can fail after being met at n = 0. Recipe: pre-specified Stefan anchor plus CatBoost residual (λ = 0.25) with all "
            "labels (Holm p = 0.004 for the mean)."),
        data=(
            "Source data (data/processed): h3/h25b_targets.csv, h4/a2_curve.csv, h4/c2_coverage.csv, h3/h25b_breakeven.csv, "
            "h4/a2_minn.csv, h4/a1_recipe_table.csv, m1/m1_sc_tests.csv; fidelity_base_v3.csv (Russia C, Greenland counts)."),
    )


def _drop_draft_caption() -> bool:
    """CAPTIONS.md 의 옛 '## Table1_region_summary_draft' 절 제거(최종 절이 생긴 뒤)."""
    with _locked(CAPTIONS):
        cur = CAPTIONS.read_text()
        pat = re.compile(rf"^## {re.escape(NAME)}_draft\n.*?(?=^## |\Z)", re.S | re.M)
        if not pat.search(cur):
            return False
        CAPTIONS.write_text(pat.sub("", cur))
    return True


def build(draft: bool = False):
    ps.use_paper()
    T, info = collect()
    info["log"] += value_check(T, info)
    rec_p = float(T.set_index("region").loc[MEAN_KEY, "recipe_p_holm"])
    assert abs(rec_p - 0.004) < 1e-9, f"recipe Holm p {rec_p} ≠ 0.004 (caption)"
    stem = NAME
    C, feet, _ = cells(T)
    csv = T.drop(columns=["marks"]).copy()
    for k, *_ in COLS:
        csv[f"disp_{k}"] = [plain(c[k][0]) for c in C]
    csv["name"] = csv["name"].map(plain)
    fmap = {m: plain(t) for m, t in feet}
    csv["footnotes"] = [";".join(f"{m}: {fmap[m]}" for m in sorted({m for v in c.values() for m in v[1]})) for c in C]
    csv.to_csv(OUT / f"{stem}.csv", index=False)
    tex = to_latex(T)
    (OUT / f"{stem}.tex").write_text(tex)
    lc = latex_check(tex, stem)
    info["log"].append(f"LaTeX check: {lc}")
    if lc.get("ok") is False:
        raise RuntimeError(f"Table 1 LaTeX 컴파일 실패: {QA_DIR / 'table1_tex'}")
    fig = render(T)
    info["log"].append(f"layout: {fig._table_layout}")
    (QA_DIR / f"{stem}_values.txt").write_text("\n".join(info["log"]) + "\n")
    qa = save_paper(fig, NAME, caption=caption(T), draft=False, sources={"table": csv},
                    spec=dict(id="paper_table1", intent="Table 1 대상별 요약: 대상(Fig 1a 지역 번호), 셀·블록, E_own/E0·오차 유형(수준/중간/구조), "
                              "라벨 0 원천 잔차 Δ·계층 conformal 커버리지와 구간 폭(MAIN6), H25b·A2 주 정의 n*(미달성 '> n_max'), "
                              "라벨 전량 사전 지정 레시피 Δ(AB4). 표 본체는 LaTeX booktabs, PNG/PDF 는 미리보기.",
                              design_note=("행 = H25b 분석 대상 14(F1–F11 분석 단위) + 라벨 0 평가만 있는 Russia C·Greenland + 요약 2행. "
                                           "스펙 §3 초안의 외부 홀드아웃 행은 Fig 7c·ST7, 'Worst deploy RMSE' 는 Fig 7b·ST4a, "
                                           "'Workflow branch' 는 Fig 7a 태그(workflow_tags.json), H27 손익분기는 ST6a 로 둔다. "
                                           "AL-6 라벨 0 잔차 Δ(+6.0 cm) 극단값은 각주(물리식 RMSE 최저 10.4 cm, 3분할 모두 악화)."),
                              assets=info["sources"], exports_table=[f"outputs/figures/paper/{stem}.csv", f"outputs/figures/paper/{stem}.tex"],
                              ci_kinds=dict(n0_residual="block (split-averaged)", coverage="block (h4 C2)", coverage_mean="strat_AB4",
                                            nstar="block (three-criterion main definition)", recipe_regions="block (M1)",
                                            recipe_mean="strat_AB4"),
                              error_type_rule="level ≥ 0.20, intermediate 0.15–0.20, structure < 0.15 (|ln(E_own/E0)|, h25b_targets)",
                              coverage_method=info["cov_method"], latex_check=lc, draft=False,
                              facts="scratchpad results_facts.md 2026-09-26 20:45 대조 통과(_qa/Table1_region_summary_values.txt)"))
    qa["latex_check"] = lc
    if qa.get("ok"):
        _drop_draft_caption()
        for p in list(OUT.glob(f"{NAME}_draft.*")) + list(OUT.glob(f"{NAME}_FAIL.*")) + list(OUT.glob(f"{NAME}_proto.*")) \
                + list(QA_DIR.glob(f"{NAME}_draft*")) + list((QA_DIR / "table1_tex").glob(f"{NAME}_draft*")) \
                + [QA_DIR / "Table1_nstar_probe.png"]:
            if p.exists():
                p.unlink()
    return qa
