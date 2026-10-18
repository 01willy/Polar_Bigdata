# -*- coding: utf-8 -*-
"""v10_si_tables.py: 보충 표 S4(방법 × 검증 분할 × 기후 입력 RMSE)와 S5(제품 비교)를 결과 CSV 에서 LaTeX 로 만든다(영문 · 국문 머리글).
입력 data/processed/v5/runs/core_v6/{summary_metrics,common_cells_eval}.csv. 출력 paper/manuscript/shared_v10/si_tables_{en,ko}.tex
실행: python3 scripts/2_evaluation/v10_si_tables.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; RUN = ROOT / "data" / "processed" / "v5" / "runs" / "core_v6"; OUT = ROOT / "paper" / "manuscript" / "shared_v10"
REG = [("Alaska", "Alaska", "알래스카"), ("Lena", "Lena Delta", "레나 델타"), ("Canada", "Canada (NW)", "캐나다 북서부")]
METH = [("P0", "-", -1.0, "Stefan, source coeff.", "Stefan(원천 계수)"), ("P1", "-", -1.0, "Stefan, recalibrated", "Stefan(재보정)"),
        ("Ku", "-", -1.0, "Kudryavtsev, scaled", "Kudryavtsev(척도 보정)"), ("D0", "catboost", -1.0, "Direct ML", "직접 ML"),
        ("F1", "catboost", -1.0, "Physics-input ML", "물리 입력 ML"), ("D1", "catboost", -1.0, "Physics pseudo-labels", "물리 유사라벨"),
        ("R1", "catboost", 0.5, "Physics + residual ML", "물리 + 잔차 ML")]
SCH = [("random", "Random", "무작위"), ("block", "Block", "블록"), ("year", "Year", "연도"), ("block_year", "Block × year", "블록 × 연도"), ("region", "Region", "지역")]
FOR = [("year_obs", "Year-specific (to observation day)", "관측 연도(관측일까지)"), ("year", "Year-specific (full season)", "관측 연도(계절 전체)"), ("clim", "Mean 2011–2024", "평년(2011–2024 평균)")]


def f1(v):
    return "--" if pd.isna(v) else f"{v:.1f}"


SHORT_KO = {"P0": "Stefan 원천", "P1": "Stefan 재보정", "Ku": "Kudryavtsev", "D0": "직접 ML", "F1": "물리 입력 ML", "D1": "유사라벨", "R1": "물리 + 잔차 ML"}
SHORT_EN = {"P0": "Stefan, source", "P1": "Stefan, recal.", "Ku": "Kudryavtsev", "D0": "Direct ML", "F1": "Phys.-input ML", "D1": "Pseudo-labels", "R1": "Phys. + residual ML"}


def s4(lang):
    """지역별로 쌓은 표: 행 = 방법, 열 = 검증 설계 5종 × 기후 입력 2종(관측 연도 · 평년). 계절 전체 입력 값은 결과 표에 있다."""
    m = pd.read_csv(RUN / "summary_metrics.csv"); m = m[m.featset == "F8TS"]
    L = 0 if lang == "en" else 1
    fo2 = [FOR[0], FOR[2]]
    cap = ("RMSE (cm) of all methods by validation design, for the year-specific climate input (observation year, accumulated to the observation day) and the mean climate input (2011--2024 mean). "
           "Labels 2010--2024, out-of-fold predictions, three seeds averaged. Region hold-out scores each region with models trained on the other two regions. Values for the full-season input are in the deposited result tables." if lang == "en" else
           "방법별 · 검증 설계별 RMSE(cm). 관측 연도 기후 입력(관측일까지 누적)과 평년 기후 입력(2011--2024년 평균). 라벨 2010--2024년, 겹 밖 예측, 씨앗 3개 평균. 지역 홀드아웃은 다른 두 지역으로 학습한 모델로 각 지역을 채점하였다. 계절 전체 입력의 값은 공개 결과 표에 있다.")
    out = [r"\begin{table}[p]", r"\caption{" + cap + r"}\label{tab:s4}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
           r"\begin{tabular}{@{}l" + "r" * (len(SCH) * 2) + "@{}}", r"\toprule"]
    HEAD = {"en": {"year_obs": "Year-specific input", "clim": "Mean input 2011--2024"}, "ko": {"year_obs": "관측 연도 입력", "clim": "평년 입력(2011--2024년)"}}[lang]
    out.append(("Method" if lang == "en" else "방법") + " & " + " & ".join(r"\multicolumn{5}{c}{" + HEAD[fo] + "}" for fo, _, _ in fo2) + r" \\")
    SH = {"en": ["Rand.", "Block", "Year", r"B$\times$Y", "Reg."], "ko": ["무작위", "블록", "연도", r"블록$\times$연도", "지역"]}[lang]
    out.append(" & " + " & ".join(SH[j] for _ in fo2 for j in range(len(SCH))) + r" \\")
    for rk, ren, rko in REG:
        out.append(r"\midrule")
        out.append(r"\multicolumn{11}{@{}l}{\textit{" + (ren if lang == "en" else rko) + r"}} \\")
        for mk, lr, lam, men, mko in METH:
            row = [SHORT_EN.get(mk, men) if lang == "en" else SHORT_KO.get(mk, mko)]
            for fo, *_ in fo2:
                for sc, *_ in SCH:
                    q = m[(m.forcing == fo) & (m.method == mk) & (m.scheme == sc) & (m.learner == lr) & ((m.lam - lam).abs() < 1e-9) & (m.region == rk)]
                    row.append(f1(q.rmse.iloc[0]) if len(q) else "--")
            out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def s5(lang):
    """지역별로 쌓은 표: 행 = 계열, 열 = RMSE 전체 · RMSE 25 km 밖 · 편향 전체 · n."""
    c = pd.read_csv(RUN / "common_cells_eval.csv")
    L = 0 if lang == "en" else 1
    ser = [("P1 (Stefan, recal.)", "Stefan, recalibrated", "Stefan(재보정)"), ("R1 (CatBoost)", "Physics + residual ML", "물리 + 잔차 ML"), ("D0 (CatBoost)", "Direct ML", "직접 ML"),
           ("Wei 2026", "Wei et al. 2026", "Wei 등 2026"), ("Liu 2024", "Liu et al. 2024", "Liu 등 2024"), ("ESA CCI v4", "ESA CCI v4", "ESA CCI v4"),
           ("ESA CCI v5", "ESA CCI v5", "ESA CCI v5"), ("Aalto 2018", "Aalto et al. 2018", "Aalto 등 2018")]
    cap = ("Comparison with products on common cell-years (Wei et al., Liu et al. and ESA CCI v4 all available for the observation year; 2010--2020). "
           "All = all common cell-years; $>$25 km = cells more than 25 km from the training sites of Wei et al. RMSE and bias (prediction minus observation) in cm." if lang == "en" else
           "공통 셀-연도(Wei 등, Liu 등, ESA CCI v4 모두 관측 연도 값이 있는 셀-연도, 2010--2020년)에서의 제품 비교. 전체 = 모든 공통 셀-연도, 25 km 밖 = Wei 등 학습 지점에서 25 km 넘게 떨어진 셀. RMSE · 편향(예측 − 관측) 단위 cm.")
    hdr = ("Series", "RMSE, all", "RMSE, $>$25 km", "Bias, all") if lang == "en" else ("계열", "RMSE 전체", "RMSE 25 km 밖", "편향 전체")
    out = [r"\begin{table}[p]", r"\caption{" + cap + r"}\label{tab:s5}", r"\scriptsize", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", " & ".join(hdr) + r" \\"]
    for rk, ren, rko in REG:
        a0 = c[(c.region == rk) & (c.mask_km == 0)]; b0 = c[(c.region == rk) & (c.mask_km == 25)]
        nn = (f"$n$ = {int(a0.n.max()):,} / {int(b0.n.max()):,}" if len(a0) and len(b0) else "")
        out += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{" + (ren if lang == "en" else rko) + r"} (" + nn + r")} \\"]
        for key, en, ko in ser:
            a = a0[a0.series == key]; b = b0[b0.series == key]
            out.append(" & ".join([en if lang == "en" else ko, f1(a.rmse.iloc[0]) if len(a) else "--", f1(b.rmse.iloc[0]) if len(b) else "--",
                                   (f"{a.bias.iloc[0]:+.1f}".replace("-", "−") if len(a) else "--")]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


for lang in ("en", "ko"):
    (OUT / f"si_tables_{lang}.tex").write_text("% si_tables_" + lang + ".tex: scripts/2_evaluation/v10_si_tables.py 가 만든다.\n" + s4(lang) + "\n\n" + s5(lang) + "\n", encoding="utf-8")
    print("written", lang)
