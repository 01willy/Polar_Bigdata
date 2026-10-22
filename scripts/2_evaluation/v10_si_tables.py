# -*- coding: utf-8 -*-
"""v10_si_tables.py: 보충 표 S4(방법 × 검증 분할 × 기후 입력 RMSE)와 S5(제품 비교)를 결과 CSV 에서 LaTeX 로 만든다(영문 · 국문 머리글).
입력 data/processed/v5/runs/core_v6/{summary_metrics,common_cells_eval}.csv. 출력 paper/manuscript/shared_v10/si_tables_{en,ko}.tex
실행: python3 scripts/2_evaluation/v10_si_tables.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; RUN = ROOT / "data" / "processed" / "v5" / "runs" / "core_v6"; OUT = ROOT / "paper" / "manuscript" / "shared_v11"   # v15.2: the file the v15 supplement reads
REG = [("Alaska", "Alaska", "알래스카"), ("Lena", "Lena Delta", "레나 델타"), ("Canada", "Canada (NW)", "캐나다 북서부")]
METH = [("R1", "catboost", 0.5, "Physics + residual ML", "물리 + 잔차 ML"), ("P1", "-", -1.0, "Stefan, recalibrated", "Stefan(재보정)"),
        ("P0", "-", -1.0, "Stefan, source coeff.", "Stefan(원천 계수)"), ("Ku", "-", -1.0, "Kudryavtsev, scaled", "Kudryavtsev(척도 보정)"),
        ("D0", "catboost", -1.0, "Direct ML", "직접 ML"), ("F1", "catboost", -1.0, "Physics-input ML", "물리 입력 ML"),
        ("D1", "catboost", -1.0, "Physics pseudo-labels", "물리 유사라벨")]                                   # v15.2: physics-residual first
SCH = [("random", "Random", "무작위"), ("block", "Spatial block", "공간 블록"), ("year", "Year", "연도"), ("block_year", "Block-year", "블록-연도"), ("region", "Region held out", "지역 제외")]
FOR = [("year_obs", "Year-specific (to observation day)", "관측 연도(관측일까지)"), ("year", "Year-specific (full season)", "관측 연도(계절 전체)"), ("clim", "Mean 2011–2024", "평년(2011–2024 평균)")]


def f1(v):
    return "--" if pd.isna(v) else f"{v:.1f}"


SHORT_KO = {"P0": "Stefan 원천", "P1": "Stefan 재보정", "Ku": "Kudryavtsev", "D0": "직접 ML", "F1": "물리 입력 ML", "D1": "유사라벨", "R1": "물리 + 잔차 ML"}
SHORT_EN = {"P0": "Stefan, source", "P1": "Stefan, recal.", "Ku": "Kudryavtsev", "D0": "Direct ML", "F1": "Phys.-input ML", "D1": "Pseudo-labels", "R1": "Phys. + residual ML"}


def s4(lang):
    """지역별로 쌓은 표: 행 = 방법(물리 잔차 먼저), 열 = 검증 설계 5종, 관측 연도 입력만(평년 입력 효과는 본문 그림 3a; 전체 값은 결과 표)."""
    m = pd.read_csv(RUN / "summary_metrics.csv"); m = m[(m.featset == "F8TS") & (m.forcing == "year_obs")]
    L = 0 if lang == "en" else 1
    cap = ("RMSE (cm) of all methods under the five validation designs, from the least to the most demanding (climate input of the observation year accumulated to the observation day; "
           "labels 2010--2024; out-of-fold predictions, three seeds averaged). Random, site-years assigned to folds at random; spatial block, 0.5$^\\circ$ blocks held out; year, observation years held out; "
           "block-year, blocks and years held out together; region held out, each region scored with models trained on the other two regions (the recalibrated Stefan model then equals the "
           "source-coefficient model). Reading: the random and year designs reward the learners that fit the sites, and the region hold-out exposes them (direct ML \\nrmseLeregDz{} cm in the Lena Delta), "
           "whereas the physics-anchored models change least across designs. Values for the mean climate input are in the deposited result tables." if lang == "en" else
           "다섯 검증 설계(덜 엄격한 것부터)에서 모든 방법의 RMSE(cm). 관측 연도의 기후 입력(관측일까지 누적), 라벨 2010--2024년, 겹 밖 예측, 씨앗 3개 평균. 무작위는 지점-연도를 무작위로 겹에 배정, 공간 블록은 0.5$^\\circ$ 블록 제외, "
           "연도는 관측 연도 제외, 블록-연도는 블록과 연도를 함께 제외, 지역 제외는 다른 두 지역으로 학습한 모델로 각 지역을 채점한 것이다(이때 재보정 Stefan 모델은 원천 계수 모델과 같다). "
           "읽는 법: 무작위 · 연도 설계는 지점을 맞추는 학습기에 유리하고 지역 제외는 그 학습기를 드러내며(레나 델타 직접 ML \\nrmseLeregDz{} cm), 물리에 고정한 모델은 설계에 따른 변화가 가장 작다. 평년 기후 입력 값은 기탁 결과 표에 있다.")
    out = [r"\begin{table}[htbp]", r"\caption{" + cap + r"}\label{tab:s4}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l" + "r" * len(SCH) + "@{}}", r"\toprule"]
    out.append(("Method" if lang == "en" else "방법") + " & " + " & ".join((sc[1] if lang == "en" else sc[2]) for sc in SCH) + r" \\")
    for rk, ren, rko in REG:
        out.append(r"\midrule")
        out.append(r"\multicolumn{6}{@{}l}{\textit{" + (ren if lang == "en" else rko) + r"}} \\")
        for mk, lr, lam, men, mko in METH:
            row = [men if lang == "en" else mko]
            for sc, *_ in SCH:
                q = m[(m.method == mk) & (m.scheme == sc) & (m.learner == lr) & ((m.lam - lam).abs() < 1e-9) & (m.region == rk)]
                row.append(f1(q.rmse.iloc[0]) if len(q) else "--")
            out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def s5(lang):
    """지역별로 쌓은 표: 행 = 계열, 열 = RMSE 전체 · RMSE 25 km 밖 · 편향 전체 · n."""
    c = pd.read_csv(RUN / "common_cells_eval.csv")
    L = 0 if lang == "en" else 1
    ser = [("R1 (CatBoost)", "Physics + residual ML", "물리 + 잔차 ML"), ("P1 (Stefan, recal.)", "Stefan, recalibrated", "Stefan(재보정)"),
           ("Wei 2026", "Wei et al. 2026", "Wei 등 2026"), ("Liu 2024", "Liu et al. 2024", "Liu 등 2024"), ("ESA CCI v4", "ESA CCI v4", "ESA CCI v4"),
           ("ESA CCI v5", "ESA CCI v5", "ESA CCI v5"), ("Aalto 2018", "Aalto et al. 2018", "Aalto 등 2018")]
    cap = ("Comparison with the products on the common site-years, that is, the site-years for which Wei et al., Liu et al. and ESA CCI v4 all provide a value of the observation year (2010--2020; "
           "each site-year is scored with the product value and the out-of-fold prediction of its own year). All = all common site-years; $>$25 km = site-years in cells more than 25 km from the "
           "training sites of Wei et al. RMSE and mean bias (prediction minus observation) in cm. Our models are spatial-block out-of-fold predictions." if lang == "en" else
           "공통 지점-연도(Wei 등, Liu 등, ESA CCI v4 모두 관측 연도 값을 제공하는 지점-연도, 2010--2020년; 각 지점-연도는 그 해의 제품 값과 겹 밖 예측으로 채점)에서의 제품 비교. 전체 = 모든 공통 지점-연도, 25 km 밖 = Wei 등 학습 지점에서 25 km 넘게 떨어진 셀의 지점-연도. 우리 모델은 공간 블록 겹 밖 예측이다. RMSE · 편향(예측 − 관측) 단위 cm.")
    hdr = ("Series", "RMSE, all", "RMSE, $>$25 km", "Bias, all") if lang == "en" else ("계열", "RMSE 전체", "RMSE 25 km 밖", "편향 전체")
    out = [r"\begin{table}[htbp]", r"\caption{" + cap + r"}\label{tab:s6}", r"\scriptsize", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", " & ".join(hdr) + r" \\"]
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
    (OUT / f"si_table_designs_{lang}.tex").write_text("% si_table_designs_" + lang + ".tex: scripts/2_evaluation/v10_si_tables.py 가 만든다 (보충 표 S4).\n" + s4(lang) + "\n", encoding="utf-8")
    (OUT / f"si_table_products_{lang}.tex").write_text("% si_table_products_" + lang + ".tex: scripts/2_evaluation/v10_si_tables.py 가 만든다 (보충 표 S6).\n" + s5(lang) + "\n", encoding="utf-8")
    print("written", lang)
