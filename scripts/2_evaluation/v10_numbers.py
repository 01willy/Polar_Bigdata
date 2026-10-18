# -*- coding: utf-8 -*-
"""v10_numbers.py: 논문 초안 v10 의 연도 범위 · 보조 수치 매크로(영문 · 국문). 출력 paper/manuscript/shared_v10/numbers_v10.tex, docs/numbers_v10.json.
연도 범위는 자료에서 센다(라벨 표, 제품 표, 연도별 평가 표, 지도 메타). 실행: python3 scripts/2_evaluation/v10_numbers.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "data" / "processed"; N = {}
RK = {"Alaska": "Ak", "Lena": "Le", "Canada": "Ca"}


def put(k, v):
    assert re.fullmatch(r"[A-Za-z]+", k), k; N[k] = v


def ranges(years):
    """정렬된 연도 목록 → 연속 구간 문자열 목록(예: ['2013–2017', '2020'])."""
    ys = sorted(set(int(y) for y in years)); out = []; s = p = ys[0]
    for y in ys[1:]:
        if y == p + 1:
            p = y; continue
        out.append(f"{s}–{p}" if p > s else f"{s}"); s = p = y
    out.append(f"{s}–{p}" if p > s else f"{s}")
    return out


def en_join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def ko_join(parts):
    return parts[0] + "년" if len(parts) == 1 else ", ".join(p + "년" for p in parts[:-1]) + "과 " + parts[-1] + "년"


lab = pd.read_csv(P / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
for r, g in lab.groupby("region"):
    rg = ranges(g.year.unique()); put(f"yrs{RK[r]}", en_join(rg)); put(f"yrs{RK[r]}ko", ko_join(rg))
    put(f"yfirst{RK[r]}", str(int(g.year.min()))); put(f"ylast{RK[r]}", str(int(g.year.max())))
# 공통 셀(Wei · Liu · CCI v4 모두 같은 해 값) 연도
pr = pd.read_csv(P / "v5" / "products_cell_year_v5.csv")
w = lab[["loc_id", "year", "region"]].copy()
for p in ("wei", "liu", "cci4y"):
    q = pr[(pr["product"] == p) & pr.value_cm.notna()][["loc_id", "year"]]; q[p] = 1
    w = w.merge(q, on=["loc_id", "year"], how="left")
cc = w.dropna(subset=["wei", "liu", "cci4y"])
put("yrscc", f"{int(cc.year.min())}–{int(cc.year.max())}")
for r, g in cc.groupby("region"):
    put(f"yrscc{RK[r]}", f"{int(g.year.min())}–{int(g.year.max())}")
# 제품별 비교에 쓴 연도(라벨 셀에 값이 있는 해)
for p, k in (("wei", "Wei"), ("liu", "Liu"), ("cci4y", "Ccifour"), ("cci5y", "Ccifive"), ("yk", "Yk")):
    q = lab.merge(pr[(pr["product"] == p) & pr.value_cm.notna()][["loc_id", "year"]], on=["loc_id", "year"])
    if len(q):
        put(f"yrsprod{k}", f"{int(q.year.min())}–{int(q.year.max())}")
# 연도별 평가(셀-연도 30개 이상인 해)
Y = pd.read_csv(ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F5_yearwise_rmse.csv"); Y = Y[(Y.mask_km == 0) & Y.shown]
for r, g in Y.groupby("region"):
    rg = ranges(g.year.unique()); put(f"yrsyw{RK[r]}", en_join(rg)); put(f"yrsyw{RK[r]}ko", ko_join(rg)); put(f"nyw{RK[r]}", str(g.year.nunique()))
# 지도 연도, 구간 폭, 적용 범위
for r in ("Alaska", "Lena"):
    m = json.load(open(P / "v5" / "maps" / r / "meta.json"))
    put(f"yrsmap{RK[r]}", f"{min(m['years'])}–{max(m['years'])}"); put(f"hw{RK[r]}", f"{m['interval']['half_width_90']:.1f}")
# 관측일 누적 TDD / 계절 전체 TDD 비(라벨 셀-연도 중앙값)
import sys; sys.path.insert(0, str(ROOT / "src")); import polar.testbed_v5 as T  # noqa: E402
df = T.load_v5(P / "v5", products=False); d = T.with_forcing(df, "year_obs")
for r, g in d.groupby("region"):
    put(f"tddfrac{RK[r]}", f"{np.nanmedian(g.tdd_obs_frac):.2f}")
# 관측일(DOY) 분포
for r, g in lab.groupby("region"):
    doy = g.obs_doy.dropna()
    put(f"doymed{RK[r]}", f"{doy.median():.0f}" if len(doy) else "–")
# 외부 CALM 자료 연도
ex = pd.read_csv(P / "v9_ext" / "labels_cell_year_v5.csv"); put("yrsext", f"{int(ex.year.min())}–{int(ex.year.max())}")
# 연도별 지도의 지역 평균 편차(그 해 r1 − 2011–2024 평균 r1, 지도 셀 평균)
for r in ("Alaska", "Lena"):
    md = P / "v5" / "maps" / r; mean = pd.read_csv(md / "mean_2011_2024.csv.gz", usecols=["cell_id", "r1"]).set_index("cell_id").r1
    m = json.load(open(md / "meta.json")); an = {}
    for yr in m["years"]:
        v = pd.read_csv(md / f"pred_{yr}.csv.gz", usecols=["cell_id", "r1"]).set_index("cell_id").r1
        an[yr] = float((v - mean.reindex(v.index)).mean())
    ymin = min(an, key=an.get); ymax = max(an, key=an.get)
    put(f"anmin{RK[r]}", f"{an[ymin]:+.1f}".replace("-", "−")); put(f"anminyr{RK[r]}", str(ymin))
    put(f"anmax{RK[r]}", f"{an[ymax]:+.1f}".replace("-", "−")); put(f"anmaxyr{RK[r]}", str(ymax))
    put(f"anrange{RK[r]}", f"{an[ymax] - an[ymin]:.1f}")
    put(f"ncellmap{RK[r]}", f"{len(mean):,}")
# 지역 평균 편차: 예측 대 관측 셀 평균(셀-연도 50개 이상인 해, F7 원천 JSON)
F7 = json.load(open(ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F7_series_significance.json"))
for r in ("Alaska", "Lena"):
    pa = F7["b"][r]["predicted_anomaly_cm"]; oa = F7["b"][r]["observed_anomaly_cm"]; yrs = sorted(set(pa) & set(oa))
    pv = np.array([pa[y] for y in yrs]); ov = np.array([oa[y] for y in yrs])
    put(f"anobsr{RK[r]}", f"{np.corrcoef(pv, ov)[0, 1]:.2f}"); put(f"anobssame{RK[r]}", str(int((np.sign(pv) == np.sign(ov)).sum()))); put(f"anobsn{RK[r]}", str(len(yrs)))
    put(f"anobsrange{RK[r]}", f"{ov.max() - ov.min():.1f}"); put(f"anpredrange{RK[r]}", f"{pv.max() - pv.min():.1f}")
    put(f"anobsyrs{RK[r]}", f"{yrs[0]}–{yrs[-1]}")
# 라벨 정제 단계별 개수(labels_cell_year_v5_meta.json)
LM = json.load(open(P / "v5" / "labels_cell_year_v5_meta.json"))["counts"]
put("qcAboveRaw", f"{LM['above']['raw']:,}"); put("qcAboveKept", f"{LM['above']['qc']:,}"); put("qcAboveEarly", f"{LM['above']['early_excluded']:,}")
put("qcCalmSy", f"{LM['calm']['site_years']:,}"); put("qcCalmNoloc", f"{LM['calm']['no_loc']:,}"); put("qcCalmCy", f"{LM['calm']['cell_years']:,}")
put("qcLenaRaw", f"{LM['allena']['raw']:,}"); put("qcLenaEos", f"{LM['allena']['eos_qc']:,}"); put("qcLenaCy", f"{LM['allena']['cell_years']:,}")
put("qcTotalCy", f"{LM['total']['cell_years']:,}"); put("qcTotalCyIn", f"{LM['total']['cell_years_in_forcing']:,}")
EM = json.load(open(P / "v9_ext" / "build_meta.json"))
put("extRawSy", f"{EM['raw_site_years']:,}"); put("extCens", str(EM["censored_removed"])); put("extQcRm", str(EM["qc_comment_removed"]))
# 점추정(pt)과 절댓값(abs) 변형: 구간이 붙은 v6 · v9 매크로에서 만든다(본문 · 초록에서 점추정만 쓸 때)
for src in ("numbers_v6.json", "numbers_v9.json"):
    for k, v in json.load(open(ROOT / "docs" / src, encoding="utf-8")).items():
        if not isinstance(v, str):
            continue
        pt = v.split(" [")[0].strip()
        if " [" in v and re.fullmatch(r"[A-Za-z]+", k + "pt"):
            N.setdefault(k + "pt", pt)
        if re.fullmatch(r"[+−-]?[0-9.,]+", pt) and re.fullmatch(r"[A-Za-z]+", k + "abs"):
            N.setdefault(k + "abs", pt.lstrip("+−-"))
json.dump(N, open(ROOT / "docs" / "numbers_v10.json", "w"), ensure_ascii=False, indent=0)
tex = ["% numbers_v10.tex: scripts/2_evaluation/v10_numbers.py 가 만든다(연도 범위 · 보조 수치). 손으로 고치지 않는다."]
for k, v in N.items():
    tex.append("\\newcommand{\\n" + k + "}{" + v.replace("%", "\\%") + "}")
(ROOT / "paper" / "manuscript" / "shared_v10" / "numbers_v10.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print(json.dumps(N, ensure_ascii=False, indent=0))
