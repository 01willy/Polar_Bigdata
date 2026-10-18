# -*- coding: utf-8 -*-
"""v9_numbers.py: v9 추가 실험·질의 점검 수치 매크로 → docs/numbers_v9.json(덱 {nMacro} 토큰, 글자만 쓰는 이름).
입력 data/processed/v5/runs/extra_v9/{pa_metrics,pa_deltas,ext_metrics,ext_deltas,shap_groups}.csv, data/processed/v9_qa/*.csv, qa_summary.json
실행: python3 scripts/2_evaluation/v9_numbers.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; X = ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9"; Q = ROOT / "data" / "processed" / "v9_qa"
N = {}
RK = {"Alaska": "Ak", "Lena": "Le", "Canada": "Ca"}
SUBK = {"W Siberia": "Ws", "C Siberia": "Cs", "European Russia": "Er", "NE Siberia": "Ns", "Chukotka": "Ch", "Kamchatka": "Ka", "Svalbard": "Sv", "Greenland": "Gr", "All sites": "All"}
MK = {"R1": "Ro", "P1": "Po", "P0": "Pz", "RP": "Rp", "RP1": "Rpo", "Pcal": "Pcal", "Praw": "Praw", "R1x": "Rox", "D0": "Dz", "F1": "Fo", "R1loc": "Roloc", "P1loc": "Poloc",
      "wei": "Wei", "liu": "Liu", "cci5y": "Ccifive", "cci4y": "Ccifour", "aalto": "Aalto", "Wei": "Wei", "Liu": "Liu", "P1n": "Pn", "D0S": "Ds", "R1S": "Rs"}
PK = {"wei": "", "liu": "Liu"}


def m(x):
    return f"{x:.1f}".replace("-", "−")


def d(r):
    return f"{r.delta:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]".replace("-", "−")


def put(k, v):
    assert re.fullmatch(r"[A-Za-z]+", k), k; N[k] = v


pm = pd.read_csv(X / "pa_metrics.csv")
for _, r in pm.iterrows():
    put(f"pa{RK[r.region]}{MK[r.method]}{PK[r['product']]}", m(r.rmse)); put(f"pan{RK[r.region]}{PK[r['product']]}", f"{int(r.n):,}")
for _, r in pd.read_csv(X / "pa_deltas.csv").iterrows():
    put(f"pad{RK[r.region]}{MK[r.A]}{MK[r.B]}{PK[r['product']]}", d(r))
em = pd.read_csv(X / "ext_metrics.csv"); em = em[em.rows == "own"]
for _, r in em.iterrows():
    sub = "All" if r.subset == "in_range" else ("Allx" if r.subset == "all" else SUBK.get(r.subset.split("|")[-1]))
    if sub and r.series in MK:
        put(f"ext{sub}{MK[r.series]}", m(r.rmse)); put(f"extn{sub}", f"{int(em[(em.subset == r.subset) & (em.series == 'R1')].n.iloc[0]):,}")
for _, r in pd.read_csv(X / "ext_deltas.csv").iterrows():
    put(f"extd{'All' if r.subset == 'in_range' else 'Allx'}{MK[r.A]}{MK[r.B]}", d(r))
for _, r in pd.read_csv(X / "shap_groups.csv").iterrows():
    put(f"shap{RK[r.region]}{'Res' if r.kind == 'residual' else 'Dir'}{r.group.capitalize()}", f"{100 * r.share:.0f}")
tv = pd.read_csv(Q / "transfer_vs_products.csv")
for _, r in tv.iterrows():
    nn = {0: "zero", 25: "twfive"}[int(r.n_label)]; p = "Wei" if r["product"] == "wei" else "Liu"
    put(f"tlvp{RK[r.region]}{MK[r.method]}{p}{nn}", d(r)); put(f"tlvprmse{RK[r.region]}{MK[r.method]}{p}{nn}", m(r.rmse_method)); put(f"tlvprmse{RK[r.region]}{p}{nn}", m(r.rmse_product))
lb = pd.read_csv(Q / "lat_band_bias.csv")
for _, r in lb.iterrows():
    tag = {"(70, 72]": "North", "(68, 70]": "Foot", "(55, 64]": "South", "(64, 66]": "Int", "(66, 68]": "Mid", "(69, 72]": "South", "(72, 74]": "North", "(60, 64]": "South"}.get(r.band)
    if tag:
        for s in ("R1", "P1", "wei", "liu", "cci4y"):
            put(f"bias{RK[r.region]}{tag}{MK[s]}", f"{r[f'{s}_bias']:+.1f}".replace("-", "−") if pd.notna(r[f"{s}_bias"]) else "–")
        put(f"biasobs{RK[r.region]}{tag}", m(r.obs_mean))
me = pd.read_csv(Q / "map_band_alaska2014_by_elev.csv")
for _, r in me.iterrows():
    tag = {"(59, 64]": "South", "(64, 66]": "Int", "(66, 68]": "Mid", "(68, 70]": "Foot", "(70, 72]": "North"}[r.band] + ("Mtn" if ">" in r.mtn else "Low")
    put(f"mapAk{tag}Wei", f"{r.wei_minus:+.1f}".replace("-", "−")); put(f"mapAk{tag}Liu", f"{r.liu_minus:+.1f}".replace("-", "−"))
S = json.load(open(Q / "qa_summary.json"))
sw = S["sw_alaska"]; put("swpfrmed", f"{sw['pfr_median']:.0f}"); put("swpfrge", f"{100 * sw['share_pfr_ge50']:.0f}"); put("maplatmin", f"{sw['map_lat_min']:.1f}")
for r, v in S["dist_wei_train"].items():
    put(f"distwei{RK[r]}med", f"{v['median_km']:.0f}"); put(f"distwei{RK[r]}le", f"{100 * v['share_le25']:.0f}"); put(f"distwei{RK[r]}min", f"{v['min_km']:.0f}")
lab = S["external_calm"]; put("extcells", str(lab["cells"])); put("extcy", f"{lab['cell_years']:,}"); put("extblocks", str(lab["blocks"])); put("extsubs", str(len(lab["by_region"])))
json.dump(N, open(ROOT / "docs" / "numbers_v9.json", "w"), ensure_ascii=False, indent=0)
print(len(N), "macros"); print({k: N[k] for k in list(N)[:12]})
print({k: v for k, v in N.items() if k.startswith(("padLe", "tlvpLe", "extAll", "distwei", "mapAkFoot"))})

# F8 제품 지도 쪽(R11·R12)의 관측 셀 RMSE(공통 셀, 지도 값 = 계절 말; oof = 공간 블록 검증 예측)
F8 = json.load(open(ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F8_product_maps.json"))
for r in ("Alaska", "Lena"):
    for k, v in F8["rows"][r]["residual"]["rmse_common_cells_cm"].items():
        kk = {"ours": "Ours", "oof": "Oof", "wei": "Wei", "liu": "Liu"}.get(k)
        if kk and v is not None:
            put(f"fe{RK[r]}{kk}", m(float(v)))
json.dump(N, open(ROOT / "docs" / "numbers_v9.json", "w"), ensure_ascii=False, indent=0)
print({k: v for k, v in N.items() if k.startswith("fe")})

# 연도별 우위 횟수(F5 R05 원천 표, 마스크 0, 표시 연도)
Y = pd.read_csv(ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F5_yearwise_rmse.csv")
Y = Y[(Y.mask_km == 0) & Y.shown]
for r, g in Y.groupby("region"):
    put(f"yw{RK[r]}n", str(len(g)))
    for k, col in (("Wei", "rmse_wei"), ("Liu", "rmse_liu"), ("Po", "rmse_p1")):
        q = g[g[col].notna()]; put(f"yw{RK[r]}Ro{k}win", str(int((q.rmse_r1 < q[col]).sum()))); put(f"yw{RK[r]}Ro{k}n", str(len(q)))
# 조건 열지도(F4e): 0 을 제외하는 칸 수, 칸 수, 3분위 Stefan RMSE 와 이득의 상관
G = pd.read_csv(ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F4e_gain_conditions.csv")
G = G[(G.tercile >= 0) & G.shown]
for r, g in G.groupby("region"):
    put(f"gc{RK[r]}sig", str(int(g.excludes_0.sum()))); put(f"gc{RK[r]}cells", str(len(g)))
    put(f"gc{RK[r]}maxabs", m(g.delta.abs().max())); put(f"gc{RK[r]}corr", f"{g[['rmse_stefan', 'delta']].corr().iloc[0, 1]:.2f}".replace("-", "−"))
json.dump(N, open(ROOT / "docs" / "numbers_v9.json", "w"), ensure_ascii=False, indent=0)
print({k: v for k, v in N.items() if k.startswith(("yw", "gc"))})

# LaTeX 매크로(답변서용): \n<Key>
tex = ["% numbers_v9.tex: scripts/2_evaluation/v9_numbers.py 가 docs/numbers_v9.json 에서 만든다. 손으로 고치지 않는다."]
for k, v in N.items():
    vv = v.replace("%", "\\%")
    tex.append("\\newcommand{\\n" + k + "}{" + vv + "}")
(ROOT / "outputs" / "report_v9").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "report_v9" / "numbers_v9.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print("tex macros", len(tex) - 1)

# 답변서 보조 수치: 라벨 분포, 위도대별 편향 표본 수, Wei 학습 지점(알래스카 70°N 이북)
import numpy as np
LAB = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv"); LAB = LAB[LAB.in_forcing == 1]
put("labpten", f"{np.percentile(LAB.alt_cm, 10):.0f}"); put("labpninety", f"{np.percentile(LAB.alt_cm, 90):.0f}")
for _, r in lb.iterrows():
    tag = {"(70, 72]": "North", "(68, 70]": "Foot", "(55, 64]": "South", "(64, 66]": "Int", "(66, 68]": "Mid", "(69, 72]": "South", "(72, 74]": "North", "(60, 64]": "South"}.get(r.band)
    if tag:
        for s in ("wei", "liu"):
            put(f"biasn{RK[r.region]}{tag}{MK[s]}", f"{int(r[f'{s}_n']):,}")
W = pd.read_excel(ROOT / "data" / "raw" / "qtp_alt_open_2026-09-29" / "essd-2026-447_supplement" / "ALT_variables.xlsx", usecols=["LAT", "LONG"]).drop_duplicates()
put("weisitesAkN", str(int(((W.LONG.between(-170, -140)) & (W.LAT >= 70)).sum())))
json.dump(N, open(ROOT / "docs" / "numbers_v9.json", "w"), ensure_ascii=False, indent=0)
tex = ["% numbers_v9.tex: scripts/2_evaluation/v9_numbers.py 가 docs/numbers_v9.json 에서 만든다. 손으로 고치지 않는다."]
for k, v in N.items():
    tex.append("\\newcommand{\\n" + k + "}{" + v.replace("%", "\\%") + "}")
(ROOT / "outputs" / "report_v9" / "numbers_v9.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print({k: N[k] for k in ("labpten", "labpninety", "biasnLeSouthWei", "biasnLeNorthWei", "weisitesAkN")})

# 외부 지역 소수 라벨 적응(EXTFT, Rescale dfuMW): 세 제품 값이 모두 있는 같은 행(F10 FT 그림 원천 표)
FT = ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F10_extft.csv"
if FT.exists():
    ft = pd.read_csv(FT); ft["n_label"] = ft.n_label.astype(str)
    SUBK2 = {"all": "All", "W Siberia": "Ws", "C Siberia": "Cs", "European Russia": "Er", "NE Siberia": "Ns", "Chukotka": "Ch", "Svalbard": "Sv", "Kamchatka": "Ka", "Greenland": "Gr"}
    NK = {"0": "zero", "5": "five", "10": "ten", "25": "twfive", "50": "fifty", "all": "all"}
    SK = {"R1S": "Rs", "P1n": "Pn", "D0S": "Ds", "wei": "Wei", "liu": "Liu", "cci5y": "Ccifive"}
    for _, r in ft.iterrows():
        if r.scope in SUBK2 and r.series in SK:
            put(f"ft{SUBK2[r.scope]}{SK[r.series]}{NK[r.n_label]}", m(r.rmse))
    put("ftrows", f"{int(ft[ft.scope == 'all'].n.iloc[0]):,}")
    D = pd.read_csv(ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9" / "extft_deltas.csv"); D["n_label"] = D.n_label.astype(str)
    for _, r in D.iterrows():
        k = f"ftd{SK.get(r.A, r.A)}{({'ESA CCI v5': 'Ccifive', 'Wei 2026': 'Wei', 'Liu 2024': 'Liu'}).get(r.B, SK.get(r.B, r.B))}{NK[r.n_label]}"
        if re.fullmatch(r"[A-Za-z]+", k):
            put(k, d(r))
    json.dump(N, open(ROOT / "docs" / "numbers_v9.json", "w"), ensure_ascii=False, indent=0)
    tex = ["% numbers_v9.tex: scripts/2_evaluation/v9_numbers.py 가 docs/numbers_v9.json 에서 만든다. 손으로 고치지 않는다."]
    for k, v in N.items():
        tex.append("\\newcommand{\\n" + k + "}{" + v.replace("%", "\\%") + "}")
    (ROOT / "outputs" / "report_v9" / "numbers_v9.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print({k: N[k] for k in ("ftAllRszero", "ftAllRsfive", "ftAllCcifivezero", "ftAllWeizero", "ftAllLiuzero", "ftWsRszero", "ftWsRsfive", "ftdRsCcifivefive", "ftrows")})
