# -*- coding: utf-8 -*-
"""v6_numbers.py: 원고·덱 수치 매크로 생성. 결과 표(CSV)에서 읽어 LaTeX 매크로(numbers.tex)와 JSON 으로 쓴다. 수치를 손으로 옮기지 않는다.

입력: data/processed/v5/runs/<run>/{summary_metrics, deltas_vs_ref, deltas_forcing, conditions_tdd, common_cells_eval, products_eval_mask25}.csv,
      <run>/yearwise/{multiyear_changes, yearwise_metrics}.csv, featsets 실행의 deltas_vs_ref.csv, 학습기 실행(rf, nn)의 deltas_vs_ref.csv
출력: paper/manuscript/en_v6/numbers.tex, docs/numbers_v6.json
실행: python3 scripts/2_evaluation/v6_numbers.py [--run core_v6] [--forcing year_obs]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; PROC = ROOT / "data" / "processed" / "v5"; RUNS = PROC / "runs"
REG = ["Alaska", "Lena", "Canada"]; RK = {"Alaska": "Ak", "Lena": "Le", "Canada": "Ca"}
MK = {"P0": "Pz", "P1": "Po", "Ku": "Ku", "Ed": "Ed", "PE": "Pe", "D0": "Dz", "D1": "Do", "F1": "Fo", "R1": "Ro", "R1cv": "Rc", "S": "St"}
SK = {"random": "rand", "block": "blk", "year": "yr", "block_year": "blkyr", "region": "reg"}
FK = {"year_obs": "obs", "year": "yr", "clim": "clim"}


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")


def sfmt(x, nd=1):
    """부호 있는 값."""
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:+.{nd}f}".replace("-", "−")
    return s


def ci(d, lo, hi, nd=1):
    return f"{sfmt(d, nd)} [{sfmt(lo, nd)}, {sfmt(hi, nd)}]"


def main(run, forcing, featset="F8TS", lam=0.5):
    out = {}; R = RUNS / run
    sm = pd.read_csv(R / "summary_metrics.csv"); dv = pd.read_csv(R / "deltas_vs_ref.csv"); df = pd.read_csv(R / "deltas_forcing.csv")
    main_rows = lambda t: t[(t.featset == featset) & ((t.method != "R1") | (t.lam == lam)) & (t.learner.isin(["catboost", "-"]))]
    sm = main_rows(sm); dv = main_rows(dv); df = main_rows(df)
    if "pair" in df:
        df = df[df.pair == f"{forcing}:clim"]
    # --- RMSE by region × scheme × method (주 강제력)
    for r in REG:
        for s_, sk in SK.items():
            for m, mk in MK.items():
                q = sm[(sm.region == r) & (sm.scheme == s_) & (sm.forcing == forcing) & (sm.method == m)]
                if len(q):
                    out[f"rmse{RK[r]}{sk}{mk}"] = fmt(q.rmse.iloc[0]); out[f"bias{RK[r]}{sk}{mk}"] = sfmt(q.bias.iloc[0])
                    out[f"ranom{RK[r]}{sk}{mk}"] = fmt(q.r_anom.iloc[0], 2)
        # 무작위 − 블록 (D0): 과대평가
        a = sm[(sm.region == r) & (sm.forcing == forcing) & (sm.method == "D0")].set_index("scheme").rmse
        if {"random", "block"} <= set(a.index):
            out[f"optim{RK[r]}Dz"] = fmt(a["block"] - a["random"])
    # --- 강제력 효과 (forcing − clim), 블록
    for r in REG:
        for m, mk in MK.items():
            q = df[(df.region == r) & (df.scheme == "block") & (df.method == m)]
            if len(q):
                q = q.iloc[0]; out[f"dforc{RK[r]}{mk}"] = ci(q.delta, q.lo, q.hi)
    # --- 방법 − P1 (블록·연도·지역)
    for r in REG:
        for s_, sk in SK.items():
            for m, mk in MK.items():
                q = dv[(dv.region == r) & (dv.scheme == s_) & (dv.forcing == forcing) & (dv.method == m) & (dv.ref == "P1")]
                if len(q):
                    q = q.iloc[0]; out[f"dref{RK[r]}{sk}{mk}"] = ci(q.delta, q.lo, q.hi); out[f"drefbeq{RK[r]}{sk}{mk}"] = ci(q.delta_beq, q.lo_beq, q.hi_beq)
    # --- TDD 3분위 (R1 − P1, 블록)
    c = pd.read_csv(R / "conditions_tdd.csv"); c = c[(c.scheme == "block") & (c.forcing == forcing) & (c.featset == featset) & (c.method == "R1") & (c.lam == lam)]
    for r in REG:
        for b, nm in ((0, "cold"), (1, "mid"), (2, "warm")):
            q = c[(c.region == r) & (c.tdd_bin == b)]
            if len(q):
                q = q.iloc[0]; out[f"dtdd{RK[r]}{nm}"] = ci(q.delta, q.lo, q.hi)
    # --- 공변량 사다리 (R1 − P1, 블록) : featsets 실행
    for t in ("modis_v6", "featsets_v6", "featsets", "featsets2", "featsets3"):
        f = RUNS / t / "deltas_vs_ref.csv"
        if f.exists():
            d = pd.read_csv(f); d = d[(d.scheme == "block") & (d.ref == "P1") & (d.method == "R1") & (d.lam == lam)]
            if "forcing" in d and (d.forcing == forcing).any():
                d = d[d.forcing == forcing]
            for r in REG:
                for fs, fk in (("F8", "Fclim"), ("F8T", "Fterr"), ("F8TS", "Fsoil"), ("F8TS_M", "Fmodis"), ("F8TS_X", "Fhires"), ("F8TS_ALL", "Fall"), ("F8TS_L", "Flag")):
                    q = d[(d.region == r) & (d.featset == fs)]
                    k = f"dfeat{RK[r]}{fk}"
                    if len(q) and k not in out:
                        q = q.iloc[0]; out[k] = ci(q.delta, q.lo, q.hi)
    # --- 학습기 (R1 − P1 블록, D0 − P1 블록)
    for t in ("core_v6", "rf_v6", "nn_v6", "core_cb", "core_rf", "core_nn"):
        f = RUNS / t / "deltas_vs_ref.csv"
        if f.exists():
            d = pd.read_csv(f); d = d[(d.scheme == "block") & (d.ref == "P1") & (d.featset == featset)]
            if "forcing" in d and (d.forcing == forcing).any():
                d = d[d.forcing == forcing]
            for r in REG:
                for lr in ("catboost", "rf", "mlp", "tabm", "ftt"):
                    for m, mk in (("R1", "Ro"), ("D0", "Dz")):
                        q = d[(d.region == r) & (d.learner == lr) & (d.method == m) & ((d.lam == lam) | (m != "R1"))]
                        k = f"dlrn{RK[r]}{lr}{mk}"
                        if len(q) and k not in out:
                            q = q.iloc[0]; out[k] = ci(q.delta, q.lo, q.hi)
    # --- 공통 셀 제품 비교
    cc = pd.read_csv(R / "common_cells_eval.csv")
    SER = {"P1 (Stefan, recal.)": "Po", "R1 (CatBoost)": "Ro", "D0 (CatBoost)": "Dz", "Wei 2026": "Wei", "Liu 2024": "Liu", "ESA CCI v4": "Ccifour", "ESA CCI v5": "Ccifive", "Aalto 2018": "Aalto", "Yi-Kimball 2019": "Yk"}
    for r in REG:
        for ser, sk in SER.items():
            for mask in (0, 25):
                q = cc[(cc.region == r) & (cc.mask_km == mask) & (cc.series == ser)]
                if len(q):
                    mk_ = "mask" if mask else "all"; out[f"cc{RK[r]}{sk}{mk_}"] = fmt(q.rmse.iloc[0]); out[f"ccn{RK[r]}{mk_}"] = f"{int(q.n.iloc[0]):,}"
    # --- 제품 짝지은 차이 (ours − product, 연도 정합, 25 km 마스크 / 전체)
    for mask, mk_ in ((25, "mask"), (0, "all")):
        f = R / ("products_eval_mask25.csv" if mask == 25 else "products_eval.csv")
        if not f.exists():
            continue
        pe = pd.read_csv(f)
        for r in REG:
            for pr, pk in (("wei", "wei"), ("liu", "liu"), ("cci4y", "ccifour"), ("cci5y", "ccifive"), ("aalto", "aalto"), ("yk", "yk")):
                for ser, sk in (("ours:R1:0.5", "Ro"), ("ours:P1", "Po"), ("ours:D0", "Dz")):
                    q = pe[(pe.region == r) & (pe.series == ser) & (pe.definition == f"vs {pr} (year-matched)")]
                    if len(q):
                        q = q.iloc[0]; out[f"dprod{RK[r]}{pk}{sk}{mk_}"] = ci(q.delta_vs_product, q.delta_lo, q.delta_hi); out[f"nprod{RK[r]}{pk}{mk_}"] = f"{int(q.n):,}"
                q = pe[(pe.region == r) & (pe.series == pr) & (pe.definition == "year-matched")]
                if len(q):
                    out[f"rmseprod{RK[r]}{pk}{mk_}"] = fmt(q.rmse.iloc[0]); out[f"biasprod{RK[r]}{pk}{mk_}"] = sfmt(q.bias.iloc[0])
    # --- 연도별 · 다년 변화
    yw = R / "yearwise"
    if (yw / "multiyear_changes.csv").exists():
        mc = pd.read_csv(yw / "multiyear_changes.csv")
        for r in REG:
            for ser, sk in (("ours:R1:0.5:blockCV", "Ro"), ("ours:R1:blockCV", "Ro"), ("ours:P1:blockCV", "Po"), ("ours:D0:blockCV", "Dz"), ("prod:wei", "Wei"), ("prod:liu", "Liu"), ("prod:cci5y", "Ccifive")):
                q = mc[(mc.region == r) & (mc.series == ser)]
                if len(q) and f"rdelta{RK[r]}{sk}" not in out:
                    out[f"rdelta{RK[r]}{sk}"] = fmt(q.r_delta.iloc[0], 2); out[f"npairs{RK[r]}"] = f"{int(q.n_pairs.iloc[0]):,}"
    for ywd, suf in ((yw, ""), (R / "yearwise_year_F8TS", "Yr")):   # 주 강제력 / 그 해 전체 강제력(비교용)
        if (ywd / "yearwise_metrics.csv").exists():
            ym = pd.read_csv(ywd / "yearwise_metrics.csv"); ym = ym[ym.series.str.startswith("ours:R1") & ym.series.str.endswith("blockCV") & (ym.n >= 50)]
            for r in REG:
                q = ym[ym.region == r]
                if len(q):
                    out[f"ybiasmin{RK[r]}{suf}"] = sfmt(q.bias.min()); out[f"ybiasmax{RK[r]}{suf}"] = sfmt(q.bias.max()); out[f"ybiasyrmin{RK[r]}{suf}"] = str(int(q.loc[q.bias.idxmin(), "year"]))
                    out[f"ybiassd{RK[r]}{suf}"] = fmt(q.bias.std())
    # --- 자료 수
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    for r in REG:
        q = lab[lab.region == r]
        out[f"ncy{RK[r]}"] = f"{len(q):,}"; out[f"ncell{RK[r]}"] = f"{q.loc_id.nunique():,}"; out[f"nblk{RK[r]}"] = str(q.block.nunique()); out[f"nyr{RK[r]}"] = str(q.year.nunique())
        out[f"nmulti{RK[r]}"] = str(int((q.groupby("loc_id").year.nunique() >= 4).sum()))
        top = q.year.value_counts(normalize=True)
        out[f"topyr{RK[r]}"] = str(int(top.index[0])); out[f"topyrshare{RK[r]}"] = f"{top.iloc[0]*100:.0f}"
    out["ncyAll"] = f"{len(lab):,}"; out["ncellAll"] = f"{lab.loc_id.nunique():,}"
    for r, key in (("Alaska", "Alaska"), ("Lena", "Lena")):
        mj = PROC / "maps" / key / "meta.json"
        if mj.exists():
            m = json.load(open(mj)); a_ = m.get("aoa", {}); v = None
            if isinstance(a_, dict):
                for k_ in ("inside_frac", "frac_inside", "inside", "frac", "share_inside"):
                    if k_ in a_:
                        v = a_[k_]; break
            if v is not None:
                out[f"aoa{RK[r]}"] = f"{float(v)*100:.0f}" if float(v) <= 1 else f"{float(v):.0f}"
    # --- 라벨 수 사다리(LL·TL)
    lad = RUNS / "ladder_v6" / "ladder_deltas.csv"
    NW = {"0": "zero", "25": "twfive", "50": "fifty", "100": "hundred", "250": "twofifty", "500": "fivehundred", "1000": "thousand", "2500": "twofivek", "all": "all"}
    if lad.exists():
        ld = pd.read_csv(lad); ld["n_label"] = ld.n_label.astype(str); lm = pd.read_csv(RUNS / "ladder_v6" / "ladder_metrics.csv"); lm["n_label"] = lm.n_label.astype(str)
        for r in REG:
            for nl in ("25", "50", "100", "250", "500", "1000", "2500", "all"):
                for m, mk in (("D0", "Dz"), ("D1", "Do"), ("F1", "Fo"), ("R1", "Ro")):
                    q = ld[(ld.exp == "LL") & (ld.region == r) & (ld.ref == "P1") & (ld.n_label == nl) & (ld.method == m)]
                    if len(q):
                        q = q.iloc[0]; out[f"ll{RK[r]}{mk}{NW[nl]}"] = ci(q.delta, q.lo, q.hi)
                q = ld[(ld.exp == "LL") & (ld.region == r) & (ld.ref == "P1") & (ld.n_label == nl) & (ld.method == "D1")]
                q0 = ld[(ld.exp == "LL") & (ld.region == r) & (ld.ref == "P1") & (ld.n_label == nl) & (ld.method == "D0")]
                if len(q) and len(q0):
                    out[f"llpseudo{RK[r]}{NW[nl]}"] = sfmt(q.delta.iloc[0] - q0.delta.iloc[0])   # 유사라벨 − 직접 학습(같은 n)
            for nl in ("0", "25", "50", "100", "250", "500"):
                for m, mk in (("P0", "Pz"), ("P1n", "Pn"), ("D0S", "Ds"), ("D0T", "Dt"), ("D1S", "Dps"), ("R1S", "Rs"), ("R1T", "Rt")):
                    q = lm[(lm.exp == "TL") & (lm.region == r) & (lm.n_label == nl) & (lm.method == m)]
                    if len(q):
                        out[f"tlrmse{RK[r]}{mk}{NW[nl]}"] = fmt(q.rmse.iloc[0])
                for m, mk in (("D0S", "Ds"), ("R1S", "Rs"), ("D1S", "Dps"), ("R1T", "Rt"), ("D0T", "Dt")):
                    q = ld[(ld.exp == "TL") & (ld.region == r) & (ld.ref == "P1n") & (ld.n_label == nl) & (ld.method == m)]
                    if len(q):
                        q = q.iloc[0]; out[f"tl{RK[r]}{mk}{NW[nl]}"] = ci(q.delta, q.lo, q.hi)
    out["runTag"] = run.replace("_", "\\_"); out["forcingMain"] = forcing.replace("_", "\\_")
    # --- 쓰기
    od = ROOT / "paper" / "manuscript" / "en_v6"; od.mkdir(parents=True, exist_ok=True)
    with open(od / "numbers.tex", "w") as fh:
        fh.write(f"% generated by scripts/2_evaluation/v6_numbers.py from runs/{run} (forcing {forcing}); do not edit\n")
        for k, v in sorted(out.items()):
            assert k.isalpha(), f"macro name must be letters only: {k}"
            fh.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    json.dump(out, open(ROOT / "docs" / "numbers_v6.json", "w"), ensure_ascii=False, indent=1)
    print(f"[numbers] {len(out)} macros from runs/{run} forcing {forcing} -> {od / 'numbers.tex'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="core_v6" if (RUNS / "core_v6" / "summary_metrics.csv").exists() else "core_cb")
    ap.add_argument("--forcing", default=None); a = ap.parse_args()
    main(a.run, a.forcing or ("year_obs" if a.run == "core_v6" else "year"))
