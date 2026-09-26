"""H18–H24 확인적 검정 통합 채점 (개정 09-21 규약: seed 짝지음·지역 층화 블록 부트스트랩·세 채점·가족별 Holm).

입력
  --tag h1819 : data/processed/m1/h1819_shard*.csv + _preds.npz (m1_master_factorial.py --ext --axis ext 산출)
  H22        : data/processed/h2/h22_rows.csv + h22_preds.npz
  H21        : 자체 tests CSV(data/processed/h2/h21_tests.csv, 이미 부트스트랩)의 행만 결합.
               'test|AB4' 행은 같은 대조를 AB4 집합으로 다시 적은 중복이므로 confirmatory=False 로 두어 Holm 가족에서 뺀다(감사 반영).
  H23·H24    : 결합하지 않는다. 두 가설은 n 곡선형 검정(h23_tests.csv 의 delta_vs_shrink, h24_tests.csv 의 delta_vs_random)이라
               이 파일의 지역 평균 Δ·Holm 형식과 맞지 않으므로 각자의 CSV 로만 보고한다.
산출 data/processed/h2/h_tests_all.csv (가설·대조·조건·지역·Δ·CI·p·Holm), h_tests_main.csv(확인적 가설만)
실행: python3 scripts/2_evaluation/h_analysis.py --tag h1819
      python3 scripts/2_evaluation/h_analysis.py --reholm   (부트스트랩 재계산 없이 기존 h_tests_all.csv 의 확인적 표지·Holm 만 다시 적용)
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.m1_stats import boot_delta, summarize_delta, seed_of, holm                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="h1819")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--reholm", action="store_true", help="기존 h_tests_all.csv 로 확인적 표지·Holm 만 재적용")
args = ap.parse_args()
PROC = ROOT / "data" / "processed"; OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
DEF = dict(dset="alaska", xset="x25", anchor="stefan", pseudo="none", r=0.0, resid="catboost_lo", iw=0, lam=0.0)
ALL_ROWS = []
H21_CONFIRM = ("H21_nested_vs_EAK", "H20_cci_blockE_vs_stefan")


def h21_confirm(test):
    """H21 확인적 대조 판정. 'test|AB4' 중복 행은 확인적이 아니다."""
    return ("|AB4" not in str(test)) and str(test).split("|")[0] in H21_CONFIRM


def finalize(T):
    """지역 평균 행 표지, 가족별 Holm(확인적·지역 평균 행만), 저장·출력."""
    T["is_mean"] = T.target.astype(str).str.startswith("MEAN")
    T["p_holm"] = np.nan
    for fam, sub in T[T.confirmatory & T.is_mean].groupby("family"):
        T.loc[sub.index, "p_holm"] = holm(sub.p_boot.values)
    T.to_csv(OUT / "h_tests_all.csv", index=False)
    M = T[T.is_mean & T.confirmatory][["family", "source", "test", "cond", "target", "rmse_A", "rmse_B", "delta", "ci_lo", "ci_hi", "p_boot", "p_holm", "delta_blockeq", "block_majority", "ci_flag"]]
    M.to_csv(OUT / "h_tests_main.csv", index=False)
    pd.set_option("display.width", 250)
    print(M.round(3).to_string())
    X = T[T.is_mean & ~T.confirmatory][["source", "test", "cond", "rmse_A", "rmse_B", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_flag"]]
    print(X.round(2).to_string())
    print(f"rows {len(T)}")


if args.reholm:
    T = pd.read_csv(OUT / "h_tests_all.csv")
    T["confirmatory"] = T.confirmatory.astype(bool)
    h21 = T.source == "h21"
    T.loc[h21, "confirmatory"] = T.loc[h21, "test"].map(h21_confirm).astype(bool)
    finalize(T)
    sys.exit(0)


# ---------------------------------------------------------------- m1 형식 적재
def load_m1(tag):
    P, EV, ANC = {}, {}, {}
    for f in sorted(glob.glob(str(PROC / "m1" / f"{tag}_shard*_preds.npz"))):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if k.startswith("g::") or k.startswith("anchor::"):
                P[k] = z[k]
            elif k.startswith("eval::"):
                _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
    return P, EV


def spec(**kw):
    s = dict(DEF); s.update(kw)
    if s["pseudo"] != "none" and s["r"] == 0.0:
        s["r"] = 10.0
    return s


def pred_seeds(P, EV, cond, tg, s):
    """(S, n) seed 짝지은 예측을 split 풀링. anchor=none 이면 직접."""
    tks = sorted([k for k in EV if k.startswith(f"{cond}|{tg}|")], key=lambda x: int(x.split("|")[2]))
    outs, ys, bs = [], [], []
    for tk in tks:
        seeds = sorted({int(k.split("|")[3]) for k in P if k.startswith(f"g::{tk}|") and k.endswith(f"|{s['dset']}|{s['xset']}|{s['anchor']}|{s['pseudo']}|{s['r']}|{s['resid']}|{s['iw']}")})
        if s["lam"] == 0.0 and s["anchor"] != "none":
            ak = f"anchor::{tk}|{s['dset']}|{s['anchor']}"
            if ak not in P:
                return None, None, None
            outs.append(np.stack([P[ak]] * max(1, len(seeds) or 1)))
        else:
            if not seeds:
                return None, None, None
            G = np.stack([P[f"g::{tk}|{sd}|{s['dset']}|{s['xset']}|{s['anchor']}|{s['pseudo']}|{s['r']}|{s['resid']}|{s['iw']}"] for sd in seeds])
            outs.append(G if s["anchor"] == "none" else P[f"anchor::{tk}|{s['dset']}|{s['anchor']}"][None, :] + s["lam"] * G)
        ys.append(EV[tk]["y"]); bs.append(EV[tk]["block"])
    if not outs:
        return None, None, None
    S = min(o.shape[0] for o in outs)
    return np.concatenate([o[:S] for o in outs], 1), np.concatenate(ys), np.concatenate(bs)


def test_m1(P, EV, name, family, cond, targets, A, B, confirm=False):
    per = {}
    for tg in targets:
        PA, y, blk = pred_seeds(P, EV, cond, tg, A); PB, y2, _ = pred_seeds(P, EV, cond, tg, B)
        if PA is None or PB is None:
            continue
        S = min(PA.shape[0], PB.shape[0])
        per[tg] = boot_delta(y, blk, PA[:S], PB[:S], args.nboot, seed_of(name, cond, tg))
    for r in summarize_delta(name, per, targets, 0):
        ALL_ROWS.append(dict(family=family, cond=cond, confirmatory=confirm, source=args.tag, **r))


P, EV = load_m1(args.tag)
if P:
    have_lst = any("stefan_lst" in k for k in P)
    xs_all = sorted({k.split("|")[5] for k in P if k.startswith("g::")})
    print(f"[{args.tag}] 예측 {len(P)} · 평가 {len(EV)} · 입력 집합 {xs_all} · LST {have_lst}", flush=True)
    for cond, tg_set in (("noinfo", MAIN6), ("covonly", AB4)):
        st0 = spec()
        if have_lst:
            test_m1(P, EV, "H18a_stefan_lst_vs_stefan", "info", cond, tg_set, spec(anchor="stefan_lst"), st0, confirm=True)
            test_m1(P, EV, "H18x_stefan_lst_k2_vs_stefan_k2", "info", cond, tg_set, spec(anchor="stefan_lst_k2"), spec(anchor="stefan_k2"))
            test_m1(P, EV, "H18x_stefan_lst_cci_vs_stefan_cci", "info", cond, tg_set, spec(anchor="stefan_lst_cci"), spec(anchor="stefan_cci"))
            test_m1(P, EV, "H18x_stefan_soil_vs_stefan", "info", cond, tg_set, spec(anchor="stefan_soil"), st0)
            test_m1(P, EV, "H18K1_stefan_lst_resid25_vs_stefan", "info", cond, tg_set, spec(anchor="stefan_lst", lam=0.25), st0)
        for x in xs_all:
            if x == "x25":
                continue
            # 직접 회귀(정보 없음: pseudo none / 공변량만: pseudo stefan r=10) — 입력 집합 요인
            if cond == "noinfo":
                test_m1(P, EV, f"H19d_{x}_vs_x25_direct", "info", cond, tg_set, spec(xset=x, anchor="none", lam=1.0), spec(anchor="none", lam=1.0),
                        confirm=(x == "x25_A"))
            else:
                test_m1(P, EV, f"H19p_{x}_vs_x25_pseudo", "info", cond, tg_set, spec(xset=x, anchor="none", pseudo="stefan", lam=1.0),
                        spec(anchor="none", pseudo="stefan", lam=1.0), confirm=(x in ("x25_A", "x25_lst")))
                test_m1(P, EV, f"H19s_{x}_pseudo_vs_stefan", "info", cond, tg_set, spec(xset=x, anchor="none", pseudo="stefan", lam=1.0), st0)
            test_m1(P, EV, f"H19r_{x}_resid25_vs_x25_resid25", "info", cond, tg_set, spec(xset=x, lam=0.25), spec(lam=0.25))
            test_m1(P, EV, f"H19r_{x}_resid25_vs_stefan", "info", cond, tg_set, spec(xset=x, lam=0.25), st0)
        if have_lst and cond == "covonly":
            test_m1(P, EV, "H18Q1_pseudo_stefan_lst_vs_pseudo_stefan", "info", cond, tg_set, spec(anchor="none", pseudo="stefan_lst", lam=1.0),
                    spec(anchor="none", pseudo="stefan", lam=1.0))
            test_m1(P, EV, "H18C_x25lst_anc_lst_ps_lst_resid25_vs_stefan", "info", cond, tg_set,
                    spec(xset="x25_lst", anchor="stefan_lst", pseudo="stefan_lst", lam=0.25), st0)
            test_m1(P, EV, "H18C_x25Alst_anc_lst_ps_lst_resid25_vs_stefan", "info", cond, tg_set,
                    spec(xset="x25_A_lst", anchor="stefan_lst", pseudo="stefan_lst", lam=0.25), st0)
            test_m1(P, EV, "H18C_anc_lst_ps_lst_resid25_vs_stefan", "info", cond, tg_set, spec(anchor="stefan_lst", pseudo="stefan_lst", lam=0.25), st0)

# ---------------------------------------------------------------- H22
f22 = OUT / "h22_preds.npz"
if f22.exists():
    z = np.load(f22, allow_pickle=True); rows22 = pd.read_csv(OUT / "h22_rows.csv")
    EV22 = {}
    for k in z.files:
        if k.startswith("eval::"):
            _, tk, fld = k.split("::"); EV22.setdefault(tk, {})[fld] = z[k]

    def p22(cond, tg, ytype, method, lam, use_nested=True):
        outs, ys, bs = [], [], []
        for tk in sorted([k for k in EV22 if k.startswith(f"{cond}|{tg}|")], key=lambda x: int(x.split("|")[2])):
            sub = rows22[(rows22.cond == cond) & (rows22.target == tg) & (rows22.split == int(tk.split("|")[2])) & (rows22.ytype == ytype) & (rows22.method == method)]
            if not len(sub):
                return None, None, None
            hp = sub.hp_nested.iloc[0] if use_nested else sub.hp.iloc[0]
            seeds = sorted(sub[sub.hp == hp].seed.unique())
            G = np.stack([z[f"g::{tk}|{ytype}|{method}|{hp:g}|{sd}"] for sd in seeds])
            anc = z[f"anchor::{tk}"]
            outs.append(anc[None, :] + lam * G if ytype == "resid" else G); ys.append(EV22[tk]["y"]); bs.append(EV22[tk]["block"])
        if not outs:
            return None, None, None
        S = min(o.shape[0] for o in outs)
        return np.concatenate([o[:S] for o in outs], 1), np.concatenate(ys), np.concatenate(bs)

    def anchor22(cond, tg):
        tks = sorted([k for k in EV22 if k.startswith(f"{cond}|{tg}|")], key=lambda x: int(x.split("|")[2]))
        return np.concatenate([z[f"anchor::{tk}"] for tk in tks])[None, :] if tks else None

    methods = sorted(rows22.method.unique())
    for cond, tg_set in (("noinfo", MAIN6), ("covonly", AB4)):
        for ytype in ("resid", "direct"):
            lams = [0.25, 0.5, 1.0] if ytype == "resid" else [1.0]
            for m in methods:
                if m == "erm":
                    continue
                for lam in lams:
                    per_a, per_e = {}, {}
                    for tg in tg_set:
                        PA, y, blk = p22(cond, tg, ytype, m, lam)
                        if PA is None or anchor22(cond, tg) is None:
                            continue
                        per_a[tg] = boot_delta(y, blk, PA, np.repeat(anchor22(cond, tg), PA.shape[0], 0), args.nboot, seed_of("H22", cond, tg, m, ytype, lam))
                        PE, _, _ = p22(cond, tg, ytype, "erm", lam)
                        if PE is not None:
                            S = min(PA.shape[0], PE.shape[0]); per_e[tg] = boot_delta(y, blk, PA[:S], PE[:S], args.nboot, seed_of("H22e", cond, tg, m, ytype, lam))
                    for r in summarize_delta(f"H22_{m}_{ytype}_lam{lam:g}_vs_stefan", per_a, tg_set, 0):
                        ALL_ROWS.append(dict(family="objective", cond=cond, confirmatory=(ytype == "resid" and lam == 0.25 and m in ("irm", "vrex", "dann")), source="h22", **r))
                    for r in summarize_delta(f"H22_{m}_{ytype}_lam{lam:g}_vs_erm", per_e, tg_set, 0):
                        ALL_ROWS.append(dict(family="objective", cond=cond, confirmatory=False, source="h22", **r))
            for lam in lams:
                per = {}
                for tg in tg_set:
                    PE, y, blk = p22(cond, tg, ytype, "erm", lam)
                    if PE is not None and anchor22(cond, tg) is not None:
                        per[tg] = boot_delta(y, blk, PE, np.repeat(anchor22(cond, tg), PE.shape[0], 0), args.nboot, seed_of("H22", cond, tg, "erm", ytype, lam))
                for r in summarize_delta(f"H22_erm_{ytype}_lam{lam:g}_vs_stefan", per, tg_set, 0):
                    ALL_ROWS.append(dict(family="objective", cond=cond, confirmatory=False, source="h22", **r))

# ---------------------------------------------------------------- H21(자체 검정 결합)
f21 = OUT / "h21_tests.csv"
if f21.exists():
    t21 = pd.read_csv(f21)
    for _, r in t21.iterrows():
        ALL_ROWS.append(dict(family="info", cond="noinfo" if "|AB4" not in r.test else "noinfo|AB4", confirmatory=h21_confirm(r.test),
                             source="h21", **{k: r[k] for k in t21.columns}))

T = pd.DataFrame(ALL_ROWS)
if len(T):
    finalize(T)
else:
    print("rows 0")
