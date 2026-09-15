"""E1 결과 분석 — 사전 등록 주 추정치(H1·H2·H3·H6)와 통합 요인표를 짝지은 블록 부트스트랩으로 산출.

입력: data/processed/e1_factorial{tag}_shard*.csv, e1_factorial{tag}_shard*_preds.npz, e1_factorial{tag}_meta_shard*.json
산출: data/processed/e1_summary{tag}.csv         (조건×대상×구성×λ seed 평균 RMSE·bias·n)
      data/processed/e1_prereg_tests{tag}.csv    (H1·H2·H3·H6 짝지은 ΔRMSE·95% CI·지역 부호)
      data/processed/e1_table_conditions{tag}.csv (원고 표: 3조건 × 방법 계열, 같은 셀)
실행: python3 scripts/3_deep_learning/e1_analysis.py [--tag _e3] [--targets Lena,Canada]

짝지은 부트스트랩: 같은 평가 셀에서 두 방법의 예측을 seed 평균한 뒤 0.5° 블록을 재표집해 ΔRMSE 분포(400회).
지역 수준 요약: 대상 지역별 Δ의 부호와 비가중 평균. 지역 6 이상이면 부호검정 p(이항, 양측) 병기.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
sys_path_root = str(Path(__file__).resolve().parents[2] / "src")
import sys
sys.path.insert(0, sys_path_root)
from polar.fidelity import TRANSFER_MAIN, TRANSFER_DEEP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--targets", default="")
ap.add_argument("--nboot", type=int, default=400)
args = ap.parse_args()
TAG = args.tag
LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
RNG = np.random.RandomState(0)

# ---------------------------------------------------------------- 로드
csvs = sorted(glob.glob(str(PROC / f"e1_factorial{TAG}_shard*.csv")))
csvs = [c for c in csvs if "_preds" not in c]
res = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
if "error" in res:
    print(f"[load] {len(csvs)} shard · {len(res)}행 · 오류행 {res.error.notna().sum()}")
    res = res[res.error.isna()]
else:
    print(f"[load] {len(csvs)} shard · {len(res)}행")
G, ANCH, EVAL = {}, {}, {}
for f in sorted(glob.glob(str(PROC / f"e1_factorial{TAG}_shard*_preds.npz"))):
    z = np.load(f, allow_pickle=False)
    for k in z.files:
        kind, rest = k.split("::", 1)
        if kind == "g":
            G[rest] = z[k]
        elif kind == "anchor":
            ANCH[rest] = z[k]
        else:
            tk, fld = rest.rsplit("::", 1)
            EVAL.setdefault(tk, {})[fld] = z[k]
targets = [t for t in args.targets.split(",") if t] or sorted(set(res.target) - {"Alaska"})
print(f"[preds] g {len(G)} · anchor {len(ANCH)} · 평가집합 {len(EVAL)} · 대상 {targets}")

# ---------------------------------------------------------------- seed 평균 요약
key = ["cond", "target", "anchor", "pseudo", "r", "resid", "lam"]
summ = (res.groupby(key + ["fold"], as_index=False).agg(rmse=("rmse_cm", "mean"), bias=("bias_cm", "mean"), n=("n", "first"),
                                                        n_seed=("seed", "nunique"))
           .groupby(key, as_index=False).agg(rmse=("rmse", "mean"), bias=("bias", "mean"), n=("n", "sum"),
                                             n_fold=("fold", "nunique"), n_seed=("n_seed", "min")))
summ.to_csv(PROC / f"e1_summary{TAG}.csv", index=False)


# ---------------------------------------------------------------- 예측 재구성
def seeds_of(cond, tg, fold):
    pre = f"{cond}|{tg}|{fold}|"
    return sorted({int(k[len(pre):].split("|")[0]) for k in G if k.startswith(pre)})


def predict(cond, tg, fold, anchor, pseudo, r, resid, lam):
    """seed 평균 예측 벡터(평가 셀 순서). 없으면 None."""
    tk = f"{cond}|{tg}|{fold}"
    ps = []
    for s in seeds_of(cond, tg, fold):
        k = f"{tk}|{s}|{anchor}|{pseudo}|{float(r)}|{resid}"
        if k not in G:
            continue
        g = G[k].astype(float)
        if anchor == "none":
            ps.append(g)
        else:
            ps.append(ANCH[f"{tk}|{anchor}"].astype(float) + lam * g)
    return None if not ps else np.mean(ps, axis=0)


def anchor_only(cond, tg, fold, anchor):
    return ANCH.get(f"{cond}|{tg}|{fold}|{anchor}")


def paired(cond, tg, fold, pA, pB):
    """ΔRMSE = RMSE(A) − RMSE(B) 점추정 + 블록 짝지은 부트스트랩 CI."""
    ev = EVAL[f"{cond}|{tg}|{fold}"]
    y, blocks = ev["y"].astype(float), ev["block"]
    m = np.isfinite(pA) & np.isfinite(pB) & np.isfinite(y)
    y, pA, pB, blocks = y[m], pA[m], pB[m], blocks[m]
    d0 = np.sqrt(np.mean((y - pA) ** 2)) - np.sqrt(np.mean((y - pB) ** 2))
    ub = np.unique(blocks)
    out = []
    for _ in range(args.nboot):
        pick = RNG.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        out.append(np.sqrt(np.mean((y[idx] - pA[idx]) ** 2)) - np.sqrt(np.mean((y[idx] - pB[idx]) ** 2)))
    return d0, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(m.sum()), len(ub)


def folds_of(cond, tg):
    """평가집합 키에서 정수 fold만(지역 합본 'pooled' 키 제외)."""
    return sorted({int(k.split("|")[2]) for k in EVAL if k.startswith(f"{cond}|{tg}|") and k.split("|")[2].isdigit()})


# ---------------------------------------------------------------- 사전 등록 검정
TESTS = []


def add_test(hid, label, cond, A, B):
    """A·B = dict(anchor,pseudo,r,resid,lam) 또는 dict(anchor_only=...)"""
    def pred_of(spec, tg, fold):
        if "anchor_only" in spec:
            return anchor_only(cond, tg, fold, spec["anchor_only"])
        return predict(cond, tg, fold, spec["anchor"], spec["pseudo"], spec["r"], spec["resid"], spec["lam"])
    deltas, tg_list = [], []
    for tg in targets:
        folds = folds_of(cond, tg)
        # fold를 지역 단위로 합쳐(OOF 연결) 한 번의 짝지은 블록 부트스트랩. 알래스카 6-fold는 74블록 전체가 재표집 단위.
        yy, bb, pa, pb = [], [], [], []
        for fold in folds:
            pA, pB = pred_of(A, tg, fold), pred_of(B, tg, fold)
            if pA is None or pB is None:
                continue
            ev = EVAL[f"{cond}|{tg}|{fold}"]
            yy.append(ev["y"].astype(float)); bb.append(ev["block"]); pa.append(pA); pb.append(pB)
        if not yy:
            continue
        EVAL[f"{cond}|{tg}|pooled"] = dict(y=np.concatenate(yy), block=np.concatenate(bb))
        d0, lo, hi, n, nb = paired(cond, tg, "pooled", np.concatenate(pa), np.concatenate(pb))
        TESTS.append(dict(H=hid, label=label, cond=cond, target=tg, fold=len(yy), delta_rmse=d0, ci_lo=lo, ci_hi=hi,
                          n_cells=n, n_blocks=nb, A=json.dumps(A), B=json.dumps(B)))
        deltas.append(d0); tg_list.append(tg)
    if deltas:
        for grp, members in [("REGION_SUMMARY_ALL", None), ("REGION_SUMMARY_MAIN", TRANSFER_MAIN), ("REGION_SUMMARY_DEEP", TRANSFER_DEEP)]:
            sel = [(t, d) for t, d in zip(tg_list, deltas) if members is None or t in members]
            if not sel or (members is not None and len(sel) == len(deltas) and grp != "REGION_SUMMARY_ALL" and members is None):
                continue
            ds = [d for _, d in sel]
            neg = sum(1 for x in ds if x < 0)
            p = binomtest(neg, len(ds), 0.5).pvalue if len(ds) >= 5 else np.nan
            TESTS.append(dict(H=hid, label=label, cond=cond, target=grp, fold=-1,
                              delta_rmse=float(np.mean(ds)), ci_lo=np.nan, ci_hi=np.nan, n_cells=len(ds),
                              n_blocks=neg, A=f"neg_regions={neg}/{len(ds)} [{','.join(t for t, _ in sel)}]", B=f"sign_test_p={p}"))


ST, SC, SS, SSC = "stefan", "stefan_cci", "stefan_soil", "stefan_soil_cci"
# H1: Stefan+CCI 앵커 vs Stefan 앵커 (λ=0), 정보 없음
add_test("H1", "Stefan+CCI 앵커 − Stefan 앵커 (λ=0)", "noinfo", dict(anchor_only=SC), dict(anchor_only=ST))
add_test("H1b", "Stefan+CCI 앵커 − Stefan 앵커 (λ=0), 공변량만", "covonly", dict(anchor_only=SC), dict(anchor_only=ST))
# H2: 앵커+ridge λ=0.25 vs 앵커 단독
for cond in ["noinfo", "covonly"]:
    add_test("H2", "Stefan+CCI 앵커 + ridge(λ=.25) − 앵커 단독", cond,
             dict(anchor=SC, pseudo="none", r=0.0, resid="ridge", lam=0.25), dict(anchor_only=SC))
    add_test("H2s", "Stefan 앵커 + ridge(λ=.25) − Stefan 앵커", cond,
             dict(anchor=ST, pseudo="none", r=0.0, resid="ridge", lam=0.25), dict(anchor_only=ST))
# H3: 증강 대조군 사다리 (공변량만, r=10, CatBoost 직접 회귀)
for ctrl in ["const", "shuffle", "tddlin"]:
    add_test("H3", f"Stefan 유사라벨 − {ctrl} 유사라벨 (direct catboost_lo, r=10)", "covonly",
             dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0),
             dict(anchor="none", pseudo=ctrl, r=10.0, resid="catboost_lo", lam=1.0))
add_test("H3b", "Stefan 유사라벨 − 증강 없음 (direct catboost_lo)", "covonly",
         dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0),
         dict(anchor="none", pseudo="none", r=0.0, resid="catboost_lo", lam=1.0))
# 탐색적(E2 이후 추가): 토양 도일 Stefan 앵커
add_test("X-F7", "stefan_soil 앵커 − stefan 앵커 (λ=0) [탐색적]", "noinfo", dict(anchor_only=SS), dict(anchor_only=ST))
add_test("X-F7c", "stefan_soil+CCI − Stefan+CCI (λ=0) [탐색적]", "noinfo", dict(anchor_only=SSC), dict(anchor_only=SC))
# 탐색적(E2 v2 이후 추가): Kudryavtsev 보정·다중 물리 결합 앵커
KU, SKU, SKC = "ku_cal", "stefan_ku", "stefan_ku_cci"
add_test("X-Ku", "ku_cal 앵커 − stefan 앵커 (λ=0) [탐색적]", "noinfo", dict(anchor_only=KU), dict(anchor_only=ST))
add_test("X-SKu", "stefan_ku − stefan (λ=0) [탐색적]", "noinfo", dict(anchor_only=SKU), dict(anchor_only=ST))
add_test("X-SKuC", "stefan_ku_cci − stefan_cci (λ=0) [탐색적]", "noinfo", dict(anchor_only=SKC), dict(anchor_only=SC))
add_test("X-SKuC2", "stefan_ku_cci + ridge(λ=.25) − stefan_cci + ridge(λ=.25) [탐색적]", "noinfo",
         dict(anchor=SKC, pseudo="none", r=0.0, resid="ridge", lam=0.25), dict(anchor=SC, pseudo="none", r=0.0, resid="ridge", lam=0.25))
# H6: 알래스카 지역 내
if "Alaska" in set(res.target):
    _t = targets
    targets = ["Alaska"]
    add_test("H6", "Stefan+CCI 앵커 − Stefan 앵커 (λ=0), 알래스카 지역 내", "labels", dict(anchor_only=SC), dict(anchor_only=ST))
    add_test("H6r", "Stefan+CCI + ridge(λ=.25) − Stefan + ridge(λ=.25), 알래스카", "labels",
             dict(anchor=SC, pseudo="none", r=0.0, resid="ridge", lam=0.25),
             dict(anchor=ST, pseudo="none", r=0.0, resid="ridge", lam=0.25))
    add_test("X-F7ak", "stefan_soil − stefan (λ=0), 알래스카 [탐색적]", "labels", dict(anchor_only=SS), dict(anchor_only=ST))
    targets = _t

tests = pd.DataFrame(TESTS)
tests.to_csv(PROC / f"e1_prereg_tests{TAG}.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== 사전 등록 검정 (ΔRMSE = A − B, 음수 = A 우세) ===")
print(tests[["H", "label", "cond", "target", "fold", "delta_rmse", "ci_lo", "ci_hi", "n_cells", "n_blocks"]]
      .round(2).to_string(index=False))

# ---------------------------------------------------------------- 원고 표: 3조건 × 방법 계열(같은 셀, seed 평균)
FAMILIES = [
    ("기계학습 단독(CatBoost)", dict(anchor="none", pseudo="none", r=0.0, resid="catboost_lo", lam=1.0)),
    ("기계학습 단독(MLP)", dict(anchor="none", pseudo="none", r=0.0, resid="mlp", lam=1.0)),
    ("Stefan 유사라벨 증강(CatBoost, r=10)", dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0)),
    ("물리식 단독(Stefan)", dict(anchor_only=ST)),
    ("물리식 단독(토양 도일 Stefan)", dict(anchor_only=SS)),
    ("위성 제품 단독(보정)", dict(anchor_only="cci_cal")),
    ("Stefan+위성 제품", dict(anchor_only=SC)),
    ("Stefan+위성 제품 + ridge(λ=.25)", dict(anchor=SC, pseudo="none", r=0.0, resid="ridge", lam=0.25)),
    ("토양 도일 Stefan+위성 제품", dict(anchor_only=SSC)),
    ("토양 도일 Stefan+위성 제품 + ridge(λ=.25)", dict(anchor=SSC, pseudo="none", r=0.0, resid="ridge", lam=0.25)),
    ("Kudryavtsev 보정 단독", dict(anchor_only="ku_cal")),
    ("Stefan+Kudryavtsev", dict(anchor_only="stefan_ku")),
    ("Stefan+Kudryavtsev+위성 제품", dict(anchor_only="stefan_ku_cci")),
    ("Stefan+Kudryavtsev+위성 제품 + ridge(λ=.25)", dict(anchor="stefan_ku_cci", pseudo="none", r=0.0, resid="ridge", lam=0.25)),
]
tab = []
for cond in ["labels", "covonly", "noinfo"]:
    for name, spec in FAMILIES:
        if cond != "covonly" and spec.get("pseudo", "none") != "none":
            continue
        vals = {}
        for tg in targets:
            folds = folds_of(cond, tg)
            rm = []
            for fold in folds:
                p = (anchor_only(cond, tg, fold, spec["anchor_only"]) if "anchor_only" in spec
                     else predict(cond, tg, fold, spec["anchor"], spec["pseudo"], spec["r"], spec["resid"], spec["lam"]))
                if p is None:
                    continue
                y = EVAL[f"{cond}|{tg}|{fold}"]["y"].astype(float)
                m = np.isfinite(p) & np.isfinite(y)
                rm.append(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
            if rm:
                vals[tg] = float(np.mean(rm))
        if vals:
            main = [v for k, v in vals.items() if k in TRANSFER_MAIN]
            deep = [v for k, v in vals.items() if k in TRANSFER_DEEP]
            tab.append(dict(cond=cond, method=name, **{f"rmse_{k}": v for k, v in vals.items()},
                            rmse_mean=float(np.mean(list(vals.values()))), n_regions=len(vals),
                            rmse_mean_main=float(np.mean(main)) if main else np.nan, n_main=len(main),
                            rmse_mean_deep=float(np.mean(deep)) if deep else np.nan, n_deep=len(deep)))
tab = pd.DataFrame(tab)
tab.to_csv(PROC / f"e1_table_conditions{TAG}.csv", index=False)
print("\n=== 원고 표: 조건 × 방법 (지역 비가중 평균, 같은 평가 셀) ===")
print(tab.pivot_table(index="method", columns="cond", values="rmse_mean", sort=False).round(2).to_string())
if tab.n_main.max() > 0 and tab.n_regions.max() > 2:
    print("\n=== 주 전이 집합 평균 / 심부 집합 평균 (noinfo) ===")
    print(tab[tab.cond == "noinfo"][["method", "rmse_mean_main", "n_main", "rmse_mean_deep", "n_deep"]].round(2).to_string(index=False))
print(f"\nsaved: e1_summary{TAG}.csv · e1_prereg_tests{TAG}.csv · e1_table_conditions{TAG}.csv")
