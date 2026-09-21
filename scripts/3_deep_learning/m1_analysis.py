"""M1 결과 분석 — 단일 결과 DB에서 요약표·짝지은 검정·중첩 선택·재현 게이트를 산출.

입력: data/processed/m1/<tag>_shard*.csv, _preds.npz, _meta.json  (여러 tag를 쉼표로 병합 가능)
산출: data/processed/m1/<out>_summary.csv        (조건·대상·구성·λ, seed 평균·seed 앙상블·분할 평균·완전 사례 재채점·구성 표기)
      data/processed/m1/<out>_tests.csv          (사전 등록·탐색적 짝지은 검정: ΔRMSE 3종 채점, 블록 부트스트랩 CI, 블록 검정, Holm)
      data/processed/m1/<out>_splits.csv         (검정별 분할(split/fold)별 Δ와 분할 간 SD)
      data/processed/m1/<out>_combo_decomp.csv   (C2-A: Stefan+CCI 결합 앵커의 오차 상관·편향²/분산 분해·이론 최적 가중)
      data/processed/m1/<out>_consistency.csv    (물리 일관성: 범위 위배율·√TDD 단조성 위배율·암묵 E IQR)
      data/processed/m1/<out>_uq.csv             (H15: 생성 모델 90% 구간의 커버리지·폭·interval score, 참조 행 = S11 CQR·상수 폭)
      data/processed/m1/<out>_nested.csv         (중첩 선택: 5지역 선택 → 1지역 채점)
      data/processed/m1/<out>_gate.csv           (--gate: 옛 표 수치 대 새 하네스 수치, 허용 0.1 cm)

통계 규약(마스터 계획 §1 + 09-18 감사 + 개정 2026-09-21 §A·C·E)
  - 짝지음(§A): 기본 `per_seed` = seed별 예측으로 Δ_s = RMSE(A_s) − RMSE(B_s)를 계산해 seed 평균. 보조 `ensemble` = seed 평균
    예측을 채점한 Δ. 두 값을 `delta_per_seed`·`delta_ensemble` 열에, CI는 짝지음 모드(기본 per_seed) 기준, 앙상블 CI는 `ci_lo_ens/ci_hi_ens`.
  - 채점 3종(§A): 셀 가중(주, `delta_cell`), 블록 등가중(블록별 RMSE의 평균, `delta_blockeq`), 블록당 셀 수 상한 100 가중
    (√(Σw e²/Σw), w=min(1,100/n_block), `delta_cap100`). 블록 다수결 `block_majority` = Δ_block<0 인 블록 비율.
  - 0.5° 블록 재표집 부트스트랩(기본 1,000회): 블록 재표집 뒤 seed 평균 Δ를 통계량으로. 셀 가중·블록 등가중 CI 둘 다.
    블록 8개 미만 지역은 CI 미산출(NaN)·플래그. 부트스트랩 p(`p_boot`) = 분포에서 0 반대편 비율의 2배(최소 1/nboot).
  - 지역 층화 부트스트랩: 주 집합 각 지역의 블록을 동시에 재표집해 '지역 비가중 평균 Δ'의 CI(주 추정치). 분포가 없는
    지역(블록<8: 러시아 C·그린란드)은 CI에서 제외하고 `ci_flag`에 기록(`delta_rmse_ci_regions` = CI 산출 지역만의 평균 Δ).
    `REGION_SUMMARY_AB4` = 셀 수가 충분한 주 집합 4지역(레나·캐나다·러시아 W·러시아 E, §B) 요약 행.
    부호검정(n=6)은 기술 통계로 강등(`sign_test_p_descriptive`). 블록 단위 보조: Wilcoxon 부호순위(`wilcoxon_p`), 지역 층화
    순열검정(지역 내 부호 뒤집기 10,000회, `strat_perm_p`). 단위 = 블록별 ΔRMSE(seed 평균), 양측.
  - 앙상블(트리+신경망): 같은 (cond,target,split,seed,anchor,pseudo,r,iw)에서 catboost_lo·mlp의 g 평균(`resid=ens_cb_mlp`),
    catboost_lo·mlp·cfm 평균(`ens_cb_mlp_cfm`)을 분석 단계에서 합성해 summary·검정에 일반 구성처럼 포함.
  - H15 UQ: 생성 모델의 `q::` 5%·95% 분위(잔차 모드는 앵커 + λ·분위)로 90% 구간의 커버리지(블록 부트스트랩 CI)·평균 폭·
    interval score(α=0.1: 폭 + (2/α)(하한−y)⁺ + (2/α)(y−상한)⁺)를 구성·조건·지역·seed별로 계산. 참조 행 = S11 CQR·상수 폭.
  - 다중 비교(§C): 가족 = transfer(H1·H7–H11·H14) / within_region(H6·H13) / uq(H15) / primary(H12, 무보정) / exploratory
    (X-* 및 나머지, 보정 제외). 가족 내 Holm 보정 `p_holm`(주 행 = REGION_SUMMARY_MAIN, 없으면 단일 대상 행; DEEP·ALL은 별도 군).
  - 구성 표기(§E) `family_tag`: analytic(λ=0 또는 앵커 단독) / residual(anchor≠none, pseudo=none) / augment_direct(anchor=none,
    pseudo≠none) / formula_mix(둘 다 ≠none) / zero_resid_shrink(anchor==pseudo) / direct(둘 다 none).
  - 완전 사례 재채점(§E): SoilGrids 9종·CCI 모두 유효한 평가 셀만으로 seed별 RMSE 평균을 다시 계산(`rmse_cc`, `n_cc`).
  - MDE(80% 검정력 근사) = 2.8 × 부트스트랩 SD(Δ). 분할 반복(split>0)은 분할 평균과 분할 간 SD를 함께 기록(+ `_splits.csv`).
실행: python3 scripts/3_deep_learning/m1_analysis.py --tags m1_model,m1_anchor --out m1_main [--gate] [--pairing per_seed|ensemble]
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
import warnings
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

warnings.filterwarnings("ignore", message="Mean of empty slice")   # SOC 전부 결측 셀의 nanmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TRANSFER_MAIN, TRANSFER_DEEP, SOIL   # noqa: E402

M1 = ROOT / "data" / "processed" / "m1"
ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="m1")
ap.add_argument("--out", default="")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--nperm", type=int, default=10000, help="층화 순열검정 반복 수")
ap.add_argument("--npairs", type=int, default=2000, help="√TDD 단조성 검사 셀쌍 수(지역·조건별)")
ap.add_argument("--pairing", default="per_seed", choices=["per_seed", "ensemble"], help="주 짝지음 규약(개정 09-21 §A: per_seed)")
ap.add_argument("--base", default="", help="평가 셀 공변량 파일(기본: meta의 base 또는 fidelity_base_v3.csv)")
ap.add_argument("--gate", action="store_true")
ap.add_argument("--tests", default="", help="검정 정의 JSON(없으면 내장 H12–H15). 항목: H,label,cond,A,B[,targets,splits,family]")
args = ap.parse_args()
TAGS = [t for t in args.tags.split(",") if t]
OUT = args.out or TAGS[0]
CFG = ["dset", "xset", "anchor", "pseudo", "r", "resid", "iw"]
MIN_BLOCKS_CI = 8        # 지역 CI 산출 최소 블록 수
MIN_BLOCKS_TEST = 5      # 블록 단위 검정(Wilcoxon·순열) 최소 블록 수
CAP_CELLS = 100          # 블록당 셀 수 상한 가중
RANGE_LO, RANGE_HI = 0.0, 300.0   # 물리 범위(cm)
ALPHA = 0.1                        # H15 구간 수준(90%)
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]   # 개정 09-21 §B: 공변량만·라벨 있음이 성립하는 주 집합 4지역
ENSEMBLES = {"ens_cb_mlp": ["catboost_lo", "mlp"], "ens_cb_mlp_cfm": ["catboost_lo", "mlp", "cfm"]}
FAMILY = {"H12": "primary", "H6": "within_region", "H13": "within_region", "H15": "uq",
          **{h: "transfer" for h in ["H1", "H7", "H8", "H9", "H10", "H11", "H14"]}}

# ---------------------------------------------------------------- 로드
res, G, Q, ANCH, EVAL, META = [], {}, {}, {}, {}, []
SEEDS = defaultdict(set)
for tag in TAGS:
    for c in sorted(glob.glob(str(M1 / f"{tag}_shard*.csv"))):
        res.append(pd.read_csv(c))
    for f in sorted(glob.glob(str(M1 / f"{tag}_shard*_preds.npz"))):
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            kind, rest = k.split("::", 1)
            if kind == "g":
                G[rest] = z[k]
                parts = rest.split("|")
                SEEDS["|".join(parts[:3])].add(int(parts[3]))
            elif kind == "q":
                Q[rest] = z[k]
            elif kind == "anchor":
                ANCH[rest] = z[k]
            else:
                tk, fld = rest.rsplit("::", 1)
                EVAL.setdefault(tk, {})[fld] = z[k]
    for f in sorted(glob.glob(str(M1 / f"{tag}_shard*_meta.json"))):
        META.append(json.loads(Path(f).read_text()))
res = pd.concat(res, ignore_index=True)
if "error" in res:
    print(f"[load] {len(res)}행 · 오류행 {res.error.notna().sum()}")
    errs = res[res.error.notna()]
    if len(errs):
        print(errs.groupby(["resid", "error"]).size().to_string())
    res = res[res.error.isna()].copy()
if "iw" not in res:
    res["iw"] = 0
res["iw"] = res["iw"].fillna(0).astype(int)
print(f"[load] tags {TAGS} · {len(res)}행 · g {len(G)} · anchor {len(ANCH)} · 평가집합 {len(EVAL)}")

# 평가 셀 공변량(완전 사례·물리 일관성·결합 앵커 분해용). loc_id로 결합.
BASE_FILE = args.base or (META[0].get("base", "fidelity_base_v3.csv") if META else "fidelity_base_v3.csv")
SOC_COLS = [c for c in SOIL if "soc" in c]
BASE = None
try:
    BASE = pd.read_csv(ROOT / "data" / "processed" / BASE_FILE, usecols=["loc_id", "e5_sqrt_tdd", "cci_alt"] + SOIL).set_index("loc_id")
    print(f"[base] {BASE_FILE} · {len(BASE):,}셀 (평가 셀 공변량)")
except Exception as e:                                       # noqa: BLE001
    print(f"[base] 공변량 파일 없음({BASE_FILE}: {e}) → rmse_cc·consistency·combo 분해 생략")


def cov_of(tk):
    """평가 셀 순서의 공변량 DataFrame(없으면 None)."""
    return BASE.reindex(EVAL[tk]["loc_id"]) if (BASE is not None and tk in EVAL) else None


_CC = {}


def cc_mask(tk):
    """완전 사례: SoilGrids 9종·CCI 모두 유효(개정 09-21 §E)."""
    if tk not in _CC:
        cv = cov_of(tk)
        _CC[tk] = None if cv is None else (cv[SOIL].notna().all(axis=1) & cv.cci_alt.notna()).values
    return _CC[tk]


# ---------------------------------------------------------------- 예측 재구성
def seeds_of(tk):
    return sorted(SEEDS.get(tk, ()))


def key_of(tk, seed, c):
    return f"{tk}|{seed}|{c['dset']}|{c['xset']}|{c['anchor']}|{c['pseudo']}|{float(c['r'])}|{c['resid']}|{int(c.get('iw', 0))}"


def predict_seeds(tk, c, lam):
    """seed별 예측 dict {seed: 벡터}(평가 셀 순서). anchor=none 이면 직접 회귀. 없으면 None."""
    out = {}
    for s in seeds_of(tk):
        k = key_of(tk, s, c)
        if k not in G:
            continue
        g = G[k].astype(float)
        out[s] = g if c["anchor"] == "none" else ANCH[f"{tk}|{c['dset']}|{c['anchor']}"].astype(float) + lam * g
    return out or None


def predict(tk, c, lam, per_seed=False):
    """seed 평균 예측(평가 셀 순서). per_seed=True 이면 seed 순 목록."""
    d = predict_seeds(tk, c, lam)
    if d is None:
        return None
    ps = [d[s] for s in sorted(d)]
    return ps if per_seed else np.mean(ps, axis=0)


def anchor_only(tk, dset, anchor):
    return ANCH.get(f"{tk}|{dset}|{anchor}")


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.sum() else np.nan


def full_cfg(spec):
    c = dict(dset="alaska", xset="x25", r=0.0, iw=0)
    c.update({k: v for k, v in spec.items() if k != "lam"})
    return c


def pred_spec(spec, tk):
    """spec → seed 평균 예측(앵커 단독이면 앵커)."""
    if "anchor_only" in spec:
        return anchor_only(tk, spec.get("dset", "alaska"), spec["anchor_only"])
    return predict(tk, full_cfg(spec), spec.get("lam", 0.0))


def pred_seeds(spec, tk):
    """spec → {seed: 예측}. 앵커 단독은 seed 무관이므로 {-1: 앵커}."""
    if "anchor_only" in spec:
        a = anchor_only(tk, spec.get("dset", "alaska"), spec["anchor_only"])
        return None if a is None else {-1: a.astype(float)}
    return predict_seeds(tk, full_cfg(spec), spec.get("lam", 0.0))


def pair_seeds(dA, dB):
    """seed 짝지음: 같은 seed끼리, 한쪽이 seed 무관(앵커)이면 방송. 반환 (S,n) 두 행렬."""
    sa, sb = sorted(dA), sorted(dB)
    if sa == [-1] and sb == [-1]:
        pairs = [(-1, -1)]
    elif sa == [-1]:
        pairs = [(-1, s) for s in sb]
    elif sb == [-1]:
        pairs = [(s, -1) for s in sa]
    else:
        common = sorted(set(sa) & set(sb))
        pairs = [(s, s) for s in common] if common else list(zip(sa, sb))
    return np.stack([dA[a] for a, _ in pairs]), np.stack([dB[b] for _, b in pairs])


def family_tag(spec):
    """구성 표기(개정 09-21 §E)."""
    if "anchor_only" in spec:
        return "analytic"
    a, p, lam = spec.get("anchor", "none"), spec.get("pseudo", "none"), float(spec.get("lam", 1.0))
    if a != "none" and lam == 0.0:
        return "analytic"
    if a != "none" and p != "none":
        return "zero_resid_shrink" if a == p else "formula_mix"
    if a != "none":
        return "residual"
    if p != "none":
        return "augment_direct"
    return "direct"


# ---------------------------------------------------------------- 앙상블(트리+신경망) 합성: 구성원 g 평균 → G·res에 일반 구성처럼 추가
def add_ensembles():
    global res
    base_cols = ["cond", "target", "split", "seed", "dset", "xset", "anchor", "pseudo", "r", "iw"]
    new = []
    for name, members in ENSEMBLES.items():
        sub = res[res.resid.isin(members)]
        for vals, grp in sub.groupby(base_cols, sort=False):
            if set(grp.resid) != set(members):
                continue
            c = dict(zip(base_cols, vals))
            c["split"], c["seed"], c["r"], c["iw"] = int(c["split"]), int(c["seed"]), float(c["r"]), int(c["iw"])
            tk = f"{c['cond']}|{c['target']}|{c['split']}"
            keys = [key_of(tk, c["seed"], dict(c, resid=m)) for m in members]
            if tk not in EVAL or not all(k in G for k in keys):
                continue
            g = np.mean([G[k].astype(float) for k in keys], axis=0)
            G[key_of(tk, c["seed"], dict(c, resid=name))] = g.astype(np.float32)
            y = EVAL[tk]["y"].astype(float)
            anc = None if c["anchor"] == "none" else ANCH[f"{tk}|{c['dset']}|{c['anchor']}"].astype(float)
            lams = sorted(set.intersection(*(set(grp[grp.resid == m].lam) for m in members)))
            first = grp.iloc[0]
            fit_s = float(sum(grp[grp.resid == m].fit_s.iloc[0] for m in members))
            for lam in lams:
                p = g if anc is None else anc + lam * g
                ok = np.isfinite(y) & np.isfinite(p)
                e = p[ok] - y[ok]
                sst = float(((y[ok] - y[ok].mean()) ** 2).sum())
                new.append(dict(c, resid=name, n=int(first.n), n_blocks=int(first.n_blocks), n_train=first.n_train, n_train_real=first.n_train_real,
                                lam=float(lam), rmse_cm=float(np.sqrt(np.mean(e ** 2))), bias_cm=float(e.mean()), mae_cm=float(np.abs(e).mean()),
                                r2=(1.0 - float((e ** 2).sum()) / sst) if sst > 0 else np.nan, fit_s=fit_s))
    if new:
        res = pd.concat([res, pd.DataFrame(new)], ignore_index=True)
        print(f"[ensemble] 합성 행 {len(new)} ({', '.join(f'{k}={v}' for k, v in ENSEMBLES.items())})")


add_ensembles()


# ---------------------------------------------------------------- 요약표 (seed 평균 RMSE · seed 앙상블 RMSE · 분할 평균 · 완전 사례)
key = ["cond", "target"] + CFG + ["lam"]
per = (res.groupby(key + ["split"], as_index=False)
          .agg(rmse=("rmse_cm", "mean"), bias=("bias_cm", "mean"), n=("n", "first"), n_blocks=("n_blocks", "first"),
               n_seed=("seed", "nunique"), fit_s=("fit_s", "mean")))
ens, rcc, ncc, chk = [], [], [], []
for _, r in per.iterrows():
    tk = f"{r.cond}|{r.target}|{int(r.split)}"
    c = {f: r[f] for f in CFG}
    ps = predict(tk, c, r.lam, per_seed=True)
    if ps is None or tk not in EVAL:
        ens.append(np.nan); rcc.append(np.nan); ncc.append(np.nan)
        continue
    y = EVAL[tk]["y"].astype(float)
    ens.append(rmse(y, np.mean(ps, axis=0)))                                   # seed 앙상블(예측 평균 후 채점)
    chk.append(abs(float(np.mean([rmse(y, p) for p in ps])) - float(r.rmse)))  # seed별 RMSE 평균 재계산 대 하네스 값
    cc = cc_mask(tk)
    if cc is None or not cc.any():
        rcc.append(np.nan); ncc.append(np.nan)
    else:
        rcc.append(float(np.mean([rmse(y[cc], p[cc]) for p in ps]))); ncc.append(int(cc.sum()))
per["rmse_ens"], per["rmse_cc"], per["n_cc"] = ens, rcc, ncc
if chk:
    print(f"[summary] seed별 RMSE 재계산 대 하네스 rmse_cm 최대 차이 {max(chk):.4f} cm")
summ = (per.groupby(key, as_index=False)
           .agg(rmse=("rmse", "mean"), rmse_split_sd=("rmse", "std"), rmse_ens=("rmse_ens", "mean"), bias=("bias", "mean"),
                n=("n", "first"), n_blocks=("n_blocks", "first"), n_split=("split", "nunique"), n_seed=("n_seed", "min"),
                fit_s=("fit_s", "mean"), rmse_cc=("rmse_cc", "mean"), n_cc=("n_cc", "first")))
summ["regime"] = np.where(summ.target.isin(TRANSFER_MAIN), "main", np.where(summ.target.isin(TRANSFER_DEEP), "deep", "alaska"))
summ["family_tag"] = [family_tag(dict(anchor=r.anchor, pseudo=r.pseudo, lam=r.lam)) for r in summ.itertuples()]
summ.to_csv(M1 / f"{OUT}_summary.csv", index=False)
print(f"[summary] {len(summ)}행 → {OUT}_summary.csv")


# ---------------------------------------------------------------- 짝지은 블록 부트스트랩 (3종 채점 · seed별 짝지음)
def _seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def _block_sse(y, inv, nb, P):
    """P (S,n) → 블록별 SSE (S,nb)."""
    E2 = (P - y[None, :]) ** 2
    return np.stack([np.bincount(inv, weights=E2[s], minlength=nb) for s in range(P.shape[0])])


def _scores(SSE, n_b, pick=None):
    """세 채점의 RMSE: 셀 가중·블록 등가중·상한 100 가중. pick=None → 각 (S,), pick (R,nb) 블록 재표집 → 각 (S,R)."""
    w_b = np.minimum(1.0, CAP_CELLS / n_b)
    if pick is None:
        return (np.sqrt(SSE.sum(1) / n_b.sum()), np.sqrt(SSE / n_b).mean(1),
                np.sqrt((SSE * w_b).sum(1) / (w_b * n_b).sum()))
    S_, n_, w_ = SSE[:, pick], n_b[pick], w_b[pick]                 # (S,R,nb), (R,nb), (R,nb)
    return (np.sqrt(S_.sum(2) / n_.sum(1)[None, :]), np.sqrt(S_ / n_[None]).mean(2),
            np.sqrt((S_ * w_[None]).sum(2) / (w_ * n_).sum(1)[None, :]))


def boot_delta(y, blocks, PA, PB, nboot, seed):
    """PA·PB (S,n) seed 짝지은 예측. Δ = RMSE(A) − RMSE(B) 점추정(seed별 평균·앙상블·3종 채점), 블록별 Δ, 블록 재표집 분포.
    블록 < MIN_BLOCKS_CI 이면 분포 None."""
    m = np.isfinite(y) & np.all(np.isfinite(PA), 0) & np.all(np.isfinite(PB), 0)
    y, PA, PB, blocks = y[m], PA[:, m], PB[:, m], blocks[m]
    ub, inv = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    n_b = np.bincount(inv, minlength=nb).astype(float)
    SA, SB = _block_sse(y, inv, nb, PA), _block_sse(y, inv, nb, PB)
    EA, EB = _block_sse(y, inv, nb, PA.mean(0, keepdims=True)), _block_sse(y, inv, nb, PB.mean(0, keepdims=True))
    cA, bA, kA = _scores(SA, n_b)
    cB, bB, kB = _scores(SB, n_b)
    d_blk = (np.sqrt(SA / n_b) - np.sqrt(SB / n_b)).mean(0)        # 블록별 ΔRMSE(seed 평균)
    out = dict(n_cells=int(m.sum()), n_blocks=nb, n_seed=int(PA.shape[0]),
               delta_per_seed=float((cA - cB).mean()), delta_blockeq=float((bA - bB).mean()), delta_cap100=float((kA - kB).mean()),
               delta_ensemble=float(_scores(EA, n_b)[0][0] - _scores(EB, n_b)[0][0]),
               block_delta=d_blk, block_neg_k=int((d_blk < 0).sum()), block_majority=float((d_blk < 0).mean()),
               dist_cell=None, dist_blockeq=None, dist_ens=None)
    if nb >= MIN_BLOCKS_CI and nboot > 0:
        pick = np.random.RandomState(seed).randint(0, nb, size=(nboot, nb))
        cA_, bA_, _ = _scores(SA, n_b, pick)
        cB_, bB_, _ = _scores(SB, n_b, pick)
        out["dist_cell"], out["dist_blockeq"] = (cA_ - cB_).mean(0), (bA_ - bB_).mean(0)
        out["dist_ens"] = (_scores(EA, n_b, pick)[0] - _scores(EB, n_b, pick)[0])[0]
    return out


def ci(dist):
    return (float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))) if dist is not None else (np.nan, np.nan)


def boot_p(dist):
    """부트스트랩 p: 분포에서 0 반대편 비율의 2배(최소 1/nboot, 최대 1)."""
    if dist is None or not len(dist):
        return np.nan
    p = 2.0 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return float(min(1.0, max(p, 1.0 / len(dist))))


def strat(dists):
    """지역 층화 결합: 부트스트랩 분포가 있는 지역(블록≥8)만 독립 결합해 비가중 평균 Δ의 분포. 하나도 없으면 None."""
    ok = [d for d in dists if d is not None]
    if not ok:
        return None
    L = min(len(d) for d in ok)
    return np.mean([d[:L] for d in ok], axis=0)


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < MIN_BLOCKS_TEST or np.all(d == 0):
        return np.nan
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def strat_perm_p(groups, nperm, seed):
    """층화 순열검정: 층(지역) 내 블록 Δ의 부호 뒤집기. 통계량 = 지역별 블록 평균 Δ의 지역 비가중 평균."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups or sum(len(g) for g in groups) < MIN_BLOCKS_TEST or nperm <= 0:
        return np.nan
    T = float(np.mean([g.mean() for g in groups]))
    rng = np.random.RandomState(seed)
    Tp = np.zeros(nperm)
    for g in groups:
        Tp += (rng.choice([-1.0, 1.0], size=(nperm, len(g))) * g[None, :]).mean(1)
    Tp /= len(groups)
    return float((1 + np.sum(np.abs(Tp) >= abs(T) - 1e-12)) / (1 + nperm))


def pooled_eval(cond, tg):
    """같은 조건·대상의 split(또는 알래스카 fold)을 이어붙인 평가 집합 키 목록."""
    return sorted([k for k in EVAL if k.startswith(f"{cond}|{tg}|")], key=lambda s: int(s.split("|")[2]))


TESTS, SPLITS, COMBO_REQ = [], [], []
ST, SC = "stefan", "stefan_cci"


def add_test(hid, label, cond, A, B, targets, splits="all", family=None):
    """A·B: dict(anchor_only=...) 또는 dict(anchor,pseudo,r,resid,lam[,dset,xset,iw]). 지역별 Δ + 지역 요약."""
    fam = family or FAMILY.get(hid, "exploratory")
    tagA, tagB = family_tag(A), family_tag(B)
    prim = "dist_ens" if args.pairing == "ensemble" else "dist_cell"
    pkey = "delta_ensemble" if args.pairing == "ensemble" else "delta_per_seed"
    reg = []
    for tg in targets:
        yy, bb, PA_, PB_, sps = [], [], [], [], []
        for tk in pooled_eval(cond, tg):
            sp = int(tk.split("|")[2])
            if splits != "all" and sp not in splits:
                continue
            dA, dB = pred_seeds(A, tk), pred_seeds(B, tk)
            if dA is None or dB is None:
                continue
            PA, PB = pair_seeds(dA, dB)
            ev = EVAL[tk]
            yy.append(ev["y"].astype(float)); bb.append(ev["block"]); PA_.append(PA); PB_.append(PB); sps.append(sp)
        if not yy:
            continue
        S = min(p.shape[0] for p in PA_)                        # 분할 간 seed 수가 다르면 최소로 절단
        PA_, PB_ = [p[:S] for p in PA_], [p[:S] for p in PB_]
        # 분할 반복은 같은 셀이 여러 분할의 B에 등장하므로 분할별 Δ의 평균과 분할 간 SD를 별도 기록
        per_split = [boot_delta(y_, b_, a_, b2_, 0, 0) for y_, b_, a_, b2_ in zip(yy, bb, PA_, PB_)]
        y, blocks = np.concatenate(yy), np.concatenate(bb)
        r = boot_delta(y, blocks, np.concatenate(PA_, 1), np.concatenate(PB_, 1), args.nboot, _seed_of(hid, label, cond, tg))
        d0, dist = r[pkey], r[prim]
        lo, hi = ci(dist)
        lo_e, hi_e = ci(r["dist_ens"])
        lo_b, hi_b = ci(r["dist_blockeq"])
        sd = float(dist.std()) if dist is not None else np.nan
        ps = np.array([s[pkey] for s in per_split])
        row = dict(H=hid, label=label, cond=cond, target=tg, n_splits=len(yy), delta_rmse=d0, ci_lo=lo, ci_hi=hi,
                   boot_sd=sd, mde80=(2.8 * sd if np.isfinite(sd) else np.nan), n_cells=r["n_cells"], n_blocks=r["n_blocks"],
                   ci_flag=("ok" if r["n_blocks"] >= MIN_BLOCKS_CI else f"blocks<{MIN_BLOCKS_CI}"),
                   delta_split_mean=float(ps.mean()), delta_split_sd=float(ps.std()) if len(ps) > 1 else np.nan,
                   A=json.dumps(A), B=json.dumps(B),
                   pairing=args.pairing, delta_per_seed=r["delta_per_seed"], delta_ensemble=r["delta_ensemble"],
                   ci_lo_ens=lo_e, ci_hi_ens=hi_e, delta_cell=r["delta_per_seed"], delta_blockeq=r["delta_blockeq"],
                   delta_cap100=r["delta_cap100"], ci_lo_blockeq=lo_b, ci_hi_blockeq=hi_b,
                   block_neg_k=r["block_neg_k"], block_majority=r["block_majority"], p_boot=boot_p(dist),
                   wilcoxon_p=wilcoxon_p(r["block_delta"]),
                   strat_perm_p=strat_perm_p([r["block_delta"]], args.nperm, _seed_of("perm", hid, label, cond, tg)),
                   sign_test_p_descriptive=np.nan, n_regions=1, neg_regions=int(d0 < 0), delta_region_median=d0,
                   delta_region_cellw=d0, delta_rmse_ci_regions=(d0 if dist is not None else np.nan), n_regions_ci=int(dist is not None),
                   n_seed=r["n_seed"], family=fam, family_tag=tagA, family_tag_B=tagB, p_holm=np.nan)
        TESTS.append(row)
        for sp, s in zip(sps, per_split):
            SPLITS.append(dict(H=hid, label=label, cond=cond, target=tg, split=sp, delta_rmse=s[pkey],
                               delta_per_seed=s["delta_per_seed"], delta_ensemble=s["delta_ensemble"], delta_blockeq=s["delta_blockeq"],
                               delta_cap100=s["delta_cap100"], n_cells=s["n_cells"], n_blocks=s["n_blocks"],
                               block_majority=s["block_majority"], n_splits=len(yy), delta_split_mean=row["delta_split_mean"],
                               delta_split_sd=row["delta_split_sd"], family=fam, family_tag=tagA))
        reg.append(dict(row, _r=r))
    if not reg:
        return
    if A == dict(anchor_only=SC) and B == dict(anchor_only=ST):
        COMBO_REQ.append((hid, label, cond, [x["target"] for x in reg], splits))
    groups = [("REGION_SUMMARY_MAIN", TRANSFER_MAIN), ("REGION_SUMMARY_AB4", AB4), ("REGION_SUMMARY_DEEP", TRANSFER_DEEP), ("REGION_SUMMARY_ALL", None)]
    for grp, members in groups:
        sel = [x for x in reg if members is None or x["target"] in members]
        if grp == "REGION_SUMMARY_AB4":                          # AB4 = 명목 4지역 중 CI 산출 가능(블록≥8) 지역만
            sel = [x for x in sel if x["_r"][prim] is not None]
        if not sel:
            continue
        ds = np.array([x["delta_rmse"] for x in sel]); neg = int((ds < 0).sum())
        p_sign = binomtest(neg, len(ds), 0.5).pvalue if len(ds) >= 5 else np.nan
        w = np.array([x["n_cells"] for x in sel], float)
        used = [x for x in sel if x["_r"][prim] is not None]        # 층화 CI는 분포가 있는 지역만(개정 09-21 §B)
        excl = [x["target"] for x in sel if x["_r"][prim] is None]
        flag = "stratified_boot" + (f"(excl:{','.join(excl)})" if excl else "")
        if grp == "REGION_SUMMARY_AB4":
            miss = [t for t in AB4 if t not in [x["target"] for x in sel]]
            flag += f"(missing:{','.join(miss)})" if miss else ""
        s_prim, s_ens, s_beq = (strat([x["_r"][k] for x in used]) for k in (prim, "dist_ens", "dist_blockeq"))
        lo, hi = ci(s_prim); lo_e, hi_e = ci(s_ens); lo_b, hi_b = ci(s_beq)
        sd = float(s_prim.std()) if s_prim is not None else np.nan
        blk = [x["_r"]["block_delta"] for x in sel]
        nblk = int(sum(len(b) for b in blk)); kneg = int(sum(x["block_neg_k"] for x in sel))
        mean_of = lambda k: float(np.mean([x[k] for x in sel]))   # noqa: E731
        TESTS.append(dict(H=hid, label=label, cond=cond, target=grp, n_splits=np.nan, delta_rmse=float(ds.mean()), ci_lo=lo, ci_hi=hi,
                          boot_sd=sd, mde80=(2.8 * sd if np.isfinite(sd) else np.nan), n_cells=len(ds), n_blocks=neg, ci_flag=flag,
                          delta_split_mean=float(np.median(ds)), delta_split_sd=float((ds * w).sum() / w.sum()),
                          A=f"neg_regions={neg}/{len(ds)} [{','.join(x['target'] for x in sel)}]", B=f"sign_test_p={p_sign}",
                          pairing=args.pairing, delta_per_seed=mean_of("delta_per_seed"), delta_ensemble=mean_of("delta_ensemble"),
                          ci_lo_ens=lo_e, ci_hi_ens=hi_e, delta_cell=mean_of("delta_cell"), delta_blockeq=mean_of("delta_blockeq"),
                          delta_cap100=mean_of("delta_cap100"), ci_lo_blockeq=lo_b, ci_hi_blockeq=hi_b,
                          block_neg_k=kneg, block_majority=(kneg / nblk if nblk else np.nan), p_boot=boot_p(s_prim),
                          wilcoxon_p=wilcoxon_p(np.concatenate(blk)),
                          strat_perm_p=strat_perm_p(blk, args.nperm, _seed_of("perm", hid, label, cond, grp)),
                          sign_test_p_descriptive=p_sign, n_regions=len(ds), neg_regions=neg, delta_region_median=float(np.median(ds)),
                          delta_region_cellw=float((ds * w).sum() / w.sum()),
                          delta_rmse_ci_regions=(float(np.mean([x["delta_rmse"] for x in used])) if used else np.nan), n_regions_ci=len(used),
                          n_seed=int(min(x["n_seed"] for x in sel)), family=fam, family_tag=tagA, family_tag_B=tagB, p_holm=np.nan))


MAIN = [t for t in TRANSFER_MAIN if any(k.split("|")[1] == t for k in EVAL)]
DEEP = [t for t in TRANSFER_DEEP if any(k.split("|")[1] == t for k in EVAL)]
ALL_T = MAIN + DEEP

if args.tests:
    for t in json.loads(Path(args.tests).read_text()):
        add_test(t["H"], t["label"], t["cond"], t["A"], t["B"], t.get("targets", ALL_T), t.get("splits", "all"), t.get("family"))
else:
    # H1/H1b 재확인: Stefan+CCI 앵커 − Stefan 앵커 (λ=0)
    for cond in ["noinfo", "covonly", "deploy"]:
        add_test("H1", "Stefan+CCI 앵커 − Stefan 앵커 (λ=0)", cond, dict(anchor_only=SC), dict(anchor_only=ST), ALL_T)
    # H13: 라벨 있음 조건에서 잔차 결합 − 앵커 단독 (catboost_lo λ=.25·.5)
    for lam in [0.25, 0.5]:
        add_test("H13", f"labels: Stefan 앵커+catboost_lo(λ={lam}) − Stefan 앵커", "labels",
                 dict(anchor=ST, pseudo="none", r=0.0, resid="catboost_lo", lam=lam), dict(anchor_only=ST), ALL_T)
    # H2 재확인(공변량만·정보 없음): 앵커+catboost_lo λ=.25 − 앵커
    for cond in ["covonly", "noinfo"]:
        add_test("H2", f"{cond}: Stefan+CCI 앵커+catboost_lo(λ=.25) − 앵커 단독", cond,
                 dict(anchor=SC, pseudo="none", r=0.0, resid="catboost_lo", lam=0.25), dict(anchor_only=SC), ALL_T)
    # H3 대조군 사다리(공변량만, 직접 catboost_lo, r=10)
    for ctrl in ["const", "const_t", "shuffle", "tddlin", "none"]:
        add_test("H3", f"covonly: Stefan 유사라벨 − {ctrl} (직접 catboost_lo, r=10)", "covonly",
                 dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0),
                 dict(anchor="none", pseudo=ctrl, r=(0.0 if ctrl == "none" else 10.0), resid="catboost_lo", lam=1.0), MAIN)
    # H10: 물리 계수 지도 앵커 − Stefan 앵커
    for em in ["emap_ridge", "emap_cb"]:
        for cond in ["covonly", "noinfo"]:
            add_test("H10", f"{cond}: {em} − Stefan (λ=0)", cond, dict(anchor_only=em), dict(anchor_only=ST), ALL_T)
    # H14: 학습 지역 집합 확장 − 알래스카만 (Stefan 앵커 + catboost_lo λ=.25, 정보 없음)
    for d in ["loro", "loro_main"]:
        for cond in ["noinfo", "covonly"]:
            add_test("H14", f"{cond}: D={d} − D=alaska (Stefan+catboost_lo λ=.25)", cond,
                     dict(dset=d, anchor=ST, pseudo="none", r=0.0, resid="catboost_lo", lam=0.25),
                     dict(dset="alaska", anchor=ST, pseudo="none", r=0.0, resid="catboost_lo", lam=0.25), ALL_T)
    # 탐색적 앵커 축(λ=0): 각 앵커 − Stefan
    for a in ["stefan_k2", "stefan_med", "cci_cal", "cci_raw", "stefan_cci_cal", "stefan_soil", "stefan_soil_cci", "ku_cal", "stefan_ku", "stefan_ku_cci"]:
        for cond in ["covonly", "noinfo"]:
            add_test("X-anchor", f"{cond}: {a} − stefan (λ=0)", cond, dict(anchor_only=a), dict(anchor_only=ST), ALL_T)
    # 탐색적 모델 축: 각 모델 직접 − Stefan 앵커, 각 모델 Stefan 잔차(λ=.25) − Stefan
    for m in ["ridge", "catboost_lo", "catboost", "mlp", "ftt", "tabm", "realmlp", "cfm", "ddpm", "nflow"]:
        for cond in ["covonly", "noinfo", "labels"]:
            add_test("X-model", f"{cond}: {m} 직접 − Stefan 앵커", cond, dict(anchor="none", pseudo="none", r=0.0, resid=m, lam=1.0), dict(anchor_only=ST), ALL_T)
            add_test("X-model", f"{cond}: Stefan+{m}(λ=.25) − Stefan", cond, dict(anchor=ST, pseudo="none", r=0.0, resid=m, lam=0.25), dict(anchor_only=ST), ALL_T)
    # 탐색적 T3 중요도 가중
    for m in ["catboost_lo", "mlp"]:
        for cond in ["covonly", "noinfo"]:
            add_test("X-iw", f"{cond}: {m} 직접 iw=1 − iw=0", cond, dict(anchor="none", pseudo="none", r=0.0, resid=m, iw=1, lam=1.0),
                     dict(anchor="none", pseudo="none", r=0.0, resid=m, iw=0, lam=1.0), ALL_T)
    # 탐색적 유사라벨 축(태그 m1_pseudo): 각 유사라벨 − Stefan 유사라벨 (직접 catboost_lo, r=10), 공변량만·배포형
    for cond in ["covonly", "deploy"]:
        for pl in ["stefan_k2", "stefan_soil", "ku_cal", "ku_k2", "edaphic_k2", "cci", "const_t", "shuffle", "tddlin", "const", "none"]:
            add_test("X-pseudo", f"{cond}: 유사라벨 {pl} − stefan (직접 catboost_lo, r=10)", cond,
                     dict(anchor="none", pseudo=pl, r=(0.0 if pl == "none" else 10.0), resid="catboost_lo", lam=1.0),
                     dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0), ALL_T)
    # 알래스카 지역 내
    if any(k.startswith("labels|Alaska|") for k in EVAL):
        add_test("H6", "알래스카: Stefan+CCI 앵커 − Stefan (λ=0)", "labels", dict(anchor_only=SC), dict(anchor_only=ST), ["Alaska"])
        add_test("H6c", "알래스카: Stefan+CCI(보정) 앵커 − Stefan (λ=0)", "labels", dict(anchor_only="stefan_cci_cal"), dict(anchor_only=ST), ["Alaska"])
        for m in ["ridge", "catboost_lo", "mlp", "cfm", "nflow", "realmlp"]:
            add_test("X-ak", f"알래스카: Stefan+{m}(λ=.75) − Stefan", "labels", dict(anchor=ST, pseudo="none", r=0.0, resid=m, lam=0.75), dict(anchor_only=ST), ["Alaska"])
            add_test("X-ak", f"알래스카: {m} 직접 − Stefan", "labels", dict(anchor="none", pseudo="none", r=0.0, resid=m, lam=1.0), dict(anchor_only=ST), ["Alaska"])


# ---------------------------------------------------------------- Holm 보정 (가족 내, 개정 09-21 §C)
def holm(p):
    """Holm step-down 보정. NaN은 제외(가족 크기에 미포함)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    m, run = len(idx), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


tests = pd.DataFrame(TESTS)
if len(tests):
    tests["holm_group"], tests["holm_m"] = "", np.nan
    # 검정 변형 단위 = (H, label, cond). 주 행 = REGION_SUMMARY_MAIN, 없으면 단일 대상 행. DEEP·ALL 요약은 별도 군.
    for (_, _, _), grp in tests.groupby(["H", "label", "cond"], sort=False):
        if grp.family.iloc[0] == "exploratory":
            continue
        is_sum = grp.target.str.startswith("REGION_SUMMARY")
        if (grp.target == "REGION_SUMMARY_MAIN").any():
            tests.loc[grp.index[grp.target == "REGION_SUMMARY_MAIN"][0], "holm_group"] = "primary"
        elif (~is_sum).sum() == 1:
            tests.loc[grp.index[~is_sum][0], "holm_group"] = "primary"
        for g in ["REGION_SUMMARY_AB4", "REGION_SUMMARY_DEEP", "REGION_SUMMARY_ALL"]:
            ii = grp.index[grp.target == g]
            if len(ii):
                tests.loc[ii[0], "holm_group"] = g.replace("REGION_SUMMARY_", "")
    for fam in [f for f in tests.family.unique() if f != "exploratory"]:
        for hg in ["primary", "AB4", "DEEP", "ALL"]:
            sel = tests.index[(tests.family == fam) & (tests.holm_group == hg)]
            if not len(sel):
                continue
            pv = tests.loc[sel, "p_boot"].values.astype(float)
            tests.loc[sel, "p_holm"] = pv if fam == "primary" else holm(pv)   # 주 가설 H12는 무보정
            tests.loc[sel, "holm_m"] = int(np.isfinite(pv).sum())
    tests.to_csv(M1 / f"{OUT}_tests.csv", index=False)
    pd.DataFrame(SPLITS).to_csv(M1 / f"{OUT}_splits.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== 짝지은 검정 (ΔRMSE = A − B, 음수 = A 우세; 주 짝지음 =", args.pairing, ") ===")
    show = tests[tests.target.str.startswith("REGION_SUMMARY") | tests.target.isin(["Alaska"])]
    cols = ["H", "label", "cond", "target", "delta_rmse", "ci_lo", "ci_hi", "delta_per_seed", "delta_ensemble", "delta_blockeq",
            "delta_cap100", "block_majority", "p_boot", "p_holm", "wilcoxon_p", "strat_perm_p", "n_cells", "n_blocks", "family"]
    print(show[cols].round(3).to_string(index=False, max_colwidth=48))


# ---------------------------------------------------------------- C2-A 결합 앵커 분해 (Stefan+CCI 대 Stefan)
def combo_decomp(hid, label, cond, targets, splits):
    rows = []
    for tg in targets:
        ys, sts, scs, cs, bs, dmax = [], [], [], [], [], 0.0
        for tk in pooled_eval(cond, tg):
            if splits != "all" and int(tk.split("|")[2]) not in splits:
                continue
            st, sc = anchor_only(tk, "alaska", ST), anchor_only(tk, "alaska", SC)
            if st is None or sc is None:
                continue
            st, sc = st.astype(float), sc.astype(float)
            recon = 2.0 * sc - st                                # 등가중 결합의 역산 CCI
            cv = cov_of(tk)
            c = recon if cv is None else np.where(np.isfinite(cv.cci_alt.values), cv.cci_alt.values, recon)
            if len(c) == 0:                                      # 평가 셀 0개(그린란드 A/B) 방어
                continue
            dmax = max(dmax, float(np.nanmax(np.abs(c - recon))))
            ys.append(EVAL[tk]["y"].astype(float)); sts.append(st); scs.append(sc); cs.append(c); bs.append(EVAL[tk]["block"])
        if not ys:
            continue
        y, st, sc, c, b = map(np.concatenate, (ys, sts, scs, cs, bs))
        m = np.isfinite(y) & np.isfinite(st) & np.isfinite(sc) & np.isfinite(c)
        y, st, sc, c, b = y[m], st[m], sc[m], c[m], b[m]
        es, ec, esc = st - y, c - y, sc - y
        S2s, S2c, Ssc = float(np.mean(es ** 2)), float(np.mean(ec ** 2)), float(np.mean(es * ec))
        vs, vc, cov = float(es.var()), float(ec.var()), float(np.cov(es, ec, bias=True)[0, 1])
        den_m, den_v = S2s + S2c - 2 * Ssc, vs + vc - 2 * cov
        w_mse = (S2c - Ssc) / den_m if den_m > 0 else np.nan          # MSE 최적 고정 가중(Stefan 쪽)
        w_var = (vc - cov) / den_v if den_v > 0 else np.nan            # 분산만 고려한 최적 가중(편향 보정 가정)
        rmse_w = float(np.sqrt(np.mean((w_mse * es + (1 - w_mse) * ec) ** 2))) if np.isfinite(w_mse) else np.nan
        ub = np.unique(b)
        rb_s = np.array([np.sqrt(np.mean(es[b == u] ** 2)) for u in ub])
        rb_c = np.array([np.sqrt(np.mean(esc[b == u] ** 2)) for u in ub])
        k = int((rb_c < rb_s).sum())
        rows.append(dict(H=hid, label=label, cond=cond, target=tg, n_cells=int(m.sum()), n_blocks=len(ub),
                         rho_err=float(np.corrcoef(es, ec)[0, 1]) if vs > 0 and vc > 0 else np.nan,
                         bias_stefan=float(es.mean()), var_stefan=vs, rmse_stefan=float(np.sqrt(S2s)),
                         bias_cci=float(ec.mean()), var_cci=vc, rmse_cci=float(np.sqrt(S2c)),
                         bias_combo=float(esc.mean()), var_combo=float(esc.var()), rmse_combo=float(np.sqrt(np.mean(esc ** 2))),
                         delta_combo_minus_stefan=float(np.sqrt(np.mean(esc ** 2)) - np.sqrt(S2s)),
                         w_opt_mse=w_mse, rmse_w_opt=rmse_w, w_opt_var=w_var,
                         block_neg_k=k, block_majority=k / len(ub), cci_recon_maxdiff=dmax))
    return rows


COMBO = [r for req in COMBO_REQ for r in combo_decomp(*req)]
if COMBO:
    combo = pd.DataFrame(COMBO)
    combo.to_csv(M1 / f"{OUT}_combo_decomp.csv", index=False)
    print("\n=== 결합 앵커 분해 (Stefan+CCI 대 Stefan; ρ = 오차 상관, w* = MSE 최적 Stefan 가중) ===")
    print(combo[["H", "cond", "target", "n_cells", "rho_err", "bias_stefan", "bias_cci", "rmse_stefan", "rmse_cci", "rmse_combo",
                 "delta_combo_minus_stefan", "w_opt_mse", "rmse_w_opt", "block_majority"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- 물리 일관성 지표 (예측 벡터 + 평가 셀 공변량)
_PAIRS = {}


def _pool_cov(cond, tg, tks):
    """tks 순서로 이어붙인 평가 셀의 y·√TDD·SOC 3분위 셀쌍(고정 seed, 같은 tk 집합이면 같은 쌍)."""
    kk = (cond, tg, tuple(tks))
    if kk in _PAIRS:
        return _PAIRS[kk]
    cv = pd.concat([cov_of(tk) for tk in tks])
    y = np.concatenate([EVAL[tk]["y"].astype(float) for tk in tks])
    s = cv.e5_sqrt_tdd.values.astype(float)
    soc = np.nanmean(cv[SOC_COLS].values.astype(float), axis=1) if SOC_COLS else np.zeros(len(cv))
    ok = np.isfinite(s) & (s > 0) & np.isfinite(soc)
    idx = np.where(ok)[0]
    rng = np.random.RandomState(_seed_of("pairs", cond, tg, *tks))
    pairs = []
    if len(idx) >= 4:
        ter = pd.qcut(soc[idx], 3, labels=False, duplicates="drop")
        for t in np.unique(ter):
            cells = idx[ter == t]
            if len(cells) < 2:
                continue
            n_t = int(np.ceil(args.npairs * len(cells) / len(idx)))
            i, j = rng.choice(cells, 6 * n_t), rng.choice(cells, 6 * n_t)
            keep = s[i] != s[j]
            i, j = i[keep][:n_t], j[keep][:n_t]
            pairs.append(np.c_[np.where(s[i] < s[j], i, j), np.where(s[i] < s[j], j, i)])
    P = np.vstack(pairs) if pairs else np.empty((0, 2), int)
    if len(P) > args.npairs:
        P = P[rng.permutation(len(P))[:args.npairs]]
    E_obs = y[ok] / s[ok]
    out = dict(y=y, s=s, ok=ok, pairs=P, n_pairs=len(P),
               mono_viol_obs=float(np.mean(y[P[:, 1]] < y[P[:, 0]])) if len(P) else np.nan,
               E_iqr_obs=float(np.subtract(*np.percentile(E_obs, [75, 25]))) if ok.any() else np.nan,
               E_med_obs=float(np.median(E_obs)) if ok.any() else np.nan)
    _PAIRS[kk] = out
    return out


def consistency_rows():
    if BASE is None:
        return []
    rows = []
    for r in summ.itertuples():
        c = {f: getattr(r, f) for f in CFG}
        preds = {}
        for tk in pooled_eval(r.cond, r.target):
            d = predict_seeds(tk, c, r.lam)
            if d is not None:
                preds[tk] = d
        if not preds:
            continue
        tks = list(preds)
        seeds = sorted(set.intersection(*(set(d) for d in preds.values())))
        if not seeds:
            continue
        pc = _pool_cov(r.cond, r.target, tks)
        s, ok, P = pc["s"], pc["ok"], pc["pairs"]
        rv, mv, iq, md = [], [], [], []
        for sd_ in seeds:
            p = np.concatenate([preds[tk][sd_] for tk in tks])
            f = np.isfinite(p)
            rv.append(float(np.mean((p[f] < RANGE_LO) | (p[f] > RANGE_HI))) if f.any() else np.nan)
            mv.append(float(np.mean(p[P[:, 1]] < p[P[:, 0]])) if len(P) else np.nan)
            E = p[ok & f] / s[ok & f]
            iq.append(float(np.subtract(*np.percentile(E, [75, 25]))) if len(E) else np.nan)
            md.append(float(np.median(E)) if len(E) else np.nan)
        rows.append(dict(cond=r.cond, target=r.target, **c, lam=r.lam, family_tag=r.family_tag, n_cells=len(pc["y"]), n_seed=len(seeds),
                         range_viol=float(np.mean(rv)), mono_viol=float(np.mean(mv)), mono_viol_obs=pc["mono_viol_obs"], n_pairs=pc["n_pairs"],
                         E_iqr_pred=float(np.mean(iq)), E_iqr_obs=pc["E_iqr_obs"],
                         E_iqr_ratio=(float(np.mean(iq)) / pc["E_iqr_obs"] if pc["E_iqr_obs"] else np.nan),
                         E_med_pred=float(np.mean(md)), E_med_obs=pc["E_med_obs"]))
    return rows


CONS = consistency_rows()
if CONS:
    cons = pd.DataFrame(CONS)
    cons.to_csv(M1 / f"{OUT}_consistency.csv", index=False)
    print("\n=== 물리 일관성 (구성 표기별 평균: 범위 위배율 · √TDD 단조성 위배율(관측 대비) · 암묵 E IQR 비) ===")
    print(cons.groupby(["cond", "family_tag"])[["range_viol", "mono_viol", "mono_viol_obs", "E_iqr_ratio"]].mean().round(3).to_string())


# ---------------------------------------------------------------- H15 UQ: 생성 모델 90% 구간 (q:: 5%·95% 분위; 잔차 모드는 앵커 + λ·분위)
UQ_NUM = ["n_cells", "n_blocks", "coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "width_median", "interval_score",
          "is_ci_lo", "is_ci_hi", "is_width_part", "is_under_part", "is_over_part", "rmse_point"]


def uq_rows():
    if not Q:
        return []
    by = defaultdict(dict)                                     # (cond, target, cfg) → {(split, seed): q키}
    for qk in Q:
        p = qk.split("|")
        by[(p[0], p[1], (p[4], p[5], p[6], p[7], float(p[8]), p[9], int(p[10])))][(int(p[2]), int(p[3]))] = qk
    rows = []
    for (cond, tg, cfg), d in by.items():
        c = dict(zip(CFG, cfg))
        lams = [1.0] if c["anchor"] == "none" else [0.25, 0.5, 0.75, 1.0]
        seeds, splits = sorted({s for _, s in d}), sorted({sp for sp, _ in d})
        for lam in lams:
            per_seed = []
            for s in seeds:
                yy, ll, hh, bb, pp = [], [], [], [], []
                for sp in splits:
                    qk = d.get((sp, s))
                    tk = f"{cond}|{tg}|{sp}"
                    if qk is None or tk not in EVAL:
                        continue
                    q = Q[qk].astype(float)
                    pt = G.get(key_of(tk, s, c))
                    pt = np.full(q.shape[1], np.nan) if pt is None else pt.astype(float)
                    if c["anchor"] == "none":
                        lo, hi = q[0], q[1]
                    else:
                        anc = ANCH[f"{tk}|{c['dset']}|{c['anchor']}"].astype(float)
                        lo, hi, pt = anc + lam * q[0], anc + lam * q[1], anc + lam * pt
                    yy.append(EVAL[tk]["y"].astype(float)); ll.append(np.minimum(lo, hi)); hh.append(np.maximum(lo, hi))
                    bb.append(EVAL[tk]["block"]); pp.append(pt)
                if not yy:
                    continue
                y, lo, hi, b, pt = map(np.concatenate, (yy, ll, hh, bb, pp))
                m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                y, lo, hi, b, pt = y[m], lo[m], hi[m], b[m], pt[m]
                hit = ((y >= lo) & (y <= hi)).astype(float)
                width, under, over = hi - lo, (2 / ALPHA) * np.maximum(lo - y, 0), (2 / ALPHA) * np.maximum(y - hi, 0)
                isc = width + under + over
                ub, inv = np.unique(b, return_inverse=True); nb = len(ub)
                n_b = np.bincount(inv, minlength=nb).astype(float)
                H_b, I_b = np.bincount(inv, weights=hit, minlength=nb), np.bincount(inv, weights=isc, minlength=nb)
                cov_ci = is_ci = (np.nan, np.nan)
                if nb >= MIN_BLOCKS_CI and args.nboot > 0:
                    pick = np.random.RandomState(_seed_of("uq", cond, tg, *cfg, lam, s)).randint(0, nb, size=(args.nboot, nb))
                    nn = n_b[pick].sum(1)
                    cov_ci, is_ci = ci(H_b[pick].sum(1) / nn), ci(I_b[pick].sum(1) / nn)
                per_seed.append(dict(cond=cond, target=tg, **c, lam=lam, seed=s, n_splits=len(yy), n_cells=int(m.sum()), n_blocks=nb,
                                     coverage=float(hit.mean()), cov_ci_lo=cov_ci[0], cov_ci_hi=cov_ci[1], width_mean=float(width.mean()),
                                     width_median=float(np.median(width)), interval_score=float(isc.mean()), is_ci_lo=is_ci[0], is_ci_hi=is_ci[1],
                                     is_width_part=float(width.mean()), is_under_part=float(under.mean()), is_over_part=float(over.mean()),
                                     rmse_point=rmse(y, pt), n_seed=1, family_tag=family_tag(dict(c, lam=lam)), note=""))
            if per_seed:
                rows += per_seed
                mean_row = dict(per_seed[0], seed="mean", n_seed=len(per_seed))
                for k in UQ_NUM:
                    mean_row[k] = float(np.nanmean([r[k] for r in per_seed]))
                rows.append(mean_row)
    ref = ROOT / "data" / "processed" / "e4_interval_score.csv"      # 참조 행: S11 알래스카 지역 내 CQR·상수 폭
    if ref.exists():
        for r in pd.read_csv(ref).itertuples():
            if r.interval not in ("cqr", "const_width"):
                continue
            rows.append(dict(cond="labels", target="Alaska", dset="ref", xset="ref", anchor="none", pseudo="none", r=0.0, resid=f"ref_{r.interval}",
                             iw=0, lam=1.0, seed="ref", n_splits=6, n_cells=int(r.n), n_blocks=np.nan, coverage=r.coverage, cov_ci_lo=r.cov_ci_lo,
                             cov_ci_hi=r.cov_ci_hi, width_mean=r.width_mean, width_median=r.width_median, interval_score=r.interval_score,
                             is_ci_lo=r.is_ci_lo, is_ci_hi=r.is_ci_hi, is_width_part=r.is_width_part, is_under_part=r.is_under_part,
                             is_over_part=r.is_over_part, rmse_point=np.nan, n_seed=np.nan, family_tag="reference",
                             note="e4_interval_score.csv (S11 알래스카 지역 내 6-fold, 참조)"))
    return rows


UQ = uq_rows()
if UQ:
    uq = pd.DataFrame(UQ)
    uq.to_csv(M1 / f"{OUT}_uq.csv", index=False)
    print("\n=== H15 UQ (90% 구간; seed 평균 행과 참조 행) ===")
    show = uq[uq.seed.astype(str).isin(["mean", "ref"])]
    print(show[["cond", "target", "anchor", "resid", "lam", "n_cells", "coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score",
                "rmse_point", "note"]].round(3).to_string(index=False, max_colwidth=40))


# ---------------------------------------------------------------- 중첩 선택 (leave-one-target-out, 주 집합)
def nested_selection(cond, cands, label):
    """cands: 구성 dict 목록(lam 포함). 각 대상 지역 t: 다른 주 지역 비가중 평균 RMSE 최소 구성을 골라 t에서 채점."""
    tab = summ[(summ.cond == cond) & (summ.target.isin(MAIN))]
    rows = []
    for t in MAIN:
        others = [o for o in MAIN if o != t]
        best, best_v = None, np.inf
        for c in cands:
            vals = []
            for o in others:
                q = tab[(tab.target == o)]
                for f in CFG + ["lam"]:
                    q = q[q[f] == c.get(f, {"dset": "alaska", "xset": "x25", "r": 0.0, "iw": 0}.get(f))]
                if len(q):
                    vals.append(float(q.rmse.iloc[0]))
            if len(vals) == len(others) and np.mean(vals) < best_v:
                best, best_v = c, float(np.mean(vals))
        if best is None:
            continue
        q = tab[tab.target == t]
        for f in CFG + ["lam"]:
            q = q[q[f] == best.get(f, {"dset": "alaska", "xset": "x25", "r": 0.0, "iw": 0}.get(f))]
        rows.append(dict(family=label, cond=cond, target=t, chosen=json.dumps(best), select_mean_others=best_v,
                         score=float(q.rmse.iloc[0]) if len(q) else np.nan))
    return rows


NEST = []
if len(summ):
    for cond in ["covonly", "noinfo", "labels", "deploy"]:
        sub = summ[(summ.cond == cond) & (summ.target.isin(MAIN))]
        if not len(sub):
            continue
        cands = [dict(zip(CFG + ["lam"], v)) for v in sub[CFG + ["lam"]].drop_duplicates().values]
        NEST += nested_selection(cond, cands, "all")
        NEST += nested_selection(cond, [c for c in cands if c["anchor"] != "none"], "anchored")
        NEST += nested_selection(cond, [c for c in cands if c["anchor"] != "none" and c["lam"] == 0.0], "analytic")
if NEST:
    nest = pd.DataFrame(NEST)
    nest.to_csv(M1 / f"{OUT}_nested.csv", index=False)
    print("\n=== 중첩 선택 (5지역 선택 → 1지역 채점, 주 집합) ===")
    print(nest.groupby(["cond", "family"]).score.agg(["mean", "count"]).round(2).to_string())

# ---------------------------------------------------------------- 재현 게이트
if args.gate:
    LEGACY = [  # (라벨, 조건, 대상, 구성/앵커, λ, 옛 값, 출처)
        ("Stefan 최소제곱 E 지역 내", "labels", "Alaska", dict(anchor_only=ST), 0, 14.46, "tab:transfer/E2"),
        ("Stefan 중앙값비 E 지역 내", "labels", "Alaska", dict(anchor_only="stefan_med"), 0, 14.56, "tab:physics S2"),
        ("Stefan+ridge λ=.75 (13.33)", "labels", "Alaska", dict(anchor=ST, pseudo="none", r=0.0, resid="ridge", lam=0.75), 0.75, 13.33, "tab:transfer S4"),
        ("ridge 직접 25종", "labels", "Alaska", dict(anchor="none", pseudo="none", r=0.0, resid="ridge", lam=1.0), 1, 13.62, "09-14 대조"),
        ("Stefan 정보 없음 레나", "noinfo", "Lena", dict(anchor_only=ST), 0, 21.62, "E1-E3"),
        ("Stefan 정보 없음 캐나다", "noinfo", "Canada", dict(anchor_only=ST), 0, 26.63, "E1-E3"),
        ("Stefan+CCI 정보 없음 레나", "noinfo", "Lena", dict(anchor_only=SC), 0, 20.76, "E1-E3"),
        ("Stefan+CCI 정보 없음 캐나다", "noinfo", "Canada", dict(anchor_only=SC), 0, 25.82, "E1-E3"),
        ("Stefan 공변량만 레나(split0)", "covonly", "Lena", dict(anchor_only=ST), 0, 16.11, "E3 adaptive/S12"),
        ("Stefan+CCI 공변량만 레나(split0)", "covonly", "Lena", dict(anchor_only=SC), 0, 14.57, "S12"),
        ("Stefan 유사라벨 r=10 catboost 레나", "covonly", "Lena", dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost", lam=1.0), 1, 16.55, "tab:aug S3"),
        ("Stefan 유사라벨 r=10 catboost 캐나다", "covonly", "Canada", dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost", lam=1.0), 1, 31.81, "tab:aug S3"),
        ("Stefan 유사라벨 r=10 catboost_lo 레나", "covonly", "Lena", dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0), 1, 16.48, "E1"),
        ("Stefan 유사라벨 r=10 catboost_lo 캐나다", "covonly", "Canada", dict(anchor="none", pseudo="stefan", r=10.0, resid="catboost_lo", lam=1.0), 1, 30.21, "E1"),
        ("증강 없음 catboost 레나", "covonly", "Lena", dict(anchor="none", pseudo="none", r=0.0, resid="catboost", lam=1.0), 1, 21.9, "tab:aug S3"),
        ("증강 없음 catboost 캐나다", "covonly", "Canada", dict(anchor="none", pseudo="none", r=0.0, resid="catboost", lam=1.0), 1, 34.0, "tab:aug S3"),
        ("MLP 34종 3-seed 앙상블", "labels", "Alaska", dict(xset="x34", anchor="none", pseudo="none", r=0.0, resid="mlp", lam=1.0), 1, 14.37, "tab:models S1(앙상블)"),
        ("CatBoost 34종 3-seed 앙상블", "labels", "Alaska", dict(xset="x34", anchor="none", pseudo="none", r=0.0, resid="catboost", lam=1.0), 1, 15.61, "tab:models S1(앙상블)"),
    ]
    rows = []
    for label, cond, tg, spec, lam, old, src in LEGACY:
        tks = [k for k in pooled_eval(cond, tg) if int(k.split("|")[2]) == 0 or tg == "Alaska"]
        yy, pp, pe = [], [], []
        for tk in tks:
            p = pred_spec(dict(spec, lam=lam), tk)
            if p is None:
                continue
            yy.append(EVAL[tk]["y"].astype(float)); pp.append(p)
            if "anchor_only" not in spec:
                pe.append(predict(tk, full_cfg(spec), lam, per_seed=True))
        if not yy:
            rows.append(dict(label=label, cond=cond, target=tg, old=old, new=np.nan, new_seedmean=np.nan, diff=np.nan, pass_=False, source=src, note="미산출"))
            continue
        y, p = np.concatenate(yy), np.concatenate(pp)
        new = rmse(y, p)                                         # seed 앙상블(풀링)
        # seed별 RMSE 평균(옛 규약 다수)
        if pe:
            n_seed = min(len(x) for x in pe)
            sm = float(np.mean([rmse(y, np.concatenate([x[s] for x in pe])) for s in range(n_seed)]))
        else:
            sm = new
        cand = [new, sm]
        best = min(cand, key=lambda v: abs(v - old))
        rows.append(dict(label=label, cond=cond, target=tg, old=old, new=new, new_seedmean=sm, diff=best - old,
                         pass_=abs(best - old) <= 0.1, source=src, note="풀링 앙상블·seed평균 중 근접값으로 판정"))
    gate = pd.DataFrame(rows)
    gate.to_csv(M1 / f"{OUT}_gate.csv", index=False)
    print("\n=== 재현 게이트 (허용 0.1 cm) ===")
    print(gate[["label", "old", "new", "new_seedmean", "diff", "pass_", "source"]].round(3).to_string(index=False))
    print(f"통과 {int(gate.pass_.sum())}/{len(gate)}")
extra = ([f"{OUT}_splits.csv"] + ([f"{OUT}_combo_decomp.csv"] if COMBO else []) + ([f"{OUT}_consistency.csv"] if CONS else [])
         + ([f"{OUT}_uq.csv"] if UQ else []))
print(f"\nsaved: {OUT}_summary.csv · {OUT}_tests.csv · {' · '.join(extra)} · {OUT}_nested.csv" + (" · gate" if args.gate else ""))
