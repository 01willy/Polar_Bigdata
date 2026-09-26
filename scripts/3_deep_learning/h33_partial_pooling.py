"""H33 Track B2 부분 풀링: E 추정기(κ=10 수축 · log E 오프셋 최대우도 · 혼합효과 부스팅 · PPI++)를 같은 라벨 추출에서 비교. CPU.

목적: 현행 κ=10 수축(h25 S1/S3)을 경험적 베이즈 계층 모형·혼합효과 부스팅·PPI++ 로 대체했을 때, 같은 n 격자·같은 분할·같은 라벨
      추출에서 물리식 대비 Δ 가 유지되는지, 그리고 라벨 전량(allA)에서 캐나다·CA-3 의 악화가 사라지는지(F5)를 판정한다.
      부수로 이론 필요 라벨 수 n_theory = σ²/(τ²·ε) 를 대상별로 내고 h25 경험 손익분기 n 과의 Spearman(F6)을 계산한다.
계획: docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3 공통 설계, §4 Track B2, §5 F5·F6, §8 산출물.
골격: scripts/3_deep_learning/h25_label_budget.py (대상 15개·LORO 원천 + 100 km 버퍼·A/B 블록 분할·라벨 추출 seed 규약).
공용: src/polar/h4_common.py (블록 SSE 저장·블록 부트스트랩 Δ·최소 n·절단 순위 상관·오프셋 최대우도).

입력: data/processed/fidelity_base_v3.csv + e5_soil_tdd_v3.csv (m1_core.load_base), 입력 x25, 물리 예측 E·sqrt(TDD)(열 e5_sqrt_tdd).
      F6 대조용 경험 손익분기 n: data/processed/h3/h25b_breakeven.csv(있으면, h4 규약 재실행본) 또는 h3/h25_breakeven.csv.
대상: 주 4지역(레나·캐나다·러시아 W·E) + 하위 지역(알래스카 6·캐나다 3·레나 2; h25 와 같은 k-means, subregions.csv 와 셀 수 대조).
원천(LORO): 대상 셀을 뺀 전체 라벨 셀에서 대상 경계 100 km 이내 제외. 앵커 E0 = 원천 최소제곱.
분할: 대상 블록 A/B 2분할(split 0..K-1). 라벨 풀 = A, 채점 = B(실측·CCI·토양 도일 유효). 라벨 n 개만 학습에 쓰고 나머지 A 는 무라벨 풀.
라벨 추출: RandomState(7919*sp + 1009*n + 31*rule_idx + rep), 규칙 random · kmedoid (h25 와 동일).

누설 규약: 대상 라벨은 선택된 n 개(allA 는 A 전체)만 추정기에 들어간다. B 의 라벨은 채점에만 쓴다.
  PPI 무라벨 풀 = A 중 선택되지 않은 셀의 공변량 예측값 f 만 사용(라벨 미사용). f 는 원천 라벨만으로 학습한다.
  mixed_boost 학습 행 = 원천 계층 지역 셀 + 대상 선택 라벨 n 개. k-중심 선택·표준화는 A 공변량만 쓴다.
  대상 전체 라벨로 구한 E_own·|log(E_own/E0)| 는 이론표(b2_theory)와 F5 의 구조/수준 구분에만 쓰고 추정기에는 넣지 않는다.

E 추정기
  shrink_k10  : E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10 (현행 h25 S1).
  offset_mle  : h4_common.offset_mle_prior/offset_mle_estimate. z = log y − log √TDD, 사전 N(log E0, τ²)(E0 = 원천 최소제곱),
                σ²·τ² 는 원천 macro 지역(셀 ≥ 3) 적률 추정. E = exp(사후 평균).
  offset_mle30: 민감도. 같은 함수에 min_cells = 30(그린란드 3셀·러시아 C 7셀 제외).
  offset_ls   : 추정 대상 대조. offset_mle 와 같은 가중 w = n/(n + σ²/τ²) 를 최소제곱 척도에 적용한다.
                log E = (1 − w)·log E0 + w·log E_ls(sel). 변형 E 단독만 채점한다.
  추정 대상 차이: shrink_k10·물리식 E0 는 최소제곱(LS) E 이고, offset_mle·ppi·mixed_boost 는 z 평균, 즉 기하평균 E 를 추정한다.
                w → 1(allA)이면 offset_mle 와 offset_ls 의 차는 전부 추정 대상 차이이고, offset_ls 와 shrink_k10 의 차는 전부
                풀링 가중(w 대 n/(n+κ)) 차이다. runs·ppi_ci 의 gm_ls_ratio = exp(mean z_sel)/E_ls(sel) 로 그 크기를 기록한다.
  mixed_boost : gpboost 1.7.4. 반응 z, 고정효과 = x25 트리 부스팅, 그룹 랜덤 절편 = 원천 계층 지역 + 대상. n = 0 이면 대상 b = 0(신규 그룹),
                n > 0 이면 대상 라벨 n 개를 학습에 넣고 group_data_pred 로 대상 그룹의 수축 b 를 얻는다. E 는 셀별 exp(F(x) + b).
                gpboost 는 C++ 단언 실패(Eigen)로 프로세스가 중단될 수 있으므로 적합을 별도 파이썬 프로세스(--gpb-server)에서 수행한다.
                그 프로세스가 죽거나(단언 실패) 시간 초과·예외가 나면 해당 적합만 ridge 고정효과 + 랜덤 절편 적률 추정으로 대체하고
                작업 단위 meta 에 횟수를 기록한다(runs 열 mb_fallback).
  ppi         : PPI++. 원천 CatBoost(x25 → z) 예측 f 의 무라벨 A 풀 평균 + n 개 골드 라벨의 rectifier 평균(z − λ f).
                λ = Cov(z, f)/(Var(f)·(1 + n/N)) 을 [0, 1] 로 자른 분산 최소화 해. 95 % 신뢰구간 = θ ± 1.96·se.
채점 변형: (E) E·sqrt(TDD) 단독, (E+res) E 앵커 + 잔차 CatBoost(원천 잔차(E0 앵커) ∪ 대상 n 개 잔차(E_n 앵커), λ ∈ {0.25, 0.5}).
채점 규약(h4 공통 §3): Δ = 방법 − 비교 대상(물리식 E0 또는 shrink_k10), 음수 = 개선.
  주 CI = 분할 안 채점 블록 부트스트랩(h4_common.boot_delta_blocks, 1000회, 방법·반복·seed 에 같은 재표집 인덱스),
  분할 승률(분할별 Δ < 0 비율)·반복 승률(반복별 Δ < 0 비율)을 함께 낸다. 보조 'ci_rep' = (분할, 반복) 행 재표집 CI.
  최소 n(주) = 블록 CI 상한 < 0 · 분할 승률 ≥ 2/3 · 반복 승률 ≥ 0.75 인 최소 n(h4_common.min_n). 미달성은 censored.
  지역 평균(AB4·ALL) = 대상별 블록 부트스트랩 분포의 같은 번호끼리 평균(대상 층화, 대상마다 다른 seed).

출력(data/processed/h4/)
  b2_runs.csv        행 = (대상, 분할, 추정기, 변형, n, 규칙, 반복, seed, λ) 의 RMSE·E_used (allA 는 n = -1, 실제 라벨 수는 n_lab)
  b2_blocksse.npz    h4_common.save_stores 형식. 키 = (est, variant, rule, scope, n, rep, seed, lam)
  b2_curve.csv       (대상, 추정기, 변형, λ, 규칙, 범위, n) 별 Δ 대 물리식·대 shrink_k10 의 블록 CI·분할/반복 승률·ci_rep
  b2_minn.csv        (대상, 추정기, 변형, λ, 규칙) 별 물리식 대비 최소 n(주 정의)·censored·n_max
  b2_pooling.csv     n ∈ {3, 5, 10, allA} 행과 구조/수준 구분
  b2_f5.csv          F5 판정량: (추정기, 변형, λ, 규칙) 별 캐나다·CA-3 allA Δ(대 물리식)·수준 지역 n ∈ {3, 5, 10} 대 shrink CI 하한 ≤ 0 여부
                     + allA w_offset·κ 가중·E_offset/E_ls·gm_ls_ratio, 보조 구조 판정 n ∈ {10, 20, 40}, struct_interp(추정 대상 차이 여부)
  유효 블록 수: curve·minn·pooling·f5 의 n_eff_blocks_min = 분할별 (Σn)²/Σn² 의 최솟값. 5 미만이면 ci_flag 에 'eff_blocks<5'.
  b2_regional.csv    대상 층화 지역 평균(AB4·ALL)
  b2_ppi_ci.csv      PPI++ θ̂·λ·se·95 % CI 와 오프셋 사후(대상·분할·n·규칙·반복)
  b2_theory.csv      대상별 σ²·τ²(셀 ≥ 3·≥ 30)·κ_eff·n_theory(ε ∈ {0.05, 0.1})·E_own·|log(E_own/E0)|·구조/수준
  b2_theory_corr.csv n_theory 대 경험 손익분기 n 의 절단 반영 Spearman·Kendall(F6). primary 열이 사전 지정 주 비교
  b2_meta.json       인자·git commit·경과 시간·행 수·대상별 사전·gpboost 대체 통계
실행(ROOT, CPU): python3 scripts/3_deep_learning/h33_partial_pooling.py --workers 4 [--smoke] [--tag b2] [--summarize-only]
스모크: 대상 3(레나·AL-1·러시아 W)·분할 1·n {3, 20}·반복 2·CatBoost 반복 1·seed 1, 산출 b2_smoke_*.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import pickle


# ---------------------------------------------------------------- gpboost 격리 프로세스(C++ 단언 실패가 작업자를 죽이지 않도록)
def _gpb_fit_predict(gpb, X, z, groups, init, threads, preds):
    import numpy as np
    gp = gpb.GPModel(group_data=groups, likelihood="gaussian", num_parallel_threads=threads)
    # 기본 lbfgs 는 일부 원천 구성에서 Eigen 단언 실패로 프로세스가 중단된다(2026-09-26 확인). gradient_descent + 적률 초기값으로 고정.
    opt = dict(optimizer_cov="gradient_descent")
    if init is not None and all(v is not None and v == v and v > 0 for v in init):
        opt["init_cov_pars"] = np.array([float(init[0]), float(init[1])])
    gp.set_optim_params(params=opt)
    bst = gpb.train(params=dict(objective="regression_l2", learning_rate=0.05, max_depth=3, num_leaves=8, min_data_in_leaf=20,
                                verbose=-1, num_threads=threads), train_set=gpb.Dataset(X, z), gp_model=gp, num_boost_round=200)
    cp = gp.get_cov_pars()
    out = {}
    for name, (Xp, gp_) in preds.items():
        p = bst.predict(data=Xp, group_data_pred=gp_, pred_latent=True)
        out[name] = np.asarray(p["fixed_effect"], float) + np.asarray(p["random_effect_mean"], float)
    return dict(preds=out, sigma2=float(cp.iloc[0, 0]), tau2=float(cp.iloc[0, 1]))


def _gpb_server_main():
    """표준 입력으로 pickle 요청을 받고, 원래 표준 출력 fd 로 pickle 응답을 보낸다. C 수준 출력은 표준 오류로 돌린다."""
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    import warnings as _w
    _w.filterwarnings("ignore")
    import gpboost as gpb
    inp = sys.stdin.buffer
    while True:
        try:
            req = pickle.load(inp)
        except EOFError:
            return
        try:
            res = _gpb_fit_predict(gpb, **req); res["ok"] = True
        except Exception as e:                                                                        # noqa: BLE001
            res = dict(ok=False, err=repr(e)[:300])
        pickle.dump(res, out, protocol=4); out.flush()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--gpb-server":
    _gpb_server_main()
    sys.exit(0)

import argparse                                                                                       # noqa: E402
import json                                                                                           # noqa: E402
import multiprocessing                                                                                # noqa: E402
import select as _select                                                                              # noqa: E402
import subprocess                                                                                     # noqa: E402
import time                                                                                           # noqa: E402
import warnings                                                                                       # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed                                      # noqa: E402
from pathlib import Path                                                                              # noqa: E402

import numpy as np                                                                                    # noqa: E402
import pandas as pd                                                                                   # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
from polar.h4_common import (BlockStore, save_stores, load_stores, stores_for_target, boot_delta_blocks,  # noqa: E402
                             rep_boot_ci, min_n, spearman_censored, kendall_censored, offset_mle_prior,
                             offset_mle_estimate, n_theory, seed_of, SPLIT_WIN_MIN, REP_WIN_MIN)

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역 전부)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40,80,160,320")
ap.add_argument("--reps", type=int, default=10, help="해석적 추정기(shrink·offset·ppi·mixed_boost E 단독) 반복")
ap.add_argument("--reps-cb", type=int, default=3, help="잔차 CatBoost 반복(앞에서부터)")
ap.add_argument("--seeds", type=int, default=2, help="잔차 CatBoost seed 수")
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--rules", default="random,kmedoid")
ap.add_argument("--eps", default="0.05,0.1")
ap.add_argument("--min-region-cells", type=int, default=3, help="계층 지역으로 인정할 원천 macro 지역의 최소 셀 수(주)")
ap.add_argument("--extra-offset-min-cells", type=int, default=30, help="민감도 offset_mle30 의 최소 셀 수(0 이면 생략)")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--gpb-timeout", type=float, default=900.0, help="gpboost 적합 1회 시간 한도(초). 넘으면 대체 추정")
ap.add_argument("--breakeven", default="", help="F6 경험 손익분기 CSV(기본: h3/h25b_breakeven.csv 가 있으면 그것, 없으면 h3/h25_breakeven.csv)")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="b2")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs·blocksse·ppi_ci 로 집계만 다시")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True, parents=True)
H3 = PROC / "h3"
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3, 20]
REPS = args.reps if not args.smoke else 2
REPS_CB = args.reps_cb if not args.smoke else 1
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
LAMS = [float(v) for v in args.lams.split(",")]
EPS = [float(v) for v in args.eps.split(",")]
RULES = args.rules.split(",")
MC30 = args.extra_offset_min_cells
ESTS = ["shrink_k10", "offset_mle", "mixed_boost", "ppi"] + (["offset_mle30"] if MC30 > 0 else []) + ["offset_ls"]
SCALAR_ESTS = [e for e in ESTS if e != "mixed_boost"]
RES_ESTS = [e for e in SCALAR_ESTS if e != "offset_ls"]           # 잔차 CatBoost(E+res) 를 붙이는 스칼라 추정기(offset_ls 는 E 단독 대조)
ESTIMAND_ESTS = {"offset_mle", "offset_mle30", "ppi"}             # 기하평균 척도 스칼라 추정기(allA 구조 판정 = 추정 대상 차이)
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
STRUCT_F5 = ["Canada", "CA-3"]                      # F5 의 구조 지역(계획서 §4 B2)
STRUCT_THR = 0.15                                   # 구조/수준 구분 |log(E_own/E0)| 문턱(계획서 F1)
TAG = args.tag + ("_smoke" if args.smoke else "")
KEY_COLS = ["est", "variant", "rule", "scope", "n", "rep", "seed", "lam"]      # BlockStore 키 순서
GRP_COLS = ["est", "variant", "rule", "scope", "n", "lam"]                     # 곡선 한 행 = 키에서 rep·seed 를 뺀 것
PHYS_KEY = ("physics", "E", "none", "n0", 0, -1, -1, 0.0)

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    DF["z"] = np.where((DF.y.values > 0) & (DF.s.values > 0), np.log(DF.y.values) - np.log(DF.s.values), np.nan)


# ---------------------------------------------------------------- 하위 지역(h25 와 동일 정의: 블록 중심 좌표 k-means, 라벨 미사용)
def make_subregions(df):
    from sklearn.cluster import KMeans
    sub = np.array(["" for _ in range(len(df))], dtype=object)
    rows = []
    for spec in args.k_sub.split(","):
        reg, k = spec.split(":"); k = int(k)
        idx = np.where(df.macro.values == reg)[0]
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        for i in idx:
            sub[i] = b2s[df.block.values[i]]
        for c in range(k):
            m = lab == c
            rows.append(dict(subregion=name_of[c], parent=reg, n_blocks=int(m.sum()), n_cells=int(bt.n.values[m].sum())))
    return sub, pd.DataFrame(rows)


DF["sub"], SUBS = make_subregions(DF)
if (H3 / "subregions.csv").exists():                          # h25 정의와 셀 수 대조(불일치 시 중단)
    ref = pd.read_csv(H3 / "subregions.csv").set_index("subregion").n_cells
    chk = SUBS.set_index("subregion").n_cells
    assert all(int(ref.get(k, -1)) == int(v) for k, v in chk.items()), "하위 지역 정의가 h3/subregions.csv 와 다름"
TARGETS = args.targets.split(",") if args.targets else MAIN4 + sorted(SUBS.subregion.tolist())
if args.smoke:
    TARGETS = ["Lena", "AL-1", "Russia_W"]


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def parent_of(t):
    return t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte, seed, w=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def select(rule, n, rng, Z):
    if rule == "random":
        return rng.choice(len(Z), n, replace=False)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


# ---------------------------------------------------------------- 원천·계층 사전(대상 라벨 미사용)
def source_idx_of(t_idx):
    src_idx = np.where(np.isfinite(DF.y.values))[0]
    src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:
        la, lo = DF.lat.values, DF.lon.values
        keep = np.ones(len(src_idx), bool); tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    return src_idx


def hier_prior(src, E0, min_cells):
    """h4_common.offset_mle_prior(사전 평균 = log 원천 최소제곱 E0) + 계층 지역 마스크. 지역 < 2 이면 None."""
    try:
        pri = offset_mle_prior(src, region_col="macro", min_cells=min_cells, y_col="y", s_col="s", logE0=float(np.log(E0)))
    except ValueError:
        return None, np.zeros(len(src), bool)
    pri = dict(pri); pri["kappa_eff"] = pri["sigma2"] / pri["tau2"]
    pri["logE0_regmean"] = float(np.mean(list(pri["region_means"].values())))       # 참고: 지역 평균의 비가중 평균
    pri["hier_regions"] = ",".join(f"{r}:{pri['region_n'][r]}" for r in pri["regions"])
    return pri, np.isin(src.macro.values, pri["regions"])


def _pri_flat(pri, suffix=""):
    keys = ("sigma2", "tau2", "tau2_raw", "kappa_eff", "logE0", "logE0_regmean", "n_regions", "hier_regions")
    return {f"{k}{suffix}": (pri.get(k, np.nan) if pri else np.nan) for k in keys}


def target_diag(t):
    """대상 진단(대상 전체 라벨 사용, 이론표·F5 구조/수준 구분 전용. 추정기에는 쓰지 않음)."""
    t_idx = target_idx(t); src_idx = source_idx_of(t_idx); src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    pri, _ = hier_prior(src, E0, args.min_region_cells)
    pri30 = hier_prior(src, E0, MC30)[0] if MC30 > 0 else None
    d = DF.iloc[t_idx]; E_own = ls_E(d.y.values, d.s.values); z = d.z.values[np.isfinite(d.z.values)]
    alr = float(abs(np.log(E_own / E0))) if E_own > 0 and E0 > 0 else np.nan
    out = dict(target=t, parent=parent_of(t), n_cells=int(len(t_idx)), n_blocks=int(d.block.nunique()), n_src=int(len(src_idx)), E0=E0, E_own=E_own,
               abs_logE_ratio=alr, struct_class=("structural" if alr < STRUCT_THR else "level") if np.isfinite(alr) else "",
               zbar_target=float(z.mean()) if len(z) else np.nan, sigma2_target=float(z.var(ddof=1)) if len(z) >= 2 else np.nan,
               **_pri_flat(pri), **(_pri_flat(pri30, f"_r{MC30}") if MC30 > 0 else {}))
    for e in EPS:
        out[f"n_theory_prior_eps{e}"] = n_theory(pri, e) if pri else np.nan
        if pri30:
            out[f"n_theory_prior{MC30}_eps{e}"] = n_theory(pri30, e)
        out[f"n_theory_target_eps{e}"] = out["sigma2_target"] / (pri["tau2"] * e) if pri else np.nan
        out[f"n_theory_shift_eps{e}"] = pri["kappa_eff"] * max(alr / e - 1.0, 0.0) if pri and np.isfinite(alr) else np.nan
    return out


# ---------------------------------------------------------------- 추정기
def ppi_pp(z_n, f_lab, f_unl):
    """PPI++ 평균 추정: θ̂ = λ·mean(f_unl) + mean(z_n − λ f_lab). λ = Cov(z,f)/(Var(f)(1 + n/N)) ∈ [0, 1]."""
    m = np.isfinite(z_n) & np.isfinite(f_lab); z_n, f_lab = z_n[m], f_lab[m]
    f_unl = f_unl[np.isfinite(f_unl)]; n, N = len(z_n), len(f_unl)
    if n == 0:
        return dict(theta=np.nan, lam=np.nan, se=np.nan, lo=np.nan, hi=np.nan, n_lab=0, N_unl=N)
    if N == 0:
        lam = 0.0
    elif n >= 2:
        vf = float(np.var(f_lab, ddof=1)); cv = float(np.cov(z_n, f_lab, ddof=1)[0, 1])
        lam = 1.0 if vf < 1e-12 else float(np.clip(cv / (vf * (1.0 + n / N)), 0.0, 1.0))
    else:
        lam = 1.0
    rect = z_n - lam * f_lab
    theta = (lam * f_unl.mean() if N else 0.0) + rect.mean()
    v_unl = float(np.var(f_unl, ddof=1)) / N if N >= 2 else 0.0
    v_rect = float(np.var(rect, ddof=1)) / n if n >= 2 else np.nan
    se = float(np.sqrt(lam ** 2 * v_unl + v_rect)) if np.isfinite(v_rect) else np.nan
    return dict(theta=float(theta), lam=lam, se=se, lo=float(theta - 1.96 * se) if np.isfinite(se) else np.nan,
                hi=float(theta + 1.96 * se) if np.isfinite(se) else np.nan, n_lab=int(n), N_unl=int(N))


class GPBClient:
    """작업자 프로세스마다 하나. gpboost 적합을 자식 파이썬 프로세스(--gpb-server)에 맡기고, 죽거나 멈추면 재기동한다."""

    def __init__(self, threads, timeout):
        self.threads = threads; self.timeout = timeout; self.p = None
        self.stats = dict(calls=0, ok=0, crash=0, timeout=0, error=0, restarts=0); self.last_err = ""

    def _start(self):
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads))
        self.p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--gpb-server"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env, cwd=str(ROOT))
        self.stats["restarts"] += 1

    def _kill(self):
        try:
            self.p.kill(); self.p.wait(timeout=10)
        except Exception:                                                                             # noqa: BLE001
            pass
        self.p = None

    def call(self, req):
        self.stats["calls"] += 1
        if self.p is None or self.p.poll() is not None:
            self._start()
        try:
            pickle.dump(dict(req, threads=self.threads), self.p.stdin, protocol=4); self.p.stdin.flush()
            ready, _, _ = _select.select([self.p.stdout], [], [], self.timeout)
            if not ready:
                self.stats["timeout"] += 1; self.last_err = "timeout"; self._kill(); return None
            res = pickle.load(self.p.stdout)
        except (EOFError, BrokenPipeError, OSError, pickle.UnpicklingError) as e:
            rc = self.p.poll() if self.p is not None else None
            self.stats["crash"] += 1; self.last_err = f"crash rc={rc} {repr(e)[:80]}"; self._kill(); return None
        if not res.get("ok"):
            self.stats["error"] += 1; self.last_err = res.get("err", ""); return None
        self.stats["ok"] += 1
        return res

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.close(); self.p.wait(timeout=30)
            except Exception:                                                                         # noqa: BLE001
                self._kill()
            self.p = None


_GPB = None


def gpb_client():
    global _GPB
    if _GPB is None:
        _GPB = GPBClient(args.threads, args.gpb_timeout)
    return _GPB


def _ridge_mixed(X, z, groups, pred_sets):
    """대체 추정: ridge 고정효과 + 그룹 랜덤 절편(적률 σ²·τ², 수축 b = n·ē/(n + σ²/τ²)). 신규 그룹 b = 0."""
    from sklearn.linear_model import Ridge
    Xf = X.astype(float); med = np.nanmedian(Xf, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isnan(Xf), med, Xf); mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    rd = Ridge(alpha=10.0).fit((Xf - mu) / sd, z)
    e = z - rd.predict((Xf - mu) / sd)
    g = pd.DataFrame(dict(g=groups, e=e)).groupby("g").e; n_r, m_r, v_r = g.size(), g.mean(), g.var(ddof=1).fillna(0.0)
    sigma2 = float(((n_r - 1) * v_r).sum() / max((n_r - 1).sum(), 1))
    tau2 = float(max(m_r.var(ddof=1) - (sigma2 / n_r).mean(), 1e-4)) if len(m_r) >= 2 else 1e-4
    b = {r: float(n_r[r] * m_r[r] / (n_r[r] + sigma2 / tau2)) for r in n_r.index}
    out = {}
    for name, (Xp, gp) in pred_sets.items():
        Xq = np.where(np.isnan(Xp.astype(float)), med, Xp.astype(float))
        out[name] = rd.predict((Xq - mu) / sd) + np.array([b.get(v, 0.0) for v in gp])
    return out, sigma2, tau2


def mixed_boost(X, z, groups, init, pred_sets):
    """혼합효과 부스팅 적합 후 pred_sets{이름: (X, 그룹)} 의 잠재 예측(고정 + 랜덤 절편 평균)을 반환. (preds, info)."""
    ok = np.isfinite(z); X, z, groups = X[ok], z[ok], np.asarray(groups, dtype=object)[ok]
    ps = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], dtype=object).astype(str)) for k, v in pred_sets.items()}
    res = gpb_client().call(dict(X=X.astype(np.float32), z=z.astype(float), groups=groups.astype(str),
                                 init=[float(init[0]), float(init[1])], preds=ps))
    if res is not None:
        return {k: np.asarray(v, float) for k, v in res["preds"].items()}, dict(fallback=False, sigma2=res["sigma2"], tau2=res["tau2"], err="")
    preds, s2, t2 = _ridge_mixed(X, z, groups.astype(str), ps)
    return preds, dict(fallback=True, sigma2=s2, tau2=t2, err=gpb_client().last_err)


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t); parent = parent_of(t)
    src_idx = source_idx_of(t_idx); src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values); logE0 = float(np.log(E0))
    pri, hier_m = hier_prior(src, E0, args.min_region_cells)
    pri30 = hier_prior(src, E0, MC30)[0] if MC30 > 0 else None
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values; z_src = src.z.values
    XH, zH, gH, yH, sH = X_src[hier_m], z_src[hier_m], src.macro.values[hier_m].astype(str), y_src[hier_m], s_src[hier_m]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, zA, XA = A.y.values, A.s.values, A.z.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    blocksB = B.block.values.astype(str)
    _, codesB = np.unique(blocksB, return_inverse=True); nbB = int(codesB.max()) + 1 if len(B) else 0
    meta = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), n_src_hier=int(hier_m.sum()), E0=E0, n_fit=0, mb_fits=0, mb_fallback_fits=0,
                mb_sigma2_n0=np.nan, mb_tau2_n0=np.nan, mb_last_err="", **_pri_flat(pri), **(_pri_flat(pri30, f"_r{MC30}") if MC30 > 0 else {}))
    if len(B) == 0 or nA < 2 or pri is None:                    # 채점 셀 없음·라벨 풀 부족·계층 지역 부족: 빈 결과
        meta["elapsed_s"] = round(time.time() - t0, 1)
        return [], [], meta, None
    st = BlockStore(t, sp, blocksB, meta=dict(n_A=nA, n_eval=len(evB), E0=E0))
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    gT = np.array([t] * len(B), dtype=object)
    init = (pri["sigma2"], pri["tau2"])
    # PPI 원천 모형 f: 원천 z 에 CatBoost(x25), seed 0. 대상 라벨 미사용, A 공변량에만 예측
    okz = np.isfinite(z_src)
    fA = cb_fit_predict(X_src[okz], z_src[okz], XA, 0)
    rows, ppi_rows = [], []
    n_fit = 1
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=meta["n_blocks_A"], n_eval=len(evB), n_blocks_eval=nbB,
                n_src=meta["n_src"], n_src_hier=meta["n_src_hier"], E0=E0, sigma2=pri["sigma2"], tau2=pri["tau2"])

    def add(est, variant, n, n_lab, rep, seed, lam, rule, scope, pred, E_used, b_t=np.nan, mb_fb=False, gm=np.nan):
        rows.append(dict(**base, est=est, variant=variant, scope=scope, rule=rule, n=int(n), n_lab=int(n_lab), rep=int(rep), seed=int(seed), lam=float(lam),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)),
                         E_used=float(E_used), b_target=float(b_t), mb_fallback=bool(mb_fb), gm_ls_ratio=float(gm)))
        st.add((est, variant, rule, scope, int(n), int(rep), int(seed), float(lam)), yB, pred)

    def residual_pred(anchor_src, anchor_sel, XS, yS, sel, seed):
        """원천 잔차(앵커 anchor_src) ∪ 대상 n 개 잔차(앵커 anchor_sel) 로 CatBoost 를 적합, B 에서 g 를 반환."""
        r_s = yS - anchor_src; okr = np.isfinite(r_s)
        if sel is None or len(sel) == 0:
            return cb_fit_predict(XS[okr], r_s[okr], XB, seed)
        r_n = yA[sel] - anchor_sel; okn = np.isfinite(r_n)
        return cb_fit_predict(np.vstack([XS[okr], XA[sel][okn]]), np.concatenate([r_s[okr], r_n[okn]]), XB, seed)

    # n = 0: 물리식, 원천 잔차만(E0 앵커), mixed_boost(b = 0)
    phys = E0 * sB
    add("physics", "E", 0, 0, -1, -1, 0.0, "none", "n0", phys, E0)
    for seed in SEEDS:
        g0 = residual_pred(E0 * s_src, None, X_src, y_src, None, seed); n_fit += 1
        for lam in LAMS:
            add("shrink_k10", "E+res", 0, 0, -1, seed, lam, "none", "n0", phys + lam * g0, E0)
    P0, info0 = mixed_boost(XH, zH, gH, init, dict(B=(XB, gT), H=(XH, gH)))
    n_fit += 1; meta["mb_fits"] += 1; meta["mb_fallback_fits"] += int(info0["fallback"])
    meta["mb_sigma2_n0"], meta["mb_tau2_n0"] = info0["sigma2"], info0["tau2"]
    pB0 = np.exp(P0["B"]) * sB; E_mb0 = float(np.exp(np.mean(P0["B"])))
    add("mixed_boost", "E", 0, 0, -1, -1, 0.0, "none", "n0", pB0, E_mb0, 0.0, info0["fallback"])
    aS0 = np.exp(P0["H"]) * sH
    for seed in SEEDS:
        g = residual_pred(aS0, None, XH, yH, None, seed); n_fit += 1
        for lam in LAMS:
            add("mixed_boost", "E+res", 0, 0, -1, seed, lam, "none", "n0", pB0 + lam * g, E_mb0, 0.0, info0["fallback"])

    def evaluate(sel, n_key, rep, rule, scope, do_cb):
        nonlocal n_fit
        n = len(sel)
        E_ls = ls_E(yA[sel], sA[sel])
        E = {"shrink_k10": (n * E_ls + args.kappa * E0) / (n + args.kappa)}
        po = offset_mle_estimate(zA[sel], pri); E["offset_mle"] = po["E"]
        po30 = offset_mle_estimate(zA[sel], pri30) if pri30 is not None else None
        if po30 is not None:
            E["offset_mle30"] = po30["E"]
        zs = zA[sel][np.isfinite(zA[sel])]
        gm = float(np.exp(zs.mean()) / E_ls) if len(zs) and np.isfinite(E_ls) and E_ls > 0 else np.nan   # 기하평균 E / LS E
        w = float(po["w"])                                      # offset_mle 와 같은 가중을 LS 척도에 적용(추정 대상 대조)
        E["offset_ls"] = float(np.exp((1.0 - w) * logE0 + w * np.log(E_ls))) if np.isfinite(E_ls) and E_ls > 0 else E0
        unl = np.setdiff1d(np.arange(nA), sel)                  # 무라벨 풀: 선택되지 않은 A 셀의 f(공변량 예측)만 사용
        pp = ppi_pp(zA[sel], fA[sel], fA[unl]); E["ppi"] = float(np.exp(pp["theta"])) if np.isfinite(pp["theta"]) else E0
        ppi_rows.append(dict(target=t, parent=parent, split=sp, scope=scope, rule=rule, n=int(n_key), rep=int(rep), E0=E0, logE0=logE0, **pp,
                             E_ppi=E["ppi"], E_ls=E_ls, E_shrink=E["shrink_k10"], E_offset=po["E"], logE_offset=po["logE_post"], logE_offset_sd=po["logE_sd"],
                             w_offset=po["w"], E_offset30=po30["E"] if po30 else np.nan, w_offset30=po30["w"] if po30 else np.nan,
                             w_kappa=n / (n + args.kappa), E_offset_ls=E["offset_ls"], gm_ls_ratio=gm,
                             E_offset_over_ls=po["E"] / E_ls if np.isfinite(E_ls) and E_ls > 0 else np.nan))
        for est in SCALAR_ESTS:
            add(est, "E", n_key, n, rep, -1, 0.0, rule, scope, E[est] * sB, E[est], gm=gm)
        # 혼합효과 부스팅: 학습 행 = 원천 계층 지역 + 대상 선택 라벨 n 개(그룹 = 대상)
        gsel = np.array([t] * n, dtype=object)
        ps = dict(B=(XB, gT), sel=(XA[sel], gsel), sel_new=(XA[sel], np.array(["__new__"] * n, dtype=object)))
        if do_cb:
            ps["H"] = (XH, gH)
        P, info = mixed_boost(np.vstack([XH, XA[sel]]), np.concatenate([zH, zA[sel]]), np.concatenate([gH.astype(object), gsel]), init, ps)
        n_fit += 1; meta["mb_fits"] += 1; meta["mb_fallback_fits"] += int(info["fallback"])
        if info["fallback"]:
            meta["mb_last_err"] = info["err"]
        pB = np.exp(P["B"]) * sB
        b_t = float(P["sel"].mean() - P["sel_new"].mean())     # 대상 랜덤 절편(신규 그룹 대비 차)
        E_mb = float(np.exp(np.mean(P["B"])))
        add("mixed_boost", "E", n_key, n, rep, -1, 0.0, rule, scope, pB, E_mb, b_t, info["fallback"], gm=gm)
        if not do_cb:
            return
        aS = np.exp(P["H"]) * sH; a_sel = np.exp(P["sel"]) * sA[sel]
        for seed in SEEDS:
            for est in RES_ESTS:
                g = residual_pred(E0 * s_src, E[est] * sA[sel], X_src, y_src, sel, seed); n_fit += 1
                for lam in LAMS:
                    add(est, "E+res", n_key, n, rep, seed, lam, rule, scope, E[est] * sB + lam * g, E[est], gm=gm)
            g = residual_pred(aS, a_sel, XH, yH, sel, seed); n_fit += 1
            for lam in LAMS:
                add("mixed_boost", "E+res", n_key, n, rep, seed, lam, rule, scope, pB + lam * g, E_mb, b_t, info["fallback"], gm=gm)

    evaluate(np.arange(nA), -1, -1, "all", "allA", do_cb=True)     # 라벨 전량: 분할마다 |A| 가 달라 n = -1 로 통일(n_lab 에 실제 수)
    for n in N_GRID:
        if n >= nA:
            continue
        for rule in RULES:
            for rep in range(REPS):
                rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * RULES.index(rule) + rep)
                sel = select(rule, n, rng, Z)
                evaluate(sel, n, rep, rule, "n", do_cb=rep < REPS_CB)
    meta.update(n_fit=n_fit, gpb_worker_stats=json.dumps(gpb_client().stats), elapsed_s=round(time.time() - t0, 1))   # 작업자 누적(작업 단위 아님)
    return rows, ppi_rows, meta, st


# ---------------------------------------------------------------- 집계
def _grp_of(k):
    return (k[0], k[1], k[2], k[3], int(k[4]), float(k[7]))


def _shrink_ref(grp):
    est, variant, rule, scope, n, lam = grp
    if est == "shrink_k10":
        return None
    if scope == "n0" and variant == "E":
        return None                                            # n = 0 의 E 단독 shrink 는 물리식과 같다(Δ 대 물리식으로 대신)
    return lambda k: k[0] == "shrink_k10" and (k[1], k[2], k[3], int(k[4]), float(k[7])) == (variant, rule, scope, n, lam)


def block_stats(stores):
    """대상별 곡선 그룹마다 물리식·shrink_k10 대비 블록 부트스트랩 Δ. (행 DataFrame, {(grp, vs): {target: dist}})."""
    out, dists = [], {}
    for t in sorted({t for t, _ in stores}):
        by_split = stores_for_target(stores, t)
        groups: dict = {}
        for sp, st_ in by_split.items():
            for k in st_.keys:
                if k[0] != "physics":
                    groups.setdefault(_grp_of(k), set()).add(k)
        bseed = seed_of("b2", t)                                 # 대상마다 다른 재표집(지역 평균 층화), 대상 안에서는 모든 비교가 공유
        # 유효 블록 수 (Σn)²/Σn² (분할별 채점 셀 수 기준). 블록 크기 불균형이 크면 명목 nb 보다 훨씬 작다.
        neff = {sp: float(st_.ncell.sum() ** 2 / max((st_.ncell.astype(float) ** 2).sum(), 1.0)) for sp, st_ in by_split.items()}
        neff_min = min(neff.values()) if neff else np.nan
        neff_json = json.dumps({str(a): round(b, 2) for a, b in sorted(neff.items())})
        for grp, keys in groups.items():
            row = dict(target=t, **dict(zip(GRP_COLS, grp)))
            for vs, ref in (("phys", [PHYS_KEY]), ("shrink", _shrink_ref(grp))):
                if ref is None:
                    continue
                r = boot_delta_blocks(by_split, list(keys), ref, nboot=args.nboot, seed=bseed, rep_fn=lambda k: k[5], return_dist=True)
                if r["n_splits"] == 0:
                    continue
                row.update({f"d_{vs}_blk": r["delta"], f"d_{vs}_lo": r["ci_lo"], f"d_{vs}_hi": r["ci_hi"], f"p_{vs}": r["p_boot"],
                            f"split_win_{vs}": r["split_win"], f"rep_win_{vs}": r["rep_win"], f"d_{vs}_beq": r["delta_beq"],
                            f"d_{vs}_beq_lo": r["ci_lo_beq"], f"d_{vs}_beq_hi": r["ci_hi_beq"], f"ref_{vs}_blk": r["rmse_B"],
                            f"d_{vs}_splits": json.dumps({str(a): round(b, 4) for a, b in r["delta_split"].items()})})
                if vs == "phys":
                    flag = [f for f in str(r["ci_flag"] or "").split(";") if f]
                    if np.isfinite(neff_min) and neff_min < 5:
                        flag.append("eff_blocks<5")
                    row.update(rmse_blk=r["rmse_A"], n_splits_blk=r["n_splits"], n_blocks_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0,
                               n_eff_blocks_min=neff_min, n_eff_blocks=neff_json, ci_flag=";".join(flag))
                if "dist" in r:
                    dists.setdefault((grp, vs), {})[t] = (r["delta"], r["dist"])
            out.append(row)
    return pd.DataFrame(out), dists


def summarize(runs, stores):
    r = runs[runs.est != "physics"].copy()
    phys = runs[runs.est == "physics"].set_index(["target", "split"]).rmse_cm
    r["phys"] = [phys.loc[(a, b)] for a, b in zip(r.target, r.split)]
    r["d_phys"] = r.rmse_cm - r.phys; r["win"] = (r.rmse_cm < r.phys).astype(float)
    if "gm_ls_ratio" not in r:
        r["gm_ls_ratio"] = np.nan
    key = ["target", "parent", "split", "est", "variant", "rule", "scope", "lam", "n", "rep"]
    g = r.groupby(key, as_index=False).agg(rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"),
                                           win=("win", "mean"), E_used=("E_used", "mean"), bias_cm=("bias_cm", "mean"), n_lab=("n_lab", "mean"),
                                           mb_fallback=("mb_fallback", "mean"), gm_ls_ratio=("gm_ls_ratio", "mean"))
    # 보조 Δ 대 shrink_k10(같은 분할·변형·규칙·범위·λ·n·반복, seed 는 위에서 평균)
    ref = g[g.est == "shrink_k10"].set_index(["target", "split", "variant", "rule", "scope", "lam", "n", "rep"]).rmse_cm
    idx = list(zip(g.target, g.split, g.variant, g.rule, g.scope, g.lam, g.n, g.rep))
    g["d_shrink"] = g.rmse_cm.values - np.array([ref.get(i, np.nan) for i in idx], float)
    g.loc[g.est == "shrink_k10", "d_shrink"] = np.nan
    ck = ["target", "parent", "est", "variant", "rule", "scope", "lam", "n"]
    curve = g.groupby(ck, as_index=False).agg(rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_beq_mean=("rmse_beq_cm", "mean"),
                                              d_phys_mean=("d_phys", "mean"), d_shrink_mean=("d_shrink", "mean"), win_rate=("win", "mean"),
                                              E_used_mean=("E_used", "mean"), bias_mean=("bias_cm", "mean"), n_lab_mean=("n_lab", "mean"),
                                              mb_fallback_rate=("mb_fallback", "mean"), gm_ls_ratio_mean=("gm_ls_ratio", "mean"),
                                              n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"))
    rep_ci = {kk: (rep_boot_ci(sub.d_phys.values), rep_boot_ci(sub.d_shrink.values)) for kk, sub in g.groupby(ck)}
    tk = [tuple(v) for v in curve[ck].itertuples(index=False)]
    curve["ci_rep_phys_lo"] = [rep_ci[k][0][0] for k in tk]; curve["ci_rep_phys_hi"] = [rep_ci[k][0][1] for k in tk]
    curve["ci_rep_shrink_lo"] = [rep_ci[k][1][0] for k in tk]; curve["ci_rep_shrink_hi"] = [rep_ci[k][1][1] for k in tk]
    bs, dists = block_stats(stores)
    curve = curve.merge(bs, on=["target"] + GRP_COLS, how="left", validate="one_to_one")
    return curve, dists


def minn_table(curve):
    """(대상, 추정기, 변형, λ, 규칙) 별 물리식 대비 최소 n(주 정의). 분할 일부에만 있는 n 은 제외(h25 규약)."""
    rows = []
    cn = curve[(curve.scope == "n") & curve.rule.isin(RULES)]
    for (t, est, variant, lam, rule), sub in cn.groupby(["target", "est", "variant", "lam", "rule"]):
        sub = sub[sub.n_splits == sub.n_splits.max()].sort_values("n")
        mn = min_n(sub, n_col="n", ci_hi_col="d_phys_hi", split_win_col="split_win_phys", rep_win_col="rep_win_phys")
        best = sub.loc[sub.d_phys_blk.idxmin()] if sub.d_phys_blk.notna().any() else None
        flags = sorted({f for v in sub.ci_flag.fillna("").astype(str) for f in v.split(";") if f}) if "ci_flag" in sub else []
        rows.append(dict(target=t, est=est, variant=variant, lam=lam, rule=rule, **mn,
                         best_n=int(best.n) if best is not None else -1, best_d_phys=float(best.d_phys_blk) if best is not None else np.nan,
                         n_eff_blocks_min=float(sub.n_eff_blocks_min.min()) if "n_eff_blocks_min" in sub else np.nan, ci_flag=";".join(flags)))
    return pd.DataFrame(rows)


def pooling_table(curve, theory):
    """n ∈ {3, 5, 10} 과 allA 행(대 물리식·대 shrink 블록 CI), 구조/수준 구분 부착."""
    c = curve[(curve.scope == "allA") | ((curve.scope == "n") & curve.n.isin([3, 5, 10]))].copy()
    c["n_label"] = np.where(c.scope == "allA", "allA", c.n.astype(str))
    c = c.merge(theory[["target", "abs_logE_ratio", "struct_class"]], on="target", how="left")
    c["worse_than_phys"] = c.d_phys_blk > 0
    c["ci_excl0_phys"] = (c.d_phys_hi < 0) | (c.d_phys_lo > 0)
    c["not_worse_than_shrink"] = c.d_shrink_lo <= 0                  # κ=10 대비 CI 가 0 을 포함하거나 개선
    cols = ["target", "parent", "struct_class", "abs_logE_ratio", "est", "variant", "lam", "rule", "n_label", "n", "n_lab_mean", "rmse_mean",
            "d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win_phys", "rep_win_phys", "d_shrink_blk", "d_shrink_lo", "d_shrink_hi",
            "split_win_shrink", "rep_win_shrink", "ci_rep_phys_lo", "ci_rep_phys_hi", "worse_than_phys", "ci_excl0_phys", "not_worse_than_shrink",
            "gm_ls_ratio_mean", "n_eff_blocks_min", "ci_flag", "n_runs", "n_splits"]
    return c[[x for x in cols if x in c.columns]].sort_values(["est", "variant", "lam", "rule", "n", "target"])


F5_MID_N = [10, 20, 40]                             # 보조 구조 판정 n(풀링 가중이 1 보다 작은 구간)


def f5_table(pool, curve, ppi):
    """F5 판정량. 추정기 e(≠ shrink_k10)·변형·λ·규칙마다:
       struct: 캐나다·CA-3 allA 의 Δ(e − 물리식) ≤ 0 (둘 다 있어야 판정), 참고로 shrink_k10 allA Δ 도 기록.
               allA 에서는 풀링 가중이 거의 1(κ 가중 n/(n+κ) ≈ 0.95–0.97, w_offset ≥ 0.996)이므로, 스칼라 기하평균 추정기
               (offset_mle·offset_mle30·ppi)의 struct 결과는 부분 풀링이 아니라 추정 대상 차이(LS 대 기하평균)로 해석한다
               (struct_interp = 'estimand_diff'). 이를 위해 allA 의 w_offset·κ 가중·E_offset/E_ls·gm_ls_ratio 를 함께 기록한다.
       struct_mid(보조): 같은 두 지역의 n ∈ {10, 20, 40} Δ(e − 물리식) 블록 점 추정이 전부 ≤ 0. 풀링 가중이 실제로 1 보다 작은 구간.
       level : 수준 지역(|log E비| ≥ 0.15) 전 대상 × n ∈ {3, 5, 10} 에서 Δ(e − shrink_k10) 블록 CI 하한 ≤ 0.
       F5_pass = struct 전부 충족 ∧ level 위반 0 (사전 정의 그대로). 해석은 struct_interp·struct_mid_fixed·level_eff_blocks_min 을 병기한다."""
    rows = []
    allA = pool[pool.n_label == "allA"]
    nn = pool[pool.n_label != "allA"]
    pa = ppi[ppi.scope == "allA"] if len(ppi) and "scope" in ppi else pd.DataFrame()
    cm = curve[(curve.scope == "n") & curve.n.isin(F5_MID_N)]
    for (est, variant, lam), sub_all in allA.groupby(["est", "variant", "lam"]):
        if est == "shrink_k10":
            continue
        for rule in RULES:
            d = dict(est=est, variant=variant, lam=lam, rule=rule,
                     struct_interp="estimand_diff" if est in ESTIMAND_ESTS else ("pooling_weight" if est == "offset_ls" else "pooling"))
            ok_s, ok_mid = [], []
            for tt in STRUCT_F5:
                a = sub_all[sub_all.target == tt]
                sh = allA[(allA.est == "shrink_k10") & (allA.variant == variant) & (allA.lam == lam) & (allA.target == tt)]
                v = float(a.d_phys_blk.iloc[0]) if len(a) else np.nan
                d[f"{tt}_allA_d_phys"] = v; d[f"{tt}_allA_d_phys_hi"] = float(a.d_phys_hi.iloc[0]) if len(a) else np.nan
                d[f"{tt}_allA_d_shrink"] = float(a.d_shrink_blk.iloc[0]) if len(a) else np.nan
                d[f"{tt}_shrink_allA_d_phys"] = float(sh.d_phys_blk.iloc[0]) if len(sh) else np.nan
                ok_s.append(v <= 0 if np.isfinite(v) else np.nan)
                q = pa[pa.target == tt] if len(pa) else pa
                for c_out, c_in in (("w_offset", "w_offset"), ("w_kappa", "w_kappa"), ("Eoff_over_Els", "E_offset_over_ls"), ("gm_ls_ratio", "gm_ls_ratio")):
                    d[f"{tt}_allA_{c_out}"] = float(q[c_in].mean()) if len(q) and c_in in q else np.nan
                for n_mid in F5_MID_N:
                    m = cm[(cm.target == tt) & (cm.est == est) & (cm.variant == variant) & (cm.lam == lam) & (cm.rule == rule) & (cm.n == n_mid)]
                    vm = float(m.d_phys_blk.iloc[0]) if len(m) else np.nan
                    d[f"{tt}_n{n_mid}_d_phys"] = vm
                    if np.isfinite(vm):
                        ok_mid.append(vm <= 0)
            d["struct_n"] = int(sum(np.isfinite(x) for x in ok_s)); d["struct_fixed"] = bool(all(x is True for x in ok_s)) if d["struct_n"] == len(STRUCT_F5) else np.nan
            d["struct_mid_n"] = len(ok_mid); d["struct_mid_fixed"] = bool(all(ok_mid)) if ok_mid else np.nan
            lv = nn[(nn.est == est) & (nn.variant == variant) & (nn.lam == lam) & (nn.rule == rule) & (nn.struct_class == "level")]
            lv = lv[np.isfinite(lv.d_shrink_lo)]
            d["level_checked"] = int(len(lv)); d["level_targets"] = ",".join(sorted(lv.target.unique()))
            viol = lv[lv.d_shrink_lo > 0]
            d["level_violations"] = int(len(viol)); d["level_violation_cells"] = ";".join(f"{a}@{b}" for a, b in zip(viol.target, viol.n_label))
            d["level_mean_d_shrink"] = float(lv.d_shrink_blk.mean()) if len(lv) else np.nan
            d["level_eff_blocks_min"] = float(lv.n_eff_blocks_min.min()) if len(lv) and "n_eff_blocks_min" in lv else np.nan
            d["level_eff_lt5"] = int((lv.n_eff_blocks_min < 5).sum()) if len(lv) and "n_eff_blocks_min" in lv else 0
            d["F5_pass"] = bool(d["struct_fixed"] is True and d["level_checked"] > 0 and d["level_violations"] == 0)
            d["primary"] = bool(variant == "E" and est in ("offset_mle", "mixed_boost"))
            rows.append(d)
    return pd.DataFrame(rows)


def regional_table(curve, dists, targets):
    """대상 층화 지역 평균: 대상별 블록 부트스트랩 분포를 같은 번호끼리 평균(AB4 = 주 4지역, ALL = 전 대상)."""
    out = []
    for (grp, vs), per in dists.items():
        for set_name, regs in (("AB4", [r for r in MAIN4 if r in targets]), ("ALL", list(targets))):
            have = [r for r in regs if r in per]
            if not have:
                continue
            dl = np.mean([per[r][0] for r in have]); dist = np.mean([per[r][1] for r in have], 0)
            out.append(dict(**dict(zip(GRP_COLS, grp)), vs=vs, region_set=set_name, n_targets=len(have), complete=len(have) == len(regs),
                            targets=",".join(have), delta=float(dl), ci_lo=float(np.nanpercentile(dist, 2.5)), ci_hi=float(np.nanpercentile(dist, 97.5)),
                            n_targets_improved=int(sum(per[r][0] < 0 for r in have))))
    return pd.DataFrame(out)


def breakeven_path():
    if args.breakeven:
        return Path(args.breakeven)
    for p in (H3 / "h25b_breakeven.csv", H3 / "h25_breakeven.csv"):
        if p.exists():
            return p
    return None


def theory_corr(theory, minn):
    """F6: n_theory 대 경험 손익분기 n 의 절단 반영 Spearman·Kendall. 절단 대상은 최대 순위 동률(h4_common).
       경험 출처: h25(h3 손익분기 CSV, 미달성 = 음수) · b2_self(이 실행의 shrink_k10/offset_mle E 최소 n).
       주(primary) = h25 · S1 · kmedoid · breakeven_n_ci · n_theory_prior(셀 ≥ 3). ε 는 prior 정의에서 순위에 영향이 없다."""
    corr = []
    th_cols = [c for c in theory.columns if c.startswith("n_theory_")]
    emp_sets = []
    bp = breakeven_path()
    if bp is not None and bp.exists():
        be = pd.read_csv(bp)
        be = be[(be.alpha == 1.0) & (((be.stage == "S1") & (be.lam == 0.0)) | ((be.stage == "S3") & (be.lam == 0.25)))]
        for (stage, rule), sub in be.groupby(["stage", "rule"]):
            if rule == "none":
                continue
            for bdef in ("breakeven_n_ci", "breakeven_n_rec50"):
                if bdef not in sub:
                    continue
                v = sub[bdef].values.astype(float)
                cens = sub["censored"].values.astype(bool) if (bdef == "breakeven_n_ci" and "censored" in sub) else (v < 0) | ~np.isfinite(v)
                emp_sets.append((f"h25:{bp.name}", stage, rule, bdef, dict(zip(sub.target, zip(np.where(cens, np.nan, v), cens)))))
    if len(minn):
        for (est, rule), sub in minn[(minn.variant == "E") & minn.est.isin(["shrink_k10", "offset_mle"])].groupby(["est", "rule"]):
            emp_sets.append(("b2_self", est, rule, "min_n_block", dict(zip(sub.target, zip(sub.n_star.values.astype(float), sub.censored.values.astype(bool))))))
    for src_name, stage, rule, bdef, emp in emp_sets:
        tt = theory[theory.target.isin(list(emp))]
        ev = np.array([emp[t][0] for t in tt.target], float); ec = np.array([emp[t][1] for t in tt.target], bool)
        for col in th_cols:
            th = tt[col].values.astype(float)
            rho, p, n = spearman_censored(th, ev, censored_y=ec)
            tau, pk, _ = kendall_censored(th, ev, censored_y=ec)
            keep = ~ec & np.isfinite(ev) & np.isfinite(th)
            if keep.sum() >= 3 and np.std(th[keep]) > 0 and np.std(ev[keep]) > 0:
                from scipy.stats import spearmanr
                rho_x, p_x = spearmanr(th[keep], ev[keep])
            else:
                rho_x, p_x = np.nan, np.nan
            corr.append(dict(source=src_name, stage=stage, rule=rule, breakeven_def=bdef, theory_col=col, n_targets=n, n_censored=int(ec.sum()),
                             spearman=rho, p=p, kendall=tau, p_kendall=pk, spearman_uncensored_only=float(rho_x), p_uncensored_only=float(p_x),
                             F6_pass=bool(np.isfinite(rho) and rho >= 0.6),
                             primary=bool(src_name.startswith("h25") and stage == "S1" and rule == "kmedoid" and bdef == "breakeven_n_ci"
                                          and col == f"n_theory_prior_eps{EPS[0]}")))
    return pd.DataFrame(corr)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        return "NA"


def _worker_init():
    import atexit
    atexit.register(lambda: _GPB.close() if _GPB is not None else None)


def main():
    t0 = time.time()
    tasks = [(t, sp) for t in TARGETS for sp in SPLITS]
    print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS} · 분할 {SPLITS} · n {N_GRID} · 반복 {REPS}/{REPS_CB} · seed {SEEDS} · tag {TAG}", flush=True)
    metas = []
    if args.summarize_only:
        runs = pd.read_csv(OUT / f"{TAG}_runs.csv")
        ppi = pd.read_csv(OUT / f"{TAG}_ppi_ci.csv") if (OUT / f"{TAG}_ppi_ci.csv").exists() else pd.DataFrame()
        stores = load_stores(OUT / f"{TAG}_blocksse.npz")
        if (OUT / f"{TAG}_targets.csv").exists():
            metas = pd.read_csv(OUT / f"{TAG}_targets.csv").to_dict("records")
    else:
        rows, ppi_rows, store_list = [], [], []

        def _done(r, p, m, st_):
            rows.extend(r); ppi_rows.extend(p); metas.append(m)
            if st_ is not None:
                store_list.append(st_)
            print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']} · 채점 {m['n_eval']} · 원천 {m['n_src']} · E0 {m['E0']:.3f}"
                  f" · σ² {m['sigma2']:.3f} · τ² {m['tau2']:.4f} · κ_eff {m['kappa_eff']:.2f} · gpb {m['mb_fits'] - m['mb_fallback_fits']}/{m['mb_fits']}"
                  f" · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s", flush=True)
        if args.workers <= 1:
            for t, sp in tasks:
                _done(*run_task(t, sp))
            if _GPB is not None:
                _GPB.close()
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as ex:
                futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
                for f in as_completed(futs):
                    _done(*f.result())
        runs = pd.DataFrame(rows); ppi = pd.DataFrame(ppi_rows)
        runs.to_csv(OUT / f"{TAG}_runs.csv", index=False); ppi.to_csv(OUT / f"{TAG}_ppi_ci.csv", index=False)
        pd.DataFrame(metas).to_csv(OUT / f"{TAG}_targets.csv", index=False)
        save_stores(store_list, OUT / f"{TAG}_blocksse.npz")
        stores = load_stores(OUT / f"{TAG}_blocksse.npz")          # 저장본으로 집계(재집계 경로와 같은 입력)
    curve, dists = summarize(runs, stores); curve.to_csv(OUT / f"{TAG}_curve.csv", index=False)
    theory = pd.DataFrame([target_diag(t) for t in TARGETS]); theory.to_csv(OUT / f"{TAG}_theory.csv", index=False)
    minn = minn_table(curve); minn.to_csv(OUT / f"{TAG}_minn.csv", index=False)
    pool = pooling_table(curve, theory); pool.to_csv(OUT / f"{TAG}_pooling.csv", index=False)
    f5 = f5_table(pool, curve, ppi); f5.to_csv(OUT / f"{TAG}_f5.csv", index=False)
    regional = regional_table(curve, dists, [t for t in TARGETS if t in set(curve.target)]); regional.to_csv(OUT / f"{TAG}_regional.csv", index=False)
    corr = theory_corr(theory, minn); corr.to_csv(OUT / f"{TAG}_theory_corr.csv", index=False)
    meta = dict(stage="H33/B2", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 Track B2, §5 F5·F6", targets=TARGETS, splits=SPLITS, n_grid=N_GRID, reps=REPS,
                reps_cb=REPS_CB, seeds=SEEDS, lams=LAMS, eps=EPS, kappa=args.kappa, buffer_km=args.buffer_km, k_sub=args.k_sub, rules=RULES, ests=ESTS,
                min_region_cells=args.min_region_cells, extra_offset_min_cells=MC30, nboot=args.nboot, gpb_timeout=args.gpb_timeout,
                ci_rule="h4_common.boot_delta_blocks(분할 안 채점 블록 재표집, 방법·반복·seed 공유 인덱스), ci_rep = 보조",
                min_n_rule=f"ci_hi<0 & split_win>={SPLIT_WIN_MIN:.3f} & rep_win>={REP_WIN_MIN}", struct_threshold=STRUCT_THR, struct_f5=STRUCT_F5,
                breakeven_file=str(breakeven_path()), workers=args.workers, threads=args.threads, smoke=args.smoke, summarize_only=args.summarize_only,
                git_commit=git_commit(), elapsed_s=round(time.time() - t0, 1),
                n_rows=dict(runs=int(len(runs)), curve=int(len(curve)), minn=int(len(minn)), pooling=int(len(pool)), f5=int(len(f5)), regional=int(len(regional)),
                            ppi_ci=int(len(ppi)), theory=int(len(theory)), theory_corr=int(len(corr))),
                mb_fits=int(sum(int(m.get("mb_fits", 0)) for m in metas)), mb_fallback_fits=int(sum(int(m.get("mb_fallback_fits", 0)) for m in metas)),
                mb_fallback_by_task={f"{m['target']}|{m['split']}": int(m.get("mb_fallback_fits", 0)) for m in metas},
                mb_last_err={f"{m['target']}|{m['split']}": m.get("mb_last_err", "") for m in metas if m.get("mb_last_err")},
                mixed_boost_covpars_n0={f"{m['target']}|{m['split']}": dict(sigma2=m.get("mb_sigma2_n0"), tau2=m.get("mb_tau2_n0")) for m in metas},
                prior_by_target={r.target: dict(sigma2=r.sigma2, tau2=r.tau2, tau2_raw=r.tau2_raw, kappa_eff=r.kappa_eff, logE0=r.logE0,
                                                 logE0_regmean=r.logE0_regmean, hier_regions=r.hier_regions) for r in theory.itertuples()})
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    show = pool[(pool.rule.isin(["kmedoid", "all"])) & (pool.variant == "E") & (pool.n_label.isin(["3", "10", "allA"]))]
    with pd.option_context("display.width", 250):
        print(show[["target", "est", "n_label", "rmse_mean", "d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win_phys", "rep_win_phys",
                    "d_shrink_blk", "d_shrink_lo", "d_shrink_hi"]].to_string(index=False))
        print(f5[f5.primary].to_string(index=False))
        print(theory[["target", "struct_class", "E0", "E_own", "abs_logE_ratio", "sigma2", "tau2", "kappa_eff", f"n_theory_prior_eps{EPS[0]}"]].to_string(index=False))
        if len(corr):
            print(corr[corr.primary | (corr.source == "b2_self")].head(12).to_string(index=False))
    print(f"saved {TAG}_* · runs {len(runs)} · curve {len(curve)} · gpb 대체 {meta['mb_fallback_fits']}/{meta['mb_fits']} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
