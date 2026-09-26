"""H34 Track B3 라벨 선택 규칙 5종 비교. 대상 14개(풀 ≥ 5셀) × 분할 3 × n {3, 5, 10, 20, 40} × 규칙 5 × 반복. CPU.

목적. 라벨 희소 지역에서 소수 라벨 n개를 어떤 규칙으로 고를 것인가를 두 목적 함수로 판정한다.
계획서 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3(공통 설계), §4 Track B3, §5 F7, §8 산출물.
골격은 scripts/3_deep_learning/h25_label_budget.py(대상 정의·LORO 원천 + 100 km 버퍼·A/B 블록 2분할·채점 셀·E 수축·잔차 CatBoost)를
그대로 따르고, 선택 규칙만 5종으로 넓힌다. 통계 규약은 src/polar/h4_common.py 를 따른다.

대상. 주 4지역(레나·캐나다·러시아 W·러시아 E) + 하위 지역(알래스카 k=6·캐나다 k=3·레나 k=2, 블록 중심 k-means, 라벨 미사용).
  어느 분할에서든 풀 |A| < --min-pool(기본 5)인 대상만 제외한다(CA-1, 풀 최소 3셀). 기본 대상은 주 4지역 + 하위 지역 10개 = 14개,
  작업 14 × 3 분할 = 42개다. 풀 A 가 한 분할이라도 0.5° 블록 1개뿐인 대상(AL-1·AL-4·AL-5·AL-6·LE-1·LE-2, 지배 블록 1개가 풀 전체)은
  RMSE 목적과 반복 추출 분산(var_rep)에는 그대로 포함한다. 이런 풀은 블록 부트스트랩이 항상 같은 풀을 돌려주므로 E 부트스트랩의 재표집
  단위를 그 분할에서만 셀로 바꾸고(boot_unit=cell), F7 주 판정 집계에서는 뺀다(아래 F7 참조).
원천(LORO). 대상 셀을 뺀 전체 F4 라벨 셀에서 대상 경계 --buffer-km 이내 셀을 제외. 앵커 E0 = 원천 최소제곱.
분할. 대상 블록 A/B 2분할(split 0..K-1). 라벨 풀 = A(√TDD 유효 셀), 채점 = B(실측·CCI·토양 도일 유효). 대상 라벨은 n개만 학습에 쓴다.

선택 규칙(모두 풀 A 안에서 n개, 비복원).
  random            균등 무작위.
  kmedoid           표준화 x25 k-means(k=n) 중심의 최근접 셀(h25 select 와 동일). k-means 초기화만 난수이므로 반복 간 거의 같은 집합이다.
  kmedoid_weighted  표준화 x25 k-means(K = min(--k-clusters, max(2, |A|//2)))로 군집을 만든 뒤, 군집 크기에 비례해 대표점 수를 배정한다.
                    n < K 이면 큰 군집부터 n개 군집에 1개씩. n ≥ K 이면 군집마다 1개를 먼저 주고 나머지 n − K 를 크기 비례(최대 잉여법)로 배정.
                    군집 안 a개는 군집 내부 k-means(k=a) 중심의 최근접 셀.
  doptimal          E 의 Fisher 정보 Σ TDD_i(= Σ s_i², s = √TDD)를 최대화하는 결정적 탐욕 규칙이다. 한 번에 한 셀씩 고르며 우선순위는
                    (1) 레버리지 s(--lev-col, 기본 e5_sqrt_tdd, 1e-6 반올림) 큰 셀, (2) 이미 고른 셀이 적은 블록의 셀(서로 다른 블록 우선),
                    (3) 표준화 x25 공간에서 이미 고른 셀까지의 최소 거리가 큰 셀(첫 셀은 풀 중심에 가까운 셀), (4) 풀 안 순번이다.
                    같은 0.5° 블록에서 ⌈n/3⌉개를 넘지 않는다(블록 분산 제약). 상한을 만족하는 후보가 없으면 상한을 풀고 같은 우선순위로 채운다.
                    v3 자료의 ERA5 도일(e5_sqrt_tdd·e5_sqrt_tdd_soil·p4_ku)은 모두 0.5° 블록 상수라 (1)의 동률은 거의 항상 같은 블록 안에서
                    생기고, 실제 동작은 "s 상위 블록에서 상한만큼, 블록 안에서는 x25 공간 분산 최대"가 된다. 난수를 쓰지 않으므로 같은 풀에서
                    반복은 모두 같은 집합이다. 대상별 최대 s 동률 셀 수(n_tie_max_A)와 Fisher 정보 비(fisher_ratio)를 함께 기록한다.
  active_ppi        원천 잔차 모델의 예측 불확실성 프록시에 비례한 확률로 추출한다. 프록시 u_i = |ĝ(x_i)|, ĝ = 원천 잔차(y − E0·s)를
                    x25 로 학습한 CatBoost(seed 0)의 A 셀 예측 잔차. 추출 확률 p_i ∝ u_i + 0.1·mean(u) (0 확률 방지), 비복원 순차 추출.

목적 함수.
  (1) E 추정. Ê = 선택 셀 최소제곱(κ=0). 기준 E_own = 대상 전 셀 최소제곱(E_ownA = 풀 A 최소제곱 병기).
      주 정의(2026-09-26 재정의): E 추정 분산 V_tot = 분할 간 분산 + 블록 부트스트랩 분산.
        각 분할에서 풀 A 의 블록을 --nboot-e 회 복원 추출(복제 블록은 별개 블록으로 재명명)하여 재표집 풀을 만들고, 그 풀에 규칙을 다시
        적용해 Ê_{s,b} 를 얻는다. 재표집 인덱스는 모든 규칙·n 에 공유한다(짝지음). 풀 블록이 1개인 분할은 셀 단위로 복원 추출한다(boot_unit=cell).
        n 은 본 실행과 같은 집합(n < 원 풀 크기 |A|)만 쓴다. 재표집 풀 크기 m ≤ n 인 재표집은 건너뛰며, 분할 안에서 살아남은 재표집이
        --nboot-e 의 80 % 미만이면 그 (규칙, n) 의 V_boot·V_tot·MSE_tot 을 NaN 으로 둔다(boot_ok=False, n_boot_min 기록).
        V_boot = mean_s Var_b(Ê_{s,b}), V_between = Var_s(mean_b Ê_{s,b})(ddof=1, 분할 1개면 0), V_tot = V_boot + V_between.
        MSE_tot = mean_{s,b} (Ê_{s,b} − E_own)² (편향 포함).
      이 정의는 모든 규칙에 같게 적용된다. random 의 V_boot 에는 추출 난수 분산이 자동으로 들어간다.
      보조(사전 등록 문구 '추출 분산'): 같은 (분할, n)에서 반복 --reps 회 추출의 Ê 분산 var_rep 과 MSE mse_rep.
      결정적 규칙은 var_rep 이 0 에 가까워 비가 자명하게 유리하므로 주 판정에서 뺀다(n_distinct_E 로 확인).
  (2) B 채점 RMSE. S1 = 수축 E_n(κ=--kappa)·s 단독. S3 = E_n 앵커 + λ·잔차 CatBoost(원천 잔차(E0 앵커) ∪ 대상 n개 잔차(E_n 앵커), 대상 행 가중 α).
      S3src = E_n 앵커 + λ·원천 잔차만의 CatBoost(선택과 무관한 잔차 모델, E_n 만 규칙에 의존). λ = --lam(0.25), α = --alpha(1).
      S1·S3src 는 전 반복, S3 는 앞 --reps-cb 회 × --seeds 개 seed. 같은 선택 집합·seed 의 CatBoost 적합은 재사용한다(결과 동일).

채점 규약(불변). Δ = 방법 − 물리식(E0·s), 음수 = 개선. 규칙 간 비교는 Δ_rand = 규칙 − random.
  지역 안 95 % CI 의 주 정의는 h4_common.boot_delta_blocks 의 블록 부트스트랩이다. 각 분할 안에서 채점 블록을 --nboot 회 복원 추출하고,
  같은 재표집 인덱스를 규칙·random·물리식의 모든 반복·seed 에 공유하며, 반복·seed 평균 후 차를 구하고, 분할별 분포를 같은 번호끼리 평균한다.
  반복 승률은 같은 rep 번호끼리 짝지어 계산한다. 결정적 규칙에서도 정의되는 CI 다.
  보조: (분할, 반복) 행 재표집 CI(h4_common.rep_boot_ci)를 *_ci_rep_lo/hi 로 병기한다. 결정적 규칙에서는 폭이 0 에 가까울 수 있어 판정에 쓰지 않는다.
  모든 실행 단위의 채점 블록별 SSE·셀 수는 <tag>_blocksse.npz(h4_common.save_stores)로 저장한다.
  지역 평균은 두 층을 따로 요약한다(m1_stats.summarize_delta 의 층화 결합): layer=main(주 4지역), layer=sub(하위 지역).
  하위 지역은 주 지역의 부분집합이므로 한 평균에 섞으면 같은 채점 셀이 두 번 들어간다(검토 2026-09-26).

판정 F7(재정의). 사전 등록 문구는 "n=3 에서 D-최적의 E 추정 분산(추출 분산)이 random 대비 ≥ 30 % 감소한 대상이 14 중 10 이상"이다.
  doptimal 은 결정적 규칙이라 추출 분산이 0 이 되어 원 문구가 자명하게 참이 되므로, E 추정 분산을 위 V_tot 로 재정의한다.
  주 판정: vtot_ratio = V_tot(doptimal) / V_tot(random) ≤ 0.7 인 대상 수 ≥ ⌈(10/14)·N_eval⌉. N_eval 은 판정 가능 대상 수로,
  전 분할 풀 블록 ≥ 2(블록 부트스트랩 정의 가능)이고 boot_ok 인 대상이다. 사전 등록 문구 '14 중 10'을 비율로 옮긴 것이며, 분모는 표에 적는다.
  보조 판정: 셀 단위 부트스트랩으로 대체한 대상까지 포함한 전체(분모 = vtot_ratio 가 유한한 대상 수), 같은 비율 기준.
  같은 표에 다음을 병기한다. (i) 편향 포함 MSE 비 mse_tot_ratio ≤ 0.7. (ii) 사전 등록 문구 그대로의 var_rep_ratio·mse_rep_ratio(보조, 자명성 표시).
  (iii) fisher_ratio > 1.1 대상 한정 통과 수. (iv) 전 분할 풀 A 블록 수 ≥ --min-blocks-a(기본 3) 대상 한정 통과 수.
  RMSE 목적은 n=10 에서 어느 규칙도 random 보다 유의하게 나쁘지 않은지(블록 CI 하한 > 0 인 대상 수, 주)와 행 재표집 CI(보조)를 기록한다.

입력. data/processed/fidelity_base_v3.csv + e5_soil_tdd_v3.csv(m1_core.load_base 병합), 입력 x25, 물리 예측 = E·e5_sqrt_tdd.
출력(data/processed/h4/). <tag>_runs.csv(실행 단위 행), <tag>_blocksse.npz(블록 SSE), <tag>_eboot.csv(E 블록 부트스트랩 Ê),
  <tag>_summary.csv(대상 × 규칙 × n × 단계 RMSE 목적), <tag>_evar.csv(E 추정 목적), <tag>_selection.csv(규칙 비교 통합표, 계획서 산출물),
  <tag>_region.csv(층별 지역 평균), <tag>_f7.csv(판정표), <tag>_targets.csv, <tag>_meta.json. 재집계는 <tag>_meta_summarize.json 에 따로 쓴다.
  스모크는 <tag>_smoke_*.
실행(ROOT, CPU).
  스모크: python3 scripts/3_deep_learning/h34_label_design.py --smoke --workers 2
  본:     python3 scripts/3_deep_learning/h34_label_design.py --tag b3 --workers 4
  재집계: python3 scripts/3_deep_learning/h34_label_design.py --tag b3 --summarize-only (runs·blocksse·eboot 재사용, 본 실행 meta 보존)
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import traceback
import warnings


def _quiet_unraisable(u, _default=sys.unraisablehook):
    """threadpoolctl 의 OpenBLAS 버전 조회 실패(ctypes 콜백 AttributeError)는 동작과 무관한 소음이므로 버린다. 그 밖의 예외는 기본 처리."""
    try:
        tb = "".join(traceback.format_tb(u.exc_traceback)) if u.exc_traceback is not None else ""
        if isinstance(u.exc_value, AttributeError) and "threadpoolctl" in tb:
            return
    except Exception:                                                                                 # noqa: BLE001
        pass
    _default(u)


sys.unraisablehook = _quiet_unraisable
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", module="threadpoolctl")
warnings.filterwarnings("ignore", message="Number of distinct clusters")          # 블록 부트스트랩 풀의 복제 셀로 인한 k-means 경고(선택 결과는 n개 유지)

import argparse                                                                                       # noqa: E402
import json                                                                                           # noqa: E402
import math                                                                                           # noqa: E402
import multiprocessing                                                                                # noqa: E402
import subprocess                                                                                     # noqa: E402
import time                                                                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed                                      # noqa: E402
from pathlib import Path                                                                              # noqa: E402

import numpy as np                                                                                    # noqa: E402
import pandas as pd                                                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
from polar.m1_stats import summarize_delta                                                           # noqa: E402
from polar.h4_common import (BlockStore, save_stores, load_stores, stores_for_target, boot_delta_blocks,  # noqa: E402
                             rep_boot_ci, seed_of)

IS_MAIN = __name__ == "__main__"

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역, 풀 조건 미달 제외)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40")
ap.add_argument("--reps", type=int, default=20, help="해석적 단계(S1·S3src) 반복")
ap.add_argument("--reps-cb", type=int, default=5, help="CatBoost 단계(S3) 반복(앞에서부터)")
ap.add_argument("--seeds", type=int, default=1, help="CatBoost seed 개수")
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--k-clusters", type=int, default=8, help="kmedoid_weighted 군집 수 상한")
ap.add_argument("--min-pool", type=int, default=5, help="어느 분할에서든 풀 |A| 가 이보다 작으면 대상 제외(풀 블록 1개 대상은 포함하고 F7 주 집계에서만 제외)")
ap.add_argument("--min-blocks-a", type=int, default=3, help="F7 보조 집계: 전 분할에서 풀 A 블록 수가 이 값 이상인 대상만")
ap.add_argument("--lev-col", default="e5_sqrt_tdd", help="doptimal 레버리지 열(기본 √TDD; v3 자료의 후보는 모두 블록 상수)")
ap.add_argument("--rules", default="random,kmedoid,kmedoid_weighted,doptimal,active_ppi")
ap.add_argument("--nboot", type=int, default=1000, help="채점 블록 부트스트랩 횟수")
ap.add_argument("--nboot-e", type=int, default=100, help="E 추정 분산용 풀 A 블록 부트스트랩 횟수(규칙 재적용)")
ap.add_argument("--boot-min-frac", type=float, default=0.8, help="분할 안에서 살아남은 E 재표집 비율이 이보다 작으면 V_boot = NaN")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="b3")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs·blocksse·eboot 로 집계만 다시(meta 는 별도 파일)")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3, 10]
REPS = args.reps if not args.smoke else 3
REPS_CB = args.reps_cb if not args.smoke else 1
NBOOT_E = args.nboot_e if not args.smoke else 20
SEEDS = list(range(args.seeds))
RULES = args.rules.split(",")
if "random" not in RULES:
    RULES = ["random"] + RULES
STAGES = ["S1", "S3src", "S3"]
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
TAG = args.tag + ("_smoke" if args.smoke else "")
PHYS_KEY = ("none", "physics", 0, -1, -1)                  # BlockStore 키 = (rule, stage, n, rep, seed)

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
DF["lev"] = DF[args.lev_col].values.astype(float)                                                   # doptimal 레버리지(기본 = s)


# ---------------------------------------------------------------- 하위 지역(h25 make_subregions 와 동일, 라벨 미사용)
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


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def parent_of(t):
    return t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])


def layer_of(t):
    return "main" if t in MAIN4 else "sub"


def pool_ok(t):
    """모든 분할에서 풀 |A| ≥ --min-pool 인지. 반환 (통과, 최소 풀 크기, 최소 풀 블록 수). 풀 블록 수는 F7 집계 구분에만 쓴다."""
    ti = target_idx(t); nmin, bmin = 10 ** 9, 10 ** 9
    for sp in SPLITS:
        A = half_split_blocks(DF, ti, sp)[0]
        A = A[np.isfinite(DF.s.values[A]) & (DF.s.values[A] > 0)]
        nmin = min(nmin, len(A)); bmin = min(bmin, len(np.unique(DF.block.values[A])))
    return (nmin >= args.min_pool), nmin, bmin


if args.targets:
    CAND = args.targets.split(",")
elif args.smoke:
    CAND = ["Lena", "AL-3", "Russia_W", "AL-4"]                       # AL-4: 풀 블록 1개 분할 경로 확인
else:
    CAND = MAIN4 + sorted(SUBS.subregion.tolist())
TARGETS, SKIPPED, POOL_BLOCKS_MIN = [], {}, {}
for _t in CAND:
    _ok, _nmin, _bmin = pool_ok(_t)
    POOL_BLOCKS_MIN[_t] = int(_bmin)
    if _ok:
        TARGETS.append(_t)
        if _bmin < 2 and IS_MAIN:
            print(f"[note] {_t}: 풀 블록 최소 {_bmin}. RMSE·var_rep 포함, E 부트스트랩은 해당 분할 셀 단위, F7 주 집계 제외", flush=True)
    else:
        SKIPPED[_t] = dict(pool_min=int(_nmin), pool_blocks_min=int(_bmin))
        if IS_MAIN:
            print(f"[skip] {_t}: 풀 최소 {_nmin}셀(기준 {args.min_pool}) · 풀 블록 최소 {_bmin}", flush=True)
if IS_MAIN:
    print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS} · 규칙 {RULES} · n {N_GRID} · 분할 {SPLITS} · 반복 {REPS}/{REPS_CB} · "
          f"E 부트스트랩 {NBOOT_E}", flush=True)


# ---------------------------------------------------------------- 기본 연산(h25 와 동일)
def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte_list, seed, w=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return [np.asarray(m.predict(X), float) for X in Xte_list]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


# ---------------------------------------------------------------- 선택 규칙
def _nearest_to_centers(Z, centers, taken):
    sel = list(taken)
    for c in centers:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return sel[len(taken):]


def _kmeans(Z, k, rng):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)


def _doptimal(n, lev, blocks, Z):
    """결정적 탐욕 D-최적. 우선순위 (1) lev 큰 셀, (2) 이미 고른 셀이 적은 블록, (3) 선택 집합까지 최소 거리(첫 셀은 풀 중심 근접),
    (4) 순번. 블록 상한 ⌈n/3⌉, 만족 후보가 없으면 상한 해제."""
    m = len(lev); cap = math.ceil(n / 3)
    lv = np.round(np.where(np.isfinite(lev), lev, -np.inf), 6)
    _, bcode = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    bcnt = np.zeros(bcode.max() + 1, int)
    spread = -((Z - Z.mean(0)) ** 2).sum(1)                 # 첫 셀: 중심에 가까울수록 큼
    avail = np.ones(m, bool); sel = []; order_idx = np.arange(m)
    for _ in range(n):
        cb = bcnt[bcode]
        ok = avail & (cb < cap)
        if not ok.any():
            ok = avail
        cand = np.where(ok)[0]
        o = np.lexsort((order_idx[cand], -spread[cand], cb[cand], -lv[cand]))
        i = int(cand[o[0]])
        sel.append(i); avail[i] = False; bcnt[bcode[i]] += 1
        d = ((Z - Z[i]) ** 2).sum(1)
        spread = d if len(sel) == 1 else np.minimum(spread, d)
    return np.array(sel)


def select(rule, n, rng, Z, lev, blocks, u):
    m = len(Z)
    if rule == "random":
        return rng.choice(m, n, replace=False)
    if rule == "kmedoid":
        km = _kmeans(Z, n, rng)
        return np.array(_nearest_to_centers(Z, km.cluster_centers_, []))
    if rule == "kmedoid_weighted":
        K = min(args.k_clusters, max(2, m // 2))
        km = _kmeans(Z, K, rng); lab = km.labels_; size = np.bincount(lab, minlength=K)
        alloc = np.zeros(K, int)
        if n < K:
            alloc[np.argsort(-size, kind="stable")[:n]] = 1
        else:
            alloc[:] = 1
            rem = n - K
            if rem > 0:
                q = rem * size / size.sum(); base = np.floor(q).astype(int); alloc += base
                left = rem - int(base.sum())
                for c in np.argsort(-(q - base), kind="stable")[:left]:
                    alloc[c] += 1
        alloc = np.minimum(alloc, size)
        short = n - int(alloc.sum())                        # 군집 크기 상한에 걸린 몫은 여유 있는 군집으로
        while short > 0:
            room = size - alloc
            c = int(np.argmax(room)); alloc[c] += 1; short -= 1
        sel = []
        for c in range(K):
            a = int(alloc[c])
            if a == 0:
                continue
            idx_c = np.where(lab == c)[0]
            if a >= len(idx_c):
                sel.extend(idx_c.tolist()); continue
            if a == 1:
                d = ((Z[idx_c] - km.cluster_centers_[c]) ** 2).sum(1); sel.append(int(idx_c[np.argmin(d)])); continue
            kc = _kmeans(Z[idx_c], a, rng)
            loc = _nearest_to_centers(Z[idx_c], kc.cluster_centers_, [])
            sel.extend(idx_c[loc].tolist())
        return np.array(sel[:n])
    if rule == "doptimal":
        return _doptimal(n, lev, blocks, Z)
    if rule == "active_ppi":
        p = u + 0.1 * float(np.mean(u)) + 1e-12; p = p / p.sum()
        return rng.choice(m, n, replace=False, p=p)
    raise ValueError(rule)


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def source_index(t_idx):
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


def e_bootstrap(t, sp, yA, sA, levA, bA, Z, u):
    """풀 A 블록 부트스트랩으로 규칙을 다시 적용한 Ê. 재표집 인덱스는 규칙·n 에 공유(짝지음). 복제 블록은 별개 블록으로 재명명.
    풀 블록이 1개면 블록 재표집이 항상 원 풀을 돌려주므로 셀 단위로 복원 추출한다(boot_unit=cell, 블록 식별자는 원 블록 유지).
    n 은 본 실행과 같은 집합(n < 원 풀 크기)만 쓰고, 재표집 풀 크기 m ≤ n 인 재표집은 건너뛴다."""
    ub = np.unique(bA); cells = {b: np.where(bA == b)[0] for b in ub}
    nA = len(bA); unit = "block" if len(ub) >= 2 else "cell"
    out = []
    for b in range(NBOOT_E):
        rs = np.random.RandomState(seed_of("h34eboot", t, sp, b))
        if unit == "block":
            pick = rs.randint(0, len(ub), len(ub))
            idx = np.concatenate([cells[ub[j]] for j in pick])
            blk = np.concatenate([np.full(len(cells[ub[j]]), f"{ub[j]}#{r}", dtype=object) for r, j in enumerate(pick)])
            nb_res = len(np.unique(ub[pick]))
        else:
            idx = rs.randint(0, nA, nA); blk = bA[idx].astype(object); nb_res = 1
        m = len(idx)
        for n in N_GRID:
            if n >= nA or n >= m:
                continue
            for rule in RULES:
                rng = np.random.RandomState(seed_of("h34eboot_sel", t, sp, n, rule, b))
                sel = np.asarray(select(rule, n, rng, Z[idx], levA[idx], blk, u[idx]), int)
                out.append(dict(target=t, split=sp, rule=rule, n=int(n), b=int(b), E_hat=ls_E(yA[idx][sel], sA[idx][sel]),
                                n_sel=int(len(np.unique(sel))), n_pool=int(m), n_blocks_distinct=int(nb_res), boot_unit=unit))
    return out


def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t); parent = parent_of(t)
    src_idx = source_index(t_idx)
    src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values
    r_src = y_src - E0 * s_src; ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    A_idx = A_idx[np.isfinite(DF.s.values[A_idx]) & (DF.s.values[A_idx] > 0)]           # 풀 = √TDD 유효
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, XA, bA, levA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32), A.block.values.astype(str), A.lev.values
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    E_own = ls_E(DF.y.values[t_idx], DF.s.values[t_idx]); E_ownA = ls_E(yA, sA)
    lv = np.round(levA, 6); n_tie_max = int((lv == np.nanmax(lv)).sum())
    # 원천 잔차 모델(선택과 무관): active_ppi 프록시 + S3src
    gA, gB = cb_fit_predict(XR_src, R_src, [XA, XB], 0); n_fit = 1
    u = np.abs(gA)
    phys = E0 * sB
    rows = []
    base = dict(target=t, parent=parent, layer=layer_of(t), split=sp, n_A=nA, n_blocks_A=int(len(np.unique(bA))), n_tie_max_A=n_tie_max,
                n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), E0=E0, E_own=E_own, E_ownA=E_ownA)
    st = BlockStore(t, sp, B.block.values, meta=dict(n_A=nA, n_eval=int(len(evB)), E0=E0, E_own=E_own))

    def add(rule, stage, n, rep, seed, pred, E_ls, E_n, fisher, nb_sel):
        rows.append(dict(**base, rule=rule, stage=stage, n=int(n), rep=int(rep), seed=int(seed), lam=float(args.lam), alpha=float(args.alpha),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)),
                         E_hat=float(E_ls), E_n=float(E_n), fisher=float(fisher), n_blocks_sel=int(nb_sel)))
        st.add((rule, stage, int(n), int(rep), int(seed)), yB, pred)

    add("none", "physics", 0, -1, -1, phys, E0, E0, np.nan, 0)
    rows[-1].update(lam=0.0, alpha=1.0, fisher=np.nan)

    cb_cache = {}
    for n in N_GRID:
        if n >= nA:
            continue
        for rule in RULES:
            for rep in range(REPS):
                rng = np.random.RandomState(seed_of("h34", t, sp, n, rule, rep))
                sel = np.asarray(select(rule, n, rng, Z, levA, bA, u), int)
                assert len(sel) == n and len(np.unique(sel)) == n, (t, sp, rule, n, rep, len(sel))
                E_ls = ls_E(yA[sel], sA[sel]); E_n = (n * E_ls + args.kappa * E0) / (n + args.kappa)
                fisher = float((sA[sel] ** 2).sum()); nb_sel = len(np.unique(bA[sel]))
                add(rule, "S1", n, rep, -1, E_n * sB, E_ls, E_n, fisher, nb_sel)
                add(rule, "S3src", n, rep, 0, E_n * sB + args.lam * gB, E_ls, E_n, fisher, nb_sel)
                if rep < REPS_CB:
                    r_n = yA[sel] - E_n * sA[sel]
                    Xtr = np.vstack([XR_src, XA[sel]]); ytr = np.concatenate([R_src, r_n])
                    w = np.concatenate([np.ones(len(R_src)), np.full(n, args.alpha)])
                    for seed in SEEDS:
                        ck = (tuple(sorted(sel.tolist())), seed)
                        if ck not in cb_cache:                                  # 같은 선택·seed 는 적합 결과가 같으므로 재사용
                            (cb_cache[ck],) = cb_fit_predict(Xtr, ytr, [XB], seed, w=w); n_fit += 1
                        add(rule, "S3", n, rep, seed, E_n * sB + args.lam * cb_cache[ck], E_ls, E_n, fisher, nb_sel)
    t1 = time.time()
    erows = e_bootstrap(t, sp, yA, sA, levA, bA, Z, u)
    meta = dict(target=t, parent=parent, layer=layer_of(t), split=sp, n_A=nA, n_blocks_A=int(len(np.unique(bA))), n_tie_max_A=n_tie_max,
                n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), E0=E0, E_own=E_own, E_ownA=E_ownA, n_fit=n_fit,
                n_fit_cached=len(cb_cache), elapsed_main_s=round(t1 - t0, 1), elapsed_eboot_s=round(time.time() - t1, 1),
                elapsed_s=round(time.time() - t0, 1))
    return rows, st, erows, meta


# ---------------------------------------------------------------- 집계
def _kf(rule, stage, n):
    return lambda k: k[0] == rule and k[1] == stage and k[2] == int(n)


def _rep(k):
    return k[3]


def _as_region_dict(r, by_split):
    """boot_delta_blocks 결과 → m1_stats.summarize_delta 입력 dict."""
    ncell = float(np.mean([st.ncell.sum() for st in by_split.values()]))
    return dict(n_cells=int(round(ncell)), n_blocks=int(round(np.mean(r["n_blocks"]))) if r["n_blocks"] else 0, n_seed=r["n_keys_A"],
                rmse_A=r["rmse_A"], rmse_B=r["rmse_B"], delta_per_seed=r["delta"], delta_blockeq=r["delta_beq"], delta_cap100=np.nan,
                block_majority=np.nan, dist_cell=r.get("dist"), dist_blockeq=r.get("dist_beq"))


def block_tests(stores, runs):
    """(대상, 규칙, 단계, n, 기준) 마다 블록 부트스트랩 Δ. 대상 안 모든 비교에 같은 재표집 seed(짝지음)."""
    res = {}
    keys = runs[runs.stage != "physics"][["target", "rule", "stage", "n"]].drop_duplicates().itertuples(index=False)
    by_t = {}
    for t, rule, stage, n in keys:
        if t not in by_t:
            by_t[t] = stores_for_target(stores, t)
        bs = by_t[t]; sd = seed_of("h34blk", t)
        res[(t, rule, stage, int(n), "phys")] = boot_delta_blocks(bs, _kf(rule, stage, n), PHYS_KEY, nboot=args.nboot, seed=sd,
                                                                  rep_fn=_rep, return_dist=True)
        if rule != "random":
            res[(t, rule, stage, int(n), "random")] = boot_delta_blocks(bs, _kf(rule, stage, n), _kf("random", stage, n), nboot=args.nboot,
                                                                        seed=sd, rep_fn=_rep, return_dist=True)
    return res, by_t


def summarize(runs, stores, eboot):
    r = runs[runs.stage != "physics"].copy()
    phys = runs[runs.stage == "physics"].set_index(["target", "split"]).rmse_cm
    r["d_phys"] = r.rmse_cm.values - np.array([phys.loc[(t, s)] for t, s in zip(r.target, r.split)])
    key = ["target", "split", "stage", "n", "rep", "seed"]
    rnd = r[r.rule == "random"][key + ["rmse_cm", "rmse_beq_cm"]].rename(columns={"rmse_cm": "rmse_rand", "rmse_beq_cm": "rmse_beq_rand"})
    r = r.merge(rnd, on=key, how="left")
    r["d_rand"] = r.rmse_cm - r.rmse_rand; r["d_rand_beq"] = r.rmse_beq_cm - r.rmse_beq_rand
    g = r.groupby(["target", "parent", "layer", "split", "rule", "stage", "n", "rep"], as_index=False).agg(
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"), d_rand=("d_rand", "mean"),
        d_rand_beq=("d_rand_beq", "mean"), n_eval=("n_eval", "first"))
    bt, by_t = block_tests(stores, runs)
    out = []
    for (t, par, lay, rule, stage, n), sub in g.groupby(["target", "parent", "layer", "rule", "stage", "n"]):
        lo, hi = rep_boot_ci(sub.d_phys.values, nboot=args.nboot, seed=1); rlo, rhi = rep_boot_ci(sub.d_rand.values, nboot=args.nboot, seed=2)
        ndist = float(sub.groupby("split").d_phys.apply(lambda v: len(np.unique(np.round(v, 9)))).mean())
        row = dict(target=t, parent=par, layer=lay, rule=rule, stage=stage, n=n, n_runs=len(sub), n_splits=sub.split.nunique(),
                   n_eval_mean=float(sub.n_eval.mean()), rmse_mean=float(sub.rmse_cm.mean()), rmse_sd=float(sub.rmse_cm.std()),
                   rmse_beq_mean=float(sub.rmse_beq_cm.mean()), d_phys_mean=float(sub.d_phys.mean()), win_phys_rows=float((sub.d_phys < 0).mean()),
                   d_rand_mean=float(sub.d_rand.mean()), d_rand_beq_mean=float(sub.d_rand_beq.mean()), n_distinct_dphys=ndist,
                   d_phys_ci_rep_lo=lo, d_phys_ci_rep_hi=hi, d_rand_ci_rep_lo=rlo, d_rand_ci_rep_hi=rhi)
        for ref, pre in (("phys", "blk_d_phys"), ("random", "blk_d_rand")):
            b = bt.get((t, rule, stage, int(n), ref))
            if b is None:
                continue
            row.update({pre: b["delta"], f"{pre}_lo": b["ci_lo"], f"{pre}_hi": b["ci_hi"], f"{pre}_p": b["p_boot"],
                        f"{pre}_beq": b["delta_beq"], f"{pre}_beq_lo": b["ci_lo_beq"], f"{pre}_beq_hi": b["ci_hi_beq"],
                        f"{pre}_split_win": b["split_win"], f"{pre}_rep_win": b["rep_win"], f"{pre}_ci_flag": b["ci_flag"]})
            if ref == "phys":
                row.update(n_blocks_eval_mean=float(np.mean(b["n_blocks"])), rmse_phys=b["rmse_B"])
        out.append(row)
    summary = pd.DataFrame(out)
    # 층별 지역 평균(주 4지역 층, 하위 지역 층 분리)
    reg_rows = []
    targets = sorted(summary.target.unique())
    layers = {"main": [t for t in targets if layer_of(t) == "main"], "sub": [t for t in targets if layer_of(t) == "sub"]}
    for (rule, stage, n) in summary[["rule", "stage", "n"]].drop_duplicates().itertuples(index=False):
        for ref in ("phys", "random"):
            if ref == "random" and rule == "random":
                continue
            for lay, tl in layers.items():
                per = {t: _as_region_dict(bt[(t, rule, stage, int(n), ref)], by_t[t]) for t in tl if (t, rule, stage, int(n), ref) in bt}
                if not per:
                    continue
                for rr in summarize_delta(f"{rule}|{stage}|n{n}|vs_{ref}|{lay}", per, tl, seed_of("reg", rule, stage, n, ref, lay)):
                    rr.update(rule=rule, stage=stage, n=n, ref=ref, layer=lay); reg_rows.append(rr)
    region = pd.DataFrame(reg_rows)
    ev = e_objective(runs, eboot)
    f7 = f7_table(ev, summary)
    sel = selection_table(ev, summary)
    return summary, ev, region, f7, sel


def e_objective(runs, eboot):
    """E 추정 목적. 주 = 블록 부트스트랩 V_tot·MSE_tot, 보조 = 반복 추출 var_rep·mse_rep."""
    info = runs[runs.stage == "physics"].groupby("target").agg(E_own=("E_own", "first"), parent=("parent", "first"), layer=("layer", "first"),
                                                              n_blocks_A_min=("n_blocks_A", "min"), n_A_min=("n_A", "min"),
                                                              n_tie_max_A=("n_tie_max_A", "max")).reset_index()
    # 보조: 반복 추출(S1 행 = 반복마다 1행)
    e = runs[runs.stage == "S1"].copy(); e["err"] = e.E_hat - e.E_own; e["errA"] = e.E_hat - e.E_ownA
    ps = e.groupby(["target", "rule", "n", "split"], as_index=False).agg(
        var_E=("E_hat", "var"), fisher=("fisher", "mean"), n_rep=("rep", "size"),
        n_distinct=("E_hat", lambda v: len(np.unique(np.round(v, 9)))))
    rep = ps.groupby(["target", "rule", "n"], as_index=False).agg(var_rep=("var_E", "mean"), fisher_mean=("fisher", "mean"),
                                                                 n_distinct_E=("n_distinct", "mean"), n_rep=("n_rep", "min"))
    rep2 = e.groupby(["target", "rule", "n"], as_index=False).agg(mean_E_rep=("E_hat", "mean"), mse_rep=("err", lambda v: float(np.mean(v ** 2))),
                                                                 mse_rep_A=("errA", lambda v: float(np.mean(v ** 2))),
                                                                 n_blocks_sel_mean=("n_blocks_sel", "mean"), E_ownA=("E_ownA", "mean"))
    rep = rep.merge(rep2, on=["target", "rule", "n"])
    # 주: 블록 부트스트랩
    eb = eboot.merge(info[["target", "E_own"]], on="target", how="left"); eb["err2"] = (eb.E_hat - eb.E_own) ** 2
    if "boot_unit" not in eb:
        eb["boot_unit"] = "block"
    bs = eb.groupby(["target", "rule", "n", "split"], as_index=False).agg(m_b=("E_hat", "mean"), v_b=("E_hat", "var"), n_b=("E_hat", "count"),
                                                                      cell_unit=("boot_unit", lambda v: bool((v == "cell").any())))
    bt = bs.groupby(["target", "rule", "n"], as_index=False).agg(
        V_boot=("v_b", "mean"), V_between=("m_b", lambda v: float(np.var(v, ddof=1)) if len(v) > 1 else 0.0),
        n_splits_boot=("split", "nunique"), n_boot_min=("n_b", "min"), n_splits_cellboot=("cell_unit", "sum"))
    # 본 실행에서 해당 n 이 돈 분할 수(= n < |A| 인 분할)와 비교해 부트스트랩이 빠진 분할이 없어야 하고, 분할마다 재표집 생존 비율 ≥ --boot-min-frac
    nsp = runs[runs.stage == "S1"].groupby(["target", "rule", "n"], as_index=False).agg(n_splits_run=("split", "nunique"))
    bt = bt.merge(nsp, on=["target", "rule", "n"], how="left")
    bt["boot_frac_min"] = bt.n_boot_min / float(NBOOT_E)
    bt["boot_ok"] = (bt.boot_frac_min >= args.boot_min_frac) & (bt.n_splits_boot >= bt.n_splits_run.fillna(0))
    bt["boot_unit"] = np.where(bt.n_splits_cellboot > 0, "cell(일부 분할)", "block")
    bt["V_tot"] = bt.V_boot + bt.V_between
    bm = eb.groupby(["target", "rule", "n"], as_index=False).agg(mean_E_boot=("E_hat", "mean"), MSE_tot=("err2", "mean"))
    bt = bt.merge(bm, on=["target", "rule", "n"], how="left")
    for c in ("V_boot", "V_between", "V_tot", "MSE_tot"):
        bt.loc[~bt.boot_ok, c] = np.nan
    ev = rep.merge(bt, on=["target", "rule", "n"], how="outer").merge(info, on="target", how="left")
    ev["bias_boot"] = ev.mean_E_boot - ev.E_own; ev["bias_rep"] = ev.mean_E_rep - ev.E_own
    rcols = dict(V_tot="V_tot_rand", MSE_tot="MSE_tot_rand", var_rep="var_rep_rand", mse_rep="mse_rep_rand", fisher_mean="fisher_rand")
    rnd = ev[ev.rule == "random"][["target", "n"] + list(rcols)].rename(columns=rcols)
    ev = ev.merge(rnd, on=["target", "n"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        ev["vtot_ratio"] = ev.V_tot / ev.V_tot_rand; ev["mse_tot_ratio"] = ev.MSE_tot / ev.MSE_tot_rand
        ev["var_rep_ratio"] = ev.var_rep / ev.var_rep_rand; ev["mse_rep_ratio"] = ev.mse_rep / ev.mse_rep_rand
        ev["fisher_ratio"] = ev.fisher_mean / ev.fisher_rand
    ev["rep_degenerate"] = ev.n_distinct_E <= 1.0 + 1e-9                                              # 반복 추출 분산이 0(결정적)
    ev["boot_ok_rand"] = ev.merge(ev[ev.rule == "random"][["target", "n", "boot_ok"]].rename(columns={"boot_ok": "_br"}),
                                  on=["target", "n"], how="left")["_br"].fillna(False).astype(bool).values
    # F7 주 집계 가능: 전 분할 풀 블록 ≥ 2(블록 부트스트랩 정의 가능), 규칙·random 모두 boot_ok, 비가 유한
    ev["f7_eval"] = (ev.n_blocks_A_min >= 2) & ev.boot_ok.fillna(False).astype(bool) & ev.boot_ok_rand & np.isfinite(ev.vtot_ratio)
    ev["f7_excl_reason"] = np.select([ev.n_blocks_A_min < 2, ~ev.boot_ok.fillna(False).astype(bool) | ~ev.boot_ok_rand,
                                      ~np.isfinite(ev.vtot_ratio)],
                                     ["pool_blocks<2(V_boot 셀 단위 대체)", "boot_frac<min 또는 분할 누락", "ratio 비유한"], "")
    return ev.sort_values(["n", "rule", "target"]).reset_index(drop=True)


F7_FRAC = 10.0 / 14.0                                          # 사전 등록 '14 중 10' 을 비율로


def f7_need(n_den):
    return int(math.ceil(F7_FRAC * n_den - 1e-9)) if n_den > 0 else np.nan


def f7_table(ev, summary):
    f7 = []

    def cnt(sub, col, thr=0.7, label=None, note=""):
        v = sub[col]; ok = v <= thr; nd = int(v.notna().sum())
        excl = sub[~sub.f7_eval] if "f7_eval" in sub else sub.iloc[:0]
        f7.append(dict(rule=rule, n=n, metric=label or f"{col}<={thr}", n_targets=nd, n_pass=int(ok.sum()), n_need=f7_need(nd),
                       median_ratio=float(v.median()) if v.notna().any() else np.nan, pass_targets=",".join(sub.target[ok]),
                       n_boot_min=float(sub.n_boot_min.min()) if "n_boot_min" in sub and sub.n_boot_min.notna().any() else np.nan,
                       excluded=";".join(f"{a}:{b}" for a, b in zip(excl.target, excl.f7_excl_reason)), note=note))

    for (rule, n), sub in ev[ev.rule != "random"].groupby(["rule", "n"]):
        se = sub[sub.f7_eval]
        cnt(se, "vtot_ratio", label="vtot_ratio<=0.7|f7_eval",
            note="주 판정(재정의): V_tot = 분할 간 분산 + 블록 부트스트랩 분산. 분모 = 전 분할 풀 블록 ≥ 2·boot_ok 대상, 기준 ⌈(10/14)·분모⌉")
        f7[-1]["excluded"] = ";".join(f"{a}:{b}" for a, b in zip(sub.target[~sub.f7_eval], sub.f7_excl_reason[~sub.f7_eval]))
        cnt(sub, "vtot_ratio", note="보조: 셀 단위 부트스트랩 대체 대상 포함 전체, 기준 ⌈(10/14)·분모⌉")
        cnt(se, "mse_tot_ratio", label="mse_tot_ratio<=0.7|f7_eval", note="편향 포함 MSE(블록 부트스트랩, E_own 기준), 주 집계 대상")
        cnt(sub, "var_rep_ratio", note=f"사전 등록 문구 '추출 분산'(보조), 결정적 대상 {int(sub.rep_degenerate.sum())}개는 자명")
        cnt(sub, "mse_rep_ratio", note="반복 추출 MSE(보조)")
        s2 = se[se.fisher_ratio > 1.1]
        cnt(s2, "vtot_ratio", label="vtot_ratio<=0.7|f7_eval&fisher_ratio>1.1", note="정보 이득 있는 대상 한정(주 집계 대상 안)")
        s3 = sub[sub.n_blocks_A_min >= args.min_blocks_a]
        cnt(s3, "vtot_ratio", label=f"vtot_ratio<=0.7|n_blocks_A_min>={args.min_blocks_a}", note="풀 블록 충분 대상 한정")
    for (rule, stage, n), sub in summary[summary.rule != "random"].groupby(["rule", "stage", "n"]):
        for lab, lo, hi, note in (("blockCI", "blk_d_rand_lo", "blk_d_rand_hi", "주: 블록 부트스트랩 CI"),
                                  ("repCI", "d_rand_ci_rep_lo", "d_rand_ci_rep_hi", "보조: (분할, 반복) 행 재표집 CI")):
            if lo not in sub:
                continue
            worse = sub[sub[lo] > 0]; better = sub[sub[hi] < 0]
            med = float(sub.blk_d_rand.median()) if "blk_d_rand" in sub else float(sub.d_rand_mean.median())
            f7.append(dict(rule=rule, n=n, metric=f"{stage}:worse_than_random({lab})", n_targets=len(sub), n_pass=len(worse), n_need=np.nan,
                           median_ratio=med, pass_targets=",".join(worse.target), note=note))
            f7.append(dict(rule=rule, n=n, metric=f"{stage}:better_than_random({lab})", n_targets=len(sub), n_pass=len(better), n_need=np.nan,
                           median_ratio=med, pass_targets=",".join(better.target), note=note))
    f7 = pd.DataFrame(f7)
    f7["verdict"] = ""
    for met, tag in (("vtot_ratio<=0.7|f7_eval", "주"), ("vtot_ratio<=0.7", "보조")):
        m = (f7.rule == "doptimal") & (f7.n == 3) & (f7.metric == met)
        f7.loc[m, "verdict"] = [f"{tag}:{'채택' if (np.isfinite(k) and p >= k) else '기각'}({p}/{d}, 기준 {k})"
                                for p, d, k in zip(f7.loc[m, "n_pass"], f7.loc[m, "n_targets"], f7.loc[m, "n_need"])]
    return f7


def selection_table(ev, summary):
    """계획서 산출물 b3_selection: 대상 × 규칙 × n 의 E 추정 목적 + 단계별 RMSE 대 random·물리식(블록 CI)."""
    cols = ["target", "parent", "layer", "rule", "n", "V_tot", "V_boot", "V_between", "MSE_tot", "bias_boot", "vtot_ratio", "mse_tot_ratio",
            "var_rep", "var_rep_ratio", "mse_rep_ratio", "rep_degenerate", "fisher_ratio", "n_blocks_A_min", "n_tie_max_A",
            "boot_unit", "n_boot_min", "boot_ok", "f7_eval", "f7_excl_reason"]
    sel = ev[[c for c in cols if c in ev.columns]].copy()
    for stage in STAGES:
        s = summary[summary.stage == stage]
        keep = {"blk_d_rand": f"{stage}_d_rand", "blk_d_rand_lo": f"{stage}_d_rand_lo", "blk_d_rand_hi": f"{stage}_d_rand_hi",
                "blk_d_phys": f"{stage}_d_phys", "blk_d_phys_lo": f"{stage}_d_phys_lo", "blk_d_phys_hi": f"{stage}_d_phys_hi",
                "rmse_mean": f"{stage}_rmse"}
        s = s[["target", "rule", "n"] + [c for c in keep if c in s.columns]].rename(columns=keep)
        sel = sel.merge(s, on=["target", "rule", "n"], how="left")
    return sel


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        return "NA"


def save_all(runs, stores, eboot, metas, t0, extra, meta_name):
    summary, ev, region, f7, sel = summarize(runs, stores, eboot)
    summary.to_csv(OUT / f"{TAG}_summary.csv", index=False); ev.to_csv(OUT / f"{TAG}_evar.csv", index=False)
    region.to_csv(OUT / f"{TAG}_region.csv", index=False); f7.to_csv(OUT / f"{TAG}_f7.csv", index=False)
    sel.to_csv(OUT / f"{TAG}_selection.csv", index=False)
    if metas is not None:
        pd.DataFrame(metas).to_csv(OUT / f"{TAG}_targets.csv", index=False)
    meta = dict(stage="H34/B3", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 B3, §5 F7", stats="src/polar/h4_common.py",
                targets=TARGETS, skipped=SKIPPED, splits=SPLITS, n_grid=N_GRID, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, rules=RULES,
                stages=STAGES, lam=args.lam, alpha=args.alpha, kappa=args.kappa, buffer_km=args.buffer_km, k_sub=args.k_sub,
                k_clusters=args.k_clusters, min_pool=args.min_pool, min_blocks_a=args.min_blocks_a, lev_col=args.lev_col,
                nboot=args.nboot, nboot_e=NBOOT_E, boot_min_frac=args.boot_min_frac,
                pool_one_block_targets=[t for t in TARGETS if POOL_BLOCKS_MIN.get(t, 2) < 2],
                f7_definition=("주: vtot_ratio = V_tot(doptimal)/V_tot(random) ≤ 0.7 인 대상 수 ≥ ⌈(10/14)·N_eval⌉ (n=3). N_eval = 전 분할 풀 A "
                               "블록 ≥ 2 이고 doptimal·random 모두 boot_ok(분할마다 재표집 생존 비율 ≥ boot_min_frac) 인 대상 수. V_tot = 분할 간 분산 + "
                               "풀 A 블록 부트스트랩 분산(규칙 재적용). 사전 등록 문구 '14 중 10'과의 차이: 풀 블록이 1개인 하위 지역은 블록 "
                               "부트스트랩이 퇴화하여 분모에서 빼고, 기준을 같은 비율로 옮겼다. 보조로 셀 단위 부트스트랩 대체 대상을 포함한 전체 집계를 "
                               "병기한다. 사전 등록 문구의 '추출 분산'(var_rep)은 결정적 규칙에서 0 이 되어 자명하므로 보조로 강등."),
                ci_definition="주: h4_common.boot_delta_blocks(채점 블록 부트스트랩, 짝지음, rep 짝 승률). 보조: rep_boot_ci(ci_rep).",
                region_layers="main = 주 4지역, sub = 하위 지역(따로 요약, 겹치는 셀 이중 계산 방지)",
                git_commit=git_commit(), elapsed_s=round(time.time() - t0, 1), n_rows=int(len(runs)), n_rows_eboot=int(len(eboot)),
                n_rows_summary=int(len(summary)), n_rows_evar=int(len(ev)), n_rows_region=int(len(region)), argv=sys.argv[1:], **extra)
    (OUT / meta_name).write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    pd.set_option("display.width", 220)
    print("\n[F7] n=3 doptimal:"); print(f7[(f7.rule == "doptimal") & (f7.n == 3)].drop(columns=["note"]).to_string(index=False))
    for col in ("vtot_ratio", "mse_tot_ratio", "var_rep_ratio"):
        print(f"\n[E 추정 n=3] {col}(대상 × 규칙):")
        print(ev[ev.n == 3].pivot(index="target", columns="rule", values=col).round(3).to_string())
    print("\n[지역 평균] vs random, 층별:")
    mr = region[region.target.str.startswith("MEAN") & (region.ref == "random")] if len(region) else region
    if len(mr):
        print(mr[["layer", "rule", "stage", "n", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_flag"]].round(3).to_string(index=False))
    print(f"\nsaved {TAG}_* · meta {meta_name} · runs {len(runs)} · eboot {len(eboot)} · summary {len(summary)} · evar {len(ev)} · "
          f"region {len(region)} · {time.time()-t0:.0f}s", flush=True)


def main():
    t0 = time.time()
    if args.summarize_only:
        runs = pd.read_csv(OUT / f"{TAG}_runs.csv")
        stores = load_stores(OUT / f"{TAG}_blocksse.npz")
        eboot = pd.read_csv(OUT / f"{TAG}_eboot.csv")
        save_all(runs, stores, eboot, None, t0, dict(summarize_only=True), f"{TAG}_meta_summarize.json")
        return
    tasks = [(t, sp) for t in TARGETS for sp in SPLITS]
    rows, metas, stores, erows = [], [], [], []
    print(f"[start] 작업 {len(tasks)}개 · 워커 {args.workers} · 스레드 {args.threads}", flush=True)

    def _done(r, st, er, m):
        rows.extend(r); metas.append(m); stores.append(st); erows.extend(er)
        print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']}/{m['n_blocks_A']}블록(최대 s 동률 {m['n_tie_max_A']}) · "
              f"채점 {m['n_eval']}/{m['n_blocks_eval']}블록 · 원천 {m['n_src']} · E0 {m['E0']:.3f} · E_own {m['E_own']:.3f} · "
              f"{m['elapsed_main_s']}+{m['elapsed_eboot_s']}s · 누적 {time.time()-t0:.0f}s · {len(metas)}/{len(tasks)}", flush=True)

    if args.workers <= 1:
        for t, sp in tasks:
            _done(*run_task(t, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows); eboot = pd.DataFrame(erows)
    runs.to_csv(OUT / f"{TAG}_runs.csv", index=False); eboot.to_csv(OUT / f"{TAG}_eboot.csv", index=False)
    save_stores(stores, OUT / f"{TAG}_blocksse.npz")
    save_all(runs, {(st.target, st.split): st for st in stores}, eboot, metas, t0,
             dict(summarize_only=False, n_tasks=len(tasks), n_fit_total=int(sum(m["n_fit"] for m in metas))), f"{TAG}_meta.json")


if __name__ == "__main__":
    main()
