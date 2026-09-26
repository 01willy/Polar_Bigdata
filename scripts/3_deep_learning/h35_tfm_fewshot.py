"""H35 Track B4 표형 파운데이션 모델(TFM) few-shot 잔차 예측기 비교. GPU 1개.

목적
  계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 Track B4, 가설 F8. TabPFN 회귀를 물리 앵커의 잔차 예측기로
  쓰고, 같은 앵커·같은 라벨·같은 λ 의 CatBoost 잔차와 짝지어 비교한다. 판정: AB4 평균 Δ(TFM − CatBoost)의 CI 가 0 을
  포함하면 동급, 하한 > 0 이면 기각. 라벨 0 에서는 실행하지 않는다.

설계(h25_label_budget.py 골격 재사용)
  대상 14 = 주 4지역(레나·캐나다·러시아 W·E) + 하위 지역(data/processed/h3/subregions.csv 와 같은 k-means 정의, |A| 가
  n 격자 최소값 이하인 CA-1 제외). 원천 = LORO(대상 셀 제외) + 대상 경계 100 km 버퍼 제외. 앵커 E0 = 원천 최소제곱.
  분할 = 대상 블록 A/B 2분할(split 0..2), 풀 A, 채점 B(eval_mask). 라벨 n ∈ {3, 10, 40}(|A| 미만), 규칙 kmedoid, 반복 3.
  대상 계수 Ê = log E 오프셋 최대우도(B2 (i)). 공용 규약에 따라 src/polar/h4_common.py 의 offset_mle_prior·offset_mle_estimate
    만 쓴다(검토 09-26: 자체 구현의 τ²(지역별 최소제곱 log E 분산)를 공용 적률 추정으로 교체).
    z = log y − log √TDD. σ² = 원천 macro 지역 안 z 합동 분산, τ² = Var_j(z̄_j) − mean_j(σ²/n_j)(하한 1e-4), 셀 ≥ 3 지역.
    사전 평균 = log E0(원천 최소제곱, B2 와 같음). log Ê = 사후 평균, Ê = exp(사후 평균).
  잔차 컨텍스트 = 원천 잔차(y − E0·√TDD, 지역별 비례 층화 서브샘플 ≤ 10,000 − 3·n_max 행) ∪ 대상 n개 잔차(y − Ê·√TDD, 3배 중복).
  입력 = x25 + 지역 id(원천 macro 정수 코드, 대상은 새 코드; TabPFN·CatBoost 모두 범주형으로 지정). 결측은 두 모델의 기본 처리(NaN native).
  예측 = Ê·√TDD + λ·g, λ ∈ {0.25, 0.5}. 방법: tfm(TabPFN), cb(CatBoost x25 만, h25 cb_fit_predict 와 같은 초모수·같은 컨텍스트 행),
  cbr(CatBoost x25 + 지역 id), anchor(λ=0, Ê 만), physics(E0).
  TabPFN 가중치: 8.0.7 기본(v3)은 라이선스 토큰(TABPFN_TOKEN) 이 없으면 내려받을 수 없으므로 계획 §9 대체안대로 공개 v2
  회귀 가중치(HF Prior-Labs/TabPFN-v2-reg, tabpfn-v2-regressor.ckpt)를 쓴다. --model-path 로 바꿀 수 있다.
  TabICL 은 설치되어 있지 않으므로 시도하지 않고 메타에 기록만 한다.

채점(불변)
  Δ = 방법 − 물리식(E0), 음수 = 개선. 같은 셀·분할·반복으로 짝지음.
  지역 안 CI(주) = h4_common.boot_delta_blocks: 분할마다 채점 블록을 복원 추출(모든 방법·반복에 같은 인덱스), 반복 평균 후
    A − B, 분할별 분포를 같은 번호끼리 평균, 1,000회 percentile 95 %. 분할 승률·반복 승률 병기. 최소 n = h4_common.min_n.
  보조 ci_rep = (분할, 반복) 행 재표집 CI(h4_common.rep_boot_ci). 실행 단위별 블록 SSE 는 <tag>_blocksse.npz 에 저장.
  지역 평균(AB4·전체 14) = 층화 블록 부트스트랩(m1_stats.boot_delta/summarize_delta; 분할을 풀링하고 블록 = 분할:블록, S = 반복).
  셀 가중 RMSE(주)·블록 등가중 RMSE(보조) 병기, 채점 셀 수 저장. 가족(검정 이름)별 Holm.

입력  data/processed/fidelity_base_v3.csv + e5_soil_tdd_v3.csv (m1_core.load_base), 열 e5_sqrt_tdd(√TDD).
출력  data/processed/h4/<tag>_runs.csv(실행 행), <tag>_summary.csv(대상별 곡선·블록 부트스트랩 CI, 같은 내용을 <tag>_tfm.csv 로도 저장),
      <tag>_minn.csv(최소 n, 절단 표시), <tag>_tests.csv(지역 평균 층화 검정·Holm), <tag>_targets.csv(대상·분할 메타: σ²·τ²·E0·행 수·시간),
      <tag>_blocksse.npz(실행 단위 블록 SSE), <tag>_preds.npz(셀 예측), <tag>_meta.json. 스모크는 <tag>_smoke_*.
실행  python3 scripts/3_deep_learning/h35_tfm_fewshot.py --gpu 9 [--smoke] [--tag b4]
      본 실행: python3 scripts/3_deep_learning/h35_tfm_fewshot.py --gpu 9 --tag b4
"""
from __future__ import annotations
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="9", help="CUDA_VISIBLE_DEVICES 값(기본 9)")
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역, CA-1 제외)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,10,40")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--dup", type=int, default=3, help="대상 잔차 행 중복 배수")
ap.add_argument("--ctx-max", type=int, default=10_000, help="TabPFN 컨텍스트 총 행 상한")
ap.add_argument("--n-est", type=int, default=8, help="TabPFN n_estimators")
ap.add_argument("--model-path", default="tabpfn-v2-regressor.ckpt", help="TabPFN 가중치(기본 공개 v2 회귀)")
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--threads", type=int, default=4, help="CatBoost thread_count")
ap.add_argument("--tag", default="b4")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs·preds 로 집계만 다시")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json                                                                                          # noqa: E402
import subprocess                                                                                    # noqa: E402
import sys                                                                                           # noqa: E402
import time                                                                                          # noqa: E402
import warnings                                                                                      # noqa: E402
from pathlib import Path                                                                             # noqa: E402

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
from polar.m1_stats import boot_delta, summarize_delta, holm, seed_of                                # noqa: E402
from polar.h4_common import (BlockStore, save_stores, load_stores, stores_for_target, boot_delta_blocks,  # noqa: E402
                             rep_boot_ci, min_n, offset_mle_prior, offset_mle_estimate, offset_z)

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3]
REPS = args.reps if not args.smoke else 1
LAMS = [float(v) for v in args.lams.split(",")]
METHODS = ["tfm", "cb", "cbr"]
TAG = args.tag + ("_smoke" if args.smoke else "")

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


# ---------------------------------------------------------------- 하위 지역(h25 와 같은 정의: 블록 중심 k-means, 라벨 미사용)
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
_ref = PROC / "h3" / "subregions.csv"
if _ref.exists():                                          # h25 정의와 일치 확인(블록 수·셀 수)
    ref = pd.read_csv(_ref).set_index("subregion"); chk = SUBS.set_index("subregion")
    bad = [s for s in chk.index if s in ref.index and (ref.loc[s, "n_blocks"] != chk.loc[s, "n_blocks"] or ref.loc[s, "n_cells"] != chk.loc[s, "n_cells"])]
    assert not bad, f"하위 지역 정의가 h3/subregions.csv 와 다름: {bad}"
SUB_OK = [s for s in sorted(SUBS.subregion) if int(SUBS.set_index("subregion").loc[s, "n_cells"]) > 2 * min(N_GRID)]   # CA-1(6셀) 제외
TARGETS = args.targets.split(",") if args.targets else MAIN4 + SUB_OK
if args.smoke and not args.targets:
    TARGETS = ["Russia_W", "LE-2"]
MACRO_CODE = {m: i for i, m in enumerate(sorted(DF.macro.unique()))}
NEW_CODE = len(MACRO_CODE)
print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS} · GPU {args.gpu} · n {N_GRID} · 분할 {SPLITS} · 반복 {REPS}", flush=True)


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def store_key(method, n, rep, lam):
    """BlockStore 키 = (방법, n, 반복, λ). physics 는 (physics, 0, -1, 0.0), anchor 는 λ = 0.0."""
    return (str(method), int(n), int(rep), float(lam))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def select_kmedoid(n, rng, Z):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


# ---------------------------------------------------------------- log E 오프셋 최대우도(B2 (i)): h4_common 단일 구현 사용
OFFSET_MIN_CELLS = 3                                         # 공용 규약 주 분석(민감도 30 은 B2 담당)


# ---------------------------------------------------------------- 모델
_TFM = {}


def tfm_fit_predict(Xtr, ytr, Xte, seed):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device="cuda", random_state=seed, n_estimators=args.n_est, model_path=args.model_path,
                        ignore_pretraining_limits=True, categorical_features_indices=[Xtr.shape[1] - 1], memory_saving_mode=False)
    m.fit(Xtr, ytr)
    p = np.asarray(m.predict(Xte), float)
    if not _TFM:
        _TFM["model_path"] = str(getattr(m, "model_path", args.model_path)); _TFM["n_estimators"] = int(args.n_est)
    return p


def cb_fit_predict(Xtr, ytr, Xte, seed, cat_idx=None):
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    if cat_idx is None:
        m.fit(Xtr, ytr); return np.asarray(m.predict(Xte), float)
    def pool(X, y=None):
        d = pd.DataFrame(X); d[cat_idx] = d[cat_idx].astype(int)
        return Pool(d, y, cat_features=[cat_idx])
    m.fit(pool(Xtr, ytr)); return np.asarray(m.predict(pool(Xte)), float)


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t)
    parent = t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])
    src_idx = np.where(np.isfinite(DF.y.values))[0]; src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:
        la, lo = DF.lat.values, DF.lon.values; keep = np.ones(len(src_idx), bool); tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    prior = offset_mle_prior(src, region_col="macro", min_cells=OFFSET_MIN_CELLS, y_col="y", s_col="s", logE0=np.log(E0))
    sigma2, tau2, n_reg = prior["sigma2"], prior["tau2"], prior["n_regions"]
    # 원천 컨텍스트: 지역별 비례 층화 서브샘플(잔차 유효 행만)
    r_src_all = src.y.values - E0 * src.s.values; ok = np.isfinite(r_src_all) & np.isfinite(src.s.values)
    src = src[ok]; r_src_all = r_src_all[ok]
    n_ctx_src = min(len(src), args.ctx_max - args.dup * max(N_GRID))
    rng_sub = np.random.RandomState(seed_of("b4sub", t, sp))
    if n_ctx_src < len(src):
        mac = src.macro.values; cnt = pd.Series(mac).value_counts()
        take = {m: int(np.floor(n_ctx_src * c / len(src))) for m, c in cnt.items()}
        short = n_ctx_src - sum(take.values())
        for m in cnt.index[:short]:                          # 내림 나머지는 큰 지역부터 1행씩
            take[m] += 1
        pick = np.concatenate([rng_sub.choice(np.where(mac == m)[0], k, replace=False) for m, k in take.items() if k > 0])
        pick.sort()
    else:
        pick = np.arange(len(src))
    ctx = src.iloc[pick]
    X_ctx = np.c_[ctx[FEATS].values.astype(np.float32), np.array([MACRO_CODE[m] for m in ctx.macro.values], np.float32)]
    R_ctx = r_src_all[pick]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]; nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    XA_r = np.c_[XA, np.full(len(XA), NEW_CODE, np.float32)]; XB_r = np.c_[XB, np.full(len(XB), NEW_CODE, np.float32)]
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), n_ctx_src=int(len(pick)), E0=E0, sigma2=sigma2, tau2=tau2)
    rows, preds, timing = [], {f"eval::{t}|{sp}": dict(y=yB, block=B.block.values.astype(str), loc_id=B.loc_id.values)}, dict(tfm=0.0, cb=0.0, cbr=0.0)
    store = BlockStore(t, sp, B.block.values, meta=dict(n_A=int(nA), n_eval=int(len(evB)), E0=float(E0)))

    def add(method, n, rep, lam, pred, E_used, w=np.nan):
        rows.append(dict(**base, method=method, n=int(n), rep=int(rep), lam=float(lam), rmse_cm=rmse(yB, pred),
                         rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)), E_used=float(E_used),
                         w_shrink=float(w)))
        store.add(store_key(method, n, rep, lam), yB, pred)          # 키 = (방법, n, 반복, λ)

    add("physics", 0, -1, 0.0, E0 * sB, E0)
    preds[f"anchor::{t}|{sp}|0|-1"] = E0 * sB
    n_fit = 0
    for n in N_GRID:
        if n >= nA:
            continue
        for rep in range(REPS):
            rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * 1 + rep)      # h25 kmedoid(RULES.index=1) 와 같은 선택
            sel = select_kmedoid(n, rng, Z)
            post = offset_mle_estimate(offset_z(A.iloc[sel], y_col="y", s_col="s"), prior)
            E_hat, w_hat = post["E"], post["w"]
            r_n = yA[sel] - E_hat * sA[sel]; okn = np.isfinite(r_n); sel, r_n = sel[okn], r_n[okn]
            rep_idx = np.repeat(sel, args.dup)
            Xtr_r = np.vstack([X_ctx, XA_r[rep_idx]]); ytr = np.concatenate([R_ctx, np.repeat(r_n, args.dup)])
            Xtr = Xtr_r[:, :-1]
            seed = seed_of("b4", t, sp, n, rep)
            anchor = E_hat * sB
            add("anchor", n, rep, 0.0, anchor, E_hat, w_hat)
            preds[f"anchor::{t}|{sp}|{n}|{rep}"] = anchor
            g = {}
            t1 = time.time(); g["tfm"] = tfm_fit_predict(Xtr_r, ytr, XB_r, seed); timing["tfm"] += time.time() - t1
            t1 = time.time(); g["cb"] = cb_fit_predict(Xtr, ytr, XB, seed); timing["cb"] += time.time() - t1
            t1 = time.time(); g["cbr"] = cb_fit_predict(Xtr_r, ytr, XB_r, seed, cat_idx=Xtr_r.shape[1] - 1); timing["cbr"] += time.time() - t1
            n_fit += 3
            for mth in METHODS:
                preds[f"g::{t}|{sp}|{n}|{rep}|{mth}"] = g[mth]
                for lam in LAMS:
                    add(mth, n, rep, lam, anchor + lam * g[mth], E_hat, w_hat)
    meta = dict(target=t, parent=parent, split=sp, n_A=nA, n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), n_ctx_src=int(len(pick)),
                E0=E0, E_own_A=ls_E(yA, sA), sigma2=sigma2, tau2=tau2, tau2_raw=prior["tau2_raw"], logE0_prior=prior["logE0"],
                n_prior_regions=n_reg, prior_regions=";".join(prior["regions"]), n_fit=n_fit,
                t_tfm_s=round(timing["tfm"], 1), t_cb_s=round(timing["cb"], 1), t_cbr_s=round(timing["cbr"], 1), elapsed_s=round(time.time() - t0, 1))
    return rows, preds, meta, store


# ---------------------------------------------------------------- 집계
def stores_from_preds(preds):
    """저장된 preds(eval y·block, anchor, g)로 실행 단위 BlockStore 를 다시 만든다(blocksse 파일이 없을 때의 재집계 경로)."""
    stores = {}
    for ek, e in preds.items():
        if not ek.startswith("eval::"):
            continue
        t, sp = ek[len("eval::"):].split("|"); sp = int(sp)
        st = BlockStore(t, sp, e["block"]); y = e["y"]
        st.add(store_key("physics", 0, -1, 0.0), y, preds[f"anchor::{t}|{sp}|0|-1"])
        for k, v in preds.items():
            if k.startswith(f"anchor::{t}|{sp}|") and not k.endswith("|0|-1"):
                n, rp = k.split("|")[2:4]; st.add(store_key("anchor", n, rp, 0.0), y, v)
            elif k.startswith(f"g::{t}|{sp}|"):
                n, rp, mth = k.split("|")[2:5]
                a = preds[f"anchor::{t}|{sp}|{n}|{rp}"]
                for lam in LAMS:
                    st.add(store_key(mth, n, rp, lam), y, a + lam * v)
        stores[(t, sp)] = st
    return stores


def summarize(runs, stores):
    """대상 × 방법 × λ × n 곡선. 주 CI = 분할 안 블록 부트스트랩(h4_common), 보조 ci_rep = (분할, 반복) 행 재표집.

    한 대상 안에서는 모든 비교가 같은 부트스트랩 seed(= seed_of("b4blk", 대상))를 써서 블록 재표집 인덱스를 공유한다.
    """
    r = runs[runs.method != "physics"].copy()
    phys = runs[runs.method == "physics"].set_index(["target", "split"]).rmse_cm
    r["phys"] = [phys.loc[(t, s)] for t, s in zip(r.target, r.split)]
    r["d_phys"] = r.rmse_cm - r.phys; r["win"] = (r.rmse_cm < r.phys).astype(float)
    cb = r[r.method == "cb"].set_index(["target", "split", "n", "rep", "lam"]).rmse_cm
    r["d_cb"] = [rmse_ - cb.get((t, s, n, rp, lam), np.nan) if m == "tfm" else np.nan
                 for t, s, n, rp, lam, m, rmse_ in zip(r.target, r.split, r.n, r.rep, r.lam, r.method, r.rmse_cm)]
    keys = ["target", "parent", "method", "lam", "n"]
    out = r.groupby(keys, as_index=False).agg(rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_beq_mean=("rmse_beq_cm", "mean"),
                                              phys_mean=("phys", "mean"), d_phys_mean=("d_phys", "mean"), d_cb_mean=("d_cb", "mean"),
                                              win_rate=("win", "mean"), n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"),
                                              n_eval_total=("n_eval", "sum"), E_used_mean=("E_used", "mean"), w_shrink_mean=("w_shrink", "mean"))
    by_t = {t: stores_for_target(stores, t) for t in out.target.unique()}
    rep_fn = lambda k: k[2]                                                                           # noqa: E731
    recs = []
    for _, row in out.iterrows():
        t, m, lam, n = row.target, row.method, float(row.lam), int(row.n)
        bseed = seed_of("b4blk", t)
        fA = lambda k, m=m, n=n, lam=lam: k[0] == m and k[1] == n and abs(k[3] - lam) < 1e-9        # noqa: E731
        rp = boot_delta_blocks(by_t[t], fA, lambda k: k[0] == "physics", nboot=1000, seed=bseed, rep_fn=rep_fn)
        sub = r[(r.target == t) & (r.method == m) & (r.lam == lam) & (r.n == n)]
        cr = rep_boot_ci(sub.d_phys.values)
        rec = dict(d_phys_blk=rp["delta"], d_phys_lo=rp["ci_lo"], d_phys_hi=rp["ci_hi"], p_boot=rp["p_boot"],
                   split_win=rp["split_win"], rep_win=rp["rep_win"], d_phys_beq=rp["delta_beq"], d_phys_lo_beq=rp["ci_lo_beq"],
                   d_phys_hi_beq=rp["ci_hi_beq"], ci_flag=rp["ci_flag"], n_blocks_min=min(rp["n_blocks"]) if rp["n_blocks"] else 0,
                   ci_rep_lo=cr[0], ci_rep_hi=cr[1])
        for ref in ("cb", "cbr"):
            if m == "tfm":
                fB = lambda k, ref=ref, n=n, lam=lam: k[0] == ref and k[1] == n and abs(k[3] - lam) < 1e-9  # noqa: E731
                rc = boot_delta_blocks(by_t[t], fA, fB, nboot=1000, seed=bseed, rep_fn=rep_fn)
                cbs = r[(r.target == t) & (r.method == ref) & (r.lam == lam) & (r.n == n)].set_index(["split", "rep"]).rmse_cm
                d_rows = sub.set_index(["split", "rep"]).rmse_cm - cbs
                crc = rep_boot_ci(d_rows.values)
                rec.update({f"d_{ref}_blk": rc["delta"], f"d_{ref}_lo": rc["ci_lo"], f"d_{ref}_hi": rc["ci_hi"], f"d_{ref}_p": rc["p_boot"],
                            f"d_{ref}_split_win": rc["split_win"], f"d_{ref}_rep_win": rc["rep_win"],
                            f"d_{ref}_ci_rep_lo": crc[0], f"d_{ref}_ci_rep_hi": crc[1]})
            else:
                rec.update({f"d_{ref}_{c}": np.nan for c in ("blk", "lo", "hi", "p", "split_win", "rep_win", "ci_rep_lo", "ci_rep_hi")})
        recs.append(rec)
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    return out


def region_tests(runs, preds, targets):
    """지역 평균 검정: 분할 풀링(블록 = 분할:블록), S = 반복. 가족 = 검정 이름(n × λ 에 Holm)."""
    PAIRS = {"tfm_vs_cb": ("tfm", "cb"), "tfm_vs_cbr": ("tfm", "cbr"), "tfm_vs_phys": ("tfm", "physics"), "cb_vs_phys": ("cb", "physics"),
             "cbr_vs_phys": ("cbr", "physics"), "anchor_vs_phys": ("anchor", "physics")}
    ab4 = [t for t in MAIN4 if t in targets]
    rows = []
    for n in sorted(set(runs.n[runs.n > 0])):
        for lam in LAMS:
            for name, (ma, mb) in PAIRS.items():
                if ma == "anchor" and lam != LAMS[0]:
                    continue
                per = {}
                for t in targets:
                    ys, bl, PA, PB = [], [], [], []
                    for sp in SPLITS:
                        ek = f"eval::{t}|{sp}"
                        if ek not in preds:
                            continue
                        reps = sorted({int(k.split("|")[3]) for k in preds if k.startswith(f"g::{t}|{sp}|{n}|")})
                        if not reps:
                            continue
                        e = preds[ek]; ys.append(e["y"]); bl.append(np.char.add(f"{sp}:", e["block"].astype(str)))
                        def pred_of(m, rp):
                            if m == "physics":
                                return preds[f"anchor::{t}|{sp}|0|-1"]
                            a = preds[f"anchor::{t}|{sp}|{n}|{rp}"]
                            return a if m == "anchor" else a + lam * preds[f"g::{t}|{sp}|{n}|{rp}|{m}"]
                        PA.append(np.stack([pred_of(ma, rp) for rp in reps])); PB.append(np.stack([pred_of(mb, rp) for rp in reps]))
                    if not ys:
                        continue
                    S = min(p.shape[0] for p in PA)
                    per[t] = boot_delta(np.concatenate(ys), np.concatenate(bl), np.hstack([p[:S] for p in PA]), np.hstack([p[:S] for p in PB]),
                                        nboot=1000, seed=seed_of("b4boot", name, n, lam))
                for scope, regs in (("AB4", ab4), ("ALL", targets)):
                    for rr in summarize_delta(f"{name}|n{n}|lam{lam}", per, regs, seed=0):
                        is_mean = rr["target"].startswith("MEAN[")
                        if scope == "AB4" and not is_mean:
                            continue                          # 지역 행은 ALL 에서 한 번만
                        rr.update(test_name=name, n=int(n), lam=float(lam), scope=scope if is_mean else "region")
                        rows.append(rr)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(subset=["test", "target"])
        for (name, scope), sub in df[df.scope != "region"].groupby(["test_name", "scope"]):
            df.loc[sub.index, "p_holm"] = holm(sub.p_boot.values)
    return df


def verdict(tests):
    if not len(tests):
        return "판정 불가(행 없음)"
    m = tests[(tests.test_name == "tfm_vs_cb") & (tests.scope == "AB4")]
    lines = []
    for _, r in m.iterrows():
        v = "동급(CI 0 포함)" if (np.isfinite(r.ci_lo) and r.ci_lo <= 0 <= r.ci_hi) else ("기각(하한 > 0)" if r.ci_lo > 0 else ("TFM 우위(상한 < 0)" if r.ci_hi < 0 else "CI 없음"))
        lines.append(f"F8 n={r.n} λ={r.lam}: Δ(TFM−CB)={r.delta:+.2f} cm [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p_holm={r.p_holm:.3f} · {v}")
    return "\n".join(lines)


def save_all(runs, preds, stores, metas, elapsed, t_start):
    runs.to_csv(OUT / f"{TAG}_runs.csv", index=False)
    if metas is not None:                                    # 새 실행일 때만 예측·블록 SSE 저장(재집계는 읽기만)
        np.savez_compressed(OUT / f"{TAG}_preds.npz", **{k: (v if not isinstance(v, dict) else np.array([json.dumps(dict(y=v["y"].tolist(), block=v["block"].tolist(), loc_id=[int(x) for x in v["loc_id"]]))])) for k, v in preds.items()})
        save_stores(list(stores.values()), OUT / f"{TAG}_blocksse.npz")
    summ = summarize(runs, stores); summ.to_csv(OUT / f"{TAG}_summary.csv", index=False); summ.to_csv(OUT / f"{TAG}_tfm.csv", index=False)
    minn = min_n(summ, group_cols=["target", "parent", "method", "lam"], ci_hi_col="d_phys_hi")
    minn.to_csv(OUT / f"{TAG}_minn.csv", index=False)
    tests = region_tests(runs, preds, [t for t in TARGETS if (runs.target == t).any()]); tests.to_csv(OUT / f"{TAG}_tests.csv", index=False)
    if metas is not None:
        pd.DataFrame(metas).to_csv(OUT / f"{TAG}_targets.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    try:
        import tabpfn; tabpfn_ver = tabpfn.__version__
    except Exception:                                                                                 # noqa: BLE001
        tabpfn_ver = "NA"
    meta = dict(stage="H35/B4", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 B4, F8", args=vars(args), targets=TARGETS, splits=SPLITS,
                n_grid=N_GRID, reps=REPS, lams=LAMS, methods=METHODS + ["anchor", "physics"], gpu=str(args.gpu),
                tabpfn=dict(version=tabpfn_ver, model_path=_TFM.get("model_path", args.model_path), n_estimators=args.n_est,
                            note="8.0.7 기본 v3 가중치는 TABPFN_TOKEN 라이선스 토큰 없이는 내려받을 수 없어 계획 §9 대체안(공개 v2 회귀 가중치)을 사용"),
                tabicl=dict(installed=False, attempted=False, note="설치되어 있지 않아 시도하지 않음(지시)"),
                anchor=f"log E 오프셋 최대우도(h4_common.offset_mle_prior/estimate, 원천 macro 셀 ≥ {OFFSET_MIN_CELLS}, 적률 τ², 사전 평균 log E0 원천 최소제곱)",
                ci="지역 안: h4_common.boot_delta_blocks(분할 안 블록 재표집, 대상별 공유 seed, 반복 짝지음), 보조 ci_rep = (분할, 반복) 행 재표집; "
                   "지역 평균: m1_stats.boot_delta/summarize_delta 층화 블록 부트스트랩",
                git_commit=commit, started=t_start, elapsed_s=round(elapsed, 1), n_rows=int(len(runs)), n_summary_rows=int(len(summ)), n_test_rows=int(len(tests)),
                verdict=verdict(tests))
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    return summ, tests


def load_preds(path):
    z = np.load(path, allow_pickle=False); out = {}
    for k in z.files:
        if k.startswith("eval::"):
            d = json.loads(str(z[k][0])); out[k] = dict(y=np.array(d["y"], float), block=np.array(d["block"], str), loc_id=np.array(d["loc_id"]))
        else:
            out[k] = z[k]
    return out


def main():
    t0 = time.time(); t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.summarize_only:
        runs = pd.read_csv(OUT / f"{TAG}_runs.csv"); preds = load_preds(OUT / f"{TAG}_preds.npz")
        if "w_shrink" not in runs:
            runs["w_shrink"] = np.nan
        bs = OUT / f"{TAG}_blocksse.npz"
        stores = load_stores(bs) if bs.exists() else stores_from_preds(preds)
        print(f"블록 SSE: {'파일 ' + bs.name if bs.exists() else 'preds 에서 재구성'} · 단위 {len(stores)}", flush=True)
        summ, tests = save_all(runs, preds, stores, None, time.time() - t0, t_start)
        print(verdict(tests)); print(f"summarized · 행 {len(runs)}")
        return
    tasks = [(t, sp) for t in TARGETS for sp in SPLITS]
    rows, preds, metas, stores = [], {}, [], {}
    for k, (t, sp) in enumerate(tasks):
        r, p, m, st = run_task(t, sp)
        rows.extend(r); preds.update(p); metas.append(m); stores[(t, sp)] = st
        print(f"  [{k+1}/{len(tasks)} {t}|{sp}] 적합 {m['n_fit']} · A {m['n_A']} · 채점 {m['n_eval']} · 컨텍스트 {m['n_ctx_src']} · E0 {m['E0']:.3f} "
              f"· σ² {m['sigma2']:.3f} τ² {m['tau2']:.4f} · tfm {m['t_tfm_s']}s cb {m['t_cb_s']}s cbr {m['t_cbr_s']}s · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s", flush=True)
    runs = pd.DataFrame(rows)
    summ, tests = save_all(runs, preds, stores, metas, time.time() - t0, t_start)
    show = summ[summ.method.isin(["tfm", "cb", "anchor"])][["target", "method", "lam", "n", "rmse_mean", "phys_mean", "d_phys_mean", "d_phys_lo", "d_phys_hi",
                                                            "split_win", "rep_win", "d_cb_mean", "d_cb_lo", "d_cb_hi"]]
    print(show.round(2).to_string(index=False))
    print(verdict(tests))
    print(f"saved {TAG}_* · runs 행 {len(runs)} · summary 행 {len(summ)} · tests 행 {len(tests)} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
