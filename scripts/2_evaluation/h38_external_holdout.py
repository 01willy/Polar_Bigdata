"""H38 Track D1 외부 홀드아웃(몽골·중앙아시아) 확인적 검정 + D2 GTNP 신규 지점 점검. CPU.

계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3 공통 설계, §4 Track D1·D2, §5 F11, §8 산출물.

목적
  D1: 지금까지 어떤 실험에도 쓰지 않은 몽골·중앙아시아(CALM 46셀·21블록, source_id F4_calm_temp)에
      2단계 프로토콜(B1)을 한 번만 적용하여 사전 예측과 대조한다. 순서가 중요하다.
      1단계(라벨 열람 금지): 공변량만 읽어 라벨 미사용 이질성 지표(SMD·분산비·AOA DI)와 사전 예측 문구를
                            d1_prediction.json 에 먼저 기록하고 파일 mtime 을 남긴다. 기존 파일이 있으면 덮어쓰지 않는다.
      2단계(라벨 열람): h25 규약(원천 = 대상 외 F4_direct 전 라벨, 100 km 버퍼, 분할 3, A/B, 채점 B eval_mask)으로
                        (i) 물리식 E0, (ii) 대표점 3개(kmedoid) E_hat·r_hat = |log(E_hat/E0)| 20회,
                        (iii) 2단계 프로토콜(r_hat >= tau 이면 kappa=10 수축 E 단독, 아니면 E0 고정 + n개 라벨 잔차 CatBoost lambda),
                        (iv) 비교: 항상 수축+잔차(S3), 항상 E0 고정+잔차, 라벨 전량 E 재적합(+ 전량 수축, 원천 잔차만, 오라클 분기).
      3단계(D2): gtnp_alt_inventory.csv 의 status == new_coord 5지점을 v3 셀·블록과 대조한 표만 만든다(v3 수정 없음).
                 현재 평가 셀 수는 load_base(좌표 2차 병합 포함)+eval_mask 로 세고, h25 규약의 분할 0–2 B 채점 셀 수를 함께 기록한다.
사전 등록 순서 공개(검증자 지적 반영)
  개발 단계 스모크(--tag d1 --smoke, 2026-09-26 13:15:11 사전 예측 기록, 13:15:12 최초 라벨 열람)와 병렬 점검(d1test, 삭제)이
  몽골 46셀의 실측 결과(E_allA 약 7.5, r 약 1.55, 수축 분기)를 본 실행 전에 산출·열람했다. 본 실행의 사전 예측 파일
  d1_prediction.json 은 그 스모크 1단계 파일을 mtime 보존 복사(cp -p)한 것이며, 사전 예측 문구는 스모크 이전에 스크립트에
  적혀 있던 두 범주 그대로이다. meta 의 prereg_disclosure 에 이 사실을 기록하고 원고 방법 절에 한 문장으로 적는다.
  이후 스모크는 몽골 라벨을 열지 않도록 기본 대상을 이미 열린 지역(--smoke-target, 기본 Russia_W)으로 두며, 대상이 기본
  (Mongolia_CAsia)과 다르면 산출 접두어에 대상 이름을 붙인다(d1_Russia_W_smoke_*).
채점 규약
  Delta = 방법 - 물리식(E0), 음수 = 개선. 같은 셀·분할·반복·seed 짝지음. 지역 안 CI(주) = h4_common 블록 부트스트랩(분할 안 B 블록 재표집, 반복·seed 평균,
  분할 분포 평균) 1000회 percentile 95 %. (분할, 반복) 부트스트랩은 *_rep 열로 보조 병기(09-26 감사: 채점 블록 분산 누락).
  offset_only(B2 log E 오프셋 최대우도, h4_common) 행은 결과 열람 후 추가한 탐색 행이며 F11 판정에 쓰지 않는다.
  셀 가중 RMSE(주)와 블록 등가중 RMSE(보조) 병기. 절대 RMSE 는 채점 셀 수(n_eval)와 함께 저장.
입력
  data/processed/fidelity_base_v3.csv, e5_soil_tdd_v3.csv (polar.m1_core.load_base 병합), gtnp_alt_inventory.csv,
  m1/a2_shift_diagnostics.csv (있으면 알래스카 기준 DI 를 사전 예측에 인용).
출력(data/processed/h4/)
  <tag>_prediction.json, <tag>_runs.csv, <tag>_summary.csv, <tag>_branch.csv, d2_gtnp_new.csv(스모크는 d2_smoke_gtnp_new.csv), <tag>_meta.json
실행(ROOT)
  스모크: python3 scripts/2_evaluation/h38_external_holdout.py --smoke --workers 1   (대상 Russia_W, 산출 d1_Russia_W_smoke_*)
  본:     python3 scripts/2_evaluation/h38_external_holdout.py --workers 3
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import datetime as dt
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, BLOCK_DEG, macro_region, spatial_block_splits                    # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
from polar.h4_common import BlockStore, boot_delta_blocks, save_stores, load_stores, offset_mle_prior, offset_mle_estimate, offset_z   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target", default=None, help="대상 매크로 지역(기본 Mongolia_CAsia, --smoke 이면 --smoke-target)")
ap.add_argument("--smoke-target", default="Russia_W", help="스모크 기본 대상(이미 열린 지역, 몽골 라벨을 열지 않기 위함)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--reps", type=int, default=20, help="대표점 추출 반복(CatBoost 포함)")
ap.add_argument("--seeds", type=int, default=2, help="CatBoost seed 수")
ap.add_argument("--budgets", default="3,10", help="라벨 예산 n")
ap.add_argument("--taus", default="0.10,0.15,0.20")
ap.add_argument("--tau-main", type=float, default=0.15)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--alpha", type=float, default=1.0, help="잔차 CatBoost 대상 행 가중")
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="d1")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--force-prediction", action="store_true", help="기존 사전 예측 파일을 덮어쓴다(원칙적으로 금지)")
ap.add_argument("--skip-d2", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
MAIN_TARGET = "Mongolia_CAsia"
TGT = args.target or (args.smoke_target if args.smoke else MAIN_TARGET)
TAG = args.tag + (f"_{TGT}" if TGT != MAIN_TARGET else "") + ("_smoke" if args.smoke else "")
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
REPS = args.reps if not args.smoke else 2
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
BUDGETS = [int(v) for v in args.budgets.split(",")]
TAUS = [float(v) for v in args.taus.split(",")]
SRC_SOURCES = ("F4_direct",)                     # 원천 = 기존 실험과 같은 F4_direct 라벨(대상 제외)
TGT_SOURCES = ("F4_direct", "F4_calm_temp")      # 대상(몽골)은 F4_calm_temp 이므로 로드 시 포함
T_ALL0 = time.time()
PREREG_DISCLOSURE = dict(
    smoke_results_viewed_before_full_run=True,
    smoke_first_prediction_written="2026-09-26T13:15:11.927 (d1_smoke_prediction.json, --tag d1 --smoke, 대상 Mongolia_CAsia)",
    smoke_first_label_read="2026-09-26T13:15:12 (같은 스모크 실행의 2단계)",
    parallel_check="--tag d1test(분할 3, 반복 1, 13:15:54) 실행 후 삭제, 실측 E_allA 7.20–7.75 열람",
    prediction_source="smoke stage1 file copied with cp -p (mtime 13:15:11.93 < first label read 13:15:12); 문구는 스모크 이전 스크립트 고정 두 범주",
    manuscript_note="방법 절: 몽골 홀드아웃 결과는 개발 스모크에서 본 실행 전에 한 차례 열람되었고, 사전 예측 문구는 그 이전에 고정되었다.",
)


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ================================================================ 1단계: 라벨 미사용 사전 예측
def source_index(df, t_idx):
    """원천 = F4_direct 이고 대상이 아닌 라벨 셀, 대상 경계 buffer_km 이내 제외(좌표만 사용)."""
    src_idx = np.where(np.isin(df.source_id.values, SRC_SOURCES) & (df.macro.values != TGT))[0]
    if args.buffer_km > 0:
        la, lo = df.lat.values, df.lon.values; tl, tn = la[t_idx], lo[t_idx]
        keep = np.ones(len(src_idx), bool)
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    return src_idx


def aoa_di_source(df, src_idx, t_idx, n_folds=6):
    """Meyer & Pebesma(2021) DI(가중 없음). 원천 표준화, 원천 6-fold 공간블록 CV 최근접 거리 평균으로 정규화,
    임계 = 원천 DI 의 Q3 + 1.5 IQR. 라벨 미사용."""
    from sklearn.neighbors import NearestNeighbors
    X = df[FEATS].values.astype(float); Xs = X[src_idx]
    med = np.nanmedian(Xs, 0); med = np.where(np.isfinite(med), med, 0.0); X = np.where(np.isnan(X), med, X)
    mu, sd = X[src_idx].mean(0), X[src_idx].std(0) + 1e-6; Z = (X - mu) / sd
    d_cv = np.full(len(src_idx), np.nan); pos = {g: i for i, g in enumerate(src_idx)}
    for tr, te in spatial_block_splits(df, n_splits=n_folds, sub_idx=src_idx):
        d, _ = NearestNeighbors(n_neighbors=1).fit(Z[tr]).kneighbors(Z[te]); d_cv[[pos[i] for i in te]] = d[:, 0]
    dbar = float(np.nanmean(d_cv)); di_tr = d_cv / dbar
    q75, q25 = np.nanpercentile(di_tr, [75, 25]); thr = float(q75 + 1.5 * (q75 - q25))
    d_t, _ = NearestNeighbors(n_neighbors=1).fit(Z[src_idx]).kneighbors(Z[t_idx]); di_t = d_t[:, 0] / dbar
    return di_t, thr


def label_free_metrics(df, t_idx, src_idx):
    Xt = df.iloc[t_idx][FEATS].values.astype(float); Xs = df.iloc[src_idx][FEATS].values.astype(float)
    mt, ms = np.nanmean(Xt, 0), np.nanmean(Xs, 0); sd_s = np.nanstd(Xs, 0) + 1e-9
    smd_var = np.abs(mt - ms) / sd_s
    di_t, thr = aoa_di_source(df, src_idx, t_idx)
    ev = (df.iloc[t_idx].cci_alt.notna()).values                       # 채점 셀 조건 중 라벨 무관 부분(CCI 유효)
    return dict(n_cells=int(len(t_idx)), n_blocks=int(df.iloc[t_idx].block.nunique()), n_cells_cci_valid=int(ev.sum()),
                n_src=int(len(src_idx)), smd_x25=float(smd_var.mean()), smd_x25_max_var=FEATS[int(np.argmax(smd_var))],
                smd_x25_max=float(smd_var.max()), var_ratio_x25=float(np.nanmean(np.nanvar(Xt, 0) / (np.nanvar(Xs, 0) + 1e-9))),
                di_src_median=float(np.median(di_t)), di_src_q3=float(np.percentile(di_t, 75)), di_src_thr=thr,
                out_aoa_src=float(np.mean(di_t > thr)), lat_mean=float(df.lat.values[t_idx].mean()), lon_mean=float(df.lon.values[t_idx].mean()),
                sqrt_tdd_mean=float(np.nanmean(df.e5_sqrt_tdd.values[t_idx])), sqrt_tdd_src_mean=float(np.nanmean(df.e5_sqrt_tdd.values[src_idx])))


def stage1_prediction():
    """라벨 열(alt_cm)을 읽지 않고 사전 예측 파일을 쓴다. 기존 파일은 보존(mtime 유지)."""
    pred_path = OUT / f"{TAG}_prediction.json"
    if pred_path.exists() and not args.force_prediction:
        prev = json.loads(pred_path.read_text())
        assert prev.get("target") == TGT, f"기존 사전 예측 파일의 대상({prev.get('target')})이 현재 대상({TGT})과 다르다"
        log(f"[1단계] 기존 사전 예측 보존: {pred_path} (mtime {dt.datetime.fromtimestamp(pred_path.stat().st_mtime)})")
        return pred_path, prev, True
    cols = ["loc_id", "lat", "lon", "region", "source_id"] + [c for c in FEATS if c != "cci_alt"] + ["cci_alt"]
    df = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=lambda c: c in cols, low_memory=False)
    assert TARGET not in df.columns, "1단계는 라벨 열을 읽지 않는다"
    df["macro"] = macro_region(df)
    df["block"] = (np.floor(df.lat / BLOCK_DEG).astype(int) * 100000 + np.floor(df.lon / BLOCK_DEG).astype(int))
    df = df[df.source_id.isin(TGT_SOURCES)].reset_index(drop=True)
    t_idx = np.where(df.macro.values == TGT)[0]
    src_idx = source_index(df, t_idx)
    met = label_free_metrics(df, t_idx, src_idx)
    ak = {}
    f_a2 = PROC / "m1" / "a2_shift_diagnostics.csv"
    if f_a2.exists():
        a2 = pd.read_csv(f_a2); r = a2[a2.target == TGT]
        if len(r):
            ak = {k: float(r[k].iloc[0]) for k in ("di_x25_median", "di_x25_q3", "out_aoa_x25") if k in r}
    pred = dict(
        track="D1", target=TGT, plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 D1, §5 F11",
        written_at=dt.datetime.now().isoformat(timespec="seconds"), labels_read=False,
        label_free_metrics=met, alaska_reference_di_from_a2=ak,
        hypothesis_F11="외부 홀드아웃에서 사전 예측한 분기·손익분기 n 범주가 실측과 일치하고, 최종 프로토콜 Δ ≤ 0",
        prediction=[
            "E비 진단(수준 오차 대 구조 오차)은 대표점 3개 라벨로만 가능하므로, 라벨 미사용 지표(SMD·분산비·DI)로는 분기 범주를 맞힐 수 없다"
            "(H27: 라벨 미사용 특징의 LOO R² < 0, E_only 만 유의).",
            "2단계 프로토콜(주 τ=0.15, n=3·10, kmedoid)의 물리식(E0) 대비 Δ(셀 가중 RMSE) ≤ 0 이며, 항상 수축+잔차(S3)보다 악화 반복 수가 적거나 같다.",
        ],
        expected_categories=dict(branch="라벨 미사용 지표로는 판정 불가(사전 예측 없음)", protocol_delta="Δ ≤ 0(점 추정), CI 상한 < 0 이면 확정 개선"),
        decision_rule=dict(tau_main=args.tau_main, taus=TAUS, kappa=args.kappa, lam=args.lam, alpha=args.alpha, budgets=BUDGETS, reps=REPS,
                           selection="kmedoid(표준화 x25 k-means 중심 최근접 셀, A 블록 풀)", scoring="B 블록 eval_mask, Δ = 방법 − E0, (분할, 반복) 짝지음 부트스트랩 1000회"),
        note="라벨 미사용 지표가 원천 대비 극단(AOA 밖 비율 1.0 수준)이면 공변량 외삽 자체가 강하므로 잔차 ML 분기의 이득은 제한적일 것으로 본다(방향만, 수치 예측 없음).",
    )
    pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=1, default=float))
    log(f"[1단계] 사전 예측 기록: {pred_path} · 셀 {met['n_cells']} · 블록 {met['n_blocks']} · SMD {met['smd_x25']:.2f} · DI 중앙값 {met['di_src_median']:.2f} · AOA 밖 {met['out_aoa_src']:.2f}")
    return pred_path, pred, False


PRED_PATH, PRED, PRED_PREEXISTING = stage1_prediction()
PRED_MTIME = dt.datetime.fromtimestamp(PRED_PATH.stat().st_mtime).isoformat(timespec="microseconds")

# ================================================================ 2단계: 라벨 열람(여기서부터 alt_cm 사용)
LABEL_READ_AT = dt.datetime.now().isoformat(timespec="microseconds")
DF = load_base(PROC, sources=TGT_SOURCES)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
T_IDX = np.where(DF.macro.values == TGT)[0]
log(f"[2단계] 라벨 열람 {LABEL_READ_AT} (사전 예측 mtime {PRED_MTIME}) · 전체 {len(DF):,}셀 · 대상 {len(T_IDX)}셀/{DF.iloc[T_IDX].block.nunique()}블록")


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


def kmedoid(n, rng, Z):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def run_task(sp):
    t0 = time.time()
    src_idx = source_index(DF, T_IDX); src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); r_src = src.y.values - E0 * src.s.values
    ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, T_IDX, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]; nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    E_allA = ls_E(yA, sA); E_own = ls_E(DF.y.values[T_IDX], DF.s.values[T_IDX])
    base = dict(target=TGT, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), E0=E0, E_allA=E_allA, r_allA=abs(np.log(E_allA / E0)), E_own=E_own, r_own=abs(np.log(E_own / E0)))
    rows, n_fit = [], 0
    store = BlockStore(TGT, sp, B.block.values, meta=dict(E0=E0, n_eval=len(evB)))
    prior = offset_mle_prior(src.assign(y=src.y.values, s=src.s.values), region_col="macro", min_cells=3, logE0=float(np.log(E0)))

    def add(method, n, rep, seed, tau, pred, E_used, E_hat3=np.nan, r_hat=np.nan, branch=""):
        store.add((method, int(n), int(rep), int(seed), float(tau)), yB, pred)
        rows.append(dict(**base, method=method, n=int(n), rep=int(rep), seed=int(seed), tau=float(tau), E_hat3=float(E_hat3), r_hat=float(r_hat),
                         branch=branch, E_used=float(E_used), rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB),
                         bias_cm=float(np.mean(pred - yB))))

    def resid(sel, E_anchor, seed):
        """원천 잔차(E0 앵커) ∪ 대상 sel 잔차(E_anchor 앵커, 가중 alpha) → CatBoost → B 잔차 예측."""
        if sel is None or len(sel) == 0:
            return cb_fit_predict(XR_src, R_src, XB, seed)
        r_n = yA[sel] - E_anchor * sA[sel]
        w = np.concatenate([np.ones(len(R_src)), np.full(len(sel), args.alpha)])
        return cb_fit_predict(np.vstack([XR_src, XA[sel]]), np.concatenate([R_src, r_n]), XB, seed, w=w)

    # 기준: 물리식 E0, 원천 잔차만(n=0), 라벨 전량 재적합·수축
    add("physics", 0, -1, -1, -1, E0 * sB, E0)
    for seed in SEEDS:
        g = resid(None, E0, seed); n_fit += 1; add("resid_src_only", 0, -1, seed, -1, E0 * sB + args.lam * g, E0)
    add("refit_allA", nA, -1, -1, -1, E_allA * sB, E_allA)
    E_shr_all = (nA * E_allA + args.kappa * E0) / (nA + args.kappa)
    add("shrink_allA", nA, -1, -1, -1, E_shr_all * sB, E_shr_all)
    # 반복: 대표점 3개 → r_hat → 예산 n 별 분기
    for rep in range(REPS):
        rng = np.random.RandomState(7_919 * sp + 101 * rep + 1)
        sel3 = kmedoid(3, rng, Z); E3 = ls_E(yA[sel3], sA[sel3]); r_hat = abs(np.log(E3 / E0))
        for n in BUDGETS:
            if n >= nA:
                continue
            sel = sel3 if n == 3 else kmedoid(n, rng, Z)
            E_ls = ls_E(yA[sel], sA[sel]); E_n = (n * E_ls + args.kappa * E0) / (n + args.kappa)
            p_shr = E_n * sB
            add("shrink_only", n, rep, -1, -1, p_shr, E_n, E3, r_hat)
            add("refit_only", n, rep, -1, -1, E_ls * sB, E_ls, E3, r_hat)
            E_off = offset_mle_estimate(offset_z(A.iloc[sel].assign(y=yA[sel], s=sA[sel])), prior)["E"]   # 탐색(사전 등록 F11 판정 밖): B2 적응형 수축
            add("offset_only", n, rep, -1, -1, E_off * sB, E_off, E3, r_hat)
            for seed in SEEDS:
                g_fix = resid(sel, E0, seed); n_fit += 1; p_fix = E0 * sB + args.lam * g_fix
                g_shr = resid(sel, E_n, seed); n_fit += 1; p_s3 = E_n * sB + args.lam * g_shr
                add("e0fix_resid", n, rep, seed, -1, p_fix, E0, E3, r_hat)
                add("shrink_resid_S3", n, rep, seed, -1, p_s3, E_n, E3, r_hat)
                for tau in TAUS:
                    br = "shrink" if r_hat >= tau else "resid"
                    add("protocol", n, rep, seed, tau, p_shr if br == "shrink" else p_fix, E_n if br == "shrink" else E0, E3, r_hat, br)
                orc = "shrink" if rmse(yB, p_shr) <= rmse(yB, p_fix) else "resid"          # 사후 최선(참고)
                add("oracle", n, rep, seed, -1, p_shr if orc == "shrink" else p_fix, E_n if orc == "shrink" else E0, E3, r_hat, orc)
    meta = dict(split=sp, n_A=nA, n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), E0=E0, E_allA=E_allA, E_own=E_own,
                n_fit=n_fit, elapsed_s=round(time.time() - t0, 1), prior_tau2=prior["tau2"], prior_sigma2=prior["sigma2"], prior_regions=prior["n_regions"])
    return rows, meta, store


# ================================================================ 집계
def block_ci(stores, runs):
    """h4_common 규약: 분할 안 B 블록 재표집(방법·반복 공통 인덱스), 반복·seed 평균, 분할 분포 평균. Δ = 방법 − 물리식."""
    by_split = {st.split: st for st in stores}
    out = []
    for (m, n, tau), _ in runs[runs.method != "physics"].groupby(["method", "n", "tau"]):
        fa = lambda k, m=m, n=n, tau=tau: k[0] == m and k[1] == n and abs(k[4] - tau) < 1e-9
        fb = lambda k: k[0] == "physics"
        r = boot_delta_blocks(by_split, fa, fb, nboot=1000, seed=11, rep_fn=lambda k: k[2])
        out.append(dict(method=m, n=n, tau=tau, d_blk=r["delta"], d_blk_lo=r["ci_lo"], d_blk_hi=r["ci_hi"], p_blk=r["p_boot"], split_win=r["split_win"],
                        rep_win=r["rep_win"], d_blk_beq=r["delta_beq"], d_blk_beq_lo=r["ci_lo_beq"], d_blk_beq_hi=r["ci_hi_beq"], blk_flag=r["ci_flag"],
                        d_split=json.dumps({int(k): round(v, 3) for k, v in r["delta_split"].items()})))
    return pd.DataFrame(out)


def summarize(runs):
    phys = runs[runs.method == "physics"].set_index("split").rmse_cm
    physb = runs[runs.method == "physics"].set_index("split").rmse_beq_cm
    r = runs[runs.method != "physics"].copy()
    r["d_phys"] = r.rmse_cm - r.split.map(phys); r["d_beq"] = r.rmse_beq_cm - r.split.map(physb); r["win"] = (r.d_phys < 0).astype(float)
    r["rep_key"] = np.where(r.rep < 0, 0, r.rep)                                              # 반복 없는 방법은 분할만 짝지음
    g = r.groupby(["method", "n", "tau", "split", "rep_key"], as_index=False).agg(               # seed 평균
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"), d_beq=("d_beq", "mean"),
        win=("win", "mean"), bias_cm=("bias_cm", "mean"), shrink_frac=("branch", lambda s: float(np.mean(s == "shrink"))))
    out = []
    for (m, n, tau), sub in g.groupby(["method", "n", "tau"]):
        d, db = sub.d_phys.values, sub.d_beq.values
        lo = hi = lob = hib = np.nan
        if len(d) >= 3:
            rng = np.random.RandomState(1); ix = rng.randint(0, len(d), (1000, len(d)))
            bs, bsb = d[ix].mean(1), db[ix].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5]); lob, hib = np.percentile(bsb, [2.5, 97.5])
        out.append(dict(target=TGT, method=m, n=n, tau=tau, n_pairs=len(sub), n_splits=sub.split.nunique(),
                        rmse_mean=sub.rmse_cm.mean(), rmse_beq_mean=sub.rmse_beq_cm.mean(), rmse_phys=float(phys.mean()), rmse_phys_beq=float(physb.mean()),
                        n_eval_mean=float(runs.groupby("split").n_eval.first().mean()),
                        d_phys_mean=d.mean(), d_phys_lo_rep=lo, d_phys_hi_rep=hi, d_beq_mean=db.mean(), d_beq_lo_rep=lob, d_beq_hi_rep=hib,
                        win_rate=sub.win.mean(), bias_mean=sub.bias_cm.mean(), shrink_frac=sub.shrink_frac.mean(),
                        ci_flag="" if len(d) >= 3 else "pairs<3"))
    return pd.DataFrame(out)


def branch_table(runs):
    """대표점 3개 진단 r_hat 분포와 τ별 분기 비율, 실측 E비(전량 A·전체)와의 대조."""
    r3 = runs[(runs.method == "shrink_only") & (runs.n == BUDGETS[0])][["split", "rep", "E_hat3", "r_hat", "E0", "E_allA", "r_allA", "E_own", "r_own"]].drop_duplicates(["split", "rep"])
    rows = []
    for tau in TAUS:
        for sp, sub in r3.groupby("split"):
            rows.append(dict(target=TGT, tau=tau, split=sp, n_reps=len(sub), E0=sub.E0.iloc[0], E_allA=sub.E_allA.iloc[0], r_allA=sub.r_allA.iloc[0],
                             E_own=sub.E_own.iloc[0], r_own=sub.r_own.iloc[0], true_branch_allA="shrink" if sub.r_allA.iloc[0] >= tau else "resid",
                             r_hat_mean=sub.r_hat.mean(), r_hat_min=sub.r_hat.min(), r_hat_max=sub.r_hat.max(), E_hat3_mean=sub.E_hat3.mean(),
                             frac_shrink=float(np.mean(sub.r_hat >= tau)),
                             frac_agree_allA=float(np.mean((sub.r_hat >= tau) == (sub.r_allA.iloc[0] >= tau)))))
    return pd.DataFrame(rows)


# ================================================================ 3단계(D2): GTNP 신규 지점 대조표
def d2_table():
    inv = pd.read_csv(PROC / "gtnp_alt_inventory.csv")
    new = inv[inv.status == "new_coord"].copy()
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "region", "source_id", "cci_alt", TARGET], low_memory=False)
    v3["macro"] = macro_region(v3)
    v3["block"] = (np.floor(v3.lat / BLOCK_DEG).astype(int) * 100000 + np.floor(v3.lon / BLOCK_DEG).astype(int))
    f4 = v3[v3.source_id == "F4_direct"]
    # 현재 평가 셀 수: load_base(loc_id 병합 + CALM 좌표 2차 병합) 결과 DF 에 eval_mask 적용(v3 직접 병합은 loc_id<0 행을 놓쳐 0 이 된다).
    ev_now, ev_split, n_lab = {}, {}, {}
    for m in ("Russia_W", "Russia_E"):
        mi = np.where((DF.macro.values == m) & np.isin(DF.source_id.values, SRC_SOURCES))[0]
        n_lab[m] = int(len(mi)); ev_now[m] = int(eval_mask(DF.iloc[mi]).sum())
        parts = []
        for sp in range(3):                                                    # h25 규약 분할 0–2 의 B 채점 셀 수
            _, bi = half_split_blocks(DF, mi, sp); parts.append(int(eval_mask(DF.iloc[bi]).sum()))
        ev_split[m] = "/".join(map(str, parts))
    rows = []
    for _, r in new.iterrows():
        blk = int(np.floor(r.lat / BLOCK_DEG) * 100000 + np.floor(r.lon / BLOCK_DEG))
        d_all = haversine_km(r.lat, r.lon, v3.lat.values, v3.lon.values); j = int(np.argmin(d_all))
        fm = f4[f4.macro == r.macro]; d_m = haversine_km(r.lat, r.lon, fm.lat.values, fm.lon.values); k = int(np.argmin(d_m))
        rows.append(dict(dataset_id=int(r.dataset_id), al_name=r.al_name, macro=r.macro, lat=r.lat, lon=r.lon, block=blk,
                         block_new_vs_v3_all=bool(blk not in set(v3.block)), block_new_vs_macro_F4=bool(blk not in set(fm.block)),
                         nearest_v3_km=round(float(d_all[j]), 2), nearest_v3_loc_id=int(v3.loc_id.iloc[j]), nearest_v3_region=v3.region.iloc[j],
                         nearest_v3_source=v3.source_id.iloc[j], nearest_v3_cci_valid=bool(pd.notna(v3.cci_alt.iloc[j])),
                         nearest_macroF4_km=round(float(d_m[k]), 1), nearest_macroF4_loc_id=int(fm.loc_id.iloc[k]),
                         n_years=int(r.n_years), year_min=int(r.year_min), year_max=int(r.year_max), n_valid=int(r.n_valid),
                         value_mean_cm=float(r.value_mean), new_cell_candidate=bool(d_all[j] > 1.0),
                         quality=r.quality, labeled_cells_now_macro=n_lab[r.macro], eval_cells_now_macro=ev_now[r.macro],
                         eval_cells_B_by_split=ev_split[r.macro]))
    t = pd.DataFrame(rows)
    # 후보 셀 그룹: 같은 블록 안 0.2 km 이내 쌍(WET/DRY 소구획)은 별도 셀이지만 같은 블록. 매크로별 요약을 열로 붙인다.
    inc = t[t.new_cell_candidate].groupby("macro").agg(new_cells=("dataset_id", "size"), new_blocks=("block", "nunique")).to_dict("index")
    t["macro_new_cells_upper"] = t.macro.map(lambda m: inc.get(m, {}).get("new_cells", 0))
    t["macro_new_blocks"] = t.macro.map(lambda m: inc.get(m, {}).get("new_blocks", 0))
    t["macro_verdict"] = t.macro_new_cells_upper.map(lambda v: "편입 후보(>=3셀, 공변량 추출·CCI 유효 확인 필요)" if v >= 3 else "기록만(<3셀)")
    t["note"] = "평가 셀 편입은 CCI·토양 도일 유효 조건을 만족해야 하며 공변량 추출 전에는 상한만 알 수 있다. v3 미수정, 편입은 사용자 결정."
    return t


def main():
    t0 = time.time()
    rows, metas, stores = [], [], []

    def _done(r, m, st):
        rows.extend(r); metas.append(m); stores.append(st)
        log(f"  [split {m['split']}] 적합 {m['n_fit']} · A {m['n_A']} · 채점 {m['n_eval']}/{m['n_blocks_eval']}블록 · 원천 {m['n_src']} · E0 {m['E0']:.3f} · {m['elapsed_s']}s")
    if args.workers <= 1 or len(SPLITS) == 1:
        for sp in SPLITS:
            _done(*run_task(sp))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(SPLITS)), mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, sp): sp for sp in SPLITS}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows); runs.to_csv(OUT / f"{TAG}_runs.csv", index=False)
    save_stores(stores, OUT / f"{TAG}_blocksse.npz")
    summ = summarize(runs).merge(block_ci(stores, runs), on=["method", "n", "tau"], how="left")
    summ["d_phys_lo"], summ["d_phys_hi"] = summ.d_blk_lo, summ.d_blk_hi                     # 주 CI = 블록 부트스트랩(h4_common), _rep 는 보조
    summ.to_csv(OUT / f"{TAG}_summary.csv", index=False)
    br = branch_table(runs); br.to_csv(OUT / f"{TAG}_branch.csv", index=False)
    d2 = None
    if not args.skip_d2:
        d2 = d2_table(); d2.to_csv(OUT / ("d2_smoke_gtnp_new.csv" if args.smoke else "d2_gtnp_new.csv"), index=False)
    # F11 대조: 사전 예측 대 실측
    main_rows = summ[(summ.method == "protocol") & (np.isclose(summ.tau, args.tau_main))]
    s3_rows = summ[summ.method == "shrink_resid_S3"]
    verdict = {}
    for n in BUDGETS:
        p = main_rows[main_rows.n == n]; s = s3_rows[s3_rows.n == n]
        if len(p):
            p = p.iloc[0]; s = s.iloc[0] if len(s) else None
            verdict[f"n{n}"] = dict(delta=float(p.d_phys_mean), ci=[float(p.d_phys_lo), float(p.d_phys_hi)], win_rate=float(p.win_rate),
                                    delta_le_0=bool(p.d_phys_mean <= 0), ci_hi_lt_0=bool(p.d_phys_hi < 0) if np.isfinite(p.d_phys_hi) else None,
                                    shrink_frac=float(p.shrink_frac), s3_delta=float(s.d_phys_mean) if s is not None else None,
                                    protocol_win_rate_ge_S3=bool(p.win_rate >= s.win_rate) if s is not None else None)
    b_main = br[np.isclose(br.tau, args.tau_main)]
    actual = dict(r_allA_by_split=b_main.set_index("split").r_allA.round(4).to_dict(), r_own=float(b_main.r_own.iloc[0]) if len(b_main) else None,
                  true_branch_allA=b_main.set_index("split").true_branch_allA.to_dict(),
                  frac_shrink_by_split=b_main.set_index("split").frac_shrink.round(3).to_dict())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    meta = dict(stage="H38 D1/D2", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 D1·D2, §5 F11", tag=TAG, target=TGT,
                args=vars(args), splits=SPLITS, reps=REPS, seeds=SEEDS, budgets=BUDGETS, taus=TAUS, feats=FEATS,
                src_sources=list(SRC_SOURCES), tgt_sources=list(TGT_SOURCES),
                prediction_file=str(PRED_PATH.relative_to(ROOT)), prediction_mtime=PRED_MTIME, label_read_at=LABEL_READ_AT,
                prediction_before_labels=bool(PRED_MTIME <= LABEL_READ_AT), prediction_preexisting=PRED_PREEXISTING,
                prediction_written_at=PRED.get("written_at"), prediction_target=PRED.get("target"),
                prereg_disclosure=PREREG_DISCLOSURE if TGT == MAIN_TARGET else {"note": "본 실행 대상이 아님(스모크·점검 대상)"},
                task_meta=metas, f11_actual=actual, f11_verdict=verdict, git_commit=commit,
                n_rows=dict(runs=int(len(runs)), summary=int(len(summ)), branch=int(len(br)), d2=int(len(d2)) if d2 is not None else 0),
                elapsed_s=round(time.time() - T_ALL0, 1))
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    show = summ[summ.method.isin(["protocol", "shrink_resid_S3", "e0fix_resid", "shrink_only", "refit_allA", "resid_src_only", "oracle"])]
    print(show[["method", "n", "tau", "n_pairs", "rmse_phys", "rmse_mean", "d_phys_mean", "d_phys_lo", "d_phys_hi", "win_rate", "shrink_frac"]].round(3).to_string(index=False))
    print(br[["tau", "split", "E0", "E_allA", "r_allA", "r_hat_mean", "frac_shrink", "frac_agree_allA"]].round(3).to_string(index=False))
    if d2 is not None:
        print(d2[["dataset_id", "al_name", "macro", "block", "block_new_vs_macro_F4", "nearest_v3_km", "nearest_macroF4_km", "n_years", "value_mean_cm",
                  "labeled_cells_now_macro", "eval_cells_now_macro", "eval_cells_B_by_split", "macro_new_cells_upper", "macro_verdict"]].to_string(index=False))
    log(f"saved {TAG}_* · runs {len(runs)}행 · summary {len(summ)}행 · {time.time() - t0:.0f}s (전체 {time.time() - T_ALL0:.0f}s)")


if __name__ == "__main__":
    main()
