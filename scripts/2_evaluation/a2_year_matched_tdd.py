"""A2 보조 실험: 셀별 관측 연도에 정합한 ERA5-Land TDD로 Stefan 계수·앵커·유사라벨을 재산출하고 전이 결과가 바뀌는지 본다
(감사 대안 설명 C2-E, 2026-09-21).

배경: 라벨·공변량·CCI의 관측 기간이 지역마다 다르다(알래스카 라벨 대부분 2013–2018 단일 연도, ERA5-Land 공변량은 2015–2020 평년,
러시아 CALM은 1992–2024 다년 평균). 지역 "편향"의 일부가 이 기간 불일치일 수 있다. 라벨 쪽 부분집합 민감도
(scripts/3_deep_learning/m1_label_year_sensitivity.py, 결론: 불일치가 결과를 좌우하지 않음)를 보완해, 여기서는 공변량 쪽
(Stefan 강제력 TDD)을 셀별 관측 연도에 맞춘다. 규약은 m1_master_factorial.py·m1_analysis.py를 따른다.

설계
 1) 연별 TDD: nh_monthly_2010-2024.nc의 월평균 2 m 기온에서 era5land_covariates.py와 같은 정의(월평균 0 °C 초과 달의 (T−0)×일수 합)로
    2010–2024 각 연도 TDD를 셀별 최근접 육지 격자(enrich_new_regions.py 규칙: ±2셀 → ±5셀 유클리드 최근접 폴백)에서 산출.
 2) 공변량 판(version):
      clim     = 현행 열(2015–2020 월기후값을 만든 뒤 클립; fidelity_base_v3)
      clim_re  = 같은 정의·같은 격자 규칙을 2010–2024 파일에서 재산출(파일·격자 재현 점검)
      clim_yr  = 2015–2020 연별 TDD 평균(클립 후 평균; matched·full과 정의가 같은 기준선)
      matched  = [year_min, year_max] ∩ [2010, 2024] 연별 TDD 평균. 범위가 2010 이전이면 2010–2014 평균으로 대체하고 플래그(pre2010),
                 일부만 2010 이전이면 partial 플래그
      full     = 2010–2024 연별 TDD 평균
    e5_tdd·e5_sqrt_tdd 두 열만 치환한다(다른 기후 공변량 6종은 현행 평년 유지).
 3) 판별 반복: E(알래스카 최소제곱, fit_coefs) → 정보 없음 전이(알래스카 학습 → 대상 전체 셀, 평가 셀 = y·CCI·토양 도일 유효)
    Stefan·Stefan+CCI RMSE·bias(주 6지역 + 풀링) + H1 ΔRMSE(Stefan+CCI − Stefan, 0.5° 블록 부트스트랩 400, 블록 8 미만 CI NaN)
    → 공변량만(레나·캐나다·러시아 W·E; half_split_blocks split 0·1·2, A블록 유사라벨 r=10, catboost_lo 직접, B 채점, seed 3)
    H3 Δ(Stefan 유사라벨 − const·tddlin·none). 알래스카 6-fold OOF Stefan은 판별 강제력 적합도 점검용.
 4) 판 간 짝지은 Δ(matched − clim 등, 같은 셀·같은 블록 재표집)로 연도 정합이 전이 오차·편향을 바꾸는지 검정.

부호 규약: bias = 예측 − 관측(양수 = 과대예측). Δ = 첫째 − 둘째(음수 = 첫째가 낫다).
산출 data/processed/m1/a2_year_matched_tdd.csv(long; table 열 = tdd_diff·coefs·indomain·noinfo·h1·covonly_seed·covonly·h3·delta_version),
     a2_year_matched_tdd_cells.csv(셀별 연별·판별 TDD·플래그), a2_year_matched_tdd_meta.json
실행(ROOT, CPU 전용): python3 scripts/2_evaluation/a2_year_matched_tdd.py [--smoke]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # CPU 전용. GPU 5–9는 다른 실험이 점유 중이므로 사용 금지
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")                           # 공유 서버 CPU 보호(하네스와 동일 규약)

import argparse
import calendar
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TRANSFER_MAIN, spatial_block_splits                          # noqa: E402
from polar.eval_metrics import all_metrics                                                      # noqa: E402
from polar.m1_core import (INPUT_SETS, load_base, eval_mask, half_split_blocks, fit_coefs,     # noqa: E402
                           anchor_pred, pseudo_label)

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=400)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--r", type=float, default=10.0, help="유사라벨 비율(알래스카 실측 수 대비)")
ap.add_argument("--threads", type=int, default=4, help="CatBoost thread_count (4 이하)")
ap.add_argument("--versions", default="clim,clim_re,clim_yr,matched,full")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
assert args.threads <= 4, "CatBoost thread_count는 4 이하"

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"; OUT.mkdir(exist_ok=True)
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2010-2024.nc"
TAG = "a2_year_matched_tdd" + ("_smoke" if args.smoke else "")
CLIM_LO, CLIM_HI = 2015, 2020                                  # 현행 평년(era5land_covariates.py)
NC_LO, NC_HI = 2010, 2024                                      # 월별 파일 범위
PRE_LO, PRE_HI = 2010, 2014                                    # 2010 이전 관측 셀의 대체 기간
COV_REGIONS = ["Lena", "Canada", "Russia_W", "Russia_E"]      # 공변량만(H3) 대상
CTRLS = ["const", "tddlin", "none"]
PSEUDOS = ["stefan"] + CTRLS
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
NBOOT = args.nboot if not args.smoke else 20
VERSIONS = args.versions.split(",") if not args.smoke else ["clim", "matched"]
assert VERSIONS[0] == "clim", "첫 판은 현행(clim)이어야 한다(짝지은 Δ 기준)"
MIN_BLOCKS_CI = 8                                              # m1_analysis.py 규약
FEATS = INPUT_SETS["x25"]
T0 = time.time()

# ---------------------------------------------------------------- 1) 자료 + 연도 조인
df = load_base(PROC)
yr = pd.read_csv(PROC / "m1" / "label_year_sensitivity_cells.csv",
                 usecols=["loc_id", "year_min", "year_max", "n_years", "mid_year", "subset"])
df = df.merge(yr, on="loc_id", how="left")
assert df.year_min.notna().all(), "연도 조인 실패 셀 존재"
EV = eval_mask(df)
BLK = df.block.values
Y = df[TARGET].values.astype(float)
ak = np.where(df.macro.values == "Alaska")[0]
print(f"[data] {len(df):,}셀 · 평가 {int(EV.sum()):,} · 알래스카 {len(ak):,} · {time.time()-T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 2) 연별 TDD (nc → 셀)
ds = xr.open_dataset(NC)
tn = "valid_time" if "valid_time" in ds.coords else "time"
glat, glon = ds.latitude.values, ds.longitude.values
years, months = ds[tn].dt.year.values, ds[tn].dt.month.values
YEARS = np.arange(NC_LO, NC_HI + 1)
assert set(YEARS) == set(np.unique(years)) and len(years) == 12 * len(YEARS)
t2m = ds["t2m"].values.astype(np.float32) - 273.15           # (180, lat, lon) °C
land = np.isfinite(t2m).all(axis=0)
print(f"[nc] t2m {t2m.shape} · 육지 격자 {land.mean():.1%} · {time.time()-T0:.0f}s", flush=True)


def nearest(arr, v):
    return int(np.abs(arr - v).argmin())


def fallback(iy, ix, la, lo, nb):
    """±nb셀 내 유클리드(위경도) 최근접 육지 격자(enrich_new_regions.py·expand_calm_regions.py 규칙). 없으면 None."""
    best = None
    for dy in range(-nb, nb + 1):
        for dx in range(-nb, nb + 1):
            y2, x2 = iy + dy, (ix + dx) % len(glon)
            if 0 <= y2 < len(glat) and land[y2, x2]:
                d2 = (glat[y2] - la) ** 2 + (glon[x2] - lo) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, y2, x2)
    return best


IY, IX = np.empty(len(df), int), np.empty(len(df), int)
FB = np.zeros(len(df))
n_fb = {"direct": 0, "0.2": 0, "0.5": 0, "fail": 0}
for i, (la, lo) in enumerate(zip(df.lat.values, df.lon.values)):
    iy, ix = nearest(glat, la), nearest(glon, lo)
    if land[iy, ix]:
        n_fb["direct"] += 1
    else:
        b = fallback(iy, ix, la, lo, 2)
        if b is not None:
            FB[i] = 0.2; n_fb["0.2"] += 1
        else:
            b = fallback(iy, ix, la, lo, 5)
            if b is not None:
                FB[i] = 0.5; n_fb["0.5"] += 1
            else:
                FB[i] = np.nan; n_fb["fail"] += 1
        if b is not None:
            iy, ix = b[1], b[2]
    IY[i], IX[i] = iy, ix
TM = t2m[:, IY, IX]                                           # (180, n) °C
del t2m
days_y = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(years, months)])   # 해당 연도 실제 일수
TDD_Y = np.stack([(np.clip(TM[years == y], 0, None) * days_y[years == y][:, None]).sum(0) for y in YEARS])   # (15, n)
days19 = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])                     # 현행 평년 정의의 일수
selc = (years >= CLIM_LO) & (years <= CLIM_HI)
clim12 = np.stack([TM[selc & (months == m)].mean(0) for m in range(1, 13)])
tdd_clim_re = (np.clip(clim12, 0, None) * days19[:, None]).sum(0)          # 평균 후 클립(현행 정의 재현)
tdd_clim_yr = TDD_Y[(YEARS >= CLIM_LO) & (YEARS <= CLIM_HI)].mean(0)      # 클립 후 평균
tdd_full = TDD_Y.mean(0)
ymin, ymax = df.year_min.values.astype(int), df.year_max.values.astype(int)
pre = ymax < NC_LO
partial = (ymin < NC_LO) & ~pre
lo_y = np.where(pre, PRE_LO, np.clip(ymin, NC_LO, NC_HI))
hi_y = np.where(pre, PRE_HI, np.clip(ymax, NC_LO, NC_HI))
MY = (YEARS[:, None] >= lo_y[None, :]) & (YEARS[:, None] <= hi_y[None, :])
n_match = MY.sum(0)
assert (n_match >= 1).all()
tdd_matched = (np.where(MY, TDD_Y, 0.0)).sum(0) / n_match
match_flag = np.where(pre, "pre2010", np.where(partial, "partial", "full"))
d_re = tdd_clim_re - df.e5_tdd.values
repro = dict(median_abs=float(np.nanmedian(np.abs(d_re))), p95_abs=float(np.nanpercentile(np.abs(d_re), 95)),
             max_abs=float(np.nanmax(np.abs(d_re))), n_gt20=int((np.abs(d_re) > 20).sum()),
             n_gt20_by_region={r: int(v) for r, v in pd.Series(np.abs(d_re) > 20).groupby(df.macro.values).sum().items()})
print(f"[tdd] 폴백 {n_fb} · 평년 재현 |차| 중앙값 {repro['median_abs']:.3f} · p95 {repro['p95_abs']:.1f} · >20 °C·day {repro['n_gt20']}셀 "
      f"{repro['n_gt20_by_region']} · pre2010 {int(pre.sum())} · partial {int(partial.sum())} · {time.time()-T0:.0f}s", flush=True)

VER = {"clim": df.e5_tdd.values.astype(float), "clim_re": tdd_clim_re, "clim_yr": tdd_clim_yr,
       "matched": tdd_matched, "full": tdd_full}
VER_DESC = {"clim": "현행 열(2015–2020 월기후값 후 클립, fidelity_base_v3)",
            "clim_re": "현행 정의를 2010–2024 파일·유클리드 폴백 격자에서 재산출",
            "clim_yr": "2015–2020 연별 TDD 평균(클립 후 평균)",
            "matched": "[year_min, year_max] ∩ [2010, 2024] 연별 TDD 평균(2010 이전 관측은 2010–2014 평균)",
            "full": "2010–2024 연별 TDD 평균"}


def make_df(v):
    if v == "clim":
        return df
    d = df.copy()
    d["e5_tdd"] = VER[v]
    d["e5_sqrt_tdd"] = np.sqrt(np.clip(VER[v], 0, None))
    return d


rows = []


def add(table, **kw):
    rows.append(dict(table=table, **kw))


def region_idx(name):
    if name == "MAIN6":
        return np.where(df.macro.isin(TRANSFER_MAIN).values)[0]
    if name == "ALL":
        return np.arange(len(df))
    return np.where(df.macro.values == name)[0]


def rmse_(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan


def bias_(y, p):
    return float(np.mean(p - y)) if len(y) else np.nan


def score(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return dict(rmse=np.nan, bias=np.nan, mae=np.nan, r2=np.nan, n=int(m.sum()))
    mm = all_metrics(y[m], p[m])
    return dict(rmse=mm["rmse_cm"], bias=mm["bias_cm"], mae=mm["mae_cm"], r2=mm["r2"], n=int(m.sum()))


def block_boot(blocks, fn, nboot, seed):
    """0.5° 블록 재표집. fn(idx) → 튜플. 같은 재표집을 두 예측에 적용(짝지은 Δ)."""
    ub = np.unique(blocks)
    pos = [np.where(blocks == b)[0] for b in ub]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(nboot):
        pick = rng.randint(0, len(ub), len(ub))
        out.append(fn(np.concatenate([pos[j] for j in pick])))
    return np.asarray(out, float)


def ci(a):
    a = np.asarray(a, float); ok = np.isfinite(a)
    if len(a) == 0 or ok.sum() < 0.8 * len(a):
        return np.nan, np.nan
    return float(np.percentile(a[ok], 2.5)), float(np.percentile(a[ok], 97.5))


def paired_delta(y, pA, pB, blocks, seed):
    """ΔRMSE·Δbias(A − B) 점추정 + 블록 부트스트랩 CI. 블록 < MIN_BLOCKS_CI 이면 CI NaN."""
    m = np.isfinite(y) & np.isfinite(pA) & np.isfinite(pB)
    y, pA, pB, blocks = y[m], pA[m], pB[m], blocks[m]
    nb = len(np.unique(blocks))
    d0 = (rmse_(y, pA) - rmse_(y, pB), bias_(y, pA) - bias_(y, pB))
    if nb < MIN_BLOCKS_CI or len(y) < 5:
        return dict(n=int(len(y)), n_blocks=nb, d_rmse=d0[0], d_rmse_lo=np.nan, d_rmse_hi=np.nan, d_bias=d0[1],
                    d_bias_lo=np.nan, d_bias_hi=np.nan, ci_flag=f"blocks<{MIN_BLOCKS_CI}", rmse_a=rmse_(y, pA), rmse_b=rmse_(y, pB))

    def fn(ii):
        return (rmse_(y[ii], pA[ii]) - rmse_(y[ii], pB[ii]), bias_(y[ii], pA[ii]) - bias_(y[ii], pB[ii]))
    bo = block_boot(blocks, fn, NBOOT, seed)
    (rl, rh), (bl, bh) = ci(bo[:, 0]), ci(bo[:, 1])
    return dict(n=int(len(y)), n_blocks=nb, d_rmse=d0[0], d_rmse_lo=rl, d_rmse_hi=rh, d_bias=d0[1], d_bias_lo=bl, d_bias_hi=bh,
                ci_flag="ok", rmse_a=rmse_(y, pA), rmse_b=rmse_(y, pB))


def sig(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


# ---------------------------------------------------------------- 3) TDD 차이 분포 표
REG_ROWS = ["Alaska"] + TRANSFER_MAIN + ["MAIN6", "ALL"]
PAIRS = [(v, "clim") for v in VER if v != "clim"] + [("matched", "clim_yr"), ("full", "clim_yr")]
for reg in REG_ROWS:
    idx = region_idx(reg)
    for v, ref in PAIRS:
        a, b = VER[v][idx], VER[ref][idx]
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        d = a[ok] - b[ok]; rel = 100.0 * d / b[ok]
        sa, sb = np.sqrt(np.clip(a[ok], 0, None)), np.sqrt(b[ok])
        add("tdd_diff", region=reg, version=v, ref=ref, n=int(ok.sum()), n_blocks=int(len(np.unique(BLK[idx]))),
            d_median=float(np.median(d)), d_q25=float(np.percentile(d, 25)), d_q75=float(np.percentile(d, 75)),
            d_mean=float(d.mean()), d_sd=float(d.std()),
            rel_median=float(np.median(rel)), rel_q25=float(np.percentile(rel, 25)), rel_q75=float(np.percentile(rel, 75)),
            corr_tdd=float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and b[ok].std() > 0 else np.nan,
            sqrt_ratio_mean=float(np.mean(sa / sb)),
            tdd_ref_mean=float(b[ok].mean()), tdd_ref_sd=float(b[ok].std()),
            interannual_sd_mean=float(np.nanmean(TDD_Y[:, idx].std(0))),
            frac_pre2010=float(pre[idx].mean()), frac_partial=float(partial[idx].mean()),
            n_years_matched_median=float(np.median(n_match[idx])),
            year_min_median=float(np.median(ymin[idx])), year_max_median=float(np.median(ymax[idx])),
            single_year_frac=float((df.n_years.values[idx] == 1).mean()))

# ---------------------------------------------------------------- 4) 판별 계수 + 알래스카 지역 내 Stefan(적합도 점검)
K, DV = {}, {}
ak_folds = spatial_block_splits(df, n_splits=6, sub_idx=ak)
ak_ev = ak[EV[ak]]
OOF = {}
for v in VERSIONS:
    dv = make_df(v); DV[v] = dv
    k = fit_coefs(dv.iloc[ak]); K[v] = k
    p_in = anchor_pred("stefan", dv.iloc[ak_ev], k)
    sc_in = score(Y[ak_ev], p_in)
    oof = np.full(len(df), np.nan)
    for tr, te in ak_folds:
        kk = fit_coefs(dv.iloc[tr]); te = te[EV[te]]
        oof[te] = anchor_pred("stefan", dv.iloc[te], kk)
    OOF[v] = oof
    sc_oof = score(Y[ak_ev], oof[ak_ev])
    add("coefs", version=v, E=k["E"], E_med=k["E_med"], st_a=k["st_a"], st_E=k["st_E"], tdd_a=k["tdd_a"], tdd_b=k["tdd_b"],
        cci_a=k["cci_a"], cci_b=k["cci_b"], ymean=k["ymean"], n_train_real=k["n_train_real"],
        sqrt_tdd_mean_alaska=float(np.nanmean(dv.e5_sqrt_tdd.values[ak])))
    add("indomain", version=v, region="Alaska", method="stefan_insample", **sc_in, n_blocks=int(len(np.unique(BLK[ak_ev]))))
    add("indomain", version=v, region="Alaska", method="stefan_oof6", **sc_oof, n_blocks=int(len(np.unique(BLK[ak_ev]))))
    print(f"  [coef {v:8s}] E={k['E']:.4f} · 알래스카 Stefan in-sample RMSE {sc_in['rmse']:.2f} · OOF6 {sc_oof['rmse']:.2f} (R² {sc_oof['r2']:.3f})", flush=True)

# ---------------------------------------------------------------- 5) 정보 없음 전이 + H1
PT = {}
bs = 1000
for v in VERSIONS:
    dv, k = DV[v], K[v]
    tg_all = region_idx("MAIN6"); tg_ev = tg_all[EV[tg_all]]
    for me in ["stefan", "stefan_cci"]:
        p = np.full(len(df), np.nan); p[tg_ev] = anchor_pred(me, dv.iloc[tg_ev], k); PT[(v, me)] = p
    h1_d = []
    for reg in TRANSFER_MAIN + ["MAIN6"]:
        idx = region_idx(reg); idx = idx[EV[idx]]
        for me in ["stefan", "stefan_cci"]:
            sc = score(Y[idx], PT[(v, me)][idx])
            add("noinfo", version=v, region=reg, method=me, **sc, n_blocks=int(len(np.unique(BLK[idx]))),
                alt_mean=float(Y[idx].mean()) if len(idx) else np.nan,
                sqrt_tdd_mean=float(np.nanmean(dv.e5_sqrt_tdd.values[idx])) if len(idx) else np.nan)
        pd_ = paired_delta(Y[idx], PT[(v, "stefan_cci")][idx], PT[(v, "stefan")][idx], BLK[idx], bs); bs += 1
        add("h1", version=v, region=reg, label="noinfo: Stefan+CCI − Stefan (λ=0)", **pd_, sig=int(sig(pd_["d_rmse_lo"], pd_["d_rmse_hi"])))
        if reg != "MAIN6":
            h1_d.append(pd_["d_rmse"])
    add("h1", version=v, region="REGION_MEAN_MAIN6", label="noinfo: Stefan+CCI − Stefan (λ=0)", n=len(h1_d),
        n_blocks=int(np.sum(np.array(h1_d) < 0)), d_rmse=float(np.mean(h1_d)), d_rmse_lo=np.nan, d_rmse_hi=np.nan,
        d_bias=np.nan, d_bias_lo=np.nan, d_bias_hi=np.nan, ci_flag="region_mean(n_blocks=neg_count)", rmse_a=np.nan, rmse_b=np.nan, sig=0)
print(f"[noinfo] 완료 · {time.time()-T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 6) 공변량만 + H3
SPL = {}
for reg in COV_REGIONS:
    t_idx = np.where(df.macro.values == reg)[0]
    for sp in SPLITS:
        A, B = half_split_blocks(df, t_idx, sp)
        SPL[(reg, sp)] = (A, B, B[EV[B]])


def fit_cb(Xtr, ytr, seed):
    """m1_core.fit_model('catboost_lo')와 동일 하이퍼파라미터, thread_count만 4 이하."""
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr)
    return m


PC = {}                                                        # (version, region, split, pseudo, seed) → evB 예측
n_fit = 0
for v in VERSIONS:
    dv, k = DV[v], K[v]
    Xreal = dv.iloc[ak][FEATS].values.astype(np.float32); yreal = Y[ak]
    n_src = len(ak); n_ps = int(args.r * n_src)
    none_models = {seed: fit_cb(Xreal, yreal, seed) for seed in SEEDS}          # 유사라벨 없음: 분할 무관, seed별 1회
    n_fit += len(SEEDS)
    for reg in COV_REGIONS:
        for sp in SPLITS:
            A, B, evB = SPL[(reg, sp)]
            Xte = dv.iloc[evB][FEATS].values.astype(np.float32); yte = Y[evB]
            P = dv.iloc[A]
            pool_mean = float(np.nanmean(k["E"] * P.e5_sqrt_tdd.values))
            for seed in SEEDS:
                PC[(v, reg, sp, "none", seed)] = np.asarray(none_models[seed].predict(Xte), float)
                for pk in ["stefan", "const", "tddlin"]:
                    rng = np.random.RandomState(seed)                            # 하네스 규약: (구성, seed)별 새 rng → 같은 seed면 같은 A 표본
                    sel = rng.choice(len(P), n_ps, replace=n_ps > len(P))
                    ps = P.iloc[sel]
                    yps = pseudo_label(pk, ps, k, rng, pool_stefan_mean=pool_mean)
                    Xtr = np.vstack([Xreal, ps[FEATS].values.astype(np.float32)])
                    ytr = np.concatenate([yreal, yps])
                    ok = np.isfinite(ytr)
                    PC[(v, reg, sp, pk, seed)] = np.asarray(fit_cb(Xtr[ok], ytr[ok], seed).predict(Xte), float); n_fit += 1
                for pk in PSEUDOS:
                    sc = score(yte, PC[(v, reg, sp, pk, seed)])
                    add("covonly_seed", version=v, region=reg, split=sp, pseudo=pk, seed=seed, **sc,
                        n_blocks=int(len(np.unique(BLK[evB]))), n_train=int(len(yreal) + (0 if pk == "none" else n_ps)))
            print(f"  [covonly {v:8s} {reg:9s} split{sp}] 평가 {len(evB):,} · 누적 적합 {n_fit} · {time.time()-T0:.0f}s", flush=True)

# 집계(하네스 summary 규약: seed 평균 RMSE → 분할 평균; 분할 간 SD 병기)
res_seed = pd.DataFrame([r for r in rows if r["table"] == "covonly_seed"])
per = res_seed.groupby(["version", "region", "split", "pseudo"], as_index=False).agg(rmse=("rmse", "mean"), bias=("bias", "mean"), n=("n", "first"), n_blocks=("n_blocks", "first"))
for (v, reg, pk), g in per.groupby(["version", "region", "pseudo"]):
    add("covonly", version=v, region=reg, pseudo=pk, rmse=float(g.rmse.mean()), rmse_split_sd=float(g.rmse.std()) if len(g) > 1 else np.nan,
        bias=float(g.bias.mean()), n=int(g.n.sum()), n_blocks=int(g.n_blocks.max()), n_split=int(len(g)), n_seed=len(SEEDS))


def pooled(v, reg, pk):
    """분할 3회의 B 평가 집합을 이어붙인 (y, block, seed 평균 예측)."""
    yy, bb, pp = [], [], []
    for sp in SPLITS:
        _, _, evB = SPL[(reg, sp)]
        yy.append(Y[evB]); bb.append(BLK[evB])
        pp.append(np.mean(np.stack([PC[(v, reg, sp, pk, s)] for s in SEEDS]), 0))
    return np.concatenate(yy), np.concatenate(bb), np.concatenate(pp), pp


for v in VERSIONS:
    for ctrl in CTRLS:
        d_reg = []
        for reg in COV_REGIONS:
            y, b, pa, pa_s = pooled(v, reg, "stefan")
            _, _, pb, pb_s = pooled(v, reg, ctrl)
            per_split = [rmse_(Y[SPL[(reg, sp)][2]], a) - rmse_(Y[SPL[(reg, sp)][2]], c) for sp, a, c in zip(SPLITS, pa_s, pb_s)]
            pd_ = paired_delta(y, pa, pb, b, bs); bs += 1
            add("h3", version=v, region=reg, ctrl=ctrl, label=f"covonly: Stefan 유사라벨 − {ctrl} (직접 catboost_lo, r={args.r:g})", **pd_,
                n_splits=len(SPLITS), d_split_mean=float(np.mean(per_split)), d_split_sd=float(np.std(per_split)) if len(per_split) > 1 else np.nan,
                sig=int(sig(pd_["d_rmse_lo"], pd_["d_rmse_hi"])))
            d_reg.append(pd_["d_rmse"])
        add("h3", version=v, region="REGION_MEAN_COV4", ctrl=ctrl, label=f"covonly: Stefan 유사라벨 − {ctrl} (직접 catboost_lo, r={args.r:g})",
            n=len(d_reg), n_blocks=int(np.sum(np.array(d_reg) < 0)), d_rmse=float(np.mean(d_reg)), d_rmse_lo=np.nan, d_rmse_hi=np.nan,
            d_bias=np.nan, d_bias_lo=np.nan, d_bias_hi=np.nan, ci_flag="region_mean(n_blocks=neg_count)", rmse_a=np.nan, rmse_b=np.nan,
            n_splits=len(SPLITS), d_split_mean=np.nan, d_split_sd=np.nan, sig=0)
print(f"[covonly] 완료 · 적합 {n_fit} · {time.time()-T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 7) 판 간 짝지은 Δ (version − clim, 같은 셀·같은 재표집)
for v in VERSIONS[1:]:
    # 알래스카 지역 내 OOF Stefan
    pd_ = paired_delta(Y[ak_ev], OOF[v][ak_ev], OOF["clim"][ak_ev], BLK[ak_ev], bs); bs += 1
    add("delta_version", version=v, ref="clim", cond="indomain", region="Alaska", method="stefan_oof6", **pd_, sig=int(sig(pd_["d_rmse_lo"], pd_["d_rmse_hi"])))
    # 정보 없음 전이
    for reg in TRANSFER_MAIN + ["MAIN6"]:
        idx = region_idx(reg); idx = idx[EV[idx]]
        for me in ["stefan", "stefan_cci"]:
            pd_ = paired_delta(Y[idx], PT[(v, me)][idx], PT[("clim", me)][idx], BLK[idx], bs); bs += 1
            add("delta_version", version=v, ref="clim", cond="noinfo", region=reg, method=me, **pd_, sig=int(sig(pd_["d_rmse_lo"], pd_["d_rmse_hi"])))
    # 공변량만(분할 풀링, seed 평균)
    for reg in COV_REGIONS:
        for pk in PSEUDOS:
            y, b, pa, _ = pooled(v, reg, pk)
            _, _, pb, _ = pooled("clim", reg, pk)
            pd_ = paired_delta(y, pa, pb, b, bs); bs += 1
            add("delta_version", version=v, ref="clim", cond="covonly", region=reg, method=pk, **pd_, sig=int(sig(pd_["d_rmse_lo"], pd_["d_rmse_hi"])))

# ---------------------------------------------------------------- 8) 판정: 연도 정합이 방향·유의성을 바꾸는가
res = pd.DataFrame(rows)


def conclusions(v):
    """(검정, 지역) → (Δ 부호, 유의) 사전 + 정보 없음 Stefan bias 부호."""
    out = {}
    reg_ok = ~res.region.fillna("").str.startswith("REGION")
    for _, r in res[(res.table == "h1") & (res.version == v) & reg_ok].iterrows():
        if np.isfinite(r.d_rmse):
            out[("H1", r.region)] = (int(np.sign(r.d_rmse)), int(r.sig))
    for _, r in res[(res.table == "h3") & (res.version == v) & reg_ok].iterrows():
        if np.isfinite(r.d_rmse):
            out[("H3-" + r.ctrl, r.region)] = (int(np.sign(r.d_rmse)), int(r.sig))
    for _, r in res[(res.table == "noinfo") & (res.version == v) & (res.method == "stefan")].iterrows():
        if np.isfinite(r.bias):
            out[("bias_stefan", r.region)] = (int(np.sign(r.bias)), int(abs(r.bias) > 5))    # 유의 대신 |bias|>5 cm 기준
    return out


base_c = conclusions("clim")
verdict = {}
for v in VERSIONS[1:]:
    c = conclusions(v)
    flips = {"sign": [], "sig": []}
    for key, (s0, g0) in base_c.items():
        s1, g1 = c.get(key, (None, None))
        if s1 is None:
            continue
        if s1 != s0 and (s0 != 0):
            flips["sign"].append(f"{key[0]}@{key[1]}")
        if g1 != g0:
            flips["sig"].append(f"{key[0]}@{key[1]}")
    dv_ = res[(res.table == "delta_version") & (res.version == v)]
    verdict[v] = dict(n_conclusions=len(base_c), sign_flips=flips["sign"], sig_flips=flips["sig"],
                      n_sig_delta_version=int(dv_.sig.sum()), n_delta_version=int(len(dv_)),
                      max_abs_d_rmse_noinfo_stefan=float(dv_[(dv_.cond == "noinfo") & (dv_.method == "stefan") & (dv_.region != "MAIN6")].d_rmse.abs().max()),
                      max_abs_d_bias_noinfo_stefan=float(dv_[(dv_.cond == "noinfo") & (dv_.method == "stefan") & (dv_.region != "MAIN6")].d_bias.abs().max()),
                      main6_d_rmse_stefan=float(dv_[(dv_.cond == "noinfo") & (dv_.method == "stefan") & (dv_.region == "MAIN6")].d_rmse.iloc[0]),
                      main6_d_bias_stefan=float(dv_[(dv_.cond == "noinfo") & (dv_.method == "stefan") & (dv_.region == "MAIN6")].d_bias.iloc[0]))

# ---------------------------------------------------------------- 9) 저장
res.to_csv(OUT / f"{TAG}.csv", index=False)
cells = df[["loc_id", "macro", "region", "block", "lat", "lon", TARGET, "cci_alt", "year_min", "year_max", "n_years", "mid_year", "subset"]].copy()
cells["eval"] = EV.astype(int)
cells["grid_lat"], cells["grid_lon"], cells["fallback_deg"] = glat[IY], glon[IX], FB
cells["match_flag"], cells["n_years_matched"], cells["match_lo"], cells["match_hi"] = match_flag, n_match, lo_y, hi_y
for v in VER:
    cells[f"tdd_{v}"] = VER[v]
for j, y in enumerate(YEARS):
    cells[f"tdd_{y}"] = TDD_Y[j]
for v in VERSIONS:
    cells[f"noinfo_stefan_{v}"] = PT[(v, "stefan")]
    cells[f"oof_stefan_{v}"] = OOF[v]
cells.to_csv(OUT / f"{TAG}_cells.csv", index=False)
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                               # noqa: BLE001
    commit = "NA"
meta = dict(stage="A2-aux year_matched_tdd (audit C2-E)", created="2026-09-21", base="fidelity_base_v3.csv", soil="e5_soil_tdd_v3.csv",
            sources=["F4_direct"], nc=str(NC.relative_to(ROOT)), nc_years=[NC_LO, NC_HI], clim_period=[CLIM_LO, CLIM_HI],
            pre2010_substitute=[PRE_LO, PRE_HI], year_source="data/processed/m1/label_year_sensitivity_cells.csv",
            tdd_definition="sum over months with monthly-mean t2m > 0 °C of (T − 0) × days in that month/year; clim uses 2019 day counts after averaging months",
            grid_rule="nearest 0.1° grid; if ocean (NaN), Euclidean-nearest land grid within ±2 then ±5 cells (enrich_new_regions.py rule)",
            fallback_counts=n_fb, clim_reproduction=repro, versions={v: VER_DESC[v] for v in VERSIONS}, replaced_columns=["e5_tdd", "e5_sqrt_tdd"],
            n_cells=int(len(df)), n_eval=int(EV.sum()), n_pre2010=int(pre.sum()), n_partial=int(partial.sum()),
            regions={r: int(n) for r, n in df.macro.value_counts().items()},
            cov_regions=COV_REGIONS, pseudos=PSEUDOS, ctrls=CTRLS, r=args.r, seeds=SEEDS, splits=SPLITS, nboot=NBOOT, min_blocks_ci=MIN_BLOCKS_CI,
            catboost=dict(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, thread_count=args.threads), feats="x25",
            n_fit=n_fit, E={v: K[v]["E"] for v in VERSIONS}, verdict=verdict,
            bias_sign="bias = pred - obs (positive = overprediction); delta = first - second (negative = first better)",
            git_commit=commit, elapsed_s=round(time.time() - T0, 1), smoke=bool(args.smoke),
            related="scripts/3_deep_learning/m1_label_year_sensitivity.py (label-side subset sensitivity)")
(OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))

# ---------------------------------------------------------------- 10) 콘솔 요약
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\n=== TDD 차이 분포 (version − ref, °C·day) ===")
td = res[res.table == "tdd_diff"]
print(td[["region", "version", "ref", "n", "d_median", "d_q25", "d_q75", "rel_median", "corr_tdd", "sqrt_ratio_mean", "interannual_sd_mean",
          "frac_pre2010", "frac_partial", "year_min_median", "year_max_median"]].round(3).to_string(index=False))
print("\n=== 판별 E · 알래스카 Stefan ===")
print(res[res.table == "coefs"][["version", "E", "st_a", "st_E", "tdd_a", "tdd_b", "sqrt_tdd_mean_alaska"]].round(4).to_string(index=False))
print(res[res.table == "indomain"].pivot_table(index="version", columns="method", values=["rmse", "r2"]).round(3).to_string())
print("\n=== 정보 없음 전이 RMSE / bias (지역 × 판) ===")
ni = res[res.table == "noinfo"]
print(ni.pivot_table(index=["region", "method"], columns="version", values="rmse").reindex(columns=VERSIONS).round(2).to_string())
print(ni.pivot_table(index=["region", "method"], columns="version", values="bias").reindex(columns=VERSIONS).round(2).to_string())
print("\n=== H1 ΔRMSE (Stefan+CCI − Stefan) ===")
print(res[res.table == "h1"][["version", "region", "n", "n_blocks", "d_rmse", "d_rmse_lo", "d_rmse_hi", "d_bias", "sig"]].round(2).to_string(index=False))
print("\n=== 공변량만 RMSE (지역 × 유사라벨 × 판) ===")
print(res[res.table == "covonly"].pivot_table(index=["region", "pseudo"], columns="version", values="rmse").reindex(columns=VERSIONS).round(2).to_string())
print("\n=== H3 ΔRMSE (Stefan 유사라벨 − 대조) ===")
print(res[res.table == "h3"][["version", "region", "ctrl", "n", "n_blocks", "d_rmse", "d_rmse_lo", "d_rmse_hi", "d_split_sd", "sig"]].round(2).to_string(index=False))
print("\n=== 판 간 짝지은 Δ (version − clim) ===")
print(res[res.table == "delta_version"][["version", "cond", "region", "method", "n", "n_blocks", "d_rmse", "d_rmse_lo", "d_rmse_hi", "d_bias", "d_bias_lo", "d_bias_hi", "sig"]].round(2).to_string(index=False))
print("\n=== 판정 ===")
for v, w in verdict.items():
    print(f"  {v:8s}: 결론 {w['n_conclusions']}건 중 부호 뒤집힘 {len(w['sign_flips'])} {w['sign_flips']} · 유의성 변화 {len(w['sig_flips'])} {w['sig_flips']}"
          f" · 판 간 Δ 유의 {w['n_sig_delta_version']}/{w['n_delta_version']} · MAIN6 Stefan ΔRMSE {w['main6_d_rmse_stefan']:+.2f} Δbias {w['main6_d_bias_stefan']:+.2f}")
print(f"\nsaved: {OUT / (TAG + '.csv')} ({len(res)}행) · {TAG}_cells.csv · {TAG}_meta.json · 적합 {n_fit} · {time.time()-T0:.0f}s")
