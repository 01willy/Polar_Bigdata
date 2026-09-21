"""M1 보조 실험: 라벨 관측 연도(1990–2024) 대 ERA5-Land 평년(2015–2020) 시간 불일치 민감도 (리뷰 대응, 2026-09-21).

배경: 셀 라벨은 관측 연도 전체의 다년 평균이고 기후 공변량은 2015–2020 평년이다(Karjalainen 2019·Ran 2022 관행과 동일,
docs/EXPERIMENT_LOG.md 09-21). 불일치가 결과를 좌우하지 않음을 수치로 보인다. 규약은 m1_master_factorial.py를 따른다
(알래스카 6-fold 0.5° 공간블록, 계수는 fold 학습 셀에서만; 정보 없음 전이 = 알래스카 학습 → 대상 지역 전체 셀;
평가 셀 = y·CCI·토양 도일 유효).

설계
 1) 연도 열(year_min·year_max·n_years·n_obs)을 dl_dataset_cell_v2.csv·alt_calm_global_cell.csv에서 loc_id(폴백: 좌표 4자리)로
    fidelity_base_v3 셀에 조인. 부분집합: overlap(i) year_max≥2015 & year_min≤2020 / inside(ii) year_min≥2015 & year_max≤2020 (⊂ i) /
    before(iii) year_max<2015 / after(보조) year_min>2020.  (i)∪(iii)∪(after) = 전체. 관측 중앙 연도 = (year_min+year_max)/2.
 2) 알래스카 지역 내 6-fold: Stefan(λ=0)·Stefan+ridge(λ=0.75, x25)·catboost_lo 직접(x25). 학습 = fold 학습 셀 전체(train_set=all),
    같은 OOF 예측을 부분집합별로 채점. 변형: 학습 셀을 (i)로 제한(train_set=overlap, 주 채점 (i)). 통제: 학습 셀을 (iii)로 제한
    (train_set=before; 표본 수 교란 분리용).
 3) 정보 없음 전이: 알래스카 전체 셀 계수 → 주 전이 6지역(레나·캐나다·러시아 W/E/C·그린란드) 전체 셀 + 풀링(MAIN6),
    Stefan·Stefan+CCI(λ=0, 해석적) 부분집합별 RMSE·bias.
 4) 편향 추세: 잔차(관측 − Stefan 예측) ~ 관측 중앙 연도 회귀(OLS, 블록 고정효과), 기울기 cm/yr, 블록 부트스트랩 CI.
    문헌 추세 0.8 cm/yr(Streletskiy 2026)과 비교하고, 그 추세가 성립할 때 기대되는 부분집합 간 Δbias를 병기.
 5) seed 3(catboost_lo; Stefan·ridge는 결정적), 0.5° 블록 부트스트랩 400회로 부분집합 간 ΔRMSE·Δbias CI(블록 8개 미만이면 CI NaN).

부호 규약: 표의 bias = 예측 − 관측(하네스 규약, 양수=과대예측). 추세 회귀의 잔차 = 관측 − 예측.
산출 data/processed/m1/label_year_sensitivity.csv(long; table 열 = counts·indomain·indomain_foldmean·delta_subset·delta_trainset·transfer·trend),
     label_year_sensitivity_meta.json, label_year_sensitivity_cells.csv(셀별 연도·부분집합·OOF 예측)
실행(ROOT, CPU 전용): python3 scripts/3_deep_learning/m1_label_year_sensitivity.py [--smoke]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # CPU 전용. GPU 5–9는 다른 실험이 점유 중이므로 사용 금지
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")                           # 공유 서버 CPU 보호(하네스와 동일 규약)

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TRANSFER_MAIN, spatial_block_splits          # noqa: E402
from polar.preprocessing import fold_prep                                       # noqa: E402
from polar.eval_metrics import all_metrics                                      # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, fit_coefs, anchor_pred   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=400)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--threads", type=int, default=4, help="CatBoost thread_count (4 이하)")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
assert args.threads <= 4, "CatBoost thread_count는 4 이하"

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"; OUT.mkdir(exist_ok=True)
TAG = "label_year_sensitivity" + ("_smoke" if args.smoke else "")
CLIM_LO, CLIM_HI, CLIM_MID = 2015, 2020, 2017.5                # ERA5-Land 평년 기간(era5land_covariates.py)
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
NBOOT = args.nboot if not args.smoke else 20
NFOLD = 6
LAM_RIDGE = 0.75
MIN_BLOCKS_CI = 8                                              # m1_analysis.py 규약
TREND_LIT = 0.8                                                # cm/yr (Streletskiy 2026)
SUBSETS = ["all", "overlap", "inside", "before", "after"]
FEATS = INPUT_SETS["x25"]
T0 = time.time()

# ---------------------------------------------------------------- 1) 자료 + 연도 조인
df = load_base(PROC)
YC = ["year_min", "year_max", "n_years", "n_obs"]
parts = []
for f in ["dl_dataset_cell_v2.csv", "alt_calm_global_cell.csv"]:
    t = pd.read_csv(PROC / f, usecols=["loc_id", "lat", "lon"] + YC, low_memory=False)
    t["year_src"] = f
    parts.append(t)
yr = pd.concat(parts, ignore_index=True).drop_duplicates("loc_id", keep="first")
df = df.merge(yr.rename(columns={"lat": "lat_y", "lon": "lon_y"}), on="loc_id", how="left")
hit = df.year_min.notna().values
coord_mismatch = int((((df.lat - df.lat_y).abs() > 1e-4) | ((df.lon - df.lon_y).abs() > 1e-4))[hit].sum())
miss = ~hit
n_join_coord = 0
if miss.any():                                                 # loc_id 실패 셀은 좌표(4자리) 폴백
    y2 = yr.copy(); y2["klat"], y2["klon"] = y2.lat.round(4), y2.lon.round(4)
    y2 = y2.drop_duplicates(["klat", "klon"])
    key = pd.DataFrame({"klat": df.lat.round(4), "klon": df.lon.round(4)})
    fb = key.merge(y2[["klat", "klon", "year_src"] + YC], on=["klat", "klon"], how="left")
    for c in YC + ["year_src"]:
        df.loc[miss, c] = fb.loc[miss, c].values
    n_join_coord = int(df.year_min.notna().values[miss].sum())
n_join_fail = int(df.year_min.isna().sum())
df = df.drop(columns=["lat_y", "lon_y"])
df["mid_year"] = (df.year_min + df.year_max) / 2.0
df["year_gap"] = CLIM_MID - df.mid_year                       # 양수 = 평년보다 먼저 관측
ymin, ymax = df.year_min.values.astype(float), df.year_max.values.astype(float)
SUB = {"all": np.ones(len(df), bool),
       "overlap": (ymax >= CLIM_LO) & (ymin <= CLIM_HI),
       "inside": (ymin >= CLIM_LO) & (ymax <= CLIM_HI),
       "before": ymax < CLIM_LO,
       "after": ymin > CLIM_HI}
df["subset"] = np.select([SUB["inside"], SUB["overlap"], SUB["before"], SUB["after"]],
                         ["inside", "overlap_only", "before", "after"], "unknown")
EV = eval_mask(df)
BLK = df.block.values
Y = df[TARGET].values.astype(float)
X = df[FEATS].values.astype(np.float32)
print(f"[data] {len(df):,}셀 · 연도 조인 loc_id {int(hit.sum()):,} + 좌표 {n_join_coord} · 실패 {n_join_fail} · 좌표 불일치 {coord_mismatch}"
      f" · 부분집합 {df.subset.value_counts().to_dict()}", flush=True)

rows = []


def add(table, **kw):
    rows.append(dict(table=table, **kw))


def region_idx(name):
    if name == "MAIN6":
        return np.where(df.macro.isin(TRANSFER_MAIN).values)[0]
    return np.where(df.macro.values == name)[0]


def score(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return dict(rmse=np.nan, bias=np.nan, mae=np.nan, r2=np.nan, n=int(m.sum()))
    mm = all_metrics(y[m], p[m])
    return dict(rmse=mm["rmse_cm"], bias=mm["bias_cm"], mae=mm["mae_cm"], r2=mm["r2"], n=int(m.sum()))


def rmse_(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan


def block_boot(blocks, fn, nboot, seed):
    """0.5° 블록 재표집. fn(idx) → 스칼라 또는 튜플."""
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


# ---------------------------------------------------------------- 2) 부분집합 셀 수 표
for reg in ["Alaska"] + TRANSFER_MAIN + ["MAIN6"]:
    idx = region_idx(reg)
    for s in SUBSETS:
        m = idx[SUB[s][idx]]
        d = df.iloc[m]
        add("counts", region=reg, subset=s, n=int(len(m)), n_eval=int(EV[m].sum()), n_blocks=int(d.block.nunique()),
            frac=float(len(m) / max(len(idx), 1)), alt_mean=float(d[TARGET].mean()) if len(m) else np.nan,
            alt_sd=float(d[TARGET].std()) if len(m) > 1 else np.nan,
            sqrt_tdd_mean=float(d.e5_sqrt_tdd.mean()) if len(m) else np.nan,
            mid_year_mean=float(d.mid_year.mean()) if len(m) else np.nan,
            year_gap_mean=float(d.year_gap.mean()) if len(m) else np.nan,
            n_years_median=float(d.n_years.median()) if len(m) else np.nan,
            frac_multi_year=float((d.n_years > 1).mean()) if len(m) else np.nan)

# ---------------------------------------------------------------- 3) 알래스카 지역 내 6-fold
ak = np.where(df.macro.values == "Alaska")[0]
folds = spatial_block_splits(df, n_splits=NFOLD, sub_idx=ak)
if args.smoke:
    folds = folds[:2]
TRAIN_SETS = ["all", "overlap", "before"]
METHODS = ["stefan", "stefan_ridge", "catboost_lo"]
P = {}                                                          # (train_set, method, seed) → 전체 길이 배열(OOF)
coef_log = []


def arr():
    return np.full(len(df), np.nan)


def fit_catboost(Xtr, ytr, Xte, seed):
    """m1_core.fit_model('catboost_lo')와 동일 하이퍼파라미터, thread_count만 4 이하."""
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


for ts in TRAIN_SETS:
    for fi, (tr, te) in enumerate(folds):
        if ts != "all":
            tr = tr[SUB[ts][tr]]
        te = te[EV[te]]
        trd, ted = df.iloc[tr], df.iloc[te]
        k = fit_coefs(trd)
        coef_log.append(dict(train_set=ts, fold=fi, E=k["E"], n_train=int(len(tr)), n_test=int(len(te)),
                             n_blocks_train=int(trd.block.nunique())))
        st_tr, st_te = anchor_pred("stefan", trd, k), anchor_pred("stefan", ted, k)
        P.setdefault((ts, "stefan", 0), arr())[te] = st_te
        Xtr, Xte, ytr = X[tr], X[te], Y[tr]
        Xtr2, Xte2 = fold_prep(Xtr, Xte, nan_native=False)      # ridge: 학습 중앙값 대체 + z-score(하네스 규약)
        g = Ridge(alpha=10.0).fit(Xtr2, ytr - st_tr).predict(Xte2)
        P.setdefault((ts, "stefan_ridge", 0), arr())[te] = st_te + LAM_RIDGE * g
        for seed in SEEDS:
            P.setdefault((ts, "catboost_lo", seed), arr())[te] = fit_catboost(Xtr, ytr, Xte, seed)
        print(f"  [AK {ts} fold{fi}] 학습 {len(tr):,} · 평가 {len(te):,} · E={k['E']:.4f} · {time.time()-T0:.0f}s", flush=True)

ak_ev = ak[EV[ak]]
ENS = {}                                                        # (train_set, method) → seed 평균 예측
for ts in TRAIN_SETS:
    for me in METHODS:
        ps = [P[kk] for kk in P if kk[0] == ts and kk[1] == me]
        pm = np.mean(np.stack(ps), 0)
        ENS[(ts, me)] = pm
        for s in SUBSETS:
            idx = ak_ev[SUB[s][ak_ev] & np.isfinite(pm[ak_ev])]
            per = [score(Y[idx], p[idx]) for p in ps]
            sc = score(Y[idx], pm[idx])
            primary = int(ts == "all" or s == ts or (ts == "overlap" and s == "inside"))
            add("indomain", region="Alaska", train_set=ts, method=me, subset=s, n=sc["n"],
                n_blocks=int(len(np.unique(BLK[idx]))), n_seed=len(ps), primary=primary,
                rmse=float(np.mean([q["rmse"] for q in per])), rmse_seed_sd=float(np.std([q["rmse"] for q in per])),
                rmse_ens=sc["rmse"], bias=float(np.mean([q["bias"] for q in per])), mae=sc["mae"], r2=sc["r2"],
                alt_mean=float(Y[idx].mean()) if len(idx) else np.nan,
                year_gap_mean=float(df.year_gap.values[idx].mean()) if len(idx) else np.nan)

# fold 평균 집계(게이트 요약 m1_gate_shard0.csv 규약: fold별 RMSE의 평균; 위 indomain 표는 셀 가중 풀링 OOF)
for ts in TRAIN_SETS:
    for me in METHODS:
        pm = ENS[(ts, me)]
        for s in SUBSETS:
            fr = []
            for fi, (_, te) in enumerate(folds):
                ii = te[EV[te] & SUB[s][te] & np.isfinite(pm[te])]
                if len(ii) >= 5:
                    fr.append((rmse_(Y[ii], pm[ii]), float((pm[ii] - Y[ii]).mean()), len(ii)))
            if not fr:
                continue
            add("indomain_foldmean", region="Alaska", train_set=ts, method=me, subset=s, n=int(sum(f[2] for f in fr)),
                n_folds=len(fr), rmse=float(np.mean([f[0] for f in fr])), rmse_fold_sd=float(np.std([f[0] for f in fr])),
                bias=float(np.mean([f[1] for f in fr])), primary=int(ts == "all" or s == ts or (ts == "overlap" and s == "inside")))

# 부분집합 간 Δ(블록 부트스트랩, 같은 재표집에서 두 부분집합 RMSE·bias를 계산)
PAIRS = [("overlap", "before"), ("inside", "before"), ("after", "before"), ("inside", "overlap")]


def delta_subset_rows(label_region, train_set, method, idx, p, bseed):
    y, pp, b = Y[idx], p[idx], BLK[idx]
    masks = {s: SUB[s][idx] for s in SUBSETS}
    gap = df.year_gap.values[idx]
    nb = len(np.unique(b))
    for sa, sb in PAIRS:
        ma, mb = masks[sa], masks[sb]
        if ma.sum() < 5 or mb.sum() < 5:
            continue

        def fn(ii, ma=ma, mb=mb):
            a, c = ma[ii], mb[ii]
            if a.sum() < 5 or c.sum() < 5:
                return (np.nan, np.nan)
            return (rmse_(y[ii][a], pp[ii][a]) - rmse_(y[ii][c], pp[ii][c]),
                    float((pp[ii][a] - y[ii][a]).mean() - (pp[ii][c] - y[ii][c]).mean()))
        d0 = fn(np.arange(len(idx)))
        if nb >= MIN_BLOCKS_CI:
            bo = block_boot(b, fn, NBOOT, seed=bseed)
            (r_lo, r_hi), (b_lo, b_hi) = ci(bo[:, 0]), ci(bo[:, 1])
        else:
            r_lo = r_hi = b_lo = b_hi = np.nan
        add("delta_subset", region=label_region, train_set=train_set, method=method, subset=f"{sa}-{sb}",
            n=int(ma.sum() + mb.sum()), n_a=int(ma.sum()), n_b=int(mb.sum()), n_blocks=nb, nboot=NBOOT,
            d_rmse=d0[0], d_rmse_lo=r_lo, d_rmse_hi=r_hi, d_bias=d0[1], d_bias_lo=b_lo, d_bias_hi=b_hi,
            rmse_a=rmse_(y[ma], pp[ma]), rmse_b=rmse_(y[mb], pp[mb]),
            # 문헌 추세(0.8 cm/yr)가 성립할 때 기대되는 Δbias(예측−관측): 먼저 관측된 집합일수록 관측이 작아 bias 커짐
            d_bias_expected_lit=float(TREND_LIT * (gap[ma].mean() - gap[mb].mean())),
            year_gap_a=float(gap[ma].mean()), year_gap_b=float(gap[mb].mean()))


bs = 100
for ts in TRAIN_SETS:
    for me in METHODS:
        pm = ENS[(ts, me)]
        idx = ak_ev[np.isfinite(pm[ak_ev])]
        delta_subset_rows("Alaska", ts, me, idx, pm, bs); bs += 1

# 학습집합 제한 효과: (i) 셀에서 짝지은 Δ = RMSE(train=overlap 또는 before) − RMSE(train=all)
for me in METHODS:
    for ts in ["overlap", "before"]:
        pa, pb = ENS[(ts, me)], ENS[("all", me)]
        for s in ["overlap", "inside", "before"]:
            idx = ak_ev[SUB[s][ak_ev] & np.isfinite(pa[ak_ev]) & np.isfinite(pb[ak_ev])]
            y, b = Y[idx], BLK[idx]
            qa, qb = pa[idx], pb[idx]

            def fn(ii, qa=qa, qb=qb, y=y):
                return rmse_(y[ii], qa[ii]) - rmse_(y[ii], qb[ii])
            d0 = fn(np.arange(len(idx)))
            nb = len(np.unique(b))
            lo, hi = ci(block_boot(b, fn, NBOOT, seed=bs)) if nb >= MIN_BLOCKS_CI else (np.nan, np.nan); bs += 1
            n_tr = int(np.mean([c["n_train"] for c in coef_log if c["train_set"] == ts]))
            n_tr_all = int(np.mean([c["n_train"] for c in coef_log if c["train_set"] == "all"]))
            add("delta_trainset", region="Alaska", train_set=f"{ts}-all", method=me, subset=s, n=int(len(idx)), n_blocks=nb,
                nboot=NBOOT, d_rmse=d0, d_rmse_lo=lo, d_rmse_hi=hi, rmse_a=rmse_(y, qa), rmse_b=rmse_(y, qb),
                n_train_a=n_tr, n_train_b=n_tr_all, primary=int(s == ts or (ts == "overlap" and s == "inside")))

# ---------------------------------------------------------------- 4) 정보 없음 전이 (알래스카 전체 계수 → 주 전이 6지역)
k_all = fit_coefs(df.iloc[ak])
tg_all = region_idx("MAIN6"); tg_ev = tg_all[EV[tg_all]]
PT = {"stefan": arr(), "stefan_cci": arr()}
for me in PT:
    PT[me][tg_ev] = anchor_pred(me, df.iloc[tg_ev], k_all)
for reg in TRANSFER_MAIN + ["MAIN6"]:
    idx = region_idx(reg); idx = idx[EV[idx]]
    for me in PT:
        for s in SUBSETS:
            ii = idx[SUB[s][idx]]
            sc = score(Y[ii], PT[me][ii])
            add("transfer", region=reg, train_set="alaska_all", method=me, subset=s, n=sc["n"],
                n_blocks=int(len(np.unique(BLK[ii]))), n_seed=1, primary=1, rmse=sc["rmse"], rmse_ens=sc["rmse"],
                bias=sc["bias"], mae=sc["mae"], r2=sc["r2"], alt_mean=float(Y[ii].mean()) if len(ii) else np.nan,
                year_gap_mean=float(df.year_gap.values[ii].mean()) if len(ii) else np.nan)
        if len(idx) >= 10:
            delta_subset_rows(reg, "alaska_all", me, idx, PT[me], bs); bs += 1

# ---------------------------------------------------------------- 5) 편향 추세: 잔차(관측 − Stefan) ~ 관측 중앙 연도
RES = arr()
RES[ak_ev] = Y[ak_ev] - ENS[("all", "stefan")][ak_ev]         # 알래스카: OOF Stefan(train_set=all)
RES[tg_ev] = Y[tg_ev] - PT["stefan"][tg_ev]                   # 전이: 알래스카 전체 계수 Stefan
single = (df.n_years.values == 1)
DOMAINS = {"Alaska": ak_ev, "Alaska_single_year": ak_ev[single[ak_ev]], "MAIN6": tg_ev,
           "Alaska+MAIN6": np.concatenate([ak_ev, tg_ev])}
for lab, idx in DOMAINS.items():
    idx = idx[np.isfinite(RES[idx]) & np.isfinite(df.mid_year.values[idx])]
    t, r, b = df.mid_year.values[idx].astype(float), RES[idx], BLK[idx]
    if len(idx) < 10 or t.std() == 0:
        continue
    slope, icpt = np.polyfit(t, r, 1)
    resid = r - (icpt + slope * t); tc = t - t.mean()
    se = float(np.sqrt(resid @ resid / (len(t) - 2) / (tc @ tc)))

    def fn_ols(ii, t=t, r=r):
        return np.polyfit(t[ii], r[ii], 1)[0] if t[ii].std() > 0 else np.nan
    lo, hi = ci(block_boot(b, fn_ols, NBOOT, seed=bs)); bs += 1
    add("trend", region=lab, method="ols", n=int(len(idx)), n_blocks=int(len(np.unique(b))), nboot=NBOOT,
        slope_cm_yr=float(slope), slope_se_ols=se, slope_lo=lo, slope_hi=hi, lit_trend_cm_yr=TREND_LIT,
        mid_year_mean=float(t.mean()), mid_year_sd=float(t.std()), year_min=float(t.min()), year_max=float(t.max()),
        resid_mean=float(r.mean()))
    # 블록 고정효과(블록 내 편차만 사용: 공간 교란 통제)
    tb = pd.Series(t).groupby(b).transform("mean").values; rb = pd.Series(r).groupby(b).transform("mean").values
    tw, rw = t - tb, r - rb
    n_blk_var = int(pd.Series(t).groupby(b).nunique().gt(1).sum())
    if tw @ tw > 0:
        slope_fe = float((tw @ rw) / (tw @ tw))
        dof = len(t) - len(np.unique(b)) - 1
        se_fe = float(np.sqrt(((rw - slope_fe * tw) ** 2).sum() / max(dof, 1) / (tw @ tw)))

        def fn_fe(ii, tw=tw, rw=rw):
            return (tw[ii] @ rw[ii]) / (tw[ii] @ tw[ii]) if tw[ii] @ tw[ii] > 0 else np.nan
        lo, hi = ci(block_boot(b, fn_fe, NBOOT, seed=bs)); bs += 1
        add("trend", region=lab, method="block_fe", n=int(len(idx)), n_blocks=int(len(np.unique(b))), nboot=NBOOT,
            n_blocks_with_year_var=n_blk_var, slope_cm_yr=slope_fe, slope_se_ols=se_fe, slope_lo=lo, slope_hi=hi,
            lit_trend_cm_yr=TREND_LIT, mid_year_mean=float(t.mean()), mid_year_sd=float(np.sqrt(tw @ tw / len(tw))),
            year_min=float(t.min()), year_max=float(t.max()), resid_mean=float(r.mean()))

# ---------------------------------------------------------------- 6) 저장
res = pd.DataFrame(rows)
res.to_csv(OUT / f"{TAG}.csv", index=False)
cells = df[["loc_id", "macro", "region", "block", "lat", "lon", TARGET, "e5_sqrt_tdd", "cci_alt", "year_min", "year_max",
            "n_years", "n_obs", "mid_year", "year_gap", "subset", "year_src"]].copy()
cells["eval"] = EV.astype(int)
for (ts, me), p in ENS.items():
    cells[f"oof_{ts}_{me}"] = p
for me, p in PT.items():
    cells[f"transfer_{me}"] = p
cells["resid_stefan"] = RES
cells.to_csv(OUT / f"{TAG}_cells.csv", index=False)
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                               # noqa: BLE001
    commit = "NA"
# 문헌 추세(0.8 cm/yr)가 성립할 때 불일치의 기여 상한: 평균 연도차는 E 보정이 흡수(수준 오프셋), 연도차 SD만 잔여 잡음(직교 가정)
trend_bound = {}
for lab, idx, col in [("Alaska", ak_ev, ENS[("all", "stefan")]), ("MAIN6", tg_ev, PT["stefan"])]:
    gap = df.year_gap.values[idx]; r0 = rmse_(Y[idx], col[idx]); noise = TREND_LIT * gap.std()
    trend_bound[lab] = dict(year_gap_mean=float(gap.mean()), year_gap_sd=float(gap.std()),
                            level_offset_absorbed_cm=float(TREND_LIT * gap.mean()), residual_noise_cm=float(noise),
                            rmse_stefan=r0, rmse_without_mismatch=float(np.sqrt(max(r0 ** 2 - noise ** 2, 0.0))),
                            rmse_inflation_cm=float(r0 - np.sqrt(max(r0 ** 2 - noise ** 2, 0.0))))
meta = dict(stage="M1-aux label_year_sensitivity", created="2026-09-21", base="fidelity_base_v3.csv", soil="e5_soil_tdd_v3.csv",
            sources=["F4_direct"], year_sources=["dl_dataset_cell_v2.csv", "alt_calm_global_cell.csv"],
            join=dict(by_loc_id=int(hit.sum()), by_coord_fallback=n_join_coord, failed=n_join_fail, coord_mismatch_gt_1e4=coord_mismatch),
            clim_period=[CLIM_LO, CLIM_HI], clim_mid=CLIM_MID,
            subset_def=dict(overlap="year_max>=2015 & year_min<=2020", inside="year_min>=2015 & year_max<=2020 (subset of overlap)",
                            before="year_max<2015", after="year_min>2020 (supplementary; overlap+before+after=all)"),
            subset_counts_all=df.subset.value_counts().to_dict(),
            n_cells=int(len(df)), n_eval=int(EV.sum()), regions={r: int(n) for r, n in df.macro.value_counts().items()},
            nfold=len(folds), seeds=SEEDS, nboot=NBOOT, min_blocks_ci=MIN_BLOCKS_CI, lam_ridge=LAM_RIDGE, feats="x25",
            catboost=dict(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, thread_count=args.threads),
            E_alaska_all=k_all["E"], coefs_fold=coef_log, trend_lit_cm_yr=TREND_LIT, trend_bound=trend_bound,
            aggregation="indomain = cell-weighted pooled OOF over 6 folds; indomain_foldmean = mean of per-fold metrics (gate convention)",
            bias_sign="bias = pred - obs (positive = overprediction); trend residual = obs - pred",
            git_commit=commit, elapsed_s=round(time.time() - T0, 1), smoke=bool(args.smoke),
            plan="docs/EXPERIMENT_LOG.md 2026-09-21 (연도별 대 다년 평균 ALT 라벨)")
(OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))

# ---------------------------------------------------------------- 7) 콘솔 요약
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\n=== 부분집합 셀 수 ===")
print(res[res.table == "counts"].pivot_table(index="region", columns="subset", values="n").reindex(["Alaska"] + TRANSFER_MAIN + ["MAIN6"]).reindex(columns=SUBSETS).to_string())
print("\n=== 알래스카 지역 내 RMSE(seed 평균) / bias ===")
ind = res[res.table == "indomain"]
print(ind.pivot_table(index=["train_set", "method"], columns="subset", values="rmse").reindex(columns=SUBSETS).round(2).to_string())
print(ind.pivot_table(index=["train_set", "method"], columns="subset", values="bias").reindex(columns=SUBSETS).round(2).to_string())
print("\n=== Δ 부분집합(알래스카·전이) ===")
print(res[res.table == "delta_subset"][["region", "train_set", "method", "subset", "n_a", "n_b", "n_blocks", "d_rmse", "d_rmse_lo", "d_rmse_hi",
                                        "d_bias", "d_bias_lo", "d_bias_hi", "d_bias_expected_lit"]].round(2).to_string())
print("\n=== Δ 학습집합(알래스카) ===")
print(res[res.table == "delta_trainset"][["train_set", "method", "subset", "n", "d_rmse", "d_rmse_lo", "d_rmse_hi", "rmse_a", "rmse_b", "n_train_a", "n_train_b"]].round(2).to_string())
print("\n=== 전이 RMSE / bias ===")
tr = res[res.table == "transfer"]
print(tr.pivot_table(index=["region", "method"], columns="subset", values="rmse").reindex(columns=SUBSETS).round(2).to_string())
print(tr.pivot_table(index=["region", "method"], columns="subset", values="bias").reindex(columns=SUBSETS).round(2).to_string())
print(tr.pivot_table(index=["region", "method"], columns="subset", values="n").reindex(columns=SUBSETS).to_string())
print("\n=== 잔차-연도 기울기(cm/yr) ===")
print(res[res.table == "trend"][["region", "method", "n", "n_blocks", "slope_cm_yr", "slope_se_ols", "slope_lo", "slope_hi", "mid_year_sd", "year_min", "year_max"]].round(3).to_string())
print(f"\nsaved: {OUT / (TAG + '.csv')} ({len(res)}행) · {TAG}_meta.json · {TAG}_cells.csv · {time.time()-T0:.0f}s")
