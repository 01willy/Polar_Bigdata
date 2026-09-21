"""L2-4 지역별 Stefan 계수·n-factor 표 + 알래스카 블록 E 분포와 물리 계수 지도 가능성 진단.

지역: 알래스카 · 주 전이 6(레나·캐나다·러시아 W/C/E·그린란드) · 심부 5(스발바르·몽골·알프스·티베트·스칸디나비아).
자료: --sources F4_direct,F4_calm_temp (심부 지역은 지온 유도 라벨 포함). E 적합 셀 = y 유효 ∩ √TDD > 0(평가 마스크 미적용).
  - E_LS = (s·y)/(s·s), E_med = median(y/s) [cm/√(°C·day)], E_soil(토양 도일 √TDD_stl1 기준), 블록 부트스트랩 95% CI(1,000회, 블록 8 미만 미산출).
  - n-factor 대리 = median(e5_tdd_soil / e5_tdd) (ERA5-Land 표층 토양 0–7 cm 도일 / 2 m 기온 도일; e5_tdd_soil ≤ 0 은 결측 처리).
  - 공변량 평균(SOC 0–5·점토·용적밀도·MAAT·TDD), ALT 평균·SD, 셀·블록 수, 알래스카 E 대비 비.
  - 알래스카 0.5° 블록별 E(셀 ≥ 3) 분포(SD·IQR)와 블록 E ~ 토양 9 + 기후 8 ridge(α=10, 표준화, √n 가중) 의 블록 leave-one-out R².
    보조: catboost_lo, 기후만/토양만, 전 지역 블록의 leave-one-region-out(탐색적).
산출 data/processed/m1/l2_regional_E.csv · l2_regional_E_blocks.csv · l2_regional_E_emap.csv · l2_regional_E_meta.json
실행 python3 scripts/2_evaluation/l2_regional_E.py [--sources F4_direct,F4_calm_temp] [--nboot 1000]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET, SOIL, CLIMATE, TRANSFER_MAIN, TRANSFER_DEEP   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--min-cells", type=int, default=3)
args = ap.parse_args()
t0 = time.time()
EMAP = list(SOIL + CLIMATE)
REGIONS = ["Alaska"] + TRANSFER_MAIN + TRANSFER_DEEP

df = L.load(sources=tuple(args.sources.split(",")))
df["ratio_nf"] = np.where((df.e5_tdd > 0) & (df.e5_tdd_soil > 0), df.e5_tdd_soil / df.e5_tdd, np.nan)
print(f"[data] {len(df):,}셀 · 출처 {args.sources} · 지역 {df.macro.nunique()}", flush=True)


def E_ls(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 3 else np.nan


def E_med(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float(np.median(y[m] / s[m])) if m.sum() >= 3 else np.nan


def boot_E(y, s, blocks, nboot, seed):
    ub = np.unique(blocks)
    if len(ub) < L.MIN_BLOCKS_CI:
        return [np.nan] * 4
    pos = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.RandomState(seed); a, b_ = np.empty(nboot), np.empty(nboot)
    for i in range(nboot):
        idx = np.concatenate([pos[b] for b in rng.choice(ub, len(ub), replace=True)])
        a[i], b_[i] = E_ls(y[idx], s[idx]), E_med(y[idx], s[idx])
    return [*np.percentile(a, [2.5, 97.5]), *np.percentile(b_, [2.5, 97.5])]


# ---------------------------------------------------------------- 지역 표
ak = df[df.macro == "Alaska"]
E_ak = E_ls(ak[TARGET].values, ak.e5_sqrt_tdd.values)
rows = []
for reg in REGIONS:
    d = df[df.macro == reg]
    if len(d) == 0:
        continue
    y, s, ss = d[TARGET].values.astype(float), d.e5_sqrt_tdd.values.astype(float), d.e5_sqrt_tdd_soil.values.astype(float)
    m = np.isfinite(y) & (s > 0)
    blocks = d.block.values
    lo, hi, mlo, mhi = boot_E(y[m], s[m], blocks[m], args.nboot, seed=abs(hash(reg)) % (2 ** 31))
    e = E_ls(y, s)
    st_ak = E_ak * s[m]; st_own = e * s[m]
    rows.append(dict(region=reg, regime=("alaska" if reg == "Alaska" else "main" if reg in TRANSFER_MAIN else "deep"),
                     n_cells=int(m.sum()), n_blocks=int(len(np.unique(blocks[m]))), sources=",".join(sorted(set(d.source_id))),
                     alt_mean_cm=float(y[m].mean()), alt_sd_cm=float(y[m].std()), E_LS=e, E_LS_lo=lo, E_LS_hi=hi, E_med=E_med(y, s),
                     E_med_lo=mlo, E_med_hi=mhi, E_soil_LS=E_ls(y, ss), E_ratio_vs_alaska=e / E_ak,
                     rmse_stefan_alaskaE=L.rmse(y[m], st_ak), bias_stefan_alaskaE=L.bias(y[m], st_ak), rmse_stefan_ownE=L.rmse(y[m], st_own),
                     nfactor_proxy_median=float(np.nanmedian(d.ratio_nf.values)), nfactor_proxy_iqr_lo=float(np.nanpercentile(d.ratio_nf.values, 25)),
                     nfactor_proxy_iqr_hi=float(np.nanpercentile(d.ratio_nf.values, 75)), n_nfactor=int(np.isfinite(d.ratio_nf.values).sum()),
                     soc_0_5_mean=float(np.nanmean(d.sg_soc_0_5)), clay_5_15_mean=float(np.nanmean(d.sg_clay_5_15)),
                     bdod_5_15_mean=float(np.nanmean(d.sg_bdod_5_15)), maat_mean_c=float(np.nanmean(d.e5_maat)), tdd_mean=float(np.nanmean(d.e5_tdd)),
                     sqrt_tdd_mean=float(np.nanmean(s[m])), cci_alt_mean=float(np.nanmean(d.cci_alt)),
                     ci_flag="ok" if len(np.unique(blocks[m])) >= L.MIN_BLOCKS_CI else f"blocks<{L.MIN_BLOCKS_CI}"))
reg_tab = pd.DataFrame(rows)
reg_tab.to_csv(L.OUT / "l2_regional_E.csv", index=False)
print(reg_tab[["region", "n_cells", "n_blocks", "alt_mean_cm", "E_LS", "E_LS_lo", "E_LS_hi", "E_med", "nfactor_proxy_median", "E_ratio_vs_alaska"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- 블록 E 표(전 지역, 셀 ≥ min_cells)
def block_table(d):
    out = []
    for b, sub in d[np.isfinite(d[TARGET].values) & (d.e5_sqrt_tdd.values > 0)].groupby("block"):
        if len(sub) < args.min_cells:
            continue
        y, s = sub[TARGET].values.astype(float), sub.e5_sqrt_tdd.values.astype(float)
        out.append(dict(block=b, region=sub.macro.iloc[0], n=len(sub), lat=float(sub.lat.mean()), lon=float(sub.lon.mean()), E=E_ls(y, s),
                        E_med=E_med(y, s), alt_mean=float(y.mean()), **{f: float(np.nanmean(sub[f].values.astype(float))) for f in EMAP}))
    return pd.DataFrame(out)


bt = block_table(df)
bt.to_csv(L.OUT / "l2_regional_E_blocks.csv", index=False)
bak = bt[bt.region == "Alaska"].reset_index(drop=True)
print(f"[blocks] 알래스카 블록 {len(bak)} (셀≥{args.min_cells}) · E SD {bak.E.std():.3f} · IQR {bak.E.quantile(.25):.3f}–{bak.E.quantile(.75):.3f} "
      f"· 범위 {bak.E.min():.3f}–{bak.E.max():.3f}", flush=True)


def prep_X(X, ref):
    med = np.nanmedian(ref, 0); med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X); R = np.where(np.isnan(ref), med, ref)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    return (X - mu) / sd, (R - mu) / sd


def fit_pred(kind, Xtr, ytr, wtr, Xte):
    if kind == "ridge":
        Xz_te, Xz_tr = prep_X(Xte, Xtr)
        return Ridge(alpha=10.0).fit(Xz_tr, ytr, sample_weight=wtr).predict(Xz_te)
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=0, verbose=0,
                           allow_writing_files=False, thread_count=L.THREADS)
    cb.fit(Xtr, ytr, sample_weight=wtr)
    return cb.predict(Xte)


def r2(y, p, w=None):
    w = np.ones_like(y) if w is None else w
    ybar = np.average(y, weights=w)
    return float(1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2))


erows = []
FEATSETS = {"soil9+climate8": EMAP, "climate8": list(CLIMATE), "soil9": list(SOIL)}
for fname, feats in FEATSETS.items():
    for kind in (["ridge", "catboost_lo"] if fname == "soil9+climate8" else ["ridge"]):
        X, yb, w = bak[feats].values.astype(float), bak.E.values, np.sqrt(bak.n.values)
        pin = fit_pred(kind, X, yb, w, X)
        pcv = np.empty(len(bak))
        for i in range(len(bak)):                                         # 블록 leave-one-out
            tr = np.arange(len(bak)) != i
            pcv[i] = fit_pred(kind, X[tr], yb[tr], w[tr], X[i:i + 1])[0]
        erows.append(dict(scope="alaska_blocks_LOBO", model=kind, features=fname, n_blocks=len(bak), E_sd=float(yb.std()),
                          E_iqr_lo=float(np.percentile(yb, 25)), E_iqr_hi=float(np.percentile(yb, 75)), r2_insample=r2(yb, pin),
                          r2_cv=r2(yb, pcv), r2_cv_weighted=r2(yb, pcv, w), rmse_E_cv=float(np.sqrt(np.mean((yb - pcv) ** 2))),
                          rmse_E_const=float(yb.std()), rmse_E_cv_weighted=float(np.sqrt(np.average((yb - pcv) ** 2, weights=w)))))
        print(f"  [emap] {kind:12s} {fname:16s} R²_in {erows[-1]['r2_insample']:.3f} · R²_LOBO {erows[-1]['r2_cv']:.3f} "
              f"(√n 가중 {erows[-1]['r2_cv_weighted']:.3f}) · {time.time()-t0:.0f}s", flush=True)
# 탐색적: 주 집합(알래스카 + 주 전이 6) 블록 leave-one-region-out. 심부 레짐(E 7–11)은 제외(포함 시 외삽 폭주로 R² −100대).
bm = bt[bt.region.isin(["Alaska"] + TRANSFER_MAIN)].reset_index(drop=True)
regs = [r for r in bm.region.unique()]
X, yb, w = bm[EMAP].values.astype(float), bm.E.values, np.sqrt(bm.n.values)
pcv = np.full(len(bm), np.nan)
for r in regs:
    te = (bm.region == r).values
    pcv[te] = fit_pred("ridge", X[~te], yb[~te], w[~te], X[te])
erows.append(dict(scope="main_regions_blocks_LORO", model="ridge", features="soil9+climate8", n_blocks=len(bm), E_sd=float(yb.std()),
                  E_iqr_lo=float(np.percentile(yb, 25)), E_iqr_hi=float(np.percentile(yb, 75)), r2_insample=np.nan, r2_cv=r2(yb, pcv),
                  r2_cv_weighted=r2(yb, pcv, w), rmse_E_cv=float(np.sqrt(np.mean((yb - pcv) ** 2))), rmse_E_const=float(yb.std()),
                  rmse_E_cv_weighted=float(np.sqrt(np.average((yb - pcv) ** 2, weights=w)))))
for r in regs:
    te = (bm.region == r).values
    if te.sum() >= 3:
        erows.append(dict(scope=f"LORO_{r}", model="ridge", features="soil9+climate8", n_blocks=int(te.sum()), E_sd=float(yb[te].std()),
                          E_iqr_lo=np.nan, E_iqr_hi=np.nan, r2_insample=np.nan, r2_cv=r2(yb[te], pcv[te]), r2_cv_weighted=r2(yb[te], pcv[te], w[te]),
                          rmse_E_cv=float(np.sqrt(np.mean((yb[te] - pcv[te]) ** 2))), rmse_E_const=float(np.sqrt(np.mean((yb[te] - yb[~te].mean()) ** 2))),
                          rmse_E_cv_weighted=np.nan))
emap = pd.DataFrame(erows)
emap.to_csv(L.OUT / "l2_regional_E_emap.csv", index=False)
print(emap[["scope", "model", "features", "n_blocks", "r2_insample", "r2_cv", "rmse_E_cv", "rmse_E_const"]].round(3).to_string(index=False))

LIT = [
    dict(source="Nelson & Outcalt (1987) Arctic Alpine Res. 19(3):279–288", quantity="edaphic factor E (Stefan 해 ALT = E·√(n·DDT))",
         value="정의 원전(지표 n-factor·토양 열물성을 한 계수로 묶음)", units="", status="정의 확인(웹 2차 요약)·수치 범위는 원문 미확인"),
    dict(source="Nelson et al. (1997) Arctic Alpine Res. 29(4):367–378, Fig. 4 (Kuparuk Flux Study plots)",
         quantity="플롯별 해빙률 β (Z = β·√DDT + α, 1995 여름)", value="β = 1.36, 1.37, 1.41, 2.11, 2.49, 3.80 (α = −9.5~+13.9 cm); 95-1 Betty Pingo(습윤 비산성 툰드라) 1.37, 95-3 Sagwon 1(습윤 산성 툰드라) 2.11",
         units="cm/√(°C·day)", status="확인(원문 PDF 텍스트 추출, geobotany.org 공개본)"),
    dict(source="Peng et al. (2018) J. Climate 31:251–266 (2차 요약)", quantity="북반구 CALM·GTN-P 사이트 E 분포",
         value="0–18, 대부분 0–12", units="cm/√(°C·day) 로 추정(원문 단위 미확인)", status="웹 요약만 확인·단위 미확인"),
    dict(source="Klene et al. (2001) Arctic Antarct. Alpine Res. 33(2):140–148", quantity="쿠파룩 식생·토양 계급별 여름 n-factor",
         value="계급별 값 원문 미확인", units="무차원", status="미확인(초록만 확인). 본 연구 대리값은 ERA5-Land 표층 토양(0–7 cm) 도일/2 m 기온 도일"),
    dict(source="Shiklomanov & Nelson (1999) Ecol. Model. 123:105–125", quantity="쿠파룩 ALT 장 해석 표현(E 공간 지도 선례)", value="", units="", status="미확인"),
]
L.write_meta(L.OUT / "l2_regional_E_meta.json", script="scripts/2_evaluation/l2_regional_E.py", sources=args.sources, n_boot=args.nboot,
             min_cells_block=args.min_cells, E_alaska_LS=E_ak, regions=REGIONS, emap_features=EMAP,
             units=dict(E="cm/√(°C·day)", nfactor_proxy="e5_tdd_soil/e5_tdd (무차원)", soc="sg_soc_0_5 SoilGrids 원 단위(파이프라인 g/kg 표기)",
                        clay="%", bdod="g/cm³ 표기(파이프라인 kg/dm³)", maat="°C", tdd="°C·day"),
             literature_notes=LIT, note="E 적합 셀 = y 유효 ∩ √TDD>0(평가 마스크 미적용). CI 블록 부트스트랩 1,000회, 블록 8 미만 NaN. "
             "rmse_stefan_alaskaE 는 알래스카 E 고정(정보 없음), rmse_stefan_ownE 는 지역 자체 E(in-sample 상한).",
             elapsed_s=round(time.time() - t0, 1))
print(f"done {time.time()-t0:.0f}s")
