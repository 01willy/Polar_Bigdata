"""L2-5 표본 대표성 잡음 하한 표.

자료: data/processed/dl_dataset_cell_v2.csv(셀 alt_sd·n_obs·n_years), alt_calm_global_cell.csv(CALM 셀), CALM 원자료
      data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab(사이트-연도 ALD).
지역별 (a) 셀 내 SD의 RMS(n_obs ≥ 2; ABoVE 250 m 셀 안 점 관측 SD), (b) 연도 간 SD의 RMS(CALM 사이트-연도, n_years ≥ 5),
(c) 다년 평균 라벨의 표준오차 SD/√n_years 의 RMS 를 방법별 RMSE(알래스카 14.46 Stefan·13.33 Stefan+ridge, 전이 정보 없음 Stefan·Stefan+CCI)와
나란히 표기. 설명 가능한 분산 상한 = 1 − 잡음²/Var(y) (Var(y) = 평가 셀 관측 ALT 분산). 달성 R² = 1 − RMSE²/Var(y).
산출 data/processed/m1/l2_noise_floor.csv · l2_noise_floor_calm_sites.csv · l2_noise_floor_meta.json
실행 python3 scripts/2_evaluation/l2_noise_floor.py
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET, TRANSFER_MAIN                         # noqa: E402
from polar.m1_core import eval_mask                                      # noqa: E402

t0 = time.time()
CANON = {"Alaska": dict(rmse_stefan=14.46, rmse_best=13.33, best_label="Stefan+ridge λ=0.75 (지역 내 6-fold)")}
RAW = L.ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"


def rms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


# ---------------------------------------------------------------- (a) 셀 내 SD (v2 셀 표)
v2 = pd.read_csv(L.PROC / "dl_dataset_cell_v2.csv", low_memory=False)
V2MAP = {"ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska", "ABoVE_CA": "Canada", "Canada": "Canada", "Lena_RU": "Lena"}
v2["macro"] = v2.region.map(V2MAP)
within = {}
for reg, src in [("Alaska", "ABoVE_AK"), ("Canada", "ABoVE_CA")]:
    d = v2[(v2.region == src) & (v2.n_obs >= 2)]
    d1 = d[d.n_years <= 1]
    within[reg] = dict(within_sd_rms=rms(d.alt_sd), n_within=int(len(d)), within_sd_rms_1yr=rms(d1.alt_sd), n_within_1yr=int(len(d1)),
                       within_sd_median=float(d.alt_sd.median()), n_obs_median=float(d.n_obs.median()), within_source=f"{src} 셀(n_obs≥2)")
# 레나: 연도 간 SD(alt_sd = 연간 최대값의 연도 간 SD, n_years ≥ 2)
lena = v2[(v2.region == "Lena_RU") & (v2.n_years >= 2)]
lena5 = v2[(v2.region == "Lena_RU") & (v2.n_years >= 5)]

# ---------------------------------------------------------------- (b)(c) CALM 사이트-연도
raw = pd.read_csv(RAW, sep="\t", skiprows=290)
raw["ald"] = pd.to_numeric(raw["ALD [cm]"], errors="coerce")
n_nonnum = int(raw["ALD [cm]"].notna().sum() - raw.ald.notna().sum())
raw["year"] = pd.to_numeric(raw["Date/Time"].astype(str).str[:4], errors="coerce")
raw = raw[raw.ald.notna() & raw.year.notna()]


def calm_macro(country, lat, lon):
    c = str(country)
    if "Alaska" in c:
        return "Alaska"
    if c == "Canada":
        return "Canada"
    if "Greenland" in c:
        return "Greenland"
    if "Svalbard" in c:
        return "Scandinavia" if lat < 74 else "Svalbard"
    if c in ("Switzerland", "Italy"):
        return "Alps"
    if c in ("Mongolia", "Kazakstan"):
        return "Mongolia_CAsia"
    if c == "China":
        return "Tibet"
    if c == "Russia":
        return "Russia_W" if lon < 90 else ("Russia_C" if lon < 140 else "Russia_E")
    return c


raw["macro"] = [calm_macro(c, la, lo) for c, la, lo in zip(raw.Country, raw.Latitude, raw.Longitude)]
sites = (raw.groupby(["Event", "macro"], as_index=False)
            .agg(lat=("Latitude", "first"), lon=("Longitude", "first"), n_years=("year", "nunique"), year_min=("year", "min"),
                 year_max=("year", "max"), alt_mean=("ald", "mean"), alt_sd=("ald", "std"), alt_min=("ald", "min"), alt_max=("ald", "max")))
def detrended_sd(g):
    """사이트 내 연도 선형 추세 제거 후 잔차 SD(연 ≥ 3). 장기 추세를 잡음에서 분리."""
    if g.year.nunique() < 3:
        return np.nan
    b, a = np.polyfit(g.year.values.astype(float), g.ald.values.astype(float), 1)
    return float(np.std(g.ald.values - (a + b * g.year.values), ddof=2))


sites = sites.merge(raw.groupby("Event").apply(detrended_sd).rename("alt_sd_detr").reset_index(), on="Event", how="left")
sites["se_multiyear"] = sites.alt_sd / np.sqrt(sites.n_years)
sites["cv"] = sites.alt_sd / sites.alt_mean
sites.to_csv(L.OUT / "l2_noise_floor_calm_sites.csv", index=False)
s5 = sites[sites.n_years >= 5]
inter = (s5.groupby("macro").agg(interannual_sd_rms=("alt_sd", rms), interannual_sd_median=("alt_sd", "median"),
                                 interannual_sd_detr_rms=("alt_sd_detr", rms), se_multiyear_rms=("se_multiyear", rms),
                                 n_sites_5yr=("Event", "size"), n_years_median=("n_years", "median"), alt_mean_calm=("alt_mean", "mean"),
                                 cv_median=("cv", "median")))
print(f"[calm] 사이트-연도 {len(raw):,} · 비수치 ALD {n_nonnum} · 사이트 {len(sites)} · n_years≥5 {len(s5)} · 기간 {int(raw.year.min())}–{int(raw.year.max())}", flush=True)
print(inter.round(2).to_string())

# ---------------------------------------------------------------- 방법별 RMSE·Var(y) (평가 셀)
df = L.load()
regs = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
no, _ = L.run_noinfo(df, regs, ["stefan", "stefan_cci"], verbose=False)
ak = df.iloc[L.region_idx(df, "Alaska")]
var_y = {"Alaska": float(ak[TARGET].var())}
rm = {"Alaska": CANON["Alaska"]}
for r in regs:
    d = no[no.macro == r]
    if len(d) < 2:
        continue
    var_y[r] = float(d[TARGET].var())
    rm[r] = dict(rmse_stefan=L.rmse(d[TARGET], d.pred_stefan), rmse_best=L.rmse(d[TARGET], d.pred_stefan_cci), best_label="Stefan+CCI (정보 없음)")

rows = []
for reg in ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland", "Svalbard", "Scandinavia", "Mongolia_CAsia", "Alps", "Tibet"]:
    r = dict(region=reg, regime=("alaska" if reg == "Alaska" else "main" if reg in TRANSFER_MAIN else "deep"))
    r.update(within.get(reg, dict(within_sd_rms=np.nan, n_within=0, within_sd_rms_1yr=np.nan, n_within_1yr=0, within_sd_median=np.nan,
                                  n_obs_median=np.nan, within_source="")))
    if reg in inter.index:
        r.update(inter.loc[reg].to_dict()); r["interannual_source"] = "CALM 사이트-연도(PANGAEA 972777)"
    elif reg == "Lena":
        r.update(dict(interannual_sd_rms=rms(lena.alt_sd), interannual_sd_median=float(lena.alt_sd.median()), interannual_sd_detr_rms=np.nan,
                      se_multiyear_rms=rms(lena.alt_sd / np.sqrt(lena.n_years)),
                      n_sites_5yr=int(len(lena5)), n_years_median=float(lena.n_years.median()), alt_mean_calm=np.nan, cv_median=np.nan,
                      interannual_source=f"Lena_RU 셀 n_years≥2 (n={len(lena)}; n_years≥5 는 {len(lena5)})"))
    else:
        r.update(dict(interannual_sd_rms=np.nan, interannual_sd_median=np.nan, interannual_sd_detr_rms=np.nan, se_multiyear_rms=np.nan, n_sites_5yr=0, n_years_median=np.nan,
                      alt_mean_calm=np.nan, cv_median=np.nan, interannual_source=""))
    vy = var_y.get(reg, np.nan); r["var_y_eval_cm2"] = vy; r["sd_y_eval_cm"] = np.sqrt(vy) if np.isfinite(vy) else np.nan
    r["n_eval"] = int(len(ak)) if reg == "Alaska" else int((no.macro == reg).sum())
    r.update(rm.get(reg, dict(rmse_stefan=np.nan, rmse_best=np.nan, best_label="")))
    for key, col in [("a_within", "within_sd_rms"), ("b_interannual", "interannual_sd_rms"), ("b_interannual_detr", "interannual_sd_detr_rms"),
                     ("c_se_multiyear", "se_multiyear_rms")]:
        nz = r[col]
        r[f"bound_r2_{key}"] = 1 - nz ** 2 / vy if np.isfinite(nz) and np.isfinite(vy) else np.nan
        r[f"noise_share_of_mse_{key}"] = nz ** 2 / r["rmse_stefan"] ** 2 if np.isfinite(nz) and np.isfinite(r["rmse_stefan"]) else np.nan
    r["r2_achieved_stefan"] = 1 - r["rmse_stefan"] ** 2 / vy if np.isfinite(vy) and np.isfinite(r["rmse_stefan"]) else np.nan
    r["r2_achieved_best"] = 1 - r["rmse_best"] ** 2 / vy if np.isfinite(vy) and np.isfinite(r["rmse_best"]) else np.nan
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(L.OUT / "l2_noise_floor.csv", index=False)
cols = ["region", "n_eval", "sd_y_eval_cm", "within_sd_rms", "n_within", "interannual_sd_rms", "interannual_sd_detr_rms", "n_sites_5yr", "se_multiyear_rms",
        "rmse_stefan", "rmse_best", "bound_r2_a_within", "bound_r2_b_interannual", "bound_r2_b_interannual_detr", "r2_achieved_best"]
print(tab[cols].round(3).to_string(index=False))
L.write_meta(L.OUT / "l2_noise_floor_meta.json", script="scripts/2_evaluation/l2_noise_floor.py", calm_raw=str(RAW), calm_siteyears=int(len(raw)),
             calm_nonnumeric_ald=n_nonnum, calm_years=[int(raw.year.min()), int(raw.year.max())], canonical_alaska=CANON["Alaska"],
             note="(a) ABoVE 셀 내 점 관측 SD(n_obs≥2; 1yr 열은 n_years≤1 셀만), (b) CALM 사이트 연도 간 SD(n_years≥5), "
             "(b_detr) 사이트 내 연도 선형 추세 제거 잔차 SD, (c) SD/√n_years. Var(y)=평가 셀 관측 분산. 전이 RMSE는 알래스카 학습 정보 없음(l2_common.run_noinfo). "
             "레나 연도 간 SD는 셀 alt_sd(연간 최대값 연도 간 SD, n_years≥2) 기반.", elapsed_s=round(time.time() - t0, 1))
print(f"done {time.time()-t0:.0f}s")
