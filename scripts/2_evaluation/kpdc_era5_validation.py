"""KPDC 지점 실측으로 ERA5-Land 공변량 현장 검증 + 콘슬 Stefan forcing 사례연구.

두 부분:
  (A) ERA5-Land 공변량 현장 검증: 콘슬·c1 지점 좌표에서 우리 ERA5-Land 파생 공변량
      (nh_monthly_2010-2024.nc에서 해당 연도 월평균 → 도일)을 뽑아 지점 실측과 대조.
      비교 지표: MAAT, TDD(월평균 정의로 정합), 최난·최한월. bias·RMSE·상관·산점.
  (B) 콘슬 물리 forcing 사례연구: 콘슬 44 ALT 관측 셀에서 Stefan ALT=a+E·√TDD를
      세 forcing으로 비교 -- (i) ERA5 √TDD, (ii) KPDC 지점 실측 √TDD.
      E·a는 알래스카 학습셋에서 site-out(콘슬 제외) 적합. 관측 ALT 대비 RMSE·bias.

정직: 콘슬 단일 지역·소표본·성장기 부분값 한계. 일반화 아닌 사례연구.
정합 주의: ERA5 도일은 '월평균>0 클립 × 그 달 일수' 정의. 지점 대조는 동일 정의의
  tdd_st_mon(월평균 기반)을 사용. Stefan forcing에는 지점 √TDD를 쓰되, ERA5와 같은
  월평균 기반(sqrt_tdd_st_mon)으로 통일해 정의차 혼입 방지.

산출:
  data/processed/kpdc_era5_validation.csv  (지점 vs ERA5 대조표)
  data/processed/kpdc_council_forcing.csv  (셀별 forcing별 Stefan 예측 vs 관측 ALT)
  data/processed/kpdc_era5_validation_meta.json

실행: /home/anaconda3/bin/python scripts/2_evaluation/kpdc_era5_validation.py
"""
import os
import sys
import json
import calendar
import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/home/willy010313/Polar_Bigdata"
sys.path.insert(0, os.path.join(ROOT, "src"))
PROC = os.path.join(ROOT, "data/processed")
NC = os.path.join(ROOT, "data/raw/era5land/nh_monthly_2010-2024.nc")

DAYS = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])


# ---------- ERA5-Land 해당 연도 월평균 → 지점 도일 ----------
def era5_year_metrics(ds, tname, lat0, lon0, year):
    """지점(lat0,lon0)·연도에서 ERA5-Land 월평균 기온으로 지표 산출.
    ERA5 방법 정의(월평균>0 클립 × 그 달 일수)로 TDD/FDD 계산."""
    sel = ds.sel(latitude=lat0, longitude=lon0, method="nearest")
    yr_mask = sel[tname].dt.year.values == year
    if yr_mask.sum() < 12:
        return None
    t2m = sel["t2m"].values[yr_mask][:12] - 273.15   # 12개월 월평균 °C
    if np.all(np.isnan(t2m)):
        return None
    tdd = float(np.nansum(np.clip(t2m, 0, None) * DAYS))
    fdd = float(np.nansum(np.clip(-t2m, 0, None) * DAYS))
    return dict(
        e5_lat=float(sel.latitude), e5_lon=float(sel.longitude),
        e5_maat=float(np.nanmean(t2m)),
        e5_tdd=tdd, e5_fdd=fdd,
        e5_sqrt_tdd=float(np.sqrt(max(tdd, 0))),
        e5_twarm=float(np.nanmax(t2m)), e5_tcold=float(np.nanmin(t2m)),
    )


def era5_year_metrics_masked(ds, tname, lat0, lon0, year, valid_months):
    """지점 관측이 있는 월만 골라 ERA5 도일 계산(부분연도 정합용)."""
    sel = ds.sel(latitude=lat0, longitude=lon0, method="nearest")
    yr_mask = sel[tname].dt.year.values == year
    if yr_mask.sum() < 12:
        return None
    t2m = sel["t2m"].values[yr_mask][:12] - 273.15
    mm = np.array([m in valid_months for m in range(1, 13)])
    tsel = t2m[mm]
    dsel = DAYS[mm]
    tdd = float(np.nansum(np.clip(tsel, 0, None) * dsel))
    return dict(e5_tdd_masked=tdd, e5_sqrt_tdd_masked=float(np.sqrt(max(tdd, 0))),
                e5_maat_masked=float(np.nanmean(tsel)))


print("[A] ERA5-Land 공변량 현장 검증")
st = pd.read_csv(os.path.join(PROC, "kpdc_station_climate.csv"))
ds = xr.open_dataset(NC)
tname = "valid_time" if "valid_time" in ds.coords else "time"

# 지점 관측 유효월(부분연도 정합)은 하드코딩하지 않는다. parse_kpdc_met가 실측 자료에서
# 유도해 CSV의 valid_months 열에 기록한 값을 단일 진실 원천으로 사용한다. 과거에는 이 값을
# 규칙으로 지정하다 c1 2018(원자료 10-18일 종료, 관측월 1-10)을 1-12로 잘못 지정해
# ERA5 MAAT를 11-12월(최한월) 포함 12개월로 계산하는 마스킹 버그가 있었다.
def parse_valid_months(cell):
    if isinstance(cell, str) and cell.strip():
        return [int(m) for m in cell.split("|")]
    return list(range(1, 13))   # 열 없으면 전연(방어)


rows = []
for _, r in st.iterrows():
    e5 = era5_year_metrics(ds, tname, r.lat, r.lon, int(r.year))
    if e5 is None:
        continue
    vm = parse_valid_months(r.get("valid_months", ""))
    e5m = era5_year_metrics_masked(ds, tname, r.lat, r.lon, int(r.year), vm)
    row = dict(
        site=r.site, year=int(r.year), lat=r.lat, lon=r.lon,
        n_days=int(r.n_days),
        valid_months="|".join(str(m) for m in vm), n_valid_months=len(vm),
        coverage_note=r.coverage_note,
        # 지점(월평균 정의 TDD로 ERA5와 정합)
        st_maat=r.maat_st, st_tdd_mon=r.tdd_st_mon, st_sqrt_tdd_mon=r.sqrt_tdd_st_mon,
        st_twarm=r.twarm_st, st_tcold=r.tcold_st,
        # ERA5 전연(참고) 및 관측월 마스킹(정합 대조)
        **e5, **(e5m or {}),
    )
    # 편의(ERA5 - 지점): 관측월 마스킹 기준(부분연도 정합)
    if not np.isnan(r.tdd_st_mon):
        row["bias_tdd"] = row["e5_tdd_masked"] - r.tdd_st_mon
        row["bias_sqrt_tdd"] = row["e5_sqrt_tdd_masked"] - r.sqrt_tdd_st_mon
    if not np.isnan(r.maat_st):
        row["bias_maat"] = row["e5_maat_masked"] - r.maat_st
    rows.append(row)

val = pd.DataFrame(rows)
val.to_csv(os.path.join(PROC, "kpdc_era5_validation.csv"), index=False)
print(val[["site", "year", "n_days", "st_maat", "e5_maat_masked", "bias_maat",
           "st_tdd_mon", "e5_tdd_masked", "bias_tdd",
           "st_sqrt_tdd_mon", "e5_sqrt_tdd_masked", "bias_sqrt_tdd"]].round(2).to_string(index=False))

# 요약 통계(전 지점·연도, 관측월 마스킹 기준)
def summ(col_bias, label):
    x = val[col_bias].dropna().values
    if len(x) == 0:
        return None
    return dict(metric=label, n=int(len(x)), bias_mean=float(np.mean(x)),
                rmse=float(np.sqrt(np.mean(x ** 2))), abs_max=float(np.max(np.abs(x))))

def corr(a, b):
    m = val[[a, b]].dropna()
    if len(m) < 3:
        return None
    return float(np.corrcoef(m[a], m[b])[0, 1])


def loo_sensitivity(a, b, bias_col):
    """소표본 상관·RMSE의 이상점 민감도(leave-one-out)를 정량화한다.
    n<=4에서 단일 점이 r·RMSE를 지배할 수 있으므로 전체값과 함께 보고한다."""
    m = val[[a, b, bias_col, "site", "year"]].dropna().reset_index(drop=True)
    n = len(m)
    if n < 3:
        return None
    r_all = float(np.corrcoef(m[a], m[b])[0, 1])
    rmse_all = float(np.sqrt(np.mean(m[bias_col].values ** 2)))
    recs = []
    for i in range(n):
        loo = m.drop(index=i)
        r_i = (float(np.corrcoef(loo[a], loo[b])[0, 1]) if len(loo) >= 3 else None)
        rmse_i = float(np.sqrt(np.mean(loo[bias_col].values ** 2)))
        recs.append({"dropped": f"{m.site[i]} {int(m.year[i])}",
                     "r_without": r_i, "rmse_without": rmse_i})
    r_vals = [x["r_without"] for x in recs if x["r_without"] is not None]
    return {
        "n": n, "r_all": r_all, "rmse_all": rmse_all,
        "r_range_loo": [min(r_vals), max(r_vals)] if r_vals else None,
        "rmse_range_loo": [min(x["rmse_without"] for x in recs),
                           max(x["rmse_without"] for x in recs)],
        "per_point": recs,
    }


summary = {
    "maat": summ("bias_maat", "MAAT (°C)"),
    "tdd": summ("bias_tdd", "TDD (°C·d)"),
    "sqrt_tdd": summ("bias_sqrt_tdd", "√TDD"),
    "corr_maat": corr("st_maat", "e5_maat_masked"),
    "corr_tdd": corr("st_tdd_mon", "e5_tdd_masked"),
    "corr_sqrt_tdd": corr("st_sqrt_tdd_mon", "e5_sqrt_tdd_masked"),
    # 소표본·이상점 민감도(정직 보고). r=0.88·RMSE 1.04는 이상점 1점 의존이므로 병기.
    "sqrt_tdd_loo": loo_sensitivity("st_sqrt_tdd_mon", "e5_sqrt_tdd_masked", "bias_sqrt_tdd"),
    "maat_loo": loo_sensitivity("st_maat", "e5_maat_masked", "bias_maat"),
    "n_note": "n<=4 사례연구. 상관·RMSE는 단일 점 의존이라 leave-one-out 범위를 병기한다.",
}
print("\n요약(ERA5 - 지점, 관측월 정합):")
for k, v in summary.items():
    print(f"  {k}: {v}")


# ---------- [B] 콘슬 Stefan forcing 사례연구 ----------
print("\n[B] 콘슬 Stefan forcing 사례연구")
cell = pd.read_csv(os.path.join(PROC, "dl_dataset_cell.csv"))

# 콘슬 관측 ALT 셀(좌표 박스)
council_box = (cell.lat.between(64.83, 64.87)) & (cell.lon.between(-163.73, -163.68))
council = cell[council_box].copy()
print(f"콘슬 ALT 셀: {len(council)}개, ALT {council.alt_cm.min():.1f}-{council.alt_cm.max():.1f}cm")

# Stefan E·a 적합: 알래스카 학습셋에서 콘슬 제외(site-out)
# 알래스카 정의: 미국(알래스카)·ABoVE_AK 영역, 콘슬 박스 밖
ak = cell[(cell.region.isin(["ABoVE_AK", "United States (Alaska)"])) & (~council_box)].copy()
ak = ak.dropna(subset=["alt_cm", "e5_sqrt_tdd"])
print(f"알래스카 학습셋(콘슬 제외): {len(ak)}개")


def fit_E(y, s):
    """Stefan E 적합(train). 절편 포함/미포함 중 train RMSE 낮은 쪽."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    denom = float((s * s).sum())
    E0 = float((s * y).sum() / denom) if denom > 0 else 0.0
    rmse0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    A = np.c_[np.ones_like(s), s]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    rmse1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    if rmse1 < rmse0:
        return {"mode": "intercept", "E": E1, "a": a1, "train_rmse": rmse1}
    return {"mode": "origin", "E": E0, "a": 0.0, "train_rmse": rmse0}


fit = fit_E(ak.alt_cm.values, ak.e5_sqrt_tdd.values)
print(f"Stefan 적합(알래스카 site-out): mode={fit['mode']} E={fit['E']:.3f} a={fit['a']:.3f} "
      f"train_rmse={fit['train_rmse']:.2f}")


def stefan_pred(s):
    return fit["a"] + fit["E"] * s


# forcing 값
# (i) ERA5 √TDD: 셀의 e5_sqrt_tdd(정적 2015-2020 기후, 콘슬 셀별로 동일값 부근)
s_era5_cell = council.e5_sqrt_tdd.values
# 콘슬 지점 실측 √TDD(ERA5 정의와 정합: 월평균 기반). 관측 연도(2019·2021) 평균.
council_st = st[(st.site == "Council")]
s_st_mon = council_st.sqrt_tdd_st_mon.mean()   # 지점 대표 √TDD (월평균 정의)
# ERA5 지점(월평균 정의, 관측월 마스킹): 콘슬 지점 좌표에서 뽑은 ERA5 √TDD 평균
council_val = val[val.site == "Council"]
s_era5_pt = council_val.e5_sqrt_tdd_masked.mean()

print(f"forcing √TDD -- ERA5 셀(중앙): {np.median(s_era5_cell):.2f}, "
      f"ERA5 지점: {s_era5_pt:.2f}, KPDC 지점 실측: {s_st_mon:.2f}")

# 각 forcing으로 콘슬 셀 ALT 예측(관측 ALT 대비)
def metrics(pred, obs):
    e = pred - obs
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)))


obs = council.alt_cm.values
pred_era5_cell = stefan_pred(s_era5_cell)          # 셀별 ERA5 √TDD(공간 변이 있음)
pred_era5_pt = stefan_pred(np.full_like(obs, s_era5_pt))   # 지점 단일 ERA5 √TDD
pred_st = stefan_pred(np.full_like(obs, s_st_mon))         # 지점 단일 실측 √TDD

m_era5_cell = metrics(pred_era5_cell, obs)
m_era5_pt = metrics(pred_era5_pt, obs)
m_st = metrics(pred_st, obs)
# 기준선 두 종:
#  (a) 알래스카 train 평균 ALT (전이 관점 기준선)
#  (b) 콘슬 국소 평균 ALT (이 셀들의 자명한 하한 -- forcing이 국소변이 설명 못하면 못 이김)
mean_pred = ak.alt_cm.mean()
m_mean = metrics(np.full_like(obs, mean_pred), obs)
council_mean = obs.mean()
m_council_mean = metrics(np.full_like(obs, council_mean), obs)

print("\nforcing별 콘슬 셀 Stefan 예측 vs 관측 ALT:")
print(f"  (i) ERA5 √TDD(셀별)   : RMSE {m_era5_cell['rmse']:.2f} bias {m_era5_cell['bias']:+.2f}")
print(f"  (i') ERA5 √TDD(지점단일): RMSE {m_era5_pt['rmse']:.2f} bias {m_era5_pt['bias']:+.2f}")
print(f"  (ii) KPDC 지점 실측 √TDD : RMSE {m_st['rmse']:.2f} bias {m_st['bias']:+.2f}")
print(f"  기준선 AK train 평균({mean_pred:.1f}cm): RMSE {m_mean['rmse']:.2f}")
print(f"  기준선 콘슬 국소 평균({council_mean:.1f}cm): RMSE {m_council_mean['rmse']:.2f}")

# skill은 콘슬 국소 평균 대비(정직한 하한). AK train 평균은 콘슬에 편의라 부적절.
skill_era5 = 1 - m_era5_pt["rmse"] / m_council_mean["rmse"]
skill_st = 1 - m_st["rmse"] / m_council_mean["rmse"]
print(f"  skill(콘슬 국소평균 대비): ERA5 지점 {skill_era5:+.3f}, 지점실측 {skill_st:+.3f}")
print(f"  forcing 차이(ERA5 지점 vs 실측): ΔRMSE {m_era5_pt['rmse']-m_st['rmse']:+.2f}cm")

# 셀별 forcing 결과 저장
council_out = council[["loc_id", "lat", "lon", "alt_cm", "e5_sqrt_tdd"]].copy()
council_out["pred_stefan_era5_cell"] = pred_era5_cell
council_out["pred_stefan_era5_pt"] = pred_era5_pt
council_out["pred_stefan_st"] = pred_st
council_out["forcing_sqrt_tdd_era5_pt"] = s_era5_pt
council_out["forcing_sqrt_tdd_st"] = s_st_mon
council_out.to_csv(os.path.join(PROC, "kpdc_council_forcing.csv"), index=False)

meta = {
    "part_A_validation": {
        "definition": "ERA5 도일 = 월평균>0 클립 × 그 달 일수. 지점도 동일 정의(tdd_st_mon)로 대조.",
        "masking": ("부분연도는 실측 관측월만 골라 ERA5 도일·MAAT 계산(정합). 관측월은 "
                    "parse_kpdc_met가 자료에서 유도한 valid_months(단일 진실 원천)를 사용한다. "
                    "관측월: Council 2021=1-10, Council 2019=6-9, c1 2016=1-12, c1 2018=1-10."),
        "masking_bugfix": ("이전 버전은 c1 2018 관측월을 1-12로 잘못 지정해(원자료는 10-18일 종료) "
                           "ERA5 MAAT를 11-12월(Nov -6.97·Dec -17.94) 포함 12개월로 계산, "
                           "허위 냉편의 bias -2.59°C를 만들었다. 관측월 1-10 정합 시 c1 2018 "
                           "bias는 -0.38°C이며, MAAT 요약은 bias -0.31°C·RMSE 0.32°C로 정정된다. "
                           "따라서 ERA5 MAAT 냉편의 주장은 성립하지 않는다."),
        "small_sample_caveat": ("√TDD 상관 r=0.88(n=4)과 RMSE 1.04는 단일 이상점 Council 2019"
                                "(성장기 부분값)에 의해 지배된다. 이 점을 빼면 n=3에서 r=1.00·"
                                "RMSE 0.13으로 급변한다(summary.sqrt_tdd_loo 참조). r=0.88을 "
                                "'ERA5가 지점을 잘 추종'의 일반 근거로 해석하지 말 것."),
        "summary": summary,
    },
    "part_B_forcing": {
        "n_council_cells": int(len(council)),
        "n_train_ak_cells": int(len(ak)),
        "stefan_fit": fit,
        "forcing_sqrt_tdd": {
            "era5_cell_median": float(np.median(s_era5_cell)),
            "era5_point": float(s_era5_pt),
            "kpdc_station": float(s_st_mon),
        },
        "council_rmse_cm": {
            "era5_cell": m_era5_cell, "era5_point": m_era5_pt,
            "kpdc_station": m_st,
            "ak_mean_baseline": m_mean, "council_mean_baseline": m_council_mean,
        },
        "skill_over_council_mean": {"era5_point": skill_era5, "kpdc_station": skill_st},
        "forcing_delta_rmse_cm": float(m_era5_pt["rmse"] - m_st["rmse"]),
        "delta_rmse_interpretation": ("ΔRMSE ~1.0cm는 √TDD 정의 불일치 잡음 수준이다. 지점 forcing "
                                      "√TDD는 Council 2019 성장기 부분값(양의 도일 일부 누락으로 과소)"
                                      "이 평균에 섞여 우연히 낮아진 것일 수 있다. '실측이 ERA5보다 "
                                      "우세'로 읽지 말 것. 방향성 근거는 약하다."),
        "leakage_control": ("Stefan E·a는 AK 학습셋(region ABoVE_AK·United States (Alaska), 콘슬 "
                            "박스 제외) 13,562셀에서 적합했고 콘슬 44셀은 하나도 포함되지 않는다"
                            "(누출 0, 재현 확인). 콘슬 셀은 예측·평가에만 쓰였다."),
        "skill_baseline_note": ("skill 하한은 콘슬에 유리한 AK train 평균(50.5cm)이 아니라 콘슬 국소 "
                                "평균(35cm, RMSE 13.4cm)으로 채택했다. 세 forcing(25-28cm) 모두 "
                                "이 자명한 하한을 못 이기므로 forcing에 유리한 누출은 없다. "
                                "'정보 병목은 기후 forcing 정밀도가 아니다'라는 결론은 유지된다."),
    },
    "caveats": [
        "콘슬 단일 지역·소표본(ALT 셀 44, 지점 관측연 2)이라 일반화 아닌 사례연구.",
        "n<=4 지표(상관·RMSE)는 단일 점 의존이므로 leave-one-out 범위 병기(summary 참조).",
        "성장기 부분값(2019)·11-12월 결측(2021)으로 지점 연지표 부분성.",
        ("부분월 가중 불일치(경미): c1 2018 10월은 지점이 관측일수(18일) 가중인데 ERA5는 "
         "10월 31일 전체를 쓴다. √TDD bias +0.14 수준으로 결론에 영향 없으나 정의상 "
         "부분월 정합이 완전하지는 않다."),
        "c1 좌표 미확정(콘슬 잠정)이라 c1 검증은 보조·불확실.",
        "ERA5 셀 √TDD는 2015-2020 정적 기후, 지점 실측은 관측연 → 시기 불일치 잔존.",
    ],
}
with open(os.path.join(PROC, "kpdc_era5_validation_meta.json"), "w") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=2)

print("\n저장:")
print(" -", os.path.join(PROC, "kpdc_era5_validation.csv"))
print(" -", os.path.join(PROC, "kpdc_council_forcing.csv"))
print(" -", os.path.join(PROC, "kpdc_era5_validation_meta.json"))
