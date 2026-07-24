"""KPDC 콘슬(Council)·c1 현장 기상 파싱 → 지점 연도별 기후 지표.

목표: 대회 요구(KPDC 데이터 활용)의 정직한 실현. 각 원시 파일을 일평균 기온으로
정리한 뒤 지점 연도별 지표를 유도한다. 지표: MAAT(연평균, 부분연도 표기),
TDD(융해도일, 일평균>0 누적), FDD(동결도일), √TDD, 최난·최한월.

주의: ERA5-Land 파생 공변량은 '월평균 기온'에서 도일을 계산한다(월평균>0 클립 후
일수 가중). 지점 실측 검증의 정합을 위해 두 가지 TDD를 함께 산출한다.
  - tdd_st       : 일평균 기반(물리적으로 올바른 정의). 결측일은 제외(부분연도 표기).
  - tdd_st_mon   : 월평균 기반(ERA5 방법과 동일 정의). ERA5 대조 전용.

산출: data/processed/kpdc_station_climate.csv
  컬럼: site, lat, lon, year, n_days, maat_st, tdd_st, fdd_st, sqrt_tdd_st,
        tdd_st_mon, sqrt_tdd_st_mon, twarm_st, tcold_st, coverage_note
메타:  data/processed/kpdc_station_climate_meta.json

실행: /home/anaconda3/bin/python scripts/1_data_prep/parse_kpdc_met.py
"""
import os
import json
import calendar
import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
KPDC = os.path.join(ROOT, "kpdc")
PROC = os.path.join(ROOT, "data/processed")
os.makedirs(PROC, exist_ok=True)

# 좌표 (CLAUDE 지시·데이터셋 ALT 셀 위치 확인)
# 콘슬(Council) 약 64.85N, -163.70W: dl_dataset_cell에 44 ALT 셀 존재(64.841~64.860N).
# c1 좌표는 KPDC 메타로 확정 불가 → 콘슬 좌표로 잠정, 불확실 표기(coverage_note).
COORD = {
    "Council_2021": (64.85, -163.70),
    "Council_2019": (64.85, -163.70),
    "c1_2016": (64.85, -163.70),   # 불확실: c1 좌표 미확정, 콘슬 좌표 잠정
    "c1_2018": (64.85, -163.70),   # 불확실: c1 좌표 미확정, 콘슬 좌표 잠정
}


def daily_climate(dser, site, lat, lon, year, cover):
    """일평균 기온 시계열(dser: index=날짜, value=일평균 degC)에서 지표 산출."""
    d = dser.dropna()
    n_days = int(len(d))
    if n_days == 0:
        return None
    # 일평균 기반 도일(물리 정의)
    tdd = float(np.clip(d.values, 0, None).sum())
    fdd = float(np.clip(-d.values, 0, None).sum())
    maat = float(d.mean())
    # 월평균 기반 도일(ERA5 방법 정의): 월평균>0 클립 × 그 달 관측일수
    dm = d.copy()
    dm.index = pd.to_datetime(dm.index)
    mon_mean = dm.groupby(dm.index.month).mean()
    mon_ndays = dm.groupby(dm.index.month).size()
    tdd_mon = float((np.clip(mon_mean.values, 0, None) * mon_ndays.values).sum())
    twarm = float(mon_mean.max())
    tcold = float(mon_mean.min())
    # 실측 관측월(ERA5 대조 마스킹의 단일 진실 원천). 하드코딩 대신 자료에서 유도.
    valid_months = sorted(int(m) for m in mon_mean.index.tolist())
    return dict(
        site=site, lat=lat, lon=lon, year=int(year), n_days=n_days,
        maat_st=round(maat, 3), tdd_st=round(tdd, 2), fdd_st=round(fdd, 2),
        sqrt_tdd_st=round(float(np.sqrt(max(tdd, 0))), 3),
        tdd_st_mon=round(tdd_mon, 2),
        sqrt_tdd_st_mon=round(float(np.sqrt(max(tdd_mon, 0))), 3),
        twarm_st=round(twarm, 3), tcold_st=round(tcold, 3),
        valid_months="|".join(str(m) for m in valid_months),
        coverage_note=cover,
    )


recs = []
notes = []

# --- Council 2021: 반시간별 2mTair (Time (UTC), Air Temp (degC)) ---
f = os.path.join(KPDC, "council", "aws_met", "Council_2021_2mTair.csv")
d = pd.read_csv(f)
d["t"] = pd.to_datetime(d["Time (UTC)"], errors="coerce")
d = d.dropna(subset=["t"])
d["tair"] = pd.to_numeric(d["Air Temp (degC)"], errors="coerce")
d = d[d["tair"] > -900]                     # 결측코드 방어(관측상 없음, 안전)
day = d.set_index("t")["tair"].resample("1D").mean()
# 전년도 일단위 인덱스로 재색인 → 관측 없는 말일도 결측(NaN)으로 명시
day = day.reindex(pd.date_range("2021-01-01", "2021-12-31", freq="1D"))
lat, lon = COORD["Council_2021"]
frac = day.notna().sum() / 365.0
# 결측이 11-12월(최한월)에 집중 → TDD(융해기)는 완전, FDD·연 MAAT는 과소
miss_mon = sorted(day[day.isna()].index.month.unique().tolist())
cov = (f"연 {day.notna().sum()}일 관측(커버 {frac:.0%}); 결측월={miss_mon}(최한월); "
       f"TDD(1-10월 융해기)는 완전·신뢰, FDD·연 MAAT는 과소(11-12월 결측); 반시간→일평균")
r = daily_climate(day, "Council", lat, lon, 2021, cov)
if r:
    recs.append(r)
notes.append(f"Council 2021: {day.notna().sum()} valid days, MAAT={r['maat_st']}")

# --- Council 2019: 일별 gap-filled MET (탭구분, Group.1=날짜, Tair_f=기온, 결측 -9999) ---
f = os.path.join(KPDC, "council", "aws_met", "COUNCIL_2019_MET_f_day.txt")
d = pd.read_csv(f, sep="\t")
d = d[d["Group.1"] != "-"].copy()
d["t"] = pd.to_datetime(d["Group.1"], errors="coerce")
d = d.dropna(subset=["t"])
d["tair"] = pd.to_numeric(d["Tair_f"], errors="coerce")
d.loc[d["tair"] <= -9000, "tair"] = np.nan
day = d.set_index("t")["tair"]
lat, lon = COORD["Council_2019"]
mons = sorted(pd.to_datetime(day.dropna().index).month.unique())
cov = (f"성장기 부분관측만({min(mons)}-{max(mons)}월, {day.notna().sum()}일); "
       f"연 MAAT 산출 불가(성장기 부분값); TDD는 성장기 부분 누적")
r = daily_climate(day, "Council", lat, lon, 2019, cov)
if r:
    # 부분연도이므로 연평균 지표는 오해 소지 → maat_st를 NaN 처리(성장기 평균은 twarm에 반영)
    r["maat_st"] = np.nan
    recs.append(r)
notes.append(f"Council 2019: {day.notna().sum()} growing-season days (months {mons})")

# --- c1 2016 / 2018: 시간별 (Date, Air Temp @ 1m/3m, 결측 6999.0) ---
for yr, fn in [(2016, "c1_met_2016 (1).csv"), (2018, "c1_met_2018.csv")]:
    f = os.path.join(KPDC, "c1_toolik", "aws_met", fn)
    d = pd.read_csv(f)
    d["t"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["t"])
    # 3m 우선(2018 1m 대량 결측), 결측 시 1m 보완
    t3 = pd.to_numeric(d["Air Temp @ 3m"], errors="coerce")
    t1 = pd.to_numeric(d["Air Temp @ 1m"], errors="coerce")
    t3[t3 >= 6000] = np.nan
    t1[t1 >= 6000] = np.nan
    tair = t3.fillna(t1)
    d = d.assign(tair=tair.values)
    day = d.set_index("t")["tair"].resample("1D").mean()
    lat, lon = COORD[f"c1_{yr}"]
    days_in_year = 366 if calendar.isleap(yr) else 365   # 윤년(2016) 분모 보정
    n_valid = int(day.notna().sum())
    frac = n_valid / days_in_year
    mons_valid = sorted(pd.to_datetime(day.dropna().index).month.unique().tolist())
    cov = (f"c1 지점(좌표 불확실, 콘슬 잠정); 연 {n_valid}/{days_in_year}일 관측"
           f"(커버 {frac:.1%}); 관측월={mons_valid}; 3m 기온 우선·1m 보완; 시간별→일평균")
    r = daily_climate(day, "c1", lat, lon, yr, cov)
    if r:
        recs.append(r)
    notes.append(f"c1 {yr}: {day.notna().sum()} valid days, MAAT={r['maat_st']}")

out = pd.DataFrame(recs)
cols = ["site", "lat", "lon", "year", "n_days", "maat_st", "tdd_st", "fdd_st",
        "sqrt_tdd_st", "tdd_st_mon", "sqrt_tdd_st_mon", "twarm_st", "tcold_st",
        "valid_months", "coverage_note"]
out = out[cols]
opath = os.path.join(PROC, "kpdc_station_climate.csv")
out.to_csv(opath, index=False)

meta = {
    "purpose": "KPDC 콘슬·c1 현장 기상 → 지점 연도별 기후 지표(ERA5 검증·물리 forcing용)",
    "n_records": int(len(out)),
    "sites": sorted(out.site.unique().tolist()),
    "years": sorted(int(y) for y in out.year.unique()),
    "tdd_definitions": {
        "tdd_st": "일평균>0 누적(물리 정의). 결측일 제외(부분연도).",
        "tdd_st_mon": "월평균>0 클립 × 관측일수(ERA5-Land 방법과 동일 정의). ERA5 대조 전용.",
    },
    "valid_months_note": ("valid_months = 실측 자료에 존재하는 월(pipe 구분). ERA5 대조 마스킹의 "
                          "단일 진실 원천이다. c1 2018은 원자료가 10-18일에 끝나 관측월=1-10월이다."),
    "coord_note": "콘슬 64.85N/-163.70W(ALT 44셀 확인). c1 좌표 미확정 → 콘슬 좌표 잠정·불확실 표기.",
    "coverage_notes": notes,
    "missing_codes_removed": [-9999, 6999.0],
}
with open(os.path.join(PROC, "kpdc_station_climate_meta.json"), "w") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=2)

print(out.to_string(index=False))
print("\n저장:", opath)
for n in notes:
    print(" -", n)
