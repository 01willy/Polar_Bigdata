"""ALLena 융해깊이(PANGAEA 973813, Lena Delta 일대) → 위치별 ALT 셀 라벨.

규칙
- 월 8-9만 사용(계절 최대 근사). 6-7월 관측은 연중 최대의 하한이므로 제외.
- QF TD==2(측정한계 초과, probe 길이보다 깊음)는 제외(우측 censored).
- QF TD==1(그림에서 추출된 값)은 qc_flag=1로 보존.
- (lat.round(4), lon.round(4), year)별 max(Thaw depth mean) = 연중 최대 융해깊이 근사.
- 위치별 재집계: alt_cm=연도별 최대의 평균, alt_sd=연도간 SD(단일연도 0.0).

산출: data/processed/alt_allena_cell.csv
      (lat,lon,region,alt_cm,alt_sd,alt_min,alt_max,n_obs,year_min,year_max,n_years,qc_flag)
실행: /home/anaconda3/bin/python scripts/1_data_prep/parse_allena.py  (ROOT에서)
"""
import numpy as np
import pandas as pd

SRC = "data/raw/allena/PANGAEA_973813_ALLena_main.txt"
OUT = "data/processed/alt_allena_cell.csv"

df = pd.read_csv(SRC, sep="\t", skiprows=437)
print(f"[load] {SRC}: {df.shape[0]:,}행 x {df.shape[1]}열")


def col(prefix):
    hits = [c for c in df.columns if c.startswith(prefix)]
    assert len(hits) == 1, f"컬럼 패턴 '{prefix}' 매칭 {len(hits)}개: {hits}"
    return hits[0]


LAT = col("Latitude (")
LON = col("Longitude (")
DT = col("Date/Time (")
TD = col("Thaw depth mean [cm]")
QF = col("QF TD")
NOBS = col("NOBS")

df["_dt"] = pd.to_datetime(df[DT])
df["year"] = df["_dt"].dt.year
df["month"] = df["_dt"].dt.month

n0 = len(df)
df = df[df["month"].isin([8, 9])]
print(f"[filter] 월 8-9만: {n0:,} -> {len(df):,}행 (6-7월은 계절 하한이라 제외)")

n1 = len(df)
df = df[df[QF] != 2]
print(f"[filter] QF TD==2(측정한계 초과) 제외: {n1:,} -> {len(df):,}행")

df = df.dropna(subset=[TD, LAT, LON])
df["qc_flag"] = (df[QF] == 1).astype(int)
print(f"[qc] QF TD==1(그림 추출) 보존: {df.qc_flag.sum():,}행 (qc_flag=1)")

df["klat"] = df[LAT].round(4)
df["klon"] = df[LON].round(4)

# 위치·연도별 연중 최대 융해깊이
yr = (df.groupby(["klat", "klon", "year"])
        .agg(td_max=(TD, "max"), n_obs=(TD, "size"), qc_flag=("qc_flag", "max"))
        .reset_index())

# 위치별 집계
loc = (yr.groupby(["klat", "klon"])
         .agg(alt_cm=("td_max", "mean"),
              alt_sd=("td_max", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
              alt_min=("td_max", "min"), alt_max=("td_max", "max"),
              n_obs=("n_obs", "sum"),
              year_min=("year", "min"), year_max=("year", "max"),
              n_years=("year", "nunique"),
              qc_flag=("qc_flag", "max"))
         .reset_index()
         .rename(columns={"klat": "lat", "klon": "lon"}))
loc.insert(2, "region", "Lena_RU")
loc["alt_sd"] = loc["alt_sd"].fillna(0.0)

loc.to_csv(OUT, index=False)
print(f"[saved] {OUT}: {len(loc):,} 위치 "
      f"(alt_cm 중앙 {loc.alt_cm.median():.1f}cm, 5/95% "
      f"{np.percentile(loc.alt_cm, [5, 95]).round(1)}, "
      f"다연도 위치 {(loc.n_years > 1).sum():,}, qc_flag=1 위치 {loc.qc_flag.sum():,})")
print(f"[bbox] lat {loc.lat.min():.3f}~{loc.lat.max():.3f}, "
      f"lon {loc.lon.min():.3f}~{loc.lon.max():.3f}, 연도 {loc.year_min.min()}~{loc.year_max.max()}")
