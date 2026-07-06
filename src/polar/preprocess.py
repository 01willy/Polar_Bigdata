"""
② 전처리 — 원본 GTN-P borehole CSV → 분석용 평형(MAGT) 테이블.

단계:
  1) 데이터셋별 CSV(id,date,depth,temperature,flag,...) 통합 → long table
  2) QC: 결측/불량 flag 제거, 좌표 join
  3) (borehole, depth)별 MAGT(평형 연평균) + 연진폭 산정
  4) borehole별 DZAA(연진폭 0 깊이) + 지온경사(심부 선형) + 0 °C base 외삽

산출:
  data/interim/alaska_long_table.parquet   QC된 (x,y,z,t,T) long table
  data/processed/borehole_profiles.csv     (borehole,depth)별 MAGT/연진폭
  data/processed/borehole_summary.csv      borehole별 요약(DZAA/지온경사/base)
"""
import numpy as np
import pandas as pd

from . import config as C

# "양호"로 간주하지 않을 flag 키워드(소문자 부분일치)
BAD_FLAG_KEYS = ("no data", "missing", "invalid", "bad", "error")


def load_long_table():
    """PT_CSV_DIR의 모든 pt_dataset_*.csv를 통합하고 좌표를 join한 long table."""
    files = sorted(C.PT_CSV_DIR.glob("pt_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {C.PT_CSV_DIR}. run 01_download_gtnp.py first.")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # 좌표 join (boreholes.csv: borehole_id,lat,lon,elev,country,site)
    bh = pd.read_csv(C.BOREHOLES_CSV)
    df = df.merge(bh[["borehole_id", "lat", "lon", "elev", "site"]],
                  on="borehole_id", how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def qc(df):
    """결측 온도/불량 flag/좌표 없음 제거."""
    n0 = len(df)
    df = df.dropna(subset=["temperature", "depth", "date", "lat", "lon"])
    flag = df["flag"].astype(str).str.lower()
    bad = flag.apply(lambda f: any(k in f for k in BAD_FLAG_KEYS))
    df = df[~bad]
    # 지중(양수 깊이)만: 표층/공기(음수 depth) 제외
    df = df[df["depth"] >= 0]
    # 물리 범위(센서 오류 제거): 지중온도 [-40, 30] °C
    df = df[df["temperature"].between(-40, 30)]
    print(f"  QC: {n0:,} -> {len(df):,} rows")
    return df.reset_index(drop=True)


def compute_magt(df):
    """(borehole, depth)별 평형 MAGT + 연진폭. 시계열·연간스냅샷 혼합 대응.

    월평균 -> 월 기후값(연간 평균) -> MAGT = 가용 월들의 평균.
      - 일/월 시계열: 12개월 기후값 평균 = 비편향 MAGT
      - 연간 스냅샷(여름 1회): 해당 월만 존재 -> 그 평균(심부는 무편향, 천부는 여름 warm bias)
    연진폭: 월 기후값이 충분(>=8개월)할 때만 (max-min)/2, 아니면 NaN.
    sampling: 'seasonal'(>=8개월) / 'sparse'(<8개월, 주로 연간) 플래그.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    mon = (df.groupby(["borehole_id", "depth", "year", "month"])["temperature"]
             .mean().reset_index())
    clim = (mon.groupby(["borehole_id", "depth", "month"])["temperature"]
              .mean().reset_index())                       # 월 기후값
    agg = (clim.groupby(["borehole_id", "depth"])
               .agg(magt=("temperature", "mean"),
                    t_max=("temperature", "max"),
                    t_min=("temperature", "min"),
                    n_months=("month", "nunique")).reset_index())
    meta = (mon.groupby(["borehole_id", "depth"])
               .agg(n_years=("year", "nunique"),
                    n_obs=("temperature", "size")).reset_index())
    prof = agg.merge(meta, on=["borehole_id", "depth"])
    prof["amp"] = np.where(prof["n_months"] >= 8,
                           (prof["t_max"] - prof["t_min"]) / 2.0, np.nan)
    prof["sampling"] = np.where(prof["n_months"] >= 8, "seasonal", "sparse")
    prof = prof.drop(columns=["t_max", "t_min"])

    bh = pd.read_csv(C.BOREHOLES_CSV)[["borehole_id", "lat", "lon", "elev", "site"]]
    prof = prof.merge(bh, on="borehole_id", how="left").sort_values(["borehole_id", "depth"])
    return prof.reset_index(drop=True)


def summarize_boreholes(prof):
    """borehole별: 최대깊이, 표층 MAGT, DZAA, 지온경사(°C/m), 0 °C base 깊이 외삽."""
    rows = []
    for bid, g in prof.groupby("borehole_id"):
        g = g.sort_values("depth")
        depths = g["depth"].to_numpy()
        magt = g["magt"].to_numpy()
        amp = g["amp"].to_numpy()
        lat, lon, elev, site = g[["lat", "lon", "elev", "site"]].iloc[0]

        # DZAA: 연진폭 < threshold 인 가장 얕은 깊이(연진폭 산정 가능한 경우)
        below = g[g["amp"] < C.DZAA_AMP_THRESHOLD]
        dzaa = float(below["depth"].min()) if len(below) else np.nan

        # 깊이순 정렬 후 가장 깊은 ~40%(최소 MIN_DEPTHS)로 지온경사 선형적합
        order = np.argsort(depths)
        d_s, m_s = depths[order], magt[order]
        k = max(C.MIN_DEPTHS_FOR_GRADIENT, int(np.ceil(0.4 * len(d_s))))
        deep_d, deep_m = d_s[-k:], m_s[-k:]
        grad = base0 = np.nan
        if len(np.unique(deep_d)) >= C.MIN_DEPTHS_FOR_GRADIENT:
            a, b = np.polyfit(deep_d, deep_m, 1)
            grad = float(a)                       # °C/m (양수=깊을수록 따뜻)
            # base 외삽: 심부에서 (1)아래로 따뜻해지고 (2)심부가 얼어있을 때만, 물리적 범위에서
            if a > 0.003 and m_s[-1] < 0:
                cross = -b / a
                if d_s.max() < cross < 1500:
                    base0 = float(cross)          # magt=0 깊이 = 영구동토 base 추정

        surf_magt = float(m_s[0]) if len(m_s) else np.nan
        deep_magt = float(m_s[-1]) if len(m_s) else np.nan
        # 계절편향 적은 지표: 10 m(없으면 [5,20]m 최근접) MAGT
        cand = g[(g["depth"] >= 5) & (g["depth"] <= 20)]
        magt_10m = (float(cand.iloc[(cand["depth"] - 10).abs().argmin()]["magt"])
                    if len(cand) else np.nan)
        permafrost = bool(np.nanmin(m_s) < 0) if len(m_s) else False
        n_sparse = int((g["sampling"] == "sparse").sum())

        rows.append(dict(borehole_id=bid, site=site, lat=lat, lon=lon, elev=elev,
                         n_depths=len(depths), max_depth=float(depths.max()),
                         surface_magt=surf_magt, magt_10m=magt_10m, deep_magt=deep_magt,
                         dzaa=dzaa, geo_gradient_C_per_m=grad,
                         base_depth_0C=base0, permafrost_present=permafrost,
                         sampling="sparse" if n_sparse > len(g) / 2 else "seasonal"))
    return pd.DataFrame(rows).sort_values("max_depth", ascending=False).reset_index(drop=True)


def run():
    C.ensure_dirs()
    print("Loading + QC ...")
    df = qc(load_long_table())
    df.to_parquet(C.LONGTABLE_PARQUET, index=False)
    print(f"  long table -> {C.LONGTABLE_PARQUET}  ({len(df):,} rows, "
          f"{df['borehole_id'].nunique()} boreholes)")

    print("Computing MAGT profiles ...")
    prof = compute_magt(df)
    prof.to_csv(C.PROFILES_CSV, index=False)
    print(f"  profiles -> {C.PROFILES_CSV}  ({len(prof):,} (borehole,depth) rows)")

    print("Summarizing boreholes ...")
    summ = summarize_boreholes(prof)
    summ.to_csv(C.SUMMARY_CSV, index=False)
    print(f"  summary -> {C.SUMMARY_CSV}  ({len(summ)} boreholes)")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summ.head(12).to_string(index=False))
    return df, prof, summ
