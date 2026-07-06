"""
전 지구 ALT(활성층 두께) 학습 데이터셋 구축.

소스: PANGAEA CALM 컴파일 (Streletskiy et al. 2025, GTN-P/CALM, DOI 10.1594/PANGAEA.972777)
      — 263개 NH 사이트, site-year ALT(cm) + lat/lon/연도/국가.
      (CALM이 GTN-P ALT를 먹이므로 이 컴파일이 전 지구 ALT의 종합본)

처리: tab 파싱 → 정제 → WorldClim 공변량 부착 → data/processed/alt_global.csv
"""
import numpy as np
import pandas as pd

from . import config as C
from . import covariates as cov

CALM_TAB = C.DATA / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"
ALT_GLOBAL = C.PROCESSED / "alt_global.csv"


def parse_pangaea_calm(path=CALM_TAB):
    lines = path.read_text(encoding="utf-8").splitlines()
    di = next(i for i, l in enumerate(lines) if l.strip() == "*/")
    header = lines[di + 1].split("\t")
    rows = [l.split("\t") for l in lines[di + 2:] if l.strip()]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"Latitude": "lat", "Longitude": "lon", "Date/Time": "year",
                            "ALD [cm]": "alt_cm", "Name": "name", "Country": "country",
                            "Area": "area", "Site": "site", "ID": "calm_id"})
    for c in ["lat", "lon", "alt_cm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "alt_cm", "year"])
    df = df[(df.alt_cm > 0) & (df.alt_cm < 600)]            # 물리 범위 QC
    return df[["site", "name", "calm_id", "country", "area", "lat", "lon", "year", "alt_cm"]]


def build():
    C.ensure_dirs()
    df = parse_pangaea_calm()
    print(f"  CALM: {len(df):,} site-year, {df.site.nunique()} 사이트, "
          f"{df.country.nunique()} 국가, {int(df.year.min())}~{int(df.year.max())}")
    print("  국가별 사이트 수:")
    sc = df.groupby("country")["site"].nunique().sort_values(ascending=False)
    for k, v in sc.head(12).items():
        print(f"    {v:3d}  {k}")

    # WorldClim 공변량 (사이트 위치에서 샘플)
    cov.download_worldclim()
    wc = cov.sample_worldclim(df["lon"].to_numpy(), df["lat"].to_numpy())
    for var, vals in wc.items():
        df[f"wc_{var}"] = vals
    df.to_csv(ALT_GLOBAL, index=False)
    print(f"\n  saved {ALT_GLOBAL}  ({len(df):,} 행, {df.shape[1]} 열)")
    print("  ALT(cm) 5/50/95%:", np.nanpercentile(df.alt_cm, [5, 50, 95]).round(0))
    return df


if __name__ == "__main__":
    build()
