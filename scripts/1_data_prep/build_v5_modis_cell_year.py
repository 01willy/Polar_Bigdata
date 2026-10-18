"""v5 MODIS 식생·적설 셀-연도 입력(재설정 문서 5.1 단계 A5). AppEEARS 점 추출 결과(data/raw/appeears_xe/results/)에서 만든다.

입력
- 식생 MOD13Q1.061(250 m, 16일): NDVI·EVI(AppEEARS CSV 는 배율 적용값. 중앙값이 1.5 를 넘으면 1e-4 를 곱한다), pixel_reliability(0 = 양호, 1 = 한계). 요청 기간 2015–2020 의 6–8월.
- 적설 MOD10A2.061(500 m, 8일): Maximum_Snow_Extent(200 = 눈, 25 = 눈 없음, 그 밖은 구름·결측 등). 요청 기간 2014-09-01 – 2021-08-31.
- 점 ↔ 셀: data/raw/appeears_xe/points/xe_labels_pixels.csv(loc_id → pix250_id, pix500_id).
변수(셀, 연도 y)
  ndvi_summer, evi_summer   6–8월 pixel_reliability ≤ 1 인 합성값 평균(×1e-4). n_vi = 쓴 합성 수
  snow_off_doy              수문년 Oct(y−1)–Sep(y) 에서 1–7월 마지막 '눈' 8일 합성의 시작 DOY(눈 소실 시기)
  snow_on_doy               가을(9–12월, y−1) 첫 '눈' 합성 시작 DOY
  snow_days                 수문년 안 '눈' 합성 수 × 8 (대략 적설 일수)
  n_snow_valid              눈/눈 없음으로 판정된 합성 수(구름·결측 제외)
다년 평균 열(ndvi_summer_mean 등)도 만든다. 라벨 값은 읽지 않는다.
출력: data/processed/v5/modis_cell_year_v5.csv, modis_cell_year_v5_meta.json
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser(); _ap.add_argument("--src", default="appeears_xe", help="data/raw/<src>/results"); _ap.add_argument("--prefix", default="xe", help="결과 폴더 접두(xe | v5)")
_ap.add_argument("--out-name", default="modis_cell_year_v5.csv"); _a = _ap.parse_args()
RES = ROOT / "data" / "raw" / _a.src / "results"
PTS = ROOT / "data" / "raw" / "appeears_xe" / "points" / "xe_labels_pixels.csv"   # v5 요청도 같은 점(픽셀) 목록을 썼다
OUT = ROOT / "data" / "processed" / "v5"


def read_many(pattern, usecols):
    fs = sorted(glob.glob(str(RES / pattern)))
    parts = [pd.read_csv(f, usecols=lambda c: c in usecols or c in ("ID", "Date"), low_memory=False) for f in fs]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), fs


def main():
    t0 = time.time()
    pts = pd.read_csv(PTS)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), files={})
    # ---------------- 식생
    vi_cols = {"MOD13Q1_061__250m_16_days_NDVI", "MOD13Q1_061__250m_16_days_EVI", "MOD13Q1_061__250m_16_days_pixel_reliability"}
    vi, fs = read_many(f"{_a.prefix}_mod13q1_labels_*/*-results.csv", vi_cols)
    meta["files"]["mod13q1_labels"] = [Path(f).name for f in fs]
    vi = vi.rename(columns={"MOD13Q1_061__250m_16_days_NDVI": "ndvi", "MOD13Q1_061__250m_16_days_EVI": "evi",
                            "MOD13Q1_061__250m_16_days_pixel_reliability": "rel"})
    vi["dt"] = pd.to_datetime(vi.Date); vi["year"] = vi.dt.dt.year; vi["month"] = vi.dt.dt.month
    vi = vi[vi.month.isin([6, 7, 8]) & (vi.rel <= 1) & vi.ndvi.notna()]
    scale = 1e-4 if np.nanmedian(np.abs(vi.ndvi.values)) > 1.5 else 1.0          # AppEEARS CSV 는 보통 배율이 이미 적용돼 있다
    meta["vi_scale_applied"] = scale
    vi["ndvi"] = vi.ndvi * scale; vi["evi"] = vi.evi * scale
    g = vi.groupby(["ID", "year"]).agg(ndvi_summer=("ndvi", "mean"), evi_summer=("evi", "mean"), n_vi=("ndvi", "size")).reset_index()
    g = g[g.n_vi >= 2]
    vi_cell = pts[["loc_id", "pix250_id"]].merge(g.rename(columns={"ID": "pix250_id"}), on="pix250_id", how="inner").drop(columns="pix250_id")
    # ---------------- 적설
    sn_cols = {"MOD10A2_061_Maximum_Snow_Extent"}
    sn, fs = read_many(f"{_a.prefix}_mod10a2_labels_*/*-results.csv", sn_cols)
    meta["files"]["mod10a2_labels"] = [Path(f).name for f in fs]
    sn = sn.rename(columns={"MOD10A2_061_Maximum_Snow_Extent": "mse"})
    sn["dt"] = pd.to_datetime(sn.Date); sn["doy"] = sn.dt.dt.dayofyear; sn["cal"] = sn.dt.dt.year; sn["month"] = sn.dt.dt.month
    sn["hy"] = np.where(sn.month >= 10, sn.cal + 1, sn.cal)                     # 수문년 = 끝나는 해
    sn["snow"] = np.where(sn.mse == 200, 1.0, np.where(sn.mse == 25, 0.0, np.nan))
    sv = sn.dropna(subset=["snow"])
    def per_hy(s):
        spring = s[(s.month <= 7) & (s.snow == 1)]
        fall = s[(s.month >= 9) & (s.cal == s.hy - 1) & (s.snow == 1)]
        return pd.Series(dict(snow_off_doy=float(spring.doy.max()) if len(spring) else np.nan,
                              snow_on_doy=float(fall.doy.min()) if len(fall) else np.nan,
                              snow_days=float(s.snow.sum() * 8), n_snow_valid=int(len(s))))
    h = sv.groupby(["ID", "hy"]).apply(per_hy).reset_index().rename(columns={"hy": "year"})
    h = h[h.n_snow_valid >= 20]
    sn_cell = pts[["loc_id", "pix500_id"]].merge(h.rename(columns={"ID": "pix500_id"}), on="pix500_id", how="inner").drop(columns="pix500_id")
    # ---------------- 결합·다년 평균
    out = vi_cell.merge(sn_cell, on=["loc_id", "year"], how="outer").sort_values(["loc_id", "year"])
    for c in ("ndvi_summer", "evi_summer", "snow_off_doy", "snow_on_doy", "snow_days"):
        out[f"{c}_mean"] = out.loc_id.map(out.groupby("loc_id")[c].mean())
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / _a.out_name, index=False)
    meta.update(rows=int(len(out)), cells=int(out.loc_id.nunique()), years_vi=sorted(int(y) for y in vi_cell.year.unique()),
                years_snow=sorted(int(y) for y in sn_cell.year.unique()), cells_vi=int(vi_cell.loc_id.nunique()), cells_snow=int(sn_cell.loc_id.nunique()),
                rules=dict(vi="6–8월 pixel_reliability ≤ 1, 합성 2개 이상", snow="MOD10A2 200 = 눈, 25 = 눈 없음, 수문년 유효 합성 20개 이상",
                           snow_off="1–7월 마지막 눈 합성 시작 DOY", snow_on="전년 9–12월 첫 눈 합성 시작 DOY"),
                summary={c: dict(median=float(out[c].median()), p10=float(out[c].quantile(.1)), p90=float(out[c].quantile(.9)))
                         for c in ("ndvi_summer", "evi_summer", "snow_off_doy", "snow_on_doy", "snow_days") if out[c].notna().any()},
                elapsed_s=round(time.time() - t0, 1))
    meta["src"] = _a.src; json.dump(meta, open(OUT / _a.out_name.replace(".csv", "_meta.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
