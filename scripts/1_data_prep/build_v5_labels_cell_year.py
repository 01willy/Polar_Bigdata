"""v5 셀-연도 라벨 표(재설정 문서 docs/RESET_2026-10-10_KOPRI_FEEDBACK.md 5.1 단계 A1·A3).

목표 변수 ALT(c, y) = 셀 c, 연도 y 의 계절 말 융해 깊이(관측 평균). 셀은 v4 loc_id(공변량 동일 단위)를 그대로 쓴다.

규칙
- 셀 식별: ABoVE·CALM 은 dl_dataset.csv 의 (lat 4자리, lon 4자리) → loc_id. ALLENA 는 alt_allena_cell.csv 의 (lat, lon) → v4 loc_id.
- 계절 말 관측만 쓴다. ABoVE 는 월 8–10(1·2·5·6·7월 관측은 'early' 로 표시하고 뺀다), ALLENA 는 월 8–9(기존 parse_allena 규칙),
  CALM 은 프로토콜상 계절 말 관측(월 미상, obs_months 'eos').
- QC: ABoVE 0 < ALT < 300 cm, 좌표 범위(parse_above_alt.py 와 같다). ALLENA QF TD != 2(측정 한계 초과 제외), 결측 제외.
- 집계: ABoVE·CALM 은 (loc_id, year) 평균. ALLENA 는 위치·연도별 최대(기존 규칙)이며 위치 = 셀이므로 그대로 쓴다.
- 지역: Alaska = ABoVE_AK + CALM United States (Alaska), Lena = ALLENA, Canada = ABoVE_CA + CALM Canada.
  캐나다 확충(F4_ext_direct, LGD)은 관측 수준 연도가 있고 약관이 확인된 자료만 뒤에 더한다(이 판에서는 넣지 않고 메타에 적는다).
- 모든 연도를 표에 남기고 in_forcing(2010–2024) 열로 표시한다. 분석은 in_forcing == 1 만 쓴다.

출력: data/processed/v5/labels_cell_year_v5.csv, static_v5.csv(v4 정적 공변량 + 2015–2020 기후값 열, 대조용), labels_cell_year_v5_meta.json
실행: python3 scripts/1_data_prep/build_v5_labels_cell_year.py
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TERRAIN, SOIL, INSAR, POLSAR, CCI, FLAGS, CLIMATE  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "v5"; OUT.mkdir(exist_ok=True)
ABOVE = ROOT / "data" / "raw" / "above" / "ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv"
ALLENA = ROOT / "data" / "raw" / "allena" / "PANGAEA_973813_ALLena_main.txt"
FORCING_YEARS = (2010, 2024)
EOS_ABOVE = (8, 9, 10)
EOS_ALLENA = (8, 9)
KEY = 4


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def months_str(s):
    return ";".join(str(m) for m in sorted(set(int(v) for v in s)))


t0 = time.time()
v4 = pd.read_csv(PROC / "fidelity_base_v4.csv", low_memory=False)
dl = pd.read_csv(PROC / "dl_dataset.csv", usecols=["lat", "lon", "year", "alt_cm", "source", "region", "loc_id"], low_memory=False)
dl["klat"], dl["klon"] = dl.lat.round(KEY), dl.lon.round(KEY)
key2loc = dl.drop_duplicates(["klat", "klon"]).set_index(["klat", "klon"]).loc_id
assert key2loc.index.is_unique
meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rules": {}, "counts": {}, "inputs": {}}

# ---------------------------------------------------------------- ABoVE (Alaska, Canada)
ab = pd.read_csv(ABOVE, usecols=["latitude", "longitude", "date", "ALT", "ALT_instrument", "survey_technique"], low_memory=False)
n_raw = len(ab)
ab = ab[(ab.ALT != -9999) & ab.ALT.notna() & (ab.latitude != -9999) & (ab.longitude != -9999)]
ab["dt"] = pd.to_datetime(ab.date, errors="coerce")
ab = ab.dropna(subset=["dt"])
ab = ab[(ab.ALT > 0) & (ab.ALT < 300) & ab.latitude.between(45, 80) & ab.longitude.between(-170, -60)]
ab["year"], ab["month"], ab["doy"] = ab.dt.dt.year, ab.dt.dt.month, ab.dt.dt.dayofyear
ab["klat"], ab["klon"] = ab.latitude.round(KEY), ab.longitude.round(KEY)
ab["loc_id"] = pd.MultiIndex.from_frame(ab[["klat", "klon"]]).map(key2loc)
n_noloc = int(ab.loc_id.isna().sum())
ab = ab.dropna(subset=["loc_id"]); ab["loc_id"] = ab.loc_id.astype(int)
reg = v4.set_index("loc_id").region
ab["region_v4"] = ab.loc_id.map(reg)
ab = ab[ab.region_v4.isin(["ABoVE_AK", "ABoVE_CA"])]
early = ~ab.month.isin(EOS_ABOVE)
meta["counts"]["above"] = dict(raw=n_raw, qc=int(len(ab) + early.sum() * 0), no_loc=n_noloc, early_excluded=int(early.sum()),
                               early_by_month=ab[early].month.value_counts().sort_index().to_dict())
ab = ab[~early]
g = (ab.groupby(["loc_id", "year"]).agg(alt_cm=("ALT", "mean"), alt_sd=("ALT", "std"), n_obs=("ALT", "size"),
                                         obs_months=("month", months_str), obs_doy=("doy", "mean"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                                         region_v4=("region_v4", "first")).reset_index())
g["region"] = g.region_v4.map({"ABoVE_AK": "Alaska", "ABoVE_CA": "Canada"})
g["source"] = "ABoVE_ds2369"; g["method"] = "probe"; g["license"] = "EOSDIS free"
above = g.drop(columns=["region_v4"])

# ---------------------------------------------------------------- CALM (Alaska, Canada)
calm = pd.read_csv(PROC / "alt_global.csv", usecols=["site", "name", "country", "lat", "lon", "year", "alt_cm"])
calm = calm[calm.country.isin(["United States (Alaska)", "Canada"])].dropna(subset=["alt_cm"])
calm["klat"], calm["klon"] = calm.lat.round(KEY), calm.lon.round(KEY)
calm["loc_id"] = pd.MultiIndex.from_frame(calm[["klat", "klon"]]).map(key2loc)
n_calm_noloc = int(calm.loc_id.isna().sum())
calm = calm.dropna(subset=["loc_id"]); calm["loc_id"] = calm.loc_id.astype(int)
calm["region_v4"] = calm.loc_id.map(reg)
calm = calm[calm.region_v4.isin(["United States (Alaska)", "Canada"])]
gc = (calm.groupby(["loc_id", "year"]).agg(alt_cm=("alt_cm", "mean"), alt_sd=("alt_cm", "std"), n_obs=("alt_cm", "size"),
                                           lat=("lat", "mean"), lon=("lon", "mean"), region_v4=("region_v4", "first")).reset_index())
gc["obs_months"] = "eos"; gc["obs_doy"] = np.nan                     # CALM: 날짜 없음(계절 말 프로토콜)
gc["region"] = gc.region_v4.map({"United States (Alaska)": "Alaska", "Canada": "Canada"})
gc["source"] = "CALM"; gc["method"] = "probe_grid"; gc["license"] = "CALM (citation)"
calm_cy = gc.drop(columns=["region_v4"])
meta["counts"]["calm"] = dict(site_years=int(len(calm)), no_loc=n_calm_noloc, cell_years=int(len(calm_cy)))

# ---------------------------------------------------------------- ALLENA (Lena)
al = pd.read_csv(ALLENA, sep="\t", skiprows=437)
def col(prefix):
    hits = [c for c in al.columns if c.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]
LAT, LON, DT, TD, QF = col("Latitude ("), col("Longitude ("), col("Date/Time ("), col("Thaw depth mean [cm]"), col("QF TD")
al["dt"] = pd.to_datetime(al[DT]); al["year"], al["month"], al["doy"] = al.dt.dt.year, al.dt.dt.month, al.dt.dt.dayofyear
n_al_raw = len(al)
al = al[al.month.isin(EOS_ALLENA) & (al[QF] != 2)].dropna(subset=[TD, LAT, LON])
al["qc_fig"] = (al[QF] == 1).astype(int)
al["klat"], al["klon"] = al[LAT].round(4), al[LON].round(4)
yr = (al.groupby(["klat", "klon", "year"]).agg(alt_cm=(TD, "max"), alt_sd=(TD, "std"), n_obs=(TD, "size"),
                                              obs_months=("month", months_str), obs_doy=("doy", "mean"), qc_fig=("qc_fig", "max")).reset_index())
lena_v4 = v4[v4.region == "Lena_RU"][["loc_id", "lat", "lon"]].copy()
lena_v4["klat"], lena_v4["klon"] = lena_v4.lat.round(4), lena_v4.lon.round(4)
yr = yr.merge(lena_v4[["loc_id", "klat", "klon", "lat", "lon"]], on=["klat", "klon"], how="inner")
yr["region"] = "Lena"; yr["source"] = "ALLENA_PANGAEA_973813"; yr["method"] = "probe"; yr["license"] = "CC BY 4.0"
lena_cy = yr.drop(columns=["klat", "klon"])
meta["counts"]["allena"] = dict(raw=n_al_raw, eos_qc=int(len(al)), cell_years=int(len(lena_cy)), cells=int(lena_cy.loc_id.nunique()),
                                qc_fig_cell_years=int(lena_cy.qc_fig.sum()))

# ---------------------------------------------------------------- 결합
lab = pd.concat([above, calm_cy, lena_cy], ignore_index=True, sort=False)
lab["qc_fig"] = lab.qc_fig.fillna(0).astype(int)
lab["alt_sd"] = lab.alt_sd.fillna(0.0)
lab["in_forcing"] = lab.year.between(*FORCING_YEARS).astype(int)
lab["block"] = (np.floor(lab.lat / 0.5).astype(int) * 100000 + np.floor(lab.lon / 0.5).astype(int))
lab = lab.sort_values(["region", "loc_id", "year"]).reset_index(drop=True)
cols = ["loc_id", "region", "source", "lat", "lon", "block", "year", "in_forcing", "alt_cm", "alt_sd", "n_obs", "obs_months", "obs_doy", "method",
        "license", "qc_fig"]
lab = lab[cols]
assert not lab.duplicated(["loc_id", "year"]).any()
lab.to_csv(OUT / "labels_cell_year_v5.csv", index=False)

# ---------------------------------------------------------------- 정적 공변량(v4 에서 loc_id 로)
stat_cols = ["loc_id", "lat", "lon", "region"] + TERRAIN + SOIL + INSAR + POLSAR + CCI + FLAGS + CLIMATE
st = v4[v4.loc_id.isin(lab.loc_id.unique())][stat_cols].copy()
st = st.rename(columns={c: c.replace("e5_", "e5clim1520_") for c in CLIMATE}).rename(columns={"region": "region_v4"})
st["region"] = st.loc_id.map(lab.drop_duplicates("loc_id").set_index("loc_id").region)
st.to_csv(OUT / "static_v5.csv", index=False)

# ---------------------------------------------------------------- 메타
f = lab[lab.in_forcing == 1]
meta["rules"] = dict(eos_months_above=list(EOS_ABOVE), eos_months_allena=list(EOS_ALLENA), calm="protocol end-of-season, month unknown",
                     above_qc="0<ALT<300 cm, lat 45–80, lon −170–−60", allena_qc="QF TD != 2, drop NA", cell="v4 loc_id",
                     forcing_years=list(FORCING_YEARS), aggregation="ABoVE/CALM mean per (loc_id, year); ALLENA max per (location, year)")
meta["counts"]["total"] = dict(cell_years=int(len(lab)), cells=int(lab.loc_id.nunique()), cell_years_in_forcing=int(len(f)),
                               cells_in_forcing=int(f.loc_id.nunique()))
meta["counts"]["by_region_in_forcing"] = {r: dict(cell_years=int(len(s)), cells=int(s.loc_id.nunique()), blocks=int(s.block.nunique()),
                                                   years=sorted(int(y) for y in s.year.unique()),
                                                   multi_year_cells=int((s.groupby("loc_id").year.nunique() > 1).sum()))
                                          for r, s in f.groupby("region")}
meta["counts"]["by_region_source_all_years"] = {f"{r}|{s}": int(n) for (r, s), n in lab.groupby(["region", "source"]).size().items()}
meta["counts"]["excluded_pre2010_cell_years"] = {r: int(n) for r, n in lab[lab.in_forcing == 0].groupby("region").size().items()}
meta["inputs"] = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size)
                  for p in [ABOVE, ALLENA, PROC / "alt_global.csv", PROC / "dl_dataset.csv", PROC / "fidelity_base_v4.csv"]}
meta["outputs"] = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size, rows=int(len(pd.read_csv(p, usecols=[0]))))
                   for p in [OUT / "labels_cell_year_v5.csv", OUT / "static_v5.csv"]}
meta["elapsed_s"] = round(time.time() - t0, 1)
json.dump(meta, open(OUT / "labels_cell_year_v5_meta.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(meta["counts"], ensure_ascii=False, indent=1))
