# -*- coding: utf-8 -*-
"""build_v9_external.py: 외부 지역 무라벨 시험용 셀-연도 표(2026-10-15, 사용자 질문 5 '추가 분석 가능한 지역').

자료: CALM 활동층 두께(PANGAEA 972777, CC BY 4.0) 2010–2024. 세 연구 지역(알래스카·캐나다 북서부·레나 델타)과 산악 영구동토(스위스·카자흐스탄)를 뺀다.
규칙
- 지점 → v4 셀(loc_id): 최근접 v4 셀(1 km 안). 같은 셀·같은 해 값은 평균.
- 세 지역 라벨 셀에서 100 km 안 지점은 뺀다(레나 델타 안 사모일로프 R51, 비코프스키 R29A·C).
- 검열값('>260' 등)과 이른 측정·화재·침수·탐침 미도달·비활성 표기 행은 뺀다.
- CALM 은 계절 말 측정 규약(관측 월 미상) → obs_months 'eos', obs_doy 결측(시험지 규칙상 9월 1일로 누적).
- 정적 공변량은 v4 표(fidelity_base_v4.csv)의 같은 열. 기후 입력과 제품 값은 v5 구축 스크립트를 --cells 로 다시 쓴다.
출력: data/processed/v9_ext/{labels_cell_year_v5.csv, static_v5.csv, forcing_year.csv, forcing_clim.csv, products_cell_year_v5.csv, build_meta.json}
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/1_data_prep/build_v9_external.py
"""
from __future__ import annotations
import json, subprocess, sys, time
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"; OUT = PROC / "v9_ext"; OUT.mkdir(parents=True, exist_ok=True)
CALM = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"
EXCL_COUNTRY = {"United States (Alaska)", "Canada", "Switzerland", "Kazakstan"}
AREA_KEY = {"West Siberia": "W Siberia", "Central Siberia": "C Siberia", "North East Siberia": "NE Siberia", "Chukotka": "Chukotka",
            "Russian European north": "European Russia", "Kamchatka": "Kamchatka", "Svalbard": "Svalbard"}
t0 = time.time(); meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), source="PANGAEA 972777 (CALM, CC BY 4.0)")

txt = CALM.read_text(encoding="utf-8", errors="ignore"); i = txt.find("*/")
d = pd.read_csv(StringIO(txt[i + 2:].lstrip("\n")), sep="\t", dtype={"Date/Time": str})
d["year"] = d["Date/Time"].str[:4].astype(int)
d = d[d.year.between(2010, 2024) & d["ALD [cm]"].notna() & ~d.Country.isin(EXCL_COUNTRY)].copy()
d["alt_num"] = pd.to_numeric(d["ALD [cm]"], errors="coerce")
meta["censored_removed"] = int((d.alt_num.isna()).sum())                       # '>260', '<460' 같은 검열값
QC_BAD = r"early|July|August1|burn|flood|not reached|inactive|not reliable|deeper than lowest|thermistors broken"
bad = d["Sample comment"].fillna("").str.contains(QC_BAD, case=False, regex=True)
meta["qc_comment_removed"] = int((bad & d.alt_num.notna()).sum()); meta["qc_rule"] = f"Sample comment 정규식 제외: {QC_BAD}; 검열값 제외"
d = d[d.alt_num.notna() & ~bad].copy(); d["ALD [cm]"] = d.alt_num
d["area"] = d.Area.fillna(d.Country).map(lambda a: AREA_KEY.get(a, "Greenland" if "Greenland" in str(a) else str(a)))
meta["raw_site_years"] = int(len(d))

v4 = pd.read_csv(PROC / "fidelity_base_v4.csv", low_memory=False)
tree = BallTree(np.radians(v4[["lat", "lon"]].values), metric="haversine")
dist, idx = tree.query(np.radians(d[["Latitude", "Longitude"]].values), k=1)
d["loc_id"] = v4.loc_id.values[idx[:, 0]]; d["d_cell_km"] = dist[:, 0] * 6371.0
d = d[d.d_cell_km < 1.0]
lab5 = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv", usecols=["lat", "lon"]).drop_duplicates()
t5 = BallTree(np.radians(lab5.values), metric="haversine")
d["d_study_km"] = t5.query(np.radians(d[["Latitude", "Longitude"]].values), k=1)[0][:, 0] * 6371.0
excl = d[d.d_study_km <= 100.0]
meta["excluded_near_study_regions"] = sorted(set(f"{s} {n}" for s, n in zip(excl.Site, excl.Name)))
d = d[d.d_study_km > 100.0]

g = d.groupby(["loc_id", "year"]).agg(alt_cm=("ALD [cm]", "mean"), alt_sd=("ALD [cm]", "std"), n_obs=("ALD [cm]", "size"), area=("area", "first"),
                                      sites=("Site", lambda s: ";".join(sorted(set(s))))).reset_index()
cell = v4.set_index("loc_id").loc[g.loc_id, ["lat", "lon"]].reset_index(drop=True)
g["lat"], g["lon"] = cell.lat.values, cell.lon.values
g["region"] = g.area; g["source"] = "CALM"; g["in_forcing"] = 1; g["obs_months"] = "eos"; g["obs_doy"] = np.nan
g["method"] = "probe_grid"; g["license"] = "CC BY 4.0"; g["qc_fig"] = 0
g["block"] = (np.floor(g.lat / 0.5).astype(int) * 100000 + np.floor(g.lon / 0.5).astype(int))
cols = ["loc_id", "region", "source", "lat", "lon", "block", "year", "in_forcing", "alt_cm", "alt_sd", "n_obs", "obs_months", "obs_doy", "method", "license", "qc_fig", "sites"]
g[cols].to_csv(OUT / "labels_cell_year_v5.csv", index=False)
meta["labels"] = dict(cell_years=int(len(g)), cells=int(g.loc_id.nunique()), blocks=int(g.block.nunique()),
                      by_region={r: dict(cell_years=int(len(s)), cells=int(s.loc_id.nunique()), years=[int(s.year.min()), int(s.year.max())])
                                 for r, s in g.groupby("region")})

st5 = pd.read_csv(PROC / "v5" / "static_v5.csv", nrows=1)
st = v4[v4.loc_id.isin(g.loc_id.unique())].copy()
for c in st5.columns:
    if c not in st:
        st[c] = np.nan
st["region"] = st.loc_id.map(g.drop_duplicates("loc_id").set_index("loc_id").region)
st[st5.columns].to_csv(OUT / "static_v5.csv", index=False)
meta["static_missing_share"] = {c: float(st[c].isna().mean()) for c in st5.columns if c.startswith(("dem_", "sg_"))}
print(f"[labels] {meta['labels']['cell_years']} cell-years, {meta['labels']['cells']} cells; excluded {meta['excluded_near_study_regions']}", flush=True)

rel = str(OUT.relative_to(ROOT))
r1 = subprocess.run([sys.executable, "scripts/1_data_prep/build_v5_forcing_year.py", "--cells", f"{rel}/static_v5.csv", "--out-prefix", f"{rel}/forcing"], cwd=ROOT,
                    capture_output=True, text=True)
meta["forcing_rc"] = r1.returncode; meta["forcing_log_tail"] = (r1.stdout + r1.stderr)[-1500:]
print(f"[forcing] rc {r1.returncode}", flush=True)
r2 = subprocess.run([sys.executable, "scripts/1_data_prep/build_v5_products_cell_year.py", "--cells", f"{rel}/static_v5.csv", "--products", "cci4y,cci5y,wei,liu,aalto",
                     "--out", f"{rel}/products_cell_year_v5.csv"], cwd=ROOT, capture_output=True, text=True)
meta["products_rc"] = r2.returncode; meta["products_log_tail"] = (r2.stdout + r2.stderr)[-1500:]
print(f"[products] rc {r2.returncode}", flush=True)
meta["elapsed_s"] = round(time.time() - t0, 1)
json.dump(meta, open(OUT / "build_meta.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: meta[k] for k in ("labels", "forcing_rc", "products_rc", "elapsed_s")}, ensure_ascii=False)[:1500])
