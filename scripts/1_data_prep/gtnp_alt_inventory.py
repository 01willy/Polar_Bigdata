"""S-A 5항: GTN-P ALT 106 데이터셋 편입 검토(분석만, 편입하지 않음).

입력: data/raw/gtnp/alt_manifest.json(106 데이터셋; activelayer_id·method·statistics),
      data/raw/gtnp/sites.json(92 사이트; activelayers[].id/lat/lon), data/raw/gtnp/alt_csv/*.csv(관측 행),
      data/processed/fidelity_base_v3.csv(중복 판정 대상 셀), data/processed/alt_global.csv(CALM 사이트명 대조)
판정
  - dup_v3_001: fidelity_base_v3 셀과 체비쇼프 0.01° 이내 → 중복(편입 불가)
  - same_calm_site: 0.01° 초과이나 alt_global.csv(CALM 원자료)의 사이트명이 일치하고 0.1° 이내 → 같은 CALM 격자의 좌표 정밀도 차이(중복 취급)
  - new_coord: 위 둘 다 아님 → 신규 좌표
  - no_coords: activelayer_id가 sites.json에 없어 좌표 미상
산출: data/processed/gtnp_alt_inventory.csv, gtnp_alt_inventory_meta.json
실행(ROOT): python3 scripts/1_data_prep/gtnp_alt_inventory.py
"""
from __future__ import annotations
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
GT = ROOT / "data" / "raw" / "gtnp"
DUP_DEG = 0.01
NAME_DEG = 0.1
NAME_DEG_WIDE = 0.5        # GTN-P activelayer 좌표가 사이트 대표 좌표(소수 1자리)인 경우(알래스카 U 사이트)


def macro_of(country: str, lat: float, lon: float) -> str:
    if country == "United States":
        return "Alaska" if (lat > 50 and lon < -130) else "US_other"
    if country in ("Canada", "Greenland", "Svalbard", "Antarctica", "Mongolia", "Switzerland", "Norway", "Austria",
                   "Kazakhstan", "Sweden", "China"):
        return {"Mongolia": "Mongolia_CAsia", "Kazakhstan": "Mongolia_CAsia", "Switzerland": "Alps", "Austria": "Alps",
                "China": "Tibet", "Norway": "Scandinavia", "Sweden": "Scandinavia"}.get(country, country)
    if country == "Russia":
        return "Russia_W" if lon < 90 else ("Russia_C" if lon < 140 else "Russia_E")
    return country


def norm(s: str) -> str:
    """소문자·영숫자만. 러시아어 전사 변형(Jakutskoe/Yakutskoe)은 j→y로 통일."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower()).replace("j", "y")


man = json.load(open(GT / "alt_manifest.json"))
sites = json.load(open(GT / "sites.json"))
al = {a["id"]: dict(site_id=x["id"], site_name=x["name"], country=x["country_name"], al_name=a["name"],
                    lat=a["latitude"], lon=a["longitude"]) for x in sites for a in x["activelayers"]}

rows = []
for d in man:
    f = glob.glob(str(GT / "alt_csv" / f"alt_dataset_{d['id']}_*.csv"))
    r = dict(dataset_id=d["id"], variable=d["variable"], method=(d.get("method") or {}).get("name"),
             activelayer_id=d["activelayer_id"], quality=d.get("quality"), file=os.path.basename(f[0]) if f else None)
    r.update(al.get(d["activelayer_id"], dict(site_id=None, site_name=None, country=None, al_name=None, lat=np.nan, lon=np.nan)))
    if f:
        t = pd.read_csv(f[0])
        col = "alt" if "alt" in t.columns else "soil_moisture"
        v = pd.to_numeric(t[col], errors="coerce")
        yr = pd.to_datetime(t.date, errors="coerce").dt.year
        if r["site_id"] is None and "site_id" in t.columns and len(t):
            r["site_id"] = int(t.site_id.iloc[0])          # sites.json에 없는 사이트 id(좌표 미상)
        r.update(n_rows=int(len(t)), n_valid=int(v.notna().sum()),
                 n_positions=int(t[["offset_x", "offset_y"]].drop_duplicates().shape[0]),
                 year_min=int(yr.min()) if yr.notna().any() else None, year_max=int(yr.max()) if yr.notna().any() else None,
                 n_years=int(yr.nunique()), value_mean=round(float(v.mean()), 1) if v.notna().any() else None,
                 value_median=round(float(v.median()), 1) if v.notna().any() else None)
    rows.append(r)
inv = pd.DataFrame(rows)
inv["is_alt"] = inv.variable.str.startswith("Active Layer")

# ---- 중복 판정: v3 셀 0.01° ----
v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "region", "source_id"], low_memory=False)
has = inv.lat.notna()
dist, j = cKDTree(v3[["lat", "lon"]].values).query(inv.loc[has, ["lat", "lon"]].values, p=np.inf)
inv.loc[has, "near_v3_dist"] = dist
inv.loc[has, "near_v3_loc_id"] = v3.loc_id.values[j]
inv.loc[has, "near_v3_region"] = v3.region.values[j]
inv["dup_v3_001"] = inv.near_v3_dist < DUP_DEG

# ---- CALM 원자료(alt_global.csv) 사이트명·좌표 대조 ----
g = pd.read_csv(PROC / "alt_global.csv", low_memory=False).dropna(subset=["lat", "lon"])
gs = g.groupby(["site", "name", "country"], as_index=False).agg(lat=("lat", "first"), lon=("lon", "first"))
gs["nname"] = gs.name.map(norm)
dist2, j2 = cKDTree(gs[["lat", "lon"]].values).query(inv.loc[has, ["lat", "lon"]].values, p=np.inf)
inv.loc[has, "near_calm_site"] = gs.site.values[j2]
inv.loc[has, "near_calm_name"] = gs.name.values[j2]
inv.loc[has, "near_calm_dist"] = dist2


STOP = ("grid", "site", "creek", "lake", "river", "island", "north", "south", "west", "east")


def name_match(r, deg):
    """al_name 토큰(4자 이상, 일반어 제외)이 deg° 이내 CALM 사이트명에 포함되면 True."""
    if not isinstance(r.al_name, str) or not np.isfinite(r.lat):
        return False
    toks = [t for t in re.split(r"[\s,;()/-]+", r.al_name) if len(t) >= 4 and t.lower() not in STOP]
    cand = gs[(np.abs(gs.lat - r.lat) < deg) & (np.abs(gs.lon - r.lon) < deg)]
    return any(norm(t) in n for t in toks for n in cand.nname)


inv["calm_name_match"] = inv.apply(lambda r: name_match(r, NAME_DEG), axis=1)
inv["calm_name_match_wide"] = inv.apply(lambda r: name_match(r, NAME_DEG_WIDE), axis=1)
inv["macro"] = [macro_of(c, la, lo) if isinstance(c, str) else None for c, la, lo in zip(inv.country, inv.lat, inv.lon)]
inv["status"] = np.select(
    [~has, inv.dup_v3_001, inv.calm_name_match, inv.calm_name_match_wide],
    ["no_coords", "dup_v3_001", "same_calm_site", "same_calm_site_coarse"], default="new_coord")
inv.loc[~inv.is_alt, "status"] = inv.loc[~inv.is_alt, "status"].map(lambda s: "non_alt:" + s)

cols = ["dataset_id", "variable", "method", "activelayer_id", "site_id", "site_name", "al_name", "country", "macro", "lat", "lon",
        "n_rows", "n_valid", "n_positions", "year_min", "year_max", "n_years", "value_mean", "value_median",
        "near_v3_dist", "near_v3_loc_id", "near_v3_region", "dup_v3_001", "near_calm_site", "near_calm_name", "near_calm_dist",
        "calm_name_match", "calm_name_match_wide", "status", "quality", "file"]
inv = inv[cols].sort_values(["status", "country", "lat"]).reset_index(drop=True)
inv.to_csv(PROC / "gtnp_alt_inventory.csv", index=False)

# ---- 요약 ----
alt = inv[inv.variable.str.startswith("Active Layer")]
print(f"[inv] 데이터셋 {len(inv)} (ALT {len(alt)} · 토양수분 {len(inv) - len(alt)}) · 좌표 있음 {int(alt.lat.notna().sum())}")
print(alt.status.value_counts().to_string())
print("\n[status × macro] ALT 데이터셋 수")
print(pd.crosstab(alt.macro.fillna("(좌표 미상)"), alt.status).to_string())
newc = alt[alt.status == "new_coord"].copy()
newc["coord"] = newc.lat.round(3).astype(str) + "," + newc.lon.round(3).astype(str)
print("\n[new_coord] 데이터셋")
print(newc[["dataset_id", "site_name", "al_name", "country", "macro", "lat", "lon", "n_valid", "year_min", "year_max", "n_years",
            "value_mean", "near_v3_dist", "near_v3_region", "near_calm_site", "near_calm_dist"]].round(3).to_string(index=False))
uniq = newc.groupby("macro").coord.nunique()
print("\n[new_coord] 매크로별 고유 좌표 수:", uniq.to_dict())
print("\n[same_calm_site(_coarse)] 데이터셋(CALM 동일 사이트, 좌표 정밀도 차이)")
sc = alt[alt.status.str.startswith("same_calm_site")]
print(sc[["dataset_id", "al_name", "country", "macro", "lat", "lon", "near_calm_site", "near_calm_name", "near_calm_dist", "near_v3_dist", "near_v3_region"]].round(3).to_string(index=False))
print("\n[no_coords] site_id(sites.json에 없음):", sorted(set(int(s) for s in alt[alt.status == "no_coords"].site_id.dropna())))

v3c = v3.groupby(v3.region.map(lambda r: r)).size()
meta = dict(created="2026-09-18", n_datasets=int(len(inv)), n_alt=int(len(alt)),
            status_counts=alt.status.value_counts().to_dict(),
            new_coord_unique_by_macro={k: int(v) for k, v in uniq.items()},
            new_coord_datasets=newc[["dataset_id", "al_name", "country", "macro", "lat", "lon", "n_valid", "year_min", "year_max", "n_years", "value_mean"]].round(3).to_dict("records"),
            same_calm_site_datasets=sc[["dataset_id", "al_name", "near_calm_site", "near_calm_name", "near_calm_dist", "near_v3_region"]].round(3).to_dict("records"),
            no_coords_site_ids=sorted(set(int(s) for s in alt[alt.status == "no_coords"].site_id.dropna())),
            rules=dict(dup=f"v3 셀 체비쇼프 {DUP_DEG}° 이내", same_site=f"alt_global.csv 사이트명 토큰 일치 + {NAME_DEG}° 이내",
                       same_site_coarse=f"사이트명 토큰 일치 + {NAME_DEG_WIDE}° 이내(GTN-P 대표 좌표가 소수 1자리인 경우)"),
            verdict_rule="편입 가치: 신규 매크로 지역 ≥ 1 또는 기존 소표본 지역 셀 ≥ 2배")
(PROC / "gtnp_alt_inventory_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (o.item() if hasattr(o, "item") else str(o))))
print("[done] gtnp_alt_inventory.csv · gtnp_alt_inventory_meta.json")
