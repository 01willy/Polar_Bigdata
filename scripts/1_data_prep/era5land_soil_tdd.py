"""E2 F7용 신규 공변량: ERA5-Land 토양 0–7 cm 온도(stl1) 월별 기후값에서 토양 도일 산출.

동기
----
Stefan 식의 강제력은 지표(토양 상단) 온도인데 현행 e5_tdd는 2 m 기온 도일이다. 둘의 비(n-factor)가
식생·적설로 지역마다 달라 전이 편향의 한 원인이 된다. stl1 도일은 그 차이를 강제력 쪽에서 흡수한다.
계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §2.1 F7.

방법
----
`era5land_covariates.py`와 동일 규약: nh_monthly_2015-2020.nc → 12개월 기후값 → 최근접 격자.
ERA5-Land는 육지 전용이므로 최근접 격자가 NaN이면 `enrich_new_regions.py`와 같이 ±2셀(0.2°) →
±5셀(0.5°) 순으로 최근접 유효 격자로 폴백하고 사용 수를 기록한다.
산출 열: e5_tdd_soil, e5_fdd_soil, e5_sqrt_tdd_soil, e5_stl1_twarm, e5_stl1_tcold, soil_fallback_deg(폴백 거리 °, 0=직접).

입력: data/processed/fidelity_base.csv (loc_id·lat·lon), data/processed/alt_calm_new_sites.csv (lat·lon)
산출: data/processed/e5_soil_tdd.csv  (키: lat·lon 반올림 4자리 + loc_id(있으면))
실행(ROOT): python3 scripts/1_data_prep/era5land_soil_tdd.py
"""
from __future__ import annotations
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2015-2020.nc"
OUT = PROC / "e5_soil_tdd.csv"

ds = xr.open_dataset(NC)
tname = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tname].dt.month).groupby("month").mean(tname)
stl = clim["stl1"].values - 273.15                    # (12, lat, lon) °C
lat = clim["latitude"].values
lon = clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])
valid = np.isfinite(stl).all(axis=0)                  # 육지 마스크
print(f"[nc] stl1 기후값 {stl.shape} · 육지 격자 {valid.mean():.1%}")

# 대상 위치
fb = pd.read_csv(PROC / "fidelity_base.csv", usecols=["loc_id", "lat", "lon"])
fb["src"] = "fidelity_base"
new = pd.read_csv(PROC / "alt_calm_new_sites.csv", usecols=["lat", "lon"])
new["loc_id"] = -1
new["src"] = "calm_new"
locs = pd.concat([fb, new], ignore_index=True)
print(f"[locs] fidelity_base {len(fb):,} + calm_new {len(new)} = {len(locs):,}")


def nearest_valid(iy, ix, max_off):
    """(iy, ix)에서 체비쇼프 거리 max_off 이내 최근접 유효 격자. 없으면 None."""
    best, bestd = None, None
    for dy in range(-max_off, max_off + 1):
        for dx in range(-max_off, max_off + 1):
            y, x = iy + dy, (ix + dx) % len(lon)
            if 0 <= y < len(lat) and valid[y, x]:
                d = max(abs(dy), abs(dx))
                if bestd is None or d < bestd:
                    best, bestd = (y, x), d
    return best, bestd


recs = []
n_fb = {0: 0, 2: 0, 5: 0, None: 0}
for r in locs.itertuples(index=False):
    iy = int(np.abs(lat - r.lat).argmin())
    ix = int(np.abs(lon - r.lon).argmin())
    off = 0
    if not valid[iy, ix]:
        cell, d = nearest_valid(iy, ix, 2)
        if cell is None:
            cell, d = nearest_valid(iy, ix, 5)
            off = 5 if cell is not None else None
        else:
            off = 2
        if cell is None:
            n_fb[None] += 1
            recs.append(dict(loc_id=r.loc_id, lat=r.lat, lon=r.lon, src=r.src, soil_fallback_deg=np.nan))
            continue
        iy, ix = cell
        n_fb[off] += 1
        fallback = d * 0.1
    else:
        n_fb[0] += 1
        fallback = 0.0
    tm = stl[:, iy, ix]
    tdd = float(np.sum(np.clip(tm, 0, None) * days))
    fdd = float(np.sum(np.clip(-tm, 0, None) * days))
    recs.append(dict(loc_id=r.loc_id, lat=r.lat, lon=r.lon, src=r.src,
                     e5_tdd_soil=tdd, e5_fdd_soil=fdd, e5_sqrt_tdd_soil=float(np.sqrt(tdd)),
                     e5_stl1_twarm=float(tm.max()), e5_stl1_tcold=float(tm.min()),
                     soil_fallback_deg=fallback))

out = pd.DataFrame(recs)
out.to_csv(OUT, index=False)
print(f"[fallback] 직접 {n_fb[0]:,} · 0.2° {n_fb[2]} · 0.5° {n_fb[5]} · 실패 {n_fb[None]}")
print(out[["e5_tdd_soil", "e5_fdd_soil", "e5_sqrt_tdd_soil"]].describe().round(1).to_string())

# 기존 기온 도일과의 관계(참고): 알래스카 셀에서 sqrt 비 = 실효 n-factor의 제곱근
chk = out[out.src == "fidelity_base"].merge(
    pd.read_csv(PROC / "fidelity_base.csv", usecols=["loc_id", "region", "e5_sqrt_tdd"]), on="loc_id")
chk["ratio"] = chk.e5_sqrt_tdd_soil / chk.e5_sqrt_tdd
print("\n[참고] √TDD_soil / √TDD_air 지역별 중앙값:")
print(chk.groupby("region").ratio.median().round(3).to_string())

(PROC / "e5_soil_tdd_meta.json").write_text(json.dumps(dict(
    stage="E2-F7", nc=NC.name, clim_years="2015-2020", n_locs=int(len(out)),
    fallback_counts={str(k): v for k, v in n_fb.items()},
    columns=["e5_tdd_soil", "e5_fdd_soil", "e5_sqrt_tdd_soil", "e5_stl1_twarm", "e5_stl1_tcold", "soil_fallback_deg"],
    note="fidelity_base.csv는 수정하지 않음. E2 스크립트가 loc_id로 병합.",
), ensure_ascii=False, indent=1))
print(f"saved: {OUT}")
