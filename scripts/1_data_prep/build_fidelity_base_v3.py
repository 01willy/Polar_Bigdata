"""S-A 데이터 정비(2026-09-18): fidelity_base_v3 조립.

정본: docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md §3 S-A, docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md §4 1–3항,
      docs/RESULTS_RECONCILIATION_2026-09-14.md §6 "데이터·코드 후속 수정" 1–3항.

작업
----
1. Svalbard 매크로 분리: region == "CALM_Svalbard" 이면서 lat < 76 인 셀(노르웨이 본토 Juvvasshøe·Snøheim, 스웨덴 Abisko)을
   "CALM_Scandinavia"로 바꾼다. 진짜 스발바르(lat > 76) 4셀은 "CALM_Svalbard" 유지. GTNPenv_SJ(F2 포락선)는 대상 아님.
2. CALM 관측법 태그: PANGAEA_972777 헤더의 Event(s) 블록에서 "COMMENT: ... Method: <문자열>"을 파싱해
   지온(Ground temperature)·융해관(Thaw tube) 유도 사이트를 source_id = "F4_calm_temp"(fidelity_level 3)로 낮춘다.
   탐침(Spatially-oriented mechanical probing)은 F4_direct 유지. Method 정보가 없는 이벤트는 F4_direct 유지·메타에 플래그.
   셀↔이벤트 대응은 .tab 데이터 열(Site, Latitude, Longitude → Event)로 만든다(alt_global.csv의 site 열이 Site 코드).
3. 토양 도일 결측: e5_soil_tdd.csv에서 e5_tdd_soil <= 0 인 행의 e5_tdd_soil·e5_sqrt_tdd_soil을 NaN으로 바꾼
   e5_soil_tdd_v3.csv 저장(원본 불변).
4. v3 조립: v2 17,572행의 loc_id·좌표·공변량·라벨은 그대로 두고 region·source_id·fidelity_level 열만 갱신.
   불변 열 해시로 검증하고 fidelity_base_v3_meta.json에 변경 내역·지역별 셀·블록 수·관측법 근거·git commit 기록.

실행(ROOT): python3 scripts/1_data_prep/build_fidelity_base_v3.py
산출: data/processed/fidelity_base_v3.csv, fidelity_base_v3_meta.json, e5_soil_tdd_v3.csv
"""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (FIDELITY, MACRO_REGION, TRANSFER_MAIN, TRANSFER_DEEP,  # noqa: E402
                            add_group_keys, macro_region, loro_splits)

SVALBARD_LAT = 76.0          # 스발바르 군도 남단(Bjørnøya 74.4 제외) 기준. 본토 사이트는 전부 lat < 69.
UPDATED_COLS = ["region", "source_id", "fidelity_level"]


def hash_df(df: pd.DataFrame, cols=None) -> str:
    d = df if cols is None else df[cols]
    return hashlib.md5(pd.util.hash_pandas_object(d.reset_index(drop=True), index=False).values).hexdigest()[:12]


# ---------------------------------------------------------------- 입력
v2 = pd.read_csv(PROC / "fidelity_base_v2.csv", low_memory=False)
cell = pd.read_csv(PROC / "alt_calm_global_cell.csv")
assert cell.loc_id.isin(v2.loc_id).all() and len(cell) == 149, "alt_calm_global_cell.csv ↔ v2 loc_id 불일치"
chk = cell.merge(v2[["loc_id", "lat", "lon", "region"]], on="loc_id", suffixes=("", "_v2"))
assert np.allclose(chk.lat, chk.lat_v2) and np.allclose(chk.lon, chk.lon_v2) and (chk.region == chk.region_v2).all()
print(f"[in] v2 {v2.shape} · CALM 신규 셀 {len(cell)} · v2 해시 {hash_df(v2)}")

# ---------------------------------------------------------------- 2a. PANGAEA 이벤트 헤더 파싱
lines = TAB.read_text(encoding="utf-8", errors="replace").split("\n")
i0 = next(i for i, l in enumerate(lines) if l.startswith("Event(s):"))
i1 = next(i for i, l in enumerate(lines) if l.startswith("Parameter(s):"))
i_end = next(i for i, l in enumerate(lines) if l.strip() == "*/")
ev_rows = []
for l in lines[i0:i1]:
    l = l.replace("Event(s):\t", "").strip()
    parts = l.split(" * ")
    d = dict(event=parts[0].split(" ")[0])
    for p in parts[1:]:
        k, _, v = p.partition(":")
        if k in ("LATITUDE", "LONGITUDE"):
            d[k.lower()] = float(v)
        elif k in ("LOCATION", "COMMENT", "METHOD/DEVICE"):
            d[k.lower().replace("/", "_")] = v.strip()
    m = re.search(r"Method:\s*(.*)$", d.get("comment", ""))
    d["method_raw"] = m.group(1).strip() if m else None
    ev_rows.append(d)
ev = pd.DataFrame(ev_rows)
assert ev.event.is_unique and len(ev) == 263
print(f"[tab] Event 헤더 {len(ev)}건 · Method 문자열 있음 {ev.method_raw.notna().sum()} · 없음 {ev.method_raw.isna().sum()}")

# ---------------------------------------------------------------- 2b. 데이터 열로 셀↔Event 대응
tab = pd.read_csv(TAB, sep="\t", skiprows=i_end + 1, encoding="utf-8", encoding_errors="replace", low_memory=False)
key = tab.groupby(["Site", "Latitude", "Longitude"]).Event.agg(lambda s: "|".join(sorted(set(s)))).reset_index()
assert not key.Event.str.contains(r"\|").any(), "(Site, 좌표)에 Event 둘 이상"
key["klat"], key["klon"] = key.Latitude.round(4), key.Longitude.round(4)
cell["klat"], cell["klon"] = cell.lat.round(4), cell.lon.round(4)
cell = cell.merge(key[["Site", "klat", "klon", "Event"]], left_on=["site", "klat", "klon"],
                  right_on=["Site", "klat", "klon"], how="left")
assert cell.Event.notna().all(), f"Event 미대응 셀 {cell.Event.isna().sum()}"
cell = cell.merge(ev[["event", "method_raw", "location"]], left_on="Event", right_on="event", how="left")


def classify(s):
    """Method 문자열 → 관측법. 탐침만 F4_direct 유지."""
    if not isinstance(s, str):
        return "unknown"
    t = s.lower()
    if "mechanical probing" in t and "temperature" not in t:
        return "probe"
    if "thaw tube" in t:
        return "thaw_tube"
    if "temperature" in t:
        return "temperature"
    raise ValueError(f"분류 불가 Method 문자열: {s}")


cell["obs_method"] = cell.method_raw.map(classify)
print("[method] region × obs_method (v2 region 기준)")
print(pd.crosstab(cell.region, cell.obs_method).to_string())

# ---------------------------------------------------------------- 1. Svalbard 분리
sval = cell.region == "CALM_Svalbard"
to_scand = sval & (cell.lat < SVALBARD_LAT)
cell["region_v3"] = cell.region.where(~to_scand, "CALM_Scandinavia")
print("[svalbard] 분리 대상(→ CALM_Scandinavia):")
print(cell.loc[to_scand, ["loc_id", "site", "name", "country", "lat", "lon", "alt_cm", "obs_method"]].round(3).to_string(index=False))
print("[svalbard] 유지(CALM_Svalbard):")
print(cell.loc[sval & ~to_scand, ["loc_id", "site", "name", "country", "lat", "lon", "alt_cm", "obs_method"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- 2c. 태그
is_temp = cell.obs_method.isin(["temperature", "thaw_tube"])
cell["source_id_v3"] = np.where(is_temp, "F4_calm_temp", "F4_direct")
cell["fidelity_level_v3"] = cell.source_id_v3.map(FIDELITY)
assert FIDELITY["F4_calm_temp"] == 3
print(f"[tag] F4_calm_temp {int(is_temp.sum())}셀 · F4_direct 유지 {int((~is_temp).sum())}셀 · Method 미상 {int((cell.obs_method == 'unknown').sum())}셀")
print(pd.crosstab(cell.region_v3, cell.source_id_v3).to_string())

# ---------------------------------------------------------------- 4. v3 조립
v3 = v2.copy()
pos = v3.set_index("loc_id").index.get_indexer(cell.loc_id)
assert (pos >= 0).all()
v3.loc[pos, "region"] = cell.region_v3.values
v3.loc[pos, "source_id"] = cell.source_id_v3.values
v3.loc[pos, "fidelity_level"] = cell.fidelity_level_v3.values
fixed_cols = [c for c in v2.columns if c not in UPDATED_COLS]
assert list(v3.columns) == list(v2.columns)
assert hash_df(v2, fixed_cols) == hash_df(v3, fixed_cols), "불변 열이 바뀜"
n_region_changed = int((v2.region != v3.region).sum())
n_source_changed = int((v2.source_id != v3.source_id).sum())
assert n_region_changed == int(to_scand.sum()) and n_source_changed == int(is_temp.sum())
for r in v3.region.unique():
    assert r in MACRO_REGION or r in ("QTP_CN",) or True  # 매핑 확인은 아래 macro 단계에서

# 저장은 pandas 재직렬화(마지막 ULP 표기 차이 발생) 대신 v2 텍스트 행에서 세 열만 치환 → 나머지 필드 바이트 동일.
import csv  # noqa: E402
upd = {int(l): (r, s, str(int(f))) for l, r, s, f in
       zip(cell.loc_id, cell.region_v3, cell.source_id_v3, cell.fidelity_level_v3)}
with open(PROC / "fidelity_base_v2.csv", newline="", encoding="utf-8") as fi, \
        open(PROC / "fidelity_base_v3.csv", "w", newline="", encoding="utf-8") as fo:
    rd, wr = csv.reader(fi), csv.writer(fo, lineterminator="\n")
    header = next(rd); wr.writerow(header)
    ci = {c: header.index(c) for c in ["loc_id"] + UPDATED_COLS}
    n_txt = 0
    for row in rd:
        u = upd.get(int(row[ci["loc_id"]]))
        if u is not None:
            row[ci["region"]], row[ci["source_id"]], row[ci["fidelity_level"]] = u
            n_txt += 1
        wr.writerow(row)
assert n_txt == len(cell)
# 바이트 수준 검증: 갱신 3열 외 모든 필드가 v2와 동일
with open(PROC / "fidelity_base_v2.csv", newline="", encoding="utf-8") as fa, \
        open(PROC / "fidelity_base_v3.csv", newline="", encoding="utf-8") as fb:
    ra, rb = csv.reader(fa), csv.reader(fb)
    ha, hb = next(ra), next(rb)
    assert ha == hb
    keep = [k for k, c in enumerate(ha) if c not in UPDATED_COLS]
    for a, b in zip(ra, rb):
        assert [a[k] for k in keep] == [b[k] for k in keep], "불변 필드 텍스트 불일치"
v3r = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
assert len(v3r) == len(v2) and v3r.loc_id.is_unique
assert hash_df(v3r, fixed_cols) == hash_df(v2, fixed_cols), "저장 후 재적재 시 불변 열 해시 불일치"
fixed_hash = hash_df(v2, fixed_cols)
print(f"[v3] 저장 {v3r.shape} · region 변경 {n_region_changed}행 · source_id 변경 {n_source_changed}행 · 불변 열 해시 {fixed_hash}")

# ---------------------------------------------------------------- 매크로·전이 집합 검증
v3g = add_group_keys(v3r)
assert (v3g.block == v2.block).all(), "block 재계산 불일치"
v3g["macro"] = macro_region(v3g)
unmapped = sorted(set(v3g.region.unique()) - set(MACRO_REGION))
assert not unmapped, f"MACRO_REGION 미매핑 region {unmapped}"
new_macro = set(v3g.macro[v3g.region.str.startswith("CALM_")])
assert new_macro <= set(TRANSFER_MAIN + TRANSFER_DEEP), new_macro - set(TRANSFER_MAIN + TRANSFER_DEEP)
for r, tr, te in loro_splits(v3g, min_test=3):
    assert set(v3g.macro.values[tr]).isdisjoint(set(v3g.macro.values[te]))

lab = v3g[v3g.alt_cm.notna()]
tbl = (lab.groupby(["macro", "source_id"]).agg(n_cells=("loc_id", "size"), n_blocks=("block", "nunique"),
                                                alt_mean=("alt_cm", "mean")).round(1).reset_index())
print("[table] macro × source_id 셀·블록 수 (라벨 있는 셀)")
print(tbl.to_string(index=False))

v2g = add_group_keys(v2); v2g["macro"] = macro_region(v2g)
main_chk = []
for mreg in TRANSFER_MAIN:
    a = int(((v2g.macro == mreg) & (v2g.source_id == "F4_direct") & v2g.alt_cm.notna()).sum())
    b = int(((v3g.macro == mreg) & (v3g.source_id == "F4_direct") & v3g.alt_cm.notna()).sum())
    p = int(((v3g.macro == mreg) & (v3g.source_id == "F4_direct") & v3g.alt_cm.notna()
             & v3g.region.str.startswith("CALM_")).sum())
    main_chk.append(dict(macro=mreg, f4_direct_v2=a, f4_direct_v3=b, dropped_to_calm_temp=a - b, calm_probe_cells_v3=p))
main_chk = pd.DataFrame(main_chk)
print("[main-set] TRANSFER_MAIN F4_direct 셀 수 v2 → v3")
print(main_chk.to_string(index=False))

# ---------------------------------------------------------------- 3. 토양 도일 결측
soil = pd.read_csv(PROC / "e5_soil_tdd.csv")
zero = (soil.e5_tdd_soil <= 0).fillna(False)
soil3 = soil.copy()
soil3.loc[zero, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan
soil3.to_csv(PROC / "e5_soil_tdd_v3.csv", index=False)
z = soil.loc[zero, ["loc_id", "lat", "lon", "src", "e5_fdd_soil"]].copy()
z["klat"], z["klon"] = z.lat.round(4), z.lon.round(4)
v3k = v3g[["loc_id", "lat", "lon", "region", "macro", "source_id", "alt_cm"]].copy()
v3k["klat"], v3k["klon"] = v3k.lat.round(4), v3k.lon.round(4)
z = z.merge(v3k[["klat", "klon", "loc_id", "region", "macro", "source_id", "alt_cm"]].rename(columns={"loc_id": "loc_id_v3"}),
            on=["klat", "klon"], how="left")
names = cell[["klat", "klon", "site", "name"]]
z = z.merge(names, on=["klat", "klon"], how="left")
print(f"[soil] e5_tdd_soil <= 0 → NaN: {int(zero.sum())}행 (원본 e5_soil_tdd.csv 불변)")
print(z[["loc_id", "loc_id_v3", "lat", "lon", "src", "region", "macro", "source_id", "site", "name", "alt_cm"]].to_string(index=False))

# ---------------------------------------------------------------- 메타
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
except Exception:
    commit = None
evidence = cell[["loc_id", "site", "name", "country", "lat", "lon", "Event", "method_raw", "obs_method",
                 "region", "region_v3", "source_id_v3", "fidelity_level_v3", "alt_cm"]].copy()
evidence.columns = ["loc_id", "site", "name", "country", "lat", "lon", "pangaea_event", "method_raw", "obs_method",
                    "region_v2", "region_v3", "source_id_v3", "fidelity_level_v3", "alt_cm"]
meta = dict(
    stage="S-A", created="2026-09-18", git_commit=commit,
    plan=["docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md §3 S-A", "docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md §4 1-3",
          "docs/RESULTS_RECONCILIATION_2026-09-14.md §6 1-3"],
    inputs=dict(fidelity_base_v2="data/processed/fidelity_base_v2.csv", calm_cells="data/processed/alt_calm_global_cell.csv",
                pangaea_tab=str(TAB.relative_to(ROOT)), soil_tdd="data/processed/e5_soil_tdd.csv"),
    n_rows=int(len(v3r)), n_rows_v2=int(len(v2)),
    updated_cols=UPDATED_COLS, fixed_cols_hash_v2=fixed_hash, fixed_cols_hash_v3=hash_df(v3r, fixed_cols),
    fixed_cols_unchanged=True, v2_full_hash=hash_df(v2), v3_full_hash=hash_df(v3r),
    n_region_changed=n_region_changed, n_source_changed=n_source_changed,
    svalbard_split=dict(rule=f"region == 'CALM_Svalbard' and lat < {SVALBARD_LAT} → 'CALM_Scandinavia'",
                        moved=cell.loc[to_scand, ["loc_id", "site", "name", "country", "lat", "lon", "alt_cm", "obs_method"]].round(4).to_dict("records"),
                        kept_svalbard=cell.loc[sval & ~to_scand, ["loc_id", "site", "name", "lat", "lon", "alt_cm", "obs_method"]].round(4).to_dict("records"),
                        note="GTNPenv_SJ(F2_gtnp_env 10셀)는 대상 아님. Scandinavia는 TRANSFER_DEEP(셀 3·블록 3, 지역 수준 주장 불가)."),
    obs_method=dict(
        source="PANGAEA_972777 헤더 'Event(s):' 블록의 각 이벤트 'COMMENT: PI: ..., Method: <문자열>' 필드",
        cell_event_link="데이터 열 (Site, Latitude, Longitude) → Event. alt_calm_global_cell.csv의 site·lat·lon(4자리 반올림)과 대응",
        rule={"probe": "'mechanical probing' 포함(예: 'Spatially-oriented mechanical probing at 100 m Grid or Transect') → F4_direct 유지",
              "thaw_tube": "'Thaw tube' 포함 → F4_calm_temp", "temperature": "'temperature' 포함(예: 'Ground temperature measurements from a 10 m borehole', 'Transect/Ground temperature measurements') → F4_calm_temp",
              "unknown": "COMMENT에 Method 없음 → F4_direct 유지·플래그"},
        counts=cell.groupby(["region_v3", "obs_method"]).size().reset_index(name="n").to_dict("records"),
        n_f4_calm_temp=int(is_temp.sum()), n_probe=int((cell.obs_method == "probe").sum()),
        unknown_cells=cell.loc[cell.obs_method == "unknown", ["loc_id", "site", "name", "region_v3"]].to_dict("records"),
        document_expectation="RESULTS_RECONCILIATION §6-2: 알프스·몽골·QTP 전부 + 스발바르 5/7 + 캐나다 2/10 = 68셀",
        matches_document=bool(int(is_temp.sum()) == 68),
        per_cell=evidence.round(4).to_dict("records")),
    fidelity_levels={"F4_direct": FIDELITY["F4_direct"], "F4_calm_temp": FIDELITY["F4_calm_temp"]},
    macro_table=tbl.to_dict("records"),
    main_set_check=main_chk.to_dict("records"),
    soil_tdd=dict(output="data/processed/e5_soil_tdd_v3.csv", rule="e5_tdd_soil <= 0 → e5_tdd_soil, e5_sqrt_tdd_soil = NaN (e5_fdd_soil 등 유지)",
                  n_set_nan=int(zero.sum()),
                  rows=z[["loc_id", "loc_id_v3", "lat", "lon", "src", "region", "macro", "source_id", "site", "name", "alt_cm"]].round(4).to_dict("records"),
                  note="F7(토양 도일 Stefan) 앵커의 대기 도일 폴백은 하네스 쪽 수정 사항(S-A 4항, 본 스크립트 범위 밖)."),
    transfer_sets=dict(TRANSFER_MAIN=TRANSFER_MAIN, TRANSFER_DEEP=TRANSFER_DEEP),
)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


(PROC / "fidelity_base_v3_meta.json").write_text(json.dumps(_clean(meta), ensure_ascii=False, indent=1))
print("[done] fidelity_base_v3.csv · fidelity_base_v3_meta.json · e5_soil_tdd_v3.csv")
