#!/usr/bin/env python3
"""ext_labels 파서: ru_liang_mounds (Liang 등 2023, PANGAEA 961876).

자료
- Liang 등 (2023), Thaw depth of tree mounds from northeastern Siberia (S4). PANGAEA,
  https://doi.org/10.1594/PANGAEA.961876 (CC BY 4.0). 상위 묶음은 PANGAEA 926686 이다.
- 인디기르카 저지(초쿠르다흐 부근)의 지점 5개 표기(V, A, B, K tree mound, K tree wet), 좌표 4개,
  2010-2013년의 융해 깊이 기록 212건이다. 기록마다 지점 좌표, 관측일, 값(cm), 'Sample method' 가 있다.
- 같은 묶음의 PANGAEA 961879(Allaiha 격자)는 CALM R31 과 같은 지점이라 쓰지 않는다.

입력(data/raw/ru_liang_mounds/)
  PANGAEA_961876.tab                          원자료. 라벨의 출처다
    curl -L -o PANGAEA_961876.tab 'https://doi.pangaea.de/10.1594/PANGAEA.961876?format=textfile'
  ref/shingubara2019_bg-16-755.html           선택. 'Published data' 기록의 원 논문 본문(측정 방법 확인용)
출력
  data/raw/ru_liang_mounds/thaw_depth_by_visit.csv            방문 단위 요약(지점, 관측일별 기록 수와 통계)
  data/processed/ext_labels/ru_liang_mounds_points.csv        표준 점 자료(_schema.json 1.0)
  data/processed/ext_labels/ru_liang_mounds_meta.json         메타와 품질 요약

규칙(계획서 6B.3, _schema.json 1.0)
  1. 한 행은 (site_id, year) 이다. alt_cm 은 그 해 방문의 기록 평균이고 n_obs 는 기록 수다.
     한 해에 방문이 여럿이면 8월 1일 이후 방문의 평균 가운데 최댓값을 쓴다(이 자료에는 그런 해가 없다).
  2. 관측 월이 8월 또는 9월이면 label_def = direct_eos, eos_basis = record_date 다.
     7월 기록은 label_def = direct_dated, eos_basis = none 이다. 기록은 지우지 않는다.
  3. method: 자료 표기(Sample method)를 notes 에 적는다.
     'Leveling measurement', 'Soil sampling' 은 측정 기기를 확인하지 못해 other 로 둔다.
     'Published data'(Comment: Shingubara et al. (2019))는 원 논문 2.2절이 강철봉 삽입으로
     융해 깊이를 쟀다고 적으므로 probe 로 둔다. 원 논문 사본(ref/)에서 그 문장을 찾지 못하면 other 로 둔다.
     --published-as-other 를 주면 세 표기를 모두 other 로 둔다.
  4. K tree mound 와 K tree wet 은 좌표가 같고 미지형이 다르다. site_id 를 따로 준다.
  5. 값의 단위는 cm 다(열 이름 'Thaw depth [cm]'). 변환하지 않는다. 범위 QC 는 0 < alt_cm <= 600 이다.

실행(저장소 최상위에서, 스레드 1개). 입력은 표 한 개(24 KB)다
  OMP_NUM_THREADS=1 python3 scripts/1_data_prep/parse_ext_ru_liang_mounds.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
from collections import Counter, OrderedDict
from datetime import date
from pathlib import Path

SRC_ID = "ru_liang_mounds"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
SCHEMA = OUT_DIR / "_schema.json"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
FIDELITY_PY = ROOT / "src" / "polar" / "fidelity.py"
CALM_TAB = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"

TAB = "PANGAEA_961876.tab"
VISIT_CSV = "thaw_depth_by_visit.csv"
REF_HTML = "ref/shingubara2019_bg-16-755.html"
AUX_FILES = OrderedDict([
    ("PANGAEA_961876_metadata.jsonld", "https://doi.pangaea.de/10.1594/PANGAEA.961876?format=metadata_jsonld"),
    ("PANGAEA_926686_metadata.jsonld", "https://doi.pangaea.de/10.1594/PANGAEA.926686?format=metadata_jsonld"),
    (REF_HTML, "https://bg.copernicus.org/articles/16/755/2019/"),
    ("ref/shingubara2019_bg-16-755_supplement.pdf",
     "https://bg.copernicus.org/articles/16/755/2019/bg-16-755-2019-supplement.pdf"),
    ("ref/shingubara2019_bg-16-755_table1.xlsx",
     "https://bg.copernicus.org/articles/16/755/2019/bg-16-755-2019-t01.xlsx"),
])

URL = "https://doi.org/10.1594/PANGAEA.961876"
URL_FILE = "https://doi.pangaea.de/10.1594/PANGAEA.961876?format=textfile"
DOI = "10.1594/PANGAEA.961876"
ACCESSED = "2026-09-29"
LICENSE = "CC-BY-4.0"
COUNTRY = "Russia"
SUBUNIT = "Indigirka"
COORD_PREC = "0.00001"
ORIG_SHINGUBARA = ("Shingubara, R. et al. (2019): Multi-year effect of wetting on CH4 flux at taiga-tundra boundary "
                   "in northeastern Siberia deduced from stable isotope ratios of CH4. Biogeosciences 16, 755-768, "
                   "https://doi.org/10.5194/bg-16-755-2019")
RELATED_PAPER = ("Liang 등 2023, Journal of Geophysical Research: Biogeosciences 128(10), e2022JG007135, "
                 "https://doi.org/10.1029/2022JG007135")

NEED_COLS = ["Location", "Latitude", "Longitude", "Year obs [a AD]", "Date/Time", "Date/time end", "Site",
             "Thaw depth [cm]", "Sample method", "Tree ID", "Comment"]
EOS_MONTHS = (8, 9)
L41B_DAY = (8, 15)                 # 계획서 6B.5 L41 변형 (b): 8월 15일 이전의 단일 방문
RANGE_CM = (0.0, 600.0)
BBOX = {"lat": (69.5, 71.5), "lon": (146.0, 150.0)}     # 초쿠르다흐 부근(좌표 부호와 자릿수 확인용)
MISSING_MARKS = {"", "-999", "-9999", "-999.0", "-9999.0", "nan", "NaN", "n.d.", "NA", "-"}
STEEL_ROD_PHRASE = "inserting a steel rod"
MAX_THAW_PHRASE = "maximum thaw depth occurs from the latter half of August to the first half of September"

# 지점 표기와 이름. V, K, B 의 이름과 분 단위 좌표는 Shingubara 등 2019 의 표 1 에서 확인했다.
# A 는 그 표에 없고 이름을 확인하지 못했다.
SITES = OrderedDict([
    ("V", {"site_id": "V", "name": "Verkhny Khatistakha (V)", "micro": "",
           "table1_min": ((70, 15), (147, 28))}),
    ("A", {"site_id": "A", "name": "A", "micro": "", "table1_min": None}),
    ("K tree mound", {"site_id": "K_tree_mound", "name": "Kodac (K), tree mound", "micro": "tree_mound",
                      "table1_min": ((70, 34), (148, 16))}),
    ("K tree wet", {"site_id": "K_tree_wet", "name": "Kodac (K), wet area", "micro": "wet_area",
                    "table1_min": ((70, 34), (148, 16))}),
    ("B", {"site_id": "B", "name": "Boydom (B)", "micro": "", "table1_min": ((70, 38), (148, 9))}),
])

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
            "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
            "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_of(lat: float, lon: float) -> str:
    """계획서 6B.4 의 지리 정의(러시아 안에서만 쓴다). 라벨 값을 쓰지 않는다."""
    if 71.5 <= lat <= 73.6 and 123.3 <= lon <= 130.1:
        return "Lena"
    if lon < 0 or lon >= 140.0:
        return "Russia_E"
    if lon < 90.0:
        return "Russia_W"
    return "Russia_C"


def read_pangaea_tab(path: Path, need_cols):
    """PANGAEA 탭 파일을 머리말(/* ... */)과 표로 나눈다. 값은 문자열 그대로 둔다."""
    text = path.read_text(encoding="utf-8")
    end = text.index("*/")
    header = text[: end + 2]
    body = text[end + 2:].lstrip("\r\n")
    rows = list(csv.reader(body.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE))
    cols, data = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    missing = [c for c in need_cols if c not in cols]
    if missing:
        raise SystemExit(f"{path.name}: 열이 없다 {missing}")
    recs = [{k: v.strip() for k, v in zip(cols, r + [""] * (len(cols) - len(r)))} for r in data]
    return header, cols, recs


def header_field(header: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\t(.+)$", header, re.M)
    return m.group(1).strip() if m else ""


def plain_text(path: Path):
    """HTML 본문을 꼬리표 없는 글로 바꾼다. 파일이 없으면 None."""
    if not path.exists():
        return None
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def cell_index(lat: float, lon: float):
    """계획서 6B.3 의 약 1 km 셀 색인."""
    ky = math.floor(lat / 0.009)
    phi = math.radians((ky + 0.5) * 0.009)
    kx = math.floor(lon * math.cos(phi) / 0.009)
    return ky, kx


def block_index(lat: float, lon: float) -> int:
    return math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5)


def haversine_km(la1, lo1, la2, lo2) -> float:
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def stats_of(vals):
    vals = [float(v) for v in vals]
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
            "min": min(vals), "median": round(statistics.median(vals), 4), "max": max(vals)}


def v3_macro_map():
    """src/polar/fidelity.py 의 MACRO_REGION 을 글로 읽는다(모듈을 불러오지 않는다)."""
    if not FIDELITY_PY.exists():
        return None
    tree = ast.parse(FIDELITY_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MACRO_REGION" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def v3_check(sites, macro_here: str):
    """v3 의 F4_direct 셀과의 거리, 체비쇼프 0.01° 중복, 다른 macro 와의 최소 거리. v3 는 읽기만 한다."""
    if not V3_PATH.exists():
        return None
    mmap = v3_macro_map() or {}
    ref = []
    with open(V3_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] == "F4_direct":
                ref.append((float(r["lat"]), float(r["lon"]), r["region"], r["loc_id"], float(r["alt_cm"])))
    out = {"n_v3_f4_direct_cells": len(ref), "macro_map_source": "src/polar/fidelity.py MACRO_REGION(글로 읽음)",
           "by_site": {}}
    for sid, (la, lo) in sites.items():
        best = best_other = None
        n_dup = 0
        for (rla, rlo, reg, lid, alt) in ref:
            if abs(rla - la) > 3.0:
                continue
            d = haversine_km(la, lo, rla, rlo)
            if max(abs(la - rla), abs(lo - rlo)) <= 0.01:
                n_dup += 1
            if best is None or d < best[0]:
                best = (d, reg, lid, rla, rlo, alt)
            if mmap.get(reg, reg) != macro_here and (best_other is None or d < best_other[0]):
                best_other = (d, reg, lid)
        out["by_site"][sid] = {
            "nearest_loc_id": int(best[2]), "nearest_region": best[1], "nearest_lat": best[3],
            "nearest_lon": best[4], "nearest_alt_cm": best[5], "dist_km": round(best[0], 2),
            "chebyshev_deg": round(max(abs(la - best[3]), abs(lo - best[4])), 5),
            "n_v3_f4_direct_within_cheb_0p01deg": n_dup,
            "nearest_other_macro_within_3deg_lat": (
                {"dist_km": round(best_other[0], 1), "region": best_other[1], "loc_id": int(best_other[2])}
                if best_other else "위도 3° 안에 다른 macro 의 F4_direct 셀이 없다(100 km 보다 멀다)"),
        }
    return out


def calm_r31_values(years):
    """CALM R31(Allaiha)의 연 값을 PANGAEA 972777 에서 읽는다. 라벨이 아니라 참고 값이다."""
    if not CALM_TAB.exists():
        return None
    text = CALM_TAB.read_text(encoding="utf-8")
    body = text[text.index("*/") + 2:].lstrip("\r\n").splitlines()
    cols = body[0].split("\t")
    i_ev, i_la, i_lo, i_yr, i_v = (cols.index(c) for c in ("Event", "Latitude", "Longitude", "Date/Time",
                                                           "ALD [cm]"))
    out = {}
    for line in body[1:]:
        r = line.split("\t")
        if r[i_ev] == "CALM_R31" and r[i_yr][:4].isdigit() and int(r[i_yr][:4]) in years and r[i_v].strip():
            out[int(r[i_yr][:4])] = {"alt_cm": float(r[i_v]), "lat": float(r[i_la]), "lon": float(r[i_lo])}
    return out


def cross_source_check(sites):
    """같은 폴더의 다른 점 자료와 체비쇼프 0.01° 이내인 지점을 센다. 실행 시점에 있는 파일만 본다."""
    files = sorted(p for p in glob.glob(str(OUT_DIR / "*_points.csv")) if not p.endswith(f"{SRC_ID}_points.csv"))
    hits, nearest = [], {}
    for p in files:
        with open(p, newline="", encoding="utf-8") as f:
            seen = set()
            for r in csv.DictReader(f):
                try:
                    la, lo = float(r["lat"]), float(r["lon"])
                except (KeyError, ValueError):
                    continue
                key = (r.get("site_id", ""), la, lo)
                if key in seen:
                    continue
                seen.add(key)
                for sid, (sla, slo) in sites.items():
                    if abs(la - sla) > 3.0:
                        continue
                    d = haversine_km(sla, slo, la, lo)
                    if sid not in nearest or d < nearest[sid]["dist_km"]:
                        nearest[sid] = {"dist_km": round(d, 2), "src_id": r.get("src_id", ""),
                                        "site_id": r.get("site_id", "")}
                    if max(abs(la - sla), abs(lo - slo)) <= 0.01:
                        hits.append({"site_id": sid, "other": f"{r.get('src_id', '')}:{r.get('site_id', '')}",
                                     "dist_km": round(d, 3)})
    return {"files_compared": [Path(p).name for p in files], "n_within_cheb_0p01deg": len(hits),
            "within_cheb_0p01deg": hits, "nearest_within_3deg_lat": nearest}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--published-as-other", action="store_true",
                    help="'Published data' 기록의 method 도 other 로 둔다")
    args = ap.parse_args()

    tab = RAW_DIR / TAB
    if not tab.exists():
        raise SystemExit(f"원자료가 없다: {tab}")
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    assert [c["name"] for c in schema["columns"]] == COLUMNS, "열 구성이 형식 정의와 다르다"

    header, cols, recs = read_pangaea_tab(tab, NEED_COLS)
    n_raw = len(recs)
    citation = header_field(header, "Citation").rstrip(", ")
    license_raw = header_field(header, "License")
    if "CC-BY-4.0" not in license_raw:
        raise SystemExit(f"약관이 예상과 다르다: {license_raw}")
    if DOI not in citation:
        raise SystemExit(f"인용문에 DOI 가 없다: {citation}")

    # ---------- 원문 근거(측정 방법, 최대 융해 시기) ----------
    ref_text = plain_text(RAW_DIR / REF_HTML)
    steel_rod_found = bool(ref_text) and STEEL_ROD_PHRASE in ref_text
    max_thaw_found = bool(ref_text) and MAX_THAW_PHRASE in ref_text
    published_is_probe = steel_rod_found and not args.published_as_other

    def method_of(sample_method: str, comment: str):
        """(method, notes 에 적을 문장)"""
        if sample_method == "Published data":
            src = f"(Comment: {comment})" if comment else ""
            if published_is_probe:
                return "probe", (f"자료 표기 Sample method = Published data{src}. 원 논문(Shingubara 등 2019) "
                                 "2.2절은 강철봉을 땅에 꽂아 융해 깊이를 쟀다고 적는다")
            return "other", (f"자료 표기 Sample method = Published data{src}. "
                             "측정 기기는 이 실행에서 확인하지 않았다")
        return "other", f"자료 표기 Sample method = {sample_method}. 측정 기기는 확인하지 못했다"

    # ---------- 기록 단위 점검 ----------
    chk = Counter()
    top_values = Counter()
    clean = []
    for i, r in enumerate(recs):
        raw_v = r["Thaw depth [cm]"]
        if raw_v in MISSING_MARKS:
            chk["missing_thaw_values"] += 1
            continue
        if re.search(r"[<>]", raw_v):
            chk["greater_than_notation"] += 1
        try:
            v = float(raw_v.lstrip("<>"))
        except ValueError:
            chk["nonnumeric_thaw_values"] += 1
            continue
        if v < 0:
            chk["negative_thaw_values"] += 1
        if not (RANGE_CM[0] < abs(v) <= RANGE_CM[1]):
            chk["thaw_outside_range"] += 1
        if abs(v * 2 - round(v * 2)) > 1e-9:
            chk["not_multiple_of_0p5cm"] += 1
        lat, lon = float(r["Latitude"]), float(r["Longitude"])
        if not (BBOX["lat"][0] <= lat <= BBOX["lat"][1] and BBOX["lon"][0] <= lon <= BBOX["lon"][1]):
            chk["coord_outside_bbox"] += 1
        d0 = date.fromisoformat(r["Date/Time"][:10])
        d1 = date.fromisoformat(r["Date/time end"][:10]) if r["Date/time end"] else None
        if str(d0.year) != r["Year obs [a AD]"]:
            chk["year_column_differs_from_date"] += 1
        if d1 is not None:
            chk["records_with_date_range"] += 1
            if d1 < d0:
                chk["date_end_before_start"] += 1
            if (d1.year, d1.month) != (d0.year, d0.month):
                chk["date_range_crosses_month"] += 1
        if r["Site"] not in SITES:
            chk["site_label_not_in_table"] += 1
        top_values[v] += 1
        clean.append({"row": i, "site": r["Site"], "lat_str": r["Latitude"], "lon_str": r["Longitude"],
                      "lat": lat, "lon": lon, "d0": d0, "d1": d1, "v": abs(v), "gt": ">" in raw_v,
                      "method_raw": r["Sample method"], "tree_id": r["Tree ID"], "comment": r["Comment"]})
    full_dup = Counter(tuple(r[c] for c in cols) for r in recs)
    chk["identical_records_beyond_first"] = sum(n - 1 for n in full_dup.values() if n > 1)
    with_tree = Counter((r["Site"], r["Date/Time"], r["Tree ID"]) for r in recs if r["Tree ID"])
    chk["duplicate_tree_id_within_visit"] = sum(n - 1 for n in with_tree.values() if n > 1)
    for k in ("missing_thaw_values", "greater_than_notation", "nonnumeric_thaw_values", "negative_thaw_values",
              "thaw_outside_range", "not_multiple_of_0p5cm", "coord_outside_bbox",
              "year_column_differs_from_date", "records_with_date_range", "date_end_before_start",
              "date_range_crosses_month", "site_label_not_in_table"):
        chk.setdefault(k, 0)

    # 지점 표기별 좌표가 하나인지, 표 1 의 분 단위 좌표와 맞는지
    site_coords = OrderedDict()
    for c in clean:
        site_coords.setdefault(c["site"], set()).add((c["lat_str"], c["lon_str"]))
    multi_coord = {s: sorted(v) for s, v in site_coords.items() if len(v) > 1}
    if multi_coord:
        raise SystemExit(f"지점 표기 하나에 좌표가 여럿이다: {multi_coord}")
    table1 = {}
    for s, v in site_coords.items():
        la, lo = map(float, next(iter(v)))
        t1 = SITES.get(s, {}).get("table1_min")
        if t1 is None:
            table1[s] = "표 1 에 없는 지점이다"
            continue
        dla = abs(la - (t1[0][0] + t1[0][1] / 60.0))
        dlo = abs(lo - (t1[1][0] + t1[1][1] / 60.0))
        table1[s] = {"table1": f"{t1[0][0]}°{t1[0][1]:02d}′N {t1[1][0]}°{t1[1][1]:02d}′E",
                     "abs_diff_deg": [round(dla, 5), round(dlo, 5)],
                     "within_half_minute": bool(dla <= 0.5 / 60 + 1e-9 and dlo <= 0.5 / 60 + 1e-9)}

    # ---------- 방문 단위 요약 ----------
    visits = OrderedDict()
    for c in clean:
        key = (c["site"], c["d0"], c["d1"])
        visits.setdefault(key, []).append(c)
    visit_rows = []
    for (site, d0, d1), rs in sorted(visits.items(), key=lambda kv: (list(SITES).index(kv[0][0])
                                                                      if kv[0][0] in SITES else 99, kv[0][1])):
        methods = sorted({r["method_raw"] for r in rs})
        st = stats_of([r["v"] for r in rs])
        visit_rows.append({
            "site": site, "site_id": SITES.get(site, {}).get("site_id", re.sub(r"\W+", "_", site)),
            "lat": rs[0]["lat_str"], "lon": rs[0]["lon_str"], "date": d0.isoformat(),
            "date_end": d1.isoformat() if d1 else "", "year": d0.year, "month": d0.month,
            "sample_method": ";".join(methods), "comment": ";".join(sorted({r["comment"] for r in rs if r["comment"]})),
            "tree_ids": ";".join(r["tree_id"] for r in rs if r["tree_id"]),
            "n": st["n"], "mean_cm": st["mean"], "sd_cm": "" if st["sd"] is None else st["sd"],
            "min_cm": st["min"], "median_cm": st["median"], "max_cm": st["max"],
            "n_gt_notation": sum(r["gt"] for r in rs)})
    with open(RAW_DIR / VISIT_CSV, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(visit_rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        wr.writerows(visit_rows)
    n_mixed_method = sum(";" in v["sample_method"] for v in visit_rows)

    # ---------- (site_id, year) 연 값 ----------
    by_sy = OrderedDict()
    for v in visit_rows:
        by_sy.setdefault((v["site"], v["year"]), []).append(v)
    n_multi_visit_years = sum(len(v) > 1 for v in by_sy.values())
    rows = []
    for (site, year), vs in by_sy.items():
        info = SITES.get(site, {"site_id": re.sub(r"\W+", "_", site), "name": site, "micro": ""})
        elig = [v for v in vs if v["month"] in EOS_MONTHS
                and (not v["date_end"] or int(v["date_end"][5:7]) in EOS_MONTHS)]
        if elig:
            best = max(elig, key=lambda v: (v["mean_cm"], v["date"]))       # 같으면 늦은 방문
            label_def = "direct_eos"
            eos = "record_date" if len(vs) == 1 else "series_max"
        else:
            best = max(vs, key=lambda v: v["date"])                          # 가장 늦은 방문
            label_def, eos = "direct_dated", "none"
        raw_methods = best["sample_method"].split(";")
        method, mnote = method_of(raw_methods[0], best["comment"])
        if len(raw_methods) > 1:
            method, mnote = "other", f"자료 표기 Sample method 가 한 방문에 여럿이다({best['sample_method']})"
        lat, lon = float(best["lat"]), float(best["lon"])
        d0 = date.fromisoformat(best["date"])
        flags = []
        if not (RANGE_CM[0] < best["mean_cm"] <= RANGE_CM[1]):
            flags.append("range")
        if not (BBOX["lat"][0] <= lat <= BBOX["lat"][1] and BBOX["lon"][0] <= lon <= BBOX["lon"][1]):
            flags.append("coord")
        if year < 1990:
            flags.append("year_pre1990")
        notes = [mnote]
        if best["date_end"]:
            notes.append(f"관측 기간 {best['date']} 에서 {best['date_end']}. date 와 month 는 시작일 기준이다")
        if label_def == "direct_eos":
            if (d0.month, d0.day) < L41B_DAY:
                notes.append(f"관측일 {best['date']} 은 8월 15일 이전이다(계획서 L41 변형 (b)의 제외 대상)")
            if max_thaw_found:
                notes.append("Shingubara 등 2019 의 2.1절은 이 지역의 최대 융해 깊이 시기를 8월 하순에서 "
                             "9월 전반으로 적는다. 값이 연 최대보다 작을 수 있다")
        else:
            notes.append(f"관측 월이 {d0.month}월이라 계절 말 조건(8-9월)을 만족하지 않는다")
        if info["micro"] == "tree_mound":
            notes.append("낙엽송 둔덕(tree mound) 미지형의 값이다. 같은 좌표의 습지 값은 site_id K_tree_wet 에 있다")
        elif info["micro"] == "wet_area":
            notes.append("둔덕 사이 습지(wet area) 미지형의 값이다. 같은 좌표의 둔덕 값은 site_id K_tree_mound 에 있다. "
                         "수체 바닥 교란으로 표시하지 않았다")
        else:
            notes.append("자료에 미지형(둔덕, 습지) 구분이 없다")
        if best["tree_ids"]:
            notes.append(f"Tree ID {best['tree_ids'].replace(';', ', ')}")
        if len(vs) > 1:
            notes.append("그 해의 방문별 평균: " + ", ".join(f"{v['date']} {v['mean_cm']:.2f} cm" for v in vs))
        rows.append({
            "src_id": SRC_ID, "site_id": info["site_id"], "site_name": info["name"],
            "lat": best["lat"], "lon": best["lon"], "year": year, "month": d0.month,
            "alt_cm": best["mean_cm"], "method": method, "label_def": label_def, "n_obs": best["n"],
            "country": COUNTRY, "macro": macro_of(lat, lon), "citation": citation, "license": LICENSE,
            "subunit": SUBUNIT, "date": best["date"], "eos_basis": eos,
            "value_kind": "single_visit" if len(vs) == 1 else "annual_value",
            "year_min": "", "year_max": "", "alt_sd_cm": best["sd_cm"], "coord_prec_deg": COORD_PREC,
            "disturbed": 0, "disturb_type": "", "right_censored": 1 if best["n_gt_notation"] else 0,
            "orig_source": ORIG_SHINGUBARA if "Published data" in raw_methods else "",
            "dup_of": "", "qc_flag": ";".join(flags) if flags else "ok", "notes": ". ".join(notes)})
    order = {info["site_id"]: i for i, info in enumerate(SITES.values())}
    rows.sort(key=lambda r: (order.get(r["site_id"], 99), r["year"]))
    assert all(list(r.keys()) == COLUMNS for r in rows), "열 순서가 형식 정의와 다르다"
    assert len({(r["site_id"], r["year"]) for r in rows}) == len(rows), "(site_id, year) 가 겹친다"
    if n_multi_visit_years == 0:
        assert sum(r["n_obs"] for r in rows) == len(clean), "행의 기록 수 합이 원자료 기록 수와 다르다"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = OUT_DIR / f"{SRC_ID}_points.csv"
    with open(points, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)

    # ---------- 대상 행과 셀 미리보기 ----------
    def is_target(r):
        return (r["label_def"] == "direct_eos" and r["qc_flag"] == "ok" and r["disturbed"] == 0
                and r["right_censored"] == 0 and r["year"] >= 1990)

    target = [r for r in rows if is_target(r)]

    def cell_table(rs, by):
        """by = 'site_id' 또는 'coord'. 위치별 다년 평균을 낸 뒤 셀 안에서 평균한다."""
        loc = OrderedDict()
        for r in rs:
            key = r["site_id"] if by == "site_id" else (r["lat"], r["lon"])
            loc.setdefault(key, []).append(r)
        cells = OrderedDict()
        for key, rr in loc.items():
            la, lo = float(rr[0]["lat"]), float(rr[0]["lon"])
            cells.setdefault(cell_index(la, lo), []).append(
                {"loc": key if isinstance(key, str) else "/".join(sorted({r["site_id"] for r in rr})),
                 "lat": la, "lon": lo, "mean": statistics.fmean(r["alt_cm"] for r in rr),
                 "years": sorted({r["year"] for r in rr}), "n_obs": sum(r["n_obs"] for r in rr)})
        out = []
        for (ky, kx), ls in cells.items():
            la = statistics.fmean(x["lat"] for x in ls)
            lo = statistics.fmean(x["lon"] for x in ls)
            out.append({"cell": [ky, kx], "block": block_index(la, lo), "lat": round(la, 5), "lon": round(lo, 5),
                        "alt_cm": round(statistics.fmean(x["mean"] for x in ls), 4),
                        "locations": [x["loc"] for x in ls],
                        "years": sorted({y for x in ls for y in x["years"]}),
                        "n_obs": sum(x["n_obs"] for x in ls)})
        return out

    def preview(rs):
        a, b = cell_table(rs, "site_id"), cell_table(rs, "coord")
        return {"n_rows": len(rs), "n_cells_1km": len(a), "n_blocks_0p5deg": len({c["block"] for c in a}),
                "blocks": sorted({c["block"] for c in a}),
                "cells_location_by_site_id": a, "cells_location_by_coordinate": b}

    early = [r for r in target if (int(r["date"][5:7]), int(r["date"][8:10])) < L41B_DAY]
    probe_only = [r for r in target if r["method"] in ("probe", "thaw_tube", "frost_tube")]
    cell_prev = {
        "_설명": "v4 조립 전의 참고 값이다. 셀 색인은 계획서 6B.3 의 식, 블록은 0.5° 다. K_tree_mound 와 "
                "K_tree_wet 은 좌표가 같다. 위치를 site_id 로 묶은 값과 좌표로 묶은 값을 함께 적는다",
        "target_rows": preview(target),
        "all_rows": {k: v for k, v in preview(rows).items() if k in ("n_rows", "n_cells_1km", "n_blocks_0p5deg",
                                                                     "blocks")},
        "target_rows_L41a_probe_tube_only": {k: v for k, v in preview(probe_only).items()
                                             if k in ("n_rows", "n_cells_1km", "n_blocks_0p5deg", "blocks")}
        if probe_only else {"n_rows": 0, "n_cells_1km": 0, "n_blocks_0p5deg": 0, "blocks": []},
        "target_rows_L41b_after_excluding_single_visit_before_aug15": {
            "n_rows": len(target) - len(early),
            "n_cells_1km": len(cell_table([r for r in target if r not in early], "site_id")),
        },
    }

    k_rows = [r for r in target if r["site_id"] in ("K_tree_mound", "K_tree_wet")]
    if k_rows:
        km = [r for r in k_rows if r["site_id"] == "K_tree_mound"]
        kw = [r for r in k_rows if r["site_id"] == "K_tree_wet"]
        n_all = sum(r["n_obs"] for r in k_rows)
        cell_prev["k_cell_alternatives"] = {
            "_설명": "K 지점 셀의 값은 둔덕과 습지를 어떻게 묶는지에 따라 달라진다. 대상 행의 연 값으로 계산했다",
            "tree_mound_only_cm": round(statistics.fmean(r["alt_cm"] for r in km), 4) if km else None,
            "wet_area_only_cm": round(statistics.fmean(r["alt_cm"] for r in kw), 4) if kw else None,
            "equal_weight_of_two_site_ids_cm": round(statistics.fmean(
                statistics.fmean(r["alt_cm"] for r in g) for g in (km, kw) if g), 4),
            "record_weighted_cm": round(sum(r["alt_cm"] * r["n_obs"] for r in k_rows) / n_all, 4),
            "n_records": {"tree_mound": sum(r["n_obs"] for r in km), "wet_area": sum(r["n_obs"] for r in kw)},
        }

    site_xy = OrderedDict((SITES.get(s, {}).get("site_id", s), tuple(map(float, next(iter(v)))))
                          for s, v in site_coords.items())
    v3c = v3_check(site_xy, "Russia_E")
    calm_r31 = calm_r31_values(sorted({r["year"] for r in rows}))
    xsrc = cross_source_check(site_xy)

    v_2012_max = max((v["max_cm"] for v in visit_rows if v["site"] == "V" and v["date"] == "2012-08-07"),
                     default=float("nan"))

    # ---------- 메타 ----------
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = ""
    files = [{"name": TAB, "path": str((RAW_DIR / TAB).relative_to(ROOT)), "url": URL_FILE,
              "size_bytes": tab.stat().st_size, "sha256": sha256_of(tab), "kind": "원자료(내려받은 파일)"},
             {"name": VISIT_CSV, "path": str((RAW_DIR / VISIT_CSV).relative_to(ROOT)), "url": "",
              "size_bytes": (RAW_DIR / VISIT_CSV).stat().st_size, "sha256": sha256_of(RAW_DIR / VISIT_CSV),
              "kind": "파서가 만든 방문 단위 요약"}]
    for name, u in AUX_FILES.items():
        p = RAW_DIR / name
        if p.exists():
            files.append({"name": name, "path": str(p.relative_to(ROOT)), "url": u,
                          "size_bytes": p.stat().st_size, "sha256": sha256_of(p),
                          "kind": "참고 자료(라벨 값의 출처가 아니다)"})

    n_by = lambda k, rs=rows: dict(Counter(r[k] for r in rs))          # noqa: E731
    raw_by_method = dict(Counter(c["method_raw"] for c in clean))
    raw_by_month = dict(Counter(f"{c['d0'].month:02d}" for c in clean))
    excl = {
        "_설명": "점 자료에서 지운 행은 없다. 아래는 대상 라벨 규칙(계획서 6B.3)을 통과하지 못하는 행의 수다",
        "dropped_from_points": 0,
        "raw_records_not_parsed": n_raw - len(clean),
        "label_def_not_direct_eos_july_visit": sum(r["label_def"] != "direct_eos" for r in rows),
        "right_censored": sum(r["right_censored"] for r in rows),
        "disturbed": sum(r["disturbed"] for r in rows),
        "qc_flag_not_ok": sum(r["qc_flag"] != "ok" for r in rows),
        "year_pre1990": sum(r["year"] < 1990 for r in rows),
        "raw_records_in_non_target_rows": sum(r["n_obs"] for r in rows if not is_target(r)),
        "raw_records_in_target_rows": sum(r["n_obs"] for r in target),
    }
    deviation = []
    if published_is_probe:
        deviation.append(
            "작업 지시의 라벨 규칙은 method 를 other 로 두라고 했다. 이 파서는 'Published data' 기록"
            f"({raw_by_method.get('Published data', 0)}건, 점 자료 "
            f"{sum(r['method'] == 'probe' for r in rows)}행)만 probe 로 두었다. 근거는 원 논문 2.2절의 문장 "
            f"'{STEEL_ROD_PHRASE}' 이다. 모두 other 로 두려면 --published-as-other 로 다시 실행한다")
    meta = OrderedDict([
        ("src_id", SRC_ID),
        ("name", "Liang 등 2023, 시베리아 북동부 낙엽송 둔덕 융해 깊이(PANGAEA 961876)"),
        ("schema_version", schema["version"]),
        ("url", URL), ("url_file", URL_FILE), ("doi", DOI), ("accessed", ACCESSED),
        ("parent_bundle", "https://doi.org/10.1594/PANGAEA.926686"),
        ("files", files),
        ("license", LICENSE), ("license_statement", license_raw),
        ("license_evidence", "원자료 파일 머리말의 License 항목과 PANGAEA 메타(JSON-LD)의 license 항목"),
        ("citation", citation),
        ("related_paper", RELATED_PAPER),
        ("orig_source_published_data", ORIG_SHINGUBARA),
        ("parser", str(Path(__file__).resolve().relative_to(ROOT))),
        ("parser_sha256", sha256_of(Path(__file__).resolve())),
        ("parser_git_commit", commit),
        ("parser_git_note", "파서는 이 커밋의 작업 트리에서 새로 만든 파일이고 아직 커밋하지 않았다"),
        ("parser_options", {"published_as_other": bool(args.published_as_other)}),
        ("n_rows_raw", n_raw),
        ("n_rows_points", len(rows)),
        ("n_sites", len({r["site_id"] for r in rows})),
        ("n_coordinates", len({(r["lat"], r["lon"]) for r in rows})),
        ("n_visits", len(visit_rows)),
        ("n_by_label_def", n_by("label_def")),
        ("n_by_method", n_by("method")),
        ("n_by_macro", n_by("macro")),
        ("n_by_site", n_by("site_id")),
        ("n_by_qc_flag", n_by("qc_flag")),
        ("n_by_eos_basis", n_by("eos_basis")),
        ("n_by_label_def_and_method", dict(Counter(f"{r['label_def']}|{r['method']}" for r in rows))),
        ("n_raw_records_by_sample_method", raw_by_method),
        ("n_raw_records_by_month", raw_by_month),
        ("n_raw_records_by_site", dict(Counter(c["site"] for c in clean))),
        ("year_range", [min(r["year"] for r in rows), max(r["year"] for r in rows)]),
        ("year_range_target", [min(r["year"] for r in target), max(r["year"] for r in target)] if target else []),
        ("n_target_rows", len(target)),
        ("target_rows", [{k: r[k] for k in ("site_id", "year", "date", "alt_cm", "alt_sd_cm", "n_obs", "method")}
                         for r in target]),
        ("alt_cm_stats_all_rows", stats_of([r["alt_cm"] for r in rows])),
        ("alt_cm_stats_target_rows", stats_of([r["alt_cm"] for r in target]) if target else None),
        ("alt_cm_stats_direct_dated_rows",
         stats_of([r["alt_cm"] for r in rows if r["label_def"] == "direct_dated"])),
        ("thaw_cm_stats_raw_records", stats_of([c["v"] for c in clean])),
        ("n_excluded_by_reason", excl),
        ("label_rule", "관측 월이 8월 또는 9월인 방문은 direct_eos(eos_basis = record_date), 7월 방문은 "
                       "direct_dated(eos_basis = none)다. 이 자료의 8월 방문은 2012-08-02, 2012-08-07, "
                       "2012-08-09, 2013-08-02 이고 9월 방문은 없다"),
        ("label_rule_deviation", deviation),
        ("method_evidence", {
            "ref_file": REF_HTML, "ref_file_found": ref_text is not None,
            "steel_rod_sentence_found": steel_rod_found,
            "max_thaw_period_sentence_found": max_thaw_found,
            "published_data_mapped_to": "probe" if published_is_probe else "other"}),
        ("checks", OrderedDict([
            ("thaw_depth_unit", "cm(열 이름 'Thaw depth [cm]', 매개변수 설명 'Thaw depth of active layer [cm]')"),
            ("coordinate_sign", "위도와 경도가 모두 양수다(북위, 동경). 지점 4곳이 모두 초쿠르다흐 부근 상자 "
                                f"(위도 {BBOX['lat'][0]}-{BBOX['lat'][1]}, 경도 {BBOX['lon'][0]}-{BBOX['lon'][1]}) 안에 있다"),
            ("record_checks", dict(sorted(chk.items()))),
            ("identical_records_note", "모든 열이 같은 기록이다. 기록에 측정점 식별자가 없어 같은 값의 다른 "
                                       "측정점인지 중복 입력인지 구분할 수 없다. 지우지 않았다"),
            ("most_frequent_values_cm", [[v, n] for v, n in top_values.most_common(5)]),
            ("largest_values_cm", [[v, top_values[v]] for v in sorted(top_values, reverse=True)[:5]]),
            ("coordinates_per_site_label", {s: list(next(iter(v))) for s, v in site_coords.items()}),
            ("shingubara2019_table1_coordinate_check", table1),
            ("visits_with_mixed_sample_method", n_mixed_method),
            ("site_years_with_more_than_one_visit", n_multi_visit_years),
        ])),
        ("visit_summary", visit_rows),
        ("cell_preview", cell_prev),
        ("v3_f4_direct_check", v3c),
        ("cross_source_check", xsrc),
        ("context_values", {
            "_설명": "라벨이 아니다. 값의 크기를 읽는 데 쓰는 참고 값이다",
            "calm_r31_allaiha": {
                "source": "data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab 의 이벤트 CALM_R31(격자 탐침, 계절 말 값)",
                "by_year": calm_r31,
                "dist_km_from_site_A": v3c["by_site"]["A"]["dist_km"] if v3c and "A" in v3c["by_site"] else None},
            "shingubara2019_table1": "둔덕 20-23 cm, 습지 31-56 cm(2010-2013년 7월 초에서 8월 초 관측의 평균)",
        }),
        ("coordinate_conversion", "변환 없음. 원자료가 WGS84 십진 도(소수 5자리)이고 동경은 양수다. "
                                  "좌표는 지점 단위다(기록마다 같은 지점 좌표가 반복된다)"),
        ("units", "cm. 변환하지 않았다"),
        ("missing_value_marks", "값 열에 빈 칸과 결측 부호가 없다. 'Date/time end', 'Tree ID', 'Comment' 는 "
                                "해당 없는 기록에서 빈 칸이다"),
        ("unverified_items", [
            "'Leveling measurement'(173건)와 'Soil sampling'(15건)의 측정 기기. PANGAEA 자료 설명에 없다. "
            "관련 논문(Liang 등 2023, JGR Biogeosciences)은 출판사 누리집이 HTTP 403 을 돌려주어 본문을 읽지 못했다",
            "8월 방문(8월 2일, 7일, 9일) 값과 연 최대 융해 깊이의 차이. 자료에 8월 중순 이후 방문이 없다. "
            "Shingubara 등 2019 는 최대 융해 깊이 시기를 8월 하순에서 9월 전반으로 적는다. 지점 A 에서 약 6 km, "
            "나머지 지점에서 28-35 km 떨어진 CALM R31 의 계절 말 값(context_values)과 이 자료의 8월 초 값은 "
            "지점, 미지형, 시기가 모두 달라 차이의 원인을 나눌 수 없다",
            "지점 A 의 이름과 경관. Shingubara 등 2019 의 표 1 에 없다. A 는 7월 기록만 있어 대상 행이 없다",
            "V 와 B 기록의 미지형 구성. 자료 제목은 둔덕(tree mounds)이나 V 의 2012-08-07 기록의 최댓값은 "
            f"{v_2012_max:g} cm 이고 Shingubara 등 2019 보충 표 S1 의 2012년 둔덕 범위는 16-27 cm 다. "
            "습지 관측이 섞였는지 확인하지 못했다",
            "'Published data' 기록과 Shingubara 등 2019 의 개별 관측값의 일대일 대응. 논문과 보충 자료에는 "
            "평균과 범위만 있다. 관측일은 보충 표 S2 와 맞는다(V 2012-08-07, 2013-08-02, B 2012-08-09, "
            "2013-07-16, K 2013-07-25 와 07-31). 개별 값은 ADS 자료(A20190211-001)에 있다고 하나 받지 않았다",
            "모든 열이 같은 기록이 다른 측정점인지 중복 입력인지 여부",
            "지점 안 측정점의 위치와 배치. 좌표는 지점 단위 하나뿐이다",
            "2011년 여름의 습윤 사건(큰 강수)이 융해 깊이에 준 영향. 교란으로 표시하지 않았다",
            "K_tree_wet 의 식생(물이끼, 사초)과 수위. Shingubara 등 2019 는 사초 습지의 수위가 지표보다 "
            "10 cm 넘게 높았던 때가 있다고 적는다. 이 자료의 습지 기록이 어느 식생인지 확인하지 못했다. "
            "disturbed = 0 으로 두었다",
        ]),
    ])
    meta_path = OUT_DIR / f"{SRC_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---------- 품질 요약 출력 ----------
    print(f"[{SRC_ID}] 원자료 {n_raw}건, 읽은 기록 {len(clean)}건, 방문 {len(visit_rows)}회, "
          f"점 자료 {len(rows)}행, 지점 표기 {meta['n_sites']}개, 좌표 {meta['n_coordinates']}개")
    print("기록 점검:", json.dumps(dict(sorted(chk.items())), ensure_ascii=False))
    print("원문 근거: 강철봉 문장", steel_rod_found, "/ 최대 융해 시기 문장", max_thaw_found)
    print("표 1 좌표 대조:", json.dumps(table1, ensure_ascii=False))
    print("label_def:", meta["n_by_label_def"], "method:", meta["n_by_method"], "macro:", meta["n_by_macro"])
    print("자료 표기별 기록 수:", raw_by_method, "월별 기록 수:", raw_by_month)
    print("연도 범위:", meta["year_range"], "대상 행 연도:", meta["year_range_target"])
    print("ALT 전 행:", meta["alt_cm_stats_all_rows"])
    print("ALT 대상 행:", meta["alt_cm_stats_target_rows"])
    print("대상 규칙 미통과:", json.dumps(excl, ensure_ascii=False))
    hdr = ("site_id", "year", "month", "date", "alt_cm", "alt_sd_cm", "n_obs", "method", "label_def", "eos_basis",
           "qc_flag")
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[k]) for k in hdr))
    tp = cell_prev["target_rows"]
    print(f"대상 셀 {tp['n_cells_1km']}개, 블록 {tp['n_blocks_0p5deg']}개 {tp['blocks']}")
    for c in tp["cells_location_by_site_id"]:
        print("  site_id 기준", c)
    for c in tp["cells_location_by_coordinate"]:
        print("  좌표 기준  ", c)
    print("v3 대조:", json.dumps(v3c["by_site"] if v3c else None, ensure_ascii=False))
    print("다른 점 자료 대조:", json.dumps(xsrc, ensure_ascii=False))
    print("산출:", points, meta_path, RAW_DIR / VISIT_CSV, sep="\n  ")


if __name__ == "__main__":
    main()
