#!/usr/bin/env python3
"""ext_labels 파서: ru_mamontovklyk_grosse.

자료: Grosse (2007), Active layer data of the geo-located sites around Cape Mamontov Klyk,
Appendix 4.2. PANGAEA, https://doi.org/10.1594/PANGAEA.611409 (CC BY 3.0).
원 출처: Schirrmeister 등 (2004), Reports on Polar and Marine Research 489 의 부록 4-1, 4-2.

입력(data/raw/ru_mamontovklyk_grosse/)
  PANGAEA_611409.tab          활동층 깊이 표(지점 98개). 라벨의 출처다
  PANGAEA_611407.tab          같은 조사의 지표 특성 표(지점 178개, 부록 4.1). 지형 분류의 출처다
  BerPolarforsch2004489.pdf   원정 보고서(선택). 부록 4-2 의 값 대조에만 쓴다
출력
  data/processed/ext_labels/ru_mamontovklyk_grosse_points.csv   표준 점 자료(형식 1.0)
  data/processed/ext_labels/ru_mamontovklyk_grosse_meta.json    메타
  data/raw/ru_mamontovklyk_grosse/report_appendix_4-2_check.csv 보고서 부록 4-2 와 PANGAEA 값의 대조

규칙(계획서 6B.3, _schema.json 1.0)
  1. 한 행은 (site_id, year) 이다. 이 자료는 2003년 단일 조사라 지점당 한 행이다.
  2. alt_cm 은 'ALD [cm]'(지점 평균)이다. 단위는 cm 이고 변환하지 않는다.
  3. 평균이 없는 행은 지우지 않는다. 'ALD min' 값을 alt_cm 에 넣고 right_censored = 1 로 둔다.
     '>' 표기는 측정이 동결면에 닿지 않은 하한이다. '>' 없이 최솟값만 있는 행은 평균의 하한이다.
  4. 관측 월이 8월 또는 9월이면 label_def = direct_eos, eos_basis = record_date 다.
     그 밖의 월이면 label_def = direct_dated, eos_basis = none 이다.
  5. method 는 other 다. 보고서 4.4.5 절과 부록은 측정 기기를 적지 않았다
     (PANGAEA 의 METHOD/DEVICE 는 'Visual observation' 이다).
  6. 지표 특성 표의 Relief type 이 열침식 지형(Log, Ovrag, Ovrag (mouth), Thermoerosional terrace)이거나
     Comment 에 'disturbed surface' 가 있으면 disturbed = 1, disturb_type = thermo_erosion 이다.
     지표 특성 표에 없는 지점은 빈 칸이다.

실행: 스레드 1개. 입력은 표 두 개(7 KB, 32 KB)와 PDF 한 쪽(180쪽)의 본문이다.
  OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_ru_mamontovklyk_grosse.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path

SRC_ID = "ru_mamontovklyk_grosse"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

TAB_ALD = "PANGAEA_611409.tab"
TAB_SURF = "PANGAEA_611407.tab"
JSONLD_NAME = "PANGAEA_611409_metadata.jsonld"
PDF_NAME = "BerPolarforsch2004489.pdf"
PDF_PAGE_APP42 = 180
ACCESSED = "2026-09-29"

URL = "https://doi.org/10.1594/PANGAEA.611409"
DOI = "10.1594/PANGAEA.611409"
FILE_URLS = {
    TAB_ALD: "https://doi.pangaea.de/10.1594/PANGAEA.611409?format=textfile",
    TAB_SURF: "https://doi.pangaea.de/10.1594/PANGAEA.611407?format=textfile",
    JSONLD_NAME: "https://doi.pangaea.de/10.1594/PANGAEA.611409?format=metadata_jsonld",
    PDF_NAME: "https://hdl.handle.net/10013/epic.10494.d001",
}
CITATION = ("Grosse, Guido (2007): Active layer data of the geo-located sites around Cape Mamontov Klyk, "
            "Appendix 4.2 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.611409")
CITATION_SURF = ("Grosse, Guido (2007): Surface parameters for the studied geolocated sites around Cape Mamontov "
                 "Klyk, Appendix 4.1 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.611407")
ORIG_SOURCE = ("Schirrmeister, L.; Grigoriev, M. N.; Kutzbach, L.; Wagner, D.; Bolshiyanov, D. Yu. (2004): "
               "Russian-German Cooperation System Laptev Sea: The Expedition Lena-Anabar 2003. "
               "Reports on Polar and Marine Research 489, 1-210, hdl:10013/epic.10494.d001 (Appendix 4-1, 4-2)")
LICENSE = "CC-BY-3.0"
COUNTRY = "Russia"
SUBUNIT = "Laptev_W_coast"
COORD_PREC = 0.0001

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
            "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
            "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

ALD_COLS = ["Sample ID", "Latitude", "Longitude", "Date/Time", "ALD [cm]", "ALD min [cm]", "ALD max [cm]",
            "NOBS [#]"]
SURF_COLS = ["Sample ID", "Latitude", "Longitude", "Date/Time", "Relief type", "Position", "Features",
             "Soil moisture", "Water bodies", "Comment"]
# 열침식 지형으로 보는 Relief type(보고서 표 4.4-4 의 thermo-erosional valley, ravine, terrace)
THERMO_EROSION = {"Log", "Ovrag", "Ovrag (mouth)", "Thermoerosional terrace"}


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
    return header, recs


def header_license(header: str) -> str:
    m = re.search(r"^License:\t(.+)$", header, re.M)
    return m.group(1).strip() if m else ""


def pdf_text(pdf: Path, first: int, last: int):
    if not pdf.exists() or shutil.which("pdftotext") is None:
        return None
    cmd = ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(pdf), "-"]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace")


def parse_appendix42(text: str):
    """보고서 부록 4-2(두 단 표)의 지점별 평균, 최소, 최대, N 을 읽는다."""
    i = text.rfind("Appendix 4-2. Active layer data")
    if i < 0:
        return {}
    rx = re.compile(r"(?<!\d)(\d{3})\s+(-|\d+)\s+(>?\d+)\s+(-|\d+)\s+(\d+)(?!\d)")
    out = {}
    for line in text[i:].splitlines():
        if "total" in line:
            continue
        for m in rx.finditer(line):
            out[m.group(1)] = {"mean": m.group(2), "min": m.group(3), "max": m.group(4), "n": m.group(5)}
    return out


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


def v3_check(points, v3_path: Path):
    """v3 의 F4_direct 셀과의 최소 거리와 체비쇼프 0.01° 중복을 센다. 표를 한 줄씩 읽고 영역 부분 집합만 비교한다."""
    if not v3_path.exists():
        return None
    ref = []
    with open(v3_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] == "F4_direct":
                la, lo = float(r["lat"]), float(r["lon"])
                if la > 60.0 and 90.0 < lo < 150.0:
                    ref.append((la, lo, r["region"], r["loc_id"]))
    best, n_dup = None, 0
    for p in points:
        la, lo = float(p["lat"]), float(p["lon"])
        for (rla, rlo, reg, lid) in ref:
            if max(abs(la - rla), abs(lo - rlo)) <= 0.01:
                n_dup += 1
            d = haversine_km(la, lo, rla, rlo)
            if best is None or d < best[0]:
                best = (d, reg, lid, p["site_id"])
    return {"n_ref_cells_compared": len(ref), "bbox_compared": "lat > 60, 90 < lon < 150",
            "min_distance_km": round(best[0], 1) if best else None,
            "nearest_v3_region": best[1] if best else None, "nearest_v3_loc_id": best[2] if best else None,
            "nearest_site_id": best[3] if best else None, "n_chebyshev_0p01_duplicates": n_dup}


def git_head(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True)
        return out.stdout.decode().strip()
    except Exception:
        return ""


def stats_of(vals):
    if not vals:
        return None
    return {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals),
            "mean": round(statistics.fmean(vals), 2)}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(";", ",")).strip()


def main():
    ap = argparse.ArgumentParser(description="ext_labels 파서: ru_mamontovklyk_grosse")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--v3", type=Path, default=V3_PATH)
    ap.add_argument("--no-report", action="store_true", help="보고서 PDF 를 읽지 않는다")
    args = ap.parse_args()

    header, recs = read_pangaea_tab(args.raw_dir / TAB_ALD, ALD_COLS)
    n_raw = len(recs)
    lic_ald = header_license(header)

    # ---- 지표 특성 표(PANGAEA 611407)
    surf, lic_surf, n_surf = {}, "", 0
    surf_path = args.raw_dir / TAB_SURF
    if surf_path.exists():
        h2, srecs = read_pangaea_tab(surf_path, SURF_COLS)
        lic_surf, n_surf = header_license(h2), len(srecs)
        for r in srecs:
            if r["Sample ID"] in surf:
                raise SystemExit(f"{TAB_SURF}: Sample ID 가 중복된다 {r['Sample ID']}")
            surf[r["Sample ID"]] = r

    # ---- 보고서 부록 4-2(값 대조)
    app42 = {}
    if not args.no_report:
        text = pdf_text(args.raw_dir / PDF_NAME, PDF_PAGE_APP42, PDF_PAGE_APP42)
        if text:
            app42 = parse_appendix42(text)

    # ---- 점 자료
    points, check42 = [], []
    n_surf_missing, n_surf_mismatch = 0, 0
    for r in recs:
        sid = r["Sample ID"]
        lat_s, lon_s = r["Latitude"], r["Longitude"]
        lat, lon = float(lat_s), float(lon_s)
        date = r["Date/Time"]
        md = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date)
        year, month = (int(md.group(1)), int(md.group(2))) if md else ("", "")
        mean_s, min_s, max_s = r["ALD [cm]"], r["ALD min [cm]"], r["ALD max [cm]"]
        n_obs = int(r["NOBS [#]"]) if r["NOBS [#]"] else 1

        notes, flags = [], []
        censored = 0
        if mean_s:
            alt, alt_out = float(mean_s), mean_s
            notes.append(f"ald_min_cm={min_s}")
            notes.append(f"ald_max_cm={max_s}")
        elif min_s.startswith(">"):
            alt, alt_out = float(min_s[1:]), min_s[1:]
            censored = 1
            notes.append(f"원자료 표기 '{min_s}'(ALD min 칸). 평균과 최대는 없다. alt_cm 은 하한이다")
        elif min_s:
            alt, alt_out = float(min_s), min_s
            censored = 1
            notes.append(f"평균과 최대가 없고 ALD min 값({min_s})만 있다. 사유는 확인하지 못했다. "
                         "alt_cm 은 지점 평균의 하한이다")
        else:
            alt, alt_out = None, ""
            notes.append("ALD 값이 없다")

        if alt is None or not (0 < alt <= 600):
            flags.append("range")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            flags.append("coord")
        if not md:
            flags.append("date")
        elif year < 1990:
            flags.append("year_pre1990")

        if md and month in (8, 9):
            label_def, eos_basis = "direct_eos", "record_date"
        else:
            label_def, eos_basis = "direct_dated", "none"

        disturbed, dtype = "", ""
        s = surf.get(sid)
        if s:
            if (s["Latitude"], s["Longitude"], s["Date/Time"]) != (lat_s, lon_s, date):
                n_surf_mismatch += 1
                notes.append("지표 특성 표(611407)의 좌표 또는 날짜가 다르다")
            is_te = s["Relief type"] in THERMO_EROSION or "disturbed surface" in s["Comment"].lower()
            disturbed = 1 if is_te else 0
            dtype = "thermo_erosion" if is_te else ""
            for key, col in [("relief", "Relief type"), ("position", "Position"), ("features", "Features"),
                             ("moisture", "Soil moisture"), ("water_bodies", "Water bodies"),
                             ("comment", "Comment")]:
                if s[col]:
                    notes.append(f"{key}={clean(s[col])}")
        elif surf:
            n_surf_missing += 1
            notes.append("지표 특성 표(611407)에 이 지점이 없다")

        if app42:
            a = app42.get(sid)
            same = bool(a) and (a["mean"].replace("-", "") == mean_s and a["min"] == min_s
                                and a["max"].replace("-", "") == max_s and a["n"] == str(n_obs))
            check42.append({"site_id": sid, "pangaea_mean": mean_s, "pangaea_min": min_s, "pangaea_max": max_s,
                            "pangaea_n": n_obs, "report_mean": a["mean"] if a else "",
                            "report_min": a["min"] if a else "", "report_max": a["max"] if a else "",
                            "report_n": a["n"] if a else "", "match": int(same)})

        points.append({
            "src_id": SRC_ID, "site_id": sid, "site_name": sid, "lat": lat_s, "lon": lon_s,
            "year": year, "month": month, "alt_cm": alt_out, "method": "other", "label_def": label_def,
            "n_obs": n_obs, "country": COUNTRY, "macro": macro_of(lat, lon), "citation": CITATION,
            "license": LICENSE, "subunit": SUBUNIT, "date": date, "eos_basis": eos_basis,
            "value_kind": "single_visit", "year_min": "", "year_max": "", "alt_sd_cm": "",
            "coord_prec_deg": COORD_PREC, "disturbed": disturbed, "disturb_type": dtype,
            "right_censored": censored, "orig_source": ORIG_SOURCE, "dup_of": "",
            "qc_flag": ";".join(flags) if flags else "ok", "notes": "; ".join(notes),
        })

    ids = [p["site_id"] for p in points]
    if len(set(ids)) != len(ids):
        raise SystemExit("site_id 가 중복된다")
    coords = [(p["lat"], p["lon"]) for p in points]
    n_dup_coord = len(coords) - len(set(coords))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{SRC_ID}_points.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(points)

    if check42:
        with open(args.raw_dir / "report_appendix_4-2_check.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(check42[0].keys()))
            w.writeheader()
            w.writerows(check42)

    # ---- 요약
    def is_target(p, use_disturbed=True):
        ok = (p["label_def"] == "direct_eos" and p["right_censored"] == 0 and p["qc_flag"] == "ok"
              and p["year"] != "" and p["year"] >= 1990)
        if use_disturbed:
            ok = ok and p["disturbed"] == 0
        return ok

    def cells_blocks(rows):
        cells = {cell_index(float(p["lat"]), float(p["lon"])) for p in rows}
        blocks = {block_index(float(p["lat"]), float(p["lon"])) for p in rows}
        return {"n_rows": len(rows), "n_cells_1km": len(cells), "n_blocks_0p5deg": len(blocks)}

    def relief_of(p):
        s = surf.get(p["site_id"])
        return s["Relief type"] if s and s["Relief type"] else "(없음)"

    mean_rows = [p for p in points if p["right_censored"] == 0]
    tgt = [p for p in points if is_target(p, True)]
    tgt_nodist = [p for p in points if is_target(p, False)]
    n_cens_gt = sum(1 for r in recs if r["ALD min [cm]"].startswith(">"))
    n_min_only = sum(1 for r in recs if not r["ALD [cm]"] and r["ALD min [cm]"]
                     and not r["ALD min [cm]"].startswith(">"))
    n_dist = sum(1 for p in points if p["disturbed"] == 1)
    n_dist_mean = sum(1 for p in mean_rows if p["disturbed"] == 1)
    n42_match = sum(c["match"] for c in check42)

    files = []
    for name in [TAB_ALD, TAB_SURF, JSONLD_NAME, PDF_NAME]:
        fp = args.raw_dir / name
        if fp.exists():
            files.append({"name": name, "size_bytes": fp.stat().st_size, "sha256": sha256_of(fp),
                          "url": FILE_URLS[name]})

    parser_path = Path(__file__).resolve()
    meta = {
        "src_id": SRC_ID,
        "name": "Grosse 2007, Cape Mamontov Klyk 주변 활동층 자료(PANGAEA 611409)",
        "schema_version": "1.0",
        "url": URL, "doi": DOI, "accessed": ACCESSED,
        "files": files,
        "license": LICENSE,
        "license_evidence": {TAB_ALD: lic_ald, TAB_SURF: lic_surf,
                             PDF_NAME: "약관을 확인하지 못했다. data/raw 에만 두고 값 대조와 방법 확인에 썼다"},
        "citation": CITATION,
        "citation_auxiliary": CITATION_SURF,
        "orig_source": ORIG_SOURCE,
        "parser": str(parser_path.relative_to(ROOT)),
        "parser_sha256": sha256_of(parser_path),
        "parser_git_commit": git_head(ROOT),
        "parser_git_note": "파서는 실행 시점에 커밋되지 않은 새 파일이다. 위 값은 저장소 HEAD 다",
        "n_rows_raw": n_raw,
        "n_rows_raw_surface_table": n_surf,
        "n_rows_points": len(points),
        "n_sites": len(set(ids)),
        "n_by_label_def": dict(Counter(p["label_def"] for p in points)),
        "n_by_method": dict(Counter(p["method"] for p in points)),
        "n_by_macro": dict(Counter(p["macro"] for p in points)),
        "n_by_qc_flag": dict(Counter(p["qc_flag"] for p in points)),
        "n_by_date": dict(sorted(Counter(p["date"] for p in points).items())),
        "year_range": [min(p["year"] for p in points), max(p["year"] for p in points)],
        "n_with_mean": len(mean_rows),
        "n_right_censored": sum(p["right_censored"] for p in points),
        "n_disturbed": n_dist,
        "n_obs_total": sum(p["n_obs"] for p in points),
        "n_obs_with_mean": sum(p["n_obs"] for p in mean_rows),
        "alt_cm_stats_with_mean": stats_of([float(p["alt_cm"]) for p in mean_rows]),
        "alt_cm_stats_target": stats_of([float(p["alt_cm"]) for p in tgt]),
        "alt_cm_stats_all_rows_including_lower_bounds": stats_of([float(p["alt_cm"]) for p in points]),
        "n_excluded_by_reason": {
            "_설명": ("점 자료에서 지운 행은 없다. 아래는 대상 라벨 규칙(계획서 6B.3)을 통과하지 못하는 행의 수다. "
                     "이 자료에서는 사유가 겹치는 행이 없다"),
            "dropped_from_points": 0,
            "right_censored_gt_mark": n_cens_gt,
            "right_censored_mean_missing_min_only": n_min_only,
            "disturbed_thermo_erosion": n_dist,
            "disturbed_thermo_erosion_among_rows_with_mean": n_dist_mean,
            "label_def_not_direct_eos": sum(1 for p in points if p["label_def"] != "direct_eos"),
            "qc_flag_not_ok": sum(1 for p in points if p["qc_flag"] != "ok"),
        },
        "n_target_rows": len(tgt),
        "n_target_rows_if_disturbed_ignored": len(tgt_nodist),
        "cell_preview": {
            "_설명": "v4 조립 전의 참고 값이다. 셀 색인은 계획서 6B.3 의 식, 블록은 0.5° 다",
            "rows_with_mean": cells_blocks(mean_rows),
            "target_rows": cells_blocks(tgt),
            "target_rows_if_disturbed_ignored": cells_blocks(tgt_nodist),
        },
        "relief_counts_all_rows": dict(Counter(relief_of(p) for p in points)),
        "relief_counts_rows_with_mean": dict(Counter(relief_of(p) for p in mean_rows)),
        "relief_counts_target_rows": dict(Counter(relief_of(p) for p in tgt)),
        "label_def_basis": ("개정 13(WRAPUP 7.3 (a)12 기록): direct_eos 로 둔 근거는 세 가지다. (1) 원정 조사 중 지점에서 잰 활동층 "
                            "깊이(ALD, 지점 평균)이고 모형이나 지온에서 유도한 값이 아니다. (2) 기록 날짜가 2003-08-15–28 로 6B.3 의 "
                            "record_date 창(8–9월)에 든다. (3) 동결면에 닿지 않은 값은 '>' 로 표기되어 절단으로 따로 처리했다. 측정 기기는 "
                            "원 보고서와 PANGAEA 에 없고(METHOD/DEVICE 'Visual observation'), 그래서 method = other 이며 L41 (a)에서 빠진다. "
                            "2003년 단일 조사라 L41 (d)(year_max >= 2010)에서도 빠진다"),
        "disturbed_rule": ("PANGAEA 611407 의 Relief type 이 Log(thermo-erosional valley), Ovrag(thermo-erosional ravine), "
                           "Ovrag (mouth), Thermoerosional terrace 이거나 Comment 에 'disturbed surface' 가 있으면 "
                           "disturbed = 1, disturb_type = thermo_erosion. 그 밖의 분류(Edoma, Elevated plain (no Edoma), "
                           "TK depression, River valley, Marine terrace, Tidal flat)는 0. "
                           "Relief type 은 조사자가 지점마다 적은 경관 단위이고 조사 시점의 활성 침식 여부를 뜻하지 않는다"),
        "surface_table_check": {
            "used": bool(surf),
            "points_without_surface_row": n_surf_missing,
            "coordinate_or_date_mismatch": n_surf_mismatch,
        },
        "report_check": {
            "report_used": bool(app42),
            "pdf_page_read": PDF_PAGE_APP42 if app42 else "",
            "appendix_4-2_sites_parsed": len(app42),
            "appendix_4-2_value_match": n42_match,
            "appendix_4-2_value_mismatch": len(check42) - n42_match,
        },
        "duplicate_check": {"duplicate_site_id": 0, "duplicate_coordinates": n_dup_coord},
        "v3_check": v3_check(points, args.v3),
        "coordinate_conversion": "변환 없음. 원자료가 십진 도(소수 4자리, 동경 양수)다. 문자열 그대로 옮겼다. 측지계 표기는 원자료에 없다",
        "units": "ALD 는 cm 다. 변환하지 않았다",
        "missing_value_marks": "PANGAEA 표는 빈 칸, 보고서 부록 4-2 는 '-' 다. '>' 는 하한 표기다",
        "unverified_items": [
            "측정 기기. 보고서 4.4.5 절과 표 4.4-3 은 'Active layer depth: Up to 15 measurements per site' 만 적었다. "
            "탐침 사용을 확인하지 못해 method 를 other 로 두었다. 같은 보고서 3.6 절의 강철봉 기술은 레나 델타 Samoylov 섬 조사의 것이다",
            "평균 없이 ALD min 값만 있는 두 지점(320, 351)의 사유",
            "'>' 표기 5지점(299, 486, 487, 488, 489)의 측정 한계가 탐침 길이인지 굴착 깊이인지",
            "보고서 부록 4-2 의 주석은 조사 기간을 2004-08-15 에서 2004-08-28 로 적었다. 보고서 4.1 절은 조사를 2003년 여름으로 적었고 "
            "PANGAEA 두 표의 날짜가 2003-08-15 에서 2003-08-28 이므로 2003년으로 두었다",
            "Relief type 이 Log, Ovrag 인 지점이 계획서 6B.3 의 '열침식 지형' 교란 규칙의 대상인지. 이 파서는 문자 그대로 적용했다",
            "Water bodies 칸은 지점의 수체 유무와 수심이다. 측정 위치가 수체 바닥인지는 확인하지 못해 water 로 표시하지 않았다",
            "좌표의 측지계. 원자료에 표기가 없다. GPS 좌표로 보고 WGS84 로 다루었다",
            "보고서 PDF 의 이용 약관",
        ],
    }
    with open(args.out_dir / f"{SRC_ID}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---- 화면 요약
    print(f"[{SRC_ID}] 원자료 {n_raw}행, 점 자료 {len(points)}행, 지점 {len(set(ids))}개")
    print(f"  평균 있음 {len(mean_rows)}, 하한 표기 {n_cens_gt}, 최솟값만 {n_min_only}, 열침식 지형 {n_dist}")
    print(f"  label_def {meta['n_by_label_def']}, method {meta['n_by_method']}, macro {meta['n_by_macro']}")
    print(f"  qc_flag {meta['n_by_qc_flag']}, 날짜 {meta['n_by_date']}")
    print(f"  ALT(평균 있는 행) {meta['alt_cm_stats_with_mean']}")
    print(f"  ALT(대상 행) {meta['alt_cm_stats_target']}")
    print(f"  대상 행 {len(tgt)} (교란 표시를 쓰지 않으면 {len(tgt_nodist)})")
    print(f"  셀 미리보기 {meta['cell_preview']}")
    print(f"  지형(평균 있는 행) {meta['relief_counts_rows_with_mean']}")
    print(f"  지표 특성 표 대조 {meta['surface_table_check']}")
    print(f"  보고서 대조 {meta['report_check']}")
    print(f"  v3 대조 {meta['v3_check']}")
    print(f"  출력 {out_csv}")


if __name__ == "__main__":
    main()
