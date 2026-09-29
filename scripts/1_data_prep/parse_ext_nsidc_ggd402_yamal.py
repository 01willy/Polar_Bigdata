#!/usr/bin/env python3
"""ext_labels 파서: nsidc_ggd402_yamal.

자료: Minkin, M. & Melnikov, E. (1998), Borehole and environmental protection descriptive and
numerical data, Yamal Peninsula, Russia, Version 1 (NSIDC GGD402). https://doi.org/10.7265/h303-nt88
원 출처: Fundamentproekt 설계 연구소(모스크바)의 시추공 자료 가운데 고른 160공. CAPS 1.0 CD-ROM(1998) 수록.

입력(data/raw/nsidc_ggd402_yamal/)
  yabrhl.txt   시추공 기록, 기록 단위 서식(158공)
  yabrhl.dat   시추공 기록, 탭 구분 표(124공)
  yadesc.dat   자연 보호 구역 55개의 설명 표(구역별 계절 융해 깊이 범위)
출력
  data/processed/ext_labels/nsidc_ggd402_yamal_points.csv   표준 점 자료(형식 1.0)
  data/processed/ext_labels/nsidc_ggd402_yamal_meta.json    메타
  data/raw/nsidc_ggd402_yamal/yabrhl_merged.csv             두 파일을 합친 시추공 단위 표(18칸 원문과 출처 파일)

규칙(계획서 6B.2 보조 라벨, 6B.3, _schema.json 1.0, _sources_plan.json 의 label_rule)
  1. 한 행은 시추공 하나다. (구역 번호, 시추공 번호)가 열쇠다. 시추공 번호만으로는 유일하지 않다.
  2. 두 파일의 합집합을 쓴다. 두 파일에 모두 있는 시추공은 수치 칸이 같은지 확인하고 한 행으로 둔다.
     Topography 는 yabrhl.dat 의 값이 길어 그 값을 쓴다(yabrhl.txt 는 첫 낱말만 있다).
  3. alt_cm 은 'Max depth seasonal freeze(thaw),m' 에 100 을 곱한 값이다.
  4. method = pit_core, label_def = unknown, eos_basis = none 이다. 이 값을 어떻게 정했는지
     (코어 관찰인지 계산인지)를 자료 설명에서 확인하지 못했다. 시추일이 3-5월인 공은 시추일의 융해 깊이가
     0 인데도 값이 있으므로 시추일에 잰 융해 깊이가 아니다.
  5. date 와 month 는 시추일이다. 라벨 값의 관측일이 아니다.
  6. 연도가 모두 1990년 이전이라 qc_flag 에 year_pre1990 이 붙는다. 확인적 풀과 대상 셀에 넣지 않는다.
  7. 좌표는 DDMMSS 정수다. 십진 도 = DD + MM/60 + SS/3600. 북위, 동경이고 부호 변환은 없다.
     coord_prec_deg 는 초 자리가 둘 다 00 이면 0.0167(분 단위), 아니면 0.0003(초 단위)이다.
  8. 값에 '>' 표기가 없어 right_censored 는 모두 0 이다.

실행: 스레드 1개. 입력은 합쳐 약 130 KB 이고 v3 표는 좌표 칸만 읽는다.
  OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_nsidc_ggd402_yamal.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

SRC_ID = "nsidc_ggd402_yamal"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

ACCESSED = "2026-09-29"
URL = "https://nsidc.org/data/ggd402/versions/1"
DOI = "10.7265/h303-nt88"
FTP_DIR = "ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd402_boreholes_russia/"
FILE_URLS = {
    "yabrhl.txt": FTP_DIR + "yabrhl.txt",
    "yabrhl.dat": FTP_DIR + "yabrhl.dat",
    "yadesc.dat": FTP_DIR + "yadesc.dat",
    "README.txt": FTP_DIR + "README.txt",
    "ggd402-userguide-v1.pdf": "https://nsidc.org/sites/default/files/ggd402-userguide-v1.pdf",
    "GGD402.001_dif.xml": ("https://nsidc.org/api/dataset/metadata/v2/oai?verb=GetRecord"
                           "&metadataPrefix=dif&identifier=GGD402.001"),
}
CITATION = ("Minkin, M. & Melnikov, E. (1998). Borehole and environmental protection descriptive and numerical "
            "data, Yamal Peninsula, Russia (GGD402, Version 1) [Data Set]. Boulder, Colorado USA. National Snow "
            "and Ice Data Center. https://doi.org/10.7265/h303-nt88. Date Accessed 09-29-2026.")
ORIG_SOURCE = ("Fundamentproekt Design Institute (Moscow; PI M. A. Minkin), borehole database 1977-1990, "
               "Kharasavey and Bovanenkovo gas fields and the Yamal-Ukhta, Yamal-Uzhgorod pipeline routes; "
               "CAPS Version 1.0 CD-ROM (IPA Data and Information Working Group, 1998)")
LICENSE = "unverified"
COUNTRY = "Russia"
SUBUNIT = "Yamal"
PREC_MINUTE = 0.0167
PREC_SECOND = 0.0003
# 자료 누리집이 적은 공간 범위(북위 66-72.5, 동경 65.5-74). 좌표 부호와 단위 점검에 쓴다
BBOX = dict(lat_min=66.0, lat_max=72.5, lon_min=65.5, lon_max=74.0)

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
            "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
            "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

# 표준 칸 이름(yabrhl.dat 머리글 순서). yabrhl.txt 의 이름은 두 칸이 조금 다르다
FIELDS = ["region", "borehole", "lat_ddmmss", "lon_ddmmss", "altitude_m", "date", "borehole_depth_m",
          "thaw_at_coring_m", "max_seasonal_thaw_m", "permafrost_table_m", "t10m_degc", "soil_active_layer",
          "soil_permafrost_1", "soil_permafrost_2", "ice_content_lenses", "salinity_type", "dsal_pct",
          "topography"]
DAT_HEADER = ["# Region", "# Borehole", "Latitude", "Longitude", "Altitude,m", "Date", "Depth of borehole, m",
              "Depth of freeze (thaw)at a coring date,m", "Max depth seasonal freeze(thaw),m",
              "Depth of permafrost upper limit, m", "Ground temperature at 10 m deep,degree C",
              "Type of geological section, active layer", "Type of geological section,permafrost,1-st layer",
              "Type of geological section,permafrost,2-nd layer", "Ice content(lenses)",
              "Value and type of salinity, permafrost", "Degree of salinity, Dsal, %", "Topography"]
TXT_KEYS = list(DAT_HEADER)
TXT_KEYS[6] = "Depth of borehole"
TXT_KEYS[7] = "Depth of freeze (thaw)at a coring date"
NUMERIC = ["altitude_m", "borehole_depth_m", "thaw_at_coring_m", "max_seasonal_thaw_m", "permafrost_table_m",
           "t10m_degc", "ice_content_lenses"]
MISSING_MARKS = {"", "-"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True)
        return out.stdout.decode().strip()
    except Exception:
        return ""


def macro_of(lat: float, lon: float) -> str:
    """계획서 6B.4 의 지리 정의(러시아 안에서만 쓴다). 라벨 값을 쓰지 않는다."""
    if 71.5 <= lat <= 73.6 and 123.3 <= lon <= 130.1:
        return "Lena"
    if lon < 0 or lon >= 140.0:
        return "Russia_E"
    if lon < 90.0:
        return "Russia_W"
    return "Russia_C"


def read_txt(path: Path):
    """yabrhl.txt 를 기록 단위로 읽는다. 기록은 '-' 줄로 시작하고 칸은 '이름:값' 이다."""
    recs, cur = [], None
    for ln, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if line.startswith("-----"):
            if cur is not None:
                recs.append(cur)
            cur = {"_line": ln}
            continue
        if not line.strip():
            continue
        if cur is None:
            raise SystemExit(f"{path.name}:{ln} 구분 줄 앞에 자료가 있다")
        key, sep, val = line.partition(":")
        if not sep:
            raise SystemExit(f"{path.name}:{ln} ':' 이 없는 줄이다: {line!r}")
        if key in cur:
            raise SystemExit(f"{path.name}:{ln} 한 기록에 칸 이름이 두 번 나온다: {key!r}")
        cur[key] = val.strip()
    if cur is not None:
        recs.append(cur)
    out = []
    for r in recs:
        miss = [k for k in TXT_KEYS if k not in r]
        extra = [k for k in r if k not in TXT_KEYS and k != "_line"]
        if miss or extra:
            raise SystemExit(f"{path.name}:{r['_line']} 칸이 맞지 않는다. 없음 {miss}, 남음 {extra}")
        rec = {f: r[k] for f, k in zip(FIELDS, TXT_KEYS)}
        rec["_where"] = f"{path.name}:{r['_line']}"
        out.append(rec)
    return out


def read_dat(path: Path):
    """yabrhl.dat 를 읽는다. 탭 구분이고 쉼표가 든 머리글은 큰따옴표로 싸여 있다."""
    with open(path, encoding="ascii", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if rows[0] != DAT_HEADER:
        raise SystemExit(f"{path.name} 머리글이 예상과 다르다: {rows[0]}")
    out = []
    for i, r in enumerate(rows[1:], 2):
        if len(r) != len(FIELDS):
            raise SystemExit(f"{path.name}:{i} 칸 수가 {len(r)} 이다")
        rec = {f: v.strip() for f, v in zip(FIELDS, r)}
        rec["_where"] = f"{path.name}:{i}"
        out.append(rec)
    return out


def read_desc(path: Path):
    """yadesc.dat 에서 구역별 계절 융해 깊이 범위(m)를 읽는다."""
    with open(path, encoding="ascii", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    h = rows[0]
    i_reg = h.index("NATURE PROTECTION REGION")
    i_sub = h.index("SUBPROVINCE")
    i_lo, i_hi = h.index("SEASONAL THAW DEPTH MIN"), h.index("SEASONAL THAW DEPTH MAX")
    out = {}
    for r in rows[1:]:
        reg = int(r[i_reg])
        if reg in out:
            raise SystemExit(f"{path.name} 구역 번호가 중복된다: {reg}")
        out[reg] = {"subprovince": r[i_sub].strip(), "thaw_min_m": r[i_lo].strip(), "thaw_max_m": r[i_hi].strip()}
    return out


def ddmmss(s: str):
    """DDMMSS 정수 문자열을 (도, 분, 초, 십진 도)로 바꾼다."""
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"DDMMSS 6자리가 아니다: {s!r}")
    d, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
    if m >= 60 or sec >= 60:
        raise ValueError(f"분 또는 초가 60 이상이다: {s!r}")
    return d, m, sec, d + m / 60.0 + sec / 3600.0


def fmt_num(x: Decimal) -> str:
    s = format(x.normalize(), "f")
    return s


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / 0.009)
    phi = math.radians((ky + 0.5) * 0.009)
    kx = math.floor(lon * math.cos(phi) / 0.009)
    return ky, kx


def block_id(lat: float, lon: float) -> int:
    return math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5)


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def stats_of(vals):
    if not vals:
        return None
    return {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals),
            "mean": round(statistics.fmean(vals), 2)}


def merge(txt, dat):
    """두 파일의 합집합을 만든다. 열쇠는 (구역 번호, 시추공 번호)의 정수 쌍이다."""
    merged, order = {}, []
    for name, recs in (("txt", txt), ("dat", dat)):
        seen = set()
        for r in recs:
            key = (int(r["region"]), int(r["borehole"]))
            if key in seen:
                raise SystemExit(f"{r['_where']} 같은 파일 안에 열쇠가 중복된다: {key}")
            seen.add(key)
            if key not in merged:
                merged[key] = {"rec": dict(r), "in": [name], "where": [r["_where"]], "diff": []}
                order.append(key)
                continue
            m = merged[key]
            m["in"].append(name)
            m["where"].append(r["_where"])
            for f in FIELDS[2:]:
                a, b = m["rec"][f], r[f]
                if a == b:
                    continue
                if f in NUMERIC or f == "dsal_pct":
                    try:
                        same = Decimal(a) == Decimal(b)
                    except Exception:
                        same = False
                    if same:
                        continue
                m["diff"].append(f)
                if f == "topography" and name == "dat":
                    m["rec"]["topography_txt"] = a
                    m["rec"][f] = b
    return merged, order


def v3_reference(path: Path):
    """v3 표에서 좌표, 지역, 자료 구분만 읽는다. 표는 고치지 않는다."""
    ref = []
    if not path.exists():
        return ref
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lat, lon = float(r["lat"]), float(r["lon"])
            if 60.0 <= lat <= 80.0 and 50.0 <= lon <= 95.0:
                ref.append({"loc_id": r["loc_id"], "lat": lat, "lon": lon, "region": r["region"],
                            "source_id": r["source_id"]})
    return ref


def other_sources_nearby(out_dir: Path, pts):
    """같은 폴더의 다른 자료원 점 자료 가운데 야말 주변(북위 66-73, 동경 65-75) 지점과의 거리를 본다.

    다른 자료원의 점 자료는 이 파서를 실행한 시점에 있던 파일만 읽는다. 자료원 사이의 중복 판정은
    v4 조립 단계의 일이다. 여기서는 거리만 적는다.
    """
    res = {}
    for p in sorted(out_dir.glob("*_points.csv")):
        if p.name == f"{SRC_ID}_points.csv":
            continue
        try:
            sites = {}
            with open(p, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        lat, lon = float(r["lat"]), float(r["lon"])
                    except (ValueError, KeyError):
                        continue
                    if 66.0 <= lat <= 73.0 and 65.0 <= lon <= 75.0:
                        sites[r["site_id"]] = (lat, lon, r.get("site_name", ""))
        except Exception as e:  # 다른 작업이 쓰는 중인 파일일 수 있다
            res[p.name] = {"error": str(e)}
            continue
        if not sites:
            continue
        best = None
        n_cheb = 0
        for sid, (lat, lon, name) in sites.items():
            for q in pts:
                d = km(lat, lon, q["_lat"], q["_lon"])
                if max(abs(lat - q["_lat"]), abs(lon - q["_lon"])) <= 0.01:
                    n_cheb += 1
                if best is None or d < best[0]:
                    best = (d, sid, name, q["site_id"])
        res[p.name] = {"n_sites_in_bbox": len(sites), "min_distance_km": round(best[0], 2),
                       "nearest_other_site_id": best[1], "nearest_other_site_name": best[2],
                       "nearest_site_id": best[3], "n_pairs_chebyshev_0p01": n_cheb}
    return res


def main():
    ap = argparse.ArgumentParser(description="ext_labels 파서: nsidc_ggd402_yamal")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--v3", type=Path, default=V3_PATH)
    args = ap.parse_args()

    txt = read_txt(args.raw_dir / "yabrhl.txt")
    dat = read_dat(args.raw_dir / "yabrhl.dat")
    desc = read_desc(args.raw_dir / "yadesc.dat")
    merged, order = merge(txt, dat)

    n_in = Counter("+".join(merged[k]["in"]) for k in order)
    diff_fields = Counter(f for k in order for f in merged[k]["diff"])
    bad_numeric_diff = {f: n for f, n in diff_fields.items() if f != "topography"}
    if bad_numeric_diff:
        raise SystemExit(f"두 파일의 값이 다른 칸이 있다: {bad_numeric_diff}")

    points, raw_rows = [], []
    missing_counts = Counter()
    n_gt_mark = 0
    outside_region_range = []
    for key in sorted(order):
        m = merged[key]
        r = m["rec"]
        region, bh = key
        site_id = f"R{region:02d}_B{bh:06d}"
        for f in FIELDS:
            if r[f] in MISSING_MARKS:
                missing_counts[f] += 1
        flags, notes = [], []

        # ---- 좌표
        try:
            la_d, la_m, la_s, lat = ddmmss(r["lat_ddmmss"])
            lo_d, lo_m, lo_s, lon = ddmmss(r["lon_ddmmss"])
        except ValueError as e:
            raise SystemExit(f"{m['where']} 좌표를 읽지 못했다: {e}")
        if not (BBOX["lat_min"] <= lat <= BBOX["lat_max"] and BBOX["lon_min"] <= lon <= BBOX["lon_max"]):
            flags.append("coord")
        minute_only = (la_s == 0 and lo_s == 0)
        prec = PREC_MINUTE if minute_only else PREC_SECOND

        # ---- 시추일
        try:
            y, mo, d = (int(x) for x in r["date"].split("."))
            date = dt.date(y, mo, d)
        except Exception:
            date = None
        if date is None:
            flags.append("date")
            year, month, date_s = "", "", ""
        else:
            year, month, date_s = date.year, date.month, date.isoformat()

        # ---- 라벨
        raw_val = r["max_seasonal_thaw_m"]
        if raw_val.startswith((">", "<")):
            n_gt_mark += 1
        try:
            val_m = Decimal(raw_val.lstrip("><"))
        except Exception:
            val_m = None
        if val_m is None:
            alt_out, alt = "", None
        else:
            alt_dec = abs(val_m) * 100
            alt_out, alt = fmt_num(alt_dec), float(alt_dec)
        if alt is None or not (0 < alt <= 600):
            flags.append("range")
        if year != "" and year < 1990:
            flags.append("year_pre1990")
        censored = 1 if raw_val.startswith(">") else 0

        # ---- 구역 설명 표의 융해 깊이 범위와 견준다(표시만 한다)
        dsc = desc.get(region)
        in_range = ""
        if dsc and val_m is not None and dsc["thaw_min_m"] and dsc["thaw_max_m"]:
            lo, hi = Decimal(dsc["thaw_min_m"]), Decimal(dsc["thaw_max_m"])
            in_range = int(lo <= val_m <= hi)
            if not in_range:
                outside_region_range.append(site_id)

        notes.append("date 와 month 는 시추일이다. alt_cm 은 시추일에 잰 값이 아니다")
        notes.append(f"thaw_at_coring_cm={fmt_num(Decimal(r['thaw_at_coring_m']) * 100)}")
        notes.append(f"permafrost_table_cm={fmt_num(Decimal(r['permafrost_table_m']) * 100)}")
        notes.append(f"region={region}")
        if dsc:
            notes.append(f"subprovince={dsc['subprovince']}")
            notes.append(f"region_thaw_range_m={dsc['thaw_min_m']}-{dsc['thaw_max_m']}")
        notes.append(f"topography={r['topography']}")
        notes.append(f"soil_active_layer={r['soil_active_layer']}")
        notes.append(f"t10m_degC={r['t10m_degc']}")
        notes.append(f"altitude_m={r['altitude_m']}")
        notes.append(f"borehole_depth_m={r['borehole_depth_m']}")
        notes.append(f"coord_raw={r['lat_ddmmss']}N,{r['lon_ddmmss']}E")
        notes.append("src_file=" + "+".join(m["in"]))

        points.append({
            "src_id": SRC_ID, "site_id": site_id, "site_name": f"borehole {bh:06d} (region {region:02d})",
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "year": year, "month": month, "alt_cm": alt_out,
            "method": "pit_core", "label_def": "unknown", "n_obs": 1, "country": COUNTRY,
            "macro": macro_of(lat, lon), "citation": CITATION, "license": LICENSE, "subunit": SUBUNIT,
            "date": date_s, "eos_basis": "none", "value_kind": "", "year_min": "", "year_max": "",
            "alt_sd_cm": "", "coord_prec_deg": prec, "disturbed": "", "disturb_type": "",
            "right_censored": censored, "orig_source": ORIG_SOURCE, "dup_of": "",
            "qc_flag": ";".join(flags) if flags else "ok",
            "notes": "; ".join(n.replace(";", ",") for n in notes),
            "_lat": lat, "_lon": lon, "_alt": alt, "_minute_only": minute_only,
            "_thaw_coring": float(r["thaw_at_coring_m"]), "_ptab": float(r["permafrost_table_m"]),
            "_val_m": float(val_m) if val_m is not None else None, "_region": region,
        })
        raw = {"site_id": site_id}
        raw.update({f: r[f] for f in FIELDS})
        raw["topography_txt"] = r.get("topography_txt", r["topography"] if m["in"] == ["txt"] else "")
        raw["in_files"] = "+".join(m["in"])
        raw["where"] = " | ".join(m["where"])
        raw["lat_deg"] = f"{lat:.6f}"
        raw["lon_deg"] = f"{lon:.6f}"
        raw["region_thaw_min_m"] = dsc["thaw_min_m"] if dsc else ""
        raw["region_thaw_max_m"] = dsc["thaw_max_m"] if dsc else ""
        raw["max_thaw_in_region_range"] = in_range
        raw_rows.append(raw)

    # ---- 중복 점검
    sid_dup = [s for s, n in Counter(p["site_id"] for p in points).items() if n > 1]
    if sid_dup:
        raise SystemExit(f"site_id 가 중복된다: {sid_dup}")
    bh_groups = defaultdict(list)
    for key in order:
        bh_groups[key[1]].append(key[0])
    bh_repeated = {f"{b:06d}": sorted(v) for b, v in bh_groups.items() if len(v) > 1}
    coord_groups = defaultdict(list)
    for p in points:
        coord_groups[(p["lat"], p["lon"])].append(p["site_id"])
    coord_shared = [{"lat": k[0], "lon": k[1], "site_ids": v} for k, v in coord_groups.items() if len(v) > 1]
    full_dups = defaultdict(list)
    for p in points:
        full_dups[(p["lat"], p["lon"], p["date"], p["alt_cm"])].append(p["site_id"])
    full_dup_groups = [v for v in full_dups.values() if len(v) > 1]

    # ---- 쓰기: 점 자료
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.out_dir / f"{SRC_ID}_points.csv"
    with open(points_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for p in points:
            w.writerow(p)

    # ---- 쓰기: 합친 시추공 표(원자료 폴더)
    merged_path = args.raw_dir / "yabrhl_merged.csv"
    raw_cols = (["site_id"] + FIELDS + ["topography_txt", "in_files", "where", "lat_deg", "lon_deg",
                                        "region_thaw_min_m", "region_thaw_max_m", "max_thaw_in_region_range"])
    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_cols, lineterminator="\n")
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    # ---- 요약
    alts = [p["_alt"] for p in points if p["_alt"] is not None]
    years = [p["year"] for p in points if p["year"] != ""]
    months = Counter(p["month"] for p in points)
    by_month_ratio = {}
    for mo in sorted(months):
        rr = [p["_thaw_coring"] / p["_val_m"] for p in points if p["month"] == mo and p["_val_m"]]
        by_month_ratio[str(mo)] = {"n": len(rr), "median": round(statistics.median(rr), 2),
                                   "min": round(min(rr), 2), "max": round(max(rr), 2)}
    n_eq_ptab = sum(1 for p in points if p["_val_m"] == p["_ptab"])
    n_coring_zero = sum(1 for p in points if p["_thaw_coring"] == 0)
    n_coring_gt = sum(1 for p in points if p["_val_m"] is not None and p["_thaw_coring"] > p["_val_m"])

    # ---- 셀 미리 보기와 v3 대조
    ref = v3_reference(args.v3)
    f4 = [v for v in ref if v["source_id"] == "F4_direct"]
    oth = [v for v in ref if v["source_id"] != "F4_direct"]
    cells = defaultdict(list)
    for p in points:
        cells[cell_index(p["_lat"], p["_lon"])].append(p)
    cell_rows = []
    for ck, ps in cells.items():
        clat = statistics.fmean(q["_lat"] for q in ps)
        clon = statistics.fmean(q["_lon"] for q in ps)
        dup = any(max(abs(v["lat"] - clat), abs(v["lon"] - clon)) <= 0.01 for v in f4)
        near_oth = any(max(abs(v["lat"] - clat), abs(v["lon"] - clon)) <= 0.01 for v in oth)
        cell_rows.append({"lat": clat, "lon": clon, "n": len(ps), "dup_v3_f4": dup, "near_v3_other": near_oth,
                          "alt": statistics.fmean(q["_alt"] for q in ps),
                          "minute_only": all(q["_minute_only"] for q in ps)})
    new_cells = [c for c in cell_rows if not c["dup_v3_f4"]]
    v3_f4_blocks = set()
    if args.v3.exists():
        with open(args.v3, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["source_id"] == "F4_direct":
                    v3_f4_blocks.add(block_id(float(r["lat"]), float(r["lon"])))
    blocks_new = set(block_id(c["lat"], c["lon"]) for c in new_cells)
    grid001 = set((math.floor(p["_lat"] / 0.01), math.floor(p["_lon"] / 0.01)) for p in points)

    nearest = None
    n_pt_cheb = 0
    for p in points:
        for v in f4:
            d = km(p["_lat"], p["_lon"], v["lat"], v["lon"])
            if max(abs(v["lat"] - p["_lat"]), abs(v["lon"] - p["_lon"])) <= 0.01:
                n_pt_cheb += 1
            if nearest is None or d < nearest[0]:
                nearest = (d, v, p["site_id"])
    within = {f"n_points_within_{t}km_of_v3_f4": sum(
        1 for p in points if any(km(p["_lat"], p["_lon"], v["lat"], v["lon"]) <= t for v in f4))
        for t in (5, 25, 100)}
    nearest_other = None
    for p in points:
        for v in oth:
            d = km(p["_lat"], p["_lon"], v["lat"], v["lon"])
            if nearest_other is None or d < nearest_other[0]:
                nearest_other = (d, v, p["site_id"])

    files_meta = []
    for name in ["yabrhl.txt", "yabrhl.dat", "yadesc.dat", "README.txt", "ggd402-userguide-v1.pdf",
                 "GGD402.001_dif.xml"]:
        p = args.raw_dir / name
        if p.exists():
            files_meta.append({"name": name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p),
                               "url": FILE_URLS[name],
                               "role": "parsed" if name in ("yabrhl.txt", "yabrhl.dat", "yadesc.dat") else "documentation"})
    derived_meta = [{"name": merged_path.name, "size_bytes": merged_path.stat().st_size,
                     "sha256": sha256_of(merged_path),
                     "note": "이 파서가 만든 표다. yabrhl.txt 와 yabrhl.dat 의 합집합(시추공 단위, 원문 값)"}]

    flag_counts = Counter()
    for p in points:
        for fl in p["qc_flag"].split(";"):
            flag_counts[fl] += 1

    meta = {
        "src_id": SRC_ID,
        "name": "NSIDC GGD402 야말 반도 시추공 기재 자료(Borehole and environmental protection descriptive and "
                "numerical data, Yamal Peninsula, Russia, Version 1)",
        "schema_version": "1.0",
        "url": URL,
        "doi": DOI,
        "download_dir": FTP_DIR,
        "accessed": ACCESSED,
        "files": files_meta,
        "derived_files_in_raw_dir": derived_meta,
        "license": LICENSE,
        "license_evidence": (
            "자료 누리집과 사용 안내서는 'As a condition of using these data, you must cite the use of this data set' "
            "라는 인용 조건만 적었다. 이용 허락 표시(CC 등)와 재배포 조건은 누리집, 사용 안내서, README.txt, DIF 메타 "
            "기록(Use_Constraints, Access_Constraints 요소 없음)에서 찾지 못했다. NSIDC 자료 정책 쪽은 자료 제공자가 "
            "조건을 둘 수 있다고만 적었다. 재배포 가능 여부를 확인하지 못했으므로 점 자료를 커밋하지 않는다"
            "(data/processed/ext_labels/.gitignore 에 이미 들어 있다)"),
        "access_note": ("익명 FTP 다. 계정이 필요 없다(서버 응답 'Only anonymous FTP is allowed here', "
                        "'Anonymous user logged in')"),
        "citation": CITATION,
        "citation_auxiliary": ("Minkin, M.A. 1998. Borehole and environmental protection descriptive and numerical "
                               "data, Yamal Peninsula, Russia. In: International Permafrost Association, Data and "
                               "Information Working Group, comp. Circumpolar Active-Layer Permafrost System (CAPS), "
                               "version 1.0. CD-ROM. Boulder, Colorado: NSIDC, University of Colorado at Boulder. "
                               "(사용 안내서 끝의 인용 표기)"),
        "orig_source": ORIG_SOURCE,
        "parser": f"scripts/1_data_prep/{Path(__file__).name}",
        "parser_sha256": sha256_of(Path(__file__)),
        "parser_git_commit": f"미커밋(실행 시점 HEAD {git_head(ROOT)})",
        "points_file": f"data/processed/ext_labels/{points_path.name}",
        "points_sha256": sha256_of(points_path),
        "n_rows_raw": {"yabrhl.txt": len(txt), "yabrhl.dat": len(dat), "union_boreholes": len(order),
                       "yadesc.dat_regions": len(desc)},
        "n_rows_points": len(points),
        "n_sites": len(set(p["site_id"] for p in points)),
        "n_by_source_file": dict(n_in),
        "n_by_label_def": dict(Counter(p["label_def"] for p in points)),
        "n_by_method": dict(Counter(p["method"] for p in points)),
        "n_by_macro": dict(Counter(p["macro"] for p in points)),
        "n_by_subunit": dict(Counter(p["subunit"] for p in points)),
        "n_by_qc_flag": dict(flag_counts),
        "n_by_coord_prec_deg": {str(k): v for k, v in Counter(p["coord_prec_deg"] for p in points).items()},
        "n_by_year": {str(k): v for k, v in sorted(Counter(years).items())},
        "n_by_drill_month": {str(k): v for k, v in sorted(months.items())},
        "n_by_nature_protection_region": {str(k): v for k, v in sorted(Counter(p["_region"] for p in points).items())},
        "year_range": [min(years), max(years)] if years else None,
        "alt_cm_stats": stats_of(alts),
        "alt_cm_value_counts": {fmt_num(Decimal(str(k))): v for k, v in sorted(Counter(alts).items())},
        "n_right_censored": sum(p["right_censored"] for p in points),
        "n_excluded_by_reason": {
            "_설명": "점 자료에서 지운 행은 없다. 아래는 대상 라벨 규칙(6B.3, 6B.4)을 통과하지 못하는 행의 수다. "
                    "사유는 겹친다",
            "dropped_from_points": 0,
            "year_pre1990": flag_counts.get("year_pre1990", 0),
            "label_def_not_direct_eos": sum(1 for p in points if p["label_def"] != "direct_eos"),
            "qc_range": flag_counts.get("range", 0),
            "qc_coord": flag_counts.get("coord", 0),
            "qc_date": flag_counts.get("date", 0),
            "right_censored": sum(p["right_censored"] for p in points),
        },
        "n_target_rows": sum(1 for p in points if p["label_def"] == "direct_eos" and p["qc_flag"] == "ok"),
        "use": "확인적 풀과 대상 셀에 넣지 않는다. 서술과 보조 분석에만 쓴다(계획서 6B.2 보조 라벨 표)",
        "label_semantics": {
            "column_used": "Max depth seasonal freeze(thaw),m",
            "equal_to_permafrost_table_column": f"{n_eq_ptab} / {len(points)}",
            "n_thaw_at_coring_date_zero": n_coring_zero,
            "n_thaw_at_coring_gt_label": n_coring_gt,
            "ratio_thaw_at_coring_to_label_by_drill_month": by_month_ratio,
            "value_resolution_m": "0.05",
            "comment": ("라벨 칸은 'Depth of permafrost upper limit, m' 칸과 모든 시추공에서 같다. 시추일의 융해 깊이"
                        "('Depth of freeze (thaw)at a coring date') 는 3-5월에 0 이고 달이 늦을수록 라벨 값에 가까워진다. "
                        "따라서 라벨 값은 시추일의 관측값이 아니라 계절 최대 융해 깊이(영구동토 상한 깊이)로 적은 값이다. "
                        "코어의 동결 구조에서 읽은 값인지 계산값인지는 자료 설명에 없다"),
        },
        "region_range_check": {
            "_설명": "yadesc.dat 의 구역별 SEASONAL THAW DEPTH MIN, MAX 범위 안에 시추공의 라벨 값이 드는지 본 것이다. "
                    "표시만 하고 qc_flag 에는 쓰지 않았다",
            "n_inside": len(points) - len(outside_region_range),
            "n_outside": len(outside_region_range),
            "outside_site_ids": outside_region_range,
        },
        "duplicate_check": {
            "duplicate_site_id": 0,
            "in_both_files_merged_to_one_row": n_in.get("txt+dat", 0),
            "fields_differing_between_files": dict(diff_fields),
            "borehole_numbers_used_in_two_regions": bh_repeated,
            "coordinates_shared_by_two_boreholes": coord_shared,
            "rows_identical_in_coord_date_value": full_dup_groups,
        },
        "missing_value_marks": {
            "_설명": "빈 칸과 '-' 를 결측으로 보았다. 라벨, 좌표, 날짜 칸에는 결측이 없다",
            "n_missing_by_field": dict(missing_counts),
            "n_label_with_gt_or_lt_mark": n_gt_mark,
        },
        "coordinate_conversion": (
            "원자료는 DDMMSS 6자리 정수다(예: 685600 은 68도 56분 00초). 십진 도 = DD + MM/60 + SS/3600 으로 바꾸고 "
            "소수 6자리로 적었다. 반구 표기는 파일에 없다. 자료 누리집의 공간 범위(북위 66-72.5, 동경 65.5-74) 안에 "
            "전부 들어가므로 북위, 동경으로 두었다. 측지계 변환은 하지 않았다"),
        "coordinate_check": {
            "bbox_from_landing_page": BBOX,
            "lat_range": [min(p["_lat"] for p in points), max(p["_lat"] for p in points)],
            "lon_range": [min(p["_lon"] for p in points), max(p["_lon"] for p in points)],
            "n_outside_bbox": flag_counts.get("coord", 0),
            "n_minute_precision": sum(1 for p in points if p["_minute_only"]),
            "n_second_precision": sum(1 for p in points if not p["_minute_only"]),
        },
        "coord_prec_rule": (
            "_sources_plan.json 의 label_rule 은 coord_prec_deg 를 0.0167 하나로 적었다. 파일을 읽어 보니 124공은 "
            "초 자리가 위도와 경도 모두 00 이고 36공(구역 47, 72. yabrhl.txt 에만 있다)은 초 자리에 값이 있다. "
            "앞의 124공은 0.0167, 뒤의 36공은 0.0003(1초는 0.000278도)으로 적었다"),
        "units": "라벨 칸의 단위는 m 다. 100 을 곱해 cm 로 적었다",
        "cell_preview": {
            "_설명": "v4 조립 전의 참고 값이다. 이 자료원은 대상 셀에 넣지 않는다. 셀 색인은 계획서 6B.3 의 식, "
                    "블록은 0.5도다. 분 단위 좌표(위도 1분은 약 1.85 km)는 1 km 셀보다 거칠어 셀 배정이 확정적이지 않다",
            "cell_rule": "ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 도",
            "n_points": len(points),
            "n_cells_1km": len(cell_rows),
            "n_cells_dup_v3_f4_chebyshev_0p01": sum(1 for c in cell_rows if c["dup_v3_f4"]),
            "n_new_cells": len(new_cells),
            "n_new_cells_from_minute_precision_only": sum(1 for c in new_cells if c["minute_only"]),
            "n_blocks_0p5deg_new_cells": len(blocks_new),
            "n_blocks_not_in_v3_f4": len(blocks_new - v3_f4_blocks),
            "n_new_cells_within_0p01deg_of_v3_f2_or_calm_temp": sum(1 for c in new_cells if c["near_v3_other"]),
            "n_cells_if_0p01deg_grid": len(grid001),
            "alt_cm_cell_mean_stats": stats_of([round(c["alt"], 2) for c in new_cells]),
            "expected_in_plan": "새 셀 147개(블록 21개), 탐색 단계 추정",
        },
        "v3_check": {
            "bbox_compared": "60 <= lat <= 80, 50 <= lon <= 95",
            "n_ref_rows_compared": len(ref),
            "n_ref_f4_direct": len(f4),
            "min_distance_km_to_v3_f4": round(nearest[0], 2) if nearest else None,
            "nearest_v3_loc_id": nearest[1]["loc_id"] if nearest else None,
            "nearest_v3_region": nearest[1]["region"] if nearest else None,
            "nearest_v3_lat_lon": [nearest[1]["lat"], nearest[1]["lon"]] if nearest else None,
            "nearest_site_id": nearest[2] if nearest else None,
            "n_point_pairs_chebyshev_0p01": n_pt_cheb,
            **within,
            "min_distance_km_to_v3_other_source": round(nearest_other[0], 2) if nearest_other else None,
            "nearest_v3_other_loc_id": nearest_other[1]["loc_id"] if nearest_other else None,
            "nearest_v3_other_source_id": nearest_other[1]["source_id"] if nearest_other else None,
        },
        "other_ext_sources_nearby": {
            "_설명": "실행 시점에 같은 폴더에 있던 다른 자료원 점 자료 가운데 북위 66-73, 동경 65-75 안의 지점과의 "
                    "거리다. 자료원 사이의 중복 판정은 v4 조립에서 한다. dup_of 는 채우지 않았다(시추공은 다른 자료원의 "
                    "지점과 같은 지점으로 확인된 것이 없다)",
            "by_file": other_sources_nearby(args.out_dir, points),
        },
        "temporal_note": ("파일의 시추일은 1976-03 에서 1989-05 사이다. 자료 누리집과 DIF 기록의 시간 범위는 "
                          "1977-01-01 에서 1990-12-31 이다. 두 표기가 다르다"),
        "field_choices": {
            "date, month": "시추일. 라벨 값의 관측일이 아니다",
            "value_kind": "빈 칸. 한 해의 값인지 여러 해의 대표값인지 확인하지 못했다",
            "disturbed": "빈 칸. 시추 위치의 교란 여부는 자료에 없다. Topography 값은 notes 에 옮겼다",
            "n_obs": "1(시추공 하나)",
            "dup_of": "빈 칸",
        },
        "unverified_items": [
            "이용 약관과 재배포 조건. 인용 조건만 확인했다",
            "'Max depth seasonal freeze(thaw),m' 값의 산정 방법(코어의 동결 구조 관찰인지, 계산값인지, 다른 시기의 측정인지)",
            "라벨 값이 시추 연도의 값인지 여러 해를 대표하는 값인지",
            "좌표의 측지계. 파일과 사용 안내서에 없다. 조사 시기와 기관으로 보아 WGS84 가 아닐 수 있다. 변환하지 않았다",
            "초 자리가 00 인 124공의 좌표가 반올림인지 버림인지",
            "시추 위치가 시설 부지나 교란지인지. 자료는 가스전과 관로 노선의 조사 시추라고만 적었다",
            "Topography 의 ravine, gully, closed thermokarst bog, deflation 이 계획서 6B.3 의 교란 규칙(열침식 지형, 수체)에 "
            "해당하는지",
            "사용 안내서가 말한 색인 지도(.gif)는 FTP 폴더에 없다. 구역 경계를 확인하지 못했다",
            "사용 안내서가 말한 두 번째 표(암상 단면 유형의 층별 설명)는 FTP 폴더에 없다",
            "yabrhl.dat 에만 있는 2공과 yabrhl.txt 에만 있는 36공이 다른 파일에서 빠진 까닭",
            "자료 누리집의 시간 범위(1977-1990)와 파일의 시추일(1976-1989)이 다른 까닭",
        ],
    }
    meta_path = args.out_dir / f"{SRC_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"원자료: txt {len(txt)}공, dat {len(dat)}공, 합집합 {len(order)}공 {dict(n_in)}")
    print(f"점 자료: {len(points)}행 -> {points_path}")
    print(f"연도 {meta['year_range']}, ALT {meta['alt_cm_stats']}")
    print(f"qc_flag {dict(flag_counts)}, 좌표 정밀도 {meta['n_by_coord_prec_deg']}")
    print(f"셀 {len(cell_rows)}개, v3 F4 중복 {meta['cell_preview']['n_cells_dup_v3_f4_chebyshev_0p01']}개, "
          f"새 셀 {len(new_cells)}개, 블록 {len(blocks_new)}개")
    print(f"v3 F4 최근접 {meta['v3_check']['min_distance_km_to_v3_f4']} km "
          f"({meta['v3_check']['nearest_v3_region']} {meta['v3_check']['nearest_v3_loc_id']})")
    print(f"메타 -> {meta_path}")


if __name__ == "__main__":
    main()
