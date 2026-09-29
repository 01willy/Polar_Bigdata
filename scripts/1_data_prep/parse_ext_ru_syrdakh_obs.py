#!/usr/bin/env python3
"""ext_labels 파서: ru_syrdakh_obs.

자료: Pohl 등 (2026), Thermo-hydrological observatory in a permafrost river valley landscape
in Syrdakh, Central Yakutia. Zenodo, https://doi.org/10.5281/zenodo.19890671
(모든 판의 DOI 10.5281/zenodo.14619854, CC BY 4.0).
자료 논문: Pohl 등 (2026), Earth Syst. Sci. Data 18, 3525-3557,
https://doi.org/10.5194/essd-18-3525-2026.

입력(data/raw/ru_syrdakh_obs/)
  Syrdakh-DB_without-imagery-dsm.zip   압축 파일을 풀지 않고 아래 항목만 읽는다
    ground/td/td_*.csv                     융해 깊이 7개 파일(연 1회 방문, 2012-2018)
    auxiliary/gps/coordinates_*.csv        지점 좌표(UTM 52N 의 x, y 와 WGS84 의 lon, lat, label)
    auxiliary/geopackage/*/*.gpkg          지점 도형과 날짜 속성(좌표 대조용)
출력
  data/processed/ext_labels/ru_syrdakh_obs_points.csv   표준 점 자료(형식 1.0)
  data/processed/ext_labels/ru_syrdakh_obs_meta.json    메타
  data/raw/ru_syrdakh_obs/ru_syrdakh_obs_visits.csv     기록 단위 표(원자료 기록 전부와 처리 상태)

규칙(계획서 6B.3, _schema.json 1.0, 자료원 계획의 label_rule)
  1. 한 행은 (site_id, year) 이다. site_id 는 td 파일의 point 값이다. 방문은 해마다 1회다.
  2. alt_cm 은 value 열이다. 단위는 파일 머리말의 unit(cm)이고 변환하지 않는다.
     결측 부호는 머리말의 na_value(NA)다. 값이 NA 인 기록은 점 자료에 넣지 않고 수를 센다.
  3. method: PVC-TUBE(관 속 탐침)와 HANDBAR(금속봉)는 probe, DRILL(굴착)과 PIT(구덩이)은 pit_core.
  4. max_exceed = 1 이면 값은 하한이고 right_censored = 1 이다. 같은 지점과 같은 방문에
     하한 기록과 하한이 아닌 기록이 함께 있으면 행은 right_censored = 1 로 두고 alt_cm 에
     하한 값을 넣는다(하한이 아닌 값은 notes 에 적는다).
  5. 같은 지점과 같은 방문의 기록이 여럿이면 alt_cm 은 그 평균, n_obs 는 기록 수,
     alt_sd_cm 은 표본 표준편차(n - 1)다.
  6. 방문 월이 8월 또는 9월이면 label_def = direct_eos, eos_basis = record_date.
     10월 방문(2012-10-07)은 label_def = direct_dated, eos_basis = none.
  7. 같은 파일 안에서 (value, max_exceed, method) 의 연속 4개 이상이 앞선 구간과 같은 순서로
     되풀이되면 뒤 구간을 반복 입력으로 본다. 지점 이름까지 같으면 정확한 중복이고 점 자료에
     넣지 않는다. 지점 이름이 다르면 위치가 다시 지정된 중복 의심 기록이다. 지우지 않고
     dup_of 에 앞선 기록의 지점을 적고 qc_flag 를 coord 로 둔다.
  8. 수체 아래 지점: 자료 저자의 그림 스크립트(td 2026-04-28.R)가 정한 하천 띠
     (기준점 scale-CS9, scale-CS2 에서 남북 부호를 붙인 거리 -3 m 에서 3 m)에 드는 지점이면
     disturbed = 1, disturb_type = water. 자료와 논문은 수체 아래 지점이나 탈릭 지점을
     따로 표시하지 않았다.
  9. PIT 기록의 구덩이 날짜(geopackage)와 파일 날짜의 연도가 다르면 qc_flag 에 date 를 붙인다.
     DRILL, HANDBAR 기록의 지점 이름에 든 연도는 위치를 측량한 해라 notes 에만 적는다.
 10. 좌표는 자료의 coordinates CSV 의 lon, lat 을 자릿수 그대로 쓴다. td 파일의 hobo-F<i> 는
     좌표 표의 HOBO-F<i> 와 대소문자만 다르므로 대소문자를 무시하고 맞춘다.

실행: 스레드 1개. 입력은 90 MB 압축 파일 안의 작은 표 몇 개다.
  OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_ru_syrdakh_obs.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import statistics
import struct
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SRC_ID = "ru_syrdakh_obs"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

ZIP_NAME = "Syrdakh-DB_without-imagery-dsm.zip"
RECORD_JSON = "zenodo_record_19890671.json"
ARTICLE_HTML = "essd-18-3525-2026_article.html"
ACCESSED = "2026-09-29"

URL = "https://doi.org/10.5281/zenodo.14619854"
DOI_VERSION = "10.5281/zenodo.19890671"
DOI_CONCEPT = "10.5281/zenodo.14619854"
FILE_URLS = {
    ZIP_NAME: "https://zenodo.org/api/records/19890671/files/Syrdakh-DB_without-imagery-dsm.zip/content",
    RECORD_JSON: "https://zenodo.org/api/records/19890671",
    ARTICLE_HTML: "https://essd.copernicus.org/articles/18/3525/2026/",
}
ZENODO_MD5 = {ZIP_NAME: "d1d193691d22009e1347c8944828720f"}

LICENSE = "CC-BY-4.0"
CITATION_ROW = (
    "Pohl, E. et al. (2026): Thermo-hydrological observatory in a permafrost river valley landscape in "
    "Syrdakh, Central Yakutia [dataset]. Zenodo, https://doi.org/10.5281/zenodo.19890671; "
    "Pohl, E. et al. (2026), Earth Syst. Sci. Data 18, 3525-3557, https://doi.org/10.5194/essd-18-3525-2026"
)
CITATION_FULL = (
    "Pohl, E., Grenier, C., Séjourné, A., Bouchard, F., Léger, E., Saintenoy, A., Konstantinov, P., "
    "Cuynet, A., Ottlé, C., Hatté, C., Noret, A., Danilov, K., Bazhin, K., Khristoforov, I., Fortier, D., "
    "Fedorov, A., and Mouche, E. (2026): Thermo-hydrological observatory in a permafrost river valley "
    "landscape in Syrdakh, Central Yakutia [dataset]. Zenodo, https://doi.org/10.5281/zenodo.19890671 "
    "(all versions: https://doi.org/10.5281/zenodo.14619854). Data description article: Pohl, E. et al.: "
    "Thermo-hydrological river valley observatory in Yedoma permafrost from 2012 through 2022 in Syrdakh, "
    "Central Yakutia, Earth Syst. Sci. Data, 18, 3525-3557, https://doi.org/10.5194/essd-18-3525-2026, 2026."
)

COLUMNS = [
    "src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
    "n_obs", "country", "macro", "citation", "license",
    "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
    "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes",
]

METHOD_MAP = {"PVC-TUBE": "probe", "PVC_TUBE": "probe", "HANDBAR": "probe", "DRILL": "pit_core", "PIT": "pit_core"}
# 자료 논문 4.1.3 절이 가정한 방법별 최대 불확도(cm). notes 에 적는다.
METHOD_ERR_CM = {"PVC-TUBE": 15, "PVC_TUBE": 15, "HANDBAR": 15, "DRILL": 25, "PIT": 2}

EOS_MONTHS = (8, 9)
RANGE_MAX_CM = 600.0
REPEAT_MIN_RUN = 4          # 반복 입력으로 보는 최소 연속 길이
RIVER_BAND_M = 3.0          # 자료 저자 스크립트의 river_coords = c(-3, 3)
SITE_REF = {"Site2": ("scale-CS9", "CS-9"), "Site1": ("scale-CS2", "CS-2")}
COORD_PREC_DEG = 0.00001    # 자료 설명의 dGPS 정확도 0.5 m 에 해당하는 자릿수


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def cell_index(lat: float, lon: float):
    """계획서 6B.3 의 약 1 km 셀 색인."""
    ky = math.floor(lat / 0.009)
    phi = math.radians((ky + 0.5) * 0.009)
    kx = math.floor(lon * math.cos(phi) / 0.009)
    return ky, kx


def block_index(lat: float, lon: float) -> int:
    return math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5)


def fmt_num(v: float) -> str:
    """값을 짧게 적는다. 정수면 정수로, 아니면 소수 4자리까지."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- 입력 읽기
def read_td_file(zf: zipfile.ZipFile, member: str):
    """td 파일 하나를 읽는다. 머리말(키, 값)과 기록 목록을 돌려준다."""
    text = zf.read(member).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    start = next(i for i, r in enumerate(rows) if r and r[0].strip() == "point")
    header = {}
    for r in rows[:start]:
        if r and r[0] and not r[0].startswith("#"):
            header[r[0].strip()] = r[1].strip() if len(r) > 1 else ""
    cols = [c.strip() for c in rows[start]]
    assert cols[:4] == ["point", "value", "max_exceed", "method"], (member, cols)
    unit = header.get("unit", "")
    if unit == "cm":
        factor = 1.0
    elif unit == "m":
        factor = 100.0
    else:
        raise ValueError(f"{member}: 단위를 알 수 없다({unit!r})")
    t0, t1 = header.get("time_start", ""), header.get("time_end", "")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", t0) and t0 == t1, (member, t0, t1)
    name_date = re.search(r"_(\d{8})-mean\.csv$", member).group(1)
    assert name_date == t0.replace("-", ""), (member, t0)
    na_tokens = {s.strip() for s in header.get("na_value", "NA").split(",")} | {""}
    recs = []
    for k, r in enumerate(rows[start + 1:], start=1):
        if not r or not r[0].strip():
            continue
        point, value_raw, exc, meth = (x.strip() for x in r[:4])
        assert exc in ("0", "1"), (member, r)
        assert meth in METHOD_MAP, (member, r)
        value = None if value_raw in na_tokens else float(value_raw) * factor
        recs.append({
            "file": member, "row_in_file": k, "date": t0, "year": int(t0[:4]), "month": int(t0[5:7]),
            "point": point, "value_raw": value_raw, "unit": unit, "value_cm": value,
            "max_exceed": int(exc), "method_raw": meth, "method": METHOD_MAP[meth],
            "status": "used", "repeat_of_row": "",
        })
    return header, recs


def read_coordinates(zf: zipfile.ZipFile):
    member = sorted(n for n in zf.namelist() if re.fullmatch(r"auxiliary/gps/coordinates_.*\.csv", n))[-1]
    text = zf.read(member).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    return member, rows, by_label


def read_gpkg_points(zf: zipfile.ZipFile):
    """geopackage 의 점 도형(x, y)과 날짜 속성을 읽는다. 좌표 대조와 notes 용이다."""
    out = {}
    members = sorted(n for n in zf.namelist() if n.endswith("_point.gpkg"))
    with tempfile.TemporaryDirectory() as td:
        for m in members:
            p = Path(td) / Path(m).name
            p.write_bytes(zf.read(m))
            con = sqlite3.connect(str(p))
            try:
                table, srs = con.execute(
                    "select table_name, srs_id from gpkg_contents where data_type='features'").fetchone()
                gcol = con.execute(
                    "select column_name from gpkg_geometry_columns where table_name=?", (table,)).fetchone()[0]
                for label, d0, d1, exact, blob in con.execute(
                        f'select label, date_start, date_end, exact_date, "{gcol}" from "{table}"'):
                    if blob is None or blob[:2] != b"GP":
                        continue
                    flags = blob[3]
                    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
                    wkb = blob[8 + env_len:]
                    bo = "<" if wkb[0] == 1 else ">"
                    gtype = struct.unpack(bo + "I", wkb[1:5])[0]
                    if gtype % 1000 != 1:
                        continue
                    x, y = struct.unpack(bo + "dd", wkb[5:21])
                    out.setdefault(label, []).append({
                        "gpkg": m, "srs_id": srs, "x": x, "y": y,
                        "date_start": d0, "date_end": d1, "exact_date": exact})
            finally:
                con.close()
    return members, out


def read_river_transects(zf: zipfile.ZipFile):
    """water/levl 의 단면 수심 표(scale-CS9, scale-CS2)에서 수면 폭을 읽는다.

    표의 point 는 단면을 따라 잰 거리(cm), value 는 수심(cm)이다. 양 끝의 수심이 0 이므로
    마지막 point 를 수면 폭으로 본다. 기준점(scale-CS*)이 단면의 어디에 있는지는 표에 없다.
    """
    out = {}
    for n in sorted(zf.namelist()):
        m = re.fullmatch(r"water/levl/wl_Wtest_syrdakh_(scale-CS[29])_L1_(\d{4})-avg\.csv", n)
        if not m:
            continue
        rows = list(csv.reader(io.StringIO(zf.read(n).decode("utf-8-sig"))))
        start = next(i for i, r in enumerate(rows) if r and r[0].strip() == "point")
        pts = [(float(r[0]), float(r[1])) for r in rows[start + 1:] if r and r[0].strip()]
        out.setdefault(m.group(1), []).append({
            "file": n, "year": int(m.group(2)), "width_m": round(pts[-1][0] / 100.0, 2),
            "max_depth_cm": max(v for _, v in pts)})
    return out


# ---------------------------------------------------------------- 반복 입력 찾기
def mark_repeats(recs):
    """같은 파일 안의 반복 구간과 정확한 중복을 표시한다."""
    key = [(r["value_raw"], r["max_exceed"], r["method_raw"]) for r in recs]
    n = len(recs)
    blocks = []
    taken = set()
    for j in range(n):
        if j in taken:
            continue
        best = None
        for i in range(j):
            if i in taken:
                continue
            length = 0
            while j + length < n and i + length < j and key[i + length] == key[j + length]:
                length += 1
            if length >= REPEAT_MIN_RUN and (best is None or length > best[1]):
                best = (i, length)
        if best:
            i, length = best
            blocks.append({"first_row": recs[i]["row_in_file"], "repeat_row": recs[j]["row_in_file"],
                           "length": length})
            for k in range(length):
                a, b = recs[i + k], recs[j + k]
                taken.add(j + k)
                b["repeat_of_row"] = a["row_in_file"]
                if b["point"] == a["point"]:
                    b["status"] = "dup_exact"
                else:
                    b["status"] = "dup_position"
                    b["dup_point"] = a["point"]
    # 반복 구간 밖의 정확한 중복(지점, 값, 하한 표시, 방법이 모두 같다)
    seen = {}
    for r in recs:
        k4 = (r["point"], r["value_raw"], r["max_exceed"], r["method_raw"])
        if r["status"] == "used":
            if k4 in seen:
                r["status"] = "dup_exact"
                r["repeat_of_row"] = seen[k4]
            else:
                seen[k4] = r["row_in_file"]
    return blocks


# ---------------------------------------------------------------- v3 대조
def v3_check(sites):
    """v3 의 F4_direct 셀과의 체비쇼프 거리. lat, lon, source_id 만 읽는다."""
    if not V3_PATH.exists():
        return {"checked": False}
    best = None
    n_within = 0
    with open(V3_PATH, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        v3 = [(float(r["lat"]), float(r["lon"]), r["loc_id"], r["region"]) for r in rd
              if r["source_id"] == "F4_direct"]
    km_by_region = {}
    for s in sites:
        dmin = None
        p1 = math.radians(s["lat_f"])
        for la, lo, lid, reg in v3:
            d = max(abs(la - s["lat_f"]), abs(lo - s["lon_f"]))
            if dmin is None or d < dmin[0]:
                dmin = (d, lid, reg)
            # 대권 거리(km, 구 반지름 6371 km). 계획서 6B.4 의 독립성 규칙(다른 macro 의 셀에서 100 km)을 보는 데 쓴다
            p2 = math.radians(la)
            h = (math.sin((p2 - p1) / 2) ** 2
                 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo - s["lon_f"]) / 2) ** 2)
            dk = 2 * 6371.0 * math.asin(math.sqrt(h))
            if reg not in km_by_region or dk < km_by_region[reg][0]:
                km_by_region[reg] = (dk, lid)
        if dmin[0] <= 0.01:
            n_within += 1
        if best is None or dmin[0] < best[0]:
            best = dmin
    nearest_km = [{"v3_region": reg, "min_km": round(d, 1), "v3_loc_id": lid}
                  for reg, (d, lid) in sorted(km_by_region.items(), key=lambda kv: kv[1][0])[:4]]
    return {"checked": True, "n_v3_f4_direct_cells": len(v3), "n_sites_within_chebyshev_0p01deg": n_within,
            "min_chebyshev_deg": round(best[0], 4), "nearest_v3_loc_id": best[1], "nearest_v3_region": best[2],
            "nearest_km_by_v3_region": nearest_km,
            "nearest_km_note": "지점 85개 가운데 가장 가까운 지점에서 잰 v3 F4_direct 셀까지의 대권 거리다. "
                               "CALM_Russia_C 는 같은 macro(Russia_C)다. 다른 macro 의 가장 가까운 셀은 "
                               "Lena_RU 이고 100 km 규칙에 걸리지 않는다"}


# ---------------------------------------------------------------- 본문
def main():
    zip_path = RAW_DIR / ZIP_NAME
    assert zip_path.exists(), f"원자료가 없다: {zip_path}"
    zip_md5 = md5_of(zip_path)
    assert zip_md5 == ZENODO_MD5[ZIP_NAME], "Zenodo 의 md5 와 다르다"

    with zipfile.ZipFile(zip_path) as zf:
        td_members = sorted(n for n in zf.namelist() if re.fullmatch(r"ground/td/td_.*\.csv", n))
        assert len(td_members) == 7, td_members
        headers, recs_by_file = {}, {}
        for m in td_members:
            headers[m], recs_by_file[m] = read_td_file(zf, m)
        coord_member, coord_rows, coord_by_label = read_coordinates(zf)
        gpkg_members, gpkg_pts = read_gpkg_points(zf)
        river_tr = read_river_transects(zf)
        member_info = {m: {"size_bytes": zf.getinfo(m).file_size,
                           "sha256": hashlib.sha256(zf.read(m)).hexdigest()}
                       for m in td_members + [coord_member]}

    # 방문은 해마다 1회인지 확인한다
    years = [recs_by_file[m][0]["year"] for m in td_members]
    assert len(set(years)) == len(years), "한 해에 파일이 둘 이상이다"

    # 반복 입력 표시
    repeat_blocks = {}
    for m in td_members:
        b = mark_repeats(recs_by_file[m])
        if b:
            repeat_blocks[m] = b
    for m in td_members:
        for r in recs_by_file[m]:
            if r["value_cm"] is None:
                r["status"] = "na_value" if r["status"] == "used" else r["status"] + "+na_value"

    # 좌표 붙이기
    lower_map = defaultdict(list)
    for lab in coord_by_label:
        lower_map[lab.lower()].append(lab)
    ref = {}
    for site, (lab, _cs) in SITE_REF.items():
        rr = coord_by_label[lab]
        assert len(rr) == 1
        ref[site] = (float(rr[0]["x"]), float(rr[0]["y"]))

    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
    except Exception:
        tr = None

    site_info = {}
    coord_match = Counter()
    max_dev_proj = 0.0
    max_dev_gpkg = 0.0
    n_gpkg_checked = 0
    all_points = sorted({r["point"] for m in td_members for r in recs_by_file[m]})
    for pt in all_points:
        if pt in coord_by_label:
            lab, how = pt, "exact"
        elif len(lower_map.get(pt.lower(), [])) == 1:
            lab, how = lower_map[pt.lower()][0], "case_insensitive"
        else:
            lab, how = None, "missing"
        coord_match[how] += 1
        if lab is None:
            site_info[pt] = None
            continue
        rows = coord_by_label[lab]
        assert len(rows) == 1, f"좌표 표에 같은 이름이 여럿이다: {lab}"
        c = rows[0]
        x, y = float(c["x"]), float(c["y"])
        lon_f, lat_f = float(c["lon"]), float(c["lat"])
        if tr is not None:
            lo2, la2 = tr.transform(x, y)
            max_dev_proj = max(max_dev_proj, abs(lo2 - lon_f), abs(la2 - lat_f))
        g = gpkg_pts.get(lab, [])
        if len(g) == 1:
            n_gpkg_checked += 1
            assert g[0]["srs_id"] == 32652
            max_dev_gpkg = max(max_dev_gpkg, abs(g[0]["x"] - x), abs(g[0]["y"] - y))
        dist = {s: math.hypot(x - rx, y - ry) for s, (rx, ry) in ref.items()}
        site = min(dist, key=dist.get)
        sign = 1.0 if y - ref[site][1] >= 0 else -1.0
        prof = sign * dist[site]          # 자료 저자 스크립트의 calc_dist_wdir 과 같은 정의
        site_info[pt] = {
            "coord_label": lab, "coord_match": how, "lat": c["lat"], "lon": c["lon"],
            "lat_f": lat_f, "lon_f": lon_f, "x": x, "y": y, "site": site, "cs": SITE_REF[site][1],
            "dist_ref_m": dist[site], "profile_m": prof,
            "gpkg_date_start": g[0]["date_start"] if len(g) == 1 else "",
            "gpkg_exact_date": g[0]["exact_date"] if len(g) == 1 else "",
        }
    assert coord_match["missing"] == 0, "좌표가 없는 지점이 있다"

    # 기록 단위 표(원자료 디렉터리)
    all_recs = [r for m in td_members for r in recs_by_file[m]]
    visits_path = RAW_DIR / f"{SRC_ID}_visits.csv"
    vcols = ["file", "row_in_file", "date", "point", "value_raw", "unit", "value_cm", "max_exceed",
             "method_raw", "method", "status", "repeat_of_row", "coord_label", "coord_match", "lat", "lon",
             "x_utm52n", "y_utm52n", "site", "profile_dist_m", "gpkg_date_start", "gpkg_exact_date"]
    with open(visits_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(vcols)
        for r in all_recs:
            s = site_info[r["point"]]
            w.writerow([r["file"], r["row_in_file"], r["date"], r["point"], r["value_raw"], r["unit"],
                        "" if r["value_cm"] is None else fmt_num(r["value_cm"]), r["max_exceed"],
                        r["method_raw"], r["method"], r["status"], r["repeat_of_row"],
                        s["coord_label"], s["coord_match"], s["lat"], s["lon"],
                        f"{s['x']:.3f}", f"{s['y']:.3f}", s["site"], f"{s['profile_m']:.2f}",
                        s["gpkg_date_start"], s["gpkg_exact_date"]])

    # 행 만들기
    groups = defaultdict(list)
    for r in all_recs:
        if r["value_cm"] is None or r["status"] == "dup_exact" or "na_value" in r["status"]:
            continue
        groups[(r["point"], r["year"], r["status"] == "dup_position")].append(r)
    keys_sy = Counter((k[0], k[1]) for k in groups)
    assert max(keys_sy.values()) == 1, "같은 (site_id, year) 에 중복 의심 기록과 일반 기록이 함께 있다"

    points = []
    for (pt, year, is_dup), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s = site_info[pt]
        cen = [r for r in rs if r["max_exceed"] == 1]
        unc = [r for r in rs if r["max_exceed"] == 0]
        notes = [f"site={s['site']}({s['cs']})", f"dist_river_ref_m={s['profile_m']:.1f}"]
        if cen:
            top = max(r["value_cm"] for r in cen)
            used = [r for r in cen if r["value_cm"] == top]
            alt, n_obs, sd, rc = top, len(used), "", 1
            notes.append("값은 하한이다(max_exceed=1)")
            if unc:
                notes.append("같은 방문의 하한이 아닌 기록은 값에 쓰지 않았다: " + "; ".join(
                    f"{r['method_raw']} {fmt_num(r['value_cm'])} cm" for r in unc))
        else:
            used = unc
            vals = [r["value_cm"] for r in used]
            alt, n_obs, rc = sum(vals) / len(vals), len(vals), 0
            sd = statistics.stdev(vals) if len(vals) >= 2 else ""
        mcount = Counter(r["method"] for r in used)
        if len(mcount) == 1:
            method = next(iter(mcount))
        else:
            top_n = max(mcount.values())
            cands = sorted(k for k, v in mcount.items() if v == top_n)
            method = "probe" if "probe" in cands else cands[0]
        raw_methods = sorted({r["method_raw"] for r in used})
        notes.append("method_raw=" + "+".join(raw_methods))
        notes.append("err_assumed_cm=" + "+".join(str(METHOD_ERR_CM[m]) for m in raw_methods))
        if len(used) >= 2:
            notes.append("같은 방문의 기록 " + str(len(used)) + "개 평균: " + "; ".join(
                f"{r['method_raw']} {fmt_num(r['value_cm'])}" for r in used))
        month = rs[0]["month"]
        if month in EOS_MONTHS:
            label_def, eos = "direct_eos", "record_date"
        else:
            label_def, eos = "direct_dated", "none"
        qc = []
        if not (0 < alt <= RANGE_MAX_CM):
            qc.append("range")
        if year < 1990:
            qc.append("year_pre1990")
        dup_of = ""
        if is_dup:
            dup_pts = sorted({r["dup_point"] for r in rs})
            assert len(dup_pts) == 1
            dup_of = f"{SRC_ID}:{dup_pts[0]}"
            qc.append("coord")
            notes.append("같은 파일의 앞선 구간과 값, 방법, 순서가 같은 반복 구간의 기록이다. "
                         "지점 이름만 다르다. 어느 위치가 맞는지 확인하지 못했다")
        multi_val = len({r["value_cm"] for r in used}) >= 2 and len({r["method_raw"] for r in used}) == 1
        if multi_val and not is_dup:
            qc.append("coord")
            notes.append("같은 방법의 서로 다른 값이 한 지점 이름에 붙어 있다. 위치 지정을 확인하지 못했다")
        label_year = re.match(r"(?:drill|pit)-(\d{4})-", pt)
        if label_year and int(label_year.group(1)) != year:
            pit_used = any(r["method_raw"] == "PIT" for r in used)
            gd = s["gpkg_date_start"]
            if pit_used and gd and int(gd[:4]) != year:
                # 구덩이 값은 구덩이 자체에서 잰 값이다. 구덩이 날짜와 파일 날짜의 연도가 다르면 날짜 의심이다
                qc.append("date")
                notes.append(f"파일의 관측일은 {rs[0]['date']} 이고 geopackage 의 구덩이 날짜는 {gd}"
                             f"(exact_date={s['gpkg_exact_date']})다. 어느 쪽이 관측일인지 확인하지 못했다")
            else:
                notes.append(f"지점 이름의 연도({label_year.group(1)})는 위치를 측량한 해이고 관측 연도와 다르다")
        if s["coord_match"] == "case_insensitive":
            notes.append(f"좌표 표의 이름은 {s['coord_label']}")
        in_band = abs(s["profile_m"]) <= RIVER_BAND_M
        disturbed, dtype = (1, "water") if in_band else (0, "")
        if pt == "hobo-F2" and year == 2012:
            notes.append("같은 지점의 2013년 값 166 cm 의 2.2배다. 탈릭 여부는 자료에서 확인하지 못했다")
        points.append({
            "src_id": SRC_ID, "site_id": pt,
            "site_name": f"Syrdakh {s['site']} ({s['cs']}) {pt}",
            "lat": s["lat"], "lon": s["lon"], "year": year, "month": month,
            "alt_cm": fmt_num(alt), "method": method, "label_def": label_def, "n_obs": n_obs,
            "country": "Russia", "macro": "Russia_C", "citation": CITATION_ROW, "license": LICENSE,
            "subunit": "Yakutia_C", "date": rs[0]["date"], "eos_basis": eos, "value_kind": "single_visit",
            "year_min": "", "year_max": "", "alt_sd_cm": "" if sd == "" else fmt_num(sd),
            "coord_prec_deg": f"{COORD_PREC_DEG:.5f}", "disturbed": disturbed, "disturb_type": dtype,
            "right_censored": rc, "orig_source": rs[0]["file"], "dup_of": dup_of,
            "qc_flag": ";".join(qc) if qc else "ok", "notes": "; ".join(notes),
            "_alt": alt, "_lat": s["lat_f"], "_lon": s["lon_f"],
        })

    # 형식 확인
    assert len({(p["site_id"], p["year"]) for p in points}) == len(points)
    for p in points:
        assert 50 < p["_lat"] < 75 and 100 < p["_lon"] < 140

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points_path = OUT_DIR / f"{SRC_ID}_points.csv"
    with open(points_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in points:
            w.writerow(p)

    # ------------------------------------------------------------ 요약
    def stats(rows):
        v = sorted(p["_alt"] for p in rows)
        if not v:
            return {"n": 0}
        return {"n": len(v), "min": round(v[0], 2), "median": round(statistics.median(v), 2),
                "max": round(v[-1], 2), "mean": round(sum(v) / len(v), 2)}

    def cells_blocks(rows):
        return (len({cell_index(p["_lat"], p["_lon"]) for p in rows}),
                len({block_index(p["_lat"], p["_lon"]) for p in rows}))

    def is_target(p, ignore_dist=False):
        return (p["label_def"] == "direct_eos" and p["right_censored"] == 0 and p["qc_flag"] == "ok"
                and p["dup_of"] == "" and p["year"] >= 1990 and (ignore_dist or p["disturbed"] == 0))

    tgt = [p for p in points if is_target(p)]
    status_count = Counter(r["status"] for r in all_recs)
    sites_all = {p["site_id"] for p in points}
    sites_tgt = {p["site_id"] for p in tgt}

    # 셀 미리 보기: 위치별 다년 평균의 평균
    cell_rows = defaultdict(lambda: defaultdict(list))
    for p in tgt:
        cell_rows[cell_index(p["_lat"], p["_lon"])][p["site_id"]].append(p)
    cell_table = []
    for ck, by_site in sorted(cell_rows.items()):
        site_means = [sum(q["_alt"] for q in v) / len(v) for v in by_site.values()]
        yrs = sorted({q["year"] for v in by_site.values() for q in v})
        lat_m = sum(v[0]["_lat"] for v in by_site.values()) / len(by_site)
        lon_m = sum(v[0]["_lon"] for v in by_site.values()) / len(by_site)
        cell_table.append({
            "ky": ck[0], "kx": ck[1], "lat_mean": round(lat_m, 6), "lon_mean": round(lon_m, 6),
            "n_sites": len(by_site), "n_site_years": sum(len(v) for v in by_site.values()),
            "years": yrs, "alt_cm_cell": round(sum(site_means) / len(site_means), 2),
            "alt_cm_site_min": round(min(site_means), 2), "alt_cm_site_max": round(max(site_means), 2),
            "sites_by_group": dict(Counter(v[0]["site_name"].split(" ")[1] for v in by_site.values())),
        })

    v3 = v3_check([site_info[pt] for pt in sorted(sites_all)])

    # 하천 기준점 5 m 이내 지점의 대상 행 값과 Site 2 대상 행의 중앙값
    near_pts = {pt for pt in all_points if abs(site_info[pt]["profile_m"]) <= 5.0}
    near_vals = sorted(p["_alt"] for p in tgt if p["site_id"] in near_pts)
    s2_med = statistics.median(p["_alt"] for p in tgt if site_info[p["site_id"]]["site"] == "Site2")
    near_dists = sorted(abs(site_info[pt]["profile_m"]) for pt in near_pts if pt in sites_all)

    files_meta = []
    for name in (ZIP_NAME, RECORD_JSON, ARTICLE_HTML):
        p = RAW_DIR / name
        if p.exists():
            d = {"name": name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p), "url": FILE_URLS[name]}
            if name in ZENODO_MD5:
                d["md5"] = zip_md5
                d["md5_matches_zenodo"] = True
            files_meta.append(d)
    for m in td_members + [coord_member]:
        files_meta.append({"name": f"{ZIP_NAME}::{m}", **member_info[m]})

    parser_path = Path(__file__).resolve()
    meta = {
        "src_id": SRC_ID,
        "name": "Syrdakh 열·수문 관측소 융해 깊이 2012-2018(Zenodo 19890671)",
        "schema_version": "1.0",
        "url": URL,
        "doi": DOI_VERSION,
        "doi_all_versions": DOI_CONCEPT,
        "doi_article": "10.5194/essd-18-3525-2026",
        "accessed": ACCESSED,
        "files": files_meta,
        "files_not_downloaded": [
            {"name": "Syrdakh-DB_dsm.zip", "size_bytes": 2629534640, "reason": "수치 표면 모형. 라벨에 쓰지 않는다"},
            {"name": "Syrdakh-DB_imagery.zip", "size_bytes": 3486939641, "reason": "정사 영상. 라벨에 쓰지 않는다"},
        ],
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0). Zenodo 기록의 license.id 가 "
                   "cc-by-4.0 이고 자료 논문 5절이 'CC 4.0 license' 로 적었다",
        "citation": CITATION_FULL,
        "parser": "scripts/1_data_prep/parse_ext_ru_syrdakh_obs.py",
        "parser_sha256": sha256_of(parser_path),
        "parser_git_commit": git_head(),
        "parser_git_note": "파서는 실행 시점에 커밋되지 않은 새 파일이다. 위 값은 저장소 HEAD 다",
        "visits_table": f"data/raw/{SRC_ID}/{SRC_ID}_visits.csv",
        "n_rows_raw": len(all_recs),
        "n_rows_raw_by_file": {m: len(recs_by_file[m]) for m in td_members},
        "n_rows_raw_by_status": dict(status_count),
        "n_rows_points": len(points),
        "n_sites": len(sites_all),
        "n_sites_by_group": dict(sorted(Counter(site_info[pt]["site"] for pt in sites_all).items())),
        "n_by_label_def": dict(Counter(p["label_def"] for p in points)),
        "n_by_method": dict(Counter(p["method"] for p in points)),
        "n_raw_by_method_raw": dict(Counter(r["method_raw"] for r in all_recs)),
        "n_by_macro": dict(Counter(p["macro"] for p in points)),
        "n_by_qc_flag": dict(Counter(p["qc_flag"] for p in points)),
        "n_by_year": dict(sorted(Counter(p["year"] for p in points).items())),
        "n_by_date": dict(sorted(Counter(p["date"] for p in points).items())),
        "year_range": [min(p["year"] for p in points), max(p["year"] for p in points)],
        "n_right_censored": sum(p["right_censored"] for p in points),
        "n_disturbed": sum(p["disturbed"] for p in points),
        "n_dup_of": sum(1 for p in points if p["dup_of"]),
        "n_obs_total": sum(p["n_obs"] for p in points),
        "alt_cm_stats_all_rows": stats(points),
        "alt_cm_stats_target": stats(tgt),
        "alt_cm_stats_direct_dated": stats([p for p in points if p["label_def"] == "direct_dated"]),
        "n_excluded_by_reason": {
            "_설명": "dropped_* 는 점 자료에 넣지 않은 원자료 기록의 수다. 나머지는 점 자료에 남아 있으나 "
                    "대상 라벨 규칙(6B.3)을 통과하지 못하는 행의 수이고 사유는 겹칠 수 있다",
            "dropped_value_na": sum(1 for r in all_recs if "na_value" in r["status"]),
            "dropped_exact_duplicate": sum(1 for r in all_recs if r["status"] == "dup_exact"),
            "merged_into_site_visit_mean": sum(p["n_obs"] - 1 for p in points if p["right_censored"] == 0),
            "not_used_in_value_uncensored_beside_censored": sum(
                1 for (pt, y, d), rs in groups.items()
                if any(r["max_exceed"] == 1 for r in rs) for r in rs if r["max_exceed"] == 0),
            "label_def_direct_dated_october_visit": sum(1 for p in points if p["label_def"] == "direct_dated"),
            "right_censored": sum(p["right_censored"] for p in points),
            "dup_of_suspected_position_duplicate": sum(1 for p in points if p["dup_of"]),
            "qc_flag_not_ok": sum(1 for p in points if p["qc_flag"] != "ok"),
            "disturbed_water": sum(p["disturbed"] for p in points),
        },
        "n_target_rows": len(tgt),
        "n_target_sites": len(sites_tgt),
        "target_rule": "label_def = direct_eos, right_censored = 0, disturbed = 0, qc_flag = ok, dup_of 빈 칸, year >= 1990",
        "cell_preview": {
            "_설명": "v4 조립 전의 참고 값이다. 셀 색인은 6B.3 의 식, 블록은 0.5° 다. 셀 값은 셀 안 위치별 "
                    "다년 평균의 평균이다",
            "all_rows": dict(zip(["n_cells_1km", "n_blocks_0p5deg"], cells_blocks(points))),
            "target_rows": dict(zip(["n_cells_1km", "n_blocks_0p5deg"], cells_blocks(tgt))),
            "target_cells": cell_table,
        },
        "v3_overlap_check": v3,
        "repeat_blocks": repeat_blocks,
        "river_transects": {
            "_설명": "water/levl 의 단면 수심 표에서 읽은 수면 폭(m)과 최대 수심(cm)이다. 하천 기준점(scale-CS*)이 "
                    "단면의 어디에 있는지는 확인하지 못했다. near_ref_sites 는 기준점에서 5 m 이내인 융해 깊이 "
                    "지점과 부호 붙인 거리(m)다",
            "transects": river_tr,
            "near_ref_sites": sorted(
                [{"site_id": pt, "site": site_info[pt]["site"], "dist_river_ref_m": round(site_info[pt]["profile_m"], 2),
                  "in_points": pt in sites_all}
                 for pt in all_points if abs(site_info[pt]["profile_m"]) <= 5.0],
                key=lambda d: abs(d["dist_river_ref_m"])),
        },
        "coordinate_conversion": {
            "source": f"{ZIP_NAME}::{coord_member} 의 lon, lat 열(WGS84 십진 도)을 자릿수 그대로 썼다",
            "origin": "자료 저자가 geopackage(EPSG:32652, WGS 84 / UTM zone 52N)의 점 도형을 geopandas 로 "
                      "EPSG:4326 으로 바꾼 값이다(auxiliary/scripts/coordinates/get_coords_from_shapefiles.py)",
            "check_pyproj": ("pyproj 로 x, y 를 다시 변환해 비교했다" if tr is not None else "pyproj 가 없어 하지 못했다"),
            "max_abs_diff_deg_vs_pyproj": (float(f"{max_dev_proj:.3e}") if tr is not None else None),
            "check_gpkg": f"geopackage 점 도형의 x, y 와 좌표 표의 x, y 를 {n_gpkg_checked}개 지점에서 비교했다",
            "max_abs_diff_m_vs_gpkg": float(f"{max_dev_gpkg:.3e}"),
            "label_match": dict(coord_match),
            "label_match_note": "td 파일의 hobo-F<i> 는 좌표 표와 geopackage 에서 HOBO-F<i> 다. 대소문자를 "
                                "무시하고 맞췄다. 자료의 파일 이름 규약 표는 hobo-F<i> 를 '위치 i 의 HOBO "
                                "서미스터 줄'로 적었다",
            "coord_prec_deg": COORD_PREC_DEG,
            "coord_prec_note": "자료의 변수 표는 dGPS(Leica Viva Uno 10) 정확도를 0.5 m 로 적었다. geopackage 의 "
                               "exact_date = 0 인 지점은 날짜가 정확하지 않다는 표시이고 좌표 정확도 표시는 아니다",
            "sign_check": "경도 130.93-130.96°E, 위도 62.556-62.558°N. 동경이고 부호 변환은 없다",
        },
        "label_rules_applied": {
            "method": "PVC-TUBE, HANDBAR 는 probe. DRILL, PIT 는 pit_core. 한 행의 기록이 두 분류에 걸치면 "
                      "기록 수가 많은 분류, 같으면 probe 를 적고 notes 의 method_raw 에 원래 값을 적었다",
            "right_censored": "max_exceed = 1 인 기록이 있는 행은 1 이고 alt_cm 은 하한 값이다",
            "label_def": "방문 월 8-9월은 direct_eos(eos_basis = record_date), 10월은 direct_dated(none)",
            "visit_dates": {m: headers[m]["time_start"] for m in td_members},
            "disturbed": "자료 저자 스크립트의 하천 띠(기준점에서 부호 붙인 거리 -3 m 에서 3 m) 안이면 1, water. "
                         "해당 지점은 없다. 하천 기준점까지의 부호 붙인 거리는 notes 의 dist_river_ref_m 에 있다",
            "site_group": "scale-CS9(Site 2)와 scale-CS2(Site 1) 가운데 가까운 쪽. 자료 저자 스크립트는 "
                          "Site 2 를 50 m 이내, Site 1 을 90 m 이내로 잡았고 그 밖의 지점은 그림에 넣지 않았다",
            "alt_sd_cm": "같은 방문의 기록이 2개 이상일 때 표본 표준편차(n - 1)",
            "qc_flag": "coord 는 위치 지정을 확인하지 못한 행(반복 구간의 중복 의심 3행, 한 지점 이름에 같은 "
                       "방법의 다른 값이 둘 붙은 1행). date 는 PIT 기록의 구덩이 날짜와 파일 날짜의 연도가 다른 행",
        },
        "unverified_items": [
            "자료 논문 4.1.3 절은 이 값을 활동층 두께가 아니라 관측 시점의 융해 깊이라고 적었다. 현장 조사는 "
            "대개 9월이고 융해는 9월 또는 10월까지 이어진다고 했다. 9월 6일 방문(2013, 2014, 2015년)의 값이 "
            "연 최대보다 얼마나 작은지는 확인하지 못했다. direct_eos 는 관측 월 규칙(8-9월)에 따른 분류다",
            "수체 아래 지점과 탈릭 지점을 표시한 열은 자료에 없다. 자료 논문 서론은 이 관측소의 작은 하천이 "
            "겨울에 바닥까지 얼 수 있고 탈릭을 유지할 열을 주지 않는다고 적었다('Here, the water bodies do not "
            "provide heat to maintain a talik'). 하천 띠 규칙(기준점에서 -3 m 에서 3 m)에 드는 지점은 없어 "
            "disturbed = 1 인 행은 0개다",
            "단면 수심 표의 수면 폭은 CS9(Site 2) 2.5-3.8 m, CS2(Site 1) 7.4-7.8 m 다(meta 의 river_transects). "
            f"Site 2 의 굴착 지점 가운데 기준점에서 {near_dists[0]:.1f}-{near_dists[-1]:.1f} m 인 지점이 "
            f"{len(near_dists)}개 있다. 기준점이 수로 가운데에 있으면 이 지점들은 둑이고, 수로 한쪽 끝에 있으면 "
            "수로 가장자리일 수 있다. 기준점의 단면 안 위치는 확인하지 못했다. 이 지점들의 대상 행 "
            f"{len(near_vals)}개는 {fmt_num(near_vals[0])}-{fmt_num(near_vals[-1])} cm 이고 "
            f"{sum(v > s2_med for v in near_vals)}개가 Site 2 대상 행의 중앙값 {fmt_num(s2_med)} cm 보다 크다. "
            "하천의 열 영향을 받은 값일 수 있다",
            "자료 논문은 CS9(Site 2)를 호수의 열 영향이 없는 단면이라 중점 관측했다고 적었다. Site 1(CS2)은 상류 "
            "열카르스트 호수 가까이 있다. Site 1 의 값에 호수의 열 영향이 들었는지는 확인하지 못했다. "
            "disturbed 는 0 으로 두었다",
            "hobo-F2 의 2012-10-07 값 370 cm(DRILL)는 같은 지점의 2013-09-06 값 166 cm 의 2.2배다. 설치 직후"
            "(2012-10-09)의 300 cm, 400 cm 지온 기록은 -0.20, -0.26 °C 로 센서 정확도(0.25 °C) 안에서 0 °C "
            "부근이다. 탈릭인지 굴착 판독의 오차인지 확인하지 못했다. 이 행은 10월 방문이라 direct_dated 다",
            "2013년 파일의 반복 구간(기록 24-31 과 35-42)은 값, 방법, 순서가 같고 지점 이름 3개가 다르다. "
            "어느 위치 지정이 맞는지 확인하지 못했다. 앞선 구간을 남기고 뒤 구간의 3개 기록을 dup_of 로 "
            "표시했다. 두 지정의 위치 차이는 25 m 이내이고 같은 1 km 셀이다",
            "td 파일 머리말의 comments 는 'sometimes drill can be metal rod; some uncertainties in the combined "
            "xlsx file' 이라고 적었다. DRILL 로 적힌 기록 가운데 금속봉 측정이 섞였을 수 있다",
            "지점 이름의 연도는 위치를 측량한 해다. 2013년 파일의 drill-2017-* 처럼 관측 연도와 다른 경우가 "
            "있다. 관측일은 파일 머리말의 time_start 를 썼다. geopackage 의 2012년 굴착 지점 날짜는 "
            "2012-10-05, 2012-10-06 으로 파일의 2012-10-07 과 1-2일 다르다",
            "2012년 파일(2012-10-07)의 pit-2013-Pit-S1 기록(PIT, 160 cm)은 geopackage 의 구덩이 날짜가 "
            "2013-09-07(exact_date = 1)이다. 관측 연도가 2012년인지 2013년인지 확인하지 못했다. 파일 날짜를 "
            "쓰고 qc_flag 를 date 로 두었다",
            "방문일은 파일마다 하나다. 자료 논문은 일부 측정이 여러 날에 걸쳐 이루어졌다고 적었다. 기록 단위 "
            "날짜는 확인하지 못했다",
            "PVC-TUBE 의 측정 범위는 변수 표에 0-200 cm 로 적혀 있다. piezo-CS9-P2 의 2018년 기록은 158 cm 에 "
            "max_exceed = 1 이다. 관 길이가 지점마다 다른지 확인하지 못했다",
            "auxiliary/external 의 GPR, ERT 자료(Léger 등 2023)는 원시 단면과 비저항 표이고 융해 깊이 판독 "
            "값이 아니어서 쓰지 않았다",
        ],
    }
    meta_path = OUT_DIR / f"{SRC_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ------------------------------------------------------------ 화면 요약
    print(f"원자료 기록 {len(all_recs)}개, 상태별 {dict(status_count)}")
    print(f"점 자료 {len(points)}행, 지점 {len(sites_all)}개, 연도 {meta['year_range']}")
    print("label_def", meta["n_by_label_def"], "method", meta["n_by_method"])
    print("qc_flag", meta["n_by_qc_flag"], "right_censored", meta["n_right_censored"],
          "disturbed", meta["n_disturbed"], "dup_of", meta["n_dup_of"])
    print("ALT 전체", meta["alt_cm_stats_all_rows"])
    print("ALT 대상", meta["alt_cm_stats_target"], "대상 지점", len(sites_tgt))
    print("셀", meta["cell_preview"]["all_rows"], meta["cell_preview"]["target_rows"])
    for c in cell_table:
        print("  ", c)
    print("반복 구간", repeat_blocks)
    print("좌표 대조", meta["coordinate_conversion"]["max_abs_diff_deg_vs_pyproj"],
          meta["coordinate_conversion"]["max_abs_diff_m_vs_gpkg"], dict(coord_match))
    print("v3", v3)
    print("출력", points_path, meta_path, visits_path)


if __name__ == "__main__":
    main()
