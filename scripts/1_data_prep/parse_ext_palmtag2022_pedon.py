"""Palmtag 등 2022 토양 단면 자료(Bolin Centre)의 러시아 단면을 표준 점 자료로 만든다.

계획서: docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B 절(LGD, 개정 10), 자료원 id palmtag2022_pedon.
형식 정의: data/processed/ext_labels/_schema.json (버전 1.0).

입력(data/raw/palmtag2022_pedon/palmtag-2022-pedon-1/)
- pedon-data-on-soil-carbon-and-nitrogen.csv   단면 651개. 머리글 2행(이름, 단위). 41열.
- sample-data-on-soil-carbon-and-nitrogen.csv  시료 6,529개. 열 'PF (0=no; 1=yes)' 를 대조 점검에 쓴다.
- coordinates-study-sites.csv                  조사지 16곳의 대표 좌표. 좌표 점검에 쓴다.

라벨 규칙(_sources_plan.json 의 label_rule)
- 열 'Depth Top Permafrost' 값이 있는 러시아 단면만 쓴다. 같은 이름의 열이 둘이다(12번째, 13번째).
  대상 조사지는 12번째 열에만 값이 있어 12번째 열을 쓴다. 13번째 열은 Herschel 과 Lena Delta 에만 값이 있다.
- method = pit_core. 관측 월이 8-9월이면 label_def = direct_eos, eos_basis = record_date.
- Lena Delta 단면은 Lena 영역이라 뺀다. Seida, Spasskaya Pad 는 값이 없다.
- Aktru(알타이 고산)는 값이 2개 있다. 날짜가 'Aug-17' 이고 일이 없으며 고산 지점의 계절 말 기준을
  자료 설명에서 확인하지 못했다. 점 자료에 남기되 label_def = direct_dated, qc_flag = date 로 둔다.

좌표
- Kytalyk, Logata: 'Standardized latitude/longitude' 열의 십진 도를 원자료 문자열 그대로 쓴다.
- Ary-Mas: UTM 47 구역(EPSG:32647), Cherskii 와 Shalaurovo: UTM 57 구역(EPSG:32657). pyproj 로 바꾼다.
  원자료의 열 이름은 'Field coordinates, latitude' 이지만 단위 행은 'text, easting' 이고 값도 동거 좌표다.
  다음 열은 북거 좌표다. 측지 기준계는 파일에 없다. WGS84 로 가정했다.
- Aktru: 'Standardized' 열에 도와 소수 분('50 04.891')이 들어 있다. 십진 도로 바꾼다.

산출
- data/processed/ext_labels/palmtag2022_pedon_points.csv   표준 점 자료(30열)
- data/processed/ext_labels/palmtag2022_pedon_meta.json    메타와 품질 요약
- data/raw/palmtag2022_pedon/pedon_disposition.csv          단면 651개의 처리 결과(포함, 제외 사유)

실행(저장소 루트에서, 스레드 1개)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python scripts/1_data_prep/parse_ext_palmtag2022_pedon.py [--download]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
SRC_ID = "palmtag2022_pedon"
RAW = ROOT / "data" / "raw" / SRC_ID
UNZ = RAW / "palmtag-2022-pedon-1"
ZIP_NAME = "palmtag-2022-pedon-1.zip"
ZIP_URL = "https://bolin.su.se/data/uploads/palmtag-2022-pedon-1.zip"
LANDING = "https://bolin.su.se/data/palmtag-2022-pedon-1"
DOI = "10.17043/palmtag-2022-pedon-1"
ACCESSED = "2026-09-29"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"
SCHEMA = OUT_DIR / "_schema.json"
DISPO = RAW / "pedon_disposition.csv"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
FIDELITY_PY = ROOT / "src" / "polar" / "fidelity.py"

CITATION = ("Palmtag, J., Obu, J., Kuhry, P., Siewert, M., Weiss, N., Hugelius, G. (2022). "
            "Detailed pedon data on soil carbon and nitrogen for the northern permafrost region. "
            "Dataset version 1. Bolin Centre Database. https://doi.org/10.17043/palmtag-2022-pedon-1")
LICENSE = "ODC-By"
PAPER = ("Palmtag, J. 등 (2022). A high spatial resolution soil carbon and nitrogen dataset for the northern "
         "permafrost region based on circumpolar land cover upscaling. Earth Syst. Sci. Data 14, 4095. "
         "https://doi.org/10.5194/essd-14-4095-2022")

# 단면 표의 열 번호(0 부터). 같은 이름의 열이 있어 번호로 읽는다.
C_NAME, C_AREA, C_TRANSECT, C_PROFILE, C_DATE, C_CRS = 0, 1, 2, 3, 4, 5
C_EAST, C_NORTH, C_STDLAT, C_STDLON = 6, 7, 8, 9
C_FROZEN_AL, C_PFTOP, C_PFTOP2 = 10, 11, 12
C_WATER, C_PATTERN, C_DESC, C_DESC2, C_LANDCOVER = 15, 17, 28, 29, 31
C_PEDON_DEPTH, C_STOP = 38, 39
N_COL = 41

# 조사지 이름(열 'Study Area abbreviation')별 설정.
# country 는 조사지 이름과 coordinates-study-sites.csv 의 표기에서 읽었다. coord 는 좌표 읽는 방식이다.
AREAS = {
    "Taymyr, Logata":        dict(country="Russia", subunit="Taymyr_Logata", coord="std", site="Logata, Taymyr, Russia"),
    "Taymyr, Arymas":        dict(country="Russia", subunit="Taymyr_AryMas", coord="utm", epsg=32647, site="Arymas, Taymyr, Russia"),
    "Kytalyk":               dict(country="Russia", subunit="Indigirka_Kytalyk", coord="std", epsg_check=32655, site="Tjokurdach, Russia"),
    "Cherskii":              dict(country="Russia", subunit="Kolyma_Cherskii", coord="utm", epsg=32657, site="Cherskiy, Russia"),
    "Shalaurovo":            dict(country="Russia", subunit="Kolyma_Shalaurovo", coord="utm", epsg=32657, site="Shalaurovo, Russia"),
    "Shalaurovo/Cherskii":   dict(country="Russia", subunit="Kolyma_Shalaurovo-Cherskii", coord="utm", epsg=32657, site="Shalaurovo, Russia"),
    "Aktru, Russia":         dict(country="Russia", subunit="Altai_Aktru", coord="dm", site="Aktru, Russia", alpine=True),
    "Lena Delta":            dict(country="Russia", subunit="Lena_Delta", coord="none", site="Lena Delta, Russia"),
    "Seida":                 dict(country="Russia", subunit="Usa_Seida", coord="none", site="Seida, Usa River Basin, European Russia"),
    "Spasskaya Pad":         dict(country="Russia", subunit="Yakutia_C_SpasskayaPad", coord="none", site="Spasskaya Pad, Russia"),
    "Abisko":                dict(country="Sweden"),
    "Tarfala":               dict(country="Sweden"),
    "Adventdalen, Svalbard": dict(country="Svalbard (Norway)"),
    "Ny Ålesund, Svalbard":  dict(country="Svalbard (Norway)"),
    "Zackenberg":            dict(country="Greenland"),
    "HE":                    dict(country="Canada"),
    "Tulemalu lake, Canada": dict(country="Canada"),
}

# 교란 표시. 근거는 단면 표의 현장 기술(열 'Transect/site field description')이다.
# 어휘(burned, water, thermo_erosion, infrastructure)에 사면 이동 항목이 없어 CH T2-8 은 thermo_erosion 으로 적었다.
DISTURBED = {
    "AM EXP 1": ("thermo_erosion", "하천 노두. 구덩이가 노두 가장자리에서 3 m. 원문: Exposure on the northern side of the river. Permafrost angle ca. 45"),
    "AM EXP 2": ("thermo_erosion", "하천 노두. 구덩이가 노두 가장자리에서 3 m. 원문: Exposure on the southern side of the river. PF angle 40"),
    "AM T2-10": ("thermo_erosion", "침식 골짜기. 원문: Eroden gully (V-shape) in the forest tundra down to the river bank"),
    "CH T2-8":  ("thermo_erosion", "교란 사면. 원문: steep, soliflucted/disturbed slope into a river valley (disturbance regime)"),
}

LENA_BOX = (71.5, 73.6, 123.3, 130.1)  # 6B.4 의 레나 델타 영역(위도 하한, 상한, 경도 하한, 상한)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


# ---------------------------------------------------------------- 읽기
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw(download: bool) -> None:
    """원자료가 없으면 내려받아 푼다. --download 를 주지 않으면 없을 때 멈춘다."""
    z = RAW / ZIP_NAME
    if not z.exists():
        if not download:
            sys.exit(f"원자료가 없다: {z}. --download 를 주거나 SOURCE.md 의 방법으로 내려받는다.")
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ZIP_URL, z)
    if not (UNZ / "pedon-data-on-soil-carbon-and-nitrogen.csv").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)


def read_rows(path: Path) -> list[list[str]]:
    txt = path.read_text(encoding="utf-8-sig")
    return [[" ".join(c.split()) for c in r] for r in csv.reader(io.StringIO(txt, newline=""))]


# ---------------------------------------------------------------- 값 해석
def parse_depth(s: str):
    """깊이 문자열을 (값, 우측 절단, 상태)로 바꾼다. 값이 없으면 값은 None 이다."""
    s = s.strip()
    if s == "":
        return None, 0, "blank"
    if s.lower() == "na":
        return None, 0, "na"
    if s == "no_pf":
        return None, 0, "no_pf"
    if s == "?":
        return None, 0, "question_mark"
    m = re.fullmatch(r">\s*(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), 1, "censored"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\?", s)
    if m:
        return float(m.group(1)), 0, "uncertain"
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s), 0, "numeric"
    return None, 0, "unparsed"


def parse_date(s: str):
    """날짜 문자열을 (연, 월, 일, 형식)으로 바꾼다. 읽지 못한 항목은 None 이다.
    dd/mm/yyyy: Kytalyk, Cherskii, Shalaurovo(17/08/2010 처럼 앞 숫자가 12 를 넘는 값이 있어 일이 먼저다).
    dd.mm.yy: Logata, Ary-Mas(05.08.11-11.08.11). mmm-yy: Aktru(Aug-17), Spasskaya Pad(Jul-12)."""
    s = s.strip()
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1)), "dd/mm/yyyy"
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1)), "dd.mm.yyyy"
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{2})", s)
    if m:
        return 2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)), "dd.mm.yy"
    m = re.fullmatch(r"([A-Za-z]{3})-(\d{2})", s)
    if m and m.group(1).lower() in MONTHS:
        return 2000 + int(m.group(2)), MONTHS[m.group(1).lower()], None, "mmm-yy"
    return None, None, None, "unparsed" if s else "blank"


def n_dec(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def parse_dm(s: str):
    """'50 04.891' 같은 도와 소수 분을 십진 도로 바꾼다."""
    m = re.fullmatch(r"(-?\d+)\s+(\d+(?:\.\d+)?)", s.strip())
    if not m:
        return None
    d, mi = int(m.group(1)), float(m.group(2))
    return math.copysign(abs(d) + mi / 60.0, d if d != 0 else 1)


def macro_of(country: str, lat: float, lon: float) -> str:
    """6B.4 의 지리 정의. 라벨 값을 쓰지 않는다."""
    if country != "Russia":
        return "other"
    la0, la1, lo0, lo1 = LENA_BOX
    if la0 <= lat <= la1 and lo0 <= lon <= lo1:
        return "Lena"
    if lon < 0 or lon >= 140:
        return "Russia_E"
    if lon >= 90:
        return "Russia_C"
    return "Russia_W"


def transformer(cache: dict, epsg: int) -> Transformer:
    if epsg not in cache:
        cache[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
    return cache[epsg]


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


# ---------------------------------------------------------------- 본 처리
def build(download: bool = False) -> dict:
    ensure_raw(download)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    cols = [c["name"] for c in schema["columns"]]

    rows = read_rows(UNZ / "pedon-data-on-soil-carbon-and-nitrogen.csv")
    header, units, body = rows[0], rows[1], rows[2:]
    assert all(len(r) == N_COL for r in rows), "단면 표의 열 수가 41이 아니다"
    assert header[C_PFTOP] == "Depth Top Permafrost" and header[C_PFTOP2] == "Depth Top Permafrost"
    assert header[C_FROZEN_AL] == "Depth frozen AL"
    names = [r[C_NAME] for r in body]
    assert len(set(names)) == len(names), "단면 이름이 겹친다"

    # 시료 표: 단면별 PF = 1 시료의 최소 상단 깊이(대조 점검용)
    srows = read_rows(UNZ / "sample-data-on-soil-carbon-and-nitrogen.csv")
    sh = srows[0]
    i_up, i_pf, i_nm = sh.index("Upper sample depth"), sh.index("PF (0=no; 1=yes)"), sh.index("Sample name with depth")
    samples = [(r[i_nm], r[i_up], r[i_pf]) for r in srows[1:]]

    def pf_top_from_samples(pedon: str):
        tops, n = [], 0
        pat = re.compile(re.escape(pedon) + r"[ _]")
        for nm, up, pf in samples:
            if pat.match(nm):
                n += 1
                if pf == "1":
                    try:
                        tops.append(float(up))
                    except ValueError:
                        pass
        return (min(tops) if tops else None), n

    # 조사지 대표 좌표
    site_xy = {}
    for r in read_rows(UNZ / "coordinates-study-sites.csv")[1:]:
        if len(r) >= 3 and r[2]:
            site_xy[r[2]] = (float(r[1]), float(r[0]))  # (lat, lon)

    # 십진 도 좌표의 정밀도는 조사지 단위로 정한다(끝자리 0 이 빠진 값이 있어 행 단위로 세면 달라진다)
    std_dec = {}
    for r in body:
        if AREAS[r[C_AREA]].get("coord") == "std" and r[C_STDLAT] and r[C_STDLON]:
            std_dec[r[C_AREA]] = max(std_dec.get(r[C_AREA], 0), n_dec(r[C_STDLAT]), n_dec(r[C_STDLON]))

    tf = {}
    out, dispo, excl, excl_area = [], [], Counter(), {}
    ky_check, site_dist = [], {}
    for r in body:
        name, area = r[C_NAME], r[C_AREA]
        cfg = AREAS.get(area)
        assert cfg is not None, f"모르는 조사지 이름: {area}"
        val, cens, dstat = parse_depth(r[C_PFTOP])
        val2, _, dstat2 = parse_depth(r[C_PFTOP2])
        fz, _, fstat = parse_depth(r[C_FROZEN_AL])
        yy, mm, dd, dfmt = parse_date(r[C_DATE])
        d = OrderedDict(pedon=name, study_area=area, country=cfg["country"], date_raw=r[C_DATE],
                        crs_raw=r[C_CRS], easting_raw=r[C_EAST], northing_raw=r[C_NORTH],
                        std_lat_raw=r[C_STDLAT], std_lon_raw=r[C_STDLON],
                        depth_frozen_al_raw=r[C_FROZEN_AL], depth_top_pf_raw=r[C_PFTOP],
                        depth_top_pf_2nd_raw=r[C_PFTOP2], depth_status=dstat, status="", reason="")

        # 제외 사유(순서대로 하나만 적는다)
        if cfg["country"] != "Russia":
            reason = "non_russia_with_value" if val is not None or val2 is not None else "non_russia_no_value"
        elif area == "Lena Delta":
            reason = "lena_delta_region_with_value" if val is not None or val2 is not None else "lena_delta_region_no_value"
        elif val is None:
            reason = f"russia_no_depth_value_{dstat}"
        else:
            reason = ""
        if reason:
            excl[reason] += 1
            excl_area.setdefault(reason, Counter())[area] += 1
            d.update(status="excluded", reason=reason)
            dispo.append(d)
            continue

        # 좌표
        conv = ""
        if cfg["coord"] == "std":
            lat_s, lon_s = r[C_STDLAT], r[C_STDLON]
            lat, lon = float(lat_s), float(lon_s)
            prec = 10.0 ** (-std_dec[area])
            conv = "standardized_decimal_degree"
            if "epsg_check" in cfg and r[C_EAST] and r[C_NORTH]:
                t = transformer(tf, cfg["epsg_check"])
                lo2, la2 = t.transform(float(r[C_EAST]), float(r[C_NORTH]))
                ky_check.append(float(km(lat, lon, la2, lo2)) * 1000.0)
        elif cfg["coord"] == "utm":
            t = transformer(tf, cfg["epsg"])
            lon, lat = t.transform(float(r[C_EAST]), float(r[C_NORTH]))
            lat_s, lon_s = f"{lat:.6f}", f"{lon:.6f}"
            prec = 0.00001
            conv = f"utm_epsg{cfg['epsg']}"
        elif cfg["coord"] == "dm":
            lat, lon = parse_dm(r[C_STDLAT]), parse_dm(r[C_STDLON])
            assert lat is not None and lon is not None, f"좌표를 읽지 못했다: {name}"
            lat_s, lon_s = f"{lat:.6f}", f"{lon:.6f}"
            prec = 0.0000167
            conv = "degree_decimal_minute"
        else:
            raise AssertionError(f"좌표 방식이 없다: {area}")

        qc = []
        if not (40.0 <= lat <= 85.0 and -180.0 <= lon <= 180.0):
            qc.append("coord")
        if cfg["site"] in site_xy:
            sl, so = site_xy[cfg["site"]]
            dist = float(km(lat, lon, sl, so))
            site_dist.setdefault(area, []).append(dist)
            if dist > 100.0:
                qc.append("coord")
        if not (0.0 < val <= 600.0):
            qc.append("range")
        if yy is None or mm is None:
            qc.append("date")
        elif dd is None:
            qc.append("date")
        if yy is not None and yy < 1990:
            qc.append("year_pre1990")

        # 라벨 정의
        alpine = bool(cfg.get("alpine"))
        if mm in (8, 9) and dd is not None and not alpine:
            label_def, eos = "direct_eos", "record_date"
        else:
            label_def, eos = "direct_dated", "none"

        dis = DISTURBED.get(name)
        pf_s, n_s = pf_top_from_samples(name)

        notes = []
        if dstat == "uncertain":
            notes.append(f"원자료 값 '{r[C_PFTOP]}'(물음표 표기)")
        if fstat == "numeric":
            notes.append(f"Depth frozen AL {fmt_num(fz)} cm")
        elif r[C_FROZEN_AL]:
            notes.append(f"Depth frozen AL '{r[C_FROZEN_AL]}'")
        if r[C_WATER] and r[C_WATER].lower() != "na":
            notes.append(f"water table '{r[C_WATER]}'(음수는 지표 위 물)")
        if pf_s is not None:
            notes.append(f"시료 표 PF=1 최소 상단 깊이 {fmt_num(pf_s)} cm")
        if dfmt == "mmm-yy":
            notes.append(f"날짜 '{r[C_DATE]}' 를 mmm-yy 로 읽었다(일 없음)")
        if alpine:
            notes.append("알타이 고산 지점. 계절 말 기준을 자료 설명에서 확인하지 못했다")
        if dis:
            notes.append(dis[1])
        desc = r[C_DESC] or r[C_LANDCOVER]
        if desc:
            notes.append("현장 기술: " + (desc[:160] + ("..." if len(desc) > 160 else "")))

        date_iso = f"{yy:04d}-{mm:02d}-{dd:02d}" if None not in (yy, mm, dd) else ""
        rec = OrderedDict(
            src_id=SRC_ID, site_id=name.replace(" ", "_"), site_name=name, lat=lat_s, lon=lon_s,
            year="" if yy is None else yy, month="" if mm is None else mm, alt_cm=fmt_num(val),
            method="pit_core", label_def=label_def, n_obs=1, country=cfg["country"],
            macro=macro_of(cfg["country"], lat, lon), citation=CITATION, license=LICENSE,
            subunit=cfg["subunit"], date=date_iso, eos_basis=eos, value_kind="single_visit",
            year_min="", year_max="", alt_sd_cm="", coord_prec_deg=f"{prec:.7f}".rstrip("0"),
            disturbed=1 if dis else 0, disturb_type=dis[0] if dis else "", right_censored=cens,
            orig_source="", dup_of="", qc_flag=";".join(dict.fromkeys(qc)) if qc else "ok",
            notes="; ".join(notes))
        assert list(rec.keys()) == cols, "열 순서가 형식 정의와 다르다"
        rec["h_lat"], rec["h_lon"], rec["h_alt"], rec["h_frozen"], rec["h_pf_s"] = lat, lon, val, (fz if fstat == "numeric" else None), pf_s
        rec["h_conv"] = conv
        out.append(rec)
        d.update(status="included", reason="", lat=lat_s, lon=lon_s, coord_conversion=conv,
                 date_iso=date_iso, year=yy, month=mm, alt_cm=fmt_num(val), label_def=label_def,
                 macro=rec["macro"], disturbed=rec["disturbed"], pf_top_from_samples=fmt_num(pf_s),
                 n_samples_matched=n_s)
        dispo.append(d)

    # 같은 좌표의 단면
    key = Counter((r["lat"], r["lon"]) for r in out)
    same_xy = [r["site_name"] for r in out if key[(r["lat"], r["lon"])] > 1]
    for r in out:
        if key[(r["lat"], r["lon"])] > 1:
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + "같은 좌표의 다른 단면이 있다"

    # ------------------------------------------------------------ 쓰기
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POINTS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    dcols = list(OrderedDict((k, None) for d in dispo for k in d))
    with open(DISPO, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dcols, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(dispo)

    # ------------------------------------------------------------ 품질 요약
    df = pd.DataFrame(out)
    alt = df["h_alt"].astype(float)

    def stats(s: pd.Series) -> dict:
        s = s.astype(float)
        return dict(n=int(len(s)), min=float(s.min()), median=float(s.median()), mean=round(float(s.mean()), 2),
                    max=float(s.max())) if len(s) else dict(n=0)

    elig = df[(df.label_def == "direct_eos") & (df.qc_flag == "ok") & (df.disturbed == 0)
              & (df.right_censored == 0) & (df.year.astype(float) >= 1990)]

    # 시료 표 대조
    cmp_ = df[df["h_pf_s"].notna()]
    diff = (cmp_["h_pf_s"].astype(float) - cmp_["h_alt"].astype(float))
    pf_check = dict(
        n_points=int(len(df)), n_compared=int(len(cmp_)),
        n_abs_diff_le_5cm=int((diff.abs() <= 5).sum()),
        n_abs_diff_gt_5cm=int((diff.abs() > 5).sum()),
        diff_gt_5cm={r.site_name: dict(alt_cm=float(r.h_alt), pf_top_from_samples_cm=float(r.h_pf_s))
                     for r in cmp_[diff.abs() > 5].itertuples()},
        note="차 = 시료 표에서 PF=1 인 시료의 최소 상단 깊이 - 단면 표의 Depth Top Permafrost. "
             "거친 점검이다. 시료는 5-20 cm 간격으로 떠서 첫 PF 시료의 상단이 상한 깊이보다 깊을 수 있다. "
             "Kytalyk 은 동결된 활동층 시료에도 PF=1 이 붙어 있어 시료 쪽이 얕다. "
             "CH T1-2B 와 CH T6-4 는 시료 표에서 지표(0 cm) 시료부터 PF=1 이다. 이유는 확인하지 못했다")

    # Depth frozen AL 과의 비교(Kytalyk 만 값이 있다)
    fa = df[df["h_frozen"].notna()]
    fdiff = fa["h_alt"].astype(float) - fa["h_frozen"].astype(float)
    frozen_check = dict(
        n_with_frozen_al=int(len(fa)), by_subunit=fa.subunit.value_counts().to_dict(),
        n_top_pf_deeper=int((fdiff > 0).sum()), n_equal=int((fdiff == 0).sum()), n_top_pf_shallower=int((fdiff < 0).sum()),
        median_diff_cm=float(fdiff.median()) if len(fa) else None, max_diff_cm=float(fdiff.max()) if len(fa) else None,
        note="차 = Depth Top Permafrost - Depth frozen AL")

    # 1 km 셀 미리 보기(확정 값은 v4 조립에서 정한다)
    cell_preview = cell_summary(elig)

    n_raw = len(body)
    meta = OrderedDict(
        src_id=SRC_ID,
        name="Palmtag 등 2022, 북부 영구동토 지역 토양 단면 자료(Bolin Centre Database)",
        url=LANDING, download_url=ZIP_URL, doi=DOI, accessed=ACCESSED,
        files=[dict(name=str(p.relative_to(RAW)), size=p.stat().st_size, sha256=sha256(p))
               for p in [RAW / ZIP_NAME] + sorted(UNZ.iterdir())],
        license=LICENSE,
        license_basis="자료 안내 쪽(url)의 License 항목: Open Data Commons Attribution License (ODC-By). 판 번호는 적혀 있지 않다",
        citation=CITATION, related_paper=PAPER,
        parser="scripts/1_data_prep/parse_ext_palmtag2022_pedon.py",
        parser_git_commit=f"미커밋(작성 시점 HEAD {git_head()})",
        schema_version=schema["version"],
        n_rows_raw=n_raw, n_rows_points=len(out), n_sites=int(df.site_id.nunique()),
        n_by_label_def=df.label_def.value_counts().to_dict(),
        n_by_method=df.method.value_counts().to_dict(),
        n_by_macro=df.macro.value_counts().to_dict(),
        n_by_subunit=df.subunit.value_counts().to_dict(),
        n_by_eos_basis=df.eos_basis.value_counts().to_dict(),
        n_by_qc_flag=df.qc_flag.value_counts().to_dict(),
        n_disturbed=int(df.disturbed.sum()), disturbed_sites={k: v[1] for k, v in DISTURBED.items()},
        n_right_censored=int(df.right_censored.sum()),
        n_excluded_by_reason=dict(sorted(excl.items())),
        n_excluded_total=int(sum(excl.values())),
        n_excluded_by_reason_and_area={k: dict(sorted(v.items())) for k, v in sorted(excl_area.items())},
        year_range=[int(df.year.min()), int(df.year.max())],
        date_range_direct_eos=[elig.date.min(), elig.date.max()] if len(elig) else [],
        dates_by_subunit={k: [g.date.min(), g.date.max()] for k, g in df[df.date != ""].groupby("subunit")},
        n_direct_eos_before_aug15=int((elig.date < elig.date.str[:4] + "-08-15").sum()),
        august_day_range_direct_eos=[int(elig.date.str[8:10].astype(int).min()),
                                     int(elig.date.str[8:10].astype(int).max())] if len(elig) else [],
        alt_cm_all=stats(alt),
        alt_cm_by_macro={k: stats(g["h_alt"]) for k, g in df.groupby("macro")},
        alt_cm_by_subunit={k: stats(g["h_alt"]) for k, g in df.groupby("subunit")},
        n_target_eligible=int(len(elig)),
        target_eligible_rule="label_def = direct_eos, qc_flag = ok, disturbed = 0, right_censored = 0, year >= 1990",
        n_target_eligible_by_macro=elig.macro.value_counts().to_dict(),
        alt_cm_target_eligible_by_macro={k: stats(g["h_alt"]) for k, g in elig.groupby("macro")},
        coordinate_conversion=dict(
            Taymyr_Logata="원자료의 Standardized latitude/longitude(십진 도, 소수 5자리 이하)를 그대로 썼다",
            Indigirka_Kytalyk="원자료의 Standardized latitude/longitude(십진 도)를 그대로 썼다. UTM 55 구역(EPSG:32655) 환산값과 대조했다",
            Taymyr_AryMas="UTM 47 구역(원자료 표기 'UTM 47X', EPSG:32647) 동거, 북거를 pyproj 로 WGS84 십진 도로 바꿨다. 소수 6자리",
            Kolyma="UTM 57 구역(원자료 표기 'UTM zone 57N', EPSG:32657) 동거, 북거를 pyproj 로 WGS84 십진 도로 바꿨다. 소수 6자리",
            Altai_Aktru="Standardized 열의 도와 소수 분(예: '50 04.891')을 십진 도로 바꿨다. 소수 6자리",
            column_note="원자료 열 'Field coordinates, latitude' 의 단위 행은 'text, easting' 이고 값도 동거다. 다음 열은 북거다",
            datum="원자료에 측지 기준계가 없다. WGS84 로 가정했다",
            kytalyk_std_vs_utm55_m=dict(n=len(ky_check), max=round(max(ky_check), 2), median=round(float(np.median(ky_check)), 2)) if ky_check else {},
            distance_to_study_site_km={k: dict(min=round(min(v), 1), max=round(max(v), 1)) for k, v in site_dist.items()},
            distance_note="coordinates-study-sites.csv 의 대표 좌표와의 거리다. 100 km 를 넘으면 qc_flag 에 coord 를 붙인다. "
                          "대표 좌표는 소수 1-2자리이고 Aktru(50.05, 87.47)는 단면 표의 도와 분(50 05, 87 47)과 숫자가 같아 "
                          "도.분 표기로 보인다. 거친 점검으로만 쓴다",
            pyproj_version=__import__("pyproj").__version__),
        date_formats=dict(
            Kytalyk_Cherskii_Shalaurovo="dd/mm/yyyy. 17/08/2010 처럼 앞 숫자가 12 를 넘는 값이 같은 열에 있다",
            Logata_AryMas="dd.mm.yy(05.08.11-11.08.11 을 2011-08-05 에서 2011-08-11 로 읽었다)",
            Aktru="'Aug-17'. mmm-yy 로 읽어 2017년 8월로 두었다. 일은 없다"),
        label_semantics=dict(
            column_used="단면 표의 12번째 열 'Depth Top Permafrost'(단위 행 비어 있음, 값은 cm 로 읽었다)",
            second_column="13번째 열도 이름이 'Depth Top Permafrost' 다. 대상 조사지에는 값이 없다",
            frozen_al="Kytalyk 은 'Depth frozen AL'(관측일의 동결면 깊이로 보인다)과 'Depth Top Permafrost' 가 따로 있다. "
                      "추가 표의 Major Unit 에서 두 깊이 사이의 시료는 'AL/SI' 로, 그 아래는 'PF' 로 적혀 있다. "
                      "Kytalyk 의 라벨은 관측일의 융해 깊이가 아니라 단면에서 판정한 영구동토 상한 깊이다",
            other_areas="Logata, Ary-Mas, Cherskii, Shalaurovo 는 'Depth frozen AL' 이 비어 있다. "
                        "'Depth Top Permafrost' 가 관측일의 동결면 깊이인지 판정한 영구동토 상한인지는 기록 단위로 구분되지 않는다",
            paper_statement="관련 논문 2.2 절: 조사는 6월 말에서 9월 초이고 8월과 9월이 가장 많다. "
                            "활동층 두께는 눈금 있는 강철 탐침 또는 굴착 구덩이의 줄자로 쟀다. 기록 단위의 측정 도구는 표에 없다"),
        qc_checks=dict(
            units="깊이 열의 단위 행은 비어 있다. 시료 표의 깊이(cm)와 같은 크기라 cm 로 읽었다",
            missing_codes="빈 칸, NA, na, no_pf, ? 는 값 없음으로 처리했다",
            duplicate_names="단면 이름 651개에 중복이 없다",
            same_coordinate_sites=same_xy,
            pf_flag_crosscheck=pf_check,
            frozen_al_vs_top_pf=frozen_check,
            range="0 < alt_cm <= 600 을 벗어난 값 수: %d" % int(df.qc_flag.str.contains("range").sum())),
        cell_preview=cell_preview,
        expected_cells_in_plan="Russia_C 새 셀 23개(블록 2개), Russia_E 새 셀 17개(탐색 단계 추정)",
        not_used_but_available=dict(
            note="규칙에 따라 러시아 밖 단면과 Lena Delta 단면은 점 자료에 넣지 않았다. "
                 "12번째 열에 수치가 있는 단면 수를 조사지별로 적는다(우측 절단 표기 포함)",
            counts=non_russia_counts(body)),
        unverified_items=[
            "깊이 열의 단위(단위 행이 비어 있다). cm 로 읽었다",
            "UTM 좌표의 측지 기준계. WGS84 로 가정했다",
            "Logata, Ary-Mas, Cherskii, Shalaurovo 의 'Depth Top Permafrost' 가 관측일의 동결면 깊이인지 단면에서 판정한 영구동토 상한인지",
            "기록 단위의 측정 도구(탐침 또는 구덩이의 줄자). method 는 자료원 규칙에 따라 pit_core 로 두었다",
            "Aktru 의 날짜 'Aug-17' 의 해석(2017년 8월로 읽었다)과 고산 지점의 계절 말 기준",
            "조사지별 원 출처 논문(관련 논문의 표 1). orig_source 를 비워 두었다",
            "ODC-By 의 판 번호",
            "교란 표시 4건은 현장 기술의 문구로 판단했다. 원 저자의 교란 분류는 표에 없다",
            "v4 조립 단계의 100 km 독립성 규칙과 공변량 유효 여부는 이 파서에서 판정하지 않았다"],
    )
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def non_russia_counts(body) -> dict:
    c = Counter()
    for r in body:
        cfg = AREAS[r[C_AREA]]
        if cfg["country"] != "Russia" or r[C_AREA] == "Lena Delta":
            v, _, _ = parse_depth(r[C_PFTOP])
            if v is not None:
                c[r[C_AREA]] += 1
    return dict(sorted(c.items()))


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "확인하지 못했다"


def v3_macro_map() -> dict:
    """src/polar/fidelity.py 의 MACRO_REGION 을 불러오지 않고 읽는다(모듈을 고치지 않고, sklearn 도 불러오지 않는다)."""
    tree = ast.parse(FIDELITY_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MACRO_REGION" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def cell_summary(elig: pd.DataFrame) -> dict:
    """6B.3 의 셀 색인으로 묶고 v3 의 F4_direct 셀과 대조한다. 읽기만 한다."""
    if not len(elig):
        return {}
    e = elig.copy()
    e["ky"] = np.floor(e["h_lat"].astype(float) / 0.009).astype(int)
    phi = np.radians((e.ky + 0.5) * 0.009)
    e["kx"] = np.floor(e["h_lon"].astype(float) * np.cos(phi) / 0.009).astype(int)
    cells = e.groupby(["macro", "subunit", "ky", "kx"]).agg(
        lat=("h_lat", "mean"), lon=("h_lon", "mean"), alt_cm=("h_alt", "mean"), n_sites=("site_id", "size")).reset_index()
    cells["block"] = (np.floor(cells.lat / 0.5).astype(int) * 100000 + np.floor(cells.lon / 0.5).astype(int))
    res = dict(note="미리 보기다. 확정 셀 수는 v4 조립에서 정한다. 대상 행 조건은 target_eligible_rule 과 같다. "
                    "n_cells_if_0p01deg_grid 는 위도와 경도를 0.01 도로 나눈 격자의 셀 수다(탐색 단계 추정과 견주기 위한 값)",
               cell_rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 도")
    if V3.exists():
        v = pd.read_csv(V3, usecols=["lat", "lon", "region", "source_id"], low_memory=False)
        mm = v3_macro_map()
        v["macro"] = v.region.map(lambda x: mm.get(x, x))
        f4 = v[v.source_id == "F4_direct"]
        nf = v[v.source_id != "F4_direct"]
        nl, no = nf.lat.to_numpy(), nf.lon.to_numpy()
        cells["near_v3_other_source"] = [
            bool((np.maximum(np.abs(nl - c.lat), np.abs(no - c.lon)) <= 0.01).any()) for c in cells.itertuples()]
        vl, vo = f4.lat.to_numpy(), f4.lon.to_numpy()
        dup, dmin_same, dmin_other, other_name = [], [], [], []
        for c in cells.itertuples():
            cheb = np.maximum(np.abs(vl - c.lat), np.abs(vo - c.lon))
            dup.append(bool((cheb <= 0.01).any()))
            dk = km(c.lat, c.lon, vl, vo)
            oth = (f4.macro != c.macro).to_numpy()
            j = int(np.argmin(np.where(oth, dk, np.inf)))
            dmin_other.append(float(dk[j]))
            other_name.append(str(f4.macro.iloc[j]))
            dmin_same.append(float(np.min(np.where(~oth, dk, np.inf))))
        cells["dup_v3_f4"] = dup
        cells["km_other_macro_v3_f4"] = dmin_other
        cells["other_macro"] = other_name
        cells["km_same_macro_v3_f4"] = dmin_same
        v3_blocks = set((np.floor(f4.lat / 0.5).astype(int) * 100000 + np.floor(f4.lon / 0.5).astype(int)).tolist())
    else:
        cells["dup_v3_f4"] = False
        cells["near_v3_other_source"] = False
        v3_blocks = set()
        res["v3"] = "fidelity_base_v3.csv 가 없어 대조하지 못했다"
    by = {}
    for mcr, g in cells.groupby("macro"):
        new = g[~g.dup_v3_f4]
        d = dict(n_sites=int(g.n_sites.sum()), n_cells=int(len(g)), n_cells_dup_v3_f4=int(g.dup_v3_f4.sum()),
                 n_new_cells=int(len(new)), n_blocks_new_cells=int(new.block.nunique()),
                 n_blocks_not_in_v3_f4=int(len(set(new.block) - v3_blocks)),
                 n_new_cells_within_0p01deg_of_v3_f2_or_calm_temp=int(new.near_v3_other_source.sum()),
                 n_cells_if_0p01deg_grid=int(len(set(zip(
                     np.floor(e.loc[e.macro == mcr, "h_lat"].astype(float) / 0.01).astype(int),
                     np.floor(e.loc[e.macro == mcr, "h_lon"].astype(float) / 0.01).astype(int))))),
                 alt_cm_cell_mean=round(float(new.alt_cm.mean()), 1) if len(new) else None,
                 by_subunit={k: dict(n_sites=int(s.n_sites.sum()), n_new_cells=int((~s.dup_v3_f4).sum()),
                                     n_blocks=int(s[~s.dup_v3_f4].block.nunique()))
                             for k, s in g.groupby("subunit")})
        if "km_other_macro_v3_f4" in g:
            d["min_km_to_other_macro_v3_f4"] = round(float(new.km_other_macro_v3_f4.min()), 1) if len(new) else None
            d["n_new_cells_within_100km_of_other_macro"] = int((new.km_other_macro_v3_f4 < 100).sum())
            d["nearest_other_macro"] = str(new.loc[new.km_other_macro_v3_f4.idxmin(), "other_macro"]) if len(new) else None
            d["min_km_to_same_macro_v3_f4"] = (round(float(new.km_same_macro_v3_f4.min()), 1)
                                               if len(new) and np.isfinite(new.km_same_macro_v3_f4.min()) else None)
        by[mcr] = d
    res["by_macro"] = by
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Palmtag 2022 단면 자료를 표준 점 자료로 만든다")
    ap.add_argument("--download", action="store_true", help="원자료가 없으면 내려받는다")
    a = ap.parse_args()
    m = build(download=a.download)
    keys = ["n_rows_raw", "n_rows_points", "n_sites", "n_by_label_def", "n_by_method", "n_by_macro", "n_by_subunit",
            "n_by_qc_flag", "n_disturbed", "n_right_censored", "n_excluded_by_reason", "year_range",
            "date_range_direct_eos", "n_direct_eos_before_aug15", "alt_cm_all", "alt_cm_by_macro",
            "n_target_eligible_by_macro", "cell_preview"]
    print(json.dumps({k: m[k] for k in keys}, ensure_ascii=False, indent=1))
    print("점 자료:", POINTS)
    print("메타:", META)
    print("처리 결과 표:", DISPO)
