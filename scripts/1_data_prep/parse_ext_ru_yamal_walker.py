#!/usr/bin/env python3
"""ext_labels 파서: ru_yamal_walker.

자료: Walker 등 (2009), 야말 측선 2007–2008 조사.
  PANGAEA 842711 (Table 20) Thaw depth of active layer at transects and relevés (CC BY 3.0). 라벨의 출처다
  PANGAEA 842691 (Table 22) Vegetation, soil and site characteristics of all relevés (CC BY 3.0).
                 relevé 의 관측일, 좌표, 지점 특성, Nadym-2 relevé 의 융해 깊이의 출처다
원 출처: Walker, D.A. 등 (2009), Data Report of the 2007 and 2008 Yamal Expeditions. Alaska Geobotany Center,
  University of Alaska Fairbanks(WalkerDA2009_yamal_dr090401.pdf). 두 PANGAEA 표는 이 보고서의 표 20, 표 22 를 옮긴 것이다.

입력(data/raw/ru_yamal_walker/)
  PANGAEA_842711.tab                 측선과 relevé 의 융해 깊이(73행)
  PANGAEA_842691.tab                 relevé 특성 표(49행)
  WalkerDA2009_yamal_dr090401.pdf    원 보고서(선택). 방법 확인과 값 대조에만 쓴다. 약관은 확인하지 못했다
출력
  data/processed/ext_labels/ru_yamal_walker_points.csv   표준 점 자료(형식 1.0)
  data/processed/ext_labels/ru_yamal_walker_meta.json    메타
  data/raw/ru_yamal_walker/releve_crosscheck.csv          relevé 값의 표 20(842711)과 표 22(842691) 대조

규칙(계획서 6B.3, _schema.json 1.0)
  1. 한 행은 측선 하나 또는 relevé 하나의 단일 방문 값이다. 각 단위는 한 번만 관측되었으므로 (site_id, year) 가 한 행이다.
  2. alt_cm 은 측선의 평균(Thaw depth average)과 relevé 의 단일 값이다. 단위는 cm 이고 변환하지 않는다.
  3. method = probe. 보고서 방법 절: 'measured ... along each transect using a 2-m long steel probe'.
  4. 관측일은 842691 의 relevé 날짜다. 측선의 날짜는 원자료에 없으므로 같은 이벤트(연구 지점) relevé 의 날짜를 쓴다.
     관측 월이 8월 또는 9월이면 label_def = direct_eos, eos_basis = record_date 다. 이 자료는 전부 8월이다.
  5. 842691 의 융해 깊이 0 은 결측으로 본다(Vaskiny Dachi-3 relevé 35–39. 842711 에는 값이 있다).
  6. '>' 표기는 융해 깊이 칸에 없다. 탐침 길이는 2 m 이고 최댓값은 136 cm 라 right_censored 는 모두 0 이다.
  7. 중복: v3 의 F4_direct 셀과 체비쇼프 0.01° 이내이면 dup_of = v3:<loc_id>(Vaskiny Dachi, CALM R5a–c).
     그다음 calm_pangaea_v4add 의 지점과 0.01° 이내이면 dup_of = calm_pangaea_v4add:<site_id>(Nadym-2, CALM R1).
  8. 842711 의 Kharasavey1 블록은 원 보고서 표 20 과 다르다(아래 KH1_REPORT 설명). 보고서 값을 새 행으로 넣고
     (license = unverified), 842711 의 Kharasavey1 행은 지우지 않고 dup_of 로 보고서 행 또는 Kharasavey2b 행을 가리킨다.

실행: 스레드 1개. 입력은 표 두 개(7 KB, 16 KB)다. PDF 는 읽지 않는다(보고서 값은 아래 상수로 옮겨 적었다).
  OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_ru_yamal_walker.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SRC_ID = "ru_yamal_walker"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
CALM_ADD_PATH = OUT_DIR / "calm_pangaea_v4add_points.csv"

TAB_THAW = "PANGAEA_842711.tab"
TAB_REL = "PANGAEA_842691.tab"
JSONLD_THAW = "PANGAEA_842711_metadata.jsonld"
JSONLD_REL = "PANGAEA_842691_metadata.jsonld"
JSONLD_PARENT = "PANGAEA_842707_parent_metadata.jsonld"
PDF_NAME = "WalkerDA2009_yamal_dr090401.pdf"
ACCESSED = "2026-09-29"

URL = "https://doi.org/10.1594/PANGAEA.842711"
DOI = "10.1594/PANGAEA.842711"
FILE_URLS = {
    TAB_THAW: "https://doi.pangaea.de/10.1594/PANGAEA.842711?format=textfile",
    TAB_REL: "https://doi.pangaea.de/10.1594/PANGAEA.842691?format=textfile",
    JSONLD_THAW: "https://doi.pangaea.de/10.1594/PANGAEA.842711?format=metadata_jsonld",
    JSONLD_REL: "https://doi.pangaea.de/10.1594/PANGAEA.842691?format=metadata_jsonld",
    JSONLD_PARENT: "https://doi.pangaea.de/10.1594/PANGAEA.842707?format=metadata_jsonld",
    PDF_NAME: "https://www.geobotany.uaf.edu/library/reports/WalkerDA2009_yamal_dr090401.pdf",
}
AUTHORS = ("Walker, Donald A; Epstein, Howard E; Leibman, Marina O; Moskalenko, Nataliya G; Orekhov, Pavel; "
           "Kuss, Patrick; Matyshak, George V; Kaarlejärvi, Elina; Forbes, Bruce C; Barbour, E M; Gobroski, K (2009)")
CITATION_THAW = (AUTHORS + ": (Table 20) Thaw depth of active layer at transects and relevés sampled along the "
                 "Yamal Transect, 2007-2008 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.842711")
CITATION_REL = (AUTHORS + ": (Table 22) Vegetation, soil and site characteristics of all relevés sampled along the "
                "Yamal Transect, 2007-2008 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.842691")
CITATION_REPORT = ("Walker, D.A.; Epstein, H.E.; Leibman, M.E.; Moskalenko, N.G.; Orekhov, P.; Kuss, J.P.; "
                   "Matyshak, G.V.; Kaarlejärvi, E.; Forbes, B.C.; Barbour, E.M.; Gobroski, K. (2009): Data Report of "
                   "the 2007 and 2008 Yamal Expeditions. Alaska Geobotany Center, Institute of Arctic Biology, "
                   "University of Alaska Fairbanks, Fairbanks, AK, 121 pp. "
                   "https://www.geobotany.uaf.edu/library/reports/WalkerDA2009_yamal_dr090401.pdf")
ORIG_SOURCE = CITATION_REPORT
LICENSE_PANGAEA = "CC-BY-3.0"
LICENSE_REPORT = "unverified"
COUNTRY = "Russia"
SUBUNIT = "Yamal"
PREC_EVENT = "0.00001"   # 이벤트 좌표(머리말, 소수 6자리이고 끝자리가 모두 0)
PREC_RELEVE = "0.0001"   # relevé 좌표(842691, 소수 4자리)

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
            "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
            "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

THAW_COLS = ["Event", "Latitude", "Longitude", "Sample ID (transect/relevé no.)", "N [#]", "Thaw depth [cm] (max)",
             "Thaw depth [cm] (min)", "Thaw depth [cm] (average)", "Std dev [±]"]
EVENT_NAMES = {"Nadym1": "Nadym-1", "Nadym2": "Nadym-2", "Laborovaya1": "Laborovaya-1",
               "Laborovaya2": "Laborovaya-2", "VaskinyDachi1": "Vaskiny Dachi-1", "VaskinyDachi2": "Vaskiny Dachi-2",
               "VaskinyDachi3": "Vaskiny Dachi-3", "Kharasavey1": "Kharasavey-1", "Kharasavey2a": "Kharasavey-2a",
               "Kharasavey2b": "Kharasavey-2b"}

# 842691 의 relevé 번호 표기 정정. 근거는 원 보고서 표 22 의 행 순서와 값, 표 7 의 GPS 좌표다.
#   (이벤트, 원표기, 위도, 경도) -> relevé 번호
RELEVE_ID_FIX = {
    ("Nadym2", "8", "65.3147", "72.8617"): (6, "원표기 '8'. Nadym2 에 '8' 이 두 번 나온다. 보고서 표 22 의 06 행(융해 깊이 40, "
                                            "미세 기복 30)과 표 7 의 ND RV 06 좌표(65°18.883', 72°51.703')와 같아 06 으로 보았다"),
    ("VaskinyDachi1", "2B", "70.2758", "68.8913"): (28, "원표기 '2B'. 보고서 표 22 의 28 행과 표 7 의 VD RV 28 "
                                                    "좌표(70°16.547', 68°53.475')와 같아 28 로 보았다"),
}

# 원 보고서 표 20(인쇄 쪽 37, PDF 49쪽)의 Kharasavey-1 블록. 표는 그림으로 들어 있어 원해상도 그림을 눈으로 읽어 옮겼다.
#   842711 의 Kharasavey1 블록은 이 값과 다르다. 842711 은 측선 번호를 T-46..T-50(Kharasavey-2b 의 번호), N 을 6 으로 적었고
#   표준편차 5개만 아래 값과 순서대로 같다. 842711 의 T-49, T-50 은 N = 6 에서 (최대 - 최소) 와 표준편차가 양립하지 않는다.
#   842711 의 Kharasavey1 relevé RV-47..49(71, 60, 76.5)는 보고서 표 20 의 Kharasavey-2b relevé 값과 같다.
#   보고서 표 7(인쇄 쪽 22)의 GPS 표도 Kharasavey-1 의 측선을 T36–T40, relevé 를 RV 40–44 로 적었다.
KH1_REPORT = [
    # (표기, N, 최대, 최소, 평균, 표준편차)
    ("T36", 11, "80", "53", "62.8", "8.75"),
    ("T37", 11, "73", "52", "59.5", "5.47"),
    ("T38", 11, "64", "52", "59.3", "3.77"),
    ("T39", 11, "67", "55", "61.8", "3.87"),
    ("T40", 11, "70", "56", "62.9", "4.53"),
    ("RV-40", 1, "", "", "67", ""),
    ("RV-41", 1, "", "", "59", ""),
    ("RV-42", 1, "", "", "65", ""),
    ("RV-43", 1, "", "", "54", ""),
    ("RV-44", 1, "", "", "57", ""),
]
# 842711 의 Kharasavey1 행이 가리킬 행(dup_of)
KH1_PANGAEA_DUP = {"T-46": "Kharasavey1_T36", "T-47": "Kharasavey1_T37", "T-48": "Kharasavey1_T38",
                   "T-49": "Kharasavey1_T39", "T-50": "Kharasavey1_T40",
                   "RV-47": "Kharasavey2b_RV-47", "RV-48": "Kharasavey2b_RV-48", "RV-49": "Kharasavey2b_RV-49"}

# 842691 의 코드 설명(PANGAEA 머리말)
DIST_TYPE = {"0": "none", "1": "ptarmigan scat", "2": "caribou tracks", "3": "caribou scat", "6": "vole tracks and scat",
             "7": "vehicle tracks"}


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


def read_pangaea_tab(path: Path, need_cols=None, unique_cols=True):
    """PANGAEA 탭 파일을 머리말(/* ... */)과 표로 나눈다. 값은 문자열 그대로 둔다.

    842691 은 'Thaw depth [cm]' 와 'Characteristic (...)' 열 이름이 겹치므로 열 번호로도 돌려준다."""
    text = path.read_text(encoding="utf-8")
    end = text.index("*/")
    header = text[: end + 2]
    body = text[end + 2:].lstrip("\r\n")
    rows = list(csv.reader(body.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE))
    cols, data = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    if need_cols:
        missing = [c for c in need_cols if c not in cols]
        if missing:
            raise SystemExit(f"{path.name}: 열이 없다 {missing}")
    padded = [[v.strip() for v in (r + [""] * (len(cols) - len(r)))] for r in data]
    if unique_cols:
        return header, cols, [dict(zip(cols, r)) for r in padded]
    return header, cols, padded


def header_license(header: str) -> str:
    m = re.search(r"^License:\t(.+)$", header, re.M)
    return m.group(1).strip() if m else ""


def header_events(header: str):
    """머리말 Event(s) 의 이벤트 좌표 문자열(소수 6자리)과 고도."""
    out = {}
    for m in re.finditer(r"(\w+) \((\w+)\) \* LATITUDE: ([\d.\-]+) \* LONGITUDE: ([\d.\-]+) \* ELEVATION: ([\d.\-]+) m",
                         header):
        out[m.group(1)] = {"lat": m.group(3), "lon": m.group(4), "elev_m": m.group(5)}
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


def load_v3_ref(v3_path: Path):
    """v3 의 F4_direct 셀 가운데 야말 주변(lat 60–75, lon 55–90)만 읽는다. 표를 한 줄씩 읽는다."""
    ref = []
    if not v3_path.exists():
        return ref
    with open(v3_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] != "F4_direct":
                continue
            la, lo = float(r["lat"]), float(r["lon"])
            if 60.0 < la < 75.0 and 55.0 < lo < 90.0:
                ref.append((la, lo, r["region"], r["loc_id"], r["alt_cm"]))
    return ref


def load_calm_add(path: Path):
    ref = {}
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            la, lo = float(r["lat"]), float(r["lon"])
            if 60.0 < la < 75.0 and 55.0 < lo < 90.0:
                ref.setdefault(r["site_id"], (la, lo, r["site_name"]))
    return [(v[0], v[1], k, v[2]) for k, v in ref.items()]


def nearest_cheb(lat, lon, ref, key_idx):
    best = None
    for item in ref:
        d = max(abs(lat - item[0]), abs(lon - item[1]))
        if best is None or d < best[0]:
            best = (d, item[key_idx], item)
    return best


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


def releve_no(label: str):
    m = re.fullmatch(r"RV-?(\d+)\*?", label)
    return int(m.group(1)) if m else None


def cross_status(td1: str, td2: str, avg: str) -> str:
    """relevé 값의 표 20(842711 또는 보고서)과 842691(보고서 표 22) 대조. 842691 의 빈 칸과 0 은 결측이다."""
    if td1 == "" or float(td1) == 0:
        return "missing_842691"
    if td2 == "" and abs(float(td1) - float(avg)) < 1e-9:
        return "match"
    return "differ"


def sd_consistent(n: int, vmax: str, vmin: str, sd: str):
    """(최대 - 최소) 와 표준편차가 양립하는지 본다. 모표준편차의 하한 R / sqrt(2N) 을 쓴다(가장 느슨한 기준)."""
    if not (vmax and vmin and sd) or n < 2:
        return None
    r = float(vmax) - float(vmin)
    return float(sd) + 1e-9 >= r / math.sqrt(2 * n)


def main():
    ap = argparse.ArgumentParser(description="ext_labels 파서: ru_yamal_walker")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--v3", type=Path, default=V3_PATH)
    ap.add_argument("--calm-add", type=Path, default=CALM_ADD_PATH)
    args = ap.parse_args()

    h_thaw, _, thaw = read_pangaea_tab(args.raw_dir / TAB_THAW, THAW_COLS)
    h_rel, rel_cols, rel_rows = read_pangaea_tab(args.raw_dir / TAB_REL, unique_cols=False)
    events = header_events(h_thaw)
    events_rel = header_events(h_rel)
    lic_thaw, lic_rel = header_license(h_thaw), header_license(h_rel)

    # ---- 842691 relevé 표. 'Thaw depth [cm]' 열이 두 개다(보고서 표 22 의 NSC/Inter 두 값)
    ci = {name: i for i, name in reversed(list(enumerate(rel_cols)))}
    i_td = [i for i, c in enumerate(rel_cols) if c == "Thaw depth [cm]"]
    if len(i_td) != 2:
        raise SystemExit(f"{TAB_REL}: 'Thaw depth [cm]' 열이 2개가 아니다({len(i_td)})")
    i_dist = [i for i, c in enumerate(rel_cols) if c.startswith("Characteristic (disturbance")]
    i_micro = [i for i, c in enumerate(rel_cols) if c.startswith("Characteristic (micro-site")]
    releve = {}              # (event, 번호) -> dict
    rel_no_thaw, rel_fix_notes = [], {}
    for r in rel_rows:
        ev, sid, lat_s, lon_s = r[ci["Event"]], r[ci["Sample ID (relevee)"]], r[ci["Latitude"]], r[ci["Longitude"]]
        fix = RELEVE_ID_FIX.get((ev, sid, lat_s, lon_s))
        if fix:
            no, why = fix
            rel_fix_notes[(ev, no)] = why
        elif sid.isdigit():
            no = int(sid)
        else:
            raise SystemExit(f"{TAB_REL}: relevé 번호를 읽지 못했다 {ev} {sid}")
        if (ev, no) in releve:
            raise SystemExit(f"{TAB_REL}: relevé 번호가 중복된다 {ev} {no}")
        td1, td2 = r[i_td[0]], r[i_td[1]]
        rec = {"event": ev, "no": no, "raw_id": sid, "lat": lat_s, "lon": lon_s, "date": r[ci["Date/Time"]],
               "site": r[ci["Site"]], "td1": td1, "td2": td2,
               "dist_deg": r[i_dist[0]] if i_dist else "", "dist_type": r[i_dist[1]] if len(i_dist) > 1 else "",
               "micro": r[i_micro[0]] if i_micro else ""}
        releve[(ev, no)] = rec
        # 0 은 결측으로 본다(규칙 5)
        v1 = float(td1) if td1 not in ("",) else None
        if v1 is None or v1 == 0:
            rel_no_thaw.append(rec)

    # 이벤트별 관측일(relevé 날짜). 이벤트 안에서 날짜가 하나여야 한다
    ev_dates = defaultdict(set)
    ev_site = defaultdict(set)
    for rec in releve.values():
        ev_dates[rec["event"]].add(rec["date"])
        ev_site[rec["event"]].add(rec["site"].replace("CALM-gnd.", "CALM-grid").split(",")[0].split(".")[0])
    for ev, ds in ev_dates.items():
        if len(ds) != 1:
            raise SystemExit(f"{TAB_REL}: 이벤트 {ev} 의 relevé 날짜가 여러 개다 {ds}")
    ev_date = {ev: next(iter(ds)) for ev, ds in ev_dates.items()}

    v3_ref = load_v3_ref(args.v3)
    calm_ref = load_calm_add(args.calm_add)

    def dup_of_for(lat, lon):
        """규칙 7. v3 F4_direct 먼저, 그다음 calm_pangaea_v4add."""
        b = nearest_cheb(lat, lon, v3_ref, 3) if v3_ref else None
        if b and b[0] <= 0.01:
            return f"v3:{b[1]}", f"v3 F4_direct loc_id {b[1]}({b[2][2]}, alt_cm {float(b[2][4]):.1f})와 체비쇼프 {b[0]:.4f}°"
        c = nearest_cheb(lat, lon, calm_ref, 2) if calm_ref else None
        if c and c[0] <= 0.01:
            return (f"calm_pangaea_v4add:{c[1]}",
                    f"calm_pangaea_v4add {c[1]}({c[2][3]})와 체비쇼프 {c[0]:.4f}°")
        return "", ""

    points, cross = [], []

    def make_row(ev, label, n, vmax, vmin, avg, sd, *, from_report=False, dup_override="", extra_notes=(),
                 orig_label=None):
        is_rv = label.startswith("RV")
        no = releve_no(label) if is_rv else None
        rv = releve.get((ev, no)) if is_rv else None
        notes, flags = [], []
        # 좌표
        if rv:
            lat_s, lon_s, prec = rv["lat"], rv["lon"], PREC_RELEVE
            notes.append("좌표는 842691 의 relevé 좌표")
        else:
            e = events.get(ev) or events_rel.get(ev)
            lat_s, lon_s, prec = e["lat"], e["lon"], PREC_EVENT
            if is_rv:
                notes.append(f"842691 에 이벤트 {ev} 의 relevé {no} 가 없다. 좌표는 이벤트(연구 지점) 좌표")
            else:
                notes.append("좌표는 이벤트(연구 지점) 좌표다. 측선 자체의 좌표가 아니다(보고서 표 7 에서 측선 끝점은 "
                             "이벤트 좌표에서 약 100 m 안이다)")
        lat, lon = float(lat_s), float(lon_s)
        # 날짜
        date = rv["date"] if rv else ev_date.get(ev, "")
        if rv:
            notes.append("관측일은 842691 의 relevé 날짜")
        elif is_rv:
            notes.append("관측일은 842691 에 있는 같은 이벤트 relevé 의 날짜")
        else:
            notes.append("관측일은 842691 에 있는 같은 이벤트 relevé 의 날짜다. 측선의 관측일은 원자료에 없다")
        md = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date)
        year, month = (int(md.group(1)), int(md.group(2))) if md else ("", "")
        # 값
        alt = float(avg) if avg else None
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
        if is_rv:
            notes.insert(0, f"relevé(식생 방형구) 단일 값. 원표기 '{orig_label or label}'")
        else:
            notes.insert(0, f"측선 탐침 관측 {n}점의 평균. max_cm={vmax}; min_cm={vmin}")
            ok = sd_consistent(n, vmax, vmin, sd)
            if ok is False:
                notes.append(f"N={n} 에서 (최대 - 최소)={float(vmax) - float(vmin):g} cm 와 표준편차 {sd} 가 양립하지 않는다"
                             f"(모표준편차 하한 {(float(vmax) - float(vmin)) / math.sqrt(2 * n):.2f})")
        if rv:
            site_desc = rv["site"]
            dist_t = ",".join(DIST_TYPE.get(x.strip(), x.strip()) for x in rv["dist_type"].split(",") if x.strip())
            notes.append(f"site={site_desc}; micro_site={rv['micro']}; disturbance_degree={rv['dist_deg']}; "
                         f"disturbance_type={dist_t or '없음'}")
            if (ev, no) in rel_fix_notes:
                notes.append(rel_fix_notes[(ev, no)])
        else:
            site_desc = "/".join(sorted(ev_site.get(ev, [])))
        notes.extend(extra_notes)
        dup, dup_note = dup_override, ""
        if not dup:
            dup, dup_note = dup_of_for(lat, lon)
        if dup_note:
            notes.append("중복: " + dup_note)
        kind = "relevé" if is_rv else "transect"
        name = f"{EVENT_NAMES.get(ev, ev)} ({site_desc}) {kind} {label.rstrip('*')}"
        row = {
            "src_id": SRC_ID, "site_id": f"{ev}_{label.rstrip('*')}", "site_name": name,
            "lat": lat_s, "lon": lon_s, "year": year, "month": month, "alt_cm": avg, "method": "probe",
            "label_def": label_def, "n_obs": n, "country": COUNTRY, "macro": macro_of(lat, lon),
            "citation": CITATION_REPORT if from_report else CITATION_THAW,
            "license": LICENSE_REPORT if from_report else LICENSE_PANGAEA,
            "subunit": SUBUNIT, "date": date, "eos_basis": eos_basis, "value_kind": "single_visit",
            "year_min": "", "year_max": "", "alt_sd_cm": sd, "coord_prec_deg": prec,
            "disturbed": 0, "disturb_type": "", "right_censored": 0,
            "orig_source": ORIG_SOURCE, "dup_of": dup,
            "qc_flag": ";".join(flags) if flags else "ok", "notes": "; ".join(notes),
        }
        return row, rv

    # ---- 842711 의 행
    n_raw_thaw = len(thaw)
    for r in thaw:
        ev, label = r["Event"], r["Sample ID (transect/relevé no.)"]
        n = int(r["N [#]"])
        vmax, vmin, avg, sd = (r["Thaw depth [cm] (max)"], r["Thaw depth [cm] (min)"],
                               r["Thaw depth [cm] (average)"], r["Std dev [±]"])
        # 표의 좌표는 이벤트 좌표를 소수 4자리로 반올림한 값이다(반올림 차 0.00005 이하)
        if (abs(float(r["Latitude"]) - float(events[ev]["lat"])) > 0.000051
                or abs(float(r["Longitude"]) - float(events[ev]["lon"])) > 0.000051):
            raise SystemExit(f"{TAB_THAW}: 표의 좌표가 이벤트 좌표와 다르다 {ev} {label}")
        extra, dup = [], ""
        if ev == "Kharasavey1":
            key = label.rstrip("*")
            dup = f"{SRC_ID}:{KH1_PANGAEA_DUP[key]}"
            if key.startswith("T"):
                extra.append("842711 의 Kharasavey1 측선 행이다. 원 보고서 표 20 의 Kharasavey-1 측선은 T36–T40, N=11 이고 "
                             "평균, 최대, 최소, N 이 이 행과 다르다. 표준편차만 순서대로 같아 같은 측선의 잘못 옮겨진 기록으로 "
                             f"보고 보고서 행({KH1_PANGAEA_DUP[key]})을 가리킨다")
            else:
                extra.append("842711 의 Kharasavey1 relevé 행이다. 값이 842711 의 Kharasavey2b 같은 번호 행과 같고, "
                             "842691 과 보고서 표 7, 표 22 는 relevé 47–49 를 Kharasavey-2b 에 둔다. 원 보고서 표 20 의 "
                             "Kharasavey-1 relevé 는 RV-40..44 다")
        row, rv = make_row(ev, label, n, vmax, vmin, avg, sd, dup_override=dup, extra_notes=extra)
        if label.endswith("*"):
            row["notes"] += "; 원표기에 '*' 가 붙어 있다. 의미는 원자료에 적혀 있지 않다"
        if rv:
            td1 = rv["td1"]
            status = cross_status(td1, rv["td2"], avg)
            cross.append({"event": ev, "releve_842711": label, "releve_842691": rv["raw_id"],
                          "thaw_842711_cm": avg, "thaw_842691_cm": td1, "thaw_842691_second_cm": rv["td2"],
                          "thaw_842691_zero_as_missing": int(td1 != "" and float(td1) == 0),
                          "status": status, "date_842691": rv["date"]})
            if status == "differ":
                t = f"{td1}/{rv['td2']}(NSC/Inter)" if rv["td2"] else td1
                row["notes"] += f"; 842691(보고서 표 22)의 값은 {t}로 이 행(842711, 보고서 표 20)과 다르다"
            elif status == "missing_842691":
                t = "0" if td1 != "" else "빈 칸"
                row["notes"] += f"; 842691 의 융해 깊이는 {t}이고 결측으로 본다"
        points.append(row)

    # ---- 원 보고서 표 20 의 Kharasavey-1 행
    for (label, n, vmax, vmin, avg, sd) in KH1_REPORT:
        extra = ["값은 원 보고서 표 20(인쇄 쪽 37)의 Kharasavey-1 블록을 그림에서 옮겨 적은 것이다. "
                 "842711 의 Kharasavey1 블록이 보고서와 달라 이 행을 더했다. 약관은 확인하지 못했다"]
        row, rv = make_row("Kharasavey1", label, n, vmax, vmin, avg, sd, from_report=True, extra_notes=extra)
        if rv:
            td = rv["td1"] + (f"/{rv['td2']}(NSC/Inter)" if rv["td2"] else "")
            if not (rv["td2"] == "" and rv["td1"] != "" and abs(float(rv["td1"]) - float(avg)) < 1e-9):
                row["notes"] += f"; 842691(보고서 표 22)의 값은 {td}로 이 행(보고서 표 20)과 다르다"
            cross.append({"event": "Kharasavey1", "releve_842711": f"{label}(보고서 표 20)",
                          "releve_842691": rv["raw_id"], "thaw_842711_cm": avg, "thaw_842691_cm": rv["td1"],
                          "thaw_842691_second_cm": rv["td2"], "thaw_842691_zero_as_missing": 0,
                          "status": cross_status(rv["td1"], rv["td2"], avg),
                          "date_842691": rv["date"]})
        points.append(row)

    # ---- 842691 에만 있는 relevé 의 융해 깊이(842711 에 없는 이벤트: Nadym)
    thaw_events = {r["Event"] for r in thaw}
    n_rel_only_rows, rel_only_excluded = 0, Counter()
    for (ev, no), rv in sorted(releve.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if ev in thaw_events or ev == "Kharasavey1":
            continue
        td1 = rv["td1"]
        if td1 == "" or float(td1) == 0:
            rel_only_excluded["842691_only_releve_without_thaw"] += 1
            continue
        label = f"RV{no:02d}"
        extra = ["값은 842691(보고서 표 22 의 Mean thaw depth)이다. 842711 에는 이 이벤트가 없다"
                 "(보고서 표 20: 'Nadym-1 (no permafrost)', 'Nadym-2 ... No data from transects')"]
        # make_row 는 relevé 번호로 842691 을 찾으므로 표기를 RV<번호> 로 준다
        row, _ = make_row(ev, label, 1, "", "", td1, "", extra_notes=extra, orig_label=f"{rv['raw_id']}(842691)")
        row["citation"] = CITATION_REL
        if rv["td2"]:
            row["notes"] += f"; 두 번째 융해 깊이 값 {rv['td2']}"
        points.append(row)
        n_rel_only_rows += 1

    ids = [p["site_id"] for p in points]
    if len(set(ids)) != len(ids):
        dup_ids = [k for k, v in Counter(ids).items() if v > 1]
        raise SystemExit(f"site_id 가 중복된다 {dup_ids}")
    # dup_of 가 이 자료 안을 가리키면 대상 행이 있어야 한다
    for p in points:
        if p["dup_of"].startswith(SRC_ID + ":") and p["dup_of"].split(":", 1)[1] not in set(ids):
            raise SystemExit(f"dup_of 대상이 없다 {p['site_id']} -> {p['dup_of']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{SRC_ID}_points.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(points)
    with open(args.raw_dir / "releve_crosscheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader()
        w.writerows(cross)

    # ---- 요약
    def is_target(p):
        return (p["label_def"] == "direct_eos" and p["right_censored"] == 0 and p["qc_flag"] == "ok"
                and p["year"] != "" and p["year"] >= 1990 and p["disturbed"] == 0 and p["dup_of"] == "")

    def cells_blocks(rows):
        cells = {cell_index(float(p["lat"]), float(p["lon"])) for p in rows}
        blocks = {block_index(float(p["lat"]), float(p["lon"])) for p in rows}
        return {"n_rows": len(rows), "n_cells_1km": len(cells), "n_blocks_0p5deg": len(blocks)}

    tgt = [p for p in points if is_target(p)]
    tgt_pangaea_only = [p for p in tgt if p["license"] == LICENSE_PANGAEA]
    # 셀 미리보기(대상 행). 셀 값은 셀 안 위치(site_id)별 값의 평균, 좌표는 위치 좌표의 평균
    cells = defaultdict(list)
    for p in tgt:
        cells[cell_index(float(p["lat"]), float(p["lon"]))].append(p)
    cell_rows = []
    for k, rows in sorted(cells.items()):
        la = statistics.fmean(float(p["lat"]) for p in rows)
        lo = statistics.fmean(float(p["lon"]) for p in rows)
        b = nearest_cheb(la, lo, v3_ref, 3) if v3_ref else None
        dmin = min((haversine_km(la, lo, x[0], x[1]) for x in v3_ref), default=None)
        cell_rows.append({
            "cell_ky_kx": list(k), "lat": round(la, 5), "lon": round(lo, 5),
            "events": sorted({p["site_id"].split("_")[0] for p in rows}),
            "n_positions": len(rows), "n_transects": sum(1 for p in rows if "_T" in p["site_id"]),
            "n_releves": sum(1 for p in rows if "_RV" in p["site_id"]),
            "n_report_rows": sum(1 for p in rows if p["license"] == LICENSE_REPORT),
            "alt_cm_mean_of_positions": round(statistics.fmean(float(p["alt_cm"]) for p in rows), 2),
            "block_0p5deg": block_index(la, lo),
            "v3_f4_chebyshev_deg": round(b[0], 4) if b else None,
            "v3_f4_nearest_km": round(dmin, 1) if dmin is not None else None,
        })
    n_dup_v3 = sum(1 for p in points if p["dup_of"].startswith("v3:"))
    n_dup_calm = sum(1 for p in points if p["dup_of"].startswith("calm_pangaea_v4add:"))
    n_dup_self = sum(1 for p in points if p["dup_of"].startswith(SRC_ID + ":"))
    n_sd_bad = sum(1 for p in points if "양립하지 않는다" in p["notes"])
    n_cross_status = dict(Counter(c["status"] for c in cross))
    n_cross_mismatch = n_cross_status.get("differ", 0)

    files = []
    for name in [TAB_THAW, TAB_REL, JSONLD_THAW, JSONLD_REL, JSONLD_PARENT, PDF_NAME]:
        fp = args.raw_dir / name
        if fp.exists():
            files.append({"name": name, "size_bytes": fp.stat().st_size, "sha256": sha256_of(fp),
                          "url": FILE_URLS[name]})

    parser_path = Path(__file__).resolve()
    meta = {
        "src_id": SRC_ID,
        "name": "Walker 등 2009, 야말 측선 융해 깊이 2007–2008(PANGAEA 842711, 842691)",
        "schema_version": "1.0",
        "url": URL, "doi": DOI, "doi_auxiliary": "10.1594/PANGAEA.842691", "doi_parent": "10.1594/PANGAEA.842707",
        "accessed": ACCESSED,
        "files": files,
        "license": LICENSE_PANGAEA,
        "license_by_rows": dict(Counter(p["license"] for p in points)),
        "license_evidence": {TAB_THAW: lic_thaw, TAB_REL: lic_rel,
                             PDF_NAME: ("약관 표기가 없다(보고서 본문에서 copyright, license 를 찾지 못했다). "
                                        "data/raw 에만 둔다. Kharasavey-1 의 10행(license = unverified)이 이 보고서 값이다")},
        "commit_policy": ("점 자료에 약관 미확인 행(보고서 값 10행)이 있어 같은 폴더의 .gitignore 에 이 점 자료 파일을 올렸다. "
                          "보고서 값의 사용이 확인되면 그 줄을 지운다. 보고서 행을 빼면 나머지는 모두 CC BY 3.0 이다"),
        "citation": CITATION_THAW,
        "citation_auxiliary": [CITATION_REL],
        "orig_source": ORIG_SOURCE,
        "parser": str(parser_path.relative_to(ROOT)),
        "parser_sha256": sha256_of(parser_path),
        "parser_git_commit": git_head(ROOT),
        "parser_git_note": "파서는 실행 시점에 커밋되지 않은 새 파일이다. 위 값은 저장소 HEAD 다",
        "n_rows_raw": n_raw_thaw,
        "n_rows_raw_releve_table": len(rel_rows),
        "n_rows_points": len(points),
        "n_rows_points_by_origin": {
            "842711": n_raw_thaw,
            "report_table20_kharasavey1": len(KH1_REPORT),
            "842691_only_nadym": n_rel_only_rows,
        },
        "n_sites": len(set(ids)),
        "n_events": len({p["site_id"].split("_")[0] for p in points}),
        "n_by_label_def": dict(Counter(p["label_def"] for p in points)),
        "n_by_method": dict(Counter(p["method"] for p in points)),
        "n_by_macro": dict(Counter(p["macro"] for p in points)),
        "n_by_qc_flag": dict(Counter(p["qc_flag"] for p in points)),
        "n_by_event": dict(sorted(Counter(p["site_id"].split("_")[0] for p in points).items())),
        "n_by_date": dict(sorted(Counter(p["date"] for p in points).items())),
        "n_by_unit_kind": {"transect": sum(1 for p in points if "_T" in p["site_id"]),
                           "releve": sum(1 for p in points if "_RV" in p["site_id"])},
        "year_range": [min(p["year"] for p in points), max(p["year"] for p in points)],
        "n_right_censored": sum(p["right_censored"] for p in points),
        "n_disturbed": sum(p["disturbed"] for p in points),
        "n_obs_total": sum(p["n_obs"] for p in points),
        "alt_cm_stats_all_rows": stats_of([float(p["alt_cm"]) for p in points]),
        "alt_cm_stats_target": stats_of([float(p["alt_cm"]) for p in tgt]),
        "alt_cm_stats_target_transects": stats_of([float(p["alt_cm"]) for p in tgt if "_T" in p["site_id"]]),
        "n_excluded_by_reason": {
            "_설명": (f"점 자료에서 지운 행은 없다(842711 의 {n_raw_thaw}행은 모두 점 자료에 있다). 아래 dup_* 는 대상 라벨 규칙에서 빠지는 "
                     "행의 수이고 사유는 겹치지 않는다. not_in_points 는 842691 에만 있고 융해 깊이가 없어 점 자료 행을 "
                     "만들지 않은 relevé 수다"),
            "dropped_from_points": 0,
            "dup_v3_f4_direct_vaskiny_dachi": n_dup_v3,
            "dup_calm_pangaea_v4add_nadym": n_dup_calm,
            "dup_within_source_kharasavey1_pangaea_block": n_dup_self,
            "label_def_not_direct_eos": sum(1 for p in points if p["label_def"] != "direct_eos"),
            "qc_flag_not_ok": sum(1 for p in points if p["qc_flag"] != "ok"),
            "right_censored": 0,
            "disturbed": 0,
            "not_in_points_842691_releve_without_thaw": rel_only_excluded["842691_only_releve_without_thaw"],
            "not_in_points_detail": "Nadym1 relevé 1–5(보고서: no permafrost), Nadym2 relevé 8, 11, 12, 13, 14",
        },
        "n_target_rows": len(tgt),
        "n_target_rows_pangaea_only": len(tgt_pangaea_only),
        "cell_preview": {
            "_설명": ("v4 조립 전의 참고 값이다. 셀 색인은 계획서 6B.3 의 식, 블록은 0.5° 다. 대상 행은 direct_eos, qc ok, "
                     "1990년 이후, 검열·교란 아님, dup_of 없음이다. 셀 값은 셀 안 위치(site_id)별 값의 평균이다"),
            "target_rows": cells_blocks(tgt),
            "target_rows_pangaea_only": cells_blocks(tgt_pangaea_only),
            "cells": cell_rows,
        },
        "kharasavey1_discrepancy": {
            "_설명": ("842711 의 Kharasavey1 블록(측선 T-46..T-50, N=6, relevé RV-47..49)은 원 보고서 표 20 의 Kharasavey-1 "
                     "블록(측선 T36–T40, N=11, relevé RV-40..44)과 다르다. 842711 의 표준편차 5개만 보고서와 순서대로 같다. "
                     "842711 의 relevé RV-47..49 값은 Kharasavey2b 의 같은 번호 행과 같다. 보고서 표 7(GPS)도 Kharasavey-1 의 "
                     "측선을 T36–T40 으로 적었다. 842711 의 T-49, T-50 은 N=6 에서 (최대 - 최소) 와 표준편차가 양립하지 않는다"),
            "pangaea_rows_sd_inconsistent": n_sd_bad,
            "report_rows_added": len(KH1_REPORT),
            "report_values": [dict(zip(["label", "n", "max_cm", "min_cm", "avg_cm", "sd_cm"], x)) for x in KH1_REPORT],
            "transcription": "보고서 PDF 49쪽(인쇄 쪽 37)에 들어 있는 원해상도 그림(594 × 56 화소 띠 16개)을 이어 붙여 눈으로 읽었다",
        },
        "report_visual_check": {
            "_설명": ("842711 의 나머지 이벤트(Laborovaya1, 2, VaskinyDachi1–3, Kharasavey2a, 2b)는 보고서 표 20 과 눈으로 대조해 "
                     "N, 최대, 최소, 평균, 표준편차, relevé 값이 같음을 확인했다. 자동 대조가 아니다"),
            "releve_table20_vs_table22_status": n_cross_status,
            "_status_설명": ("match 는 같은 값, differ 는 다른 값(NSC/Inter 두 값 포함), missing_842691 은 842691 이 빈 칸 또는 0(결측)인 "
                             "경우다"),
            "releve_table20_vs_table22_rows": len(cross),
            "releve_crosscheck_file": f"data/raw/{SRC_ID}/releve_crosscheck.csv",
            "note_table22": ("보고서 표 22(=842691)의 relevé 융해 깊이는 표 20(=842711)과 여러 relevé 에서 다르다"
                             "(VaskinyDachi1 25, VaskinyDachi2 30–34, Kharasavey 40–44, 47, 48). 보고서 안의 두 표가 다르다. "
                             "842691 은 보고서 표 22 와 두 곳에서 다르다(relevé 17 은 보고서 91, 842691 빈 칸. relevé 20 은 "
                             "보고서 118, 842691 116). 라벨은 융해 깊이 전용 표인 표 20(842711)의 값을 쓴다"),
        },
        "releve_id_fixes": {f"{k[0]}:{k[1]}": v[1] for k, v in RELEVE_ID_FIX.items()},
        "v3_check": {
            "n_v3_f4_ref_compared": len(v3_ref), "bbox_compared": "lat 60–75, lon 55–90",
            "n_calm_pangaea_v4add_sites_compared": len(calm_ref),
            "rows_dup_v3": n_dup_v3, "rows_dup_calm_pangaea_v4add": n_dup_calm,
        },
        "sampling_periods_report": {"Nadym": "2007-08-03–10", "Laborovaya": "2007-08-13–21",
                                    "Vaskiny Dachi": "2007-08-21–30", "Kharasavey": "2008-08-18–25"},
        "method_evidence": ("보고서 방법 절(인쇄 쪽 15): 'The active layer summer thaw depth was measured at 1-m intervals along "
                            "each transect using a 2-m long steel probe.' 표 7 은 Laborovaya, Vaskiny Dachi 측선을 50 m, "
                            "Kharasavey 측선을 10 m 로 적었다"),
        "coordinate_conversion": ("변환 없음. 측선 행은 842711 머리말의 이벤트 좌표(소수 6자리, 끝자리 0), relevé 행은 842691 의 "
                                  "relevé 좌표(소수 4자리)를 문자열 그대로 옮겼다. 동경 양수다. 측지계 표기는 원자료에 없다"),
        "units": "융해 깊이는 cm 다. 변환하지 않았다",
        "missing_value_marks": "PANGAEA 표는 빈 칸이다. 842691 의 0 은 결측으로 본다. 보고서 표 22 는 NA 와 '?' 다",
        "unverified_items": [
            "측선의 관측일. 원자료는 relevé 날짜만 적었다. 같은 이벤트 relevé 의 날짜를 썼다(모두 8월이고 보고서의 지점별 "
            "조사 기간 안이다)",
            "측선 관측 간격. 방법 절은 1 m 간격이라 적었으나 50 m 측선의 N 은 11(Laborovaya1 T09 는 31, 일부 측선은 5, 8, 10)이다",
            "Kharasavey-1 의 보고서 값(10행)의 이용 약관. 보고서에 약관 표기가 없다",
            "842711 의 Kharasavey1 블록 값(N=6, 평균 73.2–81.7 cm)의 출처. 보고서 표 20 어디에도 같은 값이 없다",
            "보고서 표 20 과 표 22 의 relevé 값이 다른 이유",
            "Laborovaya2 T18 의 최솟값 5 cm 와 표준편차 60.4 cm(N=5)의 사유",
            "842711 의 'RV-49*' 에 붙은 '*' 의 의미",
            "좌표의 측지계. 원자료에 표기가 없다. GPS 좌표로 보고 WGS84 로 다루었다",
            "교란 표시. relevé 표의 교란 코드(순록 발자국, 차량 흔적 등)는 스키마의 교란 범주(연소, 수체, 열침식, 시설)가 "
            "아니라 disturbed = 0 으로 두고 notes 에 적었다. 측선의 교란 정보는 원자료에 없다",
        ],
    }
    with open(args.out_dir / f"{SRC_ID}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---- 화면 요약
    print(f"[{SRC_ID}] 원자료 {n_raw_thaw}행(842711) + relevé 표 {len(rel_rows)}행(842691), 점 자료 {len(points)}행, "
          f"지점 {len(set(ids))}개")
    print(f"  행 출처 {meta['n_rows_points_by_origin']}")
    print(f"  label_def {meta['n_by_label_def']}, method {meta['n_by_method']}, macro {meta['n_by_macro']}")
    print(f"  qc_flag {meta['n_by_qc_flag']}, 라이선스 {meta['license_by_rows']}")
    print(f"  날짜 {meta['n_by_date']}")
    print(f"  이벤트 {meta['n_by_event']}")
    print(f"  ALT(전체) {meta['alt_cm_stats_all_rows']}")
    print(f"  ALT(대상) {meta['alt_cm_stats_target']}")
    print(f"  제외 {meta['n_excluded_by_reason']}")
    print(f"  대상 행 {len(tgt)} (PANGAEA 행만 {len(tgt_pangaea_only)})")
    print(f"  셀 {meta['cell_preview']['target_rows']} / PANGAEA 만 {meta['cell_preview']['target_rows_pangaea_only']}")
    for c in cell_rows:
        print(f"    {c}")
    print(f"  relevé 표 대조 {n_cross_status} (전체 {len(cross)}), SD 불일치 측선 {n_sd_bad}")
    print(f"  출력 {out_csv}")


if __name__ == "__main__":
    main()
