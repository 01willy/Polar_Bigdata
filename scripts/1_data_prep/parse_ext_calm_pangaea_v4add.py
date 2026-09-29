"""calm_pangaea_v4add: CALM PANGAEA 972777 의 v3 미편입 탐침 이벤트 9개를 표준 점 자료로 만든다.

근거
----
계획서 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 절(LGD, 개정 10), 형식 정의
`data/processed/ext_labels/_schema.json`(버전 1.0).

입력
----
data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab (PANGAEA 탭 구분 텍스트, 머리말 주석 뒤 자료 행렬)
data/processed/fidelity_base_v3.csv (읽기 전용. 좌표와 source_id 만 읽어 v3 와의 관계를 기록한다)

규칙
----
1. 대상은 TARGET_EVENTS 의 9개 이벤트다. 이벤트 머리말의 Method 가
   'Spatially-oriented mechanical probing' 으로 시작하는지 확인한다.
2. 융해관 이벤트 C3A 와 지온 이벤트(CH6, M3, CN6, U20)는 넣지 않는다.
3. 연 값은 ALD [cm] 열이다. 자료 행렬의 Date/Time 은 연도(4자리)뿐이므로 month, date 는 빈 칸이다.
4. 계절 말 근거는 CALM 규약(eos_basis = protocol)이다. ALD 변수 설명은
   'maximum thaw depth at the end of the summer' 다.
5. Sample comment 가 이른 관측을 적은 행은 label_def = direct_dated, eos_basis = none,
   qc_flag = date 로 둔다(지우지 않는다).
6. ALD 가 빈 행(inactive 등)은 값이 없으므로 점 자료에 넣지 않고 제외 수에 센다.
7. '>' 표기는 right_censored = 1 로 둔다. '<' 표기는 대상 이벤트에 없어야 한다(있으면 중단).
8. 1990년 이전은 qc_flag 에 year_pre1990, 범위(0 < ALT <= 600 cm) 밖은 range 를 붙인다.
9. 좌표는 이벤트 머리말의 값(소수 6자리 표기)을 쓰고 자료 행렬의 값(소수 4자리)과
   반올림 범위에서 같은지 확인한다. 좌표 변환은 없다(원자료가 십진 도, 서경 음수).

산출
----
data/processed/ext_labels/calm_pangaea_v4add_points.csv
data/processed/ext_labels/calm_pangaea_v4add_meta.json

실행(ROOT): OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_calm_pangaea_v4add.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ID = "calm_pangaea_v4add"
SRC_NAME = "CALM PANGAEA 972777 의 v3 미편입 탐침 이벤트 9개"
URL = "https://doi.org/10.1594/PANGAEA.972777"
DOI = "10.1594/PANGAEA.972777"
ACCESSED = "2026-09-29"
LICENSE = "CC BY 4.0"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

TARGET_EVENTS = ["CALM_R27", "CALM_R41", "CALM_R30A", "CALM_R1", "CALM_N3",
                 "CALM_IT1", "CALM_C3B", "CALM_U5", "CALM_U4"]
# 규칙으로 넣지 않는 이벤트(후보 목록에 있었으나 탐침이 아니다)
RULE_EXCLUDED = {"CALM_C3A": "thaw_tube", "CALM_CH6": "borehole_temp", "CALM_M3": "borehole_temp",
                 "CALM_CN6": "borehole_temp", "CALM_U20": "borehole_temp"}
PROBE_PREFIX = "Spatially-oriented mechanical probing"

YEAR_MIN = 1990
ALT_MAX_CM = 600.0
CELL_DEG = 0.009           # 약 1 km 셀(6B.3)
DUP_DEG = 0.01             # v3 F4_direct 와의 중복 기준(체비쇼프)

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
            "label_def", "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
            "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
            "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

# v3 의 region 을 6B.4 의 대상 지역 이름으로 옮기는 표(독립성 거리 확인용, 참고 값)
V3_REGION_TO_MACRO = {
    "ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska",
    "ABoVE_CA": "Canada", "Canada": "Canada", "CALM_Canada": "Canada",
    "Lena_RU": "Lena",
    "CALM_Greenland": "NAtlantic", "CALM_Svalbard": "NAtlantic", "CALM_Scandinavia": "NAtlantic",
    "CALM_Russia_W": "Russia_W", "CALM_Russia_C": "Russia_C", "CALM_Russia_E": "Russia_E",
    "QTP_CN": "Tibet", "CALM_QTP_China": "Tibet",
}
INDEP_KM = 100.0           # 독립성 기준(6B.4)

COUNTRY_MAP = {"Russia": "Russia", "Canada": "Canada", "United States (Alaska)": "United States",
               "Norway/Svalbard": "Norway", "Italy/Svalbard": "Norway"}


# ---------------------------------------------------------------- 원자료 읽기
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pangaea_tab(path: Path):
    """머리말 주석 행, 열 이름, 자료 행을 돌려준다."""
    lines = path.read_text(encoding="utf-8").splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "*/")
    head = lines[:end]
    cols = lines[end + 1].split("\t")
    rows = []
    for l in lines[end + 2:]:
        if not l.strip():
            continue
        r = l.split("\t")
        if len(r) != len(cols):
            raise ValueError(f"열 수 불일치: {len(r)} != {len(cols)}: {l[:80]}")
        rows.append(r)
    return head, cols, rows


def header_field(head, key: str) -> str:
    for l in head:
        if l.startswith(key + ":"):
            return l.split("\t", 1)[1].strip()
    return ""


def parse_events(head) -> dict:
    """머리말의 Event(s) 구역을 이벤트별 사전으로 만든다. 좌표는 문자열 그대로 둔다."""
    ev = {}
    inside = False
    for l in head:
        if l.startswith("Event(s):"):
            inside = True
        elif inside and not l.startswith("\t"):
            break
        if not inside:
            continue
        body = l.split("\t", 1)[1].strip()
        m = re.match(r"^(CALM_\S+)(?:\s+\(([^)]*)\))?", body)
        if not m:
            continue
        eid = m.group(1)
        label = m.group(2) or ""      # 표지와 Method 가 없는 이벤트가 있다(예: CALM_C13)
        d = dict(event=eid, label=label, uri="", lat_str="", lon_str="", elev_m="", location="",
                 pi="", method_raw="")
        u = re.search(r"\(URI: ([^)]+)\)", body)
        if u:
            d["uri"] = u.group(1).strip()
        for part in body.split(" * ")[1:]:
            k, _, v = part.partition(": ")
            k, v = k.strip(), v.strip()
            if k == "LATITUDE":
                d["lat_str"] = v
            elif k == "LONGITUDE":
                d["lon_str"] = v
            elif k == "ELEVATION":
                d["elev_m"] = v.replace(" m", "")
            elif k == "LOCATION":
                d["location"] = v
            elif k == "COMMENT":
                mm = re.match(r"^PI: (.*), Method: (.*)$", v)
                if mm:
                    d["pi"], d["method_raw"] = mm.group(1).strip(), mm.group(2).strip()
                else:
                    mm = re.match(r"^Method: (.*)$", v)
                    if mm:
                        d["method_raw"] = mm.group(1).strip()
        ev[eid] = d
    return ev


# ---------------------------------------------------------------- 값 해석
def strip_zeros(s: str) -> str:
    """'65.600000' 을 '65.6' 으로 만든다(값은 바꾸지 않는다)."""
    if "." not in s:
        return s
    s = s.rstrip("0")
    return s + "0" if s.endswith(".") else s


def n_decimals(s: str) -> int:
    s = strip_zeros(s)
    return 0 if s.endswith(".0") else len(s.split(".")[1])


def is_whole_minute(s: str, tol_min: float = 0.005) -> bool:
    x = abs(float(s))
    mins = (x - math.floor(x)) * 60.0
    return abs(mins - round(mins)) <= tol_min


def coord_precision(lat_s: str, lon_s: str) -> float:
    """표기 자릿수에서 추정한 좌표 정밀도(도). 위경도가 모두 분 단위 값이면 0.0167 이다."""
    if is_whole_minute(lat_s) and is_whole_minute(lon_s):
        return 0.0167
    return float(10.0 ** (-max(n_decimals(lat_s), n_decimals(lon_s))))


def fmt_prec(x: float) -> str:
    """정밀도를 지수 표기 없이 적는다(예: 0.000001)."""
    return f"{x:.6f}".rstrip("0")


def parse_ald(s: str):
    """ALD 문자열을 (값, 부호 표기)로 바꾼다. 빈 칸이면 (None, '')."""
    s = s.strip()
    if s == "":
        return None, ""
    sign = ""
    if s[0] in "<>":
        sign, s = s[0], s[1:].strip()
    return float(s), sign


def macro_rule(country_raw: str, lat: float, lon: float) -> str:
    """6B.4 의 지리 규칙. 라벨 값을 쓰지 않는다."""
    c = str(country_raw)
    if "Alaska" in c:
        return "Alaska"
    if c == "Canada":
        return "Canada"
    if "Svalbard" in c or "Greenland" in c or c in ("Norway", "Sweden", "Finland"):
        return "NAtlantic"
    if c == "Russia":
        if lon < 0:
            return "Russia_E"      # 추코트카(서경 표기)
        if lon < 90:
            return "Russia_W"
        if lon < 140:
            if 71.5 <= lat <= 73.6 and 123.3 <= lon <= 130.1:
                return "Lena"
            return "Russia_C"
        return "Russia_E"
    return "other"


def classify_sample_comment(txt: str) -> dict:
    """Sample comment 를 표시로 바꾼다. 해당 없으면 빈 사전."""
    t = txt.strip().lower()
    out = {}
    if t == "":
        return out
    if re.search(r"\bearly\b", t) or "july" in t:
        out["early"] = True
    if "flooded" in t:
        out["disturb_type"] = "water"
    if "burned" in t:
        out["disturb_type"] = "burned"
    if "not reached" in t:
        out["censored"] = True
    return out


def grid_support(method_raw: str) -> str:
    m = re.search(r"at (\d+) m Grid", method_raw)
    return m.group(1) if m else ""


def haversine_km(lat0: float, lon0: float, lat, lon):
    r = 6371.0088
    p0, p1 = np.radians(lat0), np.radians(np.asarray(lat, dtype=float))
    dl = np.radians(np.asarray(lon, dtype=float) - lon0)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return int(ky), int(kx)


# ---------------------------------------------------------------- 점 자료
def build_points(head, cols, rows, events):
    citation = header_field(head, "Citation")
    raw = pd.DataFrame(rows, columns=cols)
    n_raw = len(raw)
    excluded = {}

    # 대상 이벤트 확인
    for e in TARGET_EVENTS:
        if e not in events:
            raise ValueError(f"이벤트 머리말에 없음: {e}")
        if not events[e]["method_raw"].startswith(PROBE_PREFIX):
            raise ValueError(f"탐침 이벤트가 아님: {e}: {events[e]['method_raw']}")
    for e, kind in RULE_EXCLUDED.items():
        if events[e]["method_raw"].startswith(PROBE_PREFIX):
            raise ValueError(f"제외 이벤트의 Method 가 탐침임: {e}")
        key = "융해관 이벤트(C3A)" if kind == "thaw_tube" else "지온 이벤트(CH6, M3, CN6, U20)"
        excluded[key] = excluded.get(key, 0) + int((raw["Event"] == e).sum())

    tgt = raw[raw["Event"].isin(TARGET_EVENTS)].copy()
    n_other = n_raw - len(tgt) - sum(excluded.values())
    excluded["대상 외 이벤트(v3 편입 셀이거나 탐침이 아닌 이벤트)"] = int(n_other)

    dup = int(tgt.duplicated(["Event", "Date/Time"]).sum())
    if dup:
        raise ValueError(f"(이벤트, 연도) 중복 {dup}건")

    out, sites, n_missing = [], [], 0
    for e in TARGET_EVENTS:
        ev = events[e]
        sub = tgt[tgt["Event"] == e]
        lat_s, lon_s = strip_zeros(ev["lat_str"]), strip_zeros(ev["lon_str"])
        lat, lon = float(lat_s), float(lon_s)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"좌표 범위 밖: {e}")
        mlat, mlon = set(sub["Latitude"]), set(sub["Longitude"])
        if len(mlat) != 1 or len(mlon) != 1:
            raise ValueError(f"자료 행렬의 좌표가 행마다 다름: {e}")
        mlat_s, mlon_s = mlat.pop(), mlon.pop()
        if abs(float(mlat_s) - lat) > 5.1e-5 or abs(float(mlon_s) - lon) > 5.1e-5:
            raise ValueError(f"머리말 좌표와 자료 행렬 좌표 불일치: {e}")
        country_raw = sub["Country"].iloc[0].strip()
        area = sub["Area"].iloc[0].strip()
        name = sub["Name"].iloc[0].strip().strip('"').strip()
        macro = macro_rule(country_raw, lat, lon)
        prec = coord_precision(ev["lat_str"], ev["lon_str"])
        support = grid_support(ev["method_raw"])
        ky, kx = cell_index(lat, lon)
        sites.append(dict(site_id=e, site_name=name, lat=lat_s, lon=lon_s, lat_matrix=mlat_s, lon_matrix=mlon_s,
                          elev_m=ev["elev_m"], country=COUNTRY_MAP[country_raw], country_raw=country_raw,
                          area=area, macro=macro, method_raw=ev["method_raw"], grid_m=support, pi=ev["pi"],
                          uri=ev["uri"], gtnp_id=sub["ID"].iloc[0].strip(), coord_prec_deg=prec,
                          cell_ky=ky, cell_kx=kx, n_rows_raw=int(len(sub))))
        for _, r in sub.iterrows():
            year_s = r["Date/Time"].strip()
            if not re.fullmatch(r"\d{4}", year_s):
                raise ValueError(f"Date/Time 이 연도 4자리가 아님: {e}: {year_s}")
            year = int(year_s)
            val, sign = parse_ald(r["ALD [cm]"])
            sc, cc = r["Sample comment"].strip(), r["Comment"].strip()
            if val is None:
                n_missing += 1
                continue
            if sign == "<":
                raise ValueError(f"대상 이벤트에 '<' 표기: {e} {year}")
            flags = classify_sample_comment(sc)
            qc = []
            if year < YEAR_MIN:
                qc.append("year_pre1990")
            if not (0 < val <= ALT_MAX_CM):
                qc.append("range")
            label_def, eos = "direct_eos", "protocol"
            if flags.get("early"):
                label_def, eos = "direct_dated", "none"
                qc.append("date")
            censored = 1 if (sign == ">" or flags.get("censored")) else 0
            dtype = flags.get("disturb_type", "")
            notes = [f"grid_m={support}" if support else "grid_m=unknown"]
            if ev["elev_m"]:
                notes.append(f"elev_m={ev['elev_m']}")
            notes.append(f"country_raw={country_raw}")
            if sc:
                notes.append(f"sample_comment={sc}")
            if cc:
                notes.append(f"cci_comment={cc}")
            out.append(dict(
                src_id=SRC_ID, site_id=e, site_name=name, lat=lat_s, lon=lon_s, year=year, month="",
                alt_cm=val, method="probe", label_def=label_def, n_obs=1,
                country=COUNTRY_MAP[country_raw], macro=macro, citation=citation, license=LICENSE,
                subunit=area, date="", eos_basis=eos, value_kind="annual_value", year_min="", year_max="",
                alt_sd_cm="", coord_prec_deg=fmt_prec(prec), disturbed=1 if dtype else 0, disturb_type=dtype,
                right_censored=censored,
                orig_source=f"CALM site {ev['label'].split(',')[0].strip()} ({ev['uri']}); PI: {ev['pi']}",
                dup_of="", qc_flag=";".join(qc) if qc else "ok", notes="; ".join(notes)))
    excluded["ALD 결측(Sample comment: inactive)"] = int(n_missing)
    pts = pd.DataFrame(out, columns=COLUMNS)
    return pts, pd.DataFrame(sites), n_raw, len(tgt), excluded, citation


# ---------------------------------------------------------------- v3 대조
def v3_relation(sites: pd.DataFrame, raw_rows, cols, events) -> dict:
    """v3 표(읽기 전용)와의 관계. 새 셀 제외 기준은 F4_direct 셀과 체비쇼프 0.01° 이내다."""
    v3 = pd.read_csv(V3, usecols=["loc_id", "lat", "lon", "region", "alt_cm", "source_id"])
    f4 = v3[v3["source_id"] == "F4_direct"].copy()
    f4["macro"] = f4["region"].map(lambda r: V3_REGION_TO_MACRO.get(r, "other"))
    rel = []
    for _, s in sites.iterrows():
        la, lo = float(s["lat"]), float(s["lon"])
        d_all = np.maximum((v3["lat"] - la).abs(), (v3["lon"] - lo).abs())
        d_f4 = np.maximum((f4["lat"] - la).abs(), (f4["lon"] - lo).abs())
        near = v3[d_all <= DUP_DEG + 1e-9].assign(cheb=d_all[d_all <= DUP_DEG + 1e-9])
        km = pd.Series(haversine_km(la, lo, f4["lat"].values, f4["lon"].values), index=f4.index)
        other = km[f4["macro"] != s["macro"]]
        rel.append(dict(
            site_id=s["site_id"],
            nearest_f4_direct_km=round(float(km.min()), 1),
            nearest_f4_direct_km_macro=str(f4.loc[km.idxmin(), "macro"]),
            nearest_other_macro_f4_direct_km=round(float(other.min()), 1),
            nearest_other_macro=str(f4.loc[other.idxmin(), "macro"]),
            within_100km_of_other_macro=bool(other.min() < INDEP_KM),
            nearest_f4_direct_cheb_deg=round(float(d_f4.min()), 4),
            nearest_f4_direct_loc_id=int(f4.loc[d_f4.idxmin(), "loc_id"]),
            nearest_f4_direct_region=str(f4.loc[d_f4.idxmin(), "region"]),
            dropped_by_v3_f4_rule=bool(d_f4.min() <= DUP_DEG),
            v3_cells_within_0p01=[dict(loc_id=int(r.loc_id), region=str(r.region), source_id=str(r.source_id),
                                       alt_cm=round(float(r.alt_cm), 1), cheb_deg=round(float(r.cheb), 4))
                                  for r in near.sort_values("cheb").itertuples()]))
    # 대상 집합 검증: 유효 값이 있는 탐침 이벤트 가운데 v3 F4_direct 와 0.01° 이내가 아닌 것
    raw = pd.DataFrame(raw_rows, columns=cols)
    raw["v"] = pd.to_numeric(raw["ALD [cm]"], errors="coerce")
    ok = raw[(raw["v"] > 0) & (raw["v"] <= ALT_MAX_CM) & (raw["Date/Time"].astype(int) >= YEAR_MIN)]
    found = []
    for e, g in ok.groupby("Event"):
        if not events[e]["method_raw"].startswith(PROBE_PREFIX):
            continue
        la, lo = float(g["Latitude"].iloc[0]), float(g["Longitude"].iloc[0])
        d = np.maximum((f4["lat"] - la).abs(), (f4["lon"] - lo).abs()).min()
        if d > DUP_DEG:
            found.append(e)
    check = dict(rule="ALD 유효 값(0 < ALT <= 600, 1990년 이후)이 있는 탐침 이벤트 가운데 v3 F4_direct 셀과 "
                      "체비쇼프 0.01° 이내가 아닌 이벤트(자료 행렬 좌표 기준)",
                 found=sorted(found), target=sorted(TARGET_EVENTS),
                 identical=sorted(found) == sorted(TARGET_EVENTS))
    return dict(per_site=rel, target_set_check=check)


# ---------------------------------------------------------------- 요약
def summarize(pts: pd.DataFrame, sites: pd.DataFrame) -> dict:
    def dist(s):
        return dict(n=int(len(s)), min=float(s.min()), median=float(s.median()), max=float(s.max()),
                    mean=round(float(s.mean()), 2))
    main = pts[(pts["label_def"] == "direct_eos") & (pts["qc_flag"] == "ok")
               & (pts["right_censored"] == 0) & (pts["disturbed"] == 0)]
    per_site = []
    for _, s in sites.iterrows():
        a, m = pts[pts["site_id"] == s["site_id"]], main[main["site_id"] == s["site_id"]]
        per_site.append(dict(site_id=s["site_id"], site_name=s["site_name"], macro=s["macro"],
                             n_rows=int(len(a)), n_main=int(len(m)),
                             year_min=int(m["year"].min()), year_max=int(m["year"].max()),
                             alt_mean_main_cm=round(float(m["alt_cm"].mean()), 1),
                             alt_min_main_cm=float(m["alt_cm"].min()), alt_max_main_cm=float(m["alt_cm"].max())))
    site_mean = main.groupby("site_id")["alt_cm"].mean()
    return dict(
        n_sites=int(pts["site_id"].nunique()), n_rows=int(len(pts)), n_rows_main=int(len(main)),
        main_set_rule="label_def = direct_eos, qc_flag = ok, right_censored = 0, disturbed = 0",
        n_sites_by_macro={k: int(v) for k, v in sites.groupby("macro").size().items()},
        n_rows_by_macro={k: int(v) for k, v in pts.groupby("macro").size().items()},
        n_rows_by_method={k: int(v) for k, v in pts.groupby("method").size().items()},
        n_rows_by_qc_flag={k: int(v) for k, v in pts.groupby("qc_flag").size().items()},
        n_right_censored=int(pts["right_censored"].sum()), n_disturbed=int(pts["disturbed"].sum()),
        year_range=[int(pts["year"].min()), int(pts["year"].max())],
        alt_cm_rows_all=dist(pts["alt_cm"]), alt_cm_rows_main=dist(main["alt_cm"]),
        alt_cm_site_multiyear_mean=dict(n=int(len(site_mean)), min=round(float(site_mean.min()), 1),
                                        median=round(float(site_mean.median()), 1),
                                        max=round(float(site_mean.max()), 1)),
        n_distinct_1km_cells=int(sites[["cell_ky", "cell_kx"]].drop_duplicates().shape[0]),
        per_site=per_site)


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-v3", action="store_true", help="v3 표 대조를 건너뛴다")
    args = ap.parse_args()

    head, cols, rows = read_pangaea_tab(RAW)
    events = parse_events(head)
    pts, sites, n_raw, n_tgt, excluded, citation = build_points(head, cols, rows, events)

    # 형식 확인
    assert list(pts.columns) == COLUMNS
    assert pts[REQUIRED].drop(columns=["month"]).notna().all().all()
    assert (pts[REQUIRED].drop(columns=["month"]).astype(str) != "").all().all()
    assert not pts.duplicated(["site_id", "year"]).any()
    assert n_tgt == len(pts) + excluded["ALD 결측(Sample comment: inactive)"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{SRC_ID}_points.csv"
    pts.to_csv(out_csv, index=False, encoding="utf-8")

    rel = None if args.no_v3 else v3_relation(sites, rows, cols, events)
    summ = summarize(pts, sites)
    head_commit = git_head()
    meta = dict(
        src_id=SRC_ID, name=SRC_NAME, url=URL, doi=DOI, accessed=ACCESSED,
        files=[dict(name=str(RAW.relative_to(ROOT)), size_bytes=RAW.stat().st_size, sha256=sha256_of(RAW),
                    source_url="https://doi.pangaea.de/10.1594/PANGAEA.972777?format=textfile")],
        license=LICENSE, license_basis="원자료 머리말의 License 행: " + header_field(head, "License"),
        citation=citation,
        parser=f"scripts/1_data_prep/parse_ext_{SRC_ID}.py",
        parser_sha256=sha256_of(Path(__file__).resolve()),
        parser_git_commit=head_commit,
        parser_git_note="파서 파일은 이 커밋에 들어 있지 않다(작업 트리의 새 파일). 값은 실행 시점의 HEAD 다",
        schema_version="1.0",
        n_rows_raw=int(n_raw), n_rows_raw_target_events=int(n_tgt), n_events_raw=int(len(events)),
        n_rows_points=int(len(pts)),
        n_by_label_def={k: int(v) for k, v in pts.groupby("label_def").size().items()},
        n_by_method={k: int(v) for k, v in pts.groupby("method").size().items()},
        n_excluded_by_reason=excluded,
        coordinate_conversion="변환 없음. 원자료가 WGS84 십진 도이고 서경은 음수다. lat, lon 은 이벤트 머리말의 값"
                              "(소수 6자리 표기, 뒤의 0 은 뗐다)이고 자료 행렬의 값(소수 4자리)과 반올림 범위에서 "
                              "같음을 확인했다. 추코트카 R27, R41 의 경도는 원자료에서 음수다",
        coord_prec_rule="위경도가 모두 분 단위 값이면 0.0167, 아니면 10^-(표기 소수 자릿수). 표기에서 추정한 값이다",
        units="ALD [cm]. 단위 변환 없음",
        missing_code="빈 칸. 대상 이벤트의 결측 2행은 Sample comment 가 inactive 다",
        season_end_basis="CALM 규약(eos_basis = protocol). ALD 변수 설명: maximum thaw depth at the end of the summer. "
                         "자료 행렬의 Date/Time 은 연도뿐이라 month 와 date 는 빈 칸이다",
        qc_summary=summ, sites=sites.to_dict(orient="records"), v3_relation=rel,
        unverified_items=[
            "행별 관측 날짜와 월. 자료 행렬에는 연도만 있다",
            "격자 노드 수. 자료에 없어 n_obs 는 1 로 두었다",
            "행 안 표준편차(alt_sd_cm). 자료에 없다",
            "좌표의 실제 위치 정확도. coord_prec_deg 는 표기 자릿수에서 추정했다. R27 과 U5 는 분 단위 표기다",
            "격자 중심과 표기 좌표의 관계(격자 모서리인지 중심인지)",
            "ERA5-Land 육지 폴백, CCI ALT 유효 여부, 공변량 부착 결과. v4 조립 단계의 일이다",
            "R1 2009년 'Early measurement' 의 실제 관측일",
        ])
    out_meta = OUT_DIR / f"{SRC_ID}_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[원자료] 행 {n_raw}, 이벤트 {len(events)}, 대상 이벤트 행 {n_tgt}")
    print(f"[점 자료] {out_csv.relative_to(ROOT)}: 행 {len(pts)}, 지점 {summ['n_sites']}")
    print("[제외]", json.dumps(excluded, ensure_ascii=False))
    print("[label_def]", meta["n_by_label_def"], "[method]", meta["n_by_method"])
    print("[macro 지점]", summ["n_sites_by_macro"], "[macro 행]", summ["n_rows_by_macro"])
    print("[연도]", summ["year_range"], "[ALT 전체 행]", summ["alt_cm_rows_all"])
    print("[ALT 주 집합 행]", summ["alt_cm_rows_main"])
    print("[ALT 지점 다년 평균]", summ["alt_cm_site_multiyear_mean"])
    print(pd.DataFrame(summ["per_site"]).to_string(index=False))
    if rel is not None:
        print("[v3 대조] 대상 집합 일치:", rel["target_set_check"]["identical"],
              "| 찾은 이벤트:", rel["target_set_check"]["found"])
        for r in rel["per_site"]:
            print("  ", r["site_id"], "F4_direct 최근접", r["nearest_f4_direct_cheb_deg"], "°",
                  r["nearest_f4_direct_region"], f"({r['nearest_f4_direct_km']} km)",
                  "| 다른 macro 최근접", r["nearest_other_macro"], r["nearest_other_macro_f4_direct_km"], "km",
                  "| 0.01° 이내 v3 셀:",
                  [(c["loc_id"], c["source_id"], c["alt_cm"], c["cheb_deg"]) for c in r["v3_cells_within_0p01"]])


if __name__ == "__main__":
    main()
