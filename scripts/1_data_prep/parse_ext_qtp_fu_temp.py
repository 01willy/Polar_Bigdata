#!/usr/bin/env python3
"""외부 라벨 자료원 qtp_fu_temp 의 파서(계획서 6B, LGD).

자료: Fu, Ziteng (2025). Permafrost ground temperature and active layer thickness data
      on the Qinghai-Tibet Plateau (2001-2020). figshare. doi:10.6084/m9.figshare.29206613.v1
      약관 CC BY 4.0.

입력(data/raw/qtp_fu_temp/)
- Permafrost_Profile_QTP_2001-2020.xlsx
  - 시트 SITE: 지점 54개. 열은 Site, Region, Latitude, Longitude, Altitude (m) 와 지표 속성.
    좌표는 십진 도 수치이거나 수식이다(도 + 분/60, 도 + (분 + 초/60)/60).
  - 시트 ALT: 첫 열 Year(2001-2020), 나머지 열은 지점 코드 55개. 결측은 문자열 '####'.
    README 의 단위 표기는 'Centimeters (m)' 로 모순이다. 값의 범위가 0.82-7.8 이므로 m 로 본다.
- README.txt

산출
- data/processed/ext_labels/qtp_fu_temp_points.csv  (형식 _schema.json 1.0, 행은 (지점, 연도))
- data/processed/ext_labels/qtp_fu_temp_meta.json

라벨 규칙(계획서 6B.2 보조 라벨)
- method = borehole_temp, label_def = temp_derived, eos_basis = none, value_kind = annual_value.
- alt_cm = 시트 ALT 의 값(m) x 100.
- 이 자료원은 확인적 풀에 넣지 않는다. L39 의 비교 대상이다.

QC 규칙
- range: alt_cm 이 0 초과 600 이하가 아닌 행.
- coord: (R1) 보고 고도와 Copernicus DEM(GLO-30) 고도의 차가 50 m 를 넘는 지점.
         (R2) 위도 또는 경도가 다른 지점과 1e-5 도 안에서 같은 지점 쌍 가운데
              DEM 고도 차의 절댓값이 큰 쪽. DEM 을 읽지 못하면 쌍의 두 지점 모두.
- 좌표와 값은 고치지 않는다. 표시만 한다.

실행(저장소 루트에서, 스레드 1개)
  OMP_NUM_THREADS=1 /home/anaconda3/bin/python3 scripts/1_data_prep/parse_ext_qtp_fu_temp.py
읽기 전용 입력: data/processed/fidelity_base_v3.csv(근접 셀 확인), data/raw/dem/(고도 대조).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import ast
import csv
import datetime
import hashlib
import itertools
import json
import math
import re
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
SRC_ID = "qtp_fu_temp"
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
XLSX = RAW_DIR / "Permafrost_Profile_QTP_2001-2020.xlsx"
README = RAW_DIR / "README.txt"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
DEM_DIR = ROOT / "data" / "raw" / "dem"

ACCESSED = "2026-09-29"
URL = "https://doi.org/10.6084/m9.figshare.29206613.v1"
DOI = "10.6084/m9.figshare.29206613.v1"
LICENSE = "CC BY 4.0"
CITATION = (
    "Fu, Ziteng (2025). Permafrost ground temperature and active layer thickness data on the "
    "Qinghai–Tibet Plateau (2001–2020). figshare. Dataset. "
    "https://doi.org/10.6084/m9.figshare.29206613.v1"
)
# figshare API(2026-09-29) 가 돌려준 파일별 md5. 로컬 사본과 대조한다.
FIGSHARE_FILES = {
    "Permafrost_Profile_QTP_2001-2020.xlsx": {
        "url": "https://ndownloader.figshare.com/files/55143119",
        "md5": "3c7218c5eb343450cd13ff91a151315e",
        "size": 69157,
    },
    "README.txt": {
        "url": "https://ndownloader.figshare.com/files/55023443",
        "md5": "75a52e66137bb415670d3348f5123cff",
        "size": 2457,
        "orig_name": "READEME- Dataset for Qinghai-Tibet Plateau Permafrost Profiles (2001–2020).txt",
    },
}

MISSING = "####"
UNIT_FACTOR = 100.0          # m 에서 cm
RANGE_MAX_CM = 600.0
DEM_TOL_M = 50.0             # R1 의 허용 차
SAME_COORD_TOL = 1e-5        # R2 의 좌표 일치 판정(도)
SAME_SERIES_MIN_YEARS = 5    # 두 지점의 연 값이 같은 연도 수가 이 값 이상이면 기록한다
TIBET_BOX = (26.0, 40.0, 73.0, 105.0)   # 계획서 6B.4 의 Tibet 정의
V3_NEAR_DEG = 0.01           # 체비쇼프 거리(도)
CELL_DEG = 0.009             # 약 1 km 셀(계획서 6B.3)

COLUMNS = [
    "src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
    "label_def", "n_obs", "country", "macro", "citation", "license",
    "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
    "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
    "qc_flag", "notes",
]


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_formula(expr):
    """'=35+42.948/60' 같은 수식을 계산한다. 숫자, 더하기, 나누기, 괄호만 허용한다."""
    s = str(expr).strip()
    if s.startswith("="):
        s = s[1:]
    if not re.fullmatch(r"[0-9+/(). ]+", s):
        raise ValueError(f"허용하지 않는 수식: {expr!r}")

    def ev(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            return a + b if isinstance(n.op, ast.Add) else a / b
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        raise ValueError(f"허용하지 않는 수식 요소: {expr!r}")

    return ev(ast.parse(s, mode="eval").body)


def n_decimals(text):
    text = str(text)
    return len(text.split(".")[1]) if "." in text else 0


def coord_kind_and_prec(raw):
    """원자료 표기의 종류와 표기상 정밀도(도)를 돌려준다. 정확도가 아니다.

    수식은 분 또는 초의 소수 자릿수로 정한다. 십진 도 수치는 소수 5자리 이하이면 그 자릿수로 정한다.
    소수 6자리 이상인 수치는 도와 분, 도분초에서 환산한 값으로 보고, 분(소수 0-3자리) 또는
    초(소수 0-2자리)로 되돌렸을 때 맞아떨어지는 자릿수로 정한다. 맞지 않으면 소수 자릿수(최대 7)로 정한다.
    끝자리 0 은 표기에서 사라지므로 이 값은 실제보다 거칠게 나올 수 있다.
    """
    if isinstance(raw, str) and raw.strip().startswith("="):
        s = raw.strip()[1:].replace(" ", "")
        m = re.fullmatch(r"(\d+)\+\((\d+)\+([\d.]+)/60\)/60", s)
        if m:
            return "dms", 10.0 ** (-n_decimals(m.group(3))) / 3600.0
        m = re.fullmatch(r"(\d+)\+\(?([\d.]+)/60\)?", s)
        if m:
            return "deg_decmin", 10.0 ** (-n_decimals(m.group(2))) / 60.0
        raise ValueError(f"해석하지 못한 좌표 수식: {raw!r}")
    x = float(raw)
    k = n_decimals(repr(x))
    if k <= 5:
        return "decimal", 10.0 ** (-k)
    kk = min(k, 7)
    frac = abs(x) - math.floor(abs(x))
    minutes, tol_m = frac * 60.0, 60.0 * 0.5 * 10.0 ** (-kk) * 1.2
    for d in (0, 1, 2, 3):
        if abs(minutes - round(minutes, d)) <= tol_m:
            return "decimal_from_decmin", 10.0 ** (-d) / 60.0
    seconds, tol_s = frac * 3600.0, 3600.0 * 0.5 * 10.0 ** (-kk) * 1.2
    for d in (0, 1, 2):
        if abs(seconds - round(seconds, d)) <= tol_s:
            return "decimal_from_dms", 10.0 ** (-d) / 3600.0
    return "decimal", 10.0 ** (-kk)


def fmt_coord(v):
    return f"{v:.7f}".rstrip("0").rstrip(".")


def fmt_prec(v):
    return f"{v:.2g}"


def read_sites():
    wb_f = openpyxl.load_workbook(XLSX, data_only=False)
    wb_v = openpyxl.load_workbook(XLSX, data_only=True)
    ws_f, ws_v = wb_f["SITE"], wb_v["SITE"]
    hdr = [c.value for c in next(ws_f.iter_rows(min_row=1, max_row=1, max_col=11))]
    expect = ["Site", "Region", "Latitude", "Longitude", "Altitude (m)", "Ecological type",
              "Vegetation cover(%)", "VMC(%)", "MAGT(oC)", "Geomorphic unit", "Soil type"]
    if hdr != expect:
        raise ValueError(f"SITE 머리글이 예상과 다르다: {hdr}")
    sites = OrderedDict()
    max_cache_diff = 0.0
    kinds = Counter()
    for rf, rv in zip(ws_f.iter_rows(min_row=2, max_col=11), ws_v.iter_rows(min_row=2, max_col=11)):
        code = rf[0].value
        if code is None:
            continue
        code_s = str(code).strip()
        if code_s in sites:
            raise ValueError(f"SITE 시트에 지점 코드가 중복된다: {code_s}")
        vals = {}
        precs, kinds_site = [], []
        for name, idx in (("lat", 2), ("lon", 3)):
            raw = rf[idx].value
            kind, prec = coord_kind_and_prec(raw)
            val = eval_formula(raw) if kind in ("dms", "deg_decmin") else float(raw)
            cached = rv[idx].value
            if cached is not None:
                max_cache_diff = max(max_cache_diff, abs(float(cached) - val))
            vals[name] = val
            precs.append(prec)
            kinds_site.append(kind)
        form = "formula" if kinds_site[0] in ("dms", "deg_decmin") else "decimal_number"
        kinds[f"{form}:{kinds_site[0]}" if kinds_site[0] == kinds_site[1]
              else f"{form}:{kinds_site[0]}+{kinds_site[1]}"] += 1
        sites[code_s] = dict(
            site_id=code_s,
            locality=re.sub(r"\s+", " ", str(rf[1].value)).strip(),
            lat=vals["lat"], lon=vals["lon"],
            coord_kind=kinds_site[0] if kinds_site[0] == kinds_site[1] else "mixed",
            coord_prec=min(precs),
            elev_m=float(rf[4].value) if rf[4].value is not None else None,
            magt_c=float(rf[8].value) if rf[8].value is not None else None,
        )
    return sites, max_cache_diff, kinds


def read_alt():
    wb = openpyxl.load_workbook(XLSX, data_only=False)
    ws = wb["ALT"]
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    rows = [r for r in rows if any(v is not None for v in r)]
    hdr = rows[0]
    if hdr[0] != "Year":
        raise ValueError(f"ALT 시트의 첫 열이 Year 가 아니다: {hdr[0]!r}")
    codes = []
    for v in hdr[1:]:
        if v is None:
            continue
        codes.append(str(v).strip())
    if len(set(codes)) != len(codes):
        raise ValueError("ALT 시트에 지점 코드가 중복된다")
    recs = []        # (code, year, value 또는 None, 결측 사유)
    years = []
    for r in rows[1:]:
        y = r[0]
        if not isinstance(y, (int, float)) or int(y) != y:
            raise ValueError(f"ALT 시트의 연도 값이 정수가 아니다: {y!r}")
        y = int(y)
        if y in years:
            raise ValueError(f"ALT 시트에 연도가 중복된다: {y}")
        years.append(y)
        for j, code in enumerate(codes, start=1):
            v = r[j] if j < len(r) else None
            if isinstance(v, str):
                if v.strip().startswith("="):
                    raise ValueError(f"ALT 시트에 수식이 있다: {code} {y}")
                if v.strip() == MISSING:
                    recs.append((code, y, None, "missing_code"))
                else:
                    recs.append((code, y, None, "non_numeric"))
            elif v is None:
                recs.append((code, y, None, "empty_cell"))
            else:
                recs.append((code, y, float(v), ""))
    return codes, years, recs


def dem_elevation(lat, lon, dem_dir):
    """Copernicus GLO-30 타일에서 좌표가 속한 화소의 고도를 읽는다. 없으면 None."""
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception:
        return None
    ns = f"N{int(math.floor(lat)):02d}" if lat >= 0 else f"S{int(math.ceil(-lat)):02d}"
    ew = f"E{int(math.floor(lon)):03d}" if lon >= 0 else f"W{int(math.ceil(-lon)):03d}"
    tif = Path(dem_dir) / f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"
    if not tif.exists():
        return None
    with rasterio.open(tif) as ds:
        r, c = ds.index(lon, lat)
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            return None
        return float(ds.read(1, window=Window(c, r, 1, 1))[0, 0])


def cell_index(lat, lon):
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return ky, kx


def block_id(lat, lon):
    return int(math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5))


def git_head():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        head = out.stdout.strip()
        st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                             str(Path(__file__).resolve().relative_to(ROOT))],
                            capture_output=True, text=True, timeout=20).stdout.strip()
        return head, (st[:2].strip() if st else "clean")
    except Exception:
        return "", "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dem-dir", default=str(DEM_DIR))
    ap.add_argument("--v3", default=str(V3))
    args = ap.parse_args()

    # ---------- 원자료 확인 ----------
    files_meta = []
    for name, info in FIGSHARE_FILES.items():
        p = RAW_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"원자료가 없다: {p}. 내려받기: curl -L -o {name} {info['url']}")
        md5 = file_hash(p, "md5")
        files_meta.append(OrderedDict(
            name=name, size=p.stat().st_size, sha256=file_hash(p, "sha256"), md5=md5,
            url=info["url"], md5_matches_figshare=bool(md5 == info["md5"]),
            orig_name=info.get("orig_name", name)))
        if md5 != info["md5"]:
            raise ValueError(f"{name} 의 md5 가 figshare 기록과 다르다")

    sites, max_cache_diff, coord_kinds = read_sites()
    codes, years, recs = read_alt()

    # ---------- 지점 대조 ----------
    alt_only = [c for c in codes if c not in sites]
    site_only = [c for c in sites if c not in codes]

    # ---------- 좌표 확인 ----------
    la0, la1, lo0, lo1 = TIBET_BOX
    out_of_box = [c for c, s in sites.items() if not (la0 <= s["lat"] <= la1 and lo0 <= s["lon"] <= lo1)]
    if out_of_box:
        raise ValueError(f"Tibet 범위 밖 좌표: {out_of_box}")

    dem = OrderedDict()
    for c, s in sites.items():
        z = dem_elevation(s["lat"], s["lon"], args.dem_dir)
        s["dem_m"] = z
        s["dem_diff_m"] = None if (z is None or s["elev_m"] is None) else z - s["elev_m"]
        dem[c] = s["dem_diff_m"]

    coord_reason = {c: [] for c in sites}
    for c, s in sites.items():
        if s["dem_diff_m"] is not None and abs(s["dem_diff_m"]) > DEM_TOL_M:
            coord_reason[c].append(f"R1 dem_minus_elev_m={int(round(s['dem_diff_m'])):+d}")
    shared_pairs = []
    for a, b in itertools.combinations(sites, 2):
        sa, sb = sites[a], sites[b]
        same_lat = abs(sa["lat"] - sb["lat"]) < SAME_COORD_TOL
        same_lon = abs(sa["lon"] - sb["lon"]) < SAME_COORD_TOL
        if not (same_lat or same_lon):
            continue
        what = "latlon" if (same_lat and same_lon) else ("lat" if same_lat else "lon")
        da, db = sa["dem_diff_m"], sb["dem_diff_m"]
        if da is None or db is None:
            flagged = [a, b]
        else:
            flagged = [a] if abs(da) > abs(db) else [b]
        shared_pairs.append(OrderedDict(site_a=a, site_b=b, shared=what, flagged=flagged))
        for f in flagged:
            other = b if f == a else a
            coord_reason[f].append(f"R2 {what}_same_as={other}")
        for f, other in ((a, b), (b, a)):
            sites[f].setdefault("coord_shared", []).append(f"{what}_same_as={other}")

    # ---------- 연 값 계열의 중복 ----------
    series = {c: {} for c in codes}
    for c, y, v, why in recs:
        if v is not None:
            series[c][y] = v
    same_series = []
    for a, b in itertools.combinations([c for c in codes if series[c]], 2):
        yrs = sorted(y for y in series[a] if y in series[b] and series[a][y] == series[b][y])
        if len(yrs) >= SAME_SERIES_MIN_YEARS:
            same_series.append(OrderedDict(site_a=a, site_b=b, n_years=len(yrs),
                                           year_min=yrs[0], year_max=yrs[-1], years=yrs))
    same_years = {c: {} for c in codes}
    for p in same_series:
        for y in p["years"]:
            same_years[p["site_a"]].setdefault(y, []).append(p["site_b"])
            same_years[p["site_b"]].setdefault(y, []).append(p["site_a"])

    # ---------- v3 근접 셀(읽기 전용) ----------
    v3_near = {c: [] for c in sites}
    v3_info = OrderedDict(read=False)
    v3_direct = None
    if Path(args.v3).exists():
        import pandas as pd
        v3 = pd.read_csv(args.v3, usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm"],
                         low_memory=False)
        v3 = v3[(v3.lat.between(la0, la1)) & (v3.lon.between(lo0, lo1))].reset_index(drop=True)
        v3_info = OrderedDict(read=True, file=str(Path(args.v3).relative_to(ROOT)),
                              n_rows_in_tibet_box=int(len(v3)))
        v3_direct = v3[v3.source_id == "F4_direct"]
        for c, s in sites.items():
            d = np.maximum((v3.lat - s["lat"]).abs(), (v3.lon - s["lon"]).abs())
            for _, r in v3[d <= V3_NEAR_DEG].iterrows():
                v3_near[c].append(OrderedDict(loc_id=int(r.loc_id), source_id=str(r.source_id),
                                              region=str(r.region), alt_cm=round(float(r.alt_cm), 1)))

    # ---------- 점 자료 ----------
    rows_out = []
    excluded = Counter()
    for c, y, v, why in recs:
        if v is None:
            key = f"{why}"
            if c not in sites:
                key += "_and_no_site_row"
            excluded[key] += 1
            continue
        if c not in sites:
            excluded["value_but_no_site_row"] += 1
            continue
        s = sites[c]
        alt_cm = round(v * UNIT_FACTOR, 1)
        flags = []
        if not (0.0 < alt_cm <= RANGE_MAX_CM):
            flags.append("range")
        if coord_reason[c]:
            flags.append("coord")
        notes = [f"locality={s['locality']}"]
        if s["elev_m"] is not None:
            notes.append(f"elev_m={s['elev_m']:.0f}")
        if s["magt_c"] is not None:
            notes.append(f"magt_c={s['magt_c']:.2f}")
        if s["dem_diff_m"] is not None:
            notes.append(f"dem_minus_elev_m={int(round(s['dem_diff_m'])):+d}")
        for t in s.get("coord_shared", []):
            notes.append(t)
        if y in same_years[c]:
            notes.append("value_same_as=" + "|".join(same_years[c][y]))
        for n in v3_near[c]:
            notes.append(f"v3_near={n['loc_id']}:{n['source_id']}:{n['region']}")
        rows_out.append(OrderedDict(
            src_id=SRC_ID, site_id=c, site_name=f"{s['locality']} {c}",
            lat=fmt_coord(s["lat"]), lon=fmt_coord(s["lon"]), year=y, month="",
            alt_cm=f"{alt_cm:.1f}", method="borehole_temp", label_def="temp_derived", n_obs=1,
            country="China", macro="Tibet", citation=CITATION, license=LICENSE,
            subunit="QTEC", date="", eos_basis="none", value_kind="annual_value",
            year_min="", year_max="", alt_sd_cm="", coord_prec_deg=fmt_prec(s["coord_prec"]),
            disturbed="", disturb_type="", right_censored=0, orig_source="", dup_of="",
            qc_flag=";".join(flags) if flags else "ok", notes="; ".join(notes)))

    keys = [(r["site_id"], r["year"]) for r in rows_out]
    if len(set(keys)) != len(keys):
        raise ValueError("(site_id, year) 가 중복된다")
    rows_out.sort(key=lambda r: (list(sites).index(r["site_id"]), r["year"]))

    vals_m = np.array([v for _, _, v, _ in recs if v is not None])
    if not (0.3 <= float(np.median(vals_m)) <= 10.0):
        raise ValueError("ALT 값의 중앙값이 m 단위 가정(0.3-10)과 맞지 않는다")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POINTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # ---------- 요약 ----------
    alt = np.array([float(r["alt_cm"]) for r in rows_out])
    ok_mask = np.array([r["qc_flag"] == "ok" for r in rows_out])
    flag_counts = Counter()
    for r in rows_out:
        for t in r["qc_flag"].split(";"):
            flag_counts[t] += 1
    site_years = Counter(r["site_id"] for r in rows_out)
    coord_sites = [c for c in sites if coord_reason[c]]
    range_sites = sorted({r["site_id"] for r in rows_out if "range" in r["qc_flag"].split(";")},
                         key=list(sites).index)

    def dist(a):
        if len(a) == 0:
            return None
        return OrderedDict(n=int(len(a)), min=float(np.min(a)), p25=float(np.percentile(a, 25)),
                           median=float(np.median(a)), mean=round(float(np.mean(a)), 1),
                           p75=float(np.percentile(a, 75)), max=float(np.max(a)))

    # 셀 추정(v4 조립 전 참고값). 지점의 다년 평균은 range 표시가 없는 행으로 계산한다.
    def cell_estimate(use_sites):
        cells = {}
        for c in use_sites:
            s = sites[c]
            cells.setdefault(cell_index(s["lat"], s["lon"]), []).append(c)
        out = []
        for k, members in cells.items():
            lat = float(np.mean([sites[m]["lat"] for m in members]))
            lon = float(np.mean([sites[m]["lon"] for m in members]))
            dup = False
            if v3_direct is not None and len(v3_direct):
                d = np.maximum((v3_direct.lat - lat).abs(), (v3_direct.lon - lon).abs())
                dup = bool((d <= V3_NEAR_DEG).any())
            out.append(dict(cell=k, sites=members, lat=lat, lon=lon, block=block_id(lat, lon),
                            dup_v3_direct=dup))
        new = [o for o in out if not o["dup_v3_direct"]]
        return OrderedDict(
            n_sites=len(use_sites), n_cells=len(out),
            n_cells_dup_v3_f4_direct=len(out) - len(new),
            n_new_cells=len(new), n_new_blocks=len({o["block"] for o in new}),
            n_blocks_all=len({o["block"] for o in out}),
            multi_site_cells=[o["sites"] for o in out if len(o["sites"]) > 1],
            dup_cells=[o["sites"] for o in out if o["dup_v3_direct"]])

    cells_all = cell_estimate(list(sites))
    cells_ok = cell_estimate([c for c in sites if not coord_reason[c]])

    dem_checked = [c for c in sites if sites[c]["dem_diff_m"] is not None]
    dem_abs = np.array([abs(sites[c]["dem_diff_m"]) for c in dem_checked])
    unit_check = []
    for c in sites:
        ys = [float(r["alt_cm"]) for r in rows_out if r["site_id"] == c]
        for n in v3_near[c]:
            unit_check.append(OrderedDict(site_id=c, site_mean_cm=round(float(np.mean(ys)), 1),
                                          v3_loc_id=n["loc_id"], v3_source_id=n["source_id"],
                                          v3_alt_cm=n["alt_cm"]))

    head, parser_state = git_head()
    parser_rel = str(Path(__file__).resolve().relative_to(ROOT))
    meta = OrderedDict(
        src_id=SRC_ID,
        name="Fu 2025, 청장고원 동토 지온과 ALT 자료 2001-2020(figshare 29206613)",
        url=URL, doi=DOI, accessed=ACCESSED,
        generated=datetime.date.today().isoformat(),
        schema_version="1.0",
        files=files_meta,
        raw_dir=str(RAW_DIR.relative_to(ROOT)),
        license=LICENSE,
        license_basis="figshare 기록의 license 항목과 README 의 License 절",
        citation=CITATION,
        related_article="README 가 적은 연결 논문 제목: Non-temperature Environmental Drivers Modulate "
                        "Warming-induced 21st Century Permafrost Degradation on the Tibetan Plateau. "
                        "DOI 와 게재 여부는 확인하지 못했다.",
        parser=parser_rel,
        parser_sha256=file_hash(Path(__file__).resolve(), "sha256"),
        parser_git_commit=head,
        parser_git_state=f"HEAD {head}, 파서 파일 상태 {parser_state}(?? 는 미추적)",
        role="보조 라벨(지온 유도). 확인적 풀에 넣지 않는다. 계획서 6B.5 의 L39 비교 대상",
        n_rows_raw=len(recs),
        n_rows_raw_detail=OrderedDict(
            sheet_SITE_rows=len(sites), sheet_ALT_year_rows=len(years),
            sheet_ALT_site_columns=len(codes), sheet_ALT_cells=len(recs),
            sheet_ALT_numeric_cells=int(len(vals_m)),
            sheet_ALT_missing_cells=int(len(recs) - len(vals_m))),
        n_rows_points=len(rows_out),
        n_sites=len(site_years),
        n_years_per_site=OrderedDict(min=min(site_years.values()), max=max(site_years.values())),
        year_range=[min(r["year"] for r in rows_out), max(r["year"] for r in rows_out)],
        n_by_label_def=dict(Counter(r["label_def"] for r in rows_out)),
        n_by_method=dict(Counter(r["method"] for r in rows_out)),
        n_by_macro=dict(Counter(r["macro"] for r in rows_out)),
        n_by_subunit=dict(Counter(r["subunit"] for r in rows_out)),
        n_by_qc_flag=dict(flag_counts),
        n_rows_qc_ok=int(ok_mask.sum()),
        n_excluded_by_reason=dict(excluded),
        excluded_detail="지점 WL3 은 시트 ALT 의 20개 연도가 모두 '####' 이고 시트 SITE 에 행이 없다. "
                        "값과 좌표가 없으므로 점 자료에 넣지 않았다.",
        sites_in_ALT_not_in_SITE=alt_only,
        sites_in_SITE_not_in_ALT=site_only,
        alt_cm_distribution=OrderedDict(all_rows=dist(alt), qc_ok_rows=dist(alt[ok_mask])),
        units=OrderedDict(
            raw_unit="m",
            factor=UNIT_FACTOR,
            basis="README 의 단위 표기는 'Centimeters (m)' 로 모순이다. 원자료 값의 범위가 "
                  f"{float(vals_m.min()):.2f}-{float(vals_m.max()):.2f}, 중앙값 {float(np.median(vals_m)):.3f} 이고 "
                  "소수 둘째 자리까지 적혀 있다. v3 표에서 0.01도 안에 있는 셀의 ALT(cm)와 지점 평균(cm)을 "
                  "대조했다(unit_check_vs_v3).",
            unit_check_vs_v3=unit_check),
        missing_code=MISSING,
        right_censored="원자료에 '>' 표기나 하한 표시가 없다. 모든 행을 0 으로 두었다.",
        coordinate_conversion=OrderedDict(
            method="시트 SITE 의 좌표는 십진 도 수치 또는 수식이다. 수식은 도 + 분/60 과 "
                   "도 + (분 + 초/60)/60 두 형태이고 파서가 직접 계산했다. 북위와 동경이며 부호 변환은 없다. "
                   "출력은 소수 7자리까지 적고 뒤의 0 을 뺐다.",
            n_sites_by_raw_form=dict(coord_kinds),
            max_abs_diff_formula_vs_cached_value_deg=max_cache_diff,
            coord_prec_deg="원자료 표기의 자릿수에서 계산한 표기상 정밀도다(위도와 경도 가운데 작은 값. "
                           "끝자리 0 이 표기에서 사라지기 때문이다). 정확도가 아니다. README 는 일부 지점 위치가 "
                           "보안을 이유로 일반화되었을 수 있다고 적었다.",
            datum="원자료에 측지계 표기가 없다. WGS84 로 가정했다."),
        qc_rules=OrderedDict(
            range=f"0 < alt_cm <= {RANGE_MAX_CM:.0f} 이 아닌 행",
            coord_R1=f"보고 고도와 Copernicus DEM GLO-30 화소 고도의 차가 {DEM_TOL_M:.0f} m 를 넘는 지점",
            coord_R2=f"위도 또는 경도가 다른 지점과 {SAME_COORD_TOL} 도 안에서 같은 쌍 가운데 DEM 고도 차의 "
                     "절댓값이 큰 쪽. DEM 을 읽지 못하면 두 지점 모두",
            same_series=f"두 지점의 연 값이 같은 연도가 {SAME_SERIES_MIN_YEARS}개 이상이면 notes 에 "
                        "value_same_as 로 적는다. qc_flag 는 바꾸지 않는다"),
        coord_flagged_sites=OrderedDict((c, coord_reason[c]) for c in coord_sites),
        coord_shared_pairs=shared_pairs,
        range_flagged_sites=OrderedDict(
            (c, sum(1 for r in rows_out if r["site_id"] == c and "range" in r["qc_flag"].split(";")))
            for c in range_sites),
        same_series_pairs=same_series,
        dem_elevation_check=OrderedDict(
            dem="Copernicus DEM GLO-30(data/raw/dem, 1도 타일)",
            n_sites_checked=len(dem_checked), n_sites_unchecked=len(sites) - len(dem_checked),
            abs_diff_m_median=None if len(dem_abs) == 0 else round(float(np.median(dem_abs)), 1),
            abs_diff_m_max=None if len(dem_abs) == 0 else round(float(np.max(dem_abs)), 1),
            n_sites_over_tol=int((dem_abs > DEM_TOL_M).sum()) if len(dem_abs) else 0,
            per_site=OrderedDict((c, None if sites[c]["dem_diff_m"] is None
                                  else round(sites[c]["dem_diff_m"], 1) + 0.0) for c in sites)),
        v3_proximity=OrderedDict(
            v3=v3_info, rule=f"체비쇼프 거리 {V3_NEAR_DEG} 도 이내",
            sites_near_v3=OrderedDict((c, v3_near[c]) for c in sites if v3_near[c])),
        cell_estimate=OrderedDict(
            note="v4 조립 전의 참고값이다. 셀 색인은 계획서 6B.3 의 식을 썼고, v3 의 F4_direct 셀과 "
                 "체비쇼프 0.01도 이내인 셀을 중복으로 셌다.",
            all_sites=cells_all, excluding_coord_flagged_sites=cells_ok),
        unverified_items=[
            "시트 ALT 는 54개 지점 모두 2001-2020년의 20개 연도 값이 빠짐없이 있다. README 는 "
            "자료 공백을 보정했다고 적었으나 어느 연도가 관측에서 나온 값이고 어느 연도가 "
            "보간 또는 재구성 값인지는 자료에 없다. 확인하지 못했다.",
            "ALT 산정 방법의 세부(온도 센서 깊이, 0도 등온선 내삽 방식, 연 최대의 기준일)는 "
            "README 에 없다. README 의 설명은 '지온 단면 해석으로 추정, 불확도 약 5 cm' 이다.",
            "README 는 일부 지점 위치가 일반화되었을 수 있다고 적었다. 어느 지점인지는 적지 않았다.",
            "좌표의 측지계는 자료에 없다.",
            "coord 표시 지점의 원인(좌표 오기인지 고도 오기인지)은 확인하지 못했다. 좌표와 값은 고치지 않았다.",
            "QSH-2 와 QSH-3 의 같은 값이 원자료의 복사 오류인지 같은 시추공 자료의 공유인지는 확인하지 못했다.",
            "시추공이 자연 지반에 있는지 도로나 철도 노반 가까이에 있는지는 자료에 없다. disturbed 열은 비워 두었다.",
            "연결 논문의 DOI 와 게재 여부는 확인하지 못했다.",
            "MAGT 가 0도 이상인 지점(notes 의 magt_c)에서 ALT 값이 영구동토 상한 깊이를 뜻하는지는 확인하지 못했다.",
        ],
    )
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---------- 화면 요약 ----------
    print(f"[{SRC_ID}] 원자료 (지점, 연도) 칸 {len(recs)}개, 수치 {len(vals_m)}개, 결측 {len(recs) - len(vals_m)}개")
    print(f"  지점 {len(site_years)}개, 점 자료 {len(rows_out)}행, 연도 {meta['year_range']}")
    print(f"  ALT(cm) 최소 {alt.min():.0f}, 중앙 {np.median(alt):.1f}, 최대 {alt.max():.0f}")
    print(f"  qc_flag: {dict(flag_counts)}")
    print(f"  coord 표시 지점: {dict(meta['coord_flagged_sites'])}")
    print(f"  range 표시 지점(행 수): {dict(meta['range_flagged_sites'])}")
    print(f"  값이 같은 계열: {[(p['site_a'], p['site_b'], p['n_years'], p['year_min'], p['year_max']) for p in same_series]}")
    print(f"  제외: {dict(excluded)}")
    print(f"  DEM 대조: {len(dem_checked)}개 지점, 절대 차 중앙 {meta['dem_elevation_check']['abs_diff_m_median']} m")
    print(f"  셀 추정(전체): {json.dumps({k: v for k, v in cells_all.items() if k not in ('multi_site_cells', 'dup_cells')})}")
    print(f"  셀 추정(coord 제외): {json.dumps({k: v for k, v in cells_ok.items() if k not in ('multi_site_cells', 'dup_cells')})}")
    print(f"  산출: {POINTS.relative_to(ROOT)}, {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
