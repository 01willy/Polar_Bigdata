"""Talucci 등 FireALT(Arctic Data Center) 연소지·미연소지 융해 깊이 파싱 → ext_labels 표준 점 자료.

계획서: docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B 절(LGD). 형식: data/processed/ext_labels/_schema.json (1.0).

입력(data/raw/firealt_talucci2025/)
- FireAltEstimatedRawData.csv   측정 기록 48,669행, 32열. 한 행이 탐침 측정 1건이다.
    msrDepth  현장 측정 융해 깊이(cm). 라벨로 쓴다.
    estDepth  ERA5-Land 기온의 √(융해 도일) 비로 계절 말 값으로 환산한 값(cm). 라벨로 쓰지 않는다.
    결측 부호: 수치 열은 -9999, 문자 열은 NA.
- FireAltEstimatedPlotLevel.csv, FireALTEstimatedPairsBurnedUnburned.csv 는 estDepth 집계만 담고 있어 읽지 않는다.

라벨 규칙(자료원 규칙과 _schema.json 의 규칙)
- alt_cm 은 msrDepth 의 평균이다. estDepth 는 어느 산출물에도 넣지 않는다(물리 기준선과 순환한다).
- method 는 probe 다. msrType(thaw, active)으로 거르지 않고 notes 에 적는다.
- 지점 단위: 위도와 경도를 소수 4자리로 반올림한 좌표와 연소 여부(distur)의 조합이다.
  같은 좌표를 여러 plot 이 공유하면 좌표 단위로 평균한다. 같은 좌표에 연소, 미연소 기록이 함께 있으면
  두 지점으로 나눈다. 원자료는 같은 위치를 정밀도가 다른 문자열 두 가지로 적은 경우가 있어
  (예: 59.36916667,-119.3211111 과 59.36917,-119.3211) 소수 4자리에서 묶는다(v3 의 ABoVE, ALLena 행 단위와 같다).
- 방문 단위: 지점과 관측일의 조합. 방문 값은 그 날 그 지점의 모든 msrDepth 의 평균이다.
- 연 값(행 단위는 지점과 연도)
    8월 1일 이후 방문이 없다                     → 그 해의 마지막 방문 값, direct_dated, eos_basis none
    8월 1일 이후 방문이 10월 이후뿐이다          → 그 방문 가운데 최댓값, direct_dated, eos_basis none
    8월 1일 이후 방문이 1회이고 8-9월이다        → 그 방문 값, direct_eos, eos_basis record_date
    8월 1일 이후 방문이 2회 이상이고 8-9월 방문이 있다
                                                 → 8월 1일 이후 방문 값 가운데 최댓값, direct_eos, eos_basis series_max
    (개정 13, 검증 지적 반영) series_max 의 최댓값 방문이 10월 이후이면 그 방문의 기록 수가 8-9월 방문의 최대 기록
    수와 같고 msrType 이 8-9월 방문과 같을 때만 쓴다. 조건을 채우지 못하면 8-9월 방문의 최댓값을 쓴다(8-9월 방문이
    2회 이상이면 series_max, 1회면 record_date). 10월 이후 방문은 측정점 구성이나 측정 종류가 달라 같은 지점 평균의
    최댓값이라는 6B.3 정의와 맞지 않을 수 있기 때문이다.
  month 와 date 는 값을 준 방문의 것이다. n_obs 와 alt_sd_cm 도 그 방문의 기록 수와 표준편차다.
- 연소 지점은 disturbed = 1, disturb_type = burned 다. 점 자료에 남기고 대상 셀에서는 뺀다(v4 조립 단계).
- gtProbe = y(탐침 길이 초과)는 right_censored = 1 이다. 공개 파일에는 gtProbe = y 와 hitRock = y 기록이 없다
  (자료 제작 스크립트 DataCleanPointCombine.Rmd 가 두 종류를 지운 뒤 공개했다).
- 범위 QC 는 행의 alt_cm 에 적용한다(0 < alt_cm <= 600). 값이 0 인 측정 기록은 지우지 않고 방문 평균에 넣으며
  그 수를 notes 와 메타에 적는다.
- 측정값 묶음이 다른 지점과 완전히 같은 행은 dup_of 에 같은 자료원의 첫 지점을 적는다
  (원자료 제작 과정의 결합에서 한 측정 묶음이 여러 좌표에 복제된 경우다).

산출
- data/processed/ext_labels/firealt_talucci2025_points.csv
- data/processed/ext_labels/firealt_talucci2025_meta.json
- data/raw/firealt_talucci2025/derived_visits.csv   방문 단위 표(연 값으로 뽑히지 않은 방문 포함)
- data/raw/firealt_talucci2025/derived_sites.csv    지점 표(원좌표 문자열, siteId, plot 수)

실행(저장소 루트에서, 스레드 1개)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 scripts/1_data_prep/parse_ext_firealt_talucci2025.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from functools import reduce

import numpy as np
import pandas as pd

SRC_ID = "firealt_talucci2025"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw", SRC_ID)
F_RAW = os.path.join(RAW_DIR, "FireAltEstimatedRawData.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed", "ext_labels")
F_POINTS = os.path.join(OUT_DIR, f"{SRC_ID}_points.csv")
F_META = os.path.join(OUT_DIR, f"{SRC_ID}_meta.json")
F_VISITS = os.path.join(RAW_DIR, "derived_visits.csv")
F_SITES = os.path.join(RAW_DIR, "derived_sites.csv")
F_V3 = os.path.join(ROOT, "data", "processed", "fidelity_base_v3.csv")

NAME = "Talucci 등 2025, FireALT 연소지·미연소지 융해 깊이(Arctic Data Center)"
URL = "https://doi.org/10.18739/A2RN3092P"
URL_OBJECT = "https://arcticdata.io/metacat/d1/mn/v2/object/"
DOI = "10.18739/A2RN3092P"
ACCESSED = "2026-09-29"
LICENSE = "CC BY 4.0"
CITATION = ("Talucci, A., Loranty, M., Holloway, J., Rogers, B., Alexander, H., et al. (2024) FireALT dataset: "
            "estimated active layer thickness for paired burned unburned sites measured from 2001-2023. "
            "NSF Arctic Data Center. https://doi.org/10.18739/A2RN3092P. 자료 논문: Talucci, A. C., et al. (2025) "
            "Permafrost-wildfire interactions: active layer thickness estimates for paired burned and unburned "
            "sites in northern high latitudes. Earth System Science Data 17, 2887-2909, "
            "https://doi.org/10.5194/essd-17-2887-2025")
RAW_FILES = ["FireAltEstimatedRawData.csv", "FireAltEstimatedPlotLevel.csv",
             "FireALTEstimatedPairsBurnedUnburned.csv", "FireALT_eml.xml", "FireALT_eml_prev_urn-dab3aa1f.xml",
             "DataCleanPointCombine.Rmd", "ERA5GeeScript.Rmd", "ERA5CleanOrganizeData.Rmd", "EstimateALT.Rmd",
             "PlotLevelAggregate.Rmd", "SpatialAdds.Rmd", "PairsAggregate.Rmd"]

EXPECTED_COLS = ["submitNm", "lastNm", "plotId", "siteId", "year", "cntryId", "lat", "lon", "month", "day", "biome",
                 "distur", "fireYr", "fireId", "paired", "gtProbe", "hitRock", "orgDepth", "srfH2O", "topoPos",
                 "slope", "vegCvr", "msrDoy", "msrType", "msrDepth", "estDoy", "estDepth", "tsf", "tsfClass",
                 "resName", "resBiome", "permaExtent"]
POINT_COLS = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
              "n_obs", "country", "macro", "citation", "license", "subunit", "date", "eos_basis", "value_kind",
              "year_min", "year_max", "alt_sd_cm", "coord_prec_deg", "disturbed", "disturb_type", "right_censored",
              "orig_source", "dup_of", "qc_flag", "notes"]
COUNTRY = {"US": "United States", "CA": "Canada", "RU": "Russia"}
MISSING = {"-9999", "-9999.0", "NA", ""}
KEY_DEC = Decimal("0.0001")        # 지점 단위 좌표 자리수(소수 4자리)
EOS_MONTHS = (8, 9)
LATE_MONTH = 8                     # 8월 1일 이후
RANGE_MAX = 600.0
YEAR_MIN_MAIN = 1990
CELL_DEG = 0.009
BLOCK_DEG = 0.5
DUP_DEG = 0.01                     # v3 F4_direct 셀과의 중복 판정(체비쇼프)
INDEP_KM = 100.0                   # 다른 macro 의 v3 F4_direct 셀과의 독립성 거리
IDENT_MIN_N = 3                    # 동일 측정값 묶음 판정에 쓰는 최소 값 수(중복도 제거 뒤)
LENA_BOX = (71.5, 73.6, 123.3, 130.1)
V3_MACRO = {"ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska", "ABoVE_CA": "Canada", "Canada": "Canada",
            "Lena_RU": "Lena", "GTNPenv_RU": "Siberia_GTNP", "GTNPenv_SJ": "Svalbard", "GTNPenv_US": "US_GTNP",
            "GTNPenv_CH": "Alps", "GTNPenv_AQ": "Antarctica", "QTP_CN": "Tibet", "CALM_Canada": "Canada",
            "CALM_Greenland": "Greenland", "CALM_Svalbard": "Svalbard", "CALM_Scandinavia": "Scandinavia",
            "CALM_Russia_W": "Russia_W", "CALM_Russia_C": "Russia_C", "CALM_Russia_E": "Russia_E",
            "CALM_Mongolia_CAsia": "Mongolia_CAsia", "CALM_Alps": "Alps", "CALM_QTP_China": "Tibet"}
# v3 의 macro 이름(src/polar/fidelity.py 의 MACRO_REGION 사본. 읽기만 한다)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ndec(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def q4(s: str) -> str:
    return str(Decimal(s).quantize(KEY_DEC, rounding=ROUND_HALF_UP))


def macro_of(cc: str, lat: float, lon: float) -> tuple[str, bool]:
    """6B.4 의 지리 정의. 둘째 값은 국가 부호와 좌표가 맞는지 여부다."""
    if cc == "US":
        ok = lat > 50 and -180 <= lon < -130
        return ("Alaska" if ok else "other"), ok
    if cc == "CA":
        ok = lat > 41 and -141.1 <= lon <= -52
        return "Canada", ok
    if cc == "RU":
        ok = lat > 41 and (lon >= 19 or lon <= -168)
        if lon < 0:
            return "Russia_E", ok
        if lon < 90:
            return "Russia_W", ok
        if lon < 140:
            a, b, c, d = LENA_BOX
            if a <= lat <= b and c <= lon <= d:
                return "Lena", ok
            return "Russia_C", ok
        return "Russia_E", ok
    return "other", False


def subunit_of(macro: str, eco: str) -> str:
    """하위 단위는 계획서가 이름을 정한 Russia_C(중앙 야쿠티아)만 채운다. 나머지는 빈 칸으로 두고
    자료가 붙인 생태지역 이름(resName)은 notes 에 적는다(지명으로 읽히면 위치를 오해할 수 있다)."""
    return "Yakutia_C" if macro == "Russia_C" else ""


def value_sig(vals: np.ndarray) -> tuple[str, int, int]:
    """측정값 묶음의 서명. 값별 개수를 최대공약수로 나눈 다중집합의 해시, 최대공약수, 나눈 뒤의 값 수."""
    c = Counter(np.round(vals, 3).tolist())
    g = reduce(math.gcd, c.values())
    key = str(sorted((k, v // g) for k, v in c.items()))
    return hashlib.md5(key.encode()).hexdigest()[:12], g, len(vals) // g


def uniq_join(s, sep="/", limit=4) -> str:
    u = sorted({str(x) for x in s if str(x) not in ("", "NA")})
    if len(u) > limit:
        return sep.join(u[:limit]) + f"{sep}(외 {len(u) - limit}개)"
    return sep.join(u)


def fmt(x, nd=4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def cell_index(lat: float, lon: float) -> tuple[int, int]:
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return ky, kx


def describe(a) -> dict:
    a = np.asarray(a, float)
    if len(a) == 0:
        return dict(n=0)
    return dict(n=int(len(a)), min=round(float(a.min()), 2), median=round(float(np.median(a)), 2),
                mean=round(float(a.mean()), 2), max=round(float(a.max()), 2))


# ------------------------------------------------------------------ 1. 읽기와 기록 단위 확인
raw = pd.read_csv(F_RAW, dtype=str, keep_default_na=False)
assert list(raw.columns) == EXPECTED_COLS, f"열 구성이 다르다: {list(raw.columns)}"
n_raw = len(raw)
excluded = Counter()
for c in ["lat", "lon", "year", "month", "day", "msrDepth"]:
    m = raw[c].isin(MISSING)
    excluded[f"missing_{c}"] = int(m.sum())
    raw = raw[~m]
m = raw.hitRock.eq("y")
excluded["hit_rock"] = int(m.sum())
raw = raw[~m].copy()
raw["latf"] = raw.lat.astype(float)
raw["lonf"] = raw.lon.astype(float)
raw["dep"] = raw.msrDepth.astype(float)
raw["yr"] = raw.year.astype(int)
raw["mo"] = raw.month.astype(int)
raw["dy"] = raw.day.astype(int)
raw["dt"] = pd.to_datetime(dict(year=raw.yr, month=raw.mo, day=raw.dy), errors="coerce")
m = raw.dt.isna()
excluded["invalid_date"] = int(m.sum())
raw = raw[~m].copy()
raw["date"] = raw.dt.dt.strftime("%Y-%m-%d")
raw["doy"] = raw.dt.dt.dayofyear
n_doy_mismatch = int((raw.doy != raw.msrDoy.astype(int)).sum())
n_used = len(raw)
n_zero_records = int((raw.dep <= 0).sum())
n_over_records = int((raw.dep > RANGE_MAX).sum())
n_gtprobe_y = int(raw.gtProbe.eq("y").sum())
n_fullrow_dup = int(raw.duplicated(subset=EXPECTED_COLS, keep="first").sum())
unknown_cc = sorted(set(raw.cntryId) - set(COUNTRY))
assert not unknown_cc, f"국가 부호 미정의: {unknown_cc}"
assert set(raw.distur) <= {"burned", "unburned"}, set(raw.distur)

mm = [macro_of(c, a, b) for c, a, b in zip(raw.cntryId, raw.latf, raw.lonf)]
raw["macro"] = [x[0] for x in mm]
raw["coord_ok"] = [x[1] for x in mm]
raw["lat4"] = raw.lat.map(q4)
raw["lon4"] = raw.lon.map(q4)
raw["skey"] = raw.lat4 + "," + raw.lon4 + "," + raw.distur
raw["rawcoord"] = raw.lat + "," + raw.lon
raw["plotkey"] = raw.submitNm + "|" + raw.siteId + "|" + raw.plotId

# ------------------------------------------------------------------ 2. 지점 표
chk = raw.groupby("skey").agg(ncc=("cntryId", "nunique"), nmac=("macro", "nunique"))
assert (chk.ncc == 1).all() and (chk.nmac == 1).all(), "한 지점에 국가나 macro 가 둘 이상이다"

site_rows = []
for skey, g in raw.groupby("skey", sort=False):
    rc = g.drop_duplicates("rawcoord")
    if len(rc) == 1:
        lat_s, lon_s = rc.lat.iloc[0], rc.lon.iloc[0]
    else:
        lat_s, lon_s = f"{rc.latf.mean():.6f}", f"{rc.lonf.mean():.6f}"
    dec = min(max(ndec(s) for s in rc.lat), max(ndec(s) for s in rc.lon))
    site_rows.append(dict(
        skey=skey, cntryId=g.cntryId.iloc[0], macro=g.macro.iloc[0], distur=g.distur.iloc[0],
        lat4=float(g.lat4.iloc[0]), lon4=float(g.lon4.iloc[0]), lat=lat_s, lon=lon_s,
        coord_prec_deg=10.0 ** (-dec), n_rawcoord=len(rc), rawcoords=" | ".join(sorted(rc.rawcoord)),
        contributors=uniq_join(g.submitNm), lastNm=uniq_join(g.lastNm, sep="; "),
        siteIds=uniq_join(g.siteId), n_siteId=g.siteId.nunique(), n_plots=g.plotkey.nunique(),
        n_records=len(g), years=uniq_join(g.year, sep=",", limit=30), coord_ok=bool(g.coord_ok.all()),
        ecoregion=uniq_join(g.resName), biome=uniq_join(g.biome), permaExtent=uniq_join(g.permaExtent)))
sites = pd.DataFrame(site_rows).sort_values(["cntryId", "lat4", "lon4", "distur"]).reset_index(drop=True)
sites["site_id"] = [f"{SRC_ID}_{i + 1:04d}" for i in range(len(sites))]
sites["site_name"] = sites.contributors + " " + sites.siteIds + " (" + sites.distur + ")"
sid = dict(zip(sites.skey, sites.site_id))
raw["site_id"] = raw.skey.map(sid)

# ------------------------------------------------------------------ 3. 방문 표
vis_rows = []
for (site_id, date), g in raw.groupby(["site_id", "date"], sort=True):
    v = g.dep.values
    sig, mult, n_norm = value_sig(v)
    vis_rows.append(dict(
        site_id=site_id, date=date, year=int(g.yr.iloc[0]), month=int(g.mo.iloc[0]), doy=int(g.doy.iloc[0]),
        n_records=len(v), mean_cm=float(v.mean()), sd_cm=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        min_cm=float(v.min()), max_cm=float(v.max()), n_zero=int((v <= 0).sum()),
        n_gt_probe=int(g.gtProbe.eq("y").sum()), n_plots=g.plotkey.nunique(), msrType=uniq_join(g.msrType),
        contributors=uniq_join(g.submitNm), siteIds=uniq_join(g.siteId), vegCvr=uniq_join(g.vegCvr),
        fireYr=uniq_join([x for x in g.fireYr if x not in MISSING], sep=","),
        tsf=uniq_join([x for x in g.tsf if x not in MISSING], sep=",", limit=6),
        value_sig=sig, value_mult=mult, n_norm=n_norm))
vis = pd.DataFrame(vis_rows)
vis["is_late"] = (vis.month >= LATE_MONTH).astype(int)
vis["in_window"] = vis.month.isin(EOS_MONTHS).astype(int)
# 동일 측정값 묶음: 같은 날짜, 같은 서명, 값 수 IDENT_MIN_N 이상, 지점 2곳 이상
grp = vis[vis.n_norm >= IDENT_MIN_N].groupby(["date", "value_sig"]).site_id.agg(list)
ident_first, ident_size = {}, {}
for (date, sig), ids in grp.items():
    if len(ids) > 1:
        for s in ids:
            ident_first[(s, date)] = min(ids)
            ident_size[(s, date)] = len(ids)
vis["ident_first"] = [ident_first.get((s, d), "") for s, d in zip(vis.site_id, vis.date)]
vis["ident_n_sites"] = [ident_size.get((s, d), 0) for s, d in zip(vis.site_id, vis.date)]

# ------------------------------------------------------------------ 4. 연 값
smeta = sites.set_index("site_id")
chosen_idx, ann = [], []
for (site_id, year), g in vis.groupby(["site_id", "year"], sort=True):
    g = g.sort_values("date")
    late = g[g.is_late == 1]
    win = late[late.in_window == 1]
    if len(late) == 0:
        c = g.iloc[-1]
        label_def, eos_basis = "direct_dated", "none"
    elif len(win) == 0:
        c = late.sort_values(["mean_cm", "date"]).iloc[-1]
        label_def, eos_basis = "direct_dated", "none"
    elif len(late) == 1:
        c = late.iloc[0]
        label_def, eos_basis = "direct_eos", "record_date"
    else:
        c = late.sort_values(["mean_cm", "date"]).iloc[-1]
        label_def, eos_basis = "direct_eos", "series_max"
    oct_rule = ""
    if eos_basis == "series_max" and c.month not in EOS_MONTHS:      # 개정 13: 10월 이후 최댓값 방문의 비교 조건
        if int(c.n_records) == int(win.n_records.max()) and set(win.msrType) == {c.msrType}:
            oct_rule = f"10월 최댓값 방문 유지(기록 수 {int(c.n_records)} = 8-9월 최대 {int(win.n_records.max())}, msrType {c.msrType})"
        else:
            old = c
            c = win.sort_values(["mean_cm", "date"]).iloc[-1]
            eos_basis = "series_max" if len(win) >= 2 else "record_date"
            oct_rule = (f"10월 최댓값 방문 기각(개정 13): {old.date} 평균 {old.mean_cm:.1f} cm, 기록 {int(old.n_records)}건, "
                        f"msrType {old.msrType} 대 8-9월 최대 기록 {int(win.n_records.max())}건, msrType "
                        f"{'/'.join(sorted(set(win.msrType)))}")
    chosen_idx.append(c.name)
    s = smeta.loc[site_id]
    late_dates = pd.to_datetime(late.date)
    gap_min = int(late_dates.diff().dt.days.min()) if len(late) > 1 else None
    flags = []
    if not (0 < c.mean_cm <= RANGE_MAX):
        flags.append("range")
    if not s.coord_ok:
        flags.append("coord")
    if year < YEAR_MIN_MAIN:
        flags.append("year_pre1990")
    notes = [f"기여자 {c.contributors}", f"siteId {c.siteIds}", f"plot {c.n_plots}개",
             f"연중 방문 {len(g)}회(8월 1일 이후 {len(late)}회)", f"msrType {c.msrType}",
             f"식생 {c.vegCvr}", f"영구동토 분포 {s.permaExtent}", f"생태지역 {s.ecoregion}"]
    if s.distur == "burned":
        notes.append(f"화재 연도 {c.fireYr}, 화재 후 경과 연수 {c.tsf}")
    if eos_basis == "series_max" and c.month not in EOS_MONTHS:
        notes.append(f"최댓값을 준 방문이 {c.month}월이다")
    if oct_rule:
        notes.append(oct_rule)
    if gap_min is not None and gap_min <= 4:
        notes.append(f"8월 1일 이후 방문 간격 최소 {gap_min}일(같은 조사 기간의 서로 다른 plot 일 수 있다)")
    if c.n_zero > 0:
        notes.append(f"값을 준 방문에 0 cm 기록 {c.n_zero}건 포함")
    if s.n_rawcoord > 1:
        notes.append(f"원좌표 문자열 {s.n_rawcoord}개를 소수 4자리에서 묶음")
    dup_of = ""
    if c.ident_first:
        notes.append(f"측정값 묶음이 같은 날짜의 다른 지점과 같다(지점 {c.ident_n_sites}곳, 값 중복도 {c.value_mult})")
        if c.ident_first != site_id:
            dup_of = f"{SRC_ID}:{c.ident_first}"
    ann.append(dict(
        src_id=SRC_ID, site_id=site_id, site_name=s.site_name, lat=s.lat, lon=s.lon, year=int(year),
        month=int(c.month), alt_cm=fmt(c.mean_cm, 4), method="probe", label_def=label_def, n_obs=int(c.n_records),
        country=COUNTRY[s.cntryId], macro=s.macro, citation=CITATION, license=LICENSE,
        subunit=subunit_of(s.macro, s.ecoregion), date=c.date, eos_basis=eos_basis,
        value_kind="single_visit" if len(g) == 1 else "annual_value", year_min="", year_max="",
        alt_sd_cm=fmt(c.sd_cm, 4), coord_prec_deg=fmt(s.coord_prec_deg, 8),
        disturbed=1 if s.distur == "burned" else 0, disturb_type="burned" if s.distur == "burned" else "",
        right_censored=1 if c.n_gt_probe > 0 else 0,
        orig_source=f"FireALT 기여 자료(lastNm 열): {s.lastNm}", dup_of=dup_of,
        qc_flag=";".join(flags) if flags else "ok", notes="; ".join(notes),
        x_alt=float(c.mean_cm), x_latf=float(s.lat), x_lonf=float(s.lon), x_n_vis=len(g), x_n_late=len(late),
        x_gap_min=gap_min, x_n_zero=int(c.n_zero)))
pts = pd.DataFrame(ann)
vis["chosen"] = 0
vis.loc[chosen_idx, "chosen"] = 1
assert not pts.duplicated(["site_id", "year"]).any()
assert set(pts.label_def) <= {"direct_eos", "direct_dated"}

# ------------------------------------------------------------------ 5. v3 와의 거리(읽기만 한다)
v3 = pd.read_csv(F_V3, usecols=["loc_id", "lat", "lon", "region", "source_id"])
v3 = v3[v3.source_id == "F4_direct"].reset_index(drop=True)
v3["macro"] = v3.region.map(lambda r: V3_MACRO.get(r, r))
v3lat, v3lon = v3.lat.values, v3.lon.values


def v3_near(lat, lon, macro):
    cheb = np.maximum(np.abs(v3lat - lat), np.abs(v3lon - lon))
    i = int(cheb.argmin())
    km = haversine_km(lat, lon, v3lat, v3lon)
    other = (v3.macro != macro).values
    j = int(np.where(other, km, np.inf).argmin())
    return dict(cheb_deg=float(cheb[i]), cheb_loc_id=str(v3.loc_id[i]), cheb_region=v3.region[i],
                km_nearest=float(km.min()), km_other_macro=float(km[j]), other_macro=v3.macro[j],
                other_region=v3.region[j])


near = {s.site_id: v3_near(float(s.lat), float(s.lon), s.macro) for s in sites.itertuples()}
sites["v3_cheb_deg"] = [round(near[s]["cheb_deg"], 5) for s in sites.site_id]
sites["v3_cheb_loc_id"] = [near[s]["cheb_loc_id"] for s in sites.site_id]
sites["v3_cheb_region"] = [near[s]["cheb_region"] for s in sites.site_id]
sites["v3_km_other_macro"] = [round(near[s]["km_other_macro"], 1) for s in sites.site_id]
sites["v3_other_macro"] = [near[s]["other_macro"] for s in sites.site_id]
add = []
for r in pts.itertuples():
    nr = near[r.site_id]
    add.append(f"v3 F4_direct 셀 {nr['cheb_loc_id']}({nr['cheb_region']})과 체비쇼프 {nr['cheb_deg']:.4f}° 로 "
               f"{DUP_DEG}° 이내" if nr["cheb_deg"] <= DUP_DEG else "")
pts["notes"] = [n + ("; " + a if a else "") for n, a in zip(pts.notes, add)]
pts["x_v3_dup"] = [1 if a else 0 for a in add]
pts["x_km_other"] = [near[s]["km_other_macro"] for s in pts.site_id]

# ------------------------------------------------------------------ 6. 쓰기
os.makedirs(OUT_DIR, exist_ok=True)
pts = pts.sort_values(["site_id", "year"]).reset_index(drop=True)
pts[POINT_COLS].to_csv(F_POINTS, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
vcols = ["site_id", "date", "year", "month", "doy", "n_records", "mean_cm", "sd_cm", "min_cm", "max_cm", "n_zero",
         "n_gt_probe", "n_plots", "msrType", "contributors", "siteIds", "vegCvr", "fireYr", "tsf", "is_late",
         "in_window", "chosen", "value_sig", "value_mult", "n_norm", "ident_first", "ident_n_sites"]
vout = vis.sort_values(["site_id", "date"])[vcols].copy()
for c in ["mean_cm", "sd_cm", "min_cm", "max_cm"]:
    vout[c] = vout[c].map(lambda x: fmt(x, 4))
vout.to_csv(F_VISITS, index=False, encoding="utf-8", lineterminator="\n")
scols = ["site_id", "site_name", "cntryId", "macro", "distur", "lat", "lon", "lat4", "lon4", "coord_prec_deg",
         "n_rawcoord", "rawcoords", "contributors", "lastNm", "siteIds", "n_siteId", "n_plots", "n_records",
         "years", "ecoregion", "biome", "permaExtent", "coord_ok", "v3_cheb_deg", "v3_cheb_loc_id",
         "v3_cheb_region", "v3_km_other_macro", "v3_other_macro"]
sites[scols].to_csv(F_SITES, index=False, encoding="utf-8", lineterminator="\n")

# ------------------------------------------------------------------ 7. 품질 요약과 메타
pts["x_main"] = ((pts.label_def == "direct_eos") & (pts.disturbed == 0) & (pts.right_censored == 0)
                & (pts.qc_flag == "ok") & (pts.year >= YEAR_MIN_MAIN) & (pts.dup_of == ""))


def count(col, sub=None):
    d = pts if sub is None else sub
    return {str(k): int(v) for k, v in d[col].value_counts().sort_index().items()}


def cells_of(sub: pd.DataFrame) -> pd.DataFrame:
    """지점의 다년 평균을 약 1 km 셀로 모은다(6B.3 의 행 단위). v3 와의 거리는 셀 좌표로 다시 잰다."""
    if len(sub) == 0:
        return pd.DataFrame()
    sm = sub.groupby("site_id").agg(lat=("x_latf", "first"), lon=("x_lonf", "first"), macro=("macro", "first"),
                                    alt=("x_alt", "mean"), n_years=("year", "nunique"),
                                    months=("month", lambda s: ",".join(str(x) for x in sorted(set(s)))),
                                    years=("year", lambda s: ",".join(str(x) for x in sorted(set(s))))).reset_index()
    idx = [cell_index(a, b) for a, b in zip(sm.lat, sm.lon)]
    sm["ky"] = [i[0] for i in idx]
    sm["kx"] = [i[1] for i in idx]
    ce = sm.groupby(["macro", "ky", "kx"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), alt_cm=("alt", "mean"),
                                               n_sites=("site_id", "size"),
                                               months=("months", lambda s: ",".join(sorted(set(",".join(s).split(",")), key=int))),
                                               years=("years", lambda s: ",".join(sorted(set(",".join(s).split(","))))),
                                               ).reset_index()
    ce["block"] = (np.floor(ce.lat / BLOCK_DEG).astype(int) * 100000 + np.floor(ce.lon / BLOCK_DEG).astype(int))
    nn = [v3_near(a, b, m) for a, b, m in zip(ce.lat, ce.lon, ce.macro)]
    ce["v3_cheb_deg"] = [x["cheb_deg"] for x in nn]
    ce["v3_dup"] = (ce.v3_cheb_deg <= DUP_DEG).astype(int)
    ce["km_other_macro"] = [x["km_other_macro"] for x in nn]
    ce["other_macro"] = [x["other_macro"] for x in nn]
    return ce


def cell_summary(ce: pd.DataFrame) -> dict:
    out = {}
    if len(ce) == 0:
        return out
    for mac, g in ce.groupby("macro"):
        keep = g[g.v3_dup == 0]
        indep = keep[keep.km_other_macro > INDEP_KM]
        out[mac] = dict(
            n_cells=int(len(g)), n_blocks=int(g.block.nunique()),
            n_cells_v3_dup=int(g.v3_dup.sum()), n_cells_after_v3_dedup=int(len(keep)),
            n_blocks_after_v3_dedup=int(keep.block.nunique()),
            n_cells_after_dedup_and_100km=int(len(indep)), n_blocks_after_dedup_and_100km=int(indep.block.nunique()),
            min_km_to_other_macro_v3=round(float(g.km_other_macro.min()), 1),
            nearest_other_macro=str(g.loc[g.km_other_macro.idxmin(), "other_macro"]),
            alt_cm=describe(keep.alt_cm) if len(keep) else describe(g.alt_cm),
            months=",".join(sorted(set(",".join(g.months).split(",")), key=int)),
            years=",".join(sorted(set(",".join(g.years).split(",")))),
            cells=[dict(ky=int(r.ky), kx=int(r.kx), lat=round(r.lat, 5), lon=round(r.lon, 5),
                        alt_cm=round(r.alt_cm, 2), n_sites=int(r.n_sites), block=int(r.block), months=r.months,
                        v3_dup=int(r.v3_dup)) for r in g.itertuples()] if len(g) <= 40 else "40셀 초과라 생략")
    return out


main = pts[pts.x_main]
ce_main = cells_of(main)
dated_unb = pts[(pts.label_def == "direct_dated") & (pts.disturbed == 0) & (pts.qc_flag == "ok") & (pts.dup_of == "")]
ce_dated = cells_of(dated_unb)
if len(ce_dated):
    ce_dated = ce_dated.assign(cells_key=list(zip(ce_dated.macro, ce_dated.ky, ce_dated.kx)))
    if len(ce_main):
        mk = set(zip(ce_main.macro, ce_main.ky, ce_main.kx))
        ce_dated_only = ce_dated[~ce_dated.cells_key.isin(mk)]
    else:
        ce_dated_only = ce_dated
else:
    ce_dated_only = ce_dated

by_macro = {}
for mac, g in pts.groupby("macro"):
    by_macro[mac] = dict(
        n_rows=int(len(g)), n_sites=int(g.site_id.nunique()),
        n_sites_unburned=int(g[g.disturbed == 0].site_id.nunique()),
        n_sites_burned=int(g[g.disturbed == 1].site_id.nunique()),
        n_rows_by_label_def=count("label_def", g), n_rows_by_month=count("month", g),
        n_rows_main=int(g.x_main.sum()), n_sites_main=int(g[g.x_main].site_id.nunique()),
        years=f"{int(g.year.min())}-{int(g.year.max())}",
        alt_cm_all=describe(g.x_alt), alt_cm_main=describe(g[g.x_main].x_alt),
        contributors=sorted({c for s in sites[sites.macro == mac].contributors for c in s.split("/")}))

series_oct = pts[(pts.eos_basis == "series_max") & (~pts.month.isin(EOS_MONTHS))]
short_gap = pts[pts.x_gap_min.notna() & (pts.x_gap_min <= 4)]
ident_rows = pts[pts.notes.str.contains("측정값 묶음이 같은")]
try:
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
except Exception:
    head = ""

meta = dict(
    src_id=SRC_ID, name=NAME, url=URL, download_url_pattern=URL_OBJECT + "<식별자>", doi=DOI, accessed=ACCESSED,
    files=[dict(name=f, size_bytes=os.path.getsize(os.path.join(RAW_DIR, f)),
                sha256=sha256(os.path.join(RAW_DIR, f))) for f in RAW_FILES
           if os.path.exists(os.path.join(RAW_DIR, f))],
    license=LICENSE,
    license_basis="EML 메타자료(FireALT_eml.xml)의 intellectualRights 항목: Creative Commons Attribution 4.0 "
                  "International License",
    citation=CITATION,
    parser="scripts/1_data_prep/parse_ext_firealt_talucci2025.py", parser_sha256=sha256(os.path.abspath(__file__)),
    parser_git_commit=head,
    parser_git_note="파서는 커밋하지 않은 작업 트리 파일이다. parser_git_commit 은 실행 시점의 HEAD 다",
    schema_version="1.0",
    n_rows_raw=int(n_raw), n_rows_raw_used=int(n_used), n_rows_points=int(len(pts)),
    n_sites_points=int(pts.site_id.nunique()), n_raw_coordinate_strings=int(raw.rawcoord.nunique()),
    n_visits=int(len(vis)),
    n_by_label_def=count("label_def"), n_by_method=count("method"), n_by_macro=count("macro"),
    n_by_country=count("country"), n_by_eos_basis=count("eos_basis"), n_by_value_kind=count("value_kind"),
    n_by_disturbed=count("disturbed"), n_by_right_censored=count("right_censored"), n_by_qc_flag=count("qc_flag"),
    n_by_month=count("month"), n_by_year=count("year"),
    n_rows_with_dup_of=int((pts.dup_of != "").sum()),
    n_excluded_by_reason={k: int(v) for k, v in excluded.items()},
    n_excluded_note="측정 기록 단위의 제외 수다. 공개 파일에는 결측 부호, 날짜 오류, hitRock = y 기록이 없어 모두 0 이다. "
                    "연 값으로 뽑히지 않은 방문은 지우지 않고 data/raw/firealt_talucci2025/derived_visits.csv 에 남겼다",
    n_visits_not_chosen=int((vis.chosen == 0).sum()),
    n_records_in_chosen_visits=int(vis[vis.chosen == 1].n_records.sum()),
    record_checks=dict(
        n_msrDepth_zero=n_zero_records, n_msrDepth_over_600=n_over_records, n_gtProbe_y=n_gtprobe_y,
        n_msrDoy_mismatch_with_date=n_doy_mismatch, n_full_row_duplicates=n_fullrow_dup,
        n_msrDepth_below_10cm_in_aug_sep=int(((raw.dep < 10) & raw.mo.isin(EOS_MONTHS)).sum()),
        msrDepth=describe(raw.dep),
        note="전 열이 같은 행은 지우지 않았다. 원자료에 측정점 식별자가 없어 같은 plot 의 반복 측정과 "
             "복제 기록을 구분할 수 없다. 평균에는 영향이 없거나 작고 n_obs 는 부풀 수 있다"),
    year_min=int(pts.year.min()), year_max=int(pts.year.max()),
    alt_cm_all_rows=describe(pts.x_alt),
    main_set=dict(
        definition="label_def = direct_eos, disturbed = 0, right_censored = 0, qc_flag = ok, year >= 1990, "
                   "dup_of 빈 칸",
        n_rows=int(len(main)), n_sites=int(main.site_id.nunique()), alt_cm=describe(main.x_alt),
        n_rows_by_macro=count("macro", main), n_rows_by_eos_basis=count("eos_basis", main),
        n_rows_by_month=count("month", main)),
    by_macro=by_macro,
    cell_preview=dict(
        rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 도. 셀 값은 지점 다년 "
             "평균의 평균, 셀 좌표는 지점 좌표의 평균. block = floor(lat/0.5)*100000 + floor(lon/0.5)",
        note="미리보기다. 확정은 v4 조립 단계에서 한다. 다른 자료원의 셀과 합치기 전의 수다",
        note_expected="탐색 단계의 기대값은 Russia_C 7셀이었다. 6B.3 의 0.009° 규칙으로는 미연소 8월 지점 7곳이 "
                      "6셀(블록 2개)이다. 셀 크기를 1/111.32° 로 두면 7셀이 되고 ky 가 탐색 사본"
                      "(ru_unburned_cells_probe.csv)과 같다. kx 는 1 차이가 나는 셀이 있어 탐색 단계의 식은 "
                      "확인하지 못했다",
        v3_dup_rule=f"셀 좌표가 v3 F4_direct 셀과 체비쇼프 {DUP_DEG}° 이내이면 v3_dup = 1",
        independence_rule=f"다른 macro 의 v3 F4_direct 셀에서 {INDEP_KM:.0f} km 안이면 대상 셀에서 뺀다(6B.4)",
        main_set_cells=cell_summary(ce_main),
        direct_dated_unburned_cells_not_in_main=cell_summary(ce_dated_only)),
    special_cases=dict(
        series_max_value_from_month_outside_8_9=dict(
            rule_rev13=("10월 이후 최댓값 방문은 기록 수가 8-9월 방문의 최대 기록 수와 같고 msrType 이 같을 때만 쓴다. "
                        "아니면 8-9월 방문의 최댓값을 쓴다(개정 13). 기각한 행은 notes 에 '10월 최댓값 방문 기각' 으로 적었다"),
            n_rows_rejected_rev13=int(pts.notes.str.contains("10월 최댓값 방문 기각").sum()),
            n_rows=int(len(series_oct)), n_rows_main=int(series_oct.x_main.sum()),
            rows=[dict(site_id=r.site_id, year=int(r.year), date=r.date, alt_cm=round(r.x_alt, 2), macro=r.macro,
                       disturbed=int(r.disturbed)) for r in series_oct.itertuples()]),
        late_visits_within_4_days=dict(
            n_rows=int(len(short_gap)), n_rows_main=int(short_gap.x_main.sum()),
            rows=[dict(site_id=r.site_id, year=int(r.year), date=r.date, alt_cm=round(r.x_alt, 2), macro=r.macro,
                       disturbed=int(r.disturbed)) for r in short_gap.itertuples()]),
        chosen_visit_with_zero_records=dict(
            n_rows=int((pts.x_n_zero > 0).sum()),
            rows=[dict(site_id=r.site_id, year=int(r.year), date=r.date, alt_cm=round(r.x_alt, 2), n_zero=int(r.x_n_zero),
                       label_def=r.label_def, disturbed=int(r.disturbed)) for r in pts[pts.x_n_zero > 0].itertuples()]),
        identical_value_sets=dict(
            n_rows=int(len(ident_rows)), n_rows_with_dup_of=int((pts.dup_of != "").sum()),
            contributors=sorted(set(vis[vis.ident_first != ""].contributors)),
            rule=f"같은 날짜에 값의 다중집합(개수를 최대공약수로 나눈 것)이 같은 지점이 2곳 이상이고 값 수가 "
                 f"{IDENT_MIN_N} 이상인 경우")),
    site_unit="위도, 경도를 소수 4자리로 반올림(Decimal ROUND_HALF_UP)한 좌표와 distur 의 조합. "
              "site_id 는 (cntryId, 위도, 경도, distur) 순의 일련번호다",
    n_sites_with_merged_raw_coordinates=int((sites.n_rawcoord > 1).sum()),
    n_coordinates_with_both_burned_and_unburned=int(
        sites.groupby(["lat4", "lon4"]).distur.nunique().gt(1).sum()),
    coordinate_conversion="원자료는 WGS84 십진 도다(EML 의 lat, lon 설명). 변환하지 않았다. 서경은 음수로 적혀 있다"
                          "(US, CA 기록의 경도는 모두 음수, RU 기록은 모두 양수). 원좌표 문자열이 하나인 지점은 그 "
                          "문자열을 그대로 적었고 둘 이상인 지점은 서로 다른 원좌표의 평균을 소수 6자리로 적었다. "
                          "coord_prec_deg 는 원좌표 문자열의 소수 자리수에서 구했다. 공개 파일은 끝자리 0 을 적지 "
                          "않으므로 실제 정밀도는 이 값보다 좋을 수 있다",
    units="msrDepth 의 단위는 cm 다(EML 의 단위 centimeter). 변환하지 않았다",
    missing_codes="수치 열 -9999, 문자 열 NA. lat, lon, year, month, day, msrDepth 에는 결측 부호가 없다",
    censoring="공개 파일의 gtProbe 와 hitRock 은 모두 n 이다. 자료 제작 스크립트(DataCleanPointCombine.Rmd)가 "
              "gtProbe = y 2,237건과 hitRock = y 853건(스크립트 주석의 수)을 지운 뒤 공개했다. 그래서 "
              "right_censored 는 모두 0 이지만, 탐침 길이를 넘은 측정점이 빠진 지점의 평균은 실제보다 얕을 수 있다. "
              "어느 지점에서 몇 건이 빠졌는지는 공개 파일로 알 수 없다",
    visits_file="data/raw/firealt_talucci2025/derived_visits.csv",
    sites_file="data/raw/firealt_talucci2025/derived_sites.csv",
    v3_proximity=dict(
        n_sites_within_dup_deg=int((sites.v3_cheb_deg <= DUP_DEG).sum()),
        n_sites_within_dup_deg_by_macro={str(k): int(v) for k, v in
                                         sites[sites.v3_cheb_deg <= DUP_DEG].macro.value_counts().items()},
        min_km_to_other_macro_by_macro={str(k): round(float(v), 1) for k, v in
                                        sites.groupby("macro").v3_km_other_macro.min().items()}),
    subunit_rule="Russia_C 는 Yakutia_C, 나머지는 빈 칸. 자료의 생태지역 이름(resName)은 notes 에 적었다",
    unverified_items=[
        "8-9월 방문 기록 가운데 10 cm 미만인 값이 있다(record_checks.n_msrDepth_below_10cm_in_aug_sep). "
        "입력 오류인지 확인하지 못했고 지우지 않았다",
        "Russia_W 의 미연소 2지점(Sizov)은 지점마다 측정 기록이 1건이다(n_obs = 1, 110 cm 와 210 cm). "
        "지점 평균이 아니라 단일 측정값이다",
        "기여자별 원 자료의 인용문과 측정 규약(탐침 길이, plot 안의 측정점 배치)은 확인하지 못했다. 자료 논문"
        "(doi:10.5194/essd-17-2887-2025)의 본문은 읽지 않았다. orig_source 에는 lastNm 열의 기여자 이름만 적었다",
        "탐침 길이 초과 기록과 암석 접촉 기록이 공개 전에 지워졌다. 지점별 제거 수를 알 수 없어 우측 절단에 따른 "
        "얕은 쪽 편향의 크기를 확인하지 못했다",
        "8월 초·중순의 단일 방문 값(direct_eos, record_date)은 계절 최대보다 얕을 수 있다(계획서 6B.9). "
        "기여자가 붙인 msrType 이 thaw 인 8-9월 기록도 규칙에 따라 direct_eos 로 두었다",
        "미연소 지점은 연소 지점의 대조 지점이다. 과거 화재 이력이 없는지는 기여자 판단(distur 열)을 따랐고 "
        "따로 확인하지 못했다",
        "같은 위치가 정밀도가 다른 좌표 문자열로 적힌 경우를 소수 4자리에서 묶었다. 반올림 경계에 걸려 묶이지 "
        "않은 쌍이 남아 있을 수 있다. 1 km 셀 집계에서는 같은 셀에 들어간다",
        "방문 단위는 관측일이다. 같은 조사 기간에 서로 다른 plot 을 며칠에 나눠 잰 경우 날짜별 평균의 최댓값은 "
        "plot 구성의 차이를 포함한다(special_cases.late_visits_within_4_days)",
        "FireALT 기여 자료와 v3 의 ABoVE 자료, CALM 자료가 같은 현장 조사를 담고 있는지는 좌표 거리로만 보았다"
        "(v3_proximity). 원 조사 단위의 대조는 하지 않았다",
    ],
)
with open(F_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"원자료 {n_raw}행, 사용 {n_used}행, 지점 {pts.site_id.nunique()}곳, 방문 {len(vis)}건, 점 자료 {len(pts)}행")
print("label_def", count("label_def"), "| macro", count("macro"))
print("주 집합", len(main), "행,", main.site_id.nunique(), "지점 |", count("macro", main))
for k, v in meta["cell_preview"]["main_set_cells"].items():
    print(" 주 집합 셀", k, {a: b for a, b in v.items() if a != "cells"})
for k, v in meta["cell_preview"]["direct_dated_unburned_cells_not_in_main"].items():
    print(" direct_dated 미연소 셀", k, {a: b for a, b in v.items() if a != "cells"})
