"""ru_kwbs_makarieva: 콜리마 물수지 관측소(KWBS) 융해 깊이 시계열을 표준 점 자료로 만든다.

근거
----
계획서 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 절(LGD, 개정 10), 형식 정의
`data/processed/ext_labels/_schema.json`(버전 1.0).

입력
----
data/raw/ru_kwbs_makarieva/PANGAEA_881754.tab
    PANGAEA 탭 구분 텍스트. 머리말 주석(/* ... */) 뒤에 자료 행렬이 온다.
    열: Event, Latitude, Longitude, Elevation [m], Date/Time, Comment, Station,
        Thaw depth [cm] (upper boundary), Thaw depth [cm] (lower boundary), Snow h [m], Comment.
data/processed/fidelity_base_v3.csv (읽기 전용. 좌표와 source_id 만 읽어 v3 셀과의 거리를 기록한다)

측정 방법
---------
자료 논문(Makarieva 등 2018, ESSD 10:689, 3.6 절)은 기기를 Danilin cryopedometer(frost tube)로 적는다.
증류수를 채운 고무관(외경 1 cm, 눈금 1 cm)을 시추공의 에보나이트 관에 넣고, 관을 꺼내 얼음 기둥의
아래 끝을 읽는다. 따라서 method = frost_tube 다.

규칙
----
1. 지점 단위는 자료 행렬의 Station 열(43개)이다. Event(37개) 하나에 Station 이 둘 이상인 경우가 있다.
2. 읽은 값은 하한 경계 깊이(lower boundary) 열이다. 빈 칸은 값이 없는 행이다(Fully Frozen 등).
   '>' 로 시작하는 값은 기기 깊이를 넘은 하한값이다.
3. 연 값은 달력 연도 안의 읽은 값(숫자와 '>' 값 모두)의 최댓값이다. date 와 month 는 그 값이 처음
   기록된 날이다. 같은 값이 이어진 횟수와 마지막 날은 notes 에 적는다.
4. qc_flag 의 date: 연 최대가 처음 기록된 달이 6–10월 밖이다.
5. label_def: 연 최대가 6–10월에 기록되고 8월 1일 이후 읽은 값의 최댓값과 같으면 direct_eos,
   eos_basis = series_max 다. 8월 1일 이후 읽은 값이 없거나 그 최댓값이 연 최대보다 작으면
   direct_dated, eos_basis = none 이다. qc_flag 에 date 가 붙는 연도도 direct_dated 다.
6. right_censored = 1 인 경우는 둘이다.
   (a) source_mark: 그 연도에 '>' 값이 하나라도 있다. 숫자 최댓값이 '>' 값보다 큰 연도도 포함한다.
   (b) plateau_rule: '>' 값이 없는 연도인데 연 최대가 9월 1일 전에 처음 기록되고 같은 값이
       PLATEAU_MIN_N 회 이상 기록되었다. 이것은 파서의 판단이고 원자료의 표기가 아니다.
       notes 의 censor_basis 로 (a) 와 구분한다.
7. qc_flag: 범위(0 < alt_cm <= 600) 밖은 range, 1990년 이전은 year_pre1990. 여러 개면 세미콜론으로 잇는다.
8. 좌표는 자료 행렬의 표기를 그대로 쓴다(십진 도, 동경 양수). 변환은 없다.
9. 규칙에 맞지 않는 연도도 지우지 않는다. 주 집합은 qc_flag = ok, label_def = direct_eos,
   right_censored = 0, disturbed = 0 인 행이다.

산출
----
data/processed/ext_labels/ru_kwbs_makarieva_points.csv
data/processed/ext_labels/ru_kwbs_makarieva_meta.json

실행(ROOT): OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_ru_kwbs_makarieva.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ID = "ru_kwbs_makarieva"
SRC_NAME = "Makarieva 등 2017, 콜리마 물수지 관측소 융해 깊이 시계열 1954–1997(PANGAEA 881754)"
URL = "https://doi.org/10.1594/PANGAEA.881754"
DOI = "10.1594/PANGAEA.881754"
ACCESSED = "2026-09-29"
LICENSE = "CC BY 3.0"
CITATION = ("Makarieva, O.; Nesterova, N.; Lebedeva, L.; Sushansky, S. (2017): Thaw depth and snow height "
            "time series at different sites within Kolyma Water-Balance Station (KWBS), 1954-1997. PANGAEA, "
            "https://doi.org/10.1594/PANGAEA.881754. In supplement to: Makarieva, O. et al. (2018), "
            "Earth Syst. Sci. Data, 10, 689-710, https://doi.org/10.5194/essd-10-689-2018")
ORIG_SOURCE = ("Observation Reports of the Kolyma Water-Balance Station (1959-1997), Hydrometeorological "
               "Service of the USSR and Russia; digitized by O. Makarieva and N. Nesterova")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
RAW = RAW_DIR / "PANGAEA_881754.tab"
RAW_AUX = [RAW_DIR / "KWBS_gauges_details.zip", RAW_DIR / "essd-10-689-2018.pdf"]
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

EXPECTED_COLS = ["Event", "Latitude", "Longitude", "Elevation [m]", "Date/Time", "Comment", "Station",
                 "Thaw depth [cm] (upper boundary)", "Thaw depth [cm] (lower boundary)", "Snow h [m]", "Comment.1"]
COLS = ["event", "lat", "lon", "elev", "dt", "ev_comment", "station", "up", "low", "snow", "comment"]

YEAR_MIN_MAIN = 1990
ALT_MAX_CM = 600.0
MAX_MONTHS_OK = (6, 7, 8, 9, 10)       # 연 최대가 기록될 수 있는 달(자료원 규칙)
EOS_FROM = (8, 1)                      # 계절 말 창의 시작(계획서 6B.3 의 series_max)
PLATEAU_MIN_N = 10                     # 같은 값이 이어진 횟수의 기준(5일 간격으로 약 45일)
PLATEAU_FIRST_BEFORE = (9, 1)          # 연 최대가 이 날 전에 처음 기록된 경우만 본다
CELL_DEG = 0.009

POINT_COLS = [
    "src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def", "n_obs",
    "country", "macro", "citation", "license",
    "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
    "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
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


def read_pangaea(path: Path) -> tuple[str, pd.DataFrame]:
    """머리말 주석과 자료 행렬을 나누어 읽는다. 모든 열은 문자열로 둔다."""
    txt = path.read_text(encoding="utf-8")
    end = txt.index("\n*/\n")
    header, body = txt[:end], txt[end + 4:]
    df = pd.read_csv(io.StringIO(body), sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) != EXPECTED_COLS:
        raise SystemExit(f"[중단] 자료 행렬의 열이 예상과 다르다: {list(df.columns)}")
    df.columns = COLS
    return header, df


def parse_events(header: str) -> dict[str, dict]:
    """머리말의 Event 목록에서 좌표를 읽는다(자료 행렬의 좌표와 대조하는 데만 쓴다)."""
    ev = {}
    pat = re.compile(r"(KWBS_cryo_\S+) \(([^)]*)\) \* LATITUDE: ([-\d.]+) \* LONGITUDE: ([-\d.]+)")
    for m in pat.finditer(header):
        ev[m.group(1)] = dict(label=m.group(2), lat=float(m.group(3)), lon=float(m.group(4)))
    return ev


def parse_low(s: str) -> tuple[float, int]:
    """하한 경계 값을 (값, 검열 여부)로 바꾼다. 빈 칸은 (nan, 0)."""
    s = s.strip()
    if s == "":
        return (np.nan, 0)
    if s.startswith(">"):
        return (float(s[1:]), 1)
    if s.startswith("<"):
        raise SystemExit(f"[중단] '<' 표기는 예상하지 않았다: {s}")
    return (float(s), 0)


def site_id_of(station: str) -> str:
    return "kwbs_" + station.strip().replace(" ", "-")


def cell_of(lat: float, lon: float) -> tuple[int, int]:
    ky = math.floor(lat / CELL_DEG)
    phi = (ky + 0.5) * CELL_DEG
    kx = math.floor(lon * math.cos(math.radians(phi)) / CELL_DEG)
    return ky, kx


def haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat0), np.radians(lat)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_points(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = OrderedDict()
    stats["n_rows_raw"] = int(len(df))
    n_exact_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    stats["n_rows_exact_duplicate_dropped"] = n_exact_dup

    df["date"] = pd.to_datetime(df["dt"], format="%Y-%m-%d", errors="raise")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    parsed = df["low"].map(parse_low)
    df["lowv"] = [p[0] for p in parsed]
    df["cens"] = [p[1] for p in parsed]

    no_val = df[df["lowv"].isna()]
    stats["n_rows_no_lower_value"] = int(len(no_val))
    stats["n_rows_no_lower_value_fully_frozen"] = int((no_val["comment"] == "Fully Frozen").sum())
    stats["n_rows_no_lower_value_blank"] = int((no_val["comment"] != "Fully Frozen").sum())
    stats["n_rows_censored_mark"] = int(df["cens"].sum())
    stats["n_rows_numeric"] = int((df["lowv"].notna() & (df["cens"] == 0)).sum())
    dup_sd = df[df["lowv"].notna()].duplicated(["station", "dt"], keep=False)
    stats["n_rows_same_station_date"] = int(dup_sd.sum())

    rows = []
    n_sy_total, n_sy_no_reading = 0, 0
    for (station, year), g in df.groupby(["station", "year"], sort=False):
        n_sy_total += 1
        r = g[g["lowv"].notna()].sort_values("date")
        if len(r) == 0:
            n_sy_no_reading += 1
            continue
        alt = float(r["lowv"].max())
        at_max = r[r["lowv"] == alt]
        d_first, d_last = at_max["date"].iloc[0], at_max["date"].iloc[-1]
        n_at_max = int(len(at_max))
        month = int(d_first.month)
        n_cens = int(r["cens"].sum())
        late = r[r["date"] >= pd.Timestamp(year=int(year), month=EOS_FROM[0], day=EOS_FROM[1])]
        late_max = float(late["lowv"].max()) if len(late) else np.nan

        qc = []
        if not (0 < alt <= ALT_MAX_CM):
            qc.append("range")
        date_bad = month not in MAX_MONTHS_OK
        if date_bad:
            qc.append("date")
        if int(year) < YEAR_MIN_MAIN:
            qc.append("year_pre1990")

        if (not date_bad) and len(late) > 0 and late_max == alt:
            label_def, eos_basis = "direct_eos", "series_max"
        else:
            label_def, eos_basis = "direct_dated", "none"

        censor_basis = ""
        if n_cens > 0:
            censor_basis = "source_mark"
        elif (n_at_max >= PLATEAU_MIN_N
              and d_first < pd.Timestamp(year=int(year), month=PLATEAU_FIRST_BEFORE[0],
                                         day=PLATEAU_FIRST_BEFORE[1])):
            censor_basis = "plateau_rule"
        right_censored = 1 if censor_basis else 0

        notes = [f"event={g['event'].iloc[0]}"]
        elev = g["elev"].iloc[0].strip()
        if elev:
            notes.append(f"elev_m={elev}")
        notes += [f"n_at_max={n_at_max}", f"max_last={d_last.date().isoformat()}",
                  f"obs_first={r['date'].iloc[0].date().isoformat()}",
                  f"obs_last={r['date'].iloc[-1].date().isoformat()}"]
        if censor_basis:
            notes.append(f"censor_basis={censor_basis}")
        if n_cens > 0:
            cmax = float(r.loc[r["cens"] == 1, "lowv"].max())
            num = r.loc[r["cens"] == 0, "lowv"]
            notes.append(f"n_cens={n_cens}")
            notes.append(f"cens_max_cm={cmax:g}")
            if len(num):
                notes.append(f"num_max_cm={float(num.max()):g}")
        if date_bad:
            win = r[r["month"].isin(MAX_MONTHS_OK)]
            if len(win):
                wmax = float(win["lowv"].max())
                wd = win[win["lowv"] == wmax]["date"].iloc[0]
                notes.append(f"max_jun_oct_cm={wmax:g}@{wd.date().isoformat()}")
        if label_def == "direct_dated" and not date_bad:
            notes.append("late_none" if len(late) == 0 else f"late_max_cm={late_max:g}")
        n_dup_dates = int(r.duplicated(["dt"], keep=False).sum())
        if n_dup_dates:
            notes.append(f"n_same_date_rows={n_dup_dates}")
            if at_max["dt"].isin(r.loc[r.duplicated(["dt"], keep=False), "dt"]).any():
                notes.append("max_on_same_date_rows")

        rows.append(OrderedDict(
            src_id=SRC_ID,
            site_id=site_id_of(station),
            site_name=f"KWBS cryopedometer {station.strip()}: {g['ev_comment'].iloc[0].strip()}",
            lat=g["lat"].iloc[0], lon=g["lon"].iloc[0],
            year=int(year), month=month,
            alt_cm=f"{alt:g}",
            method="frost_tube", label_def=label_def, n_obs=int(len(r)),
            country="Russia", macro="Russia_E", citation=CITATION, license=LICENSE,
            subunit="Kolyma_KWBS", date=d_first.date().isoformat(), eos_basis=eos_basis,
            value_kind="annual_value", year_min="", year_max="", alt_sd_cm="", coord_prec_deg=0.001,
            disturbed=0, disturb_type="", right_censored=right_censored, orig_source=ORIG_SOURCE,
            dup_of="", qc_flag=";".join(qc) if qc else "ok", notes=";".join(notes),
        ))
    stats["n_station_years"] = n_sy_total
    stats["n_station_years_no_reading"] = n_sy_no_reading
    pts = pd.DataFrame(rows, columns=POINT_COLS)
    pts = pts.sort_values(["site_id", "year"], kind="stable").reset_index(drop=True)
    return pts, stats


def check_coords(df: pd.DataFrame, events: dict) -> dict:
    """좌표 표기, 부호, 머리말과의 일치를 확인한다."""
    out = {}
    la = df["lat"].astype(float)
    lo = df["lon"].astype(float)
    out["lat_range"] = [float(la.min()), float(la.max())]
    out["lon_range"] = [float(lo.min()), float(lo.max())]
    if not ((la.between(61.8, 61.9)).all() and (lo.between(147.5, 147.8)).all()):
        raise SystemExit("[중단] 좌표가 관측소 영역(61.8–61.9°N, 147.5–147.8°E) 밖에 있다")
    n_per_station = df.groupby("station")[["lat", "lon"]].nunique().max().max()
    if n_per_station != 1:
        raise SystemExit("[중단] 한 Station 에 좌표가 둘 이상이다")
    mism = []
    for ev, g in df.groupby("event"):
        if ev not in events:
            mism.append(f"{ev}: 머리말에 없음")
            continue
        if abs(float(g["lat"].iloc[0]) - events[ev]["lat"]) > 5e-5 or \
                abs(float(g["lon"].iloc[0]) - events[ev]["lon"]) > 5e-5:
            mism.append(f"{ev}: 머리말과 좌표가 다름")
    out["header_mismatch"] = mism
    out["lon_4th_decimal_all_zero"] = bool(df["lon"].str.endswith("0").all())
    out["n_unique_coords"] = int(df[["lat", "lon"]].drop_duplicates().shape[0])
    return out


def summarize(pts: pd.DataFrame) -> dict:
    s: dict = OrderedDict()
    alt = pts["alt_cm"].astype(float)
    rc = pts["right_censored"].astype(int)
    s["n_sites"] = int(pts["site_id"].nunique())
    s["n_rows_points"] = int(len(pts))
    s["year_range_all"] = [int(pts["year"].min()), int(pts["year"].max())]
    s["alt_cm_all"] = dict(min=float(alt.min()), median=float(alt.median()), max=float(alt.max()))
    s["n_by_macro"] = {k: int(v) for k, v in pts["macro"].value_counts().items()}
    s["n_by_method"] = {k: int(v) for k, v in pts["method"].value_counts().items()}
    s["n_by_label_def"] = {k: int(v) for k, v in pts["label_def"].value_counts().items()}
    s["n_by_eos_basis"] = {k: int(v) for k, v in pts["eos_basis"].value_counts().items()}
    s["n_by_qc_flag"] = {k: int(v) for k, v in pts["qc_flag"].value_counts().items()}
    basis = pts["notes"].str.extract(r"censor_basis=([a-z_]+)")[0].fillna("none")
    s["n_by_censor_basis"] = {k: int(v) for k, v in basis.value_counts().items()}
    s["n_by_max_month"] = {int(k): int(v) for k, v in pts["month"].value_counts().sort_index().items()}

    has = lambda f: pts["qc_flag"].str.split(";").map(lambda x: f in x)
    pre, dat, rng = has("year_pre1990"), has("date"), has("range")
    dd = pts["label_def"] != "direct_eos"
    cen = rc == 1
    # 겹침을 허용한 수
    s["n_flag_any"] = dict(year_pre1990=int(pre.sum()), date=int(dat.sum()), range=int(rng.sum()),
                           label_def_not_direct_eos=int(dd.sum()), right_censored=int(cen.sum()))
    # 순서대로 처음 걸린 사유(합이 제외 행 수와 같다)
    first = pd.Series("main", index=pts.index)
    for name, mask in [("year_pre1990", pre), ("range", rng), ("date", dat),
                       ("label_def_not_direct_eos", dd), ("right_censored", cen)]:
        first[(first == "main") & mask] = name
    s["n_excluded_by_reason"] = {k: int(v) for k, v in first.value_counts().items() if k != "main"}
    main = pts[first == "main"]
    s["n_main"] = int(len(main))
    s["n_main_sites"] = int(main["site_id"].nunique())

    # 1990년 이후 행의 내역
    post = pts[~pre]
    s["post1990"] = OrderedDict(
        n_rows=int(len(post)), n_sites=int(post["site_id"].nunique()),
        sites=sorted(post["site_id"].unique().tolist()),
        n_right_censored=int((post["right_censored"].astype(int) == 1).sum()),
        n_by_censor_basis={k: int(v) for k, v in basis[post.index].value_counts().items()},
        n_by_max_month={int(k): int(v) for k, v in post["month"].value_counts().sort_index().items()},
    )
    if len(main):
        ma = main["alt_cm"].astype(float)
        s["main"] = OrderedDict(
            year_range=[int(main["year"].min()), int(main["year"].max())],
            alt_cm=dict(min=float(ma.min()), median=float(ma.median()), max=float(ma.max())),
            n_by_max_month={int(k): int(v) for k, v in main["month"].value_counts().sort_index().items()},
            n_max_month_aug_sep=int(main["month"].isin([8, 9]).sum()),
            per_site=[],
        )
        cells: dict = OrderedDict()
        for sid, g in main.groupby("site_id"):
            la, lo = float(g["lat"].iloc[0]), float(g["lon"].iloc[0])
            ky, kx = cell_of(la, lo)
            m = float(g["alt_cm"].astype(float).mean())
            s["main"]["per_site"].append(dict(
                site_id=sid, lat=g["lat"].iloc[0], lon=g["lon"].iloc[0], n_years=int(len(g)),
                years=[int(y) for y in g["year"]], alt_mean_cm=round(m, 2),
                alt_min_cm=float(g["alt_cm"].astype(float).min()),
                alt_max_cm=float(g["alt_cm"].astype(float).max()), cell=[ky, kx],
                block_0p5deg=[math.floor(la / 0.5), math.floor(lo / 0.5)]))
            cells.setdefault((ky, kx), []).append((sid, la, lo, m))
        s["main"]["cells_1km"] = [dict(cell=list(k), n_sites=len(v), sites=[x[0] for x in v],
                                       lat=round(float(np.mean([x[1] for x in v])), 6),
                                       lon=round(float(np.mean([x[2] for x in v])), 6),
                                       alt_cm=round(float(np.mean([x[3] for x in v])), 2))
                                  for k, v in cells.items()]
        s["main"]["n_cells_1km"] = len(cells)
        s["main"]["n_blocks_0p5deg"] = len({tuple(p["block_0p5deg"]) for p in s["main"]["per_site"]})
    return s


def v3_relation(pts: pd.DataFrame) -> dict:
    v = pd.read_csv(V3, usecols=["loc_id", "lat", "lon", "region", "source_id"])
    la0 = float(pts["lat"].astype(float).mean())
    lo0 = float(pts["lon"].astype(float).mean())
    d = haversine_km(la0, lo0, v["lat"].to_numpy(), v["lon"].to_numpy())
    i = int(np.argmin(d))
    f = v["source_id"].to_numpy() == "F4_direct"
    j = int(np.argmin(np.where(f, d, np.inf)))
    coords = pts[["lat", "lon"]].drop_duplicates().astype(float).to_numpy()
    vf = v.loc[f, ["lat", "lon"]].to_numpy()
    n_dup = 0
    for la, lo in coords:
        cheb = np.maximum(np.abs(vf[:, 0] - la), np.abs(vf[:, 1] - lo))
        n_dup += int((cheb <= 0.01).any())
    return dict(center=[round(la0, 4), round(lo0, 4)],
                nearest_v3_any=dict(loc_id=int(v["loc_id"].iloc[i]), region=str(v["region"].iloc[i]),
                                    source_id=str(v["source_id"].iloc[i]), dist_km=round(float(d[i]), 1)),
                nearest_v3_f4_direct=dict(loc_id=int(v["loc_id"].iloc[j]), region=str(v["region"].iloc[j]),
                                          dist_km=round(float(d[j]), 1)),
                n_coords_within_cheb_0p01_of_v3_f4_direct=n_dup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SRC_ID}_points.csv"
    out_meta = out_dir / f"{SRC_ID}_meta.json"

    header, df = read_pangaea(RAW)
    if "CC-BY-3.0" not in header:
        raise SystemExit("[중단] 머리말에서 약관(CC-BY-3.0) 표기를 찾지 못했다")
    events = parse_events(header)
    coords = check_coords(df, events)
    pts, stats = build_points(df)

    # 형식 확인
    assert list(pts.columns) == POINT_COLS
    assert not pts.duplicated(["site_id", "year"]).any(), "(site_id, year) 가 겹친다"
    assert pts["alt_cm"].astype(float).notna().all()
    pts.to_csv(out_csv, index=False, encoding="utf-8")

    summ = summarize(pts)
    rel = v3_relation(pts)
    files = [dict(name=str(p.relative_to(ROOT)), size_bytes=p.stat().st_size, sha256=sha256_of(p))
             for p in [RAW] + [a for a in RAW_AUX if a.exists()]]
    meta = OrderedDict(
        src_id=SRC_ID, name=SRC_NAME, url=URL, doi=DOI, accessed=ACCESSED,
        download_url="https://doi.pangaea.de/10.1594/PANGAEA.881754?format=textfile",
        files=files, license=LICENSE,
        license_basis="PANGAEA 머리말의 License 항목(Creative Commons Attribution 3.0 Unported)",
        citation=CITATION,
        parser=str(Path(__file__).resolve().relative_to(ROOT)),
        parser_sha256=sha256_of(Path(__file__).resolve()),
        parser_git_commit=git_head(),
        parser_git_note="파서 파일은 이 커밋에 들어 있지 않다(작업 트리의 새 파일). 값은 실행 시점의 HEAD 다",
        points_file=str(out_csv.resolve().relative_to(ROOT)) if out_csv.resolve().is_relative_to(ROOT)
        else str(out_csv),
        points_sha256=sha256_of(out_csv),
        n_rows_raw=stats["n_rows_raw"], n_rows_points=int(len(pts)),
        n_by_label_def=summ["n_by_label_def"], n_by_method=summ["n_by_method"],
        n_excluded_by_reason=summ["n_excluded_by_reason"],
        n_raw_rows_not_used=OrderedDict(
            exact_duplicate=stats["n_rows_exact_duplicate_dropped"],
            no_lower_value=stats["n_rows_no_lower_value"],
            no_lower_value_fully_frozen=stats["n_rows_no_lower_value_fully_frozen"],
            no_lower_value_blank=stats["n_rows_no_lower_value_blank"]),
        raw_stats=stats,
        coordinate_conversion="없음. 원자료가 WGS84 십진 도이고 동경 양수다. 자료 행렬의 표기를 그대로 썼다",
        coordinate_check=coords,
        method_basis=("자료 논문 ESSD 10:689 의 3.6 절: 'Danilin cryopedometers (frost tubes) were installed at "
                      "permafrost observation sites'. 증류수를 채운 고무관(외경 1 cm, 눈금 1 cm)의 얼음 기둥 "
                      "아래 끝을 읽는다"),
        rules=OrderedDict(
            annual_value="달력 연도 안의 하한 경계 값(숫자와 '>' 값)의 최댓값. date 는 처음 기록된 날",
            qc_date=f"연 최대가 처음 기록된 달이 {MAX_MONTHS_OK[0]}–{MAX_MONTHS_OK[-1]}월 밖",
            label_def=("연 최대가 6–10월에 기록되고 8월 1일 이후 읽은 값의 최댓값과 같으면 direct_eos"
                       "(series_max), 아니면 direct_dated"),
            right_censored_source_mark="그 연도에 '>' 값이 하나라도 있다",
            right_censored_plateau_rule=(f"'>' 값이 없는 연도에서 연 최대가 {PLATEAU_FIRST_BEFORE[0]}월 "
                                         f"{PLATEAU_FIRST_BEFORE[1]}일 전에 처음 기록되고 같은 값이 "
                                         f"{PLATEAU_MIN_N}회 이상 기록되었다. 파서의 판단이다"),
            main_set="qc_flag = ok, label_def = direct_eos, right_censored = 0, disturbed = 0",
        ),
        summary=summ,
        v3_relation=rel,
        unverified_items=[
            "기기 깊이(관의 길이)는 보조 자료 cryopedometer_gauges.txt 에 6개 지점만 적혀 있다. 1990–1997년에 "
            "쓰인 지점 8, 17.5, 23 의 기기 깊이는 확인하지 못했다(지점 12 는 1.5 m 로 적혀 있다)",
            "plateau_rule 로 표시한 연도가 실제로 기기 깊이를 넘었는지는 원자료로 확인할 수 없다",
            "'>' 표기가 없고 plateau_rule 에도 걸리지 않은 연도 가운데 기기 깊이를 넘은 연도가 있을 수 있다"
            "(예: 지점 12 의 1957년은 150 cm 가 9월 2일부터 27회 기록되었다)",
            "자료 논문은 기기 주변의 지표 훼손이 융해 깊이에 영향을 주기 시작했다고 적는다(Sushansky, 1988). "
            "지점과 시기는 적혀 있지 않아 disturbed 는 모두 0 으로 두었다",
            "같은 지점과 날짜에 값이 다른 행이 둘인 경우가 있다(날짜 입력 오류로 보이나 확인하지 못했다). "
            "두 행을 모두 읽은 값으로 썼고 notes 에 n_same_date_rows 로 적었다. 1990년 이후에는 없다",
            "PANGAEA 이벤트 KWBS_cryo_23 은 보조 자료의 22번과 위치 설명이 같다. 번호가 다른 이유는 확인하지 "
            "못했다. Station 19(1980년 한 해)와 19.1 의 관계도 확인하지 못했다",
            "좌표는 위도 소수 4자리, 경도 소수 3자리다(자료 행렬의 경도 넷째 자리는 모두 0). 같은 좌표를 "
            "여러 지점이 함께 쓰는 경우가 있다(17–17.7 의 8개 지점 등). coord_prec_deg 는 0.001 로 두었다",
            "Snow h 열은 머리글이 [m] 이나 값의 크기와 보조 자료의 설명(H, cm)으로 보아 cm 로 보인다. "
            "이 파서는 적설 열을 쓰지 않는다",
            "관측 연도(1990–1997)와 공변량 기후값 기간(ERA5-Land 2015–2020, CCI 1997–2021)이 겹치지 않는다",
        ],
    )
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print("[점 자료]", out_csv, len(pts), "행")
    print("[메타]", out_meta)
    print("[원자료]", json.dumps(stats, ensure_ascii=False))
    print("[요약]", json.dumps({k: v for k, v in summ.items() if k != "main"}, ensure_ascii=False))
    if "main" in summ:
        print("[주 집합]", json.dumps(summ["main"], ensure_ascii=False))
    print("[v3 관계]", json.dumps(rel, ensure_ascii=False))
    print("[좌표]", json.dumps(coords, ensure_ascii=False))


if __name__ == "__main__":
    main()
