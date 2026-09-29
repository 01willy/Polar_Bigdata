#!/usr/bin/env python3
"""ext_labels 파서: grl_disko_zastruzny (Zastruzny 등 2024, PANGAEA 967139).

자료
- 그린란드 서부 디스코섬 사면의 탐침 측선이다. 이벤트 Disko2015_frost_table 은
  2015-08-04 와 2015-08-21 두 방문에서 같은 38점의 동결면 깊이(cm)를 기록한다.
- 원자료 파일 P967139.tab 에는 적설 두께, 지하수위, 전기 전도도, 표고 기록이 함께 있다.
  이 파서는 동결면 깊이 이벤트만 쓴다.

입력
- data/raw/grl_disko_zastruzny/P967139.tab
  (curl -L -o P967139.tab 'https://doi.pangaea.de/10.1594/PANGAEA.967139?format=textfile')
- data/raw/grl_disko_zastruzny/Zastruzny2024_WRR_e2023WR036147.pdf (선택, 관련 논문 전문. 파일 목록에만 쓴다)
  (curl -L -o Zastruzny2024_WRR_e2023WR036147.pdf 'https://umu.diva-portal.org/smash/get/diva2:1917379/FULLTEXT01')

산출
- data/raw/grl_disko_zastruzny/P967139_frost_table.tsv       이벤트 행 추출본(원자료 열 그대로)
- data/raw/grl_disko_zastruzny/frost_table_by_visit.csv      방문 단위 점 자료(점 번호, 구간, 검열 표시)
- data/processed/ext_labels/grl_disko_zastruzny_points.csv   표준 점 자료(_schema.json 1.0)
- data/processed/ext_labels/grl_disko_zastruzny_meta.json    메타와 품질 요약

라벨 규칙(계획서 6B.3)
- method = probe. 연 값은 8월 1일 이후 방문의 점 평균 가운데 최댓값이다
  (label_def = direct_eos, eos_basis = series_max). 평균이 같으면 늦은 방문을 쓴다.
- 두 방문의 점 구성(좌표)이 같은지 확인하고 다르면 공통 점만 쓴다.
- 값 120 cm 는 자료의 최댓값이고 두 방문에서 같은 상부 4점에 반복된다. 관련 논문
  (Zastruzny 등 2024, 5.1절, 8쪽)은 사면 상부의 활동층 두께가 1.2 m 를 넘는다고 적는다.
  따라서 이 값은 하한이다. 탐침 막대 길이는 논문과 자료 설명에 없다. 이 점들은 별도
  구간(U)으로 나누고 right_censored = 1 로 표시한다. 나머지 점은 구간 L 이다.
- 기록은 지우지 않는다. 검열 의심 점은 표시와 함께 점 자료에 남긴다.

실행(저장소 최상위에서, 스레드 1개)
  OMP_NUM_THREADS=1 python3 scripts/1_data_prep/parse_ext_grl_disko_zastruzny.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ID = "grl_disko_zastruzny"
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
RAW_FILE = RAW_DIR / "P967139.tab"
RAW_SUBSET = RAW_DIR / "P967139_frost_table.tsv"
RAW_VISIT = RAW_DIR / "frost_table_by_visit.csv"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
SCHEMA = OUT_DIR / "_schema.json"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
PAPER_PDF = RAW_DIR / "Zastruzny2024_WRR_e2023WR036147.pdf"
PAPER_URL_OA = "https://umu.diva-portal.org/smash/get/diva2:1917379/FULLTEXT01"

URL = "https://doi.org/10.1594/PANGAEA.967139"
URL_FILE = "https://doi.pangaea.de/10.1594/PANGAEA.967139?format=textfile"
DOI = "10.1594/PANGAEA.967139"
ACCESSED = "2026-09-29"
EVENT = "Disko2015_frost_table"
PROBE_LIMIT_CM = 120.0          # 탐침 길이 한계로 추정한 값(확인하지 못함)
EOS_MONTHS = (8, 9)             # 계절 말 창
RANGE_CM = (0.0, 600.0)         # 0 < alt_cm <= 600
BBOX = dict(lat=(69.0, 70.5), lon=(-55.5, -51.5))   # 디스코섬 범위(좌표 부호 확인용)
CELL_DEG = 0.009
TRENCH_DATE = "2015-08-06"       # 논문 3절: 추적자 트렌치 굴착일
TRENCH_POS_M = (15.0, 44.4)     # 논문 3절: 트렌치의 사면 위치(m). 원점은 상부로 추정(논문 그림 6)
R_EARTH_M = 6371008.8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pangaea(path):
    """PANGAEA 탭 파일을 머리말(dict)과 문자열 표로 읽는다."""
    head, skip = [], None
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("*/"):
                skip = i + 1
                break
            head.append(line.rstrip("\n"))
    if skip is None:
        raise RuntimeError("머리말 끝(*/)을 찾지 못했다")
    info = {}
    for line in head:
        if "\t" in line and not line.startswith("\t"):
            k, v = line.split("\t", 1)
            info[k.rstrip(":")] = v.strip()
    df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, keep_default_na=False)
    return info, df, skip


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(a)) / 1000.0


def cell_index(lat, lon):
    ky = int(np.floor(lat / CELL_DEG))
    phi = (ky + 0.5) * CELL_DEG
    kx = int(np.floor(lon * np.cos(np.radians(phi)) / CELL_DEG))
    return ky, kx


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(n=int(x.size), mean=round(float(x.mean()), 4),
                sd=round(float(x.std(ddof=1)), 4) if x.size > 1 else 0.0,
                min=float(x.min()), median=float(np.median(x)), max=float(x.max()))


def main():
    if not RAW_FILE.exists():
        sys.exit(f"원자료가 없다: {RAW_FILE}")
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    columns = [c["name"] for c in schema["columns"]]

    info, df, skip = read_pangaea(RAW_FILE)
    n_raw = len(df)
    citation = info["Citation"].rstrip(", ")
    license_raw = info.get("License", "")
    if "CC-BY-4.0" not in license_raw:
        sys.exit(f"약관이 예상과 다르다: {license_raw}")
    license_txt = "CC-BY-4.0"

    col = list(df.columns)
    c_lon, c_lat, c_elev, c_dt, c_thaw = col[2], col[3], col[4], col[5], col[6]
    assert c_lon.startswith("Longitude") and c_lat.startswith("Latitude"), col
    assert c_thaw.startswith("Thaw depth [cm]"), c_thaw     # 단위 cm 확인

    # 동결면 깊이 값은 해당 이벤트에만 있어야 한다
    n_thaw_other = int(((df["Event"] != EVENT) & (df[c_thaw] != "")).sum())
    ft = df[df["Event"] == EVENT].copy()
    n_event = len(ft)
    header_line = open(RAW_FILE, encoding="utf-8").read().split("\n")[skip]
    with open(RAW_SUBSET, "w", encoding="utf-8", newline="") as f:
        f.write(header_line + "\n")
        ft.to_csv(f, sep="\t", header=False, index=False, lineterminator="\n")

    # ---------- 방문 단위 점 자료 ----------
    v = pd.DataFrame(dict(lon_str=ft[c_lon], lat_str=ft[c_lat], elev_m=ft[c_elev],
                          datetime=ft[c_dt], thaw_str=ft[c_thaw])).reset_index(drop=True)
    n_missing = int((v.thaw_str.str.strip() == "").sum())
    n_gt = int(v.thaw_str.str.contains(r"[<>]").sum())      # '>' 표기 여부
    v["thaw_cm"] = pd.to_numeric(v.thaw_str, errors="coerce")
    n_nonnum = int(v.thaw_cm.isna().sum()) - n_missing
    v["lat"] = v.lat_str.astype(float)
    v["lon"] = v.lon_str.astype(float)
    v["date"] = v.datetime.str.slice(0, 10)
    dt = pd.to_datetime(v.date, format="%Y-%m-%d")
    v["year"], v["month"] = dt.dt.year, dt.dt.month
    n_dup = int(v.duplicated(subset=["lon_str", "lat_str", "date"]).sum())
    n_coord_bad = int((~v.lat.between(*BBOX["lat"]) | ~v.lon.between(*BBOX["lon"])).sum())
    n_range_bad = int((~((v.thaw_cm > RANGE_CM[0]) & (v.thaw_cm <= RANGE_CM[1]))).sum())

    visits = sorted(v.date.unique())
    sets = {d: set(zip(v.lon_str[v.date == d], v.lat_str[v.date == d])) for d in visits}
    common = set.intersection(*sets.values())
    same_points = all(s == common for s in sets.values())
    n_not_common = int((~pd.Series(list(zip(v.lon_str, v.lat_str))).isin(common)).sum())
    v = v[[k in common for k in zip(v.lon_str, v.lat_str)]].reset_index(drop=True)

    # 점 번호: 첫 방문의 파일 순서(사면 하부에서 상부로)
    first = v[v.date == visits[0]].reset_index(drop=True)
    pid = {k: i + 1 for i, k in enumerate(zip(first.lon_str, first.lat_str))}
    v["point_no"] = [pid[k] for k in zip(v.lon_str, v.lat_str)]

    # 사면 위치: 상부 끝 점(마지막 점)에서 측선을 따라 잰 거리(m). 논문의 위치 축(상부 0-10 m,
    # 하부 70-80 m, 그림 6)과 방향이 같다고 추정한다. 원점이 같은지는 확인하지 못했다
    la_, lo_ = first.lat.values, first.lon.values
    step_m = haversine_km(la_[:-1], lo_[:-1], la_[1:], lo_[1:]) * 1000.0
    cum_m = np.concatenate([[0.0], np.cumsum(step_m)])
    pos_top = {i + 1: round(float(cum_m[-1] - c), 2) for i, c in enumerate(cum_m)}
    v["pos_from_top_m"] = v.point_no.map(pos_top)

    # 검열 의심 점: 어느 방문에서든 값이 탐침 한계 추정값 이상인 점
    cens_by_visit = {d: set(v.point_no[(v.date == d) & (v.thaw_cm >= PROBE_LIMIT_CM)]) for d in visits}
    cens_pts = set.union(*cens_by_visit.values())
    cens_same = all(s == cens_pts for s in cens_by_visit.values())
    v["censor_suspect"] = v.point_no.isin(cens_pts).astype(int)
    v["segment"] = np.where(v.censor_suspect == 1, "U", "L")
    v.sort_values(["date", "point_no"], inplace=True)
    v[["point_no", "segment", "lon_str", "lat_str", "elev_m", "pos_from_top_m", "date", "thaw_cm",
       "censor_suspect"]].rename(
        columns={"lon_str": "lon", "lat_str": "lat"}).to_csv(RAW_VISIT, index=False, lineterminator="\n")

    # 방문 사이 변화(같은 점)
    w = v.pivot(index="point_no", columns="date", values="thaw_cm")
    d_all = (w[visits[-1]] - w[visits[0]])
    change = dict(mean_cm=round(float(d_all.mean()), 4),
                  n_deeper=int((d_all > 0).sum()), n_equal=int((d_all == 0).sum()),
                  n_shallower=int((d_all < 0).sum()))
    trench = {}
    for tp in TRENCH_POS_M:
        pn = min(pos_top, key=lambda k: abs(pos_top[k] - tp))
        trench[f"{tp:.1f}"] = dict(nearest_point=int(pn), pos_from_top_m=pos_top[pn],
                                   thaw_cm={d: float(w.loc[pn, d]) for d in visits})

    # ---------- 구간별 연 값 ----------
    seg_def = {
        "L": dict(name="Disko Island slope transect, lower segment (frost table reached)",
                  censored=0),
        "U": dict(name="Disko Island slope transect, upper segment (thaw deeper than 120 cm, right-censored)",
                  censored=1),
    }
    rows, seg_stats = [], {}
    for seg, sd in seg_def.items():
        s = v[v.segment == seg]
        if s.empty:
            continue
        pts = sorted(s.point_no.unique())
        eligible = [d for d in visits if int(d[5:7]) >= 8]          # 8월 1일 이후 방문
        vm = {d: float(s.thaw_cm[s.date == d].mean()) for d in visits}
        seg_stats[seg] = {d: stats(s.thaw_cm[s.date == d]) for d in visits}
        if eligible:
            best = max(eligible, key=lambda d: (round(vm[d], 6), d))   # 같으면 늦은 방문
            label_def, eos = "direct_eos", "series_max"
        else:
            best = max(visits, key=lambda d: (round(vm[d], 6), d))
            label_def, eos = "direct_dated", "none"
        if int(best[5:7]) not in EOS_MONTHS:
            label_def, eos = "direct_dated", "none"
        sb = s[s.date == best]
        alt = round(vm[best], 4)
        flags = []
        if not (RANGE_CM[0] < alt <= RANGE_CM[1]):
            flags.append("range")
        lat_m, lon_m = float(sb.lat.mean()), float(sb.lon.mean())
        if not (BBOX["lat"][0] <= lat_m <= BBOX["lat"][1] and BBOX["lon"][0] <= lon_m <= BBOX["lon"][1]):
            flags.append("coord")
        if int(best[:4]) < 1990:
            flags.append("year_pre1990")
        vm_txt = ", ".join(f"{d} {vm[d]:.2f} cm" for d in visits)
        if seg == "L":
            note = (f"측선 38점 가운데 점 {pts[0]}-{pts[-1]}({len(pts)}점). 방문별 점 평균: {vm_txt}. "
                    f"늦은 방문의 평균이 이른 방문보다 작다(점별 변화 평균 {change['mean_cm']:.2f} cm, 전 측선 기준). "
                    "원인은 확인하지 못했다. 측선의 두 위치에서 NaCl 추적자 트렌치(폭 30 cm)를 "
                    f"{TRENCH_DATE} 에 팠다(논문 3절)")
            if best < TRENCH_DATE:
                note += ". 연 값은 트렌치 이전 방문의 값이다"
            note += ". 논문의 모형 모의는 2015년 최대 융해가 9월 20일에서 10월 3일 사이라고 적는다(5.3절)"
        else:
            note = (f"측선 38점 가운데 점 {pts[0]}-{pts[-1]}({len(pts)}점). 두 방문 모두 값이 "
                    f"{PROBE_LIMIT_CM:.0f} cm 로 같다. 관련 논문(5.1절)은 사면 상부의 활동층 두께가 1.2 m 를 "
                    "넘는다고 적는다. 값은 하한이다. 탐침 막대 길이는 확인하지 못했다")
        rows.append(dict(
            src_id=SRC_ID, site_id=f"{EVENT}_{seg}", site_name=sd["name"],
            lat=f"{lat_m:.8f}", lon=f"{lon_m:.8f}", year=int(best[:4]), month=int(best[5:7]),
            alt_cm=alt, method="probe", label_def=label_def, n_obs=len(sb),
            country="Greenland", macro="NAtlantic", citation=citation, license=license_txt,
            subunit="Greenland", date=best, eos_basis=eos, value_kind="annual_value",
            year_min="", year_max="",
            alt_sd_cm=round(float(sb.thaw_cm.std(ddof=1)), 4) if len(sb) > 1 else 0.0,
            coord_prec_deg="0.00000001", disturbed=0, disturb_type="",
            right_censored=sd["censored"], orig_source="", dup_of="",
            qc_flag=";".join(flags) if flags else "ok", notes=note))

    assert all(set(r) == set(columns) for r in rows), "열 구성이 형식 정의와 다르다"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POINTS, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)

    # ---------- 셀 색인과 v3 대조(읽기 전용) ----------
    pts_df = pd.DataFrame(rows)
    cells = {r["site_id"]: cell_index(float(r["lat"]), float(r["lon"])) for r in rows}
    point_cells = sorted({cell_index(a, b) for a, b in zip(first.lat, first.lon)})
    v3_check = None
    if V3.exists():
        b = pd.read_csv(V3, usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm"])
        b = b[b.source_id == "F4_direct"]
        v3_check = {}
        for r in rows:
            la, lo = float(r["lat"]), float(r["lon"])
            near = b[b.lat.between(la - 2, la + 2)]
            dist = haversine_km(la, lo, near.lat.values, near.lon.values)
            i = int(np.argmin(dist))
            nb = near.iloc[i]
            cheb = np.maximum(np.abs(b.lat.values - la), np.abs(b.lon.values - lo))
            v3_check[r["site_id"]] = dict(
                nearest_loc_id=int(nb.loc_id), nearest_region=str(nb.region),
                nearest_lat=float(nb.lat), nearest_lon=float(nb.lon), nearest_alt_cm=float(nb.alt_cm),
                dist_km=round(float(dist[i]), 3),
                chebyshev_deg=round(float(max(abs(nb.lat - la), abs(nb.lon - lo))), 5),
                n_v3_f4_direct_within_cheb_0p01deg=int((cheb <= 0.01).sum()))

    # ---------- 메타 ----------
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = ""
    files = [dict(name=p.name, path=str(p.relative_to(ROOT)), size_bytes=p.stat().st_size, sha256=sha256(p),
                  kind=k)
             for p, k in [(RAW_FILE, "원자료(내려받은 파일)"), (RAW_SUBSET, "파서가 만든 이벤트 추출본"),
                          (RAW_VISIT, "파서가 만든 방문 단위 점 자료"),
                          (PAPER_PDF, "관련 논문 전문(CC BY, DiVA 공개본, 참고용)")] if p.exists()]
    whole = {d: stats(v.thaw_cm[v.date == d]) for d in visits}
    alt_main = pts_df[pts_df.right_censored == 0].alt_cm
    meta = dict(
        src_id=SRC_ID,
        name="Zastruzny 등 2024, 디스코섬 사면 동결면 깊이 2015(PANGAEA 967139)",
        url=URL, url_file=URL_FILE, doi=DOI, accessed=ACCESSED,
        files=files, license=license_txt, license_statement=license_raw, citation=citation,
        related_paper="Zastruzny 등 2024, Water Resources Research 60(11), e2023WR036147, "
                      "https://doi.org/10.1029/2023WR036147",
        related_paper_oa_copy=PAPER_URL_OA,
        paper_evidence=[
            dict(where="3절, 4쪽", text="2015-08-04 와 08-21 에 80 m 측선을 1 m 간격으로 금속 막대로 탐침했고, "
                 "막대가 단단한 층에 닿은 깊이를 동결면으로 해석했다"),
            dict(where="3절, 4쪽", text="사면 상부의 피에조미터 #24 는 동결토를 찾지 못해 148 cm 까지 넣었다"),
            dict(where="3절, 5쪽", text=f"추적자(NaCl) 트렌치는 폭 30 cm, 영구동토 상면까지 팠고 {TRENCH_DATE} 에 "
                 "사면 위치 15 m 와 44.4 m 두 곳에 두었다"),
            dict(where="4.2.3절, 7쪽", text="Pedersen 등 2022 는 같은 사면 50-80 m 구간에서 2014, 2018 년 활동층 두께를 "
                 "측정했고 그 자료는 Pedersen 등 2022 의 보충 자료에 있다"),
            dict(where="5.1절, 8쪽", text="사면 상부는 활동층 두께가 1.2 m 를 넘고, 하부는 0.20-0.60 m 이다"),
            dict(where="5.3절, 11쪽", text="모형 모의에서 2015년 최대 활동층 두께는 9월 20일에서 10월 3일 사이에 "
                 "나타나고 상부 1.0 m, 하부 0.8 m 이다(모형값이므로 라벨로 쓰지 않는다)"),
        ],
        parser=str(Path(__file__).resolve().relative_to(ROOT)),
        parser_sha256=sha256(Path(__file__).resolve()),
        parser_git_commit=commit,
        parser_git_note="파서는 이 커밋의 작업 트리에서 새로 만든 파일이고 아직 커밋하지 않았다",
        schema_version=schema["version"],
        n_rows_raw=n_raw, n_rows_raw_event=n_event, n_rows_points=len(rows),
        n_by_label_def=pts_df.label_def.value_counts().to_dict(),
        n_by_method=pts_df.method.value_counts().to_dict(),
        n_by_macro=pts_df.macro.value_counts().to_dict(),
        n_right_censored_rows=int(pts_df.right_censored.sum()),
        n_excluded_by_reason=dict(
            other_event_without_thaw_depth=n_raw - n_event,
            not_common_to_both_visits=n_not_common,
            removed_from_points=0),
        checks=dict(
            thaw_depth_unit="cm(열 이름 'Thaw depth [cm]')",
            thaw_values_outside_event=n_thaw_other,
            missing_thaw_values=n_missing, nonnumeric_thaw_values=n_nonnum,
            greater_than_notation=n_gt, duplicate_records=n_dup,
            coord_outside_disko_bbox=n_coord_bad, thaw_outside_range=n_range_bad,
            visits=visits, n_points_per_visit={d: len(sets[d]) for d in visits},
            same_point_set_in_all_visits=bool(same_points), n_common_points=len(common),
            censor_suspect_points=sorted(int(p) for p in cens_pts),
            censor_suspect_same_in_all_visits=bool(cens_same),
            transect_length_m=round(float(cum_m[-1]), 2),
            mean_point_spacing_m=round(float(step_m.mean()), 3),
            tracer_trench_nearest_points=trench,
            recorded_time="두 방문 모두 14:00:00 으로 적혀 있다. 실제 측정 시각인지 확인하지 못했다"),
        visit_stats=dict(whole_transect=whole, by_segment=seg_stats,
                         change_last_minus_first=change,
                         whole_transect_series_max_cm=max(s["mean"] for s in whole.values())),
        alt_distribution_main=dict(note="right_censored = 0 인 행", n=int(alt_main.size),
                                   min=float(alt_main.min()), median=float(alt_main.median()),
                                   max=float(alt_main.max())),
        year_range=[int(pts_df.year.min()), int(pts_df.year.max())],
        cell_index=dict(rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009",
                        by_site={k: list(c) for k, c in cells.items()},
                        cells_of_38_points=[list(c) for c in point_cells]),
        v3_f4_direct_check=v3_check,
        coordinate_conversion="변환 없음. 원자료가 WGS84 십진 도(소수 8자리, RTK DGPS)이고 서경은 음수다. "
                              "행 좌표는 구간 안 점 좌표의 평균이다",
        label_rule="연 값은 8월 1일 이후 방문의 점 평균 가운데 최댓값(평균이 같으면 늦은 방문). "
                   f"값이 {PROBE_LIMIT_CM:.0f} cm 이상인 점은 구간 U 로 나누고 right_censored = 1 로 둔다",
        unverified_items=[
            f"값 {PROBE_LIMIT_CM:.0f} cm 가 탐침 막대 길이인지 기록 상한인지. 논문(5.1절)이 상부 활동층을 1.2 m 초과로 "
            "적으므로 하한이라는 점은 확인했다. 막대 길이는 논문과 자료 설명에 없다. 상부 피에조미터 #24 가 "
            "148 cm 까지 동결토를 만나지 않았으므로(3절) 그 위치의 영구동토 존재 여부도 확인하지 못했다",
            "늦은 방문(08-21)의 점 평균이 이른 방문(08-04)보다 작은 이유. 논문 본문에서 설명을 찾지 못했다. "
            "측정자, 측정 위치의 미세한 차이, 자갈에 의한 탐침 저항 가운데 무엇인지 확인하지 못했다",
            "2015-08-04 에 58 cm 가 5점에서 반복되는 이유",
            "두 방문 뒤(8월 말에서 10월 초)의 융해 깊이. 자료에 없다. 논문의 모형 모의는 최대 융해가 9월 20일에서 "
            "10월 3일 사이라고 적으므로 8월 값은 연 최대보다 작을 수 있다. 관측으로는 확인하지 못했다",
            "기록 시각 14:00:00 이 실제 측정 시각인지 여부",
            f"점 간격. 논문은 1 m 간격 탐침이라고 적으나 자료에는 약 {step_m.mean():.2f} m 간격 38점이 있다. "
            "공개 과정에서 점을 추렸는지 확인하지 못했다",
            "추적자 트렌치의 영향. 논문의 사면 위치 축 원점을 상부로 추정했고(그림 6) 확인하지 못했다. "
            "트렌치 부근 점의 08-21 값에 준 영향은 확인하지 못했다. 교란이 두 위치에 국한되고 구간 L 의 연 값은 "
            "트렌치 이전 방문(08-04)이므로 disturbed 는 0 으로 두었다",
        ],
        leads=[
            "Pedersen 등 2022(Journal of Ecology 110(8), 1896-1912, https://doi.org/10.1111/1365-2745.13925) 보충 자료에 "
            "같은 사면 50-80 m 구간의 2014, 2018 년 활동층 두께가 있다(논문 4.2.3절). 이 작업에서는 내려받지 않았다",
        ])
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---------- 품질 요약 출력 ----------
    print(f"[{SRC_ID}] 원자료 {n_raw}행, 이벤트 {EVENT} {n_event}행")
    print(f"방문 {visits}, 방문별 점 수 {[len(sets[d]) for d in visits]}, 점 구성 동일 {same_points}")
    print(f"결측 {n_missing}, 비수치 {n_nonnum}, '>' 표기 {n_gt}, 중복 {n_dup}, "
          f"좌표 범위 밖 {n_coord_bad}, 값 범위 밖 {n_range_bad}")
    print(f"검열 의심 점 {sorted(cens_pts)}, 방문 사이 동일 {cens_same}")
    print("전 측선 방문별:", json.dumps(whole, ensure_ascii=False))
    print("구간별:", json.dumps(seg_stats, ensure_ascii=False))
    print("방문 사이 변화:", change)
    print(pts_df[["site_id", "lat", "lon", "year", "month", "date", "alt_cm", "alt_sd_cm", "n_obs",
                  "label_def", "eos_basis", "right_censored", "qc_flag"]].to_string(index=False))
    print("셀 색인:", cells, "38점의 셀:", point_cells)
    print("v3 대조:", json.dumps(v3_check, ensure_ascii=False))
    print("산출:", POINTS, META, RAW_SUBSET, RAW_VISIT, sep="\n  ")


if __name__ == "__main__":
    main()
