"""XE 2단계 취득(사전 등록 문서 2.5절, docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 개정 1, T0 = 2678100).

대상 군과 주 경로(등록 문서 2.5절 '2단계 취득의 마감과 대체 경로' 표):
  V(WorldCover)  ESA WorldCover 2021 v200 10 m 타일(S3 HTTPS, 무서명)          마감 T0 + 72 h
  V(CAVM)        Raster CAVM(Mendeley Data c4xj5rv6kv)                          마감 T0 + 72 h
  O(NCSCD)       NCSCDv2 0.012° 래스터(Bolin Centre)                            마감 T0 + 72 h
  O(SoilGrids)   ISRIC SoilGrids 250 m VRT 창 읽기(부족 창만, 새 폴더에 저장)   마감 T0 + 72 h
  V(MOD13Q1)·S   AppEEARS 점 요청. 이 스크립트는 0일 소형 시험 요청 제출과
                 전체 요청 JSON 준비만 한다(전체 요청은 제출하지 않는다)          마감 T0 + 96 h

규칙:
- 라벨 표(fidelity_base_v3.csv)에서는 좌표 열(loc_id, lat, lon, region)만 읽는다. 라벨 값 열은 읽지 않는다
  (등록 문서 2.5절 시험 (e)).
- 기존 자료 파일은 고치지 않는다. 산출은 data/raw/ 아래 새 폴더에만 쓴다.
- 표준 출력에는 개수, 크기, 상태만 적는다.

하위 명령:
  points          대상 좌표 집합과 MODIS 고유 화소 표를 만든다(data/raw/appeears_xe/points/)
  worldcover      WorldCover 타일 내려받기
  cavm            CAVM 래스터 내려받기
  ncscd           NCSCDv2 0.012° 래스터 내려받기
  soilgrids       SoilGrids 250 m 기존 창 포함 여부 점검과 부족 창 내려받기
  appeears-test   AppEEARS 소형 시험 요청 제출(10점, 1개월, 제품별 1건)
  appeears-poll   시험 요청 상태 기록과 완료 시 결과 내려받기(대기 시간 측정)
  appeears-prep   전체 점 요청 JSON 준비(제출하지 않음)
  appeears-submit 준비된 전체 점 요청을 (set:product) 순서대로 제출(task_id 기록, 재실행 시 이미 낸 것은 건너뜀)
  appeears-status 제출된 과제 상태를 한 번 조회해 기록(--fetch 면 done 과제의 결과 묶음 내려받기)
  source-md       각 폴더의 SOURCE.md(약관, 체크섬)를 다시 쓴다(--only 폴더 이름)

  worldcover --scope all_labels: v3 ∪ v4 의 모든 지역 라벨 위치까지 타일을 넓힌다(2차 범위, 2026-10-04 저녁)
"""
import argparse
import hashlib
import json
import netrc
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/willy010313/Polar_Bigdata")
RAW = ROOT / "data/raw"
LABELS = ROOT / "data/processed/fidelity_base_v3.csv"
LENA_CELLS = ROOT / "data/processed/map_lena/lena_grid_cells_v1.csv"
LENA_MASK = ROOT / "data/processed/map_lena/lena_grid_mask_v1.csv"
PLAN = "docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절(개정 1, T0 = git 2678100, 2026-10-04 14:22:39 +0900)"
KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 4, 14, 22, 39, tzinfo=KST)

# XE 대상 정의(within_grid_inputs.md 부록 A 와 같다)
REG_TARGET = {"ABoVE_AK": "alaska", "United States (Alaska)": "alaska", "Lena_RU": "lena",
              "ABoVE_CA": "canada", "Canada": "canada"}

D_WC = RAW / "worldcover_v200"
D_CAVM = RAW / "cavm_raster"
D_NCSCD = RAW / "ncscd_v2"
D_SG = RAW / "soilgrids_xe250"
D_SG_OLD = RAW / "soilgrids_multi"
D_AE = RAW / "appeears_xe"

UA = {"User-Agent": "Polar_Bigdata-XE-stage2/1.0 (research download)"}


def now_kst():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %z")


def log(msg):
    print(f"[{datetime.now(KST).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hashes_file(p, chunk=1 << 22):
    """한 번 읽어 (sha256, md5) 를 돌려준다."""
    h, m = hashlib.sha256(), hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b); m.update(b)
    return h.hexdigest(), m.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    tmp.replace(p)


def http_download(url, dst, expect_size=None, expect_md5=None, expect_sha256=None,
                  retries=5, timeout=120, session=None):
    """스트리밍 내려받기. .part 에 쓰고 크기·해시를 확인한 뒤 이름을 바꾼다.
    반환: dict(path, bytes, sha256, md5, status, sec)."""
    import requests
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and (expect_size is None or dst.stat().st_size == expect_size):
        sh, md = hashes_file(dst)
        if (expect_sha256 is None or sh == expect_sha256) and (expect_md5 is None or md == expect_md5):
            return dict(path=str(dst), bytes=dst.stat().st_size, sha256=sh, md5=md, status="cached", sec=0.0)
    s = session or requests.Session()
    last = None
    for k in range(retries):
        t0 = time.time()
        part = dst.with_suffix(dst.suffix + ".part")
        try:
            with s.get(url, stream=True, timeout=timeout, headers=UA, allow_redirects=True) as r:
                r.raise_for_status()
                h, m, n = hashlib.sha256(), hashlib.md5(), 0
                with open(part, "wb") as f:
                    for b in r.iter_content(1 << 20):
                        if b:
                            f.write(b); h.update(b); m.update(b); n += len(b)
                clen = r.headers.get("Content-Length")
            if clen is not None and int(clen) != n:
                raise IOError(f"크기 불일치 {n} != {clen}")
            if expect_size is not None and n != expect_size:
                raise IOError(f"크기 불일치 {n} != 기대 {expect_size}")
            if expect_md5 is not None and m.hexdigest() != expect_md5:
                raise IOError("md5 불일치")
            if expect_sha256 is not None and h.hexdigest() != expect_sha256:
                raise IOError("sha256 불일치")
            part.replace(dst)
            return dict(path=str(dst), bytes=n, sha256=h.hexdigest(), md5=m.hexdigest(), status="ok",
                        sec=round(time.time() - t0, 1))
        except Exception as e:  # 재시도
            last = str(e)[:200]
            if part.exists():
                part.unlink()
            time.sleep(3 * (k + 1))
    return dict(path=str(dst), bytes=0, sha256=None, md5=None, status=f"error: {last}", sec=None)


# ---------------------------------------------------------------- 좌표 집합

def label_points():
    """XE 대상 라벨 위치. 좌표 열만 읽는다."""
    d = pd.read_csv(LABELS, usecols=["loc_id", "lat", "lon", "region"])
    d = d[d.region.isin(REG_TARGET)].copy()
    d["target"] = d.region.map(REG_TARGET)
    return d[["loc_id", "lat", "lon", "target"]].reset_index(drop=True)


LABELS_V4 = ROOT / "data/processed/fidelity_base_v4.csv"


def all_label_points():
    """v3 ∪ v4 의 모든 지역 라벨 위치(좌표 열만 읽는다). 열 tag 는 v3 의 XE 대상 행이면 `labels_<target>`
    (1차 범위의 17,385점과 같다), 그 밖이면 `v3:<region>`·`v4:<region>` 이다(한 위치가 여러 표에 있으면 ';' 로 잇는다)."""
    tags = {}
    rows = {}
    for ver, path in (("v3", LABELS), ("v4", LABELS_V4)):
        d = pd.read_csv(path, usecols=["loc_id", "lat", "lon", "region"])
        for lid, la, lo, rg in zip(d.loc_id, d.lat, d.lon, d.region):
            if lid in rows:
                assert rows[lid] == (la, lo), f"좌표 불일치 {lid}"
            rows[lid] = (la, lo)
            xe = ver == "v3" and rg in REG_TARGET
            tags.setdefault(lid, set()).add(f"labels_{REG_TARGET[rg]}" if xe else f"{ver}:{rg}")
    out = pd.DataFrame([(k, v[0], v[1], ";".join(sorted(tags[k]))) for k, v in rows.items()],
                       columns=["loc_id", "lat", "lon", "tag"])
    return out.reset_index(drop=True)


def lena_points(land_only=False):
    g = pd.read_csv(LENA_CELLS, usecols=["cell_id", "lat", "lon"])
    if land_only:
        m = pd.read_csv(LENA_MASK, usecols=["cell_id", "land"])
        g = g.merge(m, on="cell_id", how="left")
        g = g[g.land == 1].drop(columns="land")
    return g.reset_index(drop=True)


# MODIS 사인 곡선 격자(MOD13Q1 250 m, MOD10A1·A2 500 m)
SINU = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
SINU_X0, SINU_Y0 = -20015109.354, 10007554.677
TILE_M = 1111950.5197665
PX = {250: TILE_M / 4800.0, 500: TILE_M / 2400.0}


def modis_pixels(lat, lon, res):
    """위경도 → (전역 행, 열) 정수와 화소 중심 위경도. 왕복 시 같은 화소인지 확인한다."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", SINU, always_xy=True)
    inv = Transformer.from_crs(SINU, "EPSG:4326", always_xy=True)
    x, y = fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
    px = PX[res]
    col = np.floor((np.asarray(x) - SINU_X0) / px).astype(np.int64)
    row = np.floor((SINU_Y0 - np.asarray(y)) / px).astype(np.int64)
    xc = SINU_X0 + (col + 0.5) * px
    yc = SINU_Y0 - (row + 0.5) * px
    lonc, latc = inv.transform(xc, yc)
    lonc, latc = np.round(np.asarray(lonc), 6), np.round(np.asarray(latc), 6)
    x2, y2 = fwd.transform(lonc, latc)
    col2 = np.floor((np.asarray(x2) - SINU_X0) / px).astype(np.int64)
    row2 = np.floor((SINU_Y0 - np.asarray(y2)) / px).astype(np.int64)
    assert (col2 == col).all() and (row2 == row).all(), "화소 중심 왕복 불일치"
    return row, col, latc, lonc


def cmd_points(a):
    out = D_AE / "points"
    out.mkdir(parents=True, exist_ok=True)
    lab = label_points()
    lena = lena_points(land_only=True)
    lena_all = lena_points(land_only=False)
    meta = dict(created=now_kst(), plan=PLAN, script="scripts/0_download/xe_stage2_acquire.py points",
                labels=str(LABELS.relative_to(ROOT)), labels_cols_read=["loc_id", "lat", "lon", "region"],
                lena_cells=str(LENA_CELLS.relative_to(ROOT)), lena_mask=str(LENA_MASK.relative_to(ROOT)),
                lena_rule="land == 1(지도 예측 대상 육지 셀)", sinusoidal=SINU,
                pixel_m={str(k): v for k, v in PX.items()}, sets={})
    for name, df, key in (("labels", lab, "loc_id"), ("lena_land", lena, "cell_id")):
        tab = df.copy()
        for res in (250, 500):
            r, c, la, lo = modis_pixels(tab.lat.values, tab.lon.values, res)
            tab[f"pix{res}_id"] = [f"r{a_}c{b_}" for a_, b_ in zip(r, c)]
            tab[f"pix{res}_lat"], tab[f"pix{res}_lon"] = la, lo
        tab.to_csv(out / f"xe_{name}_pixels.csv", index=False)
        st = dict(n_points=int(len(tab)),
                  n_pix250=int(tab.pix250_id.nunique()), n_pix500=int(tab.pix500_id.nunique()))
        if "target" in tab:
            st["by_target"] = {t: dict(n_points=int(len(g)), n_pix250=int(g.pix250_id.nunique()),
                                       n_pix500=int(g.pix500_id.nunique()))
                               for t, g in tab.groupby("target")}
        meta["sets"][name] = st
        log(f"[points] {name}: 점 {st['n_points']}, 250 m 고유 화소 {st['n_pix250']}, 500 m 고유 화소 {st['n_pix500']}")
    meta["sets"]["lena_all_cells"] = dict(n_points=int(len(lena_all)))
    write_json(out / "points_meta.json", meta)


# ---------------------------------------------------------------- WorldCover

WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WC_KEY = "v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"


def wc_name(lat0, lon0):
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"


def wc_tiles_for(lat, lon, dlat, dlon):
    """점 주변 상자(±dlat, ±dlon)가 닿는 3° 타일 이름 집합."""
    names = set()
    for sl in (-1, 1):
        for so in (-1, 1):
            la = np.floor((np.asarray(lat) + sl * dlat) / 3.0).astype(int) * 3
            lo = np.floor((np.asarray(lon) + so * dlon) / 3.0).astype(int) * 3
            names.update(wc_name(int(x), int(y)) for x, y in zip(la, lo))
    return names


def cmd_worldcover(a):
    import requests
    D_WC.mkdir(parents=True, exist_ok=True)
    (D_WC / "tiles").mkdir(exist_ok=True)
    s = requests.Session()
    grid_p = D_WC / "esa_worldcover_grid.geojson"
    if not grid_p.exists():
        http_download(f"{WC_BASE}/esa_worldcover_grid.geojson", grid_p, session=s)
    grid = {f["properties"]["ll_tile"] for f in json.loads(grid_p.read_text())["features"]}
    # 3 × 3 창(30 m)과 위치 오차를 덮는 여유: 위도 0.001°(약 110 m), 경도 0.001°/cosφ
    dlat = 0.001
    need = {}
    scope = getattr(a, "scope", "xe")
    if scope == "xe":
        lab = label_points()
        groups = [(f"labels_{t}", g) for t, g in lab.groupby("target")]
        n_lab = len(lab)
    else:  # all_labels: v3 ∪ v4 의 모든 지역(XE 밖 지역은 v3:·v4: 꼬리표)
        lab = all_label_points()
        groups = [(t, g) for t, g in lab.groupby("tag")]
        n_lab = len(lab)
    for t, g in groups:
        dlon = dlat / np.cos(np.radians(g.lat.values))
        for nm in wc_tiles_for(g.lat.values, g.lon.values, dlat, dlon.max()):
            for tg in t.split(";"):
                need.setdefault(nm, set()).add(tg)
    lena = lena_points(land_only=False)
    la0, la1 = lena.lat.min() - 0.01, lena.lat.max() + 0.01
    lo0, lo1 = lena.lon.min() - 0.03, lena.lon.max() + 0.03
    for la in range(int(np.floor(la0 / 3) * 3), int(np.floor(la1 / 3) * 3) + 1, 3):
        for lo in range(int(np.floor(lo0 / 3) * 3), int(np.floor(lo1 / 3) * 3) + 1, 3):
            need.setdefault(wc_name(la, lo), set()).add("lena_box")
    names = sorted(need)
    absent = [n for n in names if n not in grid]
    todo = [n for n in names if n in grid]
    have = {p.name.split("_")[-2] for p in (D_WC / "tiles").glob("ESA_WorldCover_10m_2021_v200_*_Map.tif")}
    missing = [n for n in todo if n not in have]
    log(f"[worldcover] 범위 {scope}(라벨 {n_lab}점): 필요 타일 {len(names)}, 격자 목록에 없음(바다·범위 밖) {len(absent)}, "
        f"내려받기 대상 {len(todo)}, 그 가운데 디스크에 없음 {len(missing)}")
    if getattr(a, "dry_run", False):
        for n in missing:
            log(f"[worldcover]   없음 {n} used_by={';'.join(sorted(need[n]))}")
        for n in absent:
            log(f"[worldcover]   격자에 없음 {n} used_by={';'.join(sorted(need[n]))}")
        return
    # 크기·ETag 확인(HEAD)
    rows = []

    def one(nm):
        url = f"{WC_BASE}/{WC_KEY.format(t=nm)}"
        h, err = None, None
        for k in range(5):  # HEAD 도 재시도한다(원격 연결 끊김으로 1차 전체 실행이 중단된 적이 있다)
            try:
                h = s.head(url, timeout=60, headers=UA)
                break
            except Exception as e:
                err = str(e)[:200]
                time.sleep(3 * (k + 1))
        if h is None:
            return dict(tile=nm, url=url, status=f"head error: {err}", bytes=0, sha256=None,
                        used_by=";".join(sorted(need[nm])))
        if h.status_code != 200:
            return dict(tile=nm, url=url, status=f"head {h.status_code}", bytes=0, sha256=None,
                        used_by=";".join(sorted(need[nm])))
        size = int(h.headers.get("Content-Length", 0))
        etag = h.headers.get("ETag", "").strip('"')
        md5 = etag if (etag and "-" not in etag) else None
        r = http_download(url, D_WC / "tiles" / Path(url).name, expect_size=size, expect_md5=md5)
        r.update(tile=nm, url=url, etag=etag, used_by=";".join(sorted(need[nm])))
        return r

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, n): n for n in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:  # 한 타일의 예외가 전체 실행을 멈추지 않게 한다
                r = dict(tile=futs[fu], url=f"{WC_BASE}/{WC_KEY.format(t=futs[fu])}", status=f"error: {str(e)[:200]}",
                         bytes=0, sha256=None, used_by=";".join(sorted(need[futs[fu]])))
            rows.append(r)
            if i % 10 == 0 or not str(r["status"]).startswith(("ok", "cached")):
                log(f"[worldcover] {i}/{len(todo)} {r['tile']} {r['status']}")
    for n in absent:
        rows.append(dict(tile=n, url=None, status="absent_in_grid(바다 또는 제품 범위 밖)", bytes=0, sha256=None,
                         used_by=";".join(sorted(need[n]))))
    man = pd.DataFrame(rows).sort_values("tile")
    man["path"] = man["path"].map(lambda p: str(Path(p).relative_to(ROOT)) if isinstance(p, str) else p) \
        if "path" in man else None
    man.to_csv(D_WC / "manifest_tiles.csv", index=False)
    # 범례 근거(제품 사용 설명서)
    pum = http_download(f"{WC_BASE}/v200/2021/docs/WorldCover_PUM_V2.0.pdf", D_WC / "docs" / "WorldCover_PUM_V2.0.pdf",
                        session=s)
    ok = man.status.astype(str).str.match(r"^(ok|cached)").sum()
    n_new = int(man.status.astype(str).eq("ok").sum())
    meta_p = D_WC / "download_meta.json"
    prev = json.loads(meta_p.read_text()) if meta_p.exists() else None
    meta = dict(created=now_kst(), plan=PLAN, product="ESA WorldCover 10 m 2021 v200", base=WC_BASE,
                scope=scope, n_label_points=int(n_lab),
                scope_note=("XE 대상(알래스카·레나·캐나다) 라벨 + 레나 지도 상자" if scope == "xe" else
                            "fidelity_base_v3 ∪ v4 의 모든 지역 라벨 + 레나 지도 상자. XE 밖 지역은 used_by 의 v3:·v4: 꼬리표"),
                n_needed=len(names), n_absent_in_grid=len(absent), n_ok=int(ok), n_new_downloaded=n_new,
                n_cached=int(man.status.astype(str).eq("cached").sum()),
                bytes_total=int(man.bytes.fillna(0).sum()), margin_deg=dict(lat=dlat, lon="0.001/cos(lat)"),
                lena_box=[float(lo0), float(lo1), float(la0), float(la1)], pum_sha256=pum.get("sha256"),
                elapsed_note="T0 + 72 h 마감(2026-10-07 14:22:39 +0900)")
    if prev is not None:
        hist = prev.pop("history", [])
        meta["history"] = hist + [prev]
    write_json(meta_p, meta)
    log(f"[worldcover] 완료 {ok}/{len(todo)}(새로 받음 {n_new}), 합계 {meta['bytes_total']/1e9:.2f} GB")


# ---------------------------------------------------------------- CAVM

MENDELEY_API = "https://data.mendeley.com/public-api/datasets/c4xj5rv6kv"


def cmd_cavm(a):
    import requests
    D_CAVM.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    info = s.get(MENDELEY_API, timeout=60, headers=UA).json()
    rows = []
    for ver in (2, 1):
        files = s.get(f"{MENDELEY_API}/files", params=dict(folder_id="root", version=ver), timeout=60,
                      headers=UA).json()
        for f in files:
            cd = f["content_details"]
            dst = D_CAVM / f"v{ver}" / f["filename"].replace(" ", "_")
            r = http_download(cd["download_url"], dst, expect_size=f["size"], expect_sha256=cd["sha256_hash"],
                              session=s)
            r.update(version=ver, filename=f["filename"], api_sha256=cd["sha256_hash"])
            rows.append(r)
            if r["status"] in ("ok", "cached"):
                with zipfile.ZipFile(dst) as z:
                    z.extractall(dst.parent / "unzipped")
                    r["members"] = [(i.filename, i.file_size) for i in z.infolist()]
            log(f"[cavm] v{ver} {f['filename']} {r['status']} {r['bytes']/1e6:.1f} MB")
    meta = dict(created=now_kst(), plan=PLAN, dataset=info.get("name"), doi_latest=info.get("doi", {}).get("id"),
                version_latest=info.get("version"), licence=info.get("data_licence"), files=rows,
                primary="v2(최신 판, 2022-01-17 공개). v1(2019-05-16)은 within_grid_inputs.md 부록 B 가 인용한 판으로 대조용")
    for r in rows:
        r["path"] = str(Path(r["path"]).relative_to(ROOT))
    write_json(D_CAVM / "download_meta.json", meta)


# ---------------------------------------------------------------- NCSCD

NCSCD_URL = "https://bolin.su.se/data/ncscd/data/v3/raster/NCSCDv2_Circumpolar_raster_0012deg.zip"


def cmd_ncscd(a):
    D_NCSCD.mkdir(parents=True, exist_ok=True)
    dst = D_NCSCD / Path(NCSCD_URL).name
    r = http_download(NCSCD_URL, dst, timeout=300)
    log(f"[ncscd] {r['status']} {r['bytes']/1e6:.1f} MB")
    members = []
    if r["status"] in ("ok", "cached"):
        with zipfile.ZipFile(dst) as z:
            z.extractall(D_NCSCD / "unzipped")
            members = [(i.filename, i.file_size) for i in z.infolist()]
    r["path"] = str(Path(r["path"]).relative_to(ROOT))
    write_json(D_NCSCD / "download_meta.json",
               dict(created=now_kst(), plan=PLAN, url=NCSCD_URL, file=r, members=members,
                    unit_note="격자 자료의 SOCC 단위는 hg C m-2(Bolin technical.php). 벡터 자료는 kg C m-2"))


# ---------------------------------------------------------------- SoilGrids 250 m 부족 창

SG_BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
SG_LAYERS = ["soc/soc_0-5cm_mean", "soc/soc_5-15cm_mean", "soc/soc_15-30cm_mean", "bdod/bdod_5-15cm_mean",
             "clay/clay_5-15cm_mean", "sand/sand_5-15cm_mean", "silt/silt_5-15cm_mean"]
SG_BIN = 2.0
SG_MARGIN_M = 1.2e4  # soilgrids_multiregion.py 와 같은 여유


def _igh_bounds(lo0, lo1, la0, la1):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
    lons = np.r_[np.linspace(lo0, lo1, 9), np.linspace(lo0, lo1, 9), np.full(9, lo0), np.full(9, lo1)]
    lats = np.r_[np.full(9, la0), np.full(9, la1), np.linspace(la0, la1, 9), np.linspace(la0, la1, 9)]
    xs, ys = tr.transform(lons, lats)
    return (float(np.min(xs) - SG_MARGIN_M), float(np.max(xs) + SG_MARGIN_M),
            float(np.min(ys) - SG_MARGIN_M), float(np.max(ys) + SG_MARGIN_M))


def sg_coverage(pts, dirs):
    """점별로 창 디렉터리 목록에서 7층이 모두 유효(> 0, nodata 아님)인 창이 있는지 본다.
    반환: in_window(창 경계 안), valid_all(7층 모두 유효), win(채운 창 이름)."""
    import rasterio
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
    x, y = tr.transform(pts.lon.values, pts.lat.values)
    x, y = np.asarray(x), np.asarray(y)
    n = len(pts)
    in_win = np.zeros(n, bool)
    valid = np.zeros(n, bool)
    win = np.array([""] * n, dtype=object)
    for d in dirs:
        lay = [d / f"{L.split('/')[-1]}.tif" for L in SG_LAYERS]
        if not all(p.exists() for p in lay):
            continue
        ok_all = None
        inside = None
        for p in lay:
            with rasterio.open(p) as src:
                b = src.read(1)
                nod = src.nodata
                fc, fr = (~src.transform) * (x, y)
                r, c = np.floor(fr).astype(int), np.floor(fc).astype(int)
                ins = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
                v = b[np.clip(r, 0, src.height - 1), np.clip(c, 0, src.width - 1)].astype(float)
                ok = ins & (v > 0) & ((v != nod) if nod is not None else True)
            ok_all = ok if ok_all is None else (ok_all & ok)
            inside = ins if inside is None else (inside & ins)
        in_win |= inside
        newly = ok_all & ~valid
        win[newly] = d.name
        valid |= ok_all
    return in_win, valid, win


def _aligned_window(src, minx, maxx, miny, maxy):
    """원격 VRT 격자에 정수 화소로 맞춘 창(바깥쪽으로 넓힘). 소수 창(from_bounds)은 읽을 때 반올림·재표집이
    일어나 저장 변환과 자료가 최대 0.5 화소 어긋난다(기존 soilgrids_multi 창에서 확인, impl_notes 참고)."""
    from rasterio.windows import Window
    T = src.transform
    c0 = int(np.floor((minx - T.c) / T.a))
    c1 = int(np.ceil((maxx - T.c) / T.a))
    r0 = int(np.floor((maxy - T.f) / T.e))
    r1 = int(np.ceil((miny - T.f) / T.e))
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, src.width), min(r1, src.height)
    return Window(c0, r0, c1 - c0, r1 - r0)


def _sg_fetch(lyr, bdir, bounds):
    import rasterio
    name = lyr.split("/")[-1]
    dst = bdir / f"{name}.tif"
    if dst.exists() and dst.stat().st_size > 1000:
        return name, dict(status="cached", bytes=dst.stat().st_size, sha256=sha256_file(dst))
    t0 = time.time()
    minx, maxx, miny, maxy = bounds
    err = None
    for k in range(4):
        try:
            with rasterio.open(f"{SG_BASE}/{lyr}.vrt") as src:
                w = _aligned_window(src, minx, maxx, miny, maxy)
                assert all(float(v).is_integer() for v in (w.col_off, w.row_off, w.width, w.height))
                arr = src.read(1, window=w)
                assert arr.shape == (int(w.height), int(w.width))
                wt = src.window_transform(w)
                prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
                            crs=IGH, transform=wt, nodata=src.nodata, compress="deflate")
                tmp = bdir / f"{name}.tif.part"
                with rasterio.open(tmp, "w", **prof) as d:
                    d.write(arr, 1)
                tmp.replace(dst)
                nv = int(((arr != src.nodata) & (arr > 0)).sum()) if src.nodata is not None else int((arr > 0).sum())
            return name, dict(status="ok", shape=list(arr.shape), n_valid=nv, bytes=dst.stat().st_size,
                              window=[int(w.col_off), int(w.row_off), int(w.width), int(w.height)],
                              sha256=sha256_file(dst), sec=round(time.time() - t0, 1))
        except Exception as e:
            err = str(e)[:200]
            time.sleep(10 * (k + 1))
    return name, dict(status=f"error: {err}")


def cmd_soilgrids(a):
    from rasterio.env import Env
    D_SG.mkdir(parents=True, exist_ok=True)
    lab = label_points()
    lena = lena_points(land_only=True)
    pts = pd.concat([lab.assign(set="labels")[["lat", "lon", "target", "set"]],
                     lena.assign(set="lena_land", target="lena_grid")[["lat", "lon", "target", "set"]]],
                    ignore_index=True)
    old_dirs = sorted(p for p in D_SG_OLD.iterdir() if p.is_dir())
    in_old, val_old, _ = sg_coverage(pts, old_dirs)
    pts["in_old"], pts["valid_old"] = in_old, val_old
    cov = {}
    for (s_, t), g in pts.groupby(["set", "target"]):
        cov[f"{s_}:{t}"] = dict(n=int(len(g)), in_old_window=int(g.in_old.sum()), valid_old=int(g.valid_old.sum()))
    log("[soilgrids] 기존 창 포함(창 경계 안 / 7층 유효): " +
        "; ".join(f"{k} {v['in_old_window']}/{v['valid_old']} of {v['n']}" for k, v in cov.items()))
    # 받을 칸: (1) 기존 창 경계 밖 점이 있는 2° 칸(부족 창, 등록 문서의 문자 그대로의 대상),
    # (2) 기존 창이 있는 칸의 정렬 재취득. 기존 창은 소수 창으로 읽어 저장 변환과 자료가 최대 0.5 화소
    # 어긋나므로(impl_notes/xe_stage2_acquire.md), 같은 제품을 정수 화소 창으로 다시 받아 새 폴더에 둔다.
    # 기존 창 경계 안이지만 값이 무효인 점은 정렬 재취득 뒤에 다시 센다.
    bl = (np.floor(pts.lon / SG_BIN) * SG_BIN).astype(int)
    bt = (np.floor(pts.lat / SG_BIN) * SG_BIN).astype(int)
    pts["bin"] = [f"lon{a_:+04d}_lat{b_:+03d}" for a_, b_ in zip(bl, bt)]
    old_names = {p.name for p in old_dirs}
    agg = pts.groupby("bin").agg(n=("lat", "size"), n_out_old=("in_old", lambda v: int((~v).sum()))).reset_index()
    agg["kind"] = np.where(agg.n_out_old > 0, "missing_window", "reacquire_aligned")
    agg["bl"] = [int(b.split("_")[0][3:]) for b in agg.bin]
    agg["bt"] = [int(b.split("_")[1][3:]) for b in agg.bin]
    bins = agg.sort_values(["kind", "n"], ascending=[True, False]).reset_index(drop=True)
    log(f"[soilgrids] 기존 창 밖 점 {int((~pts.in_old).sum())}, 2° 칸 {len(bins)}"
        f"(부족 창 {int((bins.kind == 'missing_window').sum())}, 정렬 재취득 {int((bins.kind == 'reacquire_aligned').sum())})")
    meta_p = D_SG / "windows_meta.json"
    wmeta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    env = dict(GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="5",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".vrt,.tif", VSI_CACHE="TRUE", GDAL_HTTP_MULTIPLEX="YES",
               CPL_VSIL_CURL_CACHE_SIZE="200000000", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               GDAL_NUM_THREADS="2")
    with Env(**env):
        for i, r in enumerate(bins.itertuples(), 1):
            lo0, la0 = float(r.bl), float(r.bt)
            bname = f"lon{r.bl:+04d}_lat{r.bt:+03d}"
            bdir = D_SG / bname
            bdir.mkdir(exist_ok=True)
            bounds = _igh_bounds(lo0, lo0 + SG_BIN, la0, la0 + SG_BIN)
            ent = wmeta.setdefault(bname, dict(wgs84_bbox=[lo0, lo0 + SG_BIN, la0, la0 + SG_BIN],
                                               igh_bounds=[round(v) for v in bounds], n_points=int(r.n),
                                               n_points_outside_old=int(r.n_out_old), kind=r.kind,
                                               same_bin_in_soilgrids_multi=bname in old_names, layers={}))
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(_sg_fetch, L, bdir, bounds) for L in SG_LAYERS]
                for fu in as_completed(futs):
                    nm, st = fu.result()
                    ent["layers"][nm] = st
            bad = [k for k, v in ent["layers"].items() if not str(v.get("status")).startswith(("ok", "cached"))]
            log(f"[soilgrids] [{i}/{len(bins)}] {bname} {r.kind} 점 {r.n} 실패 층 {len(bad)}")
            write_json(meta_p, wmeta)
    # 정렬 창(새 폴더)만으로 센 포함률
    new_dirs = sorted(p for p in D_SG.iterdir() if p.is_dir())
    in_new, val_new, _ = sg_coverage(pts, new_dirs)
    pts["in_new"], pts["valid_new"] = in_new, val_new
    for (s_, t), g in pts.groupby(["set", "target"]):
        cov[f"{s_}:{t}"].update(in_window_after=int(g.in_new.sum()), valid_after=int(g.valid_new.sum()))
    log("[soilgrids] 취득 뒤(창 경계 안 / 7층 유효): " +
        "; ".join(f"{k} {v['in_window_after']}/{v['valid_after']} of {v['n']}" for k, v in cov.items()))
    write_json(D_SG / "coverage_meta.json",
               dict(created=now_kst(), plan=PLAN, layers=SG_LAYERS, bin_deg=SG_BIN, margin_m=SG_MARGIN_M,
                    old_dirs=[str(p.relative_to(ROOT)) for p in old_dirs],
                    after_rule="취득 뒤 값은 이 폴더의 정수 화소 정렬 창만으로 센다",
                    bins=bins[["bin", "kind", "n", "n_out_old"]].to_dict("records"),
                    valid_rule="창 경계 안이고 7층 모두 값 > 0 이며 nodata 아님(within_grid_inputs.md 부록 A 의 '값 > 0' 규칙을 7층으로 확장)",
                    coverage=cov))


# ---------------------------------------------------------------- AppEEARS

AE = "https://appeears.earthdatacloud.nasa.gov/api"
LAYERS = {
    "MOD13Q1.061": ["_250m_16_days_NDVI", "_250m_16_days_EVI", "_250m_16_days_NIR_reflectance",
                    "_250m_16_days_MIR_reflectance", "_250m_16_days_pixel_reliability", "_250m_16_days_VI_Quality",
                    "_250m_16_days_composite_day_of_the_year"],
    "MOD10A1.061": ["NDSI_Snow_Cover", "NDSI_Snow_Cover_Basic_QA", "NDSI_Snow_Cover_Algorithm_Flags_QA"],
    "MOD10A2.061": ["Maximum_Snow_Extent", "Eight_Day_Snow_Cover"],
}
PIXRES = {"MOD13Q1.061": 250, "MOD10A1.061": 500, "MOD10A2.061": 500}


def ae_login():
    import requests
    auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise SystemExit("~/.netrc 에 urs.earthdata.nasa.gov 항목이 없다")
    r = requests.post(f"{AE}/login", auth=(auth[0], auth[2]), timeout=60)
    r.raise_for_status()
    return r.json()["token"]


def ae_task(name, coords, product, dates):
    return {"task_type": "point", "task_name": name,
            "params": {"dates": dates,
                       "layers": [{"product": product, "layer": L} for L in LAYERS[product]],
                       "coordinates": coords}}


def _coords(df, idcol, latcol, loncol, cat):
    return [{"id": str(i), "category": cat, "latitude": float(la), "longitude": float(lo)}
            for i, la, lo in zip(df[idcol], df[latcol], df[loncol])]


def cmd_appeears_test(a):
    import requests
    out = D_AE / "test"
    out.mkdir(parents=True, exist_ok=True)
    sub_p = out / "submitted.json"
    if sub_p.exists() and not a.force:
        log("[ae-test] 이미 제출됨(submitted.json). --force 없이 다시 내지 않는다")
        return
    pix = pd.read_csv(D_AE / "points" / "xe_labels_pixels.csv")
    # 고유 250 m 화소 가운데 대상별 고정 개수(알래스카 4, 레나 3, 캐나다 3)를 seed 20261004 로 고른다.
    # 500 m 화소도 서로 다르게 고른다. 좌표는 250 m 화소 중심(500 m 화소 안쪽에 있다)
    rng = np.random.default_rng(20261004)
    u = pix.drop_duplicates("pix500_id").drop_duplicates("pix250_id")
    pick = []
    for t, k in (("alaska", 4), ("lena", 3), ("canada", 3)):
        g = u[u.target == t].sort_values("pix250_id")
        pick.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    pick = pd.concat(pick, ignore_index=True)
    pick["test_id"] = [f"t{i:02d}_{t}" for i, t in enumerate(pick.target)]
    pick[["test_id", "target", "pix250_id", "pix500_id", "pix250_lat", "pix250_lon"]].to_csv(
        out / "test_points.csv", index=False)
    coords = _coords(pick, "test_id", "pix250_lat", "pix250_lon", "xe_test")
    dates = [{"startDate": "06-01-2018", "endDate": "06-30-2018"}]
    token = ae_login()
    hdr = {"Authorization": f"Bearer {token}"}
    subs = {}
    for prod in ("MOD13Q1.061", "MOD10A1.061", "MOD10A2.061"):
        name = f"xe_test_{prod.split('.')[0].lower()}_20261004"
        task = ae_task(name, coords, prod, dates)
        write_json(out / f"request_{prod.split('.')[0].lower()}.json", task)
        t_sub = now_kst()
        r = requests.post(f"{AE}/task", json=task, headers=hdr, timeout=120)
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text[:500]}
        subs[prod] = dict(task_name=name, http=r.status_code, response=body, submitted_kst=t_sub,
                          submitted_epoch=time.time())
        log(f"[ae-test] {prod} 제출 HTTP {r.status_code} {body.get('task_id', body)}")
    write_json(sub_p, dict(plan=PLAN, created=now_kst(), n_points=len(coords), dates=dates, tasks=subs))


def cmd_appeears_poll(a):
    """시험 요청 상태를 a.interval 초마다 기록하고, 완료되면 결과 묶음을 내려받는다."""
    import requests
    out = D_AE / "test"
    sub = json.loads((out / "submitted.json").read_text())
    tasks = {p: v["response"]["task_id"] for p, v in sub["tasks"].items() if "task_id" in v.get("response", {})}
    logp = out / "status_log.csv"
    first = not logp.exists()
    token, t_tok = ae_login(), time.time()
    done = {}
    t_end = time.time() + a.max_hours * 3600
    while time.time() < t_end and len(done) < len(tasks):
        if time.time() - t_tok > 12 * 3600:
            token, t_tok = ae_login(), time.time()
        hdr = {"Authorization": f"Bearer {token}"}
        for prod, tid in tasks.items():
            if prod in done:
                continue
            try:
                st = requests.get(f"{AE}/task/{tid}", headers=hdr, timeout=60).json()
            except Exception as e:
                st = {"status": f"poll_error {str(e)[:80]}"}
            with open(logp, "a") as f:
                if first:
                    f.write("time_kst,product,task_id,status,created,updated,completed\n")
                    first = False
                f.write(f"{now_kst()},{prod},{tid},{st.get('status')},{st.get('created')},{st.get('updated')},"
                        f"{st.get('completed')}\n")
            if st.get("status") in ("done", "error", "expired", "deleted"):
                done[prod] = st
                if st.get("status") == "done":
                    b = requests.get(f"{AE}/bundle/{tid}", headers=hdr, timeout=60).json()
                    files = []
                    for f_ in b.get("files", []):
                        dst = out / prod.split(".")[0].lower() / f_["file_name"]
                        r = http_download(f"{AE}/bundle/{tid}/{f_['file_id']}", dst,
                                          session=_auth_session(token))
                        files.append(dict(file_name=f_["file_name"], file_type=f_.get("file_type"),
                                          bytes=r["bytes"], sha256=r["sha256"], status=r["status"]))
                    st["bundle_files"] = files
                write_json(out / f"result_{prod.split('.')[0].lower()}.json", st)
                log(f"[ae-poll] {prod} 상태 {st.get('status')}")
        if len(done) < len(tasks):
            time.sleep(a.interval)
    # 대기 시간 요약
    lat = {}
    for prod, st in done.items():
        s0 = datetime.fromtimestamp(sub["tasks"][prod]["submitted_epoch"], KST)
        lat[prod] = dict(status=st.get("status"), submitted_kst=sub["tasks"][prod]["submitted_kst"],
                         api_created=st.get("created"), api_completed=st.get("completed"),
                         first_seen_done_kst=None)
        rows = pd.read_csv(logp)
        rr = rows[(rows["product"] == prod) & (rows["status"] == st.get("status"))]
        if len(rr):
            seen = datetime.strptime(rr.time_kst.iloc[0], "%Y-%m-%d %H:%M:%S %z")
            lat[prod]["first_seen_done_kst"] = rr.time_kst.iloc[0]
            lat[prod]["latency_min_upper"] = round((seen - s0).total_seconds() / 60, 1)
        try:
            c = datetime.fromisoformat(str(st.get("completed")).replace("Z", "+00:00"))
            if c.tzinfo is None:  # API 시각은 시간대 표기 없는 UTC
                c = c.replace(tzinfo=timezone.utc)
            lat[prod]["latency_min_api"] = round((c - s0).total_seconds() / 60, 1)
            cr = datetime.fromisoformat(str(st.get("created"))).replace(tzinfo=timezone.utc)
            st_proc = rows[(rows["product"] == prod) & (rows["status"] == "processing")]
            if len(st_proc):
                p0 = datetime.strptime(st_proc.time_kst.iloc[0], "%Y-%m-%d %H:%M:%S %z")
                lat[prod]["queue_min_upper"] = round((p0 - cr).total_seconds() / 60, 1)
        except Exception:
            pass
    write_json(out / "latency_summary.json", dict(created=now_kst(), poll_interval_s=a.interval, tasks=lat,
                                                  pending=[p for p in tasks if p not in done]))


def _auth_session(token):
    import requests
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def cmd_appeears_prep(a):
    """전체 점 요청 JSON 을 만든다. 제출하지 않는다."""
    out = D_AE / "requests"
    out.mkdir(parents=True, exist_ok=True)
    sets = {"labels": pd.read_csv(D_AE / "points" / "xe_labels_pixels.csv"),
            "lena_land": pd.read_csv(D_AE / "points" / "xe_lena_land_pixels.csv")}
    # 날짜: V 는 2015–2020 의 6–8월(반복 구간), S 는 2014-09-01 – 2021-08-31(수문년 9–8월 7개. '2015–2020'
    # 수문년의 정의를 추출 단계에서 고정하도록 앞뒤 1개 수문년을 더 받는다)
    DATES = {"MOD13Q1.061": [{"startDate": "06-01", "endDate": "08-31", "recurring": True, "yearRange": [2015, 2020]}],
             "MOD10A1.061": [{"startDate": "09-01-2014", "endDate": "08-31-2021"}],
             "MOD10A2.061": [{"startDate": "09-01-2014", "endDate": "08-31-2021"}]}
    ROLE = {"MOD13Q1.061": "V 주 경로", "MOD10A1.061": "S 주 경로", "MOD10A2.061": "S 대체 경로(MOD10A2 정의 고정)"}
    idx = []
    for sname, tab in sets.items():
        for prod in LAYERS:
            res = PIXRES[prod]
            u = tab.drop_duplicates(f"pix{res}_id").sort_values(f"pix{res}_id").reset_index(drop=True)
            nchunk = int(np.ceil(len(u) / a.chunk))
            for k in range(nchunk):
                part = u.iloc[k * a.chunk:(k + 1) * a.chunk]
                cat = sname if sname != "labels" else None
                coords = [{"id": pid, "category": (cat or tg), "latitude": float(la), "longitude": float(lo)}
                          for pid, tg, la, lo in zip(part[f"pix{res}_id"],
                                                     part["target"] if "target" in part else [sname] * len(part),
                                                     part[f"pix{res}_lat"], part[f"pix{res}_lon"])]
                name = f"xe_{prod.split('.')[0].lower()}_{sname}_p{k + 1:02d}of{nchunk:02d}"
                task = ae_task(name, coords, prod, DATES[prod])
                p = out / f"{name}.json"
                write_json(p, task)
                n_dates = None
                idx.append(dict(file=p.name, set=sname, product=prod, role=ROLE[prod], pixel_m=res,
                                n_points=len(coords), part=k + 1, n_parts=nchunk, dates=json.dumps(DATES[prod]),
                                sha256=sha256_file(p), submitted=False))
    # 레나 S 군 대체안: 면적 요청(GeoTIFF, 원 투영). 점 요청 행 수가 크면 이것을 쓴다(등록 경로 아님, 참고)
    lena = lena_points(land_only=False)
    box = [[float(lena.lon.min()) - 0.01, float(lena.lat.min()) - 0.01],
           [float(lena.lon.max()) + 0.01, float(lena.lat.min()) - 0.01],
           [float(lena.lon.max()) + 0.01, float(lena.lat.max()) + 0.01],
           [float(lena.lon.min()) - 0.01, float(lena.lat.max()) + 0.01],
           [float(lena.lon.min()) - 0.01, float(lena.lat.min()) - 0.01]]
    geo = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "lena_box"},
                                                      "geometry": {"type": "Polygon", "coordinates": [box]}}]}
    for prod in ("MOD10A1.061", "MOD13Q1.061"):
        name = f"xe_{prod.split('.')[0].lower()}_lena_box_area"
        task = {"task_type": "area", "task_name": name,
                "params": {"dates": DATES[prod], "layers": [{"product": prod, "layer": L} for L in LAYERS[prod]],
                           "geo": geo, "output": {"format": {"type": "geotiff"}, "projection": "native"}}}
        p = out / "alternative_area" / f"{name}.json"
        write_json(p, task)
        idx.append(dict(file=f"alternative_area/{p.name}", set="lena_box", product=prod,
                        role="참고 대체안(면적 요청, 등록 경로 아님)", pixel_m=PIXRES[prod], n_points=None, part=1,
                        n_parts=1, dates=json.dumps(DATES[prod]), sha256=sha256_file(p), submitted=False))
    pd.DataFrame(idx).to_csv(out / "requests_index.csv", index=False)
    write_json(out / "requests_meta.json",
               dict(created=now_kst(), plan=PLAN, chunk_points=a.chunk,
                    chunk_note="AppEEARS 점 수 상한은 문서에서 찾지 못했다[미확인]. 한 요청 1,000점으로 나눈 것은 가정이다",
                    layers=LAYERS, dates=DATES, submitted=False,
                    submit_rule="시험 요청 대기 시간을 본 뒤 제출한다. 마감 T0 + 96 h(2026-10-08 14:22:39 +0900)"))
    for sname in sets:
        for prod in LAYERS:
            sel = [r for r in idx if r["set"] == sname and r["product"] == prod]
            log(f"[ae-prep] {sname} {prod}: 요청 {len(sel)}건, 점 {sum(r['n_points'] for r in sel)}")


SUBMIT_P = D_AE / "requests" / "submitted_full.json"
STATUS_LOG = D_AE / "requests" / "status_log_full.csv"


def cmd_appeears_submit(a):
    """준비된 전체 점 요청 JSON 을 (set:product) 순서대로 제출한다. 이미 task_id 가 있는 요청은 다시 내지 않는다.
    제출 기록은 requests/submitted_full.json(요청마다 바로 저장), requests_index.csv 의 submitted·task_id 열."""
    import requests
    idx_p = D_AE / "requests" / "requests_index.csv"
    idx = pd.read_csv(idx_p)
    sub = json.loads(SUBMIT_P.read_text()) if SUBMIT_P.exists() else {}
    sel = [s.split(":") for s in a.select.split(",") if s]
    token = ae_login()
    hdr = {"Authorization": f"Bearer {token}"}
    n_err = 0
    stop = False
    for set_, prod in sel:
        rows = idx[(idx["set"] == set_) & (idx["product"] == prod) &
                   ~idx.file.astype(str).str.startswith("alternative_area")].sort_values("part")
        for r in rows.itertuples():
            name = Path(r.file).stem
            if sub.get(name, {}).get("task_id"):
                continue
            if "superseded_by" in idx and isinstance(getattr(r, "superseded_by", None), str) and r.superseded_by:
                continue  # 조각으로 대체된 원 요청은 다시 내지 않는다
            p = D_AE / "requests" / r.file
            sh = sha256_file(p)
            assert sh == r.sha256, f"요청 파일 해시가 색인과 다르다: {r.file}"
            task = json.loads(p.read_text())
            assert task["task_name"] == name and len(task["params"]["coordinates"]) == int(r.n_points)
            resp, err = None, None
            for k in range(4):
                try:
                    resp = requests.post(f"{AE}/task", json=task, headers=hdr, timeout=300)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        err = f"HTTP {resp.status_code}"
                        time.sleep(30 * (k + 1))
                        continue
                    break
                except Exception as e:
                    err = str(e)[:200]
                    time.sleep(30 * (k + 1))
            t_sub = now_kst()
            if resp is None:
                body, http = {"error": err}, None
            else:
                http = resp.status_code
                try:
                    body = resp.json()
                except Exception:
                    body = {"text": resp.text[:500]}
            tid = body.get("task_id") if isinstance(body, dict) else None
            sub[name] = dict(file=r.file, set=set_, product=prod, role=r.role, part=int(r.part), n_parts=int(r.n_parts),
                             n_points=int(r.n_points), request_sha256=sh, http=http, task_id=tid,
                             api_status=body.get("status") if isinstance(body, dict) else None,
                             response=(None if tid else body), submitted_kst=t_sub, submitted_epoch=time.time())
            write_json(SUBMIT_P, sub)
            log(f"[ae-submit] {name} HTTP {http} task_id {tid}" + ("" if tid else f" 응답 {str(body)[:200]}"))
            if tid:
                n_err = 0
            else:
                n_err += 1
                if n_err >= a.max_errors:
                    log(f"[ae-submit] 연속 실패 {n_err}회, 중단")
                    stop = True
                    break
            time.sleep(a.pause)
        if stop:
            break
    idx["task_id"] = [sub.get(Path(f).stem, {}).get("task_id") for f in idx.file.astype(str)]
    idx["submitted"] = idx.task_id.notna()
    idx["submitted_kst"] = [sub.get(Path(f).stem, {}).get("submitted_kst") for f in idx.file.astype(str)]
    idx.to_csv(idx_p, index=False)
    meta_p = D_AE / "requests" / "requests_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["submitted"] = bool(idx.submitted.any())
    meta["submission"] = dict(last_run_kst=now_kst(), select=a.select, n_submitted=int(idx.submitted.sum()),
                              n_requests_total=int((~idx.file.astype(str).str.startswith("alternative_area")).sum()),
                              record="requests/submitted_full.json(task_id), requests_index.csv(task_id·submitted 열)")
    write_json(meta_p, meta)
    for set_, prod in sel:
        m = (idx["set"] == set_) & (idx["product"] == prod) & ~idx.file.astype(str).str.startswith("alternative_area")
        log(f"[ae-submit] {set_} {prod}: 제출 {int(idx[m].submitted.sum())}/{int(m.sum())}")


def cmd_appeears_split(a):
    """거절된(값 수 상한 초과) 요청 파일을 같은 좌표 순서로 N 조각으로 나눠 새 요청 파일을 만들고 색인에 넣는다.
    원 요청 행에는 superseded_by 를 적어 다시 내지 않게 한다. 제출은 appeears-submit 이 한다."""
    idx_p = D_AE / "requests" / "requests_index.csv"
    idx = pd.read_csv(idx_p)
    if "superseded_by" not in idx:
        idx["superseded_by"] = None
    src = D_AE / "requests" / a.file
    task = json.loads(src.read_text())
    coords = task["params"]["coordinates"]
    row = idx[idx.file == a.file].iloc[0]
    n = a.parts
    bounds = np.linspace(0, len(coords), n + 1).astype(int)
    new_files = []
    for k in range(n):
        part = coords[bounds[k]:bounds[k + 1]]
        name = f"{src.stem}_s{k + 1}of{n}"
        t = {"task_type": task["task_type"], "task_name": name,
             "params": {"dates": task["params"]["dates"], "layers": task["params"]["layers"], "coordinates": part}}
        p = src.with_name(f"{name}.json")
        write_json(p, t)
        new_files.append(p.name)
        idx = pd.concat([idx, pd.DataFrame([dict(file=p.name, set=row["set"], product=row["product"], role=row["role"],
                                                 pixel_m=row["pixel_m"], n_points=len(part), part=row["part"],
                                                 n_parts=row["n_parts"], dates=row["dates"], sha256=sha256_file(p),
                                                 submitted=False, superseded_by=None,
                                                 note=f"split {k + 1}/{n} of {a.file}({a.reason})")])],
                        ignore_index=True)
        log(f"[ae-split] {p.name} 점 {len(part)}")
    idx.loc[idx.file == a.file, "superseded_by"] = ";".join(new_files)
    idx.to_csv(idx_p, index=False)


def cmd_appeears_status(a):
    """제출된 전체 요청의 상태를 한 번 조회해 requests/status_log_full.csv 에 덧붙이고 요약을 쓴다.
    --fetch 면 done 인 과제의 결과 묶음을 results/<task_name>/ 에 내려받는다(이미 받은 것은 건너뜀)."""
    import requests
    sub = json.loads(SUBMIT_P.read_text())
    token = ae_login()
    hdr = {"Authorization": f"Bearer {token}"}
    first = not STATUS_LOG.exists()
    rows = []
    res_root = D_AE / "results"
    for name, s in sub.items():
        tid = s.get("task_id")
        if not tid:
            continue
        try:
            st = requests.get(f"{AE}/task/{tid}", headers=hdr, timeout=120).json()
        except Exception as e:
            st = {"status": f"poll_error {str(e)[:80]}"}
        row = dict(time_kst=now_kst(), task_name=name, set=s["set"], product=s["product"], part=s["part"],
                   n_points=s["n_points"], task_id=tid, status=st.get("status"), created=st.get("created"),
                   updated=st.get("updated"), completed=st.get("completed"),
                   progress=json.dumps(st.get("progress")) if st.get("progress") is not None else None,
                   error=json.dumps(st.get("error"))[:300] if st.get("error") else None, fetched=False)
        done_p = res_root / name / "bundle_meta.json"
        if done_p.exists():
            row["fetched"] = True
        elif a.fetch and st.get("status") == "done":
            b = requests.get(f"{AE}/bundle/{tid}", headers=hdr, timeout=120).json()
            files = []
            ses = _auth_session(token)
            for f_ in b.get("files", []):
                dst = res_root / name / f_["file_name"]
                r = http_download(f"{AE}/bundle/{tid}/{f_['file_id']}", dst, session=ses,
                                  expect_size=f_.get("file_size"), expect_sha256=f_.get("sha256"))
                files.append(dict(file_name=f_["file_name"], file_type=f_.get("file_type"), api_sha256=f_.get("sha256"),
                                  bytes=r["bytes"], sha256=r["sha256"], status=r["status"]))
            ok = all(str(f["status"]).startswith(("ok", "cached")) for f in files) and files
            if ok:
                write_json(done_p, dict(task_name=name, task_id=tid, fetched_kst=now_kst(), task=st, files=files))
                row["fetched"] = True
            log(f"[ae-status] {name} 결과 묶음 {len(files)}파일 {'완료' if ok else '일부 실패'}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(STATUS_LOG, mode="a", header=first, index=False)
    summ = {}
    for (s_, p_), g in df.groupby(["set", "product"]):
        summ[f"{s_}:{p_}"] = dict(n_tasks=int(len(g)), status=g.status.value_counts().to_dict(),
                                   n_fetched=int(g.fetched.sum()))
    write_json(D_AE / "requests" / "status_summary.json",
               dict(created=now_kst(), plan=PLAN, deadline="T0 + 96 h(2026-10-08 14:22:39 +0900)",
                    n_tasks=int(len(df)), by_set_product=summ,
                    overall=df.status.value_counts().to_dict(), n_fetched=int(df.fetched.sum())))
    for k, v in summ.items():
        log(f"[ae-status] {k}: {v['status']} 받음 {v['n_fetched']}/{v['n_tasks']}")


# ---------------------------------------------------------------- SOURCE.md

SOURCES = {
    D_WC: dict(
        title="ESA WorldCover 10 m 2021 v200(XE V군 wc_* 7열)",
        lines=[
            "- 제품: ESA WorldCover 10 m 2021 v200(algorithm V2.0.0). 3° × 3° 타일, EPSG:4326, 화소 8.333e-5°(약 10 m), uint8, nodata 0.",
            "- 취득 경로: AWS S3 무서명 HTTPS `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_<타일>_Map.tif`. 타일 목록은 `esa_worldcover_grid.geojson`(같은 버킷).",
            "- 약관: CC BY 4.0. 근거는 타일 GeoTIFF 태그 `license = 'CC-BY 4.0 - https://creativecommons.org/licenses/by/4.0/'` 와 제품 사용 설명서(`docs/WorldCover_PUM_V2.0.pdf`) 5.1절 'provided free of charge, without restriction of use'.",
            "- 지도 표기 문구(PUM 5.2): '© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium'.",
            "- 인용: Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O., 2022. ESA WorldCover 10 m 2021 v200. doi:10.5281/zenodo.7254221.",
            "- 범례(타일 태그 `legend` 와 PUM 표가 같다): 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up, 60 Bare/sparse vegetation, 70 Snow and ice, 80 Permanent water bodies, 90 Herbaceous wetland, 95 Mangroves, 100 Moss and lichen. 등록 열과의 대응(wc_tree 등)은 추출 단계에서 라벨 결합 전에 고정한다(등록 문서 2.5절).",
            "- 범위(1차, 2026-10-04 14:36): XE 대상 라벨 위치(`fidelity_base_v3.csv` 의 알래스카·레나·캐나다 17,385행, 좌표 열만 읽음) 주변 위도 ±0.001°, 경도 ±0.001°/cosφ 상자가 닿는 타일과 레나 지도 상자(`lena_grid_cells_v1.csv` 범위 ±0.01°/±0.03°)의 타일. 필요 43타일, 모두 내려받음. 라벨 17,385점과 레나 격자 53,011셀이 모두 내려받은 타일 안에 있음을 확인했다.",
            "- 범위(2차, `--scope all_labels`): `fidelity_base_v3.csv` ∪ `fidelity_base_v4.csv` 의 모든 지역 라벨 위치(v4 18,088행이 v3 17,572행을 포함, 좌표 동일) 주변 같은 여유 상자. 라벨 위치용 131타일(제품 격자에 있음 130, 범위 밖 1: S63W063, GTN-P 남극 사이트 1점)과 레나 지도 상자용 타일을 합쳐 132타일이고, 격자에 있는 131타일을 모두 받았다(2026-10-04 21:24:40 +0900, 7.45 GB. 이 가운데 1차 범위 43타일 2.46 GB, XE 밖 지역 전용 89타일 4.99 GB). 타일별 용도는 `manifest_tiles.csv` 의 `used_by`(XE 대상은 `labels_<대상>`·`lena_box`, 그 밖은 `v3:<region>`·`v4:<region>`). `download_meta.json` 의 `history` 에 1차 메타를 남겼다.",
            "- 검증: 타일마다 HEAD 의 Content-Length 와 받은 크기가 같다. S3 ETag 이 단일 파트(md5)인 타일은 md5 도 확인했다. 다중 파트 ETag(`-N` 접미)은 md5 가 아니어서 크기만 확인했다. 타일별 sha256 은 `manifest_tiles.csv` 와 아래 표에 있다.",
        ]),
    D_CAVM: dict(
        title="Raster Circumpolar Arctic Vegetation Map(XE V군 cavm_class)",
        lines=[
            "- 제품: Raster CAVM(Raynolds et al. 2019). LAEA(구, R = 6,370,997 m), 1,000 m, int8, nodata 127. 식생 등급 1–43, 91 FW(담수), 92 SW(염수), 93 GL(빙하), 99 NA(비북극 육지). 범례는 `v2/unzipped/Raster CAVM legend.csv`.",
            "- 취득 경로: Mendeley Data 공개 API(`https://data.mendeley.com/public-api/datasets/c4xj5rv6kv`). 판 2(2022-01-17, doi:10.17632/c4xj5rv6kv.2)를 주 판으로 쓴다. 판 1(2019-05-16, within_grid_inputs.md 부록 B 가 인용한 판)은 대조용이다. 두 판의 `raster_cavm_v1.tif` 는 sha256 이 같고, 판 2 는 범례 CSV 를 더했다.",
            "- 검증: zip 의 sha256 이 Mendeley API 의 `sha256_hash` 와 같다.",
            "- 약관: CC BY-NC 3.0(Mendeley API `data_licence`: 'You are free to adapt, copy or redistribute the material, providing you attribute appropriately and do not use the material for commercial purposes.'). 비상업 조건이 있다.",
            "- 인용: Raynolds, M.K., Walker, D.A., Balser, A., et al., 2019. A raster version of the Circumpolar Arctic Vegetation Map (CAVM). Remote Sensing of Environment 232, 111297. doi:10.1016/j.rse.2019.111297. 자료: doi:10.17632/c4xj5rv6kv.2.",
            "- 한계: 툰드라만 분류한다. 내륙 알래스카 산림 셀은 99(비북극)다(within_grid_inputs.md 5.10).",
        ]),
    D_NCSCD: dict(
        title="NCSCD v2 0.012° 격자(XE O군 ncscd_soc_0_30, ncscd_soc_0_100)",
        lines=[
            "- 제품: Northern Circumpolar Soil Carbon Database v2 격자 GeoTIFF(WGS84, 0.012°). `SOCC30`(0–30 cm), `SOCC100`(0–100 cm), `SOCC200`, `SOCC300`, 토양 등급 비율, 영구동토 구역. int16, nodata −32768.",
            "- 단위: 격자 자료의 SOCC 는 hg C m⁻²(Bolin Centre technical.php: 'In the vector database the SOCC is given in the unit kg C m-2 while it is given as hg C m-2 in the gridded datasets').",
            "- 취득 경로: `https://bolin.su.se/data/ncscd/data/v3/raster/NCSCDv2_Circumpolar_raster_0012deg.zip`(Bolin Centre Database, 계정 없음).",
            "- 약관: 다운로드 페이지(https://bolin.su.se/data/ncscd/)와 기술 페이지(technical.php)에 라이선스 명시가 없다[미확인, 2026-10-04 재확인]. 페이지 문구: 'Please refer to this paper if you make use of any dataset downloaded from this site'(아래 두 논문 인용 요구). 판 1 의 SND 기록(doi:10.5879/ecds/00000001)은 접근 수준을 'Data are openly accessible' 로 적는다.",
            "- 인용: Hugelius, G., Tarnocai, C., Broll, G., Canadell, J.G., Kuhry, P., Swanson, D.K., 2013. The Northern Circumpolar Soil Carbon Database: spatially distributed datasets of soil coverage and soil carbon storage in the northern permafrost regions. Earth Syst. Sci. Data 5, 3–13. doi:10.5194/essd-5-3-2013. Hugelius, G., Bockheim, J.G., Camill, P., et al., 2013. A new data set for estimating organic carbon storage to 3 m depth in soils of the northern circumpolar permafrost region. Earth Syst. Sci. Data 5, 393–402. doi:10.5194/essd-5-393-2013.",
        ]),
    D_SG: dict(
        title="SoilGrids 2.0 250 m 원해상도 부족 창(XE O군 sg250_* 7열)",
        lines=[
            "- 제품: SoilGrids 2.0(ISRIC) 250 m, IGH(Interrupted Goode Homolosine). 7층: soc 0–5·5–15·15–30 cm(dg/kg), bdod 5–15 cm(cg/cm³), clay·sand·silt 5–15 cm(g/kg). 기존 `data/raw/soilgrids_multi/` 와 같은 층·같은 저장 형식이다. 기존 폴더는 고치지 않았다.",
            "- 취득 경로: 등록 주 경로인 ISRIC VRT 창 읽기(`/vsicurl/https://files.isric.org/soilgrids/latest/data/<prop>/<layer>.vrt`). 2° × 2° 칸 단위, IGH 여유 12 km(`soilgrids_multiregion.py` 와 같다).",
            "- 범위 결정: XE 대상 라벨 위치(17,385점)와 레나 지도 육지 셀(38,173셀)이 든 2° 칸 57개를 받았다. (1) 부족 창 36칸: 기존 `soilgrids_multi` 창 경계 밖 점이 있는 칸(등록 문서의 문자 그대로의 대상). (2) 정렬 재취득 21칸: 기존 창이 덮는 칸. 기존 창은 소수 창으로 읽어 저장 변환과 자료가 최대 0.5 화소 어긋나므로 같은 제품을 정수 화소 창으로 다시 받았다(`docs/research/2026-10-04/impl_notes/xe_stage2_acquire.md` 3절). 칸 구분은 `windows_meta.json` 의 `kind`. 모든 창은 원격 VRT 격자에 정수 화소로 맞췄고, 표본 칸에서 원격 VRT 값과 화소 단위로 같음을 확인했다. 층마다 VRT 원점이 달라 같은 칸에서도 층별 변환이 다르므로 추출은 층별 변환으로 한다. 포함률은 `coverage_meta.json`.",
            "- 약관: CC BY 4.0(ISRIC, https://docs.isric.org/globaldata/soilgrids/ 'SoilGrids maps are publicly available under the CC-BY 4.0 License').",
            "- 인용: Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, G.B.M., Kempen, B., Ribeiro, E., Rossiter, D., 2021. SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. SOIL 7, 217–240. doi:10.5194/soil-7-217-2021.",
        ]),
    D_AE: dict(
        title="AppEEARS 점 요청(XE V군 MOD13Q1, S군 MOD10A1·MOD10A2)",
        lines=[
            "- 서비스: NASA AppEEARS API(https://appeears.earthdatacloud.nasa.gov/api). Earthdata 로그인은 `~/.netrc` 의 urs.earthdata.nasa.gov 항목을 쓴다. 토큰은 파일에 쓰지 않는다.",
            "- 제품: MOD13Q1.061(250 m, 16일, LP DAAC), MOD10A1.061(500 m, 일, NSIDC), MOD10A2.061(500 m, 8일, NSIDC). 층 목록은 `requests/requests_meta.json`.",
            "- `points/`: XE 대상 라벨 위치와 레나 지도 육지 셀을 MODIS 사인 곡선 격자(R = 6,371,007.181 m, 250 m 화소 231.656 m, 500 m 화소 463.313 m)의 고유 화소로 묶은 표. 요청 좌표는 화소 중심이다(왕복 변환으로 같은 화소임을 확인). 라벨 표에서는 좌표 열만 읽었다.",
            "- `test/`: 0일 소형 시험 요청(10점, 2018-06-01 – 2018-06-30, 제품별 1건, 3건). 제출 시각·상태 기록·결과 묶음·대기 시간 요약(`latency_summary.json`). 제출부터 완료까지 MOD13Q1 17.4분, MOD10A2 18.3분, MOD10A1 19.8분(대기열 약 15.7분, 처리 2–4분). 결과 묶음의 `*-request.json` 은 AppEEARS 가 돌려준 요청 기록이며 계정 식별자를 담고 있다(공유 금지).",
            "- `requests/`: 전체 점 요청 JSON. V 는 2015–2020 의 06-01 – 08-31 반복 구간, S 는 2014-09-01 – 2021-08-31. 한 요청 1,000점으로 나눴다. 점 수·동시 과제 수 상한은 API 문서·LP DAAC 안내에 없다(2026-10-04 확인). 대신 요청이 만드는 값의 총수에 상한이 있다: `xe_mod10a1_labels_p01of02`(1,000점 × 2,557일 × 3층 = 7,671,000값)가 HTTP 400 'The total number of values that this request will generate exceeds the maximum allowed by 119.2%' 로 거절됐다(상한 약 3.5 × 10⁶ 값으로 역산). 이 요청은 같은 좌표 순서로 3조각(333·333·334점, `_s1of3`–`_s3of3`, `appeears-split`)으로 나눠 다시 냈고 원 요청 행은 `superseded_by` 로 표시했다. 제출 기록은 `requests/submitted_full.json`(task_id, 제출 시각, HTTP 응답)과 `requests_index.csv` 의 `submitted`·`task_id`·`submitted_kst` 열, 상태 조회 기록은 `requests/status_log_full.csv`·`status_summary.json`. 제출 범위(2026-10-04 21:11–21:22 +0900, 87건 수락): 라벨 위치 MOD13Q1(3건)·MOD10A1(4건, 등록 주 경로)·MOD10A2(2건), 레나 육지 셀 MOD13Q1(39건)·MOD10A2(39건). 레나 육지 MOD10A1(39건)은 출력 약 16 GB 추정이고 지도는 해석 규칙 1·2 에서만 다시 만들므로 내지 않았다(JSON 만 보존). `alternative_area/` 는 레나 상자 면적 요청 참고안이다(등록 경로 아님, 미제출).",
            "- `results/`: 완료된 전체 요청의 결과 묶음(`appeears-status --fetch`). 과제별 폴더 `results/<task_name>/` 에 CSV·granule 목록·metadata·README 와 `bundle_meta.json`(API 가 준 파일 sha256 과 대조). 결과 묶음의 `*-request.json` 은 계정 식별자를 담는다(공유 금지).",
            "- 약관: NASA EOSDIS 자료는 사용 제한이 없다(NASA Earth Science Data and Information Policy). 각 제품 DOI 인용을 요청한다.",
            "- 인용: Didan, K., 2021. MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS LP DAAC. doi:10.5067/MODIS/MOD13Q1.061. Hall, D.K., Riggs, G.A., 2021. MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid, Version 61. NSIDC DAAC. doi:10.5067/MODIS/MOD10A1.061. Hall, D.K., Riggs, G.A., 2021. MODIS/Terra Snow Cover 8-Day L3 Global 500m SIN Grid, Version 61. NSIDC DAAC. doi:10.5067/MODIS/MOD10A2.061. AppEEARS Team, 2026. Application for Extracting and Exploring Analysis Ready Samples (AppEEARS). Ver. 3.130(시험 요청 응답의 svc_version). NASA EOSDIS LP DAAC, USGS EROS Center, Sioux Falls, South Dakota, USA. 접속 2026-10-04.",
        ]),
}


def cmd_source_md(a):
    only = set(getattr(a, "only", None) or [])
    for d, info in SOURCES.items():
        if not d.exists() or (only and d.name not in only):
            continue
        rows = []
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != "SOURCE.md" and not p.name.endswith((".part", ".tmp")):
                rows.append((str(p.relative_to(d)), p.stat().st_size, sha256_file(p)))
        tot = sum(r[1] for r in rows)
        txt = [f"# {info['title']}", "",
               f"- 작성: {now_kst()}, `scripts/0_download/xe_stage2_acquire.py source-md`",
               f"- 근거: {PLAN}. 마감은 V(WorldCover, CAVM)·O 가 T0 + 72 h(2026-10-07 14:22:39 +0900), "
               f"V(MOD13Q1)·S 가 T0 + 96 h(2026-10-08 14:22:39 +0900)다.",
               f"- 파일 {len(rows)}개, 합계 {tot / 1e6:,.1f} MB", ""] + info["lines"] + \
              ["", "## 파일 체크섬(sha256)", "", "| 경로 | 바이트 | sha256 |", "|---|---:|---|"] + \
              [f"| `{r[0]}` | {r[1]:,} | `{r[2]}` |" for r in rows] + [""]
        (d / "SOURCE.md").write_text("\n".join(txt))
        log(f"[source-md] {d.relative_to(ROOT)}: 파일 {len(rows)}, {tot / 1e6:,.1f} MB")


# ---------------------------------------------------------------- 진입점

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("points")
    p = sp.add_parser("worldcover"); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--scope", choices=["xe", "all_labels"], default="xe",
                   help="xe: XE 대상 라벨(알래스카·레나·캐나다)+레나 상자. all_labels: v3 ∪ v4 모든 지역 라벨+레나 상자")
    p.add_argument("--dry-run", action="store_true", help="없는 타일 목록만 적고 내려받지 않는다")
    sp.add_parser("cavm")
    sp.add_parser("ncscd")
    p = sp.add_parser("soilgrids"); p.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("appeears-test"); p.add_argument("--force", action="store_true")
    p = sp.add_parser("appeears-poll"); p.add_argument("--interval", type=int, default=60)
    p.add_argument("--max-hours", type=float, default=30.0)
    p = sp.add_parser("appeears-prep"); p.add_argument("--chunk", type=int, default=1000)
    p = sp.add_parser("appeears-submit")
    p.add_argument("--select", default="labels:MOD13Q1.061,labels:MOD10A1.061,labels:MOD10A2.061,"
                   "lena_land:MOD13Q1.061,lena_land:MOD10A2.061",
                   help="제출할 (set:product) 목록, 쉼표 구분. 순서대로 낸다")
    p.add_argument("--pause", type=float, default=2.0, help="제출 사이 대기(초)")
    p.add_argument("--max-errors", type=int, default=3, help="연속 실패가 이 수에 이르면 멈춘다")
    p = sp.add_parser("appeears-split"); p.add_argument("--file", required=True, help="requests/ 안의 요청 파일 이름")
    p.add_argument("--parts", type=int, required=True); p.add_argument("--reason", default="값 수 상한 초과")
    p = sp.add_parser("appeears-status"); p.add_argument("--fetch", action="store_true",
                                                         help="done 인 과제의 결과 묶음을 results/ 에 내려받는다")
    p = sp.add_parser("source-md"); p.add_argument("--only", nargs="*", default=None,
                                                   help="폴더 이름(worldcover_v200 등)만 다시 쓴다")
    a = ap.parse_args()
    fn = {"points": cmd_points, "worldcover": cmd_worldcover, "cavm": cmd_cavm, "ncscd": cmd_ncscd,
          "soilgrids": cmd_soilgrids, "appeears-test": cmd_appeears_test, "appeears-poll": cmd_appeears_poll,
          "appeears-prep": cmd_appeears_prep, "appeears-submit": cmd_appeears_submit,
          "appeears-split": cmd_appeears_split, "appeears-status": cmd_appeears_status,
          "source-md": cmd_source_md}[a.cmd]
    log(f"시작 {a.cmd} (T0 경과 {(datetime.now(KST) - T0).total_seconds() / 3600:.1f} h)")
    fn(a)
    log(f"끝 {a.cmd}")


if __name__ == "__main__":
    main()
