"""
① 데이터 확보 — GTN-P (Global Terrestrial Network for Permafrost).

브루트포스(ID 1..5000) 대신 LIST + BULK 엔드포인트 사용:
  GET /api/list-pt-datasets / list-alt-datasets  -> 전체 데이터셋 열거(각 2회 호출)
  GET /api/list-sites?boreholes=true&metadata=true -> borehole 좌표/국가
  GET /api/data?pt_data=<≤20 IDs>&combined=false  -> ZIP(데이터셋별 CSV + metadata.json)

지역 필터(예: Alaska)는 sites의 lat/lon/country로 borehole을 골라 해당 데이터셋만 받는다.
주의: API의 data_access 필드는 내부 호스트(http://fastapi:8000)라 무시하고 공개 BASE로 요청.
"""
import io
import json
import time
import zipfile

import pandas as pd
import requests

from . import config as C


def _get(url):
    for attempt in range(C.HTTP_RETRIES):
        try:
            r = requests.get(url, timeout=C.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code}: {url}")
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}/{C.HTTP_RETRIES}: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def _get_data(url):
    """/api/data 전용: 네트워크 오류만 재시도(HTTP 500은 용량 초과이므로 즉시 실패→분할)."""
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=C.HTTP_TIMEOUT)
            if r.status_code == 200 and r.content.startswith(b"PK"):
                return r.content
            return None  # 500 등: 분할로 처리
        except requests.RequestException as e:
            print(f"  net retry {attempt + 1}: {e}")
            time.sleep(3)
    return None


def list_datasets(kind):
    """kind in {'pt','alt'} -> 데이터셋 dict 리스트."""
    r = _get(f"{C.GTNP_BASE}/list-{kind}-datasets")
    return r.json().get("datasets", []) if r else []


def fetch_sites():
    r = _get(f"{C.GTNP_BASE}/list-sites?boreholes=true&activelayers=true&metadata=true")
    return r.json() if r else []


def build_borehole_index(sites):
    """borehole_id -> dict(lat, lon, elev, country, site) 매핑 + DataFrame 저장용 레코드."""
    idx, records = {}, []
    for s in sites:
        cc, cn, sname = s.get("country_code"), s.get("country_name"), s.get("name")
        loc = s.get("location") or {}
        for b in s.get("boreholes", []):
            rec = dict(borehole_id=b["id"],
                       lat=b.get("latitude", loc.get("latitude_avg")),
                       lon=b.get("longitude", loc.get("longitude_avg")),
                       elev=b.get("elevation"), country=cc, country_name=cn, site=sname)
            idx[b["id"]] = rec
            records.append(rec)
    return idx, pd.DataFrame(records)


def build_activelayer_index(sites):
    """activelayer_id -> dict(lat, lon, country, site) 매핑 (ALT 지역 필터용)."""
    idx = {}
    for s in sites:
        cc, sname = s.get("country_code"), s.get("name")
        loc = s.get("location") or {}
        for a in s.get("activelayers", []):
            idx[a["id"]] = dict(lat=a.get("latitude", loc.get("latitude_avg")),
                                lon=a.get("longitude", loc.get("longitude_avg")),
                                country=cc, site=sname)
    return idx


def in_alaska(rec):
    if not rec:
        return False
    lat, lon, cc = rec.get("lat"), rec.get("lon"), rec.get("country")
    bb = C.ALASKA_BBOX
    return (cc == C.ALASKA_COUNTRY and lat is not None and lon is not None
            and bb["lat_min"] <= lat <= bb["lat_max"]
            and bb["lon_min"] <= lon <= bb["lon_max"])


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _extract_zip(content, out_dir, tag):
    """zip 바이트를 추출. metadata json은 덮어쓰기 방지용 tag 접미사."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            data = z.read(name)
            if name.endswith("_metadata.json"):
                stem = name[:-len(".json")]
                (out_dir / f"{stem}_{tag}.json").write_bytes(data)
            else:
                (out_dir / name).write_bytes(data)


def _existing_ids(out_dir, kind):
    """이미 받은 데이터셋 ID(파일명 {kind}_dataset_{id}_*.csv)."""
    ids = set()
    for f in out_dir.glob(f"{kind}_dataset_*.csv"):
        try:
            ids.add(int(f.stem.split("_")[2]))
        except (IndexError, ValueError):
            pass
    return ids


def _download_recursive(batch, key, out_dir, label, failed):
    """배치 다운로드. 실패(HTTP 500=용량초과) 시 절반으로 분할 재귀. 단건 실패는 failed에 기록."""
    url = f"{C.GTNP_BASE}/data?{key}={','.join(map(str, batch))}&combined=false"
    content = _get_data(url)
    if content:
        try:
            _extract_zip(content, out_dir, tag=f"{key}_{batch[0]}_{len(batch)}")
            return len(batch)
        except zipfile.BadZipFile:
            pass
    if len(batch) == 1:
        failed.append(batch[0])
        return 0
    mid = len(batch) // 2
    print(f"  [{label}] split {len(batch)} -> {mid}+{len(batch) - mid}")
    time.sleep(0.3)
    return (_download_recursive(batch[:mid], key, out_dir, label, failed)
            + _download_recursive(batch[mid:], key, out_dir, label, failed))


_STD_COLS = ["id", "date", "depth", "temperature", "flag", "dataset_id", "borehole_id", "site_id"]


def _download_combined(did, key, out_dir, variable):
    """combined=false가 서버 버그로 500인 대용량 데이터셋용 폴백: combined=true로 받아
    컬럼(local_time->date)을 표준 스키마로 정규화 후 저장."""
    content = _get_data(f"{C.GTNP_BASE}/data?{key}={did}&combined=true")
    if not content:
        return False
    kind = "pt" if key == "pt_data" else "alt"
    slug = (variable or "data").lower().replace(" ", "_").replace("/", "_")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for n in z.namelist():
            if not n.endswith(".csv"):
                continue
            df = pd.read_csv(io.BytesIO(z.read(n)))
            if "date" not in df.columns and "local_time" in df.columns:
                df = df.rename(columns={"local_time": "date"})
            df = df[[c for c in _STD_COLS if c in df.columns]]
            df.to_csv(out_dir / f"{kind}_dataset_{did}_{slug}.csv", index=False)
            return True
    return False


def bulk_download(ids, key, out_dir, label, meta_by_id=None):
    """ID 리스트를 적응형 배치로 받아 CSV 추출. 기존 건너뜀, 실패 분할, 단건 실패는 날짜분할."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "pt" if key == "pt_data" else "alt"
    have = _existing_ids(out_dir, kind)
    todo = [i for i in ids if i not in have]
    if have:
        print(f"[{label}] skip {len(have)} already present; {len(todo)} to fetch")

    ok, failed = 0, []
    for bi, batch in enumerate(_chunks(todo, C.BATCH_SIZE)):
        print(f"[{label}] batch {bi + 1} ({len(batch)} datasets)")
        ok += _download_recursive(batch, key, out_dir, label, failed)
        time.sleep(0.5)

    # 단건 실패(combined=false 서버버그) → combined=true 폴백
    if failed and meta_by_id:
        print(f"[{label}] combined=true retry for {len(failed)} datasets: {failed}")
        for did in failed:
            variable = (meta_by_id.get(did) or {}).get("variable")
            if _download_combined(did, key, out_dir, variable):
                ok += 1
                print(f"  [{label}] {did} recovered via combined=true")
            else:
                print(f"  [{label}] {did} STILL failed")
    elif failed:
        print(f"[{label}] failed singles (no meta): {failed}")

    print(f"[{label}] newly extracted {ok}/{len(todo)} (total present now: "
          f"{len(_existing_ids(out_dir, kind))}) -> {out_dir}")
    return ok


def run(region="alaska", variable="Ground Temperature", open_only=True,
        include_alt=True, manifest_only=False):
    """전체 확보 파이프라인.

    region: 'alaska' 또는 'all'
    variable: PT 필터(기본 Ground Temperature). None이면 전체.
    """
    C.ensure_dirs()

    print("Fetching manifests + sites ...")
    pt = list_datasets("pt")
    alt = list_datasets("alt")
    sites = fetch_sites()
    bh_idx, bh_df = build_borehole_index(sites)
    al_idx = build_activelayer_index(sites)

    C.PT_MANIFEST.write_text(json.dumps(pt, ensure_ascii=False, indent=2))
    C.ALT_MANIFEST.write_text(json.dumps(alt, ensure_ascii=False, indent=2))
    C.SITES_JSON.write_text(json.dumps(sites, ensure_ascii=False, indent=2))
    bh_df.to_csv(C.BOREHOLES_CSV, index=False)
    print(f"  PT={len(pt)} ALT={len(alt)} boreholes={len(bh_df)}")

    # 필터 적용
    if variable:
        pt = [d for d in pt if d.get("variable") == variable]
    if open_only:
        pt = [d for d in pt if d.get("policy") == "Open"]
        alt = [d for d in alt if d.get("policy") == "Open"]
    if region == "alaska":
        pt = [d for d in pt if in_alaska(bh_idx.get(d.get("borehole_id")))]
        alt = [d for d in alt if in_alaska(al_idx.get(d.get("activelayer_id")))]
    print(f"  filtered -> PT={len(pt)} ALT={len(alt)} (region={region}, variable={variable}, open_only={open_only})")

    if manifest_only:
        print("manifest-only; stop.")
        return

    pt_meta = {d["id"]: d for d in pt}
    alt_meta = {d["id"]: d for d in alt}
    bulk_download([d["id"] for d in pt], "pt_data", C.PT_CSV_DIR, "PT", pt_meta)
    if include_alt and alt:
        bulk_download([d["id"] for d in alt], "alt_data", C.ALT_CSV_DIR, "ALT", alt_meta)
    print(f"\nDone -> {C.RAW_GTNP}")
