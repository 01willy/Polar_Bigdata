"""v5 연별 강제력 표(재설정 문서 docs/RESET_2026-10-10_KOPRI_FEEDBACK.md 5.1 단계 A2).

ERA5-Land 월평균(data/raw/era5land/nh_monthly_2010-2024.nc: t2m, sd, stl1, 25–84°N, 0.1°)에서 셀·연도별 강제력을 만든다.

변수(연도 y, T_m = 2 m 기온 월평균 °C, S_m = 토양 0–7 cm 온도 월평균 °C, W_m = 적설 수당량 m, d_m = 그 달의 일수)
  e5_maat        y 의 1–12월 T_m 평균
  e5_tdd         Σ_{m∈y} max(T_m, 0)·d_m (융해 도일, °C·d).  e5_sqrt_tdd = √e5_tdd
  e5_twarm       y 의 월 최고 T_m
  e5_fdd         Σ_{m∈Oct(y−1)…Apr(y)} max(−T_m, 0)·d_m (전겨울 동결 도일).  e5_tcold = 그 창의 월 최저 T_m
  e5_stl1        y 의 1–12월 S_m 평균.  e5_tdd_soil = Σ_{m∈y} max(S_m, 0)·d_m.  e5_sqrt_tdd_soil = √
  e5_swe         Oct(y−1)…Sep(y) 12개월 W_m 평균(수문년).  e5_swe_max = 그 창의 월 최댓값
  tdd_cum_m      1월부터 m월 말까지의 누적 TDD(관측 월까지의 도일, 관측일 정합용). tdd_cum_12 = e5_tdd
  winter_partial 1 이면 y = 2010 이라 Oct–Dec 2009 가 없어 겨울 창이 Jan–Apr(2010)·Jan–Sep(2010) 로 줄었다
기존 2015–2020 기후값 열(e5_tdd 등, era5land_covariates.py)과 정의가 같은 부분: tdd·sqrt_tdd·twarm·maat·stl1. 다른 부분: fdd·tcold(달력년 → 전겨울),
swe(12개월 기후값 평균 → 수문년 평균). 비교용 기후값은 연별 값의 다년 평균으로 만들어 정의를 맞춘다.

격자: 최근접 격자가 육지(전 기간 t2m 유한)가 아니면 ±2셀, 다음 ±5셀 안 유클리드(위경도) 최근접 육지 격자(enrich_new_regions.py·a2 규칙).
      e5_grid_lat, e5_grid_lon, e5_fallback_deg(0 = 직접) 를 적는다.

출력(--out-prefix 기준):
  <prefix>_year.csv       셀 × 연도(2010–2024) 행, 위 변수
  <prefix>_clim.csv       셀 행. 2011–2024 연별 값 평균(clim1124_*, 2010 은 겨울 부분이라 제외), 2015–2020 연별 값 평균(clim1520y_*)
  <prefix>_meta.json      규칙, 입력 해시, 격자 폴백 통계, 교차 점검(2015–2020 월 기후값으로 기존 정의 재계산 → static_v5 의 e5clim1520_tdd 와 비교)
실행: python3 scripts/1_data_prep/build_v5_forcing_year.py [--cells data/processed/v5/static_v5.csv] [--out-prefix data/processed/v5/forcing]
      셀 파일은 loc_id|cell_id, lat, lon 열이 있으면 된다. 지도 격자(수십만 셀)도 같은 스크립트로 만든다.
"""
from __future__ import annotations
import argparse, calendar, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2010-2024.nc"
YEARS = list(range(2010, 2025))
VARS = ["e5_maat", "e5_tdd", "e5_sqrt_tdd", "e5_twarm", "e5_fdd", "e5_tcold", "e5_stl1", "e5_tdd_soil", "e5_sqrt_tdd_soil", "e5_swe", "e5_swe_max"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="data/processed/v5/static_v5.csv")
    ap.add_argument("--out-prefix", default="data/processed/v5/forcing")
    ap.add_argument("--chunk", type=int, default=200000)
    a = ap.parse_args()
    t0 = time.time()
    cells = pd.read_csv(ROOT / a.cells, low_memory=False)
    idc = "loc_id" if "loc_id" in cells else "cell_id"
    cells = cells[[idc, "lat", "lon"]].drop_duplicates(idc).reset_index(drop=True)
    print(f"[cells] {len(cells):,} ({idc})", flush=True)

    ds = xr.open_dataset(NC)
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    glat, glon = ds.latitude.values.astype(float), ds.longitude.values.astype(float)
    yrs, mos = ds[tn].dt.year.values, ds[tn].dt.month.values
    assert list(np.unique(yrs)) == YEARS and len(yrs) == 12 * len(YEARS)
    t2m = ds["t2m"].values.astype(np.float32) - 273.15
    land = np.isfinite(t2m).all(axis=0)
    print(f"[nc] t2m {t2m.shape} land {land.mean():.1%} {time.time()-t0:.0f}s", flush=True)

    # 격자 찾기
    def nearest(arr, v):
        return int(np.abs(arr - v).argmin())

    def fallback(iy, ix, la, lo, nb):
        best = None
        for dy in range(-nb, nb + 1):
            for dx in range(-nb, nb + 1):
                y2, x2 = iy + dy, (ix + dx) % len(glon)
                if 0 <= y2 < len(glat) and land[y2, x2]:
                    d2 = (glat[y2] - la) ** 2 + (glon[x2] - lo) ** 2
                    if best is None or d2 < best[0]:
                        best = (d2, y2, x2)
        return best

    n = len(cells)
    IY, IX, FB = np.full(n, -1), np.full(n, -1), np.full(n, np.nan)
    lon_in = cells.lon.values.copy()
    if glon.min() >= 0:                      # 0–360 경도 파일 대비
        lon_in = np.where(lon_in < 0, lon_in + 360, lon_in)
    lat_idx = np.abs(glat[None, :] - cells.lat.values[:, None]).argmin(1)
    lon_idx = np.abs(glon[None, :] - lon_in[:, None]).argmin(1)
    stats = {"direct": 0, "nb2": 0, "nb5": 0, "fail": 0}
    for i in range(n):
        iy, ix = int(lat_idx[i]), int(lon_idx[i])
        if land[iy, ix]:
            IY[i], IX[i], FB[i] = iy, ix, 0.0; stats["direct"] += 1; continue
        done = False
        for nb, key in ((2, "nb2"), (5, "nb5")):
            b = fallback(iy, ix, cells.lat.values[i], lon_in[i], nb)
            if b is not None:
                IY[i], IX[i], FB[i] = b[1], b[2], float(np.sqrt(b[0])); stats[key] += 1; done = True; break
        if not done:
            stats["fail"] += 1
    ok = IY >= 0
    print(f"[grid] {stats} {time.time()-t0:.0f}s", flush=True)

    sd = ds["sd"].values.astype(np.float32)
    stl = ds["stl1"].values.astype(np.float32) - 273.15
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(yrs, mos)], float)
    rows = []
    for s in range(0, n, a.chunk):
        e = min(n, s + a.chunk)
        sel = np.where(ok[s:e])[0] + s
        T = t2m[:, IY[sel], IX[sel]].T.astype(np.float64)      # (cells, 180)
        S = stl[:, IY[sel], IX[sel]].T.astype(np.float64)
        W = sd[:, IY[sel], IX[sel]].T.astype(np.float64)
        for y in YEARS:
            my = yrs == y
            win = ((yrs == y - 1) & (mos >= 10)) | ((yrs == y) & (mos <= 4))       # 전겨울 Oct(y−1)–Apr(y)
            hyd = ((yrs == y - 1) & (mos >= 10)) | ((yrs == y) & (mos <= 9))       # 수문년 Oct(y−1)–Sep(y)
            d = days
            tdd = (np.clip(T[:, my], 0, None) * d[my]).sum(1)
            tdds = (np.clip(S[:, my], 0, None) * d[my]).sum(1)
            fdd = (np.clip(-T[:, win], 0, None) * d[win]).sum(1)
            cum = np.cumsum(np.clip(T[:, my], 0, None) * d[my], axis=1)                 # 월말 누적 TDD(관측일 정합용)
            out = pd.DataFrame({idc: cells[idc].values[sel], "year": y,
                                **{f"tdd_cum_{m}": cum[:, m - 1] for m in range(1, 13)},
                                "e5_maat": T[:, my].mean(1), "e5_tdd": tdd, "e5_sqrt_tdd": np.sqrt(tdd), "e5_twarm": T[:, my].max(1),
                                "e5_fdd": fdd, "e5_tcold": T[:, win].min(1), "e5_stl1": S[:, my].mean(1),
                                "e5_tdd_soil": tdds, "e5_sqrt_tdd_soil": np.sqrt(tdds),
                                "e5_swe": W[:, hyd].mean(1), "e5_swe_max": W[:, hyd].max(1),
                                "winter_partial": int(y == YEARS[0])})
            rows.append(out)
        print(f"[calc] cells {s}–{e} {time.time()-t0:.0f}s", flush=True)
    fy = pd.concat(rows, ignore_index=True)
    grid = pd.DataFrame({idc: cells[idc].values, "e5_grid_lat": np.where(ok, glat[np.clip(IY, 0, None)], np.nan),
                         "e5_grid_lon": np.where(ok, glon[np.clip(IX, 0, None)], np.nan), "e5_fallback_deg": FB})
    fy = fy.merge(grid, on=idc, how="left").sort_values([idc, "year"]).reset_index(drop=True)
    fy.loc[fy.e5_tdd_soil <= 0, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan          # v3 규칙(토양 도일 0 이하는 결측)
    pre = ROOT / a.out_prefix
    pre.parent.mkdir(parents=True, exist_ok=True)
    fy.to_csv(f"{pre}_year.csv", index=False)

    # 기후값(연별 값의 평균)
    def climavg(lo, hi, tag):
        sub = fy[fy.year.between(lo, hi)]
        c = sub.groupby(idc)[VARS].mean()
        c.columns = [f"{tag}_{v[3:]}" for v in c.columns]
        return c
    cl = climavg(2011, 2024, "clim1124").join(climavg(2015, 2020, "clim1520y")).reset_index()
    cl = cl.merge(grid, on=idc, how="left")
    cl.to_csv(f"{pre}_clim.csv", index=False)

    # 교차 점검: 2015–2020 월 기후값에서 기존 정의(era5land_covariates.py)의 tdd 를 다시 계산해 static_v5 의 e5clim1520_tdd 와 비교
    check = {}
    st_path = ROOT / "data" / "processed" / "v5" / "static_v5.csv"
    if idc == "loc_id" and st_path.exists():
        st = pd.read_csv(st_path, usecols=["loc_id", "e5clim1520_tdd", "e5clim1520_maat"]).set_index("loc_id")
        m1520 = (yrs >= 2015) & (yrs <= 2020)
        T6 = t2m[m1520][:, IY[ok], IX[ok]].astype(np.float64)                 # (72, cells)
        mo6 = mos[m1520]
        clim12 = np.stack([T6[mo6 == m].mean(0) for m in range(1, 13)])        # (12, cells)
        d19 = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)], float)
        tdd_old = (np.clip(clim12, 0, None) * d19[:, None]).sum(0)
        maat_old = clim12.mean(0)
        ids = cells[idc].values[ok]
        ref = st.reindex(ids)
        dt = tdd_old - ref.e5clim1520_tdd.values
        dm = maat_old - ref.e5clim1520_maat.values
        fin = np.isfinite(dt)
        check = dict(n=int(fin.sum()), tdd_absdiff_median=float(np.nanmedian(np.abs(dt))), tdd_absdiff_p99=float(np.nanpercentile(np.abs(dt[fin]), 99)),
                     tdd_absdiff_max=float(np.nanmax(np.abs(dt))), maat_absdiff_max=float(np.nanmax(np.abs(dm))),
                     frac_within_0p5=float((np.abs(dt[fin]) <= 0.5).mean()),
                     note="기존 열은 2015–2020 월 기후값(별도 파일 nh_monthly_2015-2020.nc)·최근접 격자(육지 확인 없음) 규칙. 차이는 파일·격자 규칙 차이")
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), nc=str(NC.relative_to(ROOT)), nc_sha256=sha(NC), nc_bytes=NC.stat().st_size,
                cells_file=a.cells, n_cells=int(n), n_ok=int(ok.sum()), grid_stats=stats, years=YEARS, vars=VARS,
                definitions=__doc__.split("변수(")[1].split("기존 2015")[0], crosscheck_vs_static_v5=check,
                outputs={f"{Path(a.out_prefix).name}_year.csv": dict(rows=int(len(fy)), sha256=sha(f"{pre}_year.csv")),
                         f"{Path(a.out_prefix).name}_clim.csv": dict(rows=int(len(cl)), sha256=sha(f"{pre}_clim.csv"))},
                elapsed_s=round(time.time() - t0, 1))
    json.dump(meta, open(f"{pre}_meta.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(grid=stats, check=check, rows=len(fy), elapsed=meta["elapsed_s"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
