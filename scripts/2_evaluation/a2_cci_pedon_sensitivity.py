"""A2-4 CCI 준독립성 민감도(C2-B) — CryoGrid CCI 층서 입력에 현장 페돈이 쓰인 사이트 반경 내 대상 셀을 제외하고 H1을 재채점.

근거: ESA CCI+ Permafrost ATBD v5.0(2024-11-15) §"Ground stratigraphy": 약 700 페돈(6,500 시료)으로 토지피복별 층서를 생성.
사이트 = Siberia(Spasskaya Pad, Lena Delta, Kytalyk, Shalaurovo, Cherskii, Taymyr, Seida), N. America(Tulemalu Lake,
Richardson Mts, Ogilvie Mts, Herschel Island), Scandinavia(Abisko, Tarfala), Greenland(Zackenberg), Svalbard(Adventdalen, Ny-Ålesund).
좌표: CALM 사이트 목록(PANGAEA 972777) 또는 웹(2026-09-21 확인). 산맥·미상 사이트는 '근사/미확인'으로 표기.
H1: ΔRMSE = RMSE(Stefan+CCI 등가중) − RMSE(Stefan), λ=0, 정보 없음 평가 셀, 0.5° 블록 짝지은 부트스트랩 95% CI(400회).
반경 r ∈ {0, 25, 50, 100} km: 어느 사이트든 r 이내인 셀 제외. 제외 셀만의 Δ도 병기(이득이 사이트 근방에 집중하는지).
산출: data/processed/m1/a2_cci_pedon_sensitivity.csv, a2_cci_pedon_sites.csv
실행: python3 scripts/2_evaluation/a2_cci_pedon_sensitivity.py [--radii 25,50,100]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import Base, OUT, paired_boot, rmse, haversine_km, flag_small   # noqa: E402

# (이름, 위도, 경도, 좌표 출처·상태, ATBD 지역군, 감사 목록 포함 여부)
SITES = [
    ("Samoylov (Lena Delta)", 72.369775, 126.480630, "CALM R51", "Siberia", True),
    ("Seida", 67.065560, 62.925080, "CALM R52", "Siberia", True),
    ("Cherskii (NESS)", 68.740, 161.400, "web: Wikipedia Northeast Science Station", "Siberia", True),
    ("Kytalyk (Chokurdakh)", 70.817, 147.483, "web: INTERACT/PAGE21 70°49'N 147°29'E", "Siberia", True),
    ("Shalaurovo (Kolyma)", 69.450, 161.800, "web: 69°27'N 161°48'E", "Siberia", False),
    ("Spasskaya Pad (Yakutsk)", 62.255, 129.618, "근사: 문헌 62°15'N 129°37'E(웹 미확인)", "Siberia", False),
    ("Taymyr (미확인 → CALM R6 Labaz Lake)", 72.383330, 99.500000, "미확인: ATBD 지점 불명, CALM R6 좌표 대체", "Siberia", True),
    ("Zackenberg", 74.473000, -20.552800, "CALM G1", "Greenland", True),
    ("Herschel Island", 69.590, -139.099, "web: 69°35'N 139°06'W", "N. America", True),
    ("Tulemalu Lake", 62.933, -99.400, "web: 62°56'N 99°24'W", "N. America", False),
    ("Richardson Mountains (중심 근사)", 67.5, -136.0, "근사: 산맥 중심(범위 ~100 km)", "N. America", False),
    ("Ogilvie Mountains (중심 근사)", 64.5, -138.5, "근사: 산맥 중심(범위 ~100 km)", "N. America", False),
    ("Abisko", 68.300000, 18.833330, "CALM S2", "Scandinavia", False),
    ("Tarfala", 67.91147, 18.61233, "web: SITES 기상관측소", "Scandinavia", False),
    ("Adventdalen", 78.202, 15.831, "web: SIOS 기상관측소", "Svalbard", False),
    ("Ny-Ålesund (Bayelva)", 78.921, 11.831, "근사: Bayelva 78°55'N 11°50'E", "Svalbard", False),
]

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
ap.add_argument("--radii", default="25,50,100")
ap.add_argument("--nboot", type=int, default=400)
ap.add_argument("--audit-only", action="store_true", help="감사가 열거한 사이트만 사용(기본: ATBD 전체 목록)")
args = ap.parse_args()
t0 = time.time()
base = Base(sources=tuple(args.sources.split(",")))
print(base.describe(), flush=True)
radii = [float(r) for r in args.radii.split(",")]
sites = pd.DataFrame(SITES, columns=["site", "lat", "lon", "coord_source", "atbd_group", "in_audit_list"])
if args.audit_only:
    sites = sites[sites.in_audit_list]

# 사이트별 반경 내 대상 셀 수(전 대상 지역, 평가 셀)
site_rows = []
for _, s in sites.iterrows():
    row = s.to_dict()
    for tg in base.regions:
        te = base.te(tg)
        d = haversine_km(te.lat.values, te.lon.values, s.lat, s.lon)
        for r in radii:
            k = int((d <= r).sum())
            if k:
                row[f"{tg}_n_within_{int(r)}km"] = k
        row[f"{tg}_nearest_km"] = round(float(d.min()), 1) if len(d) else np.nan
    site_rows.append(row)
sites_out = pd.DataFrame(site_rows)
sites_out.to_csv(OUT / "a2_cci_pedon_sites.csv", index=False)


def h1(y, S, SC, blocks):
    d0, lo, hi, n, nb = paired_boot(y, SC, S, blocks, nboot=args.nboot)
    return dict(delta=round(d0, 3), ci_lo=round(lo, 3) if np.isfinite(lo) else np.nan,
                ci_hi=round(hi, 3) if np.isfinite(hi) else np.nan, n=n, n_blocks=nb)


rows = []
for tg in base.regions:
    te = base.te(tg)
    y, blocks = base.y(tg), base.blocks(tg)
    S, C, Cc = base.anchor("stefan", tg), base.anchor("cci_raw", tg), base.anchor("cci_cal", tg)
    SC, SCc = 0.5 * (S + C), 0.5 * (S + Cc)
    D = np.column_stack([haversine_km(te.lat.values, te.lon.values, s.lat, s.lon) for _, s in sites.iterrows()])
    dmin = D.min(1); nearest = sites.site.values[D.argmin(1)]
    near_site = pd.Series(nearest).value_counts().index[0]
    for r in [0.0] + radii:
        keep = dmin > r
        excl = ~keep
        base_row = dict(target=tg, radius_km=int(r), n_all=len(y), n_excluded=int(excl.sum()), frac_excluded=round(float(excl.mean()), 3),
                        n_kept=int(keep.sum()), n_blocks_kept=int(len(np.unique(blocks[keep]))) if keep.any() else 0,
                        nearest_site_mode=near_site, min_dist_km=round(float(dmin.min()), 1), median_dist_km=round(float(np.median(dmin)), 1),
                        all_cells_within=bool(excl.all()))
        if keep.any():
            hk = h1(y[keep], S[keep], SC[keep], blocks[keep])
            hkc = h1(y[keep], S[keep], SCc[keep], blocks[keep])
            base_row.update(rmse_S_kept=round(rmse(y[keep], S[keep]), 3), rmse_SC_kept=round(rmse(y[keep], SC[keep]), 3),
                            h1_delta_kept=hk["delta"], h1_ci_lo_kept=hk["ci_lo"], h1_ci_hi_kept=hk["ci_hi"],
                            h1cal_delta_kept=hkc["delta"], h1cal_ci_lo_kept=hkc["ci_lo"], h1cal_ci_hi_kept=hkc["ci_hi"])
        if excl.any():
            he = h1(y[excl], S[excl], SC[excl], blocks[excl])
            base_row.update(n_blocks_excluded=he["n_blocks"], rmse_S_excl=round(rmse(y[excl], S[excl]), 3),
                            rmse_SC_excl=round(rmse(y[excl], SC[excl]), 3), h1_delta_excl=he["delta"],
                            h1_ci_lo_excl=he["ci_lo"], h1_ci_hi_excl=he["ci_hi"])
        base_row["flag"] = flag_small(int(keep.sum()))
        rows.append(base_row)
    print(f"  [{tg}] n={len(y)} 최근접 사이트 {near_site} min={dmin.min():.0f} km · 제외 비율 " +
          " ".join(f"{int(r)}km:{(dmin <= r).mean():.2f}" for r in radii) + f" · {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
tag = "_auditonly" if args.audit_only else ""
out.to_csv(OUT / f"a2_cci_pedon_sensitivity{tag}.csv", index=False)
(OUT / f"a2_cci_pedon_sensitivity{tag}_meta.json").write_text(json.dumps(dict(
    sources=args.sources, radii=radii, nboot=args.nboot, audit_only=args.audit_only, n_sites=int(len(sites)),
    sites=sites.site.tolist(), reference="ESA CCI+ Permafrost ATBD v5.0 (2024-11-15), ground stratigraphy section (approx. 700 pedons)",
    regions=base.regions, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))
print("\n=== A2-4 페돈 사이트 반경 제외 후 H1 (Stefan+CCI − Stefan, λ=0, cm; 음수 = 결합 우세) ===")
cols = ["target", "radius_km", "n_all", "n_excluded", "frac_excluded", "n_kept", "n_blocks_kept", "flag", "rmse_S_kept", "rmse_SC_kept",
        "h1_delta_kept", "h1_ci_lo_kept", "h1_ci_hi_kept", "h1cal_delta_kept", "h1_delta_excl", "h1_ci_lo_excl", "h1_ci_hi_excl",
        "nearest_site_mode", "min_dist_km", "all_cells_within"]
print(out[[c for c in cols if c in out]].to_string(index=False))
print("\n사이트별 반경 내 셀 수:")
print(sites_out[[c for c in sites_out.columns if c in ("site", "coord_source") or "within" in c]].fillna("").to_string(index=False))
print(f"\n저장: {OUT/('a2_cci_pedon_sensitivity'+tag+'.csv')} · {time.time()-t0:.0f}s")
