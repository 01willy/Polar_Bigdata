"""Track C1 CCI 다층 앵커 결합(정보 없음 조건, AB4 CI·MAIN6 점 추정). 기존 M1 예측 재사용 + CCI 층 추출, CPU.

계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 C1, §5 F9, §3 채점 규약. 공용 통계는 src/polar/h4_common.py.

1단계(추출)
  data/raw/cci_alt·cci_pfr·cci_gtd 의 CRDPv4 NetCDF(0.01°, 6000×36000, 위도 25–85°N)에서 fidelity_base_v3 전 셀의
  최근접 격자 값을 연도별로 읽어 다년 평균을 만든다(enrich_cci_cell.py 최근접 방식, 1000×1000 청크 단위 읽기).
  층: ALT(m→cm, 대조용 재추출)·ALT_uncertainty(cm)·PFR(%)·PFR_uncertainty(%)·GTD T1m·T2m(°C).
  사용 연도(전량 실행 기준): ALT 1997–2021(25개), PFR 1997–2021(25개), GTD 2003–2021(19개, CEDA 공개본 기준 다운로드 범위).
  실제 읽은 파일·연도·실패 파일은 <layer_csv>.meta.json 과 <tag>_meta.json 의 extract 항목에 기록한다.
  블록 평활 층: 셀 블록(0.5°, fidelity.add_group_keys 의 floor(lat/0.5)·floor(lon/0.5) 격자)마다 CCI ALT 래스터의
  0.5° × 0.5° 창(50 × 50 화소)을 연도별로 읽어 화소별 다년 평균 래스터를 만들고, 창 안 유효 화소의 중앙값을 블록 값으로 쓴다.
  관측 셀 값은 쓰지 않으므로 라벨·표본 배치와 무관하다. 이전 정의(관측 셀 cci_alt 의 블록 중앙값)는 진단 열로만 남긴다.
  산출 data/processed/h4/cci_layers_cells.csv (스모크는 <tag>_cci_layers_cells.csv, 연도 뒤쪽 2개만).

2단계(채점)
  채점 셀: M1 npz 의 eval::noinfo|<t>|0 (y, loc_id, block) 를 그대로 쓴다. 새 셀 선택·필터는 없다.
  앵커(모든 결합은 같은 셀·같은 stefan 을 공유):
    stefan          = M1 npz anchor::noinfo|<t>|0|alaska|stefan (E0 = 알래스카 원천 최소제곱, 재계산과 일치 확인)
    stefan_cci      = 0.5(st + cci_alt)                                              (i) 현행 등가중
    cci_uncw        = (st/σ_st² + cci/u²)/(1/σ_st² + 1/u²)                           (ii) 불확실성 역가중
    cci_blocksmooth = 0.5(st + cci_alt_blockmed)                                     (iii) 블록 평활(래스터 0.5° 창 중앙값)
    gate_gtd_pfr    = PFR < 50 % 또는 GTD 1 m > −1 °C 이면 st, 아니면 0.5(st + cci)   (iv) 게이트
  불확실성 역가중의 두 분산 정의(주 분석)
    u²     = (CCI ALT_uncertainty 다년 평균, cm)², 하한 u ≥ --unc-floor-cm. 제품 정의상 이 층은 "화소 안 공간 변동성"(sub-pixel
             spatial variability)이며 현장 관측 대비 오차가 아니다. u 결측 셀은 등가중으로 대체한다.
    σ_st²  = 알래스카 원천 Stefan 잔차 평균제곱 mean((y − E0·√TDD)²) (원천 표본 안 적합 잔차, 대상 지역 라벨 미사용).
  민감도(주 판정과 별도 가족)
    cci_uncw_srcvar: σ_st² 를 σ²_src(t) = 대상 t 를 뺀 원천 macro 지역별 Stefan 잔차 평균제곱 MSE_r(y − E0·√TDD)의 비가중 평균으로
                     바꾼다(지역 셀 ≥ 3, 후보 = 알래스카·MAIN6, 외부 홀드아웃 미포함). 알래스카 표본 안 잔차는 전이 오차를 과소평가하므로
                     전이 상황의 Stefan 오차 규모를 반영하는 대안이다. 다른 대상 지역 라벨을 초모수 하나에 쓰므로 주 판정에는 넣지 않는다.
  검정: h4_common.boot_delta_blocks (대상별 분할 1개, 채점 블록 부트스트랩, 대상별 seed_of('c1', t) 로 모든 검정이 같은 재표집을 공유).
    대상별 행: AB4 대상은 CI 산출, 러시아 C·그린란드는 점 추정만(CI 열 NaN, ci_scope='point_only').
    MEAN[AB4] 행: AB4 대상 분포를 같은 부트스트랩 번호끼리 평균(지역 비가중)한 분포로 CI·p. MEAN[MAIN6] 행: 점 추정만.
    세 채점 중 셀 가중(delta)과 블록 등가중(delta_beq)을 병기한다. 단일 분할·단일 반복이므로 ci_rep 는 정의되지 않는다(NaN).
  Holm 가족(MEAN[AB4] 수준, 대상별 행은 대상 안 같은 가족으로 보조 보정)
    F9     = {stefan_cci, cci_uncw, cci_blocksmooth, gate_gtd_pfr} − stefan (4검정)
    vs_eq  = {cci_uncw, cci_blocksmooth, gate_gtd_pfr} − stefan_cci (3검정, 별도 가족)
    sens   = {cci_uncw_srcvar − stefan, cci_uncw_srcvar − stefan_cci} (2검정, 별도 가족)
  Holm 조정 CI: p 오름차순 순위 i 의 검정은, 단계 하강이 i 까지 기각을 이어 가면 수준 α/(m − i + 1) 의 구간,
    단계 하강이 멈춘 뒤의 검정은 수준 α/(m − |기각|) 의 구간을 쓴다(Strassburger·Bretz 2008 의 Holm 양립 구간 방식).
    구간은 부트스트랩 p 를 역산한 순서통계량 구간(os_ci)이므로 0 배제 여부가 p_holm < α 와 정확히 일치한다.
    일치 여부를 holm_consistent 열에 기록한다(불일치 0 이 정상). 보정 전 ci_lo·ci_hi 는 h4_common 과 같은 2.5·97.5 백분위다.
  F9 판정(Holm 가족은 두 판정 모두 결합 − stefan 4검정으로 같다. 구성원 집합만 다르다)
    F9_pass        (주 판정, 계획 §4 C1 문구): (ii) cci_uncw · (iii) cci_blocksmooth · (iv) gate_gtd_pfr 중 하나라도
                   MEAN[AB4] 에서 delta < 0 · p_holm < 0.05 · ci_hi_holm < 0 이면 통과.
    F9_pass_strict (계획 §5 F9 표 문구): (ii) cci_uncw · (iv) gate_gtd_pfr 만 구성원으로 두고 같은 기준을 적용한다.
    계획서 두 절의 구성원 정의가 다르므로 두 판정을 모두 기록한다. 두 판정이 갈리는 경우는 (iii) 블록 평활만 기준을
    만족할 때이며, 이때 F9_pass_driver 에 blocksmooth_only 로 표시한다. 원고의 주 판정은 F9_pass(§4) 로 하고 §5 판정을 병기한다.
  산출 data/processed/h4/<tag>_cci_layers.csv(판정 표), <tag>_tests.csv(전 행), <tag>_worst.csv, <tag>_cells.csv,
       <tag>_blocksse.npz(h4_common 형식, 키 = (rule,)), <tag>_meta.json.
실행(ROOT, CPU): python3 scripts/2_evaluation/h36_cci_layers.py --tag c1 --workers 4 [--smoke] [--skip-extract]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import glob
import json
import multiprocessing
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, BLOCK_DEG                                                          # noqa: E402
from polar.m1_core import load_base, fit_coefs, train_region_set                                     # noqa: E402
from polar.h4_common import BlockStore, save_stores, boot_delta_blocks, boot_p, seed_of              # noqa: E402

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
RAW = ROOT / "data" / "raw"
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]; AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SRC_CAND = ["Alaska"] + MAIN6                    # 민감도 σ²_src 후보(외부 홀드아웃 미포함)
ALPHA = 0.05
CHUNK = 1000
LAYERS = {"alt": (RAW / "cci_alt", {"ALT": ("cci_alt_files", 100.0), "ALT_uncertainty": ("cci_alt_unc", 100.0)}),
          "pfr": (RAW / "cci_pfr", {"PFR": ("cci_pfr", 1.0), "PFR_uncertainty": ("cci_pfr_unc", 1.0)}),
          "gtd": (RAW / "cci_gtd", None)}                     # GTD 변수명은 파일에서 탐지(T1m·T2m)
RULES_MAIN = ["stefan", "stefan_cci", "cci_uncw", "cci_blocksmooth", "gate_gtd_pfr"]
RULES_SENS = ["cci_uncw_srcvar"]
FAMILIES = {
    "F9": [(a, "stefan") for a in ("stefan_cci", "cci_uncw", "cci_blocksmooth", "gate_gtd_pfr")],
    "vs_eq": [(a, "stefan_cci") for a in ("cci_uncw", "cci_blocksmooth", "gate_gtd_pfr")],
    "sens": [("cci_uncw_srcvar", "stefan"), ("cci_uncw_srcvar", "stefan_cci")],
}
F9_MEMBERS = ("cci_uncw", "cci_blocksmooth", "gate_gtd_pfr")          # 주 판정(계획 §4 C1: (ii)–(iv))
F9_MEMBERS_STRICT = ("cci_uncw", "gate_gtd_pfr")                        # 병기 판정(계획 §5 F9: (ii)·(iv))


def file_year(path) -> int | None:
    m = re.search(r"_PP-(\d{4})-fv", Path(path).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 1단계: 추출
def nearest_idx(coords, vals):
    order = np.argsort(coords); cs = coords[order]
    j = np.clip(np.searchsorted(cs, vals), 1, len(cs) - 1)
    j = np.where(np.abs(vals - cs[j - 1]) <= np.abs(vals - cs[j]), j - 1, j)
    return order[j]


def gtd_varmap(ds) -> dict:
    """GTD 파일의 변수 → 출력 열. depth 차원이 있으면 (변수, depth index), 없으면 이름(T1m·T2m)에서 깊이를 읽는다."""
    out = {}
    for v in ds.variables:
        var = ds.variables[v]
        dims = var.dimensions
        if "lat" not in dims or "lon" not in dims or v.endswith("_uncertainty"):
            continue
        ddim = [d for d in dims if d not in ("time", "lat", "lon")]
        if ddim:
            dep = np.asarray(ds.variables[ddim[0]][:], float)
            for want in (1.0, 2.0):
                k = int(np.argmin(np.abs(dep - want)))
                if abs(dep[k] - want) < 0.26:
                    out[f"cci_gtd_{int(want)}m"] = (v, ddim[0], k)
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*m", v.lower()) or re.search(r"_(\d+)$", v)
            if m and float(m.group(1)) in (1.0, 2.0):
                out[f"cci_gtd_{int(float(m.group(1)))}m"] = (v, None, None)
    return out


def _read(var, dims, rows: slice, cols: slice, dd=None, k=None):
    idx = []
    for d in dims:
        if d == "lat": idx.append(rows)
        elif d == "lon": idx.append(cols)
        elif d == dd: idx.append(k)
        else: idx.append(0)
    return np.ma.filled(var[tuple(idx)].astype(np.float32), np.nan)


def extract_file(path: str, iy: np.ndarray, ix: np.ndarray, varmap: dict | None, windows: list | None = None):
    """한 파일에서 셀 최근접 값 추출(청크 단위 읽기). windows 가 있으면 ALT 의 블록 창(r0, r1, c0, c1) 배열도 반환.

    반환 (path, {열: 값(n_cells)}, {열: 변수명}, [창 배열(cm)] 또는 None)
    """
    import netCDF4
    ds = netCDF4.Dataset(path)
    ds.set_auto_maskandscale(True)
    if varmap is None:
        vm = {col: (v, dd, k, 1.0) for col, (v, dd, k) in gtd_varmap(ds).items()}
    else:
        vm = {col: (v, None, None, sc) for v, (col, sc) in varmap.items() if v in ds.variables}
    res = {}
    cy, cx = iy // CHUNK, ix // CHUNK
    keys = np.unique(cy * 1000 + cx)
    for col, (v, dd, k, sc) in vm.items():
        var = ds.variables[v]; dims = var.dimensions
        out = np.full(len(iy), np.nan, np.float32)
        for key in keys:
            ky, kx = key // 1000, key % 1000
            sel = (cy == ky) & (cx == kx)
            y0, x0 = ky * CHUNK, kx * CHUNK
            slab = _read(var, dims, slice(y0, y0 + CHUNK), slice(x0, x0 + CHUNK), dd, k)
            out[sel] = slab[iy[sel] - y0, ix[sel] - x0]
        res[col] = out * sc
    win = None
    if windows is not None and "ALT" in ds.variables:
        var = ds.variables["ALT"]; dims = var.dimensions
        win = [_read(var, dims, slice(r0, r1), slice(c0, c1)) * 100.0 for (r0, r1, c0, c1) in windows]
    ds.close()
    return path, res, {col: t[0] for col, t in vm.items()}, win


def block_windows(blocks: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """0.5° 블록 id(floor(lat/0.5)·100000 + floor(lon/0.5)) → 래스터 창 (r0, r1, c0, c1). lat·lon 은 오름차순 화소 중심."""
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0), "좌표가 오름차순이 아님"
    out = []
    for b in blocks:
        ilat = int(round(b / 100000)); ilon = int(b - ilat * 100000)
        la0, lo0 = ilat * BLOCK_DEG, ilon * BLOCK_DEG
        r0, r1 = np.searchsorted(lat, la0), np.searchsorted(lat, la0 + BLOCK_DEG)
        c0, c1 = np.searchsorted(lon, lo0), np.searchsorted(lon, lo0 + BLOCK_DEG)
        out.append((int(r0), int(r1), int(c0), int(c1)))
    return out


def extract_layers(cells: pd.DataFrame, years_limit: int | None, workers: int, log):
    import netCDF4
    files = []
    for lay, (d, vm) in LAYERS.items():
        fs = sorted(glob.glob(str(d / "*.nc")), key=lambda f: (file_year(f) or 0, f))
        if years_limit:
            fs = fs[-years_limit:]
        files += [(lay, f, vm) for f in fs]
        log(f"[extract] {lay}: 파일 {len(fs)}개 ({d}), 연도 {[file_year(f) for f in fs]}")
    if not files:
        raise SystemExit("CCI NetCDF 없음")
    ds = netCDF4.Dataset(files[0][1]); lat = ds.variables["lat"][:].astype(float); lon = ds.variables["lon"][:].astype(float); ds.close()
    iy, ix = nearest_idx(lat, cells.lat.values), nearest_idx(lon, cells.lon.values)

    # 블록 창(0.5°) 정의 확인: 셀 좌표 floor 와 블록 id 가 일치해야 한다
    ub = np.unique(cells.block.values.astype(np.int64))
    chk = (np.floor(cells.lat / BLOCK_DEG).astype(int) * 100000 + np.floor(cells.lon / BLOCK_DEG).astype(int)).values
    assert np.array_equal(chk, cells.block.values.astype(np.int64)), "블록 id 가 0.5° floor 격자와 불일치"
    wins = block_windows(ub, lat, lon)
    wshape = [(r1 - r0, c1 - c0) for r0, r1, c0, c1 in wins]
    log(f"[extract] 블록 창 {len(ub)}개, 창 크기(행×열) 분포 {pd.Series([f'{a}x{b}' for a, b in wshape]).value_counts().to_dict()}")
    empty = [int(b) for b, (a, c) in zip(ub, wshape) if a * c == 0]
    if empty:
        log(f"[extract] 래스터 범위(25–85°N) 밖 블록 {empty}: 블록 평활 결측 → 셀 cci 대체")
    wsum = [np.zeros(s) for s in wshape]; wcnt = [np.zeros(s, int) for s in wshape]

    acc, cnt, used, failed, varnames = {}, {}, {}, [], {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(extract_file, f, iy, ix, vm, wins if lay == "alt" else None): (lay, f) for lay, f, vm in files}
        for fu in as_completed(futs):
            lay, f = futs[fu]
            try:
                path, res, vn, win = fu.result()
            except Exception as e:                            # 불완전 파일 등
                failed.append(dict(layer=lay, file=Path(f).name, error=f"{type(e).__name__}: {e}"))
                log(f"[warn] {lay} {Path(f).name}: 읽기 실패 → 제외 ({type(e).__name__}: {e})"); continue
            used.setdefault(lay, []).append(Path(path).name)
            for col, v in res.items():
                if col not in acc:
                    acc[col] = np.zeros(len(cells)); cnt[col] = np.zeros(len(cells), int)
                ok = np.isfinite(v); acc[col][ok] += v[ok]; cnt[col][ok] += 1
                varnames[col] = vn[col]
            if win is not None:
                for j, a in enumerate(win):
                    ok = np.isfinite(a); wsum[j][ok] += a[ok]; wcnt[j][ok] += 1
            log(f"[extract] {lay} {Path(path).name}: {', '.join(f'{c} 유효 {np.isfinite(v).sum()}' for c, v in res.items())}")
    tab = cells[["loc_id", "lat", "lon", "region", "block", "cci_alt"]].copy()
    for col in acc:
        tab[col] = np.where(cnt[col] > 0, acc[col] / np.maximum(cnt[col], 1), np.nan).round(3)
        tab[f"n_years_{col}"] = cnt[col]
    # 블록 평활: 화소별 다년 평균 래스터 → 창 안 유효 화소 중앙값
    bmed, bnpix = {}, {}
    for b, s, c in zip(ub, wsum, wcnt):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        v = mean[np.isfinite(mean)]
        bmed[int(b)] = float(np.median(v)) if len(v) else np.nan; bnpix[int(b)] = int(len(v))
    tab["cci_alt_blockmed"] = tab.block.astype(np.int64).map(bmed).round(3)
    tab["n_pix_blockmed"] = tab.block.astype(np.int64).map(bnpix)
    tab["cci_alt_blockmed_cells"] = tab.groupby("block")["cci_alt"].transform("median")      # 이전 정의(진단 전용)
    alt_years = sorted(file_year(f) for f in used.get("alt", []))
    emeta = dict(files={lay: sorted(v, key=lambda f: (file_year(f) or 0, f)) for lay, v in used.items()},
                 years={lay: sorted(file_year(f) for f in v) for lay, v in used.items()},
                 failed=failed, varnames=varnames, years_limit=years_limit,
                 blocksmooth=dict(definition="CCI ALT 래스터 0.5° 창 화소별 다년 평균의 유효 화소 중앙값(cm)", years=alt_years,
                                  n_blocks=int(len(ub)), n_blocks_valid=int(np.isfinite(list(bmed.values())).sum()), blocks_outside=empty,
                                  window_shapes=pd.Series([f"{a}x{b}" for a, b in wshape]).value_counts().to_dict(),
                                  npix_median=float(np.median(list(bnpix.values())))))
    return tab, emeta


# ---------------------------------------------------------------- 2단계: 채점
def combos(st, cci, unc, cci_blk, pfr, gtd, sigma_st2, unc_floor_cm, sigma_src2=None):
    """앵커별 예측(각 (n,)). 결측 규칙: u 결측 → 등가중, 블록값 결측 → 셀 cci, 게이트 층 결측 → 해당 조건 미적용."""
    eq = 0.5 * (st + cci)
    u = np.where(np.isfinite(unc), np.maximum(unc, unc_floor_cm), np.nan)

    def uncw(s2):
        w_cci, w_st = 1.0 / u ** 2, 1.0 / s2
        return np.where(np.isfinite(u), (w_st * st + w_cci * cci) / (w_st + w_cci), eq)
    blk = 0.5 * (st + np.where(np.isfinite(cci_blk), cci_blk, cci))
    gate_pfr = np.where(np.isfinite(pfr), pfr < 50.0, False)
    gate_gtd = np.where(np.isfinite(gtd), gtd > -1.0, False)
    gate = gate_pfr | gate_gtd
    R = {"stefan": st, "stefan_cci": eq, "cci_uncw": uncw(sigma_st2), "cci_blocksmooth": blk, "gate_gtd_pfr": np.where(gate, st, eq)}
    if sigma_src2 is not None:
        R["cci_uncw_srcvar"] = uncw(sigma_src2)
    return R, gate


def pct_ci(dist, level_alpha):
    d = np.asarray(dist, float); d = d[np.isfinite(d)]
    if not len(d):
        return (np.nan, np.nan)
    return (float(np.percentile(d, 100 * level_alpha / 2)), float(np.percentile(d, 100 * (1 - level_alpha / 2))))


def os_ci(dist, level_alpha):
    """p 역산과 정확히 일치하는 순서통계량 구간. 각 꼬리에 허용하는 표본 수 k = ceil(a·N/2) − 1 일 때
    [정렬값[k], 정렬값[N − 1 − k]]. 상한 < 0 ⇔ #(Δ* ≥ 0) < a·N/2, 하한 > 0 ⇔ #(Δ* ≤ 0) < a·N/2 이므로
    boot_p(= 2·min 꼬리 비율) < a 와 구간의 0 배제가 같다(선형 보간 백분위는 경계에서 어긋날 수 있다)."""
    d = np.sort(np.asarray(dist, float)[np.isfinite(np.asarray(dist, float))])
    n = len(d)
    if not n:
        return (np.nan, np.nan)
    k = max(int(np.ceil(level_alpha * n / 2.0)) - 1, 0)
    return (float(d[k]), float(d[n - 1 - k]))


def holm_block(rows: list[dict], dists: list, alpha=ALPHA):
    """한 가족(같은 대상 수준)의 행들에 Holm p·순위·조정 CI·기각·일치 여부를 채운다(in place)."""
    p = np.array([r["p_boot"] for r in rows], float)
    ok = np.where(np.isfinite(p))[0]
    for r in rows:
        r.update(p_holm=np.nan, holm_rank=np.nan, holm_level=np.nan, ci_lo_holm=np.nan, ci_hi_holm=np.nan,
                 reject_holm=False, holm_consistent=np.nan)
    if not len(ok):
        return
    order = ok[np.argsort(p[ok], kind="stable")]; m = len(order)
    run, stopped, n_rej = 0.0, False, 0
    padj = {}
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); padj[j] = min(1.0, run)
        if not stopped and p[j] < alpha / (m - i):
            n_rej += 1
        else:
            stopped = True
    for i, j in enumerate(order):
        lev = alpha / (m - i) if i < n_rej else alpha / (m - n_rej)
        lo, hi = os_ci(dists[j], lev)
        rej = bool(padj[j] < alpha)
        rows[j].update(p_holm=padj[j], holm_rank=i + 1, holm_level=lev, ci_lo_holm=lo, ci_hi_holm=hi, reject_holm=rej,
                       holm_consistent=bool(rej == bool(np.isfinite(lo) and (lo > 0 or hi < 0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--targets", default="")
    ap.add_argument("--skip-extract", action="store_true", help="기존 cci_layers_cells.csv(+ .meta.json) 재사용")
    ap.add_argument("--unc-floor-cm", type=float, default=1.0, help="불확실성 역가중의 u 하한(cm)")
    ap.add_argument("--src-min-cells", type=int, default=3, help="민감도 σ²_src 의 원천 지역 최소 셀 수")
    args = ap.parse_args()
    t0 = time.time()
    tag = f"{args.tag}_smoke" if args.smoke else args.tag
    workers = min(args.workers, 2) if args.smoke else args.workers
    nboot = min(args.nboot, 200) if args.smoke else args.nboot
    targets = [t for t in args.targets.split(",") if t] or MAIN6
    logf = open(OUT / f"{tag}_log.txt", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"[start] {tag} workers={workers} nboot={nboot} targets={targets}")

    # 1단계
    layer_csv = OUT / ("cci_layers_cells.csv" if not args.smoke else f"{tag}_cci_layers_cells.csv")
    layer_meta = layer_csv.with_suffix(".meta.json")
    base_all = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False, usecols=["loc_id", "lat", "lon", "region", "block", "cci_alt"])
    cells = base_all.drop_duplicates("loc_id").reset_index(drop=True)
    if args.skip_extract and layer_csv.exists() and layer_meta.exists():
        tab = pd.read_csv(layer_csv); emeta = json.loads(layer_meta.read_text()); log(f"[extract] 재사용 {layer_csv} ({len(tab)} 행)")
        assert "cci_alt_blockmed" in tab, "재사용 층 파일에 래스터 블록 중앙값 열이 없다(재추출 필요)"
    else:
        tab, emeta = extract_layers(cells, 2 if args.smoke else None, workers, log)
        tab.to_csv(layer_csv, index=False); layer_meta.write_text(json.dumps(emeta, ensure_ascii=False, indent=1, default=float))
        log(f"[saved] {layer_csv} {tab.shape}")
    if "cci_alt_files" in tab:
        m = np.isfinite(tab.cci_alt_files) & np.isfinite(tab.cci_alt)
        log(f"[check] 파일 재추출 ALT 대 base cci_alt: n={int(m.sum())} r={np.corrcoef(tab.cci_alt_files[m], tab.cci_alt[m])[0, 1]:.4f} "
            f"|diff| 중앙 {np.nanmedian(np.abs(tab.cci_alt_files[m] - tab.cci_alt[m])):.2f} cm (base = 25년 평균, 스모크는 2년)")
    mb = np.isfinite(tab.cci_alt_blockmed) & np.isfinite(tab.cci_alt_blockmed_cells)
    log(f"[check] 블록 평활 래스터 중앙값 대 관측 셀 중앙값(이전 정의): 블록 {tab[mb].block.nunique()} "
        f"|diff| 중앙 {np.nanmedian(np.abs(tab.cci_alt_blockmed - tab.cci_alt_blockmed_cells)):.2f} cm, 창 유효 화소 중앙 {np.nanmedian(tab.n_pix_blockmed):.0f}")
    L = tab.set_index("loc_id")
    for c in ("cci_alt_unc", "cci_pfr", "cci_gtd_1m", "cci_gtd_2m", "cci_alt_blockmed"):
        if c not in L:
            L[c] = np.nan; log(f"[warn] 층 {c} 없음 → 결측 규칙 적용")

    # 알래스카 원천 Stefan 잔차 분산(E0 재계산, npz 앵커와 일치 확인)
    df = load_base(PROC)
    ak = df.iloc[train_region_set(df, "Lena", "alaska")]
    k = fit_coefs(ak)
    E0 = float(k["E"])
    s = ak.e5_sqrt_tdd.values.astype(float); y = ak[TARGET].values.astype(float)
    mm = np.isfinite(y) & np.isfinite(s) & (s > 0)
    sigma_st2 = float(np.mean((y[mm] - E0 * s[mm]) ** 2))
    log(f"[E0] 알래스카 원천 n={int(mm.sum())} E0={E0:.4f} 잔차 RMSE={np.sqrt(sigma_st2):.2f} cm (σ_st²={sigma_st2:.1f})")
    # 민감도: 원천 macro 지역별 Stefan 잔차 평균제곱(E0 고정)
    reg_mse = {}
    for r, g in df.groupby("macro"):
        if r not in SRC_CAND:
            continue
        yy, ss = g[TARGET].values.astype(float), g.e5_sqrt_tdd.values.astype(float)
        ok = np.isfinite(yy) & np.isfinite(ss) & (ss > 0)
        if ok.sum() >= args.src_min_cells:
            reg_mse[r] = dict(n=int(ok.sum()), mse=float(np.mean((yy[ok] - E0 * ss[ok]) ** 2)))
    log("[σ²_src] 지역별 Stefan 잔차 RMSE(E0 고정): " + ", ".join(f"{r} {np.sqrt(v['mse']):.1f} cm (n={v['n']})" for r, v in reg_mse.items()))
    dfi = df.set_index("loc_id")

    # M1 npz
    P, EV = {}, {}
    npz_files = sorted(glob.glob(str(PROC / "m1" / "m1_model_shard*_preds.npz")))
    for f in npz_files:
        z = np.load(f, allow_pickle=True)
        for kk in z.files:
            if kk.startswith("anchor::noinfo"):
                P[kk] = z[kk]
            elif kk.startswith("eval::noinfo"):
                _, tk, fld = kk.split("::"); EV.setdefault(tk, {})[fld] = z[kk]
    rows_cells, worst, stores, gate_info, sigma_src = [], [], {}, {}, {}
    for tg in targets:
        tk = f"noinfo|{tg}|0"
        if tk not in EV or f"anchor::{tk}|alaska|stefan" not in P:
            log(f"[skip] {tg}: npz 없음"); continue
        yv, loc, blk = EV[tk]["y"].astype(float), EV[tk]["loc_id"], EV[tk]["block"]
        st = P[f"anchor::{tk}|alaska|stefan"].astype(float)
        st_re = E0 * dfi.loc[loc, "e5_sqrt_tdd"].values.astype(float)
        cci = dfi.loc[loc, "cci_alt"].values.astype(float)
        log(f"[check] {tg}: npz stefan 대 재계산 E0·√TDD 최대차 {np.nanmax(np.abs(st - st_re)):.3f} cm; 채점 셀 n={len(yv)}, 블록 {len(np.unique(blk))}")
        if f"anchor::{tk}|alaska|stefan_cci" in P:
            log(f"[check] {tg}: npz stefan_cci 대 0.5(st+cci) 최대차 {np.nanmax(np.abs(P[f'anchor::{tk}|alaska|stefan_cci'] - 0.5 * (st + cci))):.3f} cm")
        src = {r: v for r, v in reg_mse.items() if r != tg}
        sigma_src[tg] = dict(value=float(np.mean([v["mse"] for v in src.values()])), regions=sorted(src))
        Lt = L.reindex(loc)
        R, gate = combos(st, cci, Lt.cci_alt_unc.values.astype(float), Lt.cci_alt_blockmed.values.astype(float),
                         Lt.cci_pfr.values.astype(float), Lt.cci_gtd_1m.values.astype(float), sigma_st2, args.unc_floor_cm,
                         sigma_src2=sigma_src[tg]["value"])
        uv = Lt.cci_alt_unc.values.astype(float)
        gate_info[tg] = dict(n=int(len(yv)), n_blocks=int(len(np.unique(blk))), gated_frac=float(gate.mean()),
                             unc_valid_frac=float(np.isfinite(uv).mean()),
                             pfr_valid_frac=float(np.isfinite(Lt.cci_pfr.values).mean()), gtd_valid_frac=float(np.isfinite(Lt.cci_gtd_1m.values).mean()),
                             blockmed_valid_frac=float(np.isfinite(Lt.cci_alt_blockmed.values).mean()),
                             unc_median_cm=float(np.nanmedian(uv)) if np.isfinite(uv).any() else None,
                             w_cci_median_main=float(np.nanmedian((1 / np.maximum(uv, args.unc_floor_cm) ** 2) / (1 / np.maximum(uv, args.unc_floor_cm) ** 2 + 1 / sigma_st2))) if np.isfinite(uv).any() else None,
                             w_cci_median_srcvar=float(np.nanmedian((1 / np.maximum(uv, args.unc_floor_cm) ** 2) / (1 / np.maximum(uv, args.unc_floor_cm) ** 2 + 1 / sigma_src[tg]["value"]))) if np.isfinite(uv).any() else None)
        stt = BlockStore(tg, 0, blk, meta=dict(cond="noinfo", n_cells=int(len(yv))))
        for name, p in R.items():
            stt.add((name,), yv, p)
            ok = np.isfinite(yv) & np.isfinite(p)
            worst.append(dict(target=tg, rule=name, rmse=float(np.sqrt(np.mean((p[ok] - yv[ok]) ** 2))), n=int(ok.sum()),
                              bias=float(np.mean(p[ok] - yv[ok])), gated_frac=gate_info[tg]["gated_frac"]))
        stores[tg] = stt
        cdf = pd.DataFrame(dict(target=tg, loc_id=loc, block=blk, y=yv, gate=gate.astype(int), cci_alt=cci, cci_alt_unc=uv,
                                cci_alt_blockmed=Lt.cci_alt_blockmed.values, cci_alt_blockmed_cells=Lt.cci_alt_blockmed_cells.values,
                                cci_pfr=Lt.cci_pfr.values, cci_gtd_1m=Lt.cci_gtd_1m.values, cci_gtd_2m=Lt.cci_gtd_2m.values))
        for name, p in R.items():
            cdf[f"pred_{name}"] = p
        rows_cells.append(cdf)
    if not stores:
        raise SystemExit("채점 대상 없음")
    W = pd.DataFrame(worst); W.to_csv(OUT / f"{tag}_worst.csv", index=False)
    C = pd.concat(rows_cells, ignore_index=True); C.to_csv(OUT / f"{tag}_cells.csv", index=False)
    save_stores(stores.values(), OUT / f"{tag}_blocksse.npz")

    # 검정
    ab4 = [t for t in AB4 if t in stores]; main6 = [t for t in MAIN6 if t in stores]
    rows = []            # (family, level, row, dist)
    for fam, pairs in FAMILIES.items():
        for a, b in pairs:
            test = f"C1_{a}_vs_{b}"
            per = {}
            for tg in stores:
                r = boot_delta_blocks({0: stores[tg]}, (a,), (b,), nboot=nboot, seed=seed_of("c1", tg), return_dist=True)
                per[tg] = r
                point_only = tg not in AB4
                row = dict(family=fam, test=test, method=a, ref=b, cond="noinfo", target=tg, level="target",
                           ci_scope="point_only" if point_only else "CI", n_regions=1, regions=tg,
                           n_cells=int(stores[tg].ncell.sum()), n_blocks=int(stores[tg].nb), rmse_A=r["rmse_A"], rmse_B=r["rmse_B"],
                           delta=r["delta"], ci_lo=np.nan if point_only else r["ci_lo"], ci_hi=np.nan if point_only else r["ci_hi"],
                           p_boot=np.nan if point_only else r["p_boot"], delta_beq=r["delta_beq"],
                           ci_lo_beq=np.nan if point_only else r["ci_lo_beq"], ci_hi_beq=np.nan if point_only else r["ci_hi_beq"],
                           split_win=r["split_win"], rep_win=r["rep_win"], region_win=np.nan, ci_rep_lo=np.nan, ci_rep_hi=np.nan, ci_flag=r["ci_flag"])
                rows.append((fam, tg, row, None if point_only else r["dist"]))
            if ab4:
                dist = np.mean([per[t]["dist"] for t in ab4], 0); dist_b = np.mean([per[t]["dist_beq"] for t in ab4], 0)
                dm = float(np.mean([per[t]["delta"] for t in ab4]))
                lo, hi = pct_ci(dist, ALPHA); lob, hib = pct_ci(dist_b, ALPHA)
                assert lo - 1e-9 <= dm <= hi + 1e-9 or not np.isfinite(lo), f"{test}: 점 추정 {dm:.3f} 가 CI [{lo:.3f}, {hi:.3f}] 밖"
                neg = sum(per[t]["delta"] < 0 for t in ab4)
                rows.append((fam, "AB4", dict(family=fam, test=test, method=a, ref=b, cond="noinfo", target=f"MEAN[AB4:{','.join(ab4)}]",
                                              level="AB4", ci_scope="CI", n_regions=len(ab4), regions=",".join(ab4),
                                              n_cells=int(sum(stores[t].ncell.sum() for t in ab4)), n_blocks=int(sum(stores[t].nb for t in ab4)),
                                              rmse_A=float(np.mean([per[t]["rmse_A"] for t in ab4])), rmse_B=float(np.mean([per[t]["rmse_B"] for t in ab4])),
                                              delta=dm, ci_lo=lo, ci_hi=hi, p_boot=boot_p(dist),
                                              delta_beq=float(np.mean([per[t]["delta_beq"] for t in ab4])), ci_lo_beq=lob, ci_hi_beq=hib,
                                              split_win=np.nan, rep_win=np.nan, region_win=float(np.mean([per[t]["delta"] < 0 for t in ab4])),
                                              ci_rep_lo=np.nan, ci_rep_hi=np.nan, ci_flag=f"neg {neg}/{len(ab4)}"), dist))
            if main6:
                neg = sum(per[t]["delta"] < 0 for t in main6)
                rows.append((fam, "MAIN6", dict(family=fam, test=test, method=a, ref=b, cond="noinfo", target=f"MEAN[MAIN6:{','.join(main6)}]",
                                                level="MAIN6", ci_scope="point_only", n_regions=len(main6), regions=",".join(main6),
                                                n_cells=int(sum(stores[t].ncell.sum() for t in main6)), n_blocks=int(sum(stores[t].nb for t in main6)),
                                                rmse_A=float(np.mean([per[t]["rmse_A"] for t in main6])), rmse_B=float(np.mean([per[t]["rmse_B"] for t in main6])),
                                                delta=float(np.mean([per[t]["delta"] for t in main6])), ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan,
                                                delta_beq=float(np.mean([per[t]["delta_beq"] for t in main6])), ci_lo_beq=np.nan, ci_hi_beq=np.nan,
                                                split_win=np.nan, rep_win=np.nan, region_win=float(np.mean([per[t]["delta"] < 0 for t in main6])),
                                                ci_rep_lo=np.nan, ci_rep_hi=np.nan, ci_flag=f"neg {neg}/{len(main6)}; 점 추정만"), None))
    # Holm: (가족, 수준) 단위
    groups = {}
    for fam, lvl, row, dist in rows:
        groups.setdefault((fam, lvl), []).append((row, dist))
    for (fam, lvl), items in groups.items():
        rr = [it[0] for it in items]; dd = [it[1] for it in items]
        if all(d is None for d in dd):
            for r in rr:
                r.update(p_holm=np.nan, holm_rank=np.nan, holm_level=np.nan, ci_lo_holm=np.nan, ci_hi_holm=np.nan,
                         reject_holm=False, holm_consistent=np.nan)
            continue
        holm_block(rr, dd)
    T = pd.DataFrame([it[2] for it in rows])
    T["ci_hi_neg"] = T.ci_hi < 0
    T["delta_blockeq"] = T.delta_beq                    # h4_analysis(C3 표) 호환 별칭
    T["improve_holm"] = T.reject_holm.astype(bool) & (T.delta < 0) & (T.ci_hi_holm < 0)
    T.to_csv(OUT / f"{tag}_tests.csv", index=False)
    n_incons = int((T.holm_consistent == False).sum())                                                # noqa: E712
    if n_incons:
        log(f"[warn] Holm p 와 Holm 조정 CI 불일치 {n_incons}행(백분위 보간 경계)")

    # 판정 표(계획 산출 c1_cci_layers.csv): 요약 수준 행
    D = T[T.level.isin(["AB4", "MAIN6"])].copy()
    D["f9_member"] = (D.family == "F9") & D.method.isin(F9_MEMBERS)
    f9 = D[(D.level == "AB4") & D.f9_member]
    f9_pass = bool(f9.improve_holm.any()) if len(f9) else None
    D["f9_member_strict"] = (D.family == "F9") & D.method.isin(F9_MEMBERS_STRICT)
    f9s = D[(D.level == "AB4") & D.f9_member_strict]
    f9_pass_strict = bool(f9s.improve_holm.any()) if len(f9s) else None
    f9_passing = sorted(f9.loc[f9.improve_holm.astype(bool), "method"].tolist())
    if f9_pass is None:
        f9_driver = None
    elif not f9_pass:
        f9_driver = "none"
    elif f9_pass_strict:
        f9_driver = "strict_member"
    else:
        f9_driver = "blocksmooth_only"
    D["F9_pass"] = f9_pass
    D["F9_pass_strict"] = f9_pass_strict
    D["F9_pass_driver"] = f9_driver
    D.to_csv(OUT / f"{tag}_cci_layers.csv", index=False)
    log(f"[F9] 주 판정 F9_pass(§4, (ii)–(iv))={f9_pass}; 병기 F9_pass_strict(§5, (ii)·(iv))={f9_pass_strict}; "
        f"기준 충족 구성원={f9_passing}; driver={f9_driver}")
    log(f"[F9] AB4 Holm 판정(가족 F9, m=4): 통과={f9_pass} "
        + "; ".join(f"{r.method} Δ={r.delta:.2f} CI_holm=[{r.ci_lo_holm:.2f},{r.ci_hi_holm:.2f}] p_holm={r.p_holm:.3f}" for r in f9.itertuples()))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        commit = ""
    meta = dict(tag=tag, args=vars(args), targets=targets, targets_scored=list(stores), ab4=ab4, main6=main6, git_commit=commit,
                elapsed_min=round((time.time() - t0) / 60, 2),
                n_rows=dict(tests=len(T), decision=len(D), worst=len(W), cells=len(C), layers=len(tab)),
                E0_alaska=E0, sigma_st2=sigma_st2, n_source_cells=int(mm.sum()), sigma_src2=sigma_src, src_region_mse=reg_mse,
                gate_info=gate_info, extract=emeta, layer_csv=str(layer_csv.relative_to(ROOT)),
                m1_npz=[Path(f).name for f in npz_files], scoring_cells="M1 npz eval::noinfo|<t>|0 (y, loc_id, block) 그대로",
                layer_cols=[c for c in tab.columns if c.startswith("cci_")], nboot=nboot, alpha=ALPHA,
                F9=dict(pass_=f9_pass, pass_strict=f9_pass_strict, driver=f9_driver, passing_members=f9_passing,
                        primary="pass_ (계획 §4 C1, 구성원 (ii)–(iv)); pass_strict 는 계획 §5 F9 표(구성원 (ii)·(iv)) 병기",
                        family=[f"{a}-{b}" for a, b in FAMILIES["F9"]], members=list(F9_MEMBERS),
                        members_strict=list(F9_MEMBERS_STRICT), level="MEAN[AB4]",
                        rule="delta<0 & p_holm<0.05 & ci_hi_holm<0 (Holm 가족 = 결합 − stefan 4검정, 두 판정 공통)"),
                holm_inconsistent_rows=n_incons,
                ci_scope=dict(AB4="블록 부트스트랩 CI(지역 비가중, 같은 번호 평균)", MAIN6="점 추정만", Russia_C="점 추정만", Greenland="점 추정만"),
                variance_defs=dict(u2="(ALT_uncertainty 다년 평균 cm)², 제품 정의 = 화소 안 공간 변동성, 하한 --unc-floor-cm",
                                   sigma_st2="알래스카 원천 표본 안 Stefan 잔차 평균제곱(E0 고정)",
                                   sigma_src2="민감도: 대상 제외 원천 macro 지역별 Stefan 잔차 평균제곱의 비가중 평균(셀 ≥ --src-min-cells)"),
                rules=dict(stefan="M1 npz anchor::noinfo|<t>|0|alaska|stefan", stefan_cci="0.5(st+cci_alt)",
                           cci_uncw=f"(st/σ_st²+cci/u²)/(1/σ_st²+1/u²), u 하한 {args.unc_floor_cm} cm, u 결측→등가중",
                           cci_uncw_srcvar="cci_uncw 에서 σ_st² → σ²_src(t)",
                           cci_blocksmooth="0.5(st+CCI ALT 래스터 0.5° 창 다년 평균 화소 중앙값), 결측→셀 cci",
                           gate_gtd_pfr="PFR<50 % 또는 GTD T1m>-1 °C → st, 아니면 등가중; 결측 층은 게이트 미적용"),
                ci_rep="단일 분할·단일 반복이므로 정의되지 않음(NaN)")
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    pd.set_option("display.width", 260)
    log(W.pivot_table(index="rule", columns="target", values="rmse").round(2).to_string())
    log(D[["family", "test", "level", "n_cells", "delta", "ci_lo", "ci_hi", "p_boot", "p_holm", "ci_lo_holm", "ci_hi_holm", "improve_holm",
           "holm_consistent", "delta_beq", "ci_flag"]].round(3).to_string(index=False))
    log(f"[done] {tag} 경과 {(time.time() - t0)/60:.1f} min → {OUT}/{tag}_{{cci_layers,tests,worst,cells,blocksse,meta}}")
    logf.close()


if __name__ == "__main__":
    main()
