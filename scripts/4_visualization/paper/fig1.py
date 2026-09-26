"""Fig 1. 문제 정의: 지역별 Stefan 계수 E 차이와 정보 없음 전이 오차(스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.1).

패널
  a 범북극 지도(EPSG:3413 계열 극 입체): 0.5° 블록별 라벨 셀 수(면적 비례 원), 지역 번호 1–8(Table 1 순번),
    배경 ESA CCI PFR v4.0 1997–2021 평균 2단계 회색(연속 ≥ 90 %, 불연속 50–90 %)
  b 지역별 z = log ALT − log √TDD 스트립(셀 점 + IQR 상자 + 평균 z 점, 평균의 블록 부트스트랩 95 % CI)
    E_region = exp(mean z)(h4_common.offset_z 와 같은 로그 공간 원점 통과 최소제곱). E_AK = 알래스카 같은 추정량(Table 1 의 E0 = 지역 제외 적합과 다른 양)
  c 세 조건 도식(정보 없음·공변량만·라벨 있음)과 채점 블록 A/B(공변량만 조건의 A 블록 = 물리식 유사라벨, 빗금)
  d 정보 없음 전이 오차: 직접 ML(CatBoost) − 물리식, AB4 4지역(rep_row CI) + AB4 평균(strat_AB4 CI)

자료
  라벨 셀: polar.m1_core.load_base(fidelity_base_v3, F4_direct) → macro_region(알래스카 13,606셀/74블록 등, 감사 조치 10)
  외부 홀드아웃(8): load_base(sources=('F4_calm_temp',)) 의 Mongolia_CAsia(46셀, 지온 유도 융해 깊이)
  d: m1/m1_sc_tests.csv(정정본, ci_valid), label 'noinfo: catboost 직접 − Stefan 앵커'(교차 점검 2026-09-26: catboost_lo → catboost, 사실 목록 일치)
  PFR 평균 캐시: data/processed/cci_pfr_mean_1997_2021.nc(0.1° 집계, 연도별 누적 평균, 메모리 절약)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (ps, ROOT, PROC, AB4, M1, OUT, QA_DIR, load_m1_tests, save_paper, MissingData, abslogE_map,  # noqa: F401
                     order_by_abslogE)

PFR_DIR = ROOT / "data" / "raw" / "cci_pfr"
PFR_CACHE = PROC / "cci_pfr_mean_1997_2021.nc"
PFR_COARSE = 10                                   # 0.01° × 10 = 0.1° 집계

# Table 1 순번(스펙 §3): 알래스카 1, 레나 2, 캐나다 3, 러시아 W 4, 러시아 E 5, 러시아 C 6, 그린란드 7, 외부 홀드아웃 8
REGION_NO = {"Alaska": 1, "Lena": 2, "Canada": 3, "Russia_W": 4, "Russia_E": 5, "Russia_C": 6, "Greenland": 7, "Mongolia_CAsia": 8}
STRIP_REGIONS = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
MIN_BLOCKS_CI = 8                                 # 블록 < 8 지역은 점 추정만(M1 ci_flag 규약과 동일)
D_LABEL = "noinfo: catboost 직접 − Stefan 앵커"      # 사실 목록 정본(catboost +2.31 [0.76, 4.29], AB4)과 같은 모형
NBOOT = 1000
SEED = 20260926


# ================================================================ 자료
def pfr_mean_cache(force: bool = False):
    """CCI PFR 1997–2021 화소 평균을 0.1° 로 집계해 캐시(연도·위도 조각별로 읽어 누적). 반환 (lat, lon, pfr[%])."""
    import netCDF4 as nc
    if PFR_CACHE.exists() and not force:
        with nc.Dataset(PFR_CACHE) as f:
            return f["lat"][:].data, f["lon"][:].data, f["pfr_mean"][:].filled(np.nan)
    files = sorted(PFR_DIR.glob("ESACCI-PERMAFROST-L4-PFR-*-fv04.0.nc"))
    if len(files) != 25:
        raise MissingData(f"CCI PFR 연도 파일 {len(files)}/25 개({PFR_DIR.relative_to(ROOT)})")
    k = PFR_COARSE
    S = C = lat_c = lon_c = None
    for fp in files:
        with nc.Dataset(fp) as f:
            v = f["PFR"]; v.set_auto_mask(False)
            nlat, nlon = v.shape[1], v.shape[2]
            if S is None:
                S = np.zeros((nlat // k, nlon // k)); C = np.zeros_like(S)
                lat_c = f["lat"][:].data.reshape(-1, k).mean(1); lon_c = f["lon"][:].data.reshape(-1, k).mean(1)
            step = 600                               # 600 행(≈ 21 MB) 조각
            for i0 in range(0, nlat, step):
                a = v[0, i0:i0 + step, :].astype(np.float32)
                ok = a <= 100
                a[~ok] = 0
                r = a.shape[0] // k
                S[i0 // k:i0 // k + r] += a.reshape(r, k, nlon // k, k).sum((1, 3))
                C[i0 // k:i0 // k + r] += ok.reshape(r, k, nlon // k, k).sum((1, 3))
        print(f"[fig1] PFR {fp.name.split('-')[-2]} 누적")
    with np.errstate(invalid="ignore"):
        M = np.where(C > 0, S / np.maximum(C, 1), np.nan).astype(np.float32)
    with nc.Dataset(PFR_CACHE, "w") as g:
        g.createDimension("lat", len(lat_c)); g.createDimension("lon", len(lon_c))
        g.createVariable("lat", "f4", ("lat",))[:] = lat_c
        g.createVariable("lon", "f4", ("lon",))[:] = lon_c
        p = g.createVariable("pfr_mean", "f4", ("lat", "lon"), zlib=True, fill_value=np.float32(np.nan))
        p[:] = M; p.units = "percent"
        g.title = "ESA CCI Permafrost PFR v4.0, mean of 1997-2021 (25 annual files), aggregated to 0.1 deg"
        g.source = ", ".join(x.name for x in files)
        g.note = "1997-2002 ERA5 MODIS LST bias-corrected products, 2003-2021 MODIS LST CryoGrid products; pixel-year mean of valid pixels"
    return lat_c, lon_c, M


def load_cells() -> pd.DataFrame:
    """라벨 셀(F4_direct, macro 지역) + 외부 홀드아웃(Mongolia_CAsia, F4_calm_temp). 열: macro, lat, lon, block, y, s, z."""
    from polar.m1_core import load_base
    a = load_base(PROC)
    b = load_base(PROC, sources=("F4_calm_temp",))
    b = b[b.macro == "Mongolia_CAsia"]
    d = pd.concat([a, b], ignore_index=True)
    d = d[d.macro.isin(REGION_NO)].copy()
    d.attrs["n_population"] = len(a)                                 # 실험 모집단(load_base → macro_region, 감사 조치 10)
    d.attrs["omitted"] = a[~a.macro.isin(REGION_NO)].macro.value_counts().to_dict()   # Table 1 밖 소수 셀 지역
    d["y"] = d.alt_cm.astype(float); d["s"] = d.e5_sqrt_tdd.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["z"] = np.log(d.y) - np.log(d.s)
    d.attrs["source"] = "data/processed/fidelity_base_v3.csv (polar.m1_core.load_base)"
    return d


def block_boot_mean(z: np.ndarray, blocks: np.ndarray, nboot: int = NBOOT, seed: int = SEED):
    """평균 z 의 블록 부트스트랩 95 % CI(블록 복원 재표집, 셀 평균)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=z); cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), size=(nboot, len(ub)))
    bm = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(bm, [2.5, 97.5])


def strip_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in STRIP_REGIONS:
        q = d[(d.macro == r) & np.isfinite(d.z)]
        z = q.z.values; nb = q.block.nunique()
        lo, hi = block_boot_mean(z, q.block.values) if nb >= MIN_BLOCKS_CI else (np.nan, np.nan)
        rows.append(dict(region=r, no=REGION_NO[r], n_cells=len(z), n_blocks=nb, z_mean=z.mean(), z_lo=lo, z_hi=hi,
                         z_q25=np.percentile(z, 25), z_q75=np.percentile(z, 75)))
    T = pd.DataFrame(rows)
    z0 = float(T.loc[T.region == "Alaska", "z_mean"].iloc[0])
    T["E"] = np.exp(T.z_mean); T["E_lo"] = np.exp(T.z_lo); T["E_hi"] = np.exp(T.z_hi)
    T["abslogE_ratio"] = np.abs(T.z_mean - z0)
    T["ci_kind"] = np.where(T.n_blocks >= MIN_BLOCKS_CI, "block", "none")
    return T.sort_values("abslogE_ratio", kind="stable").reset_index(drop=True)


def transfer_table(order: list) -> pd.DataFrame:
    T = load_m1_tests(valid_only=True)
    q = T[(T.H == "X-model") & (T.label == D_LABEL) & (T.cond == "noinfo")]
    reg = q[q.target.isin(AB4)].set_index("target").loc[[r for r in order if r in AB4]].reset_index()
    mean = q[q.target == "REGION_SUMMARY_AB4"]
    if len(reg) != 4 or len(mean) != 1:
        raise MissingData(f"m1_sc_tests: '{D_LABEL}' AB4 행 {len(reg)}/4, 평균 {len(mean)}/1")
    out = pd.concat([reg, mean], ignore_index=True)[["target", "delta_rmse", "ci_lo", "ci_hi", "n_cells", "n_blocks", "n_seed", "ci_kind"]]
    out.attrs["source"] = "data/processed/m1/m1_sc_tests.csv"
    return out


# ================================================================ 패널
def _proj_extent(proj, lon, lat, aspect, pad=0.06):
    """점 집합을 담는 투영 좌표 범위(가로:세로 = aspect)."""
    import cartopy.crs as ccrs
    p = proj.transform_points(ccrs.PlateCarree(), np.asarray(lon), np.asarray(lat))
    x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def _map_extent(proj, d: pd.DataFrame, extra: pd.DataFrame, aspect: float, pad: float, margin_frac: float = 0.025):
    """지도 범위: 크기는 북극 지역 1–7 셀로만 정하고(외부 홀드아웃이 축척을 줄이지 않게), 남는 여유 안에서
    틀을 평행 이동해 외부 홀드아웃 블록(extra: lon, lat, n)을 가장 많이 담는다. 1–7 셀은 틀 안 margin_frac 여백을 지킨다."""
    import cartopy.crs as ccrs
    x0, x1, y0, y1 = _proj_extent(proj, d.lon, d.lat, aspect, pad=pad)
    w, h = x1 - x0, y1 - y0
    p = proj.transform_points(ccrs.PlateCarree(), d.lon.values, d.lat.values)
    mx, my = margin_frac * w, margin_frac * h
    # 1–7 이 여백 안에 남는 이동 범위
    dx_lo, dx_hi = (p[:, 0].max() + mx) - x1, (p[:, 0].min() - mx) - x0
    dy_lo, dy_hi = (p[:, 1].max() + my) - y1, (p[:, 1].min() - my) - y0
    q = proj.transform_points(ccrs.PlateCarree(), extra.lon.values, extra.lat.values)
    wt = extra.n.values.astype(float)
    best = (-1.0, 0.0, 0.0)
    for dx in np.linspace(min(dx_lo, 0), max(dx_hi, 0), 41):
        for dy in np.linspace(min(dy_lo, 0), max(dy_hi, 0), 41):
            inside = ((q[:, 0] > x0 + dx + mx) & (q[:, 0] < x1 + dx - mx) & (q[:, 1] > y0 + dy + my) & (q[:, 1] < y1 + dy - my))
            score = wt[inside].sum() - 1e-9 * (abs(dx) + abs(dy))        # 동점이면 이동이 작은 쪽
            if score > best[0]:
                best = (score, dx, dy)
    _, dx, dy = best
    return x0 + dx, x1 + dx, y0 + dy, y1 + dy


def pfr_image(ax, proj, ext, npx=(1400, 1030)):
    """PFR 2단계 회색 래스터를 지도 투영 격자로 최근접 표본화해 imshow(래스터, 600 dpi 저장)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap
    lat, lon, M = pfr_mean_cache()
    x0, x1, y0, y1 = ext
    xs = np.linspace(x0, x1, npx[0]); ys = np.linspace(y1, y0, npx[1])
    X, Y = np.meshgrid(xs, ys)
    ll = ccrs.PlateCarree().transform_points(proj, X, Y)
    LON, LAT = ll[..., 0], ll[..., 1]
    dlat = lat[1] - lat[0]; dlon = lon[1] - lon[0]
    i = np.clip(np.round((LAT - lat[0]) / dlat).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((LON - lon[0]) / dlon).astype(int), 0, len(lon) - 1)
    v = M[i, j]
    v[(LAT < lat.min() - 0.1) | (LAT > lat.max() + 0.1)] = np.nan
    cls = np.full(v.shape, np.nan)
    cls[(v >= 50) & (v < 90)] = 0; cls[v >= 90] = 1
    cm = ListedColormap([ps.GREY["pf_disc"], ps.GREY["pf_cont"]])
    im = ax.imshow(np.ma.masked_invalid(cls), extent=(x0, x1, y0, y1), origin="upper", cmap=cm, vmin=0, vmax=1,
                   interpolation="nearest", transform=proj, zorder=0.3, rasterized=True)
    return im


def panel_map(fig, d: pd.DataFrame, rect=(0.0, 52.0, 98.0, 72.0)):
    """a: 범북극 지도(블록별 셀 수 원, 지역 번호, PFR 2단계 배경). 반환 (ax, B, n_off_blocks, n_off_cells)."""
    import cartopy.crs as ccrs
    ax = ps.paper_map_ax(fig, rect, extent=(-180, 180, 40, 90), lon0=-45, grid_lat=10, grid_lon=30,
                         scalebar_km=None, inset=False)
    proj = ax._paper_proj
    B = d.groupby(["macro", "block"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("y", "size")).reset_index()
    ext_mask = (B.macro == "Mongolia_CAsia").values
    ext = _map_extent(proj, d[d.macro != "Mongolia_CAsia"], B[ext_mask], rect[2] / rect[3], pad=0.05)
    ax.set_extent(ext, crs=proj)
    pfr_image(ax, proj, ext)
    ps.density_circles(ax, B.lon[~ext_mask], B.lat[~ext_mask], B.n[~ext_mask], color=ps.COLOR["refit"])
    ps.density_circles(ax, B.lon[ext_mask], B.lat[ext_mask], B.n[ext_mask], color=EXT_COLOR, alpha=0.6)
    # 틀 밖 외부 홀드아웃 블록: 가장 가까운 틀 가장자리 안쪽에 채운 삼각형(off-scale 규약과 같은 문법, 축 안 1.4 mm)
    # + 삼각형 옆 직접 표지(캡션 없이 읽히도록, round 2 검토 반영)
    PB = proj.transform_points(ccrs.PlateCarree(), B.lon.values, B.lat.values)
    x0, x1, y0, y1 = ext
    mpp = (ext[1] - ext[0]) / rect[2]                          # 투영 m / mm
    inset_m = OFFSCALE_INSET_MM * mpp
    off = ext_mask & ~((PB[:, 0] > x0) & (PB[:, 0] < x1) & (PB[:, 1] > y0) & (PB[:, 1] < y1))
    edge_pts = {}
    for i in np.where(off)[0]:
        px, py = PB[i, 0], PB[i, 1]
        if py > y1:
            key = "^"                                                   # 가장자리별 삼각형 1개로 합침
        elif px > x1:
            key = ">"
        elif py < y0:
            key = "v"
        else:
            key = "<"
        edge_pts.setdefault(key, []).append((px, py))
    for mk, pts in edge_pts.items():
        pts = np.array(pts)
        ex = float(np.clip(pts[:, 0].mean(), x0, x1)); ey = float(np.clip(pts[:, 1].mean(), y0, y1))
        if mk == "^":
            ey = y1 - inset_m
        elif mk == "v":
            ey = y0 + inset_m
        elif mk == ">":
            ex = x1 - inset_m
        else:
            ex = x0 + inset_m
        ax.plot([ex], [ey], mk, color=EXT_COLOR, mec=EXT_COLOR, ms=ps.MS["main"] + 0.6, transform=proj, clip_on=True,
                zorder=6)
        # 직접 표지: 삼각형 왼쪽(가로 가장자리) 또는 아래쪽(세로 가장자리), 흰 바탕
        ha, dxy = (("right", (-2.2, 0.0)) if mk in "^v" else (("right" if mk == ">" else "left"), (0.0, -2.6)))
        nb = len(pts)
        t = ax.text(ex + dxy[0] * mpp, ey + dxy[1] * mpp, f"{nb} blocks beyond frame", transform=proj, ha=ha,
                    va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
        t.set_gid("offscale_key")
    # 지역 번호 표지(흰 바탕 정사각 0.5 pt #4d4d4d). 위치 = 지역 셀 중심 + mm 오프셋(외부 홀드아웃은 틀 안 블록 기준)
    P = proj.transform_points(ccrs.PlateCarree(), d.lon.values, d.lat.values)
    d = d.assign(px=P[:, 0], py=P[:, 1])
    inside_d = (d.px > x0) & (d.px < x1) & (d.py > y0) & (d.py < y1)
    badges = {}
    for r, no in REGION_NO.items():
        q = d[(d.macro == r) & inside_d]
        cx, cy = np.median(q.px), np.median(q.py)
        dx, dy = BADGE_OFFSET_MM.get(r, (3.0, 3.0))
        badges[r] = badge(ax, cx + dx * mpp, cy + dy * mpp, no, proj)
    for r, lon_min, (dx, dy) in BADGE_EXTRA:                  # 떨어진 사이트 묶음(캐나다 북극 제도 CALM)
        q = d[(d.macro == r) & (d.lon > lon_min)]
        if len(q):
            bx = badge(ax, np.median(q.px) + dx * mpp, np.median(q.py) + dy * mpp, REGION_NO[r], proj)
            # 같은 번호 두 표지를 가는 점선으로 연결(한 지역의 두 묶음임을 캡션 없이 표시)
            if r in badges:
                ax.figure.canvas.draw()                        # 표지 bbox 패치 크기 확정(shrink 계산용)
                ax.annotate("", xy=badges[r].get_position(), xycoords=proj._as_mpl_transform(ax),
                            xytext=bx.get_position(), textcoords=proj._as_mpl_transform(ax),
                            arrowprops=dict(arrowstyle="-", color=ps.GREY["text2"], lw=0.4, ls=(0, (1.2, 1.2)),
                                            patchA=bx.get_bbox_patch(), patchB=badges[r].get_bbox_patch(),
                                            shrinkA=0.5, shrinkB=0.5), zorder=5.5).set_gid("region_link")
    ps.scale_bar(ax, 1000, loc=SCALEBAR_LOC)
    return ax, B.assign(in_frame=~off), int(off.sum()), int(B.n[off].sum())


def panel_strip(ax, d: pd.DataFrame, T: pd.DataFrame):
    """b: 지역별 z 스트립(셀 점, IQR 상자, 평균 ± 블록 CI), E_AK 세로선."""
    from matplotlib.patches import Rectangle
    n = len(T)
    ys = np.arange(n)[::-1].astype(float)
    rng = np.random.default_rng(SEED)
    zlim = np.log(E_LIM)
    z0 = float(T.loc[T.region == "Alaska", "z_mean"].iloc[0])
    ax.axvline(z0, color=ps.COLOR["phys"], lw=ps.LW["ref"], ls="-", zorder=1.5)
    ax.annotate(r"E$_\mathrm{AK}$", xy=(z0, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], annotation_clip=False)
    n_out = 0
    for yv, (_, r) in zip(ys, T.iterrows()):
        z = d[(d.macro == r.region) & np.isfinite(d.z)].z.values
        inside = (z >= zlim[0]) & (z <= zlim[1]); n_out += int((~inside).sum())
        # 점 수 상한(MAX_PTS): 셀 수가 두 자릿수 넘게 다른 지역 사이에서 겹침 농도가 범주처럼 보이지 않게(round 2)
        zi = z[inside]
        if len(zi) > MAX_PTS:
            zi = rng.choice(zi, MAX_PTS, replace=False)
        jit = rng.uniform(-0.3, 0.3, len(zi))
        ax.scatter(zi, yv + jit, s=ps.MS["point"] ** 2, color=ps.COLOR["direct"], alpha=0.2, lw=0,
                   zorder=2, rasterized=True)
        if r.n_cells >= 10:
            ax.add_patch(Rectangle((r.z_q25, yv - 0.3), r.z_q75 - r.z_q25, 0.6, fc="none", ec=ps.GREY["text2"],
                                   lw=0.6, zorder=3))
        if np.isfinite(r.z_lo):
            ax.plot([r.z_lo, r.z_hi], [yv, yv], color="#000000", lw=ps.LW["ci_forest"], solid_capstyle="round", zorder=4)
        ax.plot([r.z_mean], [yv], "o", color="#000000", ms=ps.MS["main"], mec="white", mew=0.4, zorder=5)
    ax.set_yticks(ys); ax.set_yticklabels([ps.region_name(r) for r in T.region])
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0, pad=2); ax.spines["left"].set_visible(False)
    ax.set_xlim(*zlim)
    ax.set_xticks(np.log(E_TICKS)); ax.set_xticklabels([f"{v:g}" for v in E_TICKS])
    ax.set_xlabel(r"Stefan coefficient E (cm (°C d)$^{-1/2}$)")
    return n_out


def panel_schematic(ax):
    """c: 세 조건 도식(학습 자료 → 채점 블록). 좌표 = mm(축 상자 98 × 40 mm).
    학습 자료 정의(m1_master_factorial.py): noinfo = 원천 실측만, covonly = 원천 실측 ∪ A 블록 물리식 유사라벨
    (A 블록 공변량에 Stefan 앵커 적용), labels = 원천 실측 ∪ A 블록 실측. 채점은 noinfo 전체 셀, 나머지 B 블록."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    W, H = ax._box_mm[2], ax._box_mm[3]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    blue_t = "#d5e0ec"                                           # refit 파랑 tint(실측 자료)
    hatch_c = "#8fa9c6"                                          # 유사라벨 빗금(파랑 계열, 실측 채움과 모양으로 구분)
    grey_s = "#bdbdbd"                                           # 채점 회색(phys tint)
    # 칩 = (문구, 종류, 다음 줄과 붙임). 종류: data(실측), pseudo(유사라벨, 빗금), absent(없음, 점선)
    cols = [("No information", [("Source labels (Alaska)", "data", False), ("No target data", "absent", False)], "all"),
            ("Covariates only", [("Source labels (Alaska)", "data", False), ("Physics pseudo-labels", "pseudo", True),
                                 ("in A blocks", "pseudo", False)], "B"),
            ("Labels", [("Source labels (Alaska)", "data", False), ("Target labels in A blocks", "data", False)], "AB")]
    cw, gap = (W - 2 * 2.0) / 3, 2.0
    pattern = np.array([[1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1]])          # 1 = A 블록, 0 = B 블록
    sq, sg = 3.1, 0.55

    def hatched(x, y, w, h, rounded):
        kw = dict(fc="white", ec=hatch_c, lw=0, hatch="//////")
        if rounded:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", **kw))
        else:
            ax.add_patch(Rectangle((x, y), w, h, **kw))

    for k, (head, chips, score) in enumerate(cols):
        x0 = k * (cw + gap)
        # 머리글 = 범주 표지(7.5 pt 굵게, 축 제목과 같은 크기). 상자 대신 0.4 pt 규칙선(round 2: 슬라이드형 카드 제거)
        ax.text(x0 + cw / 2, H - 2.3, head, ha="center", va="center", fontsize=ps.FS["label"], fontweight="bold")
        ax.plot([x0 + 1.0, x0 + cw - 1.0], [H - 4.4, H - 4.4], color=ps.GREY["text2"], lw=0.4, solid_capstyle="butt")
        # 학습 자료 칩
        yc = H - 7.2
        tbox = dict(boxstyle="square,pad=0.08", fc="white", ec="none")
        for text, kind, joined in chips:
            if kind == "pseudo":
                hatched(x0 + 1.0, yc - 3.4, cw - 2.0, 3.4, rounded=False)
            else:
                ax.add_patch(FancyBboxPatch((x0 + 1.0, yc - 3.4), cw - 2.0, 3.4, boxstyle="round,pad=0,rounding_size=0.6",
                                            fc="white" if kind == "absent" else blue_t,
                                            ec=ps.GREY["mid"] if kind == "absent" else "none",
                                            lw=0.5 if kind == "absent" else 0, ls=(0, (2, 1.5)) if kind == "absent" else "-"))
            ax.text(x0 + cw / 2, yc - 1.7, text, ha="center", va="center", fontsize=ps.FS["annot"],
                    color=ps.GREY["text2"] if kind == "absent" else "#000000",
                    bbox=tbox if kind == "pseudo" else None)
            yc -= 3.4 if joined else 4.2
        # 화살표(1종)
        ya = 16.6
        ax.annotate("", xy=(x0 + cw / 2, ya - 2.4), xytext=(x0 + cw / 2, ya + 1.0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=ps.GREY["text2"], lw=0.6,
                                    shrinkA=0, shrinkB=0))
        # 대상 지역 블록(A/B)
        nx, ny = pattern.shape[1], pattern.shape[0]
        gx0 = x0 + (cw - (nx * sq + (nx - 1) * sg)) / 2
        gy0 = 5.2
        for iy in range(ny):
            for ix in range(nx):
                isA = pattern[iy, ix] == 1
                xx = gx0 + ix * (sq + sg); yy = gy0 + (ny - 1 - iy) * (sq + sg)
                if score == "all":
                    fc, lab = grey_s, ""
                elif score == "B":
                    fc, lab = ((None, "A") if isA else (grey_s, "B"))
                else:
                    fc, lab = ((blue_t, "A") if isA else (grey_s, "B"))
                if fc is None:
                    hatched(xx, yy, sq, sq, rounded=False)
                    ax.add_patch(Rectangle((xx, yy), sq, sq, fc="none", ec=ps.GREY["text2"], lw=0.4))
                else:
                    ax.add_patch(Rectangle((xx, yy), sq, sq, fc=fc, ec=ps.GREY["text2"], lw=0.4))
                if lab:
                    ax.text(xx + sq / 2, yy + sq / 2, lab, ha="center", va="center", fontsize=ps.FS["annot"],
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if fc is None else None)
        ax.text(x0 + cw / 2, 2.4, "Scored: all target cells" if score == "all" else "Scored: B blocks",
                ha="center", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    return ax


def panel_transfer(ax, D: pd.DataFrame):
    """d: 정보 없음 직접 ML − 물리식(가로 포레스트). 지역 rep_row CI, 평균 strat_AB4 CI."""
    reg = D[D.target != "REGION_SUMMARY_AB4"].reset_index(drop=True)
    mean = D[D.target == "REGION_SUMMARY_AB4"].iloc[0]
    y_reg = np.arange(len(reg))[::-1] + 1.3
    s = ps.style_of("direct")
    for yv, (_, r) in zip(y_reg, reg.iterrows()):
        ps.ci_errorbar(ax, yv, r.ci_lo, r.ci_hi, orient="h", method="direct", lw=ps.LW["ci_forest"])
        ax.plot([r.delta_rmse], [yv], ls="none", marker=s["marker"], ms=ps.MS["main"] + 0.6, mec=s["color"], mew=0.9, zorder=4)
    ps.ci_errorbar(ax, 0.0, mean.ci_lo, mean.ci_hi, orient="h", method="direct", lw=ps.LW["ci_forest"])
    ax.plot([mean.delta_rmse], [0.0], ls="none", marker=s["marker"], ms=ps.MS["mean"] + 1.2, mec=s["color"], mew=1.3, zorder=4)
    ax.axhline(0.65, color=ps.GREY["light"], lw=0.4, zorder=0.5)
    ax.set_yticks(list(y_reg) + [0.0])
    ax.set_yticklabels([ps.region_name(t) for t in reg.target] + ["Mean (4 regions)"])
    ax.set_ylim(-0.6, y_reg.max() + 0.6)
    ax.tick_params(axis="y", length=0, pad=2); ax.spines["left"].set_visible(False)
    ax.set_xlim(*D_XLIM); ax.set_xticks(D_XTICKS)
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ps.zero_line(ax, "x", better_text=True)
    ax.set_xlabel("ΔRMSE vs physics (cm)")
    return ax


def badge(fig_or_ax, x, y, no, transform, **kw):
    """지역 번호 표지(흰 바탕 정사각, 0.5 pt #4d4d4d, 6.5 pt). a 지도와 b·d 행에서 같은 모양."""
    t = fig_or_ax.text(x, y, str(no), transform=transform, ha="center", va="center", fontsize=ps.FS["annot"], zorder=6,
                       bbox=dict(boxstyle="square,pad=0.22", fc="white", ec=ps.GREY["text2"], lw=0.5), **kw)
    t.set_gid("region_badge")
    return t


def row_badges(fig, ax, regions, x_mm):
    """b·d 행 라벨 왼쪽(슬롯 좌단)에 지역 번호 표지. 행 y 는 축의 y 눈금 순서와 같다."""
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(fig.transFigure, ax.transData)
    ys = ax.get_yticks()
    fx = x_mm / ps.fig_mm(fig)[0]
    for r, yv in zip(regions, ys):
        badge(fig, fx, yv, REGION_NO[r], tr)


# ================================================================ 배치 상수(mm)
EXT_COLOR = "#4d4d4d"
BADGE_OFFSET_MM = {"Alaska": (-5.0, 4.0), "Lena": (0.0, 5.0), "Canada": (-5.0, -4.0), "Russia_W": (5.0, 0.0),
                   "Russia_E": (-4.0, 4.0), "Russia_C": (-4.0, -4.0), "Greenland": (4.0, 0.0), "Mongolia_CAsia": (5.0, -3.0)}
SCALEBAR_LOC = (0.05, 0.06)
BADGE_X_MM = 104.0
BADGE_EXTRA = [("Canada", -100.0, (-4.5, 0.0))]
E_LIM = (0.2, 6.0)
MAX_PTS = 500                                     # b 지역별 표시 점 상한(무작위 부분 표본, 통계는 전체 셀)
OFFSCALE_INSET_MM = 1.4                           # a 틀 밖 삼각형의 틀 안쪽 거리
E_TICKS = (0.25, 0.5, 1, 2, 4)
D_XLIM = (-5.0, 8.0)
D_XTICKS = (-4, 0, 4, 8)


# ================================================================ 그림
def build(draft: bool = False):
    ps.use_paper()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = load_cells()
    T = strip_table(d)
    D = transfer_table(list(T.region))

    fig = ps.paper_figure(180, FIG_H)
    ax_a, B, n_off_b, n_off_c = panel_map(fig, d, rect=(0.0, ROW1_Y, 98.0, 71.0))
    ax_b = ps.slot_mm(fig, 102.0, ROW1_Y, 20.0, 56.0, 71.0)
    n_out = panel_strip(ax_b, d, T)
    ax_c = ps.axes_mm(fig, 0.0, 2.0, 98.0, 40.0)
    panel_schematic(ax_c)
    ax_d = ps.slot_mm(fig, 102.0, 12.0, 20.0, 56.0, 29.5)
    panel_transfer(ax_d, D)
    row_badges(fig, ax_b, list(T.region), BADGE_X_MM)
    row_badges(fig, ax_d, [t for t in D.target if t in REGION_NO], BADGE_X_MM)

    # 범례 1개(a 아래): 원 크기 3단계 + 외부 홀드아웃 + PFR 2단계
    hs = ps.size_legend_handles((6, 50, 200), fmt="{:,}")
    hs[0].set_label("≤ 6 cells"); hs[1].set_label("50 cells"); hs[2].set_label("≥ 200 cells")
    hs.append(Line2D([], [], ls="none", marker="o", ms=float(np.sqrt(ps.circle_size(50))), mfc=EXT_COLOR, mec=ps.GREY["edge"],
                     mew=0.3, alpha=0.6, label="External holdout"))
    hs += [Patch(fc=ps.GREY["pf_cont"], ec="#808080", lw=0.3, label="PFR ≥ 90 %"),
           Patch(fc=ps.GREY["pf_disc"], ec="#808080", lw=0.3, label="PFR 50–90 %")]
    ps.legend_below(fig, hs, LEGEND_RECT, ncol=3, handletextpad=0.4, columnspacing=1.0)
    ps.label_panels([ax_a, ax_b, ax_c, ax_d], "abcd")

    src_b = T[["no", "region", "n_cells", "n_blocks", "z_mean", "z_lo", "z_hi", "z_q25", "z_q75", "E", "E_lo", "E_hi", "abslogE_ratio", "ci_kind"]]
    src_a = B.rename(columns={"macro": "region", "n": "n_cells"}).assign(no=lambda x: x.region.map(REGION_NO))
    cap = caption(T, D, B, d, n_out, n_off_b, n_off_c)
    res = save_paper(fig, "Fig1_problem", caption=cap, draft=draft,
                      spec=dict(intent="Regions differ in the Stefan coefficient E, and without target information direct ML "
                                       "transfers worse than the physics anchor.",
                                panels=4, layout_mm="row 1 (y 58-129): a map 98 + gap 4 + b 20+56, axis height 71; "
                                                    "legend band under a (y 50.3-57.7); row 2 (y 2-42): c schematic 98 + gap 4 + "
                                                    "d 20+56 (axis y 12-41.5); b x-title to d 'better' label gap >= 3 mm; total 134 mm",
                                deviations=["legend: horizontal band beneath panel a instead of inside a (bottom-left), "
                                            "because the lower-left map area holds Canada sites and the scale bar",
                                            "map extent sized on regions 1-7 only; external holdout 8 blocks outside the frame "
                                            "are marked with one filled triangle 1.4 mm inside the nearest frame edge plus a direct label "
                                            "('6 blocks beyond frame')",
                                            "b: plotted points subsampled to <= 500 per region (statistics use all cells) so "
                                            "overplot density does not read as an encoded class",
                                            "b: reference line labelled E_AK (Alaska exp(mean z)); Table 1 E0 is a leave-region-out "
                                            "fit and is a different quantity",
                                            "b/d row order uses the log-space E (spec 2.1); Table 1 uses least-squares E and "
                                            "orders Canada before Russia E (disclosed in caption)",
                                            "c: covariates-only A blocks hatched (physics pseudo-labels on A-block covariates, "
                                            "m1_master_factorial.py covonly: train = source labels + A pseudo-labels)",
                                            "canvas 134 mm (spec 128) to separate b x-axis title from d"],
                                assets=["data/processed/fidelity_base_v3.csv", "data/processed/cci_pfr_mean_1997_2021.nc",
                                        "data/raw/cci_pfr/ESACCI-PERMAFROST-L4-PFR-*-fv04.0.nc", "data/processed/m1/m1_sc_tests.csv"],
                                ci_kinds=dict(b="block (1,000 block resamples of the region mean z)", d_regions="rep_row", d_mean="strat_AB4"),
                                source_data=["outputs/figures/paper/source_data/Fig1_a.csv", "Fig1_b.csv", "Fig1_d.csv"]),
                      sources={"a": src_a, "b": src_b, "d": D})
    write_values(T, D, B, d, draft, fig=fig)
    return res


def write_values(T, D, B, d, draft=False, fig=None):
    """스펙 §5.1 6단계: 그림 값 ≥ 3개를 원천 CSV 에서 다시 읽어 대조(_qa/Fig1_problem_values.txt)."""
    log = []
    ok_all = True

    def chk(name, fig_v, src_v, tol):
        nonlocal ok_all
        ok = bool(np.isclose(fig_v, src_v, atol=tol, rtol=0))
        ok_all &= ok
        log.append(f"{'OK  ' if ok else 'FAIL'} {name}: figure {fig_v:.6g} vs source {src_v:.6g} (tol {tol:g})")

    # d: m1_sc_tests.csv 원천 재독(로더 우회)
    R = pd.read_csv(M1 / "m1_sc_tests.csv")
    R = R[(R.H == "X-model") & (R.cond == "noinfo") & (R.label == D_LABEL)]
    for _, r in D.iterrows():
        q = R[R.target == r.target].iloc[0]
        for c in ("delta_rmse", "ci_lo", "ci_hi"):
            chk(f"d {r.target} {c}", float(r[c]), float(q[c]), 1e-9)
    # a·b: 셀·블록 수를 fidelity_base_v3 원천(h21_physics_rules 은 v1 기준이라 알래스카만)·Table 1 과 대조
    H21 = pd.read_csv(PROC / "h2" / "h21_physics_rules.csv").set_index("region")
    chk("b Alaska n_cells vs h21", float(T.set_index("region").loc["Alaska", "n_cells"]), float(H21.loc["Alaska", "n"]), 0)
    chk("b Alaska n_blocks vs h21", float(T.set_index("region").loc["Alaska", "n_blocks"]), float(H21.loc["Alaska", "n_blocks"]), 0)
    t1p = OUT / "Table1_region_summary.csv"
    t1p = t1p if t1p.exists() else OUT / "Table1_region_summary_draft.csv"
    if t1p.exists():
        T1 = pd.read_csv(t1p)
        # Table 1 CSV 스키마: 구판은 'no' 열로 자료 행을 구분, 현행은 kind == 'data'(교차 점검 2026-09-26 수정)
        T1 = (T1[T1.kind == "data"] if "kind" in T1 else T1.dropna(subset=["no"])).set_index("region")
        for r in T.region:
            if r in T1.index:
                chk(f"b {r} n_cells vs {t1p.name}", float(T.set_index('region').loc[r, 'n_cells']), float(T1.loc[r, 'n_cells']), 0)
                chk(f"b {r} n_blocks vs {t1p.name}", float(T.set_index('region').loc[r, 'n_blocks']), float(T1.loc[r, 'n_blocks']), 0)
        m = d[d.macro == "Mongolia_CAsia"]
        if "Mongolia_CAsia" in T1.index:
            chk("a Mongolia n_cells vs Table 1", float(len(m)), float(T1.loc["Mongolia_CAsia", "n_cells"]), 0)
            chk("a Mongolia n_blocks vs Table 1", float(m.block.nunique()), float(T1.loc["Mongolia_CAsia", "n_blocks"]), 0)
        else:
            log.append(f"info Mongolia not in {t1p.name} (holdout 8 is not a Table 1 row); figure {len(m)} cells, {m.block.nunique()} blocks")
        log.append("info Table 1 E_own (least squares) vs figure exp(mean z): "
                   + ", ".join(f"{r} {T1.loc[r, 'E_own']:.3f}/{e:.3f}" for r, e in zip(T.region, T.E) if r in T1.index))
    # a: 원 크기 합 = 셀 수, 모집단 = load_base 행 수
    chk("a sum of block cells (regions 1-7)", float(B[B.macro != "Mongolia_CAsia"].n.sum()),
        float((d.macro != "Mongolia_CAsia").sum()), 0)
    log.append(f"info population load_base {d.attrs.get('n_population')}, omitted {d.attrs.get('omitted')}")
    log.append(f"info external holdout blocks beyond frame {int((~B.in_frame).sum())} ({int(B.n[~B.in_frame].sum())} cells)")
    # 원천 파일 전체 행 수(17,572)는 실험 모집단(load_base)과 구분해 기록(감사 조치 10)
    raw = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=lambda c: c in ("alt_cm", "e5_sqrt_tdd", "macro_region", "source",
                                                                              "fidelity_source"))
    log.append(f"info fidelity_base_v3 rows {len(raw)}")
    if fig is not None:                                   # 글자 크기 위계: 패널 문자(9 pt) 외 모든 글자 ≤ 8 pt
        from matplotlib.text import Text
        sz = sorted({(round(t.get_fontsize(), 2), t.get_fontweight()) for t in fig.findobj(Text)
                     if t.get_visible() and t.get_text().strip()}, reverse=True)
        big = [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
               and t.get_fontsize() > 8.0 and t.get_text() not in "abcd"]
        chk("max text size excluding panel letters (pt, <= 8)", float(len(big)), 0.0, 0)
        log.append(f"info text sizes (pt, weight): {sz}")
    log.insert(0, f"Fig1_problem values cross-check: {'PASS' if ok_all else 'FAIL'}")
    stem = "Fig1_problem" + ("_draft" if draft else "")
    (QA_DIR / f"{stem}_values.txt").write_text("\n".join(log) + "\n")
    print(f"[fig1] values cross-check {'PASS' if ok_all else 'FAIL'} -> _qa/{stem}_values.txt")


FIG_H = 134.0
ROW1_Y = 58.0
LEGEND_RECT = (4.0, 50.3, 92.0, 7.4)


def caption(T, D, B, d, n_out, n_off_b=0, n_off_c=0) -> dict:
    """영문 캡션(블록 예산 80/80/150/40). 수치는 자료에서 읽는다."""
    lab = d[d.macro != "Mongolia_CAsia"]
    ext = d[d.macro == "Mongolia_CAsia"]
    nb_lab = int(lab.groupby(["macro", "block"]).ngroups)
    small = T[(T.ci_kind == "none") | (T.n_cells < 10)]
    small_txt = " and ".join(f"{ps.region_name(r.region)} ({int(r.n_cells)} cells)"
                             for _, r in small.iterrows())
    n_pop = int(d.attrs.get("n_population", len(lab)))
    omitted = d.attrs.get("omitted", {})
    n_omit = int(sum(omitted.values()))
    # Table 1(최소제곱 E, abslogE_map) 기준 AB4 순서. 그림 b 순서(로그 공간 E)와 다를 때만 문장을 넣는다
    t1_order = order_by_abslogE(list(AB4), abslogE_map())
    fig_order = [r for r in T.region if r in AB4]
    order_txt = ("" if t1_order == fig_order else
                 " Table 1's least-squares |log(E_own/E0)| ranks regions 2–5 as "
                 + " < ".join(ps.region_name(r) for r in t1_order) + ".")
    off_txt = (f" Triangle: {n_off_b} of its blocks ({n_off_c} cells) beyond the frame." if n_off_b else "")
    omit_txt = (f" ({n_pop:,}-cell population minus {len(omitted)} single-cell regions)"
                if n_omit else "")
    return dict(
        definition=("Fig. 1 | Regional Stefan coefficients and transfer error without target information. "
                    "The physics anchor predicts active-layer thickness (ALT) as E·√TDD, with TDD the ERA5-Land thawing "
                    "degree-day sum and E fitted on the source region (Alaska; E_AK in b). Table 1's E0 is a different, "
                    "leave-region-out fit. "
                    "ΔRMSE = direct-ML RMSE minus physics-anchor RMSE on the same cells (cm); negative favours ML. "
                    "Region numbers follow Table 1."),
        statistics=("b, E = exp(mean z), z = log ALT − log √TDD per cell; 95 % CI of the mean from 1,000 resamples of 0.5° "
                    f"blocks for regions with ≥ 8 blocks, boxes for ≥ 10 cells; {small_txt}: mean only."
                    + order_txt +
                    " d, CIs resample (split, repeat) rows for regions and strata of blocks for the mean."),
        panels=(f"a, {len(lab):,} label cells in {nb_lab} blocks of 0.5°{omit_txt}; circle area proportional to cells per "
                "block (bounded at 6 and 200). Region 3 has mainland and Arctic Archipelago clusters. "
                f"Dark circles: external holdout 8 ({len(ext)} cells, thaw depth derived from ground temperature).{off_txt} "
                "Background: ESA CCI Permafrost PFR v4.0, mean of 1997–2021 (ERA5-based to 2002, MODIS LST "
                "CryoGrid from 2003), 0.1°; polar stereographic, true scale at 70°N. "
                f"b, cells (subsampled to ≤ {MAX_PTS} per region; {n_out} off-axis omitted), "
                "interquartile boxes and means, ordered by |log(E/E_AK)|. "
                "c, training data and scored cells per condition; A and B are halves of the target 0.5° blocks (three "
                "splits); hatching, physics pseudo-labels on A-block covariates. d, CatBoost trained on Alaska, three seeds, "
                "all target cells; cell counts in Supplementary Table 1."),
        data=("Source data: data/processed/fidelity_base_v3.csv (polar.m1_core.load_base), "
              "data/processed/cci_pfr_mean_1997_2021.nc, data/processed/m1/m1_sc_tests.csv (X-model, noinfo, "
              "catboost direct minus Stefan anchor)."))
