"""Fig 5 라벨 배치(H29 블록 라벨 가치 지도 + B3 선택 규칙). 스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.5·§1.6·§5.

패널
  a  캐나다 블록 라벨 가치 지도(S3, 고유 블록 33개, 분할 평균)
  b  레나 블록 라벨 가치 지도(S3, 고유 블록 18개). a·b 색막대 1개 공유
  c  선택 규칙 4종의 E 추정 총분산 비(무작위 대비, n = 3), 로그 축
  d  선택 규칙 4종의 B블록 ΔRMSE 대 무작위(n = 10, S3)

부호 규약(원천과 다름, 주의)
  h29 원천 value_S* = 물리식 RMSE − 방법 RMSE(양수 = 개선). 논문 Δ 규약(§1.6, 음수 = 개선)에 맞추려고
  그림에는 ΔRMSE = −value_S3 을 그린다. 해로운 블록 = ΔRMSE > 0 = h29b 의 frac_harm_S3(value < 0) 와 같은 집합.

자료
  a·b  h3/h29b_blocks.csv(감사 조치 6: 고유 블록 단위 재집계), 요약 h3/h29b_summary.csv
  c    h4/b3_evar.csv(n = 3, vtot_ratio). 주 집계 대상 = f7_eval(블록 부트스트랩 가능 8 대상),
       셀 단위 부트스트랩 대체 4 대상은 회색, V_tot 이 없는 2 대상(AL-1·AL-6)은 제외. 판정 표 h4/b3_f7.csv
  d    h4/b3_summary.csv(blk_d_rand, 블록 CI 점 추정) + h4/b3_region.csv(MEAN 행, 층화 블록 부트스트랩 CI)

설계 메모(스펙 대비 조정, 근거)
  - 색 정규화: 스펙은 TwoSlopeNorm(±99 백분위를 5 cm 올림)이지만 블록 값 분포가 |Δ| ≤ 2 cm 다수 + 레나 −27 cm 극단이라
    선형이면 대부분의 원이 흰색에 가깝다. Fig 3·4 의 Δ 축과 같은 symlog(선형 ±2 cm) 0 중심 정규화를 쓰고 캡션에 적는다.
  - 원 크기: 셀 수가 1–1,183 으로 세 자릿수에 걸쳐 s = clip(0.6 n, 4, 120) 이면 1셀 블록 색이 보이지 않는다.
    면적 ∝ n 을 유지하되 하한 27셀·상한 200셀에서 자른다(12–90 pt²). 하한은 조밀 해칭(간격 약 0.6 mm)의 사선이
    최소 원에도 2개 이상 들어가게 정했다(검토 round 2: 흑백 인쇄에서 부호 정보 유지). 크기 표지는 색막대 아래 별도 키.
  - 겹침(검토 round 2): 모든 원이 서로 겹치지 않게 국소 반발 배치(_repel, 바깥 테두리 사이 ≥ 0.3 mm). 참 위치가
    원 밖으로 벗어난 원은 참 위치 점과 가는 회색 선으로 잇는다. 이동량은 source_data offset_mm·캡션에 기재.
    레나 지도는 NE 10 m 해안선. 원 위치는 라벨 셀 중심이라 수로를 걸친 블록은 물 위에 찍힌다(캡션 문장, † 표지는 시험 후 제외).
  - 지역 마커 범례는 Russia W·E 를 한 항목으로 묶어 범례 8항목 안에 넣는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (ps, load_h29_blocks, load_h29_summary, load_b3, save_paper, MissingData, H4, ROOT)  # noqa: F401

RULES = ["kmedoid", "kmedoid_weighted", "doptimal", "active_ppi"]
RULE_LABEL = {"kmedoid": "k-medoid", "kmedoid_weighted": "Weighted k-medoid", "doptimal": "D-optimal", "active_ppi": "Active PPI"}
# 규칙 범주 라벨: c(x 축)·d(y 축)가 같은 문자열·같은 줄바꿈을 쓴다(검토 round 2, 타이포그래피 일관성)
RULE_TICK = {"kmedoid": "k-medoid", "kmedoid_weighted": "Weighted\nk-medoid", "doptimal": "D-optimal", "active_ppi": "Active\nPPI"}

# 지도 설정: (lon/lat 범위, 중심 경도, 축척 막대 km, 경도 격자)
# res = Natural Earth 해안선·육지 해상도. 레나는 삼각주 섬·수로가 50 m 판에서 사라져 10 m 판을 쓴다(검토 round 1).
MAPS = {"Canada": dict(ll=(-142.0, -62.0, 58.5, 82.5), lon0=-102.0, bar=500, glon=10, glat=5, res="50m"),
        "Lena": dict(ll=(121.5, 132.5, 71.2, 74.0), lon0=127.0, bar=100, glon=2, glat=1, res="10m")}
SIZE_K, SIZE_MIN, SIZE_MAX = 0.45, 12.0, 90.0         # 원 면적 pt² = clip(0.45 n, 12, 90) → 27셀 이하·200셀 이상 절단
HATCH = "/" * 12                                      # 해칭 간격 약 0.6 mm: 최소 원(지름 1.4 mm)에도 사선 2개 이상(검토 round 2)
LW_PREF, LW_OTHER = 0.8, 0.3                          # 원 테두리 pt: k-중심 선호 블록 굵은 검정, 나머지 회색
GAP_MM = 0.3                                          # 원 바깥 테두리 사이 최소 여백(선호 테두리끼리 맞닿음 방지)
LEAD_MM = 0.4                                         # 참 위치가 원 테두리 밖으로 이만큼 넘게 벗어나면 점 + 회색 선
LINTHRESH, VMAX = 2.0, 30.0                          # 색 symlog 선형 구간 ±2 cm, 범위 ±30 cm(블록 최대 |Δ| 27.0 cm)
KMED_THR = 0.5                                        # k-중심 선호 블록: 10개 k-중심 라벨 중 평균 ≥ 0.5 개가 이 블록에 떨어짐
MARKER_GREY = "#4d4d4d"


def _size(n):
    return np.clip(SIZE_K * np.asarray(n, float), SIZE_MIN, SIZE_MAX)


def _norm():
    from matplotlib.colors import SymLogNorm
    return SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=-VMAX, vmax=VMAX, base=10)


def _map_ax(fig, rect_mm, region):
    """극 입체(true scale 70°N) 지도 축. 축 상자 비율에 맞춰 투영 좌표 범위를 넓혀 상자를 꽉 채운다(빈 여백 방지)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cfg = MAPS[region]
    proj = ccrs.NorthPolarStereo(central_longitude=cfg["lon0"], true_scale_latitude=70)
    x, y, w, h = rect_mm
    ax = ps.axes_mm(fig, x, y, w, h, projection=proj)
    lo0, lo1, la0, la1 = cfg["ll"]
    lons = np.r_[np.linspace(lo0, lo1, 60), np.full(30, lo1), np.linspace(lo1, lo0, 60), np.full(30, lo0)]
    lats = np.r_[np.full(60, la0), np.linspace(la0, la1, 30), np.full(60, la1), np.linspace(la1, la0, 30)]
    P = proj.transform_points(ccrs.PlateCarree(), lons, lats)
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    cx, cy, dx, dy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if dx / dy < w / h:
        dx = dy * w / h
    else:
        dy = dx * h / w
    ax.set_extent((cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2), crs=proj)
    res = cfg["res"]
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor=ps.GREY["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="white", edgecolor=ps.GREY["edge"], lw=0.25, zorder=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale(res), edgecolor=ps.GREY["edge"], lw=0.4 if res == "50m" else 0.3, zorder=1)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, lw=0.4, color=ps.GREY["grid"], alpha=0.6, zorder=0.5,
                 xlocs=np.arange(-180, 181, cfg["glon"]), ylocs=np.arange(40, 91, cfg["glat"]))
    ax.spines["geo"].set_linewidth(0.5)
    ax._paper_proj = proj
    sb = ps.scale_bar(ax, cfg["bar"], loc=(0.05, 0.05))
    sb.set_bbox(dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    ps.inset_locator(fig, ax, cfg["ll"], frac=0.2, corner="upper right")
    # 지역 이름(조건 표지, 제목 아님)
    ax.text(0.025, 0.975, region if region != "Lena" else "Lena Delta", transform=ax.transAxes, ha="left", va="top",
            fontsize=ps.FS["label"], zorder=6, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    return ax


def _gridline_labels(ax, region):
    """위도선 값 표지(지도 안 좌측, 6.5 pt 회색). 위도 판독용 최소 표기."""
    import cartopy.crs as ccrs
    cfg = MAPS[region]
    lo0 = cfg["ll"][0]
    lats = {"Canada": [60, 70, 80], "Lena": [72, 73]}[region]
    x0, x1 = ax.get_xlim()
    out = []
    for la in lats:
        # 위도선이 축 왼쪽 끝과 만나는 곳 근처에 둔다: 경도 lo0 쪽 점을 투영해 x 를 축 안으로 제한
        px, py = ax._paper_proj.transform_point(lo0 + (1.5 if region == "Lena" else 3.0), la, ccrs.PlateCarree())
        px = min(max(px, x0 + 0.01 * (x1 - x0)), x1)
        t = ax.text(px, py, f"{la}°N", fontsize=ps.FS["annot"], color=ps.GREY["text2"], ha="left", va="bottom",
                    transform=ax.transData, zorder=4, clip_on=True,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=ps.GREY["land"], edgecolor="none", alpha=0.9))
        out.append(t)
    return out


def _repel(ax, B, gap_mm=GAP_MM, iters=400):
    """겹침 제거 배치(표시 위치만 이동, 자료값 불변, 검토 round 2). 좌표는 축 위 mm.
    round 1 의 쌍별 초승달 규칙(_dodge)은 3개 이상 원이 한 점에 몰린 군집(Mackenzie 삼각주, Great Slave 호)에서
    저대비 쌍과 k-중심 선호 테두리 접촉을 남겼다. 그래서 모든 원 쌍의 바깥 테두리(반지름 + 선 굵기/2) 사이에
    gap_mm 이상의 여백이 생길 때까지 겹친 쌍을 서로 민다(국소 반발 이완). 이동 분담은 상대 면적의 역비율이라
    큰 원은 적게, 작은 원은 많이 움직인다. 축 밖으로 나가지 않게 가둔다. 반환: (x_mm, y_mm, 이동량 mm, 바깥 반지름 mm)."""
    import cartopy.crs as ccrs
    P = ax._paper_proj.transform_points(ccrs.PlateCarree(), B.lon.to_numpy(float), B.lat.to_numpy(float))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w_mm, h_mm = ax._box_mm[2], ax._box_mm[3]
    X = (P[:, 0] - x0) / (x1 - x0) * w_mm; Y = (P[:, 1] - y0) / (y1 - y0) * h_mm
    X0, Y0 = X.copy(), Y.copy()
    PT = 25.4 / 72.0
    R = np.sqrt(_size(B.n_cells.to_numpy()) / np.pi) * PT                       # 원 반지름 mm(면적 pt² → mm)
    lw = np.where(B.kmedoid_freq.to_numpy(float) >= KMED_THR, LW_PREF, LW_OTHER) * PT
    Ro = R + lw / 2                                                             # 바깥 테두리 반지름
    A = Ro ** 2
    rng = np.random.default_rng(3)
    for _ in range(iters):
        moved = False
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                dx, dy = X[j] - X[i], Y[j] - Y[i]
                d = float(np.hypot(dx, dy))
                need = Ro[i] + Ro[j] + gap_mm
                if d < need - 1e-4:
                    if d < 1e-3:                                                # 같은 위치: 결정적 임의 방향
                        a = rng.uniform(0, 2 * np.pi); ux, uy = np.cos(a), np.sin(a)
                    else:
                        ux, uy = dx / d, dy / d
                    ov = (need - d) * 0.5                                        # 이완 계수 0.5(진동 방지)
                    wi, wj = A[j] / (A[i] + A[j]), A[i] / (A[i] + A[j])          # 큰 원이 덜 움직임
                    X[i] -= ux * ov * wi; Y[i] -= uy * ov * wi
                    X[j] += ux * ov * wj; Y[j] += uy * ov * wj
                    moved = True
        X = np.clip(X, Ro + 0.2, w_mm - Ro - 0.2); Y = np.clip(Y, Ro + 0.2, h_mm - Ro - 0.2)
        if not moved:
            break
    # 최종 확인: 남은 겹침이 있으면 실패(여백 기준의 90 % 미만)
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            if np.hypot(X[j] - X[i], Y[j] - Y[i]) < Ro[i] + Ro[j] + 0.9 * gap_mm:
                raise RuntimeError(f"_repel 미수렴: {B.block.iloc[i]}·{B.block.iloc[j]}")
    return X, Y, np.hypot(X - X0, Y - Y0), Ro, X0, Y0


def _block_map(ax, B, norm, cm):
    """면적 비례 원(색 = ΔRMSE, 해칭 = 해로움, 굵은 검정 테두리 = k-중심 선호).
    겹침: _repel 로 모든 원을 서로 떨어뜨린다(바깥 테두리 사이 ≥ GAP_MM). 참 위치가 원 밖으로 벗어나면
    참 위치의 작은 점과 원 테두리를 가는 회색 선으로 잇는다. 그리는 순서는 셀 수 내림차순(겹침이 없어 표시에 영향 없음).
    물 위에 찍히는 원(라벨 셀 중심이 육지 다각형 밖)은 그림 기호 없이 캡션 문장으로 설명한다. 검토 round 2 에서
    † 표지를 시험했으나 지도에서 묘지 기호로 읽히고 해안선과 겹쳐 뺐다(source_data in_water 열로 기록)."""
    B = B.assign(_absd=B.d_S3.abs()).sort_values(["n_cells", "_absd"], ascending=[False, True]).reset_index(drop=True)
    X, Y, mv, Ro, X0, Y0 = _repel(ax, B)
    w_mm, h_mm = ax._box_mm[2], ax._box_mm[3]
    n_lead = 0
    for i, r in B.iterrows():
        pref = r.kmedoid_freq >= KMED_THR
        sc = ax.scatter([X[i] / w_mm], [Y[i] / h_mm], s=_size(r.n_cells), c=[r.d_S3], norm=norm, cmap=cm, transform=ax.transAxes,
                        linewidths=LW_PREF if pref else LW_OTHER, edgecolors="#000000" if pref else ps.GREY["edge"],
                        zorder=3.0 + 0.9 * i / max(len(B) - 1, 1))
        if r.d_S3 > 0:
            sc.set_hatch(HATCH)
        # 참 위치 표지: 참 위치가 원 테두리 밖으로 LEAD_MM 넘게 벗어날 때만(짧은 토막 선은 잡음이라 생략)
        if mv[i] > Ro[i] + LEAD_MM:
            ux, uy = (X[i] - X0[i]) / mv[i], (Y[i] - Y0[i]) / mv[i]
            ex, ey = X[i] - ux * Ro[i], Y[i] - uy * Ro[i]
            ax.plot([X0[i] / w_mm, ex / w_mm], [Y0[i] / h_mm, ey / h_mm], transform=ax.transAxes, color=ps.GREY["text2"],
                    lw=0.35, solid_capstyle="butt", zorder=2.8)
            ax.plot([X0[i] / w_mm], [Y0[i] / h_mm], transform=ax.transAxes, ls="none", marker="o", ms=1.1,
                    mfc=ps.GREY["text2"], mec="none", zorder=2.85)
            n_lead += 1
    return B.assign(offset_mm=mv, leader=mv > Ro + LEAD_MM)


def _water_blocks(B, region):
    """라벨 셀 중심이 Natural Earth 육지 다각형 밖(바다·하구 수로)에 있는 블록 번호 집합. 지도와 같은 해상도를 쓴다."""
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from shapely.prepared import prep
    q = B[B.target == region]
    bb = box(q.lon.min() - 2, q.lat.min() - 2, q.lon.max() + 2, q.lat.max() + 2)
    geoms = [g.intersection(bb) for g in shp.Reader(shp.natural_earth(MAPS[region]["res"], "physical", "land")).geometries()
             if g.intersects(bb)]
    L = prep(unary_union(geoms))
    return {int(b) for b, x, y in zip(q.block, q.lon, q.lat) if not L.contains(Point(x, y))}


def _colorbar_and_key(fig, rect_cb, rect_key, norm, cm):
    """세로 색막대(양수 쪽 해칭 = 해로운 블록) + 원 크기 키(범례 객체 아님)."""
    import matplotlib as mpl
    from matplotlib.ticker import FixedLocator, FuncFormatter
    cax = ps.axes_mm(fig, *rect_cb)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation="vertical")
    ticks = [-20, -5, -2, 0, 2, 5, 20]
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    cb.ax.tick_params(labelsize=ps.FS["tick"], width=ps.LW["tick"], length=2.0, pad=1.2)
    cb.outline.set_linewidth(0.5)
    # 양수(해로움) 구간 해칭: 지도 원의 해칭과 같은 부호
    from matplotlib.patches import Rectangle
    cax.add_patch(Rectangle((0, 0), 1, VMAX, transform=cax.transData, facecolor="none", hatch=HATCH,
                            edgecolor=ps.GREY["edge"], lw=0, zorder=3))
    cb.set_label("ΔRMSE vs physics (cm)", fontsize=ps.FS["label"], labelpad=1.5)
    cax.set_gid("colorbar")
    # 크기 키
    kx, ky, kw, kh = rect_key
    kax = ps.axes_mm(fig, kx, ky, kw, kh)
    kax.set_xlim(0, 1); kax.set_ylim(0, 1); kax.axis("off")
    kax.set_gid("size_key")
    kax.text(0.0, 1.0, "Cells per\nblock", fontsize=ps.FS["annot"], ha="left", va="top", transform=kax.transAxes)
    for yy, n, lab in [(0.56, 27, "≤ 27"), (0.36, 100, "100"), (0.13, 200, "≥ 200")]:
        kax.scatter([0.16], [yy], s=_size(n), facecolor="white", edgecolor=ps.GREY["mid"], linewidths=0.5, clip_on=False)
        kax.text(0.38, yy, lab, fontsize=ps.FS["annot"], ha="left", va="center")
    return cax, kax


def _jitter(k, m, width=0.30, seed=5):
    """결정적 흩뿌림(대상 순서 고정): [−width, width] 균등 간격을 섞어 배정."""
    if m <= 1:
        return np.zeros(m)
    v = np.linspace(-width, width, m)
    rng = np.random.default_rng(seed + k)
    return rng.permutation(v)


def _region_style(t):
    mk, sc = ps.region_marker(t)
    return mk, sc


def build(draft: bool = False):
    ps.use_paper()
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    # ------------------------------------------------------------ 자료
    B = load_h29_blocks()
    S29 = load_h29_summary()
    if "value_S3" not in B:
        raise MissingData("h29 블록 가치 열(value_S3)")
    B = B.copy(); B["d_S3"] = -B["value_S3"]                     # 논문 Δ 규약(음수 = 개선)
    B["harmful"] = B.d_S3 > 0
    B["kmedoid_pref"] = B.kmedoid_freq >= KMED_THR
    E = load_b3("evar", draft); Ssum = load_b3("summary", draft); R = load_b3("region", draft); F7 = load_b3("f7", draft)
    ev = E[(E.n == 3) & E.rule.isin(RULES)].copy()
    ev["f7_eval"] = ev.f7_eval.astype(str).str.lower().isin(["true", "1"])
    ev = ev[np.isfinite(ev.vtot_ratio)]
    sd = Ssum[(Ssum.n == 10) & (Ssum.stage == "S3") & Ssum.rule.isin(RULES)].copy()
    rg = R[(R.n == 10) & (R.stage == "S3") & (R.ref == "random") & R.rule.isin(RULES) & R.target.str.startswith("MEAN[")].copy()
    rg["layer"] = np.where(rg.target.str.count(",") == 3, "main", np.where(rg.target.str.count(",") == 9, "sub", "other"))
    rg = rg[rg.layer.isin(["main", "sub"])]

    # ------------------------------------------------------------ 배치(mm, 좌하 원점). 180 × 134
    H = 134.0
    fig = ps.paper_figure(180, H)
    cm = ps.cmap("diverging"); norm = _norm()
    MAP_Y, MAP_H = 66.0, 63.0
    ax_a = _map_ax(fig, (0.0, MAP_Y, 80.0, MAP_H), "Canada")
    ax_b = _map_ax(fig, (82.0, MAP_Y, 80.0, MAP_H), "Lena")
    offs = []
    water = {reg: _water_blocks(B, reg) for reg in ("Canada", "Lena")}
    for ax, reg in ((ax_a, "Canada"), (ax_b, "Lena")):
        offs.append(_block_map(ax, B[B.target == reg], norm, cm)[["target", "block", "offset_mm", "leader"]])
        _gridline_labels(ax, reg)
    offs = pd.concat(offs)
    B = B.merge(offs, on=["target", "block"], how="left")
    B["in_water"] = [int(b) in water[t] for t, b in zip(B.target, B.block)]
    n_off = int((B.offset_mm > 0.01).sum()); max_off = float(B.offset_mm.max()); n_lead = int(B.leader.sum())
    n_water = {reg: len(water[reg]) for reg in water}
    _colorbar_and_key(fig, (164.5, MAP_Y + 27.0, 2.5, 36.0), (163.5, MAP_Y, 16.0, 22.0), norm, cm)

    # c: 분산 비(로그 y), x = 규칙
    ax_c = ps.slot_mm(fig, 0.0, 12.0, 12.0, 58.0, 38.0)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.15, 4.0)
    ax_c.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 4]))
    ax_c.yaxis.set_minor_locator(NullLocator())
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax_c.axhline(1.0, **ps.DELTA_ZERO)
    ax_c.axhline(0.7, color=ps.GREY["text2"], lw=0.6, ls=(0, (1, 1.5)), zorder=1.4)
    ax_c.set_xlim(-0.6, len(RULES) - 0.4)
    # 기준선 표지는 축 오른쪽 밖(점과 겹치지 않게)
    import matplotlib.transforms as mtr
    trc = mtr.blended_transform_factory(ax_c.transAxes, ax_c.transData)
    ax_c.text(1.02, 1.0, "random", transform=trc, ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False)
    ax_c.text(1.02, 0.7, "0.7", transform=trc, ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False)
    src_c = []
    col_c = ps.COLOR["refit"]
    for k, rule in enumerate(RULES):
        q = ev[ev.rule == rule].sort_values("target").reset_index(drop=True)
        jit = _jitter(k, len(q), 0.26)
        for j, r in q.iterrows():
            mk, scl = _region_style(r.target)
            prim = bool(r.f7_eval)
            c = col_c if prim else ps.GREY["light"]
            ax_c.plot([k + jit[j]], [r.vtot_ratio], ls="none", marker=mk, ms=3.4 * scl, mfc=c, mec=c, mew=0.4,
                      alpha=0.75 if prim else 0.9, zorder=3 if prim else 2.5)
            src_c.append(dict(rule=rule, target=r.target, vtot_ratio=r.vtot_ratio, primary_set=prim, x_jitter=round(jit[j], 3)))
        med = float(np.median(q.loc[q.f7_eval, "vtot_ratio"])) if q.f7_eval.any() else np.nan
        ax_c.plot([k - 0.32, k + 0.32], [med, med], color=col_c, lw=1.6, solid_capstyle="butt", zorder=4)
        src_c.append(dict(rule=rule, target="MEDIAN(primary)", vtot_ratio=med, primary_set=True, x_jitter=0.0))
    ax_c.set_xticks(range(len(RULES)))
    ax_c.set_xticklabels([RULE_TICK[r] for r in RULES])
    ax_c.tick_params(axis="x", length=0, pad=2.5)
    ax_c.set_ylabel("Variance of E estimate\n(ratio to random, n = 3)")
    # 방향 표지(d 의 'better ←' 와 같은 문법): 비 < 1 이 개선
    bt = ax_c.annotate("better ↓", xy=(1.0, 1.0), xycoords="axes fraction", xytext=(-1.0, -1.0), textcoords="offset points",
                       ha="right", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    bt.set_gid("better")

    # d: 규칙별 Δ 대 무작위(가로 포레스트), n = 10, S3
    ax_d = ps.slot_mm(fig, 84.0, 12.0, 17.0, 77.0, 38.0)
    XLIM = (-3.0, 7.6)
    ylab = [RULE_TICK[r] for r in RULES]
    yrow = np.arange(len(RULES))[::-1].astype(float)
    col_d = ps.COLOR["residual"]
    src_d = []
    for k, (rule, yr) in enumerate(zip(RULES, yrow)):
        q = sd[sd.rule == rule].sort_values("target").reset_index(drop=True)
        jit = _jitter(k, len(q), 0.30, seed=11)
        for j, r in q.iterrows():
            mk, scl = _region_style(r.target)
            v = r.blk_d_rand
            if not np.isfinite(v):
                continue
            ax_d.plot([np.clip(v, *XLIM)], [yr + jit[j]], ls="none", marker=mk, ms=3.2 * scl, mfc=col_d, mec=col_d, mew=0.4,
                      alpha=0.45, zorder=2.6)
            src_d.append(dict(rule=rule, row=f"target:{r.target}", estimate=v, ci_lo=r.blk_d_rand_lo, ci_hi=r.blk_d_rand_hi,
                              ci_kind="block", y_jitter=round(jit[j], 3)))
    for layer, off, var in (("main", 0.0, "solid"), ("sub", 0.0, "dashed")):
        est, lo, hi = [], [], []
        for rule in RULES:
            m = rg[(rg.rule == rule) & (rg.layer == layer)]
            if len(m):
                est.append(float(m.delta.iloc[0])); lo.append(float(m.ci_lo.iloc[0])); hi.append(float(m.ci_hi.iloc[0]))
            else:
                est.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            # CI 종류: 4 주 지역 평균 = strat_AB4(§1.6 표). 10 하위 지역 평균 = 같은 층화 블록 부트스트랩(m1_stats.strat)을
            # 하위 지역 10개 층에 적용한 것 → 별도 표기 strat_sub10(정의는 spec.ci_kind_defs, §1.6 표에 행 추가 요청).
            src_d.append(dict(rule=rule, row=f"pooled_mean:{layer}", estimate=est[-1], ci_lo=lo[-1], ci_hi=hi[-1],
                              ci_kind="strat_AB4" if layer == "main" else "strat_sub10", y_jitter=0.0))
        o = 0.13 if layer == "main" else -0.13
        ps.forest(ax_d, ylab, est, lo, hi, method="residual", offset=o, ls=var, offscale=XLIM, ms=ps.MS["mean"],
                  set_labels=(layer == "main"), zorder=4, better_text=(layer == "main"))
    ax_d.set_xlim(*XLIM)
    ax_d.set_xlabel("ΔRMSE vs random labels (cm)")
    ax_d.set_xticks([-2, 0, 2, 4, 6])
    ax_d.xaxis.set_major_formatter(FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))

    # ------------------------------------------------------------ 범례(그림당 1개, 8항목)
    def rh(reg):
        mk, _ = ps.region_marker(reg)
        return Line2D([], [], ls="none", marker=mk, ms=3.4, mfc=MARKER_GREY, mec=MARKER_GREY, mew=0.4)
    h_rus = (rh("Russia_W"), rh("Russia_E"))
    handles = [
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc="white", mec="#000000", mew=0.8, label="k-medoid-preferred block (a, b)"),
        Line2D([], [], color=col_c, lw=1.6, solid_capstyle="butt", label="Median, primary targets (c)"),
        Line2D([], [], color=col_d, lw=ps.LW["ci_forest"], marker="D", ms=ps.MS["mean"], mfc=col_d, mec=col_d, label="Pooled mean, 4 main regions (d)"),
        Line2D([], [], color=col_d, lw=ps.LW["ci_forest"], ls=ps.VARIANT_LS["dashed"], marker="D", ms=ps.MS["mean"], mfc=col_d, mec=col_d,
               label="Pooled mean, 10 subregions (d)"),
    ]
    reg_handles = [rh("Alaska"), rh("Canada"), rh("Lena"), h_rus]
    reg_labels = ["Alaska (AL-1–6)", "Canada, CA-2, CA-3", "Lena, LE-1, LE-2", "Russia W, E"]
    for hh, ll in zip(reg_handles[:3], reg_labels[:3]):
        hh.set_label(ll)
    allh = handles + reg_handles
    alll = [h.get_label() for h in handles] + reg_labels
    lg = fig.legend(allh, alll, loc="center", ncol=4, bbox_to_anchor=ps.mm_to_fig(fig, 6.0, 55.5) + ps.mm_to_fig(fig, 168.0, 9.0),
                    bbox_transform=fig.transFigure, fontsize=ps.FS["legend"], frameon=False, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                    columnspacing=1.6, handlelength=2.2)
    lg.set_gid("paper_legend")

    ps.label_panels([ax_a, ax_b, ax_c, ax_d], "abcd")

    # ------------------------------------------------------------ 캡션 수치
    s29 = S29.set_index("target")
    def harm(t):
        r = s29.loc[t]
        return (f"{int(round(r.frac_harm_S3 * r.n_blocks_unique))} of {int(r.n_blocks_unique)} blocks harmful "
                f"({100 * r.frac_harm_S3:.0f} %, 95 % CI {100 * r.frac_harm_S3_lo:.0f}–{100 * r.frac_harm_S3_hi:.0f} %)")
    nprim = int(ev[ev.rule == RULES[0]].f7_eval.sum()); nsec = int((~ev[ev.rule == RULES[0]].f7_eval).sum())
    # F7 판정(사전 등록 주 정의: D-optimal V_tot 비 ≤ 0.7 대상 수, n = 3, 주 집합). b3_f7.csv 주 행에서 읽는다.
    f7 = F7[(F7.rule == "doptimal") & (F7.n == 3) & (F7.metric == "vtot_ratio<=0.7|f7_eval")]
    if len(f7) != 1:
        raise MissingData("b3_f7 doptimal n=3 vtot_ratio<=0.7|f7_eval 행")
    f7 = f7.iloc[0]
    f7_ok = int(f7.n_pass) >= float(f7.n_need)
    dopt = rg[(rg.rule == "doptimal") & (rg.layer == "main")]
    if len(dopt) != 1:
        raise MissingData("b3_region D-optimal n=10 S3 main 평균 행")
    dopt = dopt.iloc[0]
    f7_sentence = (f"F7 {'supported' if f7_ok else 'not supported'}: D-optimal cut V_tot by ≥ 30 % in {int(f7.n_pass)} of "
                   f"{int(f7.n_targets)} primary targets (criterion {int(f7.n_need)}). ")
    caption = dict(
        definition=("Fig. 5 | Label placement. Block label value is ΔRMSE on B blocks (method minus physics anchor, cm; negative is better) "
                    "when only one A block is labelled (E shrinkage κ = 10 plus residual ML, λ = 0.25), averaged over splits for each unique block. "
                    "Rules: k-medoid, weighted k-medoid (cluster-size weights), D-optimal (√TDD leverage), Active PPI; random is the reference."),
        statistics=("a, b: hatching, ΔRMSE > 0 by sign only, not significance; harmful fraction with Clopper-Pearson 95 % CI. "
                    "c: V_tot = between-split variance plus block-bootstrap variance of E (100 resamples of pooled A blocks, rule re-applied). "
                    + f7_sentence +
                    "d: target points, block-bootstrap estimates (1,000 resamples, paired); pooled means, stratified block-bootstrap 95 % CIs "
                    "with the 4 main regions or the 10 subregions as strata."),
        panels=(f"a, Canada: {harm('Canada')}. b, Lena Delta: {harm('Lena')}. Polar stereographic, true scale 70°N; "
                "circle area ∝ labelled cells (clipped at 27 and 200); circles displaced to avoid overlap "
                f"({n_off} moved, ≤ {max_off:.1f} mm), grey lines join {n_lead} to true centroids (dots). "
                "Symlog colour scale, linear within ±2 cm. "
                f"Circles at labelled-cell centroids; {n_water['Lena']} in b fall on delta channels "
                "(coastline: Natural Earth 1:50 m in a, 1:10 m in b). "
                "Black outline: block receiving ≥ 0.5 of 10 k-medoid labels (mean). "
                f"c, n = 3; blue, {nprim} targets with block-level bootstrap (primary set); grey, {nsec} targets with cell-level fallback; "
                "AL-1 and AL-6 lack V_tot. Dotted line, pre-specified 0.7 criterion. "
                "d, n = 10, 14 targets; smaller markers, subregions; D-optimal minus random, main regions "
                f"{'+' if dopt.delta > 0 else ''}{ps.fmt_num(dopt.delta, 2)} [{ps.fmt_num(dopt.ci_lo, 2)}, {ps.fmt_num(dopt.ci_hi, 2)}] cm."),
        data=("Source: data/processed/h3/h29b_blocks.csv, h29b_summary.csv; data/processed/h4/b3_evar.csv, b3_f7.csv, "
              "b3_summary.csv, b3_region.csv. Per-target values in Supplementary Table 3."),
    )
    spec = dict(intent=("라벨 배치가 수보다 중요하다. 단일 블록 라벨은 캐나다 27 %·레나 56 % 블록에서 해로웠다(H29 고유 블록 평균, 확정). "
                        "F7 기각: D-optimal 의 E 추정 분산(V_tot) 30 % 감소는 주 8 대상 중 2개(기준 6). 캡션 판정 문장은 b3_f7.csv 에서 계산."),
                panels=4, ci_kinds=dict(a="none(부호 해칭)", b="none(부호 해칭)", c="none(개별 점)", d="block(대상 점)·strat_AB4(4 주 지역 평균)·strat_sub10(10 하위 지역 평균)"),
                ci_kind_defs=dict(strat_sub10="10 하위 지역 평균의 층화 블록 부트스트랩(m1_stats.strat, 층 = 하위 지역 10개). "
                                              "strat_AB4 와 같은 함수, 층 집합만 다름. §1.6 표에 행 추가 필요"),
                assets=["data/processed/h3/h29b_blocks.csv", "data/processed/h3/h29b_summary.csv", "data/processed/h4/b3_evar.csv",
                        "data/processed/h4/b3_f7.csv", "data/processed/h4/b3_summary.csv", "data/processed/h4/b3_region.csv"],
                deviations=["색 정규화 symlog(선형 ±2 cm, ±30 cm): 선형 TwoSlopeNorm 이면 |Δ| ≤ 2 cm 블록이 흰색에 가까움",
                            "원 면적 clip(0.45 n, 12, 90) pt²: 1셀 블록 색·해칭 판독 보장",
                            "h29b 고유 블록(33·18) 사용: 스펙의 61·33 은 행 수(감사 조치 6)",
                            "c 회색 = 셀 단위 부트스트랩 대체 대상(f7_eval False); rep_degenerate 는 D-optimal 전 대상이 True 라 표시 기준으로 쓰지 않음",
                            "k-중심 선호는 테두리만(검토 round 1)",
                            f"표시 위치 이동(_repel, 검토 round 2): 모든 원 쌍의 바깥 테두리 사이 ≥ {GAP_MM} mm, {n_off}개·최대 {max_off:.2f} mm, "
                            f"참 위치가 원 밖인 {n_lead}개는 점 + 회색 선(source_data offset_mm·leader)",
                            "해칭 12 사선(간격 약 0.6 mm), 원 면적 하한 12 pt²: 최소 원에도 사선 2개 이상(검토 round 2)",
                            "b 해안선 NE 10 m(삼각주 수로 표현), a 는 50 m. 원 위치 = 라벨 셀 중심, 육지 다각형 밖 중심은 캡션 문장(source_data in_water)",
                            "c·d 규칙 범주 라벨 동일 문자열·줄바꿈(Weighted/k-medoid, Active/PPI 두 줄, 검토 round 2)",
                            "캔버스 180 × 134 mm(§2.5 표 128 mm 대비 +6 mm): c 의 두 줄 범주 라벨과 두 줄 y 축 제목 아래 여백 12 mm, "
                            "지도 위 패널 문자 여백 5 mm 를 둔 결과. 200 mm 상한 안"],
                placeholders=[])
    src_ab = B[["target", "block", "lat", "lon", "n_cells", "n_splits", "value_S3", "d_S3", "harmful", "kmedoid_freq", "kmedoid_pref",
                "offset_mm", "leader", "in_water"]].copy()
    return save_paper(fig, "Fig5_label_placement", caption=caption, spec=spec, draft=draft,
                      sources={"ab": src_ab, "c": pd.DataFrame(src_c), "d": pd.DataFrame(src_d)})
