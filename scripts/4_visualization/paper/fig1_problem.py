"""Fig 1 (v2). 문제 정의와 자료 범위. 스펙: figures/figure_spec.json display_items 'Fig 1' 의 f3_v2 블록.

패널
  a  범북극 지도(극 입체, 참척 70°N): 0.5° 블록의 1 km 위치 수에 면적 비례하는 원. v3 라벨 셀은 채운 원, LGD 추가 대상 셀(약관 확인분)은
     검은 윤곽 원. NAtlantic 은 약관 확인분 판에서 부적격이라 대상으로 표시하지 않고 직접 표지(SI only)만 둔다. 약관 미확인 셀은 그리지 않는다.
     레나 지도 영역 파선 사각형, 티베트(LGD) 경위도 삽도, 배경 CCI PFR 2단계 회색.
  b  지역별 z = ln(ALT/√TDD)(셀 점 ≤ 400, 사분위 상자, 평균) 와 대상에 적용된 ln E0 세로 눈금. 위 축 E.
  c  설계 도식(지역 홀드아웃 + 100 km 버퍼, A/B 블록 절반, 라벨 추출, 채점, 방법 계열).
  d  원천 풀의 지역 구성(누적 막대, 회색 단계 + 무늬)과 E0.
자료: data/processed/paper_figs/fig1_*.csv(v2_data.py). 이 모듈은 PFR 배경 래스터 외에 계산하지 않는다.
실행: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/fig1_problem.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402
import matplotlib.path as mpath                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle  # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps                                                  # noqa: E402
import _common as C                                                      # noqa: E402

NAME = "Fig1_problem"
SPEC_ID = "paper_fig1_problem"
LON0 = -100.0                    # 아래쪽 경선
LAT_MIN = 50.0
FILL_V3 = ps.COLOR["refit"]
S_K, S_MIN, S_MAX = 2.6, 2.6, 200.0   # 원 면적(pt²) = clip(k·위치 수)
SIZES = (1, 10, 50)
REGION_LL = {  # 지역 이름 직접 표지 기준점(경도, 위도). 셀 무리 옆 빈 곳
    "Alaska": ("Alaska (ref.)", (-163.0, 56.0)), "Canada": ("Canada", (-128.0, 55.0)), "Lena": ("Lena", (125.0, 78.2)),
    "Russia_W": ("Russia W", (72.0, 61.0)), "Russia_E": ("Russia E", (150.0, 54.0)), "Russia_C": ("Russia C", (100.0, 58.5)),
    "Greenland": ("Greenland", (-40.0, 76.5)),
}
NATL_LL = (-8.0, 66.0)
COMP = [("share_alaska", "Alaska", "#4d4d4d", None), ("share_lena", "Lena", "#a6a6a6", None),
        ("share_canada", "Canada", "#e0e0e0", "////"), ("share_other", "Other (< 2 %)", "white", "....")]


def _pfr_cache():
    import netCDF4 as nc
    with nc.Dataset(C.PROC / "cci_pfr_mean_1997_2021.nc") as f:
        return f["lat"][:].data, f["lon"][:].data, f["pfr_mean"][:].filled(np.nan)


def _pfr_image(ax, proj, npx=900):
    """PFR 2단계 회색(연속 ≥ 90 %, 불연속 50–90 %)을 투영 격자에 최근접 표본화(래스터)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap
    lat, lon, M = _pfr_cache()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, npx); ys = np.linspace(y1, y0, npx)
    X, Y = np.meshgrid(xs, ys)
    ll = ccrs.PlateCarree().transform_points(proj, X, Y)
    LON, LAT = ll[..., 0], ll[..., 1]
    i = np.clip(np.round((LAT - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((LON - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    v = M[i, j]
    v[(LAT < lat.min() - 0.1) | (LAT > lat.max() + 0.1)] = np.nan
    cls = np.full(v.shape, np.nan)
    cls[(v >= 50) & (v < 90)] = 0; cls[v >= 90] = 1
    cm = ListedColormap([ps.GREY["pf_disc"], ps.GREY["pf_cont"]])
    ax.imshow(np.ma.masked_invalid(cls), extent=(x0, x1, y0, y1), origin="upper", cmap=cm, vmin=0, vmax=1, interpolation="nearest",
              transform=proj, zorder=0.3, rasterized=True)


def _circles(ax, q, filled: bool, zorder=3):
    import cartopy.crs as ccrs
    s = np.clip(S_K * q.n_loc_1km.values, S_MIN, S_MAX)
    o = np.argsort(-s)
    if filled:
        ax.scatter(q.lon.values[o], q.lat.values[o], s=s[o], color=FILL_V3, alpha=0.55, linewidths=0.3, edgecolors=ps.GREY["edge"],
                   transform=ccrs.PlateCarree(), zorder=zorder)
    else:
        ax.scatter(q.lon.values[o], q.lat.values[o], s=s[o], facecolors="none", linewidths=0.7, edgecolors="#000000",
                   transform=ccrs.PlateCarree(), zorder=zorder + 0.5)


def panel_map(fig, rect):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    Bk = V.rd("fig1_blocks")
    M = V.rd("fig1_meta").iloc[0]
    proj = ccrs.NorthPolarStereo(central_longitude=LON0, true_scale_latitude=70)
    ax = ps.axes_mm(fig, *rect, projection=proj)
    ax.set_extent((-180, 180, LAT_MIN, 90), crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 200)
    circ = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax.set_boundary(circ, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=ps.GREY["land"], edgecolor="none", zorder=0)
    _pfr_image(ax, proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=ps.GREY["edge"], lw=0.35, zorder=1)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, lw=0.35, color=ps.GREY["grid"], alpha=0.8, zorder=0.5,
                 xlocs=np.arange(-180, 181, 30), ylocs=[60, 70, 80])
    ax.spines["geo"].set_linewidth(0.5)
    v3 = Bk[(Bk.kind == "v3") & (Bk.region != "Tibet")]
    _circles(ax, v3, True)
    add = Bk[(Bk.kind == "lgd_added") & (Bk.region != "Tibet_LGD")]
    _circles(ax, add, False)
    # North Atlantic 의 LGD 추가 셀은 그리지 않는다(약관 확인분 판에서 부적격, SI 전체 판에만). 직접 표지만 둔다
    P = proj.transform_points(ccrs.PlateCarree(), Bk.lon.values, Bk.lat.values)
    Bk = Bk.assign(px=P[:, 0], py=P[:, 1])
    pc = ccrs.PlateCarree()
    ptr = proj._as_mpl_transform(ax)
    for reg, (txt, (lo_, la_)) in REGION_LL.items():
        L0 = proj.transform_points(pc, np.array([lo_]), np.array([la_]))[0]
        q = Bk[(Bk.region == reg) & (Bk.kind == "v3")]
        k = int(np.argmin(np.hypot(q.px.values - L0[0], q.py.values - L0[1])))
        t = ax.annotate(txt, xy=(q.px.values[k], q.py.values[k]), xycoords=ptr, xytext=(L0[0], L0[1]), textcoords=ptr, ha="center",
                        va="center", fontsize=ps.FS["annot"], zorder=6, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", lw=0.45, color=ps.GREY["text2"], shrinkA=0, shrinkB=1.5))
        t.set_gid("region_label")
    # NAtlantic: SI 전용 직접 표지(범례에 넣지 않는다)
    t = ax.text(*NATL_LL, "North Atlantic: SI only\n(licence check pending)", transform=pc, ha="center", va="center",
                fontsize=ps.FS["annot"], color=ps.GREY["text2"], zorder=6, linespacing=1.0,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    t.set_gid("natl_si")
    # 레나 지도 영역
    lo = np.r_[np.linspace(M.lena_lon0, M.lena_lon1, 20), np.full(8, M.lena_lon1), np.linspace(M.lena_lon1, M.lena_lon0, 20), np.full(8, M.lena_lon0)]
    la = np.r_[np.full(20, M.lena_lat0), np.linspace(M.lena_lat0, M.lena_lat1, 8), np.full(20, M.lena_lat1), np.linspace(M.lena_lat1, M.lena_lat0, 8)]
    import matplotlib.patheffects as pe
    ln = ax.plot(lo, la, color="#000000", lw=0.8, ls=(0, (2.0, 1.2)), transform=ccrs.PlateCarree(), zorder=5)[0]
    ln.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    ps.scale_bar(ax, 1000, loc=(0.56, 0.10))
    ax.set_gid("map")
    return ax


def panel_tibet(fig, rect):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    Bk = V.rd("fig1_blocks")
    ax = ps.axes_mm(fig, *rect, projection=ccrs.PlateCarree())
    ax.set_extent((77, 103, 29.5, 40.5), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=ps.GREY["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=ps.GREY["edge"], lw=0.35, zorder=1)
    _circles(ax, Bk[(Bk.kind == "lgd_added") & (Bk.region == "Tibet_LGD")], False)
    ax.spines["geo"].set_linewidth(0.5)
    ax.text(0.03, 0.95, "Tibet (LGD)", transform=ax.transAxes, ha="left", va="top", fontsize=ps.FS["annot"])
    ax.set_gid("inset")
    return ax


def panel_z(fig, rect):
    Z = V.rd("fig1_z_points")
    S = V.rd("fig1_z_summary")
    ax = ps.axes_mm(fig, *rect)
    n = len(S)
    ys = np.arange(n)[::-1].astype(float)
    rng = np.random.default_rng(7)
    for y, r in zip(ys, S.itertuples()):
        z = Z[Z.row == r.row].z.values
        jit = rng.uniform(-0.28, 0.28, len(z))
        ax.scatter(z, y + jit, s=ps.MS["point"] ** 2 * 0.55, color=ps.COLOR["direct"], alpha=0.25, lw=0, zorder=2, rasterized=True)
        if r.n_cells >= 10:
            ax.add_patch(Rectangle((r.z_q25, y - 0.3), r.z_q75 - r.z_q25, 0.6, fc="none", ec="#000000", lw=0.6, zorder=3))
        ax.plot([r.z_mean], [y], "o", ms=ps.MS["main"], color="#000000", mec="white", mew=0.4, zorder=4)
        if np.isfinite(r.lnE0):
            ax.plot([r.lnE0, r.lnE0], [y - 0.3, y + 0.3], color=ps.COLOR["phys"], lw=1.4, solid_capstyle="butt", zorder=4.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.label.replace('Alaska (reference)', 'Alaska, ref.')} ({r.n_cells:,})" for r in S.itertuples()])
    r0 = S.iloc[0]
    t = ax.annotate("ln E0 of each target", xy=(r0.lnE0, ys[0] + 0.3), xytext=(0, 1.5), textcoords="offset points", ha="left", va="bottom",
                    fontsize=ps.FS["annot"], color=ps.GREY["text2"], annotation_clip=False)
    t.set_gid("e0_label")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlim(-1.6, 3.6)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ax.set_xlabel("z = ln(ALT / √TDD)")
    top = ax.secondary_xaxis("top", functions=(np.exp, np.log))
    top.set_xticks([0.5, 1, 2, 5, 10, 20])
    top.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    top.tick_params(axis="x", length=2.0, labelsize=ps.FS["tick"])
    top.set_xlabel("E = exp(z)  (cm (°C d)$^{−1/2}$)", labelpad=2.0)
    for sp in top.spines.values():
        sp.set_linewidth(ps.LW["axis"])
    # 구획선: v3 대상 / Alaska / LGD
    ax.axhline(ys[6] + 0.5, color=ps.GREY["light"], lw=0.5, zorder=0.5)
    ax.axhline(ys[7] + 0.5, color=ps.GREY["light"], lw=0.5, zorder=0.5)
    return ax, S


def panel_source(fig, rect, ecol_x):
    D = V.rd("fig1_source")
    ax = ps.axes_mm(fig, *rect)
    order = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland", "Alaska"]
    D = D.set_index("target").loc[order].reset_index()
    ys = np.arange(len(D))[::-1].astype(float)
    for y, r in zip(ys, D.itertuples()):
        left = 0.0
        for col, lab, fc, hatch in COMP:
            w = float(getattr(r, col))
            if w > 0:
                ax.barh(y, w, left=left, height=0.62, color=fc, edgecolor="#000000", lw=0.4, hatch=hatch, zorder=2)
            left += w
        a = float(r.share_alaska)
        if a > 0.2:
            ax.text(a / 2, y, f"{100 * a:.1f} %", ha="center", va="center", fontsize=ps.FS["annot"], color="white", zorder=3)
        else:
            ax.text(0.01 + float(r.share_lena) / 2, y, f"Lena {100 * float(r.share_lena):.1f} %", ha="center", va="center",
                    fontsize=ps.FS["annot"], color="#000000", zorder=3)
    ax.axhline(ys[-1] + 0.5, color=ps.GREY["light"], lw=0.5, zorder=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([("Alaska (x)" if t == "Alaska" else ps.region_name(t)) for t in D.target])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(D) + 0.9)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Share of source cells")
    # E0 글자 열
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(ecol_x, len(D) - 0.55, "E0", transform=tr, ha="center", va="bottom", fontsize=ps.FS["tick"], clip_on=False)
    for y, r in zip(ys, D.itertuples()):
        ax.text(ecol_x, y, f"{r.E0:.2f}", transform=tr, ha="center", va="center", fontsize=ps.FS["annot"], clip_on=False)
    # 범주 직접 표지(막대 위 두 줄, 글자 폭으로 간격)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for k, grp in enumerate((COMP[:2], COMP[2:])):
        y0 = len(D) + 0.62 - 0.62 * k
        x = 0.0
        for col, lab, fc, hatch in grp:
            ax.add_patch(Rectangle((x, y0 - 0.2), 0.05, 0.4, facecolor=fc, edgecolor="#000000", lw=0.4, hatch=hatch, clip_on=False,
                                   zorder=3))
            t = ax.text(x + 0.065, y0, lab, ha="left", va="center", fontsize=ps.FS["annot"], clip_on=False)
            x = max(0.42, inv.transform((t.get_window_extent(rend).x1, 0))[0] + 0.06)
    return ax, D


def panel_schematic(fig, rect):
    """c: 설계 도식. 좌표 = mm(축 상자). 세 열: 지역 홀드아웃, A/B 블록 절반, 방법과 채점. 화살표는 실선 한 종류."""
    ax = ps.axes_mm(fig, *rect)
    W, H = rect[2], rect[3]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fs = ps.FS["annot"]
    rng = np.random.default_rng(3)
    ytitle = H - 0.5
    # 1) 지역 홀드아웃(x 0–33)
    ax.text(0.5, ytitle, "1  Region holdout", ha="left", va="top", fontsize=ps.FS["base"])
    cx, cy, R = 15.5, 26.0, 6.0
    ax.add_patch(Circle((cx, cy), R + 4.0, fc="none", ec="#000000", lw=0.6, ls=(0, (2.0, 1.2)), zorder=2))
    ax.add_patch(Circle((cx, cy), R, fc="#d5e0ec", ec=ps.COLOR["refit"], lw=0.8, zorder=2))
    ax.text(cx, cy, "target\nregion", ha="center", va="center", fontsize=fs, linespacing=1.0, zorder=4)
    pts = [np.c_[ox + rng.normal(0, 1.4, n), oy + rng.normal(0, 1.2, n)] for ox, oy, n in ((3.0, 35.5, 12), (29.0, 36.0, 12), (30.5, 21.0, 10),
                                                                                         (2.0, 22.0, 8))]
    P = np.vstack(pts)
    P = P[np.hypot(P[:, 0] - cx, P[:, 1] - cy) > R + 4.4]
    ax.scatter(P[:, 0], P[:, 1], s=3.0, color=ps.COLOR["direct"], lw=0, zorder=3)
    ang = np.array([0.5, 1.7, 2.9, 4.2, 5.5])
    ax.scatter(cx + (R + 2.0) * np.cos(ang), cy + (R + 2.0) * np.sin(ang), s=7, marker="x", color=ps.GREY["mid"], lw=0.6, zorder=3)
    ax.text(cx, 13.2, "100 km buffer: excluded (×)", ha="center", va="center", fontsize=fs)
    ax.text(0.5, 8.0, "Source: all other labelled cells", ha="left", va="center", fontsize=fs)
    ax.text(0.5, 3.5, "E0 = Σ s y / Σ s², s = √TDD", ha="left", va="center", fontsize=fs)
    # 2) A/B 블록 절반(x 37–72)
    x0 = 37.0
    ax.text(x0, ytitle, "2  A/B split of 0.5° blocks", ha="left", va="top", fontsize=ps.FS["base"])
    pattern = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
    sq, gp = 5.0, 0.6
    gx0, gy0 = x0 + 0.5, 20.5
    for i in range(3):
        for j in range(4):
            a_ = pattern[i, j] == 1
            xx, yy = gx0 + j * (sq + gp), gy0 + (2 - i) * (sq + gp)
            ax.add_patch(Rectangle((xx, yy), sq, sq, fc="white" if a_ else "#bdbdbd", ec="#000000", lw=0.5, zorder=2))
            cxy = np.c_[xx + rng.uniform(0.8, sq - 0.8, 4), yy + rng.uniform(0.8, sq - 0.8, 4)]
            if a_:
                ax.scatter(cxy[:, 0], cxy[:, 1], s=2.2, color=ps.GREY["mid"], lw=0, zorder=3)
                ax.scatter(cxy[:1, 0], cxy[:1, 1], s=7.0, color="#000000", lw=0, zorder=3.5)
            else:
                ax.scatter(cxy[:, 0], cxy[:, 1], s=2.2, color="#000000", lw=0, zorder=3)
    gx1 = gx0 + 4 * (sq + gp) - gp
    ky = gy0 - 3.2
    ax.add_patch(Rectangle((x0 + 0.5, ky - 1.2), 2.4, 2.4, fc="white", ec="#000000", lw=0.5))
    ax.text(x0 + 3.6, ky, "A: labels drawn", ha="left", va="center", fontsize=fs)
    ax.add_patch(Rectangle((x0 + 22.5, ky - 1.2), 2.4, 2.4, fc="#bdbdbd", ec="#000000", lw=0.5))
    ax.text(x0 + 25.6, ky, "B: scored", ha="left", va="center", fontsize=fs)
    ax.scatter([x0 + 1.7], [ky - 4.0], s=7.0, color="#000000", lw=0)
    ax.text(x0 + 3.6, ky - 4.0, "drawn label", ha="left", va="center", fontsize=fs)
    ax.scatter([x0 + 18.9], [ky - 4.0], s=2.2, color=ps.GREY["mid"], lw=0)
    ax.text(x0 + 20.3, ky - 4.0, "other A cells", ha="left", va="center", fontsize=fs)
    ax.text(x0 + 0.5, 8.0, "5 splits; n = 0, 3, 10, 40, 160,", ha="left", va="center", fontsize=fs)
    ax.text(x0 + 0.5, 3.5, "320, 1,000 and all labels", ha="left", va="center", fontsize=fs)
    # 3) 방법과 채점(x 75–110)
    x1 = 75.5
    ax.text(x1, ytitle, "3  Methods and score", ha="left", va="top", fontsize=ps.FS["base"])
    rows = [("P0", "E0 √TDD (n = 0)"), ("P1", "E$_n$ √TDD"), ("D0", "direct CatBoost"), ("R0, R1", "P0, P1 + residual ML")]
    for k, (a_, b_) in enumerate(rows):
        yy = H - 7.5 - k * 4.4
        ax.text(x1, yy, a_, ha="left", va="center", fontsize=fs, fontweight="bold")
        ax.text(x1 + 9.0, yy, b_, ha="left", va="center", fontsize=fs)
    ax.text(x1, 17.0, "E$_n$ = (n E$_{ls}$ + 10 E0) / (n + 10)", ha="left", va="center", fontsize=fs)
    ax.text(x1, 12.5, "Draws: cell-random; block-spread (L43)", ha="left", va="center", fontsize=fs)
    ax.text(x1, 8.0, "Score on B cells:", ha="left", va="center", fontsize=fs)
    ax.text(x1, 3.5, "ΔRMSE = RMSE − RMSE(P0)", ha="left", va="center", fontsize=fs)
    kw = dict(arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#000000", zorder=5)
    ax.add_patch(FancyArrowPatch((cx + R + 4.3, cy + 2.0), (gx0 - 0.6, cy + 2.0), **kw))
    ax.add_patch(FancyArrowPatch((gx1 + 0.6, cy + 2.0), (x1 - 0.8, cy + 2.0), **kw))
    ax.set_gid("schematic")
    return ax


def build():
    ps.use_paper()
    W, H = 180.0, 168.0
    fig = ps.paper_figure(W, H)
    ax_a = panel_map(fig, (3.0, 64.0, 88.0, 88.0))
    ax_t = panel_tibet(fig, (1.5, 57.0, 24.0, 12.0))                    # noqa: F841
    ax_b, S = panel_z(fig, (122.0, 74.0, 54.0, 64.0))
    ax_c = panel_schematic(fig, (2.0, 4.0, 110.0, 46.0))
    ax_d, D = panel_source(fig, (132.0, 12.0, 36.0, 40.0), ecol_x=1.13)
    ps.panel_label(ax_a, "a", dx_mm=-1.5, dy_mm=0.5)
    ps.panel_label(ax_b, "b", dx_mm=-28.0, dy_mm=8.0)
    ps.panel_label(ax_c, "c", dx_mm=-1.0, dy_mm=0.5)
    ps.panel_label(ax_d, "d", dx_mm=-20.0, dy_mm=0.5)
    # 범례(하나)
    h = [Line2D([], [], ls="none", marker="o", ms=float(np.sqrt(np.clip(S_K * v, S_MIN, S_MAX))), mfc=FILL_V3, mec=ps.GREY["edge"], mew=0.3,
                alpha=0.55, label=f"{v} location{'s' if v > 1 else ''} (1 km) per 0.5° block") for v in SIZES]
    h += [Line2D([], [], ls="none", marker="o", ms=5.0, mfc=FILL_V3, mec=ps.GREY["edge"], mew=0.3, alpha=0.55, label="v3 label cells"),
          Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none", mec="#000000", mew=0.7, label="LGD cells added (licence verified; same size scale)"),
          Line2D([], [], color="#000000", lw=0.8, ls=(0, (2.0, 1.2)), label="Lena map domain"),
          Rectangle((0, 0), 1, 1, fc=ps.GREY["pf_cont"], ec="none", label="Permafrost ≥ 90 %"),
          Rectangle((0, 0), 1, 1, fc=ps.GREY["pf_disc"], ec="none", label="Permafrost 50–90 %")]
    ps.legend_below(fig, h, (2.0, H - 11.5, W - 4.0, 10.0), ncol=4)
    M = V.rd("fig1_meta").iloc[0]
    nd = V.rd("fig1_not_drawn")
    nd_txt = ", ".join(f"{'North Atlantic' if r == 'NAtlantic' else ps.region_name(r)} {int(n)}" for r, n in zip(nd.region, nd.n_cells_not_drawn))
    share = D[D.target.isin(["Lena", "Canada", "Russia_W", "Russia_E"])].share_alaska
    cap = dict(
        definition=("z = ln(ALT/√TDD), with TDD the ERA5-Land air thawing index (°C d, 2015–2020 climatology); E = exp(z) is the "
                    "Stefan coefficient. E0 is the least-squares slope of ALT on √TDD through the origin over the source pool; P0 "
                    "predicts E0 √TDD. Circle area is proportional to the number of 1 km label locations in each 0.5° block."),
        statistics=("Descriptive; no tests. b, boxes span the quartiles, black dots are means and points are up to 400 cells per row "
                    "(random subsample)."),
        panels=(f"a, labelled cells of the v3 table ({int(M.n_v3_cells):,} cells) and cells added by LGD in the licence-verified "
                f"edition ({int(M.n_lgd_added_drawn)} cells; Tibet in the inset). Cells without verified licence are not drawn "
                f"({nd_txt}). North Atlantic is ineligible in the licence-verified edition; its {int(M.n_natl_si)} added cells are not "
                "drawn and appear only in the SI (full edition). "
                "The dashed box is the Lena map domain (SI). Grey shading, ESA CCI permafrost extent (1997–2021 mean). "
                "b, regions in Table 1 order; LGD rows are the licence-verified target sets. c, evaluation design. "
                f"d, source-pool composition for mode x targets (Alaskan cells {100 * share.min():.1f}–{100 * share.max():.1f} % "
                "for the four main targets) and E0 in cm (°C d)^−1/2."),
        data=("v3 and v4 label tables; E0 from the LG target table (Rescale) and the LGD unit files (local). Polar stereographic "
              "projection, true scale at 70°N."),
    )
    src = dict(a_blocks=V.rd("fig1_blocks"), a_not_drawn=nd, b_summary=S, b_points=V.rd("fig1_z_points"), d=D)
    spec_extra = dict(module="fig1_problem.py", panels=4, message="region holdout transfer with Alaska-dominated E0; E differs by region",
                      edition="licence-verified (LG 7.3); NAtlantic SI only")
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Transfer design, regional Stefan coefficients and data domain.",
                     spec_extra=spec_extra)


if __name__ == "__main__":
    build()
