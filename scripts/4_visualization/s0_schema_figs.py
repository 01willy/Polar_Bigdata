"""S0 시각화: source overlap 히트맵 + 0.5° 공간블록 GroupKFold 지도.

`docs/RESEARCH_PLAN_...` §11.5 규약: 냉색(cmcrameri), 지도 우선, PNG300+PDF.
누설통제(블록이 폴드로 통째 배정)를 눈으로 확인 가능하게 한다.

실행: python scripts/4_visualization/s0_schema_figs.py [그림이름 ...]
      인자 없으면 3종 전부, 인자를 주면 해당 그림만 재생성한다.
      예: python scripts/4_visualization/s0_schema_figs.py spatial_block_folds_map
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP
from polar.fidelity import add_group_keys, spatial_block_splits

use_polar()
plt.rcParams["pdf.fonttype"] = 42   # PDF 텍스트를 편집 가능한 벡터로(래스터화 금지)
plt.rcParams["ps.fonttype"] = 42
PROC = C.PROCESSED
OUT = C.FIGURES / "s0_schema"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


# ---------------- 1. source overlap 히트맵 ----------------
def fig_source_overlap_heatmap():
    ov = pd.read_csv(PROC / "source_overlap_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(ov.values, cmap=CMAP.count, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ov.columns))); ax.set_xticklabels(ov.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ov.index))); ax.set_yticklabels(ov.index, fontsize=8)
    for i in range(len(ov.index)):
        for j in range(len(ov.columns)):
            v = ov.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 55 else "black")
    ax.set_title("자료원 쌍 셀 overlap (%)\nStefan·CCI만 clean → full source-aware 가능", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("overlap %", fontsize=8)
    save(fig, "source_overlap_heatmap")


# ---------------- 2. 0.5° 공간블록 GroupKFold 지도 ----------------
# 저채도 이산 팔레트(냉색 규약). 폴드는 명목 범주이므로 순차맵 대신 구분 가능한 6색.
# CIELAB 최소 색차 15.3, 육지(#efece6)·해양(#eaf1f4) 배경 대비 20 이상.
FOLD_COLORS = ["#2f4b6e", "#4a7c8c", "#59636b", "#7fa38a", "#9b93ac", "#aeb9c2"]
LINE = "#5a6169"     # 주석선·스케일바(무채색에 가까운 청회)
TXT = "#444444"      # 값·주석 텍스트
BLOCK_DEG_ = 0.5

# 확대 패널 창: 알래스카 북사면(달턴 도로 축) 0.5° 블록 6×2칸
ZOOM = ("확대", -151.0, -148.0, 68.5, 69.5)


def _darken(hex_color, f=0.55):
    """같은 색상을 어둡게(점 테두리용). 색 부호화는 유지하면서 밝은 폴드의 대비를 확보."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def _scalebar_km(ax, km, frac=(0.055, 0.06), color=LINE, fontsize=8.5):
    """투영 좌표(m) 기반 축척 막대. 상자·화살표 없이 선 하나와 라벨만.

    반환한 아티스트는 다른 라벨의 겹침 회피에 장애물로 쓸 수 있다.
    """
    x0, x1, y0, y1 = ax.get_extent()
    x = x0 + frac[0] * (x1 - x0)
    y = y0 + frac[1] * (y1 - y0)
    ln, = ax.plot([x, x + km * 1000.0], [y, y], color=color, lw=1.4,
                  solid_capstyle="butt", zorder=6, transform=ax.projection)
    tx = ax.text(x + km * 500.0, y + 0.018 * (y1 - y0), f"{km:g} km", ha="center",
                 va="bottom", fontsize=fontsize, color=color, zorder=6,
                 transform=ax.projection)
    return ln, tx


def _block_cells(df, fold_of):
    """관측이 존재하는 0.5° 블록의 좌하단 좌표와 폴드 번호."""
    lo = np.floor(df.lon.values / BLOCK_DEG_) * BLOCK_DEG_
    la = np.floor(df.lat.values / BLOCK_DEG_) * BLOCK_DEG_
    cells = pd.DataFrame({"lon0": lo, "lat0": la, "fold": fold_of})
    return cells.groupby(["lon0", "lat0"], as_index=False).agg(
        fold=("fold", "first"), n=("fold", "size"))


def _draw_cells(ax, cells, alpha=1.0, edge="white", lw=0.35, zorder=3):
    """0.5° 블록을 폴드 색으로 채운 사각형으로 그린다(경위도 격자에 정확히 정렬)."""
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    for r in cells.itertuples():
        ax.add_patch(mpatches.Rectangle(
            (r.lon0, r.lat0), BLOCK_DEG_, BLOCK_DEG_,
            facecolor=FOLD_COLORS[int(r.fold) % len(FOLD_COLORS)],
            edgecolor=edge, linewidth=lw, alpha=alpha, zorder=zorder,
            transform=ccrs.PlateCarree()))


def _draw_bbox(ax, extent, label=None, color="#2f3b47", lw=1.0):
    """다른 패널의 범위를 본 지도에 얇은 사각형으로 표시(색칠 상자 없음)."""
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    _, lo0, lo1, la0, la1 = extent
    ax.add_patch(mpatches.Rectangle((lo0, la0), lo1 - lo0, la1 - la0,
                                    fill=False, edgecolor=color, linewidth=lw,
                                    zorder=7, transform=ccrs.PlateCarree()))
    if label:  # 투영으로 사각형이 기울 수 있어 투영좌표 좌상단 바깥에 라벨을 둔다
        import matplotlib.patheffects as pe
        from polar.geomap import to_proj
        xs, ys = to_proj(ax, [lo0, lo1, lo1, lo0], [la0, la0, la1, la1])
        ax.annotate(label, xy=(float(np.min(xs)), float(np.max(ys))),
                    xycoords=ax.projection._as_mpl_transform(ax),
                    xytext=(-3, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=color, zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def _draw_block_grid(ax, extent, color="#79818a", lw=0.7):
    """0.5° 블록 경계를 실제 선으로 그린다. '같은 칸=같은 폴드'를 눈으로 확인 가능하게."""
    import cartopy.crs as ccrs
    _, lo0, lo1, la0, la1 = extent
    for x in np.arange(lo0, lo1 + 1e-9, BLOCK_DEG_):
        ax.plot([x, x], [la0, la1], color=color, lw=lw, zorder=2,
                transform=ccrs.PlateCarree())
    for y in np.arange(la0, la1 + 1e-9, BLOCK_DEG_):
        ax.plot([lo0, lo1], [y, y], color=color, lw=lw, zorder=2,
                transform=ccrs.PlateCarree())


def _panel_labels(fig, items, y, dx_name=0.0255):
    """패널 라벨을 그림 좌표의 공통 y에 배치한다(패널 높이를 조정해도 (a)(b)(c)가 한 줄).

    식별자는 볼드, 설명어는 regular. 그림 내부 제목은 두지 않는다.
    """
    for ax, letter, name in items:
        x = ax.get_position().x0
        fig.text(x, y, f"({letter})", fontsize=11, fontweight="bold",
                 color="#1f2933", ha="left", va="bottom")
        fig.text(x + dx_name, y, name, fontsize=10, color="#1f2933",
                 ha="left", va="bottom")


def _lonlat_to_proj(ax, lon, lat):
    """경위도 → 투영 좌표(m). transData가 실제 mpl 변환이라 bbox 계산에 그대로 쓸 수 있다."""
    from polar.geomap import to_proj
    x, y = to_proj(ax, np.atleast_1d(lon), np.atleast_1d(lat))
    return np.column_stack([x, y])


def _place_cell_labels(fig, ax, cells, lon_pts, lat_pts, marker_s=8.0,
                       fontsize=7.5, obstacles=()):
    """블록별 폴드 라벨을 셀 안에서 관측 마커·축척 막대·다른 라벨과 겹치지 않게 놓는다.

    후보는 셀 안 3열(좌·우·가운데) × 7행(위→아래)이다. 행을 촘촘히 두어야 관측이
    세로로 이어진 블록에서도 빈 띠를 찾을 수 있다. 각 후보의 텍스트 bbox를 실제
    렌더러로 재고 두 가지를 구분해 센다. 하나는 실제 가림(마커 반지름·테두리·흰
    외곽선을 더한 범위 안에 마커 중심이 들어오는 경우, 축척 막대·다른 라벨과의
    교차 포함)이고, 다른 하나는 그보다 조금 넓은 여유 범위 안의 마커 수다. 앞의
    것이 0인 후보만 채택 대상이며, 여러 개면 뒤의 것이 작은 쪽을 고른다. 좌표가
    거의 겹치는 관측이 많아 후자를 실격 사유로 쓰면 배치가 불필요하게 밀린다.

    한 블록이라도 가림 0을 못 찾으면 글자를 한 단계 줄여 전체 배치를 다시 시도한다
    (크기는 모든 라벨에 동일하게 적용해 겉보기를 일정하게 유지). 마지막 단계에서도
    남으면 최소 가림 후보에 두꺼운 흰 마스크를 씌운다.
    """
    import matplotlib.patheffects as pe
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    disp = ax.transData.transform(_lonlat_to_proj(ax, np.asarray(lon_pts),
                                                  np.asarray(lat_pts)))
    HALO, MARK_LW = 1.8, 0.35
    px = fig.dpi / 72.0
    # 가림 판정: 마커 반지름(면적 s pt²) + 테두리 절반 + 흰 외곽선 절반 + 최소 여유
    hard = (np.sqrt(marker_s / np.pi) + 0.5 * MARK_LW + 0.5 * HALO + 0.3) * px
    soft = hard + 2.5 * px                          # 숨돌릴 여백(동점 정리용)
    # 라벨끼리의 최소 간격. 이웃 블록의 두 라벨이 같은 경계선에 붙으면 소속이
    # 흐려지므로 글자 크기에 준하는 간격을 강제한다.
    lab_pad = 3.4 * px
    obs = [a.get_window_extent(renderer=rend) for a in obstacles]
    halo = [pe.withStroke(linewidth=HALO, foreground="white")]
    inset = 0.045
    spots = []
    for fy in np.linspace(BLOCK_DEG_ - inset, inset, 7):
        va = ("top" if fy > BLOCK_DEG_ - inset - 1e-9 else
              "bottom" if fy < inset + 1e-9 else "center")
        for fx, ha in ((inset, "left"), (BLOCK_DEG_ - inset, "right"),
                       (0.5 * BLOCK_DEG_, "center")):
            spots.append((fx, fy, ha, va))

    def _count(b, grow):
        m = Bbox.from_extents(b.x0 - grow, b.y0 - grow, b.x1 + grow, b.y1 + grow)
        return int(((disp[:, 0] > m.x0) & (disp[:, 0] < m.x1)
                    & (disp[:, 1] > m.y0) & (disp[:, 1] < m.y1)).sum())

    def _attempt(fs):
        """폰트 fs로 전체 블록을 배치하고 (텍스트, 가림수) 목록을 돌려준다."""
        out, placed = [], []
        for r in cells.itertuples():
            label = f"폴드 {int(r.fold) + 1}"
            best_t, best_key = None, None
            for dx, dy, ha, va in spots:
                x, y = _lonlat_to_proj(ax, r.lon0 + dx, r.lat0 + dy)[0]
                t = ax.text(x, y, label, transform=ax.transData, fontsize=fs,
                            color=TXT, ha=ha, va=va, zorder=6, path_effects=halo)
                b = t.get_window_extent(renderer=rend)
                mb = Bbox.from_extents(b.x0 - hard, b.y0 - hard,
                                       b.x1 + hard, b.y1 + hard)
                lb = Bbox.from_extents(b.x0 - lab_pad, b.y0 - lab_pad,
                                       b.x1 + lab_pad, b.y1 + lab_pad)
                n = _count(b, hard)
                n += sum(bool(Bbox.intersection(mb, o)) for o in obs)
                n += sum(bool(Bbox.intersection(lb, o)) for o in placed)
                key = (n, _count(b, soft))
                if best_key is None or key < best_key:
                    if best_t is not None:
                        best_t.remove()
                    best_t, best_key = t, key
                else:
                    t.remove()
                if best_key == (0, 0):
                    break
            b = best_t.get_window_extent(renderer=rend)
            placed.append(Bbox.from_extents(b.x0 - lab_pad, b.y0 - lab_pad,
                                            b.x1 + lab_pad, b.y1 + lab_pad))
            out.append((best_t, best_key[0]))
        return out

    ladder = np.round(np.arange(fontsize, fontsize - 1.01, -0.25), 2)
    for fs in ladder:                               # 0.25pt씩만 줄여 최소한으로 축소
        res = _attempt(fs)
        if not any(n for _, n in res):
            break
        if fs > ladder[-1]:                         # 마지막 시도는 남겨 둔다
            for t, _ in res:
                t.remove()
    for t, n in res:
        if n:  # 회피 실패 시: 상자 대신 글자 윤곽을 따르는 두꺼운 흰 마스크
            t.set_path_effects([pe.withStroke(linewidth=3.6, foreground="white")])


def fig_spatial_block_folds_map():
    """0.5° 블록 단위 폴드 배정 지도. 블록 경계를 실제로 그려 '같은 블록=같은 폴드'를 노출."""
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    from polar.geomap import (make_ax, map_projection, projected_aspect,
                              circular_boundary, ALASKA, PANARCTIC)

    df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
    folds = spatial_block_splits(df)
    fold_of = np.full(len(df), -1)
    for k, (_, te) in enumerate(folds):
        fold_of[te] = k
    cells = _block_cells(df, fold_of)
    n_fold = len(folds)

    # 패널 높이를 맞추기 위해 투영 후 가로/세로 비를 width_ratios로 사용한다.
    # (b)(c)는 아래에 눈금 라벨 대역이 붙으므로, 원형 패널 (a)는 그 대역까지 내려
    # 세로를 채운다. 그만큼 (a)의 폭도 커지므로 슬롯 예약분에 LABEL_BAND를 반영.
    FIGW, FIGH = 11.2, 3.9
    TOP, BOT, LABEL_BAND = 0.92, 0.19, 0.135
    pa, pb, pc = [projected_aspect(e) for e in (PANARCTIC, ALASKA, ZOOM)]
    ratios = [pa * (1.0 + LABEL_BAND / (TOP - BOT)), pb, pc]
    fig = plt.figure(figsize=(FIGW, FIGH))
    gs = fig.add_gridspec(1, 3, width_ratios=ratios, wspace=0.16,
                          left=0.035, right=0.985, bottom=BOT, top=TOP)

    # (a) 범북극 위치 맥락 -------------------------------------------------
    _, ax_a = make_ax(PANARCTIC, fig=fig, grid_labels=False,
                      ax=fig.add_subplot(gs[0, 0], projection=map_projection(PANARCTIC)))
    circular_boundary(ax_a)
    # 북극권(66.5°N) 참조선: 눈금이 없는 원형 개관도에 최소한의 위도 기준을 준다
    ax_a.plot(np.linspace(-180, 180, 721), np.full(721, 66.5), color="#8d949c",
              lw=0.6, ls=(0, (4, 3)), zorder=3, transform=ccrs.PlateCarree())
    for k in range(n_fold):
        m = fold_of == k
        ax_a.scatter(df.lon[m], df.lat[m], s=5.0, c=FOLD_COLORS[k], alpha=0.9,
                     linewidths=0, zorder=4, transform=ccrs.PlateCarree())
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]  # 상자 대신 얇은 흰 외곽선
    ax_a.text(30, 63.0, "66.5°N", transform=ccrs.PlateCarree(), fontsize=8,
              color="#5a6169", ha="center", va="center", zorder=6, path_effects=halo)
    _draw_bbox(ax_a, ALASKA, label="(b)")

    # (b) 알래스카: 0.5° 블록이 폴드로 통째 배정된 모습 ---------------------
    _, ax_b = make_ax(ALASKA, fig=fig,
                      ax=fig.add_subplot(gs[0, 1], projection=map_projection(ALASKA)),
                      grid_lon=np.arange(-170, -137.9, 5.0),
                      grid_lat=np.arange(60, 72.1, 3.0),
                      grid_kw=dict(rotate_labels=False))
    ak = cells[(cells.lon0 >= ALASKA[1]) & (cells.lon0 < ALASKA[2])
               & (cells.lat0 >= ALASKA[3]) & (cells.lat0 < ALASKA[4])]
    _draw_cells(ax_b, ak, alpha=0.95, edge="white", lw=0.3)
    _draw_bbox(ax_b, ZOOM, label="(c)")
    _scalebar_km(ax_b, 300)

    # (c) 확대: 블록 격자 + 블록 내부 관측이 모두 같은 색 --------------------
    # 눈금 라벨은 1° 간격(중첩 방지), 블록 경계선은 0.5° 간격으로 따로 그린다.
    _, ax_c = make_ax(ZOOM, fig=fig,
                      ax=fig.add_subplot(gs[0, 2], projection=map_projection(ZOOM)),
                      grid_lon=np.arange(ZOOM[1], ZOOM[2] + 0.01, 1.0),
                      grid_lat=np.arange(ZOOM[3], ZOOM[4] + 0.01, BLOCK_DEG_),
                      grid_kw=dict(linewidth=0.0, color="none", alpha=0.0,
                                   rotate_labels=False))
    for _ax in (ax_b, ax_c):  # 축소 게재 대비 눈금 라벨 9pt
        _ax._gl.xlabel_style = _ax._gl.ylabel_style = {"size": 9, "color": "0.3"}
    _draw_block_grid(ax_c, ZOOM)
    zc = cells[(cells.lon0 >= ZOOM[1]) & (cells.lon0 < ZOOM[2])
               & (cells.lat0 >= ZOOM[3]) & (cells.lat0 < ZOOM[4])]
    _draw_cells(ax_c, zc, alpha=0.45, edge="none", lw=0, zorder=1)
    zm = (df.lon.between(ZOOM[1], ZOOM[2]) & df.lat.between(ZOOM[3], ZOOM[4])).values
    MARK_S = 8.0
    for k in range(n_fold):
        m = zm & (fold_of == k)
        if m.any():
            # alpha<1: 최밀집 구역에서 겹침 정도가 보이도록(단일 블롭 방지)
            ax_c.scatter(df.lon[m], df.lat[m], s=MARK_S, c=FOLD_COLORS[k], alpha=0.7,
                         linewidths=0.35, edgecolors=_darken(FOLD_COLORS[k]),
                         zorder=4, transform=ccrs.PlateCarree())
    sb_c = _scalebar_km(ax_c, 10, frac=(0.86, 0.05))   # 관측 없는 우하단 블록에 배치

    # 공통 범례: 색 = 폴드 번호 -------------------------------------------
    handles = [mpatches.Patch(facecolor=FOLD_COLORS[k], edgecolor="none",
                              label=f"{k + 1}") for k in range(n_fold)]
    handles.append(mpatches.Patch(facecolor="#efece6", edgecolor="#b9b5ac",
                                  linewidth=0.5, label="관측 없는 블록"))
    fig.legend(handles=handles, title="교차검증 폴드", ncol=len(handles),
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               fontsize=9.5, title_fontsize=9.5, handlelength=1.2,
               handleheight=0.9, columnspacing=1.1, handletextpad=0.5)

    # 레이아웃 마감: (a)를 (b)(c)의 눈금 라벨 하단까지 늘려 세로 여백을 제거한다
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ybot = min(ax.get_tightbbox(rend).transformed(inv).y0 for ax in (ax_b, ax_c))
    slot = ax_a.get_position(original=True)
    h = TOP - ybot
    w = h * FIGH / FIGW * pa
    ax_a.set_position([slot.x0 + 0.5 * (slot.width - w), ybot, w, h])

    # 블록별 폴드 라벨: 마커와 겹치지 않는 모서리로 배치(최종 위치 확정 후)
    _place_cell_labels(fig, ax_c, zc, df.lon[zm].values, df.lat[zm].values,
                       marker_s=MARK_S, obstacles=sb_c)

    fig.canvas.draw()
    ytop = max(ax.get_position().y1 for ax in (ax_a, ax_b, ax_c))
    _panel_labels(fig, [(ax_a, "a", "범북극 관측 분포"),
                        (ax_b, "b", "알래스카 · 0.5° 블록의 폴드 배정"),
                        (ax_c, "c", "블록 격자 확대 · 북부 알래스카")], ytop + 0.022)
    save(fig, "spatial_block_folds_map")


# ---------------- 3. 지역별 공변량 가용성 막대(트랙 결정 근거) ----------------
def fig_covariate_availability_bars():
    cp = pd.read_csv(PROC / "covariate_availability_by_region.csv")
    main = cp[cp.region.isin(["ABoVE_AK", "Lena_RU", "ABoVE_CA"])]
    piv = main.pivot_table(index="group", columns="region", values="valid_pct", aggfunc="first")
    piv = piv.reindex(["terrain", "climate", "soil", "cci", "insar", "polsar"])
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax, colormap="cmc.batlow", width=0.78)
    ax.axhline(50, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("유효율 (%)", fontsize=9); ax.set_xlabel("공변량 그룹", fontsize=9)
    ax.set_title("지역×공변량 가용성 — SAR(InSAR/PolSAR)는 알래스카에만 존재\n→ 정확도=AK in-domain, 전이=공유피처+물리", fontsize=10)
    ax.legend(title="", fontsize=8); ax.tick_params(axis="x", rotation=0)
    save(fig, "covariate_availability_bars")


FIGS = {
    "source_overlap_heatmap": fig_source_overlap_heatmap,
    "spatial_block_folds_map": fig_spatial_block_folds_map,
    "covariate_availability_bars": fig_covariate_availability_bars,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for name in names:
        FIGS[name]()
    print(f"[done] S0 시각화 {len(names)}종 완료")
