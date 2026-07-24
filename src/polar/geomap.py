"""실제 지도 배경 위 매핑 헬퍼 (cartopy) — 논문급 공간 시각화.

`docs/RESEARCH_PLAN_...` §11.5 + 사용자 요구: within-site 등 매핑 시 실제 해안선·육지·
격자를 깔아 위경도가 어느 지역인지 알 수 있게. 냉색 규약(plotstyle.CMAP) 준수.

사용:
    from polar.geomap import make_ax, scatter_map, ALASKA, PANARCTIC
    fig, ax = make_ax(ALASKA)
    sc = scatter_map(ax, lon, lat, vals, cmap=CMAP.alt, vmin=20, vmax=110, label="ALT (cm)")
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    _HAS_CARTOPY = True
except Exception:  # pragma: no cover
    _HAS_CARTOPY = False

# 논문급 지물 색 (은은한 웜그레이 육지 / 옅은 청 해양)
_LAND = "#efece6"
_OCEAN = "#eaf1f4"
_COAST = "#7d7d7d"
_BORDER = "#b8b3aa"

# 지역 프리셋: (이름, lon0, lon1, lat0, lat1)
ALASKA = ("알래스카", -170, -138, 59, 72)
PANARCTIC = ("범북극", -180, 180, 45, 82)
LENA = ("레나 삼각주", 120, 133, 70, 75)


def _proj(extent):
    """지역 중심 정사영(AlbersEqualArea)으로 왜곡 최소화."""
    _, lo0, lo1, la0, la1 = extent
    lon0 = 0.5 * (lo0 + lo1)
    lat0 = 0.5 * (la0 + la1)
    if not _HAS_CARTOPY:
        return None
    if lo1 - lo0 >= 300:  # 범북극은 정사(북극 중심)
        return ccrs.NorthPolarStereo(central_longitude=-100)
    return ccrs.AlbersEqualArea(central_longitude=lon0, central_latitude=lat0,
                                standard_parallels=(la0 + 3, la1 - 3))


def make_ax(extent=ALASKA, figsize=(7.2, 6.0), ax=None, fig=None, title=None):
    """실제 지도 배경(육지·해양·해안선·격자) 위 axes 반환. cartopy 없으면 평면 fallback."""
    name, lo0, lo1, la0, la1 = extent
    if not _HAS_CARTOPY:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (la0 + la1))))
        ax.set_xlabel("경도 (°)"); ax.set_ylabel("위도 (°)")
        if title:
            ax.set_title(title)
        ax._is_geo = False
        return fig, ax

    proj = _proj(extent)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([lo0, lo1, la0, la1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=_LAND, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=_OCEAN, alpha=0.6, zorder=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=_COAST, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=_BORDER, linewidth=0.4, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.6, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": "0.3"}
    if title:
        ax.set_title(title, fontsize=11)
    ax._is_geo = True
    return fig, ax


def scatter_map(ax, lon, lat, vals, cmap=None, vmin=None, vmax=None, s=9,
                label=None, norm=None, edge=False):
    """지도 위 값 산점. cartopy면 PlateCarree transform, 아니면 평면."""
    kw = dict(c=vals, cmap=cmap, s=s, alpha=0.9,
              linewidths=0.15 if edge else 0, edgecolors="0.25" if edge else "none")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin; kw["vmax"] = vmax
    if getattr(ax, "_is_geo", False):
        kw["transform"] = ccrs.PlateCarree()
    sc = ax.scatter(np.asarray(lon), np.asarray(lat), **kw)
    return sc


def add_colorbar(fig, sc, ax, label, shrink=0.8):
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, shrink=shrink)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def add_scalebar_note(ax, extent):
    """간단 축척 참고(위도 기반 km 눈금 대체 주석). cartopy scalebar는 버전의존이라 텍스트로."""
    name, lo0, lo1, la0, la1 = extent
    ax.text(0.02, 0.02, name, transform=ax.transAxes, fontsize=9, color="0.25",
            va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))


# ============================================================
# 논문형 고도화 (viz 워크플로 스펙: Ran 2022·Whitcomb 2024·Obu 2019 사례)
# ============================================================
import numpy as _np

# ALT 이산 단계(논문형 BoundaryNorm) — CCI·TC 지도 관행
ALT_LEVELS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]


def to_proj(ax, lon, lat):
    """경위도 → ax.projection 좌표(m). transform kwarg 의존보다 안전(hexbin·마스크 공통 기반)."""
    if not getattr(ax, "_is_geo", False):
        return _np.asarray(lon), _np.asarray(lat)
    pts = ax.projection.transform_points(ccrs.PlateCarree(), _np.asarray(lon), _np.asarray(lat))
    return pts[..., 0], pts[..., 1]


def hexbin_map(ax, lon, lat, vals=None, gridsize=48, reduce=None, mincnt=1,
               cmap=None, vmin=None, vmax=None, norm=None, log_count=False, rasterized=True):
    """관측을 육각 셀 통계로 필드화(보간 없음, 통계 왜곡 없음). 성긴 산점 대체.
    vals=None이면 밀도 모드(개수), 지정 시 셀 reduce(기본 median). 반환 PolyCollection."""
    import numpy as np
    x, y = to_proj(ax, lon, lat)
    kw = dict(gridsize=gridsize, mincnt=mincnt, cmap=cmap, rasterized=rasterized, linewidths=0.15, edgecolors="face")
    if vals is None:
        kw.update(bins="log" if log_count else None)
    else:
        kw.update(C=np.asarray(vals), reduce_C_function=(reduce or np.median))
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin; kw["vmax"] = vmax
    return ax.hexbin(x, y, **kw)


def field_map(ax, lon2d, lat2d, grid, cmap=None, norm=None, vmin=None, vmax=None,
              levels=None, alpha=1.0, mask=None, rasterized=True):
    """격자 필드 pcolormesh. levels 지정 시 BoundaryNorm 이산 단계(논문형). mask=True 영역 회색."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm
    g = np.array(grid, float)
    if mask is not None:
        g = np.where(mask, np.nan, g)
    g = np.ma.masked_invalid(g)
    if cmap is not None:
        try:
            cmap = cmap.copy(); cmap.set_bad("#d9d9d9")
        except Exception:
            pass
    if levels is not None:
        norm = BoundaryNorm(levels, getattr(cmap, "N", 256), extend="both")
    kw = dict(cmap=cmap, alpha=alpha, rasterized=rasterized, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin; kw["vmax"] = vmax
    if getattr(ax, "_is_geo", False):
        kw["transform"] = ccrs.PlateCarree()
    return ax.pcolormesh(lon2d, lat2d, g, **kw)


def support_mask(lon2d, lat2d, lon_obs, lat_obs, thresh_km=25.0):
    """관측 최근접거리 > thresh_km 격자셀 = True(지원범위 밖, AOA 간이). Whitcomb2024 관행."""
    import numpy as np
    from scipy.spatial import cKDTree
    # 근사 평면거리(위도 보정). 극지라 대충이면 충분.
    latm = np.deg2rad(np.nanmean(lat_obs))
    xo = np.asarray(lon_obs) * np.cos(latm) * 111.0
    yo = np.asarray(lat_obs) * 111.0
    tree = cKDTree(np.c_[xo, yo])
    xg = (lon2d * np.cos(latm) * 111.0).ravel()
    yg = (lat2d * 111.0).ravel()
    d, _ = tree.query(np.c_[xg, yg])
    return (d.reshape(lon2d.shape) > thresh_km)


def add_scalebar(ax, length_km=200, loc="lower right"):
    """Albers 등 m 단위 투영에 물리 축척. 실패 시 조용히 스킵."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        if getattr(ax, "_is_geo", False):
            ax.add_artist(ScaleBar(1, units="m", location=loc, length_fraction=0.25,
                                   box_alpha=0.7, color="0.2", font_properties={"size": 7}))
    except Exception:
        pass


def add_inset_locator(fig, ax, extent, size=0.24):
    """범북극 원형 inset에 현재 지역 위치 박스 표시(어디인지 한눈에)."""
    if not _HAS_CARTOPY:
        return None
    name, lo0, lo1, la0, la1 = extent
    axins = ax.inset_axes([0.0, 0.72, size, size], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    axins.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    axins.add_feature(cfeature.LAND.with_scale("110m"), facecolor=_LAND, zorder=0)
    axins.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=_OCEAN, zorder=0)
    axins.coastlines("110m", linewidth=0.3, color=_COAST)
    _circular(axins)
    import matplotlib.patches as mpatches
    axins.add_patch(mpatches.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=ccrs.PlateCarree(),
                                       fill=False, edgecolor="#c0392b", linewidth=1.3, zorder=5))
    return axins


def _circular(ax):
    """극지 스테레오 축을 원형 경계로."""
    import matplotlib.path as mpath
    theta = _np.linspace(0, 2 * _np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = _np.vstack([_np.sin(theta), _np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)


def circular_boundary(ax):
    """PANARCTIC NorthPolarStereo 지도를 원형으로(공개 API)."""
    if getattr(ax, "_is_geo", False):
        _circular(ax)
