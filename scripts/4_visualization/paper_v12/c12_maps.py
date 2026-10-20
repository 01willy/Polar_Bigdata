"""Map helpers for the v11 paper figures of role C (Fig. 6 product maps, Fig. 7 yearly maps, Supplementary Figs S7 and S8).
Copied from scripts/4_visualization/paper_v10/maps10.py and edited for the v11 style.

Reuses the grid/cache machinery of scripts/4_visualization/v7_f6_maps.py (Base: 250 m x step hillshade grid with the 1 km cell index per pixel,
interp_cells: display interpolation of 1 km cell centres, fill_rows: <= 2-cell gap filling, coast, graticule, obs_cells) with the v11 map palette:
land outside the mapped domain flat light grey, water very light blue-grey, cells outside the area of applicability blended 40 % towards the land
grey, optional subtle hillshade on mapped cells only, coast/graticule/frame in greys, text #262626. Colormaps: ALT cmcrameri oslo_r (truncated so
that the low end stays bluish and distinct from the land grey), anomaly/difference cmcrameri vik with symmetric limits.
The v7 cache files are reused read-only through symlinks in outputs/figures/paper/v12/_cache (the F6 code only writes a cache file when it does
not exist); new caches are written to outputs/figures/paper/v12/_cache (never into v7 or v10).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c12_common as C  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb, ListedColormap  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

V7_CACHE = C.ROOT / "outputs" / "figures" / "paper" / "v7" / "_cache"
CACHE = C.OUT_MAIN / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
for f in V7_CACHE.glob("*.npz"):                 # read-only reuse of existing caches
    t = CACHE / f.name
    if not t.exists() and not t.is_symlink():
        try:
            t.symlink_to(f)
        except OSError:
            pass
os.environ["F6_CACHE"] = str(CACHE)
import v7_f6_maps as F6  # noqa: E402
import cmcrameri.cm as cmc  # noqa: E402

F6.CACHE = CACHE
LAND_HEX = "#DDDDDD"                            # land outside the mapped domain (and grid cells without a value); darker than the vik centre #ece5e0
WATER_HEX = "#EEF3F7"                           # water: very light blue-grey
LAND = np.array(to_rgb(LAND_HEX), np.float32); WATER = np.array(to_rgb(WATER_HEX), np.float32)
FADE = 0.40                                     # outside the area of applicability: blend towards LAND
COAST = "#6E6E6E"; GRAT = "#BDBDBD"; FRAME = C.AXIS; OBS = C.TEXT
ALT_LUT = (0.16, 0.88)                          # oslo_r section used for ALT (low end bluish grey #c2c8d3, high end dark navy)
CM_ALT = F6.MV.lut(cmc.oslo_r, *ALT_LUT)
CM_ANOM = ListedColormap(cmc.vik(np.linspace(0.0, 1.0, 256)))   # diverging anomaly / difference (symmetric limits)
HALO = [pe.withStroke(linewidth=1.4, foreground="white")]


class Med(F6.Med):
    """F6 paper sizes with the v12 adjustments (black frame 0.5 pt, coast 0.5 pt, 7 pt map text)."""

    def __init__(self):
        super().__init__("paper")
        self.fs_tick = 7.0; self.fs_label = 7.0; self.fs_cbar = 7.0          # v12: one text size
        self.lw_frame = 0.5; self.lw_coast = 0.5; self.lw_grat = 0.35; self.lw_scale = 0.9; self.lw_box = 0.8; self.lw_link = 0.5
        self.tick_len = 2.0; self.tick_w = 0.5; self.tick_pad = 1.4
        self.halo = 1.6; self.halo_fx = [pe.withStroke(linewidth=self.halo, foreground="white")]


def patch_f6():
    """Point the v7 map helpers to the v11 colours (module globals read at call time; nothing is written to the v7 module file)."""
    F6.COAST = COAST; F6.GRAT = GRAT; F6.FRAME = FRAME; F6.OBS = OBS
    F6.LAND = LAND; F6.WATER = WATER; F6.HATCH = LAND; F6.FADE = FADE


patch_f6()


def render(base, val, cmap, norm, aoa=None, smooth=False, fill_max=0, w_shade=0.0, feather=0.0, aoa_sigma=0.0, fade=FADE):
    """Cell values -> RGB uint8 on the base grid. Drawn = finite value and DEM shade present (as F6.render). Everything else on land = LAND
    (flat, no hatching, no shade); water cells and sea = WATER. Outside the area of applicability (aoa == 0) colours are blended towards LAND by
    `fade`. Hillshade (w_shade) multiplies drawn pixels only. smooth/fill_max as F6.render (display interpolation, <= fill_max-cell gaps)."""
    idx = base.idx; has = idx >= 0
    v = np.full(idx.shape, np.nan, np.float32); v[has] = val[idx[has]]
    drawn = np.isfinite(v) & base.valid; drawn_src = drawn; water_cell = base.water_cell
    if smooth:
        vs = F6.interp_cells(base, val)
        if fill_max > 0:
            fill = np.zeros(idx.shape, bool); fill[has] = F6.fill_rows(base, val, fill_max)[idx[has]]; fill &= np.isfinite(vs)
            drawn = drawn | fill; water_cell = water_cell & ~fill; del fill
        v = np.where(drawn & np.isfinite(vs), vs, v); del vs
    water = water_cell | (~base.land & ~drawn)
    rgb = np.empty(idx.shape + (3,), np.float32); rgb[:] = LAND; rgb[water] = WATER
    vv = np.where(drawn, v, np.nanmean(v[drawn]) if drawn.any() else 0.0)
    col = np.empty(idx.shape + (3,), np.float32)
    for r0 in range(0, idx.shape[0], 512):                      # chunked colour mapping (float64 RGBA of the full grid is large)
        col[r0:r0 + 512] = cmap(norm(vv[r0:r0 + 512]))[..., :3]
    del vv
    if aoa is not None:
        w = np.zeros(idx.shape, np.float32); w[has] = aoa[idx[has]]
        if aoa_sigma > 0:
            d = drawn_src.astype(np.float32); den = gaussian_filter(d, aoa_sigma); num = gaussian_filter(w * d, aoa_sigma)
            w = np.where(den > 1e-3, num / np.maximum(den, 1e-6), w)
        f = (fade * (1.0 - w))[..., None]
        col = col * (1.0 - f) + f * LAND; del w, f
    if w_shade > 0:
        col = col * ((1 - w_shade) + w_shade * base.shade)[..., None]
    if feather > 0:
        d = drawn.astype(np.float32); a = gaussian_filter(d, feather)
        num = np.stack([gaussian_filter(col[..., c] * d, feather) for c in range(3)], axis=-1)
        cs = np.where(a[..., None] > 1e-3, num / np.maximum(a, 1e-6)[..., None], col)
        a = np.clip(a, 0, 1)[..., None]; rgb = a * cs + (1 - a) * rgb
    else:
        rgb[drawn] = col[drawn]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), int(drawn_src.sum())


def faded(cmap, x=0.55):
    """Swatch colour of a mapped cell outside the area of applicability (cmap(x) blended FADE towards LAND), for legends."""
    mid = np.array(cmap(x)[:3], np.float32); return tuple((mid * (1 - FADE) + FADE * LAND).tolist())


def colorbar(fig, cax, cmap, norm, label, ticks, extend="neither", minor=None, label_top=False, orientation="horizontal", fs_label=None):
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation=orientation, extend=extend, ticks=ticks, extendfrac=0.06)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(C.AXIS)
    cb.ax.tick_params(labelsize=C.FS_SMALL, width=0.4, length=1.6, pad=1.2, colors=C.AXIS, labelcolor=C.TEXT)
    labs = [f"{t:g}".replace("-", "−") for t in ticks]
    (cb.ax.set_xticklabels if orientation == "horizontal" else cb.ax.set_yticklabels)(labs)
    if minor is not None:
        (cb.ax.set_xticks if orientation == "horizontal" else cb.ax.set_yticks)(minor, minor=True)
        cb.ax.tick_params(which="minor", length=1.0, width=0.4)
    fs = fs_label or C.FS_TICK
    if label_top:
        cb.ax.set_title(label, fontsize=fs, pad=2.0, loc="center", color=C.TEXT)
    else:
        cb.set_label(label, fontsize=fs, labelpad=1.5, color=C.TEXT)
    return cb


def frame(ax, rgb, extent, lim):
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="antialiased", zorder=1)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor(FRAME)


def corner_label(ax, s, M, x=0.03, y=0.965, ha="left", fs=None):
    return ax.text(x, y, s, transform=ax.transAxes, va="top", ha=ha, fontsize=fs or M.fs_label, color=C.TEXT, zorder=9,
                   path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])


def scalebar(ax, lim, km, M, fx=0.05, fy=0.06):
    """Scale bar (as F6.scalebar, dark-grey ink): length label above the bar, end ticks, white halo."""
    x0, x1, y0, y1 = lim; L = km * 1000.0; tick = 0.014 * (y1 - y0)
    xs = x0 + fx * (x1 - x0); ys = y0 + fy * (y1 - y0)
    halo = [pe.withStroke(linewidth=M.halo + 0.4, foreground="white")]
    ax.plot([xs, xs + L], [ys, ys], color=C.TEXT, lw=M.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    for x in (xs, xs + L):
        ax.plot([x, x], [ys, ys + tick], color=C.TEXT, lw=0.8 * M.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    return ax.text(xs + L / 2, ys + 1.4 * tick, f"{km} km", ha="center", va="bottom", fontsize=M.fs_tick, color=C.TEXT, zorder=9, path_effects=M.halo_fx)


def box(ax, lim, lw=0.6, color=None):
    x0, x1, y0, y1 = lim
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color or C.TEXT, lw=lw, zorder=7, solid_joinstyle="miter",
            path_effects=[pe.withStroke(linewidth=lw + 1.0, foreground="white")])


from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


class FadedRamp:
    """Legend handle placeholder for 'outside the area of applicability': drawn by RampHandler as the colormap blended FADE towards LAND."""

    def __init__(self, cmap, fade=FADE, n=10, lo=0.08, hi=0.92):
        self.cmap, self.fade, self.n, self.lo, self.hi = cmap, fade, n, lo, hi


class RampHandler(HandlerBase):
    """Draws a FadedRamp as a small horizontal colour ramp (the same blending rule as the maps), thin grey outline."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        h = orig_handle; x0, y0 = -xdescent, -ydescent; arts = []
        for k in range(h.n):
            c = np.array(h.cmap(h.lo + (h.hi - h.lo) * (k + 0.5) / h.n)[:3]); c = c * (1 - h.fade) + h.fade * LAND
            arts.append(Rectangle((x0 + k * width / h.n, y0), width / h.n + 0.05, height, facecolor=tuple(c.tolist()), edgecolor="none",
                                  transform=trans))
        arts.append(Rectangle((x0, y0), width, height, facecolor="none", edgecolor="#B5B5B5", lw=0.3, transform=trans))
        return arts
