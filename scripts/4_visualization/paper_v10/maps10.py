"""Map helpers for the v10 paper figures (Fig. 6, Supplementary Figs S6, S7, S9).

Reuses the grid/cache machinery of scripts/4_visualization/v7_f6_maps.py (Base: 250 m x step hillshade grid with the 1 km cell index per pixel,
interp_cells: display interpolation of 1 km cell centres, fill_rows: <= 2-cell gap filling, coast, graticule, scalebar, obs_cells) but renders with
the paper palette: land outside the mapped domain light grey (flat, no hatching), water very light blue-grey, cells outside the area of
applicability blended 40 % towards the land grey, optional subtle hillshade on mapped cells only. The v7 cache folder is read through symlinks;
new caches are written to outputs/figures/paper/v10/_cache (never into v7).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb, Normalize, ListedColormap  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

V7_CACHE = C.ROOT / "outputs" / "figures" / "paper" / "v7" / "_cache"
CACHE = C.OUT_MAIN / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
for f in V7_CACHE.glob("*.npz"):                 # read-only reuse of existing caches
    t = CACHE / f.name
    if not t.exists():
        try:
            t.symlink_to(f)
        except OSError:
            pass
os.environ["F6_CACHE"] = str(CACHE)
import v7_f6_maps as F6  # noqa: E402
import cmcrameri.cm as cmc  # noqa: E402

F6.CACHE = CACHE
LAND_HEX = "#E1E1E1"                            # land outside the mapped domain (and grid cells without a value)
WATER_HEX = "#EDF2F6"                           # water: very light blue-grey
LAND = np.array(to_rgb(LAND_HEX), np.float32); WATER = np.array(to_rgb(WATER_HEX), np.float32)
FADE = 0.40                                     # outside the area of applicability: blend towards LAND
COAST = "#7F7F7F"; GRAT = "#BDBDBD"; FRAME = "#000000"
CM_ALT = F6.MV.lut(cmc.oslo_r, 0.12, 0.90)      # sequential ALT (truncated so the ends stay visible on white/grey)
CM_ANOM = ListedColormap(cmc.vik(np.linspace(0.0, 1.0, 250)))   # diverging anomaly / difference (symmetric limits)
OBS = "#000000"


class Med(F6.Med):
    """F6 paper sizes with the v10 adjustments (thin coast/graticule/frame, 6 pt map text)."""

    def __init__(self):
        super().__init__("paper")
        self.fs_tick = 6.0; self.fs_label = 6.5; self.fs_cbar = 6.5
        self.lw_frame = 0.4; self.lw_coast = 0.3; self.lw_grat = 0.3; self.lw_scale = 0.8; self.lw_box = 0.6; self.lw_link = 0.4
        self.tick_len = 1.5; self.tick_w = 0.4; self.tick_pad = 1.2
        self.halo = 1.6; self.halo_fx = [pe.withStroke(linewidth=self.halo, foreground="white")]


def patch_f6():
    """Point the v7 map helpers to the v10 colours (module globals read at call time)."""
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


def colorbar(fig, cax, cmap, norm, label, ticks, extend="neither", minor=None, label_top=False):
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=extend, ticks=ticks, extendfrac=0.06)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor("black")
    cb.ax.tick_params(labelsize=C.FS_SMALL, width=0.4, length=1.6, pad=1.2, colors="black")
    cb.ax.set_xticklabels([f"{t:g}".replace("-", "−") for t in ticks])
    if minor is not None:
        cb.ax.set_xticks(minor, minor=True); cb.ax.tick_params(which="minor", length=1.0, width=0.4)
    if label_top:
        cb.ax.set_title(label, fontsize=C.FS_TICK, pad=2.0, loc="center")
    else:
        cb.set_label(label, fontsize=C.FS_TICK, labelpad=1.5)
    return cb


def frame(ax, rgb, extent, lim):
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="antialiased", zorder=1)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor(FRAME)


def corner_label(ax, s, M, x=0.03, y=0.965, ha="left"):
    return ax.text(x, y, s, transform=ax.transAxes, va="top", ha=ha, fontsize=M.fs_label, color="black", zorder=9,
                   path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
