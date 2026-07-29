"""연별 ALT 지도(조밀 격자). 그해 ERA5-Land 기후를 물리식에 대입해 연속장을 산출한다.

기존 그림은 관측 지점에만 예측을 얹어 성글고 연도 차이가 보이지 않았다. 기후 구동장은
격자 전역에서 계산되므로 조밀 격자로 그리면 그해 기후의 공간 구조가 드러난다.
아래 행에는 기간 평균 대비 편차를 그려 연도 간 신호를 직접 보인다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar                      # noqa: E402
from polar.fidelity import macro_region, fit_stefan_E      # noqa: E402
from polar.geomap import make_ax, mask_ocean  # noqa: E402
import pandas as pd                                        # noqa: E402
from cmcrameri import cm as cmc                            # noqa: E402

use_polar()
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2010-2024.nc"
OUTD = ROOT / "outputs" / "figures" / "s9_timelapse"
EXT = ("alaska", -168.5, -138.0, 58.5, 71.5)   # (이름, 경도, 위도) 창
YEARS = [2013, 2016, 2019, 2022]
RES = 0.05
GREY = "#5b6670"

# ---------------------------------------------------------------- 계수
df = pd.read_csv(ROOT / "data" / "processed" / "fidelity_base.csv")
df["macro"] = macro_region(df)
ak = df[df.macro == "Alaska"]
E = fit_stefan_E(ak["alt_cm"].values, ak["e5_sqrt_tdd"].values)
print(f"Stefan 계수 E = {E:.4f} (알래스카 실측 적합)")

# ---------------------------------------------------------------- 연별 TDD 격자
ds = xr.open_dataset(NC)
lat_name = "latitude" if "latitude" in ds.coords else "lat"
lon_name = "longitude" if "longitude" in ds.coords else "lon"
tname = "valid_time" if "valid_time" in ds.coords else "time"
lon = ds[lon_name].values
lon180 = np.where(lon > 180, lon - 360, lon)
sel_lon = (lon180 >= EXT[1] - 1) & (lon180 <= EXT[2] + 1)
sel_lat = (ds[lat_name].values >= EXT[3] - 1) & (ds[lat_name].values <= EXT[4] + 1)
sub = ds.isel({lon_name: np.where(sel_lon)[0], lat_name: np.where(sel_lat)[0]})
t2m = sub["t2m"] - 273.15
days = xr.DataArray(
    [pd.Period(str(v)[:10], freq="M").days_in_month for v in sub[tname].values],
    coords={tname: sub[tname]}, dims=[tname])
tdd_all = (t2m.clip(min=0) * days).groupby(f"{tname}.year").sum(tname)   # (year, lat, lon)

glat0 = sub[lat_name].values
glon0 = np.sort(lon180[sel_lon])
order = np.argsort(lon180[sel_lon])
if glat0[0] > glat0[-1]:                    # 위도 오름차순 정렬
    glat0 = glat0[::-1]; flip = True
else:
    flip = False

glat = np.arange(EXT[3], EXT[4] + RES, RES)
glon = np.arange(EXT[1], EXT[2] + RES, RES)

def field(year):
    a = tdd_all.sel(year=year).values[:, order]
    if flip:
        a = a[::-1]
    da = xr.DataArray(a, coords={"lat": glat0, "lon": glon0}, dims=["lat", "lon"])
    fine = da.interp(lat=glat, lon=glon, method="linear")
    return E * np.sqrt(np.clip(fine.values, 0, None))

alts = {y: field(y) for y in YEARS}
mean_years = list(range(2010, 2025))
base = np.mean([field(y) for y in mean_years], axis=0)
print(f"격자 {len(glat)}x{len(glon)} · 기간평균 ALT {np.nanmean(base):.1f} cm")

# 영구동토 마스크: 기간평균 연평균기온 < 0 인 곳만
maat = (t2m.groupby(f"{tname}.year").mean(tname).mean("year").values[:, order])
if flip:
    maat = maat[::-1]
maat_f = xr.DataArray(maat, coords={"lat": glat0, "lon": glon0},
                      dims=["lat", "lon"]).interp(lat=glat, lon=glon).values
pf = maat_f < 0.0
for y in YEARS:
    alts[y] = np.where(pf, alts[y], np.nan)
base = np.where(pf, base, np.nan)

# ---------------------------------------------------------------- 그림
vals = np.concatenate([alts[y][np.isfinite(alts[y])] for y in YEARS])
vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
lv = np.arange(np.floor(vmin / 5) * 5, np.ceil(vmax / 5) * 5 + 5, 5)
amax = max(abs(np.nanpercentile(alts[y] - base, [2, 98])).max() for y in YEARS)
amax = np.ceil(amax / 2) * 2                    # 눈금이 정수가 되도록 반올림
lva = np.arange(-amax, amax + 1e-9, amax / 5)

fig = plt.figure(figsize=(11.6, 5.6))
gs = fig.add_gridspec(2, len(YEARS), hspace=0.06, wspace=0.06,
                      left=0.035, right=0.90, top=0.94, bottom=0.06)
for j, y in enumerate(YEARS):
    ax = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree())
    make_ax(EXT, ax=ax, fig=fig, grid=False)
    cf = ax.contourf(glon, glat, alts[y], levels=lv, cmap=cmc.oslo_r,
                     extend="both", transform=ccrs.PlateCarree(), zorder=1.5)
    for c in cf.collections:
        c.set_edgecolor("face"); c.set_rasterized(True)
    mask_ocean(ax, zorder=3)
    ax.set_title(f"{y}년", fontsize=10.5, pad=4, color="#222")
    if j == 0:
        ax.text(-0.02, 0.5, "예측 ALT", transform=ax.transAxes, rotation=90,
                va="center", ha="right", fontsize=10, color=GREY)
    if j == len(YEARS) - 1:
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.03, extend="both")
        cb.set_label("ALT (cm)", fontsize=9.5, color=GREY)
        cb.ax.tick_params(labelsize=8.5, labelcolor=GREY, length=2)

    axa = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())
    make_ax(EXT, ax=axa, fig=fig, grid=False)
    cfa = axa.contourf(glon, glat, alts[y] - base, levels=lva, cmap=cmc.broc_r,
                       extend="both", transform=ccrs.PlateCarree(), zorder=1.5)
    for c in cfa.collections:
        c.set_edgecolor("face"); c.set_rasterized(True)
    mask_ocean(axa, zorder=3)
    if j == 0:
        axa.text(-0.02, 0.5, "기간평균 대비 편차", transform=axa.transAxes, rotation=90,
                 va="center", ha="right", fontsize=10, color=GREY)
    if j == len(YEARS) - 1:
        cb = fig.colorbar(cfa, ax=axa, fraction=0.05, pad=0.03, extend="both")
        cb.set_label("편차 (cm)", fontsize=9.5, color=GREY)
        cb.ax.tick_params(labelsize=8.5, labelcolor=GREY, length=2)

fig.text(0.035, 0.012,
         f"격자 {RES}° · 그해 ERA5-Land 융해도일을 물리식에 대입 · "
         f"기간평균은 2010–2024년 · 연평균기온 0 °C 이상 육지는 제외",
         fontsize=7.6, color="#7a7a7a", ha="left", va="bottom")
for ext in ("png", "pdf"):
    fig.savefig(OUTD / f"alt_annual_fields.{ext}", dpi=300, bbox_inches="tight")
print(f"saved: {OUTD}/alt_annual_fields.png")
