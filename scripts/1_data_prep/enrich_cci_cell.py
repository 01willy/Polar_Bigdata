"""데이터 확충 — ESA CCI ALT(관측기반 아닌 CryoGrid 제품)를 prior 공변량으로 셀에 추출.
- 25년(1997-2021) NetCDF ALT(m) 다년평균 → cm. 알래스카 bbox 서브셋으로 메모리 절약.
- dl_dataset_cell.csv의 14,348 셀 좌표에 최근접 샘플 → cci_alt(cm), cci_valid 추가(in place).
- CCI는 truth가 아니라 prior/benchmark. ablation의 CCI 그룹으로 자동 편입(M8).
실행: python3 scripts/1_data_prep/enrich_cci_cell.py
"""
import glob, numpy as np, pandas as pd, xarray as xr

CELL = "data/processed/dl_dataset_cell.csv"
cell = pd.read_csv(CELL)
la, lo = cell.lat.values, cell.lon.values
mlo, Mlo, mla, Mla = lo.min()-0.2, lo.max()+0.2, la.min()-0.2, la.max()+0.2
print(f"[bbox] lat {mla:.1f}~{Mla:.1f}  lon {mlo:.1f}~{Mlo:.1f}  cells={len(cell)}")

files = sorted(glob.glob("data/raw/cci_alt/*.nc"))
acc = None; cnt = None; latc = lonc = None
for f in files:
    ds = xr.open_dataset(f)
    sub = ds["ALT"].sel(lat=slice(mla, Mla), lon=slice(mlo, Mlo))
    if sub.ndim == 3:
        sub = sub.isel(time=0)
    a = sub.values.astype("float32")
    if acc is None:
        acc = np.zeros_like(a); cnt = np.zeros_like(a)
        latc = sub.lat.values; lonc = sub.lon.values
    valid = np.isfinite(a)
    acc[valid] += a[valid]; cnt[valid] += 1
    ds.close()
mean_m = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)  # m
mean_cm = mean_m * 100.0
print(f"[cci] grid {mean_cm.shape}  valid {np.isfinite(mean_cm).sum():,}  "
      f"ALT(cm) 5/50/95%={np.nanpercentile(mean_cm,[5,50,95]).round(1)}")

# 최근접 인덱스 샘플
def nearest_idx(coords, vals):
    order = np.argsort(coords)
    cs = coords[order]
    j = np.searchsorted(cs, vals)
    j = np.clip(j, 1, len(cs)-1)
    left = cs[j-1]; right = cs[j]
    j = np.where(np.abs(vals-left) <= np.abs(vals-right), j-1, j)
    return order[j]
iy = nearest_idx(latc, la); ix = nearest_idx(lonc, lo)
cci = mean_cm[iy, ix]
cell["cci_alt"] = np.round(cci, 2)
cell["cci_valid"] = np.isfinite(cci).astype(int)
print(f"[sample] 셀 CCI 유효 {cell.cci_valid.sum():,}/{len(cell)} "
      f"({100*cell.cci_valid.mean():.0f}%)  중앙 {np.nanmedian(cci):.1f}cm")

# 관측 ALT와 상관(정성)
m = np.isfinite(cci) & np.isfinite(cell.alt_cm.values)
if m.sum() > 100:
    r = np.corrcoef(cci[m], cell.alt_cm.values[m])[0, 1]
    print(f"[corr] CCI prior vs 관측 셀평균 ALT: r={r:.3f} (n={m.sum():,})")

cell.to_csv(CELL, index=False)
print("[saved]", CELL, "→ +cci_alt, cci_valid")
