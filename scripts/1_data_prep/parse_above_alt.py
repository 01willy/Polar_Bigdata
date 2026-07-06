"""ABoVE ds2369 ALT(22.4만 관측) 파싱·QC → 점단위 + site-year 집계,
기존 CALM(alt_global.csv, 3,604 site-year)과 통합 → alt_combined.csv + 시각화.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_korean
plt = use_korean()

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
SRC = "data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv"

# ---------- 로드 + QC ----------
d = pd.read_csv(SRC, usecols=["site_name", "latitude", "longitude", "date",
                              "ALT", "ALT_err"], low_memory=False)
n0 = len(d)
d = d[(d.ALT != -9999) & (d.ALT.notna())]
d = d[(d.latitude != -9999) & (d.longitude != -9999)]
d["year"] = pd.to_datetime(d.date, errors="coerce").dt.year
d = d.dropna(subset=["year"])
d["year"] = d["year"].astype(int)
# 물리적 QC: 0 < ALT < 300 cm (활성층 두께 현실범위), 좌표 범위
d = d[(d.ALT > 0) & (d.ALT < 300)]
d = d[(d.latitude.between(45, 80)) & (d.longitude.between(-170, -60))]
print(f"ABoVE QC: {n0} → {len(d)} 유효 점관측 (ALT 0~300cm, 좌표 정상)")
print(f"  ALT 5/50/95%: {np.percentile(d.ALT,[5,50,95]).round(1)} cm")
print(f"  범위 lat {d.latitude.min():.1f}~{d.latitude.max():.1f}, "
      f"lon {d.longitude.min():.1f}~{d.longitude.max():.1f}, 연도 {d.year.min()}~{d.year.max()}")

# 점단위 저장(공간 DL/패치용)
pt = d.rename(columns={"latitude": "lat", "longitude": "lon", "ALT": "alt_cm",
                       "ALT_err": "alt_err"})[["site_name", "lat", "lon", "year", "date", "alt_cm", "alt_err"]]
pt.to_csv("data/processed/alt_above_pointlevel.csv", index=False)

# site-year 집계(CALM 호환)
sy = (pt.groupby(["site_name", "year"])
        .agg(lat=("lat", "mean"), lon=("lon", "mean"),
             alt_cm=("alt_cm", "mean"), n_pts=("alt_cm", "size")).reset_index())
sy.to_csv("data/processed/alt_above_siteyear.csv", index=False)
print(f"  점단위 {len(pt)} → site-year {len(sy)} (고유 site {sy.site_name.nunique()})")

# ---------- CALM과 통합 ----------
calm = pd.read_csv("data/processed/alt_global.csv")
calm_min = calm[["site", "lat", "lon", "year", "alt_cm"]].copy()
calm_min["source"] = "CALM"
calm_min = calm_min.rename(columns={"site": "site_id"})
above_min = sy.rename(columns={"site_name": "site_id"})[["site_id", "lat", "lon", "year", "alt_cm"]].copy()
above_min["source"] = "ABoVE"
comb = pd.concat([calm_min, above_min], ignore_index=True)
comb.to_csv("data/processed/alt_combined.csv", index=False)
print(f"[통합] CALM {len(calm_min)} + ABoVE {len(above_min)} = {len(comb)} site-year "
      f"→ alt_combined.csv  ({len(comb)/len(calm_min):.1f}배)")

# ================= 시각화 =================
fig = plt.figure(figsize=(15, 6))

# (1) 전지구 커버리지 + ABoVE 밀도
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(calm.lon, calm.lat, s=10, c="#888", label=f"기존 CALM ({calm.groupby(['lat','lon']).ngroups} 사이트, 전지구)",
            zorder=2, edgecolors="none")
sub = pt.sample(min(8000, len(pt)), random_state=0)
ax1.scatter(sub.lon, sub.lat, s=4, c="#e41a1c", alpha=0.3,
            label=f"NEW ABoVE ({sy.site_name.nunique()} 사이트, {len(pt):,} 점)", zorder=3)
ax1.axhline(66.56, color="navy", ls="--", lw=0.7, alpha=0.6)
ax1.set_xlim(-180, 180); ax1.set_ylim(20, 85)
ax1.set_xlabel("경도"); ax1.set_ylabel("위도")
ax1.set_title("전지구 ALT 라벨 커버리지", fontsize=12, weight="bold")
ax1.legend(loc="lower left", fontsize=9); ax1.grid(alpha=0.25)

# (2) 라벨 규모 성장 + ALT 분포
ax2 = fig.add_subplot(2, 2, 2)
bars = ax2.bar(["CALM\n(기존)", "CALM+ABoVE\n(통합, site-year)", "ABoVE\n(점단위)"],
               [len(calm_min), len(comb), len(pt)],
               color=["#888", "#4daf4a", "#e41a1c"])
ax2.set_yscale("log"); ax2.set_ylabel("라벨 수 (log)")
for b, v in zip(bars, [len(calm_min), len(comb), len(pt)]):
    ax2.text(b.get_x()+b.get_width()/2, v*1.15, f"{v:,}", ha="center", fontsize=9, weight="bold")
ax2.set_title("ALT 라벨 규모 성장", fontsize=11, weight="bold")
ax2.grid(alpha=0.25, axis="y")

ax3 = fig.add_subplot(2, 2, 4)
ax3.hist(calm.alt_cm.dropna(), bins=50, alpha=0.55, density=True, color="#888", label="CALM")
ax3.hist(pt.alt_cm, bins=50, alpha=0.55, density=True, color="#e41a1c", label="ABoVE")
ax3.set_xlabel("ALT (cm)"); ax3.set_ylabel("밀도"); ax3.set_xlim(0, 300)
ax3.set_title("ALT 분포 비교", fontsize=11, weight="bold")
ax3.legend(fontsize=9); ax3.grid(alpha=0.25)

fig.suptitle("ABoVE 통합 — 주 출력(ALT) 라벨 밀도 대폭 강화 (전이·공간 DL 학습용)", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig("outputs/figures/17_above_alt_integration.png", dpi=130)
plt.close(fig)
print("saved outputs/figures/17_above_alt_integration.png")
