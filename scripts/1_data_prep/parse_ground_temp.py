"""신규 지중온도 3종 중 좌표 완비된 2종(GGD200 서시베리아·PERMOS 스위스) 파싱 →
공통 스키마 data/processed/ground_temp.csv + 직관적 시각화.
공통 스키마: source, site, lat, lon, depth_m, temp_c, year, elev_m, gtnp_id
(G10015 알래스카는 좌표 없음+GTN-P 중복가능 → 별도 처리 예정)
"""
import os, csv, glob, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_korean
plt = use_korean()

RAW = "data/raw/boreholes"
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ---------------- GGD200 (서시베리아, 정적 깊이별 암석온도) ----------------
def parse_ggd200():
    depths = [20, 50, 100, 200, 300, 400, 500, 600, 1000, 2000]
    rows = []
    path = os.path.join(RAW, "ggd200/deepwell.dat")
    with open(path, encoding="latin-1") as f:
        for r in csv.reader(f):
            if len(r) < 14:
                continue
            c0 = r[0].strip()
            if not c0.isdigit():         # 사이트 시작 행에만 index
                continue
            name = r[1].strip().strip('",; ')
            try:
                nl, el = float(r[2]), float(r[3])
            except ValueError:
                continue
            if nl <= 0 or el <= 0:
                continue
            lat = int(nl // 100) + (nl % 100) / 60.0     # DDMM → decimal
            lon = int(el // 100) + (el % 100) / 60.0
            for d, idx in zip(depths, range(13, 23)):
                if idx >= len(r):
                    break
                v = r[idx].strip().replace(" ", "")
                if v in ("", "-", "?"):
                    continue
                try:
                    t = float(v)
                except ValueError:
                    continue
                rows.append(dict(source="GGD200", site=name, lat=round(lat, 4),
                                 lon=round(lon, 4), depth_m=d, temp_c=t,
                                 year=np.nan, elev_m=np.nan, gtnp_id=""))
    return pd.DataFrame(rows)

# ---------------- PERMOS (스위스 알프스, 연평균 지중온도) ----------------
def parse_permos():
    base = glob.glob(os.path.join(RAW, "permos/extracted/*/Data"))[0]
    bh = pd.read_csv(glob.glob(f"{base}/borehole_*.csv")[0])
    bhy = pd.read_csv(glob.glob(f"{base}/bht_year_*.csv")[0])
    m = bhy.merge(bh[["id", "name", "lat", "lon", "h", "gtnp_id"]],
                  left_on="borehole_id", right_on="id", how="left")
    out = pd.DataFrame(dict(
        source="PERMOS", site=m["name"], lat=m["lat"].round(4), lon=m["lon"].round(4),
        depth_m=m["depth"], temp_c=m["temp"], year=m["time"],
        elev_m=m["h"], gtnp_id=m["gtnp_id"].fillna("").astype(str),
    ))
    return out.dropna(subset=["lat", "lon", "temp_c"])

ggd = parse_ggd200()
per = parse_permos()
allgt = pd.concat([ggd, per], ignore_index=True)
allgt.to_csv("data/processed/ground_temp.csv", index=False)

def summ(df, nm):
    print(f"[{nm}] rows={len(df)}, sites={df['site'].nunique()}, "
          f"depth {df.depth_m.min():.0f}~{df.depth_m.max():.0f}m, "
          f"temp {df.temp_c.min():.1f}~{df.temp_c.max():.1f}°C")
summ(ggd, "GGD200 서시베리아")
summ(per, "PERMOS 스위스")
print(f"[통합] {len(allgt)} rows → data/processed/ground_temp.csv")
n_gtnp = (per["gtnp_id"].astype(str).str.strip() != "").sum()
print(f"  (PERMOS 중 gtnp_id 보유 = GTN-P 중복가능: {n_gtnp} rows)")

# ================= 시각화 1: 신규 지중온도 사이트 세계지도 =================
try:
    gtnp = pd.read_csv("data/processed/gtnp_borehole_sites.csv")
except Exception:
    gtnp = None
    for cand in glob.glob("data/processed/*.csv"):
        try:
            d = pd.read_csv(cand)
            if {"lat", "lon"}.issubset(d.columns) and "gtnp" in cand.lower():
                gtnp = d; break
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(13, 6.5))
if gtnp is not None and {"lat", "lon"}.issubset(gtnp.columns):
    ax.scatter(gtnp.lon, gtnp.lat, s=8, c="lightgray", label=f"기존 GTN-P ({gtnp['lat'].notna().sum()})",
               zorder=1, edgecolors="none")
gg = ggd.drop_duplicates("site")
pp = per.drop_duplicates("site")
ax.scatter(gg.lon, gg.lat, s=45, c="#d62728", marker="^",
           label=f"NEW GGD200 서시베리아 ({len(gg)} 사이트)", zorder=3, edgecolors="k", linewidths=0.4)
ax.scatter(pp.lon, pp.lat, s=45, c="#1f77b4", marker="s",
           label=f"NEW PERMOS 스위스 ({len(pp)} 사이트)", zorder=3, edgecolors="k", linewidths=0.4)
ax.set_xlim(-180, 180); ax.set_ylim(20, 85)
ax.axhline(66.56, color="navy", ls="--", lw=0.7, alpha=0.6)
ax.text(-178, 67.3, "북극권 (66.56°N)", color="navy", fontsize=8)
ax.set_xlabel("경도"); ax.set_ylabel("위도")
ax.set_title("신규 확보 지중온도(borehole) 사이트 — GTN-P 공간 격차 보강", fontsize=13, weight="bold")
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("outputs/figures/15_ground_temp_map.png", dpi=130)
plt.close(fig)
print("saved outputs/figures/15_ground_temp_map.png")

# ================= 시각화 2: 깊이-온도 프로파일 =================
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
# GGD200: 정적 심부 프로파일 (동결 근지표 → 온난 심부 = 지열)
ax = axes[0]
for name, g in ggd.groupby("site"):
    if len(g) >= 4:
        g = g.sort_values("depth_m")
        ax.plot(g.temp_c, g.depth_m, "-o", ms=3, lw=0.8, alpha=0.55)
ax.axvline(0, color="k", ls="--", lw=1)
ax.invert_yaxis(); ax.set_yscale("log")
ax.set_xlabel("암석온도 (°C)"); ax.set_ylabel("깊이 (m, log)")
ax.set_title(f"GGD200 서시베리아 심부 프로파일\n(≥4깊이 사이트, 동결↔지열 심부)", fontsize=10)
ax.grid(alpha=0.3)
# PERMOS: 얕은(0~20m) 연평균 프로파일 예시
ax = axes[1]
recent = per[per.year == per.year.max()]
shown = 0
for name, g in recent.groupby("site"):
    if len(g) >= 3 and shown < 12:
        g = g.sort_values("depth_m")
        ax.plot(g.temp_c, g.depth_m, "-o", ms=3, lw=0.9, alpha=0.7, label=name)
        shown += 1
ax.axvline(0, color="k", ls="--", lw=1)
ax.invert_yaxis()
ax.set_xlabel("연평균 지중온도 (°C)"); ax.set_ylabel("깊이 (m)")
ax.set_title(f"PERMOS 스위스 얕은 프로파일 ({int(per.year.max())}년)\n(0°C 좌측=동결)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2, loc="lower left")
fig.suptitle("깊이별 지중온도 프로파일 — 우리 3D 열구조 학습의 핵심 라벨", fontsize=12, weight="bold")
fig.tight_layout(); fig.savefig("outputs/figures/16_ground_temp_profiles.png", dpi=130)
plt.close(fig)
print("saved outputs/figures/16_ground_temp_profiles.png")
