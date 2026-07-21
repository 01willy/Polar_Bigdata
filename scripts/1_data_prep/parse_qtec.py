"""TPDC QTEC(칭짱공정회랑, Zenodo 5009871) 지중온도 파싱 → 지온 프로파일 + ALT 라벨.

입력(data/raw/tpdc_qtec_zenodo5009871/)
- GT_52818_Golmud_1955-2020.csv     일단위, Year/Month/Day + GT_{0..320}_AVG, 결측 32766
- GT_52908_Wudaoliang_1956-2020.csv 일단위, 동일 스키마(' GT_0_MIN' 앞공백 주의)
- GT_00000_Slopes_2014-2019.csv     월평균, Soil_Temp_1cm~260cm

좌표
- Wudaoliang 35.22N 93.08E, Golmud 36.42N 94.90E (WMO 지점좌표).
  README의 Space scope(35°39'N, 90°3'E)는 경도 오타로 보인다(같은 README의 지도
  embed는 94.06E, 35.65N을 가리킴). WMO 공식 지점좌표를 사용한다.
- Slopes(사면 관측지)는 README 지도 중심 좌표 35.6517N 94.0612E(쿤룬산구 인근)를 사용.

산출
(a) data/processed/ground_temp_qtec.csv  사이트·깊이별 연평균/연최대 온도
(b) data/processed/alt_qtec_cell.csv     Wudaoliang ALT(연최대 프로파일 0°C 교차 깊이)
    - 2014=280.0cm(GT_160/320은 11-12월만 관측되나 심부 최대는 하계 융해 신호를 반영),
      2019=316.2cm 두 연도만 유효.
    - 2015는 교차깊이=320cm(센서 하단, GT_320 연최대=0.0)로 censored 제외.
    - 2016/2017/2018/2020은 GT_320 연최대>0(0.2~0.8°C)로 교차 없음(ALT>320cm censored).
    - censored 연도 제외로 다년 평균 alt_cm은 하한 성격(하향 편향)이며, 이를
      censor_flag=1로 데이터에 명시한다(침묵 편향 방지). site='Wudaoliang' 병기.
    - Golmud은 GT_320 연평균 약 +8°C(계절동토, 비영구동토)이므로 ALT 라벨에서 제외.
실행: /home/anaconda3/bin/python scripts/1_data_prep/parse_qtec.py  (ROOT에서)
"""
import numpy as np
import pandas as pd

RAW = "data/raw/tpdc_qtec_zenodo5009871"
MISS = 32766
SITES = {
    "Golmud": dict(file=f"{RAW}/GT_52818_Golmud_1955-2020.csv", lat=36.42, lon=94.90),
    "Wudaoliang": dict(file=f"{RAW}/GT_52908_Wudaoliang_1956-2020.csv", lat=35.22, lon=93.08),
}
SLOPES = dict(file=f"{RAW}/GT_00000_Slopes_2014-2019.csv", lat=35.6517, lon=94.0612)
MIN_DAYS = 300     # 일단위 연집계 최소 유효일수
MIN_MONTHS = 10    # 월단위 연집계 최소 유효월수

recs = []
annual_max = {}    # site -> {year: {depth_cm: 연최대}}  (ALT 산출용, 일수 필터 없이 별도 보관)

# ---------- 일단위 두 지점 ----------
for site, info in SITES.items():
    df = pd.read_csv(info["file"])
    df.columns = [c.strip() for c in df.columns]   # ' GT_0_MIN' 앞공백 정리
    df = df.replace(MISS, np.nan)
    dcols = {c: int(c.split("_")[1]) for c in df.columns
             if c.startswith("GT_") and c.endswith("_AVG")}
    for c, dcm in dcols.items():
        g = df.groupby("Year")[c].agg(["mean", "max", "count"])
        ok = g[g["count"] >= MIN_DAYS]
        if len(ok) == 0:
            continue
        recs.append(dict(site=site, lat=info["lat"], lon=info["lon"], depth_m=dcm / 100.0,
                         temp_c_mean=float(ok["mean"].mean()),
                         temp_c_max=float(ok["max"].mean()),
                         n_years=int(len(ok))))
    # ALT용 연최대(부분연도 포함; 2014 GT_160/320은 11-12월만 존재)
    am = {}
    for y, g in df.groupby("Year"):
        prof = {dcm: float(g[c].max()) for c, dcm in dcols.items() if g[c].notna().any()}
        if prof:
            am[int(y)] = prof
    annual_max[site] = am

# ---------- Slopes 월평균 ----------
df = pd.read_csv(SLOPES["file"])
df = df.replace(MISS, np.nan)
dcols = {c: int(c.replace("Soil_Temp_", "").replace("cm", ""))
         for c in df.columns if c.startswith("Soil_Temp_")}
for c, dcm in dcols.items():
    g = df.groupby("Year")[c].agg(["mean", "max", "count"])
    ok = g[g["count"] >= MIN_MONTHS]
    if len(ok) == 0:
        continue
    recs.append(dict(site="Slopes", lat=SLOPES["lat"], lon=SLOPES["lon"], depth_m=dcm / 100.0,
                     temp_c_mean=float(ok["mean"].mean()),
                     temp_c_max=float(ok["max"].mean()),
                     n_years=int(len(ok))))

gt = pd.DataFrame(recs).sort_values(["site", "depth_m"]).reset_index(drop=True)
gt["region"] = "QTP_CN"
gt["source"] = "TPDC_QTEC"
gt.to_csv("data/processed/ground_temp_qtec.csv", index=False)
print(f"[saved] data/processed/ground_temp_qtec.csv: {len(gt)}행 "
      f"(사이트 {gt.site.nunique()}개, 깊이 0~{gt.depth_m.max():.1f}m)")
gol320 = gt[(gt.site == "Golmud") & (gt.depth_m == 3.2)]
print(f"[check] Golmud GT_320 연평균 {gol320.temp_c_mean.iloc[0]:.1f}°C "
      f"(+8°C대, 비영구동토 -> ALT 라벨 제외)")

# ---------- Wudaoliang ALT: 연최대 프로파일의 0°C 교차 깊이 ----------
def cross_depth(prof):
    """깊이 오름차순 연최대 프로파일에서 가장 깊은 +→- 교차의 선형보간 깊이(cm)."""
    ds = sorted(prof)
    alt = None
    for i in range(len(ds) - 1):
        t0, t1 = prof[ds[i]], prof[ds[i + 1]]
        if t0 > 0 and t1 <= 0:
            alt = ds[i] + (ds[i + 1] - ds[i]) * t0 / (t0 - t1)
    return alt

alts = {}
n_censored = 0
for y, prof in sorted(annual_max["Wudaoliang"].items()):
    if 320 not in prof or 160 not in prof:
        continue                       # 심부 센서 없는 연도(2013 이전) 제외
    a = cross_depth(prof)
    status = "valid"
    if a is None:
        status = "censored(ALT>320cm, GT_320 연최대>0)"
    elif a >= 320.0:
        status = "censored(교차=센서 하단 320cm)"
        a = None
    print(f"[wudaoliang] {y}: GT_160max={prof[160]:+.2f} GT_320max={prof[320]:+.2f} "
          f"-> {'ALT %.1fcm' % a if a else status}")
    if a is not None:
        alts[y] = a
    else:
        n_censored += 1

vals = np.array([alts[y] for y in sorted(alts)])
years = sorted(alts)
assert years == [2014, 2019], f"유효 연도 예상(2014,2019)과 불일치: {years}"
assert abs(alts[2014] - 280.0) < 0.5 and abs(alts[2019] - 316.2) < 0.5, alts

row = dict(lat=SITES["Wudaoliang"]["lat"], lon=SITES["Wudaoliang"]["lon"], region="QTP_CN",
           alt_cm=float(vals.mean()),
           alt_sd=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
           alt_min=float(vals.min()), alt_max=float(vals.max()),
           n_obs=int(len(vals)), year_min=int(min(years)), year_max=int(max(years)),
           n_years=int(len(years)),
           censor_flag=int(n_censored > 0),   # censored 연도 제외 -> 라벨 하한 성격
           site="Wudaoliang")
out = pd.DataFrame([row])
out.to_csv("data/processed/alt_qtec_cell.csv", index=False)
print(f"[saved] data/processed/alt_qtec_cell.csv: 1행 "
      f"(alt_cm={row['alt_cm']:.1f} sd={row['alt_sd']:.1f}, {years}, "
      f"censored 연도 {n_censored}개 제외 -> censor_flag={row['censor_flag']})")
