# 격자 안 입력(토양 수분·유기층·식생·적설·미지형) 검토와 XE_hires_covariates 설계

- 작성일: 2026-10-04
- 대상 질문(사용자 질문 3): "토양 수분·유기층·식생·적설 같은 격자 안 입력 자료를 선행 논문에서는 안 썼나? 우리가 확보한 자료나 확보할 수 있는 자료에 없었는가?"
- 범위: (1) 2010–2026 문헌 점검, (2) 보유 자료 점검, (3) 취득 가능 제품 점검, (4) XE_hires_covariates 사전 등록안.
- 이 문서는 모형을 돌리지 않았다. 3.3절의 분산 분해는 라벨만 쓴 자료 서술이고(모형 예측 없음), 공변량과 ALT 의 관계는 계산하지 않았다(사전 등록 전 금지 항목).
- 근거 표기: 로컬 파일은 저장소 상대 경로, 웹 자료는 URL. 문헌 검증 수준은 A(원문 PDF 또는 전문을 직접 읽음), B(출판사·저장소 페이지를 WebFetch 로 읽음. 요약 모델을 거치므로 문맥 오류 가능), C(검색 결과 요약만 확인), [미확인](찾지 못했거나 확인하지 못함)로 적는다.

---

## 1. 답변 요약

1. **선행 연구도 이 변수들을 썼다.** 다만 범북극·지역 지도 연구는 대부분 250 m–1 km 제품(MODIS LAI·NDVI, 적설 지속 기간 0.05°, SoilGrids SOC, 토지 피복, DEM 지형 습윤 지수)을 썼고, 중요도는 기후(TDD·FDD·연평균 기온)와 일사가 앞섰다(Ran 2022, Karjalainen 2019, Gautam 2025). 토양 수분 제품은 통계·ML 지도 연구에서 거의 쓰이지 않았고, 과정 모형(Yi 2018)에서 SMAP 9 km 로 쓰였으며 ALT 불확실성 기여는 SOC·적설 밀도보다 작았다. 유기층 두께 지도를 ALT 예측 입력으로 직접 쓴 범지역 ML 연구는 이번 검색 범위에서 찾지 못했다(SOC 가 대용으로 쓰였다).
2. **격자 안 변동을 맞힌 사례는 수 km² 규모의 현장 연구뿐이다.** LiDAR 0.25 m 와 WorldView-2 NDVI 2 m 로 Barrow 5 km² 에서 r² 0.76(Gangodagamage 2014), UAS·AVIRIS-NG·Sentinel-2 로 Seward 반도에서 설명 분산 8–39 %(Hantson 2025)다. 해상도를 30 m 에서 1 km 로 거칠게 할 때 정규화 오차 증가는 약 5 % 로, 1 m 에서 30 m 구간(약 10 %)보다 작았다(Du 2026). 즉 ALT 의 공간 변동 대부분은 30 m 이하 규모에 있다.
3. **우리 보유 자료에는 격자 안에서 변하는 입력이 일부 이미 있다.** x25 의 DEM 6열은 라벨 점마다 계산한 값이라 1 km 위치 안에서도 변한다(알래스카 행의 99 % 가 위치 안 값이 2개 이상인 위치에 있다). 그런데도 WF9 에서 직접 ML(D0)의 격자 안 설명 비율은 알래스카 −2.0 % 였다. 격자 안에서 값이 같은 입력은 ERA5(8열), SoilGrids(9열, 약 5 km 추출), MODIS CMG 0.05°(H19), 대부분의 CCI 다. 토양 수분은 ERA5-Land 격자값과 라벨 쪽 현장 측정(ABoVE VWC)만 있다.
4. **확보 가능한 공개 제품**: 30 m 이하(ArcticDEM 2·10 m, ESA WorldCover 10 m, Copernicus DEM 30 m, Hansen·GSW 30 m), 250–500 m(SoilGrids 250 m, MODIS NDVI 250 m, MODIS 적설 500 m), 1 km(NCSCD, CAVM)는 계정 범위 안에서 확보할 수 있다. **북위 60° 이북에서 3 km 이하 토양 수분 위성 제품은 없다**(SMAP/Sentinel-1 3 km 는 60°S–60°N). 유기층 두께는 범지역 제품이 없다.
5. **XE 의 목적은 하나다.** 취득 가능한 고해상 입력을 더하면 물리 기반 ML(R1)이 ERA5 격자 안 변동을 맞히는지, WF9 와 같은 분할·라벨 수·분해로 사전 등록해 판정한다. 라벨만 쓴 자료 서술로 보면 알래스카 격자 안 분산의 약 49 % 는 100 m 셀 안 변동(미지형·관측 잡음·연도 차이)이어서 30–500 m 입력으로 닿을 수 있는 몫은 그 나머지(약 51 %)가 상한이다.

---

## 2. 선행 연구 점검(2010–2026)

### 2.1 연구별 정리

| 연구 | 규모·해상도 | 격자 안 성격의 입력(제품·해상도) | 보고된 효과·중요도 | 검증 방식 | 검증 수준 |
|---|---|---|---|---|---|
| Ran et al. 2022, ESSD 14, 865–884 | 북반구, 1 km | FDD·TDD(MODIS LST 1 km), LAI(GLASS 1 km), 적설 지속 기간 SCD(Hori 2017, 0.05°), SoilGrids250(SOC·용적밀도·조립질, 30″로 집계), WorldClim 일사·강수 | ALT 사이트 452곳. 앙상블 ALT RMSE 86.93 ± 19.61 cm. FDD·TDD·SCD·LAI·SOC·일사가 유의, 강수·용적밀도는 유의하지 않음 | 거리 블록 10겹 CV, 1,000회 | A (`references/00_core10/ran2022_panarctic.pdf`) + B |
| Karjalainen et al. 2019, Sci Data | 범북극, 30″(약 1 km) | TDD, FDD, 0 °C 위·아래 강수 합, 잠재 일사(PISR), SOC(SoilGrids1km). 수체는 MAGT 모형에만 | ALT 사이트 303곳. ALT 는 GLM 예측만 사용. 현재 조건 R² 0.37(요약 경유, 원문 문맥 [미확인]) | 500 km 거리 블록 CV | B |
| Karjalainen et al. 2019, The Cryosphere 13, 693 | 북반구, 30″ | FDD, TDD, 강우, 강설, 일사, NDVI(MODIS), SOC, 조립·세립 퇴적물 | ALT n = 298. ALT 중요도: 일사 0.37, 강우 0.05, SOC 0.04, 조립 퇴적물 0.03(원문 인용) | 무작위 70/30, 100회 | B |
| Aalto et al. 2018, GRL 45, 4889–4898 | 범북극, 30″ | 기후 + 국지 요인(지형·일사, SoilGrids1km SOC) | 국지 예측변수를 넣으면 ALT 모형이 개선(GAM 제외). RF RMSE 89 cm, 앙상블 104 cm | 거리 블록 CV | C (원문 403, 자료 기록은 B) |
| Gautam et al. 2025, Sci Rep 15, 42420 | 알래스카, 250 m | DEM 250 m 지형 지수(경사·향·유량 누적·SPI·STI·TWI), WorldClim 1 km, ESA 토지 피복(논문 표기 250 m), 암상 | CALM 68곳(2014). RF 시험 R² 0.24(RMSE 22 cm) 대 Stefan 0.54(18 cm). 중요도: 연평균 기온 19 %, 경사 18 %, STI 14 %, SPI 11 %, 강수 약 10 %. 토양 습윤·토지 피복은 하위. 토양 수분·적설·유기층 입력 없음 | 무작위 70/30 1회 | A (`references/00_core10/gautam2025_alaska_alt.pdf`) |
| Pastick et al. 2015, RSE 168, 301–315 | 알래스카, 30 m | Landsat 7 ETM+ 모자이크(2008–2011) 밴드·NDVI·gNDVI·NDII, DEM(고도·향·CTI·잠재 일사), 토지 피복, 습지, 축소 기후 | ALT 회귀 나무. 무작위 보류 n = 216: MAE 13 cm, MBE −1 cm, r 0.61. 1 m 넘는 ALT 는 과소 예측 | 무작위 10겹 CV + 무작위 보류 + 독립 자료(MAE 15 cm) | A (`references/02_alt_dl_mapping/pastick2015_nsp_alaska.pdf`) |
| Mishra & Riley 2014, SSSAJ 78, 894–902 | 알래스카, 60 m | 지형, 기후, 토지 피복(지리 가중 회귀) | 토양 단면 153개. 예측 오차 0.11 m, RPD 1.8 | [미확인] | C |
| Yi et al. 2018, The Cryosphere | 알래스카, 1 km 과정 모형 | MODIS LST 1 km, SMAP L4 토양 수분 9 km, SOC 50 m, NLCD 30 m, MERRA-2 적설 0.5° + MODIS 적설 500 m | SOC 불확실성이 ALT 에 미치는 영향은 연속 영구동토 약 5 %, 산발 영구동토 약 45 %. 낮은 적설 밀도 시나리오에서 ALT +56 %. SMAP 토양 수분의 기여는 SOC·적설 밀도보다 작음 | 레이더 산출 ALT 와 비교 | B |
| ESA CCI Permafrost(CryoGrid CCI), PUG v3.0 | 북반구, 1 km | MODIS LST, ESA Land Cover CCI 로 지층 배정 | 알려진 한계: 지층. "극단적인 경우 수 m 편차" | 시추공·현장 비교 | A (`references/01_benchmark/esa_cci_permafrost_v4.pdf`) |
| Obu et al. 2019, ESR 193, 299–316 | 북반구, 1 km TTOP | MODIS LST, 축소 ERA-Interim, 툰드라 습윤 등급, ESA CCI 토지 피복. 화소마다 200개 앙상블로 적설·피복의 화소 안 변동 표현 | MAGT 정확도 ±2 °C(시추공 대비). ALT 는 산출하지 않음 | 시추공 | A (`references/02_alt_dl_mapping/obu2019_ttop_nh.pdf`) |
| Whitcomb et al. 2024, ERL 19, 014046 | 북부 알래스카, 30 m | AirMOSS P-band PolSAR ALT 를 RF 로 확장. 입력: NLCD 피복, 고도·경사·향·잠재 일사, SOC·용적밀도·점토·모래, TDD·FDD, 계절 기온, 여름 월별 NDVI, 수체 근접도 | CALM 대비 RMSE 약 11–12 cm, 검증 화소 대비 7.5–9.1 cm | PolSAR 검증 화소, CALM 23곳 이상 | A (PDF 텍스트) |
| Chen et al. 2019(ORNL DAAC 1657) | 북부 알래스카 비행선 12곳, 30 m | P-band SAR 로 ALT 와 토양 수분 단면을 함께 산출 | 자료 제품(2014·2015·2017) | 현장 비교 | B |
| Clayton et al. 2021, ERL 16, 055028 | 알래스카·캐나다 48곳 | 현장 VWC(HydroSense 6·12·20 cm, GPR·DualEM 층 평균) | 동시 관측 7,499건. 층 평균 VWC 는 ALT 와 음의 관계, 상부 12 cm VWC 는 양의 관계, 20 cm 는 관계 없음. R² 낮음. 관계는 사이트 사이에서 일관 | 회귀(사이트·지역) | B |
| Gangodagamage et al. 2014, WRR 50, 6339–6357 | Barrow 5 km², 2 m | LiDAR 0.25 m(경사·곡률), WorldView-2 NDVI 2 m | r² 0.76, RMSE ±4.4 cm | 현장 측정 대비 | A (PDF 텍스트) |
| Hantson et al. 2025, Environ. Res.: Ecol. 4, 015001 | Seward 반도 2곳 | UAS(5.2·10 m), AVIRIS-NG, Sentinel-2 | RF 설명 분산 8–39 %, RMSE 9.91–21.91 cm. 최적 해상도는 지형·식생 유형에 따라 다름(물길·관목 지형은 5 m) | RF 출력 기준 | A (PDF 텍스트) |
| Du et al. 2026, The Cryosphere 20, 4277–4291 | North Slope 구릉, 5 m 격자 | 드론 광학·다중분광, Sentinel-2, 경사·향·고도·NDVI·NDWI·적색 경계 | 탐침 약 1,700건. 5 m RF RMSE 6.53 cm. 0.1 m 기준 대비 정규화 RMSE 는 1 m 까지 약 10 %, 1–30 m 에서 약 10 %, 30–1,000 m 에서 약 5 % 증가 | 현장 측정, 해상도별 집계 | B |
| Thaler et al. 2023, Earth Space Sci. 10, e2023EA003015 | Seward 반도 유역 3곳, m 규모 | LiDAR 지형·식생, 고해상 위성 적설 지수 | 근지표 영구동토 유무 분류. NDVI 가 가장 중요. 사이트 안 정확도 70–90 %, 다른 사이트 전이 50–77 % | 사이트 안·사이트 간 | C |
| Yin et al. 2024, IntechOpen | 캐나다 NWT | Sentinel-1·2, Landsat-8, PALSAR, 지형 등 116 변수 | RF R² 0.476(검증), 교차검증 0.432 | k겹 CV(공간 분리 여부 [미확인]) | A (`references/02_alt_dl_mapping/yin2024_alt_upscaling_airborne_gee.pdf`) |
| Zhang C. et al. 2021, IJAEO 102, 102455 | 내륙 알래스카 | 항공 초분광 | 6년 ALT, ML 앙상블. 수치 [미확인] | [미확인] | C |
| Li C. et al. 2022, JGR Atmos. | 북반구, 1 km | Stefan 식 + ERA5-Land(격자 안 입력 없음) | 2000–2018 평균 ALT 127.19 → 145.37 cm | [미확인] | C |
| Peng et al. 2018, J. Clim. 31, 251–266 / Peng et al. 2023, Earth's Future | 북반구 / 25 km(Gautam 2025 서술) | [미확인](원문 403) | [미확인] | [미확인] | C |
| Siewert et al. 2021, GBC | 툰드라 3개 지형 | 토양 단면 SOC·지하 얼음 | SOC 변동 계수: 단면 규모 21–73 %, 경관 규모 24–67 % | 서술 | C |
| Schaefer et al. 2015, Remote Sensing(ReSALT) | Barrow | InSAR 침하 | GPR 대비 검증. 수치 [미확인] | GPR | C |
| Streletskiy et al. 2012, Polar Geography 35, 95–116 | 북부 알래스카 | 경관별 ALT 변동 | [미확인] | [미확인] | C |
| Nyland et al.(과업 문서 언급) | | 해당 저자의 ALT 지도화 연구를 찾지 못함 | | | [미확인] |

### 2.2 변수별 정리

| 변수 | 선행 연구의 사용 | 해상도 | 관찰 |
|---|---|---|---|
| 토양 수분 | 과정 모형 입력(Yi 2018, SMAP 9 km), P-band 산출(Chen 2019, 30 m 비행선), 현장 VWC(Clayton 2021). 통계·ML 지도에서는 TWI·CTI·NDII 같은 대용 변수(Gautam 2025, Pastick 2015) | 9 km(제품), 30 m(비행선), 점(현장) | Yi 2018 에서 기여가 작았고, Clayton 2021 에서 측정 깊이에 따라 부호가 바뀌고 R² 가 낮았다 |
| 유기층 두께 | 범지역 ML 의 직접 입력으로는 찾지 못함. SOC 가 대용(Ran 2022, Aalto 2018, Karjalainen 2019, Whitcomb 2024). 과정 모형은 SOC 50 m 로 유기층을 배분(Yi 2018) | 250 m–1 km(SoilGrids), 50 m(Yi) | 과정 모형에서는 가장 큰 불확실성 요인(Yi 2018). 범지역 SOC 지도는 단면 규모 변동(Siewert 2021)을 담지 못한다 |
| 식생 | LAI 1 km(Ran 2022), NDVI(Karjalainen TC 2019, Whitcomb 2024, Pastick 2015, Thaler 2023), 토지 피복(Gautam 2025, Mishra & Riley 2014, CCI·Obu) | 2 m–1 km | 현장·국지 연구에서 NDVI 가 상위 중요도(Thaler 2023, Gangodagamage 2014). 범북극 연구에서는 기후보다 하위 |
| 적설 | SCD 0.05°(Ran 2022), 강설 합(Karjalainen 2019), MERRA-2 + MODIS 500 m(Yi 2018), 화소 안 적설 앙상블(Obu 2019) | 500 m–0.5° | Ran 2022 에서 유의, Yi 2018 에서 밀도 가정이 ALT 를 크게 바꿈. 미세 적설 재분배(수 m)는 UAV 연구에만 있다 |
| 미지형 | LiDAR 0.25 m(Gangodagamage 2014), UAS(Hantson 2025, Du 2026), DEM 250 m 지수(Gautam 2025) | 0.25 m–250 m | 국지에서 경사·곡률·향이 상위. 30 m 이상에서는 일사·경사 정도만 남는다 |

### 2.3 문헌에서 얻는 설계 시사점

- 범지역 연구의 검증은 무작위 분할(Gautam 2025, Pastick 2015, Karjalainen TC 2019) 또는 거리 블록(Ran 2022, Aalto 2018, Karjalainen Sci Data 2019)이었다. 기후 격자 안 성분을 따로 떼어 고해상 입력의 몫을 잰 연구는 이번 검색 범위에서 찾지 못했다(검색 범위 한정 진술이다).
- 고해상 입력이 큰 설명력을 보인 연구는 모두 수 km² 안의 현장 연구이고, 사이트 사이 전이는 약했다(Thaler 2023: 50–77 %). XE 는 이 차이를 시험하는 위치에 있다. 같은 블록 분할(지역 내, 블록 단위 홀드아웃)에서 30–500 m 공개 제품이 격자 안 변동을 맞히는지를 본다.
- Du 2026 의 해상도별 오차 곡선은 ALT 변동의 상당 부분이 1 m 이하와 1–30 m 구간에 있음을 보인다. 이 결과와 3.3절의 자료 서술(100 m 셀 안 분산이 알래스카 격자 안 분산의 약 49 %)은 같은 방향이다. 따라서 XE 의 사전 기대는 '작은 이득 또는 동등' 이며, 해석 규칙도 이 경우를 먼저 적는다.

---

## 3. 보유 자료 점검

### 3.1 현재 입력과 격자 안 변동 여부

알래스카 라벨 행(13,606)에 대해 1 km 위치(EPSG:3338 1 km 정사각 격자, 원점에 따라 333곳. QA 문서의 343곳과 원점 차이) 안에서 값이 2개 이상인 위치에 속한 행의 비율을 셌다(`data/processed/fidelity_base_v3.csv`, `data/processed/covariates_ext_v1.csv`, 계산 코드는 부록 A).

| 입력 | 원자료 해상도 | 추출 방식(근거) | 고유값 수(알래스카) | 위치 안 변동 행 비율 | 격자 안 성격 |
|---|---|---|---|---|---|
| `dem_elev`, `dem_slope`, `dem_aspect_*` | Copernicus DEM 30 m | 라벨 점의 중심 화소(`scripts/1_data_prep/terrain_features_dem.py`) | 경사 6,027 | 0.99 | 점 규모(이미 x25 에 있음) |
| `dem_tpi`, `dem_rough` | 30 m | 점을 중심으로 한 33 × 33 창(약 1 km) 대비 | 약 4,080 | 0.99 | 점 중심 이동 창 |
| H19 T군(`twi`, `flowacc_log`, `curv_*`, `rel_1km`) | 30 m, pysheds | 1 km 창 평균(`docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` 2.1) | 1,960 | 0.98 | 이동 창 평균 |
| H19 W군(`water_dist_km` 등) | JRC GSW 30 m | 거리·1·5 km 평균 | 12,827(거리) | 0.98 | 점 규모(거리) |
| H19 `treecover_1km` | Hansen 30 m | 1 km 평균 | 2,523 | 0.80 | 이동 창 평균 |
| H19 `ndvi_summer`, `lst_*` | MODIS CMG 0.05° | 최근접 격자 | 207, 197 | 0.39, 0.33 | 거친 격자 |
| `e5_*` 8열 | ERA5-Land 0.1° | 최근접 격자 | 164 | 0.22(격자 경계 위치) | 격자 |
| `sg_*` 9열 | SoilGrids 250 m 제품 | WCS 약 5 km 요청(`scripts/1_data_prep/enrich_soilgrids_wcs.py`, `data/processed/dl_dataset_cell_v3_soil_meta.json` 의 source = WCS) | 179(SOC 0–5) | 0.20 | 약 5 km |
| `cci_alt`, `cci_valid` | 약 1 km | 최근접 화소 | 150 | 0.70 | 1 km |
| SAR 9열(x34, 알래스카만) | 30 m(PolSAR) | 5 × 5 화소(150 m) 창 평균(`scripts/3_deep_learning/polsar_residual.py`) | | | 점 규모. WF1-b 기각(아래) |

정정 사항:
- 과업 문서의 "창 기반 공변량은 위치 안에서 모두 같다" 는 정확하지 않다. x25 의 DEM 6열과 H19 의 T·W군은 라벨 점마다 계산해 1 km 위치 안에서도 변한다. 위치 안에서 같은 것은 ERA5, SoilGrids(5 km 추출), MODIS CMG, 대부분의 CCI 다.
- QA 문서(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q8 표)의 "지형: 33 × 33 화소 창의 고도·경사·향·TPI·거칠기" 는 "고도·경사·향은 점 화소, TPI·거칠기는 33 × 33 창" 으로 고쳐야 한다. 같은 표의 "SAR: 5 km 안 점들의 통계" 는 PolSAR 의 경우 150 m 창이다(InSAR 는 이번에 확인하지 않음 [미확인]).
- 점 규모 지형은 이미 x25 에 들어 있었고, WF9 지역 내 라벨 전량에서 직접 ML(D0)의 격자 안 설명 비율은 알래스카 −2.0 %, 레나 6.2 %, 캐나다 −8.3 % 였다(`results/rescale_wf3/data/processed/wf/wf3b_decomp.csv`, `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5.3). 30 m 지형만으로는 격자 안 변동을 맞히지 못했다는 뜻이다.

### 3.2 디스크의 관련 원자료

| 자료 | 경로 | 크기 | 상태 |
|---|---|---|---|
| Copernicus DEM 30 m | `data/raw/dem/` | 455 타일, 8.6 GB | 알래스카·레나·티베트(N31–N38) 포함 |
| JRC GSW occurrence 30 m | `data/raw/gsw/` | 48 타일, 3.1 GB | 점 값 추출 가능(현재는 1·5 km 평균과 거리만 사용) |
| Hansen GFC treecover2000 30 m | `data/raw/hansen/` | 48 타일, 11 GB | 점 값 추출 가능(현재는 1 km 평균) |
| MODIS CMG(MOD11C3·MOD13C2) 0.05° | `data/raw/modis_cmg/` | 144 파일, 12 GB | 격자 규모 |
| SoilGrids 250 m 원해상도 창 | `data/raw/soilgrids_multi/` | 16 창, 81 MB, 7층(SOC 3깊이, bdod, clay, sand, silt) | 라벨 점 포함률: 알래스카 94.6 %(12,877/13,606), 레나 75.4 %(2,289/3,037), 캐나다 51.1 %(379/742). cfvo·phh2o 는 없음 |
| SoilGrids WCS 약 5 km | `data/raw/soilgrids_wcs/` | 11 MB | 현재 x25 의 출처 |
| ERA5-Land(적설 SWE·적설 깊이·토양 수분 swvl1·2) | `data/raw/era5land/` | 1.7 GB | 격자 규모(x25 의 `e5_swe`, H19 의 `e5_swvl*`, `e5_sde_winter`) |
| ABoVE ReSALT, PolSAR 상향 ALT 30 m | `data/raw/resalt/`, `data/raw/polsar_alt/` | 6.9 GB, 7.0 GB | ALT 유래 제품(입력 누설 위험). x34 로 시험, WF1-b 기각 |
| ABoVE 토양 수분·ALT 현장 자료 V2 | `data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` | 529,862행 | 라벨 쪽 현장 VWC(아래 3.4) |
| FireALT, Palmtag 2022 단면 | `data/raw/firealt_talucci2025/`, `data/raw/palmtag2022_pedon/` | | 화재·유기층 단면(지도 공변량 아님) |

기존 시험에서 확인된 것:
- **H19(전이)**: 지형 수문·수체·식생·ERA5 확장 23열을 더한 직접 ML 은 공변량만 AB4 조건에서 +0.64 cm [0.05, 1.12] 로 악화, 기각이었다(`docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` 237행). 지역 내 격자 안 분해로는 시험하지 않았다.
- **WF1-b(지역 내, 알래스카)**: SAR 9열(x34) R1 − x25 R1 은 n 500 +0.07, 1,000 −0.11, 2,000 −0.09, 5,000 −0.08(동등), 전량 −0.04(미결정)로 기각이었다(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 92행).
- **N2 오차 하한**: 거친 공변량 묶음(기후 8, 토양 9, CCI 2열) 안의 합동 표준편차는 알래스카 11.31, 레나 14.83, 캐나다 17.90 cm 다(`data/processed/lgx/lgx_floor.csv`, 정의는 `scripts/3_deep_learning/h42_label_grid_ext.py` floor_table). 이 값 아래로 내려가려면 거친 공변량 묶음 안에서 변하는 입력이 필요하다.

### 3.3 라벨 분산의 공간 규모별 분해(자료 서술, 모형 없음)

`data/processed/fidelity_base_v3.csv` 의 ALT 를 중첩 묶음으로 나눈 묶음 안 제곱합 비율이다. 격자 = (0.5° 블록, 기온 √TDD 값), 즉 WF9 기온 정의. 1 km·250 m·100 m 는 격자 안에 중첩한 EPSG:3338 정사각 셀이다(코드 부록 A). 레나에 알래스카 Albers 투영을 쓴 것은 서술용 근사이며, XE 본 실행은 LG 6B.3 셀 색인(0.009°)을 쓴다.

| 지역 | 셀 수 | 전체 SD(cm) | 블록 안 | 격자 안 | 1 km 안 | 250 m 안 | 100 m 안 | 격자 안 SD | 100 m 안 SD |
|---|---|---|---|---|---|---|---|---|---|
| 알래스카 | 13,606 | 17.38 | 0.535 | 0.504 (171묶음) | 0.409 (370) | 0.324 (910) | 0.248 (1,866) | 12.42 | 9.31 |
| 레나 | 3,037 | 21.10 | 0.757 | 0.627 (59) | 0.327 (223) | 0.204 (572) | 0.119 (1,038) | 16.86 | 8.98 |
| 캐나다 | 742 | 30.90 | 0.431 | 0.379 (50) | 0.312 (82) | 0.260 (110) | 0.228 (171) | 19.67 | 16.81 |

격자 안 분산을 다시 나누면(격자 안 = 100 %):

| 지역 | 격자 안·1 km 사이 | 1 km 안·100 m 사이 | 100 m 안 |
|---|---|---|---|
| 알래스카 | 18.8 % | 31.9 % | 49.2 % |
| 레나 | 47.8 % | 33.2 % | 19.0 % |
| 캐나다 | 17.7 % | 22.2 % | 60.2 % |

해석(서술):
- 1 km 제품(CCI, NCSCD, CAVM)이 닿을 수 있는 몫은 '격자 안·1 km 사이' 이고, 30–250 m 제품은 여기에 '1 km 안·100 m 사이' 를 더할 수 있다. 100 m 안 변동은 10 m 이하 자료(ArcticDEM 2 m, WorldCover 10 m) 외에는 닿지 않는다.
- 100 m 안 변동에는 관측 잡음이 섞여 있다. ABoVE 알래스카 행의 점별 ALT_err 중앙값은 GPR 8.33 cm(192,548행), 탐침 3.00 cm(12,533행)다. 소수 4자리 위치 13,568곳 가운데 2개 연도 이상 관측된 위치는 5.1 % 이고 관측 연도는 2005–2022 다(같은 파일). 즉 위치 사이 차이에는 연도 차이도 섞인다.
- 이 분산 비율은 ALT 자체의 분산이다. WF9 의 '물리식 오차 제곱의 72 % 가 격자 안' 은 P1 의 SSE 비율이라 정의가 다르다(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5.3).

### 3.4 ABoVE 현장 토양 수분(라벨 쪽 자료)

`data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` 를 가볍게 읽은 결과다(코드 부록 A).

- VWC(0–100 % 범위) 179,378행, 소수 4자리 위치 2,427곳, 연도 2008–2024. 기기 표기가 있는 행은 HydroSense II 9,208, HydroSense I 5,872, GPR 3,555, CD659 12 cm 막대 2,787, CD658 20 cm 막대 989, DualEM 424 이다. 측정 하단 깊이는 10 cm(64,466행), 6 cm(41,086), 18 cm(32,229), 20 cm(13,259), 12 cm(10,390) 순이다.
- 같은 위치·같은 날짜의 ALT·VWC 쌍은 2,465건(위치 1,961곳)이다.
- 라벨 셀 기준으로 15 m 안에 VWC 위치가 있는 셀은 알래스카 2,468셀(31블록), 캐나다 602/742셀(15블록)이다. 50 m 로 넓히면 알래스카 4,250셀이다.
- 이 자료는 라벨 위치에만 있으므로 지도 공변량이 될 수 없다. XE 에서는 '토양 수분을 완벽히 안다면 격자 안 변동을 얼마나 설명하는가' 의 상한 진단(XE-e)에만 쓴다.
- 주의: GPR 의 VWC 와 ALT 는 같은 유전율·전파 속도 추정에서 함께 나오므로 오차를 공유한다(설계 판단). 주 진단은 GPR 이 아닌 VWC(HydroSense, 막대, DualEM)로 하고 GPR VWC 는 민감도로만 둔다.

---

## 4. 취득 가능 제품

계정 상태: NASA Earthdata(`~/.netrc`), Copernicus CDS(`~/.cdsapirc`) 있음. TPDC·Google Earth Engine 없음. 지역 표기: AK 알래스카, LE 레나(71.5–73.6°N, 123.3–130.1°E), CA 캐나다 NWT·유콘(ABoVE_CA 라벨 위도 60.5° 이상), RU 러시아 CALM 사이트, TB 티베트. 점 추출 대상은 라벨 약 17,500셀과 레나 지도 격자 53,011셀(`data/processed/map_lena/lena_grid_cells_v1.csv`)이다. 소요 시간과 디스크는 추정값이다[추정].

| 변수군 | 제품 | 원해상도 | AK | LE | CA | RU | TB | 접근 | 보유 | 점 추출 작업량[추정] | XE 에서의 역할 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 미지형 | Copernicus DEM GLO-30 | 30 m | O | O | O | O | O | 공개 | 디스크 | 점 TWI·다중 규모 TPI 3–4 h | T2군 |
| 미지형 | ArcticDEM 모자이크 v4.1 | 2 m(10·32·100·500 m, 1 km 판도 있음) | O(2022-06 이전 공개분) | O | O | 60°N 이북만 | X | PGC·AWS STAC COG, 인용 규정 | 없음 | /vsicurl 창 읽기 4–6 h, 디스크 5 GB 이하 | M군 |
| 토양(SOC 등) | SoilGrids 2.0 | 250 m | O | O | O | O | O | ISRIC WebDAV VRT·WCS | 7층 일부(3.2) | 부족 창·cfvo·phh2o 2–3 h, 200 MB 이하 | O군 |
| 토양(SOC) | NCSCD v2 | 0.012°(약 1 km) | O | O | O | O | [미확인] | Bolin Centre·APGC | 없음 | 1 h, 1 GB 이하 | O군(1 km) |
| 유기층 두께 | 범지역 제품 없음. 유콘강 유역 30 m(USGS, Pastick·Wylie) | 30 m | 일부[미확인] | X | X | X | X | ScienceBase[미확인] | 없음 | 범위 확인 필요 | 서술 민감도 후보 |
| 식생 | MODIS MOD13Q1 v061(NDVI·EVI·NIR·MIR) | 250 m, 16일 | O | O | O | O | O | Earthdata, AppEEARS 점 추출 | 없음 | 점 요청 3–4 h + 대기. 점 수 상한 [미확인] | V군 |
| 토지 피복 | ESA WorldCover 2021 v200 | 10 m, 11등급 | O | O | O | O | O | AWS 무서명, CC BY 4.0 | 없음 | 필요 타일만 1–3 GB, 2 h | V군 |
| 토지 피복 | ABoVE 연간 우점 피복(ORNL 1691, 1984–2014) | 30 m | O | X | O | X | X | Earthdata(ORNL) | 없음 | 2–3 h | 알래스카 민감도 |
| 식생형 | CAVM 래스터 | 1 km | 툰드라만 | O | 툰드라만 | 툰드라만 | X | Mendeley Data | 없음 | 1 h, 수 MB | V군(1 km) |
| 레나 피복 | Schneider 2009(30 m), Lisovski 2025 서식지(10 m) | 30 m, 10 m | X | O | X | X | X | PANGAEA, CC BY 4.0 | 없음 | 1 h | 레나 민감도 |
| 적설 | MODIS MOD10A1 v061(NDSI 적설) | 500 m, 일 | O | O | O | O | O | Earthdata, AppEEARS 점 추출 | 없음 | 적설 지속 기간·소멸일 계산 3–4 h + 대기. 레나 지도는 타일 내려받기 수 GB | S군 |
| 적설 | ERA5-Land SWE·적설 깊이 | 0.1° | O | O | O | O | O | CDS | 디스크 | 0 | 기존(격자) |
| 토양 수분 | SMAP L4 SPL4SMGP v7 | 9 km | O | O | O | O | O | Earthdata | 없음 | | ERA5 와 같은 규모라 격자 안 입력 아님. 제외 |
| 토양 수분 | SMAP/Sentinel-1 SPL2SMAP_S | 3 km(내부 1 km 후방산란) | X | X | X | X | O | Earthdata | 없음 | | 60°S–60°N 만 제공. 제외 |
| 토양 수분 | ESA CCI SM | 0.25°[미확인] | O | O | O | O | O | CDS 등 | 없음 | | ERA5 보다 거침. 제외 |
| 토양 수분 대용 | 점 TWI(DEM), NDWI(MOD13Q1 NIR·MIR), WorldCover 습지 등급, GSW 점 수체 빈도 | 10–250 m | O | O | O | O | O | 위 제품 | 일부 디스크 | 위 작업에 포함 | T2·V군 |
| 수목·수체 | Hansen 30 m, GSW 30 m 점 값 | 30 m | O | O | O | O | O | 공개 | 디스크 | 1 h | T2군 |
| 토양 수분·ALT 산출 | AirMOSS P-band(ORNL 1657) | 30 m, 비행선 12곳 | 일부 | X | X | X | X | Earthdata | 없음 | | ALT 와 함께 산출(누설·오차 공유). 제외 |

요약:
- **토양 수분**: 북위 60° 이북에서 격자 안 해상도의 공개 위성 제품은 확인하지 못했다. XE 는 대용 변수(점 TWI, NDWI, 습지 등급, 수체 빈도)와 현장 상한 진단(XE-e)으로 대신한다.
- **유기층 두께**: 범지역 제품이 없다. SoilGrids 250 m SOC, NCSCD, 이끼·습지 피복 등급이 대용이다.
- **식생·적설·미지형**: 공개 제품으로 확보 가능하다. MODIS 는 AppEEARS 점 추출로 타일 전체를 받지 않아도 된다(NSIDC MOD10A1 페이지의 접근 방법 목록).
- **티베트**는 XE 대상이 아니다(지역 내 대상은 알래스카·레나·캐나다). 티베트만 3 km 토양 수분 제품 범위 안에 있다.

---

## 5. XE_hires_covariates 사전 등록안

### 5.1 질문

WF9 에서 물리 기반 ML(R1)은 x25 로 ERA5 격자 안 변동을 맞히지 못했다(WF9-a 기각, 라벨 전량 +0.09 cm 열세, 크기 0.5 cm 미만). 이 결과가 입력 해상도 때문인지 시험한다. "취득 가능한 공개 고해상 입력(10–500 m)을 더하면 R1 의 격자 안 RMSE 가 x25 보다 작아지고, 재보정 물리식(P1)보다도 작아지는가."

### 5.2 입력 집합(등록 시 열 목록 고정)

| 군 | 열(초안, 등록 때 확정) | 출처 | 지원 규모 |
|---|---|---|---|
| H0(취득 없음) | `twi`, `twi_sd`, `flowacc_log`, `curv_plan`, `curv_prof`, `rel_1km`, `water_occ_1km`, `water_occ_5km`, `water_dist_km`, `treecover_1km` (10열) | `data/processed/covariates_ext_v1.csv` | 점 중심 1 km 이동 창 |
| T2(디스크) | `twi_pt`(점 화소 TWI), `tpi_90`, `tpi_270`, `slope_90`(3 × 3 평균), `gsw_occ_pt`(3 × 3), `treecover_pt`(3 × 3) (6열) | Copernicus DEM, GSW, Hansen | 30–270 m |
| M(취득) | `ad_slope_30`, `ad_curv_30`, `ad_tpi_50`, `ad_tpi_150`, `ad_rough_50` (5열) | ArcticDEM 10 m | 30–150 m |
| O(일부 디스크) | `sg250_soc_0_5`, `sg250_soc_5_15`, `sg250_soc_15_30`, `sg250_bdod_5_15`, `sg250_clay_5_15`, `sg250_sand_5_15`, `sg250_silt_5_15`, `ncscd_soc_0_30`, `ncscd_soc_0_100` (9열) | SoilGrids 250 m, NCSCD | 250 m, 1 km |
| V(취득) | `ndvi250_jja`, `evi250_jja`, `ndwi250_jja`, `ndvi250_max`(2015–2020 6–8월), `wc_tree`, `wc_shrub`, `wc_grass`, `wc_wetland`, `wc_moss`, `wc_bare`, `wc_water`(WorldCover 3 × 3 = 30 m 창 등급 비율), `cavm_class` (12열) | MOD13Q1, WorldCover, CAVM | 30 m–1 km |
| S(취득) | `scd500`(9–8월 수문년 적설 일수, 2015–2020 평균), `snowoff_doy500`, `snowon_doy500` (3열) | MOD10A1 | 500 m |
| H | H0 ∪ T2 ∪ M ∪ O ∪ V ∪ S (45열) | | |

결합 규칙(등록 시 고정):
- 연도: MODIS 는 ERA5 기후값과 같은 2015–2020, WorldCover 는 2021 판. 라벨 연도(2005–2022, 위치당 대부분 1개 연도)와의 차이는 한계로 적는다.
- 위치 오차 흡수: 30 m 이하 제품은 점 화소가 아니라 3 × 3 창 통계를 주 값으로 쓴다. 점 화소 값은 민감도.
- 결측: 대상 지역에서 한 군의 유한값 비율이 90 % 미만이면 그 군은 그 대상에서 빼고 개정 이력에 적는다(예: 캐나다 SoilGrids 원해상도 창 부족 시 취득 후 재평가). 나머지 결측은 CatBoost 기본 결측 처리(x34 와 같은 규약)에 맡긴다.
- 누설 차단: ALT 를 직접 산출하거나 ALT 로 학습한 제품(CCI 추가 판, ReSALT, PolSAR 상향 ALT, AirMOSS)은 H 에 넣지 않는다. CCI 는 x25 의 기존 2열만 유지한다.
- 등록 전에 허용하는 확인: 새 열의 유한값 비율, 격자·위치 안 변동 비율(3.1절 표와 같은 방식), 새 열 사이의 상관. 금지: 새 열과 ALT 또는 P1 잔차의 어떤 통계도 등록 전에는 계산하지 않는다.

### 5.3 대상·분할·라벨 수·방법

- 대상: 알래스카, 레나, 캐나다(지역 내, 모드 r). WF6·WF9 와 같은 분할 1–25(half_split_blocks, 무효·중복 규칙 같음), n {200, 500, 1,000, 전량}(캐나다는 200·전량), 추출 3(전량 1), seed 0·1.
- 입력 변형: `x25`(재현용, WF9 저장소와 같은 키는 |ΔSSE| ≤ 1e-6 이어야 한다), `xh0` = x25 + H0, `xh` = x25 + H. 서술용 변형: x25 + 군 하나씩(T2, M, O, V, S), SoilGrids 교체 변형(5 km 9열 대신 250 m 열), 알래스카 ABoVE 피복 변형, 레나 Lisovski 2025 피복 변형.
- 방법: P1(라벨로 맞춘 E), R1(catboost_lo 잔차, 교차검증 λ ∈ {0.25, 0.5, 1.0}, λ 0.25 고정값도 저장), D0(catboost 기본 용량). WF9 와 같은 함수(`scripts/3_deep_learning/h54_workflow.py` R9Unit)를 부른다.
- 플랫폼: CatBoost·물리식만 쓰므로 Rescale CPU(elm)에서 한 플랫폼으로 돈다. 한 대비 안에서 플랫폼을 섞지 않는다. GPU 는 쓰지 않는다.

### 5.4 분해 정의

- 격자 묶음: WF9 구현 정의 그대로(채점 셀의 (기온 √TDD 값, 0.5° 블록)). 격자 안 RMSE 는 2셀 이상 묶음 셀의 (y − ȳ_g) − (ŷ − ŷ̄_g) 로 계산한다(`grid_groups`, `decomp_fns`).
- 위치 묶음(새로 더함): (블록, √TDD, LG 6B.3 셀 색인 ky = floor(lat/0.009), kx = floor(lon·cosφ/0.009)). 위치 묶음은 격자 묶음 안에 중첩된다. 저장소: `<대상>~l`(위치 안: 2셀 이상 위치 묶음 셀의 위치 평균 편차), `<대상>~gl`(격자 안·위치 사이: 위치 평균의 격자 평균 편차, 위치 셀 수 가중). 항등식 SSE_w(격자) = SSE(격자 안·위치 사이) + SSE(위치 안)을 단위 시험으로 확인한다.
- 설명 비율: 1 − SSE_w(M)/SSE_w(P1)(WF9-d 와 같은 정의), 격자 안·위치 안 각각.

### 5.5 가설

| ID | 종류 | 대비 | 조건 |
|---|---|---|---|
| XE-a | 주 | 격자 안 RMSE: R1(교차검증 λ, xh) − R1(교차검증 λ, x25) | 세 대상 층화 평균, n ∈ {500, 1,000, 전량}(캐나다는 전량만이라 '지역 2/3' 표기) |
| XE-b | 주 | 격자 안 RMSE: R1(교차검증 λ, xh) − P1 | 같은 풀, 같은 n. WF9-a(x25 에서 기각)의 재시험 |
| XE-c | 보조 | 위치 안 RMSE 의 같은 두 대비, D0(xh) − D0(x25) 격자 안, 총 RMSE R1(xh) − R1(x25)(동등 여부), xh0 의 같은 대비 | 대상별 행 포함, n 200 포함 |
| XE-d | 서술 | 군별 추가 변형, SoilGrids 교체, 지역별 피복 변형의 설명 비율, 격자 사이 성분 | 판정어는 붙이되 확인적 가설로 세지 않음 |
| XE-e | 진단(탐색) | 알래스카·캐나다 VWC 부분 집합에서 P1 격자 안 잔차를 현장 VWC 로 설명하는 비율 | 3.4절, 5.8절 |

### 5.6 판정 규칙

- 통계: 같은 분할·추출·seed 의 짝 대비, 셀 가중과 블록 등가중의 블록 재표집 95 % CI(재표집 10,000회), 4분 판정(우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값 |·| ≤ 0.5 cm, 그 밖 미결정). 층화 평균은 h42.pool_rows(풀 지역 2개 이상, 채점 블록 합집합 8 이상). WF9 와 같은 규칙(`results/rescale_wf3/data/processed/wf/wf3b_meta.json` rules).
- 가설 문구(`_rule3`): 등록 n 모두 우세 = 지지, 열세 없이 일부 우세 = 부분 지지, 열세가 있거나 우세가 없으면 기각, 판정할 수 없는 대비가 있고 우세가 없으면 판정 불가.
- Holm 보정은 XE-a·XE-b 각각의 n 묶음 안에서 보조 열로만 적는다.

### 5.7 해석 규칙(결과 전 고정)

1. XE-a 가 지지 또는 부분 지지이고 XE-b 도 지지 또는 부분 지지: "공개 고해상 입력(10–500 m)을 더하면 물리 기반 ML 이 ERA5 격자 안 변동의 일부(설명 비율 x %)를 맞힌다. WF9 의 격자 안 결론은 입력 해상도의 한계였다." C8 의 격자 안 문장을 고친다.
2. XE-a 가 지지 또는 부분 지지이고 XE-b 가 기각: "고해상 입력은 ML 의 격자 안 오차를 줄였으나 재보정 물리식(격자 안 실측 표준편차)보다 작게 만들지는 못했다."
3. XE-a 가 기각이고 열세가 없음: "취득 가능한 공개 고해상 입력으로도 격자 안 변동을 맞히지 못했다. 남은 변동은 30 m 이하 미지형, 관측 잡음, 연도 차이로 본다." 3.3절의 100 m 안 분산 비율을 상한 근거로 함께 적는다.
4. XE-a 에 열세가 있음: "고해상 입력은 격자 안 오차를 늘렸다."
5. 어느 경우든 총 RMSE 차가 0.5 cm 미만인 대상은 함께 적고, 지도 해상도 문구('1 km 셀로 표시한 기후 격자 단위 보정 지도')는 1 또는 2 일 때만 고친다.
6. XE-e 는 판정을 바꾸지 않는다. 현장 VWC(비 GPR)의 설명 비율이 대상의 0.5 cm 등가 비율(5.8절) 미만이면 "현장 토양 수분도 격자 안 변동의 0.5 cm 이상을 설명하지 못했다", 이상이면 "격자 안 해상도의 토양 수분 지도가 있으면 이득의 여지가 있다(상한 x %)" 를 쓴다.

### 5.8 검정력 근사와 진단 설계

- 0.5 cm 등가 설명 비율(근사): 격자 안 RMSE 가 격자 안 SD 와 같다고 두면, 0.5 cm 감소는 1 − ((SD − 0.5)/SD)² 의 설명 비율에 해당한다. 3.3절의 격자 안 SD 로 계산하면 알래스카 7.9 %(12.42 cm), 레나 5.8 %(16.86 cm), 캐나다 5.0 %(19.67 cm)다. WF9 의 R1 설명 비율(알래스카 0.1 %, 레나 R1(λ 0.25) 5.6 %, 캐나다 −2.2 %)과 비교하면, XE 가 0.5 cm 우세를 보이려면 알래스카에서 설명 비율이 약 8 % 포인트 늘어야 한다.
- XE-e(현장 VWC 상한 진단): 알래스카 2,468셀(31블록), 캐나다 602셀(15블록)에서 P1 의 격자 안 잔차를 (i) 비 GPR 층별 VWC(측정 하단 ≤ 12 cm 와 > 12 cm 를 따로), (ii) GPR·DualEM 층 평균 VWC(민감도)로 설명한다. 학습기는 단조 제약 없는 1차 회귀와 catboost_lo 둘, 평가는 블록 단위 교차검증이다. 사전 기대 부호는 Clayton 2021 을 따른다(층 평균 음, 상부 12 cm 양). 날짜는 같은 위치의 ALT 측정과 같은 계절 캠페인 값을 우선하고, 없으면 위치·기기별 평균을 쓴다. 이 진단은 탐색으로 표시한다.

### 5.9 실행 계획

| 단계 | 내용 | 산출 | 소요[추정] | 디스크[추정] |
|---|---|---|---|---|
| 0 | 등록 문서 작성(이 설계를 `docs/EXPERIMENT_PLAN_XE_*.md` 로 옮겨 열 목록·규칙 확정) | 등록 시각 기록 | 1 h | |
| 1 | H0 결합, T2 계산(디스크 DEM·GSW·Hansen) | `data/processed/xe/xe_feat_v1.csv`(loc_id 키) | 4–5 h | 1 GB 이하 |
| 2 | SoilGrids 250 m 부족 창·2층 취득, NCSCD·CAVM 내려받기 | O군 열 | 3–4 h | 1 GB 이하 |
| 3 | ArcticDEM 10 m 창 읽기(라벨 위치 주변, 레나 지도는 32 m) | M군 열 | 4–6 h | 5 GB 이하 |
| 4 | WorldCover 타일, MODIS MOD13Q1·MOD10A1 AppEEARS 점 요청(고유 화소로 중복 제거), 레나 지도용 MODIS 타일 | V·S군 열 | 6–8 h + 처리 대기(최대 1일 [미확인]) | 5–15 GB |
| 5 | 등록 전 허용 확인(유한값 비율, 격자·위치 안 변동 비율, 열 사이 상관) | QA 표 | 1 h | |
| 6 | 하네스 확장: `--exp xe`, 변형 `xh0`·`xh`·군별, 위치 묶음 저장소 `~l`·`~gl`, 단위 시험(분해 항등식, x25 재현) | `tests/test_h54_workflow.py` 추가 시험 | 4–6 h | |
| 7 | 로컬 스모크(2스레드, nice 10), `--count-only`, Rescale elm 본 실행 | `xe_*` 표 | 실행 1–3 h | |
| 8 | 판정 표 열람, 해석 규칙 적용, C8·지도 문구 갱신 여부 결정 | 판정 기록 | 2 h | |

- 비용 근거: WF 3차 보강(wf9·wf9x·wf10) 129 작업 단위의 단위 실행 시간 합은 3,178초, 적합 14,818건이었다(`results/rescale_wf3/data/processed/wf/wf3b_meta.json`). XE 는 wf9 arm 을 입력 변형 약 9개로 반복하므로 수 CPU-시간 규모이고, elm(3.24 달러/시간, `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 214행)에서 20 달러 상한 안으로 본다[추정]. 실제 값은 `--count-only` 와 스모크 실측으로 다시 적는다.
- 순서: 단계 1 이 끝나면 H0·T2 변형을 먼저 돌릴 수 있다. H 의 열 목록은 단계 0 에서 고정했으므로 먼저 본 결과가 H 구성을 바꾸지 못한다. 단계 2–4 에서 취득에 실패한 군은 5.2절 결측 규칙으로만 뺀다.
- 지도: XE 판정이 해석 규칙 1 또는 2 일 때만 레나 지도 격자(53,011셀, 육지 38,086셀)에 H 를 붙여 지도를 다시 만든다. 판정이 기각이면 지도는 x25 판을 유지한다.

### 5.10 위험과 한계

- 라벨 지원 규모: ABoVE 라벨은 GPR·탐침 점을 소수 4자리 좌표(위도 약 11 m)로 묶어 연도에 걸쳐 평균한 셀이다(`scripts/1_data_prep/parse_above_alt.py`, `aggregate_alt_cell.py`). 10 m 제품과 맞추기에는 좌표 해상도가 거칠다. 3 × 3 창으로 흡수하지만 위치 안 대비(XE-c)의 해석은 이 한계를 함께 적는다.
- 시간 불일치: 라벨 2005–2022, MODIS 2015–2020, WorldCover 2021. 화재 같은 교란 뒤의 식생 변화가 섞일 수 있다.
- 공간 자기상관: 위치 안 셀은 같은 GPR 측선의 인접 점이라 위치 안 변동의 상당 부분이 측정 잡음일 수 있다(GPR 점별 ALT_err 중앙값 8.33 cm).
- 포괄 범위: ArcticDEM 은 60°N 이북만, CAVM 은 툰드라만 분류한다. 내륙 알래스카 산림 셀은 CAVM 비분류다.
- 다중성: H 가 45열이라 n 200 에서 과적합 위험이 있다. R1 은 교차검증 λ 로 잔차를 줄이므로 손해 방지 장치가 있으나, D0 는 그렇지 않다(XE-c 는 보조).
- 이 실험은 전이(지역 홀드아웃)를 시험하지 않는다. 전이에서 고해상 입력의 가치(H19 의 연장)는 별도 질문이며, XE 결과가 지지일 때만 후속으로 검토한다.

---

## 6. 다른 계획 실험과의 관계

- XD_placement_policy: XE 가 지지이면 배치 알고리즘의 공변량 공간에 H 를 넣는 변형을 둔다. 기각이면 x25 공간을 유지한다.
- XC_workflow_end_to_end: XE 결과와 독립으로 x25 로 등록한다. XE 판정은 XC 판정 뒤에 열람하도록 순서를 둘 필요는 없다(서로 다른 대비).
- XG_product_comparison: Whitcomb 2024(30 m) 같은 고해상 ALT 제품은 XG 의 비교 대상이고 XE 의 입력이 아니다.
- XF_new_regions: 새 지역은 x25 로 핵심 대비를 다시 돌린다. 고해상 입력은 북위 60° 이북 지역에서만 같은 구성이 가능하다.

---

## 7. 미해결 항목

1. AppEEARS 점 요청의 점 수·처리 시간 상한 [미확인]. 고유 250 m·500 m 화소로 중복을 없앤 점 수를 먼저 센다.
2. NCSCD 의 티베트 포함 여부, ESA CCI SM 해상도, 유콘강 유역 유기층 두께 지도의 범위·접근 [미확인].
3. Karjalainen 2019(Sci Data)의 ALT R² 0.37 의 원문 문맥, Aalto 2018 의 예측변수 전체 목록, Peng 2018·2023 의 입력 [미확인. 출판사 페이지 403].
4. 3.3절 분해의 1 km·250 m·100 m 셀은 EPSG:3338 원점에 따라 묶음 수가 달라진다(1 km 위치 333곳 대 QA 문서 343곳). 등록 문서에서는 LG 6B.3 셀 색인으로 다시 계산해 적는다.
5. QA 문서 Q8 표의 지형·SAR 추출 서술 정정(3.1절)은 이 문서에 기록만 했고 원문은 고치지 않았다.

---

## 부록 A. 자료 서술 계산(재현용)

모두 `OMP_NUM_THREADS=1` 로 1분 안에 끝나는 판다스 읽기다. 모형 예측은 없다.

```python
# 3.3절 분산 분해
import pandas as pd, numpy as np
from pyproj import Transformer
df = pd.read_csv('data/processed/fidelity_base_v3.csv')
t = Transformer.from_crs(4326, 3338, always_xy=True)
def key(x, y, s):
    return pd.Series([f'{a}_{b}' for a, b in zip(np.floor(x/s).astype(int), np.floor(y/s).astype(int))])
def dec(d):
    d = d.reset_index(drop=True); x, y = t.transform(d.lon.values, d.lat.values)
    d['g9'] = d.block.astype(str) + '|' + d.e5_sqrt_tdd.round(9).astype(str)
    for s, nm in ((1000, 'l1'), (250, 'l250'), (100, 'l100')):
        d[nm] = d['g9'] + '#' + key(x, y, s)
    yv = d.alt_cm.values; tot = ((yv - yv.mean())**2).sum()
    return {lv: ((yv - d.groupby(lv).alt_cm.transform('mean'))**2).sum()/tot
            for lv in ['block', 'g9', 'l1', 'l250', 'l100']}
# 알래스카 = region ∈ {ABoVE_AK, United States (Alaska)}, 레나 = Lena_RU, 캐나다 = {ABoVE_CA, Canada}
```

- 3.1절 위치 안 변동 비율: 위 1 km 키로 묶어 열별 `nunique() > 1` 인 위치에 속한 행 비율(`covariates_ext_v1.csv` 를 `loc_id` 로 결합).
- 3.2절 SoilGrids 포함률: `data/raw/soilgrids_multi/*/soc_0-5cm_mean.tif` 의 경계와 IGH 변환 좌표를 대조해 값 > 0 인 셀 비율.
- 3.4절 VWC 집계: VWC 0–100, 위도 유효 행. 라벨 셀과의 거리는 EPSG:3338 좌표의 최근접 거리(scipy cKDTree).
- ALT_err·연도: ALT 0–300 cm, 경도 −170 ~ −141(알래스카) 행.

## 부록 B. 웹 근거 목록

- Ran et al. 2022: https://essd.copernicus.org/articles/14/865/2022/
- Karjalainen et al. 2019(Sci Data): https://pmc.ncbi.nlm.nih.gov/articles/PMC6413688/
- Karjalainen et al. 2019(TC): https://tc.copernicus.org/articles/13/693/2019/
- Aalto et al. 2018: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018GL078007 (403), 자료 기록 https://zenodo.org/records/5003804
- Gautam et al. 2025: https://www.nature.com/articles/s41598-025-26586-w (로컬 PDF 로 확인)
- Mishra & Riley 2014: https://www.researchgate.net/publication/262449337 (검색 요약)
- Yi et al. 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC7309651/
- Whitcomb et al. 2024: https://iopscience.iop.org/article/10.1088/1748-9326/ad127f , PDF https://plymsea.ac.uk/id/eprint/10090/1/Whitcomb_2024_Environ._Res._Lett._19_014046.pdf
- AirMOSS ALT·토양 수분(ORNL 1657): https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-above-pband-sar-1657-1
- Clayton et al. 2021: https://iopscience.iop.org/article/10.1088/1748-9326/abfa4c
- Gangodagamage et al. 2014: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2013WR014283 , PDF https://www.osti.gov/pages/servlets/purl/1623437
- Hantson et al. 2025: https://iopscience.iop.org/article/10.1088/2752-664X/ad9f6c , PDF https://www.osti.gov/pages/servlets/purl/2538206
- Du et al. 2026: https://tc.copernicus.org/articles/20/4277/2026/
- Thaler et al. 2023: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA003015 (검색 요약)
- Zhang C. et al. 2021: https://www.sciencedirect.com/science/article/pii/S0303243421001628 (검색 요약)
- Li C. et al. 2022: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JD036785 (검색 요약)
- Peng et al. 2018: https://journals.ametsoc.org/view/journals/clim/31/1/jcli-d-16-0721.1.xml (403), Peng et al. 2023: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023EF003573 (403)
- Siewert et al. 2021: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GB006659 (검색 요약)
- Schaefer et al. 2015: https://www.researchgate.net/publication/274699542 (검색 요약)
- ArcticDEM: https://www.pgc.umn.edu/data/arcticdem/
- SoilGrids 2.0: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html
- NCSCD v2: https://apgc.awi.de/dataset/ncscdv2-circumarctic-shapefile (검색 요약)
- MOD10A1 v061: https://nsidc.org/data/mod10a1/versions/61
- MOD13Q1 v061: https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061 (검색 요약)
- AppEEARS: https://appeears.earthdatacloud.nasa.gov/api/ (점 수 상한 [미확인])
- SMAP L4 SPL4SMGP v7: https://nsidc.org/data/spl4smgp/versions/7 (검색 요약)
- SMAP/Sentinel-1 SPL2SMAP_S: https://nsidc.org/data/spl2smap_s/versions/2
- ESA WorldCover 2021: https://esa-worldcover.org/en/data-access (검색 요약)
- ABoVE 토지 피복(ORNL 1691): https://daac.ornl.gov/ABOVE/guides/Annual_Landcover_ABoVE.html (검색 요약)
- CAVM 래스터: https://www.sciencedirect.com/science/article/pii/S0034425719303165 , https://data.mendeley.com/datasets/c4xj5rv6kv/1 (검색 요약)
- 레나 피복: https://doi.pangaea.de/10.1594/PANGAEA.759631 (검색 요약), https://essd.copernicus.org/articles/17/1707/2025/
