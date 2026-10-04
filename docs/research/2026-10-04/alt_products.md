# 기존 ALT 지도 제품 조사: 같은 시험지 비교(XG)와 다중 소스 적층 후보(XB)

- 작성일: 2026-10-04
- 대상 실험: XG_product_comparison, XB_multisource_stacking
- 근거: 이 문서의 모든 수치는 아래에 경로나 URL 을 적은 파일과 웹 페이지에서 가져왔다. 확인하지 못한 항목은 [미확인]으로 적었다. 검색 엔진 요약에서만 본 내용은 [검색 요약]으로 적었다.
- 계산 범위: 이 조사에서 한 계산은 라벨 표와 제품 학습 지점의 거리 계산뿐이다(단일 스레드, 1분 이내). 모형 학습과 제품 래스터 추출은 하지 않았다.

---

## 1. 요약

1. 조사한 제품은 13종이다. 같은 시험지 비교에 바로 쓸 수 있는 제품은 ESA CCI ALT(v4 보유, v5 공개)와 Wei 등 2026(Zenodo v2)이다. 둘 다 약관이 열려 있고, 주 대상 지역을 모두 덮고, 연도별 값이 있다.
2. 누설의 핵심은 CALM 이다. 우리 러시아 W·C·E, 그린란드 라벨은 모두 CALM 지점이므로, CALM 으로 학습한 제품(Wei 2026, Liu 2024, Ran 2022, Aalto 2018, Li 2022)은 이 지역에서 학습 자료로 채점받는 셈이다. 이 지역에서 공정하게 비교할 수 있는 제품은 ALT 라벨로 학습하지 않은 CCI 뿐이다.
3. Liu 2024 는 ABoVE SMALT(ORNL ds1903, 2008-2020)로 학습했다. 우리 알래스카·캐나다 ABoVE 라벨(ds2369 V2)은 이 자료를 확장한 판이므로 Liu 는 알래스카에서 공정 비교 대상이 될 수 없다.
4. Wei 2026 은 학습 지점 좌표 표(보충 자료 `ALT_variables.xlsx`, 2,196행)가 공개되어 정확한 거리 마스크를 만들 수 있다. 블록 단위 5 km 마스크를 쓰면 레나 19/20 블록(3,009셀), 캐나다 20/39 블록(664셀), 알래스카 28/74 블록(4,524셀)이 남는다.
5. 기존 LGX 결과에서 CCI 원값은 라벨 0 에서 원천 계수 Stefan(P0)보다 블록 등가중 RMSE 가 4.99–13.93 cm 컸다(레나, 캐나다, 러시아 E, 알래스카. 러시아 W 는 +0.41 미결정). 선형 보정 CCI 는 레나에서 셀 가중 −0.51 cm 였다. 제품 비교의 CCI 부분은 이미 일부 답이 있다.
6. XG 는 기존 LG·WF6 조각의 블록별 SSE 를 다시 쓰는 블록 마스크 방식을 주 설정으로 권한다. 우리 방법(P0, P1, R1, D0)을 다시 적합하지 않아도 같은 B 블록에서 제품과 짝지은 비교를 할 수 있다.
7. XB 적층의 기본 후보는 누설이 없는 6종(P1, P*, 보정 Kudryavtsev, 토양형 Stefan, 선형 보정 CCI, Stefan·CCI 평균)이다. 확장 후보는 Wei 2026(마스크 통과 대상만)과 Yi·Kimball 2018(알래스카만)이다. Liu 2024, Li 2022, SAR 제품, 접근이 막힌 제품은 적층에서 뺀다.

---

## 2. 우리 라벨의 출처와 누설 판정 기준

### 2.1 라벨 출처(v3 직접 라벨, `data/processed/fidelity_base_v3.csv`)

| 대상 지역 | 셀 수 | 0.5° 블록 | 라벨 원자료 | 근거 |
|---|---|---|---|---|
| 알래스카 | 13,606(ABoVE_AK 13,542, region 'United States (Alaska)' 의 CALM 64) | 74 | ORNL ds2369 V2(ABoVE 토양 융해 깊이, 2005-2024) | `scripts/1_data_prep/parse_above_alt.py`, `data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` |
| 캐나다 | 750(ABoVE_CA 726, CALM 등 24) | 39 | ds2369 V2, CALM | 같음 |
| 레나 | 3,037 | 20 | ALLena(PANGAEA 973813, 1998-2022) | `data/raw/allena/PANGAEA_973813_ALLena_main.txt` |
| 러시아 W·C·E | 31·7·30 | 21·6·21 | CALM(PANGAEA 972777) | `data/raw/calm/SOURCE.md` |
| 새 지역(v4) | 티베트 132, 러시아 C 51, 북대서양 38 등 | 34·8·12 | Du 등 GPR(Zenodo 21999366), FireALT(Talucci 2025) 등 | `data/processed/fidelity_base_v4.csv`, `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B |

셀 수와 블록 수는 v3 표를 `polar.fidelity.MACRO_REGION` 으로 묶어 이번에 다시 셌다. v4 새 셀 수와 블록 수는 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 실행 기록(Tibet 132(34), Russia_C 51(8), NAtlantic 38(12))과 같다.

ds2369 V2 는 ds1903(SMALT)을 확장한 판이다. ds2369 는 2005-01-10 – 2024-08-25, ALT 관측 224,200건이고(https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-above-soil-thawmoisture-2369-1), ds1903 은 2008-06-22 – 2020-08-15, ALT 관측 약 206,000건이다(https://data.nasa.gov/dataset/above-soil-moisture-and-active-layer-thickness-in-alaska-and-nwt-canada-2008-2020-4d334). 우리 알래스카 셀 13,542개 가운데 13,423개, 캐나다 ABoVE 셀 726개 가운데 350개가 year_min 2020 이하다(`data/processed/m1/a2_year_matched_tdd_cells.csv`).

### 2.2 누설 등급

| 등급 | 정의 | 처리 |
|---|---|---|
| L0 독립 | 제품 학습·보정에 ALT 현장 라벨을 쓰지 않았다(문서 기준) | 모든 채점 셀에서 비교한다 |
| L1 지점 공개 | ALT 라벨로 학습했고 학습 지점 좌표가 공개되었다 | 정확한 거리 마스크 뒤 비교한다 |
| L2 출처만 공개 | ALT 라벨로 학습했고 출처 목록만 공개되었다 | 출처 전체 좌표로 만든 대리 마스크 뒤 서술 비교만 한다 |
| L3 같은 원자료 | 우리 라벨과 같은 원자료로 학습했다 | 그 지역에서는 비교하지 않는다. 학습 표본 안 참고값으로만 적는다 |

거리 마스크의 기준은 두 가지다. 셀 단위 마스크는 학습 지점에서 d km 안의 채점 셀을 뺀다. 블록 단위 마스크는 학습 지점에서 d km 안의 셀이 하나라도 있는 0.5° 블록을 통째로 뺀다. d 는 5 km(주)와 25 km(민감도)로 둔다. 5 km 는 1 km 셀 크기와 CALM 격자(1 km × 1 km)의 좌표 오차를 덮는 값이다.

---

## 3. 제품별 조사표

### 3.1 기본 정보

| 제품 | 식별자(URL/DOI) | 약관 | 해상도 | 기간 | 형식·크기 | 접근 |
|---|---|---|---|---|---|---|
| ESA CCI ALT v4 | https://dx.doi.org/10.5285/d34330ce3f604e368c06d76de1987ce5 | ESA CCI Permafrost 약관(원문 [미확인]) | 0.01°(위경도), 북위 25–85° | 1997–2021 연별 | NetCDF 25개, 로컬 828 MB. ALT, ALT_uncertainty 변수 | 보유(`data/raw/cci_alt/`), CEDA 공개 |
| ESA CCI ALT v5 | https://dx.doi.org/10.5285/a6fbedd8ee5b472c8e84e55f746c1704 | 같음 | 1 km | 1997–2023 연별 | NetCDF 28개, 804 MB(CEDA 카탈로그) | CEDA 공개, 계정 없음 |
| Wei 등 2026 | 자료 https://doi.org/10.5281/zenodo.21667583 (v2), 논문 https://doi.org/10.5194/essd-2026-447 | CC BY 4.0 | 1 km | 2000–2024 연별 | ALT zip 3개 2.56 GB, 상대 불확도 zip 3개 2.61 GB | Zenodo 공개 |
| Liu Z. 등 2024 | 자료 https://doi.org/10.5281/zenodo.10070610, 논문 https://doi.org/10.1088/1748-9326/ad0f73 | CC BY 4.0 | 1 km(0.0083°) | 2003–2020 연별 | GeoTIFF 18개, 각 약 103 MB(합 약 1.86 GB) | Zenodo 공개 |
| Ran 등 2022 | 자료 https://doi.org/10.11888/Geocry.tpdc.271190, 논문 https://doi.org/10.5194/essd-14-865-2022 | 논문 표기 CC BY 4.0 | 1 km | 2000–2016 기간 대표값(연별 아님) | GeoTIFF, 크기 [미확인] | TPDC 계정 필요(`docs/NOVELTY_POSITIONING_2026-09-29.md` 234행). 이번 조회에서 TPDC 페이지 내용은 받지 못했다 |
| Aalto 등 2018 | 자료 https://zenodo.org/records/5003804 (DOI 10.5061/dryad.886pr72), 논문 GRL 45, 4889–4898 | CC0 | 30 초각(WGS84) | 기준기 2000–2014 대표값 | `ALT_Rasters.zip` 672,583,929 바이트(기준기 1 + 미래 7) | Zenodo 공개 |
| Li C. 등 2022 | 자료 https://zenodo.org/records/6615133, 논문 https://doi.org/10.1029/2022JD036785 | CC BY 4.0 | 1 km(논문 초록) | 2000–2018(논문), 기록 제목은 2001–2017 | 평균 ALT tif 598,566,806 바이트, 계수 E tif 598,566,808 바이트 | Zenodo 공개. 연별 지도는 없다 |
| Yi·Kimball(ORNL ds1760) | https://doi.org/10.3334/ORNLDAAC/1760 | EOSDIS 자유 이용 | 1 km | 2001–2015 연별 | NetCDF 1개, 35.063 MB | Earthdata 계정 |
| ABoVE L·P 밴드 SAR ALT v3(ds2004) | https://doi.org/10.3334/ORNLDAAC/2004 | EOSDIS 자유 이용 | 30 m | 2017 | 52파일, 6.891 GB | 보유(`data/raw/resalt/`, 6.9 GB). 알래스카 39, 캐나다 북서부 12 사이트 |
| ABoVE 확장 ALT(ds2332) | https://doi.org/10.3334/ORNLDAAC/2332 | EOSDIS 자유 이용 | 30 m | 2014, 2015, 2017 | COG 3개, 6.96 GB | 보유(`data/raw/polsar_alt/upscaled_alt_*.tif`) |
| Pastick 등 2015 | 자료 https://doi.org/10.5066/F7C53HX6, 논문 https://doi.org/10.1016/j.rse.2015.07.019 | USGS 자료 공개 [미확인] | 30 m | 현재 기준 [미확인] | 크기 [미확인] | ScienceBase 가 이 서버의 자동 요청을 막았다. 공개본에 ALT 층이 있는지 [미확인] |
| Peng 등 2024(NCDC) | DOI 10.12072/ncdc.nieer.db4213.2024 [검색 요약] | [미확인] | km 급 | 1850–2100 | NetCDF 77.6 GiB [검색 요약] | NCDC 접속 실패(이번 조회, `data/raw/qtp_alt_open_2026-09-29/SOURCES.json` 기록). DataCite 조회 404 |
| 티베트 지역 제품 2종 | https://zenodo.org/records/18320588 (30 m, GPR 기반), https://zenodo.org/records/18425147 (1 km, 기준기 2006–2020) | CC BY 4.0 | 30 m, 1 km | 위와 같음 | 1.1 MB, 255.6 MB | 둘 다 파일 접근 제한(로그인·승인) |

참고로 조사했으나 표에서 뺀 자료가 있다. SNAP·GIPL2 알래스카 지도(2 km)는 GCM 5종 합성 A1B 강제력의 모의 결과라 관측 기간 재현 지도가 아니다(https://akevt.gi.alaska.edu/GIPL2-GCM-CRREL/, [검색 요약]). Gautam 등 2025(Sci Rep)는 격자 산출을 공개하지 않았다. 논문의 자료 공개 문장은 'Data is provided within the manuscript & supplementary information files' 이다(https://www.nature.com/articles/s41598-025-26586-w). Du 등의 Zenodo 21999366 은 활동층 수분 제품의 코드 묶음이고 ALT 지도가 아니다(`data/raw/qtp_alt_open_2026-09-29/zenodo21999366/Code_Archive_QTB_ALM_v2s.zip` 목록). 우리 티베트 GPR 라벨의 원자료다.

### 3.2 방법, 학습 자료, 보고 정확도, 누설 등급

| 제품 | 방법 | ALT 학습·보정 자료 | 보고 검증 | 우리 라벨 대비 누설 등급 |
|---|---|---|---|---|
| CCI v4·v5 | CryoGrid 열전달 모형. MODIS LST 와 축소 ERA5 강제력(2003년부터), 1997–2002 는 ERA5 편향 보정. 화소마다 앙상블 | ALT 라벨로 학습하지 않는다. 지층은 약 700개 토양 단면의 약 6,500 시료로 만든 지표 피복별 지층 자료다. 단면 위치에 레나 삼각주, Kytalyk, Spasskaya Pad, Cherskii, Herschel Island 가 있다(ATBD v4) | 고위도 ALT: 평균 편향 0.07 m, SD ±0.56 m, RMSE 0.56 m. 중앙아시아·몽골·티베트·알프스는 지층 미모수화로 평가에서 뺐다(CAR v5, 52쪽) | L0. 지층 시료 위치가 일부 대상 지역과 겹치나 ALT 값이 아니다 |
| Wei 2026 | CatBoost·LightGBM 평균. 예측 변수 17종 | CALM·GTN-P 201지점, 연 관측 2,196건(2000–2020) | 지점 하나 제외 교차검증(LOSO) R² 0.761, RMSE 60.48 cm | L1(보충 자료에 좌표). 러시아 W·C·E 는 사실상 L3 |
| Liu 2024 | 랜덤 포레스트 상위 10개 평균(h2o) | 2,966 지점-연: CALM 2,409, ABoVE SMALT 257(Clayton 2021, ds1903), ABCflux 142, 티베트 문헌 158 | 무작위 10겹 교차검증 RMSE 21.60 cm, 편향 4.02 cm, R² 0.97 | 알래스카·캐나다 L3(SMALT), 러시아 L3(CALM), 레나 L2 |
| Ran 2022 | 통계·ML 앙상블(GAM, SVR, RF, XGB) | ALT 452지점. Aalto 2018 의 자료에 티베트·톈산·중국 동북부 149지점을 더했다. 측정의 99 % 가 2000–2016 | 거리 블록 10겹 교차검증 1,000회. ALT RMSE 86.93 ± 19.61 cm, 편향 2.71 ± 16.46 cm | L2(CALM·GTN-P 대리 마스크). 러시아 L3 |
| Aalto 2018 | 통계 모형 앙상블 | ALT 303지점(Ran 2022 의 452 − 149 로 계산). GTN-P 와 추가 자료, 그 가운데 CALM 174 [검색 요약, 원문 미확인] | [미확인](논문 원문 접근 실패) | L2. 러시아 L3 |
| Li C. 2022 | Stefan 식, ERA5-Land 기온 | 초록: 'based on permafrost monitoring data'. 계수 E 의 공간화 방법 [미확인] | [미확인](원문 접근 실패) | L2(CALM 대리 마스크). 구조상 우리 계수 지도류(V1)와 같은 유형 |
| Yi·Kimball | 위성 기반 토양 과정 모형(MODIS LST, SMAP 토양 수분, MERRA-2 적설) | 현장 ALT 는 검증에만 썼다(CALM 51지점). 감도 분석의 공극률·포화도 보정은 Dalton Highway 구간의 레이더 산출과 맞췄다 | PP ≥ 70 % 지역 n = 33, R 0.60, 편향 1.58 cm, RMSE 20.32 cm(Yi 등 2018, TC 12, 145–161) | L0(알래스카만) |
| ds2004 | ReSALT InSAR 와 PolSAR 결합 산출 | 산출식 보정에 쓴 현장 자료 [미확인]. ds2369 현장 자료는 UAVSAR 검증용으로 수집되었다 | [미확인] | L0 로 두되 단서를 붙인다. 이미 공변량 insar_alt 로 쓴다 |
| ds2332 | PolSAR 산출을 랜덤 포레스트로 확장 | 학습은 PolSAR 산출만 썼고 현장 자료는 검증에만 썼다(Whitcomb 등 2024) | 현장 대비 RMSE 11.39–12.07 cm, 편향 −2.71 – −6.51 cm | L0 로 두되 위와 같은 단서. 이미 공변량 polsar_alt 로 쓴다 |
| Pastick 2015 | 의사결정·회귀 나무 | 알래스카 현장 관측 [검색 요약: 탐침 측정 17,000건 이상] | 근지표 영구동토 분류 정확도 약 85 % [검색 요약] | [미확인] |
| Peng 2024 | Peng 2023 초록은 Stefan 해, Wei 2026 은 ML 기반으로 기술 | [미확인] | [미확인] | [미확인] |

근거:
- CCI: CEDA 카탈로그 https://catalogue.ceda.ac.uk/uuid/a6fbedd8ee5b472c8e84e55f746c1704/ , ATBD v4 https://climate.esa.int/media/documents/CCI_PERMA_D2.2_ATBD_v4.0.pdf (4.1.2절과 지층 절), CAR v5 https://climate.esa.int/documents/3155/CCI_PERMA_CAR_v5.0_20251115.pdf (3.9절). 로컬 파일 구조는 `data/raw/cci_alt/ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-2015-fv04.0.nc` 를 열어 확인했다.
- Wei: 원고 PDF https://essd.copernicus.org/preprints/essd-2026-447/essd-2026-447.pdf (2.3–2.5절), Zenodo API https://zenodo.org/api/records/21667583 . v1(https://zenodo.org/records/20602924)은 관측 2,435건으로 적혀 있고 새 판이 있다고 표시된다.
- Liu: 논문 PDF https://eprints.whiterose.ac.uk/id/eprint/206431/1/Liu_2024_Environ._Res._Lett._19_014020.pdf (2.2절, 2.4.1절, 3.1절), Zenodo API https://zenodo.org/api/records/10070610 .
- Ran: 논문 PDF https://essd.copernicus.org/articles/14/865/2022/essd-14-865-2022.pdf (2.1절, 초록).
- Aalto: Zenodo API https://zenodo.org/api/records/5003804 .
- Li: Semantic Scholar 초록 https://api.semanticscholar.org/graph/v1/paper/DOI:10.1029/2022JD036785 , Zenodo API https://zenodo.org/api/records/6615133 .
- Yi·Kimball: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-sat-activelayer-thickness-maps-1760-1 , 논문 https://tc.copernicus.org/articles/12/145/2018/tc-12-145-2018.pdf .
- ds2004, ds2332: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-above-resalt-insar-polsar-v3-2004-3 , https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-alt-maps-ak-ca-2332-1 , https://iopscience.iop.org/article/10.1088/1748-9326/ad127f .
- Peng: Semantic Scholar 초록 https://api.semanticscholar.org/graph/v1/paper/DOI:10.1029/2023EF003573 .

### 3.3 보고 정확도의 해석

제품이 보고한 RMSE(Wei 60.48 cm, Ran 86.93 cm, Liu 21.60 cm, CCI 0.56 m)는 우리 채점 오차와 직접 비교할 수 없다. 검증 지점 구성(심부 티베트·몽골 포함 여부), 분할 방식(무작위, 지점 하나 제외, 거리 블록), 집계 단위가 모두 다르다. Liu 의 21.60 cm 는 무작위 10겹 교차검증 값이라 우리 검증 사다리(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q15)의 무작위 단계에 해당한다. 같은 시험지 비교가 필요한 이유가 여기에 있다.

---

## 4. 누설 정량화(이번 계산)

### 4.1 방법

- 채점 셀: v3 의 source_id = F4_direct 셀(`data/processed/fidelity_base_v3.csv`), v4 새 셀은 loc_id ≥ 20000 이고 source_id = F4_ext_direct 인 셀(`data/processed/fidelity_base_v4.csv`).
- 학습 지점 좌표 집합:
  - Wei: 보충 자료 `data/raw/qtp_alt_open_2026-09-29/essd-2026-447_supplement/ALT_variables.xlsx` 의 (LAT, LONG) 고유값 195개. 행 2,196, Site Name 고유값 196.
  - CALM: `data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab` 자료 행의 (Latitude, Longitude) 고유값 247개.
  - GTN-P ALT: `data/raw/gtnp/sites.json` 의 activelayers 좌표 고유값 225개.
  - Ran·Aalto 대리 집합: CALM ∪ GTN-P. Liu 대리 집합: CALM ∪ GTN-P ∪ 2020년 이전 연도를 가진 ABoVE 셀(`data/processed/m1/a2_year_matched_tdd_cells.csv`, year_min ≤ 2020). ABCflux 와 티베트 문헌 지점은 좌표 표를 확보하지 못해 넣지 않았다.
- 거리: 반지름 6,371 km 의 대원 거리로 각 셀에서 가장 가까운 학습 지점까지 잰다.
- 계산 스크립트는 이번 세션의 임시 폴더에만 두었다. 구현 단계에서 7.7절의 마스크 스크립트로 옮긴다.

### 4.2 학습 지점 근접 셀 수(v3, 셀 단위)

| 대상 | 셀 | 블록 | Wei ≤ 1 km | Wei ≤ 5 km | Wei ≤ 25 km | CALM ≤ 5 km | CALM·GTN-P ≤ 5 km | CALM·GTN-P ≤ 25 km |
|---|---|---|---|---|---|---|---|---|
| 알래스카 | 13,606 | 74 | 1,183 | 4,474 | 9,062 | 4,618 | 4,638 | 9,113 |
| 캐나다 | 750 | 39 | 18 | 19 | 79 | 24 | 24 | 84 |
| 레나 | 3,037 | 20 | 0 | 12 | 30 | 433 | 446 | 2,233 |
| 러시아 W | 31 | 21 | 26 | 26 | 28 | 31 | 31 | 31 |
| 러시아 C | 7 | 6 | 5 | 5 | 5 | 7 | 7 | 7 |
| 러시아 E | 30 | 21 | 22 | 22 | 25 | 30 | 30 | 30 |
| 그린란드 | 3 | 2 | 0 | 0 | 0 | 3 | 3 | 3 |

Wei 의 Site Name 196개 가운데 189개는 1 km 안에 우리 v3 셀이 있다. Wei 학습 지점 거의 전부가 우리 라벨 위치와 겹친다는 뜻이다.

### 4.3 마스크 뒤 남는 채점 단위

| 제품(마스크 집합) | 대상 | 셀 단위 5 km: 셀(블록) | 셀 단위 25 km: 셀(블록) | 블록 단위 5 km: 블록(셀) | 블록 단위 25 km: 블록(셀) |
|---|---|---|---|---|---|
| Wei(정확) | 알래스카 | 9,132(38) | 4,544(25) | 28(4,524) | 23(4,405) |
| Wei(정확) | 캐나다 | 731(21) | 671(21) | 20(664) | 20(664) |
| Wei(정확) | 레나 | 3,025(20) | 3,007(19) | 19(3,009) | 18(2,116) |
| Wei(정확) | 러시아 W·C·E | 5(4)·2(2)·8(7) | 3(3)·2(2)·5(5) | 3·2·6 | 3·2·4 |
| Ran·Aalto(대리) | 알래스카 | 8,968(36) | 4,493(25) | 27(4,523) | 23(4,405) |
| Ran·Aalto(대리) | 캐나다 | 726(16) | 666(16) | 15(659) | 15(659) |
| Ran·Aalto(대리) | 레나 | 2,591(20) | 804(13) | 16(821) | 13(804) |
| Ran·Aalto(대리) | 러시아 W·C·E | 0 | 0 | 0 | 0 |
| Liu(대리) | 알래스카 | 0 | 0 | 0 | 0 |
| Liu(대리) | 캐나다 | 242(7) | 140(4) | 5(173) | 3(133) |
| Liu(대리) | 레나 | 2,591(20) | 804(13) | 16(821) | 13(804) |

레나는 대리 집합의 블록 단위 마스크에서 셀이 3,037개에서 821개로 준다. 사모일로프 섬 주변의 CALM 지점이 셀이 많은 블록 4개 안에 있기 때문이다. Wei 는 이 지점을 학습에 쓰지 않아 레나 블록이 거의 그대로 남는다.

캐나다의 2021년 이후 ABoVE 셀 376개 가운데 200개(4블록)는 2020년 이전 ABoVE 셀에서 25 km 넘게 떨어져 있다. 알래스카의 2021년 이후 셀 119개는 모두 2020년 이전 셀에서 1.5 km 안에 있다. Liu 를 알래스카에서 시간 분리로 구제할 수 없다는 뜻이다.

### 4.4 v4 새 지역(셀 단위)

| 대상 | 새 셀 | 블록 | Wei ≤ 5 km | Wei ≤ 25 km | CALM ≤ 5 km | CALM ≤ 25 km | Wei 까지 중앙 거리(km) |
|---|---|---|---|---|---|---|---|
| 티베트(GPR) | 132 | 34 | 1 | 14 | 5 | 20 | 173.5 |
| 러시아 C | 51 | 8 | 0 | 0 | 0 | 0 | 447.1 |
| 북대서양 | 38 | 12 | 10 | 13 | 9 | 14 | 597.4 |
| 캐나다 확충 | 75 | 38 | 6 | 21 | 9 | 26 | 43.6 |
| 러시아 E 확충 | 25 | 11 | 5 | 5 | 6 | 6 | 30.8 |
| 러시아 W 확충 | 8 | 6 | 0 | 1 | 1 | 1 | 112.0 |

러시아 C 새 셀(FireALT 등)은 Wei·CALM 과 겹치지 않는다. 러시아에서 CALM 학습 제품을 비교할 수 있는 유일한 셀 집합이다. 다만 블록이 8개이고 북대서양 라벨은 약관 미확인분이 섞여 있다(`docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B). 티베트는 Wei 기준으로 거의 독립이다. Liu·Ran 이 쓴 티베트 문헌 자료와 Du 등 GPR 지점의 중복은 좌표 표가 없어 [미확인]이다.

---

## 5. 기존 근거: LGX 의 CCI v4 결과

LGX L29 에서 CCI v4 다년 평균(공변량 cci_alt)을 기준선으로 이미 채점했다. 아래 값은 `data/processed/lgx/lgx_curve.csv` 에서 mode = x(대상 지역 전체 제외), n = 0 행을 읽은 것이다. Δ 는 방법 − P0 이고 괄호는 블록 재표집 95 % CI 다.

| 대상 | P0 RMSE(셀 가중) | CCI 원값 RMSE | Δ 셀 가중 | Δ 블록 등가중 | 선형 보정 CCI Δ 셀 가중 | 선형 보정 CCI Δ 블록 등가중 |
|---|---|---|---|---|---|---|
| 레나 | 21.69 | 22.69 | +1.00 [−0.61, 3.43] | +6.62 [4.35, 8.89] | −0.51 [−0.87, −0.01] | +0.31 [−0.11, 0.75] |
| 캐나다 | 28.16 | 29.20 | +1.04 [−1.37, 8.59] | +13.93 [10.18, 17.93] | −1.14 [−1.78, 0.78] | +0.95 [−0.57, 2.50] |
| 러시아 W | 42.98 | 43.55 | +0.57 [−1.43, 2.77] | +0.41 [−1.65, 2.50] | −1.42 [−2.92, −0.39] | −2.68 [−5.52, −0.40] |
| 러시아 E | 29.72 | 30.18 | +0.45 [−1.08, 6.41] | +4.99 [2.17, 7.76] | −0.32 [−1.43, 1.26] | +0.06 [−1.11, 1.23] |
| 알래스카(x) | 14.54 | 18.81 | +4.27 [2.22, 6.00] | +7.71 [5.38, 10.24] | +0.51 [0.08, 1.67] | +2.65 [1.47, 3.92] |

해석:
- CCI 원값은 셀이 많은 블록에서는 P0 와 비슷하고 셀이 적은 블록에서 크게 틀린다. 셀 가중 Δ 는 작고 블록 등가중 Δ 가 크다. XG 는 두 가중을 모두 보고해야 한다.
- 선형 보정(원천 라벨로 a + b·CCI 적합)은 러시아 W 에서 두 가중 모두 P0 보다 작았다. 이 결과는 L29 판정 문서의 기록과 함께 읽어야 한다(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q3·Q5).
- 오차 하한은 알래스카 11.31, 레나 14.83, 캐나다 17.90 cm 다(`data/processed/lgx/lgx_floor.csv`, scope = eval).

따라서 XG 의 CCI 부분은 새 계산보다 (1) v5 연도 정합판 추가, (2) WF6 지역 내 설정 추가, (3) 다른 제품과 같은 표에 싣는 정리가 주된 일이다.

---

## 6. 공정 비교 가능 판정

| 제품 | 알래스카 | 캐나다 | 레나 | 러시아 W·E | 러시아 C(v4 새 셀) | 티베트(v4) | 판정 |
|---|---|---|---|---|---|---|---|
| CCI v4·v5 | 가능(전 셀) | 가능 | 가능 | 가능 | 가능 | 가능. 단 CCI 스스로 티베트 지층을 모수화하지 않았다고 밝혔다 | **주 비교** |
| Wei 2026 | 가능(마스크 뒤 28블록) | 가능(20블록) | 가능(19블록) | 불가(3–6블록, 학습 지점) | 가능(8블록, 서술) | 가능(서술) | **주 비교** |
| Aalto 2018 | 부분(대리 마스크) | 부분(15블록) | 부분(16블록, 821셀) | 불가 | 서술 | [미확인] | 서술 비교 |
| Ran 2022 | Aalto 와 같음 | 같음 | 같음 | 불가 | 서술 | 티베트 학습 자료 중복 [미확인] | 접근 확보 시 서술 비교 |
| Liu 2024 | 불가(SMALT) | 부분(5블록, 173셀) | 부분(16블록) | 불가 | 서술 | [미확인] | 레나만 서술 비교 |
| Li 2022 | 부분 | 부분 | 부분 | 불가 | 서술 | [미확인] | 서술(계수 지도 유형 참고) |
| Yi·Kimball | 가능(연도 2001–2015 셀 64.7 %) | 해당 없음 | 해당 없음 | 해당 없음 | 해당 없음 | 해당 없음 | 알래스카 보조 비교 |
| ds2004·ds2332 | 덮인 셀만, 단서 부착 | ds2004 일부 | 해당 없음 | 해당 없음 | 해당 없음 | 해당 없음 | 서술(이미 공변량) |
| Pastick, Peng, 티베트 제한 제품 | 접근 미확보 |  |  |  |  |  | 제외 |

알래스카 연도 비율은 `a2_year_matched_tdd_cells.csv` 에서 year_min ≥ 2001 이고 year_max ≤ 2015 인 셀의 비율이다.

---

## 7. XG 설계

### 7.1 질문

1. 새 지역에 라벨이 없을 때, 기존 ALT 지도를 그대로 쓰는 것과 원천 계수 Stefan(P0)은 어느 쪽이 나은가.
2. 대상 라벨 n 개(10, 전량)를 얻었을 때, 우리 워크플로(P1, R1)는 기존 지도보다 나은가. 같은 라벨로 지도를 재보정하면(P1@제품) 결론이 바뀌는가.
3. 라벨이 많은 지역 안(WF6 설정)에서는 어떤가.

질문 1 은 '기존 지도가 있는데 왜 물리 앵커 워크플로가 필요한가'라는 심사 질문(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q9, Q12 의 6)에 대한 직접 답이다.

### 7.2 비교 방법(LGX 키 이름 규칙을 따른다)

| 키 | 정의 | n |
|---|---|---|
| B:{p}_raw | 제품 값 그대로(cm) | 0 |
| B:{p}_aff | 원천 라벨로 a + b·제품을 적합해 대상에 적용(B:cci_aff 와 같은 식) | 0 |
| P1@{p} | 대상 라벨 n 개로 제품의 비례 계수를 κ = 10 수축으로 재보정(P1@cci 와 같은 `coefs` 함수) | 3, 10, 40, 160, 전량 |
| P0, P1, P*, R1(λ 0.25), D0 | 기존 LG·LGX·WF6 키 그대로 | 같은 격자 |

{p} 는 cci4(v4 다년 평균, 기존 cci_alt), cci5y(v5 연도 정합), wei(Wei v2 연도 정합), aalto, liu, li, yk(Yi·Kimball) 다.

### 7.3 채점 셀과 누설 통제

- 분할과 추출: LG 와 같다(`half_split_blocks` split_seed 1–5, `h40.draw_cells`). WF6 는 분할 1–25 와 같은 추출을 쓴다.
- 주 마스크: 블록 단위 5 km. 제품 학습 지점에서 5 km 안의 셀이 있는 B 블록을 빼고, 남은 블록에서 모든 방법을 같은 블록으로 채점한다. L0 제품(CCI, Yi·Kimball)은 마스크가 없다.
- 이 방식을 주로 두는 이유: LG·LGX·WF6 조각은 키별·블록별 SSE 를 저장한다(`results/rescale_lg/data/processed/lg/shards/*_blocksse.npz` 의 u0_keys, u0_sse, u0_blocks). 블록을 빼는 마스크는 P0·P1·R1·D0 를 다시 적합하지 않고 기존 SSE 로 다시 채점할 수 있다. 셀 예측 저장(`*_cells.npz`)은 F1k, F1n, V2, RM 키만 있어 셀 단위 마스크에는 쓸 수 없다(`data/processed/lgx/shards/lgx1__cpu__Lena__x__s1_cells.npz` 확인).
- 민감도: 블록 단위 25 km, 셀 단위 5 km. 셀 단위 판은 R1·D0 를 마스크된 셀에서 다시 채점해야 하므로 Rescale CPU 에서 재실행한다.
- 라벨 쪽 마스크: P1@{p} 와 R1 의 A 추출은 LG 와 같은 추출을 그대로 쓴다(짝지음 유지). A 셀이 제품 학습 지점이어도 B 채점 셀로 정보가 새지는 않는다. 다만 XB 의 가중치 학습에서는 A 쪽도 마스크한다(8.3절).
- 학습 표본 안 참고값: 러시아 W·E 에서 CALM 학습 제품의 값은 '학습 지점 포함'을 붙여 별표로만 싣는다. 판정에 쓰지 않는다.

### 7.4 시간 정합

- 주 설정: 셀의 라벨 연도(year_min–year_max)와 제품 기간이 겹치면 겹치는 연도의 평균을 쓴다. 겹치지 않으면 제품 기간 전체 평균을 쓰고 match_flag 를 남긴다.
- 민감도: 제품 기간 평균(정적)만 쓴다. 기존 cci_alt(1997–2021 평균)와 같은 방식이다.
- 지역별 연도 덮임(라벨 연도가 제품 기간 안에 들어가는 셀 비율, 이번 계산): 캐나다는 CCI v4 0.288, CCI v5 0.877, Wei 0.977, Liu 0.280 이다. 캐나다 라벨의 중앙 연도가 2022 이기 때문이다. 알래스카는 CCI v4 0.973, Wei 0.998, Liu 0.959, 레나는 CCI v4 0.998, Wei 0.987, Liu 0.870 이다. 캐나다에서 연도 정합 비교를 하려면 CCI 는 v5 가 필요하다.

### 7.5 지표와 통계

- 지표: RMSE(셀 가중, 블록 등가중), 평균 편향, MAE, 오차 하한 대비 환원 가능 오차 비율(알래스카, 레나, 캐나다).
- 대비: Δ = 우리 방법 − 제품. CI 는 `h4_common.boot_delta_blocks`(분할 안 B 블록 재표집, 10,000회)로 낸다. 4분 판정(우세, 열세, 동등, 미결정)과 동등 한계 0.5 cm 는 LGX 와 같다. 두 가중의 CI 가 모두 0 을 제외할 때만 우세·열세로 쓴다.
- 풀: 레나·캐나다 층화 평균(Wei, CCI 공통). CCI 는 주 4지역 층화 평균도 낸다. 알래스카(x)와 WF6 알래스카는 별도 행이다(LG 의 풀 규칙: 알래스카와 하위 지역은 한 풀에 넣지 않는다).

### 7.6 등록 대비

제품 값을 채점 셀에서 추출하기 전에 아래 대비를 계획서에 등록한다. 확인적 대비는 6개이고 Holm 보정을 붙인다.

| id | 대비 | 제품 | 풀 | 성격 |
|---|---|---|---|---|
| XG-1w | B:wei_raw − P0(n 0) | Wei | 레나·캐나다, 블록 마스크 5 km | 확인적 |
| XG-2w | R1 − B:wei_raw(n 10) | Wei | 같음 | 확인적 |
| XG-3w | R1 − B:wei_raw(전량) | Wei | 같음 | 확인적 |
| XG-1c | B:cci5y_raw − P0(n 0) | CCI v5 | 주 4지역 | 확인적 |
| XG-2c | R1 − B:cci5y_raw(n 10) | CCI v5 | 주 4지역 | 확인적 |
| XG-3c | R1 − B:cci5y_raw(전량) | CCI v5 | 주 4지역 | 확인적 |
| XG-4 | P1@{p} − P1(n 10, 전량) | 전 제품 | 제품별 마스크 | 서술 |
| XG-5 | WF6 지역 내: R1·P1·D0 − B:{p}_raw(n 200, 500, 1,000, 전량) | CCI v5, Wei, Yi·Kimball(알래스카) | 알래스카, 레나, 캐나다 | 서술 |
| XG-6 | 알래스카(x), 러시아 C·티베트(v4) 행 | 전 제품 | 대상별 | 서술 |
| XG-7 | 학습 표본 안 참고값(러시아 W·E 의 CALM 학습 제품) | Wei, Liu, Aalto | 대상별 | 별표, 판정 없음 |

방향은 미리 가정하지 않는다. CCI 원값이 블록 등가중에서 P0 보다 4.99–13.93 cm 나빴던 기존 결과(5절)는 비맹검 정보이므로 계획서의 '비맹검 항목'에 적는다.

### 7.7 실행 단계, 비용, 산출물

| 단계 | 내용 | 실행 위치 | 비용(추정) |
|---|---|---|---|
| 1 | 내려받기: CCI v5(804 MB), Wei ALT zip 3개(2.56 GB, 불확도 zip 은 선택 2.61 GB), Aalto(0.67 GB), Liu(1.86 GB), Li 평균·E(1.2 GB), ds1760(35 MB) | 로컬 네트워크 | 디스크 약 7.1 GB(불확도 포함 9.7 GB) |
| 2 | 추출: 새 스크립트 `scripts/1_data_prep/xg_extract_products_v1.py`. v4 전 셀 좌표에서 최근접 화소(주)와 3×3 평균(민감도), 연도별 값. 단위·좌표계 확인 기록 | 로컬, nice 10, 4 스레드 이하, 창 읽기(전체 래스터를 메모리에 올리지 않는다) | 수십 분 [추정] |
| 3 | 마스크: 새 스크립트 `scripts/2_evaluation/xg_leak_mask_v1.py`. 4.1절의 집합으로 셀·블록 마스크 표 생성 | 로컬 | 1분 이내 |
| 4 | 채점: 새 스크립트 `scripts/2_evaluation/h55_xg_products.py`. h40·h42 의 함수를 불러 제품 키를 해석식으로 계산하고, 기존 조각의 블록 SSE 와 짝지어 블록 마스크로 다시 채점 | Rescale CPU(elm) 또는 로컬 경량. 학습 적합 없음 | 1 h 이내 [추정] |
| 5 | 셀 단위 민감도: R1·D0 재적합 | Rescale CPU | 1–2 h, 1–3 달러 [추정, `docs/QA_FINAL_REVIEW_2026-10-02.md` Q11 의 X1 비용 기준] |

산출물: `data/processed/xg/xg_product_values_v1.csv`(loc_id, product, year_rule, value_cm, valid, pix_dist_m), `xg_leak_mask_v1.csv`, `xg_curve.csv`, `xg_tests.csv`, `xg_meta.json`(입력 sha256, Zenodo md5 대조 결과, 제품 판 번호). 플랫폼 규칙에 따라 제품 키와 P·R 키는 모두 CPU 결정적 계산이다.

### 7.8 한계

- 제품의 ALT 정의가 다르다. CCI 는 모형 화소의 연 최대 융해 깊이, Wei 는 탐침·지온·융해관이 섞인 CALM 라벨, Liu 는 GPR 비중이 큰 SMALT 를 따른다. 우리 라벨도 GPR 과 탐침이 섞여 있다.
- Wei 의 LOSO 는 지점 하나만 뺀다. 이웃 지점이 남으므로 블록 분리 채점보다 낙관적일 수 있다. 우리 비교는 제품을 '사용자가 받는 그대로' 채점하므로 이 점은 결과 해석에만 영향을 준다.
- 라벨 0 대비에서 P0 는 다른 지역 라벨로 계수를 정하고, 제품은 CALM 전 지역 라벨로 학습했다. 두 쪽이 쓴 정보량이 다르다. 이 차이는 문장으로 밝힌다.
- 블록 마스크는 Wei 의 알래스카 블록을 74개에서 28개로 줄인다. 남는 블록은 CALM 에서 먼 내륙·남부 쪽으로 치우칠 수 있다. 마스크 전후의 P0 RMSE 를 함께 보고해 표본 구성 변화를 보인다.

---

## 8. XB 적층 후보 권고

### 8.1 기본 후보(모든 대상, 누설 없음)

| 후보 | 근거 열 또는 키 | 비고 |
|---|---|---|
| P1(재보정 Stefan) | P1 | 기준 |
| P*(연도 정합 도일 Stefan) | P1@tddm(`lgx_tdd_matched_v1.csv`) | 라벨 0 에서 유일하게 유의한 개선(−1.00) |
| 보정 Kudryavtsev | P1@ku(p4_ku) | 결측 대체 규칙 주의(LG 계획서의 결측 대체 통계 예외) |
| 토양형 Stefan | P1@ed(p2_edaphic) | 같음 |
| 선형 보정 CCI v4 | B:cci_aff, P1@cci | 기존 cci_alt 그대로. 이전 실험과 연속성 |
| Stefan·CCI 평균 | WF7 앵커 | 기존 |
| CCI v5 연도 정합 | 새 열 cci5y | v4 를 대체하지 않고 민감도 판으로 둔다 |

### 8.2 확장 후보(마스크 통과 대상만)

| 후보 | 쓰는 대상 | 쓰지 않는 대상 | 이유 |
|---|---|---|---|
| Wei 2026 v2 | 레나, 캐나다, 알래스카(마스크 뒤), 러시아 C·티베트(v4) | 러시아 W·E, 그린란드 | 정확한 학습 지점 표가 있다. 주 대상 대부분을 덮는다. 연도별 값이 있다 |
| Yi·Kimball ds1760 | 알래스카(WF6, 알래스카 x) | 그 밖 | L0 이고 CCI 와 다른 모형 계열이다. 2001–2015 만 있다 |
| Aalto 2018 | 민감도 판에만(레나·캐나다·알래스카 마스크 뒤) | 러시아 | CC0 라 쓰기 쉽지만 L2 대리 마스크라 레나 셀이 크게 준다 |

### 8.3 적층의 누설 규칙

- 제품이 적층 후보로 들어가려면 그 대상의 A 추출 풀과 B 채점 셀이 모두 마스크를 통과해야 한다. A 셀이 제품 학습 지점이면 제품의 A 잔차가 작게 나와 가중치가 부풀고, 그 부풀림은 B 에서 손해로 나타난다.
- 통과하지 못한 대상에서는 그 후보를 빼고 적층한다(대체값을 넣지 않는다). 빠진 후보를 대상별로 기록한다.
- 결측 화소(예: 레나 CCI 유효 2,958/3,037, `fidelity_base_v3.csv` 의 cci_valid)는 기존 규칙(셀 단위 대체 없음, 결측 셀 기록)을 따른다.

### 8.4 적층에서 뺄 제품

| 제품 | 이유 |
|---|---|
| Liu 2024 | 알래스카·캐나다 학습 자료가 우리 라벨 원자료(SMALT)와 같다. 레나에서만 쓸 수 있어 대상 간 비교가 어렵다 |
| Li 2022 | CALM 으로 정한 계수 지도를 쓴 Stefan 이다. 우리 시험에서 계수 지도 유형은 새 지역에서 고정 계수보다 나빴다(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q4). 연별 지도가 없고 평균 ALT 파일의 값 의미가 [미확인]이다 |
| ds2004·ds2332 | 알래스카 일부만 덮는다. SAR 입력은 이미 시험해 이득이 없었다(WF1-b, `docs/QA_FINAL_REVIEW_2026-10-02.md` Q3) |
| Ran 2022 | 접근 미확보. 확보하면 Aalto 2018 과 같은 등급으로 민감도 판에 넣을 수 있다 |
| Pastick, Peng, 티베트 제한 제품 | 접근 미확보 |

---

## 9. 내려받기 주소(구현 단계)

| 제품 | 주소 | 검증값 |
|---|---|---|
| CCI v5 | https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/active_layer_thickness/L4/area4/pp/v05.0/northern_hemisphere/{YYYY}/ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-{YYYY}-fv05.0.nc (YYYY = 1997–2023. 1997–2002 의 파일 이름 접두는 디렉터리에서 확인한다) | 2015 파일 31,374,886 바이트(디렉터리 목록) |
| CCI v4(보유분 대조용) | https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/active_layer_thickness/L4/area4/pp/v04.0/ | 1997 파일 33,742,151 바이트로 로컬과 같다 |
| Wei v2 ALT | https://zenodo.org/api/records/21667583/files/ALT_2000_2009.zip/content , .../ALT_2010_2019.zip/content , .../ALT_2020_2024.zip/content | md5 74d778dd80abfaa552013c60416e8358, c7e673a22225c58c693aa49f7a46da39, a2ac7a831765519de819826d50066fd2 |
| Wei v2 불확도(선택) | https://zenodo.org/api/records/21667583/files/Relative_uncertainty_2000_2009.zip/content , .../Relative_uncerrtainty_2010_2019.zip/content(원본 철자 그대로) , .../Relative_uncertainty_2020_2024.zip/content | md5 e3018edc76e45313b5c65717ba1fb0ef, 38ec0aa19f80b814f0da62905754ed4c, 98b3e758682983ec35f68d0d19047fe0 |
| Wei 학습 표 | 보유(`data/raw/qtp_alt_open_2026-09-29/essd-2026-447_supplement/ALT_variables.xlsx`, DOI 10.5194/essd-2026-447-supplement) |  |
| Liu 2024 | https://zenodo.org/api/records/10070610/files/ALT_{YYYY}.tif/content (YYYY = 2003–2020) | 예: ALT_2015.tif md5 9095fc69e6cf212a90cfca5701b52fc7 |
| Aalto 2018 | https://zenodo.org/api/records/5003804/files/ALT_Rasters.zip/content | md5 47752e4d140dda92d9ddbf806adc4508 |
| Li 2022 | https://zenodo.org/api/records/6615133/files/Spatial_distribution_of_average_ALT_in_permafrost_in_Northern_Hemisphere.tif/content , .../Spatial_distribution_of_average_soil_factor_E_in_permafrost_in_northern_Hemisphere.tif/content | md5 9dbd462a96f746425c4ca32974446429, 6a87cabb6a808ac3188401dfc14d469c |
| Yi·Kimball | https://doi.org/10.3334/ORNLDAAC/1760 (Earthdata 로그인, `.netrc` 또는 earthaccess) |  |
| Ran 2022 | https://data.tpdc.ac.cn/en/data/5093d9ff-a5fc-4f10-a53f-c01e7b781368/ (TPDC 계정) |  |
| ds2332 사용자 안내서 | https://daac.ornl.gov/ABOVE/guides/ALT_Maps_AK_CA.html |  |

Zenodo md5 는 https://zenodo.org/api/records/{id} 응답의 checksum 값이다. 내려받은 뒤 같은 값을 `xg_meta.json` 에 대조 결과로 남긴다.

---

## 10. 부수 발견

ds2332(우리 `data/raw/polsar_alt/upscaled_alt_*.tif`)는 화소 값 1.0 을 개방 수면, 2.0 을 산림으로 부호화한다고 검색 결과의 ORNL 안내서 요약에 적혀 있다 [검색 요약, 안내서 원문 미확인]. 단위는 m 다. 기존 점검에서 'polsar_alt 99.5 이상 1,247셀은 분류 부호 혼입 의심'으로 적었던 문제(`docs/EXPERIMENT_PLAN_LGU_2026-09-29.md`, `docs/GENERATIVE_MODEL_REVIEW_2026-09-29.md`)와 맞는다. 사용자 안내서로 부호를 확인하면 과거 PolSAR 실험의 재확인 범위를 정할 수 있다. XG 에서 ds2332 를 서술 비교할 때는 1.0·2.0 화소를 결측으로 둔다.

---

## 11. 미확인 사항

1. Ran 2022 의 TPDC 페이지 내용(파일 목록, 크기, 계정·신청 절차)을 받지 못했다.
2. Wei 원고는 201지점이라 적었으나 보충 표의 Site Name 은 196개, 좌표 고유값은 195개다. 차이의 원인을 확인하지 못했다.
3. Wei, Liu, Aalto, Li 래스터의 단위(cm 또는 m), 좌표계, 결측 값을 아직 확인하지 않았다. 추출 단계의 첫 점검 항목이다.
4. Li 2022 의 계수 E 공간화 방법과 '평균 ALT' 파일의 값 의미를 확인하지 못했다. Zenodo 의 설명 문서는 −2–2 의 추세 등급만 설명한다.
5. Aalto 2018 의 ALT 지점 수(303, 그 가운데 CALM 174)는 Ran 2022 본문과 검색 요약으로만 맞췄다. 원문은 출판사 차단으로 읽지 못했다.
6. CCI 약관 원문(https://artefacts.ceda.ac.uk/licences/specific_licences/esacci_permafrost_terms_and_conditions.pdf)을 읽지 않았다. 지층 외의 모형 조정에 ALT 관측을 썼는지도 ATBD v4 수준에서만 확인했다.
7. Liu 마스크에 넣을 ABCflux 지점 좌표와 티베트 문헌 지점 좌표를 확보하지 못했다.
8. Peng 2024 자료의 방법(Stefan 해 또는 ML)이 문헌마다 다르게 기술된다. 접근도 막혀 있다.
9. Pastick 2015 공개본에 ALT 층이 있는지 확인하지 못했다.
10. Wei 2026 은 심사 중이다. 판이 바뀌면 학습 표와 지도가 바뀔 수 있으므로 v2 와 md5 를 고정해 쓴다.
