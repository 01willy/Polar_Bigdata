# 독립 지역 확충용 공개 현장 ALT 자료 조사 (2026-10-04)

작성 범위: 사용자 질문 1(공개 자료로 독립 지역을 더할 수 있는가, 알래스카에서 개발하고 보유한 다른 지역에서 시험하는 방향은 어떤가)에 대한 자료 측면의 답이다. 실험 ID 는 XF_new_regions, XD_placement_policy, XJ_tempderived_aux_labels 와 연결된다. 모델 학습과 대용량 내려받기는 하지 않았다. 웹 메타데이터와 소형 파일(최대 2.5 MB 1건, 나머지 140 KB 이하)만 scratchpad 에 받아 지점·날짜를 확인했고 `data/raw/` 에는 아무것도 두지 않았다.

## 1. 요약

1. LGD 적격 규칙(직접 측정·계절 말 라벨, 1990년 이후, 기존 대상과 100 km 이상, 채점 블록 합집합 8개 이상)을 그대로 적용하면, 1–2일 안에 확보 가능한 공개 자료로 **새 독립 macro 지역을 기준 충족 상태로 더할 수 있는 경우는 찾지 못했다(0개)**.
2. 이유는 세 가지다. (가) 공개 ALT 자료의 대부분이 이미 보유한 CALM, ABoVE ds2369, ALLena, FireALT, LGD 자료원과 겹친다. (나) 보유 지역 밖의 후보(몽골, 알프스, 천산, 노르웨이 산지, 스발바르 시추공)는 지온 유도 라벨이다. (다) 남반구(남극, 안데스)는 채점 마스크가 요구하는 CCI ALT 의 범위(25–85°N) 밖이다.
3. 조건부로 가장 가까운 것은 **NAtlantic 약관 확인분 판의 적격화**다. 지금 약관 확인분 판은 22셀, 블록 8개, 채점 블록 합집합 6 < 8 이라 점 추정에 머문다. CC BY 4.0 인 Stordalen(Zenodo 10420396)과 Adventdalen 2023(Zenodo 11187360)이 새 채점 블록을 1–2개 더할 가능성이 있다. 확정은 셀 조립과 `--count-only` 뒤에만 할 수 있다.
4. 하위 지역 과제는 1–2개 늘릴 수 있다. 가장 큰 후보는 Bonanza Creek LTER 의 내륙 알래스카 흑가문비 150지점 최대 ALT(2000–2002, 9–10월)다. 알래스카 동부 내륙과 도로망을 따라 새 셀을 줄 수 있으나 약관과 내려받기 경로가 [미확인]이다.
5. 캐나다 동부·고위도 북극(CA-1)은 이미 CALM 행으로 v3 에 들어 있다(6셀). Parks Canada 공개 격자의 대부분이 이 CALM 지점과 같고, 새로 더할 수 있는 것은 3–6셀 수준이다.
6. 학습형 배치 정책은 '알래스카 하위 지역에서 개발하고 알래스카 밖 과제에서 시험'하는 설계가 자료 측면에서 가능하다. 지금 보유 자료로 시험 과제는 LE-1, LE-2, CA-2, CA-3, Tibet 의 5개이고 Russia_C 가 경계다. 과제 5개면 단측 부호 검정에서 '5개 모두 같은 방향'만 p < 0.05(0.5^5 = 0.031)다. 이 기준을 결과 열람 전에 등록할 것을 권한다.

## 2. 조사 방법

| 항목 | 내용 |
|---|---|
| 보유 자료 대조 기준 | `data/processed/fidelity_base_v4_meta.json` 의 by_source(18개 자료원), `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B.2(선정·보류·제외 목록), `data/processed/ext_labels/_sources_plan.json` 의 deferred(14건), `docs/LGD_LICENSE_CONTACTS_2026-09-30.md`, `data/raw/` 목록(41개 폴더) |
| ds2369 내부 구성 | `data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` 의 team_name 별 범위를 직접 집계했다(유효 ALT 224,199행) |
| 연합 검색 | DataONE 연합 색인(ADC, LTER, ESS-DIVE, PANGAEA, NSIDC, PDC, USGS 등) 668건, PANGAEA 검색 API 166건, NASA CMR(ORNL DAAC, NSIDC), 캐나다 정부 공개자료 CKAN, Zenodo·figshare API, 웹 검색 |
| 거리 계산 | 후보 좌표와 v4 의 F4 계열 셀 사이 대권 거리 최솟값(km) |
| 판정 규칙 | LGD 6B.3(라벨 정의, 계절 말 근거, 1990년 이후, 교란 제외), 6B.4(100 km 독립성, 채점 블록 8개 이상). 6B.3 의 `record_date` 는 8월 또는 9월 단일 방문이다. 7월 단일 방문은 `direct_dated` 로 대상 셀에 넣지 않는다 |
| 채점 범위 | CCI ALT 파일 속성 geospatial_lat_min 25.0, geospatial_lat_max 85.0(`data/raw/cci_alt/ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-2021-fv04.0.nc`). 남반구 지점은 eval_mask 에서 빠진다 |

## 3. 보유 자료의 과제 구조

### 3.1 ds2369 에 이미 들어 있는 알래스카·캐나다 조사

| team_name | 유효 ALT 행 | 범위(위도, 경도) | 연도 |
|---|---|---|---|
| Schaefer | 200,380 | 61.2–71.3°N, -163.3 ~ -141.1° | 2013–2018 |
| NGEE | 6,555 | Seward 반도(65°N, -164 ~ -166°), Utqiagvik(71°N) | 2013–2022 |
| Douglas | 3,934 | 64.87–64.95°N, -147.74 ~ -147.61° | 2005–2022 |
| KTurner(Brock 대학) | 2,550 | 67.84–67.93°N, -139.76 ~ -139.35°(올드크로) | 2017–2022 |
| AlaskaTundraFire | 2,473 | 65.0–69.7°N, -164.8 ~ -148.6°(노아탁, 수어드 반도) | 2016–2018 |
| Natali | 1,961 | 61.3–68.6°N | 2013 |
| Bourgeau-Chavez | 1,186 | 60.5–62.6°N, -117.4 ~ -113.0° | 2008–2022 |
| ABoVE-Iwahana, Zhang, Rocha, Scotty Creek, Eight Mile Lake, Trail Valley Creek | 215–1,168 | 북부 사면, 내륙 알래스카, 캐나다 북서부 | 2013–2024 |

ESS-DIVE 의 NGEE Seward·Utqiagvik 자료, Old Crow Flats(Nordicana D115, 2017–2022), Trail Valley Creek, Scotty Creek 는 같은 조사지가 이미 들어 있어 새 지역이 되지 않는다.

### 3.2 지금 쓸 수 있는 과제

| 과제 | 평가 셀 | 블록 | 채점 블록 최소 | 근거 |
|---|---|---|---|---|
| AL-1 | 5,154 | 10 | 1 | `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md` 2절 |
| AL-2 | 3,491 | 20 | 10 | 같은 곳 |
| AL-3 | 651 | 16 | 4 | 같은 곳 |
| AL-4 | 471 | 4 | 1 | 같은 곳 |
| AL-5 | 619 | 17 | 9 | 같은 곳 |
| AL-6 | 3,220 | 7 | 1 | 같은 곳 |
| CA-1 | 6 | 6 | 해당 없음(LG 제외) | 같은 곳, `data/processed/lg_subregion_map_v1.csv` |
| CA-2 | 395 | 14 | 2 | 같은 곳 |
| CA-3 | 349 | 19 | 3 | 같은 곳 |
| LE-1 | 1,786 | 14 | 3 | 같은 곳 |
| LE-2 | 1,251 | 6 | 4 | 같은 곳 |
| Tibet_LGD | 132 | 34(합집합 34) | 14 | `data/processed/lgd_eligibility_v1.csv` |
| Russia_C(확충) | 57 | 13(합집합 13) | 6 | 같은 곳 |
| Russia_W, Russia_E(v3) | 28, 27 | 18, 19 | 7, 8 | 커버리지 문서 2절 |
| NAtlantic(전체 판) | 34(대상 41) | 13(채점 10, 합집합 10) | 3 | `lgd_eligibility_v1.csv` |
| NAtlantic(약관 확인분 판) | 22 | 8(합집합 6) | 2 | `data/processed/lgd/lgd_eligibility_lic.csv`, 부적격 |

CA-1 의 6셀은 v3 CALM_Canada 행 가운데 66.88°N(-64.70°), 72.81°N(-79.33°), 73.01°N(-80.69°), 78.88°N, 80.02°N, 81.40°N(-76.71°) 행으로 본다. 여섯 위도의 평균 75.5°N 이 `lg_subregion_map_v1.csv` 의 CA-1 평균과 같다. 같은 지역 묶음 밖의 CALM_Canada 행으로 65.88°N(-89.37°)와 64.20°N(-95.50°)이 있다(`fidelity_base_v4.csv` 직접 조회).

## 4. 후보 자료 평가

순위 기준: (a) 독립 지역에 채점 블록 8개 이상 또는 1 km 셀 30개 이상을 더하는가, (b) 약관의 명확성, (c) 작업량. '관계' 열의 '새 macro' 는 기존 대상 셀에서 100 km 밖, '하위' 는 기존 macro 안의 새 하위 지역 또는 셀 추가다.

### 4.1 NAtlantic 약관 확인분 판의 적격화(우선순위 1)

현재 약관 확인분 판의 블록 8개는 Russell Glacier(67.14°N, -50.09°), Tavvavuoma(68.46°N, 20.90°), 디스코(69.26°N, -53.48°), 일루리사트(69.23°N, -51.07°, 13셀), Kevo(69.82°N, 27.17°), Zackenberg(74.47°N, -20.56°), Adventdalen UNISCALM(78.20°N, 15.84°), Bayelva(78.92°N, 11.86°)다(`data/processed/lgd/run_tables/NAtlantic_lic/fidelity_base_v3.csv` 직접 조회). 빠진 것은 calm_web_subsites 13셀(Abisko·Kapp Linné, 블록 4)과 cusp_v1_1 6셀이다(`lgd_eligibility_lic.csv` 의 lic_dropped_sources).

| 순위 | 자료 | DOI/URL | 약관 | 방법·시기 | 규모 | 위치·관계 | 내려받기 | 판정 |
|---|---|---|---|---|---|---|---|---|
| 1 | Crill & McCalley, Stordalen 자동 챔버 지점 수동 ALT·지하수위 2003–2017 | https://doi.org/10.5281/zenodo.10420396 | CC BY 4.0 | 금속봉, 시즌 중 반복 측정(`series_max` 적용 가능) | 3지점(팔사, Sphagnum, Eriophorum), xlsx 194,902 B | 68°21′N, 19°03′E 부근(ICOS SE-Sto 소개, https://www.icos-sweden.se/abisko-stordalen, 검색 요약 기준. 챔버 좌표 [미확인]). 0.5° 블록 경도 19.0–19.5 로 기존 Tavvavuoma 블록(20.5–21.0)과 다르다. 하위 | Zenodo 직접 | 새 채점 블록 1개 후보. Sphagnum·Eriophorum 지점은 영구동토가 없거나 탐침 한계일 수 있어 교란·검열 규칙 확인 필요 |
| 2 | Wendt 2024, Adventdalen 12개 시추 지점 지중 얼음과 2023년 ALT | https://doi.org/10.5281/zenodo.11187360 (개념 DOI 10.5281/zenodo.11187359) | CC BY 4.0 | 탐침, 2023년 9월 초(TC 20, 1179, 2026 본문) | 12지점, 지점당 최대 9회 측정 | 78.2°N, 15.8°E 부근. UNISCALM 블록과 같거나 동쪽 인접 블록 [미확인] | Zenodo 직접 | 블록 0–1개. 단년(2023, 이례적 고온 해) 값이라 연도 편차 표시 필요 |
| 3 | Petrone 등 2016, 캉에를루수아크 Two Boat Lake GPR·탐침 | https://doi.org/10.1594/PANGAEA.845258 | CC BY-NC-SA 3.0 | 탐침 측선(Probe_transects.xlsx 14,247 B), GPR(GPR_profiles.xlsx 90,252 B) | 1 집수역 | Russell Glacier 와 같은 블록으로 추정. 하위 | PANGAEA 직접(zip 약 2.5 MB) | 새 블록 없음. CUSP 의 Petrone 행을 원출처로 바꾸는 약관 경로다. 파생 표 재배포는 NC-SA 조건을 따라야 한다 |
| 4 | Rönkkö 2003, Vaisjeäggi 팔사 ALT(NSIDC GGD622) | https://doi.org/10.7265/dave-b458 | 인용 요구(재배포 조항 표기 없음 [미확인]) | 탐침, 1993-09-08–2002-10-14, 76회 | 1지점 | 69.82°N, 27.17°E. 기존 Kevo 셀과 같은 위치 | NSIDC FTP | 새 블록 없음. 연도 보강만 |

### 4.2 알래스카 하위 지역 과제(우선순위 2)

| 순위 | 자료 | DOI/URL | 약관 | 방법·시기 | 규모 | 범위·관계 | 내려받기 | 판정 |
|---|---|---|---|---|---|---|---|---|
| 1 | Bonanza Creek LTER, 흑가문비 150지점 최대 ALT 2000–2002 (knb-lter-bnz.140) | https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=140 | [미확인] (EDI PASTA 가 2026-10-04 공개 읽기를 거부했다) | 'Maximum active layer depths', 기간 2000-09-01–2002-10-30(DataONE 색인) | 150지점 | 63.31–67.34°N, -150.47 ~ -142.08°. Taylor, Alaska, Parks, Elliott, Steese, Dalton 도로 | EDI 포털(브라우저 확인 필요) | 셀 30개 이상의 새 내륙 하위 과제 후보. 기존 AL-5 와 겹치는 블록 수 [미확인] |
| 2 | Bonanza Creek LTER, 지역 지점망 ALT 또는 영구동토 유무 2000–2025 (knb-lter-bnz.605) | https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=605 | [미확인] | 2.5 m 안의 영구동토 유무와 깊이, 10월 측정 포함 | [미확인] | 63.70–66.27°N, -150.40 ~ -144.33° | EDI 포털 | 1번과 합칠 때 보조. 유무형 값은 검열 처리 필요 |
| 3 | ViPER, 알래스카 시추공 지점 융해 깊이·유기층 2015, 2017, 2018 | https://doi.org/10.18739/A22J6848J | CC BY 4.0 | 2015년 7–8월, 2017년 9월, 2018년 8–9월 | csv 8개, 각 0.9–22.8 KB | 62.17–70.37°N, -150.62 ~ -145.47°. 시추공 지점이 기존 CALM·ds2369 셀과 겹칠 가능성 [미확인] | ADC 직접 | 2017, 2018년 값만 계절 말. 셀 추가 소량 |
| 4 | Anaktuvuk Pass·Selawik 융해 깊이 측선 2011 | https://doi.org/10.18739/A2N596 | CC BY 4.0 | 1 m 간격 탐침 측선 12개(50–400 m), 2011-08-30–09-28 | 2개 마을 | 66–68°N, -160 ~ -151°. 마을 부지라 교란 표시 필요 | ADC 직접 | 새 블록 1–2개 가능, 셀은 적다 |
| 5 | 유콘강 유역 Indigenous Observation Network ALT 2019, 2020 | https://doi.org/10.18739/A2X05XD00, https://doi.org/10.18739/A23N20F83 | CC0 1.0 | 100 m 격자 10 m 간격 탐침. 2020년 측정 2020-08-26–09-14 | 파일 u42, u44, u46, u47, u49, u50, u51, u55, u57(2019) | CALM U42–U57 와 같은 지점 | ADC 직접 | 새 셀 없음. 기존 셀의 연도 보강 |
| 6 | 북서 알래스카 비버 연못 융해 깊이 2022–2026 | https://doi.org/10.18739/A2W37KX7F | CC0 1.0 | 8월과 3월 말 측정 | csv 127,589 B | 64.45–66.96°N, -165.93 ~ -161.90° | ADC 직접 | 연못 가장자리는 교란·수체 규칙으로 대부분 빠질 것으로 본다. 낮음 |
| 7 | NPS Arctic Network 식생 조사구 2002–2008 (ORNL 1542) | https://doi.org/10.3334/ORNLDAAC/1542 | ORNL DAAC 공개(Earthdata 로그인) | 성장기 최성기 단일 방문, 2002-07-10–2008-08-10 | 조사구 수 [미확인] | 65.28–68.49°N, -165.32 ~ -150.84° | Earthdata | 7월 방문이 주이므로 대부분 `direct_dated`. 8월 방문분만 대상 |
| 8 | FireALT 의 알래스카 비교란 계절 말 지점 | 보유(`data/processed/ext_labels/firealt_talucci2025_points.csv`) | CC BY 4.0 | 탐침 | 122지점 | 기존 셀과 거리 중앙값 0.003 km. 10 km 밖은 14지점(10블록) | 보유 | 새 하위 과제가 되지 않는다 |

### 4.3 캐나다 하위 지역(CA-1 고위도 북극, 허드슨만, 유콘)

| 순위 | 자료 | DOI/URL | 약관 | 방법·시기 | 규모·관계 | 판정 |
|---|---|---|---|---|---|---|
| 1 | North American Arctic Transect 의 NWT·북극 군도 지점(ORNL 1386) | https://doi.org/10.3334/ORNLDAAC/1386 | ORNL DAAC 공개(Earthdata 로그인) | 10 × 10 m 격자 융해 깊이. Green Cabin 2003·2006, Mould Bay 2004·2006, Isachsen 2005–2006 | 북극 군도 3지점. CA-1 에 3셀 | 측정 월 [미확인]. CA-1 6 → 최대 9셀 |
| 2 | Parks Canada CALM 격자(Auyuittuq, Quttinirpaaq, Sirmilik, Ukkusiksalik, Qausuittuq) | open.canada.ca 데이터셋 b5662fca…, 5e8d13ca…, a29dba8f…, 0a827e84…, ed8981ff… | Open Government Licence - Canada | 100점 격자 탐침. 측정일: Sirmilik 대부분 7월 11–25일, Quttinirpaaq 26개 연도 가운데 16개 연도 8월(나머지 7월), Auyuittuq 대부분 8월 1–17일, Ukkusiksalik 대부분 8월, Qausuittuq 2024-08-06·2025-07-22. Sirmilik·Auyuittuq 는 융해관의 연 최대값 표도 있다 | 공원 격자 CSV 에는 좌표가 없다. 좌표 근접으로 보면 Auyuittuq(66.88°N), Sirmilik(72.81°N, 73.01°N), Quttinirpaaq(81.40°N), Ukkusiksalik(65.88°N)는 v3 CALM_Canada 행과 같은 지점이다 [추정] | 새 셀은 Qausuittuq 1개. 나머지는 연도 보강과 융해관 최대값으로 라벨 정의 점검에 쓴다 |
| 3 | Tarnocai & Bockheim 2011, 북극 캐나다 동결토 단면 | https://doi.org/10.1594/PANGAEA.838950 | CC BY 3.0 | 단면 기재의 융해 깊이, 날짜 [미확인] | 90 data points, 61.60–81.84°N, -131.58 ~ -71.30° | 관측 연도가 1990년 이후인지 [미확인]. 보조 |
| 4 | Boike 등 2022, 투크토약툭섬과 매켄지 삼각주 호수 주변 ALT | https://doi.org/10.1594/PANGAEA.949181 | CC BY 4.0 | 1.5 m 탐침, 2021-09-14–09-20 | 28위치, 68.78–69.46°N | CA-2 셀 보강(GGD353, CALM 지점 근처) |
| 5 | USGS 유콘강 유역 Active Layer Network | https://doi.org/10.5066/F7NC5ZFM | 사용·접근 제약 'none'(FGDC 메타데이터) | 45 m 격자 100점 탐침, 늦여름 최대 융해기, 2009– | 알래스카 지점은 CALM U 코드와 같다. 캐나다 지점은 Dawson, Teslin(USGS 자료 쪽 검색 요약 기준 [미확인]) | 유콘 2셀. ScienceBase 가 서버에서 Cloudflare 403 이라 브라우저로 받아야 한다 |
| 6 | Langer 등 2020, Churchill(매니토바)·Deadhorse ALT | https://doi.org/10.1594/PANGAEA.913423 | CC BY 4.0 | 금속봉. Churchill 2018-07-21–25(69점), 2019-07-29–31(71점) | Churchill 140점(2018년 69, 2019년 71). 경도 -100° 동쪽 점 142개가 1 km 셀 4개, 블록 2개 | 7월 단일 방문이라 `direct_dated`. 대상 셀 아님 |
| 7 | Tulemalu Lake 토양 단면(키발리크) | https://doi.org/10.1594/PANGAEA.786465 | CC BY 3.0 | 2006-08-07–08-15 단면 융해 깊이 | 62.90°N, -99.16°, 1조사지 | 1셀. `record_date` 충족 |

### 4.4 러시아·티베트

| 자료 | DOI/URL | 약관 | 방법·시기 | 규모·관계 | 판정 |
|---|---|---|---|---|---|
| Xiao 등 2026, 원취안(Wenquan) GPR ALT 810점 | https://doi.org/10.5281/zenodo.21366503 (WQ_GPR_ALT_810.csv 34,559 B) | CC BY 4.0 | GPR. 제목은 'late-thaw ALT', 점별 날짜 없음 | 810점이 1 km 셀 50개, 블록 4개. 보유 Tibet 셀과 거리 중앙값 1.0 km, 최대 7.5 km | Du 2026 과 같은 조사지. 블록 안 밀도만 늘린다 |
| Polaris II, 북동 시베리아 식물·토양·융해 깊이 | https://doi.org/10.5065/D6NG4NP0 | CC BY 4.0 | 2012-07-08–2013-08-03 | 68.51–69.51°N, 158.90–162.21°E(콜리마 CALM 군집과 같은 지역) | 7월 관측이 많아 대부분 `direct_dated` |
| 체르스키 화재 실험·낙엽송 밀도 구배 융해 깊이(ADC, LTER) | DataONE 색인 제목 기준 | [미확인] | 화재 실험 또는 7월 관측 | 콜리마 CALM 군집 1–2 km 안 | 교란 또는 계절 말 아님. 제외 |
| 일본 ADS 의 Spasskaya Pad(야쿠츠크) 자료 | https://ads.nipr.ac.jp/portal/oai?verb=GetRecord&identifier=A20150303-009&metadataPrefix=DIF | [미확인] | 지온·수분 자료 확인. 9월 중순 ALT 측정은 문헌 언급뿐 [미확인] | Russia_C 의 야쿠츠크 CALM(R42, R43) 부근 | 낮음 |
| TPDC 302540, 270324 등(기존 보류 목록) | `_sources_plan.json` deferred | 계정 필요 | GPR | 티베트 북부, 치롄 | 사용자 계정이 있어야 한다. 1–2일 안 확보 불확실 |

### 4.5 확인 후 제외한 자료

| 자료 | 근거 | 제외 이유 |
|---|---|---|
| Streletskiy 등 2026 CALM 지점 표(figshare 32885756, CC BY 4.0, 642,009 B) | 지점 140곳, 3,500행을 v4 와 대조했다. 25°N 이북 지점의 최소 거리 최댓값은 112.4 km(Stelvio, 시추공)이고 탐침 지점은 모두 2.6 km 이내다 | 새 북반구 탐침 지점 없음. 남극 16곳, 아르헨티나 3곳은 CCI 범위 밖 |
| 몽골 CALM(M 지점) | 같은 표의 METHOD 가 Borehole | 지온 유도. XJ 보조 후보 |
| 스발바르 SESS 2018 PermaSval(Zenodo 4777825) | 설명문: 시추공과 ALT 격자 2개 | 시추공 보간값 위주. XJ 보조 후보 |
| Yukon Permafrost Database(OGL-Yukon) | 지반공학 시추공, 지온, 보고서 색인 | 계절 말 ALT 라벨이 아니다 |
| Nordicana D115 Old Crow Flats 2017–2022 | Nordicana 목록 | ds2369 KTurner 팀(Brock 대학, 2017–2022, 같은 범위)과 같은 조사로 본다 [추정] |
| NGEE ESS-DIVE Seward·Utqiagvik(2016–2023 개별 패키지) | ds2369 NGEE 팀이 같은 지점 포함 | 같은 블록 안 연도 보강만 가능 |
| Sjöberg 등 2026 캉에를루수아크 융해 깊이 측선(PANGAEA 989941, 989942) | 제목: May 2017 | 5월 관측 |
| Resolute 2025 생태 조사(https://doi.org/10.18739/A20C4SN0H) | 설명문: 자갈·암석으로 탐침 시도 | 7월 관측, 유효 측정 불확실 |
| NSIDC GGD314 핀란드 라플란드 팔사 | 1972–1982, NSIDC 미보관 | 1990년 이전 |
| Voigt 등 2023 메탄 흡수 조사(PANGAEA 953119) | 변수에 'Thaw depth of active layer' 있음. Kilpisjärvi 부근 고지대 | 영구동토 유무 [미확인], 플럭스 캠페인 부속값 |
| 중국 동북(다싱안링) | 검색에서 공개 점 자료를 찾지 못했다 | 논문 본문 값만 확인 |
| 남극(PERMATHERMAL Deception, Livingston 등) | CMR 목록 | CCI ALT 25–85°N 밖 |

## 5. 새 macro 지역이 나오지 않는 이유

1. **중복 구조**: 북반구의 장기 탐침 자료는 CALM 이 거의 모두 모았고(PANGAEA 972777 과 figshare 32885756), 알래스카·캐나다 북서부의 단기 조사는 ds2369 가 모았다. 두 묶음 밖의 공개 자료는 같은 조사지의 다른 연도나 실험구가 대부분이다.
2. **라벨 정의**: 보유 지역 밖의 장기 관측(몽골, 알프스, 노르웨이 산지, 스발바르 시추공, 퀘벡·래브라도)은 지온 유도다. LGD 는 이를 확인적 풀에서 빼도록 정했다(6B.1).
3. **계절 말 조건**: 생태 조사 부속 융해 깊이(NPS ARCN, Polaris, Churchill, Parks Canada Sirmilik)는 7월 단일 방문이 많다.
4. **채점 범위**: CCI ALT 가 25–85°N 이라 남반구 자료는 채점 셀이 될 수 없다.
5. **계정**: 티베트(TPDC), 중국 동북(NCDC), 그린란드(GEM)의 직접 측정 자료는 계정이나 승인이 필요하다.

## 6. 1–2일 안의 현실적 추정

| 항목 | 추정 | 근거와 조건 |
|---|---|---|
| 기준을 채우는 새 독립 macro 지역 | 0개 | 4절 전체 |
| 조건부 적격 지역 | 0–1개(NAtlantic 약관 확인분 판) | 현재 채점 블록 합집합 6. Stordalen, Adventdalen 2023 이 새 채점 블록 2개를 줘야 8에 닿는다. 셀 조립, CCI·토양 도일 유효성, 분할 구조 확인 전에는 판정할 수 없다 |
| 새 하위 지역 과제(30셀 이상) | 1개 유력, 최대 2개 | BNZ 150지점(내륙 도로망). EDI 접근이 열려야 한다. 두 번째는 BNZ RSN, ViPER, Anaktuvuk·Selawik 를 묶을 때의 [추정] |
| 기존 과제의 셀 보강 | CA-1 6 → 9–10셀, Tibet 원취안 50셀(같은 블록), CA-2 소량, Russia 소량 | 4.3, 4.4 |
| 약관 회신 의존 | calm_web_subsites(13셀, 블록 4), cusp_v1_1, GGD353, Walker 보고서 값 | `docs/LGD_LICENSE_CONTACTS_2026-09-30.md`. 회신이 오면 NAtlantic 은 공개 자료 추가 없이 적격이다(전체 판 합집합 10) |

## 7. 알래스카 개발·타 지역 시험 설계(질문 1 후반부)

자료 측면에서 이 방향은 가능하다. 다만 '독립 지역 수'의 약점을 없애지는 못하고, 정책 학습의 과적합을 시험하는 구조를 만든다.

| 단계 | 과제 | 쓰임 |
|---|---|---|
| 개발 | AL-1–AL-6(셀 471–5,154). 새 내륙 과제(BNZ)가 확보되면 추가 | 정책(GBM, DeepSets) 학습과 하이퍼파라미터 선택. 알래스카 안에서 하위 지역 하나씩 빼는 교차 검증 |
| 동결 | 개발 단계 종료 시 정책과 비교 규칙을 고정하고 등록 | XD 사전 등록 문서에 해시와 함께 기록 |
| 시험 | LE-1, LE-2, CA-2, CA-3, Tibet_LGD(5개). Russia_C(57셀)는 n = 10 까지만 | 알래스카 밖에서 한 번만 평가. 비교 대상은 무작위와 블록 층화 k-center 규칙 |
| 판정 | 과제 5개 단측 부호 검정. 5개 모두 같은 방향일 때만 p = 0.031 | 이 기준을 결과 열람 전에 고정한다. 4/5 는 '방향 일치 우세'로만 서술 |

주의점:

- AL-1, AL-4, AL-6 은 채점 블록 최소가 1이라 개발 단계의 과제별 CI 가 불안정하다(커버리지 문서 2절).
- 시험 과제의 채점 블록 최소도 2–4(CA-2 2, CA-3 3, LE-1 3, LE-2 4)라 과제 단위 CI 보다 과제 간 부호 일치로 판정하는 편이 맞다.
- Tibet 는 계수 비가 원천 범위 밖인 심부 레짐이라(6B.7) 배치 정책의 시험에서도 따로 표시한다.
- 새 자료로 만든 과제는 LGD 규칙대로 새 판(v5)과 사전 등록 뒤에만 쓴다. LGD 는 '새 지역 실행을 제출한 뒤 확보한 자료는 그 실행의 대상 셀에 넣지 않는다'고 정했다(6B.2).

## 8. 구현 단계 취득 목록

순서는 기대 이득과 약관 명확성 순이다. 파서 이름은 기존 규약(`scripts/1_data_prep/parse_ext_<id>.py`, `data/processed/ext_labels/<id>_points.csv`)을 따른다.

| 순서 | id(제안) | URL | 형식·크기 | 받는 방법 | 다음 처리 |
|---|---|---|---|---|---|
| 1 | swe_stordalen_crill | https://zenodo.org/records/10420396 | xlsx 194,902 B, pdf 2,245,763 B | Zenodo 직접 | `series_max`(8월 1일 이후 최댓값), 지점 좌표는 pdf 지도와 ICOS SE-Sto 좌표로 확인 |
| 2 | sjm_adventdalen_wendt2023 | https://zenodo.org/records/11187360 | 파일 목록 [미확인] | Zenodo 직접 | 2023년 9월 초 `record_date`, 지점 좌표 확인 |
| 3 | (NAtlantic_lic 재계산) | 위 1, 2 를 넣은 v5 후보 표 | 로컬 가벼운 처리 | `lgd_eligibility` 와 같은 방식의 `--count-only` | 채점 블록 합집합이 8 이상인지 확인 후 XF 등록 |
| 4 | ak_bnz_blackspruce150 | https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=140 | [미확인] | 브라우저(EDI API 가 서버 요청을 거부) | 약관 확인, 측정 월 확인, 1 km 셀 집계, 기존 AL 블록과 겹침 계산 |
| 5 | ak_bnz_rsn | https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=605 | [미확인] | 브라우저 | 영구동토 부재 값의 검열 처리 |
| 6 | ak_viper_td | https://doi.org/10.18739/A22J6848J | csv 8개(최대 22,787 B) | ADC 직접 | 2017, 2018년 값만 대상 |
| 7 | ak_anaktuvuk_selawik | https://doi.org/10.18739/A2N596 | [미확인] | ADC 직접 | 마을 부지 교란 표시 |
| 8 | ca_naat_ornl1386 | https://doi.org/10.3334/ORNLDAAC/1386 | csv(크기 [미확인]) | Earthdata(기존 .netrc) | 측정 월 확인 후 CA-1 보강 |
| 9 | ca_parks_qausuittuq | https://open.canada.ca/data/en/dataset/ed8981ff-c8b3-42e7-b02c-1d1b0373a991 | csv 8,686 B | 직접 | 2024-08-06 값만 대상 |
| 10 | ca_moses_tuk2021 | https://doi.org/10.1594/PANGAEA.949181 | 106 data points | PANGAEA 직접 | CA-2 보강 |
| 11 | ca_usgs_aln | https://doi.org/10.5066/F7NC5ZFM | 지점별 xlsx/csv [미확인] | 브라우저(ScienceBase 403) | Dawson, Teslin 2셀 |
| 12 | grl_petrone_pangaea | https://doi.org/10.1594/PANGAEA.845258 | zip 약 2.5 MB | PANGAEA 직접 | CUSP Petrone 행의 원출처 대체(약관 경로) |
| 13 | qtp_xiao_wenquan | https://doi.org/10.5281/zenodo.21366503 | csv 34,559 B | Zenodo 직접 | 기존 Tibet 블록 안 밀도 보강, 날짜 없음 표시 |
| 14 | (XJ 보조 후보) | Nordicana D8, Zenodo 4777825, 몽골·알프스 보유 행 | 다양 | 다양 | 지온 유도 라벨을 낮은 가중의 원천 행으로만 사용 |

## 9. 남은 문제와 미확인 항목

1. BNZ 140, 605 의 약관과 측정 월, 그리고 EDI 내려받기 경로. 2026-10-04 서버 요청은 'not authorized to execute service method readMetadata' 로 거부되었다.
2. Stordalen 과 Adventdalen 2023 지점이 실제로 새 0.5° 채점 블록이 되는지, CCI ALT 와 토양 도일이 유효한지.
3. USGS ALN 의 캐나다 지점 목록(Dawson, Teslin)은 검색 요약에서만 확인했다. 원문 열람은 ScienceBase 차단으로 하지 못했다.
4. NAAT(ORNL 1386)와 NPS ARCN(ORNL 1542)의 측정 월별 행 수. 파일 열람에 Earthdata 로그인이 필요하다.
5. Nordicana D115 와 ds2369 KTurner 팀 자료의 동일성은 범위와 연도로 추정했다.
6. 새 자료를 넣는 판은 v5 와 XF 사전 등록이 필요하다. LGD 결과(L1e, L4e, L8e, L38–L42)는 바꾸지 않는다.
7. 약관 회신(CALM 누리집, CUSP, GGD353, Walker 보고서)은 사용자 연락 사항으로 그대로 남는다. 회신이 오면 공개 자료 추가 없이 NAtlantic 전체 판이 주 판정이 된다.

## 10. 근거

**저장소 파일**

- `data/processed/fidelity_base_v4_meta.json`(by_source, n_rows)
- `data/processed/fidelity_base_v4.csv`(CALM_Canada 행, 거리 계산)
- `data/processed/lgd_eligibility_v1.csv`, `data/processed/lgd/lgd_eligibility_lic.csv`, `data/processed/lgd/run_tables/NAtlantic_lic/fidelity_base_v3.csv`
- `data/processed/lg_subregion_map_v1.csv`, `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md`
- `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B, `docs/LGD_LICENSE_CONTACTS_2026-09-30.md`, `data/processed/ext_labels/_sources_plan.json`
- `data/processed/ext_labels/firealt_talucci2025_points.csv`
- `data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv`
- `data/raw/cci_alt/ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-2021-fv04.0.nc`(위도 범위 속성)

**웹(열람일 2026-10-04)**

- DataONE 연합 색인: https://cn.dataone.org/cn/v2/query/solr/
- PANGAEA 검색 API: https://www.pangaea.de/advanced/search.php
- NASA CMR: https://cmr.earthdata.nasa.gov/search/collections.json
- 캐나다 공개자료 CKAN: https://open.canada.ca/data/api/action/package_search
- Streletskiy 등 2026 자료: https://doi.org/10.6084/m9.figshare.32885756, 논문 https://www.nature.com/articles/s43247-026-03824-1
- ION: https://doi.org/10.18739/A2X05XD00, https://doi.org/10.18739/A23N20F83
- USGS ALN 메타데이터: https://data.usgs.gov/datacatalog/metadata/USGS.5941496be4b0764e6c64a4db.xml, https://catalog.data.gov/dataset/yukon-river-basin-active-layer-network-active-layer-depth-measurements, https://www.usgs.gov/data/active-layer-data-yukon-river-basin-alaska-and-canada
- NPS ARCN 사용자 안내: https://daac.ornl.gov/ABOVE/guides/Arctic_Network_Veg_plots.html
- NAAT 사용자 안내: https://daac.ornl.gov/ABOVE/guides/North_Slope_Transect_Veg_Maps.html
- Parks Canada: https://open.canada.ca/data/en/dataset/a29dba8f-a791-4012-8c27-084ae7b76da9, https://open.canada.ca/data/en/dataset/b5662fca-2082-428e-a9c3-3234822bbecd, https://open.canada.ca/data/en/dataset/0a827e84-278c-4150-aeef-77eb9c9612ce, https://open.canada.ca/data/en/dataset/5e8d13ca-2af8-4335-b770-20b898a5b530, https://open.canada.ca/data/en/dataset/ed8981ff-c8b3-42e7-b02c-1d1b0373a991
- Stordalen: https://zenodo.org/records/10420396
- Adventdalen: https://zenodo.org/records/11187360, https://tc.copernicus.org/articles/20/1179/2026/
- SESS 2018: https://zenodo.org/records/4777825
- Petrone 2016: https://doi.org/10.1594/PANGAEA.845258
- NSIDC GGD622, GGD314: https://nsidc.org/data/ggd622/versions/1, https://nsidc.org/data/ggd314/versions/1
- Langer 2020: https://doi.org/10.1594/PANGAEA.913423
- MOSES 2021: https://doi.org/10.1594/PANGAEA.949181
- Tarnocai & Bockheim 2011: https://doi.org/10.1594/PANGAEA.838950
- Tulemalu Lake: https://doi.org/10.1594/PANGAEA.786465
- Voigt 2023: https://doi.org/10.1594/PANGAEA.953119
- Xiao 2026: https://zenodo.org/records/21366503
- Nordicana D 목록: https://nordicana.cen.ulaval.ca/en/list-of-publications
- Yukon Permafrost Database: https://open.canada.ca/data/en/dataset/42bc4eff-c34d-49bc-9970-b1e2218e212f
- Abisko 공개자료 안내: https://www.polar.se/en/research-support/open-data/data-from-abisko-scientific-research-station/
- ADC: https://doi.org/10.18739/A22J6848J, https://doi.org/10.18739/A2N596, https://doi.org/10.18739/A2W37KX7F, https://doi.org/10.18739/A2707WP16, https://doi.org/10.5065/D6NG4NP0, https://doi.org/10.18739/A20C4SN0H
- EDI: https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=140, https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-bnz&identifier=605
