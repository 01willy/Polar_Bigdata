# 데이터 취득 실행계획 (Data Acquisition Plan)

> **[2026-07-10 갱신]** 이번 세션 진행/상태 (상세: `docs/EXPERIMENT_LOG.md` 2026-07-10 항목)
> - **완료 — ESA CCI ALT prior 추출**: 기보유 25년 NetCDF(828MB)를 다년평균 → 14,348 셀 최근접 추출(`scripts/1_data_prep/enrich_cci_cell.py`). 전 셀 유효, 관측 셀평균과 r=0.53. ablation **M8 +CCI는 무익**(기후와 중복) — prior/benchmark로만 활용.
> - **정체 — SoilGrids 250m**: ISRIC WebDAV VRT vsicurl 원격 windowed-read가 20분+ 산출 0으로 정체. 중단. **재시도 방안**: (a) 사전 타일 다운로드 후 로컬 창 읽기, (b) 시간대 바꿔 재시도, (c) 대체 미러. 스크립트 `scripts/0_download/soilgrids_alaska.py` 유지.
> - **다음 우선순위**: SoilGrids(M6) → Sentinel-2/NDVI(M7) → PANGAEA CALM 2025(T-lite 월별 forcing). 각 모달리티는 동일 fold/target/가중/지표로 M3 대비 Δskill·Δcoverage·ΔAOA 보고 후 채택.
> - **채택 원칙**: RMSE↓라도 전이(LORO)·커버리지 악화 시 주력 채택 안 함. 고차원 EO/SAR encoder(DL)는 handcrafted feature가 효과를 보인 뒤 게이트 통과 시만.

> 작성일: 2026-06-30
> 연구 목표: 전 지구 borehole 지중온도 + CALM ALT 관측 + 공변량 → 공간인식 딥러닝으로
> (1) 주 출력 ALT 지도, (2) 0~20m 3D 지중 열구조, (3) 셀별 불확실성 예측, 그리고
> "알래스카 학습 → 타 영구동토 지대 전이(LORO)" 검증.
> 현 한계: 공변량이 거칠어(WorldClim 18km) LORO RMSE ~97cm. → **지중온도(borehole) 보강 + 공변량 고해상화**가 핵심.

> **디스크 경고 (선행 확인 결과)**: `/home`은 13T 중 12T 사용, **여유 307G(98% 사용)**.
> 즉시 수GB는 가능하나 ERA5-Land/Daymet/MODIS/SoilGrids/ArcticDEM 전체 미러는 **불가**.
> 모든 격자 데이터는 반드시 **알래스카+전이 target 영역으로 서브셋 후** 받을 것. (상세는 D·E절)

---

## A. 무계정 즉시 다운로드 가능 (사용자 계정 불필요)

| 데이터셋 | 변수 | 해상도 | 용량 | 실제 다운 방법 (URL/명령 힌트) | 우선순위 |
|---|---|---|---|---|---|
| **NSIDC GGD200 — W. Siberia deep wells** (지중온도) | 깊이별(20·50·100·200·…·2000m) 암석온도, 좌표, 지질, 열류량, 동토두께 (736개소, 1960-95 정적) | point(736개소) | **~104 KB** | **FTP 프로토콜 필수**(https 아님). `wget -np -nd "ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd200_boreholes_siberia/deepwell.dat"` (+ README.txt, deepwell.gif, deepwell.xls). .xls는 키릴 cp1251 인코딩 | **1 (지중온도)** |
| **NSIDC G10015 — Alaska Arctic Slope deep boreholes** (지중온도) | 깊이별 thermistor 로그(Depth m vs ITS-90 ℃), 24개 시추공, 1973-2014 | point(24개) | ~1.2 MB(zip), 해제 ~3.8MB | `wget "https://noaadata.apps.nsidc.org/NOAA/G10015/G10015.zip"` (HTTPS, 인증 불필요). 시추공×로그날짜별 2열 ASCII 184개 → 파싱·QC 필요 | **2 (source 정밀화)** |
| **Nordicana D8 — NE Canada borehole + near-surface GT** (지중온도) | 시추공+근지표 온도 시계열(시간단위), 11지역 44측정지, 1988-2023 | point(~44) | ~수백KB~2MB(ZIP ~45개) | publication 페이지의 "Get file"=`/en/download?id=<숫자>` 직링크(예 id=2240,2241,2285…) wget 순회. **주의: 본 샌드박스 egress 차단(HTTP 000) — 외부망/브라우저에서 받을 것** | **3 (전이 target=캐나다)** |
| **PERMOS — Swiss Alpine borehole GT** (지중온도) | 보어홀 다깊이 thermistor 시계열 + 지표온도 + ERT, 1987-2023, ~30 시추공 | point(~30) | **~38 MB** | DOI ZIP 직다운: `wget https://www.permos.ch/doi/permos-dataset-2024-1`(전체 ZIP). 또는 Shiny 브라우저. 산악 전이 target | 4 |
| **ESA CCI Permafrost v4** — GT/ALT/Extent (모델 product, 벤치마크) | GT 연평균(표면/1m/2m/5m/10m), ALT, Extent, **셀별 불확실성**. 1997-2021. 북반구(area4) | 1km / 연 | **GT ~9.5GB, ALT ~0.9GB, Extent 소형** | CEDA HTTPS 디렉토리(로그인 불필요): `wget -r -np -nH --cut-dirs=N -A '*.nc' https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/ground_temperature/L4/area4/pp/v04.0/` . **디스크 주의: ALT만 우선, GT는 알래스카 잘라쓰기** | 5 (벤치마크/weak-label) |
| **SoilGrids 2.0** (공변량: 토양물성) | texture/bulkdensity/SOC/coarse frag (0~200cm 6깊이) + quantile 불확실성. **지중온도 없음** | 250m | 맵당 ~5GB(전체 수백GB~TB) | WebDAV 익명(user/pw 모두 `anonymous`): `https://files.isric.org/soilgrids/latest/data/` . GDAL VRT로 **알래스카 bbox만** gdal_translate | 6 (공변량) |
| **CHELSA V2.1** (공변량: 기후) | tas/tasmax/tasmin/pr/bio 등 지표기후. **지중온도 없음** | 1km(30 arcsec) | 파일당 ~140-165MB(전체 수TB) | 익명 HTTPS: `wget https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/tas/1981-2010/CHELSA_tas_01_1981-2010_V.2.1.tif` . **메인 chelsa-climate.org/downloads 는 404** — S3 버킷 사용 | 7 (공변량, WorldClim 대체) |
| **Liu et al. 2024 — Gridded ALT 2003-2020** (모델 product) | 연 ALT (BigTIFF). **지중온도 없음** | 1km/연 | **~1.86GB**(18파일×~103MB) | Zenodo 직다운: `https://zenodo.org/api/records/10070610/files/ALT_2020.tif/content` . 최신 GDAL/rasterio(BigTIFF) 필요 | 8 (ALT 벤치마크) |
| **ArcticDEM Mosaic v4.1** (공변량: 지형) | slope/aspect/curvature/TWI/TPI 산출용 DSM. **지중온도 없음, >60N만** | 10m/32m(2m 일부) | 전체 ~17TB | AWS 익명: `aws s3 ls --no-sign-request s3://pgc-opendata-dems/arcticdem/mosaics/` . **알래스카 최신은 NGA 제약으로 제외(E절 참조)**. 32m·타일 단위로만 | 9 (지형 공변량) |
| **Daymet V4 R1** (공변량: 북미 기후) | tmax/tmin/prcp/srad/vp/**swe**/dayl 일별. **북미만, 지중온도 없음** | 1km/일 | 전체 3.2TB | 단일픽셀 API 무인증: `https://daymet.ornl.gov/single-pixel/api/data?lat=64.8&lon=-147.7&vars=tmax,tmin,swe&years=2020` . 격자 서브셋은 THREDDS/NCSS | 10 (source 공변량) |

> 참고: A절 다운로드 중 인증이 전혀 없는 것은 GGD200/G10015/Nordicana/PERMOS/SoilGrids/CHELSA/Liu/ArcticDEM(AWS). CCI는 익명이나 막히면 무료 CEDA 계정.

---

## B. 계정 필요 (사이트별로 정확히)

| 데이터셋 | 사이트(포털) | 만들 계정 | 로그인 후 자동다운(API키/토큰) | 용량 | 왜 필요 |
|---|---|---|---|---|---|
| **TPDC Qinghai-Tibet Permafrost Thermal State (ESSD 2021)** — 84시추공 **3/6/10/20m** 다깊이 시계열 + ALT 12사이트 | data.tpdc.ac.cn | **TPDC 무료 회원가입**(실명, 이메일) | **불가(수동)** — 공개 REST API/토큰/.netrc 미문서화, 세션 쿠키 기반 브라우저 다운만. 일부 데이터는 오프라인 신청·승인 가능 | 수십~수백 MB(추정, 미명시) | **0~20m 목표층과 깊이 정확 일치 + ALT + 고원 전이**를 단일 셋으로 충족. 지중온도 신규성 최우선 |
| **TPDC MAGT15m (ESSD 2025)** — 231개 15m GT point + **1km GeoTIFF** | data.tpdc.ac.cn | TPDC 무료(CC BY 4.0) | 불가(수동, SPA·세션) | 수~수십 MB(추정) | 18km→1km 격자 GT로 **거친 공변량 한계 직접 완화** + 고원 baseline. 단 단일깊이·연평균 |
| **TPDC China-Russia Pipeline GT (ESSD 2022)** — 20시추공 **10~60m** 10년 시계열 | data.tpdc.ac.cn (DOI 10.11888/Cryos.tpdc.272357) | TPDC 무료 | 불가(수동, 세션·일부 승인대기) | 수MB~수십MB(추정) | 동북아 **불연속 영구동토 전이 도메인** 추가, 0~20m 다깊이 시계열 |
| **ABoVE Soil Moisture & ALT, AK/Canada 2005-2024 (ds 2369)** — **~224,200 in-situ ALT**(기존 3,604 site-year의 ~60배) | ORNL DAAC / NASA Earthdata | **NASA Earthdata Login**(무료, 즉시) | **가능** — `.netrc`(machine urs.earthdata.nasa.gov) / Bearer 토큰 / `earthaccess` 라이브러리 | **~116 MB**(CSV 1개) | **주출력 ALT 라벨 밀도 압도** → 97cm RMSE 정면 공략. 단 지중온도는 없음(ALT/수분만) |
| **ABoVE ReSALT L/P-band ALT, AK 2017 v3 (ds 2004)** — 30m 격자 ALT+계절침하+불확실성, 51사이트 | ORNL DAAC / NASA Earthdata | NASA Earthdata Login(무료) | 가능(.netrc+쿠키 / earthaccess / S3 us-west-2) | **6.89 GB**(52파일) | 고해상 ALT 지도+**셀별 불확실성** = 우리 출력 구조와 일치. 단 단일연도·수분만(지중온도 없음) |
| **ERA5-Land hourly (1950-)** — 4깊이 **soil_temperature(0~289cm)** + 적설(SWE/cover/density) | Copernicus CDS | **CDS/ECMWF 무료** + Personal Access Token | **가능(권장)** — `~/.cdsapirc`(url+key) + `pip install cdsapi`. 웹폼이 cdsapi 스크립트 생성 | 알래스카 서브셋 변수·연도당 수십~수백MB(전지구 다년은 TB) | **WorldClim의 결정적 대체** — 지중 열구조 물리 사전정보 + 최강 ALT 공변량, 전 전이지역 균일 |
| **MODIS MOD11A2 LST v6.1 (8일,1km)** — 주/야 LST(상부경계 강제력) | NASA Earthdata / LP DAAC | NASA Earthdata Login(무료) + 앱 사전승인 | 가능(.netrc+urs_cookies / AppEEARS 재투영 / GEE 대안) | 그래뉼당 4-8MB(알래스카 4타일×다년 수GB~수십GB) | 0~20m·ALT 상부 경계 강제력. ERA5-Land와 일부 중복(nice-to-have) |

> **B절 자동화 등급 요약**:
> - **완전 자동화 가능(API/토큰)**: ERA5-Land(cdsapi), ABoVE 2369/2004(earthaccess/.netrc), MODIS(.netrc).
> - **수동 다운만(자동화 불가)**: TPDC 3종 전부. → 디스크/시간 예산엔 수동 작업으로 잡을 것.

---

## C. 지중온도(ground temperature) 신규 소스 하이라이트

> 기준: **우리 GTN-P(397 borehole/450 dataset, 전지구)** 를 보강하는 가치 = 깊이밀도 × 신규지역(전이 target) × 0~20m 목표층 적합성.

1. **NSIDC GGD200 (서시베리아 736개소)** — **단일 최대 시추공 밀도**로 우리가 가장 약한 서시베리아를 채움. 20m부터 다깊이라 0~20m 목표층 완전 커버. 무계정·정적(1960-95)이라 정적 모델과 정합. **우리 데이터의 공변량 격차를 가장 직접 메우는 신규 지중온도.** (단 점 요약표, 시계열 아님, 결측 많음.)

2. **TPDC Qinghai-Tibet Thermal State (84시추공 3/6/10/20m)** — 깊이층이 **우리 0~20m 3D 목표와 정확히 일치**하고 ALT 라벨까지 동봉. 고원(고고도)은 GTN-P에 희박한 강력한 신규 전이 도메인. **깊이·ALT·전이를 한 셋으로 충족하는 최고가치.** (계정·수동 다운, 지역=고원.)

3. **TPDC China-Russia Pipeline (20시추공 10~60m, 10년)** — **불연속(discontinuous) 영구동토**라는 우리 분류에 드문 전이 도메인. 깊이·시간 밀도 높음. (회랑형 좁은 범위, 계정·수동.)

4. **Nordicana D8 (NE 캐나다 44측정지, 시간단위)** — **알래스카→캐나다 전이의 직접 target-domain 깊이별 라벨**. 무계정·최신(2023). 근지표+시추공 동시라 0~20m 얕은 열구조 보정에 직접 사용. (Nunavik/Nunavut 한정, 샌드박스 egress 차단.)

5. **NSIDC G10015 (알래스카 Arctic Slope 24 심부 시추공)** — source 도메인 정밀화. GTN-P와 일부 중복 가능하나 장기 고해상 thermistor 로그라 **0~20m 학습/검증 라벨 강화**. 무계정·소형.

6. **PERMOS (스위스 알프스 ~30 시추공)** / **NORPERM(노르웨이·스발바르)** — 산악·유럽북극이라는 **전이 다양성** 보강용. PERMOS는 무계정·정밀 시계열. NORPERM은 공식 URL 死(404), 사실상 **GTN-P 경유**가 현실적이라 신규성 낮음(이미 보유 카탈로그와 중복 가능) → 후순위.

> **핵심 메시지**: 지중온도 신규성은 **GGD200(서시베리아·무계정) + TPDC 3종(고원·동북아·계정)** 이 양대 축.
> Nordicana/G10015/PERMOS는 전이 검증의 target 라벨 보조.

---

## D. 권장 취득 순서 (디스크 307G·전이목표 고려)

### 1순위 — 무계정·소형·고가치 지중온도 (당장, 합계 < 50 MB)
1. **NSIDC GGD200** (서시베리아 지중온도, ~104KB) — FTP wget
2. **NSIDC G10015** (알래스카 Arctic Slope, ~1.2MB) — HTTPS wget
3. **PERMOS** (스위스 산악 GT, ~38MB) — DOI ZIP
4. **Nordicana D8** (NE 캐나다 GT, ~2MB) — 외부망/브라우저(샌드박스 egress 차단)

> 이유: 디스크 부담 없음, 인증 불필요, 전이 target 지중온도를 즉시 확보. 파서/QC 먼저 정비.

### 2순위 — 계정 발급 후 핵심 보강 (며칠, 자동화 우선)
5. **NASA Earthdata Login 발급** → **ABoVE ds 2369**(~116MB, ALT 22만건 = 주출력 라벨 밀도) `earthaccess`로 자동
6. **TPDC 무료 가입** → **Qinghai-Tibet Thermal State(2021)** 수동 다운(깊이 정확일치+ALT+고원 전이) → 이어서 **China-Russia Pipeline(2022)**, **MAGT15m(2025)**
7. **Copernicus CDS 가입 + .cdsapirc** → **ERA5-Land** **알래스카+전이 target bbox로만** soil_temperature 4층 + SWE 서브셋(WorldClim 대체 = 거친 공변량 1차 해결)

> 이유: 인증 1회로 자동 파이프라인 확보(earthaccess/cdsapi). TPDC는 수동이지만 지중온도 신규성 최상위라 병행.

### 3순위 — 공변량 고해상화 & 벤치마크 (디스크 관리하며 선택적, 모두 서브셋)
8. **CHELSA V2.1** — 알래스카+전이 target 타일만(WorldClim 1km 대체/보강)
9. **SoilGrids 2.0** — 알래스카 bbox만 VRT 추출(열전도도/열용량 공변량)
10. **ESA CCI Permafrost v4** — **ALT(~0.9GB) 먼저**, GT는 알래스카 잘라서(전체 ~9.5GB 받지 말 것). 벤치마크/weak-label
11. **Liu et al. ALT(~1.86GB)**, **ArcticDEM 32m 타일**, **Daymet 서브셋**, **MODIS LST** — 여유·필요시

> **디스크 규율**: 8~11은 전체 미러 금지. 307G 중 안전 운용은 ~수십GB. 받기 전 `df -h` 확인, 영역 서브셋 후 다운, 사용 후 raw 정리.

---

## E. 주의 / 함정

### 인증·승인 방식
- **TPDC 3종**: 무료지만 **실명 가입 필요**, 일부 데이터셋은 **오프라인 신청→검토→승인 대기**가 있을 수 있음. 공개 API/토큰/.netrc **미문서화 → 스크립트 자동화 불가, 브라우저 수동 다운만**. 중국 서버라 해외 접속 속도/간헐 접근성 이슈.
- **NASA Earthdata(ABoVE 2369/2004, MODIS)**: 가입 즉시 사용(승인대기 없음)이나 **다운로드엔 반드시 로그인**. `.netrc`(machine urs.earthdata.nasa.gov) + `~/.urs_cookies` 미설정 시 인증 리다이렉트에서 wget **실패**. MODIS는 추가로 **앱(LP DAAC Data Pool) 사전 authorize** 필요.
- **Copernicus CDS(ERA5-Land)**: ① **데이터셋별 라이선스 1회 수락** 안 하면 cdsapi가 license 에러로 실패(웹폼에서 1회 동의). ② 인증은 `.netrc` 아닌 **`~/.cdsapirc`(url+key, 2024 CDS-Beta Personal Access Token)** — 구 UID:APIKEY/구 cdsapi는 동작 안 함(최신으로 업데이트). ③ 큐 대기·요청크기 제한(작게 분할). ④ 데이터 2~3개월 지연.

### 프로토콜·경로 함정 (그대로 치면 실패)
- **GGD200**: 랜딩의 `ftp://` 링크를 https/브라우저로 치면 실패(HTTP 000/404). **반드시 FTP 클라이언트(curl/wget ftp://)**. NOAA@NSIDC HTTPS 전환은 이 fgdc/ 트리엔 미적용.
- **CHELSA**: 프롬프트의 `chelsa-climate.org/downloads/` 는 **404**. 실제는 `os.unil.cloud.switch.ch/chelsa02/` S3. EnviDat "Login disabled" 안내는 데이터 접근과 무관(익명).
- **NORPERM**: 논문 인쇄 URL(ngu.no/kart/permafrost)은 **404**. geo.ngu.no는 지도 뷰어/WMS만(정형 다운로드 없음) → 사실상 **GTN-P 경유**. 후순위.
- **Nordicana D8**: 데이터는 살아있으나 **이 샌드박스에서 Université Laval 서버 egress 차단(TCP 타임아웃, HTTP 000)**. 외부망/브라우저로 받을 것.
- **ESA CCI**: 프롬프트 UUID(7479606004…)는 **컬렉션 레코드**(직다운 링크 없음). 실제 파일은 `dap.ceda.ac.uk/neodc/…` 변수별 경로. GT의 1m/2m/5m/10m는 **측정값 아님(MODIS LST+ERA5+CryoGrid 산출물)**, 1997-2002(Case B)와 2003-2021(Case A) 방법 갈림.

### 지역·내용 제약 (목표와의 정합)
- **지중온도 "없음" 주의**: ABoVE 2369/2004(ALT·수분만), SoilGrids/CHELSA/Daymet(기후·토양물성만), MODIS(표면 LST만), ArcticDEM(지표 DSM만), Liu ALT(ALT만), CCI의 ALT/Extent. **깊이별 지중온도는 GGD200·G10015·Nordicana·PERMOS·TPDC 3종·CCI-GT·ERA5-Land(soil_temp)** 만 제공.
- **알래스카 미커버(전이 target 전용)**: GGD200(서시베리아), Nordicana(NE캐나다), PERMOS(알프스), NORPERM(노르웨이/스발바르), TPDC 전부(고원·동북아). → 학습용 아님, **전이 검증 라벨**로 사용.
- **ArcticDEM 알래스카 함정**: 2022.6 이후 알래스카는 **NGA/EOCL 계약 제약으로 공개 배포 제외**. 최신·완전 알래스카 커버 불가(GEE 카탈로그 표기와 상충, PGC/AWS 제약이 실질 기준). 2m는 .edu 계정/유료.
- **시점 편중·정적성**: ABoVE 2369 측정은 8~9월(해빙 최대기) 편중. ReSALT는 2017 단일연도 스냅샷. GGD200/G10015는 시계열 아닌 캠페인/정적 로그 → 파싱·QC 필수.

### 디스크 (재강조)
- `/home` **98% 사용, 307G 여유**. ERA5-Land/Daymet/MODIS/SoilGrids/ArcticDEM/CHELSA **전체 미러 금지**. 전부 **알래스카+전이 target bbox 서브셋** 후 다운. 받기 전 `df -h`, 사용 후 raw 즉시 정리. SoilGrids는 Homolosine 좌표라 재투영 필요할 수 있음.
