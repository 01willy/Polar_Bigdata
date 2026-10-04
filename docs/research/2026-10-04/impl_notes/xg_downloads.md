# XG·XB 제품 내려받기 구현 기록 (2026-10-04)

- 범위: 계획서 0.3절의 라벨과 결합하지 않는 내려받기 가운데 ALT 제품 4종(CCI v5, Wei 2026 v2 ALT zip 3개, Aalto 2018, Yi·Kimball ds1760).
- 라벨 표는 열지 않았다. 제품 파일은 머리 정보(차원, 좌표계, 자료형, 결측 값, 파일에 저장된 통계 태그)만 읽었고 화소 값은 읽지 않았다.
- 계획서와 충돌하는 점은 찾지 못했다. 아래는 2단계 추출(`scripts/1_data_prep/xg_extract_products_v1.py`)과 1c 마스크 단계가 알아야 할 사실이다.

## 1. 저장 위치(계획서에 폴더 이름이 정해져 있지 않아 새로 정했다)

| 제품 | 폴더 | 자료 파일 | 합계(바이트) | 대조 |
|---|---|---|---:|---|
| CCI v5 | `data/raw/cci_alt_v5/` | NetCDF 27개(1997-2023) | 843,453,733 | CEDA 목록 크기 27/27 일치(CEDA 는 체크섬 미공개) |
| Wei 2026 v2 | `data/raw/wei2026_alt_v2/` | ALT zip 3개 | 2,560,802,049 | Zenodo md5 3/3 일치 |
| Aalto 2018 | `data/raw/aalto2018_alt/` | `ALT_Rasters.zip` | 672,583,929 | Zenodo md5 일치 |
| Yi·Kimball ds1760 | `data/raw/ornl_ds1760/` | NetCDF4 1개 | 36,766,169 | ORNL sha256·CMR sha256 일치 |

합계 4,113,605,880 바이트로 계획서 2.7절의 '약 4.1 GB' 와 맞는다. 각 폴더에 `SOURCE.md`, `download_manifest.json`, `checksums.md5`, `checksums.sha256` 이 있다. 계획서 0.3절의 '약관 문구와 md5 를 메타에 적는다' 는 `SOURCE.md` 와 `download_manifest.json` 에 적었고, 2.7절의 `xg_meta.json` 은 추출 스크립트가 이 기록을 옮겨 만든다.

## 2. 추출 단계가 다룰 형식 차이

| 제품 | 좌표계 | 해상도 | 단위 | 결측 값 | 시간 |
|---|---|---|---|---|---|
| CCI v5 | 위경도 WGS84, GeoTransform −180 0.01 0 90 0 −0.01, lat 6,000 × lon 36,000 | 0.01° | m(uint16 × 0.01) | 65535(척도 적용 전) | 연별 파일 1997-2023 |
| Wei v2 | EPSG:3995(Arctic Polar Stereographic), 13,639 × 10,287 | 1,000 m | cm(단위 태그 없음, 근거는 SOURCE.md) | −9999.0 | 연별 tif 2000-2024 |
| Aalto 2018 | EPSG:4326, 43,200 × 7,200, >30°N | 30 초각 | cm(Zenodo 설명문) | float32 최솟값 | 기준기 2000-2014 한 장(`ALT_Baseline.tif`) |
| ds1760 | Albers 등적 원추(사용 안내서 EPSG:3338), x 1,776 × y 1,650 | 1 km | m | −999.0 | 2001-2015 연별(time 15) |

- Wei 와 ds1760 은 투영 좌표계라서 셀 위경도를 변환한 뒤 최근접 화소를 골라야 한다. 계획서 2.7절 산출물의 pix_dist_m 은 투영 좌표에서 재는 것이 자연스럽다.
- CCI 와 ds1760 은 m 이므로 100 을 곱해 cm 로 맞춘다. Wei 와 Aalto 는 cm 그대로다.
- CCI 1997-2002 파일은 강제력이 다른 계열(ERA5 편향 보정, 파일 접두 ERA5_MODISLST_BIASCORRECTED)이다. 연도 정합 평균에 들어가는 연도가 이 범위면 표지를 남기는 것이 좋다(계획서의 규칙은 바꾸지 않는다).
- CCI 카탈로그 초록은 '북위 30° 이북'이라 적었고 파일 속성은 25-85°N 이다. 실제 유효 화소 범위는 추출 때 결측 표지로 드러난다.
- Wei·Aalto 의 tif 는 zip 안에서 deflate 압축이다. `/vsizip/` 임의 접근이 느리면 필요한 tif 만 작업 임시 폴더에 풀어 창 읽기를 한다(원 zip 은 유지).

## 3. 계획서와 조사 보고서 사이의 작은 차이(계획서가 우선)

- `alt_products.md` 8.2절은 Aalto 를 XB 민감도 판 후보로 제안했으나 계획서 2.2절 XB 후보 목록에는 Aalto 가 없다. 계획서대로 Aalto 는 XG 서술(XG-4, XG-6, XG-7)에만 쓴다.
- `alt_products.md` 3.1절의 'CCI v5 NetCDF 28개' 는 CEDA 카탈로그의 '28 Files' 를 옮긴 값이다. 실제 북반구 NetCDF 는 27개(1997-2023)이고 나머지 1개는 안내문(`00README_catalogue_and_licence.txt`)이다. `Antarctica/` 폴더는 받지 않았다.
- ds1760 의 CMR 약관 주소(https://science.nasa.gov/earth-science/earth-science-data/data-information-policy)는 2026-10-04 에 404 였다. 현행 NASA Earthdata 자료 이용 안내 쪽의 문구(CC0, 인용 권고)를 대신 적었다. 계획서의 'EOSDIS 자유 이용' 표기와 맞는다.

## 4. 받지 않은 것

- Wei 상대 불확도 zip 3개(합계 2,610,400,473 바이트): 계획서 2.7절이 받지 않기로 했다.
- Aalto `MAGT_Rasters.zip`(2,869,846,127 바이트): 어떤 키에도 쓰이지 않는다.
- Liu 2024, Li 2022: 계획서 0.3절 내려받기 목록에 없다. 2.7절에서 Liu 는 알래스카에서 쓰지 않고, XB 는 Liu·Li 를 쓰지 않는다.
- 막힌 항목: 없다. 네 제품 모두 받았다(ds1760 은 `~/.netrc` 의 Earthdata 계정으로 받았다).
