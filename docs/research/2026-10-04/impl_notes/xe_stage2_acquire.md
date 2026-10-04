# XE 2단계 취득 구현 기록(xe_stage2_acquire)

- 작성: 2026-10-04, 모듈 `scripts/0_download/xe_stage2_acquire.py`
- 근거: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 2.5절(개정 1, T0 = git 2678100, 2026-10-04 14:22:39 +0900), `docs/research/2026-10-04/within_grid_inputs.md` 4·5절
- 이 문서는 구현 중 확인한 불일치와 가정을 적는다. 등록 문서는 고치지 않았다. 판정·가설·결정 규칙은 바꾸지 않았다.
- 라벨 값은 읽지 않았다. 라벨 표에서는 `loc_id, lat, lon, region` 열만 읽었다(시험 (e)의 조건).

## 1. 작업 지시와 등록 문서 사이의 불일치

| 번호 | 작업 지시 | 등록 문서 2.5절 | 구현(가장 문자 그대로의 읽기) |
|---|---|---|---|
| 1 | SoilGrids 부족 창을 WCS 로 취득 | O 군 주 경로는 'ISRIC SoilGrids VRT 창 읽기(부족 창)', 대체 경로 없음 | VRT 창 읽기로 받았다. WCS(SCALESIZE 없는 GetCoverage)는 같은 250 m 값을 주지만 등록 경로가 아니므로 쓰지 않았다 |
| 2 | AppEEARS 시험 요청은 MOD13Q1·MOD10A2 | S 군 주 경로는 AppEEARS MOD10A1 점 요청, 대체 순서는 earthaccess MOD10A1 → MOD10A2 | 시험 요청에 MOD10A1 을 더해 제품별 1건, 3건을 냈다(MOD13Q1, MOD10A1, MOD10A2). 주 경로와 대체 경로의 대기 시간을 함께 잰다. 전체 요청 JSON 은 MOD10A1(주)과 MOD10A2(대체 준비) 둘 다 만들었다. 등록 문서는 MOD10A2 의 취득 경로를 정하지 않았으므로 AppEEARS MOD10A2 JSON 은 준비물일 뿐이다 |
| 3(저녁 세션) | 전체 요청은 MOD13Q1·MOD10A2 를 라벨 위치와 레나 격자 중심에 제출 | S 군 주 경로는 MOD10A1 | 지시대로 MOD13Q1·MOD10A2 를 두 집합에 냈고, 등록 주 경로를 지키기 위해 라벨 위치 MOD10A1(1,234화소, 출력 약 0.5 GB 추정)도 함께 냈다. 레나 육지 MOD10A1(출력 약 16 GB 추정)은 내지 않았다. 지도는 해석 규칙 1·2 에서만 다시 만들고(within_grid_inputs.md 5.9) 그때 MOD10A2 대체 정의(등록 문서 2.5절에 고정)를 쓴다. 이 선택이 등록 이탈인지는 추출 단계에서 S 군 정의를 고정할 때 개정 이력에 적는다 |
| 4(저녁 세션) | WorldCover 는 v3·v4 의 모든 라벨 위치를 덮을 것 | V 군 대상은 XE 3지역 | 1차(14:36) 43타일은 XE 대상만 덮었다. 2차(21:24, `--scope all_labels`)에서 v3 ∪ v4 모든 지역 라벨 위치까지 넓혀 131타일로 완료했다. XE 추출은 1차 43타일만 쓰면 되고, 나머지 89타일은 XF 등 다른 실험의 참고용이다. 라벨 값은 읽지 않았다(좌표 열만) |
| 5(저녁 세션) | SoilGrids 250 m 부족 창을 `data/raw/soilgrids_wcs` 와 비교 | O 군 출처는 `data/raw/soilgrids_multi/`(250 m 창) + 부족 창 | `soilgrids_wcs/` 는 x25 의 약 5 km WCS 추출(지역 상자 31개, 9층)이며 250 m 창이 아니다. 부족 창 판정의 기준은 등록 문서대로 `soilgrids_multi/` 로 두었다(3절). 기준 차이만 기록한다 |

## 2. 등록 문서 안의 미정 사항과 이번 구현의 가정

1. **S 군 '2015–2020 평균'의 수문년 범위**: 9–8월 수문년 6개가 2014/15–2019/20 인지 2015/16–2020/21 인지 정해져 있지 않다. 요청은 2014-09-01 – 2021-08-31(수문년 7개)로 두 읽기를 모두 덮는다. 어느 6개를 쓸지는 추출 단계(`xe_point_covariates.py`)에서 라벨 결합 전에 고정해야 한다.
2. **V 군 '6–8월'의 MOD13Q1 합성 기준**: 요청은 2015–2020 의 06-01 – 08-31 반복 구간이다. 시험 요청(06-01 – 06-30)에서 AppEEARS 는 요청 구간과 겹치는 합성을 모두 돌려주었다(MOD13Q1 2018-05-25·06-10·06-26 시작 합성, MOD10A2 05-25 – 06-26 시작 합성 5개). 따라서 전체 요청은 5월 말 시작 합성과 8월 말 시작 합성을 포함하는 상위 집합이다. `_250m_16_days_composite_day_of_the_year` 층을 함께 받아 추출 단계가 합성 시작일 기준 또는 화소 관측일 기준으로 '6–8월'을 정할 수 있게 했다. 정의는 추출 단계에서 라벨 결합 전에 고정한다.
3. **AppEEARS 점 수 상한**: API 문서와 검색에서 상한을 찾지 못했다[미확인]. 한 요청 1,000점으로 나눴다(가정). 라벨 위치는 MOD13Q1 3건(2,080 화소), MOD10A1 2건(1,234 화소), MOD10A2 2건이다.
4. **레나 지도 격자의 범위**: 마스크 `land == 1` 인 38,173셀을 썼다. within_grid_inputs.md 5.9 의 '육지 38,086셀'은 회색 87셀을 뺀 표시 셀 수다(WRAPUP 6.5). 육지 전체가 표시 셀을 포함한다.
5. **레나 S 군 점 요청의 규모**: MOD10A1 은 38,173점 × 2,557일 × 3층이다. AppEEARS 점 출력 CSV 로는 수십 GB 규모가 될 수 있다[추정]. 레나 상자 면적 요청(GeoTIFF, 원 투영) JSON 을 `data/raw/appeears_xe/requests/alternative_area/` 에 참고안으로 두었다. 이 안은 등록 경로가 아니다. 지도 재작성은 해석 규칙 1·2 일 때만 하므로(within_grid_inputs.md 5.9) 라벨 위치 요청을 먼저 낸다.
6. **WorldCover 범위**: XE 대상(알래스카·레나·캐나다, `fidelity_base_v3.csv` 17,385행)과 레나 지도 상자만 덮었다. XE 밖 지역 라벨은 넣지 않았다.
7. **CAVM 판**: Mendeley 판 2(2022-01-17)를 주 판으로 두었다. 판 1 의 `raster_cavm_v1.tif` 와 sha256 이 같고, 판 2 는 범례 CSV 를 더했다.

## 3. 기존 SoilGrids 창의 위치 어긋남(새로 확인한 결함)

- `data/raw/soilgrids_multi/` 의 창은 `from_bounds` 가 준 소수 창으로 읽고 `window_transform(소수 창)` 을 저장했다(`scripts/0_download/soilgrids_multiregion.py`). 저장 변환의 원점은 원격 VRT 격자에서 정수 화소가 아니다(예: `lon-164_lat+60` 원점은 VRT 화소 (18616.967, 6480.364)).
- 원격 VRT 를 정수 창으로 읽어 30 × 30 화소를 맞춰 보면, 창마다 일치하는 이동량이 반올림·내림 어느 한 규칙으로 정해지지 않았다(15창 가운데 반올림과 일치 7(1창은 내림과도 일치), 내림만 일치 1, 둘 다 불일치 4, 유효값이 적어 판단 불가 3). 읽기 단계에서 소수 창이 재표집되어 자료가 저장 변환과 최대 0.5 화소(125 m) 어긋난다고 본다.
- 영향: 기존 창 경계 안이지만 7층이 무효로 나온 라벨 점 1,283개(알래스카 521, 캐나다 10, 레나 752) 가운데 원격 VRT 에서는 soc 0–5 cm 가 양수인 점이 315개였다(알래스카 188, 캐나다 9, 레나 118).
- 조치: 이번 취득은 정수 화소 정렬 창(`_aligned_window`)으로 받는다. 부족 창 36칸(등록 문서의 문자 그대로의 대상) 외에, 기존 창이 덮는 21칸도 같은 제품을 정렬 창으로 다시 받아 `data/raw/soilgrids_xe250/` 에 두었다. 기존 폴더는 고치지 않았다. 정렬 재취득 칸은 `windows_meta.json` 의 `kind = reacquire_aligned` 로 구분한다.
- 추출 단계 권고: XE 의 `sg250_*` 는 `soilgrids_xe250/` 의 정렬 창만으로 추출하는 것이 위치 오차 흡수 원칙(2.5절 결합 규칙)에 맞다. 기존 창을 쓰는 판단은 추출 모듈이 하고, 어느 쪽이든 라벨 결합 전에 고정한다.
- 이 결함은 SoilGrids 원해상도 창을 쓴 기존 산출(`scripts/1_data_prep/enrich_soilgrids_cell.py`, `scripts/4_visualization/soilgrids_gate_figs.py`)에도 해당할 수 있다. x25 의 `sg_*` 는 WCS 약 5 km 추출이라 해당하지 않는다. 이번 작업 범위 밖이므로 기록만 한다.

## 4. 취득 뒤 확인한 범위(라벨 값 없이 좌표만 사용, 등록 전 허용 확인의 유한값 비율)

| 항목 | 알래스카 | 캐나다 | 레나 라벨 | 레나 지도 육지 |
|---|---|---|---|---|
| NCSCD SOCC30·SOCC100 유효 비율 | 1.000 | 1.000 | 0.408 | 0.918 |
| CAVM 식생 등급(1–43) 비율 | 0.931 | 0.004 | 0.715 | 0.853 |
| CAVM 수면(91·92) 비율 | 0.021 | 0.000 | 0.285 | 0.059 |
| CAVM 비북극 육지(99) 비율 | 0.048 | 0.996 | 0.000 | 0.088 |

- 레나 라벨의 NCSCD 무효(3,037점 가운데 1,797점)는 제품 쪽 결측이다(1 km 화소 값이 nodata). 다시 받아도 채워지지 않는다. O 군의 레나 유한값 비율은 SoilGrids 정렬 창 결과와 함께 추출 단계에서 90 % 규칙으로 판단한다.
- CAVM 은 캐나다 라벨(ABoVE_CA 북방림)에서 거의 모두 99(비북극 육지)다. 결측은 아니지만 정보가 없다. within_grid_inputs.md 5.10 의 '툰드라만 분류' 한계와 같다.
- SoilGrids 정렬 창 포함률은 `data/raw/soilgrids_xe250/coverage_meta.json` 에 적는다(취득 완료 뒤).

## 5. 마감(2026-10-04 21:30 +0900 갱신)

T0 = 2026-10-04 14:22:39 +0900. 마감 A = T0 + 72 h = 2026-10-07 14:22:39, 마감 B = T0 + 96 h = 2026-10-08 14:22:39, xe_feat 최종판 = T0 + 100 h = 2026-10-08 18:22:39.

| 군 | 마감 | 상태 | 완료 시각(T0 기준) | 마감까지 남은 시간 |
|---|---|---|---|---|
| V(WorldCover) | A | 완료. 1차 43타일(XE 대상+레나 상자), 2차 131타일(v3 ∪ v4 모든 라벨 위치, 7.45 GB). 제품 범위 밖 1타일(S63W063, 남극 GTN-P 1점) | 1차 14:36:50(T0 + 0.24 h), 2차 21:24:40(T0 + 7.03 h) | 65.0 h |
| V(CAVM) | A | 완료(판 2 주, 판 1 대조, zip sha256 = Mendeley API) | 14:33(T0 + 0.17 h) | 71.8 h |
| O(NCSCD) | A | 완료(0.012° 래스터 zip 97.6 MB, 73파일 해제) | 14:35(T0 + 0.21 h) | 71.8 h |
| O(SoilGrids 250 m 창) | A | 완료. 57칸 × 7층 모두 ok(부족 창 36 + 정렬 재취득 21). XE 라벨 17,385점·레나 육지 38,173셀 모두 창 경계 안 | 16:48:55(T0 + 2.44 h) | 69.6 h |
| V(MOD13Q1) | B | 제출 완료, 처리 대기. 라벨 위치 3건·레나 육지 39건(task_id 는 `appeears_xe/requests/submitted_full.json`) | 제출 21:11:01–21:13:34(T0 + 6.8 h) | 89.0 h |
| S(MOD10A1 주 경로, 라벨 위치) | B | 제출 완료, 처리 대기. 4건(1,000점 요청 1건이 값 수 상한으로 거절되어 3조각으로 재제출, 8절) | 제출 21:11:13–21:21:44(T0 + 7.0 h) | 89.0 h |
| S(MOD10A2 대체 준비) | B | 제출 완료, 처리 대기. 라벨 위치 2건·레나 육지 39건 | 제출 21:11:16–21:15:48(T0 + 6.9 h) | 89.0 h |
| S(MOD10A1, 레나 육지) | B | 미제출(출력 약 16 GB 추정, 지도용은 해석 규칙 1·2 에서만 필요). JSON 39건 보존 | | |
| M(ArcticDEM 10 m) | A | 이 세션 범위 밖. 디스크 상태만 기록: 색인 JSON(10 m·32 m) 과 스모크 창 8개(10 m ok 6, 타일 없음 2, 1.1 MB), SOURCE.md 없음(`arcticdem_xe/fetch_meta.json`) | 15:58(스모크) | 65 h |

상태 조회 1회(21:23:08, T0 + 7.01 h): 87건 가운데 pending 86, processing 1(`appeears_xe/requests/status_summary.json`, `status_log_full.csv`). 시험 요청에서 대기열 약 16분·처리 2–4분이었으므로 87건이 순차 처리되면 수 시간이 걸릴 수 있다[추정]. 결과 내려받기는 `appeears-status --fetch`(9절).

## 6. AppEEARS 시험 요청 결과(대기 시간)

- 요청: 10점(알래스카 4, 레나 3, 캐나다 3, 서로 다른 250 m·500 m 화소, seed 20261004), 2018-06-01 – 2018-06-30, 제품별 1건. 2026-10-04 14:36:36–38 +0900 제출.
- 상태 전이: pending → queued(약 2–3분 뒤) → processing(제출 뒤 15.7분 이하, 60초 간격 기록) → done.
- 제출부터 완료(API `completed`)까지: MOD13Q1 17.4분, MOD10A2 18.3분, MOD10A1 19.8분. 대부분이 대기열 시간이고 처리 시간은 2–4분이었다.
- 결과 행 수: MOD13Q1 30행(10점 × 합성 3), MOD10A2 50행(10점 × 합성 5), MOD10A1 300행(10점 × 30일). 행당 크기는 MOD13Q1 약 440 B, MOD10A2 약 84 B, MOD10A1 약 168 B 다.
- 전체 요청 출력 크기 추정(행당 크기 × 점 × 날짜 수)[추정]: 라벨 위치 MOD10A1 약 0.5 GB, MOD13Q1 약 30 MB, MOD10A2 약 30 MB. 레나 육지 MOD10A1 약 16 GB, MOD13Q1 약 0.6 GB, MOD10A2 약 1 GB. 처리 시간은 점 × 날짜 수에 따라 늘 것이므로 시험 결과만으로는 추정하지 못했다[미확인].
- 제출 권고 순서(마감 T0 + 96 h 안): 라벨 위치 MOD13Q1·MOD10A1(주 경로) → 라벨 위치 MOD10A2(대체 대비) → 레나 지도용 요청(해석 규칙 1·2 에서만 쓰므로 마지막).

- 확인(21:10, 저녁 세션): API 재조회에서 3건 모두 `done`, 묶음 5파일씩(결과 CSV, granule 목록, metadata XML, request JSON, README) 디스크에 있고 `result_*.json` 의 sha256 과 같다. 계정에는 이 3건 외 과제가 없었다.

## 7. 저녁 세션(2026-10-04 20:58–21:30 +0900) 디스크 재고 조사

앞 세션은 세션 한도로 중단됐다. 다시 시작하며 `data/raw/` 아래 2단계 폴더를 조사했다. 체크섬 검증은 기존 SOURCE.md 표의 sha256 과 크기를 파일마다 다시 계산해 비교한 것이다.

| 폴더 | 파일 | 크기 | SOURCE.md | 체크섬 검증(조사 시점) | 판정 |
|---|---:|---:|---|---|---|
| `worldcover_v200/` | 49(타일 43) | 2.46 GB | 있음(14:58) | 타일 43 sha256 일치(2차 실행의 cached 재해시) | XE 범위 완료, 전 라벨 범위는 미완(7절 뒤 2차 실행) |
| `cavm_raster/` | 19 | 310 MB | 있음(14:58) | 19/19 일치 | 완료, 변경 없음 |
| `ncscd_v2/` | 73 | 187 MB | 있음(14:58) | 73/73 일치 | 완료. SOURCE.md 약관 문구만 보강(인용 요구 문장 인용) |
| `soilgrids_xe250/` | 404(57칸 × 7층 + 메타 4) | 341 MB | 있음이나 낡음(14:58, 4칸 32행만) | 표의 자료 파일 30/30 일치, 로그·메타 2건은 갱신됨 | 취득은 16:48 완료였으나 SOURCE.md 가 중간 상태. 다시 썼다(403행) |
| `appeears_xe/` | 143 | 20 MB | 있음(14:58) | 159/159 일치 | 시험 요청 완료, 전체 요청 미제출 → 8절 |
| `arcticdem_xe/` | 11 | 1.1 MB | 없음 | (표 없음) | M 군 스모크만. 이 세션 범위 밖 |

자원: `/` 97 %(여유 59 GB), `/home` 96 %(여유 583 GB), RAM 251 GB 가운데 가용 43 GB·스왑 0 여유. 모든 산출은 `/home` 의 `data/raw/<source>/` 에만 썼고, 압축 해제·대용량 처리는 하지 않았다(WorldCover 는 스트리밍 저장, SoilGrids 는 창 단위). 메모리 사용은 10 GB 미만이다.

## 8. 저녁 세션에서 한 작업

1. **WorldCover 2차 범위**(`worldcover --scope all_labels`): `fidelity_base_v3.csv` ∪ `fidelity_base_v4.csv` 의 좌표 열(loc_id, lat, lon, region)만 읽어 모든 지역 라벨 18,088점(v4 가 v3 를 포함, 좌표 동일)을 덮는 타일을 구했다. 필요 132타일(라벨 131 + 레나 상자 포함), 격자에 없음 1(S63W063), 디스크에 없음 88. 88타일을 받아 131/131 로 끝냈다(7.45 GB, 그 가운데 XE 밖 지역 전용 89타일 4.99 GB). 1회차 실행은 86타일 뒤 HEAD 요청의 원격 연결 끊김 예외로 중단됐고(manifest 미작성), HEAD 재시도와 타일별 예외 격리를 더한 2회차가 남은 2타일을 받고 129타일을 재해시했다. `manifest_tiles.csv` 의 `used_by` 로 XE 용(`labels_<대상>`, `lena_box`)과 그 밖(`v3:<region>`, `v4:<region>`)을 구분한다. `download_meta.json` 은 `history` 에 1차 메타를 남겼다.
2. **SoilGrids 250 m**: 취득 완료 확인(57칸 × 7층 ok, `windows_meta.json`). 포함률(`coverage_meta.json`, 좌표만 사용): 창 경계 안 비율은 네 집합 모두 100 %, 7층 유효 비율은 알래스카 13,123/13,606, 캐나다 692/742, 레나 라벨 2,099/3,037, 레나 육지 33,079/38,173. 90 % 규칙 판단은 추출 단계의 몫이다. v3 ∪ v4 의 XE 밖 라벨 703점은 250 m 창이 없다(2° 칸 127개, 기존 칸과 겹침 18. 칸당 약 2.3분이면 약 5 h[추정]). O 군 범위가 XE 3지역이므로 받지 않았다.
3. **AppEEARS 전체 요청 제출**(`appeears-submit`, 21:10:58–21:21:46): 준비된 JSON 85건을 라벨 MOD13Q1 → 라벨 MOD10A1 → 라벨 MOD10A2 → 레나 MOD13Q1 → 레나 MOD10A2 순으로 냈다. 84건 HTTP 202, 1건 HTTP 400. 거절 사유는 요청이 만드는 값의 총수 상한이다: `xe_mod10a1_labels_p01of02`(1,000점 × 2,557일 × 3층 = 7,671,000값)에 'The total number of values that this request will generate exceeds the maximum allowed by 119.2%' → 상한 약 3.5 × 10⁶ 값(역산). 이 상한은 API 문서·LP DAAC 안내에 없다. 다른 요청 종류는 MOD13Q1 약 2.9 × 10⁵, MOD10A2 약 6.4 × 10⁵, MOD10A1 234점 약 1.8 × 10⁶ 값으로 상한 아래다. 거절 요청은 같은 좌표 순서로 3조각(333·333·334점, 조각당 약 2.6 × 10⁶ 값)으로 나눠(`appeears-split`) 다시 냈고 모두 202 였다(21:21:38–21:21:44). 원 요청 행은 `requests_index.csv` 의 `superseded_by` 로 표시했다. 합계 87건 수락.
4. **상태 조회 1회**(`appeears-status`, 21:22:06–21:23:08): pending 86, processing 1(`xe_mod13q1_labels_p01of03`). 기록은 `requests/status_log_full.csv`, `requests/status_summary.json`.
5. **SOURCE.md 갱신**: `worldcover_v200`(136파일), `soilgrids_xe250`(403파일), `ncscd_v2`(약관 문구 보강), `appeears_xe`(167파일, 제출·상한 기록) 를 `source-md --only` 로 다시 썼다. `cavm_raster` 는 바뀐 것이 없어 두었다.
6. **코드**(`scripts/0_download/xe_stage2_acquire.py`): `hashes_file`(sha256·md5 한 번에), `all_label_points`, `worldcover --scope/--dry-run`, HEAD 재시도·예외 격리, `appeears-submit`(제출마다 저장, 재실행 시 건너뜀), `appeears-split`, `appeears-status [--fetch]`, `source-md --only`. 라벨 값 열은 어디서도 읽지 않는다.

## 9. 남은 작업(다음 세션)

- AppEEARS 처리 대기: `python3 scripts/0_download/xe_stage2_acquire.py appeears-status --fetch` 를 주기적으로 실행해 `done` 과제의 결과 묶음을 `data/raw/appeears_xe/results/<task_name>/` 에 받는다(API 가 준 sha256 과 대조, `bundle_meta.json`). 마감 B(2026-10-08 14:22:39 +0900)까지 끝나지 않은 과제는 등록 문서 2.5절의 대체 경로 순서(earthaccess → MOD10A2 → 군 제외)로 간다. 모두 받은 뒤 `source-md --only appeears_xe` 로 체크섬 표를 다시 쓴다.
- 결과 묶음을 받은 뒤 추출 단계(`xe_point_covariates.py`)에서 S 군 수문년 범위, V 군 6–8월 합성 기준, MOD10A1 대 MOD10A2 선택을 라벨 결합 전에 고정하고, 선택 결과를 개정 이력에 적는다(2절 1·2 항, 1절 3 항).
- M 군(ArcticDEM 10 m)은 스모크 창 8개 상태다. 마감 A 까지 본 취득이 필요하다(이 세션 범위 밖).
- 이 세션은 git 커밋을 하지 않았다. 바뀐 파일: 이 문서, `scripts/0_download/xe_stage2_acquire.py`, `data/raw/{worldcover_v200,soilgrids_xe250,ncscd_v2,appeears_xe}/SOURCE.md`, `worldcover_v200/{manifest_tiles.csv,download_meta.json,download.log,tiles/*(88타일)}`, `appeears_xe/requests/{submitted_full.json,requests_index.csv,requests_meta.json,status_log_full.csv,status_summary.json,submit_full.log,status_poll.log,xe_mod10a1_labels_p01of02_s[1-3]of3.json}`.
