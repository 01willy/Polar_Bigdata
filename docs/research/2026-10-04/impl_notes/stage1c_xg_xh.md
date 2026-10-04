# 로컬 1c 실행 기록: XG(제품 비교)·XH(검증 사다리)

- 작성: 2026-10-04 21:39. 실행 위치 ROOT(`/home/willy010313/Polar_Bigdata`), HEAD d8a3758(T0 = 2678100 은 HEAD 의 조상, `git merge-base --is-ancestor` 확인). 커밋·push 는 하지 않았다.
- 근거: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 0.3(열람 규칙·출력 제한), 1절(로컬 자원), 2.7(XG), 2.8(XH), 5절 1c. 명령은 `impl_notes/x_product_comparison.md` 9.2 와 `impl_notes/x_validation_ladder.md` 9.2 의 '다음 단계 명령'을 그대로 썼다.
- 범위: XG 의 누설 마스크 표·관문 갱신·CCI v4 행 집계, XH 의 kNNDM 색인·예측 영역·세기·묶음 점검. XG 의 `--extract`(CCI v5·Wei·Aalto·ds1760 추출, 2단계)는 실행하지 않았다.
- 라벨 사용: 모든 단계가 라벨 값을 쓰지 않거나(마스크·색인·세기·묶음 점검) 기존 조각의 블록 SSE 를 다시 채점하는 집계(cci4 행)다. 집계 산출은 봉인 폴더에만 썼고 봉인 파일은 열지 않았다. 화면 출력은 모듈의 `restricted_output` 과 셸 필터를 함께 거쳤다.
- 모듈 변경: 없다. 실패(traceback)도 없었다. 동결 모듈·새 모듈의 sha256 앞 16자: `x_product_comparison.py` b9c01d8931d1e969, `x_validation_ladder.py` 2979b4d0e44c06ce, `xbatch_core.py` aba34a3346d86aeb, `src/polar/cv_schemes.py` e18b3604e324a8a7.
- 환경: python 3.9.18, scikit-learn 1.3.0, numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, pyproj 3.6.1, xarray 2023.6.0(kNNDM 메타의 deps 와 같다).

## 1. 실행 명령과 자원(시각순)

공통 접두: `CUDA_VISIBLE_DEVICES= nice -n 10 /usr/bin/time -v`. OMP_NUM_THREADS 는 표의 값. 시작 전 `free -g` 가용 메모리는 모두 43–44 GB(하한 30 GB 이상)여서 대기(60 s 간격, 최대 30 분)는 발생하지 않았다. 스왑은 28/28 GB 로 차 있었다(기존 기록과 같다). 무거운 작업은 동시 2개 이하로 유지했다(XH `--build-knndm` 배경 실행 중에 XG `--gate-only`·`--summarize-only` 를 차례로 전경 실행).

| 순서 | 묶음 | 명령(ROOT 기준) | OMP | 시작 | 종료 | 종료 코드 | 벽시계 | 최대 RSS | 가용 메모리(시작 전) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | XG | `python3 scripts/3_deep_learning/x_product_comparison.py --mask` | 2 | 21:27:06 | 21:27:09 | 0 | 3.5 s | 269 MB | 43 GB |
| 2 | XH | `python3 scripts/3_deep_learning/x_validation_ladder.py --build-knndm --threads 4`(배경) | 4 | 21:27:38 | 21:36:03 | 0 | 8 min 25 s | 706 MB | 43 GB |
| 3 | XG | `python3 scripts/3_deep_learning/x_product_comparison.py --gate-only` | 2 | 21:28:10 | 21:28:24 | 0 | 14.3 s | 301 MB | 44 GB |
| 4 | XG | `python3 scripts/3_deep_learning/x_product_comparison.py --summarize-only --allow-local --products cci4 2>&1 \| grep -v -E '판정\|verdict\|Δ\|delta\|rmse\|RMSE\|우세\|열세\|동등\|미결정\|지지\|기각'` | 2 | 21:29:01 | 21:29:13 | 0(python)·0(grep) | 11.6 s | 374 MB | 43 GB |
| 5 | XG | `python3 scripts/3_deep_learning/x_product_comparison.py --payload-check` | 1 | 21:31:01 | 21:31:03 | 1(설계상: 없는 입력 1개) | 1.8 s | 182 MB | 해당 없음(가벼움) |
| 6 | XH | `python3 scripts/3_deep_learning/x_validation_ladder.py --count-only` | 1 | 21:37:00 | 21:37:09 | 0 | 8.6 s | 223 MB | 43 GB |
| 7 | XH | `python3 scripts/3_deep_learning/x_validation_ladder.py --payload-check` | 1 | 21:37:23 | 21:37:25 | 0 | 1.7 s | 181 MB | 해당 없음(가벼움) |

- 메모리 상한: 두 모듈은 로컬(WF_RESCALE·LG_RESCALE 미설정)에서 허용 표지와 무관하게 상주 메모리 감시 스레드(`rss_watchdog`, 10 GB)를 건다. 모듈이 스스로 제한하므로 `prlimit --data` 는 쓰지 않았다(RLIMIT_AS 는 CatBoost 적합을 멈추게 한 기록이 있다. `x_product_comparison.md` 7절). 실제 최대 RSS 는 모두 1 GB 아래였다.
- GDAL: 추출을 돌리지 않아 rasterio 를 부르지 않았고 `data/raw/tmp_gdal` 은 생기지 않았다.
- 순서 5 의 종료 코드 1 은 `payload_check` 가 '없는 필수 입력'이 있을 때 내는 값이다(제품 값 표는 2단계 산출).
- 산출 경로: 이 기록의 모든 산출은 `data/processed/xbatch/XG_product_comparison/`, `data/processed/xbatch/XH_validation_ladder/` 안에만 있다. 같은 시간대에 저장소의 다른 경로(deck/, outputs/figures/, paper/, XE 모듈)가 바뀌었으나 병렬로 진행 중인 다른 작업의 것이며 이 단계와 무관하다.

## 2. XG 산출(비봉인 파일: sha256 전체, 행 수는 머리글 제외)

| 파일 | sha256 | 행 수 | 바이트 | 만든 단계 |
|---|---|---|---|---|
| `xg_leak_mask_v1.csv` | bb2af49a55142b73d9e5c39ddf4aa0ffd5e09ed3cdf08bb63cb81efcc6eabdf7 | 4,970 | 223,452 | 1 `--mask` |
| `xg_leak_cells_v1.csv` | e7e3ba1337dde038e779dc965748cb47f5ef38c33866f49429b4770a9d994383 | 105,108 | 2,998,444 | 1 `--mask` |
| `xg_meta.json`(mask 절만) | c1011ef36365cdff7960794d3661f04a4329bf8c9964fb004db3eff75cd02372 | 해당 없음 | 421 | 1 `--mask` |
| `xg_gate.csv` | 1f8090cf25dc3440d4cac40d440ecb9a03bbb152a7ab8159488cba87b7fe8a61 | 1,972 | 276,635 | 3 `--gate-only` |
| `xg_gate_meta.json` | 4c23af5bd65661c8b296d7f450da8af48e991432db4ba98f5065dbb3627d2a9c | 해당 없음 | 302 | 3 `--gate-only` |
| `xg_payload_manifest.json` | e95d31a799aaa1cac2682c17a1317aced8084b880825176a8a17a7545c7178bf | 해당 없음 | 12,459 | 5 `--payload-check` |

**마스크(단계 1, 좌표만)**: 학습 지점 Wei 195개(고유 좌표, `ALT_variables.xlsx`), CALM ∪ GTN-P 464개(Aalto 대리 마스크). 저장소 10개 × 제품 7종 × d {5, 25} km. 학습 지점이 없는 제품(cci4·cci5y·cci5s·yk)은 가린 블록 0 이다. 블록 수와 가린 블록 수(Wei 5 km / 25 km, Aalto 5 km / 25 km):

| 저장소 | 블록 수 | Wei 5 km | Wei 25 km | Aalto 5 km | Aalto 25 km |
|---|---|---|---|---|---|
| Lena\|x, Lena\|r | 20 | 1 | 2 | 4 | 7 |
| Canada\|x, Canada\|r | 39 | 19 | 19 | 24 | 24 |
| Russia_W\|x | 21 | 18 | 18 | 21 | 21 |
| Russia_E\|x | 21 | 15 | 17 | 21 | 21 |
| Alaska\|x, Alaska\|r | 74 | 46 | 51 | 47 | 51 |
| Russia_C~lgd\|x | 13 | 3 | 3 | 5 | 5 |
| Tibet_LGD\|x | 34 | 1 | 3 | 3 | 5 |

- 셀 거리 표(`xg_leak_cells_v1.csv`)는 학습 지점이 있는 제품(wei·weis·aalto) × 저장소 10개의 셀별 최근접 학습 지점 거리다. 5 km 안 셀 수(Wei): 레나 12/3,037, 캐나다 19/750, 알래스카 4,474/13,606, 러시아 W 26/31, 러시아 E 22/30, Russia_C 4/57, 티베트 1/132. 러시아 W·E 의 Aalto 대리 마스크는 모든 블록을 가린다(라벨이 모두 CALM 지점이라는 계획 2.7 문구와 맞는다).
- weis 는 wei 와 학습 지점이 같아 표 값이 같다.

**관문(단계 3)**: g1 1,300키 실패 0, g1b 650키 실패 0, g2 21키 실패 0(모두 통과, 2026-10-04 16:09 기록과 같다). g3 는 'md5 기록 없음'(추출 전이라 `xg_meta.json` 에 extract 절이 없다)으로 실패 1 로 적혔다. 이는 2단계 `--extract` 뒤 `--gate-only` 를 다시 돌려야 해소된다. 관문 기록(21:28:24)이 마스크 표(21:27:09)보다 새로워 집계 때 stale 이 아니었다.

**CCI v4 행 집계(단계 4, 봉인)**: 제품 값 표가 없어 모듈이 cci4(자료의 `cci_alt` 열)만 계산했다. 재표집 10,000회. 봉인 폴더 `data/processed/xbatch/XG_product_comparison/sealed/` 에 쓴 파일과 화면에 나온 행 수·sha256 앞 16자(파일은 열지 않았다):

| 봉인 파일 | 행 수(화면) | sha256 앞 16자(화면) |
|---|---|---|
| `xg_descriptive.csv` | 28 | 9aa4b64bc482a4ef |
| `xg_key_scores.csv` | 75 | f0cb62da631cd78d |
| `xg_product_key_notes.csv` | 109 | 43e55601ff2fae72 |
| `xg_summary_meta.json` | 12 | 14207b155ad5b979 |
| `README.txt`, `sealed_manifest.json` | `write_sealed` 가 쓴 보조 파일 | 열지 않음 |

- 화면 요약: '관문 기록 있음 · 모두 통과 False · 오래됨 False'. '모두 통과 False' 는 g3(md5 기록 없음)만의 영향이다.
- 확인 대비 6개(xg_tests)는 wei·cci5y 가 없어 만들어지지 않았다(모듈 설계대로). 2단계 뒤 전체 집계가 같은 이름으로 다시 쓴다.

**묶음 점검(단계 5)**: 필수 입력 3개 가운데 없음 1개(`xg_product_values_v1.csv`). 21:15 점검의 없음 3개에서 마스크 단계가 셀 거리 표와 `xg_meta.json` 을 만들어 1개로 줄었다. 묶음 정보 파일 82개 sha256, `git_uncommitted_paths` 31개(다른 작업의 미커밋 변경 포함).

## 3. XH 산출(비봉인 파일: sha256 전체)

| 파일 | sha256 | 행 수 | 바이트 | 만든 단계 |
|---|---|---|---|---|
| `xh_knndm_index_v1.csv` | a7738be429b4d7bfe85dd15f381554762cc400129c2c29f053b8ba2364ec25b0 | 54,429 | 933,787 | 2 `--build-knndm` |
| `xh_knndm_meta_v1.json` | 12cb854a810eb8e2d636c2c445c593980f74dd93a36c66fd7f1c158d59b22d99 | 항목 12 | 22,037 | 2 `--build-knndm` |
| `xh_pred_domain_v1.csv.gz` | 386c6e09c585ef815400a6376dc8418449470c2422b101784109e5ea6631cc60 | 322,671 | 947,708 | 2 `--build-knndm` |
| `xh_payload_manifest.json` | c0be8de4ab0c919e3b183842c28293e7945e859c77a5aa974d094d050e52c903 | 해당 없음 | 12,386(갱신) | 7 `--payload-check` |

**kNNDM 색인(단계 2)**: k 5, 후보 q 100(로그 간격, 중복 제거 뒤 알래스카 96·레나 93·캐나다 88개), maxp 0.5, seed = seed_of('xh','W1K',지역,변형,반복). 색인 행 54,429 = 알래스카 13,606 × 3 + 레나 3,037 × 3 + 캐나다 750 × 3 × 변형 2. 모든 (지역, 변형, 반복)에서 묶음 번호 0–4 가 5개 모두 있고 loc_id 중복 0 이다. 메타의 `index_sha256` 은 파일 해시와 같다.

| 지역 | 변형 | 투영 | 셀 수 | 예측 영역 점 | gij 중앙값(km) | 선택 q(반복 1·2·3) | W 선택(km, 반복 1·2·3) | W 무작위(km) | 묶음 크기(반복 1) | 1회 소요 |
|---|---|---|---|---|---|---|---|---|---|---|
| Alaska | 주 | EPSG:3338 | 13,606 | 20,000(m1 정의: 0.02° 세분 육지 698,774 · 영구동토 581,385 에서 seed 0 표본) | 81.6 | 15 · 15 · 17 | 31.4 · 31.4 · 31.4 | 103.9 | 4,723 / 4,161 / 2,013 / 1,742 / 967 | 132–136 s |
| Lena | 주 | LAEA(72.13°N, 127.81°E) | 3,037 | 53,011(`lena_grid_x25_v1` 전 점) | 30.7 | 7 · 6 · 6 | 13.0 · 15.5 · 15.5 | 39.0 | 895 / 1,312 / 747 / 80 / 3 | 21–22 s |
| Canada | 주(± 1°) | LAEA(65.17°N, 127.30°W) | 750 | 122,209(ERA5-Land 원 격자 육지 141,530 ∩ MAAT < 0) | 331.9 | 6 · 5 · 6 | 255.6 · 159.1 · 255.6 | 456–458 | 198 / 203 / 192 / 8 / 149 | 4 s |
| Canada | pm2(± 2°) | 같음 | 750 | 127,451(육지 151,298 ∩ MAAT < 0) | 331.3 | 6 · 13 · 5 | 267.8 · 273.7 · 270.9 | 471–472 | 198 / 203 / 192 / 8 / 149 | 4 s |

- 모든 항목에서 `clustered` 참, `random_would_match_better` 거짓(무작위 묶음이 예측 영역 거리 분포에 더 가깝지 않다).
- 구조 관찰(라벨 무관): 묶음 크기가 매우 작은 경우가 있다. 레나 반복 1 의 다섯째 묶음 3셀(채점 3), 캐나다 주 반복 2 의 셋째·다섯째 묶음 5·3셀(채점 4·2), 캐나다 주 반복 1·3 과 pm2 반복 1 의 넷째 묶음 8셀(채점 6). kNNDM 의 k-평균 병합이 만든 결과이며 계획 2.8 의 정의대로 둔다. 집계는 블록 SSE 합으로 하므로 묶음 크기가 집계 가중에 직접 들어가지는 않는다. 서술 때 참고한다.
- 캐나다 예측 영역 계산 중 `cv_schemes.py:199` 의 `RuntimeWarning: Mean of empty slice` 2회(상자 안 바다 격자의 전체 결측 평균). 결과에 영향 없음.

**세기(단계 6)**: 단위 42개, 적합 2,514건(색인 있음), 추정 0.63 워커·시간(catboost·rf 는 h41 가정값). 21:15 의 1,764건에 W1K 12단위 750건(알래스카 70 × 3 + 레나 60 × 3 + 캐나다 60 × 6)이 더해져 `x_validation_ladder.md` 2절의 적합 수와 같다. 캐나다 W1B 반복 1 의 학습 519·채점 231 은 그대로다.

**묶음 점검(단계 7)**: 필수 입력 3개 가운데 없음 0개. 묶음 정보 파일 87개 sha256(21:15 의 82개 + 1c 산출 3개 + XE·XB 쪽 신규 입력), git d8a3758, plan_commit 2678100, `git_uncommitted_paths` 32개.

## 4. 남은 일

**R1b(XH, 계획 5절 2a)**
1. 색인 sha256 `a7738be429b4d7bfe85dd15f381554762cc400129c2c29f053b8ba2364ec25b0` 을 계획 개정 이력(또는 묶음 tar 의 sha256 기록)에 적는다(계획 1절 '묶음과 코드 출처'. 이 기록은 그 전 단계다).
2. 묶음(payload)에 `xh_knndm_index_v1.csv`, `xh_knndm_meta_v1.json`, `xh_pred_domain_v1.csv.gz` 를 넣는다(`PAYLOAD_EXTRA`, 점검 통과). 제출 전에 '커밋 뒤 제출' 또는 '묶음 tar 와 sha256 보관' 가운데 하나를 지킨다.
3. R0 스모크 뒤 R1b 제출(XB 핵심, XH, XE xh0·xt2 와 함께. XE 의 `xe_feat_v1.csv` 는 이 기록의 범위 밖이다). Rescale 실행은 `--knndm-sha256` 또는 메타의 값과 색인을 대조한 뒤 적합한다.
4. 회수 뒤 로컬에서 `--gate-only`(관문 (1) W1B 대 h41 W1, (2) M1C 대 m1 표)를 본 다음 봉인 표를 열람 순서 2 에 따라 연다.

**R3(XG, 계획 5절 '병렬'·3)**
1. 2단계 추출: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --extract --threads 4`. 제품 4종은 이미 `data/raw/{cci_alt_v5, wei2026_alt_v2, aalto2018_alt, ornl_ds1760}` 에 있다(`xg_downloads.md`). T0 조상 조건은 충족한다. Wei 단위가 값 범위로 cm 가 아니면 `--units wei=cm`. GDAL 캐시 512 MB·`CPL_TMPDIR=data/raw/tmp_gdal` 은 모듈이 설정한다. Aalto zip 안 deflate 읽기가 느리면 `data/raw/aalto2018_alt/` 아래에 푼다.
2. `--gate-only`(g3 md5 대조 포함, 추출 기록보다 새로워야 한다) → `--payload-check`(없음 0개여야 R3 제출) → 전체 집계 `--summarize-only --allow-local`(단계 4 와 같은 셸 필터, 봉인 표를 같은 이름으로 다시 쓴다).
3. R3 전 스모크(XG 셀 단위 경로) → R3 제출(셀 단위 5 km 민감도, 레나·캐나다 모드 x, 적합 280건). 묶음에 `xg_product_values_v1.csv`, `xg_leak_cells_v1.csv`, `xg_meta.json` 이 들어가야 한다.
4. 판정 표(xg_tests)는 R3 회수와 재현 관문 확인 뒤 열람 순서에 따라 연다. 모든 판정 행에 '등록 이탈(WRAPUP 10)' 표지가 붙는다(모듈 설정).

## 5. 로그 보관

- 실행 로그와 `/usr/bin/time -v` 기록은 세션 scratchpad(`/tmp/claude-1025/-home-willy010313-Polar-Bigdata/8d61e134-5378-44fd-9785-fd61ec207ca1/scratchpad/logs/`)에 두었다. 저장소 안에는 두 산출 폴더 밖에 아무것도 쓰지 않았다. 집계 로그는 셸 필터를 거친 판이다.
