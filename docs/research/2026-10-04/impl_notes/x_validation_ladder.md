# x_validation_ladder 구현 기록(XH, 계획 2.8)

- 대상: `scripts/3_deep_learning/x_validation_ladder.py`, `src/polar/cv_schemes.py`, `tests/test_x_xg_xh.py`(XH 시험 (k)–(s))
- 근거: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 2.8, 1절, 0.3, `docs/research/2026-10-04/harness_implementation_plan.md` 3.4·4.8, `scripts/2_evaluation/h41_validation_ladder.py`(동결), `scripts/3_deep_learning/m1_cv_scheme_comparison.py`
- 작성: 2026-10-04. 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리다.
- 이 기록을 쓰면서 한 계산: 합성 시험, `--count-only`, 제한 스모크(캐나다, W1R·W1B, 반복 1, seed 1)와 그 조각의 관문 (1). kNNDM 색인은 만들지 않았다(로컬 1c 단계).

## 1. 파일 위치

| 계획 문구 | 처리 |
|---|---|
| `scripts/2_evaluation/x_validation_ladder_regional.py`, `tests/test_x_ladder.py` | 작업 배정에 따라 `scripts/3_deep_learning/x_validation_ladder.py` 와 `tests/test_x_xg_xh.py` 로 두었다. 계획이 정한 시험(묶음이 겹치지 않고 셀을 덮는다, cv_schemes.knndm 이 m1 함수와 같은 입력에서 같은 묶음, W1K 학습·채점 교집합 0, 예측 영역 결정성, 색인 파일 해시 대조)을 모두 넣었다 |
| `src/polar/cv_schemes.py` | 계획대로 두었다. m1 의 함수 본문을 바꾸지 않고 옮겼고, 예측 영역 함수에 상자·파일 인자만 더했다 |

## 2. 단과 묶음

| 항목 | 처리 | 근거 |
|---|---|---|
| 묶음을 나누는 셀 | 대상 지역의 라벨 있는 전 셀(h40 target_idx). 학습 = 학습 묶음 ∩ 학습 가능, 채점 = 채점 묶음 ∩ eval_mask | h41 W1 규칙(unit_cells 의 지역 내 단) |
| W1R·W1S seed | seed_of('xh', 단, 지역, 반복). h41 V-R·V-S 는 지역을 섞은 통합 자료 단이라 seed 를 따로 정했다 | 계획 미기재 |
| W1B | h41 W1 과 같은 묶음(seed_of('lgv','W',지역,반복))이라 반복 1–3 이 h41 W1 조각과 같은 셀·같은 모형이다(관문 (1)) | 계획 '재현용으로 다시 실행' |
| 반복과 seed | 반복 3(1–3), CatBoost seed 0·1 | 1절 '추출·seed' 기본값. h41 W1 조각의 seed 도 0·1 이다 |
| 모형 | PSw, D0w, RSw(λ 0.25·1.0). h41 models_pooled 의 F1kw·V2w 는 등록되지 않아 적합하지 않는다. 식은 models_pooled 그대로이고 시험 (o)가 같은 예측을 확인한다 | 계획 2.8 '모형' |
| 알래스카 대회 구성 | W1 단마다 직접 능형(D0w[ridge])과 RSw[ridge, λ 0.75](m1 의 stefan_ridge_l075 식, fold_prep + Ridge α 10)를 더한다. 관문 (2)용 M1C 단위는 m1 정의(6겹, seed 0–2, random_cell·site_0.05·block_0.5, 모형 4종)를 그대로 다시 돈다 | 계획 2.8 '연속성 확인용', 재현 관문 (2) |
| 적합 수 | 2,160(3지역 × 4단 × 반복 3 × 5겹 × 12) + 캐나다 pm2 180 + 알래스카 능형 120 + M1C 54. 계획의 '약 810건'은 seed 1·새 단 3개 기준의 근사로 읽었다. 판정이 없는 서술 실험이라 비용만 바뀐다. 세기 추정(h41 가정값) 0.47 워커·시간(W1K 제외), W1K 를 더하면 약 0.65 워커·시간 | 계획 2.8 '약 0.5–1 워커·시간' 범위 안 |

## 3. kNNDM

| 항목 | 처리 | 근거 |
|---|---|---|
| 만드는 곳 | 로컬 1c 에서 `--build-knndm` 으로 한 번 만들고 색인(`xh_knndm_index_v1.csv`), 예측 영역(`xh_pred_domain_v1.csv.gz`), 메타(`xh_knndm_meta_v1.json`, 색인 sha256·scikit-learn 판·선택 q·W)를 쓴다. 실행은 색인 sha256 을 메타 또는 `--knndm-sha256` 과 대조하고 다르면 중단한다 | 1절 재현 관문 행 |
| 매개변수 | k 5, 후보 수 100(로그 간격), maxp 0.5, k-평균 n_init 1·max_iter 50(m1 값). seed = seed_of('xh','W1K',지역,변형,반복) | m1 정의, 계획 '5겹' |
| 투영 | 알래스카는 m1 과 같은 EPSG:3338. 레나·캐나다는 라벨 중심의 람베르트 등적 방위 투영 | 계획 미기재. EPSG:3338 은 레나에서 쓸 수 없다 |
| 예측 영역 | 알래스카 = m1 정의(0.02° 세분, 20,000점, seed 0). 레나 = 격자 파일의 전 점(53,011점). 캐나다 = 라벨 경계 ± 1°(주)·± 2°(민감도 변형 pm2) 상자의 ERA5-Land 원 격자에서 육지 ∩ 다년평균 MAAT < 0 인 전 점. ERA5-Land 파일은 m1 과 같은 nh_monthly_2015-2020.nc | 계획 문구는 알래스카에만 '세분'과 '20,000점'을 적었다. 문자 그대로 레나·캐나다는 격자 전체를 쓴다 |

## 4. 집계(봉인)

- 셀 단위 제곱 오차의 반복·seed 평균은 블록 SSE 의 반복·seed 평균과 같다(반복마다 채점 셀을 한 번씩 예측). 그 블록 합으로 RMSE(셀 가중, 블록 등가중)를 낸다.
- 재표집: 지역마다 seed_of('xhboot', 지역)의 블록 다중도를 모든 단·모형·변형이 공유한다. ΔΔ 의 네 항은 같은 재표집 번호로 계산한다. 3지역 평균은 채점 블록 8개 이상 지역의 m1_stats.strat 이다.
- 표: xh_ladder(단·지역·모형별 RMSE 와 거리 중앙값), xh_dd(XH-1·XH-2, 학습기 3종, 캐나다 pm2 변형), xh_stage_contrasts(D0 − P1, R1 − P1), xh_reuse_h41(V-G 와 h41 W1 반복 5 행을 lgv_metrics.csv 에서 옮김). 모두 `data/processed/xbatch/XH_validation_ladder/sealed/` 에 쓴다(R1b 산출, 열람 순서 2).
- 판정어를 쓰지 않는다(계획 '다중성: 없음').

## 5. 재현 관문

| 관문 | 처리 | 이번 결과 |
|---|---|---|
| (1) W1B 와 h41 W1 | 키별 블록 SSE. h41 은 예측을 float32 로 저장했으므로 두 쪽 모두 float32 로 바꾼 예측에서 만든다. 1절 허용 오차(`--gate-level`, 기본 local_rescale) 또는 예측 차 1 float32 ulp 이내면 통과. rf 키는 넣지 않는다 | 스모크 조각(캐나다 반복 1, seed 0): 7키 실패 0 |
| (2) M1C 와 m1 표 | m1 표의 RMSE 가 소수 3자리 반올림이라 |반올림(새 값) − 표 값| ≤ 0.0011 cm 면 통과. 실패하면 '대회 연속성 재현 실패(환경 미기록)'를 적고 알래스카 능형 키만 뺀다 | 스모크에는 M1C 가 없어 '조각 없음' |

## 6. 결정성

- 같은 스모크를 두 번 돌리면 rf 키(D0w·RSw[rf])의 예측만 달랐다(다른 키는 비트 단위로 같다). scikit-learn 의 숲 예측이 스레드(n_jobs 2)에서 나무 값을 더하는 순서에 따라 반올림 수준으로 달라지는 것으로 본다. FitterV 는 동결 모듈이라 바꾸지 않았다. 계획대로 rf 는 관문에 넣지 않는다. 봉인 표의 sha256 은 실행마다 다를 수 있다.

## 7. 로컬 실행 기록(2026-10-04)

- 시험 21개 통과(XG·XH 합계, 12 s).
- `--count-only`: 단위 42개(W1K 12개는 색인 없음으로 건너뜀), 적합 1,764건(W1K 제외), 캐나다 W1B 반복 1 의 묶음 0 이 학습 519·채점 231 로 h41 W1 조각과 같다.
- 스모크: 단위 2개 완료, 최대 RSS 357 MB, 봉인 표 4개(열지 않음).
- 메모리: RLIMIT_AS 10 GiB 에서 CatBoost 기본 용량 적합이 멈춰(x_product_comparison 기록 7절) 상주 메모리 감시 스레드로 바꿨다.

## 8. 남은 일

1. 로컬 1c: `--build-knndm`(알래스카 k-평균 후보 100개 × 반복 3, 수십 분 예상). 색인과 메타를 Rescale 묶음에 넣어야 한다(`xbatch_core.PAYLOAD_INPUTS` 에 없다).
2. 관문 (1)·(2)는 Rescale 회수 뒤 로컬에서 `--gate-only` 로 본다(h41 조각과 m1 표가 로컬에만 있다).

## 9. 검토 반영(2026-10-04 적대 검토, 결함 14건 가운데 XH 해당 7건)

- 적용 범위: 검토 결함 목록의 1, 3, 10, 11, 12, 13, 14. 등록 단·모형·서술 항목·CI 방법(블록 재표집 10,000회)·'판정어 없음' 규칙은 바꾸지 않았다.
- 시험 표지 (u)·(y)는 `tests/test_x_xg_xh.py` 머리말의 '검토 반영' 목록이다.

| 결함 | 내용 | 처리 | 시험 |
|---|---|---|---|
| 1·13 | `--allow-local` 이 상주 메모리 감시를 끄고, `require_memory(30)` 이 기다리지 않았으며 집계·관문 경로가 가용 메모리를 보지 않았다 | 주 함수: `on_rescale()`(환경 변수 WF_RESCALE·LG_RESCALE)이 거짓이면 허용 표지와 무관하게 `rss_watchdog(10 GB)`. `--build-knndm`, `--gate-only`, 본 실행·스모크, 집계 경로 모두 `wait_memory`(30 GB 하한, 최대 3,600 s 대기, 60 s 간격, Rescale 에서는 대기 없음)를 거친다. 세기·묶음 점검은 기다리지 않는다 | (y) |
| 3 | 묶음에 kNNDM 색인·메타·예측 영역이 없어 모든 W1K 단위가 단위 안에서 실패했다 | 모듈 `PAYLOAD_EXTRA`(색인, 메타, 예측 영역)와 `--payload-check`(`payload_manifest(extra=)` 의 missing, 묶음 정보 `xh_payload_manifest.json`, 없으면 종료 코드 1). 공용 골격 `xbatch_core.PAYLOAD_OPTIONAL`·`PAYLOAD_REQUIRES` 에 같은 세 경로를 더했다(추가만). `preflight_w1k`: W1K 단위가 하나라도 있으면 execute 전에 색인을 읽어 sha256 을 메타(또는 `--knndm-sha256`)와 대조하고 (지역, 변형, 반복) 행의 존재를 확인한다. 없거나 다르면 적합 전에 SystemExit | (y) |
| 10 | 관문 (2) 실패가 집계에 반영되지 않았고 봉인 메타에 관문 상태가 없었다 | `gate_status` 가 `xh_gate_meta.json` 을 읽는다(조각 unit.json 보다 오래되면 stale). 관문 (2)가 '실패'이고 오래되지 않았을 때만 알래스카 연속성 행(learner ridge: D0w[ridge], RSw[ridge, λ 0.75])을 xh_ladder 에서 빼고(cont_action drop) ridge_note 에 '대회 연속성 재현 실패(환경 미기록)'를 적는다. 통과면 continuity_gate '관문 2 통과', 기록 없음·미실행·오래됨이면 행을 두고 '관문 2 미확인(사유)'. 관문 (1)·(2) 요약, 뺀 행 수, stale 여부를 `xh_summary_meta.json` 의 gates 에 쓴다. xh_dd·xh_stage_contrasts 는 영향을 받지 않는다(ridge 는 LEARNERS 에 없다) | (u) |
| 11 | 관문 (1)에 1절 허용 오차 밖의 '예측 차 1 float32 ulp 이내' 통과 경로가 있었다 | `gate1_decision`: 유한 셀 집합이 같고 1절 허용 오차(`--gate-level`, 기본 local_rescale: \|ΔSSE\| ≤ 1e-4 cm² 또는 상대 1e-9)이내일 때만 통과. max_ulp 와 pass_ulp_only(허용 오차는 넘고 예측 차 1 ulp 이내)는 정보 열이고 관문 (1) 요약의 n_ulp_only 로만 센다. 5절 표 (1)행의 '또는 예측 차 1 float32 ulp 이내면 통과'는 이 절로 대체된다 | (u) |
| 12 | 1,000회 재표집으로 본 봉인 폴더에 썼다 | `check_nboot`: 스모크가 아니고 a.nboot < 10,000 이면 집계 전에 멈춘다(주 함수에서 재표집 전에, summarize(write=True) 안에서 다시) | (y) |
| 14 | 파일 위치 이탈 기록 | 1절의 기록을 유지하고 묶음 정보 `module_inputs.file_location` 과 봉인 메타 file_location 에 적는다. `--gate-level` 은 관문 (1)이 쓰므로 유지 | (y) |

- 8절 남은 일 1 의 '색인과 메타를 Rescale 묶음에 넣어야 한다'는 결함 3 처리로 해소됐다. 색인 생성(`--build-knndm`)과 관문 (1)·(2)의 Rescale 회수 뒤 확인은 그대로 남아 있다.
- 거절한 항목: 없다.

### 9.1 이번 확인(2026-10-04)

- 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q -p no:cacheprovider tests/test_x_xg_xh.py` → 27개 통과. `tests/test_xbatch_core.py` 25개 통과.
- `--count-only`: 단위 42개(W1K 12개는 색인 없음으로 건너뜀), 적합 1,764건, 추정 0.47 워커·시간(7절과 같다).
- `--payload-check`: 필수 입력 3개 가운데 없음 3개(1c 전이라 예상된 상태). 묶음 정보 `data/processed/xbatch/XH_validation_ladder/xh_payload_manifest.json`.
- 동결 모듈 h41 sha256 앞 16자 492373e4d37b5ea4 불변(h40·h42·h54 도 불변).
- kNNDM 색인·스모크·집계는 이번에 돌리지 않았다(로컬 1c).

### 9.2 로컬 1c 명령(ROOT 에서, 라벨 값 미사용)

1. `free -g` 로 가용 메모리 30 GB 이상 확인(모자라면 wait_memory 가 최대 1시간 기다린다). 무거운 로컬 작업 동시 2개 이하(계획 1절).
2. 예측 영역 표와 kNNDM 색인(알래스카 k-평균 후보 100개 × 반복 3, 수십 분 예상. 입력은 좌표와 ERA5-Land 기후 격자 `data/raw/era5land/nh_monthly_2015-2020.nc`, 레나 격자 `data/processed/map_lena/lena_grid_x25_v1.csv.gz`):
   `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python3 scripts/3_deep_learning/x_validation_ladder.py --build-knndm --threads 4`
   - 산출 `data/processed/xbatch/XH_validation_ladder/xh_knndm_index_v1.csv`, `xh_knndm_meta_v1.json`(index_sha256, scikit-learn 판, 선택 q·W), `xh_pred_domain_v1.csv.gz`. 화면에는 행 수·sha256·영역 점 수·최대 RSS 만 나온다.
3. 세기 재확인(단위 42개는 그대로이고 W1K 12개 단위의 적합 750건이 더해져 적합 2,514건이 돼야 한다. 2절의 적합 수와 같다): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 10 python3 scripts/3_deep_learning/x_validation_ladder.py --count-only`
4. 묶음 점검(없음 0개여야 R1b 제출): `python3 scripts/3_deep_learning/x_validation_ladder.py --payload-check`
5. 색인 sha256 을 개정 이력(또는 묶음 tar 의 sha256 기록)에 적는다(계획 1절 '묶음과 코드 출처').
