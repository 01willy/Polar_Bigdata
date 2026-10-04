# xbatch_core 구현 기록(계획 해석과 구현 결정)

- 대상: `scripts/3_deep_learning/xbatch_core.py`, `tests/test_xbatch_core.py`, `scripts/rescale/xbatch_constraints.txt`
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 1절·4절·부록 A, `docs/research/2026-10-04/harness_implementation_plan.md` 2절
- 작성: 2026-10-04. 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리다. 처리는 모두 가장 문자 그대로의 해석을 따랐고, 두 해석의 결과가 다른 곳은 부록 A 의 기록으로 판별했다.
- 이 기록을 쓰면서 계산한 것: 분할 구조(셀 수·블록 수, 라벨 값 미사용), warm_trim 의 제외 셀 수(공변량 √TDD 만 사용). 모형 적합과 결과 표 계산은 하지 않았다.

## 1. 분할 제외 규칙(1절 '분할 seed' 행)

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| 앞선 분할과의 비교 | 'A 블록 집합이 같거나 여집합인 분할' | 같음(prior_dup)과 여집합(prior_mirror)을 모두 뺀다 | 문구 그대로. AL-1, AL-4, AL-6, LE-2, NAtlantic~lic(seed 209)의 제외 가운데 다수가 여집합이다 |
| 새 범위 안의 중복 | '새 범위 안의 중복' | A 블록 집합이 같은 경우(dup_new)와 여집합인 경우(mirror_new)를 모두 중복으로 본다. 앞 seed 가 다른 사유로 빠졌어도 비교에 넣는다 | NAtlantic~lic 행(유효 4, 앞선 분할과 같음 1, 새 범위 중복 5, 채점 블록 4–5)은 이 해석에서만 재현된다. 여집합을 빼면 유효 5, 채점 블록 2–5 로 부록 A 와 다르다 |
| 사유의 적용 순서 | 정하지 않음 | 앞선 분할 → 새 범위 중복 → 채점 블록 2 미만 → 채점 블록 5 미만(XC·XC-r 의 PE1·PE2·알래스카 대상) | 부록 A 의 사유별 수(AL-1 '7개 앞선 분할과 같음(그 가운데 채점 블록 2 미만 포함)', NAtlantic '1개 앞선 분할과 같음, 5개 새 범위 안 중복')가 이 순서에서 재현된다 |
| XC-r(211–220)의 앞선 분할 | '그 대상에서 이미 평가에 쓴 분할' | 1–200(LGX-N4). 201–210 은 넣지 않는다(등록 시점에 평가에 쓰지 않았다) | 201–210 을 넣어도 결과가 같았다(알래스카 10, 레나 8(215·217 제외), 캐나다 10) |
| LGX-N4·LGD 밖 대상의 앞선 분할 | 정하지 않음 | 1–5(LG·LGD·WF4 의 분할) | 현재 새 seed 를 쓰는 대상에는 해당하지 않는다(Canada~exp~lic 등은 분할 1–5 만 쓴다) |
| 분할 1–5 재계산 기록의 'Russia_C 6' | 대상 판 미기재 | LGD 실행 표(Russia_C~lgd)의 값으로 본다 | 시험 (q)에서 Russia_C~lgd 의 분할 1–5 최소 채점 블록이 6 이다 |

- 재현 결과: `xbatch_core.py --split-plan` 과 시험 (q)가 부록 A 의 21행(XC 18행, XC-r 3행)의 유효 분할 수, 뺀 seed 와 사유, |A| 범위, 분할별 채점 블록 범위, 채점 블록 합집합을 모두 다시 냈다. 분할 1–5 의 최소 채점 블록(레나 5, 캐나다 12, 러시아 W 7, 러시아 E 8, 알래스카 21, Russia_C 6, Tibet 14)도 같다. `split_plan(..., check=True)` 는 실행 때마다 이 대조를 단언한다.
- 실행 분할에서 만든 문맥의 메타(dup_of −1, valid True, n_valid_splits)는 새 seed 규칙의 값으로 덮어쓴다(`build_tctx`, `build_rctx` 의 plan_row). h40.Data.split_structure 의 유효성은 중복·채점 블록 2 미만만 보므로 새 규칙과 다를 수 있다. 실행 여부는 새 규칙이 정한다.

## 2. 2단 재표집(1절 '대비와 CI(두 팔의 라벨 집합이 다른 대비)')

- 구현은 h39 `l43_region` 의 계산을 곡선 키 두 개로 일반화한 것이다. 분할마다 블록 가중 W(seed_of(seed, 분할))를 두 팔이 공유하고, 팔마다 추출 번호를 복원 추출한다(seed_of(seed, 분할, 팔 표지)). 시험 (h1)에서 팔 표지 ('block', 'cell')로 `l43_region` 의 분포와 점 추정이 1e-12 안에서 같다.
- 팔 표지의 기본값: 곡선 키의 문자열. 계획은 표지를 정하지 않았다. 이 기본값에서는 같은 대상의 여러 대비가 같은 팔에 같은 추출 재표집을 쓰므로 재표집마다 Δ(A−C) = Δ(A−B) + Δ(B−C) 가 성립한다(시험 (h3)). 블록 가중 seed 의 기본값은 seed_of('xbatch-2s', 저장소 이름)이며 대비와 무관하다.
- '2단 공통'(보조): 블록 가중을 사용 분할 채점 블록의 합집합에서 한 번 뽑는다(seed_of(seed, 'U')). 추출 재표집 색인은 주 2단과 같다. 추출이 하나뿐이면 h42.boot_delta_common 과 같은 분포가 된다(시험 (h2)).
- '추출 조건부' 보조 열: 계획 문구('boot_delta_blocks 의 CI')대로 h40.contrast(h4_common.boot_delta_blocks, seed = tm.seed)를 쓴다. 이 함수는 두 쪽 모두 추출이 여럿이면 공통 추출 번호로 제한하므로, 라벨 집합이 다른 두 팔도 추출 번호끼리 짝지어진다. 판정이 주 2단과 다르면 '추출 변동 의존'을 붙인다.
- CI 를 내는 조건은 같은 라벨 집합 대비와 같다(has_ci, 채점 블록 합집합 8 이상). 점 추정만 내는 대상(NAtlantic~lic)은 분포 없이 점 추정만 남는다.

## 3. 판정·p·Holm

| 항목 | 처리 |
|---|---|
| δ_rel | 0.02 × P0 RMSE(가중별). 지역 행은 그 지역의 P0 RMSE, 풀 행은 풀에 쓴 지역의 P0 RMSE 평균. P0 는 n 과 무관하므로 '같은 풀·n·가중'은 같은 채점 셀과 같은 풀을 뜻하는 것으로 읽었다. 세 판정(δ 0.5, 1.0, δ_rel)이 판정 가능하고 서로 다르면 '한계 의존' |
| 양측 p | 두 가중 가운데 큰 h4_common.boot_p(주 분포. 2단 대비는 2단 분포) |
| 비열등 p | 두 가중 가운데 큰 P(Δ^b ≥ 0.5)(h54.ni_p 와 같은 정의, 1/재표집 수 하한 없음). Holm 에는 2배(1 에서 자름)를 넣는다(`p_ni_x2`) |
| Holm | 행 없음·판정 불가는 p 1 로 넣어 가족 크기 m 을 유지한다. 입력 수가 m 보다 적으면 1 로 채워 계산한다. XC-2s 의 '보정 배수'는 m − 순위 + 1(동률은 입력 순서)이다(`holm_table`) |
| XC 주 가설의 CI 기준 | `criterion_met`: 2단 CI 와 2단 공통 CI 의 상한이 두 가중 모두 0(우월) 또는 0.5(비열등) 미만. Holm 조건은 실험 모듈이 본다 |
| _rule3 | h54._rule3 과 같은 순서(반대 판정 → 기각, 충족 없음 → 판정할 수 없는 대비가 있으면 판정 불가 아니면 기각, 모두 충족 → 지지, 그 밖 → 부분 지지) |
| 소수 블록 | 사용 분할의 최소 채점 블록 < 5 인 지역 행에 '소수 블록'. 풀 행에는 해당 지역 이름을 적는다 |
| 풀 표지 | CI 풀 지역 수 k 와 등록 지역 수 m 으로 '지역 k/m', '부분(지역 k/m)', '판정 불가'(k < 2) |
| 지역 수준 추론 | h42.region_inference(부호, 부호 뒤집기, HK). 지역 분산 = 주 분포(셀 가중)의 분산(h42.region_inference_main 과 같다). region_general = HK CI 가 0 을 제외 |

## 4. 실행 보호·출력 제한·경로

- 실행 보호: 계획 문구는 'WF_RESCALE=1 또는 --allow-local(h54 와 같다)'이다. h54 는 LG_RESCALE=1 도 받으므로 같게 받는다. 허용 표지가 없으면 스레드 ≤ 4, 집계 재표집 ≤ 1,000(h54 와 같다).
- 출력 제한 패턴: WRAPUP 783행의 grep 패턴('판정|verdict|Δ|delta')에 RMSE·rmse, 판정어(우세, 열세, 동등, 미결정, 지지, 기각), 표 열 이름(ci_lo, ci_hi, p_two, p_boot, p_ni)을 더했다. CatBoost 같은 C 확장이 파일 기술자에 직접 쓰는 출력은 Python 쪽 필터를 지나지 않으므로 셸에서 `SHELL_FILTER` 를 함께 쓴다.
- 산출 경로: 기본 허용 뿌리는 `data/processed/xbatch/` 하나다(환경 변수 XBATCH_OUT_ROOTS 로 뿌리를 더할 수 있다. 내려받기 모듈은 `data/raw/<새 자료원>/` 을 명시적으로 넘긴다). 묶음 정보 JSON 은 `data/processed/xbatch/` 또는 `work/` 에만 쓴다.
- 봉인 폴더: `data/processed/xbatch/<새 이름>/sealed/`. 목록 파일(sealed_manifest.json, 파일별 행 수·sha256·시각)과 README 를 같은 폴더에 둔다. 읽기 보호 `assert_not_sealed` 는 경로에 'sealed' 마디가 있으면 거부한다.

## 5. 자료

- 실행 표 별칭: h54.LGD_SPECS 의 세 항목(Tibet_LGD, NAtlantic~lic, Russia_C~lgd)에 Canada~exp~lic(디렉터리 Canada_expanded_lic, 표 안의 대상 Canada)을 더했다. 이름은 계획 2.2 xb_t 표의 이름이다. h54.build_tctx 는 이 별칭을 모르므로 `xbatch_core.build_tctx` 가 h40.build_ctx 를 직접 부른다.
- 점 추정 여부: 'Russia_C~lgd' 의 '~' 앞 이름이 v3 의 점 추정 대상(Russia_C)과 같으므로 별칭 표를 먼저 본다.
- 약관 점검(4절): 네 실행 표 모두 fidelity_base_v4_labels.csv 의 lic_unverified = 1 셀이 0 이다(Tibet 17,704행, NAtlantic_lic 17,591행, Russia_C 17,622행, Canada_expanded_lic 17,609행). 토양 표는 모두 `../../../e5_soil_tdd_v4.csv` 로의 기호 연결이다.

## 6. XI warm_trim(계획 2.9)

- 계획 문구 'A 에서 s ≥ min(s_W) 인 셀을 뺀다'의 s_W 를 W 블록의 모든 셀로 읽었다(문구 'W 셀'). W 채점 셀만 쓰는 정의와 Canada_expanded_lic 분할 1–10 에서 뺄 셀 수가 같았다(분할별 14, 24, 7, 15, 7, 10, 10, 18, 17, 18셀. 하네스 보고서 3.1 의 7–24셀과 같다). s 가 결측인 A 셀은 남긴다. warm 변형에만 정의한다.
- `R10TrimUnit` 은 h54.R10Unit 의 방법·격자·추출 규칙을 그대로 쓰고 저장소 이름만 `<대상>~warm_trim|r`(W·I 저장소는 `~warm_trimW`, `~warm_trimI`)이다. 추출 seed 의 모드 표지는 R10Unit 과 같은 'r10w' 이고 |A| 가 줄어 추출은 다르다.

## 7. 패키지 판

- 제약 파일은 1절의 다섯 판(catboost 1.2.10, scikit-learn 1.9.1, pandas 2.3.3, scipy 1.17.1, numpy 2.1.1)이다. 로컬 판은 catboost 1.2.10, scikit-learn 1.3.0, pandas 2.1.4, scipy 1.11.4, numpy 1.26.4, torch 2.6.0 이라 네 판이 다르다. 로컬 스모크와 단위 시험은 로컬 판으로 돈다. 로컬·Rescale 사이 관문은 1절 허용 오차(1e-4 cm² 또는 상대 1e-9)와 CatBoost·물리식·ridge 키 한정을 따른다(`gate_compare(level='local_rescale', key_fn=...)`). `--deps-check` 는 차이가 있으면 종료 코드 1 이다(Rescale 설치 뒤에는 0 이어야 한다).

## 8. 시험 실행 기록

- 명령: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --as=10737418240 taskset -c 100-103 python3 -m pytest -q -p no:cacheprovider tests/test_xbatch_core.py`(출력은 계획 0.3 의 grep 필터를 거쳤다).
- 실제 자료를 읽는 시험은 (q) 부록 A, (r) 약관 두 개이며 셀 수·블록 수만 쓴다(세기 범주).
