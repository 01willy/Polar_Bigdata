# x_placement_policy 구현 기록(계획 해석과 구현 결정)

- 대상: `scripts/3_deep_learning/x_placement_policy.py`(XD-alg, Algorithm P 선택, XD-learn 학습 자료, S9-GBM, 동결, XD-4·5), `scripts/3_deep_learning/x_placement_deepsets.py`(S9-DS), `tests/test_x_xd.py`
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 2.3절(Algorithm P 고정 규칙), 2.4절, 0.3절, 1절. `docs/research/2026-10-04/placement_and_workflow_algorithm.md` 4·6절, `harness_implementation_plan.md` 4.4절
- 작성: 2026-10-04. 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리다. 모두 가장 문자 그대로의 해석을 따랐다.
- 이 기록을 쓰면서 계산한 것: 실제 자료의 세기(`--count-only`, 라벨 값 미사용), 제한 스모크(코어 4개, 스레드 2, 산출 표는 봉인 폴더, 화면에는 수와 해시만), 합성 자료 시험. 판정 표·봉인 파일은 열지 않았다. 동결 모듈 4개는 바꾸지 않았다(sha256 앞 16자 7098f59dabe2b73f, 22e215e435e157be, cf9a1f6d0929c892, 492373e4d37b5ea4).

## 1. 파일 이름

| 계획 2.4 '구현·시험' | 작업 지시 | 처리 |
|---|---|---|
| `x_placement_policy.py`(S8 계열, algorithm_p_order, power_alloc, coefs_w, coefs_re, 집합 효용 단위, select_algorithm_p) | 같음 | 같은 파일. 함수 이름은 `coef_w`, `coef_re`(계획의 coefs_w·coefs_re 와 같은 역할) |
| `x_placement_policy_learn.py`(특징, GBM, DeepSets, 동결, 선택 내보내기) | `x_placement_deepsets.py`(DeepSets) | 특징·GBM·동결·색인 내보내기는 `x_placement_policy.py` 에, DeepSets 는 `x_placement_deepsets.py` 에 두었다. `x_placement_policy_learn.py` 는 만들지 않았다 |
| `tests/test_x_placement.py`(a)–(h) | `tests/test_x_xd.py` | `tests/test_x_xd.py` 가 (a)–(h)를 같은 기호로 담고 누설·세기·재현·봉인 시험을 더한다 |

XC 모듈(`x_workflow_end_to_end.py`)은 이 모듈의 `placement_order(name, Z, blk, schedule, seed)` 를 부른다. 2026-10-04 에 XC 의 참조 구현과 무작위 문제 400개(S8a·S8b·S8c·S8p·S2·S4)에서 순서가 모두 같음을 확인했다(라벨 미사용).

## 2. 배치 S8(γ)(계획 2.4 의사코드)

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| 누적 예산 일정 | 입력 '(10, 40, 160)' | XD-alg 은 그 팔의 n 격자(xd_t 10·40, xd_r 20·50·100·200·500, 학습 집합은 DEV_N 가운데 n 이하와 n). XC 와 XD-4 의 고정된 Algorithm P 는 (10, 40, 160)(`AP_SCHEDULE`, 15절 2) | xd_t 의 10·40 은 10·40·160 의 앞부분과 같은 순서를 낸다. xd_r 은 n 200·500 이 160 을 넘으므로 격자를 일정으로 둔다 |
| 블록 번호 정렬 | 없음 | 문자열 정렬(BlockStore 와 XC 참조 구현의 규약) | 이 자료의 블록 번호는 음수와 7–9자리가 섞여 숫자 정렬과 문자열 정렬이 다르다. 두 모듈의 순서를 같게 했다 |
| 첫 블록 seed | 'seed 로 고른다' | RandomState(seed % 2^32).randint(블록 수), seed = seed_of('xd-S8', 대상, 모드, 분할, 추출). S8 네 변형이 같은 seed 를 쓴다(규칙만 다른 짝) | XC 는 seed_of('xc-P', …) |
| S8b 의 맨 첫 셀 | '블록 안 첫 셀도 farthest-first' | π ∪ L0 가 비어 있으면 블록 중심 근접 셀 | farthest-first 는 기준 집합이 있어야 정의된다. XC 참조 구현도 같다 |
| 상한 N_b | '상한 N_b' | 상한에 걸린 블록을 N_b 로 고정하고 남은 수를 나머지 블록에 같은 비례로 다시 나눈 뒤(반복) 최대 나머지 | 합 = min(n, ΣN_b) 를 보장한다 |
| 소수부 동률 | 'β 순서 앞 블록 우선' | 소수부를 12자리에서 반올림한 뒤 같은 값이면 β 순위 | 1/3 같은 부동소수 잡음을 동률로 본다 |
| deficit 소진 | 정하지 않음 | 남은 셀이 있는 블록 가운데 deficit 최대(동률 β 순서)를 계속 고른다. deficit 은 음수가 될 수 있다 | 단계 사이 배분이 단조가 아닌 경우(최대 나머지 방식의 역설)와 상한 때문에 deficit 합이 모자랄 수 있다 |
| L0 | '이미 있는 라벨' | 후보 색인으로 받는다. 블록 순서는 L0 의 블록들에서 시작하고, 거리·블록 수에 L0 를 넣는다 | 모의에서는 L0 = ∅ 다 |
| 동률 | 정하지 않음 | 블록 안 셀은 작은 색인, 블록은 β 순서 | |
| 거리 | 표준화 공변량 거리 | 셀 사이 거리는 차의 제곱합의 제곱근(행렬 곱 전개를 쓰지 않는다). 특징 계산은 scipy cdist | BLAS 스레드 수와 무관하게 같은 값을 낸다 |

## 3. 계수 변형 S8w·S8r

- 계수 수축(2026-10-04 16:40 고침, 15절 5): S8w·S8r 의 P1·R1 앵커는 수축하지 않은 E_w·E_RE 다. 계획 2.4 는 S8w 를 'S8a 의 라벨 + 설계 가중 계수 E_w = Σ w s y / Σ w s²', S8w·S8r 을 '계수만 바꿔 계산'으로 등록했고 E_w 의 수축은 정하지 않았다. 가장 문자 그대로의 해석은 수축하지 않은 E_w 다. 원시 계수가 비유한이면 E0 를 쓰고 표지(coef_nonfinite_E0)를 단다. 초판은 h42.shrink(κ 10)로 E0 쪽으로 수축했다(n 10 에서 수축 가중 0.5 라 S8w − S8a 의 계수 차이가 절반이 된다). 수축판(예: S8wk·S8rk)은 계획에 등록되지 않은 팔이므로 만들지 않았다.
- 방법: 계획 2.4 '(P1 은 닫힌 형식, R1 은 다시 적합)'대로 P1·R1 만 낸다. 지역 내 팔의 D1 은 S8w·S8r 에 계산하지 않는다.
- S8r 의 추정식(계획은 '블록 E 의 무작위 효과 평균, τ² 는 라벨로 추정'만 정했다): DerSimonian–Laird. 블록 계수 E_b = Σ_b s y / Σ_b s², 표집 분산 v_b = σ²/Σ_b s². σ² 는 블록 안 잔차 분산(자유도 Σ(n_b − 1) ≥ 1)이고, 블록마다 라벨이 하나뿐이면 합동 최소제곱 잔차 분산(자유도 n − 1)이다. 이 퇴화 경우 Q = n − 1 이 되어 τ² = 0, E_RE = E_ls 다(라벨 10개가 서로 다른 10블록에 놓이는 S8a n 10 이 이 경우다). 각 추출의 블록 수·자유도·σ² 출처를 unit.json 의 notes.s8r 에 적는다.

## 4. XD-alg 단위와 집계

| 항목 | 처리 |
|---|---|
| S1·S2·S4 | h54 의 WF2(RUnit.strategy_sets·s4_order·std_stats)·WF8(T8Unit) 함수와 seed 를 그대로 쓴다. 시험 (j)가 합성 자료에서 T8Unit·RUnit(wf2)과 블록 SSE 가 같음을 확인한다 |
| 같은 라벨 집합의 추출 | 두 팔 모두 WF8 규칙(적합을 재사용하고 그 추출 번호로 저장)을 쓴다. WF2 는 첫 추출만 저장했으므로 xd_r 의 키는 WF2 의 키를 포함하는 집합이다. 재현 관문은 공통 키만 비교한다. 2단 CI 의 추출 재표집에서 결정적 전략의 추출 수가 줄지 않게 하려는 것이다 |
| 판정 방법 | XD-1·2·3 은 R1(λ 0.25)로 판정한다(Algorithm P 규칙, WF8 과 같다). P1(두 팔)과 D1(지역 내)은 대비 표의 보조 행이다 |
| CI | 라벨 집합이 다른 대비(S_k − S1, S8 변형 − S2·S4, S8w·S8r − S1)는 2단 CI(xbatch_core.region_stat_two_stage), 같은 라벨 대비(S8w·S8r − S8a)는 블록 재표집(region_stat_same). 추출 조건부 보조 열·'추출 변동 의존'·'분할 독립 가정 의존'·δ 1.0·δ_rel 과 '한계 의존'은 xbatch_core.pool 이 붙인다 |
| XD-1 | PE1 풀(지역 5개 등록, n 40 은 레나·캐나다만이라 '부분(지역 2/5)')의 S8a − S1. 종합은 _rule3(우세 기준) |
| XD-2 | 레나 전이 S8w − S8a 의 n 10·40 이 모두 우세이고, 알래스카·레나·캐나다의 S8w − S8a(R1) 행(전이 n 10·40, 지역 내 n 20–500)에 열세가 없으면 지지, 그 밖은 기각. '어느 행에도'를 두 팔의 행으로 읽었다. 판정 순서(`xd2_verdict`, 15절 7): (1) 열세 행이 있으면 기각(열세 문장. 레나 행이 없어도 지지가 될 수 없으므로 판정할 수 있다), (2) 레나 n 10·40 가운데 행이 없거나 판정할 수 없으면(CI 없음) 판정 불가(1절 '행 없음·실패'), (3) 둘 다 우세이면 지지, (4) 둘 다 동등이면 기각(동등 문장), (5) 그 밖은 기각('확인하지 못했다'). 레나 밖 계열 행의 결측은 판정 불가로 보지 않는다(없는 행은 열세일 수 없다) |
| XD-3 | S8a·S8b·S8c·S8p − S2·S4 의 PE1 풀 비열등(두 가중 2단 CI 상한 < 0.5)과 4분 판정, n 10·40. 문장 갈래(`xd3_parts`): 행 없음·판정할 수 없음 → 판정 불가, 비열등이면서 열세(CI 가 (0, 0.5) 안)이면 두 문장을 함께 쓴다, 그 밖은 비열등·열세·동등·미결정 가운데 하나 |
| Holm | 가족 m 20 = XD-1 n 2개(양측 p) + XD-2 n 2개(레나 행의 양측 p) + XD-3 16개(비열등 단측 p × 2). 행 없음은 p 1. 보조 열 |
| 해석 문장 | 계획 2.4 의 사전 고정 문장. XD-1·2·3 모두 Holm 보정 뒤 xbatch_core.compose_sentence 로 만든다(보정 전 유의, 0.5 cm 미만, 판정 불가 사유, XD-1 은 지역 열세 덧붙임). XD-1 의 지역 수는 풀 표지를 그대로 쓴다('…작았다(지역 5/5)', '…(부분(지역 2/5))'). XD-2 지지 문장은 두 레나 행의 Holm p 가운데 큰 값과 |Δ| 가 작은 값으로 표기하고, 열세 문장은 행마다 표기를 괄호로 붙인다(Holm p 는 레나 전이 행에만 있다). XD-3 의 Holm 가족 p 는 비열등 p 이므로 '보정 전 유의'는 비열등 문장에만 붙고, 4분 판정 열세 문장에는 등록된 Holm p 가 없어 0.5 cm 미만 표기만 붙는다(1절 문구 'Holm 보정 p ≥ 0.05 인 …'의 문자 그대로 적용). 계획 2.4 의 XD-3 판정 불가 칸은 비어 있어 1절 다섯 갈래의 '판정할 수 없었다'를 썼다 |
| XD-6 | 단위의 notes.traits(블록 수, 유효 블록 수, 공변량 블록 사이 비율, 셀 가중 E, 블록 평균 E(셀 3개 이상 블록), 로그 비)의 분할 평균과 S2·S4·S8a − S1(R1) 지역 행의 표. 셀 가중 E·블록 평균 E 는 A 라벨 전체로 계산한 서술 값이고 선택·예측에는 쓰지 않는다 |

## 5. Algorithm P 선택(계획 2.3 고정 규칙)

- 읽는 범위: 조각 파일 이름(대상, 모드)으로 알래스카 계열(전이 Alaska·AL-1–AL-6, 지역 내 Alaska·AL-1·AL-2)만 고른 뒤 연다. 다른 조각은 열지 않는다(시험 (h)가 레나·티베트 조각을 깨진 파일로 두어 확인한다).
- 제외: 행마다 S_k − S1(R1@0.25)의 2단 CI(주 분포) 하한이 셀 가중 또는 블록 등가중에서 0 을 넘으면 그 후보를 뺀다. CI 를 낼 수 없는 행(채점 블록 합집합 8 미만 등)은 제외 근거가 되지 않는다(하한이 없다). 2단 공통 CI 는 쓰지 않는다(계획 문구 '2단 CI').
- 점수: 전이 알래스카 계열 대상마다 상대 변화 (RMSE(S_k) − RMSE(S1))/RMSE(S1) 를 가중별로 n 10·40(있는 n)에서 평균하고 큰 값을 그 대상 값으로 둔 뒤 대상 단순 평균. RMSE 는 추출·seed 평균 뒤 분할 평균(xbatch_core.grp_rmse2)이다. 지역 내 행은 제외에만 쓴다.
- 동률(1e-9 미만): S8a, S8c, S8p, S8b, S2, S4 순. 모든 후보가 빠지면(또는 점수가 없으면) S1.
- 산출: `selection/algorithm_p_selection.json`(고른 후보, 설정, 후보별 제외 행·점수, 읽은 저장소와 조각 파일, 재표집 수, 이 파일의 sha256, 선택 함수 원문의 sha256, xbatch_core sha256, 동결 표지)과 `selection/algorithm_p_alaska_table.csv`, 각 `.sha256` 사이드카. 화면에는 저장소 수, 행 수, 해시 앞 16자, 고른 후보 이름만 쓴다. 부록 XC-0 에는 선택 파일의 sha256, `script_sha256`, `selection_fn_sha256` 을 적는다.
- 재표집은 10,000회로만 한다(허용 표지 없는 로컬 실행은 거부). R1a 의 집계 단계(`--summarize-only`)가 봉인 표와 함께 선택 파일을 쓴다. 로컬에서 같은 조각으로 다시 돌리면(`--select-p --allow-local`) 같은 결과가 나와야 한다.
- 마감 초과(R1a 제출 + 48 h): `--select-default` 가 P_default(S8a) 선택 파일을 쓴다. 시각 판단은 사용자가 한다.

## 6. XD-learn 학습 자료

| 항목 | 처리 |
|---|---|
| 집합 구성 | n_sets 200 = S1 80(seed_of('xdl', 과제, 모드, 분할, n, 번호, 'S1')의 무작위) + 구조 60(S2·S4·S8a·S8b·S8c·S8p 를 돌아가며 10개씩, 다른 seed) + 섭동 60(구조 집합 j mod 60 의 비율 f ~ U(0.2, 0.5), round(f·n)개를 그 집합 밖 무작위 셀로 바꾼다) |
| 중복 집합 | 지우지 않는다(적합은 재사용). 순위 손실에서 효용이 같은 쌍은 빠진다 |
| 효용 기준 | 기준 S1 = 같은 단위의 h40.draw_cells 추출 d 0–4(XD-alg·WF8 의 S1 과 같다)의 R1@0.25(seed 0) RMSE 평균. 효용은 집계 때 runs 의 RMSE 로 계산한다(조각에는 RMSE 만 있다) |
| 집합 특징 | 덮임(후보 부분표본 1,000개(seed_of('xd-cov', …))에서 가장 가까운 라벨까지 거리의 평균·90 백분위·최댓값), 에너지 거리(V 통계: 2·E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖, 대각 포함), 블록(덮은 블록 비율, 라벨 배분 엔트로피, Σ_b |라벨 비율 − 셀 비율|), √TDD(평균, 표준편차(ddof 0), 후보 평균 대비 비, Σs²), 비가중 AOA 비유사도 평균(h54.aoa_di, 원천 통계 표준화, seed_of('wf-aoa', 대상, 분할) = WF2 S5 와 같은 값), 과제 문맥 5개(후보 수, 블록 수, 유효 블록 수, 공변량 블록 사이 비율, 원천 대비 공변량 차이 = 열별 |평균 차|/원천 표준편차의 평균) |
| 집합 크기 | 특징 'n_set' 을 더했다. 모형 하나가 n 10·20·40 을 함께 배우므로 필요하다(계획 목록에 없다) |
| √TDD 결측 | 후보 평균으로 채운다(특징 계산에서만) |
| 원소 특징(DS) | [Z(25), s 표준화(과제 안), 블록 셀 비율, DI, 같은 집합 안 최근접 라벨 거리(원소 1개면 0)] 29개 |
| 저장 | `<조각>_sets.npz`(종류, 생성 seed, 색인, 특징, 원소 고정 특징, 문맥, A 지문)를 unit.json 보다 먼저 쓴다 |
| 과제 | 개발 과제 7개만. 그 밖의 과제는 `--xd5`(XD-4 표 열람 뒤)가 있어야 만든다 |

## 7. S9-GBM, S9-DS, 교차검증

- 개발 교차검증 효용: 검증 묶음(과제, 분할, n)마다 후보 200개 가운데 예측 효용이 가장 작은 집합(동률은 작은 번호)의 실제 효용 U(두 가중 평균)를 묶음 평균한 값. 탐욕 정책이 새로 고른 집합은 CatBoost 적합 없이 채점할 수 없어서 이 정의를 썼다. 두 변형이 같은 정의를 쓴다.
- 접기: AL-k(k 1–6)마다 검증 = AL-k, 학습 = 나머지 AL 과제(알래스카 x 는 학습·검증 어디에도 없다). 최종 적합은 개발 과제 전부.
- S9-GBM: CatBoost(학습률 0.05, 깊이 4, l2 3, seed 0, 4스레드). 조기 종료 = 반복 수 {100, 200, 400, 800} 가운데 교차검증 효용이 가장 작은 값(동률은 작은 값). 하이퍼파라미터는 계획이 정하지 않아 구현에서 정했다.
- S9-DS: φ = Linear(29 → 128) → ReLU → Linear(128 → 128) → ReLU, 합·평균 풀링(256), ρ = Linear(256 + 문맥 6 → 128) → ReLU → Linear(128 → 1). 계획의 'φ = MLP(32 → 128 → 128)'의 32 는 계획이 적은 원소 특징 수(29)와 맞지 않는다. 입력 차원은 원소 특징 수로 두고 층 구조(→ 128 → 128)는 계획대로 했다. 문맥 = 과제 문맥 5개 + n. 원소·문맥 특징은 학습 묶음의 평균·표준편차로 표준화한다.
- 손실: RankNet 형, 묶음 안 모든 집합 쌍(U_i < U_j 이면 log(1 + exp(f_i − f_j))), 묶음 평균. 쌍을 표집하지 않고 밀집 행렬로 계산한다(인덱스 누적 연산을 피해 결정성을 지킨다). AdamW(1e-3, 가중 감쇠 1e-4), 한 걸음 = 묶음 8개, epoch 마다 묶음 순서를 seed_of('xd-ds', 접기, seed, epoch)로 섞는다.
- 조기 종료: 접기·seed(0·1·2)마다 50 epoch 를 돌리며 epoch 마다 검증 점수를 저장하고, 세 seed 평균 점수의 교차검증 효용을 접기 평균한 곡선의 최소 epoch(동률은 앞)를 고른다. 최종 = 그 epoch 수로 seed 0·1·2 를 학습하고 점수를 평균한다.
- 결정성: CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True), cuDNN 결정성, seed 고정(xbatch_core.torch_deterministic). 2026-10-04 스모크에서 GPU 0 과 GPU 1 의 학습 가중치와 선택 순서가 비트 단위로 같았다(같은 기종). GPU 는 0–4 가운데 nvidia-smi 의 사용 메모리 500 MiB 이하이고 계산 프로세스가 없는 첫 번호를 쓴다. 21:14 재점검(16절)에서 `setup_device` 가 CUBLAS_WORKSPACE_CONFIG 을 환경 값과 무관하게 `:4096:8` 로 덮어쓰고, CUDA_DEVICE_ORDER=PCI_BUS_ID 로 nvidia-smi 번호와 CUDA 번호를 같게 한 뒤 CUDA_VISIBLE_DEVICES 에 허용 GPU 번호 하나만 두며, 설정 뒤 `torch.are_deterministic_algorithms_enabled()`·보이는 장치 수 1 을 확인하고 어긋나면 거부하도록 고쳤다.
- 동결 선택: 교차검증 효용이 작은 변형, 동률은 S9-GBM(계획 목록 순서). 한 변형의 기록이 없으면 있는 변형으로 정하고 `variants_available` 에 적는다. 동결 기록에 모형 sha256, torch·CUDA·cuDNN·드라이버 판, GPU 번호, seed, 특징 목록, 색인 파일 sha256 을 적는다. 마감(R1a 회수 + 72 h)은 기록의 생성 시각으로 사용자가 판단한다.

## 8. 탐욕 정책

- 첫 셀 = RandomState(seed_of('xd-pol0', 대상, 모드, 분할, 정책 seed)) 무작위. 정책 seed 는 첫 셀만 바꾼다.
- 단계 크기: 5개 단위가 되도록 더한다(1 → 5 → 10 → …). '빈 집합에서 5개씩'과 '정책 seed 가 첫 셀만 바꾼다'를 함께 만족시키는 읽기다. 크기 20·40 의 선택은 한 경로의 앞부분이다(중첩).
- 후보 = farthest-first 상위 300(현재 집합까지 최소 거리 큰 순, 동률은 작은 색인) ∪ 무작위 200(seed_of('xd-polr', 대상, 모드, 분할, 현재 크기), 정책 seed 와 무관). 예측 효용이 작은 순(동률은 작은 색인)으로 그 단계의 수를 한 번에 더한다.
- S9-GBM 은 집합 특징의 증분 계산(features_batch)을, S9-DS 는 L ∪ {c} 의 원소 특징(최근접 거리 갱신 포함)을 후보마다 다시 만들어 한 번에 평가한다.

## 9. 시험 팔(XD-4, 작업 R2b)

- 색인 파일: 항목마다 |A| 와 A 지문(공변량·√TDD·블록의 sha1, 라벨 미사용)을 적고, 단위가 문맥과 대조한다. 색인 파일 sha256 은 사이드카와 동결 기록 둘 다와 대조한다. 동결 기록이 없으면 xd_test 를 거부한다.
- Algorithm P 의 재적합(15절 2 에서 고침): 선택 파일의 후보가 S8 계열이면 `placement_order(후보, Z, 블록, (10, 40, 160), seed_of('xd-S8', 대상, 모드, 분할, 추출))` 의 앞 n 개를 쓴다. 일정은 계획 2.4 의사코드와 XC 의 Algorithm P 팔과 같고 시험 격자(20, 40)가 아니다. S2·S4 는 일정이 없으므로 XD-alg 과 같은 WF8 구현(h54 strategy_sets)을 쓴다. 선택이 S1 이면 AP = S1 이다.
- 시작 전 색인 확인(15절 3): `run_units` 가 단위를 시작하기 전에 색인 파일이 xd_test 의 모든 (별칭, 유효 분할)과 정책 seed 0–4 를 덮는지, n_max 가 시험 격자 최댓값 이상인지 확인한다. 시험 과제(TEST_TASKS)의 항목이 빠지면 거부하고, 서술 과제(Russia_C~lgd)의 항목이 빠지면 그 단위를 'not_indexed' 로 건너뛴다. 확인을 parse 가 아닌 run_units 에 둔 이유: 유효 분할은 자료를 읽어야 정해지고, parse 는 워커마다 다시 불린다. 시작 뒤 단위 안에서 시험 과제 항목이 없으면 SystemExit 가 아닌 RuntimeError(실패 표)로 남긴다.
- 동결 커밋 확인(15절 9): xd_test 본 실행은 동결 기록이 git 에 있고(`git ls-files --error-unmatch`) HEAD 대비 수정되지 않았을 때(`git diff --quiet HEAD --`)만 한다. Rescale 묶음에는 .git 이 없으므로, 로컬에서 커밋 뒤 `--attest-freeze --freeze-manifest <기록>` 이 커밋 확인 기록(`<기록 이름>_commit.json` + sha256 사이드카: 커밋 해시, 동결 기록 sha256)을 쓰고, git 이 없는 환경은 이 기록의 사이드카와 동결 기록 sha256 을 대조한다. 확인 기록은 R2b 묶음에 넣는다.
- 색인 경로: 동결 기록에 저장소 뿌리 기준 상대 경로(index_relpath)를 더하고, 다른 기계의 절대 경로는 'data/processed/' 뒤를 이 저장소 뿌리에 붙여 찾는다(`rebase_path`, Rescale 묶음용).
- v_w: (분할, n)마다 S9 정책 seed p 와 AP 추출 d 의 모든 (p, d) 쌍의 (RMSE(S9_p) − RMSE(AP_d))/RMSE(AP_d)(seed 평균 RMSE) 평균, n 20·40 평균. 계획의 '평균_{분할, 추출, n, 정책 seed}'를 짝 지정 없이 모든 쌍으로 읽었다.
- 러시아 C(Russia_C~lgd): 계획 '서술로만 둔다'에 따라 시험 팔에 넣되(n 20 만) k/5 와 계열 j/3 에는 넣지 않는다(counted False).
- XD-4·5 표는 봉인 폴더에 쓴다(판정어·p 없음, '과제 5개 가운데 k개, 계열 3개 가운데 j개').

## 10. XD-5(XD-4 표 열람 뒤)

- 시험 과제 학습 자료: `--exp xd_learn --dev-tasks <시험 과제> --xd5`. `--xd5`(x_placement_policy 와 x_placement_deepsets 모두)는 봉인 폴더에 XD-4 시험 표 `xd4_summary.json`(접미사 없음)이 있고 `sealed_manifest.json` 에 그 항목이 있어야 한다(표는 열지 않고 존재와 봉인 기록만 본다, 15절 9). 사람이 표를 연 시각은 코드가 확인할 수 없으므로 계획 8절 기록으로 둔다.
- 오라클 대비 후회: (과제, 분할, n)마다 U(S9*)(정책 seed 평균, seed 0), U(AP), 후보 200개의 최소 U. U 의 기준은 학습 자료와 같은 S1 추출(같은 함수·seed 라 같은 예측)이다.
- 계열 하나 제외(검증 = 한 계열의 과제 전부, 학습 = 나머지 계열)와 계열 안 과제 하나 제외(검증 = 시험 과제 하나, 학습 = 나머지 과제 전부)의 교차검증 효용을 동결된 반복 수(S9-GBM)·epoch 수(S9-DS)로 낸다. 두 값의 차이가 과제 의존으로 생기는 낙관의 크기다.

## 11. 열람 순서와 산출 위치(계획 0.3)

| 산출 | 위치 | 열람 |
|---|---|---|
| XD-alg 대비·판정 표(XD-1·2·3, Holm, XD-6) | `sealed/xd_alg_*.csv`, `sealed/xd6_traits.csv` | 비알래스카 행은 S9* 동결 커밋 뒤 |
| 알래스카 계열이 아닌 xd_t·xd_r 조각(PE1, Tibet_LGD, 레나·캐나다 지역 내) | `sealed/shards/` | S9* 동결 커밋 뒤. 계획 0.3 의 '봉인 폴더' 규칙(열람 순서 3·6, 출력 제한)을 원자료 조각에도 적용한다. 집계(`summarize_alg`)와 재현 관문(`--gate`)만 읽고, 선택(`select_algorithm_p`)과 학습 자료(`load_learning`, S9-DS)는 이 폴더를 읽으려 하면 거부한다 |
| 알래스카 계열 xd_t·xd_r 조각(전이 Alaska·AL-1–AL-6, 지역 내 Alaska·AL-1·AL-2) | `shards/` | R1a 회수 뒤(열람 순서 1) |
| Algorithm P 선택(알래스카 계열 행과 고른 후보) | `selection/` | R1a 회수 뒤 바로(열람 순서 1) |
| 학습 자료 조각, S9-GBM·S9-DS 모형·교차검증 기록, 색인 파일 | `shards/`, `policy/` | 개발 과제(알래스카 계열)만 담는다. 효용 값은 화면에 쓰지 않는다. `--xd5` 의 시험 과제 학습 자료는 XD-4 표 뒤에 만들어지므로 `shards/` 에 둔다 |
| XD-4 시험 팔 조각(xdx) | `shards/` | 동결 커밋 뒤에 만들어지고 판정어가 없는 서술이므로 봉인하지 않았다(집계 표는 `sealed/xd4_*`) |
| 동결 기록 | `freeze/` | 커밋 대상 |
| XD-4·5 표 | `sealed/xd4_*`, `sealed/xd5_*` | XD-4 는 R2b 뒤, XD-5 는 XD-4 뒤 |
| 스모크 조각 | `sealed/shards_smoke/` | 열지 않는다(스모크 대상은 알래스카 계열만) |
| 스모크 산출 표(선택·모형 기록 포함) | `sealed/*_smoke.*` | 열지 않는다 |
| 세기 표, 재현 관문 표 | `count/`, `gate/` | 라벨 통계 없음 |

## 12. 로컬 자원과 실행 보호

- CatBoost 1.2.10 은 주소 공간 상한(RLIMIT_AS, `prlimit --as=10G`) 아래에서 가끔 특징 중요도 계산(`_calc_fstr`)에서 멈췄다(2026-10-04 로컬 시험에서 두 번 재현, faulthandler 추적). 상한 없이, 또는 자료 영역 상한(RLIMIT_DATA, `prlimit --data=10G`) 아래에서는 같은 시험이 반복해 통과했다. 이 모듈의 로컬 시험·스모크·세기는 `prlimit --data` 를 쓴다. `systemd-run --user --scope` 는 이 서버에서 실패했다. 계획 1절은 '`prlimit --as` 로 강제한다'라고 적었으므로 다른 모듈의 로컬 실행에도 같은 위험이 있다(남은 문제 1).
- 허용 표지가 없으면 워커는 2개(스레드 2와 함께 코어 4개) 이하, 스레드 4 이하이고, 본 실행·S9-GBM 학습·색인 내보내기와 10,000회 집계·선택은 거부한다.
- 스모크 대상은 알래스카 계열만이다(`SMOKE_TARGETS`: xd_r AL-2, xd_t AL-3·AL-5, xd_learn AL-3·AL-5, xd_test AL-5. 모듈 적재 때와 parse 에서 ⊂ ALASKA_X ∪ ALASKA_R 를 단언한다). 초판의 스모크는 xd_t 에 Russia_W(PE1)를 넣었다(15절 1). 스모크 조각은 `sealed/shards_smoke/` 에 쓴다. 시험 팔 경로는 AL-5 로 지난다.
- 로컬 실행(WF_RESCALE·LG_RESCALE 없음, `--allow-local` 포함)은 main 시작 때 `local_resources` 가 가용 메모리 30 GB 를 확인하고(모자라면 `--mem-wait` 초(기본 1,800) 동안 60 초 간격으로 기다린 뒤 거부) 자료 영역 상한 10 GB(RLIMIT_DATA, `limit_data`)를 건다. 상한은 spawn 워커에 물려진다(작업 하나 10 GB, 워커 2개 합 20 GB). S9-DS 는 스모크를 포함해 시작 때 메모리 30 GB 를 확인한다.

## 13. 세기와 스모크 기록(2026-10-04, 라벨 값 미사용)

- `--count-only --exp xd_r,xd_t,xd_learn,xd_test`: 작업 단위 715(건너뜀 12), 적합 48,241. 실험별: xd_r 168 단위·18,080 적합·0.50 워커·시간, xd_t 427·9,626·0.88, xd_learn 93·19,065·1.10, xd_test 27·1,470·0.20(h54 추정식, 4스레드). 합 2.69 워커·시간, 1차 Rescale 실측 비 0.62 를 적용하면 1.67 워커·시간이다. 계획 2.4 의 추정(3–14 워커·시간)보다 작다. 세기의 학습 집합은 무작위 순열로 대신 세므로 중복 집합의 재사용 수는 실제와 다를 수 있다.
- 스모크(`--smoke --threads 2 --workers 0` 와 `--workers 2`, taskset 코어 4개, `prlimit --data=10G`): 23 단위 + 시험 팔 1 단위가 실패 없이 끝났다(19–24 s, 최대 RSS 311 MB). 걸러진 출력 줄은 0 이다. S9-DS 스모크(GPU 0, 8–22 s, 최대 RSS 4.1 GB)도 끝났다.

- 검토 반영 뒤 스모크(2026-10-04 16:3x, 출력 폴더는 세션 임시 폴더 `XBATCH_OUT_ROOTS` 아래라 저장소의 기존 산출을 건드리지 않았다): `--smoke --threads 2 --workers 0`, taskset 코어 4개, `prlimit --data=10G`, 출력 거름. 23 단위 + 시험 팔 1 단위, 실패 0, 최대 RSS 308 MB. 조각은 모두 `sealed/shards_smoke/` 에 있고 `shards/` 폴더는 만들어지지 않았다. S9-DS 스모크(GPU 0, 최대 RSS 4.1 GB)도 새 경로에서 끝났다. `--count-only`(AL-3·Lena, CA-3·Russia_C~lgd, 분할 1)는 '[자원]' 줄을 쓰고 걸러진 줄이 없었다.

## 14. 남은 문제

1. `prlimit --as` 아래 CatBoost 멈춤(12절). 계획 1절의 강제 방법('`prlimit --as`')과 이 모듈의 방법(RLIMIT_DATA, `prlimit --data`)이 다르다. 계획 개정 사항으로 조정 담당에게 넘긴다(사용자와 조정 담당이 정한다).
2. R1a 의 실행 스크립트(Rescale 단계 순서: 단위 → `--summarize-only` → `--train-gbm --threads 4` → `--export-selection gbm`)는 묶음 담당이 만든다. 이 모듈은 명령만 제공한다.
3. 재현 관문(`--gate`)은 결과 회수 뒤 로컬에서 돈다. WF2·WF8 조각 경로(`results/rescale_wf*/data/processed/wf/shards`)를 기본값으로 둔다.
4. 동결 마감(R1a 회수 + 72 h)과 XD-alg 마감(R1a 제출 + 48 h)은 코드가 시각을 강제하지 않는다. 기록의 생성 시각으로 사용자가 판단한다.
5. 초판 스모크의 비봉인 조각(15절 1): `data/processed/xbatch/XD_placement_policy/shards/` 에 초판 스모크 조각 74개가 있다. 그 가운데 `xdt_smoke__cpu__Russia_W__x__s1__{S1,S2,S4,S8a,S8b,S8c,S8p}_{runs.csv,blocksse.npz,unit.json}` 21개는 PE1 행(R1a 본 결과의 부분집합)이다. 이 파일들은 열지 않았고 옮기거나 지우지도 않았다(사용자 확인 전). 사용자 확인 뒤 조정 담당이 봉인 폴더로 옮기거나 지우고 그 시각을 계획 8절에 적는다. 옮기는 명령(파일을 열지 않는다): `mkdir -p data/processed/xbatch/XD_placement_policy/sealed/shards_smoke_v0 && mv data/processed/xbatch/XD_placement_policy/shards/x*_smoke__* data/processed/xbatch/XD_placement_policy/sealed/shards_smoke_v0/`. 나머지 53개(AL-2·AL-3·AL-5)도 RMSE 열이 있는 스모크 산출이므로 같은 명령으로 함께 옮긴다. 초판 스모크의 봉인 표(`sealed/xd_alg_*_smoke.*`, `sealed/xd6_traits_smoke.csv`)에도 Russia_W 행이 있으나 이미 봉인 폴더 안이다. 이 모듈은 이제 스모크 조각을 `sealed/shards_smoke/` 에서만 읽으므로 옛 파일은 어떤 경로도 읽지 않는다.
6. S8w − S8a 의 해석(15절 5): S8a 의 앵커는 κ 수축 최소제곱(h40 E1)이고 S8w 는 수축하지 않은 E_w 이므로, S8w − S8a 는 계수 추정식의 차이와 수축 제거의 차이를 함께 담는다. 계획 문구를 따랐고 가설은 바꾸지 않았다. 원고의 XD-2 문장에서 이 점을 한계로 적을지는 조정 담당이 정한다.
7. XD-4 의 Algorithm P 가 S2·S4 이면 XD-alg 과 같은 WF8 구현(분할·n·추출마다 seed, S2 는 n 마다 독립)을 쓴다. XC 의 `placement_order`('S2' = 고정 seed 중첩 순환, 'S4' = seed_of('xc-P') k-center)와는 같은 절차의 다른 실현이다(일정이 없으므로 절차는 같다). S8 계열은 XC 와 같은 순서 함수·일정을 쓴다.
8. 동결 커밋 확인은 동결 기록 파일의 커밋만 본다. 계획 0.3 의 '개정 이력에 적고'(계획 문서의 개정 이력 줄)는 코드가 확인하지 않는다.
9. 색인 항목의 |A|·A 지문 불일치는 여전히 단위 안에서 SystemExit 로 작업 전체를 멈춘다(자료가 색인과 다르면 전체 중단이 맞다고 보았다). 미리 확인하려면 시작 전에 문맥을 모두 만들어야 한다.
10. XD-4 시험 팔 조각(xdx)과 `--xd5` 학습 자료 조각은 봉인하지 않았다(11절 표). 계획 0.3 은 이 둘보다 먼저 만들어지는 표만 봉인하도록 정했다.

## 15. 2026-10-04 검토 반영 기록(16:40)

검토 결함 10건을 고치고 각각 계획 문구와 대조했다. 계획은 고치지 않았다. 판정 표·봉인 파일은 열지 않았다.

| 번호 | 결함 | 고친 내용 | 계획 근거 | 시험 |
|---|---|---|---|---|
| 1 | 스모크가 PE1 대상 Russia_W 를 적합하고 조각을 비봉인 `shards/` 에 썼다 | 스모크 대상을 알래스카 계열(`SMOKE_TARGETS`)로 바꾸고 ⊂ ALASKA_X ∪ ALASKA_R 단언, 스모크 조각은 `sealed/shards_smoke/`. 옛 조각 21개는 열지 않고 사용자 확인을 기다린다(14절 5) | 0.3 열람 순서 3·6, 출력 제한 | (x) |
| 2 | XD-4 의 고정된 Algorithm P 를 시험 격자 일정(20, 40)으로 다시 만들었다 | S8 후보는 `AP_SCHEDULE`(10, 40, 160)과 XD-alg seed 로 `placement_order` 를 부른다(`XDXUnit.ap_sets`). 선택 파일 spec 에 `schedule_fixed` 를 적는다 | 2.4 의사코드 '누적 예산 일정 (10, 40, 160)', 2.3 부록 XC-0 | (y): XD-4 AP = XD placement_order = XC 참조 구현 순서, 시험 격자 일정과는 다르다 |
| 3 | S9-DS 색인 기본 과제에 Russia_C~lgd 가 없고, 빠진 항목이 SystemExit 로 R2b 전체를 멈춘다 | DS `--test-tasks` 기본 = TEST_TASKS + TEST_DESC. `run_units` 가 시작 전에 색인 덮임을 확인(시험 과제 결측은 거부, 서술 과제 결측은 not_indexed 로 건너뜀). 단위 안 결측은 RuntimeError | 2.4 시험 과제·'러시아 C 는 서술로만' | (z) |
| 4 | 비알래스카 xd_t·xd_r 조각이 비봉인 `shards/` 에 RMSE 열과 함께 쓰였다 | 그 조각은 `sealed/shards/`. 집계·관문은 두 폴더, 선택·학습 자료는 비봉인 폴더만(봉인 폴더면 거부) | 0.3 열람 순서 3·6(봉인 규칙을 원자료 조각에도 적용) | (n), (h) |
| 5 | S8w·S8r 계수를 κ 10 으로 수축했다 | 수축하지 않은 E_w·E_RE(비유한이면 E0). 설정 해시에 `coef_shrink: none` | 2.4 'E_w = Σ w s y / Σ w s²', '계수만 바꿔 계산' | (k) |
| 6 | XD-1 문장이 '(지역 지역 5/5/5)'가 되었다 | 템플릿 '…작았다({m}).'(m = 풀 표지) | 2.4 사전 고정 문장 | (aa) |
| 7 | XD-2 의 결측이 기각·'확인하지 못했다'가 되고, XD-3 의 행 없음이 미결정, 비열등+열세에서 열세가 빠졌으며, XD-2·3 에 보정 전 유의·0.5 cm 미만 표기가 없었다 | `xd2_verdict`·`xd3_parts`, 모든 문장을 Holm 뒤 compose_sentence 로 | 1절 '행 없음·실패', '해석 문장의 다섯 갈래', '효과 크기 표기', '다중성' | (aa) |
| 8 | 로컬 실행에 메모리 확인·상한이 없었다 | main 시작 때 `local_resources`(30 GB 확인·대기, RLIMIT_DATA 10 GB). `--as` 와 `--data` 의 차이는 14절 1 | 1절 '로컬 자원' | (ac), (m) |
| 9 | `--xd5` 자기 선언, xd_test 동결 기록의 커밋 미확인 | `--xd5` 는 `sealed/xd4_summary.json` 과 봉인 기록 항목이 있어야 한다. xd_test 본 실행은 git 커밋 확인(git 이 없으면 `--attest-freeze` 확인 기록) | 0.3 '커밋', 열람 순서 4 | (u), (ad) |
| 10 | 시험의 빈틈 | (h)에 깨진 `xdr__cpu__Lena__r`·`Canada__r` 조각과 지역 내 행만으로 제외되는 후보(S4)를 더했다. (x)–(ad)를 더했다(스모크 대상, XD-4 AP 순서, 색인 과제 집합과 시작 전 확인, XD-2·3 갈래, XD-4 집계 화면 출력이 '[봉인]' 줄뿐, 로컬 자원, 동결 커밋) | 2.4 '구현·시험' | `tests/test_x_xd.py` 39개 통과 |

시험 실행: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --data=10737418240 taskset -c 100-103 python3 -m pytest -q tests/test_x_xd.py`(39 통과, 약 42–133 s). XC 연결 시험(`tests/test_x_xc.py -k "xd or placement"`, 11 통과)도 다시 돌렸다.

## 16. 검토 반영(재점검, 2026-10-04 21:14)

15절의 반영 작업이 세션 한도로 중단되어, 결함 10건을 계획 문구와 다시 대조하고 코드·시험을 점검했다. 계획·자료·다른 문서는 고치지 않았다. 판정 표·봉인 파일·옛 스모크 조각은 열지 않았다. 동결 모듈 4개의 sha256 앞 16자는 작업 전후 모두 7098f59dabe2b73f, 22e215e435e157be, cf9a1f6d0929c892, 492373e4d37b5ea4 였다.

### 16.1 결함별 재점검 결과

| 번호 | 재점검 | 계획 대조 | 상태 |
|---|---|---|---|
| 1 | `SMOKE_TARGETS` 가 알래스카 계열(AL-2, AL-3, AL-5)만이고 모듈 적재·parse 에서 단언한다. 스모크 조각·표는 `sealed/shards_smoke/`·`sealed/*_smoke.*` 에 쓴다 | 0.3 열람 순서 3·6, 출력 제한 | 코드 반영됨. 시험 (x). 옛 조각 74개(Russia_W 21개 포함)는 `shards/` 에 그대로 있다(16.3) |
| 2 | `XDXUnit.ap_sets` 가 S8 후보에 `AP_SCHEDULE`(10, 40, 160)과 seed_of('xd-S8', …) 를 쓴다. 선택 파일 spec 에 `schedule_fixed` | 2.4 의사코드 '누적 예산 일정 (10, 40, 160)', 2.3 부록 XC-0 | 반영됨. 시험 (y)가 XC 참조 구현과 순서 일치, 시험 격자 일정과 불일치를 확인 |
| 3 | DS `--test-tasks` 기본 = TEST_TASKS + TEST_DESC. `run_units` 가 시작 전 `index_coverage` 로 시험 과제 결측은 거부, 서술 과제 결측은 not_indexed 로 건너뜀. 단위 안 결측은 RuntimeError(실패 표) | 2.4 시험 과제, '러시아 C 는 서술로만' | 반영됨. 시험 (z). 검토가 제안한 parse 단계 확인 대신 run_units 에 둔 이유는 9절 |
| 4 | `unit_sealed`·`shard_dir_of` 로 비알래스카 xd_t·xd_r 조각은 `sealed/shards/`. 집계·관문은 두 폴더(`alg_dirs`), 선택·학습 자료는 `refuse_sealed_rows` 로 봉인 폴더 거부 | 0.3 열람 순서 3·6 | 반영됨. 시험 (n), (h) |
| 5 | `coef_variant` 가 수축하지 않은 E_w·E_RE 를 앵커로 쓴다(비유한이면 E0, 표지). 설정 해시 `coef_shrink: none` | 2.4 'E_w = Σ w s y / Σ w s²', '계수만 바꿔 계산' | 반영됨. 시험 (k) |
| 6 | XD-1 템플릿 '…작았다({m}).', m = 풀 표지 | 2.4 사전 고정 문장 | 반영됨. 시험 (aa) |
| 7 | `xd2_verdict`(열세 → 기각, 레나 행 없음·CI 없음 → 판정 불가, 둘 다 우세 → 지지, 둘 다 동등 → 기각(동등), 그 밖 기각), `xd3_parts`(행 없음 → 판정 불가, 비열등+열세 병기), 모든 문장을 Holm 뒤 `compose_sentence` 로 | 1절 '행 없음·실패', '다섯 갈래', '효과 크기 표기', '다중성' | 반영됨. 시험 (aa) |
| 8 | main 시작 때 `local_resources`(가용 30 GB 확인·`--mem-wait` 대기, RLIMIT_DATA 10 GB). Rescale(WF_RESCALE·LG_RESCALE)에서는 걸지 않는다 | 1절 로컬 자원 | 반영됨. 시험 (ac), (m). `--as` 와 `--data` 의 차이는 14절 1 그대로 |
| 9 | `--xd5` 는 `sealed/xd4_summary.json` 과 봉인 기록 항목이 있어야 한다. xd_test 본 실행은 동결 기록의 git 커밋 확인(없으면 `--attest-freeze` 확인 기록) | 0.3 '커밋', 열람 순서 4 | 반영됨. 시험 (u), (ad) |
| 10 | (h)에 깨진 `xdr__cpu__Lena__r`·`Canada__r` 조각과 r 행만으로 제외되는 후보(S4), (x)–(ad) | 2.4 '구현·시험' | 반영됨 |

### 16.2 이번 재점검에서 더 고친 것(계획의 '특별 주의' 항목)

| 항목 | 고친 내용 | 계획 근거 | 시험 |
|---|---|---|---|
| CUBLAS 설정 | `x_placement_deepsets.setup_device` 가 `CUBLAS_WORKSPACE_CONFIG` 을 `setdefault` 가 아니라 대입으로 `:4096:8` 로 둔다(환경에 다른 값이 있어도 계획 값). 설정 뒤 `torch.are_deterministic_algorithms_enabled()` 와 환경 값을 확인하고 어긋나면 거부한다 | 1절 플랫폼 '`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`' | (g): 환경 `:16:8` 을 준 뒤 `:4096:8` 로 덮어써짐 |
| GPU 번호 | `CUDA_DEVICE_ORDER=PCI_BUS_ID` 를 CUDA 문맥 전에 두어 nvidia-smi 번호(허용 목록 0–4 의 기준)와 CUDA 번호를 같게 한다. `--gpu` 번호가 허용 목록 밖이면 nvidia-smi 를 보기 전에 거부하고, 고른 GPU 도 허용 목록을 다시 확인한다. CUDA 를 만든 뒤 보이는 장치 수가 1 이 아니면 거부한다. torch 가 보는 장치 이름(`gpu_name_torch`)을 기록에 더한다 | 1절 '로컬 GPU 0–4', 2.4 'GPU 번호' 동결 기록 | (ae): nvidia-smi 표를 대리로 두고 CUDA 를 만들지 않은 채 거부 규칙·선택 규칙을 확인 |
| 메모리 대기 | S9-DS 에도 `--mem-wait`(기본 1,800 s)를 두어 x_placement_policy 와 같은 30 GB 대기 규약을 쓴다(이전에는 즉시 거부) | 1절 '30 GB 아래면 기다린다' | (ae) |
| 동결 기록 | `freeze()` 가 S9-DS 기록의 `cublas`, `deterministic`, `device_order`, `gpu_name` 을 동결 기록에 옮긴다(S9-GBM 이면 빈 값) | 2.4 '동결' | (q) 통과 유지 |

### 16.3 고치지 않은 것과 이유

1. 옛 스모크 조각 74개(`data/processed/xbatch/XD_placement_policy/shards/x*_smoke__*`, 그 가운데 PE1 행 Russia_W 21개)는 옮기거나 지우지 않았다. 자료 변경과 계획 8절 기록은 이 작업의 범위 밖(사용자 확인 뒤 조정 담당)이다. 옮기는 명령은 14절 5 에 있다. 현재 코드는 이 파일들을 어떤 경로에서도 읽지 않는다(스모크 조각 폴더는 `sealed/shards_smoke/`).
2. 검토 결함 1 의 '그 시각을 계획 8절에 적는다', 결함 8 의 '`--as`·`--data` 차이의 계획 개정'은 계획 문서 수정이므로 하지 않았다(조정 담당).
3. 결함 3 의 '`parse()` 에서 색인 덮임 확인'은 `run_units` 시작 전 확인으로 두었다(유효 분할은 자료를 읽어야 정해지고 parse 는 워커마다 다시 불린다, 9절). 효과(시작 전 거부, 서술 과제 건너뜀)는 같다.
4. 결함 7 의 XD-2 순서는 '알래스카·레나·캐나다 행에 열세가 있으면 레나 행이 없어도 기각'으로 두었다(4절). 계획 2.4 의 판정 규칙 '충족이면 지지, 그 밖은 기각'에서 열세 행은 그 자체로 불충족이고, 1절 '행 없음·실패'는 지지 쪽으로 세지 않는 규칙이므로 가설·판정 규칙을 바꾸지 않는다. 열세 행이 없고 레나 행이 없거나 CI 가 없으면 판정 불가다.
5. S9-DS 의 GPU 경로에는 RLIMIT_DATA 를 걸지 않았다(CPU 경로만 `limit_data(10)`). CUDA 드라이버의 메모리 매핑 아래에서 RLIMIT_DATA 의 동작을 이 세션에서 시험할 수 없었다(로컬 계산은 합성 시험과 `--count-only` 만). 스모크의 최대 RSS 4.1 GB 는 상한 10 GB 안이다. GPU 경로의 상한 강제는 남은 문제로 둔다.
6. `xbatch_core.torch_deterministic` 의 `setdefault` 는 공용 모듈이라 고치지 않았고, 이 모듈의 `setup_device` 가 먼저 대입해 같은 효과를 낸다.

### 16.4 시험·세기 기록(라벨 값 미사용)

- `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --data=10737418240 taskset -c 100-103 python3 -m pytest -q -p no:cacheprovider -W ignore tests/test_x_xd.py`: 40 passed(41.6 s). 시작 전 가용 메모리 43 GB.
- `tests/test_x_xc.py -k "xd or placement"`: 11 passed(XC 의 `placement_order` 계약).
- `--count-only --exp xd_r,xd_t,xd_learn,xd_test`(출력 폴더는 `XBATCH_OUT_ROOTS` 로 세션 임시 폴더, 저장소의 `data/` 는 건드리지 않았다): 작업 단위 715(건너뜀 12), 적합 48,241. xd_r 168 단위·18,080 적합·0.50 워커·시간, xd_t 427·9,626·0.88, xd_learn 93·19,065·1.10, xd_test 27·1,470·0.20. 합 2.69 워커·시간(실측 비 0.62 적용 1.67). 13절의 값과 같다. 세기 25 s, '[자원] 가용 메모리 43.3 GB · 자료 영역 상한 10 GB 설정' 줄이 찍히고 걸러진 줄은 0 이다.
- 동결 모듈 sha256 앞 16자 재확인: 7098f59dabe2b73f, 22e215e435e157be, cf9a1f6d0929c892, 492373e4d37b5ea4.
- 바꾼 파일: `scripts/3_deep_learning/x_placement_deepsets.py`(setup_device, --mem-wait, 기록 항목), `scripts/3_deep_learning/x_placement_policy.py`(freeze 기록 항목), `tests/test_x_xd.py`((g) 확장, (ae) 추가), 이 기록. git 커밋·push 는 하지 않았다.
