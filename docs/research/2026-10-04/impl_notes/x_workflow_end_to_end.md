# x_workflow_end_to_end 구현 기록(계획 해석과 구현 결정)

- 대상: `scripts/3_deep_learning/x_workflow_end_to_end.py`, `tests/test_x_xc.py`
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 1절·2.3절·2.4절(S8 정의)·2.6절(XF 마감)·부록 A, `docs/research/2026-10-04/placement_and_workflow_algorithm.md` 2·4·5절, `harness_implementation_plan.md` 4.3절
- 작성: 2026-10-04. 같은 날 검토 지적 15건을 반영해 고쳤다(12절). 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리다. 처리는 가장 문자 그대로의 해석을 따랐다.
- 이 기록을 쓰면서 계산한 것: 합성 자료 단위 시험, 라벨 값을 지운 문맥의 적합 수 세기(`--count-only`). 실제 자료의 모형 적합과 결과 표 계산은 하지 않았다. 봉인 폴더의 표와 조각은 열지 않았다(파일 이름만 보았다).

## 1. 계획과 작업 지시가 다른 곳

| 항목 | 계획 | 작업 지시 | 처리 |
|---|---|---|---|
| 시험 파일 이름 | `tests/test_x_workflow.py`(2.3절 '구현·시험') | `tests/test_x_xc.py` | 작업 지시의 이름으로 썼다. 내용은 계획의 (a)부터 (i)까지를 모두 담는다. 묶음 목록(4절)의 `tests/test_x_*.py` 에 둘 다 들어간다 |
| 배치 함수 | `x_placement_policy.py` 의 algorithm_p_order·power_alloc 을 쓴다 | 같음 | 본 실행·집계(`_need_file`)는 `--placement-impl xd` 로 강제한다(auto 는 xd 로 바꾸고 ref 는 거부, XD 모듈을 불러오다 난 예외는 삼키지 않는다). 이 모듈의 참조 구현(2절)은 스모크·세기·시험과 대조(`--placement-check`)용이다 |
| W+ 의 적층 후보 | XB 의 적층 함수(코드 해시를 R1 제출 전에 커밋) | 같음 | 3절의 공급자 계약(`xc_wplus_cv`, `xc_wplus_final`, 선택 `attach_cands_t`)으로 연결했다. 본 실행·집계는 `--wplus require` 로 강제한다(auto 는 require 로 바꾸고 off 는 거부) |

## 2. Algorithm P(배치)

### 2.1 XD 모듈과의 연결
- `x_placement_policy` 의 `xc_placement_order` 또는 `placement_order`(인자 (name, Z, blk, schedule, seed))를 쓴다. 어느 쪽을 썼는지(`placement_impl`)와 XD 모듈 sha1 앞 12자(`placement_sha`)를 설정 해시와 unit.json 에 적는다. 본 실행 전에 XD 모듈이 바뀌면 설정 해시가 바뀌어 `--resume` 이 그 조각을 다시 돈다.
- 대조: 합성 후보 풀 150개 경우(블록 3–29개, 후보 6종, 시험 g4)에서 참조 구현과 XD 의 순서·배분이 모두 같다. 실제 A 풀 대조는 `--placement-check`(라벨 미사용)로 한다. 다르면 XD 쪽이 기준이다. 머리말의 명령 목록에 'R2a 제출 전 `--placement-check` 통과'를 조건으로 적었다.
- 대조 중 찾은 차이 하나를 참조 구현에서 고쳤다: 배분의 소수부가 수학적으로 같은데(예: 40 × 4/48 과 40 × 40/48 의 소수부 1/3) 부동소수 잡음으로 달라 β 순서 동률 규칙이 적용되지 않았다. XD 와 같이 소수부를 소수 12자리로 반올림해 동률로 본다(계획 문구 '소수부 동률은 β 순서 앞 블록 우선'의 뜻에 맞다).
- 부록 XC-0 선택 파일: XD 선택 스크립트의 산출 `data/processed/xbatch/XD_placement_policy/selection/algorithm_p_selection.json`(기본 경로, `--xc0` 로 바꾼다). 후보 이름 키는 `chosen`(XD), `algorithm_p`, `selected` 가운데 있는 것. 값은 S2, S4, S8a, S8b, S8c, S8p 또는 규칙 5 의 S1. 스칼라 항목만 기록하고 근거 표·후보 점수는 읽지 않는다. 파일 sha256 을 설정 해시에 넣는다. 본 실행과 집계는 이 파일만 받고 `--algorithm-p` 는 스모크·세기·시험 전용이다. 파일이 없으면 스모크·세기는 P_default(S8a)에 '부록 XC-0 미확정' 표지를 붙인다.
- 본 실행·집계의 선택 파일 검사(XD 의 `verify_sidecar` 와 같은 수준): `<파일>.sha256` 사이드카가 없거나 다르면 거부한다. 파일이 XD 선택 스크립트의 산출인지 본다. 규칙 1–5 의 산출(`select_algorithm_p`)은 'rule' = '계획 2.3 Algorithm P 의 고정 규칙' 과 alaska_family 항목이 있어야 하고, 규칙 6 의 산출(`select_default`)은 'rule' 이 없고 chosen = S8a, reason 에 'P_default' 가 있어야 한다. `--xc0-sha256 <부록 XC-0 에 적은 sha256 앞 16자 이상>` 을 주면 파일 sha256 과 대조한다. 스모크·세기·시험은 사이드카가 있을 때만 대조한다.

### 2.2 S8 의사코드(2.4절)의 문자 그대로 판에서 정한 세부

| 항목 | 계획 문구 | 처리 |
|---|---|---|
| 표준화 | 중앙값 대체 X − 평균, 표준편차 + 1e-6, 후보 풀 통계 | `h54.standardize_fit`·`standardize`(h40._prep_stats, float32 경유) |
| 첫 블록 | seed 로 고른다 | `RandomState(seed).randint(블록 수)`, 블록 번호 = A 블록 이름의 정렬 순서 |
| 블록 순서 β | 블록 중심에 farthest-first | 동률은 작은 블록 번호(`np.argmax` 의 첫 값) |
| power_alloc 의 상한 | 최대 나머지 배분, 상한 N_b | 상한에 걸린 블록을 N_b 로 고정하고 남은 예산을 나머지 블록에 같은 규칙으로 다시 나눈다(물 채우기). 소수부 동률은 β 순서 |
| 할당이 단계 사이에 줄어드는 경우 | 정하지 않음 | 최대 나머지 배분은 n 이 커질 때 한 블록의 몫이 줄 수 있다. 이미 놓은 라벨은 옮기지 않으므로 그 블록의 수가 q 보다 클 수 있다. deficit 은 max(0, ·)이고 모든 deficit 이 0 이면 β 순서의 앞 블록(남은 셀이 있는)을 고른다 |
| 블록 안 첫 셀 | S8a·c·p: 중심 근접, S8b: farthest-first | S8b 의 맨 첫 라벨(라벨이 하나도 없을 때)은 farthest-first 가 정의되지 않아 중심 근접 셀로 둔다 |
| 블록 안 동률 | 정하지 않음 | 작은 A 색인 |
| 누적 예산 일정 | (10, 40, 160) | XC: 등록 격자 가운데 \|A\| 미만 n. XC-r: (200, 500, 1,000) 가운데 \|A\| 미만 n(지역 내 판의 일정으로 읽었다). 전량은 A 전체라 배치가 필요 없다 |

### 2.3 S2·S4 가 Algorithm P 로 뽑힐 때, 규칙 5(S1)
- 계획은 'Algorithm P 팔은 seed_of("xc-P", 대상, 모드, 분할, d) 로 첫 블록을 정한다'만 적었다. S2 는 h40.draw_blocks 와 같은 절차(블록 순열, 블록 안 순열, 순환 배정)를 이 seed 하나로 n_max 까지 돌린 중첩 판이다(앞 n 개 = n 에서 멈춘 결과). S4 는 h54.kcenter_order 의 첫 점을 이 seed 로 고른다.
- 규칙 5(S1): Algorithm P 라벨 = 무작위 중첩 라벨이다. algP 키를 따로 적합하지 않고 A1·A4·A1+ 는 rand 키를 쓴다. 두 팔이 같은 곡선 키가 되는 대비(Δ ≡ 0)는 모두 '시험하지 않음'으로 둔다(`identical_arms`). 규칙 5 가 이름으로 정한 XC-5b·5c 와 S-XC3 은 규칙 문구 그대로('시험하지 않음(Algorithm P = S1, 부록 XC-0 규칙 5)'), 그 밖의 같은 키 대비(XC-F3·S-XC0 의 n 10 A1 − A2, S-XC8 의 n 10 A1 − A2 와 n 40·160 A1 − A5, F2 의 같은 대비)는 '시험하지 않음(Algorithm P = S1, 라벨 k개 에서 A = B, 부록 XC-0 규칙 5 와 같은 이유)'다. 주 가설은 Holm 가족에 p 1 로 남는다(m 8 유지). A6 은 자료에 따라 정해지므로 넣지 않는다(후회값 0 은 결과다). 9절 4 참조.

### 2.4 seed 의 대상 이름
- 'xc-rand'·'xc-P' 의 대상 인자는 h54 의 규약(LGD 실행 표는 표 안의 이름, 예: Russia_C~lgd → 'Russia_C')을 따랐다. 조각 이름과 저장소 이름은 별칭이다.
- 적합에 넣는 라벨 색인은 정렬했다(같은 집합이면 같은 적합, CatBoost 는 행 순서에 따라 결과가 다르다).

### 2.5 결정성 점검(재현 관문 항목)
- 계획 2.3 재현 관문의 'Algorithm P 의 결정성(같은 seed 두 번 실행 순서 일치)'을 실제 자료 경로에 두었다(`placement_determinism`, 라벨 미사용). XC 대상마다 분할 계획(201–210)의 실행 분할과 추출 0–4 에서 placement_order 를 표준화부터 두 번 계산해 같은지 보고, R2a 조각(`--r2a-shards`, 기본 `--out-dir` 의 shards)의 unit.json 에 있는 orders_head(순서의 앞 10개)와 같은지 본다. 일정과 seed 는 XCUnit 과 같다.
- `--placement-check` 는 XD 대 참조 구현 대조와 이 점검을 함께 하고, `--gate-check` 는 WF4 블록 SSE 대조와 이 점검을 함께 한다. 관문 통과 = 두 항목 모두 통과. 결과 표(`<tag>_placement_determinism.csv`)에는 대상·분할·추출·일치 여부만 있다. unit.json 에서는 orders_head 만 읽는다.

## 3. W+(A1+, 보조)

- 공급자 계약: `x_multisource_stacking.xc_wplus_cv(unit, tr, te, n, d, fold, seed)` 와 `xc_wplus_final(unit, sel, n, d, seed)` 가 {'Stack', 'StackR@0.25'} 예측을 돌려준다. unit 은 XCUnit(h54.TUnit)이고, 라벨은 tr(또는 sel)만 쓰며 학습기 적합은 `unit.fit` 으로 부른다(세기·추적·누설 시험).
- W+ 의 선택은 W 후보 + 적층 후보의 같은 교차검증 SSE 에서 고른다(동률은 W 후보 다음 Stack, StackR). W 는 같은 SSE 의 W 후보 부분에서 고르므로 추가 적합이 없다(구현 보고서 4.3).
- 본 실행·집계는 `--wplus require` 로 강제한다(1절). 스모크·세기·시험의 `--wplus auto` 는 공급자가 있을 때만 켜고, 없으면 A1+ 키를 만들지 않아 S-XC2 가 '행 없음'이 된다.
- XB 의 `attach_cands_t` 는 문맥의 라벨이 표와 같은지 단언하므로, 세기 범주에서 라벨을 지우기 전에 후보 표를 붙인다(`build_unit`). 실제 XB 공급자를 쓴 누설 시험(시험 b3 의 실제 자료 판 `test_b3_leakage_real_xb_provider_one_split`, 캐나다 x 분할 201, n 40, 작은 크기, `XC_RUN_HEAVY=1`)이 통과했다.
- XC-F3 단위(part xcf3)에는 W+ 공급자를 붙이지 않는다. XC-F3 대비(A1 − A2, n 10)에는 W+ 가 필요 없고, 새 지역 실행 표에 XB 후보 열이 없으면 `attach_cands_t` 의 예외로 단위 전체가 실패할 수 있기 때문이다. 설정 해시의 wplus 항목은 '미사용(XC-F3 는 A1 − A2, n 10)'이다.

## 4. 팔과 적합

- A1 의 n 10 은 R1(λ 0.25)이다. 이때 algP 라벨에는 P1·P2(닫힌 형식)와 R1 만 적합하고 R2·D1·규칙 W 는 적합하지 않는다(등록 팔에 쓰이지 않는다).
- 무작위 라벨에는 모든 n 에서 P1, P2, R1(λ 0.25·0.5·1.0), R2, D1 과 규칙 W 를 적합한다. A2·A3·A5 와 A6 의 후보가 모두 여기서 나온다.
- 전량(n = −1)은 두 배치가 같은 라벨(A 전체)이라 placement 'rand' 하나만 저장한다. A1@전량 = A5@전량 = W, A4@전량 = A2@전량 = R1(0.25).
- A6: 그 (대상, n)의 무작위 라벨 W 후보 가운데 셀 가중 RMSE(추출·seed 평균 뒤 분할 평균)가 가장 작은 것. '평균'의 가중을 정하지 않아 프로젝트의 기본 RMSE(셀 가중)로 읽었다. 동률은 W 후보 순서.
- 편향 진단 b10 = mean(E0·s − y)를 n 10 의 각 추출에서 rand·algP 두 라벨 집합 모두에 대해 unit.json 의 diag 에 둔다(시험 (c)). 집계(S-XC6·7·9·12)가 이 값을 읽으므로 unit.json 에 남긴다.
- 재현 관문 단위(`--gate`): 분할 1, 레나 x·캐나다 x, WF4 격자(10, 40, 160, 전량), 추출 5, seed 0·1, 무작위 팔만, 추출 = h40.draw_cells, placement 'cell'. 키 이름이 WF4 조각과 같아 key_map 이 필요 없다. `--gate-check <WF4 조각 폴더>` 가 W 와 R1(λ 0.25) 키의 블록 SSE 를 `XB.gate_compare` 로 대조한다(기본 수준 same_node, `--gate-level elm_hematite|local_rescale`). 같은 명령이 2.5절의 결정성 점검도 한다.

## 5. XC-r(지역 내, 서술)

- 무작위·Algorithm P 의 seed 표지는 XC 와 같다('xc-rand', 'xc-P', 모드 'r'). 계획은 XC-r 의 무작위가 중첩인지 적지 않았다. XC 의 보조 팔이라 XC 의 규약(중첩)을 따랐다.
- 규칙 W(모드 r 판): 계획은 '무작위 + 규칙 W(모드 r 판)'만 적었다. 후보(P0, P1, P2, R1@0.25, R1@1.0, R2@0.25, D1)와 동률 순서, 5겹 블록 교차검증(h42.cv_folds_of, seed 0 적합)은 전이 규칙 W 와 같고, 적합은 h54.RUnit 의 지역 내 적합(선택 라벨과 유사라벨 행, 원천 행 없음)을 쓴다. 교차검증 묶음의 유사라벨 색인 표지는 'wcv{묶음}'이다.
- R1(교차검증 λ)는 WF6 레시피(RUnit.lam_cv)와 같다. 고정 λ 키(0.25·0.5·1.0)도 함께 저장한다.
- 4분 판정만 붙이고 Holm 가족에 넣지 않는다. 대비: Algorithm P + R1 − 무작위 + R1(2단 CI), 규칙 W − R1, R1 − P1, 규칙 W − P1(같은 라벨 집합), Algorithm P + R1 − P1(2단 CI). 레나 n 1,000 의 분할 완결성(7/8)은 h42.pool_rows 의 splits_short 열에 남는다. XC-r 의 알래스카 행은 Algorithm P 선택 계열(지역 내 알래스카)이라 selection_family 열이 '선택 계열'이다.

## 6. 판정과 해석 문장

| 항목 | 처리 |
|---|---|
| 우월 가설(XC-1, XC-5, XC-F3)의 갈래 | 주 2단 CI 의 4분 판정(δ 0.5). '기준 충족'은 별도 열(2단 CI 와 2단 공통 CI 의 상한이 두 가중 모두 0 미만이고 Holm 보정 p < 0.05) |
| 비열등 가설(XC-2)의 갈래 | 주 2단 CI 상한이 두 가중 모두 0.5 미만이면 표의 첫 갈래(비열등 성립 문장), 아니면 열세·동등·미결정. '기준 충족'은 2단·2단 공통 CI 의 비열등과 Holm 보정 비열등 p(단측 p × 2) < 0.05 |
| 문장의 의존 표지 | 갈래는 주 2단 CI 로 정하되, 판정을 말하는 갈래(우세 또는 비열등 성립, 열세, 동등)에서 2단 공통 CI 의 기준 판단(우월 0, 비열등 0.5) 또는 4분 판정이 주 CI 와 다르면 문장 괄호에 '분할 독립 가정 의존'을, 추출 조건부 CI 의 기준 판단 또는 4분 판정이 다르면 '추출 변동 의존'을 넣는다(1절 '주 CI 와 판정이 다르면 붙인다'). 예: 주 2단 CI 로 우세인데 2단 공통 CI 가 기준을 못 넘으면 criterion_met 은 False 이고 문장은 '...작았다(...)(분할 독립 가정 의존).'이다. 같은 표지는 tags 열과 sentence_notes 열에도 있다 |
| '보정 전 유의' | 우세·열세 갈래(비열등 성립 갈래 포함)에서 Holm 보정 p ≥ 0.05 일 때. 비열등 갈래에는 효과 크기 표기('0.5 cm 미만')를 붙이지 않는다 |
| 그 밖의 표지 | '한계 의존', '소수 블록(지역)', '지역 일반 문장 허용(HK CI 가 0 을 제외)'은 tags 열에 둔다 |
| 행 없음·판정 불가 | p 1 로 Holm 에 넣어 m = 8 을 유지한다 |
| XC-2s | XC-2 가 기준을 충족한 n 에서만 시험한다. 우월 기준(2단·2단 공통 CI 상한 < 0, 두 가중)과 우월 양측 p × 그 n 의 Holm 보정 배수 < 0.05. 그 밖은 '시험하지 않음(비열등 불성립)'. 보정 배수는 p 동률(비열등 p 0, 양측 p 하한 1/B)에서 동률 묶음의 첫 순위 배수(m − 첫 순위 + 1)다. Holm 보정 p 는 동률 묶음 안에서 같은 값(누적 최댓값)이고 그 값이 첫 순위 배수에서 나오기 때문이다(`holm_with_ties`, Holm 보정 p ≥ min(1, 배수 × p) 를 단언). 안정 정렬 순위의 배수는 holm_multiplier_stable 열에 남긴다 |
| 최소 검출 효과 | 2.3절 해석 표가 미결정 문장에 최소 검출 효과를 둔 XC-1 과, 검정력 표에서 '검정력 낮음'·'미결정 가능성 큼'인 XC-5b·5c 에 붙인다. 값 = 2.80 × (2단 CI 셀 가중 반폭 / 1.96) |
| 지역 열세 문장 | 풀 문장 뒤에 지역 행 가운데 열세(두 가중 CI 하한 > 0)인 지역을 덧붙인다(XB.compose_sentence 와 같은 형식). 레나 행은 늘 따로 적는다('레나 행: Δ, CI, 판정') |
| XC-2 의 독립 지역 문장 | 독립 지역 7곳(PE1 5곳, 알래스카 x, Tibet_LGD)의 A1 − A3 지역 행(2단 CI)에서 '0.5 cm 넘게 나빴다'를 셀 가중 점 추정 Δ > 0.5 cm 로 읽었다. 블록 등가중 Δ 와 CI 는 S-XC4 (b) 표에 함께 둔다. 문장 틀은 그대로 두고 끝에 '(이 라벨 수에서 평가한 지역 k곳)'을 붙인다(9절 6). 알래스카 x 는 지역 목록에서 'Alaska\|x(선택 계열)'로 쓴다 |
| '같은 지역을 다시 무작위로 나눈 분할에서' | 2.3절 표 머리 문구('모든 문장에')대로 XC-1, XC-2, XC-5, XC-F3 의 모든 갈래에 넣었다. XC-F3 은 새 지역이라 '다시'가 사실과 맞지 않을 수 있다(9절 1). 'F3 0곳' 문장은 시험이 없으므로 넣지 않았다 |
| 공통 문장 | 2.3절 공통 문장을 sentences.json 의 common 에 둔다 |

## 7. 보조 S-XC0–S-XC12

| 항목 | 처리 |
|---|---|
| S-XC4 (a) | WF0 위험표와 같은 정의: RMSE(팔) − RMSE(P0) > 2 cm(셀 가중 점 추정)인 대상 수. A1, A2, A3 × n 10·40·160·전량, 대상 범위 F1·F2·전체 |
| S-XC4 (b) | harm_b 표에 n 별 평가 지역 수(n_regions_evaluated)와 등록 지역 수(7)를 둔다 |
| S-XC6 '최대 n' | 계획은 'A1@최대 n'이다. 설계 보고서 5.4 의 'n 160 또는 최대 n'에 따라 그 대상의 최대 유한 등록 n(160, 40, 10 순)을 주 열로 두고 전량 열을 함께 둔다 |
| S-XC7 | 분할 안 추출 사이 표준편차(ddof 1)의 분할 평균, Algorithm P 라벨과 무작위 라벨 따로 |
| S-XC8 | 3지역 보조 열(AUX3)의 알래스카 x 행에 selection_family = '선택 계열', 그 풀의 평균 행에 '선택 계열 포함(Alaska\|x)'을 둔다(2.3절 대상 표). 같은 표지를 target 열이 있는 모든 표(대비, 손해표, 진단표, 라벨 등가, XC-r)에 붙인다(알래스카, AL-k 의 모든 모드) |
| S-XC9 | 지역 수 R, 지역당 평균 예산 B̄ ∈ {40, 160}, 총 예산 R·B̄. 균등 = 지역마다 B̄, 비례 = n 10 뒤 남은 예산을 \|b10\|(A1 라벨) 비례. 배정값을 그 지역에서 쓸 수 있는 격자 n 가운데 log 거리가 가장 가까운 값으로 맞추고 실제 배정 합을 적는다. 층화 평균 = 지역 A1 RMSE 의 비가중 평균. PE1 과 3지역 보조 열 |
| S-XC11 | (a) A2 곡선(n 10·40·160·전량, 전량의 n = 분할 평균 \|A\|)을 log n 선형 보간해 A1@n 의 RMSE 에 처음 닿는 n, 곡선 범위 밖이면 '범위 밖'. 셀 가중과 블록 등가중 따로. (b) 셀 가중만(하한이 셀 수준 값), `lgx_floor.csv` 의 eval 행 |
| S-XC12 | ρ(\|b10\|_A1, (RMSE(P1@전량) − RMSE(A1@전량)) / RMSE(P0)). 결합 재표집 b: 계열 복원 추출(알래스카·캐나다·레나 계열은 하위 지역 포함), 이득은 h40.contrast 분포의 b 번째 값, \|b10\| 은 (분할, 추출) 복원 재표집. 셀 가중·블록 등가중 따로. 계열 3개 미만 재표집 비율을 적는다(판정어 없음) |
| 지역 하나 제외 평균 | 풀 지역 가운데 하나를 뺀 나머지의 층화 평균과 CI(loo 표) |
| F2 | 하위 지역 14대상(7 × 모드 i·x)의 판정별 대상 수(우세 k, 열세 m, 동등 e, 미결정 j, 판정 불가). detail 의 대상별 기록에 사용 분할의 최소 채점 블록이 5 미만이면 '소수 블록'(부록 A 의 AL-1, AL-3, CA-2, CA-3, LE-1 행), AL-k 이면 '선택 계열'을 괄호로 붙이고, 소수 블록 대상은 few_block_targets 열에도 둔다 |
| 같은 라벨 집합 XC 팔 대비의 보조 2단 CI | S-XC1(A5 − A2), S-XC2(A1+ − A1), S-XC5(A2 − A6), 전량 행(A1 − A2, A1 − A3), S1 일 때의 주 가설 등 두 팔이 XC 팔(A1, A1+, A2–A6)이고 라벨 집합이 같은 대비에 2단 CI(주 2단과 같은 seed 규약)를 ci_lo_2s·ci_hi_2s·ci_lo_beq_2s·ci_hi_beq_2s·verdict4_2s·noninf_2s·superior_2s 열로 함께 낸다. 주 CI 와 판정은 같은 라벨 집합 CI 그대로다(9절 5). P0 가 한쪽인 대비(S-XC10)와 XC-r 팔(rA*)에는 내지 않는다 |

## 8. XC-F3, 집계 상태와 출력

- `--f3 별칭=실행 표 디렉터리[@표 안 대상]` 로 부록 XC-F3 의 새 지역을 등록한다(XB.RUN_TABLES 에 더함, 점 추정 아님). 조각 tag 는 `<tag>f3` 이라 R2a 의 XC 조각과 설정 해시가 섞이지 않는다. 분할 제외 규칙은 1절 그대로(앞선 분할 1–5, 채점 블록 2 미만).
- 적격은 분할 201–210 에서 다시 센 구조 조건(유효 분할 1개 이상, 채점 블록 합집합 8 이상)으로 본다. 라벨 종류·거리·약관 조건은 XF 의 적격 판정(부록 XC-F3)을 따른다. 적격 2곳 이상이면 층화 평균, 1곳이면 그 지역 행, 0곳이면 p 1 과 'F3 0곳' 문장.
- 부록 XC-F3 상태(`--f3-status`): R2a 집계 시점에는 부록 XC-F3 이 정해지지 않았다(XF 마감 T0 + 7일, 2.6절). pending(부록 XC-F3 전)이면 XC-F3 은 갈래 '미정(부록 XC-F3 전)', p 1, 문장은 자리표시이고 'F3 0곳' 문장을 쓰지 않는다. 이때 Holm 순위와 보정 p 는 XC-F3 의 p 가 정해지면 바뀔 수 있으므로 Holm 의존 표(hyp, holm, sentences)의 봉인 파일 이름에 `_provisional` 을 붙이고 meta 에 holm_final = false 를 적는다. 'F3 0곳' 문장과 최종 Holm 표는 부록 XC-F3 이 0곳으로 확정되었을 때(`--f3-status none`) 또는 R4 의 xcf3 조각이 있을 때(`auto` 또는 `final`)만 낸다. auto 는 xcf3 조각이 있으면 final, 없으면 pending 이다. 조각과 맞지 않는 명시값(조각이 있는데 none·pending, 조각이 없는데 final)은 거부한다. Algorithm P = S1 이면 XC-F3 은 F3 상태와 관계없이 '시험하지 않음'(n 10 에서 A1 = A2)이라 Holm 표가 확정이다(0곳 확정이면 'F3 0곳' 문장이 먼저다).
- 재표집 수: 계획 1절은 10,000회를 등록했다. 허용 표지 없는 로컬 집계는 공용 골격의 상한(1,000)으로 줄어든다. 스모크 밖 집계에서 재표집 수가 10,000 이 아니면 모든 봉인 파일 이름에 `_nboot<N>` 을 붙이고 meta 에 registration_deviation('재표집 N회(등록 10000회)')을 적으며 화면에도 그 사실을 쓴다. 등록 이름(xc_hyp.csv 등)은 10,000회 집계만 쓴다.
- NAtlantic~lic 은 2.3절 대상 표에 없어(부록 A 의 참고 행) 기본 대상에 넣지 않았다(`--targets` 로 더할 수 있다).
- 출력 제한: 집계 표는 모두 `data/processed/xbatch/XC_workflow_end_to_end/sealed/` 에 쓴다(재현 관문 통과 전 판정 표 열람 금지, 0.3 의 5). 스모크 집계도 봉인 폴더에 `xc_smoke_*` 로 쓴다. 세기 표(`xc_count.csv`, `xc_count_detail.csv`), 관문 표(`xcgate_gate_<수준>.csv`), 결정성 표(`*_placement_determinism.csv`)에는 RMSE·Δ·판정이 없어 봉인하지 않는다.
- 조각의 결과 열: 조각 runs.csv 에서 rmse_cm·rmse_beq_cm·bias_cm 을 뺀다(집계는 blocksse.npz 만 쓴다). unit.json 의 notes 에서 규칙 W 의 후보별 교차검증 RMSE(w_cv_rmse)를 빼 봉인 폴더 `unit_cv/<조각 기본 이름>_wcv.json` 에 둔다(워커가 동시에 쓰므로 매니페스트 없이 파일 하나씩 원자적으로 쓴다). b10(diag)은 집계가 읽으므로 unit.json 에 남는다. 조각 자체도 결과 파일이므로 재현 관문 전에는 열지 않는다(집계기와 결정성 점검의 orders_head 읽기만 한다).
- 스모크: 계획은 스모크 분할을 정하지 않았다. 평가 분할(XC 201–210, XC-r 211–220)을 쓰지 않는다. XC 는 관문과 같은 분할 1(WF4·LG 가 평가에 쓴 분할), XC-r 은 지역 내 분할 1–25(WF6·WF9 가 쓴 분할) 가운데 1 을 쓰고, 분할 계획(split_plan_201)을 거치지 않고 h40 분할 구조로 문맥을 만든다(분할 계획 경로는 세기가 시험한다). 스모크 조각은 봉인 폴더 `sealed/shards_smoke/` 에 쓴다(계획 0.3 '걸러지지 않은 출력이 필요하면 봉인 폴더').
- 첫 스모크(2026-10-04 16:01–16:03)는 평가 분할(XC 201, XC-r 211)의 첫 유효 분할을 적합해 조각 9개 파일(3단위 × 3파일)을 봉인 폴더 밖 `shards/` 에 썼다. 본 실행 R2a 가 같은 단위를 결정적으로 다시 내므로 주 가설 지역(캐나다 x)의 실제 결과 일부다. 열지 않고(파일 이름만 보았다) `sealed/shards_smoke_s201_legacy/` 로 옮겼고 그 폴더에 README 를 두었다. 같은 조각에서 나온 첫 스모크 집계 표(`sealed/xc_smoke_*`)는 처음부터 봉인 폴더에 있다. 옮긴 조각은 집계에 쓰지 않는다.
- 세기(`--count-only`)는 문맥의 라벨(yA, yB, 원천 y)과 라벨 유래 메타(E0, E_*, y_*)를 지운 뒤 dry 단위로 센다. 배치는 공변량만 쓰므로 실제 순서로 교차검증 묶음 수까지 센다.

## 9. 계획 안의 불일치(구현은 문자 그대로)

1. 2.3절 해석 표 머리의 '모든 문장에 같은 지역을 다시 무작위로 나눈 분할에서를 넣는다'는 XC-F3 문장(새 독립 지역, '독립 지역 확인')과 맞지 않는다. 문자 그대로 넣었고 원고 작성 때 확인이 필요하다.
2. 2.3절의 'n ∈ {0, 10, 40, 160, 전량}'과 팔 표의 '단계 0: P0' 외에 n 0 의 대비는 정의되지 않았다. P0 만 저장한다.
3. 2.3절 '구현·시험' 목록의 split_plan_201 은 공용 골격의 split_plan(부록 A 단언)을 그대로 부른다.
4. 2.3절 고정 규칙 5 는 Algorithm P = S1 일 때 'XC-5b·XC-5c 와 S-XC3 은 시험하지 않음'만 적었다. 그러나 S1 이면 n 10 의 A1 = R1(λ 0.25, 무작위) = A2 라서 XC-F3(A1 − A2, n 10)과 S-XC0 도 같은 키끼리의 대비가 되어 Δ ≡ 0, p 1, 4분 판정 '동등'이 나오고 XC-F3 문장이 '0.5 cm 안에서 같았다'가 된다(합성 자료에서 재현). S-XC8 의 n 10 A1 − A2, n 40·160 A1 − A5 와 F2 의 같은 대비도 같다. 규칙 5 의 이유(두 팔이 같다)를 같은 키 대비 모두에 적용해 '시험하지 않음'으로 두고(2.3절), 주 가설은 p 1 로 Holm 에 넣어 m 8 을 유지한다. 규칙 5 가 이름으로 정한 대비는 규칙 문구를, 그 밖은 '규칙 5 와 같은 이유'를 사유로 적는다.
5. 1절 '대비와 CI(두 팔의 라벨 집합이 다른 대비)' 행은 머리 정의가 '라벨 집합이 다른 대비'인데 '적용: XC 의 모든 팔 대비(A1·A2·A3·A4·A5)'라고 적었다. 앞의 정의를 따르면 같은 라벨 집합인 A5 − A2(S-XC1), A1+ − A1(S-XC2), A2 − A6(S-XC5), 전량 행, S1 일 때의 주 가설은 같은 라벨 집합 CI(h40.contrast, boot_delta_common)이고, 뒤의 적용 문구를 따르면 2단 CI 다. 적용 목록의 팔(A1·A2·A3·A4·A5)은 라벨 집합이 다른 대비(A1 − A2, A1 − A3, A1 − A5, A4 − A2)에 나오는 팔과 같으므로 머리 정의를 주 해석으로 두었다(주 CI·판정·Holm 은 그대로). 대신 같은 라벨 집합인 XC 팔 대비에도 2단 CI 를 보조 열(_2s)로 함께 내어 두 읽기를 모두 표에 남긴다(7절).
6. 2.3절 XC-2 덧붙임 문장은 '독립 지역 7곳(PE1 5곳, 알래스카 x, Tibet_LGD) 가운데 ... j곳'으로 n 과 관계없이 분모가 7이다. 실제로 평가하는 지역은 n 10 에서 7곳, n 40 에서 4곳(레나, 캐나다, Tibet_LGD, 알래스카 x), n 160 에서 3곳(레나, 캐나다, 알래스카 x)이다(부록 A 의 |A|). 문장 틀은 그대로 두고 끝에 '(이 라벨 수에서 평가한 지역 k곳)'을 붙이며, harm_b 표에 n 별 평가 지역 수 열을 둔다.
7. 0.3 은 부록 XC-0 커밋 직후 R2a 를 제출하고, XC-F3 목록은 XF 마감(T0 + 7일) 뒤 부록 XC-F3 에 적는다고 정했다. 그런데 XC-F3 은 주 Holm 가족(m 8)의 한 가설이라 R2a 집계 시점에는 Holm 순위와 보정 p 가 정해지지 않는다(XC-F3 의 p 가 작아지면 다른 7개 가설의 순위와 배수가 바뀐다). R2a 집계 표를 잠정 표(`_provisional`, holm_final = false)로 두고, 부록 XC-F3 확정(0곳) 또는 R4 의 xcf3 조각 뒤의 집계를 최종 표로 한다(8절). 잠정 표는 XC-F3 을 p 1 로 넣었으므로 다른 가설에 대해 보수적이다.
8. 계획 2.3 규칙 6 의 P_default 기록(XD `select_default`)에는 'rule' 항목과 알래스카 계열 항목이 없다. 본 실행의 선택 파일 검사는 규칙 1–5 의 산출과 규칙 6 의 산출을 각각의 형식으로 받는다(2.1절).
9. 계획은 로컬 스모크의 분할을 정하지 않았다. 평가 분할을 쓰면 R2a 의 실제 결과 일부가 미리 생기므로(8절), 평가에 쓰지 않는 분할 1 을 쓴다.

## 10. 로컬 자원(공용 골격에도 해당)

- 1절 로컬 자원 규약의 `prlimit --as=10737418240`(공용 골격 `XB.limit_memory(10)` 도 RLIMIT_AS)는 가상 주소 공간 상한이다. CatBoost 적합을 되풀이하면 glibc 의 스레드별 malloc 아레나 예약 때문에 가상 크기가 커진다. 합성 XC-r 단위 세 번에서 VmPeak 10.8 GB(RSS 0.24 GB)였고, 상한 아래에서 CatBoost 가 `get_feature_importance` 안에서 멈췄다(faulthandler 추적, 시험 b2 가 10분 넘게 끝나지 않음).
- `MALLOC_ARENA_MAX=2` 이면 같은 계산의 VmPeak 가 2.8 GB(RSS 0.22 GB)였다. 이 모듈은 허용 표지가 없는 로컬 실행에서 `limit_malloc_arenas(2)`(mallopt 와 환경 변수)를 `XB.limit_memory` 보다 먼저 부른다. 시험 파일도 불러올 때 같은 설정을 한다. 로컬 시험·스모크 명령에는 `MALLOC_ARENA_MAX=2` 를 함께 준다.
- `systemd-run --user --scope -p MemoryMax=10G`(1절이 먼저 권한 방법)는 이 서버(cgroup v1)에서 작업 생성이 실패해 쓸 수 없었다.
- 다른 x_* 모듈도 같은 상한을 쓰면 같은 멈춤이 생길 수 있다(공용 골격 담당에게 알릴 사항).
- '시작 전 가용 메모리가 30 GB 아래면 기다린다'는 `XB.require_memory(30, wait_s=1800, poll_s=60)` 으로 구현했다(60초 간격으로 최대 30분 기다린 뒤에도 모자라면 끝낸다). 이전 판은 기다리지 않고 바로 끝냈다.
- 허용 표지 없는 실행은 nice 를 10 이상으로 맞춘다(`os.nice` 는 증분이라 현재 값과의 차이만 더한다). 로컬 스모크는 스레드를 2 로 자르고, 시작 환경에서 `XB.smoke_env_report()` 가 경고(코어 묶음 4개 초과, OMP_NUM_THREADS 2 초과)를 내면 중단한다(5절 1a '코어 4개, 2스레드').

## 11. 시험과 세기 실행 기록

- 단위 시험(검토 반영 전): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 prlimit --as=10737418240 taskset -c 116-119 python3 -m pytest -q -p no:cacheprovider tests/test_x_xc.py`(출력은 0.3 의 grep 필터를 거쳤다). 36 통과, 1 건너뜀(실제 자료 적합 시험은 `XC_RUN_HEAVY=1` 일 때만), 118.7 s. 실제 자료 적합 시험은 따로 한 번 돌려 통과했다(캐나다 x 분할 201, n 40, 반복 10, 실제 XB 공급자). 실제 자료를 읽는 시험은 (e) 부록 A, 세기 CLI 두 개, 실제 XB 공급자 시험이며 화면에는 통과 여부만 남는다.
- 단위 시험(검토 반영 뒤, 2026-10-04 21:20 재실행): 같은 명령(코어 116–119). 48 통과, 1 건너뜀(실제 자료 적합 시험, `XC_RUN_HEAVY=1` 전용), 227.8 s. 화면 출력에 RMSE·Δ·판정어가 없음을 `XB.FORBIDDEN_OUT` 패턴으로 확인했다(일치 0줄). 반영 뒤 첫 실행에서 1건이 실패했고(12절 '검증 중 고친 것'), 고친 뒤 전부 통과했다.
- 세기(검토 반영 뒤, `--count-only --threads 1`, 산출은 세션 임시 폴더로 보내 저장소의 `xc_count.csv` 는 그대로 두었다): 작업 단위 208(XC 180, XC-r 28), 건너뛴 분할 32(채점 블록 5 미만 4, 앞선 분할과 같음 12, 여집합 16), 적합 91,135(XC 85,064, XC-r 6,071), 실측 기반 16.29 워커·시간, 워커 22개 약 0.74 h. 반영 전과 같은 수다(Algorithm P = P_default, 배치 구현 xd, W+ 켜짐).
- 세기(`--count-only --threads 1`, 라벨을 지운 문맥, 배치 구현 xd, W+ 켜짐, Algorithm P = P_default 로 셈, 검토 반영 전): 작업 단위 208(XC 180 = F1 68 + F2 112, XC-r 28), 건너뛴 분할 32(채점 블록 5 미만 4, 앞선 분할과 같음 12, 여집합 16), 적합 91,135(XC 85,064, 그 가운데 W+ 의 StackRcv 7,694, XC-r 6,071). WF4·WF6 실측 초/적합으로 16.29 워커·시간, 워커 22개로 약 0.74 h(설치, 단위 크기 불균형, 2단 재표집 집계 제외). h40 의 LG 시간 모형으로는 28.6 워커·시간이고 가장 긴 단위는 0.25 h 다. 계획 2.3절의 예상(약 8만 건, 7–17 워커·시간, 상한 3 h)과 맞는다. 검토 반영으로 적합 수가 바뀌는 곳은 XC-F3 단위의 W+ 제거뿐이다(현재 F3 단위 0개라 세기는 같다). 산출 `data/processed/xbatch/XC_workflow_end_to_end/xc_count.csv`, `xc_count_detail.csv`(셀 수·적합 수만).
- 첫 제한 스모크(검토 반영 전, `--smoke --threads 2 --workers 0`, 코어 4개): 러시아 W x 분할 201, 캐나다 x 분할 201, 캐나다 r 분할 211, 추출 1, seed 1. 실패 0. 집계는 봉인 폴더에 `xc_smoke_*`(15개 파일)로 썼고 화면에는 행 수와 sha256 만 남았다. 이 스모크가 평가 분할을 쓴 문제와 조각 처리는 8절과 12절 1.
- 동결 모듈 sha256 앞 16자는 그대로다(h40 7098f59dabe2b73f, h42 22e215e435e157be, h54 cf9a1f6d0929c892, h41 492373e4d37b5ea4). git 에서 동결 파일과 계획 문서는 바뀌지 않았다.

## 12. 검토 반영(2026-10-04, 검토 지적 15건)

- 지적 목록: 적대적 코드 검토의 결함 15건(심각도 medium 3, low 12). 각 지적을 계획 문구(0.3절 출력·열람 규칙, 1절 공통 규약, 2.3절, 부록 A)와 다시 대조한 뒤 고쳤다. 계획 문서는 고치지 않았다.
- 등록 사항(가설, 팔, 분할, 라벨 격자, CI 방법, 판정 규칙, Holm 가족)을 바꾸는 지적은 없었다. 거절한 지적은 없다. 3번만 규칙 5 의 문구를 넘어 그 사유를 같은 키 대비 전체에 적용한 해석이며, 가설·Holm 가족(m 8, p 1)은 그대로다(9절 4 에 근거를 적었다). 6번은 검토가 제시한 두 처리(거부 또는 표지) 가운데 표지를 택했다.
- 위치 표기: M = `scripts/3_deep_learning/x_workflow_end_to_end.py`, T = `tests/test_x_xc.py`(행 번호는 2026-10-04 21:20 판).

| 번호 | 지적(요지) | 계획 근거 | 처리 | 변경 위치 | 시험 |
|---|---|---|---|---|---|
| 1 | 스모크가 평가 분할(XC 201, XC-r 211)을 적합하고 조각을 봉인 폴더 밖 `shards/` 에 썼다. runs.csv 에 RMSE·편향 열, unit.json 에 교차검증 RMSE 가 있었다 | 0.3 출력 제한('걸러지지 않은 출력은 봉인 폴더'), 0.3 열람 순서 5 | 스모크 분할을 XC 1, XC-r 1 로 바꾸고(분할 계획을 거치지 않는 h40 문맥) 조각을 `sealed/shards_smoke/` 에 쓴다. 조각 runs.csv 에서 rmse_cm·rmse_beq_cm·bias_cm 을 빼고 w_cv_rmse 를 `sealed/unit_cv/` 로 옮겼다. 기존 조각 9개 파일은 열지 않고 `sealed/shards_smoke_s201_legacy/` 로 옮기고 README 를 두었다(8절) | M:44–45(머리말), M:131–134(상수), M:848–849(스모크 조각 폴더), M:950–953(스모크 분할), M:1012–1016(스모크 문맥), M:1048·1056–1058(run_unit), M:1062–1077(strip_result_cols, write_unit_cv) | T:779 test_k, T:802 test_l |
| 2 | R2a 집계 시점에 부록 XC-F3 이 없는데 'F3 0곳' 문장과 최종 이름의 Holm 표를 냈다 | 0.3 2단계 등록(부록 XC-F3 은 XF 마감 뒤), 2.3 XC-F3 행(0곳이면 p 1), 5절 단계 4 | `--f3-status auto\|pending\|none\|final`. pending 이면 XC-F3 갈래 '미정(부록 XC-F3 전)', 자리표시 문장, Holm 의존 표(hyp, holm, sentences)에 `_provisional`, meta holm_final = false. 'F3 0곳' 문장과 최종 표는 none 또는 xcf3 조각이 있을 때만(8절, 9절 7) | M:141–145(상수), M:785–787(인자), M:1453–1456(자리표시), M:1515–1552(tests_xc), M:1933–1962(resolve_f3_status, sealed_names), M:1970–1997(summarize) | T:420 test_f5, T:749 test_j |
| 3 | Algorithm P = S1 이면 n 10 의 A1 = A2 라 XC-F3·S-XC0 이 Δ ≡ 0 의 '동등'이 된다 | 2.3 규칙 5(XC-5b·5c·S-XC3 만 명시), 2.3 'n 10 에서 A1 과 A4 는 같은 예측' | 같은 곡선 키가 되는 대비 전체(XC-F3, S-XC0, S-XC8 의 같은 키 행, F2 의 같은 대비)를 '시험하지 않음'으로 두고 주 가설은 p 1 로 Holm 에 넣는다(m 8 유지). 규칙 5 가 이름으로 정한 대비는 규칙 문구, 그 밖은 '규칙 5 와 같은 이유'(2.3절, 9절 4) | M:146–147(RULE5_TXT), M:1179–1197(identical_arms, untested_reason), M:1527–1541(tests_xc), M:1618–1622(aux_tables), M:1663–1667(f2_counts) | T:502 test_f3 |
| 4 | 문장 갈래가 주 2단 CI 만 보고 정해지고 '분할 독립 가정 의존'·'추출 변동 의존'이 문장에 없다 | 1절 보조 CI 행('주 CI 와 판정이 다르면 붙인다', '두 CI 모두 기준 충족') | 판정을 말하는 갈래(우세 또는 비열등 성립, 열세, 동등)에서 2단 공통 CI 의 기준 판단 또는 4분 판정이 다르면 '분할 독립 가정 의존', 추출 조건부 CI 가 다르면 '추출 변동 의존'을 문장 괄호에 넣는다. criterion_met 은 두 CI 와 Holm 조건을 그대로 본다(6절) | M:1381–1393(dep_flags), M:1418–1438(sentence_deps, _add_notes), M:1445–1460(compose_hyp_sentence), M:1559–1560(tests_xc) | T:468 test_f4 |
| 5 | 본 실행에서도 `--placement-impl auto`·`--wplus auto` 가 기본이라 XD 오류를 삼키고 참조 구현·W+ 없음으로 갈 수 있었다 | 2.3 구현·시험(배치 함수는 x_placement_policy, A1+ 는 커밋된 XB 적층 함수) | `_need_file` 이면 xd·require 로 강제(auto 는 바꾸고 ref·off 는 거부), XD 모듈 예외를 삼키지 않는다. 머리말 본 실행·집계 명령에 두 인자와 R2a 전 `--placement-check` 통과 조건을 적었다(1절) | M:55–64(머리말), M:397–423(xd_placement strict, resolve_placement_impl), M:882–894(enforce_main_impl), M:1086(_worker_init), M:2217(main) | T:825 test_n |
| 6 | 허용 표지 없는 로컬 집계가 재표집을 1,000 으로 줄이고 등록 이름으로 덮어썼다 | 1절 재표집 10,000회 | 스모크 밖에서 재표집 ≠ 10,000 이면 모든 봉인 파일 이름에 `_nboot<N>`, meta 에 registration_deviation, 화면에 알림(8절). 등록 이름은 10,000회 집계만 쓴다 | M:1945–1962(sealed_names), M:1973–1975(summarize) | T:420 test_f5, T:749 test_j |
| 7 | 누설 시험 (b)가 모든 칸의 합집합 밖 라벨만 바꿔 집합 사이 누설을 잡지 못했다 | 1절 누설 시험('선택되지 않은 A 라벨', 칸 단위) | 칸 (배치, n, 추출)마다 그 칸의 선택 집합 밖 라벨을 바꿔 그 칸의 키 예측이 같은지 보는 시험(XC 단위 W+ 가짜 공급자 포함, XC-r 단위). 음성 대조 'cross' 공급자(같은 n·추출의 무작위 집합 라벨을 읽음)는 합집합 판을 통과하고 칸 단위 판에 걸린다. 모듈 변경 없음 | T:115–139(FakeStack leak='cross'), T:209–227(per_cell_leakage) | T:230 test_b3_per_cell, T:253 test_b4 |
| 8 | 재현 관문의 'Algorithm P 결정성' 확인이 실제 자료 경로에 없었다 | 2.3 재현 관문 | `placement_determinism`: 분할 계획(201–210)의 실행 분할·추출마다 순서를 두 번 계산해 일치를 보고 R2a 조각 unit.json 의 orders_head 와 대조한다(라벨 미사용). `--gate-check` 와 `--placement-check` 에 넣고 관문 통과 = 블록 SSE 대조와 결정성 모두(2.5절) | M:59(머리말), M:636–637(orders_head), M:790(--r2a-shards), M:2079–2083(gate_check), M:2086–2137(_shard_heads, placement_determinism), M:2164–2166(placement_check) | T:868 test_p |
| 9 | 3지역 보조 열의 알래스카 x 행에 '선택 계열', F2 대상 기록에 '소수 블록' 표지가 없었다 | 2.3 대상 표(3지역 보조 열), 부록 A(AL-1·AL-3·CA-2·CA-3·LE-1 소수 블록), 1절 소수 블록 | target 열이 있는 모든 표에 selection_family 열(알래스카, AL-k), 풀 평균 행에 '선택 계열 포함(...)', 독립 지역 문장의 'Alaska\|x(선택 계열)', f2_counts detail 의 '소수 블록'·'선택 계열'과 few_block_targets 열(7절) | M:138(SEL_FAMILY_TXT), M:1199–1217(selection_family, mark_family), M:1314–1321(cpool), M:1440–1443·1466(_fam_mark), M:1573–1576(harm_b), M:1604(mark_family 적용), M:1657–1684(f2_counts) | T:381 test_f |
| 10 | 본 실행·집계가 부록 XC-0 선택 파일을 sha256 사이드카 없이 받고 XD 산출인지 확인하지 않았다 | 0.3 2단계 등록(선택 스크립트 해시를 부록 XC-0 에) | 본 실행·집계는 사이드카 필수, XD 산출 형식 확인('rule' = 고정 규칙 문구와 alaska_family, 또는 규칙 6 의 P_default 기록), `--xc0-sha256` 으로 부록 XC-0 의 값과 대조(2.1절, 9절 8). 스모크·세기·시험은 사이드카가 있을 때만 대조 | M:157–159(XC0_RULE, XC0_DEFAULT_MARK), M:229–265(xc0_origin, load_xc0), M:779(인자), M:867–880(resolve_algp) | T:839 test_o |
| 11 | XC-F3 단위(part xcf3)에도 W+ 공급자를 붙여 XB 후보 열이 없는 새 지역 표에서 단위가 실패할 수 있었다 | 2.3 XC-F3(A1 − A2, n 10), A1+ 는 보조 | xcf3 단위는 W+ 없음(관문·XC-r 과 같다). 설정 해시의 wplus 는 '미사용(XC-F3 는 A1 − A2, n 10)'(3절) | M:926–928(unit_cfg), M:1022(build_unit) | T:813 test_m |
| 12 | 메모리 30 GB 하한에서 기다리지 않고 끝냈고, 스모크의 코어 4개·스레드 2·nice 10 을 확인하지 않았다 | 1절 로컬 자원, 5절 1a | `XB.require_memory(30, wait_s 1800, poll_s 60)`, 허용 표지 없는 실행의 nice 10, 스모크 스레드 2, `XB.smoke_env_report()` 경고 시 중단(10절) | M:135–137(상수), M:823–824(finalize), M:2198–2204(set_local_nice), M:2217–2229(main) | T:898 test_q |
| 13 | XC-2s 의 보정 배수가 안정 정렬 순위에서 나와 p 동률에서 Holm 이 실제로 쓴 배수보다 작을 수 있었다 | 2.3 XC-2s('그 n 이 Holm 에서 받은 보정 배수') | `holm_with_ties`: 동률 묶음의 첫 순위 배수(m − 첫 순위 + 1)를 쓰고 Holm 보정 p ≥ min(1, 배수 × p) 를 단언한다. 안정 정렬 배수는 holm_multiplier_stable 열에 남긴다(6절) | M:1477–1504(xc2s_test, holm_with_ties), M:1550–1565(tests_xc) | T:491 test_f6 |
| 14 | XC-2 덧붙임 문장의 분모 '독립 지역 7곳'이 n 40(4곳)·160(3곳)의 실제 평가 지역 수와 다르다 | 2.3 XC-2 덧붙임 문장(문구 고정) | 문장 틀은 그대로 두고 끝에 '(이 라벨 수에서 평가한 지역 k곳)'을 붙이고 harm_b 표에 n_regions_evaluated·n_regions_registered 열을 둔다(9절 6) | M:1465–1470(compose_hyp_sentence), M:1506–1512(indep_worse), M:1567–1576(tests_xc) | T:381 test_f |
| 15 | 1절 2단 CI 의 '적용: XC 의 모든 팔 대비' 문구와 머리 정의('라벨 집합이 다른 대비')의 차이가 기록되지 않았다 | 1절 대비와 CI 두 행 | 머리 정의를 주 해석으로 두고(주 CI·판정·Holm 그대로) 같은 라벨 집합인 XC 팔 대비(S-XC1, S-XC2, S-XC5, 전량 행, S1 일 때의 주 가설)에 2단 CI 를 보조 열(_2s)로 함께 낸다(7절, 9절 5) | M:147(AUX2S_ARMS), M:1219–1258(aux_two_stage, _attach_aux2s), M:1305–1314(cpool) | T:381 test_f |

**검증 중 고친 것(2026-10-04 21:00–21:20, 반영 뒤 첫 실행)**
- `summarize` 의 meta 가 nboot 를 두 번 받아(`sealed_names` 의 dev 에 nboot 가 있는데 meta 에도 nboot=nb 를 넣었다) 집계가 TypeError 로 끝났다. 시험 test_j(조각 → 집계 → 봉인 왕복)가 잡았다. meta 의 nboot 는 dev 의 값만 쓴다(M:1989–1992). 반영 뒤 첫 실행은 이 1건 실패·47 통과였고, 고친 뒤 48 통과다.
- 정리(동작 변화 없음): 쓰지 않는 `warnings` import 를 지웠다. XC-2s 문장 틀을 자리표시만 있는 f-string 에서 일반 문자열로 바꿨다(값은 같다, M:200).

**검증 결과(반영 뒤)**
- 단위 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 prlimit --as=10737418240 taskset -c 116-119 python3 -m pytest -q -p no:cacheprovider tests/test_x_xc.py` → 48 passed, 1 skipped, 227.8 s. 건너뜀 1 은 실제 자료 적합 시험(`XC_RUN_HEAVY=1` 전용). 화면 출력에 `XB.FORBIDDEN_OUT` 일치 줄 0.
- 세기: 작업 단위 208, 건너뛴 분할 32, 적합 91,135(XC 85,064, XC-r 6,071), 16.29 워커·시간(11절). 화면 출력에 금지 패턴 일치 줄 0. 산출은 세션 임시 폴더(`XBATCH_OUT_ROOTS`·`--out-dir`)에 썼고 저장소의 `data/processed/xbatch/XC_workflow_end_to_end/` 는 바꾸지 않았다.
- 최종 실행에 쓴 파일 sha256 앞 16자: 모듈 f412cfa3db58c959, 시험 00e4f29a787b5845, `xbatch_core.py` aba34a3346d86aeb, `x_placement_policy.py` b2ef7464df799ee3, `x_multisource_stacking.py` 8d0d79884fec2b4d(세 공용·이웃 모듈은 다른 묶음의 검토 반영으로 같은 날 바뀌었다. 이 모듈이 쓰는 함수·인자는 모두 있음을 확인했다. `xbatch_core.py` 는 이 반영에서 고치지 않았다).
- 동결 모듈 sha256 앞 16자는 그대로다(h40 7098f59dabe2b73f, h42 22e215e435e157be, h54 cf9a1f6d0929c892, h41 492373e4d37b5ea4). 계획 문서와 기존 자료는 바꾸지 않았다. git 커밋은 하지 않았다.
