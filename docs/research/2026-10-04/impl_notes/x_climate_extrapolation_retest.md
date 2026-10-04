# XI_climate_extrapolation_retest 구현 기록(계획 해석과 구현 결정)

- 대상: `scripts/3_deep_learning/x_climate_extrapolation_retest.py`, `tests/test_x_xi_xj.py`(시험 a–g, p, q–v)
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 2.9절, 1절, 0.3, 7.2-2. `docs/research/2026-10-04/harness_implementation_plan.md` 3.1·4.9절. 공용 골격 `xbatch_core.py` 와 `impl_notes/xbatch_core.md` 6절(warm_trim).
- 작성: 2026-10-04. 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리이며, 모두 가장 문자 그대로의 해석을 따랐다.
- 이 기록을 쓰면서 계산한 것: 합성 자료 단위 시험과 `--count-only`(셀·블록 수, 공변량 √TDD 만). 실제 자료로 모형을 적합하지 않았고 결과 표를 만들지 않았다.

## 1. 판(자료)과 역할

| 판 별칭 | 자료 | 변형 | 역할(표의 role 열) | 근거 |
|---|---|---|---|---|
| Canada~exp~lic | `lgd/run_tables/Canada_expanded_lic`(787셀) | warm_trim | 주(XI-a·b·c) | 2.9 '판 구성', 7.2-2 |
| Canada~exp~lic | 같음 | warm | 민감도(기본 판) | 2.9 '기본 분할 판은 민감도' |
| Canada~exp~lic | 같음 | cold | 서술(XI-d) | 2.9 설계 'warm·cold', XI-d 'cold' |
| Canada~exp | `lgd/run_tables/Canada_expanded`(825셀) | warm, cold | 서술(XI-d 전체판) | XI-d '전체판(825셀)' |
| Alaska | v3 | warm, cold | 서술(재현(비맹검) 행) | 2.9 '알래스카(v3) 행은 재현(비맹검) 행' |
| Canada | v3 | warm, cold | 재현 관문 전용 | 2.9 재현 관문 '캐나다 v3 판을 한 번 돌려' |

- warm_trim 은 계획대로 캐나다 v4 약관 확인분에만 둔다. 전체판·알래스카에는 warm_trim 을 두지 않는다(등록되지 않았다).
- 캐나다 v3 판은 재현 관문에만 쓰고 가설·서술 표(`xi_tests.csv`, `xi_hyp.csv`)에 넣지 않는다(WF10 에서 이미 판정한 대상이다).
- 알래스카 cold 는 h54.wf10_plan 에서 유효 분할이 없다(W 1블록, WF10 과 같다). 건너뛴 기록에만 남는다.

## 2. 설계 세부

| 항목 | 계획 문구 | 처리 |
|---|---|---|
| 저장소 이름 | '저장소 `<대상>~wW`, `<대상>~wI`' | h54 의 이름 규칙 `<대상>~<변형>W|r` 을 따르고 대상 자리에 판 별칭을 둔다(`Canada~exp~lic~warm_trimW|r` 등). v3 판(Alaska, Canada)은 별칭이 대상 이름이라 Vppbeb 조각과 저장소 이름이 같고, 재현 관문은 이 이름으로 키를 대조한다. 계획의 `~wW` 는 줄임 표기로 읽었다 |
| n 격자 | 'n {100, 전량}(캐나다 v4 의 \|A\| 237–383 이라 500 은 없다)', 'WF10 과 같다' | 격자는 h54.WF10_GRID = {100, 500, 전량} 그대로 두고 |A| 미만 규칙(h54.cells_for)이 캐나다의 500 을 뺀다. 알래스카 재현 행에는 500 이 있다(WF10 과 같다) |
| 방법과 단위 | 'WF10 과 같다', 'h54 를 고치지 않고' | h54.R10Unit 을 하위 클래스로 감싸 저장소 이름의 대상 자리만 바꾼다(XIUnit). 방법·추출·seed·교차검증 λ 는 R10Unit 그대로다. 시험 (a)가 별칭 = 대상 이름일 때 R10Unit 과 키·블록 SSE 가 같음을 확인한다 |
| warm_trim | 'A 에서 s ≥ min(s_W) 인 셀을 빼' | xbatch_core.R10TrimUnit(s_W = W 블록의 모든 셀, `impl_notes/xbatch_core.md` 6절)을 같은 방식으로 감싼다(XITrimUnit). 분할 구조는 warm 의 wf10_plan 과 같다 |
| 실행 방식 | 'h54 를 고치지 않고 `--exp wf10 --data-dir <실행 표>` 로 부른다' | h54 의 명령행을 직접 부르지 않고 같은 함수(h54.build_rctx10, wf10_plan, R10Unit)를 판마다 `--data-dir` 를 바꾼 h54 인자로 부른다. 이유: (1) h54 의 집계는 판정 표를 화면에 쓰고 봉인 폴더에 쓰지 않는다(0.3 열람 순서 6 위반). (2) warm_trim 과 판 별칭 저장소 이름은 h54 명령행에 없다. 적합 경로는 같다 |
| 분할 | 'WF10·XI 1–10' | h54.wf10_split·wf10_plan 그대로(분할 1–10, 중복·W·I 채점 블록 2 미만 무효). 새 seed 제외 규칙(부록 A)은 XC 전용이라 적용하지 않는다 |
| 원천 계수 E0 | 정하지 않음 | h54.build_rctx10 그대로(실행 표의 캐나다 밖 원천, 모드 x, 100 km 버퍼). 하네스 보고서 4.9 의 '캐나다 E0 는 v3 와 같다' 와 같은 경로 |

## 3. 판정·사전 규칙·문장

| 항목 | 처리 |
|---|---|
| EP 의 CI | 같은 라벨 집합 대비(두 방법이 같은 추출을 쓴다)이므로 1절의 h40.contrast 를 W·I 저장소에 따로 적용하고 같은 재표집 번호끼리 뺀다(h54.region_ep, WF10-a 규칙). 보조 CI 는 h42.boot_delta_common 의 같은 차. 2단 재표집은 쓰지 않는다(라벨 집합이 다른 두 팔의 대비가 아니다) |
| δ_rel 의 P0 RMSE | EP 행은 총 저장소(W ∪ I)의 P0 RMSE, XI-c 행은 W 저장소의 P0 RMSE. 1절 '그 행과 같은 풀·n·가중의 P0 RMSE' 를 채점 셀 집합 기준으로 읽었다. EP 는 W·I 두 채점 집합의 차라 총 저장소를 썼다 |
| XI-a·b 가설 수준 판정 | n {100, 전량} 의 지역 행 4분 판정을 xbatch_core.rule3(h54._rule3 과 같은 순서)으로 묶는다. XI-a 기준 '열세', XI-b 기준 '우세' |
| XI-c | h54._ni_text(두 가중 CI 상한 < 0.5 cm). 보조 δ 1.0 cm 판(같은 함수, 한계 1.0)을 `verdict_noninf_d10` 에 병기한다. n 별 판정은 비열등 상태(ni_status: h54._ni_text 의 n 별 규칙과 같이 h42.valid_row 거짓이면 판정 불가, 두 가중 CI 상한의 nanmax < 0.5 이면 충족, 그 밖은 미충족)다. 두 판의 비교와 약한 쪽은 이 상태로 하고(아래 '약한 쪽'), 문장 꼬리 '비열등 기준은 …이다'도 약한 쪽 상태로 쓰며 두 상태가 다르면 둘 다 적는다. 문장 갈래(다섯 갈래)는 W 대비 4분 판정의 약한 쪽(verdict4_sentence)이다 |
| frac_W_outside_A | 조각 unit.json 의 값(h54.build_rctx10 의 정의: W 채점 셀 √TDD 가 A 의 최댓값보다 큰 비율, cold 는 최솟값보다 작은 비율. warm_trim 은 xbatch_core.trim_ctx_warm 이 다시 계산)을 실행 분할에서 평균한다. 평균이 0.5 미만이면 그 판의 판정을 '외삽 시험 아님'으로 둔다(0.5 는 통과). 가설 표뿐 아니라 `xi_tests.csv` 의 그 판 행(XI-d 서술 행 포함)도 verdict4, verdict4_d10, verdict4_rel, verdict4_common(과 값이 있으면 verdict4_draw_cond)을 '외삽 시험 아님'으로 덮고, 판정 부속 표기(한계 의존, 분할 독립 가정 의존, 추출 변동 의존, 효과 크기)를 비우며 판정 표지(noninf 등)를 빈 값으로 둔다. CI·p 열은 남기고 `extrap_rule` 열에 적용 여부를 적는다(apply_extrap_rule) |
| 외삽 폭 | warm: W 채점 셀 √TDD 중앙값 − A 의 √TDD 최댓값(계획 문구). cold 는 계획에 정의가 없어 거울 정의(A 최솟값 − W 중앙값)를 썼다. 공변량만 쓰므로 세기 출력과 봉인 밖 구조 표에 둔다 |
| '약한 쪽' | 계획은 순서를 정하지 않았다. 주장의 강도 순서로 정했다. (1) n 별 4분 판정(XI-a·b, XI-c 의 문장 갈래): 우세·열세 2 > 동등 1 > 미결정 0 > 판정 불가·행 없음 −1 > 외삽 시험 아님 −2. 같은 순위의 반대 방향(우세 대 열세)은 미결정으로 쓴다. (2) XI-c 의 n 별 비열등 상태: 충족 1 > 미충족 0 > 판정 불가·행 없음 −1 > 외삽 시험 아님 −2. (3) 가설 수준 판정(rule3, _ni_text 문구의 앞머리): 지지 2 > 부분 지지 1 > 기각 0 > 판정 불가 −1 > 외삽 시험 아님 −2(같은 순위이면 warm_trim 문구). `xi_hyp.csv` 에 n 별 판정(verdict_trim, verdict_basic, verdict_sentence, both_written; XI-c 는 비열등 상태), 4분 판정(verdict4_trim, verdict4_basic, verdict4_sentence, both_written_v4), 가설 수준 판정(verdict_hypothesis, verdict_hypothesis_basic, verdict_hypothesis_written, hyp_both_written)을 모두 남긴다. both_written 류는 갈래(판정어)가 다를 때 참이다 |
| 기본 판이 '외삽 시험 아님' | 문구 그대로 두 판의 판정이 다르므로 약한 쪽(외삽 시험 아님 → 문장 갈래 판정 불가, 사유 '외삽 시험 아님')으로 쓴다. 공변량 세기에서 기본 판(약관 확인분 warm)의 분할 평균은 0.584 로 하네스 보고서 3.1 과 같았다(규칙 통과). 전체판(825셀)은 0.304 로 규칙 미충족이 예상된다(서술 행) |
| Holm | 주 판(warm_trim)의 n 별 p 6개: XI-a·b 는 양측 p(두 가중 가운데 큰 값), XI-c 는 비열등 단측 p × 2(1절). 주 판이 '외삽 시험 아님' 이거나 행이 없으면 p 1 로 넣어 m = 6 을 유지한다. 보조 열이다(4분 판정이 판정) |
| 문장 | 1절 다섯 갈래를 가설별로 쓰고, 모든 문장에 '같은 시기의 따뜻한 블록을 쓴 공간 대용', 외삽 폭(√TDD 차, 주 판의 분할 평균), n 을 넣는다. 두 판의 판정이 다르면 두 판정과 기본 판의 외삽 폭을 문장 괄호에 넣는다. 주 행 6개가 모두 판정 불가 갈래이면 전체 문장은 '기후 외삽은 판정할 수 없었다.' 다 |
| 맹검 표지 | XI-a·b·c 와 캐나다 v4 서술 행 '맹검', 알래스카 행 '재현(비맹검)'. 설계 표지는 모두 '결과 열람 뒤 설계' |

## 4. 출력·봉인·관문

- 판정 표(`xi_tests.csv`, `xi_hyp.csv`, `xi_holm.csv`, `xi_curve.csv`(저장소별 곡선 값), `xi_meta.json`)는 `data/processed/xbatch/XI_climate_extrapolation_retest/sealed/` 에만 쓴다(xbatch_core.write_sealed, 화면에는 행 수와 sha256). 0.3 열람 순서 2 에 따라 작업 R2a 제출 뒤 연다.
- 봉인 밖 표는 라벨 값에서 나온 통계가 없는 것만 둔다: `xi_count.csv`(허용 열 COUNT_COLS), `xi_structure.csv`(셀·블록 수, 공변량 √TDD, frac, 외삽 폭), `xi_timing.csv`(적합 수와 시간), `xi_failed.csv`(식별 열만), `xi_gate.csv`(블록 SSE 차와 범위 결손, 패키지 판), `xi_gate_meta.json`.
- 조각: `runs.csv` 에는 키별 RMSE·편향 열(rmse_cm, rmse_beq_cm, bias_cm)을 쓰지 않는다(public_rows). 이 열로 0.3 열람 순서 2 보다 먼저 warm_trim·기본 판 대비를 계산할 수 있었기 때문이다. 집계는 `blocksse.npz` 의 SSE 만 쓰고, 실패 표는 n_nonfinite·fit_flag 를 쓴다. `unit.json` 의 문맥 메타는 h54.build_rctx10 의 구조 열(셀·블록 수, 공변량 √TDD, 원천 계수 E0)뿐이며 대상 라벨 통계는 없다. 패키지 판(`deps`)을 적는다.
- 재현 관문(`--gate`)은 v3 판 조각과 Vppbeb 조각(`results/rescale_wf3/data/processed/wf/shards/wf10__cpu__*`)의 공통 키 블록 SSE 를 비교한다. 기본 수준은 elm_hematite(0)다. Vppbeb 는 hematite(3차 본 실행), XI 는 elm 실행을 가정했다. 다른 노드에서 돌리면 `--gate-level` 로 1절 수준을 고른다.
- 관문 범위 점검(gate_coverage): 등록 관문 단위는 알래스카 warm, 캐나다 v3 warm·cold 다(2.9 재현 관문. 알래스카 cold 는 유효 분할이 없다). 단위마다 기대 분할(두 쪽 unit.json 의 expected_splits ∩ 실행 분할, 두 쪽 조각의 분할)이 하나 이상이어야 하고, 기대 분할마다 두 쪽 조각과 store_names 의 저장소(총, W, I)가 모두 있어야 한다. 이번 실행의 키는 모두 기준에 있어야 하고(기준 키 없음), 설계 범위(이번 n 격자와 P0 의 n 0, h54 추출 수 규칙, CatBoost seed 수) 안의 기준 키는 모두 이번 실행에 있어야 한다(이번 실행 키 없음). 하나라도 어긋나면 실패다. Vppbeb 조각은 알래스카 warm 1–10, 캐나다 warm·cold 1–10 이다(unit.json 구조 확인).
- 관문 패키지 판: 기준은 `results/rescale_wf3/logs_wf/deps.log` 의 '[deps] 확인' 줄, 이번 실행은 조각 unit.json 의 deps 다. 두 값을 `xi_gate.csv`(deps_ref, deps_new)와 `xi_gate_meta.json`(numpy_mismatch)에 적는다. 허용 오차는 1절 그대로 0 이다(5절 4).

## 5. 불일치와 확인 필요 사항

1. **전체판(825셀)의 묶음과 약관**: 2.9 XI-d 와 7.2-2 는 전체판(`run_tables/Canada_expanded`)을 서술 행으로 등록했다. 그러나 4절 묶음 목록은 실행 표 네 개(Tibet, Russia_C, NAtlantic_lic, Canada_expanded_lic)만 넣고, 약관 점검은 그 네 표에 약관 미확인 셀이 없음을 요구한다. 전체판에는 약관 미확인 38셀이 있다(`lgd_eligibility_lic.csv`). 구현은 문자 그대로 두 규정을 모두 지킨다: 전체판을 판 목록에 두되 실행 표가 없으면 '실행 표 없음'으로 건너뛰고 meta·세기 출력에 남긴다. 전체판 서술 행이 필요하면 묶음에 `Canada_expanded` 를 더하는 결정(약관 점검 범위 밖, 서술 전용)을 개정 이력에 적어야 한다.
2. **적합 수**: 계획 2.9 는 '적합 약 3,400건, 약 0.2 워커·시간' 이다. 이 구현의 `--count-only`(전체 판 목록)는 캐나다 v3 재현 관문(warm·cold)과 전체판을 포함하므로 그보다 많다(보고 수치는 작업 요약에 적었다). 캐나다 단위는 |A| 가 작아 단위당 수 초다(Vppbeb 캐나다 단위 실측 6.0 s, 80건).
3. **저장소 이름 표기**: 1절과 2.9 의 `~wW` 표기와 다르다(2절 첫 행). 판정에는 영향이 없다.
4. **재현 관문과 numpy 판(미해결)**: 1절은 같은 노드 종류와 elm·hematite 사이의 관문 허용 오차를 |ΔSSE| 0 으로 정했다. 그러나 기준인 3차 본 실행 Vppbeb(XI 관문)는 numpy 2.4.6 이었고(`results/rescale_wf3/logs_wf/deps.log`), R1a 는 1절 '패키지 판'에 따라 numpy 2.1.1 을 고정한다(`scripts/rescale/xbatch_constraints.txt`). 1절 근거 기록(WF9 재현 점검 11,814키 최대 0)은 numpy 판이 같은 두 노드 사이의 값이라, numpy 판이 다른 이번 관문에서 차 0 이 보장되지 않는다. 물리식 키(P0, P1, P2)는 numpy 산술만 써서 비트 수준 차가 날 수 있고, CatBoost 키는 입력 행렬이 같으면 numpy 판과 무관할 가능성이 높다[미확인]. 구현은 문자 그대로 허용 오차 0 을 유지하고, 두 쪽 판을 관문 표와 메타에 적는다. 관문이 numpy 판 때문에 실패하면 1절에 따라 판정을 멈추고 원인을 개정 이력에 적는다. 계획 결정(관문 수준 또는 R1a 의 numpy 판)이 필요하다.
5. **묶음 입력**: XI 의 주 판은 `run_tables/Canada_expanded_lic/e5_soil_tdd_v3.csv`(e5_soil_tdd_v4.csv 로 가는 기호 연결)를 읽는다. 4절 묶음 목록의 실행 표 문구에 토양 표 연결이 명시되지 않아 xbatch_core.PAYLOAD_INPUTS 에 네 실행 표의 `e5_soil_tdd_v3.csv` 연결을 더했다(모듈 PAYLOAD_EXTRA 와 시험 r). 묶음을 만들 때 연결을 보존하거나 파일로 풀어야 한다. 허용 표지가 있는 본 실행(스모크 아님)에서 등록 판(Canada~exp~lic, 관문용 Alaska·Canada v3)의 실행 표가 없으면 '실행 표 없음'으로 건너뛰지 않고 중단한다(require_versions). 전체판(825셀)만 1 의 이유로 건너뛸 수 있다.

## 6. 로컬 시험·세기·스모크 기록(2026-10-04, 라벨 값·RMSE·Δ·판정 미출력)

- 단위 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 prlimit --as=10737418240 taskset -c 104-107 python3 -m pytest -q -p no:cacheprovider tests/test_x_xi_xj.py`(XI·XJ 19개 통과, 약 43 s, 출력은 0.3 의 grep 필터를 거쳤다).
- 주소 공간 상한: MALLOC_ARENA_MAX 없이 `prlimit --as=10G` 를 걸면 glibc 할당 영역 때문에 가상 크기가 약 9.1 GB(RSS 약 230 MB)가 되어 CatBoost 의 특성 중요도 계산에서 멈췄다. MALLOC_ARENA_MAX=2 에서는 VmPeak 2.8 GB 였다. 모듈은 MALLOC_ARENA_MAX 가 있을 때만 10 GB 상한(xbatch_core.limit_memory)을 건다(limit_local_memory).
- `--count-only`(전체 판, 공변량·셀 수만): 작업 단위 80(건너뜀 10 = 알래스카 cold 무효), 적합 6,980건, h54 추정식 누적 0.27 CPU-h(4스레드 프로세스). 판·변형별 frac_W_outside_A 분할 평균: Canada~exp~lic warm_trim 1.000(제외 셀 7–24), warm 0.584, cold 1.000, Canada~exp warm 0.304(사전 규칙 미충족 예상), cold 1.000, Alaska warm 1.000, Canada(v3) warm 0.953, cold 1.000. 하네스 보고서 3.1 의 값과 같다.
- 스모크(`--smoke --threads 2 --workers 0`, 코어 4개): Canada~exp~lic warm_trim·warm, Canada(v3) warm 의 분할 1(추출 1, seed 1, n {100, 전량}), 3단위 17 s, 최대 RSS 294 MB. 집계 표는 봉인 폴더의 `xi_smoke_*` 에만 썼다(열지 않았다).
- 재현 관문 경로 점검(`--gate --smoke --gate-level local_rescale`): 스모크의 v3 캐나다 warm 저장소 3개(총, W, I)의 공통 키 81개가 Vppbeb 조각과 같았다(최대 상대 SSE 차 8.1e-16, 셀 수 일치). 본 관문은 Rescale 산출로 다시 한다(기본 수준 elm_hematite).

## 7. 검토 반영(2026-10-04 16시대, 계획 해석은 바꾸지 않았다)

| 검토 항목 | 처리 |
|---|---|
| XI-c 의 약한 쪽을 4분 판정으로 비교하고 문장 꼬리가 주 판만 읽음 | 3절 'XI-c', '약한 쪽'. 비열등 상태로 비교하고 꼬리는 약한 쪽 상태, 다르면 두 상태를 적는다(시험 q) |
| 사전 규칙 미충족 판의 xi_tests.csv 행이 일반 판정을 가짐 | 3절 'frac_W_outside_A'. apply_extrap_rule 로 판정 열을 덮는다(시험 e) |
| 묶음 입력 누락 시 등록 판이 조용히 건너뛰어짐 | 5절 5. require_versions, PAYLOAD_INPUTS(시험 r) |
| 봉인 밖 조각에 키별 RMSE·편향 | 4절 '조각'. public_rows(시험 s) |
| 관문이 범위를 보지 않음 | 4절 '관문 범위 점검'(시험 t) |
| 관문 기준과 R1a 의 numpy 판 차이 | 5절 4(미해결 불일치), 관문 표·메타에 두 쪽 판 기록 |
| 로컬 자원 규칙의 일부 미적용 | 로컬 실행은 모드와 관계없이(세기·스모크·집계·관문·로컬 본 실행) local_resources 를 거친다: MALLOC_ARENA_MAX 가 없으면 명령행은 MALLOC_ARENA_MAX=2 로 다시 시작하고(os.execve) 함수 호출은 거부, 가용 메모리 30 GB 미만이면 60초 간격으로 최대 1시간 대기 뒤 중단, 주소 공간 10 GB 상한(시험 v, p) |
| 스모크 스레드 상한 4 | 허용 표지 없는 스모크는 스레드 2 로 자른다(5절 1a, 시험 v) |
| 가설 수준 판정의 약한 쪽 없음 | 3절 '약한 쪽' (3). verdict_hypothesis_basic, verdict_hypothesis_written, hyp_both_written(시험 q) |

- 다시 돌린 것(2026-10-04 16시대, 라벨 값·RMSE·Δ·판정 미출력): 단위 시험 `tests/test_x_xi_xj.py` 26개 통과(약 47 s, 0.3 grep 필터), `tests/test_xbatch_core.py` 25개 통과. 스모크(`--smoke --threads 2 --workers 0`, 코어 104–107, MALLOC_ARENA_MAX=2): 3단위 17 s, 최대 RSS 292 MB, 앞선 스모크 조각을 새 형식(runs.csv 에 RMSE·편향 열 없음, unit.json 에 deps)으로 덮었다. 봉인 표 `xi_smoke_*` 는 열지 않았다. 관문 경로(`--gate --smoke --gate-level local_rescale`, MALLOC_ARENA_MAX 없이 시작해 다시 시작 경로 확인): 캐나다 v3 warm 분할 1 의 저장소 3개, 공통 키 81, 불일치 0, 범위 결손 0, 통과. 로컬 판(numpy 1.26.4)과 기준 판(numpy 2.4.6)이 달랐으나 local_rescale 허용 오차 안이었다.
