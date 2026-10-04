# x_multisource_stacking 구현 기록(XB_multisource_stacking, 계획 해석과 구현 결정)

- 대상: `scripts/3_deep_learning/x_multisource_stacking.py`, `tests/test_x_xb.py`
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 1절·2.2절·2.3절(W+)·4절, `docs/research/2026-10-04/harness_implementation_plan.md` 3.3·4.2절, `alt_products.md` 8절, `impl_notes/xbatch_core.md`
- 작성: 2026-10-04. 계획은 고치지 않았다. 동결 모듈 4개는 고치지 않았고 `xbatch_core` 를 거쳐 sha256 앞 16자 대조 뒤 불러 쓴다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리이며, 모두 가장 문자 그대로의 해석을 따랐다.
- 이 기록을 쓰면서 계산한 것: 합성 자료 단위 시험, 라벨 값을 쓰지 않는 `--count-only`(적합 수, 셀·블록 수, 라벨 없이 정한 대체 사유 수), 제한 스모크(화면 출력은 출력 제한 필터를 거쳤고 조각과 집계 표는 봉인 폴더에만 썼다. 열지 않았다).
- 2026-10-04 검토 반영(10절): 검토 결함 11건을 고쳤거나 처리 방식을 적었다. 아래 1–9절의 해당 행은 고친 뒤의 구현을 적는다.

## 1. 후보(계획 2.2 '후보')

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| 핵심 후보 | P1, P*, Ku, Ed, Ss, Cr, Ca(K ≤ 7). Pe 는 기준선 팔 | 이 7개를 이 순서로 둔다(순서 = 동률 순서, P1 첫째). Pe 는 후보에 넣지 않는다 | 계획 그대로. Pe = 0.5·P1 + 0.5·Cr 가 정확히 성립한다(Cr 의 무효 셀이 P1 값이므로 Pe 의 무효 셀 값 P1 과 같다). 작업 지시 요약의 'Stefan-CCI mean' 은 기준선 팔로 구현했다 |
| 척도 보정 | c_n = shrink(LS, c0, n, κ 10) | c0 = 원천 셀 최소제곱(ls_E), LS 가 비유한이거나 n = 0 이면 c0. P1 은 h54 의 E_n 과 같은 식·같은 연산 순서 | P1 의 적층 값이 h54 P1 과 비트 단위로 같아야 대체(적층 = P1) 판정이 정확하다(시험 (d)) |
| anchor_fill 범위 | Ku 만 'anchor_fill' 을 명시 | P*, Ku, Ed, Ss 모두 h42.anchor_fill(비유한·0 이하 셀 → ρ·s, ρ = 원천 b/s 중앙값)을 쓴다. SI 에 'anchor_fill 은 척도 앵커 후보 P*·Ku·Ed·Ss 모두에 썼다'를 적는다(모듈 상수 ANCHOR_NOTE, 집계 메타 notes) | 이 후보들은 h42 x9 의 앵커 경로(P1@tddm, P1@ku, P1@ed, P1@soil, `alt_products.md` 8.1 의 근거 키)와 같은 계산이다. 그 경로가 모든 앵커에 anchor_fill 을 쓴다. 계획이 Ku 에만 적은 것은 결측이 실제로 있는 열이기 때문으로 읽었다 |
| P* 의 b | 연도 정합 도일 Stefan | b = √tdd_matched(tdd ≤ 0 은 결측). 값 = `lgx_tdd_matched_v1.csv`(v3) + `xb_tdd_matched_v4.csv`(v4 새 셀, 5절) | h42 anchor_ctx 의 tddm 변환과 같다 |
| P* 결측 규칙 | '대상의 A 또는 채점 셀 가운데 5 % 넘게 P* 가 없으면 그 대상에서 P* 후보를 빼고' | 대상 수준에서 한 번 정한다(`pstar_target_miss`). A 쪽 = 어느 분할에서든 A 풀이 될 수 있는 대상 셀 전체, 채점 쪽 = 대상의 eval_mask 셀 전체. 채우기 전(비유한·0 이하) 결측 비율이 하나라도 0.05 를 넘으면 그 대상의 모든 분할에서 뺀다(같으면 남긴다). 분할 집합에 기대지 않으므로 XC 의 새 seed 분할에도 같은 결정이 나온다. 남긴 대상의 결측 셀은 anchor_fill. 뺀 대상은 xb_t 의 P* 키도 저장하지 않는다. 분할 단위 비율도 xb_cands.info 에 남긴다 | 문구의 '그 대상에서'를 분할마다가 아닌 대상 단위로 읽었다(검토 결함 8). 원천 결측은 규칙에 넣지 않는다(문구가 A·채점 셀만 적었다). 현재 표(v4 표 포함)에서는 211 단위 모두 K = 7 이라 결과가 바뀌지 않는다 |
| Cr·Ca 의 무효 셀 | Cr 은 '무효 셀은 P1 값', Ca 는 정하지 않음 | 유효 = cci_alt 유한 ∩ cci_valid 유한 ∩ cci_valid ≥ 0.5(h54.re_anchor 와 같은 정의). Ca 도 무효 셀은 같은 보정의 P1 값이다. Ca 의 (a, b)는 유효 셀만으로 맞춘다(원천 (a0, b0)도 유효 원천 셀) | Cr 과 같은 규칙을 Ca 에 적용했다. h42 B:cci_aff 는 무효 셀을 ρ·s 로 채운 값에 아핀을 적용했으나 그 행은 Ca 와 다른 기준선이다 |
| Ca 의 수축 | '원천 (a0, b0)에서 κ 10 수축' | a_n = shrink(a_LS, a0, m, κ), b_n = shrink(b_LS, b0, m, κ), m = 선택 라벨 가운데 유효 셀 수. affine_ls 가 비유한(유효 3개 미만 등)이면 (a0, b0) | h42.shrink 의 규칙 그대로 |
| 무효 셀의 P1 | 정하지 않음 | 같은 보정 단계의 P1 값(교차 적합 묶음 j 에서는 L \ L_j 로 보정한 P1, 원천 행에서는 E0·s) | 후보 하나의 값이 다른 단계의 계수를 쓰지 않게 했다 |

## 2. 가중 학습(계획 2.2 '가중 학습' 1–8)

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| '묶음이 2개 미만' | 'F = h42.cv_folds_of(L 의 A 블록). n < 10 이거나 묶음이 2개 미만이면 적층 = P1' | 문구 그대로: F 는 cv_folds_of 가 낸 묶음이고, 그 묶음 수 K 가 2 미만일 때만 대체(folds<2)다. 라벨이 한 블록에만 있어 cv_folds_of 가 셀 묶음(표지 cell_folds, K = min(5, n))을 낸 경우는 그 묶음으로 적층한다(대체가 아니다). 셀 묶음 여부는 적층 기록의 fold_flag 와 세기 표의 n_cell_folds 열(서술)에 남긴다 | 작업 규칙 '가장 문자 그대로의 해석'을 따른다(검토 결함 2). 이전 판은 셀 묶음을 대체로 셌다(문장 'n 10 에서 … 묶음 부족 대체가 많을 것'과 맞추려는 해석). 그 해석은 등록 이탈이므로 쓰지 않았다. 결과적으로 n ≥ 10 에서 K < 2 는 생기지 않는다(cv_folds_of 는 라벨 2개 이상이면 K ≥ 2). 계획의 예상('묶음 부족 대체가 많을 것')과 다르다는 점은 9절 세기와 해석 문장(N10_NOTE: 대체 사유 n<10·묶음<2·γ̂ = 1 과 그 행의 대체 비율)으로 적는다. 셀 묶음 판을 대체로 다시 바꾸려면 사용자 승인과 등록 이탈 기록이 필요하다 |
| 대체 사유 | 'stack_fallback(사유: n, 묶음, γ̂ = 1)' | runs 의 stack_fallback 열에 `n<10`, `folds<2`, `gamma=1` 을 적는다(Stack·StackR 행). unit.json 의 stack 목록에 추출마다 사유, γ̂, ŵ, w*, 지지 집합, 묶음 수를 적는다 | 계획 그대로 |
| 단체 제약 최소제곱 | '지지 집합 2^K − 1 개를 모두 풀어 음수가 없는 해 가운데 SSE 가 가장 작은 것. 동률은 P1 을 포함하고 원소가 적은 지지 집합' | 지지 집합마다 마지막 원소를 기준으로 한 차분 최소제곱(등식 제약의 닫힌 해, numpy lstsq). 차분 행렬의 계수가 모자라면(해가 하나가 아님) 그 지지 집합을 건너뛴다. 음수 허용 −1e-12, 동률 허용 상대 1e-10. 동률이 남으면 색인 사전 순 | 계수가 모자란 지지 집합의 SSE 는 더 작은 지지 집합이 낸다. 허용치는 플랫폼 사이 마지막 자리 반올림 차이가 선택을 바꾸지 않게 하는 값이다 |
| γ 의 2단 교차검증 | '같은 묶음의 2단 교차검증. SSE 가 가장 작은 γ, 동률이면 큰 γ' | 구현 보고서 3.3 의 절차 그대로: 묶음 j 를 뺀 Z 행으로 w_j = simplex_ls 를 맞추고 (1 − γ)·w_j + γ·e_P1 로 묶음 j 의 Z 행을 채점한다. 1단 = 교차 적합 Z, 2단 = 가중의 묶음 교차검증. SI 에 비중첩임을 적는다(모듈 상수 GAMMA_CV_NOTE, 집계 메타 notes) | 계획이 인용한 구현 보고서의 문장을 따랐다. Z 의 다른 묶음 행은 묶음 j 의 라벨이 들어간 계수로 만들어졌으므로 엄밀한 중첩 교차검증은 아니고, y_j 가 w_j 에 들어가 γ 선택의 채점이 작은 γ(적층 쪽)로 낙관적일 수 있다(검토 결함 7). 중첩판(묶음 j 를 뺀 라벨 안에서 Z 를 다시 교차 적합)은 등록되지 않아 넣지 않았다. 바꾸려면 사용자 승인과 등록 이탈 기록이 필요하다. 대상 밖 라벨은 어느 쪽도 쓰지 않는다 |
| w* 와 대체 | w* = (1 − γ̂)·ŵ + γ̂·e_P1 | γ̂ = 1 이면 w* = e_P1 으로 두고 대체(gamma=1)로 적는다. 적층 예측은 0 이 아닌 가중만 더하고, w* = e_P1 이면 P1 예측을 그대로 쓴다 | '정확히 같음'(시험 (d))을 0·NaN 이나 반올림 없이 보장한다 |
| StackR 학습 행 | 라벨 행 y − 적층(표본 안, 최종 계수와 w*), 전이에서는 원천 행 y − 적층(원천 계수 c0, w*) | 그대로. 적층 = P1(대체이거나 ŵ = e_P1)이면 학습 행렬이 R1 과 같으므로(원천 c0 = E0, 라벨 E_n) 같은 seed 의 R1 잔차 성분을 다시 쓰고 fit_flag 에 reuse_R1 을 적는다 | 같은 행렬·seed 의 CatBoost 적합은 같은 결과를 낸다. 적합 수가 준다(세기는 라벨 없이 정한 대체만 재사용으로 센다) |
| StackR 의 교차검증 λ(지역 내) | '교차검증 λ' | h54.RUnit.lam_cv 와 같은 묶음(cv_folds_of, 셀 묶음 포함)·seed 0 적합·동률은 작은 λ. 묶음 j 마다 학습 쪽 라벨로 적층을 다시 맞추고(안쪽 블록 묶음 seed 의 추출 표지 '추출.묶음', h54 Pbest 의 관례), 잔차 목표 = y − 적층(표본 안) | h54 R1 의 교차검증 λ 가 묶음마다 E_n 을 다시 구하는 것과 같은 구성이다 |
| StackR 저장 키 | λ 0.25 와 교차검증 λ | 지역 내는 h54 emit 과 같이 λ 0.25·0.5·1.0·교차검증을 저장한다(0.5·1.0 은 보조 키). 전이는 λ 0.25 만 | 같은 적합에서 나오므로 비용이 없다 |
| Stack0 | '원천 행에서 원천 거시 지역 하나 제외 교차 적합으로 가중을 맞춘다' | 원천 거시 지역(h40 macro_src)마다 그 지역을 뺀 원천 행의 최소제곱 계수(scale: ls_E, affine: 유효 셀 아핀, 수축 없음)로 그 지역 행을 예측해 Z_src 를 만들고 simplex_ls 를 맞춘다. γ 수축은 없다. 채점 셀 예측은 원천 전체 계수 c0. 거시 지역이 2개 미만이면 P1 | 문구에 γ 가 없다. 수축의 사전값이 될 라벨 계수가 n 0 에는 없다 |
| B:ens | 'B:ens(등가중)' | h42 x9 의 N1 정의: (E0·s + c0_ku·ku + c0_cci·cci)/3, ku 와 cci 는 anchor_fill, c0 = 원천 최소제곱 | LGX L29 의 기준선과 같은 정의 |
| 누설 규약 | 'B 라벨과 선택되지 않은 A 라벨은 가중, 계수, γ, 잔차 어디에도 쓰지 않는다. anchor_fill 의 ρ 는 원천에서만' | 후보 표(CandSet)는 대상 라벨을 담지 않는다. 보정·교차 적합·γ·StackR 은 호출 때 선택 라벨 색인만 읽고, 쓴 색인을 h54 추적(WF_TRACE)에 남긴다 | 시험 (b)(c)(e)(j) |

## 3. 팔·키·조각

- xb_r: h54.RUnit 하위 클래스(XBRUnit). 추출 h40.draw_cells(대상, 'r', 분할, n, 추출), 분할 1–25 의 구조는 h54.split_structure_ext + split_plan(WF6 규칙: 레나 분할 24 무효). P1 은 h54.physics 와 같은 키·값, R1·Re 는 WF6 와 같은 함수·seed(lam_cv, fit_resid, emit). 개별 보정 후보는 서술 키 `P*`, `cand_Ku`, `cand_Ed`, `cand_Ss`, `cand_Cr`, `cand_Ca` 로 저장한다.
- xb_t: h54.T7Unit 하위 클래스(XBTUnit). 문맥은 xbatch_core.build_tctx(h40.build_ctx, 실행 표 별칭 포함), 추출·유사라벨 색인은 LG·WF7 과 같다. P0·P1·Pe·R1(0.25)·Re(0.25)는 T7Unit 과 같은 행렬·seed. 대상 = LG 27 (대상, 모드)(h40.default_targets(i, x)) + Tibet_LGD, Russia_C~lgd, NAtlantic~lic(점 추정), Canada~exp~lic(민감도).
- 저장소 이름: 핵심 판 `<대상>|<모드>`(WF6·WF7 과 같아 재현 관문의 키 대조가 바로 된다), 변형 판 `<대상>~<변형>|<모드>`. 조각 이름 `xbr__cpu__<대상>__r__s<분할>[__<변형>]`, `xbt__cpu__<대상>__<모드>__s<분할>[__<변형>]`(1절 '산출 경로'). 출력 경로 `data/processed/xbatch/XB_multisource_stacking/shards/`.
- runs 의 추가 열: stack_fallback, stack_gamma, stack_K, stack_w(0 이 아닌 가중의 JSON). unit.json 에 stack(추출별 기록), xb_cands(후보 이름, K, 뺀 후보와 사유, ρ, c0, P* 결측 비율, pre2010 비율, tdd 표 출처), stack0(가중, 제외 후보), K, pstar_dropped, pstar_target, tdd_src, tdd_v4_missing_allowed, resid_src_excluded_km5, runs_sealed_cols, 스모크이면 smoke_env 를 적는다.
- 조각 runs.csv 에는 rmse_cm, rmse_beq_cm, bias_cm 열을 쓰지 않는다(`public_runs`, RUNS_SEALED_COLS, 검토 결함 10). h54 UnitBase.add 가 키마다 이 열을 채우므로 봉인 밖 조각에서 Stack − P1 방향을 읽을 수 있었다(계획 0.3 열람 순서 6). 집계(블록 SSE 저장소, 대체 비율), 재현 관문, --resume 은 blocksse.npz 와 나머지 열만 쓴다. 조각 경로는 계획 1절 '산출 경로'의 `data/processed/xbatch/<새 이름>/` 를 그대로 둔다.
- --shard i/N: 단위 목록을 이름 순으로 정렬해 k % N = i − 1 인 단위만 돈다(N 개 작업이 겹치지 않고 모두를 덮는다. 시험 (m)). h54 에는 없는 선택이며 작업 지시의 '--shard' 를 이렇게 정했다.
- 재현 관문(--gate): xb_r 의 P0·P1·R1(λ 0.25·교차검증)과 WF6 조각(`results/rescale_wf2/data/processed/wf/shards/wf6__*`), xb_t 의 P0·P1·R1(0.25)·Re(0.25)와 WF7 조각(같은 폴더 `wf7__*`)의 블록 SSE 를 xbatch_core.gate_compare 로 대조한다(기본 허용 수준 same_node = 0). 표에는 |ΔSSE| 만 있다.

## 4. 변형(2단계 cci5y, 3단계 확장)

- --variant cci5y|ext 와 --product 이름=경로[:열]. 제품 표 형식(XG 표집 표의 인터페이스로 정했다): `loc_id`, 값 열(기본 `value`, ALT cm), `mask_ok`(1 = XG 누설 마스크 통과), `train_dist_km`(가장 가까운 제품 학습 지점까지의 거리). Wei(L1, 마스크·5 km 규칙 제품)는 두 열이 반드시 있어야 하고 없으면 중단한다(`_read_product`, 검토 결함 3). L0 제품(CCI v5, YK)은 XG 마스크가 없으므로(`alt_products.md` 7.7 '주 마스크') mask_ok 가 없으면 모든 셀이 통과다(PRODUCT_MASK_REQUIRED).
- 모든 제품은 아핀 보정(원천 (a0, b0) → κ 10 수축, `harness_implementation_plan.md` 3.3 표의 '공개 ALT 제품 | 아핀').
- cci5y: 결측 셀은 P1 값(CCI v4 의 Cr·Ca 와 같은 규칙).
- Wei: A 풀과 채점 셀이 모두 마스크를 통과하고 값이 있을 때만 그 대상의 후보로 넣는다(`alt_products.md` 8.3 '대체값을 넣지 않는다'). Stack0 후보에서 뺀다. 확장 판 StackR 의 원천 행 가운데 train_dist_km < 5 인 행을 잔차 학습에서 뺀다. 모든 원천 행의 train_dist_km 가 유한해야 하고(표에 없는 원천 행도 비유한으로 센다) 아니면 중단한다. 5 km 제외는 Wei 표를 준 확장 판 전체의 StackR 에 적용한다(그 대상에서 Wei 가 마스크로 빠져도 적용한다. 문구 '확장 판의 StackR 원천 행 가운데 … 그 판의 잔차 학습에서 뺀다'의 문자 그대로의 범위). 뺀 행 수는 unit.json 의 resid_src_excluded_km5 와 xb_cands.km5_src_excluded_Wei 에 적는다. 원천 계수 (a0, b0)는 유한한 원천 셀 전부로 맞춘다(계획이 정한 두 규칙 밖의 원천 셀 제외는 넣지 않았다).
- YK(Yi·Kimball): 알래스카 계열(상위 지역 Alaska)만. 결측 셀이 A 풀이나 채점 셀에 하나라도 있으면 그 대상에서 뺀다. L0 라 XG 마스크는 표에 mask_ok 가 있을 때만 적용한다. 5 km 규칙은 없다.
- 확장 판은 원천 행 마스크 때문에 적층 = P1 이어도 R1 성분을 다시 쓰지 않는다.

## 5. 1c 입력: v4 새 셀 tdd_matched(--write-tdd-v4)

- 정의: `scripts/2_evaluation/a2_year_matched_tdd.py` 2) 단계를 같은 연산 순서로 옮겼다(최근접 격자, ±2 → ±5셀 유클리드 육지 폴백, 연별 TDD = Σ max(T, 0)·그 해 일수, matched = [year_min, year_max] ∩ [2010, 2024] 평균, 2010 이전은 2010–2014 평균과 pre2010 표지, 일부만 이전이면 partial). a2 스크립트는 모듈 최상위에서 인자를 읽고 학습을 돌리므로 불러 쓰지 않았다.
- 대상 셀: 네 실행 표(Tibet, Russia_C, NAtlantic_lic, Canada_expanded_lic)의 F4_direct 셀(load_base 가 쓰는 셀) 가운데 `lgx_tdd_matched_v1.csv` 에 없는 셀. 관측 연도는 `fidelity_base_v4_labels.csv` 의 year_min·year_max(라벨 값은 읽지 않는다).
- 재현 점검: 같은 코드로 v3 셀 400개(seed_of('xb-tdd-check'))를 다시 계산해 `lgx_tdd_matched_v1.csv` 와 최대 |차| ≤ 1e-6 °C·day, 표지 일치를 확인해야 쓴다.
- 산출 `data/processed/xbatch/XB_multisource_stacking/inputs/xb_tdd_matched_v4.csv`(v3 셀은 a2 셀 표의 값·표지, 새 셀은 계산값)와 `_meta.json`. XB 는 값을 v1 표에서 먼저 읽고 새 셀만 이 표에서 채우며, pre2010 비율은 이 표의 표지로 unit.json 에 적는다.
- 묶음(4절): `xbatch_core.PAYLOAD_INPUTS` 에 이 표와 `_meta.json` 을 더했다(검토 결함 1, 공용 골격의 다른 줄은 고치지 않았다). 모듈 상수 PAYLOAD_EXTRA 에도 같은 두 경로를 두었다.
- 입력 점검: `tdd_tables` 는 v4 표가 없거나 sha256 앞 16자가 기록값 `4b22ca458685950c`(TDD_V4_SHA16)와 다르면 중단한다. `--allow-missing-tdd-v4` 를 줄 때만 표 없이 진행하고(LGD 4대상의 P* 가 빠져 그 20 단위의 K 가 7 에서 6 으로 준다), 출처 dict(설정 해시에 들어간다)와 unit.json 의 tdd_v4_missing_allowed 에 적는다. main 은 세기·스모크·본 실행·집계 전에 `check_inputs` 로 이 점검을 하고 출처와 해시만 화면에 쓴다. XC 의 W+ 공급자(attach_cands_t)도 같은 점검을 거친다(XC 인자에는 허용 표지가 없다).

## 6. 집계(봉인)와 판정 구현

- 표: `<out-dir>/sealed/` 에 xb_tests, xb_weights(추출별), xb_weights_summary, xb_cci_relation, xb_fallback, xb_group_error(저장소·곡선 키별 오차), xb_meta. 화면에는 xbatch_core.write_sealed 의 행 수와 sha256 앞 16자만 남는다. 스모크 조각과 표도 봉인 폴더(`sealed/smoke_shards`, `sealed/smoke_*`)에 둔다.
- XB-1·XB-2: 주 4지역(모드 x), 같은 라벨 집합 대비(xbatch_core.contrast_pool kind same). 등록 지역 수는 n 10·전량 4, n 40·160 2(레나·캐나다). 판정 = 층화 평균의 4분 판정, 가설 종합 = _rule3(우세).
- XB-3: 지역 내 3대상 층화 평균, StackR(교차검증 λ) − R1(교차검증 λ), 등록 지역 3('지역 k/3').
- XB-4: 알래스카 지역 내 지역 행의 비열등(두 가중 CI 상한 < 0.5 cm). 등록 n 모두 비열등이면 지지, 일부면 부분 지지(판정 불가 n 을 덧붙인다), 비열등이 하나도 없을 때 판정할 수 없는 n(행 없음, 판정 불가)이 있으면 '판정 불가(사유)', 그 밖은 기각(검토 결함 4. 1절 '행 없음·실패 … 판정 불가 갈래'와 _rule3 의 '판정할 수 없는 대비가 있고 충족이 없음 = 판정 불가'). 판정 불가 행의 문장은 '비열등을 확인하지 못했다' 가 아니라 '적층이 작동하지 않아(대체 비율 r) 판정할 수 없었다' 또는 '비열등을 판정할 수 없었다(사유)' 다.
- 변형 조각: 집계(`load_tms_variant`)와 재현 관문은 이 변형(--variant)의 조각만 읽는다(조각 이름 여섯째 마디, `variant_shards`). 핵심·cci5y·ext 조각이 한 폴더에 있어도 설정 해시 충돌로 멈추거나 --allow-mixed-cfg 에서 변형 저장소가 핵심 XB-5 대상별 행과 K 표에 섞이지 않는다(검토 결함 6). xbatch_core.load_tms 와 같은 경로(load_stores, h54.make_tm, h54.units_for)를 걸러낸 목록으로 밟는다.
- 퇴화 추출 규칙: runs 의 Stack 행에서 (저장소, n)의 대체 비율을 센다(세 사유 모두). 50 % 를 넘는 지역 행은 '판정 불가(적층 미작동)'로 두고 그 지역을 풀에서 뺀다(풀 표지가 '부분(지역 k/m)'이 된다). 풀 행에는 풀 전체의 대체 비율을 적는다. 문구가 (대상, n) 행만 정했으므로 풀 행 자체를 비율로 판정 불가로 두지는 않았다. 다만 모든 지역이 퇴화해 풀 행이 없으면 판정 불가 풀 행(verdict_note '적층 미작동', 풀 대체 비율, flagged_regions)을 더하고, 퇴화 지역을 빼서 풀 지역이 2 미만이 된 판정 불가 풀 행에는 verdict_note '적층 미작동(…)'을 적는다(검토 결함 5).
- 판정 불가 문장: '적층이 작동하지 않아(대체 비율 r) 판정할 수 없었다' 는 verdict_note 가 '적층 미작동'으로 시작할 때만 쓴다. 그 밖의 판정 불가(풀 지역 2 미만, CI 비유한, 점 추정이 CI 밖)는 '… 차이를 판정할 수 없었다(라벨 k개, 사유 …)' 로 쓴다(1절 다섯 갈래의 '판정할 수 없었다(사유)').
- 동등 재확인: 판정이 동등인 지역·풀 행은 대체가 있었던 (분할, n, 추출)의 키를 모두 뺀 저장소로 다시 판정하고, 동등이 아니면 판정을 '미결정'으로 두고 사유를 적는다('동등 판정은 … 같을 때만 쓴다'를 동등 문장을 쓰지 않는 것으로 읽었다).
- Holm: XB-1–XB-3 의 n 별 층화 평균 양측 p(두 가중 가운데 큰 값), m 12, 행 없음·판정 불가 p 1. XB-4 는 비열등 단측 p × 2, m 4. 둘 다 보조 열이다(1절 다중성: XB 는 4분 판정이 판정).
- XB-5(보조): Stack0 − P0, Stack0 − B:ens(n 0), Stack − Pe, StackR − Re(전이 0.25, 지역 내 교차검증 λ)를 격자 전부에서, 주 4지역·PE1·PE2·3지역 보조 열 풀로 낸다. '대상별 행' 은 저장소 전부의 지역 행으로 두고 하위 지역에는 subregion 표지를 붙인다(본문은 대상 수로 쓴다). 'PE1·PE2·3지역 보조 열' 을 주 대비(Stack − P1, StackR − R1)에도 적용했고 P* − P1 을 서술로 더했다(확인적으로 세지 않는다).
- XB-6(서술): 추출별 w*·ŵ·γ̂·대체 사유, (대상, n) 요약, 뺀 후보, 대상별 K, 전량 n 의 CCI 가중(Cr + Ca)과 P1 RMSE 의 순위상관(서술, 판정어 없음). 풀 행에 K_by_target 을 적는다.
- 해석 문장: 2.2 표의 문장을 갈래마다 두고 xbatch_core.compose_sentence 로 보정 전 유의·효과 크기·지역 열세 덧붙임을 붙인다. XB-1·2 우세 문장 뒤에는 지역 행을 적고(WF7 처럼 캐나다가 풀을 정할 수 있다), XB-1 n 10 문장에는 'P1 대 P1 비교일 수 있다'를 붙인다.

## 7. W+(XC 의 A1+ 팔, 계획 2.3)

- `select_w_wplus(unit, sel, n, d)`: h54.TUnit.select_w 와 같은 묶음(cv_folds_of)·seed 0 적합(R1cv, R2cv, D1cv)·SSE 누적 순서로 W 를 고르고, 같은 묶음 고리에서 Stack(묶음 j 의 학습 쪽 라벨로 다시 맞춘 적층)과 StackR@0.25(원천 행 + 학습 쪽 라벨의 잔차, 적층 = P1 이면 R1cv 성분 재사용)를 더해 W+ 를 고른다. 후보 순서(동률 순서) = W_CANDS 다음 Stack, StackR@0.25. 라벨 < 10 이면 둘 다 P1, 묶음 < 2 이면 P1(h54 와 같다). W 의 선택과 묶음별 RMSE 는 h54 select_w 와 같다(시험 (j)).
- `wplus_predictions(unit, sel, n, d, seed, g_r1)`: 채점 셀의 Stack, StackR@0.25 예측과 적층 기록.
- XC 공급자 계약(`x_workflow_end_to_end.WPlusProvider`, XC 작성자가 정한 이름): `xc_wplus_cv(unit, tr, te, n, d, fold, seed)` → {'Stack', 'StackR@0.25'}(te 예측), `xc_wplus_final(unit, sel, n, d, seed)` → 같은 키의 채점 셀 예측. 묶음 안 적층은 학습 쪽 라벨 tr 로 다시 맞추고(안쪽 묶음 추출 표지 '추출.묶음'), StackR 은 unit.fit 으로 적합해 세기와 추적을 받는다. select_w_wplus 의 묶음 안 값과 같다(시험 (j) 둘째). 공급자 경로에서는 R1cv 성분을 넘겨받지 않으므로 적층 = P1 이어도 StackR 을 다시 적합한다(같은 행렬·seed 라 값은 같고 적합 수만 늘어난다).
- XC 단위의 준비: 문맥에 후보 표가 없으면 공급자 함수가 `attach_cands_t(unit.a, unit.c, unit.alias, mode, split)` 로 붙인다(새 seed 분할 201–220 도 같은 색인 규칙과 일치 단언). 인자에 TDDM·TDDV4·VARIANT·PRODUCTS 가 없으면 기본 tdd 표와 핵심 판을 쓴다.
- 계획 0.3: 이 파일(W+ 코드)과 해시를 R1a·R1b 제출 전에 커밋한다. 커밋은 하지 않았다(작업 지시).

## 8. 시험 파일 이름

- 계획 2.2 는 `tests/test_x_stacking.py` 를 등록했고 작업 지시는 `tests/test_x_xb.py` 를 지정했다. 등록 항목 (a)–(i)와 추가 항목은 `tests/test_x_xb.py` 에 있다.
- 등록 이름의 얇은 파일 `tests/test_x_stacking.py` 를 더했다(검토 결함 11). 그 파일 이름만 줄 때(`python3 -m pytest tests/test_x_stacking.py`) test_x_xb 의 시험을 모두 다시 내보내 돌리고, 같은 실행에 test_x_xb.py 도 모이면(디렉터리, `tests/test_x_*.py` 글롭, 경로 인자 없음) 이름 대응 확인 시험 하나만 돌려 같은 시험이 두 번 돌지 않게 한다. 묶음 글롭 `tests/test_x_*.py` 는 두 파일을 모두 덮는다.
- 계획 개정 이력에 넣을 문장(다음 허용 개정에서 사용자가 정한다. 이 작업은 계획을 고치지 않는다): 'XB 시험의 본문은 작업 지시에 따라 `tests/test_x_xb.py` 에 두었고 등록 이름 `tests/test_x_stacking.py` 는 그 시험을 다시 내보낸다. 항목 (a)–(i)는 같다.'

## 9. 로컬 확인 기록(2026-10-04, 라벨 값과 결과 수치는 보지 않았다)

- 단위 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c 108-111 python3 -m pytest -q -p no:cacheprovider tests/test_x_xb.py`(출력은 0.3 의 grep 필터를 거쳤다). 18개 모두 통과, 59 s, 최대 RSS 342 MB.
- 주의(가상 메모리 상한): `prlimit --as=10737418240`(1절 로컬 자원의 대체 강제 방법, xbatch_core.limit_memory 와 같은 RLIMIT_AS)을 걸면 CatBoost 적합이 수백 건 쌓인 뒤 `get_feature_importance` 안에서 멈춘다. 멈춘 프로세스의 VmPeak 10.48 GB(상한)·RSS 234 MB 였다. CatBoost 의 가상 주소 사용이 적합마다 늘어 mmap 이 실패하는 것으로 보인다. 시험은 AS 상한 없이 돌리고 최대 RSS(342 MB)를 적었다. systemd-run --user --scope 는 이 서버에서 실패한다. 이 모듈의 main 은 허용 표지 없는 실행에서 xbatch_core.limit_memory(10)을 부르므로, 적합이 많은 로컬 실행(스모크 확대 등)은 같은 위험이 있다(스모크 55건은 문제없었다). 공용 골격 작성자에게 넘길 사항이다. 이 문제는 10절에서 이 모듈 안에서 처리했다(main 의 `apply_local_resources`: 적합이 있는 모드는 glibc 할당 영역을 2개로 묶은 뒤에만 주소 공간 상한을 건다).
- 1c 입력(--write-tdd-v4): v4 새 셀 239개(Tibet_LGD 132, Russia_C 51, Canada 37, NAtlantic 19), 그 가운데 pre2010 27개(모두 Russia_C, 계획 2.2 의 27/51 과 같다). 재현 점검: v3 셀 400개의 최대 |차| 4.5e-13 °C·day, 표지 400개 일치. 표 sha256 앞 16자 4b22ca458685950c, 최대 RSS 3.5 GB.
- --count-only(라벨 값 미사용, 스레드 1, 23 s): 작업 단위 211(xb_r 74 = 알래스카 25, 레나 24, 캐나다 25; xb_t 137 = 31 (대상, 모드) × 유효 분할), 건너뛴 분할 19(채점 블록 < 2 11, 채점 셀 없음 1, 중복 7). 적합 xb_r 25,140건(R1·Re·StackR 각 2,506, 교차검증 적합 R1cv·Recv·StackRcv 각 5,874), xb_t 16,406건(R1·Re 각 6,318, StackR 3,770 = 라벨 없이 정한 대체(n < 10, 블록 묶음 < 2)를 R1 재사용으로 뺀 상한). 추정 누적(h40 BASE_FIT, 4스레드 프로세스): xb_r 0.60 CPU-h, xb_t 1.85 CPU-h, 1차 Rescale 실측 비 0.62 적용 합계 약 1.5 워커·시간(계획 2.2 의 약 1.4 워커·시간과 같은 규모). 22 워커면 가장 긴 단위가 벽시계를 정한다. v4 tdd 표가 있을 때 211 단위 모두 K = 7(표가 없으면 LGD 4대상 20 단위에서 P* 가 빠져 K = 6).
- (아래 두 항목은 검토 전 규칙, 곧 셀 묶음을 대체로 센 판의 기록이다. 문구 그대로의 규칙으로 다시 센 값은 10절에 있다.)
- 라벨 없이 센 적층 대체(추출 수, xb_t): n 0 은 137/137, n 3 은 675/675(n < 10). 블록 묶음 < 2 는 n 10 에서 640 추출 가운데 166, n 40 555 가운데 121, n 160 490 가운데 85, n 320 345 가운데 54, n 1000 180 가운데 19(모두 하위 지역 모드 i·x 와 소수 블록 대상), 전량 137 가운데 n < 10 9·블록 묶음 < 2 8. 주 4지역·Russia_C~lgd·Tibet_LGD·알래스카 x 의 n 10 블록 묶음 대체는 레나 2/25 뿐이다. 계획 2.2 가 n 10 에서 '묶음 부족 대체가 많을 것'으로 본 예상과 달리 주 4지역에서는 적다(γ̂ = 1 대체는 라벨이 있어야 셀 수 있어 실행 뒤에 센다). xb_r 은 대체 0. 표: `data/processed/xbatch/XB_multisource_stacking/xb_count_fallback.csv`. 계획의 '개정 이력에 적는다' 는 사용자 결정 사항으로 남긴다(이 기록은 계획을 고치지 않는다).
- 제한 스모크(코어 4개, 스레드 2, 출력 필터): xb_r 캐나다 분할 1(n 20·전량), xb_t 러시아 W·Tibet_LGD 분할 1(n 0·3·10·전량), 추출 1, seed 1. 3 단위 모두 상태 ok, 적합 55건, 적합 실패·비유한 키 0, 16 s, 최대 RSS 315 MB. 조각은 `sealed/smoke_shards`, 표는 `sealed/smoke_*` 에만 썼고 열지 않았다. 같은 스모크를 두 번 돌려 봉인 표의 sha256 이 같았다(결정성).
- 재현 관문(--smoke --gate --gate-level local_rescale): 스모크 조각과 WF6·WF7 Rescale 조각의 공통 키 16개(xb_r 캐나다 3, xb_t 러시아 W 13; P0·P1·R1·Re) 모두 통과, 최대 |ΔSSE| 2.3e-10 cm²(상대 7e-16), 셀 수 일치. 표 `xb_gate_smoke.csv`(|ΔSSE| 만 있다).

## 10. 검토 반영(2026-10-04, 결함 11건)

계획은 고치지 않았다. 동결 모듈 4개는 고치지 않았다. 공용 골격 `xbatch_core.py` 는 PAYLOAD_INPUTS 에 두 줄만 더했다.

| 결함 | 처리 | 시험 |
|---|---|---|
| 1 (주요) v4 tdd 표가 묶음에 없고 없을 때 조용히 진행 | PAYLOAD_INPUTS 와 PAYLOAD_EXTRA 에 표·메타 추가. 없거나 sha256 앞 16자가 4b22ca458685950c 가 아니면 중단, `--allow-missing-tdd-v4` 일 때만 진행하고 unit.json·설정에 적는다(5절) | (o) |
| 2 (중간) 묶음 부족 대체가 문구 그대로가 아님 | cv_folds_of 의 K < 2 일 때만 대체. 셀 묶음은 적층하고 fold_flag·n_cell_folds 에 남긴다(2절). 설정 해시가 바뀌었다(시험 (h)의 고정값 갱신) | (d), (g), (h) |
| 3 (중간) Wei·YK 누설 통제가 선택 열에 기댐 | Wei 표의 mask_ok·train_dist_km 필수, 모든 원천 행 거리 유한 필수(아니면 중단), 5 km 제외는 확장 판 전체에 적용하고 뺀 행 수 기록. YK 는 L0 라 마스크 열이 없으면 통과(4절) | (i), (i2) |
| 4 (중간) XB-4 판정 불가가 기각으로 셈 | 비열등이 없고 판정할 수 없는 n 이 있으면 '판정 불가(사유)'. 판정 불가 행 문장은 대체 비율 또는 사유 문장(6절) | (q) |
| 5 (낮음) 모든 지역 퇴화 시 풀 행 없음, 판정 불가 문장 오용 | 판정 불가(적층 미작동) 풀 행 추가, 퇴화로 풀 지역 2 미만이 된 행 표지, 적층 미작동 문장은 verdict_note 가 그럴 때만(6절) | (q) |
| 6 (낮음) 변형 조각이 섞임 | 집계·관문이 이 변형의 조각만 읽는다(6절) | (r) |
| 7 (낮음) γ 교차검증 비중첩 | 구현 보고서 3.3 그대로 두고 SI 문장을 GAMMA_CV_NOTE 로 집계 메타에 넣었다. 중첩판은 사용자 승인과 등록 이탈 기록이 있을 때만(2절) | 없음(서술) |
| 8 (낮음) anchor_fill 범위, P* 결정 단위 | anchor_fill 은 그대로 두고 SI 문장(ANCHOR_NOTE). P* 는 대상 수준에서 한 번 정한다(1절) | (p) |
| 9 (낮음) 로컬 자원 규약 누락 | Rescale 밖에서는 모든 모드에서 가용 메모리 30 GB 대기(`--mem-wait`, 기본 1,800 s). 세기·집계는 주소 공간 10 GB, 적합 모드(스모크, --allow-local 실행)는 할당 영역 2개를 묶은 뒤에만 10 GB 상한(묶지 못하면 상한 없이 경고). smoke_env_report 를 스모크 집계 메타와 조각 unit.json 에 적고 경고를 화면에 쓴다 | (s) |
| 10 (낮음) 봉인 밖 runs.csv 의 RMSE 열 | runs.csv 에서 rmse_cm·rmse_beq_cm·bias_cm 를 뺀다(3절) | (s) |
| 11 (낮음) 등록 시험 파일 이름 | 얇은 `tests/test_x_stacking.py` 추가, 개정 이력 문장안을 8절에 적었다 | 이름 대응 시험 |

**다시 한 확인(2026-10-04, 라벨 값과 결과 수치는 보지 않았다)**
- 단위 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c 108-111 python3 -m pytest -q -p no:cacheprovider tests/test_x_xb.py tests/test_x_stacking.py`(0.3 의 grep 필터). 25개 통과(test_x_xb 24, 이름 대응 1), 62 s, 최대 RSS 317 MB. `tests/test_x_stacking.py` 만 주면 24개가 모인다. XC 의 W+ 공급자 시험(`tests/test_x_xc.py -k wplus`) 2개 통과.
- --count-only(라벨 값 미사용, 스레드 1, 26 s, 최대 RSS 272 MB): 작업 단위 211(xb_r 74, xb_t 137), 건너뜀 19, 211 단위 모두 K = 7, 후보를 뺀 단위 없음(P* 대상 수준 결정 포함). 적합 xb_r 25,140건(바뀌지 않음), xb_t 17,312건(R1·Re 각 6,318, StackR 4,676. 셀 묶음 추출이 대체에서 빠져 R1 재사용이 줄었다. 이전 16,406건). 추정 누적 xb_r 0.60, xb_t 1.94 CPU-h, 실측 비 0.62 적용 합계 약 1.57 워커·시간(계획 2.2 의 약 1.4 와 같은 규모).
- 라벨 없이 센 대체(추출 수, xb_t): n 0 은 137/137, n 3 은 675/675(n < 10), 전량은 137 가운데 n < 10 9. 묶음 수 K < 2 대체는 모든 n 에서 0 이다. 셀 묶음 적층(대체 아님, 서술)은 n 10 640 가운데 166, n 40 555 가운데 121, n 160 490 가운데 85, n 320 345 가운데 54, n 1000 180 가운데 19, 전량 137 가운데 8. n 10 의 셀 묶음은 하위 지역(AL-1·AL-2·AL-4·AL-6 모드 i·x 에서 14–20/15–25, AL-5 1·4/25, LE-1 5·6/20, LE-2 5·7/15)과 레나 2/25, NAtlantic~lic 3/10 이다. 주 4지역의 n 10 대체는 라벨 없이 정해지는 사유로는 없고, γ̂ = 1 대체는 실행 뒤에 센다. 계획의 예상('n 10 에서 묶음 부족 대체가 많을 것')과 다르다는 점과 이 수는 개정 이력에 적을 사항이다(사용자 결정). 표: `data/processed/xbatch/XB_multisource_stacking/xb_count_fallback.csv`(n_cell_folds 열 추가).
- 제한 스모크(main 경로, 코어 4개, 스레드 2, MALLOC_ARENA_MAX 를 주지 않아 mallopt 경로를 지났다, 출력 필터): v4 표 sha256 확인, 3 단위 상태 ok, 적합 55건, 적합 실패·비유한 키 0, 17 s, 최대 RSS 301 MB. 조각과 표는 봉인 폴더에만 썼고 열지 않았다. 화면에는 봉인 표의 행 수와 해시만 남았다.
- 재현 관문(--smoke --gate --gate-level local_rescale, 변형 필터 경로): xb_r ↔ WF6 공통 키 3, xb_t ↔ WF7 공통 키 13, 실패 0.
- 남은 사용자 결정: (a) 개정 이력에 시험 파일 이름 대응(8절 문장안)과 셀 묶음 세기 결과를 적는 일, (b) γ 의 중첩 교차검증 판을 등록 이탈로 넣을지, (c) 셀 묶음을 대체로 세는 해석을 등록 이탈로 택할지(택하지 않으면 현재 구현이 문구 그대로다).
