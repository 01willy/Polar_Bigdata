# 신규 실험 묶음(XA–XJ) 하네스 구현 계획 (2026-10-04)

**성격**: 구현 계획서다. 코드, 자료, 기존 문서를 바꾸지 않았고 모형 학습을 하지 않았다. 이번 작성에서 한 계산은 세 종류뿐이다. (1) XI 의 설계 점검(공변량 √TDD 만 쓰는 블록 정렬, 분할 구조, 학습 없는 적합 수 세기), (2) XE 의 근거가 되는 라벨 분산 서술(모형 결과 아님), (3) XA 입력 표의 재계산. 모두 스레드 1개, 각 1분 안팎이고 파일을 쓰지 않았다.

**열람 상태(등록 문서에 옮겨 적을 내용)**
- XA: QA 문서의 C2 점검 값(ρ −0.08)을 `data/processed/wf/wf0_misspec.csv` 에서 다시 계산해 확인했다(ρ −0.082, p 0.666). 같은 표에서 ρ(재보정 이득, P1 − R1(λ 0.25)) = 0.07(p 0.715)을 새로 보았다. 이 값은 사후 열람이다. WF4 조각의 unit.json 에서 계수 오차 열(E0, E_own, logE_ratio_own)의 존재를 확인했고 상관은 계산하지 않았다.
- XI: 캐나다 W 블록 수와 W 셀의 외삽 비율(공변량만)을 계산했다. 모형 예측은 보지 않았다.
- XE: 알래스카·레나·캐나다의 격자 안·1 km 칸 안 ALT 분산 비율을 계산했다(자료 서술).

---

## 1. 요약

1. 열 개 실험 가운데 XA(C2 분해)와 XG(제품 비교의 전이 부분)는 기존 조각만으로 계산할 수 있다. 새 적합이 필요 없다.
2. XB, XC, XD, XE, XH, XI, XJ 는 새 적합이 필요하다. 모두 CatBoost CPU 와 물리식이라 Rescale CPU 노드에서 돌린다. 기존 측정 시간으로 환산한 합계는 4스레드 워커 기준 약 8 워커·시간이고, elm 노드(워커 22개) 두 작업으로 각 1시간 안, 합계 10 달러 안으로 본다.
3. GPU 는 XD 의 학습형 샘플링 정책(DeepSets)에만 쓴다. 로컬 GPU 0–4 는 13:04 기준 비어 있다.
4. 기존 하네스(h40, h41, h42, h54)는 고치지 않고 파일 경로로 불러 쓴다(h54 가 h40·h42 를 부르는 방식과 같다). WF 결과를 낸 h54 의 해시가 그대로 남으므로 WF 조각과의 비트 단위 재현 점검이 가능하다.
5. 지정 점검 결과
   - (i) XI: v4 약관 확인분 캐나다 확충판(787셀, 58블록)에서 warm W 는 11블록(채점 블록 11개)이다. v3 는 3블록이었다. CI 조건(블록 합집합 8 이상)을 만족하므로 WF10 재시험이 가능하다. 다만 캐나다의 기후 외삽 폭이 작다. W 셀의 중앙 √TDD 는 45.54 이고 A 의 최대값은 분할에 따라 45.19–45.93 이며, 10개 분할 가운데 4개에서 W 셀의 2.4 % 만 A 범위 밖이다. 외삽을 보장하는 변형(warm_trim)을 함께 등록할 것을 권한다.
   - (ii) XA: `wf0_misspec.csv` 는 LG·LGD 곡선 표의 점 추정만 담고 있어 CI 를 줄 수 없다. 대상별 재표집 분포는 WF4 조각(30 (대상, 모드), 132 단위)의 BlockStore 와 `h40.contrast(return_dist=True)` 로 다시 만들 수 있다. 같은 대상 안에서 재표집 번호가 공통이라 P0 − W = (P0 − P1) + (P1 − W) 가 재표집마다 성립한다.
   - (iii) XB: 앵커 P1, Kudryavtsev 보정, 토양형 Stefan, 토양 도일 Stefan, CCI 원값·선형 보정, Stefan·CCI 평균은 지역 내·전이 모두 계산할 수 있다. P*(연도 정합 도일)는 v3 직접 라벨 셀에는 모두 있으나 v4 새 셀 516개에는 없다. 가중은 선택 라벨의 교차 적합 기저 예측에 단체(simplex) 제약 최소제곱을 맞추고, P1 쪽 수축 강도 γ 를 2단 블록 교차검증으로 고른다.
   - (iv) XH: h41 은 통합 모형(V-R, V-P, V-S, V-B, V-C, V-G)과 지역 내 블록 5겹(W1, W2)을 이미 냈다. kNNDM 은 알래스카(m1, 6겹)에만 있다. 최소 추가 실행은 지역 내 학습(W1 정의)에 셀 무작위, 0.05° 지점, kNNDM 세 단을 더하는 것이고 지역 홀드아웃은 h41 V-G 를 쓴다.

---

## 2. 공통 설계

### 2.1 코드 구조

**동결 모듈(고치지 않는다. 묶음 스크립트가 sha256 앞 16자로 대조한다)**

| 파일 | sha256 앞 16자 | 쓰는 것 |
|---|---|---|
| `scripts/3_deep_learning/h40_label_grid.py` | 7098f59dabe2b73f | Data, build_ctx, draw_cells, draw_blocks, Fitter·cb_fit, TMx, contrast, build_curve, ls_E, _prep_stats |
| `scripts/3_deep_learning/h42_label_grid_ext.py` | 22e215e435e157be | shrink, cv_folds_of, stack_rows, ps_index, pseudo_values, anchor_fill, affine_ls, build_unit·ext_tables(앵커 열), TestBook, region_stats, pool_rows, verdict4, FitterX |
| `scripts/3_deep_learning/h54_workflow.py` | cf9a1f6d0929c892(마지막 변경 7da0064, 2026-10-02 10:51) | RCtx, build_rctx, build_tctx, split_structure_ext, RUnit(draw, coefs, folds, lam_cv, fit_resid, fit_direct, physics, strategy_sets, s4_order), TUnit(rows_R, rows_D, select_w), T7Unit, R9Unit·GridDecompMixin, R10Unit, wf10_split·wf10_plan, kcenter_order, standardize, re_anchor, make_tm, write_shard 의 형식 |
| `scripts/2_evaluation/h41_validation_ladder.py` | 492373e4d37b5ea4 | folds_random, folds_grouped, groups_s, VData·unit_cells, models_pooled, models_w2 |

`scripts/3_deep_learning/m1_cv_scheme_comparison.py` 는 모듈 최상위에서 `ap.parse_args()`(86행)를 부르므로 불러 쓸 수 없다. kNNDM 함수(`knndm`, `merge_clusters`, `prediction_domain`, `nnd_km`)는 새 모듈로 옮기고 같은 입력에서 같은 결과를 내는지 시험한다.

**새 파일(제안)**

| 파일 | 역할 |
|---|---|
| `scripts/3_deep_learning/xbatch_core.py` | 공용 골격. h40·h42·h54 를 importlib 로 등록하고, h54 인자 계약(a.SEEDS, a.threads, a.cb_iters, a._ha, a.PROC, a.LGD, a.subregion_map, a.SPLITS, a.G)을 만족하는 인자 객체, 실험별 unit_cfg·cfg_hash, 조각 쓰기(h54.write_shard 와 같은 형식, 실험 고유 cfg), 작업 단위 열거, 프로세스 풀(h54.execute 와 같은 코어 고정·재시도), 집계(h40.build_curve, h42.TestBook, h54.make_tm)를 둔다. 자료 디렉터리를 대상별로 받는 resolve(`--region-tables 이름:디렉터리`)를 둔다(h54 의 LGD_SPECS 는 상수라 확장할 수 없다) |
| `scripts/3_deep_learning/x_multisource_stacking.py` | XB 단위(XBRUnit ⊂ h54.RUnit, XBTUnit ⊂ h54.TUnit)와 가중 함수 |
| `scripts/3_deep_learning/x_workflow_end_to_end.py` | XC 단위(XCUnit ⊂ h54.TUnit)와 지역 간 배분 분석 |
| `scripts/3_deep_learning/x_placement_policy.py` | XD 배치 알고리즘(H 계열) 평가 단위와 후보 집합 생성 단위 |
| `scripts/3_deep_learning/x_placement_policy_learn.py` | XD 학습형 정책(GBM CPU, DeepSets GPU)과 지역 하나 제외 검증 |
| `scripts/3_deep_learning/x_hires_covariates.py` | XE 단위(R9Unit 의 입력 집합 교체판) |
| `scripts/1_data_prep/xe_point_covariates.py` | XE 점 규모 공변량 추출(로컬 원자료) |
| `scripts/2_evaluation/xa_c2_gain_decomposition.py` | XA 분석(학습 없음) |
| `scripts/2_evaluation/x_validation_ladder_regional.py` | XH 지역 내 무작위·지점·kNNDM 단(h41 단위 형식 재사용) |
| `src/polar/cv_schemes.py` | m1 의 kNNDM·예측 영역 함수 사본(XH 전용) |
| `scripts/1_data_prep/xg_sample_alt_products.py`, `scripts/2_evaluation/x_product_comparison.py` | XG 제품 표집, 누설 표, 비교 분석 |
| `scripts/3_deep_learning/x_new_regions.py`, `scripts/2_evaluation/x_new_regions_pool.py`, `scripts/1_data_prep/build_fidelity_base_v5.py`, `scripts/1_data_prep/lgd_eligibility_v2.py` | XF 새 지역 자료 조립, 실행 표, 풀 집계 |
| `scripts/3_deep_learning/x_tempderived_aux_labels.py` | XJ 단위 |
| `scripts/rescale/make_payload_xbatch.sh`, `scripts/rescale/run_xbatch.sh`, `configs/rescale/xbatch_smoke.yaml`, `xbatch_a_elm.yaml`, `xbatch_c_hematite.yaml` | Rescale 묶음·실행(run_wf.sh 의 단계 구조 그대로, 실험 목록 XB_EXPS) |
| `tests/test_x_*.py` | 실험별 단위 시험(4절) |

XI 는 새 코드가 필요 없다. h54 를 그대로 `--exp wf10 --data-dir <실행 표>` 로 부른다(4.9절). warm_trim 변형만 xbatch_core 에 R10Unit 의 하위 클래스로 둔다.

### 2.2 조각·키·집계 규약(h54 와 같다)

- 저장 키: (method, learner, alpha, placement, n, draw, seed, lam). 교차검증 λ 는 lam = −1.0, 고른 값은 alpha_sel 열.
- 조각: `<tag><실험>__cpu__<대상>__<모드>__s<분할>[__<변형>]_{runs.csv, blocksse.npz, unit.json}`. unit.json 은 마지막에 원자적으로 쓴다. `--resume` 은 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다. 공통 해시에서 variant, data_sha 를 뺀다(h54 CFG_UNIT_KEYS 와 같다). 산출 디렉터리는 `data/processed/xbatch/<실험>/`.
- 통계: 분할 안 채점 블록 재표집(공통 인덱스, 셀 가중·블록 등가중), 재표집 10,000회, 4분 판정(동등 ±0.5 cm), 층화 평균 h42.pool_rows(CI 풀 = 채점 블록 합집합 8 이상), Holm 은 보조 열.
- 실행 보호: `WF_RESCALE=1` 또는 `--allow-local` 이 없으면 스모크·세기·집계 외 실행을 거부하고 스레드 상한 4, 집계 재표집 상한 1,000(h54 와 같다).
- 누설 시험 규약: 선택 라벨 밖의 A 라벨과 B 라벨을 바꿔도 모든 저장 예측과 선택이 같아야 한다(h54 시험 b, d, m 과 같은 형식).

### 2.3 플랫폼과 자원

| 실험 | 계산 | 플랫폼 |
|---|---|---|
| XA, XG(전이) | 분석(재표집) | 로컬 CPU, 4스레드 이하, nice 10 |
| XB, XC, XD(알고리즘·후보 집합), XE(적합), XH, XI, XJ | CatBoost CPU·물리식·ridge·RF | Rescale CPU(elm 96코어 3.24 달러/시간, 대안 hematite 64코어 4.22 달러/시간; `configs/rescale/wf2_full.yaml`, `wf3_full_hematite.yaml`) |
| XD(학습형 정책) | DeepSets(PyTorch), GBM | 로컬 GPU 0–4(DeepSets), GBM 은 Rescale 또는 로컬 CPU 8스레드 이하 |
| XE(공변량 추출), XG(제품 표집) | 래스터 읽기 | 로컬 CPU 4스레드 이하, 메모리 10 GB 이하(원자료 DEM 8.6 GB, Hansen 11 GB, GSW 3.1 GB 가 로컬에만 있다) |
| XF | 자료 조립 로컬, 적합 Rescale | 혼합(대비는 지역 안에서 닫는다) |

- 플랫폼 규칙: 한 대비(같은 표의 두 방법)는 한 작업·한 노드 종류의 조각 안에서 닫는다. CPU CatBoost·물리식은 elm·hematite 사이에서 비트 단위로 같았고(WF9 재현 점검 공통 키 11,814개, 최대 |ΔSSE| 0, `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5.3) 로컬과 Rescale 사이 최대 |ΔSSE| 6.8e-6 cm² 였다(같은 문서 개정 이력 2026-10-02 11:32). 그래도 각 실험은 비교 기준(P1, R1, S1 등)을 자기 조각에서 다시 적합한다.
- GPU 상태(`nvidia-smi`, 2026-10-04 13:04): 0–4 사용 0 MiB, 5–9 는 각 14,532 MiB·사용률 100 %(다른 사용자). 메모리 `free -g` 기준 가용 113 GB, 작업 지시의 상한 55 GB 를 계획 기준으로 둔다.

### 2.4 사전 등록과 표지

- 실행 전에 `docs/EXPERIMENT_PLAN_X_2026-10-04.md` 를 커밋한다. 가설, 판정 규칙, 해석 규칙, 대상·격자·분할·seed, 열람 상태를 적는다.
- 표지: XA, XB, XD, XE, XH, XI, XJ 는 '결과 열람 뒤 설계'다. XC 는 구성 요소의 결과를 본 뒤 설계했으나 결합 워크플로의 성능은 계산한 적이 없으므로 '실행 전 등록 확인 시험'으로 둔다. XF 의 새 지역 결과는 맹검이다(라벨 통계를 실행 전에 계산하지 않는다. LGD 의 WRAPUP 7.2 (a)7 규칙과 같다).

---

## 3. 지정 점검 결과

### 3.1 (i) XI: 캐나다 warm 블록 수

h54 의 `wf10_split`·`wf10_plan` 을 그대로 불러 계산했다(`--data-dir` 만 바꿈, 학습 없음). 블록 정렬 열은 h54 가 실제로 쓰는 기온 √TDD(`e5_sqrt_tdd`, h40 의 s)다. 토양 √TDD 로 바꿔도 warm W 블록 수는 같았다.

| 자료 | 캐나다 셀 | 블록 | warm W 블록 | W 채점 셀(채점 블록) | 유효 분할 / 10 | I 채점 블록 범위 | W 셀 가운데 A 의 √TDD 범위 밖 비율(분할 평균, 최소–최대) |
|---|---|---|---|---|---|---|---|
| v3(`data/processed`) | 750 | 39 | 3 | 192(3) | 10 | 10–19 | 0.953(0.953–0.953) |
| v4 전체판(`lgd/run_tables/Canada_expanded`) | 825 | 67 | 11 | 209(11) | 10 | 8–35 | 0.304(0.024–0.957) |
| v4 약관 확인분(`lgd/run_tables/Canada_expanded_lic`) | 787 | 58 | 11 | 207(11) | 10 | 7–30 | 0.584(0.024–0.957) |

- 판정: WF10 재시험은 가능하다. W 채점 블록이 11개라 CI 조건(블록 합집합 8 이상, `h42.pool_rows`)을 만족하고, 알래스카(warm W 17블록, 기온 √TDD)와 함께 두 지역 층화 평균을 낼 수 있다. 주 판정 판은 약관 확인분 판이다(LG 개정 15 (m)2, `docs/LGD_LICENSE_CONTACTS_2026-09-30.md`).
- 위험: 캐나다의 기후 외삽 폭이 작다. W 블록 평균 √TDD 는 45.54–46.55, 나머지 블록 평균의 최대는 45.52 다. 셀 단위로는 W 셀의 최소 45.19, 중앙 45.54 이고 A 의 최대는 분할 1, 2, 4, 5 에서 45.93 이다. 이 네 분할에서는 W 셀의 2.4 % 만 A 범위 밖이다. v3 는 모든 분할에서 95.3 % 였다. 새 확충 셀이 따뜻한 칸 근처에 있어 A 에 섞이기 때문이다.
- 권고: (1) 주 시험은 등록 규칙 그대로(WF10-a·b·c 와 같은 판정). (2) 등록 변형 warm_trim: A 에서 s ≥ min(s_W) 인 셀을 뺀다(분할마다 7–24셀). W 셀이 모두 A 범위의 경계 밖이 된다. (3) 분할별 frac_W_outside_A 를 표에 적는다(h54 R10Unit meta 에 이미 있다).
- 적합 수(학습 없는 세기): 캐나다 v4 약관 확인분 warm·cold 20 단위, 1,600건(v3 1,598건). n 격자 {100, 500, 전량} 가운데 |A| 237–383 이라 500 은 없다.
- cold 변형: v4 약관 확인분 W 17블록(채점 블록 14), 전체판 21블록(18), v3 15블록(12). 알래스카 cold 는 W 1블록(4,716셀, 대상 셀의 34.7 %)이라 무효로 남는다.

### 3.2 (ii) XA: C2 분해의 입력

**`wf0_misspec.csv` 의 입력과 열**(`scripts/2_evaluation/wf0_reanalysis.py`)
- 입력: `results/rescale_lg/data/processed/lg/lg_curve.csv`(LG, Rescale)와 `data/processed/lgd/lgd_curve_lic.csv`(LGD 약관 확인분). 셀 무작위 추출, α 1, catboost_lo(물리식은 none), 점 추정만 대상과 그린란드 제외, 30대상.
- 열: P0(RMSE), P1·R1_0.25·R1_1.0·R2_1.0·D0_1.0·D1_1.0(각각 RMSE − RMSE(P0)), sig_*(곡선 표의 유의 표지), recal_gain = −(P1 − P0), ml_best_delta = 다섯 ML 레시피의 최소 Δ(사후 선택), ml_vs_p1 = ml_best_delta − P1.
- 라벨 전량(n = −1)의 점 추정만 있다. 재표집 분포가 없어 CI 를 줄 수 없다.
- 재계산: Spearman(recal_gain, −ml_vs_p1) = −0.082(p 0.666), QA 문서의 −0.08 과 같다. Spearman(recal_gain, −(R1_0.25 − P1)) = 0.07(p 0.715)이다(사후 열람).
- 주의: recal_gain 은 계수 오차의 대리값으로 부적절하다. 캐나다 x 는 라벨 전량에서 P1 이 P0 보다 4.64 cm 나쁘다(라벨 위치 이질성). 계수 오차는 별도 열로 정의해야 한다(아래).

**재표집 분포를 만들 수 있는 원천**
- WF4 조각: `results/rescale_wf/data/processed/wf/shards/wf4__cpu__*`, 30 (대상, 모드), 132 단위(elm 한 플랫폼). 키 P0, P1, P2, R1(λ 0.25·0.5·1.0), R2, D1, W 가 n ∈ {10, 40, 160, 전량}에 있다. unit.json 에 diag(n = 10 추출별 bias, abs_bias), E0, E_own, E_A, E_B, logE_ratio_own, logE_ratio_AB 가 있다(예: `wf4__cpu__Lena__x__s1_unit.json`).
- `h40.contrast(tm, gA, gB, return_dist=True)` 는 대상마다 seed_of('lgboot', 이름)의 재표집 행렬을 분할별로 공유한다(`h4_common.boot_delta_blocks`). 같은 대상의 모든 방법 쌍이 같은 재표집 번호를 쓰므로 Δ(P0 − W)^b = Δ(P0 − P1)^b + Δ(P1 − W)^b 가 재표집 b 마다 정확히 성립한다(셀 가중·블록 등가중 모두).
- LG 조각(`results/rescale_lg/data/processed/lg/shards`, 1,044 파일)은 WF0 의 사후 최선 ML 재현용이다.

**분석 명세(등록안)**
- 계수 오차 x: CE1 = |log(E_own/E0)|(대상 전체 라벨, 오라클), CE2 = |편향10|(WF4-c 진단값, 라벨 10개, 실용), CE3 = RMSE(P0) − RMSE(P1@전량)(QA 와의 연속성, 보조).
- 이득 y(n ∈ {10, 40, 160, 전량}, 주 = 전량): G_recal = P0 − P1, G_ML = P1 − R1(λ 0.25), G_W = P1 − W, G_tot = P0 − W = G_recal + G_W. 보조로 R1(λ 1.0), R2, D1.
- 통계: Spearman ρ(x, y). CI 는 WF4-c 와 같은 결합 재표집이다. 이득은 대상별 재표집 분포의 같은 번호 b, CE2 는 (분할, 추출) 재표집(seed_of('xa', …)), CE1 은 고정. 10,000회, 2.5–97.5 백분위, p_one = mean(ρ^b ≤ 0).
- 독립성 민감도: 30대상(하위 지역 모드 i·x 가 셀을 공유한다), 모드 x 만 20대상, 독립 거시 지역 8(알래스카 x, 레나, 캐나다, 러시아 W·E, Tibet_LGD, Russia_C~lgd, NAtlantic~lic).
- 분해 서술: 대상별 G_recal / G_tot(|G_tot| ≥ 0.5 cm 인 대상만), 주 4지역 층화 평균의 G_recal 과 G_W.
- 해석 규칙(결과 전 고정): ρ(CE2, G_recal) 의 CI 하한 > 0 이고 ρ(CE2, G_W) 의 CI 가 0 을 포함하면 C2 를 '라벨을 쓰는 보정(재보정)의 이득은 계수 오차 크기와 관련되고, 재보정을 넘어선 ML 몫은 관련이 확인되지 않았다'로 쓴다. ρ(CE2, G_W) 의 CI 하한도 > 0 이면 'ML 몫도 계수 오차와 함께 커진다'를 더한다. 두 CI 가 모두 0 을 포함하면 C2 의 비례 문장을 쓰지 않는다.
- 재현 관문: 같은 코드 경로로 WF4-c 의 ρ 0.38 [0.17, 0.55](대상 28)와 WF0 의 ρ 0.66 을 다시 낸다. 다르면 분석을 멈춘다.

### 3.3 (iii) XB: 앵커 계산 가능성과 가중 학습

**앵커별 계산 경로**

| 앵커 | 열(자료) | 지역 내 arm(h54.build_rctx) | 전이 arm(h40.build_ctx + h42.build_unit) | 새 지역(LGD 실행 표) | 보정 |
|---|---|---|---|---|---|
| P1 Stefan | e5_sqrt_tdd | c.sA, c.sB | c.s_src, c.sA, c.sB | 있음 | 척도 c_n = shrink(LS, E0, n, κ 10) |
| P* 연도 정합 도일 | √tdd_matched(`data/processed/lgx_tdd_matched_v1.csv`, loc_id 결합) | df 색인으로 결합. v3 직접 라벨 17,467셀 모두 있음 | ext['tdd_matched'] | 없음(v4 새 셀 516개가 표에 없음) | 척도(c0 원천, c_n 수축) |
| Kudryavtsev 보정 | p4_ku(load_base) | df 열을 A·평가 색인으로 뽑는다 | ext['p4_ku'] | 있음 | 척도 + anchor_fill(비유한·0 이하는 ρ·s, ρ 는 원천 중앙값) |
| 토양형 Stefan | p2_edaphic | 같음 | ext['p2_edaphic'] | 있음 | 같음 |
| 토양 도일 Stefan | e5_sqrt_tdd_soil | B 는 c.sB_soil, A 는 df 에서 | ext['e5_sqrt_tdd_soil'] | 있음(v4 토양 표) | 척도 |
| CCI 원값 | cci_alt, cci_valid | c.XA[:, CCI_COL] | 같음 | 있음 | 없음. 무효 셀(cci_valid < 0.5 또는 결측)은 P1 값 |
| CCI 선형 보정 | cci_alt | 같음 | 같음 | 있음 | 아핀(원천 (a0, b0) → 각 계수 κ 10 수축) |
| Stefan·CCI 평균(Pe) | 위 둘 | h54.re_anchor | 같음 | 있음 | 기준선. P1 과 CCI 원값의 볼록 결합 안에 들어 있어 후보에서 뺀다 |
| 공개 ALT 제품 | XG 표집 표 | XG 뒤 | XG 뒤 | 제품 범위에 따라 | 아핀 |

- 지역 내 arm 은 build_rctx 가 쓰는 색인(half_split_blocks, eval_mask)을 다시 계산하고 c.yA, c.yB 와 같은지 단언한다(h42.build_unit 의 단언과 같은 방식). 원천 계수 c0 는 build_rctx 의 원천(LG 모드 x, 100 km 버퍼) 셀에서 구한다.
- 전이 arm 은 h42.build_unit 을 쓴다. ExtTables 는 표에 없는 loc_id 가 하나라도 있으면 중단하므로, LGD 실행 표에서는 tdd 표에 새 셀을 결측으로 덧붙인 판을 따로 두거나 P* 를 그 대상의 후보에서 뺀다. 규칙: 대상의 A 또는 채점 셀 가운데 5 % 넘게 tdd_matched 가 없으면 P* 를 뺀다(unit.json 에 기록).
- 이미 알려진 예외(라벨 미사용): p4_ku, p2_edaphic 은 load_base 가 공변량 결측을 전체 셀 중앙값으로 채운 열이다(토양 결측 399셀, 그 가운데 396셀이 레나, h42 머리 주석). h42.phys_train_fill(원천 중앙값) 판을 레나 민감도로 둔다.

**가중 학습(누설 없는 절차)**
1. 선택 라벨 L(n 개)의 묶음 F = h42.cv_folds_of(blkA[L], 대상, 모드, 분할, n, 추출). 묶음 2개 미만이거나 n < 10 이면 가중 = P1 단위 벡터(적층 = P1).
2. 교차 적합 기저 행렬 Z(n × K): 묶음 j 마다 후보 k 의 보정 계수를 L \ L_j 로 구해 L_j 를 예측한다.
3. 단체 제약 최소제곱: w = argmin ||y − Z w||², w ≥ 0, Σw = 1. K ≤ 8 이므로 지지 집합 2^K − 1 개를 모두 풀어(등식 제약 최소제곱의 닫힌 해) 음수가 없는 해 가운데 SSE 가 가장 작은 것을 고른다. 동률은 P1 을 포함하고 원소가 적은 지지 집합이다. 반복 해법을 쓰지 않아 플랫폼 사이 결과가 같다.
4. 수축 강도 γ ∈ {0, 0.25, 0.5, 0.75, 1}: 같은 묶음으로 2단 교차검증을 한다. 묶음 j 를 빼고 Z 로 w_j 를 맞춘 뒤 w_j,γ = (1 − γ) w_j + γ e_P1 로 L_j 를 채점한다. SSE 가 가장 작은 γ, 동률이면 큰 γ(보수적).
5. 최종: Z 전체로 ŵ, w* = (1 − γ̂) ŵ + γ̂ e_P1. 채점 셀의 기저 예측은 L 전체로 보정한 계수로 낸다.
6. 잔차 판(StackR): 잔차 모형의 학습 목표는 라벨 행 y − 적층(표본 안, 최종 계수와 w*), 전이 arm 의 원천 행 y − 적층(원천 계수 c0, w*). h40 R1 의 구성(원천 행 E0 앵커, 라벨 행 E_n 앵커)과 같다. λ 0.25 와(지역 내) 교차검증 λ.
7. 라벨 0(전이): 원천 행에서 원천 거시 지역 하나 제외 교차 적합으로 가중을 맞춘 라벨 없는 적층(Stack0). B:ens(등가중), P0, P* 와 비교한다.
- B 라벨과 선택되지 않은 A 라벨은 가중, 계수, γ, 잔차 어디에도 쓰지 않는다. ρ(anchor_fill)는 원천에서만 구한다.

### 3.4 (iv) XH: 검증 사다리의 기존 산출과 최소 추가

**이미 있는 것**
- h41(`data/processed/lgx/ladder/lgv_metrics.csv`, 1,687행): 단 V-R(셀 무작위 5겹), V-P(거친 공변량 묶음), V-S(0.05° 지점), V-B(0.5° 블록 5겹), V-C0·C100·C500(공간 군집 제외·버퍼), V-G(지역 홀드아웃, 원천 = source_idx(지역, 'x'), 100 km 버퍼), W1-지역(지역 내 학습만, 블록 5겹, 반복 5), W2-지역(지역 내 + 타 지역 원천). 지역 알래스카, 레나, 캐나다.
- 단 V-* 의 모형은 통합 모형이다. PS = E_TR·s 의 E_TR 은 학습 집합 전체(여러 지역) 최소제곱이라 지역 재보정 P1 이 아니다. V-G 의 PS, RS, D0 은 각각 P0, R0, D0 에 해당한다.
- 예(알래스카, catboost_lo): V-R D0 12.31, PS 14.26, RS(λ 0.25) 13.57 / V-B D0 16.81, PS 14.41, RS 14.02 / V-G D0 35.56, PS 14.68, RS 15.00 / W1-Alaska D0w 16.08, PSw 14.40, RSw(λ 0.25) 13.92 cm.
- m1(`data/processed/m1/cv_scheme_comparison.csv`, 84행): 알래스카만, 6겹, seed 3, 단 random_cell, site_0.05, block_0.5(·canonical), block_1.0·2.0, knndm, 모형 stefan, ridge, catboost_lo, stefan_ridge_l075. QA Q15 표의 출처다.

**빠진 것**
1. 레나·캐나다의 kNNDM(어느 모형으로도 없다).
2. 지역 재보정 P1·잔차 R1·D0 정의(W1 정의)의 셀 무작위·지점·kNNDM 단. V-R, V-S 는 통합 모형이다.

**최소 추가 실행**
- 단: W1R(셀 무작위 5겹), W1S(0.05° 지점 묶음 5겹, h41.groups_s), W1K(kNNDM 5겹). 지역 알래스카, 레나, 캐나다. 반복 3.
- 모형: h41 의 W1 정의 그대로(PSw = 학습 묶음 E·s, D0w·RSw × catboost_lo·catboost·rf, λ 0.25·1.0). 블록 단은 h41 W1 결과를 쓰고, 지역 홀드아웃은 V-G 를 쓴다. 대회 구성(ridge 잔차 λ 0.75)은 알래스카 연속성 확인용으로 둔다.
- kNNDM 예측 영역: 알래스카는 m1 정의(ERA5-Land 0.02° 세분, 육지 ∩ MAAT < 0, 20,000점), 레나는 `data/processed/map_lena/lena_grid_x25_v1.csv.gz`(WF5 격자), 캐나다는 라벨 경계 상자 ± 1° 의 ERA5-Land 육지 ∩ MAAT < 0 격자[정의는 등록 때 고정].
- 그림: 가로축 = 채점 셀에서 가장 가까운 학습 셀까지의 거리 중앙값(km), 세로축 = RMSE, 선 = P1(지역 홀드아웃은 P0), R1(R0), D0. 무작위 수치를 함께 실어 선행 연구 수치와 잇는다(QA Q15 권고).

### 3.5 XE·XD 의 근거 자료 서술(모형 결과 아님)

**격자 안·1 km 칸 안 ALT 분산 비율**(v3 직접 라벨, 격자 = (블록, 기온 √TDD 값), 칸 = floor(lat/0.01), floor(lon/0.01))

| 지역 | 셀 | 0.01° 칸 | 격자 묶음 | 총 분산(cm²) | 격자 안 비율 | 칸 안 비율 | 칸 안 표준편차(cm) | 칸 안 고도 표준편차(m) |
|---|---|---|---|---|---|---|---|---|
| 알래스카 | 13,606 | 452 | 171 | 302.1 | 0.504 | 0.389 | 10.85 | 3.11 |
| 레나 | 3,037 | 296 | 59 | 444.9 | 0.627 | 0.316 | 11.86 | 3.48 |
| 캐나다 | 750 | 99 | 58 | 961.3 | 0.371 | 0.285 | 16.55 | 0.65 |

- 알래스카 격자 안 분산의 77 %(0.389/0.504)가 0.01° 칸 안에 있다. 그런데 칸 안의 DEM 공변량은 거의 같다(고도 표준편차 3.1 m). 지금 지형 6종은 점마다 33 × 33 화소(약 1 km) 창이고(`scripts/1_data_prep/terrain_features_dem.py`), H19 의 TWI·수체·수목 피복은 1 km 평균이다(`scripts/1_data_prep/build_covariates_ext.py`). 칸 안 변동을 설명할 입력이 사실상 없다.
- 0.01° 칸 수(452)는 QA 의 '1 km 위치 343곳'과 정의가 다르다(이번 점검 정의).
- 오차 하한 11.31 cm(`data/processed/lgx/lgx_floor.csv`)는 기후·토양·CCI 의 같은 값 묶음 안 분산이라 DEM 과 점 규모 입력으로는 더 낮아질 수 있다(`h42.floor_table`).

**블록 크기 편중과 배치 전략의 가중 의존**
- 셀 기준 상위 3블록의 비율: 알래스카 0.593(74블록, 블록 셀 수 중앙값 3), 레나 0.798(20블록, 중앙값 11), 캐나다 0.561(39블록, 중앙값 1).
- WF2 지역 내 R1(λ 0.25) n = 100: 레나 S2 − S1 +1.65 [0.90, 2.40](셀 가중) / [0.15, 1.57](블록 등가중) 열세. 알래스카 S2 − S1 +0.34 [−0.45, 0.66] / [−2.47, −0.97], S4 − S1 +0.44 [−0.40, 0.81] / [−2.48, −0.94] 미결정(`results/rescale_wf/data/processed/wf/wf_tests.csv`).
- 블록마다 같은 수를 주는 배치(S2)는 작은 블록을 많이 재므로 블록 등가중 오차를 줄이지만, 셀이 몰린 큰 블록을 덜 재서 셀 가중 오차를 줄이지 못하거나 늘린다. XD 의 혼합 알고리즘은 블록 배분을 블록 크기에 비례(또는 제곱근 비례)하게 둔다.

---

## 4. 실험별 계획

### 4.1 XA_c2_gain_decomposition(C2 의 재보정 몫과 ML 몫 분리)

- 목적: C2 문장을 재보정 몫(P0 − P1)과 재보정을 넘어선 ML 몫(P1 − R1, P1 − W)으로 나눠 CI 와 함께 보고한다.
- 기존 코드: h54.tests_wf4 의 WF4-c(결합 재표집 Spearman, `_spearman_rows`, seed_of('wf4c', …)), h40.contrast(return_dist), h54.make_tm·find_shards 형식, `wf0_reanalysis.py`.
- 추가: `scripts/2_evaluation/xa_c2_gain_decomposition.py`. 함수 load_wf4_tms(조각 → TMx), gains(tm, n) → 이득 분포 4종, ce_values(units) → CE1·CE2·CE3, joint_spearman(x 표본, y 분포), decomposition_table.
- CLI: `nice -n 10 python3 scripts/2_evaluation/xa_c2_gain_decomposition.py --wf4-shards results/rescale_wf/data/processed/wf/shards --nboot 10000 --threads 4 --allow-local --out-dir data/processed/xbatch/xa`
- 산출: xa_gains.csv(대상 × n × 이득 종류의 점 추정·CI), xa_spearman.csv, xa_decomp.csv, xa_meta.json.
- 적합 수: 0. 시간: 대상 30 × n 4 × 대비 4 의 재표집 10,000회, 로컬 프로세스 1개로 수십 분 이내[추정, 재지 않음].
- 시험(`tests/test_xa_c2.py`): (a) 합성 저장소에서 재표집마다 G_recal + G_W = G_tot(셀 가중·블록 등가중). (b) 동률이 있는 Spearman. (c) [HEAVY] WF4 조각에서 WF4-c 의 ρ 와 CI 를 다시 낸다.
- 재현 점검: WF4-c ρ 0.38 [0.17, 0.55], WF0 ρ 0.66 의 재현(3.2절).
- 위험: 30대상이 독립이 아니다(하위 지역 i·x 공유). CE1 은 대상 전체 라벨을 쓴 오라클이다. 이 분석의 일부(ρ −0.08, 0.07)는 이미 열람했다.

### 4.2 XB_multisource_stacking(여러 물리식·제품의 라벨 학습 가중)

- 목적: 대상 라벨로 물리식·제품의 볼록 가중을 학습해 P1, R1 보다 나은지 본다(QA Q11 의 X1).
- 기존 코드: h42 의 앵커 경로(ANCHORS soil·ku·ed·cci·tddm, anchor_fill, affine_ls, anchor_ctx, B:ens), h54 의 re_anchor·T7Unit(원천 행 잔차 목표)·RUnit.lam_cv, h42.cv_folds_of·shrink. LGX L29 의 기준선 결과(라벨 0·10·전량).
- 추가(`x_multisource_stacking.py`)
  - 순수 함수: simplex_ls(Z, y), oof_base(L, folds, 후보), choose_gamma(Z, y, folds, Γ), calibrate(kind, b_L, y_L, c0, n), anchors_inregion(D, c, …), anchors_transfer(c, ext).
  - XBRUnit(h54.RUnit): 방법 P1, R1(교차검증 λ·고정 λ), Stack, StackR(교차검증 λ·0.25), Pe(기준선), 개별 보정 앵커 P1@b(서술). 같은 추출(h40.draw_cells(대상, 'r', 분할, n, 추출))이라 WF6 와 키가 맞는다.
  - XBTUnit(h54.TUnit): P0, P1, R1(0.25), Re(0.25, WF7 와 같음), Stack, StackR(0.25), Stack0(라벨 0), B:ens.
  - unit.json 에 추출마다 가중 w*, γ̂, 빠진 후보를 적는다(가중 분포 표의 원천).
- 범위: 지역 내 xb_r = 알래스카·레나·캐나다, 분할 1–25(WF6 규칙), n {20, 50, 100, 200, 500, 1,000, 전량}(|A| 미만), 추출 3(전량 1), seed 2. 전이 xb_t = LG 27 (대상, 모드) + LGD 약관 확인분 3 + 캐나다 확충 약관 확인분, 분할 1–5, n {0, 3, 10, 40, 160, 320, 1,000, 전량}, 추출 5, seed 2.
- CLI: `WF_RESCALE=1 python3 scripts/3_deep_learning/x_multisource_stacking.py --exp xb_r,xb_t --workers 22 --threads 4 --resume --no-summarize`, 세기 `--count-only --threads 1`, 스모크 `--smoke --threads 2 --workers 0`(taskset 코어 4개).
- 적합 수·시간(기존 실측으로 환산)
  - xb_r: 단위 74(알래스카 25, 레나 24, 캐나다 25), (n, 추출) 칸 약 1,256개 × 적합 14(R1 2 + R1 교차검증 5 + StackR 2 + StackR 교차검증 5) ≈ 17,600건. WF6 실측 catboost_lo 0.057–0.063 s/건(`results/rescale_wf2/data/processed/wf/wf2b_timing.csv`)으로 약 0.3 워커·시간.
  - xb_t: WF7(117 단위, R1·Re 11,356건, 0.209 s/건, 단위 합 0.675 h)에 StackR 를 더한 규모. 약 19,000건, 약 1.1 워커·시간.
- 가설(등록안)
  - XB-a(대상별, 지역 내): 알래스카·레나·캐나다 각각 Stack 또는 StackR(교차검증 λ)가 P1 보다 우세인 n 이 n ≥ 200 에 있다.
  - XB-b(주 4지역 층화 평균, 전이): StackR(0.25) − R1(0.25) 가 n ∈ {10, 40, 160, 전량} 가운데 하나 이상에서 우세이고 열세가 없다.
  - XB-c(안전, 지역 내 알래스카): Stack − P1 의 두 가중 CI 상한 < 0.5 cm(n ≥ 200 모두).
  - XB-d(서술): 지역·n 별 가중 분포, 가중이 큰 앵커와 그 지역의 P1 오차의 관계.
- 시험(`tests/test_x_stacking.py`): (a) simplex_ls 가 무작위 문제에서 w ≥ 0, Σw = 1 이고 SLSQP 기준해와 SSE 가 같거나 작다. (b) 교차 적합 Z 의 각 행은 자기 묶음 밖 라벨로만 보정된다(추적). (c) B·비선택 A 라벨을 바꿔도 예측·가중이 같다. (d) γ = 1 이거나 n < 10 이면 Stack = P1(정확히 같음). (e) anchor_fill 의 ρ 는 원천에서만. (f) P1·R1·Re 키가 h54 RUnit·T7Unit 과 같다(합성 자료). (g) dry 세기. (h) 설정 해시 고정.
- 재현 점검: xb_r 의 P1·R1(교차검증 λ, 고정 λ) 키가 WF6 조각(`results/rescale_wf2/.../shards/wf6__*`)과 같다. xb_t 의 P0·P1·R1(0.25)·Re 가 WF7 조각과 같다. 허용 차: 같은 노드 종류 |ΔSSE| 0, 다른 노드 1e-6 cm².
- 위험: (1) 결과 열람 뒤 설계(CCI 는 캐나다에서 도움, 알래스카에서 해로움이 알려져 있다). (2) n 이 작을 때 가중의 과적합(γ 교차검증과 n < 10 규칙으로 막는다). (3) 후보 사이 공선성(지지 집합 열거로 해는 결정적이지만 가중 해석은 불안정하다. 가중은 서술만). (4) 새 지역의 P* 부재. (5) p4_ku·p2_edaphic 의 공변량 중앙값 대체 예외.

### 4.3 XC_workflow_end_to_end(워크플로 전체의 순차 적용, 확인 시험)

- 목적: 배치 알고리즘 → 라벨 10개 편향 진단 → 방법 선택 규칙 → 라벨 추가의 순서를 새 지역에 차례로 적용했을 때, 무작위 배치 + 고정 레시피보다 나은지를 실행 전 등록한 규칙으로 시험한다.
- 워크플로 WFX(등록안)
  1. 배치: XD 의 H-prop(블록 크기 비례 배분 + 블록 안 공변량 k-중심, 4.4절)으로 A 풀의 순서를 만든다. 순서는 내포적이다(n = 40 은 n = 10 을 포함).
  2. n = 10: R1(λ 0.25)(C3, AB6). 진단값 d10 = |mean(E0·s − y)|(선택 10개)을 기록한다.
  3. n = 40, 160: 규칙 W+ = W 의 후보(P0, P1, P2, R1@0.25, R1@1.0, R2@0.25, D1)에 Stack, StackR@0.25 를 더한 집합에서 선택 라벨 안 5겹 블록 교차검증으로 고른다(h54.TUnit.select_w 를 확장).
  4. 지역 사이 배분(서술): 대상마다 n = 10 뒤 d10 으로 남은 예산을 나누는 규칙과 균등 배분을 비교한다. 같은 총 예산이 격자 n 위에서만 정의되므로 확인적 가설로 세지 않는다.
- 비교 arm(같은 조각에서 다시 적합): Base = S1 + R1(0.25), S1 + W, S1 + W+, H + R1(0.25), H + W, WFX(= H + W+). W 의 선택은 W+ 의 교차검증 SSE 에서 부분 후보로 다시 고르므로 추가 적합이 없다.
- 기존 코드: TUnit.select_w, diag, h40.draw_cells(S1), RUnit.strategy_sets·kcenter_order, XB 의 Stack 함수.
- 추가(`x_workflow_end_to_end.py`): placement_order(H 계열, 내포적), XCUnit(h54.TUnit), select_wplus, allocation_analysis(대상별 곡선 → 배분 규칙별 층화 평균).
- 대상: 독립 거시 지역 풀 레나 x, 캐나다 x, 러시아 W x, 러시아 E x, 알래스카 x, Tibet_LGD, Russia_C~lgd(+ NAtlantic~lic 점 추정), 분할 1–5, n {10, 40, 160}(|A| 미만), 추출 5, seed 2. 보조 대상으로 하위 지역(AL-1–6, CA-2·3, LE-1·2, 모드 x)을 서술한다.
- |A| 제약(WF4 unit.json): 러시아 W 14–16, 러시아 E 15–17, Russia_C~lgd 26–35 → n = 10 만. Tibet_LGD 63–69 → n 10·40. 레나 1,120–1,737, 캐나다 325–406, 알래스카 x 6,322–7,436 → n 10·40·160.
- 가설(등록안, 확인적)
  - XC-1(주): Δ = WFX − Base 의 층화 평균이 n = 40(풀: 레나, 캐나다, 알래스카 x, Tibet_LGD)과 n = 160(풀: 레나, 캐나다, 알래스카 x)에서 우세. 둘 다 우세면 지지, 하나면 부분 지지, 열세가 있으면 기각.
  - XC-2(안전): n = 10, 풀 7지역에서 WFX − Base 의 두 가중 CI 상한 < 0.5 cm.
  - XC-3(서술): 배치 효과(H + R1 − S1 + R1), 방법 효과(S1 + W+ − S1 + R1), 상호작용, W 와 W+ 의 차이(적층 후보의 몫).
  - XC-4(서술): d10 기반 배분 대 균등 배분.
- 적합 수·시간: (배치, n, 추출) 칸마다 W+ 교차검증 5묶음 × (R1 0.21 s, R2 1.0 s, D1 1.0 s, StackR 0.21 s) + 최종 4방법 × seed 2 ≈ 17 s(적합 28건). 배치 2종, 분할당 칸 75개(15 × 3 + 10 + 5 × 4), 분할 5 → 375 칸 × 34 s ≈ 3.5 워커·시간, 적합 약 21,000건(WF4 실측 `results/rescale_wf/.../wf_timing.csv`: R2·D1 약 1.0 s/건, R1 0.22 s/건). 하위 지역 보조를 넣으면 약 2배.
- 플랫폼: Rescale hematite 또는 elm 의 별도 작업(작업 B). 모든 arm 이 같은 작업 안에서 닫힌다.
- 시험(`tests/test_x_workflow.py`): (a) H 순서의 앞 n 개가 n 의 집합과 같다(내포). (b) W+ 선택이 선택 라벨만 쓴다(B 라벨 교체 불변). (c) d10 은 첫 10개 라벨만 쓴다. (d) S1 + W arm 이 h54 TUnit 의 W 와 같다(합성 자료). (e) 등록 풀과 판정 함수가 합성 저장소에서 기대 문구를 낸다.
- 재현 점검: S1 + W, S1 + R1(0.25) 키가 WF4 조각(elm)과 같다(n 10·40·160).
- 위험: (1) n ≥ 40 의 독립 지역이 3–4개라 검정력이 낮다. (2) W+ 에 적층을 넣는 결정은 XB 결과 전에 고정한다(XB 와 병렬 실행을 위해). (3) 하위 지역은 상위 지역과 셀을 공유해 독립 근거가 아니다. (4) 구성 요소(W, 배치) 결과를 본 뒤 설계했다는 표지를 단다.

### 4.4 XD_placement_policy(배치 알고리즘의 명세와 학습형 정책 시범)

**알고리즘 명세(사용자 질문 2 의 답이 되는 형태)**

현재 근거로 이미 정확히 적을 수 있는 알고리즘은 둘이다.
- S4(공변량 탐욕 k-중심, `h54.kcenter_order`): A 풀의 x25 를 중앙값 대체 뒤 A 풀 평균·표준편차로 표준화한다(라벨 미사용). 첫 점은 seed 로 고르고, 그 뒤 '이미 고른 점까지의 최소 거리가 가장 큰 점'을 차례로 더한다.
- S2(블록 순환, `h40.draw_blocks`): A 의 0.5° 블록을 seed 순열로 놓고 블록마다 하나씩 차례로 뽑는다(블록 안은 무작위).

새로 시험할 혼합 알고리즘 H(등록안). 입력 = A 풀 공변량 X, 블록 b, 예산 n, seed, 이미 있는 라벨 L0(선택).
1. Z = standardize(X)(h54.standardize_fit, 라벨 미사용).
2. 블록 배분 n_b: H-eq 는 균등(S2 와 같은 순환), H-prop 는 블록 셀 수 비례(최대 나머지 방식, `h54.proportional_alloc`), H-sqrt 는 셀 수 제곱근 비례. 배분이 블록 셀 수를 넘지 않게 한다.
3. 순서: 블록을 seed 순열로 놓고, 배분이 남은 블록을 차례로 돌며 블록 안에서 '이미 고른 점(L0 포함)까지의 최소 거리가 가장 큰 셀'을 고른다(동률은 작은 색인).
4. 결과는 내포적 순서이고 결정적이다. 계산량 O(n·|A|).
- 근거: 3.5절의 블록 편중(알래스카 상위 3블록 59 %, 레나 80 %)과 S2·S4 의 셀 가중·블록 등가중 결과 차이. H-prop 는 셀 분포를 보존하면서(무작위와 같은 기대 배분) 블록 안의 공변량 덮임을 넓힌다.

**평가 arm**
- 지역 내(WF2 형식): 알래스카, 레나, 캐나다, AL-1, AL-2, 예산 {20, 50, 100, 200, 500}(|A| 미만), 추출 5, 분할 1–5. 전략 S1, S2, S4(같은 조각에서 다시 적합), H-eq, H-prop, H-sqrt. 방법 P1, R1(0.25), D1.
- 전이(WF8 형식): 레나, 캐나다, 러시아 W·E, 알래스카(모드 x), n {10, 40}, 추출 5, 분할 1–5. 전략 S1, S2, S4, H-prop, H-sqrt. 방법 P1, R1(0.25).
- 가설(등록안): XD-a 각 대상·n 에서 H-prop − S1 의 두 가중 CI 상한 < 0.5 cm(비열등), XD-b 풀(층화 평균)에서 n ≤ 100 의 H-prop − S1 이 우세, XD-c(서술) H-prop 와 S2·S4 의 차이.
- 적합 수: 지역 내 전략당 R1 약 1,024건(0.038 s)·D1 약 1,024건(0.075 s)(WF2 실측 평균), 전략 6종 약 0.2 워커·시간. 전이 전략당 R1 약 390건(0.22 s), 5종 약 0.12 워커·시간.

**학습형 정책 시범(사용자 질문 1 의 앞부분)**
- 과제(task) = (대상, 분할). 대상 = 지역 내 |A| ≥ 200 인 알래스카, 레나, 캐나다, AL-1–AL-6, LE-1, LE-2(10개 안팎). 하위 지역은 같은 거시 지역 안이라 검증은 거시 지역 하나 제외(알래스카 계열 / 레나 계열 / 캐나다)로 한다. XF 에서 새 지역이 생기면 시험 과제로 더한다.
- 후보 집합 생성: 과제·n ∈ {20, 50, 100} 마다 256개 집합(S1, S2, S3, S4, H 계열을 seed 를 바꿔 섞고 일부 셀을 무작위로 바꾼 섭동 집합). 결과 = 그 집합으로 맞춘 R1(0.25, seed 0)과 P1 의 채점 RMSE 와 블록 SSE(BlockStore 저장). 약 10 × 5 × 3 × 256 = 38,400건 × 약 0.04 s ≈ 0.45 워커·시간(Rescale, 작업 A).
- 정책: 집합 특징(라벨 미사용: 덮임 반경, 블록 수, 블록 배분의 크기 상관, 쌍 거리 평균, √TDD 범위 덮임, A 풀의 반경 r 안 비율)을 넣은 GBM(CatBoost CPU)과, 셀 특징 집합을 받는 DeepSets(φ: MLP → 평균 풀링 → ρ: MLP, PyTorch, 로컬 GPU 0–4, seed 5개)를 학습해 '같은 (과제, n)의 무작위 평균 대비 RMSE 변화'를 예측한다.
- 평가: 남겨 둔 거시 지역의 각 (과제, n)에서 정책이 256개 후보 가운데 고른 집합의 실제 RMSE 를 S1 평균, H-prop, 오라클(256개 중 최선)과 비교한다. 결과값은 모두 Rescale CPU 적합에서 왔고 GPU 는 선택만 한다. 선택 목록, 모형 해시, torch 판, seed 를 기록한다.
- 위치: 원고 본문 주장이 아니라 SI 탐색 분석이다. XD-a·b 가 지지되면 본문의 배치 규칙은 H-prop 다.
- 시험(`tests/test_x_placement.py`): (a) H 계열이 A 색인 안의 서로 다른 정수를 내고 결정적이며 내포적이다. (b) 배분 합 = n, 블록 셀 수를 넘지 않는다. (c) 블록 안 선택이 작은 문제에서 전수 계산의 최대 최소 거리와 같다. (d) 후보 생성 재현. (e) 정책 특징이 라벨을 읽지 않는다(추적). (f) 거시 지역 하나 제외 분할에 과제 누설이 없다.
- 재현 점검: S1, S2, S4 의 키가 WF2·WF8 조각과 같다.
- 위험: (1) 독립 거시 지역이 3개라 학습형 정책의 일반화 근거가 약하다. (2) 후보 집합 밖의 배치는 평가하지 못한다. (3) DeepSets 는 환경마다 결과가 달라질 수 있어 같은 환경에서만 비교한다. (4) 셀 가중과 블록 등가중의 결론이 갈릴 수 있다(두 지표 모두 보고).

### 4.5 XE_hires_covariates(격자 안 변동을 설명할 고해상도 입력)

- 목적: 격자 안·칸 안 변동을 설명할 점 규모 입력이 있으면 R1 의 격자 안 RMSE 가 줄어드는지 본다(사용자 질문 3, C8).
- 이미 시험한 거친 판(QA Q3·Q8, `data/processed/covariates_ext_v1_meta.json`): MODIS LST(0.05°), TWI·유역 누적·곡률(DEM 30 m → 90 m 평균, 1 km 평균), JRC 수체 빈도(1 km·5 km 창), Hansen 수목 피복(1 km 평균), MODIS NDVI·EVI·NDWI(0.05°), ERA5-Land 토양 수분 swvl1·2, 피부 온도 도일, LAI, 적설 밀도·깊이, 강수(0.1°). 모두 이득이 없었다(H18·H19). 토양 유기 탄소는 SoilGrids(약 5 km 요청 추출)로 x25 에 들어 있다.
- 새 입력(1단계, 로컬 원자료만): P30 = Copernicus DEM 30 m 화소의 고도·경사·향, 90 m(3 × 3)와 약 330 m(11 × 11) TPI, 3 × 3 곡률, 30 m TWI(타일 창 안 pysheds), V30 = Hansen 2000 수목 피복 30 m 화소와 90 m 평균, GSW 수체 빈도 30 m 와 빈도 50 % 이상 화소까지의 거리. 원자료: `data/raw/dem`(타일 455개, 8.6 GB), `data/raw/hansen`(11 GB), `data/raw/gsw`(3.1 GB). DEM 타일이 모든 라벨 위치를 덮는지 `data/processed/dem_tiles_missing.csv`(126행)와 대조해 다시 확인한다[미확인].
- 새 입력(2단계, 내려받기 필요)[미확인]: SoilGrids 250 m 원해상도, MODIS 적설 지속 기간(500 m), Landsat·Sentinel-2 식생 지수(30 m·10 m), 10 m 토지 피복. 자료원·약관·해상도는 조사 과제의 확인 뒤 정한다.
- 추가: `scripts/1_data_prep/xe_point_covariates.py`(loc_id 키 표 `data/processed/xe_point_cov_v1.csv`와 메타), `x_hires_covariates.py`(XERUnit ⊂ h54.R9Unit: XAf·XBf 를 입력 집합 x25, x25+P30, x25+P30+V30 으로 바꾼다. 저장소 = 총, 격자 안 `~w`, 격자 사이 `~b`, 칸 안 `~l`(새 묶음 = (격자, 0.01° 칸))).
- 범위: 알래스카·레나·캐나다 지역 내, 분할 1–25, n {200, 500, 1,000, 전량}, 추출 3, seed 2(WF9 와 같은 추출). 방법 P1, R1(교차검증 λ, 0.25), D0(catboost).
- 가설(등록안): XE-a 알래스카 전량의 격자 안 RMSE: R1(x25+P30+V30) − R1(x25) 우세. XE-b 세 대상 층화 평균의 같은 대비(n ≥ 500). XE-c(서술) 칸 안 설명 비율, 총 RMSE 대비.
- 해석 규칙: XE-a 가 지지면 C8 에 '점 규모 지형·피복 입력으로 격자 안 변동의 x % 를 설명했다'를 쓴다. 기각이면 '공개 30 m 입력으로도 격자 안 상세도는 늘지 않았다'를 쓰고 한계(측정 잡음, 점과 1 km 셀의 지지 규모 차이)를 적는다.
- 적합 수: 입력 집합당 WF9 규모(R1 1,174 + R1 교차검증 2,858 + D0 1,174건, 0.063·0.056·0.42 s/건, `results/rescale_wf3/.../wf3b_timing.csv`) 약 0.2 워커·시간, 3종 약 0.6(x25 는 WF9 재현용으로 다시 돈다).
- 시험(`tests/test_x_hires.py`): (a) 합성 래스터에서 점 추출·창 통계가 수동 계산과 같다. (b) 입력 집합 전환이 맞는 열을 넘긴다. (c) 네 저장소의 분해 항등식(총 = 격자 안 + 격자 사이, 격자 안 = 칸 안 + 칸 사이). (d) x25 판이 h54.R9Unit 과 같다.
- 재현 점검: x25 판의 총 저장소가 WF9 조각과 같다(WF9 는 WF6 와 74/74 일치했다).
- 위험: 고해상도 입력이 원천 지역에 없으면 전이 arm 은 할 수 없다(지역 내만). 측정 잡음이 칸 안 분산의 상당 부분일 수 있다. 래스터 추출이 로컬 입출력에 오래 걸릴 수 있다.

### 4.6 XF_new_regions(공개 자료로 독립 지역 추가)

- 목적: 독립 지역 수를 늘려 일반화 근거를 보강하고 핵심 대비(LG L1·L4·L8 형식, WF4 W·진단, XB, XC)를 다시 계산한다(사용자 질문 1).
- 기존 코드: parse_ext_* 20여 개, `build_ext_cells_v1.py`, `ext_cells_covariates_v1.py`, `era5land_soil_tdd.py`, `build_fidelity_base_v4.py`(v3 행 해시 점검), `lgd_eligibility_v1.py`(유효 분할 ≥ 1, 사용 분할 채점 블록 합집합 ≥ 8), h51(실행 표 → h40.main), h52(P4·PE1·PE2 풀).
- 추가: 자료원별 parse 스크립트, `build_ext_cells_v2.py`, 새 셀의 연도 정합 도일(라벨 연도 + `data/raw/era5land/nh_monthly_2010-2024.nc`, `a2_year_matched_tdd.py` 의 정의), `build_fidelity_base_v5.py`(v3·v4 행 불변 단언), `lgd_eligibility_v2.py`, `x_new_regions.py`(h51 이 v4 표 이름에 묶여 있어 실행 표 생성과 h40·xbatch 호출을 일반화), `x_new_regions_pool.py`(h52 규칙 재사용).
- 독립성 규칙(등록안): 새 지역의 셀은 기존 대상 셀에서 100 km 넘게 떨어져 있어야 한다(h40 버퍼와 같은 값). 약관 확인분만 주 판정에 쓴다.
- 적합 수: 새 지역 표 하나당 LG CPU 방법 축 890–1,265건, 프로세스 1개 0.23–0.32 h(`data/processed/lgd/lgd_count_summary.csv`의 Tibet·NAtlantic·Russia_C). 여기에 WF4·XB·XC arm 이 표마다 약 0.5 워커·시간[추정].
- 시험: v5 의 v3·v4 행이 바이트 단위로 같다, 실행 표의 대상 셀 수가 적격 표와 같다, 새 셀의 공변량 결측률, 맹검 규칙(새 지역 라벨 통계를 실행 전 출력하지 않는다).
- 위험: 자료 확보와 약관이 일정의 대부분을 정한다. 적격 조건(채점 블록 합집합 8)을 못 넘으면 점 추정만 남는다. 후보 자료 목록은 조사 과제의 산출을 입력으로 받는다(이 문서에서는 정하지 않는다).

### 4.7 XG_product_comparison(기존 ALT 지도 제품과 같은 시험지 비교)

- 목적: '기존 지도보다 나은가'에 같은 채점 셀로 답한다(QA Q9, Q12).
- 기존 코드·자료: LGX 의 셀 단위 예측(`data/processed/lgx/shards/lgxb__*_cells.npz`, `lgx1__*_cells.npz`, 250 파일; 키 = (method, learner, alpha, placement, n, draw, seed, comp), comp = pred 또는 잔차 성분 g, 키별 앵커 계수 E, 셀 loc_id·y·s·block). n ∈ {0, 10, 40, 160, 전량}. P1 = E·s, R1 = E·s + λ·g 를 다시 만들 수 있다.
- 추가: `xg_sample_alt_products.py`(제품 래스터를 모든 셀 loc_id 에서 최근접·이중선형으로 표집, 제품 기간 기록), 누설 표(제품 학습 지점 목록과 각 셀의 거리, 학습 포함·10 km 안 표지), `x_product_comparison.py`(cells.npz 와 제품 값을 loc_id 로 결합해 같은 채점 셀에서 블록 SSE 를 만들고 재표집). 제품은 XB 의 앵커 후보로도 넣는다(누설 점검 통과분만).
- 비교: 제품 원값, 제품 아핀 보정(라벨 n 개), P0, P1, R1(0.25), Stack(XB). 누설 층: 전체, 제품 학습 지점 제외, 10 km 안 제외.
- 적합 수: 전이 부분 0(분석만). 지역 내 비교가 필요하면 XB 의 xb_r 에 제품 앵커를 넣어 함께 돈다.
- 시험: cells.npz 에서 다시 만든 블록 SSE 가 lgxb BlockStore 와 같다, 누설 마스크가 지정 셀을 뺀다, 표집 함수가 합성 래스터에서 맞다.
- 위험: 제품의 확보 가능성·약관·학습 자료 목록·기간·해상도가 [미확인]이다(Ran 2022, Wei 2026, Liu Z. 2024 는 QA 문서가 든 이름이다). 학습 목록이 공개되지 않은 제품은 '누설 가능' 표지를 달고 SI 로만 쓴다.

### 4.8 XH_validation_ladder(검증 사다리)

- 계획은 3.4절. 추가 스크립트 `scripts/2_evaluation/x_validation_ladder_regional.py`(h41 을 동결 모듈로 불러 단위 형식·모형 함수·집계를 재사용하고 단 W1R·W1S·W1K 를 더한다), `src/polar/cv_schemes.py`(kNNDM).
- 적합 수: 지역 3 × 단 3 × 반복 3 × 묶음 5 × (D0w·RSw × 학습기 3) = 810건. 알래스카 학습 묶음 약 10,900행에서 catboost_lo 약 0.23 s, catboost·rf 는 h42 의 가정값(catboost 2.2 s, rf 4.0 s @ 16,700행)이라 재지 않았다. 약 0.5–1 워커·시간[가정 포함]. kNNDM 묶음 구성은 m1 전체 실행이 약 15–20분이었다(m1 머리 주석).
- 시험(`tests/test_x_ladder.py`): 묶음이 서로 겹치지 않고 셀을 덮는다, cv_schemes.knndm 이 m1 함수와 같은 입력에서 같은 묶음을 낸다(작은 합성 자료), W1K 의 학습·채점 교집합 0, 예측 영역 정의의 결정성.
- 재현 점검: (1) 블록 단을 새 스크립트로 한 번 다시 돌려 h41 W1 지표와 같음을 확인한다. (2) 알래스카 6겹 random_cell·site_0.05·block_0.5 의 stefan·ridge·catboost_lo 를 m1 정의로 다시 돌려 `cv_scheme_comparison.csv` 와 1e-6 cm 안에서 같음을 확인한다.
- 위험: 캐나다 kNNDM 의 예측 영역 정의가 결과를 바꿀 수 있다. m1(6겹)과 h41(5겹)의 묶음 수가 달라 QA Q15 표와 숫자가 조금 다르다(그림은 새 5겹 값, 6겹은 SI 교차 확인).

### 4.9 XI_climate_extrapolation_retest(WF10 재시험)

- 실행(새 코드 없음, h54 그대로)
  - 알래스카: `WF_RESCALE=1 python3 scripts/3_deep_learning/h54_workflow.py --exp wf10 --wf10-targets Alaska --data-dir data/processed --tag xi --out-dir data/processed/xbatch/xi --workers 22 --threads 4 --resume --no-summarize`
  - 캐나다: 같은 명령에 `--wf10-targets Canada --data-dir data/processed/lgd/run_tables/Canada_expanded_lic`
  - 집계: `--summarize-only --exp wf10 --wf10-targets Alaska,Canada --tag xi --out-dir data/processed/xbatch/xi`(표 xi3b_*). 공통 설정 해시는 data_sha 를 빼므로 두 자료판의 조각을 함께 집계할 수 있다.
- 확인 사항: 실행 표 디렉터리에는 하위 지역 대응표가 없어 h40 이 k-평균으로 하위 지역을 계산한다(거시 대상 캐나다에는 영향 없음). 캐나다 E0 는 캐나다 밖 v3 원천에서 구하므로 v3 와 같다.
- 변형 warm_trim(xbatch_core, R10Unit 하위 클래스): A 에서 s ≥ min(s_W) 셀을 뺀다. 판정 규칙은 WF10-a·b·c 와 같고 서술 표지를 단다.
- 적합 수: 캐나다 20 단위 1,600건(이번 세기), 알래스카 warm 10 단위, warm_trim 10 단위. WF10 실측 합 0.085 워커·시간(`wf3b_timing.csv`)의 약 2배, 약 0.2 워커·시간.
- 재현 점검: 알래스카 wf10 키가 3차 본 실행 Vppbeb 조각(`results/rescale_wf3/data/processed/wf/shards/wf10__cpu__Alaska__*`)과 같아야 한다. 대조로 캐나다 v3 판도 한 번 돌려 Vppbeb 캐나다 키와 같음을 확인한다.
- 가설(등록안): XI-a·b·c = WF10-a·b·c 와 같은 규칙, 풀 = 알래스카(v3) + 캐나다(v4 약관 확인분). XI-d(서술) cold, warm_trim, 분할별 외삽 비율.
- 위험: 3.1절의 외삽 폭 문제. 알래스카 WF10 대상 행은 이미 열람했다. 자료판이 섞인 풀(알래스카 v3, 캐나다 v4)임을 표에 적는다.

### 4.10 XJ_tempderived_aux_labels(지온 유도 라벨의 보조 사용)

- 자료(v4, `data/processed/fidelity_base_v4.csv`): F4_calm_temp 68행(몽골·중앙아시아 46, 알프스 9, 티베트 6, 스발바르 3, 캐나다 2, 스칸디나비아 2), F3_ext_temp 39행(Tibet_LGD), F2_gtnp_env 37행(시베리아 15, 스발바르 10, 미국 9, 알프스 2, 남극 1), F2_ext_unknown 148행(러시아 W 야말, 약관 규칙상 어느 표에도 넣지 않는다).
- 설계: (1) 원천 쪽: 보조 행을 가중 w ∈ {0, 0.1, 0.3, 1}로 원천 행에 더한다(h42.stack_rows 의 가중, 지온 유도 표지 열을 입력에 더해 정의 차이를 모형이 흡수하게 한 판 포함). 대상 주 4지역, n {0, 10, 40}. (2) 대상 쪽: Tibet_LGD 에서 지온 유도 39행을 A 풀의 보조 라벨로 두고 직접 라벨 n ∈ {0, 3, 10} 일 때 계수 보정에 가중 w 로 넣는다. 채점은 직접 라벨 셀만.
- 추가: `x_tempderived_aux_labels.py`(load_base(sources=…)로 보조 행을 읽는 Data 판, 채점 제외 단언).
- 적합 수: 약 0.3 워커·시간[추정].
- 가설: 확인적 가설로 세지 않는다(서술). 원천 쪽 보조 행은 원천 1만 4천–1만 7천 행에 비해 100여 행이라 효과가 작을 것으로 본다.
- 우선순위: 가장 낮다. 시간이 모자라면 뺀다. 이유: 원천 쪽 표본이 작고, 대상 쪽은 티베트 한 지역뿐이며, L39(티베트 지온 유도 라벨을 직접 라벨로 둔 변형)가 이미 '강건'이었다(`data/processed/lgd/lgd_tests_lic.csv`).
- 시험: 보조 행이 채점에 들어가지 않는다, w = 0 이면 기준과 같다, 가중이 학습 행렬에 맞게 들어간다.

---

## 5. 적합 수·시간·비용 종합

per-fit 시간은 Rescale 4스레드 워커의 실측이다(`results/rescale_wf/data/processed/wf/wf_timing.csv`, `results/rescale_wf2/data/processed/wf/wf2b_timing.csv`, `results/rescale_wf3/data/processed/wf/wf3b_timing.csv`). 지역 내 catboost_lo 0.04–0.07 s, 지역 내 유사라벨 R2·D1 0.10–0.14 s, catboost 기본 용량 0.26–0.42 s, 전이 R1·Re 0.21–0.25 s, 전이 R2·D1 1.0–1.09 s 다. 단위 경과 시간의 합은 적합 시간의 합과 거의 같았다(WF4 6.61 대 6.63 h).

| 실험 | 적합(건, 추정) | 워커·시간(4스레드) | 플랫폼·작업 |
|---|---|---|---|
| XA | 0 | 로컬 분석 | 로컬 CPU |
| XB(xb_r, xb_t) | 약 36,600 | 약 1.4 | Rescale 작업 A |
| XC(핵심 7–8지역) | 약 21,000 | 약 3.5(하위 지역 보조 포함 약 7) | Rescale 작업 B |
| XD(알고리즘 arm, 후보 집합) | 약 52,600 | 약 0.8 | Rescale 작업 A |
| XD(학습형 정책) | GBM·DeepSets | GPU 수 시간 이내[추정] | 로컬 GPU 0–4 |
| XE(적합) | 약 15,600 | 약 0.6 | Rescale 작업 A |
| XE(공변량 추출) | 0 | 로컬 입출력 | 로컬 CPU |
| XF | 지역당 약 1,000–2,000 + arm | 지역당 약 0.8[추정] | 자료 확보 뒤 별도 작업 |
| XG | 0(전이) | 로컬 분석 | 로컬 CPU |
| XH | 약 810 + kNNDM 구성 | 약 0.5–1[가정 포함] | Rescale 작업 A |
| XI | 약 3,400 | 약 0.2 | Rescale 작업 A |
| XJ | 소량 | 약 0.3 | Rescale 작업 A |

- 작업 A(elm 96코어, 워커 22): 약 4 워커·시간 → 실행 약 0.2 h, 설치 0.1 h, 집계(재표집 10,000회, 분할 25개 실험 포함) 0.3–0.5 h. 상한 2 h, 최악 6.48 달러(3.24 달러/시간).
- 작업 B(hematite 64코어 워커 14 또는 elm): 약 3.5–7 워커·시간 → 실행 0.3–0.5 h, 집계 0.2 h. 상한 2 h, 최악 8.44 달러(4.22 달러/시간).
- 두 작업 합계 예상 5–10 달러. elm 이 배정되지 않으면(10-02 의 60분 대기 사례) hematite 판으로 다시 제출한다.

---

## 6. 실행 순서와 병렬화

| 단계 | 내용 | 자원 | 의존 |
|---|---|---|---|
| 0 | 등록 문서 커밋(가설·판정·해석 규칙·열람 상태) | 로컬 | 없음 |
| 1a | xbatch_core, XB, XD, XI 변형, XH 코드와 단위 시험(taskset 코어 4개, 스레드 2) | 로컬 | 0 |
| 1b | XC 코드(XB·XD 함수 사용)와 시험 | 로컬 | 1a 의 함수(결과 아님) |
| 1c | XE 점 공변량 추출, XG 제품 표집(확보분), XH 예측 영역 표 | 로컬 CPU 4스레드, nice 10 | 0 |
| 1d | XA 분석(학습 없음) | 로컬 CPU 4스레드 | 0 |
| 2 | Rescale 스모크(단계 env·deps·smoke·pytest·count·precheck·project, run_wf.sh 와 같은 구조) | Rescale 1 h 상한 | 1a–1c |
| 3a | 작업 A: XB, XD, XE, XH, XI, XJ | Rescale elm | 2 |
| 3b | 작업 B: XC | Rescale hematite 또는 elm | 2 |
| 4 | XD 학습형 정책(작업 A 의 후보 집합 결과를 받아 GPU 0–4 에 과제 묶음·seed 를 나눔) | 로컬 GPU | 3a |
| 5 | 재현 관문 확인(4절 각 항목) 뒤 판정 표 열람, 등록 문서에 판정 기록 | 로컬 | 3a, 3b, 4 |
| 6 | XF(자료 확보·약관 확인 뒤), XG 의 지역 내 부분 | 로컬 + Rescale | 조사 과제 |

- 원고·PPT 작업과는 판정 표 열람(5단계) 시점만 맞추면 된다. 1–4 단계는 원고 작업과 병렬로 진행할 수 있다.

---

## 7. 사용자 질문과 실험의 대응

| 질문 | 실험 | 이 계획이 답하는 범위 |
|---|---|---|
| 1. 독립 지역을 늘릴 수 있나, 알래스카에서 시험하고 다른 지역으로 옮겨 보는 방향 | XF(새 지역 파이프라인), XD 학습형 정책(거시 지역 하나 제외: 알래스카 계열로 학습해 레나·캐나다에서 검증), XC(새 지역 포함 확인 시험) | 파이프라인과 검증 방식. 후보 자료 목록은 조사 과제 몫 |
| 2. '고르게 퍼뜨린다'를 알고리즘으로 | XD 의 S2·S4 명세(이미 코드로 정의)와 H-prop 등록, XC 의 순차 적용 | 4.4절 알고리즘. H-prop 의 근거는 블록 편중과 가중별 결과 차이(3.5절) |
| 3. 토양 수분·유기층·식생·적설 입력을 썼나 | XE | 거친 판(ERA5-Land 0.1°, MODIS 0.05°, 1 km 평균)은 이미 썼고 이득이 없었다. 점 규모(30 m) 판은 쓴 적이 없다 |
| Q11·Q9·Q12(이전 질문) | XB, XG, XH | 적층, 제품 비교, 검증 사다리 |
| C2 문장 | XA | 재보정 몫과 ML 몫의 분리 |
| WF10 판정 불가 | XI | 캐나다 v4 로 CI 조건 충족, 외삽 폭 한계 |

---

## 8. 열린 문제

1. XC 의 W+ 에 적층 후보를 넣는 결정을 XB 결과 전에 고정할지(병렬 실행 가능), XB 뒤로 미룰지(조건부 등록) 사용자 결정이 필요하다. 이 문서는 전자를 권한다.
2. XI 의 주 판정 판: 약관 확인분(787셀) 대 전체판(825셀). 등록 규칙상 약관 확인분이 주다. 외삽 폭 문제로 warm_trim 을 주 판정으로 올릴지도 정해야 한다.
3. XE 2단계 자료(적설 지속, 고해상 식생, SoilGrids 원해상도)의 자료원·계정·약관 [미확인].
4. XG 제품(Ran 2022, Wei 2026, Liu Z. 2024)의 확보 경로·학습 자료 목록·약관 [미확인].
5. XF 후보 자료 목록과 약관 [미확인]. 몽골은 라벨이 모두 지온 유도라 직접 라벨 지역으로 쓸 수 없다(`docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md`).
6. 캐나다 kNNDM 예측 영역의 정의(경계 상자 폭, 영구동토 마스크).
7. catboost 기본 용량과 RF 의 1만 행 규모 적합 시간은 재지 않았다(XH 의 시간 추정에 가정값이 들어 있다). Rescale 스모크의 사전 점검으로 다시 잰다.
8. P*(연도 정합 도일)를 새 지역 셀에 만들려면 라벨 연도 정보가 필요하다. v4 새 셀의 연도 열 유무를 확인해야 한다[미확인].
9. 이번 점검에서 XA 관련 상관 하나(ρ 0.07)를 새로 보았다. 등록 문서의 열람 상태에 적어야 한다.

---

## 9. 근거 경로

- 코드: `scripts/3_deep_learning/h40_label_grid.py`, `h42_label_grid_ext.py`, `h54_workflow.py`, `h51_lgd_run.py`, `m1_cv_scheme_comparison.py`, `scripts/2_evaluation/h41_validation_ladder.py`, `h52_lgd_pool.py`, `wf0_reanalysis.py`, `src/polar/m1_core.py`, `src/polar/h4_common.py`, `src/polar/fidelity.py`, `scripts/1_data_prep/terrain_features_dem.py`, `build_covariates_ext.py`, `tests/test_h54_workflow.py`, `tools/rescale_client.py`, `scripts/rescale/run_wf.sh`, `make_payload_wf.sh`, `configs/rescale/wf_full.yaml`, `wf2_full.yaml`, `wf3_full.yaml`, `wf3_full_hematite.yaml`.
- 시간 표: `results/rescale_wf/data/processed/wf/wf_timing.csv`, `results/rescale_wf2/data/processed/wf/wf2b_timing.csv`, `results/rescale_wf3/data/processed/wf/wf3b_timing.csv`.
- 판정·결과 표: `results/rescale_wf/data/processed/wf/wf_tests.csv`, `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`, `data/processed/wf/wf0_misspec.csv`, `data/processed/lgx/ladder/lgv_metrics.csv`, `lgv_meta.json`, `data/processed/m1/cv_scheme_comparison.csv`, `data/processed/lgx/lgx_floor.csv`, `data/processed/lgd/lgd_tests_lic.csv`, `data/processed/lgd/lgd_count_summary.csv`, `data/processed/lgd/lgd_run_manifest.json`.
- 자료: `data/processed/fidelity_base_v3.csv`, `fidelity_base_v4.csv`, `fidelity_base_v4_meta.json`, `data/processed/lgd/run_tables/Canada_expanded/`, `Canada_expanded_lic/`, `data/processed/lgx_tdd_matched_v1.csv`, `data/processed/covariates_ext_v1_meta.json`, `data/processed/dem_tiles_missing.csv`.
- 조각: `results/rescale_wf/data/processed/wf/shards/wf4__*`(132 단위), `results/rescale_lg/data/processed/lg/shards/`(1,044 파일), `data/processed/lgx/shards/*_cells.npz`(250 파일), `results/rescale_wf3/data/processed/wf/shards/wf10__*`.
- 문서: `docs/QA_FINAL_REVIEW_2026-10-02.md`, `docs/RESEARCH_OVERVIEW_2026-10-02.md`, `docs/EXPERIMENT_PLAN_WF_2026-10-01.md`(5.1–5.3, 개정 이력), `docs/LGD_LICENSE_CONTACTS_2026-09-30.md`, `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md`.
- 이번 점검의 일회성 계산(파일 저장 없음): XI 블록·분할 표(3.1절), 분산 분해(3.5절), 블록 편중(3.5절), wf0_misspec 상관 재계산(3.2절), `nvidia-smi`·`free -g`(2.3절).
