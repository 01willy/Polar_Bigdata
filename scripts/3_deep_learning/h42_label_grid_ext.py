"""H42 · 라벨 격자 확장 하네스(LGX). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A(개정 5 사전 등록, 개정 7 조작적 정의)의 구현.

목적
  본 실행(LG, h40)의 설계를 그대로 두고 확장 항목 X1, X2, X3c, X5, X6, X9, N1–N4 를 같은 분할, 같은 라벨 추출, 같은 채점으로 돌린다.
  검증 사다리 X3a·X3b(가설 L19–L22)는 scripts/2_evaluation/h41_validation_ladder.py 가 맡는다. 이 파일은 L9–L18, L23–L31 을 판정한다.

동결 규칙
  h40(scripts/3_deep_learning/h40_label_grid.py)과 src/polar 의 기존 모듈을 고치지 않는다. h40 은 importlib 로 파일 경로에서 읽어
  sys.modules['h40_label_grid'] 에 등록하고(모듈 최상위) 자료 적재(Data), 분할과 추출(build_ctx, draw_cells, cells_of), 학습(Fitter, cb_fit),
  조각 기록(cfg_hash, _atomic_text), 채점(BlockStore), 집계(TMx, contrast, strat_mean, build_curve)를 그대로 쓴다.
  본 실행 조각(data/processed/lg/shards, tag lg)은 읽기만 한다. 확장의 산출은 data/processed/lgx 에 둔다.

기호
  s = e5_sqrt_tdd. 원천 행 S, 대상 A 풀, 선택 라벨 L(h40.draw_cells 의 sel), 채점 셀 B(B 블록의 eval_mask 셀).
  E0 = Σ_S s·y / Σ_S s². E_ls = Σ_L s·y / Σ_L s². E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10, n = 0 또는 E_ls 비유한이면 E0. x = x25.
  CB = catboost_lo(반복 200, 학습률 0.05, 깊이 3, l2 3, 스레드 4). 유사라벨 색인 ps(r) = RandomState(seed_of('lg-ps', 대상, 모드, 분할, seed))
  .choice(|A|, round(r·|S|), replace=True). r = 10 이면 h40 과 같은 색인이다.

축(--axes), 조각 tag, 범위
  cpu  base  lgxb   대상 27, 분할 1–5, n 전 격자, 추출 5, seed 2. P0, P1, P2, D0, D1, R0, R1, R2(h40 과 같은 학습 행렬).
                    셀 단위 저장(cells.npz)과 거리 3분위 층 키(#near, #mid, #far. n ∈ {10, 40, 160, 전량}).
       x1    lgx1   같은 범위. D0t, D0c, D0m, F1a, F1k, F1n, V2, V2c, V2r, RM, RMc, RM0, R1@f1k. F1k, F1n, V2, RM 은 셀 단위 저장.
       x2    lgx2   n {0, 10, 40, 160}. D1@r·R2@r(r 1, 3, 30), D1@shuffle·const_t·const_src·tddlin·ku(r 10), R1·R2 의 alpha '10'·'100'
                    (n 10, 40, 160), R1s·P1s(n 10, 40, 160, 전량). 모방 지수는 runs 의 imit_rms 열.
       x9    lgx9   κ {3, 30} 의 P1@k·R1@k, 앵커 b ∈ {soil, ku, ed, cci, tddm} 의 P0@b·P1@b·R0@b·R1@b, R0@x23c·R1@x23c, N1 의 B:* 행.
                    해석식은 n 전 격자, 적합은 n {0, 3, 10, 40, 160, 전량}.
                    결측 대체 민감도(개정 7): P0@ku_tr, P0@ed_tr, B:ku_raw_tr, B:ed_raw_tr, B:ens_tr(n = 0 해석식), F1k@tr(n {0, 10} 적합).
       n3    lgxn3  대상 12, n {0, 10, 40, 160, 전량}, 추출 2, seed 2. 학습기 catboost(600, 깊이 6, 0.03), rf(500), catboost_tuned.
                    방법 D0, R0, R1(catboost 는 R2 포함). 선택 표 <tag>_tuned_select.csv 를 먼저 만든다.
       prox  lgxp   S_rand5, S_randm(반복 3, 채점 셀 무작위 5-fold), inblk(n 10, 40, 160, 추출 5), S_buf(10–500 km). 방법 P1, D0, R1.
       x5    lgx5   교차 적합 구간(점수 |log(y/ŷ)|, P1·R0·R1, 목표 80 %·90 %, seed 0, λ 0.25), n = 0 구간, ShapValues(D0, F1k, R1 의 g).
       x9f   lgx9f  셀 제외 변형 noearly, probe. P0, P1, D0, R0, R1. 저장소 이름 '<대상>~<변형>|<모드>'.
       n4    lgxn4  대상 15, 분할 1–50, n {0, 10, 40, 전량}, 추출 2, seed 2. P0, P1, P2, D0, R0, R1.
       n4a   lgxn4a 대상 15, 분할 1–200, n {3, 10, 40, 160, 전량}, 추출 5. 해석식 P0, P1, P2 만.
       ridge lgxr0  대상 12, 학습기 축 격자. ridge 로 D0, R0, R1, R2, F1k, F1n, V2, RM.
  gpu  r0    lgxr0  R0 × realmlp, ftt, tabm, cfm, ddpm, nflow. 분할 1–5.
       g45   lgxg   h40 의 gpu 부분(D0·R1·R2 × 학습기 7종, MLP α 축)을 분할 4·5 에만 실행(h40._worker_run 호출).
       nn    lgxnn  F1k, V2, RM × mlp, tabm, ftt.
  --targets 를 주면 모든 축이 그 대상만 쓴다. --splits, --n-grid, --draws, --seeds 는 축 기본값을 줄이는 용도다(스모크, 사전 점검).
  --smoke: Russia_W x, AL-3 i, Canada x, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1. gpu 는 epochs 3, 학습기 mlp, cfm, tabm.
  --precheck: Canada x, 분할 1, n {0, 40}, 추출 1, seed 1, 나머지는 본 실행과 같은 설정(r = 30, 고용량 학습기, 교차 적합 포함).

방법 정의(계획서 §6A.3 과 같다. 아래는 계획서에 없는 구현 규약)
  키 = (method, learner, alpha, placement, n, draw, seed, lam). 물리식과 B:* 는 learner 'none', seed −1, lam 0.0. 직접 계열 lam 1.0,
  V2·V2c lam 0.0, 잔차·곱셈 계열 lam ∈ {0.25, 0.5, 1.0}, RMc lam 1.0. placement: cell, rand5, randm, inblk, buf<d>.
  - V2, RM: z 또는 h 가 비유한인 행은 학습에서 뺀다(h40.v1_fit 과 같은 규약). 예측은 학습 목표의 범위로 자른다.
  - n = 0 의 공유: R1 = R0, F1n = F1a, RM = V2 의 적합에서 ĥ = ẑ − log E0(RM0 과 같은 값), R1@k = R0. 같은 적합을 다시 하지 않는다.
  - D1@ku 의 계수: c0 = 원천 최소제곱, c_n = κ = 10 수축. b_ku 는 앵커 대체 규칙(비유한 또는 0 이하 셀은 ρ·s, ρ = 원천 b/s 중앙값).
  - R1s 의 순열 seed 는 seed_of('lgx-lshuf', 대상, 모드, 분할, n, 추출)이고 전량은 n = −1 로 넣는다.
  - inblk 의 추출 d 는 fold 분할 seed_of('lgx-fold', 대상, 모드, 분할, d)를 쓴다(추출 0–2 는 S_rand5·S_randm 의 반복 0–2 와 같은 fold).
    라벨 추출 seed 는 seed_of('lgx-inblk', 대상, 모드, 분할, n, d, fold), S_randm 의 채움 seed 는 seed_of('lgx-randm', 대상, 모드, 분할, 반복, fold).
  - S_buf 는 채점 셀(eval_mask 셀)에서 d km 안의 A 셀을 뺀다.
  - 근접 단의 runs 행에서 n_lab 과 E_used 는 fold 평균이다.
  - 구간의 n = 0 행은 방법 이름 P0, R0 으로 저장한다. k > m 이면 구간이 무한대이므로 포함률 1 로 저장하고 폭 키는 저장하지 않는다(cv_flag 에 q_inf).
  - 셀 제외 변형에서 표지 셀을 지운 뒤 라벨이 0 개가 되면 원천만으로 적합한다(실제 라벨 수는 n_lab 열).
  - catboost_tuned 의 선택 기준은 λ = 1.0 의 예측 RMSE 다(직접은 y, 잔차는 학습 쪽 원천으로 구한 E 의 잔차를 학습). λ = 0.25 값은 보조 열이다.
    동률이면 반복 × 2^깊이 가 작은 설정. 30셀 이상 원천 지역이 2개 미만이면 가장 작은 설정을 고르고 표시한다.
  - N1 의 p4_ku 결측 셀: 대체 방식(B:ku_raw, P0@ku, P1@ku)과 채점 제외 방식(같은 이름에 #kuok, 비교 기준 P0#kuok, P1#kuok)을 모두 저장한다.
  - L24 의 대상 단위 집계는 등록한 n(40, 160) 모두에서 판정이 나온 대상만 확정으로 센다. 판정 불가와 행 없음은 알 수 없는 대상이다.
    어느 한 n 기준의 수는 보조 값으로 적는다.
  - L28 의 앵커 변형에서 핵심 대비는 P0, P1, R1 을 P0@b, P1@b, R1@b 로 바꾼 것이다(D0 는 그대로). CCI 입력 제외 변형은 D0 → D0c, R1 → R1@x23c.
    기준 행과 변형 행의 풀 지역 집합이 다르면 그 대비의 안정성은 '판정 불가(풀 지역 불일치)'다. 앵커 교체 판정은 n ∈ {10, 전량}의 대비로 한다.
  - 동등성 p 는 셀 가중과 블록 등가중 분포에서 각각 구한 값 가운데 큰 값이다. 4분 판정의 '동등'은 p_eq ≤ 0.025 에 대응한다.
  - 결측 대체 민감도의 두 열은 phys_train_fill 이 만든다. 공변량 결측을 원천 셀(학습 쪽) 중앙값으로 채운 뒤 physics_ensemble 을 다시 부른다.

판정 문구의 공통 규칙(계획서 §6A.2 의 개정 7. TestBook.verdict 와 build_tests_x 가 구현한다)
  - 풀 지역 수: 판정에 쓴 층화 평균의 풀 지역 수가 대상 지역 수보다 적으면 문구 앞에 '부분(지역 k/N): '을 붙인다(LG 의 L2·L4·L8 과 같다).
  - 분할 완결성: 기대 분할 수 = 중복이 아닌 유효 분할 수(h40.enumerate_units 와 같은 규칙). 판정에 쓴 대비의 풀 지역 가운데 분할이 기대보다
    적은 지역이 있으면 '부분(분할 k/K): '을 붙인다. 확인적 가설(L10, L12, L15, L29, L30)은 이 경우 '판정 불가(분할 k/K)'다.
  - 판정할 수 없는 대비(행 없음, 판정 불가): 확인적 가설은 지지 문구를 쓰지 않는다. 기각 조건이 판정한 대비만으로 이미 충족되면
    '기각(일부 대비 판정 불가, 대비 k/m)', 아니면 '판정 불가(대비 k/m; 없는 대비: …)'다. '우세가 없으면 지지' 형식의 보조 가설(L13 보조, L18,
    L28 앵커 교체)은 '부분(대비 k/m(없는 대비: …)): '을 붙인다. 그 밖의 보조 가설(L9, L14, L16, L17, L26)은 판정 불가다.
  - 세는 형식의 가설(L24 대상 4, L25 칸 6, L27 학습기 8, L31 지역 4): 알 수 없는 항목이 있으면 기준을 이미 넘겼거나 넘길 수 없을 때만
    '부분(… k/N)'으로 판정하고, 남은 항목에 따라 결과가 바뀔 수 있으면 판정 불가다.
  - L17 의 동등은 기준 λ 와 λ = 1.0 의 네 대비가 모두 동등일 때다. 우세는 기준 λ 의 대비로 판정하고 λ = 1.0 의 대비는 stat 에 병기한다.
  - L29 의 라벨 n개 조건: n ∈ {10, 40, 160, 전량}에서 재보정 기준선(B:ens, P1@b) − P1 을 계산하고, 우세인 것이 있으면 점 추정이 가장 낮은 것을
    P1* 로 두어 R1 − P1* 를 병기한다(보조 판정 행).
  - 효과 크기: 우세 또는 열세이면서 |Δ| < 0.5 cm 인 대비는 small_note 열과 판정 문구에 표기를 붙인다.
  - 표의 confirmatory 열은 가설 단위다(L10, L12, L15, L29, L30 = True). L11, L13 의 판정 행은 보조다.

누설 규약
  대상 라벨은 선택된 n개만 학습, 계수, 선택에 쓴다. 표준화와 결측 대체 통계는 학습 행렬에서만 구한다. 채점 셀 라벨은 채점 전용이다.
  물리 입력 특징(F1a, F1n)의 계수는 E0 또는 선택 라벨의 E_n 이다. 앵커 대체의 ρ 와 위약의 (a, b)는 원천에서만 구한다.
  근접 단(prox)은 설계상 채점 fold 밖의 B 셀 라벨을 학습에 쓴다(채점 fold 의 셀은 쓰지 않는다. 실행 중 단언).
  예외(라벨은 쓰지 않는다): p4_ku 와 p2_edaphic 은 load_base 가 physics._fallback 으로 공변량 결측을 전체 셀 중앙값으로 채워 계산한 열이다.
  이 통계에는 대상 지역과 채점 셀의 공변량이 들어간다. 동결 모듈이라 그대로 둔다. 자료 확인 결과 토양 열 결측은 399셀이고 그 가운데 396셀이
  레나에 있다(레나 3,037셀의 13 %). 영향을 받는 방법은 F1k, R1@f1k, D1@ku, 앵커 ku·ed 의 P0@b·P1@b·R0@b·R1@b, N1 의 B:ku_raw, B:ed_raw, B:ens 다.
  x9 축의 결측 대체 민감도 행(학습 쪽 중앙값으로 채운 열)으로 영향의 크기를 본다. 두 값의 차이는 unit.json 의 notes(phys_tr_ku, phys_tr_ed)에 남는다.

산출(<out-dir>, 기본 data/processed/lgx)
  shards/<tag>__<부분>__<대상>__<모드>__s<분할>[__<학습기>]_{runs.csv, blocksse.npz, unit.json} · _cells.npz(base, x1) · _shap.csv(x5)
    unit.json 이 완료 표지다(마지막에 원자적으로 쓴다). cfg, cfg_hash, cfg_common, code_sha(h42), code_sha_h40 을 남긴다.
    --resume 은 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  runs.csv 의 열 = h40 의 열 + variant, kappa, r, n_train_rows, imit_rms, smear, q, cv_folds, cv_flag (구간 행에는 stat 열이 더 있다).
  cells.npz: loc_id, y, s, block, lat, lon, d_src_km, keys(JSON [method, learner, alpha, placement, n, draw, seed, comp]), P(float32, 키 × 셀),
    E(키별 앵커 계수), nd_keys 와 d_lab_km((n, 추출)별 최근접 라벨 거리. lgxb 에만 있다). comp = pred(직접 예측), g(잔차 성분), h(ĥ).
    크기 추정: 분할 하나의 채점 셀 합계 약 2.6만(27 대상), 분할 5 에서 13만 셀. 키는 lgxb 170개(5성분 × seed 2 × (n, 추출) 17),
    lgx1 136개. 비압축 약 160 MB(88 + 71), 압축 뒤 130–150 MB 로 본다(재지 않았다).
  집계: <tag>_curve.csv, _tests.csv(L9–L18, L23–L31), _gate.csv, _lg_aux.csv, _region_inference.csv, _splitdist.csv, _conformal.csv,
    _distance.csv, _shap.csv, _floor.csv, _tuned_select.csv(실행 중 작성), _timing.csv, _failed.csv, _targets.csv, _meta.json, _count_<부분>.csv

명세(implementation_spec_h42)와 다르게 구현한 점
  1. 시험 파일 이름은 tests/test_h42_ext.py 다(과제 지시). 명세는 tests/test_h42_smoke.py 였다.
  2. 자료 객체는 h40.Data 를 직접 만들어 분할 범위별로 둔다(get_data_x). 분할 범위만 다른 축(n4, n4a)은 자료와 원천 색인을 공유한다.
     h40.get_data 는 분할 범위가 바뀔 때마다 자료를 다시 읽기 때문이다. 값은 같다. 분할 4·5 축은 h40.run_unit 안에서 h40.get_data 를 쓴다.
  3. 분할 4·5 축(g45)은 별도 풀이 아니라 gpu 부분의 공용 풀에서 돈다. 워커 초기화가 h40._worker_init(argv_g, gpu_queue, threads)를 부르고
     작업은 h40._worker_run 을 부른다. 학습기 등급 순서를 세 gpu 축에 함께 적용하기 위해서다.
  4. ridge 축의 방법에 D0, R1, R2 를 더했다(명세는 R0, F1k, F1n, V2, RM). 본 실행의 ridge 는 분할 1–3 뿐이라 L26 의 5분할 판정에서 빠지기
     때문이다. 가설과 판정은 바꾸지 않았다. ridge 적합이 약 3,600건 늘어난다.
  5. run_ctx_x 의 반환은 (rows, BlockStore, stats, cells) 이고 셀 제외 변형의 저장소는 BlockStore.extra 에, 변수 기여 행은 stats['shap_rows'] 에 둔다.
  6. catboost_tuned 의 선택은 주 프로세스가 아니라 워커 풀의 선택 작업(대상·모드 단위)으로 돌리고 주 프로세스가 표를 쓴다. n3 축의 작업 단위는
     선택 표가 끝난 뒤에 제출한다.
  7. 모든 축의 조각에 P0 과 그 축이 돈 (n, 추출)의 P1 을 넣는다(조각 하나로 두 기준선 대비를 계산할 수 있게 한다).
  8. 사전 점검(--precheck)의 gpu 기본 축은 r0, nn 이다. 분할 4·5 의 적합(D0·R1·R2)은 LG 사전 점검(qOkSo)이 이미 쟀다.
  9. x1 축의 cells.npz 에는 최근접 라벨 거리(nd_keys, d_lab_km)를 넣지 않는다. lgxb 의 같은 단위 파일에 있다.
  10. gpu 부분의 실행 순서는 H.LEARNER_TIER 다음에 둘째 키를 두어 FT-Transformer 를 nflow 뒤로 보낸다(계획서 §6A.3 X6 의 순서).
  11. 셀 표지 표와 연도 정합 도일 표가 없을 때 학습 없는 실행(--count-only)은 표지 0 으로 두고 진행한다. 그 밖의 실행은 --axes 에 x9 또는 x9f 가
      있으면 주 프로세스가 실행 전에 확인해 그 부분(--part) 전체를 시작하지 않는다(비용이 들기 전에 중단한다). 표와 자료 판이 어긋난 경우도 같다.
      표가 없는 상태의 적합 수에는 x9 의 tddm 앵커가 빠진다(x9 의 적합 수가 표가 있을 때보다 적다).
  12. 집계의 곡선 계산은 대상·모드 단위로 나눠 --workers 개의 프로세스에서 돈다. 값은 프로세스 수와 무관하다. 워커가 실패한 대상·모드는
      주 프로세스에서 다시 계산하고, 그래도 실패하면 meta 의 curve_failed 와 <tag>_failed.csv 에 남기고 종료 코드를 1 로 둔다.
  13. Holm 묶음의 '주 대비'는 가설 판정에 들어가는 층화 평균 대비(L9–L18, L23, L26, L28–L30)와 L24 의 대상별 이중 차분으로 정했다.
      4분 판정이 나오지 않은 행은 묶음에 넣지 않는다. 동등성 대비(L9, L12, L14, L17, L26 의 주 대비)는 같은 항목 안에서 p_eq 에 Holm 을 따로
      적용한다(holm_p_eq). 명세는 묶음의 구성원을 정하지 않았다. Holm 보정 p 는 보조 열이고 판정에 쓰지 않는다.
  14. 구간 행의 runs 에 stat 열(미포함 비율 또는 평균 폭)을 더했다.
  15. cpu 부분의 실행 순서는 확인적 등급이 먼저다(축 base, x1, x2, x9, n3 의 주 4지역 x 모드). 그다음 --axes 의 순서, h40 의 우선순위다.
      catboost_tuned 의 선택 작업도 주 4지역을 먼저 제출하고, 선택 표는 대상·모드마다 바로 쓴다(그 대상의 n3 단위가 나머지 선택 작업을 기다리지 않는다).
  16. catboost_tuned 의 선택 설정이 없는 n3 단위는 catboost, rf 만 돌고 unit.json 에 tuned_missing 을 남긴다. 이 조각은 --resume 에서 미완료다.
  17. 프로세스 풀이 깨질 때 두 번 제출되어 있던 단위는 원인 후보로 빼서 마지막에 단위마다 새 풀(프로세스 1개)에서 돌린다. 거기서도 깨지면 그
      단위만 실패다. --pool-retries 는 원인 후보를 가려내지 못한 풀 재생성의 상한이다.
  18. 결측 대체 민감도 행(x9 축)을 더했다. 계획서 §6A.3 X9 의 개정 7 에 등록했다. 가설의 판정은 바꾸지 않고 보조 판정 행(L10, L29)을 낸다.

확인하지 못한 것
  로컬에서 학습, 스모크, pytest 를 실행하지 않았다(사용자 지시 2026-09-29). 로컬 확인은 py_compile, pyflakes, 스레드 1개의 --count-only,
  그리고 학습기를 대체 함수로 바꾼 경로 확인(학습 없음, 스레드 1개, 수 초. 산출은 임시 디렉터리)뿐이다. CatBoost, RF, 신경망을 실제로 적합한 적은 없다.
  CatBoost 의 노드 종류 간 재현성, r = 30 과 고용량 학습기의 적합 시간, 셀 단위 저장의 실제 크기, nboot 10,000 집계 시간은 Rescale 사전 점검에서 잰다.
  집계 시간의 추정: 곡선 약 3만 행 × 대비 3개 × 분할 5 × 재표집 10,000회로 프로세스 1개 기준 약 100분, 워커 14개 기준 약 8분(재지 않았다).
  BASE_FIT_X 의 catboost, rf, catboost_tuned 기준 시간은 가정값이다. gpu 학습기의 기준 시간은 h40.BASE_FIT(로컬 GPU 값)이라 T4 실측과 다르다.
  --count-only 의 cpu 기본 축 전체는 스레드 1개에서 1분을 넘는다. --axes 로 나눠 한 번에 하나씩 돌린다(예: base,x1,x2,x9 / n3,prox,x5,x9f,ridge
  / n4 / n4a). 허용 표지 없이 축을 여럿 주면 안내문을 출력한다.
  개정 7 의 판정 규칙(부분 표기, 세는 형식의 판정, 분할 완결성)은 합성 저장소 시험(tests/test_h42_ext.py 의 p–s)으로만 확인한다.
  시험은 Rescale 사전 점검에서 실행한다. 결측 대체 민감도 열이 load_base 의 열과 결측이 없는 셀에서 일치하는지는 시험 (t)가, 프로세스 풀 경로
  (--workers 2, 곡선 워커 포함)는 HEAVY 시험 (E)가 확인한다. 풀이 깨졌을 때의 원인 단위 격리, 곡선 워커 실패 뒤의 재계산, 선택 설정이 없는
  n3 단위의 실행은 시험이 없다(코드 읽기로만 확인했다).

실행 환경(사용자 지시 2026-09-29)
  로컬에서는 --count-only, --write-label-flags, --write-tdd-matched, --summarize-only 만 실행한다(스레드 1).
  학습을 하는 실행(스모크, 사전 점검, 본 실행)은 --allow-local 또는 환경 변수 LG_RESCALE=1 이 없으면 자료를 읽기 전에 거부한다(h41 과 같다).
  허용 표지가 없으면 스레드는 --threads 와 무관하게 1 이고, --summarize-only 는 프로세스 1개, 재표집 1,000회 이하로 낮춘다(표의 nboot 열과
  meta 의 nboot_capped 에 적는다. 이 값은 등록한 재표집 10,000회의 값이 아니다). --gpus 의 기본값은 빈 문자열이다.

실행(ROOT)
  적합 수:   python3 scripts/3_deep_learning/h42_label_grid_ext.py --part cpu --count-only --threads 1
  선행 작업: python3 scripts/3_deep_learning/h42_label_grid_ext.py --write-label-flags --threads 1
             python3 scripts/3_deep_learning/h42_label_grid_ext.py --write-tdd-matched --threads 1
  사전 점검: LG_RESCALE=1 python3 scripts/3_deep_learning/h42_label_grid_ext.py --part cpu --precheck --workers 1 --threads 4 --gate-tag lg_precheck
  본 실행:   LG_RESCALE=1 python3 scripts/3_deep_learning/h42_label_grid_ext.py --part cpu --workers 14 --threads 4 --resume --no-summarize
             LG_RESCALE=1 python3 scripts/3_deep_learning/h42_label_grid_ext.py --part gpu --gpus 0,1,2,3 --procs-per-gpu 3 --threads 2 \\
                 --learners mlp,cfm,ddpm,tabm,nflow,ftt --resume --no-summarize
  집계:      LG_RESCALE=1 python3 scripts/3_deep_learning/h42_label_grid_ext.py --summarize-only --workers 14 --threads 4   (Rescale 작업 안에서)
             python3 scripts/3_deep_learning/h42_label_grid_ext.py --summarize-only --workers 1 --threads 1 --nboot 1000   (로컬 확인용, 값이 다르다)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from itertools import product
from pathlib import Path


def _peek_threads(default="4"):
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽는다. h40 과 같은 방식).
    허용 표지(환경 변수 LG_RESCALE=1 또는 --allow-local)가 없으면 --threads 와 무관하게 1 이다(공유 서버 보호)."""
    av = sys.argv
    if not (os.environ.get("LG_RESCALE", "") == "1" or "--allow-local" in av):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _peek_threads())

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_h40():
    """h40 을 파일 경로에서 읽어 sys.modules 에 등록한다. 모듈 최상위에서 부른다(spawn 워커가 주 모듈을 다시 읽을 때 등록되어야
    H._worker_init 의 역직렬화가 된다). 이미 등록되어 있으면 그 모듈을 쓴다(시험이 먼저 읽은 경우)."""
    name = "h40_label_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / "h40_label_grid.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_h40()

from polar.m1_core import eval_mask, half_split_blocks, load_base                                     # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                             # noqa: E402
from polar import m1_stats as MS                                                                     # noqa: E402

# ================================================================ 고정 설계값(계획서 §6A. 바꾸면 사전 등록에서 벗어난다)
KAPPA = 10.0
R_BASE = 10.0
R_GRID = (1.0, 3.0, 30.0)
PLACEBOS = ("shuffle", "const_t", "const_src", "tddlin")
KAPPA_VAR = (3.0, 30.0)
LAMS = (0.25, 0.5, 1.0)
LAM_BASE = 0.25
CV_FOLDS = 5
PROX_FOLDS = 5
PROX_REPS = 3
BUFFERS = (10, 25, 50, 100, 200, 500)
ANCHORS = ("soil", "ku", "ed", "cci", "tddm")
ANCHOR_COL = dict(soil="e5_sqrt_tdd_soil", ku="p4_ku", ed="p2_edaphic", cci="cci_alt", tddm="tdd_matched")
ALPHAS_X2 = ("10", "100")
ALPHA_N = (10, 40, 160)
INBLK_N = (10, 40, 160)
CELL_N = (0, 10, 40, 160, -1)                     # 셀 단위 저장과 거리 층을 두는 n
X9_FIT_N = (0, 3, 10, 40, 160, -1)                # x9 축의 적합 격자(해석식은 n 전 격자)
TR_FIT_N = (0, 10)                                # 결측 대체 민감도의 F1k@tr 을 적합하는 n(L10 의 확인적 대비가 쓰는 n)
TR_ANCHORS = (("ku", "p4_ku"), ("ed", "p2_edaphic"))   # 결측 대체 민감도의 해석식 기준선(앵커 이름, 열)
# physics.load_physics_inputs 가 결측을 채우는 입력 열. load_base 는 이 열의 결측을 전체 셀 중앙값으로 채워 p4_ku, p2_edaphic 을 만든다
PHYS_INPUT_COLS = ("e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_maat", "e5_twarm", "e5_tcold", "e5_swe", "sg_bdod_5_15", "sg_sand_5_15",
                   "sg_cfvo_5_15", "sg_soc_5_15")
SHAP_N = (0, 40, -1)
CONF_LEVELS = ((90, 0.10), (80, 0.20))
CONF_LAM = 0.25
Y_FLOOR_CM = 1.0                                  # 구간 점수의 ŷ 하한
MIN_REGION_CELLS = 30                             # 원천 지역 하나 제외 방식에 넣는 지역의 최소 셀 수
VARIANTS = ("noearly", "probe")
VARIANT_FLAG = dict(noearly="early", probe="gpr")
CB_THREADS = 4                                    # CPU 축의 CatBoost 스레드(본 실행과 같은 값. 재현 점검의 전제)
CB_HI = dict(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0)
RF_CFG = dict(n_estimators=500, max_features=1.0 / 3.0, min_samples_leaf=5)
TUNE_GRID = [(int(d), int(it)) for d in (3, 6, 8) for it in (200, 600, 1500)]        # (depth, iterations), 학습률 0.05
TUNE_LR = 0.05
GATE_TOL_PHYS = 1e-9
GATE_TOL_ML = 0.02
L9_POINT_MAX = 0.3
SMALL_EFFECT_CM = 0.5
SMALL_EFFECT_TXT = "통계적으로 구별되나 크기는 0.5 cm 미만"
NA_VERDICTS = ("행 없음", "판정 불가")             # 4분 판정이 나오지 않은 대비의 표시
CONFIRMATORY = ("L10", "L12", "L15", "L29", "L30")   # 이 파일이 판정하는 확인적 가설(L19 는 h41)
EQ_ALPHA = 0.025                                  # 4분 판정의 '동등'(95 % CI 가 한계 안)에 대응하는 동등성 p 의 기준
N_LEARNERS_L27 = 8                                # L27 의 등록 학습기 수(ridge, 신경망·생성 7종)
FLOOR_MIN_CELLS = 5
FLOOR_REGIONS = ("Alaska", "Lena", "Canada")

FEATS = list(H.FEATS)
MAIN4 = list(H.MAIN4)
BASE_LEARNER = H.BASE_LEARNER
X_LEARNERS = ("catboost", "rf", "catboost_tuned")
N3_LEARNERS = ("catboost", "rf", "catboost_tuned")
NAN_NATIVE = ("catboost_lo", "catboost", "catboost_tuned")
BASE_METHODS = ("P0", "P1", "P2", "D0", "D1", "R0", "R1", "R2")
X1_METHODS = ("D0t", "D0c", "D0m", "F1a", "F1k", "F1n", "V2", "V2c", "V2r", "RM", "RMc", "RM0", "R1@f1k")
RIDGE_METHODS = ("D0", "R0", "R1", "R2", "F1k", "F1n", "V2", "RM")          # 명세의 R0·F1k·F1n·V2·RM 에 D0·R1·R2 를 더했다(docstring 참조)
NN_METHODS = ("F1k", "V2", "RM")
R0_LEARNERS = ("realmlp", "ftt", "tabm", "cfm", "ddpm", "nflow")
NN_LEARNERS = ("mlp", "tabm", "ftt")
GPU_TIER = dict(mlp=0, cfm=0, ddpm=0, tabm=0, nflow=1, ftt=2, realmlp=3)     # H.LEARNER_TIER 다음의 둘째 순서 키(FT-Transformer 는 빠른 학습기 뒤)
DROP_T = ("e5_tdd", "e5_sqrt_tdd")
DROP_C = ("cci_alt", "cci_valid")
FLOOR_COLS = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe",
              "sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15", "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15",
              "sg_soc_15_30", "cci_alt", "cci_valid"]
N4_TARGETS = [(t, "x") for t in MAIN4 + [H.ALASKA]] + [(t, "i") for t in H.SUB_AL + H.SUB_OTHER]
FULL_GRID = "0,3,10,40,160,320,1000,all"
LEARNER_GRID = "0,10,40,160,all"

# 축 정의. tag = --tag 뒤에 붙는 마디, targets = all(27) · learner(12) · n4(15), grid·draws·seeds·splits = 축의 기본 범위.
AX = {
    "base": dict(part="cpu", tag="b", targets="all", splits=5, grid=FULL_GRID, draws=5, seeds=2),
    "x1": dict(part="cpu", tag="1", targets="all", splits=5, grid=FULL_GRID, draws=5, seeds=2),
    "x2": dict(part="cpu", tag="2", targets="all", splits=5, grid="0,10,40,160,all", draws=5, seeds=2),
    "x9": dict(part="cpu", tag="9", targets="all", splits=5, grid=FULL_GRID, draws=5, seeds=2),
    "n3": dict(part="cpu", tag="n3", targets="learner", splits=5, grid=LEARNER_GRID, draws=2, seeds=2),
    "prox": dict(part="cpu", tag="p", targets="all", splits=5, grid="10,40,160", draws=5, seeds=2),
    "x5": dict(part="cpu", tag="5", targets="all", splits=5, grid=LEARNER_GRID, draws=5, seeds=1),
    "x9f": dict(part="cpu", tag="9f", targets="all", splits=5, grid=LEARNER_GRID, draws=5, seeds=2),
    "n4": dict(part="cpu", tag="n4", targets="n4", splits=50, grid="0,10,40,all", draws=2, seeds=2),
    "n4a": dict(part="cpu", tag="n4a", targets="n4", splits=200, grid="3,10,40,160,all", draws=5, seeds=1),
    "ridge": dict(part="cpu", tag="r0", targets="learner", splits=5, grid=LEARNER_GRID, draws=2, seeds=2),
    "r0": dict(part="gpu", tag="r0", targets="learner", splits=5, grid=LEARNER_GRID, draws=2, seeds=2),
    "g45": dict(part="gpu", tag="g", targets="learner", splits=5, grid=FULL_GRID, draws=5, seeds=2),
    "nn": dict(part="gpu", tag="nn", targets="learner", splits=5, grid=LEARNER_GRID, draws=2, seeds=2),
}
AXES_CPU = ["base", "x1", "x2", "x9", "n3", "prox", "x5", "x9f", "n4", "n4a", "ridge"]
AXES_GPU = ["r0", "g45", "nn"]
CONFIRM_AXES = ("base", "x1", "x2", "x9", "n3")   # 확인적 가설 L10·L12(x1), L15(x2), L29(x9), L30(n3)와 그 기준 방법(base)의 축
AXIS_METHODS = {
    "base": list(BASE_METHODS), "x1": list(X1_METHODS),
    "x2": ["D1@r1", "D1@r3", "D1@r30", "R2@r1", "R2@r3", "R2@r30", "D1@shuffle", "D1@const_t", "D1@const_src", "D1@tddlin", "D1@ku",
           "R1[a10,a100]", "R2[a10,a100]", "R1s", "P1s"],
    "x9": ["P1@k3", "R1@k3", "P1@k30", "R1@k30"] + [f"{m}@{b}" for b in ANCHORS for m in ("P0", "P1", "R0", "R1")] + ["R0@x23c", "R1@x23c", "B:*"]
          + ["P0@ku_tr", "P0@ed_tr", "B:ku_raw_tr", "B:ed_raw_tr", "B:ens_tr", "F1k@tr[n0,n10]"],
    "x9f": ["P0", "P1", "D0", "R0", "R1"], "n3": ["D0", "R0", "R1", "R2[catboost]"], "n4": ["P0", "P1", "P2", "D0", "R0", "R1"],
    "n4a": ["P0", "P1", "P2"], "prox": ["P1", "D0", "R1"], "x5": ["cov·wid:P0,P1,R0,R1", "shap:D0,F1k,R1"],
    "ridge": list(RIDGE_METHODS), "r0": ["R0"], "nn": list(NN_METHODS), "g45": ["h40 gpu 부분(D0·R1·R2, MLP α 축)"],
}
AUX_MARKS = ("#", "cov", "wid")                   # 거리 층과 구간 키(곡선 표에서 뺀다)
TRACE_X = None                                    # 시험용: 리스트를 넣으면 셀 제외 변형 등 h42 고유 경로의 정보를 기록한다
# 적합 시간 추정용 기준값(--count-only 전용, 결과에는 쓰지 않는다): (기준 행 수에서 1건의 시간 s, 기준 행 수, 행 수 지수).
# catboost_lo 는 LG 사전 점검(qOkSo)의 실측 0.3 s(캐나다 x, 4스레드). catboost·rf·catboost_tuned 는 재지 않은 가정값이다.
BASE_FIT_X = dict(H.BASE_FIT)
BASE_FIT_X.update({"catboost_lo": (0.3, 16700, 0.6), "catboost": (2.2, 16700, 0.6), "catboost_tuned": (2.2, 16700, 0.6), "rf": (4.0, 16700, 1.0)})


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H42 라벨 격자 확장 하네스(LGX)")
    ap.add_argument("--part", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--axes", default="", help=f"쉼표 목록. 기본: cpu = {','.join(AXES_CPU)} / gpu = {','.join(AXES_GPU)}")
    ap.add_argument("--learners", default="", help="gpu 축의 학습기 부분집합(기본: 축마다 정해진 학습기 전부)")
    ap.add_argument("--targets", default="", help="쉼표 목록. '이름' 또는 '이름:모드'. 주면 모든 축이 이 대상만 쓴다(기본: 축별 대상)")
    ap.add_argument("--modes", default="i,x")
    ap.add_argument("--splits", type=int, default=0, help="half_split_blocks split_seed 1..K. 0 = 축별 기본값(5, n4 50, n4a 200)")
    ap.add_argument("--n-grid", default="", help="주면 축별 격자와의 교집합을 쓴다(기본: 축별 격자)")
    ap.add_argument("--draws", type=int, default=0, help="0 = 축별 기본값. 주면 축 기본값과 비교해 작은 쪽")
    ap.add_argument("--seeds", type=int, default=0, help="0 = 축별 기본값. 주면 축 기본값과 비교해 작은 쪽")
    ap.add_argument("--workers", type=int, default=2, help="CPU 프로세스 수(gpu 부분은 --gpus 가 빈 문자열일 때만 사용). 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=0, help="BLAS·torch 스레드 수. 0 = 허용 표지가 있으면 4, 없으면 1. 허용 표지가 없으면 1 로 "
                                                         "낮춘다. CPU 축의 CatBoost·RF 스레드는 4 로 고정한다")
    ap.add_argument("--gpus", default="", help="gpu 부분: 쉼표 목록. 기본은 빈 문자열(CPU). Rescale 작업 명령에서만 명시한다")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--realmlp-epochs", type=int, default=256)
    ap.add_argument("--out-dir", default="data/processed/lgx")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="본 실행(LG) 산출 디렉터리. 읽기 전용")
    ap.add_argument("--lg-tag", default="lg", help="본 실행 조각의 tag")
    ap.add_argument("--gate-dir", default="", help="재현 점검에 쓸 조각 디렉터리(기본: --lg-dir/shards)")
    ap.add_argument("--gate-tag", default="", help="재현 점검에 쓸 조각의 tag(예: lg_precheck. 기본: --lg-tag)")
    ap.add_argument("--label-flags", default="lgx_label_flags_v1.csv")
    ap.add_argument("--tdd-matched", default="lgx_tdd_matched_v1.csv")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="lgx")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="집계: 조각 사이 설정 해시가 달라도 진행한다(기본은 오류)")
    ap.add_argument("--allow-local", action="store_true", help="학습을 하는 실행(스모크 포함)을 허용한다(환경 변수 LG_RESCALE=1 과 같은 효과)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--write-label-flags", action="store_true", help="로컬 선행 작업: ABoVE 원자료에서 셀 표지 표를 만들고 끝낸다")
    ap.add_argument("--write-tdd-matched", action="store_true", help="로컬 선행 작업: 연도 정합 도일 표를 만들고 끝낸다")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(path, base=ROOT):
    return Path(path) if os.path.isabs(str(path)) else Path(base) / str(path)


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    a.SUFFIX = ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.TAG = a.tag + a.SUFFIX
    a.PERMIT = bool(a.allow_local) or os.environ.get("LG_RESCALE", "") == "1"
    a.threads_asked = int(a.threads)
    a.threads = (int(a.threads) if int(a.threads) > 0 else 4) if a.PERMIT else 1      # 허용 표지가 없으면 1(학습 없는 로컬 확인 전용)
    a.MODES = [m for m in a.modes.split(",") if m]
    allowed = AXES_CPU if a.part == "cpu" else AXES_GPU
    if a.axes:
        a.AXES = [v.strip() for v in a.axes.split(",") if v.strip()]
    elif a.precheck and a.part == "gpu":
        a.AXES = ["r0", "nn"]                                       # 분할 4·5 의 적합(D0·R1·R2)은 LG 사전 점검이 이미 쟀다
    else:
        a.AXES = list(allowed)
    bad = [v for v in a.AXES if v not in allowed]
    if bad:
        raise SystemExit(f"--part {a.part} 에서 쓸 수 없는 축: {bad} (가능: {allowed})")
    if a.smoke:                                                     # 스모크: 대상 3, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1
        a.splits = a.splits or 1; a.n_grid = a.n_grid or "0,10,40,all"; a.draws = a.draws or 1; a.seeds = a.seeds or 1
        a.targets = a.targets or "Russia_W:x,AL-3:i,Canada:x"
        if a.part == "gpu":
            a.epochs = min(a.epochs, 3); a.realmlp_epochs = min(a.realmlp_epochs, 3)
            a.learners = a.learners or "mlp,cfm,tabm"
    if a.precheck:                                                  # 사전 점검: 범위는 작게, 설정은 본 실행과 같게
        a.splits = a.splits or 1; a.n_grid = a.n_grid or "0,40"; a.draws = a.draws or 1; a.seeds = a.seeds or 1
        a.targets = a.targets or "Canada:x"
    a.LEARNERS = [v.strip() for v in a.learners.split(",") if v.strip()]
    bad = [v for v in a.LEARNERS if v not in H.GPU_LEARNERS]
    if bad:
        raise SystemExit(f"알 수 없는 gpu 학습기: {bad}")
    a.GPUS = [g.strip() for g in str(a.gpus).split(",") if g.strip() != ""]
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGDIR = _abs(a.lg_dir)
    a.SHARDS = a.OUT / "shards"
    a.GATE_DIR = _abs(a.gate_dir) if a.gate_dir else a.LGDIR / "shards"
    a.GATE_TAG = a.gate_tag or a.lg_tag
    a.FLAGS = _abs(a.label_flags, a.PROC); a.TDDM = _abs(a.tdd_matched, a.PROC)
    a.SELECT = a.OUT / f"{a.TAG}_tuned_select.csv"
    a._ha = {}
    return a


def axis_tag(a, axis):
    return f"{a.tag}{AX[axis]['tag']}{a.SUFFIX}"


def axis_learners(a, axis):
    """축의 학습기 목록. cpu 축은 조각 하나에 학습기를 함께 넣으므로 조각 단위의 학습기가 없다."""
    if axis == "r0":
        base = list(R0_LEARNERS)
    elif axis == "nn":
        base = list(NN_LEARNERS)
    elif axis == "g45":
        base = list(H.GPU_LEARNERS)
    else:
        return []
    return [v for v in base if v in a.LEARNERS] if a.LEARNERS else base


def _grid_txt(a, axis):
    base = [v.strip() for v in AX[axis]["grid"].split(",")]
    if a.n_grid:
        want = [v.strip() for v in a.n_grid.split(",") if v.strip()]
        base = [v for v in base if v in want]
    return ",".join(base)


def _targets_txt(a, axis):
    if a.targets:
        return a.targets
    kind = AX[axis]["targets"]
    if kind == "all":
        return ""
    tg = H.LEARNER_TARGETS if kind == "learner" else N4_TARGETS
    return ",".join(f"{t}:{m}" for t, m in tg if m in a.MODES)


def h40_args(a, axis):
    """축 하나의 h40 형식 인자(HA). 분할·격자·추출·대상이 축마다 다르다. cpu 축의 threads 는 4 로 고정한다."""
    if axis in a._ha:
        return a._ha[axis]
    sp = AX[axis]
    splits = int(a.splits) if a.splits else int(sp["splits"])
    draws = min(int(a.draws), sp["draws"]) if a.draws else sp["draws"]
    seeds = min(int(a.seeds), sp["seeds"]) if a.seeds else sp["seeds"]
    grid = _grid_txt(a, axis)
    tg = _targets_txt(a, axis)
    th = CB_THREADS if sp["part"] == "cpu" else int(a.threads)
    argv = ["--part", sp["part"], "--tag", axis_tag(a, axis), "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", a.subregion_map,
            "--modes", a.modes, "--splits", str(splits), "--n-grid", grid or "0", "--draws", str(draws), "--seeds", str(seeds), "--threads", str(th),
            "--epochs", str(a.epochs), "--realmlp-epochs", str(a.realmlp_epochs), "--learner-splits", str(splits), "--learner-draws", str(draws),
            "--learner-n-grid", grid or "0", "--kappa", str(KAPPA), "--r", str(R_BASE), "--lams", ",".join(str(v) for v in LAMS)]
    if tg:
        argv += ["--targets", tg, "--learner-targets", tg]
    if sp["part"] == "gpu":
        lrs = axis_learners(a, axis)
        if lrs:
            argv += ["--learners", ",".join(lrs)]
    HA = H.parse_args(argv)
    if not grid:
        HA.N_GRID = []; HA.LEARNER_N = []
    if not tg and AX[axis]["targets"] == "all":
        HA.LEARNER_TARGETS = list(HA.TARGETS)
    if sp["part"] == "gpu" and not axis_learners(a, axis):
        HA.LEARNERS = []
    HA.AXIS = axis
    a._ha[axis] = HA
    return HA


def g45_argv(a):
    """X6 분할 4·5 의 h40 인자. 기본 인자에서는 cfg_common 이 본 실행의 gpu 조각과 같다(unit_cfg 에 분할과 tag 가 없다).
    스모크와 사전 점검은 범위를 줄이므로 cfg_common 이 본 실행과 다르다(집계에서 본 실행 조각과 합치지 않는다)."""
    lrs = axis_learners(a, "g45")
    argv = ["--part", "gpu", "--splits", "5", "--learner-splits", "5", "--learners", ",".join(lrs), "--gpus", ",".join(a.GPUS),
            "--procs-per-gpu", str(a.procs_per_gpu), "--threads", str(a.threads), "--epochs", str(a.epochs),
            "--realmlp-epochs", str(a.realmlp_epochs), "--tag", axis_tag(a, "g45"), "--out-dir", str(a.OUT), "--data-dir", str(a.PROC),
            "--subregion-map", a.subregion_map, "--resume", "--no-summarize"]
    if a.smoke or a.precheck or a.targets or a.n_grid or a.draws or a.seeds:
        grid = _grid_txt(a, "g45") or "0"
        lg = ",".join(v for v in grid.split(",") if v in LEARNER_GRID.split(",")) or "0"
        argv += ["--n-grid", grid, "--learner-n-grid", lg]
        if a.draws:
            argv += ["--draws", str(a.draws), "--learner-draws", str(min(a.draws, 2))]
        if a.seeds:
            argv += ["--seeds", str(a.seeds)]
        if a.targets:
            argv += ["--targets", a.targets, "--learner-targets", a.targets]
    return argv


def g45_splits(a):
    """분할 4·5. 스모크와 사전 점검은 분할 4 하나만 돈다(경로 확인용)."""
    return (4,) if (a.smoke or a.precheck) else (4, 5)


# ================================================================ 자료
_DX: dict = {}


def get_data_x(HA):
    """h40 의 Data 를 분할 범위별로 하나씩 둔다. 분할 범위만 다른 경우에는 자료와 원천 색인(분할과 무관)을 공유하고 분할 구조만 다시 계산한다.
    h40 의 모듈 상태(H._DATA)는 건드리지 않는다."""
    sig = (str(HA.PROC), HA.k_sub, HA.buffer_km, tuple(HA.SPLITS), str(getattr(HA, "SUBMAP", "")))
    if sig in _DX:
        return _DX[sig]
    twin = next((d for s, d in _DX.items() if s[:3] == sig[:3] and s[4] == sig[4]), None)
    if twin is None:
        D = H.Data(HA)
    else:
        D = copy.copy(twin); D.args = HA; D._split = {}
    D.sig = sig
    _DX[sig] = D
    return D


def file_sha(path):
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "none"


def code_sha_x():
    return file_sha(__file__)


class ExtTables:
    """확장 열(셀 표지, 연도 정합 도일). loc_id 로 결합한다. 표에 없는 loc_id 가 있으면 중단한다.
    표 파일이 없을 때: 학습 없는 실행(dry)은 표지 0, 도일 NaN 으로 두고 경고한다. 그 밖의 실행은 주 프로세스가 실행 전에 확인해(check_ext_tables)
    그 표가 필요한 축(x9, x9f)이 --axes 에 있으면 부분 전체를 시작하지 않는다. 워커 안에서 난 같은 오류는 그 단위의 실패로 기록된다."""

    def __init__(self, a, df, dry=False):
        loc = df.loc_id.values
        self.n = len(df)
        self.flags_src, self.tdd_src = "none", "none"
        self.early = np.zeros(self.n, bool); self.gpr = np.zeros(self.n, bool); self.tddm = np.full(self.n, np.nan)
        if a.FLAGS.exists():
            f = pd.read_csv(a.FLAGS, usecols=["loc_id", "early", "gpr"]).drop_duplicates("loc_id").set_index("loc_id")
            miss = int((~np.isin(loc, f.index.values)).sum())
            if miss:
                raise SystemExit(f"[ext] 표지 표 {a.FLAGS} 에 없는 loc_id 가 {miss}개다. 자료 판과 표를 확인한다")
            self.early = f.early.reindex(loc).values.astype(float) >= 0.5
            self.gpr = f.gpr.reindex(loc).values.astype(float) >= 0.5
            self.flags_src = f"{a.FLAGS.name}:{file_sha(a.FLAGS)}"
        if a.TDDM.exists():
            t = pd.read_csv(a.TDDM, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id").set_index("loc_id")
            miss = int((~np.isin(loc, t.index.values)).sum())
            if miss:
                raise SystemExit(f"[ext] 연도 정합 도일 표 {a.TDDM} 에 없는 loc_id 가 {miss}개다. 자료 판과 표를 확인한다")
            self.tddm = t.tdd_matched.reindex(loc).values.astype(float)
            self.tdd_src = f"{a.TDDM.name}:{file_sha(a.TDDM)}"
        self.dry = bool(dry)
        self.warned: set = set()

    def require(self, axis):
        need = []
        if axis == "x9f" and self.flags_src == "none":
            need.append("셀 표지 표(--write-label-flags)")
        if axis == "x9" and self.tdd_src == "none":
            need.append("연도 정합 도일 표(--write-tdd-matched)")
        if need and not self.dry:
            raise SystemExit(f"[ext] 축 {axis} 에 필요한 입력 표가 없다: {', '.join(need)}")
        if need and axis not in self.warned:
            self.warned.add(axis)
            print(f"  [warn] 축 {axis}: {', '.join(need)} 가 없다. 적합 수는 표지 0, 도일 결측으로 센 값이다"
                  f"(x9 는 tddm 앵커가 빠지고 x9f 는 셀을 빼지 않는다)", flush=True)
        return need


_XT: dict = {}


def ext_tables(a, D, dry=False):
    k = (id(D.df), str(a.FLAGS), str(a.TDDM), bool(dry))
    if k not in _XT:
        _XT[k] = ExtTables(a, D.df, dry)
    return _XT[k]


EXT_DF_COLS = ("loc_id", "lat", "lon", "e5_tdd", "p4_ku", "p2_edaphic", "cci_alt", "e5_sqrt_tdd_soil")


def ext_of(df, XT, idx):
    out = {c_: (df[c_].values[idx] if c_ == "loc_id" else df[c_].values[idx].astype(float)) for c_ in EXT_DF_COLS}
    out.update(tdd_matched=XT.tddm[idx], early=XT.early[idx], gpr=XT.gpr[idx])
    return out


def phys_train_fill(df, src_idx, A_idx, B_idx):
    """결측 대체 민감도용 열. load_base 의 p4_ku, p2_edaphic 은 공변량 결측을 전체 셀 중앙값으로 채운 입력에서 계산한 값이다
    (physics._fallback, 대상 지역과 채점 셀의 공변량이 통계에 들어간다. 라벨은 쓰지 않는다). 여기서는 같은 식(physics_ensemble)을
    공변량 결측을 원천 셀의 중앙값으로 채운 입력에 다시 적용한다. 반환 dict(src, A, B → dict(p4_ku, p2_edaphic), info).
    원천 셀에 유한한 값이 하나도 없는 열은 채우지 못하고 physics._fallback 이 세 집합의 중앙값으로 채운다(info 의 n_unfilled 에 적는다)."""
    from polar.physics import physics_ensemble
    parts = [np.asarray(src_idx, int), np.asarray(A_idx, int), np.asarray(B_idx, int)]
    idx = np.concatenate(parts)
    ns = len(parts[0])
    cols, n_fill, n_unfilled = {}, {}, 0
    for c_ in PHYS_INPUT_COLS:
        v = df[c_].values[idx].astype(float).copy()
        bad = ~np.isfinite(v)
        if bad.any():
            ref = v[:ns]
            if np.isfinite(ref).any():
                v[bad] = float(np.nanmedian(ref))
            else:
                n_unfilled += int(bad.sum())
            n_fill[c_] = int(bad.sum())
        cols[c_] = v
    pe = physics_ensemble(pd.DataFrame(cols), E=1.0)
    ku, ed = np.asarray(pe["p4_kudryavtsev"], float), np.asarray(pe["p2_edaphic"], float)
    cut = np.cumsum([len(q) for q in parts])[:-1]
    out = {k: dict(p4_ku=a_, p2_edaphic=b_) for k, a_, b_ in zip(("src", "A", "B"), np.split(ku, cut), np.split(ed, cut))}
    out["info"] = dict(n_filled=n_fill, n_unfilled=int(n_unfilled))
    return out


def build_unit(D, HA, XT, target, mode, split):
    """작업 단위의 Ctx(h40.build_ctx)와 확장 열. 확장 열은 build_ctx 와 같은 색인으로 다시 구하고 일치를 단언한다.
    ext['phys_tr'] 은 결측 대체 민감도용 열을 계산하는 함수다(x9 축이 부를 때만 계산한다)."""
    c = H.build_ctx(D, HA, target, mode, split)
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(target, mode)
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src_ok = src_idx[ok]
    A_idx, B_idx = half_split_blocks(df, t_idx, split)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    assert len(src_ok) == len(c.y_src) and np.array_equal(df.y.values[src_ok], c.y_src), "원천 색인이 Ctx 와 다르다"
    assert len(A_idx) == len(c.yA) and np.array_equal(df.y.values[A_idx], c.yA, equal_nan=True), "A 색인이 Ctx 와 다르다"
    assert len(evB) == len(c.yB) and np.array_equal(df.y.values[evB], c.yB, equal_nan=True), "채점 색인이 Ctx 와 다르다"
    assert np.array_equal(df.block.values[A_idx], c.blkA) and np.array_equal(df.block.values[evB], c.blkB), "블록이 Ctx 와 다르다"
    ext = dict(src=ext_of(df, XT, src_ok), A=ext_of(df, XT, A_idx), B=ext_of(df, XT, evB), flags_src=XT.flags_src, tdd_src=XT.tdd_src)
    ext["phys_tr"] = lambda: phys_train_fill(df, src_ok, A_idx, evB)
    return c, ext


# ================================================================ 순수 함수(시험 대상)
def stack_rows(Xs, ys, Xl=None, yl=None, alpha=None, pseudo=None):
    """학습 행렬: 원천 행, 대상 라벨 행, 유사라벨 행 순서(h40 의 train_rows 와 같은 순서와 가중 규칙)."""
    X = [Xs]; Y = [np.asarray(ys, float)]; W = [np.ones(len(ys))]
    weighted = alpha not in (None, "1", "cont")
    if Xl is not None and len(Xl):
        X.append(Xl); Y.append(np.asarray(yl, float)); W.append(np.full(len(yl), float(alpha) if weighted else 1.0))
    if pseudo is not None:
        Xp, yp = pseudo() if callable(pseudo) else pseudo
        X.append(Xp); Y.append(np.asarray(yp, float)); W.append(np.ones(len(yp)))
    return np.vstack(X), np.concatenate(Y), (np.concatenate(W) if weighted else None)


def shrink(E_ls, E0, m, kappa):
    """κ 수축 계수. m = 라벨 수. E_ls 가 비유한이거나 m = 0 이면 E0."""
    if m <= 0 or not np.isfinite(E_ls):
        return float(E0)
    return float((m * E_ls + kappa * E0) / (m + kappa))


def ps_index(target, mode, split, seed, nA, nsrc, r):
    """유사라벨 행의 복원 추출 색인. r = 10 이면 h40 과 같은 색인이다."""
    return np.random.RandomState(seed_of("lg-ps", target, mode, split, seed)).choice(nA, int(round(float(r) * nsrc)), replace=True)


def pseudo_values(v_pool, ps, sel, y_real, resid_anchor=None):
    """유사라벨 값. v_pool = 풀 값(길이 |A|, 잔차 유사 행이면 None), ps = 복원 추출 색인, sel = 선택 라벨.
    ps 가 sel 에 속한 행은 실측(y_real) 또는 실측 잔차(y_real − resid_anchor)로 바꾼다."""
    ps = np.asarray(ps, int)
    out = np.zeros(len(ps)) if v_pool is None else np.asarray(v_pool, float)[ps].copy()
    if len(sel):
        m = np.isin(ps, sel)
        out[m] = (y_real[ps[m]] - resid_anchor[ps[m]]) if v_pool is None else y_real[ps[m]]
    return out


def placebo_pool(kind, base, y_src, target, mode, split, seed, tdd=None, tdd_ab=None, ratio=1.0):
    """위약의 풀 값(길이 |A|). base = E_n·s_A. shuffle 은 풀 셀 사이 순열, const_t 는 풀 평균, const_src 는 원천 y 평균,
    tddlin 은 (E_n/E0)·(a + b·TDD)."""
    base = np.asarray(base, float)
    if kind == "shuffle":
        pi = np.random.RandomState(seed_of("lgx-shuf", target, mode, split, seed)).permutation(len(base))
        return base[pi]
    if kind == "const_t":
        return np.full(len(base), float(np.nanmean(base)) if len(base) else np.nan)
    if kind == "const_src":
        return np.full(len(base), float(np.nanmean(y_src)))
    if kind == "tddlin":
        return float(ratio) * (tdd_ab[0] + tdd_ab[1] * np.asarray(tdd, float))
    raise ValueError(kind)


def shuffled_labels(y_sel, target, mode, split, n, draw):
    """라벨 셔플 음성 대조: 선택 라벨의 y 를 선택 셀 사이에서 순열한다."""
    y_sel = np.asarray(y_sel, float)
    pi = np.random.RandomState(seed_of("lgx-lshuf", target, mode, split, n, draw)).permutation(len(y_sel))
    return y_sel[pi]


def anchor_fill(raw_src, raw_A, raw_B, s_src, sA, sB):
    """앵커 값의 대체. 비유한 또는 0 이하 셀은 ρ·s 로 바꾼다. ρ = 원천 셀의 b/s 중앙값(대상 셀과 라벨을 쓰지 않는다)."""
    raw_src = np.asarray(raw_src, float); s_src = np.asarray(s_src, float)
    ok = np.isfinite(raw_src) & (raw_src > 0) & np.isfinite(s_src) & (s_src > 0)
    rho = float(np.median(raw_src[ok] / s_src[ok])) if ok.any() else np.nan
    out, nrep = {}, {}
    for k, v, s in (("src", raw_src, s_src), ("A", raw_A, sA), ("B", raw_B, sB)):
        v = np.asarray(v, float); bad = ~(np.isfinite(v) & (v > 0))
        out[k] = np.where(bad, rho * np.asarray(s, float), v); nrep[k] = int(bad.sum())
    return out, rho, nrep


def affine_ls(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.nanstd(x[m]) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(x[m], y[m], 1)
    return float(a), float(b)


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def nearest_km(lat, lon, lat_ref, lon_ref):
    """각 점에서 가장 가까운 기준 점까지의 대권 거리(km). 기준 점이 없으면 NaN. 단위 구의 현 길이로 최근접을 찾고 2R·arcsin(현/2)로 바꾼다
    (haversine 과 같은 값)."""
    n = len(np.atleast_1d(lat))
    if n == 0:
        return np.zeros(0)
    if len(np.atleast_1d(lat_ref)) == 0:
        return np.full(n, np.nan)
    from scipy.spatial import cKDTree
    ch, _ = cKDTree(_xyz(lat_ref, lon_ref)).query(_xyz(lat, lon), k=1)
    return 2.0 * 6371.0 * np.arcsin(np.clip(ch / 2.0, 0.0, 1.0))


def strata3(dist):
    """최근접 라벨 거리의 3분위 층. 근 = 33.3 백분위 이하, 원 = 66.7 백분위 초과, 중 = 나머지."""
    dist = np.asarray(dist, float)
    q1, q2 = np.nanpercentile(dist, [100.0 / 3.0, 200.0 / 3.0])
    near = dist <= q1; far = dist > q2
    return dict(near=near, mid=~near & ~far, far=far)


def conformal_k(m, alpha):
    """분위 색인 k = ceil((1 − α)(m + 1)). k > m 이면 구간이 무한대다."""
    return int(math.ceil((1.0 - float(alpha)) * (int(m) + 1) - 1e-12))


def conformal_q(scores, alpha):
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    k = conformal_k(len(s), alpha)
    return float(s[k - 1]) if 1 <= k <= len(s) else np.inf


def weighted_quantile(scores, weights, p):
    """가중 혼합 분포의 p 분위(가중 누적 분포가 p 이상이 되는 가장 작은 값)."""
    s = np.asarray(scores, float); w = np.asarray(weights, float)
    m = np.isfinite(s) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.inf
    o = np.argsort(s[m]); s, w = s[m][o], w[m][o]
    cw = np.cumsum(w) / w.sum()
    j = int(np.searchsorted(cw, float(p) - 1e-12, side="left"))
    return float(s[min(j, len(s) - 1)])


def log_score(y, yhat):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.log(np.asarray(y, float) / np.maximum(np.asarray(yhat, float), Y_FLOOR_CM)))


def cv_folds_of(blk_sel, target, mode, split, n, draw, k_max=CV_FOLDS):
    """교차 적합 묶음. 선택 라벨의 A 블록을 순열해 돌려 배정한다. 라벨이 한 블록에만 있으면 셀 단위 묶음으로 바꾼다.
    반환 (묶음 번호 배열, 묶음 수, 표지)."""
    blk_sel = np.asarray(blk_sel); m = len(blk_sel)
    rng = np.random.RandomState(seed_of("lgx-cv", target, mode, split, n, draw))
    ub = np.unique(blk_sel)
    if m < 2:
        return np.zeros(m, int), 0, "n<2"
    if len(ub) >= 2:
        K = min(int(k_max), len(ub))
        fo = dict(zip(ub[rng.permutation(len(ub))], np.arange(len(ub)) % K))
        return np.array([fo[b] for b in blk_sel], int), K, ""
    K = min(int(k_max), m)
    f = np.empty(m, int); f[rng.permutation(m)] = np.arange(m) % K
    return f, K, "cell_folds"


def prox_folds(target, mode, split, rep, nB, k=PROX_FOLDS):
    """채점 셀의 무작위 fold 번호."""
    pi = np.random.RandomState(seed_of("lgx-fold", target, mode, split, rep)).permutation(nB)
    f = np.empty(nB, int); f[pi] = np.arange(nB) % int(k)
    return f


def prox_labels(tier, target, mode, split, rep, k, Nk, nA, n=0, draw=0):
    """근접 단의 라벨 집합 (A 색인, B 색인). rand5 = A ∪ N_k. randm = 라벨 수 |A|(N_k 먼저, 부족분은 A 에서 무작위).
    inblk = N_k 에서 n 개 무작위."""
    Nk = np.asarray(Nk, int)
    if tier == "rand5":
        return np.arange(nA), Nk
    if tier == "randm":
        rng = np.random.RandomState(seed_of("lgx-randm", target, mode, split, rep, k))
        if len(Nk) >= nA:
            return np.zeros(0, int), np.sort(rng.choice(Nk, nA, replace=False))
        return np.sort(rng.choice(nA, nA - len(Nk), replace=False)), Nk
    if tier == "inblk":
        rng = np.random.RandomState(seed_of("lgx-inblk", target, mode, split, n, draw, k))
        return np.zeros(0, int), np.sort(rng.choice(Nk, int(n), replace=False))
    raise ValueError(tier)


def verdict4(lo, hi, lob, hib, delta):
    """4분 판정. 우세 = 두 가중의 CI 상한 < 0, 열세 = 두 가중의 CI 하한 > 0, 동등 = 그 밖에서 네 끝값의 절댓값이 모두 δ 이하, 미결정 = 그 외.
    끝값이 비유한이면 판정 불가."""
    v = [lo, hi, lob, hib]
    if not all(x is not None and np.isfinite(x) for x in v):
        return "판정 불가"
    if hi < 0 and hib < 0:
        return "우세"
    if lo > 0 and lob > 0:
        return "열세"
    if max(abs(float(x)) for x in v) <= float(delta):
        return "동등"
    return "미결정"


def eq_p(dist, delta):
    """동등성의 부트스트랩 p: max(P(Δ* ≤ −δ), P(Δ* ≥ +δ))."""
    d = np.asarray(dist, float); d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return float(max(np.mean(d <= -float(delta)), np.mean(d >= float(delta)), 1.0 / len(d)))


def base_lam_x(method):
    """판정에 쓰는 방법별 기준 λ. 물리식·기준선·V2·V2c = 0.0, 직접 계열 = 1.0, RMc = 1.0, 잔차·곱셈 계열 = 0.25."""
    m = str(method).split("#")[0]
    if m.startswith("P") or m.startswith("B:") or m in ("V2", "V2c") or m.startswith("cov") or m.startswith("wid"):
        return 0.0
    if m.startswith("D") or m.startswith("F1") or m == "RMc":
        return 1.0
    return LAM_BASE


def G(method, n, lam=None, learner=None, alpha="1", placement="cell"):
    """곡선 키 (method, learner, alpha, placement, n, lam). 물리식과 기준선의 학습기는 none, P0 은 n = 0 의 한 키다."""
    m = str(method); core = m.split("#")[0]
    if core.startswith("P") or core.startswith("B:"):
        if m == "P0":
            return H.P0_GRP
        return (m, "none", "1", placement, int(n), 0.0)
    return (m, learner or BASE_LEARNER, str(alpha), placement, int(n), float(base_lam_x(m) if lam is None else lam))


# ================================================================ 학습기
class FitterX(H.Fitter):
    """h40 의 Fitter 에 catboost(600, 깊이 6), catboost_tuned, rf 를 더한다. 나머지 학습기는 h40 의 절차 그대로다."""

    def __init__(self, args, dry=False):
        super().__init__(args, dry)
        self.models: dict = {}

    def _count(self, axis, learner, info):
        k = f"{axis}|{learner}|{(info or {}).get('method', '')}"
        self.n[axis] += 1; self.nd[k] += 1; self.last_flag = ""
        if self.nrow_fn is not None:
            nr = max(int(self.nrow_fn(axis, learner, info or {})), 1)
            self.rowsd[k] += nr
            b, ref, ex = BASE_FIT_X.get(learner, (1.0, 12000, 1.0))
            self.estd[k] += float(b * (nr / ref) ** ex)
        return k

    def _fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
        if learner not in X_LEARNERS:
            return super()._fit(learner, axis, build, seed, preds, info, init, iters, cont)
        Xtr, ytr, w = build()
        self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)
        ytr = np.asarray(ytr, float)
        if not np.all(np.isfinite(ytr)):
            raise ValueError("학습 목표에 비유한 값이 있다(앞 단계 적합 실패)")
        th = int(self.args.threads)
        if learner == "rf":
            from sklearn.ensemble import RandomForestRegressor
            med = H._prep_stats(Xtr)[0]                                  # 결측은 학습 행렬 중앙값(표준화 없음)

            def fill(X):
                X = np.asarray(X, float)
                return np.where(np.isnan(X), med, X)
            m = RandomForestRegressor(random_state=int(seed), n_jobs=th, **RF_CFG)
            m.fit(fill(Xtr), ytr, sample_weight=w)
            return m, [np.asarray(m.predict(fill(p)), float) if len(p) else np.zeros(0) for p in preds]
        cfg = dict(CB_HI) if learner == "catboost" else dict((info or {}).get("cb_cfg") or {})
        if not cfg:
            raise RuntimeError("catboost_tuned 의 선택 설정이 없다(선택 표를 먼저 만든다)")
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=int(cfg["iterations"]), learning_rate=float(cfg["learning_rate"]), depth=int(cfg["depth"]),
                              l2_leaf_reg=float(cfg.get("l2_leaf_reg", 3.0)), random_seed=int(seed), verbose=0, allow_writing_files=False,
                              thread_count=th)
        m.fit(Xtr, ytr, sample_weight=w)
        return m, [np.asarray(m.predict(p, thread_count=th), float) if len(p) else np.zeros(0) for p in preds]


class CellBook:
    """채점 셀 단위 예측 성분의 저장(float32). 직접 계열은 예측, 잔차 계열은 g, 곱셈 보정은 ĥ 이다.
    키 = [method, learner, alpha, placement, n, draw, seed, comp], comp ∈ {pred, g, h}. E 는 그 키의 앵커 계수(없으면 NaN)."""

    def __init__(self, c, ext, d_src):
        self.static = dict(loc_id=np.asarray(ext["B"]["loc_id"]), y=np.asarray(c.yB, np.float32), s=np.asarray(c.sB, np.float32),
                           block=np.asarray(c.blkB).astype(str), lat=np.asarray(ext["B"]["lat"], np.float32),
                           lon=np.asarray(ext["B"]["lon"], np.float32), d_src_km=np.asarray(d_src, np.float32))
        self.nB = len(c.yB)
        self.keys, self.P, self.E = [], [], []
        self.nd, self.D = [], []

    def add(self, method, learner, alpha, placement, n, d, seed, comp, vec, E=np.nan):
        vec = np.asarray(vec, np.float32)
        if vec.shape != (self.nB,):
            return
        self.keys.append([str(method), str(learner), str(alpha), str(placement), int(n), int(d), int(seed), str(comp)])
        self.P.append(vec); self.E.append(float(E))

    def add_nd(self, n, d, dist):
        self.nd.append([int(n), int(d)])
        self.D.append(np.full(self.nB, np.nan, np.float32) if dist is None else np.asarray(dist, np.float32))

    def save(self, path):
        path = Path(path)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
        P = np.vstack(self.P).astype(np.float32) if self.P else np.zeros((0, self.nB), np.float32)
        Dm = np.vstack(self.D).astype(np.float32) if self.D else np.zeros((0, self.nB), np.float32)
        np.savez_compressed(tmp, **self.static, keys=np.array([json.dumps(k, ensure_ascii=False) for k in self.keys], dtype=str), P=P,
                            E=np.asarray(self.E, float), nd_keys=np.array([json.dumps(k) for k in self.nd], dtype=str), d_lab_km=Dm,
                            meta=np.array(json.dumps(dict(format="lgx_cells_v1", n_cells=int(self.nB), n_keys=len(self.keys)))))
        os.replace(tmp, path)
        return path


# ================================================================ 작업 단위 실행
class Runner:
    """작업 단위(대상, 모드, 분할) 하나에서 축 하나를 실행한다. 키 = (method, learner, alpha, placement, n, draw, seed, lam)."""

    def __init__(self, c, ext, axis, HA, dry=False, tuned=None):
        self.c, self.ext, self.axis, self.HA, self.dry = c, ext, axis, HA, bool(dry)
        self.part = AX[axis]["part"]
        self.tuned = dict(tuned or {})
        self.F = FitterX(HA, dry)
        self.F.nrow_fn = lambda ax, lr, info: int(info.get("nrow", 1))
        self.name = f"{c.target}|{c.mode}"
        self.st = BlockStore(self.name, c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part=self.part, axis=axis))
        self.extra: list = []
        self.rows: list = []; self.shap: list = []
        self.n_rows = 0; self.n_bad = 0
        self.nA, self.nB, self.nsrc = len(c.yA), len(c.yB), len(c.y_src)
        self.lams = [float(v) for v in HA.LAMS]
        self.kappa, self.r = float(HA.kappa), float(HA.r)
        self.seeds = [int(v) for v in HA.SEEDS]
        self.grid = [int(v) for v in HA.N_GRID]
        self.draws = int(HA.DRAWS)
        self.notes: dict = {}
        self._ps: dict = {}
        self.keep_cols = dict(drop_t=[i for i, f in enumerate(FEATS) if f not in DROP_T],
                              drop_c=[i for i, f in enumerate(FEATS) if f not in DROP_C],
                              drop_m=[i for i, f in enumerate(FEATS) if f not in DROP_T + DROP_C])
        self.base_row = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=self.part, axis=axis)
        self.cells = None
        if axis in ("base", "x1") and not self.dry:
            d_src = nearest_km(ext["B"]["lat"], ext["B"]["lon"], ext["src"]["lat"], ext["src"]["lon"])
            self.cells = CellBook(c, ext, d_src)
        self.add("P0", "none", "1", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)          # P0 은 모든 비교의 기준이므로 항상 저장한다

    # ------------------------------------------------------------ 저장
    def add(self, method, learner, alpha, placement, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", st=None, y=None, **kw):
        self.n_rows += 1
        if self.dry:
            return None
        st = self.st if st is None else st
        y = self.c.yB if y is None else y
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = dict(self.base_row, variant=str(kw.get("variant", "")), method=method, learner=learner, alpha=str(alpha), placement=placement, n=int(n),
                   n_lab=int(n_lab), draw=int(d), seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan,
                   E_used=float(E_used), alpha_sel="", n_blocks_lab=int(nb_lab), n_nonfinite=nf,
                   fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"), kappa=float(kw.get("kappa", self.kappa)),
                   r=float(kw.get("r", np.nan)), n_train_rows=int(kw.get("nrow", 0)), imit_rms=float(kw.get("imit", np.nan)),
                   smear=float(kw.get("smear", np.nan)), q=float(kw.get("q", np.nan)), cv_folds=int(kw.get("cv_folds", 0)),
                   cv_flag=str(kw.get("cv_flag", "")))
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다(h40 과 같다)
            self.n_bad += 1
            self.rows.append(row)
            return None
        key = (method, learner, str(alpha), placement, int(n), int(d), int(seed), float(lam))
        st.add(key, y, pred)
        sse, cnt = st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
            rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
        row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.nanmean(pred - y)) if len(y) else np.nan)
        self.rows.append(row)
        return key

    def add_masked(self, method, learner, alpha, placement, n, d, seed, lam, pred, masks, st=None, y=None):
        """층 키: 층 밖 셀을 NaN 으로 둔 예측을 저장소에 직접 넣는다(block_sse 가 비유한 셀을 뺀다). 예측이 모두 유한한 키만 넣는다."""
        if self.dry or not masks:
            return
        st = self.st if st is None else st
        y = self.c.yB if y is None else y
        pred = np.asarray(pred, float)
        if not np.all(np.isfinite(pred)):
            return
        for nm, mk in masks.items():
            st.add((f"{method}#{nm}", learner, str(alpha), placement, int(n), int(d), int(seed), float(lam)), y, np.where(mk, pred, np.nan))

    def add_stat(self, name, learner, n, d, seed, lam, num, den=None, st=None, **kw):
        """구간 키: 블록별 합(미포함 셀 수 또는 폭 합)과 블록별 셀 수를 저장한다."""
        self.n_rows += 1
        if self.dry:
            return
        st = self.st if st is None else st
        num = np.asarray(num, float)
        sse = np.bincount(st.codes, weights=num, minlength=st.nb).astype(float)
        cnt = st.ncell.copy() if den is None else np.asarray(den, np.int64)
        st.add_sse((name, learner, "1", "cell", int(n), int(d), int(seed), float(lam)), sse, cnt)
        tot = float(cnt.sum())
        self.rows.append(dict(self.base_row, variant="", method=name, learner=learner, alpha="1", placement="cell", n=int(n),
                              n_lab=int(kw.get("n_lab", 0)), draw=int(d), seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan,
                              bias_cm=np.nan, E_used=float(kw.get("E_used", np.nan)), alpha_sel="", n_blocks_lab=int(kw.get("nb_lab", 0)),
                              n_nonfinite=0, fit_flag="", kappa=self.kappa, r=np.nan, n_train_rows=0, imit_rms=np.nan, smear=np.nan,
                              q=float(kw.get("q", np.nan)), cv_folds=int(kw.get("cv_folds", 0)), cv_flag=str(kw.get("cv_flag", "")),
                              stat=float(sse.sum() / tot) if tot > 0 else np.nan))

    def emit_resid(self, method, learner, alpha, placement, n, d, seed, anchorB, g, E_used, n_lab, nb_lab=0, flag="", st=None, y=None,
                   masks=None, cell=False, **kw):
        for lam in self.lams:
            p = np.asarray(anchorB, float) + lam * np.asarray(g, float)
            self.add(method, learner, alpha, placement, n, d, seed, lam, p, E_used, n_lab, nb_lab, flag, st=st, y=y, **kw)
            self.add_masked(method, learner, alpha, placement, n, d, seed, lam, p, masks, st=st, y=y)
        if cell and self.cells is not None and int(n) in CELL_N:
            self.cells.add(method, learner, alpha, placement, n, d, seed, "g", g, E_used)

    # ------------------------------------------------------------ 계수·색인·입력
    def coefs(self, sel, cx=None, kappa=None):
        """(E_n, E_ls). 라벨이 없거나 E_ls 가 비유한이면 둘 다 E0(h40 과 같은 규약)."""
        cx = self.c if cx is None else cx
        if len(sel) == 0:
            return float(cx.E0), float(cx.E0)
        E_ls = H.ls_E(cx.yA[sel], cx.sA[sel])
        if not np.isfinite(E_ls):
            return float(cx.E0), float(cx.E0)
        return shrink(E_ls, cx.E0, len(sel), self.kappa if kappa is None else kappa), float(E_ls)

    def n_ps(self, r):
        return int(round(float(r) * self.nsrc))

    def ps_of(self, seed, r):
        k = (int(seed), float(r))
        if k not in self._ps:
            self._ps[k] = ps_index(self.c.target, self.c.mode, self.c.split, seed, self.nA, self.nsrc, r)
        return self._ps[k]

    def cells_iter(self, grid=None, draws=None):
        return H.cells_of(self.grid if grid is None else grid, self.draws if draws is None else draws, self.nA)

    def sel_of(self, n, d):
        return H.draw_cells(self.c.target, self.c.mode, self.c.split, n, d, self.nA)

    def phys_tr(self):
        """결측 대체 민감도용 열(공변량 결측을 원천 셀 중앙값으로 채워 다시 계산한 p4_ku, p2_edaphic). 없거나 계산에 실패하면 None."""
        if not hasattr(self, "_phys_tr"):
            pt = self.ext.get("phys_tr")
            try:
                self._phys_tr = pt() if callable(pt) else pt
            except Exception as e:                                        # noqa: BLE001
                self.notes["phys_tr_error"] = repr(e)[:160]
                self._phys_tr = None
        return self._phys_tr

    def inputs(self, kind, sel, E1=None, cx=None):
        """(원천 행렬, 라벨 행렬, 채점 행렬). kind: x25, drop_t, drop_c, drop_m, f1a, f1k, f1k_tr, f1n. 학습 없는 실행은 길이만 맞는 행렬을 돌려준다."""
        c = self.c if cx is None else cx
        if self.dry or kind == "x25":
            return c.X_src, c.XA[sel], c.XB
        if kind in self.keep_cols:
            k = self.keep_cols[kind]
            return c.X_src[:, k], c.XA[sel][:, k], c.XB[:, k]
        assert c is self.c, "물리 입력 열은 기본 Ctx 에서만 만든다"
        e = self.ext
        if kind in ("f1k", "f1k_tr"):
            q = e if kind == "f1k" else self.phys_tr()
            cs = np.c_[q["src"]["p4_ku"], q["src"]["p2_edaphic"]]
            cl = np.c_[np.asarray(q["A"]["p4_ku"])[sel], np.asarray(q["A"]["p2_edaphic"])[sel]]
            cB = np.c_[q["B"]["p4_ku"], q["B"]["p2_edaphic"]]
        elif kind == "f1a":
            cs, cl, cB = c.E0 * c.s_src, c.E0 * c.sA[sel], c.E0 * c.sB
        elif kind == "f1n":
            cs, cl, cB = c.E0 * c.s_src, float(E1) * c.sA[sel], float(E1) * c.sB
        else:
            raise ValueError(kind)
        return (np.column_stack([c.X_src, cs]).astype(np.float32), np.column_stack([c.XA[sel], cl]).astype(np.float32),
                np.column_stack([c.XB, cB]).astype(np.float32))

    def tuned_cfg(self, method):
        kind = "D" if str(method).startswith("D") else "R"
        cfg = self.tuned.get(kind)
        if cfg is None:
            return dict(iterations=600, depth=6, learning_rate=TUNE_LR, l2_leaf_reg=3.0) if self.dry else {}
        return dict(cfg)

    def fit(self, lr, method, build, seed, preds, n, d, sel, nrow, alpha="1", placement="cell", **info):
        inf = dict(method=method, n=int(n), draw=int(d), alpha=str(alpha), placement=placement, sel=np.asarray(sel, int), nrow=int(nrow), **info)
        if lr == "catboost_tuned":
            inf["cb_cfg"] = self.tuned_cfg(method)
        return self.F.fit(lr, self.axis, build, seed, list(preds), info=inf)

    # ------------------------------------------------------------ 직접·잔차 적합(공용)
    def direct(self, lr, method, n, d, sel, seed, kind="x25", E1=None, cx=None, pseudo=None, nps=0, alpha=None, st=None, variant="",
               masks=None, cell=False, r=np.nan, imit_ref=None, E_used=np.nan, placement="cell", nb=0):
        c = self.c if cx is None else cx
        Xs, Xl, XB = self.inputs(kind, sel, E1, c)
        nrow = len(c.y_src) + len(sel) + int(nps)
        a = str(alpha) if alpha else "1"
        _, (p,) = self.fit(lr, method, lambda: stack_rows(Xs, c.y_src, Xl, c.yA[sel], alpha, pseudo), seed, [XB], n, d, sel, nrow, alpha=a,
                           placement=placement, variant=variant)
        imit = float(np.sqrt(np.mean((np.asarray(p, float) - imit_ref) ** 2))) if (imit_ref is not None and not self.dry and len(p)) else np.nan
        self.add(method, lr, a, placement, n, d, seed, 1.0, p, E_used, len(sel), nb, self.F.last_flag, st=st, y=c.yB, variant=variant, nrow=nrow,
                 r=r, imit=imit)
        self.add_masked(method, lr, a, placement, n, d, seed, 1.0, p, masks, st=st, y=c.yB)
        if cell and self.cells is not None and int(n) in CELL_N:
            self.cells.add(method, lr, a, placement, n, d, seed, "pred", p, E_used)
        return p

    def resid(self, lr, method, n, d, sel, seed, anchor_sel, anchorB, E_used, kind="x25", E1=None, cx=None, pseudo=None, nps=0, alpha=None,
              st=None, variant="", masks=None, cell=False, r=np.nan, kappa=None, y_sel=None, placement="cell", nb=0, emit=True):
        c = self.c if cx is None else cx
        Xs, Xl, XB = self.inputs(kind, sel, E1, c)
        yl = (c.yA[sel] if y_sel is None else np.asarray(y_sel, float)) - np.asarray(anchor_sel, float)
        nrow = len(c.y_src) + len(sel) + int(nps)
        a = str(alpha) if alpha else "1"
        _, (g,) = self.fit(lr, method, lambda: stack_rows(Xs, c.r0_src, Xl, yl, alpha, pseudo), seed, [XB], n, d, sel, nrow, alpha=a,
                           placement=placement, variant=variant)
        if emit:
            self.emit_resid(method, lr, a, placement, n, d, seed, anchorB, g, E_used, len(sel), nb, self.F.last_flag, st=st, y=c.yB, masks=masks,
                            cell=cell, variant=variant, nrow=nrow, r=r, kappa=self.kappa if kappa is None else kappa)
        return g

    def base_cell(self, lr, n, d, sel, E1, seed, methods, masks=None, cell=False, cx=None, st=None, variant="", suffix="", kind="x25",
                  kappa=None):
        """기준 방법 D0, D1, R0, R1, R2(h40 의 cb_cell·learner_cell 과 같은 학습 행렬). suffix 는 방법 문자열의 변형 표기다."""
        c = self.c if cx is None else cx
        E0 = c.E0; nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
        a0, a1 = E0 * c.sA[sel], E1 * c.sA[sel]
        kw = dict(cx=c, st=st, variant=variant, masks=masks, cell=cell, kind=kind, nb=nb)
        if "D0" in methods:
            self.direct(lr, "D0" + suffix, n, d, sel, seed, **kw)
        if "D1" in methods:
            self.direct(lr, "D1" + suffix, n, d, sel, seed, nps=self.n_ps(self.r), r=self.r, imit_ref=E1 * c.sB, E_used=E1,
                        pseudo=lambda: (c.XA[self.ps_of(seed, self.r)], pseudo_values(E1 * c.sA, self.ps_of(seed, self.r), sel, c.yA)), **kw)
        g0 = None
        if "R0" in methods or ("R1" in methods and nl == 0):
            g0 = self.resid(lr, "R0" + suffix, n, d, sel, seed, a0, E0 * c.sB, E0, emit="R0" in methods, **kw)
        if "R1" in methods:
            if nl == 0:                                                   # n = 0 에서 R1 은 R0 과 같다(E_n = E0, 원천 잔차만)
                self.emit_resid("R1" + suffix, lr, "1", "cell", n, d, seed, E1 * c.sB, g0, E1, 0, 0, self.F.last_flag, st=st, y=c.yB, masks=masks,
                                cell=cell, variant=variant, nrow=len(c.y_src), kappa=self.kappa if kappa is None else kappa)
            else:
                self.resid(lr, "R1" + suffix, n, d, sel, seed, a1, E1 * c.sB, E1, kappa=kappa, **kw)
        if "R2" in methods:
            self.resid(lr, "R2" + suffix, n, d, sel, seed, a1, E1 * c.sB, E1, nps=self.n_ps(self.r), r=self.r,
                       pseudo=lambda: (c.XA[self.ps_of(seed, self.r)], pseudo_values(None, self.ps_of(seed, self.r), sel, c.yA, E1 * c.sA)), **kw)

    def analytic(self, n, d, sel, E1, E2, methods=("P1", "P2"), masks=None, cx=None, st=None, variant=""):
        c = self.c if cx is None else cx
        nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
        for m, E in (("P1", E1), ("P2", E2)):
            if m in methods:
                self.add(m, "none", "1", "cell", n, d, -1, 0.0, E * c.sB, E, nl, nb, st=st, y=c.yB, variant=variant)
                self.add_masked(m, "none", "1", "cell", n, d, -1, 0.0, E * c.sB, masks, st=st, y=c.yB)
        if masks:
            self.add_masked("P0", "none", "1", "cell", n, d, -1, 0.0, c.E0 * c.sB, masks, st=st, y=c.yB)

    def masks_of(self, n, d, sel):
        """거리 3분위 층과 최근접 라벨 거리. n = 0 이거나 저장 대상 n 이 아니면 없다."""
        if self.dry or len(sel) == 0 or int(n) not in CELL_N:
            return None, None
        e = self.ext
        dist = nearest_km(e["B"]["lat"], e["B"]["lon"], e["A"]["lat"][sel], e["A"]["lon"][sel])
        if not np.all(np.isfinite(dist)):
            return None, dist
        return strata3(dist), dist

    # ------------------------------------------------------------ 축 base · n4 · n4a
    def run_base(self, methods=BASE_METHODS, lr=BASE_LEARNER, strata=True, cell=True):
        for n, d in self.cells_iter():
            sel = self.sel_of(n, d)
            E1, E2 = self.coefs(sel)
            masks, dist = self.masks_of(n, d, sel) if strata else (None, None)
            if self.cells is not None and n in CELL_N:
                self.cells.add_nd(n, d, dist)
            self.analytic(n, d, sel, E1, E2, methods, masks)
            for seed in self.seeds:
                self.base_cell(lr, n, d, sel, E1, seed, [m for m in methods if not m.startswith("P")], masks=masks, cell=cell)

    # ------------------------------------------------------------ 축 x1(결합 방식)과 학습기 축
    def x1_cell(self, lr, n, d, sel, E1, seed, methods, cell=False):
        c = self.c; nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
        pF1a, flag_f1a = None, ""
        for m, kind in (("D0t", "drop_t"), ("D0c", "drop_c"), ("D0m", "drop_m"), ("F1a", "f1a"), ("F1k", "f1k")):
            if m in methods:
                p = self.direct(lr, m, n, d, sel, seed, kind=kind, E1=E1, cell=cell and m == "F1k", nb=nb, E_used=c.E0 if m == "F1a" else np.nan)
                if m == "F1a":
                    pF1a, flag_f1a = p, self.F.last_flag                  # F1a 적합 직후의 표지(뒤따르는 F1k 적합이 last_flag 를 바꾼다)
        if "F1n" in methods:
            if nl == 0 and pF1a is not None:                              # n = 0 에서 F1n 은 F1a 와 같다(같은 적합을 다시 하지 않는다)
                self.add("F1n", lr, "1", "cell", n, d, seed, 1.0, pF1a, E1, 0, 0, flag_f1a, nrow=self.nsrc)
                if cell and self.cells is not None and n in CELL_N:
                    self.cells.add("F1n", lr, "1", "cell", n, d, seed, "pred", pF1a, E1)
            else:
                self.direct(lr, "F1n", n, d, sel, seed, kind="f1n", E1=E1, cell=cell, nb=nb, E_used=E1)
        if any(m in methods for m in ("V2", "V2c", "V2r", "RM", "RMc", "RM0")):
            self.v2_block(lr, n, d, sel, E1, seed, methods, cell, nl, nb)
        if "R1@f1k" in methods:
            self.resid(lr, "R1@f1k", n, d, sel, seed, E1 * c.sA[sel], E1 * c.sB, E1, kind="f1k", nb=nb)

    def v2_block(self, lr, n, d, sel, E1, seed, methods, cell, nl, nb):
        """계수 예측 V2(z = log(y/s)), 재변환 보정 V2c, V2 앵커 잔차 V2r, 곱셈 보정 RM·RMc·RM0. 비유한 z 의 행은 학습에서 뺀다."""
        c = self.c; E0 = c.E0
        with np.errstate(invalid="ignore", divide="ignore"):
            zs = np.log(c.y_src) - np.log(c.s_src)
            lE0, lE1 = np.log(E0), np.log(E1)
        zl = c.zA[sel]
        z = np.concatenate([zs, zl]); ok = np.isfinite(z)
        s_all = np.concatenate([c.s_src, c.sA[sel]]); y_all = np.concatenate([c.y_src, c.yA[sel]])
        rm_own = nl > 0 and ("RM" in methods or "RMc" in methods)
        need_v2 = any(m in methods for m in ("V2", "V2c", "V2r", "RM0")) or (nl == 0 and ("RM" in methods or "RMc" in methods))
        need_in = any(m in methods for m in ("V2c", "V2r")) or (nl == 0 and "RMc" in methods)

        def clip_fit(name, tgt, okm, insample):
            preds = [c.XB] + ([c.X_src, c.XA[sel]] if insample else [])
            _, out = self.fit(lr, name, lambda: (np.vstack([c.X_src, c.XA[sel]])[okm], tgt[okm], None), seed, preds, n, d, sel, int(okm.sum()))
            lo, hi = (float(tgt[okm].min()), float(tgt[okm].max())) if okm.any() else (0.0, 0.0)
            tB = np.clip(out[0], lo, hi)
            t_in = np.clip(np.concatenate([out[1], out[2]]), lo, hi) if insample else None
            with np.errstate(invalid="ignore", over="ignore"):
                sm = float(np.mean(np.exp(tgt[okm] - t_in[okm]))) if (insample and okm.any()) else np.nan
            return tB, t_in, sm, self.F.last_flag

        zB = None
        if need_v2:
            zB, z_in, sm, fl = clip_fit("V2", z, ok, need_in)
            with np.errstate(invalid="ignore", over="ignore"):
                vB = np.exp(zB) * c.sB
            nrow = int(ok.sum())
            if "V2" in methods:
                self.add("V2", lr, "1", "cell", n, d, seed, 0.0, vB, np.nan, nl, nb, fl, nrow=nrow)
                if cell and self.cells is not None and n in CELL_N:
                    self.cells.add("V2", lr, "1", "cell", n, d, seed, "pred", vB, np.nan)
            if "V2c" in methods:
                self.add("V2c", lr, "1", "cell", n, d, seed, 0.0, vB * sm, np.nan, nl, nb, fl, nrow=nrow, smear=sm)
            if "RM0" in methods:
                for lam in self.lams:
                    with np.errstate(invalid="ignore", over="ignore"):
                        p = E0 * c.sB * np.exp(lam * (zB - lE0))
                    self.add("RM0", lr, "1", "cell", n, d, seed, lam, p, E0, nl, nb, fl, nrow=nrow)
            if "V2r" in methods:
                with np.errstate(invalid="ignore", over="ignore"):
                    rt = y_all - np.exp(z_in) * s_all
                okr = np.isfinite(rt)
                _, (g,) = self.fit(lr, "V2r", lambda: (np.vstack([c.X_src, c.XA[sel]])[okr], rt[okr], None), seed, [c.XB], n, d, sel,
                                   int(okr.sum()))
                self.emit_resid("V2r", lr, "1", "cell", n, d, seed, vB, g, np.nan, nl, nb, self.F.last_flag, nrow=int(okr.sum()))
        if "RM" in methods or "RMc" in methods:
            if rm_own:
                h = np.concatenate([zs - lE0, zl - lE1]); okh = np.isfinite(h)
                hB, _, smh, fl = clip_fit("RM", h, okh, "RMc" in methods)
                nrow = int(okh.sum())
            else:                                                        # n = 0: V2 의 적합에서 ĥ = ẑ − log E0(추가 적합 없음)
                hB, smh, nrow = zB - lE0, sm, int(ok.sum())
            if "RM" in methods:
                for lam in self.lams:
                    with np.errstate(invalid="ignore", over="ignore"):
                        p = E1 * c.sB * np.exp(lam * hB)
                    self.add("RM", lr, "1", "cell", n, d, seed, lam, p, E1, nl, nb, fl, nrow=nrow)
                if cell and self.cells is not None and n in CELL_N:
                    self.cells.add("RM", lr, "1", "cell", n, d, seed, "h", hB, E1)
            if "RMc" in methods:
                with np.errstate(invalid="ignore", over="ignore"):
                    p = E1 * c.sB * np.exp(hB) * smh
                self.add("RMc", lr, "1", "cell", n, d, seed, 1.0, p, E1, nl, nb, fl, nrow=nrow, smear=smh)

    def run_x1(self):
        for n, d in self.cells_iter():
            sel = self.sel_of(n, d)
            E1, E2 = self.coefs(sel)
            self.analytic(n, d, sel, E1, E2, ("P1",))
            for seed in self.seeds:
                self.x1_cell(BASE_LEARNER, n, d, sel, E1, seed, X1_METHODS, cell=True)

    def run_learner(self, lr, methods):
        """학습기 하나로 기준 방법과 결합 방식 방법을 적합한다(ridge 축, gpu 의 r0·nn 축, n3 축)."""
        bm = [m for m in methods if m in BASE_METHODS and not m.startswith("P")]
        xm = [m for m in methods if m in X1_METHODS]
        for n, d in self.cells_iter():
            sel = self.sel_of(n, d)
            E1, E2 = self.coefs(sel)
            self.analytic(n, d, sel, E1, E2, ("P1",))
            for seed in self.seeds:
                if bm:
                    self.base_cell(lr, n, d, sel, E1, seed, bm)
                if xm:
                    self.x1_cell(lr, n, d, sel, E1, seed, xm)

    def run_n3(self, learners=N3_LEARNERS):
        for lr in learners:
            self.run_learner(lr, ("D0", "R0", "R1", "R2") if lr == "catboost" else ("D0", "R0", "R1"))

    # ------------------------------------------------------------ 축 x2(유사라벨)
    def run_x2(self):
        c = self.c; lr = BASE_LEARNER; E0 = c.E0; eA = self.ext["A"]
        tdd_ab = affine_ls(self.ext["src"]["e5_tdd"], c.y_src)
        bku, rho, _ = anchor_fill(self.ext["src"]["p4_ku"], eA["p4_ku"], self.ext["B"]["p4_ku"], c.s_src, c.sA, c.sB)
        c0_ku = H.ls_E(c.y_src, bku["src"])
        self.notes.update(tddlin_a=tdd_ab[0], tddlin_b=tdd_ab[1], ku_rho=rho, ku_c0=c0_ku)
        for n, d in self.cells_iter():
            sel = self.sel_of(n, d)
            E1, E2 = self.coefs(sel)
            nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
            self.analytic(n, d, sel, E1, E2, ("P1",))
            y_sh, E1s = None, E0
            if n != 0 and nl >= 2:                                        # 라벨 셔플 음성 대조(R1s, P1s)
                y_sh = shuffled_labels(c.yA[sel], c.target, c.mode, c.split, n, d)
                E1s = shrink(H.ls_E(y_sh, c.sA[sel]), E0, nl, self.kappa)
                self.add("P1s", "none", "1", "cell", n, d, -1, 0.0, E1s * c.sB, E1s, nl, nb)
            c_ls = H.ls_E(c.yA[sel], bku["A"][sel]) if nl else np.nan
            c_n = shrink(c_ls, c0_ku, nl, self.kappa)
            a1 = E1 * c.sA[sel]; base = E1 * c.sA
            for seed in self.seeds:
                if y_sh is not None:
                    self.resid(lr, "R1s", n, d, sel, seed, E1s * c.sA[sel], E1s * c.sB, E1s, y_sel=y_sh, nb=nb)
                if n == -1:                                               # 전량은 R1s, P1s 만
                    continue
                for r in R_GRID:
                    tg = f"@r{int(r)}"
                    self.direct(lr, "D1" + tg, n, d, sel, seed, nps=self.n_ps(r), r=r, imit_ref=E1 * c.sB, E_used=E1, nb=nb,
                                pseudo=lambda r=r: (c.XA[self.ps_of(seed, r)], pseudo_values(base, self.ps_of(seed, r), sel, c.yA)))
                    self.resid(lr, "R2" + tg, n, d, sel, seed, a1, E1 * c.sB, E1, nps=self.n_ps(r), r=r, nb=nb,
                               pseudo=lambda r=r: (c.XA[self.ps_of(seed, r)], pseudo_values(None, self.ps_of(seed, r), sel, c.yA, base)))
                for kind in PLACEBOS + ("ku",):
                    def pool(kind=kind):
                        if kind == "ku":
                            return c_n * bku["A"]
                        return placebo_pool(kind, base, c.y_src, c.target, c.mode, c.split, seed, tdd=eA["e5_tdd"], tdd_ab=tdd_ab, ratio=E1 / E0)
                    self.direct(lr, f"D1@{kind}", n, d, sel, seed, nps=self.n_ps(self.r), r=self.r, imit_ref=E1 * c.sB,
                                E_used=c_n if kind == "ku" else E1, nb=nb,
                                pseudo=lambda pool=pool: (c.XA[self.ps_of(seed, self.r)], pseudo_values(pool(), self.ps_of(seed, self.r), sel, c.yA)))
                if n in ALPHA_N and nl > 0:
                    for al in ALPHAS_X2:
                        self.resid(lr, "R1", n, d, sel, seed, a1, E1 * c.sB, E1, alpha=al, nb=nb)
                        self.resid(lr, "R2", n, d, sel, seed, a1, E1 * c.sB, E1, alpha=al, nps=self.n_ps(self.r), r=self.r, nb=nb,
                                   pseudo=lambda: (c.XA[self.ps_of(seed, self.r)], pseudo_values(None, self.ps_of(seed, self.r), sel, c.yA, base)))

    # ------------------------------------------------------------ 축 x9(κ, 앵커, 입력)와 N1 기준선
    def anchor_ctx(self, kind):
        """앵커 b 를 s 자리에 넣은 Ctx. E0 자리에 c0, r0_src 자리에 y − c0·b 가 계산된다. 원천에 쓸 수 있는 값이 없으면 None."""
        c, e = self.c, self.ext
        raw = {k: np.asarray(e[k][ANCHOR_COL[kind]], float) for k in ("src", "A", "B")}
        if kind == "tddm":
            with np.errstate(invalid="ignore"):
                raw = {k: np.sqrt(np.where(v > 0, v, np.nan)) for k, v in raw.items()}
        b, rho, nrep = anchor_fill(raw["src"], raw["A"], raw["B"], c.s_src, c.sA, c.sB)
        self.notes[f"anchor_{kind}"] = dict(rho=rho, n_replaced=nrep)
        if not np.isfinite(rho) or not np.all(np.isfinite(b["src"])):
            return None, raw
        cb = H.Ctx(c.target, c.mode, c.split, c.parent, c.X_src, c.y_src, b["src"], c.macro_src, c.XA, c.yA, b["A"], c.blkA, c.XB, c.yB, b["B"],
                   c.blkB, min_cells_prior=self.HA.min_cells_prior, meta=c.meta)
        assert len(cb.y_src) == len(c.y_src), "앵커 Ctx 의 원천 행 수가 달라졌다"
        return cb, raw

    def run_x9(self):
        c = self.c; lr = BASE_LEARNER; E0 = c.E0; e = self.ext
        anc = {}
        for kind in ANCHORS:
            cb, raw = self.anchor_ctx(kind)
            anc[kind] = (cb, raw)
            if cb is not None:
                self.add(f"P0@{kind}", "none", "1", "cell", 0, 0, -1, 0.0, cb.E0 * cb.sB, cb.E0, 0)
        # N1 의 기준선(해석 계산). 라벨과 무관한 행은 n = 0 에 한 번 저장한다
        kuok = np.isfinite(anc["ku"][1]["B"]) & (anc["ku"][1]["B"] > 0)
        mk = dict(kuok=kuok)
        for kind, nm in (("ed", "B:ed_raw"), ("ku", "B:ku_raw"), ("cci", "B:cci_raw")):
            cb = anc[kind][0]
            if cb is not None:
                self.add(nm, "none", "1", "cell", 0, 0, -1, 0.0, cb.sB, 1.0, 0)
        a_s, b_s = affine_ls(c.s_src, c.y_src)
        self.add("B:s_aff", "none", "1", "cell", 0, 0, -1, 0.0, a_s + b_s * c.sB, b_s, 0)
        a_c, b_c = affine_ls(e["src"]["cci_alt"], c.y_src)
        if anc["cci"][0] is not None:
            self.add("B:cci_aff", "none", "1", "cell", 0, 0, -1, 0.0, a_c + b_c * anc["cci"][0].sB, b_c, 0)
        self.notes.update(s_aff=(a_s, b_s), cci_aff=(a_c, b_c), n_ku_missing_B=int((~kuok).sum()))
        ens_ok = anc["ku"][0] is not None and anc["cci"][0] is not None
        if ens_ok:
            self.add("B:ens", "none", "1", "cell", 0, 0, -1, 0.0, (E0 * c.sB + anc["ku"][0].E0 * anc["ku"][0].sB + anc["cci"][0].E0 * anc["cci"][0].sB) / 3.0,
                     np.nan, 0)
        if anc["ku"][0] is not None and not self.dry:                     # p4_ku 결측 셀을 채점에서 뺀 방식(대체 방식과 병기)
            self.add_masked("P0", "none", "1", "cell", 0, 0, -1, 0.0, E0 * c.sB, mk)
            self.add_masked("B:ku_raw", "none", "1", "cell", 0, 0, -1, 0.0, anc["ku"][0].sB, mk)
            self.add_masked("P0@ku", "none", "1", "cell", 0, 0, -1, 0.0, anc["ku"][0].E0 * anc["ku"][0].sB, mk)
        # 결측 대체 민감도(계획서 §6A.3 X9, 개정 7): 공변량 결측을 원천 셀 중앙값으로 채워 다시 계산한 p4_ku, p2_edaphic 의 기준선
        pt = self.phys_tr()
        tr_ok = pt is not None
        if tr_ok:
            tr = {}
            for kind, col in TR_ANCHORS:
                b, rho, nrep = anchor_fill(pt["src"][col], pt["A"][col], pt["B"][col], c.s_src, c.sA, c.sB)
                if not np.isfinite(rho) or not np.all(np.isfinite(b["src"])):
                    continue
                c0 = H.ls_E(c.y_src, b["src"])
                tr[kind] = (c0, b)
                self.add(f"P0@{kind}_tr", "none", "1", "cell", 0, 0, -1, 0.0, c0 * b["B"], c0, 0)
                self.add(f"B:{kind}_raw_tr", "none", "1", "cell", 0, 0, -1, 0.0, b["B"], 1.0, 0)
                with np.errstate(invalid="ignore"):
                    dB = np.abs(np.asarray(pt["B"][col], float) - np.asarray(e["B"][col], float))
                    dS = np.abs(np.asarray(pt["src"][col], float) - np.asarray(e["src"][col], float))
                self.notes[f"phys_tr_{kind}"] = dict(rho=rho, c0=c0, n_replaced=nrep, n_changed_B=int(np.nansum(dB > 1e-9)),
                                                     max_abs_diff_B=float(np.nanmax(dB)) if np.isfinite(dB).any() else 0.0,
                                                     n_changed_src=int(np.nansum(dS > 1e-9)),
                                                     max_abs_diff_src=float(np.nanmax(dS)) if np.isfinite(dS).any() else 0.0)
            if "ku" in tr and anc["cci"][0] is not None:
                self.add("B:ens_tr", "none", "1", "cell", 0, 0, -1, 0.0,
                         (E0 * c.sB + tr["ku"][0] * tr["ku"][1]["B"] + anc["cci"][0].E0 * anc["cci"][0].sB) / 3.0, np.nan, 0)
            self.notes["phys_tr_info"] = pt.get("info", {})
        for n, d in self.cells_iter():
            sel = self.sel_of(n, d)
            E1, E2 = self.coefs(sel)
            nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
            self.analytic(n, d, sel, E1, E2, ("P1",))
            fit_n = n in X9_FIT_N
            p1b = {}
            for kind in ANCHORS:                                          # 해석식 P1@b 는 n 전 격자
                cb = anc[kind][0]
                if cb is None:
                    continue
                c1, _ = self.coefs(sel, cb)
                p1b[kind] = (c1, c1 * cb.sB)
                self.add(f"P1@{kind}", "none", "1", "cell", n, d, -1, 0.0, c1 * cb.sB, c1, nl, nb)
            if "ku" in p1b:
                self.add_masked("P1", "none", "1", "cell", n, d, -1, 0.0, E1 * c.sB, mk)
                self.add_masked("P1@ku", "none", "1", "cell", n, d, -1, 0.0, p1b["ku"][1], mk)
            if ens_ok and n != 0:
                self.add("B:ens", "none", "1", "cell", n, d, -1, 0.0, (E1 * c.sB + p1b["ku"][1] + p1b["cci"][1]) / 3.0, np.nan, nl, nb)
            Ek = {k: self.coefs(sel, kappa=k)[0] for k in KAPPA_VAR}
            for k in KAPPA_VAR:
                self.add(f"P1@k{int(k)}", "none", "1", "cell", n, d, -1, 0.0, Ek[k] * c.sB, Ek[k], nl, nb, kappa=k)
            if not fit_n:
                continue
            for seed in self.seeds:
                if tr_ok and n in TR_FIT_N:                               # 결측 대체 민감도: 학습 쪽 중앙값으로 채운 두 열을 입력으로 준 F1k
                    self.direct(lr, "F1k@tr", n, d, sel, seed, kind="f1k_tr", nb=nb)
                if nl == 0:                                               # n = 0: R1@k = R0(적합 하나를 나눠 쓴다)
                    g0 = self.resid(lr, "R0", n, d, sel, seed, E0 * c.sA[sel], E0 * c.sB, E0, emit=False)
                    for k in KAPPA_VAR:
                        self.emit_resid(f"R1@k{int(k)}", lr, "1", "cell", n, d, seed, E0 * c.sB, g0, E0, 0, 0, self.F.last_flag, nrow=self.nsrc,
                                        kappa=k)
                else:
                    for k in KAPPA_VAR:
                        self.resid(lr, f"R1@k{int(k)}", n, d, sel, seed, Ek[k] * c.sA[sel], Ek[k] * c.sB, Ek[k], kappa=k, nb=nb)
                for kind in ANCHORS:
                    cb = anc[kind][0]
                    if cb is not None:
                        self.base_cell(lr, n, d, sel, p1b[kind][0], seed, ("R0", "R1"), cx=cb, suffix=f"@{kind}")
                self.base_cell(lr, n, d, sel, E1, seed, ("R0", "R1"), suffix="@x23c", kind="drop_c")

    # ------------------------------------------------------------ 축 x9f(표지 셀 제외 변형)
    def variant_ctx(self, variant):
        """표지 셀을 원천과 채점에서 뺀 Ctx 와 A 풀의 유지 표지(keepA). 라벨은 전체 풀에서 뽑은 뒤 표지 셀을 지운다."""
        c, e = self.c, self.ext
        col = VARIANT_FLAG[variant]
        ks, kA, kB = ~np.asarray(e["src"][col], bool), ~np.asarray(e["A"][col], bool), ~np.asarray(e["B"][col], bool)
        info = dict(n_src_drop=int((~ks).sum()), n_A_drop=int((~kA).sum()), n_B_drop=int((~kB).sum()), n_src=int(ks.sum()), n_B=int(kB.sum()))
        self.notes[f"variant_{variant}"] = info
        if ks.sum() < 3 or kB.sum() == 0:
            return None, kA
        cv = H.Ctx(c.target, c.mode, c.split, c.parent, c.X_src[ks], c.y_src[ks], c.s_src[ks], c.macro_src[ks], c.XA, c.yA, c.sA, c.blkA,
                   c.XB[kB], c.yB[kB], c.sB[kB], c.blkB[kB], min_cells_prior=self.HA.min_cells_prior, meta=c.meta)
        if TRACE_X is not None:
            TRACE_X.append(dict(kind="variant", variant=variant, src_loc=np.asarray(e["src"]["loc_id"])[ks], B_loc=np.asarray(e["B"]["loc_id"])[kB],
                                keepA=kA.copy()))
        return cv, kA

    def run_x9f(self, methods=("P1", "D0", "R0", "R1")):
        c = self.c; lr = BASE_LEARNER
        for variant in VARIANTS:
            cv, kA = self.variant_ctx(variant)
            if cv is None:
                continue
            st = BlockStore(f"{c.target}~{variant}|{c.mode}", c.split, cv.blkB,
                            meta=dict(target=c.target, mode=c.mode, part=self.part, axis=self.axis, variant=variant))
            self.extra.append(st)
            self.add("P0", "none", "1", "cell", 0, 0, -1, 0.0, cv.E0 * cv.sB, cv.E0, 0, st=st, y=cv.yB, variant=variant)
            for n, d in self.cells_iter():
                sel0 = self.sel_of(n, d)
                sel = sel0[kA[sel0]]
                if TRACE_X is not None:
                    TRACE_X.append(dict(kind="variant_sel", variant=variant, n=n, draw=d, sel=sel.copy(), sel0=sel0.copy()))
                E1, E2 = self.coefs(sel, cv)
                self.analytic(n, d, sel, E1, E2, ("P1",), cx=cv, st=st, variant=variant)
                for seed in self.seeds:
                    self.base_cell(lr, n, d, sel, E1, seed, [m for m in methods if not m.startswith("P")], cx=cv, st=st, variant=variant)

    # ------------------------------------------------------------ 축 prox(채점 셀 고정 근접 단)
    def run_prox(self):
        c = self.c; lr = BASE_LEARNER; E0 = c.E0; nA, nB = self.nA, self.nB

        def tier(placement, n, d, rep, kind):
            fid = prox_folds(c.target, c.mode, c.split, rep, nB)
            sizes = np.bincount(fid, minlength=PROX_FOLDS)
            if kind == "inblk" and (nB - sizes.max()) < n:
                return
            P1 = np.full(nB, np.nan); cnt = np.zeros(nB, int)
            PD = {s: np.full(nB, np.nan) for s in self.seeds}; A1 = np.full(nB, np.nan); GR = {s: np.full(nB, np.nan) for s in self.seeds}
            nlab, Es, fl = [], [], ""
            for k in range(PROX_FOLDS):
                te = np.where(fid == k)[0]
                if len(te) == 0:
                    continue
                iA, iB = prox_labels(kind, c.target, c.mode, c.split, rep, k, np.where(fid != k)[0], nA, n=n, draw=d)
                assert len(np.intersect1d(iB, te)) == 0, "채점 fold 의 셀이 라벨에 들어갔다"
                if kind == "randm":
                    assert len(iA) + len(iB) == nA, "S_randm 의 라벨 수가 |A| 가 아니다"
                Xl = np.vstack([c.XA[iA], c.XB[iB]]); yl = np.concatenate([c.yA[iA], c.yB[iB]]); sl = np.concatenate([c.sA[iA], c.sB[iB]])
                m_ = len(yl)
                E1 = shrink(H.ls_E(yl, sl), E0, m_, self.kappa) if m_ else E0
                P1[te] = E1 * c.sB[te]; A1[te] = E1 * c.sB[te]; cnt[te] += 1
                nlab.append(m_); Es.append(E1)
                sel = np.concatenate([iA, 100000000 + iB]).astype(int)       # 추적용 색인(B 셀은 1e8 을 더해 구분한다)
                for seed in self.seeds:
                    _, (p,) = self.fit(lr, "D0", lambda: stack_rows(c.X_src, c.y_src, Xl, yl), seed, [c.XB[te]], n, d, sel, self.nsrc + m_,
                                       placement=placement, fold=k)
                    PD[seed][te] = p; fl = fl or self.F.last_flag
                    _, (g,) = self.fit(lr, "R1", lambda: stack_rows(c.X_src, c.r0_src, Xl, yl - E1 * sl), seed, [c.XB[te]], n, d, sel,
                                       self.nsrc + m_, placement=placement, fold=k)
                    GR[seed][te] = g; fl = fl or self.F.last_flag
            assert np.all(cnt == 1), "채점 셀이 반복마다 한 번씩 채점되지 않았다"
            nl = int(round(float(np.mean(nlab)))) if nlab else 0
            Em = float(np.mean(Es)) if Es else E0
            self.add("P1", "none", "1", placement, n, d, -1, 0.0, P1, Em, nl)
            for seed in self.seeds:
                self.add("D0", lr, "1", placement, n, d, seed, 1.0, PD[seed], np.nan, nl, flag=fl, nrow=self.nsrc + nl)
                self.emit_resid("R1", lr, "1", placement, n, d, seed, A1, GR[seed], Em, nl, flag=fl, nrow=self.nsrc + nl)

        for rep in range(PROX_REPS):
            tier("rand5", -1, rep, rep, "rand5")
            tier("randm", -1, rep, rep, "randm")
        for n in [v for v in INBLK_N if v in self.grid]:
            for d in range(self.draws):
                tier("inblk", n, d, d, "inblk")
        dAB = None if self.dry else nearest_km(self.ext["A"]["lat"], self.ext["A"]["lon"], self.ext["B"]["lat"], self.ext["B"]["lon"])
        for dk in BUFFERS:                                                # S_buf: A 에서 채점 셀 d km 안의 셀을 뺀다(fold 없음, 서술 통계)
            sel = np.arange(nA) if dAB is None else np.where(~(dAB < float(dk)))[0]
            E1, E2 = self.coefs(sel)
            nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
            pl = f"buf{int(dk)}"
            self.add("P1", "none", "1", pl, -1, 0, -1, 0.0, E1 * c.sB, E1, nl, nb)
            for seed in self.seeds:
                self.direct(lr, "D0", -1, 0, sel, seed, placement=pl, nb=nb)
                self.resid(lr, "R1", -1, 0, sel, seed, E1 * c.sA[sel], E1 * c.sB, E1, placement=pl, nb=nb)

    # ------------------------------------------------------------ 축 x5(교차 적합 구간, 변수 기여)
    def _interval(self, M, learner, n, d, seed, lam, yhat, q_by_level, **kw):
        c = self.c
        if self.dry:
            for q in q_by_level:
                self.n_rows += 2 if np.isfinite(q) else 1
            return
        yh = np.maximum(np.asarray(yhat, float), Y_FLOOR_CM)
        if not np.all(np.isfinite(yh)):
            self.n_bad += 1
            return
        for (lv, _), q in zip(CONF_LEVELS, q_by_level):
            if not np.isfinite(q):                                        # k > m: 구간이 무한대다. 포함률 1 로 저장하고 폭은 저장하지 않는다
                self.add_stat(f"cov{lv}:{M}", learner, n, d, seed, lam, np.zeros(self.nB), q=np.inf, **dict(kw, cv_flag=(kw.get("cv_flag", "") + ";q_inf").strip(";")))
                continue
            lo, hi = yh * np.exp(-q), yh * np.exp(q)
            miss = ((c.yB < lo) | (c.yB > hi)).astype(float)
            self.add_stat(f"cov{lv}:{M}", learner, n, d, seed, lam, miss, q=q, **kw)
            self.add_stat(f"wid{lv}:{M}", learner, n, d, seed, lam, hi - lo, q=q, **kw)

    def run_x5(self):
        c = self.c; lr = BASE_LEARNER; E0 = c.E0; seed = self.seeds[0]; lam = CONF_LAM
        # (1) n = 0 구간: 원천 지역 하나 제외 방식의 표본 밖 점수를 지역 등가중으로 합친 분위
        if 0 in self.grid:
            regs = [r for r, k in Counter(c.macro_src.tolist()).items() if k >= MIN_REGION_CELLS]
            self.notes["n0_regions"] = sorted(str(r) for r in regs)
            if len(regs) >= 2:
                sP, sR, w = [], [], []
                none = np.zeros(0, int)
                for r in sorted(regs, key=str):
                    te = c.macro_src == r; tr = ~te
                    E_r = H.ls_E(c.y_src[tr], c.s_src[tr])
                    _, (g,) = self.fit(lr, "R0", lambda: (c.X_src[tr], c.y_src[tr] - E_r * c.s_src[tr], None), seed, [c.X_src[te]], 0, 0, none,
                                       int(tr.sum()), cv_region=str(r))
                    sP.append(log_score(c.y_src[te], E_r * c.s_src[te])); sR.append(log_score(c.y_src[te], E_r * c.s_src[te] + lam * g))
                    w.append(np.full(int(te.sum()), 1.0 / max(int(te.sum()), 1)))
                g0 = self.resid(lr, "R0", 0, 0, none, seed, E0 * c.sA[none], E0 * c.sB, E0, emit=False)
                sP, sR, w = np.concatenate(sP), np.concatenate(sR), np.concatenate(w)
                kw = dict(cv_folds=len(regs), cv_flag="loro_src")
                self._interval("P0", "none", 0, 0, -1, 0.0, E0 * c.sB, [weighted_quantile(sP, w, 1 - al) for _, al in CONF_LEVELS], E_used=E0, **kw)
                self._interval("R0", lr, 0, 0, seed, lam, E0 * c.sB + lam * g0, [weighted_quantile(sR, w, 1 - al) for _, al in CONF_LEVELS],
                               E_used=E0, **kw)
        # (2) 라벨 n개 구간: A 블록 단위 교차 적합 점수의 분위
        for n, d in self.cells_iter([v for v in self.grid if v != 0]):
            sel = self.sel_of(n, d)
            nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
            fid, K, flag = cv_folds_of(c.blkA[sel], c.target, c.mode, c.split, n, d)
            if K < 2:
                self.notes[f"cv_skip_{n}_{d}"] = flag or "folds<2"
                continue
            sc = {m: np.full(nl, np.nan) for m in ("P1", "R0", "R1")}
            for j in range(K):
                tem = fid == j
                te, tr = sel[tem], sel[~tem]
                E_tr, _ = self.coefs(tr)                                   # 묶음마다 E 를 학습 쪽 라벨로 다시 구한다
                _, (g_0,) = self.fit(lr, "R0", lambda: stack_rows(c.X_src, c.r0_src, c.XA[tr], c.yA[tr] - E0 * c.sA[tr]), seed, [c.XA[te]], n, d, tr,
                                     self.nsrc + len(tr), cv_fold=int(j))
                _, (g_1,) = self.fit(lr, "R1", lambda: stack_rows(c.X_src, c.r0_src, c.XA[tr], c.yA[tr] - E_tr * c.sA[tr]), seed, [c.XA[te]], n, d,
                                     tr, self.nsrc + len(tr), cv_fold=int(j))
                sc["P1"][tem] = log_score(c.yA[te], E_tr * c.sA[te])
                sc["R0"][tem] = log_score(c.yA[te], E0 * c.sA[te] + lam * g_0)
                sc["R1"][tem] = log_score(c.yA[te], E_tr * c.sA[te] + lam * g_1)
            E1, _ = self.coefs(sel)
            g0 = self.resid(lr, "R0", n, d, sel, seed, E0 * c.sA[sel], E0 * c.sB, E0, emit=False)
            g1 = self.resid(lr, "R1", n, d, sel, seed, E1 * c.sA[sel], E1 * c.sB, E1, emit=False)
            kw = dict(n_lab=nl, nb_lab=nb, cv_folds=int(K), cv_flag=flag)
            self._interval("P1", "none", n, d, -1, 0.0, E1 * c.sB, [conformal_q(sc["P1"], al) for _, al in CONF_LEVELS], E_used=E1, **kw)
            self._interval("R0", lr, n, d, seed, lam, E0 * c.sB + lam * g0, [conformal_q(sc["R0"], al) for _, al in CONF_LEVELS], E_used=E0, **kw)
            self._interval("R1", lr, n, d, seed, lam, E1 * c.sB + lam * g1, [conformal_q(sc["R1"], al) for _, al in CONF_LEVELS], E_used=E1, **kw)
        # (3) 변수 기여: CatBoost ShapValues 의 평균 절댓값(D0, F1k, R1 의 g). 추출 0, seed 0
        for n in [v for v in SHAP_N if v in self.grid and (v in (0, -1) or 0 < v < self.nA)]:
            sel = self.sel_of(n, 0)
            E1, _ = self.coefs(sel)
            for m, kind, resid in (("D0", "x25", False), ("F1k", "f1k", False), ("R1", "x25", True)):
                Xs, Xl, XB = self.inputs(kind, sel, E1)
                ys = c.r0_src if resid else c.y_src
                yl = (c.yA[sel] - E1 * c.sA[sel]) if resid else c.yA[sel]
                mdl, _ = self.fit(lr, m, lambda: stack_rows(Xs, ys, Xl, yl), seed, [XB], n, 0, sel, self.nsrc + len(sel), shap=True)
                if self.dry or mdl is None:
                    continue
                try:
                    from catboost import Pool
                    sv = np.asarray(mdl.get_feature_importance(Pool(XB), type="ShapValues", thread_count=int(self.HA.threads)), float)
                except Exception as ex:                                   # noqa: BLE001
                    self.notes[f"shap_fail_{m}_{n}"] = repr(ex)[:160]
                    continue
                names = FEATS + (["p4_ku", "p2_edaphic"] if kind == "f1k" else [])
                imp = np.abs(sv[:, :-1]).mean(0)
                self.shap += [dict(self.base_row, method=m, learner=lr, n=int(n), n_lab=int(len(sel)), draw=0, seed=int(seed), feature=f,
                                   mean_abs_shap=float(v), n_cells=int(self.nB)) for f, v in zip(names, imp)]

    # ------------------------------------------------------------ 실행과 마무리
    def run(self, learners=None):
        ax = self.axis
        if ax == "base":
            self.run_base()
        elif ax == "x1":
            self.run_x1()
        elif ax == "x2":
            self.run_x2()
        elif ax == "x9":
            self.run_x9()
        elif ax == "x9f":
            self.run_x9f()
        elif ax == "n3":
            self.run_n3([v for v in N3_LEARNERS if (not learners or v in learners)])
        elif ax == "n4":
            self.run_base(("P1", "P2", "D0", "R0", "R1"), strata=False, cell=False)
        elif ax == "n4a":
            self.run_base(("P1", "P2"), strata=False, cell=False)
        elif ax == "prox":
            self.run_prox()
        elif ax == "x5":
            self.run_x5()
        elif ax == "ridge":
            self.run_learner("ridge", RIDGE_METHODS)
        elif ax == "r0":
            for lr in (learners or R0_LEARNERS):
                self.run_learner(lr, ("R0",))
        elif ax == "nn":
            for lr in (learners or NN_LEARNERS):
                self.run_learner(lr, NN_METHODS)
        else:
            raise ValueError(f"알 수 없는 축 {ax}")
        return self

    def finish(self):
        F = self.F
        stores = [self.st] + list(self.extra)
        n_ml = sum(1 for st in stores for k in st.keys if k[1] != "none")
        n_fail = int(sum(F.fail.values())) + int(self.n_bad)
        status = "ok" if n_fail == 0 else ("failed" if n_ml == 0 else "partial")
        stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(self.n_rows), status=status,
                     n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                     est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(sum(len(st) for st in stores)),
                     n_stored_ml=int(n_ml), n_nonfinite_keys=int(self.n_bad), notes=self.notes, shap_rows=self.shap,
                     n_stores=len(stores))
        self.st.extra = list(self.extra)
        return self.rows, self.st, stats, self.cells


def run_ctx_x(c, ext, axis, HA, dry=False, learners=None, tuned=None):
    """작업 단위 하나에서 축 하나를 실행한다. 반환 (rows, BlockStore, 통계 dict, CellBook 또는 None).
    셀 제외 변형의 저장소는 BlockStore 의 extra 속성(목록)에, 변수 기여 행은 통계의 shap_rows 에 있다."""
    if len(c.yB) == 0 or len(c.yA) == 0:
        return [], None, dict(n_fit={}, sec={}, fail={}, n_rows=0, status="no_eval", n_fit_detail={}, sec_detail={}, rows_detail={}, est_detail={},
                              errors=[], n_stored=0, n_stored_ml=0, n_nonfinite_keys=0, notes={}, shap_rows=[], n_stores=0), None
    return Runner(c, ext, axis, HA, dry=dry, tuned=tuned).run(learners).finish()


# ================================================================ 조각 입출력
def shard_paths_x(a, axis, target, mode, split, learner=None):
    b = a.SHARDS / (f"{axis_tag(a, axis)}__{AX[axis]['part']}__{target}__{mode}__s{split}" + (f"__{learner}" if learner else ""))
    return dict(runs=Path(str(b) + "_runs.csv"), npz=Path(str(b) + "_blocksse.npz"), unit=Path(str(b) + "_unit.json"),
                cells=Path(str(b) + "_cells.npz"), shap=Path(str(b) + "_shap.csv"))


_SHA: dict = {}


def _sha_cached(path):
    k = str(path)
    if k not in _SHA:
        _SHA[k] = file_sha(path)
    return _SHA[k]


def unit_cfg_x(a, axis, learner=None):
    """결과에 영향을 주는 설정 요약. 대상, 분할, tag, 워커, GPU 목록은 넣지 않는다(h40 의 unit_cfg 와 같은 원칙)."""
    HA = h40_args(a, axis)
    part = AX[axis]["part"]
    d = dict(axis=axis, part=part, learners=[learner] if learner else (axis_learners(a, axis) or ["cpu"]), n_grid=list(HA.N_GRID),
             draws=int(HA.DRAWS), seeds=list(HA.SEEDS), methods=list(AXIS_METHODS[axis]), lams=list(HA.LAMS), kappa=float(HA.kappa), r=float(HA.r),
             buffer_km=float(HA.buffer_km), k_sub=str(HA.k_sub), min_cells_prior=int(HA.min_cells_prior),
             subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans",
             design=dict(r_grid=list(R_GRID), placebos=list(PLACEBOS), kappa_var=list(KAPPA_VAR), cv_folds=CV_FOLDS, prox_folds=PROX_FOLDS,
                         prox_reps=PROX_REPS, buffers=list(BUFFERS), anchors=list(ANCHORS), alphas=list(ALPHAS_X2), cell_n=list(CELL_N),
                         conf=[list(v) for v in CONF_LEVELS], conf_lam=CONF_LAM, y_floor=Y_FLOOR_CM, min_region_cells=MIN_REGION_CELLS))
    if part == "cpu":
        d.update(cb_iters=int(HA.cb_iters), cb_threads=int(HA.threads))
    else:
        d.update(epochs=int(HA.epochs), realmlp_epochs=int(HA.realmlp_epochs))
    if axis == "x9f":
        d.update(label_flags=f"{a.FLAGS.name}:{_sha_cached(a.FLAGS)}")
    if axis == "x9":
        d.update(tdd_matched=f"{a.TDDM.name}:{_sha_cached(a.TDDM)}")
    if axis == "n3":
        d.update(cb_hi=dict(CB_HI), rf=dict(RF_CFG), tune_grid=[list(v) for v in TUNE_GRID], tune_lr=TUNE_LR)
    return d


def unit_state_x(a, axis, target, mode, split, learner=None):
    """(완료 여부, 사유). 완료 = 조각 파일이 있고, 설정 해시가 같고, status 가 failed 가 아니다."""
    if axis == "g45":
        return H.unit_state(a.HG, "gpu", target, mode, split, learner)
    p = shard_paths_x(a, axis, target, mode, split, learner)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != H.cfg_hash(unit_cfg_x(a, axis, learner)):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    if u.get("tuned_missing"):
        return False, "catboost_tuned 선택 설정 없음(선택 표가 준비되면 다시 실행)"
    return True, str(u.get("status", "ok"))


def _atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, path)


def write_shard_x(a, axis, c, rows, st, stats, cells, elapsed, learner=None, ext=None):
    p = shard_paths_x(a, axis, c.target, c.mode, c.split, learner)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(rows), p["runs"])
    save_stores([st] + list(getattr(st, "extra", [])), p["npz"])
    if cells is not None:
        cells.save(p["cells"])
    stats = dict(stats)
    shap = stats.pop("shap_rows", [])
    if shap:
        _atomic_csv(pd.DataFrame(shap), p["shap"])
    HA = h40_args(a, axis)
    cfg = unit_cfg_x(a, axis, learner)
    unit = {**stats, **c.meta}
    unit.update(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=AX[axis]["part"], axis=axis, learner=learner or "",
                tag=axis_tag(a, axis), elapsed_s=round(elapsed, 1), n_fit_total=int(sum(stats["n_fit"].values())), cfg=cfg,
                cfg_hash=H.cfg_hash(cfg), cfg_common=H.cfg_hash(cfg, common=True), code_sha=code_sha_x(), code_sha_h40=H.code_sha(),
                threads=int(HA.threads), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), n_shap_rows=len(shap),
                has_cells=bool(cells is not None), flags_src=(ext or {}).get("flags_src", ""), tdd_src=(ext or {}).get("tdd_src", ""))
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))      # 완료 표지는 마지막에 쓴다
    return unit


# ================================================================ catboost_tuned 의 설정 선택(원천 지역 하나 제외 교차검증)
def select_sig(a):
    HA = h40_args(a, "n3")
    d = dict(grid=[list(v) for v in TUNE_GRID], lr=TUNE_LR, min_cells=MIN_REGION_CELLS, buffer_km=float(HA.buffer_km), k_sub=str(HA.k_sub),
             subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans", threads=CB_THREADS)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def select_rows(a, target, mode, dry=False):
    """대상·모드 하나의 선택 표. 원천 macro 지역(30셀 이상) 하나 제외 교차검증의 지역 등가중 RMSE 를 9개 설정에 대해 구한다.
    직접(D)은 y 를, 잔차(R)는 학습 쪽 원천으로 구한 E 의 잔차를 학습한다(선택 기준은 λ = 1.0 의 예측 RMSE, λ = 0.25 값은 보조 열).
    대상 라벨과 채점 셀은 쓰지 않는다. seed 0."""
    HA = h40_args(a, "n3")
    D = get_data_x(HA)
    df = D.df
    _, _, src_idx, _ = D.source_idx(target, mode)
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    y, s, mac = df.y.values[src], df.s.values[src], df.macro.values[src]
    regs = sorted(str(r) for r, k in Counter(mac.tolist()).items() if k >= MIN_REGION_CELLS)
    base = dict(target=target, mode=mode, n_src=int(len(src)), n_regions=len(regs), regions=",".join(regs), sel_sig=select_sig(a))
    if dry:
        return [dict(base, kind=k, depth=dp, iterations=it, n_fit=len(regs)) for k in ("D", "R") for dp, it in TUNE_GRID]
    X = df[FEATS].values[src].astype(np.float32)
    from catboost import CatBoostRegressor
    rows = []
    for kind in ("D", "R"):
        for dp, it in TUNE_GRID:
            t0 = time.time(); r1, r25 = [], []
            for r in regs:
                te = mac == r; tr = ~te
                E = H.ls_E(y[tr], s[tr]) if kind == "R" else 0.0
                mdl = CatBoostRegressor(iterations=int(it), learning_rate=TUNE_LR, depth=int(dp), l2_leaf_reg=3.0, random_seed=0, verbose=0,
                                        allow_writing_files=False, thread_count=CB_THREADS)
                mdl.fit(X[tr], y[tr] - E * s[tr])
                g = np.asarray(mdl.predict(X[te], thread_count=CB_THREADS), float)
                r1.append(float(np.sqrt(np.mean((E * s[te] + g - y[te]) ** 2))))
                r25.append(float(np.sqrt(np.mean((E * s[te] + LAM_BASE * g - y[te]) ** 2))) if kind == "R" else np.nan)
            rows.append(dict(base, kind=kind, depth=int(dp), iterations=int(it), learning_rate=TUNE_LR,
                             score_rmse=float(np.mean(r1)) if r1 else np.nan, score_rmse_lam025=float(np.mean(r25)) if r25 else np.nan,
                             rmse_by_region=json.dumps(dict(zip(regs, [round(v, 4) for v in r1]))), n_fit=len(regs),
                             sec=round(time.time() - t0, 1)))
    return rows


def pick_tuned(rows):
    """선택 표에 selected 열을 붙인다. (대상, 모드, 직접·잔차)마다 점수가 가장 작은 설정, 동률이면 용량(반복 × 2^깊이)이 작은 설정.
    교차검증을 할 수 없으면(30셀 이상 원천 지역이 2개 미만) 가장 작은 설정을 고르고 표시한다."""
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["capacity"] = df.iterations.astype(float) * (2.0 ** df.depth.astype(float))
    df["selected"] = False; df["select_flag"] = ""
    for _, g in df.groupby(["target", "mode", "kind"], sort=False):
        sc = g.score_rmse.values.astype(float) if "score_rmse" in g else np.full(len(g), np.nan)
        if np.isfinite(sc).any() and int(g.n_regions.iloc[0]) >= 2:
            o = sorted(range(len(g)), key=lambda i: (sc[i] if np.isfinite(sc[i]) else np.inf, g.capacity.values[i]))
            df.loc[g.index[o[0]], "selected"] = True
        else:
            j = int(np.argmin(g.capacity.values))
            df.loc[g.index[j], "selected"] = True; df.loc[g.index, "select_flag"] = "no_cv(regions<2)"
    return df


def load_tuned(a, target, mode):
    """선택 표에서 대상·모드의 설정 {"D": {...}, "R": {...}} 을 읽는다. 표가 없거나, 비었거나, 읽을 수 없거나, 설정 표지가 다르거나,
    행이 없으면 빈 dict 다(호출한 쪽이 catboost_tuned 만 건너뛴다)."""
    if not a.SELECT.exists():
        return {}
    try:
        df = pd.read_csv(a.SELECT)
    except Exception:                                                     # noqa: BLE001  빈 파일(EmptyDataError)과 깨진 파일
        return {}
    need = {"target", "mode", "kind", "selected", "iterations", "depth", "learning_rate"}
    if not len(df) or not need <= set(df.columns):
        return {}
    if "sel_sig" in df:
        df = df[df.sel_sig.astype(str) == select_sig(a)]
    df = df[(df.target == target) & (df["mode"] == mode) & (df.selected.astype(str).str.lower() == "true")]
    return {str(r.kind): dict(iterations=int(r.iterations), depth=int(r.depth), learning_rate=float(r.learning_rate), l2_leaf_reg=3.0)
            for r in df.itertuples()}


def select_ready(a, pairs):
    """선택 표가 있고 설정 표지가 같으며 필요한 대상·모드를 모두 덮는지."""
    if not a.SELECT.exists():
        return False
    try:
        df = pd.read_csv(a.SELECT)
    except Exception:                                                     # noqa: BLE001
        return False
    if not len(df) or "sel_sig" not in df or set(df.sel_sig.astype(str)) != {select_sig(a)}:
        return False
    have = {(t, m, k) for t, m, k, s_ in zip(df.target, df["mode"], df.kind, df.selected.astype(str).str.lower()) if s_ == "true"}
    return all((t, m, k) in have for t, m in pairs for k in ("D", "R"))


def write_select(a, rows):
    """선택 표를 쓴다. 새 행이 없으면 표를 건드리지 않는다(선택 작업이 모두 실패했을 때 기존 표를 빈 표로 덮어쓰지 않는다).
    같은 설정 표지의 기존 행은 보존하고 같은 대상·모드의 행만 새 행으로 바꾼다."""
    df = pick_tuned(rows)
    if not len(df):
        print(f"  [warn] catboost_tuned 선택 표에 쓸 새 행이 없다. 표를 쓰지 않는다: {a.SELECT}", flush=True)
        return df
    a.OUT.mkdir(parents=True, exist_ok=True)
    if a.SELECT.exists():                                                 # 이어서 실행: 같은 설정 표지의 기존 행을 보존하고 새 행으로 바꾼다
        try:
            old = pd.read_csv(a.SELECT)
            if len(old) and "sel_sig" in old:
                new_keys = set(zip(df.target, df["mode"]))
                old = old[(old.sel_sig.astype(str) == select_sig(a)) & ~pd.Series(list(zip(old.target, old["mode"])), index=old.index).isin(new_keys)]
                df = pd.concat([old, df], ignore_index=True)
        except Exception:                                                 # noqa: BLE001  빈 파일이나 깨진 파일은 새 표로 바꾼다
            pass
    _atomic_csv(df, a.SELECT)
    return df


# ================================================================ 작업 단위
_UC: dict = {}


def run_unit_x(a, axis, target, mode, split, learner=None, dry=False):
    t0 = time.time()
    HA = h40_args(a, axis)
    D = get_data_x(HA)
    XT = ext_tables(a, D, dry)
    XT.require(axis)
    if dry:                                                              # 적합 수 세기: 같은 (대상, 모드, 분할)의 Ctx 를 축 사이에 나눠 쓴다
        k = (target, mode, int(split))
        if k not in _UC:
            if len(_UC) > 256:
                _UC.clear()
            _UC[k] = build_unit(D, HA, XT, target, mode, split)
        c, ext = _UC[k]
    else:
        c, ext = build_unit(D, HA, XT, target, mode, split)
        old = shard_paths_x(a, axis, target, mode, split, learner)
        for k_ in ("unit", "cells", "shap"):                              # 이전 세대의 완료 표지와 부속 파일을 먼저 지운다(새 세대가 쓰지 않는 파일이 남지 않게)
            old[k_].unlink(missing_ok=True)
    tuned = load_tuned(a, target, mode) if (axis == "n3" and not dry) else {}
    lrs = [learner] if learner else None
    tuned_missing = bool(axis == "n3" and not dry and not all(k_ in tuned for k_ in ("D", "R")))
    if tuned_missing:                                                     # 선택 설정이 없으면 catboost_tuned 만 건너뛴다(catboost, rf 는 남긴다)
        lrs = [v for v in N3_LEARNERS if v != "catboost_tuned"]
        print(f"  [warn] {axis}|{target}|{mode}|s{split}: catboost_tuned 의 선택 설정이 없다. catboost_tuned 를 건너뛴다"
              f"(조각은 미완료로 남고 --resume 에서 다시 실행된다)", flush=True)
    rows, st, stats, cells = run_ctx_x(c, ext, axis, HA, dry=dry, learners=lrs, tuned=tuned)
    if dry:
        return dict(axis=axis, target=target, mode=mode, split=split, part=AX[axis]["part"], learner=learner or "", n_A=c.meta["n_A"],
                    n_eval=c.meta["n_eval"], nb_eval=c.meta["nb_eval"], n_src=c.meta["n_src"], valid=c.meta["valid"], n_rows=stats["n_rows"],
                    est_s=round(float(sum(stats["est_detail"].values())), 1), fit_total=int(sum(stats["n_fit"].values())),
                    _detail=stats["n_fit_detail"], _rows=stats["rows_detail"], _est=stats["est_detail"])
    if st is None:
        raise RuntimeError(f"{axis}|{target}|{mode}|s{split}: 채점 셀 또는 A 셀이 없다")
    if axis == "n3":
        stats["tuned"] = tuned
        stats["tuned_missing"] = tuned_missing
        if tuned_missing and stats.get("status") == "ok":
            stats["status"] = "partial"
    return write_shard_x(a, axis, c, rows, st, stats, cells, time.time() - t0, learner, ext)


def enumerate_x(a):
    """작업 단위 목록 (축, 대상, 모드, 분할, 학습기)과 건너뛴 분할의 기록. 열거는 h40.enumerate_units 를 쓴다(중복·무효 분할 규칙이 본 실행과 같다)."""
    units, skipped = [], []
    a.DX = {}
    for axis in a.AXES:
        part = AX[axis]["part"]
        if axis == "g45":
            if not axis_learners(a, "g45"):
                continue
            a.HG = H.parse_args(g45_argv(a))
            D = get_data_x(a.HG)
            us, sk = H.enumerate_units(a.HG, D, "gpu")
            us = [u for u in us if u[2] in g45_splits(a)]
            sk = [s_ for s_ in sk if s_["split"] in g45_splits(a)]
            a.DX[axis] = (a.HG, D)
        else:
            HA = h40_args(a, axis)
            D = get_data_x(HA)
            us, sk = H.enumerate_units(HA, D, part) if (part == "cpu" or HA.LEARNERS) else ([], [])
            a.DX[axis] = (HA, D)
        units += [(axis,) + tuple(u) for u in us]
        skipped += [dict(axis=axis, **s_) for s_ in sk]
    return units, skipped


def confirm_grade(axis, target, mode):
    """확인적 등급. 0 = 확인적 가설(L10, L12, L15, L29, L30)의 주 4지역 층화 평균에 쓰는 단위(축 base, x1, x2, x9, n3 의 주 4지역 x 모드), 1 = 그 밖."""
    return 0 if (axis in CONFIRM_AXES and target in MAIN4 and mode == "x") else 1


def priority_x(a, u):
    """실행 순서. cpu 는 확인적 등급(confirm_grade)이 먼저, 그다음 --axes 의 순서다. gpu 는 학습기 등급(H.LEARNER_TIER, 그 안에서
    FT-Transformer 를 뒤로)이 먼저다. 그 안에서 h40 의 unit_priority(무효 분할은 뒤, 주 4지역 → Alaska → 알래스카 하위 i → x → 나머지,
    분할 번호)를 쓴다. 벽시계 상한에 걸려도 확인적 가설의 구성 단위가 먼저 채워진다."""
    axis, t, m, sp, lr = u
    HA, D = a.DX[axis]
    base = tuple(H.unit_priority(HA, D, (t, m, sp, lr)))
    ai = a.AXES.index(axis)
    if AX[axis]["part"] == "gpu":
        return (H.LEARNER_TIER.get(lr, 0), GPU_TIER.get(lr, 0), ai) + base
    return (confirm_grade(axis, t, m), ai) + base


def unit_name_x(u):
    return f"{u[0]}|{u[1]}|{u[2]}|s{u[3]}" + (f"|{u[4]}" if len(u) > 4 and u[4] else "")


# ---------------------------------------------------------------- 워커
_WX = None


def _worker_init_x(argv, argv_g, gpu_queue, threads):
    global _WX
    warnings.filterwarnings("ignore")
    if argv_g is not None:                                                # 분할 4·5 축: h40 의 워커 초기화를 그대로 부른다(GPU 배정 포함)
        H._worker_init(argv_g, gpu_queue, threads)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get()) if gpu_queue is not None else ""
    _WX = parse_args(argv)
    if _WX.part == "gpu" and argv_g is None:                              # gpu 부분만 torch 를 읽는다(cpu 축은 CatBoost, RF, ridge 뿐이다)
        try:
            import torch
            torch.set_num_threads(int(threads))
        except Exception:                                                 # noqa: BLE001
            pass
    if any(ax == "g45" for ax in _WX.AXES) and axis_learners(_WX, "g45"):
        _WX.HG = H.parse_args(g45_argv(_WX))


def _worker_run_x(axis, target, mode, split, learner=None):
    t0 = time.time()
    if axis == "n3sel":
        return dict(axis=axis, target=target, mode=mode, rows=select_rows(_WX, target, mode), wall_s=round(time.time() - t0, 1))
    if axis == "g45":
        u = dict(H._worker_run("gpu", target, mode, split, learner))
        u["axis"] = axis
        return u
    u = run_unit_x(_WX, axis, target, mode, split, learner)
    u["wall_s"] = round(time.time() - t0, 1); u["device"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    u["n_fail"] = int(sum(u.get("fail", {}).values())) + int(u.get("n_nonfinite_keys", 0))
    return {k: u.get(k) for k in ("axis", "target", "mode", "split", "part", "learner", "n_A", "n_eval", "nb_eval", "n_src", "E0", "n_fit_total",
                                  "n_rows", "elapsed_s", "wall_s", "device", "status", "valid", "n_fail", "tuned_missing")}


# ================================================================ 로컬 선행 작업(학습 없음)
def _write_fixed(df, path, keys, what):
    """고정 입력 표를 쓴다. 기존 파일과 내용이 다르면 덮어쓰지 않고 중단한다(하위 지역 대응표와 같은 절차)."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        same = list(old.columns) == list(df.columns) and len(old) == len(df)
        if same:
            a_, b_ = old.sort_values(keys).reset_index(drop=True), df.sort_values(keys).reset_index(drop=True)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(b_[col]) and pd.api.types.is_numeric_dtype(a_[col]):
                    same &= bool(np.allclose(a_[col].values.astype(float), b_[col].values.astype(float), rtol=0, atol=1e-9, equal_nan=True))
                else:
                    same &= bool((a_[col].astype(str).values == b_[col].astype(str).values).all())
        if not same:
            raise SystemExit(f"[{what}] 기존 표 {path} 와 다르다. 덮어쓰지 않는다(정의를 바꾸려면 파일을 지우고 다시 실행)")
        print(f"[{what}] 기존 표와 같다. 다시 쓰지 않는다: {path}", flush=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(df, path)
    print(f"[{what}] {len(df):,}행 → {path}", flush=True)
    return True


def write_label_flags(a):
    """셀 표지 표. ABoVE 원자료를 좌표 소수 4자리로 결합한다(fa_failure_analysis.attach_label_meta 와 같은 규칙, 비율 0.5 이상).
    ABoVE 가 아닌 셀은 early = 0, gpr = 0 이다(CALM 과 레나 셀은 계절 최대 관측으로 가정. 원자료 문서로 확인하지 못했다)."""
    df = load_base(a.PROC)
    raw = pd.read_csv(ROOT / "data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv",
                      usecols=["team_name", "survey_technique", "latitude", "longitude", "date", "ALT_instrument", "ALT"], low_memory=False)
    raw = raw[(raw.ALT != -9999) & raw.ALT.notna() & (raw.ALT > 0) & (raw.ALT < 300) & (raw.latitude != -9999)].copy()
    dt = pd.to_datetime(raw.date, errors="coerce")
    raw["month"] = dt.dt.month
    raw["klat"], raw["klon"] = raw.latitude.round(4), raw.longitude.round(4)
    raw["is_gpr"] = (raw.ALT_instrument == "GPR").astype(float)
    raw["is_probe"] = (raw.ALT_instrument == "Probe").astype(float)
    raw["is_early"] = (raw.month <= 7).astype(float)
    raw["team_name"] = raw.team_name.astype(str).str.strip("'")
    cm = raw.groupby(["klat", "klon"], as_index=False).agg(n_pts=("ALT", "size"), frac_gpr=("is_gpr", "mean"), frac_probe=("is_probe", "mean"),
                                                         frac_early=("is_early", "mean"), team=("team_name", lambda v: v.mode().iloc[0]))
    k = pd.DataFrame(dict(loc_id=df.loc_id.values, klat=df.lat.round(4).values, klon=df.lon.round(4).values))
    d = k.merge(cm, on=["klat", "klon"], how="left")
    assert len(d) == len(df), "좌표 결합에서 행 수가 달라졌다"
    above = df.region.isin(["ABoVE_AK", "ABoVE_CA"]).values
    out = pd.DataFrame(dict(loc_id=df.loc_id.values,
                            early=(above & (d.frac_early.values >= 0.5)).astype(int), gpr=(above & (d.frac_gpr.values >= 0.5)).astype(int),
                            probe=np.where(above, (d.frac_probe.values >= 0.5).astype(int), 1),
                            team=np.where(above, d.team.fillna("미상").values, df.region.values),
                            n_pts=np.where(above, d.n_pts.fillna(0).values, 0).astype(int),
                            above_matched=np.where(above, d.n_pts.notna().values, True).astype(int)))
    print(f"[label-flags] 셀 {len(out):,} · ABoVE {int(above.sum()):,}(결합 안 됨 {int((above & (out.above_matched.values == 0)).sum())}) · "
          f"early {int(out.early.sum()):,} · gpr {int(out.gpr.sum()):,}", flush=True)
    _write_fixed(out, a.FLAGS, ["loc_id"], "label-flags")
    return dict(executed=[], skipped=[], resumed=[])


def write_tdd_matched(a):
    """연도 정합 도일 표. data/processed/m1/a2_year_matched_tdd_cells.csv 에서 loc_id, tdd_matched 만 뽑는다."""
    df = load_base(a.PROC)
    src = a.PROC / "m1" / "a2_year_matched_tdd_cells.csv"
    t = pd.read_csv(src, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id")
    miss = int((~np.isin(df.loc_id.values, t.loc_id.values)).sum())
    out = t[t.loc_id.isin(df.loc_id.values)].sort_values("loc_id").reset_index(drop=True)
    print(f"[tdd-matched] 자료 셀 {len(df):,} · 표의 셀 {len(out):,} · 빠진 셀 {miss} · tdd_matched 결측 {int(out.tdd_matched.isna().sum())}", flush=True)
    if miss:
        raise SystemExit(f"[tdd-matched] {src} 가 자료의 셀 {miss}개를 덮지 못한다. 표를 쓰지 않는다")
    _write_fixed(out, a.TDDM, ["loc_id"], "tdd-matched")
    return dict(executed=[], skipped=[], resumed=[])


# ================================================================ 집계: 저장소와 통계
def find_shards_x(shards_dir, tag):
    out = []
    d = Path(shards_dir)
    if not d.exists():
        return out
    for p in sorted(d.glob(f"{tag}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")                  # cpu 5마디, gpu 6마디(끝에 학습기)
        if len(parts) not in (5, 6) or parts[0] != tag:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), cells=Path(b + "_cells.npz"),
                            shap=Path(b + "_shap.csv"), tag=parts[0], part=parts[1], target=parts[2], mode=parts[3], split=int(parts[4][1:]),
                            learner=parts[5] if len(parts) == 6 else ""))
    return out


def is_aux_method(m):
    m = str(m)
    return ("#" in m) or m.startswith("cov") or m.startswith("wid")


def make_tm(name, by_split, D, nboot):
    """h40 의 TMx. 변형 저장소(이름에 ~ 가 있다)는 기본 대상의 분할 구조를 쓰고 point_only 를 기본 대상 기준으로 다시 정한다.
    변형에서 채점 블록이 2개 미만이 된 분할은 무효로 둔다."""
    base = name.split("|")[0].split("~")[0]
    info0 = D.split_structure(base)
    if not set(info0) >= set(by_split):
        info0 = {}
    info = {sp: dict(info0.get(sp, {}), valid=bool(info0.get(sp, {}).get("valid", True) and st.nb >= 2)) for sp, st in by_split.items()}
    tm = H.TMx(name, by_split, info, nboot)
    tm.base_target = base
    tm.point_only = base in H.MAIN_POINT
    tm.has_ci = len(tm.by_valid) > 0 and not tm.point_only
    tm.nboot = nboot if tm.has_ci else 0
    tm.expected_splits = expected_splits(info0) if info0 else sorted(tm.used)
    return tm


def expected_splits(info):
    """분할 구조(h40.Data.split_structure)에서 실행하기로 한 분할의 목록. h40.enumerate_units 와 같은 규칙이다: 중복 분할과 채점 셀 또는
    A 셀이 없는 분할은 빼고, 유효 분할이 하나라도 있으면 유효 분할만 센다."""
    uniq = [int(sp) for sp, v in info.items() if v.get("dup_of", -1) < 0 and v.get("n_eval", 1) > 0 and v.get("n_A", 1) > 0]
    val = [sp for sp in uniq if info[sp].get("valid", True)]
    return sorted(val if val else uniq)


def n_expected(tm):
    """대상의 기대 분할 수. 분할 구조를 모르면(시험의 합성 저장소) 저장소의 사용 분할 수다."""
    ex = getattr(tm, "expected_splits", None)
    return int(len(ex)) if ex is not None else int(len(tm.used))


def sub_tm(tm, splits):
    """분할 부분집합만 쓰는 TMx 사본."""
    t2 = copy.copy(tm)
    keep = set(int(s_) for s_ in splits)
    t2.by_all = {sp: st for sp, st in tm.by_all.items() if sp in keep}
    t2.by_valid = {sp: st for sp, st in tm.by_valid.items() if sp in keep}
    t2.used = {sp: st for sp, st in tm.used.items() if sp in keep}
    t2.idx = {sp: g for sp, g in tm.idx.items() if sp in keep}
    t2.nb_union = len(set().union(*[set(st.blocks.tolist()) for st in t2.used.values()])) if t2.used else 0
    t2.n_valid = len(t2.by_valid); t2.n_unique = len(t2.by_all)
    t2.has_ci = bool(tm.has_ci and len(t2.used) > 0)
    t2.nboot = tm.nboot if t2.has_ci else 0
    ex = getattr(tm, "expected_splits", None)
    if ex is not None:
        t2.expected_splits = [sp for sp in ex if sp in keep]
    return t2


def curve_view(tm):
    """곡선 표에 쓰는 TMx 사본(거리 층과 구간 키를 뺀다)."""
    t2 = copy.copy(tm)
    t2.idx = {sp: {g: ks for g, ks in gd.items() if not is_aux_method(g[0])} for sp, gd in tm.idx.items()}
    return t2


def _pair_keys(tm, sp, gA, gB):
    """분할 하나에서 두 쪽의 키. 추출이 여럿인 두 쪽은 공통 추출 번호로 제한한다(h40.contrast 와 같은 규칙)."""
    a_, b_ = tm.idx[sp].get(gA), tm.idx[sp].get(gB)
    if not a_ or not b_:
        return None
    da, db = {k[5] for k in a_}, {k[5] for k in b_}
    if len(da) > 1 and len(db) > 1:
        cm = da & db
        a_ = [k for k in a_ if k[5] in cm]; b_ = [k for k in b_ if k[5] in cm]
        if not a_ or not b_:
            return None
    return a_, b_


def grp_rmse(tm, g):
    """곡선 키의 셀 가중 RMSE(추출·seed 평균 뒤 분할 평균)."""
    v = []
    for sp, st in tm.used.items():
        ks = tm.idx[sp].get(g)
        if not ks:
            continue
        S, C = st.matrices(ks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v.append(float(np.nanmean(H4._rmse_rows(S, C))))
    return float(np.mean(v)) if v else np.nan


def boot_delta_common(tm, gA, gB, nboot=None):
    """보조 CI(지역 블록 공통 재표집). 사용 분할 채점 블록의 합집합 U 에 대해 재표집 다중도 W 를 한 번 뽑고, 분할마다 자기 채점 블록의 열을 읽는다.
    분할별 Δ 분포는 boot_delta_blocks 와 같은 식이고 분할 평균은 nanmean 이다(채점 블록의 다중도가 모두 0 인 반복은 그 분할만 빠진다)."""
    nboot = int(tm.nboot if nboot is None else nboot)
    if not tm.used:
        return None
    U = sorted(set().union(*[set(st.blocks.tolist()) for st in tm.used.values()]))
    pos = {b: i for i, b in enumerate(U)}
    W = H4.boot_weights(len(U), nboot, seed_of("lgxboot", tm.name)) if nboot > 0 else None
    dc, db, pt, ptb = [], [], [], []
    for sp, st in sorted(tm.used.items()):
        pk = _pair_keys(tm, sp, gA, gB)
        if pk is None:
            continue
        SA, CA = st.matrices(pk[0]); SB, CB = st.matrices(pk[1])
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            pt.append(float(np.nanmean(H4._rmse_rows(SA, CA)) - np.nanmean(H4._rmse_rows(SB, CB))))
            ptb.append(float(np.nanmean(H4._beq_rows(SA, CA)) - np.nanmean(H4._beq_rows(SB, CB))))
            if W is not None:
                Ws = W[:, [pos[b] for b in st.blocks.tolist()]]
                bA = np.sqrt((SA @ Ws.T) / (CA @ Ws.T)); bB = np.sqrt((SB @ Ws.T) / (CB @ Ws.T))
                dc.append(np.nanmean(bA, 0) - np.nanmean(bB, 0))
                vA = np.where(CA > 0, np.sqrt(SA / np.where(CA > 0, CA, 1)), 0.0); mA = (CA > 0).astype(float)
                vB = np.where(CB > 0, np.sqrt(SB / np.where(CB > 0, CB, 1)), 0.0); mB = (CB > 0).astype(float)
                db.append(np.nanmean((vA @ Ws.T) / (mA @ Ws.T), 0) - np.nanmean((vB @ Ws.T) / (mB @ Ws.T), 0))
    if not pt:
        return None
    out = dict(delta=float(np.mean(pt)), delta_beq=float(np.mean(ptb)), n_splits=len(pt), n_blocks_union=len(U))
    if dc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist, distb = np.nanmean(np.vstack(dc), 0), np.nanmean(np.vstack(db), 0)
        out.update(dist=dist, dist_beq=distb)
    return out


def boot_mean_blocks(tm, g, nboot=None):
    """비율형 통계 Σ(W·S)/Σ(W·C)의 점 추정과 블록 부트스트랩 CI(구간의 커버리지와 폭). 분할 안 채점 블록 재표집, 분할 분포 평균."""
    nboot = int(tm.nboot if nboot is None else nboot)
    pts, dists = [], []
    for sp, st in sorted(tm.used.items()):
        ks = tm.idx[sp].get(g)
        if not ks:
            continue
        S, C = st.matrices(ks)
        with np.errstate(invalid="ignore", divide="ignore"):
            pts.append(float(np.nanmean(S.sum(1) / C.sum(1))))
            if nboot > 0:
                W = H4.boot_weights(st.nb, nboot, seed_of(tm.seed, sp))
                dists.append(np.nanmean((S @ W.T) / (C @ W.T), 0))
    if not pts:
        return None
    out = dict(value=float(np.mean(pts)), n_splits=len(pts), ci_lo=np.nan, ci_hi=np.nan)
    if dists:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.nanmean(np.vstack(dists), 0)
        out.update(ci_lo=float(np.nanpercentile(d, 2.5)), ci_hi=float(np.nanpercentile(d, 97.5)))
    return out


def _ci(d):
    if d is None:
        return np.nan, np.nan
    d = np.asarray(d, float)
    if not np.isfinite(d).any():
        return np.nan, np.nan
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def region_stats(tm, gA, gB):
    """지역 하나의 대비: 점 추정, 주 분포(h40.contrast), 공통 재표집 분포. CI 풀 조건은 h40.strat_mean 과 같다."""
    r = H.contrast(tm, gA, gB, return_dist=True)
    if r is None or not np.isfinite(r["delta"]):
        return None
    ok = bool(tm.has_ci and tm.nb_union >= MS.MIN_BLOCKS_CI and "dist" in r)
    rc = boot_delta_common(tm, gA, gB) if ok else None
    return dict(delta=float(r["delta"]), delta_beq=float(r["delta_beq"]), rmse_A=float(r["rmse_A"]), rmse_B=float(r["rmse_B"]),
                n_splits=int(r["n_splits"]), n_splits_expected=n_expected(tm), n_blocks_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0, ok=ok,
                dist=r["dist"] if ok else None, dist_beq=r["dist_beq"] if ok else None,
                cdist=rc.get("dist") if rc else None, cdist_beq=rc.get("dist_beq") if rc else None)


def region_dd(tm, p1, p2):
    """이중 차분 [p1 의 (A − B)] − [p2 의 (A − B)]. 같은 TMx 의 두 대비 분포를 같은 번호끼리 뺀다."""
    a_, b_ = region_stats(tm, *p1), region_stats(tm, *p2)
    if a_ is None or b_ is None:
        return None
    ok = bool(a_["ok"] and b_["ok"])

    def sub(k):
        return (a_[k] - b_[k]) if (ok and a_[k] is not None and b_[k] is not None and len(a_[k]) == len(b_[k])) else None
    return dict(delta=a_["delta"] - b_["delta"], delta_beq=a_["delta_beq"] - b_["delta_beq"], rmse_A=a_["delta"], rmse_B=b_["delta"],
                n_splits=min(a_["n_splits"], b_["n_splits"]), n_splits_expected=max(a_["n_splits_expected"], b_["n_splits_expected"]),
                n_blocks_min=min(a_["n_blocks_min"], b_["n_blocks_min"]), ok=ok,
                dist=sub("dist"), dist_beq=sub("dist_beq"), cdist=sub("cdist"), cdist_beq=sub("cdist_beq"))


def stats_row(s_, a, target, extra=None):
    """대비 통계 → 표의 한 행(주 CI, 공통 재표집 CI, 4분 판정, 동등성 p)."""
    lo, hi = _ci(s_.get("dist")); lob, hib = _ci(s_.get("dist_beq"))
    clo, chi = _ci(s_.get("cdist")); clob, chib = _ci(s_.get("cdist_beq"))
    pe = [eq_p(s_[k], a.delta_eq) for k in ("dist", "dist_beq") if s_.get(k) is not None]
    row = dict(target=target, delta=s_["delta"], ci_lo=lo, ci_hi=hi, delta_blockeq=s_["delta_beq"], ci_lo_beq=lob, ci_hi_beq=hib,
               p_boot=H4.boot_p(s_["dist"]) if s_.get("dist") is not None else np.nan, p_eq=float(np.nanmax(pe)) if pe else np.nan, rmse_A=s_.get("rmse_A", np.nan),
               rmse_B=s_.get("rmse_B", np.nan), ci_lo_c=clo, ci_hi_c=chi, ci_lo_beq_c=clob, ci_hi_beq_c=chib,
               verdict4=verdict4(lo, hi, lob, hib, a.delta_eq), verdict4_d10=verdict4(lo, hi, lob, hib, a.delta_eq_aux),
               verdict4_common=verdict4(clo, chi, clob, chib, a.delta_eq))
    row.update(extra or {})
    return row


def _short_splits(per, use):
    """분할이 기대보다 적은 지역의 (분할 수, 기대 분할 수, 이름) 목록. 비율이 작은 순이다."""
    out = [(int(per[nm]["n_splits"]), int(per[nm].get("n_splits_expected", per[nm]["n_splits"])), nm) for nm in use]
    return sorted([q for q in out if q[0] < q[1]], key=lambda q: (q[0] / max(q[1], 1), q[2]))


def pool_rows(per, names, a, label):
    """지역별 통계 dict → 지역 행과 층화 평균 행. 풀 = 분포가 있는 지역(없으면 점 추정만). 풀 지역 수 2 미만은 판정 불가.
    층화 평균 행에는 풀 지역 이름(pool_regions)과 분할 완결성(splits_min, splits_expected, splits_short)을 적는다."""
    have = [nm for nm in names if nm in per]
    if not have:
        return []
    rows = [stats_row(per[nm], a, nm, dict(scope="region", n_splits=per[nm]["n_splits"],
                                           n_splits_expected=per[nm].get("n_splits_expected", per[nm]["n_splits"]),
                                           n_blocks_split_min=per[nm]["n_blocks_min"])) for nm in have]
    pool = [nm for nm in have if per[nm]["dist"] is not None]
    use = pool if pool else have
    short = _short_splits(per, use)
    m = dict(delta=float(np.mean([per[nm]["delta"] for nm in use])), delta_beq=float(np.mean([per[nm]["delta_beq"] for nm in use])),
             rmse_A=float(np.mean([per[nm]["rmse_A"] for nm in use])), rmse_B=float(np.mean([per[nm]["rmse_B"] for nm in use])),
             dist=MS.strat([per[nm]["dist"] for nm in pool]) if pool else None,
             dist_beq=MS.strat([per[nm]["dist_beq"] for nm in pool]) if pool else None,
             cdist=MS.strat([per[nm]["cdist"] for nm in pool if per[nm]["cdist"] is not None]) if pool else None,
             cdist_beq=MS.strat([per[nm]["cdist_beq"] for nm in pool if per[nm]["cdist_beq"] is not None]) if pool else None)
    neg = sum(per[nm]["delta"] < 0 for nm in have)
    mr = stats_row(m, a, f"MEAN[{','.join(use)}]", dict(scope="MEAN", n_ci_regions=len(pool), n_regions_target=len(names),
                                                       ci_flag=f"neg {neg}/{len(have)}; ci_regions {len(pool)}",
                                                       delta_allregions=float(np.mean([per[nm]["delta"] for nm in have])),
                                                       pool_regions=",".join(pool),
                                                       splits_min=short[0][0] if short else int(min(per[nm]["n_splits"] for nm in use)),
                                                       splits_expected=short[0][1] if short else int(max(
                                                           per[nm].get("n_splits_expected", per[nm]["n_splits"]) for nm in use)),
                                                       splits_short="; ".join(f"{nm} {k}/{K}" for k, K, nm in short)))
    lo, hi = mr["ci_lo"], mr["ci_hi"]
    bad = len(pool) < H.MIN_POOL_REGIONS or not np.isfinite(lo) or (mr["delta"] < lo - 1e-9 or mr["delta"] > hi + 1e-9)
    if bad:                                                              # 판정 불가: 풀 지역 수 미달, CI 비유한, 점 추정치가 백분위 CI 밖
        why = "pool<2" if len(pool) < H.MIN_POOL_REGIONS else ("ci nonfinite" if not np.isfinite(lo) else "assert: point outside CI")
        mr.update(verdict4="판정 불가", verdict4_d10="판정 불가", verdict4_common="판정 불가", ci_flag=f"{why}; {mr['ci_flag']}")
    mr["undetermined"] = bool(bad)
    return rows + [mr]


def floor_table(df):
    """N2 오차 하한. 거친 공변량 묶음(기후 8, 토양 9, CCI 2열을 소수 6자리로 반올림한 값이 같은 셀) 가운데 5셀 이상 묶음 안의
    합동 분산(ddof 1)으로 지역별 하한 RMSE 를 구한다. 채점 셀(eval_mask) 기준이 주 값이고 전체 셀 기준을 병기한다."""
    X = df[FLOOR_COLS].values.astype(float)
    key = pd.Series([tuple(v) for v in np.where(np.isnan(X), -9999.0, np.round(X, 6))])
    gid = pd.factorize(key)[0]
    ev = eval_mask(df)
    y = df.y.values.astype(float)
    rows = []
    for reg in FLOOR_REGIONS:
        for scope, m in (("eval", ev & (df.macro.values == reg)), ("all", (df.macro.values == reg) & np.isfinite(y))):
            d = pd.DataFrame(dict(g=gid[m], y=y[m]))
            ag = d.groupby("g").y.agg(["size", "var"])
            ag = ag[ag["size"] >= FLOOR_MIN_CELLS]
            dfree = float((ag["size"] - 1).sum())
            var = float(((ag["size"] - 1) * ag["var"]).sum() / dfree) if dfree > 0 else np.nan
            rows.append(dict(region=reg, scope=scope, n_cells=int(m.sum()), n_bundles=int(d.g.nunique()), n_bundles_used=int(len(ag)),
                             n_cells_used=int(ag["size"].sum()) if len(ag) else 0, floor_rmse_cm=float(np.sqrt(var)) if np.isfinite(var) else np.nan))
    meta = dict(n_bundles_total=int(len(np.unique(gid))), n_bundles_ge5=int((np.bincount(gid) >= FLOOR_MIN_CELLS).sum()), columns=list(FLOOR_COLS),
                rule="소수 6자리 반올림, 결측은 표지값 −9999, 5셀 이상 묶음의 합동 분산(ddof 1)")
    return pd.DataFrame(rows), meta


def floor_of(floor, D, tm):
    """대상의 하한 RMSE(상위 macro 지역의 채점 셀 기준 값). 하한이 없는 지역은 NaN."""
    if floor is None or not len(floor):
        return np.nan
    try:
        par = D.parent_of(getattr(tm, "base_target", tm.target))
    except Exception:                                                     # noqa: BLE001
        return np.nan
    q = floor[(floor.region == par) & (floor.scope == "eval")]
    return float(q.floor_rmse_cm.iloc[0]) if len(q) else np.nan


def effect_cols(delta, rmse_p0, fl):
    """효과 크기 표기: cm, P0 RMSE 대비 %, 환원 가능 오차 대비 비율(개선이 양수), 0.5 cm 미만 표시."""
    out = dict(delta_cm=float(delta) if np.isfinite(delta) else np.nan, delta_pct_p0=np.nan, frac_reducible=np.nan,
               small_effect=bool(np.isfinite(delta) and abs(delta) < SMALL_EFFECT_CM))
    if np.isfinite(delta) and np.isfinite(rmse_p0) and rmse_p0 > 0:
        out["delta_pct_p0"] = float(100.0 * delta / rmse_p0)
        if np.isfinite(fl) and rmse_p0 - fl > 0:
            out["frac_reducible"] = float(-delta / (rmse_p0 - fl))
    return out


class TestBook:
    """가설 판정 표의 행을 쌓는다. 대비 행(지역, 층화 평균)과 판정 행(verdict)을 함께 둔다.
    판정 행의 문구에는 판정에 쓴 대비(used)에서 구한 부분 표기(풀 지역 수, 분할 완결성)와 효과 크기 표기를 붙인다(계획서 §6A.2, 개정 7)."""

    def __init__(self, a, tms, D, floor):
        self.a, self.tms, self.D, self.floor = a, tms, D, floor
        self.rows: list = []
        self.m4 = [f"{t}|x" for t in MAIN4]
        self.m3 = ["Lena|x", "Canada|x", f"{H.ALASKA}|x"]
        self._p0: dict = {}

    def p0(self, nm):
        if nm not in self._p0:
            tm = self.tms.get(nm)
            self._p0[nm] = (grp_rmse(tm, H.P0_GRP), floor_of(self.floor, self.D, tm)) if tm is not None else (np.nan, np.nan)
        return self._p0[nm]

    def _missing(self, test, item, label, scope, role, blind, kw):
        """키가 없는 대비의 자리 표시 행."""
        r = dict(kw)
        r.update(test_id=test, item=item, contrast=label, scope=scope, role=role, primary=False, blind=bool(blind), verdict4="행 없음",
                 note=";".join(v for v in (str(kw.get("note", "")), "키가 없다") if v))
        self.rows.append(r)

    def _emit(self, rows, test, item, label, names, primary, blind, role, **kw):
        out = None
        for r in rows:
            if r["scope"] == "region":
                rp, fl = self.p0(r["target"])
            else:
                use = [nm for nm in names if nm in self.tms]
                rp = float(np.nanmean([self.p0(nm)[0] for nm in use])) if use else np.nan
                fl = np.nan
                fr = [effect_cols(q["delta"], *self.p0(q["target"]))["frac_reducible"] for q in rows if q["scope"] == "region"]
                r["frac_reducible_mean"] = float(np.nanmean(fr)) if np.isfinite(fr).any() else np.nan
                r["n_regions_floor"] = int(np.isfinite(fr).sum())           # 환원 가능 오차 비율의 평균에 들어간 지역 수(하한이 있는 지역만)
                r["pool"] = f"지역 {int(r.get('n_ci_regions', 0))}/{len(names)}"
            r.update(effect_cols(r["delta"], rp, fl))
            r["small_note"] = SMALL_EFFECT_TXT if (str(r.get("verdict4", "")) in ("우세", "열세") and r.get("small_effect")) else ""
            r.update(test_id=test, item=item, contrast=label, role=role, primary=bool(primary and r["scope"] != "region"), blind=bool(blind), **kw)
            self.rows.append(r)
            if r["scope"] != "region":
                out = r
        return out

    def contrast(self, test, item, label, gA, gB, names=None, primary=True, blind=True, role="주", aux3=True, **kw):
        """층화 평균 대비. 주 CI 와 판정 불가 조건은 h40.strat_mean 의 값을 그대로 쓰고, 공통 재표집 CI 와 동등성 p 는 지역별 분포에서 다시 구한다."""
        names = self.m4 if names is None else list(names)
        fa = gA if callable(gA) else (lambda tm: gA)
        fb = gB if callable(gB) else (lambda tm: gB)
        per = {}
        for nm in names:
            tm = self.tms.get(nm)
            if tm is None:
                continue
            s_ = region_stats(tm, fa(tm), fb(tm))
            if s_ is not None:
                per[nm] = s_
        rows = pool_rows(per, names, self.a, label)
        if not rows:
            self._missing(test, item, label, "MEAN", role, blind, kw)
            return None
        ref = [r for r in H.strat_mean(label, self.tms, names, fa, fb) if str(r.get("target", "")).startswith("MEAN[")]
        mr = rows[-1]
        if ref:                                                          # 주 CI 와 판정 불가 조건은 h40.strat_mean 의 값
            q = ref[0]
            mr.update(delta=q["delta"], ci_lo=q["ci_lo"], ci_hi=q["ci_hi"], delta_blockeq=q["delta_blockeq"], ci_lo_beq=q["ci_lo_beq"],
                      ci_hi_beq=q["ci_hi_beq"], p_boot=q.get("p_boot", np.nan), n_ci_regions=int(q.get("n_ci_regions", 0)),
                      ci_flag=str(q.get("ci_flag", "")))
            if H.mean_state(q) == "undetermined":
                mr.update(verdict4="판정 불가", verdict4_d10="판정 불가", undetermined=True)
            else:
                mr.update(verdict4=verdict4(q["ci_lo"], q["ci_hi"], q["ci_lo_beq"], q["ci_hi_beq"], self.a.delta_eq),
                          verdict4_d10=verdict4(q["ci_lo"], q["ci_hi"], q["ci_lo_beq"], q["ci_hi_beq"], self.a.delta_eq_aux), undetermined=False)
        out = self._emit(rows, test, item, label, names, primary, blind, role, **kw)
        if aux3 and names == self.m4:                                    # 보조: 레나·캐나다·Alaska(x) 3지역 평균
            per3 = {nm: per[nm] for nm in self.m3 if nm in per}
            tm = self.tms.get(self.m3[2])
            if tm is not None:
                s_ = region_stats(tm, fa(tm), fb(tm))
                if s_ is not None:
                    per3[self.m3[2]] = s_
            r3 = [r for r in pool_rows(per3, self.m3, self.a, label) if r["scope"] != "region" or r["target"] == self.m3[2]]
            for r in r3:
                if r["scope"] != "region":
                    r["scope"] = "MEAN3"
            self._emit(r3, test, item, label, self.m3, False, blind, "보조(3지역)", **kw)
        return out

    def dd(self, test, item, label, p1, p2, names=None, primary=True, blind=True, role="주", **kw):
        """이중 차분의 층화 평균. p1, p2 = (gA, gB)."""
        names = self.m4 if names is None else list(names)
        per = {}
        for nm in names:
            tm = self.tms.get(nm)
            if tm is None:
                continue
            s_ = region_dd(tm, p1, p2)
            if s_ is not None:
                per[nm] = s_
        rows = pool_rows(per, names, self.a, label)
        if not rows:
            self._missing(test, item, label, "MEAN", role, blind, kw)
            return None
        return self._emit(rows, test, item, label, names, primary, blind, role, kind="이중 차분", **kw)

    def single(self, test, item, label, name, stats, primary=False, blind=True, role="주", **kw):
        """대상 하나의 대비(층화 평균 없음)."""
        if stats is None:
            self._missing(test, item, label, "region", role, blind, dict(kw, target=name))
            return None
        r = stats_row(stats, self.a, name, dict(scope="region", n_splits=stats["n_splits"],
                                                n_splits_expected=stats.get("n_splits_expected", stats["n_splits"]),
                                                n_blocks_split_min=stats["n_blocks_min"]))
        rp, fl = self.p0(name)
        r.update(effect_cols(r["delta"], rp, fl))
        r["small_note"] = SMALL_EFFECT_TXT if (str(r.get("verdict4", "")) in ("우세", "열세") and r.get("small_effect")) else ""
        r.update(test_id=test, item=item, contrast=label, role=role, primary=bool(primary), blind=bool(blind), **kw)
        self.rows.append(r)
        return r

    @staticmethod
    def marks(used, point=False):
        """판정에 쓴 대비 행들의 부분 표기와 효과 크기 표기. used = [(이름, 행 또는 None)].
        반환 dict(pool = (k, N) 또는 None, splits = (k, K) 또는 None, small = 이름 목록). 4분 판정이 나온 행만 본다.
        point = True(점 추정으로 판정하는 가설)이면 행이 있는 대비를 모두 보고 분할 완결성만 구한다."""
        rows = [(k, r) for k, r in (used or []) if (r is not None if point else valid_row(r))]
        pools = [] if point else [(int(r["n_ci_regions"]), int(r["n_regions_target"])) for _, r in rows
                                  if r.get("scope") == "MEAN" and r.get("n_regions_target") and int(r["n_ci_regions"]) < int(r["n_regions_target"])]
        sp = []
        for _, r in rows:                                                 # 층화 평균 행은 splits_min·splits_expected, 지역 행은 n_splits·n_splits_expected
            k_ = r.get("splits_min") if r.get("splits_min") is not None else r.get("n_splits")
            K_ = r.get("splits_expected") if r.get("splits_expected") is not None else r.get("n_splits_expected")
            if k_ is not None and K_ is not None and np.isfinite(float(k_)) and np.isfinite(float(K_)) and int(k_) < int(K_):
                sp.append((int(k_), int(K_)))
        small = [] if point else [str(k) for k, r in rows if r["verdict4"] in ("우세", "열세") and np.isfinite(float(r.get("delta", np.nan)))
                                  and abs(float(r["delta"])) < SMALL_EFFECT_CM]
        return dict(pool=min(pools, key=lambda q: q[0] / q[1]) if pools else None,
                    splits=min(sp, key=lambda q: q[0] / q[1]) if sp else None, small=small)

    def verdict(self, test, item, text, stat="", role="주", blind=True, used=None, partial=None, point=False, **kw):
        """판정 행. used 를 주면 문구에 부분 표기와 효과 크기 표기를 붙인다(판정 불가 문구에는 붙이지 않는다).
        partial = 호출한 쪽이 더하는 부분 표기(예: '대비 5/6'). point = True 는 점 추정으로 판정하는 가설이다(분할 완결성만 본다).
        확인적 가설은 판정에 쓴 대비의 분할이 기대보다 적으면 판정하지 않는다(계획서 §6A.2, 개정 7)."""
        text = str(text)
        conf = test in CONFIRMATORY
        mk = self.marks(used, point)
        pre = [str(v) for v in (partial or []) if v]
        if not text.startswith("판정 불가"):
            if mk["splits"] is not None and conf:
                text = f"판정 불가(분할 {mk['splits'][0]}/{mk['splits'][1]}. 등록한 분할이 모두 채워진 뒤 판정한다)"
            else:
                if mk["pool"] is not None:
                    pre.append(f"지역 {mk['pool'][0]}/{mk['pool'][1]}")
                if mk["splits"] is not None:
                    pre.append(f"분할 {mk['splits'][0]}/{mk['splits'][1]}")
                if pre:
                    text = f"부분({', '.join(pre)}): {text}"
                if mk["small"]:
                    text = f"{text} [{SMALL_EFFECT_TXT}: {', '.join(mk['small'])}]"
        self.rows.append(dict(test_id=test, item=item, scope="verdict" if role == "주" else "verdict_aux", verdict=text, stat=str(stat), role=role,
                              primary=False, blind=bool(blind), n_used=len(used or []),
                              n_used_valid=int(sum((r is not None) if point else valid_row(r) for _, r in (used or []))), **kw))

    def frame(self):
        df = pd.DataFrame(self.rows)
        if not len(df):
            return df
        df["holm_p"] = np.nan
        df["holm_note"] = ""
        df["holm_p_eq"] = np.nan
        df["holm_note_eq"] = ""
        for c_ in ("p_boot", "p_eq", "verdict4", "primary", "eq_test"):
            if c_ not in df:
                df[c_] = np.nan
        prim = df.primary.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
        judged = df.verdict4.notna() & ~df.verdict4.isin(list(NA_VERDICTS))  # 판정 불가와 행 없음은 묶음에 넣지 않는다(묶음 크기를 키우지 않는다)
        eqt = df.eq_test.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
        for item, g in df[prim & judged & df.p_boot.notna()].groupby("item"):   # 항목별 주 대비 묶음의 Holm 보정(양측 부트스트랩 p)
            hp = MS.holm(g.p_boot.values.astype(float))
            df.loc[g.index, "holm_p"] = hp
            sig = g.verdict4.isin(["우세", "열세"]).values & (hp >= 0.05)
            df.loc[g.index[sig], "holm_note"] = "보정 전 유의"
        for item, g in df[prim & judged & eqt & df.p_eq.notna()].groupby("item"):   # 동등성 대비 묶음의 Holm 보정(두 한쪽 p 의 최댓값)
            hq = MS.holm(g.p_eq.values.astype(float))
            df.loc[g.index, "holm_p_eq"] = hq
            weak = (g.verdict4 == "동등").values & (hq > EQ_ALPHA)
            df.loc[g.index[weak], "holm_note_eq"] = "보정 전 동등"
        if "verdict4_common" in df:
            m = df.verdict4.notna() & df.verdict4_common.notna() & (df.verdict4 != df.verdict4_common) & df.scope.isin(["MEAN", "MEAN3", "region"]) \
                & ~df.verdict4.isin(["행 없음"])
            df["ci_dependence"] = np.where(m, "분할 독립 가정 의존", "")
        df["confirmatory"] = df.test_id.isin(list(CONFIRMATORY))
        front = ["test_id", "item", "contrast", "scope", "role", "confirmatory", "primary", "blind", "target", "n", "lam", "delta", "ci_lo", "ci_hi",
                 "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "p_boot", "p_eq", "verdict4", "verdict4_common", "verdict4_d10", "small_note", "holm_p",
                 "holm_p_eq", "n_ci_regions", "pool", "splits_min", "splits_expected", "verdict", "stat"]
        return df[[c_ for c_ in front if c_ in df] + [c_ for c_ in df.columns if c_ not in front]]


def valid_row(r):
    """4분 판정(우세, 열세, 동등, 미결정)이 나온 대비 행인지."""
    return r is not None and str(r.get("verdict4", "")) not in NA_VERDICTS and str(r.get("verdict4", "")) != ""


def _fmt(res):
    return "; ".join(f"{k}: {('행 없음' if r is None else r['verdict4'])}" + ("" if r is None else f"(Δ {r['delta']:.2f}, {r.get('pool', '')}"
                                                                                                  + (f", {r['small_note']}" if r.get("small_note") else "")
                                                                                                  + (f", 분할 {r['splits_short']}" if r.get("splits_short") else "")
                                                                                                  + ")")
                     for k, r in res)


def _v(r):
    return "행 없음" if r is None else str(r["verdict4"])


def count_valid(res):
    """(판정 가능한 대비 수, 등록한 대비 수, 판정할 수 없는 대비의 이름)."""
    miss = [str(k) for k, r in res if not valid_row(r)]
    return len(res) - len(miss), len(res), miss


def na_text(res, what="대비"):
    k, m, miss = count_valid(res)
    return f"판정 불가({what} {k}/{m}; 없는 {what}: {', '.join(miss)})"


def part_mark(res, what="대비"):
    """판정할 수 없는 대비가 있을 때의 부분 표기(없으면 빈 목록)."""
    k, m, miss = count_valid(res)
    return [f"{what} {k}/{m}(없는 {what}: {', '.join(miss)})"] if k < m else []


def tri_count(states, need):
    """세 값(True, False, None = 알 수 없음)의 집계 판정. True 가 need 개 이상이면 'yes', 알 수 없는 것을 모두 True 로 쳐도 need 에
    못 미치면 'no', 그 밖은 'unknown'(남은 항목에 따라 결과가 바뀔 수 있다)."""
    yes = sum(v is True for v in states); unk = sum(v is None for v in states)
    if yes >= need:
        return "yes"
    if yes + unk < need:
        return "no"
    return "unknown"


# ================================================================ 집계: 가설 판정(L9–L18, L23–L31)
def build_tests_x(a, tms, D, floor, runs, conf=None, n4=None):
    """L9–L18, L23–L31 의 대비 행과 판정 행(계획서 §6A.5 와 §6A.2 의 개정 7 규칙).
    - 판정할 수 없는 대비(행 없음, 판정 불가)가 있을 때: 확인적 가설은 지지 문구를 쓰지 않는다. 기각 조건이 판정한 대비만으로 이미 충족되면
      '기각(일부 대비 판정 불가, 대비 k/m)', 아니면 '판정 불가(대비 k/m; 없는 대비: …)'다. '우세가 없으면 지지' 형식의 보조 가설(L13 보조,
      L18, L28 앵커 교체)은 '부분(대비 k/m(없는 대비: …))'을 붙인다. 그 밖의 보조 가설은 판정 불가다.
    - 풀 지역 수가 대상 지역 수보다 적은 층화 평균을 쓴 판정은 '부분(지역 k/N)', 분할이 기대보다 적은 대비를 쓴 판정은 '부분(분할 k/K)'를
      붙인다. 확인적 가설은 분할이 기대보다 적으면 판정 불가다.
    - 세는 형식의 가설(L24, L25, L27, L31)은 알 수 없는 항목이 있어도 기준을 이미 넘겼거나 넘길 수 없을 때만 판정하고(부분 표기),
      남은 항목에 따라 결과가 바뀔 수 있으면 판정 불가다(tri_count)."""
    T = TestBook(a, tms, D, floor)
    V = T.verdict
    N_ALL = -1
    memo: dict = {}

    def C(test, item, label, gA, gB, **kw):
        """층화 평균 대비. 같은 가설의 같은 이름·같은 지역 목록의 대비는 한 번만 계산한다."""
        k = (test, label, tuple(kw.get("names") or ()))
        if k not in memo:
            memo[k] = T.contrast(test, item, label, gA, gB, **kw)
        return memo[k]

    def rej(k, m):
        return "기각" if k == m else f"기각(일부 대비 판정 불가, 대비 {k}/{m})"

    # ---------------- L9 중복성 점검(보조)
    res = [(f"n={n}", C("L9", "X1", f"F1a-D0|n{n}", G("F1a", n), G("D0", n), n=n, eq_test=True)) for n in (0, 10, N_ALL)]
    k, m, _ = count_valid(res)
    ok = k == m and all(r["verdict4"] == "동등" and abs(r["delta"]) <= L9_POINT_MAX for _, r in res)
    V("L9", "X1", na_text(res) if k < m else ("지지: √TDD 가 이미 입력에 있으므로 상수 계수 Stefan 출력의 추가는 정보를 더하지 않는다" if ok
                                                else "미지지: 구현을 점검하고 표만 싣는다"), _fmt(res), role="보조", used=res)

    # ---------------- L10 물리 입력 대 잔차(확인적)
    r0 = [("n=0 R0-F1k", C("L10", "X1", "R0-F1k|n0", G("R0", 0), G("F1k", 0), n=0)),
          ("n=0 R0-F1a", C("L10", "X1", "R0-F1a|n0", G("R0", 0), G("F1a", 0), n=0))]
    r10 = [("n=10 R1-F1k", C("L10", "X1", "R1-F1k|n10", G("R1", 10), G("F1k", 10), n=10)),
           ("n=10 R1-F1n", C("L10", "X1", "R1-F1n|n10", G("R1", 10), G("F1n", 10), n=10))]
    for n in (3, 40, 160, N_ALL):
        for m_ in ("F1k", "F1n"):
            C("L10", "X1", f"R1-{m_}|n{n}", G("R1", n), G(m_, n), n=n, primary=False, role="보조")
    four = r0 + r10
    st0, st10 = [_v(r) for _, r in r0], [_v(r) for _, r in r10]
    allv = st0 + st10
    k, m, _ = count_valid(four)
    worse = [lab for (lab, _), v in zip(four, allv) if v == "열세"]
    if worse:
        txt = f"{rej(k, m)}: 열세인 대비가 있다({', '.join(worse)}). 해당 방법의 곡선을 마스터 곡선에 더한다"
    elif k < m:
        txt = na_text(four)
    elif all(v == "우세" for v in allv):
        txt = "지지: 잔차 구조는 물리 입력 구조보다 오차가 작다"
    elif all(v == "우세" for v in st0) and not any(v == "우세" for v in st10):
        txt = "부분 지지(n = 0)"
    elif all(v == "우세" for v in st10) and not any(v == "우세" for v in st0):
        txt = "부분 지지(n = 10)"
    elif not any(v == "우세" for v in allv):
        txt = "결합 방식 사이의 차이는 확인되지 않았다(덧셈 잔차를 방법 기여로 내세우지 않는다)"
    else:
        txt = f"혼재(우세 {sum(v == '우세' for v in allv)}/4, 등록한 규칙 밖의 조합): 대비별로 서술한다"
    V("L10", "X1", txt, _fmt(four), used=four)
    # 결측 대체 민감도(보조): F1k 의 두 입력 열을 학습 쪽 중앙값으로 채운 값으로 바꾼 F1k@tr
    sens = [("n=0 R0-F1k@tr", C("L10", "X1", "R0-F1k@tr|n0", G("R0", 0), G("F1k@tr", 0), n=0, primary=False, role="보조(결측 대체 민감도)",
                                aux3=False)),
            ("n=10 R1-F1k@tr", C("L10", "X1", "R1-F1k@tr|n10", G("R1", 10), G("F1k@tr", 10), n=10, primary=False, role="보조(결측 대체 민감도)",
                                 aux3=False))]
    for n in TR_FIT_N:
        C("L10", "X1", f"F1k@tr-F1k|n{n}", G("F1k@tr", n), G("F1k", n), n=n, primary=False, role="보조(결측 대체 민감도)", aux3=False)
    pair = [(r0[0], sens[0]), (r10[0], sens[1])]
    if any(not valid_row(b_[1]) or not valid_row(s_[1]) for b_, s_ in pair):
        txt = na_text([b_ for b_, _ in pair] + sens)
    else:
        diff = [f"{s_[0]} {_v(s_[1])}(기준 {_v(b_[1])})" for b_, s_ in pair if _v(b_[1]) != _v(s_[1])]
        txt = "결측 대체 통계를 학습 쪽 중앙값으로 바꿔도 4분 판정이 같다" if not diff else "결측 대체 통계에 따라 4분 판정이 달라진다: " + ", ".join(diff)
    V("L10", "X1", txt, _fmt(sens), role="보조", used=sens, note="결측 대체 민감도: p4_ku, p2_edaphic 의 공변량 결측을 원천 셀 중앙값으로 채운 값")

    # ---------------- L11 라벨 0 물리 입력(보조)
    r = C("L11", "X1", "F1k-P0|n0", G("F1k", 0), G("P0", 0), n=0, blind=False, note="L1 관련(비맹검)")
    V("L11", "X1", na_text([("F1k-P0", r)]) if not valid_row(r) else ("기각: F1k 가 n = 0 에서 P0 보다 우세" if _v(r) == "우세" else "지지"),
      _fmt([("F1k-P0", r)]), role="보조", blind=False, used=[("F1k-P0", r)], note="L1 관련(비맹검)")

    # ---------------- L12 덧셈 대 곱셈(확인적)
    res = [(f"n={n} λ={lam}", C("L12", "X1", f"RM-R1|n{n}|lam{lam}", G("RM", n, lam), G("R1", n, lam), n=n, lam=lam, eq_test=True))
           for n in (10, N_ALL) for lam in (LAM_BASE, 1.0)]
    for n in (10, N_ALL):
        C("L12", "X1", f"RMc-R1|n{n}|lam1.0", G("RMc", n), G("R1", n, 1.0), n=n, lam=1.0, primary=False, role="보조")
    vs = [_v(r_) for _, r_ in res]
    k, m, _ = count_valid(res)
    diff = [f"{lab} RM {v}" for (lab, _), v in zip(res, vs) if v in ("우세", "열세")]
    if diff:
        txt = f"{rej(k, m)}: " + ", ".join(diff) + ". RMc 행을 함께 보고한다"
    elif k < m:
        txt = na_text(res)
    elif all(v == "동등" for v in vs):
        txt = "지지: 곱셈 보정과 덧셈 잔차는 동등하다(한계 0.5 cm, λ 0.25 와 1.0)"
    else:
        txt = "차이를 확인하지 못함"
    V("L12", "X1", txt, _fmt(res), used=res)

    # ---------------- L13 비선형 계수 예측(보조)
    ra = C("L13", "X1", "V2-P1|n10", G("V2", 10), G("P1", 10), n=10)
    rb = C("L13", "X1", "V2-R1|n10", G("V2", 10), G("R1", 10), n=10)
    two = [("V2-P1", ra), ("V2-R1", rb)]
    k, m, _ = count_valid(two)
    V("L13", "X1", na_text(two) if k < m else ("지지" if (_v(ra) == "우세" and _v(rb) in ("동등", "미결정")) else "미지지"), _fmt(two), role="보조",
      used=two)
    reg13 = [("Canada", "x"), ("CA-3", "i")]
    det, allok, used13, have13 = [], True, [], []
    for t, md in reg13:
        tm = tms.get(f"{t}|{md}")
        if tm is None:
            continue
        ns = sorted({g[4] for gd in tm.idx.values() for g in gd if g[0] == "V2" and (g[4] >= 40 or g[4] == -1)}, key=lambda v: (v == -1, v))
        for n in ns:
            for tag, tmx in (("분할 전부", tm), ("분할 3–5", sub_tm(tm, (3, 4, 5)))):
                s0, s1 = region_stats(tmx, G("V2", n), G("P0", 0)), region_stats(tmx, G("V2", n), G("P1", n))
                if s0 is None or s1 is None:
                    continue
                ok_ = bool(s0["delta"] <= 0 and s1["delta"] < 0)
                row = T.single("L13", "X1", f"V2-P0|n{n}|{tag}", f"{t}|{md}", s0, role="주" if tag == "분할 전부" else "보조", n=n, passed=ok_,
                               delta_vs_p1=s1["delta"], note=f"재현(비맹검), {tag}", blind=False)
                if tag == "분할 전부":
                    allok &= ok_; det.append(f"{t}|{md} n={n}:{'O' if ok_ else 'X'}"); used13.append((f"{t}|{md} n={n}", row))
                    if f"{t}|{md}" not in have13:
                        have13.append(f"{t}|{md}")
    miss13 = [f"{t}|{md}" for t, md in reg13 if f"{t}|{md}" not in have13]
    if not have13:
        txt = "판정 불가(행 없음)"
    elif not allok:
        txt = "미지지"
    else:
        txt = "지지" if not miss13 else "지지 조건 충족"
    V("L13", "X1", txt, "캐나다(x)·CA-3(i), L7 형식: " + "; ".join(det), role="보조", blind=False, note="재현(비맹검). 점 추정 기준", used=used13, point=True,
      partial=[f"대상 {len(have13)}/{len(reg13)}(없는 대상: {', '.join(miss13)})"] if (miss13 and have13) else None)

    # ---------------- L14 물리 관련 입력 제거(방향 중립)
    res = [(f"{m_} n={n}", C("L14", "X1", f"{m_}-D0|n{n}", G(m_, n), G("D0", n), n=n, eq_test=True)) for m_ in ("D0t", "D0c", "D0m") for n in (0, 10, N_ALL)]
    vs = [_v(r_) for _, r_ in res]
    dm = [v for (k_, _), v in zip(res, vs) if k_.startswith("D0m")]
    k, m, _ = count_valid(res)
    if k < m:
        txt = na_text(res)
    elif all(v == "동등" for v in vs):
        txt = "직접 ML 의 전이 오차는 물리 관련 입력의 유무에 의존하지 않는다"
    elif "열세" in dm:
        txt = "D0 는 물리 관련 입력을 가진 직접 ML 이다(D0m 열세). 방법 절에 적는다"
    else:
        txt = "혼재 또는 미결정: 표로 보고한다"
    V("L14", "X1", txt, _fmt(res), role="보조", used=res)

    # ---------------- L15 위약(확인적)
    res = {(p, n): C("L15", "X2", f"D1-D1@{p}|n{n}", G("D1", n), G(f"D1@{p}", n), n=n, blind=(n != 0),
                     note="재현(비맹검)" if n == 0 else "") for p in PLACEBOS for n in (0, 10)}
    vs = {k_: _v(r_) for k_, r_ in res.items()}
    lst = [(f"{p} n={n}", r_) for (p, n), r_ in res.items()]
    k, m, _ = count_valid(lst)
    if k < m:
        txt = na_text(lst)
    elif all(v == "우세" for v in vs.values()):
        txt = "증강 이득은 물리식의 셀 단위 정보에서 온다"
    elif all(vs[("const_src", n)] == "우세" for n in (0, 10)) and not any(vs[(p, n)] == "우세" for p in ("shuffle", "const_t") for n in (0, 10)):
        txt = "증강 이득은 대상 지역의 수준과 분포 정보에서 오며 셀 단위 물리 정보의 추가 기여는 확인되지 않았다"
    elif not any(v == "우세" for v in vs.values()):
        txt = "증강 이득은 유사라벨의 내용과 구분되지 않는다"
    else:
        txt = "혼재: n 별로 서술한다"
    V("L15", "X2", txt, _fmt(lst), used=lst, blind=False, note="n = 0 부분은 재현(비맹검)")
    for n in (0, 10, 40, 160):                                            # 대체 공급원(위약이 아니다. L15 판정에 넣지 않는다)
        C("L15", "X2", f"D1-D1@ku|n{n}", G("D1", n), G("D1@ku", n), n=n, primary=False, role="보조", aux3=False)

    # ---------------- L16 비율
    res = [(f"{m_}@r{int(r_)} n={n}", C("L16", "X2", f"{m_}@r{int(r_)}-{m_}|n{n}", G(f"{m_}@r{int(r_)}", n), G(m_, n), n=n))
           for m_ in ("D1", "R2") for r_ in R_GRID for n in (0, 10)]
    vs = [_v(r_) for _, r_ in res]
    rp = C("L16", "X2", "D1@r30-P0|n0", G("D1@r30", 0), G("P0", 0), n=0, primary=False, role="보조")
    imit = {}
    if runs is not None and len(runs) and "imit_rms" in runs:
        q = runs[(runs.n == 0) & runs.method.isin(["D1@r1", "D1@r3", "D1", "D1@r30"]) & runs.target.isin(MAIN4) & (runs["mode"] == "x")
                 & (runs.learner == BASE_LEARNER) & (runs.alpha.astype(str) == "1") & (runs.placement == "cell")]
        q = q[q.variant.fillna("").astype(str) == ""] if "variant" in q else q
        imit = q.groupby("method").imit_rms.mean().to_dict()
    seq = [imit.get(k_, np.nan) for k_ in ("D1@r1", "D1@r3", "D1", "D1@r30")]
    mono = bool(all(np.isfinite(seq)) and all(seq[i] > seq[i + 1] for i in range(3)))
    k, m, _ = count_valid(res)
    if k < m:
        txt = na_text(res)
    elif all(v in ("동등", "미결정") for v in vs):
        txt = "비율에 둔감"
    else:
        txt = "비율 의존: 표로 보고한다(기준 r = 10 은 바꾸지 않는다)"
    V("L16", "X2", txt, _fmt(res), role="보조", used=res)
    V("L16", "X2", "판정 불가(D1@r30 − P0 또는 모방 지수가 없다)" if (not valid_row(rp) or not all(np.isfinite(seq))) else (
        "증강은 직접 ML 을 물리식 수준으로 올리는 수단이고 물리식을 넘는 근거가 아니다" if (_v(rp) == "동등" and mono)
        else "수렴 문장의 조건을 충족하지 못함"), f"D1@r30-P0(n = 0) {_v(rp)}; 모방 지수(r 1, 3, 10, 30) {[round(float(v), 2) for v in seq]}, 단조 감소 {mono}",
      role="보조", used=[("D1@r30-P0", rp)])

    # ---------------- L17 라벨 셔플 음성 대조(동등 판정은 기준 λ 와 λ = 1.0 을 모두 본다. 우세 판정은 기준 λ 의 대비로 한다)
    res = [(f"n={n} λ={lam}", C("L17", "X2", f"R1-R1s|n{n}|lam{lam}", G("R1", n, lam), G("R1s", n, lam), n=n, lam=lam, eq_test=True))
           for n in (10, N_ALL) for lam in (LAM_BASE, 1.0)]
    for n in (40, 160):
        C("L17", "X2", f"R1-R1s|n{n}|lam{LAM_BASE}", G("R1", n), G("R1s", n), n=n, lam=LAM_BASE, primary=False, role="보조")
        C("L17", "X2", f"P1-P1s|n{n}", G("P1", n), G("P1s", n), n=n, primary=False, role="보조", aux3=False)
    vs = [_v(r_) for _, r_ in res]
    base17 = [(k_, r_) for k_, r_ in res if k_.endswith(f"λ={LAM_BASE}")]
    k, m, _ = count_valid(res)
    dom = [k_ for k_, r_ in base17 if _v(r_) == "우세"]
    if k < m:
        txt = na_text(res)
    elif all(v == "동등" for v in vs):
        txt = "대상 라벨의 기여는 수준 정보다"
    elif dom:
        txt = "대상 라벨의 공변량과 잔차의 관계가 기여한다: " + ", ".join(dom)
    else:
        txt = "확인하지 못함"
    V("L17", "X2", txt, _fmt(res), role="보조", used=res if (k == m and all(v == "동등" for v in vs)) else base17,
      note="동등은 λ 0.25 와 1.0 의 네 대비가 모두 동등일 때다. 우세는 기준 λ 0.25 의 대비로 판정하고 λ 1.0 의 대비는 stat 에 병기한다")

    # ---------------- L18 α × 결합
    res = [(f"α={al} n={n}", C("L18", "X2", f"R2-R1|a{al}|n{n}", G("R2", n, alpha=al), G("R1", n, alpha=al), n=n, alpha=al))
           for al in ALPHAS_X2 for n in ALPHA_N]
    k, m, _ = count_valid(res)
    if k == 0:
        txt = na_text(res)
    elif any(_v(r_) == "우세" for _, r_ in res):
        txt = "미지지: 우세인 조합이 있다. 표로 보고한다"
    else:
        txt = "지지: 가중을 주어도 증강 결합은 R1 보다 낫지 않다"
    V("L18", "X2", txt, _fmt(res), role="보조", used=res, partial=part_mark(res) if k else None)

    # ---------------- L23 근접 효과
    ra = C("L23", "X3c", "D0[S_rand5]-D0[S_block]", G("D0", N_ALL, placement="rand5"), G("D0", N_ALL), n=N_ALL)
    rb = {m_: T.dd("L23", "X3c", f"[{m_}-P1](S_randm)-[{m_}-P1](S_block)", (G(m_, N_ALL, placement="randm"), G("P1", N_ALL, placement="randm")),
                   (G(m_, N_ALL), G("P1", N_ALL)), n=N_ALL) for m_ in ("D0", "R1")}
    rc = T.dd("L23", "X3c", "[R1-P1](inblk)-[R1-P1](A)|n10", (G("R1", 10, placement="inblk"), G("P1", 10, placement="inblk")),
              (G("R1", 10), G("P1", 10)), n=10)
    V("L23", "X3c", na_text([("(b) D0", rb["D0"])]) if not valid_row(rb["D0"]) else (
        "무작위 분할의 이득에는 라벨 수와 무관한 근접 성분이 있다" if _v(rb["D0"]) == "우세" else "근접 성분은 확인되지 않았다"),
      _fmt([("(a)", ra), ("(b) D0", rb["D0"]), ("(b) R1", rb["R1"]), ("(c)", rc)]), role="보조", used=[("(b) D0", rb["D0"])])
    for bk in BUFFERS:                                                    # S_buf: 서술 통계
        for m_ in ("D0", "R1"):
            C("L23", "X3c", f"{m_}[buf{bk}]-{m_}[S_block]", G(m_, N_ALL, placement=f"buf{bk}"), G(m_, N_ALL), n=N_ALL, primary=False, role="서술",
              aux3=False)

    # ---------------- L24 거리 의존(관측 분석). 대상 단위 집계는 등록한 n(40, 160) 모두에서 판정이 나온 대상만 확정으로 센다
    elig = [("Lena", "x"), ("Canada", "x"), (H.ALASKA, "x"), ("AL-2", "i")]
    n24 = (40, 160)
    r_state, p_state, det, used_r, used_p, r_any = [], [], [], [], [], 0
    for t, md in elig:
        nm = f"{t}|{md}"; tm = tms.get(nm)
        vr, vp = [], []
        for n in n24:
            sr = sp_ = None
            if tm is not None:
                sr = region_dd(tm, (G("R1#near", n), G("P1#near", n)), (G("R1#far", n), G("P1#far", n)))
                sp_ = region_dd(tm, (G("P1#near", n), G("P0#near", n)), (G("P1#far", n), G("P0#far", n)))
            a1 = T.single("L24", "X5", f"[R1-P1](근)-[R1-P1](원)|n{n}", nm, sr, primary=True, n=n, kind="이중 차분")
            a2 = T.single("L24", "X5", f"[P1-P0](근)-[P1-P0](원)|n{n}", nm, sp_, primary=True, n=n, kind="이중 차분")
            vr.append(_v(a1)); vp.append(_v(a2))
            used_r.append((f"{nm} n={n}", a1)); used_p.append((f"{nm} n={n}", a2))
        ok_r = [v for v in vr if v not in NA_VERDICTS]; ok_p = [v for v in vp if v not in NA_VERDICTS]
        r_state.append(True if (len(ok_r) == len(n24) and all(v == "우세" for v in ok_r)) else (False if any(v != "우세" for v in ok_r) else None))
        p_state.append(True if (len(ok_p) == len(n24) and not any(v in ("우세", "열세") for v in ok_p))
                       else (False if any(v in ("우세", "열세") for v in ok_p) else None))
        r_any += bool("우세" in ok_r)
        det.append(f"{nm}: 잔차 {vr}, 재보정 {vp}")
    for states, need, used_, yes_txt, no_txt, lab in (
            (r_state, 2, used_r, "잔차 학습의 순가치는 라벨 가까운 셀에 집중된다", "거리 의존은 확인되지 않았다", "우세 대상"),
            (p_state, 3, used_p, "재보정 이득은 라벨과의 거리에 의존하지 않는다", "재보정 이득의 거리 무의존은 확인되지 않았다", "우세도 열세도 아닌 대상")):
        nd = sum(v is not None for v in states); ny = sum(v is True for v in states)
        tc = tri_count(states, need)
        txt = f"판정 불가(대상 {nd}/{len(elig)}, 기준 {need}개 이상. 남은 대상에 따라 결과가 바뀔 수 있다)" if tc == "unknown" else (yes_txt if tc == "yes" else no_txt)
        V("L24", "X5", txt, f"{lab} {ny}/{nd}(판정한 대상, 등록한 n 모두 기준), 등록 대상 {len(elig)}" + (f", 어느 한 n 우세 {r_any}" if need == 2 else "")
          + " | " + "; ".join(det), role="보조", used=used_, partial=[f"대상 {nd}/{len(elig)}"] if nd < len(elig) else None,
          note="관측 분석. 통제 실험은 L23. 대상 단위 집계는 등록한 n(40, 160) 모두에서 판정이 나온 대상만 센다")

    # ---------------- L25 구간(방향 중립). 6칸 가운데 행이 없는 칸과 n = 0 의 R0 폭이 없는 칸은 알 수 없는 칸이다
    cells25 = [(t, n) for t in ("Lena", "Canada", H.ALASKA) for n in (40, 160)]
    if conf is not None and len(conf):
        s1, s2, det, short = [], [], [], []
        for t, n in cells25:
            w0 = conf[(conf.target == t) & (conf["mode"] == "x") & (conf.method == "R0") & (conf.n == 0) & (conf.level == 90)]
            q = conf[(conf.target == t) & (conf["mode"] == "x") & (conf.method == "R1") & (conf.n == n) & (conf.level == 90)]
            w0v = float(w0.width_cm.iloc[0]) if len(w0) else np.nan
            if not len(q) or not np.isfinite(float(q.coverage.iloc[0])):
                s1.append(None); s2.append(None); det.append(f"{t} n={n}: 행 없음")
                continue
            cov, wid = float(q.coverage.iloc[0]), float(q.width_cm.iloc[0])
            inb = bool(0.85 <= cov <= 0.95)
            nar = False if not inb else (bool(wid < w0v) if (np.isfinite(wid) and np.isfinite(w0v)) else None)
            s1.append(inb); s2.append(nar)
            for src_ in (q, w0):
                if len(src_) and "n_splits_expected" in src_ and int(src_.n_splits.iloc[0]) < int(src_.n_splits_expected.iloc[0]):
                    short.append((int(src_.n_splits.iloc[0]), int(src_.n_splits_expected.iloc[0])))
            det.append(f"{t} n={n}: 커버리지 {cov:.3f}, 폭 {wid:.1f} cm(n = 0 의 R0 폭 {w0v:.1f})")
            T.rows.append(dict(test_id="L25", item="X5", contrast="R1 90 % 구간", scope="region", target=f"{t}|x", n=n, role="주", primary=False,
                               blind=True, coverage=cov, width_cm=wid, width_n0_r0=w0v, in_band=inb, narrower=nar))
        c1 = sum(v is True for v in s1); c2 = sum(v is True for v in s2)
        k1 = sum(v is not None for v in s1); k2 = sum(v is not None for v in s2)
        t1, t2 = tri_count(s1, 4), tri_count(s2, 4)
        if t2 == "yes":
            txt = "라벨이 구간을 좁힌다"
        elif t2 == "no" and t1 == "yes":
            txt = "라벨은 커버리지를 맞추나 폭을 줄이지 못한다"
        elif t2 == "no" and t1 == "no":
            txt = "라벨 기반 보정이 목표 커버리지를 주지 못한다"
        else:
            txt = f"판정 불가(칸 {k2}/{len(cells25)}. 남은 칸에 따라 결과가 바뀔 수 있다)"
        part = ([f"칸 {k2}/{len(cells25)}"] if k2 < len(cells25) else []) + ([f"분할 {min(short)[0]}/{min(short)[1]}"] if short else [])
        V("L25", "X5", txt, f"c1 = {c1}(판정한 칸 {k1}), c2 = {c2}(판정한 칸 {k2}), 등록 칸 {len(cells25)} | " + "; ".join(det), role="보조", partial=part)
    else:
        V("L25", "X5", "판정 불가(행 없음)", "", role="보조")

    # ---------------- L26 학습기 동등성, L27 E0 고정 앵커 경로
    lrs = sorted({g[1] for tm in tms.values() for gd in tm.idx.values() for g in gd
                  if g[0] in ("R1", "R0", "D0") and g[1] not in ("none", BASE_LEARNER) + X_LEARNERS})
    for lr in lrs:
        res, full5 = [], True
        for n in (0, 10, N_ALL):
            for lam in (LAM_BASE, 1.0):
                r_ = C("L26", "X6", f"R1[{lr}]-R1[{BASE_LEARNER}]|n{n}|lam{lam}", G("R1", n, lam, learner=lr), G("R1", n, lam), n=n, lam=lam,
                       learner=lr, aux3=False, eq_test=True)
                res.append((f"n={n} λ={lam}", r_))
        for nm in T.m4:                                                  # 5분할 여부: 학습기의 키가 대상의 사용 분할 전부에 있는지
            tm = tms.get(nm)
            if tm is not None and tm.used:
                have = sum(1 for sp in tm.used if tm.idx[sp].get(G("R1", 0, learner=lr)))
                full5 &= bool(have == len(tm.used) and have >= n_expected(tm))
        for m_ in ("D0", "R0"):
            for n in (0, 10, N_ALL):
                C("L26", "X6", f"{m_}[{lr}]-{m_}[{BASE_LEARNER}]|n{n}", G(m_, n, learner=lr), G(m_, n), n=n, learner=lr, primary=False, role="보조",
                  aux3=False)
        vs = [_v(r_) for _, r_ in res]
        k, m, _ = count_valid(res)
        if not full5:
            txt = "판정 불가(5분할 판정에서 제외: 분할 일부만 있다. LG 의 3분할 판정을 유지한다)"
        elif k < m:
            txt = na_text(res)
        elif all(v == "동등" for v in vs):
            txt = "동등(한계 0.5 cm)"
        elif any(v in ("우세", "열세") for v in vs):
            txt = "차이 있음: " + ", ".join(f"{k_} {v}" for (k_, _), v in zip(res, vs) if v in ("우세", "열세"))
        else:
            txt = "차이를 확인하지 못함"
        V("L26", "X6", txt, _fmt(res), role="보조", learner=lr, used=res, note="분할 1–3 은 본 실행 조각, 분할 4–5 는 확장 조각이다(분할과 실행 작업이 교락)")
    if not lrs:
        V("L26", "X6", "판정 불가(행 없음)", "학습기 축의 키가 없다", role="보조")
    tmc = tms.get("Canada|x")
    n27 = (40, 160, N_ALL)
    reg27 = ["ridge"] + list(H.GPU_LEARNERS)
    if tmc is None:
        V("L27", "X6", "판정 불가(행 없음)", "", role="보조")
    else:
        def sgn(lr, n):
            s_ = region_stats(tmc, G("R0", n, learner=lr), G("P0", 0))
            return None if s_ is None else (float(s_["delta"]), int(np.sign(s_["delta"])), int(s_["n_splits"]), int(s_["n_splits_expected"]))
        ref = {n: sgn(BASE_LEARNER, n) for n in n27}
        ref_ok = all(v is not None and v[2] >= v[3] for v in ref.values())
        states, det = [], []
        for lr in reg27:
            got = {n: sgn(lr, n) for n in n27}
            if any(v is None for v in got.values()):
                states.append(None); det.append(f"{lr}: 행 없음"); continue
            if any(v[2] < v[3] for v in got.values()):
                w = min(got.values(), key=lambda v: v[2] / max(v[3], 1))
                states.append(None); det.append(f"{lr}: 분할 {w[2]}/{w[3]}(집계에서 뺀다)"); continue
            if not ref_ok:
                states.append(None); det.append(f"{lr}: 기준 학습기의 행 없음 또는 분할 부족"); continue
            ok_ = all(got[n][1] == ref[n][1] for n in n27)
            states.append(bool(ok_))
            det.append(f"{lr}: {'같음' if ok_ else '다름'} {[round(got[n][0], 2) for n in n27]}")
            for n in n27:
                T.rows.append(dict(test_id="L27", item="X6", contrast="R0-P0", scope="region", target="Canada|x", n=n, learner=lr, role="주",
                                   primary=False, blind=True, delta=got[n][0], delta_ref=ref[n][0], same_sign=bool(got[n][1] == ref[n][1])))
        tested = sum(v is not None for v in states); same = sum(v is True for v in states)
        tc = tri_count(states, 6)
        txt = (f"판정 불가(시험한 학습기 {tested}/{len(reg27)}, 부호가 같은 학습기 {same}. 남은 학습기에 따라 결과가 바뀔 수 있다)" if tc == "unknown"
               else ("학습기와 무관" if tc == "yes" else "CatBoost 한정"))
        V("L27", "X6", txt, f"부호가 같은 학습기 {same}/{tested}(등록 기준: {len(reg27)}종 중 6종 이상) | " + "; ".join(det), role="보조",
          partial=[f"시험 {tested}/{len(reg27)}"] if tested < len(reg27) else None)
        for t in ("Canada|x", "CA-3|x"):
            tm = tms.get(t)
            if tm is None:
                continue
            for n in n27:
                vd = {}
                for lr in [BASE_LEARNER] + reg27:
                    s_ = region_stats(tm, G("R0", n, learner=lr), G("R1", n, learner=lr))
                    vd[lr] = "행 없음" if s_ is None else verdict4(*_ci(s_["dist"]), *_ci(s_["dist_beq"]), a.delta_eq)
                dom_ = [lr for lr, v in vd.items() if v == "우세"]
                na_ = [lr for lr, v in vd.items() if v in NA_VERDICTS]
                T.rows.append(dict(test_id="L27", item="X6", contrast="R0-R1(같은 학습기)", scope="count", target=t, n=n, role="보조", primary=False,
                                   blind=True, n_learners_dominant=len(dom_), learners_dominant=",".join(dom_),
                                   n_learners_judged=len(vd) - len(na_), n_learners_na=len(na_), learners_na=",".join(na_)))

    # ---------------- L28 판정 안정성
    core = [("D0-P0|n0", "D0", "P0", 0), ("P1-P0|n10", "P1", "P0", 10), ("R1-P1|n10", "R1", "P1", 10), ("R1-P1|all", "R1", "P1", N_ALL),
            ("R1-P0|all", "R1", "P0", N_ALL)]
    base_r = {lab: C("L28", "X9", f"기준|{lab}", G(ma, n), G(mb, 0 if mb == "P0" else n), n=n, primary=False, role="기준", aux3=False)
              for lab, ma, mb, n in core}

    def pool_set(r_):
        """층화 평균 행의 풀 지역 집합(셀 제외 변형의 표지는 뗀다)."""
        return frozenset(v.split("|")[0].split("~")[0] + "|" + v.split("|")[-1] for v in str(r_.get("pool_regions", "")).split(",") if v)

    def shift(r_0, r_1):
        v0, v1 = _v(r_0), _v(r_1)
        if v0 in NA_VERDICTS or v1 in NA_VERDICTS:
            return "판정 불가"
        if pool_set(r_0) != pool_set(r_1):
            return "판정 불가(풀 지역 불일치)"
        if v0 == v1:
            return "강건"
        if {v0, v1} == {"우세", "열세"}:
            return "의존"
        if (v0 in ("우세", "열세")) != (v1 in ("우세", "열세")):
            return "약화"
        return "강건"                                                     # 동등과 미결정 사이의 변화

    def sub_m(m_, var):
        """변형 설정에서 핵심 대비의 방법 이름."""
        kind, val = var
        if kind == "kappa":
            return f"{m_}@k{val}" if m_ in ("P1", "R1") else m_
        if kind == "anchor":
            return f"{m_}@{val}" if m_ in ("P0", "P1", "R1") else m_
        if kind == "x23c":
            return {"R1": "R1@x23c", "D0": "D0c"}.get(m_, m_)
        return m_
    # (이름, 변형, 셀 제외 표지, 맹검 여부). κ 와 앵커는 앞선 실험에서 방향을 보았다(계획서 §6A.7)
    variants = [("κ 3", ("kappa", 3), None, False), ("κ 30", ("kappa", 30), None, False), ("6–7월 제외", ("cell", "noearly"), "noearly", True),
                ("탐침만", ("cell", "probe"), "probe", True)] + [(f"앵커 {b}", ("anchor", b), None, False) for b in ANCHORS] \
        + [("CCI 입력 제외", ("x23c", ""), None, True)]
    summ = {}
    for vname, var, cellvar, vblind in variants:
        names = [f"{t}~{cellvar}|x" for t in MAIN4] if cellvar else None
        out, used = {}, []
        for lab, ma, mb, n in core:
            r_ = C("L28", "X9", f"{vname}|{lab}", G(sub_m(ma, var), n), G(sub_m(mb, var), 0 if sub_m(mb, var) in ("P0",) or mb == "P0" else n),
                   names=names, n=n, variant=vname, blind=vblind, aux3=False, note="" if vblind else "κ, λ, 앵커는 비맹검")
            out[lab] = shift(base_r[lab], r_)
            if r_ is not None:
                r_["stability"] = out[lab]; r_["verdict4_base"] = _v(base_r[lab])
            if not out[lab].startswith("판정 불가"):
                used += [(lab, r_), (f"기준 {lab}", base_r[lab])]
        n_ok = sum(not v.startswith("판정 불가") for v in out.values())
        worst = "의존" if "의존" in out.values() else ("약화" if "약화" in out.values() else (
            f"판정 불가(판정한 대비 0/{len(out)})" if n_ok == 0 else "강건"))
        summ[vname] = worst
        V("L28", "X9", worst, "; ".join(f"{k_}: {v}" for k_, v in out.items()), role="보조", variant=vname, blind=vblind, used=used,
          partial=[f"대비 {n_ok}/{len(out)}"] if 0 < n_ok < len(out) else None)
    if summ.get("탐침만", "") in ("약화", "의존"):
        V("L28", "X9", "결론을 'GPR 유래 라벨을 포함한 원천 조건'으로 한정한다", f"탐침만 변형: {summ['탐침만']}", role="보조", blind=True)
    res = [(f"{b} n={n}", C("L28", "X9", f"R1@{b}-R1|n{n}", G(f"R1@{b}", n), G("R1", n), n=n, primary=False, role="보조", blind=False, aux3=False))
           for b in ANCHORS for n in (0, 10, N_ALL)]
    judged = [(k_, r_) for k_, r_ in res if not k_.endswith("n=0")]        # n = 0 의 R1@b − R1 은 R0@b − R0 과 같다(라벨과 무관). 보조 행으로만 둔다
    k, m, _ = count_valid(judged)
    if k == 0:
        txt = na_text(judged)
    elif any(_v(r_) == "우세" for _, r_ in judged):
        txt = "앵커 교체에 우세인 조합이 있다: 표로 보고한다"
    else:
        txt = "앵커 교체의 이득 없음"
    V("L28", "X9", txt, _fmt(res), role="보조", blind=False, used=judged, partial=part_mark(judged) if k else None,
      note="판정은 n ∈ {10, 전량}의 대비로 한다. n = 0 의 대비는 라벨과 무관한 보조 행이다")
    sel_tab = loto_kappa_lam(tms, MAIN4 + [H.ALASKA])
    if len(sel_tab):
        for r_ in sel_tab.to_dict("records"):
            T.rows.append(dict(test_id="L28", item="X9", contrast="대상 하나 제외 κ·λ 선택", scope="region", role="보조", primary=False, blind=False, **r_))
        dmax = float(np.nanmax(np.abs(sel_tab.delta_sel.values - sel_tab.delta_fixed.values)))
        V("L28", "X9", "사전 고정값의 선택 편향은 확인되지 않았다" if dmax <= a.delta_eq else "고른 κ·λ 의 결과가 고정값과 0.5 cm 넘게 다르다",
          f"최대 차 {dmax:.2f} cm(R1 − P0, n = 10), 대상 {len(sel_tab)}/{len(MAIN4) + 1}", role="보조", blind=False,
          partial=[f"대상 {len(sel_tab)}/{len(MAIN4) + 1}"] if len(sel_tab) < len(MAIN4) + 1 else None)

    # ---------------- L29 물리 기준선(확인적)
    bl = ["B:s_aff", "B:ed_raw", "B:ku_raw", "B:cci_raw", "B:cci_aff", "B:ens"] + [f"P0@{b}" for b in ANCHORS]
    res = [(m_, C("L29", "N1", f"{m_}-P0|n0", G(m_, 0), G("P0", 0), n=0)) for m_ in bl]
    k, m, miss = count_valid(res)
    dom = [(m_, r_) for m_, r_ in res if _v(r_) == "우세"]
    if not dom and k < m:
        V("L29", "N1", na_text(res, "기준선"), _fmt(res), used=res)
    elif not dom:
        V("L29", "N1", "원천 계수 Stefan 은 시험한 기준선 가운데 약한 기준선이 아니다", _fmt(res), used=res)
    else:
        ps, _r = min(dom, key=lambda v: v[1]["delta"])
        b_ = ps.split("@")[1] if ps.startswith("P0@") else None
        p1s = f"P1@{b_}" if b_ else ("B:ens" if ps == "B:ens" else ps)

        def gp1(n):
            return G(p1s, n) if (b_ or ps == "B:ens") else G(ps, 0)
        rr = [("L8 R1-P*|all", C("L29", "N1", f"R1-{ps}|all", G("R1", N_ALL), G(ps, 0), n=N_ALL, primary=False, role="보조", aux3=False))]
        rr += [(f"L4 R1-P1*|n{n}", C("L29", "N1", f"R1-{p1s}|n{n}", G("R1", n), gp1(n), n=n, primary=False, role="보조", aux3=False))
               for n in (10, 40, 160, N_ALL)]
        if b_:
            rr += [("L10 R0*-F1k|n0", C("L29", "N1", f"R0@{b_}-F1k|n0", G(f"R0@{b_}", 0), G("F1k", 0), n=0, primary=False, role="보조", aux3=False)),
                   ("L10 R1*-F1k|n10", C("L29", "N1", f"R1@{b_}-F1k|n10", G(f"R1@{b_}", 10), G("F1k", 10), n=10, primary=False, role="보조",
                                        aux3=False))]
        V("L29", "N1", f"P0 보다 우세인 기준선이 있다. P* = {ps}. L4, L8, L10 의 대비를 P* 대비로 병기한다. ML 이 P* 를 넘지 못하면 '물리식을 넘는다'를 "
                       "쓰지 않는다" + ("" if k == m else f" (일부 기준선 판정 불가, 기준선 {k}/{m}. 없는 기준선: {', '.join(miss)}. P* 는 판정한 기준선 "
                                                          "가운데 점 추정이 가장 낮은 것이다)"), _fmt(res) + " | " + _fmt(rr), used=res)
    # 라벨 n개 조건: 재보정 기준선 − P1 에 같은 절차를 쓴다(n ∈ {10, 40, 160, 전량}. n = 40, 160 은 레나·캐나다 2지역 평균이다)
    recal = ["B:ens"] + [f"P1@{b}" for b in ANCHORS]
    ln_used, ln_det, ln_dom = [], [], []
    for n in (10, 40, 160, N_ALL):
        rn = [(m_, C("L29", "N1", f"{m_}-P1|n{n}", G(m_, n), G("P1", n), n=n, primary=False, role="보조", aux3=False)) for m_ in recal]
        ln_used += [(f"{m_} n={n}", r_) for m_, r_ in rn]
        dn = [(m_, r_) for m_, r_ in rn if _v(r_) == "우세"]
        kk, mm, _ = count_valid(rn)
        if dn:
            p1n, r1n = min(dn, key=lambda v: v[1]["delta"])
            rr_ = C("L29", "N1", f"R1-{p1n}|n{n}", G("R1", n), G(p1n, n), n=n, primary=False, role="보조", aux3=False)
            ln_dom.append(str(n) if n != N_ALL else "전량")
            ln_det.append(f"n={n}: P1* = {p1n}(Δ {r1n['delta']:.2f}, 판정 {kk}/{mm}), R1 − P1* {_v(rr_)}")
        else:
            ln_det.append(f"n={n}: 우세 없음(판정 {kk}/{mm})")
    k, m, _ = count_valid(ln_used)
    if ln_dom:
        txt = (f"라벨 n개 조건에서 P1 보다 우세인 재보정 기준선이 있다(n = {', '.join(ln_dom)}). R1 − P1* 를 병기한다. R1 이 P1* 보다 우세가 아닌 n 에서는 "
               "'재보정 물리식을 넘는다'를 쓰지 않는다")
    elif k < m:
        txt = na_text(ln_used)
    else:
        txt = "라벨 n개 조건에서 P1 보다 우세인 재보정 기준선은 없다"
    V("L29", "N1", txt, "; ".join(ln_det), role="보조", used=ln_used, partial=part_mark(ln_used) if (ln_dom and k < m) else None,
      note="라벨 n개 조건의 절차. n = 40, 160 은 레나·캐나다 2지역 평균이다")
    for tag in ("B:ku_raw", "P0@ku"):                                     # p4_ku 결측 셀을 채점에서 뺀 방식
        C("L29", "N1", f"{tag}-P0|n0|kuok", G(f"{tag}#kuok", 0), G("P0#kuok", 0), n=0, primary=False, role="보조(채점 제외 방식)", aux3=False)
    # 결측 대체 민감도(보조): 공변량 결측을 원천 셀 중앙값으로 채워 다시 계산한 p4_ku, p2_edaphic 의 기준선
    base_of = dict(res)
    trb = [("B:ku_raw_tr", "B:ku_raw"), ("B:ed_raw_tr", "B:ed_raw"), ("P0@ku_tr", "P0@ku"), ("P0@ed_tr", "P0@ed"), ("B:ens_tr", "B:ens")]
    sens = [(mt, C("L29", "N1", f"{mt}-P0|n0", G(mt, 0), G("P0", 0), n=0, primary=False, role="보조(결측 대체 민감도)", aux3=False)) for mt, _ in trb]
    for mt, mb in trb:
        C("L29", "N1", f"{mt}-{mb}|n0", G(mt, 0), G(mb, 0), n=0, primary=False, role="보조(결측 대체 민감도)", aux3=False)
    if any(not valid_row(r_) or not valid_row(base_of.get(mb)) for (mt, r_), (_, mb) in zip(sens, trb)):
        txt = na_text(sens + [(mb, base_of.get(mb)) for _, mb in trb], "기준선")
    else:
        diff = [f"{mt} {_v(r_)}(기준 {_v(base_of[mb])})" for (mt, r_), (_, mb) in zip(sens, trb) if _v(r_) != _v(base_of[mb])]
        txt = "결측 대체 통계를 학습 쪽 중앙값으로 바꿔도 4분 판정이 같다" if not diff else "결측 대체 통계에 따라 4분 판정이 달라진다: " + ", ".join(diff)
    V("L29", "N1", txt, _fmt(sens), role="보조", used=sens, note="결측 대체 민감도: p4_ku, p2_edaphic 의 공변량 결측을 원천 셀 중앙값으로 채운 값")

    # ---------------- L30 학습기 용량(확인적)
    res = [(lr, C("L30", "N3", f"D0[{lr}]-P0|n0", G("D0", 0, learner=lr), G("P0", 0), n=0, learner=lr, blind=False, note="L1 관련(비맹검)"))
           for lr in N3_LEARNERS]
    vs = [_v(r_) for _, r_ in res]
    k, m, _ = count_valid(res)
    dom = [k_ for (k_, _), v in zip(res, vs) if v == "우세"]
    if dom:
        txt = f"{rej(k, m)}: L1 의 결론을 '저용량 설정 한정'으로 적는다. 우세 학습기 " + ", ".join(dom)
    elif k < m:
        txt = na_text(res)
    else:
        txt = "지지: 용량을 키우거나 원천 교차검증으로 조정해도 직접 ML 은 라벨 0 전이에서 물리식을 넘지 못한다"
    V("L30", "N3", txt, _fmt(res), blind=False, used=res, note="L1 관련(비맹검)")
    for lr in N3_LEARNERS:
        for n in (0, 10, N_ALL):
            C("L30", "N3", f"R1[{lr}]-R1[{BASE_LEARNER}]|n{n}", G("R1", n, learner=lr), G("R1", n), n=n, learner=lr, primary=False, role="보조",
              aux3=False)

    # ---------------- L31 분할 구성. 판정 행은 캐나다를 포함하므로 비맹검이다(계획서 §6A.7)
    m4n = [f"{t}|x" for t in MAIN4]
    if n4 is not None and len(n4.get("splitdist", [])):
        sd = n4["splitdist"]
        q = sd[(sd.contrast == "R1-P0|all") & sd.target.isin(m4n)]
        if "primary_source" in q:
            q = q[q.primary_source.astype(bool)]
        dep = sd[(sd.contrast == "R1-P0|all") & (sd.lg5_outside_10_90.astype(str).str.lower() == "true")].target.tolist()
        states, short = [], []
        for nm in m4n:
            r_ = q[q.target == nm]
            if not len(r_) or not np.isfinite(float(r_.frac_neg.iloc[0])):
                states.append(None); continue
            states.append(bool(float(r_.frac_neg.iloc[0]) >= 0.8))
            if "n_splits_expected" in r_ and int(r_.n_splits.iloc[0]) < int(r_.n_splits_expected.iloc[0]):
                short.append((int(r_.n_splits.iloc[0]), int(r_.n_splits_expected.iloc[0])))
        for r_ in q.to_dict("records"):
            T.rows.append(dict(test_id="L31", item="N4", scope="region", role="주", primary=False, blind=r_["target"] not in ("Canada|x",), **r_))
        kh = sum(v is not None for v in states); ky = sum(v is True for v in states)
        tc = tri_count(states, 3)
        txt = (f"판정 불가(지역 {kh}/{len(m4n)}. 남은 지역에 따라 결과가 바뀔 수 있다)" if tc == "unknown"
               else ("분할 구성에 강건" if tc == "yes" else "지역 의존"))
        V("L31", "N4", txt, f"음수 분할 비율 ≥ 0.8 인 주 지역 {ky}/{kh}(등록 지역 {len(m4n)}); 분할 구성 의존 표시 대상 {dep}", role="보조", blind=False,
          partial=([f"지역 {kh}/{len(m4n)}"] if kh < len(m4n) else []) + ([f"분할 {min(short)[0]}/{min(short)[1]}"] if short else []),
          note="지역 수준 검정(부호, 순열, 임의효과)은 판정에 쓰지 않는다. lgx_region_inference.csv 참조. 캐나다는 재현(비맹검)")
    else:
        V("L31", "N4", "판정 불가(행 없음)", "", role="보조", blind=False)
    return T.frame()


def loto_kappa_lam(tms, targets):
    """대상 하나 제외 방식의 κ·λ 선택. 대상 t 를 뺀 나머지 대상의 평균 Δ(R1 − P0, n = 10)가 가장 작은 (κ, λ)를 t 에 적용한다."""
    def key(k, lam):
        return G("R1" if k == 10 else f"R1@k{int(k)}", 10, lam)
    tab = {}
    for t in targets:
        tm = tms.get(f"{t}|x")
        if tm is None:
            continue
        for k, lam in product((3, 10, 30), LAMS):
            r = H.contrast(curve_view(tm), key(k, lam), H.P0_GRP)
            if r is not None and np.isfinite(r["delta"]):
                tab[(t, k, lam)] = float(r["delta"])
    rows = []
    for t in targets:
        others = [o for o in targets if o != t]
        cand = []
        for k, lam in product((3, 10, 30), LAMS):
            v = [tab.get((o, k, lam), np.nan) for o in others]
            if np.isfinite(v).sum() >= 2:
                cand.append((float(np.nanmean(v)), k, lam))
        if not cand or (t, 10, LAM_BASE) not in tab:
            continue
        _, k, lam = min(cand)
        if (t, k, lam) not in tab:
            continue
        rows.append(dict(target=f"{t}|x", kappa_sel=float(k), lam_sel=float(lam), delta_sel=tab[(t, k, lam)], delta_fixed=tab[(t, 10, LAM_BASE)],
                         n=10, n_others=len(others)))
    return pd.DataFrame(rows)


# ================================================================ 집계: 표
def conformal_table(tms, a):
    """구간의 커버리지와 평균 폭(비율형 통계의 블록 부트스트랩 CI)."""
    rows = []
    for nm, tm in tms.items():
        if "~" in nm:
            continue
        gs = sorted({g for gd in tm.idx.values() for g in gd if str(g[0]).startswith("cov")}, key=str)
        for g in gs:
            lv = int(str(g[0])[3:5]); M = str(g[0]).split(":")[1]
            c_ = boot_mean_blocks(tm, g)
            w_ = boot_mean_blocks(tm, ("wid" + str(g[0])[3:],) + tuple(g[1:]))
            if c_ is None:
                continue
            rows.append(dict(target=tm.target, mode=tm.mode, method=M, learner=g[1], n=int(g[4]), lam=float(g[5]), level=lv, target_cov=lv / 100.0,
                             coverage=1.0 - c_["value"], cov_lo=1.0 - c_["ci_hi"], cov_hi=1.0 - c_["ci_lo"],
                             width_cm=w_["value"] if w_ else np.nan, width_lo=w_["ci_lo"] if w_ else np.nan, width_hi=w_["ci_hi"] if w_ else np.nan,
                             n_splits=c_["n_splits"], n_splits_expected=n_expected(tm), point_only=bool(not tm.has_ci)))
    return pd.DataFrame(rows)


def distance_table(tms, a):
    """거리 층별 RMSE 와 층별 대비(R1 − P1, P1 − P0, D0 − P0, R1 − P0)."""
    rows = []
    for nm, tm in tms.items():
        if "~" in nm:
            continue
        ns = sorted({g[4] for gd in tm.idx.values() for g in gd if str(g[0]).endswith("#near")})
        for n in ns:
            for stx in ("near", "mid", "far"):
                for lab, ma, mb in (("R1-P1", "R1", "P1"), ("P1-P0", "P1", "P0"), ("D0-P0", "D0", "P0"), ("R1-P0", "R1", "P0"), ("R0-P0", "R0", "P0")):
                    s_ = region_stats(tm, G(f"{ma}#{stx}", n), G(f"{mb}#{stx}", n))
                    if s_ is None:
                        continue
                    rows.append(stats_row(s_, a, nm, dict(mode=tm.mode, n=int(n), stratum=stx, contrast=lab, n_splits=s_["n_splits"])))
    return pd.DataFrame(rows)


def n4_tables(a, stores50, stores200, D50, D200):
    """N4 분할 확장: 분할별 점 Δ 의 분포(부트스트랩 없음)와 지역 수준 추론."""
    out = dict(splitdist=pd.DataFrame(), region=pd.DataFrame())
    rows, per_t = [], {}
    contrasts = [("D0-P0|n0", G("D0", 0), H.P0_GRP), ("P1-P0|n10", G("P1", 10), H.P0_GRP), ("R1-P1|n10", G("R1", 10), G("P1", 10)),
                 ("R1-P1|all", G("R1", -1), G("P1", -1)), ("R1-P0|all", G("R1", -1), H.P0_GRP), ("P1-P0|all", G("P1", -1), H.P0_GRP),
                 ("P2-P0|all", G("P2", -1), H.P0_GRP)]
    for src, stores, D in (("n4", stores50, D50), ("n4a", stores200, D200)):
        if not stores or D is None:
            continue
        for nm in sorted({k[0] for k in stores}):
            tm = make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, 0)
            for lab, gA, gB in contrasts:
                r = H.contrast(tm, gA, gB)
                if r is None or not r["delta_split"]:
                    continue
                ds = pd.Series(r["delta_split"]).sort_index()
                v = ds.values.astype(float)
                lg5 = ds[ds.index.isin([1, 2, 3, 4, 5])]
                m5 = float(lg5.mean()) if len(lg5) else np.nan
                rng = np.random.RandomState(seed_of("lgx-sub5", nm, lab))
                sub5 = np.array([v[rng.choice(len(v), min(5, len(v)), replace=False)].mean() for _ in range(2000)]) if len(v) >= 5 else np.array([])
                p10, p90 = (np.percentile(v, 10), np.percentile(v, 90)) if len(v) else (np.nan, np.nan)
                rows.append(dict(source=src, target=nm, contrast=lab, n_splits=int(len(v)), n_splits_expected=n_expected(tm), delta_mean=float(v.mean()),
                                 delta_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                 frac_neg=float(np.mean(v < 0)), p05=float(np.percentile(v, 5)), p10=float(p10), p50=float(np.percentile(v, 50)),
                                 p90=float(p90), p95=float(np.percentile(v, 95)), lg5_mean=m5, lg5_n=int(len(lg5)),
                                 lg5_pct_single=float(np.mean(v <= m5) * 100) if np.isfinite(m5) else np.nan,
                                 lg5_pct_sub5=float(np.mean(sub5 <= m5) * 100) if (np.isfinite(m5) and len(sub5)) else np.nan,
                                 lg5_outside_10_90=bool(np.isfinite(m5) and (m5 < p10 or m5 > p90)),
                                 delta_splits=json.dumps({str(int(k)): round(float(x), 4) for k, x in ds.items()})))
                if src == "n4" or lab in ("P1-P0|n10", "P1-P0|all", "P2-P0|all"):
                    per_t[(src, lab, nm)] = (float(v.mean()), float(v.var(ddof=1) / len(v)) if len(v) > 1 else np.nan)
    sd = pd.DataFrame(rows)
    if len(sd):
        dup = sd.duplicated(subset=["target", "contrast"], keep="first") & (sd.source == "n4a")      # 같은 대비는 분할 50회 값을 주로 쓴다
        sd["primary_source"] = ~dup
    out["splitdist"] = sd
    # 지역 수준 추론: 지역 Δ = 분할 평균, 분산 = 분할 간 분산 / 분할 수(분할 사이 의존을 무시한 근사. 판정에 쓰지 않는다)
    reg = []
    sets = dict(main4=[f"{t}|x" for t in MAIN4], main4_alaska=[f"{t}|x" for t in MAIN4 + [H.ALASKA]],
                no_russia_w=[f"{t}|x" for t in MAIN4 if t != "Russia_W"],
                main4_sub_i=[f"{t}|x" for t in MAIN4] + [f"{t}|i" for t in H.SUB_AL + H.SUB_OTHER])
    for (src, lab) in sorted({(k[0], k[1]) for k in per_t}):
        for sname, names in sets.items():
            th = np.array([per_t[(src, lab, nm)][0] for nm in names if (src, lab, nm) in per_t], float)
            vv = np.array([per_t[(src, lab, nm)][1] for nm in names if (src, lab, nm) in per_t], float)
            used = [nm for nm in names if (src, lab, nm) in per_t]
            if len(th) < 2:
                continue
            reg.append(dict(source=src, contrast=lab, region_set=sname, regions=",".join(used), **region_inference(th, vv),
                            note="하위 지역 i 모드는 알래스카 원천을 공유하므로 독립 지역이 아니다" if sname == "main4_sub_i" else ""))
            if sname == "main4":
                for j, nm in enumerate(used):
                    reg.append(dict(source=src, contrast=lab, region_set=f"main4 − {nm}", regions=",".join(u for u in used if u != nm),
                                    k=len(th) - 1, mean=float(np.delete(th, j).mean()), note="지역 하나 제외 평균"))
    out["region"] = pd.DataFrame(reg)
    return out


def region_inference_main(tms, a):
    """주 묶음(분할 1–5)의 지역 수준 추론. 지역 Δ = 분할 평균 점 추정, 분산 = 블록 부트스트랩 분포의 분산.
    지역 집합: 주 4지역, 주 4지역 + Alaska, 러시아 W 제외, 주 4지역 + 하위 지역 x 모드 10대상. 판정에 쓰지 않는다."""
    core = [("D0-P0|n0", G("D0", 0), H.P0_GRP), ("P1-P0|n10", G("P1", 10), H.P0_GRP), ("R1-P1|n10", G("R1", 10), G("P1", 10)),
            ("R1-P1|all", G("R1", -1), G("P1", -1)), ("R1-P0|all", G("R1", -1), H.P0_GRP)]
    sets = dict(main4=[f"{t}|x" for t in MAIN4], main4_alaska=[f"{t}|x" for t in MAIN4 + [H.ALASKA]],
                no_russia_w=[f"{t}|x" for t in MAIN4 if t != "Russia_W"],
                main4_sub_x=[f"{t}|x" for t in MAIN4] + [f"{t}|x" for t in H.SUB_AL + H.SUB_OTHER])
    every = sorted({nm for v in sets.values() for nm in v})
    rows = []
    for lab, gA, gB in core:
        per = {}
        for nm in every:
            tm = tms.get(nm)
            if tm is None:
                continue
            s_ = region_stats(tm, gA, gB)
            if s_ is not None:
                d_ = s_["dist"]
                per[nm] = (s_["delta"], float(np.nanvar(d_, ddof=1)) if d_ is not None and np.isfinite(d_).sum() > 2 else np.nan)
        for sname, names in sets.items():
            used = [nm for nm in names if nm in per]
            if len(used) < 2:
                continue
            th = np.array([per[nm][0] for nm in used], float); vv = np.array([per[nm][1] for nm in used], float)
            rows.append(dict(source="main(분할 1–5)", contrast=lab, region_set=sname, regions=",".join(used), **region_inference(th, vv),
                             note="하위 지역은 상위 지역과 셀이 겹친다(독립 지역이 아니다)" if sname == "main4_sub_x" else ""))
            if sname == "main4":
                for j, nm in enumerate(used):
                    rows.append(dict(source="main(분할 1–5)", contrast=lab, region_set=f"main4 − {nm}", regions=",".join(u for u in used if u != nm),
                                     k=len(th) - 1, mean=float(np.delete(th, j).mean()), note="지역 하나 제외 평균"))
        H4.boot_weights.cache_clear()
    return pd.DataFrame(rows)


def region_inference(theta, var):
    """지역 단위 추론: 부호 검정(양측 정확), 부호 뒤집기 순열 검정(2^k 전수), DerSimonian-Laird τ² 와 Hartung-Knapp CI."""
    from scipy import stats as sps
    th = np.asarray(theta, float); k = len(th)
    neg = int((th < 0).sum()); pos = int((th > 0).sum())
    n_ = neg + pos
    p_sign = float(min(1.0, 2.0 * sps.binom.cdf(min(neg, pos), n_, 0.5))) if n_ > 0 else np.nan
    obs = abs(th.mean())
    cnt = sum(1 for sg in product((-1.0, 1.0), repeat=k) if abs(float(np.mean(th * np.array(sg)))) >= obs - 1e-12)
    p_perm = float(cnt / (2 ** k))
    out = dict(k=int(k), mean=float(th.mean()), n_neg=neg, p_sign=p_sign, p_perm=p_perm, p_perm_min=float(2.0 / (2 ** k)),
               k_re=0, tau2=np.nan, mean_re=np.nan, hk_lo=np.nan, hk_hi=np.nan, hk_p=np.nan)
    v = np.asarray(var, float)
    okv = np.isfinite(v) & (v > 0)                                        # 임의효과 평균은 분산이 있는 지역만 쓴다(k_re)
    th, v, k = th[okv], v[okv], int(okv.sum())
    out["k_re"] = k
    if k >= 2:
        w = 1.0 / v
        mu_f = float((w * th).sum() / w.sum())
        Q = float((w * (th - mu_f) ** 2).sum())
        c_ = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - (k - 1)) / c_) if c_ > 0 else 0.0
        ws = 1.0 / (v + tau2)
        mu = float((ws * th).sum() / ws.sum())
        qv = float((ws * (th - mu) ** 2).sum() / ((k - 1) * ws.sum()))
        se = float(np.sqrt(qv))
        tcrit = float(sps.t.ppf(0.975, k - 1))
        out.update(tau2=float(tau2), mean_re=mu, hk_lo=mu - tcrit * se, hk_hi=mu + tcrit * se,
                   hk_p=float(2 * sps.t.sf(abs(mu / se), k - 1)) if se > 0 else np.nan)
    return out


def gate_table(a, sh_base, gate_dir, gate_tag):
    """재현 점검. 확장의 기준 방법 값(lgxb)과 본 실행 cpu 조각의 공통 키를 키별 셀 가중 RMSE 로 대조한다.
    허용 차는 물리식 1e-9 cm, CatBoost 방법 0.02 cm 다."""
    rows = []
    h40_now = H.code_sha()
    for s_ in sh_base:
        ref = Path(gate_dir) / f"{gate_tag}__cpu__{s_['target']}__{s_['mode']}__s{s_['split']}_blocksse.npz"
        refu = Path(str(ref)[:-len("_blocksse.npz")] + "_unit.json")
        row = dict(target=s_["target"], mode=s_["mode"], split=s_["split"], ref=str(ref.name), status="", n_keys=0, n_keys_phys=0, n_keys_ml=0,
                   max_diff_phys=np.nan, max_diff_ml=np.nan, pass_phys=None, pass_ml=None, passed=None, blocks_equal=None, code_sha_h40_ref="",
                   code_sha_h40_now=h40_now, worst_key="")
        if not ref.exists():
            row["status"] = "기준 조각 없음"; rows.append(row); continue
        try:
            A = load_stores(s_["npz"]); B = load_stores(ref)
            nm = f"{s_['target']}|{s_['mode']}"
            sa, sb = A.get((nm, s_["split"])), B.get((nm, s_["split"]))
            if refu.exists():
                row["code_sha_h40_ref"] = str(json.loads(refu.read_text()).get("code_sha", ""))
            if sa is None or sb is None:
                row["status"] = "저장소 없음"; rows.append(row); continue
            row["blocks_equal"] = bool(np.array_equal(sa.blocks, sb.blocks) and np.array_equal(sa.ncell, sb.ncell))
            keys = [k for k in sa.keys if k in sb and k[0] in BASE_METHODS and k[1] in ("none", BASE_LEARNER) and str(k[2]) == "1" and k[3] == "cell"]
            dp, dm, worst = [], [], (0.0, "")
            for k in keys:
                d_ = abs(sa.rmse(k) - sb.rmse(k))
                (dp if k[1] == "none" else dm).append(d_)
                if d_ >= worst[0]:
                    worst = (d_, json.dumps(list(k)))
            row.update(n_keys=len(keys), n_keys_phys=len(dp), n_keys_ml=len(dm), max_diff_phys=float(max(dp)) if dp else np.nan,
                       max_diff_ml=float(max(dm)) if dm else np.nan, worst_key=worst[1])
            row["pass_phys"] = bool(dp and max(dp) <= GATE_TOL_PHYS); row["pass_ml"] = bool((not dm) or max(dm) <= GATE_TOL_ML)
            row["passed"] = bool(row["pass_phys"] and row["pass_ml"] and row["blocks_equal"] and len(keys) > 0)
            row["status"] = "ok" if row["passed"] else "불일치"
        except Exception as e:                                            # noqa: BLE001
            row["status"] = f"오류: {repr(e)[:120]}"
        rows.append(row)
    return pd.DataFrame(rows)


def lg_aux_table(a, D, floor):
    """본 실행(LG) 조각의 보조 열. L2, L4, L5, L8 의 주 4지역 층화 평균 대비에 지역 블록 공통 재표집 CI, Holm 보정 p, 효과 크기 표기를 붙인다.
    L1–L8 의 판정은 h40 집계 그대로이고 이 표는 판정에 쓰지 않는다. 재표집 횟수는 --nboot 이다(h40 집계는 1,000회)."""
    sh = find_shards_x(a.LGDIR / "shards", a.lg_tag)
    if not sh:
        return pd.DataFrame()
    try:
        stores = load_stores([s_["npz"] for s_ in sh])
    except AssertionError as e:
        print(f"  [warn] 본 실행 조각을 합치지 못했다: {repr(e)[:160]}", flush=True)
        return pd.DataFrame()
    tms = {}
    for nm in sorted({k[0] for k in stores}):
        tms[nm] = make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot)
    T = TestBook(a, tms, D, floor)
    ns = sorted({g[4] for tm in tms.values() for gd in tm.idx.values() for g in gd})
    for m in ("R2", "R3"):
        for n in (10, 40, 160):
            T.contrast("L2", "LG", f"{m}-R1|n{n}", G(m, n), G("R1", n), n=n)
    for n in [v for v in ns if v > 0] + ([-1] if -1 in ns else []):
        T.contrast("L4", "LG", f"R1-P1|n{n}", G("R1", n), G("P1", n), n=n)
    lrs = sorted({g[1] for tm in tms.values() for gd in tm.idx.values() for g in gd if g[0] == "R1" and g[1] not in ("none", BASE_LEARNER)})
    for lr in lrs:
        for n in (0, 10, 40, 160, -1):
            T.contrast("L5", "LG", f"R1[{lr}]-R1[{BASE_LEARNER}]|n{n}", G("R1", n, learner=lr), G("R1", n), n=n, learner=lr, aux3=False)
    T.contrast("L8", "LG", "R1-P0|all", G("R1", -1), H.P0_GRP, n=-1)
    for nm in [f"{t}|x" for t in MAIN4]:
        tm = tms.get(nm)
        if tm is None:
            continue
        for n in [v for v in ns if 0 <= v <= 40]:
            T.single("L1", "LG", f"D0-P0|n{n}", nm, region_stats(tm, G("D0", n), H.P0_GRP), n=n)
    df = T.frame()
    if len(df):
        df["note_lg"] = "보조 열(판정에 쓰지 않는다). Holm 묶음은 가설 id 별 층화 평균 대비"
        prim = df.primary.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False) & df.p_boot.notna()
        df["holm_p"] = np.nan; df["holm_note"] = ""
        for _, g in df[prim].groupby("test_id"):
            hp = MS.holm(g.p_boot.values.astype(float))
            df.loc[g.index, "holm_p"] = hp
            df.loc[g.index[g.verdict4.isin(["우세", "열세"]).values & (hp >= 0.05)], "holm_note"] = "보정 전 유의"
    H4.boot_weights.cache_clear()
    return df


# ================================================================ 집계
def _read_runs(shards):
    frames = []
    for s_ in shards:
        if s_["runs"].stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, variant=str, cv_flag=str), keep_default_na=False,
                            na_values=["", "nan", "NaN"])
            if len(f):
                f["tag"] = s_["tag"]
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_, v in (("alpha_sel", ""), ("fit_flag", ""), ("variant", ""), ("cv_flag", "")):
        runs[c_] = runs[c_].fillna(v).astype(str) if c_ in runs else v
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int) if "n_nonfinite" in runs else 0
    return runs


def check_cfg_x(a, shards, units):
    """조각 사이 설정 일치 확인. (tag, 부분)마다 공통 설정 해시가 하나여야 한다."""
    info = {}
    for key in sorted({(s_["tag"], s_["part"]) for s_ in shards}):
        us = [u for s_, u in zip(shards, units) if (s_["tag"], s_["part"]) == key]
        hs = Counter(str(u.get("cfg_common", "legacy")) for u in us)
        cs = Counter(str(u.get("code_sha", "legacy")) for u in us)
        ch = Counter(str(u.get("code_sha_h40", u.get("code_sha", "legacy"))) for u in us)
        info["|".join(key)] = dict(cfg_common=dict(hs), code_sha=dict(cs), code_sha_h40=dict(ch), n=len(us))
        if len(cs) > 1:
            print(f"  [warn] {key}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
        if len(hs) > 1:
            msg = f"{key}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def main_shards(a):
    """주 묶음의 조각(분할 1–5 의 확장 축)과 분할 4·5 축의 조각, 본 실행 gpu 조각(읽기 전용. 스모크와 사전 점검에서는 읽지 않는다)."""
    tags = []
    for ax in ("base", "x1", "x2", "x9", "x9f", "n3", "prox", "x5", "ridge", "nn"):
        tg = axis_tag(a, ax)
        if tg not in tags:
            tags.append(tg)
    sh_main = [s_ for tg in tags for s_ in find_shards_x(a.SHARDS, tg)]
    sh_g = find_shards_x(a.SHARDS, axis_tag(a, "g45"))
    sh_lg = [] if a.SUFFIX else [s_ for s_ in find_shards_x(a.LGDIR / "shards", a.lg_tag) if s_["part"] == "gpu"]
    return sh_main, sh_g, sh_lg


def load_group(a, sh_own, sh_lg):
    """조각을 읽어 (대상|모드, 분할)로 병합한 저장소와 runs 표. 본 실행 gpu 조각은 확장 조각에 있는 단위에만 합친다."""
    runs = _read_runs(sh_own)
    if len(runs):
        runs = runs.drop_duplicates(subset=["tag", "target", "mode", "split", "variant"] + H.KEY_COLS, keep="first")
    stores = load_stores([s_["npz"] for s_ in sh_own]) if sh_own else {}
    if sh_lg:
        try:
            for k, st in load_stores([s_["npz"] for s_ in sh_lg]).items():
                if k in stores:
                    stores[k].merge(st)
        except AssertionError as e:
            msg = f"본 실행 gpu 조각의 채점 블록이 확장 조각과 다르다: {repr(e)[:160]}"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg)
            print("  [warn] " + msg, flush=True)
    return stores, runs


def curve_rows(runs):
    """곡선 표에 쓰는 runs. 변형 행은 대상 이름에 표지를 붙이고 거리 층과 구간 행은 뺀다."""
    if not len(runs):
        return runs
    rc = runs.copy()
    v = rc.variant.astype(str)
    rc["target"] = np.where(v != "", rc.target.astype(str) + "~" + v, rc.target.astype(str))
    rc = rc[~rc.method.map(is_aux_method)]
    return rc.drop_duplicates(subset=["target", "mode", "split"] + H.KEY_COLS, keep="first")


def curve_of(a, D, stores, runs, floor, names=None, failed=None):
    """대상·모드의 곡선(h40.build_curve)과 4분 판정, 효과 크기 열. 재표집 행렬은 대상마다 비운다.
    failed 에 목록을 주면 계산에 실패한 저장소 이름과 사유를 넣는다(없으면 경고만 한다)."""
    rc = curve_rows(runs)
    out = []
    for nm in (sorted({k[0] for k in stores}) if names is None else names):
        tm = make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot)
        try:
            cur = H.build_curve({nm: curve_view(tm)}, rc[(rc.target == tm.target) & (rc["mode"] == tm.mode)]) if len(rc) else pd.DataFrame()
        except Exception as e:                                            # noqa: BLE001
            print(f"  [warn] 곡선 계산 실패 {nm}: {repr(e)[:200]}", flush=True)
            if failed is not None:
                failed.append(dict(store=nm, target=tm.target, mode=tm.mode, reason=repr(e)[:200]))
            cur = pd.DataFrame()
        if len(cur):
            fl = floor_of(floor, D, tm)
            for kind in ("p0", "p1"):
                cur[f"verdict4_{kind}"] = [verdict4(r_[f"d_{kind}_lo"], r_[f"d_{kind}_hi"], r_[f"d_{kind}_beq_lo"], r_[f"d_{kind}_beq_hi"], a.delta_eq)
                                           for r_ in cur.to_dict("records")]
            eff = [effect_cols(d_, p_, fl) for d_, p_ in zip(cur.d_p0.values, cur.rmse_p0.values)]
            cur["delta_cm"] = [e_["delta_cm"] for e_ in eff]; cur["delta_pct_p0"] = [e_["delta_pct_p0"] for e_ in eff]
            cur["frac_reducible"] = [e_["frac_reducible"] for e_ in eff]; cur["small_effect"] = [e_["small_effect"] for e_ in eff]
            cur["floor_rmse_cm"] = fl
            out.append(cur)
        H4.boot_weights.cache_clear()                                     # nboot 10,000 의 재표집 행렬은 대상마다 비운다
    return out


def _curve_worker(argv, target, mode, floor_records):
    """집계 워커: 대상·모드 하나의 조각만 읽어 곡선을 계산한다."""
    warnings.filterwarnings("ignore")
    a = parse_args(argv)
    sh_main, sh_g, sh_lg = main_shards(a)

    def pick(sh):
        return [s_ for s_ in sh if s_["target"] == target and s_["mode"] == mode]
    stores, runs = load_group(a, pick(sh_main) + pick(sh_g), pick(sh_lg))
    D = get_data_x(h40_args(a, "base"))
    bad = []
    cur = curve_of(a, D, stores, runs, pd.DataFrame(floor_records), failed=bad)
    return dict(curve=pd.concat(cur, ignore_index=True) if cur else pd.DataFrame(), failed=bad)


def curve_argv(a):
    """곡선 워커에 넘기는 인자. 주 프로세스가 낮춘 재표집 횟수(limit_local)를 워커도 쓰게 한다."""
    av = list(a.ARGV)
    if getattr(a, "nboot_asked", None) is not None:
        av += ["--nboot", str(int(a.nboot))]
    return av


def summarize(a, elapsed=0.0, skipped=None):
    """집계. 곡선은 대상·모드 단위로 나눠 --workers 개의 프로세스에서 계산한다(재표집 10,000회에서 곡선 계산이 가장 오래 걸린다).
    곡선 워커가 실패하면(워커 비정상 종료 포함) 그 대상·모드를 주 프로세스에서 다시 계산하고, 그래도 실패한 저장소는 meta 의 curve_failed 와
    <tag>_failed.csv 에 남긴다. 호출한 쪽(main)이 종료 코드에 반영한다."""
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")      # 거리 층 대비: 재표집에서 셀이 하나도 뽑히지 않은 반복
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    sh_main, sh_g, sh_lg = main_shards(a)
    sh_n4 = find_shards_x(a.SHARDS, axis_tag(a, "n4"))
    sh_n4a = find_shards_x(a.SHARDS, axis_tag(a, "n4a"))
    everything = sh_main + sh_g + sh_n4 + sh_n4a
    if not everything:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{a.tag}*", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in everything]
    cfg_info = check_cfg_x(a, everything, units)
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    D5 = get_data_x(h40_args(a, "base"))
    # ---------------- 본 실행의 gpu 조각(분할 1–3)을 읽기 전용으로 합친다. 공통 설정 해시가 확장의 분할 4·5 조각과 다르면 중단한다
    x6 = dict(n_lg_gpu=len(sh_lg), n_lgxg=len(sh_g), cfg_equal=None, note="분할 1–3 은 본 실행 작업, 분할 4–5 는 확장 작업이다(분할과 실행 작업이 교락)")
    h40_now = H.code_sha()
    if sh_lg:
        ul = [json.loads(s_["unit"].read_text()) for s_ in sh_lg]
        hl = {str(u.get("cfg_common", "legacy")) for u in ul}
        hg = {str(u.get("cfg_common", "legacy")) for s_, u in zip(everything, units) if s_ in sh_g}
        shl = {str(u.get("code_sha", "")) for u in ul}
        x6.update(cfg_common_lg=sorted(hl), cfg_common_lgxg=sorted(hg), code_sha_h40_lg=sorted(shl), code_sha_h40_now=h40_now)
        if shl - {h40_now}:
            print(f"  [warn] h40 의 코드 해시({h40_now})가 본 실행 조각의 값({sorted(shl)})과 다르다", flush=True)
        if sh_g:
            x6["cfg_equal"] = bool(len(hl) == 1 and hl == hg)
            if not x6["cfg_equal"]:
                msg = f"X6: 본 실행 gpu 조각의 공통 설정 해시 {sorted(hl)} 와 확장 분할 4·5 조각의 값 {sorted(hg)} 가 다르다"
                if not a.allow_mixed_cfg:
                    raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
                print("  [warn] " + msg, flush=True)
    stores, runs = load_group(a, sh_main + sh_g, sh_lg)
    failed = runs[(runs.n_nonfinite > 0) | (runs.fit_flag != "")] if len(runs) else runs
    floor, floor_meta = floor_table(D5.df)
    tms = {nm: make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D5, a.nboot) for nm in sorted({k[0] for k in stores})}
    # ---------------- 곡선
    pairs = sorted({(s_["target"], s_["mode"]) for s_ in sh_main + sh_g})
    nw = min(max(int(a.workers), 1), max(len(pairs), 1))
    curves, curve_failed = [], []

    def stores_of(t, m):
        """대상·모드의 저장소 이름(셀 제외 변형 '<대상>~<변형>|<모드>' 포함)."""
        return sorted(nm for nm in {k[0] for k in stores} if nm.split("|")[0].split("~")[0] == t and nm.split("|")[1] == m)
    if nw > 1:
        redo = []
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            futs = {ex.submit(_curve_worker, curve_argv(a), t, m, floor.to_dict("records")): (t, m) for t, m in pairs}
            for f, tmn in futs.items():
                try:
                    r_ = f.result()
                    curves.append(r_["curve"])
                    redo += [(tmn, [q["store"] for q in r_["failed"]])] if r_["failed"] else []
                except (Exception, SystemExit) as e:                      # noqa: BLE001  BrokenProcessPool(워커 비정상 종료)도 여기로 온다
                    print(f"  [warn] 곡선 계산 워커 실패 {tmn}: {repr(e)[:200]}. 주 프로세스에서 다시 계산한다", flush=True)
                    redo.append((tmn, None))
        for (t, m), names in redo:                                        # 실패한 대상·모드는 주 프로세스에서 차례로 다시 계산한다
            bad = []
            curves += curve_of(a, D5, stores, runs, floor, names=stores_of(t, m) if names is None else names, failed=bad)
            curve_failed += bad
    else:
        curves = curve_of(a, D5, stores, runs, floor, failed=curve_failed)
    curves = [c_ for c_ in curves if len(c_)]
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if len(curve):
        curve = curve.sort_values(["target", "mode", "method", "learner", "alpha", "placement", "lam", "n"]).reset_index(drop=True)
    t_curve = time.time() - t0
    conf = conformal_table(tms, a); H4.boot_weights.cache_clear()
    dist = distance_table(tms, a); H4.boot_weights.cache_clear()
    # ---------------- N4
    n4 = dict(splitdist=pd.DataFrame(), region=pd.DataFrame())
    if sh_n4 or sh_n4a:
        s50 = load_stores([s_["npz"] for s_ in sh_n4]) if sh_n4 else {}
        s200 = load_stores([s_["npz"] for s_ in sh_n4a]) if sh_n4a else {}
        n4 = n4_tables(a, s50, s200, get_data_x(h40_args(a, "n4")) if sh_n4 else None, get_data_x(h40_args(a, "n4a")) if sh_n4a else None)
    tests = build_tests_x(a, tms, D5, floor, runs, conf=conf, n4=n4)
    H4.boot_weights.cache_clear()
    n4["region"] = pd.concat([v for v in (region_inference_main(tms, a), n4["region"]) if len(v)], ignore_index=True) \
        if (len(n4["region"]) or len(tms)) else n4["region"]
    sh_base = [s_ for s_ in sh_main if s_["tag"] == axis_tag(a, "base")]
    gate = gate_table(a, sh_base, a.GATE_DIR, a.GATE_TAG) if sh_base else pd.DataFrame()
    lg_aux = pd.DataFrame()
    if not a.SUFFIX:
        try:
            lg_aux = lg_aux_table(a, D5, floor)
        except Exception as e:                                            # noqa: BLE001
            print(f"  [warn] 본 실행 보조 열 계산 실패: {repr(e)[:200]}", flush=True)
    n_shap = {str(s_["unit"]): int(u.get("n_shap_rows", 0) or 0) for s_, u in zip(everything, units)}
    shp = [pd.read_csv(s_["shap"]) for s_ in sh_main if n_shap.get(str(s_["unit"]), 0) > 0 and s_["shap"].exists() and s_["shap"].stat().st_size > 1]
    shap = pd.concat(shp, ignore_index=True) if shp else pd.DataFrame()
    if len(shap):
        shap = shap.groupby(["target", "mode", "method", "learner", "n", "feature"], as_index=False).agg(
            mean_abs_shap=("mean_abs_shap", "mean"), sd_splits=("mean_abs_shap", "std"), n_splits=("split", "nunique"), n_lab=("n_lab", "mean"))
        shap["rank"] = shap.groupby(["target", "mode", "method", "n"]).mean_abs_shap.rank(ascending=False, method="min").astype(int)
    tt = H.timing_table(units)
    if len(tt):
        tt["axis_x"] = [str(u.get("axis", "")) for u in units for _ in (u.get("n_fit_detail") or {})]
    pref = a.TAG
    for df_ in (tests, curve):                                            # 재표집 횟수를 표에 적는다(허용 표지 없는 집계는 1,000 으로 낮춘 값이다)
        if len(df_):
            df_["nboot"] = int(a.nboot)
    curve.to_csv(O / f"{pref}_curve.csv", index=False); tests.to_csv(O / f"{pref}_tests.csv", index=False)
    gate.to_csv(O / f"{pref}_gate.csv", index=False); lg_aux.to_csv(O / f"{pref}_lg_aux.csv", index=False)
    n4["region"].to_csv(O / f"{pref}_region_inference.csv", index=False); n4["splitdist"].to_csv(O / f"{pref}_splitdist.csv", index=False)
    conf.to_csv(O / f"{pref}_conformal.csv", index=False); dist.to_csv(O / f"{pref}_distance.csv", index=False)
    shap.to_csv(O / f"{pref}_shap.csv", index=False); floor.to_csv(O / f"{pref}_floor.csv", index=False)
    if curve_failed:                                                      # 곡선 계산에 실패한 저장소도 실패 표에 남긴다
        cf = pd.DataFrame([dict(target=q["target"], mode=q["mode"], method="(곡선 계산)", fit_flag="curve_failed", reason=q["reason"], store=q["store"])
                           for q in curve_failed])
        failed = pd.concat([failed, cf], ignore_index=True)
    tt.to_csv(O / f"{pref}_timing.csv", index=False); failed.to_csv(O / f"{pref}_failed.csv", index=False)
    tg = pd.DataFrame([{k: v for k, v in u.items() if not isinstance(v, (dict, list))} for u in units])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    tg.to_csv(O / f"{pref}_targets.csv", index=False)
    splits_by_learner = {}
    for s_ in sh_g + sh_lg + [q for q in sh_main if q["part"] == "gpu"]:
        splits_by_learner.setdefault(f"{s_['tag']}|{s_['learner']}", set()).add(int(s_["split"]))
    n_unit_bad = Counter(str(u.get("status", "ok")) for u in units)
    fit_tot = Counter()
    for u in units:
        for k, v in (u.get("n_fit") or {}).items():
            fit_tot[f"{u.get('axis', u.get('part', ''))}:{k}"] += int(v)
    n_gate_fail = int((gate.passed.astype(str) != "True").sum()) if len(gate) else 0
    meta = dict(stage="H42/LGX", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A", git_commit=H.git_commit(), tag=a.TAG,
                args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")}, code_sha=code_sha_x(), code_sha_h40=h40_now,
                key_fields=H.KEY_COLS, n_units=len(units), unit_status=dict(n_unit_bad), n_fit=dict(fit_tot), n_fit_total=int(sum(fit_tot.values())),
                unit_elapsed_s_sum=float(sum(u.get("elapsed_s", 0) for u in units)), elapsed_s=round(elapsed, 1),
                summarize_s=round(time.time() - t0, 1), curve_s=round(t_curve, 1), summarize_workers=int(nw), n_rows=int(len(runs)),
                n_curve=int(len(curve)), n_tests=int(len(tests)), curve_failed=list(curve_failed),
                nboot_capped=getattr(a, "nboot_asked", None) is not None, nboot_asked=int(getattr(a, "nboot_asked", None) or a.nboot),
                n_tuned_missing=int(sum(1 for u in units if u.get("tuned_missing"))),
                n_failed_keys=int(len(failed)), shard_cfg=cfg_info, x6=x6, splits_by_learner={k: sorted(v) for k, v in splits_by_learner.items()},
                gate=dict(dir=str(a.GATE_DIR), tag=a.GATE_TAG, n_units=int(len(gate)), n_fail=n_gate_fail, tol_phys=GATE_TOL_PHYS, tol_ml=GATE_TOL_ML),
                floor=floor_meta, nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux), skipped=skipped or [],
                design=unit_cfg_x(a, "base")["design"],
                rules=dict(ci_main="h4_common.boot_delta_blocks(분할 안 채점 블록 재표집, 분할 분포 평균)과 h40.strat_mean. 재표집 10,000회",
                           ci_common="지역 블록 공통 재표집: 사용 분할 채점 블록의 합집합에 다중도를 한 번 뽑고 분할마다 자기 블록의 열을 읽는다",
                           verdict4="우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 그 밖에서 네 끝값이 [−δ, +δ] 안, 미결정 = 그 외",
                           p_eq="동등성 p = max(P(Δ* ≤ −δ), P(Δ* ≥ +δ)), 두 가중 가운데 큰 값. 4분 판정의 '동등'(95 % CI 가 한계 안)은 p_eq ≤ 0.025 에 "
                                "대응한다(우세·열세가 먼저 판정되므로 CI 가 한계 안이면서 0 을 제외하면 우세 또는 열세다)",
                           holm="항목(X1, X2, …)별 주 대비 묶음의 부트스트랩 p 에 m1_stats.holm. 주 대비 = 가설 판정에 들어가는 층화 평균 대비와 "
                                "L24 의 대상별 이중 차분(보조 n, 3지역 평균, 서술 행은 뺀다). 4분 판정이 나오지 않은 행(판정 불가, 행 없음)은 묶음에 "
                                "넣지 않는다. 동등성 대비(L9, L12, L14, L17, L26 의 주 대비)는 같은 항목 안에서 p_eq 에 Holm 을 따로 적용한다"
                                "(holm_p_eq, 보정 뒤 0.025 를 넘는 동등은 '보정 전 동등'). 보조 열이며 판정에 쓰지 않는다",
                           partial="판정 문구의 부분 표기: 풀 지역 수가 대상 지역 수보다 적으면 '부분(지역 k/N)', 판정에 쓴 대비의 분할이 기대 분할 "
                                   "수보다 적으면 '부분(분할 k/K)'(확인적 가설은 판정 불가), 판정할 수 없는 대비가 있으면 '부분(대비 k/m)' 또는 "
                                   "'판정 불가(대비 k/m)'. 기대 분할 수 = 중복이 아닌 유효 분할 수(h40.enumerate_units 와 같은 규칙)",
                           confirmatory="확인적 가설 L10, L12, L15, L29, L30(L19 는 h41). 판정할 수 없는 대비가 있으면 지지 문구를 쓰지 않는다",
                           small="우세 또는 열세이면서 |Δ| < 0.5 cm 인 대비는 small_note 열과 판정 문구에 '통계적으로 구별되나 크기는 0.5 cm 미만'을 붙인다",
                           phys_tr="결측 대체 민감도: p4_ku, p2_edaphic 의 공변량 결측을 원천 셀 중앙값으로 채워 다시 계산한 기준선(P0@ku_tr 등)과 "
                                   "F1k@tr(n 0, 10). load_base 의 두 열은 전체 셀 중앙값으로 채운 값이다",
                           effect="delta_cm, delta_pct_p0 = 100·Δ/RMSE_P0, frac_reducible = −Δ/(RMSE_P0 − 하한), small_effect = |Δ| < 0.5 cm",
                           l24="대상 단위 집계는 등록한 n(40, 160) 모두에서 판정이 나온 대상만 확정으로 센다. 판정 불가와 행 없음은 알 수 없는 "
                               "대상이고 기준(2개, 3개)을 이미 넘겼거나 넘길 수 없을 때만 판정한다",
                           tuned="선택 기준은 λ = 1.0 의 예측 RMSE(원천 지역 하나 제외, 지역 등가중). 동률이면 반복 × 2^깊이 가 작은 설정"))
    (O / f"{pref}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    if len(failed) or set(n_unit_bad) - {"ok"}:
        print(f"[summarize] 실패 기록: 조각 상태 {dict(n_unit_bad)} · 저장에서 빠졌거나 대체된 키 {len(failed):,} → {pref}_failed.csv", flush=True)
    if curve_failed:
        print(f"[summarize] 곡선 계산 실패 {len(curve_failed)}건(주 프로세스 재계산 뒤): {[q['store'] for q in curve_failed]}", flush=True)
    if len(gate):
        print(f"[summarize] 재현 점검: {int(len(gate)) - n_gate_fail}/{len(gate)} 단위 통과 → {pref}_gate.csv", flush=True)
    print(f"[summarize] 조각 {len(everything)} · runs {len(runs):,} · curve {len(curve):,}({t_curve:.0f}s, 프로세스 {nw}) · tests {len(tests):,} · "
          f"{time.time() - t0:.0f}s → {O}/{pref}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v):
        print(v[["test_id", "role", "verdict"]].to_string(index=False), flush=True)
    return dict(curve=curve, tests=tests, gate=gate, conformal=conf, distance=dist, shap=shap, floor=floor, meta=meta, failed=failed,
                splitdist=n4["splitdist"], region=n4["region"], lg_aux=lg_aux, timing=tt)


# ================================================================ 실행
def count_only(a, units, skipped):
    """학습 없이 적합 수와 추정 시간을 출력한다. 추정 = 학습기별 기준 시간 × (학습 행 수 / 기준 행 수)^지수 의 합(BASE_FIT_X).
    n4a 는 해석식만 있으므로 작업 단위 수만 센다."""
    rows, det, nrw, est = [], Counter(), Counter(), Counter()
    for u in units:
        axis, t, m, sp, lr = u
        if axis == "n4a":
            v = a.DX[axis][1].split_structure(t)[sp]
            rows.append(dict(axis=axis, target=t, mode=m, split=sp, part="cpu", learner="", n_A=v["n_A"], n_eval=v["n_eval"], nb_eval=v["nb_eval"],
                             n_src=0, valid=v["valid"], n_rows=1 + 2 * len(H.cells_of(a.DX[axis][0].N_GRID, a.DX[axis][0].DRAWS, v["n_A"])),
                             est_s=0.0, fit_total=0))
            continue
        if axis == "g45":
            r = H.run_unit(a.HG, "gpu", t, m, sp, lr, dry=True)
            r = dict(r, axis=axis, fit_total=int(sum(v for k, v in r.items() if k.startswith("fit_"))))
        else:
            r = run_unit_x(a, axis, t, m, sp, lr, dry=True)
        for k, v in r.pop("_detail").items():
            det[f"{axis}|{k}"] += v
        for k, v in r.pop("_rows").items():
            nrw[f"{axis}|{k}"] += v
        for k, v in r.pop("_est").items():
            est[f"{axis}|{k}"] += v
        rows.append(r)
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    print(f"[count-only] 부분 {a.part} · 축 {a.AXES} · 작업 단위 {len(units)} · 건너뜀 {len(skipped)}", flush=True)
    if not len(df):
        return df
    df["order"] = list(range(len(df)))
    df.to_csv(a.OUT / f"{a.TAG}_count_{a.part}.csv", index=False)
    by = df.groupby("axis", sort=False).agg(units=("split", "size"), targets=("target", "nunique"), fit_total=("fit_total", "sum"),
                                            rows=("n_rows", "sum"), est_h=("est_s", "sum"))
    by["est_h"] = (by.est_h / 3600).round(2)
    print(by.to_string(), flush=True)
    tab = []
    for k in det:
        ax, _, lr, mth = (k.split("|") + ["", "", ""])[:4]
        tab.append(dict(axis=ax, learner=lr, method=mth, n_fit=int(det[k]), rows_per_fit=round(nrw[k] / max(det[k], 1)), est_h=round(est[k] / 3600, 3)))
    tab = pd.DataFrame(tab)
    if len(tab):
        tab = tab.sort_values(["axis", "learner", "method"])
        tab.to_csv(a.OUT / f"{a.TAG}_count_{a.part}_detail.csv", index=False)
        bl = tab.groupby(["axis", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), est_h=("est_h", "sum"))
        print("[count-only] 축·학습기별 적합 수와 추정 누적 시간(프로세스 1개 기준. catboost·rf·catboost_tuned 의 기준 시간은 가정값이다)", flush=True)
        print(bl.to_string(index=False), flush=True)
    n_sel = 0
    if "n3" in a.AXES:
        pairs = sorted({(u[1], u[2]) for u in units if u[0] == "n3"})
        n_sel = int(sum(r_["n_fit"] for t, m in pairs for r_ in select_rows(a, t, m, dry=True)))
        print(f"[count-only] catboost_tuned 선택 단계: 대상 {len(pairs)} · 적합 {n_sel:,}(본 적합 수에 들어 있지 않다)", flush=True)
    tot_h = float(df.est_s.sum()) / 3600
    print(f"[count-only] 총 적합 {int(df.fit_total.sum()):,}(선택 단계 {n_sel:,} 별도) · 저장 행 {int(df.n_rows.sum()):,} · 추정 누적 {tot_h:.1f} h", flush=True)
    return df


def run_permitted(a):
    """학습을 하는 실행(스모크, 사전 점검, 본 실행)의 허용 여부. Rescale 작업 명령에서 LG_RESCALE=1 또는 --allow-local 을 준다."""
    return bool(a.allow_local) or os.environ.get("LG_RESCALE", "") == "1"


LOCAL_NBOOT_MAX = 1000                            # 허용 표지 없는 집계(--summarize-only)의 재표집 횟수 상한
LOCAL_COUNT_AXES = 6                              # 허용 표지 없는 --count-only 에서 축을 나누라고 안내하는 축 수


def limit_local(a):
    """허용 표지가 없는 실행(--count-only, --summarize-only)의 자원을 제한한다. 스레드는 finalize 가 1 로 낮춘다."""
    if run_permitted(a):
        return a
    if int(a.threads_asked) > 1:
        print(f"[local] 허용 표지가 없다: --threads {a.threads_asked} → 1", flush=True)
    if a.summarize_only:
        if int(a.workers) > 1:
            print(f"[local] 허용 표지가 없다: --workers {a.workers} → 1", flush=True)
        a.workers = min(int(a.workers), 1)
        if int(a.nboot) > LOCAL_NBOOT_MAX:
            print(f"[local] 허용 표지가 없다: --nboot {a.nboot} → {LOCAL_NBOOT_MAX}. 이 집계의 CI 와 판정은 등록한 재표집 횟수(10,000)의 값이 아니다"
                  f"(표의 nboot 열과 meta 의 nboot_capped 에 적는다)", flush=True)
            a.nboot_asked = int(a.nboot); a.nboot = LOCAL_NBOOT_MAX
    if a.count_only and (len(a.AXES) > LOCAL_COUNT_AXES or (len(a.AXES) > 1 and any(v in a.AXES for v in ("n4", "n4a")))):
        print(f"[local] --count-only 의 축이 {len(a.AXES)}개다. 스레드 1개에서 1분을 넘을 수 있다. --axes 로 나눠 한 번에 하나씩 돌린다"
              f"(예: base,x1,x2,x9 / n3,prox,x5,x9f,ridge / n4 / n4a)", flush=True)
    return a


def check_ext_tables(a):
    """실행 전에 확장 열의 표를 주 프로세스에서 한 번 읽는다(축과 무관). 표와 자료 판이 어긋나면 비용이 들기 전에 여기서 중단한다.
    x9 축은 연도 정합 도일 표, x9f 축은 셀 표지 표가 있어야 한다. 표가 없으면 그 부분(--part) 전체를 시작하지 않는다."""
    for ax in a.AXES:
        if ax == "g45" or ax not in a.DX:                                 # 분할 4·5 축은 h40.run_unit 을 그대로 부른다(확장 열을 쓰지 않는다)
            continue
        ext_tables(a, a.DX[ax][1], dry=False).require(ax)


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)                                     # 워커(spawn)가 물려받는다
    if a.GPUS and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        print("[warn] 부모 환경의 CUDA_VISIBLE_DEVICES 가 빈 문자열이다. --gpus 를 무시하고 CPU 로 돈다", flush=True)
        a.GPUS = []
    if a.part == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.write_label_flags:
        return write_label_flags(a)
    if a.write_tdd_matched:
        return write_tdd_matched(a)
    will_run = not (a.count_only or a.summarize_only)
    if will_run and not run_permitted(a):                                  # 자료를 읽기 전에 거부한다(스모크 포함)
        raise SystemExit("[거부] 학습을 하는 실행(스모크·사전 점검·본 실행)은 Rescale 작업에서 한다. Rescale 작업 명령에 LG_RESCALE=1 또는 "
                         "--allow-local 을 준다(사용자 지시 2026-09-29: 공유 서버의 CPU·GPU 를 쓰지 않는다)")
    limit_local(a)
    if a.summarize_only:
        out = summarize(a, 0.0, [])
        n_cf = len((out or {}).get("meta", {}).get("curve_failed", []))
        return dict(executed=[], skipped=[], resumed=[], n_fail=n_cf, n_curve_failed=n_cf)
    units, skipped = enumerate_x(a)
    units.sort(key=lambda u: priority_x(a, u))
    print(f"[data] 부분 {a.part} · 축 {a.AXES} · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 축별 "
          f"{dict(Counter(u[0] for u in units))}", flush=True)
    if a.count_only:
        count_only(a, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[])
    check_ext_tables(a)                                                   # 필요한 입력 표를 실행 전에 확인한다(없거나 어긋나면 부분 전체를 시작하지 않는다)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    resumed, todo = [], []
    for u in units:
        ok, why = unit_state_x(a, *u) if a.resume else (False, "")
        if ok:
            resumed.append(u)
            print(f"  [resume] 건너뜀 {unit_name_x(u)} (조각 있음, 상태 {why})", flush=True)
        else:
            todo.append(u)
            if a.resume and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name_x(u)} ({why})", flush=True)
    use_gpu = a.part == "gpu" and len(a.GPUS) > 0
    nproc = len(a.GPUS) * a.procs_per_gpu if use_gpu else max(a.workers, 1)
    inline = (not use_gpu) and a.workers <= 0
    argv_g = g45_argv(a) if any(u[0] == "g45" for u in todo) else None
    pairs = sorted({(u[1], u[2]) for u in todo if u[0] == "n3"}, key=lambda v: (confirm_grade("n3", v[0], v[1]), v))
    need = [v for v in pairs if not (a.resume and select_ready(a, [v]))]   # 선택 표가 이미 덮는 대상·모드는 다시 고르지 않는다
    sel_tasks = [("n3sel", t, m, 0, None) for t, m in need]
    if pairs and len(need) < len(pairs):
        print(f"  [resume] catboost_tuned 선택 표에 있는 대상 {len(pairs) - len(need)}개는 다시 고르지 않는다: {a.SELECT}", flush=True)
    print(f"[plan] 실행 {len(todo)} · 선택 작업 {len(sel_tasks)} · 재개로 건너뜀 {len(resumed)} · 프로세스 {nproc} · 스레드 {a.threads} · "
          f"장치 {'GPU ' + ','.join(a.GPUS) if use_gpu else 'CPU'} · epochs {a.epochs} · RealMLP epochs {a.realmlp_epochs}", flush=True)
    done, failed = [], []
    state = dict(sel_wait={(t, m) for t, m in need}, sel_failed=[])

    def log(u):
        done.append(u)
        print(f"  [{u.get('axis', '')}|{u['target']}|{u['mode']}|s{u['split']}{'|' + u['learner'] if u.get('learner') else ''}] 적합 {u['n_fit_total']} · "
              f"행 {u['n_rows']} · A {u['n_A']} · 채점 {u['n_eval']}/{u['nb_eval']}블록{'' if u['valid'] else '(무효 분할)'} · 원천 {u['n_src']} · "
              f"{u['elapsed_s']}s · 장치 '{u['device']}' · 상태 {u['status']}(실패 {u['n_fail']}) · 완료 {len(done)}/{len(todo)} · "
              f"누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {unit_name_x(u)}: {repr(e)[:300]}", flush=True)

    def sel_done(u, res):
        """선택 작업 하나가 끝났다. 표는 대상·모드마다 바로 쓴다(그 대상의 n3 단위가 나머지 선택 작업을 기다리지 않는다).
        실패한 대상·모드의 n3 단위는 catboost_tuned 없이 돌고 조각에 tuned_missing 을 남긴다."""
        state["sel_wait"].discard((u[1], u[2]))
        rows = list((res or {}).get("rows", []))
        if rows:
            df = write_select(a, rows)
            print(f"  [select] {u[1]}|{u[2]}: catboost_tuned 선택 표 {len(df)}행 → {a.SELECT} · 남은 선택 작업 {len(state['sel_wait'])}", flush=True)
        else:
            state["sel_failed"].append((u[1], u[2]))

    def handle(u, fn):
        try:
            res = fn()
        except BrokenProcessPool:
            raise
        except (Exception, SystemExit) as e:                              # noqa: BLE001  한 단위의 실패(워커 안의 SystemExit 포함)가 나머지를 막지 않게 한다
            fail(u, e)
            if u[0] == "n3sel":
                sel_done(u, None)
            return
        if u[0] == "n3sel":
            sel_done(u, res)
        else:
            log(res)

    def blocked(u):
        return u[0] == "n3" and (u[1], u[2]) in state["sel_wait"]          # 그 대상·모드의 선택 작업이 끝나야 n3 단위를 돌린다

    if inline and (todo or sel_tasks):
        _worker_init_x(argv, argv_g, None, a.threads)
        for u in sel_tasks + todo:
            handle(u, lambda u=u: _worker_run_x(*u))
    elif todo or sel_tasks:
        ctx = multiprocessing.get_context("spawn")                        # fork 후 OpenMP 충돌 회피

        def gpu_queue(n_slots=None):
            if not use_gpu:
                return None
            q = ctx.Queue()
            slots = [g for g in a.GPUS for _ in range(a.procs_per_gpu)]
            for g in (slots if n_slots is None else slots[:n_slots]):
                q.put(g)
            return q
        pending, attempt = list(sel_tasks) + list(todo), 0
        hits, suspects = Counter(), []                                    # 단위별 '풀이 깨질 때 제출되어 있던 횟수'와 따로 돌릴 단위
        while pending:
            broken, running, pool_ok = [], {}, True

            def pull():
                for i, u in enumerate(pending):
                    if blocked(u):
                        continue
                    return pending.pop(i)
                return None
            with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_worker_init_x,
                                     initargs=(argv, argv_g, gpu_queue(), a.threads)) as ex:
                while True:
                    while pool_ok and len(running) < 2 * nproc:
                        u = pull()
                        if u is None:
                            break
                        try:
                            running[ex.submit(_worker_run_x, *u)] = u
                        except BrokenProcessPool:
                            broken.append(u); pool_ok = False
                    if not running:
                        break
                    fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for f in fin:
                        u = running.pop(f)
                        try:
                            handle(u, f.result)
                        except BrokenProcessPool:                        # 워커 비정상 종료(메모리 부족 등): 남은 단위는 새 풀에서 다시 돈다
                            broken.append(u); pool_ok = False
            if not broken:
                if pending:                                               # 선택 작업이 끝나지 않아 막힌 단위(선택 작업이 따로 돌릴 단위로 빠진 경우)
                    if any(s_[0] == "n3sel" for s_ in suspects):
                        break
                    for u in pending:
                        fail(u, RuntimeError("catboost_tuned 선택 작업이 끝나지 않아 실행하지 못했다"))
                    pending = []
                break
            for u in broken:
                hits[u] += 1
            sus = [u for u in broken if hits[u] >= 2]                      # 풀이 깨질 때 두 번 제출되어 있던 단위는 원인 후보다. 마지막에 따로 돌린다
            suspects += sus
            broken = [u for u in broken if hits[u] < 2]
            if not sus:
                attempt += 1                                              # 원인 후보를 가려내지 못한 풀 재생성만 센다
            if attempt > a.pool_retries:
                for u in broken + pending:
                    fail(u, RuntimeError(f"프로세스 풀이 원인 단위를 가려내지 못한 채 {attempt}회 깨졌다. 재시도 상한 {a.pool_retries}"))
                pending = []
                break
            both = broken + pending
            pending = [u for u in both if u[0] == "n3sel"] + sorted([u for u in both if u[0] != "n3sel"], key=lambda u: priority_x(a, u))
            print(f"[pool] 워커 비정상 종료. 남은 {len(pending)} 단위로 풀을 다시 만든다(원인 후보 {len(suspects)} 단위는 마지막에 따로 돌린다. "
                  f"재시도 {attempt}/{a.pool_retries})", flush=True)
        tail = [u for u in suspects if u[0] == "n3sel"] + [u for u in suspects if u[0] != "n3sel"] + list(pending)
        for u in tail:                                                    # 원인 후보: 단위마다 새 풀(프로세스 1개)에서 돌린다. 여기서도 깨지면 그 단위만 실패다
            if blocked(u):
                fail(u, RuntimeError("catboost_tuned 선택 작업이 끝나지 않아 실행하지 못했다"))
                continue
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init_x,
                                         initargs=(argv, argv_g, gpu_queue(1), a.threads)) as ex:
                    handle(u, ex.submit(_worker_run_x, *u).result)
            except BrokenProcessPool as e:
                fail(u, RuntimeError(f"단위를 따로 돌린 풀에서도 워커가 비정상 종료했다: {repr(e)[:120]}"))
                if u[0] == "n3sel":
                    sel_done(u, None)
    if state["sel_failed"]:
        print(f"  [warn] catboost_tuned 선택이 실패한 대상·모드 {state['sel_failed']}: 그 대상의 n3 조각은 catboost_tuned 없이 돌았고 미완료로 남는다"
              f"(--resume 으로 다시 실행)", flush=True)
    n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
    n_part = sum(1 for u in done if u["status"] == "partial")
    n_tm = sum(1 for u in done if u.get("tuned_missing"))
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {n_part} · 선택 설정 없는 n3 조각 {n_tm} · {time.time() - t0:.0f}s", flush=True)
    n_cf = 0
    if not a.no_summarize:
        out = summarize(a, time.time() - t0, skipped)
        n_cf = len((out or {}).get("meta", {}).get("curve_failed", []))

    def ident(u):
        return (u.get("axis", ""), u["target"], u["mode"], u["split"]) + ((u["learner"],) if u.get("learner") else ())
    return dict(executed=[ident(u) for u in done], skipped=skipped, resumed=[tuple(v for v in u if v is not None) for u in resumed],
                n_fail=n_fail + n_tm + n_cf, n_partial=n_part, n_tuned_missing=n_tm, n_curve_failed=n_cf)


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
