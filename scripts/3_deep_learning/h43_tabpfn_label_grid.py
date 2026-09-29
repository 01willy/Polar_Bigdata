"""H43 · TabPFN 라벨 격자(LGT). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6C(개정 9 사전 등록)의 구현.

목적
  로컬 GPU 에서만 돌릴 수 있는 TabPFN 을 본 실행(LG, h40)과 같은 대상, 분할, 라벨 추출 seed, 채점으로 돌려 학습기 축의 빈칸을 메운다.
  짝 비교 대조군은 같은 컨텍스트 행으로 학습한 CatBoost(catboost_ctx)이고 두 학습기를 같은 프로세스에서 짝지어 적합한다.
  가설 L32–L37(모두 보조)의 대비는 LGT 조각 안에서 닫힌다. 본 실행 조각은 집계의 교차 비교(--cross-lg, 보조)에서만 읽는다.

동결 규칙
  h40, h42, h35, h41 과 src/polar 의 기존 모듈을 고치지 않는다. h40 은 importlib 로 파일 경로에서 읽어 sys.modules['h40_label_grid'] 에
  등록하고(모듈 최상위) 자료 적재(Data, get_data), 대상과 분할(build_ctx, enumerate_units), 라벨 추출(draw_cells, cells_of), 계수(ls_E),
  CatBoost 적합(cb_fit), 조각 이름과 설정 해시(shard_paths, cfg_hash, _atomic_text), 집계(TMx, contrast, build_curve, build_minn,
  timing_table)를 그대로 쓴다. h42 는 집계 함수 안에서만 읽는다(find_shards_x, make_tm, verdict4, G, TestBook, floor_table, region_stats).
  src/polar 에서는 h4_common(BlockStore, save_stores, load_stores, seed_of)만 직접 부른다.

기호
  s = e5_sqrt_tdd. 원천 행 S, 대상 A 풀, 선택 라벨 L(h40.draw_cells 의 sel), 채점 셀 B(B 블록의 eval_mask 셀). x = x25.
  E0 = 원천 전체의 최소제곱(h40.Ctx.E0). E_ls = 선택 라벨의 최소제곱. E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10, n = 0 또는 E_ls 비유한이면 E0.
  E0 와 원천 잔차 y − E0·s 는 부분 추출한 행이 아니라 원천 전체로 구한 값이다(h40.Ctx 의 E0, r0_src).

축(--axes)과 조각 tag
  main  lgt   대상 27, 분할 1–5, n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5, seed 2, 방법 D0·R0·R1 × 학습기 2종.
  sens  lgts  대상 12(h40.LEARNER_TARGETS), 분할 1–5, 추출 2, seed 2, 방법 D0·R1 × 학습기 2종. 변형(--sens)마다 한 요인만 바꾼다.
              d3   대상 라벨 행 3배 중복, 키의 alpha '3', n {3, 10, 40, 160}
              d10  대상 라벨 행 10배 중복, 키의 alpha '10', n {10, 40, 160}
              rid  지역 id 범주형 입력(26번째 열), 방법 이름 D0@rid·R1@rid, n {0, 10, 40, 160, 전량}
              tgt  원천 행 없이 대상 라벨 행만, 방법 이름 D0@tgt·R1@tgt, n {40, 160, 320, 1000, 전량} 가운데 실제 라벨 수 40 이상
  --smoke: GPU 1장, 대상 Russia_W x 와 Canada x, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1, 축 main·sens, 집계 재표집 1,000회. tag 뒤에 _smoke.

학습기
  tabpfn        tabpfn 8.0.7 의 TabPFNRegressor. 공개 v2 회귀 가중치(--model-path, 기본 tabpfn-v2-regressor.ckpt), n_estimators 8,
                random_state = 학습기 seed, device cuda, ignore_pretraining_limits True, memory_saving_mode False, 출력은 평균.
                입력은 x25 원값(float32, 결측은 NaN 그대로, 표준화 없음). 범주형 지정은 없다(rid 변형만 26번째 열을 지정한다).
  catboost_ctx  catboost_lo 와 같은 초모수(반복 200, 학습률 0.05, 깊이 3, l2 3. h40.cb_fit), random_seed = 학습기 seed, 입력 x25 원값.
                학습 행은 같은 (n, 추출, seed)의 TabPFN 컨텍스트 행과 같다. 중복 행은 행 반복으로 준다(표본 가중이 아니다).

방법과 키
  P0 = E0·s, P1 = E_n·s (해석식. 모든 조각, 모든 (n, 추출) 칸에 저장한다. n = 0 의 P1 은 E0 다)
  D0  목표 = 원천 y ∪ 대상 라벨 y. 예측 f(x). λ 1.0 한 키.
  R0  목표 = 원천 y − E0·s ∪ 대상 라벨 y − E0·s. 예측 E0·s + λ·g(x).
  R1  목표 = 원천 y − E0·s ∪ 대상 라벨 y − E_n·s. 예측 E_n·s + λ·g(x). n = 0 에서는 R0 의 g 를 그대로 쓴다(추가 적합 없음).
  λ ∈ {0.25, 0.5, 1.0} 은 같은 적합에서 사후 적용한다. 판정의 기준 λ 는 D0 1.0, R0·R1 0.25 다.
  키 = (method, learner, alpha, placement, n, draw, seed, lam). 전량은 n = −1. P0 = h40.P0_KEY, P1 = ('P1', 'none', '1', 'cell', n, d, −1, 0.0).
  세 방법은 같은 (n, 추출, seed)에서 같은 컨텍스트 행렬을 쓰고 목표만 다르다. 두 학습기는 같은 행렬과 같은 목표를 받는다.
  R2, D1, R3(유사라벨을 넣는 방법)은 넣지 않는다(계획서 §6C.3: 컨텍스트 상한 안에서 LG 와 같은 정의를 쓸 수 없다).

컨텍스트 규칙(계획서 §6C.4)
  상한 C = --ctx-max(10,000). 대상 행 묶음 T = 중복 배수 × 라벨 수. 원천 행 예산 m = min(원천 행 수, C − max(T, --ctx-reserve)).
  원천 부분 추출(ctx_order): 지역 이름 오름차순으로 지역마다 RandomState(seed_of('lgt-ctx', 대상, 모드, 분할, 학습기 seed)).permutation 을
  뽑고, 순위 p 의 행에 키 u = (p + 0.5)/c_j 를 준 뒤 (u, 지역 이름, p) 순으로 정렬해 앞에서 m행을 쓴다. 작은 예산의 집합은 큰 예산의
  집합에 포함된다. seed 는 n, 추출 번호, 방법, 학습기와 무관하다.
  T > C − --ctx-src-min 이면 라벨 셀을 (C − ctx_src_min) // 중복 배수 개로 무작위 부분 추출하고(seed_of('lgt-tsub', 대상, 모드, 분할, n, 추출))
  fit_flag 에 tsub 를 남긴다. 계수 E_n 은 이 경우에도 선택 라벨 전체로 구한다. 등록 범위에서는 일어나지 않는다.
  행 순서는 원천 행(정렬 순) 다음에 대상 행이다. 채점 셀은 한 번에 예측하고 CUDA 메모리 부족 예외가 난 적합만 --pred-chunk 행 묶음으로
  다시 예측한다(fit_flag 에 chunk).

누설 규약
  대상 라벨은 선택된 n개만 컨텍스트와 계수에 쓴다. 채점 셀 라벨은 채점에만 쓴다. 표준화와 결측 대체는 하지 않는다.
  rid 변형의 지역 부호는 전체 자료의 macro 이름을 정렬한 정수이고 대상 행과 채점 셀은 새 부호(부호 수)를 쓴다(라벨을 쓰지 않는다).

조각(<out-dir>/shards, 기본 data/processed/lgt/shards)
  <tag>__gpu__<대상>__<모드>__s<분할>_{runs.csv, blocksse.npz, cells.npz, unit.json}. 조각 하나에 두 학습기의 결과와 P0, P1 이 함께 들어간다.
  기록 순서는 runs.csv → blocksse.npz → cells.npz → unit.json 이고 모두 임시 파일 뒤 os.replace 다. unit.json 이 완료 표지다.
  실행 시작 때 이전 unit.json 과 cells.npz 를 지운다(--no-cells 재실행에서 세대가 다른 셀 파일이 남지 않게 한다. 셀 단위 자료를 읽는 코드는
  unit.json 의 has_cells 를 먼저 본다). --resume 은 runs, blocksse, unit 이 있고 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  status(계획서 §6C.8, 개정 12): 실패가 없으면 ok. 적합을 시도한 학습기 가운데 저장 키가 0 이거나 실패 적합(예외, 비유한 예측, 점검 실패)
  비율이 FAIL_RATIO_MAX(0.2)를 넘는 학습기가 있으면 failed(unit.json 의 failed_learners). 그 밖의 실패가 있으면 partial.
  같은 학습기의 학습기 실패(적합·예측 예외, 비유한 예측)가 FAIL_STREAK_MAX(5)회 연속되면 단위를 중단한다(unit.json 을 쓰지 않는다).
  예외 뒤 CUDA 문맥을 쓸 수 없으면 워커를 끝내 풀을 새로 만들게 한다.
  runs.csv 의 열 = h40 의 열 + ctx_set(main, d3, d10, rid, tgt), n_ctx, n_ctx_src, n_ctx_tgt, fit_s, ctx_sha(컨텍스트 행렬과 목표의 SHA-1 앞 12자).
  cells.npz: y, s, block, loc_id, lat, lon, keys(JSON [method, learner, alpha, placement, n, draw, seed, comp]), P(float32, 키 × 채점 셀),
    E(키별 앵커 계수), E0, coef_keys(JSON [n, draw])와 coef_E(E_n). comp = pred(D0 의 예측) 또는 g(잔차 성분). 예측은 E·s + λ·g 로 복원한다.
  비유한 예측이 있는 키는 BlockStore 에 넣지 않고 runs 에 n_nonfinite, fit_flag 로 남긴다. 적합 한 건의 예외는 그 키만 실패로 기록한다.
  ImportError 와 MemoryError 는 조각 전체를 실패로 둔다(unit.json 을 쓰지 않는다).

실행 제어(공유 서버 규칙, 사용자 지시 2026-09-29 밤)
  --gpus 는 필수다(빈 문자열이면 거부. CPU 로 TabPFN 을 돌리지 않는다). GPU 0–4 가 목록에 있으면 거부한다. 실행 직전에 nvidia-smi 로
  메모리 사용을 확인해 --gpu-mem-max-mib(기본 0 MiB)를 넘는 GPU 는 빼고 경고한다. GPU 하나에 워커 프로세스 하나(spawn)다.
  워커는 CUDA_VISIBLE_DEVICES 를 자기 GPU 하나로 두고 OMP, MKL, OpenBLAS, NUMEXPR, torch, CatBoost 의 스레드를 --threads 로 제한한다.
  실행은 --threads 2–4 만 허용한다. 가중치 파일이 없으면 실행하지 않는다(내려받기 시도 방지). --count-only 와 --summarize-only 는
  GPU 없이 스레드 1–2 로 돈다. 실행 순서는 (1) 주 설정·주 4지역 (2) 민감도·주 4지역 (3) 주 설정·Alaska x (4) 주 설정·나머지 학습기 축 대상
  (5) 주 설정·나머지 (6) 민감도·나머지이고 묶음 안에서는 분할 번호, 큰 대상 순이다.
  검증 지적 반영(개정 12):
  - CUDA_DEVICE_ORDER = PCI_BUS_ID 를 모듈 최상위와 워커에서 둔다. 부모는 torch.cuda 를 부르지 않는다(CUDA 확인은 짧은 자식 프로세스).
  - 워커는 단위를 시작할 때마다 배정 GPU 를 nvidia-smi 로 다시 본다. 첫 CUDA 초기화 전에는 메모리 사용이 --gpu-mem-max-mib 이하이고 계산
    프로세스가 없어야 한다. 그 뒤에는 자기 PID 가 아닌 계산 프로세스가 없어야 한다. 걸리면 점유 표지(run_<tag>/gpu_busy__<번호>.json)를 남기고
    워커를 끝낸다. 첫 초기화 직후에는 자기 PID(없으면 torch UUID)가 있는 GPU 가 배정 번호의 GPU 와 같은지 확인하고 다르면 실행 전체를 멈춘다.
  - 풀이 깨지면 표지가 남은 GPU 를 빼고 nvidia-smi 로 다시 확인한 뒤 새 풀을 만든다. 실행 중 표지(run_<tag>/running__*.json)로 원인 단위를
    가리고 풀을 POOL_CRASH_MAX(2)회 깬 단위만 격리한다. 이번 실행(run_id)에서 이미 완료된 조각은 다시 돌리지 않는다.
  - 중단: Ctrl-C, SIGTERM, SIGHUP 은 대기 작업을 취소하고 워커를 종료한다. 워커는 SIGINT 를 무시하고 부모가 죽으면 SIGTERM 을 받는다
    (prctl PR_SET_PDEATHSIG). 적합 사이에 부모 PID 가 바뀌었으면 스스로 끝난다. 워커 PID 와 배정 GPU 는 run_<tag>/worker__<PID>.json 에 남는다.
  - 실행 잠금(run_<tag>/lock.json): 같은 tag 의 살아 있는 실행이 있으면 거부한다. 완료 조각이 있는데 --resume 도 --overwrite 도 없으면 거부한다
    (스모크 tag 는 예외).
  - 종료 코드: 중단 130, 실패(단위 예외, status failed, 집계 실패) 1, partial 만 있으면 2, 그 밖은 0. 실행 상태는 <tag>_run_status.json 에 남는다.
    본 실행 뒤 두 번째 통과는 --resume --rerun-partial 로 돌린다.
  - CatBoost 는 학습과 채점의 Pool 을 thread_count = --threads 로 직접 만든다. torch inter-op 스레드 1, TabPFN n_preprocessing_jobs = 1 을 명시한다.
  - CUDA 메모리 부족 뒤의 묶음 재예측은 except 블록 밖에서 gc.collect() 와 캐시 비우기 뒤에 한다.

집계(--summarize-only)
  lgt, lgts 조각을 읽어 곡선(h40.build_curve + 4분 판정 열), 최소 n(h40.build_minn), 가설 표(L32–L37, h42.TestBook)를 만든다.
  산출: <tag>_curve.csv, _minn.csv, _tests.csv, _cross.csv, _gate.csv, _timing.csv, _failed.csv, _targets.csv, _meta.json, _count.csv.
  실패(저장하지 못한 키, 상태가 ok 가 아닌 조각, 곡선 계산 실패)가 남으면 종료 코드 1 이다.
  교차 비교(--cross-lg)는 본 실행 cpu 조각과 P0·P1 을 대조한 뒤(허용 차 1e-9 cm, <tag>_gate.csv) 통과한 단위에만 catboost_lo 키를 병합한다.
  LGT 의 P0, P1 키 가운데 기준 조각에 없는 키가 있으면(열 n_missing_ref) 통과로 두지 않는다.
  <tag>_cross.csv 의 모든 행은 '교차 환경(보조)'이고 <tag>_tests.csv 에 넣지 않는다.
  집계의 조작적 정의(계획서 §6C.6, 개정 12. 가설과 판정 문구는 바꾸지 않았다):
  - 학습기 짝 대비(L32 의 R1·D0·R0 대비, L37 의 R1[T] − R1[C])는 두 학습기에 모두 있는 (method, alpha, placement, n, draw, seed, lam) 키만
    쓴다(pair_store). 뺀 키 수는 대비 행의 n_unpaired 열과 <tag>_meta.json 의 n_unpaired_keys 에 적는다. 대비 행의 key_set 열이 키 집합이다.
  - L37 의 안정성 분류는 주 기준(주 설정의 추출 5개)과 보조 기준(주 설정의 추출 0–1, 민감도 축과 같은 추출 수)으로 함께 계산해
    stability, stability_d01, stability_same 열에 병기한다. 판정 문구의 분할 완결성 표기에는 기준 행도 넣는다.
    tgt 의 변형 − 주 설정 대비 문장은 안정성 판정과 따로 판정 행(clause '변형 − 주 설정')에 적는다.
  - L32 에서 기준 λ 0.25 의 대비가 모두 동등이고 λ 1.0 에 우세 또는 열세가 있으면 '기준 λ 0.25 에서는 동등, λ 1.0 에서는 차이가 있다(n, 방향).
    동등으로 쓰지 않는다'로 적는다. L32 의 D0, R0 보조 대비는 모든 n 이 맹검이다(계획서 §6C.7: TabPFN 의 D0, R0 은 기존 결과가 없다).
  - 스모크는 곡선, 최소 n, 판정 표, 교차 표를 쓰지 않고 판정 행과 Δ 를 화면에 내지 않는다. 경로 점검(smoke_report), 적합 시간, 구조 수만 낸다.

명세(implementation_spec)와 다르게 구현한 점
  1. 시험 파일 이름은 tests/test_h43_tabpfn.py 다(과제 지시). 명세는 tests/test_h43_lgt.py 였다.
  2. --threads 의 기본값은 2 다(구현 규칙). 명세는 4 였다. 시간 추정(계획서 §6C.10)은 4스레드 기록에서 나온 값이다.
  3. GPU 를 빼는 기준은 --gpu-mem-max-mib 인자로 두고 기본값을 0 MiB 로 했다(계획서 §6C.8, 공유 서버 규칙). 구현 규칙의 500 MiB 보다 엄격하다.
  4. 셀 단위 저장(cells.npz, float32)을 조각에 더했다(구현 규칙). 명세의 조각은 세 파일이었다. --no-cells 로 끌 수 있고 --resume 의
     완료 조건에는 넣지 않는다.
  5. 가중치 경로는 h43 이 절대 경로로 바꿔 TabPFN 에 넘긴다. 파일 이름만 주면 TabPFN 캐시 디렉터리(환경 변수 TABPFN_MODEL_CACHE_DIR,
     XDG_CACHE_HOME, 없으면 ~/.cache/tabpfn)에서만 찾는다. tabpfn 8.0.7 은 파일 이름만 받으면 현재 작업 디렉터리를 먼저 본다.
  6. Holm 묶음의 주 대비(primary)는 계획서 §6C.6 대로 L32 의 기준 λ 대비 5개, L33 의 6개, L34 의 2개, L35 의 1개다. L32 의 λ 1.0 대비,
     L36, L37, 병기 대비는 primary 가 아니다. 명세는 등록 대비 전부를 primary 로 적었다.
  7. --gpus 목록은 번호가 큰 순으로 정렬해 쓴다(스모크는 가장 큰 번호 하나).
  8. --targets 를 주면 민감도 축은 그 대상과 학습기 축 12대상의 교집합만 돈다(등록 범위 밖의 민감도 조각을 만들지 않는다).
  9. 워커는 TABPFN_DISABLE_TELEMETRY=1 을 둔다(적합마다 나가는 외부 통신을 막는다. 예측값과 무관하다).
  10. 인자 --rerun-partial(--resume 에서 일부 적합이 실패한 조각도 다시 실행), --dry-build(--count-only 에서 컨텍스트 행렬을 실제로 만들어
      크기와 목표를 확인), --no-cells 를 더했다.
  11. L33 의 '우세인 최소 n' 에서 전량은 격자의 마지막 순서로 두고 'n ≤ 40' 조건에는 넣지 않는다(전량의 실제 라벨 수가 지역마다 다르다).
  12. L37 의 기준 행은 주 설정의 추출 5개를 쓴 대비이고 변형 행은 민감도의 추출 2개를 쓴 대비다. 변형 − 주 설정 대비는 공통 추출 번호로
      짝짓는다(h40.contrast 의 규칙). 추출 수의 차이를 보기 위해 추출 0–1 로 제한한 보조 기준 행을 함께 계산한다(개정 12).
  13. 검증 지적(2026-09-30)에 따라 실행 제어(잠금, GPU 재확인, 중단 처리, 풀 재생성의 원인 단위 격리, 종료 코드)와 조각 상태 규칙을 더했다.
      위 '실행 제어'와 '조각' 항목에 적었다.

확인한 것(코드 읽기, tabpfn 8.0.7)
  - 가중치 경로: model_loading.resolve_model_path 는 파일 이름만 받으면 현재 작업 디렉터리, 그다음 캐시 디렉터리를 본다(위 5번).
  - CUDA 메모리 부족 예외: predict 는 torch.OutOfMemoryError 를 tabpfn.errors.TabPFNCUDAOutOfMemoryError 로 바꿔 올린다.
    is_cuda_oom 은 예외 클래스 이름(OutOfMemoryError 계열)과 RuntimeError 의 'out of memory' 문구를 함께 본다.
  - 범주형 추론: categorical_features_indices 를 빈 목록으로 줘도 컨텍스트가 100행을 넘으면 고유값이 4개 미만인 수치 열을 범주형으로
    추론한다(preprocessing.modality_detection, inference_config 의 MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE = 100,
    MIN_UNIQUE_FOR_NUMERICAL_FEATURES = 4). 고유값 1개 이하인 열은 상수 열로 분류된다. 추론은 적합마다(컨텍스트마다) 다시 정해진다.
    x25 에 그런 열이 있으면 TabPFN 안에서 범주형으로 처리된다. --count-only 가 해당 열을 출력하고, 실행은 적합 뒤의 분류
    (inferred_feature_schema_ 의 범주형·상수 열 색인)를 변형과 컨텍스트 크기(100행 이하, 초과)별로 세어 unit.json 의 tabpfn_schema 에 남긴다.
  - h42.TestBook 이 읽는 인자 속성은 delta_eq, delta_eq_aux 다. 재표집 횟수는 TMx 의 nboot 에서 읽는다. h42 는 import 때 파일을 쓰지 않는다.

구현 뒤 --count-only 로 확인한 값(2026-09-29, 스레드 1개, 약 22초. --dry-build 를 더하면 약 25초)
  - 작업 단위 172(main 117, sens 55), 건너뛴 분할 23. TabPFN 적합 수는 main 16,800, d3 1,424, d10 1,000, rid 1,440, tgt 1,108 로 설계 단계
    값(계획서 §6C.5)과 같다. catboost_ctx 도 같은 수다. 추정 누적 시간은 TabPFN 35.0 GPU-h, CatBoost 1.9 h 다.
  - 컨텍스트 행 수의 최댓값은 10,000, 대상 행 부분 추출(tsub)은 0건이다. --dry-build 의 행렬 점검 실패는 0건이다.
  - 지역별 행 수와 비례 배분 m·c_j/N 의 차이는 최댓값 2.78행이다. 계획서 §6C.4 는 '1행 이내'로 적었으나 정렬 키 규칙에서 나오는 상한은
    0.5·(1 − w) + w·(J − 1)/2 다(w = 그 지역의 비중, J = 원천 지역 수). 규칙은 계획서대로 구현했고 문구만 실제와 다르다.
  - 고유값이 4개 미만인 x25 열은 cci_valid 하나다(TabPFN 이 범주형으로 추론한다. CatBoost 는 수치로 쓴다).

스모크(2026-09-30 01:29–01:36, GPU 9, 스레드 2, --gpu-mem-max-mib 2, 4단위 98적합, 6분 46초, 종료 코드 0)로 확인한 것
  - 4단위 모두 status ok, 저장하지 못한 키 0, 짝이 없는 키 0. 같은 컨텍스트 행렬(짝 51), n = 0 의 R1 = R0(키 12), CatBoost Pool 경로 차 0.0 cm,
    장치 대응(자기 PID), 실제 n_estimators 8, 잠금 해제와 GPU 반환. 집계 경로(summarize, build_tests_t)가 끝까지 돌았다(스모크 판정 행 11개는
    모두 판정 불가: 2대상·분할 1 이라 풀 조건을 채우지 못한다).
  - 적합 1건: TabPFN 5.6–6.3 s(컨텍스트 9,000–9,400행, 채점 12–393행), tgt 0.75–0.86 s, 워커 첫 적합 12.6–19.4 s. CatBoost 0.33–0.47 s.
    추정식 대비 실측 비 TabPFN 0.92, CatBoost 1.09. GPU 메모리 최댓값 할당 1,619 MiB, 예약 5,534 MiB.
  - 묶음 예측과 한 번 예측의 차는 최대 0.056 cm 로 허용 차 0.001 cm 를 넘었다. tabpfn 기본값(inference_precision 'auto' → CUDA 혼합 정밀도)과
    묶음 크기에 따른 계산 순서 차이로 보이나 실험으로 확인하지 않았다. 묶음 재예측은 CUDA 메모리 부족 때만 쓰이고 표지(chunk)가 남는다.
  - 빈 GPU 의 nvidia-smi 메모리 표시가 2 MiB 였다(계산 프로세스 없음). 기본값 --gpu-mem-max-mib 0 이면 빈 GPU 도 빠지므로 실행 때 표시값을 준다.

확인하지 못한 것
  pytest 는 이 단계에서 실행하지 않았다(워크플로 규칙: 로컬 실행은 py_compile, pyflakes, --count-only, 스모크 1회). 시험 파일은 작성만 했다.
  채점 6,000–7,300행(Alaska x)의 시간과 GPU 메모리, 스레드 4개일 때의 적합 시간, 같은 입력의 GPU 반복 예측이 같은 값을 내는지, 풀 재생성과
  중단 처리의 실제 동작(워커 비정상 종료, 신호)은 실행으로 확인하지 못했다. CatBoost 의 스레드 수에 따른 재현성(2스레드와 4스레드)과
  재표집 10,000회의 집계 시간도 확인하지 못했다.

실행(ROOT)
  적합 수:   python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --count-only --threads 1
  스모크:    python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --smoke --gpus 9 --threads 2
  본 실행:   python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --gpus <실행 직전 nvidia-smi 로 확인한 빈 GPU, 예: 9,7,6,5> --threads 4
             --resume --no-summarize            (첫 실행도 --resume 을 준다. 종료 코드 0 완료, 2 partial 있음, 1 실패, 130 중단)
  두 번째 통과: 같은 명령에 --rerun-partial 을 더한다(partial 조각과 실패 조각만 다시 돈다)
  집계:      python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --summarize-only --threads 2
  호출자는 실행 중에도 nvidia-smi 로 GPU 점유를 주기적으로 확인한다(워커는 단위 시작 때만 확인한다).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

THREADS_DEFAULT = 2                                 # --threads 기본값(구현 규칙)
THREADS_MAX = 4
THREADS_RUN_MIN = 2                                 # 실행은 2–4 만 허용한다
THREADS_LIGHT_MAX = 2                               # --count-only, --summarize-only 의 상한
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(default=THREADS_DEFAULT):
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽어 1–4 로 자른다). 학습 없는 실행은 2 이하로 둔다."""
    av = sys.argv
    val = default
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        n = int(val)
    except (TypeError, ValueError):
        n = int(default)
    n = max(1, min(n, THREADS_MAX))
    if "--count-only" in av or "--summarize-only" in av:
        n = min(n, THREADS_LIGHT_MAX)
    return str(n)


for _v in THREAD_VARS:
    os.environ[_v] = _peek_threads()                # setdefault 가 아니라 대입이다(부모 환경의 큰 값을 물려받지 않는다)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # CUDA 장치 번호를 nvidia-smi 번호(PCI 버스 순)와 맞춘다(torch 를 부르기 전)

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, filename):
    """파일 경로에서 모듈을 읽어 sys.modules 에 등록한다. 이미 등록되어 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_h40():
    """h40 을 읽는다. 모듈 최상위에서 부른다(spawn 워커가 주 모듈을 다시 읽을 때 등록되어야 한다)."""
    return _load_module("h40_label_grid", "h40_label_grid.py")


def _load_h42():
    """h42 를 읽는다. 집계 함수 안에서만 부른다. h42 는 스레드 환경 변수를 setdefault 로만 건드리므로 h43 의 값이 유지된다."""
    return _load_module("h42_label_grid_ext", "h42_label_grid_ext.py")


H = _load_h40()

import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                             # noqa: E402

# ================================================================ 고정 설계값(계획서 §6C. 바꾸면 사전 등록에서 벗어난다)
AXES = ("main", "sens")
AX_T = dict(main=dict(sfx="", draws=5, methods=("D0", "R0", "R1")),
            sens=dict(sfx="s", draws=2, methods=("D0", "R1")))
LEARNERS_T = ("tabpfn", "catboost_ctx")
TP, CBX = LEARNERS_T
FULL_GRID = "0,3,10,40,160,320,1000,all"
MAIN_SET = dict(dup=1, alpha="1", sfx="", rid=False, tgt=False, min_lab=0)
SENS_SETS = dict(
    d3=dict(dup=3, alpha="3", sfx="", rid=False, tgt=False, min_lab=0, grid=(3, 10, 40, 160)),
    d10=dict(dup=10, alpha="10", sfx="", rid=False, tgt=False, min_lab=0, grid=(10, 40, 160)),
    rid=dict(dup=1, alpha="1", sfx="@rid", rid=True, tgt=False, min_lab=0, grid=(0, 10, 40, 160, -1)),
    tgt=dict(dup=1, alpha="1", sfx="@tgt", rid=False, tgt=True, min_lab=40, grid=(40, 160, 320, 1000, -1)))
RESERVED_GPUS = (0, 1, 2, 3, 4)                     # 남겨 두는 GPU(공유 서버 규칙)
DESIGN_FITS = dict(main=16800, d3=1424, d10=1000, rid=1440, tgt=1108)      # 설계 단계의 TabPFN 적합 수(계획서 §6C.5)
NB_SEEN = (3, 10, 40)                               # b4 에서 방향을 본 n(비맹검)
GATE_TOL = 1e-9                                     # 재현 점검의 허용 차(cm)
SMOKE_CHUNK_TOL = 1e-3                              # 스모크: 묶음 예측과 한 번 예측의 허용 차(cm)
TRACE = None                                        # 시험용: 리스트를 넣으면 모든 적합의 컨텍스트 행렬과 정보를 기록한다
FAIL_STREAK_MAX = 5                                 # 같은 학습기의 연속 실패(적합·예측 예외, 비유한 예측) 상한. 이르면 단위를 중단한다
FAIL_RATIO_MAX = 0.2                                # 학습기별 실패 적합 비율이 이 값을 넘으면 조각 상태를 failed 로 둔다
POOL_CRASH_MAX = 2                                  # 실행 중에 풀을 이 횟수만큼 깬 단위는 격리한다(실패로 기록하고 다시 돌리지 않는다)
GPU_SETTLE_S = 30                                   # 풀을 다시 만들 때 종료한 워커의 GPU 메모리가 풀리기를 기다리는 최대 시간(s)
AUX_DRAWS = (0, 1)                                  # L37 보조 기준 행의 추출 번호(민감도 축과 같은 추출 수)
EXIT_FAIL, EXIT_PARTIAL, EXIT_INTERRUPT = 1, 2, 130
TAG_ABORT, TAG_MISMATCH = "[적합 중단]", "[GPU 불일치]"   # 워커 예외의 문구 표지(워커 밖으로는 내장 예외로만 넘긴다)
RUN_COLS = ["target", "mode", "parent", "split", "part", "axis", "method", "learner", "alpha", "placement", "n", "n_lab", "draw", "seed", "lam",
            "rmse_cm", "rmse_beq_cm", "bias_cm", "E_used", "alpha_sel", "n_blocks_lab", "n_nonfinite", "fit_flag",
            "ctx_set", "n_ctx", "n_ctx_src", "n_ctx_tgt", "fit_s", "ctx_sha"]


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H43 TabPFN 라벨 격자(LGT)")
    ap.add_argument("--axes", default="main,sens", help="쉼표 목록: main(주 설정), sens(민감도)")
    ap.add_argument("--sens", default="d3,d10,rid,tgt", help="민감도 변형의 쉼표 목록")
    ap.add_argument("--targets", default="", help="쉼표 목록. '이름' 또는 '이름:모드'. 기본: main 은 계획서 §1 의 27대상, sens 는 학습기 축 12대상")
    ap.add_argument("--splits", type=int, default=5, help="half_split_blocks split_seed 1..K")
    ap.add_argument("--n-grid", default=FULL_GRID, help="sens 는 변형별 고정 격자와의 교집합을 쓴다")
    ap.add_argument("--draws", type=int, default=0, help="0 = 축 기본값(main 5, sens 2). 주면 축 기본값과 비교해 작은 쪽")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--ctx-max", type=int, default=10000, help="컨텍스트 총 행 수의 상한 C")
    ap.add_argument("--ctx-reserve", type=int, default=1000, help="대상 행 몫으로 비워 두는 행 수 R(원천 예산 = C − max(T, R))")
    ap.add_argument("--ctx-src-min", type=int, default=1000, help="원천 행의 최소 몫(T 가 C − 이 값을 넘으면 대상 행을 부분 추출)")
    ap.add_argument("--n-est", type=int, default=8, help="TabPFN n_estimators")
    ap.add_argument("--model-path", default="tabpfn-v2-regressor.ckpt", help="TabPFN 가중치. 파일 이름만 주면 TabPFN 캐시 디렉터리에서 찾는다")
    ap.add_argument("--pred-chunk", type=int, default=4096, help="CUDA 메모리 부족 때 다시 예측하는 묶음의 행 수")
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--gpus", default="", help="쉼표 목록(예: 9,7,6,5). 실행에 필수다. GPU 하나에 워커 하나")
    ap.add_argument("--gpu-mem-max-mib", type=int, default=0, help="실행 직전 메모리 사용이 이 값을 넘는 GPU 는 뺀다")
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT, help="프로세스당 스레드 수. 실행은 2–4 만 허용한다")
    ap.add_argument("--out-dir", default="data/processed/lgt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="lgt")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--cross-lg", action="store_true", help="집계: 본 실행 조각과의 교차 비교(보조)와 재현 점검을 켠다")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="본 실행(LG) 산출 디렉터리. 읽기 전용")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--resume", action="store_true", help="설정 해시가 같고 실패가 아닌 조각은 건너뛴다")
    ap.add_argument("--overwrite", action="store_true", help="완료 표지가 있는 조각이 있어도 --resume 없이 전부 다시 실행해 덮어쓴다(없으면 거부)")
    ap.add_argument("--rerun-partial", action="store_true", help="--resume 에서 일부 적합이 실패한 조각(status partial)도 다시 실행한다")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 작업 단위, 칸, 적합 수와 추정 시간만 낸다")
    ap.add_argument("--dry-build", action="store_true", help="--count-only 에서 컨텍스트 행렬과 목표를 실제로 만들어 크기를 확인한다(학습 없음)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true", help="실행 뒤 집계를 생략한다(조각만 남긴다)")
    ap.add_argument("--no-cells", action="store_true", help="셀 단위 저장(cells.npz)을 생략한다")
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="집계: 조각 사이 설정 해시가 달라도 진행한다(기본은 중단)")
    ap.add_argument("--pool-retries", type=int, default=2, help="진전(단위 완료 또는 격리) 없이 풀을 연속으로 다시 만드는 횟수의 상한")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(path, base=ROOT):
    return Path(path) if os.path.isabs(str(path)) else Path(base) / str(path)


def tabpfn_cache_dir():
    """TabPFN 의 캐시 디렉터리(tabpfn.model_loading.get_cache_dir 의 리눅스 규칙과 같다)."""
    v = os.environ.get("TABPFN_MODEL_CACHE_DIR", "").strip()
    if v:
        return Path(v)
    x = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(x) if x else Path.home() / ".cache") / "tabpfn"


def resolve_model(path_txt):
    """가중치 파일의 절대 경로. 절대 경로는 그대로, 디렉터리가 있는 상대 경로는 ROOT 기준, 파일 이름만 있으면 TabPFN 캐시 디렉터리."""
    p = Path(str(path_txt)).expanduser()
    if p.is_absolute():
        return p
    if p.parent != Path("."):
        return ROOT / p
    return tabpfn_cache_dir() / p.name


def finalize(a):
    a.SUFFIX = "_smoke" if a.smoke else ""
    a.TAG = a.tag + a.SUFFIX
    a.AXES = [v.strip() for v in str(a.axes).split(",") if v.strip()]
    bad = [v for v in a.AXES if v not in AXES]
    if bad:
        raise SystemExit(f"알 수 없는 축: {bad} (가능: {list(AXES)})")
    a.SENS = [v.strip() for v in str(a.sens).split(",") if v.strip()]
    bad = [v for v in a.SENS if v not in SENS_SETS]
    if bad:
        raise SystemExit(f"알 수 없는 민감도 변형: {bad} (가능: {list(SENS_SETS)})")
    if a.smoke:                                                     # 스모크: 대상 2, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1
        a.splits = 1; a.n_grid = "0,10,40,all"; a.draws = 1; a.seeds = 1
        a.targets = a.targets or "Russia_W:x,Canada:x"
        a.nboot = min(int(a.nboot), 1000)
    a.threads_asked = int(a.threads)
    light = bool(a.count_only or a.summarize_only)
    a.threads = max(1, min(int(a.threads), THREADS_LIGHT_MAX if light else THREADS_MAX))
    try:
        a.GPUS = sorted({int(g) for g in str(a.gpus).split(",") if g.strip() != ""}, reverse=True)      # 번호가 큰 것부터 쓴다
    except ValueError:
        raise SystemExit(f"--gpus 는 정수의 쉼표 목록이다: '{a.gpus}'")
    a.N_USER = H._n_list(a.n_grid)
    a.LAMS = [float(v) for v in str(a.lams).split(",") if v.strip()]
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGDIR = _abs(a.lg_dir)
    a.SHARDS = a.OUT / "shards"
    a.RUN_DIR = a.OUT / f"run_{a.TAG}"                               # 실행 잠금, 워커 PID, 실행 중 표지, GPU 점유 표지
    a.RUN_ID = ""                                                    # main 이 실행마다 정한다(조각의 unit.json 에 기록)
    a.MODEL = resolve_model(a.model_path)
    a.save_cells = not a.no_cells
    a._ha = {}
    return a


def axis_tag(a, axis):
    return f"{a.tag}{AX_T[axis]['sfx']}{a.SUFFIX}"


def sens_grid(a, v):
    """변형 v 의 n 격자 = 변형의 고정 격자 ∩ --n-grid."""
    return [n for n in SENS_SETS[v]["grid"] if n in a.N_USER]


def axis_grid(a, axis):
    if axis == "main":
        return list(a.N_USER)
    have = {n for v in a.SENS for n in sens_grid(a, v)}
    return [n for n in H._n_list(FULL_GRID) if n in have]


def axis_targets(a, axis):
    """축의 대상 목록. None 은 h40 의 기본값(계획서 §1 의 27대상)이다."""
    want = H._pairs(a.targets, ["i", "x"]) if a.targets else None
    if axis == "main":
        return want
    base = list(H.LEARNER_TARGETS)
    return base if want is None else [p for p in base if p in want]


def _grid_txt(grid):
    return ",".join("all" if n == -1 else str(int(n)) for n in grid)


def h40_args_t(a, axis):
    """축 하나의 h40 형식 인자(HA). part 를 cpu 로 주는 이유는 h40.enumerate_units 가 cpu 부분에서 학습기 축 제한 없이
    (대상, 모드, 분할)을 열거하기 때문이다. 조각 이름의 부분은 gpu 다(shard_paths 에 따로 준다)."""
    if axis in a._ha:
        return a._ha[axis]
    sp = AX_T[axis]
    draws = min(int(a.draws), sp["draws"]) if a.draws else sp["draws"]
    grid = axis_grid(a, axis)
    pairs = axis_targets(a, axis)
    argv = ["--part", "cpu", "--tag", axis_tag(a, axis), "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", a.subregion_map,
            "--splits", str(int(a.splits)), "--n-grid", _grid_txt(grid) or "0", "--draws", str(draws), "--seeds", str(int(a.seeds)),
            "--threads", str(int(a.threads)), "--kappa", str(float(a.kappa)), "--lams", ",".join(str(v) for v in a.LAMS),
            "--cb-iters", str(int(a.cb_iters))]
    if pairs:
        argv += ["--targets", ",".join(f"{t}:{m}" for t, m in pairs)]
    HA = H.parse_args(argv)
    if not grid:
        HA.N_GRID = []
    if pairs is not None and not pairs:
        HA.TARGETS = []
    HA.AXIS = axis
    a._ha[axis] = HA
    return HA


# ================================================================ 컨텍스트 구성
def ctx_order(macro_src, target, mode, split, seed):
    """원천 행 색인의 전체 순서(지역 비례 층화, 중첩 구조). 앞에서 m행을 쓰면 지역별 행 수가 비례 배분에 가깝고, 작은 m 의 집합은
    큰 m 의 집합에 포함된다. 난수는 지역 이름 오름차순으로 지역마다 permutation 하나를 뽑는다(소비 순서 고정)."""
    mac = np.asarray(macro_src).astype(str)
    n = len(mac)
    rng = np.random.RandomState(seed_of("lgt-ctx", target, mode, split, seed))
    u = np.zeros(n); rank = np.zeros(n, np.int64); code = np.zeros(n, np.int64)
    for j, nm in enumerate(sorted(set(mac.tolist()))):
        idx = np.where(mac == nm)[0]
        perm = rng.permutation(len(idx))
        rank[idx[perm]] = np.arange(len(idx))                       # 순열의 p번째 행이 순위 p 를 받는다
        u[idx] = (rank[idx] + 0.5) / len(idx)
        code[idx] = j
    return np.lexsort((rank, code, u)).astype(np.int64)             # 정렬 키 (u, 지역 이름, p)


def src_budget(n_src, T, C, R):
    """원천 행 예산 m = min(원천 행 수, C − max(T, R))."""
    return int(max(min(int(n_src), int(C) - max(int(T), int(R))), 0))


def region_counts(macro_src, rows):
    mac = np.asarray(macro_src).astype(str)[np.asarray(rows, int)]
    return {str(k): int(v) for k, v in sorted(Counter(mac.tolist()).items())}


def region_dev(macro_src, rows):
    """선택한 원천 행의 지역별 행 수와 비례 배분 m·c_j/N 의 차이(절댓값)의 최댓값."""
    mac = np.asarray(macro_src).astype(str)
    rows = np.asarray(rows, int)
    if len(rows) == 0 or len(mac) == 0:
        return 0.0
    tot = Counter(mac.tolist()); got = Counter(mac[rows].tolist())
    return float(max(abs(got.get(k, 0) - len(rows) * c_ / len(mac)) for k, c_ in tot.items()))


def region_dev_bound(macro_src):
    """region_dev 의 이론 상한. 정렬 키 (u, 지역 이름, p)의 앞 m행에서 지역 j 의 행 수는 u*·c_j 와 0.5 이내로 같으므로
    비례 배분과의 차이는 0.5·(1 − w_j) + w_j·(J − 1)/2 이하다(w_j = 지역의 비중, J = 지역 수)."""
    mac = np.asarray(macro_src).astype(str)
    if len(mac) == 0:
        return 0.0
    cnt = Counter(mac.tolist())
    J = len(cnt)
    return float(max(0.5 * (1.0 - c_ / len(mac)) + (c_ / len(mac)) * (J - 1) / 2.0 for c_ in cnt.values()))


def coef_n(c, sel, kappa):
    """E_n = (n·E_ls + κ·E0)/(n + κ). n = 0 이거나 E_ls 가 비유한이면 E0 (h40.run_ctx 의 coefs 와 같은 식)."""
    m_ = len(sel)
    if m_ == 0:
        return float(c.E0)
    E_ls = H.ls_E(c.yA[sel], c.sA[sel])
    if not np.isfinite(E_ls):
        return float(c.E0)
    return float((m_ * E_ls + float(kappa) * c.E0) / (m_ + float(kappa)))


def context_rows(c, order, sel, vs, a, n, draw):
    """(원천 행 색인, 컨텍스트에 넣는 라벨 셀 색인, 표지). 대상 행이 상한에 가까우면 라벨 셀을 부분 추출한다(표지 tsub)."""
    dup = int(vs["dup"])
    tsel = np.asarray(sel, int)
    cap = int(a.ctx_max) if vs["tgt"] else int(a.ctx_max) - int(a.ctx_src_min)
    flag = ""
    if dup * len(tsel) > cap:
        k = max(cap // dup, 0)
        rng = np.random.RandomState(seed_of("lgt-tsub", c.target, c.mode, c.split, n, draw))
        tsel = np.sort(rng.choice(tsel, k, replace=False))
        flag = "tsub"
    if vs["tgt"]:
        src = np.zeros(0, np.int64)
    else:
        src = np.asarray(order[:src_budget(len(order), dup * len(tsel), a.ctx_max, a.ctx_reserve)], np.int64)
    return src, tsel, flag


def macro_codes(names):
    """지역 부호(rid 변형): 이름을 정렬한 정수. 새 부호 = 부호 수(h35 와 같다)."""
    code = {str(m): i for i, m in enumerate(sorted({str(v) for v in names}))}
    return code, len(code)


def build_X(c, src, tsel, dup, code=None, new_code=0):
    """컨텍스트 행렬(원천 행 다음에 대상 행)과 채점 행렬. code 를 주면 26번째 열로 지역 부호를 붙인다."""
    X = np.vstack([c.X_src[src], np.repeat(c.XA[tsel], int(dup), axis=0)]).astype(np.float32)
    XB = np.asarray(c.XB, np.float32)
    if code is not None:
        cs = np.array([code.get(str(m), new_code) for m in np.asarray(c.macro_src)[src]], np.float32)
        col = np.concatenate([cs, np.full(int(dup) * len(tsel), float(new_code), np.float32)])
        X = np.c_[X, col].astype(np.float32)
        XB = np.c_[XB, np.full(len(XB), float(new_code), np.float32)].astype(np.float32)
    return X, XB


def build_y(c, src, tsel, dup, kind, E_anchor=None):
    """컨텍스트 목표. kind = 'D'(y) 또는 'R'(잔차: 원천은 E0 앵커, 대상은 E_anchor 앵커)."""
    if kind == "D":
        ys, yt = c.y_src[src], c.yA[tsel]
    else:
        ys, yt = c.r0_src[src], c.yA[tsel] - float(E_anchor) * c.sA[tsel]
    return np.concatenate([np.asarray(ys, float), np.repeat(np.asarray(yt, float), int(dup))])


def ctx_sha(X, y):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X, np.float32).tobytes()); h.update(np.ascontiguousarray(y, np.float64).tobytes())
    return h.hexdigest()[:12]


# ================================================================ 학습기
def is_cuda_oom(e):
    """CUDA 메모리 부족 예외인지. tabpfn 8.0.7 의 predict 는 torch.OutOfMemoryError 를 TabPFNCUDAOutOfMemoryError 로 바꿔 올린다.
    파이썬의 MemoryError(호스트 메모리)는 여기에 들지 않는다."""
    names = {k.__name__ for k in type(e).__mro__}
    if names & {"OutOfMemoryError", "TabPFNOutOfMemoryError", "TabPFNCUDAOutOfMemoryError"}:
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _free_cuda():
    """GPU 캐시를 비운다. 참조가 남은 텐서는 반환되지 않으므로 예외 블록 밖에서 gc.collect() 뒤에 부른다."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                     # noqa: BLE001
        pass


def _predict_chunks(m, XB, chunk):
    chunk = max(int(chunk), 1)
    return np.concatenate([np.asarray(m.predict(XB[k:k + chunk]), float) for k in range(0, len(XB), chunk)])


def tabpfn_schema(m):
    """적합 뒤 TabPFN 이 정한 열 분류(범주형 열과 상수 열의 색인). tabpfn 8.0.7 의 inferred_feature_schema_ 를 읽는다.
    범주형 추론은 컨텍스트가 100행을 넘을 때만 한다(inference_config.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE). 읽을 수 없으면 빈 dict."""
    n_est = getattr(m, "n_estimators_", None)                             # 적합에 실제로 쓴 n_estimators(8.0.7 의 auto_scale_n_estimators 확인용)
    try:
        from tabpfn.preprocessing.datamodel import FeatureModality
        sc = m.inferred_feature_schema_
        return dict(cat_cols=[int(i) for i in sc.indices_for(FeatureModality.CATEGORICAL)],
                    const_cols=[int(i) for i in sc.indices_for(FeatureModality.CONSTANT)], n_est_used=int(n_est) if n_est is not None else -1)
    except Exception:                                                     # noqa: BLE001
        return {}


def tabpfn_fit_predict(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
    """TabPFN 적합과 예측. 생성자 인자는 h35.tfm_fit_predict 와 같고 범주형 지정(기본은 지정 없음, rid 변형은 [25])과
    n_preprocessing_jobs = 1(tabpfn 8.0.7 의 기본값을 명시)만 다르다. 반환 (예측, 표지, 정보).
    예측은 한 번에 한다. CUDA 메모리 부족 예외는 except 블록에서 표지만 세우고, 블록을 벗어나 예외 객체와 그 traceback 이 잡은 텐서가 풀린 뒤
    gc.collect() 와 캐시 비우기를 하고 --pred-chunk 행 묶음으로 다시 예측한다(표지 chunk).
    정보: 열 분류(cat_cols, const_cols). check_chunk > 0 이면 같은 모델로 묶음 예측을 한 번 더 해 한 번 예측과의 차이(절댓값의 최댓값)와
    그 점검 시간(check_s)을 남긴다(스모크 전용)."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device="cuda", random_state=int(seed), n_estimators=int(a.n_est), model_path=str(a.MODEL),
                        ignore_pretraining_limits=True, categorical_features_indices=[] if cat_idx is None else [int(cat_idx)],
                        memory_saving_mode=False, n_preprocessing_jobs=1)
    m.fit(X, y)
    flag, info = "", tabpfn_schema(m)
    oom, p = False, None
    try:
        p = np.asarray(m.predict(XB), float)
    except Exception as e:                                                # noqa: BLE001
        if not is_cuda_oom(e):
            raise
        oom = True                                                        # 처리는 블록 밖에서 한다
    if oom:
        gc.collect()
        _free_cuda()
        p = _predict_chunks(m, XB, a.pred_chunk)
        flag = "chunk"
    if check_chunk and flag == "" and len(XB) > int(check_chunk):
        t0 = time.time()
        q = _predict_chunks(m, XB, check_chunk)
        info.update(chunk_rows=int(check_chunk), chunk_maxdiff=float(np.max(np.abs(q - p))), check_s=round(time.time() - t0, 3))
    del m
    return p, flag, info


def cb_fit_predict(HA, X, y, XB, seed, cat_idx=None):
    """같은 컨텍스트 행으로 학습하는 CatBoost. 초모수는 catboost_lo(h40.cb_fit)와 같다(반복 --cb-iters, 학습률 0.05, 깊이 3, l2 3,
    random_seed = 학습기 seed). h40.cb_fit 은 numpy 배열을 fit 에 넘겨 catboost 가 학습 Pool 을 스레드 제한 없이(Pool 의 기본값
    thread_count = −1) 만든다. 여기서는 학습과 채점의 Pool 을 thread_count = --threads 로 직접 만든다. rid 변형은 26번째 열을 범주형(정수)으로
    지정한다(h35.cb_fit_predict 의 cat_idx 경로와 같은 방식). 반환 (예측, 표지, 정보)."""
    th = int(HA.threads)
    from catboost import CatBoostRegressor, Pool

    def pool(Z, t=None):
        if cat_idx is None:
            return Pool(np.asarray(Z), t, thread_count=th)
        d = pd.DataFrame(np.asarray(Z))
        d[int(cat_idx)] = d[int(cat_idx)].astype(int)
        return Pool(d, t, cat_features=[int(cat_idx)], thread_count=th)
    m = CatBoostRegressor(iterations=int(HA.cb_iters), learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=int(seed), verbose=0,
                          allow_writing_files=False, thread_count=th)
    m.fit(pool(X, y))
    return np.asarray(m.predict(pool(XB), thread_count=th), float), "", {}


def est_fit_s(learner, n_ctx, n_eval):
    """적합 1건의 추정 시간(s). 계획서 §6C.10 의 식이다(--count-only 전용. 결과에는 쓰지 않는다)."""
    if learner == TP:
        return 0.5 + (6.7 + 0.6 * float(n_eval) / 1000.0) * (float(n_ctx) / 9900.0)
    return 0.4 * (max(float(n_ctx), 1.0) / 10000.0) ** 0.6


class FitAbort(RuntimeError):
    """같은 학습기의 연속 실패가 FAIL_STREAK_MAX 에 이르렀다. 단위를 중단한다(unit.json 을 쓰지 않는다).
    워커 밖으로는 _worker_run_t 가 내장 RuntimeError 로 바꿔 넘긴다(주 모듈의 예외 클래스는 부모 프로세스에서 풀 수 없다)."""


class FitterT:
    """적합 실행, 계수, 기록. dry = True 이면 학습 없이 수와 추정 시간만 센다(예측은 0).
    실패의 구분: 점검 실패(컨텍스트 행렬이나 목표가 규칙에 어긋남)는 그 적합만 실패로 남긴다. 학습기 실패(적합·예측 예외, 비유한 예측)는
    학습기별 연속 횟수를 세고 FAIL_STREAK_MAX 에 이르면 FitAbort 로 단위를 중단한다. 성공한 적합이 그 학습기의 연속 횟수를 0 으로 되돌린다."""

    def __init__(self, a, HA, n_eval, dry=False):
        self.a, self.HA, self.n_eval, self.dry = a, HA, int(n_eval), bool(dry)
        self.n = Counter(); self.sec = Counter(); self.fail = Counter(); self.nonfin = Counter()       # 키 = "변형:학습기"
        self.nd = Counter(); self.secd = Counter(); self.rowsd = Counter(); self.estd = Counter()      # 키 = "축|학습기|방법"(h40.timing_table 의 형식)
        self.errors: list = []
        self.smoke: list = []                                             # 스모크의 묶음 예측 점검과 CatBoost Pool 점검 기록
        self.schema: dict = {}                                            # TabPFN 열 분류: "변형|ctx<=100" 또는 "변형|ctx>100" → {분류 JSON: 적합 수}
        self.streak = Counter()                                           # 학습기별 연속 실패 수
        self._checked: set = set()
        self._cb_checked = False

    def check(self, X, y, XB, n_ctx, cat_idx):
        """컨텍스트 행렬과 목표의 점검. 어긋나면 ValueError 다(그 적합만 실패로 남는다)."""
        if len(y) != len(X) or len(X) != int(n_ctx):
            raise ValueError(f"컨텍스트 행 수 불일치: X {len(X)}, y {len(y)}, 기대 {int(n_ctx)}")
        if len(X) > int(self.a.ctx_max):
            raise ValueError(f"컨텍스트 행 수 {len(X)} 가 상한 {int(self.a.ctx_max)} 을 넘는다")
        if X.shape[1] != XB.shape[1] or (cat_idx is not None and int(cat_idx) != X.shape[1] - 1):
            raise ValueError(f"입력 열 수 불일치: 컨텍스트 {X.shape[1]}, 채점 {XB.shape[1]}, 범주형 열 {cat_idx}")
        if not np.all(np.isfinite(np.asarray(y, float))):
            raise ValueError("학습 목표에 비유한 값이 있다")

    def _note_schema(self, ctx_set, n_ctx, ex):
        """TabPFN 이 정한 열 분류를 변형과 컨텍스트 크기(100행 이하, 초과)별로 센다(unit.json 의 tabpfn_schema)."""
        if "cat_cols" not in ex:
            return
        key = f"{ctx_set}|{'ctx<=100' if int(n_ctx) <= 100 else 'ctx>100'}"
        sig = json.dumps(dict(cat=list(ex["cat_cols"]), const=list(ex["const_cols"]), n_est=int(ex.get("n_est_used", -1))), sort_keys=True)
        slot = self.schema.setdefault(key, {})
        slot[sig] = int(slot.get(sig, 0)) + 1

    def _fail(self, k, ctx_set, learner, err):
        self.fail[f"{ctx_set}:{learner}"] += 1
        if len(self.errors) < 20:
            self.errors.append(f"{k}: {err}")
        print(f"    [warn] 적합 실패({k}): {err[:160]}", flush=True)

    def fit(self, ctx_set, learner, method, X, y, XB, seed, cat_idx, n_ctx, info=None):
        """반환 (예측 또는 잔차 성분, 표지, 시간 s, 컨텍스트 해시). 적합 한 건의 예외는 그 적합만 실패로 남긴다(예측 NaN, 표지 fail).
        같은 학습기의 연속 실패가 FAIL_STREAK_MAX 에 이르면 FitAbort 를 올린다. 시간에는 스모크 점검(묶음 예측, CatBoost Pool 대조)을 넣지 않는다."""
        k = f"{ctx_set}|{learner}|{method}"
        self.n[f"{ctx_set}:{learner}"] += 1; self.nd[k] += 1; self.rowsd[k] += int(n_ctx)
        self.estd[k] += est_fit_s(learner, n_ctx, self.n_eval)
        if self.dry:
            if X is not None:                                             # --dry-build: 행렬의 크기와 목표만 확인한다(학습 없음)
                try:
                    self.check(X, y, XB, n_ctx, cat_idx)
                except ValueError as e:
                    self.fail[f"{ctx_set}:{learner}"] += 1
                    if len(self.errors) < 20:
                        self.errors.append(f"{k}: {repr(e)[:200]}")
            return np.zeros(self.n_eval), "", 0.0, ""
        _orphan_check()
        t0 = time.time(); flag, sha, err, oom, lfail, extra_s, p = "", "", "", False, False, 0.0, None
        try:
            if TRACE is not None:
                TRACE.append(dict(info or {}, ctx_set=ctx_set, learner=learner, method=method, seed=int(seed), cat_idx=cat_idx,
                                  X=np.array(X, copy=True), y=np.array(y, copy=True), XB=np.array(XB, copy=True)))
            self.check(X, y, XB, n_ctx, cat_idx)
        except ValueError as e:
            err = "점검: " + repr(e)[:200]                                   # 점검 실패는 연속 실패에 세지 않는다
        if not err:
            try:
                sha = ctx_sha(X, y)
                if learner == TP:
                    chk = 0
                    if self.a.smoke and ctx_set not in self._checked and self.n_eval >= 2:
                        chk = int(math.ceil(self.n_eval / 2.0))           # 스모크: 변형마다 첫 TabPFN 적합에서 채점 셀을 두 묶음으로 나눠 본다
                    p, flag, ex = tabpfn_fit_predict(self.a, X, y, XB, seed, cat_idx, check_chunk=chk)
                    self._note_schema(ctx_set, n_ctx, ex)
                    extra_s += float(ex.get("check_s", 0.0) or 0.0)
                    if chk:
                        self._checked.add(ctx_set)
                        self.smoke.append(dict({q: v for q, v in ex.items() if q.startswith("chunk") or q == "check_s"}, ctx_set=ctx_set,
                                               method=method, n_eval=self.n_eval, n_ctx=int(n_ctx)))
                else:
                    p, flag, ex = cb_fit_predict(self.HA, X, y, XB, seed, cat_idx)
                    if self.a.smoke and cat_idx is None and not self._cb_checked:      # 스모크: Pool 의 스레드 지정이 예측을 바꾸지 않는지 h40.cb_fit 과 대조
                        t1 = time.time()
                        q = np.asarray(H.cb_fit(self.HA, X, y, seed).predict(XB, thread_count=int(self.HA.threads)), float)
                        self._cb_checked = True
                        self.smoke.append(dict(cb_pool_maxdiff=float(np.max(np.abs(q - np.asarray(p, float)))), ctx_set=ctx_set, method=method,
                                               n_ctx=int(n_ctx)))
                        extra_s += time.time() - t1
                p = np.asarray(p, float)
                if p.shape != (len(XB),):
                    raise ValueError(f"예측의 모양 {p.shape} 이 채점 셀 수 {len(XB)} 와 다르다")
            except (ImportError, MemoryError):                            # 환경 문제: 조각 전체를 실패로 둔다(재개 때 다시 실행)
                raise
            except Exception as e:                                        # noqa: BLE001
                err, oom, lfail = repr(e)[:200], is_cuda_oom(e), True     # 처리는 블록 밖에서 한다(예외 객체가 잡은 텐서를 먼저 놓는다)
        if err:
            self._fail(k, ctx_set, learner, err)
            p, flag = np.full(len(XB), np.nan), "fail"
            if oom:
                gc.collect()
                _free_cuda()
        elif not np.all(np.isfinite(p)):
            self.nonfin[f"{ctx_set}:{learner}"] += 1
            lfail = True
        if lfail:
            self.streak[learner] += 1
            if self.streak[learner] >= FAIL_STREAK_MAX:
                raise FitAbort(f"{TAG_ABORT} {learner} 적합이 {self.streak[learner]}회 연속 실패했다({k}). 마지막: {err or '비유한 예측'}")
        elif not err:
            self.streak[learner] = 0
        dt = max(time.time() - t0 - extra_s, 0.0)
        self.sec[f"{ctx_set}:{learner}"] += dt; self.secd[k] += dt
        return p, flag, dt, sha


class CellBookT:
    """셀 단위 저장(float32). 성분(comp)은 D0 의 예측(pred) 또는 잔차 성분(g)이다. 예측은 E·s + λ·g 로 복원한다."""

    def __init__(self, c, loc=None, E0=np.nan):
        self.y = np.asarray(c.yB, np.float32); self.s = np.asarray(c.sB, np.float32); self.block = np.asarray(c.blkB).astype(str)
        self.loc = dict(loc or {}); self.E0 = float(E0)
        self.keys: list = []; self.P: list = []; self.E: list = []
        self.coef: dict = {}

    def add(self, method, learner, alpha, n, d, seed, comp, vec, E=np.nan):
        self.keys.append(json.dumps([str(method), str(learner), str(alpha), "cell", int(n), int(d), int(seed), str(comp)]))
        self.P.append(np.asarray(vec, np.float32)); self.E.append(float(E))

    def add_coef(self, n, d, E):
        self.coef[(int(n), int(d))] = float(E)

    def save(self, path):
        path = Path(path)
        P = np.stack(self.P).astype(np.float32) if self.P else np.zeros((0, len(self.y)), np.float32)
        ck = sorted(self.coef)
        arrs = dict(y=self.y, s=self.s, block=self.block, keys=np.array(self.keys, dtype=str), P=P, E=np.asarray(self.E, float),
                    E0=np.array(self.E0), coef_keys=np.array([json.dumps(list(k)) for k in ck], dtype=str),
                    coef_E=np.array([self.coef[k] for k in ck], float))
        for k in ("loc_id", "lat", "lon"):
            if k in self.loc and len(self.loc[k]) == len(self.y):
                arrs[k] = np.asarray(self.loc[k])
        tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path)
        return path


# ================================================================ 작업 단위 실행
def run_ctx_t(c, axis, a, HA, dry=False, code=None, new_code=0, loc=None):
    """작업 단위 하나(축, 대상, 모드, 분할)를 실행한다. 반환 (rows, BlockStore, 통계 dict, CellBookT 또는 None).
    code = 지역 부호(rid 변형). 주지 않으면 원천 지역 이름으로 만든다(시험용 합성 자료)."""
    F = FitterT(a, HA, len(c.yB), dry)
    empty = dict(n_fit={}, sec={}, fail={}, n_rows=0, status="no_eval", n_fit_detail={}, sec_detail={}, rows_detail={}, est_detail={}, errors=[],
                 n_stored=0, n_stored_ml=0, n_nonfinite_keys=0, ctx={}, ctx_dev_max=0.0, ctx_dev_bound=0.0, n_src_regions=0, n_ctx_max=0, n_ctx_min=0,
                 flags={}, smoke_check=[], tabpfn_schema={}, nonfinite_fits={}, n_fit_learner={}, n_fail_learner={}, n_stored_learner={},
                 failed_learners=[])
    if len(c.yB) == 0 or len(c.yA) == 0:
        return [], None, empty, None
    st = BlockStore(f"{c.target}|{c.mode}", c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part="gpu", axis=axis))
    rows, n_rows, n_bad = [], [0], [0]
    nA = len(c.yA)
    E0 = float(c.E0)
    build = (not dry) or bool(getattr(a, "dry_build", False))
    cells = CellBookT(c, loc, E0) if (a.save_cells and not dry) else None
    if code is None:
        code, new_code = macro_codes(np.asarray(c.macro_src).tolist() + [c.parent])
    base = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part="gpu", axis=axis)
    orders = {int(s_): ctx_order(c.macro_src, c.target, c.mode, c.split, int(s_)) for s_ in HA.SEEDS}
    ctx_info: dict = {}
    dev_max, nctx_all, flags = [0.0], [], Counter()

    def add(ctx_set, method, learner, alpha, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", nc=(0, 0, 0), fit_s=0.0, sha=""):
        n_rows[0] += 1
        if dry:
            return
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = dict(**base, method=method, learner=learner, alpha=str(alpha), placement="cell", n=int(n), n_lab=int(n_lab), draw=int(d),
                   seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used), alpha_sel="",
                   n_blocks_lab=int(nb_lab), n_nonfinite=nf, fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"), ctx_set=ctx_set,
                   n_ctx=int(nc[0]), n_ctx_src=int(nc[1]), n_ctx_tgt=int(nc[2]), fit_s=round(float(fit_s), 3), ctx_sha=str(sha))
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다(방법 간 채점 셀 집합 통일)
            n_bad[0] += 1
            rows.append(row)
            return
        key = (method, learner, str(alpha), "cell", int(n), int(d), int(seed), float(lam))
        st.add(key, c.yB, pred)
        sse, cnt = st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
            rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
        row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.nanmean(pred - c.yB)) if len(c.yB) else np.nan)
        rows.append(row)

    seen_p1: set = set()

    def analytic(ctx_set, n, d, E_n, nl, nb):
        """P1 = E_n·s. (n, 추출)마다 한 번 저장한다(변형과 무관한 값이다)."""
        if (n, d) in seen_p1:
            return
        seen_p1.add((n, d))
        add(ctx_set, "P1", "none", "1", n, d, -1, 0.0, E_n * c.sB, E_n, nl, nb)
        if cells is not None:
            cells.add_coef(n, d, E_n)

    add("main" if axis == "main" else "sens", "P0", "none", "1", 0, 0, -1, 0.0, E0 * c.sB, E0, 0)      # P0 은 모든 비교의 기준이므로 항상 저장한다

    def note_ctx(seed, src):
        m_ = int(len(src))
        slot = ctx_info.setdefault(str(int(seed)), {})
        if str(m_) not in slot:
            slot[str(m_)] = dict(n_src=m_, regions=region_counts(c.macro_src, src))
            dev_max[0] = max(dev_max[0], region_dev(c.macro_src, src))

    def run_set(name, vs, grid, methods):
        dup, alpha, sfx = int(vs["dup"]), str(vs["alpha"]), str(vs["sfx"])
        cat_idx = c.XB.shape[1] if vs["rid"] else None                     # rid: 26번째 열(색인 25)이 지역 부호다
        for n, d in H.cells_of(grid, HA.DRAWS, nA):
            sel = H.draw_cells(c.target, c.mode, c.split, n, d, nA)
            nl = len(sel)
            if nl < int(vs["min_lab"]):                                   # tgt: 실제 라벨 수가 40 미만인 칸은 돌리지 않는다
                continue
            nb = len(np.unique(c.blkA[sel])) if nl else 0
            E_n = coef_n(c, sel, HA.kappa)
            analytic(name, n, d, E_n, nl, nb)
            for seed in HA.SEEDS:
                src, tsel, tflag = context_rows(c, orders[int(seed)], sel, vs, a, n, d)
                nc = (len(src) + dup * len(tsel), len(src), dup * len(tsel))
                if nc[0] == 0:                                            # 컨텍스트가 비면 적합하지 않는다(원천이 없고 n = 0)
                    continue
                note_ctx(seed, src)
                nctx_all.append(nc[0])
                if tflag:
                    flags[tflag] += 1
                X = XB = None
                if build:
                    X, XB = build_X(c, src, tsel, dup, code if vs["rid"] else None, new_code)
                g0: dict = {}
                for method in methods:
                    mname = method + sfx
                    if method == "R1" and nl == 0 and g0:                 # n = 0 에서 R1 은 R0 과 같다(E_n = E0, 같은 적합)
                        for lr in LEARNERS_T:
                            g, fl, sha = g0[lr]
                            for lam in HA.LAMS:
                                add(name, mname, lr, alpha, n, d, seed, lam, E0 * c.sB + lam * g, E0, nl, nb, fl, nc, 0.0, sha)
                            if cells is not None:
                                cells.add(mname, lr, alpha, n, d, seed, "g", g, E0)
                        continue
                    kind = "D" if method == "D0" else "R"
                    E_a = None if kind == "D" else (E0 if method == "R0" else E_n)
                    y = build_y(c, src, tsel, dup, kind, E_a) if build else None
                    for lr in LEARNERS_T:
                        info = dict(n=int(n), draw=int(d), alpha=alpha, sel=np.array(sel, copy=True), tsel=np.array(tsel, copy=True),
                                    src=np.array(src, copy=True), dup=dup, E_anchor=E_a)
                        g, fl, dt, sha = F.fit(name, lr, mname, X, y, XB, seed, cat_idx, nc[0], info)
                        fl = ";".join(v for v in (tflag, fl) if v)
                        if kind == "D":
                            add(name, mname, lr, alpha, n, d, seed, 1.0, g, np.nan, nl, nb, fl, nc, dt, sha)
                        else:
                            for lam in HA.LAMS:
                                add(name, mname, lr, alpha, n, d, seed, lam, float(E_a) * c.sB + lam * g, E_a, nl, nb, fl, nc, dt, sha)
                        if method == "R0" and nl == 0:
                            g0[lr] = (g, fl, sha)
                        if cells is not None:
                            cells.add(mname, lr, alpha, n, d, seed, "pred" if kind == "D" else "g", g, np.nan if kind == "D" else E_a)

    if axis == "main":
        run_set("main", MAIN_SET, list(HA.N_GRID), AX_T["main"]["methods"])
    else:
        for v in a.SENS:
            run_set(v, SENS_SETS[v], [n for n in sens_grid(a, v) if n in HA.N_GRID], AX_T["sens"]["methods"])
    n_ml = sum(1 for k in st.keys if k[1] != "none")
    n_fail = int(sum(F.fail.values())) + int(n_bad[0])
    lc = learner_counts_t(F, st)
    status = "ok" if n_fail == 0 else ("failed" if lc["failed_learners"] else "partial")
    stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(n_rows[0]), status=status,
                 n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                 est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(len(st)), n_stored_ml=int(n_ml),
                 n_nonfinite_keys=int(n_bad[0]), ctx=ctx_info, ctx_dev_max=round(float(dev_max[0]), 3),
                 ctx_dev_bound=round(region_dev_bound(c.macro_src), 3), n_src_regions=int(len(set(np.asarray(c.macro_src).astype(str).tolist()))),
                 n_ctx_max=int(max(nctx_all)) if nctx_all else 0, n_ctx_min=int(min(nctx_all)) if nctx_all else 0, flags=dict(flags),
                 smoke_check=list(F.smoke), tabpfn_schema=F.schema, nonfinite_fits=dict(F.nonfin), **lc)
    return rows, st, stats, cells


def learner_counts_t(F, st):
    """학습기별 적합 수, 실패 적합 수(예외와 비유한 예측), 저장 키 수와 조각을 failed 로 두게 하는 학습기(계획서 §6C.8, 개정 12).
    적합을 시도한 학습기 가운데 저장 키가 0 이거나 실패 적합 비율이 FAIL_RATIO_MAX 를 넘는 학습기가 있으면 조각 상태는 failed 다
    (--resume 이 다시 실행한다). 그 밖의 실패가 있으면 partial 이다."""
    fit_by, fail_by, stored_by = Counter(), Counter(), Counter()
    for kk, v in F.n.items():
        fit_by[kk.split(":", 1)[-1]] += int(v)
    for src in (F.fail, F.nonfin):
        for kk, v in src.items():
            fail_by[kk.split(":", 1)[-1]] += int(v)
    for k in (st.keys if st is not None else []):
        if k[1] != "none":
            stored_by[k[1]] += 1
    bad = [lr for lr in LEARNERS_T if fit_by[lr] > 0 and (stored_by[lr] == 0 or fail_by[lr] / fit_by[lr] > FAIL_RATIO_MAX)]
    return dict(n_fit_learner=dict(fit_by), n_fail_learner=dict(fail_by), n_stored_learner=dict(stored_by), failed_learners=bad)


# ================================================================ 조각 입출력
_SHA: dict = {}


def file_sha(path, short=0):
    """파일의 SHA-1. 읽을 수 없으면 'none'."""
    k = str(path)
    if k not in _SHA:
        try:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            _SHA[k] = h.hexdigest()
        except OSError:
            _SHA[k] = "none"
    v = _SHA[k]
    return v[:short] if (short and v != "none") else v


def code_sha_t():
    return file_sha(__file__, 12)


def pkg_version(name):
    try:
        from importlib import metadata
        return str(metadata.version(name))
    except Exception:                                                     # noqa: BLE001
        return "NA"


def env_info(gpu=False):
    """실행 환경 기록. gpu = True 이면 torch 를 불러 GPU 이름을 적는다(워커 전용)."""
    d = dict(python=platform.python_version(), numpy=str(np.__version__), pandas=str(pd.__version__), torch=pkg_version("torch"),
             tabpfn=pkg_version("tabpfn"), catboost=pkg_version("catboost"), sklearn=pkg_version("scikit-learn"), gpu_name="")
    if gpu:
        try:
            import torch
            d["torch"] = str(torch.__version__)
            if torch.cuda.is_available():
                d["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:                                                 # noqa: BLE001
            pass
    return d


def shard_paths_t(a, axis, target, mode, split):
    """조각 경로. 이름은 h40.shard_paths 가 만든다(<tag>__gpu__<대상>__<모드>__s<분할>). cells 는 h43 이 더한 파일이다."""
    p = dict(H.shard_paths(h40_args_t(a, axis), "gpu", target, mode, split))
    p["cells"] = Path(str(p["unit"])[:-len("_unit.json")] + "_cells.npz")
    return p


def unit_cfg_t(a, axis):
    """결과에 영향을 주는 설정 요약. 대상, 분할, tag, 스레드 수, GPU 번호는 넣지 않는다(조각의 정체 또는 실행 자원)."""
    HA = h40_args_t(a, axis)
    return dict(axis=axis, sens=list(a.SENS) if axis == "sens" else [], methods=list(AX_T[axis]["methods"]), learners=list(LEARNERS_T),
                n_grid=list(HA.N_GRID), sens_grid={v: sens_grid(a, v) for v in a.SENS} if axis == "sens" else {},
                draws=int(HA.DRAWS), seeds=list(HA.SEEDS), lams=list(HA.LAMS), kappa=float(HA.kappa), buffer_km=float(HA.buffer_km),
                k_sub=str(HA.k_sub), min_cells_prior=int(HA.min_cells_prior),
                subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans",
                ctx_max=int(a.ctx_max), ctx_reserve=int(a.ctx_reserve), ctx_src_min=int(a.ctx_src_min), n_est=int(a.n_est),
                model=Path(a.MODEL).name, model_sha1=file_sha(a.MODEL), tabpfn=pkg_version("tabpfn"), cb_iters=int(HA.cb_iters), feats="x25",
                pred_chunk=int(a.pred_chunk))


def unit_state_t(a, axis, target, mode, split):
    """(완료 여부, 사유). 완료 = runs, blocksse, unit 이 있고, 설정 해시가 같고, status 가 failed 가 아니다."""
    p = shard_paths_t(a, axis, target, mode, split)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != H.cfg_hash(unit_cfg_t(a, axis)):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    if getattr(a, "rerun_partial", False) and u.get("status") == "partial":
        return False, "일부 적합 실패(--rerun-partial)"
    return True, str(u.get("status", "ok"))


def _atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, path)


def write_shard_t(a, axis, c, rows, st, stats, cells, elapsed, extra=None):
    p = shard_paths_t(a, axis, c.target, c.mode, c.split)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    _atomic_csv(df[[k for k in RUN_COLS if k in df] + [k for k in df.columns if k not in RUN_COLS]] if len(df) else df, p["runs"])
    save_stores([st], p["npz"])
    if cells is not None:
        cells.save(p["cells"])
    cfg = unit_cfg_t(a, axis)
    unit = {**stats, **c.meta}
    unit.update(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part="gpu", axis=axis, learner="", tag=axis_tag(a, axis),
                elapsed_s=round(float(elapsed), 1), n_fit_total=int(sum(stats["n_fit"].values())), cfg=cfg, cfg_hash=H.cfg_hash(cfg),
                cfg_common=H.cfg_hash(cfg), code_sha=code_sha_t(), code_sha_h40=H.code_sha(), model_sha1=cfg["model_sha1"],
                threads=int(a.threads), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), has_cells=bool(cells is not None))
    unit.update(extra or {})
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))      # 완료 표지는 마지막에 쓴다
    return unit


# ================================================================ 작업 단위
def eval_cells(D, c, target, split):
    """채점 셀의 식별자와 좌표(셀 단위 저장용). h40.build_ctx 와 같은 순서로 다시 구하고 y 가 같은지 확인한다. 어긋나면 빈 dict."""
    try:
        df = D.df
        t_idx = D.target_idx(target)
        _, B_idx = H.half_split_blocks(df, t_idx, split)
        ev = B_idx[H.eval_mask(df.iloc[B_idx])]
        if len(ev) != len(c.yB) or not np.allclose(df.y.values[ev], c.yB, equal_nan=True):
            return {}
        return {k: df[k].values[ev] for k in ("loc_id", "lat", "lon") if k in df}
    except Exception:                                                     # noqa: BLE001
        return {}


def require_cuda():
    """GPU 를 쓸 수 없으면 실행하지 않는다(CPU 대체 없음)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch.cuda 를 쓸 수 없다(CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'). "
                           "TabPFN 을 CPU 로 돌리지 않는다")
    return torch


def cuda_backend():
    """GPU 백엔드(torch). CUDA 를 쓸 수 없으면 예외다. 시험에서 GPU 없이 실행 경로를 확인할 때 대체한다."""
    return require_cuda()


def run_unit_t(a, axis, target, mode, split, dry=False):
    t0 = time.time()
    HA = h40_args_t(a, axis)
    D = H.get_data(HA)
    c = H.build_ctx(D, HA, target, mode, split)
    code, new_code = macro_codes(D.df.macro.unique())
    torch = None
    if not dry:
        torch = cuda_backend()
        torch.cuda.reset_peak_memory_stats()
        p_ = shard_paths_t(a, axis, target, mode, split)
        for k_ in ("unit", "cells"):                                      # 이전 세대의 완료 표지와 셀 파일을 먼저 지운다(--no-cells 재실행에서 세대가 섞이지 않게)
            p_[k_].unlink(missing_ok=True)
    loc = eval_cells(D, c, target, split) if (a.save_cells and not dry) else None
    rows, st, stats, cells = run_ctx_t(c, axis, a, HA, dry=dry, code=code, new_code=new_code, loc=loc)
    if dry:
        return dict(axis=axis, target=target, mode=mode, split=split, n_A=c.meta["n_A"], n_eval=c.meta["n_eval"], nb_eval=c.meta["nb_eval"],
                    n_src=c.meta["n_src"], valid=c.meta["valid"], n_rows=stats["n_rows"], n_ctx_max=stats["n_ctx_max"],
                    n_ctx_min=stats["n_ctx_min"], ctx_dev_max=stats["ctx_dev_max"], ctx_dev_bound=stats["ctx_dev_bound"],
                    n_src_regions=stats["n_src_regions"], n_tsub=int(stats["flags"].get("tsub", 0)),
                    n_check_fail=int(sum(stats["fail"].values())), _detail=stats["n_fit_detail"], _rows=stats["rows_detail"],
                    _est=stats["est_detail"], _errors=stats["errors"])
    if st is None:                                                        # 채점 셀 또는 A 셀이 없다(열거 단계에서 걸러지는 경우)
        raise RuntimeError(f"{target}|{mode}|s{split}: 채점 셀 또는 A 셀이 없다")
    peak = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)          # torch 가 할당한 텐서의 최댓값
    resv = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2)           # torch 가 잡아 둔 메모리의 최댓값(nvidia-smi 값에 가깝다)
    unit = write_shard_t(a, axis, c, rows, st, stats, cells, time.time() - t0,
                         extra=dict(env=env_info(gpu=True), gpu_mem_peak_mib=round(peak, 1), gpu_mem_reserved_peak_mib=round(resv, 1),
                                    run_id=str(getattr(a, "RUN_ID", "") or ""), gpu_uuid=str(_WINFO.get("gpu_uuid", "")),
                                    device_check=str(_WINFO.get("device_check", ""))))
    del rows, st, cells
    _free_cuda()                                                          # 단위가 끝나면 GPU 캐시를 비운다(모델은 적합마다 지운다)
    return unit


def enumerate_t(a):
    """작업 단위 (축, 대상, 모드, 분할)의 목록과 건너뛴 분할의 기록. 분할 규칙은 h40.enumerate_units 와 같다."""
    units, skipped = [], []
    for axis in a.AXES:
        HA = h40_args_t(a, axis)
        if not HA.TARGETS or not HA.N_GRID or (axis == "sens" and not a.SENS):
            continue
        D = H.get_data(HA)
        us, sk = H.enumerate_units(HA, D, "cpu")
        units += [(axis, t, m, int(sp)) for t, m, sp, _ in us]
        skipped += [dict(s_, axis=axis, part="gpu") for s_ in sk]
    return units, skipped


def unit_group(axis, target, mode):
    """실행 순서의 묶음. 0 주 설정·주 4지역, 1 민감도·주 4지역, 2 주 설정·Alaska x, 3 주 설정·나머지 학습기 축 대상, 4 주 설정·나머지,
    5 민감도·나머지."""
    main4 = target in H.MAIN4 and mode == "x"
    if axis == "sens":
        return 1 if main4 else 5
    if main4:
        return 0
    if target == H.ALASKA:
        return 2
    return 3 if (target, mode) in H.LEARNER_TARGETS else 4


def priority_t(a, D, u):
    axis, t, m, sp = u
    v = D.split_structure(t)[sp]
    return (unit_group(axis, t, m), int(sp), -int(v["n_A"]), str(t), str(m))


def unit_name_t(u):
    return f"{u[0]}|{u[1]}|{u[2]}|s{u[3]}"


# ---------------------------------------------------------------- 워커
_WT = None
_WINFO: dict = {}                                                         # 워커 정보(배정 GPU, 부모 PID, 실행 디렉터리). 부모와 시험에서는 비어 있다
UNIT_KEEP = ("axis", "target", "mode", "split", "n_A", "n_eval", "nb_eval", "n_src", "E0", "n_fit_total", "n_rows", "n_ctx_max", "elapsed_s", "wall_s",
             "device", "status", "valid", "gpu_mem_peak_mib", "gpu_mem_reserved_peak_mib", "failed_learners", "run_id")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json(path, obj):
    """작은 기록 파일(임시 파일 뒤 교체). 쓰지 못해도 실행을 멈추지 않는다."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        H._atomic_text(Path(path), json.dumps(obj, ensure_ascii=False, default=str))
    except OSError:
        pass


def unit_summary(u):
    """부모에 돌려주는 단위 요약(워커의 결과, 또는 풀이 깨져 결과를 받지 못한 단위의 unit.json)."""
    d = {k: u.get(k) for k in UNIT_KEEP}
    d["n_fail"] = int(sum((u.get("fail") or {}).values())) + int(u.get("n_nonfinite_keys", 0) or 0)
    return d


def _set_pdeathsig(sig=signal.SIGTERM):
    """부모 프로세스가 죽으면 이 프로세스가 sig 를 받게 한다(리눅스 prctl PR_SET_PDEATHSIG = 1). 성공하면 True."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return int(libc.prctl(1, int(sig), 0, 0, 0)) == 0
    except Exception:                                                     # noqa: BLE001
        return False


def _orphan_check():
    """워커 전용: 부모가 바뀌었으면(부모 종료) 바로 끝낸다. 부모 PID 를 모르면(부모 프로세스, 시험) 아무것도 하지 않는다."""
    pp = _WINFO.get("ppid")
    if pp and os.getppid() != int(pp):
        os._exit(0)


def _worker_init_t(argv, gpu_queue, threads, run_dir="", ppid=0, run_id=""):
    """워커 초기화. SIGINT 는 무시하고(중단은 부모가 처리한다) 부모 종료 때 SIGTERM 을 받도록 한다. 큐에서 GPU 번호 하나를 받아
    CUDA_DEVICE_ORDER = PCI_BUS_ID 와 CUDA_VISIBLE_DEVICES 를 둔다(torch 를 부르기 전). 스레드 수를 제한하고(torch intra-op = --threads,
    inter-op = 1) 워커 기록(run_dir/worker__<PID>.json)을 남긴다. run_id 는 부모가 정한 실행 식별자이고 조각의 unit.json 에 적는다
    (풀이 깨졌을 때 부모가 이번 실행에서 완료된 조각을 가려내는 데 쓴다)."""
    global _WT, _WINFO
    warnings.filterwarnings("ignore")
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    pds = _set_pdeathsig()
    if int(ppid) and os.getppid() != int(ppid):                           # 초기화 전에 부모가 이미 끝났다
        os._exit(0)
    gpu = str(gpu_queue.get()) if gpu_queue is not None else ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for v in THREAD_VARS:
        os.environ[v] = str(int(threads))
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    _WT = parse_args(argv)
    _WT.RUN_ID = str(run_id or "")
    _WINFO = dict(gpu=gpu, ppid=int(ppid) or os.getppid(), run_dir=str(run_dir), cuda_checked=False, pdeathsig=bool(pds))
    try:
        import torch
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:                                                     # noqa: BLE001
        pass
    if run_dir:
        _write_json(Path(run_dir) / f"worker__{os.getpid()}.json", dict(pid=os.getpid(), ppid=_WINFO["ppid"], gpu=gpu, start=_now(),
                                                                        pdeathsig=bool(pds), threads=int(threads)))


def cuda_healthy():
    """CUDA 문맥을 쓸 수 있는지(작은 연산과 동기화). 워커 전용."""
    try:
        import torch
        x = torch.ones(8, device="cuda")
        v = float((x * 2.0).sum().item())
        torch.cuda.synchronize()
        return v == 16.0
    except Exception:                                                     # noqa: BLE001
        return False


def _norm_uuid(v):
    s = str(v or "").strip().lower()
    return s[4:] if s.startswith("gpu-") else s


def _gpu_busy_exit(g, why):
    """배정 GPU 를 다른 프로세스가 쓰고 있다: 표지(run_dir/gpu_busy__<번호>.json)를 남기고 워커를 끝낸다. 부모는 풀이 깨진 것을 보고
    그 GPU 를 뺀 뒤 풀을 다시 만든다(이때 다른 워커의 실행 중 단위도 끝나며 다시 돈다)."""
    rd = _WINFO.get("run_dir")
    if rd:
        _write_json(Path(rd) / f"gpu_busy__{int(g)}.json", dict(gpu=int(g), pid=os.getpid(), why=str(why), time=_now()))
    print(f"[worker] GPU {g} 를 쓰지 않는다: {why}. 워커를 끝낸다", flush=True)
    os._exit(75)


def gpu_guard_t():
    """워커 전용 GPU 확인. 단위를 시작할 때마다 부른다.
    (1) 첫 CUDA 초기화 전: 배정 GPU 의 메모리 사용이 --gpu-mem-max-mib 이하이고 계산 프로세스가 없어야 한다.
    (2) 그 뒤: 배정 GPU 에 자기 PID 가 아닌 계산 프로세스가 없어야 한다.
    (1)(2)에 걸리면 점유 표지를 남기고 워커를 끝낸다. (3) 첫 초기화 직후: nvidia-smi 에서 자기 PID 가 있는 GPU(없으면 torch 장치 0 의 UUID)가
    배정 번호의 GPU 와 같아야 한다. 어긋나면 TAG_MISMATCH 문구의 RuntimeError 를 올린다(부모가 실행 전체를 멈춘다).
    nvidia-smi 를 부를 수 없으면 경고하고 확인 없이 진행한다."""
    gpu = _WINFO.get("gpu")
    if gpu in (None, ""):
        return
    g = int(gpu)
    try:
        info, apps = query_gpu_info(), query_gpu_apps()
    except Exception as e:                                                # noqa: BLE001
        print(f"[worker] GPU {g}: nvidia-smi 확인 실패({repr(e)[:120]}). 확인 없이 진행한다", flush=True)
        return
    me = info.get(g)
    if me is None:
        print(f"[worker] GPU {g}: nvidia-smi 목록에 없다. 확인 없이 진행한다", flush=True)
        return
    foreign = sorted({int(p) for p, u in apps if u == me["uuid"] and int(p) != os.getpid()})
    if _WINFO.get("cuda_checked"):
        if foreign:
            _gpu_busy_exit(g, f"다른 계산 프로세스 {foreign}")
        return
    if int(me["used"]) > int(_WT.gpu_mem_max_mib) or foreign:
        _gpu_busy_exit(g, f"첫 CUDA 초기화 전 메모리 사용 {int(me['used'])} MiB, 다른 계산 프로세스 {foreign}")
    import torch
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    tu = _norm_uuid(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    try:
        mine = sorted({u for p, u in query_gpu_apps() if int(p) == os.getpid()})
    except Exception:                                                     # noqa: BLE001
        mine = []
    if mine:
        ok = mine == [me["uuid"]]
    elif re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", tu):
        ok = tu == _norm_uuid(me["uuid"])
    else:
        ok = None
    if ok is False:
        raise RuntimeError(f"{TAG_MISMATCH} 배정 GPU {g}(UUID {me['uuid']})와 CUDA 장치가 다르다: 자기 PID 의 GPU {mine or '없음'}, "
                           f"torch UUID {tu or '없음'}, CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '')}")
    if ok is None:
        print(f"[worker] GPU {g}: 장치 대응을 확인하지 못했다(nvidia-smi 에 자기 PID 가 없고 torch UUID 를 읽지 못했다)", flush=True)
    _WINFO.update(cuda_checked=True, gpu_uuid=me["uuid"], device_check=("PID" if mine else ("UUID" if ok else "없음")))


def _marker_path(name):
    rd = _WINFO.get("run_dir")
    return Path(rd) / f"running__{name.replace('|', '__')}__{os.getpid()}.json" if rd else None


def _worker_run_t(axis, target, mode, split):
    """작업 단위 하나(워커). 시작 때 GPU 를 확인하고 실행 중 표지(run_dir/running__*.json)를 쓴다. 단위가 끝나면(성공이든 예외든) 표지를 지운다.
    예외 뒤 CUDA 문맥을 쓸 수 없으면 표지를 남긴 채 워커를 끝낸다(부모가 풀을 다시 만들고 표지로 원인 단위를 센다).
    주 모듈에서 정의한 예외는 내장 RuntimeError 로 바꿔 넘긴다(부모 프로세스는 __mp_main__ 의 클래스를 풀 수 없다)."""
    _orphan_check()
    name = unit_name_t((axis, target, mode, split))
    gpu_guard_t()
    mk = _marker_path(name)
    if mk is not None:
        _write_json(mk, dict(unit=name, pid=os.getpid(), gpu=_WINFO.get("gpu"), start=_now()))
    t0 = time.time()
    try:
        u = run_unit_t(_WT, axis, target, mode, split)
    except BaseException as e:                                            # noqa: BLE001
        if _WINFO.get("gpu") not in (None, "") and not cuda_healthy():
            print(f"[worker] {name}: CUDA 문맥을 쓸 수 없다({repr(e)[:160]}). 워커를 끝낸다", flush=True)
            os._exit(76)
        if mk is not None:
            mk.unlink(missing_ok=True)
        if type(e).__module__ == "builtins":
            raise
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:500]}") from None
    if mk is not None:
        mk.unlink(missing_ok=True)
    u["wall_s"] = round(time.time() - t0, 1)
    return unit_summary(u)


# ---------------------------------------------------------------- GPU 확인
def _smi(args):
    return subprocess.check_output(["nvidia-smi"] + list(args), text=True, timeout=60)


def parse_gpu_info(text):
    """nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 의 해석. {번호: dict(uuid, used)}."""
    out = {}
    for line in str(text).strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 3:
            try:
                out[int(p[0])] = dict(uuid=p[1], used=int(float(p[2])))
            except ValueError:
                continue
    return out


def parse_gpu_apps(text):
    """nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 의 해석. [(PID, UUID)]. 계산 프로세스가 없으면 빈 목록."""
    out = []
    for line in str(text).strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 2:
            try:
                out.append((int(p[0]), p[1]))
            except ValueError:
                continue
    return out


def query_gpu_info():
    return parse_gpu_info(_smi(["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]))


def query_gpu_apps():
    return parse_gpu_apps(_smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]))


def query_gpu_memory():
    """nvidia-smi 의 GPU 별 메모리 사용(MiB). 부를 수 없으면 예외를 올린다."""
    return {g: int(v["used"]) for g, v in query_gpu_info().items()}


def gpu_apps_by_index(info, apps):
    """{GPU 번호: [계산 프로세스 PID]}."""
    by = {v["uuid"]: g for g, v in info.items()}
    out: dict = {}
    for pid, u in apps:
        if u in by:
            out.setdefault(by[u], []).append(int(pid))
    return out


def screen_gpus(gpus, used, mem_max=0, reserved=RESERVED_GPUS, apps=None):
    """쓸 수 있는 GPU 목록과 뺀 GPU 의 (번호, 사유) 목록. 남겨 두는 GPU 가 목록에 있으면 거부한다(SystemExit).
    apps = {번호: [PID]} 를 주면 계산 프로세스가 있는 GPU 도 뺀다."""
    res = [int(g) for g in gpus if int(g) in set(reserved)]
    if res:
        raise SystemExit(f"[거부] 남겨 두는 GPU {sorted(set(reserved))} 가 --gpus 에 있다: {res}")
    ok, dropped = [], []
    for g in gpus:
        g = int(g)
        if g not in used:
            dropped.append((g, "nvidia-smi 목록에 없음"))
        elif int(used[g]) > int(mem_max):
            dropped.append((g, f"메모리 사용 {int(used[g])} MiB > {int(mem_max)} MiB"))
        elif apps and apps.get(g):
            dropped.append((g, f"계산 프로세스 {sorted(apps[g])}"))
        else:
            ok.append(g)
    return ok, dropped


def rescreen_gpus(a, cand, excluded=()):
    """풀을 다시 만들 때의 GPU 재확인. 워커가 점유 표지를 남긴 GPU 를 빼고, 종료한 워커의 메모리가 풀리기를 GPU_SETTLE_S 초까지 기다린 뒤
    메모리 사용이 --gpu-mem-max-mib 를 넘거나 계산 프로세스가 있는 GPU 를 뺀다. nvidia-smi 를 부를 수 없으면 빈 목록(더 돌리지 않는다)."""
    cand = [int(g) for g in cand if int(g) not in set(int(x) for x in excluded)]
    t_end = time.time() + float(GPU_SETTLE_S)
    while True:
        try:
            info, apps = query_gpu_info(), query_gpu_apps()
        except Exception as e:                                            # noqa: BLE001
            print(f"[pool] nvidia-smi 확인 실패({repr(e)[:160]}). 풀을 다시 만들지 않는다", flush=True)
            return []
        ok, dropped = screen_gpus(cand, {g: v["used"] for g, v in info.items()}, a.gpu_mem_max_mib, apps=gpu_apps_by_index(info, apps))
        if not dropped or time.time() >= t_end:
            break
        time.sleep(3.0)
    for g_, why in dropped:
        print(f"[pool] GPU {g_} 를 뺀다: {why}", flush=True)
    for g_ in sorted(set(int(x) for x in excluded)):
        print(f"[pool] GPU {g_} 를 뺀다: 워커가 점유 표지를 남겼다", flush=True)
    return ok


def check_run_args(a):
    """실행의 거부 조건. 자료를 읽기 전에 확인한다."""
    if not a.GPUS:
        raise SystemExit("[거부] --gpus 가 비어 있다. TabPFN 은 GPU 에서만 돌린다(CPU 대체 없음). 예: --gpus 9,7,6,5")
    screen_gpus(a.GPUS, {g: 0 for g in a.GPUS})                            # 남겨 두는 GPU 확인
    if not (THREADS_RUN_MIN <= int(a.threads_asked) <= THREADS_MAX):
        raise SystemExit(f"[거부] --threads {a.threads_asked}: 실행은 프로세스당 스레드 {THREADS_RUN_MIN}–{THREADS_MAX}개만 허용한다")
    if not Path(a.MODEL).exists():
        raise SystemExit(f"[거부] TabPFN 가중치 파일이 없다: {a.MODEL}. 내려받기를 시도하지 않는다")


def check_overwrite(a, units):
    """완료 표지(unit.json)가 있는 조각이 있는데 --resume 도 --overwrite 도 없으면 거부한다(스모크 tag 는 예외). 반환 해당 단위 목록."""
    prev = [u for u in units if shard_paths_t(a, *u)["unit"].exists()]
    if prev and not (a.resume or a.overwrite or a.smoke):
        raise SystemExit(f"[거부] 완료 표지가 있는 조각 {len(prev)}개가 있다(예: {unit_name_t(prev[0])}). 이어 돌리려면 --resume, "
                         "전부 다시 돌려 덮어쓰려면 --overwrite 를 준다")
    return prev


# ---------------------------------------------------------------- 실행 제어(잠금, 풀, 종료 코드)
def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False


def acquire_lock(a):
    """실행 잠금(run_<tag>/lock.json, PID 기록). 같은 tag 의 살아 있는 실행이 있으면 거부한다. 죽은 PID 의 잠금은 지우고 다시 만든다."""
    d = Path(a.RUN_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "lock.json"
    for _ in range(2):
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                old = json.loads(p.read_text())
            except (OSError, ValueError):
                old = {}
            pid = int(old.get("pid", 0) or 0)
            if pid and pid_alive(pid):
                raise SystemExit(f"[거부] 같은 tag 의 실행이 이미 있다(PID {pid}, 잠금 {p}). 끝난 실행이면 잠금 파일을 지운다")
            p.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(dict(pid=os.getpid(), start=_now(), run_id=str(a.RUN_ID), argv=list(a.ARGV), host=platform.node()), f, ensure_ascii=False)
        return p
    raise SystemExit(f"[거부] 실행 잠금을 만들 수 없다: {p}")


def release_lock(p):
    """자기 PID 의 잠금만 지운다."""
    try:
        if p is not None and Path(p).exists() and int(json.loads(Path(p).read_text()).get("pid", 0)) == os.getpid():
            Path(p).unlink()
    except (OSError, ValueError):
        pass


def read_run_notes(run_dir, prefix):
    out = []
    for p in sorted(Path(run_dir).glob(f"{prefix}__*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def clear_run_notes(run_dir, prefixes=("running", "gpu_busy")):
    for pre in prefixes:
        for p in Path(run_dir).glob(f"{pre}__*.json"):
            p.unlink(missing_ok=True)


def culprit_pids(exitcodes):
    """풀이 깨진 뒤 워커 종료 코드({PID: exitcode})에서 스스로 끝난 워커. 풀이 나머지 워커에 보낸 SIGTERM(−15)과 정상 종료(0)는 뺀다."""
    return {int(p) for p, c in exitcodes.items() if c is not None and int(c) not in (0, -int(signal.SIGTERM))}


def triage_broken(broken, running, crashes, done_fn):
    """풀이 깨진 뒤 결과를 받지 못한 단위의 분류. running = 원인으로 보는 실행 중 단위 이름의 집합, crashes = 단위 이름별 누적 횟수(갱신한다),
    done_fn(u) = 이번 실행에서 완료된 조각의 unit dict 또는 None. 반환 (완료 [(단위, unit dict)], 격리할 단위, 다시 돌릴 단위)."""
    fin, quar, again = [], [], []
    for u in broken:
        uj = done_fn(u)
        if uj is not None:
            fin.append((u, uj))
            continue
        nm = unit_name_t(u)
        if nm in running:
            crashes[nm] += 1
        (quar if crashes[nm] >= POOL_CRASH_MAX else again).append(u)
    return fin, quar, again


def unit_done_run(a, u):
    """이번 실행(run_id)에서 쓴 완료 조각의 unit dict. 풀이 깨져 결과를 받지 못한 단위를 다시 돌리기 전에 확인한다. 없으면 None."""
    p = shard_paths_t(a, *u)
    if not (a.RUN_ID and p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return None
    try:
        uj = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return None
    ok = uj.get("run_id") == a.RUN_ID and uj.get("cfg_hash") == H.cfg_hash(unit_cfg_t(a, u[0]))
    return uj if ok else None


def kill_pool(ex):
    """풀을 바로 멈춘다: 대기 중인 작업을 취소하고 워커 프로세스를 종료한다(부모의 중단, 신호, GPU 불일치)."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:                                                     # noqa: BLE001
        pass
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:                                                 # noqa: BLE001
            pass
    t_end = time.time() + 15.0
    for p in procs:
        try:
            p.join(timeout=max(t_end - time.time(), 0.1))
            if p.is_alive():
                p.kill()
                p.join(timeout=5.0)
        except Exception:                                                 # noqa: BLE001
            pass
    return [int(p.pid) for p in procs if p.pid is not None]


def _stop_signal(signum, frame):
    raise KeyboardInterrupt(f"신호 {signum}")


def install_stop_handlers():
    """SIGTERM 과 SIGHUP 을 KeyboardInterrupt 로 바꾼다(주 스레드). nohup 등으로 SIGHUP 을 무시하게 띄운 실행은 그대로 둔다."""
    old = {}
    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            if s == signal.SIGHUP and signal.getsignal(s) == signal.SIG_IGN:
                continue
            old[s] = signal.signal(s, _stop_signal)
        except (ValueError, OSError):
            pass
    return old


def restore_handlers(old):
    for s, h in (old or {}).items():
        try:
            signal.signal(s, h)
        except (ValueError, OSError):
            pass


def exit_code(res):
    """종료 코드: 중단 130, 실패(단위 예외, 상태 failed, 집계 실패) 1, 일부 적합 실패(상태 partial)만 있으면 2, 그 밖은 0."""
    if res.get("interrupted"):
        return EXIT_INTERRUPT
    if res.get("n_fail"):
        return EXIT_FAIL
    if res.get("n_partial"):
        return EXIT_PARTIAL
    return 0


# ================================================================ 집계: 조각 읽기와 곡선
def find_shards_t(a, X):
    """주 설정과 민감도 조각(--axes 와 무관하게 있는 것을 모두 읽는다)."""
    out = []
    for axis in AXES:
        out += [dict(s_, axis=axis) for s_ in X.find_shards_x(a.SHARDS, axis_tag(a, axis)) if s_["part"] == "gpu"]
    return out


def read_runs_t(shards):
    frames = []
    for s_ in shards:
        if s_["runs"].stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, ctx_set=str, ctx_sha=str), keep_default_na=False,
                            na_values=["", "nan", "NaN"])
            if len(f):
                f["tag"] = s_["tag"]
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "ctx_set", "ctx_sha"):
        runs[c_] = runs[c_].fillna("").astype(str) if c_ in runs else ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int) if "n_nonfinite" in runs else 0
    return runs.drop_duplicates(subset=["target", "mode", "split"] + list(H.KEY_COLS), keep="first")      # P0·P1 은 두 축에 모두 있다


def check_cfg_t(a, shards, units):
    """조각 사이 설정 일치 확인. (tag, 축)마다 설정 해시가 하나여야 한다. 다르면 중단한다(--allow-mixed-cfg 로 진행)."""
    info = {}
    for key in sorted({(s_["tag"], s_["axis"]) for s_ in shards}):
        us = [u for s_, u in zip(shards, units) if (s_["tag"], s_["axis"]) == key]
        hs = Counter(str(u.get("cfg_hash", "legacy")) for u in us)
        cs = Counter(str(u.get("code_sha", "legacy")) for u in us)
        ch = Counter(str(u.get("code_sha_h40", "legacy")) for u in us)
        ms = Counter(str(u.get("model_sha1", "legacy")) for u in us)
        info["|".join(key)] = dict(cfg_hash=dict(hs), code_sha=dict(cs), code_sha_h40=dict(ch), model_sha1=dict(ms), n=len(us))
        if len(cs) > 1:
            print(f"  [warn] {key}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
        if len(hs) > 1:
            msg = f"{key}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def curve_t(a, X, D, tms, runs, floor, failed):
    """대상·모드의 곡선(h40.build_curve)에 4분 판정 열(기준 P0, P1, 같은 방법의 n = 0)과 효과 크기 열을 붙인다."""
    out = []
    for nm, tm in tms.items():
        rr = runs[(runs.target == tm.target) & (runs["mode"] == tm.mode)] if len(runs) else runs
        try:
            cur = H.build_curve({nm: tm}, rr) if len(rr) else pd.DataFrame()
        except Exception as e:                                            # noqa: BLE001
            print(f"  [warn] 곡선 계산 실패 {nm}: {repr(e)[:200]}", flush=True)
            failed.append(dict(store=nm, target=tm.target, mode=tm.mode, reason=repr(e)[:200]))
            cur = pd.DataFrame()
        if len(cur):
            rec = cur.to_dict("records")
            for kind, pre in (("p0", "d_p0"), ("p1", "d_p1"), ("n0", "net_value")):
                cur[f"verdict4_{kind}"] = [X.verdict4(r_.get(f"{pre}_lo"), r_.get(f"{pre}_hi"), r_.get(f"{pre}_beq_lo"), r_.get(f"{pre}_beq_hi"),
                                                      a.delta_eq) for r_ in rec]
            fl = X.floor_of(floor, D, tm)
            eff = [X.effect_cols(d_, p_, fl) for d_, p_ in zip(cur.d_p0.values, cur.rmse_p0.values)]
            cur["delta_cm"] = [e_["delta_cm"] for e_ in eff]; cur["delta_pct_p0"] = [e_["delta_pct_p0"] for e_ in eff]
            cur["frac_reducible"] = [e_["frac_reducible"] for e_ in eff]; cur["small_effect"] = [e_["small_effect"] for e_ in eff]
            cur["floor_rmse_cm"] = fl
            out.append(cur)
        if hasattr(H4.boot_weights, "cache_clear"):
            H4.boot_weights.cache_clear()                                 # 재표집 행렬은 대상마다 비운다
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ================================================================ 집계: 가설 판정(L32–L37)
def _nl(n):
    return "all" if int(n) == -1 else str(int(n))


def _nt(n):
    return "전량" if int(n) == -1 else f"n={int(n)}"


def _pool_set(r_):
    return frozenset(v for v in str((r_ or {}).get("pool_regions", "")).split(",") if v)


def stability(X, r0, r1):
    """판정 안정성(계획서 §6A 의 L28 과 같은 규칙). 같거나 동등과 미결정 사이의 변화 = 강건, 우세·열세와 동등·미결정 사이의 변화 = 약화,
    우세와 열세가 바뀜 = 의존. 4분 판정이 없는 쪽이 있으면 판정 불가, 풀 지역 집합이 다르면 판정 불가(풀 지역 불일치)."""
    v0, v1 = X._v(r0), X._v(r1)
    if v0 in X.NA_VERDICTS or v1 in X.NA_VERDICTS or not v0 or not v1:
        return "판정 불가"
    if _pool_set(r0) != _pool_set(r1):
        return "판정 불가(풀 지역 불일치)"
    if v0 == v1:
        return "강건"
    if {v0, v1} == {"우세", "열세"}:
        return "의존"
    if (v0 in ("우세", "열세")) != (v1 in ("우세", "열세")):
        return "약화"
    return "강건"


def stab_text(items):
    """L37 안정성 분류 목록 [(이름, 분류)] → 판정 문구(계획서 §6C.6 의 규칙)."""
    states = [s_ for _, s_ in items]
    if any(s_.startswith("판정 불가") for s_ in states):
        return "판정 불가(" + "; ".join(f"{lab} {s_}" for lab, s_ in items if s_.startswith("판정 불가")) + ")"
    if "의존" in states:
        return "의존: " + ", ".join(lab for lab, s_ in items if s_ == "의존") + ". 한계 절에 조건을 적는다"
    if "약화" in states:
        return "약화: " + ", ".join(lab for lab, s_ in items if s_ == "약화")
    return "강건"


def _sub_store(st, keep):
    S, C_ = st.matrices(keep)
    return BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, list(keep), S, C_, dict(st.meta))


def pair_store(st):
    """학습기 짝 대비용 저장소 사본(계획서 §6C.6 실패 처리, 개정 12). 학습기 키(tabpfn, catboost_ctx)는 상대 학습기의 같은
    (method, alpha, placement, n, draw, seed, lam) 키가 있을 때만 남긴다. 물리식 키(learner none)는 그대로 둔다. 반환 (사본, 뺀 키 목록)."""
    other = {TP: CBX, CBX: TP}
    keep, drop = [], []
    for k in st.keys:
        if k[1] in other and (tuple(k[:1]) + (other[k[1]],) + tuple(k[2:])) not in st:
            drop.append(k)
        else:
            keep.append(k)
    return (_sub_store(st, keep), drop) if drop else (st, [])


def pair_stores(stores):
    """{(이름, 분할): 저장소} 전체에 pair_store 를 적용한다. 반환 (사본 dict, {(이름, 분할): 뺀 키 목록})."""
    out, drop = {}, {}
    for k, st in stores.items():
        out[k], dr = pair_store(st)
        if dr:
            drop[k] = dr
    return out, drop


def draw_filter(draws):
    """추출 번호가 draws 에 드는 키만 남기는 저장소 변환(P0 은 추출 0 이므로 남는다)."""
    keep_d = {int(d) for d in draws}

    def fn(st):
        drop = [k for k in st.keys if int(k[5]) not in keep_d]
        return (_sub_store(st, [k for k in st.keys if int(k[5]) in keep_d]), drop) if drop else (st, [])
    return fn


def tm_view(tm, fn):
    """TMx 사본. 저장소마다 fn(저장소) → (사본, 뺀 키 목록)을 적용하고 키 색인을 h40.TMx 와 같은 규칙으로 다시 만든다. 분할 구조와 속성은
    그대로다. 반환 (사본, {분할: 뺀 키 목록})."""
    t2 = copy.copy(tm)
    new, dropped = {}, {}
    for sp, st in tm.by_all.items():
        new[sp], dr = fn(st)
        if dr:
            dropped[sp] = dr
    t2.by_all = new
    t2.by_valid = {sp: new[sp] for sp in tm.by_valid}
    t2.used = {sp: new[sp] for sp in tm.used}
    t2.idx = {}
    for sp, st in t2.used.items():
        grp: dict = {}
        for k in st.keys:
            grp.setdefault((k[0], k[1], str(k[2]), k[3], int(k[4]), float(k[7])), []).append(k)
        t2.idx[sp] = grp
    return t2, dropped


def build_tests_t(a, tms, D, floor, X=None):
    """L32–L37 의 대비 행과 판정 행(계획서 §6C.6). LGT 의 가설은 모두 보조다(h42.CONFIRMATORY 에 없다). 분할이 기대보다 적은 대비를 쓴
    판정에는 h42.TestBook.verdict 가 '부분(분할 k/K): '을 붙인다. Holm 묶음(item 'LGT' 의 primary 행)은 L32 의 기준 λ 대비 5개,
    L33 의 6개, L34 의 2개, L35 의 1개다. L32 의 동등성 p 는 같은 묶음 안에서 따로 보정한다(eq_test).
    키 집합(key_set 열, 개정 12): all = 저장된 키 전부, pair = 두 학습기에 모두 있는 키만(pair_store. 학습기 짝 대비 L32, L37 의
    R1[T] − R1[C]), d01 = 추출 0–1 만(L37 의 보조 기준 행), pair_d01 = 둘 다. 짝 대비 행의 n_unpaired 는 그 대비의 두 곡선 키에서
    짝이 없어 뺀 키 수다(층화 평균에 든 지역과 사용 분할의 합)."""
    X = X or _load_h42()
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    V = T.verdict
    ITEM, ALL = "LGT", -1
    LB = float(H.LAM_BASE)
    memo: dict = {}
    views: dict = {"all": tms}
    unp: dict = {}                                                        # 키 집합 → 이름 → Counter((method, alpha, placement, n, lam) → 뺀 키 수)

    def get_view(name):
        if name not in views:
            out, cnt_by = {}, {}
            for nm, tm in tms.items():
                t2, cnt = tm, Counter()
                if "pair" in name:
                    t2, dr = tm_view(t2, pair_store)
                    for sp, ks in dr.items():
                        if sp in tm.used:
                            for k in ks:
                                if "d01" not in name or int(k[5]) in AUX_DRAWS:
                                    cnt[(k[0], str(k[2]), k[3], int(k[4]), float(k[7]))] += 1
                if "d01" in name:
                    t2, _ = tm_view(t2, draw_filter(AUX_DRAWS))
                out[nm], cnt_by[nm] = t2, cnt
            views[name], unp[name] = out, cnt_by
        return views[name]

    def n_unpaired(name, gA, gB, names):
        tab = unp.get(name, {})
        grp = {(q[0], str(q[2]), q[3], int(q[4]), float(q[5])) for q in (gA, gB)}
        return int(sum(tab.get(nm, Counter())[q] for nm in names for q in grp))

    def C(test, label, gA, gB, view="all", **kw):
        k = (test, label, tuple(kw.get("names") or ()), view)
        if k not in memo:
            kw.setdefault("role", "보조")
            vw = get_view(view)
            if view.startswith("pair"):
                names = list(kw.get("names") or T.m4) + ([T.m3[2]] if kw.get("aux3", True) and not kw.get("names") else [])
                kw["n_unpaired"] = n_unpaired(view, gA, gB, names)
            T.tms = vw
            try:
                memo[k] = T.contrast(test, ITEM, label, gA, gB, key_set=view, **kw)
            finally:
                T.tms = tms
        return memo[k]

    def g(method, n, lr=None, lam=None, alpha="1"):
        return X.G(method, n, lam, lr, alpha)

    def single(test, label, nm, gA, gB, view="all", **kw):
        tm = get_view(view).get(nm)
        if view.startswith("pair"):
            kw["n_unpaired"] = n_unpaired(view, gA, gB, [nm])
        return T.single(test, ITEM, label, nm, X.region_stats(tm, gA, gB) if tm is not None else None, key_set=view, **kw)

    rows_1000 = ["Lena|x", f"{H.ALASKA}|x"]                                 # n = 1,000 은 층화 평균을 내지 않고 지역 행으로 보고한다

    # ---------------- L32 학습기 동등성: R1[T] − R1[C]
    l32_n = (0, 10, 40, 160, ALL)
    base32 = [(_nt(n), C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TP, LB), g("R1", n, CBX, LB), view="pair", n=n, lam=LB,
                         eq_test=True, blind=n not in NB_SEEN)) for n in l32_n]
    lam32 = [(f"{_nt(n)} λ=1.0", C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam1.0", g("R1", n, TP, 1.0), g("R1", n, CBX, 1.0), view="pair", n=n, lam=1.0,
                                   eq_test=True, primary=False, blind=n not in NB_SEEN, role="보조(λ 1.0)", aux3=False)) for n in l32_n]
    for n in (3, 320):
        C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TP, LB), g("R1", n, CBX, LB), view="pair", n=n, lam=LB, primary=False,
          blind=n not in NB_SEEN, role="보조(추가 n)")
    for nm in rows_1000:
        single("L32", f"R1[T]-R1[C]|n1000|lam{LB}", nm, g("R1", 1000, TP, LB), g("R1", 1000, CBX, LB), view="pair", n=1000, lam=LB,
               role="보조(지역 행)")
    for m_, lam in (("D0", 1.0), ("R0", LB)):                            # TabPFN 의 D0, R0 은 기존 결과가 없다(계획서 §6C.7): 모든 n 이 맹검
        for n in l32_n:
            C("L32", f"{m_}[T]-{m_}[C]|n{_nl(n)}", g(m_, n, TP, lam), g(m_, n, CBX, lam), view="pair", n=n, lam=lam, primary=False, blind=True,
              role=f"보조({m_})", aux3=False)
    vb, v1 = [X._v(r_) for _, r_ in base32], [X._v(r_) for _, r_ in lam32]
    k, m, _ = X.count_valid(base32)
    k1, m1, _ = X.count_valid(lam32)
    dom = [(lab, r_) for (lab, r_), v in zip(base32, vb) if v in ("우세", "열세")]
    dom1 = [f"{lab} TabPFN {v}" for (lab, _), v in zip(lam32, v1) if v in ("우세", "열세")]
    used32, part32 = base32, None
    if dom:
        txt = "TabPFN 과 같은 컨텍스트 CatBoost 의 차이가 있는 라벨 수: " + ", ".join(f"{lab} TabPFN {r_['verdict4']}(Δ {r_['delta']:+.2f} cm)"
                                                                                  for lab, r_ in dom)
        used32, part32 = dom, X.part_mark(base32)
    elif k < m:
        txt = X.na_text(base32)
    elif all(v == "동등" for v in vb) and k1 < m1:
        txt = X.na_text(base32 + lam32)
    elif all(v == "동등" for v in vb + v1):
        txt = f"동등(한계 {float(a.delta_eq):g} cm, λ {LB:g} 와 1.0)"
        used32 = base32 + lam32
    elif all(v == "동등" for v in vb) and dom1:                           # 개정 12: 기준 λ 는 동등, λ 1.0 에 우세 또는 열세
        txt = f"기준 λ {LB:g} 에서는 동등, λ 1.0 에서는 차이가 있다({', '.join(dom1)}). 동등으로 쓰지 않는다"
        used32 = base32 + [(lab, r_) for (lab, r_), v in zip(lam32, v1) if v in ("우세", "열세")]
        dom1 = []
    else:
        txt = "차이를 확인하지 못함"
    if dom1 and not txt.startswith("판정 불가"):
        txt += f" [λ 1.0 의 대비(병기): {', '.join(dom1)}]"
    V("L32", ITEM, txt, X._fmt(base32 + lam32), role="보조", blind=False, used=used32, partial=part32,
      note="재현(비맹검): n ∈ {3, 10, 40} 은 b4 에서 방향을 보았다. n = 0, 160, 전량의 대비 행은 맹검이다")

    # ---------------- L33 순가치: R1[T] − P1
    l33_n = (3, 10, 40, 160, 320, ALL)
    resT = [(_nt(n), n, C("L33", f"R1[T]-P1|n{_nl(n)}", g("R1", n, TP), g("P1", n), n=n, lam=LB, blind=n not in NB_SEEN)) for n in l33_n]
    resC = [(_nt(n), n, C("L33", f"R1[C]-P1|n{_nl(n)}", g("R1", n, CBX), g("P1", n), n=n, lam=LB, primary=False, blind=n not in NB_SEEN,
                          role="보조(병기 CatBoost)")) for n in l33_n]
    for nm in rows_1000:
        single("L33", "R1[T]-P1|n1000", nm, g("R1", 1000, TP), g("P1", 1000), n=1000, lam=LB, role="보조(지역 행)")
        single("L33", "R1[C]-P1|n1000", nm, g("R1", 1000, CBX), g("P1", 1000), n=1000, lam=LB, role="보조(지역 행, 병기 CatBoost)")
    pT, pC = [(lab, r_) for lab, _, r_ in resT], [(lab, r_) for lab, _, r_ in resC]
    order33 = {n: i for i, n in enumerate(l33_n)}
    ok33 = [(lab, n, r_) for lab, n, r_ in resT if X.valid_row(r_)]
    win = sorted([q for q in ok33 if q[2]["verdict4"] == "우세"], key=lambda q: order33[q[1]])
    lose = [q for q in ok33 if q[2]["verdict4"] == "열세"]
    w4 = [q for q in win if int(q[2].get("n_ci_regions", 0)) >= len(T.m4)]
    wp = [q for q in win if int(q[2].get("n_ci_regions", 0)) < len(T.m4)]
    s4 = [q for q in w4 if 0 < q[1] <= 40]
    sp_ = [q for q in wp if 0 < q[1] <= 40]
    if not ok33:
        txt, used33 = X.na_text(pT), pT
    elif s4 or sp_:
        txt, q = "희소 라벨에서도 TabPFN 잔차의 순가치가 있다", (s4 or sp_)[0]
        used33 = [(q[0], q[2])]
    elif win:
        txt, used33 = "TabPFN 잔차의 순가치는 n > 40 에서만 확인되었다", [(win[0][0], win[0][2])]
    else:
        txt, used33 = "TabPFN 잔차의 재보정 물리식 대비 순가치는 확인되지 않았다", [(lab, r_) for lab, _, r_ in ok33]
    if ok33:
        txt += f"(우세인 최소 n: 4지역 평균 {w4[0][0] if w4 else '없음'}, 2–3지역 평균 {wp[0][0] if wp else '없음'})"
        if lose:
            txt += f". 열세인 n: {', '.join(q[0] for q in lose)}"
    diff = [lab for (lab, _, rt), (_, _, rc) in zip(resT, resC) if X.valid_row(rt) and X.valid_row(rc) and rt["verdict4"] != rc["verdict4"]]
    V("L33", ITEM, txt, f"{X._fmt(pT)} | 병기 CatBoost: {X._fmt(pC)} | 두 학습기의 4분 판정이 다른 n: {', '.join(diff) if diff else '없음'}",
      role="보조", blind=False, used=used33, partial=X.part_mark(pT) if ok33 else None,
      note="재현(비맹검): n ∈ {3, 10, 40} 은 b4 에서 방향을 보았다. n ≥ 160 과 전량의 대비 행은 맹검이다")

    # ---------------- L34 라벨 0
    ra = C("L34", "D0[T]-P0|n0", g("D0", 0, TP), g("P0", 0), n=0, lam=1.0)
    rb = C("L34", "R0[T]-P0|n0", g("R0", 0, TP), g("P0", 0), n=0, lam=LB)
    ca = C("L34", "D0[C]-P0|n0", g("D0", 0, CBX), g("P0", 0), n=0, lam=1.0, primary=False, blind=False, role="보조(병기 CatBoost)")
    cb = C("L34", "R0[C]-P0|n0", g("R0", 0, CBX), g("P0", 0), n=0, lam=LB, primary=False, blind=False, role="보조(병기 CatBoost)")
    ua = [("D0[T]-P0", ra)]
    if not X.valid_row(ra):
        txt = X.na_text(ua)
    elif X._v(ra) == "우세":
        txt = ("기각: TabPFN 직접 예측이 라벨 0 에서 P0 보다 우세하다. L1 의 결론을 'TabPFN 을 제외한 학습기 한정'으로 적고 TabPFN 의 곡선을 "
               "마스터 곡선에 더한다")
    else:
        txt = f"지지: 표형 파운데이션 모델을 학습기로 써도 직접 ML 은 라벨 0 전이에서 물리식을 넘지 못한다(4분 판정 {X._v(ra)})"
    V("L34", ITEM, txt, X._fmt(ua + [("병기 D0[C]-P0", ca)]), role="보조", blind=True, used=ua, clause="(a) 직접 예측")
    ub = [("R0[T]-P0", rb)]
    say = dict(우세="라벨 0 에서 TabPFN 잔차는 원천 계수 물리식보다 오차가 작다", 열세="라벨 0 에서 TabPFN 잔차는 원천 계수 물리식보다 오차가 크다",
               동등=f"라벨 0 에서 TabPFN 잔차와 원천 계수 물리식은 구별되지 않는다(한계 {float(a.delta_eq):g} cm)", 미결정="차이를 확인하지 못함")
    V("L34", ITEM, X.na_text(ub) if not X.valid_row(rb) else say.get(X._v(rb), X._v(rb)), X._fmt(ub + [("병기 R0[C]-P0", cb)]), role="보조",
      blind=True, used=ub, clause="(b) 잔차(방향 중립)")

    # ---------------- L35 전량
    r35 = C("L35", "R1[T]-P0|all", g("R1", ALL, TP), g("P0", 0), n=ALL, lam=LB)
    c35 = C("L35", "R1[C]-P0|all", g("R1", ALL, CBX), g("P0", 0), n=ALL, lam=LB, primary=False, blind=False, role="보조(병기 CatBoost)")
    d35 = C("L35", "D0[T]-P0|all", g("D0", ALL, TP), g("P0", 0), n=ALL, lam=1.0, primary=False, role="보조(병기 D0)")
    u35 = [("R1[T]-P0 전량", r35)]
    if not X.valid_row(r35):
        txt = X.na_text(u35)
    else:
        txt = "지지" if X._v(r35) == "우세" else f"기각(4분 판정 {X._v(r35)})"
    V("L35", ITEM, txt, X._fmt(u35 + [("병기 R1[C]-P0", c35), ("병기 D0[T]-P0", d35)]), role="보조", blind=True, used=u35,
      note="러시아 W·E 의 전량은 라벨 14–17개다")

    # ---------------- L36 직접 대 잔차(방향 중립): D0[T](λ 1.0) − R1[T](λ 0.25)
    r36 = [(_nt(n), C("L36", f"D0[T]-R1[T]|n{_nl(n)}", g("D0", n, TP), g("R1", n, TP), n=n, primary=False)) for n in l32_n]
    v36 = [X._v(r_) for _, r_ in r36]
    k, m, _ = X.count_valid(r36)
    if k < m:
        txt = X.na_text(r36)
    elif all(v == "열세" for v in v36):
        txt = "TabPFN 에서도 앵커 + 잔차 구조가 직접 구조보다 오차가 작다"
    elif any(v == "우세" for v in v36):
        up = [lab for (lab, _), v in zip(r36, v36) if v == "우세"]
        txt = (f"{', '.join(up)} 에서는 TabPFN 직접 예측이 앵커 + 잔차보다 오차가 작다. 나머지: "
               + ", ".join(f"{lab} {v}" for (lab, _), v in zip(r36, v36) if v != "우세"))
    else:
        txt = "n 별 서술: " + ", ".join(f"{lab} {v}" for (lab, _), v in zip(r36, v36))
    V("L36", ITEM, txt, X._fmt(r36), role="보조", blind=True, used=r36)

    # ---------------- L37 컨텍스트 구성 민감도
    variants = [("d3", SENS_SETS["d3"], (10, 40), False), ("d10", SENS_SETS["d10"], (10, 40), True),
                ("rid", SENS_SETS["rid"], (10, 40), False), ("tgt", SENS_SETS["tgt"], (40,), True)]
    aux = f"추출 {AUX_DRAWS[0]}–{AUX_DRAWS[-1]}"
    for v, vs, ns_, blind in variants:
        al, mv = str(vs["alpha"]), "R1" + str(vs["sfx"])
        res, stab, dv, base_rows = [], [], [], []
        for n in ns_:
            pairs = [("R1[T]-R1[C]", "pair", g("R1", n, TP), g("R1", n, CBX), g(mv, n, TP, alpha=al), g(mv, n, CBX, alpha=al)),
                     ("R1[T]-P1", "all", g("R1", n, TP), g("P1", n), g(mv, n, TP, alpha=al), g("P1", n))]
            for lab, ks, a0, b0, a1, b1 in pairs:
                ks_aux = "pair_d01" if ks == "pair" else "d01"
                r0 = C("L37", f"기준|{lab}|n{_nl(n)}", a0, b0, view=ks, n=n, lam=LB, primary=False, blind=False, role="기준", aux3=False)
                r0b = C("L37", f"기준({aux})|{lab}|n{_nl(n)}", a0, b0, view=ks_aux, n=n, lam=LB, primary=False, blind=False, role=f"기준({aux}, 보조)",
                        aux3=False)
                r1 = C("L37", f"{v}|{lab}|n{_nl(n)}", a1, b1, view=ks, n=n, lam=LB, primary=False, blind=blind, role="보조(변형)", aux3=False, variant=v)
                s_, sb = stability(X, r0, r1), stability(X, r0b, r1)
                res.append((f"{lab} {_nt(n)}", r1)); base_rows.append((f"기준 {lab} {_nt(n)}", r0))
                stab.append((f"{lab} {_nt(n)}", s_, X._v(r0), X._v(r1), sb, X._v(r0b)))
                T.rows.append(dict(test_id="L37", item=ITEM, contrast=f"{v}|{lab}|n{_nl(n)}", scope="stability", variant=v, n=n, lam=LB, role="보조",
                                   primary=False, blind=bool(blind), verdict4_base=X._v(r0), verdict4_variant=X._v(r1), stability=s_,
                                   verdict4_base_d01=X._v(r0b), stability_d01=sb, stability_same=bool(s_ == sb)))
            dv.append((_nt(n), C("L37", f"{v}-main|R1[T]|n{_nl(n)}", g(mv, n, TP, alpha=al), g("R1", n, TP), n=n, lam=LB, primary=False,
                                 blind=blind, role="보조(변형 − 주 설정)", aux3=False, variant=v)))
        txt = stab_text([(lab, s_) for lab, s_, _, _, _, _ in stab])
        txt_b = stab_text([(lab, sb) for lab, _, _, _, sb, _ in stab])
        det = "; ".join(f"{lab}: {s_}(주 설정 {b0}, 변형 {b1})" for lab, s_, b0, b1, _, _ in stab)
        det_b = "; ".join(f"{lab}: {sb}(주 설정 {aux} {bb})" for lab, _, _, _, sb, bb in stab)
        mkb = T.marks(base_rows)                                          # 기준 행의 분할 완결성도 부분 표기에 넣는다(개정 12)
        part = [f"분할 {mkb['splits'][0]}/{mkb['splits'][1]}(기준 행)"] if mkb["splits"] is not None else []
        V("L37", ITEM, txt, f"{det} | 보조 기준({aux}): {det_b} · 두 기준의 분류 {'같음' if txt == txt_b else '다름'} | 변형 − 주 설정(R1[T]): "
          f"{X._fmt(dv)}", role="보조", blind=bool(blind), used=res, partial=part, variant=v, clause="안정성", stability_d01=txt_b,
          stability_same=bool(txt == txt_b))
        if v == "tgt":                                                    # 변형 − 주 설정 대비는 안정성 판정과 따로 적는다(개정 12)
            if X.count_valid(dv)[0] < len(dv):
                t2 = X.na_text(dv)
            else:
                say = dict(우세="원천 컨텍스트는 TabPFN 잔차 예측에 기여하지 않는다", 열세="원천 컨텍스트가 기여한다")
                t2 = ". ".join(f"{say.get(X._v(r_), '변형 − 주 설정 4분 판정 ' + X._v(r_))}({lab})" for lab, r_ in dv)
            V("L37", ITEM, t2, f"변형 − 주 설정(R1[T]): {X._fmt(dv)}", role="보조", blind=bool(blind), used=dv, variant=v, clause="변형 − 주 설정",
              note="tgt 의 n = 40 은 컨텍스트가 40행이라 TabPFN 의 범주형 추론이 꺼진다(100행 이하). 원천 행 제거 외에 입력 처리도 다르다")
    return T.frame()


# ================================================================ 집계: 교차 비교(보조)와 재현 점검
def gate_compare(st, sb):
    """재현 점검 한 단위. LGT 저장소 st 의 P0, P1 키(학습기 none, cell) 가운데 기준 저장소 sb 에 없는 키 수(n_missing_ref)와 공통 키의
    셀 가중 RMSE 최대 차. 통과 = 공통 키가 있고, 없는 키가 0 이고, 최대 차가 GATE_TOL 이하이고, 채점 블록 집합이 같다."""
    mine = [k for k in st.keys if k[0] in ("P0", "P1") and k[1] == "none" and k[3] == "cell"]
    keys = [k for k in mine if k in sb]
    worst = (0.0, "")
    for k in keys:
        d_ = abs(st.rmse(k) - sb.rmse(k))
        if d_ >= worst[0]:
            worst = (float(d_), json.dumps(list(k)))
    beq = bool(np.array_equal(st.blocks, sb.blocks) and np.array_equal(st.ncell, sb.ncell))
    miss = len(mine) - len(keys)
    return dict(n_keys=len(keys), n_missing_ref=int(miss), max_diff=worst[0] if keys else np.nan, worst_key=worst[1], blocks_equal=beq,
                passed=bool(keys and miss == 0 and worst[0] <= GATE_TOL and beq))


def cross_tables(a, X, D, floor, stores):
    """본 실행 cpu 조각(읽기 전용)과의 재현 점검과 교차 비교. 반환 (gate 표, cross 표, 사유).
    재현 점검은 P0, P1 의 키별 셀 가중 RMSE 를 대조한다(허용 차 1e-9 cm, gate_compare). LGT 의 P0, P1 키 가운데 기준 조각에 없는 키가
    있으면 통과로 두지 않는다. 통과한 단위만 catboost_lo(alpha '1', cell)의 D0·R0·R1 키를 저장소 사본에 병합해 R1[T] − R1[catboost_lo],
    R1[C] − R1[catboost_lo] 를 계산한다. stores 는 짝 맞춘 저장소(pair_stores)를 받는다. 가설 판정에 쓰지 않는다."""
    ref = {(s_["target"], s_["mode"], int(s_["split"])): s_ for s_ in X.find_shards_x(a.LGDIR / "shards", a.lg_tag) if s_["part"] == "cpu"}
    if not ref:
        return pd.DataFrame(), pd.DataFrame(), f"본 실행 조각 없음({a.LGDIR / 'shards'}, tag {a.lg_tag}, 부분 cpu)"
    h40_now = H.code_sha()
    gate, merged = [], {}
    for (nm, sp), st in sorted(stores.items()):
        t, m = nm.split("|")
        row = dict(target=t, mode=m, split=int(sp), status="", n_keys=0, n_missing_ref=np.nan, max_diff=np.nan, worst_key="", blocks_equal=None,
                   passed=False, code_sha_h40_ref="", code_sha_h40_now=h40_now, tol=GATE_TOL)
        s_ = ref.get((t, m, int(sp)))
        if s_ is None:
            row["status"] = "기준 조각 없음"; gate.append(row); continue
        try:
            sb = load_stores(s_["npz"]).get((nm, int(sp)))
            try:
                row["code_sha_h40_ref"] = str(json.loads(s_["unit"].read_text()).get("code_sha", ""))
            except (OSError, ValueError):
                pass
            if sb is None:
                row["status"] = "저장소 없음"; gate.append(row); continue
            row.update(gate_compare(st, sb))
            row["status"] = "ok" if row["passed"] else ("기준에 없는 키" if row["n_missing_ref"] else "불일치")
            if row["passed"]:
                S, C_ = st.matrices(st.keys)
                cp = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, list(st.keys), S, C_, dict(st.meta))
                for k in sb.keys:
                    if k[0] in ("D0", "R0", "R1") and k[1] == H.BASE_LEARNER and str(k[2]) == "1" and k[3] == "cell":
                        cp.add_sse(k, *sb.get(k))
                merged[(nm, int(sp))] = cp
        except Exception as e:                                            # noqa: BLE001
            row["status"] = f"오류: {repr(e)[:120]}"
        gate.append(row)
    gate = pd.DataFrame(gate)
    if not merged:
        return gate, pd.DataFrame(), "재현 점검을 통과한 단위가 없다"
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in merged.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in merged})}
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    LB = float(H.LAM_BASE)
    note = "교차 환경(보조): 실행 환경(하드웨어, python 과 라이브러리 판)과 교락된다. 가설 판정에 쓰지 않는다"
    for n in (0, 10, 40, 160, -1):
        for lr, lab in ((TP, "T"), (CBX, "C")):
            T.contrast("LGT-cross", "LGT", f"R1[{lab}]-R1[catboost_lo]|n{_nl(n)}", X.G("R1", n, LB, lr), X.G("R1", n, LB, H.BASE_LEARNER), n=n, lam=LB,
                       primary=False, blind=True, role="교차 환경(보조)", note=note)
        T.contrast("LGT-cross", "LGT", f"D0[C]-D0[catboost_lo]|n{_nl(n)}", X.G("D0", n, 1.0, CBX), X.G("D0", n, 1.0, H.BASE_LEARNER), n=n, lam=1.0,
                   primary=False, blind=True, role="교차 환경(보조)", note=note, aux3=False)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    return gate, T.frame(), ""


# ================================================================ 집계: 스모크 점검
def smoke_report(units, runs, stores):
    """스모크에서 확인할 것: 두 학습기가 같은 행렬을 받았는지, n = 0 의 R1 = R0, 묶음 예측과 한 번 예측의 차이, 적합 1건 시간, GPU 메모리."""
    out = dict(same_matrix=None, n_pairs=0, r1_eq_r0_n0=None, n_r1_r0=0, chunk_maxdiff=None, chunk_ok=None, gpu_mem_peak_mib=None, sec_per_fit={})
    ml = runs[(runs.learner != "none") & (runs.ctx_sha != "")] if len(runs) else runs
    if len(ml):
        K = ["target", "mode", "split", "ctx_set", "method", "alpha", "n", "draw", "seed"]
        gsha = ml.drop_duplicates(subset=K + ["learner"]).groupby(K).ctx_sha.agg(["nunique", "size"])
        gsha = gsha[gsha["size"] >= 2]
        out.update(same_matrix=bool((gsha["nunique"] == 1).all()) if len(gsha) else None, n_pairs=int(len(gsha)))
    diffs = []
    for st in stores.values():
        for k in st.keys:
            if k[0] == "R1" and int(k[4]) == 0:
                k0 = ("R0",) + tuple(k[1:])
                if k0 in st:
                    diffs.append(float(np.max(np.abs(st.get(k)[0] - st.get(k0)[0]))))
    if diffs:
        out.update(r1_eq_r0_n0=bool(max(diffs) == 0.0), n_r1_r0=len(diffs))
    ch = [float(q["chunk_maxdiff"]) for u in units for q in (u.get("smoke_check") or []) if "chunk_maxdiff" in q]
    if ch:
        out.update(chunk_maxdiff=float(max(ch)), chunk_ok=bool(max(ch) <= SMOKE_CHUNK_TOL))
    pk = [float(u["gpu_mem_peak_mib"]) for u in units if u.get("gpu_mem_peak_mib") is not None]
    if pk:
        out["gpu_mem_peak_mib"] = float(max(pk))
    rv = [float(u["gpu_mem_reserved_peak_mib"]) for u in units if u.get("gpu_mem_reserved_peak_mib") is not None]
    out["gpu_mem_reserved_peak_mib"] = float(max(rv)) if rv else None
    cb = [float(q["cb_pool_maxdiff"]) for u in units for q in (u.get("smoke_check") or []) if "cb_pool_maxdiff" in q]
    out["cb_pool_maxdiff"] = float(max(cb)) if cb else None               # Pool 스레드 지정(h43) 대 h40.cb_fit 경로의 예측 차(cm)
    sch: dict = {}
    for u in units:
        for k, v in (u.get("tabpfn_schema") or {}).items():
            slot = sch.setdefault(k, {})
            for sig, c_ in v.items():
                slot[sig] = int(slot.get(sig, 0)) + int(c_)
    out["tabpfn_schema"] = sch
    out["device_check"] = sorted({str(u.get("device_check", "")) for u in units})
    return out


def smoke_timing(runs, units):
    """스모크: 적합 1건 시간(변형, 학습기, 방법, n 별)과 계획서 §6C.10 추정식 대비 실측 비(학습기별). 키마다 한 번 센다(λ 행의 중복 제거).
    적합 없이 재사용한 행(n = 0 의 R1, fit_s 0)은 뺀다. 결과 값(RMSE)은 읽지 않는다."""
    if not len(runs):
        return pd.DataFrame(), {}
    ml = runs[(runs.learner != "none") & (runs.fit_s > 0)]
    ml = ml.drop_duplicates(subset=["target", "mode", "split", "ctx_set", "method", "learner", "alpha", "n", "draw", "seed"]).copy()
    if not len(ml):
        return pd.DataFrame(), {}
    ne = {(str(u["target"]), str(u["mode"]), int(u["split"])): int(u.get("n_eval", 0) or 0) for u in units}
    ml["n_eval"] = [ne.get((str(t), str(m), int(s)), 0) for t, m, s in zip(ml.target, ml["mode"], ml.split)]
    ml["est_s"] = [est_fit_s(lr, nc, nv) for lr, nc, nv in zip(ml.learner, ml.n_ctx, ml.n_eval)]
    tab = ml.groupby(["ctx_set", "learner", "method", "n"], as_index=False).agg(
        n_fit=("fit_s", "size"), sec_mean=("fit_s", "mean"), sec_max=("fit_s", "max"), n_ctx_mean=("n_ctx", "mean"), n_eval_mean=("n_eval", "mean"),
        est_mean=("est_s", "mean"))
    for c_ in ("sec_mean", "sec_max", "n_ctx_mean", "n_eval_mean", "est_mean"):
        tab[c_] = tab[c_].astype(float).round(3)
    ratio = {str(lr): round(float(g_.fit_s.sum() / max(float(g_.est_s.sum()), 1e-9)), 3) for lr, g_ in ml.groupby("learner")}
    return tab, ratio


# ================================================================ 집계
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                                     # noqa: BLE001
        return "NA"


def failed_rows(runs):
    """저장하지 못한 키(비유한 예측 또는 적합 실패). chunk, tsub 표지는 실패가 아니다(저장된 키다)."""
    if not len(runs):
        return runs
    bad = runs.fit_flag.astype(str).map(lambda v: any(q in ("fail", "nonfinite") for q in v.split(";")))
    return runs[(runs.n_nonfinite > 0) | bad]


def summarize(a, elapsed=0.0, skipped=None):
    """집계(로컬, 프로세스 1개). 반환 dict 의 n_fail 은 저장하지 못한 키, 상태가 ok 가 아닌 조각, 곡선 계산 실패의 합이다."""
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    X = _load_h42()
    sh = find_shards_t(a, X)
    if not sh:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{axis_tag(a, 'main')}__gpu__*, {axis_tag(a, 'sens')}__gpu__*", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    cfg_info = check_cfg_t(a, sh, units)
    runs = read_runs_t(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    stores_pair, unpaired = pair_stores(stores)                           # 학습기 짝 대비용(개정 12). 판정 표는 build_tests_t 가 같은 규칙으로 만든다
    unp_lr = Counter(k[1] for ks in unpaired.values() for k in ks)
    if unpaired:
        print(f"[summarize] 짝이 없는 학습기 키 {sum(unp_lr.values()):,}개(학습기별 {dict(unp_lr)}, 단위 {len(unpaired)}개)는 학습기 짝 대비에서 뺀다",
              flush=True)
    D = H.get_data(h40_args_t(a, "main"))
    floor, floor_meta = X.floor_table(D.df)
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    curve_failed: list = []
    cur = curve_t(a, X, D, tms, runs, floor, curve_failed)
    mn = H.build_minn(cur) if len(cur) else pd.DataFrame()
    tests = build_tests_t(a, tms, D, floor, X)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    gate, cross, cross_note = pd.DataFrame(), pd.DataFrame(), "교차 비교를 켜지 않았다(--cross-lg)"
    if a.cross_lg:
        if a.SUFFIX:
            cross_note = "스모크 조각은 본 실행 조각과 교차 비교하지 않는다"
        else:
            gate, cross, cross_note = cross_tables(a, X, D, floor, stores_pair)
        if cross_note:
            print(f"[summarize] 교차 비교: {cross_note}", flush=True)
    failed = failed_rows(runs)
    flagged = runs[runs.fit_flag.astype(str) != ""] if len(runs) else runs
    tg = pd.DataFrame([dict({k: v for k, v in u.items() if not isinstance(v, (dict, list))}, n_fit=json.dumps(u.get("n_fit", {})),
                            sec=json.dumps(u.get("sec", {})), fail=json.dumps(u.get("fail", {})),
                            errors=json.dumps(u.get("errors", []), ensure_ascii=False), env=json.dumps(u.get("env", {}), ensure_ascii=False),
                            ctx=json.dumps(u.get("ctx", {}), ensure_ascii=False), tabpfn_schema=json.dumps(u.get("tabpfn_schema", {})),
                            failed_learners=",".join(u.get("failed_learners", []) or [])) for u in units])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    tt = H.timing_table(units)
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    if a.smoke:                                                           # 스모크: 결과 표(곡선, 최소 n, 판정, 교차)는 쓰지 않는다(개정 12)
        for nm_ in ("curve", "minn", "tests", "cross", "gate"):
            (O / f"{a.TAG}_{nm_}.csv").unlink(missing_ok=True)
    else:
        _atomic_csv(cur, O / f"{a.TAG}_curve.csv"); _atomic_csv(mn, O / f"{a.TAG}_minn.csv"); _atomic_csv(tests, O / f"{a.TAG}_tests.csv")
        _atomic_csv(cross, O / f"{a.TAG}_cross.csv"); _atomic_csv(gate, O / f"{a.TAG}_gate.csv")
    _atomic_csv(tt, O / f"{a.TAG}_timing.csv")
    fa = failed.copy() if len(failed) else pd.DataFrame(columns=list(runs.columns) if len(runs) else RUN_COLS)
    if curve_failed:
        fa = pd.concat([fa, pd.DataFrame([dict(target=q["target"], mode=q["mode"], fit_flag="curve_failed", method=q["reason"]) for q in curve_failed])],
                       ignore_index=True)
    _atomic_csv(fa, O / f"{a.TAG}_failed.csv")
    _atomic_csv(tg.sort_values(["target", "mode", "split", "axis"]) if len(tg) else tg, O / f"{a.TAG}_targets.csv")
    status = Counter(str(u.get("status", "ok")) for u in units)
    n_fail = int(len(failed)) + int(sum(v for k, v in status.items() if k != "ok")) + int(len(curve_failed))
    lt = pd.DataFrame()
    if len(tt):
        lt = tt.groupby(["axis", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"), rows=("rows_per_fit", "mean"))
        lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
        print("[summarize] 변형·학습기별 적합 시간(전 조각 합계)\n" + lt.to_string(index=False), flush=True)
    smoke = smoke_report(units, runs, stores) if a.smoke else {}
    if smoke:
        smoke["sec_per_fit"] = {f"{r_.axis}|{r_.learner}": round(float(r_.sec_per_fit), 3) for r_ in lt.itertuples()} if len(lt) else {}
        print(f"[smoke] 같은 행렬 {smoke['same_matrix']}(짝 {smoke['n_pairs']}) · n = 0 의 R1 = R0 {smoke['r1_eq_r0_n0']}(키 {smoke['n_r1_r0']}) · "
              f"묶음 예측 차이 {smoke['chunk_maxdiff']} cm(허용 {SMOKE_CHUNK_TOL}) · CatBoost Pool 경로 차이 {smoke['cb_pool_maxdiff']} cm · "
              f"GPU 메모리 최댓값 할당 {smoke['gpu_mem_peak_mib']} MiB, 예약 {smoke['gpu_mem_reserved_peak_mib']} MiB · 장치 대응 확인 "
              f"{smoke['device_check']}", flush=True)
        print(f"[smoke] TabPFN 열 분류(변형|컨텍스트 크기 → {{분류: 적합 수}}): {json.dumps(smoke['tabpfn_schema'], ensure_ascii=False)}", flush=True)
        stab_, ratio_ = smoke_timing(runs, units)
        _atomic_csv(stab_, O / f"{a.TAG}_fit_timing.csv")
        smoke.update(sec_ratio_to_est=ratio_, fit_timing_rows=int(len(stab_)),
                     structure=dict(n_curve=int(len(cur)), n_minn=int(len(mn)), n_tests=int(len(tests)),
                                    n_tests_delta=int(np.isfinite(pd.to_numeric(tests["delta"], errors="coerce")).sum()) if "delta" in tests else 0,
                                    n_verdict=int(tests.scope.isin(["verdict", "verdict_aux"]).sum()) if "scope" in tests else 0,
                                    n_verdict_na=int((tests.scope.isin(["verdict", "verdict_aux"])
                                                      & tests.verdict.astype(str).str.startswith("판정 불가")).sum()) if "verdict" in tests else 0,
                                    test_ids=sorted({str(v) for v in tests.test_id}) if "test_id" in tests else [],
                                    n_unpaired_keys=int(sum(unp_lr.values()))))
        if len(stab_):
            print("[smoke] 적합 1건 시간(s, 변형·학습기·방법·n 별, 스모크 점검 시간 제외)\n" + stab_.to_string(index=False), flush=True)
        print(f"[smoke] 실측/추정식(계획서 §6C.10) 비: {ratio_} · 집계 경로: {smoke['structure']}", flush=True)
    fit_tot = Counter()
    for u in units:
        for k, v in (u.get("n_fit") or {}).items():
            fit_tot[k] += int(v)
    meta = dict(stage="H43/LGT", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6C", git_commit=git_commit(), tag=a.TAG,
                tags={ax: axis_tag(a, ax) for ax in AXES}, args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")},
                key_fields=list(H.KEY_COLS), p0_key=list(H.P0_KEY), all_n=-1, learners=list(LEARNERS_T), n_units=len(units),
                n_units_axis=dict(Counter(s_["axis"] for s_ in sh)), n_fit=dict(fit_tot), n_fit_total=int(sum(fit_tot.values())),
                unit_elapsed_s_sum=float(sum(float(u.get("elapsed_s", 0) or 0) for u in units)), elapsed_s=round(float(elapsed), 1),
                n_rows=int(len(runs)), n_curve=int(len(cur)), n_minn=int(len(mn)), n_tests=int(len(tests)), n_cross=int(len(cross)),
                n_gate=int(len(gate)), cross_note=cross_note, shard_cfg=cfg_info, unit_status=dict(status), n_failed_keys=int(len(failed)),
                n_flagged_rows=int(len(flagged)), flags=dict(Counter(flagged.fit_flag.astype(str))) if len(flagged) else {},
                curve_failed=curve_failed, n_fail=n_fail, skipped=skipped or [], floor=floor_meta, smoke=smoke,
                n_unpaired_keys=int(sum(unp_lr.values())), n_unpaired_learner=dict(unp_lr),
                unpaired_units={f"{k[0]}|s{k[1]}": len(v) for k, v in sorted(unpaired.items())},
                gpu_mem_peak_mib=max([float(u.get("gpu_mem_peak_mib") or 0.0) for u in units] or [0.0]),
                env=sorted({json.dumps(u.get("env", {}), ensure_ascii=False, sort_keys=True) for u in units}),
                code_sha=code_sha_t(), code_sha_h40=H.code_sha(), code_sha_h42=file_sha(SCRIPT_DIR / "h42_label_grid_ext.py", 12),
                delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux), nboot=int(a.nboot),
                rules=dict(ci="h4_common.boot_delta_blocks: 분할 안 채점 블록 재표집, 방법·추출·seed 공통 인덱스, 분할 분포 평균",
                           verdict4="우세 = 두 가중의 CI 상한 < 0, 열세 = 두 가중의 CI 하한 > 0, 동등 = 네 끝값의 절댓값이 한계 이하, 그 밖은 미결정",
                           hypotheses="L32–L37 은 모두 보조다. 확인적 가설 집합(계획서 §6A.2)을 바꾸지 않는다",
                           holm="item LGT 의 primary 행: L32 기준 λ 5개, L33 6개, L34 2개, L35 1개. 보조 열이며 판정에 쓰지 않는다",
                           base_lam="D0 1.0, R0·R1 0.25", cross="교차 환경(보조). 가설 판정에 쓰지 않는다",
                           failed="비유한 예측 또는 적합 실패 키. chunk 와 tsub 표지는 저장된 키다"),
                summarize_s=round(time.time() - t0, 1))
    H._atomic_text(O / f"{a.TAG}_meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    if n_fail:
        print(f"[summarize] 실패 기록: 조각 상태 {dict(status)} · 저장하지 못한 키 {len(failed):,} · 곡선 실패 {len(curve_failed)} → {a.TAG}_failed.csv",
              flush=True)
    print(f"[summarize] 조각 {len(sh)} · runs {len(runs):,} · curve {len(cur):,} · minn {len(mn):,} · tests {len(tests):,} · cross {len(cross):,} · "
          f"{time.time() - t0:.0f}s → {O}/{a.TAG}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v) and not a.smoke:                                            # 스모크는 판정 행과 Δ 를 출력하지 않는다(개정 12)
        print(v[[c_ for c_ in ("test_id", "role", "blind", "verdict", "stat") if c_ in v]].to_string(index=False), flush=True)
    return dict(curve=cur, minn=mn, tests=tests, cross=cross, gate=gate, targets=tg, timing=tt, failed=fa, meta=meta, n_fail=n_fail)


# ================================================================ 적합 수(--count-only)
def low_card_columns(D, thr=4):
    """고유값이 thr 개 미만인 x25 열. tabpfn 8.0.7 은 컨텍스트가 100행을 넘으면 이런 수치 열을 범주형으로 추론한다."""
    out = {}
    for c_ in H.FEATS:
        v = pd.Series(D.df[c_].values).dropna()
        k = int(v.nunique())
        if k < int(thr):
            out[str(c_)] = k
    return out


def count_only(a, D, units, skipped):
    """학습 없이 작업 단위, 적합 수, 추정 시간을 낸다. 추정식은 계획서 §6C.10 이다. <tag>_count.csv 에 쓴다."""
    t0 = time.time()
    rows, det, nrw, est, errs = [], Counter(), Counter(), Counter(), []
    for i, u in enumerate(units):
        r_ = run_unit_t(a, *u, dry=True)
        d_, w_, e_ = r_.pop("_detail"), r_.pop("_rows"), r_.pop("_est")
        errs += [f"{unit_name_t(u)}: {v}" for v in r_.pop("_errors")]
        for k, v in d_.items():
            det[k] += v; nrw[k] += w_.get(k, 0); est[k] += e_.get(k, 0.0)
        per = Counter()
        for k, v in d_.items():
            cs, lr, _ = (k.split("|") + ["", ""])[:3]
            per[f"fit_{cs}_{lr}"] += v; per[f"fit_{lr}"] += v
            per[f"est_{lr}_s"] += e_.get(k, 0.0)
        r_.update({k: (round(v, 1) if k.startswith("est_") else int(v)) for k, v in per.items()})
        r_.update(group=unit_group(u[0], u[1], u[2]), order=i)
        rows.append(r_)
    df = pd.DataFrame(rows).fillna(0)
    a.OUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(df, a.OUT / f"{a.TAG}_count.csv")
    tab = []
    for k in sorted(det):
        cs, lr, m_ = (k.split("|") + ["", ""])[:3]
        tab.append(dict(ctx_set=cs, learner=lr, method=m_, n_fit=int(det[k]), rows_per_fit=round(nrw[k] / max(det[k], 1)), est_h=est[k] / 3600.0))
    tab = pd.DataFrame(tab)
    print(f"[count-only] 작업 단위 {len(units)}(축별 {dict(Counter(u[0] for u in units))}) · 건너뜀 {len(skipped)}"
          f"({', '.join(sorted({str(s_['status']) for s_ in skipped})) or '없음'}) · {time.time() - t0:.0f}s", flush=True)
    summary = dict(units=len(units), units_axis=dict(Counter(u[0] for u in units)), fits={}, est_h={}, design={}, design_match=None)
    if len(tab):
        by = tab.groupby(["ctx_set", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), est_h=("est_h", "sum"))
        by["est_h"] = by.est_h.round(2)
        print("[count-only] 변형·학습기별 적합 수와 추정 누적 시간(h)\n" + by.to_string(index=False), flush=True)
        bm = tab.groupby(["ctx_set", "method", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), rows_per_fit=("rows_per_fit", "mean"))
        print("[count-only] 변형·방법별 적합 수\n" + bm.to_string(index=False), flush=True)
        tp = by[by.learner == TP].set_index("ctx_set").n_fit.to_dict()
        full = (not a.smoke) and (not a.targets) and a.N_USER == H._n_list(FULL_GRID) and int(a.splits) == 5 and not a.draws and int(a.seeds) == 2 \
            and set(a.SENS) == set(SENS_SETS) and set(a.AXES) == set(AXES)
        cmp_ = {k: dict(design=int(v), count=int(tp.get(k, 0)), same=bool(int(tp.get(k, 0)) == int(v))) for k, v in DESIGN_FITS.items()}
        summary.update(fits={f"{r_.ctx_set}|{r_.learner}": int(r_.n_fit) for r_ in by.itertuples()},
                       est_h={f"{r_.ctx_set}|{r_.learner}": float(r_.est_h) for r_ in by.itertuples()}, design=cmp_,
                       design_match=bool(all(v["same"] for v in cmp_.values())) if full else None)
        if full:
            print("[count-only] 설계 단계 값(계획서 §6C.5)과의 대조(TabPFN 적합 수): "
                  + " · ".join(f"{k} 설계 {v['design']:,} / 계산 {v['count']:,}{'' if v['same'] else ' (다르다)'}" for k, v in cmp_.items()), flush=True)
            if not summary["design_match"]:
                print("[count-only] 설계 단계 값과 다르다. 계획서 §6C.5 의 표를 이 값으로 고치고 개정 이력에 적는다", flush=True)
        else:
            print("[count-only] 등록 범위 전체가 아니므로 설계 단계 값과 대조하지 않는다", flush=True)
        h_tp = float(by[by.learner == TP].est_h.sum()); h_cb = float(by[by.learner == CBX].est_h.sum())
        tot = int(by[by.learner == TP].n_fit.sum())
        print(f"[count-only] TabPFN 적합 {tot:,}건 · 추정 누적 {h_tp:.1f} GPU-h · catboost_ctx {int(by[by.learner == CBX].n_fit.sum()):,}건 "
              f"추정 누적 {h_cb:.1f} h(4스레드 기준) · 합계 {h_tp + h_cb:.1f} h", flush=True)
        for w in (1, 2, 4):
            print(f"  GPU {w}장(프로세스 {w}개): 벽시계 추정 {(h_tp + h_cb) / w:.1f} h · 추정의 2배 {2 * (h_tp + h_cb) / w:.1f} h", flush=True)
        summary.update(n_fit_tabpfn=tot, est_h_tabpfn=round(h_tp, 2), est_h_catboost=round(h_cb, 2))
    if len(df):
        g = df.groupby(["axis", "group"], as_index=False).agg(units=("split", "size"), fit_tabpfn=(f"fit_{TP}", "sum"), est_tabpfn_s=(f"est_{TP}_s", "sum"),
                                                               n_ctx_max=("n_ctx_max", "max"), ctx_dev_max=("ctx_dev_max", "max"))
        g["est_tabpfn_h"] = (g.est_tabpfn_s / 3600.0).round(2)
        print("[count-only] 실행 순서 묶음별(0 주 설정·주 4지역, 1 민감도·주 4지역, 2 주 설정·Alaska x, 3 주 설정·나머지 학습기 축, 4 주 설정·나머지, "
              "5 민감도·나머지)\n" + g.drop(columns="est_tabpfn_s").to_string(index=False), flush=True)
        print(f"[count-only] 컨텍스트 행 수: 최댓값 {int(df.n_ctx_max.max()):,}(상한 {int(a.ctx_max):,}) · 최솟값 {int(df[df.n_ctx_min > 0].n_ctx_min.min()) if (df.n_ctx_min > 0).any() else 0:,} · "
              f"대상 행 부분 추출(tsub) {int(df.n_tsub.sum())}건 · 지역별 행 수와 비례 배분의 차이 최댓값 {float(df.ctx_dev_max.max()):.2f}행"
              f"(원천 지역 수 {int(df.n_src_regions.min())}–{int(df.n_src_regions.max())}, 이론 상한을 넘는 단위 "
              f"{int((df.ctx_dev_max > df.ctx_dev_bound + 1e-6).sum())}건)", flush=True)
        summary.update(n_ctx_max=int(df.n_ctx_max.max()), n_tsub=int(df.n_tsub.sum()), ctx_dev_max=float(df.ctx_dev_max.max()),
                       ctx_dev_bound_max=float(df.ctx_dev_bound.max()), n_over_bound=int((df.ctx_dev_max > df.ctx_dev_bound + 1e-6).sum()),
                       n_src_regions=[int(df.n_src_regions.min()), int(df.n_src_regions.max())], n_check_fail=int(df.n_check_fail.sum()))
        if a.dry_build:
            print(f"[count-only] --dry-build: 컨텍스트 행렬 점검 실패 {int(df.n_check_fail.sum())}건" + (f" · {errs[:5]}" if errs else ""), flush=True)
    lc = low_card_columns(D)
    summary["low_card_x25"] = lc
    print(f"[count-only] 고유값 4개 미만인 x25 열(TabPFN 이 범주형으로 추론하는 열): {lc if lc else '없음'}", flush=True)
    print(f"[count-only] 가중치 파일 {a.MODEL}: {'있음' if Path(a.MODEL).exists() else '없음'}"
          + (f" · {Path(a.MODEL).stat().st_size:,}바이트 · SHA-1 {file_sha(a.MODEL)}" if Path(a.MODEL).exists() else ""), flush=True)
    H._atomic_text(a.OUT / f"{a.TAG}_count_meta.json", json.dumps(dict(summary, skipped=skipped, errors=errs[:50], tag=a.TAG, code_sha=code_sha_t(),
                                                                      args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")}),
                                                                 ensure_ascii=False, indent=1, default=str))
    return df, summary


# ================================================================ 실행
def cuda_available(gpu):
    """GPU 하나(nvidia-smi 번호)에서 torch.cuda 를 쓸 수 있는지. 부모 프로세스가 CUDA 를 초기화해 워커의 GPU 에 문맥을 남기지 않도록
    짧은 자식 프로세스에서 확인한다(자식이 끝나면 문맥도 없어진다). 부모는 torch 를 부르지 않는다."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    code = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"
    try:
        return subprocess.run([sys.executable, "-c", code], env=env, timeout=300, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except Exception:                                                     # noqa: BLE001
        return False


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in THREAD_VARS:
        os.environ[v] = str(int(a.threads))                               # 워커(spawn)가 물려받는다
    will_run = not (a.count_only or a.summarize_only)
    if will_run:
        check_run_args(a)                                                 # 자료를 읽기 전에 거부한다
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                           # 학습 없는 실행은 GPU 를 쓰지 않는다
    if a.summarize_only:
        res = summarize(a, 0.0, None)
        return dict(executed=[], skipped=[], resumed=[], n_fail=int(res["n_fail"]) if res else 0)
    units, skipped = enumerate_t(a)
    D = H.get_data(h40_args_t(a, a.AXES[0] if a.AXES else "main"))
    units.sort(key=lambda u: priority_t(a, D, u))
    print(f"[data] {len(D.df):,}셀 · 축 {a.AXES} · 민감도 {a.SENS} · 분할 1–{int(a.splits)} · n {a.N_USER} · seed {int(a.seeds)} · λ {a.LAMS} · "
          f"하위 지역 {D.sub_src} · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 스레드 {a.threads}", flush=True)
    if a.count_only:
        count_only(a, D, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[], n_fail=0)
    check_overwrite(a, units)                                             # 완료 조각이 있는데 --resume 도 --overwrite 도 없으면 거부한다
    try:
        info, apps = query_gpu_info(), query_gpu_apps()
    except Exception as e:                                                # noqa: BLE001
        raise SystemExit(f"[거부] nvidia-smi 로 GPU 상태를 확인할 수 없다: {repr(e)[:200]}")
    gpus, dropped = screen_gpus(a.GPUS, {g: v["used"] for g, v in info.items()}, a.gpu_mem_max_mib, apps=gpu_apps_by_index(info, apps))
    for g_, why in dropped:
        print(f"[warn] GPU {g_} 를 뺀다: {why}", flush=True)
    if not gpus:
        raise SystemExit("[거부] 쓸 수 있는 GPU 가 없다(지정한 GPU 가 모두 사용 중이다)")
    if a.smoke:
        gpus = gpus[:1]                                                   # 스모크는 GPU 1장
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])                     # 부모 쪽 코드가 다른 GPU 를 건드리지 않게 한다(워커는 자기 값으로 바꾼다)
    if not cuda_available(gpus[0]):                                       # 자식 프로세스에서 확인한다(부모는 CUDA 문맥을 만들지 않는다)
        raise SystemExit("[거부] torch.cuda 를 쓸 수 없다. TabPFN 을 CPU 로 돌리지 않는다")
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    a.RUN_ID = f"{os.getpid()}-{time.strftime('%Y%m%dT%H%M%S')}"
    lock = acquire_lock(a)                                                # 같은 tag 의 두 번째 실행을 거부한다
    clear_run_notes(a.RUN_DIR, ("running", "gpu_busy", "worker"))
    done, failed, quarantined, resumed, todo = [], [], [], [], []
    done_names, failed_names = set(), set()
    interrupted, abort, handlers, busy_gpus, gpu_all = "", "", {}, set(), list(gpus)

    def log(u):
        done.append(u); done_names.add(unit_name_t((u["axis"], u["target"], u["mode"], int(u["split"]))))
        print(f"  [{u['axis']}|{u['target']}|{u['mode']}|s{u['split']}] 적합 {u['n_fit_total']} · 행 {u['n_rows']} · 컨텍스트 최대 {u['n_ctx_max']} · "
              f"A {u['n_A']} · 채점 {u['n_eval']}/{u['nb_eval']}블록{'' if u['valid'] else '(무효 분할)'} · 원천 {u['n_src']} · {u['elapsed_s']}s · "
              f"GPU '{u['device']}' 메모리 최댓값 {u['gpu_mem_peak_mib']} MiB(예약 {u['gpu_mem_reserved_peak_mib']} MiB) · 상태 {u['status']}(실패 {u['n_fail']}"
              f"{', failed 학습기 ' + ','.join(u['failed_learners']) if u.get('failed_learners') else ''}) · 완료 {len(done)}/{len(todo)} · "
              f"누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append((u, repr(e)[:300])); failed_names.add(unit_name_t(u))
        print(f"  [FAIL] {unit_name_t(u)}: {repr(e)[:300]}", flush=True)

    try:
        for u in units:
            ok, why = unit_state_t(a, *u) if a.resume else (False, "")
            if ok:
                resumed.append(u)
                print(f"  [resume] 건너뜀 {unit_name_t(u)} (조각 있음, 상태 {why})", flush=True)
            else:
                todo.append(u)
                if a.resume and why != "조각 없음":
                    print(f"  [resume] 다시 실행 {unit_name_t(u)} ({why})", flush=True)
        print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · GPU {gpus}(프로세스 {len(gpus)}개) · 스레드 {a.threads} · 가중치 {a.MODEL.name} "
              f"SHA-1 {file_sha(a.MODEL, 12)} · tabpfn {pkg_version('tabpfn')} · n_estimators {a.n_est} · 컨텍스트 상한 {a.ctx_max:,} · "
              f"run_id {a.RUN_ID} · 기록 {a.RUN_DIR}", flush=True)
        if todo:
            handlers = install_stop_handlers()                            # SIGTERM, SIGHUP → KeyboardInterrupt(워커를 정리하고 끝낸다)
            ctx = multiprocessing.get_context("spawn")                    # fork 후 OpenMP·CUDA 충돌 회피
            remaining, stall, rebuilds, crashes = list(todo), 0, 0, Counter()
            while remaining:
                if not gpus:
                    for u in remaining:
                        fail(u, RuntimeError("쓸 수 있는 GPU 가 없다(풀 재생성 때 재확인)"))
                    break
                q = ctx.Queue()
                for g_ in gpus:
                    q.put(g_)
                broken, n_done0, clean, procs = [], len(done), False, {}
                ex = ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx, initializer=_worker_init_t,
                                         initargs=(argv, q, a.threads, str(a.RUN_DIR), os.getpid(), a.RUN_ID))
                try:
                    futs = {ex.submit(_worker_run_t, *u): u for u in remaining}
                    procs = dict(getattr(ex, "_processes", None) or {})
                    for f in as_completed(futs):
                        u = futs[f]
                        try:
                            log(f.result())
                        except BrokenProcessPool:                         # 워커 비정상 종료: 원인 단위를 가린 뒤 새 풀에서 다시 돈다
                            broken.append(u)
                        except Exception as e:                            # noqa: BLE001  한 단위의 실패가 나머지를 막지 않게 한다
                            fail(u, e)
                            if TAG_MISMATCH in str(e):                    # 장치 대응이 어긋났다: 실행 전체를 멈춘다
                                abort = str(e)
                                break
                    clean = not abort
                finally:
                    if clean:
                        ex.shutdown(wait=True)
                    else:
                        kill_pool(ex)
                if abort:
                    for u in remaining:
                        if unit_name_t(u) not in done_names | failed_names:
                            fail(u, RuntimeError(f"GPU 장치 대응 불일치로 실행을 멈췄다: {abort[:160]}"))
                    break
                if not broken:
                    break
                rebuilds += 1
                codes = {int(pid): p.exitcode for pid, p in procs.items()}
                culprit = culprit_pids(codes)
                notes = read_run_notes(a.RUN_DIR, "running")
                running = {str(m_.get("unit", "")) for m_ in notes if not culprit or int(m_.get("pid", -1)) in culprit}
                busy = {int(m_["gpu"]) for m_ in read_run_notes(a.RUN_DIR, "gpu_busy") if "gpu" in m_}
                clear_run_notes(a.RUN_DIR, ("running", "gpu_busy"))
                fin, quar, again = triage_broken(broken, running, crashes, lambda u: unit_done_run(a, u))
                for u, uj in fin:                                         # 결과 전달 전에 풀이 깨졌으나 조각은 이번 실행에서 완료되었다
                    log(unit_summary(uj))
                for u in quar:
                    quarantined.append(unit_name_t(u))
                    fail(u, RuntimeError(f"실행 중에 풀이 {crashes[unit_name_t(u)]}회 깨졌다(격리, 상한 {POOL_CRASH_MAX})"))
                busy_gpus |= busy
                stall = 0 if (len(done) > n_done0 or quar) else stall + 1
                print(f"[pool] 워커 비정상 종료(재생성 {rebuilds}회째, 진전 없는 연속 {stall}회). 종료 코드 {codes} · 원인으로 본 실행 중 단위 "
                      f"{sorted(running) or '없음'} · 점유 표지 GPU {sorted(busy) or '없음'} · 완료 확인 {len(fin)} · 격리 {len(quar)} · 남은 {len(again)}",
                      flush=True)
                if stall > int(a.pool_retries):
                    for u in again:
                        fail(u, RuntimeError(f"진전 없이 풀이 {stall}회 연속 깨졌다(상한 {a.pool_retries})"))
                    break
                gpus = rescreen_gpus(a, gpu_all, busy_gpus)
                if a.smoke:
                    gpus = gpus[:1]
                remaining = sorted(again, key=lambda u: priority_t(a, D, u))
    except KeyboardInterrupt as e:
        interrupted = str(e) or "KeyboardInterrupt"
        print(f"[중단] {interrupted}. 워커를 종료했다. 이어 돌리려면 같은 명령에 --resume 을 준다", flush=True)
    finally:
        restore_handlers(handlers)
        n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
        part = [unit_name_t((u["axis"], u["target"], u["mode"], int(u["split"]))) for u in done if u["status"] == "partial"]
        stat = dict(run_id=a.RUN_ID, tag=a.TAG, argv=argv, start_s=round(t0, 1), elapsed_s=round(time.time() - t0, 1), gpus_first=gpu_all,
                    gpus_last=gpus, busy_gpus=sorted(busy_gpus), n_todo=len(todo), n_done=len(done), n_resumed=len(resumed), n_fail=n_fail,
                    n_partial=len(part), partial_units=part, failed_units=[dict(unit=unit_name_t(u), reason=r_) for u, r_ in failed],
                    failed_status_units=[unit_name_t((u["axis"], u["target"], u["mode"], int(u["split"]))) for u in done if u["status"] == "failed"],
                    quarantined=quarantined, interrupted=interrupted, abort=abort,
                    next_step="python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --gpus <확인한 목록> --threads 4 --resume --rerun-partial --no-summarize")
        _write_json(a.OUT / f"{a.TAG}_run_status.json", stat)
        release_lock(lock)
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {len(part)} · 격리 {len(quarantined)} · {time.time() - t0:.0f}s · "
          f"상태 기록 {a.OUT / (a.TAG + '_run_status.json')}", flush=True)
    if part:
        print(f"[done] 일부 적합이 실패한 조각(partial) {len(part)}개: {part[:10]}{' …' if len(part) > 10 else ''}. "
              "두 번째 통과는 --resume --rerun-partial 로 돌린다", flush=True)
    res = dict(executed=[(u["axis"], u["target"], u["mode"], u["split"]) for u in done], skipped=skipped, resumed=list(resumed), n_fail=n_fail,
               n_partial=len(part), interrupted=interrupted)
    if interrupted:
        return res
    if not a.no_summarize:
        sres = summarize(a, time.time() - t0, skipped)
        if sres:
            res["n_fail"] += int(sres["n_fail"])
    return res


if __name__ == "__main__":
    _res = main()
    sys.exit(exit_code(_res))
