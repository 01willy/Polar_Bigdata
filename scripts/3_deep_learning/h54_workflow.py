"""H54 · 희소·충분 라벨 워크플로 하네스(WF). 계획 docs/EXPERIMENT_PLAN_WF_2026-10-01.md(커밋 4168331, 결과 열람 뒤 설계)의 구현.
2차 보강 실험 WF6–WF8 은 같은 계획서 §6(커밋 1b42b4d, 결과 열람 뒤 설계)의 구현이다.

실험(--exps 또는 --exp, 기본 wf1,wf2,wf3,wf4,wf6,wf7,wf8. WF0 은 재분석이라 계산이 없고, WF5 지도는 wf5_priority 함수와 --wf5-map 으로
따로 만든다. 1차 = wf1–wf4(계획서 §2), 2차 보강 = wf6–wf8(계획서 §6))
  wf1  충분 라벨 지역 안의 라벨 수 곡선. 지역 내 모드(모드 이름 r): 학습 라벨 = 대상 A 블록에서 뽑은 n 개뿐이고 다른 지역 원천 행이 없다.
       P0 과 κ 수축의 사전값만 LG 의 원천 계수 E0(모드 x 원천, 100 km 버퍼)을 쓴다.
       대상 Alaska, Lena, Canada, AL-1, AL-2, AL-6. Alaska 는 x34(x25 + SAR 9열) 조각을 따로 둔다(--x34-targets).
       n {20, 50, 100, 200, 500, 1000, 2000, 5000, 전량}(|A| 미만과 전량), 추출 5(n ≤ 1000)·3(그 밖)·1(전량), seed 2.
       방법 P0, P1, P2, Pk, Pc@c2·c4·c8, Pc(교차검증 k), Pbest(P1·P2·Pk·Pc 가운데 교차검증), D0, D1, R1(λ 0.25·0.5·1.0·교차검증),
       R2(같은 λ), Re(λ 0.25·교차검증), RK(λ 0.25·0.5·1.0·교차검증). D0·R1 은 catboost_lo 와 catboost(기본 용량) 둘 다,
       그 밖의 학습 방법은 catboost_lo. x34 조각은 catboost_lo 의 D0, D1, R1, R2(λ 격자와 교차검증 λ)만 둔다.
  wf2  라벨 위치 선택. 대상 Alaska, Lena, Canada, AL-1, AL-2(지역 내, 후보 = A 셀). 전략 S1–S7, 예산 {20, 50, 100, 200, 500}(|A| 미만),
       추출 5, seed 2. 방법 P1, R1(λ 0.25·0.5·1.0, 판정 λ 0.25), D1. 조각은 (대상, 분할, 전략) 단위이고 키의 placement 가 전략이다.
  wf3  하위 군집. 대상 Alaska, Lena(지역 내). n {100, 500, 2000, 전량}, 추출 규칙은 wf1 과 같다. 방법 P1, Pc@c2·c4·c8, Pc(교차검증 k),
       R1(전체 잔차), R1c@c{k}(군집별 잔차), R1i@c{k}(군집 원핫 입력을 더한 R1). λ 0.25·0.5·1.0.
  wf4  방법 선택 규칙 W 와 편향 진단. 대상 30 = LG 의 27 (대상, 모드)와 LGD 약관 확인분 새 지역 3(Tibet_LGD, NAtlantic~lic, Russia_C~lgd).
       전이 모드(원천 = LG 와 같은 다른 지역 셀, h40.build_ctx). n {10, 40, 160, 전량}, 추출 5, seed 2, 추출과 유사라벨 색인은 LG 와 같다.
       방법 P0, P1, P2, R1(λ 0.25·0.5·1.0), R2(같은 λ), D1, W. W 는 선택한 n 개 라벨 안 5겹 블록 교차검증(seed 0 적합)으로
       {P0, P1, P2, R1(0.25), R1(1.0), R2(0.25), D1} 가운데 오차가 가장 작은 방법이고, 라벨이 10개 미만이면 P1 이다.
       n = 10 의 추출마다 편향 진단값(선택 라벨 위치의 P0 평균 오차, E0·s − y 의 평균)을 unit.json 의 diag 에 남긴다.
  wf6  검정력 보강(계획서 §6 WF6). 지역 내 모드 r(wf1 과 같다). 대상 Alaska(x25), Lena, Canada. n {200, 500, 1000, 전량}(|A| 미만과 전량),
       추출 3(전량 1), seed 2. 분할 = half_split_blocks 의 split_seed 1–25(LG 분할 1–5 에 새 분할 6–25. --wf6-splits). 중복·무효 분할은
       h40.Data.split_structure 의 규칙을 분할 1–25 에 그대로 적용해 뺀다(분할 1–5 의 구조는 LG 와 같다).
       방법 P1, Pk, R1(λ 0.25·0.5·1.0·교차검증), R2(같은 λ), Re(λ 0.25·교차검증), D0(catboost 기본 용량만). R1·R2·Re 는 catboost_lo 다.
       WF6-a 의 보조 대비(Pbest)를 위해 wf1 과 같은 물리 보정식 묶음(P2, Pc@c2·c4·c8, Pc, Pbest)도 계산해 저장한다(학습 적합 없음).
  wf7  Stefan·CCI 평균 앵커의 전이 성능(계획서 §6 WF7). 전이 모드(h40.build_ctx). 대상 = LG 의 27 (대상, 모드)(러시아 C·그린란드는 점 추정만),
       분할 1–5, n {0, 3, 10, 40, 160, 320, 1000, 전량}(0·전량과 |A| 미만), 추출 5(0·전량 1), seed 2, 추출은 h40.draw_cells(LG 와 같다).
       방법 P0, P1, Pe(앵커만), R1(λ 0.25), Re(앵커 + 잔차 λ 0.25, catboost_lo). P0·P1·R1 은 같은 플랫폼 대비를 위해 이 조각에서 다시 적합한다.
       앵커(E) = 0.5·(E·s + cci_alt)(cci_alt 유한이고 cci_valid ≥ 0.5 인 셀), 그 밖 E·s. Pe = 앵커(E_n). Re 의 잔차 모형 학습 행은
       h40 의 R1 과 같은 구성이다: 원천 행의 목표 = y − 앵커(E0)(원천 계수), 선택 라벨 행의 목표 = y − 앵커(E_n).
  wf8  전이 조건의 라벨 위치 선택(계획서 §6 WF8). 대상 Lena, Canada, Russia_W, Russia_E, Alaska(모드 x, 원천 = LG). n {10, 40}(|A| 미만.
       러시아 W·E 는 |A| 14–17 이라 10 만), 분할 1–5, 추출 5, seed 2. 전략 S1(셀 무작위 = h40.draw_cells), S2(블록 층화 = h40.draw_blocks),
       S4(대상 A 풀 x25 표준화의 탐욕 k-중심)는 wf2 의 구현(RUnit.strategy_sets·s4_order·std_stats)을 그대로 빌려 쓴다.
       방법 P1, R1(λ 0.25, 원천 행 + 선택 라벨). 조각은 (대상, 분할, 전략) 단위이고 키의 placement 가 전략이다.

기호와 공통 규약
  s = e5_sqrt_tdd. E_ls = Σ s·y / Σ s². E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10(h42.shrink). x25 = h40.FEATS, x34 = x25 + SAR 9열.
  분할 = m1_core.half_split_blocks(분할 1–5), 채점 = B 블록의 eval_mask 셀(h40 과 같다). 중복 분할과 채점 블록 2개 미만 분할의 규칙도 h40 과 같다.
  지역 내 추출 seed = seed_of(대상, 'r', 분할, n, 추출)(h40.draw_cells 에 모드 'r' 를 넣는다). wf2 의 S1 은 같은 추출이다.
  키 = (method, learner, alpha, placement, n, draw, seed, lam)(h40 의 BlockStore 키). alpha 는 항상 '1'. 교차검증 λ 의 키는 lam = −1.0 이고
  고른 λ 는 runs 의 alpha_sel 에 'lam=0.5' 형식으로 적는다. 선택형 방법(Pc, Pbest, W)의 고른 값도 alpha_sel 에 적는다.
  교차검증은 h42.cv_folds_of(선택 라벨의 A 블록을 순열해 5묶음, 블록이 하나면 셀 묶음)를 쓰고 seed 0 적합으로 고른 값을 모든 seed 에 쓴다.
  묶음마다 E_n 을 학습 쪽 라벨로 다시 구한다. 동률이면 λ 는 작은 값, 방법은 후보 순서의 앞, k 는 작은 값이다.
  통계: h40.contrast(분할 안 채점 블록 재표집, 셀 가중·블록 등가중), h42 의 verdict4(우세, 열세, 동등 ±0.5 cm, 미결정), region_stats,
  pool_rows, TestBook(층화 평균, Holm 보조 열). 재표집 10,000회(--nboot).

구현 결정(계획서가 정하지 않은 세부. 계획서 개정 이력에 '구현 결정, 결과 실행 전'으로 적었다)
  1. 크리깅(Pk, RK): 자체 구현 국소 정규 크리깅이다. 좌표는 구면 위 3차원 직교 좌표(km), 최근접 라벨 32개, 지수 변이도
     γ(h) = c0 + c·(1 − exp(−3h/a))(pykrige 'exponential' 과 같은 모수화, a = 실용 범위). 변이도는 라벨(최대 1,500개 무작위 부분표본)의
     경험 변이도(12구간, 최대 거리의 절반까지)에 구간 쌍 수의 제곱근 가중 최소제곱으로 맞춘다. a 는 60개 로그 격자에서 고르고
     (c, c0) 는 비음수 최소제곱이다. 라벨이 10개 미만이거나 구간이 3개 미만이면 크리깅하지 않고 P1(E_n)으로 대체하고 fit_flag 에 적는다.
     pykrige 1.7.3 이 로컬에 있으나 Rescale 이미지에는 없어 계산 경로가 플랫폼마다 달라질 수 있으므로 쓰지 않는다.
     시험 (k)가 같은 변이도 모수에서 pykrige 의 국소 크리깅(backend loop, n_closest_points)과 예측이 같은지 확인한다.
     Pk 는 라벨 위치의 E = y/s 를 크리깅해 Ê·s 로 예측한다. RK 는 R1(λ) 의 표본 안 잔차 y − (E_n·s + λ·g) 를 크리깅해 더한다
     (교차 적합 잔차가 아니다). RK 의 교차검증 λ 는 R1 의 교차검증 λ 를 그대로 쓴다.
  2. 군집(Pc, R1c, R1i, S3): A 풀 셀의 x25 를 중앙값 대체 뒤 A 풀의 평균·표준편차로 표준화하고(h40._prep_stats, 라벨 미사용)
     k-평균(n_init 10, random_state = seed_of('wf-km', 대상, 분할, k))을 A 풀에 맞춘다. B 셀은 가장 가까운 중심에 배정한다.
     Pc 는 군집 c 의 라벨로 구한 E_ls,c 를 전체 E_n 으로 κ = 10 수축한다(E_c = (n_c·E_ls,c + κ·E_n)/(n_c + κ), 라벨 없는 군집은 E_n).
     R1c 는 군집 라벨이 20개 이상인 군집만 따로 잔차 모형을 맞추고 그 밖의 군집은 전체 R1 의 g 를 쓴다. 앵커는 전체 E_n·s 다.
     R1i 는 군집 번호를 원핫 k 열로 입력에 더한다. S3 는 k = 8 군집에 예산을 군집 크기 비례(최대 나머지 방식)로 나누고 군집 안에서 무작위로 뽑는다.
  3. S6(순차 능동 선택): 20개 무작위(S1 의 n = 20 추출과 같다)에서 시작해 다음 예산 경계를 넘지 않게 최대 20개씩 더한다(20, 40, 50, 70,
     90, 100, 120, …, 500). 선택 모형은 R1 잔차 모형(목표 y − E_n·s, E_n 은 현재 라벨로 다시 구한다)이고 catboost_lo 설정에
     posterior_sampling = True 를 준 CatBoost 다. virtual_ensembles_predict(prediction_type = 'VirtEnsembles', virtual_ensembles_count = 10)
     의 성분 사이 분산이 큰 A 풀 셀을 고른다(동률은 seed 로 섞는다). seed = seed_of('wf-s6', 대상, 분할, 추출, 현재 라벨 수).
     선택 적합은 학습기 이름 catboost_ve, 방법 이름 S6sel 로 센다.
     S4 는 표준화 x25 에서 탐욕 k-중심(첫 점은 추출 seed 로 무작위)이고, S5 는 LG 원천(모드 x) 셀의 표준화 x25 까지의 최근접 거리를 원천 안
     평균 쌍 거리(부분표본 2,000)로 나눈 값(AOA 비유사도, 변수 가중 없음)이 큰 순서, S7 은 √TDD 가 큰 순서(동률은 추출 seed 로 섞는다)다.
     S5 는 결정적이라 추출이 모두 같다. 같은 라벨 집합이 앞 추출과 겹치면 그 추출은 적합하지 않는다(S5 는 추출 0 만 저장된다).
  4. x34 의 SAR 결측(PolSAR 은 알래스카 셀의 74 % 만 유한)은 CatBoost 의 기본 결측 처리(Min)에 맡긴다. 대체하지 않는다.
  5. LGD 새 지역은 data/processed/lgd/run_tables/{Tibet, NAtlantic_lic, Russia_C}/ 의 실행 표(약관 미확인 행 0)를 h40.Data 로 읽어
     LGD 와 같은 방식으로 돌린다(대상 이름 Tibet_LGD, NAtlantic, Russia_C, 저장소 이름 Tibet_LGD|x, NAtlantic~lic|x, Russia_C~lgd|x).
     NAtlantic~lic 은 LGD 와 같이 점 추정만이다. 표가 없으면 그 대상을 건너뛰고 meta 의 skipped 에 적는다(--no-lgd 로 끌 수 있다).
  6. 지역 내 D1·R2 의 유사라벨 행 수는 round(r × 학습 라벨 수), r = 10 이다(LG 는 r × 원천 행 수. 지역 내 모드에는 원천 행이 없다).
     색인은 A 풀에서 복원 추출(seed_of('wf-ps', 대상, 'r', 분할, n, 추출, seed, 묶음))이고 라벨 셀은 실측으로 바꾼다(h40 과 같은 규칙).
  7. Re 의 앵커 = 0.5·(E_n·s + cci_alt)(cci_alt 유한이고 cci_valid ≥ 0.5 인 셀), 그 밖은 E_n·s 다. 채점 셀은 eval_mask 상 모두 CCI 가 있다.
  8. 판정 정의. WF1-b 는 교차검증 λ 의 R1@x34 − R1(catboost_lo)이다(고정 λ 행은 보조). WF1-a·c 의 R1, R2 는 교차검증 λ 다.
     WF2-a 는 (전략, n) 18개 대비 가운데 하나 이상이 우세이면 지지다. 세 n 모두 우세인 전략은 보조로 적는다.
     WF2-b 의 '가장 좋은 전략' 은 알래스카 n = 100 의 R1(λ 0.25) 점 추정 RMSE 가 가장 작은 S2–S7 이고, 레나·캐나다의 n = 100 대비로 판정한다.
     WF3-a 는 두 대상 × n {500, 2000, 전량} 이 모두 우세이면 지지, 일부면 부분 지지, 우세가 없거나 열세가 있으면 기각이다.
     WF4-a 의 비열등은 주 4지역 층화 평균 W − R1(0.25) 의 두 가중 CI 상한이 모두 0.5 cm 미만인 것이다. WF4-b 는 n 160, 전량 각각에서
     W − P1 이 우세인 대상 수가 R1(0.25) − P1 이 우세인 대상 수보다 많은지 본다(두 n 모두면 지지). WF4-c 의 진단값은 편향의 크기
     |mean(E0·s − y)|(추출·분할 평균), ML 이득은 전량 라벨의 RMSE(P0) − RMSE(W)이고, CI 는 이득의 블록 재표집 분포(h40.contrast)와
     진단값의 (분할, 추출) 재표집을 같은 번호로 묶은 Spearman 분포의 2.5–97.5 백분위다.
  9. 대상 30 은 LG 와 같이 하위 지역의 모드 i·x 를 모두 둔다(WF0 의 30대상과 같은 구성). --wf4-modes x 로 모드 x 만 쓰면 20대상이다.

구현 결정(2차 보강 WF6–WF8. 계획서 개정 이력에 '구현 결정, 결과 실행 전'으로 적었다)
  10. wf6 분할: split_seed 1–25 의 구조를 h40.Data.split_structure 의 코드로 계산한다(분할 목록만 1–25 로 바꾼 얇은 보기, _SplitView).
      중복(A 블록 집합이 앞 분할과 같음)은 빼고, 채점 블록 2개 미만 분할은 유효 분할이 있으면 뺀다(h40 과 같다). 빈 자리를 다른 seed 로
      채우지 않는다. 계획서의 '분할 1–5 에 새 분할 6–25 를 더한 20회' 는 seed 목록(1–25, 25개)과 횟수(20)가 맞지 않는다. 명시한 seed 목록을
      따라 1–25 를 쓰고 '20' 은 새 분할의 수(6–25)로 읽었다(--wf6-splits 20 이면 1–20). 자료 v3 에서 알래스카·캐나다 25회, 레나 24회
      (분할 24 의 채점 블록 1개)이고 분할 1–25 사이에 중복 분할은 없다.
  11. wf6 방법: 교차검증 λ 는 wf1 과 같은 함수(lam_cv: h42.cv_folds_of 블록 5묶음, seed 0 적합, 묶음마다 E_n 재계산)다. 고정 λ 키(R1·R2 의
      0.25·0.5·1.0, Re 의 0.25)는 사후 적용이라 함께 저장한다(보조). D0 은 catboost(600, 깊이 6, 0.03)만 적합한다.
  12. wf7 Re 의 원천 행 목표는 원천 계수 E0 로 만든 앵커의 잔차 y − 앵커(E0)다(h40 의 R1 이 원천 행에 E0 앵커를 쓰는 것과 같은 구성이고,
      대회 구성 S12 도 원천 계수 앵커를 썼다). n = 0 에서 P1 = P0, R1·Re 는 원천 행만으로 적합한다. R1·Re 는 λ 0.25 만 저장한다.
  13. wf8 전략은 RUnit 의 함수를 그대로 쓴다. S4 의 seed 는 wf2 와 같은 seed_of('wf-s4', 대상, 분할, 추출)라 같은 A 풀에서 wf2 의 S4 와 같은
      순서를 낸다. S1·S2 는 모드 x 의 LG 추출(셀, 블록 분산)과 같다. 같은 라벨 집합이 여러 추출에서 나오면 적합은 한 번만 하고 예측을
      추출마다 저장한다(wf2 는 첫 추출만 저장했다. wf8 은 S1 과 추출 번호를 맞추고 중복 집합의 가중을 보존한다).
  14. 판정 정의. WF6-a(대상별): (레시피 ∈ {R1, R2, Re(교차검증 λ), D0(catboost)}, n) 대비 가운데 하나라도 P1 보다 우세(두 가중)면 지지.
      Pbest 대비는 보조 판정 행(verdict_aux)이다. WF6-b: 세 대상 층화 평균의 R2(교차검증 λ) − P1 을 n ∈ {500, 1000, 전량}(전량은
      n ≥ 500 에 넣는다. wf1 과 같은 규약)에서 보고 모두 우세면 지지, 열세가 없고 일부 우세면 부분 지지, 열세가 있거나 우세가 없으면 기각이다.
      WF7-a·b 는 방향 중립 서술이다(주 4지역 층화 평균의 4분 판정, 확인적 가설로 세지 않는다. tests 의 hypothesis 열 False).
      WF8-a: 레나·캐나다의 n = 40 R1 에서 S4 − S1 이 우세인 지역이 하나 이상이면 지지. WF8-b: 다섯 대상의 n = 10 R1 에서 S2 − S1,
      S4 − S1 가운데 열세가 하나라도 있으면 기각, 판정할 수 없는 대비가 있으면 판정 불가, 그 밖은 지지다.
  15. 2차 보강의 집계 표는 1차 표를 덮어쓰지 않게 <tag>2b_* 에 따로 쓴다(아래 산출). 1차(wf1–wf4)의 조각 이름, 설정 해시, 표 이름은 바뀌지 않는다
      (시험 (p)가 1차 설정 해시와 조각 이름을 고정값과 대조한다).

누설 규약
  대상 라벨은 선택된 n 개만 계수, 학습, 크리깅, 교차검증, 전략(S6)에 쓴다. 표준화와 군집은 A 풀의 공변량(라벨 미사용)으로만 정한다.
  B 블록 셀의 라벨은 채점 전용이다. 시험이 B 라벨과 비선택 A 라벨을 바꿔도 예측과 선택이 같음을 확인한다.

산출(<out-dir>, 기본 data/processed/wf)
  shards/<tag><실험번호>__cpu__<대상>__<모드>__s<분할>[__<변형>]_{runs.csv, blocksse.npz, unit.json}. 변형 = x34(wf1), 전략(wf2).
    unit.json 이 완료 표지다(마지막에 원자적으로 쓴다). --resume 은 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  <tag>_curve.csv(h40.build_curve 형식 + exp, verdict4_p0·p1), <tag>_tests.csv(WF1-a·b·c, WF2-a·b, WF3-a·b, WF4-a·b·c 와 서술 대비),
  <tag>_meta.json, <tag>_targets.csv, <tag>_timing.csv, <tag>_failed.csv, <tag>_count.csv, <tag>_count_detail.csv(--count-only).
  2차 보강(wf6–wf8): 조각은 shards/<tag>{6,7,8}__cpu__… (wf8 은 변형 = 전략), 표는 <tag>2b_curve.csv, <tag>2b_tests.csv(WF6-a·b, WF7-a·b 서술,
  WF8-a·b 와 서술 대비), <tag>2b_meta.json, <tag>2b_targets.csv, <tag>2b_timing.csv, <tag>2b_failed.csv, <tag>2b_count.csv, <tag>2b_count_detail.csv.
  3차 보강(--exp wf9,wf9x,wf10, 계획서 §7): <tag>3b_curve.csv, <tag>3b_tests.csv(WF9-a·c, WF10-a·b·c 와 서술 WF9-b·d, WF10-d), <tag>3b_meta.json,
  <tag>3b_targets.csv, <tag>3b_timing.csv, <tag>3b_failed.csv, <tag>3b_rmse.csv(저장소별 RMSE), <tag>3b_decomp.csv(격자 안·사이 분해), <tag>3b_count.csv.
  두 차수를 한 번에 집계하면 차수마다 자기 표에 쓴다. 스모크는 wf_smoke_*, wf2b_smoke_* 다.

실행 환경
  로컬 공유 서버에서는 --count-only, --summarize-only(재표집 1,000회 상한), --smoke(스레드 2, 워커 0)만 한다. 본 실행과 사전 점검은
  Rescale CPU 노드에서 한다(WF_RESCALE=1 또는 LG_RESCALE=1 또는 --allow-local 이 없으면 거부). 스레드 상한은 허용 표지가 없으면 4다.
  CatBoost 는 thread_count 밖의 고정 비용 부분도 여러 스레드로 돌려 프로세스 CPU 사용이 thread_count 의 약 1.5–3배다(로컬 측정, 작은 적합일수록
  크다. 명시적 Pool 의 thread_count 로도 줄지 않았다). 로컬 스모크와 시험은 taskset 으로 코어 4개에 묶어 공유 서버의 상한을 지킨다.
  적합 수:  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/h54_workflow.py --count-only --threads 1 [--exp wf6,wf7,wf8]
  스모크:   CUDA_VISIBLE_DEVICES= nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/h54_workflow.py --smoke --threads 2 --workers 0
            [--exp wf6,wf7,wf8]
  본 실행:  WF_RESCALE=1 python3 scripts/3_deep_learning/h54_workflow.py --workers 22 --threads 4 --resume --no-summarize
  집계:     WF_RESCALE=1 python3 scripts/3_deep_learning/h54_workflow.py --summarize-only --workers 22 --threads 4
  Rescale:  scripts/rescale/run_wf.sh(WF_MODE=smoke|full, WF_EXPS), configs/rescale/wf_smoke.yaml, wf_full.yaml(1차, WF_EXPS=wf1,wf2,wf3,wf4),
            wf2_smoke.yaml, wf2_full.yaml(2차 보강, WF_EXPS=wf6,wf7,wf8), 묶음 scripts/rescale/make_payload_wf.sh
  WF5 지도: python3 scripts/3_deep_learning/h54_workflow.py --wf5-map --wf5-strategy S4 --wf5-n 50,200 --threads 2 (WF2 판정 뒤)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import warnings
from collections import Counter
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

LOCAL_MAX_THREADS = 4                             # 허용 표지 없는 실행의 스레드 상한(공유 서버)


def run_permitted_env(argv=None):
    """본 실행 허용 표지: 환경 변수 WF_RESCALE=1 또는 LG_RESCALE=1, 또는 --allow-local."""
    av = sys.argv if argv is None else argv
    return os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1" or "--allow-local" in av


def _peek_threads():
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽는다). 허용 표지가 없으면 상한 LOCAL_MAX_THREADS."""
    val = None
    av = sys.argv
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        t = int(val) if val is not None else 1
    except ValueError:
        t = 1
    if not run_permitted_env():
        t = min(t, LOCAL_MAX_THREADS)
    return str(max(t, 1))


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _peek_threads())
warnings.filterwarnings("ignore", module="threadpoolctl")
warnings.filterwarnings("ignore", message=".*glibc.*")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, fname):
    """동결 하네스(h40, h42)를 파일 경로에서 읽어 sys.modules 에 등록한다(h42 와 같은 방식). 이미 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("h40_label_grid", "h40_label_grid.py")
X = _load("h42_label_grid_ext", "h42_label_grid_ext.py")

from polar.m1_core import INPUT_SETS, eval_mask, half_split_blocks                                   # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                             # noqa: E402
from polar import m1_stats as MS                                                                     # noqa: E402

# ================================================================ 고정 설계값(계획서 §1–§2. 바꾸면 사전 등록에서 벗어난다)
FEATS = list(H.FEATS)                                                       # x25
SAR_COLS = [v for v in INPUT_SETS["x34"] if v not in FEATS]                 # insar 5, polsar 3, insar_miss
FEATS34 = FEATS + SAR_COLS                                                  # x34 = x25 다음에 SAR 9열
CCI_COL, CCIV_COL = FEATS.index("cci_alt"), FEATS.index("cci_valid")
MODE_R = "r"                                                                # 지역 내 모드
KAPPA = 10.0
R_PS = 10.0
LAMS = (0.25, 0.5, 1.0)
LAM_BASE = 0.25
LAM_CV = -1.0                                                               # 교차검증으로 고른 λ 의 키
CV_K = 5
DRAWS_SMALL, DRAWS_LARGE, DRAW_SPLIT_N = 5, 3, 1000                         # n ≤ 1000 은 추출 5, 그 밖은 3
SEEDS = (0, 1)
LO, HI = "catboost_lo", "catboost"
KRIGE_K = 32
KRIGE_MIN = 10
VARIO_MAX_PTS = 1500
VARIO_NLAGS = 12
VARIO_NRANGE = 60
CLUSTER_KS = (2, 4, 8)
S3_K = 8
R1C_MIN = 20
S6_START, S6_BATCH, VE_COUNT = 20, 20, 10
AOA_SUB = 2000
W_CANDS = ("P0", "P1", "P2", "R1@0.25", "R1@1.0", "R2@0.25", "D1")       # 규칙 W 의 후보(계획서 순서 = 동률 순서)
W_MIN_N = 10
DIAG_N = 10
NI_MARGIN = 0.5
PBEST_CANDS = ("P1", "P2", "Pk", "Pc")
STRATEGIES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
EXPS_R1 = ("wf1", "wf2", "wf3", "wf4")                                      # 1차(계획서 §2)
EXPS_R2 = ("wf6", "wf7", "wf8")                                             # 2차 보강(계획서 §6)
EXPS_R3 = ("wf9", "wf9x", "wf10")                                           # 3차 보강(계획서 §7). 기본 실행 목록에는 넣지 않는다(--exp 로 지정)
EXPS_ALL = EXPS_R1 + EXPS_R2                                                # --exps 의 기본값(1·2차)
EXPS_KNOWN = EXPS_ALL + EXPS_R3
TRANSFER_EXPS = ("wf4", "wf7", "wf8", "wf9x")                               # 전이 모드(원천 = LG, h40.build_ctx)
PLAN_COMMIT = {"r1": "4168331", "r2": "1b42b4d", "r3": "8180632"}           # 차수별 사전 등록 커밋
WF1_TARGETS = ("Alaska", "Lena", "Canada", "AL-1", "AL-2", "AL-6")
WF1_X34 = ("Alaska",)
WF1_GRID = (20, 50, 100, 200, 500, 1000, 2000, 5000, -1)
WF2_TARGETS = ("Alaska", "Lena", "Canada", "AL-1", "AL-2")
WF2_BUDGETS = (20, 50, 100, 200, 500)
WF2_DRAWS = 5
WF3_TARGETS = ("Alaska", "Lena")
WF3_GRID = (100, 500, 2000, -1)
WF4_GRID = (10, 40, 160, -1)
WF4_DRAWS = 5
# 2차 보강(계획서 §6). 바꾸면 사전 등록에서 벗어난다
WF6_TARGETS = ("Alaska", "Lena", "Canada")                                  # 지역 내(모드 r), x25
WF6_GRID = (200, 500, 1000, -1)
WF6_SPLITS = 25                                                             # split_seed 1–25 = LG 분할 1–5 + 새 분할 6–25
WF6_DRAWS = 3
WF7_GRID = (0, 3, 10, 40, 160, 320, 1000, -1)                               # LG 의 n 격자
WF7_DRAWS = 5
WF7_TEST_N = (0, 10, 40, 160, -1)                                           # WF7-a·b 를 보고하는 n
WF8_TARGETS = ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska")          # 모드 x
WF8_GRID = (10, 40)
WF8_DRAWS = 5
WF8_STRATEGIES = ("S1", "S2", "S4")
# 3차 보강(계획서 §7). 바꾸면 사전 등록에서 벗어난다
WF9_TARGETS = ("Alaska", "Lena", "Canada")                                  # 지역 내(모드 r), x25, WF6 와 같은 대상
WF9_GRID = (200, 500, 1000, -1)
WF9_SPLITS = 25                                                             # split_seed 1–25(WF6 와 같다)
WF9_DRAWS = 3
WF9X_TARGETS = ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska")         # 전이(모드 x)
WF9X_GRID = (0, 10, 40, 160, -1)
WF9X_DRAWS = 5
WF10_TARGETS = ("Alaska", "Canada")                                         # 지역 내, 블록 평균 √TDD 범위가 넓은 대상
WF10_GRID = (100, 500, -1)
WF10_SPLITS = 10                                                            # I 블록 순열 seed 1–10
WF10_DRAWS = 3
WF10_FRAC = 0.25                                                            # W·I 의 대상 셀 비율 하한
WF10_VARIANTS = ("warm", "cold")
ZERO_N_EXPS = ("wf7", "wf9x")                                               # n = 0(라벨 없음)을 격자에 두는 실험
# 1차 Rescale 스모크(nyFtT) 사전 점검 대표 단위 7개의 실측/추정 비(계획서 개정 이력 2026-10-01 13:20). --count-only 의 환산 출력에만 쓴다
RESCALE_RATIO = 0.62
# LGD 약관 확인분 새 지역: 저장소 이름 → (실행 표 디렉터리, 표 안의 대상 이름, 점 추정만)
LGD_SPECS = {"Tibet_LGD": ("Tibet", "Tibet_LGD", False), "NAtlantic~lic": ("NAtlantic_lic", "NAtlantic", True),
             "Russia_C~lgd": ("Russia_C", "Russia_C", False)}
MAIN4 = list(H.MAIN4)
# 적합 시간 추정(--count-only 전용, 결과에는 쓰지 않는다). LG 의 BASE_FIT 모형(h40): 1건 = b·(행 수/기준 행 수)^지수.
# catboost 는 h42.BASE_FIT_X 의 가정값(재지 않았다). 소표본 하한(고정 비용)은 로컬 스모크(2스레드, 2026-10-01)의 실측
# (catboost_lo 0.10 s, catboost 0.5 s, 가상 앙상블 0.09 s)을 4스레드로 줄여 잡은 가정값이다. Rescale 스모크의 실측으로 바꿔 읽는다.
EST_BASE = {LO: H.BASE_FIT[LO], HI: X.BASE_FIT_X[HI], "catboost_ve": H.BASE_FIT[LO]}
EST_FLOOR = {LO: 0.08, HI: 0.35, "catboost_ve": 0.09}
EST_KRIGE = (0.02, 1.0e-5)                                                  # 크리깅 1회 = a + b·(예측 셀 수) s(스모크 실측 0.02 s 근처)
EST_KMEANS = 0.05                                                           # k-평균 1회(스모크 실측 0.03–0.04 s)
PRED_TRACE = None                                                           # 시험용: dict 를 넣으면 저장한 모든 키의 예측을 기록한다
WF_TRACE = None                                                             # 시험용: 리스트를 넣으면 적합·계수·선택이 쓴 라벨 색인을 기록한다


# ================================================================ 인자
def _list(txt, cast=str):
    return [cast(v.strip()) for v in str(txt).split(",") if v.strip()]


def _n_list(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def _grid_txt(g):
    return ",".join("all" if int(v) == -1 else str(int(v)) for v in g)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H54 희소·충분 라벨 워크플로 하네스(WF)")
    ap.add_argument("--exps", "--exp", dest="exps", default=",".join(EXPS_ALL),
                    help="실행·집계할 실험(쉼표 목록. 1차 wf1, wf2, wf3, wf4, 2차 보강 wf6, wf7, wf8, 3차 보강 wf9, wf9x, wf10. 기본 = 1·2차)")
    ap.add_argument("--wf1-targets", default=",".join(WF1_TARGETS))
    ap.add_argument("--x34-targets", default=",".join(WF1_X34), help="wf1 에서 x34 조각을 둘 대상(SAR 열이 있는 알래스카 계열)")
    ap.add_argument("--wf2-targets", default=",".join(WF2_TARGETS))
    ap.add_argument("--wf3-targets", default=",".join(WF3_TARGETS))
    ap.add_argument("--wf4-targets", default="", help="'이름:모드' 쉼표 목록. 기본 = LG 27 (대상, 모드) + LGD 새 지역 3")
    ap.add_argument("--wf4-modes", default="i,x", help="기본 대상 목록에 쓰는 모드(하위 지역). x 만 주면 20대상")
    ap.add_argument("--no-lgd", action="store_true", help="wf4 에서 LGD 새 지역을 뺀다")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--wf1-grid", default=_grid_txt(WF1_GRID))
    ap.add_argument("--wf2-budgets", default=_grid_txt(WF2_BUDGETS))
    ap.add_argument("--wf3-grid", default=_grid_txt(WF3_GRID))
    ap.add_argument("--wf4-grid", default=_grid_txt(WF4_GRID))
    ap.add_argument("--wf6-targets", default=",".join(WF6_TARGETS))
    ap.add_argument("--wf6-grid", default=_grid_txt(WF6_GRID))
    ap.add_argument("--wf6-splits", type=int, default=WF6_SPLITS, help="wf6 의 half_split_blocks split_seed 1..K(LG 분할 1–5 + 새 분할)")
    ap.add_argument("--wf7-targets", default="", help="'이름:모드' 쉼표 목록. 기본 = LG 27 (대상, 모드)")
    ap.add_argument("--wf7-grid", default=_grid_txt(WF7_GRID))
    ap.add_argument("--wf8-targets", default=",".join(WF8_TARGETS), help="'이름' 또는 '이름:모드' 쉼표 목록(모드 생략 = x)")
    ap.add_argument("--wf8-grid", default=_grid_txt(WF8_GRID))
    ap.add_argument("--wf8-strategies", default=",".join(WF8_STRATEGIES))
    ap.add_argument("--wf9-targets", default=",".join(WF9_TARGETS))
    ap.add_argument("--wf9-grid", default=_grid_txt(WF9_GRID))
    ap.add_argument("--wf9-splits", type=int, default=WF9_SPLITS, help="wf9 의 half_split_blocks split_seed 1..K(WF6 와 같은 25)")
    ap.add_argument("--wf9x-targets", default=",".join(WF9X_TARGETS), help="'이름' 또는 '이름:모드' 쉼표 목록(모드 생략 = x)")
    ap.add_argument("--wf9x-grid", default=_grid_txt(WF9X_GRID))
    ap.add_argument("--wf10-targets", default=",".join(WF10_TARGETS))
    ap.add_argument("--wf10-grid", default=_grid_txt(WF10_GRID))
    ap.add_argument("--wf10-splits", type=int, default=WF10_SPLITS, help="wf10 의 I 블록 순열 seed 1..K")
    ap.add_argument("--wf10-variants", default=",".join(WF10_VARIANTS))
    ap.add_argument("--splits", type=int, default=5, help="half_split_blocks split_seed 1..K")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--draws-cap", type=int, default=0, help="추출 수 상한(0 = 설계값). 스모크·시험용")
    ap.add_argument("--workers", type=int, default=1, help="프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=1, help="프로세스당 스레드(BLAS·CatBoost). 허용 표지가 없으면 상한 4")
    ap.add_argument("--cb-iters", type=int, default=200, help="catboost_lo 반복 수(LG 와 같은 200)")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5, help="4분 판정의 동등 한계(cm)")
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--ni-margin", type=float, default=NI_MARGIN, help="WF4-a 비열등 한계(cm)")
    ap.add_argument("--out-dir", default="data/processed/wf")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="wf")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(분할 1, 추출 1, seed 1, 대상 축소). 로컬 허용(스레드 2 권장)")
    ap.add_argument("--precheck", action="store_true", help="Rescale 사전 점검: 실험마다 본 실행 크기의 대표 단위 하나씩(시간 측정)")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-local", action="store_true", help="본 실행을 로컬에서 허용한다(WF_RESCALE=1 과 같은 효과)")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--no-pin-cores", action="store_true", help="워커를 코어 묶음에 고정하지 않는다(기본: 허용 표지가 있고 코어가 충분하면 워커마다 "
                                                              "스레드 수만큼의 코어에 고정)")
    ap.add_argument("--wf5-map", action="store_true", help="WF5: 레나 격자에 전략을 적용해 다음 관측 우선순위를 쓴다(학습 적합 포함)")
    ap.add_argument("--wf5-strategy", default="", help="WF5 전략(WF2-b 규칙으로 정한 전략. S1–S7, S2 는 격자 좌표의 0.5° 블록)")
    ap.add_argument("--wf5-n", default="50,200")
    ap.add_argument("--wf5-grid", default="data/processed/map_lena/lena_grid_x25_v1.csv.gz")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(p, base=ROOT):
    return Path(p) if os.path.isabs(str(p)) else Path(base) / str(p)


def default_wf4_targets(modes):
    out = [f"{t}:{m}" for t, m in H.default_targets(["i", "x"]) if m in modes or t in H.MAIN4 + H.MAIN_POINT + [H.ALASKA]]
    return out


def _pair(v, mode="x"):
    """'이름:모드' → (이름, 모드). 모드를 생략하면 mode."""
    return tuple(v.split(":", 1)) if ":" in v else (v, mode)


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    a.SUFFIX = ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.TAG = a.tag + a.SUFFIX
    a.PERMIT = run_permitted_env(a.ARGV)
    a.threads_asked = int(a.threads)
    a.threads = max(1, int(a.threads) if a.PERMIT else min(int(a.threads), LOCAL_MAX_THREADS))
    a.EXPS = [v for v in _list(a.exps) if v in EXPS_KNOWN]
    a.SPLITS = list(range(1, int(a.splits) + 1))
    a.SEEDS = list(range(int(a.seeds)))
    a.T = {"wf1": _list(a.wf1_targets), "wf2": _list(a.wf2_targets), "wf3": _list(a.wf3_targets)}
    a.X34 = _list(a.x34_targets)
    modes = _list(a.wf4_modes)
    t4 = _list(a.wf4_targets) if a.wf4_targets else default_wf4_targets(modes) + ([] if a.no_lgd else [f"{k}:x" for k in LGD_SPECS])
    a.T["wf4"] = [tuple(v.split(":")) if ":" in v else (v, "x") for v in t4]
    a.STRAT = [v for v in _list(a.strategies) if v in STRATEGIES]
    a.G = {"wf1": _n_list(a.wf1_grid), "wf2": _n_list(a.wf2_budgets), "wf3": _n_list(a.wf3_grid), "wf4": _n_list(a.wf4_grid)}
    # 2차 보강(wf6–wf8)
    a.SPLITS6 = list(range(1, int(a.wf6_splits) + 1))
    a.T["wf6"] = _list(a.wf6_targets)
    a.T["wf7"] = [_pair(v) for v in (_list(a.wf7_targets) if a.wf7_targets else [f"{t}:{m}" for t, m in H.default_targets(["i", "x"])])]
    a.T["wf8"] = [_pair(v) for v in _list(a.wf8_targets)]
    a.STRAT8 = [v for v in _list(a.wf8_strategies) if v in WF8_STRATEGIES]
    a.G.update(wf6=_n_list(a.wf6_grid), wf7=_n_list(a.wf7_grid), wf8=_n_list(a.wf8_grid))
    # 3차 보강(wf9, wf9x, wf10)
    a.SPLITS9 = list(range(1, int(a.wf9_splits) + 1))
    a.SPLITS10 = list(range(1, int(a.wf10_splits) + 1))
    a.VAR10 = [v for v in _list(a.wf10_variants) if v in WF10_VARIANTS]
    t3 = dict(wf9=_list(a.wf9_targets), wf9x=[_pair(v) for v in _list(a.wf9x_targets)], wf10=_list(a.wf10_targets))
    a.T.update(t3)
    a.G.update(wf9=_n_list(a.wf9_grid), wf9x=_n_list(a.wf9x_grid), wf10=_n_list(a.wf10_grid))
    if a.smoke:                                                    # 스모크: 모든 경로를 작은 대상으로 한 번씩 지난다
        a.SPLITS = [1]; a.SEEDS = [0]; a.draws_cap = a.draws_cap or 1
        a.T = {"wf1": ["Canada", "AL-3"], "wf2": ["Canada"], "wf3": ["Canada"],
               "wf4": [("Russia_W", "x")] + ([] if a.no_lgd else [("Tibet_LGD", "x")])}
        a.X34 = ["AL-3"]
        a.G = {"wf1": [20, 50, -1], "wf2": [20, 50], "wf3": [100, -1], "wf4": [10, -1]}
        a.nboot = min(int(a.nboot), 500)
        # 2차 보강: wf6 은 새 분할(6)을 지나고, wf7 은 n = 0 과 하위 지역 모드 i, wf8 은 |A| 가 작은 대상(n = 10 만)을 지난다
        a.SPLITS6 = [6]
        a.T.update(wf6=["Canada"], wf7=[("Russia_W", "x"), ("AL-3", "i")], wf8=[("Russia_W", "x"), ("Canada", "x")])
        a.G.update(wf6=[200, -1], wf7=[0, 10, -1], wf8=[10, 40])
        # 3차 보강: wf9 는 새 분할(7), wf9x 는 n = 0 과 |A| 가 작은 대상, wf10 은 warm 변형 하나
        a.SPLITS9 = [7]; a.SPLITS10 = [1]; a.VAR10 = ["warm"]
        a.T.update(wf9=["Canada"], wf9x=[("Russia_W", "x"), ("Canada", "x")], wf10=["Canada"])
        a.G.update(wf9=[200, -1], wf9x=[0, 10, -1], wf10=[100, -1])
    if a.precheck:                                                 # 사전 점검: 실험마다 본 실행 크기의 대표 단위(분할 1)
        a.SPLITS = [1]
        a.T = {"wf1": ["Alaska"], "wf2": ["Alaska"], "wf3": ["Alaska"], "wf4": [("Lena", "x"), ("AL-2", "i")]}
        a.STRAT = ["S1", "S6"]
        a.SPLITS6 = [1]
        a.T.update(wf6=["Alaska"], wf7=[("Alaska", "x"), ("Lena", "x")], wf8=[("Alaska", "x")])
        a.STRAT8 = ["S1", "S4"]
        a.SPLITS9 = [1]; a.SPLITS10 = [1]; a.VAR10 = ["warm"]
        a.T.update(wf9=["Alaska"], wf9x=[("Alaska", "x"), ("Lena", "x")], wf10=["Alaska"])
    a.TAG2 = f"{a.tag}2b{a.SUFFIX}"                                # 2차 보강의 집계 표 이름(1차 표를 덮어쓰지 않는다)
    a.TAG3 = f"{a.tag}3b{a.SUFFIX}"                                # 3차 보강의 집계 표 이름
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGD = _abs(a.lgd_dir)
    a.SHARDS = a.OUT / "shards"
    a._ha = None
    return a


def exp_tag(a, exp):
    return f"{a.tag}{exp[2:]}{a.SUFFIX}"


def rounds_of(a, exps=None):
    """실험 목록을 차수로 나눈다. 반환 [(차수, 실험 목록, 집계 표 이름)]. 1차 = wf1–wf4(<tag>_*), 2차 보강 = wf6–wf8(<tag>2b_*),
    3차 보강 = wf9, wf9x, wf10(<tag>3b_*)."""
    exps = list(a.EXPS if exps is None else exps)
    out = []
    for rnd, members, tag in (("r1", EXPS_R1, a.TAG), ("r2", EXPS_R2, a.TAG2), ("r3", EXPS_R3, a.TAG3)):
        ex = [e for e in exps if e in members]
        if ex:
            out.append((rnd, ex, tag))
    return out


def draws_for(a, exp, n):
    """(n 의 추출 수). wf1·wf3 은 n ≤ 1000 에서 5, 그 밖 3, 전량 1. wf2·wf4 는 5(전량 1). wf6 은 3, wf7·wf8 은 5(n = 0·전량 1).
    --draws-cap 이 있으면 그 값 이하."""
    if n == -1 or (n == 0 and exp in ZERO_N_EXPS):
        k = 1
    elif exp in ("wf1", "wf3"):
        k = DRAWS_SMALL if n <= DRAW_SPLIT_N else DRAWS_LARGE
    elif exp in EXPS_R2:
        k = {"wf6": WF6_DRAWS, "wf7": WF7_DRAWS, "wf8": WF8_DRAWS}[exp]
    elif exp in EXPS_R3:
        k = {"wf9": WF9_DRAWS, "wf9x": WF9X_DRAWS, "wf10": WF10_DRAWS}[exp]
    else:
        k = WF2_DRAWS if exp == "wf2" else WF4_DRAWS
    return max(1, min(k, int(a.draws_cap))) if int(a.draws_cap) > 0 else k


def cells_for(a, exp, nA, grid=None):
    """(n, 추출) 목록. 양수 n 은 |A| 미만만, 전량은 n = −1 하나(h40.cells_of 와 같은 규칙). n = 0 은 ZERO_N_EXPS(wf7)에서만 하나 둔다."""
    out = []
    for n in (a.G[exp] if grid is None else grid):
        if n == -1:
            out.append((-1, 0))
        elif n == 0 and exp in ZERO_N_EXPS:
            out.append((0, 0))
        elif 0 < n < nA:
            out += [(n, d) for d in range(draws_for(a, exp, n))]
    return out


def h40_args(a):
    """적합기(h40.Fitter, h42.FitterX)에 넘기는 h40 형식 인자. 스레드와 catboost_lo 반복 수만 쓴다."""
    if a._ha is None:
        a._ha = H.parse_args(["--threads", str(a.threads), "--cb-iters", str(a.cb_iters), "--kappa", str(KAPPA), "--r", str(R_PS),
                              "--lams", ",".join(str(v) for v in LAMS)])
    return a._ha


# ================================================================ 자료와 작업 단위 문맥
_DATA: dict = {}


def data_args(a, data_dir):
    """h40.Data 를 만드는 h40 형식 인자(자료 디렉터리별). 분할 범위는 a.SPLITS 다."""
    HA = H.parse_args(["--threads", str(a.threads), "--data-dir", str(data_dir), "--subregion-map", a.subregion_map,
                       "--splits", str(max(a.SPLITS))])
    HA.SPLITS = list(a.SPLITS)
    return HA


def get_data(a, spec=None):
    """(h40 인자, h40.Data). spec = None 은 v3 자료(data/processed), 그 밖은 LGD 실행 표 디렉터리 이름이다. 프로세스마다 한 번 적재한다."""
    key = (str(a.PROC), str(a.LGD), spec, tuple(a.SPLITS))
    if key not in _DATA:
        d = a.PROC if spec is None else a.LGD / spec
        if spec is not None and not (d / "fidelity_base_v3.csv").exists():
            raise FileNotFoundError(f"LGD 실행 표가 없다: {d}/fidelity_base_v3.csv")
        HA = data_args(a, d)
        _DATA[key] = (HA, H.Data(HA))
    return _DATA[key]


def lgd_available(a, alias):
    spec = LGD_SPECS[alias][0]
    return (a.LGD / spec / "fidelity_base_v3.csv").exists() and (a.LGD / spec / "e5_soil_tdd_v3.csv").exists()


def resolve4(a, alias):
    """wf4 대상 이름 → (자료 spec, 표 안의 대상 이름, 점 추정만)."""
    if alias in LGD_SPECS:
        spec, tgt, po = LGD_SPECS[alias]
        return spec, tgt, po
    return None, alias, alias in H.MAIN_POINT


class RCtx:
    """지역 내 작업 단위(대상, 분할). 라벨 후보 = A 블록의 모든 셀, 채점 = B 블록의 eval_mask 셀. 원천 행은 없다.
    E0 = LG 원천(모드 x) 최소제곱 계수(P0 과 κ 수축의 사전값). src_X = LG 원천 셀의 x25(S5 의 AOA 거리 전용, 라벨 미사용).
    시험에서는 합성 자료로 직접 만든다."""

    def __init__(self, target, split, parent, XA, yA, sA, blkA, latA, lonA, XB, yB, sB, blkB, latB, lonB, E0, XA34=None, XB34=None,
                 src_X=None, meta=None, mode=MODE_R):
        self.target, self.mode, self.split, self.parent = str(target), str(mode), int(split), str(parent)
        self.XA = np.asarray(XA, np.float32); self.yA = np.asarray(yA, float); self.sA = np.asarray(sA, float); self.blkA = np.asarray(blkA)
        self.XB = np.asarray(XB, np.float32); self.yB = np.asarray(yB, float); self.sB = np.asarray(sB, float); self.blkB = np.asarray(blkB)
        self.latA, self.lonA = np.asarray(latA, float), np.asarray(lonA, float)
        self.latB, self.lonB = np.asarray(latB, float), np.asarray(lonB, float)
        self.XA34 = None if XA34 is None else np.asarray(XA34, np.float32)
        self.XB34 = None if XB34 is None else np.asarray(XB34, np.float32)
        self.E0 = float(E0)
        self.src_X = None if src_X is None else np.asarray(src_X, np.float32)
        self.meta = dict(meta or {})
        self.PA, self.PB = xyz_km(self.latA, self.lonA), xyz_km(self.latB, self.lonB)

    def XAf(self, xs):
        return self.XA if xs == "x25" else self.XA34

    def XBf(self, xs):
        return self.XB if xs == "x25" else self.XB34


class _SplitView:
    """h40.Data.split_structure 를 다른 분할 목록으로 부르기 위한 얇은 보기. 같은 df, 같은 대상 색인, 같은 코드(중복·채점 블록 규칙)를 쓰고
    h40.Data 의 분할 캐시(분할 1–5)는 건드리지 않는다. wf6 의 분할 1–25 에 쓴다."""

    def __init__(self, D, splits):
        self.df, self.target_idx = D.df, D.target_idx
        self.args = SimpleNamespace(SPLITS=[int(v) for v in splits])
        self._split: dict = {}


_SPLIT_EXT: dict = {}


def split_structure_ext(D, target, splits):
    """분할 목록 splits 의 구조(h40.Data.split_structure 와 같은 dict). 앞선 분할과 A 블록 집합이 같으면 dup_of, 채점 블록 2개 미만은 무효."""
    key = (id(D), str(target), tuple(int(v) for v in splits))
    if key not in _SPLIT_EXT:
        _SPLIT_EXT[key] = H.Data.split_structure(_SplitView(D, splits), target)
    return _SPLIT_EXT[key]


def splits_of(a, exp):
    """실험의 분할 목록. wf6 은 split_seed 1–25(--wf6-splits), wf9 는 1–25(--wf9-splits), wf10 은 I 순열 seed 1–10(--wf10-splits),
    그 밖은 1–5(--splits)."""
    if exp == "wf6":
        return list(a.SPLITS6)
    if exp == "wf9":
        return list(a.SPLITS9)
    if exp == "wf10":
        return list(a.SPLITS10)
    return list(a.SPLITS)


def split_info_of(a, D, exp, target):
    """실험의 분할 구조. wf6·wf9 는 분할 1–25 안에서 다시 계산하고, 그 밖은 h40.Data 의 분할 1–5 구조(기존과 같다). wf10 은 wf10_plan 을 쓴다."""
    return split_structure_ext(D, target, splits_of(a, exp)) if exp in ("wf6", "wf9") else D.split_structure(target)


def split_plan(D, target, splits, info=None):
    """실행할 분할과 건너뛴 분할(h40.enumerate_units 와 같은 규칙: 중복 분할, 채점·A 셀 없음, 유효 분할이 있을 때의 무효 분할을 뺀다).
    info 를 주지 않으면 h40.Data 의 분할 구조를 쓴다(wf1–wf4, wf7, wf8)."""
    info = D.split_structure(target) if info is None else info
    any_valid = any(v["valid"] for v in info.values())
    keep, skip = [], []
    for sp in splits:
        v = info[sp]
        if v["dup_of"] >= 0:
            skip.append((sp, f"dup_of_{v['dup_of']}", v))
        elif v["n_eval"] == 0 or v["n_A"] == 0:
            skip.append((sp, "no_eval", v))
        elif not v["valid"] and any_valid:
            skip.append((sp, "invalid_nb_eval<2", v))
        else:
            keep.append(sp)
    return keep, skip, info


def build_rctx(a, target, split, info=None):
    """지역 내 문맥. A·B 색인과 채점 셀은 h40.build_ctx 와 같은 규칙(half_split_blocks, eval_mask)으로 만든다.
    info = 분할 구조 dict(wf6 의 분할 1–25). 주지 않으면 h40.Data 의 분할 1–5 구조를 쓴다."""
    HA, D = get_data(a)
    df = D.df
    t_idx = D.target_idx(target)
    A_idx, B_idx = half_split_blocks(df, t_idx, split)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    _, parent, src_idx, comp = D.source_idx(target, "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    E0 = H.ls_E(df.y.values[src], df.s.values[src])
    info = D.split_structure(target)[split] if info is None else info
    has34 = all(c_ in df.columns for c_ in SAR_COLS)

    def cols(idx, names):
        return df[names].values[idx].astype(np.float32)
    meta = dict(n_A=int(len(A_idx)), nb_A=int(len(np.unique(df.block.values[A_idx]))),
                n_eval=int(len(evB)), nb_eval=int(len(np.unique(df.block.values[evB]))), n_src=int(len(src)), E0=float(E0),
                E_A=float(H.ls_E(df.y.values[A_idx], df.s.values[A_idx])), n_cells=int(len(t_idx)), dup_of=int(info["dup_of"]),
                valid=bool(info["valid"]), n_valid_splits=int(info["n_valid_splits"]), n_unique_splits=int(info["n_unique_splits"]),
                src_mode="x", n_src_parent=int(comp.get("n_src_parent", 0)), subregion_src=getattr(D, "sub_src", ""))
    return RCtx(target, split, parent, cols(A_idx, FEATS), df.y.values[A_idx], df.s.values[A_idx], df.block.values[A_idx], df.lat.values[A_idx],
                df.lon.values[A_idx], cols(evB, FEATS), df.y.values[evB], df.s.values[evB], df.block.values[evB], df.lat.values[evB],
                df.lon.values[evB], E0, XA34=cols(A_idx, FEATS34) if has34 else None, XB34=cols(evB, FEATS34) if has34 else None,
                src_X=cols(src, FEATS), meta=meta)


def build_tctx(a, alias, mode, split):
    """전이 문맥(h40.build_ctx 그대로). LGD 새 지역은 그 실행 표의 h40.Data 로 만든다. 반환 (Ctx, 저장소 이름의 대상)."""
    spec, tgt, _ = resolve4(a, alias)
    HA, D = get_data(a, spec)
    c = H.build_ctx(D, HA, tgt, mode, split)
    c.meta.update(alias=alias, data_spec=spec or "v3")
    return c


def wf10_split(D, target, variant, split):
    """WF10 분할(계획서 §7): 대상 셀의 0.5° 블록을 블록 평균 √TDD(토양 도일, 유한 셀의 셀 수 가중, 공변량만 사용)로 정렬해, warm 은 가장
    따뜻한 블록부터, cold 는 가장 추운 블록부터 대상 셀 수의 25 % 이상이 될 때까지 W 로 둔다(동률은 블록 이름 순). 남은 블록을
    seed_of('wf10-I', 대상, 변형, 분할) 순열로 섞어 셀 수 25 % 이상이 될 때까지 I 로 둔다. 나머지가 A. 반환 dict(A, W, I = df 색인, 블록 목록, 블록 평균)."""
    if variant not in WF10_VARIANTS:
        raise ValueError(f"wf10 변형은 {WF10_VARIANTS} 가운데 하나다: {variant}")
    df = D.df
    t_idx = np.asarray(D.target_idx(target))
    blk = df.block.values[t_idx].astype(str)
    sv = df.s.values[t_idx].astype(float)
    ub, inv = np.unique(blk, return_inverse=True)
    nbk = len(ub)
    ncell = np.bincount(inv, minlength=nbk).astype(float)
    ok = np.isfinite(sv)
    ssum = np.bincount(inv[ok], weights=sv[ok], minlength=nbk)
    scnt = np.bincount(inv[ok], minlength=nbk).astype(float)
    bmean = np.full(nbk, np.nan)
    bmean[scnt > 0] = ssum[scnt > 0] / scnt[scnt > 0]
    quota = WF10_FRAC * float(len(t_idx))
    sign = -1.0 if variant == "warm" else 1.0
    order = sorted([j for j in range(nbk) if np.isfinite(bmean[j])], key=lambda j: (sign * bmean[j], ub[j]))
    W_b, cum = [], 0.0
    for j in order:
        if cum >= quota:
            break
        W_b.append(j); cum += ncell[j]
    wset = set(W_b)
    rest = [j for j in range(nbk) if j not in wset]
    perm = np.random.RandomState(seed_of("wf10-I", target, variant, int(split))).permutation(len(rest))
    I_b, cum = [], 0.0
    for q in perm:
        if cum >= quota:
            break
        I_b.append(rest[q]); cum += ncell[rest[q]]
    iset = set(I_b)
    A_b = [j for j in rest if j not in iset]
    pick = lambda js: t_idx[np.isin(inv, np.asarray(sorted(js), int))] if js else np.zeros(0, int)   # noqa: E731
    return dict(A=pick(A_b), W=pick(W_b), I=pick(I_b), W_blocks=sorted(ub[W_b].tolist()), I_blocks=sorted(ub[I_b].tolist()),
                A_blocks=sorted(ub[A_b].tolist()), block_mean=dict(zip(ub.tolist(), [float(v) for v in bmean])))


_WF10_PLAN: dict = {}


def wf10_plan(a, D, target, variant):
    """WF10 분할 1..K 의 (실행, 건너뜀, 구조). 앞선 분할과 I 블록 집합이 같으면 중복(dup_of), W·I 의 채점 블록이 2개 미만이거나 A 가 비면 무효."""
    key = (id(D), str(target), str(variant), tuple(a.SPLITS10))
    if key not in _WF10_PLAN:
        df = D.df
        info, seen = {}, {}
        for sp in a.SPLITS10:
            q = wf10_split(D, target, variant, sp)
            evW = q["W"][eval_mask(df.iloc[q["W"]])] if len(q["W"]) else q["W"]
            evI = q["I"][eval_mask(df.iloc[q["I"]])] if len(q["I"]) else q["I"]
            nbW, nbI = len(np.unique(df.block.values[evW])), len(np.unique(df.block.values[evI]))
            ks = frozenset(q["I_blocks"])
            dup = seen.get(ks, -1)
            seen.setdefault(ks, sp)
            info[sp] = dict(n_A=int(len(q["A"])), n_eval=int(len(evW) + len(evI)), n_eval_W=int(len(evW)), n_eval_I=int(len(evI)), nb_eval_W=int(nbW),
                            nb_eval_I=int(nbI), dup_of=int(dup), valid=bool(nbW >= 2 and nbI >= 2 and len(q["A"]) > 0))
        nu = sum(1 for v in info.values() if v["dup_of"] < 0)
        nv = sum(1 for v in info.values() if v["dup_of"] < 0 and v["valid"])
        for v in info.values():
            v.update(n_unique_splits=int(nu), n_valid_splits=int(nv))
        _WF10_PLAN[key] = info
    info = _WF10_PLAN[key]
    keep, skip = [], []
    for sp in a.SPLITS10:
        v = info[sp]
        if v["dup_of"] >= 0:
            skip.append((sp, f"dup_of_{v['dup_of']}", v))
        elif v["n_eval_W"] == 0 or v["n_eval_I"] == 0 or v["n_A"] == 0 or not v["valid"]:
            skip.append((sp, "invalid_wf10", v))
        else:
            keep.append(sp)
    return keep, skip, info


def build_rctx10(a, target, split, variant):
    """WF10 문맥: A = 라벨 후보 블록, 채점 B = W 의 eval_mask 셀 다음 I 의 eval_mask 셀(c.maskW 로 구분). E0 는 build_rctx 와 같다."""
    HA, D = get_data(a)
    df = D.df
    q = wf10_split(D, target, variant, split)
    A_idx, W_idx, I_idx = q["A"], q["W"], q["I"]
    evW = W_idx[eval_mask(df.iloc[W_idx])]
    evI = I_idx[eval_mask(df.iloc[I_idx])]
    evB = np.concatenate([evW, evI]).astype(int)
    _, parent, src_idx, comp = D.source_idx(target, "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    E0 = H.ls_E(df.y.values[src], df.s.values[src])
    _, _, info = wf10_plan(a, D, target, variant)
    inf = info.get(int(split), dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))
    sA, sW, sI = df.s.values[A_idx], df.s.values[evW], df.s.values[evI]
    outside = (sW > np.nanmax(sA)) if variant == "warm" else (sW < np.nanmin(sA))

    def cols(idx, names):
        return df[names].values[idx].astype(np.float32)
    meta = dict(n_A=int(len(A_idx)), nb_A=int(len(np.unique(df.block.values[A_idx]))), n_eval=int(len(evB)), n_eval_W=int(len(evW)),
                n_eval_I=int(len(evI)), nb_eval=int(len(np.unique(df.block.values[evB]))), nb_eval_W=int(len(np.unique(df.block.values[evW]))),
                nb_eval_I=int(len(np.unique(df.block.values[evI]))), n_src=int(len(src)), E0=float(E0), n_cells=int(len(D.target_idx(target))),
                dup_of=int(inf["dup_of"]), valid=bool(inf["valid"]), n_valid_splits=int(inf["n_valid_splits"]),
                n_unique_splits=int(inf["n_unique_splits"]), src_mode="x", n_src_parent=int(comp.get("n_src_parent", 0)), wf10_variant=variant,
                s_A_mean=float(np.nanmean(sA)), s_A_min=float(np.nanmin(sA)), s_A_max=float(np.nanmax(sA)), s_W_mean=float(np.nanmean(sW)),
                s_I_mean=float(np.nanmean(sI)), frac_W_outside_A=float(np.mean(outside)) if len(sW) else np.nan,
                W_blocks=";".join(q["W_blocks"]), I_blocks=";".join(q["I_blocks"]))
    c = RCtx(target, split, parent, cols(A_idx, FEATS), df.y.values[A_idx], df.s.values[A_idx], df.block.values[A_idx], df.lat.values[A_idx],
             df.lon.values[A_idx], cols(evB, FEATS), df.y.values[evB], df.s.values[evB], df.block.values[evB], df.lat.values[evB],
             df.lon.values[evB], E0, src_X=cols(src, FEATS), meta=meta)
    c.maskW = np.r_[np.ones(len(evW), bool), np.zeros(len(evI), bool)]
    c.variant = str(variant)
    return c


# ================================================================ 순수 함수: 크리깅
R_EARTH_KM = 6371.0


def xyz_km(lat, lon):
    """구면 위 3차원 직교 좌표(km). 두 점의 직선 거리는 대권 거리와 짧은 거리에서 같다."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return R_EARTH_KM * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def vario_exp(h, psill, rng, nugget):
    """지수 변이도. pykrige 의 'exponential' 과 같은 모수화(rng = 실용 범위): γ(h) = nugget + psill·(1 − exp(−3h/rng))."""
    return float(nugget) + float(psill) * (1.0 - np.exp(-3.0 * np.asarray(h, float) / float(rng)))


def fit_variogram(P, z, seed, max_pts=VARIO_MAX_PTS, nlags=VARIO_NLAGS, nrange=VARIO_NRANGE):
    """경험 변이도에 지수 모형을 맞춘다. 반환 dict(psill, range, nugget, n_fit, n_bins, sse) 또는 None(라벨 < KRIGE_MIN, 구간 < 3).
    점이 max_pts 보다 많으면 seed 로 부분표본을 뽑는다. 거리는 최대 거리의 절반까지 nlags 구간, 가중 = √(구간 쌍 수).
    범위는 로그 격자(nrange 개)에서 고르고 (psill, nugget)은 비음수 최소제곱으로 구한다."""
    from scipy.optimize import nnls
    from scipy.spatial.distance import pdist
    P = np.asarray(P, float); z = np.asarray(z, float)
    ok = np.isfinite(z) & np.all(np.isfinite(P), 1)
    P, z = P[ok], z[ok]
    if len(z) < KRIGE_MIN:
        return None
    if len(z) > max_pts:
        idx = np.sort(np.random.RandomState(int(seed) % (2 ** 31 - 1)).choice(len(z), int(max_pts), replace=False))
        P, z = P[idx], z[idx]
    d = pdist(P)
    g = 0.5 * pdist(z[:, None], "sqeuclidean")
    dmax = 0.5 * float(d.max()) if len(d) else 0.0
    if not np.isfinite(dmax) or dmax <= 0:
        return None
    m = d <= dmax
    d, g = d[m], g[m]
    edges = np.linspace(0.0, dmax, int(nlags) + 1)
    b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, int(nlags) - 1)
    cnt = np.bincount(b, minlength=int(nlags)).astype(float)
    gs = np.bincount(b, weights=g, minlength=int(nlags)); ds = np.bincount(b, weights=d, minlength=int(nlags))
    k = cnt > 0
    if k.sum() < 3:
        return None
    hc, gc, w = ds[k] / cnt[k], gs[k] / cnt[k], np.sqrt(cnt[k])
    best = None
    for rg in np.geomspace(max(float(hc.min()), 1e-3), 2.0 * dmax, int(nrange)):
        f = 1.0 - np.exp(-3.0 * hc / rg)
        A = np.c_[f, np.ones_like(f)] * w[:, None]
        coef, _ = nnls(A, gc * w)
        sse = float(np.sum((A @ coef - gc * w) ** 2))
        if best is None or sse < best[0] - 1e-12 * max(1.0, abs(best[0])):
            best = (sse, float(rg), coef)
    sse, rg, (psill, nug) = best
    if psill <= 0 and nug <= 0:                                           # 평탄한 자료: 순수 너깃(국소 평균이 된다)
        nug = max(float(np.var(z)), 1e-9)
    return dict(psill=float(psill), range=float(rg), nugget=float(nug), n_fit=int(len(z)), n_bins=int(k.sum()), sse=sse)


def ok_predict(P_lab, z_lab, P_new, vp, k=KRIGE_K, chunk=1024):
    """국소 정규 크리깅(최근접 k 개). pykrige 의 이동 창 계산(backend loop, n_closest_points)과 같은 연립식이다:
    [Γ 1; 1ᵀ 0][w; μ] = [γ0; 1], Γ_ii = 0, 거리 0 인 이웃은 γ0 = 0. 특이 행렬이면 유사역행렬을 쓴다."""
    from scipy.spatial import cKDTree
    P_lab = np.asarray(P_lab, float); z = np.asarray(z_lab, float); P_new = np.asarray(P_new, float)
    m = len(P_new)
    if m == 0:
        return np.zeros(0)
    kk = int(min(int(k), len(z)))
    dist, idx = cKDTree(P_lab).query(P_new, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    ps, rg, nu = vp["psill"], vp["range"], vp["nugget"]
    out = np.empty(m)
    ar = np.arange(kk)
    for s0 in range(0, m, int(chunk)):
        ii, dd = idx[s0:s0 + chunk], dist[s0:s0 + chunk]
        Q = P_lab[ii]
        Dm = np.sqrt(((Q[:, :, None, :] - Q[:, None, :, :]) ** 2).sum(-1))
        G = vario_exp(Dm, ps, rg, nu)
        G[:, ar, ar] = 0.0
        c_ = len(ii)
        A = np.ones((c_, kk + 1, kk + 1)); A[:, :kk, :kk] = G; A[:, kk, kk] = 0.0
        bvec = np.ones((c_, kk + 1)); bvec[:, :kk] = np.where(dd < 1e-9, 0.0, vario_exp(dd, ps, rg, nu))
        try:
            wv = np.linalg.solve(A, bvec[..., None])[..., 0]
        except np.linalg.LinAlgError:
            wv = np.einsum("cij,cj->ci", np.linalg.pinv(A), bvec)
        out[s0:s0 + chunk] = (wv[:, :kk] * z[ii]).sum(1)
    return out


def krige(P_lab, z_lab, P_new, seed):
    """변이도 적합 + 국소 크리깅. 반환 (예측 또는 None, 변이도 dict 또는 None). None 이면 호출한 쪽이 대체값을 쓴다."""
    vp = fit_variogram(P_lab, z_lab, seed)
    if vp is None:
        return None, None
    p = ok_predict(P_lab, z_lab, P_new, vp)
    if not np.all(np.isfinite(p)):
        return None, vp
    return p, vp


# ================================================================ 순수 함수: 표준화·군집·전략
def standardize_fit(X):
    """중앙값 대체 + 평균·표준편차(h40._prep_stats). 라벨을 쓰지 않는다."""
    return H._prep_stats(np.asarray(X, float))


def standardize(X, stats):
    return H._prep_apply(X, stats).astype(float)


def kmeans_labels(ZA, ZB, k, seed):
    """A 풀 표준화 공변량에 k-평균을 맞추고 A·B 의 군집 번호를 돌려준다. k 는 A 풀 셀 수를 넘지 않는다."""
    from sklearn.cluster import KMeans
    k = int(max(1, min(int(k), len(ZA))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=10, random_state=int(seed) % (2 ** 31 - 1)).fit(ZA)
    labB = km.predict(ZB) if len(ZB) else np.zeros(0, int)
    return np.asarray(km.labels_, int), np.asarray(labB, int)


def proportional_alloc(sizes, n):
    """군집 크기 비례 배분(최대 나머지 방식). 군집 크기를 넘지 않게 남는 몫은 여유 있는 군집에 차례로 준다."""
    sizes = np.asarray(sizes, int)
    n = int(min(n, sizes.sum()))
    q = n * sizes / max(sizes.sum(), 1)
    alloc = np.minimum(np.floor(q).astype(int), sizes)
    rem = q - np.floor(q)
    for j in np.argsort(-rem, kind="stable"):
        if alloc.sum() >= n:
            break
        if alloc[j] < sizes[j]:
            alloc[j] += 1
    while alloc.sum() < n:                                                # 몫이 군집 크기에 막힌 경우
        j = int(np.argmax(sizes - alloc))
        alloc[j] += 1
    return alloc


def kcenter_order(Z, n_max, seed, init=None):
    """탐욕 k-중심 순서. init(이미 고른 점의 좌표)가 있으면 그 집합에서 시작하고, 없으면 첫 점을 seed 로 무작위로 고른다.
    그 뒤 기존 집합까지의 최소 거리가 가장 큰 점을 차례로 더한다(동률은 작은 색인)."""
    Z = np.asarray(Z, float)
    N = len(Z)
    n_max = int(min(n_max, N))
    order = []
    if init is not None and len(init):
        from scipy.spatial import cKDTree
        mind = cKDTree(np.asarray(init, float)).query(Z, k=1)[0]
    else:
        j0 = int(np.random.RandomState(int(seed) % (2 ** 31 - 1)).randint(N))
        order.append(j0)
        mind = np.sqrt(((Z - Z[j0]) ** 2).sum(1)); mind[j0] = -1.0
    while len(order) < n_max:
        j = int(np.argmax(mind))
        order.append(j)
        mind = np.minimum(mind, np.sqrt(((Z - Z[j]) ** 2).sum(1))); mind[j] = -1.0       # 고른 점은 −1 로 남는다(min 이 유지한다)
    return np.array(order, int)


def aoa_di(Zc, Zsrc, seed, sub=AOA_SUB):
    """AOA 비유사도(Meyer·Pebesma 2021, 변수 가중 없음): 후보 셀의 원천 최근접 거리 / 원천 안 평균 쌍 거리(부분표본 sub)."""
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist
    Zsrc = np.asarray(Zsrc, float)
    dmin = cKDTree(Zsrc).query(np.asarray(Zc, float), k=1)[0]
    S = Zsrc
    if len(S) > sub:
        S = S[np.sort(np.random.RandomState(int(seed) % (2 ** 31 - 1)).choice(len(S), int(sub), replace=False))]
    dbar = float(np.mean(pdist(S))) if len(S) > 1 else 1.0
    return dmin / max(dbar, 1e-12)


def rank_desc(score, seed):
    """점수가 큰 순서의 색인(동률은 seed 로 섞는다)."""
    score = np.asarray(score, float)
    tie = np.random.RandomState(int(seed) % (2 ** 31 - 1)).rand(len(score))
    return np.lexsort((tie, -score))


def ve_variance(Xtr, ytr, Xc, seed, iters, threads):
    """CatBoost 가상 앙상블 분산(posterior_sampling = True, catboost_lo 설정). 반환 = 후보 셀별 성분 사이 분산.
    가상 앙상블 10개에는 트리가 충분해야 하므로 반복 수의 하한을 10 × VE_COUNT 로 둔다(설계값 200 에는 영향이 없다)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=max(int(iters), 10 * VE_COUNT), learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=int(seed) % (2 ** 31 - 1), verbose=0,
                          allow_writing_files=False, thread_count=int(threads), posterior_sampling=True)
    m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, float))
    p = np.asarray(m.virtual_ensembles_predict(np.asarray(Xc, np.float32), prediction_type="VirtEnsembles",
                                               virtual_ensembles_count=VE_COUNT, thread_count=int(threads)), float)
    return p.reshape(len(Xc), VE_COUNT, -1)[:, :, 0].var(1)


def re_anchor(base, cci, ccv):
    """Re 앵커: CCI 가 유효한 셀은 Stefan(E_n·s)과 CCI ALT 의 평균, 그 밖은 Stefan."""
    base = np.asarray(base, float); cci = np.asarray(cci, float); ccv = np.asarray(ccv, float)
    ok = np.isfinite(cci) & np.isfinite(ccv) & (ccv >= 0.5)
    return np.where(ok, 0.5 * (base + np.where(ok, cci, 0.0)), base)


def onehot(lab, k):
    return np.eye(int(k), dtype=np.float32)[np.asarray(lab, int)]


def est_fit_s(learner, nrow):
    """적합 1건의 추정 시간(s, 4스레드 기준). LG BASE_FIT 모형에 소표본 하한을 둔다."""
    b, ref, ex = EST_BASE.get(learner, EST_BASE[LO])
    return float(max(b * (max(int(nrow), 1) / ref) ** ex, EST_FLOOR.get(learner, 0.05)))


# ================================================================ 적합기와 작업 단위
class WFFitter(X.FitterX):
    """h42.FitterX(catboost_lo = h40.cb_fit, catboost = CB_HI 600·깊이 6·0.03)에 WF 의 시간 추정을 붙인다."""

    def _count(self, axis, learner, info):
        k = f"{axis}|{learner}|{(info or {}).get('method', '')}"
        self.n[axis] += 1; self.nd[k] += 1; self.last_flag = ""
        nr = max(int((info or {}).get("nrow", 1)), 1)
        self.rowsd[k] += nr
        self.estd[k] += est_fit_s(learner, nr)
        return k


class UnitBase:
    """작업 단위의 공용 부분: 저장(BlockStore 와 runs 행), 추적, 적합 호출, 마무리."""

    def _init_base(self, a, c, exp, name, variant, dry):
        self.a, self.c, self.exp, self.variant, self.dry = a, c, exp, str(variant or ""), bool(dry)
        self.F = WFFitter(h40_args(a), dry)
        self.F.nrow_fn = lambda ax, lr, info: int(info.get("nrow", 1))
        self.name = name
        self.st = BlockStore(name, c.split, c.blkB, meta=dict(target=name.split("|")[0], mode=c.mode, exp=exp, variant=self.variant))
        self.rows, self.n_rows, self.n_bad = [], 0, 0
        self.notes: dict = {}
        self.n_krige, self.krige_s, self.est_krige = 0, 0.0, 0.0
        self.n_km, self.km_s = 0, 0.0
        self.nA, self.nB = len(c.yA), len(c.yB)
        self.seeds = list(a.SEEDS)
        self.base_row = dict(exp=exp, target=name.split("|")[0], mode=c.mode, parent=c.parent, split=int(c.split), variant=self.variant, part="cpu",
                             axis=exp)
        self.xstores = self._make_xstores()

    def _make_xstores(self):
        """같은 예측을 다른 채점 부분집합·변환으로 함께 저장할 저장소 [(BlockStore, fn(y, pred) → (y', pred'))]. 기본은 없다(wf1–wf8)."""
        return []

    # ------------------------------------------------------------ 저장
    def add(self, method, learner, placement, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", sel_info="", **kw):
        self.n_rows += 1
        if self.dry:
            return None
        pred = np.asarray(pred, float); y = self.c.yB
        nf = int((~np.isfinite(pred)).sum())
        row = dict(self.base_row, method=method, learner=learner, alpha="1", placement=placement, n=int(n), n_lab=int(n_lab), draw=int(d),
                   seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used), alpha_sel=str(sel_info),
                   n_blocks_lab=int(nb_lab), n_nonfinite=nf, fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"),
                   n_train_rows=int(kw.get("nrow", 0)), cv_folds=int(kw.get("cv_folds", 0)), cv_flag=str(kw.get("cv_flag", "")))
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다(h40 과 같다)
            self.n_bad += 1
            self.rows.append(row)
            return None
        key = (method, learner, "1", placement, int(n), int(d), int(seed), float(lam))
        self.st.add(key, y, pred)
        for st_x, fn in self.xstores:                                    # 3차 보강: 격자 안·사이(wf9, wf9x), W·I(wf10)
            yy, pp = fn(y, pred)
            st_x.add(key, yy, pp)
        if PRED_TRACE is not None:
            PRED_TRACE[(self.exp, self.name, int(self.c.split), self.variant) + key] = pred.copy()
        sse, cnt = self.st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
            rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
        row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.mean(pred - y)) if len(y) else np.nan)
        self.rows.append(row)
        return key

    def emit(self, method, lr, n, d, seed, anchorB, g, E_used, nl, nb, lcv=None, K=0, cf="", flag="", placement="cell", sel_info="", nrow=0):
        """잔차 방법의 저장: λ 0.25·0.5·1.0 과(lcv 가 있으면) 교차검증 λ(키 lam = −1)."""
        anchorB = np.asarray(anchorB, float); g = np.asarray(g, float)
        for lam in LAMS:
            self.add(method, lr, placement, n, d, seed, lam, anchorB + lam * g, E_used, nl, nb, flag=flag, sel_info=sel_info, nrow=nrow)
        if lcv is not None:
            self.add(method, lr, placement, n, d, seed, LAM_CV, anchorB + float(lcv) * g, E_used, nl, nb, flag=flag, sel_info=f"lam={lcv}",
                     cv_folds=K, cv_flag=cf, nrow=nrow)

    def trace(self, kind, idx, **kw):
        if WF_TRACE is not None:
            WF_TRACE.append(dict(kind=kind, exp=self.exp, name=self.name, split=int(self.c.split), variant=self.variant,
                                 idx=np.array(idx, int).copy(), **kw))

    def fit(self, lr, method, build, nrow, seed, preds, n, d, lab_idx, placement="cell", **tr):
        """build() → (Xtr, ytr, None). lab_idx = 학습에 들어간 대상 라벨의 A 색인(추적용). preds 의 예측 목록을 돌려준다."""
        self.trace("fit", lab_idx, method=method, learner=lr, n=int(n), draw=str(d), seed=int(seed), **tr)
        inf = dict(method=method, n=int(n) if not isinstance(n, str) else n, draw=d, alpha="1", placement=placement,
                   sel=np.asarray(lab_idx, int), nrow=int(nrow))
        _, out = self.F.fit(lr, self.exp, build, seed, list(preds), info=inf)
        return out

    def nb(self, idx):
        return int(len(np.unique(self.c.blkA[idx]))) if len(idx) else 0

    def coefs(self, idx):
        """(E_n, E_ls). 라벨이 없거나 E_ls 가 비유한이면 둘 다 E0(h40·h42 와 같다)."""
        c = self.c
        if len(idx) == 0:
            return float(c.E0), float(c.E0)
        E_ls = H.ls_E(c.yA[idx], c.sA[idx])
        if not np.isfinite(E_ls):
            return float(c.E0), float(c.E0)
        return X.shrink(E_ls, c.E0, len(idx), KAPPA), float(E_ls)

    def folds(self, idx, n, d):
        """교차검증 묶음(h42.cv_folds_of: 라벨의 A 블록 순열 5묶음, 블록이 하나면 셀 묶음)."""
        return X.cv_folds_of(self.c.blkA[idx], self.c.target, self.c.mode, self.c.split, n, d)

    def finish(self):
        F = self.F
        n_ml = sum(1 for k in self.st.keys if k[1] != "none")
        n_fail = int(sum(F.fail.values())) + int(self.n_bad)
        status = "ok" if n_fail == 0 else ("failed" if n_ml == 0 else "partial")
        stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(self.n_rows), status=status,
                     n_fit_detail=dict(F.nd), sec_detail={k: round(v, 3) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                     est_detail={k: round(v, 3) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(len(self.st)),
                     n_stored_ml=int(n_ml), n_nonfinite_keys=int(self.n_bad), notes=self.notes, n_krige=int(self.n_krige),
                     krige_s=round(self.krige_s, 2), est_krige_s=round(self.est_krige, 2), n_kmeans=int(self.n_km), kmeans_s=round(self.km_s, 2),
                     est_kmeans_s=round(self.n_km * EST_KMEANS, 2))
        if self.xstores:
            stats["store_names"] = [self.st.target] + [st_x.target for st_x, _ in self.xstores]
            return self.rows, [self.st] + [st_x for st_x, _ in self.xstores], stats
        return self.rows, self.st, stats


class RUnit(UnitBase):
    """지역 내 작업 단위(wf1, wf2, wf3). 학습 행 = 선택 라벨(과 유사라벨 행)뿐이다."""

    def __init__(self, a, c, exp, variant="", dry=False):
        self._init_base(a, c, exp, f"{c.target}|{c.mode}", variant, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)          # P0 은 모든 비교의 기준이라 항상 저장한다

    # ------------------------------------------------------------ 추출·행렬
    def draw(self, n, d):
        return H.draw_cells(self.c.target, self.c.mode, self.c.split, n, d, self.nA)

    def ps_idx(self, n, d, seed, tag, n_lab):
        """유사라벨 행 색인: A 풀에서 round(r × 학습 라벨 수) 개 복원 추출."""
        n_ps = int(round(R_PS * int(n_lab)))
        if n_ps <= 0:
            return np.zeros(0, int)
        return np.random.RandomState(seed_of("wf-ps", self.c.target, self.c.mode, self.c.split, n, d, seed, tag)).choice(self.nA, n_ps, replace=True)

    def fit_direct(self, lr, method, idx, seed, n, d, xs="x25", pseudo_E=None, preds=None, tag="", placement="cell"):
        c = self.c; XA = c.XAf(xs)
        ps = self.ps_idx(n, d, seed, tag, len(idx)) if pseudo_E is not None else np.zeros(0, int)

        def build():
            Xr, yr = [XA[idx]], [c.yA[idx]]
            if len(ps):
                Xr.append(XA[ps]); yr.append(X.pseudo_values(float(pseudo_E) * c.sA, ps, idx, c.yA))
            return np.vstack(Xr), np.concatenate(yr), None
        return self.fit(lr, method, build, len(idx) + len(ps), seed, [c.XBf(xs)] if preds is None else preds, n, d, idx, placement=placement,
                        tag=tag)

    def fit_resid(self, lr, method, idx, anchor_lab, seed, n, d, xs="x25", pseudo_E=None, preds=None, tag="", placement="cell"):
        c = self.c; XA = c.XAf(xs)
        ps = self.ps_idx(n, d, seed, tag, len(idx)) if pseudo_E is not None else np.zeros(0, int)
        anchor_lab = np.asarray(anchor_lab, float)

        def build():
            Xr, yr = [XA[idx]], [c.yA[idx] - anchor_lab]
            if len(ps):
                Xr.append(XA[ps]); yr.append(X.pseudo_values(None, ps, idx, c.yA, float(pseudo_E) * c.sA))
            return np.vstack(Xr), np.concatenate(yr), None
        return self.fit(lr, method, build, len(idx) + len(ps), seed, [c.XBf(xs)] if preds is None else preds, n, d, idx, placement=placement,
                        tag=tag)

    def anchor_A(self, kind, E, idx):
        base = float(E) * self.c.sA[idx]
        return re_anchor(base, self.c.XA[idx, CCI_COL], self.c.XA[idx, CCIV_COL]) if kind == "Re" else base

    def anchor_B(self, kind, E):
        base = float(E) * self.c.sB
        return re_anchor(base, self.c.XB[:, CCI_COL], self.c.XB[:, CCIV_COL]) if kind == "Re" else base

    # ------------------------------------------------------------ 크리깅·군집
    def _krige(self, idx, z, P_new, n, d, tag):
        self.n_krige += 1
        self.est_krige += EST_KRIGE[0] + EST_KRIGE[1] * len(P_new)
        self.trace("krige", idx, n=n, draw=str(d), tag=tag)
        if self.dry:
            return np.zeros(len(P_new)), ""
        if len(idx) < KRIGE_MIN:
            return None, f"krige_fallback(n<{KRIGE_MIN})"
        t0 = time.time()
        p, vp = krige(self.c.PA[idx], z, P_new, seed_of("wf-vario", self.c.target, self.c.split, n, d, tag))
        self.krige_s += time.time() - t0
        return (p, "") if p is not None else (None, "krige_fallback(variogram)")

    def krige_E(self, idx, P_new, n, d, tag):
        """라벨 위치의 E = y/s 를 크리깅한다. 반환 (Ê 또는 None, 표지)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            z = self.c.yA[idx] / self.c.sA[idx]
        return self._krige(idx, z, P_new, n, d, tag)

    def std_stats(self):
        if self._std is None:
            self._std = standardize_fit(self.c.XA)
        return self._std

    def clusters(self, k):
        """(A 군집 번호, B 군집 번호). 표준화와 k-평균은 A 풀 공변량만 쓴다(라벨 미사용)."""
        if k not in self._km:
            self.n_km += 1
            if self.dry:
                labs = (np.random.RandomState(k).randint(0, k, self.nA), np.random.RandomState(k + 7).randint(0, k, self.nB))
            else:
                t0 = time.time(); st = self.std_stats()
                labs = kmeans_labels(standardize(self.c.XA, st), standardize(self.c.XB, st), k, seed_of("wf-km", self.c.target, self.c.split, k))
                self.km_s += time.time() - t0
            self._km[k] = labs
        return self._km[k]

    def pc_E(self, idx, k, labs_target):
        """군집 계수: 군집 c 의 E_ls,c 를 학습 라벨의 전체 E_n 으로 κ 수축. 라벨이 없는 군집은 E_n."""
        c = self.c
        labA, _ = self.clusters(k)
        E_n, _ = self.coefs(idx)
        Ec = np.full(int(k), float(E_n))
        li = labA[idx]
        for cl in range(int(k)):
            m = idx[li == cl]
            if len(m):
                Ec[cl] = X.shrink(H.ls_E(c.yA[m], c.sA[m]), E_n, len(m), KAPPA)
        return Ec[np.asarray(labs_target, int)]

    def pc_choose_k(self, idx, n, d):
        """Pc 의 k 를 학습 라벨 안 블록 교차검증으로 고른다(동률이면 작은 k). 묶음이 2개 미만이면 k = 2."""
        c = self.c
        fid, K, flag = self.folds(idx, n, d)
        if K < 2:
            return CLUSTER_KS[0], int(K), flag or "folds<2"
        sse = {k: 0.0 for k in CLUSTER_KS}
        for j in range(K):
            tr, te = idx[fid != j], idx[fid == j]
            self.trace("cv", tr, method="Pc", n=n, draw=str(d), fold=j, held=te)
            for k in CLUSTER_KS:
                labA, _ = self.clusters(k)
                sse[k] += float(np.sum((self.pc_E(tr, k, labA[te]) * c.sA[te] - c.yA[te]) ** 2))
        return min(CLUSTER_KS, key=lambda v: (sse[v] if np.isfinite(sse[v]) else np.inf, v)), int(K), flag

    def pbest_choose(self, sel, n, d):
        """최선 물리 보정식: P1, P2, Pk, Pc(안쪽 교차검증 k) 가운데 블록 교차검증 오차가 가장 작은 것(동률은 후보 순서)."""
        c = self.c
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return "P1", int(K), flag or "folds<2"
        sse = {m: 0.0 for m in PBEST_CANDS}
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            self.trace("cv", tr, method="Pbest", n=n, draw=str(d), fold=j, held=te)
            E1, E2 = self.coefs(tr)
            s, y = c.sA[te], c.yA[te]
            Ek, _ = self.krige_E(tr, c.PA[te], n, d, f"pb{j}")
            kj, _, _ = self.pc_choose_k(tr, n, f"{d}.{j}")
            labA, _ = self.clusters(kj)
            pr = dict(P1=E1 * s, P2=E2 * s, Pk=(Ek if Ek is not None else np.full(len(te), E1)) * s, Pc=self.pc_E(tr, kj, labA[te]) * s)
            for m in PBEST_CANDS:
                sse[m] += float(np.sum((pr[m] - y) ** 2))
        if self.dry:
            return "P1", int(K), flag
        return min(PBEST_CANDS, key=lambda m: (sse[m] if np.isfinite(sse[m]) else np.inf, PBEST_CANDS.index(m))), int(K), flag

    def lam_cv(self, kind, lr, xs, sel, n, d, suf=""):
        """잔차 방법(R1, R2, Re)의 λ 교차검증(seed 0 적합). 반환 (λ, 묶음 수, 표지). 묶음이 2개 미만이면 λ 0.25."""
        c = self.c
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return LAM_BASE, int(K), flag or "folds<2"
        sse = {lam: 0.0 for lam in LAMS}
        seed0 = self.seeds[0]
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            E_tr, _ = self.coefs(tr)
            a_tr, a_te = self.anchor_A(kind, E_tr, tr), self.anchor_A(kind, E_tr, te)
            (g,) = self.fit_resid(lr, f"{kind}{suf}cv", tr, a_tr, seed0, n, d, xs=xs, pseudo_E=E_tr if kind == "R2" else None,
                                  preds=[c.XAf(xs)[te]], tag=f"cv{j}")
            for lam in LAMS:
                sse[lam] += float(np.sum((a_te + lam * np.asarray(g, float) - c.yA[te]) ** 2))
        if self.dry:
            return LAM_BASE, int(K), flag
        if not any(np.isfinite(sse[lam]) for lam in LAMS):
            return LAM_BASE, int(K), "cv_failed"
        return float(min(LAMS, key=lambda lam: (sse[lam] if np.isfinite(sse[lam]) else np.inf, lam))), int(K), flag

    # ------------------------------------------------------------ 물리 보정식(wf1, wf3)
    def physics(self, n, d, sel, E1, E2, which=("P1", "P2", "Pk", "Pc", "Pbest"), placement="cell"):
        c = self.c; nl, nb = len(sel), self.nb(sel)
        self.trace("coef", sel, n=n, draw=str(d))
        out = dict(P1=E1 * c.sB, P2=E2 * c.sB)
        for m in ("P1", "P2"):
            if m in which:
                self.add(m, "none", placement, n, d, -1, 0.0, out[m], E1 if m == "P1" else E2, nl, nb)
        if "Pk" in which or "Pbest" in which:
            Ek, fl = self.krige_E(sel, c.PB, n, d, "full")
            out["Pk"] = (Ek if Ek is not None else np.full(self.nB, E1)) * c.sB
            if "Pk" in which:
                self.add("Pk", "none", placement, n, d, -1, 0.0, out["Pk"], np.nan, nl, nb, flag=fl)
        if "Pc" in which or "Pbest" in which:
            for k in CLUSTER_KS:
                _, labB = self.clusters(k)
                out[f"Pc@c{k}"] = self.pc_E(sel, k, labB) * c.sB
                self.add(f"Pc@c{k}", "none", placement, n, d, -1, 0.0, out[f"Pc@c{k}"], np.nan, nl, nb)
            kc, K, fl = self.pc_choose_k(sel, n, d)
            out["Pc"] = out[f"Pc@c{kc}"]
            self.add("Pc", "none", placement, n, d, -1, 0.0, out["Pc"], np.nan, nl, nb, sel_info=f"k={kc}", cv_folds=K, cv_flag=fl)
        if "Pbest" in which:
            best, K, fl = self.pbest_choose(sel, n, d)
            self.add("Pbest", "none", placement, n, d, -1, 0.0, out[best], np.nan, nl, nb, sel_info=best, cv_folds=K, cv_flag=fl)
        return out

    # ------------------------------------------------------------ wf1
    def wf1_cell(self, n, d, xs):
        c = self.c
        sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
        self.trace("select", sel, n=n, draw=str(d))
        E1, E2 = self.coefs(sel)
        if xs == "x25":
            self.physics(n, d, sel, E1, E2)
        suf = "" if xs == "x25" else "@x34"
        lrs = (LO, HI) if xs == "x25" else (LO,)
        lam = {("R1", lr): self.lam_cv("R1", lr, xs, sel, n, d, suf) for lr in lrs}
        lam[("R2", LO)] = self.lam_cv("R2", LO, xs, sel, n, d, suf)
        if xs == "x25":
            lam[("Re", LO)] = self.lam_cv("Re", LO, xs, sel, n, d)
        a1A, a1B = E1 * c.sA[sel], E1 * c.sB
        nps = int(round(R_PS * nl))
        for seed in self.seeds:
            for lr in lrs:
                (p,) = self.fit_direct(lr, "D0" + suf, sel, seed, n, d, xs=xs)
                self.add("D0" + suf, lr, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nl)
                gB, gA = self.fit_resid(lr, "R1" + suf, sel, a1A, seed, n, d, xs=xs, preds=[c.XBf(xs), c.XAf(xs)[sel]])
                fl = self.F.last_flag
                lcv, K, cf = lam[("R1", lr)]
                self.emit("R1" + suf, lr, n, d, seed, a1B, gB, E1, nl, nb, lcv, K, cf, fl, nrow=nl)
                if lr == LO and xs == "x25":
                    self.rk(n, d, seed, sel, a1A, gA, a1B, gB, lcv, E1, nl, nb)
            (p,) = self.fit_direct(LO, "D1" + suf, sel, seed, n, d, xs=xs, pseudo_E=E1)
            self.add("D1" + suf, LO, "cell", n, d, seed, 1.0, p, E1, nl, nb, flag=self.F.last_flag, nrow=nl + nps)
            (g,) = self.fit_resid(LO, "R2" + suf, sel, a1A, seed, n, d, xs=xs, pseudo_E=E1)
            lcv, K, cf = lam[("R2", LO)]
            self.emit("R2" + suf, LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl + nps)
            if xs == "x25":
                aeA, aeB = self.anchor_A("Re", E1, sel), self.anchor_B("Re", E1)
                (g,) = self.fit_resid(LO, "Re", sel, aeA, seed, n, d)
                fl = self.F.last_flag
                lcv, K, cf = lam[("Re", LO)]
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * np.asarray(g, float), E1, nl, nb, flag=fl, nrow=nl)
                self.add("Re", LO, "cell", n, d, seed, LAM_CV, aeB + lcv * np.asarray(g, float), E1, nl, nb, flag=fl, sel_info=f"lam={lcv}",
                         cv_folds=K, cv_flag=cf, nrow=nl)

    def rk(self, n, d, seed, sel, a1A, gA, a1B, gB, lcv, E1, nl, nb):
        """RK = R1(λ) + 표본 안 잔차의 국소 크리깅. λ 0.25·0.5·1.0 과 R1 의 교차검증 λ."""
        c = self.c
        gA, gB = np.asarray(gA, float), np.asarray(gB, float)
        for lam in list(LAMS) + [LAM_CV]:
            lv = float(lcv) if lam == LAM_CV else float(lam)
            r_lab = c.yA[sel] - (a1A + lv * gA)
            rB, fl = self._krige(sel, r_lab, c.PB, n, d, f"rk{seed}_{lam}")
            pred = a1B + lv * gB + (rB if rB is not None else 0.0)
            self.add("RK", LO, "cell", n, d, seed, lam, pred, E1, nl, nb, flag=fl, sel_info=f"lam={lv}" if lam == LAM_CV else "", nrow=nl)

    def run_wf1(self):
        xs = "x34" if self.variant == "x34" else "x25"
        if xs == "x34" and self.c.XA34 is None:
            raise RuntimeError(f"{self.c.target}: x34 열(SAR)이 자료에 없다")
        for n, d in cells_for(self.a, "wf1", self.nA):
            self.wf1_cell(n, d, xs)

    # ------------------------------------------------------------ wf6(검정력 보강, 계획서 §6)
    def run_wf6(self):
        """wf1 과 같은 지역 내 문맥·추출·교차검증에서 WF6 의 방법만 적합한다. 물리 보정식은 wf1 의 묶음 그대로(Pbest 는 보조 대비),
        R1·R2·Re 는 catboost_lo(교차검증 λ 와 고정 λ 키), D0 은 catboost(기본 용량)만이다."""
        c = self.c
        for n, d in cells_for(self.a, "wf6", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.physics(n, d, sel, E1, E2)
            lam = {m: self.lam_cv(m, LO, "x25", sel, n, d) for m in ("R1", "R2", "Re")}
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            aeA, aeB = self.anchor_A("Re", E1, sel), self.anchor_B("Re", E1)
            nps = int(round(R_PS * nl))
            for seed in self.seeds:
                (p,) = self.fit_direct(HI, "D0", sel, seed, n, d)
                self.add("D0", HI, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d)
                lcv, K, cf = lam["R1"]
                self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "R2", sel, a1A, seed, n, d, pseudo_E=E1)
                lcv, K, cf = lam["R2"]
                self.emit("R2", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl + nps)
                (g,) = self.fit_resid(LO, "Re", sel, aeA, seed, n, d)
                fl = self.F.last_flag
                g = np.asarray(g, float)
                lcv, K, cf = lam["Re"]
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * g, E1, nl, nb, flag=fl, nrow=nl)
                self.add("Re", LO, "cell", n, d, seed, LAM_CV, aeB + lcv * g, E1, nl, nb, flag=fl, sel_info=f"lam={lcv}", cv_folds=K, cv_flag=cf,
                         nrow=nl)

    # ------------------------------------------------------------ wf3
    def run_wf3(self):
        c = self.c
        for n, d in cells_for(self.a, "wf3", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.physics(n, d, sel, E1, E2, which=("P1", "Pc"))
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            for seed in self.seeds:
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d)
                g = np.asarray(g, float)
                self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, flag=self.F.last_flag, nrow=nl)
                for k in CLUSTER_KS:
                    labA, labB = self.clusters(k)
                    gc, own, fl = g.copy(), 0, ""
                    for cl in range(int(k)):
                        m_lab = sel[labA[sel] == cl]
                        mB = labB == cl
                        if len(m_lab) < R1C_MIN or not mB.any():
                            continue
                        (gcl,) = self.fit_resid(LO, f"R1c@c{k}", m_lab, E1 * c.sA[m_lab], seed, n, d, preds=[c.XB[mB]], tag=f"c{cl}")
                        gc[mB] = np.asarray(gcl, float); own += 1; fl = fl or self.F.last_flag
                    self.emit(f"R1c@c{k}", LO, n, d, seed, a1B, gc, E1, nl, nb, flag=fl, sel_info=f"own={own}/{k}", nrow=nl)
                    OA, OB = onehot(labA, k), onehot(labB, k)
                    (gi,) = self.fit(LO, f"R1i@c{k}", lambda OA=OA: (np.hstack([c.XA[sel], OA[sel]]), c.yA[sel] - a1A, None), nl, seed,
                                     [np.hstack([c.XB, OB])], n, d, sel)
                    self.emit(f"R1i@c{k}", LO, n, d, seed, a1B, gi, E1, nl, nb, flag=self.F.last_flag, nrow=nl)

    # ------------------------------------------------------------ wf2(라벨 위치 선택)
    def s3_set(self, n, d):
        labA, _ = self.clusters(S3_K)
        K = int(labA.max()) + 1 if len(labA) else 1
        sizes = np.bincount(labA, minlength=K)
        alloc = proportional_alloc(sizes, n)
        rng = np.random.RandomState(seed_of("wf-s3", self.c.target, self.c.mode, self.c.split, n, d))
        out = [rng.choice(np.where(labA == cl)[0], int(alloc[cl]), replace=False) for cl in range(K) if alloc[cl] > 0]
        return np.sort(np.concatenate(out).astype(int)) if out else np.zeros(0, int)

    def s4_order(self, nmax, d):
        if self.dry:
            return np.random.RandomState(d).permutation(self.nA)[:nmax]
        Z = standardize(self.c.XA, self.std_stats())
        return kcenter_order(Z, nmax, seed_of("wf-s4", self.c.target, self.c.split, d))

    def s5_order(self, d):
        if self.dry:
            return np.random.RandomState(d + 11).permutation(self.nA)
        if self._di is None:
            if self.c.src_X is None or not len(self.c.src_X):
                raise RuntimeError("S5: 원천 공변량(src_X)이 없다")
            st = standardize_fit(self.c.src_X)
            self._di = aoa_di(standardize(self.c.XA, st), standardize(self.c.src_X, st), seed_of("wf-aoa", self.c.target, self.c.split))
            self.notes["s5_di"] = dict(min=float(np.min(self._di)), median=float(np.median(self._di)), max=float(np.max(self._di)))
        return rank_desc(self._di, seed_of("wf-s5", self.c.target, self.c.split, d))

    def s6_var(self, cur, cand, d):
        """S6 의 선택 점수: R1 잔차 모형의 가상 앙상블 분산. 학습 = 현재 라벨(cur)뿐."""
        c = self.c
        E_n, _ = self.coefs(cur)
        self.trace("s6", cur, n=len(cur), draw=str(d))
        k = self.F._count(self.exp, "catboost_ve", dict(method="S6sel", nrow=len(cur)))
        if self.dry:
            return np.random.RandomState(len(cur) + d).rand(len(cand))
        t0 = time.time()
        try:
            v = ve_variance(c.XA[cur], c.yA[cur] - E_n * c.sA[cur], c.XA[cand], seed_of("wf-s6", c.target, c.split, d, len(cur)), self.a.cb_iters,
                            self.a.threads)
        except Exception as e:                                            # noqa: BLE001  선택 적합 실패: 그 단계만 무작위(표지)
            self.F.fail[self.exp] += 1
            self.F.errors.append(f"{k}: {repr(e)[:160]}")
            self.notes.setdefault("s6_fallback", []).append([int(d), int(len(cur))])
            v = np.random.RandomState(seed_of("wf-s6f", c.target, c.split, d, len(cur))).rand(len(cand))
        dt = time.time() - t0
        self.F.sec[self.exp] += dt; self.F.secd[k] += dt
        return v

    def s6_sets(self, budgets, d):
        bset = sorted(budgets)
        start = min(S6_START, bset[0])
        cur = list(self.draw(start, d))
        out = {start: np.sort(np.array(cur, int))} if start in bset else {}
        while len(cur) < bset[-1]:
            nxt = min(b for b in bset if b > len(cur))
            k = min(S6_BATCH, nxt - len(cur))
            ca = np.sort(np.array(cur, int))
            cand = np.setdiff1d(np.arange(self.nA), ca)
            v = self.s6_var(ca, cand, d)
            pick = cand[rank_desc(v, seed_of("wf-s6t", self.c.target, self.c.split, d, len(cur)))[:k]]
            cur += [int(v_) for v_ in pick]
            if len(cur) in bset:
                out[len(cur)] = np.sort(np.array(cur, int))
        return out

    def strategy_sets(self, strat, budgets, d):
        c = self.c
        if strat == "S1":
            return {n: self.draw(n, d) for n in budgets}
        if strat == "S2":
            return {n: H.draw_blocks(c.target, c.mode, c.split, n, d, c.blkA) for n in budgets}
        if strat == "S3":
            return {n: self.s3_set(n, d) for n in budgets}
        if strat == "S4":
            order = self.s4_order(max(budgets), d)
            return {n: np.sort(order[:n]) for n in budgets}
        if strat == "S5":
            order = self.s5_order(d)
            return {n: np.sort(order[:n]) for n in budgets}
        if strat == "S6":
            return self.s6_sets(budgets, d)
        if strat == "S7":
            order = rank_desc(c.sA, seed_of("wf-s7", c.target, c.split, d))
            return {n: np.sort(order[:n]) for n in budgets}
        raise ValueError(f"알 수 없는 전략 {strat}")

    def run_wf2(self):
        c = self.c; strat = self.variant
        budgets = [b for b in self.a.G["wf2"] if 0 < b < self.nA]
        if not budgets:
            return
        seen: dict = {}
        for d in range(draws_for(self.a, "wf2", budgets[0])):
            sets = self.strategy_sets(strat, budgets, d)
            for n in budgets:
                sel = np.sort(np.asarray(sets[n], int))
                key = (n, tuple(sel.tolist()))
                if key in seen:                                           # 같은 라벨 집합(결정적 전략): 첫 추출만 적합·저장한다
                    self.notes.setdefault("dup_draws", []).append([int(n), int(d), int(seen[key])])
                    continue
                seen[key] = d
                self.trace("select", sel, n=n, draw=str(d), strategy=strat)
                nl, nb = len(sel), self.nb(sel)
                E1, _ = self.coefs(sel)
                self.trace("coef", sel, n=n, draw=str(d))
                self.add("P1", "none", strat, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
                a1A, a1B = E1 * c.sA[sel], E1 * c.sB
                for seed in self.seeds:
                    (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d, placement=strat)
                    self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, flag=self.F.last_flag, placement=strat, nrow=nl)
                    (p,) = self.fit_direct(LO, "D1", sel, seed, n, d, pseudo_E=E1, placement=strat)
                    self.add("D1", LO, strat, n, d, seed, 1.0, p, E1, nl, nb, flag=self.F.last_flag, nrow=nl + int(round(R_PS * nl)))

    def run(self):
        if self.exp == "wf1":
            self.run_wf1()
        elif self.exp == "wf2":
            self.run_wf2()
        elif self.exp == "wf3":
            self.run_wf3()
        elif self.exp == "wf6":
            self.run_wf6()
        else:
            raise ValueError(self.exp)
        return self


class TUnit(UnitBase):
    """전이 작업 단위(wf4). 학습 행 = 원천 행 + 선택 라벨(+ 유사라벨 행). 행렬과 추출은 LG(h40)와 같다.
    exp·variant 는 2차 보강의 전이 실험(T7Unit = wf7, T8Unit = wf8)이 같은 문맥과 행렬 함수를 쓰려고 둔 인자다(wf4 는 기본값)."""

    def __init__(self, a, c, alias, dry=False, exp="wf4", variant=""):
        self._init_base(a, c, exp, f"{alias}|{c.mode}", variant, dry)
        self.alias = alias
        self.nsrc = len(c.y_src)
        self.n_ps = int(round(R_PS * self.nsrc))
        self._ps: dict = {}
        self.diag: list = []
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)

    def draw(self, n, d):
        return H.draw_cells(self.c.target, self.c.mode, self.c.split, n, d, self.nA)       # LG 와 같은 추출(대상 이름은 표 안의 이름)

    def ps(self, seed):
        if seed not in self._ps:
            c = self.c
            self._ps[seed] = X.ps_index(c.target, c.mode, c.split, seed, self.nA, self.nsrc, R_PS)
        return self._ps[seed]

    def rows_R(self, idx, anchor_lab, seed=None, pseudo_E=None):
        c = self.c
        pseudo = None
        if pseudo_E is not None:
            ps = self.ps(seed)
            pseudo = (c.XA[ps], X.pseudo_values(None, ps, idx, c.yA, float(pseudo_E) * c.sA))
        return X.stack_rows(c.X_src, c.r0_src, c.XA[idx], c.yA[idx] - np.asarray(anchor_lab, float), None, pseudo)

    def rows_D(self, idx, seed=None, pseudo_E=None):
        c = self.c
        pseudo = None
        if pseudo_E is not None:
            ps = self.ps(seed)
            pseudo = (c.XA[ps], X.pseudo_values(float(pseudo_E) * c.sA, ps, idx, c.yA))
        return X.stack_rows(c.X_src, c.y_src, c.XA[idx], c.yA[idx], None, pseudo)

    def select_w(self, sel, n, d):
        """규칙 W: 선택 라벨 안 5겹 블록 교차검증(seed 0 적합)으로 후보 가운데 오차가 가장 작은 방법. 라벨 < 10 이면 P1."""
        c = self.c
        if len(sel) < W_MIN_N:
            return "P1", 0, f"n<{W_MIN_N}", {}
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return "P1", int(K), flag or "folds<2", {}
        seed0 = self.seeds[0]
        sse = {m: 0.0 for m in W_CANDS}
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            self.trace("cv", tr, method="W", n=n, draw=str(d), fold=j, held=te)
            E1, E2 = self.coefs(tr)
            s, y = c.sA[te], c.yA[te]
            a_tr, a_te = E1 * c.sA[tr], E1 * s
            nl = len(tr)
            (g1,) = self.fit(LO, "R1cv", lambda: self.rows_R(tr, a_tr), self.nsrc + nl, seed0, [c.XA[te]], n, d, tr, fold=j)
            (g2,) = self.fit(LO, "R2cv", lambda: self.rows_R(tr, a_tr, seed0, E1), self.nsrc + nl + self.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
            (pd1,) = self.fit(LO, "D1cv", lambda: self.rows_D(tr, seed0, E1), self.nsrc + nl + self.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
            g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
            pr = {"P0": c.E0 * s, "P1": E1 * s, "P2": E2 * s, "R1@0.25": a_te + 0.25 * g1, "R1@1.0": a_te + g1, "R2@0.25": a_te + 0.25 * g2,
                  "D1": np.asarray(pd1, float)}
            for m in W_CANDS:
                sse[m] += float(np.sum((pr[m] - y) ** 2))
        if self.dry:
            return "P1", int(K), flag, {}
        best = min(W_CANDS, key=lambda m: (sse[m] if np.isfinite(sse[m]) else np.inf, W_CANDS.index(m)))
        return best, int(K), flag, {m: float(np.sqrt(v / max(len(sel), 1))) for m, v in sse.items()}

    def run(self):
        c = self.c
        for n, d in cells_for(self.a, "wf4", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            E1, E2 = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            pB = {"P0": c.E0 * c.sB, "P1": E1 * c.sB, "P2": E2 * c.sB}
            self.add("P1", "none", "cell", n, d, -1, 0.0, pB["P1"], E1, nl, nb)
            self.add("P2", "none", "cell", n, d, -1, 0.0, pB["P2"], E2, nl, nb)
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            preds = {}
            for seed in self.seeds:
                (g1,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.emit("R1", LO, n, d, seed, a1B, g1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl)
                (g2,) = self.fit(LO, "R2", lambda: self.rows_R(sel, a1A, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel)
                self.emit("R2", LO, n, d, seed, a1B, g2, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl + self.n_ps)
                (p1,) = self.fit(LO, "D1", lambda: self.rows_D(sel, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel)
                self.add("D1", LO, "cell", n, d, seed, 1.0, p1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl + self.n_ps)
                g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
                preds[seed] = dict(pB, **{"R1@0.25": a1B + 0.25 * g1, "R1@1.0": a1B + g1, "R2@0.25": a1B + 0.25 * g2, "D1": np.asarray(p1, float)})
            choice, K, flag, cv = self.select_w(sel, n, d)
            for seed in self.seeds:
                self.add("W", LO, "cell", n, d, seed, 0.0, preds[seed][choice], np.nan, nl, nb, sel_info=choice, cv_folds=K, cv_flag=flag)
            if cv:
                self.notes.setdefault("w_cv_rmse", {})[f"{n}|{d}"] = {m: round(v, 4) for m, v in cv.items()}
            if n == DIAG_N and nl > 0 and not self.dry:
                bias = float(np.mean(c.E0 * c.sA[sel] - c.yA[sel]))
                self.diag.append(dict(n=int(n), draw=int(d), n_lab=int(nl), bias=bias, abs_bias=abs(bias)))
        return self

    def finish(self):
        rows, st, stats = super().finish()
        stats["diag"] = list(self.diag)
        return rows, st, stats


def re_src_resid(c):
    """wf7 Re 의 원천 행 목표: y − 앵커(E0). 앵커(E0) = 0.5·(E0·s + cci_alt)(CCI 유효 셀), 그 밖 E0·s(h40 R1 의 원천 E0 앵커와 같은 구성)."""
    return np.asarray(c.y_src, float) - re_anchor(c.E0 * np.asarray(c.s_src, float), c.X_src[:, CCI_COL], c.X_src[:, CCIV_COL])


class T7Unit(TUnit):
    """wf7(계획서 §6 WF7): Stefan·CCI 평균 앵커의 전이 성능. 문맥·원천·추출·seed 는 LG(h40.build_ctx, h40.draw_cells)와 같다.
    방법 P0, P1, Pe = 앵커(E_n), R1 = E_n·s + 0.25·g1, Re = 앵커(E_n) + 0.25·ge. g1 의 학습 행 = 원천(y − E0·s) ∪ 선택 라벨(y − E_n·s),
    ge 의 학습 행 = 원천(y − 앵커(E0)) ∪ 선택 라벨(y − 앵커(E_n)). n = 0 은 E_n = E0 이고 원천 행만으로 적합한다."""

    def __init__(self, a, c, alias, dry=False):
        super().__init__(a, c, alias, dry, exp="wf7")
        self.re_src = re_src_resid(c)
        ok = np.isfinite(c.X_src[:, CCI_COL]) & np.isfinite(c.X_src[:, CCIV_COL]) & (c.X_src[:, CCIV_COL] >= 0.5)
        okB = np.isfinite(c.XB[:, CCI_COL]) & np.isfinite(c.XB[:, CCIV_COL]) & (c.XB[:, CCIV_COL] >= 0.5)
        self.notes["cci_valid_frac"] = dict(src=float(ok.mean()) if len(ok) else np.nan, B=float(okB.mean()) if len(okB) else np.nan)

    def rows_Re(self, idx, anchor_lab):
        c = self.c
        return X.stack_rows(c.X_src, self.re_src, c.XA[idx], c.yA[idx] - np.asarray(anchor_lab, float), None, None)

    def run(self):
        c = self.c
        for n, d in cells_for(self.a, "wf7", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            E1, _ = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            aeA = re_anchor(a1A, c.XA[sel, CCI_COL], c.XA[sel, CCIV_COL])
            aeB = re_anchor(a1B, c.XB[:, CCI_COL], c.XB[:, CCIV_COL])
            self.add("P1", "none", "cell", n, d, -1, 0.0, a1B, E1, nl, nb)
            self.add("Pe", "none", "cell", n, d, -1, 0.0, aeB, E1, nl, nb)
            for seed in self.seeds:
                (g1,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.add("R1", LO, "cell", n, d, seed, LAM_BASE, a1B + LAM_BASE * np.asarray(g1, float), E1, nl, nb, flag=self.F.last_flag,
                         nrow=self.nsrc + nl)
                (ge,) = self.fit(LO, "Re", lambda: self.rows_Re(sel, aeA), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * np.asarray(ge, float), E1, nl, nb, flag=self.F.last_flag,
                         nrow=self.nsrc + nl)
        return self


class T8Unit(TUnit):
    """wf8(계획서 §6 WF8): 전이 조건(모드 x)의 라벨 위치 선택. 전략 S1·S2·S4 는 wf2 의 구현(RUnit 의 함수)을 그대로 빌려 쓴다.
    S1 = h40.draw_cells(대상, x, 분할, n, 추출), S2 = h40.draw_blocks(같은 인자, A 블록), S4 = A 풀 x25 표준화(라벨 미사용)의 탐욕 k-중심.
    방법 P1, R1(λ 0.25, 원천 행 + 선택 라벨). 같은 라벨 집합이 다른 추출에서 다시 나오면 적합을 다시 하지 않고 같은 예측을 그 추출 번호로 저장한다."""

    std_stats = RUnit.std_stats                                            # wf2 의 구현을 그대로 쓴다(표준화 = A 풀 공변량)
    s4_order = RUnit.s4_order
    strategy_sets = RUnit.strategy_sets

    def __init__(self, a, c, alias, variant, dry=False):
        if variant not in WF8_STRATEGIES:
            raise ValueError(f"wf8 전략은 {WF8_STRATEGIES} 가운데 하나다: {variant}")
        super().__init__(a, c, alias, dry, exp="wf8", variant=variant)
        self._std = None

    def run(self):
        c = self.c; strat = self.variant
        budgets = [b for b in self.a.G["wf8"] if 0 < b < self.nA]
        if not budgets:
            return self
        seen: dict = {}
        for d in range(draws_for(self.a, "wf8", budgets[0])):
            sets = self.strategy_sets(strat, budgets, d)
            for n in budgets:
                sel = np.sort(np.asarray(sets[n], int))
                nl, nb = len(sel), self.nb(sel)
                key = (n, tuple(sel.tolist()))
                self.trace("select", sel, n=n, draw=str(d), strategy=strat)
                E1, _ = self.coefs(sel)
                self.trace("coef", sel, n=n, draw=str(d))
                self.add("P1", "none", strat, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
                a1A, a1B = E1 * c.sA[sel], E1 * c.sB
                dup = key in seen
                if dup:                                                   # 같은 라벨 집합: 앞 추출의 g 를 그대로 쓴다(같은 행렬·seed)
                    self.notes.setdefault("dup_draws", []).append([int(n), int(d), int(seen[key][0])])
                else:
                    seen[key] = (d, {})
                for seed in self.seeds:
                    if dup:
                        g, fl = seen[key][1][seed]
                    else:
                        (g,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel, placement=strat)
                        g, fl = np.asarray(g, float), self.F.last_flag
                        seen[key][1][seed] = (g, fl)
                    self.add("R1", LO, strat, n, d, seed, LAM_BASE, a1B + LAM_BASE * g, E1, nl, nb, flag=fl, nrow=self.nsrc + nl)
        return self


# ================================================================ 3차 보강(계획서 §7): 격자 안 분해(wf9, wf9x)와 기후 외삽(wf10)
def grid_groups(sB, blkB):
    """격자 묶음(계획서 §7 WF9): 채점 셀의 (√TDD(토양 도일) 값, 블록)이 같은 셀. √TDD 가 비유한인 셀은 혼자 묶음이다.
    반환 (묶음 번호, 2셀 이상 묶음에 속한 셀의 마스크)."""
    sB = np.asarray(sB, float); blkB = np.asarray(blkB).astype(str)
    if not len(sB):
        return np.zeros(0, np.int64), np.zeros(0, bool)
    key = [f"{b}|{v:.12g}" if np.isfinite(v) else f"{b}|nan{i}" for i, (b, v) in enumerate(zip(blkB, sB))]
    gid = pd.factorize(pd.Series(key))[0].astype(np.int64)
    cnt = np.bincount(gid)
    return gid, cnt[gid] >= 2


def decomp_fns(gid, multi):
    """격자 안·격자 사이 변환. within(y, p) = 2셀 이상 묶음 셀의 (y − ȳ_g, p − p̄_g), between(y, p) = 모든 셀의 (ȳ_g, p̄_g).
    묶음이 블록 안에 있으므로 블록마다 총 SSE = 격자 안 SSE + 격자 사이 SSE 다."""
    gid = np.asarray(gid, np.int64); multi = np.asarray(multi, bool)
    G = int(gid.max()) + 1 if len(gid) else 0
    cnt = np.maximum(np.bincount(gid, minlength=G).astype(float), 1.0)

    def means(v):
        return (np.bincount(gid, weights=np.asarray(v, float), minlength=G) / cnt)[gid]

    def within(y, p):
        return (np.asarray(y, float) - means(y))[multi], (np.asarray(p, float) - means(p))[multi]

    def between(y, p):
        return means(y), means(p)
    return within, between


class GridDecompMixin:
    """총 저장소와 함께 격자 안(<대상>~w, 2셀 이상 묶음의 셀)·격자 사이(<대상>~b, 모든 셀) 저장소를 채운다(wf9, wf9x)."""

    def _make_xstores(self):
        c = self.c
        gid, multi = grid_groups(c.sB, c.blkB)
        t, m = self.name.split("|")
        G = int(gid.max()) + 1 if len(gid) else 0
        self.notes["grid"] = dict(n_groups=G, n_multi_groups=int(np.sum(np.bincount(gid, minlength=G) >= 2)) if G else 0,
                                  n_multi_cells=int(multi.sum()), n_cells=int(len(gid)))
        w_fn, b_fn = decomp_fns(gid, multi)
        meta = dict(target=t, mode=c.mode, exp=self.exp, variant=self.variant)
        out = []
        if multi.any():
            out.append((BlockStore(f"{t}~w|{m}", c.split, np.asarray(c.blkB)[multi], meta=dict(meta, part="within")), w_fn))
        out.append((BlockStore(f"{t}~b|{m}", c.split, c.blkB, meta=dict(meta, part="between")), b_fn))
        return out


class R9Unit(GridDecompMixin, RUnit):
    """wf9 지역 내(계획서 §7 WF9): WF6 와 같은 문맥·분할·추출·교차검증. 방법 P1, Pk, Pc(교차검증 k), R1(교차검증 λ, λ 0.25·0.5·1.0),
    Re(교차검증 λ, λ 0.25), D0(catboost). 같은 예측을 총·격자 안·격자 사이 저장소에 함께 저장한다."""

    def run(self):
        c = self.c
        for n, d in cells_for(self.a, "wf9", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.physics(n, d, sel, E1, E2, which=("P1", "Pk", "Pc"))
            lam = {m: self.lam_cv(m, LO, "x25", sel, n, d) for m in ("R1", "Re")}
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            aeA, aeB = self.anchor_A("Re", E1, sel), self.anchor_B("Re", E1)
            for seed in self.seeds:
                (p,) = self.fit_direct(HI, "D0", sel, seed, n, d)
                self.add("D0", HI, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d)
                lcv, K, cf = lam["R1"]
                self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "Re", sel, aeA, seed, n, d)
                fl = self.F.last_flag
                g = np.asarray(g, float)
                lcv, K, cf = lam["Re"]
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * g, E1, nl, nb, flag=fl, nrow=nl)
                self.add("Re", LO, "cell", n, d, seed, LAM_CV, aeB + lcv * g, E1, nl, nb, flag=fl, sel_info=f"lam={lcv}", cv_folds=K, cv_flag=cf,
                         nrow=nl)
        return self


class T9Unit(GridDecompMixin, TUnit):
    """wf9x 전이(계획서 §7 WF9): LG 문맥·원천·추출·seed. 방법 P0, P1, R1(λ 0.25·0.5·1.0), R2(λ 0.25·0.5·1.0), D0(catboost_lo, 원천 행 +
    선택 라벨), D1(catboost_lo). n = 0 은 E_n = E0 이고 선택 라벨 없이 적합한다. 같은 예측을 총·격자 안·격자 사이 저장소에 함께 저장한다."""

    def __init__(self, a, c, alias, dry=False):
        super().__init__(a, c, alias, dry, exp="wf9x")

    def run(self):
        c = self.c
        for n, d in cells_for(self.a, "wf9x", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            E1, _ = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            self.add("P1", "none", "cell", n, d, -1, 0.0, a1B, E1, nl, nb)
            for seed in self.seeds:
                (g1,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.emit("R1", LO, n, d, seed, a1B, g1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl)
                (g2,) = self.fit(LO, "R2", lambda: self.rows_R(sel, a1A, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel)
                self.emit("R2", LO, n, d, seed, a1B, g2, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl + self.n_ps)
                (p0,) = self.fit(LO, "D0", lambda: self.rows_D(sel), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.add("D0", LO, "cell", n, d, seed, 1.0, p0, np.nan, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl)
                (p1,) = self.fit(LO, "D1", lambda: self.rows_D(sel, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel)
                self.add("D1", LO, "cell", n, d, seed, 1.0, p1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl + self.n_ps)
        return self


class R10Unit(RUnit):
    """wf10(계획서 §7 WF10): 블록 평균 √TDD 로 나눈 W(외삽 채점)·I(보간 채점). 같은 적합으로 W 와 I 를 따로 채점한다
    (저장소 <대상>~<변형>W, <대상>~<변형>I. 총 저장소 <대상>~<변형> 은 W ∪ I). 방법 P1, P2, R1(교차검증 λ, λ 0.25·0.5·1.0),
    R2(교차검증 λ, 고정 λ), D0(catboost 와 catboost_lo), D1(catboost_lo). 추출 seed 는 h40.draw_cells(대상, 'r10' + 변형 첫 글자, …)."""

    def __init__(self, a, c, exp, variant="", dry=False):
        if variant not in WF10_VARIANTS:
            raise ValueError(f"wf10 변형은 {WF10_VARIANTS} 가운데 하나다: {variant}")
        self._init_base(a, c, exp, f"{c.target}~{variant}|{c.mode}", variant, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)

    def _make_xstores(self):
        c = self.c
        mW = np.asarray(c.maskW, bool)
        mI = ~mW
        t, m = self.name.split("|")
        meta = dict(target=c.target, mode=c.mode, exp=self.exp, variant=self.variant)
        out = []
        for part, mk in (("W", mW), ("I", mI)):
            if mk.any():
                out.append((BlockStore(f"{t}{part}|{m}", c.split, np.asarray(c.blkB)[mk], meta=dict(meta, part=part)),
                            (lambda mk_: (lambda y, p: (np.asarray(y, float)[mk_], np.asarray(p, float)[mk_])))(mk)))
        return out

    def draw(self, n, d):
        return H.draw_cells(self.c.target, "r10" + self.variant[0], self.c.split, n, d, self.nA)

    def run(self):
        c = self.c
        for n, d in cells_for(self.a, "wf10", self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.physics(n, d, sel, E1, E2, which=("P1", "P2"))
            lam = {m: self.lam_cv(m, LO, "x25", sel, n, d) for m in ("R1", "R2")}
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            nps = int(round(R_PS * nl))
            for seed in self.seeds:
                for lr in (HI, LO):
                    (p,) = self.fit_direct(lr, "D0", sel, seed, n, d)
                    self.add("D0", lr, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nl)
                (p,) = self.fit_direct(LO, "D1", sel, seed, n, d, pseudo_E=E1)
                self.add("D1", LO, "cell", n, d, seed, 1.0, p, E1, nl, nb, flag=self.F.last_flag, nrow=nl + nps)
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d)
                lcv, K, cf = lam["R1"]
                self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "R2", sel, a1A, seed, n, d, pseudo_E=E1)
                lcv, K, cf = lam["R2"]
                self.emit("R2", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl + nps)
        return self


# ================================================================ 조각 입출력
def shard_base(a, exp, target, mode, split, variant=""):
    return a.SHARDS / (f"{exp_tag(a, exp)}__cpu__{target}__{mode}__s{int(split)}" + (f"__{variant}" if variant else ""))


def shard_paths(a, exp, target, mode, split, variant=""):
    b = str(shard_base(a, exp, target, mode, split, variant))
    return dict(runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), unit=Path(b + "_unit.json"))


_SHA: dict = {}


def file_sha(path):
    k = str(path)
    if k not in _SHA:
        try:
            _SHA[k] = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
        except OSError:
            _SHA[k] = "none"
    return _SHA[k]


def code_sha():
    return file_sha(__file__)


CFG_UNIT_KEYS = ("variant", "data_sha")                              # 조각마다 달라도 되는 항목(공통 해시에서 뺀다)


def unit_cfg(a, exp, variant="", data_sha=""):
    """결과에 영향을 주는 설정 요약. 대상·분할·tag·워커는 넣지 않는다(h40 의 unit_cfg 와 같은 원칙)."""
    d = dict(exp=exp, variant=str(variant), grid=[int(v) for v in a.G[exp]], draws_cap=int(a.draws_cap), seeds=list(a.SEEDS), lams=list(LAMS),
             kappa=KAPPA, r=R_PS, cv_k=CV_K, cb_iters=int(a.cb_iters), cb_hi=dict(X.CB_HI), data_sha=str(data_sha))
    if exp in ("wf1", "wf3"):
        d.update(draws=[DRAWS_SMALL, DRAWS_LARGE, DRAW_SPLIT_N], clusters=list(CLUSTER_KS))
    if exp == "wf1":
        d.update(krige=[KRIGE_K, KRIGE_MIN, VARIO_MAX_PTS, VARIO_NLAGS, VARIO_NRANGE], pbest=list(PBEST_CANDS), learners=[LO, HI])
    if exp == "wf2":
        d.update(draws=[WF2_DRAWS], s3_k=S3_K, s6=[S6_START, S6_BATCH, VE_COUNT], aoa_sub=AOA_SUB)
    if exp == "wf3":
        d.update(r1c_min=R1C_MIN)
    if exp == "wf4":
        d.update(draws=[WF4_DRAWS], w_cands=list(W_CANDS), w_min_n=W_MIN_N, diag_n=DIAG_N)
    if exp == "wf6":                                                       # 2차 보강. 분할 목록(1–25)은 조각마다 하나라 넣지 않는다(원칙 같음)
        d.update(draws=[WF6_DRAWS], clusters=list(CLUSTER_KS), krige=[KRIGE_K, KRIGE_MIN, VARIO_MAX_PTS, VARIO_NLAGS, VARIO_NRANGE],
                 pbest=list(PBEST_CANDS), learners=dict(D0=HI, R1=LO, R2=LO, Re=LO), mode=MODE_R)
    if exp == "wf7":
        d.update(draws=[WF7_DRAWS], methods=["P0", "P1", "Pe", "R1", "Re"], lam=LAM_BASE, re_src="y-anchor(E0)", zero_n=True)
    if exp == "wf8":
        d.update(draws=[WF8_DRAWS], methods=["P1", "R1"], lam=LAM_BASE, s4="wf2", dup="reuse_fit_store_all")
    if exp == "wf9":                                                       # 3차 보강(계획서 §7)
        d.update(draws=[WF9_DRAWS], clusters=list(CLUSTER_KS), krige=[KRIGE_K, KRIGE_MIN, VARIO_MAX_PTS, VARIO_NLAGS, VARIO_NRANGE],
                 physics=["P1", "Pk", "Pc"], learners=dict(D0=HI, R1=LO, Re=LO), mode=MODE_R, grid_group="sqrt_tdd_soil|block", decomp=["w", "b"])
    if exp == "wf9x":
        d.update(draws=[WF9X_DRAWS], methods=["P0", "P1", "R1", "R2", "D0", "D1"], learner=LO, zero_n=True, grid_group="sqrt_tdd_soil|block",
                 decomp=["w", "b"])
    if exp == "wf10":
        d.update(draws=[WF10_DRAWS], frac=WF10_FRAC, methods=["P1", "P2", "R1", "R2", "D0", "D1"], learners=dict(D0=[HI, LO], R1=LO, R2=LO, D1=LO),
                 mode=MODE_R, draw_mode="r10+variant[0]", split_rule="block_mean_sqrt_tdd_soil, I = seed_of('wf10-I', target, variant, split)")
    return d


def wf_cfg_hash(cfg, common=False):
    d = {k: v for k, v in cfg.items() if not (common and k in CFG_UNIT_KEYS)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def data_sha(a, exp, target):
    if exp == "wf4" and target in LGD_SPECS:
        d = a.LGD / LGD_SPECS[target][0]
        return f"{file_sha(d / 'fidelity_base_v3.csv')}:{file_sha(d / 'e5_soil_tdd_v3.csv')}"
    return f"{file_sha(a.PROC / 'fidelity_base_v3.csv')}:{file_sha(a.PROC / 'e5_soil_tdd_v3.csv')}:{file_sha(a.PROC / a.subregion_map)}"


def unit_state(a, exp, target, mode, split, variant=""):
    p = shard_paths(a, exp, target, mode, split, variant)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != wf_cfg_hash(unit_cfg(a, exp, variant, data_sha(a, exp, target))):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    return True, str(u.get("status", "ok"))


def write_shard(a, exp, target, mode, split, variant, c, rows, st, stats, elapsed, expected):
    p = shard_paths(a, exp, target, mode, split, variant)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    tmp = p["runs"].with_name(p["runs"].name + f".tmp{os.getpid()}")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, p["runs"])
    save_stores(st if isinstance(st, list) else [st], p["npz"])
    cfg = unit_cfg(a, exp, variant, data_sha(a, exp, target))
    unit = {**stats, **{k: v for k, v in c.meta.items() if not isinstance(v, (dict, list))}}
    unit.update(exp=exp, target=target, mode=mode, parent=c.parent, split=int(split), variant=variant, tag=exp_tag(a, exp), elapsed_s=round(elapsed, 1),
                n_fit_total=int(sum(stats["n_fit"].values())), n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0),
                expected_splits=[int(v) for v in expected], cfg=cfg, cfg_hash=wf_cfg_hash(cfg), cfg_common=wf_cfg_hash(cfg, common=True),
                code_sha=code_sha(), code_sha_h40=H.code_sha(), code_sha_h42=X.code_sha_x(), threads=int(a.threads),
                device=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))      # 완료 표지는 마지막에 쓴다
    return unit


# ================================================================ 작업 단위 목록·실행
def enumerate_units(a):
    """작업 단위 (실험, 대상, 모드, 분할, 변형)과 건너뛴 기록. 기대 분할(split_plan 의 keep)을 함께 돌려준다."""
    units, skipped, expected = [], [], {}
    for exp in a.EXPS:
        if exp in TRANSFER_EXPS:
            for alias, mode in a.T[exp]:
                spec, tgt, _ = resolve4(a, alias)
                if spec is not None and not lgd_available(a, alias):
                    skipped.append(dict(exp=exp, target=alias, mode=mode, split=-1, status="lgd_table_missing"))
                    continue
                HA, D = get_data(a, spec)
                if mode not in H.valid_modes(tgt):
                    raise SystemExit(f"{exp} 대상 {alias} 에 모드 {mode} 는 없다")
                keep, skip, _ = split_plan(D, tgt, a.SPLITS)
                expected[(exp, alias, mode)] = keep
                skipped += [dict(exp=exp, target=alias, mode=mode, split=sp, status=st_, **v) for sp, st_, v in skip]
                variants = list(a.STRAT8) if exp == "wf8" else [""]
                units += [(exp, alias, mode, sp, v) for sp in keep for v in variants]
            continue
        HA, D = get_data(a)
        for t in a.T[exp]:
            if exp == "wf10":                                              # 변형(warm, cold)마다 분할 구조가 다르다
                for v in a.VAR10:
                    keep, skip, _ = wf10_plan(a, D, t, v)
                    expected[(exp, t, MODE_R, v)] = keep
                    skipped += [dict(exp=exp, target=t, mode=MODE_R, split=sp, variant=v, status=st_, **iv) for sp, st_, iv in skip]
                    units += [(exp, t, MODE_R, sp, v) for sp in keep]
                continue
            keep, skip, _ = split_plan(D, t, splits_of(a, exp), split_info_of(a, D, exp, t) if exp in ("wf6", "wf9") else None)
            expected[(exp, t, MODE_R)] = keep
            skipped += [dict(exp=exp, target=t, mode=MODE_R, split=sp, status=st_, **v) for sp, st_, v in skip]
            variants = [""]
            if exp == "wf1" and t in a.X34:
                variants.append("x34")
            if exp == "wf2":
                variants = list(a.STRAT)
            units += [(exp, t, MODE_R, sp, v) for sp in keep for v in variants]
    return units, skipped, expected


def cost_hint(a, u):
    """실행 순서용 대략의 비용(긴 단위 먼저). 값은 순서에만 쓰고 결과에 영향이 없다."""
    exp, t, m, sp, v = u
    try:
        spec, tgt, _ = resolve4(a, t) if exp in TRANSFER_EXPS else (None, t, False)
        _, D = get_data(a, spec)
        nA = split_info_of(a, D, exp, tgt)[sp]["n_A"]
    except Exception:                                                     # noqa: BLE001
        nA = 1000
    w = dict(wf1=4.0 if v != "x34" else 1.5, wf2=0.6 if v != "S6" else 1.2, wf3=2.0, wf4=6.0, wf6=3.0, wf7=1.5, wf8=0.4, wf9=2.5, wf9x=2.0,
             wf10=2.5).get(exp, 1.0)
    return w * (1.0 + nA / 2000.0) * (3.0 if exp in ("wf4", "wf7", "wf9x") else 1.0)


def run_unit(a, exp, target, mode, split, variant="", dry=False, expected=None):
    t0 = time.time()
    if exp in TRANSFER_EXPS:
        c = build_tctx(a, target, mode, split)
        if exp == "wf9x":
            U = T9Unit(a, c, target, dry)
        else:
            U = T7Unit(a, c, target, dry) if exp == "wf7" else (T8Unit(a, c, target, variant, dry) if exp == "wf8" else TUnit(a, c, target, dry))
    elif exp == "wf10":
        c = build_rctx10(a, target, split, variant)
        U = R10Unit(a, c, exp, variant, dry)
    else:
        info = split_info_of(a, get_data(a)[1], exp, target)[split] if exp in ("wf6", "wf9") else None
        c = build_rctx(a, target, split, info)
        U = R9Unit(a, c, exp, variant, dry) if exp == "wf9" else RUnit(a, c, exp, variant, dry)
    if not dry:
        shard_paths(a, exp, target, mode, split, variant)["unit"].unlink(missing_ok=True)
    U.run()
    rows, st, stats = U.finish()
    if dry:
        return dict(exp=exp, target=target, mode=mode, split=int(split), variant=variant, n_A=len(c.yA), n_eval=len(c.yB), n_rows=stats["n_rows"],
                    fit_total=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
                    n_krige=stats["n_krige"], est_krige_s=stats["est_krige_s"], n_kmeans=stats["n_kmeans"], est_kmeans_s=stats["est_kmeans_s"],
                    _detail=stats["n_fit_detail"], _rows=stats["rows_detail"], _est=stats["est_detail"])
    if expected is None:
        expected = [int(split)]
    return write_shard(a, exp, target, mode, split, variant, c, rows, st, stats, time.time() - t0, expected)


_WA = None
_WEXP = None


def core_blocks(a):
    """워커별 코어 묶음(스레드 수만큼). CatBoost 는 thread_count 밖에서도 여러 스레드를 돌리므로 Rescale 노드에서는 워커를 서로 겹치지 않는
    코어 묶음에 고정해 과다 구독을 막는다. 허용 표지가 없거나, --no-pin-cores 이거나, 코어가 모자라면 None."""
    if a.no_pin_cores or not a.PERMIT or not hasattr(os, "sched_getaffinity") or int(a.workers) <= 0:
        return None
    avail = sorted(os.sched_getaffinity(0))
    T, Wn = int(a.threads), int(a.workers)
    if Wn * T > len(avail):
        return None
    return [avail[i * T:(i + 1) * T] for i in range(Wn)]


def _worker_init(argv, expected_items, core_queue=None):
    global _WA, _WEXP
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if core_queue is not None:
        try:
            os.sched_setaffinity(0, set(core_queue.get(timeout=30)))
        except Exception:                                                 # noqa: BLE001  고정에 실패하면 고정 없이 돈다
            pass
    _WA = parse_args(argv)
    _WEXP = {tuple(k): v for k, v in expected_items}


def _worker_run(exp, target, mode, split, variant):
    t0 = time.time()
    u = run_unit(_WA, exp, target, mode, split, variant, expected=_WEXP.get((exp, target, mode, variant), _WEXP.get((exp, target, mode))))
    return {k: u.get(k) for k in ("exp", "target", "mode", "split", "variant", "n_A", "n_eval", "n_fit_total", "n_rows", "elapsed_s", "status")} | dict(
        wall_s=round(time.time() - t0, 1), n_fail=int(sum((u.get("fail") or {}).values())) + int(u.get("n_nonfinite_keys", 0)))


def unit_name(u):
    return f"{u[0]}|{u[1]}|{u[2]}|s{u[3]}" + (f"|{u[4]}" if u[4] else "")


def execute(a, units, expected, argv):
    """작업 단위를 실행한다(--workers 0 은 이 프로세스에서 차례로, 그 밖은 spawn 프로세스 풀). 반환 (완료 목록, 실패 목록)."""
    t0 = time.time()
    done, failed = [], []
    exp_items = [(list(k), v) for k, v in expected.items()]

    def log(r):
        done.append(r)
        print(f"  [{r['exp']}|{r['target']}|{r['mode']}|s{r['split']}{'|' + r['variant'] if r['variant'] else ''}] 적합 {r['n_fit_total']} · "
              f"행 {r['n_rows']} · A {r['n_A']} · 채점 {r['n_eval']} · {r['elapsed_s']}s · 상태 {r['status']}(실패 {r['n_fail']}) · "
              f"완료 {len(done)}/{len(units)} · 누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {unit_name(u)}: {repr(e)[:300]}", flush=True)

    if not units:
        return done, failed
    if int(a.workers) <= 0:
        _worker_init(argv, exp_items)
        for u in units:
            try:
                log(_worker_run(*u))
            except Exception as e:                                       # noqa: BLE001
                fail(u, e)
        return done, failed
    ctx = multiprocessing.get_context("spawn")
    blocks = core_blocks(a)
    print(f"[pool] 워커 {a.workers} · 스레드 {a.threads} · 코어 고정 {'예(워커마다 ' + str(a.threads) + '코어)' if blocks else '아니오'}", flush=True)
    remaining, attempt = list(units), 0
    while remaining:
        broken = []
        q = None
        if blocks:
            q = ctx.Queue()
            for b_ in blocks:
                q.put(b_)
        with ProcessPoolExecutor(max_workers=int(a.workers), mp_context=ctx, initializer=_worker_init, initargs=(argv, exp_items, q)) as ex:
            futs = {ex.submit(_worker_run, *u): u for u in remaining}
            for f in as_completed(futs):
                try:
                    log(f.result())
                except BrokenProcessPool:
                    broken.append(futs[f])
                except Exception as e:                                   # noqa: BLE001
                    fail(futs[f], e)
        if not broken:
            break
        attempt += 1
        if attempt > int(a.pool_retries):
            for u in broken:
                fail(u, RuntimeError(f"프로세스 풀이 {attempt}회 깨졌다"))
            break
        remaining = broken
        print(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다({attempt}/{a.pool_retries})", flush=True)
    return done, failed


# ================================================================ 집계: 조각·저장소·곡선
def find_shards(a, exp):
    tag = exp_tag(a, exp)
    out = []
    if not a.SHARDS.exists():
        return out
    for p in sorted(a.SHARDS.glob(f"{tag}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if len(parts) not in (5, 6) or parts[0] != tag:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), exp=exp, target=parts[2], mode=parts[3],
                            split=int(parts[4][1:]), variant=parts[5] if len(parts) == 6 else ""))
    return out


def read_runs(shards):
    frames = []
    for s_ in shards:
        if s_["runs"].stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, variant=str, cv_flag=str, placement=str, target=str,
                                                   mode=str, method=str, learner=str), keep_default_na=False, na_values=["", "nan", "NaN"])
            if len(f):
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "variant", "cv_flag"):
        runs[c_] = runs[c_].fillna("").astype(str) if c_ in runs else ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int)
    return runs


def check_cfg(a, exp, units):
    hs = Counter(str(u.get("cfg_common", "legacy")) for u in units)
    cs = Counter(str(u.get("code_sha", "legacy")) for u in units)
    if len(cs) > 1:
        print(f"  [warn] {exp}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
    if len(hs) > 1:
        msg = f"{exp}: 조각의 공통 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
        if not a.allow_mixed_cfg:
            raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
        print("  [warn] " + msg, flush=True)
    return dict(cfg_common=dict(hs), code_sha=dict(cs), n=len(units))


def units_for(units, nm):
    """저장소 이름 nm 에 해당하는 조각 unit. 3차 보강 조각은 unit 의 store_names(총, 격자 안·사이, W·I 저장소 이름)로 찾고,
    그 밖은 '<대상>|<모드>' 로 찾는다(기존과 같다)."""
    return [u for u in units if nm in (u.get("store_names") or [f"{u['target']}|{u['mode']}"])]


def is_point_only(exp, name):
    """점 추정만 내는 대상. wf4 는 LG 의 러시아 C·그린란드와 LGD 의 NAtlantic~lic, wf7·wf8 은 LG 와 같이 러시아 C·그린란드다."""
    t = name.split("|")[0]
    if exp == "wf4":
        return bool((t in LGD_SPECS and LGD_SPECS[t][2]) or t in H.MAIN_POINT)
    return bool(exp in ("wf7", "wf8", "wf9x") and t in H.MAIN_POINT)


def make_tm(name, by_split, units, nboot, point_only):
    """h40.TMx. 분할 구조(dup_of, valid)와 기대 분할은 조각의 unit.json 에서 읽는다(LGD 표의 자료를 다시 읽지 않는다)."""
    info, exp_sp = {}, set()
    for u in units:
        info[int(u["split"])] = dict(dup_of=int(u.get("dup_of", -1)), valid=bool(u.get("valid", True)))
        exp_sp |= {int(v) for v in u.get("expected_splits", [])}
    tm = H.TMx(name, by_split, info, nboot)
    tm.point_only = bool(point_only)
    tm.has_ci = len(tm.by_valid) > 0 and not tm.point_only
    tm.nboot = int(nboot) if tm.has_ci else 0
    tm.expected_splits = sorted(exp_sp) if exp_sp else sorted(tm.used)
    return tm


def _curve_task(exp, name, tm, runs_sub, delta_eq):
    warnings.filterwarnings("ignore")
    try:
        cur = H.build_curve({name: tm}, runs_sub) if len(runs_sub) else pd.DataFrame()
    finally:
        H4.boot_weights.cache_clear()
    if len(cur):
        cur.insert(0, "exp", exp)
        for kind in ("p0", "p1"):
            cur[f"verdict4_{kind}"] = [X.verdict4(r_[f"d_{kind}_lo"], r_[f"d_{kind}_hi"], r_[f"d_{kind}_beq_lo"], r_[f"d_{kind}_beq_hi"], delta_eq)
                                       for r_ in cur.to_dict("records")]
    return cur


def build_curves(a, tasks):
    """tasks = [(exp, name, tm, runs_sub)]. --workers > 1 이면 프로세스 풀로 나눈다(값은 프로세스 수와 무관하다)."""
    out = []
    if int(a.workers) <= 1 or len(tasks) <= 1:
        for exp, nm, tm, rs in tasks:
            out.append(_curve_task(exp, nm, tm, rs, a.delta_eq))
        return out
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(int(a.workers), len(tasks)), mp_context=ctx) as ex:
        futs = {ex.submit(_curve_task, exp, nm, tm, rs, a.delta_eq): (exp, nm) for exp, nm, tm, rs in tasks}
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:                                       # noqa: BLE001  워커 실패: 주 프로세스에서 다시 계산
                exp, nm = futs[f]
                print(f"  [warn] 곡선 워커 실패 {exp}|{nm}: {repr(e)[:160]}. 주 프로세스에서 다시 계산한다", flush=True)
                t = next(t_ for t_ in tasks if t_[0] == exp and t_[1] == nm)
                out.append(_curve_task(*t, a.delta_eq))
    return out


# ================================================================ 집계: 가설 판정
def gk(method, n, learner=LO, lam=None, placement="cell"):
    """곡선 키 (method, learner, alpha, placement, n, lam). 물리식은 learner none, lam 0. P0 은 n = 0 의 한 키."""
    if method == "P0":
        return H.P0_GRP
    if learner == "none":
        return (method, "none", "1", placement, int(n), 0.0)
    return (method, learner, "1", placement, int(n), float(LAM_BASE if lam is None else lam))


def nlab(n):
    return "전량" if int(n) == -1 else f"{int(n)}"


def ns_of(tm, method, learner=None, placement="cell"):
    out = set()
    for gd in tm.idx.values():
        for g in gd:
            if g[0] == method and g[3] == placement and (learner is None or g[1] == learner):
                out.add(int(g[4]))
    return sorted(out, key=lambda v: (v == -1, v))


def _v(r):
    return X._v(r)


def _counts(res):
    c_ = Counter(_v(r) for _, r in res)
    return ", ".join(f"{k} {v}" for k, v in sorted(c_.items()))


def choice_freq(runs, method, by=("target", "n")):
    """선택형 방법(Pbest, Pc, W, 교차검증 λ)의 선택 분포. runs 의 alpha_sel 을 센다(seed 0 행만)."""
    if not len(runs):
        return {}
    q = runs[(runs.method == method) & (runs.seed.isin([-1, 0]))]
    if method in ("R1", "R2", "Re", "RK", "R1@x34", "R2@x34"):
        q = q[np.isclose(q.lam.astype(float), LAM_CV)]
    out = {}
    for key, g in q.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        out["|".join(str(v) for v in key)] = dict(Counter(g.alpha_sel.astype(str)))
    return out


def tests_wf1(a, tms, runs):
    T = X.TestBook(a, tms, None, None)
    for nm in sorted(tms):
        tm = tms[nm]
        main = nm == "Alaska|r"
        role = "주" if main else "서술"
        ns_all = ns_of(tm, "Pbest", "none")
        # ---------------- WF1-a: n ≥ 1,000 에서 R1(교차검증 λ) 또는 R2(교차검증 λ)가 Pbest 보다 우세인 n 이 있다
        ns = [n for n in ns_all if n == -1 or n >= 1000]
        res = []
        for n in ns:
            for m in ("R1", "R2"):
                s_ = X.region_stats(tm, gk(m, n, LO, LAM_CV), gk("Pbest", n, "none"))
                res.append((f"{m} n={nlab(n)}", T.single("WF1-a", "WF1-a", f"{m}(λ cv)-Pbest|n{nlab(n)}", nm, s_, primary=main, role=role, n=n,
                                                       method=m, lam=LAM_CV)))
            for m in ("R1", "R2"):
                for lam in LAMS:
                    T.single("WF1-a", "WF1-a", f"{m}(λ {lam})-Pbest|n{nlab(n)}", nm, X.region_stats(tm, gk(m, n, LO, lam), gk("Pbest", n, "none")),
                             role="보조", n=n, method=m, lam=lam)
        win = [k for k, r in res if _v(r) == "우세"]
        kv, km, _ = X.count_valid(res)
        if not res:
            txt = "판정 불가(행 없음: n ≥ 1,000 인 라벨 수가 없다)"
        elif win:
            txt = f"지지: 최선 물리 보정식보다 우세인 n 이 있다({', '.join(win)})"
        elif kv < km:
            txt = X.na_text(res)
        else:
            txt = "기각: n ≥ 1,000 에서 R1·R2(교차검증 λ)가 최선 물리 보정식보다 우세인 n 이 없다"
        sel = choice_freq(runs[runs.target == nm.split("|")[0]], "Pbest", ("n",)) if len(runs) else {}
        T.verdict("WF1-a", "WF1-a", txt, X._fmt(res) + f" | Pbest 선택 {json.dumps(sel, ensure_ascii=False)}", role=role, used=res, target=nm)
        # ---------------- WF1-b: n ≥ 500 에서 x34 의 R1 이 x25 의 R1 보다 우세(교차검증 λ)
        ns_b = [n for n in ns_of(tm, "R1@x34", LO) if n == -1 or n >= 500]
        if ns_b or main:
            res = []
            for n in ns_b:
                s_ = X.region_stats(tm, gk("R1@x34", n, LO, LAM_CV), gk("R1", n, LO, LAM_CV))
                res.append((f"n={nlab(n)}", T.single("WF1-b", "WF1-b", f"R1@x34-R1|n{nlab(n)}|λ cv", nm, s_, primary=main, role=role, n=n, lam=LAM_CV)))
                for lam in LAMS:
                    T.single("WF1-b", "WF1-b", f"R1@x34-R1|n{nlab(n)}|λ {lam}", nm, X.region_stats(tm, gk("R1@x34", n, LO, lam), gk("R1", n, LO, lam)),
                             role="보조", n=n, lam=lam)
                for m, lam in (("D0", 1.0), ("D1", 1.0), ("R2", LAM_CV)):
                    T.single("WF1-b", "WF1-b", f"{m}@x34-{m}|n{nlab(n)}", nm, X.region_stats(tm, gk(f"{m}@x34", n, LO, lam), gk(m, n, LO, lam)),
                             role="보조", n=n, lam=lam)
            vs = [_v(r) for _, r in res]
            kv, km, _ = X.count_valid(res)
            up = [k for (k, _), v in zip(res, vs) if v == "우세"]
            down = [k for (k, _), v in zip(res, vs) if v == "열세"]
            if not res:
                txt = "판정 불가(x34 행 없음)"
            elif down:
                txt = f"기각: x34 의 R1 이 열세인 n 이 있다({', '.join(down)})"
            elif kv < km:
                txt = X.na_text(res)
            elif len(up) == len(res):
                txt = "지지: n ≥ 500 의 모든 n 에서 x34 의 R1 이 우세(SAR 정보의 가치)"
            elif up:
                txt = f"부분 지지: 우세인 n = {', '.join(up)}"
            else:
                txt = "기각: 우세인 n 이 없다(SAR 열의 추가 가치를 확인하지 못했다)"
            T.verdict("WF1-b", "WF1-b", txt, X._fmt(res), role=role, used=res, target=nm)
        # ---------------- WF1-c: 물리 기반 레시피(R1·R2·D1)가 Pbest 대비 열세인 n 의 수가 D0 보다 적다
        cnt = {}
        for m, lr, lam in (("R1", LO, LAM_CV), ("R2", LO, LAM_CV), ("D1", LO, 1.0), ("D0", LO, 1.0)):
            res_m = [(f"{m} n={nlab(n)}", T.single("WF1-c", "WF1-c", f"{m}-Pbest|n{nlab(n)}", nm, X.region_stats(tm, gk(m, n, lr, lam),
                                                                                                                gk("Pbest", n, "none")),
                                                  role=role, n=n, method=m, lam=lam)) for n in ns_all]
            kv, km, miss = X.count_valid(res_m)
            cnt[m] = (sum(_v(r) == "열세" for _, r in res_m), kv, km, res_m)
        for m in ("R1", "R2"):                                             # 보조: 고정 λ 0.25 의 열세 수
            res_m = [(f"{m}(0.25) n={nlab(n)}", T.single("WF1-c", "WF1-c", f"{m}(λ 0.25)-Pbest|n{nlab(n)}", nm,
                                                         X.region_stats(tm, gk(m, n, LO, LAM_BASE), gk("Pbest", n, "none")), role="보조", n=n,
                                                         method=m, lam=LAM_BASE)) for n in ns_all]
            cnt[f"{m}(0.25)"] = (sum(_v(r) == "열세" for _, r in res_m), *X.count_valid(res_m)[:2], res_m)
        d0 = cnt["D0"]
        incomplete = [m for m in ("R1", "R2", "D1", "D0") if cnt[m][1] < cnt[m][2]]
        fewer = [m for m in ("R1", "R2", "D1") if cnt[m][0] < d0[0]]
        stat = "; ".join(f"{m} 열세 {v[0]}/{v[1]}(등록 {v[2]})" for m, v in cnt.items())
        if not ns_all:
            txt = "판정 불가(행 없음)"
        elif incomplete:
            txt = f"판정 불가(판정할 수 없는 대비가 있다: {', '.join(incomplete)})"
        elif len(fewer) == 3:
            txt = "지지: R1·R2·D1 모두 열세인 n 의 수가 D0 보다 적다"
        elif fewer:
            txt = f"부분 지지: 열세인 n 의 수가 D0 보다 적은 레시피 = {', '.join(fewer)}"
        else:
            txt = "기각: 열세인 n 의 수가 D0 보다 적은 물리 기반 레시피가 없다"
        T.verdict("WF1-c", "WF1-c", txt, stat, role=role, used=[(k, r) for m in ("R1", "R2", "D1", "D0") for k, r in cnt[m][3]], target=nm)
    return T.frame()


def tests_wf2(a, tms, runs):
    T = X.TestBook(a, tms, None, None)
    strat = [s for s in a.STRAT if s != "S1"]
    main_n = (50, 100, 200)
    for nm in sorted(tms):
        tm = tms[nm]
        main = nm == "Alaska|r"
        role = "주" if main else "서술"
        ns = ns_of(tm, "R1", LO, "S1")
        res, by_s = [], {}
        for s in strat:
            for n in ns:
                r = T.single("WF2-a", "WF2-a", f"{s}-S1|R1(0.25)|n{nlab(n)}", nm,
                             X.region_stats(tm, gk("R1", n, LO, LAM_BASE, s), gk("R1", n, LO, LAM_BASE, "S1")),
                             primary=bool(main and n in main_n), role=role if n in main_n else "서술", n=n, strategy=s)
                if n in main_n:
                    res.append((f"{s} n={nlab(n)}", r)); by_s.setdefault(s, []).append(r)
                for m, lam in (("P1", None), ("D1", 1.0)):
                    gA = gk("P1", n, "none", placement=s) if m == "P1" else gk("D1", n, LO, 1.0, s)
                    gB = gk("P1", n, "none", placement="S1") if m == "P1" else gk("D1", n, LO, 1.0, "S1")
                    T.single("WF2-a", "WF2-a", f"{s}-S1|{m}|n{nlab(n)}", nm, X.region_stats(tm, gA, gB), role="보조", n=n, strategy=s, method=m)
        win = [k for k, r in res if _v(r) == "우세"]
        all3 = [s for s, rs in by_s.items() if len(rs) == len(main_n) and all(_v(r) == "우세" for r in rs)]
        kv, km, _ = X.count_valid(res)
        if not res:
            txt = "판정 불가(행 없음)"
        elif win:
            txt = f"지지: S1 보다 우세인 (전략, n) = {', '.join(win)}" + (f". 세 n 모두 우세인 전략 = {', '.join(all3)}" if all3 else "")
        elif kv < km:
            txt = X.na_text(res)
        else:
            txt = "기각: n {50, 100, 200} 의 R1 에서 S1 보다 우세인 전략이 없다"
        T.verdict("WF2-a", "WF2-a", txt, X._fmt(res), role=role, used=res, target=nm)
    # ---------------- WF2-b: 알래스카 n = 100 의 R1 점 추정으로 고른 전략이 레나·캐나다에서 S1 보다 열세가 아니다
    tmA = tms.get("Alaska|r")
    if tmA is not None and strat:
        sc = {s: X.grp_rmse(tmA, gk("R1", 100, LO, LAM_BASE, s)) for s in strat}
        sc = {s: v for s, v in sc.items() if np.isfinite(v)}
        if sc:
            best = min(sc, key=lambda s: (sc[s], s))
            res = []
            for nm in ("Lena|r", "Canada|r"):
                tm = tms.get(nm)
                s_ = X.region_stats(tm, gk("R1", 100, LO, LAM_BASE, best), gk("R1", 100, LO, LAM_BASE, "S1")) if tm is not None else None
                res.append((nm, T.single("WF2-b", "WF2-b", f"{best}-S1|R1(0.25)|n100", nm, s_, primary=True, n=100, strategy=best)))
                if tm is not None:
                    for n in [v for v in ns_of(tm, "R1", LO, best) if v != 100]:
                        T.single("WF2-b", "WF2-b", f"{best}-S1|R1(0.25)|n{nlab(n)}", nm,
                                 X.region_stats(tm, gk("R1", n, LO, LAM_BASE, best), gk("R1", n, LO, LAM_BASE, "S1")), role="보조", n=n, strategy=best)
            vs = [_v(r) for _, r in res]
            kv, km, _ = X.count_valid(res)
            if "열세" in vs:
                txt = f"기각: {best} 가 S1 보다 열세인 대상이 있다({', '.join(k for (k, _), v in zip(res, vs) if v == '열세')})"
            elif kv < km:
                txt = X.na_text(res)
            else:
                txt = f"지지: 알래스카에서 고른 {best} 는 레나·캐나다에서 S1 보다 열세가 아니다"
            stat = "알래스카 n = 100 R1(0.25) RMSE: " + ", ".join(f"{s} {v:.2f}" for s, v in sorted(sc.items(), key=lambda kv_: kv_[1])) + " | " + X._fmt(res)
            T.verdict("WF2-b", "WF2-b", txt, stat, used=res, strategy=best)
    return T.frame()


def tests_wf3(a, tms, runs):
    T = X.TestBook(a, tms, None, None)
    res = []
    for nm in ("Alaska|r", "Lena|r"):
        tm = tms.get(nm)
        if tm is None:
            continue
        for n in [v for v in ns_of(tm, "Pc", "none") if v == -1 or v >= 500]:
            res.append((f"{nm.split('|')[0]} n={nlab(n)}", T.single("WF3-a", "WF3-a", f"Pc(k cv)-P1|n{nlab(n)}", nm,
                                                                    X.region_stats(tm, gk("Pc", n, "none"), gk("P1", n, "none")), primary=True, n=n)))
            for k in CLUSTER_KS:
                T.single("WF3-a", "WF3-a", f"Pc@c{k}-P1|n{nlab(n)}", nm, X.region_stats(tm, gk(f"Pc@c{k}", n, "none"), gk("P1", n, "none")),
                         role="보조", n=n)
    vs = [_v(r) for _, r in res]
    kv, km, _ = X.count_valid(res)
    up = [k for (k, _), v in zip(res, vs) if v == "우세"]
    down = [k for (k, _), v in zip(res, vs) if v == "열세"]
    if not res:
        txt = "판정 불가(행 없음)"
    elif down:
        txt = f"기각: Pc 가 P1 보다 열세인 대비가 있다({', '.join(down)})"
    elif kv < km:
        txt = X.na_text(res)
    elif len(up) == len(res):
        txt = "지지: n ≥ 500 의 모든 대비에서 Pc(교차검증 k)가 P1 보다 우세"
    elif up:
        txt = f"부분 지지: 우세인 대비 = {', '.join(up)}"
    else:
        txt = "기각: n ≥ 500 에서 Pc 가 P1 보다 우세인 대비가 없다"
    T.verdict("WF3-a", "WF3-a", txt, X._fmt(res) + f" | k 선택 {json.dumps(choice_freq(runs, 'Pc'), ensure_ascii=False)}", used=res)
    # ---------------- WF3-b(방향 중립): 군집별 잔차·군집 입력 대 전체 잔차
    for nm in ("Alaska|r", "Lena|r"):
        tm = tms.get(nm)
        if tm is None:
            continue
        out = {}
        for kind in ("R1c", "R1i"):
            lst = []
            for n in ns_of(tm, "R1", LO):
                for k in CLUSTER_KS:
                    for lam in (LAM_BASE, 1.0):
                        lst.append((f"{kind}@c{k} n={nlab(n)} λ {lam}",
                                    T.single("WF3-b", "WF3-b", f"{kind}@c{k}-R1|n{nlab(n)}|λ {lam}", nm,
                                             X.region_stats(tm, gk(f"{kind}@c{k}", n, LO, lam), gk("R1", n, LO, lam)), role="서술", n=n, lam=lam,
                                             method=f"{kind}@c{k}")))
            out[kind] = lst
        T.verdict("WF3-b", "WF3-b", "서술(방향 중립): " + "; ".join(f"{kind} {_counts(lst)}" for kind, lst in out.items()),
                  "; ".join(X._fmt(lst) for lst in out.values()), role="서술", target=nm, used=[r for lst in out.values() for r in lst])
    return T.frame()


def _spearman_rows(Dm, Gm):
    """행마다 Spearman ρ(순위의 Pearson, 동률 평균 순위)."""
    from scipy.stats import rankdata
    rd, rg = rankdata(Dm, axis=1), rankdata(Gm, axis=1)
    rd = rd - rd.mean(1, keepdims=True); rg = rg - rg.mean(1, keepdims=True)
    den = np.sqrt((rd ** 2).sum(1) * (rg ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rd * rg).sum(1) / den


def ni_p(tms, names, gA, gB, margin):
    """비열등 p: 층화 평균 분포(지역 분포의 비가중 평균, m1_stats.strat)에서 max(P(Δ ≥ 한계)) 를 두 가중에 대해 구한다."""
    dc, db = [], []
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        s_ = X.region_stats(tm, gA, gB)
        if s_ is not None and s_.get("dist") is not None:
            dc.append(s_["dist"]); db.append(s_["dist_beq"])
    if len(dc) < H.MIN_POOL_REGIONS:
        return np.nan
    sc, sb = MS.strat(dc), MS.strat(db)
    return float(max(np.mean(sc >= margin), np.mean(sb >= margin)))


def tests_wf4(a, tms, runs, units):
    T = X.TestBook(a, tms, None, None)
    main4 = [f"{t}|x" for t in MAIN4]
    gW = lambda n: gk("W", n, LO, 0.0)                                     # noqa: E731
    gR = lambda n: gk("R1", n, LO, LAM_BASE)                              # noqa: E731
    # ---------------- WF4-a: W 는 R1(0.25)보다 열세가 아니다(주 4지역 평균, 두 가중, 한계 0.5 cm)
    res = []
    for n in (10, 40, 160, -1):
        mr = T.contrast("WF4-a", "WF4-a", f"W-R1(0.25)|n{nlab(n)}", gW(n), gR(n), names=main4, primary=True, aux3=False, n=n)
        if mr is not None:
            mr["p_ni"] = ni_p(tms, main4, gW(n), gR(n), a.ni_margin)
            mr["ni"] = bool(X.valid_row(mr) and not mr.get("undetermined", False) and mr["ci_hi"] < a.ni_margin and mr["ci_hi_beq"] < a.ni_margin)
        res.append((f"n={nlab(n)}", mr))
    hp = MS.holm([r["p_ni"] if (r is not None and np.isfinite(r.get("p_ni", np.nan))) else np.nan for _, r in res])
    for (_, r), h in zip(res, hp):
        if r is not None:
            r["holm_p_ni"] = float(h)
    kv, km, _ = X.count_valid(res)
    ok_n = [k for k, r in res if r is not None and r.get("ni")]
    if kv < km:
        txt = X.na_text(res)
    elif len(ok_n) == km:
        txt = f"지지: 모든 n 에서 비열등(두 가중 CI 상한 < {a.ni_margin} cm)"
    elif ok_n:
        txt = f"부분 지지: 비열등인 n = {', '.join(ok_n)}"
    else:
        txt = f"기각: 어느 n 에서도 비열등(한계 {a.ni_margin} cm)을 보이지 못했다"
    T.verdict("WF4-a", "WF4-a", txt, "; ".join(f"{k}: Δ {r['delta']:.2f}, CI 상한 {r['ci_hi']:.2f}·{r['ci_hi_beq']:.2f}, p_ni {r.get('p_ni', np.nan):.4f}"
                                               for k, r in res if r is not None), used=res)
    # ---------------- WF4-b: n ≥ 160 에서 P1 보다 우세인 대상 수가 W > R1(0.25)
    per_n = {}
    for n in (160, -1):
        cw = cr = det = 0
        for nm in sorted(tms):
            tm = tms[nm]
            rW = T.single("WF4-b", "WF4-b", f"W-P1|n{nlab(n)}", nm, X.region_stats(tm, gW(n), gk("P1", n, "none")), role="집계", n=n, method="W")
            rR = T.single("WF4-b", "WF4-b", f"R1(0.25)-P1|n{nlab(n)}", nm, X.region_stats(tm, gR(n), gk("P1", n, "none")), role="집계", n=n,
                          method="R1")
            cw += _v(rW) == "우세"; cr += _v(rR) == "우세"; det += bool(X.valid_row(rW) and X.valid_row(rR))
        per_n[n] = (cw, cr, det)
    more = [nlab(n) for n, (cw, cr, _) in per_n.items() if cw > cr]
    none_det = [nlab(n) for n, (_, _, det) in per_n.items() if det == 0]
    stat = "; ".join(f"n={nlab(n)}: W 우세 {cw} · R1 우세 {cr}(두 대비 모두 판정 가능한 대상 {det}/{len(tms)})" for n, (cw, cr, det) in per_n.items())
    if none_det:
        txt = f"판정 불가(판정 가능한 대상이 없는 n: {', '.join(none_det)})"
    else:
        txt = ("지지: n 160·전량 모두 W 의 우세 대상 수가 더 많다" if len(more) == 2 else
               f"부분 지지: W 의 우세 대상 수가 더 많은 n = {', '.join(more)}" if more else "기각: W 의 우세 대상 수가 R1(0.25)보다 많지 않다")
    T.verdict("WF4-b", "WF4-b", txt, stat, point=True, used=[])
    # ---------------- WF4-c(진단): 편향 진단값(라벨 10개) 과 전량 라벨 ML 이득(P0 − W)의 순위상관
    diag = {}
    for u in units:
        nm = f"{u['target']}|{u['mode']}"
        diag.setdefault(nm, []).extend(u.get("diag") or [])
    rows, names = [], []
    for nm in sorted(tms):
        tm = tms[nm]
        dv = diag.get(nm, [])
        r = H.contrast(tm, gW(-1), H.P0_GRP, return_dist=True)
        rR1 = H.contrast(tm, gR(-1), H.P0_GRP, return_dist=False)
        if not dv or r is None or not np.isfinite(r["delta"]):
            continue
        names.append(nm)
        rows.append(dict(name=nm, abs_vals=np.array([v["abs_bias"] for v in dv], float), sgn_vals=np.array([v["bias"] for v in dv], float),
                         gain=-float(r["delta"]), dist=(-np.asarray(r["dist"], float)) if "dist" in r else None,
                         gain_r1=-float(rR1["delta"]) if rR1 is not None else np.nan))
    if len(rows) >= 3:
        B = int(max(a.nboot, 1))
        rng = np.random.RandomState(seed_of("wf4c", len(rows)))
        Dabs = np.zeros((B, len(rows))); Dsgn = np.zeros((B, len(rows))); Gm = np.zeros((B, len(rows)))
        for j, q in enumerate(rows):
            k = len(q["abs_vals"])
            pick = rng.randint(0, k, size=(B, k))
            Dabs[:, j] = q["abs_vals"][pick].mean(1); Dsgn[:, j] = q["sgn_vals"][pick].mean(1)
            Gm[:, j] = q["dist"][:B] if (q["dist"] is not None and len(q["dist"]) >= B) else q["gain"]
        d0 = np.array([q["abs_vals"].mean() for q in rows]); s0 = np.array([q["sgn_vals"].mean() for q in rows])
        g0 = np.array([q["gain"] for q in rows]); g1 = np.array([q["gain_r1"] for q in rows])
        rho = float(_spearman_rows(d0[None], g0[None])[0]); rho_b = _spearman_rows(Dabs, Gm)
        lo, hi = (float(np.nanpercentile(rho_b, 2.5)), float(np.nanpercentile(rho_b, 97.5)))
        rho_s = float(_spearman_rows(s0[None], g0[None])[0]); rb_s = _spearman_rows(Dsgn, Gm)
        ok1 = np.isfinite(g1)
        rho_r1 = float(_spearman_rows(d0[ok1][None], g1[ok1][None])[0]) if ok1.sum() >= 3 else np.nan
        for q, dm, sm in zip(rows, d0, s0):
            T.rows.append(dict(test_id="WF4-c", item="WF4-c", contrast="진단값-이득", scope="region", role="진단", target=q["name"], diag_abs=float(dm),
                               diag_signed=float(sm), n_diag=int(len(q["abs_vals"])), gain_w=q["gain"], gain_r1=q["gain_r1"],
                               ci_flag="" if q["dist"] is not None else "점 추정(분포 없음)"))
        T.rows.append(dict(test_id="WF4-c", item="WF4-c", contrast="Spearman(|편향|, P0 − W)", scope="MEAN", role="진단", n_targets=len(rows), rho=rho,
                           ci_lo=lo, ci_hi=hi, p_one=float(np.mean(rho_b <= 0)), rho_signed=rho_s,
                           ci_lo_signed=float(np.nanpercentile(rb_s, 2.5)), ci_hi_signed=float(np.nanpercentile(rb_s, 97.5)), rho_gain_r1=rho_r1,
                           nboot=B))
        txt = (f"지지: 양의 순위상관(ρ {rho:.2f}, CI [{lo:.2f}, {hi:.2f}])" if lo > 0 else
               f"미지지: ρ {rho:.2f}, CI [{lo:.2f}, {hi:.2f}] 가 0 을 포함하거나 음수다")
        T.verdict("WF4-c", "WF4-c", txt, f"대상 {len(rows)}(진단값과 이득이 모두 있는 대상) · 부호 있는 편향 ρ {rho_s:.2f} · R1(0.25) 이득 ρ {rho_r1:.2f}",
                  used=[], note="진단")
    else:
        T.verdict("WF4-c", "WF4-c", f"판정 불가(대상 {len(rows)} < 3)", "", used=[], note="진단")
    return T.frame()


# ================================================================ 집계: 2차 보강 판정(계획서 §6)
WF6_RECIPES = (("R1", LO, LAM_CV), ("R2", LO, LAM_CV), ("Re", LO, LAM_CV), ("D0", HI, 1.0))     # WF6-a 의 레시피(교차검증 λ, D0 = catboost)


def _rlab(m, lr, lam):
    return f"{m}(λ cv)" if lam == LAM_CV else (f"{m}({lr})" if m.startswith("D") else f"{m}(λ {lam})")


def tests_wf6(a, tms, runs):
    """WF6-a(대상별: R1·R2·Re(교차검증 λ)·D0(catboost) 가운데 하나라도 P1 보다 우세인 n 이 있다. Pbest 대비는 보조),
    WF6-b(세 대상 층화 평균의 R2(교차검증 λ) − P1 이 n ≥ 500 에서 우세)."""
    T = X.TestBook(a, tms, None, None)
    names = [f"{t}|{MODE_R}" for t in a.T["wf6"]]
    for nm in sorted(tms):
        tm = tms[nm]
        ns = ns_of(tm, "P1", "none")
        res, res_pb = [], []
        for n in ns:
            for m, lr, lam in WF6_RECIPES:
                gA = gk(m, n, lr, lam)
                res.append((f"{_rlab(m, lr, lam)} n={nlab(n)}", T.single("WF6-a", "WF6-a", f"{_rlab(m, lr, lam)}-P1|n{nlab(n)}", nm,
                                                                         X.region_stats(tm, gA, gk("P1", n, "none")), primary=True, n=n, method=m,
                                                                         learner=lr, lam=lam)))
                res_pb.append((f"{_rlab(m, lr, lam)} n={nlab(n)}", T.single("WF6-a", "WF6-a(Pbest)", f"{_rlab(m, lr, lam)}-Pbest|n{nlab(n)}", nm,
                                                                            X.region_stats(tm, gA, gk("Pbest", n, "none")), role="보조", n=n,
                                                                            method=m, learner=lr, lam=lam)))
            for m in ("Pk", "Pbest"):                                       # 서술: 물리 보정식 사이의 대비
                T.single("WF6-a", "WF6-a(물리)", f"{m}-P1|n{nlab(n)}", nm, X.region_stats(tm, gk(m, n, "none"), gk("P1", n, "none")), role="서술",
                         n=n, method=m)
            for m, lam in (("R1", LAM_BASE), ("R2", LAM_BASE), ("Re", LAM_BASE)):   # 보조: 고정 λ 0.25
                T.single("WF6-a", "WF6-a(λ 0.25)", f"{m}(λ 0.25)-P1|n{nlab(n)}", nm, X.region_stats(tm, gk(m, n, LO, lam), gk("P1", n, "none")),
                         role="보조", n=n, method=m, lam=lam)
        win = [k for k, r in res if _v(r) == "우세"]
        kv, km, _ = X.count_valid(res)
        if not res:
            txt = "판정 불가(행 없음)"
        elif win:
            txt = f"지지: P1 보다 우세인 (레시피, n) = {', '.join(win)}"
        elif kv < km:
            txt = X.na_text(res)
        else:
            txt = "기각: R1·R2·Re(교차검증 λ)와 D0(catboost) 가운데 P1 보다 우세인 n 이 없다"
        sel = choice_freq(runs[runs.target == nm.split("|")[0]], "Pbest", ("n",)) if len(runs) else {}
        T.verdict("WF6-a", "WF6-a", txt, X._fmt(res), role="주", used=res, target=nm)
        win_pb = [k for k, r in res_pb if _v(r) == "우세"]
        kv, km, _ = X.count_valid(res_pb)
        txt = ("판정 불가(행 없음)" if not res_pb else f"보조 지지: Pbest 보다 우세인 (레시피, n) = {', '.join(win_pb)}" if win_pb else
               X.na_text(res_pb) if kv < km else "보조 기각: Pbest 보다 우세인 (레시피, n) 이 없다")
        T.verdict("WF6-a", "WF6-a(Pbest)", txt, X._fmt(res_pb) + f" | Pbest 선택 {json.dumps(sel, ensure_ascii=False)}", role="보조", used=res_pb,
                  target=nm)
    # ---------------- WF6-b: 세 대상 층화 평균의 R2(교차검증 λ) − P1 이 n ≥ 500 에서 우세
    res = []
    for n in (200, 500, 1000, -1):
        main = n == -1 or n >= 500
        mr = T.contrast("WF6-b", "WF6-b", f"R2(λ cv)-P1|n{nlab(n)}", gk("R2", n, LO, LAM_CV), gk("P1", n, "none"), names=names, primary=main,
                        role="주" if main else "서술", aux3=False, n=n, lam=LAM_CV)
        if main:
            res.append((f"n={nlab(n)}", mr))
        for m, lr, lam in (("R1", LO, LAM_CV), ("Re", LO, LAM_CV), ("D0", HI, 1.0)):   # 서술: 다른 레시피의 층화 평균
            T.contrast("WF6-b", "WF6-b(서술)", f"{_rlab(m, lr, lam)}-P1|n{nlab(n)}", gk(m, n, lr, lam), gk("P1", n, "none"), names=names, primary=False,
                       role="서술", aux3=False, n=n, method=m, lam=lam)
    vs = [_v(r) for _, r in res]
    kv, km, _ = X.count_valid(res)
    up = [k for (k, _), v in zip(res, vs) if v == "우세"]
    down = [k for (k, _), v in zip(res, vs) if v == "열세"]
    if not any(r is not None for _, r in res):
        txt = "판정 불가(행 없음)"
    elif down:
        txt = f"기각: R2(교차검증 λ)가 P1 보다 열세인 n 이 있다({', '.join(down)})"
    elif kv < km:
        txt = X.na_text(res)
    elif len(up) == len(res):
        txt = "지지: n ≥ 500 의 모든 n 에서 R2(교차검증 λ)가 P1 보다 우세(세 대상 층화 평균)"
    elif up:
        txt = f"부분 지지: 우세인 n = {', '.join(up)}"
    else:
        txt = "기각: n ≥ 500 에서 R2(교차검증 λ)가 P1 보다 우세인 n 이 없다"
    T.verdict("WF6-b", "WF6-b", txt, X._fmt(res) + f" | 층화 대상 {', '.join(names)}", used=res)
    return T.frame()


def tests_wf7(a, tms, runs):
    """WF7(방향 중립 서술, 확인적 가설 아님): 주 4지역 층화 평균의 Re − R1(λ 0.25)(WF7-a), Pe − P1(WF7-b)을 n ∈ {0, 10, 40, 160, 전량}에서
    4분 판정으로 보고한다. 주 4지역 밖 대상의 지역 행과 Re − P0(보조)을 함께 둔다."""
    T = X.TestBook(a, tms, None, None)
    main4 = [f"{t}|x" for t in MAIN4]
    items = (("WF7-a", "WF7-a", "Re-R1(0.25)", lambda n: gk("Re", n, LO, LAM_BASE), lambda n: gk("R1", n, LO, LAM_BASE)),
             ("WF7-b", "WF7-b", "Pe-P1", lambda n: gk("Pe", n, "none"), lambda n: gk("P1", n, "none")),
             ("WF7-a", "WF7-a(P0)", "Re(0.25)-P0", lambda n: gk("Re", n, LO, LAM_BASE), lambda n: H.P0_GRP))
    for test, item, lbl, fA, fB in items:
        res = []
        role = "서술" if item in ("WF7-a", "WF7-b") else "보조"
        for n in WF7_TEST_N:
            mr = T.contrast(test, item, f"{lbl}|n{nlab(n)}", fA(n), fB(n), names=main4, primary=False, role=role, aux3=False, n=n)
            res.append((f"n={nlab(n)}", mr))
            for nm in sorted(tms):                                            # 주 4지역 밖 대상의 지역 행
                if nm in main4:
                    continue
                T.single(test, item, f"{lbl}|n{nlab(n)}", nm, X.region_stats(tms[nm], fA(n), fB(n)), role=f"{role}(지역)", n=n)
        pre = "서술(방향 중립, 확인적 가설 아님)" if role == "서술" else f"보조 서술({lbl}, 방향 중립)"
        txt = f"{pre}: " + ", ".join(f"{k} {_v(r)}" for k, r in res)
        T.verdict(test, item, txt, X._fmt(res), role=role, used=res, note="방향 중립")
    return T.frame()


def tests_wf8(a, tms, runs):
    """WF8-a(레나·캐나다의 n = 40 R1 에서 S4 − S1 이 우세인 지역이 하나 이상), WF8-b(다섯 대상의 n = 10 R1 에서 S2·S4 가 S1 보다 열세인
    지역이 없다). 그 밖의 (전략, n, 레시피) 대비는 서술 행이다."""
    T = X.TestBook(a, tms, None, None)
    regions = [f"{t}|{m}" for t, m in a.T["wf8"]]
    strat = [s for s in a.STRAT8 if s != "S1"]
    for nm in sorted(tms):
        tm = tms[nm]
        for s in strat:
            for n in ns_of(tm, "R1", LO, "S1"):
                for m, gA, gB in (("R1", gk("R1", n, LO, LAM_BASE, s), gk("R1", n, LO, LAM_BASE, "S1")),
                                  ("P1", gk("P1", n, "none", placement=s), gk("P1", n, "none", placement="S1"))):
                    T.single("WF8", "WF8(서술)", f"{s}-S1|{m}{'(0.25)' if m == 'R1' else ''}|n{nlab(n)}", nm, X.region_stats(tm, gA, gB), role="서술",
                             n=n, strategy=s, method=m)
    # ---------------- WF8-a
    res = []
    for nm in ("Lena|x", "Canada|x"):
        tm = tms.get(nm)
        s_ = X.region_stats(tm, gk("R1", 40, LO, LAM_BASE, "S4"), gk("R1", 40, LO, LAM_BASE, "S1")) if tm is not None else None
        res.append((nm.split("|")[0], T.single("WF8-a", "WF8-a", "S4-S1|R1(0.25)|n40", nm, s_, primary=True, n=40, strategy="S4")))
    win = [k for k, r in res if _v(r) == "우세"]
    kv, km, _ = X.count_valid(res)
    if win:
        txt = f"지지: n = 40 의 R1 에서 S4 가 S1 보다 우세인 지역 = {', '.join(win)}"
    elif kv < km:
        txt = X.na_text(res)
    else:
        txt = "기각: 레나·캐나다 모두 n = 40 의 R1 에서 S4 가 S1 보다 우세가 아니다"
    T.verdict("WF8-a", "WF8-a", txt, X._fmt(res), used=res)
    # ---------------- WF8-b(안전성)
    res = []
    for nm in regions:
        tm = tms.get(nm)
        for s in ("S2", "S4"):
            s_ = X.region_stats(tm, gk("R1", 10, LO, LAM_BASE, s), gk("R1", 10, LO, LAM_BASE, "S1")) if tm is not None else None
            res.append((f"{nm.split('|')[0]} {s}", T.single("WF8-b", "WF8-b", f"{s}-S1|R1(0.25)|n10", nm, s_, primary=True, n=10, strategy=s)))
    down = [k for k, r in res if _v(r) == "열세"]
    kv, km, _ = X.count_valid(res)
    if down:
        txt = f"기각: n = 10 의 R1 에서 S1 보다 열세인 (지역, 전략) = {', '.join(down)}"
    elif kv < km:
        txt = X.na_text(res)
    else:
        txt = "지지: n = 10 의 R1 에서 S2·S4 가 S1 보다 열세인 지역이 없다"
    T.verdict("WF8-b", "WF8-b", txt, X._fmt(res), used=res)
    return T.frame()


WF_HYP = ("WF1-a", "WF1-b", "WF1-c", "WF2-a", "WF2-b", "WF3-a", "WF4-a", "WF4-b", "WF4-c", "WF6-a", "WF6-b", "WF8-a", "WF8-b",
          "WF9-a", "WF9-c", "WF10-a", "WF10-b", "WF10-c")


# ---------------------------------------------------------------- 3차 보강 판정(계획서 §7)
def _rule3(res, kind="우세"):
    """WF9-a·c, WF10-a·b 의 판정 문구. 모든 대비가 kind 면 지지, 반대 판정 없이 일부면 부분 지지, 반대 판정이 있거나 kind 가 없으면 기각.
    판정할 수 없는 대비가 있고 kind 가 없으면 판정 불가."""
    other = "열세" if kind == "우세" else "우세"
    if not any(r is not None for _, r in res):
        return "판정 불가(행 없음)"
    vs = [_v(r) for _, r in res]
    kv, km, _ = X.count_valid(res)
    hit = [k for (k, _), v in zip(res, vs) if v == kind]
    bad = [k for (k, _), v in zip(res, vs) if v == other]
    if bad:
        return f"기각: {other}인 대비가 있다({', '.join(bad)})"
    if not hit:
        return X.na_text(res) if kv < km else f"기각: {kind}인 대비가 없다"
    if len(hit) == len(res):
        return f"지지: 모든 대비가 {kind}({', '.join(hit)})"
    return f"부분 지지: {kind}인 대비 = {', '.join(hit)}" + (f"; {X.na_text(res)}" if kv < km else "")


def _ni_text(res, margin):
    """WF10-c 비열등(두 가중 95 % CI 상한 < margin) 문구. 모두면 지지, 일부면 부분 지지, 없으면 기각."""
    if not any(r is not None for _, r in res):
        return "판정 불가(행 없음)"
    ok, no, na = [], [], []
    for k, r in res:
        if not X.valid_row(r):
            na.append(k)
            continue
        hi = np.nanmax([float(r.get("ci_hi", np.nan)), float(r.get("ci_hi_beq", np.nan))])
        (ok if np.isfinite(hi) and hi < margin else no).append(k)
    if len(ok) == len(res):
        return f"지지: 모든 n 에서 두 가중 CI 상한 < {margin:g} cm({', '.join(ok)})"
    if ok:
        return f"부분 지지: 비열등인 n = {', '.join(ok)}" + (f"; 판정 불가 {', '.join(na)}" if na else "")
    if na and not no:
        return X.na_text(res)
    return "기각: 비열등인 n 이 없다"


def _plab(m, lr, lam):
    return m if lr == "none" else _rlab(m, lr, lam)


def tests_wf9(a, tms, runs):
    """WF9-a(세 대상 층화 평균의 격자 안 RMSE: R1(교차검증 λ) − P1, n ∈ {500, 1,000, 전량}, 모두 우세 = 지지), WF9-b(대상별 격자 안 대비는 보조,
    격자 사이 대비와 총 RMSE 대비는 서술)."""
    T = X.TestBook(a, tms, None, None)
    tg = list(a.T["wf9"])
    wn = [f"{t}~w|{MODE_R}" for t in tg]
    res = []
    for n in a.G["wf9"]:
        main = n == -1 or n >= 500
        mr = T.contrast("WF9-a", "WF9-a", f"R1(λ cv)-P1[격자 안]|n{nlab(n)}", gk("R1", n, LO, LAM_CV), gk("P1", n, "none"), names=wn, primary=main,
                        role="주" if main else "서술", aux3=False, n=n, lam=LAM_CV, part="w")
        if main:
            res.append((f"n={nlab(n)}", mr))
    T.verdict("WF9-a", "WF9-a", _rule3(res, "우세"), X._fmt(res) + f" | 층화 대상 {', '.join(wn)}", used=res)
    recipes = (("R1", LO, LAM_CV), ("R1", LO, LAM_BASE), ("Re", LO, LAM_CV), ("D0", HI, 1.0), ("Pk", "none", None), ("Pc", "none", None))
    for t in tg:
        for part, role, word in (("w", "보조", "격자 안"), ("b", "서술", "격자 사이"), ("", "서술", "총")):
            nm = f"{t}~{part}|{MODE_R}" if part else f"{t}|{MODE_R}"
            tm = tms.get(nm)
            if tm is None:
                continue
            for n in ns_of(tm, "P1", "none"):
                for m, lr, lam in recipes:
                    T.single("WF9-b", f"WF9-b({word})", f"{_plab(m, lr, lam)}-P1[{word}]|n{nlab(n)}", nm,
                             X.region_stats(tm, gk(m, n, lr, lam), gk("P1", n, "none")), role=role, n=n, method=m, learner=lr,
                             lam=np.nan if lam is None else lam, part=part or "total")
    for n in a.G["wf9"]:                                                 # 서술: 다른 레시피의 격자 안 층화 평균
        for m, lr, lam in recipes[1:]:
            T.contrast("WF9-b", "WF9-b(층화 평균, 서술)", f"{_plab(m, lr, lam)}-P1[격자 안]|n{nlab(n)}", gk(m, n, lr, lam), gk("P1", n, "none"),
                       names=wn, primary=False, role="서술", aux3=False, n=n, method=m, part="w")
    return T.frame()


def tests_wf9x(a, tms, runs):
    """WF9-c(주 4지역 층화 평균의 격자 안 RMSE: R1(λ 0.25) − P1, n ∈ {0, 10, 전량}, 모두 우세 = 지지), WF9-d(서술: 대상별·방법별 격자 안·사이 대비,
    다른 레시피의 층화 평균)."""
    T = X.TestBook(a, tms, None, None)
    w4 = [f"{t}~w|x" for t in MAIN4]
    res = []
    for n in a.G["wf9x"]:
        main = n in (0, 10, -1)
        mr = T.contrast("WF9-c", "WF9-c", f"R1(λ 0.25)-P1[격자 안]|n{nlab(n)}", gk("R1", n, LO, LAM_BASE), gk("P1", n, "none"), names=w4,
                        primary=main, role="주" if main else "서술", aux3=False, n=n, lam=LAM_BASE, part="w")
        if main:
            res.append((f"n={nlab(n)}", mr))
    T.verdict("WF9-c", "WF9-c", _rule3(res, "우세"), X._fmt(res) + f" | 층화 대상 {', '.join(w4)}", used=res)
    recipes = (("R1", LO, LAM_BASE), ("R1", LO, 1.0), ("R2", LO, LAM_BASE), ("D0", LO, 1.0), ("D1", LO, 1.0))
    for nm in sorted(tms):
        t, m = nm.split("|")
        part = "w" if t.endswith("~w") else "b" if t.endswith("~b") else ""
        word = {"w": "격자 안", "b": "격자 사이", "": "총"}[part]
        tm = tms[nm]
        for n in ns_of(tm, "P1", "none"):
            for meth, lr, lam in recipes:
                T.single("WF9-d", f"WF9-d({word})", f"{_plab(meth, lr, lam)}-P1[{word}]|n{nlab(n)}", nm,
                         X.region_stats(tm, gk(meth, n, lr, lam), gk("P1", n, "none")), role="서술", n=n, method=meth, learner=lr, lam=lam,
                         part=part or "total")
    for n in a.G["wf9x"]:
        for meth, lr, lam in recipes[1:]:
            T.contrast("WF9-d", "WF9-d(층화 평균, 서술)", f"{_plab(meth, lr, lam)}-P1[격자 안]|n{nlab(n)}", gk(meth, n, lr, lam), gk("P1", n, "none"),
                       names=w4, primary=False, role="서술", aux3=False, n=n, method=meth, part="w")
    return T.frame()


def region_ep(tmW, tmI, gA, gB):
    """외삽 손실 EP = Δ_W(A − B) − Δ_I(A − B). W·I 는 블록이 겹치지 않고 재표집이 서로 독립(저장소 이름별 seed)이라 두 분포를 같은 번호끼리 뺀다.
    rmse_A·rmse_B 자리에는 Δ_W·Δ_I 를 둔다(h42.region_dd 와 같은 형식)."""
    if tmW is None or tmI is None:
        return None
    w, i = X.region_stats(tmW, gA, gB), X.region_stats(tmI, gA, gB)
    if w is None or i is None:
        return None
    ok = bool(w["ok"] and i["ok"])

    def sub(k):
        return (w[k] - i[k]) if (ok and w[k] is not None and i[k] is not None and len(w[k]) == len(i[k])) else None
    return dict(delta=w["delta"] - i["delta"], delta_beq=w["delta_beq"] - i["delta_beq"], rmse_A=w["delta"], rmse_B=i["delta"],
                n_splits=min(w["n_splits"], i["n_splits"]), n_splits_expected=max(w["n_splits_expected"], i["n_splits_expected"]),
                n_blocks_min=min(w["n_blocks_min"], i["n_blocks_min"]), ok=ok, dist=sub("dist"), dist_beq=sub("dist_beq"), cdist=sub("cdist"),
                cdist_beq=sub("cdist_beq"))


def ep_contrast(T, test, item, label, var, targets, gA, gB, primary=True, role="주", **kw):
    """외삽 손실의 대상별 행과 층화 평균 행(h42.pool_rows). 층화 평균의 이름 = <대상>~<변형>|r(W ∪ I 저장소, 효과 크기 표기용)."""
    names = [f"{t}~{var}|{MODE_R}" for t in targets]
    per = {}
    for t, nm in zip(targets, names):
        s_ = region_ep(T.tms.get(f"{t}~{var}W|{MODE_R}"), T.tms.get(f"{t}~{var}I|{MODE_R}"), gA, gB)
        if s_ is not None:
            per[nm] = s_
    rows = X.pool_rows(per, names, T.a, label)
    if not rows:
        T._missing(test, item, label, "MEAN", role, True, kw)
        return None
    return T._emit(rows, test, item, label, names, primary, True, role, kind="외삽 손실", **kw)


def tests_wf10(a, tms, runs):
    """WF10-a(warm: EP(D0, catboost) 열세), WF10-b(warm: EP(R1 교차검증 λ) − EP(D0) = Δ_W(R1 − D0) − Δ_I(R1 − D0) 우세), WF10-c(warm: W 에서
    R1(교차검증 λ) − P1 의 두 가중 CI 상한 < 0.5 cm). 두 대상 층화 평균, n ∈ {100, 전량}. WF10-d 는 cold 변형과 다른 레시피의 서술."""
    T = X.TestBook(a, tms, None, None)
    tg = list(a.T["wf10"])
    others = (("D0", LO, 1.0), ("D1", LO, 1.0), ("R2", LO, LAM_CV), ("R1", LO, LAM_BASE), ("R1", LO, 1.0), ("P2", "none", None))
    for var in a.VAR10:
        warm = var == "warm"
        res = {"a": [], "b": [], "c": []}
        for n in a.G["wf10"]:
            prim = warm and n in (100, -1)
            role = "주" if prim else "서술"
            ids = (("WF10-a", "WF10-b", "WF10-c") if warm else ("WF10-d", "WF10-d", "WF10-d"))
            ra = ep_contrast(T, ids[0], f"WF10-a({var})", f"EP[D0(catboost)-P1]|{var}|n{nlab(n)}", var, tg, gk("D0", n, HI, 1.0), gk("P1", n, "none"),
                             primary=prim, role=role, n=n, variant=var)
            rb = ep_contrast(T, ids[1], f"WF10-b({var})", f"EP[R1(λ cv)-D0(catboost)]|{var}|n{nlab(n)}", var, tg, gk("R1", n, LO, LAM_CV),
                             gk("D0", n, HI, 1.0), primary=prim, role=role, n=n, variant=var)
            rc = T.contrast(ids[2], f"WF10-c({var})", f"R1(λ cv)-P1[W]|{var}|n{nlab(n)}", gk("R1", n, LO, LAM_CV), gk("P1", n, "none"),
                            names=[f"{t}~{var}W|{MODE_R}" for t in tg], primary=prim, role=role, aux3=False, n=n, variant=var, part="W")
            if prim:
                res["a"].append((f"n={nlab(n)}", ra)); res["b"].append((f"n={nlab(n)}", rb)); res["c"].append((f"n={nlab(n)}", rc))
            for part in ("W", "I"):                                       # 서술: W·I 각각의 Δ(물리식 기준)
                for m, lr, lam in (("R1", LO, LAM_CV), ("D0", HI, 1.0)) + others:
                    T.contrast("WF10-d", f"WF10-d(Δ_{part}, {var})", f"{_plab(m, lr, lam)}-P1[{part}]|{var}|n{nlab(n)}", gk(m, n, lr, lam),
                               gk("P1", n, "none"), names=[f"{t}~{var}{part}|{MODE_R}" for t in tg], primary=False, role="서술", aux3=False, n=n,
                               variant=var, part=part, method=m)
            for m, lr, lam in others:                                      # 서술: 다른 레시피의 외삽 손실
                ep_contrast(T, "WF10-d", f"WF10-d(EP, {var})", f"EP[{_plab(m, lr, lam)}-P1]|{var}|n{nlab(n)}", var, tg, gk(m, n, lr, lam),
                            gk("P1", n, "none"), primary=False, role="서술", n=n, variant=var, method=m)
        if warm:
            T.verdict("WF10-a", "WF10-a", _rule3(res["a"], "열세"), X._fmt(res["a"]), used=res["a"])
            T.verdict("WF10-b", "WF10-b", _rule3(res["b"], "우세"), X._fmt(res["b"]), used=res["b"])
            T.verdict("WF10-c", "WF10-c", _ni_text(res["c"], a.ni_margin), X._fmt(res["c"]), used=res["c"])
        else:
            txt = "서술(cold 변형, 판정어 대신 4분 판정): " + "; ".join(f"{h} " + ", ".join(f"{k} {_v(r)}" for k, r in res_)
                                                         for h, res_ in (("EP(D0)", res["a"]), ("EP(R1) − EP(D0)", res["b"]),
                                                                         ("Δ_W(R1 − P1)", res["c"])))
            T.verdict("WF10-d", "WF10-d(cold)", txt, "", role="서술", used=[])
    return T.frame()


def store_rmse_table(tms_by):
    """3차 보강 저장소별 RMSE 서술 표: (실험, 저장소, 방법, 학습기, 배치, n, λ)마다 분할 안에서 키(추출·seed) 평균 MSE 를 구하고 분할 평균한 값의
    제곱근, 분할 평균 SSE 와 셀 수. WF9-d 의 분해 표와 WF10 의 W·I RMSE 에 쓴다."""
    rows = []
    for exp, tms in tms_by.items():
        for nm, tm in tms.items():
            acc: dict = {}
            for sp, st in tm.used.items():
                per: dict = {}
                for k in st.keys:
                    s_, c_ = st.get(k)
                    tot_c = float(c_.sum())
                    if tot_c > 0:
                        per.setdefault((k[0], k[1], str(k[3]), int(k[4]), float(k[7])), []).append((float(s_.sum()), tot_c))
                for g, v in per.items():
                    acc.setdefault(g, []).append((float(np.mean([q[0] for q in v])), float(np.mean([q[1] for q in v]))))
            for (m, lr, pl, n, lam), v in acc.items():
                sse = float(np.mean([q[0] for q in v])); cnt = float(np.mean([q[1] for q in v]))
                mse = float(np.mean([q[0] / q[1] for q in v]))
                rows.append(dict(exp=exp, store=nm, method=m, learner=lr, placement=pl, n=n, lam=lam, rmse=float(np.sqrt(mse)), mse=mse, sse_mean=sse,
                                 cells_mean=cnt, n_splits=len(v)))
    return pd.DataFrame(rows)


def decomp_table(rt):
    """WF9-d 서술: 대상·방법·n 별 총·격자 안·격자 사이 RMSE, 격자 안 설명 비율 1 − MSE_w(M)/MSE_w(P1), 총 SSE 가운데 격자 안 SSE 비율,
    총 SSE 차(P1 − M) 가운데 격자 안 성분 비율. 분할 평균 SSE 로 계산한다(판정에 쓰지 않는다)."""
    if not len(rt):
        return pd.DataFrame()
    q = rt[rt.exp.isin(["wf9", "wf9x"])].copy()
    if not len(q):
        return pd.DataFrame()
    q["base"] = q.store.str.replace(r"~[wb]\|", "|", regex=True)
    q["part"] = np.where(q.store.str.contains(r"~w\|", regex=True), "w", np.where(q.store.str.contains(r"~b\|", regex=True), "b", "tot"))
    key = ["exp", "base", "method", "learner", "placement", "n", "lam"]
    wide = q.pivot_table(index=key, columns="part", values=["rmse", "sse_mean", "cells_mean"], aggfunc="first")
    wide.columns = [f"{a_}_{b_}" for a_, b_ in wide.columns]
    wide = wide.reset_index()
    ref = wide[(wide.method == "P1") & (wide.learner == "none")].set_index(["exp", "base", "placement", "n"])
    out = []
    for r in wide.to_dict("records"):
        k = (r["exp"], r["base"], r["placement"], r["n"])
        if k not in ref.index:
            continue
        p1 = ref.loc[k]
        p1 = p1.iloc[0] if isinstance(p1, pd.DataFrame) else p1
        g = lambda d, c_: float(d.get(c_, np.nan)) if hasattr(d, "get") else np.nan   # noqa: E731
        mse_w, mse_w0 = g(r, "rmse_w") ** 2, g(p1, "rmse_w") ** 2
        sw, sb, st_ = g(r, "sse_mean_w"), g(r, "sse_mean_b"), g(r, "sse_mean_tot")
        sw0, st0 = g(p1, "sse_mean_w"), g(p1, "sse_mean_tot")
        den = st0 - st_
        out.append(dict(r, expl_within=1.0 - mse_w / mse_w0 if mse_w0 > 0 else np.nan, share_within_of_sse=sw / st_ if st_ > 0 else np.nan,
                        gain_total_sse=den, share_gain_within=(sw0 - sw) / den if np.isfinite(den) and abs(den) > 1e-9 else np.nan,
                        check_tot_eq_w_plus_b=(st_ - (np.nan_to_num(sw) + sb)) if np.isfinite(st_) and np.isfinite(sb) else np.nan))
    return pd.DataFrame(out)




def git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                                     # noqa: BLE001
        return "NA"


LOCAL_NBOOT_MAX = 1000


def summarize(a, elapsed=0.0, skipped=None):
    """차수마다 조각을 읽어 곡선(h40.build_curve)과 가설 표(h42.TestBook)를 만든다. 1차(wf1–wf4)는 <tag>_*, 2차 보강(wf6–wf8)은 <tag>2b_* 에
    쓴다(1차 표를 덮어쓰지 않는다). 실험 사이 저장소는 섞지 않는다. 반환: 차수가 하나면 그 차수의 dict, 둘이면 이어 붙인 표와 rounds."""
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    if not a.PERMIT and int(a.nboot) > LOCAL_NBOOT_MAX:
        print(f"[local] 허용 표지가 없다: --nboot {a.nboot} → {LOCAL_NBOOT_MAX}. 이 집계의 CI 는 등록한 재표집 횟수(10,000)의 값이 아니다", flush=True)
        a.nboot_asked, a.nboot = int(a.nboot), LOCAL_NBOOT_MAX
    outs = {}
    for rnd, exps, tag in rounds_of(a):
        o = summarize_round(a, rnd, exps, tag, elapsed, [s_ for s_ in (skipped or []) if s_.get("exp") in exps])
        if o is not None:
            outs[rnd] = o
    if not outs:
        print("[summarize] 조각이 하나도 없다", flush=True)
        return None
    if len(outs) == 1:
        return next(iter(outs.values()))

    def cat(k):
        fr = [o[k] for o in outs.values() if o.get(k) is not None and len(o[k])]
        return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    return dict(curve=cat("curve"), tests=cat("tests"), meta={r: o["meta"] for r, o in outs.items()}, timing=cat("timing"), failed=cat("failed"),
                rounds=outs)


def summarize_round(a, rnd, exps, tag, elapsed=0.0, skipped=None):
    """한 차수(rnd = r1 또는 r2)의 집계. exps = 그 차수의 실험, tag = 표 이름(<tag>_curve.csv 등)."""
    t0 = time.time()
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    tasks, tests, units_all, runs_all, cfg_info, tms_by = [], [], [], [], {}, {}
    for exp in exps:
        sh = find_shards(a, exp)
        if not sh:
            print(f"[summarize] {exp}: 조각 없음({a.SHARDS}/{exp_tag(a, exp)}__*)", flush=True)
            continue
        units = [json.loads(s_["unit"].read_text()) for s_ in sh]
        cfg_info[exp] = check_cfg(a, exp, units)
        runs = read_runs(sh)
        if len(runs):
            runs = runs.drop_duplicates(subset=["target", "mode", "split", "variant"] + H.KEY_COLS, keep="first")
        stores = load_stores([s_["npz"] for s_ in sh])
        by_name: dict = {}
        for (nm, sp), st in stores.items():
            by_name.setdefault(nm, {})[int(sp)] = st
        tms = {nm: make_tm(nm, bs, units_for(units, nm), a.nboot, is_point_only(exp, nm)) for nm, bs in sorted(by_name.items())}
        tms_by[exp] = tms
        for nm, tm in tms.items():
            t, m = nm.split("|")
            rs = runs[(runs.target == t) & (runs["mode"] == m)] if len(runs) else runs
            tasks.append((exp, nm, tm, rs))
        units_all += units; runs_all.append(runs)
    if not tasks:
        print(f"[summarize] {dict(r1='1차(wf1–wf4)', r2='2차 보강(wf6–wf8)', r3='3차 보강(wf9, wf9x, wf10)')[rnd]}: 조각이 하나도 없다", flush=True)
        return None
    curves = [c_ for c_ in build_curves(a, tasks) if len(c_)]
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if len(curve):
        curve = curve.sort_values(["exp", "target", "mode", "method", "learner", "placement", "lam", "n"]).reset_index(drop=True)
    runs = pd.concat([r_ for r_ in runs_all if len(r_)], ignore_index=True) if any(len(r_) for r_ in runs_all) else pd.DataFrame()
    fn = dict(wf1=tests_wf1, wf2=tests_wf2, wf3=tests_wf3, wf6=tests_wf6, wf7=tests_wf7, wf8=tests_wf8, wf9=tests_wf9, wf9x=tests_wf9x, wf10=tests_wf10)
    for exp, tms in tms_by.items():
        rx = runs[runs.exp == exp] if len(runs) else runs
        try:
            df = tests_wf4(a, tms, rx, [u for u in units_all if u.get("exp") == "wf4"]) if exp == "wf4" else fn[exp](a, tms, rx)
        except Exception as e:                                            # noqa: BLE001  한 실험의 판정 실패가 나머지 집계를 막지 않게 한다
            print(f"  [warn] {exp} 판정 계산 실패: {repr(e)[:300]}", flush=True)
            df = pd.DataFrame([dict(test_id=exp.upper(), item="error", scope="verdict", verdict=f"판정 불가(계산 실패: {repr(e)[:160]})")])
        if len(df):
            df.insert(0, "exp", exp)
            tests.append(df)
        H4.boot_weights.cache_clear()
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    if len(tests):
        tests["hypothesis"] = tests.test_id.isin(WF_HYP)
        tests["design_note"] = "결과 열람 뒤 설계"
        tests["nboot"] = int(a.nboot)
    failed = runs[(runs.n_nonfinite > 0) | runs.fit_flag.isin(["fail", "nonfinite", "fallback"])] if len(runs) else runs
    krige_fb = int(runs.fit_flag.str.startswith("krige_fallback").sum()) if len(runs) else 0
    tg = pd.DataFrame([{k: v for k, v in u.items() if not isinstance(v, (dict, list))} for u in units_all])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    tt = timing_table(units_all)
    curve.to_csv(O / f"{tag}_curve.csv", index=False)
    tests.to_csv(O / f"{tag}_tests.csv", index=False)
    tg.to_csv(O / f"{tag}_targets.csv", index=False)
    tt.to_csv(O / f"{tag}_timing.csv", index=False)
    failed.to_csv(O / f"{tag}_failed.csv", index=False)
    extra_tables = {}
    if rnd == "r3":                                                       # 3차 보강 서술 표(판정에 쓰지 않는다)
        rt = store_rmse_table(tms_by)
        dt = decomp_table(rt)
        rt.to_csv(O / f"{tag}_rmse.csv", index=False)
        dt.to_csv(O / f"{tag}_decomp.csv", index=False)
        extra_tables = dict(rmse=int(len(rt)), decomp=int(len(dt)))
    choices = {}
    if len(runs):
        ch = {"r1": (("wf1", "Pbest"), ("wf1", "Pc"), ("wf3", "Pc"), ("wf1", "R1"), ("wf1", "R2"), ("wf1", "Re"), ("wf1", "R1@x34"), ("wf4", "W")),
              "r2": (("wf6", "Pbest"), ("wf6", "Pc"), ("wf6", "R1"), ("wf6", "R2"), ("wf6", "Re")),
              "r3": (("wf9", "Pc"), ("wf9", "R1"), ("wf9", "Re"), ("wf10", "R1"), ("wf10", "R2"))}[rnd]
        for exp, m in ch:
            q = runs[runs.exp == exp]
            if len(q):
                choices[f"{exp}|{m}"] = choice_freq(q, m)
    r2 = rnd == "r2"
    r3 = rnd == "r3"
    keep = EXPS_R1 if rnd == "r1" else exps                                # meta 의 대상·격자 = 그 차수의 실험(1차는 기존과 같이 wf1–wf4 전부)
    meta = dict(stage={"r1": "H54/WF", "r2": "H54/WF2b", "r3": "H54/WF3b"}[rnd],
                plan="docs/EXPERIMENT_PLAN_WF_2026-10-01.md" + {"r1": "", "r2": " §6", "r3": " §7"}[rnd],
                plan_commit=PLAN_COMMIT[rnd], git_commit=git_commit(), tag=tag,
                code_sha=code_sha(), code_sha_h40=H.code_sha(), code_sha_h42=X.code_sha_x(), exps=list(tms_by),
                targets={k: [f"{t}:{m}" for t, m in v] if k in TRANSFER_EXPS else list(v) for k, v in a.T.items() if k in keep})
    if rnd == "r1":
        meta.update(x34_targets=a.X34)
    splits_meta = (dict(wf6=a.SPLITS6, wf7=a.SPLITS, wf8=a.SPLITS) if r2 else
                   dict(wf9=a.SPLITS9, wf9x=a.SPLITS, wf10=a.SPLITS10, wf10_variants=a.VAR10) if r3 else a.SPLITS)
    meta.update(strategies=a.STRAT8 if r2 else ([] if r3 else a.STRAT), grids={k: list(v) for k, v in a.G.items() if k in keep},
                splits=splits_meta, seeds=a.SEEDS, lams=list(LAMS), nboot=int(a.nboot), extra_tables=extra_tables,
                nboot_capped=getattr(a, "nboot_asked", None), delta_eq=a.delta_eq, ni_margin=a.ni_margin, n_units=len(units_all),
                n_fit_total=int(sum(int(u.get("n_fit_total", 0)) for u in units_all)), unit_elapsed_s_sum=float(sum(u.get("elapsed_s", 0) for u in units_all)),
                elapsed_s=round(float(elapsed), 1), summarize_s=round(time.time() - t0, 1), n_curve=int(len(curve)), n_tests=int(len(tests)),
                n_failed_keys=int(len(failed)), n_krige_fallback=int(krige_fb), unit_status=dict(Counter(str(u.get("status")) for u in units_all)),
                shard_cfg=cfg_info, skipped=skipped or [], choices=choices,
                rules=dict(verdict4="h42.verdict4: 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값 |·| ≤ 0.5 cm, 그 밖 미결정",
                           ci="h40.contrast(h4_common.boot_delta_blocks): 분할 안 채점 블록 재표집, 방법·추출·seed 공통 인덱스, 분할 분포 평균",
                           mean="h40.strat_mean·h42.pool_rows(층화 평균, CI 풀 = 유효 분할 ≥ 1 이고 채점 블록 합집합 ≥ 8)",
                           holm="h42.TestBook.frame: 항목(가설)마다 주 대비의 양측 부트스트랩 p 에 Holm(보조 열). WF4-a 는 p_ni 에 따로 holm_p_ni"),
                implementation={"r1": IMPL_NOTES, "r2": IMPL_NOTES2, "r3": IMPL_NOTES3}[rnd])
    (O / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[summarize] 실험 {list(tms_by)} · 조각 {len(units_all)} · runs {len(runs):,} · curve {len(curve):,} · tests {len(tests):,} · "
          f"{time.time() - t0:.0f}s → {O}/{tag}_*", flush=True)
    if len(tests):
        v = tests[tests.scope.isin(["verdict", "verdict_aux"])]
        cols = [c_ for c_ in ("exp", "test_id", "role", "target", "verdict") if c_ in v]
        if len(v):
            with pd.option_context("display.max_colwidth", 140, "display.width", 250):
                print(v[cols].to_string(index=False), flush=True)
    return dict(curve=curve, tests=tests, meta=meta, timing=tt, failed=failed)


IMPL_NOTES = {
    "krige": "자체 국소 정규 크리깅(구면 3차원 좌표 km, 최근접 32, 지수 변이도 실용 범위 모수화, 경험 변이도 12구간·최대 거리 절반·√쌍수 가중, "
             "범위 60개 로그 격자 + 비음수 최소제곱, 부분표본 1,500). 라벨 < 10 또는 구간 < 3 이면 P1 대체(fit_flag). RK = 표본 안 잔차 크리깅",
    "cluster": "A 풀 x25 중앙값 대체·표준화(라벨 미사용), k-평균 n_init 10, B 는 가장 가까운 중심. Pc = 군집 계수의 전체 E_n 으로의 κ 10 수축. "
               "R1c 는 군집 라벨 ≥ 20 만 따로 적합(그 밖 전체 g). R1i = 원핫 k 열 입력",
    "s6": "20개 무작위(S1 n = 20 과 같다) → 예산 경계까지 최대 20개씩. 선택 모형 = R1 잔차 CatBoost(catboost_lo, posterior_sampling), "
          "VirtEnsembles 10개 분산이 큰 셀. 실패하면 그 단계만 무작위(notes.s6_fallback)",
    "s4_s5_s7": "S4 표준화 x25 탐욕 k-중심(첫 점 무작위), S5 LG 원천(모드 x) 기준 AOA 비유사도 큰 순(결정적, 추출 0 만 저장), S7 √TDD 큰 순(동률 seed)",
    "x34": "SAR 결측은 CatBoost 결측 처리에 맡긴다. x34 조각은 catboost_lo 의 D0·D1·R1·R2 만",
    "lgd": "run_tables/{Tibet, NAtlantic_lic, Russia_C} 를 h40.Data 로 읽는다(약관 미확인 행 0). NAtlantic~lic 은 점 추정만",
    "inregion_pseudo": "지역 내 D1·R2 의 유사라벨 행 = round(10 × 학습 라벨 수), A 풀 복원 추출",
    "re_anchor": "0.5·(E_n·s + cci_alt)(CCI 유효 셀), 그 밖 E_n·s",
    "cv": "h42.cv_folds_of(블록 5묶음, 블록 1개면 셀 묶음), seed 0 적합으로 선택, 묶음마다 E_n 재계산, 동률은 작은 λ·앞 후보·작은 k",
    "p0_ref": "지역 내 실험의 P0·κ 수축 사전값 E0 = LG 원천(모드 x, 100 km 버퍼) 최소제곱. 알래스카 하위 지역도 모드 x",
    "wf4_targets": "LG 27 (대상, 모드) + LGD 새 지역 3 = 30. 추출과 유사라벨 색인은 LG 와 같다",
}

IMPL_NOTES2 = {
    "wf6_splits": "half_split_blocks split_seed 1–25(LG 1–5 + 새 6–25). h40.Data.split_structure 의 코드를 분할 1–25 에 적용(_SplitView): "
                  "중복 분할 제외, 유효 분할이 있으면 채점 블록 2개 미만 분할 제외. 빈 자리를 다른 seed 로 채우지 않는다",
    "wf6_methods": "wf1 과 같은 지역 내 문맥·추출·교차검증. 물리 보정식 묶음(P1, P2, Pk, Pc, Pbest), R1·R2(λ 0.25·0.5·1.0·교차검증), "
                   "Re(λ 0.25·교차검증), D0(catboost 600·깊이 6·0.03 만). R1·R2·Re 는 catboost_lo. 추출 3(전량 1), seed 2",
    "wf7": "LG 27 (대상, 모드), 분할 1–5, n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5, seed 2, 원천·추출 = LG. Pe = 앵커(E_n), "
           "Re = 앵커(E_n) + 0.25·ge, ge 의 원천 행 목표 = y − 앵커(E0), 선택 라벨 행 목표 = y − 앵커(E_n). 앵커(E) = 0.5·(E·s + cci_alt)(CCI 유효), "
           "그 밖 E·s. P0·P1·R1(λ 0.25)은 같은 조각에서 다시 적합(같은 플랫폼 대비). n = 0 은 원천 행만",
    "wf8": "모드 x, 원천 = LG. 전략 S1 = h40.draw_cells, S2 = h40.draw_blocks, S4 = wf2 의 탐욕 k-중심(A 풀 x25 표준화, seed 는 wf2 와 같다). "
           "n {10, 40}(|A| 미만), 추출 5, seed 2. 같은 라벨 집합의 추출은 적합을 재사용하고 추출마다 저장",
    "verdicts": "WF6-a 대상별(레시피 R1·R2·Re 교차검증 λ, D0 catboost; 하나라도 P1 대비 우세면 지지, Pbest 대비는 보조). "
                "WF6-b 세 대상 층화 평균 R2(교차검증 λ) − P1, n ∈ {500, 1000, 전량} 모두 우세 = 지지, 일부 = 부분 지지, 열세 또는 우세 없음 = 기각. "
                "WF7-a·b 방향 중립 서술(주 4지역 층화 평균, 확인적 가설 아님). WF8-a 레나·캐나다 n 40 R1 의 S4 − S1 우세 지역 ≥ 1. "
                "WF8-b 다섯 대상 n 10 R1 의 S2·S4 − S1 에 열세가 없으면 지지",
    "tables": "2차 보강 표는 <tag>2b_*(1차 <tag>_* 와 따로). 조각 이름 <tag>{6,7,8}__cpu__<대상>__<모드>__s<분할>[__<전략>]",
}


IMPL_NOTES3 = {
    "wf9_grid": "격자 묶음 = 채점 셀의 (√TDD(토양 도일) 값을 유효숫자 12자리로 적은 문자열, 블록). 비유한 √TDD 셀은 혼자 묶음. "
                "격자 안 저장소(<대상>~w) = 2셀 이상 묶음 셀의 (y − ȳ_g, ŷ − ŷ̄_g), 격자 사이 저장소(<대상>~b) = 모든 셀의 (ȳ_g, ŷ̄_g). "
                "블록마다 총 SSE = 격자 안 SSE + 격자 사이 SSE",
    "wf9_inregion": "WF6 와 같은 문맥·분할(1–25, split_structure_ext)·추출(3, 전량 1)·교차검증. 방법 P1, Pk, Pc(교차검증 k), R1(교차검증 λ, λ 0.25·0.5·1.0), "
                    "Re(교차검증 λ, λ 0.25), D0(catboost). R2, P2, Pbest 는 뺐다",
    "wf9x": "LG 문맥(h40.build_ctx)·원천·추출(5, n 0·전량 1)·seed. 방법 P0, P1, R1·R2(λ 0.25·0.5·1.0, 원천 행 + 선택 라벨, R2 는 유사라벨 행 포함), "
            "D0(catboost_lo, 원천 행 + 선택 라벨), D1(catboost_lo, 유사라벨 행 포함). n = 0 은 E_n = E0",
    "wf10_split": "블록 평균 √TDD(유한 셀의 셀 수 가중). warm 은 큰 순, cold 는 작은 순(동률 블록 이름)으로 대상 셀 25 % 이상까지 W, 나머지를 "
                  "seed_of('wf10-I', 대상, 변형, 분할) 순열로 25 % 이상까지 I, 나머지 A. 앞선 분할과 I 블록 집합이 같으면 중복, W·I 채점 블록 2개 미만이면 무효",
    "wf10_methods": "P1, P2, R1(교차검증 λ, λ 0.25·0.5·1.0), R2(교차검증 λ, 고정 λ), D0(catboost, catboost_lo), D1(catboost_lo). 같은 적합으로 W·I 를 "
                    "따로 채점(저장소 <대상>~<변형>W·I, 총 <대상>~<변형>). 추출 seed = h40.draw_cells(대상, 'r10' + 변형 첫 글자, 분할, n, 추출)",
    "wf10_ep": "EP = Δ_W − Δ_I. W·I 의 재표집은 저장소 이름별 seed 로 독립이고 두 분포를 같은 번호끼리 뺀다. 층화 평균은 h42.pool_rows",
    "verdicts": "WF9-a(세 대상 격자 안 R1(교차검증 λ) − P1, n 500·1,000·전량), WF9-c(주 4지역 격자 안 R1(λ 0.25) − P1, n 0·10·전량): 모두 우세 지지, "
                "열세 없이 일부 우세 부분 지지, 열세 또는 우세 없음 기각. WF10-a(EP(D0 catboost) 열세), WF10-b(EP(R1 교차검증 λ) − EP(D0) 우세), "
                "WF10-c(W 의 R1 − P1 두 가중 CI 상한 < 0.5 cm): warm, 두 대상 층화 평균, n 100·전량",
    "tables": "3차 보강 표는 <tag>3b_*(curve, tests, meta, targets, timing, failed, rmse, decomp). 조각 이름 <tag>{9,9x,10}__cpu__<대상>__<모드>__s<분할>[__<변형>]",
}


def timing_table(units):
    rows = []
    for u in units:
        for k, nf in (u.get("n_fit_detail") or {}).items():
            ax, lr, m = (k.split("|") + ["", ""])[:3]
            sec = float((u.get("sec_detail") or {}).get(k, 0.0)); nr = float((u.get("rows_detail") or {}).get(k, 0.0))
            rows.append(dict(exp=u.get("exp", ""), target=u.get("target", ""), mode=u.get("mode", ""), split=u.get("split", -1), variant=u.get("variant", ""),
                             learner=lr, method=m, n_fit=int(nf), sec=sec, sec_per_fit=sec / max(int(nf), 1), rows_per_fit=nr / max(int(nf), 1),
                             est_s=float((u.get("est_detail") or {}).get(k, np.nan)), threads=u.get("threads", "")))
        rows.append(dict(exp=u.get("exp", ""), target=u.get("target", ""), mode=u.get("mode", ""), split=u.get("split", -1), variant=u.get("variant", ""),
                         learner="krige", method="krige", n_fit=int(u.get("n_krige", 0)), sec=float(u.get("krige_s", 0.0)),
                         sec_per_fit=float(u.get("krige_s", 0.0)) / max(int(u.get("n_krige", 0)), 1), rows_per_fit=np.nan,
                         est_s=float(u.get("est_krige_s", 0.0)), threads=u.get("threads", "")))
        rows.append(dict(exp=u.get("exp", ""), target=u.get("target", ""), mode=u.get("mode", ""), split=u.get("split", -1), variant=u.get("variant", ""),
                         learner="kmeans", method="kmeans", n_fit=int(u.get("n_kmeans", 0)), sec=float(u.get("kmeans_s", 0.0)),
                         sec_per_fit=float(u.get("kmeans_s", 0.0)) / max(int(u.get("n_kmeans", 0)), 1), rows_per_fit=np.nan,
                         est_s=float(u.get("est_kmeans_s", 0.0)), threads=u.get("threads", ""), elapsed_s=u.get("elapsed_s", np.nan)))
    return pd.DataFrame(rows)


# ================================================================ 적합 수 세기
def count_only(a, units, skipped):
    """학습 없이(dry) 적합 수와 추정 시간을 센다. 추정 = LG BASE_FIT 모형(h40, catboost 는 h42 가정값) + 소표본 하한, 크리깅·k-평균은 가정값."""
    t0 = time.time()
    rows, det, est = [], Counter(), Counter()
    for u in units:
        r = run_unit(a, *u, dry=True)
        for k, v in r.pop("_detail").items():
            det[f"{u[0]}|{k}"] += v
        r.pop("_rows")
        for k, v in r.pop("_est").items():
            est[f"{u[0]}|{k}"] += v
        rows.append(r)
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    if not len(df):
        print("[count-only] 작업 단위 없음", flush=True)
        return df
    df["est_total_s"] = df.est_s + df.est_krige_s + df.est_kmeans_s
    tab = []
    for k in det:
        exp, _, lr, m = (k.split("|") + ["", "", ""])[:4]
        tab.append(dict(exp=exp, learner=lr, method=m, n_fit=int(det[k]), est_h=est[k] / 3600))
    tab = pd.DataFrame(tab).sort_values(["exp", "learner", "method"])
    for rnd, exps, tag in rounds_of(a):                                   # 차수마다 자기 표에 쓴다(다른 차수의 표를 덮어쓰지 않는다)
        if df.exp.isin(exps).any():
            df[df.exp.isin(exps)].to_csv(a.OUT / f"{tag}_count.csv", index=False)
        if len(tab) and tab.exp.isin(exps).any():
            tab[tab.exp.isin(exps)].to_csv(a.OUT / f"{tag}_count_detail.csv", index=False)
    by = df.groupby("exp").agg(units=("split", "size"), targets=("target", "nunique"), fits=("fit_total", "sum"), krige=("n_krige", "sum"),
                               kmeans=("n_kmeans", "sum"), est_fit_h=("est_s", "sum"), est_krige_h=("est_krige_s", "sum"), est_all_h=("est_total_s", "sum"))
    for c_ in ("est_fit_h", "est_krige_h", "est_all_h"):
        by[c_] = (by[c_] / 3600).round(2)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"[count-only] 작업 단위 {len(units)} · 건너뜀 {len(skipped)} · 세기 {time.time() - t0:.0f}s", flush=True)
        print(by.to_string(), flush=True)
        tb = tab.copy(); tb["est_h"] = tb.est_h.round(3)
        print("[count-only] 실험·학습기·방법별 적합 수와 추정 누적 시간(프로세스 1개, 스레드 4 기준)", flush=True)
        print(tb.to_string(index=False), flush=True)
    tot_h = float(df.est_total_s.sum()) / 3600
    print(f"[count-only] 총 적합 {int(df.fit_total.sum()):,} · 크리깅 {int(df.n_krige.sum()):,} · k-평균 {int(df.n_kmeans.sum()):,} · "
          f"추정 누적 {tot_h:.1f} CPU-h(4스레드 프로세스 기준)", flush=True)
    for w in (15, 22, 43):
        print(f"  워커 {w}개(4스레드) → 약 {tot_h / w:.2f} h(단위 크기 불균형과 감속 제외)", flush=True)
    for rnd, exps, tag in rounds_of(a):                                   # 차수별 합계와 1차 Rescale 실측 비(0.62)의 환산
        q = df[df.exp.isin(exps)]
        if not len(q):
            continue
        h = float(q.est_total_s.sum()) / 3600
        hx = float(q.est_total_s.max()) / 3600
        print(f"[count-only] {dict(r1='1차', r2='2차 보강', r3='3차 보강')[rnd]} {exps}: 단위 {len(q)} · 적합 {int(q.fit_total.sum()):,} · 크리깅 {int(q.n_krige.sum()):,} · "
              f"추정 누적 {h:.2f} CPU-h · 실측 비 {RESCALE_RATIO} 적용 {h * RESCALE_RATIO:.2f} CPU-h · 워커 22개 약 {h * RESCALE_RATIO / 22:.2f} h · "
              f"가장 긴 단위 {hx:.2f} h(비 적용 {hx * RESCALE_RATIO:.2f} h) → {tag}_count.csv", flush=True)
    if skipped:
        print(f"[count-only] 건너뛴 분할·대상: {dict(Counter(s_['status'] for s_ in skipped))}", flush=True)
    return df


# ================================================================ WF5 다음 관측 우선순위(지도)
def wf5_priority(strategy, cand_X, cand_s, cand_lat, cand_lon, lab_X, lab_y, lab_s, E0, n_next, seed, src_X=None, threads=1, cb_iters=200):
    """WF5: 후보 셀(격자)에서 다음 관측 n_next 개의 순서(후보 색인)와 점수를 전략으로 정한다. lab_* = 이미 관측된 라벨 셀.
    S1 무작위, S2 0.5° 블록 순환(h40.draw_blocks), S3 군집 크기 비례(k = 8), S4 기존 라벨에서 시작한 탐욕 k-중심, S5 원천(src_X) 대비 AOA
    비유사도, S6 기존 라벨로 맞춘 R1 잔차 모형의 가상 앙상블 분산(새 셀의 라벨이 없으므로 순차 갱신 없이 한 번에 고른다), S7 √TDD 큰 순.
    표준화는 후보 셀의 공변량으로 한다(S5 는 원천 통계). 반환 dict(order, score)."""
    cand_X = np.asarray(cand_X, float); N = len(cand_X)
    n_next = int(min(n_next, N))
    rs = int(seed) % (2 ** 31 - 1)
    if strategy == "S1":
        return dict(order=np.random.RandomState(rs).permutation(N)[:n_next], score=None)
    if strategy == "S2":
        blk = np.floor(np.asarray(cand_lat) / 0.5).astype(int) * 100000 + np.floor(np.asarray(cand_lon) / 0.5).astype(int)
        sel = H.draw_blocks("wf5", "map", 0, n_next, rs, blk)
        return dict(order=np.asarray(sel, int), score=None)
    st = standardize_fit(cand_X)
    Zc = standardize(cand_X, st)
    if strategy == "S3":
        labA, _ = kmeans_labels(Zc, Zc[:0], S3_K, rs)
        alloc = proportional_alloc(np.bincount(labA, minlength=int(labA.max()) + 1), n_next)
        rng = np.random.RandomState(rs)
        out = [rng.choice(np.where(labA == cl)[0], int(k), replace=False) for cl, k in enumerate(alloc) if k > 0]
        return dict(order=np.concatenate(out).astype(int), score=None)
    if strategy == "S4":
        init = standardize(lab_X, st) if lab_X is not None and len(lab_X) else None
        return dict(order=kcenter_order(Zc, n_next, rs, init=init), score=None)
    if strategy == "S5":
        if src_X is None or not len(src_X):
            raise ValueError("S5 는 원천 공변량(src_X)이 필요하다")
        ss = standardize_fit(src_X)
        di = aoa_di(standardize(cand_X, ss), standardize(src_X, ss), rs)
        return dict(order=rank_desc(di, rs)[:n_next], score=di)
    if strategy == "S6":
        lab_y = np.asarray(lab_y, float); lab_s = np.asarray(lab_s, float)
        E_ls = H.ls_E(lab_y, lab_s)
        E_n = X.shrink(E_ls, E0, len(lab_y), KAPPA)
        v = ve_variance(np.asarray(lab_X, np.float32), lab_y - E_n * lab_s, cand_X.astype(np.float32), rs, cb_iters, threads)
        return dict(order=rank_desc(v, rs)[:n_next], score=v)
    if strategy == "S7":
        return dict(order=rank_desc(np.asarray(cand_s, float), rs)[:n_next], score=np.asarray(cand_s, float))
    raise ValueError(f"알 수 없는 전략 {strategy}")


def run_wf5(a):
    """WF5 지도: 레나 격자(육지 셀, x25 결측 없음)에 --wf5-strategy 를 적용해 다음 --wf5-n 개 관측 위치의 우선순위를 쓴다.
    기대 오차 감소는 WF2 모의 결과(<tag>_curve.csv 의 전략별 곡선)에서 가져온다(아직 연결하지 않았다. 계획서 개정 이력의 TODO)."""
    strat = a.wf5_strategy
    if strat not in STRATEGIES:
        raise SystemExit("--wf5-strategy 를 S1–S7 가운데 하나로 준다(WF2-b 규칙으로 정한 전략)")
    g = pd.read_csv(_abs(a.wf5_grid))
    keep = (g.get("land", 1) == 1) & np.all(np.isfinite(g[FEATS].values.astype(float)), 1) & np.isfinite(g.e5_sqrt_tdd.values.astype(float))
    g = g[keep].reset_index(drop=True)
    HA, D = get_data(a)
    df = D.df
    li = D.target_idx("Lena")
    _, _, src_idx, _ = D.source_idx("Lena", "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    E0 = H.ls_E(df.y.values[src], df.s.values[src])
    ns = sorted(_n_list(a.wf5_n))
    res = wf5_priority(strat, g[FEATS].values, g.e5_sqrt_tdd.values, g.lat.values, g.lon.values, df[FEATS].values[li], df.y.values[li],
                       df.s.values[li], E0, max(ns), seed_of("wf5", strat), src_X=df[FEATS].values[src], threads=a.threads, cb_iters=a.cb_iters)
    order = np.asarray(res["order"], int)
    out = pd.DataFrame(dict(cell_id=g.cell_id.values[order] if "cell_id" in g else order, lat=g.lat.values[order], lon=g.lon.values[order],
                            rank=np.arange(1, len(order) + 1), score=(np.asarray(res["score"])[order] if res["score"] is not None else np.nan)))
    for n in ns:
        out[f"next_{n}"] = out["rank"] <= n
    a.OUT.mkdir(parents=True, exist_ok=True)
    path = a.OUT / f"wf5_priority_{strat}.csv"
    out.to_csv(path, index=False)
    meta = dict(strategy=strat, n=ns, n_candidates=int(len(g)), n_labels=int(len(li)), E0=float(E0), grid=str(a.wf5_grid), code_sha=code_sha(),
                expected_error_reduction="TODO: WF2 의 전략별 곡선(<tag>_curve.csv, placement = 전략)에서 예산별 Δ 를 붙인다")
    (a.OUT / f"wf5_priority_{strat}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[wf5] 전략 {strat} · 후보 {len(g):,} · 라벨 {len(li):,} · 상위 {max(ns)} → {path}", flush=True)
    return dict(executed=[], skipped=[], resumed=[])


# ================================================================ 실행
def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)                                     # 워커(spawn)가 물려받는다
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                               # CPU 전용(GPU 학습기를 쓰지 않는다. 계획서 §1)
    if a.threads_asked > a.threads:
        print(f"[local] 허용 표지가 없다: --threads {a.threads_asked} → {a.threads}", flush=True)
    if a.wf5_map:
        return run_wf5(a)
    will_run = not (a.count_only or a.summarize_only)
    if will_run and not a.smoke and not a.PERMIT:                         # 자료를 읽기 전에 거부한다
        raise SystemExit("[거부] 스모크가 아닌 실행(사전 점검, 본 실행)은 Rescale 작업에서 한다. WF_RESCALE=1(또는 LG_RESCALE=1) 또는 "
                         "--allow-local 을 준다(공유 서버 보호)")
    if will_run and not a.PERMIT and int(a.workers) > 1:
        print(f"[local] 허용 표지 없는 스모크: --workers {a.workers} → 1", flush=True)
        a.workers = 1
    if a.summarize_only:
        if not a.PERMIT and int(a.workers) > 1:
            a.workers = 1
        out = summarize(a, 0.0, [])
        return dict(executed=[], skipped=[], resumed=[], n_fail=0 if out is not None else 1)
    units, skipped, expected = enumerate_units(a)
    print(f"[data] 실험 {a.EXPS} · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 실험별 {dict(Counter(u[0] for u in units))} · 분할 {a.SPLITS} · "
          f"seed {a.SEEDS} · 스레드 {a.threads} · 워커 {a.workers}", flush=True)
    if a.count_only:
        count_only(a, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[])
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    resumed, todo = [], []
    for u in units:
        ok, why = unit_state(a, *u) if a.resume else (False, "")
        if ok:
            resumed.append(u)
        else:
            todo.append(u)
            if a.resume and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name(u)} ({why})", flush=True)
    todo.sort(key=lambda u: -cost_hint(a, u))
    print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · 프로세스 {max(int(a.workers), 1)} · 스레드 {a.threads}", flush=True)
    done, failed = execute(a, todo, expected, argv)
    n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
    n_part = sum(1 for u in done if u["status"] == "partial")
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {n_part} · {time.time() - t0:.0f}s", flush=True)
    if not a.no_summarize:
        summarize(a, time.time() - t0, skipped)
    return dict(executed=[(u["exp"], u["target"], u["mode"], u["split"], u["variant"]) for u in done], skipped=skipped, resumed=resumed,
                n_fail=n_fail, n_partial=n_part)


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
