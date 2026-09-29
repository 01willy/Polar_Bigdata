data/processed/lgw/ (부록 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 3.6·8.4 의 산출 폴더)

- h39(scripts/2_evaluation/h39_scenarios.py)의 모든 모드와 8.4 교차 환경 점검의 산출을 이 폴더 하나에 둔다.
  h39 의 --out-dir 기본값이 이 폴더다. summarize 도 이 폴더로 돌린다(lgw_tests.csv 를 하나로 유지한다).
- data/processed/wrapup 은 이 폴더를 가리키는 심볼릭 링크다. 결과 추가 1(2026-09-30 03:25–03:35)의 산출을
  2026-09-30 04:07 에 이 폴더로 옮겼고, 04:08 에 고친 h39 로 다시 돌렸다(부록 결과 절, 개정 이력 '결과 추가 1 수정 1').
- sealed/: LG·LGX 결과와 같은 값이거나 같은 형식인 표(sealed/lgw_sc3_full.csv)와 판정 행이 든 로그(sealed/logs/).
  LG·LGX 회수 전에는 열지 않는다(sealed/README_SEALED.txt).
- 결과성 수치가 든 파일(봉인하지 않았다. 인용하거나 열 때 표지를 붙인다):
  lgw_sens_unit.csv 의 row_p1, row_p2, ref_row_p1, ref_row_p2 행과 lgw_sens_unit_splits.csv, lgw_sens_unit_ref_splits.csv 의
  d_row_* 열은 해석식 P1 − P0, P2 − P0 의 분할 분포(split_seed 1–200, 추출 5)다. 레나·캐나다·Alaska(x)·러시아 W 에서
  LGX-N4a 와 같은 양이고 추출 seed 만 다르다. AB4, SC1w-P 와 같은 형식이다.
- xenv/lgx/: 8.4 (ii)의 로컬 h42·h41 집계 표. 대조 계산 외에는 열지 않는다.
- xenv/lg*, logs/xenv_i_*.log, lgw_xenv_gate_i*: 8.4 (i) 작업의 산출(별도 작업).
- logs/h39_*.log: h39 실행 로그(건수만 찍는다).
