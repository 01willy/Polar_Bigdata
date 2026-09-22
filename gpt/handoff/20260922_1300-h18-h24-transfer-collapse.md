# Handoff: 전이 붕괴 완화 실험 H18–H24(정보·계수·학습 목표·관측 설계) 실행 결과 + 논문급 그림 11종
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-22 13:00
**Session focus**: "공변량만 있는 지역에서 ML/DL 전이 붕괴를 해결 또는 완화"를 목표로, 전 세션에 사전 등록한 H18–H24 를 변인 통제 설계(한 요인만 변경)로 실행하고, 결과를 개정 09-21 채점 규약으로 확정한 뒤, 검토 에이전트 2종의 지적을 반영해 논문급 그림 11종을 산출.
**Author**: Claude Fable 5.1 + 백승원

---

## 1. TL;DR
- **라벨 없는 전이에서 물리식을 넘는 요인은 없었다.** 관측 지표 강제력(MODIS LST, H18), 수문·단열·지반 공변량 23종(H19), CCI 상대 계수(H20), 지역 간 계수 대여(H21), 불변 관계 학습(IRM·V-REx·GroupDRO·DANN·지역 임베딩, H22) 모두 Stefan(알래스카 E) 대비 Δ ≥ 0 또는 CI 가 0 을 포함(AB4, 층화 블록 부트스트랩). 공변량 23종을 전부 넣으면 오히려 +0.64 [0.05, 1.12] 악화, SoilGrids 를 빼도 +0.51 악화.
- **유일한 유의 이득은 관측 설계(H24)**: 라벨 3개를 공변량 공간 대표점(k-중심)으로 고르면 무작위보다 레나 −0.67·캐나다 −0.74·러시아 W −2.44(CI 0 제외), 알래스카 샌드박스 −1.50, 캐나다 무작위 라벨의 해(+5.6)가 +0.9 로 감소. n ≥ 10 에서는 규칙 차이 소멸. 기존 CALM 사이트 위치만 쓰면 무작위보다 나쁘다.
- **H23 MAML**: 라벨 3개 적응이 E 수축 적합을 넘지 못함(AB4 Δ +1.09; 캐나다 −0.51·레나 −0.32 우세, 러시아 W +4.03·E +1.16 열세). 수준 오차 지역에서는 라벨이 잔차 적응이 아니라 E 재추정에 쓰여야 한다.
- **진단의 핵심**: 지역 계수 E 는 지리·기후·토양·지표 산출물 어느 것으로도 예측되지 않는 "지역 고유 상수"처럼 거동(LORO R² 전 집합 음수). 레나의 수준 차이는 LST n-factor 로 설명되나(자체 E 1.42→1.62, 알래스카 1.63) RMSE 이득은 작고, 러시아 W 의 1.7배 격차는 지표 강제력이 아니라 지표 아래 조건(얼음·이탄·수분)에 있다.
- **감사 정정 2건**: 층화 요약의 점 추정치가 CI 산출 지역과 다른 집합으로 계산되던 결함(`polar.m1_stats.summarize_delta`)과 fig01 의 iw 필터 결함을 수정, 전 수치 재생성. 그림은 시각·과학 검토 후 결함 수정본.

## 2. Context
- 전신: `gpt/handoff/20260922_1039-m1-results-label-budget-next-plan.md`(M1 확정·라벨 예산 곡선·다음 계획 H18–H24). 상위 계획 `docs/EXPERIMENT_PLAN_NEXT_2026-09-22.md`, 세부 설계(사전 등록, 결과 열람 전 커밋 8385a1c) `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md`(§9 결과 요약 추가).
- 사용자 지시(09-22): 연구 의의(ML/DL 이 물리식을 넘는 모델; 라벨 있음은 달성)를 살리면서 공변량만 지역의 전이 붕괴를 해결·완화하는 실험을 변인 통제로 설계·실행. GPU 4·5·6·7 사용(8·9 타 사용자). 결과 시각화는 논문급·직관적·폴더 규칙 준수.

## 3. What we did
- **자료 취득·공변량 확장** — `scripts/1_data_prep/fetch_ext_covariates.py`, `fetch_era5_ext_resume.py`, `build_covariates_ext.py` → `data/processed/covariates_ext_v1.csv`(+meta). MODIS MOD11C3 LST·MOD13C2 NDVI(Earthdata, pyhdf), JRC GSW·Hansen 10° 타일, ERA5-Land 확장 8변수(CDS, curl 재개), Copernicus DEM → pysheds TWI(90 m 다운샘플, 176 타일 84 s). 군 LST 6·T 6·W 3·V 5·H 10.
- **하네스 확장** — `src/polar/m1_core.py`(`load_base(ext=)`, `register_ext`, `eval_mask(require)`, 앵커·유사라벨 `stefan_lst`), `m1_master_factorial.py --ext --eval-require --axis ext`(GPU 6·7 샤드 2, 12 분), 공용 채점 `src/polar/m1_stats.py`, 통합 분석 `scripts/2_evaluation/h_analysis.py` → `data/processed/h2/h_tests_{all,main}.csv`.
- **H20·H21 해석적** — `scripts/3_deep_learning/h21_coef_borrow.py`(유사도 3 × 추정량 × w, 공여 leave-one-out 중첩 선택, 물리식 조건 표) → `h21_{summary,tests,nested,physics_rules,forcing_rule}.csv`.
- **H19a 해석적** — `h19_blockE.py`(블록 E 설명력 LOBO/LORO, 계수 지도 앵커) → `h19_blockE{,_cells}.csv`.
- **H22** — `h22_invariant.py`(M1 MLP 고정, 환경 = 알래스카 6 fold ∪ 지역, 초모수 LOEO 중첩 선택, GPU 5, 45 분) → `h22_rows.csv`, `h22_preds.npz`.
- **H23** — `h23_maml.py`(FOMAML 3,000 에피소드, 대상당 1회 학습, GPU 4) → `h23_{runs,summary,tests}.csv`.
- **H24** — `h24_obs_design.py`(규칙 6 × n × 추정량 4, 4지역 + 알래스카 샌드박스, CPU 4워커 51 분) → `h24_{runs,curve,tests}.csv`.
- **배포 규칙 탐색** — `scripts/2_evaluation/h_deploy_gating.py` → `h_deploy_gating.csv`.
- **그림** — `scripts/4_visualization/h2_figs.py` → `outputs/figures/h2/`(PNG 300 dpi + PDF, 스펙 `figures/figure_spec.json` id h2_*): fig01 붕괴 구조, fig02 정보 축 포레스트, fig03 블록 E 설명력, fig04 계수 대여, fig05 학습 목표, fig06 MAML, fig07/07b 관측 설계 곡선·요약, fig08 종합 포레스트, map01 신규 공변량 지도, map02 오차 변화 지도, map03 관측 설계 지도. 팔레트는 dataviz 검증기 통과 4색(#3a6ea5·#2a9d8f·#8e6bbf·#b8791f)+중립 참조.
- **검토** — visual-reviewer·scientific-figure-reviewer 에이전트 2종 실행, blocker 2건(통계 요약 지역 집합, iw 필터)·should-fix 9건 수정 후 재렌더.

## 4. Key numbers (단위 cm, AB4 = 레나·캐나다·러시아 W·E, 층화 블록 부트스트랩 95% CI; 정보 없음의 CI 집합도 블록≥8 인 같은 4지역)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| Stefan(LST 도일) − Stefan(기온 도일) | 정보 없음 / 공변량만 | ΔRMSE | +0.16 [−0.65, 1.24] / +0.10 [−0.79, 0.91] | H18a |
| ML(x25+LST) − ML(x25) | 공변량만, 직접 CatBoost+유사라벨 | ΔRMSE | +0.03 [−0.12, 0.12] | H18b |
| ML(x25+A 23종) − ML(x25) | 공변량만 / 정보 없음 | ΔRMSE | +0.64 [0.05, 1.12](Holm 0.22) / +0.33 [−0.32, 0.88] | H19b |
| x25 − SoilGrids(x16) − x25 | 공변량만 | ΔRMSE | +0.51 [0.10, 0.95] | H19x |
| 블록 E LORO R²(비알래스카, n 가중) | 기준 17 → +A / +LST | R² | −0.59 → −0.87 / −1.04 (CatBoost) | H19a |
| cci_blockE − Stefan | 정보 없음 AB4 | ΔRMSE | −0.43 [−2.56, 3.51]; 블록 등가중 +5.5 | H20 |
| 계수 대여(중첩) − E_AK Stefan | 정보 없음 AB4 | ΔRMSE | +1.26 [−0.52, 2.61]; 자체 E 상한 −4.33 [−6.33, −1.47] | H21 |
| IRM / V-REx / DANN 잔차 λ=.25 − Stefan | 공변량만 | ΔRMSE | +0.39 [0.11, 0.70] / +0.02 [−0.26, 0.34] / +0.27 [0.05, 0.44] | H22 |
| V-REx − ERM (같은 MLP) | 공변량만 | ΔRMSE | −0.31 [−0.44, 0.00] | H22x |
| k-중심 선택 n=3(E 수축) − 무작위 | 레나/캐나다/러시아 W/러시아 E/알래스카 샌드박스 | ΔRMSE | −0.67 [−1.06, −0.32] / −0.74 [−1.02, −0.48] / −2.44 [−3.05, −1.85] / +0.23 [−0.03, 0.49] / −1.50 [−1.92, −1.07] | H24 |
| 캐나다 E 재적합 n=3 대 Stefan | 무작위 / k-중심 / 좌표 층화 / 블록 순환 | ΔRMSE | +5.6 / +0.9 / +3.1 / +6.0 | H24 |
| MAML 적응 n=3 / E 수축+MAML 잔차 − E 수축(κ=10) | 공변량만 AB4(λ=.25) | ΔRMSE | +1.09(캐나다 −0.51 [−0.71, −0.28], 러시아 W +4.03 [3.31, 4.85]) / +0.20 | H23 |
| 알래스카 샌드박스 n=20 (E 수축) | 무작위 / 좌표 층화 / k-중심 / 기존 CALM 사이트 | RMSE | 18.2 / 16.4 / 17.2 / 17.3 (물리식 22.1) | `h24_curve.csv` |
| 배포 게이팅 최선(cci_agree20) 대 Stefan+CCI | 정보 없음 AB4 | RMSE | 29.0 vs 28.9 | `h_deploy_gating.csv` |

## 5. Decisions & rationale
- 결정 규칙(개정 09-21 §D): H18–H23 전부 C(기각), H24 는 지역별 CI 3/4 + 샌드박스로 A(라벨 3–5개에 한정).
- 층화 요약 정정: 점 추정치와 CI 를 같은 지역 집합(블록≥8)으로 통일하고 전 지역 단순평균은 `delta_allregions` 로 병기. 이전 세션 산출(m1_analysis.py)은 별도 함수라 영향 없음(확인 필요 시 같은 규칙 적용 권고).
- 그림 규칙: 확인적 행 별표 = Holm p<.05, 탐색 행 = CI 기준(†); 컬러바 절단은 화살표·캡션 고지; 블록<8 지역은 흐림·점만.

## 6. Open issues / risks
- H19b 의 +0.64 는 CI 가 0 을 제외하나 Holm 후 0.22: "추가 시 악화"는 방향·크기로 서술하고 유의성은 주장하지 않는다.
- 정보 없음 조건의 확장 마스크(신규 공변량 유효)로 레나 2,958→2,710 셀, 그린란드 제외. 기준 재채점 행으로 분리 보고.
- 그림은 발표·보고서 폭(12–17.5 in). 논문 2단 폭으로 줄이려면 주석 폰트(9 pt) 재조정 필요(검토 에이전트 지적).
- H23 MAML 은 BN 없는 MLP·FOMAML 3,000 에피소드 한 설정만 시험. 2차 MAML·ANIL·더 긴 메타학습은 미시험.
- 외부 자료 한계: MODIS LST 는 청천 월 기후값(0.05°), GSW·Hansen 은 80°N 이북 없음, ERA5 확장은 모델 산출.

## 7. Next steps
1. 데이터 확장이 선행 조건: 라벨 지역 수(현재 CI 가능 4개)를 늘려야 계수 대여·불변 학습·관측 설계 규칙의 일반성을 검정할 수 있다(GTN-P/CALM 추가 지역, 시추공 유도 라벨 재편입).
2. 지표 아래 정보: InSAR 계절 침하(얼음 풍부도 대리)·이탄 두께(PEATMAP/NCSCD)·IPA 구역 변수로 E 회귀 재시도. 8일 MOD11A2·MOD10 적설 지속기간으로 n-factor 재시험.
3. 관측 설계 절 원고화: "라벨 예산 대 오차"(S-F) + "선택 규칙"(H24) 통합 그림·표, 알래스카 최소 관측망 지도.
4. 원고 결과 절 6부 재단(정보 축·계수 축·학습 목표 축 기각 + 관측 설계 채택), Sci Rep 프런트매터와 정합.

## 8. Files touched (this session)
- 신규: `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md`, `src/polar/m1_stats.py`, `scripts/1_data_prep/{fetch_ext_covariates,fetch_era5_ext_resume,build_covariates_ext}.py`, `scripts/3_deep_learning/{h21_coef_borrow,h24_obs_design,h22_invariant,h23_maml,h19_blockE}.py`, `h1819_run.sh`, `scripts/2_evaluation/{h_analysis,h_deploy_gating}.py`, `scripts/4_visualization/h2_figs.py`, `data/processed/h2/*`(요약 CSV·meta; npz·h24_runs.csv 6 MB 미추적), `data/processed/covariates_ext_v1.csv`(14 MB 미추적, 스크립트 재생성; meta JSON 은 추적), `outputs/figures/h2/*`.
- 수정: `src/polar/m1_core.py`(ext 병합·입력 집합 등록·stefan_lst), `scripts/3_deep_learning/m1_master_factorial.py`(--ext/--eval-require/--axis ext), `figures/figure_spec.json`(h2_* 11건), `docs/EXPERIMENT_LOG.md`, `SESSION_HANDOFF.md`.
- 원자료(미추적): `data/raw/{modis_cmg,gsw,hansen}/`, `data/raw/era5land/nh_monthly_ext_2015-2020.nc`.
