# Handoff: 데이터 재구조화(스레드 R) — 가중/집계는 무료 이득, 시간정합은 매핑엔 게이트 탈락

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-07-08 15:38
**Session focus**: apparent-floor 진단이 가리킨 "데이터를 잘못 넣어서 생긴 병목"을, 새 데이터 취득 전에 **기존 데이터 재구조화(㉠시간정합 ㉡집계 ㉢가중)**로 검증.
**Author**: Claude (Fable 5) + willy010313

---

## 1. TL;DR
- **㉢ 1/n 가중 = 무료 큰 이득(채택)**: 위치당 공평 채점 시 pooled(현재) skill **−0.8%(보간)/0.04%(전이)** → **가중 10.9%/7.3%**. 데이터 한 톨 안 늘리고 밀집셀 편향만 제거해 실력 회복. (`restructure_gate_results.csv`)
- **현재 "R²0.20"은 밀집셀 착시였음**: 상위 1% 셀이 전체 점의 12% 차지 → 점-단위 지표가 부풀려짐. 위치-동등 채점하면 현재 pooled는 거의 평균 수준.
- **㉡ 셀 집계(채택)**: 225k→14,348 위치당 1행, 셀평균 ALT + **셀내 SD(중앙 2cm)를 불확실성 라벨**로 분리(척도정합·정직 UQ). 전이 최고(cell-trained skill 8.1%).
- **㉠ 시간정합 = 매핑엔 게이트 탈락(정직한 음성)**: ERA5-Land 2010–2024(816MB) 확보 후 '그해 기후' 대입 → 보간 +3(8.7 vs 5.7) but **전이 −5(5.1 vs 9.9)·연도holdout −8(20.1 vs 27.5)**. ALT는 '그 해 날씨'보다 '위치 고유성질'이 지배. 시간축은 매핑 지렛대 아님 → lagged-ALT 기반 **예측(T1)** 응용에서만 유효.
- **메타**: 모델(DL) 아닌 **데이터가 지렛대** 재확인. "데이터 올바르게 넣기"의 실질 이득 = 가중/집계. 정확도 남은 지렛대 = 새 모달리티(SoilGrids/Sentinel) 또는 예측 응용 전환.

## 2. Context
- 직전 핸드오프: [20260706_1717 문헌 재조사·예측/4D 트랙 판정](20260706_1717-lit-review-forecasting-4d-tracks.md). 그 세션에서 P0(과대표현 정정)·스레드 A(다중모달 ablation)·횡단 AOA/UQ 실행하고, **apparent-floor 진단**(17cm은 물리벽 아닌 정보병목: within 13.7%/between 86.3%, 비가역하한 7.2cm ≪ 현재 16.9cm; `apparent_floor_diagnosis.csv`) + **pseudo-replication**(같은 공변량 셀에 ALT 34–96cm 공존, 정적 climatology로 연도차가 잡음) 발견.
- 이번 세션 동기: 사용자 제안 — "새 데이터 다운로드보다, 이미 있는 데이터를 올바른 구조(연도맞춤·집계·가중)로 넣는 게 먼저 아닌가?" → 검증.
- 전략 확인 결과: ㉡㉢은 무료·즉시(재구조화 먼저 맞음), **㉠은 ERA5 다년 다운로드가 전제**(디스크엔 2015–2020만, 라벨 70%가 2010–2014). 둘 병렬 실행.

## 3. What we did
- **㉡㉢ 집계·가중** — `scripts/1_data_prep/aggregate_alt_cell.py`, `scripts/2_evaluation/restructure_gate.py`
  - `dl_dataset_cell.csv`(14,348 위치): 셀평균 ALT + `alt_sd`(셀내 SD) + n. 공정비교: 세 방법 모두 **held-out 셀평균 ALT** 채점(apples-to-apples), 공간블록+LORO.
  - 결과: (1)pooled −0.8/0.04% → (2)+1/n가중 10.9/7.3% → (3)cell학습 −1.9/**8.1%(전이 최고)**.
  - 적대검증: 가중치 0.005~1.0·loc당 합=1 확인, 상위1%셀=점12% → 밀집편향 확증(버그 아님). 절대 skill은 fold 분산 큼(정성 결론).
- **ERA5 다년 다운로드** — `scripts/0_download/era5land_monthly_multiyear.py`, CDS 자동. **816MB, 180 monthly steps(2010–2024)**, t2m/sd/stl1. wallclock ~7분(CDS 큐 4분+전송). 라벨 99.4% 연도정합.
- **㉠ 시간정합** — `scripts/1_data_prep/era5land_temporal_covariates.py`, `scripts/2_evaluation/temporal_gate.py`
  - 연도별 도일/적설/토양온도 파생(`alt_era5_temporal.csv` 2.96M행, 같은 위치 연도별 TDD SD=145 → 연도신호 실재). (위치,연도) 17,800단위, 조인 100%.
  - static vs temporal GBM: 보간 8.7>5.7(+3), **전이 5.1<9.9(−5), per-year 20.1<27.5(−8)**. → 매핑 게이트 탈락.
- **정리**: EXPERIMENT_LOG/SESSION_HANDOFF/EXPERIMENTS/PLAN_FORWARD 갱신, GPU→6,7,8,9, gitignore(대용량 temporal CSV). 4커밋 push(origin/main, `5ddc286..c33b096` 검증).

## 4. Key numbers (this session)
| Method | Case | Metric | Value | Source |
|---|---|---|---|---|
| pooled(현재) | 셀평균, 공간블록 | skill / RMSE | −0.8% / 18.68cm | `restructure_gate_results.csv` |
| **pooled+1/n가중** | 셀평균, 공간블록 | skill / RMSE | **10.9% / 16.51cm** | 동 |
| pooled(현재) | 셀평균, LORO | skill | 0.04% | 동 |
| **pooled+1/n가중** | 셀평균, LORO | skill | 7.3% | 동 |
| **cell-trained** | 셀평균, LORO | skill / RMSE | **8.1% / 16.94cm** | 동 |
| static(정적기후) | (위치,연도), LORO | skill | 9.9% | `temporal_gate_results.csv` |
| temporal(그해기후) | (위치,연도), LORO | skill | 5.1% (악화) | 동 |
| static | (위치,연도), per-year | skill | 27.5% | 동 |
| temporal | (위치,연도), per-year | skill | 20.1% (악화) | 동 |
| 셀내 SD(불확실성 라벨) | 14,348 셀 | median | 2.0cm | `dl_dataset_cell.csv`(집계) |
| ERA5-Land 2010–2024 | NH monthly | steps / size | 180 / 816MB | `raw/era5land/nh_monthly_2010-2024.nc` |

## 5. Decisions made
- **㉢ 1/n 가중 채택** — 무료로 보간+11%/전이+8% 회복. 향후 모든 매핑 학습에 위치-동등 가중 적용.
- **㉡ 셀 집계 채택** — 척도정합(정답=셀평균) + 셀내 SD를 정직한 불확실성 라벨로.
- **㉠ 시간정합 매핑용 배제** — 게이트 탈락(전이·연도holdout 악화). 시간축은 예측(T1, lagged-ALT) 응용으로만.
- **평가 프레이밍 전환** — 점-단위(밀집셀 가중) 대신 **위치-동등** 채점이 정직한 지표. 기존 R²0.20은 밀집 착시로 명기.

## 6. Open questions / blockers
- **재구조화 기준선 위 다중모달 ablation** — 스레드 A(정적 pooled)를 가중/집계 기준선에서 재실행하면 각 모달리티(기후/지형/InSAR/PolSAR) 기여가 어떻게 바뀌나? (증거: `alt_feature_ablation.py`를 cell/가중 버전으로.)
- **fold 분산** — 공간블록 6개라 절대 skill이 fold 배정에 민감(fold0 pathological). 증거: kNNDM CV 또는 블록 부트스트랩으로 CI.
- **예측(T1) 응용의 실효성** — lagged-ALT(사이트 지속성)로 GRU가 정적 baseline을 이기나? 단, 다관측 사이트(≥2년) 13,177개 한정, 매핑 아닌 모니터링 예측 산출물.

## 7. Next steps (prioritized)
1. **재구조화 기준선 확정 + 스레드 A 재실행** — owner: Claude, ~반나절. 가중/집계 데이터에서 다중모달 ablation 다시 → "진짜 feature 기여". 선결: 없음(데이터 준비됨).
2. **데이터 확장(정확도 남은 지렛대)** — owner: Claude/user, ~1–2일. SoilGrids(서버복구)·Sentinel-1 InSAR 시계열·Sentinel-2/MODIS → ablation M7~M9. 선결: 계정/서버.
3. **스레드 B 3D 열구조** — owner: Claude, ~1–2일. GBM 조건장 + CCI 0/1/2/5/10m prior + 깊이인코딩 + 물리 단조성 → PyVista 인터랙티브 열큐브 + 0°C 등온면.
4. **(선택) 예측 T1 파일럿** — owner: Claude, 게이트. lagged-ALT + 월별 forcing GRU vs 정적 baseline, 모니터링 사이트 미래 ALT. 시간정합이 매핑엔 탈락했으므로 이건 별도 응용으로만.

## 8. Pointers
- Authoritative: `docs/PLAN_FORWARD.md`(방향·스레드·게이트·중단조건), `SESSION_HANDOFF.md`, `docs/EXPERIMENTS.md`, `docs/EXPERIMENT_LOG.md`.
- 결과 CSV: `data/processed/{restructure_gate_results,temporal_gate_results,apparent_floor_diagnosis,alt_feature_ablation_results,alt_conformal_aoa_results,alt_aoa_transfer_results,curated_scope_results_rescored,model_tournament_results_rescored}.csv`.
- 그림: `outputs/figures/02_evaluation/{restructure_gate,temporal_gate,apparent_floor_diagnosis,skill_reframing,coverage_calibration}.png`, `outputs/figures/06_deep_learning/alt_feature_ablation.png`, `outputs/maps/{alt_aoa_mask,alt_uncertainty_width}.png`.
- 표준지표 모듈: `src/polar/eval_metrics.py`. 문헌: `references/INDEX.md`(+`00_core10/`).
- Active jobs: 없음. 대용량 제외 데이터: `raw/era5land/nh_monthly_2010-2024.nc`(816MB, git 제외), `dl_dataset_{cell,temporal}.csv`·`alt_era5_temporal.csv`(git 제외, 스크립트로 재생성).
- 관련 이전 핸드오프: `20260706_1717-lit-review-forecasting-4d-tracks.md`, `20260706_1113-tournament-floor-curated-demo.md`.

## 9. Caveats for GPT
- **"17cm=물리하한"·"12.97cm=SOTA 돌파" 인용 금지(정정됨)** — 각각 apparent floor(covariate 병목)·범위축소 아티팩트. 모든 RMSE에 **skill-over-mean/R² 병기**(표준 `eval_metrics.py`).
- **점-단위 R²0.20은 밀집셀 착시** — 위치-동등 채점이 정직한 지표(현재 pooled ≈ 평균 수준).
- **시간정합이 정확도를 올린다고 가정 금지** — 매핑 게이트 탈락(전이·연도holdout 악화). 시간은 예측(T1) 응용에서만.
- **fold 분산 큼**(공간블록 6개) — 절대 skill 수치는 정성적으로. 상대비교(같은 fold 내)만 신뢰.
- 캐노니컬 명칭(project.yaml): B0/B0b/B1(신경장 탈락)/tournament/curated. 스레드 R=데이터 재구조화(㉠시간정합 ㉡집계 ㉢가중). GPU 최신=6,7,8,9.
