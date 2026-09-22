# Handoff: M1 마스터 요인 실험 결과 확정 · 라벨 예산 곡선 · "ML이 물리식을 넘게 하기" 다음 계획
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-22 10:39
**Session focus**: 사전 등록된 단일 하네스(M1)로 3조건 × 6축을 전면 재실행해 논문 결과를 확정하고, 회의적 감사·Sci Rep 격차 보완·희소 라벨 곡선을 마친 뒤, 연구 목적("ML/DL이 물리 경험식을 넘게 학습시키기")에 맞춘 다음 세션 계획(H18–H24)을 사전 등록.
**Author**: Claude Fable 5.1 + 백승원

---

## 1. TL;DR
- **라벨 없는 전이(공변량만·정보 없음)에서 어떤 ML/DL 구조·앵커·증강 조합도 1-파라미터 Stefan 물리식을 확실히 넘지 못했다.** 모델 10종 직접 회귀는 +2~+20 cm 악화, 잔차 모드 ±1 cm, 중첩 선택 레시피 27.9 대 Stefan 26.7(H12 기각), Stefan+CCI 결합 −0.9~−1.4 cm(CI 0 포함, 블록 등가중 +1.3, 확정 효과 아님).
- **물리 유사라벨 증강(C1)은 대조군을 강하게 잡아도 이득이 남는다**: 증강 없음 −3.6, 상수 −4.0, 대상 수준 상수 −1.8, 행 셔플 −1.9, TDD 선형 −0.7 cm(AB4, 전부 CI 0 제외). 이득의 절반은 수준 이식, 나머지는 기후 단조 관계와 √ 형태.
- **라벨 예산 곡선**: 러시아 서부는 라벨 3개로 39.8→30.6(수준 오차 지역), 레나는 40~80개에 0.1~1 cm(구조 오차), 캐나다는 무작위 라벨이 해(블록 이질성). 안전 기본값 = κ=10 수축 E. 라벨 있음 조건 잔차 결합은 −0.7~−1.1 cm 유의(H13).
- **전이 UQ**: 알래스카 보정 CQR의 전이 커버리지 0.32~0.73(명목 0.90 미달), 앵커(Stefan+CCI) ± 밀도비 가중 Q 구간만 0.84~0.93 회복(러시아 W 0.56). 생성 모델 3종 구간은 지역 내 0.62~0.72로 기각.
- **다음 세션 정본 `docs/EXPERIMENT_PLAN_NEXT_2026-09-22.md`**: 목적을 "ML이 넘으려면 어떤 정보·학습 목표가 필요한가"로 재정립. H18 MODIS LST 관측 강제력, H19 수문·단열 공변량(TWI 등), H20 시뮬레이터 상대 계수, H21 지역 간 계수 대여, H22·H23 불변 학습·MAML, H24 관측 설계(알래스카 라벨 선택 → 4지역 실전 시험).

## 2. Context
- 전신: `gpt/handoff/20260921_1600-m1-master-factorial-scirep-gap.md`(같은 세션 전반부: 데이터 v3·하네스·재현 게이트·S-C·보조 분석 12종·프런트매터 초안), 그 전 `20260915_1554-e1-e4-reconciliation-transfer-plan.md`.
- 동기: 09-16 사용자 지적(실험 축이 단계마다 달라 체계적 비교 불가) → 단일 하네스·단일 결과 DB로 전면 재실행 후 논문화. 09-18 감사 워크플로(6렌즈 79건)로 통계 설계·코드·대안 설명·Sci Rep 요건 점검.
- 09-22 사용자 관점 교정: 연구 목적은 물리식 개선이 아니라 **ML/DL이 물리식을 넘게 학습시키는 것**. 실패는 (원인 → 부족 정보 → 다음 실험)으로 기록.

## 3. What we did
- **데이터 v3** — Files: `scripts/1_data_prep/build_fidelity_base_v3.py`, `data/processed/fidelity_base_v3.csv`(미추적, 스크립트 재생성), `e5_soil_tdd_v3.csv`, `tests/test_fidelity_v3.py`. Result: 스발바르에서 노르웨이 본토·스웨덴 3사이트 분리(Scandinavia), CALM Method 파싱으로 지온·시추공 유도 68셀 `F4_calm_temp` 강등, 토양 도일 0인 8셀 결측. 주 전이 집합 탐침 셀 보존(캐나다 752→750). GTN-P ALT 106 데이터셋은 CALM 중복(신규 지역 0)이라 보류.
- **M1 하네스** — Files: `src/polar/m1_core.py`, `scripts/3_deep_learning/m1_master_factorial.py`, `m1_analysis.py`(seed별 짝지음·블록 3종 가중·층화 부트스트랩·Holm·중첩 선택·UQ·재현 게이트), `m1_run_queue.sh`, `scripts/4_visualization/m1_figs.py`, `src/polar/m1_ext.py`(S-D 확장 준비). 생성 모델 3종(cfm·ddpm·nflow)을 `src/polar/tab_models.py`에 추가. TabPFN은 `TABPFN_TOKEN` 부재로 제외.
- **재현 게이트(S-B)** — Result: 옛 표 18개 중 13개 0.1 cm 이내 재현, 캐나다 5개는 데이터 판(v1 742·v2 752·v3 750셀)과 S3 채점 셀 정의(CCI 결측 이상치 포함 371셀) 차이로 규명(v1 `--eval-all` 재실행에서 34.0/31.8/42.0 재현). Files: `data/processed/m1/m1_gate_gate.csv`, `m1_gate_v1_gate.csv`, `m1_gate_v1all_shard0.csv`.
- **S-C 주효과 7,190적합 + 심부 420** — GPU 5–9, 5레인 순차 큐, 10:27–13:33(RealMLP 레인 2.8 h). Files: `data/processed/m1/m1_{anchor,pseudo,model,realmlp,dset,xset,iw,deep}_shard*.csv`, 분석 `m1_sc_{summary,tests,nested,uq,combo_decomp,consistency,splits}.csv`, 그림 `outputs/figures/m1/m1_sc_main_effects.png`.
- **사전 등록 개정(09-21, 결과 열람 전)** — `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md` 끝: 채점 규약 3종, seed별 짝지음, 층화 부트스트랩 주 추론, AB4, 다중 비교, 결정 규칙, 공식 혼합 표기, H16·H17.
- **보조 분석 12종(CPU)** — `scripts/2_evaluation/{l2_*,a2_*}.py`, `scripts/3_deep_learning/m1_{cv_scheme_comparison,transfer_uq,label_year_sensitivity,sparse_label_curve}.py`, `scripts/2_evaluation/m1_level_structure_split.py`. 산출 `data/processed/m1/{cv_scheme_*,l2_*,a2_*,label_year_*,transfer_uq_*,sparse_label_*,m1_level_structure}.csv`.
- **문서** — `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md`(데이터·코드 가용성, 선언문, 통계 명세표, Reporting Summary, Gautam 대비 절, 용어 정정), `environment-lock.txt`, `docs/EXPERIMENT_PLAN_NEXT_2026-09-22.md`, 로그 `docs/EXPERIMENT_LOG.md` 09-18/21·09-22 항목.
- **운영 사고 2건과 조치**: (1) shard 12개 동시 실행 + torch 전 코어 스레드로 load average 213 → GPU당 1프로세스·스레드 상한(정확도 영향 0, 예측 비트 동일 확인). (2) 에이전트 9개+워크플로 동시 실행으로 사용량 한도 → 동시 에이전트 2개 이하 규칙(메모리).

## 4. Key numbers (this session, 단위 cm, AB4 = 레나·캐나다·러시아 W·E 비가중 평균, 층화 블록 부트스트랩 95% CI)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| Stefan 유사라벨 − 증강 없음 / 상수 / 대상 수준 상수 / 행 셔플 / TDD 선형 | 공변량만, 직접 CatBoost r=10 | ΔRMSE | −3.57 [−4.76, −1.85] / −4.02 / −1.80 [−2.40, −1.04] / −1.88 / −0.70 [−0.94, −0.20] | `m1_sc_tests.csv` H3 |
| Stefan+CCI − Stefan (λ=0) | 정보 없음 / 공변량만 | ΔRMSE | −0.86 [−2.02, 0.87] / −1.39 [−2.81, 1.28]; 블록 등가중 +1.28 | H1 |
| 앵커+catboost_lo λ=.25 − 앵커 | 정보 없음 / 공변량만 | ΔRMSE | +0.03 [−0.59, 0.48] / +0.19 [−0.62, 0.65] | H2 |
| 중첩 선택 레시피 vs Stefan 단독 | 공변량만 | RMSE | 27.87 vs 26.73 | `m1_sc_nested.csv`, `m1_sc_summary.csv` |
| 물리 계수 지도 앵커 − Stefan | 공변량만/정보 없음 | ΔRMSE | +3.0~+6.7(유의 악화) | H10 |
| 라벨 있음 Stefan+catboost_lo λ=.25/.5 − Stefan | A블록 실측 학습 | ΔRMSE | −0.68 [−1.01, −0.27] / −1.14 [−1.83, −0.32] | H13 |
| 학습 지역 확장 loro/loro_main − 알래스카만 | 정보 없음 | ΔRMSE | +0.06 / +0.16 (CI 0 포함) | H14 |
| 직접 회귀 10종 − Stefan | 정보 없음 | ΔRMSE | CatBoost +2.3~+2.6 · MLP +3.0 · FT-T +3.9 · TabM +4.6 · RealMLP +4.3 · 능형 +11 · cfm +12.5 · ddpm +8.0 · nflow +19.8 | X-model |
| 생성 모델 90% 구간 | 알래스카 지역 내 | 커버리지 / interval score | cfm 0.70/151 · ddpm 0.62/166 · nflow 0.67/149 vs CQR 0.93/68.9 | `m1_sc_uq.csv` |
| 알래스카 CQR 전이 커버리지 | 정보 없음, 캐나다/레나/러시아 E/W | 커버리지 | 0.60 / 0.57 / 0.73 / 0.32; 앵커±가중 Q 0.88/0.93/0.84/0.56 | `transfer_uq_summary.csv` |
| 라벨 n개 E 재적합, 러시아 W | 공변량만+라벨 | RMSE | n=0 39.8 → n=3 30.6 → n=10 29.2 | `sparse_label_curve.csv` |
| 라벨 n개, 레나 (Stefan / 잔차 λ.25) | 〃 | RMSE | 20.4 → n=80 20.3 / 20.3 (승률 0.73) | 〃 |
| 라벨 n개 E 재적합, 캐나다 | 〃 | RMSE | 25.4 → n=3 32.4 → n=80 29.6 (해); A 전량 372셀 잔차 25.1 | 〃 |
| 수준·구조 분리(ML 구조 중심화) − 앵커 | 정보 없음 | ΔRMSE | CatBoost −0.12~−0.18(캐나다 −1.40 [−2.61, −0.13]), 나머지 ≥ 0, 러시아 E +0.9~+1.3 | `m1_level_structure.csv` |
| 교차검증 방식 | 알래스카 CatBoost(저용량) / Stefan+능형 | RMSE | 무작위 11.55 · 0.5° 블록 16.10 · kNNDM 17.91 / 12.77 · 13.33 · 13.70 | `cv_scheme_comparison.csv` |
| 결합 앵커 오차 상관 ρ(Stefan, CCI) | 레나/캐나다/러시아 W/E | ρ | 0.99 / 0.67 / 0.83 / 0.90; 실현 이득 = 독립 기대의 12–17% | `a2_combo_decomp.csv` |
| 지역 내 오라클 이득(라벨 전량) | 알래스카 / 레나 / 캐나다 | ΔRMSE vs Stefan | −1.13 / −0.8 / ≈0 | `m1_sc_tests.csv` X-ak·H13, `sparse_label_curve.csv` |
| 재현 게이트 | 옛 표 18개 | 통과 | 13/18 재현 + 5 원인 규명 | `m1_gate_gate.csv` 등 |

## 5. Decisions made
- 주 결과 문장: "라벨 없는 지역에서는 물리식이 상한이며, 물리 유사라벨 증강은 ML을 그 상한에 붙이는 방법이고, 라벨이 일부 있으면 물리+ML이 상한을 넘는다." Stefan+CCI 결합은 "집계 의존 소폭 경향"으로 서술(정보원 다양성 문구 삭제).
- 확인적 요약은 AB4(A/B 분할이 성립하는 4지역), 채점 = seed별 짝지음 + 층화 블록 부트스트랩, 부호검정은 기술 통계.
- 조건 명칭: 자료 상황 2종(라벨 있음 전량·희소 / 라벨 없음 공변량만) + 비적응 기준선 1줄.
- 운영: GPU당 프로세스 1개, torch·OMP 4스레드, CatBoost 6스레드; 동시 에이전트 2개 이하, 워크플로는 명시 요청 시만.
- 연구 목적 재확인(사용자): ML/DL이 물리식을 넘게 하기. 음성 결과는 (원인 → 부족 정보 → 다음 실험)으로 기록.

## 6. Open questions / blockers
- 어떤 정보가 지역 간 E 차이를 결정하는가: MODIS LST 기반 n-factor·지표 도일(H18), 수문·단열 공변량(H19)이 블록별 E 분산의 절반 이상을 설명하는지가 핵심 미확인. 이 답이 "ML이 물리식을 넘을 수 있는가"를 정한다.
- 지역 간 계수 대여(H21)가 러시아 C·소표본 지역에서 실제로 개선되는지(해석적, 미실행).
- 알래스카 최소 관측 설계(H24)의 규칙이 다른 지역에서 재현되는지.
- TabPFN 편입 여부(사용자 `TABPFN_TOKEN` 필요), InSAR(Sentinel-1) 자료 확보 가능성.
- 영문 원고 재단 미착수.

## 7. Next steps (prioritized)
1. **해석적 실험 3건(CPU 반나일, Claude)**: H20 시뮬레이터 상대 계수 E_cci, H21 지역 간 계수 대여(유사도 3종·수축), 물리식 조건 엄밀화(멱지수·n-factor·토양 도일·Kudryavtsev 성분). 전제: 없음.
2. **H24 관측 설계(CPU 반나일)**: 알래스카 라벨 선택 전략 × n → 4지역 A블록 실전 시험. 전제: 없음.
3. **공변량 확충(반나일~1일)**: TWI·하천 거리(DEM 즉시), MODIS LST 지표 도일·n-factor(AppEEARS/GEE 계정 확인 필요), 습지·호수·이탄·지반 얼음·수목 피복·NDVI·SMAP·적설 깊이 다운로드 → 물리식 사다리·M1 모델/유사라벨 축 재실행(GPU 1–2장, 1프로세스/GPU).
4. **H22·H23 불변 학습(IRM/DANN)·MAML(GPU 1장 1일)**: 6 라벨 지역 leave-region-out.
5. **원고 재단(1일)**: 결과 절 6부 + 관측 설계 절, 프런트매터 적용, `docs/RESULTS_RECONCILIATION_2026-09-14.md` §5 문장 교체. GPT 역할: 결과 절 초안 비판 검토·서론 차별 서술 검토.
6. S-D 최소 집합(GPU 1시간): 자기훈련 대조(H16)·셀 셔플·편향 물리 용량-반응(`src/polar/m1_ext.py` 연결).

## 8. Pointers
- Authoritative reports: `outputs/report/main.tex`(국문 보고서, 영문화 대상), 결과 DB `data/processed/m1/`, 그림 `outputs/figures/m1/`.
- Active jobs: none(모든 큐 종료, GPU 5–9 유휴).
- Ckpt: 없음(모델 저장 없음, 예측 npz만; `data/processed/m1/*_preds.npz` 미추적).
- Related prior handoffs: `gpt/handoff/20260921_1600-m1-master-factorial-scirep-gap.md`, `20260915_1554-e1-e4-reconciliation-transfer-plan.md`, `20260831_1105-final-deck-audit-journal.md`.
- 계획 정본: `docs/EXPERIMENT_PLAN_NEXT_2026-09-22.md`(다음), `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`(개정 09-21 포함), `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md`.

## 9. Caveats for GPT
- 옛 헤드라인 "24.11→22.92(위성 결합+잔차)"는 철회됨. 결합 앵커 값은 23.27(정보 없음 2지역) 또는 AB4 Δ −0.86(CI 0 포함)로만 인용.
- "완전 전이"라는 용어 대신 "라벨 없음(공변량만)" + "비적응 기준선". "정보 없음"은 자료 상황이 아니라 방법.
- 캐나다 수치는 데이터 판(v1 742·v2 752·v3 750셀)에 따라 다르다. 원고는 v3 기준.
- Stefan 계수 E는 최소제곱(1.617)이 정본, 중앙값비(1.620)는 tab:physics 참고. 알래스카 6-fold는 fold 0이 단일 블록 4,716셀(35%).
- H13 "라벨 있음"은 A블록 전량(레나 1,500·캐나다 370·러시아 15셀). "라벨 몇 개"는 `sparse_label_curve.csv` 참조. 캐나다에서 무작위 라벨 E 재적합은 해.
- 생성 모델 잔차 모드의 구간은 λ 배율이라 λ=1 행만 UQ 비교에 사용.
- 명칭 정본: `.claude/project.yaml` naming_canonical(S1–S14, E1–E4, M1, AB4).
