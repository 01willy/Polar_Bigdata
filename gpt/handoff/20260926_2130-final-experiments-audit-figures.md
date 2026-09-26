# Handoff: 통합 실험 A2·B1–B4·C1·C2·D1 실행, 실험·문서 감사(21건 확정), H25 블록 CI 재실행, 논문 그림
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-26 21:30
**Session focus**: 오전에 확정한 통합 계획(`docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md`)을 사용자 지시로 같은 세션에 실행했다. 병렬로 기존 실험·결과·문서 감사(적대 검증)와 최상위 저널 그림 관행 조사·그림 재설계를 수행했다. 감사 결과 채점 CI 를 블록 부트스트랩으로 바꾸고 H25 를 재실행했으며, 절 2 의 주장을 다시 세웠다.
**Author**: Claude Fable 5.1 → Claude Opus 5.5(사용량 한도 후 전환) + 백승원

---

## 1. TL;DR
- 감사: 지적 22건 중 21건 확정. 옛 지역 안 CI(분할·반복 행 재표집)는 채점 블록 분산을 빠뜨려 좁았다. 'ρ −0.84 로 필요 라벨 수 예측'은 규칙·정의·오라클 E비·사후 지표 조합의 산물이라 폐기. H24 '채택' 철회, H28 등록 가설 기각. 공용 모듈 `src/polar/h4_common.py`(블록 부트스트랩·최소 n·절단 순위·오프셋 MLE)로 모든 새 실험과 H25 재실행을 채점했다.
- 가설 판정: 지지 F1(구조 지역 8개 중 6개가 잔차로 n ≤ 40 미달성, 레나 n* 320)·F8(TabPFN = CatBoost, n ≤ 10)·F11(몽골 외부 홀드아웃, 라벨 정의 교락); 부분 F2(라벨 분산 배치는 E 추정 시 −1.1 ~ −1.7)·F3(수준 지역 이득의 72–95 % 가 E 처리)·F10(계층 conformal 커버리지 0.86); 기각 F4(3라벨 진단 분기)·F5(적응형 풀링)·F6(이론 필요 라벨 수)·F7(D-최적)·F9(CCI 다층).
- 원고 절 2 새 축: 소수 라벨의 가치는 E 추정에서 나오고, 잔차 ML 은 수백 개가 있어야 대상에 적응하며, 라벨은 흩어 모은다. 고정 κ=10 이 관측 범위에서 가장 견고하다. 악화 지역을 미리 가려내는 라벨 기반 진단은 없다(음성 결과).
- 그림: `scripts/4_visualization/paper_figs.py` + `paper/fig1–7.py, table1.py, supp.py` + `src/polar/paperstyle.py`. 본문 7장 + Table 1 + 부록 10장·보조표 17개, 그림마다 시각·과학 검토 2회 이상.

## 2. Context
- 전신: `gpt/handoff/20260926_1200-final-paper-plan.md`(계획 확정). 사용자 지시(09-26 저녁): 실험까지 이번 세션에 실행, GPU 5–9(9번부터, 8번 타 사용자 확인 후 제외), 시각화 개선 에이전트와 실험 과정·결과 검토 에이전트 병렬, Fable 한도 시 Opus 5.5 로 계속.

## 3. What we did
- 스크립트 신규: `scripts/3_deep_learning/h31_min_labels_residual.py`(A2), `h33_partial_pooling.py`(B2), `h34_label_design.py`(B3), `h35_tfm_fewshot.py`(B4, GPU 9), `scripts/2_evaluation/h32_two_stage_protocol.py`(B1), `h36_cci_layers.py`(C1), `h37_conformal_hier.py`(C2), `h38_external_holdout.py`(D1·D2), `h4_analysis.py`, `scripts/1_data_prep/fetch_cci_layers.py`(CCI GTD 2003–2021·PFR 1997–2021 다운로드), `src/polar/h4_common.py`, `tests/test_h4_common.py`.
- 수정: `h25_label_budget.py`(블록 SSE 저장·블록 CI·주 정의·--exclude-parent·분할 구조 기록), `h27_breakeven_rule.py`(절단 순위·이진 검정·3라벨 E비), h28·h3_analysis(Holm), h_analysis, m1_analysis(요약 정정).
- 감사: 4렌즈 발견 → 지적별 반박 3표 → 조치 목록(`docs/AUDIT_2026-09-26.md`, 판정 원자료 `data/processed/h4/audit_verdicts_2026-09-26.json`). 과거 문서 정정(09-22 로그·설계서·계획서·핸드오프·메모리).
- 그림: 스펙 `figures/PAPER_FIGURE_REDESIGN_2026-09-26.md`(저널 관행 조사 12편·현행 그림 비평·스타일 시스템·심사 33건 반영). 워크플로 3개(구현·검토, 최종 자료 재작성·교차 일관성, 부록 검토).

## 4. Key numbers (cm; Δ = 방법 − 물리식(원천 E0), 블록 부트스트랩 95 % CI)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| E0 고정 + 잔차 CatBoost, 최소 n*(주 정의, all_blocks) | 레나 / LE-1·LE-2·AL-2·AL-6 / AL-4 / 캐나다·AL-5 | n* | 320 / > 320 / > 160 / 3(n=0 에서 이미 넘음) | h4/a2_minn.csv |
| 분산 − 집중(n=40, 12 대상 평균) | κ 수축 / 오프셋 MLE / E0 고정 | ΔRMSE | −1.14 [−1.76, −0.77] / −1.65 [−2.21, −1.24] / +0.04 [−0.03, 0.09] | h4/a2_spread.csv |
| 수준 지역 E 처리 몫(n=10) | 러시아 W / AL-3 / AL-1 | 비율 | 0.95 / 0.89 / 0.72 | h4/a2_curve.csv |
| 2단계 프로토콜 대 항상 수축+잔차(n=3) | 14 대상 | 평균 Δ(악화 수) | −0.50(7) 대 −0.60(6) | h4/b1_protocol.csv |
| κ=10 / 오프셋 MLE(n=3, 14 평균) | B2 | ΔRMSE | −0.62 [−0.74, −0.37] / +0.33 [−0.10, 1.15] | h4/b2_regional.csv |
| TabPFN − CatBoost 잔차(AB4) | n=3 / 10 / 40 | ΔRMSE | +0.11 [−0.18, 0.58] / −0.14 [−0.42, 0.36] / −0.33 [−0.54, −0.10] | h4/b4_tests.csv |
| CCI 불확실성 가중 − Stefan | 라벨 0, AB4 | ΔRMSE | −0.43 [−0.84, 0.04](Holm 비유의) | h4/c1_tests.csv |
| 계층 conformal 90 % | 라벨 0, AB4 | 커버리지(폭) | 0.86 [0.80, 0.92] (81 cm); 셀 풀링 0.73, CQR AB4 0.55(6지역 0.64) | h4/c2_coverage.csv |
| 몽골 외부 홀드아웃 | 3라벨 프로토콜 / 오프셋 MLE(탐색) | ΔRMSE | −52.8 [−56.9, −49.2] / −122 [−139, −100] (물리식 271) | h4/d1_summary.csv |
| H25b kmedoid S3 달성(주 정의) | 14 대상 | 수 | 6/14(random 5/14); 오라클 |log E비| 절단 ρ −0.47 (p 0.09) | h3/h27c_* |
| 레나 random S3 | n ≥ 20 | ΔRMSE | −0.62 ~ −1.38(분할 승률 0.67–1.0) | h3/h25b_curve.csv |
| 사전 지정 Stefan + CatBoost λ 0.25 | 라벨 전량 AB4 | ΔRMSE | −0.68 [−1.01, −0.27] (M1 H13 Holm 0.004) | m1/m1_sc_tests.csv |

## 5. Decisions & rationale
- 채점 CI 를 블록 부트스트랩으로 통일하고 손익분기 주 정의를 CI 기반으로 고정(감사 stats:F1·F2). 가설 문구와 판정 기준은 사전 등록 그대로 두고 정의 차이는 계획서 §10 에 기록.
- 수준/구조 분류는 오라클 최소제곱 |log(E_own/E0)| 하나로(수준 ≥ 0.20, 중간 0.15–0.20, 구조 < 0.15).
- D1 은 대상 라벨 정의 교락 때문에 스트레스 검정으로만 쓴다. 오프셋 MLE 행은 탐색.
- 권고 배포 절차(Fig 7a)는 사전 등록 2단계 분기가 아니라 증거가 지지하는 절차(라벨 0: 물리 앵커 + 계층 conformal; 3–10: κ 수축 E + 흩은 라벨 + 원천 잔차; 수백·전량: 사전 지정 레시피).

## 6. Open issues / risks
- 대상 라벨이 원천 1.6만 행에 같은 가중으로 섞여 잔차가 적응하지 못한 것이 F1 의 기제로 보인다. 대상 행 가중 α 를 키운 확인 실험은 하지 않았다(H26 은 불확정).
- 분할 규약(split 0 GroupKFold, 1·2 무작위 탐욕)과 무효 분할(AL-1·AL-4·AL-6)로 일부 대상 CI 가 불안정. h25b 곡선 점 추정은 d_phys_blk 를 써야 한다.
- 러시아 W·E 평가 셀 12–17개. GTNP 러시아 W 신규 4지점 편입은 사용자 결정 대기(v3 미수정).
- C4(InSAR 다중 충실도)와 부록 S10 은 수행하지 않았다.
- 커밋은 로컬만(푸시 안 함).

## 7. Next steps
1. 영문 원고 초안(계획서 §10.3 절 구조, 그림 7 + Table 1).
2. D2 러시아 W 지점 편입·몽골 직접 측정 라벨 여부 결정.
3. 선택 실험: α 확대 잔차, 분할 5회 이상·규약 통일, C4.
4. `git push`.

## 8. Files touched
- 신규: 위 §3 스크립트, `docs/AUDIT_2026-09-26.md`, `figures/PAPER_FIGURE_REDESIGN_2026-09-26.md`, `data/processed/h4/*`(요약·meta·blocksse; 대용량 runs 는 추적 제외), `data/processed/h3/h25b_*·h25x_*·h27b/c/x_*·h29b_*`, `outputs/figures/paper/*`, `src/polar/{h4_common,paperstyle,cvd}.py`.
- 수정: 계획서 §10·개정 이력, `docs/EXPERIMENT_LOG.md`, 09-22 문서들(정정 표기), `SESSION_HANDOFF.md`, `figures/figure_spec.json`, `data/processed/m1/m1_sc_tests.csv`(원본 `_pre0926`), 메모리.
