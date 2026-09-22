# Handoff: 라벨 예산 프로토콜(H25–H30) — 라벨 몇 개면 되는가, 어떤 라벨이 해로운가, 라벨 있는 지역 레시피 확정
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-22 19:30
**Session focus**: 논문 흐름(라벨 있음 → 라벨 희소 → 라벨 0 → 결론)에 맞춰 사전 등록한 H25–H30을 실행. 대상을 하위 지역 포함 15개로 늘려 라벨 수 n에 따른 단계별(E 수축 → 증강 → 잔차 ML) 오차 곡선·손익분기·필요 라벨 수 규칙을 만들고, 라벨 있음 조건 레시피를 중첩 선택으로 확정하며, 블록 라벨 가치와 배포 규칙을 검정. 논문급 그림 9종(시각·과학 검토 반영).
**Author**: Claude Fable 5.1 + 백승원

---

## 1. TL;DR
- **라벨 몇 개면 물리식을 넘는가는 지역의 계수 격차가 결정한다.** 손익분기 n(라벨 전량 이득의 50% 회복·승률 75%)은 러시아 서부 3, AL-1 3, AL-3 10, CA-2 40이고 나머지 10개 대상은 격자 최댓값(10–320)까지 미달성. |log(자체 E/원천 E0)|가 손익분기 n을 Spearman −0.84(p<0.001)로 예측하고, 라벨 없이 계산한 공변량 지표는 예측하지 못한다(LOO R² −0.25). → **2단계 프로토콜**: 대표점 3개로 E비를 재고, 1에서 멀면 E 재적합으로 종료, 가까우면 잔차 학습용 라벨 수십 개를 여러 블록에 분산 배치.
- **라벨의 배치가 수보다 중요하다.** 라벨 전량으로 E를 재적합해도 캐나다 +3.5·CA-3 +5.9 악화(A/B 블록 간 E 이질성). 한 블록에 몰린 라벨은 캐나다 62%·레나 58%에서 해로움. k-중심 선택은 n=3에서 일부 지역(레나·캐나다·러시아 W)만 유리하고 n≥10에서는 무작위가 나은 지역이 많다 → H24 "채택"을 "n=3에서 무작위의 최악 회피"로 축소.
- **라벨 있는 지역 레시피**: 사전 지정 Stefan 앵커 + CatBoost 잔차 λ 0.25/0.5가 −0.68 [−1.01, −0.27] / −1.14 [−1.82, −0.22]로 확정(블록 등가중도 −0.7/−1.0). 중첩 선택 레시피(Kudryavtsev 보정 앵커 + CatBoost)는 셀 가중 −3~−4이나 CI가 0을 포함하고 블록 등가중에서 0 → 집계 의존. CatBoost 대 신경망(TabM) 차이 비유의. 알래스카 지역 내 fold 중첩 −1.10 [−1.89, −0.40].
- **배포 규칙**: AOA 밖 셀에 물리식을 쓰는 잔차 ML 규칙은 직접 ML 대비 −2.52 [−3.97, −1.01](사전 등록 문구), 동일 계열 비교 −1.07 [−1.85, −0.46]이며 물리식과 동급(+0.11). AOA 직접 ML은 물리식보다 +1.56 열세. 대상 라벨 가중 α는 부차 인자(H26 기각).

## 2. Context
- 전신: `gpt/handoff/20260922_1300-h18-h24-transfer-collapse.md`(정보·계수·학습 목표 축 기각). 사용자 지적(09-22): 논문 흐름과 유기적 연결, 라벨 예산별 조정법과 해로운 조합, 라벨 0의 적용법을 밝혀야 좋은 논문. 계획서 `docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md`(§1 논문 흐름 대응표, §7 결과).

## 3. What we did
- **A1·A2 라벨 예산 프로토콜** — `scripts/3_deep_learning/h25_label_budget.py`: 대상 15(주 4 + 하위 11, 블록 중심 k-means), 원천 = 대상 외 전체 라벨(100 km 버퍼), 단계 S1/S2/S3, 규칙 random/kmedoid, n ≤ 320, 분할 3, 66,155행. CPU 6워커 2시간. `--summarize-only` 재집계 모드 추가(라벨 전량 행의 분할별 n 차이 버그 수정).
- **A4 필요 라벨 수 규칙** — `scripts/2_evaluation/h27_breakeven_rule.py`(LOO ridge, |log E비|).
- **A5 블록 라벨 가치** — `scripts/3_deep_learning/h29_block_label_value.py`.
- **B1·B2 레시피 중첩 선택** — GPU 6–9로 라벨 있음 조건 모델 축 분할 1·2 보충(`m1_master_factorial.py --axis model --conds labels --tag h28model`, 7모델 × 3앵커), `scripts/2_evaluation/h28_recipe_nested.py`.
- **A3·C3** — `scripts/2_evaluation/h3_analysis.py`(해로운 조합 통합표, AOA 게이팅 층화 검정).
- **그림** — `scripts/4_visualization/h3_figs.py` → `outputs/figures/h3/` fig00 개요 도식 · fig01 라벨 예산 계단(주 4지역) · fig01b 하위 지역 10개 · fig02 라벨 3·10개 이득 + 손익분기 · fig03 필요 라벨 수 규칙 · fig04 블록 라벨 가치 지도 · fig05 레시피 중첩 선택 · fig06 해로운 조합 열지도 · fig07 배포 규칙. 스펙 `figures/figure_spec.json` h3_*. 시각·과학 검토 에이전트 지적 반영.

## 4. Key numbers (cm; Δ = 방법 − 물리식(원천 E0 Stefan), B블록 채점, 분할·반복 짝지음 CI)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| E 수축 + 잔차 ML, k-중심, n=3 | 러시아 W / AL-1 / AL-3 / CA-2 / 캐나다 / 레나 / 러시아 E / CA-3 | ΔRMSE | −8.7 / −1.6 / −1.8 / −1.5 / −0.8 / −0.7 / +0.2 / +4.3 | h25_curve |
| 같은 방법, n=10 | 같은 순서 | ΔRMSE | −10.5 / −2.7 / −3.8 / +0.7 / −0.2 / +0.6 / −0.3 / +3.3 | h25_curve |
| 손익분기 n(S3, 회복률 50%·승률 75%) | 러시아 W / AL-1 / AL-3 / CA-2 / 나머지 10 | n | 3 / 3 / 10 / 40 / 미달성 | h25_breakeven |
| 손익분기 n 대 \|log E비\| | 14 대상 | Spearman | −0.84 (p<0.001); 단독 LOO R² 0.57 | h27_rule |
| 손익분기 n 대 라벨 미사용 지표 | 14 대상 | LOO R² | −0.25 | h27_rule |
| 라벨 전량(A 전체) E 재적합 + 잔차 | 캐나다 / CA-3 / 레나 / 러시아 W | ΔRMSE | +2.3 / +5.1 / −1.4 / −12.4 | h25_curve(allA) |
| 한 블록 전량 라벨의 가치(E 수축) | 캐나다 / 레나 | 해로운 블록 비율 | 62% / 58% | h29_summary |
| 사전 지정 Stefan+CatBoost λ.25 / .5 − Stefan | 라벨 있음 AB4 | ΔRMSE | −0.68 [−1.01, −0.27] / −1.14 [−1.82, −0.22] | h28_tests |
| 중첩 선택(전체) / CatBoost 계열 / 신경망 계열 − Stefan | 라벨 있음 AB4 | ΔRMSE | −3.07 [−5.36, 1.72] / −4.35 [−6.68, 0.65] / −2.02 [−3.12, 0.03]; 블록 등가중 +0.15 / −0.86 / +0.44 | h28_tests |
| 알래스카 fold 중첩(Stefan+ridge λ.75) − Stefan | 지역 내 | ΔRMSE | −1.10 [−1.89, −0.40] | h28_tests |
| AOA 게이팅 잔차 − 직접 ML / AOA 게이팅 직접 − 직접 ML / AOA 게이팅 잔차 − Stefan | 정보 없음 AB4 | ΔRMSE | −2.52 [−3.97, −1.01] / −1.07 [−1.85, −0.46] / +0.11 [−0.20, 0.34] | h30_deploy_tests |

## 5. Decisions & rationale
- H25 부분 지지(수준 오차 지역만), H26·H27(라벨 미사용)·H29 기각, H28 사전 지정만 확정, H30 부분 지지. 원고 절 2의 축은 "손익분기 n 은 E비로 예측되며 라벨 배치가 수보다 중요하다"로 정한다.
- 관측 설계 규칙(H24)은 축소 서술: k-중심은 n=3에서 무작위의 최악을 피하는 안전장치이고 일반적 우위는 없다. 개선안(군집 크기 가중·계층 수축)은 다음 세션 검정 대상.
- 그림 수치는 채점 셀을 명시하고 절대 RMSE 를 표 간 비교하지 않는다.

## 6. Open issues / risks
- 손익분기 정의(회복률 50% 기준)는 라벨 전량 값이 물리식과 비슷한 지역에서 불안정(분모 ≈ 0). 대안 정의(Δ의 CI 상한 < 0)도 병기했다.
- 하위 지역은 같은 지역 안에서 만든 것이라 완전한 독립 지역이 아니다(원천에 같은 지역의 다른 블록이 포함, 100 km 버퍼만 적용).
- 라벨 있음 조건 러시아 W·E는 평가 셀 12개로 불안정. CatBoost 5반복 × seed 2로 S2·S3 곡선의 잡음이 큼(레나 S2 곡선 등).
- k-중심 선택은 25차원 표준화 공변량의 k-means 대표점이라 군집 크기를 무시한다(등가중 대표). 이것이 n≥10 열세의 원인으로 추정되며 미검정.

## 7. Next steps
1. E 적합 개선 검정: 군집 크기 가중 최소제곱, 라벨 출처 블록 수에 따른 κ 자동(계층 수축) → 캐나다·CA-3형 악화 제거 여부.
2. 2단계 프로토콜(대표점 3개 → E비 → 분기)의 확인적 검정과 라벨 지역 확장.
3. 원고 재단: 절 1 확정 레시피(사전 지정) + 지역 차이, 절 2 라벨 예산 계단·손익분기·해로운 조합·2단계 프로토콜, 절 3 라벨 0 물리 앵커·AOA 게이팅·한계. 논문 2단 폭 그림 판.
4. 이전 M1 요약(m1_analysis.py)에 층화 요약 정정 적용(점 추정치·CI 지역 집합 일치).

## 8. Files touched (this session part)
- 신규: `docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md`, `scripts/3_deep_learning/{h25_label_budget,h29_block_label_value}.py`, `scripts/2_evaluation/{h27_breakeven_rule,h28_recipe_nested,h3_analysis}.py`, `scripts/4_visualization/h3_figs.py`, `data/processed/h3/*`(요약 CSV·meta; h25_runs.csv 10 MB 미추적), `data/processed/m1/h28model_shard*.csv`(+meta; npz 미추적), `outputs/figures/h3/*`.
- 수정: `docs/EXPERIMENT_LOG.md`, `SESSION_HANDOFF.md`, `figures/figure_spec.json`(h3_* 8건).
