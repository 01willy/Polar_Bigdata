# Handoff: 논문 완성 통합 실험 계획 확정(사전 등록) + 조사 3건 + 저널 규격 그림 원형
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-26 12:00
**Session focus**: 사용자 지시 "논문 완성을 위한 통합 실험 plan을 이번 세션에서 확정하고 다음 세션에 실행". 조사 에이전트 3건(ALT 전이 문헌 2022–2026, 소수 라벨 전이 ML 기법, Sci Rep 규격)을 통합해 `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md`를 작성했다. 저널 규격 그림 파이프라인 원형 1종을 만들었다. 실험은 돌리지 않았다.
**Author**: Claude Fable 5.1 + 백승원

---

## 1. TL;DR
- 확정 주장 4(순수 ML 붕괴, 증강 사다리, 사전 지정 앵커+잔차 −0.68/−1.14, 손익분기 n ∝ |log E비|), 보강 필요 3(2단계 프로토콜, κ=10 수축의 정식화, 라벨 선택 규칙), 미해결 2(잔차 학습 최소 n, 라벨 0 보정 구간).
- 핵심 신규 실험: **A2 최소 라벨 수**(E 처리 4수준 × 블록 분산 3수준 × n, 교락 제거), **B2 부분 풀링**(log E 오프셋 최대우도·혼합효과 부스팅·PPI++ 구간, κ=10 대체), **C2 계층 conformal**(라벨 0 새 지역의 유효 구간), **D1 외부 홀드아웃**(몽골·중앙아시아 46셀, 미사용 지역, 사전 예측 후 1회 적용).
- 문헌: LORO 전이 오차를 보고한 ALT 매핑 논문 없음. Gautam 2025(Sci Rep)와 방향 일치·차별점 명시 필요. E 공변량 예측 성공 사례 없음. InSAR-ALT 프록시는 알래스카 밖 불안정 → 약라벨은 한계 서술과 부록으로 축소.
- 기법: TFM은 잔차 예측기 한정(라벨 0 이득 없음이 문헌 근거). 무라벨 TTA·재가중·공변량 이동 conformal·잡음 라벨 학습은 기각 목록.
- 그림: 본문 display item 8(그림 6 + 워크플로 1 + 표 1), 부록 S1–S11, 전부 `paper_figs.py`(journal_style)에서 생성. Fig2 원형 완성.

## 2. Context
- 전신 `gpt/handoff/20260922_1930-label-budget-protocol.md`. 사용자 요구(09-26): 잔차 학습 최소 라벨 수의 명확한 조건 분리, 통합 실험 계획 확정(최신 기법 고려), 그림은 Nature/Sci Rep 상위 저널 수준(정갈함·논리·창의성)으로 기존 프로젝트보다 향상.

## 3. What we did
- 조사 3건 실행·통합(원문 확인 수준 표기, 재확인 필요 항목 계획서 §2.4).
- 계획서 작성: §1 주장별 상태표, §2 조사 반영(신규성·기법 우선순위·저널 규격), §3 공통 설계, §4 Track A–E, §5 가설 F1–F11·판정 규칙, §6 실행 순서·자원(약 12–14 h, 벽시계 8–10 h), §7 그림 설계(원칙·본문 8·부록 11·본문 대응), §8 산출물, §9 위험.
- `scripts/4_visualization/paper_figs.py` 신설, `outputs/figures/paper/Fig2_label_budget.{pdf,png}`·`CAPTIONS.md`, 스펙 `figures/figure_spec.json` paper_fig2.

## 4. Key numbers(이번 세션 신규 계산 없음, 계획 근거)
| Item | Value | Source |
|---|---|---|
| 손익분기 n(S3, κ=10) | 러시아 W 3 · AL-1 3 · AL-3 10 · CA-2 40 · 나머지 10 미달성 | h3/h25_breakeven |
| \|log(E_own/E0)\| 상위 | 러시아 W 0.51 · CA-1 0.46 · AL-3 0.30 · CA-2 0.26 · AL-1 0.21 · 러시아 E 0.17 · CA-3 0.17 | h3/h25_targets |
| 구조 지역(≤ 0.13) | LE-1 · Lena · LE-2 · AL-4 · AL-2 · AL-5 · AL-6 · Canada(0.00) | h3/h25_targets |
| 라벨 셀/블록 | 알래스카 13,542/40 · 레나 3,037/20 · 캐나다 742/32 · 러시아 W 31/21 · E 30/21 · 몽골·중앙아시아 46/21(미사용) | fidelity_base_v3 |
| GTNP 신규 지점 | 러시아 5(new_coord), 나머지 72 중복·12 동일 사이트 | gtnp_alt_inventory |

## 5. Decisions & rationale
- 손익분기 주 정의를 "Δ의 CI 상한 < 0인 최소 n"으로 바꾼다(회복률 50 %는 분모 0 문제).
- κ=10 휴리스틱은 B2의 통계 모형으로 대체하고, 이론 필요 라벨 수 n_theory = σ²/(τ²·ε)를 H27 경험 규칙과 대조한다.
- 라벨 0 결론은 "예측은 물리식 수준, 보고할 것은 계층 conformal 구간·AOA 마스크·오차 상한"으로 정한다.
- 약라벨(InSAR)은 본문 한계로, 실험은 부록 C4(캐나다 하위 지역, 여유 시)로 한정.

## 6. Open issues / risks
- GPBoost·TabPFN-2.5 설치·접근 실패 시 대체안(계획서 §9). CCI GTD·PFR 다운로드 실패 시 C1 축소.
- 러시아 W·E 평가 셀 12–17개 불안정, 하위 지역의 불완전 독립성(100 km 버퍼).
- 로컬 커밋만 존재(원격 미푸시). 미추적 대용량 CSV(h25_runs 등) 유지.

## 7. Next steps
계획서 §6 순서. 0 CCI 다운로드 → 1 A2 → 2 B2·B4 → 3 B3 → 4 B1 → 5 C1·C2 → 6 D1·D2 → 7 h4_analysis → 8 paper_figs Fig1–7·표·부록 → 9 검토 에이전트·로그·핸드오프·커밋.

## 8. Files touched
- 신규: `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md`, `scripts/4_visualization/paper_figs.py`, `outputs/figures/paper/*`, `gpt/handoff/20260926_1200-final-paper-plan.md`.
- 수정: `docs/EXPERIMENT_LOG.md`, `SESSION_HANDOFF.md`, `figures/figure_spec.json`, `gpt/handoff/INDEX.md`, 메모리.
