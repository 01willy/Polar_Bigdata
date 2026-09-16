# Handoff: 논문화 실험 E1–E4 실행 + 기존 결과 대조 감사 + 완전 전이 검증 계획
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-09-15 15:54
**Session focus**: 사전 등록 실험(E1 통합 요인·E2 물리식 사다리·E3 CALM 확충·E4 통계 보완)을 실행하고, 그 결과가 기존 보고서·덱의 주장 8건을 뒤집는지 대조·감사한 뒤, "완전 전이 한계 극복" 의의를 확정할 다음 세션 계획을 사전 등록했다.
**Author**: Claude Fable 5.1 + 01willy

---

## 1. TL;DR
- **기존 결과는 뒤집히지 않았다.** 주장 8건 판정: 뒤집힘 0 · 강화 1 · 정교화 4 · 하향 3. 옛 수치는 같은 조건에서 전부 재현(13.330·14.457 소수 셋째 자리, 등가중 앵커 24.13→23.29, 증강 16.5/30.2, 커버리지 93.4%). 신규 스크립트 5종 적대 감사: 결론 무효화 결함 0.
- **실질 변경 하나**: "CCI 결합 앵커 + 잔차 ML(22.92/21.32)"의 잔차 부분. 185조합 in-sample 최소값이며 중첩 선택(25.49/25.32)·사전 지정 재실행(주 6지역 6/6 악화)에서 이득이 사라진다. 헤드라인은 등가중 앵커 23.27/21.68로 물러난다.
- **해석 정교화**: 증강 순가치 10.8 = 수준 보정 6.2 + 기후 단조 관계 3.6 + √ 함수형 1.0(TDD 선형 대조는 보정된 도일식). 13.33은 앵커 없는 능형 13.62와 −0.29 [−0.61, 0.08]. "Stefan만 정확"은 무보정 비교의 결과이며 같은 k에서는 3지역 기준 최선 또는 동급.
- **데이터 확충**: CALM 비북미 149셀 편입(`fidelity_base_v2.csv`, 기존 불변) → 주 전이 6지역(레나·캐나다·러시아 W/E/C·그린란드) + 심부 4지역. Stefan+CCI 앵커는 10지역 중 8지역에서 물리식보다 낮으나 비유의(p=0.11/0.22).
- **사용자 입장**: 판정을 아직 신뢰하지 않음. 다음 세션은 판정 재확인(V1–V4) → 돌파 후보 총력 시험(T1–T6, 중첩 선택 필수) → 희소 라벨 전이(F1–F3) → 사전 고정 결정 규칙 A/B/C. 정본 `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`.

## 2. Context
- 직전 핸드오프 `gpt/handoff/20260831_1105-final-deck-audit-journal.md`: 본선 덱 확정, 투고지 Sci Rep, P0 = winner's curse 해소·대조군 사다리·관련연구 재배치.
- 사용자 지적 4건(2026-09-08): 워크플로 비정합(증강·잔차·결합이 실험마다 다르게 쓰임), 물리 경험식이 Stefan뿐, 전이 지역 2곳의 일반화, Sci Rep 대비 보완. 이를 E1–E4로 설계해 GPU 2·3·4·5에서 실행.
- 2026-09-14: 사용자가 "기존 결과가 다 틀렸다는 것인가"라고 이의를 제기 → 주장 8건 대조 + 스크립트 5종 감사 워크플로(13 에이전트) 실행. 09-08 로그의 내 서술 5건이 새 결과를 과장한 것으로 확인되어 정정.

## 3. What we did
- **Action**: 사전 등록 계획 작성(주 추정치 H1–H6 고정) · **Files**: `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` · **Result**: 개정 이력 6건(앵커 수준 추가·주 집합 재정의·H2 기각·대조 결과 반영)
- **Action**: E4.1 S12 중첩 선택 재평가(새 실행 없음) · **Files**: `scripts/3_deep_learning/e4_nested_selection.py` → `data/processed/e4_nested_selection.csv` · **Result**: 탐색 최소값 22.92/21.32 → leave-one-target-out 25.49/25.32(물리식 24.11/22.91보다 나쁨). 사전 지정 R2 23.27/21.68·R3 23.00/22.24
- **Action**: E2 물리식 사다리(10계열 × k∈{0,1,2}, 토양 도일 Stefan 신설) · **Files**: `e2_physics_ladder.py`, `scripts/1_data_prep/era5land_soil_tdd.py` → `e2_physics_ladder_results{,_v2main,_v2deep}.csv` · **Result**: 무보정 격차는 배율 편향(c≈0.53–0.71). 3지역: Stefan k=1 최저·k=2 동급, 토양 도일 Stefan 13.65 vs 14.46(CI 0 제외). 7지역 비가중 평균 Ku 우세는 그린란드 3셀 효과(셀 가중 Stefan 16.6 < 19.3). 멱지수 b 0.35–0.42
- **Action**: E3 CALM 비북미 확충 · **Files**: `scripts/1_data_prep/expand_calm_regions.py` → `fidelity_base_v2.csv`(gitignore, 재생성), `fidelity_base_v2_meta.json`, `src/polar/fidelity.py` TRANSFER_MAIN/DEEP · **Result**: 149셀 신규, 기존 17,423행 불변, DEM 66타일 다운로드, SoilGrids WCS 창 신규
- **Action**: E1 통합 요인 설계(3조건 × 앵커 6 × 증강 6 × 잔차 3 × λ 5 × 3 seed, 같은 B블록 셀 + 알래스카 6-fold) · **Files**: `e1_unified_factorial.py`, `e1_analysis.py` → `e1_factorial_shard*.csv`, `e1_prereg_tests.csv`, `e1_table_conditions.csv` · **Wallclock/GPU**: 4 shard × 37분, GPU 2–5, 1,188 적합 · **Result**: §4 표
- **Action**: E1-E3 정보 없음 10지역(알래스카만 학습, 지역 전체 셀) · **Files**: 같은 스크립트 `--noinfo-only --base fidelity_base_v2.csv --tag _e3` → `e1_prereg_tests_e3.csv`, `e1_table_conditions_e3.csv` · **Result**: H1 8/10 방향 일관 비유의, H2 주 6/6 악화(ridge 외삽 폭주 그린란드 +650)
- **Action**: E4.3 CQR 구간 점수 · **Files**: `e4_interval_score.py` → `e4_interval_score.csv`, `e4_conditional_coverage.csv` · **Result**: CQR 68.9 vs 상수 폭 참조 65.4, 셀별 폭과 오차 순위 상관 −0.07, 양극단 5분위 커버리지 0.83–0.85
- **Action**: E4.4 라벨 정의 민감도 · **Files**: `e4_label_sensitivity.py` → `e4_label_sensitivity.csv`, `_meta.json` · **Result**: 알래스카 라벨 86% GPR 유도. E_probe 1.613 vs E_gpr 1.654, 전이 변화 ≤0.5 cm → 결론 불변
- **Action**: 기존 주장 8건 대조 + 스크립트 5종 감사(워크플로 13 에이전트) · **Files**: `docs/RESULTS_RECONCILIATION_2026-09-14.md`(§5 원고 변경 표, §6 정정·후속) · **Result**: §1 TL;DR
- **Action**: 그림 5종 · **Files**: `outputs/figures/e2_physics_ladder/e2_ladder_delta{,_v2main}.png`, `e2_exponent_b*.png`, `outputs/figures/e1_factorial/e1_lambda_by_condition.png`, `e1_control_ladder.png`, `e1_lambda_per_region_e3.png`(figure_spec 5건 등록, 시각 검수 완료)
- **Action**: 다음 세션 계획 사전 등록 · **Files**: `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md` · **Result**: §7
- 커밋: 5488d1a(feat) · 726927d(results) · 292f8ba(viz) · ff6504a(docs). pytest 18 통과.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| Stefan+CCI 앵커 vs Stefan (λ=0) | 정보 없음, 레나·캐나다 전체 셀 | RMSE cm | 24.13 → 23.29 (옛 24.11→23.27 재현) | `e1_prereg_tests_e3.csv` H1, `e4_nested_selection.csv` R1·R2 |
| S12 탐색 최소 → 중첩 선택 | 정보 없음 / 공변량만 | RMSE cm | 22.92 → 25.49 / 21.32 → 25.32 | `e4_nested_selection.csv` R0·R4 |
| 앵커+ridge λ=0.25 − 앵커 (H2) | 정보 없음, 레나 | ΔRMSE | +0.76 [0.16, 1.82]; 주 6지역 6/6 악화 | `e1_prereg_tests.csv`, `e1_prereg_tests_e3.csv` |
| Stefan 유사라벨 − {없음, 상수, 셔플, TDD선형} (H3) | 공변량만, 캐나다 / 레나 | ΔRMSE | −5.35/−10.78/−4.59/−1.00 · −3.29/−1.71/−0.67/−0.10 | `e1_prereg_tests.csv` |
| Stefan+ridge λ=0.75 vs 앵커 없는 ridge | 알래스카 6-fold, 25종, 풀링 | RMSE | 13.330 vs 13.620 (Δ −0.29 [−0.61, 0.08]) | `e1_factorial_shard{0,2}_preds.npz` 재계산(대조 C1) |
| Stefan+CCI(무보정 등가중) − Stefan (H6) | 알래스카 지역 내 | ΔRMSE | +2.02 [0.47, 3.38]; 보정 CCI 결합은 −0.24 | `e1_prereg_tests.csv`, 감사 A2 |
| 토양 도일 Stefan k=1 vs Stefan | 알래스카 6-fold / LORO 3지역 | RMSE | 13.65 vs 14.46 (Δ −0.81 [−1.33, −0.19]) / 20.81 vs 21.26 | `e2_physics_ladder_results.csv` |
| Ku k=1·k=2 vs Stefan | 7지역 비가중 / 셀 가중 | RMSE | 30.80·30.66 vs 33.68 / 21.71·19.28 vs 16.57 | `e2_physics_ladder_results_v2main.csv` |
| CatBoost 직접 vs Stefan | 정보 없음, 그린란드 제외 5지역 | RMSE | 38.0 vs 34.5 (6지역 비가중 35.5 vs 36.0은 그린란드 3셀 효과) | `e1_table_conditions_e3.csv` |
| CQR 90% 구간 | 알래스카 S11 OOF | interval score | 68.9 vs 상수 폭 65.4 (raw 123.9) | `e4_interval_score.csv` |
| E 계수 (전체/탐침/GPR) | 알래스카 | – | 1.6167 / 1.6127 / 1.6538 | `e4_label_sensitivity_meta.json` |

## 5. Decisions made
- **원고 헤드라인 교체**: 24.11→22.92 대신 24.11→23.27(등가중 앵커). "+잔차 학습" 행은 보조 표(in-sample 최소·중첩 25.49 병기)로 강등 — 중첩 선택·사전 지정 재실행 모두에서 이득 부재.
- **C1 서술 교체**: 상수 대조 순가치 하나 → 대조군 사다리 분해(수준·단조 관계·√ 함수형). "부정확한 물리"는 "무보정 물리"로.
- **결합 앵커의 처방 범위**: 무보정 등가중 CCI 결합은 전이 전용(지역 내 악화). 보정 CCI 결합은 지역 내 악화 없음, 전이 이득 감소.
- **주 전이 집합 = 6지역**(레나·캐나다·러시아 W/C/E·그린란드), 심부 4지역(스발바르·몽골·알프스·티베트) 분리. 스발바르 이동은 결과 확인 후였으나 H1에 불리한 방향(포함 시 6/7).
- **탐색적 수준 표기**: 토양 도일 Stefan·Kudryavtsev 계열 앵커는 결과 확인 후 추가되었으므로 확인적 주장으로 승격하지 않음.
- **두 번째 의의("완전 전이 한계 극복")는 다음 세션 결정 규칙 A/B/C로 확정**. 사용자 요청.

## 6. Open questions / blockers
- **판정의 독립 재확인**: 사용자가 "잔차 이득 = 선택 효과" 판정을 신뢰하지 않음. 해소 증거: V1(옛 최적 구성 고정 재채점, 신규 4지역 포함), V2(블록 부트스트랩으로 선택 편향 크기 추정, 기준 1.19/1.59), V3(주 6지역 중첩 순환). 둘 이상 "실재"면 철회.
- **데이터 결함(주 집합 결론 불변)**: `Svalbard` 매크로에 노르웨이 본토 2·스웨덴 1 사이트 혼입(진짜 스발바르 146 cm, Stefan RMSE 123). CALM 신규 68셀(알프스·몽골·QTP 전부, 스발바르 5/7, 캐나다 2/10)이 지온 유도 라벨인데 `F4_direct` 태그. `e5_tdd_soil=0` 빙상 격자(그린란드 2셀 등) 미처리.
- **통계 표기**: 6지역 비가중 평균이 3셀·7셀 지역에 지배됨. 셀 가중·중앙값·블록 수 병기, 블록 8개 미만 CI 미산출 규칙 필요. H1 "방향 일관"은 셀 가중 진술(블록 다수결은 레나·캐나다 혼재).
- **사전 등록 이탈 2건**: Stefan+CCI 가중 고정 0.5·무보정(최소제곱 가중이 1.0 클리핑), H3 짝지음은 seed 앙상블 후.
- **S12 원표 결함**: 기계학습 계열이 CCI 결측 1셀(ALT 250 cm) 포함 전체 셀에서 채점, 21.32는 캐나다 단일 seed 구성(완비 최소 21.42).
- **미편입 자료**: `data/raw/gtnp/alt_csv/` GTN-P ALT 106 데이터셋(격자·트랜섹트 탐침)이 어떤 조립 스크립트에도 미참조. CALM 중복 제거 후 전이 지역 확충 후보.

## 7. Next steps (prioritized) — 정본 `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`
1. **데이터 정비(§4)** — Claude, 1일. Svalbard 분리, CALM 관측법 태그(`F4_calm_temp`), `e5_tdd_soil=0` 결측, 분석 스크립트 CI 게이트·셀 가중 열, GTN-P ALT 106 편입 검토.
2. **판정 재확인 V1–V4(§1)** — Claude, 반나일, CPU + GPU 1장. 둘 이상 "실재"면 09-14 판정 철회.
3. **돌파 후보 T1–T4(§2)** — Claude, GPU 2·3·4·5, 1–2일. 앵커+잔차 조건 변경(14종·저용량 트리·FT-T·log 잔차·AOA 게이팅), 보정 다중 앵커 게이팅, 공변량 이동 대응 DL(중요도 가중·DANN·자기훈련), 전수 격자(중첩 채점만 보고). 확인적 H7–H9.
4. **희소 라벨 전이 F1–F3(§3)** — Claude, 반나일. 라벨 n∈{0,5,10,20,40,80} 곡선.
5. **결정 규칙 적용(§5) → 원고 재단** — user + Claude. A/B/C 중 하나로 두 번째 의의 확정 후 `docs/RESULTS_RECONCILIATION_2026-09-14.md` §5 표대로 main.tex 수정, 영문화.
6. **GPT 검토 요청**: (a) V2 선택 편향 부트스트랩 설계의 타당성, (b) T3 공변량 이동 대응 DL 후보 우선순위, (c) Sci Rep 기준에서 결정 규칙 B("소폭 가능성 + 희소 라벨 곡선")의 게재 가능성 판단.

## 8. Pointers
- Authoritative reports: `outputs/report/main.tex`(미수정, 22.92 등 옛 헤드라인 잔존 — 변경 표는 `docs/RESULTS_RECONCILIATION_2026-09-14.md` §5)
- 계획: `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md`(E1–E4, 개정 이력), `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`(다음 세션), `docs/PAPER_PLAN_SCIREP.md` §7(진행표)
- 로그: `docs/EXPERIMENT_LOG.md` 2026-09-08/14/15 항목. 스냅샷: `SESSION_HANDOFF.md`
- 대조·감사 원본: `~/.claude/projects/.../subagents/workflows/wf_24efebe6-d79/journal.jsonl`(에이전트 13개 전문)
- Active jobs: none(GPU 2·3 유휴, 4·5 타 사용자 6 GB)
- 재생성 필요 파생(gitignore): `data/processed/fidelity_base_v2.csv`(`expand_calm_regions.py --attach`), `e5_soil_tdd.csv`(`era5land_soil_tdd.py`), `e1_factorial*_preds.npz`
- Related prior handoffs: `20260831_1105-final-deck-audit-journal.md`, `20260728_1007-contest-deliverables-s6-e2.md`

## 9. Caveats for GPT
- 09-08 로그의 다음 서술은 09-14에 정정됨: "7지역 평균 Ku 우세"(비가중·그린란드 효과), "순수 CatBoost ≈ Stefan"(그린란드 효과, 5지역 전부 ML 열세), "토양 도일 Stefan 확인 실패"(TDD_stl1=0 아티팩트, 셀 가중 최저), "CCI 결합 지역 내 해"(무보정 등가중 한정), "√ 고유 기여"(→ √ 함수형 추가 기여). 정정본만 인용할 것.
- E1 평가 셀: 태그 없음 = 레나·캐나다 B블록 ∩ CCI·토양도일 유효(1,496·370). `_e3` = 알래스카만 학습, 대상 전체 셀. 옛 tab:s12 '정보 없음'은 전체 셀, '공변량만'은 B블록. 조건 다른 수치를 한 축에 놓지 말 것.
- `e1_summary.csv`는 fold 평균 집계(13.61), 옛 13.33은 풀링 집계. 같은 실험.
- 검증 3조건 정식명: 라벨 있음(지역 내) / 공변량만 / 정보 없음. 용어 정본: 물리 잔차 결합·위성 제품(ESA CCI)·레나델타·증강 비율 r·활동층(ALT). 알래스카 13,606셀 = 0.5° 블록 74개.
- Ku·토양 도일 계열 앵커는 탐색적(사후 추가). 확인적 검정은 H1–H6(09-08 사전 등록)과 H7–H9(09-15 사전 등록)만.
- `submission/`은 7/30 예선 동결본, 미추적·수정 금지.


## 10. Addendum (2026-09-16)
- 사용자 결정: 주 축 = 공변량만(배포 조건), 정보 없음 = 비적응 기준선. "완전 전이" 프레임에 매몰하지 않음.
- 다음 세션 정본이 `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`로 바뀜: 세 조건 × 여섯 축(물리식·증강·잔차·ML/DL·학습 지역 집합·앵커 제품)을 단일 하네스로 전면 재실행. S-B 재현 게이트(기존 표 수치 0.1 cm 재현)가 선행 조건. 09-15 계획(V1–V4·T0–T6·F1–F3)은 하위 단계.
- 새 확인적 가설 H12(공변량만에서 앵커+증강+잔차 중첩 레시피 > Stefan+CCI 앵커), H13(라벨 있음 잔차 이득), H14(학습 지역 확대 무효 재검).
- GPT 검토 요청 추가: S-C/S-D 분수 요인 설계의 축별 수준 선택이 적절한지, 중첩 선택(5→1) 외에 더 나은 선택 편향 통제가 있는지.
