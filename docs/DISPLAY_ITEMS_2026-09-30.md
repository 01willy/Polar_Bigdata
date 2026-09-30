# 본문 표시 항목 대응표와 LGF-F1·N1 배치 결정 (2026-09-30, F1)

**성격**: `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` 3.5 F1 의 산출이다. 같은 계획 7절의 대응표 초안을 확정하고 `figures/figure_spec.json` 에 기록했다. 가설, 판정 문구, 범위, 확인적 가설 집합은 바꾸지 않는다.
**열람 상태**: 이 문서와 스펙을 쓰면서 LG, LGX, LGT, LGU, LGF, LGD, LGW 의 결과 표와 조각, `results/rescale_*`, 봉인 폴더를 열지 않았다. 원천 표 이름은 하네스 코드의 산출 이름으로 확인했다.
**정본**: 패널별 원천 표, 필터, 축 변수와 단위, 색표, 내보내기, QA 는 `figures/figure_spec.json` 의 `display_items`, `conditional_rules`, `si_items`, `colormap_rules`, `export_targets`, `qa_checklist` 다. 이 문서는 요지와 결정 근거만 적는다.

---

## 1. 본문 표시 항목(8개)

Scientific Reports 의 본문 표시 항목 상한은 8개다. 구성은 Fig 1–7 과 Table 1 이며, 조건부 규칙이 서도 항목 수는 늘지 않는다.

| 항목 | 제목 | 주장·틀 | 가설·대비 | 주 원천 표 | SI 로 가는 것 |
|---|---|---|---|---|---|
| Fig 1 | 문제 정의와 자료 범위 | R1–R5 의 전제, 틀 (1)–(3) | 없음(서술) | `fidelity_base_v4.csv`, `fidelity_base_v4_labels.csv`, `lgd_eligibility_v1.csv`, `cci_pfr_mean_1997_2021.nc`, `$LGS/lg_targets.csv` | 자료원별 세부, 산점, 하위 지역 지도 |
| Fig 2 | 라벨 수 곡선 Δ(n) | R2(R1, R3 의 양 끝), 틀 (1)(2) | L1, L2, L4, AB6. 조건부 곡선 L10, L30, L34(a), LGF-F1, LGF-N1 | `$LGS/lg_curve.csv`, `lg_tests.csv`, `lgx/lgx_lg_aux.csv`, 신규 `paper/pool_fixed_curve.csv` | 27대상 전체 곡선, P1 기준선 대비 곡선 |
| Fig 3 | 물리 정보의 사용 | R3, R2, 틀 (1) | L15, L10, L12(확인적), L17, L8, L31 표지, LGX N2 띠, AB3·AB4·AB7·AB8·AB9 | `lgx/lgx_tests.csv`, `lg_tests.csv`, `lgx_splitdist.csv`, `lgx_region_inference.csv`, `lgx_floor.csv`, `lgw/lgw_bundle.csv` | L16, L18, L7, L13, L27, L28, L9, L14 |
| Fig 4 | 최소 라벨 수와 대상별 분포 | R2, R3, 틀 (2) | L4, L3, AB5, n* 구간 보고, NOVELTY N1 대상별 요약. 조건부 C12 | `$LGS/lg_minn.csv`, `lg_tests.csv`, `lgw/lgw_aux4.csv`, 신규 `paper/n1_target_summary.csv` | 방법 × n 열지도, N1 수치 표 |
| Fig 5 | 관측 배치 | R4, 틀 (2) | L43, L23, L24 | `lgw/lgw_l43.csv`, `lgw_tests.csv`, `lgx/lgx_tests.csv`, `lgx_distance.csv` | L43 보조, 지역 행, LGU-C3 |
| Fig 6 | 라벨 0 전이와 불확실성 | R1, R5, 틀 (3) | L1·AB1, L29·AB2, L30, L11, L34(a), LGF-F1, LGF-N1, LGU-B2, LGU-A1·AB10, L25, C2 참조. 조건부 지도 | `lgw_bundle.csv`, `lgx_tests.csv`, `lgt/lgt_tests.csv`, `lgf/lgf_tests.csv`, `lgf/lgfn_tests.csv`, `lgu/lgu_b_intervals.csv`, `lgu_b_tests.csv`, `lgu_a_tests.csv`, `lgx_conformal.csv`, `map_lena/lena_pred_v1.csv.gz` | 지역 행, L29 전체, LGT, LGF 세부, LGU 나머지, 지도 |
| Fig 7 | 배포 절차와 초록 주 대비 | R1–R5 요약, 틀 (2) | SC1w, SC1w-P, SC2w, SC3w, AB1–AB10 | `lgw/lgw_scenarios.csv`, `lgw_scenarios_summary.csv`, `lgw_sc3.csv`, `lgw_tests.csv`, `lgw_bundle.csv` | 칸 단위 표, Holm 두 묶음 표, δ_rel |
| Table 1 | 자료와 대상 | 자료 범위 | 없음(서술) | `lgw/lgw_label_units.csv`, `$LGS/lg_targets.csv`, `fidelity_base_v4_labels.csv`, `lgd_eligibility_v1.csv` | 대상 × 분할 표, 자료원별 세부 |

- `$LGS` 는 `logs/post/lgs_dir` 에 적힌 LG 표 폴더다. `data/processed/lg` 에는 적합 수 표만 있으므로 그림 모듈은 이 폴더를 읽지 않는다.
- 판정 기호는 두 가중(셀 가중, 블록 등가중) CI 의 4분 판정이고, 그림의 CI 는 셀 가중이다. 블록 등가중 값은 SI 표에 싣는다.
- 모든 그림은 폭 180 mm, 높이 200 mm 이하, 패널 6개 이하, 범례 1개다. PDF(Type 42), SVG, 600 dpi PNG 와 그림 값 CSV 를 낸다.
- 색표: 비음수 ALT 는 `cmc.oslo_r`(지각 균등 순차형, 공통 고정 범위), 구간 폭과 오차는 `cmc.acton`, 부호 있는 차이는 `cmc.broc`(0 중심 발산형, TwoSlopeNorm)이다. 4분 판정은 모양과 채움 기호로 표시하고 색만으로 구분하지 않는다.

## 2. 조건부 규칙

| 규칙 | 내용 |
|---|---|
| Fig 2 두 판 | 풀 평균은 구성을 고정한 두 판을 모두 싣는다. E1 은 주 4지역(P4) 층화 평균의 n ∈ {0, 3, 10}, E2 는 레나·캐나다 2지역 층화 평균의 전 n 이다. n ≥ 40 에서 러시아 W·E 가 빠지며 생기는 구성 변화를 곡선에 섞지 않기 위해서다 |
| Fig 2 곡선 추가 | L10 에 열세, L30 에 우세, L34(a) 우세, LGF-F1 기각, LGF-N1 기각이 있으면 해당 곡선을 더한다. 로컬 곡선에는 플랫폼 표지와 같은 로컬 조각의 비교 곡선을 함께 넣는다. 범례 8항목을 넘는 곡선은 L10 > L30 > L34 > F1 > N1 순으로 넣고 나머지는 SI 짝 그림에 둔다 |
| Fig 4 C12 | J1 에서 L4 의 최소 n 이 160 이면 C12(n = 80, `data/processed/lg_n80_local`)를 돌리고 n = 80 점을 '교차 환경' 표지와 함께 더한다. 아니면 구간 보고('40 초과 160 이하' 형식)만 쓴다 |
| Fig 6 지도 병합 | 세 조건을 모두 만족할 때만 지도 3장(ALT 라벨 0, 90 % 구간 폭, 외삽 표시)을 Fig 6e 로 둔다. (1) `lgu_b_tests.csv` 의 LGU-B2 판정이 '전이'(네 조건 모두), (2) 지도 구간이 nflow σ(격자 σ 파일)로 만들어짐, (3) 지도 QA 통과. 그 밖은 지도 6패널을 SI 1장으로 둔다. (1)만 만족하고 σ 파일이 없으면 SI 로 두고 본문 배치를 결과 뒤에 다시 정하지 않는다 |
| Fig 6a 행 고정 | 행 목록을 결과 전에 고정한다. 판정 불가나 부분 판정인 행도 지우지 않고 상태를 적는다 |
| L29 P* | L29 가 P* 를 정하면 Fig 6a 에 P* − P0, Fig 3c 에 R1 − P*(전량) 행을 더한다. Fig 2 의 0선은 P0 로 둔다 |
| SD/SE 비 | 한 지역·대비에 'SD/SE 비 과대'와 L31 형식 표지가 함께 서면 Fig 3c 의 그 행은 LGX-N4 지역 수준 값을 주 점으로 쓴다(WRAPUP 1.6) |
| LGD 두 판 | 새 지역 결과는 전체판과 약관 확인분 판을 병기한다. Fig 1a 와 Table 1 의 공개판은 `lic_unverified = 1` 셀을 그리지 않고 수만 각주로 적는다 |

## 3. LGF-F1·N1 의 배치 결정

**질문**: 7절 끝의 점검 항목이다. LGF-F1(TabICL v2 라벨 0 직접 예측)과 LGF-N1(조정한 판별 신경망 4종의 라벨 0 직접 예측)은 확인적 가설이고 R1 의 근거인데 본문 항목이 없었다.

**선택지**

| 선택지 | 내용 |
|---|---|
| A | Fig 6a(라벨 0 포레스트)에 학습기별 요약 행으로 넣는다. 지역 행, 보조 행, F2–F6, N2–N4 는 SI |
| B | SI 에만 두고 본문 문장에서 인용한다 |

**결정**: A 를 택한다. 본문 표시 항목은 8개 그대로다.

**근거**

1. R1 은 본문 주장이고, 계획 1.5 표는 그 근거로 LG L1, LGX L30(확인적), LGT L34, LGF-F1·N1(확인적)을 든다. 7절 초안에는 L30, F1, N1 세 확인적 가설 모두 본문 위치가 없었다. 초록의 한정어('시험한 학습기 k종에서', WRAPUP 1.1)를 본문 그림에서 확인할 수 있어야 한다.
2. 행으로 넣으므로 항목 수가 늘지 않는다.
3. LGF 2.2 의 플랫폼 규칙을 지킨다. 각 행은 같은 플랫폼 조각 안의 대비(학습기 − P0)이고, 로컬 비교 행(`catboost_ctx`, 로컬 기본판)을 같은 묶음에 둔다. Rescale 묶음과 로컬 묶음 사이의 차는 그리거나 계산하지 않는다.
4. 행 목록을 결과 전에 고정하므로 판정에 따라 본문 배치가 바뀌지 않는다.
5. 심사 질문('표형 파운데이션 모델을 시험했는가', '신경망이 조정되지 않았다', LGF 1.3)에 본문 그림으로 답한다.

**비용과 대응**

- LGF 표는 창 마감(10-02 05:01) 뒤 C9 에서 나온다. Fig 6 최종판은 C9 뒤로 두며, 이는 F3 일정(10-02–10-03)과 맞는다. 그 전 초안은 LGF 행을 pending 으로 두고 파일 이름에 `_draft` 를 붙인다.
- N1 이 GPU 상실로 '판정 불가(분할 k/K)'가 되면 행을 지우거나 SI 로 옮기지 않고 상태를 적는다.
- 초록에서는 F1·N1 에 판정어를 붙이지 않는다(WRAPUP 1.1). 그림의 판정 기호는 LGF 2.2 의 두 갈래('지지(물리식보다 오차가 크거나 구별되지 않음)', '지지(우세 근거 없음)')를 구분한다.
- F1 이나 N1 이 기각이면 Fig 2 에 곡선을 더하는 규칙은 이 결정과 별도로 적용한다.

**Fig 6a 의 고정 행**(n = 0, P4 층화 평균, 대비 = 방법 − P0)

| 묶음 | 행 |
|---|---|
| 기준선(Rescale) | B:ens(AB2, L29), P*(L29 가 정할 때만) |
| 직접 ML(Rescale) | CatBoost 기본(L1, AB1), catboost·rf·catboost_tuned(L30), F1k(L11) |
| 직접 ML(로컬, LGT) | TabPFN v2(L34(a)), catboost_ctx(병기) |
| 직접 ML(로컬, LGF-F) | TabICL v2(F1, 컨텍스트 10,000행과 원천 전체), catboost_ctx(병기, 두 조건) |
| 직접 ML(로컬, LGF-N) | 조정한 MLP·TabM·FT-T·RealMLP(N1), 로컬 기본판(N1 보조 (a)) |

## 4. 7절 초안과 달라진 점

| 변경 | 근거 |
|---|---|
| L10·L12 대비 값을 Fig 2 에서 Fig 3b 로 옮겼다. Fig 2 에는 L10 기각 시 곡선 추가 규칙만 남는다 | Fig 2 는 지역 곡선 4개와 풀 판 2개로 패널 6개(재설계 스펙 5.2 상한)다. 결합 방식 비교는 틀 (1)의 주제다 |
| L3 은 Fig 4c 에만 둔다 | L3 의 판정량이 n* 다. Fig 2 는 α = 1 곡선만 그린다 |
| L29·L30·L11 을 Fig 6a 에 두었다 | 초안에 위치가 없었다. 모두 라벨 0 대비다 |
| LGF-F1·N1, L34(a) 요약 행을 Fig 6a 에 두었다 | 3절 결정 |
| Fig 6 행의 'AB4(라벨 0)'를 'P4 풀(주 4지역)의 라벨 0 대비'로 바꾸었다 | 옛 그림 코드의 `AB4`(주 4지역 목록)와 WRAPUP 1.1 의 대비 AB4(P1 − P0, n = 10)가 같은 이름이다. 대비 AB4 는 Fig 3d·Fig 7d 에 둔다 |
| Fig 4d 에 대상별 4분 판정 수 분포(NOVELTY N1)를 두었다 | RESEARCH_FRAME A.3 의 Fig 4 구성('개선/악화 분포')과 F1 의 'N1 대상별 요약 계산 위치' |
| L17 은 Fig 3b, L16·L18 은 SI | L17 은 결합 구조 해석에 직접 쓰인다 |
| L31 은 Fig 3c 표지와 SI, AK1w·L6·L22 는 SI 한 표 | 초안 누락 보완. WRAPUP 5절의 원고 규칙 |

## 5. 가설 대응 점검

7절 초안에서 위치가 없던 id 와 이번 배정은 다음과 같다. 전체 대응은 스펙의 `hypothesis_coverage` 에 있다.

| id | 지위 | 배정 |
|---|---|---|
| L29 | LGX 확인적 | Fig 6a, SI 기준선 사다리 |
| L30 | LGX 확인적 | Fig 6a, Fig 2 조건부 곡선 |
| L11 | 보조 | Fig 6a |
| L31, LGX N2(오차 하한), LGX N4a | 보조·보고 규칙 | Fig 3c 표지·띠, SI |
| L6, L22, AK1w | 보조 | SI 한 표 |
| L7, L13 | 보조 | SI 계수 모형 |
| L9, L14 | 보조 | SI 중복성 표 |
| L1e, L4e, L8e | LGD 재계산 | SI(초안은 PE1·PE2 로만 표기) |
| SC1w-P | 보조 | Fig 7b(초안은 SC1w–SC3w 로만 표기) |
| WRAPUP 1.6 SD/SE 비, 1.7 편향·MAE, 2.4 S-a·S-b, 9.4 δ_rel | 서술 | SI 표. δ_rel 은 Fig 7 캡션 한 문장 |

이 배정 뒤 표시 위치가 없는 등록 id 는 없다.

## 6. 계획과 코드의 불일치

| 번호 | 내용 | 조치 |
|---|---|---|
| 1 | Fig 2·Fig 3d 의 구성 고정 풀 평균 곡선(방법 × n 의 P0 대비 층화 평균)을 내는 표가 코드에 없다. h40 의 `lg_tests.csv` 는 L2, L4, L8 대비에만 층화 평균 행을 둔다 | 신규 모듈 `scripts/4_visualization/paper/pool_fixed_curve.py` → `data/processed/paper/pool_fixed_curve.csv`. 서술 산출이며 판정에 쓰지 않는다. F3 전에 작성한다 |
| 2 | F1 이 요구한 NOVELTY N1 대상별 요약(n 별 평균, 중앙값, 악화 대상 수, 최대 증가)을 계산하는 코드가 없다 | 신규 모듈 `n1_target_summary.py`. `lg_curve.csv` 의 두 가중 CI 에 `h42.verdict4`(δ 0.5 cm)를 적용해 센다 |
| 3 | LGD 두 판(전체, 약관 확인분)의 산출 이름이 코드에 없다(A6 진행 중) | A6 구현 뒤 스펙의 `lgd_editions` 와 `SF_LGD` 에 이름을 적는다 |
| 4 | 격자 σ 파일을 만드는 스크립트(A7)가 아직 없다. `h49_transfer_map --sigma-file` 은 파일이 없으면 멈춘다 | `fig6_map_merge` 조건 (2)로 처리한다 |
| 5 | WRAPUP 6.10 은 지도 산출을 `data/processed/map/` 로 적었으나 코드 기본값은 `data/processed/map_lena/` 다(코드 주석에 사유가 있다) | 스펙은 코드 경로를 쓴다. WRAPUP 개정 때 경로를 고친다 |
| 6 | 판정 표의 열 이름이 하네스마다 다르다. LGU(h44, h45)는 `test`, 나머지는 `test_id`. 층화 평균 행의 scope 는 LG `MEAN4`, LGT·LGF `MEAN` | `_common.py` 의 공통 로더에서 맞춘다 |
| 7 | `paperstyle.save_figure` 의 기본 형식은 PDF·PNG 뿐이다. `fig_map_lena.py` 는 PNG 를 300 dpi 로 쓴다(WRAPUP 6.8) | F3 에서 SVG 를 더하고 지도 PNG 를 600 dpi 로 맞춘다 |
| 8 | `data/processed/lg` 폴더가 있다(적합 수 표 8개). 계획 5.1 은 이 폴더를 LG 표 폴더로 쓰지 않는다 | 그림 모듈은 `$LGS` 만 읽는다 |
| 9 | 작업 지시의 `docs/PAPER_FIGURE_REDESIGN_2026-09-26.md` 는 실제로 `figures/PAPER_FIGURE_REDESIGN_2026-09-26.md` 에 있다 | 스펙은 실제 경로를 적었다 |
| 10 | h48 은 계획 7절에 없는 `lgfn_cross_gate.csv` 를 더 쓴다 | 교차 환경 보조 표로 SI 에만 둔다 |

## 7. 기존 내용의 처리

- `figures/figure_spec.json` 의 2026-09-26 본문 항목 8개(`paper_fig1_problem`–`paper_table1`)는 `figures[]` 에서 빼서 `superseded.display_items_2026_09_26` 에 원문 그대로 옮겼다. `_common.update_figure_spec` 이 같은 id 의 옛 항목과 새 항목을 병합하면 옛 열이 새 기록에 남기 때문이다.
- 옛 SI 항목(`paper_figs*`, `paper_tables*`)은 `figures[]` 에 그대로 두고 처리(재생성 유지 또는 새 SI 항목으로 대체)를 `superseded.si_items_2026_09_26` 에 적었다. 대체되는 옛 결과는 SI 음성 결과 등록표로 모은다.
- `outputs/figures/paper/CAPTIONS.md` 의 옛 절과 옛 그림 파일은 이번 커밋에서 건드리지 않았다. F3 에서 새 이름으로 다시 만든다.

## 8. 남은 위험과 가정

- Fig 6 의 layout_A 는 높이 약 195 mm 로 상한(200 mm)에 가깝다. 지도 행이 들어가면 패널 b–d 를 줄여야 할 수 있다.
- Tibet 은 범북극 극 입체 투영 밖이라 Fig 1a 에 삽도로 둔다. 삽도 크기는 F3 에서 정한다.
- Fig 6a 는 14행 안팎이다. 행 높이 4.4 mm 기준 약 70 mm 가 필요하다.
- Fig 7d 의 AB10 은 구간 점수 대비라 단위(%)와 재표집 횟수(2,000회)가 다른 행(cm, 10,000회)과 다르다. 별도 축과 캡션 문장으로 구분한다.
- 신규 모듈 두 개(1, 2번)는 계산 규칙만 정했고 코드와 시험은 아직 없다.

## 계산 모듈 규칙 확정(2026-09-30 12:35, LG 표 값 열람 전. 열 이름만 확인한 상태)

모듈 `scripts/4_visualization/paper/pool_fixed_curve.py`, `n1_target_summary.py`(커밋 390db15)의 미정 항목을 다음과 같이 정한다. 모두 서술용 그림 규칙이며 등록된 판정을 바꾸지 않는다. 그림의 판정 기호는 `lg_tests`, `lgx_lg_aux`, `lgw_bundle` 에서만 가져온다.

1. 재표집 seed: 하네스 규칙(지역 `seed_of('lgboot', 이름)`, 분할 `seed_of(지역 seed, 분할)`, 공통 `seed_of('lgxboot', 이름)`)을 쓴다. 같은 대비의 `lgx_lg_aux` 행과 값이 같아진다. 명세의 `seed_of('paper-pool', …)` 문구는 이 규칙으로 고친다.
2. 'N1' 은 NOVELTY N1(대상별 4분 판정 수, `lg_curve` 기반)이다. LGF-N1 은 `lgfn_tests` 의 확인적 판정이며 이 모듈의 보조 표(`--aux-curve`)로만 센다.
3. 대상 묶음의 주 묶음은 P4(주 4지역)다. Alaska(x)는 WRAPUP 1.2 에 따라 주 4지역 평균과 합치지 않고 별도 행으로 둔다. P4 + Alaska(x) 묶음은 SI 보조로만 낸다.
4. '악화 대상 수' 의 주 열은 열세 수(n_inferior, 두 가중 CI 기준)다. 점 추정 Δ > 0 인 수(n_increase_point)는 보조 열이다.
5. 러시아 W·E 행이 없는 n ≥ 40 에서는 평균·중앙값을 그리지 않고 대상별 점만 그린다(composition_complete 거짓 표지).
6. 분할이 모자란 풀 행은 값을 남기고 '부분(분할 k/K)' 표지를 단다(LG 판정 세부 규칙의 부분 표기와 같다).
7. E2 의 n 포함 기준은 명세 문구대로 두 지역 각각 한 분할 이상에 키가 있는 n 이다. 모든 분할에 없는 n 은 6의 표지를 단다.
8. 결과 뒤 한 번, 모듈이 읽는 CPU 조각의 P 키 값과 h40 집계(GPU 조각의 같은 키로 덮어쓰기)의 값을 대조한다. 차가 있으면 원인을 적고 h40 집계 값을 쓴다.
9. `figures/figure_spec.json` 의 derived_modules 상태를 '구현(390db15)' 으로 고치고 자원 항목을 스레드 2 로 고친다.
