# 영문 원고 4개 절의 수치·주장·규정 점검(front, methods, discussion, legends)

작성 2026-10-04. 대상: `paper/manuscript/en/sections/front.tex`, `methods.tex`, `discussion.tex`, `legends.tex`.
방법: 각 수치의 근거 행을 `paper/claims/*/README.md` 에서 찾고, 그 README 가 적은 원천 CSV·JSON 을 pandas(`OMP_NUM_THREADS=1`, 읽기 전용)로 열어 값·부호·단위·자릿수(지침 4.5)를 대조하였다. 등록 사실은 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md`, `docs/EXPERIMENT_PLAN_WF_2026-10-01.md`, `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md`, `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`, `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md`, `git log`, `configs/rescale/*.yaml`, `src/polar/*.py`, `scripts/4_visualization/paper_v3/*.py` 와 대조하였다. 그림 설명문은 `outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md`, `*_values.txt`, 렌더된 `Fig1–Fig5, Fig7.png` 과 대조하였다.
결과 요약: 점검 행 147개(대조한 수치 약 280개), 수치 정정 7건, 문구 정정 8건, 미해결 7건(사용자 결정·자리표시). 초록 199단어(자리표시 제외).

표기: OK = 원천과 일치. 정정 = 원고를 고쳤다(전 → 후). 결정 = 사용자 결정 대기.

## 1. front.tex

### 1.1 초록(판 B)

| 조각 | 원고 값 | 원천(파일, 행 필터, 열) | 파일 값 | 판정 |
|---|---|---|---|---|
| S3 'held out five regions with 100 km buffers' | 5지역, 100 km | D README 1.3, E24–E25; LG 계획서 §1 | 주 4지역 + Alaska(x), 버퍼 100 km | OK |
| S4 AB3 '−1.6 cm; 95% CI −2.0 to −1.0' | −1.6 [−2.0, −1.0] | `data/processed/lgw/lgw_bundle.csv`, ab=AB3, target=MEAN, delta/ci_lo/ci_hi | −1.629003 [−1.983740, −1.043568] | OK(\|Δ\| ≥ 1 → 소수 1자리, CI 같은 자릿수) |
| S4 AB3 판정어 'lower error' | 우세, Holm 규칙 (a) | 같은 행 verdict4, verdict4_common, holm_p | 우세 / 우세 / 0.001 | OK |
| S4 AB4 '−2.5 cm; −3.0 to −1.9' | −2.5 [−3.0, −1.9] | 같은 파일, ab=AB4, target=MEAN | −2.454509 [−3.038719, −1.883111], holm_p 0.001 | OK |
| S4 AB4 'lower in one region' | 우세 지역 1/4 | 같은 파일, ab=AB4, scope=region, verdict4 | Lena 미결정, Canada 열세, Russia_W 우세, Russia_E 미결정 | OK(J7 규칙 (e)) |
| S4 AB5 'no further difference established for residual ML' | 규칙 (b) | 같은 파일, ab=AB5, target=MEAN, holm_p, abstract_rule | −0.177787, holm_p 0.136, '(b) 초록은 차이를 확인하지 못했다' | OK |
| S5 '−0.87 and −1.3 cm' | WF4-a n 40, 160 | `results/rescale_wf/data/processed/wf/wf_tests.csv`, test_id=WF4-a, scope=MEAN, delta | −0.873044, −1.320595 | OK(\|Δ\| < 1 → 2자리; ≥ 1 → 1자리) |
| S5 'two regions, change from Canada' | 레나·캐나다 풀, 캐나다 우세 | 같은 파일, scope=region, target=Canada\|x, Lena\|x, n 40/160 | Canada −2.095831·−2.715142 우세(공통 CI 도 우세), Lena +0.349743·+0.073952 미결정 | OK |
| S6 '13.9 against 14.4 cm RMSE (1000 labels)' | 13.9 / 14.4 | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`, WF6-a, Alaska\|r, R1(λ cv)-P1\|n1000, rmse_A/rmse_B | 13.867150 / 14.407610 | OK |
| S6 'about 99% ... (all labels) between climate grid cells' | 1 − 0.9 % | `results/rescale_wf3/data/processed/wf/wf3b_decomp.csv`, exp=wf9, base=Alaska\|r, method=R1, lam=−1, n=−1, share_gain_within | 0.008612 → 99.1 % | OK |
| S7 '−2.5 cm (Canada) and +1.7 cm (Lena Delta)' | WF2-a S2−S1 n 100 | `wf_tests.csv`, test_id=WF2-a, contrast=S2-S1\|R1(0.25)\|n100, delta | Canada −2.538178, Lena +1.651555 | OK(−2.5, +1.7) |
| 판정어 범위 | AB 대비 문장(S4)에만 | WRAPUP 1.1, 명세 2.1 | WF 문장(S5–S7)은 'changed', 'reached', 'lay' | OK |
| Δ·CI 쌍 수 | 2(AB3, AB4) | 명세 2.4 | | OK |
| 단어 수(자리표시 [XE], [XC] 제외) | 199 | 명세 ≤ 200 | 이번 재계산 199 | OK |
| 제목 단어 수 | 14 | 명세 1.3 | 14(하이픈 단어 1단어) | OK |
| 핵심어 | 6 | Sci Rep ≤ 6 | 6 | OK |

### 1.2 Code availability(블록 2, Methods M15 와 중복)

| 조각 | 원고 값 | 원천 | 판정 |
|---|---|---|---|
| 스크립트 경로 h40, h42, h41, h54, h39, wf0_reanalysis, x_*.py, tests/, configs/rescale/ | 존재 | `ls` 확인(scripts/3_deep_learning, scripts/2_evaluation, x_*.py 11개, tests 31개, configs/rescale 15개) | OK |
| 패키지 판 CatBoost 1.2.10, scikit-learn 1.9.1, pandas 2.3.3, SciPy 1.17.1, NumPy 2.1.1 | FINAL_BATCH 1절 '패키지 판', `results/rescale_wf*/logs_wf/deps.log` | 같음(3차 Rescale 만 numpy 2.4.6) | OK |
| 'TabPFN v2 and TabICL v2 ... not redistributed' 와 Methods M15 의 'Built with PriorLabs-TabPFN' 문장 | 두 블록이 달랐다 | 설치본 `tabpfn-8.0.7.dist-info/licenses/LICENSE` 10항: 원본·가중치 또는 그것을 담은 제품을 배포·공개할 때 표기 의무. 논문과 저장소는 가중치·원본을 담지 않는다 | 정정(methods.tex 를 front.tex 와 같은 문장으로 통일). 저장소 README 표기 여부는 결정(아래 5절) |

### 1.3 Data availability

| 조각 | 판정 |
|---|---|
| DOI 10.1594/PANGAEA.972777, 973813, 10.3334/ORNLDAAC/2369, 10.24381/cds.68d2bb30, 10.5285/d34330ce…, 10.3334/ORNLDAAC/2004, 10.5270/ESA-c5d3d65 | OK(방법 초안 Data availability 209–215행, `refs_front.bib`, `scripts/0_download/resalt_insar.py` 와 일치) |
| [미확인] 4건(SoilGrids 판, Copernicus DEM 인용 형식, PolSAR DOI, 영구동토 범위 DOI), [DECISION] 3건 | 미해결(자리표시 유지) |

## 2. methods.tex

### 2.1 수치

| 조각 | 원고 값 | 원천 | 파일 값 | 판정 |
|---|---|---|---|---|
| M1 17,572 행 / 17,467 직접 라벨 / 68 CALM 셀 | | `data/processed/fidelity_base_v3_meta.json` n_rows, n_source_changed; macro_table F4_direct 합 | 17572 / 68 / 17467(F4_calm_temp 68) | OK |
| M1 20 지온 유도 셀(15 Alaska, 4 Canada, 1 Tibet) | | `fidelity_base_v4_meta.json` v3_method_check.counts; D README E50 | Alaska\|temperature 15, Canada\|temperature 4, QTP 1 | OK |
| M1 라벨 연도 1990–2024 | | `data/processed/m1/label_year_sensitivity_cells.csv` year_min/year_max; D README 단서 5 | 1990–2024 | OK |
| M1 행 수 30(E Russia)–13,606(Alaska) | | `data/processed/paper_figs/table1_rows.csv` label_rows | 30, 13606 | OK |
| M1 하위 지역 10개, 블록 중심 k-means, 라벨 미사용 | | LG 계획서 22행 | 같음 | OK |
| M1 39.7, 15.1, 8.7 행/1 km 위치 | | `data/processed/lgw/lgw_label_units.csv` rows_per_loc | 39.667638, 15.109453, 8.720930 | OK |
| M1 13,606행, 343 위치, 74 블록 | | `table1_rows.csv` Alaska | 13606 / 343 / 74 | OK |
| M1 새 지역 규칙(1990 이후, 100 km, 약 1 km 셀, 블록 합집합 ≥ 8) | | LG 계획서 752, 783, 785행 | 같음 | OK |
| M1 Central Russia 57행 13블록, Tibet 132행 34블록, North Atlantic 41행 | | `table1_rows.csv` Russia_C_LGD, Tibet_LGD, NAtlantic | 57/13, 132/34, 41 | OK |
| M2 공변량 25 = 6 + 8 + 9 + 2, 33 × 33 창, 2015–2020, SoilGrids 250 m → 약 5 km, CCI 1997–2021 | | `src/polar/fidelity.py` TERRAIN/CLIMATE/SOIL/CCI, `scripts/1_data_prep/terrain_features_dem.py` W = 33, D README 단서 5·6 | 같음 | OK |
| M2 채점 셀 조건(라벨·CCI·토양 도일 TDD 유효) | | `src/polar/m1_core.py` eval_mask | 같음 | OK |
| M3 κ = 10, 민감도 κ ∈ {3, 30}, 기준 λ(잔차 0.25, 직접·물리 입력 1.0), λ ∈ {0.25, 0.5, 1.0} | | LG 계획서 196, 290, 511행; `lg_meta.json` lams | 같음 | OK |
| M3 알래스카 셀 78–94 % | | `data/processed/paper_figs/fig1_source.csv` share_alaska(주 6지역) | 0.780–0.943 | OK |
| M3 E0 범위 | 1.50–1.62 | `fig1_source.csv` E0: 주 6지역 1.590–1.619, Alaska(x) 1.505(원천의 79 % 가 레나) | **정정**: '$E_0$ is close to an Alaskan value (1.50--1.62 …)' → '$E_0$ varies little between these targets (1.59--1.62 …)'. 1.50 은 알래스카를 뺀 원천의 값이라 문장의 논리와 맞지 않았다 |
| M3 연도 정합 도일(2010–2024, 2010 이전 라벨은 2010–2014) | | `scripts/2_evaluation/a2_year_matched_tdd.py` 16–17, 73–74행 | 같음 | OK |
| M3 토양 물성 Stefan 재보정(P1@ed)을 n 40·160 의 강한 기준선으로 | | LG 7.1a(936행); `lgx_tests.csv` L29 P1@ed-P1\|n40/n160 | −1.872572, −1.934447 우세(‡) | OK |
| M4 유사라벨 비율(원천 행 수의 10배), 위약 4종, tddlin 정의 | | LG 계획서 44, 47, 235–239행 | r = 10, shuffle/const_t/const_src/tddlin | OK |
| M4 CatBoost 200회, 학습률 0.05, 깊이 3, seed 2 | | `src/polar/m1_core.py` 192, 289행; `lg_meta.json` seeds [0, 1] | 같음 | OK |
| M4 학습기 10종 | | C1 README 2.1; `lgt_tests.csv`, `lgf_tests.csv`, `lgfn_tests.csv` | 10종 | OK |
| M5 대상 27 = 6 + 1 + 10 × 2, 분할 5, 추출 5, 격자 {0, 3, 10, 40, 160, 320, 1000, 전량} | | `results/rescale_lg/data/processed/lg/lg_meta.json` targets, splits, draws, n_grid | 27, 5, 5, 같은 격자 | OK |
| M5 A/B 분할 규칙(블록 순열, 셀 수가 적은 쪽) | | `src/polar/m1_core.py` half_split_blocks | 같음 | OK |
| M5 러시아 W·E A 셀 14–17 | | `lg_targets.csv` part=cpu, mode=x, valid: n_A | Russia_W 14–16, Russia_E 15–17 | OK |
| M5 티베트 고도 | '위 the elevation range of the source' | LG 계획서 897, 1114행: 원천 0.5–99.5 백분위 0.3–905.3 m, 티베트 3,537.5–5,176.0 m | **정정**: 'above the elevation range of the source' → 'above the 99.5th percentile of source-cell elevation' |
| M5 검증 사다리 8단, 5지역 | | C7 README 2.1; `lgv_metrics.csv` scheme 8종 | V-R, V-P, V-S, V-B, V-C0, V-C100, V-C500, V-G | OK |
| M6 첫 시험(분할 5, n ≥ 1000, Pbest, SAR 9열)과 재시험의 네 변경, 분할 25(레나 24) | | C4 README 1.2 (c); `wf2b_tests.csv` n_splits_expected | 25 / 24 | OK |
| M6 유사라벨 10 × 라벨 수(지역 내), 사전값 = 원천 E0, λ 블록 교차검증 | | WF 계획서 208행, 32행 | 같음 | OK |
| M6 조정 Rand 0.999 / 0.997 / 0.832 | | WF 계획서 개정 이력 243행(산문, CSV 없음); C8 README 5.1 의 1 | 같음 | OK(산문 출처) |
| M7 전략 7종, n 20–500(지역 내), 전이 n 10·40(S2·S4), L43 결과 전 등록 | | C6 README 3절, 2.5; WRAPUP 4절 | 같음 | OK |
| M7 Algorithm P 의사코드, 후보 γ 0/0.5/1, S8b, S2, S4, 알래스카 계열 선택 | | FINAL_BATCH 2.4 S8 정의, 2.3 '고정 규칙' | 같음 | OK |
| M8 규칙 W 5단계, 후보 7개, 묶음 min(5, 블록), n < 10 → P1, 동률 규칙 | | C5 README 1.4; WF 계획서 58행 | 같음 | OK |
| M8 n {10, 40, 160, 전량}, 대상 30 = 27 + 3, 비열등 한계 0.5 cm | | C5 README 3절; `wf_meta.json` ni_margin | 같음 | OK |
| M8 진단값 정의, 28대상 | | `wf_tests.csv` WF4-c MEAN n_targets | 28 | OK |
| M9 재표집 횟수 | '라벨 격자 주 가설 1000, 나머지 10,000' | `lg_meta.json` nboot 1000; `lgd_meta.json` nboot L1e_L4e_L8e 1000; `lgu_a_meta.json`, `lgu_b_meta.json` nboot 1000; LGX·LGT·LGF·WF·AB 10,000 | **정정**: 'The main hypotheses of the label-grid experiment used 1000 replicates' → 라벨 격자 하네스의 가설(본 실험·새 지역 실행)과 예측 구간 실험 1000, 나머지 10,000 |
| M9 2단 재표집의 적용 범위 | '배치 대비(as in placement)' | C6 README 5.1 의 5(WF2·WF8 CI 는 추출 조건부 `h40.contrast`); `lgw_l43.csv` ci_kind '2단 재표집'; FINAL_BATCH 1절(XC·XD 2단) | **정정**: 2단 재표집은 L43 과 워크플로 분석(XC·XD)에, 배치 시험(WF2·WF8)은 추출 조건부 CI 로 고쳤다 |
| M9 동등 한계 0.5 cm ≈ 1–2 %(21.7–43.0 cm), 보조 1.0 cm 와 2 % | | `lg_curve.csv` P0 n 0 rmse 21.69–42.98(D README E39); WRAPUP 652행 δ_rel = 0.02 × P0 RMSE | 1.2–2.3 % | OK |
| M9 n* 정의 | '원천 계수 대비 우세 … 분할 2/3, 추출 75 %' | `src/polar/h4_common.py` min_n(ci_hi < 0 & split_win ≥ 2/3 & rep_win ≥ 0.75); Fig 4a 는 P1 − P0 와 R1 − P1 에 적용 | **정정**: 특정 대비(원천 계수 대비)로 한정한 문장을 대비 일반 정의(셀 가중 CI 상한 < 0)로 고쳤다 |
| M9 α 0.05 양측, Holm m = 10, 규칙 (a)–(e) | | WRAPUP 1.1 | 같음 | OK |
| M9 블록 수 범위 | '20 to 74 per region' | `table1_rows.csv` blocks: Lena 20, Canada 39, Russia_W 21, Russia_E 21, Alaska 74, Russia_C_LGD 13, Tibet_LGD 34 | **정정**: 풀 지역이 4–7개이면 최소는 러시아 C 확충판 13 → '13 to 74' |
| M9 분할 50회, 부호 검정, 부호 뒤집기, Hartung–Knapp, 사인 검정 최소 p 0.125 | | LG 계획서 309, 327행; 2 × 0.5⁴ | 같음 | OK |
| M9 오차 하한 셀 13,333 | | `data/processed/lgx/lgx_floor.csv` Alaska eval n_cells_used | 13333 | OK |
| M9 계층 conformal 집단 2–4개 | | LGU 계획서 68, 99행 | n 0 은 4개, n ≥ 40 은 2개 | OK |
| M10 커밋·시각 40be64c 09-29 15:45:46, 316714c 09-30 03:02:42, 4168331, 1b42b4d, 8180632, 7da0064, 2678100 10-04 14:22:39, ff1be02 09-30 11:44:30 | | `git log --format=%cd`(KST) | 모두 일치 | OK |
| M10 첫 회수 07:20:22 | | `docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md` 8, 91행; LG 개정 15 (l) | 같음 | OK |
| M10 이탈 10건 | | 명세 6.10 목록 1–10 | 10건 모두 대응 | OK |
| M11 XC 가설 8개(1b, 1c, 2a–c, 5b, 5c, F3), 마지막만 맹검 | | FINAL_BATCH 2.3 주 가설 표 | 같음 | OK |
| M11 XG 대비 6개, 마스크 5·25 km; XE 입력 10–500 m; XA 계열 7개·CE1/CE2 | | FINAL_BATCH 2.7, 2.5, 2.1 | 같음 | OK |
| M12 Rescale 노드(64코어·T4 4장; elm Milan 96코어; hematite Milan-X 64코어), RTX 3090 | | `configs/rescale/lg_full.yaml`, `lgx_a.yaml`, `wf_full.yaml`, `wf3_full_hematite.yaml`; `lgt_meta.json` env | 같음 | OK |
| M12 CatBoost 1.2.10 전 플랫폼 | | `lgt_meta.json` env, `lgd_meta.json` xenv_summary, Rescale 로그(`catboost 1.2.10` 13건) | 같음 | OK |
| M12 3.6e-14 cm / 34,360 키; 11,814 공통 키 ΔSSE 0; FT-T 288 키 중 43, 최대 5.61 cm; ridge 앵커 CatBoost 0.46 cm | | SI_learners README 2.6(R-C14, R-wf9, R-gate-iii, R-gate-i); `data/processed/wf/wf9_repro_check.csv` 74행 ok, n_common 합 11814, max_abs_dsse 0; `lgd_meta.json` by_class catboost max 0.4594965 | 같음 | OK(34,360·5.61 은 README·메타 값, 조각별 표 `lgw_xenv_gate_iii.csv` 존재) |
| M13 지도 투영 중심 경도 | '−45°, EPSG:3413' | `scripts/4_visualization/paper_v3/fig1.py`(LON0_A=None → 레나 확대 중심 126.698° E), `fig5.py`(lon0 −103), `fig7.py`(LON0 −152), `fig6.py`(−45) | **정정**: 그림마다 다르므로 '진척 위도 70° N, 중심 경도는 설명문에 적는다' 로 고쳤다 |
| M13 색표 oslo_r 0.12–0.90, broc, bam, acton_r 0.05–0.95 | | 지침 2.6; `paper_v3/style.py` CMAP | 같음 | OK |
| M15 패키지 판, 경로 | | 1.2 와 같음 | | OK |

### 2.2 규정·문구

| 항목 | 판정 |
|---|---|
| 방법 약호(P0, P1, R1, D0, D1, W)와 실험 번호의 본문 노출 | 등록 소절(Internal pre-registration and deviations, Additional registered analyses) 밖 0건(정규식 점검) | OK |
| 연결어 대시(—, –) | 0건(숫자 범위의 `--` 만) | OK |
| 지침 4.6 금지 표현 | 0건('first' 는 farthest-first·first retrieval 등 일반 용법, 'significant' 는 등록 문구 인용, 'reversed' 는 색표 뒤집기) | OK |
| 'pre-registered' 단독 | 0건(모두 internally pre-registered / internal pre-registration) | OK |
| M14 [DECISION: LLM 사용 기재 문구] | 자리표시 유지 | 미해결(결정) |
| M15 Code availability 가 Methods 마지막 소절 | OK | |
| M12 '추가 분석(XA–XJ)이 Rescale CPU 에서 돌았다' | 실행 전(FINAL_BATCH 1절의 계획 플랫폼). 결과가 나오면 시제·노드 종류를 다시 확인한다 | 미해결(실행 뒤 확인) |

## 3. discussion.tex

| 조각 | 원고 값 | 원천 | 파일 값 | 판정 |
|---|---|---|---|---|
| D1 2.5 cm (95 % CI 1.9 to 3.0; lower error in one of four regions) | | `lgw_bundle.csv` AB4 MEAN, region | −2.454509 [−3.038719, −1.883111]; 우세 1/4 | OK |
| D1 'no further difference was established for the residual' | | AB5 holm_p 0.136 | | OK |
| D1 비열등(한계 0.5 cm) n 40·160, 이득은 캐나다 | | `wf_tests.csv` WF4-a MEAN ni True(n 40, 160), 공통 CI 상한 0.07·−0.29, −0.34·−0.63; Canada 우세 | | OK |
| D1 '3–160 라벨의 권고 방법은 원천 계수 Stefan 보다 점 추정이 높다' | | Fig7_values.txt [5]: R1 n 3 +0.1768, n 10 +0.7814; W n 40 +0.5539, n 160 +0.1849 | 모두 양수 | OK([DECISION Fig 7d] 유지) |
| D1 'about 99 %' | | `wf3b_decomp.csv` share_gain_within 0.008612 | | OK |
| D1 'spreading labels lowered error in Canada and raised it in the Lena Delta' | 단정형 | `wf_tests.csv` WF2-a Lena S2−S1 n 100: verdict4 열세, verdict4_common 미결정(분할 독립 가정 의존); 레나의 다른 n 은 미결정 | **정정**: 'and raised it in the Lena Delta' → 'but not in the Lena Delta'(단정형은 두 CI 판정이 같을 때만) |
| D2 85 % | | AB8 −2.635748, AB9 −0.404108 → (AB8 − AB9)/AB8 = 84.7 % | | OK |
| D2 13.5 of 14.0 cm(W Russia) | | `wf0_misspec.csv` Russia_W\|x P1 −13.498049; `lgw_bundle.csv` AB8 Russia_W −13.980876 | | OK |
| D2 캐나다 계수가 원천과 가장 가깝다·블록 사이 변동 | | C2 README CA6 (`lg_targets.csv` logE_ratio_own 0.003, E_block_cv 0.22); C4 README 1.2 (a) logE_ratio_AB −0.34–+0.35 | | OK |
| D2 +2.3 cm(캐나다, 라벨 10) | | AB4 Canada 2.260737, 열세(공통 CI 도 열세) | | OK |
| D2 잔차 가중 1.0 선택 | | `wf_meta.json` choices(C5 README E29: n 40 22/25, n 160 24/25) | | OK |
| D2 ρ 0.38 (0.17 to 0.55), 28대상 | | `wf_tests.csv` WF4-c MEAN rho 0.380952 [0.166379, 0.545170], n_targets 28 | | OK |
| D3 8 %, 39 %, 56 설정; 4 %, 19 % | | `data/processed/s4_residual_results.csv` D_indomain ridge shared25 lam 0/0.75 seed 0: 14.457/13.330(λ > 0 행 56); `wf2b_tests.csv` E1.R1.n1000 14.407610/13.867150; 하한 11.308101 | 7.80 %, 38.60 %; 3.75 %, 19.17 % | OK(정수 반올림) |
| D3 물리 입력 구조: n ≤ 10 열세, 캐나다 n 40·160 우세 | | `lgx_tests.csv` L10 MEAN·region | E01–E04 우세(공통 CI 도 우세); Canada n 40·160 열세 | OK |
| D3 13,606행, 343 위치 | | `table1_rows.csv` | | OK |
| D3 O'Malley 2026 관측 1–40개 | | `references/INDEX.md` 383행 | | OK(문헌) |
| D4 0.48 cm, 라벨 10 동등 | | `lgx_tests.csv` L15 tddlin n 0 −0.476863 우세, n 10 −0.080129 동등 | | OK |
| D4 0.86 cm(캐나다, 중앙값, 라벨 10) | | `data/processed/lgw/lgw_sens_unit.csv` quantity=unit_p1, n=10, Canada, p50 | −0.863883 | OK |
| D4 단년 라벨(러시아 W·E) | | `lgw_sens_year.csv` n 10 p50 0.046946, 0.036988 | 이득 유지 | OK |
| D4 3.6×10⁻¹⁴ cm, 5.6 cm | | SI_learners README 2.6(R-C14, R-gate-iii 5.61) | | OK |
| D5 10.3 cm (95 % CI 7.3 to 12.8) | 풀 표기 없음 | `data/processed/lgx/ladder/lgv_tests.csv` L19 D0[catboost_lo]:V-G−V-R scope=MEAN5: 10.342619 [7.294513, 12.843602] | **정정**: '(five-region mean; 95 % CI …)' 로 풀을 밝혔다(C7 README 5.2) |
| D5 격자 안 설명 1 % 미만(알래스카) | | `wf3b_decomp.csv` Alaska ML 키 전체 expl_within 최대 0.006738 | | OK |
| D5 0.39–0.54 cm | | `wf2b_tests.csv` WF6-a Alaska R1 n 500/1000/전량 −0.389715/−0.540460/−0.515289 | | OK |
| D5 4 재사용 지역 + 새 얕은 지역 1 | | D README 1.2 의 4 | | OK |
| D5 78–94 % | | `fig1_source.csv` | | OK |
| D6 0.70 cm(레나, 라벨 10) | | `wf_tests.csv` WF4-a Lena\|x n 10 0.696054 열세(공통 CI 도 열세) | | OK |
| D6 72 % | | `wf3b_decomp.csv` Alaska P1 share_within_of_sse 0.718663 | | OK |
| 금지 표현('safe', 'robust', 연결어 대시, 내부 약호) | 0건 | | OK |
| 한계 문단 요소(하지 못하는 것, 시험 안 한 조건, 시험 방식 차이, 측정 안 한 변수, 설계 표지) | 명세 5.1 의 5요소 모두 있음 | | OK |
| 단어 수 | 955(명세 목표 850; D5 216 대 160) | 목표 초과, 상한은 본문 합계 4500 | 미해결(조립 때 조정) |

## 4. legends.tex

### 4.1 수치

| 그림 | 조각 | 원천 | 파일 값 | 판정 |
|---|---|---|---|---|
| Fig 1 | 약관 미확인 셀 Canada 38, North Atlantic 19, W Russia 1 | `data/processed/paper_figs/fig1_not_drawn.csv` | 38, 19, 1 | OK |
| Fig 1 | 지역당 최대 400 셀; 라벨 셀 40개; 100 km; 5분할 중 1 | Fig1_values.txt b·c·d 절 | 400, 40, 99.6 km 축척, 분할 1 | OK |
| Fig 1 | 검증 사다리 3지역 평균 + 지역 값 | FINAL_BATCH 2.8 XH-1 | 3지역 층화 평균 | OK |
| Fig 1 | 투영 중심 경도 | '45° W' | `fig1.py` 728행(LON0_A=None → lon_c), Fig1_values.txt '중심 경도 126.698°' | **정정**: 'central meridian 45°~W' → '127°~E' |
| Fig 1 | 영구동토 구역 [미확인] | Fig1_values.txt: 'ESA CCI Permafrost PFR v4.0, mean of 1997–2021, 0.1°'(fig1.py pfr_classes) | 자료는 확인됨, 인용·DOI 미확인 | 미해결(자리표시 유지, 아래 5절) |
| Fig 2 | 2.5 cm (1.9 to 3.0), Holm 0.001; W Russia 12.0 (10.5 to 13.6); Canada 2.3 (0.7 to 2.9); −0.18 (−0.53 to −0.01), P 0.034, Holm 0.136 | `lgw_bundle.csv` AB4 MEAN/region, AB5 MEAN holm_input_p | −2.454509 [−3.038719, −1.883111]; −11.964734 [−13.619579, −10.525251]; 2.260737 [0.713885, 2.941032]; −0.177787 [−0.526863, −0.014938], 0.034, 0.136 | OK |
| Fig 2 | −12.5 cm(e 축) | `fig2_region_curves.csv` Russia_W R1 n 10 d_p0 | −12.521160 | OK |
| Fig 2 | 재표집 10,000(a, b), 1000(c–f); 분할 5 | `fig2_pool_curves.csv`·`fig2_region_curves.csv` nboot; Fig2_values.txt | 10000 / 1000 / 5 | OK |
| Fig 2 | 마스크 기준 | '14–17(W·E Russia)' | Fig2_values.txt [6]: Canada n > 406 도 마스크 | **정정**: '325--406 in Canada' 를 더했다(렌더 설명문 Fig2_legend.md 와 일치) |
| Fig 3 | 1.6 cm (1.0 to 2.0), Holm 0.001 | `lgw_bundle.csv` AB3; `lgx_tests.csv` L15 shuffle n 0 | −1.629003 [−1.983740, −1.043568] / [−1.985129, −1.035556] | OK |
| Fig 3 | 0.48 cm(n 0), ±0.5 안(n 10) | `lgx_tests.csv` L15 tddlin | −0.476863 우세; −0.080129 동등 | OK |
| Fig 3 | 2.5–3.7 cm(n ≤ 10, 두 가중 우세), Holm 0.001(F1k n 10) | `lgx_tests.csv` L10 R0−F1k\|n0, R0−F1a\|n0, R1−F1k\|n10, R1−F1n\|n10; `lgw_bundle.csv` AB7 | −2.720631, −2.492392, −3.736060, −2.516528(모두 우세, 공통 CI 도 우세); AB7 holm_p 0.001 | OK |
| Fig 3 | 1.9–2.4 cm(n 40·160), 캐나다 유래, 4대비 중 1개 분할 독립 의존 | `lgx_tests.csv` L10 n 40/160 MEAN·region | +2.377200(‡), +1.851063, +2.234208, +1.980894; Canada 열세, Lena 미결정 | OK |
| Fig 3 | 패널 a 지역·계수 | '레나델타' | Fig3_values.txt [6]·[7](1): CONCEPT_REGION = Russia_W, 셀 31, E0 1.589510, 재보정 E 2.109580; 렌더 Fig3.png 확인 | **정정**: 'Concept on Lena Delta labels' → 'Concept on the 31 labelled cells of W Russia … (Source Stefan, 1.59 …; Recalibrated Stefan, 2.11)'. 명세 초안(레나델타)과 다르며 되돌리려면 fig3.py CONCEPT_REGION = 'Lena' |
| Fig 4 | ρ 0.38 (0.17 to 0.55), 28대상, 부호 ρ −0.01; 티베트 227.7 / −159.6 | `wf_tests.csv` WF4-c MEAN, region Tibet_LGD\|x diag_abs, gain_w | 0.380952 [0.166379, 0.545170], −0.013136; 227.712134, 159.551129 | OK |
| Fig 4 | 1.3 cm (0.7 to 1.7) n 160; 0.87 n 40(분할 독립 의존); 전량 비열등 의존; n 10 불성립 | `wf_tests.csv` WF4-a MEAN: n 160 −1.320595 [−1.703930, −0.676553] 우세/우세; n 40 −0.873044 우세/미결정; 전량 ni True, 공통 CI 상한 0.62; n 10 ni False | | OK |
| Fig 4 | Canada −2.1, −2.7; Lena +0.35, +0.07 | WF4-a region n 40/160 | −2.095831, −2.715142; 0.349743, 0.073952 | OK |
| Fig 5 | Canada 2.5–3.2 (n 50·100), 1.3–1.4 (n 200) | `wf_tests.csv` WF2-a Canada S2·S4 | −2.737301, −2.538178, −2.633462, −3.206203; −1.437359, −1.251577 | OK |
| Fig 5 | Lena +1.5 to +2.0 (n 20–500), +1.7 열세(n 100, 의존) | Lena S2 n 20–500: 1.983388, 1.472234, 1.651555, 1.792451, 1.481605; n 100 verdict4 열세 / common 미결정 | | OK |
| Fig 5 | Alaska 블록 등가중 −1.1 to −1.7, 셀 가중 +0.28 to +0.74 | Alaska S2·S4 n 50–200 delta_blockeq −1.30, −1.73, −1.08, −1.15, −1.71, −1.09; delta 0.45, 0.34, 0.61, 0.28, 0.44, 0.74 | | OK |
| Fig 5 | 능동 선정 0.69 to 2.8 cm 열세(Alaska·Lena n 50–200), 6대비 중 1개 의존 | S6 rows 1.877, 1.249, 0.689, 2.122, 2.559, 2.790; Lena n 200 열세/미결정 | | OK |
| Fig 5 | 전이 Canada 2.4 (0.5 to 3.4) 의존; Lena S2 n 10 +1.9 | `wf2b_tests.csv` WF8-a Canada S4 n 40 −2.393681 [−3.400275, −0.489941] 우세/미결정; WF8-b Lena S2 n 10 1.885081 [0.298736, 3.076734] | | OK |
| Fig 5 | 패널 문자 | (a to d) 지도 4, (e) 곡선, (f) 포레스트 | 렌더 Fig5.png·Fig5_values.txt: 지도 3(a–c: Random, Block stratified, Covariate spread), d 곡선, e 포레스트(능동 선정 지도 D-12 미승인) | **정정**: (a to c)/(d)/(e) 와 'In panel d', 'In panel e' 로 고치고 지도 부호(후보 셀 점, 선정 라벨 수 ∝ 원 면적, 40 km 묶음)를 적었다 |
| Fig 6 | 2/30, 18/30; 17대상 | `data/processed/wf/wf0_risk.csv` n 0 D1_1.0 2, D0_1.0 18; 명세 8.3 | | OK |
| Fig 6 | 2.4–5.2 cm(rf, 신경망 4종) | `lgx_tests.csv` L30 rf 2.628285; `lgfn_tests.csv` LGF-N1 2.451820, 2.435923, 4.129612, 5.214119(모두 공통 CI 도 열세) | | OK |
| Fig 6 | CatBoost 2.2 (1.1 to 3.4), Holm 0.023, 분할 독립 의존; Year-matched 1.0 cm 의존 | `lgw_bundle.csv` AB1 2.245922 [1.100411, 3.421232] holm 0.023, common 미결정; `lgx_tests.csv` L29 −0.996848 우세/미결정 | | OK |
| Fig 6 | d 1000회(보정 고정), e 10,000회; 척도화 5종; 띠 0.85–0.95 | `lgu_b_meta.json` nboot 1000(1단 = 보정 고정, LGU 2.7); CAPTIONS v2/Fig6 'a, d 10,000'; `fig6_b.csv` method 5종(cfm 은 SI); `fig6_d.csv` in_band | | OK |
| Fig 7 | 0.54 (0.32 to 0.69), 19 %; 0.39, 0.52 공통 CI 미결정; 레나·캐나다 미결정 | `wf2b_tests.csv` WF6-a Alaska n 1000 −0.540460 [−0.692597, −0.324430] 우세/우세; n 500 −0.389715, 전량 −0.515289 우세/미결정; Lena·Canada 미결정 | 19.17 % | OK |
| Fig 7 | 캐나다 하한선 −10.8 to −10.6 | `lgx_floor.csv` Canada 17.900684 − rmse_B 28.55/28.66 | −10.65, −10.76 | OK |
| Fig 7 | about 99 %, 격자 안 −0.01 동등, 3지역 +0.09 의존 | `wf3b_decomp.csv` 0.008612; `wf3b_tests.csv` WF9-a Alaska~w −0.009052 동등/동등; MEAN 0.091150 열세/미결정 | | OK |
| Fig 7 | b 마스크 규칙 | 'under 10 scoring cells' | `fig7.py` 256행(25분할 합산 cnt < 10, 해칭 //) | **정정**: 'fewer than 10 scoring cells summed over the 25 splits' |
| Fig 7 | d 320·1000 미시험, 레나·캐나다 전이 풀 | Fig7_values.txt [5] | | OK |
| Table 1 | 0.009° 셀 색인, E0 단위 | `scripts/4_visualization/paper/table1_data.py` 84행; `h39_scenarios.py` LOC_DEG | | OK |

### 4.2 규정

| 항목 | 판정 |
|---|---|
| 각 설명문 ≤ 350단어(자리표시 제외, 이번 계산기) | Fig 1 295, Fig 2 302, Fig 3 321, Fig 4 292, Fig 5 348(정정 뒤 줄임), Fig 6 290, Fig 7 310, Table 1 142 | OK(Fig 1·4·6·7 의 X 자리표시 치환분 40–50단어는 명세 예산 안) |
| 첫 문장 = 주장(결과 그림) 또는 명사구(Fig 1, Table 1), 명세 7절과 같음 | OK | |
| 내부 약호·연결어 대시·금지 표현 | 0건 | OK |
| Fig 6 미렌더(v3 폴더에 Fig6 없음) | 설명문은 명세 8.6 기준. 렌더 뒤 패널·투영 재확인 필요 | 미해결 |

## 5. 미해결 항목(사용자 결정 또는 결과 대기)

1. TabPFN 표기 'Built with PriorLabs-TabPFN': 설치본 LICENSE 10항은 원본·가중치 또는 그것을 담은 제품의 배포·공개에 적용된다. 논문 본문에서는 뺐고(front·methods 통일), 저장소 README 표기 여부는 공개 때 결정한다.
2. 자리표시: front.tex [DECISION] 12건·[미확인] 5건, methods.tex [DECISION] 6건·[미확인] 4건·[XD] 1건, discussion.tex [DECISION] 2건·[X?] 5건, legends.tex [X?] 4건·[미확인] 6건. 제출 전 0건이어야 한다.
3. Fig 1 영구동토 바탕 자료: fig1.py 가 쓴 파일은 ESA CCI Permafrost PFR v4.0 의 1997–2021 평균이다. 인용 형식과 DOI 가 정해지면 Fig 1·Fig 6 설명문과 Methods M13 의 [미확인] 을 채운다.
4. Fig 3 패널 a 지역(W Russia 로 렌더)과 Fig 5 패널 구성(지도 3개)은 명세 초안과 다르다. 설명문은 렌더를 따랐다. 명세·그림 가운데 어느 쪽을 바꿀지는 그림 작업의 결정이다.
5. 고찰 단어 수 955(목표 850, D5 216 대 160). 본문 합계 4500 안에서 조립 때 조정한다.
6. Methods M12 의 추가 분석(XA–XJ) 실행 환경 문장은 계획 값이다. 실행 뒤 노드 종류·판을 다시 확인한다.
7. Fig 3c 렌더에서 전량(All) 의 F1k 점이 채운 원으로 보인다(원천 registered False). 설명문 문제는 아니며 그림 쪽 점검 항목이다.

## 6. 집계

- 점검 행 147개(1–4절 표의 행). 대조한 수치(값·CI 끝값·개수·커밋 시각·판 번호)는 약 280개.
- 수치 정정 7건: Methods E0 범위(1.50–1.62 → 1.59–1.62), 블록 수 범위(20–74 → 13–74), 재표집 횟수의 적용 범위(1000 = 라벨 격자 하네스·예측 구간, 10,000 = 나머지), Fig 1 중심 경도(45° W → 127° E), Fig 2 마스크 기준에 캐나다 325–406 추가, Fig 3a 지역·계수(레나델타 → W Russia 31셀, E0 1.59, 재보정 2.11), Fig 7b 마스크 규칙(10셀 미만 → 25분할 합산 10셀 미만).
- 문구 정정 8건: Methods 티베트 고도(99.5 백분위), 2단 재표집의 적용 범위(L43·XC·XD; 배치 시험은 추출 조건부), n* 정의(대비 일반), 지도 투영 문장(중심 경도는 설명문), TabPFN 문장(front 와 통일); 고찰 레나델타 배치 문장(단정형 해제), 10.3 cm 의 풀 표기(five-region mean); Fig 5 패널 문자(a–c, d, e)와 지도 부호.
- 미해결: 7건(5절).
- 초록 단어 수: 199(자리표시 제외). 설명문 단어 수는 4.2 절.
