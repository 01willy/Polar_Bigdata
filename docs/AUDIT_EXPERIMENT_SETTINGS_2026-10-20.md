# 실험 설정 일관성 감사 (원고 en_v13 기준, 2026-10-20)

목적: 원고(Scientific Reports 초안 v13)에서 같은 그림·같은 문장에 나란히 놓이는 실행들 사이에 설정(라벨 필터, 기후 입력, 특징, 학습기, λ, 씨앗, 분할, 부트스트랩, 지표, 부분집합)이 일치하는지 코드·실행 기록·산출 메타에서 확인한다. 파일은 읽기만 했고 이 보고서 외에는 고치지 않았다.

읽은 자료: `src/polar/testbed_v5.py`, `scripts/3_deep_learning/{v5_compare,v6_label_ladder,v9_extra,v5_map_predict}.py`, `scripts/1_data_prep/{build_v5_labels_cell_year,build_v5_forcing_year,build_v9_external,build_map_grid_alaska_v1,build_map_grid_lena_v1}.py`, `scripts/2_evaluation/{v5_aggregate,v6_label_ladder_agg,v9_extra_agg,v9_extft_agg,v5_products_eval,v5_common_cells,v5_yearwise,v6_numbers,v13_numbers}.py`, `scripts/rescale/{run_v6,run_v6b,run_v6c,run_v9,run_v9b,ingest_v6}.sh`, `src/polar/tab_models.py`, `data/processed/v5/runs/*/run_*.json`·`meta/*.json`·`preds/` 목록, `data/processed/v5/maps/*/meta.json`, `data/processed/v5/grid/*_forcing_meta.json`, `data/processed/map_{alaska,lena}/*_x25_v1_meta.json`, `data/processed/v9_ext/build_meta.json`, `paper/manuscript/en_v13/{sections/methods.tex,sections/results.tex,si.tex}`, `paper/manuscript/shared_v10/numbers_v6.tex`, `SESSION_HANDOFF.md`(Rescale 기록).

심각도 정의: **수정 필요** = 결과 해석이나 원고 진술이 코드와 어긋나 조치가 필요함. **설명 추가** = 결과는 유효하나 원고·캡션·보충에 조건을 적어야 함. **문제 없음** = 확인했고 일치함.

---

## 1. 실행별 설정 표

공통(모든 v6·v9 실행): 라벨 `data/processed/v5/labels_cell_year_v5.csv`, `in_forcing == 1`(2010–2024), 세 지역, 사용 행 = ALT 유한 ∧ √TDD 유한·양수(`testbed_v5.py:150-152`), 19,172 셀-연도(모든 run_*.json의 `rows`). 블록 = 0.5° 격자 번호(`build_v5_labels_cell_year.py:128`). 공간 블록 겹 = 블록을 행 수 균형으로 5겹 배정(`testbed_v5.py:164-176`), 겹 씨앗 0. CatBoost = 600회·학습률 0.03·깊이 6·L2 3(`testbed_v5.py:293-294`). 지표 = `testbed_v5.py:388-398`(RMSE·편향·MAE·r·R²·블록 등가중 RMSE), ΔRMSE 구간 = 블록 다항 재추출 짝지은 부트스트랩, 씨앗 0(`testbed_v5.py:415-432`). 씨앗 평균 예측으로 채점(`v5_aggregate.py:33-39`).

| 실행(태그·잡) | 스크립트 | 기후 입력 | 특징 | 방법·학습기 | λ | 씨앗 | 분할 | 부트스트랩 B | 부분집합·비고 |
|---|---|---|---|---|---|---|---|---|---|
| core_v6 (CJsgU + 로컬 Lena·region, 10-14) | `v5_compare.py`; `run_v6.sh` core/core_region | year_obs · year · clim | F8TS | P0·P1·P1a·Ku·Ed·PE·D0·F1·D1·R1, CatBoost | 0.25·0.5·1 | 0,1,2 (물리식 0) | block(k5, 겹 씨앗 0) · year(≥20) · block_year(≥10) · random · region(100 km 완충) | 2000 (`run_v6.sh` agg_core); conditions_tdd 는 1000(`v5_aggregate.py:109`, 원고 미사용) | P0 의 E0 = 다른 두 지역 전량(같은 기후 입력 판, `v5_compare.py:69,94`); region 분할에서는 완충 뒤 학습 집합의 E(P0 ≡ P1). R1cv·S 없음 |
| core_cb (로컬 10-10) | `v5_compare.py` | year · clim | F8TS | +S, −Ed·PE | 0.25·0.5·1 | 0,1,2 | 동일 | 2000 | 관측일 정합 채택 전 실행. 원고 v13 은 쓰지 않음(단, `v5_map_predict.py:21` 기본값이 여전히 core_cb) |
| methods2 (로컬 10-12) | `v5_compare.py` | year | F8TS | Ed·PE·R1cv | 0.5 | 0,1,2 | block · year | 2000 | **R1cv(내부 CV λ)는 이 실행에만 있음**(year_obs 판 없음) |
| featsets_v6 (CJsgU) | `v5_compare.py` | year_obs | F8·F8T·F8TS·F8TS_L·F8TS_M·F8TS_X·F8TS_ALL·F8TS_SAR(알래스카만) | P1·D0·R1 | 0.5 | 0,1,2 | block · year | **1000** (`run_v6.sh` agg_feat) | F8TS_M·ALL 은 CECkU 이전 MODIS 표. 그림 3c 는 F8·F8T·F8TS_X 만 여기서 가져옴(`paper_v13/fig3_accuracy.py:357`) |
| modis_v6 (CECkU) | `v5_compare.py`; `run_v6b.sh` | year_obs | F8TS·F8TS_M·F8TS_ALL (MODIS 2010–2024) | P1·D0·R1 | 0.5 | 0,1,2 | block · year | 2000 | 그림 3c 의 F8TS·F8TS_M·F8TS_ALL |
| nn_v6 (RkmTeb, A100) | `v5_compare.py` GPU 분기 | year_obs | F8TS·F8TS_M | D0·R1, mlp·tabm·ftt(`tab_models.py:45-125`: 100 epoch, 학습 겹 안 무작위 10 % 행 조기 종료(인내 6), lr 1e-3(ftt 5e-4), 배치 8192, SmoothL1, 중앙값 대치·표준화·±10 절단) | 0.5 | 0,1,2 | block · year | **1000**, P1 기준은 core_v6 에서(`--ref-run`, 로컬 재집계; `SESSION_HANDOFF.md:1093`) | TabM·FT-T 는 자체 축약 구현(TabM = 공유 몸통 + 머리 8개) |
| rf_v6 (CJsgU) | `v5_compare.py` | year_obs | F8TS | D0·R1, RF(300그루, 잎 ≥5, 특징 0.5; `testbed_v5.py:297-301`) | 0.5 | 0,1,2 | block · year | **1000**, 기준 core_v6 | |
| confirm_v6 · blk025_v6 · blk100_v6 (CJsgU) | `v5_compare.py` | year_obs | F8TS | P1·D0·R1 | 0.5 | 0,1,2 | block, 겹 씨앗 1 / 블록 0.25° / 1.0° | **1000** | 민감도 |
| ladder_v6 LL (nMsgU) | `v6_label_ladder.py:60-86`; `run_v6c.sh` | year_obs | F8TS | P0(E_src 두 지역), P1(E 를 n개로), D0, F1, D1(r 1), D1x4(r 4), R1 | 0.5 | 0,1,2 = **라벨 추출 + 학습기 씨앗**(`:70`) | `folds_block(d,5,0)`(`:62`), 시험 겹 전체 채점 | **1000**, 자체 구현(블록 행 재추출, RNG 씨앗 0 공유; `v6_label_ladder_agg.py:19-26`) | n ∈ 25…2500·all; 씨앗 평균 예측 RMSE(`rmse`)와 씨앗별 평균(`rmse_seed_mean`) 둘 다 기록 |
| ladder_v6 TL (nMsgU) | `v6_label_ladder.py:89-124` | year_obs | F8TS | P0(원천 E0), P1n(n ≥ 3 이면 대상 n개 E), D0S(원천+n), D0T(n ≥ 25), D1S(유사라벨 min(시험, 원천+n)개), R1S(원천 행 E0 앵커·대상 행 En 앵커), R1T | 0.5 | 0,1,2 (n 0 은 씨앗 0만) | 대상 블록 5겹(씨앗 0); **원천 = 다른 두 지역 전량, 100 km 완충 없음**(`:91`) | 1000 | n ∈ 0…500 |
| extra_v9 PA (gxnyhb) | `v9_extra.py:66-89` | year_obs | F8TS (+제품 열 R1x) | Praw·Pcal(학습 겹 아핀)·P1·R1·RP·RP1·R1x | 0.5 (RP1 1.0) | 0,1,2 | `folds_block(d,5,0)`, **겹 안에서 제품 값이 있는 행만**(학습 ≥ 30, 시험 ≥ 5; `:72-74`); E 도 그 행으로 | 2000 | 지역 × wei·liu |
| extra_v9 EXT (gxnyhb) | `v9_extra.py:93-120` | year_obs(관측일 없음 → 244일) | F8TS | P0(E0 = 세 지역 전량 통합, `:98`), P1loc(같은 소지역 **다른 셀** 전 연도 ≥10행으로 E, `:101-106`), D0, F1, R1, R1loc | 0.5 | 0,1,2 | 학습 = 세 지역 전량, 시험 = 외부 816 셀-연도 | 2000 | 집계 부분집합 all(816) · in_range(셀 평균 ALT ≤ 300 cm, 805) · 소지역 |
| extra_v9 EXTFT (dfuMW) | `v9_extra.py:124-172`; `run_v9b.sh` | year_obs | F8TS | P0, P1n(n ≥ 3), D0S, R1S | 0.5 | 0,1,2 | 소지역별 **블록 하나씩 제외**(그린란드는 셀), 나머지 블록에서 n ∈ 0·5·10·25·50·all | 2000 (`T.boot_delta`) | 셀 평균 ALT ≤ 300 cm 셀만(`:129`; 805 행·72 셀·49 블록 = 원고 매크로와 일치) |
| extra_v9 SHAP | `v9_extra.py:176-192` | year_obs | F8TS | CatBoost 동일 초매개변수, 씨앗 0, 지역 전량 적합 | - | 0 | 없음 | - | 잔차(y − P1, E 전량)와 직접 |
| 제품 비교(로컬, `ingest_v6.sh:28-33`) | `v5_products_eval.py`, `v5_common_cells.py`, `v5_yearwise.py` | year_obs | F8TS | core_v6 block OOF 씨앗 평균 P0·P1·Ku·D0·R1 | 0.5 | - | - | 2000 (mask5 만 1000, 원고 미사용) | 단위 = 셀-연도; 제품 = 관측 연도 값(+다년 평균 정의); Wei 학습 지점 25 km 마스크(거리 정보가 16,189 셀 전부에 있음, 확인); 공통 셀 = Wei·Liu·CCI v4 모두 있는 행; 연도별 그림은 ≥30 셀-연도(`v7_f5_products_years.py:15`), 그림 7d 는 ≥50 |
| 지도 Alaska·Lena(로컬, `ingest_v6.sh:41`) | `v5_map_predict.py --forcing year_obs --run core_v6` | **학습 year_obs, 격자는 그 해 전체 TDD**(`grid/*_forcing_year.csv` 에 tdd_cum 열 없음, 10-10 생성) | F8TS | P1(E 지역 전량: 알래스카 1.949, 레나 1.679), R1 CatBoost 3씨앗 평균 | 0.5 | 0,1,2 | 없음(전량 적합) | - | 구간 = core_v6 block·year_obs R1 OOF 잔차 5·95 %; AOA = 학습 행(관측일 판) 표준화 공간에서 블록 간 최근접 거리 95 분위 대 격자(2011–2024 평균 **전계절** 강제력); 표시 = gray == 0 (858,517 · 38,086 셀) |

라벨·외부 자료 구축(비교용):

| 항목 | 연구 지역(`build_v5_labels_cell_year.py`) | 외부 CALM(`build_v9_external.py`) |
|---|---|---|
| 자료·연도 | ABoVE v2, CALM(`alt_global.csv`), ALLena; 2010–2024 = in_forcing | PANGAEA 972777, 2010–2024(`:33`) |
| 계절 필터 | ABoVE 8–10월(`:33,73`), ALLena 8–9월(`:34,110`), CALM 날짜 없음 → 244일(`testbed_v5.py:113`) | CALM 날짜 없음 → 244일(`:59`) |
| 값 QC | ABoVE 0 < ALT < 300 cm·좌표 상자(`:64`); ALLena QF ≠ 2(`:110`); CALM 범위 필터 없음(>300 cm 5행은 1990년대라 in_forcing 0) | 검열값 1건·주석 정규식(early/burn/flood/not reached 등) 3건 제거(`:34-39`); 범위 필터 없음, EXTFT·EXT in_range 에서 셀 평균 ≤ 300 cm |
| 셀 배정·집계 | v4 loc_id, ABoVE·CALM 평균, ALLena 위치·연도 최대, 관측일 = 평균 doy(`:77-79,113-114`) | 최근접 v4 셀 < 1 km, 셀·연도 평균(`:46-47,55`) |
| 지역 간 거리 | region 분할에서만 100 km 완충(`testbed_v5.py:212-223`) | 연구 라벨 100 km 안 지점 제외(사모일로프·비코프스키 3곳, `:51-53`) |

---

## 2. 발견 사항

### 2.1 지도 학습 설정과 지도 오차 보고 설정의 불일치 (과제 항목 2)

**F1. [수정 필요] 지도는 관측일 누적 TDD 로 학습한 모델(E·잔차 학습기)에 그 해 전체 TDD 를 넣어 만들었고, 보고한 오차·구간·AOA 는 관측일 입력의 교차검증에서 나왔다.**
- 근거: `ingest_v6.sh:41`(`--forcing year_obs --run core_v6`), `data/processed/v5/maps/Alaska/meta.json`(`interval.source = Alaska__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz`, `E = 1.949`, `rmse_oof 18.42`), `v5_map_predict.py:42-43`(학습 판 = `a.forcing`), `:84-94`(격자 강제력 `grid/<region>_forcing_year.csv` 의 `e5_tdd`·`e5_sqrt_tdd` = 1–12월 합, `build_v5_forcing_year.py:121,127`; 격자 표에는 `tdd_cum_*` 열이 없음, 생성일 2026-10-10 < 셀 표 2026-10-12).
- 크기: 관측일까지의 TDD 비율 평균 알래스카 0.77(10–90 % 0.65–0.98), 레나 0.79, 캐나다 0.84. 같은 라벨에 대한 E 는 year_obs 1.88–2.00 대 year 1.68–1.78(알래스카, `core_v6/meta/Alaska__block__{year_obs,year}__F8TS__P1__-__lam-1.json`), 레나 1.58–1.74 대 1.41–1.55. 즉 E_obs/E_year ≈ 1.12 이고, 격자에 전계절 √TDD 를 곱하면 P1 지도는 전계절 입력으로 학습한 모델보다 약 12 % 높다. Stefan 식 안에서는 관측일 E 를 계절 말로 외삽한 것으로 해석할 수 있으나, (i) 잔차 학습기 g(x)는 `e5_tdd`·`e5_sqrt_tdd` 가 학습 범위(관측일 값)를 넘는 입력을 받으므로 트리 모델 특성상 최고 TDD 학습 행의 보정값으로 고정되고, (ii) 90 % 구간 반폭(결과 `results.tex:48` `\nhwAk`·`\nhwLe` = 25.5·24.5 cm)과 OOF RMSE 는 관측일 입력에서 측정한 값이며, (iii) AOA 는 관측일 강제력으로 표준화한 학습 공간에 전계절 평균 강제력 격자를 넣어 계산해(`v5_map_predict.py:63-64,103-106`) TDD 축이 한쪽으로 밀린 상태에서 56 %·62 % 가 나왔다.
- 원고: `methods.tex:22` "Yearly maps use the full-season TDD of each year and therefore represent end-of-season ALT", `:72` "using the full-season climate of each year", `:77` "The 90% prediction interval ... from the spatial-block out-of-fold residuals". 학습 입력이 관측일 TDD 라는 점과 구간이 그 입력에서 나왔다는 점이 없다. `v5_map_predict.py:3-4,8` 주석의 "오차는 같은 설정의 블록 교차검증(core_cb)에서 보고한다"는 설정이 같지 않고 참조 실행도 낡았다.
- 조치안(택일 뒤 민감도 1장): (a) 격자에도 `tdd_cum_*` 를 만들어(`build_v5_forcing_year.py` 는 셀 표에만 추가됨; 격자 재생성 필요) 기준일(예: 244일, 또는 지역 관측일 중앙값 232·236일)까지의 TDD 로 예측하고 지도를 "그 날짜의 ALT" 로 정의한다. 관측일 모델·구간·AOA 와 설정이 일치한다. (b) `--forcing year` 로 학습해 전계절 입력과 맞추고, 구간은 core_v6 의 `*__block__year__*` R1 OOF 잔차에서 가져온다(파일 있음). 이 경우 지도는 "관측일 ALT 를 전계절 TDD 로 설명한 모델"이 되어 계절 말 ALT 라고 부르기 어렵다. (c) 현행 유지 시 Methods 에 "관측일 TDD 로 적합한 계수를 전계절 TDD 에 적용한 외삽이며, 구간·AOA 는 관측일 입력 교차검증에서 얻었다" 를 명시하고 (a) 또는 (b) 와의 차이 지도를 보충에 넣는다. (a) 를 권한다.

**F2. [수정 필요] 지도 영역 규칙이 원고 서술(PFR ≥ 50 % 만)과 다르다.**
- 두 격자 모두 `build_masks`(`build_map_grid_lena_v1.py:103,510-530`, 알래스카는 `build_map_grid_alaska_v1.py:406` 에서 재사용)로 dem_tile_absent → water(GSW 발생 빈도 셀 평균 ≥ 50 %) → pfr_missing → pfr 문턱 → **cci_missing(CCI ALT 제품 값 없음) → soil_tdd_missing(토양 TDD ≤ 0) → s_missing** 순으로 회색을 정하고, 지도는 gray == 0 만 그린다(`v5_map_predict.py:81`). 알래스카 회색 41,116 셀 중 cci_missing 18,643 + soil_tdd_missing 9,217 = 27,860 셀(68 %)은 P1·R1 이 쓰지 않는 입력의 결측 때문에 빠졌다(`map_alaska/alaska_grid_x25_v1_meta.json` masks.summary.by_reason_first). 레나는 cci_missing 77, ERA5-Land 육지 격자 미확보 1,338 셀(`grid/lena_forcing_meta.json` fail 1338)이 빠진다.
- `methods.tex:25,72` 는 영역을 "ESA CCI permafrost probability of at least 50 %" 로만 적는다. 조치: Methods 에 수체·DEM·ERA5-Land 육지·(유지한다면) CCI ALT·토양 TDD 조건을 적거나, 지도용 마스크에서 cci_missing·soil_tdd_missing 을 빼고 지도를 다시 만든다(모델 입력과 무관한 조건이므로 빼는 쪽을 권함).

**F3. [설명 추가] 알래스카·레나 격자의 영구동토·수체 규칙이 서로 다르며 캐나다 격자에서 결정해야 한다(과제 항목 4).**
- 영구동토: 알래스카는 영역 정의에서 Natural Earth 육지 ∧ PFR ≥ 50 %(`build_map_grid_alaska_v1.py:89,168`); 레나는 상자 안 전체 셀에 PFR < 10 % 또는 결측이면 회색(`build_map_grid_lena_v1.py:91,518-519`). 현재 레나 표시 셀은 전부 PFR 100 % 라 결과 차이는 없지만 규칙은 다르다. 캐나다는 알래스카 규칙(50 %)을 쓰고 그렇게 적어야 원고와 맞는다.
- 수체: 두 격자 모두 GSW 발생 빈도 셀 평균 ≥ 50 %(`lena:92,517`); 알래스카 59–60°N 띠는 GSW 60N 타일이 없어 수체를 판정하지 않는다(`alaska:23-24`). 캐나다 격자는 타일 범위를 먼저 확인해야 한다.
- 결측 입력: 토양 9열이 비는 표시 셀(알래스카 916, 레나 272)은 CatBoost 결측 처리로 예측되고 AOA 는 결측을 표준화 0 으로 둔다(`v5_map_predict.py:64,104`). 설명 한 줄이면 된다.
- `v5_map_predict.py:16` 은 이미 Canada 를 받도록 바뀌었으나 `:21` 기본 `--run` 은 core_cb, `:23` 기본 `--forcing` 은 year 이다. 캐나다 지도를 만들 때 ingest 와 같은 인자를 쓰거나 기본값을 바꿔야 한다.

### 2.2 같은 그림·문장에서 비교되는 실행 사이의 설정 차이 (과제 항목 1)

**F4. [수정 필요] 부트스트랩 반복 수가 원고 진술과 다른 실행이 있다.**
- `methods.tex:53`: "B=2000 (1000 times for the label-number experiments)". 실제 1000 인 집계: featsets_v6(`run_v6.sh` agg_feat), confirm_v6·blk025_v6·blk100_v6, rf_v6, nn_v6(`SESSION_HANDOFF.md:1093`). 2000: core_v6, modis_v6(`run_v6b.sh`), PA·EXT·EXTFT, 제품 비교. 그림 3c 는 modis_v6(2000)와 featsets_v6(1000)의 구간을 한 판에 섞는다(`paper_v13/fig3_accuracy.py:107-110,357`). 학습기 민감도(보충)도 1000 이다.
- 조치: 평가만 다시 하면 된다. `v5_aggregate.py --run <tag> --nboot 2000 --refs P1`(nn_v6·rf_v6 는 `--ref-run data/processed/v5/runs/core_v6`)을 featsets_v6·confirm_v6·blk025_v6·blk100_v6·rf_v6·nn_v6 에 돌리고 수치 매크로를 재생성한다. 그렇지 않으면 Methods 문장을 "민감도 실행은 1000" 으로 고친다.

**F5. [설명 추가] 라벨 수 사다리의 세 씨앗은 라벨 추출도 바꾸며, 채점은 세 추출의 예측 평균으로 한다.**
- `v6_label_ladder.py:70` 에서 씨앗 s 가 `sample(n, random_state=s)` 와 학습기 씨앗을 함께 정한다. `v6_label_ladder_agg.py:35,39` 는 셀-연도별로 세 씨앗 예측을 평균한 뒤 RMSE·Δ를 계산하므로, 그림 4a·b 의 값은 "n개 라벨 모델 한 개"가 아니라 "서로 다른 n개 추출 세 개의 앙상블(최대 3n 개 라벨 사용)" 이다. `ladder_metrics.csv` 의 `rmse` 와 `rmse_seed_mean` 차이는 n 25 에서 알래스카 17.84 대 18.34, 캐나다 30.9 대 31.8 cm 이고 n 이 커질수록 줄어 'all' 에서는 core_v6 값과 같다(알래스카 R1 18.4223 = core_v6 block·year_obs 18.42, 겹·씨앗 동일 확인).
- 핵심 실행(core_v6)의 세 씨앗은 같은 학습 집합에서 학습기 씨앗만 바꾼 앙상블이라 의미가 다르다. 조치: `methods.tex:61` 과 그림 4 캡션에 "three draws; predictions of the three draws averaged before scoring" 을 적거나, 사다리 그림을 `rmse_seed_mean`(추출별 RMSE 의 평균)으로 바꾼다. 후자가 "라벨 n개로 얻는 정확도" 라는 질문에 맞다.

**F6. [설명 추가] 전이 사다리(TL)의 원천 집합은 지역 홀드아웃과 달리 100 km 완충을 두지 않는다.**
- `v6_label_ladder.py:91` 원천 = 다른 두 지역 전량; `testbed_v5.py:212-223` 지역 홀드아웃은 시험 지역 100 km 안 학습 셀 제외. 영향은 작다: 대상 알래스카일 때 원천 5행(1셀), 대상 캐나다일 때 20행(20셀)이 100 km 안(최소 42.9 km), 대상 레나는 0. 그림 4b 의 n = 0 과 그림 3b 의 지역 홀드아웃이 같은 실험으로 읽히므로 차이를 한 문장 적는다. P0 의 E0 도 TL 은 완충 없는 전량(`:94`), 지역 홀드아웃은 완충 뒤 학습 집합(`v5_compare.py:94`)이다.

**F7. [설명 추가] 지역 홀드아웃에서는 '원천 계수 Stefan(P0)' 과 '재보정 Stefan(P1)' 이 같은 값이다.**
- `v5_compare.py:94` 에서 region 분할의 E0 = 학습 집합(완충 뒤 두 지역)의 최소제곱 E 이고 P1 도 같은 집합으로 E 를 맞추므로 동일하다. `summary_metrics.csv` region·year_obs: P0 = P1 = 19.36(알래스카)·21.46(레나)·27.75(캐나다); 매크로 `\nrmseAkregPz` = `\nrmseAkregPo`. `methods.tex:38` 의 정의("recalibrated ... on the training labels of the target region")는 지역 홀드아웃에 대상 라벨이 없어 적용되지 않는다. 그림 3b 에 둘을 나란히 두면 중복이므로 하나만 두거나 캡션에 적는다.

**F8. [설명 추가] 외부 지역 실험 안에서 '인근 재보정' 의 홀드아웃 단위가 다르다.**
- EXT 의 P1loc 는 같은 소지역의 다른 **셀** 전 연도(≥10행)로 E 를 맞춘 셀 단위 홀드아웃(`v9_extra.py:101-106`); EXTFT 의 P1n(all) 은 **블록** 하나를 빼고 나머지 블록 전량으로 E 를 맞춘다(`:140-158`). 같은 805 행에서 P1loc 49.7 cm, P1n(all) 43.2 cm(`ext_metrics.csv` in_range, `extft_metrics.csv` all). 보충 그림 S5(ext_metrics in_range)와 그림 4d(EXTFT)가 나란히 놓이므로 단위 차이를 적는다. 또 EXT 의 P0·R1 은 E0 를 세 지역 통합으로 맞추고(`:98`) 핵심 실행의 P0 는 두 지역으로 맞추는데, 대상이 다르므로 정의 차이만 적으면 된다.

**F9. [설명 추가] 제품 기준 결합(PA)은 제품 값이 있는 행으로 학습·채점하고 E 도 그 행으로 맞춘다.**
- `v9_extra.py:72,75`. 같은 겹이지만 행 수가 다르다(알래스카 liu 14,326 대 core 15,540). 보충 S7 의 P1·R1 값(알래스카 R1 17.23)이 본문 core 값(18.42)과 다른 이유를 캡션에 적는다.

**F10. [수정 필요(작음)] 내부 교차검증으로 λ 를 고른 경우(R1cv)는 전계절 입력 실행(methods2, 10-12)에만 있다.**
- core_v6 의 방법 목록에 R1cv 가 없다(`runs/core_v6/run_*.json`, `run_v6.sh`); `methods2/run_20261012_143938.json` 은 forcing year. `methods.tex:43` 은 λ 0.25·1·"a value selected by inner block cross-validation" 을 민감도로 들고 있다. 주 설정(year_obs)으로 R1cv 를 다시 돌리거나(블록 분할·세 지역·CatBoost, 로컬 CPU 수십 분), 문장을 "tested with the full-season input" 으로 한정한다.

**F11. [문제 없음] 겹·씨앗·학습기·특징·λ 의 일치.**
- 겹 씨앗 0·k 5·블록 0.5°: core_v6(run json fold_seed 0), ladder(`v6_label_ladder.py:62,92` 상수 0), PA(`v9_extra.py:68`). confirm_v6 만 의도적으로 겹 씨앗 1. 사다리 'all' 의 RMSE 가 core_v6 와 소수점 넷째 자리까지 같다는 것으로 겹·행 순서·씨앗 일치를 확인했다.
- CatBoost 초매개변수는 `testbed_v5.py:293-294`, `v9_extra.py:185`, `v5_map_predict.py:50-51` 에서 동일. 씨앗 0,1,2 동일. F8TS 23열 동일(run json `feats`, maps meta `feats`). λ 0.5 동일.
- 제품 비교 세 스크립트 모두 `--forcing year_obs`, core_v6, 2000 회(`ingest_v6.sh:28-33`); `core_v6/yearwise` 는 `yearwise_year_obs_F8TS` 로의 링크. 누설 마스크는 16,189 라벨 셀 전부에 Wei 거리가 있어 "거리 없는 셀은 남긴다" 규칙(`v5_products_eval.py:42`)이 발동하지 않는다. 비교 단위는 셀-연도로 본문·표 S2 와 일치한다.
- 3분위 조건 분석: 2000 회, ≥30 셀-연도 ∧ ≥5 블록(`v9_f4e_gain_conditions.py:8,76`) = `methods.tex:58`.

### 2.3 앵커 계수 E 의 적합 방식 (과제 항목 3)

**F12. [문제 없음, 정의 차이는 설명] 모든 실행이 `ls_scale`(원점 통과 최소제곱, p > 0 행; `testbed_v5.py:233-235`)을 쓰고 TDD 변형은 year_obs 로 같다.** 적합 행만 다르다.
- 핵심 P1: 학습 겹(`run_method` → `fit_phys(tr)`, `:324`). 핵심 P0: 다른 두 지역 사용 가능 행 전량, 같은 기후 입력 판(`v5_compare.py:69`); region 분할은 F7.
- 사다리 LL: P1·R1 앵커 = n개 부분집합(`v6_label_ladder.py:77,83`), P0 = 두 지역 전량(`:39`). TL: P1n = 대상 n개(n ≥ 3, `:105`), R1S 의 학습 앵커는 원천 행 E0·대상 행 En(`:118`), 시험 앵커 En.
- 외부: EXT E0 = 세 지역 통합(`v9_extra.py:98`), P1loc 셀 단위(F8); EXTFT P1n = 다른 블록 n개(`:158`), n 0 이면 E0.
- PA: 제품 값이 있는 학습 겹 행(`:75`). 지도: 지역 전량(`v5_map_predict.py:43`; 알래스카 1.949, 레나 1.679).
- R1 잔차 학습기는 모든 실행에서 λ·res 를 더하고(`testbed_v5.py:356-359`, `v6_label_ladder.py:55-57`, `v9_extra.py:83,115,165`, `v5_map_predict.py:94`) 잔차 목표는 y − E·s 로 같다. R1cv 만 학습 겹 안 3겹 블록 CV 로 λ 를 고른다(`testbed_v5.py:332-346`).

### 2.4 원고 Methods 진술과 코드 (과제 항목 5)

| 진술(`methods.tex` 줄) | 코드 | 판정 |
|---|---|---|
| CatBoost 600회·0.03·깊이 6·L2 3 (`:43`) | `testbed_v5.py:293-294` | 문제 없음 |
| B = 2000, 라벨 수 실험 1000 (`:53`) | F4 | 수정 필요 |
| 지역 홀드아웃 100 km 완충 (`:48`) | `testbed_v5.py:212-223` | 문제 없음 |
| 날짜 없는 CALM = 244일 (`:5`) | `testbed_v5.py:113`, `build_v9_external.py:59` | 문제 없음 |
| ABoVE 0 < ALT < 300 cm, 8–10월 (`:5`) | `build_v5_labels_cell_year.py:33,64,73` | 문제 없음(CALM·ALLena 는 범위 필터 없음, 본문도 ABoVE 에만 적용한다고 씀; ALLena 최대 260 cm) |
| ALLena 8–9월, 탐침 한계 표지 제외, 위치·연도 최대 (`:5`) | `:34,110,113` | 문제 없음 |
| 연도 홀드아웃 ≥20 셀-연도 (`:48`) | `folds_year(min_test=20)` `:189` | 문제 없음 |
| 블록×연도 홀드아웃 (`:48`, 그림 2c 캡션 "fewer than 20 ... not scored") | `folds_block_year(min_test=10)` `:200` | 설명 추가: 블록×연도는 10 |
| 유사라벨 = 채점 겹에서 같은 수 (`:43`) | `npl = min(len(te), round(len(tr)))` `:352` | 문제 없음(시험 겹이 더 작으면 그 수) |
| 세 씨앗 예측 평균 (`:43`) | `v5_aggregate.py:35` | 문제 없음(사다리는 F5) |
| 학습기 민감도 RF·MLP·TabM·FT-Transformer (`:43`) | `tab_models.py:45-84` 자체 축약 구현, 학습 겹 안 무작위 10 % 행으로 조기 종료(공간 블록 아님) `:119` | 설명 추가: "simplified re-implementations" 와 조기 종료 규칙 |
| SHAP 지역 전량 (`:43`) | `v9_extra.py:179-186` | 문제 없음 |
| 조건 분석 ≥30·≥5 블록 (`:58`) | `v9_f4e_gain_conditions.py:76` | 문제 없음 |
| 연도별 RMSE ≥30 (`:58`, 표 S2) | `v7_f5_products_years.py:15`(그림 5b); `v5_yearwise.py` 기본 min-n 20 은 표만 | 문제 없음 |
| 외부: 검열·주석 제거, 100 km, 스발바르 시추공 제외 (`:61`) | `build_v9_external.py:33-39,51-53`, `v9_extra.py:129` | 문제 없음(제외 기준은 "셀 평균 ALT ≤ 300 cm"; 원고 문장과 같은 뜻) |
| 지도 영역 PFR ≥ 50 % (`:25,72`) | F2·F3 | 수정 필요 |
| 지도 구간·AOA (`:72-77`) | F1 | 수정 필요 |
| 2010–2024 유지, FDD = 10월(y−1)–4월(y) (`:5,22`) | 2010 행 75개(외부 2010 포함)는 2009년 10–12월이 없어 FDD·tcold 는 1–4월, SWE 는 1–9월만으로 계산(`build_v5_forcing_year.py:118-119,131`, winter_partial) | 설명 추가(또는 2010 제외; 0.4 %) |
| 계산 "Models were fitted ... on cloud nodes" (`:80`, 표 S4 "All model fitting used cloud CPU nodes") | 지도 모델과 nn_v6 재집계는 로컬(`ingest_v6.sh:41`, `maps/map_*_v6.log`, `SESSION_HANDOFF.md:1093`); core_v6 의 Lena·region 일부도 로컬 run json 이 있음 | 설명 추가 |
| λ 민감도에 내부 CV 선택 포함 (`:43`) | F10 | 수정 필요(작음) |

### 2.5 그 밖에 확인한 것

- **외부 QC 대 연구 지역 CALM QC [문제 없음, 비고]**: 연구 지역 CALM 표(`alt_global.csv`)에는 주석 열이 없어 주석 기반 QC 를 적용할 수 없었고, 외부 표에는 적용했다(제거 3건). 연구 지역 CALM 은 676 셀-연도(3.5 %)라 영향이 작다. 필요하면 보충 S3 에 한 줄.
- **featsets_v6 의 MODIS 묶음 [문제 없음]**: 그림 3c 가 F8TS_M·F8TS_ALL 을 modis_v6(2010–2024 MODIS)에서, 나머지를 featsets_v6 에서 가져오는 규칙이 소스 데이터 JSON 에 적혀 있다(`fig3_accuracy.py:357`). B 차이는 F4.
- **지도 docstring [설명 추가]**: `v5_map_predict.py:3-4,8,21` 의 core_cb·"연도 정합" 표현은 낡았다(실행은 core_v6·year_obs).
- **사다리 표의 n 열 [비고]**: `ladder_metrics.csv` 의 `n`(n_label all)은 마지막 겹 학습 행 수(알래스카 10,743)이고 그림은 preds 에서 겹별 크기를 다시 센다(`fig4_labels.py:81-85`). 결과에 영향 없음.
- **TL D1S 유사라벨 수 [비고]**: `min(len(te), len(src)+n)` 개(`v6_label_ladder.py:114`)로 LL 의 r = 1 규칙과 뜻이 같다(학습 행 수만큼).

---

## 3. 우선순위 수정 목록

1. **지도 설정 통일(F1)**: 격자 `tdd_cum_*` 생성 → 기준일 TDD 로 재예측(권장) 또는 `--forcing year` 재학습 + core_v6 `year` OOF 구간. Methods "Yearly mapping" 절·`results.tex:48` 구간 수치·그림 7·S8·S9 재생성, 두 설정 차이 지도 1장을 보충에 추가. 캐나다 지도도 같은 설정으로.
2. **지도 영역 규칙(F2·F3)**: Methods 에 실제 마스크 규칙을 적거나 cci_missing·soil_tdd_missing 을 지도 마스크에서 제거하고 재생성. 캐나다 격자는 PFR ≥ 50 %·GSW 타일 범위·Natural Earth 육지 규칙을 알래스카와 같게 두고 메타에 기록.
3. **부트스트랩 재집계(F4)**: featsets_v6·confirm_v6·blk025_v6·blk100_v6·rf_v6·nn_v6 를 `--nboot 2000` 으로 재집계(평가만, 로컬 CPU) 후 `v6_numbers.py` 재실행. 또는 Methods 문장 수정.
4. **사다리 채점 정의(F5)**: 그림 4a·b 를 `rmse_seed_mean` 기준으로 바꾸거나 "세 추출의 예측 평균" 을 Methods·캡션에 명시. TL 의 완충 없음(F6)도 한 문장.
5. **R1cv(F10)**: year_obs 설정으로 재실행하거나 문장 한정.
6. **설명 추가 묶음**: 지역 홀드아웃 P0 ≡ P1(F7), 외부 P1loc 대 P1n 단위(F8), PA 행 집합(F9), 블록×연도 최소 10, 2010 겨울 창, 신경망 축약 구현·조기 종료 규칙, 계산 위치(지도·재집계 로컬), `v5_map_predict.py` 주석·기본값 갱신.

## 4. 요약 판정

- 핵심 비교(core_v6)·공변량·학습기·블록 크기·겹 씨앗 민감도·라벨 사다리·전이·외부·제품 비교는 라벨 필터, 기후 입력(year_obs), 특징(F8TS), CatBoost 초매개변수, λ, 씨앗, 겹(k 5·씨앗 0·0.5°)이 일치한다. 불일치는 (i) 지도의 입력(전계절 TDD) 대 학습·오차 평가(관측일 TDD), (ii) 지도 영역 규칙의 서술 누락과 두 격자 간 규칙 차, (iii) 민감도 실행의 B = 1000, (iv) 사다리의 세 추출 앙상블 채점, (v) R1cv 의 입력 판, (vi) 지역 홀드아웃 P0 ≡ P1·외부 재보정 단위·TL 완충 같은 정의 차이다. (i)·(ii) 는 지도 재생성 또는 본문 수정이 필요하고, (iii)·(v) 는 평가 재실행으로 해결되며, 나머지는 문장 추가로 충분하다.
