# XE_hires_covariates 구현 기록(계획 해석과 구현 결정)

- 대상 파일: `scripts/3_deep_learning/x_hires_covariates.py`(실험 하네스), `scripts/3_deep_learning/x_hires_registry.py`(열·군·변형·90 % 규칙 등록부), `scripts/1_data_prep/xe_point_covariates.py`(특징 추출 명령), `scripts/1_data_prep/xe_feature_tools.py`(래스터 창·지형·MODIS 정의 순수 함수), `tests/test_x_xe.py`
- 근거: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 2.5절(개정 1, T0 = git 2678100, 2026-10-04 14:22:39 +0900), 1절(공통 규약), 0.3(열람 순서·출력 제한), `docs/research/2026-10-04/within_grid_inputs.md` 3–5절, `harness_implementation_plan.md` 4.5, `impl_notes/xbatch_core.md`, `impl_notes/xe_stage2_acquire.md`
- 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리이며, 모두 가장 문자 그대로의 읽기를 따랐다. 가설·판정 규칙·해석 문장은 바꾸지 않았다.
- 이 기록을 쓰면서 계산한 것: 합성 자료 시험, 실제 자료의 세기(적합 수, 셀 수), 좌표 열만 쓰는 창 묶음 수, XE-e 준비 표의 셀·블록 수(좌표·VWC 만). 모형 결과 표는 열지 않았다.
- 2026-10-04 저녁 적대적 검토(결함 17건)의 반영은 8절에 있다. 1–7절의 문장 가운데 8절로 바뀐 곳은 '(8절 n)' 로 표시했다.

## 1. 계획과 작업 지시 사이의 불일치

| 번호 | 작업 지시 | 계획 2.5 | 처리 |
|---|---|---|---|
| 1 | 시험 파일 `tests/test_x_xe.py` | `tests/test_x_hires.py` | 작업 지시의 이름을 썼다. 계획의 시험 목록 (a)–(f)를 모두 담았다(파일 머리 주석). 묶음 목록의 `tests/test_x_*.py` 에 들어간다 |
| 2 | 특징 추출을 `scripts/1_data_prep/xe_*.py` 로 | `scripts/1_data_prep/xe_point_covariates.py` | 명령 파일은 계획 이름 그대로 두고 순수 함수를 `xe_feature_tools.py` 로 나눴다 |
| 3 | '최종 해시 커밋 단계' | 'sha256 을 개정 이력에 커밋'(T0 + 100 h) | git 커밋은 하지 않는다(작업 규칙). `finalize` 가 `xe_feat_v1_final.json` 과 개정 이력 문구 `xe_feat_v1_revision_entry.md` 를 쓰고, 커밋은 사용자가 한다. 하네스는 최종 해시 파일의 sha256 이 표와 같고 그 sha256 이 git HEAD 의 계획 문서 개정 이력 절에 있을 때만 2단계 변형을 돈다(8절 12) |
| 4 | M 군 원자료 | 주 경로 ArcticDEM 10 m 색인 JSON + `/vsicurl` 창 읽기, 대체 32 m | `xe_stage2_acquire.py` 는 M 군을 다루지 않는다. `xe_point_covariates.py m-fetch` 로 구현했다(아래 4.4). 이번 작업에서는 실행하지 않았다(마감 T0 + 72 h) |
| 5 | 등록부 위치 | 정하지 않음 | 하네스가 Rescale 에서도 읽어야 하므로 `scripts/3_deep_learning/x_hires_registry.py` 에 두었다(xbatch 묶음 목록의 `x_*.py` 에 들어간다) |

## 2. 변형·방법·저장소

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| x25 의 방법 | '방법 P1, R1, D0. WF9 의 R9Unit 을 부른다', 재현 관문 'x25 의 총·격자 안·격자 사이 저장소가 WF9 조각과 같다' | x25 변형은 `h54.R9Unit.run` 을 그대로 부른다(WF9 의 P1, Pk, Pc, R1, Re, D0). 관문이 저장소 전체의 일치를 요구하므로 WF9 와 같은 방법 묶음을 둔다 | 시험 (d)가 모든 키의 예측과 세 저장소의 SSE 가 R9Unit 과 같음을 확인한다 |
| 고해상 변형의 방법 | P1, R1(교차검증 λ ∈ {0.25, 0.5, 1.0}, λ 0.25 고정 저장), D0(catboost 기본 용량) | P1, R1(교차검증 λ 와 λ 0.25·0.5·1.0 고정 키, h54 emit 과 같다), D0(catboost)만 적합한다. 키의 방법 이름은 `R1@<변형>`, `D0@<변형>`(h54 wf1 의 `R1@x34` 와 같은 방식) | 같은 적합에서 나오는 고정 λ 키(0.5, 1.0)를 함께 저장해도 적합 수는 같다 |
| 변형 사이 대비의 CI | 1절: 같은 라벨 집합 대비는 분할 안 채점 블록 재표집 | 모든 변형이 같은 추출(h40.draw_cells(대상, 'r', 분할, n, 추출))을 쓰므로 R1@xh − R1 은 같은 라벨 집합 대비다. 모든 변형 조각이 같은 저장소 이름을 써서 한 TMx 로 합쳐지고 `xbatch_core.region_stat_same`(h40.contrast)로 계산한다 | P1 은 변형 조각마다 같은 키·같은 값으로 저장된다(시험 (d)) |
| 위치 묶음 | (블록, √TDD, LG 6B.3 셀 색인 floor(lat/0.009), floor(lon·cosφ/0.009)) | φ = (ky + 0.5)·0.009°(LG 계획 756행의 정의, `parse_ext_palmtag2022_pedon.py` 와 같다). 격자 열쇠(h54.grid_groups)에 셀 색인을 붙인 열쇠라 위치 묶음은 격자 묶음 안에 중첩된다 | 시험 (c) |
| `~l`, `~gl` 저장소 | 위치 안: 2셀 이상 위치 묶음 셀의 위치 평균 편차, 격자 안·위치 사이: 위치 평균의 격자 평균 편차(위치 셀 수 가중) | `~gl` 은 2셀 이상 격자 묶음의 모든 셀에 (ȳ_l − ȳ_g, p̂_l − p̂_g) 를 둔다. 셀마다 한 항이라 위치 셀 수 가중이 된다. 블록마다 SSE_w = SSE_gl + SSE_l(한 셀 위치는 위치 안 편차 0) | 시험 (c) 함수판과 단위 저장소판 |
| 작업 묶음 | xh0·xt2 는 R1b, xh 와 SI 변형은 R3 | `--stage r1b`(x25, xh0, xt2), `--stage r3`(x25, xh, SI 변형). 1절 '한 대비는 한 작업·한 노드 종류의 조각 안에서 닫는다' 때문에 x25 를 두 작업에서 모두 다시 적합한다. 조각 tag 는 `xe_r1b`, `xe_r3` 로 나눈다 | |
| SI 변형 '군별 추가' | 'T2, M, O, V, S 각각' | x25 + 그 군(`add_T2` 등, within_grid_inputs.md 5.3 의 'x25 + 군 하나씩') | |
| SoilGrids 교체 | '5 km 9열 대신 250 m 열' | x25 에서 토양 9열을 빼고 O 군의 SoilGrids 250 m 7열을 더한다(NCSCD 는 넣지 않는다). 90 % 규칙은 군 단위이므로 O 군이 빠진 대상에서는 돌리지 않는다. 혼입: x25 의 토양 9열에는 250 m 대응 열이 없는 cfvo·phh2o(5–15 cm)가 들어 있어, 이 변형의 차이에는 해상도 교체 효과와 두 변수의 제거 효과가 섞인다. 등록 문구 그대로 두고 SI 상태 표·봉인 분해 표의 note 열에 적는다(8절 13) | |
| 점 화소 민감도 | '30 m 이하 제품은 3 × 3 창 통계를 주 값, 점 화소 값을 민감도로 둔다' | 특징 표에 점 화소 열(`slope_px`, `gsw_occ_px`, `treecover_px`, `wc_*_px` 원핫 7열, `ad_slope_px`, `ad_curv_px`)을 두고 SI 변형 `xh_px`(xh 의 창 열을 점 화소 열로 바꾼 판)를 등록했다. TPI·거칠기·점 TWI 는 정의상 창 또는 점 값이라 바꾸지 않는다 | 열 이름은 등록되지 않아 구현에서 정했다 |
| 지역 피복 변형 | '알래스카 ABoVE 30 m, 레나 Lisovski 2025 10 m' | 열 이름과 취득 경로가 등록되지 않았고 취득 목록(2단계 표)에도 없다. 취득·추출 경로를 만들지 않았으므로 이번 묶음에서는 **미실행(등록 이탈)** 이다. 세기의 건너뜀 사유, SI 상태 표 `<tag>_si_status.csv`, 모듈 머리에 그 사실을 적는다. 특징 표에 접두사 열(`abv_*`, `lis_*`)이 있을 때만 도는 규칙은 남겨 두었다(8절 5) | 미해결 항목 1 |

## 3. 판정과 해석 문장

| 항목 | 처리 |
|---|---|
| XE-a·XE-b 의 풀 | 3대상 층화 평균(`xbatch_core.contrast_pool`, 등록 지역 수 3, '지역 k/3' 표지). n 500·1,000 은 캐나다 행이 없어(|A| 308–447) '부분(지역 2/3)'이다 |
| Holm(m = 6) | XE-a, XE-b 의 n 500, 1,000, 전량 층화 평균 행의 양측 p(`p_two`, 두 가중 가운데 큰 값). 행이 없거나 판정 불가인 n 은 p 1 로 넣어 m 을 유지한다(`xbatch_core.holm_table`). 보조 열이고 판정은 보정 전 CI 의 4분 판정이다 |
| _rule3 결과 → 해석 표의 행 | '지지·부분 지지' = 우세, '기각' 가운데 열세 판정이 있으면 열세, 열세가 없고 판정이 있는 n 이 모두 동등이면 동등, 그 밖의 기각은 미결정, '판정 불가' 는 판정 불가. 해석 표의 'XE-a 동등' 을 '판정이 있는 모든 등록 n 이 동등'으로 읽었다 |
| XE-a 우세에서 XE-b 가 동등·판정 불가 | 해석 표에 없는 조합이다. 둘째 행('…차이는 확인하지 못했다')을 쓴다. XE-b 열세이면 '(여전히 컸다)' 판 |
| 문장의 a, x | (8절 11 로 바뀜) a·Δ·Holm p 는 같은 대비·같은 n 에서 가져온다. n = 그 대비의 풀 판정이 갈래의 기준(우세 또는 열세)을 충족한 등록 n 가운데 가장 큰 n(전량 > 1,000 > 500). 첫째 문장(XE-a·XE-b 모두 우세)의 a 는 XE-b 의 그 n, 둘째 문장(aw_bn·aw_bl)과 열세 문장의 a 는 XE-a 의 그 n. x = 첫째 문장의 n 에서 R1(교차검증 λ, xh)의 격자 안 설명 비율(1 − MSE_w(R1)/MSE_w(P1))의 대상 평균(%). 가설 표의 `sentence_source_hyp`, `sentence_n`, `sentence_delta`, `sentence_holm_p`, `sentence_rule` 열에 출처를 적는다 |
| 1절 공통 규칙 | '통계적으로 구별되나 크기는 0.5 cm 미만'은 문장의 a 와 같은 Δ 로 정한다. '보정 전 유의'는 같은 대비·같은 n 의 Holm 보정 p 로 정하되, 첫째 문장은 두 가설을 함께 주장하므로 두 Holm p(각자의 기준 충족 n) 가운데 큰 값이 0.05 이상이면 붙인다(보수적 읽기, `sentence_holm_p_other`·`sentence_holm_p_marker` 열). 풀 문장 뒤에 XE-a·XE-b 지역 행(격자 안, 등록 n 전부)의 열세를 '지역 X(가설, n) 에서는 오차가 컸다(Δ, CI)' 로 덧붙인다 |
| n 200 행 | XE-a·XE-b·XE-a0 의 등록 n 은 500·1,000·전량이다. 같은 대비의 n 200 행은 hypothesis 'XE-c', role '보조' 로 쓰고 원 가설은 `hypothesis_registered` 열에 둔다(8절 10) |
| 맹검 열 | `blind` 열에는 1절 어휘(맹검, 비맹검 부분 포함, 재현(비맹검), 등록 시점에 결과 존재(미열람))만 쓰고 근거 문장은 `blind_reason` 열에 둔다(8절 15). 등록 표가 비운 칸(XE-c 의 xh·xt2 행, XE-d, XE-e)은 같은 사유 규칙으로 채운다: P1 과의 대비는 '비맹검 부분 포함(WF9-a 열람)', xh0·xh 변형은 '비맹검 부분 포함(H0 10열이 H19 입력)', xt2·군별 추가·교체 변형과 XE-e 는 '맹검' |
| 재현 관문 | summarize 가 봉인 표를 쓰기 전에 x25 저장소를 WF9 조각과 대조한다(8절 9). 통과 = 공통 키 실패 0, x25 키의 WF9 포괄률 1, 같은 (저장소, 분할)의 WF9 키의 x25 포괄률 1, 기대 x25 단위(조각 `expected_splits`) 누락 0. 통과하지 않으면 XE-a·XE-b 를 '판정 불가(관문: 사유)' 로 쓰고 모든 봉인 표에 `gate` 열, 메타에 `gate_wf9`(키 수·포괄률·누락 단위)를 남긴다. `--gate-wf9 none` 이면 '판정 불가(관문 미실시)' |
| 공통 행 | 총 RMSE 차 |R1(xh) − R1(x25)| < 0.5 cm 인 대상(n 전량 지역 행), O 군이 캐나다에서 빠졌는지, 지도 문구 변경 여부(첫째·둘째 경우만), 'XE-e 는 판정을 바꾸지 않는다' 를 붙인다 |
| XE-c 의 구성 | 위치 안(~l) R1(xh) − R1(x25)·R1(xh) − P1, 격자 안 D0(xh) − D0(x25), 총 R1(xh) − R1(x25), 격자 안 R1(xh0) − R1(x25)·R1(xh0) − P1. n 200·500·1,000·전량, 지역 행과 층화 평균 행에 4분 판정(확인적으로 세지 않는다) |
| XE-a0 | 격자 안 R1(xt2) − R1(x25)(4분 판정)과 서술 행 R1(xt2) − P1 |
| XE-d | SI 변형마다 격자 안·위치 안 R1(변형) − R1(x25), R1(변형) − P1, xh 의 격자 사이 R1 − P1, 분해 표(저장소별 RMSE, 격자 안·위치 안·격자 안 위치 사이 설명 비율, 항등식 점검 열) |
| 봉인 파일 | `data/processed/xbatch/XE_hires_covariates/sealed/<tag>_{xh0,xt2,xh,si}_tests.csv`, `<tag>_xh_hypotheses.csv`, `<tag>_xh_holm.csv`, `<tag>_*_decomp.csv`, `<tag>_si_status.csv`, `<tag>_meta.json`, XE-e 의 `xe_e_xe_e.csv`·`xe_e_xe_e_meta.json`. 변형 묶음별로 나눈 이유는 0.3 열람 순서(xh0 = R2a 제출 뒤, xt2 = xe_feat 최종 해시 커밋 뒤)를 파일 단위로 지키기 위해서다. 모든 표에 `nboot`·`nboot_note`(등록 10,000회가 아니면 '재표집 n회(등록 이탈)')·`gate` 열이 있다(8절 8·9). 스모크(`--smoke`)의 집계 표와 조각은 `<OUT>/smoke/sealed/`(조각은 `smoke/sealed/shards/`)에 둔다(8절 1) |
| 화면 출력 | 모든 모드를 `xbatch_core.restricted_output()` 안에서 돈다. 봉인 파일 이름에 '판정' 영문 단어를 넣지 않았다(행 수·해시 줄이 걸러지지 않게) |
| xt2 표 열람 조건 | `--stage r1b --feat-gate-only --feat data/processed/xe/xe_feat_v1.csv`: R1b 조각 unit.json 의 H0·T2 군 열 해시가 커밋된 최종판(`xe_feat_v1_final.json`, 개정 이력 대조)의 군 열 해시와 모두 같아야 통과(`xe_r1b_feat_gate_check.json` 의 `xt2_open_ok`). 해시만 읽고 표는 열지 않는다(8절 12) |

## 4. 특징 추출(군별 정의, 라벨과 결합하기 전에 고정)

### 4.1 공통

- 대상 행: `fidelity_base_v3.csv` 의 region 이 알래스카(ABoVE_AK, United States (Alaska)), 레나(Lena_RU), 캐나다(ABoVE_CA, Canada, CALM_Canada) 인 17,395행. 좌표 열(loc_id, lat, lon, region)만 읽는다(`read_csv_cols` 가 라벨 열을 거부한다, 시험 (e)).
- 90 % 규칙의 '군의 유한값 비율' = 그 대상의 (행, 군 열) 칸 가운데 유한한 칸의 비율. 이 읽기에서는 열 하나가 크게 비면 군 전체가 빠진다. 예상되는 결과: NCSCD 가 레나 라벨의 40.8 % 에서만 유효해(`xe_stage2_acquire.md` 4절) 레나에서 O 군(SoilGrids 250 m 포함)이 빠질 가능성이 크고, MOD13Q1 이 마감까지 오지 않으면 V 군 전체(WorldCover, CAVM 포함)가 빠진다. 문구 그대로의 읽기라 바꾸지 않았다.
- 마감: 조각 메타에 원자료의 가장 늦은 수정 시각을 적고, 결합 단계에서 원자료 키(O, M, V_wc, V_cavm, V_mod13q1, S)의 시각이 마감(T0 + 72 h 또는 96 h)을 넘으면 그 열을 결측으로 둔 뒤 90 % 규칙을 적용한다('마감을 넘긴 군은 결측 규칙으로 빼고'). 최종판 고정 시각이 T0 + 100 h 를 넘으면 최종 파일에 `after_deadline` 을 적는다('그 시점까지 확보한 군으로 고정').
- 군 열 해시: 결합할 때 군마다 정준 CSV(loc_id + 군 열, 소수 17자리)의 sha256 을 메타에 적는다. R1b(1단계 표)의 xh0·xt2 조각에 그 값을 적고, 집계에서 최종판의 H0·T2 해시와 대조한다(`<tag>_feat_gate.csv`). 2단계 결합 뒤에도 1단계 군 열이 바뀌지 않음을 시험 (i)가 확인한다.
- 창 묶음: 위도 0.1° × 경도 0.25° 칸(224개). 래스터 창은 그 칸의 점 경계 상자에 여유를 더해 읽는다.
- 모자이크 화소 색인: 기준 타일 격자의 정수 화소로 구한다(`grid_rc`). 창 변환의 부동소수 오차로 화소 경계 위의 점이 한 칸 어긋나는 문제를 합성 시험에서 찾아 고쳤다.

### 4.2 T2(Copernicus DEM 30 m, GSW, Hansen)

| 열 | 정의 |
|---|---|
| twi_pt | 점 화소의 ln(a/max(tanβ, 1e-3)), a = (D8 누적 칸 수 + 1)·칸 면적/칸 변 평균. 누적은 점 묶음 경계 상자 ± 0.05°(경도는 /cosφ) DEM 창에서 pysheds(함몰 채움, 평탄 해소)로 구한다. 창 밖 상류 면적은 잘린다(근사, '타일 창 안 pysheds') |
| tpi_90, tpi_270 | z − 3 × 3 평균, z − 9 × 9 평균(가운데 포함, x25 dem_tpi 와 같은 규약) |
| slope_90 | 화소 경사(°, np.gradient)의 3 × 3 평균. 민감도 slope_px = 점 화소 |
| gsw_occ_pt, treecover_pt | GSW occurrence(255 결측), Hansen treecover2000 의 3 × 3 평균. 민감도 *_px = 점 화소 |

- 위경도 격자의 칸 크기는 창 가운데 위도의 cos 로 근사한다(`build_covariates_ext.terrain_tile` 과 같다). 60°·70°N 을 지나는 창은 가운데 타일 격자로 최근접 재표집한다(Copernicus 경도 간격이 1.5″·2″·3″ 로 바뀐다).

### 4.3 O, V, S

| 군·열 | 정의 |
|---|---|
| sg250_* | `data/raw/soilgrids_xe250/` 의 정수 화소 정렬 창만 쓴다(`xe_stage2_acquire.md` 3절 권고). 층마다 창 이름 순 첫 유효 창(값 > 0, 자료 없음 아님)의 점 화소. 단위는 SoilGrids 원값(soc dg/kg, bdod cg/cm³, 점토·모래·실트 g/kg) |
| ncscd_soc_0_30, ncscd_soc_0_100 | NCSCDv2 0.012° SOCC30·SOCC100 점 화소, kg C m-2(원값 hg C m-2 ÷ 10), −32768 결측 |
| wc_*(7) | WorldCover 2021 v200 3 × 3 창(30 m)의 등급 비율, 분모 = 자료가 있는 칸. 범례는 제품 설명서(PUM v2.0) 표로 확인했다: 10 수목, 20 관목, 30 초지, 90 초본 습지, 100 이끼·지의류, 60 나지·희소 식생, 80 영구 수면(계획의 '[미확인: 등급 이름]' 해소) |
| cavm_class | CAVM 래스터(Mendeley 판 2 의 raster_cavm_v1.tif) 점 화소의 부호(1–43 식생 단위, 91 담수, 92 염수, 93 빙하, 99 비북극 육지). 127 만 결측. 범주 부호를 수치 열로 둔다(CatBoost 범주 지정은 등록되지 않았다) |
| ndvi250_jja, evi250_jja, ndwi250_jja | MOD13Q1 16일 합성 가운데 합성 시작일(AppEEARS Date)이 2015–2020 의 6월 1일–8월 31일이고 pixel_reliability ∈ {0, 1} 인 값의 평균. NDWI = (NIR − MIR)/(NIR + MIR)(H19 와 같은 정의). AppEEARS 가 겹치는 합성(예: 5월 25일 시작)도 돌려주므로 시작일로 다시 거른다 |
| ndvi250_max | 합성 시작 일차별 해 평균(기후값)의 최댓값(H19 ndvi_max 가 월 기후값의 최댓값인 것과 같은 구성) |
| S 경로 | MOD10A1 결과가 라벨 500 m 화소의 90 % 이상을 덮으면 MOD10A1, 아니면 MOD10A2 가 같거나 더 많이 덮을 때 MOD10A2, 그 밖에 MOD10A1 결과가 있으면 MOD10A1, 없으면 군 결측(계획의 대체 순서) |
| scd500(MOD10A2, 등록 정의) | 수문년(9–8월) 안 적설 표시 합성(Maximum_Snow_Extent 200) 수 × 8. 수문년은 끝 해로 이름 붙이고 2015–2020 수문년(2014-09-01 – 2020-08-31)을 평균한다. 요청은 2014-09-01 – 2021-08-31 이라 두 읽기를 모두 덮었고, 끝 해 읽기로 고정했다(MODIS 식생의 2015–2020 여름과 같은 해) |
| snowoff·snowon(MOD10A2) | '첫·마지막 무적설 전이 합성의 시작일': 판정 있는 합성(200 적설, 25 무적설, 그 밖의 부호는 건너뛴다)만 시간 순으로 보아 적설 → 무적설로 바뀐 합성이 소멸 전이, 무적설 → 적설로 바뀐 합성이 시작 전이다. 달력 연도 2015–2020 마다 첫 소멸 전이 합성과 마지막 시작 전이 합성의 시작 일차를 구해 평균한다 |
| MOD10A1(주 경로) | 등록 문구가 MOD10A1 의 일 정의를 정하지 않았다. NDSI_Snow_Cover 10–100 = 적설(NDSI 0.1, C6 적설 판정 하한이자 MOD10A2 가 모으는 적설 판정과 같은 기준), 0–9 = 무적설, 그 밖(밤, 구름, 수면, 결측)은 시간상 가장 가까운 판정 날의 상태로 채운다(같은 거리면 앞 날). scd500 = 수문년 안 적설 일수, 전이 규칙은 MOD10A2 와 같다 |

- 캐나다 CALM 8셀(CALM_Canada)은 `xe_stage2_acquire.py` 의 점 집합(REG_TARGET)에 없어 AppEEARS 결과(V 의 MODIS 4열, S)가 결측이다. 캐나다 752행의 1.1 % 라 90 % 규칙에는 영향이 작다(미해결 항목 3).

### 4.4 M(ArcticDEM v4.1)

- 색인: `https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/10m.json`(자식 타일 2,485개, 2026-10-04 HTTP 200 확인)과 32m.json. 타일 이름 rr_cc = (floor(y/1e5) + 41, floor(x/1e5) + 41)(EPSG:3413). 07_41, 08_40, 30_30 의 STAC bbox 로 규칙을 확인했다.
- 창: EPSG:3413 2 km 칸·타일마다 점 경계 상자 ± 200 m 를 `/vsicurl` 로 읽어 `data/raw/arcticdem_xe/windows/` 에 GeoTIFF 로 둔다(pystac_client·boto3 미사용). 10 m 타일이 없거나 창이 모두 결측이거나 읽기에 실패하면 32 m 판(대체 경로).
- 열: ad_slope_30 = 3 × 3(10 m) 경사 평균(°), ad_curv_30 = 3 × 3 라플라시안 평균(1/m), ad_tpi_50 = z − 5 × 5 평균, ad_tpi_150 = z − 15 × 15 평균, ad_rough_50 = 5 × 5 표준편차(x25 dem_rough 와 같은 규약). 32 m 창은 반폭 max(1, round((L/res − 1)/2)) 화소로 근사하고 ad_res 열에 해상도를 적는다. '곡률'의 종류는 등록되지 않아 라플라시안으로 정했다.
- 약관: STAC 의 license CC-BY-4.0. 인용 문구를 `fetch_meta.json` 에 적는다.

## 5. 실행 보호·자원

- 하네스: `xbatch_core.guard`(스모크·세기·집계 외 실행은 WF_RESCALE=1 또는 --allow-local), 허용 표지가 없으면 스레드 ≤ 4, 집계 재표집 ≤ 1,000(표에 '등록 이탈' 열). 2단계 변형(xh, SI)은 최종 해시 파일의 sha256 이 표와 같고 그 sha256 이 계획 개정 이력에 커밋되어 있을 때만 돈다. `--allow-unfinal` 은 `--smoke`·`--count-only` 에서만 받는다. 요청한 변형이 '최종 해시 없음'으로 빠지면 적합 전에 멈춘다. 빈 특징 표(`--feat none`)로는 x25 밖의 변형을 적합하지 않는다. 특징 표 메타가 없거나 대상의 `group_decisions` 가 없으면 멈춘다(90 % 규칙을 알 수 없다)(8절 2·3·6).
- 로컬 자원(허용 표지 없는 실행): 시작 전 가용 메모리 30 GB 확인(모자라면 60 s 간격으로 최대 30 분 기다린 뒤 거부), 상주 메모리 10 GB 감시 스레드(넘으면 종료 코드 3, 워커에도 건다), MALLOC_ARENA_MAX 가 있으면 주소 공간 상한 10 GB 도 건다(XI·XJ 와 같은 조건). 스모크는 스레드 2 로 줄이고 코어 묶음이 4개를 넘으면 거부한다(8절 7·16).
- 특징 추출: 워커 ≤ 4(spawn), 워커당 스레드 1, nice 10, `--mem-gb`(기본 10 GB, RLIMIT_AS), 시작 전 가용 메모리 30 GB 확인, 최대 RSS 를 조각 메타에 적는다. GDAL 임시 파일·캐시는 `CPL_TMPDIR`(기본 `data/raw/xe_tmp`)·`GDAL_CACHEMAX` 512 MB 로 /home 의 `data/raw` 아래에 둔다(계획 1절 디스크, 8절 추가 변경).
- 주소 공간 상한(prlimit --as)과 CatBoost: 같은 프로세스에서 rasterio·pysheds(numba)·sklearn 을 먼저 부른 뒤 10 GB 주소 공간 상한 아래에서 CatBoost 적합을 하면 `get_feature_importance` 안에서 멈추는 것을 시험 실행에서 보았다(CPU 100 %, 4분 넘게 진행 없음). 상한 없이, 또는 32 GB 상한에서는 정상이다. 실제 사용량(RSS)은 작다. 특징 추출과 하네스는 다른 프로세스에서 돌므로 실행에는 영향이 없으나, 로컬 시험·스모크에 prlimit --as 를 쓸 때는 이 점을 고려한다.

## 6. 미해결 항목

1. 지역 피복 변형(cov_AK, cov_LE): 원자료(ABoVE ORNL 1691, Lisovski 2025)와 열 정의가 등록되지 않았고 취득되지 않았다. 이번 묶음에서는 미실행(등록 이탈)으로 기록한다(8절 5). 취득·추출 경로는 만들지 않았다.
2. (해소, 8절 4) XE-e 를 구현했다(`--xe-e-prep`, `--xe-e`). 등록되지 않은 세부의 읽기는 8절 4 에 있다.
3. `xe_stage2_acquire.py` 의 점 집합에 CALM_Canada 8셀이 없다(MODIS 결측). AppEEARS 요청을 다시 낼 때 넣을 수 있다.
4. (해소, 8절 2) `xbatch_core.PAYLOAD_OPTIONAL` 에 `xe_feat_v1_meta.json`, `xe_feat_v1_final.json`, XE-e 의 `xe_vwc_v1.csv`·메타, 계획 문서 사본(git 없는 노드의 개정 이력 대조)을 넣었다. `PAYLOAD_REQUIRES` 가 표가 있으면 메타도 있어야 함을 묶음 정보에 적는다.
5. ArcticDEM 창 받기(m-fetch)와 T2·O·V·S 의 전체 추출은 실행하지 않았다(1c·2단계 작업). 1c 명령은 8절 끝에 있다.
6. 개정 이력 기재 사항(사용자가 커밋할 때 계획 개정 이력에 적을 문구, 8절 '개정 이력 기재 사항').

## 7. 실행 기록(2026-10-04, 로컬, 세기·스모크 범주)

- 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --as=34359738368 taskset -c 100-103 python3 -m pytest -q tests/test_x_xe.py` → 22개 통과(약 60 s). 10 GB 주소 공간 상한에서는 5절의 CatBoost 멈춤이 시험 9번째에서 재현되어 32 GB 상한으로 돌렸다(실제 RSS 는 1 GB 미만).
- 세기(`--count-only --feat none`, 모든 군 포함으로 센 상한): R1b(x25·xh0·xt2) 단위 222, 적합 19,650건, 약 0.9–1.1 워커·시간(WF9 실측 환산·h54 모형 추정, 4스레드). R3(x25·xh·SI 7변형, 지역 피복 2변형은 자료 없음) 단위 666, 적합 50,886건, 약 2.9–3.4 워커·시간. 레나 분할 24 는 채점 블록 1개로 빠진다(WF9 와 같다). elm 워커 22개로 두 작업 모두 실행 0.05–0.15 h 수준이다. 계획 2.5 의 '약 2 워커·시간[추정]'보다 큰 것은 x25 를 두 작업에서 다시 적합하고 SI 변형 7개를 더했기 때문이다.
- 특징 추출 시험 실행(캐나다 752행, 출력은 스모크 경로 `data/processed/xbatch/XE_hires_covariates/smoke/xe/`): H0·T2 23 s(창 묶음 51개, 워커 2, 최대 RSS 337 MB). 전체 3대상 창 묶음은 224개라 T2 는 수 분 규모다(계획의 4–5 h 추정보다 짧다). 같은 경로에서 O·V·S 를 읽어 본 유한값 비율(캐나다, 좌표만 사용): SoilGrids 250 m 0.544(정렬 창 취득 진행 중), NCSCD 0.999, WorldCover 0.987, CAVM 1.000(대부분 99 비북극 육지), MOD13Q1·S 0(AppEEARS 본 요청 전).
- ArcticDEM 창 받기 시험(캐나다 창 8개): 10 m 6개 성공(각 약 0.4 s), 2개는 ArcticDEM 범위 밖 CALM 지점(56.6°N 등)이라 10·32 m 타일이 없다. `data/raw/arcticdem_xe/` 에 색인 JSON 2개와 창 6개(1.1 MB)를 두었다.
- 스모크(`--smoke`, 캐나다 분할 7, n 200·전량, 추출 1, seed 0, x25·xh0·xt2, 스레드 2, 코어 4개): 단위 3개 13 s, 최대 RSS 299 MB. 집계 표는 `smoke/sealed/` 에만 썼다(열지 않았다). 조각 3세트(9개 파일)는 처음 `<OUT>/shards/` 에 썼다가 검토 뒤 열지 않은 채 `smoke/sealed/shards/` 로 옮겼다(8절 1).
- 재현 관문 시험: 스모크 x25 의 총·격자 안·격자 사이 저장소를 WF9 hematite 조각(`results/rescale_wf3/.../wf9__cpu__Canada__r__s7_*`)과 대조했다(`--gate-wf9 --gate-level local_rescale`). 공통 키 81개(P0, P1, Pk, Pc, R1, Re, D0) 모두 통과, 최대 |ΔSSE| 6.4e-10 cm², 최대 상대 차 2.7e-15. 본 관문(elm·hematite 0)은 R1b·R3 결과로 다시 한다.

## 8. 검토 반영(2026-10-04 저녁, 적대적 검토 결함 17건)

- 근거: 적대적 검토가 낸 결함 목록 17건(중간 2건은 1·2·3·4번, 낮음 13건). 처리 전에 각 항목을 계획 2.5·1절·0.3 과 다시 대조했다. 계획은 고치지 않았고, 등록 가설·팔·분할·CI 방법·판정 규칙·Holm 가족(m 6)은 바꾸지 않았다. 동결 모듈 4개의 sha256 앞 16자는 계획 머리말 값과 같다(h40 7098f59dabe2b73f, h42 22e215e435e157be, h54 cf9a1f6d0929c892, h41 492373e4d37b5ea4).
- 이 절을 쓰면서 실행한 것: 합성 자료 시험(`tests/test_x_xe.py`), `--count-only`(r1b, r3, `--feat none`), `--xe-e-prep`(좌표·블록·VWC 만 읽는 세기 범주). 모형 적합과 집계 표 계산, 1c 추출(H0 결합·T2 추출)은 하지 않았다. 화면 출력은 모두 `restricted_output()` 안이었고 RMSE·Δ·판정은 없었다.
- 아래 번호는 결함 목록 순서다.

| 번호 | 결함 요지 | 처리 | 코드 위치 |
|---|---|---|---|
| 1 | 스모크 조각(실제 자료 적합 결과: 키별 RMSE·bias, unit 의 E0·E_A)이 봉인 밖 `shards/` 에 있었다 | 스모크의 조각 경로를 `<OUT>/smoke/sealed/shards/` 로 바꿨다(집계 표와 같은 봉인 폴더 아래). 기존 스모크 조각 3세트(`xe_r1b_smoke__cpu__Canada__r__s7__{x25,xh0,xt2}_{runs.csv,blocksse.npz,unit.json}`, 15:57 생성) 9개 파일은 열지 않은 채 그 폴더로 옮겼고 `<OUT>/shards/` 는 비어 있다(존재하지 않는다). 봉인 밖에 남긴 스모크 산출은 해시·SSE 차만 있는 `xe_r1b_smoke_feat_gate.csv`·`xe_r1b_smoke_gate_wf9.csv` 와 특징 표 `smoke/xe/` 다 | `finalize`(`a.SHARDS`) |
| 2 | 최종 해시가 없으면 enumerate_units 가 xh·SI 를 미리 빼서 858행의 거부가 발동하지 않았다. 묶음에 메타·최종 JSON 이 없었다 | `check_requested`: 본 실행·스모크에서 요청한 변형 가운데 '최종 해시 없음'으로 빠진 것이 하나라도 있으면 적합 전에 SystemExit. `xbatch_core.PAYLOAD_OPTIONAL` 에 `xe_feat_v1_meta.json`, `xe_feat_v1_final.json`, XE-e 의 `xe_vwc_v1.csv`·메타, 계획 문서 사본을 더했고 `PAYLOAD_REQUIRES` 로 표가 있으면 메타가 함께 있어야 함을 묶음 정보 missing 에 적는다. `payload_manifest` 는 있는 선택 파일의 sha256 을 모두 기록한다 | `check_requested`, `xbatch_core` 18절 |
| 3 | `--allow-unfinal` 이 본 실행에서도 받아지고 집계가 조각의 최종판 조건을 보지 않았다 | `finalize` 가 `--smoke`·`--count-only` 밖의 `--allow-unfinal` 을 거부한다. `summarize` 의 `check_final_shards`: needs_final 변형 조각의 `allow_unfinal` 참, `feat_final_ok` 거짓, `feat_sha256` ≠ 집계 쪽 커밋된 최종 해시 가운데 하나라도 있으면 비스모크 집계를 거부하고 사유를 메타 `unfinal_shards` 에 적는다 | `finalize`, `check_final_shards` |
| 4 | XE-e 미구현 | 구현했다. `--xe-e-prep`(세기 범주)가 `inputs/xe_vwc_v1.csv`·메타를 만들고 `--xe-e`(본 실행 보호)가 봉인 폴더에 `xe_e_xe_e.csv`·메타를 쓴다. 등록되지 않은 세부의 문자 그대로 읽기는 아래 '4 의 세부' | `vwc_prep`, `xe_e_run` |
| 5 | 지역 피복 변형(cov_AK·cov_LE)의 취득·추출 경로가 없어 기록 없이 사라진다 | 경로를 만들지 않았다(원자료 미취득, 열 정의 미등록, 2단계 취득 표에도 없다). '미실행(등록 이탈)' 을 세기의 건너뜀 사유, SI 상태 표 `<tag>_si_status.csv`, 모듈 머리에 명시한다. 개정 이력 기재 사항에 넣었다 | `x_hires_registry.COVER_NOT_RUN`, `variant_status` |
| 6 | `--feat none` 이 run 에서 거부되지 않고 decisions None 이 fail open | `check_feat_for_fit`: run·smoke 에서 빈 표와 x25 밖 변형이 함께 있으면 SystemExit. `FeatTable.decisions`: 실제 표의 메타에 `group_decisions` 가 없거나 대상이 빠지면 RuntimeError(빈 표는 세기 전용으로만 None) | `check_feat_for_fit`, `FeatTable.decisions` |
| 7 | 로컬 모드에서 메모리 확인·상한이 없었다 | `local_resources`: 허용 표지가 없으면 `XB.require_memory(30 GB, 최대 1,800 s 대기)`, 상주 메모리 10 GB 감시 스레드(넘으면 종료 코드 3), MALLOC_ARENA_MAX 가 있으면 `XB.limit_memory(10 GB)`. 풀 워커(`_worker_init`)에도 감시를 건다. 가용 메모리를 화면과 메타에 적는다 | `local_resources`, `rss_watchdog` |
| 8 | 허용 표지 없는 집계가 재표집 1,000회로 봉인 판정 표를 쓰면서 표에 표지가 없었다 | 모든 봉인 표에 `nboot`·`nboot_note`('재표집 n회(등록 이탈, 등록 10,000회)') 열, 메타에 `nboot_registered`. 거부 대신 열을 택한 이유는 로컬 집계가 스모크 점검에 필요하기 때문이다. 본 집계는 WF_RESCALE=1 또는 `--allow-local` 에서 10,000회로 돈다 | `summarize`(`mark`) |
| 9 | 관문이 선택이고 봉인 표 뒤에 돌며 누락을 잡지 못했다 | `gate_wf9` 가 봉인 표보다 먼저 기본으로 돈다. 통과 = 공통 키 실패 0, x25 키의 WF9 포괄률 1, 같은 (저장소, 분할)의 WF9 키의 x25 포괄률 1(스모크 제외), 조각 `expected_splits` 기준 x25 단위 누락 0. 결과(키 수·실패 수·포괄률·누락 단위)를 메타 `gate_wf9` 와 모든 봉인 표 `gate` 열에 적고, 통과하지 않으면 XE-a·XE-b 판정을 '판정 불가(관문: 사유)' 로 쓴다 | `gate_wf9`, `gate_coverage`, `expected_x25`, `verdict_rows` |
| 10 | n 200 행이 XE-a·XE-b '주' 로 적혔다 | `row_label`: XE-a·XE-b·XE-a0 의 등록 n(500·1,000·전량) 밖 행은 hypothesis 'XE-c', role '보조'. 원 가설은 `hypothesis_registered` 열 | `row_label`, `run_contrasts` |
| 11 | 문장의 a·'보정 전 유의'·효과 크기 표기가 다른 대비·다른 n 에서 왔다 | `pick_n`: 그 대비의 풀 판정이 갈래의 기준을 충족한 등록 n 가운데 가장 큰 n(전량 > 1,000 > 500)에서 a·Δ·Holm p 를 같은 대비로 가져온다. 첫째 문장은 XE-b, 둘째·열세 문장은 XE-a. 첫째 문장은 두 가설을 함께 주장하므로 '보정 전 유의' 는 두 Holm p 가운데 큰 값으로 정한다(보수적 읽기). 출처 열 `sentence_source_hyp`, `sentence_n`, `sentence_delta`, `sentence_holm_p`, `sentence_holm_p_other`, `sentence_holm_p_marker`, `sentence_rule` | `pick_n`, `verdict_rows` |
| 12 | final_ok 가 로컬 파일만 보고 '개정 이력에 커밋' 을 확인하지 않았다. xt2 열람 전 대조 단계가 없었다 | `final_committed`: 최종 sha256 이 git HEAD 의 계획 문서 '## 개정 이력' 절에 있어야 한다(git 이 없는 Rescale 노드는 묶음의 계획 문서 사본). `--feat-gate-only`: R1b 조각 unit.json 의 H0·T2 군 열 해시와 커밋된 최종판의 군 열 해시를 대조해 `xe_r1b_feat_gate_check.json` 의 `xt2_open_ok` 를 쓴다(해시만 읽는다). 조각 unit 에 `feat_final_commit` 문구를 적는다 | `final_committed`, `FeatTable.final_ok`, `feat_gate_check` |
| 13 | sg250 의 cfvo·phh2o 제거 효과 혼입이 기록되지 않았다 | 변형 정의는 등록 문구('5 km 9열 대신 250 m 열') 그대로 두고, 혼입을 2절 표, `SI_NOTES`, SI 상태 표와 봉인 분해 표의 `note` 열에 적는다 | `SI_NOTES`, `variant_status` |
| 14 | 90 % 규칙으로 변형의 모든 군이 빠진 대상의 처리 근거가 없었다 | 택한 읽기: 변형의 군 가운데 일부가 빠지면 남은 군으로 돈다(계획의 'xh = xt2' 규정과 같은 축소). 모든 군이 빠지면 그 대상에서 그 변형을 건너뛰고 풀은 1절 층화 평균 규칙의 '부분(지역 k/3)' 표지를 받는다. xh0 = x25 로 돌리지 않는 이유: 같은 열·같은 seed 의 적합은 Δ ≡ 0 이 되어 풀 평균을 0 쪽으로 끌면서 '지역 k/m' 표지도 붙지 않아, 지역 수가 줄었다는 사실이 표에서 사라진다. 건너뜀은 메타 `variant_target_skips` 와 봉인 표 `excluded_targets` 열에 남는다. 코드는 바꾸지 않았고 개정 이력 기재 사항에 넣었다 | `x_hires_registry.variant_columns` |
| 15 | XE-c 맹검 열에 근거 문장이 섞여 1절 어휘를 벗어났다 | `blind` 열은 1절 어휘만, 근거는 `blind_reason` 열. 등록 표가 비운 칸의 채우기 규칙은 3절 표 | `blind_of` |
| 16 | 스모크의 코어 4개·스레드 2 가 강제되지 않았다 | `finalize` 가 허용 표지 없는 스모크의 스레드를 2 로 줄이고, `check_smoke_env` 가 `sched_getaffinity` 4개 초과 또는 스레드 2 초과를 거부한다 | `check_smoke_env` |
| 17 | 시험 파일이 묶음에 없는 추출 모듈(rasterio·pysheds)을 모듈 수준에서 읽었다 | 지연 읽기(`_Lazy`)와 skipif(모듈 없음, rasterio 없음, pysheds 없음)로 바꿨다. 두 추출 모듈은 묶음에 넣지 않았다(Rescale 환경에 rasterio·pysheds 가 없고 추출은 로컬 1c 작업이다). 시험 파일 이름은 작업 지시(`tests/test_x_xe.py`)대로 두었다(1절 1) | `tests/test_x_xe.py` 머리 |

**4 의 세부(XE-e, 등록되지 않은 세부의 문자 그대로 읽기)**

- 부분 집합 소속: 계획 문구 'ABoVE 현장 VWC 부분 집합(알래스카 2,468셀·31블록, 캐나다 602셀·15블록, 15 m 안)' = 15 m 안에 VWC 위치(소수 4자리 위경도, VWC 0–100 %, 유효 좌표. 측정 하단이 결측인 층 없는 행도 위치로 센다)가 있는 라벨 셀. 거리는 EPSG:3338 좌표의 cKDTree(설계 보고서 부록 A 와 같다). `--xe-e-prep` 결과: 알래스카 2,468셀·31블록, 캐나다 602셀·15블록으로 계획 수치와 같다(`inputs/xe_vwc_v1_meta.json` counts). 비 GPR 층 값이 있는 셀은 알래스카 1,331·27블록, 캐나다 598·15블록, GPR 값이 있는 셀은 알래스카 1,470·17블록, 캐나다 0.
- P1 격자 안 잔차: 격자 묶음(블록, 기온 √TDD) 안에서 P1 = E1·s 는 상수이므로 (y − E1 s) − 묶음 평균 = y − ȳ_g 이고 E1·라벨 수·분할과 무관하다. 그래서 라벨 수와 분할을 정하지 않고 대상의 채점 가능 셀(eval_mask) 전체에서 한 번 구한다. 한 셀 묶음의 잔차는 0 이라 설명 비율은 2셀 이상 묶음 셀에서 낸다. 실행 메타 counts `<대상>|any|member` 에 소속 셀·블록 수와 2셀 이상 묶음 셀·블록 수를 함께 적어 계획 수치와 대조한다.
- 층과 기기: GPR 기기 행 = GPR 판(민감도). 그 밖(HydroSense, 막대, DualEM, 기기 미표기)은 측정 하단 0 < d ≤ 12 cm = shallow, d > 12 cm = deep 의 두 예측 변수(설계 보고서 5.8 (i)), 하단 결측은 층 없음(소속에는 세고 예측 변수는 결측). 계획 2.5 는 'GPR'/'비 GPR' 만 구분하므로 DualEM 은 비 GPR 에 둔다(설계 보고서가 GPR·DualEM 을 함께 묶은 것과 다르며 계획 문구를 우선했다).
- 날짜 맞춤: 같은 위치에서 ALT 를 잰 해(ALT_instrument 가 있는 행의 날짜. ALT 값은 읽지 않는다)의 VWC 행이 있으면 그 행만, 없으면 그 위치·층의 모든 행. 값 = 위치·층·기기 평균 → 기기 평균의 평균 → 셀 반경 15 m 안 위치 값의 평균.
- 교차검증과 학습기: 블록 K = min(5, 블록 수) 묶음(h42.cv_folds_of 와 같은 배정 규칙, seed_of('xe-e', 대상, 변수 묶음, 부분 집합, 'r', 0)), 1차 회귀(절편, 결측은 학습 묶음 평균 대치 + 결측 표시 열, 단조 제약 없음)와 catboost_lo(seed 0·1, 평균 행 추가). 설명 비율 = 1 − Σ(e − ê)²/Σe²(셀 가중, 묶음 밖 예측). 블록 2개 미만이면 '계산 불가'.
- 주 행 = 비 GPR 두 열, 등록 부분 집합. 민감도 = GPR 판, 예측 변수가 유한한 셀만 쓴 판. 0.5 cm 등가 설명 비율(알래스카 7.9 %, 캐나다 5.0 %)을 참고 열로 둔다. 맹검 표지 '맹검'(VWC 와 P1 잔차의 관계를 계산한 적이 없다). 열람 시점은 xh·SI 표와 같이 둔다. 판정을 바꾸지 않는다.
- 읽는 열: ABoVE 의 latitude, longitude, date, ALT_instrument, VWC_instrument, depth_bottom, VWC(ALT·ALT_err 값은 읽지 않는다), v3 의 loc_id, lat, lon, region, block. 시험 (l)이 확인한다.

**바꿔 적용했거나 거절한 항목(이유)**

- (가) 2 의 대안 가운데 'PAYLOAD_CODE_GLOBS 에 `scripts/1_data_prep/xe_*.py` 추가' 는 택하지 않았다. 추출은 로컬 1c 작업이고 Rescale 환경에 rasterio·pysheds 가 없어 묶음에 넣어도 돌지 않는다. 시험 쪽을 지연 읽기·skipif 로 바꿨다(17).
- (나) 8 은 거부 대신 '등록 이탈' 열을 택했다. 허용 표지 없는 집계는 스모크 점검에 필요하고, 본 집계는 10,000회로 돈다.
- (다) 14 는 코드를 바꾸지 않고 해석을 기록했다(위 표).
- (라) 5 는 추출 경로를 만들지 않았다. 열 정의가 등록되지 않아 지금 만들면 등록 뒤 설계가 된다.
- (마) 11 은 결함이 요구한 '같은 대비·같은 n' 에 더해, 첫째 문장의 '보정 전 유의' 를 두 가설의 Holm p 가운데 큰 값으로 정했다(1절 규칙의 보수적 읽기. 판정 규칙이 아니라 문장 표지 규칙이다).

**추가 변경(결함 목록 밖, 계획 1절 규약)**

- `xe_point_covariates.py`: GDAL 임시 파일·캐시를 `CPL_TMPDIR`(기본 `data/raw/xe_tmp`, 없으면 만든다)·`GDAL_CACHEMAX` 512 MB 로 /home 의 `data/raw` 아래에 둔다(1절 디스크. 루트·/tmp 는 97 % 사용).
- `_worker_init`: 허용 표지 없는 풀 워커에도 10 GB 상주 메모리 감시를 건다.
- XE-e 준비 표 `inputs/xe_vwc_v1.csv` 를 소속 기준 열 `n_loc_any` 를 넣어 다시 만들었다(21:17, 세기 범주. 이전 판 16:40 은 층이 있는 위치만 세어 알래스카 소속이 계획 수치와 달랐다).

**개정 이력 기재 사항(사용자가 커밋할 때 계획 개정 이력에 적을 문구. 이 기록은 계획을 고치지 않는다)**

1. XE 스모크 조각(캐나다 분할 7, 실제 자료 적합 결과)을 봉인 밖 `shards/` 에 썼다가 열지 않은 채 `smoke/sealed/shards/` 로 옮겼다(2026-10-04 16:20).
2. XE SI 변형 '지역 피복(알래스카 ABoVE 30 m, 레나 Lisovski 2025 10 m)' 은 취득·추출 경로를 만들지 않아 미실행이다(등록 이탈. SI 상태 표에 적는다).
3. XE-e 의 등록되지 않은 세부(부분 집합 소속 기준, 잔차 정의, 층, 날짜 맞춤, 교차검증, 학습기)는 위 '4 의 세부' 의 문자 그대로 읽기로 고정했다. 소속 셀·블록 수는 계획 수치와 같다.
4. 90 % 규칙으로 변형의 모든 군이 빠진 대상에서는 그 변형을 건너뛰고 풀에 '부분(지역 k/3)' 을 붙인다(위 14).
5. sg250 변형은 cfvo·phh2o 제거 효과가 섞인다(위 13).
6. 시험 파일 이름 `tests/test_x_xe.py`(계획의 `tests/test_x_hires.py` 대신).

**시험·세기(이 절, 2026-10-04 21시대, 로컬. `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c 100-103`, 시험은 `prlimit --as=34359738368`(5절의 CatBoost 멈춤 회피), 가용 메모리 42–44 GB)**

- `python3 -m pytest -q tests/test_x_xe.py`: 33 passed in 239.26s. (a)–(m) 모두 포함. 새 시험: XE-e 소속 기준(층 없는 위치, n_loc_any), 첫째 문장의 Holm 표지.
- `--count-only --stage r1b --feat none`: 작업 단위 222(x25·xh0·xt2 각 74), 적합 19,650건, 건너뜀 1(레나 분할 24, 채점 블록 < 2), h54 모형 추정 1.06·WF9 실측 환산 0.92 워커·시간(4스레드), 워커 22개 약 0.05 h.
- `--count-only --stage r3 --feat none`: 작업 단위 666(x25·xh·add_T2·add_M·add_O·add_V·add_S·sg250·xh_px 각 74), 적합 50,886건, 건너뜀 7(레나 분할 24 1, cov_AK·cov_LE 미실행 2, 대상 아님 4), 3.39·2.93 워커·시간, 워커 22개 약 0.15 h. 7절의 세기와 같다.
- `--xe-e-prep --feat none`: VWC 행 179,378(층 없음 6,099), 위치 2,427(층 있음 2,416), 셀 3,070. 소속 셀·블록 알래스카 2,468·31, 캐나다 602·15(계획 수치와 같다). 3 s.

**다음 작업 1c(H0 결합·T2 추출. 이 절에서는 실행하지 않았다. ROOT 에서, 가용 메모리 30 GB 이상 확인 뒤)**

```
cd /home/willy010313/Polar_Bigdata
free -g                                      # 가용(available) 30 GB 미만이면 기다린다. 명령도 --min-free-gb 30 으로 확인해 거부한다
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c 100-103 \
  python3 scripts/1_data_prep/xe_point_covariates.py h0 t2 assemble qa --workers 4 --mem-gb 10 --min-free-gb 30 \
  2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE' | tee data/processed/xbatch/XE_hires_covariates/log_1c_h0_t2.txt
# 산출: data/processed/xe/parts/xe_{H0,T2}_v1.csv(+ 메타), data/processed/xe/xe_feat_v1.csv, xe_feat_v1_meta.json(stage1, 90 % 규칙 결과·군 열 해시),
#       xe_feat_v1_qa.csv, xe_feat_v1_qa_corr.csv(등록 전 허용 확인: 유한값 비율·격자·위치 안 변동·새 열 사이 상관. ALT·잔차 통계 없음)
# 확인(세기 범주): 실제 1단계 표로 R1b 단위를 센다(xh0·xt2 가 H0·T2 포함으로 222 단위여야 한다. 90 % 규칙으로 군이 빠지면 건너뜀 사유에 나온다)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c 100-103 \
  python3 scripts/3_deep_learning/x_hires_covariates.py --count-only --stage r1b --feat data/processed/xe/xe_feat_v1.csv
```
