# x_product_comparison 구현 기록(XG, 계획 2.7)

- 대상: `scripts/3_deep_learning/x_product_comparison.py`, `tests/test_x_xg_xh.py`(XG 시험 (a)–(j), (t))
- 근거: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 2.7, 1절, 0.3·0.4(등록 이탈 WRAPUP 10), `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 1.1 (a)–(e), `docs/research/2026-10-04/alt_products.md` 2.2·4·7·9·10
- 작성: 2026-10-04. 계획은 고치지 않았다. 아래는 계획 문구가 정하지 않았거나 두 가지로 읽힐 수 있는 곳의 처리다.
- 이 기록을 쓰면서 한 계산: 합성 시험, `--count-only`(조각 수·블록 수·적합 수), 제한 스모크(레나 분할 1), 재현 관문 g1·g1b·g2(통과 여부만). 제품 값 추출과 누설 마스크 표는 만들지 않았다(로컬 1c 단계).

## 1. 파일 위치

| 계획 문구 | 처리 |
|---|---|
| `scripts/1_data_prep/xg_extract_products_v1.py`, `scripts/2_evaluation/xg_leak_mask_v1.py`, `scripts/2_evaluation/x_product_comparison.py`, `tests/test_x_products.py` | 작업 배정에 따라 한 파일 `scripts/3_deep_learning/x_product_comparison.py` 의 단계(`--extract`, `--mask`, 재채점, `--part cell`)로 두었다. 시험은 `tests/test_x_xg_xh.py` 에 계획이 정한 항목(합성 래스터 표집, 마스크가 지정 블록을 뺀다, 재채점 SSE 일치, P1@{p} 가 P1 과 같은 수축 식, 단위·좌표계·결측값 기록)을 모두 넣었다. 4절 묶음 목록의 `scripts/3_deep_learning/x_*.py`, `tests/test_x_*.py` 에 들어간다 |

## 2. 추출

| 항목 | 처리 | 근거 |
|---|---|---|
| T0 뒤 추출 | `git merge-base --is-ancestor 2678100 HEAD` 가 참일 때만 추출한다 | 0.3 '그 밖의 규칙' |
| 추출 셀 | v3 직접 라벨(F4_direct) 전부와 네 실행 표의 새 셀(loc_id 기준 18,000여 셀). 라벨 값 열은 읽지 않는다 | 채점 셀만으로는 P1@{p}(A 셀)와 B:{p}_aff·P0@{p}(원천 셀)를 계산할 수 없다. alt_products 7.7 의 'v4 전 셀 좌표' |
| 라벨 연도 | v3 셀은 `data/processed/m1/a2_year_matched_tdd_cells.csv`, 새 셀은 `fidelity_base_v4_labels.csv`(part = new)의 year_min·year_max | v3 셀의 연도는 v4 라벨 표에 없다(이번 확인: v3 17,572행 가운데 연도 결합 0) |
| 연도 정합 | 라벨 연도와 겹치는 제품 연도 가운데 값이 있는 해의 평균. 겹침이 없거나 겹친 해가 모두 결측이면 기간 평균과 match_flag = period_mean | 계획 2.7 '시간 정합' |
| 화소 | 최근접 화소(주), 3 × 3 유한 값 평균(민감도 열 value3_cm). 화소 중심까지 거리가 화소 크기 × 1.5 를 넘으면 결측 | alt_products 7.7 |
| 단위 | CCI 는 파일 속성 'metres'(×100), Yi·Kimball 은 'm'. Wei·Aalto 는 속성이 없어(이번 메타데이터 확인) 유한 값 중앙값 < 10 이면 m 로 본다. `--units wei=cm` 로 바꿀 수 있고 정한 방식과 중앙값을 `xg_meta.json` 에 쓴다 | alt_products 11절 3 '[미확인]' |
| 결측값 | CCI 는 _FillValue 65535(netCDF 자동 가림), Wei 는 nodata −9999, Aalto 는 nodata −3.4e38(< −1e30 도 결측), Yi·Kimball 은 −999. ds2332 의 1.0·2.0 화소 결측 규칙은 제품 표 MISSING_CODES 에 두었다(XG 등록 제품에는 ds2332 가 없다) | 계획 2.7 마지막 행 |
| md5 | Wei zip 3개와 Aalto zip 을 alt_products 9절 값과 대조하고 다르면 중단한다(관문 g3 의 기록). Yi·Kimball 은 내려받은 .sha256 파일과 대조한다 | 계획 2.7 재현 관문 |
| cci4 | 추출하지 않고 자료의 cci_alt 열(v4 다년 평균)을 쓴다 | 계획 2.7 '1단계는 CCI v4(보유)' |
| 민감도 제품 | cci5s·weis(기간 평균)를 서술 표에만 둔다 | alt_products 7.4 민감도 |

## 3. 키와 마스크

| 항목 | 처리 | 근거 |
|---|---|---|
| B:{p}_raw 의 결측 셀 | h42 anchor_fill 과 같이 ρ·s(ρ = 원천 셀 b/s 중앙값, 라벨 미사용)로 채운다. 채운 셀 수를 xg_product_key_notes 에 쓴다 | 'LGX 키 이름 규칙'(B:cci_raw 와 같은 처리). 채점 셀 집합을 방법 사이에 같게 둔다 |
| 알래스카 전용 제품(Yi·Kimball)의 전이 | 원천에 값이 없어 ρ·c0 를 정할 수 없다. B:yk_raw 만 A 쪽 ρ 로 채워 내고 P1@yk·B:yk_aff 는 내지 않는다. 지역 내(XG-5)는 B:yk_raw 만 쓴다 | 계획 2.7 XG-5 'Yi·Kimball(알래스카)' |
| P1@{p} | c_n = shrink(ls_E(y_A[sel], b_A[sel]), c0, n, κ 10), 추출은 h40.draw_cells(LG 추출) | 관문 g1b 에서 LGX x9 의 P1@cci 와 650키가 같았다 |
| 블록 마스크의 셀 | 대상 지역의 그 블록 안 모든 셀(채점 셀만이 아님) | 계획 문구 '그 거리 안의 셀이 하나라도 있는 B 블록'. 더 보수적이다 |
| 거리 경계 | d km 안 = 거리 ≤ d | alt_products 4.2 표('≤ 5 km') |
| L0 제품 | 학습 지점 집합이 비어 마스크가 없다. 두 공동 주 마스크의 표는 같은 값이다 | alt_products 7.3 |
| Aalto | CALM ∪ GTN-P 활동층 좌표의 대리 마스크(L2) | alt_products 4.1 |
| P* | LG 대상은 LGX x9 조각의 P0@tddm·P1@tddm 을 병합한다. LGD 대상(새 셀)은 tdd_matched 가 없어 P* 가 없다 | 계획 2.7 키 목록 |

## 4. 판정과 문장

| 항목 | 처리 |
|---|---|
| Holm | 공동 주 마스크마다 따로 6개 가족(m = 6, 두 가중 가운데 큰 양측 p, 행 없음·판정 불가 p 1)을 보정한다. 동등성 p(h42 stats_row 의 p_eq)도 같은 방식으로 따로 보정한다 |
| 문구 갈래(WRAPUP 1.1) | (a) 우세·열세이고 Holm p < 0.05 → 그 갈래. (b) Holm p ≥ 0.05 → 미결정 갈래 + '보정 전 유의'. (c) 동등이고 보정 동등성 p ≤ 0.025 → 동등, 아니면 미결정 + '보정 전 동등'. (d) 미결정. (e) 풀 표지가 '부분(지역 k/m)'이면 문장 앞 표지에 넣는다 |
| 두 마스크 결합 | 갈래가 같으면 그대로. 다르면 약한 쪽(우세·열세 > 동등 > 미결정 > 판정 불가)과 '마스크 의존'. 우세와 열세가 엇갈리면 미결정과 '마스크 의존'(계획은 '약한 쪽'만 정했다) |
| XG-1 문장의 CALM 절 | '지도 X 는 대상 지역 안 CALM 지점도 학습에 썼고 ...'는 학습 지점이 있는 제품(Wei)에만 붙인다. CCI(L0)에는 사실이 아니어서 붙이지 않는다. 판정 규칙은 바꾸지 않았다 |
| XG-2·XG-3 열세 | 학습 자료가 겹치는 제품(Wei, Aalto)이면 '(학습 자료 중복 가능)'을 붙인다(WRAPUP 10 문장 규칙) |
| 공통 표지 | 모든 판정 행과 문장에 '등록 이탈(WRAPUP 10)', 제품 ALT 정의, 마스크 전후 P0 RMSE 를 넣는다. 지역 행에 열세가 있으면 풀 문장 뒤에 덧붙인다(1절) |
| 소수 블록 | 마스크 전 분할별 최소 채점 블록 < 5 → '소수 블록', 마스크 뒤 < 8 → '마스크 뒤 소수 블록'(표지만, CI 조건은 합집합 8 그대로) |

## 5. 셀 단위 5 km 민감도(R3)

- 대상: 레나·캐나다(모드 x, Wei 풀), 분할 1–5, n {0, 10, 전량}, 추출 5, seed 0·1. R1(catboost_lo)과 D0(catboost_lo)를 LG 와 같은 행렬로 다시 적합하고 P0·P1·P1@wei·B:wei_raw 를 해석식으로 낸다. 같은 예측을 전체 저장소(`<대상>|x`)와 Wei 학습 지점 5 km 안 채점 셀을 뺀 저장소(`<대상>~c5|x`)에 함께 쓴다(h54 UnitBase 의 추가 저장소). 적합 280건(`--count-only`).
- 셀 거리는 마스크 단계의 `xg_leak_cells_v1.csv` 를 우선 쓴다(Rescale 묶음에 원 학습 지점 표를 보내지 않아도 된다).
- 조각 runs.csv 의 rmse·bias 열은 비운다(출력 제한). 블록 SSE 만 남긴다.

## 6. 로컬 실행과 관문 결과(2026-10-04)

- 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --as=10737418240 taskset -c 64-67 python3 -m pytest -q tests/test_x_xg_xh.py` → 21개 통과(12 s).
- `--count-only`: 재채점 조각 109개(LG 25, LGD 10, WF6 74), 셀 단위 단위 10개, 적합 280건. 레나 분할 2 의 채점 블록 5(기존 기록과 같다).
- 스모크(레나 분할 1, 재표집 200, 셀 단위 적합 4건): 완료, 최대 RSS 360 MB. 봉인 표는 `data/processed/xbatch/XG_product_comparison/smoke/sealed/` 에 있고 열지 않았다.
- 관문(`--gate-only`, 전체 분할): g1 1,300키 실패 0, g1b 650키 실패 0, g2 21키 실패 0, g3 은 추출 전이라 기록 없음. 표 `data/processed/xbatch/XG_product_comparison/xg_gate.csv`(통과 여부와 최대 차만).

## 7. 메모리 상한

- `xbatch_core.limit_memory(10)`(RLIMIT_AS 10 GiB)를 건 프로세스에서 CatBoost 기본 용량(600회, 깊이 6, XH)이 fit 안의 get_feature_importance 에서 멈췄다(가상 메모리 약 9.2 GiB, 상주 0.3 GB). 상한을 빼면 같은 스모크가 1분 안에 끝났다. `systemd-run --user` 는 이 서버에서 실패한다. 그래서 두 모듈은 RLIMIT_AS 대신 상주 메모리(VmRSS)를 2초마다 보고 10 GB 를 넘으면 끝내는 감시 스레드(`rss_watchdog`)를 쓴다. 공용 골격의 limit_memory 는 다른 모듈에도 같은 위험이 있어 열린 문제로 넘긴다.

## 8. 남은 일

1. 로컬 1c: `--extract`(T0 확인, md5 대조, 단위 기록), `--mask`. Wei 의 단위는 추출 때 처음 확인된다.
2. Aalto 기준기 GeoTIFF 는 zip 안에서 deflate 압축(1.26 GB)이라 /vsizip/ 창 읽기가 느릴 수 있다. 느리면 `data/raw/aalto2018_alt/` 아래에 풀어 읽는다(계획 1절 디스크 규칙).
3. Rescale R3 묶음에 `xg_product_values_v1.csv`, `xg_leak_cells_v1.csv`, `xg_meta.json` 을 넣어야 한다(`xbatch_core.PAYLOAD_INPUTS` 에 없다. `payload_manifest(extra=...)` 로 더한다).

## 9. 검토 반영(2026-10-04 적대 검토, 결함 14건 가운데 XG 해당 12건)

- 적용 범위: 검토 결함 목록의 1–9, 12–14 와 결함 10 의 XG 부분(관문 요약의 봉인 메타 기록). 등록 가설·대비·마스크·CI 방법·판정 규칙·Holm 가족(m 6)은 바꾸지 않았다. 계획 문구의 문자 그대로 해석과 다른 선택은 '부분 반영·거절' 항에 적었다.
- 시험 표지 (u)–(y)는 `tests/test_x_xg_xh.py` 머리말의 '검토 반영' 목록이다.

| 결함 | 내용 | 처리 | 시험 |
|---|---|---|---|
| 1 | `--allow-local` 이 상주 메모리 감시를 끄고, 집계·관문 경로가 가용 메모리를 보지 않았다 | 감시는 환경 변수(WF_RESCALE·LG_RESCALE)로만 정한다(`on_rescale`, 허용 표지와 무관). 추출·관문·셀 단위·집계 경로 모두 `wait_memory`(30 GB 하한, 최대 3,600 s 대기, 60 s 간격)를 거친다. 묶음 점검·세기는 기다리지 않는다 | (y) |
| 2 | 작업 R3 에서 제품 값 표가 없으면 제품 키 없이 status ok 조각이 생겼다 | 본 실행(dry·스모크 아님)은 `require_cell_inputs`(Wei 값)와 `cell_dist(strict=True)`(표 없음, 대상·제품 행 없음, 채점 셀 거리 비유한)에서 적합 전에 멈춘다. `preflight_cell` 이 execute 전에 한 번 더 확인한다. 조각 설정 `cell_cfg` 에 제품 값 표·셀 거리 표의 sha256(`input_shas`)을 넣어 다른 표로 만든 조각을 `--resume` 이 다시 쓰지 않는다 | (x), (j) |
| 3 | 묶음 목록에 R3 입력(제품 값 표, 셀 거리 표, xg_meta.json)이 없었다 | 모듈 `PAYLOAD_EXTRA` 와 `--payload-check`(`payload_manifest(extra=PAYLOAD_EXTRA)` 의 missing 을 돌려주고 묶음 정보 `xg_payload_manifest.json` 을 쓴다. 없는 입력이 있으면 종료 코드 1). 공용 골격 `xbatch_core.PAYLOAD_OPTIONAL` 과 `PAYLOAD_REQUIRES` 에도 같은 세 경로를 더했다(추가만, 다른 줄은 고치지 않았다. `tests/test_xbatch_core.py` 25개 통과) | (y) |
| 4 | ρ·s 대체 채점 셀의 비율이 판정 행에 없었다 | 모든 대비 행에 n_B_filled·n_B_scored·fill_frac_B(사용 분할 합계)·fill_frac_B_max_split·fill_note('제품 결측 대체 k%', 0.05 % 미만은 '0.1% 미만'). 판정 표 xg_tests 에 두 마스크 최댓값 fill_frac_B 와 표지. 서술 민감도 xg_fill_sensitivity(Holm 없음): 대체 셀이 있는 블록을 두 팔에서 같이 뺀 뒤 다시 계산하고 두 마스크 결합 갈래(보정 전 4분 판정)를 쓴다 | (w) |
| 5 | 판정 표에 1절 표지(한계 의존, 분할 독립 가정 의존, HK 조건, 부분 표지)가 없고 열세 지역은 5 km 행만 보았다 | `mask_cols` 가 마스크마다 MEAN 행의 verdict4_d10·verdict4_rel·limit_dependence·ci_dependence·pool(pool_label)·few_block_regions·mask_few_blocks·fill_note·delta·rmse_p0_postmask·n_ci_regions·region_general 을 `<열>_<마스크>` 로 옮긴다. region_general 은 두 마스크 모두 참일 때만 참이고 그때 tags 에 '지역 일반 문장 허용(HK CI 가 0 을 제외)'을 넣는다. `combined_notes` 는 '부분(지역 k/m)'·Holm 문구 표지·'마스크 의존'·'한계 의존'·'분할 독립 가정 의존'·효과 크기·결측 대체·판정 불가 사유의 합집합, `worse_union` 은 두 마스크 열세 지역의 합집합(두 마스크 모두 열세면 5 km 값과 '5·25 km 마스크' 표기) | (u) |
| 6 | XG-4 풀에 러시아 W·E 가 별표 없이 들어갔다 | CALM 학습 제품(train = wei, calm_gtnp)의 XG-4 풀에서 러시아 W·E 를 빼고(`MEAN[Lena|x,Canada|x]`), 그 지역 행은 ref_star '학습 지점 포함*'·역할 '참고값(판정 없음)'으로 판정 열을 비워 싣는다(XG-7). L0 제품(CCI)은 주 4지역 그대로 | (v) |
| 7 | Aalto(L2) 행에 '누설 점검 불가' 표지가 없었다 | `leak_cols`: leak_grade, leak_check('누설 점검 불가(대리 마스크)'), si_only, 역할에 'SI 전용'. xg_descriptive·xg_key_scores 의 모든 행과 봉인 메타(leak_check) | (v) |
| 8 | XG-4r 에 문장 규칙이 없었다 | 두 마스크 결합 갈래(보정 전 4분 판정, Holm 없음, holm 열 '없음(서술)')와 TEMPLATES['XG-4r'] 의 다섯 갈래 문장, 공통 표지(제품 ALT 정의, 마스크 전후 P0 RMSE, 등록 이탈)를 MEAN 행에 쓴다. 열세이고 학습 자료가 겹치는 제품이면 '(학습 자료 중복 가능)' | (v) |
| 9 | GDAL 캐시·임시 폴더가 기본값(RAM 의 5 %, /tmp)이었다 | `gdal_env`: rasterio 를 처음 부르기 전에 GDAL_CACHEMAX 512(MB), CPL_TMPDIR `data/raw/tmp_gdal`(폴더 생성). xg_meta.json 의 gdal 에 기록 | (x) |
| 10(XG 부분) | 관문 요약이 봉인 메타에 없었다 | `gate_status` 가 xg_gate_meta.json 을 읽어(제품 값·마스크·추출 기록·셀 단위 조각보다 오래되면 stale) xg_summary_meta.json 의 gates 에 넣는다. 화면에는 통과 여부와 오래됨만 쓴다 | 집계 경로 |
| 12 | 1,000회 재표집으로 본 봉인 폴더에 썼다 | `check_nboot`: 스모크가 아니고 a.nboot < 10,000 이면 집계 전에 멈춘다(주 함수에서는 무거운 재채점 전에, summarize 안에서 다시) | (x), (y) |
| 13 | `require_memory` 가 기다리지 않았다 | `wait_memory` 가 `xbatch_core.require_memory(30, wait_s=3600, poll_s=60)` 을 부른다. Rescale(환경 변수)에서는 기다리지 않는다 | (y) |
| 14 | 파일 위치 이탈 기록과 쓰이지 않는 `--gate-level` | 1절의 위치 이탈 기록을 유지하고 묶음 정보 `module_inputs.file_location` 에도 적는다. XG 의 `--gate-level` 인자는 뺐다(g2 는 1절 로컬과 Rescale 사이의 상대 허용 1e-9 고정) | (x), (y) |

- 그 밖의 변경: `Rescore.tm` 은 조각이 없는 대상을 마스크 표를 보기 전에 건너뛴다(마스크 표가 그 대상을 담지 않아도 멈추지 않는다. 실제 실행에서는 `--mask` 가 모든 대상의 표를 만든다).
- 부분 반영·거절: (1) 결함 2 의 '비 dry 실행은 모두 중단'에서 스모크를 제외했다. 스모크는 추출 전에도 돌려야 하는 로컬 점검이고 smoke/ 아래에만 쓰며 유료 작업이 아니다. 스모크 조각은 product_available=False 를 기록한다. (2) 결함 4 의 '대체 셀을 뺀 민감도'는 LG 조각에 셀 단위 예측이 없어 셀 대신 그 셀이 든 블록을 두 팔에서 같이 뺀다(FILL_SENS_TXT 에 명시, 서술). B:{p}_raw 의 ρ·s 대체 규칙 자체는 3절의 결정(채점 셀 집합을 방법 사이에 같게 두는 LGX 규칙)을 유지했고, 대신 모든 행에 비율을 싣는다. (3) 결함 6 의 XG-4r 은 등록 풀(레나·캐나다, 주 4지역)을 그대로 두었다. 계획 2.7 표가 XG-4r 의 풀을 '같음'으로 등록했기 때문이다.
- 8절 남은 일 3(묶음 입력)은 결함 3 처리로 해소됐다. 남은 일 1·2 는 그대로다.

### 9.1 이번 확인(2026-10-04)

- 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q -p no:cacheprovider tests/test_x_xg_xh.py` → 27개 통과(XG (a)–(j), (t), (u)–(y); XH (k)–(s), (u)). `tests/test_xbatch_core.py` 25개 통과.
- `--count-only`: 재채점 조각 109(LG 25, LGD 10, WF6 74), 셀 단위 단위 10, 적합 280건(6절과 같다).
- `--payload-check`: 필수 입력 3개 가운데 없음 3개(추출·마스크 전이라 예상된 상태). 묶음 정보 `data/processed/xbatch/XG_product_comparison/xg_payload_manifest.json`(파일 82개의 sha256).
- 동결 모듈 sha256 앞 16자 불변: h40 7098f59dabe2b73f, h42 22e215e435e157be, h54 cf9a1f6d0929c892, h41 492373e4d37b5ea4.
- 제품 값 추출·누설 마스크 표·스모크·집계는 이번에 돌리지 않았다(로컬 1c·2단계).

### 9.2 다음 단계 명령(ROOT 에서, 순서대로)

로컬 1c(계획 5절: XG 의 CCI v4 행·누설 마스크 표. 라벨 값 미사용, 적합 없음)
1. `free -g` 로 가용 메모리 30 GB 이상 확인(모자라면 wait_memory 가 최대 1시간 기다린다).
2. 누설 마스크 표(좌표만): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --mask`
3. 관문 기록 갱신(마스크 표가 관문 기록보다 새로워 stale 이 된다): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --gate-only`
4. CCI v4 행(cci4 는 자료의 cci_alt 열이라 추출이 없다. 재표집 10,000회, 봉인 폴더): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --summarize-only --allow-local --products cci4 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'`
   - 산출 `data/processed/xbatch/XG_product_comparison/sealed/xg_descriptive.csv, xg_key_scores.csv, xg_product_key_notes.csv, xg_summary_meta.json`(cci4 행만. 2단계 뒤 전체 집계가 같은 이름으로 다시 쓴다).

2단계(계획 5절 '병렬', 제품 추출. T0 커밋이 HEAD 의 조상이어야 한다)
5. `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --extract --threads 4`(Wei 단위가 값 범위로 cm 가 아니면 `--units wei=cm`)
6. `--gate-only`(g3 md5 포함) → `--payload-check`(없음 0개여야 R3 제출) → 전체 집계 `--summarize-only --allow-local`(위 4 와 같은 필터).
