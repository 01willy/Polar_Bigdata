# x_new_regions 구현 기록(계획 2.6 XF_new_regions)

- 대상: `scripts/3_deep_learning/x_new_regions.py`, `tests/test_x_xf.py`
- 근거 문서: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = git 2678100) 2.6절, 1절, 0.3절, 4절, 5절, 부록 A. `docs/research/2026-10-04/open_regions.md`, `harness_implementation_plan.md` 4.6, `impl_notes/xbatch_core.md`, `impl_notes/xf_downloads.md`
- 작성: 2026-10-04. 계획은 고치지 않았다. 동결 모듈 4개는 고치지 않았고 xbatch_core(XB)를 거쳐서만 쓴다. 기존 스크립트(build_ext_cells_v1, ext_cells_covariates_v1, build_fidelity_base_v4, lgd_eligibility_v1)도 고치지 않고 불러 쓴다.
- 이 기록을 쓰면서 계산한 것: 합성 자료 단위 시험뿐이다. 실제 새 자료원의 점 자료는 아직 없다(파서가 없다). 새 라벨의 평균·분산·계수 비, RMSE, Δ, 판정은 계산하지도 출력하지도 않았다.

## 1. 자료 조립(계획 2.6 '구현·시험')

| 항목 | 계획 문구 | 처리 | 근거 |
|---|---|---|---|
| 스크립트 나눔 | 자료원별 parse_ext_<id>.py, build_ext_cells_v2.py, build_fidelity_base_v5.py, lgd_eligibility_v2.py, x_new_regions.py, x_new_regions_pool.py | 작업 지시대로 셀 조립, v5 표, 적격 세기, 실행 표, 집계, XC-F3 목록을 x_new_regions.py 하나의 단계(--stage, --count-only, --summarize-only, --xc-f3)로 두었다. 파서(parse_ext_<id>.py)는 만들지 않았다 | 파서는 원파일의 라벨 열 결정(xf_downloads.md 4절 2항)이 필요하고, 이 작업 범위가 아니다 |
| 새 점 자료 위치 | 정하지 않음 | 두 곳을 읽는다. (1) `data/processed/xbatch/XF_new_regions/ext_labels/<src_id>_points.csv`(권장, 1절 산출 경로 안), (2) `data/processed/ext_labels/` 의 점 자료 가운데 v4 조립 목록(`ext_cells_v1_meta.json` 의 inputs.points, 20개)에 없는 파일. 같은 이름이 두 곳에 있으면 멈춘다 | (2)는 기존 규약(open_regions.md 8절)을 따른 파서를 받기 위해서다 |
| 셀 규칙 | LGD 6B.3·6B.4 그대로 | build_ext_cells_v1.main 을 고치지 않고 부른다. 바꾸는 것은 전역 변수 EXT(입력·산출 폴더)뿐이다. 그 함수의 화면 출력(라벨 평균 포함)은 버리고 메타에서 라벨 통계 항목(키가 `(^|_)alt(_|$)`)과 절단·값 경계 요약을 지운다 | 계획 0.3, 2.6 열람 상태('새 라벨의 평균·분산·계수 비를 출력하지 않는다') |
| v1 셀 규칙의 결함 | 해당 없음 | build_ext_cells_v1.main 은 우측 절단 행이 하나도 없으면 395행(절단 고아 위치)에서 KeyError 로 멈춘다(빈 목록 색인이 열 선택이 된다). 셀을 만들 수 없는 절단 행 하나(적도 태평양, 2000년)를 넣어 그 경로를 지나게 하고 메타의 수에서 뺀다. 재현 관문의 v1 재조립에는 넣지 않는다 | 함수를 고치면 v4 조립 기록의 스크립트 해시가 달라진다 |
| v4 행 불변 | 'v5 의 v3·v4 행이 바이트 단위로 같다' | v5 = v4 원문 줄 + 새 행(build_fidelity_base_v4.append_rows_text). 그래서 새 위치가 v4 셀과 같은 라벨 집합·셀 색인이면 v4 셀 값이 바뀌어야 하므로 v5 에 넣지 않는다(merge_v4). v4 새 직접 라벨 셀과 체비쇼프 0.01° 이내이면 v3 중복 규칙을 v4 로 넓혀 뺀다(dup_v4) | v4 가 v3 위에 쌓인 방식과 같다 |
| 새 행 규약 | 정하지 않음 | loc_id = 30000 + 일련번호(macro, ky, kx 순, v4 새 행 최댓값 20515 다음 자리를 넉넉히 비운다). 그 밖은 build_fidelity_base_v4 와 같다(source_id F4_ext_direct, region = REGION_OF 또는 macro 이름, spatial_support_m 1000, sigma_prior 3–40 자름, 토양 도일 0 이하 결측) | |
| 독립성 100 km | '다른 macro 의 v3 직접 라벨 셀에서 100 km 넘게 떨어짐' | 문구 그대로 v3 F4_direct 까지의 거리(build_ext_cells_v1 의 indep_ok)를 쓴다. v4 새 지역(Tibet_LGD 등) 셀까지 거리는 판정에 쓰지 않는다 | |
| 새 macro 판정 | '이번 조사 뒤 처음 확보한' macro 지역 | 점 자료 macro 가 기존 표(v4: v3 행과 LGD 새 행)의 macro 에 없으면 새 macro 후보다. 스키마 어휘의 'other' 도 이름 그대로 후보가 되므로 새 지역 파서는 macro 이름을 붙여야 한다 | |
| 역할 | 2.6 '단위와 역할' 표 | NAtlantic = 민감도(natl_v5), Alaska = 하위 과제(XC F2 서술, 실행 표 없음, XD 개발 과제 아님), 기존 macro(Tibet, Canada, Russia_W·E·C, Lena 등) = 셀 보강(L40 형식 민감도), 그 밖 = 새 macro | |
| 비상업 약관 | 'Petrone 은 CC BY 로 공개할 v5 파생 표에 넣지 않는다' | 약관 문자열이 NC·Non-commercial 이면 v5 에 넣지 않는다(lic_nc). 내부 판을 따로 두지 않았다 | 가장 문자 그대로의 읽기 |
| 약관 미확인 | '약관 확인분만 주 판정' | v5 에는 넣고(v4 와 같다) 실행 표의 주 판에서 뺀다. 새 macro 표와 셀 보강 표는 약관 확인분 대상 셀만 고른다 | |
| 공변량 | LGD 셀 파이프라인 | ext_cells_covariates_v1 의 단계 함수(dem, e5, cci, sg, elev)를 고치지 않고 부른다. 전역 변수 PARTS(산출), DEM, SG_RAW 를 덮개 폴더(`data/raw/xf_dem_overlay`, `data/raw/xf_soilgrids_overlay`: 기존 파일은 기호 연결, SoilGrids v4 창 메타는 복사본)로 바꿔 기존 원자료 폴더와 `windows_wcs_meta_v4.json` 을 고치지 않는다. 새 SoilGrids 창 이름이 기존 v4 창과 겹치지 않게 macro 앞에 'xf5_' 를 붙인다(창 이름에만 쓰인다). --fetch 가 없으면 내려받지 않고 빠진 타일·창은 결측이다 | 작업 지시의 쓰기 경로 규칙(data/raw/<새 자료원>) |
| 실행 표 | LGD 와 같은 방식 | lgd_eligibility_v1.write_run_table_text 를 v5 표에 그대로 쓴다(v3 행 원문 줄, 고른 새 행만 source_id 를 F4_direct 로). 토양 표는 v5 토양 도일의 기호 연결이다 | h51 과 같다 |
| NAtlantic v5 | 약관 확인분 판에 공개 자료 2건을 더한 민감도, 약관 회신 뒤 전체 판 포함 | NAtlantic~lic~v5 = LGD NAtlantic~lic 의 v4 새 행 19 + 새 셀. 새 셀이 0 이면 LGD 표와 같으므로 적합하지 않는다(run = False). 전체 판 NAtlantic~v5 는 --natl-lic-confirmed 로 회신을 확인한 자료원 이름을 받을 때만 만든다(회신 기록을 개정 이력에 먼저 적는다) | |
| 연도 정합 도일 | '새 셀의 연도 정합 도일(a2_year_matched_tdd.py 정의)' | 이 모듈에서 만들지 않았다. XB 의 P* 후보가 쓰는 표이므로 XB 의 1c 표(v4 새 셀)를 만드는 함수를 v5 새 행에도 적용해야 한다. 남은 문제로 둔다 | a2 스크립트는 모듈 최상위에서 인자를 읽어 불러 쓸 수 없다 |

## 2. 적격 세기(라벨 값 미사용)

- 분할 1–5: lgd_eligibility_v1.structure 와 같은 규칙(중복, 여집합, 채점 블록 2 미만, 사용 분할의 채점 블록 합집합, 100 km 버퍼 뒤 원천 수)을 다시 구현했다. 라벨 평균과 레짐(150 cm)은 계산하지 않는다. XF 가설은 레짐을 쓰지 않는다. 시험 (d)가 같은 입력에서 structure 의 모든 키(라벨 평균 제외)가 같음을 확인한다.
- 분할 201–210(XC-F3): XB.split_plan_new(앞선 분할 = 1–5, 채점 블록 하한 2). F3 대상은 PE1·PE2·알래스카가 아니므로 '채점 블록 5 미만 제외'(1절, PE1·PE2·알래스카 대상의 구조 규칙)를 적용하지 않았다(문자 그대로). XC 모듈이 F3 에 5 미만 규칙을 적용하기로 하면 이 세기와 맞춰야 한다.
- 적격 = 유효 분할 ≥ 1 이고 채점 블록 합집합 ≥ 8(LGD 6B.4). 부적격 새 지역은 LGD 처럼 점 추정으로 돌고 XF-1–XF-3 가족에 들지 않는다.
- 라벨은 셀 유무(채점 마스크, 원천 유효 행)에만 쓴다. 적합 수는 h51 과 같이 √TDD 를 라벨 자리에 넣은 가짜 문맥의 dry 실행(h40.run_ctx, h54.TUnit)이다. h40.Data 적재 때 z = log(y/s) 가 메모리에서 계산되지만 집계·출력하지 않는다(h51 과 같다).
- 시험 (e)는 대상 라벨 값을 1.7배로 바꿔 화면 출력과 적격·적합 수 표(표 해시 열 제외)가 같음을 확인한다.

## 3. 적합 단위(작업 R4)

- (spec, 분할) 하나에서 h40.build_ctx 문맥을 한 번 만들고 (1) LG CPU 방법 축(h51 의 LGD 범위: P0–P3, D0, D1, R0–R3, V1, V1r, catboost_lo, α 1, n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5, seed 2, λ 0.25·0.5·1.0)과 (2) WF4 팔(h54.TUnit: P0, P1, P2, R1, R2, D1, 규칙 W, n {10, 40, 160, 전량}, 추출 5, seed 2, 진단값)을 적합한다. 계획 2.6 의 비용 근거('LG CPU 방법 축 890–1,265건 + WF4·XB·XC 팔')를 따랐다. XB·XC 팔은 각 모듈이 register_xf_aliases 로 XF 실행 표를 읽어 돈다.
- 조각 이름: `<tag>__cpu__<별칭>__x__s<분할>__{lg, wf4}`(tag 기본 xf). 저장소 이름은 LG '<별칭>|x', WF4 '<별칭>~wf4|x' 라 한 tag 를 함께 읽어도 키가 섞이지 않는다. 설정 해시의 공통 부분은 모든 조각에서 같다(XB.check_cfg).
- 추출 seed 는 표 안의 대상 이름(예 NAtlantic)으로 정한다(LG·LGD·WF4 와 같다). NAtlantic v5 의 추출 seed 는 LGD NAtlantic~lic 와 같고 |A| 가 달라 추출은 다르다.
- unit.json 에는 h40·h54 와 같이 build_ctx 메타(E_own, y_mean 등)가 들어간다. 본 실행은 XF 마감과 적격 기록 뒤라 열람 규칙에 걸리지 않으며 화면에는 쓰지 않는다.

## 4. 대비와 판정(봉인 집계)

| 가설 | 키(h40 저장소 곡선 키) | CI |
|---|---|---|
| XF-1 | D0(catboost_lo, n 0, λ 키 1.0) − P0 | 같은 라벨 집합 대비(두 방법 모두 대상 라벨 미사용): 분할 안 채점 블록 재표집(XB.region_stat_same), 두 가중, 보조 공통 재표집 |
| XF-2 | R1(λ 0.25, n 10) − P1(n 10) | 같은 추출·seed 의 라벨(같은 라벨 집합) |
| XF-3 | R1(λ 0.25, 전량) − P0 | 같은 라벨 집합(P0 는 라벨 미사용) |
| XF-4 | 규칙 W − R1(λ 0.25), n 10·40·160(WF4 저장소), 진단값 |b10| 평균 | 같은 라벨 집합, 서술 |

- 4분 판정은 XB.pool 의 주 δ 0.5 와 보조 δ 1.0·δ_rel, '한계 의존', '분할 독립 가정 의존', '소수 블록' 열을 그대로 쓴다. 재표집은 10,000회(로컬 집계는 1,000회 상한).
- '적격 새 독립 지역마다, 2곳 이상이면 층화 평균도': 지역 행이 가설 판정이고 층화 평균 행은 보조로 읽었다. Holm 가족은 지역 행의 양측 p 만이다(m = 3 × 적격 지역 수, 행 없음 p 1, 보조 열). 1절 '다중성' 행에 따라 판정 문구는 보정 전 CI 의 4분 판정이 정하고 Holm 보정 p ≥ 0.05 인 우세·열세 문장에는 '보정 전 유의'를 붙인다. 층화 평균 문장에는 Holm p 가 없으므로 이 표지를 붙이지 않는다.
- 해석 문장: 계획은 XF-1 의 예문('추가 독립 지역 k곳에서 라벨 0개 직접 ML 은 원천 계수 Stefan 보다 오차가 a cm 컸다(독립 지역 확인).')만 준다. XF-2·XF-3 와 다른 갈래는 1절 다섯 갈래 문구와 같은 구조로 만들었다(_templates). 적격 0곳 문장, NAtlantic v5 문장(점 추정이면 셀 가중 점 추정 차와 '점 추정'), '하위 과제만 추가' 표지는 2.6 사전 고정 문장 그대로다.
- 봉인: 계획 0.3 은 R4 의 열람 순서를 따로 적지 않았다. 출력 제한(0.3)을 지키려고 집계는 `data/processed/xbatch/XF_new_regions/sealed/` 에만 쓰고 화면에는 행 수와 해시만 쓴다. 재현 관문 기록(xf_repro_gate.json)이 통과하지 않았으면 집계를 거부한다(0.3 열람 순서 5. --allow-no-gate 는 그 사실을 메타에 남긴다).

## 5. 부록 XC-F3 목록

- F3 = 종류가 새 macro 지역이고 분할 1–5 에서 적격('XF 에서 적격 판정을 받은')이며 분할 201–210 에서 다시 센 적격('적격은 분할 201–210 에서 다시 센다')도 만족하고 약관 확인분인 지역. NAtlantic v5 와 셀 보강, 하위 과제는 넣지 않는다.
- 0곳이면 상태는 '시험하지 못함(적격 0곳, Holm 가족에 p 1)'이고 2.3 XC-F3 표의 F3 0곳 문장을 붙인다. XF 마감(T0 + 7일 = 2026-10-11 14:22:39 +0900) 전에 만들면 '잠정'으로 적는다.
- 계획 문서는 고치지 않는다. 부록 문구는 `xc_f3_list.json` 의 appendix_text 에 두고, 개정 이력으로 옮기는 일은 사용자 확인 뒤에 한다(0.3 커밋 규칙). XC 모듈은 xc_f3_list.csv 의 별칭과 xc_keep(실행 분할)을 읽고 register_xf_aliases 로 실행 표를 연다.

## 6. 재현 관문(--repro-gate)

1. v5 의 v4 줄과 토양 도일 v5 의 v4 줄이 바이트 단위로 같다.
2. LGD 실행 표 네 개가 바뀌지 않았다(NAtlantic_lic·Canada_expanded_lic 는 lgd_eligibility_lic.csv 의 table_sha256 과 대조, 나머지는 존재).
3. Tibet 와 NAtlantic_lic 실행 표를 v5 에서 새 경로로 다시 만들면 기존 실행 표와 바이트가 같다(계획 'Tibet_LGD 를 새 파이프라인으로 다시 만든 실행 표').
4. 다시 만든 두 표의 적격 세기(대상 셀 수, 유효 분할, 합집합, 최소 채점 블록, 원천 수, 적격)가 LGD 적격 표와 같다.
5. NAtlantic~lic 의 분할 201–210 구조가 부록 A 의 행(유효 4, 앞선 분할 1, 새 범위 중복 5, |A| 5–15, 채점 블록 4–5, 합집합 6)과 같다.
6. (--gate-b1) v4 조립의 점 자료 20개로 셀을 다시 만들면 ext_cells_v1.csv 와 ext_cells_v1_locs.csv 가 바이트 단위로 같다(빌드 스크립트 해시는 v1 메타와 같다).

## 7. 시험 실행 기록

- 단위 시험: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /usr/bin/time -v nice -n 10 prlimit --as=34359738368 taskset -c 40-43 python3 -m pytest -q -p no:cacheprovider tests/test_x_xf.py`(출력은 0.3 의 grep 필터를 거쳤다). 14개 통과, 1개(HEAVY 재현 관문, XF_HEAVY=1) 건너뜀, 22 s, 최대 RSS 317 MB. 누설 시험 (f)가 가장 길다(약 16 s).
- 메모리 상한의 방식: 처음에는 xbatch_core 시험 기록과 같이 `prlimit --as=10737418240`(주소 공간 10 GB)으로 돌렸는데, 시험 전체를 한 프로세스에서 돌리면 누설 시험에서 멈췄다(RSS 246 MB, VmSize 9.5 GB, VmPeak 10.0 GB, 시스템 시간이 대부분). CatBoost·스레드·적재 모듈의 가상 예약이 주소 공간 상한에 닿아 할당 실패가 커널 안에서 되풀이된 것으로 본다. `systemd-run --user --scope -p MemoryMax=` 는 이 서버에서 실패했다. 그래서 모듈은 1절 '작업 하나 10 GB 이하'를 RSS 감시 스레드(rss_watchdog, 기본 10 GB, 넘으면 종료 코드 3)로 강제하고 주소 공간 상한은 --as-cap-gb 로만 건다. 시험은 주소 공간 32 GB 안전망과 /usr/bin/time 의 최대 RSS 기록으로 돌렸다. XB.limit_memory(주소 공간 10 GB)를 쓰는 다른 모듈도 같은 멈춤을 겪을 수 있다(8절).
- 실제 자료 점검(산출 뿌리는 세션 임시 폴더, data/processed/xbatch 에는 쓰지 않았다): 새 점 자료가 아직 없어 v5 = v4(새 행 0)이다.
  - `--stage cells,cov,v5,tables`: 점 자료 0, v4 행 18,088, v4 줄 바이트 같음, spec 은 NAtlantic~lic~v5 하나(v4 새 행 19, 새 셀 0, 적합 안 함).
  - `--count-only`: NAtlantic~lic~v5 대상 셀 22(v3 3, v4 19), 블록 8, 채점 셀 19, 원천 17,462, 유효 분할 5, 채점 블록 합집합 6, 최소 채점 블록 2, 부적격(점 추정). 분할 201–210 유효 4, 합집합 6. LGD 적격 표(lgd_eligibility_lic.csv)와 같다. 비용 참고: 단위 5, 적합 866(LG 665, WF4 201), 추정 0.26 CPU-h(4스레드), 실측 비 0.62 적용 0.16.
  - `--repro-gate --gate-b1`: 점검 14개 모두 통과(v5·토양 v4 줄 바이트, LGD 실행 표 4개 불변, Tibet·NAtlantic_lic 실행 표 재생성 바이트, 두 표의 적격 세기, 부록 A NAtlantic~lic 행, v1 셀 재조립 바이트 2파일, v1 점 자료 20개 불변). 약 30 s.
  - `--xc-f3`: F3 0곳, 잠정(마감 전), 'XC-F3 시험하지 못함'.
  - `--smoke`(합성 대상 라벨, NAtlantic~lic~v5 분할 1): 적합 18(LG 15, WF4 3), 21 s, 최대 RSS 444 MB, 봉인 파일 4개. 화면에는 행 수와 해시만 나왔다.

## 8. 남은 문제

1. 새 자료원 파서(parse_ext_swe_stordalen_crill.py, parse_ext_sjm_adventdalen_wendt2023.py 등)가 없다. T0 + 24 h 안의 NAtlantic v5 --count-only 는 파서가 점 자료를 쓴 뒤에 한다. 파서는 라벨 열 결정(Adventdalen 의 세 후보 열, Stordalen 의 series_max)을 값 계산 전에 기록해야 한다.
2. 새 행의 연도 정합 도일(XB 의 P*)은 이 모듈이 만들지 않는다.
3. 작업 R4 묶음에는 XF 실행 표(`data/processed/xbatch/XF_new_regions/run_tables/**`), v5 토양 도일, xf_specs.json, xf_eligibility.csv 를 더해야 한다(XB.payload_manifest 의 extra).
4. F3 의 분할 201–210 채점 블록 하한(2 또는 5)은 XC 모듈과 맞춰야 한다.
5. Petrone(비상업)은 v5 에서 뺐다. 내부 분석 판이 필요하면 계획 개정이 필요하다.
6. xbatch_core.limit_memory(RLIMIT_AS 10 GB)는 CatBoost 를 여러 번 적합한 프로세스에서 가상 예약에 걸려 멈출 수 있다(7절). 공용 골격 작성자에게 알릴 사항이다.

## 9. 우선 자료원 파서의 라벨 열 결정(값 계산 전 기록, 2026-10-04 16:38 KST)

이 절은 두 파서(`scripts/1_data_prep/parse_ext_swe_stordalen_crill.py`, `parse_ext_sjm_adventdalen_wendt2023.py`)를 쓰기 전, 원파일의 깊이 값을 읽거나 계산하기 전에 적었다. 근거는 `data/raw/<src_id>/SOURCE.md`(시트 구조, 측정 방법), `open_regions.md` 4.1, LGD 6B.3 이다.

| 자료원 | 라벨 열 | 직접 라벨(qc_flag ok)로 쓰는 지점 | 연 값 규칙 | 근거 |
|---|---|---|---|---|
| swe_stordalen_crill | DATA 시트의 지점별 `A L` 열(음수 표기는 절댓값, cm) | Dry(Palsa 자동 챔버 지점 1, 3, 5)와 Palsa center 만. Mesic(Sphagnum), Wet(Eriophorum), Tussok(9), E, F 는 영구동토 존재가 확인되지 않아 qc_flag `pf_uncertain` 으로 점 자료에 남기고 직접 라벨에서 뺀다 | 8월 1일 이후 값이 있는 방문이 2회 이상이면 그 최댓값(`series_max`), 1회이고 8–9월이면 `record_date`, 8월 1일 이후가 10월 이후뿐이거나 없으면 `direct_dated`(FireALT 파서의 개정 13 규칙과 같다). 그 해 후보 방문에 '>' 값이 하나라도 있으면 right_censored = 1, 값은 숫자 최댓값과 '>' 하한 가운데 큰 값(ru_kwbs_makarieva 파서 (a) 규칙과 같다) | 6B.3 series_max 정의, SOURCE.md 메모 시트('>' = 탐침보다 깊음), open_regions.md 4.1('Sphagnum·Eriophorum 지점은 영구동토가 없거나 탐침 한계일 수 있다') |
| sjm_adventdalen_wendt2023 | Overview 시트의 `Thaw depth in-situ (cm)` | 코어 12지점 모두(값이 있는 지점) | 2023년 9월 단일 방문(`record_date`, direct_eos) | `ALT (cm)` 는 현장 융해 깊이에 InSAR 지표 침하 보정을 더한 값으로 보여 6B.3 '보정값 금지'와 v3·CALM 의 보정하지 않은 탐침 규약에 맞지 않는다. 보정 열(`Surface subsidence correction ...`)도 쓰지 않는다. 측정 방법은 탐침(open_regions.md 4.1, TC 20, 1179, 2026 본문). 2023년은 이례적 고온 해라 notes 에 연도 편차 표지를 둔다(qc_flag 는 ok) |

- 좌표: Stordalen 은 원파일에 좌표가 없어 ICOS SE-Sto 소개의 68°21′N, 19°03′E(분 단위, coord_prec_deg 0.0167)를 모든 지점에 쓴다(지점들은 한 1 km 셀이 된다). Adventdalen 은 UTM X, Y 를 WGS84 UTM 33N(EPSG:32633)으로 가정해 바꾼다([가정], SOURCE.md).
- macro 는 두 자료원 모두 `NAtlantic`(country Sweden, Norway; subunit Scandinavia, Svalbard)이고 x_new_regions 의 6B.4 지리 정의와 같다.
- 파서는 화면과 메타에 라벨 통계(평균, 분산, 최솟값·최댓값, 계수 비)를 쓰지 않는다. 행 수, 지점 수, 연도 범위, label_def·qc_flag 별 행 수만 쓴다(계획 0.3, 2.6 열람 상태).
- 산출: `data/processed/xbatch/XF_new_regions/ext_labels/<src_id>_points.csv`, `<src_id>_meta.json`(1절 산출 경로. 기존 data/processed/ext_labels 에는 쓰지 않는다).

## 10. 검토 반영(2026-10-04 21:15 KST)

적대적 검토가 낸 결함 16건(세션 임시 폴더 `resume/defects_3.json`)의 반영 기록이다. 앞선 세션이 반영을 시작한 뒤 세션 한도로 끊겼고, 이 세션에서 계획 2.6·1·0.3·5절과 LGD 6B.3·6B.4 로 각 항목을 다시 대조한 뒤 남은 항목을 고치고 시험과 세기를 끝냈다. 등록된 가설, 팔, 분할, CI 방법, 판정 규칙, Holm 가족은 바꾸지 않았다. 계획 문서, 기존 자료, 동결 모듈(h40 `7098f59dabe2b73f`, h42 `22e215e435e157be`, h54 `cf9a1f6d0929c892`, h41 `492373e4d37b5ea4`, 2026-10-04 21:0x 재확인)은 고치지 않았다. 1–8절은 16:38 시점의 기록이고, 아래 표가 그 뒤의 상태를 대신한다(1절 '스크립트 나눔'·'새 점 자료 위치'·'연도 정합 도일' 행, 6절 2항, 8절 1–2항은 이 절로 대체된다).

| 번호 | 결함 요지 | 처리 | 시험 |
|---|---|---|---|
| 1 | 6B.4 지리 정의를 쓰지 않고 파서의 macro 문자열을 믿었다 | `geo_macro`(국가·위도·경도: NAtlantic = 그린란드·스발바르·노르웨이·스웨덴·핀란드, Canada, 러시아 90°E·140°E 분할과 레나 상자 제외, 알래스카 좌표, 청장고원 상자, v3 국가 macro), `cell_geo`, `role_of`, `check_point_macros`. 셀 조립 전과 `classify_cells` 에서 점 자료 macro 가 지리 정의와 다르면 멈춘다. 역할은 지리 macro 로 정한다(NAtlantic = natl_v5, Alaska = AL-7 하위 과제, 그 밖 6B.4·v3 macro = augment_L40). 새 macro 후보는 6B.4·v3·v4 어디에도 없는 이름만이다 | b0, b1, b1r(실제 known_macros 로 'Greenland', 'Yukon', 'Svalbard', 'Scandinavia' 모두 거부), b3 |
| 2 | xf_specs.json 의 표 경로가 절대 경로였다 | `table.dir` 는 산출 뿌리 기준 상대 경로이고, `spec_dir(P, 별칭)` 이 실행 시점의 뿌리에서 푼다(run_unit, unit_cfg, smoke, count_only 모두). XC-F3 목록의 run_table 은 ROOT 기준 상대 경로다 | c3, h, i |
| 3 | 마감(T0 + 24 h 세기, T0 + 7일 자료)이 기록만 되고 강제되지 않았다 | `natl_first_record`: 우선 자료원 점 자료가 셀 조립에 들어간 뒤 첫 --count-only 를 배타 생성으로 한 번만 기록(시각, 점 자료·v5·spec·스크립트 sha256, 세기 값). `natl_status`: 기록 없음, 마감 뒤, 첫 세기 부적격, 지금 부적격이면 점 추정(집계기가 쓴다). `points_manifest`: 점 자료 목록을 기록하고 T0 + 7일 뒤 첫 실행에서 동결, 그 뒤 목록 밖·해시가 바뀐 파일은 거부. `run_all`: 마감 전이거나 최종 부록 XC-F3 가 없으면 거부 | g3, k3, m1, m2 |
| 4 | --count-only 재실행이 적격 표를 덮어쓰고, 집계가 동결 해시와 조각 표 해시를 대조하지 않았다 | 최종 부록 XC-F3(마감 뒤 처음 만든 것)가 적격 동결 기록이다(`final_xcf3`, `check_frozen`: 적격 표·spec·v5 sha256 대조). 동결 뒤 `count_only` 와 cells·v5·tdd·tables 단계는 거부한다. `select_specs`, `register_xf_aliases`, `summarize` 가 대조한다. `unit_table_check` 가 조각 unit.json 의 table_sha256 을 spec 표 해시와 대조하고, 집계는 관문 기록의 v5 해시가 지금 v5 와 다르면 멈춘다 | n, i |
| 5 | v5 새 행의 연도 정합 도일이 없고, XB 가 XF 별칭을 모른다 | `--stage tdd`(`write_tdd_v5`): `x_multisource_stacking.a2_tdd_cells` 를 고치지 않고 불러 v5 새 행의 tdd_matched 를 만들고, 같은 코드로 v3 셀 400개를 다시 계산해 `lgx_tdd_matched_v1` 과 대조한다(실자료 최대 차 2.3e-13 °C·day, 표지 일치). `--xb -- <XB 인자>`(`xb_prepare`, `xb_main`): 이 프로세스 안에서 `register_xf_aliases` 로 XB.RUN_TABLES 에 별칭을 더하고 XB 의 `tdd_tables` 를 감싸 v5 표를 붙인 뒤 XB.main 을 부른다. 실제 적합은 적격 동결 기록이 있어야 하고 --workers 0 만 받는다 | o1, o2, o3 |
| 6 | 셀 보강 행이 L40 형식이 아니었다(대비 다름, P1 − P0 없음, 기준 비교·L28 없음, macro 수준) | `l40_rows`: L1 지역 조건(D0 − P0, n 0·3·10·40 과 실제 라벨 수 40 이하의 전량 행), 전량 R1(0.25) − P0, n 10 P1 − P0 를 같은 라벨 집합 대비로 계산하고 기준 → 확충판의 변화를 L28(`l28_shift`, `worst_shift`: h52 와 같은 규칙)로 분류한다. 기준 표는 `<대상>~aug~ref`(새 셀 없는 같은 표)를 같은 작업에서 다시 적합한다. 하위 지역 대상: `extended_submap`(lg_subregion_map_v1 + v5 새 블록의 최근접 중심 배정)으로 CA-1 등 블록 대응표를 실행 표 폴더에 두고 h40 의 기본 --subregion-map 이 읽는다 | g4, c, c2 |
| 7 | 우선 자료원 파서 2개가 없었다 | 9절(값 계산 전 라벨 열 결정) 뒤 `parse_ext_swe_stordalen_crill.py`, `parse_ext_sjm_adventdalen_wendt2023.py` 를 쓰고 16:39–16:40 에 실행했다. 산출은 `data/processed/xbatch/XF_new_regions/ext_labels/`(점 자료 87행·12행, 메타는 행 수·지점 수·연도·label_def·qc_flag 별 수만. 라벨 통계 없음을 메타 키로 확인했다). macro 는 NAtlantic 으로 명시했다 | 파서 메타 점검(수동) |
| 8 | 관문이 Tibet·Russia_C 표의 해시와 v4 메타를 대조하지 않았다 | LGD 실행 표 네 개 모두 `lgd_run_manifest.json` 의 specs.<표>.table_sha256 과 대조하고, NAtlantic_lic·Canada_expanded_lic 은 `lgd_eligibility_lic.csv` 와도 대조한다. v4 표·토양 v4 는 `fidelity_base_v4_meta.json` 의 outputs 해시와 대조한다 | l(HEAVY, 이 세션에서는 실행하지 않았다) |
| 9 | --allow-local 이 로컬 자원 규칙을 풀었다 | 완화 기준은 `on_rescale()`(WF_RESCALE=1 또는 LG_RESCALE=1)만이다. 그 밖(--allow-local 포함)은 가용 메모리 30 GB 대기, RSS 10 GB 감시(워커마다 `_worker_init`), 워커당 4스레드, 워커 × 스레드 32 이하(parse_args 가 거부) | k2 |
| 10 | 최대 RSS 미기록, 스모크 환경 미점검 | 세기·관문·조립·tdd 메타에 `max_rss_mb`. 스모크는 `XB.smoke_env_report()` 를 메타에 적고 경고가 있으면 허용 표지 없이 거부한다 | k4 |
| 11 | 새 알래스카 셀의 AL-7 실행 표가 없었다 | 역할 alaska_subtask, v5 region AL-7, spec `AL-7~xf`(run False, f2_only, 대상 = 새 알래스카 셀, v3 알래스카 행은 원천). XC-F3 목록의 f2_only 항목과 `register_xf_aliases` 로 XC 가 연다 | c, h |
| 12 | merge_v4 가 v4 에 들어가지 않은 v1 셀과도 충돌로 처리했다 | merge_v4 = in_v4 = 1 인 셀과 같은 cell_uid 만. v4 에 들어가지 않은 v1 셀과 같은 cell_uid 는 `v1_cell_not_v4` 로 표시만 하고 사유를 xf_note 에 적는다 | b2 |
| 13 | 빠진 지역 행의 판정 불가 행·문장이 없었다 | 적격 지역의 빠진 대비 행은 verdict4 '행 없음', p 1, 사유 '행 없음(Holm 가족에 p 1)' 행으로 넣고 문장을 만든다. 지역 행의 사유는 `region_reason`(사용 분할 수, 채점 블록 합집합, CI 유무) | g1 |
| 14 | 공변량 결측률 시험 없음, run_unit 과 누설 시험 경로가 달랐다 | e2(합성 결측 패턴의 covariate_completeness 와 xf_x25_all). `run_unit` 이 `compute_unit` 을 거쳐 누설 시험 f 와 같은 경로다. 마감(g3, k3, m1, m2), 동결(n), 지리(b0–b3) 시험을 더했다 | e2, f, 위 항목 |
| 15 | XC-F3 기록에 선택 스크립트·v5·spec 해시가 없었다 | `write_xc_f3` 의 기록과 부록 문구에 script_sha256, v5_sha256, specs_sha256, elig_sha256, points_manifest_sha256 | h |
| 16 | 기존 `data/processed/ext_labels` 도 읽었다 | `xf_point_files` 는 `<산출>/XF_new_regions/ext_labels/*_points.csv` 만 읽는다. 파서가 이 폴더에 쓴다(모듈 머리말과 9절) | j, b3 |

제안한 고침과 다르게 처리한 항목(거부한 결함은 없다)

- 5번: 제안은 XB 에 `--xf-aliases` 선택지를 더하는 것이었다. XB(`x_multisource_stacking.py`)는 따로 검토·수정 중인 모듈이라 고치지 않고, XF 쪽 진입점 `--xb` 가 같은 일을 프로세스 안에서 한다(XB.RUN_TABLES 등록, tdd_tables 감쌈). XB 파일은 바뀌지 않는다.
- 6번: 제안은 LGD 조각이나 h51 표의 기준값을 읽는 것이었다. 계획 1절 '플랫폼'(한 대비는 한 작업·한 노드 종류의 조각 안에서 닫고, 비교 기준을 자기 조각에서 다시 적합한다)에 따라 기준 표 `~aug~ref` 를 같은 작업에서 다시 적합한다. LGD 조각은 로컬 실행이라 Rescale 조각과 환경이 다르다.
- 3번(이 세션에서 더한 처리): 전체 판 `NAtlantic~v5` 는 약관 회신 뒤에만 생겨 첫 세기 기록에 있을 수 없다. 앞선 구현은 이를 '첫 세기에 없던 판' 으로 언제나 점 추정으로 두었는데, 2.6 은 전체 판을 민감도 행에 포함하고 마감 규칙의 대상을 Stordalen·Adventdalen 세기(약관 확인분 판)로 적는다. 그래서 전체 판이 기록에 없으면 약관 확인분 판의 첫 세기 기록으로 마감·'첫 세기 부적격' 을 판정하고, 지금 적격 여부는 전체 판 자신의 적격 표로 본다. 약관 확인분 판의 첫 세기가 부적격이면 전체 판도 점 추정이다(2.6 '점 추정으로 확정'). 시험 m2 에 네 경우를 더했다.

시험 실행(이 세션)

- `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /usr/bin/time -v nice -n 10 taskset -c 40-43 python3 -m pytest -q -p no:cacheprovider tests/test_x_xf.py`(출력은 0.3 의 grep 필터를 거쳤다): 59 passed, 1 skipped(HEAVY 재현 관문, XF_HEAVY=1), 25.5 s, 최대 RSS 321 MB. pyflakes 지적 없음(모듈, 시험, 파서 2개).

세기 실행(실자료, 산출 뿌리 `data/processed/xbatch/XF_new_regions`)

- 조립은 앞선 세션이 16:41–16:42 에 `--stage cells,cov,v5,tdd,tables` 로 했다(xf_build_meta.json, 최대 RSS 3,653 MB, 85 s). 점 자료 2개(99행 지리 대조 통과), 셀 13(직접 12, dated 1), in_v5 10, 대상 10. 뺀 셀 3: label_set dated 1(Stordalen), merge_v4 1(Adventdalen 코어 지점이 v4 에 이미 있는 셀과 같은 cell_uid), dup_v4 1(v4 직접 셀과 체비쇼프 0.01° 이내). v5 = v4 18,088행 + 새 10행(v4 줄 바이트 같음), 토양 v5 = 18,096 + 10, 하위 지역 대응표 새 블록 30, 연도 정합 도일 새 행 10. 새 셀 공변량 x25 완전 1.0, CCI 유효 1.0, 토양 도일 1.0, 채점 가능 10.
- 세기: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /usr/bin/time -v nice -n 10 taskset -c 40-43 python3 scripts/3_deep_learning/x_new_regions.py --count-only --threads 2`(2026-10-04 21:10:14 KST 시작, 2.7 s, 최대 RSS 216 MB, 시작 전 가용 메모리 43 GB).
- 결과(라벨 값 미사용, 셀·블록 수만): `NAtlantic~lic~v5` 대상 셀 32(v3 3, v4 19, 새 10), 블록 10(새 블록 2: Stordalen 1, Adventdalen 1. Adventdalen 의 다른 블록 1개는 v4 에 있던 블록), 채점 셀 29, 원천 17,462, 유효 분할 5(1–5 모두), 채점 블록 합집합 8, 최소 채점 블록 2(소수 블록), 적격. 분할 201–210: 유효 8(208 은 채점 블록 1, 210 은 제외), 합집합 8, 적격. 대상 행 점검 통과. 다른 macro 의 v3 직접 라벨 셀까지 최소 거리 630 km. 적합 수 단위 5, 적합 1,448(LG 축과 WF4 팔), 추정 0.48 CPU-h(4스레드), 실측 비 0.62 적용 0.30 CPU-h.
- 판정 방식: 첫 세기 기록 `xf_natl_v5_first_count.json` 이 21:10:17 KST(마감 2026-10-05 14:22:39 KST 전)에 적격(합집합 8)으로 쓰였다. 따라서 NAtlantic v5 민감도 행은 적격 동결 때도 적격이면 4분 판정이고, 그렇지 않으면 점 추정이다. LGD NAtlantic~lic(합집합 6, 점 추정)에 공개 자료 2건의 셀 10개를 더해 8에 도달했다. 2.6 의 '2개 더하면 8 은 근사' 대로 블록 배정이 바뀌어 사용 분할의 합집합이 정확히 8 이므로 이후 셀 조립이 바뀌면(마감 전 자료 추가) 다시 셀 때 적격이 바뀔 수 있다. 첫 세기 기록은 덮어쓰지 않는다.
- 기록 해시 앞 16자: v5 `82d5307d1064cce9`, xf_specs.json `89a2a982d484b613`, 실행 표 `9f227d825295fc0f`, 점 자료 Adventdalen `6dc163415cf4a37c`·Stordalen `d269a4e1e7204b3a`, 스크립트(세기 시점) `0c1612a5f546debc`.
- 라벨 통계는 화면·메타·이 기록 어디에도 쓰지 않았다. 적격 기록 뒤에도 쓰지 않았다.

남은 위험과 다음 단계

1. 재현 관문(`--repro-gate --gate-b1`)은 새 행 10개가 든 v5 로 이 세션에서 돌리지 않았다(허용된 로컬 작업 범위를 시험·세기로 한정했다). R4 전에 돌려 `xf_repro_gate.json` 을 만든다(집계가 요구한다).
2. 부록 XC-F3 최종판(적격 동결)은 XF 자료 마감(2026-10-11 14:22:39 KST) 뒤 `--xc-f3` 로 만든다. 그 전에 자료가 더 오면 cells 단계부터 다시 하고 `--count-only` 를 다시 한다(첫 세기 기록은 그대로다).
3. NAtlantic v5 는 사용 분할의 최소 채점 블록이 2 라 지역 행에 '소수 블록' 이 붙는다(1절). 민감도 행이므로 XF 확인·XC-F3 에는 들어가지 않는다.
4. XC 모듈은 `xc_f3_list.json` 의 run_table(ROOT 상대 경로)과 f2_only 항목(AL-7~xf)을 `register_xf_aliases` 로 열어야 한다. XB 의 XF-4 행은 `--xb` 진입점으로 낸다(8절 3항의 묶음 추가 사항은 그대로다).
5. 8절 3–6항은 그대로 남아 있다.
