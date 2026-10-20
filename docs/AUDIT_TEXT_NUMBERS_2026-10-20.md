# 원고 v13 본문 · 그림 · 매크로 정합 감사 (2026-10-20 12:50 기준)

대상: `paper/manuscript/en_v13` (main.tex, sections/*.tex, si.tex), 국문 쌍둥이 `paper/manuscript/ko_v13`, 매크로 `shared_v10/numbers_v6·v9·v10.tex`, `shared_v11/numbers_v11.tex`, `shared_v13/numbers_v13.tex`, 그림 원천 `outputs/figures/paper/v13/source_data/Fig*.json`, `v13_si/SI_figures.json`, 그림 스크립트 `scripts/4_visualization/paper_v13/`.
감사 중(12:43~12:46) `results.tex`(영·국문), `fig4_labels.py`, `fig5_products.py`, `fig7_maps.py`, `Fig4·Fig5 PDF/JSON`이 갱신되었다. 아래 판정은 12:48 시점의 파일을 기준으로 한다. 파일은 수정하지 않았다.

## 1. 요약

- 본문에 쓰인 매크로 237개는 모두 정의되어 있고(미정의 0, 중복 정의 0), 그림 안에 인쇄된 수치(Fig. 3b·3d·4b·4e·5b·5c·6·7, S3)와 같은 양을 가리키는 매크로는 반올림 범위 안에서 모두 일치한다.
- 영문과 국문은 절마다 매크로 집합이 같고(전체 273개, 차이 0), 그림 패널 글자도 같다(같은 오류를 공유한다).
- 수정이 필요한 항목은 패널 글자 오류 1건(그림 5b→5d), 그림 7 재렌더 필요 1건, 서로 다른 실험의 수치를 같은 주장에 섞어 쓴 1건, 그림 5a와 행 집합이 다른 매크로 1건, 보충 자료의 낡은 교차 참조와 그림 경로 오류이다.

## 2. 발견 사항 표

심각도: 상 = 투고 전 반드시 수정, 중 = 수정 권고, 하 = 선택.

| # | 심각도 | 위치 | 문제 | 수정안 |
|---|---|---|---|---|
| 1 | 상 | en/ko `results.tex` L43 "Year by year … (Fig. 5b)" / "(그림 5b)" | 연도별 RMSE는 v13에서 패널 **d**이다(Fig5 JSON `d`: year-wise; `b`는 관측 대 예측 산점도). 본문이 5b를 가리킨다. | `\ref{fig:products}b` → `d` (영·국문 모두). |
| 2 | 상 | `Fig7_maps.{pdf,json}` (12:11) vs `fig7_maps.py`·캡션(12:43) | 스크립트와 캡션은 d = 지역 평균 편차 시계열, e = 알래스카 연도별 편차 지도로 바뀌었으나, 디스크의 PDF/JSON은 옛 배치(d = 편차 지도, e = 시계열)이다. 본문 L48의 "(Fig. 7e)"(공간 분포), "(Fig. 7d)"(시계열)도 새 배치 기준이다. | `fig7_maps.py` 재실행 → JSON `d`/`e` 키가 캡션과 같은지 확인 → `v13_numbers.py` 재실행(그림 7 `c.stats`는 변함 없음) → 빌드. 스크립트 docstring 7~8행(d/e 설명)도 갱신. |
| 3 | 상 | `en_v13/main.pdf`(12:13), `ko_v13/main.pdf`(12:14), `*_submission.pdf`(12:16) | `results.tex`(12:43)와 Fig4·Fig5 PDF(12:46)보다 오래된 빌드이다. `fig6_product_maps.py`도 12:51에 수정되었으나 Fig6 PDF/JSON은 10-18 빌드이다(캡션 관련 논리는 동일: 글자 a–d/e–h, RMSE·축 밖 값 인쇄). | 항목 1·2 반영 뒤 `fig6_product_maps.py` 재실행, `build.sh`와 `tools/package_paper_v13.sh` 재실행. |
| 4 | 중 | 초록·결론 `\ntlLeDszeroabs`(18.0 cm), 고찰 `\ndrefLeregDzabs`(17.8 cm); 고찰 L13 `\ntlrmseLeDszero`(39.5), 고찰 L8·결과 L12 `\nrmseLeregDz`(39.3) | "레나 델타를 제외하면 직접 ML 오차가 Stefan보다 N cm 크다"는 한 주장에 두 실험의 값이 섞여 있다. 18.0/39.5는 ladder_v6 전이 실험 n=0(D0S−P1n, 그림 4b), 17.8/39.3은 core_v6 지역 홀드아웃(100 km 제외, 그림 3b). | 한 실험으로 통일하거나(초록·결론은 그림 3b 값 `\ndrefLeregDzabs`·`\nrmseLeregDz` 권장), 두 값이 다른 설계임을 한 번 명시. |
| 5 | 중 | en/ko `results.tex` L31 `\ndprodLeweiRomask`(−12.5), `\ndprodLeccifiveRomask`(−5.0) | 그림 5a·보충 표 S6 인용 직후의 "25 km 마스크 짝지은 차이"인데, 원천은 `core_v6/products_eval_mask25.csv`의 "vs wei (year-matched)" 행(레나, n=802: Wei 값이 있는 모든 셀-연도)과 "vs cci5y" 행(n=2,352)이다. 그림 5a의 공통 셀-연도(>25 km, n=449)와 행 집합이 다르다(공통 셀에서 R1−Wei = −13.7, R1−CCI v5 = −7.3). | 문장에 행 집합을 명시: "on all Lena Delta cell-years with a value of that product (n = \nprodLeweimask / \nprodLeccifivemask)" (매크로 `\nprodLeweimask`=802, `\nprodLeccifivemask`=2,352 존재). 또는 공통 셀 기준 차이로 교체. |
| 6 | 중 | `en_v13/si.tex`·`ko_v13/si.tex` L10 `\sifig` | `IfFileExists`는 `v13_si/#1.pdf`를 검사하고 `\includegraphics`는 `v12_si/#1.pdf`를 넣는다. 보충 PDF에 v12 그림이 들어간다(10개 모두 md5가 다름, 수치는 "unchanged" 표기). | `includegraphics` 경로를 `v13_si`로 수정 후 si.pdf 재빌드. |
| 7 | 중 | `si.tex` 표 S2 "External regions (Fig. 4c …)", "Year-wise comparison (Fig. 5b)"; S5 캡션 "pooled value of Fig. 4c"; 국문 si.tex 동일 | v13 패널 번호(외부 지역 = 그림 4d·e, 연도별 = 그림 5d)와 어긋난다. | 4c→4d,e; 5b→5d; S5 캡션 4c→4d. |
| 8 | 중 | en/ko `results.tex` L28 "(Fig. 4e) … Kamchatka (\nftKaRszero to \nftKaRsfive)" | 캄차카(16 셀-연도)는 30개 기준에 미달해 그림 4e에 없다(JSON `e`: 6개 소지역). 값 자체(7.9→11.7)는 `F10_extft.csv` 캄차카 행과 일치. | "Kamchatka (… cm; 16 cell-years, not shown in Fig. 4e)"로 표기하거나 캄차카를 S5 언급으로 돌림. |
| 9 | 하 | en/ko `results.tex` L33 "bias of all series except ESA CCI v5 decreased with the observed ALT" (알래스카) | Fig5 JSON `c` 알래스카 ESA CCI v4는 단조 감소가 아니다(3.2, −3.6, −4.7, −1.5). | "except ESA CCI v4 and v5" 또는 "from the thinnest to the 45–60 cm class". |
| 10 | 하 | 그림 5 캡션 d "Liu et al. end in 2020" | 레나 2017년(≤2020)도 Liu 없이 그려진다(Liu 행이 30개 미만이면 제외, JSON `d.definition`). | "and are omitted in years with fewer than 30 cell-years with a Liu value" 추가. |
| 11 | 하 | 그림 4 캡션 d "(observations of 2010--2024)" | 패널 d의 474행은 2010–2020년이다(JSON `d.years`). 2010–2024는 전체 외부 자료 기간. | "2010--2020"으로 쓰거나 괄호를 삭제. |
| 12 | 하 | 그림 2 캡션 b "2nd--98th percentile" | JSON `b.flow.inputs.dem_slope.percentiles` = [0, 98]. 경사 썸네일만 0–98. | "2nd--98th (slope 0--98th)" 또는 스크립트를 2–98로 통일. |
| 13 | 하 | en/ko `results.tex` L15 "both models reached about \nrmseAkblkRo (18.4) cm in Alaska" | Stefan 재보정은 18.6(JSON 3a·3b 18.58), 잔차 모델 18.4. | "about 18.4--18.6 cm" 또는 두 매크로(`\nrmseAkblkPo`, `\nrmseAkblkRo`) 병기. |
| 14 | 하 | en/ko `results.tex` L19 `\ngcLecorr`·`\ngcCacorr` "(Fig. 3d)" | 상관은 `F4e_gain_conditions.csv`의 `rmse_stefan` 열로 계산되며 패널 d에는 Stefan RMSE가 그려지지 않는다(값 −0.99/−0.79는 재계산 일치). | 유지 가능. 원하면 "computed from the tercile values behind Fig. 3d"로 한정. |
| 15 | 하 | en/ko `results.tex` L43 `\nrdelta*` | 연도 간 변화 상관(core_v6 `multiyear_changes.csv`)은 어느 패널에도 없다. 같은 문단이 그림 5(b→d)를 인용. | 유지 가능(본문 전용 수치). |
| 16 | 하 | en/ko `results.tex` L39 `\nmapAkFootMtnWei·Liu` | 산지 위도대 차이는 보충 그림 S6c의 양이며 문장은 그림 6만 인용. | "(Supplementary Fig. S6c)" 추가. |
| 17 | 하 | `\nllAkRohundred` = "−0.7 [−1.4, −0.0]" | 상한 −0.01이 "−0.0"으로 인쇄된다. 판정("구간이 0 제외")은 맞으나 표기가 어색. | 생성기 `v6_numbers.py` `sfmt`에서 −0.0 처리(예: 소수 2자리) 또는 본문에서 "[−1.4, 0.0]"로 두고 각주. |
| 18 | 하 | 국문 그림 5 캡션 d | 본문은 `\nyrsywAkko`·`\nyrsywLeko`를 쓰는데 캡션은 `\nyrsywAk`·`\nyrsywLe`·`\nyrsywCa`(영문 "and" 포함)를 쓴다. | 캡션을 `…ko` 변형으로 교체(`\nyrsywCako` 존재). |
| 19 | 하 | 영문 SI `\sifig` 라벨 | FigS1·FigS2가 모두 `fig:s1`, 이후 라벨이 한 칸씩 밀림(FigS10 = `fig:s9`). 본문은 `\ref`를 쓰지 않아 출력에는 영향 없음(LaTeX 경고만). | 라벨을 S 번호와 맞춤. |
| 20 | 하 | `SI_figures.json` script 필드, `si_s0*.py`·`fig4/5/7*.py` docstring | S4→`si_s10_…`, S5→`si_s04_…`, S6→`si_s05_…`, S7→`si_s06_…`, S8→`si_s07_…`, S9→`si_s08_…`, S10→`si_s09_…`로 낡은 파일명·번호. fig4 docstring은 c = 합산 곡선(현재 d), fig5 docstring은 "values beyond 45 cm … printed"(현재 압축 축)·b = 연도별(현재 d), fig7 docstring은 d/e 옛 배치. | 메타데이터·docstring 갱신(수치 영향 없음). |
| 21 | 하 | `en_v13/main.tex`·`ko_v13/main.tex` 주석, `ko_v13/sections/*.tex` 헤더 "국문 v11" | 매크로 출처 주석에 `shared_v13` 누락, 국문 헤더 버전 표기 낡음. | 주석 갱신. |

## 3. 과제별 점검 결과

### 3.1 패널 인용 문장과 매크로의 실험 설정 (과제 1)

| 본문 위치 | 인용 패널 | 사용 매크로 | 매크로 원천(스크립트 · 표) | 패널 설정(JSON) | 판정 |
|---|---|---|---|---|---|
| L5 | Fig. 1b | `\ntopyrshareAk`, `\ntopyrAk`, `\nnyrLe` | v6: `labels_cell_year_v5.csv` (in_forcing==1) | 같은 표 | 일치 |
| L7 | Fig. 3a | `\ndforc{Le,Ca}{Po,Ro}`, `\ndforcAkPo` | v6: core_v6 `deltas_forcing.csv`, block, F8TS, pair year_obs:clim, R1 λ0.5 | block, F8TS, clim/year_obs, catboost, λ0.5 | 일치(JSON `a.change…` −1.43/−1.58/−1.49/+1.8) |
| L12 | Fig. 3b | `\nrmse{Ak}rand{Dz}`, `\noptim*`, `\nrmseLe{blk,yr,reg}Dz`, `\nrmseLeregPz`, `\ndref{Le,Ca}reg{Dz,Ro}` | v6: core_v6 `summary_metrics.csv`·`deltas_vs_ref.csv`, year_obs, F8TS | 동일 | 일치(인쇄값 39.3 = `\nrmseLeregDz`) |
| L15 | Fig. 3c | `\nrmse*blk*`, `\ndref*blkRo` (core_v6), `\ndfeat*` (featsets_v6/modis_v6) | v6 | c: F8TS는 modis_v6, F8·F8T·F8TS_X는 featsets_v6 | 일치(−3.59/−3.74/−0.15 = core_v6와 동일) |
| L19 | Fig. 3d | `\ngc*` (F4e csv), `\ndtddLewarm` (core_v6 `conditions_tdd.csv`) | v9, v6 | 같은 F4e 표 | 일치; 상관 열은 패널에 없음(#14) |
| L22 | Fig. 4a | `\nll*` | v6: ladder_v6 `ladder_deltas.csv` LL, ref P1 | 동일 | 일치 |
| L26 | Fig. 4b | `\ntlrmse*` | v6: ladder_v6 `ladder_metrics.csv` TL | 동일 | 일치(인쇄값 39.5 = `\ntlrmseLeDszero`) |
| L28 | Fig. 4c–e, S5 | `\nft*` (F10_extft.csv), `\nftd*` (extft_deltas.csv), `\nftScoredRows` 등 (v11) | v9, v11 | d: 474행(7 소지역, 2010–2020); e: ≥30행 6 소지역 | 합산값 일치; 캄차카 불일치(#8); 700·805행 차이는 본문에 명시됨 |
| L31 | Fig. 5a, 표 S6 | `\ncc*all` | v6: core_v6 `common_cells_eval.csv` mask 0 | 동일 | 일치 |
| L31 | (같은 문단) | `\ndprodLe{wei,ccifive}Romask` | v6: `products_eval_mask25.csv`, 제품별 행 집합 | 5a는 공통 셀-연도 | 행 집합 불일치(#5) |
| L33 | Fig. 5b, 5c | `\nrLe*`, `\nbiasLeAll*`, `\nbias*Thin*`, `\nthinN*` | v13: Fig5 JSON에서 생성 | 동일 | 일치 |
| L37 | S6 | `\ndistwei*med`, `\nbiasLeNorth*`, `\ntlvp*` | v9: qa_summary, `lat_band_bias.csv` (72,74] 띠, `transfer_vs_products.csv` | S6b 중앙값 7.33/107.1/221.72 | 일치 |
| L39 | Fig. 6 | `\nfe*` (v7 F8 JSON, v11 Fig6 JSON) | v9, v11 | Fig6 JSON 17.08/15.76/16.38, 12.66/26.18/30.43, n 95/14, 85 | 일치; `\nmapAkFootMtn*`는 S6c 양(#16) |
| L43 | Fig. 5b(→d) | `\nyw*` | v9: `F5_yearwise_rmse.csv` mask 0, shown | 5d 동일 표 | 수치 일치, 패널 글자 오류(#1) |
| L45 | S7 | `\npadLe*` | v9: `pa_deltas.csv` | S7 JSON | 일치 |
| L48 | Fig. 7c–e | `\nncellmap*`, `\nan*` (v10: 지도 표·v7 F7 JSON), `\nanobs*NoTwenty`(v11), `\ntrend*` (v13 Fig7 JSON), `\naoa*`, `\nhw*` (meta.json) | v10, v11, v13, v6 | Fig7 JSON: r_Le 0.745, r_Ak 0.253, 부호 일치 8/10, 범위 10.23/17.67, AoA 0.5559/0.6172, 추세 2.694/6.398, 98/100 % | 일치; 패널 글자는 재렌더 후 유효(#2) |

### 3.2 그림 안 인쇄 수치 대 매크로 (과제 2)

- Fig. 3b "39.3" = `\nrmseLeregDz`; Fig. 3d 칸 값 "−6.2"(레나 TDD High) = `\ndtddLewarm`, 알래스카 최대 |값| 0.9 = `\ngcAkmaxabs`; 테두리 칸 19/23(레나), 17/23(캐나다) = `\ngcLesig`/`\ngcCasig` 등.
- Fig. 4b "39.5" = `\ntlrmseLeDszero`; Fig. 4e 괄호 n(153, 52, 30, 158, 33, 32) + 캄차카 16 = 474 = `\nftrows`.
- Fig. 5a: 압축 축으로 바뀌어 인쇄 수치 없음; "n = 461 / 449"의 461 = `\nccLeN`. Fig. 5b r = 0.73/0.05/0.07 = `\nrLe*`(RMSE·편향은 캡션대로 본문에만). Fig. 5c 축 아래 1,093 / 248 = `\nthinNAk`/`\nthinNLe`; 축 밖 "+77.4"는 본문 미인용.
- Fig. 6: RMSE 17.1/15.8/16.4, 12.7/26.2/30.4, n 95/14 = `\nfe*`; 축 밖 값(−95, −65, −110, −89)은 본문 미인용.
- Fig. 7(시계열) "n = 143" = `\nanobsLeTwentyN`; Fig. 1 셀 수 13,253/2,206/730 = `\nncell*`; S3 비중 50 %/39 % = `\nshapLeResTerrain`/`\nshapCaResSoil`.
- 반올림을 넘는 차이 없음.

### 3.3 캡션 대 스크립트 (과제 3)

- 일치: Fig. 1(1997–2021 평균, 2010–2024, 제품 6종·사용 연도), Fig. 2(2013년 레나 셀, 15,540 알래스카 셀-연도, 잔차 5개, 2014 알래스카 지도, λ=0.5, 20개 미만 연도 회색, 2012·2019 편차), Fig. 3(2000회 재추출, 0.5° 블록, 30 셀-연도·5 블록, 6 묶음, 8 공변량, 1.3배 규칙으로 39.3 인쇄), Fig. 4(추출 3회, 1000회 부트스트랩, 0·25·50·100·250·500, 474행, ≥30 소지역), Fig. 5(공통 셀-연도 2010–2020, 25 km 마스크, 압축 축, 1:1선·r, 10개 미만 구간 미표시, ≥30 셀-연도 해), Fig. 6(2014/2016, 50 % 영구동토 확률, 250 m 보간, 3 씨앗 평균 겹 밖 예측, 25 km 안 85/95), Fig. 7(50 km·24 km 창, 최소제곱 추세, ±15 cm, ≥50 셀-연도, 2011–2022/2012–2021), S1(≥4개 해, 2015/2020), S2(학습기 5종), S3, S4, S6(600 m, 10셀마다), S7(2013–2021, n 인쇄), S8, S9(2011–2024 평균 / 2016), S10(2010–2024).
- 불일치: 표 #2(그림 7 d/e 재렌더 필요), #10(Liu 누락 규칙), #11(4d 관측 기간), #12(경사 백분위), S5 캡션 "Fig. 4c"(#7). 그림 4b 캡션은 축 밖 인쇄값(39.5)을 언급하지 않는다(그림 3b 캡션과 달리; 선택).

### 3.4 영문 대 국문 (과제 4)

- 매크로: front 11/11, intro 1/1, methods 19/19, results 198/198, discussion 14/14, conclusions 3/3, table1 27/27. 국문은 `\nyrsAkko`·`\nyrsLeko`·`\nyrsCako`(방법), `\nyrsywAkko`·`\nyrsywLeko`(결과 본문)만 `ko` 변형을 쓴다. 순서 차이(methods #10, results #43)는 어순 차이일 뿐 집합은 같다.
- 패널 인용: 결과 절 18개 인용이 같고(영문의 `fig:labels e` 인용 1개가 국문에서는 `c--e` 뒤 문장 구조상 생략), 그림 5b 오류를 공유한다(#1). 보충 참조(S1–S10, 표 S1–S6)도 같다.
- 한쪽에만 있는 문장: 없음(절·문단 수 동일).
- 매크로가 아닌 수치: 두 언어 동일. 서론 21.6 cm·60.5 cm(문헌값), 고찰 68 지점·0.84/0.24/0.54(Gautam), "Half of the Alaska labels"(= `\ntopyrshareAk` 51 %), "about 3 m", "25–100 labels"; 결과 "less than 0.5 cm"(3c 차이 최대 0.34 cm, 확인), "below 10 cm", "about 40 and 70 cm"(국문 40--70 cm; Fig. 5b 산점의 서술값, 원천 JSON에 분포 없음), "below 30 cm"(구간 경계), "±10 cm"(S9: 알래스카 |보정| p99 12.1 cm, 레나 7.7 cm; "mostly" 범위), 방법의 초매개변수·100 km·0.5°·B=2000/1000·30 셀-연도·5 블록·244일·약 6 m, 보충 S4의 201 지점·2,196 관측·2,966 지점-연도. 표 S5·S6의 수치는 `v10_si_tables.py` 생성물(직접 입력 아님).
- 국문 그림 5 캡션의 영문 연도 매크로(#18).

### 3.5 초록 · 결론 · 고찰 (과제 5)

- 초록 `\nncyAll` 19,172, 3.6/3.7 cm(`\ndref*blkRoabs`), 23.8→19.9(`\ntlrmseLeRs*`), 56.2→40.3(`\nftAllRs*`), 21.1 vs 28.4–34.6(`\ncc*`): 모두 결과 절과 같은 매크로.
- 초록·결론 18.0 cm(`\ntlLeDszeroabs`, 전이 실험) 대 고찰 17.8 cm(`\ndrefLeregDzabs`, 지역 홀드아웃) 및 39.5 대 39.3: 두 실험 혼용(#4).
- 고찰 "random cross-validation underestimated … by 3.6–9.9 cm", "ESA CCI v5 (60.7) exceeded … (40.3)", "25 labels … 23.8 to 19.9 … below 21.5": 결과 절과 일치. "The residual learner also improved a product used as the anchor": 레나(−2.3 [−7.4, −0.4])에서만 판정, 알래스카·캐나다는 미구분(결과 L45)이므로 "in the Lena Delta"를 덧붙이는 편이 정확.
- 결론의 "five published products … 28.4–34.6"과 "on par with Wei et al. and Liu et al."은 결과 L31과 일치.

### 3.6 남은 자리표시자 (과제 6)

| 파일 | 자리표시자 |
|---|---|
| `en_v13/sections/front.tex` L6–7 | `[corresponding e-mail]`, `[Co-author]`, `[KOPRI]` |
| `en_v13/sections/front.tex` L18 | `[repository and DOI]` (Data availability) |
| `en_v13/sections/front.tex` L23, L26 | `[Funding and acknowledgements.]`, `[Co-author contributions.]` |
| `en_v13/sections/methods.tex` L83 | `[repository and DOI]` (Code availability) |
| `ko_v13/sections/front.tex` L6–7, L18, L23, L26 | `[교신 저자 전자우편]`, `[공저자]`, `[극지연구소]`, `[저장소 · DOI]`, `[연구비 · 감사의 글.]`, `[공저자 기여.]` |
| `ko_v13/sections/methods.tex` L83 | `[저장소 · DOI]` |
| `main.tex`·`si.tex` `\paperfig`/`\sifig` | "in preparation"/"준비 중"은 그림 파일이 없을 때만 나오는 대체 문구(현재 모든 PDF 존재, 출력 안 됨) |
| TODO/FIXME | 없음 |

## 4. 우선순위

1. (상) 결과 L43 그림 5b → 5d (영·국문).
2. (상) `fig7_maps.py` 재실행으로 그림 7 PDF/JSON을 새 d/e 배치에 맞추고, `v13_numbers.py` 재실행 후 `build.sh`·`package_paper_v13.sh` 재실행(현재 main.pdf·submission.pdf가 results.tex·그림 4·5보다 오래됨).
3. (중) 초록·결론·고찰의 "직접 ML − Stefan" 수치를 한 실험(18.0/39.5 또는 17.8/39.3)으로 통일.
4. (중) 결과 L31의 25 km 마스크 차이에 행 집합(n = 802 / 2,352) 명시 또는 공통 셀 기준으로 교체.
5. (중) `si.tex` `\sifig` 경로 v12_si → v13_si; 보충 표 S2·S5 캡션의 그림 4c/5b → 4d,e/5d(영·국문).
6. (중) 캄차카 문장에 "not shown in Fig. 4e" 또는 S5 참조.
7. (하) #9–#21: 표현·캡션 보완, 국문 캡션 `ko` 매크로, SI 라벨, docstring·JSON 메타데이터 갱신.
