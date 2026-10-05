# 영문 원고 조립 점검표 (Scientific Reports, 2026-10-05, 6차: 검토판, 완결성 점검)

조립 대상은 `main.tex`(본문 PDF)와 `si_main.tex`(Supplementary Information PDF)이다. 단어 수는 명세 10절 규칙대로 렌더 PDF 를 `pdftotext` 로 읽어 절 경계로 자른 뒤 공백 단위로 셌다(`tools/count_words.py`, 결과 `build/wordcount.json`). 위첨자 인용 번호 토큰은 뺐다. 자리표시를 포함한 값과 뺀 값을 함께 적는다(자리표시 정규식에 `[PENDING: ...]` 을 더했다). 2차 수정 내용은 6절, 3차 수정(결과 채움) 내용은 9절, 4차 수정(XC 채움) 내용은 10절, 5차(XB 민감도, XD-5, XE-e) 내용은 11절, 6차(검토판, 그림 번호, 완결성 점검) 내용은 12절에 모았다.

## 1. 산출물

| 파일 | 내용 |
|---|---|
| `main.tex`, `main.pdf` (39쪽, 3.4 MB) | 본문. `\documentclass[pdflatex,sn-nature]{sn-jnl}`, xelatex, bibtex(`sn-nature.bst`) |
| `si_main.tex`, `si_main.pdf` (76쪽, 7.7 MB) | SI. 같은 클래스, 본문 폭 170 mm. 지도 그림 6개(S10–S15) 때문에 커졌다 |
| `sections/si.tex` | Supplementary Methods 1–9, Notes 1–2, Supplementary Fig. S1–S9(자리 상자) |
| `sections/si_maps.tex` | Supplementary Fig. S10–S15(1 km 지도와 XK·XL 그림). `tools/make_si_maps.py` 가 `outputs/figures/paper/v3_restructure/maps/*_legend.md` 에서 캡션을 만든다 |
| `sections/si_tables.tex` | Supplementary Table S1–S18. `tools/make_si_tables.py` 가 생성(손으로 고치지 않는다). S13 c–o(XB–XL)는 `tools/si_x_results.py` 의 값(계획 8.2–8.9, 추가 등록 결과 절에서 옮김) |
| `references.bib` | `refs_*.bib` 7개 병합. `tools/merge_bib.py` 가 생성, 기록 `build/merge_bib.log` |
| `build.sh` | 병합 → xelatex → bibtex → xelatex 2회. OMP_NUM_THREADS=2, nice 10. 로그 `build/main.log`, `build/si_main.log`, `build/*.blg` |
| `tools/count_words.py` | 렌더 PDF 단어 수 |
| `../template/sn_template_dec2024.zip`, `../template/sn_template/` | Springer Nature 템플릿 v3.1(2024-12). sha256 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`, 901,814 바이트(문서 2.1절 기록과 일치) |

TinyTeX 에 설치한 패키지: threeparttable, ncctools, jknapltx, rsfs, appendix, catchfile, newunicodechar. 설치 전에 tlmgr 자체 갱신(`tlmgr update --self`)이 필요했다.

## 2. 컴파일 상태

| 문서 | 오류 | 경고(줄) | BibTeX | 쪽 |
|---|---|---|---|---|
| main | 0 | 10 | 오류 0, 경고 0, 항목 67 | 39 |
| si_main | 0 | 7 | 오류 0, 경고 0, 항목 12 | 76 |

경고 내역(두 문서 공통 유형): (1) xeCJK·ctex 판이 LaTeX 2026-06-01 을 요구하나 설치 커널은 2025-11-01 이다(3건, 동작에는 영향 없음, 커널 갱신은 하지 않았다). (2) 템플릿이 부르는 mathrsfs 의 글꼴 크기 대체(4–5건). (3) hyperref 책갈피 단계 차이 2건(`\bmhead` 가 단락 수준이라 생긴다). 본문은 Overfull 0건이다. SI 는 Overfull 125건(대부분 3 pt 안팎, 최대 약 8.6 pt, 좁은 표 열의 긴 낱말과 굵은 표 머리)이다. S2 의 Plan 열 19 pt 넘침은 줄바꿈 허용으로 없앴다.

## 3. 단어 수(렌더 PDF 기준, 4차 수정 뒤)

| 항목 | 한도 | 자리표시 포함 | 자리표시 제외 | 판정 |
|---|---|---|---|---|
| 제목 | 20단어 | 14 | 14 | 충족 |
| 초록 | 200단어 | 213 | 199 | 충족. XC 는 넣지 않았다(10.1절, 주석 `[DECISION: abstract XC sentence]`). 남은 자리는 XE PENDING |
| 본문(서론 + 결과 + 고찰) | 4,500단어(목표 4,450 이하) | 4,641 | **4,447** | 충족(5차 추가 뒤. 11절) |
| 서론 | (계획 700) | 633 | 633 | |
| 결과 | (계획 2,800) | 3,145 | 2,999 | |
| 고찰 | (계획 850) | 856 | 808 | |
| Methods | 한도 없음 | 4,979 | 4,930 | M7 에 Algorithm P = S1 의 뜻을 더했다 |
| Table 1 설명문 | 150단어(명세) | 142 | 142 | 충족 |

결과 소절별(자리표시 제외, 괄호는 명세 4절 목표): R1 284(300), R2 434(400), R3 384(450), R4 527(350), R5 491(450), R6 371(400), R7 307(250), R8 165(200). R4 는 XC 문장(약 170단어) 때문에 목표를 넘었다.

여유는 3단어다. XE-a·XE-b, XF 결과가 들어오면 다시 줄여야 한다.

그림 설명문(각 350단어 이하, 'Figure N.' 머리 두 낱말 포함, 자리표시 제외)

| 그림 | 단어 | 판정 |
|---|---|---|
| Fig. 1 | 349 | 충족. 다시 그린 그림(04:15:15, md 04:12:36)과 맞춘 판, 패널 e 의 XH 문장 35단어 |
| Fig. 2 | 308 | 충족 |
| Fig. 3 | 323 | 충족 |
| Fig. 4 | 349 | 충족(여유 1단어). 다시 그린 그림(02:49:59)의 `Fig4_legend.md` 와 맞춘 판 |
| Fig. 5 | 350 | 경계. 미확인 자리표시를 채우면 넘을 수 있다 |
| Fig. 6 | 350 | 경계. 그림 작업의 새 판(04:59:09, c 에 기존 ALT 지도 행)과 맞췄고 PENDING 표지를 풀었다(10.3절) |
| Fig. 7 | 347 | 충족. 새 판(md 05:27:56, Fig7.pdf 05:37:28)과 맞췄다. e 에 등록 단서를 더하고 a 의 두 값을 줄였다(10.3절) |

## 4. 표시 항목

본문 8개이다: Fig. 1–7 과 Table 1(Sci Rep 한도 8). 그림은 PDF 만 썼다(`outputs/figures/paper/v3_restructure/Fig1-7.pdf`, 170 mm 폭 그대로, 한 쪽에 하나). Fig. 6 은 그림 작업이 만든 `Fig6.pdf`(02:19:52)를 넣었다. Table 1 은 `Table1_data.tex` 의 tabular 만 읽어(`catchfile`) 넣었고, 제목과 주석은 `legends.tex` 의 Table caption 하나만 쓴다. 클래스의 table 환경(threeparttable)은 상자에 담은 표를 받지 못해 원래 float(`tableorg`)를 썼다. 표 자연 폭이 단일 단 본문 폭(31 pc)보다 약 14 mm 넓어 본문 폭 가운데에 맞췄다.

**PDF 크기**: 3차 컴파일에서 main.pdf 는 3.4 MB 다. 그림 작업이 Fig1.pdf 를 다시 그려(457 KB) 줄었다. 아래는 2차 때의 기록이다. 2차 main.pdf 는 12.8 MB 였다. `pdfimages -list` 로 보면 그림 PDF 안의 래스터는 모두 600 ppi 이고(Fig1 4개, Fig6 10개, Fig7 1개, Fig5 0개) 크기도 작다(합 0.3 MB 미만). 600 ppi 를 넘는 래스터가 없어 내려 표집할 대상이 없다. 크기는 벡터 경로에서 온다: Fig1 은 압축 해제 21.4 MB 의 내용 흐름에 선분 연산 678,029개와 표지 그리기 4,527개가 있고, Fig5 는 선분 292,472개다(이미 Flate 압축, qpdf 무손실 재압축은 0.1% 미만 감소). 줄이려면 그림 스크립트에서 해안선·영구동토 구역 다각형을 단순화하거나 조밀한 바탕 층을 600 ppi 래스터로 그려야 한다(그림 작업의 결정, 이번 작업은 그림을 고치지 않았다).

## 5. 참고문헌

- 본문 67편(명세 계획 71–72편; 권고 60편). 3차에서 서론의 영구동토 온도 문장을 줄이며 Biskaborn2019 가 빠졌다(병합 기록에는 남고 인용되지 않는다). SI 12편(SI 안 인용만, 번호는 SI 안에서 따로 매김).
- 병합 기록(`build/merge_bib.log`): 입력 99항목 → 고유 키 68. 여러 항목이 모인 키 23개 가운데 1개(Poggio2021)는 내용이 같았고, 22개는 내용이 달라 서지 필드가 더 많은 쪽을 남겼다(예: Streletskiy2025 는 refs_front.bib 판, Stefan1891 은 refs_methods.bib 판, Linnenbrink2024 는 권호가 있는 refs_legends.bib 판). 같은 문헌이 다른 키로 들어간 4쌍을 한 키로 합치고 절 파일의 인용 키를 고쳤다: linnenbrink2024_knndm → Linnenbrink2024(legends.tex), MunozSabater2021 → munozsabater2021_era5_land(methods.tex), dunn2023_hierarchical_conformal → Dunn2023, gneiting2007_proper_scoring_rules → GneitingRaftery2007(results_b.tex). 고치지 않았다면 참고문헌 목록에 같은 문헌이 두 번 나왔다. 병합 스크립트는 이제 같은 DOI 가 두 키로 남으면 실패한다.
- `sn-nature.bst`(템플릿 v1.1)는 편집자 없는 inproceedings 항목에서 booktitle 을 버리고 오류를 낸다. 병합본에서 5항목(Prokhorenkova2018, Gorishniy2021, Gorishniy2025, Holzmuller2024, romano2019_conformalized_quantile_regression)을 article(journal = booktitle)로 바꿨다. refs_*.bib 원본은 그대로다.
- 인용 양식: 절 파일이 `\cite` 를 낱말 바로 뒤에 두므로 위첨자 번호(`\setcitestyle{super}`)로 조판했다.
- 2차 수정으로 본문에서 빠진 문헌 인용은 없었다(68편). 3차에서 1편(Biskaborn2019)이 빠졌다. 고찰에서 지운 문장의 인용(Read2019, Jia2021, OMalley2026, Ploton2020, MeyerPebesma2022)은 서론과 결과에 남아 있다.

## 6. 2차 수정(2026-10-05) 기록

### 6.1 본문 줄이기(5,060 → 4,422단어)

원칙: 되풀이를 지우고, 부차 수치는 SI 표·SI 주석·그림 설명문으로 옮겼다. 등록 판정, 단서(분할 독립 가정 의존, 보정 전 유의, 0.5 cm 미만), 사후 표지, 결과 열람 뒤 설계 표지, 한계는 지우지 않았다. 남은 수치는 바꾸지 않았다. 바꾸기 전 파일은 세션 스크래치(`sections_before_cut/`)에 두었고 `git diff` 로 대조할 수 있다(커밋하지 않았다).

| 절 | 줄인 내용 | 옮긴 곳 |
|---|---|---|
| 서론 | 343 위치(Table 1·고찰에 있음), 방법 비교 문장(결과 R1 로 이동, 인용 Read2019·Jia2021·Steyerberg2004 함께), 발견 요약의 CI·P(결과에 있음), 연결 문장 | Table 1, R1 |
| R1 | 설계 반복, $E_0$ 범위와 알래스카 비율, 오차 하한 값, 알래스카 채점 방식 거리·RMSE, 무작위 검증 문장(서론과 중복) | Table 1, Supplementary Table S7f, Supplementary Note 2 |
| R2 | 학습기별 수치, 연도 정합 기준선 CI, CI 기준 대상 수(14, 8), 연결 문장 | Fig. 6 설명문, Supplementary Table S4, S5 |
| R3 | 러시아 서부 13.5/14.0 cm(고찰 D2 에 사후 표지와 함께 남음), 편향 부호 CI, XA 세 문장을 한 문장으로(세 등록 조각, 사후 표지, 보조 CI 단서 유지), 지역 행 수치 | Supplementary Table S6, S8, S13 |
| R4 | 캐나다 2.1·2.7 cm, 대상 수 3/4·2/8, 선택 횟수 22/25·24/25, 알래스카 0.91·0.98·2.1 cm, 캐나다 4.5–5.1 cm, 신경망 조정 1.3–3.0 cm | Fig. 4 설명문, Supplementary Table S6, S8 |
| R5 | 첫 시험 설계 세부, 79.7→64.4 cm², 토양 정의·캐나다 수치 | Methods, Supplementary Table S7g, S10 |
| R6 | 200 라벨 캐나다 값, 레나델타 +1.5–+2.0 cm, 알래스카 두 가중 값, 하위 지역 값, 능동 선정 값, CI 일부, L43 의 CI | Supplementary Table S9 |
| R7 | 러시아 중부 수치, 티베트 P0 RMSE, 티베트 60.0 cm(R2 와 중복) | Supplementary Table S11, R2 |
| R8 | 지역별 포함률 0.75·0.93 | Supplementary Table S12d |
| 고찰 | 결과 수치의 반복(2.5 cm CI, ρ, 2.3 cm, 0.70 cm, 10.3 cm CI, 0.48 cm), 대회 수치 8%·39%·4%·19%, 무작위 교차검증 문장과 라벨 수 설계 선례 문장(서론과 중복) | 결과 절, Supplementary Note 2 |

고친 문장 가운데 의미가 바뀌지 않도록 확인할 곳: R2 'counts based on intervals give the same order'(14 대 8, 원래 18 대 2 와 같은 순서), R6 'The Canadian reduction was smaller with 200 labels'(1.3–1.4 cm 대 2.5–3.2 cm), D4 'differed from them by less than 0.5 cm'(라벨 0 에서 0.48 cm, 라벨 10개에서 동등), D3 대회 문장('a larger gain ... than the stricter test here', 7.8% 대 3.75%, 38.6% 대 19.2%).

### 6.2 열린 문제 처리

| 항목 | 처리 |
|---|---|
| Methods 의 XA 계열 수 | 'Additional registered analyses' 를 실행과 맞췄다: 등록 목록은 7계열, North Atlantic 대상(목록에 없음)을 8번째 단독 계열로 재표집, 등록 7계열 27대상 판은 등록 밖 민감도이며 결과 범주가 같다(`sealed/xa_hyp.csv` 의 deviation 열, XA_result 4절). 이탈 목록에 한 항목을 더했다 |
| Fig. 6 설명문 | `Fig6_legend.md`(렌더판)와 맞췄다: 중심 경도 127° E, 알래스카 확대도 157° W, −40 cm 아래 끝 색, 레나델타 지시선, 물리 출력 입력 CatBoost 를 높은 오차 학습기에 포함, 앵커 앙상블 정의, 재표집 횟수(d 1000, c·e 10,000), 영구동토 구역 ESA CCI PFR v4.0(1997–2021 평균), Cartopy 0.23.0. 수치는 결과 절과 같은 자릿수 규칙(지침 4.5)으로 반올림했다(2.44–5.21 → 2.4–5.2, 2.25 [1.10, 3.42] → 2.2 [1.1, 3.4], 1.00 → 1.0, −59.98 → −60.0) |
| Fig. 4d | 그림 작업이 `Fig4.pdf`·`Fig4_legend.md` 를 2026-10-05 02:49:59 에 다시 만들었다(XA 패널 d). 설명문 전체를 그 판과 맞췄다: 패널 a 의 등록 조건 세 가지, b 의 티베트 값(d 의 −135.7·−5.9 cm 포함)과 북대서양 점 추정, c 의 레나델타·캐나다 평균 표기, d 의 XA 수치(재보정 몫 ρ 0.37 [−0.11, 0.93], 재보정을 넘어선 몫 ρ 0.28 [−0.19, 0.80], 대상 고정 [0.18, 0.54], 범주 '확인하지 못함', 사후 분석). 수치는 결과 절 자릿수로 반올림했다(1.32 → 1.3, 2.10·2.72 → 2.1·2.7, 227.71 → 227.7, −159.55 → −159.6, −135.65 → −135.7, −5.89 → −5.9). 350단어를 지키려고 패널 a 의 '(Supplementary Table S1)' 지시를 뺐다. main.pdf 는 새 Fig4.pdf 로 다시 컴파일했다 |
| PDF 크기 | 4절. 600 ppi 를 넘는 래스터가 없어 줄이지 않았다 |
| Supplementary Note 2 | R1 에서 옮긴 알래스카 채점 방식 거리·RMSE 를 넣고, 백분율을 C4 README 2.6 값으로 고쳤다(38.7% → 38.6%, 3.7% → 3.75%, 'several settings' → '56 settings') |
| SI 표 | S7 에 오차 하한·대회 값·파생값(C4 README 2.6), S8 에 원천 교차검증 조정(C5 README 2.6, LGF-N2)을 더했다 |

## 7. 남은 자리표시(주석 제외, 컴파일되는 본문만, 4차 뒤)

본문 35건(6차 뒤, 같은 문구를 묶으면 25항목, 검토판 첫 쪽 표): DECISION 21(front 13, methods 6, discussion 2), 미확인 5(front 4, methods 1), MISSING 3(results_a 2, results_b 1), PENDING 6(초록 XE, R5 XE, R7 XF·XC-F3, 고찰 D6 XF·XE). X 자리표시는 0건이다. 초록의 XC 결정은 주석 `[DECISION: abstract XC sentence]` 로 남겼다(검토판 목록 4번).

SI: PENDING 15(XF, XE-a·XE-b, XD-5, XC-F3 의 표 행·주석·S2 판정 열), RESULT 30, VERIFY 10, SI 그림 9(S1–S9 자리 상자), 미확인 5, UPDATE 3, verify 3, CITE 3, S1·S2·S16·S17 표 칸 8, DECISION 1(S17 KPDC 행. 제목 쪽 저자 자리는 si_main.tex 에 따로 1건).

찾는 법: `grep -n '\[\(DECISION\|미확인\|MISSING\|PENDING\|X[A-L]\):' sections/*.tex`.

## 8. 열린 문제

1. **Fig. 6c**: 그림 작업이 기존 ALT 지도 행을 더했다(Fig6.pdf 05:00:10). 설명문을 맞추고 PENDING 표지를 풀었다. md 의 'added post hoc', 'was not resolved', '47% of cells gap-filled' 는 등록 표지에 맞춰 'registration deviation; designed after earlier results were viewed', 'no difference was established', '47.4% imputed' 로 적었다(XG 는 사후 분석이 아니라 등록 이탈과 결과 열람 뒤 설계다).
2. **Fig. 1e**: 그림 작업이 Fig. 1 을 3차 작업 중에 두 번 다시 그렸다(04:03:41, 04:15:15; md 04:01:51, 04:12:36). 설명문은 마지막 md(지역 홀드아웃 포함, 속이 빈 기호 = 원천 계수 Stefan)와 맞췄다. md 의 '직접 ML 14.8 to 28.7 cm, Stefan 20.5–22.2 cm' 는 그림 스크립트가 계산한 3지역 평균이라 등록 원천에 없어 쓰지 않았고, 대신 XH 등록 문장의 수치(셀 가중 증가)를 썼다.
3. **본문 길이**: 4,447단어(여유 3). XC 를 채울 때 다시 줄여야 한다. Fig. 1·4(349), Fig. 5(350)는 한도에 붙어 있다.
4. **단일 파일 규칙**: 템플릿은 `\input` 없는 단일 .tex 를 요구한다. 제출 전 latexpand 로 펼치고, 한글 자리표시를 모두 바꾼 뒤 pdflatex 로도 오류 0 을 확인해야 한다(지금은 한글 때문에 xelatex 만 된다).
5. **Code availability 두 판**: front.tex 블록 2 와 methods.tex M15 의 문구가 다르다. 조립은 methods.tex 판을 썼다.
6. **SI 의 한계**: Supplementary Methods 1–7 과 Note 1 은 2026-09-30 영문 초안을 옮긴 것이고 다시 대조하지 않았다([verify]·[UPDATE]·[RESULT] 표지는 파랗게 남겼다). Supplementary Table S2 에는 가설별 Δ·CI·P 열이 아직 없다(S3–S13 에 값이 있다). README 표 셀 17개는 한국어 비고를 번역하지 못해 비고를 빼고 † 를 달았다(`build/si_tables.log`). SI 그림 S1–S9 는 자리 상자다. 지도 그림 S10–S15 는 그림 작업이 바꾸면 `tools/make_si_maps.py` 를 다시 돌린다(지도 작업 에이전트가 아직 실행 중이다).
7. **그 밖**: 초록의 수식 숫자가 클래스 서식으로 굵게 보인다. 본문 그림은 캡션 없이 'Figure N' 표지만 단다(설명문은 Figure legends 절).

## 9. 3차 수정(2026-10-05, X 결과 채움) 기록

원천은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8.2–8.9, `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md` 결과 절, `paper/claims/*/README.md` 6.1 이다. 수치는 옮겼을 뿐 다시 계산하지 않았다. 배치는 계획 6절(결과 전 고정)을 따랐다: XG·XH 는 본문, XB·XD·XE·XI·XJ 는 본문에 지시 문장만 두고 결과는 SI(Supplementary Table S13), XK·XL 은 SI.

### 9.1 채운 곳

| 실험 | 본문(영문·국문) | SI |
|---|---|---|
| XH | R1(직접 ML 증가 5.76, 8.17, 14.20 cm, 재보정 Stefan 0.29, 0.51, 4.28 cm, 서술, 결과 열람 뒤 설계) + Wadoux 반론 문장, Fig. 1e 설명문(35단어) | S13 e·f(단별 RMSE, ΔΔ, 등록 서술 문장) |
| XG | R2(CCI v5 +23.79 cm, Holm P 0.0006, 지역 열세 3곳; 잔차 ML −8.37·−7.51 cm; Wei v2 미결정, 결측 대체 47.4%; 등록 이탈, 결과 열람 뒤 설계, CCI 행 비맹검 부분 포함, ALT 정의 차이), 고찰 D6 | S13 d(확인 대비 6행, 지역 행, 마스크 전후 P0 RMSE, 민감도, 등록 문장) |
| XB | R3·R4·R5 지시 문장(미결정·대체 36%; 40·160 미결정; 0.55·0.498 cm 우세, 200·전량 미결정, 알래스카 비열등; 결과 열람 뒤 설계, 비맹검 부분 포함) | S13 c(16행, 등록 문장, 가중 서술) |
| XI | R5 지시 문장(같은 시기 따뜻한 블록의 공간 대용, 외삽 폭 0.97, '실제 외삽 시험이 아님', XI-c 2.44·2.41 cm 비열등, XI-a·b 미결정) | S13 g(두 판, 약한 쪽 판정) |
| XE(xh0) | R5 지시 문장(H0 10열, 격자 안 동등, 보조 대비, 비맹검 부분 포함) + PENDING(XE-a, XE-b) | S13 h, PENDING |
| XD | Methods M7(Algorithm P = S1), R6 지시 문장(XD-1 미결정, XD-3 비열등 8·열세 2·확인 못함 6, XD-2 사후 설계, 공통 한계) | S13 i–k, Supplementary Methods 8(학습 정책 정의, XD-4 서술, XD-5 PENDING) |
| XJ | R7 한 줄(8.14–83.17 cm, 일반화하지 않음, 원천 쪽 시험하지 않음(약관 근거 미기록)) | S13 l |
| XK·XL | 고찰 D5(1 km 지도 = 표시 해상도 등록 문장, Supplementary Figs S10–S14) | S13 m·n, Supplementary Methods 9(CCI v5 라벨 셀 편향 +35.63·+58.60 cm, v4 −5.09·+10.35 cm, 원인 단정 없음), Supplementary Fig. S10–S15 |
| XF | R7·고찰 D6·Supplementary Methods 2·S18 에 PENDING 표지 | |
| XC | 자리표시 그대로(실행 중) | S2, S13 o, Supplementary Methods 9 |

### 9.2 길이 조정(채운 뒤 4,727 → 4,444단어)

R3·R5 를 먼저 줄였다. 옮긴 내용은 모두 SI 표에 있다. R1: 오차 하한 지시(본문에 그 값이 더 없음). R2: 학습기 이름 목록(Fig. 6 설명문), 연도 정합 Stefan 대비 3.2 cm(S4), 티베트 CI(S4). XG 문장은 'Three results depart from this pattern' 의 세 항목과 섞이지 않도록 새 문단으로 두었다. R3: 물리 입력 대비 CI(S6), 증강 결합 CI(S6), 현지 최소제곱 문장 축약('only in W Russia and Tibet', 다섯 지역 모두 0 이 아니어서 뜻이 같다), ρ 정의 절(Fig. 4b 설명문). R4: 신경망 조정의 학습기별 값(S8). R5: 첫 시험 문장(Methods), 19% 줄일 수 있는 오차(S7), SAR 값(S7), 블록 등가중 구간(S7·S10). R6: 캐나다 200 라벨, 알래스카 전체 미결정, 하위 지역 집계, 능동 선정, L43 의 지역 값(모두 S9). 서론: Biskaborn2019 온도 문장. 고찰: 10.3 cm 반복(R1 에 있음).

등록 단서와 표지는 남겼다: 결과 열람 뒤 설계, 비맹검 부분 포함, 등록 이탈, '실제 외삽 시험이 아님', XJ 일반화 금지 문장, 후보 풀 한계, 보정 전 유의, 0.5 cm 미만, 분할 독립 가정 의존. 'partly non-blind' 는 'partly unblinded' 로 통일했다.

### 9.3 발견한 불일치

1. Fig. 6c 에 XG 행이 없다(8절 1).
2. Fig. 1e 의 md 설명문 수치가 등록 원천에 없는 3지역 평균이다(8절 2).
3. XH 봉인 재사용 표(`xh_reuse_h41.csv`)가 0행이라 V-G 값은 원천 표에서 읽었다(계획 8.3 기록, S13 주석에 적음).
4. 3차 지시는 XD-4 수치(시험 과제 5개 가운데 2개 우세·2개 열세)를 주었으나, 계획 6절과 명세 11절은 XD 를 SI 로 고정하고 R6 지시 문장을 XD-1·XD-3·공통 한계로 정했다. 본문 R6 에는 XD-1·XD-3·XD-2(사후 설계 표지)와 공통 한계만 두고 XD-4 는 SI(Supplementary Methods 8, S13 j)에 두었다.
5. 국문판 front 의 DECISION 은 12건(영문 13건, 저자 자리 구조 차이, 2차부터 같음).

## 10. 4차 수정(2026-10-05, XC 채움) 기록

원천은 계획 8.10(봉인 표 첫 열람 05:20:39 KST)과 `data/processed/xbatch/XC_workflow_end_to_end/sealed/`(`xc_hyp.csv`, `xc_sentences.json`, `xc_contrasts_main.csv`)이다. 수치는 옮겼고 다시 계산하지 않았다. 모든 XC 문장에 'in new random splits of the same regions'(국문 '같은 지역을 다시 무작위로 나눈 분할에서')를 넣었다.

### 10.1 채운 곳

| 자리 | 영문·국문 내용 |
|---|---|
| 초록 S8 | XC 를 넣지 않았다. 수치 문장은 S7 을 대신해도 214단어가 되어 200단어를 넘는다. 계획 7.1 은 제외를 허용한다. 후보 문장(34단어, 'pooled across regions; over 0.5 cm larger error in 2 of 3 regions')을 주석 `[DECISION: abstract XC sentence]` 에 두었다 |
| R4 | XC-1c −0.93 cm(레나델타·캐나다 풀; 캐나다 −2.41, 레나델타 +0.54 미결정), XC-1b 미결정(최소 검출 효과 약 0.5 cm; 레나델타 +0.21), 배치가 무작위로 바뀌어 같은 라벨에서 규칙 W 와 고정 레시피를 비교한 것이라는 문장, XC-2a–c 비열등(라벨 10개는 5지역 평균, 40·160개는 레나델타·캐나다)과 XC-2s(−1.35, −1.51 cm), 풀 평균 한정과 지역 손해(러시아 동부 +1.34, 선택 계열 알래스카 +1.51·+2.34, 레나델타 +0.53), XC-r(지역 안 규칙 W 대 잔차 ML: 200개 동등, 500·1000개 미결정). 표지: 재사용 지역의 재검정, 결과 열람 뒤 설계, 비맹검 부분 포함 |
| R6 | 절차가 셀 무작위 추출로 바뀌어 배치 단계는 기여가 없고 그 효과(XC-5)는 시험하지 않았다는 문장과 등록 공통 문장(알래스카 계열에서 고름, 평가 셀이 배치·규칙 설계에 쓰임, 같은 셀의 새 분할) |
| R7 | XC-F3 은 PENDING 표지(배치가 무작위라 라벨 10개 대비가 같은 라벨이고 적격 지역이 없음, 등록 문장은 XF 마감 2026-10-11 14:22 KST 뒤) |
| 고찰 D6 | 조건 문장: 워크플로 전체는 같은 지역의 새 분할에서만 시험, 배치는 무작위, 비열등은 풀 평균에서만(러시아 동부, 알래스카, 레나델타 제외) |
| Fig. 7e | 그림 작업의 새 설명문(수치만)에 'in new random splits of the same regions', 'placement reduced to random', 'partly unblinded' 를 더했다 |
| Methods M7 | Algorithm P = S1 이므로 배치 단계는 기여가 없고, 라벨 10개에서는 비교 팔과 같은 라벨·모델, 라벨 40·160개의 무작위 배치 + 고정 레시피 대비는 같은 라벨의 규칙 W 대 고정 레시피라는 문장 |
| SI | Supplementary Table S13 o(주 가설 8행, 지역 행 15행, 등록 문장, S-XC1–S-XC12, LOO, XC-r, Holm 표), S2 의 XC 판정 열, Supplementary Methods 9 의 XC 실행·이탈 문단 |

### 10.2 길이 조정(XC 채운 뒤 약 4,690 → 4,440단어)

R3·R5 를 먼저, 다음에 고찰을 줄였다. 옮긴 수치는 SI 표에 있다. R2: CatBoost 블록 등가중 구간(S3·S4), 선형 위약의 0.5 cm·라벨 10개 동등 설명(S3, 고찰 D4 에 남음), 학습기 이름을 '세 학습기'로, 'Three results depart from this pattern' 문장. R3: 물리 입력 대비 라벨 0개 CI(S6), 증강 결합의 Holm 동등성 P(S3), 라벨 3개 현지 최소제곱 서술 문장(서술 비교, 그림 3d·S6). R4: 비열등 미확인 CI 두 개(S8), '레나델타에서 차이 미확인' 반복, 캐나다의 λ 1.0 선택(고찰 D2 에 남음). R5: 분할 수, 시험 설명, 레나델타·캐나다 라벨 수, 레이더 공변량 문장(S7), 72% 문장 압축. R7: 5지역 풀 CI, 티베트 CI(S11). R8: 생성 분위 구간 포함률(S12), 위약 CI, AB10 문장 압축. 서론: 'These studies differ in how they were evaluated'. 고찰: D1 의 규칙 W 비열등 반복, D2 의 ρ·XA 반복(R3 에 남음), D3 의 물리 입력 비교 문장(R3·R4 에 남음, 인용은 서론에 남음)과 Wadoux 반복(R1 에 남음), D5 의 이득 범위, D6 마지막 절 압축.

등록 단서와 표지는 남겼다: 보정 전 유의, 분할 독립 가정 의존, 0.5 cm 미만, 사후 표지, 등록 판정 '혼재', 결과 열람 뒤 설계, 비맹검 부분 포함.

### 10.3 발견한 불일치

1. 조정 담당은 Fig7_legend.md 가 05:37 무렵 다시 쓰였다고 했으나, 파일 시각은 05:27:56 그대로이고 05:37:28 에 다시 그려진 것은 Fig7.pdf 다. 설명문은 05:27:56 판 내용(348단어)과 맞췄다.
2. Fig. 7 md 의 e 문장에는 등록 단서('in new random splits of the same regions', 비맹검 부분 포함)와 배치가 무작위였다는 사실이 없다. 원고 설명문에 더했고, 350단어를 지키려고 a 의 알래스카 RMSE 쌍·CI(본문 R5 에 있음)와 19% 값(S7)을 뺐다. md 와 원고 설명문이 이 세 곳에서 다르다.
3. Fig. 6 md 의 XG 문구('added post hoc', 'was not resolved', '47% of cells gap-filled')는 등록 표지와 다르다(10.3절 위, 8절 1).
4. 봉인 문장(`xc_sentences.json`)의 XC-F3 문장은 새 지역에 대한 문장인데도 '같은 지역을 다시 무작위로 나눈 분할에서'를 붙였다(생성 규칙의 결과). 원고는 XC-F3 을 PENDING 으로 두어 이 문장을 쓰지 않았다.
5. 계획 8.10 은 XC-1b·1c 가 S-XC1(같은 라벨의 규칙 W 대 고정 R1)과 같은 값이라고 적었다. 원고 R4 는 이를 '같은 라벨에서 방법 규칙과 고정 레시피를 비교한 것'으로 적었다.

## 11. 5차 수정(2026-10-05, 마지막 추가 결과) 기록

원천은 계획 8.6 끝(XE-e, 첫 열람 05:42:35), 8.11(XD-5, 05:38:08), 8.12(XB CCI v5 민감도 판, 05:43:31)이다. 수치는 옮겼고 다시 계산하지 않았다.

| 항목 | 본문(영문·국문) | SI |
|---|---|---|
| XB-3 단서 | R5 의 XB-3 문장 괄호에 'not retained when CCI v5 was added'(국문 'CCI v5를 후보로 더하면 유지되지 않음')를 넣었다. 본문에서 XB-3 이 나오는 곳은 이 한 곳이다 | S13 c 주석에 같은 문장, S13 c(이어서)에 민감도 판 표(XB-1–XB-4, 판정·Holm P, 지역 행 변화, 가중). S2 의 XB 판정 열에 같은 단서. 등록 판정은 핵심 판 그대로라고 적었다 |
| XD-5 | 없음(SI 배치) | Supplementary Methods 8 의 PENDING 을 서술 문장으로 바꿨다(후회가 작은 과제 3개·큰 과제 2개, 계열 하나 제외 효용 S9-DS 2/5·S9-GBM 5/5 음수, 사후, S9-DS 효용의 섞기 seed 민감성은 결정성 위반이 아님). S13 k(이어서) 표 두 개와 주석 |
| XE-e | 고찰 D5 한계 문장 끝에 'on-site soil moisture did not explain the within-grid residual either (diagnostic; Supplementary Table S13)'(국문 '현장 토양 수분도 격자 안 잔차를 설명하지 못했다(진단; 보충 표 S13)') | S13 h(이어서) 표(주 −3.54·−4.89 %, −4.58·−4.48 %, 민감도 −6.59–+2.55 %, 0.5 cm 등가 비율 7.9·5.0 %)와 주석(진단, 판정을 바꾸지 않음) |

길이: 위 두 문장(약 20단어)을 넣고 R1 의 'as in lake-temperature studies'(인용은 서론에 남음), 서론의 러시아 서부 $E$ 값(그림 1b 에 있음), R4 의 라벨 40개 CI(S8), R4 의 레나델타 문장 그림 지시를 줄였다. 본문 4,447단어.

SI Supplementary Methods 9 의 실행 문단에 세 실행(시각, 열람 시각)과 'XB 등록 판정은 핵심 판, XB-3 은 민감도 판에서 유지되지 않음', 'XE-e 는 판정을 바꾸지 않음'을 더했다.

## 12. 6차(2026-10-05): 검토판과 완결성 점검

### 12.1 검토판

- `main_review.tex` → `main_review.pdf`(24쪽), 국문 `../ko/main_review.tex` → `main_review.pdf`(25쪽). 만드는 법: `./build_review.sh`(국문은 `../ko/build_review.sh`). 두 판 모두 오류 0.
- `tools/make_review.py en|ko` 가 같은 절 파일을 `build/review/sections/` 에 복사하며 [DECISION]·[PENDING]·[미확인]·[MISSING] 표지를 회색 위첨자 번호 `[N]` 로 바꾸고(같은 문구는 같은 번호), 첫 쪽 '검토 메모 / Open items' 표(번호, 종류, 위치, 내용; 25항목)를 만든다. 주석과 front.tex 블록 2 는 처리하지 않는다. 초록 XC 결정(주석)은 목록에 넣고 초록 끝에 번호를 단다.
- 그림마다 본문 폭 그림과 바로 아래 설명문을 한 float 으로 묶어 처음 인용한 문단 뒤에 넣었다. 표 1 도 처음 인용한 문단 뒤에 넣었다. 본문 폭은 170 mm(검토판만).
- 제출판(`main.tex`, 그림은 끝)은 아래 수정 외에는 바꾸지 않았다.

### 12.2 그림 번호(처음 인용 순서)

본문의 처음 인용 순서가 1, 3, 6, 2, 4, 7, 5 였다. 순서를 맞추려면 번호를 바꿔야 해서 원고 안에서만 바꿨다(그림 파일과 `*_legend.md` 의 번호는 그대로). 원고 그림 4 = 파일 Fig6, 그림 5 = Fig4, 그림 6 = Fig7, 그림 7 = Fig5. 본문·설명문·SI 의 인용, 설명문 순서, `\PlaceFigure` 의 파일 대응을 함께 바꿨다(영문·국문). 그림 작업이 보내는 설명문 변경은 파일 번호이므로 이 대응으로 옮긴다. 이제 처음 인용 순서는 그림 1–7, 표 1 이 차례대로다.

### 12.3 패널 인용

빠진 패널을 인용했다: 그림 1a(서론, 지역과 라벨 수), 그림 2a,b(R1, 두 기준선), 그림 2c–f 와 3d(R3), 그림 7a–e(R6 첫 문장). 모든 그림의 모든 패널이 본문에서 인용된다. 늘어난 단어는 R1 의 표 1 반복 인용과 R7 문장 압축으로 맞췄다(본문 4,449단어).

### 12.4 확인한 것과 고친 것

- '??', 정의되지 않은 인용·참조: 세 PDF 모두 0건.
- Methods 식: 식 (1)–(6)의 기호를 모두 본문에서 정의한다. 앵커 식의 $x$ 정의를 더했다.
- 수치 추적: 서론·결과·고찰의 소수 수치 전부가 계획 8절, 추가 등록 결과 절, claims README·표, 원고 명세, 그림 값 파일 가운데 하나에 같은 표기로 있다(문자열 대조, 반올림 값은 명세의 근거 표에 있음).
- [미확인] 해결(프로젝트 기록 근거): Copernicus DEM 인용 형식과 귀속 문구(`docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md` 1.1 의 9행, 확인 상태 '확정(웹)'), 영구동토 범위·구역 자료의 이름과 판(ESA CCI Permafrost extent v4.0, netCDF 속성 product_version 04.0, 1997–2021 연별 파일), Cartopy 0.23.0(설치 판과 그림 설명문), S17 의 JRC 지표수 인용(Pekel et al. 2016).
- 남긴 [미확인](파일로 확인할 수 없음): SoilGrids 취득 시점 판 표지, PolSAR 상향 ALT 자료 DOI, CCI 영구동토 범위 v4.0 DOI, ERA5-Land 귀속 문구, Kudryavtsev 1974 서지(참고문헌 색인이 미확인으로 표시), SI 의 S5 동률 수와 티베트 현장 자료 약관.

### 12.5 아직 끝나지 않은 것

1. 사용자 결정: 저자·소속·교신 저자와 전자우편, 감사의 글·저자 기여·이해상충 문구, 연구비, LLM 사용 문구, 저장소와 DOI, 코드 공개 시점, KPDC 자료 사용, 대회 보고서 인용, 지역 이름 표기, 러시아 중부 영문 표기, 방법 초안 183행 수용, Fig 7d(파일 번호) 경로의 y 기준, 초록 XC 문장.
2. 대기 결과: XE-a·XE-b(10월 8일), XF·XC-F3(10월 11일).
3. 빠진 값: 없음(12.6절에서 채움).
4. SI(12.6절의 새 번호): 보충 그림 S7–S15 는 자리 상자, 보충 표 S17 b 부분의 [RESULT]·[VERIFY] 틀, 보충 표 S1 의 가설별 Δ·CI·P 열, 보충 표 S12 의 약관·DOI 칸(S13 의 판 칸은 채워짐), Supplementary Methods 1–7 의 [verify]·[UPDATE] 표지가 남는다. [CITE] 3곳은 12.6절에서 인용으로 바꿨다.
5. 보충 표·그림 번호는 12.6절에서 처음 인용 순서로 바꿨다. 본문에서 인용하지 않는 보충 표 5개(S14–S18)가 남는다.
6. 그림 재양식(그림 작업 진행 중): 색·기호를 말하는 설명문 문구가 바뀌면 파일 번호 대응(12.2)으로 옮기고 두 판을 다시 컴파일한다.

### 12.6 6차 추가(2026-10-05): 빠진 값, 참고문헌, 보충 번호

**빠진 값(M-04)**: `tools/build_numbers.py`(읽기 전용, pandas)가 `build/numbers_m04.json` 을 쓴다. 정의: 지역 중앙값은 지역별 셀 가중 Δ 의 중앙값, 러시아 서부 제외 값은 나머지 지역 Δ 의 등가중 평균(풀 평균 행이 지역 행의 단순 평균과 같음을 스크립트가 확인), 오차 감소·증가 지역 수는 등록 4분 규칙(두 95 % CI). 원천: R3 은 `paper/claims/C1_label0_safety/tables/lgw_bundle.csv`(AB4 P1 − P0, AB7 R1 − F1k, 라벨 10개), R7 은 `data/processed/lgd/lgd_tests_lic.csv`(약관 확인분 판, 풀 PE1, L8e R1 − P0 전량, L4e R1 − P1 라벨 10개; 지역 판정은 저장 열이 비어 CI 열에서 같은 규칙으로 만들었고, 같은 지역의 초록 묶음 판정과 일치). 값: AB4 중앙값 −0.06 cm, 러시아 서부 제외 +0.72 cm; AB7 중앙값 −2.2 cm, 러시아 서부 제외 −0.32 cm, 4지역 가운데 2지역 오차 감소; PE1 전량 중앙값 −0.78 cm, 러시아 서부 제외 −0.49 cm, 감소 1(러시아 서부)·증가 1(캐나다); PE1 라벨 10개 중앙값 −0.03 cm. [MISSING] 3건을 모두 채웠다. 늘어난 단어는 R5·R6·R7 첫 문장의 블록 등가중 구간(보충 표에 있음), 고찰의 캐나다 문장과 D6 워크플로 절, 서론의 지역 나열(R1 에서 정의)로 맞췄다(본문 4,442단어). 이어서 D6 압축에서 빠졌던 XC 지역 단서(러시아 동부·알래스카·레나델타에서는 비열등이 성립하지 않음, XC 필수 사항 3)를 되살리고, 늘어난 5단어는 R4 선정 규칙 문장의 블록 등가중 구간(보충 표 S4 의 E3 행에 있음)으로 맞췄다(본문 4,447단어, 국문 같은 자리).

**참고문헌**: 웹에서 확인한 서지로 세 항목을 `refs_methods.bib` 에 더했다. Klene et al. 2001(Crossref, doi:10.1080/15230430.2001.12003416, AAAR 33(2) 140–148), Romanovsky & Osterkamp 1997(Crossref, doi:10.1002/(SICI)1099-1530(199701)8:1<1::AID-PPP243>3.0.CO;2-U, PPP 8(1) 1–22), Kudryavtsev et al. 1974(DOI 없음; 영구동토 백과 항목의 참고문헌과 PPP 논문 참고문헌 검색 요약: MSU 1974, CRREL Draft Translation 606, Hanover NH, 1977). Methods 의 [미확인: Kudryavtsev 1974 서지]와 SI 의 [CITE] 3곳을 인용으로 바꿨다. 본문 참고문헌 68편, SI 15편.

**보충 표·그림 번호(처음 인용 순서)**: `tools/si_numbering.py` 의 대응으로 바꿨다. 표: 옛 2→1, 13→2, 5→3, 8→4, 6→5, 7→6, 10→7, 9→8, 11→9, 12→10, 1→11, 17→12, 16→13, 3→14, 4→15, 14→16, 15→17, 18→18. 그림: 옛 S10–S15(지도·XK)→S1–S6, 옛 S1–S9(자리 상자)→S7–S15. 손으로 쓴 절 파일(영문·국문, si.tex)은 한 번 변환하고 머리에 표지 주석을 달았다. 생성기(`make_si_tables.py`, `si_x_results.py`, `make_si_maps.py`)는 옛 번호로 쓰고 출력에서 대응을 적용하며, 표는 새 번호 순서로 낸다. si.tex 에서 지도 그림을 자리 상자 앞에 두었다. 점검: 영문·국문 본문의 보충 표 첫 인용 순서 1–13, 보충 그림 1–5, SI 의 표 머리 1–18 과 그림 1–15 가 차례대로다. 본문에서 인용하지 않는 보충 표는 S14(초록 묶음), S15(라벨 0 학습기), S16(배치 시나리오), S17(선택 이력), S18(XF 조사)로 SI 안에서만 인용된다. 이 문서의 앞 절(9–11절)의 보충 표 번호는 옛 번호다.

## 13. 7차 수정(2026-10-05, 그림 v4 양식 반영)

- 그림 4(파일 Fig6) 설명문을 `Fig6_legend.md`(13:01)와 맞췄다: '(+23.79 cm at axis end; 95% CI 15.34 to 30.16)', 'weighted by cell and block', 'arrow' 삭제. 늘어난 1단어는 md 와 같이 구간 괄호의 '(CIs;' 를 빼서 맞췄다(350단어). 국문판 같은 자리(축 끝에 표시, CI 약어 삭제).
- 화살표·삼각형 서술: 단어 경계 검색으로 본문, 설명문, SI, 생성기에 다른 곳이 없음을 확인했다. 다시 그린 그림 4d 의 지역 기호에는 삼각형(러시아 서부 ▲, 러시아 동부 ▼)이 남아 있으나 그림 안 범례로 표시되고 설명문은 기호를 말하지 않는다.
- SI 그림 S7–S15: `tools/make_si_figs.py` 가 `outputs/figures/paper/v3_restructure/si/FigS<n>.pdf` 와 `FigS<n>_legend.md` 를 찾아, 있으면 그림과 설명문(새 번호 그대로, 다시 매기지 않음)을 넣고, PDF 만 있으면 계획 설명문에 [SI legend: to be written] 표지를 달고, 없으면 자리 상자를 둔다. si.tex 의 자리 상자 9개는 생성 파일 `sections/si_figs.tex` 로 옮겼다(설명문 문자열 같음). 13:58 현재 si 폴더는 없다.
- `make_si_maps.py` 를 다시 실행했다(지도 PDF 13:03–13:04, 설명 md 03:55, 내용 변화 없음).
- 다시 컴파일(13:5x): 영문 본문 40쪽, SI 77쪽, 국문 본문 29쪽, 영문 검토판 24쪽, 국문 검토판 25쪽, 오류 0, 미정의 참조 0. 본문 4,447단어, 초록 199단어, 설명문 최대 350단어, 표 1 제목 142단어. 검토 메모 22항목.

## 14. 8차 수정(2026-10-05, XM 추가)

- 원천: `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md` 결과 절(14:20 열람, 커밋 e454a71). 수치는 그 표에서 옮겼고 봉인 표는 열지 않았다.
- 보충 표 S2: 'q. XM' 부분을 더했다(XM-a 격자 안 3대상 평균·지역·1,000·500 행, XM-b 총 RMSE 평균·지역 행, XM-c 레나 격자 사이 행, 두 가중 CI, 4분 판정, Holm P). 주석에 설계 표지(결과 열람 뒤 설계, 탐색(SI), 비맹검 부분 포함, XE Holm 가족 밖), 입력 정의, 등록 해석 문장(우세 갈래, '통계적으로 구별되나 0.5 cm 미만', 격자 안 설명 비율 1.2 %, C8·지도 문구 불변, 확인 시험 후보, 원인 구별 안 함), 사실 기록(레나 총 RMSE −1.50 cm, 대부분 격자 사이 −1.80 cm), 덮음 비율, 열람 시각, 재현 관문을 적었다. 표 제목은 'XA to XM', 머리 설명은 'Parts c to q' 와 XM 등록 문장으로 고쳤다.
- 보충 표 S1(가설 등록부): XM 행(addendum XM, exploratory, ff7c97f 10-05, PU; designed after viewing, 'XM-a partially supported').
- 보충 표 S12(자료와 약관): WorldCover 2021 v200(CC BY 4.0, 표기 문구, Zanaga et al. 2022 doi)과 Sentinel-2 L2A(Copernicus Sentinel 약관, 표기 문구) 두 행을 'XM only' 로 더했다(추가 등록 문서 8절).
- Supplementary Methods 9: 제목을 'XA to XM' 으로 고치고 XM 문단(등록 커밋, 표지, 입력, WorldCover 51타일, Sentinel-2 534장면, 반사도 offset 정정을 적합 전 등록 이탈로 기록한 커밋 31e19f7, 특징 표 sha256 고정, 148단위, 재현 관문, 열람 시각)을 더했다.
- 고찰 D5: XM 문장을 넣지 않았다. 본문이 4,447단어라 3단어 여유뿐이고, 예시 문구를 넣으면 약 14단어가 늘어 4,450단어를 넘는다. 국문판 본문도 바꾸지 않았다(SI 는 영문판만 있음).
- SI 그림: `make_si_figs.py` 로 S7, S14, S15 를 넣었다(14:28–14:29 파일, 설명 md 없음, 계획 설명문과 [SI legend: to be written] 표지). S15 계획 설명문의 'earlier Fig. 7' 을 'an earlier draft of the main figures' 로 고쳤다.
- 다시 컴파일: 영문 본문 40쪽, SI 82쪽, 국문 본문 29쪽, 검토판 24·25쪽, 오류 0. 본문 4,447단어, 검토 메모 22항목(변화 없음).

## 15. 9차 수정(2026-10-05, SI 그림 배치)

- SI 그림: 그림 담당이 만든 `outputs/figures/paper/v3_restructure/si/FigS7–S12, S14, S15`(PDF, PNG, _legend.md, 14:43–14:48)를 `tools/make_si_figs.py` 로 넣었다. 설명문은 `tools/si_fig_captions.py` 에 SI 설명 양식(굵은 제목 문장, 패널 글자, 공통 표기는 S8 참조)으로 줄여 두었고, 범례 파일의 모든 수치가 설명문에 있는지와 범례 파일 sha256 앞 16자를 생성 때마다 대조한다(현재 통과). 단어 수(범례 → 설명문): S7 197 → 180, S8 206 → 182, S9 212 → 176, S10 192 → 179, S11 235 → 230, S12 238 → 216, S13 151 → 149, S14 228 → 214. S11 에는 본문과 같은 표지(XE 보조 대비·비맹검 부분 포함, XI '진짜 외삽 시험 아님', XB 비맹검 부분 포함)를, S12 e 에는 'XD, exploratory' 를 더했다.
- S7: 제목과 첫 문장을 'M1 6겹 채점 방식 비교(보충 노트 2 인용)이며 XH 5단계 검증 사다리(Fig. 1e)가 아니다'로 고쳤다.
- S13(독립 지역 예측 지도): 격자 예측 산출이 없어 뺐다. 이 그림은 본문과 SI 어디에서도 인용되지 않았으므로 빼고 뒤 번호를 당겼다(S13 = 파일 FigS14, S14 = 파일 FigS15). `si_numbering.py` 의 그림 대응도 옛 S8 → S13, 옛 S9 → S14 로 고쳤고 옛 S7 을 가리키면 오류가 나게 했다. 파일 FigS15 범례의 'Supplementary Table S3'(초록 묶음의 옛 번호)은 새 번호 S14 로 썼다.
- 첫 인용 순서: S1–S5 본문(고찰), S6 Supplementary Methods 9, S7 보충 노트 2, S8–S14 는 Supplementary Figures 첫 문단에서만 인용된다. 첫 문단 문장을 생성 파일 `sections/si_figs_lead.tex` 로 옮겨 S1–S6 문장 뒤에 붙였다(문장 하나만 있는 쪽 제거).
- SI 표의 내부 표지 접두를 새 번호로 고쳤다: [S1: per-hypothesis …], [S11: from lg_targets …], [S12: licence] 3곳, [S13: Python version …].
- 다시 컴파일: 영문 본문 40쪽, SI 84쪽, 국문 본문 29쪽, 검토판 24·25쪽, 오류 0, 미정의 참조 0. SI 그림 자리 상자 0. 검토 메모 22항목.
- SI 에 남은 표지: PENDING 12, RESULT 30, VERIFY 10, verify 3, UPDATE 3, 미확인 3, DECISION 2, 내부 표지 6([S1:] 1, [S11:] 1, [S12:] 3, [S13:] 1).

## 16. 10차 수정(2026-10-05, SI 표지 정리)

SI 에 남은 표지를 (a) 기록으로 채움, (b) 등록 마감 문장, (c) 검토 메모 항목으로 나눠 처리했다. SI 본문과 표에는 표지가 없다(남은 것은 사용자 결정 DECISION 2개: SI 표지의 저자, 보충 표 S12 의 KPDC).

- (a) 채운 것: Supplementary Methods 6 의 교차 환경 점검 (i)(LG 개정 14), (ii)(WRAPUP 8.4 (ii)), (iii)(LG 개정 15 (u)) 결과와 계속 결정, 집계 출처(본 실행 작업 완료, 12:25)와 RealMLP 조각 완결(231/231, LG 개정 15 (s)); 'GPU training is not bitwise deterministic' 문장은 점검 (iii) 결과(로컬 반복 동일, 플랫폼 사이 차이)에 맞춰 고쳤다. Supplementary Methods 5 의 설정 수(MLP·TabM·RealMLP 33, FT-T 34; LGF 개정 5 의 E1). Supplementary Methods 2 의 융해관 규칙 사실(자료 README D 단서 3; v2 에서 더한 CALM 지점만 제외, 그 밖의 v3 직접 셀 가운데 융해관 대응 2, 모호 7)과 Supplementary Methods 1 의 같은 범위 문구. Supplementary Methods 1 의 GPR 비율(실험 기록 E4.4: ABoVE 알래스카 점관측 219,089개의 86 %, 탐침 7.7 %, GPR 우세 셀 10,835/13,606; 옛 표지의 '셀의 86 %'는 단위가 틀렸다). Supplementary Note 1 의 L7(기각), L13(미지지), AB4, AB5, AB9, L4(LG 7.1·7.2, 보충 표 S14). 보충 표 S17 b 의 [RESULT] 23·[VERIFY] 10: `tools/build_reclass_summary.py`(읽기 전용)가 `data/processed/lgw/lgw_rescore.csv`(M4.4 가 지정한 원천) 행을 CI 종류와 4분 판정별로 센 값, H30·M1 의 감사 이력과 관련 시험(AUDIT stats:F4, RESEARCH_FRAME). 보충 표 S11 North Atlantic 행(LGD 전체 판 조각 기록: 분할 5, A 19–24, 채점 12–19(블록 3–7), E0 1.59). 보충 표 S12 약관 3칸(ABoVE, ERA5-Land, CCI ALT; 서지 초안 1.1 의 '확정(웹)'). 보충 표 S13 의 WF·X 작업 A 판(Python 3.11.10, PyTorch 2.4.1 CUDA 12.1 빌드·CPU, pytabkit 미설치; 실행 기록 env.log·deps.log).
- (b) 등록 마감 문장: PENDING 12곳을 'Pending: registered data deadline 8 October 2026 (XE stage 2 …)' 또는 '… 11 October 2026 (XF / XC-F3 …)' 문장으로 바꿨다.
- (c) 검토 메모: 새 항목 4개([23] CALM·ALLena 파서 범위 규칙, [24] 티베트 현장 자료 약관, [25] 네 자료원 이용 허락 요청, [26] 부분 번역 칸 17개)와 기존 항목에 SI 위치를 붙였다([1], [3], [5], [6], [15], [17], [19]). `tools/make_review.py` 의 SI_MERGE·SI_ITEMS. Supplementary Methods 8 의 동률 수는 '세지 않았다'는 사실 문장으로 바꿨다. Supplementary Methods 2 의 허락 상태는 '허락 기록 없음(10월 5일), 약관 확인분 판이 주 판정' 문장으로 바꿨다.
- 내부 표지 삭제: [S1:], [S11:], [S12:]×3, [S13:], [S16]×2, [old result, recheck].
- 표지 수(SI PDF): 이전 RESULT 30, PENDING 12, VERIFY 10, verify 3, UPDATE 3, 미확인 3(소스 4), 내부 9, DECISION 2 → 이후 DECISION 2 만. 검토 메모 22 → 26항목. SI 85쪽, 검토판 24·25쪽, 오류 0.

## 17. 11차 수정(2026-10-05, 부분 번역 칸)

- 용어집으로 옮기지 못한 SI 표 칸 17개(원문 15종)를 `make_si_tables.py` 의 MANUAL_EN 에 영문으로 옮겼다(같은 뜻, 새 주장 없음). 단검표와 표 설명의 단검표 주석을 지웠다. XA 표의 '가족 밖', '음 방향 … 최대 0.31' 두 칸(용어집을 거치지 않던 칸)도 옮겼다. SI 에 남은 한글은 사용자 결정 표지 두 개뿐이고, 남은 단검표 7개는 보충 그림 S14 그림 안의 기호(설명문에 풀이)다.
- 검토 메모 [26] 을 지웠다(25항목). SI 85쪽, 검토판 24·25쪽, 오류 0.

## 18. 12차 수정(2026-10-05, Fig 1 a 지역 색)

- 그림 1 a(파일 Fig1, 15:31 재렌더)에 그림 담당 문구 'colour marks the region as in b (grey, Greenland, Svalbard and Scandinavia),' 를 'in a block,' 뒤에 넣었다(영문, 국문, Fig1_legend.md). 늘어난 단어는 c, d, 투영 문장, b 의 표기 압축과 'rectangle, area of c and d' 로 맞췄다(정보 삭제 없음, 349단어).
- 다섯 문서 다시 컴파일: 영문 본문 40쪽, SI 85쪽, 국문 본문 29쪽, 검토판 24·25쪽, 오류 0. 본문 4,447단어, 설명문 최대 350단어, 검토 메모 25항목.

