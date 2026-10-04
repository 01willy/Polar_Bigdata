# ALT·영구동토 분야 상위 학술지 그림 모범 사례 분석 (2026-10-04)

목적: 원고(Scientific Reports 목표)와 발표 자료의 그림 7개(Fig 1–7)를 설계하기 전에, 상위 학술지의 영구동토·ALT·지온 논문과 평가 설계 논문에서 그림 설계 방식을 확인하고 우리 그림에 옮길 요소를 정한다.

관련 문서
- 그림 규격과 슬라이드 원칙 전체: `docs/research/2026-10-04/figure_slide_standards.md`
- Sci Rep 원고 형식: `docs/research/2026-10-04/scirep_manuscript_exemplars.md`, `scirep_format_and_drafts.md`
- 현행 그림 계획: `docs/DISPLAY_ITEMS_2026-09-30.md`, `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`
- 내려받은 PDF 색인: `references/09_figure_exemplars/README.md`

표기: [미확인] = 이번 조사에서 원문으로 확인하지 못한 항목. 그림 기술은 모두 PDF 쪽을 렌더링해 직접 본 결과이다. 캡션에 투영명이 없는 지도는 '북극 중심 방위 투영(투영명 미기재)'으로 적는다.

---

## 1. 요지

1. 분석 대상은 15편이다. 상위 학술지(Nat. Commun. 6, Nat. Clim. Change 2, Sci. Data 1, Nat. Rev. Earth Environ. 1, The Cryosphere 2, ESSD 2, GRL 1)이며, 영구동토 논문 11편과 평가 설계·물리 결합 방법 논문 4편이다. 별도로 비교 기준 4편(목표 학술지 경쟁 논문 포함)과 반면교사 사례를 정리했다.
2. 모범 사례의 공통점은 다섯 가지이다. (1) 개념도를 상자와 문장 대신 실제 좌표·단면·기하 도형 위의 직접 라벨로 그린다. (2) 범례 상자 대신 선 끝·점 무리 옆 직접 라벨을 쓴다. (3) 성능 곡선에 기준선(영 모형, 물리 모형, 외부 기준)을 함께 그려 교차점이 결론이 되게 한다. (4) 불확실성은 정의를 밝힌 오차 막대·음영 띠·측정 정확도 이내 계급으로 보인다. (5) 소형 다중 그림은 행·열 머리만으로 읽히게 하고 범례를 한 번만 둔다.
3. 우리 Fig 2(라벨 수 곡선)와 같은 문법의 상위 학술지 예는 영구동토 문헌에서 찾지 못했다. 가장 가까운 예는 Ploton et al. 2020 Fig 5(버퍼 거리에 따른 성능 곡선과 영 모형 기준선의 교차)와 Tsai et al. 2021 Fig 7b(학습 자료 비율에 따른 성능 곡선과 외부 기준 도달점)이다.
4. 우리 Fig 3(물리 정보)과 Fig 7(워크플로)의 개념 패널에는 Nitzbon et al. 2024 Fig 3(블록도 위 행, 모의 결과 아래 행, 같은 색 체계), Burke et al. 2020 Fig 1·Biskaborn et al. 2019 Fig 7(온도·깊이 좌표 위 개념도), Talucci et al. 2025 Fig 2(봄·가을 단면 삽화), Shen et al. 2023 Fig 2(함수 공간 도식)를 쓴다. 모두 상자형 글씨가 없다.
5. 우리 Fig 6(라벨 0 전이와 불확실성)에는 Langer et al. 2024 Fig 3(예측 지도 위 관측 잔차 원, 앙상블 5–95 % 오차 막대 산점도)과 Aalto et al. 2018 Fig 3(방법별 구간 폭을 모양 부호로)이 가장 가깝다.
6. 목표 학술지의 최근접 경쟁 논문(Gautam et al. 2025, Sci. Rep.)의 ALT 지도는 연속 변수에 순서가 없는 범주 색을 쓴다. Sci Rep 게재 그림의 하한이 높지 않으므로, 위 모범 사례 수준이면 그림 품질에서 분명히 구별된다.
7. Sci Rep 지침을 원문으로 다시 확인했다. 그림 설계와 직접 관련된 항목은 "avoid excessive boxing", 산세리프 단일 서체·단일 크기, 소문자 굵은 패널 문자, 최소 선 굵기 1 pt, 벡터 제출, "schemes should not be used", 표시 항목(그림과 표 합계) 8개 상한이다(8.2절). 현행 계획(그림 7 + 본문 표)은 상한에 걸린다.

---

## 2. 조사 방법

후보 출처
- 과제에 주어진 후보 16건, 기존 문헌 색인(`references/INDEX.md`)의 관련 항목, Crossref 저널명 질의(Communications Earth & Environment, Scientific Reports, Nature Communications, Earth's Future, GRL, ESSD, Scientific Data, 2019년 이후).
- DOI 와 서지는 모두 OpenAlex 단건 조회(DOI)와 Crossref 로 확인했다. 저자 수·권·쪽은 OpenAlex 값이다.

선정 기준
- 그림 유형이 우리 그림에 대응할 것: 관측·자료 지도(Fig 1), 성능 대 자료량 곡선(Fig 2, 7), 물리 개념도(Fig 3), 조건별 성능 분포(Fig 4), 평가·배치 설계 도식(Fig 5), 불확실성 지도·구간(Fig 6).
- 그림 자체를 렌더링해 확인한 뒤 모범 사례 여부를 판정했다. 내려받았으나 그림이 모범 사례가 아닌 1건은 지웠다(9.5절).

내려받기 규칙
- 합법적 공개본만 사용했다(출판사 OA, Copernicus, OSTI, EarthArXiv). 그림자 도서관은 쓰지 않았다. Unpaywall 등 이메일을 요구하는 서비스는 쓰지 않았다.
- 파일마다 `file` 과 `pdfinfo` 로 PDF 여부와 쪽수를 확인했다. 출판사 링크만 담은 1쪽짜리 안내 파일(AWI 저장소)은 거부했다.
- `references/` 아래 다른 폴더에 같은 논문이 있으면 받지 않거나 지웠다(Wei 2026, Talucci 2025 는 `08_recent_alt_2022_2026/` 사본과 바이트 단위로 같아 09 사본을 지웠다).

제약
- OpenAlex 의 키 없는 일일 예산이 같은 IP 의 다른 작업과 공유되어 검색 질의가 소진되었다. DOI 단건 조회는 계속 가능했다. 저널별 체계적 검색은 Crossref 질의로 대체했으므로 PNAS·Sci. Adv.·GRL·Earth's Future 의 최근 논문은 완전히 훑지 못했다.
- Wiley(AGU 학술지)와 PMC·Europe PMC 의 PDF 경로는 자동 내려받기를 막는다(봇 확인 페이지 또는 403). 우회하지 않았다. PMC 소장 논문 일부는 Europe PMC 공식 REST API 로 그림 이미지만 받아 배치를 확인했다(저해상도).

---

## 3. 분석 대상 일람

| # | 논문 | 학술지 | DOI | 파일 위치 | 주로 연결되는 우리 그림 |
|---|---|---|---|---|---|
| 1 | Hjort et al. 2018 | Nat. Commun. 9, 5147 | 10.1038/s41467-018-07557-4 | 09 | Fig 1, 4, 6 |
| 2 | Karjalainen et al. 2019 | Sci. Data 6, 190037 | 10.1038/sdata.2019.37 | 09 | Fig 1, 6(SI 지도) |
| 3 | Biskaborn et al. 2019 | Nat. Commun. 10, 264 | 10.1038/s41467-018-08240-4 | 07_context | Fig 1, 3, 4 |
| 4 | Aalto et al. 2018 | GRL 45, 4889–4898 | 10.1029/2018GL078007 | 12_manuscript_refs | Fig 4, 6 |
| 5 | Nitzbon et al. 2020 | Nat. Commun. 11, 2201 | 10.1038/s41467-020-15725-8 | 09 | Fig 3, 7 |
| 6 | Langer et al. 2024 | The Cryosphere 18, 363–385 | 10.5194/tc-18-363-2024 | 09 | Fig 3, 6, 7 |
| 7 | Burke et al. 2020 | The Cryosphere 14, 3155–3174 | 10.5194/tc-14-3155-2020 | 09 | Fig 3, 4 |
| 8 | Nitzbon et al. 2024 | Nat. Clim. Change 14, 573–585 | 10.1038/s41558-024-02011-4 | 09 (EarthArXiv 프리프린트) | Fig 1, 6, 7 |
| 9 | Talucci et al. 2025 | ESSD 17, 2887–2909 | 10.5194/essd-17-2887-2025 | 08_recent_alt_2022_2026 | Fig 1, 3, 4 |
| 10 | Wei et al. 2026 | ESSD Discussions | 10.5194/essd-2026-447 | 08_recent_alt_2022_2026 | Fig 6, SI 제품 비교 |
| 11 | Ploton et al. 2020 | Nat. Commun. 11, 4540 | 10.1038/s41467-020-18321-y | 09 | Fig 1c, 2, 5 |
| 12 | Meyer & Pebesma 2022 | Nat. Commun. 13, 2208 | 10.1038/s41467-022-29838-9 | 05_uq_transfer | Fig 1c, 5 |
| 13 | Tsai et al. 2021 | Nat. Commun. 12, 5988 | 10.1038/s41467-021-26107-z | 09 | Fig 2, 7 |
| 14 | Shen et al. 2023 | Nat. Rev. Earth Environ. 4, 552–567 | 10.1038/s43017-023-00450-9 | 09 (OSTI 수락 원고) | Fig 3, 발표 개념 |
| 15 | Natali et al. 2019 | Nat. Clim. Change 9, 852–857 | 10.1038/s41558-019-0592-8 | 링크만(PMC8781060) | Fig 1, 6 |

비교 기준(모범 사례 아님, 5절): Gautam et al. 2025(Sci. Rep.), Ran et al. 2022(ESSD), Obu et al. 2019(Earth-Sci. Rev.), Clayton et al. 2021(ERL).
보조 참고(기존 폴더): Linnenbrink et al. 2024, GMD 17, 5897–5912, doi:10.5194/gmd-17-5897-2024(`05_uq_transfer`), Fig 3·5 를 Fig 5 설계에 쓴다.

---

## 4. 논문별 그림 분석

각 그림은 배치, 투영·삽입도, 색, 불확실성, 글자·주석, 범례 순으로 적는다. 해당 없는 항목은 생략한다.

### 4.1 Hjort et al. 2018, Nature Communications

서지: Hjort, J., Karjalainen, O., Aalto, J., Westermann, S. et al. (8인). Degrading permafrost puts Arctic infrastructure at risk by mid-century. *Nature Communications* 9, 5147 (2018). doi:10.1038/s41467-018-07557-4

선정 이유: 관측 지점(MAGT 시추공)과 통계 모형 산출(현재·미래 영구동토 분포)을 지도 한 장에 겹치고, 범주별 결과를 불확실성 막대로 보인다. 통계 모형 앙상블(GLM·GAM·RF·GBM) 기반이라 우리 ML 지도와 맥락이 같다.

Fig 1 (영구동토 분포와 관측 지점)
- 배치: 2단 폭 단일 패널. 원 둘레에 경도 라벨(60°W, 120°W, 180°, 120°E, 60°E), 안쪽에 위도 라벨.
- 투영·삽입도: 북극 중심 방위 투영(투영명 미기재). 확대도 2개(좌상단, 우하단)를 본 지도의 사각형과 직선으로 잇고, 확대도마다 km 축척 막대(0–100 km)를 둔다.
- 색: 면 자료는 범주 4개(현재 영구동토 진청, 2041–2060 RCP4.5 연녹, 영구동토 없음 연회색, 빙하 짙은 회색). 점 자료(MAGT)는 6계급 발산형(> 5 °C 빨강에서 < −10 °C 남색), 점마다 검은 테두리.
- 범례: 지도 안 좌하단 흰 상자 1개에 면 범례와 점 범례를 두 열로 나란히.
- 주석 밀도: 낮다. 지명 없이 위경도와 축척만 둔다.

Fig 2 (기반시설 위험 비율)
- 배치: 2 × 2(a–d). 위 행은 시나리오 비교, 아래 행은 지역 비교. 왼쪽 열 '근지표 영구동토 융해', 오른쪽 열 '고위험 지역'. 패널 제목은 명사구 한 줄.
- 축: x 는 기반시설 7종(회전 라벨, 아래 행에만), y 는 비율(%) 0–100 을 네 패널이 공유. 범주를 하나 건너 회색 세로 띠로 칠해 범주 경계를 읽기 쉽게 한다.
- 불확실성: 점(추정치)과 세로 막대(범위). 막대의 근거(MAGT·ALT 예측 불확실성)를 캡션에 명시한다.
- 색: 시나리오 3개(회색 계열과 빨강), 지역 3개(주황, 남색, 하늘색). 같은 범주 안에서 x 위치를 조금씩 어긋나게 둔다.
- 범례: 패널 안 빈 영역의 작은 목록, 테두리 없음.

Fig 3 (위험 지도)
- 배치: 본 지도 1개와 모서리 확대도 4개(중앙 알래스카 2, 서시베리아·우랄 2). 확대도마다 축척 막대와 지명.
- 색: 위험 4계급 순서형(연노랑, 주황, 갈색, 빨강 'hot spot'). 기반시설은 선 종류(도로, 철도, 파이프라인)와 점 크기(정주지 인구 3계급)로 구분해 색에만 정보를 싣지 않는다.
- 범례: 좌하단 한 상자에 세 묶음(위험 계급, 기반시설 기호, 정주지 크기)을 열로 나눈다. 자료 출처·라이선스 문구를 범례 아래 작은 글씨로.

우리 그림 연결: Fig 1(관측 지점 + 영구동토 구역 지도, 확대도 연결과 축척), Fig 4·Fig 6(범주별 점·오차 막대와 회색 띠 묶음).

### 4.2 Karjalainen et al. 2019, Scientific Data

서지: Karjalainen, O., Aalto, J., Luoto, M., Westermann, S. et al. (8인). Circumpolar permafrost maps and geohazard indices for near-future infrastructure risk assessments. *Scientific Data* 6, 190037 (2019). doi:10.1038/sdata.2019.37

선정 이유: Nature 계열 자료 논문의 지도 표준이다. MAGT 지점(n = 797)과 ALT 지점(n = 303)을 모양으로 구분한 관측 지도가 우리 Fig 1 에 직접 대응한다. Fig 1 은 상자형 흐름도라 반면교사로 기록한다.

Fig 2 (관측 지점과 영구동토 범위)
- 투영: 지구본 시점(Fig 3 캡션에 'orthographic projection' 명시), 음영 기복을 바탕에 깔고 30°N 까지 보인다.
- 색: 현재(연청), 2041–2060(청), 2061–2080(보라) 영구동토 범위를 겹쳐 칠해, 미래로 갈수록 남는 범위가 줄어드는 것을 한 장에 보인다(중첩 범위 표현).
- 점: MAGT 지점은 검은 삼각형, ALT 지점은 흰 원. 색이 아니라 모양과 명도로 자료 종류를 구분한다.
- 범례: 좌하단, 테두리 없이 지도 위에 직접.

Fig 3 (지수별·기간별 위험 지도)
- 배치: 4행 × 2열 소형 다중 지도(a–h). 행 = 지수 4종(왼쪽 행 머리 Is, Ir, Ia, Ic), 열 = 기간 2개(열 머리 'RCP 4.5 2041–2060', 'RCP 4.5 2061–2080').
- 색: 3계급 순서형 보라 단색조와 빙하 회색. 범례 1개를 오른쪽 아래에 두고 8패널이 공유한다.
- 주석: 패널 안 글자 없음. 행·열 머리만으로 읽힌다.

Fig 1 (반면교사): 상자 6개에 글머리 문장을 채운 흐름도이다. 사용자가 지적한 '상자형 글씨 설명 그림'과 같은 유형이다.

우리 그림 연결: Fig 1(자료 종류를 모양으로 구분), Fig 6 SI 지도판(행·열 머리만으로 읽히는 소형 다중 지도).

### 4.3 Biskaborn et al. 2019, Nature Communications (기존 `07_context`)

서지: Biskaborn, B. K., Smith, S. L., Noetzli, J., Matthes, H. et al. (48인). Permafrost is warming at a global scale. *Nature Communications* 10, 264 (2019). doi:10.1038/s41467-018-08240-4

선정 이유: 점 자료 지도의 겹침 처리, 측정 정확도 이내 변화를 별도 계급으로 둔 색 설계, 개념도를 좌표 위에 그린 방식이 모범적이다.

Fig 1 (장기 기록)
- (a) 위쪽 좁은 띠 지도에 시추공 ID 를 직접 적는다. (b) 시계열 선 옆에 ID 를 직접 적어 범례를 없앤다.
- y 축은 끊김 표시로 −15 °C 아래 구간을 압축한다.

Fig 2 (온도와 변화율)
- 배치: 북극 지도 2개(a, c)를 위에 크게, 남극 지도 2개(b, d)를 아래 모서리에 작게, 그 사이 가운데에 공유 범례 블록.
- 기호: 현재 온도(a, b)는 사각형, 10년 변화율(c, d)은 원. 변수마다 모양을 달리해 색표가 둘이어도 혼동하지 않게 한다.
- 색: 변화율은 냉각(파랑), 측정 정확도 ±0.1 °C 이내(초록), 온난화(노랑에서 빨강). '변화 없음'을 정확도 기준의 별도 계급으로 둔다.
- 겹침 처리: 시추공이 밀집한 곳은 지시선으로 점을 바깥으로 부채꼴 배치한다.
- 바탕: 연속(연청)·불연속(연보라) 영구동토 구역을 옅게 칠해 점이 앞에 보이게 한다.

Fig 7 (열 체제 개념도)
- 온도(x)·깊이(y) 좌표 위에 겨울 최소(파랑)·여름 최대(빨강) 곡선, 연평균 점선, 0 °C 세로선, 영년변화 깊이 Z* 의 원 표시. 활동층·영구동토·비동결 구간을 옅은 색 띠로 칠하고 이름을 띠 안에 직접 쓴다. 상자형 글씨가 없다.

우리 그림 연결: Fig 1(밀집 지점 처리, 구역 바탕), Fig 4(오차 하한 안의 차이를 중립색 계급으로 분리), Fig 3(물리 개념을 좌표 위 도식으로).

### 4.4 Aalto et al. 2018, Geophysical Research Letters (기존 `12_manuscript_refs`)

서지: Aalto, J., Karjalainen, O., Hjort, J., Luoto, M. Statistical forecasting of current and future circum-Arctic ground temperatures and active layer thickness. *Geophysical Research Letters* 45, 4889–4898 (2018). doi:10.1029/2018GL078007

선정 이유: GLM·GAM·GBM·RF 와 앙상블의 MAGT·ALT 불확실성을 흑백에서도 읽히게 비교한다. 방법 비교 그림의 간결한 기준이다.

Fig 2 (관측 대 예측)
- 산점도 2개(MAGT, ALT), 반투명 점, 1:1 점선.
- 검증 기간 2개를 색으로 나누고, 기간별 통계(Bias, RMSE, R², n)를 같은 색 글자로 해당 점 무리 근처에 둔다. 범례 상자가 없다.

Fig 3 (예측 불확실성)
- 배치: 2 × 2(MAGT·ALT × PI50·PI95).
- 축: x 는 기준기와 시나리오·기간(기간은 x 축 아래 2단 라벨로 묶음), y 는 불확실성(±°C, ±cm).
- 부호: 방법 5종을 표지 모양(빈 삼각형, 빈 마름모, ×, +, 회색 사각형)으로만 구분한다.
- 범례: 패널 안 좌상단, 테두리 없음.

우리 그림 연결: Fig 6(예측 구간 폭을 방법·조건별로 요약), Fig 4(학습기 비교를 모양 부호로).

### 4.5 Nitzbon et al. 2020, Nature Communications

서지: Nitzbon, J., Westermann, S., Langer, M., Martin, L. C. P. et al. (7인). Fast response of cold ice-rich permafrost in northeast Siberia to a warming climate. *Nature Communications* 11, 2201 (2020). doi:10.1038/s41467-020-15725-8

선정 이유: 물리 모형 결과를 개념 단면도, 소형 다중 그림, 모형 설정도로 나누어 보인다. 우리 Fig 3(물리 정보의 사용)에 대응하며 세 그림 모두 상자형 글씨가 없다.

Fig 1 (지형 단면과 지역 지도)
- (b) 지형 단면 블록도(열카르스트 호수, Yedoma, 배수 호수 분지, 홀로세 퇴적층)를 크게 두고 (a) 지역 지도를 블록도 왼쪽 아래 모서리에 겹쳐 넣는다.
- 단위 이름을 단면 위에 직접 쓰되, 분석에 넣은 단위는 파란 글자, 뺀 단위('not considered')는 회색 글자로 구분한다. 분석 범위를 글자 색 하나로 표시한다.

Fig 3 (모의 열화와 포화)
- 배치: 4행 × 3열 = 12패널(a–l). 열 = 지형 유형 3종(열 머리), 행 = 배수 조건 2 × 시나리오 2(왼쪽 2단 행 머리 'Well-drained/Water-logged', 'RCP4.5/RCP8.5').
- 축: x 2000–2100 공유. 마지막 행만 깊이 범위가 커서(0 에서 −10 m) 패널 높이를 키워, 깊이 1 m 당 길이를 행 사이에 같게 맞춘다. 눈으로 깊이를 비교할 수 있다.
- 표현: 지표(실선)와 최대 융해 깊이(점선) 사이를 불포화(빨강), 포화(남색), 수체(하늘색) 면으로 채운다.
- 주석: 패널 j 하나에만 화살표 주석(초기 활동층, 침하, 영구동토 열화, 최종 활동층)을 달아 '읽는 법'을 보인다. 나머지 11개는 주석이 없다.
- 범례: 패널 a 안에 1회.

Fig 5 (모형 설정도)
- 타일 4개(중심, 테두리, 골, 저수조)를 높이가 다른 기둥으로 그리고, 타일 사이 교환(눈, 퇴적물, 물, 열)을 색 화살표로 표시한다. 높이 차(e_R, e_T, e_res)는 치수선으로 적는다. 문장 없이 기호와 치수로 설명한다.

우리 그림 연결: Fig 3(개념 단면과 결과 소형 다중 그림), Fig 7(워크플로를 상자 대신 '읽는 법' 주석 하나로).

### 4.6 Langer et al. 2024, The Cryosphere

서지: Langer, M., Nitzbon, J., Groenke, B., Assmann, L.-M. et al. (7인). The evolution of Arctic permafrost over the last 3 centuries from ensemble simulations with the CryoGridLite permafrost model. *The Cryosphere* 18, 363–385 (2024). doi:10.5194/tc-18-363-2024

선정 이유: CALM ALT 관측과 모형 앙상블을 지도·산점도로 대조하면서 앙상블 5–95 % 범위를 오차 막대로 보인다. 우리 Fig 6 에 가장 가깝다.

Fig 1 (지층 개념도)
- 토양 기둥 하나에 구성 성분(물·얼음, 유기물, 광물, 공기)을 색 면으로, 층 경계를 수평선으로 그린다.
- 앙상블에서 무작위로 바꾼 경계(z_R, z_V, z_S, h_snow,max 등)만 자홍색으로 칠한다. '무엇을 바꾸었는가'를 색 하나로 표시한다.

Fig 3 (ALT 지도와 CALM 비교)
- 배치: 왼쪽 큰 지도(a), 오른쪽 위아래 산점도(b: ALT 평균, c: ALT 추세). 지도와 산점도 사이에 세로 색 막대 2개를 나란히.
- 색: 바탕의 모의 ALT 는 순차형(viridis 계열, 0–10 m). CALM 지점 원은 모형 − 관측 차이를 0 중심 발산형(자홍, 흰색, 녹색; −2 에서 2 m)으로. 같은 지도에서 연속 변수와 부호 있는 차이를 서로 다른 색표로 분리한다.
- 불확실성: y 오차 막대 = 앙상블 5–95 백분위, x 오차 막대 = 같은 격자 안 여러 관측의 범위. 1:1 실선, ±20 % 점선, 비교가 의미 있는 범위(ALT < 2 m)를 회색 사각형으로 표시한다. 통계(n, RMedSE, r, bias)는 패널 안 좌상단 글자.
- 투영: 북극 중심 극 투영(투영명 미기재), 육지 회색, 바다 흰색.

Fig 5 (ALT 상대 변화)
- 3패널 가로 배열. 가운데(b)는 기준기 절대값(순차형), 양옆(a, c)은 상대 변화(0 중심 RdBu 계열, −1 에서 1, 두 패널 같은 범위). 패널마다 세로 색 막대.
- 기준 상태를 가운데 두고 변화를 양옆에 두는 배치이다.

우리 그림 연결: Fig 6(예측 지도 위 관측 잔차 원, 커버리지 산점도), Fig 3(보정하는 계수만 강조색), Fig 7 분해 패널(두 차이 지도에 같은 발산 범위).

### 4.7 Burke et al. 2020, The Cryosphere

서지: Burke, E. J., Zhang, Y., Krinner, G. Evaluating permafrost physics in the Coupled Model Intercomparison Project 6 (CMIP6) models and their sensitivity to climate change. *The Cryosphere* 14, 3155–3174 (2020). doi:10.5194/tc-14-3155-2020

선정 이유: 물리 개념(지표 offset, 열 offset, ALT, Dzaa)을 실제 온도·깊이 축 위에 그린 개념도와, 기후 구간별로 모형·관측 분포를 나란히 둔 소형 다중 상자그림이 있다.

Fig 1 (연 온도 단면 개념도)
- 온도(°C, x)·깊이(y) 축 위에 연 최대(굵은 빨강), 최소(가는 빨강), 평균(빨강 점선) 곡선.
- ALT 구간(초록 띠), 영구동토 바닥(연보라 띠), Dzaa(검은 점선)를 해당 위치에 직접 적는다. 범례는 위쪽 작은 상자 하나.

Fig 3 (모형별 기후 편차)
- 막대 패널 5개를 위아래로 쌓고 x 축(모형 18개, 회전 라벨)을 맨 아래 한 번만 둔다.
- 관측보다 크면 빨강, 작으면 파랑으로 부호를 색과 방향에 이중으로 싣는다. 관측 자료 간 차이는 초록 수평선.

Fig 9 (MAAT 구간별 ALT)
- 배치: 5행 × 4열 소형 다중 그림(모형별 1칸). x = 지역 MAAT 구간(−14, −10, −6, −2 °C), y = 최대 융해 깊이(m) 공유.
- 모형(빨강)과 CALM 관측(파랑) 상자그림을 같은 구간에 나란히. 칸마다 일치 비율(%)을 좌상단 숫자로.
- 약점: 범례를 18칸 모두에 반복한다. 공유 범례 하나로 충분하다.

우리 그림 연결: Fig 3(Stefan 융해 전선 개념을 좌표 위 도식으로), Fig 4(기후·계수 오차 구간별로 ML 이득 분포를 나란히).

### 4.8 Nitzbon et al. 2024, Nature Climate Change (EarthArXiv 프리프린트)

서지: Nitzbon, J., Schneider von Deimling, T., Aliyeva, M., Chadburn, S. E. et al. (13인). No respite from permafrost-thaw impacts in the absence of a global tipping point. *Nature Climate Change* 14, 573–585 (2024). doi:10.1038/s41558-024-02011-4. 분석 파일은 EarthArXiv 프리프린트 v1(doi:10.31223/x55x08, CC BY-NC-SA)이다. 게재본 그림과 다를 수 있다 [미확인].

선정 이유: 상자 없는 개념도(3차원 블록도)와 모의 결과를 위아래로 짝지어 같은 색 체계로 잇는 구성이 이번 대상 중 가장 완성도가 높다.

Fig 1 (평형 영구동토 면적과 근거 관계)
- (a) 평형 영구동토 면적 대 ΔGMST 곡선(굵은 선)과 그럴듯한 범위(회색 음영). (b) 근거가 되는 영구동토 비율·MAAT 관계를 (a) 안 오른쪽 위 삽입 그림으로. (c) 탄소량은 같은 x 축을 공유하는 아래 패널.
- 결과 곡선과 그 근거 관계를 한 그림 안에 겹쳐 둔다.

Fig 2 (하위 지역 지도)
- 8계급을 '색상 4개(지형·생물군계) × 명도 2단계(얼음 함량)'로 만들고 범례를 4열 × 2행 행렬로 배치한다. 축척 막대(0–2,000 km).
- 약점: 면적 비율을 원그래프로 둔다. 막대가 낫다.

Fig 3 (국지 과정과 모의 예)
- 배치: 위 행 블록도 3개(a–c: 탈릭 발달, 호수 열카르스트, 열침식), 아래 행 깊이·시간 패널 4개(d–g).
- 블록도: 라벨을 지표·단면 위에 곡선을 따라 직접 쓴다('active layer', 'permafrost', 'talik', 'time / warming'). 블록도 제목 옆 색 사각형이 Fig 2 지도의 하위 지역 색과 대응한다(그림 사이 색 열쇠).
- 결과 패널: 깊이 0–15 m, 1900–2100, 블록도와 같은 흙색 계열로 활동층·탈릭·호수를 칠한다. 범례는 패널 d 안에 1회.

우리 그림 연결: Fig 7(워크플로와 분해를 '개념 위, 실제 결과 아래' 짝으로), Fig 1(지역 구분 범례를 행렬로), Fig 6(중심선, 범위 음영, 근거 관계 삽입).

### 4.9 Talucci et al. 2025, Earth System Science Data (기존 `08_recent_alt_2022_2026`)

서지: Talucci, A. C., Loranty, M. M., Holloway, J., Rogers, B. M. et al. (39인). Permafrost–wildfire interactions: active layer thickness estimates for paired burned and unburned sites in northern high latitudes. *Earth System Science Data* 17, 2887–2909 (2025). doi:10.5194/essd-17-2887-2025

선정 이유: 이른 시기 융해 깊이를 √TDD 비율로 계절 말 ALT 로 환산한다(본문 식 1–3: A = 측정일까지 TDD 합의 제곱근, B = 해빙 계절 말까지 TDD 합의 제곱근, 추정 ALT = 측정 깊이 × B/A, Stefan 식 인용). 내용이 우리 물리 기준선과 직접 겹치고 측정 개념도와 잔차 그림이 단순하다.

Fig 1 (측정 분포 지도)
- 생태구역별 측정 비율을 원 크기와 원 안 숫자(%)로 표시한 비례 기호 지도. 수만 개 측정점을 그리지 않고 구역 단위 원 11개로 요약한다. 북극권은 굵은 점선.
- 약점: 생태구역 11개를 서로 다른 색상으로 칠해 범주 색이 많다.

Fig 2 (측정 개념)
- 봄(측정 융해 깊이)과 가을(추정 ALT) 토양 단면 삽화 2개를 나란히 두고, 측정 막대와 'Frozen active layer', 'Thawed active layer', 'Permafrost table' 라벨을 지시선으로 붙인다. 문장 상자가 없다.

Fig 6 (추정 불확실성)
- (a, b) 관측 대 추정 산점도(빈 원, 1:1 주황선).
- (c) 지점별 잔차(cm): 흩뿌린 점 + 상자그림 + 평균 마름모, 0 근처 회색 띠, 오른쪽 가장자리에 'Overestimate'(위 화살표)와 'Underestimate'(아래 화살표)를 옅은 회색으로 둔다. 부호 해석을 축 옆에 붙인다.

우리 그림 연결: Fig 3(√TDD 환산 개념의 단면 삽화), Fig 4(대상별 개선·악화를 0 기준 띠와 방향 화살표로), Fig 1(지역별 라벨 수를 비례 원과 숫자로).

### 4.10 Wei et al. 2026, ESSD Discussions (기존 `08_recent_alt_2022_2026`)

서지: Wei, Y., Wu, Z., Wang, J., Shen, H., Chen, Y. A 1 km resolution dataset of Northern Hemisphere permafrost active layer thickness (2000–2024). *Earth System Science Data Discussions* (토론 시작 2026-08-12). doi:10.5194/essd-2026-447

선정 이유: 가장 최근의 반구 ALT 제품 논문이다(연간 ALT 관측 2,196건, 지점 단위 leave-one-site-out 검증). 기존 1 km 제품과의 지역별 비교와 연도별 불확실성 지도를 담는다. 우리 그림이 넘어야 할 현행 수준이며 설계 일부는 반면교사이다.

Fig 1 (연구 지역): (a) 정거원통 띠 지도 위, (b, c) 극 투영 지도 2개 아래. 범례 3묶음을 그림 아래에 둔다. ALT 지점은 빨간 점.

Fig 3 (시간 성능): (a) 관측 대 예측 Sen 기울기 산점도, 1:1 점선, 좌상단 통계 글자. (b) 변동폭 비율 히스토그램과 중앙값 점선·라벨.

Fig 7 (기존 제품 비교)
- 배치: 3열(본 연구, Westermann et al. 2024, Peng et al. 2024) × 7행(반구와 6개 지역). 맨 위 행에 빈도 막대 삽입. 지역 평균을 패널 안 숫자로.
- 색: 공유 이산 색 막대(ALT, cm, 60–300 비균등 구간). 약점: Spectral 계열(무지개형) 색표.

Fig 9 (불확실성)
- 배치: 2 × 3 연도별 상대 불확실성 지도와 (g) 연도별 불확실성 계급 면적 비율 100 % 누적 막대. 지도와 요약 막대가 같은 계급색을 쓴다(이 연결은 좋다).
- 약점: 음이 아닌 불확실성에 파랑·빨강 발산형 색표를 쓴다. 패널 안 'Mean: 45.20' 에 단위가 없다.

우리 그림 연결: Fig 6(불확실성 지도와 계급 면적 요약의 연결), SI 제품 비교(열 = 제품, 행 = 지역), Fig 1(지점 지도 구성 참고, 범례 묶음 수는 줄일 것).

### 4.11 Ploton et al. 2020, Nature Communications

서지: Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N. et al. (13인). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11, 4540 (2020). doi:10.1038/s41467-020-18321-y

선정 이유: 무작위 분할과 공간 분할의 차이를 '거리에 따른 성능 곡선과 영 모형 기준선'으로 보인다. 우리 Fig 2(라벨 수 곡선과 물리 기준선의 교차)와 같은 문법이다. 평가 설계 도식도 상자를 최소화한다.

Fig 1 (연구 지역): 지도 1개, 순차형 자홍·노랑(magma 계열) 색표. 색 막대는 지도 오른쪽 바깥에 세로로 두고 단위(Mg·ha⁻¹)와 열린 끝값(≥ 500, < 100)을 쓴다. 아프리카 위치 삽입도를 오른쪽 아래에. 축척 막대 600 km.

Fig 3 (공간 구조 도식)
- (a) 공간 fold 44개 지도 → (b) 점선 사각형 영역 확대 → (c) 다시 확대한 격자 위에 반경 r1, r2 버퍼 원.
- 세 단계 확대를 점선 테두리로 잇는다. 글자는 r1, r2, 범례 1항목, 축척 막대뿐이다.

Fig 5 (공간 구조가 CV 통계에 주는 영향)
- (a) R² 대 버퍼 반경(km), (b) RMSPE 대 버퍼 반경. 점 = 10회 반복 평균, 세로 막대 = ±SD, 굵은 반투명 선 = 평활 적합.
- 기준선: (a) 회색 점선 = 공간 K-fold 결과, (b) 검은 선 = 학습 평균을 내는 영 모형. ML 곡선이 반경 약 100 km 부근에서 영 모형과 만난다. 교차점이 곧 결론이다.
- (c) 지도: 각 격자에서 가장 가까운 관측까지의 거리를 (a)의 관계로 R² 로 바꾸어 칠한다(hot 계열, 0–0.5). 성능 곡선을 공간 지도로 옮긴다.

Fig 6 (교차검증 작업 흐름): 세 전략(무작위 K-fold, 공간 K-fold, 버퍼 LOO)을 실제 지도 축소판과 표 모양 도식으로 보인다. 회색 머리 상자 3개는 남아 있다(부분적 상자형).

우리 그림 연결: Fig 2(곡선과 기준선의 교차), Fig 5(거리 기반 배치 설명, 거리에서 기대 성능으로 바꾼 지도), Fig 1c(평가 설계를 지리 확대 도식으로).

### 4.12 Meyer & Pebesma 2022, Nature Communications (기존 `05_uq_transfer`)

서지: Meyer, H., Pebesma, E. Machine learning-based global maps of ecological variables and the challenge of assessing them. *Nature Communications* 13, 2208 (2022). doi:10.1038/s41467-022-29838-9

선정 이유: 관측 군집도를 '표본 간 최근접 거리 분포 대 표본·예측 위치 거리 분포'로 보인다. 우리 평가 설계(지역 홀드아웃이 실제 예측 거리와 맞는가)를 그림 하나로 보이는 틀이 된다.

Fig 1 (공간 거리 분포)
- 배치: 4행 짝 구성. 왼쪽 = Equal Earth 투영 세계 지도(표본 분홍, 예측 영역 청록), 오른쪽 = 최근접 거리 밀도(로그 x, 1–10,000 km).
- 행 이름은 오른쪽 세로 띠 글자. 4행째는 완전 공간 무작위 표본으로, 두 분포가 겹치는 기준 행이다.
- 범례는 맨 아래 하나를 공유한다.

우리 그림 연결: Fig 1c 또는 SI(지역별 라벨·예측 위치 거리 분포, 무작위 분할 대조 행), Fig 5(배치 전략별 거리 분포).

### 4.13 Tsai et al. 2021, Nature Communications

서지: Tsai, W.-P., Feng, D., Pan, M., Beck, H. E. et al. (8인). From calibration to parameter learning: harnessing the scaling effects of big data in geoscientific modeling. *Nature Communications* 12, 5988 (2021). doi:10.1038/s41467-021-26107-z

선정 이유: 물리 모형과 학습의 결합(미분 가능 매개변수 학습) 성능을 학습 자료량에 대해 그린 규모 곡선이 있다. 영구동토 문헌에 라벨 수 곡선의 상위 학술지 예가 드물어 대체 참고로 둔다.

Fig 2 (실행 수·계산 시간에 따른 RMSE)
- 배치: 2 × 2. (a, b) RMSE 대 격자당 실행 수, (c, d) RMSE 대 계산 시간(로그 x).
- 실선(dPL)과 점선·표지(SCE-UA), 표본 밀도 3단계를 색으로. 기능 문턱(RMSE 0.05)을 점선 수평선으로.

Fig 7b (규모 곡선)
- x = 학습에 쓴 유역 비율(%), y = 중앙 KGE. 빨간 곡선.
- 외부 기준(Beck et al. 2020) 값을 점선 수평선과 별표로 표시해 '기준에 도달하는 자료 비율'을 그림에서 바로 읽게 한다.

반면교사: Fig 1 은 색 평행사변형·상자 흐름도이다. Fig 7a 는 y 축 4개(왼쪽 1, 오른쪽 3)를 겹쳐 읽기 어렵다.

우리 그림 연결: Fig 2·Fig 7(라벨 수 곡선에 기준선과 도달점 표시). 다축 구성은 쓰지 않는다.

### 4.14 Shen et al. 2023, Nature Reviews Earth & Environment (OSTI 수락 원고)

서지: Shen, C., Appling, A., Gentine, P., Bandai, T. et al. (31인). Differentiable modelling to unify machine learning and physical models for geosciences. *Nature Reviews Earth & Environment* 4, 552–567 (2023). doi:10.1038/s43017-023-00450-9. 분석 파일은 OSTI 수락 원고이다. 게재본 그림은 출판사 재작도본일 수 있다 [미확인].

선정 이유: '물리 사전 구조가 ML 탐색 공간을 좁힌다'는 개념을 상자 없이 함수 공간 도식으로 보인다. 우리 주제(물리 증강 잔차 ML)의 개념 패널 후보이다.

Fig 2 (함수 공간 도식)
- 비용 함수 지형을 옅은 파랑 등치 배경으로 깔고, 'Machine Learning'(바깥), 'Differentiable Geosciences'(중간), 'Process-Based'(안쪽) 닫힌 곡선을 겹친다.
- 경로 A(ML 에 구조 제약)와 B(물리 모형에 학습 단위 추가)를 화살표 2개로, 최적점을 ×로 표시한다.
- 글자는 영역 이름 3개, 화살표 이름 2개, 'optimal', 'searchable function space' 뿐이다.

반면교사: Fig 1 은 파란 상자 중심의 비교 도식이다.

우리 그림 연결: Fig 3(물리 기준선과 잔차 ML 의 위치를 함수 공간 도식으로), 발표 자료의 개념 슬라이드.

### 4.15 Natali et al. 2019, Nature Climate Change (링크만 기록)

서지: Natali, S. M., Watts, J. D., Rogers, B. M., Potter, S. et al. (75인). Large loss of CO2 in winter observed across the northern permafrost region. *Nature Climate Change* 9, 852–857 (2019). doi:10.1038/s41558-019-0592-8. PDF 자동 내려받기는 출판사와 PMC 모두 막혔다. Europe PMC 공식 API 로 저자 원고(NIHMS1539129, PMC8781060)의 그림 이미지만 확인했다. 게재본 그림과 다를 수 있다 [미확인].

선정 이유: 관측 지점 지도와 층별 분포(바이올린과 점)를 한 그림에 둔 구성, 기준기 큰 지도와 미래 소형 지도 4개의 구성이 우리 Fig 1·Fig 6 에 맞는다.

Fig 1 (관측 분포)
- (a) 영구동토 구역 4단계 파랑 바탕 위 관측 지점(노란 원), (b) 생물군계별, (c) 영구동토 구역별 겨울 플럭스 바이올린과 내부 점 분포. (b)·(c)가 y 축 이름을 공유한다.

Fig 3 (겨울 CO2 플럭스 지도)
- (a) 2003–2017 기준 큰 지도, (b, c) RCP4.5·RCP8.5 × 두 기간 소형 지도 2 × 2. 10계급 범례 하나를 (a) 옆에 두고 모두 공유한다.
- 약점: 음이 아닌 플럭스에 파랑·빨강 발산형 색을 써서 가운데 밝은 색이 의미 있는 중심처럼 보인다.

우리 그림 연결: Fig 1(지도와 지역별 라벨 분포), Fig 6 지도판(기준 큰 지도와 조건별 소형 지도).

---

## 5. 비교 기준 (모범 사례 아님)

### 5.1 Gautam et al. 2025, Scientific Reports (기존 `01_benchmark`)

서지: Gautam, S., Mishra, U., Scott, S., Lara, M. J. Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska. *Scientific Reports* 15, 42420 (2025). doi:10.1038/s41598-025-26586-w

- 목표 학술지의 최근접 경쟁 논문이다.
- Fig 2: 알래스카 ALT 지도 2개(RF, Stefan)를 위아래로 두고 세로 색 막대 하나를 공유한다. 연속 변수(ALT, cm)를 비균등 구간(25, 40, 50, 60, 70, 80, 90, 105, 140)과 서로 순서가 없는 색(검정, 주황, 하늘, 초록, 노랑, 남색, 주황, 분홍)으로 칠해 크기 순서가 색에서 읽히지 않는다. 위경도 라벨과 축척 막대가 보이지 않는다.
- 판단: Sci Rep 게재 그림의 하한을 보여준다. 우리 그림은 이 수준을 분명히 넘어야 한다.

### 5.2 Ran et al. 2022, ESSD (기존 `01_benchmark`)

서지: Ran, Y., Li, X., Cheng, G., Che, J. et al. (13인). New high-resolution estimates of the permafrost thermal state and hydrothermal conditions over the Northern Hemisphere. *Earth System Science Data* 14, 865–884 (2022). doi:10.5194/essd-14-865-2022

- Fig 4: 반구 ALT 지도, 이산 9계급(남색에서 녹·황·주황·갈색까지 다색상), 호수·빙하 범례를 지도 안 좌하단에. 위경도와 축척 막대를 갖춘다. 정돈되어 있으나 색표가 무지개형에 가깝다.

### 5.3 Obu et al. 2019, Earth-Science Reviews (기존 `02_alt_dl_mapping`)

서지: Obu, J., Westermann, S., Bartsch, A. et al. Northern Hemisphere permafrost map based on TTOP modelling for 2000–2016 at 1 km² scale. *Earth-Science Reviews* 193, 299–316 (2019). doi:10.1016/j.earscirev.2019.04.023

- Fig 6: 200회 앙상블 MAGT 표준편차 지도, 초록·노랑·빨강 신호등형 색표, 범례 'High: 5 / Low: 0'.
- Fig 2: 관측 대 모형 산점도에 ±2 °C 점선은 좋으나 통계 글자의 자릿수가 많고('RMSE =1.9868'), 음수 값에 'Mean absolute error' 라벨이 붙어 있다(평균 오차로 보인다).

### 5.4 Clayton et al. 2021, Environmental Research Letters (09 폴더, 수락 원고)

서지: Clayton, L. K., Schaefer, K. M., Battaglia, M. J. et al. (23인). Active layer thickness as a function of soil water content. *Environmental Research Letters* 16, 055028 (2021). doi:10.1088/1748-9326/abfa4c

- 내용: 알래스카 현장 자료로 ALT 와 체적 함수율의 관계(잠열 가설과 열전도 가설)를 검토한다. Stefan 형 해석과 직접 관련되므로 본문 인용 후보이다.
- 그림: Fig 2b 의 2차원 밀도에 jet 계열 색표를 쓰고, Fig 5 범례를 색 채운 이름 상자로 둔다. 회귀식과 R² 를 패널마다 글자로 반복한다. 그림 설계는 따르지 않는다.

---

## 6. 반면교사 요약

| 사례 | 문제 | 대안(출처) |
|---|---|---|
| Karjalainen 2019 Fig 1 | 상자 6개에 문장을 채운 흐름도 | 실제 자료 축소판을 늘어놓는 흐름(Ploton Fig 6), 개념 위·결과 아래 짝(Nitzbon 2024 Fig 3) |
| Tsai 2021 Fig 1 | 색 평행사변형·상자 흐름도 | 함수 공간 도식(Shen Fig 2), 좌표 위 개념도(Burke Fig 1) |
| Tsai 2021 Fig 7a | y 축 4개 겹침 | 지표마다 패널 분리, 공유 x 축 |
| Shen 2023 Fig 1 | 상자 중심 비교 도식 | 같은 논문 Fig 2 방식 |
| Burke 2020 Fig 9 | 18칸 모두 범례 반복 | 공유 범례 1개 |
| Wei 2026 Fig 9 | 음이 아닌 불확실성에 발산형, 평균에 단위 없음 | 순차형 단일 색조, 'Mean 45.2 %' 처럼 단위 포함 |
| Wei 2026 Fig 7, Ran 2022 Fig 4 | 무지개형 색표 | 지각 균일 순차형(viridis, batlow 등) |
| Natali 2019 Fig 3 | 음이 아닌 양에 발산형 | 순차형 |
| Obu 2019 Fig 6 | 신호등형 색표, 'High/Low' 범례 | 순차형, 수치 눈금과 단위 |
| Obu 2019 Fig 2 | 통계 자릿수 과다, 라벨 오류 | 유효 숫자 2–3자리, 지표명 정확히 |
| Gautam 2025 Fig 2 | 연속 변수에 순서 없는 범주 색, 축척·좌표 없음 | 순차형 이산 계급, 축척 막대·좌표 |
| Clayton 2021 Fig 2, 5 | jet 색표, 이름 상자 범례 | 순차형, 선 끝 직접 라벨 |
| Talucci 2025 Fig 1 | 범주 색 11개 | 범주 수 6개 이하 또는 모양·명도 병용 |
| Nitzbon 2024 Fig 2 | 원그래프 | 가로 막대 |
| Mishra 2021 Sci. Adv.(검토 후 제외) | 무지개형 지도, 이중 y 축 막대 | 순차형, 축 하나 |
| Hugelius 2020 PNAS(검토 후 제외) | 원판 지도가 서로 겹치는 배치, 고채도 다색상 | 겹치지 않는 격자 배치, 순차형 |

---

## 7. 우리 그림별 설계 권고

현행 그림 계획(DISPLAY_ITEMS, RESTRUCTURE_PLAN)의 메시지를 바꾸지 않고, 표현 방식만 제안한다.

### Fig 1 문제 정의와 자료 범위

메시지: 새 지역은 라벨이 적고 Stefan 계수 E 가 지역마다 다르다. 평가는 같은 시험지 위에서 세 설계로 한다.

- a 지도: 북극 중심 방위 투영(투영명을 캡션에 적는다). 영구동토 구역을 옅은 2단계 바탕으로(Biskaborn Fig 2). 지역별 라벨 수를 비례 원과 원 안 숫자로(Talucci Fig 1). 자료원 구분은 모양으로(Karjalainen Fig 2: 예를 들어 CALM 은 원, 기타 탐침·시추공은 삼각형). 밀집 지역은 확대도 1–2개와 km 축척 막대(Hjort Fig 1). 지역 색은 6개 이하로 하고 Fig 2–7 에서 같은 지역 색을 유지한다(Nitzbon 2024 의 그림 간 색 열쇠).
- b E 분포: 지역별 바이올린과 내부 점(Natali Fig 1b, c), 또는 지역을 E 중앙값 순으로 정렬한 점·구간 그림. y 축 이름 하나를 공유한다.
- c 평가 설계: 지역 1–2개에 대해 라벨 간·라벨과 예측 위치 간 최근접 거리 분포를 무작위 분할 대조와 함께 그린다(Meyer Fig 1). 또는 무작위 분할, 지역 홀드아웃, 지역 내 블록 홀드아웃을 지도 확대 도식으로 보인다(Ploton Fig 3). 상자 흐름도는 쓰지 않는다.
- 피할 것: 상자형 흐름도(Karjalainen Fig 1), 그림 아래 범례 3묶음(Wei Fig 1), 범주 색 7개 이상.

### Fig 2 라벨 수 곡선 Δ(n)

- 축: x = 라벨 수 n(0, 1, 3, 10, 40, 160 의 순서 눈금 또는 로그 축, 0 은 축 끊김으로 분리), y = 물리 기준 대비 ΔRMSE(cm).
- 기준선: y = 0 수평선이 물리 기준선 P0 이다(Ploton Fig 5b 의 영 모형 선 역할). 두 번째 기준 P* 는 회색 점선.
- 곡선과 구간: 방법별 중앙값 선, 블록 부트스트랩 구간은 옅은 음영. 구간 정의는 범례 문장에 적는다.
- 교차점 n*: 곡선 위 작은 세로 눈금과 숫자로 직접 표시한다(Tsai Fig 7b 의 기준 도달 표시).
- 라벨: 곡선 이름은 선 오른쪽 끝 직접 라벨(Biskaborn Fig 1b). 범례 상자를 두지 않는다.
- 두 판(E1, E2): y 축을 공유하는 좌우 패널(Hjort Fig 2 의 공유 축).
- 곡선 수: 본문 4–5개, 나머지는 SI 짝 그림. DISPLAY_ITEMS 의 '범례 8항목' 상한은 직접 라벨 방식에서는 많다.
- 피할 것: 다축(Tsai Fig 7a), 표지와 선 종류를 동시에 많이 쓰는 구성.

### Fig 3 물리 정보의 사용

- a 개념: 깊이·시간 좌표 위에 √TDD 융해 전선 곡선을 그리고 관측 ALT 점, 계수 E 의 차이를 같은 축 위에 보인다(Burke Fig 1, Biskaborn Fig 7 의 문법). 측정 개념이 필요하면 봄·가을 단면 삽화(Talucci Fig 2). ML 이 보정하는 대상(계수 E, 잔차)만 강조색 하나로 칠한다(Langer Fig 1 의 자홍 방식).
- b–d 결과: 기전(C1)과 구조(C3)를 소형 다중 그림으로 두고 첫 패널 하나에만 '읽는 법' 화살표 주석을 단다(Nitzbon 2020 Fig 3j). 축 범위가 다른 행은 패널 높이로 축척을 맞춘다(같은 그림).
- 발표 자료: 함수 공간 도식(Shen Fig 2)으로 '물리 기준선 + 잔차 ML' 의 위치를 설명한다. 원고에는 넣지 않는다(표시 항목 상한).
- 피할 것: 'Physics → ML → Output' 상자 흐름(Tsai Fig 1, Shen Fig 1).

### Fig 4 ML 이 이득인 조건

- a 최소 라벨 수 n*: 대상·지역별 점과 구간(구간 보고 형식은 막대 끝 화살표로 '초과'를 표시). 대상 묶음은 회색 띠로(Hjort Fig 2).
- b 계수 오차 대 ML 이득 산점도: 사전에 정한 오차 하한 안의 점은 중립 회색, 밖의 점은 개선·악화 두 색(Biskaborn Fig 2 의 '정확도 이내' 계급). ρ 와 구간은 패널 안 짧은 글자 한 줄.
- c 대상별 개선·악화 분포: 흩뿌린 점 + 상자 + 0 기준 띠, 축 옆에 개선·악화 방향 화살표(Talucci Fig 6c).
- d 학습기 비교가 필요하면 모양 부호로 구분해 흑백에서도 읽히게 한다(Aalto Fig 3). 기후·계수 오차 구간별 분포를 나란히 둘 때는 공유 범례 하나(Burke Fig 9 의 약점을 피함).

### Fig 5 다음 관측 위치

- 위 행: 예시 지역 하나에서 전략별 선택 지점을 작은 지도 3–4개로(같은 예산). 아래 행: 같은 열에 예산 대 RMSE 곡선. 지도와 곡선을 열 단위로 정렬한다(Linnenbrink Fig 3 의 위·아래 정렬).
- 대안: '가장 가까운 라벨까지의 거리 → 기대 이득' 지도 하나(Ploton Fig 5c).
- 전략 7종 × 예산 전체: 점·구간 열 그림(전략 = 행, 예산 = 열 또는 x 축), 기준 전략 대비 차이를 0 중심 발산색으로. 전이 전략(S1, S2, S4, S6)은 같은 그림의 하단 행으로.
- 거리 분포로 전략의 공간 포괄성을 보일 때는 Meyer Fig 1 의 밀도 짝을 쓴다.
- 피할 것: 알고리즘 순서도 상자.

### Fig 6 라벨 0 전이와 불확실성

- a 대비 행(사전 고정 목록): 점·구간 그림, 0 수직선, 판정 상태를 행 오른쪽 짧은 표기로. 행 묶음은 회색 띠.
- b–d 구간: 방법별 구간 폭 요약(Aalto Fig 3), 관측 대 예측 산점도에 구간 막대·1:1·±20 % 선(Langer Fig 3b), 커버리지 대 명목 수준 점(대각선 기준).
- e 지도(조건부): ALT 예측(순차형), 90 % 구간 폭(다른 순차형 색표), 외삽 영역(회색 마스크 또는 해칭) 3장. 관측 잔차를 0 중심 발산색 원으로 예측 지도 위에 겹친다(Langer Fig 3a). 구간 폭 계급별 면적 비율 막대를 지도 계급색과 연결한다(Wei Fig 9g, 단 순차형 색표로).
- 지도 6패널을 SI 로 보낼 때는 행·열 머리만으로 읽히는 소형 다중 지도(Karjalainen Fig 3), 기준 큰 지도 + 조건별 소형 지도(Natali Fig 3) 중 하나.
- 피할 것: 음이 아닌 양의 발산형(Wei Fig 9, Natali Fig 3), 신호등형(Obu Fig 6).

### Fig 7 충분 라벨 지역과 워크플로

- a 지역 내 라벨 수 곡선: 오차 하한을 회색 수평 띠로(Tsai Fig 7b 의 기준선, Ploton Fig 5b 의 영 모형 선 역할). 지역 곡선은 선 끝 직접 라벨.
- b 분해(격자 단위 편향 보정 대 격자 안 상세도): 같은 0 중심 발산 범위를 공유하는 지도 2장(Langer Fig 5a, c) 또는 설명 비율 막대. 설명 비율이 작다는 결론이면 막대가 더 직접적이다.
- c 워크플로: 상자 흐름도 대신 단계별 실제 미니 그림 3–4개를 가로로 늘어놓고 단계 이름을 명사구로 위에 둔다(Nitzbon 2024 Fig 3 의 개념 위·결과 아래, Ploton Fig 6 의 실제 지도 축소판). 라벨 수 구간별 권고는 x 축 아래 구간 띠에 직접 적는다.
- 표시 항목: Sci Rep 상한(그림·표 합계 8)을 고려하면 워크플로 표는 SI 표 또는 그림 안 구간 띠로 둔다.

---

## 8. 공통 규칙

### 8.1 모범 사례에서 추출한 규칙

1. 개념도는 좌표(온도·깊이, 깊이·시간), 단면 삽화, 기하 도형 위에 직접 라벨로 그린다. 문장을 담은 상자를 쓰지 않는다(Burke Fig 1, Biskaborn Fig 7, Nitzbon 2020 Fig 1·5, Nitzbon 2024 Fig 3, Talucci Fig 2, Shen Fig 2).
2. 성능 곡선에는 기준선을 같이 그린다. 결론이 되는 교차점이나 도달점은 그림 안에 직접 표시한다(Ploton Fig 5, Tsai Fig 7b).
3. 불확실성은 오차 막대(정의를 캡션에), 음영 띠, '정확도 이내' 계급 중 하나로 보이고 정의를 반드시 적는다(Hjort Fig 2, Langer Fig 3, Nitzbon 2024 Fig 1, Biskaborn Fig 2).
4. 선과 점 무리는 직접 라벨을 우선한다. 범례가 필요하면 패널 안 빈 영역에 테두리 없이 한 번만 둔다(Biskaborn Fig 1b, Aalto Fig 2, Nitzbon 2020 Fig 3).
5. 소형 다중 그림은 행·열 머리로 읽히게 하고 축과 범례를 공유한다(Karjalainen Fig 3, Nitzbon 2020 Fig 3).
6. 같은 단위의 축은 패널 사이 축척을 맞춘다. 범위가 다른 행은 패널 높이로 맞춘다(Nitzbon 2020 Fig 3).
7. 연속 양은 순차형, 부호 있는 차이는 0 중심 발산형으로 나누고, 한 지도에 둘이 같이 나오면 색표를 따로 둔다(Langer Fig 3, Fig 5).
8. 자료 종류는 모양으로도 구분해 색에만 정보를 싣지 않는다(Karjalainen Fig 2, Biskaborn Fig 2, Aalto Fig 3, Hjort Fig 3).
9. 밀집 점은 확대도, 지시선 부채꼴, 구역 단위 비례 원 중 하나로 처리한다(Hjort Fig 1, Biskaborn Fig 2c, Talucci Fig 1).
10. 지도에는 축척 막대(길이를 막대 위에), 위경도 라벨, 투영명(캡션)을 둔다.
11. 그림 사이에는 같은 대상에 같은 색을 쓰고 색 열쇠로 잇는다(Nitzbon 2024 Fig 2–3).
12. 패널 안 통계 글자는 짧게(지표명, 값, 단위, 유효 숫자 2–3자리) 쓴다(Langer Fig 3b, Aalto Fig 2. 반대 사례 Obu Fig 2).

### 8.2 이번에 원문으로 확인한 Sci Rep 지침 중 그림 설계 항목

출처: https://www.nature.com/srep/author-instructions/submission-guidelines (2026-10-04 접속). 전체 목록은 `figure_slide_standards.md` 2.1절에 있다. 아래는 위 권고와 직접 맞닿는 문장이다.
- "Put all display items on a white background, and avoid excessive boxing, unnecessary colour, spurious decorative effects (such as three-dimensional 'skyscraper' histograms) and highly pixelated computer drawings."
- "Use a clear, sans-serif typeface (for example, Helvetica) for figure lettering. Use the same typeface in the same font size for all figures in your paper."
- "The thinnest lines in the final figure should be no smaller than one point wide."
- "Figures divided into parts should be labelled with a lower-case, bold letter (a, b, c and so on) in the same type size as used elsewhere in the figure."
- "Scale bars should be used rather than magnification factors, with the length of the bar defined on the bar itself rather than in the legend."
- "Figures should not contain more than one panel unless the parts are logically connected"
- "Display items are limited to 8 (figures and/or tables)." "Please note that schemes should not be used and should be presented as figures instead."
- 범례: "Please begin your figure legends with a brief title sentence for the whole figure ... In legends, please use visual cues rather than verbal explanations such as "open red triangles". Keep each legend total to no more than 350 words."
- 파일: 선 그림·도표·도식은 벡터(EPS, AI), 다패널 그림은 한 파일, 최종본 이미지는 가능하면 300 DPI.

---

## 9. 내려받기 결과

### 9.1 새로 받은 파일 (`references/09_figure_exemplars/`, 10건)

| 파일 | 출처 경로 | 쪽수 |
|---|---|---|
| hjort2018_arctic_infrastructure_risk.pdf | nature.com OA | 9 |
| karjalainen2019_circumpolar_geohazard_maps.pdf | nature.com OA | 16 |
| nitzbon2020_ice_rich_permafrost_siberia.pdf | nature.com OA | 11 |
| ploton2020_spatial_validation.pdf | nature.com OA | 11 |
| tsai2021_parameter_learning_scaling.pdf | nature.com OA | 13 |
| burke2020_cmip6_permafrost_physics.pdf | tc.copernicus.org | 20 |
| langer2024_cryogridlite_ensemble.pdf | tc.copernicus.org | 23 |
| nitzbon2024_no_permafrost_tipping_point_preprint.pdf | EarthArXiv(프리프린트 v1) | 15 |
| shen2023_differentiable_modelling_review.pdf | OSTI(수락 원고) | 40 |
| clayton2021_alt_soil_water.pdf | OSTI(LANL 수락 원고) | 12 |

모든 파일은 `file`(PDF)과 `pdfinfo`(쪽수)로 확인했다. 폴더 색인은 `references/09_figure_exemplars/README.md` 이다.

### 9.2 다른 폴더 사본을 쓴 항목 (중복 회피)

- Wei et al. 2026, Talucci et al. 2025: `08_recent_alt_2022_2026/` 사본과 바이트 단위로 같아 09 사본을 지웠다.
- Aalto et al. 2018: `12_manuscript_refs/` 사본(헬싱키 대학 저장소판) 사용.
- Biskaborn 2019, Meyer & Pebesma 2022, Gautam 2025, Ran 2022, Obu 2019, Linnenbrink 2024: 기존 폴더 사용.

### 9.3 공개본이나 자동 내려받기에 실패한 항목 (링크만, 브라우저로 받을 수 있음)

| 논문 | DOI | 공개 경로 | 실패 사유 | 그림 검토 |
|---|---|---|---|---|
| Natali et al. 2019, Nat. Clim. Change 9, 852–857 | 10.1038/s41558-019-0592-8 | https://europepmc.org/articles/PMC8781060 | 출판사 PDF 가 HTML 반환, Europe PMC 403 | 저자 원고 그림 이미지로 검토(4.15절) |
| Hjort et al. 2022, Nat. Rev. Earth Environ. 3, 24–38 | 10.1038/s43017-021-00247-8 | http://hdl.handle.net/10138/344541 | 저장소 봇 확인 페이지 | 미검토 |
| Mishra et al. 2021, Sci. Adv. 7(9) | 10.1126/sciadv.aaz5236 | https://europepmc.org/articles/PMC7904252 | 출판사·PMC 차단 | 검토 후 제외(무지개형, 이중 y 축) |
| Hugelius et al. 2020, PNAS 117, 20438–20446 | 10.1073/pnas.1916387117 | https://europepmc.org/articles/PMC7456150 | 출판사·PMC 차단 | 검토 후 제외(원판 겹침, 고채도) |
| Natali et al. 2021, PNAS 118(21) | 10.1073/pnas.2100163118 | https://europepmc.org/articles/PMC8166174 | PMC 차단 | 미검토 |
| Peng et al. 2023, Earth's Future 11(8) | 10.1029/2023EF003573 | https://doi.org/10.1029/2023EF003573 (gold OA) | Wiley 자동 차단 | 미검토 |
| Chen et al. 2023, Earth Space Sci. 10(1) (PDO 2, ReSALT 계열) | 10.1029/2022EA002453 | https://doi.org/10.1029/2022EA002453 (gold OA) | Wiley 자동 차단 | 미검토 |

### 9.4 구독 필요

- Smith, S. L., O'Neill, H. B., Isaksen, K., Noetzli, J. et al. (5인). The changing thermal state of permafrost. *Nature Reviews Earth & Environment* 3, 10–23 (2022). doi:10.1038/s43017-021-00240-1. 구독 필요. 랜딩: https://doi.org/10.1038/s43017-021-00240-1. 과제 후보의 'Smith et al. 2022 (Permafrost & Periglac. Process.)' 는 이 논문과 같은지 확인하지 못했다 [미확인].

### 9.5 검토 후 지운 파일

- Manos, E., Witharana, C., Liljedahl, A. Permafrost thaw-related infrastructure damage costs in Alaska are projected to double under medium and high emission scenarios. *Communications Earth & Environment* 6 (2025). doi:10.1038/s43247-025-02191-7. 알래스카 대상이라 받았으나 지도가 GIS 기본 양식(회색 상자 범례, 자릿수가 많은 구간값)이라 모범 사례에서 빼고 파일을 지웠다.
- AWI 저장소의 Nitzbon 2024 파일: 출판사 링크만 담은 1쪽 안내문이라 지웠다.

### 9.6 후보였으나 대상을 특정하지 못한 항목

- 'Westermann et al. 2023 Cryosphere': The Cryosphere 2023 에서 해당 논문을 특정하지 못했다. 가까운 후보는 Westermann, S. et al. (24인) The CryoGrid community model (version 1.0). *Geoscientific Model Development* 16, 2607–2647 (2023). doi:10.5194/gmd-16-2607-2023 이며 그림은 검토하지 않았다 [미확인].
- 'Chen et al. ReSALT': 이번 검색에서 확인한 ReSALT 명칭 논문은 Schaefer et al. 2015(Remote Sensing, doi:10.3390/rs70403735)이고, Chen 계열의 최근 논문은 9.3절의 PDO 2(2023)이다. 과제 후보가 어느 논문을 뜻하는지는 특정하지 못했다 [미확인]. 둘 다 그림 미검토.

---

## 10. 남은 확인 사항

1. 프리프린트·수락 원고·저자 원고로 분석한 4편(Nitzbon 2024, Shen 2023, Natali 2019, Clayton 2021)은 게재본 그림과 다를 수 있다 [미확인]. 그림을 원고에서 인용하거나 설계 근거로 언급할 때는 게재본을 확인한다.
2. 투영명이 캡션에 없는 지도는 관찰에 근거해 '북극 중심 방위 투영'으로만 적었다.
3. OpenAlex 검색 예산 소진으로 PNAS, Sci. Adv., GRL, Earth's Future 의 2023–2026 ALT 논문을 체계적으로 훑지 못했다. 예산이 회복되면(UTC 자정) 저널별 제목 검색을 다시 할 수 있다.
4. Wiley(AGU)·PMC 논문 4건(9.3절)은 브라우저 수동 내려받기가 필요하다.
5. Sci Rep 표시 항목 상한 8(그림과 표 합계): 현행 계획(Fig 1–7 + 본문 Table 1)은 정확히 8이다. 표를 하나 더 두거나 그림을 나누면 넘는다. 워크플로 표와 AB 묶음 표는 SI 로 둔다.
6. 라벨 수 곡선(Fig 2)의 직접 선례는 영구동토 상위 학술지에서 찾지 못했다. 이 판단은 이번 검색 범위 안의 결과이다.
