# ALT 지도 제시 방식과 ML 지구과학 그림 설계 조사 (2026-10-21)

목적: 원고 v14 의 Fig 6(제품 비교 지도)과 Fig 7(평균 · 추세 · 편차 지도)을 확정하기 전에, (Q1) ALT · 영구동토 지도 논문이 지도와 관측 비교를 어떻게 제시하는지, (Q2) 2022–2026 ML 지구과학 논문이 검증 설계 · 라벨 수 · 제품 비교라는 세 메시지를 어떤 그림으로 전달하는지 확인한다.

선행 문서와의 관계
- `docs/research/2026-10-04/alt_figure_exemplars.md`(15편, 배치 · 색 · 범례 분석)와 `docs/FIGURE_AUDIT_v12_2026-10-18.md` 1–3절(저널 규정, 글꼴 실측)은 반복하지 않는다. 이 문서는 지도 유형과 관측 비교 방식, 메시지별 차트 유형만 다룬다.
- 확인 방법: 로컬 PDF(`references/`)와 이번에 내려받은 PDF 3편(Yi 2018, Roberts 2017, Wadoux 2021)은 `pdftotext` 로 캡션 전문을 읽었다. Yi 2019 와 Nitzbon 2024 게재본은 출판사 HTML 로 읽었다. 그림을 렌더링해 본 것은 아니므로 패널 안 세부(색, 기호)는 캡션에 적힌 범위까지만 적는다.
- 표기: [미확인] = 캡션 · 본문으로 확인하지 못한 항목.

---

## 1. 요지

1. 조사 대상은 Q1 지도 논문 19편(필수 13편 + QTP · 시베리아 보강 6편)과 Q2 방법 · 응용 논문 22편이다. 필수 13편 가운데 Westermann et al. 2024 는 학술 논문이 아니라 CEDA 자료 인용(Permafrost_cci ALT v4.0)이고, 제품을 기술한 TC · ESSD 논문은 CEDA · APGC 기록과 검색에서 찾지 못했다. Liu et al. 2024 는 Environmental Research Letters 19, 014020 이며 Zenodo 10070610 의 기간은 2003–2020 이다(과제 설명의 2010–2020 은 오기).
2. 지도 유형: 본문 주 지도는 다년 평균이 지배적이다(Ran 2022, Liu 2024, Yi 2018, Langer 2024, Aalto 2018, Obu 2019, Zou 2025, Wang 2025). 추세 지도는 평균 지도 다음 그림으로 따로 둔다(Liu Fig 5, Yi Fig 5, Wang Fig 13, 지점 단위로는 Streletskiy 2026 Fig 2). 연도별 지도는 선택 연도 소형 다중(Wei 2026 Fig 4 · 9) 또는 단일 연도 예시(Zhao 2025 Fig 7c 의 2010, CCI PUG 의 2015)로만 나타나고, 연도별 편차 지도는 조사한 지도 논문에서 찾지 못했다.
3. 관측 비교: 지점 평균의 관측 대 예측 산점도가 표준이다(Liu Fig 3a, Langer Fig 3b, Yi Fig 3c, Aalto Fig 2, Obu Fig 2, Zou Fig 3). 관측을 지도 위에 그릴 때는 모형 − 관측 잔차를 원의 색 · 크기로 둔다(Langer Fig 3a, Zhao Fig 7c, Obu Fig 7–9). 지점 추세의 모형 대 관측 비교(Liu Fig 3b–d, Wei Fig 3a), 확대창(Yi Fig 4, Hjort Fig 3, Du 2026 TC Fig 4–5), 제품 열 × 지역 행 소형 다중과 화소 산점도(Wei Fig 7 · 8)가 보조 방식이다.
4. Q2: (a) 검증 설계 비교는 '기준값으로부터의 편차 + 0 기준선'(Wadoux 2021 Fig 1b–d, Mila 2022 Fig 7, Linnenbrink 2024 Fig 5) 또는 설계별 관측 대 예측 산점도의 나란한 배치(Ploton 2020 Fig 4, Khanal 2023 Fig 3)로 그린다. (b) 라벨 수 곡선은 중앙값 선 + 사분위 띠 + 수평 기준선(Staudinger 2025 Fig 3, Willard 2021 Fig 7, Tsai 2021 Fig 7)이 공통 문법이다. (c) 제품 비교는 같은 색 범위의 지도 나열 + 관측 대 각 제품 산점도(Wei Fig 7–8, Feeney 2022 Fig 1–3, Khanal 2023 Fig 7)이며, Taylor 다이어그램은 Feeney 2022 보충 그림(S4–S6)에서만 쓰였다.
5. 시사점: Fig 6 의 구성(관측 최다 연도, 제품을 우리 셀 위에 재배치, 관측 셀을 같은 색 범위의 원으로, 1:1 산점도)은 Langer Fig 3 과 Wei Fig 7–8 을 합친 형태로 분야 관행 안에 있다. 다만 단일 연도 제품 비교는 선례가 없으므로 캡션에 '같은 지점-연도 비교'라는 근거를 적어야 한다. Fig 7 의 평균 지도 + 확대창 + 추세 지도는 관행과 같고, 연도별 편차 지도(7f)는 선례가 없어 보조 패널 또는 보충 그림이 적절하다.

---

## 2. Q1. ALT · 영구동토 지도 논문의 지도 제시와 관측 비교

### 2.1 논문별 정리

| 논문 | 학술지 · 연도 | 주 지도 유형 | 관측 비교 방식 | 그림 번호 | 비고 |
|---|---|---|---|---|---|
| Wei et al. | ESSD Discussions 2026 (essd-2026-447) | 연도별: 2000–2024 중 선택 연도 6장(a–f)과 ALT 계급 면적 비율(g). 불확실성도 같은 구성(Fig 9). 다년 평균(2000–2020)은 제품 비교에만 사용 | leave-one-site-out CV. 관측 대 예측 산점도는 보충 Fig S1(모형별), 지표는 Table 1(앙상블 R² 0.761, RMSE 60.48 cm). 시간 성능은 68개 장기 지점의 Sen 기울기 관측 대 예측 산점도(1:1 점선)와 변동폭 비율 히스토그램. 제품 비교는 열 = 제품 3개(본 연구, Westermann 2024, Peng 2024), 행 = 반구 + 지역 확대, 상단 행에 빈도 막대와 영역 평균값 삽입(Fig 7), 500만 화소 무작위 추출 화소 산점도에 1:1 · RMSE · 평균 차이(Fig 8) | Fig 1(지점 위치), 3, 4, 7, 8, 9 | 패널별 연도는 캡션에 없음 [미확인]. 관측점을 산출 지도 위에 그리지 않음. 비교 제품 2개는 모두 자료 인용(CEDA v4.0, Peng 2024 dataset) |
| Liu et al. | Environ. Res. Lett. 19, 014020 (2024) | 다년 평균 2003–2020(Fig 4a)에 SD · 변동계수 · 위도 요약(4b–d). 추세 지도 2003–2020(Fig 5a)에 하한 · 상한 · SD · 위도 요약(5b–e). 연도별 지도는 본문에 없음 | CALM 지점 평균의 밀도 산점도(log10, 최적 적합선과 1:1 점선; RMSE 21.60 cm, bias 4.02 cm, R² 0.97). 지점 위치에서의 추세를 RF 와 관측으로 각각 지도에 찍고(3b–c) 추세 일치 비율 막대(3d) | Fig 1(지점, 자료원별 기호), 3, 4, 5 | 다른 자료와의 비교는 본문 4.1 절 문장(추세율 인용)뿐이며 제품 지도 비교 그림은 없음 |
| Ran et al. | ESSD 14, 865–884 (2022) | 다년 평균 2000–2016(ALT Fig 4, MAGT Fig 3), 위도 · 고도 · 건조지수 단면(Fig 5) | 거리 블록 10겹 CV × 1000회의 수치만 본문 · 초록(ALT bias 2.71 ± 16.46 cm, RMSE 86.93 ± 19.61 cm). 관측 대 예측 그림 없음. 지점 위치는 Fig 2 | Fig 2, 4, 5 | 이전 지도와의 비교는 문장으로만 |
| Aalto et al. | GRL 45, 4889–4898 (2018) | 다년 평균 2000–2014(앙상블 중앙값)와 2041–2060, 그 차이를 한 그림에(Fig 4; 다른 기간 · 시나리오는 S3–S4) | 과거 기간(1970–1984, 1985–1999) 관측 대 예측 산점도, bias · RMSE · R², 1:1 점선(Fig 2). 방법별 PI50 · PI95 불확실성(Fig 3) | Fig 1(지점), 2, 3, 4 | 예측 지도 위에 관측점 없음 |
| Westermann et al. (Permafrost_cci ALT v4.0) | CEDA 자료 인용 2024, doi:10.5285/d34330ce3f604e368c06d76de1987ce5 | 연도별 1 km 제품(1997–2021). 제품 사용 설명서(PUG, 로컬 `01_benchmark/esa_cci_permafrost_v4.pdf`)의 예시 지도는 단일 연도(MAGT 1 m 2005, ALT 2015, 범위 2015) | PUG 에는 검증 그림이 없고 별도 PVIR(Product Validation and Intercomparison Report)로 넘김 | PUG Fig 1–3 | TC · ESSD 제품 논문은 CEDA · APGC 기록과 검색에서 확인되지 않음 [미확인]. CEDA 기록의 선행판은 Obu et al. 2021 v3.0 자료 인용 |
| Obu et al. | Earth-Sci. Rev. 193, 299–316 (2019) | 다년 평균 2000–2016 MAGT(Fig 3), 영구동토 확률(Fig 4), 구역(Fig 5), 앙상블 SD(Fig 6). ALT 지도 없음 | 시추공 측정 대 모형 TTOP 산점도, 1:1 실선과 ±2 °C 점선(Fig 2). 지역 지도 3장(러시아, 캐나다 · 알래스카, 중국 · 몽골)에 구역 바탕 위 시추공 − 모형 차이를 점으로(Fig 7–9) | Fig 2, 3, 7, 8, 9 | 잔차를 지도 위 점으로 두는 방식의 원형 |
| Gautam et al. | Sci. Rep. 15, 42420 (2025) | 기준기 ALT 지도 2장(RF a, Stefan b, 세로 배치, Fig 2). 2030–2100 변화 지도(Fig 3–4), 10년 단위 변화 선과 사분위 띠(Fig 5) | 본문 그림에 관측 대 예측 산점도 없음. 10겹 CV 와 시험 자료 R² 를 본문 수치로만 제시 | Fig 1(변수 중요도), 2, 3, 4, 5 | 기준기 연도 범위가 캡션에 없음 [미확인]. 관측점을 지도에 그리지 않음 |
| Langer et al. | The Cryosphere 18, 363–385 (2024) | 10년 평균 2007–2016(ALT Fig 3a, MAGT Fig 2a). 기간 간 상대 변화 지도 2장과 기준 기간 평균을 가운데 둔 3패널(Fig 5) | 지도 위 CALM 지점에 모형 − 관측 편차 원(3a). 격자 셀 평균의 산점도: y 오차 막대 = 앙상블 5–95 백분위, x 오차 막대 = 같은 셀 안 관측 범위, ±20 % 점선, ALT < 2 m 회색 사각형(3b) | Fig 2, 3, 5 | Fig 6 설계의 가장 가까운 선례 |
| Nitzbon et al. | Nat. Clim. Change 14, 573–585 (2024) | 지도는 하위 지역 구분도 1장(Fig 2). ALT 지도 · 관측 비교 없음. 나머지는 GMST 대 평형 영구동토 면적(Fig 1, 4), 과정 삽화 + 모의 예(Fig 3), 탄소 배출(Fig 5) | 해당 없음 | Fig 1–5(게재본) | 게재본 캡션은 제목 문장만 읽음. 프리프린트는 Fig 6 까지였고 번호가 다름 |
| Talucci et al. | ESSD 17, 2887–2909 (2025) | 격자 지도 없음. 생태구역별 측정 비율을 비례 원으로(Fig 1) | 관측 대 추정 산점도를 소실 · 비소실로 나누고 1:1 주황선(Fig 6a–b), 플롯별 편향을 0 기준선과 평균 마름모로(6c). 관측 · 추정 분포는 바이올린 + 상자(Fig 5) | Fig 1, 5, 6 | 지점 자료 논문 |
| Yi et al. | The Cryosphere 12, 145–161 (2018) | 다년 평균 2001–2015 1 km ALT(Fig 3a, ALT > 300 cm 짙은 회색)와 영구동토 확률 지도(3b). 추세 지도 2001–2015(Fig 5a)와 확률 구역별 추세 분포(5b). SOC 불확실성 지도(Fig 7a) | 확률 구역별 모형 대 CALM 비교, 오차 막대 = 겹치는 기간의 SD(3c), 확률에 따른 모형 · 관측 ALT 변화(3d). 확대창: Dalton Highway 구역의 레이더 ALT 지도와 CALM 지점(WD, DHS, FB, SH)별 관측 · 레이더 · 모형 비교, 위도대 분포(Fig 4) | Fig 3, 4, 5, 7 | CALM 지점을 3a 지도에 찍었는지는 캡션에 없음 [미확인]. ORNL DAAC 자료 설명서의 평균 지도에는 검은 점으로 표시 |
| Yi et al. | The Cryosphere 13, 197–218 (2019) | 동결 과정 · 영도 유지 기간 논문. ALT 지도 없음. 지점 지도(Fig 2)와 동결 시작 · 영도 유지 기간 추세 지도 2001–2016(Fig 9) | 지점 모형 대 관측 시계열 · 산점(Fig 4, 6) | Fig 2, 9 | ABoVE ALT 지도는 2018 논문에 있음 |
| Karjalainen et al. | Sci. Data 6, 190037 (2019) | 현재 · 미래 모형 범위 위에 관측 지점(MAGT 검은 기호, ALT 흰 기호)을 겹친 지도(Fig 2). 지수 지도 4 × 2(Fig 3) | 반복 무작위 CV 에 500 km 거리 제약을 더한 설계와 과거 기간 재예측(Technical Validation 문단, Table 3). 관측 대 예측 그림은 본문에 없음 | Fig 2, 3 | 보충 그림 여부 [미확인] |
| Hjort et al. | Nat. Commun. 9, 5147 (2018) | 기준기 · 미래 영구동토 범위 위에 시추공 관측 MAGT 를 색 원으로(Fig 1). 위험 지도와 확대창 2곳(Fig 3) | 본문 그림에 관측 대 예측 없음 | Fig 1, 3 | 관측값을 지도 위에 색으로 두는 방식(우리 Fig 6 · 7 의 관측 셀 채움과 같음) |
| Wang et al. (QTP, Stefan-CatBoost-ET) | Remote Sensing 17, 2006 (2025) | 제품 표시도(Fig 11, 1958–2022 산출), 다년 평균 분포와 변화(Fig 12), 추세 지도(Fig 13) | 10겹 CV 산점(Fig 7–8), 오차 분석(Fig 9), 위도 · 경도 · 고도에 대한 잔차 그림(Fig 10) | Fig 7–13 | 캡션이 한 문장이라 패널 구성은 본문 서술로만 확인 |
| Zou et al. (QTP MAGT 15 m) | ESSD 17, 1731 (2025) | 다년 평균 2010–2019 지도(Fig 4), 고도 · 경도 · 위도 단면(Fig 5) | 관측 대 예측 산점도(Fig 3), 시추공 231개 위치(Fig 1) | Fig 1, 3, 4 | ALT 가 아닌 MAGT |
| Zhao et al. (서쿤룬) | The Cryosphere 19, 4211 (2025); 로컬은 egusphere-2024-3956 프리프린트 | 단일 연도 2010 ALT · 융해 깊이 지도에 관측 편차 원(크기 · 색)을 겹침(Fig 7c). 1980 스냅샷과 고도별 상자그림(Fig 9), 10년 단위 상대 변화 지도(Fig 10) | 시추공 · GPR 관측 대 모형 산점도, ±0.25 m 회색 띠, 1:1 빨간선(Fig 7a–b) | Fig 7, 9, 10 | 게재본 그림 번호 [미확인] |
| Du et al. (QTP 활동층 수분 90 m) | ESSD Discussions 2026 (essd-2026-330) | 융합 ALM 지도 1장(Fig 9)과 모형별 지도 4장(Fig 6), 적용 영역 · 비유사도 지도(Fig 11) | 공간 CV 바깥 겹 예측을 모은 관측 대 예측 산점도, 학습 · 시험 나란히(Fig 5). 기존 토양수분 제품 2종과의 밀도 산점도(Fig 12) | Fig 5, 6, 9, 11, 12 | ALT 가 아닌 수분. 공간 CV 에서 R² 0.30–0.38 로 떨어짐을 본문에 명시 |
| Streletskiy et al. (CALM) | Commun. Earth Environ. 2026, doi:10.1038/s43247-026-03824-1 | 격자 지도 없음. 지점 지도(기호 = 측정 방법, 색 = 기록 길이, Fig 1), 유의한 지점 추세 2000–2024 를 색으로(Fig 2) | 지역별 연도 편차 시계열(Fig 3), 지역 · 구역별 연 변화 상자그림(Fig 5) | Fig 1, 2, 3, 5 | 편차를 지점 시계열로 보이는 방식 |
| Guo et al. (앙상블 융해 조건) | ESSD Discussions 2026 (essd-2026-299) | 기존 제품 지도를 나란히(영구동토 비율 Fig 1, MAGT Fig 2, ALT Fig 3). CCI 1997–2021 추세 지도(Fig 4) | GTN-P · CALM 지점의 위성 추정 대 관측 PTI(Fig 10) | Fig 1–4, 10 | 제품 간 비교를 지도 나열로 |

시베리아 2023–2026 ML ALT 지도 논문은 검색(extended)에서 찾지 못했다. 가까운 것은 Langer 2024(과정 모형 앙상블), ALLena 융해 깊이 자료 집성(AWI, 1998–2022), Streletskiy 2026(CALM 지점)이다. 과제가 예시한 Ni et al. 은 JGR Atmos. 126, e2020JD033402 (2021), Xu et al. 은 Int. J. Digital Earth 10, 522–538 (2017)로 조사 기간 밖이며 그림은 보지 않았다 [미확인].

### 2.2 분야 관행 요약

1. 주 지도는 다년 평균 1장이고 추세 · 변화 지도는 별도 그림이다. 평균과 추세를 한 그림에 넣은 예는 없고, Langer Fig 5 처럼 기준 상태를 가운데에 두고 변화를 양옆에 두는 3패널은 있다.
2. 연도별 지도는 선택 연도 소형 다중(Wei Fig 4 · 9)이나 단일 연도 예시(Zhao Fig 7c, CCI PUG)로만 쓰인다. 연도별 편차 지도는 없고, 편차는 지점 시계열(Streletskiy Fig 3)이나 면적 비율 막대(Wei Fig 4g)로 보인다.
3. 관측 비교의 표준은 지점(또는 셀) 평균의 산점도이다. 지점-연도 전부를 찍는 산점도는 보충(Wei Fig S1) 또는 자료 논문(Du Fig 5)에 둔다. 산점도에는 1:1 선과 함께 ±20 %(Langer), ±2 °C(Obu), ±0.25 m(Zhao) 허용 띠를 자주 둔다.
4. 관측을 지도 위에 둘 때 두 유형이 있다. 위치 · 관측값(Hjort Fig 1, Karjalainen Fig 2, Streletskiy Fig 1)과 모형 − 관측 잔차(Langer Fig 3a, Zhao Fig 7c, Obu Fig 7–9). 잔차 유형은 연속 변수 색표와 0 중심 발산 색표를 한 지도에서 분리한다.
5. 제품 비교는 (i) 같은 색 범위의 지도 나열(Wei Fig 7, Guo Fig 1–3), (ii) 제품 간 화소 산점도(Wei Fig 8), (iii) 지점 관측 대 각 제품 산점도(Khanal Fig 7, Feeney Fig 3)의 조합이다. 관측 대 제품 비교를 단일 연도에서 수행한 예는 없다(Wei 는 2000–2020 평균, Feeney 는 2007 조사 자료 대 정적 지도).

### 2.3 Fig 6(선택 연도 제품 비교, 관측 셀 표시)에의 시사점

- 구성 자체(지도 3장 + 산점도 1장 × 지역 3행)는 Langer Fig 3(지도 + 산점도)과 Wei Fig 7(열 = 제품) · Fig 8(1:1 산점도)의 결합이며 분야 관행 안에 있다. 제품 지도를 우리 셀 위에 재배치한 점은 Feeney Fig 1 의 '공통 범위 · 해상도로 조화한 뒤 비교'와 같은 논리이므로 캡션에 그대로 적어 둔다.
- 관측 셀을 같은 색 범위의 관측값으로 채우는 방식은 Hjort Fig 1 유형이다. 잔차 유형(Langer, Zhao)이 아니므로 '세 지도 중 어느 쪽이 관측에 가까운가'를 눈으로 비교하게 하는 목적이 분명히 드러난다. 잔차 지도는 보충 그림으로 두면 두 유형을 모두 갖춘다.
- 단일 연도(관측 최다 연도) 비교는 선례가 없다. 캡션 첫 문장에 '같은 지점-연도에서의 비교'임을 적고, 다년 평균 비교(관행)와의 관계를 본문에서 한 문장으로 잇는다(지점-연도 라벨 사용이 연도 정합의 근거라는 점).
- 산점도 d · h · l 의 채움 · 빈 기호(Wei 학습 지점 25 km 이내 · 밖) 구분은 Meyer & Pebesma 2022 의 거리 논리를 기호로 옮긴 것이며 조사한 지도 논문에는 같은 기호 사용이 없다. 범례 문장에서 기호 의미를 먼저 적는다.
- 허용 띠(±20 % 등)는 우리 산점도에 없다. n 이 작은 지역에서는 1:1 선만으로 충분하지만, 알래스카처럼 n 이 큰 패널에는 Langer 식 ±20 % 점선을 검토할 수 있다.

### 2.4 Fig 7(평균 지도 + 추세 + 편차)에의 시사점

- 평균 지도 + 확대창 + 관측 셀 채움(7a–c)은 Yi Fig 4 · Hjort Fig 3 의 확대창, Hjort Fig 1 의 관측값 채움과 일치한다. 적용 영역 밖을 옅게 표시하는 방식은 Du 2026 Fig 11(AOA 마스크)과 같은 정보이며 지도 안에 통합한 점이 더 간결하다.
- 추세 지도(7e)는 Liu Fig 5a · Yi Fig 5a · Wang Fig 13 과 같은 유형이다. 세 논문 모두 15년 이상 기간이고, Liu 는 추세의 하한 · 상한 · SD 를 같이 둔다. 우리 추세는 2011–2024 의 14년이고 '연도별 기후에 대한 모형 반응'이므로 캡션 문구를 유지하고, 추세 검증은 Liu Fig 3b–d(지점 추세 모형 대 관측)보다 7d 의 편차 대조가 적합하다는 점을 본문에 둔다.
- 지역 평균 편차 대 관측 셀 편차(7d)는 Streletskiy Fig 3(지점 편차 시계열)의 격자판에 해당하며 선례와 맞는다.
- 연도별 편차 지도(7f)는 선례가 없다. 가장 가까운 형식은 Wei Fig 4(선택 연도 소형 다중)와 Zhao Fig 10(10년 단위 상대 변화)이다. 본문에 두려면 선택 연도 3–4장으로 줄이고 0 중심 발산 색표와 고정 범위를 쓰며, 전체 14년은 보충 그림으로 보내는 편이 관행에 가깝다.

---

## 3. Q2. 메시지별 그림 설계 사례 (2022–2026 ML 지구과학 논문)

### 3.1 (a) 검증 설계별 정확도 (무작위 · 공간 블록 · 지역 홀드아웃, 무작위 CV 의 낙관)

| 논문 · 그림 | 패널 내용 | 차트 유형 | 전달이 잘 되는 이유 |
|---|---|---|---|
| Wadoux et al. 2021, Ecol. Model. 457, 109692, Fig 1b–d | 보정 표본 설계(계통 무작위 b, 단순 무작위 c, 2단 군집 d)별로 무작위 CV · 공간 K겹 CV · 버퍼 LOO CV · 설계 기반 검증이 추정한 RMSE 의 모집단 RMSE 로부터의 편차. 0 수평선 = 참값. a 는 연구 지역 지도 | 편차 점 · 분포 + 0 기준선 | 낙관 · 비관이 0 선 위아래로 바로 읽힌다. 표본 설계를 행으로 두어 '언제 무작위 CV 가 낙관인가'를 조건부로 보인다 |
| Ploton et al. 2020, Nat. Commun. 11, 4540, Fig 4 · 5 | Fig 4: 무작위 10겹(a) 대 공간 44겹(b) CV 의 관측 대 예측 열지도. Fig 5: 버퍼 반경에 따른 R²(a) · RMSPE(b), 영 모형 선과 공간 K겹 참조선, 반경을 최근접 관측 거리로 옮긴 기대 R² 지도(c) | 산점 열지도 나란히 + 거리 곡선 | 같은 축의 두 열지도가 낙관을 직접 보이고, 곡선이 영 모형과 만나는 거리가 결론이 된다 |
| Milà et al. 2022, Methods Ecol. Evol. 13, 1304, Fig 6 · 7; Linnenbrink et al. 2024, GMD 17, 5897, Fig 5 | 검증 방법 × 표본 분포 × 자기상관 범위별 RMSE(Mila Fig 6), CV 추정과 참 RMSE 의 차이(보간 · 외삽, Mila Fig 7). Linnenbrink Fig 5 는 CV − 참값 차이를 RMSE · R² · MAE 로, 표본 분포 × CV 방법 격자 | 소형 다중 차이 그림 + 0 기준선 | 무작위 CV 의 낙관과 공간 CV 의 비관을 같은 눈금에서 보인다. 행 · 열 머리만으로 읽힌다 |
| Khanal et al. 2023, Sci. Rep. 13, 8090, Fig 3 · 4 | Fig 3: 10겹 공간 CV(A)와 무작위 CV(B)의 관측 대 예측 산점도, RMSE · R² 표기. Fig 4: 두 CV 잔차의 베리오그램과 무작위 순열 포락선 | 산점도 2패널 + 베리오그램 | 목표 학술지(Sci Rep)의 응용 예. 두 패널 같은 축이 핵심이며 잔차 자기상관으로 이유를 덧붙인다 |
| Feng et al. 2023, HESS 27, 2357, Fig 1 · 3 · 6 | Fig 1: 미계측 유역(PUB)과 미계측 지역(PUR) 분할 도식(학습 유역 검은 점, 시험 유역 빨간 별). Fig 3 · 6: 같은 지표(NSE, KGE, FLV, FHV)의 상자그림을 PUB 와 PUR 로 각각, 'in sample' 모형을 표기 | 분할 지도 도식 + 지표별 상자그림 | 설계 차이를 실제 지점 지도로 보이고 결과를 같은 지표 · 같은 축으로 반복해 설계 간 차이를 읽게 한다 |
| Meyer & Pebesma 2022, Nat. Commun. 13, 2208, Fig 1; Meyer et al. 2021, Methods Ecol. Evol. 12, 1620, Fig 9 | 표본 간 최근접 거리와 예측 위치 · 표본 거리 분포의 짝(2022). 군집 표본에서 공간 CV 와 무작위 CV 오차가 각각 적용되는 영역(AOA) 비교(2021 Fig 9) | 밀도 짝 그림, AOA 지도 | 검증 설계가 예측 거리와 맞는지를 분포로, 오차가 유효한 영역을 지도로 보인다 |
| Roberts et al. 2017, Ecography 40, 913 | 본문 그림 4장은 개념도(Fig 1 의존 구조와 블록, Fig 2 블록 크기 선택, Fig 3 비유사도 대 정확도와 예측 지평, Fig 4 블록 설계의 문제). 무작위 대 블록 오차 추정의 실증 그림은 본문에 없음 | 개념도 | 설계 원리 인용에만 쓴다 [사례 연구 그림 미확인] |

우리 그림에의 적용: 검증 사다리(무작위 → 공간 블록 → 지역 홀드아웃)는 Feng Fig 3 · 6 처럼 같은 지표 · 같은 y 축의 설계별 상자(또는 점 · 구간)로 두고, 무작위 CV 의 낙관은 Wadoux 식으로 '지역 홀드아웃 대비 차이 + 0 선'으로 한 패널에 요약한다. 거리 분포(Meyer 2022)는 보충에 둔다. Khanal Fig 3 유형의 두 산점도 나란히 배치는 n 이 큰 알래스카 한 지역에만 쓴다.

### 3.2 (b) 라벨 수에 따른 정확도 · 학습 곡선 · 새 지역 소수 라벨 전이

| 논문 · 그림 | 패널 내용 | 차트 유형 | 전달이 잘 되는 이유 |
|---|---|---|---|
| Staudinger et al. 2025, HESS 29, 5005, Fig 3 · 4 · 5 | Fig 3: 모형 × 유역별 조건부 엔트로피 대 표본 수, 30회 반복의 25–75 백분위 띠와 중앙값 선, 최대 엔트로피 기준 점선. Fig 4: 곡선 시작과 끝의 차이(상대 학습량) 막대. Fig 5: 표본 추출 방식별 곡선 | 학습 곡선 소형 다중 + 기준선 + 요약 막대 | 곡선(띠 + 기준선)과 그 요약값(막대)을 분리해 '얼마나 더 배우는가'를 수치로 남긴다 |
| Willard et al. 2021, Water Resour. Res.(수락 원고; arXiv:2011.05369), Fig 7 · 3 | Fig 7: 대상 호수에서 온도 프로파일 수에 따른 PGDL 중앙값 RMSE, 오차 막대 = 5회 무작위 선택의 사분위, 수평 색선 = 라벨 0 전이 모형(PGDL-MTL, MTL9)의 중앙값과 사분위 띠. Fig 3: 전이 모형 대 기준 모형 RMSE 산점도(1:1)와 차이 | 소수 라벨 곡선 + 수평 전이 기준선 | '몇 개의 현지 라벨이 전이 모형과 같아지는가'를 교차점으로 읽게 한다. 우리 Fig 4 와 같은 질문 |
| Tsai et al. 2021, Nat. Commun. 12, 5988, Fig 7 | a: 학습 표본 밀도(s16 → s4)에 따른 RMSE, SCE-UA 점선 대조. b: 공간 외삽(PUB) 시험에서 Beck20 자료 대비 규모 곡선 | 규모 곡선 + 외부 기준 | 외부 기준값에 도달하는 자료 비율을 그림에서 바로 읽는다 |
| Linnenbrink et al. 2024, GMD 17, 5897, Fig 7 | k 값별 kNNDM CV 오차 추정을 표본 수 100–4000 에 대해, 점 = 중앙값, 막대 = 사분위, 넷째 행에 W 통계 | 표본 수 곡선 소형 다중 | 검증 방법의 편향이 표본 수에 따라 어떻게 변하는지를 같은 눈금으로 |
| Hollmann et al. 2025, Nature 637, 319, Fig 5a | 자료 변형별 정규화 점수 비교. 캡션: TabPFN 이 학습 표본 절반으로 CatBoost 기본 설정의 정확도를 재현 | 정규화 점수 점 · 구간 | 표본 효율을 '같은 정확도에 필요한 표본 수'로 환산해 진술한다 |
| Chance et al. 2024, Earth Sci. Inform. 17, 6055, Fig 9 | 학습 자료 길이에 따른 최고 ML 모형과 기준 모형의 오차, 계절별 | 학습 곡선 + 기준 모형 | 영구동토 지온 예측에서 학습 자료량 효과를 기준 모형과 같이 보인 예 |
| O'Malley et al. 2026, arXiv:2605.16665, Fig 4 · 5 | 미국 학습 모형을 앨버타 · 호주 · 영국에 적용한 관측 대 예측 패리티 그림(95 % 구간 선)과 지역별 구성 요소 제거 실험의 MAE 개선 | 지역별 패리티 + 제거 실험 막대 | 새 지역 전이를 지역별 패널로 분리하고 구간을 함께 보인다 |

우리 그림에의 적용: Fig 4(표본 효율)는 x = 지점-연도 수(순서 눈금), y = RMSE, 중앙값 선 + 블록 부트스트랩 사분위 띠(Staudinger), Stefan 기준선과 라벨 0 전이 기준선을 수평선 · 띠로(Willard Fig 7), 기준선에 도달하는 n 을 눈금으로 표시(Tsai Fig 7b)한다. 지역은 y 축을 공유하는 열 패널로 두고 곡선 이름은 선 끝 직접 라벨로 한다.

### 3.3 (c) 새 지도와 기존 제품의 비교 (같은 지점 · 같은 연도)

| 논문 · 그림 | 패널 내용 | 차트 유형 | 전달이 잘 되는 이유 |
|---|---|---|---|
| Wei et al. 2026, ESSDD, Fig 7 · 8 | Fig 7: 열 = 제품 3개, 행 = 반구 + 지역 확대, 상단 행에 유효 화소 빈도 막대와 영역 평균값 삽입, 공유 색 막대. Fig 8: 2000–2020 평균의 제품 간 화소 산점도(500만 화소), 1:1 점선, RMSE 와 평균 차이(제품 − 본 연구) | 제품 × 지역 소형 다중 지도 + 화소 산점도 | 지역 평균값을 패널 안 숫자로 두어 지도와 수치를 같이 읽게 한다. 제품 간 차이를 '관측 없이도' 정량화 |
| Feeney et al. 2022, Sci. Rep. 12, doi:10.1038/s41598-022-05476-5, Fig 1–4, S4–S6 | Fig 1: 8개 지도의 평균 · SD · 변동계수 · 신호 대 잡음 지도. Fig 2: 지도별 분포 바이올린 + 상자, 토양 계급 띠. Fig 3: 지도별 예측 대 2007 조사 측정 산점도, 1:1 점선, 최적 적합 ± 1 SE, 측정 SE 수평선. Fig 4: 위도대 · 토지피복 · 토양군별 분포 상자와 관측 IQR 띠. Taylor 다이어그램은 S4–S6 | 조화 지도 통계 + 제품별 산점도 소형 다중 + 계급별 분포 | 공통 범위 · 해상도로 조화한 뒤 비교한다는 절차가 그림 순서에 드러난다. Taylor 는 보충으로 보낸 점이 Sci Rep 관행 |
| Khanal et al. 2023, Sci. Rep. 13, 8090, Fig 7 | 예측 SOC 대 GSOCmap(A) · SoilGrids250m(B) 산점도, 색 = 고도, 1:1 빨간선, RMSE | 제품별 산점도 2패널 | 같은 지점에서 새 지도와 기존 제품을 같은 축으로 대조. 공변량(고도)을 색으로 더해 불일치 원인을 보인다 |
| Langer et al. 2024, TC 18, 363, Fig 3 | 지도 위 CALM 셀의 모형 − 관측 편차 원(a) + 셀 평균 산점도(앙상블 5–95 % 막대, 관측 범위 막대, ±20 % 점선)(b) | 잔차 지도 + 구간 산점도 | 어디서 틀리는지(지도)와 얼마나 틀리는지(산점도)를 한 그림에 |
| Liu et al. 2024, ERL 19, 014020, Fig 3 | 지점 평균 밀도 산점도(a), 지점 위치에서의 추세를 RF 와 관측으로 각각 지도에(b–c), 추세 부호 일치 막대(d) | 밀도 산점도 + 지점 추세 지도 짝 + 일치 막대 | 평균뿐 아니라 추세의 일치를 지점 단위로 검증하는 유일한 예 |
| Guo et al. 2026, ESSDD, Fig 1–3; Du et al. 2026, ESSDD, Fig 12; Zhao et al. 2025, TC, Fig 7c | 기존 제품 지도를 같은 계급 범례로 나열(Guo). 새 지도 대 기존 제품 2종의 밀도 산점도, 1:1 과 회귀선(Du). 단일 연도 지도 위 관측 편차 원(크기 · 색)(Zhao) | 지도 나열 · 밀도 산점도 · 잔차 원 | 각각 Fig 6 의 지도 행, 산점도 열, 관측 표시 방식의 선례 |

우리 그림에의 적용: Fig 6 은 Langer Fig 3 과 Wei Fig 7–8 의 결합이므로 구조를 유지한다. 산점도는 n 이 작아 점으로 두고 밀도 · hexbin 은 쓰지 않는다. 캡션의 RMSE(전체 · 25 km 밖) 두 값은 Wei Fig 8 의 'RMSE + 평균 차이' 표기와 같은 밀도이다. 계급별 잔차(영구동토 구역 · 거리 계급)는 Feeney Fig 4 · Yi Fig 3c 유형으로 보충에 둔다. Taylor · target 다이어그램은 조사 범위의 ALT 논문에 없고 Feeney 도 보충에 두었으므로 본문에는 쓰지 않는다.

---

## 4. 출처 목록

| # | 서지 | DOI · URL | 확인 방법 | 확인 범위 |
|---|---|---|---|---|
| 1 | Wei, Y., Wu, Z., Wang, J., Shen, H., Chen, Y. A 1 km resolution dataset of Northern Hemisphere permafrost ALT (2000–2024). ESSD Discuss. (2026) | 10.5194/essd-2026-447; 자료 Zenodo 21667583 | 로컬 PDF 캡션 · 본문 | Fig 1, 3, 4, 7, 8, 9 캡션 전문. Fig 4 · 9 패널 연도 [미확인] |
| 2 | Liu, Z., Kimball, J. S., Ballantyne, A. et al. Widespread deepening of the active layer in northern permafrost regions from 2003 to 2020. ERL 19, 014020 (2024) | 10.1088/1748-9326/ad0f73; 자료 Zenodo 10070610 | 로컬 PDF | Fig 1, 3, 4, 5 캡션, 4.1 절 |
| 3 | Ran, Y. et al. ESSD 14, 865–884 (2022) | 10.5194/essd-14-865-2022 | 로컬 PDF | Fig 2, 4, 5 캡션, 초록 · 방법 문장 |
| 4 | Aalto, J., Karjalainen, O., Hjort, J., Luoto, M. GRL 45, 4889–4898 (2018) | 10.1029/2018GL078007 | 로컬 PDF | Fig 1–4 캡션 |
| 5 | Westermann, S. et al. ESA Permafrost_cci ALT for the Northern Hemisphere, v4.0 (2024) | 10.5285/d34330ce3f604e368c06d76de1987ce5; https://apgc.awi.de/dataset/permafrost-cci-alt-north-hemisphere-v4-1997-2021 | CEDA · APGC 기록 WebFetch, 로컬 PUG PDF | 자료 인용 확인. 제품 논문 [미확인] |
| 6 | Obu, J. et al. Earth-Sci. Rev. 193, 299–316 (2019) | 10.1016/j.earscirev.2019.04.023 | 로컬 PDF | Fig 2, 3, 7 캡션 |
| 7 | Gautam, S., Mishra, U., Scott, S., Lara, M. J. Sci. Rep. 15, 42420 (2025) | 10.1038/s41598-025-26586-w | 로컬 PDF | Fig 1–5 캡션, 방법 문장 |
| 8 | Langer, M. et al. The Cryosphere 18, 363–385 (2024) | 10.5194/tc-18-363-2024 | 로컬 PDF | Fig 3, 5 캡션 |
| 9 | Nitzbon, J. et al. Nat. Clim. Change 14, 573–585 (2024) | 10.1038/s41558-024-02011-4 | 게재본 HTML WebFetch(제목 문장), 프리프린트 로컬 PDF | 게재본 Fig 1–5 제목, 프리프린트 Fig 1–6 캡션 |
| 10 | Talucci, A. C. et al. ESSD 17, 2887–2909 (2025) | 10.5194/essd-17-2887-2025 | 로컬 PDF | Fig 1, 5, 6 캡션 |
| 11 | Yi, Y., Kimball, J. S., Chen, R. H. et al. The Cryosphere 12, 145–161 (2018) | 10.5194/tc-12-145-2018 | PDF 내려받기(scratchpad) · pdftotext | Fig 1–8 캡션 전문 |
| 12 | Yi, Y. et al. Sensitivity of active-layer freezing process to snow cover in Arctic Alaska. The Cryosphere 13, 197–218 (2019) | 10.5194/tc-13-197-2019 | 출판사 HTML WebFetch | Fig 1–10 캡션 |
| 13 | Karjalainen, O. et al. Sci. Data 6, 190037 (2019) | 10.1038/sdata.2019.37 | 로컬 PDF | Fig 1–3 캡션, Technical Validation 문단 |
| 14 | Hjort, J. et al. Nat. Commun. 9, 5147 (2018) | 10.1038/s41467-018-07557-4 | 로컬 PDF | Fig 1–3 캡션 |
| 15 | Wang et al. Simulation of ALT based on multi-source remote sensing data and integrated ML models: QTP. Remote Sens. 17, 2006 (2025) | 10.3390/rs17122006 | 로컬 PDF | Fig 7–13 캡션(한 문장), 본문 서술 |
| 16 | Zou, D. et al. Permafrost temperature baseline at 15 m depth on the QTP (2010–2019). ESSD 17, 1731 (2025) | 10.5194/essd-17-1731-2025 | 로컬 PDF | Fig 1, 3, 4 캡션 |
| 17 | Zhao et al. The thermal state of permafrost on the QTP (1980–2022): West Kunlun. The Cryosphere 19, 4211 (2025) | 게재본 https://tc.copernicus.org/articles/19/4211/2025/ ; 프리프린트 10.5194/egusphere-2024-3956 | 로컬 프리프린트 PDF | 프리프린트 Fig 7, 9, 10 캡션. 게재본 번호 [미확인] |
| 18 | Du et al. 90 m active layer moisture dataset, QTP. ESSD Discuss. (2026) | 10.5194/essd-2026-330 | 로컬 PDF | Fig 5, 6, 9, 11, 12 캡션 |
| 19 | Du et al. ALT heterogeneity, Arctic foothills. The Cryosphere 20, 4277 (2026) | 10.5194/tc-20-4277-2026 | 로컬 PDF | Fig 4–6 캡션 |
| 20 | Streletskiy, D. et al. Long-term monitoring of ALT confirms global permafrost degradation. Commun. Earth Environ. (2026) | 10.1038/s43247-026-03824-1 | 로컬 PDF | Fig 1, 2, 3, 5 캡션 |
| 21 | Guo et al. An ensemble dataset of permafrost thaw conditions. ESSD Discuss. (2026) | 10.5194/essd-2026-299 | 로컬 PDF | Fig 1–4, 10 캡션 |
| 22 | Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., Brus, D. J. Ecol. Model. 457, 109692 (2021) | 10.1016/j.ecolmodel.2021.109692 ; 공개 사본 https://data.imas.utas.edu.au/attachments/a81f6b58-2942-451a-a563-321aea49545a/AAP4501_HIMI_Demersal_Fish_HMSC/Manuscript/Papers%20of%20interest/Wadaoux_2021.pdf | PDF WebFetch 저장본 · pdftotext | Fig 1 캡션 전문, 방법 문단 |
| 23 | Roberts, D. R. et al. Ecography 40, 913–929 (2017) | 10.1111/ecog.02881 ; 공개 사본 https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf | PDF WebFetch 저장본 · pdftotext | Fig 1–4 캡션 |
| 24 | Ploton, P. et al. Nat. Commun. 11, 4540 (2020) | 10.1038/s41467-020-18321-y | 로컬 PDF | Fig 4, 5 캡션 |
| 25 | Meyer, H., Pebesma, E. Nat. Commun. 13, 2208 (2022) | 10.1038/s41467-022-29838-9 | 로컬 PDF | Fig 1 캡션 |
| 26 | Meyer, H., Pebesma, E. Predicting into unknown space? Methods Ecol. Evol. 12, 1620–1633 (2021) | 10.1111/2041-210X.13650 (로컬은 arXiv:2005.07939) | 로컬 PDF | Fig 3, 4, 9 캡션 |
| 27 | Milà, C., Mateu, J., Pebesma, E., Meyer, H. Methods Ecol. Evol. 13, 1304–1316 (2022) | 10.1111/2041-210X.13851 | 로컬 PDF | Fig 1–8 캡션 첫 줄 |
| 28 | Linnenbrink, J. et al. GMD 17, 5897–5912 (2024) | 10.5194/gmd-17-5897-2024 | 로컬 PDF | Fig 5, 7 캡션 |
| 29 | Khanal et al. Mapping soil organic carbon stocks in Nepal's forests. Sci. Rep. 13, 8090 (2023) | 10.1038/s41598-023-34247-z | 로컬 PDF | Fig 2–5, 7 캡션 |
| 30 | Feeney, C. J. et al. Multiple soil map comparison ... Sci. Rep. 12 (2022) | 10.1038/s41598-022-05476-5 | 로컬 PDF | Fig 1–4 캡션, Taylor 문단 |
| 31 | Feng, D. et al. HESS 27, 2357–2373 (2023) | 10.5194/hess-27-2357-2023 | 로컬 PDF | Fig 1, 3, 6 캡션 |
| 32 | Staudinger, M. et al. HESS 29, 5005–5029 (2025) | 10.5194/hess-29-5005-2025 | 로컬 PDF | Fig 3–5 캡션 |
| 33 | Willard, J. D. et al. Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. WRR (2021, 수락 원고) | arXiv:2011.05369 ; WRR DOI [미확인] | 로컬 PDF | Fig 3, 7 캡션 |
| 34 | Tsai, W.-P. et al. Nat. Commun. 12, 5988 (2021) | 10.1038/s41467-021-26107-z | 로컬 PDF | Fig 7 캡션 |
| 35 | Hollmann, N. et al. Nature 637, 319–326 (2025) | 10.1038/s41586-024-08328-6 | 로컬 PDF | Fig 4, 5 캡션 |
| 36 | Chance, R., Ahajjam, A., Putkonen, J., Pasch, T. Earth Sci. Inform. 17, 6055–6073 (2024) | 10.1007/s12145-024-01486-1 | 로컬 PDF | Fig 8, 9 캡션 |
| 37 | O'Malley, D. et al. In-context learning enables continental-scale subsurface temperature prediction. arXiv (2026) | arXiv:2605.16665 | 로컬 PDF | Fig 1–6 캡션 |
| 38 | Betti et al. Mapping on a budget. arXiv (2025); Lin et al. Fine-scale soil mapping in Alaska (MiSo). arXiv (2025) | arXiv:2509.03749 ; arXiv:2506.17302 | 로컬 PDF | Fig 4 · 5(Betti), Fig 4 · 7(Lin) 캡션. 본문에는 참고만 |
| 39 | Ludwig, M. et al. Assessing and improving the transferability of current global spatial prediction models. Glob. Ecol. Biogeogr. 32, 356–368 (2023) | 10.1111/geb.13635 | 검색 결과 서지만 | 그림 [미확인], 본문에 사용하지 않음 |

가져오지 못한 것: nature.com 은 쿠키 리디렉션 뒤 캡션 제목 문장만 반환했다(Nitzbon 2024). Wiley(Roberts 2017 원문 페이지)는 403 이었고 대학 저장소 사본으로 대체했다. Yi 2018 은 출판사 PDF 가 10 MB 를 넘어 WebFetch 가 거부해 scratchpad 에 내려받아 읽었다. 시베리아 2023–2026 ML ALT 지도 논문과 Westermann 제품 논문은 검색으로 특정하지 못했다.
