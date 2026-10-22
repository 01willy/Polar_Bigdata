# 상위 학술지 그림 설계 사례집 4차 (2026-10-22)

목적: 원고 v15 의 그림 7종(Fig 1 자료 개관, Fig 2 워크플로, Fig 3 정확도 · 전이, Fig 4 조건 · 사다리, Fig 5 라벨 수, Fig 6 제품 비교, Fig 7 지도)을 다듬기 위해, 2021–2026 상위 학술지에서 색이 절제되고 패널당 메시지가 하나인 그림을 유형별로 새로 모은다. 선행 문서(`docs/research/2026-10-04/alt_figure_exemplars.md`, `docs/research/2026-10-21/map_and_figure_practice.md`)에 이미 있는 논문(Hjort, Karjalainen, Biskaborn, Aalto, Nitzbon, Langer, Burke, Talucci, Wei, Ploton, Meyer, Tsai, Shen, Natali, Liu, Yi, Streletskiy, Wadoux, Milà, Linnenbrink, Khanal, Feng, Staudinger, Willard 2021, Hollmann, Chance, O'Malley, Feeney, Guo, Du, Zhao)은 다시 다루지 않는다.

확인 방법: 캡션은 출판사 HTML(nature.com, Copernicus) 또는 공개 PDF(저장소 · arXiv, `pdftotext`)에서 전문을 읽었다. 표의 그림 열에 '이미지' 라 적은 21개 그림은 출판사 그림 파일을 내려받아 직접 보았고, '캡션' 이라 적은 그림은 캡션으로만 확인했다. 색 · 기호 서술은 그 범위 안에서만 적었다. [미확인] = 캡션 · 이미지 어느 쪽으로도 확인하지 못한 항목. 영구동토 밖 분야(수문, 산림, 토양, 기후)의 그림이 많은데, 평가 설계 · 라벨 수 · 제품 비교 그림의 선례가 그쪽에 더 많기 때문이다.

---

## 1. 요지

1. 새로 모은 사례는 논문 30편, 그림 40여 개이다(유형별 4–8개). 상위 학술지 본문 그래프의 전형은 '계열 2–3색 + 회색 맥락' 이다. 흑백만 쓴 제품 비교 산점(Araza 2022 Fig 4), 패널당 1색과 회색 기준선(Rolf 2021 Fig 3c), 2색 상자그림과 기준 파선(Nearing 2024 Fig 2), 한 색조 3단계 성적표(Mudryk 2025 Fig 8)가 대표이다.
2. 자료 개관(유형 1)은 지도 한 장에 기간 · 수량 패널을 붙이는 구성이 표준이다. 연도 × 기법 누적 막대(Virkkala 2022 Fig 6), 활성 지점 수 시계열(Pallandt 2022 Fig 2, Kratzert 2023 Fig 3), 변수 × 연도 기록 타임라인(Boike 2019 Fig 4), 라벨의 기후 공간 분포(Araza 2022 Fig 2)가 우리 Fig 1 보강안(지역 × 연도 행렬)과 직접 대응한다.
3. 물리 · ML 결합 도식(유형 2)에서 Nature 계열은 상자를 쓰되 색을 역할에만 쓴다. ML 블록 1색 + 물리 블록 회색 + 실자료 썸네일(Reichstein 2019 Fig 4), 화살표 색이 '과정 모형 전용 · 하이브리드 추가 · 공통' 을 뜻하는 구성(Koppa 2022 Fig 1), 기호 색이 '학습 계수 · 조정 상수 · 강제' 를 뜻하는 구성(Kraft 2022 Fig 1)이 우리 Fig 2b(Stefan + 잔차 ML)에 쓸 문법이다.
4. 검증 설계 · 전이(유형 3)와 라벨 수(유형 4)의 공통 문법은 '같은 y 축에 설계를 나란히 + 기준선 1개' 이다. 무작위 대 블록 CV 를 좌우에 둔 상자그림(Kattenborn 2022 Fig 3), 분할 블록 크기에 따른 성능과 보간 기준선(Rolf 2021 Fig 3c), 학습 표본 수 곡선과 수평 기준선 · 띠(Kratzert 2024 Fig 6), 국가 간 대 국가 내 홀드아웃을 두 열로 둔 r² 눈금 그림(Yeh 2020 Fig 3a–b)이 우리 Fig 3 · Fig 5 의 직접 선례이다.
5. 제품 비교(유형 5)와 지도(유형 6)에서는 관측 대 제품을 구간 평균 + 수염 + 표본 수 크기의 단색 기호로 두고(Araza Fig 4, Santoro 2021 Fig 5), 제품 간 차이는 '기준 제품 대비 %' 나 '(새 지도 − 제품) 지점 차이 지도' 로 둔다(Kochkov 2024 Fig 2, Koppa 2022 Fig 3). 지도는 발산 색표 1개 + 회색 마스크 2단계(Webb 2022 Fig 1, Berner 2020 Fig 4), 중앙 지도 + 같은 크기 삽입도(Jasechko 2024 Fig 1)가 우리 Fig 6 · 7 의 규격이 된다.

---

## 2. 유형별 사례 표

열 설명: 색 사용 = 색상(hue) 수와 회색 사용, 효과 이유 = 메시지가 읽히는 장치, 우리 적용 = 한 문장 제안.

### 2.1 유형 1. 자료 개관(지점 · 지점-연도 · 기간 · 군집)

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Virkkala et al. 2022, ESSD 14, 179 (ABCflux) | Fig 6 (이미지), Fig 2 · 3 (캡션) | a–b 5년 구간별 월 관측 수(한대림 · 툰드라), c–d 월별 관측 수. Fig 2 지도(지점 = 측정 기법) + 현장 사진, Fig 3 국가별 관측 비율 대 면적 비율 | 누적 막대 2 × 2, 1:1 산점 | 기법 3종 = viridis 3색(보라 · 청록 · 노랑), 격자 없음, 패널 머리띠 회색 | 시간 편중과 계절 편중이 같은 y 축 · 같은 색으로 두 행에 놓여 한눈에 비교된다 | Fig 1b–c(또는 제안 B 행렬)를 지역 × 연도 누적 막대로 바꾸고 자료원(CALM · 탐침 · 시추공)을 3색 이하로 둔다 |
| Pallandt et al. 2022, Biogeosciences 19, 559 | Fig 2 (이미지), Fig 1 (캡션) | 월별 활성 EC 지점 수(검정 톱니선)와 여름(주황) · 겨울(하늘색) 장기 포락선. Fig 1 지도: 초록 = CO2, 보라 = CO2 + CH4, 눈송이 = 겨울 측정 | 시계열 선 3개 | 3색, 범례 없이 캡션에서 설명 | 톱니 자체가 겨울 결측을 보여 주고, 포락선 둘이 '여름만 있는 관측망' 을 수치로 남긴다 | Fig 1c 를 연도별 '가용 지점 수(회색)' 와 '분석 사용 지점 수(지역색)' 두 선으로 둔다 |
| Kratzert et al. 2023, Sci Data 10, 61 (Caravan) | Fig 3 (이미지), Fig 1 (캡션) | 좌: 시간에 따른 활성 유역 수를 출처 자료 7종으로 누적, 우: 유역별 기록 길이(연) 누적 히스토그램. Fig 1: 지도 + 기후대 분포 대 육지 비율 | 누적 면적 + 누적 히스토그램 | 범주 7색(많은 편), 바탕 흰색 | '언제 얼마나' 와 '얼마나 오래' 를 두 패널로 나누어 한 그림에 둔다 | 지역 3개를 3색 누적 면적으로 연도별 지점-연도 수를 그리고 기록 길이 분포는 보충으로 보낸다 |
| Jasechko et al. 2024, Nature 625, 715 | Fig 1 (이미지), Fig 2 (캡션) | a 세계 지도(연회색 육지)에 관측정 17만 개를 추세 색으로, b–o 지역 삽입도 14개를 가장자리에 같은 크기로 배치, 삽입도 제목은 지역명 명사구. Fig 2: 대수층 다각형, 안정 = 짙은 회색, 문헌 지역 = 원 | 점 지도 + 삽입도 격자 | 발산 9계급(파랑 · 회색 · 노랑 · 빨강), −0.25~0.25 m/yr 계급은 회색 | 중앙 지도의 문자 원이 삽입도 위치를 가리키고, 중립 계급이 회색이라 변화 없는 지점이 시선을 끌지 않는다 | Fig 1a 를 중앙 극지도 + 3지역 삽입도(같은 폭)로 두고, 지점 색은 기록 연수 또는 잔차의 중립 계급을 회색으로 한다 |
| Boike et al. 2019, ESSD 11, 261 | Fig 4 (캡션; 2019, 기간 밖) | 1998–2017 변수별 기록 타임라인. 초록 = 지상 센서, 갈색 = 지중 센서, 진한 색 = 이번 자료, 옅은 색 = 이전 자료 | 변수 × 연도 간트형 막대 | 2색 × 2명도 | 결측 구간이 빈칸으로 바로 보이고, 자료의 출처 판이 명도 하나로 구분된다 | Fig 1c 자료 기간 막대를 '사용 연도 진한 색 + 가용 기간 옅은 색' 두 단계로 두는 v12 계획의 근거로 쓴다 |
| Poggio et al. 2021, SOIL 7, 217 (SoilGrids 2.0) | Fig 3 · 4 (캡션) | 70,000 km² 격자당 관측 수 지도(변수별 1장) | 격자 밀도 지도 | [미확인] | 점 수만 개 대신 격자 수로 군집을 보여 라벨 편중을 지도에서 읽게 한다 | 제안 B 의 b(1 km 셀당 지점 수)를 0.5° 격자당 지점-연도 수 지도로도 검토한다 |
| Araza et al. 2022, RSE 272, 112917 | Fig 2 (캡션) | 플롯 분포를 (a) 국가 · 수관 피복 > 10 % 지도, (b) Whittaker 생물군계(강수 × 기온) 공간, (c) 인구밀도(log2) × 수관 피복 산점의 세 패널로 | 지도 + 기후 공간 산점 + 교란 산점 | [미확인] | 라벨이 지리 · 기후 · 교란 공간 어디에 있는지를 세 좌표계로 나누어 대표성을 따로 보인다 | 지역별 라벨을 TDD × 적설(또는 MAAT × 강수) 공간에 찍어 레나 · 알래스카 · 캐나다의 공변량 범위 차이를 Fig 1 보충으로 둔다 |

### 2.2 유형 2. 물리 · ML 결합 워크플로 도식

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Reichstein et al. 2019, Nature 566, 195 | Fig 4 (이미지), Fig 3 (캡션) | a 입력 → 단순 특징 → 추상 특징 → 물리 모형 매개변수 → 물리 출력의 세로 사슬, 오른쪽 회색 상자가 '학습 사상 1 … n + 2' 와 '물리 모형' 을 구분, 괄호 n · m. b 온도장 썸네일 → 회색 사다리꼴 CNN → 운동장 → 물리 왜곡 모형. c 강수 · 복사 · 기온 시계열 썸네일 → RNN → 물리 확산 · 에너지 수지 → 증산 · 광합성 시계열. Fig 3: 하위 모형 · 매개변수 · 강제 사이 ML 개입 지점을 원 번호로 | 블록 사슬 + 실자료 썸네일 | ML 블록 청록 1색, 물리 블록 회색, 썸네일 고유색 | 'ML 이 어디까지, 물리가 어디부터' 가 색 둘로 읽히고 입출력은 썸네일이 설명하므로 문장 상자가 없다 | Fig 2b 모델 행을 입력 썸네일 → ML 상자(1색) → Stefan 상자(회색) → ALT 지도 순으로 두고 잔차 경로만 강조색으로 한다 |
| Kochkov et al. 2024, Nature 632, 1060 (NeuralGCM) | Fig 1 (이미지) | a 강제 · 잡음 · 입력(지구본 썸네일) → 학습 인코더 → 상태 x_t → 역학 핵(노랑) 과 학습 물리(초록) → 두 경향이 ODE 적분기에서 합류 → 디코더 → 출력, 점선 '반복 n 회'. b 학습 물리 모듈(기둥 단위 NN) | 흐름 도식 + 썸네일 | 파스텔 상자 5색, 화살표 회색, 반복 고리 점선 | 물리와 학습이 각자 '경향' 을 내고 한 점에서 합쳐지는 구조가 화살표 합류로 드러나고, 시간 반복은 점선 하나로 끝난다 | Fig 2b 에서 Stefan 항과 잔차 항이 ⊕ 로 합류하는 구조를 같은 문법으로 그리고, 연도 반복은 점선 고리로 둔다 |
| Koppa et al. 2022, Nat Commun 13, 1912 | Fig 1 (이미지) | 입력(파란 원) → Gash(초록) · Priestley–Taylor(빨강) → 'E_i + E_p × S = E' 식 상자. 좌 초록 영역 = 증산 스트레스 DL, 우 빨강 영역 = 토양 수분 모듈. 아래: 학습 발자국(지점) 대 모형 격자(0.25°, 식생 비율) | 블록 흐름 + 발자국 도식 | 상자 4색, 화살표 3색(빨강 = 과정 모형 전용, 초록 = 하이브리드 추가, 검정 = 공통) | 화살표 색이 '무엇이 바뀌었는가' 를 설명해 과정 모형과 하이브리드의 차이가 한 그림에서 읽힌다. 상자 수가 많은 점은 약점 | Fig 2b 화살표를 3종(Stefan 전용 회색, 잔차 ML 추가 강조색, 공통 검정)으로 제한하고 지점 라벨 대 1 km 격자의 발자국 차이를 작은 도식으로 덧붙인다 |
| Kraft et al. 2022, HESS 26, 1579 | Fig 1 (이미지) | (a) 입력(기상 강제 시계열 점, 정적 변수 벡터 d = 30), (b) 신경망 FCNN → LSTM → FCNN 과 벡터 차원 표기, (c) 수문 모형(저장고 3개와 흐름 식). 회색 범례 상자가 '값 전달 · 순환 전달 · 변수 색 규칙' 을 설명 | 3구획 도식 + 수식 기호 | 색 = 변수 역할(강제 하늘색, 학습 계수 빨강, 조정 상수 파랑, 저장고 보라) | 수식의 어느 기호가 학습되는지 색만으로 읽히고 문장은 범례 상자 한 곳에만 있다 | Fig 2a · b 에서 Stefan 식 기호 가운데 학습 · 보정 대상(E, 잔차)만 강조색, 강제(TDD) · 상수는 무채색으로 둔다 |
| Irrgang et al. 2021, Nat Mach Intell 3, 667 | Fig 1 (이미지), Fig 2 (캡션) | 지식 수준이 다른 세 영역(법칙 · 매개변수 모두 앎, 법칙만 앎, 둘 다 모름)을 파스텔 원 3개로, 영역 사이 과정(극한 사상, 에너지 플럭스, 탄소 방출 등)을 가는 검정 화살표에 글자로. Fig 2: 약결합 → 강결합 단계 | 벤 다이어그램형 개념도 | 파스텔 3색, 화살표 검정, 아이콘 없음 | 원 3개와 화살표 글자만으로 '어디에 ML 이 필요한가' 를 구분한다 | 발표용 개념 슬라이드(물리 충분 · 계수 불확실 · 잔차 미지의 3영역)로만 쓰고 원고에는 넣지 않는다 |
| Willard et al. 2022, ACM Comput. Surv. 55, 66 (arXiv 2003.04919) | Fig 3 · 4 (캡션) | Fig 3: 잔차 모델링, f_ML 이 f_PHY 의 오차를 학습하고 최종 예측 = f_PHY + 잔차, 빨강 = 학습 흐름, 파랑 = 시험 흐름. Fig 4: 물리 모형 출력 Y_PHY 와 입력 D 를 함께 ML 에 넣는 하이브리드 | 블록 흐름 2종 | 2색(학습 · 시험) [이미지 미확인] | 학습 단계와 예측 단계를 색 둘로 분리해 '잔차를 무엇으로 학습하는가' 가 명확하다 | Fig 2b 잔차 경로에서 학습(관측 − Stefan)과 예측(Stefan + ML) 흐름을 두 색으로 나눈다 |
| Lang et al. 2023, Nat Ecol Evol 7, 1778 | Fig 1a (캡션) | Sentinel-2 영상 + 좌표 부호화 → CNN → 수관 높이와 예측 분산(공유 특징에서 두 출력), 희소 GEDI 발자국이 감독 신호임을 그림으로 | 모델 도식 + 희소 라벨 삽화 | [미확인] | 희소 점 라벨로 조밀 지도를 학습한다는 설정이 패널 하나에 들어가고, 불확실성 출력이 구조에 포함된다 | Fig 2b 에서 지점-연도 라벨(희소) → 1 km 지도(조밀) 관계를 썸네일 한 쌍으로 보인다 |

보조: Jia et al. 2021(ACM/IMS Trans. Data Sci. 2, 15; arXiv 2001.11086) Fig 2 는 PGRNN 흐름을 '검정 화살표 = RNN 흐름, 파랑 = 에너지 흐름' 으로 그린다(캡션). Karniadakis et al. 2021(Nat Rev Phys 3, 422) Fig 1 은 물리를 구조에 심은 신경망 예(공변 구성 네트워크, Lax–Oleinik 식) 이고 '물리 많음 · 자료 적음' 스펙트럼 그림은 Box 안에 있다 [미확인]. Kashinath et al. 2021(Phil Trans R Soc A 379, 20200093) Fig 5 는 보존 층 n 개를 덧붙인 구조 제약 NN 도식이다(Beucler 재수록, 캡션).

### 2.3 유형 3. 검증 설계별 정확도 · 공간 전이 · 무작위 CV 의 낙관

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Kattenborn et al. 2022, ISPRS Open J. Photogramm. Remote Sens. 5, 100018 | Fig 3 (이미지), Fig 4 (이미지) | 좌 '비독립 시험(무작위 CV)', 우 '독립 시험(블록 CV)', 각각 학습 정사영상 수 n = 10 · 25 · 40 × 증강 4종의 F1 상자그림, 회색 점 = 개별 지점. Fig 4: 지점 내 · 지점 간 다변량 상관도(파랑 = 예측 변수, 회색 = 반응) | 상자그림 좌우 대비 + 상관도 | 상자 4색(흰 · 파랑 · 보라 · 회색), 점 회색 | 같은 y 축에 좌우로 두어 낙관이 높이 차이로 바로 읽히고, 상관도가 원인(공간 자기상관)을 덧붙인다 | Fig 3a 의 무작위 대 블록 CV(제안 I)를 좌우 같은 축으로 두고, 보충에 거리별 잔차 상관도를 둔다 |
| Rolf et al. 2021, Nat Commun 12, 4392 (MOSAIKS) | Fig 3c (이미지) | 체스판 분할 폭 δ(0.5°–16°)에 따른 R², 과제 7개 소형 패널(패널당 1색), 회색 파선 = RBF 보간 기준선(회색 띠), δ = 0 벤치마크 원, 아래 체스판 지도 3장(검정 = 학습, 흰색 = 검증) | 거리 곡선 소형 다중 + 분할 도식 | 패널당 1색 + 회색, 분할 지도 흑백 | 패널마다 색 하나와 회색 기준선뿐이라 '보간보다 나은가' 가 바로 읽히고, 분할 방식은 지도 3장이 설명한다 | 제안 G(학습 지점 거리별 오차)나 Fig 3b 를 '블록 크기 → 오차' 곡선으로 그리고 Stefan · 보간 기준선을 회색 파선, 분할 도식을 아래 작은 지도로 둔다 |
| Yeh et al. 2020, Nat Commun 11, 2583 | Fig 3a–b (이미지), Fig 2c–d (캡션) | 모형 6종 × (a 국가 홀드아웃, b 국가 내 마을 홀드아웃)의 r²: 회색 세로 눈금 = 국가-연도별 값, 검정 눈금 = 평균, 빨강 눈금 = 통합 r², 값을 숫자로 직접 표기. Fig 2c–d: 국가별 r² | 1차원 눈금 그림(strip) 2열 | 회색 + 검정 + 빨강 | 분포(회색) · 평균(검정) · 통합(빨강) 세 층이 한 줄에 들어가고, 전이(국가 간)와 내삽(국가 내)이 두 열로 대비된다 | Fig 3b 설계 Δ 를 '설계별 한 줄 = 블록별 회색 눈금 + 중앙값 검정 + 전체 빨강' 으로, 지역 홀드아웃 대 지역 내 블록을 두 열로 둔다 |
| Nearing et al. 2024, Nature 627, 559 | Fig 2 (이미지), Fig 4 (캡션) | 재현기간(1 · 2 · 5 · 10년)별 정밀도(a) · 재현율(b) 상자그림, AI 모형 주황 대 GloFAS 파랑, 파란 파선 = GloFAS 1년 사건 중앙값 기준선, x 눈금 라벨에 N(관측소 수). Fig 4: 대륙별 같은 구성 | 2계열 상자그림 + 기준 파선 | 2색, 기준선은 기준 계열 색의 파선 | 색이 둘뿐이고 기준선이 하나라 '미계측 유역에서도 기준을 넘는가' 가 패널마다 하나의 판정으로 읽힌다. 표본 수가 축 라벨에 있어 신뢰도를 같이 읽는다 | Fig 4 사다리(설계별 분포)에서 Stefan 중앙값을 파선으로 두고 각 설계의 지점-연도 수를 눈금 라벨에 적는다 |
| Kratzert et al. 2024, HESS 28, 4187 | Fig 3 (캡션) | 531개 유역 NSE 의 누적분포: 단일 유역 학습(주황) 대 지역 학습(파랑), a 개념 · 과정 모형, b LSTM | 누적분포(CDF) 2패널 | 2색 [이미지 미확인] | 지점별 지표를 CDF 로 두면 평균 하나보다 분포 전체의 우열과 꼬리가 보인다 | Fig 5c 오차 분포(또는 보충)를 지역별 CDF 로 두는 대안을 검토하고 기준 모형은 회색으로 둔다 |
| Stock, Gregr & Chan 2023, Nat Ecol Evol 7, 1743 | Fig 1 (캡션) | 학습 · 시험 분할(위)과 실제 생태 적용(아래)을 한 도식에 두고, 누설 때문에 시험 정확도가 적용 정확도와 어긋나는 경로를 설명 | 개념 도식 | [미확인] | 평가가 적용 상황을 흉내 내야 한다는 원리를 그림 하나로 말한다 | Fig 2c 검증 설계 지도 5장의 캡션에 '지점-연도 누설 차단' 원리를 한 줄로 적는 근거로 쓴다 |

### 2.4 유형 4. 라벨 수 · 전이 학습 곡선

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Kratzert et al. 2024, HESS 28, 4187 | Fig 6 (이미지) | 학습 유역 수(0–531)에 따른 중앙 NSE: 파란 점선 + 점 + 오차 막대(무작위 분할), 주황 수평선 + 옅은 띠 = HUC 분할, 초록 수평선 + 띠 = 속성 군집 분할, 각 분할의 결과를 삼각형 · 사각형으로 해당 유역 수 위치에 | 학습 곡선 + 수평 기준선 · 띠 | 3색, 띠는 같은 색 옅게 | 곡선이 수평 기준선 · 띠를 넘는 지점이 '몇 개면 충분한가' 를 그대로 준다 | Fig 5a 에 Stefan 과 라벨 0 전이 기준을 수평선 + 띠로 두고 교차점 n* 를 눈금으로 표시한다(현행 계획과 일치) |
| Rolf et al. 2021, Nat Commun 12, 4392 | Fig 3b (이미지) | R² 대 특징 수 K(log)와 학습 표본 수 N(log), 과제별 선 7개, 띠 = 5겹 범위 | 로그 x 학습 곡선 2패널 | 과제별 7색(Fig 3a · c 와 공유), 범례는 c 에만 | 로그 축에 띠가 겹 범위를 보이고 색이 그림 전체에서 같아 범례가 한 번이면 된다 | Fig 5a x 축을 로그(1 · 3 · 10 · 40 · 160), 띠 = 블록 부트스트랩 사분위, 지역 색을 전 그림과 공유한다 |
| Yeh et al. 2020, Nat Commun 11, 2583 | Fig 3c · e (이미지) | c 학습에 쓴 자료 비율(%)에 따른 홀드아웃 국가 r², 선 4개의 이름을 선 끝에 직접 표기. e 부(富) 분위 누적별 r², 선 6개 끝 라벨 | 학습 곡선, 끝 라벨 | 4–6색, 범례 없음 | 선 끝 라벨이 범례를 없애고 포화점(0.25 이후 완만)이 바로 보인다 | Fig 5a 곡선 이름을 선 끝 직접 라벨로 두고 범례 상자를 없앤다 |
| Pallandt et al. 2022, Biogeosciences 19, 559 | Fig 6 (캡션) | 지점 1–25개 추가에 따른 관측망 대표성 개선 %, 주황 = 단계적 최적 선택, 파랑 점선 = 무작위 순서 추가 | 예산 대 개선 곡선 2개 | 2색 [이미지 미확인] | 관측망 설계의 가치를 '같은 예산에서 무작위보다 얼마나 나은가' 한 곡선으로 말한다 | Fig 5 또는 보충의 다음 관측 위치 실험(전략별 예산 대 RMSE)에 무작위 추가를 회색 기준 곡선으로 둔다 |

보조: Kashinath et al. 2021 Fig 4a 는 학습 표본 크기별 시험 RMSE(물리 유도 NN 대 NN, Daw et al. 재수록)로 '소수 라벨에서 물리 유도의 이득' 진술의 인용 근거이다(캡션, 설계 참고 가치는 낮음). Jia et al. 2021 Table 3 은 미세조정 라벨 비율(0 · 0.2 · 2 · 20 · 100 %)을 표로 둔다. Read et al. 2019(WRR 55, 9173) 의 학습 프로파일 수 대 RMSE 그림은 출판사 접근이 막혀 번호 · 구성을 확인하지 못했다 [미확인].

### 2.5 유형 5. 새 지도와 기존 제품의 비교

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Araza et al. 2022, RSE 272, 112917 | Fig 4 (이미지), Fig 5 · 6 (캡션) | 지도 4종(Baccini 2000, GEOCARBON 2008, GlobBiomass 2010, CCI 2017) × 0.1° 격자 평균 플롯 AGB 대 지도 AGB, AGB 구간별 검정 점(크기 = 격자 수), 수염 = 25–75 %, 점선 1:1, 2 × 2 공유 축. Fig 5: 지도별 편향과 95 % CI 지도. Fig 6: 네 지도의 편향 부호 일치 · 불일치 지도 | 구간 평균 산점 행렬 + 편향 지도 | 검정만, 색 없음 | 색 없이 구간 평균 · 산포 · 표본 수가 한 기호에 들어가고, 제품별 포화 구간이 1:1 이탈로 읽힌다 | Fig 5b 3 × 3 행렬과 Fig 6 산점을 구간 평균 + 수염 + 크기 = n 의 단색 기호로 단순화하고, 제품 간 편향 부호 일치 지도는 보충으로 둔다 |
| Santoro et al. 2021, ESSD 13, 3927 (GlobBiomass) | Fig 5 (이미지), Fig 10 · 11 (캡션) | a–b 현장 · 지도 AGB 히스토그램(검정 선), c 0.1° 격자 산점(회색 빈 원) + 10 Mg/ha 구간 중앙값(채운 원, 색 = 표본 수), 1:1 점선, 상대 RMSD · 편향 글자, d–g 생태 지대별 같은 구성. Fig 10: 위도별 평균 프로파일과 다른 지도 대비 상대 차이. Fig 11: 지도별 상대 차이 지도 + 위도대별 격자 수 | 회색 산점 + 구간 중앙값, 층별 소형 다중 | 회색 + 순차 램프 1개 | 전체(큰 패널)와 층별(작은 패널 4개)이 같은 문법이고, 개별 점은 회색이라 구간 중앙값이 앞에 선다 | Fig 6 산점에 회색 개별 점 + 구간 중앙값을 두고, 영구동토 구역별 분해는 보충에 같은 문법으로 둔다 |
| Mudryk et al. 2025, TC 19, 201 (SnowPEx+) | Fig 8 (이미지), Fig 3 · 6 · 7 (캡션) | 제품 23종을 순위순 누적 막대(추세 연초록, 변동 초록, 기후값 진초록), 순위 묶음(1–5, 6–10 …) 회색 세로선, 오른쪽에 순위 목록 두 열(전체 시험, 추세 제외). Fig 3 · 5: Taylor · target 도표, 회색 곡선 = 50 · 90 백분위. Fig 7: 지역 × 강제 자료 격자, 회색 음영 = 평가 앙상블 범위, 칸 안 점수 숫자 | 성적표 누적 막대 + 순위 목록 | 한 색조 3단계 + 회색 | 순위 근거(항목별 기여)가 막대 안에 있고, 두 순위 목록이 '추세 평가가 순위를 바꾸는가' 를 글자로 답한다 | Fig 6 아래 통계 표를 보충에서 지역 × 제품 성적표(지표별 점수 누적 막대 또는 숫자 열지도)로 확장한다 |
| Tolan et al. 2024, RSE 300, 113888 (arXiv 2304.07213) | Fig 12 (이미지), Fig 7 · 8 · 9 (캡션) | Fig 12: 하위 지역 20여 곳의 UA · PA, 신뢰구간 80 · 90 · 95 % 를 겹친 막대(파랑 PA, 빨강 UA), x 눈금 라벨에 표본 수. Fig 7: 입력 영상 · 본 연구 CHM · Lang 2022 · Potapov 2021 네 열 나열. Fig 8: 검증 세트별 블록 집계 밀도 산점. Fig 9: GEDI 대비 밀도 산점과 잔차 대 예측 | 제품 열 나열 + 구간 막대 + 밀도 산점 | 2색 × 3명도 | 열 = 제품, 행 = 장소의 나열이 상세도 차이를 직접 보이고, 구간 3겹이 표본 수 효과를 보인다 | Fig 6 지도 행(열 = 제품)은 유지하고, 지역별 RMSE 점에 80 · 95 % 블록 부트스트랩 구간 2겹과 n 눈금 라벨을 둔다 |
| Koppa et al. 2022, Nat Commun 13, 1912 | Fig 2 · 3 (캡션) | Fig 2: 하이브리드 대 과정 모형의 KGE 바이올린(짧은 · 키 큰 식생), 빨간 선 = KGE −0.41(계절 평균보다 나은 경계). Fig 3: 플럭스 타워 · 수액 흐름 지점 지도에 KGE 차이(파랑 = 개선, 빨강 = 악화), 지역 4묶음 | 바이올린 + 기준선, 지점 차이 지도 | 발산 2색 [이미지 미확인] | '기준보다 나은가' 는 선 하나로, '어디서 나아졌는가' 는 차이만 그린 지도로 답한다 | Fig 6 · 7 보충에 (우리 − 제품) 지점 오차 차이 지도(파랑 개선 · 빨강 악화)를 두고 Stefan 대비 기준선을 분포 패널에 넣는다. Nearing 2024 Fig 1(AI − GloFAS F1 차이 지도, 관측소 3,673개)도 같은 유형이다 |
| Kochkov et al. 2024, Nature 632, 1060 | Fig 2 (캡션) | a · c RMSE · RMSB 를 ECMWF-ENS 오차의 % 로 정규화해 모형 6종을 리드타임별로, 결정론 실선 · 확률 파선. b · d · f · h: 공간 분포 지도, 회색 = 지표면 기압이 700 hPa 미만인 영역(마스크) | 기준 제품 정규화 선 그래프 + 기량 지도 | [이미지 미확인], 마스크 회색 | 기준 제품을 100 % 로 두면 제품 간 비교가 축 하나로 끝나고, 비교 불가 영역은 회색으로 비운다 | 제품별 RMSE 를 Stefan(또는 Wei 2026) 대비 비율로 그린 보조 패널을 검토하고, Fig 7 의 적용 영역 밖은 회색 마스크로 둔다 |

보조: Lang et al. 2023 Fig 1b–d 는 높이 구간별 잔차 상자그림(구성요소 제거 실험 포함), 혼동 그림, 생물군계별 잔차와 표본 수를 한 그림에 둔다(캡션). ALT 구간별 · 구역별 제품 잔차를 보충에 둘 때의 선례이다.

### 2.6 유형 6. 지도(다지역 배치 · 차이 · 추세 · 마스크)

| 논문 | 그림 | 패널 내용 | 차트 유형 | 색 사용 | 효과 이유 | 우리 적용 |
|---|---|---|---|---|---|---|
| Webb et al. 2022, Nat Clim Change 12, 841 | Fig 1 (이미지), Fig 2 (캡션) | 북미 · 유라시아 두 패널(같은 투영군, 같은 폭), 12 km 화소의 2000–2021 지표수 추세, 색 막대는 아래 가로 하나. 연구 지역 밖은 회색, 바다 청회색, 격자선 흰색. Fig 2: 추세를 영구동토 범위 · 지하 얼음 · 호수 피복 구간별 상자그림 · 평균으로 | 2패널 추세 지도 + 구간별 상자 | 발산 2색(갈색 · 청록), 맥락은 무채색 | 색이 둘뿐이고 맥락 요소가 무채색이라 추세 부호가 지도에서 유일한 색 정보가 된다 | Fig 7 추세 · 편차 지도에 발산 색표 1개 + 적용 영역 밖 회색 마스크를 쓰고 3지역 패널 폭을 같게 한다 |
| Berner et al. 2020, Nat Commun 11, 4621 | Fig 4 (이미지), Fig 1 · 3 (캡션) | 2행(1985–2016, 2000–2016) × 4열(양 추세 비율, 음 추세 비율, SWI 추세, 상관) 극지 소형 다중, 열마다 5계급 색 막대를 머리에, 열별 색조 1개, 짙은 회색 = 불모지, 검정 = 자료 없음. Fig 1b–c: 50 km 격자당 표본 지점 수. Fig 3: 구역별 추세 비율 막대, 표본 수를 막대 위에 | 소형 다중 지도 + 열 머리 색 막대 | 열별 1색조(초록, 빨강, 노랑 · 주황 · 빨강, 자홍 · 파랑), 마스크 회색 · 검정 | 열 머리 색 막대 하나로 열 전체가 읽히고, 마스크 2단계가 '자료 없음' 과 '해당 없음' 을 구분한다 | Fig 7 소형 다중(제안 E 의 선택 연도 4장)에서 열 머리 색 막대를 공유하고 '자료 없음 · 적용 밖' 마스크를 2단계로 둔다 |
| Smith et al. 2022, Nat Rev Earth Environ 3, 10 | Fig 1 (이미지), Fig 3 (캡션) | 극 투영 지도, 영구동토 4구역을 파랑 4단계로, 지온 추세를 원 크기로(주황 = 전 기록, 빨강 = 2000–2019, 겹친 원 = 범위), 빨간 사각 = 지역 평균, 바다 지명 이탤릭, 범례 상자 2행. Fig 3: 지역별 ALT 편차 시계열 8패널, SD 음영, 'y 축 범위가 다름' 명시 | 비례 기호 지도 + 지역 시계열 | 파랑 4단계 + 주황 · 빨강 | 기간 2개는 색, 크기는 원 크기, 지역 평균은 테두리로 나누어 한 지도에 네 정보가 겹치지 않는다 | Fig 1a 지점 기호를 크기 = 기록 연수, 색 = 자료원 2종으로 두고, Fig 7d 지역 편차 시계열에 SD 띠와 y 축 범위를 명시한다 |
| Jasechko et al. 2024, Nature 625, 715 | Fig 1 (이미지), Fig 2 (캡션) | 중앙 세계 지도 + 가장자리 삽입도 14개(같은 크기, 제목 = 지역명), 중앙 지도에 문자 원으로 삽입도 위치. Fig 2: 대수층 다각형 1,693개, 안정 = 짙은 회색, 심화 = 노랑 · 주황 · 빨강, 회복 = 파랑, 문헌 지역 = 원 | 중앙 지도 + 삽입도 격자, 다각형 지도 | 발산 색 1벌, 중립 · 안정 계급 회색 | 삽입도가 모두 같은 크기라 지역 간 비교가 공정하고, 변화 없는 영역이 회색이라 변화가 앞에 선다 | Fig 1a · Fig 7 의 3지역 삽입도를 같은 크기로 두고 '변화 없음' 계급을 회색으로 둔다 |
| Runge et al. 2022, RSE 268, 112752 | Fig 1 · 5 · 7 · 8 (캡션) | Fig 1: 북시베리아 개관 지도 + 초점 지역 9개(a–i) 삽입도, 바탕 = 영구동토 구역 · Yedoma · 생태계. Fig 5: 3지역 열 × (위 첫 · 마지막 활동 연도 막대 빨강 · 파랑, 아래 연간 면적) 행 정렬. Fig 7: 40 km 격자당 RTS 밀도 지도 + 구역 지도. Fig 8: 요약 4패널(활동 연도, 위도별 빈도, 연간 면적과 추세선, 크기 분포) | 개관 + 삽입도, 지역 열 정렬 소형 다중, 밀도 격자 지도 | 2색(빨강 · 파랑) [이미지 미확인] | 지역을 열, 지표를 행으로 정렬해 지역 간 비교가 열 단위로 끝난다 | Fig 7 을 3지역 열 × (평균 지도, 추세 지도, 편차 시계열) 행으로 정렬한다 |
| Rolf et al. 2021, Nat Commun 12, 4392 | Fig 2 (캡션) | 과제 7개 행 × (좌 학습 라벨 지도, 우 5겹 CV 예측 지도, 산점) 열, 표시용으로 20 km 집계, 1:1 검정 선, 회색 = 표본 없는 영역 | 관측 지도 · 예측 지도 · 산점 행렬 | [이미지 미확인], 비표본 영역 회색 | 관측과 예측을 같은 색 범위로 나란히 두고 산점이 수치를 보탠다. 표시용 집계임을 캡션에 적는다 | Fig 6 각 지역 행(관측 셀 · 제품 · 우리 · 산점)의 선례로 삼고, 표시용 집계를 쓸 때는 캡션에 명시한다 |

---

## 3. 색 · 글자 · 범례 관행

### 3.1 학술지 지침(원문 확인)

- Nature 형식 지침(https://www.nature.com/nature/for-authors/formatting-guide, 2026-10-22 접속): 표준 그림 폭 90 mm(1단) · 180 mm(2단). "Figures should be as small and simple as is compatible with clarity." "Unnecessary figures and parts (panels) of figures should be avoided." "Figures should not contain more than one panel unless the parts are logically connected." 글자는 "lower-case type, with the first letter capitalized and no full stop." "Layering type directly over shaded or textured areas and using reversed type … should be avoided." 기호 설명은 "Where possible, text, including keys to symbols, should be provided in the legend rather than on the figure itself." 범례(캡션)는 제목 문장으로 시작하고 300단어 미만, 오차 막대 · 통계는 반드시 정의. 6쪽 논문의 표시 항목은 보통 4개, 다패널 그림은 반 쪽(약 600단어)에 해당.
- Scientific Reports 지침은 선행 문서 8.2절에 원문이 있다(과도한 상자 · 불필요한 색 금지, 산세리프 단일 서체 · 단일 크기, 소문자 굵은 패널 문자, 최소 선 굵기 1 pt, 표시 항목 8개 상한).
- 패널 문자 표기: Nature 계열 그림 안은 굵은 소문자(a, b)이고 캡션은 'a,' 로 받는다(열람한 Nature · Nat Commun · NCC 그림 모두). Nature Reviews 계열은 캡션에서 'a |' 로 받는다(Karniadakis 2021, Miner 2022). Copernicus 계열은 그림 안 · 캡션 모두 '(a)' 이다(Kraft, Virkkala, Mudryk). 우리 원고의 소문자 알파벳 순 패널 문자는 어느 쪽에도 맞는다.

### 3.2 색 지침(원문 확인)

- Wong, B. 2011, Points of view: Color blindness. Nat Methods 8, 441. Fig 2 '색각 이상자를 위해 최적화한 색(8색, Okabe–Ito)' 과 P · D 모의, Fig 3 '빨강 → 자홍 또는 초록 → 청록 대체' 가 권고의 핵심이다. 우리의 채도 낮춘 Okabe–Ito 는 이 팔레트에서 나온다.
- Katsnelson, A. 2021, Colour me better: fixing figures for colour blindness. Nature 598, 224–225(기술 특집, 2021-10-04). 요령 4개: "Do not use rainbows. Use a perceptually uniform colour map, such as viridis or cividis." "Pick a palette … Color Universal Design or Color Blind 10 Palette." "Think bigger. Use features such as shapes and line textures to disambiguate colour." "Test drive. Use a simulator such as Color Oracle or Coblis." 북유럽 남성 12명 중 1명, 여성 200명 중 1명이 색각 이상이라는 수치도 같이 적는다.
- Crameri, F., Shephard, G. E., Heron, P. J. 2020, The misuse of colour in science communication. Nat Commun 11, 5444. Fig 2(색각 이상 · 흑백 모의에서 무지개형만 실패), Fig 5(색표 분류: sequential · diverging · multi-sequential · cyclic × continuous · discrete · categorical), Fig 6(자료 성격에 맞는 색표 선택 지침). 지각 균일 색표(batlow 등)의 근거로 인용한다.

### 3.3 열람한 그림에서 읽은 관행(이미지 21개 기준)

1. 색상 수: 본문 통계 그래프는 계열 2–3색이 전형이다(Nearing 2, Webb 2, Pallandt 3, Virkkala 3, Kratzert 3, Kattenborn 3 + 회색). 흑백만 쓴 예(Araza Fig 4)와 한 색조 명도 3단계(Mudryk Fig 8, Tolan Fig 12)도 본문 그림이다. 7색을 쓴 예(Rolf Fig 3, Caravan Fig 3)는 범주가 7개일 때이고, Rolf 는 패널당 1색으로 나누어 부담을 줄였다.
2. 회색의 역할: 개별 관측(Yeh 눈금, Santoro 빈 원, Kattenborn 점), 기준선(Rolf RBF 파선), 맥락 영역(Webb 연구 지역 밖, Berner 불모지, Kochkov 마스크), 안정 계급(Jasechko)에 회색을 쓴다. 강조는 1–2색에만 남긴다.
3. 기준선: 기준 계열 색의 파선(Nearing), 실선 + 옅은 띠(Kratzert), 회색 파선 + 회색 띠(Rolf). 교차점이 결론이 되는 그림에서는 기준선이 반드시 같은 패널 안에 있다.
4. 직접 라벨: 선이 끝에서 떨어질 때는 선 끝 라벨(Yeh Fig 3c · e), 막대 묶음은 지시선 라벨(Rolf Fig 3a). 선이 겹치는 소형 다중에서는 범례를 한 패널에만 둔다(Rolf Fig 3c). Nature 지침은 기호 설명을 캡션에 두라고 하므로 그림 안 범례는 작게, 정의는 캡션에 적는다.
5. 표본 수 표기: x 눈금 라벨에 N(Nearing Fig 2, Tolan Fig 12), 막대 위 숫자(Berner Fig 3), 기호 크기 = n(Araza Fig 4, Santoro Fig 5). 제품 비교 · 설계 비교에서 n 이 다르면 반드시 표기한다.
6. 도식: ML 블록 1색 + 물리 회색(Reichstein), 파스텔 상자 + 회색 화살표(Kochkov), 화살표 색 = 변경 여부(Koppa), 기호 색 = 변수 역할(Kraft). 아이콘은 Kochkov 의 지구본 · 구름 정도이며, 문장을 담은 상자는 없다.
7. 지도: 발산 색표 1벌 + 중립 계급 회색(Jasechko, Webb), 열별 색 막대를 머리에 공유(Berner), 삽입도는 같은 크기(Jasechko), 비례 기호는 크기와 색을 다른 변수에(Smith). 바다 · 육지 바탕은 무채색 또는 옅은 파랑 1색이다.

---

## 4. 출처 목록

| # | 서지 | DOI · URL | 확인 방법 | 확인 범위 |
|---|---|---|---|---|
| 1 | Virkkala, A.-M. et al. The ABCflux database. ESSD 14, 179–208 (2022) | 10.5194/essd-14-179-2022 | Copernicus HTML, 그림 파일 | Fig 1–7 캡션, Fig 6 이미지 |
| 2 | Pallandt, M. M. T. A. et al. Representativeness assessment of the pan-Arctic eddy covariance site network. Biogeosciences 19, 559–583 (2022) | 10.5194/bg-19-559-2022 | Copernicus HTML, 그림 파일 | Fig 1–6 캡션, Fig 2 이미지 |
| 3 | Kratzert, F. et al. Caravan. Sci Data 10, 61 (2023) | 10.1038/s41597-023-01975-w | nature.com HTML(curl), 그림 파일 | Fig 1–7 캡션, Fig 3 이미지 |
| 4 | Jasechko, S. et al. Rapid groundwater decline and some cases of recovery in aquifers globally. Nature 625, 715–721 (2024) | 10.1038/s41586-023-06879-8 | nature.com HTML, 그림 파일 | Fig 1–4 캡션, Fig 1 이미지 |
| 5 | Boike, J. et al. A 16-year record … Samoylov Island. ESSD 11, 261–299 (2019) | 10.5194/essd-11-261-2019 | Copernicus HTML(WebFetch) | Fig 1–4 캡션 |
| 6 | Poggio, L. et al. SoilGrids 2.0. SOIL 7, 217–240 (2021) | 10.5194/soil-7-217-2021 | Copernicus HTML(WebFetch) | Fig 1–10 캡션 |
| 7 | Araza, A. et al. A comprehensive framework for assessing the accuracy and uncertainty of global AGB maps. RSE 272, 112917 (2022) | 10.1016/j.rse.2022.112917 ; 공개본 https://eprints.whiterose.ac.uk/id/eprint/183884/ | PDF(pdftotext), 9쪽 렌더링 | Fig 1–6 캡션, Fig 4 이미지 |
| 8 | Reichstein, M. et al. Deep learning and process understanding for data-driven Earth system science. Nature 566, 195–204 (2019) | 10.1038/s41586-019-0912-1 | nature.com HTML, 그림 파일 | Fig 1–4 캡션, Fig 4 이미지 |
| 9 | Kochkov, D. et al. Neural general circulation models for weather and climate. Nature 632, 1060–1066 (2024) | 10.1038/s41586-024-07744-y | nature.com HTML, 그림 파일 | Fig 1–4 캡션, Fig 1 이미지 |
| 10 | Koppa, A. et al. A deep learning-based hybrid model of global terrestrial evaporation. Nat Commun 13, 1912 (2022) | 10.1038/s41467-022-29543-7 | nature.com HTML, 그림 파일 | Fig 1–6 캡션, Fig 1 이미지 |
| 11 | Kraft, B. et al. Towards hybrid modeling of the global hydrological cycle. HESS 26, 1579–1614 (2022) | 10.5194/hess-26-1579-2022 | Copernicus HTML, 그림 파일 | Fig 1–8 캡션, Fig 1 이미지 |
| 12 | Irrgang, C. et al. Towards neural Earth system modelling. Nat Mach Intell 3, 667–674 (2021) | 10.1038/s42256-021-00374-3 | nature.com HTML, 그림 파일 | Fig 1–2 캡션, Fig 1 이미지 |
| 13 | Willard, J. et al. Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Comput Surv 55, 66 (2022) | 10.1145/3514228 ; arXiv:2003.04919 | arXiv PDF(pdftotext) | Fig 1–4 캡션 |
| 14 | Jia, X. et al. Physics-guided machine learning for scientific discovery: lake temperature profiles. ACM/IMS Trans Data Sci 2, 15 (2021) | 10.1145/3447814 ; arXiv:2001.11086 | arXiv PDF(pdftotext) | Fig 1–14 캡션, Table 3 |
| 15 | Lang, N. et al. A high-resolution canopy height model of the Earth. Nat Ecol Evol 7, 1778–1789 (2023) | 10.1038/s41559-023-02206-6 | nature.com HTML | Fig 1–4 캡션 |
| 16 | Kashinath, K. et al. Physics-informed machine learning: case studies for weather and climate modelling. Phil Trans R Soc A 379, 20200093 (2021) | 10.1098/rsta.2020.0093 ; 공개본 https://cfd.me.berkeley.edu/wp-content/uploads/2022/02/rsta.2020.0093.pdf | PDF(pdftotext) | Fig 1–14 캡션 |
| 17 | Karniadakis, G. E. et al. Physics-informed machine learning. Nat Rev Phys 3, 422–440 (2021) | 10.1038/s42254-021-00314-5 | nature.com HTML | Fig 1–5 캡션. Box 그림 [미확인] |
| 18 | Kattenborn, T. et al. Spatially autocorrelated training and validation samples inflate performance assessment of CNNs. ISPRS Open J Photogramm Remote Sens 5, 100018 (2022) | 10.1016/j.ophoto.2022.100018 ; 공개본 https://publikationen.bibliothek.kit.edu/1000156879 | PDF(pdftotext), 5쪽 렌더링 | Fig 1–4 캡션, Fig 3 · 4 이미지 |
| 19 | Rolf, E. et al. A generalizable and accessible approach to machine learning with global satellite imagery. Nat Commun 12, 4392 (2021) | 10.1038/s41467-021-24638-z | nature.com HTML, 그림 파일 | Fig 1–4 캡션, Fig 3 이미지 |
| 20 | Yeh, C. et al. Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. Nat Commun 11, 2583 (2020) | 10.1038/s41467-020-16185-w | nature.com HTML, 그림 파일 | Fig 1–6 캡션, Fig 3 이미지 |
| 21 | Nearing, G. et al. Global prediction of extreme floods in ungauged watersheds. Nature 627, 559–563 (2024) | 10.1038/s41586-024-07145-1 | nature.com HTML, 그림 파일 | Fig 1–6 캡션, Fig 2 이미지 |
| 22 | Kratzert, F. et al. HESS Opinions: Never train an LSTM network on a single basin. HESS 28, 4187–4201 (2024) | 10.5194/hess-28-4187-2024 | Copernicus HTML(WebFetch), 그림 파일 | Fig 1–7 · 부록 캡션, Fig 6 이미지 |
| 23 | Stock, A., Gregr, E. J., Chan, K. M. A. Data leakage jeopardizes ecological applications of machine learning. Nat Ecol Evol 7, 1743–1745 (2023) | 10.1038/s41559-023-02162-1 | nature.com HTML | Fig 1 캡션 |
| 24 | Santoro, M. et al. The global forest above-ground biomass pool for 2010. ESSD 13, 3927–3950 (2021) | 10.5194/essd-13-3927-2021 | Copernicus HTML(WebFetch), 그림 파일 | Fig 1–11 캡션, Fig 5 이미지 |
| 25 | Mudryk, L. et al. Benchmarking of SWE products based on outcomes of the SnowPEx+ Intercomparison Project. TC 19, 201–218 (2025) | 10.5194/tc-19-201-2025 | Copernicus HTML(WebFetch), 그림 파일 | Fig 1–8 캡션, Fig 8 이미지 |
| 26 | Tolan, J. et al. Very high resolution canopy height maps from RGB imagery. RSE 300, 113888 (2024) | 10.1016/j.rse.2023.113888 ; arXiv:2304.07213 | arXiv PDF(pdftotext), 20쪽 렌더링 | Fig 1–17 캡션, Fig 12 이미지 |
| 27 | Webb, E. E. et al. Permafrost thaw drives surface water decline across lake-rich regions of the Arctic. Nat Clim Change 12, 841–846 (2022) | 10.1038/s41558-022-01455-w | nature.com HTML, 그림 파일 | Fig 1–4 캡션, Fig 1 이미지 |
| 28 | Berner, L. T. et al. Summer warming explains widespread but not uniform greening in the Arctic tundra biome. Nat Commun 11, 4621 (2020) | 10.1038/s41467-020-18479-5 | nature.com HTML, 그림 파일 | Fig 1–5 캡션, Fig 4 이미지 |
| 29 | Smith, S. L. et al. The changing thermal state of permafrost. Nat Rev Earth Environ 3, 10–23 (2022) | 10.1038/s43017-021-00240-1 | nature.com HTML(구독 논문, 캡션 · 그림 파일은 공개), 그림 파일 | Fig 1–5 캡션, Fig 1 이미지 |
| 30 | Runge, A., Nitze, I., Grosse, G. Remote sensing annual dynamics of rapid permafrost thaw disturbances with LandTrendr. RSE 268, 112752 (2022) | 10.1016/j.rse.2021.112752 ; 공개본 https://epic.awi.de/id/eprint/54836/ | PDF(pdftotext) | Fig 1–8 캡션 |
| 31 | Crameri, F. et al. The misuse of colour in science communication. Nat Commun 11, 5444 (2020) | 10.1038/s41467-020-19160-7 | nature.com HTML | Fig 1–6 캡션 |
| 32 | Wong, B. Points of view: Color blindness. Nat Methods 8, 441 (2011) | 10.1038/nmeth.1618 | nature.com HTML | Fig 1–3 캡션 |
| 33 | Katsnelson, A. Colour me better: fixing figures for colour blindness. Nature 598, 224–225 (2021) | 10.1038/d41586-021-02696-z | nature.com HTML | 본문 요령 4개 원문 |
| 34 | Nature formatting guide | https://www.nature.com/nature/for-authors/formatting-guide | nature.com HTML | 그림 · 범례 항목 원문 |

확인하지 못한 항목: de Bruin et al. 2022(Ecol Inform 69, 101665, CV 추정 대 참 정확도 산점)는 WUR 저장소(504) · ScienceDirect(403) 모두 받지 못해 표에서 뺐다. Read et al. 2019(WRR 55, 9173)는 Wiley 가 403 이었다. Miner et al. 2022(Nat Rev Earth Environ 3, 55)와 Turetsky et al. 2020(Nat Geosci 13, 138)은 캡션을 읽었으나 개작 그림이거나 지도가 아니어서 표에서 뺐다. nature.com 은 WebFetch 로는 쿠키 리디렉션에 막히고 쿠키를 저장하는 curl 로는 캡션 · 그림 파일이 모두 읽힌다. 내려받은 PDF · 그림 파일은 세션 scratchpad 에만 두었고 저장소에는 넣지 않았다.
