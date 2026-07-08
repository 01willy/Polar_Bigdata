# Layout Rules — Polar_Bigdata 그림·발표 레이아웃 규칙

> `design/brand_tokens.json`의 토큰을 실제 레이아웃에 적용하는 규칙. 모든 그림 스크립트가 준수.
> 코드 진입점: `from polar.plotstyle import use_polar; from polar.outputs import figpath, mappath, volpath`.

## 1. 한 그림 = 하나의 지배 메시지
- 그림 하나가 하나의 주장(claim)을 전달. 제목은 결론형(예 "무작위 CV는 전이를 4배 과대평가").
- 근거 수치는 캡션/주석으로, 헤드라인은 공간 시각화로.

## 2. 격자·정렬
- 수동 배치 금지. `plt.subplots` 격자 사용, 축 상하/좌우 정렬.
- 다패널은 공유 컬러바(하나) 또는 패널별 컬러바를 일관되게. subplot 간격 wspace/hspace 0.25.
- 여백 8% 이상(프로젝터 크롭/PDF 변환 대비).

## 3. 축·컬러바·단위 (필수)
- 모든 축에 변수명 + 단위. 모든 컬러바에 라벨 + 단위(cm, °C, cm/yr 등).
- 지도: 위경도 격자 또는 스케일바 + 컬러바(단위). 종횡비=물리비율.
- 부호장(온도·차이·개선)은 **0중심 발산**(vik/broc, TwoSlopeNorm vcenter=0). 비음수(ALT·오차·밀도)는 순차형.

## 4. 타이포·선·마커 일관성
- 제목 15pt bold, 축라벨 11pt, 눈금 9pt, 주석 9pt (brand_tokens 참조). 인쇄 크기에서 가독.
- 선폭 1.8, 마커 5, 상/우 spine 제거, grid alpha 0.25 — 관련 그림 전체 통일.

## 5. 색 규약 (냉색 표준, 붉은 계열 금지)
- ALT=oslo_r · 온도=vik(0중심) · 오차/UQ=acton · 차이=broc · 밀도=davos_r · 지형=bukavu.
- **색으로만 정보 전달 금지**: AOA 외삽영역은 회색 마스킹 + 해칭 병행. 범주는 색+마커/라벨.

## 6. 지도 원칙 (VISUALIZATION.md 강제)
- 어떤 결과든 그래프만 내지 말 것 — 공간 시각화 ≥2종(예측지도 + 오차/불확실성지도).
- 관측점 오버레이, AOA 마스크, 0°C 등온면 등 물리 주석.

## 7. 산출·내보내기
- 벡터(SVG/PDF) 우선 + 고해상 PNG(논문/포스터 300dpi, 슬라이드 200dpi). 논문 그림에 래스터 텍스트 금지.
- 파일명은 폴더가 맥락 → 서술형(snake). 경로는 outputs 헬퍼로.

## 8. 완료 기준
- 렌더 후 `scientific-figure-reviewer`(과학적 정확성) + `visual-reviewer`(레이아웃/가독성) QA 통과 후 완료 선언. 결함 수정 전 "완료" 금지.
