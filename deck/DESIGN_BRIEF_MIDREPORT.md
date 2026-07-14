# 디자인 브리프 — 중간보고 PPT 및 본선 발표덱 재구성

작성 2026-07-13 · 용도: 클로드 디자인(ppt-art-director) 인수인계 문서. 이 문서만 읽고 작업을 시작할 수 있도록 자기완결적으로 작성.
근거: visual-reviewer 에이전트의 덱 v2 전수 감사(18슬라이드 + 독립 그림 파일, 결함 26건) + 프로젝트 디자인 시스템.

---

## 1. 목적과 제작물

| 제작물 | 목적 | 시점 |
|---|---|---|
| **중간보고 PPT** | 연구실 중간보고: ① 현재까지 연구 현황 ② 앞으로의 진행 계획 ③ 빅데이터 대회 최종 제출물 예상도 | 우선 |
| **본선 발표덱** | 극지 빅데이터 대회 본선 발표(접수 08-17~31). 중간보고 PPT의 상위 호환으로 재사용 | 예선 통과 시 |

사용자 평가: "PPT 디자인과 3D 그림 등 시각화가 전문적이지 못하고 미감이 떨어진다." 목표는 장식 추가가 아니라 **정보 위계·그리드·타이포·그림 판독성의 정상화**다.

## 2. 기존 자산 (재사용 대상)

- 덱 v2: `deck/build_report.py` + `deck/report_lib.py` → `deck/render/permafrost_report.{pptx,pdf}` (18슬라이드, 렌더 PNG `deck/render/rpt-*.png`)
- 그림: `outputs/figures/`(evaluation·deep_learning), `outputs/maps/`(AOA 마스크·불확실성 폭·예측 지도), `outputs/volumes_3d/`(**frozen_body.vtp, permafrost_zero_isotherm_rf.vtp** 존재 → PyVista 전환 준비됨)
- 디자인 시스템: `design/brand_tokens.json`(냉색 규약: ALT=oslo_r·온도=vik 0중심·오차=acton·차이=broc, categorical 6색, highlight #0b7285), `design/layout_rules.md`, `design/visual_qa_checklist.md`
- 문체: `~/.claude/rules/writing-tone.md` 준수. **em-dash(—) 금지**, 보고서 톤, 명사형 제목

## 3. 감사에서 확인된 결함 (26건 중 severity=high 요약)

1. **그림 과소 크기**: rpt-07(공변량 4패널)·rpt-08(InSAR 약지도 3패널)이 슬라이드 높이 30~35%로 축소되어 축라벨·컬러바 판독 불가. 삽입 의미 상실.
2. **KPI 컨텍스트 라벨 극소 폰트**: rpt-02 하단 수치 띠(16.1cm, 56→86% 등)의 설명 라벨이 6pt 이하로 발표 화면에서 판독 불가.
3. **캡션·출처 노출**: 그림 캡션이 푸터 근처에 분리 배치(6~7pt), `outputs/figures/...`·`data/processed/...` 내부 파일 경로가 슬라이드에 그대로 노출. 발표용에서는 제거하고 speaker notes로 이동해야 함.
4. **배경 불일치**: rpt-01~17 흰 배경 vs rpt-18만 검정 배경. 유일한 다크 슬라이드로 일관성 붕괴.
5. **밀집 bar chart**: rpt-12(셀 재분석 2패널) 막대 10개+가 밀집, x축 회전 라벨 판독 곤란, 두 패널 y축 범위 상이한데 주석 없음.
6. **rpt-15 (3D 열구조)**: 두 패널 컬러바 범위 상이 설명 없음, 오른쪽 그림 서브플롯 축라벨 판독 불가.
7. **그림 내부 폰트 vs 슬라이드 폰트 위계 붕괴**: 그림 제목이 슬라이드 섹션 제목보다 크거나 비슷하게 렌더되는 슬라이드 다수.
8. **3D 그림 (열큐브 thermal_cube_alaska.png 등)**: matplotlib Axes3D 기본 회색 pane·격자 노출, 뷰포인트 미최적화로 뒤 슬라이스가 가려짐, 비선형 깊이 간격 미표현.
9. **hero_3d_permafrost.png**: borehole 수직 막대 두께·마커 불균일, 검정 배경에서 컬러바·범례 대비 부족.
10. **ground_temp_profiles.png**: rainbow 다색 선 사용(브랜드 토큰의 냉색·색맹안전 규약 위반), 색으로만 사이트 구분.

중간 심각도(발췌): 제목 이중 공백·자간 미제어(rpt-01), 표 셀 패딩·헤더 구분 부재(rpt-05), Q/C 두 열이 쌍으로 오독되는 배치(rpt-06), 텍스트 화살표 파이프라인(rpt-09), 토큰 외 임의색(노랑 주석 박스·갈색 막대), 여백 8% 미준수 슬라이드(rpt-10·12), cross_section 제목·지리 범위 표기 부재.

## 4. 아트디렉션 (감사 제안 채택)

1. **배경 통일**: 오프화이트(#F8F8F6) 단일 배경 전체 유지. 표지만 진한 배경을 쓸 경우 표지 전용 마스터를 분리.
2. **색 제한**: 슬라이드 크롬(제목·킥커·강조)은 highlight #0b7285 하나. 경계 #E5E8EC, 보조 텍스트 #6B7280 회색 2단계만 추가 허용. 토큰 외 색 전면 제거.
3. **12컬럼 그리드**: 좌우 여백 각 6%, 거터 1%. 텍스트 전용=1~8컬럼, 그림+텍스트=텍스트 1~5 / 그림 6~12 고정. `report_lib.py`에 컬럼 좌표 상수로 정의해 전 슬라이드 공유.
4. **타이포 4계층 고정**: 슬라이드 제목 36pt bold / 섹션 킥커 11pt uppercase #0b7285 / 본문 18pt / 캡션·출처 10pt #6B7280. 그림 내부는 발표 기준 axis_label 13pt·tick 10pt·annot 10pt로 brand_tokens 상향 조정(슬라이드 축소 후 판독 보장).
5. **그림 삽입 크기 표준**: 단독 그림은 슬라이드 높이 60% 이상, 2그림이면 각각 50% 이상. 슬라이드당 그림 1개 원칙. 삽입 PNG는 slide_16x9(12.8×7.2in) 200dpi로 사전 렌더.
6. **KPI 카드 컴포넌트**: 수치 40pt bold + 단위 20pt #6B7280 + 컨텍스트 라벨 12pt + 선택적 델타 화살표, 배경 #F0F4F8 라운드 사각형. rpt-02·11 등에 일괄 적용.
7. **파이프라인 슬라이드(rpt-09) 벡터 다이어그램화**: 텍스트 화살표를 matplotlib.patches 또는 SVG 노드·엣지 다이어그램으로 교체, 각 노드에 데이터 수(n=14,348셀 등) 명기. `model-architecture-figure` 스킬 활용 권장.
8. **표 재설계(rpt-05)**: 헤더 행 #0b7285 배경+흰 글자, 교차 행 음영(#F0F4F8/흰), 컬럼 폭 20/35/45%, 본 연구 행은 bold+좌측 2px 컬러 바.
9. **경로·캡션 정리**: 파일 경로 전량 제거(speaker notes로), 출처는 "CALM(2022), GTN-P(2024)" 형태 데이터셋 명칭으로.
10. **문체**: 명사형 제목, em-dash 금지, 과장 금지(예: "SOTA 돌파" 표현 금지, RMSE 옆 R²·skill 병기).

## 5. 3D·공간 시각화 개선 지침

### 즉효 수정 (matplotlib 유지)
- Axes3D: `ax.{x,y,z}axis.pane.fill=False`, `pane.set_edgecolor('none')`, `ax.grid(False)`로 회색 박스 제거. 뷰포인트 elev=30/azim=-50 또는 elev=20/azim=-40으로 전 슬라이스 노출.
- 슬라이스 면에 0°C 등온선 contour 오버레이(white dashed, lw 1.2)로 물리 경계 강조. 비선형 깊이 간격(0.5/2/5/10/20m) 주석 명기.
- hero_3d: borehole 막대 lw 1.5·white·alpha 0.7 통일, 마커 s=20·zorder=5, 범례 facecolor #1A1A2E·labelcolor white, 컬러바 텍스트 흰색 명시.
- cross_section: 결론형 제목 추가, 경도 고정값·위도 범위 명기, 0°C contour 명시.
- ground_temp_profiles: rainbow 제거 → categorical 6색 + 나머지 회색(#B8BEC6), 범례에 사이트명.

### 발표 품질 전환 (본선 히어로 비주얼)
- **PyVista off-screen 렌더**(window_size 2560×1440): 기존 `outputs/volumes_3d/*.vtp`(frozen_body, permafrost_zero_isotherm_rf)를 add_mesh로 등온면 렌더, 조명·ambient occlusion 적용 고해상 PNG 출력. 필요 시 인터랙티브 HTML 병행 export.
- 깊이 슬라이스 GIF: 팔레트 생성으로 밴딩 저감, 프레임 간 컬러 정규화 고정(layout_rules 준수).

## 6. 중간보고 전용 추가 슬라이드 3장 (현재 덱에 없음)

A. **연구 현황 타임라인**: 완료(셀 재분석·conformal·AOA·토너먼트·음성결과 2종) / 진행(KPDC 편입·AlphaEarth ablation·예선 보고서) / 예정(3D 열큐브·본선 덱)의 3단 행. 대회 일정(7/31 예선·8월 본선자료·9월 본선)을 축에 병기.
B. **최종 제출물 예상도**: ① 분석 보고서(예선) ② ALT 2D 지도+불확실성+AOA 마스크 지도 세트 ③ 얕은 3D 열구조 시각화(PyVista 등온면·GIF) ④ 전이 검증 표 ⑤ 재현 코드 패키지. 각 항목을 실제 그림 썸네일 또는 목업으로 제시.
C. **잔여 리스크**: KPDC 주문 승인 대기, "주된 활용" 해석, 3D 조건장 구현 일정. 표 형식.

## 7. 중간보고 PPT 구성안 (12~14장)

1. 표지 2. 요약(KPI 카드 4개) 3. 대회 요강과 우리 포지션(마감·규칙·KPDC 전략) 4. 연구 정체성(1문장+개념도) 5. 데이터 지도(보유+KPDC 편입 계획) 6. 정직 평가 프로토콜(CV 누설 그림) 7. 핵심 발견: 정보병목(위경도 대조군 bar) 8. 불확실성 보정+AOA(지도 2종) 9. 전이 검증(LORO) 10. 음성 결과 요약(T-lite·CCI) 11. 현황 타임라인(신규 A) 12. 최종 제출물 예상도(신규 B) 13. 리스크·일정(신규 C) 14. 마무리

콘텐츠 수치·문구 원천: `docs/CONTEST_PLAN_2026.md` §1(확정 수치 표)·§5(일정)·§6(목차). 수치는 반드시 `data/processed/*.csv` 근거와 대조.

## 8. 작업 절차와 QA

1. `deck/deck_spec.json` 갱신(중간보고 구성 반영) → 2. 그림 사전 렌더(brand_tokens 상향 폰트 적용) → 3. `build_report.py` 계열로 빌드 → 4. PDF/PNG 렌더 → 5. `scientific-figure-reviewer`(과학 정확성) + `visual-reviewer`(레이아웃) QA → 6. 결함 수정 후 완료 선언(`design/visual_qa_checklist.md` 기준).

일정 유의: 예선 보고서(7/31 마감)가 최우선이므로 중간보고 PPT 제작은 보고서 골격 완성 이후 착수를 권장. 본선 덱은 08-17~31 접수 기간에 이 브리프의 전체 결함 목록(26건)을 소화한다.
