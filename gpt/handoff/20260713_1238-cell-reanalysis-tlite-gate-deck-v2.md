# Handoff — 셀 단위 재분석 · T-lite 게이트 · CCI 확충 · 발표덱 v2

작성 2026-07-13 12:38 · 대상 커밋 `fb83ae4`(push 반영 확인) · 이전 핸드오프 `20260708_1538-data-restructuring-thread-r.md`

## 1. 이번 세션 요약

GPT 계획(`gpt/20260709_claude_next_research_plan_dl_alt_3d.md`)의 P2/P3/P6-C를 실행하고, 결과를 학술 보고서 톤 발표덱으로 정리했다. 핵심은 "셀 단위(location-equal) 정직 재평가"에서 정보 병목이 재확인된 것이다.

## 2. 실험 결과 (모두 근거 CSV, 수치 감사 25/26 통과)

- **셀 단위 다중모달 ablation** (`alt_ablation_cell_results.csv`, 14,348셀, GBM 고정)
  - LORO: 기후(ERA5) 16.45cm(skill 10.8%), 기후+지형 16.94(지형 추가는 악화), +InSAR 16.09(12.7%, 물리 조합 최고), +PolSAR 16.98, 전체 16.43.
  - **위치 대조군(위경도 2피처) 15.72cm(skill 14.7%)**. 물리 공변량 조합보다 높다. 점-단위 옛 ablation의 15% skill이 pseudo-replication 착시였음을 셀 재평가로 확증했다. 위도가 기후 이상을 대리하는 것이며, 정보 병목의 직접 증거이다.
- **보정 UQ** (`alt_conformal_cell_results.csv`): raw 분위-GBM 90% 구간의 실제 커버리지 56.1%(심한 과신) → CQR 보정 85.9%(폭 50.6cm). 점-단위(71%)보다 raw 과신이 심하다.
- **AOA 전이** (`alt_aoa_cell_transfer.csv`): DI 구간이 커질수록 RMSE 상승(저DI 13cm → 고DI 30cm). 커버리지는 비단조(D1 61%, D3 88% 피크, D6 50%). 셀 데이터가 ABoVE_AK에 편중돼 LORO 전이 대비는 제한적이다.
- **T-lite 시계열 DL 게이트** (`tlite_sequence_gate_results.csv`, CALM 251 사이트, 3,345 시퀀스): GRU/TCN vs persistence·climatology·GBM-annual.
  - site-disjoint: GRU 16.79 < persistence 16.98 < GBM 17.33 (GRU 소폭 최우수).
  - temporal holdout: **GBM-annual 15.86 < persistence 17.02 < GRU 19.15 < TCN 23.85** (DL 붕괴).
  - **게이트 미통과**(temporal 미충족). 사전등록 규칙대로 부록/future work로 강등. 정적 tabular ALT는 GBM으로 충분함을 재확인했다.
- **CCI prior 확충** (`enrich_cci_cell.py`): ESA CCI ALT 25년 다년평균을 14,348셀에 추출(전 셀 유효, 관측과 r=0.53). ablation M8은 개선이 없다(기후와 정보 중복). CCI는 prior/benchmark로만 사용한다.
- **SoilGrids**: ISRIC vsicurl VRT 원격 읽기가 정체(산출 0)로 중단. 다음 세션 재시도 필요.

## 3. 발표덱 v2

`deck/build_report.py` + `deck/report_lib.py`(18슬라이드) → `deck/render/permafrost_report.{pptx,pdf}`. 배경·동기·선행연구·연구질문·데이터·방법·정직평가·결과(재분석·정확도·UQ·3D·T-lite)·요약 구성. 종이 배경, 세리프 제목(Noto Serif CJK KR), booktabs 표, 렌더된 수식. 사용자 지적을 반영해 em-dash를 렌더 텍스트에서 전량 제거하고 장식 위젯을 절제했다. 문체 규율은 전역 규칙 `~/.claude/rules/writing-tone.md`에 고정했다.

## 4. 현재 확정 상태

- 정적 ALT 매핑 정확도의 한계는 모델이 아니라 공변량 정보이다. 셀 재평가가 이를 확증한다.
- 헤드라인 평가는 공간블록·LORO만 사용한다. 무작위 CV는 약 4배 과대평가한다(IDW 28 대 111cm).
- 3D 엔진은 GBM 조건장으로 확정(신경장 게이트 탈락). ALT(계절 최대융해)와 MAGT(연평균장)를 분리한다.
- T-lite는 게이트 미통과로 부록 대상이다.

## 5. 다음 단계 (사용자 방향: GPT와 상의 예정)

1. **연구 지속**: 데이터 확장(SoilGrids M6 재시도, Sentinel-1/2 M7, PANGAEA CALM 월별 forcing으로 T-lite 재검증), 스레드 B(얕은 3D GBM 조건장 + CCI 0/1/2/5/10m prior + 0°C 등온면).
2. **발표자료(PPT) 전면 재구성 검토**: 현재 v2는 보고서 톤으로 정리했으나, 구성·서사를 다시 짤 여지가 있다. 대회 발표 목적에 맞는 슬라이드 구조를 GPT와 함께 재설계할 계획이다.
3. **그림 전면 재구성**: 신규 실험 4종은 논문 관례로 정리했으나, 개념도·지도류 등 재사용 그림이 남아 있다. ALT 관련 논문 그림을 참고해 시각화를 제대로 다시 구성할 계획이다.

## 6. GPT에게 묻고 싶은 것

- 위치 대조군(위경도)이 물리 공변량 조합을 능가하는 결과를 논문에서 어떻게 프레이밍할지. 정보 병목의 증거로 제시하되, "위도=기후 대리"를 과하게 단순화하지 않는 서술.
- T-lite 게이트 미통과(음성 결과)를 본문 대 부록 중 어디에 둘지, 그리고 월별 forcing 재검증의 우선순위.
- 발표 슬라이드 구조(연구 본체 대 발표 임팩트)와 핵심 그림 셋 구성에 대한 제안.

근거 파일: `data/processed/*_cell_*` `tlite_*` `overnight_cell_meta.json` · 코드: `scripts/2_evaluation/overnight_cell_experiments.py` `scripts/3_deep_learning/tlite_sequence_gate.py` `scripts/1_data_prep/enrich_cci_cell.py` · 덱: `deck/`.
