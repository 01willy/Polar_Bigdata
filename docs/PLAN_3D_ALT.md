# 연구 방향 확정 + 실험 세부 계획 v2 (2026-07-01)

> 근거: `DATA_MODEL_SCOUT.md`(멀티에이전트 조사, 정량결론은 우리 데이터로 재검증),
> 분산분해(위치간 87.7%/위치내 12.3%=잡음), ERA5 재채점(전이 −20%), 패치CNN≈GBM(+2.8%).
> 용어 정리: **"3D ALT map" = 2D ALT 지도 + 0~20m 3D 지중온도장**. ALT는 3D 온도장의
> **0°C 등온면 깊이로 유도**되는 파생물 — 예측은 "온도장 한 번"만 한다(이중 예측 아님).

## 0. 4대 질문에 대한 결론

**Q1. 멀티에이전트 조사 신뢰성 / CNN 학습 여부**
- CNN은 **이미 학습 완료** (패치CNN, GPU6, 공간블록 4-fold: CNN 17.2 vs GBM 17.7cm).
- 조사결과의 **정량 핵심은 전부 우리 실측으로 재검증되어 신뢰** (분산 87.7%, ERA5 −20%, CNN≈GBM, LORO에서 IDW/kriging 붕괴).
- 단 에이전트가 든 **외부 인용(논문 ID 등)은 미검증** — 채택 전 개별 확인. 근거 없는 문장 1건 발견("flow-matching 메모리 확정" — 그런 메모리 없음) → 정량만 믿고 인용은 걸러낸다.

**Q2. 3D ALT를 정확히 추론할 DL 모델**
- **정확도(2D ALT)**: GBM 유지(어떤 DL도 정보병목에서 못 이김 — 실측+조사 일치).
- **3D 온도장(신규 산출물)**: **조건부 신경장(Conditioned Neural Field, SIREN류)** 채택.
  T(lat,lon,depth) = f(좌표, 공변량임베딩). ALT 관측과는 "0°C 등온면 깊이=ALT" **일관성 손실**로 연결,
  borehole 깊이별 온도가 **수직 앵커**. 물리 정칙화(진폭의 깊이 감쇠, ∂²T/∂z² 평활) + 딥앙상블 UQ.
- **채택 게이트(필수)**: 깊이 포함 IDW/kriging/GBM(+depth)을 spatial CV로 **실제로 이겨야만** 채택. 못 이기면 보간+GBM으로 산출물 생성.

**Q3. 부족한 데이터, 추가 확보 순서**
1. **이미 보유했는데 미활용인 자산 먼저** (비용 0):
   - GTN-P 원본 933MB — **전지구 397 borehole 중 35개(알래스카)만 파싱됨** → 전지구 프로파일 파싱이 최대 라벨 확장.
   - InSAR ReSALT 6.9GB — 30m ALT·침하·불확실성·토양수분·수위 격자(구조 확인 완료) → 공변량+조밀 라벨.
   - GGD200(서시베리아)·PERMOS — 3D 라벨로 편입.
2. **무계정 공변량**(위치간 변동 공략): 토양 열물성(SoilGrids/Dai), 지하빙(GGD318), 이탄두께(Hugelius). 알래스카 bbox만.
3. **계정/수동**: TPDC 고원 3/6/10/20m(사용자 가입 필요), NGEE-Arctic, Nordicana D8(외부망에서 수동).

**Q4. 물리 시뮬레이션 ALT map 보간으로 피벗해야 하나?**
- **전면 피벗은 반대.** 이유: ① 순수 보간이면 novelty 소멸(ESA CCI 1km 제품 재생산), ② 물리모델 편향을 검증 없이 상속, ③ 우리 차별점(관측기반+전이+UQ)이 사라짐.
- **대신 하이브리드(residual learning)**: 물리제품(ESA CCI GT/ALT 1km, GIPL 알래스카)을 **prior/weak label + 벤치마크**로 쓰고,
  관측(우리 라벨)으로 **물리제품의 오차를 학습·보정**한다. 이러면 라벨 부족은 물리제품이 메우고, 관측이 정답 역할을 유지하며,
  "물리제품을 관측으로 보정 + 불확실성 + 전이검증"이라는 명확한 기여가 남는다.
- **게이트**: CCI prior를 피처로 넣었을 때 LORO가 나빠지면 즉시 제거(나쁜 피처가 전이 악화시키는 것 실측된 바 있음).

## 1. Phase A — 데이터 자산 총동원 (1~2일, CPU)
| # | 작업 | 산출 | 게이트/성공기준 |
|---|---|---|---|
| A1 | **GTN-P 전지구 프로파일 파싱** (397 borehole, 미활용 933MB) | `ground_temp_global.csv` (전지구 깊이-온도) | 국가≥6, borehole≥200 파싱 |
| A2 | **ReSALT 활용**: 51 granule 모자이크 → 관측점에 alt/sub/Sw0/wtd 부착 + 30m 조밀 weak label 추출 | `dl_dataset_v2.csv` + `resalt_weaklabels.csv` | 공간블록 CV에서 GBM RMSE 개선 여부 기록 |
| A3 | **무계정 공변량 3종**(열물성·지하빙·이탄, 알래스카 bbox) | 공변량 컬럼 추가 | **공간블록+LORO 재채점** — 전이 악화 피처 즉시 제거 |
| A4 | **ESA CCI GT/ALT 1km 다운**(CEDA 익명, ALT 0.9GB 우선) | prior/벤치마크 격자 | CCI prior 피처 LORO 게이트 |
| A5 | (사용자) TPDC 가입 → 고원 시추공 수동 다운 / Nordicana 외부망 | 전이 도메인 라벨 | — |

## 2. Phase B — DL 사전학습+미세조정 & 3D 신경장 (2~4일, GPU 0–3)
> **v2.1 확정(2026-07-01, 사용자 제안 반영)**: 물리제품/InSAR을 **synthetic·weak label로 써서 DL을 사전학습** → 실측으로 미세조정.
> 데이터량 문제(DL이 GBM을 못 이기는 원인)를 정면 해결. weak label 규모: **InSAR ReSALT ~64M 픽셀**(실측의 285배) + CCI 1km 수백만 셀.

| # | 작업 | 상세 |
|---|---|---|
| B0 | **DL 사전학습(2D ALT)** | weak label(ReSALT 30m ~64M + CCI 1km)로 신경장/CNN 사전학습 → 실측(ABoVE 22.4만+CALM)으로 미세조정. **DL이 주인공** |
| B1 | **baseline 구축** | GBM, CCI 자체, IDW/kriging(+3D는 depth 포함) — DL이 이겨야 할 기준선. 평가는 실측+공간블록/LORO만 |
| B2 | **Conditioned SIREN(3D)** | 같은 신경장의 depth 축 확장: (x,y,z)+공변량 FiLM → T. 손실 = 프로파일 MSE + ALT 0°C 일관성 + 물리 정칙화(PINN). GIPL/CryoGrid 합성 컬럼 사전학습 가능. GPU 0–3 앙상블 4개(=UQ) |
| B3 | **게이트 평가** | B0/B2가 B1을 못 이기면 해당 노선 중단(kill-switch). 이기면 채택 |
| B4 | **전이 검증** | 알래스카 학습 → GGD200(서시베리아)/PERMOS(알프스)/TPDC(고원) LORO |
| B5 | **주의사항** | weak label 편향: 미세조정·평가는 실측만 / "CCI 흉내" 방지 위해 CCI baseline 필수 포함 / LORO 악화 피처 즉시 제거 |

## 3. Phase C — 산출물 & 시각화 (1~2일)
- **알래스카 1km 3D 온도큐브 + ALT 지도 + UQ 지도** (신경장 or 보간, B3 결과에 따라)
- 0°C 등온면 3D 메시(pyvista), 깊이 스캔 GIF, **CCI/GIPL과 나란히 비교 지도**
- 시각화 원칙 준수: 결과마다 지도/3D/GIF ≥2종 (`docs/VISUALIZATION.md`)

## 4. 성공 기준 · kill-switch
- **성공**: (i) 3D 프로파일 예측이 baseline 대비 RMSE ≥10% 개선 or 동급+연속장/UQ 제공, (ii) ALT 유도값이 GBM 지도와 정합(편향 <10cm), (iii) 전이 3개 도메인에서 물리적으로 타당한 프로파일.
- **kill-switch**: B3 게이트 실패 → 신경장 폐기, "보간+GBM+물리 prior" 노선으로 산출물 완성(연구 기여는 데이터·전이·UQ·비교에 있음).
- **금지**: 무작위 CV 수치 인용, LORO 게이트 없는 피처 추가, 그래프-only 결과 보고.
