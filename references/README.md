# 참고문헌 (references/)

우리 프로젝트(well 온도 → 3D 영구동토/활성층 보간)와 관련된 핵심 논문·제품 자료.
PDF가 받아진 것은 파일로, 페이월/제품은 링크로 정리.

## 다운로드된 PDF

| 파일 | 분류 | 한줄 요약 / 우리와의 관계 |
|---|---|---|
| `ran2022_NH_permafrost_thermal_state_ESSD.pdf` | **데이터 융합 ML(최근접 경쟁)** | GTN-P 1002 borehole + CALM + 9공변량 ML앙상블 → NH MAGT(DZAA)·ALT·확률 1km. **우리 baseline과 같은 데이터·철학**(단 단일깊이 2D, 3D·base 없음). |
| `jafarov2012_GIPL2_alaska_TC.pdf` | **물리 forward 모델(GIPL2)** | 열전도+상변화 PDE를 기후로 구동해 알래스카 지중온도·활성층·base 시뮬레이션. **우리가 벤치마크할 incumbent**. |
| `gautam2025_ALT_alaska_ML_Stefan_SciRep.pdf` | **활성층 ML(직접 비교 대상)** | RF vs Stefan 물리모델로 알래스카 ALT 250m. RF 훈련 R²0.84→테스트 0.24(과적합), Stefan 0.54. **순수 ML의 일반화 한계 실증**. |
| `aljubran2024_InterPIGNN_arxiv.pdf` | **PINN 보간(방법 선례)** | 정상상태 3D 열전도 PDE를 손실에 넣어 sparse borehole→3D 온도체적 보간(지열,미국). **우리 PINN 핵심과 구조 동일** — 인용·차별화 필수. |
| `groenke2023_bayesian_heat_transfer_TC.pdf` | **물리+베이지안 UQ(1D)** | 1D 열전도(상변화 포함) 역산 + 앙상블칼만으로 borehole 프로파일에서 깊이별 T 복원 + 불확실성. **우리 PINN+UQ의 1D 선례** — 3D로 올리는 게 빈틈. |
| `biskaborn2019_GTNP_permafrost_warming_NatComm.pdf` | **GTN-P 데이터 기준** | GTN-P 전지구 borehole로 영구동토 온난화 정량(2007–16 +0.29°C). **우리 데이터(GTN-P)의 표준 레퍼런스**. |

## 페이월/제품 (링크)

- **Pastick et al. 2015**, *Distribution of near-surface permafrost in Alaska* (Remote Sens. Environ.) — 30m 의사결정나무로 근지표 영구동토 유무 매핑(85%). DOI 10.1016/j.rse.2015.07.019. 데이터: https://www.sciencebase.gov/catalog/item/5602ab5ae4b03bc34f5448b4
- **Obu et al. 2019**, *NH permafrost map (TTOP) 1km* (Earth-Sci. Rev.) — 범극지 MAGT@top 표준 제품. DOI 10.1016/j.earscirev.2019.04.023 · 데이터 PANGAEA 10.1594/PANGAEA.888600
- **ESA CCI Permafrost** (제품) — 위성+ERA5를 CryoGrid로 구동, 범극지 1km, 0/1/2/5/10m 지중온도·ALT. CEDA: https://catalogue.ceda.ac.uk/uuid/5675b0be944f45a8af0e7ddbeb47a011
- **GIPL2 Alaska 제품**(SNAP/UAF) — 1km 지중온도(10깊이)+영구동토 base. http://catalog.snap.uaf.edu (Permafrost)

## 분류 요약
- **물리 forward 시뮬레이션**: GIPL2(Jafarov), CCI — 기후→PDE, well은 검증만.
- **데이터 ML 보간**: Ran 2022, Gautam 2025, Pastick 2015 — well+공변량→2D 지도.
- **PINN/물리+ML**: InterPIGNN(지열 3D), Groenke(1D 베이지안).
- **우리의 빈틈**: well 직접보간으로 **3D 깊이장 + 활성층/base + 불확실성 메시**(+ 도메인 전이).
