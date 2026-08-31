# Handoff: 본선 발표덱 최종화 + 정합 감사(BLOCKER 해소) + 논문 저널 전략

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-08-31 11:05
**Session focus**: 본선 발표덱 23장을 논문급 그림 문법으로 최종화하고, 4축 정합 감사로 발견한 BLOCKER(그림-표 셀 집합 불일치)를 포함한 전건을 수정한 뒤, 이 연구의 논문화 가능성과 투고 저널을 확정하였다.
**Author**: Claude Opus 4.7 (1M) + 백승원

---

## 1. TL;DR

- **BLOCKER 발견·해소**: `s12_figs.py`가 "물리식(Stefan) 단독"을 `anchor="stefan"`(전체 셀)으로 그려 왔는데, 보고서 표 8·4.7절은 `stefan_cci_w`(CCI 유효 셀 한정)를 쓴다. 그림 두 지역 평균이 23.72/24.60인데 표는 22.91/24.11로 어긋났고, `fig:s12` 캡션의 "두 패널 평균이 표 8의 두 열"이라는 검증 가능한 진술을 위반하고 있었다. 앵커 교체 후 재생성하여 **표와 정확히 일치**(22.91/24.11, 편향 5.78/2.57)시켰다.
- **확정 오류 정정**: 알래스카 13,606셀의 공간블록 수는 **83이 아니라 74**다(`map_model_gate_meta.json` n_blocks=74로 독립 확인). 개요 그림·SESSION_HANDOFF·주석에 남아 있던 83을 전부 정정했다. 단 `submission/`(2026-07-30 예선 동결본)의 83은 수정 금지.
- **본선 발표덱 23장 확정**: `deck/render/permafrost_final.{pptx,pdf}`. 방법부 3그림을 3차에 걸쳐 재설계해 "박스는 3종만, 화살표는 직각만"이라는 평면 논문형 문법으로 확정했고 금지 목록을 스펙에 명문화했다. 발표 대본·Q&A(36문항) 문서도 생성기와 함께 저장소에 편입.
- **KPDC 유도 ALT 파싱 정당성 확증**: 독립 재파싱으로 소수점까지 재현, 깊이 라벨 3중 물리검사 통과. 같은 사이트 값이 35~153 cm로 갈리는 것은 오류가 아니라 지점·방식·연도 차이임을 값의 사다리로 정리했다.
- **논문화 판정**: 현 상태로 The Cryosphere/PPP는 무리. **1순위 CRST → 2순위 Scientific Reports**. TC 배제는 감(感)이 아니라 증거 기반(아래 §5).

---

## 2. Context

**직전 핸드오프**: [20260729_1741-report-consistency-audit-s13-dropped.md](20260729_1741-report-consistency-audit-s13-dropped.md) — 보고서 수치 전수 감사 통과, S13 통합실험 폐기, 미결로 "분량 13쪽 대 10쪽 안내"를 남김.

그 뒤 7/30에 예선 제출(`submission/` 동결)이 있었고, 이번 세션은 **본선 발표(2026-08-28) 준비**가 목적이었다. 발표덱 초안(`permafrost_final.pptx`)은 있었으나 사용자가 "페이지 6·8·9의 method 그림이 AI틱하고 전문성이 떨어진다"고 지적해, 참조덱(`IMAGE_SEUNGWONBAEK_...pseudo-MCMC.pptx`, RTM 프로젝트 산출물)의 도식 문법을 분석해 재설계하는 것에서 출발했다. 이어서 발표 대비 질의 36건에 답하는 과정에서 KPDC 해석에 대한 의문이 제기되어 원자료 재검증까지 진행했고, 마지막으로 논문화 가능성을 평가했다.

7/29 감사가 "결론 불변"으로 통과했음에도 이번에 BLOCKER가 나온 이유: 지난 감사는 **본문·표의 수치**를 원자료와 대조했고, 이번에는 **그림 생성 스크립트가 어느 행을 집어 그리는가**까지 내려갔다. 같은 이름("물리식 단독")이 서로 다른 셀 집합을 가리키고 있었던 것이라 표 대조만으로는 잡히지 않았다.

---

## 3. What we did

### 3.1 방법부 그림 3차 재설계 (덱 p6·p8·p9)

- **Action**: 참조덱 p3·5·6·8의 도식 문법을 분석해 적용. v3(저채도 구역 틴트) → v4(외곽 프레임 + 색 구역 패널) → **v5/v6(평면)**. v3·v4는 사용자가 "박스 과다·곡선 화살표·번호 틱바가 AI틱하다"고 반려.
- **Files**: `deck/mk_final_figs.py`(fig_workflow·fig_dl_arch·fig_aug_design·fig_aug_spatial), `figures/architecture_spec.json`, `deck/deck_spec_final.json`(revision_v5~v9)
- **Result**: 확정 문법 = 백색 배경 + 파선 구분선 + 주황 볼드 헤딩 + 텍스트 위계. **박스 허용은 3종뿐**(실데이터 이미지 액자 / 프로세스 셰브런 / 네트워크 연산 블록 `#F7E9AE`). 표는 북탭스 헤어라인, 판정은 채택(녹 `#2F6B33`)·제외(적 `#96382C`) 색 텍스트, **화살표는 수평·수직 직각만**. 금지 목록(외곽 프레임, 색 구역 패널, 카드 박스 일반, 판정 태그 박스, 곡선 rad 화살표, 번호 틱바, 장거리 파선 우회, 그림 안 pNN 상호참조, 그래프 사후 강조 라벨, "선+주황 라벨+큰 숫자" 통계 카드)을 스펙에 명문화.
- p6는 5열 균등 배분 + **행 높이를 열 간 정렬**(R1~R4)해 화살표를 전부 평행하게 만들었다.

### 3.2 p12 증강 효과 지도 재구성

- **Action**: "증강 후가 관측 색에 가까워지는지 육안으로 모르겠다"는 지적에 대응.
- **Result**: 전/후 직접 색 비교는 실제 변화가 5 cm 수준이라 ALT 20~120 스케일에서 원리적으로 판독 불가. 셀 단위 예측 원자료가 없어 좁은 스케일 재렌더도 불가(생성 스크립트가 7/24 세션 임시본으로 소실). **(a) 관측 · (b) 증강 전(21.9) · (c) 증강 전후 변화량 지도(±30 cm)** + 공유 컬러바 구성으로 확정. 변화량 지도가 "어디가 얼마나 하향 조정되어 관측 수준으로 갔는가"를 직접 보여주므로 슬라이드 주장(지역 전반 편향 보정)에 더 정합하다.

### 3.3 정합 감사 4축 + 최종 검수 (병렬 에이전트)

- **Action**: 참고문헌·수치(전·후반)·그림-본문 4축 감사 + 적대 검증(오탐 0) → 초견 청중 이해도 3렌즈 → 최종 오탈자/시각 QA 4그룹 → cleanup 감사 4축.
- **Result**: 확정 결함 8건 + 이해도 개선 29건 중 24건 + 시각 결함 22건 + cleanup 감사 13건 수정. 상세는 `docs/EXPERIMENT_LOG.md` 2026-08-31 항목.
- 주요: 참고문헌 4건 추가(Gorishniy×2·Prokhorenkova·Riseborough), 인용-주장 매핑 18건 전수 정합 확인.

### 3.4 BLOCKER 해소 (최우선)

- **Action**: `s12_figs.py`의 앵커 2곳(46행 series, 85행 (c)패널 편향)을 `anchor="stefan"` → `"stefan_cci_w"`로 교체 후 그림·덱 재생성.
- **Files**: `scripts/4_visualization/s12_figs.py`, `outputs/figures/s12_hybrid/s12_hybrid_summary.png`, `deck/build_final.py`
- **Result**: 검증 결과 half **22.91** / loro **24.11**로 표 8과 소수점까지 일치, 편향 5.78/2.57(본문 5.8/2.6). `w_stefan_cci=1.0`이므로 두 앵커는 **같은 Stefan 식**이고 차이는 평가 셀 집합뿐(CCI 결측 셀 포함/제외). 레나델타는 CCI 결측이 없어 동일(16.16), 캐나다만 갈렸다(31.29 → 29.66). 부수로 덱 p15의 면피성 각주를 "그림·표 모두 위성 제품이 유효한 같은 셀 집합에서 평가한 값"으로 교체.

### 3.5 KPDC 콘슬 유도 ALT 적대 재검증

- **Action**: "같은 사이트에서 35~153 cm로 갈리는 게 파싱 오류 아닌가"라는 의문에 대해 4에이전트(독립 재파싱 / 물리 정합성 / 외부 문헌·KPDC 조사 / ABoVE 출처 추적) 투입.
- **Result**: **파싱 정당성 확정**. 원자료에서 기존 스크립트를 보지 않고 재산출한 값이 소수점까지 재현(75.6/121.1/136.4/140.0/153.1, 검열 >80 13건·>160 10건). 깊이 라벨은 여름 단조 감소·겨울 역전·심부 진폭 감쇠 3중 물리검사를 전 시추공이 통과. 원파일 배치 A의 라벨 반전은 실재했으나 `s7_parse_kpdc_council.py`가 겨울 단조성 rho 검정으로 교정, ID10-14 인터리브 의심분은 배제됨.
- 정정 2건: (a) ID02 45.3·ID03 61.6은 **관측 절단(partial_end) 과소치**로, 완전 관측 비검열 스프레드는 75.6~153.1(약 2배). (b) 비교 기준 "탐침 35.0"은 **2017-05-30 초여름 측정 20% + CALM U27 그리드 1셀이 섞인 셀평균**으로 저편향이며 8월 한정 평균은 41.6 cm.

### 3.6 발표 문서 2종

- **Files**: `deck/mk_script_docx.py` → `발표대본_ALT_ctrl.docx`, `deck/mk_qa_doc.py` + `deck/qa_answers.md` → `발표QA대비_ALT_ctrl.docx`
- **Result**: 대본은 초견 청중 기준으로 전면 재작성(용어 최초 정의, 그림 안내 대사, 평이한 비유, 15분 30초 배분). Q&A 36문항은 **한줄 답 / 쉽게 설명하면 / 숫자로 / 근거** 4단 형식으로 재작성(22→13쪽). Q&A 생성기가 scratchpad에만 있어 재현 불가였던 것을 저장소로 이관.

### 3.7 논문화 가능성 평가 (3에이전트)

- **Action**: ALT-ML 논문 게재지 전수 조사(16편) + main.tex 전문 적대 심사 시뮬레이션 + 저널 스코프 조사. 이후 TC 심층(게재 통계·공개심사 리스크·ML 수용 폭·APC) 추가 조사.
- **Result**: §5·§7 참조.

---

## 4. Key numbers (this session)

이번 세션에 **새로 계산한 수치는 없다**. 아래는 감사로 검증하거나 정정한 값이다.

| 항목 | 대상 | 값 | 근거 아티팩트 |
|---|---|---|---|
| s12 물리식 단독 (수정 후) | 공변량만 / 두 지역 평균 | **22.91** cm | `data/processed/s12_hybrid_transfer_shard*.csv` (anchor=stefan_cci_w) |
| s12 물리식 단독 (수정 후) | 정보 없음 / 두 지역 평균 | **24.11** cm | 동일, 표 8(tab:s12)과 일치 |
| s12 물리식 편향 (수정 후) | 정보 없음 / 레나·캐나다 | **+5.78 / +2.57** cm | 동일, 본문 4.7절 5.8/2.6과 일치 |
| 표 8 기계학습 단독 (정정) | 공변량만 / 정보 없음 | 27.94/26.61 → **27.93/26.60** | `data/processed/s12_hybrid_gate.csv` |
| 알래스카 공간블록 수 (정정) | 13,606셀 | 83 → **74** | `data/processed/map_model_gate_meta.json` n_blocks=74 |
| 캐나다 순가치 (검증) | 증강 r=10, CatBoost | **+10.2** cm | `data/processed/s3_aug_curve_results.csv` 시드짝 재계산 |
| p12 증강 전/후 (검증) | 레나델타 RMSE | 21.91 (r=0) → 17.01 (r=1) → 16.5 (r=10) | 동일 CSV |
| p12 편향 (검증) | 레나델타 | +16.20 (r=0) → +8.30 (r=10) | 동일 CSV |
| 예측구간 폭 (정정 표기) | 알래스카 공간블록 CV | 보정 전 **14.7** → 후 **53.6** cm(평균) | `data/processed/s11_conformal_oof.csv` (셀별 p2–p98 = 39–76) |
| 커버리지 (검증) | 목표 90% | **93.4%** (CI 0.889–0.966) | `data/processed/s11_conformal_results.csv` |
| KPDC 유도 ALT (재현) | 콘슬 2025 비검열 5개 | 75.6 / 121.1 / **136.4**(중앙값) / 140.0 / 153.1 | `data/processed/s7_council_alt_derived.csv` |
| KPDC 탐침 기준 (정정) | 콘슬 반경 3 km | 35.0(초여름 포함) → **41.6**(8월 한정) | `data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` |

**주의**: 41.6과 CALM U27 68.6은 **main.tex 미수록 수치**다. 덱·보고서에 넣지 않았고(유입 0건 확인) Q&A 문서의 질의 대응용으로만 관리한다.

---

## 5. Decisions made

1. **s12 그림을 표 8 기준으로 맞춘다** (그림이 아니라 표가 옳음) — 본문 4.7절이 "위성 제품이 결측인 셀을 제외하고 모든 방법을 같은 셀에서 평가"라고 명시하고, 표 8 캡션도 동일. 그림만 전체 셀을 쓰고 있었다.
2. **p12는 변화량 지도로 간다** — 전/후 직접 비교는 스케일상 판독 불가이고 재렌더할 원자료가 없다. 변화량이 주장(지역 전반 편향 보정)에 더 정합.
3. **방법부 그림에 "박스 3종·직각 화살표" 문법을 강제한다** — 3차 반려 끝에 확정. 금지 목록을 스펙에 명문화해 회귀 방지.
4. **136.4를 콘슬 대표 ALT로 쓰지 않는다** — 심부 16층 시추공 부분표본이며, 외부 기준(CALM U27 그리드 53~89, KOPRI 2016 피트 58~73)을 상회. 보고서의 기존 단서("콘슬 전체를 대표하는 값으로 일반화하지 않는다")가 재확인됨.
5. **투고는 CRST 1순위, Scientific Reports 2순위** — The Cryosphere 배제. 근거: (a) TC의 ML 게재 사례는 전부 "ML=도구, 빙권 과학 질문=주장" 구조(Du 2026 ALT 공간이질성, Garibaldi 2026 TTOP 중요도)이고, 방법 조합 자체를 주장으로 낸 `tc-2022-9`(InSAR+RF 영구동토 매핑)는 **리젝**됐다. 우리 프로필과 정확히 일치. (b) TC는 투고 즉시 EGUsphere 공개이고 리젝 시 "revision was not accepted" 문구와 레퍼리 코멘트(각각 DOI)가 **영구 삭제 불가** — 비유의 헤드라인에 비대칭 손실. (c) Uxa et al. 2026이 이미 TC에 ALT RMSE 14.2–18.2 cm를 게재해 직접 비교당한다.
6. **`submission/`은 수정하지 않는다** — 7/30 예선 동결본. 내부에 83이 남아 있으나 제출 기록물이다. 본선 패키지는 `submission/code` 복사가 아니라 현행 `scripts/`에서 재생성한다.

---

## 6. Open questions / blockers

**차단 요소 없음**(BLOCKER 해소 완료). 아래는 판단이 필요한 열린 항목이다.

- **W1 전이 개선의 비유의성**: 24.11→22.92의 95% CI가 [−2.22, +2.76]로 0을 포함한다. 초록·결론이 이를 "핵심 차별점"으로 세우는 한 상위지 심사를 통과하기 어렵다. **해소 증거**: 전이 대상 지역을 2곳에서 4곳 이상으로 늘려 블록 수를 확충한 뒤 CI 재산정. 또는 지역 적응형 계수 E로 개선폭 자체를 키움.
- **W2 winner's curse**: 185개 조합×λ 5단계를 평가 데이터에서 채점한 뒤 최소값(22.92/21.32)을 보고했다. 선택이 개입하지 않은 수치는 등가중 앵커(24.11→23.27, 개선 0.84)뿐. **해소 증거**: 중첩 선택(내부 블록 CV로 구성 선택, 외부에서만 채점) 재평가.
- **KPDC 식별자**: 지온 프로파일은 KOPRI-KPDC-**00002707**(2024)·**00002955**(2025)로 등록되어 있는데, 덱·보고서는 코어 자료 식별자 00002125만 표기한다. 병기할지 결정 필요.
- **`report_overview_figure.py:124`의 `n_grid=892,865`**: 저장 산출물로 뒷받침되지 않는 하드코딩(main.tex 어디에도 없음). `report_alt_map_hires.py`가 격자 셀 수를 meta JSON에 기록하게 하고 그 값을 읽도록 바꾸는 것이 정공법.

---

## 7. Next steps (prioritized)

논문 투고(CRST)를 목표로 한 경로다. P0는 어느 저널이든 공통으로 필요하다.

1. **185조합 winner's curse 해소** — owner: Claude, 계산 수일, 선행조건 없음.
   중첩 선택으로 재평가하거나, 등가중 앵커(λ=0.25 고정)를 **사전 지정 주 추정치**로 교체하고 나머지는 탐색 결과로 강등. `data/processed/s12_hybrid_transfer_shard*.csv` 재분석만으로 가능.
2. **대조군 사다리 추가** — owner: Claude, 수일.
   셔플 Stefan(순서 섞은 물리값), TDD 선형식(제곱근 없이) 조건을 증강 실험에 추가해 "물리 정보의 기여"와 "저복잡도(1-파라미터)의 기여"를 분리. 현재 리뷰어가 물으면 답할 근거가 없다.
3. **관련연구 재배치 + 영문화** — owner: user/Claude, 1~2주.
   물리 모형 출력을 학습 신호로 쓰는 계보를 인용해야 novelty가 산다: Read et al. 2019(WRR, PGDL 합성라벨 사전학습), Jia et al. 2021(PGRNN), Willard et al. 2022(ACM CSUR 서베이), Liu et al. 2023(CRST, PI-LSTM). 차별점을 "**라벨 없는 지역으로의 공간 표적 증강 + 상수 대조 순가치 설계 + 부정확 물리의 해악 정량화**"로 좁혀 서술. 미이행 시 신규성 기각 위험.
4. **CRST 응용 논의 절 신설** — owner: user, 수일.
   CRST는 응용·공학지라 "이 지도·예측구간·AOA가 기반시설 위험 평가·설계에 어떻게 쓰이는가"가 사실상 필수다. 덱 p2의 인프라 위험 논거를 결론 쪽으로 확장.
5. **(P1) 전이 지역 2곳 이상 추가** — owner: Claude, 2~6주, 데이터 확보 선행.
   스칸디나비아·캐나다 군도 CALM 클러스터. 블록 수 확충 후 CI 재산정. W1·W3의 정공법이며, CI가 0을 벗어나면 PPP/TC 상향 도전 가능.
6. **(P1) 외부 제품 head-to-head** — owner: Claude, 1~2주.
   GIPL2·Obu 2019·Ran 2022와 동일 셀에서 비교. 산출물 우위의 외부 준거.
7. **(비차단) 잔여 정합** — `report_overview_figure.py`의 892,865 근거 산출물화, `figures/figure_spec.json`에 이번 재생성 그림 4종(report_overview, transfer_loro_summary, s12_hybrid_summary, alt_annual_fields) 스펙 등록, KPDC 식별자 병기.

---

## 8. Pointers

- **주 보고서**: `outputs/report/main.tex` (13쪽, xelatex+xeCJK). ※ 경로는 `outputs/report/`이고 `docs/report/`는 존재하지 않는다(구 핸드오프의 오기, 이번에 정정).
- **본선 발표덱**: `deck/render/permafrost_final.{pptx,pdf}` 23장. 빌드 `deck/build_final.py`(+`final_lib.py`), 스펙 `deck/deck_spec_final.json`, 그림 `deck/mk_final_figs.py`.
- **발표 문서**: `deck/render/발표대본_ALT_ctrl.docx`(생성기 `deck/mk_script_docx.py`), `발표QA대비_ALT_ctrl.docx`(생성기 `deck/mk_qa_doc.py`, 소스 `deck/qa_answers.md`).
- **핵심 결과 CSV**: `data/processed/{s3_aug_curve_results, s12_hybrid_transfer_shard0-3, s12_hybrid_gate, s11_conformal_results, s4_residual_results, s14_annual_results, map_model_gate_results}.csv`
- **KPDC 파싱**: `scripts/1_data_prep/s7_parse_kpdc_council.py` → `data/processed/s7_council_alt_derived.csv`, `s7_parse_meta.json`. 원자료 `kpdc/`(gitignore, 로컬 전용).
- **Active jobs**: none.
- **Ckpt**: 해당 없음(이번 세션 학습 없음).
- **직전 핸드오프**: [20260729_1741-report-consistency-audit-s13-dropped.md](20260729_1741-report-consistency-audit-s13-dropped.md)
- **아카이브**: `submission/`(2026-07-30 예선 동결본, 수정 금지), `deck/build_{deck,report,midreport,summary}.py`(재실행 금지 배너 추가됨, 폐기 수치 하드코딩)

---

## 9. Caveats for GPT

**용어 정본** (`.claude/project.yaml` naming_canonical 참조, 이번에 현행화됨):
- **물리 잔차 결합** (구 "물리 결합" 폐기 — main.tex 9건 이번에 통일)
- **위성 제품** (구 "CCI 제품"·"위성제품" 폐기 — ESA CCI는 출처명으로만 1회)
- **레나델타** (단독 "레나" 금지), **증강 비율** (붙여쓰기 "증강비율" 금지)
- 검증 3조건: **지역 내 / 공변량만 / 정보 없음**. 조건이 다른 수치는 같은 축에서 비교 금지.
- 실험 단계: 현행 **S1~S14**(S13 폐기). 구 명칭 B0/B0b/B1/B1b/tournament/curated는 main.tex 내 0건이며 폐기 이력일 뿐이다.
- 공변량: **전체 34 / 전이 공통 25 / 지형기후 14 / 광역격자 17 / 연별 9**. project.yaml에 있던 "14종=지형6+기후8"과 "물리관측 PolSAR/InSAR 4"는 낡은 기술이었고 이번에 교체했다.

**가정하지 말 것**:
- "13,606셀이 83개 블록"은 **오류**다. 74가 맞다. 구 핸드오프·`submission/`·EXPERIMENT_LOG 과거 항목에 83이 남아 있으나 이력이다.
- `docs/report/main.tex`는 **존재하지 않는다**. `outputs/report/main.tex`다.
- "제출덱 = permafrost_summary 10장"은 **예선 아카이브**다. 본선은 permafrost_final 23장.
- "17 cm 물리하한", "12.97 cm SOTA 돌파"는 이미 폐기된 주장이다(2026-07-06 정정, 각각 apparent floor·범위축소 아티팩트).
- s12 그림의 "물리식 단독"은 이제 **CCI 유효 셀 기준**이다. 이전에 렌더된 그림(커밋 0eaccaf 이전)과 수치가 다르다.
- 콘슬 KPDC의 "136.4 cm"는 심부 시추공 부분표본이지 사이트 대표값이 아니다. 탐침 "35.0"도 초여름 측정이 섞인 셀평균이라 연최대 기준으로는 41.6이 타당하다(단 41.6은 main.tex 미수록).
- 이번 세션에 **새 실험·학습은 없다**. 모든 수치는 기존 CSV에서 검증하거나 정정한 것이다.
