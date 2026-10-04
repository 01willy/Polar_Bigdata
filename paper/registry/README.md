# 실험 등록부(paper/registry)

**성격**: 논문 작업용 실험 색인이다. 원래 결과 파일은 옮기거나 이름을 바꾸지 않는다. 이 폴더의 CSV 는 원래 경로를 가리키기만 한다. 작성 2026-10-04, 경로와 행 수는 같은 날 `ls`, `wc -l`, `git ls-files`, `git check-ignore` 로 확인했다.

## 1. 파일

| 파일 | 내용 |
|---|---|
| `experiments.csv` | 실험 묶음 등록부. 66행(헤더 제외), 18열, UTF-8 |
| `naming_map.csv` | 옛 실험 id 와 새 이름의 대응. 82행(헤더 제외), 4열 |
| `README.md` | 이 문서. 열 정의와 갱신 방법 |

## 2. 행 구성

이전 세션에서 정리한 47묶음을 모두 담았다. 한 행에는 새 이름 하나만 둔다. 그래서 묶음 수와 행 수가 다르다.

| 단계(phase) | 행 수 | 내용 |
|---|---|---|
| H0 | 25 | 2026-06-30 – 08-31 탐색과 대회기(PRE-A, PRE-B, P0/P1, P2, PH1, W2.1, W3, AK-AUG, TL→S9, SH3D→S10, S0–S8, S11, E1-K, E2-D, S12, S13, S14+MAP) |
| H1 | 3 | 09-08 사전 등록 E1·E3, E2·E4, 09-14 대조 감사와 09-15 전이 계획 |
| H2 | 1 | M1 |
| H3 | 1 | H18–H24 |
| H4 | 1 | H25–H30 |
| H5 | 3 | 09-26 통합 실험, NEXT P1–P6(미실행), FA |
| T | 8 | LG, LGX, LGT, LGF, LGD, LGU, WRAPUP, LGW(J7)+레나 지도 |
| W | 12 | WF0, WF1, WF6, WF2, WF8, WF3, WF4, WF5(미발행), WF7, WF9, WF9-soil, WF10 |
| X | 12 | 새 묶음 XA–XJ(2026-10-04 등록)와 추가 등록 XK·XL(2026-10-05) |

- 47묶음 표의 'WF1–WF5', 'WF6–WF8', 'WF9–WF10' 은 새 이름이 서로 달라 실험별 행으로 나눴다. `old_ids` 열에 원래 묶음 이름을 괄호로 적었다(예: `WF2(WF1–WF5 묶음)`).
- 같은 새 이름을 여러 행이 쓸 수 있다(T3 = LGT·LGF, T6 = WRAPUP·LGW, W2 = WF1·WF6, W3 = WF2·WF8, W7 = WF9·WF9-soil, H0 = 대회기 25묶음). 행의 고유 키는 `(new_name, old_ids)` 다.

## 3. 열 정의(experiments.csv)

| 열 | 정의 | 값 규칙 |
|---|---|---|
| `new_name` | 새 이름(사용자 승인 체계, `paper/README.md` 3절) | T1–T6, W1–W8, H0–H5, X 묶음, `WF5_not_issued` |
| `old_ids` | 원래 실험 id 와 묶음 이름 | 이름이 겹치는 옛 id 는 날짜나 '…와 다름'을 붙였다 |
| `phase` | 단계 | H0–H5(이력), T(전이 실험), W(워크플로 실험), X(계획) |
| `dates` | 실행 또는 기록 날짜 | `YYYY-MM-DD`, 범위는 `–` |
| `purpose_ko` | 목적 한두 문장 | 보고서 문체 |
| `input_data` | 입력 자료 경로와 규모 | 규모는 확인한 값만 적고 나머지는 생략 |
| `covariates` | 입력 공변량 집합 | x25 = 지형 6 + ERA5-Land 기후 8 + SoilGrids 토양 9 + CCI 2(`src/polar/fidelity.py` SHARED_CORE). x34 = x25 + InSAR 5 + PolSAR 3 + 결측 표지 1(COVARIATE_CORE). 확인하지 못한 칸은 `[미확인]` |
| `protocol` | 분할·채점·통계 규약 | 블록 A/B, 분할 수, 라벨 수 격자, 재표집 횟수 |
| `code_paths` | 코드 경로(`;` 구분) | 존재 확인. 아직 없는 파일은 `[예정]` |
| `main_outputs` | 주 산출 경로(`;` 구분) | 존재 확인. 계획 행은 `[예정]` |
| `output_status` | 산출 상태 | 아래 4절 |
| `compute` | 계산 자원과 비용 | Rescale 작업 id, 노드, 목록가(달러) 또는 로컬 GPU·CPU |
| `status` | 실행 상태 | 첫 낱말이 `done`, `superseded`, `discarded`, `not run`, `planned` 가운데 하나. 괄호는 보충(예: `done(기각)`, `done(J1)`, `superseded(M1)`) |
| `verdict_summary_ko` | 판정과 핵심 수치 | 수치는 `verdict_doc` 의 판정 기록에서 옮겼다. Δ 는 cm, 음수가 개선 |
| `verdict_doc` | 판정 정본 문서와 절·행 | `LOG` 행 번호는 `docs/EXPERIMENT_LOG.md` 의 2026-10-04 기준 행 |
| `claims` | 관련 주장 폴더(`;` 구분) | `C1_label0_safety`, `C2_bias_diagnosis`, `C3_structure_by_label_count`, `C4_sufficient_labels`, `C5_method_selection`, `C6_label_placement`, `C7_evaluation_design`, `C8_gain_source`, `D_data_and_design`, `SI_uncertainty`, `SI_learners_new_regions`. 없으면 `-` |
| `paper_use` | 원고에서의 쓰임 | `main`(본문 결과), `SI`, `background`(서론·토의·이력 서술), `none`. 둘 이상이면 `;` 로 잇고 앞이 주 용도다 |
| `retracted_claims` | 이 묶음과 관련해 철회·사용 금지된 문장 | 따옴표 안이 철회 문장, 괄호가 근거 |

## 4. output_status 표기

| 표기 | 뜻 |
|---|---|
| `exists` | 파일이 있고 git 이 추적한다 |
| `untracked` | 파일이 있으나 git 이 추적하지 않는다(커밋 전 결과 표) |
| `untracked(ignored)` | 파일이 있고 `.gitignore` 로 제외된다(대용량 입력) |
| `missing(…)` | 산출이 없다. 괄호에 이유 |
| `archived` | 보관 폴더로 옮겨졌다. 2026-10-04 기준 해당 행 없음 |
| `planned(미생성)` | 계획 행. 아직 만들지 않았다 |
| `NL` | `wc -l` 줄 수(헤더 포함). 행 수는 N − 1 이다. 따옴표 안 줄바꿈이 있는 CSV 는 실제 행 수와 다를 수 있다 |

같은 파일을 여러 행이 공유하면 `(WF1–WF4 공용)` 처럼 적었다.

## 5. 갱신 방법

1. **원래 결과를 건드리지 않는다.** 결과 파일, 스크립트, 판정 문서는 옮기거나 고치지 않는다. 이 등록부만 고친다.
2. **새 실험(X 묶음 실행 등)**: 사전 등록 문서를 먼저 커밋한다. 그다음 해당 X 행의 `code_paths`, `main_outputs` 에서 `[예정]` 을 지우고 실제 경로를 적는다. 판정을 기록한 뒤 `status`, `verdict_summary_ko`, `verdict_doc` 를 채운다. 판정 정본은 계획 문서의 결과 절이고 이 CSV 는 요약이다.
3. **새 이름**: 승인된 체계(T, W, H, X)를 따른다. 새 묶음이 생기면 `naming_map.csv` 에 옛 id 행을 같이 더한다.
4. **경로와 행 수 점검**: 고친 뒤 아래 명령으로 경로를 확인하고 `output_status` 의 줄 수를 다시 적는다.

```bash
cd /home/willy010313/Polar_Bigdata
OMP_NUM_THREADS=1 python3 - <<'EOF'
import csv, os, re
for r in csv.DictReader(open('paper/registry/experiments.csv', encoding='utf-8')):
    for col in ('code_paths', 'main_outputs'):
        if '[예정]' in r[col]:
            continue
        for tok in re.findall(r'[\w./\-]+\.(?:py|csv|md|yaml|sh|json|pdf)|[\w./\-]+/(?=;|$)', r[col]):
            if not os.path.exists(tok):
                print('없음:', r['new_name'], r['old_ids'][:20], tok)
EOF
wc -l <경로>                      # 줄 수
git ls-files --error-unmatch <경로> # 추적 여부
git check-ignore -v <경로>          # 제외 여부
```

5. **문체**: 보고서 문체(~이다, ~한다), 명사형 제목, 접속 기호로서의 줄표 금지(숫자 범위의 `–` 는 허용). 확인하지 못한 값은 `[미확인]` 으로 표시하고 지어내지 않는다.
6. **기록**: 고친 날짜와 이유를 이 문서 끝 '개정 이력'에 한 줄로 적는다.

## 개정 이력

- 2026-10-04: 초판. 47묶음을 64행(계획 X 10행 포함)으로 등록. 경로 점검에서 존재하지 않는 경로는 없었다. 이전 표의 와일드카드 경로(`e4_*.py`, `h31_*` 등, `h44_*.py`–`h46_*.py`)를 실제 파일 이름으로 펼쳤고, M1 의 `cv_scheme_comparison.csv` 를 만든 `scripts/3_deep_learning/m1_cv_scheme_comparison.py`, H25–H30 의 `h29_block_label_value.py`·`h29_resummary.py`, H18–H24 의 `h19_blockE.py`·공변량 확장 스크립트, S7 의 `scripts/1_data_prep/s7_parse_kpdc_council.py` 를 더했다.
- 2026-10-05: X 묶음 봉인 표 열람 결과 반영(계획 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8.2–8.9, 추가 등록 문서 결과 절). XB, XD, XE, XG, XH, XI, XJ 행의 실행 상태·판정 요약·근거 경로를 고쳤고, XK_support_scale·XL_map_products 행을 더했다(experiments.csv 66행, naming_map.csv 82행). 경로 점검에서 없는 경로는 없었다. XA 행은 이번에 고치지 않았다(판정 기록은 계획 8.1).
- 2026-10-05 05:40: XC_workflow_end_to_end 행을 봉인 표 열람 결과로 고쳤다(계획 8.10. 실행 상태, 판정 요약, 근거 경로, 주장 C5·C6·C2·C1). 경로 점검에서 없는 경로는 없었다.
- 2026-10-05 05:55: XD(XD-5 서술), XB(CCI v5 민감도 판 cci5y), XE(XE-e 진단) 행을 계획 8.11, 8.12, 8.6 으로 고쳤다. 경로 점검에서 없는 경로는 없었다.
