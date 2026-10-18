#!/usr/bin/env python3
"""make_si_tables.py : Supplementary Tables S1-S18 -> sections/si_tables.tex (assembly 2026-10-05).

Sources (no new computation; values are copied, not recomputed):
  * paper/claims/*/README.md evidence tables (S4-S12): cells are copied from the README
    markdown tables; Korean words are replaced by the glossary below (ko -> en). Columns
    that hold only file paths or row filters are dropped; the README section is named in
    each block heading so that every row can be traced to its source row filter.
  * paper/claims/D_data_and_design/tables/Table1_data.csv (v2 Table 1 copy, S1), README E19.
  * paper/claims/C1_label0_safety/tables/lgw_bundle.csv (S3, abstract bundle, spec 8 source)
    and lgw_tests.csv (S14 deployment scenarios, registered verdict rows).
  * docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md M4.3-M4.5 (S2 register frame, S15) and
    docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md (S16 Table 4, S17 Data availability).
  * paper/claims/C2_bias_diagnosis/XA_result_2026-10-04.md section 4 (S13, XA rows).
Round 3 (5 October 2026): S13 parts c to o (XB to XL) come from tools/si_x_results.py (values copied from FINAL_BATCH 8.2-8.9
and the XK/XL addendum); XC rows stay as placeholders and XF, XE-a/XE-b and XD-5 carry [PENDING] markers.
Usage: python3 tools/make_si_tables.py   (run from paper/manuscript/en)
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import si_x_results as XR  # noqa: E402
import si_numbering as SN  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CLAIMS = os.path.join(ROOT, "paper", "claims")
OUT = os.path.join(HERE, "sections", "si_tables.tex")
LOG = os.path.join(HERE, "build", "si_tables.log")
HANGUL = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")

# ----------------------------------------------------------------------------------------
# markdown tables
# ----------------------------------------------------------------------------------------

def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, i, tick = [], "", 0, False
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if c == "`":
            tick = not tick
        if c == "|" and not tick:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    cells.append(cur.strip())
    return cells


def md_sections(path):
    out, head, tables, cur = [], None, [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            if cur:
                tables.append(cur)
                cur = None
            if head is not None:
                out.append((head, tables))
            head, tables = line.lstrip("#").strip(), []
            continue
        if line.strip().startswith("|"):
            cells = split_row(line)
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                continue
            cur = cur or []
            cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = None
    if cur:
        tables.append(cur)
    out.append((head, tables))
    return out


def md_table(path, sec_prefix, idx=0):
    for h, ts in md_sections(path):
        if h.startswith(sec_prefix + " ") or h == sec_prefix:
            return ts[idx]
    raise SystemExit(f"table not found: {path} {sec_prefix} {idx}")


# ----------------------------------------------------------------------------------------
# glossary (longest keys first). Values were read from the README tables; only words change.
# ----------------------------------------------------------------------------------------
GLOSSARY = [
    ("통계적으로 구별되나 크기는 0.5 cm 미만", "distinguishable, size below 0.5 cm"),
    ("통계적으로 구별되나 크기는", "distinguishable, size"),
    ("크기 0.5 cm 미만 표지", "flag: size below 0.5 cm"), ("크기 0.5 cm 미만", "size below 0.5 cm"),
    ("크기는 0.5 cm 미만", "size below 0.5 cm"), ("0.5 cm 미만", "below 0.5 cm"),
    ("분할 독립 가정 의존", "depends on split independence"), ("보정 전 유의", "significant before correction"),
    ("두 판정 열 비교", "two verdict columns compared"), ("공통 판정", "common-CI verdict"),
    ("공통 CI", "common CI"), ("블록 등가중", "block-equal"), ("셀 가중", "cell-weighted"),
    ("희소 라벨에서도 ML 순가치 있음", "net value of ML with sparse labels"),
    ("희소 라벨에서 순가치 있음", "net value with sparse labels"),
    ("차이를 확인하지 못했다", "no difference established"), ("확인하지 못함", "not established"),
    ("지지하지 않음", "not supported"), ("부분 지지", "partially supported"), ("판정 불가", "not decidable"),
    ("판정 가능한", "decidable"), ("규칙 비작동", "rule not operating"), ("안전성 미확인", "safety not confirmed"),
    ("우세", "lower error"), ("열세", "higher error"), ("동등", "equivalent"), ("미결정", "undetermined"),
    ("지지", "supported"), ("기각", "rejected"), ("서술", "descriptive"), ("강건", "robust"), ("약화", "weakened"),
    ("혼재", "mixed"), ("비열등", "non-inferior"), ("미확인", "not confirmed"), ("해당 없음", "not applicable"),
    ("결측", "missing"), ("사후", "post hoc"), ("없음", "none"), ("있음", "present"),
    ("결과 열람 뒤 설계", "designed after viewing results"), ("재현(비맹검)", "replication (unblinded)"),
    ("비맹검", "unblinded"), ("맹검", "blind"), ("교차 환경", "cross-environment"), ("불통과", "not passed"),
    ("예측된 결과", "predicted outcome"), ("소수 블록", "few blocks"), ("라벨 통계 열람", "label statistics viewed"),
    ("위치가 다른 라벨 집합", "label set at other locations"),
    ("레나델타", "Lena Delta"), ("레나·캐나다", "Lena Delta and Canada"), ("레나", "Lena Delta"),
    ("캐나다", "Canada"), ("러시아 서부", "W Russia"), ("러시아 동부", "E Russia"), ("러시아 중부", "Central Russia"),
    ("러시아 W", "W Russia"), ("러시아 E", "E Russia"), ("러시아 C", "Central Russia"),
    ("알래스카", "Alaska"), ("티베트", "Tibetan Plateau"), ("그린란드", "Greenland"), ("북대서양", "North Atlantic"),
    ("주 4지역 평균", "four main regions, mean"), ("주 4지역", "four main regions"), ("4지역 평균", "four-region mean"),
    ("층화 평균", "stratified mean"), ("3지역 보조 열", "three-region auxiliary column"), ("3지역", "three regions"),
    ("2지역", "two regions"), ("지역 내", "within region"), ("지역 행", "region rows"), ("지역 수", "number of regions"),
    ("독립 지역", "independent regions"), ("새 지역", "new regions"), ("하위 지역", "sub-regions"),
    ("지역", "regions"), ("대상", "targets"),
    ("라벨 전량", "all labels"), ("라벨 10개", "ten labels"), ("라벨 0", "zero labels"), ("전량", "all"),
    ("라벨", "labels"), ("유사라벨", "pseudo-labels"),
    ("원천 계수 Stefan", "source-coefficient Stefan"), ("재보정 물리식", "recalibrated physics model"),
    ("재보정 이득", "recalibration gain"), ("재보정 몫", "recalibration share"), ("재보정", "recalibration"),
    ("현지 최소제곱", "local least squares"), ("수축", "shrinkage"), ("물리 입력 구조", "physics-input structure"),
    ("잔차 구조", "residual structure"), ("물리 입력", "physics input"), ("물리식", "physics model"),
    ("최선 물리 보정식", "best physics calibration"), ("최선 ML", "best ML"),
    ("증강 결합", "augmented combination"), ("증강", "augmentation"), ("잔차 학습", "residual learning"), ("잔차", "residual"),
    ("곱셈", "multiplicative"), ("라벨 셔플", "label shuffle"), ("섞은", "shuffled"), ("위약", "placebo"),
    ("직접 ML", "direct ML"), ("학습기", "learner"), ("주 학습기", "main learner"), ("랜덤 포레스트", "random forest"),
    ("용량 확대", "higher capacity"), ("원천 교차검증 조정", "source-CV tuning"), ("원천 지역 하나 제외 교차검증 조정", "leave-one-source-region-out tuning"),
    ("다중 헤드", "multi-head"), ("축소", "reduced"), ("컨텍스트 상한 10,000행", "context cap 10,000 rows"),
    ("원천 전체 컨텍스트", "full source context"), ("같은 컨텍스트", "same context"), ("로컬 병기 행", "local companion row"),
    ("병기 행", "companion row"), ("병기", "companion"), ("로컬", "local"), ("한계 행", "limit row"),
    ("저가중 잔차", "low-weight residual"), ("잔차 가중", "residual weight"), ("물리 증강", "physics augmentation"),
    ("증강 + 앵커 + 잔차", "augmentation + anchor + residual"), ("앵커", "anchor"),
    ("구성 고정 풀 곡선", "fixed-composition pool curve"), ("연도 정합 도일", "year-matched degree-days"),
    ("물리식·제품 앙상블", "physics-model and product ensemble"), ("앙상블", "ensemble"), ("제품", "product"),
    ("격자 안", "within grid"), ("격자 사이", "between grids"), ("기온", "air"), ("토양", "soil"),
    ("설명 비율", "explained share"), ("비율", "share"), ("순위상관", "rank correlation"), ("상관", "correlation"),
    ("편향", "bias"), ("계수 오차", "coefficient error"), ("진단값", "diagnostic"), ("진단", "diagnostic"),
    ("이득", "gain"), ("몫", "share"), ("오차 하한", "error floor"), ("오차", "error"),
    ("블록 층화", "block stratification"), ("공변량 최대 최소 거리", "covariate max-min distance"),
    ("공변량 군집 층화", "covariate-cluster stratification"), ("능동 선택", "active selection"), ("능동 선정", "active selection"),
    ("셀 무작위", "random cells"), ("무작위", "random"), ("블록 분산", "block-dispersed"), ("전략", "strategy"),
    ("교차검증", "cross-validation"), ("규칙", "rule"), ("고정 레시피", "fixed recipe"), ("레시피", "recipe"),
    ("선택 횟수", "selection counts"), ("선택", "selection"), ("구간 점수", "interval score"), ("구간", "interval"),
    ("포함률", "coverage"), ("커버리지", "coverage"), ("폭", "width"), ("정규화기", "normalizer"),
    ("보정 전 분위 구간", "uncalibrated quantile interval"), ("분위", "quantile"), ("계층", "hierarchical"),
    ("셀 풀링", "cell pooling"), ("생성", "generative"), ("확인적", "confirmatory"), ("보조", "auxiliary"),
    ("판정 문구", "verdict wording"), ("판정 규칙", "decision rule"), ("판정", "verdict"), ("등록 가설", "registered hypothesis"),
    ("등록", "registered"), ("가설", "hypothesis"), ("주 근거", "main evidence"), ("첫 시험", "first test"),
    ("분할 5회", "five splits"), ("분할", "splits"), ("추출", "draws"), ("재표집", "resamples"), ("중앙값", "median"),
    ("평균", "mean"), ("최솟값", "minimum"), ("최댓값", "maximum"), ("최소", "minimum"), ("범위", "range"),
    ("같은 대비의 4분 보조 열", "four-way auxiliary column of the same contrast"), ("같은 대비", "same contrast"),
    ("같은 필터", "same filter"), ("같은 파일", "same file"), ("같은 표", "same table"), ("같음", "same"), ("같고", "same, with"),
    ("〃", "same"), ("이번 점검", "this check"), ("이번 계산", "this calculation"), ("위험 수", "risk count"),
    ("셀", "cells"), ("블록", "blocks"), ("행", "rows"), ("열", "column"), ("값", "value"), ("총", "total"),
    ("확장 풀", "extended pool"), ("지역의 판정이 확장 풀", "regional verdict in the extended pool"),
    ("유지된다", "is maintained"), ("유지되지 않는다", "is not maintained"),
    ("본 실행", "main run"), ("실행", "run"), ("원천 지지 밖", "outside source support"), ("절댓값", "absolute value"),
    ("백분위", "percentile"), ("전이", "transfer"), ("정의", "definition"), ("저장소", "store"), ("성분", "component"),
    ("키 전체의", "over all keys of"), ("분해", "decomposition"), ("상수 폭", "constant width"), ("상수", "constant"), ("반폭", "half-width"),
    ("참조", "reference"), ("짝지음", "paired"), ("조건부", "conditional"), ("관측", "observed"), ("양 끝", "both ends"),
    ("대회", "contest"), ("채택", "adopted"), ("원고 조치", "manuscript action"), ("완화", "relaxed"), ("재채점", "re-scored"),
    ("경계", "bounds"), ("주 방법", "main method"), ("목표 띠 밖", "outside the target band"), ("검정 불가", "not testable"),
    ("정밀도 미달", "precision insufficient"), ("보정 그대로", "as calibrated"), ("로그", "log"), ("정합", "matched"),
    ("보정 불확실성 의존", "depends on calibration uncertainty"), ("종합", "overall"), ("환산", "converted"), ("그림 표", "figure table"),
    ("무한", "infinite"), ("통과", "passed"), ("회복", "recovery"), ("확인", "confirmed"), ("상한", "upper bound"), ("하한", "lower bound"),
    ("넷째 자리", "fourth decimal"), ("산술", "arithmetic"), ("초록 묶음", "abstract bundle"), ("초록 규칙", "abstract rule"),
    ("초록", "abstract"), ("본문", "main text"), ("묶음", "bundle"), ("항목 안", "within item"), ("위 넷의", "of the four above"),
    ("위와", "as above"), ("모두", "all"), ("둘 다", "both"), ("다섯", "five"), ("풀 전체", "whole pool"), ("풀", "pool"),
    ("단계", "stage"), ("요약표", "summary table"), ("공개 TabM 과 다름", "not the public TabM"), ("물리 출력 입력", "physics-output input"),
    ("컨텍스트 상한", "context cap"), ("방향을 쓴다", "direction stated"), ("한계를 명시한다", "limit stated"), ("해석 주의", "note"),
    ("보정 전", "before correction"), ("포함", "included"), ("제외", "excluded"), ("심부 레짐", "deep regime"), ("형식", "form"),
    ("관계", "relation"), ("기여", "contribution"), ("민감도", "sensitivity"), ("대체", "substitute"), ("어긋난", "diverging"),
    ("불충족", "not met"), ("충족", "met"), ("대칭", "symmetric"), ("부적격", "ineligible"), ("적격", "eligible"), ("합집합", "union"),
    ("변형 불가", "variant not possible"), ("실질 변형", "substantive variants"), ("확충판", "expanded editions"),
    ("사전분포 오지정 의존", "depends on prior misspecification"), ("상대 단위", "relative units"), ("재현 실패", "replication failed"),
    ("구별되지 않는다", "indistinguishable"), ("조각", "shards"), ("근거", "basis"), ("고도", "elevation"), ("채점", "scoring"),
    ("외삽", "extrapolation"), ("완료되지 않은", "incomplete"), ("작업 단위", "work units"), ("블록 수", "number of blocks"),
    ("셀 수", "number of cells"), ("대상 수", "number of targets"), ("양의", "positive"), ("음의", "negative"), ("부호 있는", "signed"),
    ("한쪽 p", "one-sided p"), ("고정", "fixed"), ("낙관적", "optimistic"), ("가운데", "among"), ("열람", "viewing"), ("인용", "cited"),
    ("계산", "calculation"), ("분포", "distribution"), ("해석식", "analytic"), ("의존", "dependence"), ("빈 값", "empty value"),
    ("불성립", "not met"), ("성립", "met"), ("파일", "file"), ("표지", "flag"), ("판정어", "verdict word"), ("우선", "first"),
    ("순차", "sequential"), ("원천 외삽", "source extrapolation"), ("지온 유도", "temperature-derived"), ("확인분", "verified"),
    ("약관", "licence"), ("확충", "expanded"), ("대비", "contrast"), ("항목", "item"), ("단위", "unit"), ("승인", "approval"),
    ("주 4", "main 4"), ("주 설정", "main setting"),
    ("원천 교차검증 조정이 전이 오차를 키웠다", "source-CV tuning increased the transfer error"), ("조정판의 오차가 작다", "the tuned version had lower error"),
    ("동등성 판정 불가", "equivalence not decidable"), ("정밀도 미달 대비", "contrasts with insufficient precision"), ("전체 문장 조건 미충족", "condition for a learner-general sentence not met"),
    ("강건 문장 규칙을 채운 학습기", "learners meeting the robust-sentence rule"), ("다중 헤드 MLP", "multi-head MLP"), ("축소 FT-T", "reduced FT-T"), ("전체", "all"),
    ("파일", "file"), ("최솟값", "minimum"), ("의 행 수", " row count"), ("행 수", "row count"), ("평균", "mean"), ("같은 조건", "same condition"), ("최근접 거리 중앙값", "median nearest distance"),
    ("원 열 부호 그대로", "signs as in the source column"), ("개정 점검", "revision check"), ("이득 기준", "gain basis"), ("최대 차", "maximum difference"), ("이므로", "so"),
    ("알 수 없음", "unknown"), ("쓰지 않는다", "not used"), ("안에서 같다", "same within"), ("최악", "worst"), ("아래", "below"),
    ("독립", "independent"), ("칸", "cell"), ("직접", "direct"), ("전용", "only"), ("공변량", "covariate"), ("학습 쪽", "training side"),
    ("통계", "statistics"), ("바꿔도", "changed"), ("4분", "four-way"), ("있다", "exists"), ("없다", "none"), ("적어", "few"),
    ("문서", "document"), ("재현되지 않음", "not reproduced"), ("재현", "reproduced"), ("정확히", "exactly"), ("읽는 법", "reading notes"),
    ("이상", "or more"), ("미만", "below"), ("초과", "above"), ("전부", "all"), ("엇갈린", "conflicting"), ("가중", "weighting"),
    ("형태", "form"), ("문장", "sentence"), ("결과는", "result"), ("결과", "result"), ("나왔다", "came from"), ("이지만", "but"),
    ("표에서", "in the table"), ("단", "rungs"), ("부분", "partial"), ("같은", "same"), ("같다", "same"), ("중", "of"), ("판", "edition"),
    ("계수", "coefficient"), ("입력에는", "inputs have"), ("뒤", "after"), ("보았다", "viewed"), ("다름", "differs"), ("양", "quantity"),
    ("두", "two"), ("네", "four"), ("별", "by"), ("차", "difference"), ("쪽", "side"), ("못한", "not"), ("하지 않은", "not"),
    ("점 추정", "point estimate"), ("및", "and"), ("개선", "improvement"), ("악화", "deterioration"),
    ("유의", "significant"),
]
GLOSSARY.sort(key=lambda kv: -len(kv[0]))

MANUAL_EN = {  # round 11 (5 October 2026): English renderings of the 17 cells the glossary could not translate (same meaning, no new claims)
    "'지지: 잔차 구조는 물리 입력 구조보다 오차가 작다'; holm_p 0.0023(항목 안 보조 열)":
        "'Supported: the residual structure has lower error than the physics-input structure'; holm_p 0.0023 (auxiliary column within the item)",
    "−3.45 [−4.73, −2.37] / [−3.98, −2.09]; −4.37 [−5.63, −3.29] / [−5.16, −3.01]. 지역 행: R0@tddm − F1k 레나 −3.92, 캐나다 −1.28, 러시아 W −4.61, 러시아 E −3.98; R1@tddm − F1k 레나 −2.29, 캐나다 +1.88, 러시아 W −13.44, 러시아 E −3.61. verdict: 'P* = P0@tddm. L4, L8, L10 의 대비를 P* 대비로 병기한다'":
        "−3.45 [−4.73, −2.37] / [−3.98, −2.09]; −4.37 [−5.63, −3.29] / [−5.16, −3.01]. Region rows: R0@tddm − F1k Lena Delta −3.92, Canada −1.28, W Russia −4.61, E Russia −3.98; R1@tddm − F1k Lena Delta −2.29, Canada +1.88, W Russia −13.44, E Russia −3.61. Verdict: 'P* = P0@tddm. The contrasts of L4, L8 and L10 are also reported against P*'",
    "n 10: −0.12, −0.26; 전량: −0.60 [−0.80, −0.36] / [−0.64, −0.32], −1.69 [−2.48, −0.84] / [−2.01, −0.94]. '대상 라벨의 공변량과 잔차의 관계가 기여한다: n=-1 λ=0.25'":
        "n 10: −0.12, −0.26; all labels: −0.60 [−0.80, −0.36] / [−0.64, −0.32], −1.69 [−2.48, −0.84] / [−2.01, −0.94]. 'The relation between the covariates and the residuals of the target labels contributes: n=-1 λ=0.25'",
    "우세(n 3·10 은 '크기는 0.5 cm 미만', blind False 재현(비맹검); n 3 은 항목 안 Holm 보조 열 '보정 전 유의'). P1* 표지: n 3·320 'P1* 를 확인하지 않은 n', n 40·160 'P1* 대비 미계산(P1 대비만 보조)'. MEAN3 동등·동등":
        "Lower error (n 3 and 10 'size below 0.5 cm', blind False, replication (unblinded); n 3 'significant before correction' in the auxiliary Holm column within the item). P1* flag: n 3 and 320 'n at which P1* was not checked', n 40 and 160 'P1* contrast not computed (P1 contrast only, auxiliary)'. MEAN3 equivalent and equivalent",
    "n 3 미결정, n ≥ 10 우세. P1* 표지: n 3·320 'P1* 를 확인하지 않은 n', n 40·160 'P1* 대비 미계산(P1 대비만 보조)'. MEAN3 n 10 동등, n 40 우세('크기는 0.5 cm 미만')":
        "n 3 undetermined, n ≥ 10 lower error. P1* flag: n 3 and 320 'n at which P1* was not checked', n 40 and 160 'P1* contrast not computed (P1 contrast only, auxiliary)'. MEAN3 n 10 equivalent, n 40 lower error ('size below 0.5 cm')",
    "[-2.02, 0.62] / [-2.18, 0.33], 미결정(`ci_dependence` 빈 값. 셀 가중 상한 0.62 > 0.5 cm 라 공통 CI 로는 비열등 불성립, 이번 점검)":
        "[-2.02, 0.62] / [-2.18, 0.33], undetermined (ci_dependence empty; the cell-weighted upper limit 0.62 exceeds 0.5 cm, so non-inferiority is not met under the common CI; this check)",
    "지지. `scope == 'verdict'` 행의 `verdict` 열: '지지: 양의 순위상관(ρ 0.38, CI [0.17, 0.55])'. 같은 행 `role` 주, `confirmatory` False, `blind` True, `design_note` '결과 열람 뒤 설계'":
        "Supported. Column verdict of the rows with scope == 'verdict': 'Supported: positive rank correlation (ρ 0.38, CI [0.17, 0.55])'. Same rows: role main, confirmatory False, blind True, design_note 'designed after viewing results'",
    "레나·캐나다 우세, 러시아 W 미결정, 러시아 E 열세. 같은 대비의 LGX 보조 열(`data/processed/paper_figs/fig3_c.csv`, L4 행)은 러시아 E [−0.02, 1.28] 미결정이다(재표집 의존)":
        "Lena Delta and Canada lower error, W Russia undetermined, E Russia higher error. In the LGX auxiliary column of the same contrast (data/processed/paper_figs/fig3_c.csv, L4 rows), E Russia is undetermined, [−0.02, 1.28] (depends on the resampling)",
    "판정 불가(대비 0/2; 없는 대비: n=100, n=전량)":
        "Not decidable (contrasts 0/2; missing contrasts: n=100, n=all)",
    "(a)10 병기. P2 RMSE 108.12. 셀 가중 CI 가 전부 0 초과(P2 가 더 정확)이고 블록 등가중 CI 만 0 을 포함한다. '구별되지 않는다' 로 쓰지 않는다":
        "(a)10 reported alongside. P2 RMSE 108.12. All cell-weighted CIs lie above 0 (P2 more accurate) and only the block-equal CIs include 0. Not to be written as 'indistinguishable'",
    "재현 실패 지역(L38 (b)). 셀 가중 CI 는 전부 0 미만이고 블록 등가중 CI 는 0 을 포함한다(두 가중이 엇갈린 미결정). 셀 가중 점 추정과 블록 등가중 CI 를 한 괄호에 묶지 않는다":
        "Region where replication failed (L38 (b)). All cell-weighted CIs lie below 0 and the block-equal CIs include 0 (undetermined, the two weightings disagree). Cell-weighted point estimates and block-equal CIs are not combined in one bracket",
    "test_id=L41, scope=verdict, 열 `verdict`, `stat`, `note`, `same_as`; 대상 셀 수는 원천 `tables/lgd_eligibility_v1.csv` 의 `n_cells`, `eligible`":
        "test_id=L41, scope=verdict, columns verdict, stat, note, same_as; target cell counts from the columns n_cells and eligible of the source tables/lgd_eligibility_v1.csv",
    "티베트: (a)(c)(h) 는 대상 셀 0개라 '변형 불가(적격 표에서 부적격)' 다. (b)(d)(e)(f)(g) 는 '강건' 이지만 모두 주 설정과 같은 실행 표(same_as, 132셀)라 note 가 '주 설정의 TMx 를 그대로 써서 모든 행이 주 설정과 같다' 고 적었다. 따라서 티베트에는 라벨 집합 민감도가 없다. 러시아 중부: (a) 변형 불가(12셀, 부적격), (h) 점 추정만, (b) '약화'(D0-P0|n3 열세 → 미결정), (e) '약화'(D0-P0|n10 미결정 → 열세), (d)(f) '강건'(다시 적합한 변형), (c)(g) '강건' 은 same_as 라 민감도가 아니다":
        "Tibetan Plateau: (a), (c) and (h) have no target cells and are 'variant not possible (ineligible in the eligibility table)'. (b), (d), (e), (f) and (g) are 'robust' but all use the run table of the main setting (same_as, 132 cells), and the note records that every row equals the main setting because the TMx of the main setting is used unchanged; the Tibetan Plateau therefore has no label-set sensitivity. Central Russia: (a) variant not possible (12 cells, ineligible), (h) point estimate only, (b) 'weakened' (D0-P0|n3 higher error to undetermined), (e) 'weakened' (D0-P0|n10 undetermined to higher error), (d) and (f) 'robust' (refitted variants); (c) and (g) 'robust' are same_as runs and not sensitivity results",
    "원천 Delig, spec ∈ {NAtlantic_lic, Tibet_L42_lic, Russia_C_L42_lic}, 열 `n_cells, nb_union, min_nb_eval_used, eligible, regime`":
        "Source Delig, spec ∈ {NAtlantic_lic, Tibet_L42_lic, Russia_C_L42_lic}, columns n_cells, nb_union, min_nb_eval_used, eligible, regime",
    "서술. 짝지음 CI 는 원천 표에 없음. 문서의 [0.6, 6.5] 는 정확히 재현되지 않음(추출 의존, 아래 읽는 법)":
        "Descriptive. No paired CI in the source table. The interval [0.6, 6.5] of the source document is not reproduced exactly (it depends on the draws; reading notes of the source README)",
}

UNTRANSLATED = []


def tr(text, where=""):
    """glossary replacement; any Hangul left is dropped from the cell and logged."""
    if text.strip() in MANUAL_EN:
        return MANUAL_EN[text.strip()]
    t = text.replace("**", "").replace("`", "")
    t = re.sub(r"(\d)\s?개", r"\1", t)
    t = re.sub(r"(\d)\s?회", r"\1 times", t)
    t = re.sub(r"(\d)\s?대상", r"\1 targets", t)
    t = re.sub(r"(\d)\s?지역", r"\1 regions", t)
    t = re.sub(r"(\d)\s?행", r"\1 rows", t)
    for ko, en in GLOSSARY:
        if ko in t:
            t = t.replace(ko, en)
    t = re.sub(r" 대 ", " vs ", t)
    t = re.sub(r"(?<=\S)\s(?:으로|에서|이다|이고|이며|은|는|이|가|을|를|의|에|와|과|도|만|로|라)(?=[\s,.;:)\]']|$)", "", t)
    t = re.sub(r"(?<=[A-Za-z0-9)\]'%*])(?:으로|에서|이다|이고|이며|이라|라고|은|는|이|가|을|를|의|에|로|와|과|도|만|인|한|라|다)(?=[\s,.;:)\]']|$)", "", t)
    if HANGUL.search(t):
        UNTRANSLATED.append(f"{where}: {t}")
        # drop parenthetical notes and words that still contain Hangul
        t = re.sub(r"\([^()]*[\uac00-\ud7a3][^()]*\)", "", t)
        t = re.sub(r"'[^']*[\uac00-\ud7a3][^']*'", "", t)
        t = re.sub(r"\S*[\uac00-\ud7a3]+\S*", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip(" ,;.")
        t = t + "\\textsuperscript{\\dag}" if t else "\\dag"
    return t


def tex(s):
    """escape LaTeX specials in copied text (glossary output keeps its own \\dag marker)."""
    keep = []

    def stash(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"\\textsuperscript\{\\dag\}|\\dag", stash, s)
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    # break opportunities inside long codes (file names, contrasts, target keys)
    s = s.replace("\\_", "\\_\\allowbreak{}").replace("|", "|\\allowbreak{}").replace("/", "/\\allowbreak{}")
    s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def cell(text, where=""):
    return tex(tr(text, where))


# ----------------------------------------------------------------------------------------
# LaTeX emitters
# ----------------------------------------------------------------------------------------

def widths(headers, rows, total=1.0, minw=0.05):
    n = len(headers)
    lens = []
    for j in range(n):
        vals = [len(r[j]) if j < len(r) else 0 for r in rows] + [len(headers[j])]
        lens.append(max(4, min(60, sorted(vals)[int(0.9 * (len(vals) - 1))])) ** 0.75)
    tot = sum(lens)
    w = [max(minw, total * x / tot) for x in lens]
    s = sum(w)
    return [x * total / s for x in w]


def plain(s):
    """visible text of a cell (commands and braces removed), for width estimates."""
    s = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", lambda m: m.group(1)[1:-1] if m.group(1) and not m.group(0).startswith("\\textcolor") else "", s)
    return re.sub(r"[{}$\\]", "", s)


def enforce_min(ws, headers, rows, line_pt=482.0, char_pt=4.1, pad_pt=9.0):
    """widen columns whose longest unbreakable token would not fit; renormalize to 1."""
    n = len(headers)
    mins = []
    for j in range(n):
        toks = [t for r in ([headers] + rows) if j < len(r) for t in plain(r[j]).split()]
        longest = max((len(t) for t in toks), default=4)
        mins.append((longest * char_pt + pad_pt) / line_pt)
    w = [max(a, b) for a, b in zip(ws, mins)]
    tot = sum(w)
    return [x / tot for x in w]


def longtable(headers, rows, size="\\fontsize{7.5}{9}\\selectfont", ws=None, title=None):
    n = len(headers)
    ws = enforce_min(ws or widths(headers, rows), headers, rows)
    sep = 4  # pt
    spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{\\dimexpr {w:.3f}\\linewidth-{2*sep}pt\\relax}}" for w in ws)
    out = []
    if title:
        out.append(f"\\par\\medskip\\noindent{{\\small\\textit{{{title}}}}}\\par\\nopagebreak")
    out.append("{" + size + f"\\setlength{{\\tabcolsep}}{{{sep}pt}}\\setlength{{\\LTpre}}{{2pt}}\\setlength{{\\LTpost}}{{4pt}}")
    out.append(f"\\begin{{longtable}}{{@{{}}{spec}@{{}}}}")
    out.append("\\toprule")
    out.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    out.append("\\midrule\\endfirsthead")
    out.append("\\toprule")
    out.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    out.append("\\midrule\\endhead")
    out.append("\\bottomrule\\endlastfoot")
    for r in rows:
        r = list(r) + [""] * (n - len(r))
        out.append(" & ".join(r[:n]) + " \\\\")
    out.append("\\end{longtable}}")
    return "\n".join(out)


def table_head(num, title, legend):
    return (f"\\clearpage\n\\phantomsection\\label{{tab:S{num}}}\n"
            f"\\noindent\\textbf{{Supplementary Table S{num}.}} {title}\\par\\smallskip\n"
            f"\\noindent{{\\footnotesize {legend}}}\\par\n")


def readme_block(readme, sec, idx, cols, headers, title, rowfilter=None, label_fn=None, fmap=None):
    t = md_table(os.path.join(CLAIMS, readme, "README.md"), sec, idx)
    rows = []
    for r in t[1:]:
        if rowfilter and not rowfilter(r):
            continue
        out = []
        for k, j in enumerate(cols):
            v = r[j] if j < len(r) else ""
            if label_fn and k == 0:
                v = label_fn(v)
            if fmap and k in fmap:
                v = fmap[k](v)
            out.append(cell(v, f"{readme} {sec} t{idx}"))
        rows.append(out)
    return longtable(headers, rows, title=f"{title} ({readme.split('_')[0]} README {sec})")


def filt_label(v):
    """C6 row filters -> 'target; contrast (role)'."""
    kv = dict(re.findall(r"(\w+)=`([^`]*)`", v))
    for k2, v2 in re.findall(r"(\w+)=([^,`]+)", re.sub(r"\w+=`[^`]*`", "", v)):
        kv.setdefault(k2, v2.strip())
    kv["role"] = {"주": "main", "서술": "descriptive", "보조": "auxiliary"}.get(kv.get("role", ""), kv.get("role", ""))
    parts = [kv.get("target", ""), kv.get("contrast", kv.get("method", "")) + (f", n {kv['n']}" if "n" in kv and "contrast" not in kv else "")]
    role = kv.get("role", "")
    s = "; ".join(p for p in parts if p)
    return s + (f" ({role})" if role else "")


# ----------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------
CODES = ("Method codes (internal identifiers allowed in the Supplementary Information): P0 source-coefficient Stefan model; "
         "P* year-matched Stefan model; P1 recalibrated Stefan model ($\\kappa = 10$); P2 local least-squares Stefan model; "
         "P1* or P1@ed soil-property recalibrated Stefan model; R0 source anchor plus residual ML; R1 recalibrated anchor plus residual ML; "
         "R2 augmentation plus anchor plus residual; D0 direct ML; D1 physics pseudo-label augmentation; F1a, F1k, F1n physics-input ML; "
         "RM multiplicative residual; Re Stefan-CCI mean anchor plus residual; W selection rule (cross-validation within the target labels); "
         "S1 random cells, S2 block stratification, S3 covariate-cluster stratification, S4 covariate max-min distance, S5 source-extrapolation first, "
         "S6 sequential selection by prediction variance, S7 $\\sqrt{\\mathrm{TDD}}$ first. Target suffixes: $|x$ parent region excluded from the source, "
         "$|i$ rest of the parent kept, $|r$ within-region design. $n$ = number of target labels; all = all labels of half A. "
         "Verdicts: lower error, higher error, equivalent ($\\pm$0.5~cm), undetermined; both 95\\% block-bootstrap CIs (cell-weighted and block-equal) must agree.")


def s1():
    path = os.path.join(CLAIMS, "D_data_and_design", "tables", "Table1_data.csv")
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["Group"] in ("Sub-region",) or r["Role"] == "Point est.":
            rows.append([tex(r["Target"]), tex(r["Group"] if r["Role"] != "Point est." else "Point estimate only"),
                         tex(r["Modes"]), tex(r["Rows"]), tex(r["1 km loc."]), tex(r["Blocks"]), tex(r["Splits"]),
                         tex(r["A cells"]), tex(r["Scored cells (blocks)"]), tex(r["E0, x / i"]), tex(r["Label type"])])
    rows.append(["North Atlantic", "New region, full edition only", "x", "41", "n.a.", "13", "5", "19–24", "12–19 (3–7)", "1.59", "1 km cell means"])  # LGD shard records, full edition (round 10)
    h = ["Target", "Group", "Modes", "Rows", "1 km loc.", "Blocks", "Splits", "A cells", "Scored cells (blocks)", "$E_0$, x / i", "Label type"]
    legend = ("Sub-regions of Alaska (AL-1 to AL-6), Canada (CA-2, CA-3) and the Lena Delta (LE-1, LE-2) were defined by $k$-means clustering of 0.5\\textdegree{} block "
              "centroids without labels (133 blocks, assignment fixed before the main run). Mode x excludes the parent region from the source; mode i keeps the rest of the parent "
              "(simulated new area inside a labelled region). Ranges are over valid splits. $E_0$ in cm~(\\textdegree{}C~d)$^{-1/2}$. The North Atlantic group is reported only in the "
              "full-data edition (licence-unverified rows; D README E19); n.a., not available. Source: paper/claims/D\\_data\\_and\\_design/tables/Table1\\_data.csv (copy of the v2 Table 1 data) and README E19; "
              "North Atlantic splits, A cells, scored cells and $E_0$ from the shard records of the full-edition LGD run (data/processed/lgd/NAtlantic/shards, unit.json).")
    return table_head(1, "Sub-regions, targets with point estimates only and the North Atlantic group.", legend) + \
        longtable(h, rows, ws=[0.09, 0.11, 0.05, 0.06, 0.06, 0.05, 0.05, 0.10, 0.12, 0.08, 0.23])


S2_STATUS = {
    "L1": "supported (regions with both CI upper bounds below 0: 0/4)", "L2": "rejected (augmentation is for zero labels only); R3 partial",
    "L3": "supported", "L4": "net value of ML with sparse labels (minimum $n$ 10, four-region mean)",
    "L5": "learner differences (partial: region count short at $n$ 40, 160)", "L6": "not decidable (no passing combination)",
    "L7": "rejected", "L8": "supported ($-2.64$ [$-3.47$, $-1.90$]; regions 4/4)", "L10": "supported", "L12": "rejected",
    "L15": "mixed (described by $n$)", "L19": "not supported (no reversal wording; degradation size reported)", "L20": "supported",
    "L21": "supported ($\\lambda$ 0.25)", "L22": "lower error in 2 of 3 regions (Lena Delta, Canada)",
    "L29": "a baseline better than P0 exists (P* = year-matched Stefan)", "L30": "supported",
    "L1e": "supported (P4, PE1, PE2)", "L4e": "P4 and PE2: net value with sparse labels; PE1 partial (2/5)",
    "L8e": "supported (P4, PE1, PE2)", "L38": "diverging regions: Central Russia, Tibetan Plateau", "L39": "robust",
    "L40": "weakened", "L41": "robust or weakened by region", "L42": "robust or weakened by region",
    "L43": "placement effect not established", "SC1w": "safety not confirmed (non-inferior cells 4/10)",
    "SC2w": "sentence A (2 of 3 regions at $n$ 40)", "SC3w": "rule not operating (label ratio 0.91)",
    "AK1w": "majority ($r$ 0.57; 68 pairs, 5 targets)", "LGF-F1": "supported (not superior to P0)",
    "LGF-N1": "supported (not superior to P0; four networks)",
}
for k in ["AB1", "AB2", "AB3", "AB4", "AB5", "AB6", "AB7", "AB8", "AB9", "AB10"]:
    S2_STATUS[k] = "see Supplementary Table S3"

WF_ROWS = [
    ("WF0", "WF", "D", "Zero-label risk table and misspecification diagnostics over 30 targets (post hoc description)", "4168331 10-01", "post hoc", "descriptive (Supplementary Table S5)"),
    ("WF1-a", "WF", "P", "Within-region R1 and R2 ($\\lambda$ by CV) have lower error than the best physics calibration at $n \\geq 1000$ (Alaska, five splits)", "4168331 10-01", "designed after viewing", "rejected"),
    ("WF1-b", "WF", "A", "Nine SAR covariates add value to within-region R1", "4168331 10-01", "designed after viewing", "rejected"),
    ("WF1-c", "WF", "A", "Recipes with fewer higher-error $n$ than direct ML", "4168331 10-01", "designed after viewing", "partially supported (R2, D1)"),
    ("WF2-a", "WF", "P", "Placement strategies S2 to S7 have lower error than random cells (S1) within regions", "4168331 10-01", "designed after viewing", "supported (S3 at $n$ 100; size below 0.5 cm)"),
    ("WF2-b", "WF", "A", "The best Alaskan strategy (S3) transfers to the Lena Delta and Canada", "4168331 10-01", "designed after viewing", "rejected"),
    ("WF3-a", "WF", "A", "A covariate-dependent coefficient (Pc) has lower error than P1 at $n \\geq 500$", "4168331 10-01", "designed after viewing", "rejected"),
    ("WF4-a", "WF", "P", "Selection rule W is non-inferior to the fixed recipe R1($\\lambda$ 0.25) (margin 0.5 cm)", "4168331 10-01", "designed after viewing", "partial (regions 2/4): non-inferior at 40, 160 and all labels"),
    ("WF4-b", "WF", "A", "W has lower error than P1 in more targets than R1(0.25)", "4168331 10-01", "designed after viewing", "rejected"),
    ("WF4-c", "WF", "P", "The ten-label bias is rank-correlated with the all-label gain of W over P0", "4168331 10-01", "designed after viewing", "supported ($\\rho$ 0.38 [0.17, 0.55])"),
    ("WF6-a", "WF", "P", "Within-region recipes ($\\lambda$ by CV) have lower error than P1 at $n$ 200 to all (Alaska; Lena Delta; Canada)", "1b42b4d 10-01", "designed after viewing", "Alaska supported; Lena Delta rejected; Canada partial, rejected"),
    ("WF6-b", "WF", "A", "Stratified mean of three targets, R2 versus P1 at $n \\geq 500$", "1b42b4d 10-01", "designed after viewing", "partial (regions 2/3): rejected"),
    ("WF8-a", "WF", "P", "In transfer, S4 has lower error than S1 for R1 at $n$ 40", "1b42b4d 10-01", "designed after viewing", "supported (Canada)"),
    ("WF8-b", "WF", "A", "In transfer, S2 and S4 are not worse than S1 for R1 at $n$ 10", "1b42b4d 10-01", "designed after viewing", "supported"),
    ("WF9-a", "WF", "P", "Within-grid gain of R1 ($\\lambda$ by CV) over P1 within regions", "8180632 10-02", "designed after viewing", "partial (regions 2/3): rejected"),
    ("WF9-c", "WF", "P", "Within-grid gain of R1 ($\\lambda$ 0.25) over P1 in transfer", "8180632 10-02", "designed after viewing", "partial (regions 2/4): partially supported"),
    ("WF9 soil", "WF", "A", "Soil-$\\sqrt{\\mathrm{TDD}}$ grouping sensitivity of WF9", "7da0064 10-02", "designed after viewing", "see Supplementary Table S10"),
    ("WF10-a to c", "WF", "P", "Climate extrapolation (warm variant)", "8180632 10-02", "designed after viewing", "not decidable (contrasts 0/2)"),
]

X_ROWS = [
    ("XA", "FINAL\\_\\allowbreak{}BATCH 2.1", "post hoc", "Rank correlation of the recalibration share, the selection-rule share and the ML share beyond recalibration with coefficient error", "2678100 10-04", "replication (unblinded)", "not established (XA-1, XA-2, XA-3; Supplementary Table S13)"),
    ("XB", "FINAL\\_\\allowbreak{}BATCH 2.2", "see plan", "Label-weighted stacking of physics models and products", "2678100 10-04", "PU", "XB-1, XB-2 rejected (undetermined); XB-3 partially supported (lower error at $n$ 500 and 1,000; not retained when CCI v5 was added as a candidate, sensitivity edition); XB-4 supported (non-inferior)"),
    ("XC", "FINAL\\_\\allowbreak{}BATCH 2.3", "see plan", "End-to-end workflow on new splits of the same regions", "2678100 10-04", "PU (XC-1, XC-2, XC-5); B (XC-F3)", "Algorithm P = S1; XC-1b undetermined; XC-1c lower error (Canada; Lena Delta undetermined); XC-2a, XC-2b, XC-2c non-inferior for the pooled mean (XC-2s lower error at $n$ 40 and 160; regions over 0.5 cm worse: E Russia, Alaska, Lena Delta); XC-5b, XC-5c not tested; XC-F3 not tested (appendix pending: registered data deadline 11 October 2026)"),
    ("XD", "FINAL\\_\\allowbreak{}BATCH 2.4", "exploratory", "Placement algorithm and learned placement policy", "2678100 10-04", "PU (XD-1 to XD-3); B (XD-4)", "Algorithm P = S1; XD-1 rejected; XD-2 rejected (higher error in Alaska and Canada); XD-3 non-inferior 8, higher error 2, not established 6; XD-4, XD-5 descriptive"),
    ("XE", "FINAL\\_\\allowbreak{}BATCH 2.5", "see plan", "Sub-grid public covariates within regions", "2678100 10-04", "PU (xh0)", "XE-c (xh0, auxiliary) equivalent; XE-e diagnostic (no verdict); XE-a, XE-b pending: registered data deadline 8 October 2026 (XE stage 2)"),
    ("XF", "FINAL\\_\\allowbreak{}BATCH 2.6", "see plan", "New independent regions from public data", "2678100 10-04", "B (new regions)", "Pending: registered data deadline 11 October 2026 (XF)."),
    ("XG", "FINAL\\_\\allowbreak{}BATCH 2.7", "see plan", "Existing ALT products on the same blocks", "2678100 10-04 (WRAPUP 10 deviation)", "B (Wei rows); PU (CCI rows)", "XG-1c higher error; XG-2c, XG-3c lower error; XG-1w to XG-3w undetermined"),
    ("XH", "FINAL\\_\\allowbreak{}BATCH 2.8", "descriptive", "Validation ladder", "2678100 10-04", "RU (Alaska); B (Lena Delta, Canada)", "descriptive (no verdict words)"),
    ("XI", "FINAL\\_\\allowbreak{}BATCH 2.9", "see plan", "Climate-extrapolation retest (warm-block spatial proxy)", "2678100 10-04", "B (Canada); RU (Alaska)", "XI-a, XI-b rejected (undetermined); XI-c supported (lower error, non-inferior)"),
    ("XJ", "FINAL\\_\\allowbreak{}BATCH 2.10", "see plan", "Temperature-derived auxiliary labels", "2678100 10-04", "PU (L39)", "XJ-1, XJ-2 not tested (licence basis not recorded); XJ-3 lower error (six rows)"),
    ("XK", "addendum XK", "descriptive", "Error by grid size within regions", "d721d24 10-05", "designed after viewing", "descriptive (no verdict words)"),
    ("XL", "addendum XL", "descriptive", "1 km maps compared with ALT products", "d721d24 10-05", "designed after viewing", "descriptive (no verdict; no accuracy claim)"),
    ("XM", "addendum XM", "exploratory", "Open 10 to 20 m land-cover fractions and summer vegetation indices of the 1 km label cell added to residual ML within regions (within-grid RMSE)", "ff7c97f 10-05", "PU; designed after viewing", "XM-a partially supported (lower error at $n$ 1,000 and all, size below 0.5 cm; equivalent at 500)"),
]


def s2():
    t = md_table(os.path.join(ROOT, "docs", "MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md"), "M4.3", 0)
    rows = []
    for r in t[1:]:
        hid = r[0].strip()
        status = S2_STATUS.get(hid, "see the Supplementary Table of its family")
        rows.append([tex(hid), tex(r[1]), tex(r[2]), tex(r[3]).replace("\\textbackslash{}", "\\"), tex(r[4]), tex(r[5]), status])
    for r in WF_ROWS + X_ROWS:
        rows.append([tex(r[0]) if "\\" not in r[0] else r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
    h = ["ID", "Plan", "Role", "Statement (abridged)", "Registered", "Blinding", "Registered verdict"]
    legend = ("Role: C confirmatory; P primary; A auxiliary; D descriptive or diagnostic; N direction-neutral report; abstract, contrast of the abstract bundle. "
              "Blinding: B blind; PU partially unblinded; RU replication (unblinded); RE result existed on disk at registration, not viewed; Sa S-a viewing label; NS not listed. "
              "Registered: first commit containing the hypothesis text (hash, date, Korea Standard Time). The verdict column gives the first clause of the registered verdict "
              "string of each plan (scope = verdict rows of the test tables cited in the claims READMEs); contrasts, $\\Delta$, both 95\\% CIs, uncorrected and Holm-adjusted "
              "$P$ values are in Supplementary Tables S3 to S13. Rows LG to WRAPUP: frame of docs/MANUSCRIPT\\_DRAFT\\_SUPPORT\\_2026-09-30.md M4.3 (88 rows); WF and X rows added at assembly.")
    return table_head(2, "Register of registered hypotheses.", legend) + \
        longtable(h, rows, ws=[0.07, 0.08, 0.06, 0.36, 0.13, 0.12, 0.18])


def s3():
    path = os.path.join(CLAIMS, "C1_label0_safety", "tables", "lgw_bundle.csv")
    rows = []

    def f(x, d=2):
        try:
            return f"{float(x):+.{d}f}".replace("-", "\u2212")
        except ValueError:
            return "n.a."

    def p(x):
        try:
            v = float(x)
            return f"{v:.4f}" if v < 0.001 else f"{v:.3f}"
        except ValueError:
            return "n.a."

    vmap = {"우세": "lower error", "열세": "higher error", "동등": "equivalent", "미결정": "undetermined"}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("scope") != "MEAN":
            continue
        con = r["contrast"].replace("구간 점수(α 0.1) 단 (iii) − B4 | n10", "interval score ($\\alpha$ 0.1), rung (iii) $-$ B4, n10")
        con = tex(con) if "$" not in con else con
        cell_ci = f"{f(r['delta'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}]" if r["ci_lo"] else f"{f(r['delta'])} [n.a.]"
        beq_ci = f"{f(r['delta_blockeq'])} [{f(r['ci_lo_beq'])}, {f(r['ci_hi_beq'])}]" if r["delta_blockeq"] else "n.a."
        rows.append([r["ab"], con, cell_ci, beq_ci, p(r["p_cell"]), p(r["p_beq"]), p(r["holm_p"]),
                     vmap.get(r["verdict4"], r["verdict4"]), vmap.get(r["verdict4_common"], r["verdict4_common"] or "n.a."),
                     "yes" if r.get("ci_dependence") else ""])
    h = ["AB", "Contrast", "$\\Delta$ cell-weighted [95\\% CI] (cm)", "$\\Delta$ block-equal [95\\% CI] (cm)", "$P$ cell", "$P$ block",
         "Holm $P$", "Verdict", "Common-CI verdict", "Split-dependent"]
    legend = ("Four main regions, stratified means; AB1 to AB9 with 10,000 block resamples in both weightings, AB10 (interval score of the hierarchical predictive interval "
              "against the calibrated constant-width interval B4; Lena Delta, Canada, Alaska) with 1,000 resamples from LGU-A1, for which the copied bundle row has no CI. "
              "$P$ values are uncorrected one-sided bootstrap values; Holm adjustment over the ten contrasts. Common-CI verdict: the verdict under the common resampling of "
              "both arms; 'yes' marks verdicts that depend on the split-independence assumption. The abstract uses verdict words only for these contrasts. "
              "Source: paper/claims/C1\\_label0\\_safety/tables/lgw\\_bundle.csv (copy of data/processed/lgw/lgw\\_bundle.csv), rows scope = MEAN. " + CODES)
    return table_head(3, "Abstract contrast bundle AB1 to AB10.", legend) + \
        longtable(h, rows, ws=[0.05, 0.17, 0.15, 0.15, 0.07, 0.07, 0.07, 0.10, 0.10, 0.07])


def s4():
    parts = [table_head(4, "Transfer without target labels: learners, physics information and physics baselines.",
                        "Direct ML minus the source-coefficient Stefan model for ten learners (four main regions), regional rows, the three-region auxiliary column, "
                        "new regions, placebo pseudo-labels, low-weight residuals and the physics baselines. Holm families differ by row (abstract bundle, LGX, LGT, LGF). "
                        "Learners 1 to 4 ran on the cloud platform, the other learners and companion rows on the local GPU server; differences between platforms are not computed. " + CODES)]
    parts.append(readme_block("C1_label0_safety", "2.1", 0, [0, 1, 4, 5, 6, 7, 8, 9],
                              ["\\#", "Learner", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Holm $P$", "Common CI (cell / block), verdict", "Dependence"],
                              "a. Direct ML minus source-coefficient Stefan, zero labels, four main regions"))
    parts.append(readme_block("C1_label0_safety", "2.2", 0, [0, 1, 2, 3, 4, 5],
                              ["Region", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Common CI (cell / block), verdict", "Dependence"],
                              "b. AB1 regional rows"))
    parts.append(readme_block("C1_label0_safety", "2.2", 1, [0, 1, 2, 3, 4, 6, 7],
                              ["Region", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "sig", "sig (common)", "Flags", "Cross-environment"],
                              "c. New independent regions, licence-verified edition (L1e, PE2, zero labels)"))
    parts.append(readme_block("C1_label0_safety", "2.2", 2, [0, 2, 3, 4, 5, 6, 7],
                              ["Learner", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Common CI, verdict", "Dependence", "Four-region $\\Delta$"],
                              "d. Three-region auxiliary column (Lena Delta, Canada, Alaska)"))
    parts.append(readme_block("C1_label0_safety", "2.2", 3, [0, 1, 2, 3, 4, 5],
                              ["Filter", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Common CI, verdict", "Dependence"],
                              "e. Limit row: TabICL v2 with ten labels"))
    parts.append(readme_block("C1_label0_safety", "2.3", 0, [0, 2, 3, 4, 5],
                              ["Contrast", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Holm $P$"],
                              "f. Source of the physics information: placebo pseudo-labels (AB3, L15)"))
    parts.append(readme_block("C1_label0_safety", "2.4", 0, [0, 2, 3, 4, 5, 6, 7],
                              ["Contrast", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Common CI, verdict", "Dependence", "Note"],
                              "g. Low-weight residuals and combination structure at zero labels"))
    parts.append(readme_block("C1_label0_safety", "2.6", 0, list(range(7)),
                              ["Target", "Method", "RMSE", "RMSE P0", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "sig"],
                              "h. Target curves at zero labels (LG main run; 1,000 resamples)"))
    parts.append(readme_block("C1_label0_safety", "2.7", 0, [0, 2, 3, 4, 5, 6, 7],
                              ["Contrast", "$\\Delta$ cell-weighted [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Holm $P$", "Common CI, verdict", "Dependence"],
                              "i. Physics baselines at zero labels"))
    return "\n".join(parts)


def s5():
    head = table_head(5, "Zero-label risk table (WF0, post hoc description, no verdict words).",
                      "Number of the 30 targets (25 LG targets and five licence-verified LGD targets; seven independent regions and 20 sub-regions) in which the method minus P0 "
                      "exceeded +2~cm, with median and maximum (cm). Cell-random draws, $\\alpha$ = 1, CatBoost. D0\\_1.0 direct ML; R1\\_1.0 residual weight 1.0; R1\\_0.25 low-weight "
                      "residual (equal to R0 at zero labels); D1\\_1.0 physics augmentation; R2\\_1.0 augmentation plus anchor plus residual with $\\lambda$ 1.0. The count uses the point estimate only. "
                      "Counting only the seven independent regions, targets above +2~cm at zero labels were D0 5/7, R1\\_1.0 2/7, D1 1/7, R1\\_0.25 0/7 and R2\\_1.0 0/7. Counting targets whose "
                      "two CIs both exceeded zero (higher error) instead of the 2~cm threshold gives D0\\_1.0 14, R1\\_1.0 10, D1\\_1.0 8, R1\\_0.25 6 and R2\\_1.0 5 of 30 "
                      "(post hoc calculation, C1 README 2.8). Source: paper/claims/C1\\_label0\\_safety/tables/wf0\\_risk.csv.")
    return head + readme_block("C1_label0_safety", "2.5", 0, list(range(6)),
                               ["$n$", "Method", "Targets", "Targets above +2 cm", "Median (cm)", "Maximum (cm)"], "Risk table",
                               label_fn=lambda v: v.replace("−1(전량)", "all").replace("−1", "all"))


def generic8(readme, sec, idx, title, value_col=5, verdict_col=6, label_col=1):
    return readme_block(readme, sec, idx, [0, label_col, value_col, verdict_col],
                        ["ID", "Contrast or quantity", "Value (cm unless stated): cell-weighted [CI] / block-equal [CI]", "Verdict"], title)


def s6():
    parts = [table_head(6, "Combination structures and augmentation (L10, L2, L12, L17, L4, recalibration method).",
                        "Values: cell-weighted $\\Delta$ [95\\% CI] / block-equal [95\\% CI] in cm. Pools at 40 and 160 labels contain the Lena Delta and Canada only. " + CODES)]
    for sec, ttl in [("2.1", "a. Residual structure versus physics-input structure, 0 to 10 labels"), ("2.2", "b. 40 to 160 labels: reversal and regional rows"),
                     ("2.3", "c. Augmented combination and stacking"), ("2.4", "d. Multiplicative residual and label shuffle"),
                     ("2.5", "e. Net value of the residual over the recalibrated model"), ("2.6", "f. Shrinkage recalibration versus local least squares")]:
        parts.append(generic8("C3_structure_by_label_count", sec, 0, ttl))
    return "\n".join(parts)


def s7():
    parts = [table_head(7, "Within-region label-count tests (WF1, WF6, SAR).",
                         "Designed after the transfer results had been viewed and registered before running. WF6-a: 25 splits (24 valid in the Lena Delta), comparator P1. "
                         "WF1: first test, five splits, comparator the best cross-validated physics calibration (Pbest); WF1-b adds nine SAR covariates (x34) in Alaska. " + CODES)]
    parts.append(readme_block("C4_sufficient_labels", "2.1", 0, list(range(7)),
                              ["ID", "Contrast", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Common-CI verdict", "RMSE A / B"], "a. WF6-a, Alaska"))
    parts.append(readme_block("C4_sufficient_labels", "2.2", 0, list(range(7)),
                              ["ID", "Target", "Contrast", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "Splits"], "b. WF6-a, Lena Delta and Canada"))
    parts.append(readme_block("C4_sufficient_labels", "2.3", 0, list(range(5)),
                              ["ID", "$n$", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "Verdict"], "c. WF6-b, stratified mean of three targets"))
    parts.append(readme_block("C4_sufficient_labels", "2.4", 0, list(range(7)),
                              ["ID", "Target", "Contrast", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "Verdict", "RMSE A / B"], "d. WF1-a, first test"))
    parts.append(readme_block("C4_sufficient_labels", "2.5", 0, list(range(6)),
                              ["ID", "Contrast", "Role", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "Verdict"], "e. WF1-b (SAR) and WF1-c (risk counts)"))
    parts.append(readme_block("C4_sufficient_labels", "2.6", 0, [0, 3, 4],
                              ["ID", "Column", "Value"], "f. Error floors (covariate-conditional, scoring cells) and contest values (cm)"))
    parts.append(readme_block("C4_sufficient_labels", "2.6", 1, [0, 1, 2],
                              ["ID", "Calculation", "Value"], "g. Derived values: relative RMSE change and reducible squared error (F = error floor)"))
    return "\n".join(parts)


def s8():
    parts = [table_head(8, "Selection rule W and bias diagnosis (WF4-a, WF4-b, WF4-c, selection counts).",
                         "W chooses among seven candidates by block cross-validation within the target labels (Methods); fixed recipe R1 with $\\lambda$ 0.25. Non-inferiority margin 0.5~cm. "
                         "Designed after the label-grid results had been viewed and registered before running. " + CODES)]
    parts.append(readme_block("C5_method_selection", "2.1", 0, list(range(9)),
                              ["ID", "$n$ (pool)", "$\\Delta$", "CI cell / block-equal", "Verdict", "Common CI, verdict", "Holm $P$", "$P_{\\mathrm{ni}}$, non-inferior", "$\\Delta$ (\\% of P0)"],
                              "a. WF4-a pooled contrast W $-$ R1(0.25)"))
    parts.append(readme_block("C5_method_selection", "2.2", 0, list(range(7)),
                              ["ID", "Target", "$n$", "$\\Delta$", "CI cell / block-equal", "Verdict", "Common-CI verdict"], "b. WF4-a regional rows"))
    parts.append(readme_block("C5_method_selection", "2.3", 1, list(range(6)),
                              ["ID", "$n$", "W lower error", "R1(0.25) lower error", "W higher error", "R1(0.25) higher error"], "c. WF4-b, targets with lower error than P1"))
    parts.append(readme_block("C5_method_selection", "2.4", 0, list(range(7)),
                              ["ID", "Target", "$n$", "W $-$ P0 cell / block-equal", "Verdict", "W $-$ P1 cell / block-equal", "Verdict"], "d. W against the physics baselines (curve table, auxiliary)"))
    parts.append(readme_block("C5_method_selection", "2.5", 0, list(range(6)),
                              ["ID", "Target", "$n$ 10", "$n$ 40", "$n$ 160", "All"], "e. Methods chosen by W (counts over splits and draws; this check)"))
    parts.append(readme_block("C5_method_selection", "2.5", 1, list(range(5)),
                              ["ID", "Target", "$n$", "W $-$ R1(0.25)", "W $-$ R1(1.0)"], "f. Point estimates against fixed recipes (this check)"))
    parts.append(generic8("C2_bias_diagnosis", "2.1", 0, "g. Bias diagnosis records (WF4-c, WF0)"))
    parts.append(readme_block("C5_method_selection", "2.6", 0, list(range(11)),
                              ["ID", "Learner", "R1 $n$ 0 $\\lambda$ 0.25", "R1 $n$ 10 $\\lambda$ 0.25", "R1 all $\\lambda$ 0.25", "D0 $n$ 0", "D0 $n$ 10", "D0 all",
                               "R1 $n$ 0 $\\lambda$ 1.0", "R1 $n$ 10 $\\lambda$ 1.0", "R1 all $\\lambda$ 1.0"],
                              "h. Source cross-validation tuning of neural networks, tuned minus default (LGF-N2; cm, four main regions)"))
    parts.append(readme_block("C5_method_selection", "2.6", 1, [0, 1, 2, 3],
                              ["ID", "Learner", "Verdict wording (summary of the file values)", "Default kept (k\\_default)"],
                              "i. LGF-N2 verdicts by learner"))
    return "\n".join(parts)


def s9():
    parts = [table_head(9, "Label placement strategies (WF2, WF8, L43).",
                         "Strategy minus random cells (S1) at the same budget; within-region designs (target suffix $|r$) and transfer ($|x$). Values in cm; RMSE A / B are the RMSE of "
                         "the strategy and of random cells. WF2-a rows marked descriptive are reported without verdict words in the main text. " + CODES)]
    cols = [0, 2, 3, 4, 5, 6]
    hdr = ["ID", "Target; contrast (role)", "$\\Delta$ cell [CI]", "$\\Delta$ block-equal [CI]", "RMSE A / B", "Verdict"]
    for sec, ttl in [("2.1", "a. Block stratification (S2) and covariate max-min distance (S4), WF2-a"), ("2.2", "b. Sequential active selection (S6), WF2-a"),
                     ("2.3", "c. Source-extrapolation first (S5) and $\\sqrt{\\mathrm{TDD}}$ first (S7), WF2-a"), ("2.4", "d. Transfer of the best Alaskan strategy, WF2-b"),
                     ("2.5", "e. Transfer conditions, WF8"), ("2.6", "f. Block-dispersed versus cell-random placement, L43")]:
        parts.append(readme_block("C6_label_placement", sec, 0, cols, hdr, ttl, fmap={1: filt_label}))
    parts.append(readme_block("C6_label_placement", "2.9", 0, list(range(8)),
                              ["Strategy", "WF2 lower", "WF2 equivalent", "WF2 undetermined", "WF2 higher", "WF8 lower", "WF8 undetermined", "WF8 higher"],
                              "g. Verdict counts by strategy (post hoc count)"))
    return "\n".join(parts)


def s10():
    parts = [table_head(10, "Gain decomposition and climate extrapolation (WF9, WF10).",
                         "Scoring cells grouped by 0.5\\textdegree{} block and air $\\sqrt{\\mathrm{TDD}}$ (registered: soil $\\sqrt{\\mathrm{TDD}}$, sensitivity rows b and e). Shares are computed "
                         "from the squared-error sums (Methods, equation for SSE). Explained within-grid share = 1 $-$ (within-grid RMSE of the method / within-grid RMSE of P1)$^2$. " + CODES)]
    for sec, ttl in [("2.1", "a. Decomposition of the within-region gain in Alaska (all labels, $\\lambda$ by CV)"), ("2.2", "b. Shares of the squared error (descriptive, no CI)"),
                     ("2.3", "c. Explained within-grid share (descriptive, no CI)"), ("2.4", "d. WF9-a within regions, air $\\sqrt{\\mathrm{TDD}}$ grouping"),
                     ("2.5", "e. WF9-a, soil $\\sqrt{\\mathrm{TDD}}$ grouping (sensitivity)"), ("2.6", "f. WF9-c in transfer, air grouping"),
                     ("2.7", "g. WF9-c, soil grouping (sensitivity)"), ("2.8", "h. Descriptive contrasts outside the registered hypotheses"),
                     ("2.9", "i. WF10 climate extrapolation (warm variant)")]:
        parts.append(generic8("C8_gain_source", sec, 0, ttl))
    return "\n".join(parts)


def s11():
    parts = [table_head(11, "Additional independent regions (L1e, L4e, L8e, L38, L39).",
                         "Licence-verified edition is the main edition; the North Atlantic rows are from the full-data edition and carry a few-blocks flag. All Tibetan scored cells lie above "
                         "the source elevation range, so Tibetan contrasts are predicted outcomes and are not used as evidence for independent regions. New-region shards ran on the local "
                         "server and P4 shards on the cloud (cross-environment flag). " + CODES)]
    parts.append(readme_block("SI_learners_new_regions", "2.5", 0, [0, 1, 2, 3], ["ID", "Filter", "Value", "Verdict"], "a. Pool verdicts (L1e, L4e, L8e)"))
    parts.append(readme_block("SI_learners_new_regions", "2.5", 1, [0, 1, 2, 3, 4], ["ID", "Item, target, $n$", "Value (cm)", "Verdict", "Note"], "b. Region-level contrasts (L38)"))
    parts.append(readme_block("SI_learners_new_regions", "2.5", 2, [0, 1, 2, 3], ["ID", "Filter", "Value", "Verdict"], "c. Label definition in the Tibetan Plateau (L39)"))
    parts.append(readme_block("SI_learners_new_regions", "2.5", 3, [0, 1, 2, 3, 4], ["ID", "Item, $n$", "Value (cm)", "Verdict", "Note"], "d. North Atlantic, full-data edition (L38)"))
    return "\n".join(parts)


def s12():
    rows = [
        ["Within-region CQR (S11, contest; Alaska, 6-fold block out-of-fold)", "0.93 [0.89, 0.97]", "53.64", "descriptive", "U1, SI\\_uncertainty README 2.2"],
        ["Transfer CQR trained in Alaska (H17; four main regions, two conditions)", "0.32--0.73", "", "H17 confirmed (upper CI $<$ 0.90 in 4/4)", "U2, README 2.3"],
        ["Pooled generative quantile intervals, nflow (LGU-B1, $\\lambda$ 1.0)", "0.76 [0.67, 0.83]", "", "supported (replication, unblinded)", "U4, README 2.5"],
        ["Pooled generative quantile intervals, cfm (LGU-B1, $\\lambda$ 1.0)", "0.65 [0.56, 0.76]", "", "supported (replication, unblinded)", "U4, README 2.5"],
        ["Hierarchical conformal, hier2\\_cdf (h37, four main regions)", "0.86 [0.80, 0.92]", "80.54", "partially supported (F10)", "U3, README 2.4"],
        ["Hierarchical conformal, constant normalizer (LGU-B, zero labels)", "0.86 [0.83, 0.90]", "81.29", "", "U4, README 2.5"],
    ]
    head = table_head(12, "Coverage of 90\\% prediction intervals by step: within-region CQR, transfer CQR, pooled generative intervals and hierarchical conformal intervals.",
                      "Nominal coverage 0.90. Coverage [95\\% CI] and mean width (cm); cell-weighted values. The hierarchical conformal coverage of 0.86 uses a different protocol from "
                      "the zero-label interval comparison of Fig.~4d and is therefore reported here. Values are copied from paper/claims/SI\\_uncertainty/README.md section 2 (no new calculation; "
                      "section 4 recommendation). Detailed rows follow.")
    parts = [head, longtable(["Step", "Coverage [CI]", "Width (cm)", "Verdict", "Source"], rows, ws=[0.40, 0.14, 0.09, 0.22, 0.15], title="a. Steps")]
    parts.append(readme_block("SI_uncertainty", "2.2", 0, [0, 3, 4, 6], ["Item", "Columns", "Value", "Verdict"], "b. U1 within-region intervals in Alaska"))
    parts.append(readme_block("SI_uncertainty", "2.3", 0, list(range(7)), ["Method", "Condition", "Lena Delta", "Canada", "W Russia", "E Russia", "Verdict"], "c. U2 transfer CQR (coverage)"))
    parts.append(readme_block("SI_uncertainty", "2.4", 0, [0, 2, 3, 5], ["Item", "Columns", "Value", "Verdict"], "d. U3 zero-label hierarchical conformal intervals"))
    parts.append(readme_block("SI_uncertainty", "2.5", 0, [0, 2, 3, 4], ["Hypothesis", "Cell-weighted", "Block-equal", "Verdict (blinding)"], "e. U4 normalizers and generative intervals (LGU-B)"))
    parts.append(readme_block("SI_uncertainty", "2.5", 1, list(range(6)), ["Normalizer", "cov10", "Two-stage CI", "cov10 block-equal", "Width (cm)", "Interval score (cm)"], "f. U4 normalizers at zero labels"))
    parts.append(readme_block("SI_uncertainty", "2.6", 0, [0, 2, 3, 4], ["Hypothesis", "Columns", "Value", "Verdict"], "g. U5 hierarchical predictive distributions with labels and AB10 (LGU-A)"))
    parts.append(readme_block("SI_uncertainty", "2.7", 0, list(range(7)), ["Target", "$n$", "Coverage", "Width (cm)", "Width R0, $n$ 0", "In band", "Narrower"], "h. U6 R1 intervals with labels (LGX L25)"))
    return "\n".join(parts)


def s13():
    # XA rows: XA_result_2026-10-04.md section 4 (sealed/xa_hyp.csv), values copied as printed there
    t = md_table(os.path.join(CLAIMS, "C2_bias_diagnosis", "XA_result_2026-10-04.md"), "4.", 0)
    gain = {"G_recal": "recalibration share P0 $-$ P1", "G_W": "selection-rule share P1 $-$ W", "G_ML2": "ML share min(P1, P2) $-$ R1(0.25)",
            "G_ML": "ML share P1 $-$ R1(0.25)", "G_shr": "shrinkage remainder P1 $-$ P2", "G_tot": "total P0 $-$ W"}
    cat = {"확인하지 못함": "not established", "양(벗어남)": "positive", "음(벗어남)": "negative"}
    rows = []
    for r in t[1:]:
        hid = r[0].replace("(보조)", " (aux.)").replace("서술", "descriptive")
        cats = r[8].replace("**", "")
        for ko, en in sorted(cat.items(), key=lambda kv: -len(kv[0])):
            cats = cats.replace(ko, en)
        cats = cats.replace("(두 범주, 약한 쪽)", " (two categories, weaker reported)").replace("×3", "$\\times$3")
        p_one = r[6].replace("(음 방향 P(ρ^b ≥ 0) 최대 0.31)", " (negative direction: P(ρ^b ≥ 0) at most 0.31)")   # round 11 translation
        holm = r[7].replace("가족 밖", "outside the family")
        rows.append([tex(hid), gain.get(r[1].strip(), tex(r[1])), tex(r[2]), tex(r[3]), tex(r[4]), tex(r[5]), tex(p_one), tex(holm), cats])
    h = ["Hypothesis", "Gain", "$\\rho$ cell / block", "Main CI (family clusters) cell / block", "Auxiliary CI (targets fixed) cell / block",
         "$|A| \\geq 100$ CI cell / block", "$P_{\\mathrm{one}}$ cell, block", "Holm $P$", "Category main / auxiliary / final"]
    xa = longtable(h, rows, ws=[0.08, 0.13, 0.08, 0.14, 0.14, 0.13, 0.08, 0.06, 0.16],
                   title="a. XA gain decomposition (post hoc; all labels; coefficient error = absolute ten-label bias; 28 targets, 8 families; 10,000 resamples)")
    pool = [["All", "4/4", "2.23 [1.33, 3.05] / 2.13 [1.19, 3.01]", "\u22120.02 [\u22120.87, 0.57] / \u22120.43 [\u22121.23, 0.33]", "1.14 [0.22, 2.22] / 1.43 [0.52, 2.32]"],
            ["10", "4/4", "2.45 [1.86, 3.04] / 2.62 [1.88, 3.29]", "\u22120.84 [\u22121.91, \u22120.03] / \u22121.36 [\u22122.33, \u22120.42]", "0.43 [\u22120.49, 1.36] / 0.51 [\u22120.45, 1.45]"],
            ["40", "2/4 (Lena Delta, Canada)", "\u22121.99 [\u22122.71, \u22120.89] / \u22122.01 [\u22122.63, \u22121.37]", "\u22120.76 [\u22120.90, \u22120.57] / \u22120.84 [\u22120.98, \u22120.70]", "1.43 [0.73, 1.90] / 1.55 [1.03, 2.12]"],
            ["160", "2/4", "\u22122.14 [\u22122.93, \u22120.95] / \u22122.12 [\u22122.79, \u22121.44]", "\u22120.19 [\u22120.23, \u22120.14] / \u22120.21 [\u22120.24, \u22120.17]", "1.96 [1.18, 2.51] / 2.06 [1.47, 2.69]"]]
    xa6 = longtable(["$n$", "Pool", "Recalibration share (cell / block)", "Shrinkage remainder", "Selection-rule share"], pool,
                    ws=[0.06, 0.16, 0.26, 0.26, 0.26], title="b. XA-6 decomposition, four main regions, stratified mean (cm, positive = improvement; post hoc description)")
    xrest = XR.build(longtable)
    legend = ("XA was registered in commit 2678100 (4 October 2026, 14:22:39) and run locally without any new fit (sealed tables first opened 21:07:28). Both reproduction gates passed. "
              "Categories: positive or negative only if the main CI (family-cluster and block joint resampling) excludes zero in both weightings and the $|A| \\geq 100$ subset "
              "(23 targets) does too; otherwise not established. Where the main and auxiliary categories differ, both are given and the weaker is reported. The registered "
              "family list had seven families; the North Atlantic target (point estimate only) was added as an eighth single-target family (implementation deviation, sealed/xa\\_hyp.csv). "
              "All five hypotheses were robust to the scale variants and to the oracle definition of coefficient error (absolute log ratio of the all-label target coefficient to "
              "the source coefficient). Holm family: XA-1 to XA-3. Source: paper/claims/C2\\_bias\\_diagnosis/XA\\_result\\_2026-10-04.md sections 4 and 6 "
              "(data/processed/xbatch/XA\\_c2\\_gain\\_decomposition/sealed/). "
              "Parts c to q: XB to XJ were registered in the same commit (2678100) and run on 5 October 2026; sealed tables were first opened between 03:18 and 05:21 KST (XC 05:20:39) after their reproduction gates passed "
              "(docs/EXPERIMENT\\_PLAN\\_FINAL\\_BATCH\\_2026-10-04.md section 8). XK and XL were added on 5 October 2026 (commit d721d24) before computation "
              "(docs/EXPERIMENT\\_PLAN\\_FINAL\\_BATCH\\_ADDENDUM\\_XK\\_XL\\_2026-10-05.md, results), and XM on 5 October 2026 (commit ff7c97f) before feature extraction "
              "(docs/EXPERIMENT\\_PLAN\\_FINAL\\_BATCH\\_ADDENDUM\\_XM\\_2026-10-05.md, results; sealed tables first opened 14:20:47 KST). All were designed after earlier results were viewed; further flags are given with each part. "
              "$\\Delta$ is the RMSE of the first method minus that of the second (cm; negative, the first has lower error), cell-weighted and block-equal with block-bootstrap 95\\% CIs (10,000 resamples unless stated). "
              "Registered interpretation sentences are rendered in English in the notes. " + CODES)
    return table_head(13, "Additional registered analyses XA to XM.", legend) + "\n".join([xa, xa6, xrest])


def s14():
    rows = [
        ["SC1w", "main", "Few-label recipe R1 non-inferior to P0 (0.5 cm) at 3 and 10 labels in five regions",
         "safety not confirmed (non-inferior cells 4/10); no higher-error cell", "non-inferior: Lena Delta $n$ 3, Canada $n$ 3, W Russia $n$ 3 and 10; not confirmed: Lena Delta $n$ 10, Canada $n$ 10, E Russia $n$ 3 and 10, Alaska $n$ 3 and 10"],
        ["SC1w-d10", "auxiliary ($\\delta$ 1.0 cm)", "Same with a 1.0 cm margin", "safety not confirmed (non-inferior cells 5/10)", "not confirmed: Canada $n$ 10, E Russia $n$ 3 and 10, Alaska $n$ 3 and 10"],
        ["SC1w-P", "main", "Recalibration alone (P1) non-inferior to P0", "rejected", "Canada $n$ 3: $\\Delta$ +0.97 [+0.26, +1.30], Holm $P$ 0.021; Canada $n$ 10: +2.26 [+0.71, +2.93], Holm $P$ 0.008"],
        ["SC2w", "main", "Residual learning versus recalibrated Stefan at 40 labels (Lena Delta, Canada, Alaska)", "sentence A: lower error in 2 of 3 regions", "Lena Delta lower; Canada lower; Alaska undetermined"],
        ["SC2w-n160", "auxiliary", "Same at 160 labels", "sentence A: lower error in 2 of 3 regions", "Lena Delta lower; Canada lower; Alaska undetermined"],
        ["SC3w", "main", "Sequential stopping rule ($\\tau$ 0.05) halves labels against always 40 with error increase $\\leq$ 0.3 cm and no higher-error region",
         "rule not operating (label ratio 0.91); $\\Delta$ described only", "label ratio 0.909; three-region mean $\\Delta$ $-$0.048 [$-$0.079, $-$0.002], block-equal $-$0.077 [$-$0.105, $-$0.049]; condition (i) not met, (ii) to (iv) met"],
        ["SC3w, $\\tau$ 0.025", "descriptive", "$\\tau$ sensitivity", "rule not operating (label ratio 0.99)", "label ratio 0.988; $\\Delta$ $-$0.001 [$-$0.006, +0.004], block-equal $-$0.009 [$-$0.014, $-$0.003]"],
        ["SC3w, $\\tau$ 0.1", "descriptive", "$\\tau$ sensitivity", "rejected (condition (i) not met)", "label ratio 0.542; $\\Delta$ $-$0.177 [$-$0.291, $-$0.055], block-equal $-$0.310 [$-$0.394, $-$0.225]"],
        ["SC3w, $\\tau$ 0.2", "descriptive", "$\\tau$ sensitivity", "all four conditions met (not the main verdict)", "label ratio 0.185; $\\Delta$ $-$0.677 [$-$0.909, $-$0.353], block-equal $-$0.845 [$-$1.041, $-$0.640]"],
        ["S-a", "sensitivity (interpretation rule fixed before results)", "Label unit: point labels versus 1 km location means, P1 $-$ P0",
         "with point labels the few-label recalibration gain was smaller (Canada, $n$ 10, $-$0.86 cm); deployment label unit is the 1 km location mean",
         "median $\\Delta_{\\mathrm{unit}}$: Lena Delta $n$ 3 +0.31, $n$ 10 +0.34; Canada $n$ 3 $-$0.47, $n$ 10 $-$0.86; Alaska $n$ 3 +0.07, $n$ 10 +0.09"],
        ["S-b", "sensitivity", "Label year: single-year versus multi-year mean labels (W and E Russia)", "recalibration gain with 3 to 10 labels maintained",
         "median $\\Delta_{\\mathrm{year}}$: W Russia $n$ 3 +0.02, $n$ 10 +0.05; E Russia $n$ 3 +0.01, $n$ 10 +0.04"],
        ["L43", "main", "Block-dispersed versus cell-random placement for P1", "placement effect not established", "P1 $n$ 10 undetermined (regions 4/4, Holm $P$ 1); $n$ 40 undetermined (regions 2/4, Holm $P$ 1)"],
        ["AK1w", "main", "Combinations that beat the physics model in Alaskan sub-regions with an Alaskan source also do so with an outside-only source",
         "majority ($r$ 0.57, 68 pairs, 5 targets)", "$r$ 0.574; $r'$ 0.779; target-level majority 2/5; sensitivity $r$ 0.619"],
    ]
    legend = ("Registered in WRAPUP (316714c, 30 September 2026, 03:02:42), partly unblinded (earlier shrinkage curves were known). The verdict column translates the registered "
              "verdict string; values are copied from the stat column. Source: paper/claims/C1\\_label0\\_safety/tables/lgw\\_tests.csv (copy of data/processed/lgw/lgw\\_tests.csv) "
              "and docs/MANUSCRIPT\\_DRAFT\\_RESULTS\\_ABSTRACT\\_2026-09-30.md (Deployment scenarios). " + CODES)
    return table_head(14, "Deployment scenarios SC1w to SC3w and related wrap-up tests.", legend) + \
        longtable(["ID", "Role", "Question", "Registered verdict", "Values"], rows, ws=[0.08, 0.11, 0.25, 0.24, 0.32])


RECLASS_JSON = os.path.join(HERE, "build", "reclass_summary.json")   # tools/build_reclass_summary.py (round 10)
RECLASS_FIX = {  # cells of the M4.4 frame filled from opened records other than lgw_rescore.csv (round 10)
    ("M1 H-series", 7): "AUDIT stats:F4 (`f098031` 2026-09-26 21:42:16): the nested recipe rests on the pre-registered H13 (Holm $P$ 0.004 and 0.008); its "
                        "'all target regions' wording does not hold in Canada (−0.01) and E Russia (+0.33); M1 Holm family $m$ = 2 (H13 rows)",
    ("H30", 6): "SC1w to SC3w (deployment scenarios)",
    ("H30", 7): "Holm column added to the H30 family; aoa\\_resid stays significant (0.001 to 0.007) (AUDIT stats:F4, `f098031` 2026-09-26 21:42:16)",
}


def _reclass_cells(pid, row, rs):
    """fill the [VERIFY]/[RESULT] cells of one frame row from the reclassification summary (counts of stored rows)."""
    e = rs.get(pid)
    out = list(row)
    esc_ = lambda t: t.replace("_", "\\_")
    if e is None:
        return out
    ci, vc, desc = e["ci_type"], e["verdict_counts"], esc_(e["descriptive_tables"])
    if "[VERIFY]" in out[2]:
        out[2] = ci or "none"
    if "[RESULT]" in out[4]:
        parts = []
        if "two weightings" in ci:
            parts.append("reclassified (unblinded), four-way rule")
        if "cell-weighted only" in ci:
            parts.append("cell-weighted rows: four-way verdict not possible (CI protocol differs)")
        if "row resampling" in ci:
            parts.append("row-resampling rows: protocol differs, descriptive")
        if desc:
            parts.append("summary tables: descriptive")
        if "rescored" in e:
            parts.append("stored predictions re-scored (unblinded)")
        out[4] = "; ".join(parts)
    if "[RESULT]" in out[5]:
        if "rescored" in e:
            r = e["rescored"]
            out[5] = f"stored CIs: {vc}; re-scored ({r['tests']} tests, {r['rows']} rows): {r['verdict_counts']}"
        elif vc and not vc.startswith("descriptive"):
            out[5] = vc + (f"; {desc}" if desc else "")
        elif vc.startswith("descriptive"):
            out[5] = f"row-resampling CIs, not reclassified ({e['rows']:,} rows)"
        else:
            out[5] = desc
    for (k, col), txt in RECLASS_FIX.items():
        if k == pid and "[VERIFY]" in out[col]:
            out[col] = txt
    return out


def s15():
    sup = os.path.join(ROOT, "docs", "MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md")
    t5 = md_table(sup, "M4.5", 0)
    rows5 = [[tex(c).replace("[RESULT]", "\\textcolor{blue}{[RESULT]}").replace("[VERIFY]", "\\textcolor{blue}{[VERIFY]}") for c in r] for r in t5[1:]]
    t4 = md_table(sup, "M4.4", 0)
    rs = json.load(open(RECLASS_JSON, encoding="utf-8"))
    rows4 = []
    for r in t4[1:]:
        cells = _reclass_cells(r[0].strip(), [c for c in r], rs)
        rows4.append([tex(c) if ("\\" not in c and "$" not in c) else c for c in cells])
    legend = ("Part a lists claims, hypotheses and rules that were dropped, withdrawn, replaced or corrected, with the commit that recorded the action and its timing relative to the "
              "first retrieval of LG-family results (T\\_res, 30 September 2026, 07:20:22). Part b lists earlier negative results reclassified under the four-way verdict rule "
              "(WRAPUP 1.4 (b)); original verdicts are not changed. Its CI-type, reclassification and value columns count the stored rows of the reclassification table "
              "data/processed/lgw/lgw\\_rescore.csv (WRAPUP execution step C7, 1 October 2026) by CI type and by four-way verdict (tools/build\\_reclass\\_summary.py; no value "
              "recomputed); many rows are curve points of one experiment, so the counts describe the stored tables and are not tests. Audit entries: docs/AUDIT\\_2026-09-26.md. "
              "Source: docs/MANUSCRIPT\\_DRAFT\\_SUPPORT\\_2026-09-30.md M4.5 and M4.4 (draft of 30 September 2026).")
    a = longtable([tex(h) for h in t5[0]], rows5, ws=[0.03, 0.17, 0.08, 0.12, 0.30, 0.18, 0.12], title="a. Selection and withdrawal history")
    b = longtable([tex(h) for h in t4[0]], rows4, ws=[0.07, 0.15, 0.10, 0.06, 0.13, 0.17, 0.12, 0.20], title="b. Past negative results, reclassified")
    return table_head(15, "Selection and withdrawal history and earlier negative results.", legend) + a + "\n\\clearpage\n" + b   # part b starts on a new page (no orphaned title)


def s16():
    t = md_table(os.path.join(ROOT, "docs", "MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md"), "Computing environments and cross-environment rules", 0)
    rows = [[tex(c) for c in r] + [""] for r in t[1:]]
    rows[0][3] = "3.11.10"
    rows[1][3] = "NumPy 2.1.1 (third WF run 2.4.6) / pandas 2.3.3 / scikit-learn 1.9.1 / SciPy 1.17.1"
    rows[2][3] = "1.2.10 / 2.4.1 (CUDA build 12.1, run on CPU) / not installed"
    rows[3][3] = "none"
    rows[4][3] = "CPU nodes"
    h = [tex(x) for x in t[0]] + ["Cloud (WF runs, X job A)"]
    legend = ("LG main run, LGX-A and LGX-B ran on cloud nodes (Rescale iolite-4: 64 CPU cores, four NVIDIA T4 GPUs). LGT, LGF, LGU, LGD, the local continuation of LGX and the "
              "aggregations ran on a shared local server with NVIDIA RTX 3090 GPUs (24 GB) at low priority (nice 10, at most 32 CPU threads, one process per GPU). "
              "Package versions of the WF runs and of X job A (XB, XD-alg, XH, XI, XJ; 5 October 2026) are those recorded in results/rescale\\_wf*/logs\\_wf/ and "
              "results/rescale\\_xbatch\\_A/logs\\_xbatch\\_A/ (env.log, deps.log); XA, XC, XD-4, XE, XG and XM ran on the local server. Each shard records the code hash, the "
              "configuration hash, the package versions and the device. Source: docs/MANUSCRIPT\\_DRAFT\\_METHODS\\_INTRO\\_2026-09-30.md Table 4 and methods.tex (Code availability).")
    return table_head(16, "Software and execution environments.", legend) + longtable(h, rows, ws=[0.22, 0.26, 0.26, 0.26])


S17_ROWS = [
    ("Labels, main regions", "GTN-P/CALM compilation (Streletskiy et al., 2025)", "PANGAEA, doi:10.1594/PANGAEA.972777", "CC BY 4.0", "used"),
    ("Labels, main regions", "ALLena thaw depth, Lena River Delta (Veremeeva et al., 2025)", "PANGAEA, doi:10.1594/PANGAEA.973813; 10.1594/PANGAEA.974408", "CC BY 4.0", "used"),
    ("Labels, main regions", "ABoVE soil moisture and ALT, version 2 (Moore et al., 2025)", "ORNL DAAC, doi:10.3334/ORNLDAAC/2369", "NASA EOSDIS open data, no restrictions; citation required", "used"),
    ("Labels, new regions", "GPR and pit survey points, Tibetan Plateau (Du et al., 2026)", "Zenodo, doi:10.5281/zenodo.21999366; field data doi:10.12072/ncdc.permafrost.db7703.2026", "MIT (Zenodo record); licence of the field data not yet confirmed", "used"),
    ("Labels, new regions", "Ilulissat probing (Scheer et al., 2024)", "PANGAEA, doi:10.1594/PANGAEA.964306", "CC BY 4.0", "used"),
    ("Labels, new regions", "T-MOSAiC myThaw 2021--2023 (Martin et al., 2023; Boike et al., 2024; Hammar et al., 2025)", "PANGAEA, doi:10.1594/PANGAEA.956039; .971586; .974461", "CC BY 4.0", "used"),
    ("Labels, new regions", "Additional CALM events of the PANGAEA compilation", "PANGAEA, doi:10.1594/PANGAEA.972777", "CC BY 4.0", "used"),
    ("Labels, new regions", "Disko Island (Zastruzny et al., 2024)", "PANGAEA, doi:10.1594/PANGAEA.967139", "CC BY 4.0", "used"),
    ("Labels, new regions", "Tavvavuoma (Sannel, 2020)", "Bolin Centre Database, doi:10.17043/sannel-2020-temperature-1", "CC BY 4.0", "used"),
    ("Labels, new regions", "Cape Mamontov Klyk (Grosse, 2007)", "PANGAEA, doi:10.1594/PANGAEA.611409", "CC BY 3.0", "used"),
    ("Labels, new regions", "Pedon data (Palmtag et al., 2022)", "Bolin Centre Database, doi:10.17043/palmtag-2022-pedon-1", "ODC-By", "used"),
    ("Labels, new regions", "FireALT, measured depths only (Talucci et al., 2024)", "NSF Arctic Data Center, doi:10.18739/A2RN3092P", "CC BY 4.0", "used"),
    ("Labels, new regions", "Syrdakh observatory (Pohl et al., 2026)", "Zenodo, doi:10.5281/zenodo.19890671", "CC BY 4.0", "used"),
    ("Labels, new regions", "Kolyma Water-Balance Station (Makarieva et al., 2017)", "PANGAEA, doi:10.1594/PANGAEA.881754", "CC BY 3.0", "used"),
    ("Labels, new regions", "Indigirka tree mounds (Liang et al., 2023)", "PANGAEA, doi:10.1594/PANGAEA.961876", "CC BY 4.0", "used"),
    ("Labels, new regions", "Yamal transect (Walker et al., 2009)", "PANGAEA, doi:10.1594/PANGAEA.842711", "CC BY 3.0", "used (PANGAEA rows)"),
    ("Label definition test", "Temperature-derived labels (Fu, 2025)", "figshare, doi:10.6084/m9.figshare.29206613.v1", "CC BY 4.0", "L39 only"),
    ("Labels, licence not confirmed", "CALM web files, Abisko and Kapp Linn\\'e (\\AA kerman, 1998; \\AA kerman and Johansson, 2008)", "https://www2.gwu.edu/\\textasciitilde calm/data/north.htm", "not confirmed", "full-data edition only"),
    ("Labels, licence not confirmed", "CUSP v1.1 synthesis (Jorgenson and Kanevskiy, 2025; Petrone et al., 2016)", "github.com/jonschwenk/cusp v1.1; doi:10.18739/A27P8TG0G (CC0 1.0); doi:10.1594/PANGAEA.845258 (CC BY-NC-SA 3.0)", "not confirmed", "full-data edition only"),
    ("Labels, licence not confirmed", "NSIDC GGD353 thaw tubes (Nixon, 2003)", "doi:10.7265/7m84-k262", "provider asks to be consulted", "full-data edition only"),
    ("Labels, licence not confirmed", "Yamal transect, ten report-derived rows", "report", "not confirmed", "full-data edition only"),
    ("Descriptive only", "NSIDC GGD402; Yang and Qiu (2026)", "", "", "parsed, no analysis"),
    ("Covariates", "ERA5-Land monthly averaged data (Mu\\~noz-Sabater et al., 2021)", "Copernicus Climate Data Store, doi:10.24381/cds.68d2bb30", "Copernicus licence (CC BY type, CDS page)", "used"),
    ("Covariates", "ESA CCI Permafrost ALT, version 4.0 (Westermann et al., 2024)", "CEDA, doi:10.5285/d34330ce3f604e368c06d76de1987ce5", "ESA CCI Permafrost terms of use (registered and unregistered use; citation required)", "used"),
    ("Covariates", "SoilGrids 2.0 (Poggio et al., 2021)", "ISRIC WCS service", "CC BY 4.0 (release 'latest' at access; version tag not recorded)", "used"),
    ("Covariates", "Copernicus DEM GLO-30", "doi:10.5270/ESA-c5d3d65", "\\copyright{} DLR e.V. 2010--2014 and Airbus 2014--2018, Copernicus programme", "used"),
    ("Covariates, XM only", "ESA WorldCover 10 m 2021 v200 (Zanaga et al., 2022)", "Zenodo, doi:10.5281/zenodo.7254221", "CC BY 4.0; \\copyright{} ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium", "XM only"),
    ("Covariates, XM only", "Sentinel-2 L2A, July and August 2019--2023 (534 scenes)", "Earth Search v1 STAC (Element 84), collection sentinel-2-l2a; AWS Open Data sentinel-cogs", "Copernicus Sentinel data terms (free, full and open); Contains modified Copernicus Sentinel data [2019--2023]", "XM only"),
    ("Map masks", "ESA CCI Permafrost extent (permafrost fraction), version 4.0, 1997--2021 mean", "data set DOI not yet confirmed", "ESA CCI data policy: free and open access", "maps only"),
    ("Map masks", "JRC Global Surface Water occurrence (Pekel et al., 2016)", "Pekel et al., Nature 540, 418--422 (2016), doi:10.1038/nature20584", "", "maps only"),
    ("Optional", "KPDC Council case study (KOPRI-KPDC-00002125, -00002707, -00002955)", "KPDC", "KPDC access conditions", "\\textcolor{blue}{[DECISION: KPDC 자료, SI 포함 여부와 식별자, 이용 조건]}"),
]


def s17():
    rows = [list(r) for r in S17_ROWS]
    legend = ("Licences as recorded at access; rows from sources without a confirmed licence are excluded from the deposited tables and from the licence-verified edition used for "
              "the main verdicts. Values from these sources can be obtained from the original providers. Source: docs/MANUSCRIPT\\_DRAFT\\_METHODS\\_INTRO\\_2026-09-30.md "
              "(Data availability, 30 September 2026) and front.tex (Data availability); licence status per data/processed/fidelity\\_base\\_v4\\_meta.json (by\\_source).")
    return table_head(17, "Data sources, licences and excluded data.", legend) + \
        longtable(["Use", "Data set", "Repository and identifier", "Licence", "Status"], rows, ws=[0.13, 0.30, 0.29, 0.16, 0.12])


def s18():
    ph = [["Pending: registered data deadline 11 October 2026 (XF; survey rows of docs/research/2026-10-04/open\\_regions.md 4.1--4.4 with their XF eligibility).", "", "", "", "", "", ""]]
    legend = ("Public field ALT data sets surveyed for independent regions (survey of 4 October 2026, docs/research/2026-10-04/open\\_regions.md) and the outcome of the eligibility "
              "rules of Supplementary Methods 2 applied blind in XF. Rows are filled after the XF run (placement fixed before results: Supplementary Information). " + XR.PEND_XF)
    return table_head(18, "Survey of public data for independent regions (XF).", legend) + \
        longtable(["Data set", "Identifier", "Licence", "Method and period", "Size and relation to existing regions", "Survey decision", "XF eligibility"], ph,
                  ws=[0.20, 0.14, 0.10, 0.14, 0.18, 0.12, 0.12])


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    parts = ["% si_tables.tex : generated by tools/make_si_tables.py (do not edit by hand; edit the script or the claims READMEs).",
             "% Supplementary Tables S1-S18 (MANUSCRIPT_SPEC 8), numbered in order of first citation (tools/si_numbering.py)."]
    blocks = {}
    for old, fn in enumerate((s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18), start=1):
        blocks[SN.TAB[old]] = SN.remap(fn(), 'en')      # former numbers inside the generators -> first-citation numbers
    for new in sorted(blocks):
        parts.append(blocks[new])
    out = "\n\n".join(parts) + "\n"
    open(OUT, "w", encoding="utf-8").write(out)
    open(LOG, "w", encoding="utf-8").write(f"untranslated cells (Hangul dropped, marked with dagger): {len(UNTRANSLATED)}\n" + "\n".join(UNTRANSLATED) + "\n")
    print(f"wrote {OUT}; untranslated cells {len(UNTRANSLATED)} (see {LOG})")


if __name__ == "__main__":
    main()
