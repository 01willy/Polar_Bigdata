# -*- coding: utf-8 -*-
"""v6_results_doc.py: docs/numbers_v6.json(매크로)에서 결과 요약 문서 docs/RESULTS_v6_<date>.md 를 만든다. 수치를 손으로 옮기지 않는다.
실행: python3 scripts/2_evaluation/v6_results_doc.py
"""
from __future__ import annotations
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = json.load(open(ROOT / "docs" / "numbers_v6.json", encoding="utf-8"))
g = lambda k: N.get(k, "--")
REG = [("Ak", "알래스카"), ("Le", "레나 델타"), ("Ca", "캐나다 북서부")]
date = time.strftime("%Y-%m-%d")
L = []
L.append(f"# 결과 요약 v6 ({date}): 관측일 정합 주 강제력, 실행 {g('runTag')}\n")
L.append("생성: `scripts/2_evaluation/v6_results_doc.py` ← `docs/numbers_v6.json` ← `scripts/2_evaluation/v6_numbers.py` ← `data/processed/v5/runs/`. 값 형식 `Δ [95% 블록 부트스트랩 구간]`, 단위 cm, Δ = RMSE(앞) − RMSE(뒤), 음수 = 앞이 낫다.\n")
L.append("## 1. 자료\n")
L.append("| 지역 | 셀-연도 | 셀 | 블록 | 연도 수 | 다년 셀(≥4년) | 최다 연도(비율) |\n|---|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | {g('ncy'+k)} | {g('ncell'+k)} | {g('nblk'+k)} | {g('nyr'+k)} | {g('nmulti'+k)} | {g('topyr'+k)} ({g('topyrshare'+k)}%) |")
L.append("\n## 2. 질문 1: 그 해 기후 입력(관측일 정합 − 평균 입력, 공간 블록)\n")
L.append("| 지역 | 원천 Stefan | 재보정 Stefan | Kudryavtsev | 직접 학습 | 물리 입력 | 잔차 결합 |\n|---|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | {g('dforc'+k+'Pz')} | {g('dforc'+k+'Po')} | {g('dforc'+k+'Ku')} | {g('dforc'+k+'Dz')} | {g('dforc'+k+'Fo')} | {g('dforc'+k+'Ro')} |")
L.append("\n## 3. 질문 2: 검증 방식에 따른 RMSE(cm)와 과대평가\n")
L.append("| 지역 | 방법 | 무작위 | 블록 | 연도 | 블록×연도 | 지역 |\n|---|---|---|---|---|---|---|")
for k, nm in REG:
    for m, mn in (("Po", "재보정 Stefan"), ("Dz", "직접 학습"), ("Ro", "잔차 결합")):
        L.append(f"| {nm} | {mn} | {g('rmse'+k+'rand'+m)} | {g('rmse'+k+'blk'+m)} | {g('rmse'+k+'yr'+m)} | {g('rmse'+k+'blkyr'+m)} | {g('rmse'+k+'reg'+m)} |")
L.append("\n무작위 → 블록 과대평가(직접 학습): " + ", ".join(f"{nm} {g('optim'+k+'Dz')} cm" for k, nm in REG))
L.append("\n지역 홀드아웃의 재보정 Stefan 대비 차이: " + "; ".join(f"{nm} 직접 학습 {g('dref'+k+'regDz')}, 잔차 결합 {g('dref'+k+'regRo')}" for k, nm in REG))
L.append("\n## 4. 질문 3: 재보정 Stefan 대비 차이(공간 블록)\n")
L.append("| 지역 | 직접 학습 | 물리 입력 | 유사라벨 증강 | 잔차 결합 λ=0.5 | Kudryavtsev |\n|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | {g('dref'+k+'blkDz')} | {g('dref'+k+'blkFo')} | {g('dref'+k+'blkDo')} | {g('dref'+k+'blkRo')} | {g('dref'+k+'blkKu')} |")
L.append("\n공변량 사다리(잔차 결합 − Stefan, 블록):\n")
L.append("| 지역 | 기후만 | +지형 | +토양 | +MODIS | +고해상 | 전부 |\n|---|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | {g('dfeat'+k+'Fclim')} | {g('dfeat'+k+'Fterr')} | {g('dfeat'+k+'Fsoil')} | {g('dfeat'+k+'Fmodis')} | {g('dfeat'+k+'Fhires')} | {g('dfeat'+k+'Fall')} |")
L.append("\n융해 도일 3분위(잔차 결합 − Stefan): " + "; ".join(f"{nm} 추움 {g('dtdd'+k+'cold')}, 중간 {g('dtdd'+k+'mid')}, 따뜻 {g('dtdd'+k+'warm')}" for k, nm in REG))
L.append("\n학습기(잔차 결합 − Stefan / 직접 학습 − Stefan, 블록):\n")
L.append("| 지역 | CatBoost | RF | MLP | TabM | FT-T |\n|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | " + " | ".join(f"{g('dlrn'+k+l+'Ro')} / {g('dlrn'+k+l+'Dz')}" for l in ("catboost", "rf", "mlp", "tabm", "ftt")) + " |")
L.append("\n## 5. 질문 4: 기존 제품과 같은 셀·같은 해(공통 셀 RMSE cm, 전체 / 학습 지점 25 km 밖)\n")
L.append("| 지역 | n | 재보정 Stefan | 잔차 결합 | 직접 학습 | Wei 2026 | Liu 2024 | CCI v4 | CCI v5 | Aalto 2018 |\n|---|---|---|---|---|---|---|---|---|---|")
for k, nm in REG:
    L.append(f"| {nm} | {g('ccn'+k+'all')} / {g('ccn'+k+'mask')} | " + " | ".join(f"{g('cc'+k+s+'all')} / {g('cc'+k+s+'mask')}" for s in ("Po", "Ro", "Dz", "Wei", "Liu", "Ccifour", "Ccifive", "Aalto")) + " |")
L.append("\n짝지은 차이(잔차 결합 − 제품, 연도 정합, 25 km 마스크): " + "; ".join(f"{nm} Wei {g('dprod'+k+'weiRomask')}, Liu {g('dprod'+k+'liuRomask')}, CCI v4 {g('dprod'+k+'ccifourRomask')}" for k, nm in REG))
L.append("\n## 6. 연도별\n")
L.append("연도 홀드아웃 연도 편차 상관(잔차 결합 / 재보정 Stefan): " + "; ".join(f"{nm} {g('ranom'+k+'yrRo')} / {g('ranom'+k+'yrPo')}" for k, nm in REG))
L.append("\n다년 셀 연도 간 변화 상관(잔차 결합 / Wei / Liu): " + "; ".join(f"{nm} {g('rdelta'+k+'Ro')} / {g('rdelta'+k+'Wei')} / {g('rdelta'+k+'Liu')} (쌍 {g('npairs'+k)})" for k, nm in REG))
L.append("\n잔차 결합 연도별 편향 범위(블록 CV, n ≥ 50): " + "; ".join(f"{nm} {g('ybiasmin'+k)} ~ {g('ybiasmax'+k)} cm(최소 {g('ybiasyrmin'+k)}년)" for k, nm in REG))
L.append("\n## 7. 지도\n")
L.append(f"적용 가능 영역 안쪽 비율: 알래스카 {g('aoaAk')}%, 레나 델타 {g('aoaLe')}%. 지도 `data/processed/v5/maps/<Region>/pred_<year>.csv.gz`(관측일 정합 학습, 그 해 격자 강제력).\n")
out = ROOT / "docs" / f"RESULTS_v6_{date}.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8"); print(f"[doc] {out.relative_to(ROOT)} {len(L)} lines")
