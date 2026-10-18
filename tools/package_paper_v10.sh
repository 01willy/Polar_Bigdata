#!/usr/bin/env bash
# package_paper_v10.sh: 논문 초안 v10 납품 묶음(deliverables/11_paper_v10_2026-10-16). 원고 PDF, 본문 그림, 보충 그림(최종 번호), 점검표.
set -uo pipefail
cd "$(dirname "$0")/.."
D=deliverables/11_paper_v10_2026-10-16; mkdir -p "$D/figures_main" "$D/figures_supplementary"
cp paper/manuscript/en_v10/main.pdf "$D/manuscript_en.pdf"; cp paper/manuscript/en_v10/si.pdf "$D/supplementary_en.pdf"
cp paper/manuscript/en_v10/main_submission.pdf "$D/manuscript_en_submission_compressed.pdf" 2>/dev/null
cp paper/manuscript/ko_v10/main_submission.pdf "$D/원고_국문_압축본.pdf" 2>/dev/null
cp paper/manuscript/ko_v10/main.pdf "$D/원고_국문.pdf"; cp paper/manuscript/ko_v10/si.pdf "$D/보충자료_국문.pdf"
cp paper/manuscript/en_v10/COMPLIANCE_v10.md "$D/"
for f in outputs/figures/paper/v10/Fig*_*.pdf outputs/figures/paper/v10/Fig*_*.png; do [ -f "$f" ] && cp "$f" "$D/figures_main/"; done
# 보충 그림: si.tex 의 \sifig 순서 = 최종 번호
i=0
for stem in $(grep -o '^\\sifig{FigS[0-9]*_[a-z_]*}' paper/manuscript/en_v10/si.tex | sed 's/\\sifig{//; s/}//'); do
  i=$((i+1)); short=${stem#FigS*_}
  for ext in pdf png; do [ -f "outputs/figures/paper/v10_si/$stem.$ext" ] && cp "outputs/figures/paper/v10_si/$stem.$ext" "$D/figures_supplementary/Supplementary_Fig_S${i}_${short}.$ext"; done
done
ls -la "$D" "$D/figures_supplementary" | awk '{print $5, $9}'
