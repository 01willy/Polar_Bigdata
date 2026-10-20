#!/usr/bin/env bash
# package_paper_v14.sh: 논문 초안 v14 납품 묶음(deliverables/15_paper_v14_2026-10-20). 원고 PDF(전체 · 압축), 보충 자료, 본문 그림, 보충 그림(최종 번호), 점검표.
set -uo pipefail
cd "$(dirname "$0")/.."
D=deliverables/15_paper_v14_2026-10-20; rm -rf "$D/figures_main" "$D/figures_supplementary"; mkdir -p "$D/figures_main" "$D/figures_supplementary"
# compressed first-submission copy: figures as rasters (tools/build_submission_png.sh; Sci Rep single-file limit 3 MB)
VER=v14 bash tools/build_submission_png.sh en ko
cp paper/manuscript/en_v14/main.pdf "$D/manuscript_en.pdf"; cp paper/manuscript/en_v14/main_submission.pdf "$D/manuscript_en_submission_compressed.pdf"
cp paper/manuscript/en_v14/si.pdf "$D/supplementary_en.pdf"
cp paper/manuscript/ko_v14/main.pdf "$D/원고_국문.pdf"; cp paper/manuscript/ko_v14/main_submission.pdf "$D/원고_국문_압축본.pdf"
cp paper/manuscript/ko_v14/si.pdf "$D/보충자료_국문.pdf"
[ -f paper/manuscript/en_v14/COMPLIANCE_v14.md ] && cp paper/manuscript/en_v14/COMPLIANCE_v14.md "$D/"
for f in outputs/figures/paper/v14/Fig*_*.pdf outputs/figures/paper/v14/Fig*_*.png; do [ -f "$f" ] && cp "$f" "$D/figures_main/"; done
i=0
for stem in $(grep -o '^\\sifig{FigS[0-9]*_[a-z_]*}' paper/manuscript/en_v14/si.tex | sed 's/\\sifig{//; s/}//'); do
  i=$((i+1)); short=${stem#FigS*_}
  for ext in pdf png; do [ -f "outputs/figures/paper/v14_si/$stem.$ext" ] && cp "outputs/figures/paper/v14_si/$stem.$ext" "$D/figures_supplementary/Supplementary_Fig_S${i}_${short}.$ext"; done
done
ls -la "$D" | awk '{print $5, $9}'
