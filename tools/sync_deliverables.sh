#!/bin/bash
# 최종 산출물만 deliverables/ 로 모은다(복사본). 원본은 각 작업 폴더에 그대로 두고, 이 스크립트를 다시 돌리면 최신판으로 갱신된다.
# 실행: bash tools/sync_deliverables.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=deliverables
mkdir -p "$D"/{01_발표자료,02_논문초안,03_그림/{논문그림,ALT지도,보충그림,방법도식,슬라이드용},04_참고문헌,05_보고서}

cp_if() { [ -e "$1" ] && cp -p "$1" "$2" || echo "[sync] 없음(건너뜀): $1"; }

# 01 발표 자료
cp_if deck/render/permafrost_paper_report.pptx "$D/01_발표자료/연구결과_보고덱.pptx"
cp_if deck/render/permafrost_paper_report.pdf  "$D/01_발표자료/연구결과_보고덱.pdf"
cp_if "deck/render/발표대본_연구결과보고_2026-10-06.pdf" "$D/01_발표자료/발표대본과_배경지식.pdf"

# 02 논문 초안(검토용 = 그림이 본문 안, 제출형식 = 그림이 끝)
cp_if paper/manuscript/en/main_review.pdf "$D/02_논문초안/영문_본문_검토용.pdf"
cp_if paper/manuscript/en/main.pdf        "$D/02_논문초안/영문_본문_제출형식.pdf"
cp_if paper/manuscript/en/si_main.pdf     "$D/02_논문초안/영문_보충자료_SI.pdf"
cp_if paper/manuscript/ko/main_review.pdf "$D/02_논문초안/국문_본문_검토용.pdf"
cp_if paper/manuscript/ko/main.pdf        "$D/02_논문초안/국문_본문_제출형식.pdf"

# 03 그림
F=outputs/figures/paper/v3_restructure
# 원고 번호(처음 인용 순서)로 복사한다. 파일 이름(작업용) → 원고 번호: Fig1→1, Fig2→2, Fig3→3, Fig6→4, Fig4→5, Fig7→6, Fig5→7
rm -rf "$D/03_그림/논문그림" && mkdir -p "$D/03_그림/논문그림"
for pair in 1:1 2:2 3:3 6:4 4:5 7:6 5:7; do
  src=${pair%%:*}; dst=${pair##*:}
  for ext in pdf png; do cp_if "$F/Fig${src}.${ext}" "$D/03_그림/논문그림/Figure${dst}.${ext}"; done
  cp_if "$F/Fig${src}_legend.md" "$D/03_그림/논문그림/Figure${dst}_설명문(작업파일_Fig${src}).md"
done
for f in "$F"/Table1*; do [ -e "$f" ] && cp -p "$f" "$D/03_그림/논문그림/"; done
rsync -a --delete --include='*.pdf' --include='*.png' --include='*_legend.md' --exclude='*' "$F/maps/" "$D/03_그림/ALT지도/"
# 보충 그림(원고 SI 번호: 파일 FigS7–S12 = S7–S12, FigS14 = S13, FigS15 = S14)
mkdir -p "$D/03_그림/보충그림" && rsync -a --delete --include='FigS*.pdf' --include='FigS*.png' --include='FigS*_legend.md' --exclude='*' "$F/si/" "$D/03_그림/보충그림/"
[ -d deck/assets/paper_report/method ] && rsync -a --delete --include='*.png' --include='*.pdf' --include='*.md' --exclude='*' deck/assets/paper_report/method/ "$D/03_그림/방법도식/"
rsync -a --delete --include='*.png' --exclude='*' deck/assets/paper_report/slide_panels/ "$D/03_그림/슬라이드용/"
rsync -a --include='*_slide.png' --exclude='*' deck/assets/paper_report/maps/ "$D/03_그림/슬라이드용/"

# 04 참고문헌(PDF 는 references/ 에 두고 색인과 바로가기만)
cp_if references/INDEX.md "$D/04_참고문헌/INDEX.md"
cp_if references/README_STUDY_GUIDE.md "$D/04_참고문헌/README_STUDY_GUIDE.md"
ln -sfn ../../references "$D/04_참고문헌/PDF_전체폴더"

# 05 보고서·판정 기록
cp_if docs/MORNING_REPORT_2026-10-05.md "$D/05_보고서/아침보고_2026-10-05.md"
cp_if docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md "$D/05_보고서/최종묶음_실험계획과_판정기록.md"
cp_if docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md "$D/05_보고서/추가등록_XK_XL.md"
cp_if docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md "$D/05_보고서/추가등록_XM.md"
cp_if docs/QA_FINAL_REVIEW_2026-10-02.md "$D/05_보고서/질의응답_Q1-Q15.md"
cp_if docs/QA_FOLLOWUP_2026-10-04.md "$D/05_보고서/질의응답_후속.md"

# 색인
{
  echo "# 최종 산출물 모음"
  echo
  echo "갱신: $(date '+%Y-%m-%d %H:%M') · 생성 스크립트: \`tools/sync_deliverables.sh\`(원본은 각 작업 폴더, 여기는 복사본)"
  echo
  for sub in 01_발표자료 02_논문초안 03_그림/논문그림 03_그림/ALT지도 03_그림/보충그림 03_그림/방법도식 03_그림/슬라이드용 04_참고문헌 05_보고서; do
    echo "## $sub"
    ( cd "$D/$sub" 2>/dev/null && ls -1 | sed 's/^/- /' ) || true
    echo
  done
} > "$D/README.md"
echo "[sync] 완료: $D"
