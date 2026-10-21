#!/usr/bin/env bash
# sync_deliverables.sh (v2, 2026-10-21): rebuild deliverables/ as COPIES of the current outputs. Sources stay where the scripts write them
# (paper/manuscript/*_vNN = LaTeX, outputs/figures/paper/vNN = figures, deck/render = decks, docs/ = reports). Re-run after any change.
#   VER=v14 bash tools/sync_deliverables.sh            # full: compressed submission copy + copies + README
#   VER=v14 bash tools/sync_deliverables.sh --no-build # copies only (manuscripts must already be built)
# Layout (one folder per kind, current version only; older bundles live in deliverables/archive/):
#   deliverables/paper/               manuscript PDFs (en · ko, full and compressed), supplementary, compliance table, figures_main/, figures_supplementary/, source_data/, tables/
#   deliverables/figure_proposals/    design alternatives (PNG) + the proposal document
#   deliverables/deck/                latest presentation deck (PDF · PPTX)
#   deliverables/reports/             audits, research notes, session state, latest GPT handoff
#   deliverables/references/          literature index + link to the PDF folder
#   deliverables/archive/             earlier bundles (01_–15_), kept as they were
set -uo pipefail
cd "$(dirname "$0")/.."
VER=${VER:-v14}; D=deliverables; NOBUILD=${1:-}
mkdir -p "$D"/{paper/figures_main,paper/figures_supplementary,paper/source_data,paper/tables,figure_proposals,deck,reports,references,archive}
# one-time: move the dated / numbered bundles into archive/
for d in "$D"/[0-9][0-9]_*; do [ -d "$d" ] && mv "$d" "$D/archive/" && echo "[sync] archived $(basename "$d")"; done
cp_if() { [ -e "$1" ] && cp -p "$1" "$2" || echo "[sync] missing: $1"; }

# ---- paper (current version)
EN=paper/manuscript/en_$VER; KO=paper/manuscript/ko_$VER; FIG=outputs/figures/paper/$VER; SI=outputs/figures/paper/${VER}_si
if [ "$NOBUILD" != "--no-build" ]; then VER=$VER bash tools/build_submission_png.sh en ko > logs/fig_$VER/submission_png.log 2>&1 || echo "[sync] compressed copy failed (see logs/fig_$VER/submission_png.log)"; fi
cp_if "$EN/main.pdf" "$D/paper/manuscript_en.pdf"; cp_if "$EN/main_submission.pdf" "$D/paper/manuscript_en_submission_compressed.pdf"; cp_if "$EN/si.pdf" "$D/paper/supplementary_en.pdf"
cp_if "$KO/main.pdf" "$D/paper/원고_국문.pdf"; cp_if "$KO/main_submission.pdf" "$D/paper/원고_국문_압축본.pdf"; cp_if "$KO/si.pdf" "$D/paper/보충자료_국문.pdf"
cp_if "$EN/COMPLIANCE_$VER.md" "$D/paper/COMPLIANCE.md"
rm -rf "$D/paper/figures_main"/* "$D/paper/figures_supplementary"/* "$D/paper/source_data"/* "$D/paper/tables"/*
for f in "$FIG"/Fig*_*.pdf "$FIG"/Fig*_*.png; do [ -f "$f" ] && cp -p "$f" "$D/paper/figures_main/"; done
i=0
for stem in $(grep -o '^\\sifig{FigS[0-9]*_[a-z_]*}' "$EN/si.tex" | sed 's/\\sifig{//; s/}//'); do
  i=$((i+1)); short=${stem#FigS*_}
  for ext in pdf png; do [ -f "$SI/$stem.$ext" ] && cp -p "$SI/$stem.$ext" "$D/paper/figures_supplementary/Supplementary_Fig_S${i}_${short}.$ext"; done
done
for f in "$FIG"/source_data/* "$SI"/source_data/*; do [ -f "$f" ] && cp -p "$f" "$D/paper/source_data/"; done
for f in "$FIG"/source_data/TableS7_* paper/manuscript/shared_$VER/table_s7*_{en,ko}.tex "$EN/sections/table1.tex" "$KO/sections/table1.tex"; do [ -f "$f" ] && cp -p "$f" "$D/paper/tables/"; done

# ---- figure proposals
rm -rf "$D/figure_proposals"/*; for f in outputs/figures/paper/${VER}_proposals/*.png docs/FIGURE_PROPOSALS_*.md; do [ -f "$f" ] && cp -p "$f" "$D/figure_proposals/"; done

# ---- deck (latest render) and reports
rm -rf "$D/deck"/*; latest=$(ls -t deck/render/*.pdf 2>/dev/null | head -1); [ -n "$latest" ] && cp -p "$latest" "$D/deck/" && { pp="${latest%.pdf}.pptx"; [ -f "$pp" ] && cp -p "$pp" "$D/deck/"; }
rm -rf "$D/reports"/*; for f in docs/AUDIT_*2026-10-2*.md docs/research/2026-10-21/*.md docs/RESET_2026-10-10_KOPRI_FEEDBACK.md SESSION_HANDOFF.md docs/EXPERIMENT_LOG.md; do [ -f "$f" ] && cp -p "$f" "$D/reports/"; done
lh=$(ls -t gpt/handoff/2*.md 2>/dev/null | head -1); [ -n "$lh" ] && cp -p "$lh" "$D/reports/HANDOFF_latest_$(basename "$lh")"

# ---- references
cp_if references/INDEX.md "$D/references/INDEX.md"; ln -sfn ../../references "$D/references/PDF_folder"

# ---- README
{
  echo "# 최종 산출물 모음 (deliverables/)"
  echo
  echo "갱신: $(date '+%Y-%m-%d %H:%M') · 원고 판 $VER · 생성: \`VER=$VER bash tools/sync_deliverables.sh\` (여기는 복사본, 원본은 아래 표의 위치)"
  echo
  echo "| 폴더 | 내용 | 원본 위치 |"; echo "|---|---|---|"
  echo "| \`paper/\` | 영문 · 국문 원고(전체 · 압축 제출본), 보충 자료, 규정 점검표, 본문 그림(\`figures_main/\`), 보충 그림(\`figures_supplementary/\`, 보충 자료 번호), 그림별 원천 자료(\`source_data/\`), 표(\`tables/\`) | \`paper/manuscript/{en,ko}_$VER/\`, \`outputs/figures/paper/$VER/\`, \`${VER}_si/\` |"
  echo "| \`figure_proposals/\` | 그림 설계 대안 렌더와 제안 문서 | \`outputs/figures/paper/${VER}_proposals/\`, \`docs/FIGURE_PROPOSALS_*.md\` |"
  echo "| \`deck/\` | 최신 발표 덱 | \`deck/render/\` |"
  echo "| \`reports/\` | 감사 보고, 조사 노트, 세션 상태, 실험 기록, 최신 GPT 인계 | \`docs/\`, \`SESSION_HANDOFF.md\`, \`gpt/handoff/\` |"
  echo "| \`references/\` | 문헌 색인과 PDF 폴더 바로가기 | \`references/\` |"
  echo "| \`archive/\` | 이전 묶음(01_–15_, 날짜별 판) | 변경 없음 |"
  echo
  for sub in paper paper/figures_main paper/figures_supplementary paper/tables figure_proposals deck reports references; do
    echo "## $sub"; ( cd "$D/$sub" 2>/dev/null && ls -1p | grep -v '/$' | sed 's/^/- /' ) || true; echo
  done
  echo "## archive"; ( cd "$D/archive" && ls -1 | sed 's/^/- /' ); echo
} > "$D/README.md"
echo "[sync] done: $D ($(du -sh "$D/paper" | cut -f1) paper)"
