#!/usr/bin/env bash
# build_submission_png.sh (2026-10-20): compressed first-submission PDF (Scientific Reports single-file option, 3 MB). Same text as main.tex, but the
# figures are included as the 600 dpi PNG exports instead of the vector PDFs, then the rasters are downsampled to 150 dpi (JPEG). The full-resolution
# vector manuscript (main.pdf) is delivered alongside. Usage: bash tools/build_submission_png.sh en ko
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
for lang in "$@"; do
  src=paper/manuscript/${lang}_${VER:-v13}
  sed 's/\\figdirv#1\.pdf/\\figdirv#1.png/g' "$src/main.tex" > "$src/main_png.tex"
  ( cd "$src"; HERE=$(pwd); TPL="$HERE/../template/sn_template/sn-article-template"; export TEXINPUTS=".:$TPL//:" BSTINPUTS=".:$TPL/bst//:" BIBINPUTS="$HERE:"
    mkdir -p build
    run() { nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build main_png.tex > "build/main_png.pass$1.txt" 2>&1; }
    run 1; (cd build && BIBINPUTS="$HERE:" nice -n 10 bibtex main_png > main_png.bibtex.txt 2>&1); run 2; run 3 )
  nice -n 10 gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.6 -dPDFSETTINGS=/printer -dDownsampleColorImages=true -dColorImageResolution=220 \
     -dColorImageDownsampleThreshold=1.0 -dColorImageDownsampleType=/Bicubic -dAutoFilterColorImages=false -dColorImageFilter=/DCTEncode \
     -sOutputFile="$src/main_submission.pdf" "$src/build/main_png.pdf"
  errs=$(grep -c '^\(! \|[^:]*:[0-9]*: \)' "$src/build/main_png.log" 2>/dev/null || true)
  echo "$lang: main_submission.pdf $(stat -c %s "$src/main_submission.pdf") B, pages $(pdfinfo "$src/main_submission.pdf" 2>/dev/null | awk '/^Pages/{print $2}'), latex errors $errs"
done
