#!/bin/bash
set -e
cd "$(dirname "$0")"

MODE="${1:-preprint}"
TMP_TEX="main.build.tex"

case "$MODE" in
  anon|anonymous|review)
    ANON_VALUE="true"
    PREPRINT_VALUE="false"
    OUTPUT_PDF="main-anon.pdf"
    ;;
  camera|final)
    ANON_VALUE="false"
    PREPRINT_VALUE="false"
    OUTPUT_PDF="main-final.pdf"
    ;;
  preprint|arxiv|"")
    ANON_VALUE="false"
    PREPRINT_VALUE="true"
    OUTPUT_PDF="main-preprint.pdf"
    ;;
  *)
    echo "Usage: ./build.sh [preprint|anon|final]"
    exit 1
    ;;
esac

cleanup() {
  rm -f "$TMP_TEX" main.build.aux main.build.bbl main.build.blg main.build.log main.build.out main.build.pdf
}

trap cleanup EXIT

sed \
  -e "s/^\\\\anonymoussubmissiontrue$/\\\\anonymoussubmission$ANON_VALUE/" \
  -e "s/^\\\\anonymoussubmissionfalse$/\\\\anonymoussubmission$ANON_VALUE/" \
  -e "s/^\\\\preprintversiontrue$/\\\\preprintversion$PREPRINT_VALUE/" \
  -e "s/^\\\\preprintversionfalse$/\\\\preprintversion$PREPRINT_VALUE/" \
  main.tex > "$TMP_TEX"

rm -f main.build.aux main.build.bbl main.build.blg "$OUTPUT_PDF"
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" > /dev/null
bibtex main.build > /dev/null
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" > /dev/null
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" | grep -E "Output written|Warning|Error" | grep -v "^(Package|LaTeX Warning: Float)"
cp main.build.pdf "$OUTPUT_PDF"
echo "Wrote $OUTPUT_PDF"
