#!/bin/bash
# Build the COLM 2026 workshop submission.
#   ./build-colm.sh submission  -> main-colm-submission.pdf  (double-blind + line numbers; DEFAULT)
#   ./build-colm.sh preprint    -> main-colm-preprint.pdf    (named authors, no line numbers)
#   ./build-colm.sh final       -> main-colm-final.pdf       (camera-ready, named authors)
# Single source of truth is main.tex; this only swaps the colm2026_conference package option.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure figures are reachable (mirrors ../build.sh).
if [ ! -e figures ]; then
  ln -sf ../figures figures
fi

MODE="${1:-submission}"
case "$MODE" in
  submission|review|anon) OPT="submission"; OUT="main-colm-submission.pdf" ;;
  preprint)               OPT="preprint";   OUT="main-colm-preprint.pdf" ;;
  final|camera)           OPT="final";      OUT="main-colm-final.pdf" ;;
  *) echo "Usage: ./build-colm.sh [submission|preprint|final]"; exit 1 ;;
esac

TMP_TEX="main.build.tex"
cleanup() { rm -f main.build.aux main.build.bbl main.build.blg main.build.log main.build.out main.build.pdf "$TMP_TEX"; }
trap cleanup EXIT

sed -E "s/\\\\usepackage\[(submission|preprint|final)\]\{colm2026_conference\}/\\\\usepackage[$OPT]{colm2026_conference}/" \
  main.tex > "$TMP_TEX"

rm -f main.build.aux main.build.bbl "$OUT"
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" > /dev/null
bibtex main.build > /dev/null
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" > /dev/null
pdflatex -interaction=nonstopmode -jobname=main.build "$TMP_TEX" | grep -E "Output written|Warning: Citation|undefined" | grep -v "Float" || true
cp main.build.pdf "$OUT"
echo "Wrote $OUT"
