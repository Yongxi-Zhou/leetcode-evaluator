#!/bin/bash
# Build the PREPRINT / arXiv version (named authors, no line numbers, page numbers).
# Single source of truth is main.tex; this just delegates to build.sh in preprint mode.
# Output: main-preprint.pdf
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/build.sh" preprint
