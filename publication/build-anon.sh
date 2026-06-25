#!/bin/bash
# Build the ANONYMOUS ACL/ARR review submission (adds line numbers, hides authors).
# Single source of truth is main.tex; this just delegates to build.sh in anon mode.
# Output: main-anon.pdf
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/build.sh" anon
