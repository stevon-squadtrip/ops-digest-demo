#!/usr/bin/env bash
# Pre-commit gate. Three checks, one table, non zero exit on any failure.
#   1. full test suite
#   2. no drift: a fresh site build is byte identical to the committed site/index.html
#   3. dash scan: an em dash or en dash in any tracked text file fails
# Usage, from the repo root: bash scripts/gate.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=".venv/Scripts/python"; [ -x "$PY" ] || PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
status=0
rows=()

out=$("$PY" -m pytest -q 2>&1 | tail -1)
if echo "$out" | grep -q "failed\|error"; then status=1; r="FAIL"; else r="PASS"; fi
rows+=("1 full suite|$r|$out")

tmpw="$("$PY" -c "import tempfile; print(tempfile.mkdtemp())")"
if command -v cygpath >/dev/null 2>&1; then tmp="$(cygpath -u "$tmpw")"; else tmp="$tmpw"; fi
if "$PY" -c "from pathlib import Path; from src import site; site.build(Path(r'$tmpw')/'index.html')" >/dev/null 2>&1; then
  if cmp -s "$tmp/index.html" "site/index.html"; then r="PASS"; d="rebuild is byte identical to site/index.html"; else r="FAIL"; d="rebuild differs from site/index.html (run python -m src.site)"; status=1; fi
else
  r="FAIL"; d="site build failed"; status=1
fi
rm -rf "$tmp"
rows+=("2 no drift|$r|$d")

hits=0; scanned=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue
  case "$f" in *.png|*.jpg|*.jpeg|*.gif|*.pdf|*.ico|*.woff|*.woff2) continue;; esac
  scanned=$((scanned+1))
  c=$("$PY" -c "import sys; t=open(sys.argv[1],'rb').read().decode('utf-8','replace'); print(t.count(chr(0x2014))+t.count(chr(0x2013)))" "$f")
  if [ "$c" != "0" ]; then hits=$((hits+c)); echo "  dash in $f: $c"; fi
done < <(git ls-files | sed 's/^"//; s/"$//')
if [ "$hits" = "0" ] && [ "$scanned" != "0" ]; then r="PASS"; else r="FAIL"; status=1; fi
rows+=("3 dash scan|$r|$scanned tracked text files scanned, $hits dashes")

printf '%s\n' "| Check | Result | Detail |" "|---|---|---|"
for row in "${rows[@]}"; do IFS='|' read -r a b c <<< "$row"; printf '| %s | %s | %s |\n' "$a" "$b" "$c"; done
exit $status
