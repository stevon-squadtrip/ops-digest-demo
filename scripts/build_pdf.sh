#!/usr/bin/env bash
# Builds docs/digest.pdf from site/index.html with headless Edge (letter size and the print stylesheet come from the
# page), then checks the PDF text and exits non zero on any mismatch. Usage, from the repo root: bash scripts/build_pdf.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=".venv/Scripts/python"; [ -x "$PY" ] || PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
SRC="site/index.html"
OUT="docs/digest.pdf"

[ -f "$SRC" ] || { echo "Refusing: $SRC is missing. Run python -m src.site first."; exit 1; }
[ -x "$EDGE" ] || { echo "Refusing: Edge not found at $EDGE."; exit 1; }
command -v pdftotext >/dev/null 2>&1 || { echo "Refusing: pdftotext is not on the path."; exit 1; }

mkdir -p docs
rm -f "$OUT"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$ROOT/$OUT")" "file:///$(cygpath -m "$ROOT/$SRC")" >/dev/null 2>&1
[ -s "$OUT" ] || { echo "Refusing: Edge wrote no PDF."; exit 1; }

TXT="$(mktemp -t digest_pdf_XXXX.txt)"
pdftotext "$OUT" "$TXT"
"$PY" - "$TXT" "$OUT" <<'EOF'
import json, re, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
pages = raw.count("\f")
text = re.sub(r"\s+", " ", raw)
s = json.loads(Path("outputs/plan_check_summary.json").read_text(encoding="utf-8"))
first = json.loads(Path("outputs/suppressed.json").read_text(encoding="utf-8"))["suppressed"][0]
checks = [
    (f"{s['met_at_open']} of the {s['at_open_denominator']} plans with data at open had already met their target when they opened", 1),
    ("Demo on synthetic data. Built by Stevon Judd.", 2),
    ("one click away", 0),
    ("Sources and hidden items are on the live page", 1),
    ("pairs were hidden this week", 1),
    (r"\bnan\b", 0),
    (r"\bNone\b", 0),
    (f"{first['center']} {first['metric']}", 0),
]
n_items = len(json.loads(Path("outputs/items.json").read_text(encoding="utf-8"))["items"])
rows = [("page count", pages, "", "")]
ok = True
for needle, want in checks:
    got = len(re.findall(needle, text)) if needle.startswith("\\b") else text.count(needle)
    rows.append((needle, got, want, "ok" if got == want else "MISMATCH"))
    ok = ok and got == want
tails = (text.count("(medical)") + text.count("(ops)") + text.count("The metric is medical; the task is operations.")
         + text.count("The metric is operations; the task is medical."))
rows.append(("owner tails", tails, n_items, "ok" if tails == n_items else "MISMATCH"))
ok = ok and tails == n_items
print("| Check | Found | Expected | Result |")
print("|---|---|---|---|")
for n, g, w, r in rows:
    print(f"| {n} | {g} | {w} | {r} |")
print(f"{sys.argv[2]}: {Path(sys.argv[2]).stat().st_size / 1024:.0f} KB, {pages} pages")
sys.exit(0 if ok else 1)
EOF
rc=$?
rm -f "$TXT"
exit $rc
