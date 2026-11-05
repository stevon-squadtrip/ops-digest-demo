#!/usr/bin/env bash
# Screenshots of site/index.html at desktop (1200) and phone (390) width with headless Edge. Headless Edge will not
# open a window narrower than about 500 px, so the phone shot renders the page inside an iframe that is exactly
# 390 wide and crops to it. Writes outputs/site_screens/1200.png and 390.png. Usage: bash scripts/screenshots.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -f site/index.html ] || { echo "Refusing: site/index.html is missing. Run python -m src.site first."; exit 1; }
[ -x "$EDGE" ] || { echo "Refusing: Edge not found at $EDGE."; exit 1; }
mkdir -p outputs/site_screens
shot() { # width height source
  "$EDGE" --headless=new --disable-gpu --hide-scrollbars --window-size=$1,$2 \
    --screenshot="$(cygpath -w "$ROOT/outputs/site_screens/$1.png")" "file:///$(cygpath -m "$ROOT/$3")" >/dev/null 2>&1
  [ -s "outputs/site_screens/$1.png" ] && echo "wrote outputs/site_screens/$1.png" || { echo "Refusing: no screenshot at $1"; exit 1; }
}
shot 1200 2200 site/index.html
FRAME="outputs/site_screens/frame390.html"
printf '%s\n' '<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0}iframe{border:0;display:block}</style></head>' \
  '<body><iframe src="../../site/index.html" width="390" height="3200"></iframe></body></html>' > "$FRAME"
shot 390 3200 "$FRAME"
