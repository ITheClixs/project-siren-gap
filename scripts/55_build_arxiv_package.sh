#!/bin/bash
# Builds the arXiv submission tarball and then compiles it in a clean directory, because a
# package that only builds inside the working tree is the standard way to fail an arXiv upload.
#
# arXiv runs its own TeX Live over whatever is in the tarball. This paper needs no bibtex pass --
# the bibliography is an inline thebibliography -- so the tarball is self-contained once the tex,
# the tables/ and the figures actually referenced are included.
#
# Usage: bash scripts/55_build_arxiv_package.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/arxiv
STAGE="$OUT/src"
rm -rf "$OUT"
mkdir -p "$STAGE/tables" "$STAGE/figures"

cp paper/paper.tex "$STAGE/"
cp paper/tables/*.tex "$STAGE/tables/"

# Only the figures the document actually includes, in the format it asks for.
python3 - <<'PY'
import re, shutil
from pathlib import Path
tex = Path("paper/paper.tex").read_text()
wanted = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex))
for w in sorted(wanted):
    src = Path("paper") / w
    if not src.suffix:
        for ext in (".pdf", ".png"):
            if (Path("paper") / (w + ext)).exists():
                src = Path("paper") / (w + ext)
                break
    dst = Path("dist/arxiv/src") / src.relative_to("paper")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"figure: {src} -> {dst}")
PY

# Compile in a copy that has nothing else in it, which is what arXiv sees.
CLEAN=$(mktemp -d)
cp -R "$STAGE/." "$CLEAN/"
( cd "$CLEAN" && tectonic -X compile paper.tex --keep-logs >/dev/null 2>&1 )
PAGES=$(grep -oE "Output written on paper.xdv \([0-9]+ pages" "$CLEAN/paper.log" | grep -oE "[0-9]+")
UNDEF=$(grep -ciE "undefined (control sequence|reference)|multiply.defined" "$CLEAN/paper.log" || true)
echo "isolated build: ${PAGES} pages, ${UNDEF} undefined-reference warnings"
[ "$UNDEF" = "0" ] || { echo "FAIL: unresolved references in the isolated build"; exit 1; }
cp "$CLEAN/paper.pdf" "$OUT/paper.pdf"
rm -rf "$CLEAN"

( cd "$STAGE" && tar --disable-copyfile -czf ../arxiv-submission.tar.gz . )
echo "wrote $OUT/arxiv-submission.tar.gz ($(du -h "$OUT/arxiv-submission.tar.gz" | cut -f1))"
tar -tzf "$OUT/arxiv-submission.tar.gz" | sed 's/^/  /'
