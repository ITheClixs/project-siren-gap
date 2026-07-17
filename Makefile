.PHONY: test lit-scan demo figures thesis

PY := .venv/bin/python

test:
	$(PY) -m pytest tests/ -q

lit-scan:
	bash scripts/00_lit_scan.sh .lit_cache
	python3 scripts/parse_atom.py .lit_cache/*.xml > docs/lit_snapshots/latest-scan-summary.txt
	@echo "snapshot written to docs/lit_snapshots/latest-scan-summary.txt — rename to G<k>- and commit"

# Placeholders until their gates (G1+); each will fail loudly rather than pretend.
demo figures thesis fit-% study-%:
	@echo "'$@' is not available until its gate — see docs/LAB_NOTEBOOK.md for current status" && exit 1
