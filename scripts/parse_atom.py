#!/usr/bin/env python3
"""Compact listing of arXiv Atom XML: id | date | title (one line per entry).

Usage: python3 scripts/parse_atom.py FILE... [--abs] [--diff PREV_SNAPSHOT]
  --abs        also print each entry's abstract (for close reading)
  --diff PREV  print only entries whose "id | title" line is absent from the
               previous snapshot file (gate re-scan scoop check)
"""
import sys
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def main(paths: list[str], with_abs: bool, prev_snapshot: str | None) -> None:
    seen: set[str] = set()
    if prev_snapshot:
        for line in open(prev_snapshot, errors="replace"):
            parts = line.split("|")
            if len(parts) >= 3 and parts[0].strip():
                seen.add(parts[0].strip())
    n_new = 0
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:  # noqa: BLE001 — report and continue scan
            print(f"## {path}: PARSE FAIL {exc}")
            continue
        entries = root.findall("a:entry", NS)
        if not prev_snapshot:
            print(f"\n## {path} ({len(entries)} entries)")
        for e in entries:
            aid = (e.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1]
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            date = (e.findtext("a:published", "", NS) or "")[:10]
            if prev_snapshot:
                if aid and aid not in seen:
                    seen.add(aid)
                    n_new += 1
                    print(f"NEW {aid} | {date} | {title[:120]}")
                continue
            print(f"{aid} | {date} | {title[:130]}")
            if with_abs:
                abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
                print(f"    {abstract}\n")
    if prev_snapshot:
        print(f"# diff complete: {n_new} new entries vs {prev_snapshot}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    prev = None
    if "--diff" in argv:
        i = argv.index("--diff")
        prev = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    files = [a for a in argv if a != "--abs"]
    main(files, "--abs" in argv, prev)
