#!/usr/bin/env python3
"""Compact listing of arXiv Atom XML: id | date | title (one line per entry).

Usage: python3 scripts/parse_atom.py FILE... [--abs]
  --abs  also print each entry's abstract (for close reading)
"""
import sys
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def main(paths: list[str], with_abs: bool) -> None:
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:  # noqa: BLE001 — report and continue scan
            print(f"## {path}: PARSE FAIL {exc}")
            continue
        entries = root.findall("a:entry", NS)
        print(f"\n## {path} ({len(entries)} entries)")
        for e in entries:
            aid = (e.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1]
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            date = (e.findtext("a:published", "", NS) or "")[:10]
            print(f"{aid} | {date} | {title[:130]}")
            if with_abs:
                abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
                print(f"    {abstract}\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--abs"]
    main(args, "--abs" in sys.argv)
