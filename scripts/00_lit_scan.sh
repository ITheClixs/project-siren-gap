#!/bin/bash
# 00_lit_scan.sh — literature scan against the arXiv API (Atom).
# Run at every Gate (G0..G8) per docs/LIT_QUERIES.md. Idempotent; output to .lit_cache/.
# Usage: bash scripts/00_lit_scan.sh [outdir]
set -u
OUT="${1:-.lit_cache}"
mkdir -p "$OUT"
API="https://export.arxiv.org/api/query"

fetch_search() { # name query [sort] [n]
  local name="$1" q="$2" sort="${3:-submittedDate}" n="${4:-30}"
  curl -sG --max-time 60 "$API" \
    --data-urlencode "search_query=$q" \
    --data-urlencode "start=0" \
    --data-urlencode "max_results=$n" \
    --data-urlencode "sortBy=$sort" \
    --data-urlencode "sortOrder=descending" \
    -o "$OUT/$name.xml" || echo "FETCH FAIL: $name" >&2
  sleep 3  # arXiv API politeness
}

fetch_ids() { # name comma-separated-ids
  local name="$1" ids="$2"
  curl -sG --max-time 60 "$API" \
    --data-urlencode "id_list=$ids" \
    --data-urlencode "max_results=40" \
    -o "$OUT/$name.xml" || echo "FETCH FAIL: $name" >&2
  sleep 3
}

# --- Appendix D ID verification (verify before first citation) ---
fetch_ids verify1 "2006.09661,2003.08934,2201.12204,2302.03130,2302.05438,2312.02434,2301.05187,2301.12780,2302.14040,2305.13546,2402.05232,2312.04501,2403.12143,2209.04836,2110.06296,2310.13397,2211.06489,2110.03336,2405.20231,2402.16077"
fetch_ids verify2 "2002.11448,2110.15288,2209.14764,2303.17015,2209.12892,2312.10531,2402.04081,2406.10685,2506.13018,2102.10333,2102.13219,2106.07148,2306.12447,2409.11697,2605.08281,2601.23181"

# --- Seed query set (docs/LIT_QUERIES.md) ---
fetch_search s01_wsl            'all:"weight space learning"'                                         submittedDate 30
fetch_search s02_inr_cls        'abs:"implicit neural representation" AND abs:"classification"'      submittedDate 30
fetch_search s03_param_sym      'abs:"parameter symmetries" OR abs:"parameter space symmetry"'       submittedDate 30
fetch_search s04_canon          'abs:"canonicalization" AND cat:cs.LG'                                submittedDate 30
fetch_search s05_periodic_sym   'abs:"periodic activation" AND abs:"symmetry"'                        relevance     20
fetch_search s06_siren_sym      'abs:"SIREN" AND abs:"symmetry"'                                      relevance     20
fetch_search s07_metanets       'abs:"metanetworks" OR abs:"neural functionals"'                      submittedDate 30
fetch_search s08_ws_inr         'abs:"weight space" AND abs:"implicit neural"'                        submittedDate 20
fetch_search s09_ti_ws          'ti:"weight space"'                                                   submittedDate 40
fetch_search s10_nef_init       'abs:"neural field" AND abs:"initialization"'                         submittedDate 20
fetch_search s11_titles_a       'ti:"monomial matrix group" OR ti:"continuous canonicalization" OR ti:"probe generators"' relevance 20
fetch_search s12_titles_b       'ti:"implicit zoo" OR ti:"versatile weight space" OR ti:"weight space alignment"'         relevance 20
# bump the end date at every gate re-scan (G2: 2026-07-18)
fetch_search s13_recent_ws      'abs:"weight space" AND submittedDate:[202601010000 TO 202607180000]' submittedDate 60
fetch_search s14_recent_inr     'abs:"implicit neural representation" AND submittedDate:[202601010000 TO 202607180000]' submittedDate 60
fetch_search s15_class_signal   'abs:"class signal" OR all:"HyperINR"'                                relevance     20
fetch_search s16_identif        'abs:"identifiability" AND abs:"neural network" AND cat:cs.LG'        submittedDate 30

# fail-loudly audit: empty result sets are suspicious for queries that had hits before
# (advisor review G0, Systems 2). s05/s06 are legitimately near-empty (Gate-1 tripwires).
for f in "$OUT"/*.xml; do
  n=$(grep -ac "<entry" "$f" 2>/dev/null || echo 0)
  base=$(basename "$f" .xml)
  case "$base" in
    s05_periodic_sym|s06_siren_sym) ;;  # zero is the expected (good) outcome here
    *) [ "$n" -eq 0 ] && echo "WARNING: $base returned 0 entries — possible fetch failure or scoop-blind spot" >&2 ;;
  esac
done
echo "done: $(ls "$OUT" | wc -l | tr -d ' ') files in $OUT"
