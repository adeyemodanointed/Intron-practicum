#!/usr/bin/env python3
"""Create a deterministic, balanced sample from a labelled domain corpus.

The sampler streams a CSV or CSV.GZ file, so it can work with large corpora
without loading the full dataset into memory.  It retains the rows with the
lowest stable SHA-1 priorities for each domain; repeated runs therefore select
the same rows.  It does not create labels or read evaluation data.

Example
-------
python data/preprocessing/sample_balanced_domains.py \
  --input data/swa_domain_corpus.csv.gz \
  --output data/preprocessing/swahili/raw/swa_10000.csv.gz \
  --per-domain 2000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import logging
from collections import Counter
from pathlib import Path


DOMAINS = ("Agriculture", "Finance", "Health", "Law", "Telecommunications")


def open_csv(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def text_key(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def priority(seed: str, domain: str, text: str) -> int:
    return int(hashlib.sha1(f"{seed}|{domain}|{text_key(text)}".encode("utf-8")).hexdigest(), 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-domain", type=int, default=2000)
    parser.add_argument("--seed", default="13")
    parser.add_argument("--language", default="Swahili")
    parser.add_argument("--text-column", default="Text")
    parser.add_argument("--domain-column", default="Domain")
    args = parser.parse_args()

    if args.per_domain < 1:
        raise ValueError("--per-domain must be positive")

    heaps: dict[str, list[tuple[int, int, dict[str, str]]]] = {domain: [] for domain in DOMAINS}
    available = Counter()
    row_number = 0
    kept_counter = 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    with open_csv(args.input, "rt") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Input file has no CSV header.")
        missing = {"Language", args.text_column, args.domain_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=1):
            if row.get("Language", "").strip().casefold() != args.language.casefold():
                continue
            domain = row.get(args.domain_column, "").strip()
            text = row.get(args.text_column, "").strip()
            if domain not in heaps or not text:
                continue
            available[domain] += 1
            score = priority(args.seed, domain, text)
            # A min-heap over negative scores keeps the largest selected score at
            # index zero, making replacement by a smaller score straightforward.
            item = (-score, kept_counter, row)
            kept_counter += 1
            heap = heaps[domain]
            if len(heap) < args.per_domain:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
            if row_number % 250_000 == 0:
                logging.info("Scanned %s rows; eligible rows by domain: %s", f"{row_number:,}", dict(available))

    short = {domain: args.per_domain - len(heap) for domain, heap in heaps.items() if len(heap) < args.per_domain}
    if short:
        raise ValueError(f"Not enough eligible rows for requested sample: {short}. Available: {dict(available)}")

    output_rows: list[dict[str, str]] = []
    for domain in DOMAINS:
        # Sort by the stable priority to make the output order reproducible too.
        for _, _, row in sorted(heaps[domain], key=lambda item: -item[0]):
            item = dict(row)
            item[args.domain_column] = domain
            item["ID"] = hashlib.sha1(text_key(item[args.text_column]).encode("utf-8")).hexdigest()
            output_rows.append(item)

    # Use a stable, mixed order rather than domain blocks.
    output_rows.sort(key=lambda row: priority(args.seed, row[args.domain_column], row[args.text_column]))
    fieldnames = ["ID"] + [name for name in reader.fieldnames if name != "ID"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(args.output, "wt") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    logging.info("Wrote %s rows to %s", f"{len(output_rows):,}", args.output)
    logging.info("Selected rows per domain: %s", {domain: len(heaps[domain]) for domain in DOMAINS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
