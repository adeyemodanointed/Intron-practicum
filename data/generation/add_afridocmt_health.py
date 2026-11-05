#!/usr/bin/env python3
"""Append real Yoruba Health training documents from AfriDocMT to a seed CSV.

Only the doc_health/train split is used. Its dev and test splits are deliberately
left untouched. Run this only while generate_synthetic.py is stopped: both tools
append to the same seed CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_COLUMNS = ["ID", "Text", "Domain", "Label_Source", "Generator_Model"]
SOURCE_NAME = "afridocmt_doc_health_train"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def read_existing(seed_path: Path) -> tuple[set[str], set[str]]:
    texts: set[str] = set()
    sources: set[str] = set()
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file does not exist: {seed_path}")
    with seed_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(SEED_COLUMNS[:4]).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Seed file is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("Text", "").strip():
                texts.add(normalize(row["Text"]))
            sources.add(row.get("Label_Source", ""))
    return texts, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-path",
        default="data/generation/synthetic_text/yoruba/yoruba_synthetic_seed.csv",
        help="Existing seed CSV, relative to repository root or absolute.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect AfriDocMT but do not alter the seed file.")
    args = parser.parse_args()
    seed_path = project_path(args.seed_path)
    existing_texts, sources = read_existing(seed_path)
    if SOURCE_NAME in sources:
        print("AfriDocMT Health rows are already present; no changes made.")
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Downloading masakhane/AfriDocMT, config=doc_health, split=train")
    dataset = load_dataset("masakhane/AfriDocMT", "doc_health", split="train")
    if "yo" not in dataset.column_names:
        raise ValueError(f"Expected Yoruba column 'yo'; found: {dataset.column_names}")

    rows = []
    for index, item in enumerate(dataset, start=1):
        text = " ".join(str(item["yo"] or "").split())
        if text and normalize(text) not in existing_texts:
            rows.append({
                "ID": f"AFRIDOCMT-HEALTH-{index:04d}",
                "Text": text,
                "Domain": "Health",
                "Label_Source": SOURCE_NAME,
                "Generator_Model": "human_translation",
            })

    logging.info("Found %s usable, non-duplicate Yoruba Health documents.", len(rows))
    if args.dry_run:
        return 0
    with seed_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writerows(rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "masakhane/AfriDocMT",
        "config": "doc_health",
        "split": "train",
        "language_column": "yo",
        "assigned_domain": "Health",
        "rows_added": len(rows),
        "license": "CC-BY-NC-SA-3.0",
        "note": "Dev and test splits were not used.",
    }
    report_path = seed_path.parent / "afridocmt_health_import_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Added %s rows to %s", len(rows), seed_path)
    logging.info("Provenance report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
