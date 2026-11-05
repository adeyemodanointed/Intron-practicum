#!/usr/bin/env python3
"""Extract one language's in-domain manual labels into a protected evaluation CSV.

Rows labelled Other, blank, or multi-label are excluded from the closed five-way
domain score. The original manual-validation file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DOMAIN = {"Agriculture", "Finance", "Health", "Law", "Telecommunications"}
WHITESPACE = re.compile(r"\s+")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_text(value: str) -> str:
    """Return a stable key used only to prevent evaluation/train overlap."""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", value or "")).strip().casefold()


def load_excluded_texts(path: Path, column: str) -> set[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Exclusion corpus {path} is missing text column '{column}'.")
        return {canonical_text(row.get(column, "")) for row in reader if canonical_text(row.get(column, ""))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Practicum - Manual Validation.csv")
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--language-column",
        default="Column 1",
        help="Name of the column containing the language label (default: Column 1).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-texts-from",
        help="Optional training corpus whose exact text matches must be excluded from this evaluation export.",
    )
    parser.add_argument("--exclude-text-column", default="Text")
    args = parser.parse_args()
    input_path, output_path = resolve(args.input), resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_texts = load_excluded_texts(resolve(args.exclude_texts_from), args.exclude_text_column) if args.exclude_texts_from else set()

    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    language_rows = [row for row in rows if row.get(args.language_column, "").strip() == args.language]
    if not language_rows:
        raise ValueError(f"No rows found for language '{args.language}' in column '{args.language_column}'.")
    selected = []
    excluded = Counter()
    for index, row in enumerate(language_rows, start=1):
        label = row.get("Manual_Domain_Pred", "").strip()
        if label not in IN_DOMAIN:
            excluded[label or "blank"] += 1
            continue
        if canonical_text(row.get("Text", "")) in excluded_texts:
            excluded["overlap_with_training"] += 1
            continue
        selected.append({
            "ID": f"manual_{args.language.casefold().replace(' ', '_')}_{index:04d}",
            "text": row.get("Text", "").strip(),
            "domain": label,
            "source": row.get("Source", "").strip(),
            "label_source": "manual_review",
        })
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "text", "domain", "source", "label_source"])
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "language": args.language,
        "language_column": args.language_column,
        "language_rows": len(language_rows),
        "exported_five_domain_rows": len(selected),
        "excluded_rows_by_manual_label": dict(sorted(excluded.items())),
        "excluded_training_corpus": str(resolve(args.exclude_texts_from).relative_to(PROJECT_ROOT)) if args.exclude_texts_from else None,
        "rows_by_domain": dict(sorted(Counter(row["domain"] for row in selected).items())),
    }
    report_path = output_path.with_name(f"{output_path.stem}_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(selected)} in-domain manual rows to {output_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
