#!/usr/bin/env python3
"""Clean a tagged text corpus and write one train/validation data table.

The script is deliberately language-agnostic.  Language-specific paths, allowed
domains, column names, and split behaviour belong in a YAML configuration file.
It never reads from ``evaluation/`` and never creates domain labels: a supplied
corpus must already contain them.  The model-ready output retains a ``Split``
column rather than creating separate train and validation files.

Example
-------
python data/preprocessing/preprocess.py --config data/preprocessing/configs/yoruba.yaml
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHITESPACE = re.compile(r"\s+")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def project_path(value: str) -> Path:
    """Resolve configuration paths relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def open_csv(path: Path, mode: str) -> TextIO:
    """Open normal or gzip-compressed UTF-8 CSV files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def clean_text(value: str) -> str:
    """Normalize Unicode and spacing without removing Yoruba diacritics."""
    text = unicodedata.normalize("NFC", value or "")
    text = CONTROL_CHARS.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def canonical_text(value: str) -> str:
    """Create a stable key for duplicate detection; retain original text on output."""
    return clean_text(value).casefold()


def normalise_domain(value: str, aliases: dict[str, str]) -> str:
    key = clean_text(value).casefold()
    return aliases.get(key, clean_text(value))


def normalise_split(value: str, aliases: dict[str, str]) -> str:
    return aliases.get(clean_text(value).casefold(), "")


def word_count(text: str) -> int:
    return len(text.split())


def derived_split(group_id: str, seed: str, validation_fraction: float, test_fraction: float) -> str:
    """Assign a stable split from a document ID, preventing document leakage."""
    digest = hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    return "validation" if value < test_fraction + validation_fraction else "train"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    return config


def assign_exact_stratified_splits(
    path: Path, domain_column: str, split_column: str, seed: str, validation_fraction: float, test_fraction: float
) -> Counter:
    """Rewrite a cleaned output with exact, deterministic domain-level split sizes.

    This optional mode is useful for balanced experiments where each domain must
    contribute the same train/validation/test counts. It is deliberately opt-in
    because it holds the accepted rows in memory.
    """
    with open_csv(path, "rt") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        groups: dict[str, list[dict]] = {}
        for row in reader:
            groups.setdefault(row[domain_column], []).append(row)

    counts = Counter()
    rows: list[dict] = []
    for domain, group in groups.items():
        group.sort(key=lambda row: hashlib.sha1(f"{seed}:{row.get('Preprocess_Text_SHA1', '')}".encode("utf-8")).hexdigest())
        test_count = round(len(group) * test_fraction)
        validation_count = round(len(group) * validation_fraction)
        for index, row in enumerate(group):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            row[split_column] = split
            counts[split] += 1
            rows.append(row)

    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with open_csv(temporary_path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)
    return counts


def require(config: dict, key: str):
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key}")
    return config[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration, relative to the project root or absolute")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = load_config(config_path)
    input_path = project_path(require(config, "input_path"))
    cleaned_path = project_path(require(config, "cleaned_output_path"))
    data_path = project_path(require(config, "data_output_path"))
    report_path = project_path(require(config, "report_output_path"))

    text_column = config.get("text_column", "Text")
    domain_column = config.get("domain_column", "Domain")
    split_column = config.get("split_column", "Split (train/dev)")
    document_id_column = config.get("document_id_column", "Document_ID")
    required_columns = set(config.get("required_columns", ["ID", "Language", "Source", "License", text_column, domain_column]))
    required_values = config.get("require_nonempty_columns", ["ID", "Language", "Source", "License"])
    allowed_domains = set(config.get("allowed_domains", []))
    if not allowed_domains:
        raise ValueError("allowed_domains must contain the domains permitted in training data.")
    domain_aliases = {str(key).casefold(): value for key, value in config.get("domain_aliases", {}).items()}
    split_aliases = {str(key).casefold(): value for key, value in config.get("split_aliases", {}).items()}
    min_words = int(config.get("min_words", 3))
    max_words = int(config.get("max_words", 120))
    validation_fraction = float(config.get("validation_fraction", 0.15))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = str(config.get("split_seed", 13))
    exact_stratified_split = bool(config.get("exact_stratified_split", False))
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be non-negative, with validation_fraction > 0 and a total below 1.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    stats = Counter()
    domain_counts = Counter()
    split_counts = Counter()
    seen_text = set()

    with open_csv(input_path, "rt") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header row.")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input corpus is missing required columns: {', '.join(sorted(missing))}")
        output_columns = list(reader.fieldnames)
        if split_column not in output_columns:
            output_columns.append(split_column)
        if "Preprocess_Text_SHA1" not in output_columns:
            output_columns.append("Preprocess_Text_SHA1")

        with open_csv(cleaned_path, "wt") as clean_file, open_csv(data_path, "wt") as data_file:
            clean_writer = csv.DictWriter(clean_file, fieldnames=output_columns)
            data_writer = csv.DictWriter(data_file, fieldnames=output_columns)
            for writer in (clean_writer, data_writer):
                writer.writeheader()

            for row in reader:
                stats["input_rows"] += 1
                if any(not clean_text(row.get(column, "")) for column in required_values):
                    stats["rejected_missing_metadata"] += 1
                    continue

                text = clean_text(row.get(text_column, ""))
                words = word_count(text)
                if words < min_words:
                    stats["rejected_too_short"] += 1
                    continue
                if words > max_words:
                    stats["rejected_too_long"] += 1
                    continue

                fingerprint = hashlib.sha1(canonical_text(text).encode("utf-8")).hexdigest()
                if fingerprint in seen_text:
                    stats["rejected_duplicate_text"] += 1
                    continue
                seen_text.add(fingerprint)

                domain = normalise_domain(row.get(domain_column, ""), domain_aliases)
                if domain not in allowed_domains:
                    stats["rejected_invalid_domain"] += 1
                    continue
                split = normalise_split(row.get(split_column, ""), split_aliases)
                if split not in {"train", "validation", "test"}:
                    group_id = clean_text(row.get(document_id_column, "")) or clean_text(row.get("ID", ""))
                    split = derived_split(group_id, split_seed, validation_fraction, test_fraction)
                    stats["split_derived_from_document"] += 1
                else:
                    stats["split_preserved_from_input"] += 1

                row[text_column] = text
                row[domain_column] = domain
                row[split_column] = split
                row["Preprocess_Text_SHA1"] = fingerprint
                clean_writer.writerow(row)
                data_writer.writerow(row)
                stats["accepted_rows"] += 1
                domain_counts[domain] += 1
                split_counts[split] += 1

    if exact_stratified_split:
        split_counts = assign_exact_stratified_splits(
            cleaned_path, domain_column, split_column, split_seed, validation_fraction, test_fraction
        )
        data_split_counts = assign_exact_stratified_splits(
            data_path, domain_column, split_column, split_seed, validation_fraction, test_fraction
        )
        if split_counts != data_split_counts:
            raise ValueError("Cleaned and model-ready outputs have inconsistent exact split counts.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "outputs": {
            "cleaned": str(cleaned_path.relative_to(PROJECT_ROOT)),
            "data": str(data_path.relative_to(PROJECT_ROOT)),
        },
        "filters": {
            "min_words": min_words,
            "max_words": max_words,
            "allowed_domains": sorted(allowed_domains),
            "split_method": f"existing {split_column} when present; otherwise SHA-1 document split using {document_id_column}",
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "exact_stratified_split": exact_stratified_split,
            "split_seed": split_seed,
        },
        "counts": dict(stats),
        "accepted_rows_by_domain": dict(sorted(domain_counts.items())),
        "accepted_rows_by_split": dict(sorted(split_counts.items())),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Accepted {stats['accepted_rows']:,} of {stats['input_rows']:,} input rows.")
    print(
        f"Data split counts — train: {split_counts['train']:,}; "
        f"validation: {split_counts['validation']:,}; test: {split_counts['test']:,}."
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
