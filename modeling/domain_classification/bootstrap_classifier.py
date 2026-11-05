#!/usr/bin/env python3
"""Train a domain classifier and safely pseudo-label unlabelled text.

The workflow is one auditable bootstrap round:
1. train on labelled seed data (for example, generated or human-verified data);
2. score a separate unlabelled corpus;
3. save only high-confidence predictions as *unreviewed* pseudo-labels;
4. save a stratified human-audit sample.

Do not train on the pseudo-label file until reviewed rows have been copied to an
approved file.  The script refuses to read any file inside evaluation/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if "evaluation" in {part.casefold() for part in path.parts}:
        raise ValueError("Evaluation data is protected and must never be used by this script.")
    return path


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def require(config: dict, key: str):
    if key not in config:
        raise ValueError(f"Missing required configuration value: {key}")
    return config[key]


def open_output(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bootstrap_classifier")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "bootstrap.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_labelled(frame: pd.DataFrame, text_column: str, domain_column: str, allowed_domains: set[str], name: str) -> pd.DataFrame:
    missing = {text_column, domain_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame[text_column] = frame[text_column].fillna("").astype(str).str.strip()
    frame[domain_column] = frame[domain_column].fillna("").astype(str).str.strip()
    frame = frame[(frame[text_column] != "") & frame[domain_column].isin(allowed_domains)]
    if frame.empty:
        raise ValueError(f"{name} has no usable rows after validation.")
    return frame


def make_pipeline(max_features: int, random_seed: int) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=max_features)),
        ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=max_features)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)),
    ])


def audit_priority(identifier: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{identifier}".encode("utf-8")).hexdigest(), 16)


def update_audit_pool(pools: dict[str, list[tuple[int, dict]]], rows: pd.DataFrame, domain: str, count: int, id_column: str, seed: int) -> None:
    pool = pools.setdefault(domain, [])
    for _, row in rows.iterrows():
        identifier = str(row.get(id_column, row.name))
        pool.append((audit_priority(identifier, seed), row.to_dict()))
    pool.sort(key=lambda item: item[0])
    del pool[count:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration, relative to repository root or absolute")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = read_config(config_path)
    output_dir = resolve_path(require(config, "output_dir"))
    logger = setup_logger(output_dir)

    text_column = config.get("text_column", "Text")
    domain_column = config.get("domain_column", "Domain")
    id_column = config.get("id_column", "ID")
    allowed_domains = set(require(config, "allowed_domains"))
    threshold = float(config.get("confidence_threshold", 0.95))
    validation_fraction = float(config.get("validation_fraction", 0.15))
    random_seed = int(config.get("random_seed", 13))
    chunk_size = int(config.get("scoring_chunk_size", 50_000))
    audit_per_domain = int(config.get("audit_per_domain", 30))
    max_features = int(config.get("max_features", 200_000))
    if not 0 < threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1].")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    seed_path = resolve_path(require(config, "seed_data_path"))
    candidate_path = resolve_path(require(config, "candidate_data_path"))
    approved_path_value = config.get("approved_pseudo_labels_path")
    approved_path = resolve_path(approved_path_value) if approved_path_value else None
    if not seed_path.exists() or not candidate_path.exists():
        raise FileNotFoundError("seed_data_path and candidate_data_path must both exist.")

    logger.info("Starting bootstrap round; threshold=%.2f; candidate chunk size=%s", threshold, f"{chunk_size:,}")
    logger.info("Reading seed data: %s", seed_path)
    seed = validate_labelled(pd.read_csv(seed_path, keep_default_na=False), text_column, domain_column, allowed_domains, "Seed data")
    seed["Training_Label_Source"] = "seed"

    if approved_path:
        logger.info("Reading human-approved pseudo-labels: %s", approved_path)
        approved = validate_labelled(pd.read_csv(approved_path, keep_default_na=False), text_column, domain_column, allowed_domains, "Approved pseudo-labels")
        approved["Training_Label_Source"] = "approved_pseudo_label"
        training = pd.concat([seed, approved], ignore_index=True)
    else:
        training = seed

    logger.info("Usable training rows: %s | per domain: %s", f"{len(training):,}", training[domain_column].value_counts().to_dict())
    if training[domain_column].nunique() < 2:
        raise ValueError("At least two labelled domains are required to train a classifier.")

    train_rows, validation_rows = train_test_split(
        training, test_size=validation_fraction, random_state=random_seed, stratify=training[domain_column]
    )
    model = make_pipeline(max_features, random_seed)
    logger.info("Training model on %s rows; validating on %s held-out seed rows.", f"{len(train_rows):,}", f"{len(validation_rows):,}")
    model.fit(train_rows[text_column], train_rows[domain_column])
    validation_prediction = model.predict(validation_rows[text_column])
    macro_f1 = f1_score(validation_rows[domain_column], validation_prediction, average="macro", zero_division=0)
    validation_report = classification_report(validation_rows[domain_column], validation_prediction, output_dict=True, zero_division=0)
    (output_dir / "seed_validation_metrics.json").write_text(
        json.dumps({"macro_f1": macro_f1, "report": validation_report}, indent=2), encoding="utf-8"
    )
    logger.info("Seed-only validation macro-F1: %.3f. This is not final real-world evaluation.", macro_f1)

    # Refit on all approved training data before scoring the unlabelled candidate pool.
    model.fit(training[text_column], training[domain_column])
    joblib.dump(model, output_dir / "domain_classifier.joblib")

    high_confidence_path = output_dir / "pseudo_labels_unreviewed.csv.gz"
    all_predictions_path = output_dir / "candidate_predictions.csv.gz"
    audit_pools: dict[str, list[tuple[int, dict]]] = {}
    counts = {"candidate_rows": 0, "usable_candidate_rows": 0, "high_confidence_rows": 0}
    predicted_counts: dict[str, int] = {domain: 0 for domain in sorted(allowed_domains)}

    logger.info("Scoring candidate data: %s", candidate_path)
    first_prediction_write = True
    first_pseudo_label_write = True
    for chunk_number, chunk in enumerate(pd.read_csv(candidate_path, keep_default_na=False, chunksize=chunk_size), start=1):
        counts["candidate_rows"] += len(chunk)
        if text_column not in chunk.columns:
            raise ValueError(f"Candidate data is missing text column '{text_column}'.")
        chunk[text_column] = chunk[text_column].fillna("").astype(str).str.strip()
        chunk = chunk[chunk[text_column] != ""].copy()
        counts["usable_candidate_rows"] += len(chunk)
        if chunk.empty:
            continue
        probabilities = model.predict_proba(chunk[text_column])
        classes = model.named_steps["classifier"].classes_
        winners = probabilities.argmax(axis=1)
        chunk["Predicted_Domain"] = [classes[index] for index in winners]
        chunk["Confidence"] = probabilities.max(axis=1)
        chunk["Model_Version"] = output_dir.name
        chunk["Label_Quality"] = "unlabelled"
        chunk.to_csv(
            all_predictions_path,
            mode="wt" if first_prediction_write else "at",
            index=False,
            compression="gzip",
            header=first_prediction_write,
        )
        first_prediction_write = False
        selected = chunk[chunk["Confidence"] >= threshold].copy()
        if not selected.empty:
            selected[domain_column] = selected["Predicted_Domain"]
            selected["Label_Quality"] = "pseudo_label_unreviewed"
            selected.to_csv(
                high_confidence_path,
                mode="wt" if first_pseudo_label_write else "at",
                index=False,
                compression="gzip",
                header=first_pseudo_label_write,
            )
            first_pseudo_label_write = False
            counts["high_confidence_rows"] += len(selected)
            for domain, amount in selected["Predicted_Domain"].value_counts().items():
                predicted_counts[domain] = predicted_counts.get(domain, 0) + int(amount)
                update_audit_pool(audit_pools, selected[selected["Predicted_Domain"] == domain], domain, audit_per_domain, id_column, random_seed)
        logger.info("Scored chunk %s | rows=%s | high-confidence=%s", chunk_number, f"{len(chunk):,}", f"{len(selected):,}")

    audit_rows = [row for domain in sorted(audit_pools) for _, row in audit_pools[domain]]
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["Human_Review"] = ""
        audit["Human_Domain"] = ""
        audit["Review_Comment"] = ""
    audit.to_csv(output_dir / "pseudo_label_audit_sample.csv", index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "seed_data": str(seed_path.relative_to(PROJECT_ROOT)),
        "candidate_data": str(candidate_path.relative_to(PROJECT_ROOT)),
        "approved_pseudo_labels": str(approved_path.relative_to(PROJECT_ROOT)) if approved_path else None,
        "confidence_threshold": threshold,
        "seed_validation_macro_f1": macro_f1,
        "counts": counts,
        "high_confidence_by_predicted_domain": predicted_counts,
        "warning": "Pseudo-labels are unreviewed. Add only human-approved rows to the next training round.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Finished scoring %s candidate rows; retained %s high-confidence pseudo-labels.", f"{counts['candidate_rows']:,}", f"{counts['high_confidence_rows']:,}")
    logger.info("Review %s before using pseudo-labels in any later round.", output_dir / "pseudo_label_audit_sample.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        logging.getLogger("bootstrap_classifier").error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
