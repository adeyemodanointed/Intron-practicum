#!/usr/bin/env python3
"""Train comparable domain-classification learning-curve baselines.

Each run trains on the requested number of rows *per domain* from a fixed train
split and evaluates on a separate fixed validation split.  It never reads files
inside evaluation/, so final evaluation data remains protected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def path_from_config(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if "evaluation" in {part.casefold() for part in path.parts}:
        raise ValueError("Evaluation data is protected and cannot be used by this script.")
    return path


def model(seed: int, max_features: int) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=max_features)),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=max_features)),
        ])),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = path_from_config(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    data_path = path_from_config(config["data_path"])
    output_dir = path_from_config(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "learning_curve.log", encoding="utf-8")],
    )
    text_column = config.get("text_column", "Text")
    domain_column = config.get("domain_column", "Domain")
    split_column = config.get("split_column", "Split")
    source_column = config.get("source_column", "Source")
    domains = list(config["allowed_domains"])
    sizes = [int(size) for size in config["examples_per_domain"]]
    seed = int(config.get("random_seed", 13))
    max_features = int(config.get("max_features", 200_000))

    data = pd.read_csv(data_path, keep_default_na=False)
    needed = {text_column, domain_column, split_column}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"Data file is missing columns: {', '.join(sorted(missing))}")
    train = data[data[split_column] == "train"].copy()
    validation = data[data[split_column] == "validation"].copy()
    test = data[data[split_column] == "test"].copy()
    if validation.empty or test.empty:
        raise ValueError("Validation and test rows are required. Run preprocessing with both splits first.")
    available = train[domain_column].value_counts().to_dict()
    logging.info("Training rows per domain: %s", available)
    logging.info("Fixed validation rows per domain: %s", validation[domain_column].value_counts().to_dict())
    logging.info("Protected test rows per domain: %s", test[domain_column].value_counts().to_dict())

    results = []
    for size in sizes:
        short = {domain: int(available.get(domain, 0)) for domain in domains if available.get(domain, 0) < size}
        if short:
            raise ValueError(
                f"Cannot train {size} rows per domain with the current fixed validation split. Available: {short}. "
                "Use a smaller learning-curve size or collect more labelled data."
            )
        sampled = pd.concat([
            train[train[domain_column] == domain].sample(n=size, random_state=seed)
            for domain in domains
        ], ignore_index=True)
        classifier = model(seed, max_features)
        logging.info("Training %s rows per domain (%s total).", f"{size:,}", f"{len(sampled):,}")
        classifier.fit(sampled[text_column], sampled[domain_column])
        predicted = classifier.predict(validation[text_column])
        macro_f1 = f1_score(validation[domain_column], predicted, average="macro", zero_division=0)
        accuracy = accuracy_score(validation[domain_column], predicted)
        report = classification_report(validation[domain_column], predicted, labels=domains, output_dict=True, zero_division=0)
        (output_dir / f"metrics_{size}_per_domain.json").write_text(
            json.dumps({"macro_f1": macro_f1, "accuracy": accuracy, "report": report}, indent=2), encoding="utf-8"
        )
        joblib.dump(classifier, output_dir / f"domain_classifier_{size}_per_domain.joblib")
        result = {"examples_per_domain": size, "total_training_rows": len(sampled), "validation_rows": len(validation), "macro_f1": macro_f1, "accuracy": accuracy}
        results.append(result)
        logging.info("%s per domain | macro-F1=%.3f | accuracy=%.3f", f"{size:,}", macro_f1, accuracy)

        if source_column in validation.columns:
            source_metrics = []
            result_frame = validation[[domain_column, source_column]].copy()
            result_frame["Predicted_Domain"] = predicted
            for source, rows in result_frame.groupby(source_column):
                source_metrics.append({
                    "examples_per_domain": size,
                    "validation_source": source,
                    "validation_rows": len(rows),
                    "macro_f1": f1_score(rows[domain_column], rows["Predicted_Domain"], average="macro", zero_division=0),
                })
            pd.DataFrame(source_metrics).to_csv(output_dir / f"validation_by_source_{size}_per_domain.csv", index=False)

    pd.DataFrame(results).to_csv(output_dir / "learning_curve_metrics.csv", index=False)
    logging.info("Complete. Compare results in %s", output_dir / "learning_curve_metrics.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", flush=True)
        raise SystemExit(2)
