#!/usr/bin/env python3
"""Evaluate one already-selected scikit-learn classifier on a held-out test split.

Use this only after model selection is complete. It writes overall, per-class,
and per-source metrics plus predictions, without changing the trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    data = pd.read_csv(resolve(config["data_path"]), keep_default_na=False)
    model = joblib.load(resolve(config["model_path"]))
    text_column = config.get("text_column", "text")
    domain_column = config.get("domain_column", "domain")
    split_column = config.get("split_column", "Split")
    source_column = config.get("source_column", "source")
    labels = list(config["labels"])
    evaluate_all_rows = bool(config.get("evaluate_all_rows", False))
    if evaluate_all_rows:
        # An external manually reviewed file has no train/validation/test column:
        # every supplied row is the protected evaluation set.
        test = data.copy()
        evaluation_name = "External evaluation"
    else:
        if split_column not in data.columns:
            raise ValueError(
                f"Missing split column '{split_column}'. Set evaluate_all_rows: true "
                "when evaluating a separate external evaluation CSV."
            )
        test = data[data[split_column] == "test"].copy()
        evaluation_name = "Test"
    if test.empty:
        raise ValueError("No evaluation rows found.")
    predicted = model.predict(test[text_column])
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_rows": len(test),
        "accuracy": accuracy_score(test[domain_column], predicted),
        "macro_f1": f1_score(test[domain_column], predicted, average="macro", zero_division=0),
        "classification_report": classification_report(test[domain_column], predicted, labels=labels, output_dict=True, zero_division=0),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = test.copy()
    predictions["Predicted_Domain"] = predicted
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    if source_column in test.columns:
        rows = []
        for source, group in predictions.groupby(source_column):
            rows.append({
                "source": source,
                "test_rows": len(group),
                "macro_f1": f1_score(group[domain_column], group["Predicted_Domain"], average="macro", zero_division=0),
            })
        pd.DataFrame(rows).to_csv(output_dir / "test_metrics_by_source.csv", index=False)
    print(f"{evaluation_name} macro-F1: {metrics['macro_f1']:.4f}")
    print(f"{evaluation_name} accuracy: {metrics['accuracy']:.4f}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
