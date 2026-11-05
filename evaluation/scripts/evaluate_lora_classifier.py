#!/usr/bin/env python3
"""Evaluate a saved LoRA sequence classifier without training or tuning it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    text_column = config.get("text_column", "text")
    domain_column = config.get("domain_column", "domain")
    source_column = config.get("source_column", "source")
    split_column = config.get("split_column", "Split")
    evaluation_split = config.get("evaluation_split")
    evaluation_name = config.get("evaluation_name", "External evaluation")
    result_prefix = config.get("result_prefix", "external")
    labels = list(config["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    missing = {text_column, domain_column}.difference(data.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {', '.join(sorted(missing))}")
    unknown = set(data[domain_column]).difference(labels)
    if unknown:
        raise ValueError(f"Evaluation data has unknown labels: {sorted(unknown)}")
    if evaluation_split:
        if split_column not in data.columns:
            raise ValueError(f"Data is missing split column: {split_column}")
        data = data[data[split_column] == evaluation_split].copy()
        if data.empty:
            raise ValueError(f"No rows found for evaluation split: {evaluation_split}")

    base_model_path = str(resolve(config["base_model_path"]))
    adapter_path = str(resolve(config["adapter_path"]))
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = int(config.get("batch_size", 8))
    max_length = int(config.get("max_length", 128))
    predicted_ids: list[int] = []
    texts = data[text_column].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            predicted_ids.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    predicted = [id_to_label[index] for index in predicted_ids]

    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "evaluation_rows": len(data),
        "accuracy": accuracy_score(data[domain_column], predicted),
        "macro_f1": f1_score(data[domain_column], predicted, average="macro", zero_division=0),
        "classification_report": classification_report(
            data[domain_column], predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }
    (output_dir / f"{result_prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = data.copy()
    predictions["Predicted_Domain"] = predicted
    predictions.to_csv(output_dir / f"{result_prefix}_predictions.csv", index=False)
    if source_column in predictions.columns:
        rows = [
            {
                "source": source,
                "evaluation_rows": len(group),
                "macro_f1": f1_score(group[domain_column], group["Predicted_Domain"], average="macro", zero_division=0),
            }
            for source, group in predictions.groupby(source_column)
        ]
        pd.DataFrame(rows).to_csv(output_dir / f"{result_prefix}_metrics_by_source.csv", index=False)
    print(f"Device: {device}")
    print(f"{evaluation_name} macro-F1: {metrics['macro_f1']:.4f}")
    print(f"{evaluation_name} accuracy: {metrics['accuracy']:.4f}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
