#!/usr/bin/env python3
"""Fine-tune a multilingual sequence classifier with LoRA adapters.

The script trains only from the ``train`` split and selects checkpoints using
``validation``. It refuses to read the protected final evaluation directory and
does not inspect the ``test`` split; evaluate the chosen final model separately.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if "evaluation" in {part.casefold() for part in path.parts}:
        raise ValueError("Protected evaluation data cannot be used by this script.")
    return path


def metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    data_path = resolve(config["data_path"])
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "finetune.log", encoding="utf-8")],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-enabled PyTorch GPU is available. Install a CUDA-enabled PyTorch build before fine-tuning.")

    text_column = config.get("text_column", "text")
    domain_column = config.get("domain_column", "domain")
    split_column = config.get("split_column", "Split")
    labels = list(config["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    seed = int(config.get("random_seed", 13))
    set_seed(seed)
    random.seed(seed)

    data = pd.read_csv(data_path, keep_default_na=False)
    required = {text_column, domain_column, split_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {', '.join(sorted(missing))}")
    train = data[data[split_column] == "train"].copy()
    validation = data[data[split_column] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation splits must contain rows.")
    for frame_name, frame in {"train": train, "validation": validation}.items():
        unknown = set(frame[domain_column]).difference(labels)
        if unknown:
            raise ValueError(f"{frame_name} includes unknown labels: {sorted(unknown)}")
        frame["labels"] = frame[domain_column].map(label_to_id)

    logging.info("GPU: %s", torch.cuda.get_device_name(0))
    logging.info("Train rows: %s | validation rows: %s", f"{len(train):,}", f"{len(validation):,}")
    logging.info("Train rows by label: %s", train[domain_column].value_counts().to_dict())

    model_name = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(config.get("lora_rank", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.1)),
        target_modules=list(config.get("lora_target_modules", ["query", "value"])),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=int(config.get("max_length", 128)))

    train_set = Dataset.from_pandas(train[[text_column, "labels"]], preserve_index=False).map(tokenize, batched=True)
    validation_set = Dataset.from_pandas(validation[[text_column, "labels"]], preserve_index=False).map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    argument_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("train_batch_size", 4)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 8)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(config.get("epochs", 8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        fp16=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )
    # Transformers renamed this keyword across releases. Support either spelling
    # so the project is reproducible across the current supported releases.
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    argument_kwargs[strategy_name] = "epoch"
    arguments = TrainingArguments(**argument_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=collator,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config.get("early_stopping_patience", 2)))],
    )
    tokenizer_name = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[tokenizer_name] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    validation_metrics = trainer.evaluate()
    prediction = trainer.predict(validation_set)
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    predicted_labels = [id_to_label[int(item)] for item in predicted_ids]
    validation_output = validation[["ID", text_column, domain_column]].copy() if "ID" in validation.columns else validation[[text_column, domain_column]].copy()
    validation_output["Predicted_Domain"] = predicted_labels
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    report = classification_report(validation[domain_column], predicted_labels, labels=labels, output_dict=True, zero_division=0)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps({"metrics": validation_metrics, "classification_report": report}, indent=2), encoding="utf-8"
    )
    trainer.save_model(output_dir / "best_adapter")
    tokenizer.save_pretrained(output_dir / "best_adapter")
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "model_name_or_path": model_name,
        "labels": labels,
        "note": "The test split was intentionally not loaded or evaluated.",
    }, indent=2), encoding="utf-8")
    logging.info("Validation macro-F1: %.4f", validation_metrics["eval_macro_f1"])
    logging.info("Saved best LoRA adapter to %s", output_dir / "best_adapter")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
