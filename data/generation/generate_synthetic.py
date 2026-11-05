#!/usr/bin/env python3
"""Generate auditable Yoruba domain-text seeds using a Hugging Face chat model.

The script calls hosted inference; it does not download a model.  Set HF_TOKEN in
your shell, accept the selected model's Hugging Face access terms if prompted, and
then run this script with a YAML configuration.  It resumes an existing output
file, so interrupted runs do not discard already generated rows.

Generated text is *synthetic seed data*, not evaluation data.  Review a sample
before training, and never put the protected final evaluation rows in this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = ["ID", "Text", "Domain", "Label_Source", "Generator_Model"]
SECTION_AUDIT_COLUMNS = ["ID", "Domain", "Generation_Section", "Label_Source", "Generator_Model"]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    required = {"model", "output_path", "domains", "examples_per_domain"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration values: {', '.join(sorted(missing))}")
    return config


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_generation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_existing(output_path: Path, min_words: int, max_words: int) -> tuple[Counter, set[str], int]:
    """Count only rows usable under the configured training-length limits.

    Long source documents remain in the raw seed CSV for provenance, but they
    must not make a domain appear complete when the preprocessing pipeline will
    reject them.
    """
    counts: Counter = Counter()
    seen: set[str] = set()
    highest_id = 0
    if not output_path.exists():
        return counts, seen, highest_id
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row.get("Text", "").strip()
            domain = row.get("Domain", "").strip()
            if text and domain:
                if valid_text(text, min_words, max_words):
                    counts[domain] += 1
                seen.add(normalize(text))
            match = re.fullmatch(r"SYN-(\d+)", row.get("ID", ""))
            if match:
                highest_id = max(highest_id, int(match.group(1)))
    return counts, seen, highest_id


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def prompt(domain: str, section: str | None, amount: int, min_words: int, max_words: int) -> str:
    section_instruction = (
        f"Required sub-section: {section}. Every sentence must clearly concern this sub-section, "
        "not merely the broad domain."
        if section else
        "Required sub-section: none; vary the content across the broad domain."
    )
    return f"""Generate exactly {amount} distinct, natural Yoruba sentences for a text-domain classifier.

Domain: {domain}
{section_instruction}
Language: Yoruba written in Latin script, with correct diacritics where natural.
Length: each sentence must be between {min_words} and {max_words} words.
Length variation: deliberately include short, medium, and longer examples across
this range; do not make every sentence approximately the same length.
Content: each sentence must be clearly about {domain}; vary the subjects, settings, terminology, and sentence structure. Use ordinary prose such as a news, service, advisory, educational, or customer-information sentence. Do not translate a repeated template.
Safety: do not include personal data, URLs, markdown, headings, labels, numbering, explanations, or English-only sentences.

Return only a valid JSON array of strings. No code fence and no text outside the JSON array."""


def extract_sentences(response: str) -> list[str]:
    """Accept a JSON array, including when a provider adds surrounding whitespace."""
    decoder = json.JSONDecoder()
    start = response.find("[")
    if start < 0:
        raise ValueError("Model response contained no JSON array.")
    value, _ = decoder.raw_decode(response[start:])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Model response was not a JSON array of strings.")
    return [" ".join(item.split()) for item in value]


def valid_text(text: str, min_words: int, max_words: int) -> bool:
    words = text.split()
    # This is deliberately light filtering. A human review is still required.
    return min_words <= len(words) <= max_words and len(text) >= 20 and not re.search(r"https?://|www\.", text, re.I)


def append_rows(output_path: Path, rows: list[dict]) -> None:
    new_file = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def append_section_audit(audit_path: Path, rows: list[dict]) -> None:
    """Record sub-sections separately so legacy seed CSVs remain compatible."""
    new_file = not audit_path.exists()
    with audit_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SECTION_AUDIT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration, relative to the repository root or absolute")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and show planned work without calling Hugging Face")
    args = parser.parse_args()

    config = read_config(project_path(args.config))
    output_path = project_path(config["output_path"])
    if "evaluation" in {part.casefold() for part in output_path.parts}:
        raise ValueError("Synthetic output must never be written inside evaluation/.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_path.with_suffix(".log"))

    configured_domains = config["domains"]
    if isinstance(configured_domains, dict):
        domains = list(configured_domains)
        domain_sections = {
            domain: list(sections) for domain, sections in configured_domains.items()
        }
    elif isinstance(configured_domains, list):
        # Keep earlier simple configurations usable. New configurations should use
        # the mapping form so every generated row belongs to a controlled section.
        domains = list(configured_domains)
        domain_sections = {domain: [] for domain in domains}
    else:
        raise ValueError("domains must be a YAML list or a mapping of domain names to sub-sections.")
    target = int(config["examples_per_domain"])
    batch_size = int(config.get("batch_size", 25))
    min_words = int(config.get("min_words", 12))
    max_words = int(config.get("max_words", 35))
    max_attempts = int(config.get("max_attempts_per_batch", 4))
    pause_seconds = float(config.get("pause_seconds", 1))
    if not domains or target < 1 or batch_size < 1 or min_words > max_words:
        raise ValueError("Check domains, examples_per_domain, batch_size, and word-length settings.")
    for domain, sections in domain_sections.items():
        if not all(isinstance(section, str) and section.strip() for section in sections):
            raise ValueError(f"All sub-sections for {domain} must be non-empty strings.")

    counts, seen, next_id = load_existing(output_path, min_words, max_words)
    section_audit_path = output_path.with_name(f"{output_path.stem}_sections.csv")
    pending = {domain: max(target - counts[domain], 0) for domain in domains}
    provider = config.get("provider", "huggingface").casefold()
    if provider not in {"huggingface", "gemini"}:
        raise ValueError("provider must be either 'huggingface' or 'gemini'.")
    logger.info("Provider: %s | model: %s | output: %s", provider, config["model"], output_path)
    logger.info("Existing usable rows: %s | still needed: %s", sum(counts.values()), pending)
    if args.dry_run:
        return 0

    if provider == "huggingface":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN is not set. Set it in this PowerShell window before running the script.")
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=token)
    else:
        token = os.environ.get("GEMINI_API_KEY")
        if not token:
            raise EnvironmentError("GEMINI_API_KEY is not set. Set it in this PowerShell window before running the script.")
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=token)

    for domain in domains:
        sections = domain_sections[domain]
        section_index = counts[domain] % len(sections) if sections else 0
        while counts[domain] < target:
            needed = target - counts[domain]
            requested = min(batch_size, needed)
            section = sections[section_index % len(sections)] if sections else None
            accepted: list[str] = []
            for attempt in range(1, max_attempts + 1):
                logger.info(
                    "Requesting %s %s rows for section '%s' (attempt %s/%s).",
                    requested, domain, section or "broad-domain", attempt, max_attempts,
                )
                try:
                    if provider == "huggingface":
                        completion = client.chat_completion(
                            model=config["model"],
                            messages=[{"role": "user", "content": prompt(domain, section, requested, min_words, max_words)}],
                            temperature=float(config.get("temperature", 0.9)),
                            max_tokens=int(config.get("max_tokens", 1800)),
                        )
                        response_text = completion.choices[0].message.content or ""
                    else:
                        completion = client.models.generate_content(
                            model=config["model"],
                            contents=prompt(domain, section, requested, min_words, max_words),
                            config=types.GenerateContentConfig(
                                temperature=float(config.get("temperature", 0.9)),
                                max_output_tokens=int(config.get("max_tokens", 1800)),
                                response_mime_type="application/json",
                            ),
                        )
                        response_text = completion.text or ""
                    candidates = extract_sentences(response_text)
                except Exception as error:  # Provider errors must be logged but must not lose earlier work.
                    logger.warning("Generation request failed: %s", error)
                    error_text = str(error)
                    # Retrying cannot resolve a payment/credit error.  Stop here so
                    # the log remains clear and the user is not kept waiting.
                    if "402 Payment Required" in error_text or "no remaining credits" in error_text.casefold():
                        raise RuntimeError(
                            "Hugging Face Inference Providers has no available credits for this account. "
                            "Add credits/use another provider, then rerun; existing output is preserved."
                        ) from error
                    if "RESOURCE_EXHAUSTED" in error_text or "quota exceeded" in error_text.casefold():
                        delay = re.search(r"Please retry in ([^.]+)", error_text, re.I)
                        retry_hint = f" Retry after {delay.group(1)}." if delay else " Retry after the provider quota resets."
                        raise RuntimeError(
                            "Gemini API quota has been exhausted." + retry_hint +
                            " Rerun the same command; all rows already generated are preserved."
                        ) from error
                    if "no longer available to new users" in error_text.casefold():
                        raise RuntimeError(
                            "The configured generation model is retired for this account. "
                            "Update the model name in the YAML configuration, then rerun."
                        ) from error
                    time.sleep(pause_seconds)
                    continue
                for text in candidates:
                    normal = normalize(text)
                    if normal not in seen and valid_text(text, min_words, max_words):
                        accepted.append(text)
                        seen.add(normal)
                    if len(accepted) == needed:
                        break
                if accepted:
                    break
                logger.warning("No usable unique rows returned; retrying this batch.")

            if not accepted:
                raise RuntimeError(f"Could not obtain usable {domain} text after {max_attempts} attempts. Earlier rows were saved.")
            rows = []
            section_rows = []
            for text in accepted:
                next_id += 1
                row_id = f"SYN-{next_id:06d}"
                rows.append({
                    "ID": row_id, "Text": text, "Domain": domain,
                    "Label_Source": "synthetic_unreviewed", "Generator_Model": config["model"],
                })
                section_rows.append({
                    "ID": row_id, "Domain": domain,
                    "Generation_Section": section or "broad-domain",
                    "Label_Source": "synthetic_unreviewed", "Generator_Model": config["model"],
                })
            append_rows(output_path, rows)
            append_section_audit(section_audit_path, section_rows)
            counts[domain] += len(rows)
            section_index += 1
            logger.info(
                "Saved %s rows for %s / %s (%s/%s complete).",
                len(rows), domain, section or "broad-domain", counts[domain], target,
            )
            time.sleep(pause_seconds)
    logger.info("Completed generation. Review a sample before model training: %s", output_path)
    logger.info("Sub-section audit for newly generated rows: %s", section_audit_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EnvironmentError, FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as error:
        logging.getLogger("synthetic_generation").error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
