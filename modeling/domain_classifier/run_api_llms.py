"""API LLM baselines (option A): Gemini 4 Argon and GPT-6.1 Sol, zero-shot, on the 3,616 human-labelled texts.

    python run_api_llms.py

1. asks for the two API keys (hidden input; kept only in this process, never written to disk)
2. pilot: 50 texts per model, then prints the measured cost and the projected cost of the full run
3. full run only if you type y (the 50 pilot answers are reused, not paid for twice)
Results: results/llm/<model>_0shot/ (test_predictions.csv, run_info.json with tokens and USD)
"""
import getpass, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("gemini", "gemini-4-argon", "results/llm/gemini4argon_0shot"),
        ("openai", "gpt-6.1-sol", "results/llm/gpt61sol_0shot")]
PRICE = ["2", "10"]   # USD per 1M input / output tokens for both models (official pricing pages, Oct 2026)
FULL = 3616


def run(backend, model, out, limit=None):
    args = [sys.executable, "llm_baseline_domain.py", "--backend", backend, "--model", model, "--shots", "0", "--price", *PRICE, "--out", out]
    return subprocess.run(args + (["--limit", str(limit)] if limit else []), cwd=HERE).returncode


def main():
    for var, label in (("GEMINI_API_KEY", "Gemini (Google AI Studio)"), ("OPENAI_API_KEY", "OpenAI")):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"Paste your {label} API key (input is hidden), then Enter: ").strip()
    print("\n=== Pilot: 50 texts per model ===")
    for backend, model, out in RUNS:
        if run(backend, model, out, limit=50) != 0:
            sys.exit(f"Pilot for {model} failed (see the message above). Nothing more was sent.")
    total = 0.0
    for _, model, out in RUNS:
        i = json.load(open(os.path.join(HERE, out, "run_info.json")))
        usd = i.get("usd") or 0.0
        proj = usd / max(1, i["rows"]) * FULL
        total += proj
        print(f"{model}: pilot {i['rows']} texts cost USD {usd:.3f}, unreadable answers {i['parse_failures']}, "
              f"request {i.get('request_variant')} -> all {FULL:,} texts about USD {proj:.2f}")
    if input(f"\nRun the full test on all {FULL:,} texts with both models (about USD {total:.2f} in total)? (y/n) ").strip().lower() != "y":
        print("Stopped after the pilot."); return
    for backend, model, out in RUNS:
        run(backend, model, out)
    print("\nDone. Results in results/llm/. You can close this tab.")


if __name__ == "__main__":
    main()
