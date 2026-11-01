"""Download the transcripts (manifest JSON only, no audio or images) of the Afrivoice Kinyarwanda parts the classifier
has not used yet: education, scripted_education and government (all splits), plus the validation/test splits of
agriculture, financial and health. The dataset is gated (CC-BY-4.0): first accept its terms while signed in at
https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_Kinyarwanda, and create a read token at
https://huggingface.co/settings/tokens.

    python download_afrivoice.py        # asks for the token (hidden input; kept only in this process)

Files go to inputs/external/afrivoice_kin/<config>_<split>/manifest_*.json, which build_domain_data.py --v4 reads.
"""
import getpass, os, shutil, sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "DigitalUmuganda/Afrivoice_Kinyarwanda"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "external", "afrivoice_kin")
WANT = [("education", ("train", "val", "test")), ("scripted_education", ("train", "val", "test")),
        ("government", ("train", "val", "test")), ("agriculture", ("val", "test")), ("financial", ("val", "test")),
        ("health", ("val", "test"))]


def main():
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Paste your Hugging Face read token (input is hidden), then Enter: ").strip()
    files = HfApi().list_repo_files(REPO, repo_type="dataset", token=token)
    todo = [f for f in files if f.endswith(".json") and "manifest" in f
            and any(f.startswith(f"{c}_{s}_tarred/") for c, ss in WANT for s in ss)]
    if not todo:
        sys.exit("no manifest files found: check that the token's account has accepted the dataset terms")
    print(len(todo), "manifest files to download")
    for i, f in enumerate(todo, 1):
        cfg = f.split("/")[0].replace("_tarred", "")
        dest = os.path.join(OUT, cfg)
        os.makedirs(dest, exist_ok=True)
        p = hf_hub_download(REPO, f, repo_type="dataset", token=token, local_dir=os.path.join(OUT, "_hub"))
        os.replace(p, os.path.join(dest, os.path.basename(f)))
        print(f"  {i}/{len(todo)} {cfg}/{os.path.basename(f)}")
    shutil.rmtree(os.path.join(OUT, "_hub"), ignore_errors=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
