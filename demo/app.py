"""Local Streamlit demo for the trained five-domain AfroXLMR classifiers.

Run from the repository root:
    streamlit run demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Agriculture", "Finance", "Health", "Law", "Telecommunications"]
MODEL_PATHS = {
    "Yoruba": PROJECT_ROOT / "modeling/domain_classification/yoruba/outputs/afroxlmr_lora/best_adapter",
    "Kinyarwanda": PROJECT_ROOT / "modeling/domain_classification/kinyarwanda/outputs/afroxlmr_lora/best_adapter",
    "Nigerian Pidgin": PROJECT_ROOT / "modeling/domain_classification/pidgin/outputs/afroxlmr_lora/best_adapter",
    "Swahili": PROJECT_ROOT / "modeling/domain_classification/swahili/outputs/afroxlmr_lora/best_adapter",
}
LOCAL_BASE_MODEL = PROJECT_ROOT.parent / "PnC/models/afro-xlmr-base"

EXAMPLE_TEXTS = {
    "Yoruba": "Awọn agbẹ gbọdọ lo irugbin to dara ati ajile lati mu ikore wọn pọ si.",
    "Kinyarwanda": "Abahinzi barasabwa gukoresha imbuto nziza n'ifumbire kugira ngo umusaruro wiyongere.",
    "Nigerian Pidgin": "Farmers fit use better seed and fertilizer make their harvest increase.",
    "Swahili": "Wakulima wanashauriwa kutumia mbegu bora na mbolea ili kuongeza mavuno.",
}


@st.cache_resource(show_spinner="Loading the selected AfroXLMR adapter…")
def load_model(language: str):
    """Load one language adapter once per Streamlit process."""
    adapter_path = MODEL_PATHS[language]
    if not adapter_path.exists():
        raise FileNotFoundError(f"Saved adapter not found: {adapter_path}")
    if not LOCAL_BASE_MODEL.exists():
        raise FileNotFoundError(
            "Local AfroXLMR base model not found at "
            f"{LOCAL_BASE_MODEL}. Restore ../PnC/models/afro-xlmr-base first."
        )
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_BASE_MODEL,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify(text: str, language: str) -> list[tuple[str, float]]:
    tokenizer, model, device = load_model(language)
    encoded = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits[0], dim=-1).cpu().tolist()
    return sorted(zip(LABELS, probabilities), key=lambda item: item[1], reverse=True)


st.set_page_config(page_title="Intron Domain Classifier", page_icon="🧭", layout="centered")
st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 700; margin-bottom: 0;}
    .sub-title {color: #5d6d7e; margin-top: .2rem; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="main-title">🧭 Multilingual Domain Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AfroXLMR-LoRA demo · Agriculture · Finance · Health · Law · Telecommunications</p>',
    unsafe_allow_html=True,
)

language = st.selectbox("Language model", list(MODEL_PATHS))
left, right = st.columns([4, 1])
with left:
    st.caption("Paste a sentence or short paragraph in the selected language.")
with right:
    if st.button("Use example", use_container_width=True):
        st.session_state["demo_text"] = EXAMPLE_TEXTS[language]

text = st.text_area(
    "Text to classify",
    key="demo_text",
    height=150,
    placeholder="Enter text here…",
)

if st.button("Classify text", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text before classifying.")
    else:
        try:
            scores = classify(text.strip(), language)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            st.error(str(error))
        else:
            predicted_label, confidence = scores[0]
            st.success(f"Predicted domain: **{predicted_label}**")
            first, second = st.columns(2)
            first.metric("Model confidence", f"{confidence:.1%}")
            second.metric("Language", language)
            st.caption("Confidence is the model's relative preference among the five supported domains; it is not a guarantee of correctness.")
            st.subheader("All domain scores")
            for label, score in scores:
                st.write(f"**{label}** — {score:.1%}")
                st.progress(score)

with st.expander("About this demonstration"):
    st.markdown(
        """
        This interface uses the saved **AfroXLMR + LoRA** adapters produced by
        the project. It is a closed five-domain classifier, so it always selects
        one of the displayed domains. Text outside those domains should not be
        interpreted as a reliable prediction until an explicit `Other` class is
        added and evaluated.
        """
    )
