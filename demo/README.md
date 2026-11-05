# Local presentation demo

This is a small visual interface for demonstrating the saved five-domain
AfroXLMR-LoRA classifiers. It supports Yoruba, Kinyarwanda, Nigerian Pidgin,
and Swahili.

## Run it

From the `Intron-practicum` root, activate the environment used for model
training and run:

```powershell
pip install -r requirements.txt
streamlit run demo/app.py
```

The terminal will show a local address, normally `http://localhost:8501`.
Open it in a browser before your presentation.

## Presentation flow

1. Pick **Swahili** (or another language) from the dropdown.
2. Click **Use example** or paste a short sentence.
3. Click **Classify text**.
4. Explain that the large result is the predicted domain, while the bars show
   the model's relative confidence across the five supported domains.

The confidence is not calibrated real-world certainty. This is a closed
five-domain demo: it always returns one label and does not yet reject `Other`
text.

## Model files required

The interface loads the saved adapter under
`modeling/domain_classification/<language>/outputs/afroxlmr_lora/best_adapter/`
and the local base model at `../PnC/models/afro-xlmr-base`.
