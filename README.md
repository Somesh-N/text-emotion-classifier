# Text Emotion Classification
Two models on `dair-ai/emotion`:
- TF-IDF + Logistic Regression (baseline, optional)
- Tiny BERT fine-tuned (`prajjwal1/bert-tiny`) for 6 emotions

## Setup
python -m venv .venv && source .venv/bin/activate  # on Windows use .venv\Scripts\activate
pip install -r requirements.txt

## Predict
# Transformer
python predict.py --engine transformer --texts "i am thrilled!" "i'm scared of dark"
# Baseline (only if saved)
python predict.py --engine baseline --texts "i miss you" "why did you do that!"
