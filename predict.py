import json, argparse, torch, joblib, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def load_labels(p="artifacts/labels.json"):
    with open(p) as f: return json.load(f)

def predict_transformer(texts, model_dir="artifacts/distilbert_model"):
    labels = load_labels()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pipe = TextClassificationPipeline(model=model, tokenizer=tok, device=-1, return_all_scores=True)
    out=[]
    for t in texts:
        scores = pipe(t)[0]
        best = max(scores, key=lambda x: x["score"])
        out.append({"text": t, "label": best["label"], "score": float(best["score"])})
    return out

def predict_baseline(texts, path="artifacts/baseline_tfidf_logreg.joblib"):
    labels = load_labels()
    if not os.path.exists(path):
        raise FileNotFoundError("Baseline model file not found. Train/save it first or use --engine transformer.")
    model = joblib.load(path)
    preds = model.predict(texts)
    return [{"text": t, "label": labels[p]} for t,p in zip(texts, preds)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--engine", choices=["baseline","transformer"], default="transformer")
    a = ap.parse_args()
    res = predict_transformer(a.texts) if a.engine=="transformer" else predict_baseline(a.texts)
    print(json.dumps(res, indent=2))
