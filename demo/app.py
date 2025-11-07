import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_DIR = "artifacts/distilbert_model"

# load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)

def predict(text: str):
    if not text or not text.strip():
        return {}, "Type some text to analyze."
    scores = pipe(text)[0]  # list of dicts: {'label': 'JOY', 'score': 0.93} (label names depend on fine-tune)
    # Map to {label: score} for the Label component
    label_scores = {s["label"]: float(s["score"]) for s in scores}
    best = max(scores, key=lambda x: x["score"])
    summary = f"Top: {best['label']} ({best['score']:.3f})"
    return label_scores, summary

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Enter text"),
    outputs=[gr.Label(num_top_classes=6, label="Emotion probabilities"),
             gr.Textbox(label="Prediction summary")],
    title="Text Emotion Classifier",
    description="Tiny BERT fine-tuned on dair-ai/emotion (6 emotions)."
)

if __name__ == "__main__":
    demo.launch()
