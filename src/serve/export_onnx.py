import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LOCAL_MODEL_DIR = "data/artifacts/model"
FALLBACK_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_OUTPUT_PATH = "data/artifacts/model.onnx"

def export_model():
    # Verify that local directory exists AND contains a model config
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        print(f"Loading model from local path: '{LOCAL_MODEL_DIR}'")
        model_source = LOCAL_MODEL_DIR
    else:
        print(f"Valid local model config not found in '{LOCAL_MODEL_DIR}'. Using fallback base model: '{FALLBACK_MODEL_NAME}'")
        model_source = FALLBACK_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()

    # Save tokenizer files locally to data/artifacts/model for app.py
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)

    dummy_text = "Great product, highly recommended!"
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )

    print(f"Exporting model to {ONNX_OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    print(f"ONNX export completed: {ONNX_OUTPUT_PATH}")

if __name__ == "__main__":
    export_model()