import json
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.config import resolve

app = FastAPI(
    title="MooDrift REST API",
    version="1.0.0",
    description="Sentiment analysis inference using ONNX Runtime."
)

tokenizer = None
ort_session = None
model_info: dict | None = None

MODEL_DIR = resolve("data/artifacts/model")
ONNX_PATH = resolve("data/artifacts/model.onnx")
MANIFEST_PATH = resolve("data/artifacts/model_manifest.json")

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictionResponse(BaseModel):
    predicted_score: int
    probabilities: dict[str, float]
    latency_ms: float

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.on_event("startup")
def load_model_and_tokenizer():
    """Load the pre-exported artifact from disk. Never falls back to a substitute model.

    The serving container never talks to MLflow directly - resolving the registry alias
    and exporting to ONNX is `python -m src.serve.export_onnx`'s job, run ahead of time.
    If that hasn't happened, the correct behaviour is to stay unloaded (503 on /predict),
    not to quietly serve a different model with a different label space.
    """
    global tokenizer, ort_session, model_info

    config_path = MODEL_DIR / "config.json"
    if not config_path.exists() or not ONNX_PATH.exists():
        print(
            f"[serve] no exported model found at {MODEL_DIR} / {ONNX_PATH}. "
            f"Run `python -m src.serve.export_onnx` first. Staying unloaded."
        )
        return

    if MANIFEST_PATH.exists():
        model_info = json.loads(MANIFEST_PATH.read_text())
        print(f"[serve] serving {model_info['registered_model']} v{model_info['version']} "
              f"(@{model_info['alias']}, run {model_info['run_id']})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ReviewRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the export step first.")

    start_time = time.perf_counter()
    tokens = tokenizer(
        payload.text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="np"
    )

    onnx_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64)
    }

    logits = ort_session.run(None, onnx_inputs)[0]
    probs = softmax(logits)[0]
    predicted_score = int(np.argmax(probs) + 1)
    latency = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        predicted_score=predicted_score,
        probabilities={str(i + 1): round(float(p), 4) for i, p in enumerate(probs)},
        latency_ms=round(latency, 2)
    )
