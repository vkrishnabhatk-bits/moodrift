import time
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

app = FastAPI(
    title="MooDrift REST API",
    version="1.0.0",
    description="Sentiment analysis inference using ONNX Runtime."
)

tokenizer = None
ort_session = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "artifacts", "model")
ONNX_PATH = os.path.join(BASE_DIR, "data", "artifacts", "model.onnx")

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
    global tokenizer, ort_session
    model_source = MODEL_DIR if os.path.exists(MODEL_DIR) else "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    if os.path.exists(ONNX_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        ort_session = ort.InferenceSession(ONNX_PATH, sess_options=opts)
    else:
        print(f"Warning: ONNX file not found at {ONNX_PATH}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ReviewRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded.")

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