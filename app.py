import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, Query

from src.inference import load_model, predict_file

app = FastAPI(
    title="ENSC 424 SER Demo",
    description="Speech Emotion Recognition API (CRNN + Transformer)",
    version="0.1.0",
)

# --------------------------------------------------
# Global device + lazy model cache
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS: dict[str, torch.nn.Module] = {}


def get_model(model_type: str) -> torch.nn.Module:
    """
    Lazily load and cache models by type ('crnn' or 'transformer').
    """
    model_type = model_type.lower()
    if model_type not in MODELS:
        model, _, _ = load_model(model_type=model_type, device=DEVICE)
        MODELS[model_type] = model
        print(f"[INFO] Loaded {model_type} model on {DEVICE}")
    return MODELS[model_type]


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Speech Emotion Recognition API",
        "device": DEVICE,
        "available_models": ["crnn", "transformer"],
        "usage": "POST /predict with an audio file",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("crnn", enum=["crnn", "transformer"]),
):
    """
    Upload an audio file (e.g., WAV), choose a model, get predicted emotion.
    """

    # 1) Load model (from cache if already loaded)
    model = get_model(model_type)

    # 2) Save uploaded file to a temporary path
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name

    # 3) Run inference
    try:
        label, idx, probs = predict_file(
            tmp_path,
            model=model,
            device=DEVICE,
            model_type=model_type,
        )
    finally:
        # Always clean up the temp file
        os.remove(tmp_path)

    return {
        "model_type": model_type,
        "device": DEVICE,
        "filename": file.filename,
        "predicted_label": label,
        "predicted_index": idx,
        "class_probabilities": probs.tolist(),
    }
