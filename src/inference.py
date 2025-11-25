import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from . import config, features
from .models import CRNN, SERTransformer

# Map class indices back to readable labels (must match EMOTION_MAP in dataset.py)
ID2LABEL = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgust",
}


def load_model(model_type: str | None = None,
               device: str | None = None,
               checkpoint_path: str | Path | None = None):
    """
    Load a trained CRNN or Transformer model with saved weights.

    Returns:
        model      : torch.nn.Module in eval() mode
        device     : 'cuda' or 'cpu'
        model_type : 'crnn' or 'transformer'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use config.MODEL_TYPE as default
    if model_type is None:
        model_type = getattr(config, "MODEL_TYPE", "crnn")
    model_type = model_type.lower()

    # Project root (one level above src/)
    base_dir = Path(__file__).resolve().parents[1]

    if model_type == "crnn":
        model = CRNN()
        ckpt = checkpoint_path or (base_dir / "best_model_crnn.pth")
    else:
        model = SERTransformer()
        ckpt = checkpoint_path or (base_dir / "best_model_transformer.pth")

    ckpt = Path(ckpt)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, device, model_type


def predict_file(path: str | Path,
                 model: torch.nn.Module | None = None,
                 device: str | None = None,
                 model_type: str | None = None):
    """
    Run emotion prediction on a single audio file.

    Args:
        path       : path to an audio file (WAV, etc.)
        model      : optional pre-loaded model (if None, will load from disk)
        device     : 'cuda' or 'cpu' (if None, auto-detect)
        model_type : 'crnn' or 'transformer' (if None, use config.MODEL_TYPE)

    Returns:
        pred_label : string label, e.g. "happy"
        pred_idx   : integer class index (0..5)
        probs_np   : numpy array of shape (NUM_CLASSES,) with probabilities
    """
    path = Path(path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model if not provided
    if model is None:
        model, device, model_type = load_model(
            model_type=model_type,
            device=device
        )

    # Extract log-Mel features using the same pipeline as training
    mel = features.extract_features_from_path(str(path))   # (NUM_MEL, T)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, NUM_MEL, T)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)               # (1, NUM_CLASSES)
        probs = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)
        pred_idx = int(torch.argmax(probs).item())

    pred_label = ID2LABEL.get(pred_idx, str(pred_idx))
    probs_np = probs.cpu().numpy()

    return pred_label, pred_idx, probs_np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SER inference on a single audio file."
    )
    parser.add_argument("audio_path", help="Path to an audio file (e.g. WAV)")
    parser.add_argument(
        "--model_type",
        choices=["crnn", "transformer"],
        default=getattr(config, "MODEL_TYPE", "crnn"),
        help="Which model to use for inference",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to checkpoint (.pth). If not set, uses best_model_*.pth in project root.",
    )
    args = parser.parse_args()

    # Load model
    model, device, model_type = load_model(
        model_type=args.model_type,
        device=None,
        checkpoint_path=args.checkpoint,
    )

    # Predict
    label, idx, probs = predict_file(
        args.audio_path,
        model=model,
        device=device,
        model_type=model_type,
    )

    print(f"Device      : {device}")
    print(f"Model type  : {model_type}")
    print(f"Audio file  : {args.audio_path}")
    print(f"Predicted   : {label} (class {idx})")
    print(f"Probabilities: {probs}")
