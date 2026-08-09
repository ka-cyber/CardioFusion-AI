"""
Minimal inference API for CardioFusion-AI fusion models.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000

IMPORTANT: this repo does not ship a trained model checkpoint (see
PRODUCTION_READINESS.md -- no real cardiovascular-risk dataset has been
trained on yet). On startup, this API tries to load a checkpoint from
CARDIOFUSION_MODEL_PATH (or ./checkpoints/best_model.pt). If none is found,
the API still starts (so /health works for container orchestration) but
/predict returns HTTP 503 with a clear message rather than silently
returning meaningless output from an untrained/random-initialized network.
That's a deliberate design choice: a fake-looking prediction from an
untrained model is worse than an honest "not ready" response.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.logging_config import configure_logging
from utils.exceptions import ModelNotLoadedError

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("CARDIOFUSION_MODEL_PATH", "checkpoints/best_model.pt")
EXPECTED_WINDOW_SIZE = int(os.environ.get("CARDIOFUSION_WINDOW_SIZE", "1000"))

_model_state = {"model": None, "loaded_from": None}


def _try_load_model() -> None:
    """Attempt to load a trained fusion model checkpoint. Failure is logged,
    not raised -- the API should still start so orchestration health checks work."""
    if not os.path.exists(MODEL_PATH):
        log.warning(
            f"No model checkpoint found at '{MODEL_PATH}'. /predict will return 503 "
            "until CARDIOFUSION_MODEL_PATH points to a real trained checkpoint."
        )
        return
    try:
        import torch  # local import: torch is a heavy, optional-at-import-time dependency
        from models.fusion.fusion_models import FeatureLevelFusion
        from models.cnn.cnn_models import CNN1D

        # NOTE: this assumes a FeatureLevelFusion checkpoint with default
        # dims. A real deployment should save/load architecture config
        # alongside weights (e.g. in the same checkpoint dict) rather than
        # hardcoding it here -- flagged as a TODO, not silently papered over.
        ecg_encoder = CNN1D(in_channels=1, embedding_dim=128)
        ppg_encoder = CNN1D(in_channels=1, embedding_dim=128)
        model = FeatureLevelFusion(ecg_encoder, ppg_encoder, embedding_dim=128, num_classes=2)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        _model_state["model"] = model
        _model_state["loaded_from"] = MODEL_PATH
        log.info(f"Loaded model checkpoint from '{MODEL_PATH}'")
    except Exception as e:
        log.error(f"Found checkpoint at '{MODEL_PATH}' but failed to load it: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="CardioFusion-AI Inference API",
    description="Serves ECG+PPG fusion model predictions. See module docstring "
                 "for the honest story on model availability.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    ecg_window: list[float] = Field(..., description=f"Preprocessed ECG window, {EXPECTED_WINDOW_SIZE} samples.")
    ppg_window: list[float] = Field(..., description=f"Preprocessed, time-aligned PPG window, same length.")

    class Config:
        json_schema_extra = {
            "example": {
                "ecg_window": [0.0] * EXPECTED_WINDOW_SIZE,
                "ppg_window": [0.0] * EXPECTED_WINDOW_SIZE,
            }
        }


class PredictResponse(BaseModel):
    risk_class: int
    risk_probability: float
    model_checkpoint: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_checkpoint: Optional[str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe. Returns 200 even if no model is loaded --
    the process is alive; whether it can actually serve predictions is a
    separate, explicit field, not conflated with process health."""
    return HealthResponse(
        status="ok",
        model_loaded=_model_state["model"] is not None,
        model_checkpoint=_model_state["loaded_from"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint is loaded. This is expected until a real "
                "checkpoint (trained on a real, labeled cardiovascular-risk dataset -- "
                "not included in this repo) is placed at CARDIOFUSION_MODEL_PATH. "
                "See PRODUCTION_READINESS.md."
            ),
        )

    if len(request.ecg_window) != len(request.ppg_window):
        raise HTTPException(status_code=422, detail="ecg_window and ppg_window must be the same length.")
    if len(request.ecg_window) != EXPECTED_WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected windows of length {EXPECTED_WINDOW_SIZE}, got {len(request.ecg_window)}.",
        )

    try:
        import torch
        model = _model_state["model"]
        ecg_t = torch.tensor(request.ecg_window, dtype=torch.float32).view(1, 1, -1)
        ppg_t = torch.tensor(request.ppg_window, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            logits = model(ecg_t, ppg_t)
            probs = torch.softmax(logits, dim=-1)[0]
            risk_class = int(torch.argmax(probs).item())
            risk_probability = float(probs[1].item())  # P(high-risk class)
    except Exception as e:
        log.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        risk_class=risk_class,
        risk_probability=risk_probability,
        model_checkpoint=_model_state["loaded_from"],
    )
