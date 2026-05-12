from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException

from .model_loader import build_model_info, load_feature_info, load_runtime_assets, predict_price
from .schemas import ModelInfoResponse, PredictionRequest, PredictionResponse, ServiceHealth


app = FastAPI(title="Used Car Python Inference Service", version="1.0.0")


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    try:
        load_runtime_assets()
        feature_info = load_feature_info()
        return ServiceHealth(
            status="ok",
            model_loaded=True,
            selected_model=str(feature_info.get("selected_model", "Unknown")),
        )
    except Exception as exc:
        return ServiceHealth(status=f"error: {exc}", model_loaded=False, selected_model=None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**build_model_info())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        predicted_price, analytics = predict_price(request)
        return PredictionResponse(
            predicted_price=predicted_price,
            analytics=analytics,
            created_at=datetime.utcnow(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
