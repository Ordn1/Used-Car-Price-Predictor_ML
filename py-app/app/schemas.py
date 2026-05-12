from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    yr_mfr: int = Field(..., ge=1965)
    kms_run: int = Field(..., ge=0)
    fuel_type: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    times_viewed: int = Field(..., ge=0)
    body_type: str = Field(..., min_length=1)
    transmission: str = Field(..., min_length=1)


class PredictionAnalytics(BaseModel):
    carAge: str
    mileage: str
    views: str
    mileageBand: str
    demandBand: str
    estimatedRange: str
    agePenalty: str
    mileagePenalty: str
    demandBoost: str
    confidence: str
    marketPulse: str


class FeatureImportanceItem(BaseModel):
    feature: str
    label: str
    importance: float
    percentage: float


class PredictionResponse(BaseModel):
    predicted_price: float
    analytics: PredictionAnalytics
    created_at: datetime


class ServiceHealth(BaseModel):
    status: str
    model_loaded: bool
    selected_model: str | None = None


class ModelInfoResponse(BaseModel):
    selected_model: str
    feature_columns: list[str]
    raw_input_features: list[str]
    categorical_features: list[str]
    available_options: dict[str, list[str]]
    metrics: dict[str, dict[str, float]]
    feature_importance: list[FeatureImportanceItem]
    train_samples: int | None = None
    test_samples: int | None = None
    final_r2: float | None = None
    final_mae: float | None = None
    final_mse: float | None = None
