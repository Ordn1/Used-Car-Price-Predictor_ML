from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .schemas import PredictionAnalytics, PredictionRequest


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
MODELS_DIR = REPO_ROOT / "models"
ARTIFACT_MODEL = MODELS_DIR / "model.pkl"
ARTIFACT_SCALER = MODELS_DIR / "scaler.pkl"
ARTIFACT_ENCODERS = MODELS_DIR / "label_encoders.pkl"
ARTIFACT_FEATURES = MODELS_DIR / "feature_columns.pkl"
FEATURE_INFO_FILE = MODELS_DIR / "feature_info.json"

FEATURE_LABELS = {
    "kms_run": "Mileage",
    "fuel_type": "Fuel Type",
    "city": "City",
    "times_viewed": "Market Demand",
    "body_type": "Body Type",
    "transmission": "Transmission",
    "car_age": "Car Age",
    "kms_per_year": "Usage Intensity",
}


def format_model_name(raw_name: object) -> str:
    text = str(raw_name or "Unknown")
    replacements = {
        "GradientBoostingRegressor": "Gradient Boosting",
        "LinearRegression": "Linear Regression",
        "RandomForestRegressor": "Random Forest",
        "ExtraTreesRegressor": "Extra Trees",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def get_reference_year(feature_info: dict[str, object]) -> int:
    raw_value = feature_info.get("reference_year")
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return datetime.now().year


def get_selected_feature_columns(feature_info: dict[str, object], feature_columns: list[str]) -> list[str]:
    selected = feature_info.get("selected_features_tuned", [])
    if isinstance(selected, list) and selected and all(feature in feature_columns for feature in selected):
        return [str(feature) for feature in selected]
    return feature_columns


def get_selected_metric_key(feature_info: dict[str, object]) -> str | None:
    metrics = feature_info.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    preferred_keys = [
        "Refined: Gradient Boosting [FINAL]",
        "Baseline: Gradient Boosting",
        str(feature_info.get("final_model", "")),
        str(feature_info.get("selected_model", "")),
    ]
    for key in preferred_keys:
        if key and key in metrics:
            return key
    return next(iter(metrics.keys()), None)


@lru_cache(maxsize=1)
def load_runtime_assets() -> tuple[object, object, dict[str, object], list[str]]:
    required = [ARTIFACT_MODEL, ARTIFACT_SCALER, ARTIFACT_ENCODERS, ARTIFACT_FEATURES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing model artifacts: {missing_text}")

    return (
        joblib.load(ARTIFACT_MODEL),
        joblib.load(ARTIFACT_SCALER),
        joblib.load(ARTIFACT_ENCODERS),
        joblib.load(ARTIFACT_FEATURES),
    )


@lru_cache(maxsize=1)
def load_feature_info() -> dict[str, object]:
    if not FEATURE_INFO_FILE.exists():
        return {}
    return json.loads(FEATURE_INFO_FILE.read_text(encoding="utf-8"))


def encode_value(label_encoders: dict[str, object], column: str, value: str) -> int:
    encoder = label_encoders[column]
    normalized = str(value).strip().lower()
    classes = [entry.strip().lower() for entry in encoder.classes_]
    if normalized not in classes:
        raise ValueError(f"Unknown '{value}' for {column}.")
    index = classes.index(normalized)
    return int(encoder.transform([encoder.classes_[index]])[0])


def build_feature_row(
    feature_columns: list[str],
    request: PredictionRequest,
    label_encoders: dict[str, object],
    reference_year: int,
) -> pd.DataFrame:
    car_age = max(1, reference_year - int(request.yr_mfr))
    values = {
        "kms_run": int(request.kms_run),
        "times_viewed": int(request.times_viewed),
        "fuel_type": encode_value(label_encoders, "fuel_type", request.fuel_type),
        "city": encode_value(label_encoders, "city", request.city),
        "body_type": encode_value(label_encoders, "body_type", request.body_type),
        "transmission": encode_value(label_encoders, "transmission", request.transmission),
        "car_age": car_age,
        "kms_per_year": int(request.kms_run) / car_age,
    }
    missing = [column for column in feature_columns if column not in values]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Unsupported feature columns: {missing_text}")
    return pd.DataFrame([[values[column] for column in feature_columns]], columns=feature_columns)


def build_prediction_analytics(request: PredictionRequest, predicted_price: float, reference_year: int) -> PredictionAnalytics:
    car_age = max(0, reference_year - int(request.yr_mfr))
    kms_run = int(request.kms_run)
    times_viewed = int(request.times_viewed)

    if kms_run <= 30000:
        mileage_band = "Low"
    elif kms_run <= 70000:
        mileage_band = "Medium"
    else:
        mileage_band = "High"

    if times_viewed >= 2000:
        demand_band = "Hot"
    elif times_viewed >= 500:
        demand_band = "Active"
    else:
        demand_band = "Normal"

    age_penalty = min(35, round(car_age * 1.8))
    mileage_penalty = min(45, round((kms_run / 120000) * 20))
    demand_boost = min(18, round((times_viewed / 5000) * 10))

    confidence_score = max(55, min(92, 78 - (age_penalty // 3) - (mileage_penalty // 4) + (demand_boost // 2)))

    market_pulse = "Balanced"
    if demand_boost >= 10:
        market_pulse = "High Interest"
    elif mileage_penalty >= 25:
        market_pulse = "Price Sensitive"

    return PredictionAnalytics(
        carAge=f"{car_age}y",
        mileage=f"{kms_run:,} km",
        views=f"{times_viewed:,}",
        mileageBand=mileage_band,
        demandBand=demand_band,
        estimatedRange=f"INR {int(predicted_price * 0.9):,} - INR {int(predicted_price * 1.1):,}",
        agePenalty=f"-{age_penalty}%",
        mileagePenalty=f"-{mileage_penalty}%",
        demandBoost=f"+{demand_boost}%",
        confidence=f"{confidence_score}%",
        marketPulse=market_pulse,
    )


def available_options(label_encoders: dict[str, object]) -> dict[str, list[str]]:
    return {
        column: [str(value) for value in encoder.classes_.tolist()]
        for column, encoder in label_encoders.items()
    }


def build_feature_importance(model: object, feature_columns: list[str], selected_feature_columns: list[str]) -> list[dict[str, object]]:
    raw_values: list[float] = []

    if hasattr(model, "feature_importances_"):
        raw_values = [float(value) for value in getattr(model, "feature_importances_", [])]
    elif hasattr(model, "coef_"):
        coefficients = getattr(model, "coef_", [])
        raw_values = [abs(float(value)) for value in coefficients]

    active_feature_columns = feature_columns
    if len(raw_values) == len(selected_feature_columns):
        active_feature_columns = selected_feature_columns

    if len(raw_values) != len(active_feature_columns):
        return []

    total = sum(raw_values)
    if total <= 0:
        return []

    ranked = []
    for feature, importance in zip(active_feature_columns, raw_values, strict=False):
        ranked.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "importance": round(importance, 6),
                "percentage": round((importance / total) * 100, 2),
            }
        )

    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked


def build_model_info() -> dict[str, object]:
    model, _, label_encoders, feature_columns = load_runtime_assets()
    feature_info = load_feature_info()
    selected_metric_key = get_selected_metric_key(feature_info)
    selected_feature_columns = get_selected_feature_columns(feature_info, feature_columns)
    selected_model_name = feature_info.get("final_model") or feature_info.get("selected_model", "Unknown")
    return {
        "selected_model": format_model_name(selected_model_name),
        "feature_columns": feature_columns,
        "raw_input_features": feature_info.get(
            "raw_input_features",
            ["yr_mfr", "kms_run", "fuel_type", "city", "times_viewed", "body_type", "transmission"],
        ),
        "categorical_features": feature_info.get(
            "categorical_features",
            ["fuel_type", "city", "body_type", "transmission"],
        ),
        "available_options": available_options(label_encoders),
        "metrics": feature_info.get("metrics", {}),
        "feature_importance": build_feature_importance(model, feature_columns, selected_feature_columns),
        "train_samples": feature_info.get("train_samples"),
        "test_samples": feature_info.get("test_samples"),
        "final_r2": feature_info.get("final_r2"),
        "final_mae": feature_info.get("final_mae"),
        "final_mse": feature_info.get("final_mse"),
        "selected_metric_key": selected_metric_key,
        "selected_features_tuned": selected_feature_columns,
        "reference_year": get_reference_year(feature_info),
    }


def predict_price(request: PredictionRequest) -> tuple[float, PredictionAnalytics]:
    model, scaler, label_encoders, feature_columns = load_runtime_assets()
    feature_info = load_feature_info()
    reference_year = get_reference_year(feature_info)
    selected_feature_columns = get_selected_feature_columns(feature_info, feature_columns)

    row = build_feature_row(feature_columns, request, label_encoders, reference_year)
    scaled_row = scaler.transform(row)
    model_input = scaled_row

    if selected_feature_columns != feature_columns:
        selected_indices = [feature_columns.index(column) for column in selected_feature_columns]
        model_input = scaled_row[:, selected_indices]

    predicted_price = float(model.predict(model_input)[0])
    analytics = build_prediction_analytics(request, predicted_price, reference_year)
    return predicted_price, analytics
