from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

import db


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
APP_ICON_PATH = REPO_ROOT / "images" / "UCP.png"
DATA_FILE = REPO_ROOT / "ml-training" / "Used_Car_Price_Prediction.csv"
API_BASE_URL = os.getenv("USED_CAR_API_URL", "http://127.0.0.1:5000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("USED_CAR_API_TIMEOUT", "30"))
RUNTIME_MODE = os.getenv("USED_CAR_RUNTIME_MODE", "api").strip().lower()

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon=str(APP_ICON_PATH) if APP_ICON_PATH.exists() else "🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
footer, [data-testid="stSidebar"], [data-testid="stDecoration"] { display: none !important; }

[data-testid="stHeader"],
[data-testid="stAppHeader"],
.stAppHeader,
[data-testid="stAppViewHeader"] {
    background: transparent !important;
    pointer-events: none !important;
    height: 0 !important;
    min-height: 0 !important;
}

[data-testid="stHeaderActionElements"],
[data-testid="stAppToolbar"] {
    pointer-events: auto !important;
    position: fixed !important;
    right: 2rem !important;
    top: 1rem !important;
    bottom: auto !important;
    left: auto !important;
    height: 45px !important;
    width: auto !important;
    background: var(--secondary-background-color) !important;
    border-radius: 50px !important;
    padding: 0 1rem !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    z-index: 5000 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; }
.block-container { padding: 0 !important; max-width: 100vw !important; margin: 0 !important; }
[data-testid="stAppViewContainer"] { background: transparent; overflow: hidden; width: 100vw; height: 100vh; }
[data-testid="stMain"] { overflow: hidden !important; }

iframe {
    border: none !important;
    display: block !important;
    width: 100vw !important;
    height: 100vh !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    pointer-events: auto !important;
    z-index: 4000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

CITY_IMAGES = {
    "mumbai": "https://images.unsplash.com/photo-1566552881560-0be862a7c445?w=1400&q=85",
    "delhi": "https://images.unsplash.com/photo-1587474260584-136574528ed5?w=1400&q=85",
    "bangalore": "https://images.unsplash.com/photo-1596176530529-78163a4f7af2?w=1400&q=85",
    "bengaluru": "https://images.unsplash.com/photo-1596176530529-78163a4f7af2?w=1400&q=85",
    "chennai": "https://images.unsplash.com/photo-1582510003544-4d00b7f74220?w=1400&q=85",
    "hyderabad": "https://images.unsplash.com/photo-1581852017103-68ac65514cf7?w=1400&q=85",
    "pune": "https://picsum.photos/seed/usedcar-pune/1600/900",
    "ahmedabad": "https://images.unsplash.com/photo-1615209853186-e4bd66602508?w=1400&q=85",
    "kolkata": "https://images.unsplash.com/photo-1558431382-27e303142255?w=1400&q=85",
    "faridabad": "https://picsum.photos/seed/usedcar-faridabad/1600/900",
    "ghaziabad": "https://picsum.photos/seed/usedcar-ghaziabad/1600/900",
    "gurgaon": "https://picsum.photos/seed/usedcar-gurgaon/1600/900",
    "lucknow": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?w=1400&q=85",
    "new delhi": "https://picsum.photos/seed/usedcar-newdelhi/1600/900",
    "noida": "https://picsum.photos/seed/usedcar-noida/1600/900",
}

HISTORY_IMAGES = {
    "left": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1600&h=1200&q=85",
    "right": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1600&h=1200&q=85",
}

ABOUT_IMAGES = {
    "left": "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=1600&h=1200&q=85",
    "right": "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=1600&h=1200&q=85",
}

DEFAULT_FUEL_OPTIONS = ["petrol", "diesel", "cng", "electric", "lpg"]
DEFAULT_CITY_OPTIONS = ["mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune", "ahmedabad", "kolkata"]
DEFAULT_BODY_OPTIONS = ["sedan", "hatchback", "suv", "muv", "coupe", "minivan"]
DEFAULT_TRANS_OPTIONS = ["manual", "automatic"]


def api_request(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{API_BASE_URL}{path}"
    response = requests.request(method, url, json=payload, timeout=API_TIMEOUT_SECONDS)
    if response.ok:
        return response.json()

    try:
        detail = response.json()
    except ValueError:
        detail = response.text.strip() or "Unknown API error"
    raise RuntimeError(f"API request failed ({response.status_code}): {detail}")


@lru_cache(maxsize=1)
def _load_direct_runtime():
    py_app_path = str(REPO_ROOT / "py-app")
    if py_app_path not in sys.path:
        sys.path.insert(0, py_app_path)

    from app.model_loader import build_model_info, load_runtime_assets, predict_price  # type: ignore
    from app.schemas import PredictionRequest  # type: ignore

    return build_model_info, load_runtime_assets, predict_price, PredictionRequest


def service_request(method: str, path: str, payload: dict | None = None) -> dict:
    if RUNTIME_MODE == "api":
        return api_request(method, path, payload)

    build_model_info, load_runtime_assets, predict_price, prediction_request_cls = _load_direct_runtime()

    if method == "GET" and path == "/api/model-info":
        return build_model_info()

    if method == "GET" and path == "/api/health":
        load_runtime_assets()
        model_info = build_model_info()
        return {
            "status": "ok",
            "model_loaded": True,
            "selected_model": model_info.get("selected_model", "Unknown"),
        }

    if method == "POST" and path == "/api/predict":
        request = prediction_request_cls(**(payload or {}))
        predicted_price, analytics = predict_price(request)
        analytics_payload = analytics.model_dump() if hasattr(analytics, "model_dump") else analytics.dict()
        return {
            "predicted_price": predicted_price,
            "analytics": analytics_payload,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

    raise RuntimeError(f"Unsupported direct runtime request: {method} {path}")


def ensure_history_store() -> str | None:
    try:
        db.ensure_database_ready()
        return None
    except Exception as exc:
        return str(exc)


@st.cache_data(ttl=60)
def fetch_model_info() -> dict:
    return service_request("GET", "/api/model-info")


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    return service_request("GET", "/api/health")


@st.cache_data(ttl=300)
def load_dataset_overview(dataset_path: str) -> dict[str, object]:
    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return {
            "recordCount": "-",
            "featureCount": "-",
            "cityCoverage": "-",
            "medianPrice": "-",
            "medianPriceValue": None,
            "avgKms": "-",
            "priceRange": "-",
            "priceMinValue": None,
            "priceMaxValue": None,
        }

    city_coverage = int(df["city"].nunique()) if "city" in df.columns else 0
    median_price = int(df["sale_price"].median()) if "sale_price" in df.columns else 0
    avg_kms = int(df["kms_run"].mean()) if "kms_run" in df.columns else 0
    price_min = int(df["sale_price"].min()) if "sale_price" in df.columns else 0
    price_max = int(df["sale_price"].max()) if "sale_price" in df.columns else 0

    return {
        "recordCount": f"{len(df):,}",
        "featureCount": f"{max(0, len(df.columns) - 1):,}",
        "cityCoverage": f"{city_coverage}",
        "medianPrice": f"INR {median_price:,}",
        "medianPriceValue": median_price,
        "avgKms": f"{avg_kms:,} km",
        "priceRange": f"INR {price_min:,} - INR {price_max:,}",
        "priceMinValue": price_min,
        "priceMaxValue": price_max,
    }


def format_inr(value: float | int) -> str:
    return f"INR {int(round(float(value))):,}"


def parse_confidence(value: object) -> int:
    text = str(value or "0").replace("%", "").strip()
    try:
        return int(round(float(text)))
    except ValueError:
        return 0


def build_history_summary(history: list[dict]) -> dict[str, object]:
    total_predictions = len(history)
    if total_predictions == 0:
        return {
            "totalPredictions": 0,
            "avgConfidence": "0%",
            "avgConfidenceValue": 0,
            "avgPredictedPrice": "INR 0",
            "avgPredictedPriceValue": 0.0,
            "medianPredictedPrice": "INR 0",
            "medianPredictedPriceValue": 0.0,
            "highestPredictedPrice": "INR 0",
            "highestPredictedPriceValue": 0.0,
            "lowestPredictedPrice": "INR 0",
            "lowestPredictedPriceValue": 0.0,
            "topCity": "-",
            "priceTrend": [],
            "cityBreakdown": [],
            "fuelBreakdown": [],
            "mileageScatter": [],
            "insights": ["Run and save predictions to unlock charts, trend lines, and segment analytics."],
        }

    df = pd.DataFrame(history).copy()
    df["predictedPrice"] = pd.to_numeric(df.get("predictedPrice"), errors="coerce").fillna(0.0)
    df["confidenceValue"] = df.get("confidence", pd.Series(dtype=object)).apply(parse_confidence)
    df["createdAtParsed"] = pd.to_datetime(df.get("createdAt"), errors="coerce")
    df["city"] = df.get("city", pd.Series(dtype=object)).fillna("unknown").astype(str)
    df["fuel_type"] = df.get("fuel_type", pd.Series(dtype=object)).fillna("unknown").astype(str)
    df["yr_mfr"] = pd.to_numeric(df.get("yr_mfr"), errors="coerce").fillna(datetime.now().year)
    df["kms_run"] = pd.to_numeric(df.get("kms_run"), errors="coerce").fillna(0)
    df["carAge"] = (datetime.now().year - df["yr_mfr"]).clip(lower=0)

    avg_conf = int(round(df["confidenceValue"].mean()))
    avg_price = float(df["predictedPrice"].mean())
    median_price = float(df["predictedPrice"].median())
    highest_price = float(df["predictedPrice"].max())
    lowest_price = float(df["predictedPrice"].min())

    trend_df = df.sort_values("createdAtParsed").tail(12).reset_index(drop=True)
    price_trend = [
        {
            "sequence": index + 1,
            "label": row.createdAtParsed.strftime("%b %d, %I:%M %p") if pd.notna(row.createdAtParsed) else f"Run {index + 1}",
            "timestamp": row.createdAtParsed.isoformat(timespec="seconds") if pd.notna(row.createdAtParsed) else "",
            "city": str(row.city).title(),
            "price": round(float(row.predictedPrice), 2),
            "confidence": int(row.confidenceValue),
        }
        for index, row in enumerate(trend_df.itertuples(index=False))
    ]

    city_grouped = (
        df.groupby("city", dropna=False)
        .agg(count=("city", "size"), avgPrice=("predictedPrice", "mean"), avgConfidence=("confidenceValue", "mean"))
        .reset_index()
        .sort_values(["count", "avgPrice"], ascending=[False, False])
        .head(6)
    )
    city_breakdown = [
        {
            "label": str(row.city).title(),
            "count": int(row.count),
            "avgPrice": round(float(row.avgPrice), 2),
            "avgConfidence": int(round(float(row.avgConfidence))),
        }
        for row in city_grouped.itertuples(index=False)
    ]

    fuel_grouped = (
        df.groupby("fuel_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "fuel_type"], ascending=[False, True])
    )
    fuel_breakdown = [
        {
            "label": str(row.fuel_type).title(),
            "count": int(row.count),
        }
        for row in fuel_grouped.itertuples(index=False)
    ]

    scatter_df = df.sort_values(["kms_run", "predictedPrice"], ascending=[True, False]).tail(60)
    mileage_scatter = [
        {
            "x": int(row.kms_run),
            "y": round(float(row.predictedPrice), 2),
            "label": f"{int(row.yr_mfr)} | {str(row.city).title()} | {str(row.fuel_type).upper()}",
            "confidence": int(row.confidenceValue),
            "carAge": int(row.carAge),
        }
        for row in scatter_df.itertuples(index=False)
    ]

    top_city = city_breakdown[0]["label"] if city_breakdown else "-"
    insights = [
        f"Highest saved valuation so far is {format_inr(highest_price)}.",
        f"{top_city} currently leads prediction activity." if top_city != "-" else "Run more predictions to reveal city trends.",
        f"Average model confidence across saved predictions is {avg_conf}%.",
    ]

    return {
        "totalPredictions": total_predictions,
        "avgConfidence": f"{avg_conf}%",
        "avgConfidenceValue": avg_conf,
        "avgPredictedPrice": format_inr(avg_price),
        "avgPredictedPriceValue": round(avg_price, 2),
        "medianPredictedPrice": format_inr(median_price),
        "medianPredictedPriceValue": round(median_price, 2),
        "highestPredictedPrice": format_inr(highest_price),
        "highestPredictedPriceValue": round(highest_price, 2),
        "lowestPredictedPrice": format_inr(lowest_price),
        "lowestPredictedPriceValue": round(lowest_price, 2),
        "topCity": top_city,
        "priceTrend": price_trend,
        "cityBreakdown": city_breakdown,
        "fuelBreakdown": fuel_breakdown,
        "mileageScatter": mileage_scatter,
        "insights": insights,
    }


def build_about_payload(model_info: dict, history_summary: dict[str, object]) -> dict:
    dataset_overview = load_dataset_overview(str(DATA_FILE))
    storage_label = db.get_storage_label()
    storage_description = db.describe_storage()
    feature_labels = {
        "kms_run": "Mileage",
        "fuel_type": "Fuel Type",
        "city": "City",
        "times_viewed": "Market Demand",
        "body_type": "Body Type",
        "transmission": "Transmission",
        "car_age": "Car Age",
        "kms_per_year": "Usage Intensity",
    }
    selected_model = model_info.get("selected_model", "Not Loaded")
    feature_columns = model_info.get("feature_columns", [])
    feature_importance = model_info.get("feature_importance", [])
    metrics = model_info.get("metrics", {})

    preferred_metric_order = [
        "Baseline: Linear Regression",
        "Refined: Linear Regression (Ridge)",
        "Baseline: Gradient Boosting",
        "Refined: Gradient Boosting [FINAL]",
    ]
    selected_metric_key = next(
        (key for key in ["Refined: Gradient Boosting [FINAL]", selected_model, *preferred_metric_order] if key in metrics),
        next(iter(metrics.keys()), None),
    )
    selected_metrics = metrics.get(selected_metric_key or "", {})

    baseline_for_selected = {}
    if selected_metric_key == "Refined: Gradient Boosting [FINAL]":
        baseline_for_selected = metrics.get("Baseline: Gradient Boosting", {})
    elif selected_metric_key == "Refined: Linear Regression (Ridge)":
        baseline_for_selected = metrics.get("Baseline: Linear Regression", {})

    selected_r2 = float(selected_metrics.get("r2", model_info.get("final_r2") or 0.0))
    selected_mae = float(selected_metrics.get("mae", model_info.get("final_mae") or 0.0))
    selected_mse = float(selected_metrics.get("mse", model_info.get("final_mse") or 0.0))
    selected_delta = selected_r2 - float(baseline_for_selected.get("r2", selected_r2))

    if feature_importance:
        top_features = [item.get("label", item.get("feature", "-")) for item in feature_importance[:8]]
    else:
        top_features = [feature_labels.get(col, col.replace("_", " ").title()) for col in feature_columns][:8]

    model_comparison = []
    comparison_order = [name for name in preferred_metric_order if name in metrics]
    comparison_order.extend([name for name in metrics.keys() if name not in comparison_order])
    for model_name in comparison_order:
        values = metrics.get(model_name, {})
        mse_value = float(values.get("mse", 0.0))
        baseline_name = None
        if model_name == "Refined: Linear Regression (Ridge)":
            baseline_name = "Baseline: Linear Regression"
        elif model_name == "Refined: Gradient Boosting [FINAL]":
            baseline_name = "Baseline: Gradient Boosting"
        baseline_r2 = float(metrics.get(baseline_name, {}).get("r2", values.get("r2", 0.0))) if baseline_name else float(values.get("r2", 0.0))
        model_comparison.append(
            {
                "model": model_name,
                "r2": round(float(values.get("r2", 0.0)), 4),
                "mae": round(float(values.get("mae", 0.0)), 2),
                "rmse": round(mse_value ** 0.5, 2) if mse_value > 0 else 0.0,
                "gap": round(float(values.get("r2", 0.0)) - baseline_r2, 4),
            }
        )

    selected_metrics_cards = [
        {
            "label": "Train Samples",
            "value": f"{int(model_info.get('train_samples') or 0):,}",
            "format": "count",
            "rawValue": int(model_info.get("train_samples") or 0),
        },
        {
            "label": "Test Samples",
            "value": f"{int(model_info.get('test_samples') or 0):,}",
            "format": "count",
            "rawValue": int(model_info.get("test_samples") or 0),
        },
        {
            "label": "R2 Score",
            "value": f"{selected_r2:.4f}",
            "format": "score",
            "rawValue": round(selected_r2, 4),
        },
        {
            "label": "MAE",
            "value": format_inr(selected_mae),
            "format": "currency",
            "rawValue": round(selected_mae, 2),
        },
        {
            "label": "MSE",
            "value": f"{selected_mse:,.0f}",
            "format": "number",
            "rawValue": round(selected_mse, 2),
        },
        {
            "label": "Delta vs Baseline",
            "value": f"{selected_delta:+.4f}",
            "format": "delta",
            "rawValue": round(selected_delta, 4),
        },
    ]

    if RUNTIME_MODE == "direct":
        clockwork_flow = [
            "1. Streamlit captures the 7 raw vehicle and demand inputs from the predictor form.",
            "2. Streamlit loads the shared artifacts locally and reuses the Python inference module directly.",
            "3. Python engineers car_age and kms_per_year, then label-encodes categorical fields.",
            "4. The scaler transforms the ordered feature row before the regressor predicts the market value.",
            "5. Python derives heuristic analytics such as age penalty, mileage penalty, demand boost, and confidence.",
            f"6. Streamlit stores the completed prediction in {storage_label} so the History page can chart real usage patterns.",
        ]
    else:
        clockwork_flow = [
            "1. Streamlit captures the 7 raw vehicle and demand inputs from the predictor form.",
            "2. Streamlit sends the payload to the public ASP.NET Core API instead of touching the model directly.",
            "3. The C# API forwards the request to the internal Python inference service.",
            "4. Python engineers car_age and kms_per_year, then label-encodes categorical fields.",
            "5. The scaler transforms the ordered feature row before the regressor predicts the market value.",
            "6. Python derives heuristic analytics such as age penalty, mileage penalty, demand boost, and confidence.",
            f"7. Streamlit stores the completed prediction in {storage_label} so the History page can chart real usage patterns.",
        ]

    raw_inputs = [
        feature_labels.get(column, str(column).replace("_", " ").title())
        for column in model_info.get("raw_input_features", [])
    ]

    engineered_features = [
        "car_age = 2024 - yr_mfr",
        "kms_per_year = kms_run / max(car_age, 1)",
    ]

    project_structure = [
        {
            "folder": "ml-training/",
            "purpose": "Owns the training script, notebooks, and the dataset used to regenerate artifacts.",
        },
        {
            "folder": "models/",
            "purpose": "Stores the shared model, scaler, encoders, feature order, and model metadata.",
        },
        {
            "folder": "py-app/",
            "purpose": "Contains the reusable Python inference logic and the FastAPI service used in full local mode.",
        },
        {
            "folder": "web-api/",
            "purpose": "Hosts the public ASP.NET Core API used in the full local multi-service runtime.",
        },
        {
            "folder": "streamlit-app/",
            "purpose": "Contains the Streamlit front end, custom HTML component, direct runtime host, and configurable history integration.",
        },
    ]

    if RUNTIME_MODE == "direct":
        runtime_stack = [
            {
                "label": "UI Layer",
                "value": "Streamlit with an embedded custom HTML, CSS, and JavaScript interface.",
            },
            {
                "label": "Inference Layer",
                "value": "Direct Python runtime reusing the shared scikit-learn artifacts and inference module locally.",
            },
            {
                "label": "Storage",
                "value": storage_description,
            },
            {
                "label": "Deployment Mode",
                "value": "Streamlit-only hosted mode without the ASP.NET Core and FastAPI services.",
            },
        ]
    else:
        runtime_stack = [
            {
                "label": "UI Layer",
                "value": "Streamlit with an embedded custom HTML, CSS, and JavaScript interface.",
            },
            {
                "label": "Public API",
                "value": "ASP.NET Core minimal API acting as the stable integration boundary.",
            },
            {
                "label": "Inference Layer",
                "value": "FastAPI service loading scikit-learn artifacts and returning predictions plus analytics.",
            },
            {
                "label": "Storage",
                "value": storage_description,
            },
        ]

    return {
        "systemTitle": "Find Your Car's True Value",
        "purpose": "A machine learning system that estimates used-car value from key vehicle and demand signals.",
        "businessProblem": "Used-car prices are inconsistent and often subjective. This system provides a data-driven price reference for faster and more reliable decisions.",
        "modelName": selected_model,
        "historyCount": history_summary.get("totalPredictions", 0),
        "avgConfidence": history_summary.get("avgConfidence", "0%"),
        "recordCount": dataset_overview["recordCount"],
        "featureCount": dataset_overview["featureCount"],
        "cityCoverage": dataset_overview["cityCoverage"],
        "medianPrice": dataset_overview["medianPrice"],
        "medianPriceValue": dataset_overview.get("medianPriceValue"),
        "avgKms": dataset_overview["avgKms"],
        "priceRange": dataset_overview["priceRange"],
        "priceMinValue": dataset_overview.get("priceMinValue"),
        "priceMaxValue": dataset_overview.get("priceMaxValue"),
        "topFeatures": top_features,
        "trainingFlow": [
            "Select the 7 seller-provided inputs and the sale_price target.",
            "Normalize categorical text, impute missing values, and replace kms_run outliers with the median via IQR bounds.",
            "Engineer car_age and kms_per_year using the notebook reference year 2024.",
            "Label-encode categorical fields, standardize all features, and split the dataset 80/20.",
            "Benchmark baseline Linear Regression and Gradient Boosting on the held-out test set.",
            "Use SelectKBest to keep the top 7 predictive features, then tune Ridge and Gradient Boosting hyperparameters.",
            "Persist the tuned model, scaler, encoders, feature order, and notebook-accurate metrics into shared artifacts.",
        ],
        "rawInputs": raw_inputs,
        "engineeredFeatures": engineered_features,
        "featureImportance": feature_importance,
        "modelComparison": model_comparison,
        "selectedMetrics": selected_metrics_cards,
        "clockworkFlow": clockwork_flow,
        "historyInsights": history_summary.get("insights", []),
        "historyInsightStats": {
            "highestPredictedPriceValue": history_summary.get("highestPredictedPriceValue", 0.0),
            "topCity": history_summary.get("topCity", "-"),
            "avgConfidenceValue": history_summary.get("avgConfidenceValue", 0),
        },
        "confidenceRule": "Confidence = clamp(55%, 92%) after subtracting age and mileage penalties and adding a demand-based boost.",
        "projectStructure": project_structure,
        "runtimeStack": runtime_stack,
    }


def get_runtime_options() -> tuple[dict, list[str], list[str], list[str], list[str], str | None]:
    try:
        model_info = fetch_model_info()
        available_options = model_info.get("available_options", {})
        fuel_options = sorted(available_options.get("fuel_type", DEFAULT_FUEL_OPTIONS))
        city_options = sorted(available_options.get("city", DEFAULT_CITY_OPTIONS))
        body_options = sorted(available_options.get("body_type", DEFAULT_BODY_OPTIONS))
        trans_options = sorted(available_options.get("transmission", DEFAULT_TRANS_OPTIONS))
        return model_info, fuel_options, city_options, body_options, trans_options, None
    except Exception as exc:
        return (
            {},
            DEFAULT_FUEL_OPTIONS,
            DEFAULT_CITY_OPTIONS,
            DEFAULT_BODY_OPTIONS,
            DEFAULT_TRANS_OPTIONS,
            str(exc),
        )


defaults = {
    "fuel_type": DEFAULT_FUEL_OPTIONS[0],
    "transmission": DEFAULT_TRANS_OPTIONS[0],
    "body_type": DEFAULT_BODY_OPTIONS[0],
    "city": DEFAULT_CITY_OPTIONS[0],
    "yr_mfr": 2018,
    "kms_run": 45000,
    "times_viewed": 100,
    "predicted": None,
    "pred_error": None,
    "pred_analytics": None,
    "last_req_id": None,
    "last_nav_id": None,
    "current_page": "predictor",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


model_info, fuel_options, city_options, body_options, trans_options, api_error = get_runtime_options()
db_error = ensure_history_store()

try:
    prediction_history = db.fetch_prediction_history()
except Exception as exc:
    prediction_history = []
    db_error = str(exc)

history_summary = build_history_summary(prediction_history)
about_payload = build_about_payload(model_info, history_summary)
health_status = None
if not api_error:
    try:
        health_status = fetch_health()
    except Exception:
        health_status = None

ui_error = st.session_state.pred_error or api_error or db_error
years_list = [str(year) for year in range(datetime.now().year, 1964, -1)]

custom_car_ui = components.declare_component("used_car_ui", path=str(BASE_DIR))

result = custom_car_ui(
    yrVal=st.session_state.yr_mfr,
    kmsVal=st.session_state.kms_run,
    fuelVal=st.session_state.fuel_type,
    cityVal=st.session_state.city,
    bodyVal=st.session_state.body_type,
    transVal=st.session_state.transmission,
    tvVal=st.session_state.times_viewed,
    predPrice=st.session_state.predicted,
    predAnalytics=st.session_state.pred_analytics,
    errMsg=ui_error,
    currentPage=st.session_state.current_page,
    history=prediction_history,
    historySummary=history_summary,
    historyImages=HISTORY_IMAGES,
    aboutImages=ABOUT_IMAGES,
    aboutAnalytics=about_payload,
    modelName=model_info.get("selected_model", "Not Loaded"),
    featureCount=len(model_info.get("feature_columns", [])),
    cityImages=CITY_IMAGES,
    yearOptions=years_list,
    fuelOptions=fuel_options,
    cityOptions=city_options,
    bodyOptions=body_options,
    transOptions=trans_options,
    healthStatus=health_status,
    key="used_car_prediction_ui",
)

if result:
    action = result.get("action")

    if action == "navigate":
        nav_id = result.get("navId")
        target_page = result.get("page", "predictor")
        if nav_id != st.session_state.last_nav_id and target_page in {"predictor", "history", "about"}:
            st.session_state.last_nav_id = nav_id
            st.session_state.current_page = target_page
            st.rerun()

    if action == "predict" or result.get("predict") == 1:
        req_id = result.get("reqId")
        if req_id != st.session_state.last_req_id:
            st.session_state.last_req_id = req_id
            try:
                st.session_state.yr_mfr = int(result["yr"])
                st.session_state.kms_run = int(result["kms"])
                st.session_state.fuel_type = result["fuel"]
                st.session_state.city = result["city"]
                st.session_state.times_viewed = int(result["tv"])
                st.session_state.transmission = result["trans"]
                st.session_state.body_type = result["body"]
                st.session_state.current_page = "predictor"

                api_payload = {
                    "yr_mfr": st.session_state.yr_mfr,
                    "kms_run": st.session_state.kms_run,
                    "fuel_type": st.session_state.fuel_type,
                    "city": st.session_state.city,
                    "times_viewed": st.session_state.times_viewed,
                    "body_type": st.session_state.body_type,
                    "transmission": st.session_state.transmission,
                }
                prediction_response = service_request("POST", "/api/predict", api_payload)
                price = float(prediction_response["predicted_price"])
                analytics = prediction_response["analytics"]
                created_at = prediction_response.get("created_at", datetime.now().isoformat(timespec="seconds"))

                st.session_state.predicted = price
                st.session_state.pred_error = None
                st.session_state.pred_analytics = analytics

                history_entry = {
                    "createdAt": created_at,
                    "city": st.session_state.city,
                    "yr_mfr": st.session_state.yr_mfr,
                    "kms_run": st.session_state.kms_run,
                    "fuel_type": st.session_state.fuel_type,
                    "transmission": st.session_state.transmission,
                    "body_type": st.session_state.body_type,
                    "times_viewed": st.session_state.times_viewed,
                    "predictedPrice": price,
                    "confidence": analytics.get("confidence", "0%"),
                }
                db.insert_prediction(history_entry)
                st.rerun()
            except Exception as exc:
                st.session_state.pred_error = str(exc)
                st.session_state.predicted = None
                st.session_state.pred_analytics = None
                st.rerun()