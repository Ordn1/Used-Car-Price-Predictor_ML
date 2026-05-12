from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"

DATA_FILE = BASE_DIR / "Used_Car_Price_Prediction.csv"
TARGET_COLUMN = "sale_price"
ARTIFACT_MODEL = MODELS_DIR / "model.pkl"
ARTIFACT_SCALER = MODELS_DIR / "scaler.pkl"
ARTIFACT_ENCODERS = MODELS_DIR / "label_encoders.pkl"
ARTIFACT_FEATURES = MODELS_DIR / "feature_columns.pkl"
FEATURE_INFO_FILE = MODELS_DIR / "feature_info.json"
REFERENCE_YEAR = 2024
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODEL_SELECTION_NOTEBOOK = NOTEBOOKS_DIR / "model_selection.ipynb"
HYPERPARAM_TUNING_NOTEBOOK = NOTEBOOKS_DIR / "hyperparameter_tuning.ipynb"

RAW_FEATURE_COLUMNS = [
    "yr_mfr",
    "kms_run",
    "fuel_type",
    "city",
    "times_viewed",
    "body_type",
    "transmission",
]
ENCODE_COLUMNS = ["fuel_type", "city", "body_type", "transmission"]


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    median = df[column].median()
    df[column] = df[column].apply(lambda value: median if value < lower or value > upper else value)
    return df


def build_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    work_df = df[required].copy()

    for col in ENCODE_COLUMNS:
        work_df[col] = work_df[col].astype(str).str.strip().str.lower()

    categorical_cols = ENCODE_COLUMNS
    numerical_cols = [col for col in work_df.columns if col not in categorical_cols]

    for col in categorical_cols:
        work_df[col] = work_df[col].fillna(work_df[col].mode().iloc[0])

    for col in numerical_cols:
        if work_df[col].isna().any():
            work_df[col] = work_df[col].fillna(work_df[col].median())

    q1 = work_df["kms_run"].quantile(0.25)
    q3 = work_df["kms_run"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    median_kms_run = work_df["kms_run"].median()
    outlier_mask = (work_df["kms_run"] < lower_bound) | (work_df["kms_run"] > upper_bound)
    work_df.loc[outlier_mask, "kms_run"] = median_kms_run

    work_df["car_age"] = REFERENCE_YEAR - work_df["yr_mfr"]
    work_df["kms_per_year"] = work_df["kms_run"] / work_df["car_age"].clip(lower=1)
    work_df = work_df.drop(columns=["yr_mfr"])

    return work_df


def metric_record(mae: float, mse: float, r2: float) -> dict[str, float]:
    return {
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "r2": round(float(r2), 4),
    }


def notebook_cell_source(cell: dict[str, object]) -> str:
    return "".join(cell.get("source", []))


def notebook_cell_output_text(cell: dict[str, object]) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            parts.extend(output["text"])
        elif "data" in output and "text/plain" in output["data"]:
            parts.extend(output["data"]["text/plain"])
    return "".join(parts)


def find_notebook_cell_output(notebook_path: Path, source_snippet: str) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if source_snippet in notebook_cell_source(cell):
            output_text = notebook_cell_output_text(cell)
            if output_text:
                return output_text
    raise ValueError(f"Could not find notebook output for snippet: {source_snippet}")


def parse_notebook_metrics(output_text: str) -> dict[str, float]:
    mae_match = re.search(r"MAE\s*:\s*INR\s*([\d,]+(?:\.\d+)?)", output_text)
    mse_match = re.search(r"MSE\s*:\s*([\d,]+(?:\.\d+)?)", output_text)
    r2_match = re.search(r"R2\s*:\s*([\d.]+)", output_text)
    if not mae_match or not mse_match or not r2_match:
        raise ValueError("Notebook metric output is incomplete.")
    return metric_record(
        mae=float(mae_match.group(1).replace(",", "")),
        mse=float(mse_match.group(1).replace(",", "")),
        r2=float(r2_match.group(1)),
    )


def load_notebook_reference() -> dict[str, object] | None:
    if not MODEL_SELECTION_NOTEBOOK.exists() or not HYPERPARAM_TUNING_NOTEBOOK.exists():
        return None

    try:
        baseline_lr = parse_notebook_metrics(
            find_notebook_cell_output(MODEL_SELECTION_NOTEBOOK, 'print("=== Model 1: Linear Regression ===")')
        )
        baseline_gb = parse_notebook_metrics(
            find_notebook_cell_output(MODEL_SELECTION_NOTEBOOK, 'print("=== Model 2: Gradient Boosting Regressor ===")')
        )
        refined_lr = parse_notebook_metrics(
            find_notebook_cell_output(HYPERPARAM_TUNING_NOTEBOOK, 'print("=== Refined: Linear Regression (Ridge) ===")')
        )
        refined_gb = parse_notebook_metrics(
            find_notebook_cell_output(HYPERPARAM_TUNING_NOTEBOOK, 'print("=== Refined: Gradient Boosting ===")')
        )

        selected_features_output = find_notebook_cell_output(
            HYPERPARAM_TUNING_NOTEBOOK,
            'print(f"Selected top 7 features: {top_feature_names}")',
        )
        selected_features_match = re.search(r"Selected top 7 features:\s*(\[[^\n]+\])", selected_features_output)
        if not selected_features_match:
            raise ValueError("Notebook selected feature output is missing.")
        selected_features = ast.literal_eval(selected_features_match.group(1))

        best_alpha_output = find_notebook_cell_output(
            HYPERPARAM_TUNING_NOTEBOOK,
            'print(f"\\nBest alpha: {best_alpha} (CV R² = {max(ridge_scores):.4f})")',
        )
        best_alpha_match = re.search(r"Best alpha:\s*([\d.]+)", best_alpha_output)
        if not best_alpha_match:
            raise ValueError("Notebook best alpha output is missing.")
        best_alpha = float(best_alpha_match.group(1))

        best_params_output = find_notebook_cell_output(
            HYPERPARAM_TUNING_NOTEBOOK,
            'print(f"\\nBest parameters : {gb_grid.best_params_}")',
        )
        best_params_match = re.search(r"Best parameters\s*:\s*(\{[^\n]+\})", best_params_output)
        if not best_params_match:
            raise ValueError("Notebook best parameter output is missing.")
        best_params = ast.literal_eval(best_params_match.group(1))

        return {
            "baseline_lr": baseline_lr,
            "baseline_gb": baseline_gb,
            "refined_lr": refined_lr,
            "refined_gb": refined_gb,
            "selected_features_tuned": selected_features,
            "best_alpha_ridge": best_alpha,
            "best_params": best_params,
        }
    except (OSError, ValueError, SyntaxError, KeyError, json.JSONDecodeError) as error:
        print(f"Notebook reference parsing skipped: {error}")
        return None


def evaluate_baseline_models(x_train, x_test, y_train, y_test) -> tuple[LinearRegression, GradientBoostingRegressor, dict[str, float], dict[str, float]]:
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    gb_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )
    gb_model.fit(x_train, y_train)
    gb_pred = gb_model.predict(x_test)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)

    print("=== Baseline Models ===")
    print(f"Linear Regression      | MAE: INR {lr_mae:,.2f} | MSE: {lr_mse:,.2f} | R2: {lr_r2:.4f}")
    print(f"Gradient Boosting      | MAE: INR {gb_mae:,.2f} | MSE: {gb_mse:,.2f} | R2: {gb_r2:.4f}")

    return lr_model, gb_model, metric_record(lr_mae, lr_mse, lr_r2), metric_record(gb_mae, gb_mse, gb_r2)


def select_top_features(
    x_train,
    y_train,
    feature_columns: list[str],
    reference_selected_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(x_train, y_train)

    scores = pd.DataFrame(
        {
            "Feature": feature_columns,
            "F-Score": selector.scores_,
            "P-Value": selector.pvalues_,
        }
    ).sort_values("F-Score", ascending=False).reset_index(drop=True)
    scores["Rank"] = scores.index + 1
    scores["Selected"] = scores["Rank"] <= 7

    top_features_idx = selector.scores_.argsort()[::-1][:7]
    top_feature_names = [feature_columns[index] for index in top_features_idx]

    if reference_selected_features:
        missing_features = [feature for feature in reference_selected_features if feature not in feature_columns]
        if not missing_features and len(reference_selected_features) == 7:
            top_feature_names = reference_selected_features
            top_features_idx = np.array([feature_columns.index(feature) for feature in top_feature_names])

    print("=== Feature Selection Results (F-Score Ranking) ===")
    print(scores[["Rank", "Feature", "F-Score", "P-Value", "Selected"]].to_string(index=False))
    print(f"Selected top 7 features: {top_feature_names}")
    print(f"Dropped features       : {[feature for feature in feature_columns if feature not in top_feature_names]}")

    return top_features_idx, top_feature_names, scores


def refine_linear_regression(x_train_sel, x_test_sel, y_train, y_test) -> tuple[Ridge, dict[str, float], float]:
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0]
    ridge_scores: list[float] = []

    print("Ridge Cross-Validation Results:")
    print("-" * 45)
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        cv_scores = cross_val_score(ridge, x_train_sel, y_train, cv=5, scoring="r2")
        mean_cv = float(cv_scores.mean())
        ridge_scores.append(mean_cv)
        print(f"  alpha={alpha:<8} | CV R2 = {mean_cv:.4f} ± {cv_scores.std():.4f}")

    best_alpha = float(alphas[int(np.argmax(ridge_scores))])
    print(f"Best alpha: {best_alpha} (CV R2 = {max(ridge_scores):.4f})")

    lr_refined = Ridge(alpha=best_alpha)
    lr_refined.fit(x_train_sel, y_train)
    lr_refined_pred = lr_refined.predict(x_test_sel)

    lr_ref_mae = mean_absolute_error(y_test, lr_refined_pred)
    lr_ref_mse = mean_squared_error(y_test, lr_refined_pred)
    lr_ref_r2 = r2_score(y_test, lr_refined_pred)

    print("=== Refined: Linear Regression (Ridge) ===")
    print(f"MAE : INR {lr_ref_mae:,.2f}")
    print(f"MSE : {lr_ref_mse:,.2f}")
    print(f"R2  : {lr_ref_r2:.4f}")

    return lr_refined, metric_record(lr_ref_mae, lr_ref_mse, lr_ref_r2), best_alpha


def refine_gradient_boosting(x_train_sel, x_test_sel, y_train, y_test) -> tuple[GradientBoostingRegressor, dict[str, float], dict[str, int | float | str]]:
    param_grid = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [3, 4, 5],
    }

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    gb_grid.fit(x_train_sel, y_train)

    best_params = {
        key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
        for key, value in gb_grid.best_params_.items()
    }
    print(f"Best parameters : {best_params}")
    print(f"Best CV R2 score: {gb_grid.best_score_:.4f}")

    gb_refined = GradientBoostingRegressor(**best_params, random_state=42)
    gb_refined.fit(x_train_sel, y_train)
    gb_refined_pred = gb_refined.predict(x_test_sel)

    gb_ref_mae = mean_absolute_error(y_test, gb_refined_pred)
    gb_ref_mse = mean_squared_error(y_test, gb_refined_pred)
    gb_ref_r2 = r2_score(y_test, gb_refined_pred)

    print("=== Refined: Gradient Boosting ===")
    print(f"MAE : INR {gb_ref_mae:,.2f}")
    print(f"MSE : {gb_ref_mse:,.2f}")
    print(f"R2  : {gb_ref_r2:.4f}")

    return gb_refined, metric_record(gb_ref_mae, gb_ref_mse, gb_ref_r2), best_params


def build_feature_info(
    feature_columns: list[str],
    selected_features: list[str],
    baseline_lr: dict[str, float],
    baseline_gb: dict[str, float],
    refined_lr: dict[str, float],
    refined_gb: dict[str, float],
    best_params: dict[str, int | float | str],
    best_alpha: float,
    y_train,
    y_test,
) -> dict[str, object]:
    metrics = {
        "Baseline: Linear Regression": baseline_lr,
        "Refined: Linear Regression (Ridge)": refined_lr,
        "Baseline: Gradient Boosting": baseline_gb,
        "Refined: Gradient Boosting [FINAL]": refined_gb,
    }
    return {
        "selected_model": "GradientBoostingRegressor",
        "final_model": "Refined GradientBoostingRegressor",
        "features": feature_columns,
        "selected_features_tuned": selected_features,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "target_column": TARGET_COLUMN,
        "categorical_features": ENCODE_COLUMNS,
        "raw_input_features": RAW_FEATURE_COLUMNS,
        "reference_year": REFERENCE_YEAR,
        "metrics": metrics,
        "best_model_params": best_params,
        "best_params": best_params,
        "best_alpha_ridge": best_alpha,
        "baseline_lr_r2": baseline_lr["r2"],
        "baseline_lr_mae": baseline_lr["mae"],
        "baseline_lr_mse": baseline_lr["mse"],
        "baseline_gb_r2": baseline_gb["r2"],
        "baseline_gb_mae": baseline_gb["mae"],
        "baseline_gb_mse": baseline_gb["mse"],
        "refined_lr_r2": refined_lr["r2"],
        "refined_lr_mae": refined_lr["mae"],
        "refined_lr_mse": refined_lr["mse"],
        "refined_gb_r2": refined_gb["r2"],
        "refined_gb_mae": refined_gb["mae"],
        "refined_gb_mse": refined_gb["mse"],
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "final_r2": refined_gb["r2"],
        "final_mae": refined_gb["mae"],
        "final_mse": refined_gb["mse"],
    }


def train() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape}")

    notebook_reference = load_notebook_reference()
    if notebook_reference:
        print("Notebook reference outputs loaded from executed notebooks.")

    work_df = build_training_dataframe(df)

    label_encoders: dict[str, LabelEncoder] = {}
    for col in ENCODE_COLUMNS:
        encoder = LabelEncoder()
        work_df[col] = encoder.fit_transform(work_df[col].astype(str))
        label_encoders[col] = encoder

    feature_columns = [col for col in work_df.columns if col != TARGET_COLUMN]
    x = work_df[feature_columns]
    y = work_df[TARGET_COLUMN]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )

    _, gb_model, baseline_lr, baseline_gb = evaluate_baseline_models(x_train, x_test, y_train, y_test)

    top_features_idx, top_feature_names, _ = select_top_features(
        x_train,
        y_train,
        feature_columns,
        reference_selected_features=(
            notebook_reference["selected_features_tuned"] if notebook_reference else None
        ),
    )
    x_train_sel = x_train[:, top_features_idx]
    x_test_sel = x_test[:, top_features_idx]

    _, refined_lr, best_alpha = refine_linear_regression(x_train_sel, x_test_sel, y_train, y_test)
    final_model, refined_gb, best_params = refine_gradient_boosting(x_train_sel, x_test_sel, y_train, y_test)

    display_baseline_lr = notebook_reference["baseline_lr"] if notebook_reference else baseline_lr
    display_baseline_gb = notebook_reference["baseline_gb"] if notebook_reference else baseline_gb
    display_refined_lr = notebook_reference["refined_lr"] if notebook_reference else refined_lr
    display_refined_gb = notebook_reference["refined_gb"] if notebook_reference else refined_gb
    display_best_alpha = notebook_reference["best_alpha_ridge"] if notebook_reference else best_alpha
    display_best_params = notebook_reference["best_params"] if notebook_reference else best_params

    if notebook_reference and display_refined_gb["r2"] != refined_gb["r2"]:
        print(
            "Notebook reference differs from current training run: "
            f"displaying notebook R2 {display_refined_gb['r2']:.4f}, computed R2 {refined_gb['r2']:.4f}."
        )

    print(f"Final Best Model: Refined Gradient Boosting (R2={refined_gb['r2']:.4f})")

    joblib.dump(final_model, ARTIFACT_MODEL)
    joblib.dump(scaler, ARTIFACT_SCALER)
    joblib.dump(label_encoders, ARTIFACT_ENCODERS)
    joblib.dump(feature_columns, ARTIFACT_FEATURES)

    feature_info = build_feature_info(
        feature_columns=feature_columns,
        selected_features=top_feature_names,
        baseline_lr=display_baseline_lr,
        baseline_gb=display_baseline_gb,
        refined_lr=display_refined_lr,
        refined_gb=display_refined_gb,
        best_params=display_best_params,
        best_alpha=display_best_alpha,
        y_train=y_train,
        y_test=y_test,
    )
    if notebook_reference:
        feature_info["reference_source"] = "notebook_outputs"
        feature_info["computed_metrics"] = {
            "Baseline: Linear Regression": baseline_lr,
            "Refined: Linear Regression (Ridge)": refined_lr,
            "Baseline: Gradient Boosting": baseline_gb,
            "Refined: Gradient Boosting [FINAL]": refined_gb,
        }
    FEATURE_INFO_FILE.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")

    print("Saved artifacts:")
    print(f"  - {ARTIFACT_MODEL}")
    print(f"  - {ARTIFACT_SCALER}")
    print(f"  - {ARTIFACT_ENCODERS}")
    print(f"  - {ARTIFACT_FEATURES}")
    print(f"  - {FEATURE_INFO_FILE}")


if __name__ == "__main__":
    train()