from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


BASE_DIR = Path(__file__).resolve().parent
TABLE_NAME = "prediction_history"
SUPPORTED_BACKENDS = {"sqlite", "supabase"}


def _get_backend() -> str:
    backend = os.getenv("USED_CAR_DB_BACKEND", "sqlite").strip().lower() or "sqlite"
    if backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise RuntimeError(f"Unsupported history backend '{backend}'. Expected one of: {supported}.")
    return backend


def _get_db_path() -> Path:
    return Path(os.getenv("USED_CAR_DB_PATH", str(BASE_DIR / "used_car_price_app.db")))


def _get_supabase_config() -> tuple[str, str, str, float]:
    url = os.getenv("USED_CAR_SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("USED_CAR_SUPABASE_KEY", "").strip()
    table = os.getenv("USED_CAR_SUPABASE_TABLE", TABLE_NAME).strip() or TABLE_NAME
    timeout = float(os.getenv("USED_CAR_SUPABASE_TIMEOUT", "30"))
    if not url or not key:
        raise RuntimeError(
            "Supabase history backend requires USED_CAR_SUPABASE_URL and USED_CAR_SUPABASE_KEY."
        )
    return url, key, table, timeout


def get_storage_mode() -> str:
    return _get_backend()


def get_storage_label() -> str:
    return "Supabase" if _get_backend() == "supabase" else "SQLite"


def describe_storage() -> str:
    if _get_backend() == "supabase":
        return "Supabase Postgres via the REST API for persistent hosted prediction history."
    return "SQLite file storage for prediction history persistence."


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _normalize_created_at(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
            return parsed.isoformat(timespec="seconds")
        except ValueError:
            return value
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat(timespec="seconds")
    return datetime.now().isoformat(timespec="seconds")


def _normalize_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": _normalize_created_at(entry.get("createdAt")),
        "city": entry["city"],
        "yr_mfr": int(entry["yr_mfr"]),
        "kms_run": int(entry["kms_run"]),
        "fuel_type": entry["fuel_type"],
        "transmission": entry["transmission"],
        "body_type": entry["body_type"],
        "times_viewed": int(entry["times_viewed"]),
        "predicted_price": float(entry["predictedPrice"]),
        "confidence": entry["confidence"],
    }


def _history_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "createdAt": str(row["created_at"]),
        "city": row["city"],
        "yr_mfr": int(row["yr_mfr"]),
        "kms_run": int(row["kms_run"]),
        "fuel_type": row["fuel_type"],
        "transmission": row["transmission"],
        "body_type": row["body_type"],
        "times_viewed": int(row["times_viewed"]),
        "predictedPrice": float(row["predicted_price"]),
        "confidence": row["confidence"],
    }


def _supabase_headers(prefer: str | None = None) -> dict[str, str]:
    _, key, _, _ = _get_supabase_config()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_request(
    method: str,
    query_params: dict[str, Any] | None = None,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    prefer: str | None = None,
) -> Any:
    url, _, table, timeout = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{table}"
    if query_params:
        endpoint = f"{endpoint}?{urlencode(query_params)}"

    response = requests.request(
        method,
        endpoint,
        headers=_supabase_headers(prefer=prefer),
        json=payload,
        timeout=timeout,
    )
    if response.ok:
        if response.text.strip():
            return response.json()
        return None

    try:
        detail = response.json()
    except ValueError:
        detail = response.text.strip() or "Unknown Supabase error"
    raise RuntimeError(f"Supabase request failed ({response.status_code}): {detail}")


def ensure_database_ready() -> None:
    if _get_backend() == "supabase":
        _supabase_request("GET", {"select": "id", "limit": 1})
        return

    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    city TEXT NOT NULL,
                    yr_mfr INTEGER NOT NULL,
                    kms_run INTEGER NOT NULL,
                    fuel_type TEXT NOT NULL,
                    transmission TEXT NOT NULL,
                    body_type TEXT NOT NULL,
                    times_viewed INTEGER NOT NULL,
                    predicted_price REAL NOT NULL,
                    confidence TEXT NOT NULL
                )
                """
            )
        conn.commit()


def insert_prediction(entry: dict[str, Any]) -> None:
    record = _normalize_history_entry(entry)

    if _get_backend() == "supabase":
        _supabase_request("POST", payload=record, prefer="return=minimal")
        return

    ensure_database_ready()

    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                f"""
                INSERT INTO {TABLE_NAME} (
                    created_at,
                    city,
                    yr_mfr,
                    kms_run,
                    fuel_type,
                    transmission,
                    body_type,
                    times_viewed,
                    predicted_price,
                    confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["created_at"],
                    record["city"],
                    record["yr_mfr"],
                    record["kms_run"],
                    record["fuel_type"],
                    record["transmission"],
                    record["body_type"],
                    record["times_viewed"],
                    record["predicted_price"],
                    record["confidence"],
                ),
            )
        conn.commit()


def fetch_prediction_history(limit: int = 300) -> list[dict[str, Any]]:
    if _get_backend() == "supabase":
        rows = _supabase_request(
            "GET",
            {
                "select": "created_at,city,yr_mfr,kms_run,fuel_type,transmission,body_type,times_viewed,predicted_price,confidence",
                "order": "created_at.desc,id.desc",
                "limit": int(limit),
            },
        ) or []
        return [_history_row_to_dict(row) for row in rows]

    ensure_database_ready()
    with closing(_connect()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                f"""
                SELECT
                    created_at,
                    city,
                    yr_mfr,
                    kms_run,
                    fuel_type,
                    transmission,
                    body_type,
                    times_viewed,
                    predicted_price,
                    confidence
                FROM {TABLE_NAME}
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()

    return [_history_row_to_dict(row) for row in rows]
