import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


STORAGE_ROOT = Path(os.getenv("ARGUS_STORAGE_DIR", "ml"))
DB_PATH = Path(os.getenv("ARGUS_DB_PATH", STORAGE_ROOT / "argus_echo.db"))
SQLITE_TIMEOUT_SECONDS = int(os.getenv("ARGUS_SQLITE_TIMEOUT_SECONDS", "30"))


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                raw_feature_count INTEGER NOT NULL,
                expected_feature_count INTEGER NOT NULL,
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                training_rows INTEGER NOT NULL,
                training_file_path TEXT,
                model_path TEXT NOT NULL,
                scaler_path TEXT NOT NULL,
                config_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS diagnostic_runs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                uploaded_file_name TEXT NOT NULL,
                uploaded_file_path TEXT,
                result_path TEXT,
                anomaly_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS anomalies (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                primary_feature TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                mse_score REAL NOT NULL,
                variance REAL NOT NULL,
                attack_hint TEXT NOT NULL,
                echo_answer TEXT,
                logged_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (run_id) REFERENCES diagnostic_runs(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            );
            """
        )


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()
    return digest, salt


def create_user(email: str, password: str):
    email = email.strip().lower()
    password_hash, password_salt = hash_password(password)
    user_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, password_salt, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, email, password_hash, password_salt, utc_now()),
        )
    return {"id": user_id, "email": email}


def authenticate_user(email: str, password: str):
    email = email.strip().lower()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        return None

    password_hash, _ = hash_password(password, row["password_salt"])
    if password_hash != row["password_hash"]:
        return None
    return {"id": row["id"], "email": row["email"]}


def user_storage_dir(user_id: str) -> Path:
    path = STORAGE_ROOT / "users" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_storage_dir(user_id: str, model_id: str) -> Path:
    path = user_storage_dir(user_id) / "models" / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_storage_dir(user_id: str, run_id: str) -> Path:
    path = user_storage_dir(user_id) / "diagnostics" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def create_model_record(
    user_id: str,
    name: str,
    config: dict,
    training_file_path: Path,
    model_path: Path,
    scaler_path: Path,
    config_path: Path,
):
    model_id = model_path.parent.name
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (
                id, user_id, name, raw_feature_count, expected_feature_count, epochs,
                batch_size, training_rows, training_file_path, model_path, scaler_path,
                config_path, config_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                user_id,
                name,
                config["raw_feature_count"],
                config["num_features"],
                config["epochs"],
                config["batch_size"],
                config["training_rows"],
                str(training_file_path) if training_file_path else None,
                str(model_path),
                str(scaler_path),
                str(config_path),
                json.dumps(config),
                utc_now(),
            ),
        )
    return get_model(model_id, user_id)


def list_user_models(user_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM models
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_model(model_id: str, user_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ? AND user_id = ?",
            (model_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def create_diagnostic_run(
    user_id: str,
    model_id: str,
    uploaded_file_name: str,
    uploaded_file_path: Path,
    result_path: Path,
):
    run_id = result_path.parent.name
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO diagnostic_runs (
                id, user_id, model_id, uploaded_file_name, uploaded_file_path,
                result_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                user_id,
                model_id,
                uploaded_file_name,
                str(uploaded_file_path),
                str(result_path),
                utc_now(),
            ),
        )
    return run_id


def complete_diagnostic_run(run_id: str, user_id: str, anomaly_count: int):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE diagnostic_runs
            SET anomaly_count = ?
            WHERE id = ? AND user_id = ?
            """,
            (anomaly_count, run_id, user_id),
        )


def log_anomaly_record(
    user_id: str,
    run_id: str,
    model_id: str,
    primary_feature: str,
    entity_id: str,
    timestamp: int,
    mse_score: float,
    variance: float,
    attack_hint: str,
):
    anomaly_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO anomalies (
                id, user_id, run_id, model_id, primary_feature, entity_id, timestamp,
                mse_score, variance, attack_hint, logged_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anomaly_id,
                user_id,
                run_id,
                model_id,
                primary_feature,
                entity_id,
                int(timestamp),
                round(float(mse_score), 4),
                round(float(variance), 6),
                attack_hint,
                utc_now(),
            ),
        )
    return get_anomaly(anomaly_id, user_id)


def list_recent_anomalies(user_id: str, n: int = 5):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM anomalies
            WHERE user_id = ?
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (user_id, n),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_anomaly(user_id: str):
    rows = list_recent_anomalies(user_id, n=1)
    return rows[0] if rows else None


def get_anomaly(anomaly_id: str, user_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM anomalies WHERE id = ? AND user_id = ?",
            (anomaly_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def save_echo_answer(anomaly_id: str, user_id: str, answer: str):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE anomalies
            SET echo_answer = ?
            WHERE id = ? AND user_id = ?
            """,
            (answer, anomaly_id, user_id),
        )


def list_diagnostic_runs(user_id: str, n: int = 10):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT diagnostic_runs.*, models.name AS model_name
            FROM diagnostic_runs
            JOIN models ON models.id = diagnostic_runs.model_id
            WHERE diagnostic_runs.user_id = ?
            ORDER BY diagnostic_runs.created_at DESC
            LIMIT ?
            """,
            (user_id, n),
        ).fetchall()
    return [dict(row) for row in rows]
