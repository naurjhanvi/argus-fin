from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential


MODEL_PATH = Path("anomaly_detection_model.keras")
SCALER_PATH = Path("scaler.pkl")
CONFIG_PATH = Path("model_config.pkl")
DROP_KEYWORDS = ("time", "timestamp", "date", "label")
ROLLING_WINDOW = 5
TIME_STEPS = 10


def prepare_telemetry_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        col for col in df.columns if any(keyword in col.lower() for keyword in DROP_KEYWORDS)
    ]
    data = df.drop(columns=cols_to_drop)
    data = data.select_dtypes(include=[np.number])

    if data.empty:
        raise ValueError("No numeric telemetry columns were found in the uploaded CSV.")

    original_cols = list(data.columns)
    for col in original_cols:
        data[f"{col}_variance"] = data[col].rolling(window=ROLLING_WINDOW).var()

    return data.fillna(0)


def create_sequence(dataset, time_steps: int = TIME_STEPS):
    sequence = []
    for i in range(len(dataset) - time_steps):
        sequence.append(dataset[i : (i + time_steps)])
    return np.array(sequence)


def build_model(time_steps: int, num_features: int):
    model = Sequential(
        [
            Input(shape=(time_steps, num_features)),
            LSTM(
                16,
                activation="relu",
                return_sequences=False,
            ),
            RepeatVector(time_steps),
            LSTM(16, activation="relu", return_sequences=True),
            TimeDistributed(Dense(num_features)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_from_dataframe(
    df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    prepared = prepare_telemetry_frame(df)

    if len(prepared) <= TIME_STEPS:
        raise ValueError(
            f"Training data must contain more than {TIME_STEPS} rows after preprocessing."
        )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prepared)
    x_train = create_sequence(scaled_data, TIME_STEPS)
    num_features = x_train.shape[2]

    tf.keras.backend.clear_session()
    model = build_model(TIME_STEPS, num_features)
    history = model.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
    )

    config = {
        "time_steps": TIME_STEPS,
        "num_features": num_features,
        "feature_names": list(prepared.columns),
        "raw_sensor_count": len([c for c in prepared.columns if "_variance" not in c.lower()]),
        "rolling_window": ROLLING_WINDOW,
        "training_rows": len(prepared),
        "epochs": epochs,
        "batch_size": batch_size,
    }

    return model, scaler, config, history


def save_artifacts(
    model,
    scaler,
    config,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    config_path: Path = CONFIG_PATH,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(config, config_path)


def missing_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    config_path: Path = CONFIG_PATH,
):
    return [path for path in (model_path, scaler_path, config_path) if not path.exists()]
