import os
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
DROP_KEYWORDS = ("label", "isfraud", "isflaggedfraud", "islaundering", "is laundering")
ROLLING_WINDOW = 5
TIME_STEPS = 10
MAX_TRAINING_ROWS = int(os.getenv("ARGUS_MAX_TRAINING_ROWS", "20000"))
MAX_DIAGNOSTIC_ROWS = int(os.getenv("ARGUS_MAX_DIAGNOSTIC_ROWS", "20000"))


def prepare_telemetry_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Rename duplicated 'Account' column if Pandas hasn't already (e.g. Account, Account.1)
    cols = list(df.columns)
    account_count = 0
    for i, col in enumerate(cols):
        if col.strip() == "Account":
            account_count += 1
            if account_count == 2:
                cols[i] = "To_Account"
        cols[i] = cols[i].strip()
    df.columns = cols

    # Drop target columns if they exist
    cols_to_drop = [
        col for col in df.columns
        if any(keyword in str(col).lower() for keyword in DROP_KEYWORDS)
    ]
    data = df.drop(columns=cols_to_drop)
    
    # Sort by Timestamp if it exists
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        data = data.sort_values(by="Timestamp").reset_index(drop=True)

    # One-hot encode the 'Payment Format' column
    if "Payment Format" in data.columns:
        type_dummies = pd.get_dummies(data["Payment Format"], prefix="format")
        data = pd.concat([data, type_dummies], axis=1)

    # Calculate rolling features per Account (Transaction Velocity & Blind-Spot Fix)
    amount_col = "Amount Paid" if "Amount Paid" in data.columns else "amount"
    
    if "Account" in data.columns and amount_col in data.columns:
        data["amount_variance"] = data.groupby("Account")[amount_col].transform(lambda x: x.rolling(window=min(len(x), ROLLING_WINDOW), min_periods=1).var().fillna(0))
        # For velocity, if we have a Timestamp, we could do a rolling count. Since we sorted, we can just use rolling window count
        data["transaction_velocity"] = data.groupby("Account")[amount_col].transform(lambda x: x.rolling(window=min(len(x), ROLLING_WINDOW), min_periods=1).count().fillna(1))
    else:
        # Fallback if Account isn't present
        if amount_col in data.columns:
            data["amount_variance"] = data[amount_col].rolling(window=ROLLING_WINDOW, min_periods=1).var().fillna(0)

    # Drop string/identifier columns before feeding to the model
    ident_cols = ["Account", "To_Account", "Account.1", "From Bank", "To Bank", "Payment Format", "Timestamp", "Receiving Currency", "Payment Currency"]
    ident_cols = [c for c in ident_cols if c in data.columns]
    data = data.drop(columns=ident_cols)
    
    data = data.select_dtypes(include=[np.number])

    if data.empty:
        raise ValueError("No numeric features were found in the uploaded CSV after processing.")

    return data.fillna(0).astype("float32")


def create_sequence(dataset, time_steps: int = TIME_STEPS):
    dataset = np.asarray(dataset, dtype=np.float32)
    if len(dataset) <= time_steps:
        return np.empty((0, time_steps, dataset.shape[1]), dtype=np.float32)

    shape = (len(dataset) - time_steps, time_steps, dataset.shape[1])
    strides = (dataset.strides[0], dataset.strides[0], dataset.strides[1])
    return np.lib.stride_tricks.as_strided(dataset, shape=shape, strides=strides).copy()


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
    original_rows = len(df)
    if len(df) > MAX_TRAINING_ROWS:
        step = max(len(df) // MAX_TRAINING_ROWS, 1)
        df = df.iloc[::step].head(MAX_TRAINING_ROWS).copy()

    prepared = prepare_telemetry_frame(df)

    if len(prepared) <= TIME_STEPS:
        raise ValueError(
            f"Training data must contain more than {TIME_STEPS} rows after preprocessing."
        )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prepared).astype("float32")
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
        "raw_feature_count": len([c for c in prepared.columns if "_variance" not in c.lower()]),
        "rolling_window": ROLLING_WINDOW,
        "training_rows": len(prepared),
        "source_training_rows": original_rows,
        "max_training_rows": MAX_TRAINING_ROWS,
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
