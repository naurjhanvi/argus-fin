import os
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf

from argus_model import (
    prepare_telemetry_frame,
    create_sequence,
    CONFIG_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
import joblib

app = FastAPI(title="Argus Fin API", description="Fraud Detection API for Razorpay Hackathon")

# We will load the system globally if it exists
model = None
scaler = None
config = None

def load_system():
    global model, scaler, config
    if MODEL_PATH.exists() and SCALER_PATH.exists() and CONFIG_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH))
        scaler = joblib.load(SCALER_PATH)
        config = joblib.load(CONFIG_PATH)

load_system()


class TransactionList(BaseModel):
    transactions: list[dict]


@app.post("/predict")
async def predict(data: TransactionList):
    if model is None or scaler is None or config is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found. Train the model first.")

    df = pd.DataFrame(data.transactions)
    if df.empty:
        raise HTTPException(status_code=400, detail="No transactions provided")

    try:
        process_df = prepare_telemetry_frame(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    expected_features = config["num_features"]
    time_steps = config["time_steps"]

    if process_df.shape[1] != expected_features:
        # In a real API we might pad missing one-hot features with 0, but for simplicity here we assume a match
        # Let's dynamically add missing columns to avoid API crashes if a type was missing
        for col in config["feature_names"]:
            if col not in process_df.columns:
                process_df[col] = 0
        
        process_df = process_df[config["feature_names"]]
        
    scaled_data = scaler.transform(process_df)
    
    # If we have less than time_steps, we pad with zeros
    if len(scaled_data) < time_steps:
        padding = np.zeros((time_steps - len(scaled_data), scaled_data.shape[1]))
        scaled_data = np.vstack((padding, scaled_data))
        
    x_input = create_sequence(scaled_data, time_steps=time_steps)
    
    if len(x_input) == 0:
        # Fallback if somehow sequence creation fails
        raise HTTPException(status_code=500, detail="Failed to create sequence for model inference")

    predictions = model.predict(x_input, batch_size=256, verbose=0)
    mae_loss = np.mean(np.abs(predictions - x_input), axis=(1, 2))
    
    # Pad loss to match original dataframe length (since create_sequence drops first time_steps - 1)
    # The last loss corresponds to the last transaction
    padded_loss = np.concatenate([np.zeros(time_steps), mae_loss])
    
    # We want to return a normalized risk score (0-100) for the last transaction
    last_loss = padded_loss[-1]
    
    # Normalize the loss (a heuristic max loss can be 1.0 based on scaled data, but let's cap at 1.0)
    max_expected_loss = 0.5 
    risk_score = min((last_loss / max_expected_loss) * 100, 100.0)
    
    return {
        "risk_score": float(risk_score),
        "raw_mae": float(last_loss),
        "is_blocked": risk_score > 80.0
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
