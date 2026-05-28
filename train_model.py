import os
import sys

import pandas as pd

from argus_model import save_artifacts, train_from_dataframe


if len(sys.argv) < 2:
    print("Error: Missing dataset argument.")
    print("Usage: python train_model.py <dataset.csv>")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    sys.exit(1)

print(f"Initializing Edge AI Pipeline for: {file_path}")
df = pd.read_csv(file_path)

print("Training Lightweight Edge AI Architecture...")
model, scaler, config, history = train_from_dataframe(df, epochs=50)
save_artifacts(model, scaler, config)

final_loss = history.history["loss"][-1]
print(f"Detected {config['raw_sensor_count']} raw sensors.")
print(f"Feature engineering complete. Total inputs per timestep: {config['num_features']}")
print(f"Final training loss: {final_loss:.6f}")
print("Pipeline Complete! Model, Scaler, and Config saved successfully.")
