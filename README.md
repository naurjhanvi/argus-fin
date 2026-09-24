# Argus Fin

Argus Fin is a transaction anomaly detection prototype for financial fraud and suspicious payment activity. It includes a Streamlit dashboard for training and reviewing anomaly models, an optional FastAPI prediction service, and Echo, a retrieval-augmented assistant for financial fraud guidance.

The detector is an LSTM autoencoder trained on transactions treated as normal. It learns a reconstruction baseline; elevated reconstruction error flags unusual activity for review. It is a prototype and should not be treated as a production fraud decision system or as a substitute for investigation.

## Features

- Create an account and keep model metadata, diagnostic runs, and anomaly history scoped to that account in the Streamlit app.
- Train a model by uploading a CSV of normal transactions.
- Analyze a CSV with a selected model, review anomaly scores and visualizations, and explore flagged entities and transaction relationships.
- Build Echo's local document index from the supported files in `data/` and ask for guidance using retrieved material and the Groq API.
- Optionally serve a JSON prediction endpoint with FastAPI when root-level model artifacts are available.

## Requirements

- Python 3.10 or newer (the Docker image uses Python 3.11).
- A virtual environment is recommended.
- A Groq API key is needed for Echo responses. The dashboard and detector can be used without Echo.

## Run locally

From the repository root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the dashboard:

```powershell
streamlit run app.py
```

Streamlit prints the local URL, usually `http://localhost:8501`. Create an account in the sidebar to use model training and diagnostics.

Optional environment variables:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
$env:ARGUS_STORAGE_DIR = "ml"
$env:ARGUS_DB_PATH = "ml\argus_echo.db"
$env:ARGUS_SQLITE_TIMEOUT_SECONDS = "30"
$env:ARGUS_MAX_TRAINING_ROWS = "20000"
$env:ARGUS_MAX_DIAGNOSTIC_ROWS = "20000"
```

The storage variables default to `ml/`, `ml/argus_echo.db`, and a 30-second SQLite timeout. Training and diagnostic row limits default to 20,000 each. The app reads at most the configured row limit from uploaded CSV files.

## Dashboard workflow

1. Open **Train Model** and upload a CSV containing normal transactions. Choose a model name, epoch count, and batch size, then train. The app saves the model, scaler, and feature configuration for the signed-in account.
2. Open **Run Diagnostics**, select a model, and upload transactions to analyze. Use the charts, anomaly details, and history to investigate unusual activity.
3. For Echo, provide `GROQ_API_KEY`, then build the knowledge base from `data/` in the **Echo Operator Guidance** tab. Ask Echo about a recent anomaly.

Training and diagnostic files need compatible transaction fields and feature layouts. Preprocessing drops fraud/label columns (including `IsLaundering` and `IsFraud` variants), sorts by `Timestamp` when present, one-hot encodes `Payment Format`, derives rolling amount variance and transaction velocity by account when possible, drops identifier/text fields, and retains numeric features. The selected model's saved feature configuration is used for inference; a materially different input schema may not be suitable for that model.

The `data/` directory contains the financial crime and fraud reference PDFs used by Echo. Add `.pdf`, `.txt`, or `.docx` documents there before building or rebuilding its index. The generated FAISS index is stored under the configured storage directory at `vectorstore/echo/`.

## FastAPI prediction service

`api.py` defines a separate FastAPI app. It loads these artifacts from the current working directory when the module starts:

```text
anomaly_detection_model.keras
scaler.pkl
model_config.pkl
```

Start it from the repository root with:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Check `GET /health` for service/model status. `POST /predict` accepts a JSON body of the form:

```json
{
  "transactions": [
    {"Timestamp": "2024-01-01T12:00:00Z", "Account": "A1", "Amount Paid": 125.5, "Payment Format": "ACH"}
  ]
}
```

The response contains `risk_score` (a heuristic 0–100 score based on the final reconstruction error), `raw_mae`, and `is_blocked` (true when the score exceeds 80). This endpoint is a prototype signal, not a calibrated probability or a production authorization decision. The FastAPI service uses root-level artifacts; it does not automatically select the per-account models saved by the Streamlit app.

## Model and data notes

- The autoencoder uses 10-row sequences and a rolling feature window of 5.
- The default training and diagnostics row limits are 20,000. Override them with `ARGUS_MAX_TRAINING_ROWS` and `ARGUS_MAX_DIAGNOSTIC_ROWS`.
- The app's SQLite database and user model/run files are stored under `ARGUS_STORAGE_DIR` (default `ml/`). Keep this directory persistent if you need to retain local data.
- `train_model.py` is an auxiliary command-line training script; the supported workflow documented here is model training in the Streamlit app.
- `split_dataset.py` is a local dataset preparation helper with machine-specific input/output paths; edit those paths before using it.

## Docker

Build and run the Streamlit dashboard container:

```powershell
docker build -t argus-fin .
docker run --rm -p 8501:8501 -v "${PWD}/ml:/app/ml" -e GROQ_API_KEY="your_groq_api_key" argus-fin
```

The image starts Streamlit on port 8501 and includes a Streamlit health check. The volume mount preserves app storage between container runs. The Docker image installs `requirements.runtime.txt`, which does not include FastAPI or Uvicorn; run the API locally from an environment installed with `requirements.txt`, or add those packages to the runtime requirements before building an API container.

## Repository layout

```text
app.py                  Streamlit dashboard and user workflows
api.py                  Optional FastAPI prediction endpoint
argus_model.py          Transaction preprocessing, model training, artifact helpers
persistence.py          SQLite accounts, model metadata, runs, and anomaly history
argus_logger.py         Anomaly hint helper and legacy JSON logger
graph_utils.py          Transaction relationship graph utilities
echo/                   Document ingestion, retrieval, prompting, and evaluation helpers
data/                   Reference documents for Echo
k8s/                    Kubernetes manifests (currently named/configured for argus-echo)
.streamlit/config.toml  Streamlit server settings
Dockerfile              Streamlit container image
requirements.txt        Full development dependencies, including API/evaluation packages
requirements.runtime.txt Runtime dependencies used by the Docker image
```

## Deployment manifests

The files in `k8s/` are inherited deployment manifests named for `argus-echo` and currently reference its Google Cloud project, registry image, namespace, and service. They are not a ready-to-apply deployment configuration for Argus Fin. Update those values and validate the storage, secrets, and deployment settings for your environment before using them.

## Technology

Python, Streamlit, FastAPI, TensorFlow/Keras, scikit-learn, pandas, NumPy, Plotly, SQLite, FAISS, LangChain, Groq, and Docker.
