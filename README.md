# Argus Echo

### Edge AI Anomaly Detection + RAG-Powered Operator Guidance for ICS Security

**Argus detects. Echo explains.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-purple)](https://groq.com)

---

## Overview

Argus Echo is a domain-agnostic Edge AI diagnostic tool for safety-critical industrial telemetry. It combines:

- **Argus**, an LSTM autoencoder anomaly detector that learns normal telemetry behavior.
- **Echo**, a RAG-powered operator guidance layer grounded in ICS security documents.

The project was designed around a key ICS security problem: replay attacks can freeze or spoof sensor output so the control room sees normal-looking telemetry while the physical system is behaving abnormally. Standard time-series models can miss this because a flatline can have very low reconstruction error.

Argus addresses this by adding deterministic rolling variance features before model inference. When a normally dynamic industrial signal becomes suspiciously static, the variance channel helps expose the anomaly.

---

## Current App Flow

The current project is meant to be used through the Streamlit dashboard.

```bash
streamlit run app.py
```

After the app opens:

1. Create an account or log in from the sidebar.
2. Open the **Train Model** tab.
3. Upload a **training CSV** containing normal-operation telemetry.
4. Click **Train Model**.
5. Argus trains a model for that sensor layout and saves it to the current account.
6. Open the **Run Diagnostics** tab.
7. Upload a **testing CSV** containing telemetry to analyze.
8. Click **Run Diagnostics**.
9. The diagnosis appears in the dashboard with anomaly scores, plotted sensor traces, logged anomalies, and recent diagnostic history.

The training CSV and testing CSV should have the same sensor layout. If the testing CSV has a different number of numeric telemetry signals, the app will show a shape mismatch and ask you to train a model for that layout.

The model automatically reads the shape of the uploaded CSV. For example, 87 raw HAI sensors become 174 model inputs after Argus adds rolling variance features. No hardcoded sensor names are required.

---

## Echo Guidance

After diagnostics logs an anomaly, open the **Echo Guidance** tab to generate operator guidance.

If the Echo knowledge base has not been built yet, click **Build Knowledge Base From data/** inside the app. Echo indexes the ICS security PDFs in `data/` and uses them to ground its explanation.

Echo requires a Groq API key. Add it through `.env`:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

You can also paste the key into the Echo Guidance tab during the app session.

---

## Local Setup

### Prerequisites

- Python 3.10+
- Groq API key for Echo guidance
- Training and testing telemetry CSV files

### Installation

```bash
git clone https://github.com/naurjhanvi/Argus_Echo
cd Argus_Echo

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Start the Dashboard

```bash
streamlit run app.py
```

The app stores user accounts, uploaded files, trained models, diagnostic runs, and anomaly history under `ml/` by default.

---

## Project Structure

```text
Argus_Echo/
|-- app.py                         # Streamlit dashboard
|-- argus_model.py                 # Training, preprocessing, and model artifact helpers
|-- persistence.py                 # User, model, diagnostic, and anomaly storage
|-- train_model.py                 # Optional CLI training script
|-- argus_logger.py                # Anomaly hinting helpers
|-- requirements.txt               # Full local dependencies
|-- requirements.runtime.txt       # Runtime-focused dependencies
|-- data/                          # ICS security corpus PDFs
|-- echo/
|   |-- ingest.py                  # Builds the FAISS knowledge base
|   |-- rag.py                     # Echo RAG chain
|   |-- query_builder.py           # Converts anomaly records into Echo prompts
|   |-- evaluate.py                # RAG evaluation utilities
```

---

## Technical Approach

Argus uses an LSTM autoencoder for reconstruction-based anomaly detection.

Before sequence creation, every numeric sensor column is augmented with a rolling variance feature:

```python
sensor_variance = sensor_reading.rolling(window=5).var()
```

This creates two feature groups:

- raw telemetry readings
- rolling variance signals

The model learns normal behavior from the training CSV. During diagnostics, it reconstructs the testing CSV sequences and computes reconstruction error. When the anomaly score crosses the threshold, the app logs the anomaly and makes it available to Echo for explanation.

---

## ICS Security Corpus

Echo's interpretation quality depends on the quality of the documents in `data/`.

Included corpus examples:

- NIST SP 800-82 Rev 3 - ICS Security Guide
- LSTM anomaly detection research papers
- Kravchik & Shabtai ICS attack detection paper
- Project Argus technical report

---

*Argus Echo demonstrates the integration of Edge AI anomaly detection and RAG-based operator guidance for safety-critical ICS environments.*
