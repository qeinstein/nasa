"""
FastAPI wrapper for connector.py 

- Loads models via connector.py functions.
- Calls visualize_input() and ensemble_predict() with unique IDs.
- Asynchronously generates explanations for classical, QML, and ensemble predictions.
- Returns predictions, base64-encoded plots, JSON outputs, and explanations.
- Handles 8 features for classical model and 10 features for QML model.
- prob_ensemble as percentage.
"""

import os
import io
import base64
import pandas as pd
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid
import json
import numpy as np

# Import everything from connector.py
import connector as ep

# Create app
app = FastAPI(title="Ensemble Exoplanet API (Wrapper)")

# Make sure outputs dir exists
OUTPUT_DIR = getattr(ep, "OUTPUT_DIR", "/home/toheebogunade/Workspace/NASA/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5173"], only specify for specific use case, do not chnage!1!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import math

def sanitize_json(data):
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # or round(0, 4)
    return data

# Load models once
print("Loading models using functions from connector.py ...")
try:
    clf, clf_scaler = ep.load_classical()
    qml_model, qml_scaler = ep.load_qml(n_qubits=6)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Failed to load models: {e}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

# Helper: encode a file to base64 (PNG)
def file_to_base64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    return None

# Helper: find output PNGs and JSON for a given unique_id
def collect_output_images(unique_id: str) -> Dict[str, str]:
    images = {}
    if not os.path.exists(OUTPUT_DIR):
        return images
    expected_files = [
        f"hist_{unique_id}_{col}.png" for col in ep.FEATURE_COLUMNS_CLASSICAL
    ] + [
        f"pairplot_{unique_id}.png",
        f"tsne_{unique_id}.png",
        f"umap_{unique_id}.png",
        f"explain_{unique_id}_shap_summary.png",
        f"explain_{unique_id}_qml_perm_importance.png",
        f"combined_importances_{unique_id}.png"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            try:
                images[fname] = file_to_base64(path)
            except Exception as e:
                print(f"Failed to encode {path}: {e}")
    return images

# Helper: read ensemble CSV for a given unique_id
def read_ensemble_csv(unique_id: str) -> pd.DataFrame:
    csv_path = os.path.join(OUTPUT_DIR, f"ensemble_predictions_{unique_id}.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read CSV {csv_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Helper: read JSON output for a given unique_id
def read_output_json(unique_id: str) -> Dict[str, Any]:
    json_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                print(f"[DEBUG] Loaded JSON keys: {list(data.keys())}")
                print(f"[DEBUG] Checking for NaN/inf in output JSON for {unique_id} ...")
                for k, v in data.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        print(f"⚠️  Invalid float found in {k}: {v}")
                return data
        except Exception as e:
            print(f"Failed to read JSON {json_path}: {e}")
    return None


async def generate_explanations(df_input, clf, clf_scaler, qml_model, qml_scaler, unique_id: str, n_qubits: int = 6) -> Dict[str, Any]:
    explanations = {
        "classical_explanation": "",
        "qml_explanation": "",
        "ensemble_explanation": ""
    }
    try:
        df_temp = df_input.copy()
        result = None
        # If no label present, call ensemble_predict to obtain predictions (it will also save outputs)
        if "label" not in df_temp.columns:
            result = ep.ensemble_predict(df_temp, clf, clf_scaler, qml_model, qml_scaler, unique_id=unique_id)
            df_temp["label"] = result["pred_ensemble"]
        else:
            # If result not produced yet, attempt a predict to get probabilities (without re-running visualizations)
            result = ep.ensemble_predict(df_temp, clf, clf_scaler, qml_model, qml_scaler, unique_id=unique_id)

        # SHAP for classical: pass feature names and outpath prefix
        X_scaled_classical = clf_scaler.transform(df_temp[ep.FEATURE_COLUMNS_CLASSICAL])
        explain_prefix = os.path.join(ep.OUTPUT_DIR, f"explain_{unique_id}")
        shap_arr = ep.explain_shap_rf(clf, X_scaled_classical, ep.FEATURE_COLUMNS_CLASSICAL, explain_prefix)
        if shap_arr is not None and result is not None:
            # compute concise per-feature contributions (use first row where possible)
            try:
                first_shap = np.array(shap_arr)
                # If it's (n_samples, n_features) pick the first sample row if present
                if first_shap.ndim == 2:
                    sample_shap = first_shap[0]
                elif first_shap.ndim == 3:
                    sample_shap = first_shap[0, 0, :]
                else:
                    sample_shap = first_shap.ravel()
                shap_contributions = []
                for i, feature in enumerate(ep.FEATURE_COLUMNS_CLASSICAL):
                    val = sample_shap[i] if i < len(sample_shap) else 0.0
                    shap_contributions.append(f"{feature}: {val:.4f} ({'increases' if val > 0 else 'decreases'})")
                explanations["classical_explanation"] = f"Classical model probability: {result['prob_classical'][0]:.4f}. Features: {', '.join(shap_contributions)}."
            except Exception as e:
                print(f"Failed to summarise SHAP contributions: {e}")

        # Permutation importance for QML (pass feature names and outpath_prefix)
        try:
            # For perm importance use scaled QML input (keep n_qubits columns)
            X_q_for_perm, _ = ep.preprocess_input(df_temp, scaler=qml_scaler, keep_n_for_qml=n_qubits, is_qml=True)
            # Use label values for AUC
            y_perm = df_temp["label"].values
            perm_importances = ep.explain_perm_qml(qml_model, X_q_for_perm, y_perm, ep.FEATURE_COLUMNS_QML[:n_qubits], explain_prefix)
            if perm_importances is not None and result is not None:
                perm_contributions = []
                for i, feature in enumerate(ep.FEATURE_COLUMNS_QML[:n_qubits]):
                    v = perm_importances[i] if i < len(perm_importances) else 0.0
                    perm_contributions.append(f"{feature}: {v:.4f} AUC drop when shuffled")
                explanations["qml_explanation"] = f"QML model probability: {result['prob_qml'][0]:.4f}. Features: {', '.join(perm_contributions)}."
        except Exception as e:
            print(f"QML explain failed inside generate_explanations: {e}")

        # Ensemble explanation
        if result is not None:
            ensemble_prob = float(result["prob_ensemble"][0]) * 100
            explanations["ensemble_explanation"] = (
                f"Ensemble probability: {ensemble_prob:.2f}% chance of being an exoplanet. "
                f"Combines classical (weight: 0.6, prob: {result['prob_classical'][0]:.4f}) and "
                f"QML (weight: 0.4, prob: {result['prob_qml'][0]:.4f}). "
                f"{'Not classified as exoplanet (probability < 50%)' if ensemble_prob < 50 else 'Classified as exoplanet (probability >= 50%)'}."
            )

    except Exception as e:
        print(f"Failed to generate explanations: {e}")
        explanations["error"] = str(e)
    return explanations


# Pydantic model for manual JSON single row input
class ManualRow(BaseModel):
    orb_period: float
    tran_dur: float
    tran_depth: float
    planet_radius: float
    eq_temp: float
    insolation: float
    star_temp: float
    star_radius: float
    label: float = None  # Optional label

# Endpoint: Single manual input
@app.post("/predict/manual")
async def predict_manual(row: ManualRow):
    """
    Send a single candidate as JSON. Calls visualize_input() and ensemble_predict()
    with a unique ID. Returns predictions, base64-encoded plots, and explanations.
    """
    df = pd.DataFrame([row.dict()])
    unique_id = str(uuid.uuid4())
    hist_prefix = os.path.join(OUTPUT_DIR, f"hist_{unique_id}_")
    pairplot_path = os.path.join(OUTPUT_DIR, f"pairplot_{unique_id}.png")
    tsne_path = os.path.join(OUTPUT_DIR, f"tsne_{unique_id}.png")
    umap_path = os.path.join(OUTPUT_DIR, f"umap_{unique_id}.png")

    # Call visualize_input
    try:
        ep.visualize_input(df, hist_prefix, pairplot_path, tsne_path, umap_path)
    except Exception as e:
        print(f"visualize_input() raised: {e}")

    # Call ensemble_predict
    try:
        result = ep.ensemble_predict(
            df, clf, clf_scaler, qml_model, qml_scaler,
            weight_classical=0.6, weight_qml=0.4, n_qubits=6, unique_id=unique_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ensemble_predict failed: {e}")

    # Generate explanations asynchronously
    explanations = await generate_explanations(df, clf, clf_scaler, qml_model, qml_scaler, unique_id)

    # Read JSON output from connector.py
    output_json = read_output_json(unique_id)
    if output_json is None:
        # Fallback: construct response
        preds_df = read_ensemble_csv(unique_id)
        preds_json = preds_df.to_dict(orient="records") if not preds_df.empty else result.to_dict(orient="records")
        # Convert prob_ensemble to percentage
        for pred in preds_json:
            pred["prob_ensemble"] = pred["prob_ensemble"] * 100
        images = collect_output_images(unique_id)
        output_json = {
            "predictions": preds_json,
            "shap_values": output_json.get("shap_values") if output_json else None,
            "perm_importances": output_json.get("perm_importances") if output_json else None,
            "visualizations": images,
            "predictions_csv_content": preds_df.to_csv(index=False) if not preds_df.empty else None,
            "explanations": explanations,
            "message": f"Predictions and plots produced for unique_id={unique_id}; plots are base64-encoded strings."
        }
        output_json = sanitize_json(output_json)

    return output_json

# Endpoint: Get SHAP summary plot for a specific unique_id
@app.get("/shap/{unique_id}")
def get_shap_plot(unique_id: str):
    """
    Retrieve the SHAP summary plot for a given unique_id.
    """
    shap_path = os.path.join(OUTPUT_DIR, f"explain_{unique_id}_shap_summary.png")
    if os.path.exists(shap_path):
        return FileResponse(shap_path, media_type="image/png", filename=f"shap_summary_{unique_id}.png")
    else:
        raise HTTPException(status_code=404, detail=f"SHAP summary plot not found for unique_id={unique_id}")

# Endpoint: List generated plots (filenames)
@app.get("/plots")
def list_plots():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".png")]
    return {"plots": files, "output_dir": OUTPUT_DIR}

# Endpoint: Return a specific plot (raw file)
@app.get("/plot/{filename}")
def get_plot(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Plot not found")

# Endpoint: Download predictions CSV for a specific unique_id
@app.get("/download/predictions/{unique_id}")
def download_predictions(unique_id: str):
    csv_path = os.path.join(OUTPUT_DIR, f"ensemble_predictions_{unique_id}.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename=f"ensemble_predictions_{unique_id}.csv")
    else:
        raise HTTPException(status_code=404, detail=f"Predictions CSV not found for unique_id={unique_id}")

# Endpoint: Download JSON output for a specific unique_id
@app.get("/download/json/{unique_id}")
def download_json(unique_id: str):
    json_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}.json")
    if os.path.exists(json_path):
        return FileResponse(json_path, media_type="application/json", filename=f"output_{unique_id}.json")
    else:
        raise HTTPException(status_code=404, detail=f"JSON output not found for unique_id={unique_id}")

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

# Simple root
@app.get("/")
def root():
    return {
        "message": "Wrapper for connector.py — use /predict/csv, /predict/manual, /shap/{unique_id}, "
                   "/download/predictions/{unique_id}, or /download/json/{unique_id}",
        "required_columns": ep.FEATURE_COLUMNS_CLASSICAL
    }

import aiohttp
import asyncio

# Function to ping the health endpoint(No longer needed since deployed on hugging face)
async def ping_server():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get('http://localhost:8000/health') as response:
                    if response.status == 200:
                        print(f"Ping successful at {asyncio.get_event_loop().time():.2f}: {await response.json()}")
                    else:
                        print(f"Ping failed with status {response.status}")
            except Exception as e:
                print(f"Ping error: {e}")
            await asyncio.sleep(150)  # Wait 2 minutes (120 seconds)

# Start the ping task when the FastAPI app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ping_server())
