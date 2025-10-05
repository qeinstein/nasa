import os
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import base64
import uuid

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.inspection import permutation_importance
import shap
from sklearn.ensemble import RandomForestClassifier

# Suppress scikit-learn warnings for version mismatches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------
# USER PATHS - edit these
# -------------------------
CLASSICAL_MODEL_PATH = "nasa_model_finalClassical1.pkl"
CLASSICAL_SCALER_PATH = "nasa_scalerfinalClassical1.pkl"
QML_MODEL_PATH = "nasa_qml_model1.pt"
QML_SCALER_PATH = "nasa_qml_scaler1.pkl"
OUTPUT_DIR = "/home/toheebogunade/Workspace/NASA/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utility: load classical model + scaler
# -------------------------
def load_classical():
    try:
        print(f"Loading classical model from: {os.path.abspath(CLASSICAL_MODEL_PATH)}")
        print(f"Loading scaler from: {os.path.abspath(CLASSICAL_SCALER_PATH)}")
        clf = joblib.load(CLASSICAL_MODEL_PATH)
        scaler = joblib.load(CLASSICAL_SCALER_PATH) if os.path.exists(CLASSICAL_SCALER_PATH) else None
        if scaler is not None:
            print(f"Scaler expects {scaler.n_features_in_} features: {getattr(scaler, 'feature_names_in_', 'Not available')}")
            if scaler.n_features_in_ != len(FEATURE_COLUMNS_CLASSICAL):
                raise ValueError(f"Classical scaler expects {scaler.n_features_in_} features, but FEATURE_COLUMNS_CLASSICAL has {len(FEATURE_COLUMNS_CLASSICAL)}")
        return clf, scaler
    except Exception as e:
        print(f"Failed to load classical model/scaler: {e}")
        raise

# -------------------------
# Utility: load QML model & restore PyTorch module architecture
# -------------------------
import pennylane as qml
from pennylane import numpy as pnp
import torch.nn as nn
import torch.nn.functional as F

def build_qml_model(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (2, n_qubits)}  # Matches training: n_q_layers=2
    q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    class QMLHybrid(nn.Module):
        def __init__(self, q_layer, n_qubits):
            super().__init__()
            self.pre = nn.Sequential(
                nn.Linear(n_qubits, n_qubits),
                nn.ReLU()
            )
            self.q_layer = q_layer
            self.post = nn.Sequential(
                nn.Linear(n_qubits, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            x = self.pre(x)
            x = self.q_layer(x)
            x = self.post(x)
            return x

    return QMLHybrid(q_layer, n_qubits)

def load_qml(n_qubits):
    try:
        print(f"Loading QML model from: {os.path.abspath(QML_MODEL_PATH)}")
        print(f"Loading QML scaler from: {os.path.abspath(QML_SCALER_PATH)}")
        model = build_qml_model(n_qubits)
        model.load_state_dict(torch.load(QML_MODEL_PATH, map_location="cpu"))
        model.eval()
        scaler = joblib.load(QML_SCALER_PATH) if os.path.exists(QML_SCALER_PATH) else None
        if scaler is not None:
            print(f"QML scaler expects {scaler.n_features_in_} features: {getattr(scaler, 'feature_names_in_', 'Not available')}")
            if scaler.n_features_in_ != len(FEATURE_COLUMNS_QML):
                raise ValueError(f"QML scaler expects {scaler.n_features_in_} features, but FEATURE_COLUMNS_QML has {len(FEATURE_COLUMNS_QML)}")
        return model, scaler
    except Exception as e:
        print(f"Failed to load QML model/scaler: {e}")
        raise

# -------------------------
# Preprocessing: ensure feature order matches training
# -------------------------
FEATURE_COLUMNS_CLASSICAL = [
    "orb_period", "tran_dur", "tran_depth",
    "planet_radius", "eq_temp", "insolation",
    "star_temp", "star_radius"
]
FEATURE_COLUMNS_QML = [
    "orb_period", "tran_dur", "tran_depth", "planet_radius",
    "eq_temp", "insolation", "star_temp", "star_radius",
    "radius_to_star", "depth_per_radius"
]

def preprocess_input(df, scaler=None, keep_n_for_qml=None, is_qml=False):
    try:
        feature_columns = FEATURE_COLUMNS_QML if is_qml else FEATURE_COLUMNS_CLASSICAL
        # Add engineered features
        df = df.copy()
        df["radius_to_star"] = df["planet_radius"] / (df["star_radius"] + 1e-9)
        df["depth_per_radius"] = df["tran_depth"] / (df["planet_radius"] + 1e-9)
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input missing columns: {missing_cols}")
        X = df[feature_columns].astype(float)
        if scaler is not None:
            if scaler.n_features_in_ != len(feature_columns):
                raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but input has {len(feature_columns)}")
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        if keep_n_for_qml is not None:
            X_q = X_scaled[:, :keep_n_for_qml]
            return X_q, X_scaled
        return None, X_scaled
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise

# -------------------------
# Model inference helpers
# -------------------------
def predict_classical(clf, X_scaled):
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_scaled)[:, 1]
        return probs
    else:
        preds = clf.predict(X_scaled)
        return preds

def predict_qml(qml_model, X_q):
    with torch.no_grad():
        X_t = torch.tensor(X_q, dtype=torch.float32)
        logits = qml_model(X_t).cpu().numpy().reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return probs


def explain_shap_rf(clf, X_sample, feature_names, outpath_prefix):
    """
    Generate SHAP explanations for RandomForest models (handles multiclass robustly).
    Returns a 1D array of mean absolute SHAP values for each feature.
    """
    shap_array = None
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)

        # --- Step 1: Normalize to 2D numpy array ---
        if isinstance(shap_values, list):
            # Multi-class output -> average absolute shap across classes
            shap_array = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
        else:
            shap_array = np.array(shap_values)

        # --- Step 2: Ensure proper shape (n_samples, n_features) ---
        if shap_array.ndim == 1:
            shap_array = shap_array.reshape(-1, 1)
        elif shap_array.ndim == 3:
            shap_array = np.mean(np.abs(shap_array), axis=0)

        # --- Step 3: Align with X_sample safely ---
        n_samples = min(X_sample.shape[0], shap_array.shape[0])
        n_features = min(X_sample.shape[1], shap_array.shape[1])
        shap_array = shap_array[:n_samples, :n_features]
        X_trimmed = X_sample[:n_samples, :n_features]

        # --- Step 4: Compute mean absolute SHAP values for combined plot ---
        mean_shap = np.abs(shap_array).mean(axis=0)
        mean_shap = np.asarray(mean_shap).ravel()

        # --- Step 5: Plot SHAP summary safely ---
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_array,
            X_trimmed,
            feature_names=feature_names[:n_features],
            show=False
        )
        plt.tight_layout()
        shap_summary_path = f"{outpath_prefix}_shap_summary.png"
        plt.savefig(shap_summary_path, dpi=150)
        plt.close()

        print(f"SHAP summary saved successfully: {shap_summary_path}")
        return mean_shap

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        # Debugging info
        try:
            print("X_sample shape:", getattr(X_sample, "shape", None))
            print("shap_values type:", type(shap_values))
            print("shap_values shape:", getattr(np.array(shap_values), "shape", None))
        except Exception:
            pass
        # Return zeros array to prevent downstream shape errors
        return np.zeros(X_sample.shape[1])





# ---------- connector.py (replace explain_perm_qml) ----------
def explain_perm_qml(qml_model, X_val, y_val, feature_names, outpath_prefix):
    try:
        # Prediction wrapper for QML
        def qml_predict_proba(X_subset):
            return predict_qml(qml_model, X_subset)

        from sklearn.metrics import roc_auc_score
        # Ensure y_val is proper binary/numeric array
        y_arr = np.asarray(y_val).astype(float)
        X_arr = np.asarray(X_val).astype(float)

        if len(np.unique(y_arr)) < 2:
            # Can't compute AUC if y has only one class; return zeros
            importances = np.zeros(X_arr.shape[1])
        else:
            baseline = roc_auc_score(y_arr, qml_predict_proba(X_arr))
            importances = []
            rng = np.random.default_rng(0)
            for j in range(X_arr.shape[1]):
                Xp = X_arr.copy()
                rng.shuffle(Xp[:, j])
                score = roc_auc_score(y_arr, qml_predict_proba(Xp))
                importances.append(baseline - score)
            importances = np.array(importances)

        # Plot with matplotlib for robustness (avoid seaborn format issues)
        fig, ax = plt.subplots(figsize=(8, 0.6 * max(4, len(feature_names))))
        y_pos = np.arange(len(feature_names[:X_arr.shape[1]]))
        ax.barh(y_pos, importances[:len(y_pos)], height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names[:len(y_pos)])
        ax.set_xlabel("Decrease in AUC when permuted")
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_qml_perm_importance.png", dpi=150)
        plt.close()
        return importances
    except Exception as e:
        print(f"QML permutation importance failed: {e}")
        return None


# -------------------------
# Helper to encode image to base64
# -------------------------
def encode_image_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode("utf-8")
        return base64_str
    return None

# -------------------------
# Ensemble function
# -------------------------
def ensemble_predict(
    df_input,
    clf,
    clf_scaler,
    qml_model,
    qml_scaler,
    weight_classical=0.6,
    weight_qml=0.4,
    n_qubits=6,
    unique_id=str(uuid.uuid4())
):
    try:
        # --- Preprocess inputs ---
        _, X_scaled_classical = preprocess_input(
            df_input, scaler=clf_scaler, keep_n_for_qml=None, is_qml=False
        )
        X_q, X_scaled_qml = preprocess_input(
            df_input, scaler=qml_scaler, keep_n_for_qml=n_qubits, is_qml=True
        )

        # --- Make predictions ---
        p_clf = predict_classical(clf, X_scaled_classical)
        p_qml = predict_qml(qml_model, X_q) if qml_model is not None else np.zeros(len(df_input))
        p_ensemble = weight_classical * p_clf + weight_qml * p_qml
        preds = (p_ensemble >= 0.5).astype(int)

        # --- Prepare output dataframe ---
        out = df_input.copy().reset_index(drop=True)
        out["prob_classical"] = p_clf
        out["prob_qml"] = p_qml
        out["prob_ensemble"] = p_ensemble
        out["pred_ensemble"] = preds

        # --- Paths ---
        explain_prefix = os.path.join(OUTPUT_DIR, f"explain_{unique_id}")
        hist_prefix = os.path.join(OUTPUT_DIR, f"hist_{unique_id}_")
        pairplot_path = os.path.join(OUTPUT_DIR, f"pairplot_{unique_id}.png")
        tsne_path = os.path.join(OUTPUT_DIR, f"tsne_{unique_id}.png")
        umap_path = os.path.join(OUTPUT_DIR, f"umap_{unique_id}.png")
        combined_path = os.path.join(OUTPUT_DIR, f"combined_importances_{unique_id}.png")
        predictions_path = os.path.join(OUTPUT_DIR, f"ensemble_predictions_{unique_id}.csv")

        # --- Visualizations ---
        visualize_input(df_input, hist_prefix, pairplot_path, tsne_path, umap_path)

        shap_vals = None
        perm_importances = None
        shap_summary_path = f"{explain_prefix}_shap_summary.png"
        qml_perm_path = f"{explain_prefix}_qml_perm_importance.png"

        if "label" in df_input.columns:
            sample_X = X_scaled_classical[:min(200, X_scaled_classical.shape[0])]
            shap_vals = explain_shap_rf(clf, sample_X, FEATURE_COLUMNS_CLASSICAL, explain_prefix)

            # QML permutation importance
            if len(X_q) >= 200:
                X_perm = X_q[:200]
                y_perm = df_input["label"].values[:200]
            else:
                X_perm, y_perm = X_q, df_input["label"].values
            perm_importances = explain_perm_qml(qml_model, X_perm, y_perm, FEATURE_COLUMNS_QML[:n_qubits], explain_prefix)

        # --- Combined importance plot ---
        try:
            fig, ax = plt.subplots(figsize=(8,5))

            # SHAP
            if shap_vals is not None:
                mean_shap = np.abs(shap_vals).mean(axis=0)
                mean_shap = np.ravel(mean_shap)
                if len(mean_shap) < n_qubits:
                    mean_shap = np.pad(mean_shap, (0, n_qubits - len(mean_shap)), 'constant')
                else:
                    mean_shap = mean_shap[:n_qubits]
            else:
                mean_shap = np.zeros(n_qubits)

            # Perm importance
            if perm_importances is not None:
                perm_importances = np.ravel(perm_importances)
                if len(perm_importances) < n_qubits:
                    perm_importances = np.pad(perm_importances, (0, n_qubits - len(perm_importances)), 'constant')
                else:
                    perm_importances = perm_importances[:n_qubits]
            else:
                perm_importances = np.zeros(n_qubits)

            ind = np.arange(n_qubits)
            width = 0.35
            ax.barh(ind - width/2, mean_shap, height=width, label="RF | mean |SHAP|")
            ax.barh(ind + width/2, perm_importances, height=width, label="QML | perm importance")
            ax.set_yticks(ind)
            ax.set_yticklabels(FEATURE_COLUMNS_QML[:n_qubits])
            ax.legend()
            plt.tight_layout()
            plt.savefig(combined_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Combined importance plot failed: {e}")

        # --- Encode plots to base64 ---
        hist_base64 = {col: encode_image_to_base64(f"{hist_prefix}{col}.png") for col in FEATURE_COLUMNS_CLASSICAL}
        pairplot_base64 = encode_image_to_base64(pairplot_path)
        tsne_base64 = encode_image_to_base64(tsne_path)
        umap_base64 = encode_image_to_base64(umap_path)
        shap_summary_base64 = encode_image_to_base64(shap_summary_path) if shap_vals is not None else None
        qml_perm_base64 = encode_image_to_base64(qml_perm_path) if perm_importances is not None else None
        combined_base64 = encode_image_to_base64(combined_path)

        # --- Save predictions ---
        out.to_csv(predictions_path, index=False)
        with open(predictions_path, "r") as f:
            csv_content = f.read()

        output_json = {
            "predictions": out.to_dict(orient="records"),
            "shap_values": shap_vals.tolist() if shap_vals is not None else None,
            "perm_importances": perm_importances.tolist() if perm_importances is not None else None,
            "visualizations": {
                "histograms": hist_base64,
                "pairplot": pairplot_base64,
                "tsne": tsne_base64,
                "umap": umap_base64,
                "shap_summary": shap_summary_base64,
                "qml_perm_importance": qml_perm_base64,
                "combined_importances": combined_base64
            },
            "predictions_csv_content": csv_content
        }
        json_path = os.path.join(OUTPUT_DIR, f"output_{unique_id}.json")
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)

        return out

    except Exception as e:
        print(f"Ensemble prediction failed: {e}")
        raise


# -------------------------
# Visualizations of uploaded data
# -------------------------
def visualize_input(df_input, hist_prefix, pairplot_path, tsne_path, umap_path):
    try:
        for col in FEATURE_COLUMNS_CLASSICAL:
            plt.figure(figsize=(6,3))
            sns.histplot(df_input[col].dropna(), kde=True)
            plt.title(col)
            plt.tight_layout()
            plt.savefig(f"{hist_prefix}{col}.png", dpi=150)
            plt.close()

        sample = df_input[FEATURE_COLUMNS_CLASSICAL].sample(n=min(500, len(df_input))).reset_index(drop=True)
        sns.pairplot(sample)
        plt.savefig(pairplot_path, dpi=150)
        plt.close()

        X = df_input[FEATURE_COLUMNS_CLASSICAL].fillna(0).values
        if X.shape[0] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            emb = tsne.fit_transform(X[:1000]) if X.shape[0] > 1000 else tsne.fit_transform(X)
            plt.figure(figsize=(6,5)); plt.scatter(emb[:,0], emb[:,1], s=6); plt.title("t-SNE"); plt.tight_layout()
            plt.savefig(tsne_path, dpi=150); plt.close()

            reducer = umap.UMAP(random_state=42)
            emb2 = reducer.fit_transform(X[:2000]) if X.shape[0] > 2000 else reducer.fit_transform(X)
            plt.figure(figsize=(6,5)); plt.scatter(emb2[:,0], emb2[:,1], s=6); plt.title("UMAP"); plt.tight_layout()
            plt.savefig(umap_path, dpi=150); plt.close()
    except Exception as e:
        print(f"Visualization failed: {e}")
        raise

# -------------------------
# Main CLI
# -------------------------
def main(args):
    clf, clf_scaler = load_classical()
    qml_model, qml_scaler = load_qml(n_qubits=6)

    try:
        df = pd.read_csv(args.input)
        print("Input loaded:", df.shape)
    except Exception as e:
        print(f"Failed to load input CSV: {e}")
        raise

    missing_cols = set(FEATURE_COLUMNS_CLASSICAL) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")

    unique_id = str(uuid.uuid4())
    hist_prefix = os.path.join(OUTPUT_DIR, f"hist_{unique_id}_")
    pairplot_path = os.path.join(OUTPUT_DIR, f"pairplot_{unique_id}.png")
    tsne_path = os.path.join(OUTPUT_DIR, f"tsne_{unique_id}.png")
    umap_path = os.path.join(OUTPUT_DIR, f"umap_{unique_id}.png")

    visualize_input(df, hist_prefix, pairplot_path, tsne_path, umap_path)
    result = ensemble_predict(df, clf, clf_scaler, qml_model, qml_scaler, weight_classical=0.6, weight_qml=0.4, n_qubits=6, unique_id=unique_id)
    print("Saved ensemble predictions, explanation plots, and JSON output to", OUTPUT_DIR)
    print(result.head())

# -------------------------
# Quick test / sandbox
# -------------------------
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Ensemble exoplanet predictions.")
    parser.add_argument("--input", type=str, help="Path to input CSV file.")
    args = parser.parse_args()

    if args.input:
        main(args)
    else:
        print("Running quick test...")
        clf, clf_scaler = load_classical()
        qml_model, qml_scaler = load_qml(n_qubits=6)

        test_data = pd.DataFrame({
            "orb_period": [365, 10],
            "tran_dur": [10, 0.5],
            "tran_depth": [0.01, 0.0005],
            "planet_radius": [1.0, 0.1],
            "eq_temp": [288, 1200],
            "insolation": [1, 10],
            "star_temp": [5800, 4500],
            "star_radius": [1.0, 0.5]
        })

        print("Test input:\n", test_data)
        unique_id = str(uuid.uuid4())
        hist_prefix = os.path.join(OUTPUT_DIR, f"hist_{unique_id}_")
        pairplot_path = os.path.join(OUTPUT_DIR, f"pairplot_{unique_id}.png")
        tsne_path = os.path.join(OUTPUT_DIR, f"tsne_{unique_id}.png")
        umap_path = os.path.join(OUTPUT_DIR, f"umap_{unique_id}.png")

        visualize_input(test_data, hist_prefix, pairplot_path, tsne_path, umap_path)
        result = ensemble_predict(
            test_data, 
            clf, clf_scaler, 
            qml_model, qml_scaler, 
            weight_classical=0.6, weight_qml=0.4, n_qubits=6, unique_id=unique_id
        )
        print("Ensemble predictions:\n", result)
        print("Outputs saved to", OUTPUT_DIR)