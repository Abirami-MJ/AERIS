# ============================================================
# AERIS Project — SHAP Explainability for CMAPSS Transformer
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import math
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. RE-DEFINE MODEL ARCHITECTURE (must match training exactly)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RULTransformer(nn.Module):
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        pooled = x.mean(dim=1)
        return self.output_head(pooled)

# ------------------------------------------------------------
# 1. LOAD TRAINED MODEL + DATA
# ------------------------------------------------------------
print("=" * 70)
print("LOADING MODEL AND DATA FOR SHAP ANALYSIS")
print("=" * 70)

device = torch.device("cpu")
n_features = 17

model = RULTransformer(n_features=n_features, d_model=32, n_heads=4, n_layers=2, dropout=0.3)
model.load_state_dict(torch.load("cmapss_transformer_rul_v2.pt", map_location=device))
model.eval()
print("Model loaded.")

data = np.load("cmapss_fd001_windows.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_test = data["y_test"]   # needed later for the evidence object's actual_rul

feature_names = [
    "op_setting_1", "op_setting_2", "T24", "T30", "T50", "P15", "P30",
    "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32"
]
print(f"Feature names ({len(feature_names)}): {feature_names}")


from sklearn.preprocessing import MinMaxScaler

# Recreate the RUL scaler the same way as in the inference script
# (ideally you'd load a saved scaler via joblib instead of refitting — same caveat as before)
rul_scaler = MinMaxScaler()
rul_scaler.fit(data["y_train"].reshape(-1, 1))

# ------------------------------------------------------------
# 2. PREPARE BACKGROUND + TEST SAMPLES
# ------------------------------------------------------------
# SHAP needs a small "background" reference set to compare predictions against.
# Using the full training set is too slow — sample a representative subset.
np.random.seed(42)
background_idx = np.random.choice(X_train.shape[0], size=100, replace=False)
background = torch.tensor(X_train[background_idx], dtype=torch.float32)

# Explain a manageable number of test windows (SHAP on deep models is slow)
n_explain = 50
test_sample_idx = np.random.choice(X_test.shape[0], size=n_explain, replace=False)
test_samples = torch.tensor(X_test[test_sample_idx], dtype=torch.float32)

print(f"\nBackground samples: {background.shape}")
print(f"Test samples to explain: {test_samples.shape}")

# ------------------------------------------------------------
# 3. RUN SHAP GRADIENT EXPLAINER
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("RUNNING SHAP (GradientExplainer) — this may take a few minutes on CPU")
print("=" * 70)

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

# shap_values shape: (n_explain, timesteps, features) for single-output regression
shap_values = np.array(shap_values)
if shap_values.ndim == 4:          # some SHAP versions add an extra output-dim axis
    shap_values = shap_values[..., 0]

print(f"SHAP values shape: {shap_values.shape}  (samples, timesteps, features)")

# ------------------------------------------------------------
# 4. AGGREGATE IMPORTANCE PER FEATURE (across timesteps + samples)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE RANKING")
print("=" * 70)

# Mean absolute SHAP value per feature, averaged across all timesteps and samples
feature_importance = np.abs(shap_values).mean(axis=(0, 1))  # shape: (features,)

importance_ranking = sorted(
    zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
)

print(f"{'Feature':<15} {'Mean |SHAP|':>12}")
for name, val in importance_ranking:
    print(f"{name:<15} {val:>12.5f}")

top_5_features = [name for name, _ in importance_ranking[:5]]
print(f"\nTop 5 most influential features: {top_5_features}")



def get_ntsb_evidence(query_text, faiss_index, ntsb_metadata, embed_model, top_k=1):
    """
    Embeds the query, searches the NTSB FAISS index, and returns the
    top match with its REAL similarity score -- this becomes the belief
    value for Dempster-Shafer, replacing the old hardcoded 1.0.
    """
    query_embedding = embed_model.encode([query_text])
    distances, indices = faiss_index.search(query_embedding, top_k)

    top_idx = indices[0][0]
    raw_distance = distances[0][0]

    # Convert distance to a 0-1 similarity (adjust formula based on your
    # FAISS index type -- IndexFlatL2 uses squared L2 distance, so this
    # is one common normalization; IndexFlatIP would already be similarity)
    similarity = 1 / (1 + raw_distance)

    matched_report = ntsb_metadata[top_idx]  # your NTSB record for this index

    retrieved_evidence = {
        "source": "NTSB",
        "event_id": matched_report["event_id"],
        "retrieval_score": round(float(similarity), 3)
    }

    fusion_evidence = {
        "source": "NTSB",
        "hypothesis": "Medium Risk",  # or map based on damage severity of matched report
        "belief": round(float(similarity), 3),
        "uncertainty": round(1 - similarity, 3)
    }

    return retrieved_evidence, fusion_evidence

#################################

SENSOR_SUBSYSTEM_MAP = {
    "T24": "Low-Pressure Compressor",
    "T30": "High-Pressure Compressor",
    "T50": "Low-Pressure Turbine",
    "P15": "Bypass Duct",
    "P30": "High-Pressure Compressor",
    "Nf":  "Fan",
    "Nc":  "Core / High-Pressure Spool",
    "Ps30": "High-Pressure Compressor",
    "phi": "Fuel System",
    "NRf": "Fan",
    "NRc": "Core / High-Pressure Spool",
    "BPR": "Bypass System",
    "htBleed": "Bleed Air System",
    "W31": "Bypass Flow",
    "W32": "Core Flow",
}

SUBSYSTEM_QUERY_MAP = {
    "High-Pressure Compressor": "compressor pressure degradation failure",
    "Low-Pressure Compressor": "low pressure compressor degradation",
    "Low-Pressure Turbine": "turbine temperature degradation failure",
    "Fan": "fan speed anomaly failure",
    "Core / High-Pressure Spool": "core engine speed degradation",
    "Fuel System": "fuel flow anomaly",
    "Bypass Duct": "bypass duct pressure anomaly",
    "Bleed Air System": "bleed air system failure",
    "Bypass System": "bypass ratio anomaly",
    "Bypass Flow": "bypass airflow anomaly",
    "Core Flow": "core airflow anomaly",
}

def shap_to_query(top_shap_features):
    """
    Takes CMAPSS's top SHAP features (e.g. ['Ps30','NRf','phi','T50','P30'])
    and produces a probable-cause subsystem label + a FAISS search query.
    """
    from collections import Counter
    subsystems = [SENSOR_SUBSYSTEM_MAP.get(f, "Unknown") for f in top_shap_features]
    most_common_subsystem = Counter(subsystems).most_common(1)[0][0]
    query_text = SUBSYSTEM_QUERY_MAP.get(most_common_subsystem, "engine failure")
    return most_common_subsystem, query_text

# Example using your actual result:
top_5 = ["Ps30", "NRf", "phi", "T50", "P30"]
subsystem, query = shap_to_query(top_5)
print(f"Probable cause subsystem: {subsystem}")
print(f"FAISS query: {query}")

# ------------------------------------------------------------
# 5. VISUALIZE — BAR CHART OF FEATURE IMPORTANCE
# ------------------------------------------------------------
names_sorted = [x[0] for x in importance_ranking]
values_sorted = [x[1] for x in importance_ranking]

plt.figure(figsize=(10, 6))
plt.barh(names_sorted[::-1], values_sorted[::-1], color="steelblue")
plt.xlabel("Mean |SHAP value|  (impact on RUL prediction)")
plt.title("CMAPSS Transformer — Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=150)
plt.show()
print("\nSaved bar chart to shap_feature_importance.png")

# ------------------------------------------------------------
# 6. VISUALIZE — SHAP OVER TIME FOR ONE SAMPLE (explains a single prediction)
# ------------------------------------------------------------
sample_i = 0
plt.figure(figsize=(12, 6))
for f_idx, f_name in enumerate(feature_names):
    if f_name in top_5_features:
        plt.plot(shap_values[sample_i, :, f_idx], label=f_name, marker="o", markersize=3)

plt.title(f"SHAP Values Over Time — Test Window {test_sample_idx[sample_i]} (Top 5 Features)")
plt.xlabel("Timestep within window")
plt.ylabel("SHAP value (contribution to RUL prediction)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig("shap_single_window_timeline.png", dpi=150)
plt.show()
print("Saved single-window SHAP timeline to shap_single_window_timeline.png")

# ------------------------------------------------------------
# 7. UPDATE EVIDENCE OBJECT WITH REAL SHAP FEATURES
# ------------------------------------------------------------
# ------------------------------------------------------------
# FINAL COMBINED EVIDENCE OBJECT (SHAP + RUL prediction + confidence)
# ------------------------------------------------------------
# Run this using the SAME test_sample_idx[sample_i] window as your SHAP analysis,
# so the RUL prediction and SHAP explanation refer to the exact same window.

import json

sample_window_id = test_sample_idx[sample_i]   # test_window_97 in your run

with torch.no_grad():
    single_pred_scaled = model(test_samples[sample_i:sample_i+1]).numpy()

# Use your saved/refit rul_scaler here (see earlier inference script)
single_pred_rul = rul_scaler.inverse_transform(single_pred_scaled).flatten()[0]
actual_rul = y_test[sample_window_id]  # from your loaded .npz test set
confidence = round(max(1 - abs(single_pred_rul - actual_rul) / 125, 0), 3)

evidence_object = {
    "source": "CMAPSS",
    "prediction": {
        "RUL": round(float(single_pred_rul), 2),
        "anomaly_score": None   # still pending your anomaly-detection logic
    },
    "confidence": confidence,
    "shap_top_features": top_5_features,
    "timestamp_window": f"test_window_{sample_window_id}"
}



print("\nFINAL Evidence Object (CMAPSS):")
print(json.dumps(evidence_object, indent=2))

# ------------------------------------------------------------
# 8. SAVE EVIDENCE OBJECT TO FILE (for Dempster-Shafer fusion stage)
# ------------------------------------------------------------
import os

os.makedirs("evidence_objects", exist_ok=True)  # creates folder if it doesn't exist

output_path = f"evidence_objects/cmapss_evidence_{sample_window_id}.json"
with open(output_path, "w") as f:
    json.dump(evidence_object, f, indent=2)

print(f"\nEvidence object saved to: {output_path}")