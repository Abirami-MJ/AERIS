# ============================================================
# AERIS Project — CMAPSS Transformer Inference & Evidence Generation
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import math
import json
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 0. RE-DEFINE THE MODEL ARCHITECTURE (must match training exactly)
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
# 1. LOAD THE TRAINED MODEL WEIGHTS
# ------------------------------------------------------------
device = torch.device("cpu")
n_features = 17   # must match training

model = RULTransformer(n_features=n_features, d_model=32, n_heads=4, n_layers=2, dropout=0.3)
model.load_state_dict(torch.load("cmapss_transformer_rul_v2.pt", map_location=device))
model.eval()   # inference mode — disables dropout
print("Model loaded successfully.")

# ------------------------------------------------------------
# 2. LOAD TEST DATA + RECREATE THE SAME RUL SCALER
# ------------------------------------------------------------
data = np.load("cmapss_fd001_windows.npz")
X_test, y_test = data["X_test"], data["y_test"]

# IMPORTANT: this must be fit the same way as during training.
# Ideally, save your scaler during training with:
#   import joblib; joblib.dump(rul_scaler, "rul_scaler.pkl")
# and load it here instead of refitting. Refitting on y_test alone
# (as a placeholder below) is NOT correct for real deployment —
# only use this if you don't yet have the saved scaler.
rul_scaler = MinMaxScaler()
rul_scaler.fit(data["y_train"].reshape(-1, 1))   # fit on train, same as original

X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ------------------------------------------------------------
# 3. RUN INFERENCE — GET INDIVIDUAL PREDICTIONS
# ------------------------------------------------------------
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()

y_pred = rul_scaler.inverse_transform(y_pred_scaled).flatten()
y_true = y_test

# ------------------------------------------------------------
# 4. SHOW SAMPLE PREDICTIONS SIDE BY SIDE
# ------------------------------------------------------------
print("\nSample Predictions (first 15 test windows):")
print(f"{'Predicted RUL':>15} {'Actual RUL':>15} {'Abs Error':>12}")
for i in range(15):
    err = abs(y_pred[i] - y_true[i])
    print(f"{y_pred[i]:>15.2f} {y_true[i]:>15.2f} {err:>12.2f}")

# ------------------------------------------------------------
# 5. BUILD THE EVIDENCE OBJECT (for Dempster-Shafer fusion)
# ------------------------------------------------------------
# Confidence here is a simple example: inverse of normalized error spread.
# You'll refine this once SHAP is integrated (Section 6 note below).
sample_idx = 0   # example: first test window
sample_pred = float(y_pred[sample_idx])
sample_confidence = round(1 - (abs(y_pred[sample_idx] - y_true[sample_idx]) / 125), 3)

evidence_object = {
    "source": "CMAPSS",
    "prediction": {
        "RUL": round(sample_pred, 2),
        "anomaly_score": None   # placeholder — populate once anomaly detection logic is added
    },
    "confidence": max(sample_confidence, 0),  # clip at 0 in case error is large
    "shap_top_features": [],   # placeholder — populate once SHAP is run on this model
    "timestamp_window": f"test_window_{sample_idx}"
}

print("\nExample Evidence Object (single window):")
print(json.dumps(evidence_object, indent=2))

# ------------------------------------------------------------
# 6. VISUALIZE: PREDICTED VS ACTUAL RUL
# ------------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_true[:100], label="Actual RUL", marker="o", markersize=3)
plt.plot(y_pred[:100], label="Predicted RUL", marker="x", markersize=3)
plt.title("Predicted vs Actual RUL (first 100 test windows)")
plt.xlabel("Test Window Index")
plt.ylabel("RUL (cycles)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("predicted_vs_actual_rul.png", dpi=150)
plt.show()
print("\nSaved visualization to predicted_vs_actual_rul.png")

# ---- Scatter plot: perfect prediction line for reference ----
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.3, s=15)
plt.plot([0, 125], [0, 125], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Prediction Accuracy Scatter Plot")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rul_scatter_accuracy.png", dpi=150)
plt.show()
print("Saved scatter plot to rul_scatter_accuracy.png")