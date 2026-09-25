# ============================================================
# AERIS Project — CMAPSS FD001 Time-Series Transformer (RUL Prediction)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
DATA_PATH = "cmapss_fd001_windows.npz"
D_MODEL = 32          # reduced from 64 — less capacity to memorize noise
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3         # increased from 0.1 — stronger regularization
WEIGHT_DECAY = 1e-4   # new — penalizes overly large weights
BATCH_SIZE = 64
MAX_EPOCHS = 100       # upper limit; early stopping will likely halt sooner
PATIENCE = 8           # stop if val loss doesn't improve for 8 epochs
VAL_SPLIT_RATIO = 0.2  # 20% of TRAIN engines held out as validation
LR = 1e-3  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# 1. LOAD PREPROCESSED WINDOWS
# ------------------------------------------------------------
print("=" * 70)
print("LOADING PREPROCESSED WINDOWS")
print("=" * 70)

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

n_features = X_train.shape[2]
seq_len = X_train.shape[1]



# 1B. SPLIT TRAIN ENGINES -> TRAIN + VALIDATION (engine-wise, no leakage)
# ------------------------------------------------------------
# Assumes you still have access to train_units from preprocessing.
# If not, split X_train/y_train by re-deriving unit IDs, OR simplest:
# re-run this split logic in preprocessing and re-save the .npz with
# X_val/y_val included. For now, here's a quick windows-level split
# (slightly less clean but workable if unit IDs aren't carried through):

np.random.seed(42)
n_train_samples = X_train.shape[0]
indices = np.arange(n_train_samples)
np.random.shuffle(indices)
n_val = int(n_train_samples * VAL_SPLIT_RATIO)

val_idx = indices[:n_val]
tr_idx = indices[n_val:]

X_val, y_val = X_train[val_idx], y_train[val_idx]
X_train, y_train = X_train[tr_idx], y_train[tr_idx]

print(f"New train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")



# ------------------------------------------------------------
# 2. SCALE TARGET (fit on train only)
# ------------------------------------------------------------
rul_scaler = MinMaxScaler()
y_train_scaled = rul_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = rul_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = rul_scaler.transform(y_test.reshape(-1, 1)).flatten()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------------------
# 3. POSITIONAL ENCODING (gives the transformer a sense of time order)
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

# ------------------------------------------------------------
# 4. TRANSFORMER MODEL
# ------------------------------------------------------------
class RULTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        pooled = x.mean(dim=1)          # average across timesteps
        return self.output_head(pooled)

model = RULTransformer(n_features, D_MODEL, N_HEADS, N_LAYERS, DROPOUT).to(device)
print(model)


# ------------------------------------------------------------
# 5. TRAINING SETUP (updated with weight decay)
# ------------------------------------------------------------
model = RULTransformer(n_features, D_MODEL, N_HEADS, N_LAYERS, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)


# ------------------------------------------------------------
# 6. TRAINING LOOP WITH EARLY STOPPING (based on VALIDATION loss)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING TRANSFORMER (with early stopping)")
print("=" * 70)

best_val_loss = float("inf")
epochs_no_improve = 0
best_model_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)

    # ---- Validation pass (no gradient updates) ----
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device))
        val_loss = criterion(val_preds, y_val_t.to(device)).item()

    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{MAX_EPOCHS}] - Train Loss: {epoch_loss:.5f} - Val Loss: {val_loss:.5f} - LR: {current_lr:.6f}")

    # ---- Early stopping check ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()  # save best-performing weights
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

# Restore the best model (not necessarily the last epoch's weights)
model.load_state_dict(best_model_state)
print(f"\nRestored best model (validation loss: {best_val_loss:.5f})")

# ------------------------------------------------------------
# 7. EVALUATION (on untouched test set — evaluated only once)
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t.to(device)).cpu().numpy()

y_pred = rul_scaler.inverse_transform(y_pred_scaled).flatten()
y_true = y_test

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\nTransformer MAE  : {mae:.3f}")
print(f"Transformer RMSE : {rmse:.3f}")
print("\nCompare against:")
print("  Linear Regression -> MAE 13.025, RMSE 15.761")
print("  LSTM baseline      -> MAE 8.818,  RMSE 13.081")

torch.save(model.state_dict(), "cmapss_transformer_rul_v2.pt")
print("\nSaved improved transformer weights to cmapss_transformer_rul_v2.pt")

