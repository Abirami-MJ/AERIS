# ============================================================
# AERIS Project — CMAPSS FD001 Pre-Modeling Pipeline
# (RUL Capping -> Engine-wise Split -> Normalization -> Windowing -> Baseline)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
FILE_PATH = "cleaned_outputs/C-MAPSS_FD001/C-MAPSS_FD001_cleaned.csv"   # update path if needed
RUL_CAP = 125                            # standard cap used in CMAPSS literature
WINDOW_SIZE = 30                         # sequence length per training sample
                                          # (30 is standard for CMAPSS; 300 was discussed
                                          #  for NGAFID which has far longer flight sequences —
                                          #  confirm with your teammate if you want these aligned)
TEST_SPLIT_RATIO = 0.2

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
print("=" * 70)
print("LOADING CMAPSS FD001")
print("=" * 70)

df = pd.read_csv(FILE_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

unit_col = "unit_number" if "unit_number" in df.columns else df.columns[0]
cycle_col = "time_cycles" if "time_cycles" in df.columns else df.columns[1]

"""sensor_cols = [c for c in df.columns if "sensor" in c.lower()]
setting_cols = [c for c in df.columns if "setting" in c.lower()]
feature_cols = setting_cols + sensor_cols"""

# Explicitly exclude ID, cycle, RUL, and outlier-flag columns — 
# everything else is a genuine feature (op settings + real sensor readings)
exclude_cols = [unit_col, cycle_col, "RUL"]
exclude_cols += [c for c in df.columns if c.endswith("_is_outlier")]

feature_cols = [c for c in df.columns if c not in exclude_cols]


print(f"Unit column: {unit_col}")
print(f"Cycle column: {cycle_col}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# ------------------------------------------------------------
# 2. COMPUTE + CAP RUL
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("RUL COMPUTATION & CAPPING")
print("=" * 70)

if "RUL" not in df.columns:
    max_cycle = df.groupby(unit_col)[cycle_col].transform("max")
    df["RUL"] = max_cycle - df[cycle_col]
    print("RUL computed from max cycle per unit.")
else:
    print("RUL column already present — using existing values.")

print(f"RUL before capping — min: {df['RUL'].min()}, max: {df['RUL'].max()}")
df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
print(f"RUL after capping at {RUL_CAP} — min: {df['RUL'].min()}, max: {df['RUL'].max()}")

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT — BY ENGINE UNIT (no leakage)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("ENGINE-WISE TRAIN/TEST SPLIT")
print("=" * 70)

unique_units = df[unit_col].unique()
np.random.shuffle(unique_units)

n_test = int(len(unique_units) * TEST_SPLIT_RATIO)
test_units = unique_units[:n_test]
train_units = unique_units[n_test:]

train_df = df[df[unit_col].isin(train_units)].copy()
test_df = df[df[unit_col].isin(test_units)].copy()

print(f"Total engines: {len(unique_units)}")
print(f"Train engines: {len(train_units)} -> {len(train_df)} rows")
print(f"Test engines : {len(test_units)} -> {len(test_df)} rows")

# ------------------------------------------------------------
# 4. NORMALIZATION — FIT ON TRAIN ONLY
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("NORMALIZATION (fit on train, applied to test)")
print("=" * 70)

scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

print("Scaler fitted on training engines only. No leakage into test set.")
print(f"Train feature range check -> min: {train_df[feature_cols].min().min():.3f}, "
      f"max: {train_df[feature_cols].max().max():.3f}")

# ------------------------------------------------------------
# 5. SEQUENCE / WINDOW GENERATION (with padding + mask)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print(f"WINDOW GENERATION (window size = {WINDOW_SIZE})")
print("=" * 70)

def generate_windows(data, unit_col, cycle_col, feature_cols, window_size):
    """
    Converts each engine's time series into fixed-length windows.
    Sequences shorter than window_size are front-padded with zeros,
    and an attention mask marks real vs padded timesteps.
    """
    X, y, masks, unit_ids = [], [], [], []

    for uid, group in data.groupby(unit_col):
        group = group.sort_values(cycle_col)
        features = group[feature_cols].values
        ruls = group["RUL"].values
        n = len(group)

        if n >= window_size:
            # sliding windows across the full sequence
            for start in range(0, n - window_size + 1):
                X.append(features[start:start + window_size])
                y.append(ruls[start + window_size - 1])
                masks.append(np.ones(window_size))
                unit_ids.append(uid)
        else:
            # pad short sequences (rare in CMAPSS, common in NGAFID-style data)
            pad_len = window_size - n
            padded = np.vstack([np.zeros((pad_len, len(feature_cols))), features])
            mask = np.concatenate([np.zeros(pad_len), np.ones(n)])
            X.append(padded)
            y.append(ruls[-1])
            masks.append(mask)
            unit_ids.append(uid)

    return np.array(X), np.array(y), np.array(masks), np.array(unit_ids)

X_train, y_train, mask_train, units_train = generate_windows(
    train_df, unit_col, cycle_col, feature_cols, WINDOW_SIZE
)
X_test, y_test, mask_test, units_test = generate_windows(
    test_df, unit_col, cycle_col, feature_cols, WINDOW_SIZE
)

print(f"X_train shape: {X_train.shape}  (samples, timesteps, features)")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"Padded sequences in train: {(mask_train.sum(axis=1) < WINDOW_SIZE).sum()}")
print(f"Padded sequences in test : {(mask_test.sum(axis=1) < WINDOW_SIZE).sum()}")

# ------------------------------------------------------------
# 6. SAVE PROCESSED ARRAYS (so transformer training doesn't redo this)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("SAVING PROCESSED WINDOWS")
print("=" * 70)

np.savez_compressed(
    "cmapss_fd001_windows.npz",
    X_train=X_train, y_train=y_train, mask_train=mask_train,
    X_test=X_test, y_test=y_test, mask_test=mask_test
)
print("Saved to cmapss_fd001_windows.npz")
print("(Load later with: data = np.load('cmapss_fd001_windows.npz'))")

# ------------------------------------------------------------
# 7. BASELINE MODEL — LINEAR REGRESSION ON FLATTENED WINDOWS
# ------------------------------------------------------------
"""
print("\n" + "=" * 70)
print("BASELINE MODEL: LINEAR REGRESSION")
print("=" * 70)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

baseline = LinearRegression()
baseline.fit(X_train_flat, y_train)
y_pred = baseline.predict(X_test_flat)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Baseline MAE  : {mae:.3f}")
print(f"Baseline RMSE : {rmse:.3f}")
print("\nUse these numbers as your benchmark — the transformer should beat this")
print("before you conclude it's adding real value over a simple linear model.")
"""

# ------------------------------------------------------------
# 7. BASELINE MODEL — LSTM (sequence-aware baseline)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("BASELINE MODEL: LSTM")
print("=" * 70)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Scale the target too (RUL 0-125 -> 0-1), fit on train only ----
rul_scaler = MinMaxScaler()
y_train_scaled = rul_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = rul_scaler.transform(y_test.reshape(-1, 1)).flatten()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---- Define LSTM model ----
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # final layer's hidden state
        return self.fc(last_hidden)


model = LSTMBaseline(input_size=X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

EPOCHS = 50
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevents exploding gradients
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_dataset)
    scheduler.step(epoch_loss)
    if epoch % 5 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss (MSE, scaled): {epoch_loss:.5f} - LR: {current_lr:.6f}")

# ---- Evaluate: inverse-transform predictions back to real RUL scale ----
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t.to(device)).cpu().numpy()

y_pred = rul_scaler.inverse_transform(y_pred_scaled).flatten()
y_true = y_test  # original unscaled RUL

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nLSTM Baseline MAE  : {mae:.3f}")
print(f"LSTM Baseline RMSE : {rmse:.3f}")

# ---- Save baseline model for reference ----
torch.save(model.state_dict(), "cmapss_lstm_baseline.pt")
print("\nSaved LSTM baseline weights to cmapss_lstm_baseline.pt")



