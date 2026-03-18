"""
Physics-Informed Neural Network (PINN) — From Scratch in PyTorch
================================================================
Predicts Threshold Power (P_th) from Voltage (V) and Current (I)
Physics Constraint: P_th = V × I

CSV Format Expected:
    voltage, current, threshold_power
    e.g.:  5.0, 2.0, 10.5
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. CONFIGURATION  ← Edit these as needed
# ─────────────────────────────────────────────
CSV_PATH        = "your_data.csv"       # ← Path to your CSV file
VOLTAGE_COL     = "voltage"             # ← Column name for voltage
CURRENT_COL     = "current"             # ← Column name for current
TARGET_COL      = "threshold_power"     # ← Column name for P_th

HIDDEN_LAYERS   = [64, 128, 128, 64]    # Neural network architecture
LEARNING_RATE   = 1e-3
EPOCHS          = 5000
PHYSICS_WEIGHT  = 0.5                   # λ: weight of physics loss (0–1)
DATA_WEIGHT     = 1.0                   # weight of data loss
TEST_SIZE       = 0.2
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# 2. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples from '{csv_path}'")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Preview:\n{df.head()}\n")

    V = df[VOLTAGE_COL].values.reshape(-1, 1)
    I = df[CURRENT_COL].values.reshape(-1, 1)
    P = df[TARGET_COL].values.reshape(-1, 1)

    return V, I, P


def preprocess(V, I, P):
    scaler_V = MinMaxScaler()
    scaler_I = MinMaxScaler()
    scaler_P = MinMaxScaler()

    V_scaled = scaler_V.fit_transform(V)
    I_scaled = scaler_I.fit_transform(I)
    P_scaled = scaler_P.fit_transform(P)

    X = np.hstack([V_scaled, I_scaled])  # shape: (N, 2)

    X_train, X_test, P_train, P_test, V_train_raw, V_test_raw, I_train_raw, I_test_raw = \
        train_test_split(X, P_scaled, V, I,
                         test_size=TEST_SIZE,
                         random_state=RANDOM_SEED)

    # Convert to tensors
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return (to_tensor(X_train), to_tensor(X_test),
            to_tensor(P_train), to_tensor(P_test),
            to_tensor(V_train_raw), to_tensor(V_test_raw),
            to_tensor(I_train_raw), to_tensor(I_test_raw),
            scaler_V, scaler_I, scaler_P)


# ─────────────────────────────────────────────
# 3. PINN ARCHITECTURE
# ─────────────────────────────────────────────
class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    Input:  [V_scaled, I_scaled]  → shape (N, 2)
    Output: [P_th_scaled]         → shape (N, 1)
    """
    def __init__(self, hidden_layers):
        super(PINN, self).__init__()

        layers = []
        in_dim = 2  # V and I

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())          # Tanh works well for physics problems
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # Output: P_th
        self.network = nn.Sequential(*layers)

        # Weight initialization (Xavier)
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# 4. PHYSICS LOSS  →  Enforces P_th ≈ V × I
# ─────────────────────────────────────────────
def physics_loss(model, V_raw, I_raw, scaler_V, scaler_I, scaler_P):
    """
    Physics constraint: P_th = V × I
    We compute the residual: model(V, I) - scale(V × I)
    """
    # Compute physics prediction in original scale
    P_physics = V_raw * I_raw  # shape: (N, 1)

    # Scale physics prediction to match model output scale
    P_physics_scaled = torch.tensor(
        scaler_P.transform(P_physics.numpy()),
        dtype=torch.float32
    )

    # Scale inputs for model
    V_scaled = torch.tensor(scaler_V.transform(V_raw.numpy()), dtype=torch.float32)
    I_scaled = torch.tensor(scaler_I.transform(I_raw.numpy()), dtype=torch.float32)
    X = torch.cat([V_scaled, I_scaled], dim=1)

    # Model prediction
    P_pred = model(X)

    # Physics residual loss
    residual = P_pred - P_physics_scaled
    return torch.mean(residual ** 2)


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
def train(model, X_train, P_train, V_train_raw, I_train_raw,
          scaler_V, scaler_I, scaler_P):

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    history = {"total": [], "data": [], "physics": []}

    print("🚀 Training PINN...\n")
    print(f"{'Epoch':>8} | {'Total Loss':>12} | {'Data Loss':>12} | {'Physics Loss':>12}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # --- Data Loss: model predictions vs actual measurements ---
        P_pred = model(X_train)
        loss_data = torch.mean((P_pred - P_train) ** 2)

        # --- Physics Loss: enforce P_th = V × I ---
        loss_physics = physics_loss(model, V_train_raw, I_train_raw,
                                    scaler_V, scaler_I, scaler_P)

        # --- Combined Loss ---
        loss_total = DATA_WEIGHT * loss_data + PHYSICS_WEIGHT * loss_physics

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss_total.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>8} | {loss_total.item():>12.6f} | "
                  f"{loss_data.item():>12.6f} | {loss_physics.item():>12.6f}")

    print("\n✅ Training complete!\n")
    return history


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_test, P_test, scaler_P):
    model.eval()
    with torch.no_grad():
        P_pred_scaled = model(X_test).numpy()
        P_test_scaled = P_test.numpy()

    # Inverse transform to original scale
    P_pred_orig = scaler_P.inverse_transform(P_pred_scaled)
    P_test_orig = scaler_P.inverse_transform(P_test_scaled)

    r2  = r2_score(P_test_orig, P_pred_orig)
    mse = mean_squared_error(P_test_orig, P_pred_orig)
    rmse = np.sqrt(mse)

    print("📊 Evaluation on Test Set:")
    print(f"   R² Score : {r2:.4f}  (1.0 = perfect)")
    print(f"   RMSE     : {rmse:.4f}")
    print(f"   MSE      : {mse:.4f}\n")

    return P_pred_orig, P_test_orig, r2, rmse


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_results(history, P_pred, P_test, r2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PINN — Threshold Power Prediction", fontsize=14, fontweight='bold')

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(history["total"],   label="Total Loss",   color="black")
    ax1.plot(history["data"],    label="Data Loss",    color="blue",   linestyle="--")
    ax1.plot(history["physics"], label="Physics Loss", color="red",    linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training Loss Curves")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted vs Actual
    ax2 = axes[1]
    ax2.scatter(P_test, P_pred, alpha=0.7, color="steelblue", edgecolors="white", s=60)
    min_val = min(P_test.min(), P_pred.min())
    max_val = max(P_test.max(), P_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal")
    ax2.set_xlabel("Actual P_th")
    ax2.set_ylabel("Predicted P_th")
    ax2.set_title(f"Predicted vs Actual  (R² = {r2:.4f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = axes[2]
    residuals = P_pred.flatten() - P_test.flatten()
    ax3.hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel("Residual (Predicted - Actual)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pinn_results.png", dpi=150, bbox_inches="tight")
    print("📈 Plot saved as 'pinn_results.png'")
    plt.show()


# ─────────────────────────────────────────────
# 8. PREDICTION FUNCTION (use after training)
# ─────────────────────────────────────────────
def predict(model, voltage_values, current_values, scaler_V, scaler_I, scaler_P):
    """
    Predict threshold power for new V, I values.

    Usage:
        P = predict(model, [5.0, 6.0], [2.0, 3.0], scaler_V, scaler_I, scaler_P)
        print(P)  # → predicted P_th values
    """
    V = np.array(voltage_values).reshape(-1, 1)
    I = np.array(current_values).reshape(-1, 1)

    V_scaled = scaler_V.transform(V)
    I_scaled = scaler_I.transform(I)
    X = torch.tensor(np.hstack([V_scaled, I_scaled]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        P_scaled = model(X).numpy()

    return scaler_P.inverse_transform(P_scaled)


# ─────────────────────────────────────────────
# 9. SAVE & LOAD MODEL
# ─────────────────────────────────────────────
def save_model(model, path="pinn_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to '{path}'")

def load_model(path="pinn_model.pth"):
    model = PINN(HIDDEN_LAYERS)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"📂 Model loaded from '{path}'")
    return model


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Step 1: Load data
    V, I, P = load_data(CSV_PATH)

    # Step 2: Preprocess
    (X_train, X_test, P_train, P_test,
     V_train_raw, V_test_raw,
     I_train_raw, I_test_raw,
     scaler_V, scaler_I, scaler_P) = preprocess(V, I, P)

    # Step 3: Build model
    model = PINN(HIDDEN_LAYERS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 PINN built | Architecture: 2 → {HIDDEN_LAYERS} → 1")
    print(f"   Total parameters: {total_params:,}\n")

    # Step 4: Train
    history = train(model, X_train, P_train,
                    V_train_raw, I_train_raw,
                    scaler_V, scaler_I, scaler_P)

    # Step 5: Evaluate
    P_pred, P_actual, r2, rmse = evaluate(model, X_test, P_test, scaler_P)

    # Step 6: Plot
    plot_results(history, P_pred, P_actual, r2)

    # Step 7: Save model
    save_model(model)

    # Step 8: Example new prediction
    print("🔮 Example Prediction:")
    new_V = [5.0, 10.0, 15.0]
    new_I = [2.0,  3.0,  4.0]
    P_new = predict(model, new_V, new_I, scaler_V, scaler_I, scaler_P)
    for v, i, p in zip(new_V, new_I, P_new.flatten()):
        print(f"   V={v}V, I={i}A  →  P_th = {p:.4f} W")