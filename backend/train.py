import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

#  Configuring...Loading dataset

data_folder = "../data"
dataset_paths = sorted([
    os.path.join(data_folder,f)
    for f in os.listdir(data_folder)
    if f.endswith(".csv")
])

voltage_col = "voltage"
current_col = "current"
target_col  = "threshold_power"

hidden_layers  = [64, 128, 128, 64]
learning_rate  = 1e-3
epochs         = 5000
physics_weight = 0.5
data_weight    = 1.0
test_size      = 0.2
random_seed    = 42

#  Dataset Loading

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df)} samples from '{dataset_path}'")

    V = df[voltage_col].values.reshape(-1, 1)
    I = df[current_col].values.reshape(-1, 1)
    P = df[target_col].values.reshape(-1, 1)

    return V, I, P

#  Data Preprocessing

def preprocess(V, I, P):
    scaler_V = MinMaxScaler()
    scaler_I = MinMaxScaler()
    scaler_P = MinMaxScaler()

    V_scaled = scaler_V.fit_transform(V)
    I_scaled = scaler_I.fit_transform(I)
    P_scaled = scaler_P.fit_transform(P)

    X = np.hstack([V_scaled, I_scaled])
    X_train, X_test, P_train, P_test, V_train_raw, V_test_raw, I_train_raw, I_test_raw = \
        train_test_split(X, P_scaled, V, I, test_size=test_size, random_state=random_seed)

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return (
        to_tensor(X_train), to_tensor(X_test),
        to_tensor(P_train), to_tensor(P_test),
        to_tensor(V_train_raw), to_tensor(V_test_raw),
        to_tensor(I_train_raw), to_tensor(I_test_raw),
        scaler_V, scaler_I, scaler_P
    )

#  PINN Architecture

class PINN(nn.Module):
    def __init__(self, hidden_layers):
        super(PINN, self).__init__()
        layers = []
        in_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)

#  Physics Loss

def physics_loss(model, V_raw, I_raw, scaler_V, scaler_I, scaler_P):
    P_physics = V_raw * I_raw
    P_physics_scaled = torch.tensor(
        scaler_P.transform(P_physics.numpy()), dtype=torch.float32
    )
    V_scaled = torch.tensor(scaler_V.transform(V_raw.numpy()), dtype=torch.float32)
    I_scaled = torch.tensor(scaler_I.transform(I_raw.numpy()), dtype=torch.float32)
    X = torch.cat([V_scaled, I_scaled], dim=1)

    P_pred    = model(X)
    residual  = P_pred - P_physics_scaled
    return torch.mean(residual ** 2)

#  Training

def train(model, X_train, P_train, V_train_raw, I_train_raw, scaler_V, scaler_I, scaler_P):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    history = {"total": [], "data": [], "physics": []}
    print(f"  {'Epoch':>8} | {'Total Loss':>12} | {'Data Loss':>12} | {'Physics Loss':>12}")
    print("  " + "─" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        P_pred       = model(X_train)
        loss_data    = torch.mean((P_pred - P_train) ** 2)
        loss_physics = physics_loss(model, V_train_raw, I_train_raw, scaler_V, scaler_I, scaler_P)
        loss_total   = data_weight * loss_data + physics_weight * loss_physics

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss_total.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"  {epoch:>8} | {loss_total.item():>12.6f} | "
                  f"{loss_data.item():>12.6f} | {loss_physics.item():>12.6f}")

    print("  Training Complete \n")
    return history

#  Evaluation

def evaluate(model, X_test, P_test, scaler_P):
    model.eval()
    with torch.no_grad():
        P_pred_scaled = model(X_test).numpy()
        P_test_scaled = P_test.numpy()

    P_pred_orig = scaler_P.inverse_transform(P_pred_scaled)
    P_test_orig = scaler_P.inverse_transform(P_test_scaled)

    r2   = r2_score(P_test_orig, P_pred_orig)
    mse  = mean_squared_error(P_test_orig, P_pred_orig)
    rmse = np.sqrt(mse)

    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MSE  : {mse:.4f}\n")

    return P_pred_orig, P_test_orig, r2, rmse

#  Saving Model
def save_model(model, scaler_V, scaler_I, scaler_P, device_name):
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{device_name}_pinn.pth"
    torch.save(model.state_dict(), model_path)

    joblib.dump(scaler_V, f"models/{device_name}_scaler_V.pkl")
    joblib.dump(scaler_I, f"models/{device_name}_scaler_I.pkl")
    joblib.dump(scaler_P, f"models/{device_name}_scaler_P.pkl")

    print(f"  Saved → {model_path}")
    print(f"  Saved → models/{device_name}_scaler_V.pkl")
    print(f"  Saved → models/{device_name}_scaler_I.pkl")
    print(f"  Saved → models/{device_name}_scaler_P.pkl\n")

# starting
if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    results_summary = []

    for dataset_path in dataset_paths:

        # Extract clean device name:  "../data/bulb.csv" → "bulb"
        device_name = os.path.splitext(os.path.basename(dataset_path))[0]

        print("=" * 60)
        print(f"  DEVICE : {device_name.upper()}")
        print("=" * 60)

        # 1. Load
        V, I, P = load_data(dataset_path)

        # 2. Preprocess
        (X_train, X_test, P_train, P_test,
         V_train_raw, V_test_raw,
         I_train_raw, I_test_raw,
         scaler_V, scaler_I, scaler_P) = preprocess(V, I, P)

        # 3. Build model
        model = PINN(hidden_layers)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Architecture : 2 → {hidden_layers} → 1")
        print(f"  Parameters   : {total_params:,}\n")

        # 4. Train
        history = train(model, X_train, P_train,
                        V_train_raw, I_train_raw,
                        scaler_V, scaler_I, scaler_P)

        # 5. Evaluate
        P_pred, P_actual, r2, rmse = evaluate(model, X_test, P_test, scaler_P)

        # 6. Save
        save_model(model, scaler_V, scaler_I, scaler_P, device_name)

        results_summary.append({
            "device": device_name,
            "r2":     round(r2, 4),
            "rmse":   round(rmse, 4)
        })

    # Final summary
    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print("=" * 60)