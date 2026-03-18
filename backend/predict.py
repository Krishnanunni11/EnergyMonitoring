import torch
import numpy as np
import joblib
from train import PINN

hidden_layers = [64, 128, 128, 64]  # must match train.py

def predict(model, voltage_values, current_values, scaler_V, scaler_I, scaler_P):
    V = np.array(voltage_values).reshape(-1, 1)
    I = np.array(current_values).reshape(-1, 1)

    V_scaled = scaler_V.transform(V)
    I_scaled = scaler_I.transform(I)

    X = torch.tensor(np.hstack([V_scaled, I_scaled]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        P_scaled = model(X).numpy()

    return scaler_P.inverse_transform(P_scaled)

if __name__ == "__main__":
    device_name = "fan_test"  # change to whichever device you want to predict

    model = PINN(hidden_layers)
    model.load_state_dict(torch.load(f"models/{device_name}_pinn.pth"))
    model.eval()

    scaler_V = joblib.load(f"models/{device_name}_scaler_V.pkl")
    scaler_I = joblib.load(f"models/{device_name}_scaler_I.pkl")
    scaler_P = joblib.load(f"models/{device_name}_scaler_P.pkl")

    device_name = "fan"
    new_V = [210.0, 220.0, 230.0]
    new_I = [0.30,  0.45,  0.65]

    P_new = predict(model, new_V, new_I, scaler_V, scaler_I, scaler_P)

    for v, i, p in zip(new_V, new_I, P_new.flatten()):
        print(f"  V={v}V, I={i}A  →  P_threshold = {p:.4f} W")