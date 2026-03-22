import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
import os
import uvicorn
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import paho.mqtt.client as mqtt
from collections import deque

# ──────────────────────────────────────────────
# PINN Model Architecture
# ──────────────────────────────────────────────

hidden_layers = [64, 128, 128, 64]

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

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(title="PINN Energy Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

MODELS_DIR = "models"
device_registry = {}

def load_all_models():
    if not os.path.exists(MODELS_DIR):
        print(" models/ folder not found!")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_pinn.pth")]

    for model_file in model_files:
        device_name   = model_file.replace("_pinn.pth", "")
        scaler_V_path = os.path.join(MODELS_DIR, f"{device_name}_scaler_V.pkl")
        scaler_I_path = os.path.join(MODELS_DIR, f"{device_name}_scaler_I.pkl")
        scaler_P_path = os.path.join(MODELS_DIR, f"{device_name}_scaler_P.pkl")
        model_path    = os.path.join(MODELS_DIR, model_file)

        if not all(os.path.exists(p) for p in [scaler_V_path, scaler_I_path, scaler_P_path]):
            print(f"   Skipping {device_name} — missing scalers")
            continue

        model = PINN(hidden_layers)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        device_registry[device_name] = {
            "model":    model,
            "scaler_V": joblib.load(scaler_V_path),
            "scaler_I": joblib.load(scaler_I_path),
            "scaler_P": joblib.load(scaler_P_path),
        }
        print(f"  Loaded model : {device_name}")

    print(f"\n  Total devices loaded: {list(device_registry.keys())}\n")

load_all_models()


# ──────────────────────────────────────────────
# State & Config
# ──────────────────────────────────────────────

WINDOW_SIZE = 10

plug_buffers      = {}
user_thresholds   = {}
latest_predictions = {}
latest_values     = {}
prediction_enabled = {}

# FIX 1: Use an Event to track real MQTT connection state
#         instead of relying solely on is_connected() which
#         can briefly return False during keepalive cycles.
mqtt_client: Optional[mqtt.Client] = None
mqtt_connected_event = threading.Event()

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT   = 8884          # WebSocket Secure (wss://)
MQTT_TOPIC  = "smart/plug/+/codedata"
MQTT_PATH   = "/mqtt"


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class SinglePrediction(BaseModel):
    appliance_id: str
    voltage: float
    current: float

class BatchPredictionRequest(BaseModel):
    appliances: List[SinglePrediction]

class ThresholdSetting(BaseModel):
    plug_id:     str
    device:      str            # e.g. "fan" or "fan_test"
    threshold:   float          # Max acceptable Watts
    tolerance_w: Optional[float] = None   # +-band in Watts (default 5%)
    min_power_w: Optional[float] = None   # Hard floor below this = fault (default 3W)

class RelayControl(BaseModel):
    plug_id: str
    state:   str        # "ON" or "OFF"


# ──────────────────────────────────────────────
# Relay Control
# FIX 2: Wait for connection event before giving up;
#         use wait() with timeout instead of polling sleep.
# ──────────────────────────────────────────────

def control_relay(plug_id: str, state: str, wait_secs: float = 5.0) -> bool:
    """
    Publish a relay ON/OFF command to the smart plug.

    Args:
        plug_id:   Plug identifier matching the MQTT topic segment.
        state:     "ON" or "OFF".
        wait_secs: How long to wait for the MQTT connection if not yet ready.

    Returns:
        True on successful publish, False otherwise.
    """
    global mqtt_client

    # Wait for connection to be established (up to wait_secs)
    if not mqtt_connected_event.wait(timeout=wait_secs):
        print(f"   Relay [{plug_id}]  {state} FAILED — MQTT not connected after {wait_secs}s wait")
        return False

    # Double-check the client is still alive
    if mqtt_client is None or not mqtt_client.is_connected():
        print(f"   Relay [{plug_id}] → {state} FAILED — MQTT client lost connection")
        mqtt_connected_event.clear()   # reset so next call waits again
        return False

    relay_topic = "smart/plug/command"
    relay_cmd   = "on" if state == "ON" else "off"
    payload     = json.dumps({"plug": int(plug_id), "cmd": relay_cmd})

    try:
        result = mqtt_client.publish(relay_topic, payload, qos=1)
        # Do NOT call result.wait_for_publish() here — it would block
        # the calling thread waiting on the MQTT loop to flush, causing
        # a deadlock when called from _relay_async.

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"   Relay [{plug_id}] → {state}  (topic: {relay_topic})")
            return True
        else:
            print(f"   Relay [{plug_id}] publish error code: {result.rc}")
            return False

    except Exception as e:
        print(f"   Relay control exception: {e}")
        return False


# ──────────────────────────────────────────────
# MQTT Callbacks
# ──────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"  ✅ MQTT Connected → {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC, qos=1)
        print(f"  📡 Subscribed → {MQTT_TOPIC}")
        mqtt_connected_event.set()      # FIX 4: signal that we're connected
    else:
        error_messages = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized",
        }
        print(f"  ❌ MQTT Connection failed [{rc}]: {error_messages.get(rc, 'Unknown error')}")
        mqtt_connected_event.clear()

def on_disconnect(client, userdata, rc):
    mqtt_connected_event.clear()        # FIX 5: clear event on disconnect
    if rc != 0:
        print(f"    MQTT Disconnected unexpectedly (code: {rc}) — will auto-reconnect")
    else:
        print(f"  ℹ  MQTT Disconnected cleanly")

def _relay_async(plug_id: str, state: str, log_msg: str):
    """Run control_relay in its own thread so we never block the MQTT loop."""
    success = control_relay(plug_id, state)
    if success:
        print(f"  {log_msg} [{plug_id}]")
    else:
        print(f"   Relay [{plug_id}] → {state} FAILED")


def on_message(client, userdata, msg):
    try:
        # FIX 6: Guard against malformed topics
        parts = msg.topic.split("/")
        if len(parts) < 4:
            print(f"  ⚠️  Unexpected topic format: {msg.topic}")
            return

        plug_id = parts[2]
        data    = json.loads(msg.payload.decode())

        voltage = float(data["voltage"])
        current = float(data["current"])
        latest_values[plug_id] = {"voltage": voltage, "current": current}

        if plug_id not in plug_buffers:
            plug_buffers[plug_id] = deque(maxlen=WINDOW_SIZE)
        plug_buffers[plug_id].append((voltage, current))

        count = len(plug_buffers[plug_id])
        print(f"  📊 MQTT [{plug_id}] V={voltage:.2f}V  I={current:.4f}A  Buffer: {count}/{WINDOW_SIZE}")

        # Auto-predict every time the sliding window is full
        if count == WINDOW_SIZE:
            if prediction_enabled.get(plug_id, True) is False:
                latest_predictions[plug_id] = {
                    "plug_id": plug_id,
                    "status": "paused",
                    "message": "Prediction paused: relay is manually OFF. Turn relay ON to resume.",
                }
                print(f"  ⏸️ Prediction paused for plug [{plug_id}] (manual relay OFF)")
                return

            result = run_window_inference(plug_id)
            latest_predictions[plug_id] = result

            status = result["status"]
            print(f"  🤖 Prediction [{plug_id}]: {status} — {result.get('message', '')}")

            # ── Automatic relay control ──────────────────────────
            # Dispatch to a separate thread so we never call publish()
            # from inside the MQTT network thread (loop_forever), which
            # causes a deadlock: publish() waits for the network loop to
            # flush, but the network loop is blocked here waiting for us.
            if status == "Abnormal":
                threading.Thread(
                    target=_relay_async,
                    args=(plug_id, "OFF", "  ANOMALY → Relay turned OFF for safety"),
                    daemon=True,
                ).start()
            elif status == "Normal":
                threading.Thread(
                    target=_relay_async,
                    args=(plug_id, "ON", " Normal → Relay ON"),
                    daemon=True,
                ).start()

    except KeyError as e:
        print(f"   Missing key in MQTT payload: {e}  |  Topic: {msg.topic}  |  Payload: {msg.payload.decode()}")
    except json.JSONDecodeError as e:
        print(f"   JSON decode error: {e}  |  Payload: {msg.payload.decode()}")
    except Exception as e:
        print(f"   MQTT message error: {e}  |  Topic: {msg.topic}")


# ──────────────────────────────────────────────
# MQTT Startup with Auto-Reconnect
# FIX 8: loop_start() + manual reconnect loop instead of
#         loop_forever() so the thread stays responsive and
#         reconnects on network drops.
# ──────────────────────────────────────────────

def start_mqtt():
    global mqtt_client

    print(f"\n  🔌 Starting MQTT (wss://{MQTT_BROKER}:{MQTT_PORT}{MQTT_PATH})")
    print(f"     Topic: {MQTT_TOPIC}\n")

    # FIX 9: Give each run a unique client_id to avoid "Invalid client identifier"
    client_id = f"pinn-server-{os.getpid()}"

    mqtt_client = mqtt.Client(client_id=client_id, transport="websockets")
    mqtt_client.tls_set()
    mqtt_client.ws_set_options(path=MQTT_PATH)

    mqtt_client.on_connect    = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message    = on_message

    # FIX 10: reconnect_delay_set ensures exponential back-off on drops
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

    while True:
        try:
            print("   Connecting to MQTT broker…")
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqtt_client.loop_forever()          # blocks; auto-reconnects internally
        except Exception as e:
            print(f"   MQTT connection error: {e}")
            mqtt_connected_event.clear()
            print("   Retrying in 10 seconds…")
            time.sleep(10)

mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()


# ──────────────────────────────────────────────
# Inference Logic
# ──────────────────────────────────────────────

def run_window_inference(plug_id: str) -> dict:
    if prediction_enabled.get(plug_id, True) is False:
        return {
            "plug_id": plug_id,
            "status":  "paused",
            "message": "Prediction paused: relay is manually OFF. Turn relay ON to resume.",
        }

    buffer = plug_buffers.get(plug_id)
    if buffer is None or len(buffer) < WINDOW_SIZE:
        collected = len(buffer) if buffer else 0
        return {
            "plug_id": plug_id,
            "status":  "collecting",
            "message": f"Collecting data… {collected}/{WINDOW_SIZE} readings received",
        }

    if plug_id not in user_thresholds:
        return {
            "plug_id": plug_id,
            "status":  "error",
            "message": f"No threshold set for '{plug_id}'. POST /set-threshold first.",
        }

    device_name = user_thresholds[plug_id]["device"]
    user_limit  = user_thresholds[plug_id]["threshold"]

    if device_name not in device_registry:
        return {
            "plug_id": plug_id,
            "status":  "error",
            "message": f"No trained model for device '{device_name}'.",
        }

    entry    = device_registry[device_name]
    model    = entry["model"]
    scaler_V = entry["scaler_V"]
    scaler_I = entry["scaler_I"]
    scaler_P = entry["scaler_P"]

    window   = np.array(list(buffer))           # (WINDOW_SIZE, 2)
    V_window = window[:, 0].reshape(-1, 1)
    I_window = window[:, 1].reshape(-1, 1)

    V_scaled = scaler_V.transform(V_window)
    I_scaled = scaler_I.transform(I_window)
    X        = np.hstack([V_scaled, I_scaled])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        P_scaled_pred = model(X_tensor).numpy()

    P_pred_all    = scaler_P.inverse_transform(P_scaled_pred).flatten()
    avg_predicted = float(np.mean(P_pred_all))
    avg_physics   = float(np.mean(V_window.flatten() * I_window.flatten()))

    # Tolerance band (default ±5% of threshold)
    tolerance   = user_thresholds[plug_id].get("tolerance_w") or (0.05 * user_limit)
    # Hard floor — below this means unplugged or sensor fault (default 3W)
    # A charger in trickle mode (~9W) is NORMAL, so TOO LOW is NOT triggered
    # unless power falls below this hard floor.
    hard_floor  = user_thresholds[plug_id].get("min_power_w") or 3.0

    normal_max    = user_limit + tolerance
    deviation     = avg_predicted - user_limit
    deviation_pct = (deviation / user_limit * 100) if user_limit != 0 else 0.0

    if avg_predicted < hard_floor:
        # Near-zero = device unplugged or dead short
        status  = "Abnormal"
        flag    = "ALERT"
        message = f"FAULT — power {avg_predicted:.2f}W is below hard floor {hard_floor:.1f}W."
    elif avg_predicted > normal_max:
        # Overcurrent / overpower — dangerous
        status  = "Abnormal"
        flag    = "ALERT"
        message = f"TOO HIGH — {avg_predicted:.2f}W exceeds max {normal_max:.2f}W by {abs(deviation):.2f}W."
    else:
        # Everything from hard_floor up to normal_max is Normal.
        # Covers trickle (~9W), steady state (~14W), settling (~17W).
        status  = "Normal"
        flag    = "OK"
        message = f"Normal — {avg_predicted:.2f}W (floor: {hard_floor:.1f}W, max: {normal_max:.2f}W)."

    return {
        "plug_id":          plug_id,
        "device":           device_name,
        "window_size":      WINDOW_SIZE,
        "avg_predicted_w":  round(avg_predicted, 4),
        "avg_physics_w":    round(avg_physics, 4),
        "user_threshold_w": round(user_limit, 4),
        "tolerance_w":      round(tolerance, 4),
        "hard_floor_w":     round(hard_floor, 4),
        "normal_max_w":     round(normal_max, 4),
        "deviation_w":      round(deviation, 4),
        "deviation_pct":    round(deviation_pct, 2),
        "status":           status,
        "flag":             flag,
        "message":          message,
    }


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def index():
    return {
        "status":         "online",
        "model":          "PINN (Physics-Informed Neural Network)",
        "devices_loaded": list(device_registry.keys()),
        "window_size":    WINDOW_SIZE,
        "mqtt_connected": mqtt_connected_event.is_set(),
    }


@app.get("/sensor-data")
async def get_sensor_data():
    """Latest MQTT readings + buffer fill status for all plugs."""
    results = [
        {
            "plug_id":      plug_id,
            "voltage":      values["voltage"],
            "current":      values["current"],
            "buffer_count": len(plug_buffers.get(plug_id, [])),
            "window_size":  WINDOW_SIZE,
            "ready":        len(plug_buffers.get(plug_id, [])) >= WINDOW_SIZE,
        }
        for plug_id, values in latest_values.items()
    ]
    return {"data": results}


@app.post("/set-threshold")
async def set_threshold(data: ThresholdSetting):
    """
    Set the acceptable power threshold for a plug.

    Body example:
        {"plug_id": "1", "device": "fan", "threshold": 95.0}
    """
    device_name = data.device.lower().strip()

    # Accept both *_test and *_train style inputs and resolve to an actual loaded key.
    candidates = [device_name]
    if device_name.endswith("_test"):
        candidates.append(device_name.replace("_test", "_train"))
    elif device_name.endswith("_train"):
        candidates.append(device_name.replace("_train", "_test"))
    else:
        candidates.extend([f"{device_name}_test", f"{device_name}_train"])

    resolved_name = next((name for name in candidates if name in device_registry), None)
    if resolved_name is None:
        # Handle names like "phone_charger" -> "phone_charger_ac_train" when unique.
        prefix_matches = [name for name in device_registry if name.startswith(f"{device_name}_")]
        if len(prefix_matches) == 1:
            resolved_name = prefix_matches[0]

    if resolved_name is None:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{data.device}' not found. Available: {list(device_registry.keys())}",
        )

    device_name = resolved_name

    tolerance_w = data.tolerance_w
    min_power_w = data.min_power_w if data.min_power_w is not None else 3.0

    if tolerance_w is not None and tolerance_w < 0:
        raise HTTPException(status_code=400, detail="tolerance_w must be >= 0")
    if min_power_w < 0:
        raise HTTPException(status_code=400, detail="min_power_w must be >= 0")

    applied_tolerance = tolerance_w if tolerance_w is not None else (0.05 * data.threshold)

    user_thresholds[data.plug_id] = {
        "device":      device_name,
        "threshold":   data.threshold,
        "tolerance_w": tolerance_w,
        "min_power_w": min_power_w,
    }
    return {
        "message":      f"Threshold set for plug '{data.plug_id}'",
        "plug_id":      data.plug_id,
        "device":       device_name,
        "threshold":    data.threshold,
        "tolerance_w":  round(applied_tolerance, 4),
        "min_power_w":  round(min_power_w, 4),
        "normal_max_w": round(data.threshold + applied_tolerance, 4),
        "note":         "TOO LOW anomaly only triggers below min_power_w (hard floor)",
    }


@app.get("/get-thresholds")
async def get_thresholds():
    """Return all configured thresholds."""
    thresholds = []
    for pid, cfg in user_thresholds.items():
        threshold   = cfg["threshold"]
        tolerance_w = cfg.get("tolerance_w") or (0.05 * threshold)
        min_power_w = cfg.get("min_power_w") or 3.0
        thresholds.append({
            "plug_id":      pid,
            "device":       cfg["device"],
            "threshold":    threshold,
            "tolerance_w":  round(tolerance_w, 4),
            "min_power_w":  round(min_power_w, 4),
            "normal_max_w": round(threshold + tolerance_w, 4),
        })
    return {"thresholds": thresholds}


@app.get("/predict/mqtt")
async def predict_all_plugs():
    """Window-based prediction for ALL plugs currently publishing data."""
    if not device_registry:
        raise HTTPException(status_code=503, detail="No models loaded.")
    if not plug_buffers:
        return {"message": "No MQTT data received yet.", "results": []}

    return {"results": [run_window_inference(pid) for pid in plug_buffers]}


@app.get("/predict/mqtt/{plug_id}")
async def predict_single_plug(plug_id: str):
    """Window-based prediction for a specific plug.  e.g. GET /predict/mqtt/1"""
    if not device_registry:
        raise HTTPException(status_code=503, detail="No models loaded.")
    return run_window_inference(plug_id)


@app.get("/latest-predictions")
async def get_latest_predictions():
    """Last auto-triggered prediction per plug (updated every window fill)."""
    if not latest_predictions:
        return {"message": "No predictions yet.", "results": []}
    return {"results": list(latest_predictions.values())}


@app.post("/predict")
async def predict_batch(data: BatchPredictionRequest):
    """
    Manual single-point prediction (no MQTT required — useful for testing).

    Body example:
        {"appliances": [{"appliance_id": "fan", "voltage": 220, "current": 0.45}]}
    """
    if not device_registry:
        raise HTTPException(status_code=503, detail="No models loaded.")

    results = []
    for item in data.appliances:
        device_name = item.appliance_id.lower()
        if not device_name.endswith("_test"):
            device_name = f"{device_name}_test"

        if device_name not in device_registry:
            results.append({
                "appliance_id": item.appliance_id,
                "error": f"No model for '{device_name}'. Available: {list(device_registry.keys())}",
            })
            continue

        try:
            entry    = device_registry[device_name]
            V_arr    = np.array([[item.voltage]])
            I_arr    = np.array([[item.current]])
            V_scaled = entry["scaler_V"].transform(V_arr)
            I_scaled = entry["scaler_I"].transform(I_arr)
            X        = torch.tensor(np.hstack([V_scaled, I_scaled]), dtype=torch.float32)

            with torch.no_grad():
                P_scaled = entry["model"](X).numpy()

            predicted = float(entry["scaler_P"].inverse_transform(P_scaled)[0][0])
            results.append({
                "appliance_id":    item.appliance_id,
                "voltage":         item.voltage,
                "current":         item.current,
                "predicted_power": round(predicted, 4),
                "physics_power":   round(item.voltage * item.current, 4),
            })
        except Exception as e:
            results.append({"appliance_id": item.appliance_id, "error": str(e)})

    return {"results": results}


@app.post("/relay/control")
async def manual_relay_control(data: RelayControl):
    """
    Manually override relay state.

    Body example:  {"plug_id": "1", "state": "OFF"}

    Note: automatic anomaly control resumes on the next prediction window.
    """
    if data.state not in ("ON", "OFF"):
        raise HTTPException(status_code=400, detail="state must be 'ON' or 'OFF'")

    # Apply prediction mode immediately and deterministically.
    # OFF pauses predictions until an explicit ON is received.
    if data.state == "OFF":
        prediction_enabled[data.plug_id] = False
        latest_predictions[data.plug_id] = {
            "plug_id": data.plug_id,
            "status": "paused",
            "message": "Prediction paused: relay manually turned OFF.",
        }
    else:
        prediction_enabled[data.plug_id] = True

    relay_sent = control_relay(data.plug_id, data.state)

    if relay_sent:
        return {
            "message": "Relay command sent",
            "plug_id": data.plug_id,
            "state":   data.state,
            "prediction_enabled": prediction_enabled.get(data.plug_id, True),
            "relay_command_sent": True,
            "note":    "Prediction pauses on OFF and resumes on ON",
        }

    return {
        "message": "Prediction mode updated, but relay command was not delivered",
        "plug_id": data.plug_id,
        "state": data.state,
        "prediction_enabled": prediction_enabled.get(data.plug_id, True),
        "relay_command_sent": False,
        "note": "Prediction state still follows OFF/ON even if MQTT is unavailable",
    }


@app.get("/mqtt-status")
async def mqtt_status():
    """Check MQTT connection health."""
    connected = mqtt_connected_event.is_set() and mqtt_client is not None and mqtt_client.is_connected()
    return {
        "connected": connected,
        "broker":    MQTT_BROKER,
        "port":      MQTT_PORT,
        "transport": "wss (WebSocket Secure)",
    }


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)