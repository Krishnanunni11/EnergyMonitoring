# EnergyMonitor

PINN-based smart plug monitoring with real-time MQTT ingestion, FastAPI inference, automatic relay control, and a Next.js dashboard.

## Project Structure

- `backend/`: FastAPI service, training/inference scripts, model artifacts
- `frontend/`: Next.js dashboard UI
- `data/`: Device training CSV files
- `models/`: Trained model/scaler files used by backend API

## Features

- Real-time MQTT sensor ingestion from smart plugs
- Sliding-window power inference using PINN models
- Configurable anomaly policy per plug:
	- `threshold` (max expected power)
	- `tolerance_w` (allowed range above threshold)
	- `min_power_w` (hard floor fault cutoff)
- Automatic relay ON/OFF publishing based on prediction status
- Manual relay control endpoint
- Batch prediction endpoint for quick testing

## Requirements

- Python 3.10+
- Node.js 18+

## Backend Setup

From `backend/`:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run API server:

```bash
py api.py
```

Backend default URL: `http://127.0.0.1:8000`

For Render, run the backend as a web service and bind to the platform port via `PORT`.

## Frontend Setup

From `frontend/`:

```bash
npm install
npm run dev
```

Frontend default URL: `http://localhost:3000`

## Model Training

From `backend/`:

```bash
py train.py
```

This trains one model per CSV in `data/` and writes:

- `models/<device_name>_pinn.pth`
- `models/<device_name>_scaler_V.pkl`
- `models/<device_name>_scaler_I.pkl`
- `models/<device_name>_scaler_P.pkl`

## MQTT Configuration

Configured in `backend/api.py`:

- Broker: `broker.hivemq.com`
- Port: `8884` (secure websocket)
- Path: `/mqtt`
- Input topic: `smart/plug/+/codedata`
- Relay command topic: `smart/plug/command`

Expected sensor payload example:

```json
{
	"voltage": 229.8,
	"current": 0.145
}
```

Published relay payload example:

```json
{
	"plug": 1,
	"cmd": "off"
}
```

## Anomaly Logic (Current)

For each plug:

- Window size is `WINDOW_SIZE` samples (default `10`)
- Prediction uses average predicted power over the window
- `normal_max = threshold + tolerance`
- `hard_floor = min_power_w`

Status rules:

- `Abnormal` if `avg_predicted < hard_floor`
- `Abnormal` if `avg_predicted > normal_max`
- Otherwise `Normal`

Returned `flag` values are plain text:

- `ALERT`
- `OK`

## Key API Endpoints

Base URL: `http://127.0.0.1:8000`

### Health

- `GET /`
- `GET /mqtt-status`

### Threshold Configuration

- `POST /set-threshold`
- `GET /get-thresholds`

Request example:

```json
{
	"plug_id": "1",
	"device": "phone_charger",
	"threshold": 15.0,
	"tolerance_w": 2.0,
	"min_power_w": 3.0
}
```

Device name resolution in `/set-threshold` supports:

- Exact model key
- `_test` and `_train` style inputs
- Unique prefix mapping (example: `phone_charger` to a single matching model key)

### Predictions

- `GET /predict/mqtt`
- `GET /predict/mqtt/{plug_id}`
- `GET /latest-predictions`
- `POST /predict` (manual batch)

### Relay

- `POST /relay/control`

Manual relay body:

```json
{
	"plug_id": "1",
	"state": "OFF"
}
```

## Notes

- Models must exist in `backend/models` with matching scaler files.
- If frontend fails to start, run `npm install` in `frontend/` first.
- If a device is not found, check loaded keys via the API root response (`devices_loaded`).