# Automatic Relay Control for Anomaly Detection

## Overview
The system now automatically controls relay switches to cut off power when anomalies are detected by the PINN AI model.

## How It Works

### Automatic Control Flow
```
1. MQTT sensor data arrives → smart/plug/{plug_id}/codedata
2. Backend fills 10-reading buffer
3. PINN model predicts power consumption
4. Compare prediction with user threshold (±5%)
5. Publish relay command → smart/plug/{plug_id}/relay
   - Abnormal → {"relay": 0} 🔴
   - Normal   → {"relay": 1}  🟢
```

### MQTT Topics

**Input (from ESP32):**
- Topic: `smart/plug/{plug_id}/codedata`
- Payload: `{"plug": 1, "voltage": 234.1, "current": 0.166, "relay": 1, "timer": 0}`

**Output (to ESP32):**
- Topic: `smart/plug/{plug_id}/relay`
- Payload: `{"relay": 0}` (OFF) or `{"relay": 1}` (ON)

### Example Scenario

**Laptop charger on fan model with 120W threshold:**

```
Buffer fills with 10 readings...
🤖 Fan model predicts: 78W
📊 User threshold: 120W (±6W = 114-126W)
⚠️ Deviation: -42W (35% below threshold)
🔴 Status: Abnormal - TOO LOW
🔌 Relay [1] → OFF (relay: 0)
⚠️ ANOMALY DETECTED! Relay [1] turned OFF for safety
```

## ESP32 Code Requirements

Your ESP32 must subscribe to the relay control topic:

```cpp
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#define RELAY_PIN 4  // GPIO pin for relay

WiFiClientSecure espClient;
PubSubClient client(espClient);

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Relay OFF by default
  
  // Connect to WiFi and MQTT...
  client.setServer("broker.hivemq.com", 8883);
  client.setCallback(mqttCallback);
  
  // Subscribe to relay control topic
  client.subscribe("smart/plug/1/relay");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  // Parse relay command
  if (strcmp(topic, "smart/plug/1/relay") == 0) {
    StaticJsonDocument<200> doc;
    deserializeJson(doc, payload, length);
    
    int relayState = doc["relay"];  // 0 or 1
    
    if (relayState == 1) {
      digitalWrite(RELAY_PIN, HIGH);  // Turn ON relay
      Serial.println("✅ Relay ON");
    } else if (relayState == 0) {
      digitalWrite(RELAY_PIN, LOW);   // Turn OFF relay
      Serial.println("🔴 Relay OFF - Anomaly detected!");
    }
  }
}

void loop() {
  // Read current relay state
  int currentRelayState = digitalRead(RELAY_PIN);
  
  // Publish sensor data every second
  StaticJsonDocument<200> doc;
  doc["plug"] = 1;
  doc["voltage"] = readVoltage();
  doc["current"] = readCurrent();
  doc["relay"] = currentRelayState;  // Include current relay state
  doc["timer"] = 0;
  
  char buffer[256];
  serializeJson(doc, buffer);
  client.publish("smart/plug/1/codedata", buffer);
  
  delay(1000);
  client.loop();
}
```

## Manual Relay Control API

### Turn OFF relay manually:
```bash
curl -X POST http://127.0.0.1:8000/relay/control \
  -H "Content-Type: application/json" \
  -d '{"plug_id": "1", "state": "OFF"}'
```

### Turn ON relay manually:
```bash
curl -X POST http://127.0.0.1:8000/relay/control \
  -H "Content-Type: application/json" \
  -d '{"plug_id": "1", "state": "ON"}'
```

**Note:** Automatic control will resume on the next prediction cycle.

## Safety Features

1. **Immediate cutoff**: Relay turns OFF within 1 second of anomaly detection
2. **Automatic restoration**: Relay turns ON when pattern returns to normal
3. **Manual override**: Use API endpoint for emergency control
4. **Persistent monitoring**: Continuous 24/7 analysis every 10 readings

## Testing

1. **Set low threshold** (e.g., 80W for laptop on fan model)
2. **Plug in device** and wait for 10 readings
3. **Watch backend console** for relay commands
4. **Verify ESP32** receives command and controls relay
5. **Check relay state** changes based on predictions

## Console Output Example

```
📊 MQTT [1] V=234.1 I=0.166 | Buffer: 10/10
🤖 Auto-prediction [1]: Abnormal - TOO LOW — 35.0% below your threshold!
🔌 Relay [1] → OFF (relay: 0) (topic: smart/plug/1/relay)
⚠️ ANOMALY DETECTED! Relay [1] turned OFF for safety
```

## Configuration

- **Threshold**: Set via frontend modal or `/set-threshold` API
- **Tolerance**: Fixed at ±5% of threshold
- **Buffer size**: 10 readings (configurable in `WINDOW_SIZE`)
- **Auto-control**: Enabled by default, runs on every prediction
