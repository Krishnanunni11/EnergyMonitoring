import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const sensorResponse = await fetch('http://127.0.0.1:8000/sensor-data', {
      cache: 'no-store'
    });

    if (!sensorResponse.ok) {
      const errorText = await sensorResponse.text();
      return NextResponse.json({ error: `Sensor backend error: ${errorText}` }, { status: 500 });
    }

    const sensorPayload = await sensorResponse.json();
    const sensorData = Array.isArray(sensorPayload?.data) ? sensorPayload.data : [];

    const appliancesForPrediction = sensorData
      .map((row: any) => {
        const plugId = typeof row?.plug_id === 'string' ? row.plug_id.trim() : '';
        const voltage = Number(row?.voltage);
        const current = Number(row?.current);

        if (!plugId || !Number.isFinite(voltage) || !Number.isFinite(current)) {
          return null;
        }

        return {
          appliance_id: plugId,
          voltage: voltage,
          current: current
        };
      })
      .filter((item: any) => item !== null);

    if (appliancesForPrediction.length === 0) {
      return NextResponse.json({ data: sensorData });
    }

    const predictResponse = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ appliances: appliancesForPrediction }),
      cache: 'no-store'
    });

    if (!predictResponse.ok) {
      const errorText = await predictResponse.text();
      return NextResponse.json({ error: `Predict backend error: ${errorText}`, data: sensorData }, { status: 500 });
    }

    const predictionPayload = await predictResponse.json();
    const predictionById = new Map<string, any>(
      (Array.isArray(predictionPayload?.results) ? predictionPayload.results : []).map((result: any) => [result.appliance_id, result])
    );

    const enrichedData = sensorData.map((row: any) => {
      const prediction = predictionById.get(row?.plug_id);
      return {
        ...row,
        predicted_power: prediction?.predicted_power ?? null,
        physics_power: prediction?.physics_power ?? null
      };
    });

    return NextResponse.json({ data: enrichedData });
  } catch (error) {
    return NextResponse.json({ error: "Backend unreachable" }, { status: 500 });
  }
}