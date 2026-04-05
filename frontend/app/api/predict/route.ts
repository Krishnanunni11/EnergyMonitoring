import { NextResponse } from 'next/server';
import { getBackendUrl } from '../../../lib/backend';

export async function POST(req: Request) {
    try {
        const { appliances } = await req.json();

        if (!appliances || !Array.isArray(appliances)) {
            return NextResponse.json({ error: "Invalid input: 'appliances' must be an array" }, { status: 400 });
        }

        const formattedAppliances = appliances
            .map((app: any) => {
                const applianceId =
                    typeof app?.appliance_id === 'string'
                        ? app.appliance_id.trim()
                        : typeof app?.id === 'string'
                            ? app.id.trim()
                            : '';

                if (!applianceId) {
                    return null;
                }

                // Handle both sequence array and single voltage/current
                let voltage: number;
                let current: number;

                if (Array.isArray(app?.sequence) && app.sequence.length > 0) {
                    // Take the last value from sequence
                    const lastPoint = app.sequence[app.sequence.length - 1];
                    if (Array.isArray(lastPoint)) {
                        voltage = Number(lastPoint[0]);
                        current = Number(lastPoint[1]);
                    } else if (typeof lastPoint === 'object') {
                        voltage = Number(lastPoint?.voltage);
                        current = Number(lastPoint?.current);
                    } else {
                        return null;
                    }
                } else {
                    // Use direct voltage/current values
                    voltage = Number(app?.voltage);
                    current = Number(app?.current);
                }

                if (!Number.isFinite(voltage) || !Number.isFinite(current)) {
                    return null;
                }

                return {
                    appliance_id: applianceId,
                    voltage: voltage,
                    current: current
                };
            })
            .filter((app: any) => app !== null);

        if (formattedAppliances.length === 0) {
            return NextResponse.json({ error: 'No valid appliance data found' }, { status: 400 });
        }

        const response = await fetch(`${getBackendUrl()}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ appliances: formattedAppliances }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json({ error: `Python Server Error: ${errorText}` }, { status: 500 });
        }

        const data = await response.json();

        return NextResponse.json({ results: data.results });

    } catch (error) {
        console.error("Inference route error:", error);
        return NextResponse.json({ error: "Could not connect to AI backend" }, { status: 500 });
    }
}
