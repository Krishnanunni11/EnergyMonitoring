import { NextResponse } from 'next/server';
import { getBackendUrl } from '../../../lib/backend';

type RelayState = 'ON' | 'OFF';

export async function POST(req: Request) {
  try {
    const { plug_id, state } = await req.json();

    if (!plug_id || !state) {
      return NextResponse.json(
        { error: 'Missing required fields: plug_id, state' },
        { status: 400 }
      );
    }

    const normalizedState = String(state).toUpperCase() as RelayState;
    if (normalizedState !== 'ON' && normalizedState !== 'OFF') {
      return NextResponse.json(
        { error: "state must be 'ON' or 'OFF'" },
        { status: 400 }
      );
    }

    const response = await fetch(`${getBackendUrl()}/relay/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plug_id: String(plug_id), state: normalizedState }),
      cache: 'no-store'
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      return NextResponse.json(
        { error: payload?.detail || payload?.error || 'Backend relay control failed' },
        { status: response.status }
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    console.error('Relay control proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to control relay' },
      { status: 500 }
    );
  }
}
