import { NextResponse } from 'next/server';
import { getBackendUrl } from '../../../lib/backend';

export async function POST(req: Request) {
    try {
        const { plug_id, device, threshold } = await req.json();

        if (!plug_id || !device || !threshold) {
            return NextResponse.json(
                { error: 'Missing required fields: plug_id, device, threshold' },
                { status: 400 }
            );
        }

        const response = await fetch(`${getBackendUrl()}/set-threshold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plug_id, device, threshold: Number(threshold) }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json(
                { error: `Backend error: ${errorText}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Set threshold error:', error);
        return NextResponse.json(
            { error: 'Failed to set threshold' },
            { status: 500 }
        );
    }
}
