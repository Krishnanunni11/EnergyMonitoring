import { NextResponse } from 'next/server';
import { getBackendUrl } from '../../../lib/backend';

export async function GET() {
    try {
        const response = await fetch(`${getBackendUrl()}/get-thresholds`, {
            cache: 'no-store'
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
        console.error('Get thresholds error:', error);
        return NextResponse.json(
            { error: 'Failed to get thresholds' },
            { status: 500 }
        );
    }
}
