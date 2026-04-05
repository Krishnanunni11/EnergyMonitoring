import { NextResponse } from 'next/server';
import { getBackendUrl } from '../../../../lib/backend';

export async function GET() {
  try {
    const response = await fetch(`${getBackendUrl()}/latest-predictions`, {
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
    console.error('Latest predictions error:', error);
    return NextResponse.json(
      { error: 'Failed to get latest predictions' },
      { status: 500 }
    );
  }
}