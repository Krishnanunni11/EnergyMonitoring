const FALLBACK_BACKEND_URL = "https://energymonitoring-sf4p.onrender.com";

function normalizeBackendUrl(url: string | undefined) {
  return (url ?? FALLBACK_BACKEND_URL).replace(/\/$/, "");
}

export function getBackendUrl() {
  return normalizeBackendUrl(process.env.BACKEND_URL);
}

export function getClientBackendUrl() {
  return normalizeBackendUrl(process.env.NEXT_PUBLIC_BACKEND_URL);
}
