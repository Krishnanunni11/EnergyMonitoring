const FALLBACK_BACKEND_URL = "http://127.0.0.1:8000";

function normalizeBackendUrl(url: string | undefined) {
  return (url ?? FALLBACK_BACKEND_URL).replace(/\/$/, "");
}

export function getBackendUrl() {
  return normalizeBackendUrl(process.env.BACKEND_URL);
}

export function getClientBackendUrl() {
  return normalizeBackendUrl(process.env.NEXT_PUBLIC_BACKEND_URL);
}