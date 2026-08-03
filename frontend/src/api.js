const BASE = '/api'

async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return body
}

export const api = {
  health: () => getJSON('/health'),
  modelInfo: () => getJSON('/model-info'),
  backtest: () => getJSON('/backtest'),
  liveToday: () => getJSON('/live/today'),
}
