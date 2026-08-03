import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { formatLKR } from '../format.js'
import AllocationTable from './AllocationTable.jsx'
import RegimePanel from './RegimePanel.jsx'

export default function LiveModePanel({ initialCapital, assetNames }) {
  const [status, setStatus] = useState('loading') // loading | error | ready
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)

  const load = () => {
    setStatus('loading')
    setError(null)
    api
      .liveToday()
      .then((res) => {
        setData(res)
        setStatus('ready')
      })
      .catch((err) => {
        setError(err.message)
        setStatus('error')
      })
  }

  useEffect(load, [])

  if (status === 'loading') {
    return (
      <div className="card">
        <h2>Live / Today</h2>
        <p>Fetching current CSE data and running the model…</p>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="card live-error">
        <h2>Live allocation unavailable</h2>
        <p className="error-message">{error}</p>
        <p className="live-note">
          This mode needs a promoted model bundle (checkpoint + scaler + manifest) and a
          continuous local price history reaching yesterday. See the project notes for the
          steps to enable it once training finishes.
        </p>
        <button onClick={load}>Retry</button>
      </div>
    )
  }

  const allocations = assetNames.map((asset) => ({
    asset,
    name: asset,
    weight: data.weights[asset] ?? 0,
    value_lkr: (data.weights[asset] ?? 0) * initialCapital,
  }))

  return (
    <>
      <p className="as-of">
        As of {data.as_of_date} · Portfolio {formatLKR(initialCapital)}
      </p>
      <RegimePanel regimeProbs={data.regime_probs} />
      <AllocationTable allocations={allocations} />
    </>
  )
}
