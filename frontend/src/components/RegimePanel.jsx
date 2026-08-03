import { formatPct } from '../format.js'

const REGIME_LABELS = { bull: 'Bull', bear: 'Bear', sideways: 'Sideways' }

export default function RegimePanel({ regimeProbs }) {
  const entries = Object.entries(regimeProbs).sort((a, b) => b[1] - a[1])
  const top = entries[0][0]

  return (
    <div className="card">
      <h2>Detected Regime</h2>
      <div className="regime-top">{REGIME_LABELS[top] || top}</div>
      {entries.map(([label, prob]) => (
        <div className="regime-row" key={label}>
          <span className="regime-label">{REGIME_LABELS[label] || label}</span>
          <div className="bar-track">
            <div className="bar-fill regime-fill" style={{ width: `${prob * 100}%` }} />
          </div>
          <span className="regime-pct">{formatPct(prob)}</span>
        </div>
      ))}
    </div>
  )
}
