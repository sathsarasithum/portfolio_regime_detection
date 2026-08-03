import { formatLKR, formatPct, formatSignedPP } from '../format.js'

const HOLD_THRESHOLD = 0.001 // 0.1 percentage point — smaller moves read as "Hold"

function actionFor(delta) {
  if (delta === null) return { label: 'FIRST DAY', cls: 'badge-neutral' }
  if (delta > HOLD_THRESHOLD) return { label: 'BUY', cls: 'badge-buy' }
  if (delta < -HOLD_THRESHOLD) return { label: 'SELL', cls: 'badge-sell' }
  return { label: 'HOLD', cls: 'badge-neutral' }
}

export default function AllocationTable({ allocations, previousAllocations }) {
  const maxWeight = Math.max(...allocations.map((a) => a.weight), 0.0001)
  const prevMap = new Map((previousAllocations ?? []).map((a) => [a.asset, a]))

  return (
    <div className="card">
      <h2>Portfolio Allocation</h2>
      <table className="alloc-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Weight</th>
            <th></th>
            <th>Value (LKR)</th>
            <th>vs Prev Day</th>
          </tr>
        </thead>
        <tbody>
          {allocations.map((a) => {
            const prev = prevMap.get(a.asset)
            const delta = previousAllocations ? a.weight - (prev?.weight ?? 0) : null
            const action = actionFor(delta)
            return (
              <tr key={a.asset}>
                <td>
                  <span className="ticker">{a.asset}</span>
                  <span className="asset-name">{a.name}</span>
                </td>
                <td className="weight-cell">{formatPct(a.weight, 2)}</td>
                <td className="bar-cell">
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${(a.weight / maxWeight) * 100}%` }}
                    />
                  </div>
                </td>
                <td className="value-cell">{formatLKR(a.value_lkr)}</td>
                <td className="delta-cell">
                  <span className={`badge ${action.cls}`}>{action.label}</span>
                  {delta !== null && (
                    <span className={delta >= 0 ? 'positive' : 'negative'}>
                      {' '}
                      {formatSignedPP(delta)}
                    </span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
