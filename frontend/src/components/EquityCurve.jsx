import { formatDate, formatLKR } from '../format.js'

const WIDTH = 800
const HEIGHT = 220
const PAD = 32

export default function EquityCurve({ days }) {
  const values = days.map((d) => d.portfolio_value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const points = days.map((d, i) => {
    const x = PAD + (i / (days.length - 1)) * (WIDTH - 2 * PAD)
    const y = HEIGHT - PAD - ((d.portfolio_value - min) / range) * (HEIGHT - 2 * PAD)
    return [x, y]
  })

  const pathD = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const initial = days[0]
  const final = days[days.length - 1]

  return (
    <div className="card">
      <h2>Portfolio Value Over Test Period</h2>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="equity-svg" preserveAspectRatio="none">
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} className="axis" />
        <path d={pathD} className="equity-line" fill="none" />
      </svg>
      <div className="equity-labels">
        <span>{formatDate(initial.date)} · {formatLKR(initial.portfolio_value)}</span>
        <span>{formatDate(final.date)} · {formatLKR(final.portfolio_value)}</span>
      </div>
    </div>
  )
}
