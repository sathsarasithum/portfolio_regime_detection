import { formatDate, formatLKR, formatPct } from '../format.js'

export default function DailyValuesTable({ days, activeIndex, onSelect }) {
  return (
    <div className="card">
      <h2>Daily Portfolio Value ({days.length} test days)</h2>
      <div className="daily-table-scroll">
        <table className="daily-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>Date</th>
              <th>Portfolio Value</th>
              <th>Daily Return</th>
              <th>Trade Cost</th>
            </tr>
          </thead>
          <tbody>
            {days.map((day, i) => (
              <tr
                key={day.index}
                className={i === activeIndex ? 'active-row' : ''}
                onClick={() => onSelect(i)}
              >
                <td>{i + 1}</td>
                <td>{formatDate(day.date)}</td>
                <td className="value-cell">{formatLKR(day.portfolio_value)}</td>
                <td className={day.portfolio_return >= 0 ? 'positive' : 'negative'}>
                  {formatPct(day.portfolio_return, 2)}
                </td>
                <td className="value-cell">{formatLKR(day.total_cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
