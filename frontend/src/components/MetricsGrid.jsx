import { formatLKR, formatPct } from '../format.js'

export default function MetricsGrid({ summary }) {
  const tiles = [
    { label: 'Total Return', value: formatPct(summary.total_return) },
    { label: 'Annualized Return', value: formatPct(summary.annual_return) },
    { label: 'Annualized Volatility', value: formatPct(summary.annual_volatility) },
    { label: 'Sharpe Ratio', value: summary.sharpe_ratio.toFixed(2) },
    { label: 'Max Drawdown', value: formatPct(summary.max_drawdown) },
    { label: 'Transaction Costs', value: formatLKR(summary.total_transaction_costs) },
    { label: 'Final Portfolio Value', value: formatLKR(summary.final_portfolio_value) },
    { label: 'Trading Days', value: summary.n_trading_days },
  ]

  return (
    <div className="card">
      <h2>Test-Set Performance Summary</h2>
      <div className="metrics-grid">
        {tiles.map((t) => (
          <div className="metric-tile" key={t.label}>
            <div className="metric-value">{t.value}</div>
            <div className="metric-label">{t.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
