export default function Header({ modelInfo, mode, setMode }) {
  return (
    <header className="header">
      <div className="header-top">
        <div>
          <h1>CSE Regime-Aware Portfolio</h1>
          <p className="subtitle">Research demo — PPO agent over 10 CSE banking-sector stocks</p>
        </div>
        <nav className="mode-toggle">
          <button
            className={mode === 'backtest' ? 'active' : ''}
            onClick={() => setMode('backtest')}
          >
            Validated Backtest
          </button>
          <button
            className={mode === 'live' ? 'active' : ''}
            onClick={() => setMode('live')}
          >
            Live / Today
          </button>
        </nav>
      </div>
      {modelInfo && <p className="disclaimer">{modelInfo.disclaimer}</p>}
    </header>
  )
}
