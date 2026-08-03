import { useEffect, useState } from 'react'
import { api } from './api.js'
import Header from './components/Header.jsx'
import AllocationTable from './components/AllocationTable.jsx'
import EquityCurve from './components/EquityCurve.jsx'
import MetricsGrid from './components/MetricsGrid.jsx'
import DateSlider from './components/DateSlider.jsx'
import LiveModePanel from './components/LiveModePanel.jsx'
import DailyValuesTable from './components/DailyValuesTable.jsx'

export default function App() {
  const [mode, setMode] = useState('backtest')
  const [modelInfo, setModelInfo] = useState(null)
  const [backtest, setBacktest] = useState(null)
  const [backtestError, setBacktestError] = useState(null)
  const [dayIndex, setDayIndex] = useState(0)

  useEffect(() => {
    api.modelInfo().then(setModelInfo).catch(() => {})
    api
      .backtest()
      .then((data) => {
        setBacktest(data)
        setDayIndex(data.days.length - 1) // land on the most recent test day
      })
      .catch((err) => setBacktestError(err.message))
  }, [])

  return (
    <div className="app">
      <Header modelInfo={modelInfo} mode={mode} setMode={setMode} />

      <main className="main">
        {mode === 'backtest' && (
          <>
            {backtestError && (
              <div className="card live-error">
                <h2>Backtest data unavailable</h2>
                <p className="error-message">{backtestError}</p>
              </div>
            )}
            {!backtestError && !backtest && <div className="card">Loading test-set results…</div>}
            {backtest && (
              <>
                <DateSlider days={backtest.days} index={dayIndex} onChange={setDayIndex} />
                <div className="grid-2">
                  <AllocationTable
                    allocations={backtest.days[dayIndex].allocations}
                    previousAllocations={
                      dayIndex > 0 ? backtest.days[dayIndex - 1].allocations : null
                    }
                  />
                  <div className="side-col">
                    <MetricsGrid summary={backtest.summary} />
                  </div>
                </div>
                <EquityCurve days={backtest.days} />
                <DailyValuesTable
                  days={backtest.days}
                  activeIndex={dayIndex}
                  onSelect={setDayIndex}
                />
              </>
            )}
          </>
        )}

        {mode === 'live' && modelInfo && (
          <LiveModePanel
            initialCapital={modelInfo.initial_capital}
            assetNames={modelInfo.assets.map((a) => a.ticker)}
          />
        )}
      </main>

      <footer className="footer">
        Research prototype — not investment advice. Universe: 10 CSE banking-sector stocks.
      </footer>
    </div>
  )
}
