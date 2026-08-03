import { formatDate } from '../format.js'

export default function DateSlider({ days, index, onChange }) {
  const day = days[index]

  return (
    <div className="card date-slider-card">
      <div className="date-slider-header">
        <h2>Day {index + 1} of {days.length}</h2>
        <div className="date-slider-date">{formatDate(day.date)}</div>
      </div>
      <input
        type="range"
        min={0}
        max={days.length - 1}
        value={index}
        onChange={(e) => onChange(Number(e.target.value))}
        className="date-slider"
      />
      <div className="date-slider-controls">
        <button onClick={() => onChange(Math.max(0, index - 1))}>&larr; Prev day</button>
        <button onClick={() => onChange(0)}>First</button>
        <button onClick={() => onChange(days.length - 1)}>Last</button>
        <button onClick={() => onChange(Math.min(days.length - 1, index + 1))}>Next day &rarr;</button>
      </div>
      {day.decision_basis_date && (
        <p className="date-slider-note">
          Allocation decided from the 30-day window ending {formatDate(day.decision_basis_date)},
          applied for the {formatDate(day.date)} session.
        </p>
      )}
    </div>
  )
}
