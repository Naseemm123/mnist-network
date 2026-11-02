export default function Chart({ probabilities }) {
  return <div className="chart" role="img" aria-label="Live softmax probabilities">{probabilities.map((value, digit) => <div className={`bar-wrap ${value === Math.max(...probabilities) ? 'winner' : ''}`} key={digit}><span className="bar-value">{Math.round(value * 100)}%</span><div className="bar" style={{ height: `${Math.max(3, value * 100)}%` }} /><span className="bar-label">{digit}</span></div>)}</div>;
}
