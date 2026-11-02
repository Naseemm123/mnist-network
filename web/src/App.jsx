import { useEffect, useMemo, useState } from 'react';
import Canvas from './components/Canvas.jsx';
import Chart from './components/Chart.jsx';
import { forwardProp, loadWeights, normalizeDrawing } from './inference.js';

const blank = () => new Array(784).fill(0);

export default function App() {
  const [rawPixels, setRawPixels] = useState(blank); const [weights, setWeights] = useState(null);
  useEffect(() => { loadWeights().then(setWeights).catch(() => setWeights(null)); }, []);
  const normalized = useMemo(() => normalizeDrawing(rawPixels), [rawPixels]);
  const result = useMemo(() => weights ? forwardProp(weights, normalized) : { a1: new Array(128).fill(0), a2: new Array(10).fill(.1) }, [weights, normalized]);
  const prediction = result.a2.indexOf(Math.max(...result.a2));
  const hiddenStats = useMemo(() => {
    const active = result.a1.filter(value => value > 0);
    const ranked = result.a1.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, 6);
    return { active: active.length, mean: active.length ? active.reduce((sum, value) => sum + value, 0) / result.a1.length : 0, max: Math.max(...result.a1), ranked };
  }, [result.a1]);
  const clear = () => setRawPixels(blank());
  return <main className="app-shell">
    <header className="navbar"><div className="brand-mark"><span>MNIST</span><small>FROM SCRATCH</small></div><div className="nav-spec"><span>ARCHITECTURE</span><strong>784 → 128 → 10</strong></div><div className="nav-actions"><a className="nav-link" href="./about.html">TECHNICAL NOTES ↗</a><a className="nav-link" href="./weights.json" target="_blank" rel="noreferrer">VIEW WEIGHTS ↗</a></div></header>
    <section className="hero"><div><h1>Recognize handwritten<br />digits.</h1><p className="lede">Draw a digit and watch an already-trained 784 → 128 → 10 network process it from pixels to probabilities.</p></div><div className="hero-meta"><span>MODEL ACCURACY</span><strong>~97%</strong><small>MNIST / TEST SET</small></div></section>
    <section className="workspace">
      <div className="panel draw-panel"><div className="panel-heading"><span className="panel-index">01</span><div><h2>Input canvas</h2><p>Draw one handwritten digit</p></div><button onClick={clear}>CLEAR ↗</button></div><div className="canvas-frame"><Canvas pixels={rawPixels} onChange={setRawPixels} /><span className="corner corner-tl" /><span className="corner corner-br" /></div><div className="hint"><span>28 × 28 PIXELS</span><span>POINTER / STYLUS ENABLED</span></div></div>
      <div className="panel prediction-panel"><div className="panel-heading"><span className="panel-index">02</span><div><h2>Prediction</h2><p>Output layer / softmax</p></div></div><div className="prediction"><span className="prediction-label">I SEE A</span><strong>{prediction}</strong><span className="confidence">{Math.round(result.a2[prediction] * 100)}<small>% confidence</small></span></div><Chart probabilities={result.a2} /><div className="legend"><span><i className="dot cyan" /> ACTIVATION</span><span><i className="dot orange" /> TOP RESULT</span></div></div>
      <div className="panel hidden-panel"><div className="panel-heading"><span className="panel-index">03</span><div><h2>Hidden layer</h2><p>Each cell is one ReLU output; brighter means stronger response</p></div><span className="live-tag">A₁ / LIVE</span></div><div className="activation-stats"><div><span>ACTIVE</span><strong>{hiddenStats.active}<small> / 128</small></strong></div><div><span>MEAN</span><strong>{hiddenStats.mean.toFixed(2)}</strong></div><div><span>MAX</span><strong>{hiddenStats.max.toFixed(2)}</strong></div></div><div className="neuron-grid">{result.a1.map((value, i) => <span key={i} title={`Neuron ${i + 1}: ${value.toFixed(3)}`} style={{ opacity: value ? .14 + (value / Math.max(hiddenStats.max, .001)) * .86 : .08 }} />)}</div><div className="top-activations"><div className="top-heading"><span>TOP RESPONDING NEURONS</span><small>INDEX / A₁ VALUE</small></div>{hiddenStats.ranked.map(({ index, value }) => <div className="activation-row" key={index}><span>N{String(index + 1).padStart(3, '0')}</span><div><i style={{ width: `${hiddenStats.max ? (value / hiddenStats.max) * 100 : 0}%` }} /></div><strong>{value.toFixed(3)}</strong></div>)}</div><div className="layer-footer"><span>A₁ ACTIVATIONS</span><span>RELU OUTPUTS · INPUT UPDATES LIVE</span></div></div>
    </section>
  </main>;
}
