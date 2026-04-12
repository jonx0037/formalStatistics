import { useState, useMemo, useCallback, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfBeta, quantileBeta, expectationBeta } from './shared/distributions';
import { conjugatePriorPresets } from '../../data/continuous-distributions-data';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

export default function BetaPriorPosteriorExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [alpha0, setAlpha0] = useState(1);
  const [beta0, setBeta0] = useState(1);
  const [n, setN] = useState(20);
  const [k, setK] = useState(7);
  const [showLikelihood, setShowLikelihood] = useState(false);
  const [trueTheta, setTrueTheta] = useState(0.3);

  // Sequential mode state
  const [seqMode, setSeqMode] = useState(false);
  const [seqK, setSeqK] = useState(0);
  const [seqN, setSeqN] = useState(0);
  const seedRef = useRef(42);

  const chartW = Math.max(280, (width || 600) - 16);
  const chartH = 260;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Active posterior parameters
  const activeK = seqMode ? seqK : k;
  const activeN = seqMode ? seqN : n;
  const postAlpha = alpha0 + activeK;
  const postBeta = beta0 + activeN - activeK;

  // Prior curve
  const priorCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = 0.001 + (i / NUM_POINTS) * 0.998;
      data.push({ x, y: pdfBeta(x, alpha0, beta0) });
    }
    return data;
  }, [alpha0, beta0]);

  // Posterior curve
  const posteriorCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = 0.001 + (i / NUM_POINTS) * 0.998;
      data.push({ x, y: pdfBeta(x, postAlpha, postBeta) });
    }
    return data;
  }, [postAlpha, postBeta]);

  // Likelihood curve (scaled for display).
  // Subtracts the max log-likelihood (at the MLE θ̂ = k/n) before exponentiating
  // to prevent underflow when n is large.
  const likelihoodCurve = useMemo(() => {
    if (!showLikelihood || activeN === 0) return null;
    const logLs: number[] = [];
    const xs: number[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const theta = 0.001 + (i / NUM_POINTS) * 0.998;
      xs.push(theta);
      logLs.push(activeK * Math.log(theta) + (activeN - activeK) * Math.log(1 - theta));
    }
    const maxLogL = Math.max(...logLs);
    const data = xs.map((x, i) => ({ x, y: Math.exp(logLs[i] - maxLogL) }));
    // Scale to match posterior height
    const postMax = Math.max(...posteriorCurve.map((d) => d.y));
    data.forEach((d) => { d.y *= postMax; });
    return data;
  }, [showLikelihood, activeK, activeN, posteriorCurve]);

  const maxPDF = useMemo(() => {
    let m = Math.max(...priorCurve.map((d) => d.y), ...posteriorCurve.map((d) => d.y));
    if (likelihoodCurve) m = Math.max(m, ...likelihoodCurve.map((d) => d.y));
    return Math.max(m, 0.01);
  }, [priorCurve, posteriorCurve, likelihoodCurve]);

  // Credible intervals (memoized — quantileBeta is expensive)
  const priorCI = useMemo(() => {
    return [quantileBeta(0.025, alpha0, beta0), quantileBeta(0.975, alpha0, beta0)];
  }, [alpha0, beta0]);

  const postCI = useMemo(() => {
    return [quantileBeta(0.025, postAlpha, postBeta), quantileBeta(0.975, postAlpha, postBeta)];
  }, [postAlpha, postBeta]);

  const buildPath = useCallback((data: { x: number; y: number }[]) => {
    return data
      .map((d, i) => {
        const px = MARGIN.left + d.x * plotW;
        const py = MARGIN.top + plotH - (d.y / (maxPDF * 1.15)) * plotH;
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  }, [plotW, plotH, maxPDF]);

  // Sequential mode handlers
  const flipCoin = () => {
    seedRef.current = (seedRef.current * 1664525 + 1013904223) & 0xffffffff;
    const u = (seedRef.current >>> 0) / 0xffffffff;
    const success = u < trueTheta;
    setSeqN((prev) => prev + 1);
    if (success) setSeqK((prev) => prev + 1);
  };

  const resetSeq = () => {
    setSeqK(0);
    setSeqN(0);
    seedRef.current = 42;
  };

  const handlePreset = (preset: typeof conjugatePriorPresets[0]) => {
    setAlpha0(preset.alpha0);
    setBeta0(preset.beta0);
  };

  // Moments
  const priorMean = expectationBeta(alpha0, beta0);
  const postMean = expectationBeta(postAlpha, postBeta);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Beta-Bernoulli Conjugate Prior
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-1.5 justify-center mb-3">
        {conjugatePriorPresets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => handlePreset(preset)}
            className="px-2 py-0.5 text-xs rounded border transition-colors hover:bg-gray-100 dark:hover:bg-gray-800"
            style={{ borderColor: 'var(--color-border)' }}
            title={preset.description}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">α₀</span>
          <input type="range" min={0.1} max={20} step={0.1} value={alpha0} onChange={(e) => setAlpha0(Number(e.target.value))} className="w-20" />
          <span className="w-8 tabular-nums">{alpha0.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">β₀</span>
          <input type="range" min={0.1} max={20} step={0.1} value={beta0} onChange={(e) => setBeta0(Number(e.target.value))} className="w-20" />
          <span className="w-8 tabular-nums">{beta0.toFixed(1)}</span>
        </label>
      </div>

      {/* Mode toggle */}
      <div className="flex flex-wrap gap-4 justify-center mb-3 text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={seqMode} onChange={(e) => { setSeqMode(e.target.checked); resetSeq(); }} />
          <span>Sequential mode</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showLikelihood} onChange={(e) => setShowLikelihood(e.target.checked)} />
          <span>Show likelihood</span>
        </label>
      </div>

      {/* Data input */}
      {!seqMode ? (
        <div className="flex flex-wrap gap-4 justify-center mb-3">
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">n (trials)</span>
            <input type="range" min={0} max={100} step={1} value={n} onChange={(e) => { setN(Number(e.target.value)); setK(Math.min(k, Number(e.target.value))); }} className="w-24" />
            <span className="w-8 tabular-nums">{n}</span>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">k (successes)</span>
            <input type="range" min={0} max={n} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} className="w-24" />
            <span className="w-8 tabular-nums">{k}</span>
          </label>
        </div>
      ) : (
        <div className="flex flex-wrap gap-3 justify-center mb-3 items-center">
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">True θ</span>
            <input type="range" min={0.05} max={0.95} step={0.05} value={trueTheta} onChange={(e) => setTrueTheta(Number(e.target.value))} className="w-20" />
            <span className="w-8 tabular-nums">{trueTheta.toFixed(2)}</span>
          </label>
          <button onClick={flipCoin} className="px-3 py-1 text-xs rounded border hover:bg-gray-100 dark:hover:bg-gray-800" style={{ borderColor: 'var(--color-border)' }}>
            Flip coin
          </button>
          <button onClick={() => { for (let i = 0; i < 10; i++) flipCoin(); }} className="px-3 py-1 text-xs rounded border hover:bg-gray-100 dark:hover:bg-gray-800" style={{ borderColor: 'var(--color-border)' }}>
            +10 flips
          </button>
          <button onClick={resetSeq} className="px-3 py-1 text-xs rounded border hover:bg-gray-100 dark:hover:bg-gray-800" style={{ borderColor: 'var(--color-border)' }}>
            Reset
          </button>
          <span className="text-xs">n = {seqN}, k = {seqK}</span>
        </div>
      )}

      {/* Chart */}
      <svg width={chartW} height={chartH} className="block mx-auto">
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={v} x={MARGIN.left + v * plotW} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
            {v.toFixed(2)}
          </text>
        ))}

        {/* Likelihood (behind everything) */}
        {likelihoodCurve && (
          <path d={buildPath(likelihoodCurve)} fill="none" stroke="#d97706" strokeWidth={1.5} strokeDasharray="4,4" opacity={0.7} />
        )}

        {/* Prior (semi-transparent) */}
        <path d={buildPath(priorCurve)} fill="none" stroke="#2563eb" strokeWidth={1.5} opacity={0.5} />

        {/* Posterior */}
        <path d={buildPath(posteriorCurve)} fill="none" stroke="#DC2626" strokeWidth={2.5} />

        {/* True theta line (in sequential mode) */}
        {seqMode && (
          <line
            x1={MARGIN.left + trueTheta * plotW} y1={MARGIN.top}
            x2={MARGIN.left + trueTheta * plotW} y2={MARGIN.top + plotH}
            stroke="#16a34a" strokeWidth={1.5} strokeDasharray="4,3"
          />
        )}
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
        <span style={{ color: '#2563eb' }}>— Prior Beta({alpha0.toFixed(1)}, {beta0.toFixed(1)})</span>
        <span style={{ color: '#DC2626' }}>— Posterior Beta({postAlpha.toFixed(1)}, {postBeta.toFixed(1)})</span>
        {showLikelihood && <span style={{ color: '#d97706' }}>-- Likelihood (scaled)</span>}
        {seqMode && <span style={{ color: '#16a34a' }}>| True θ = {trueTheta.toFixed(2)}</span>}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 mt-3 text-xs max-w-md mx-auto">
        <div className="p-2 rounded" style={{ backgroundColor: 'rgba(37,99,235,0.05)' }}>
          <div className="font-medium" style={{ color: '#2563eb' }}>Prior</div>
          <div>E[θ] = {priorMean.toFixed(4)}</div>
          <div>95% CI: [{priorCI[0].toFixed(3)}, {priorCI[1].toFixed(3)}]</div>
          <div className="opacity-70">Eff. sample size: {(alpha0 + beta0).toFixed(1)}</div>
        </div>
        <div className="p-2 rounded" style={{ backgroundColor: 'rgba(220,38,38,0.05)' }}>
          <div className="font-medium" style={{ color: '#DC2626' }}>Posterior</div>
          <div>E[θ|data] = {postMean.toFixed(4)}</div>
          <div>95% CI: [{postCI[0].toFixed(3)}, {postCI[1].toFixed(3)}]</div>
          <div className="opacity-70">Eff. sample size: {(postAlpha + postBeta).toFixed(1)}</div>
        </div>
      </div>

      {activeN > 0 && (
        <div className="text-center text-xs mt-2 opacity-70 italic">
          As n → ∞, the posterior concentrates around the true θ regardless of the prior.
        </div>
      )}
    </div>
  );
}
