import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfGamma, pdfExponential, pdfChi2, expectationGamma, varianceGamma } from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

export default function GammaFamilyExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(3);
  const [beta, setBeta] = useState(1);
  const [showExp, setShowExp] = useState(false);
  const [showChi2, setShowChi2] = useState(false);
  const [sumN, setSumN] = useState(3);
  const [showSum, setShowSum] = useState(false);

  const chartW = Math.max(280, (width || 600) - 16);
  const chartH = 240;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const xMax = useMemo(() => {
    const mean = alpha / beta;
    const sd = Math.sqrt(alpha) / beta;
    return Math.max(mean + 4 * sd, 5);
  }, [alpha, beta]);

  // Main Gamma curve
  const gammaCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfGamma(x, alpha, beta) });
    }
    return data;
  }, [alpha, beta, xMax]);

  // Exponential overlay
  const expCurve = useMemo(() => {
    if (!showExp) return null;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfExponential(x, beta) });
    }
    return data;
  }, [showExp, beta, xMax]);

  // Chi-squared overlay
  const chi2Curve = useMemo(() => {
    if (!showChi2) return null;
    const k = Math.round(alpha * 2); // alpha = k/2
    if (k < 1) return null;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfChi2(x, k) });
    }
    return data;
  }, [showChi2, alpha, xMax]);

  // Sum-of-Exponentials histogram simulation
  const sumHistogram = useMemo(() => {
    if (!showSum) return null;
    // Generate sum-of-Exponentials samples using inverse CDF
    const nSamples = 2000;
    const samples: number[] = [];
    // Simple LCG PRNG for deterministic results.
    // Clamp away from 0 to prevent -log(0) = Infinity.
    let seed = 42;
    const nextRand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return Math.max((seed >>> 0) / 0x100000000, Number.EPSILON);
    };

    for (let s = 0; s < nSamples; s++) {
      let total = 0;
      for (let j = 0; j < sumN; j++) {
        total += -Math.log(nextRand()) / beta;
      }
      samples.push(total);
    }

    // Build histogram
    const nBins = 40;
    const binMax = xMax;
    const binW = binMax / nBins;
    const counts = new Array(nBins).fill(0);
    for (const val of samples) {
      const bin = Math.min(Math.floor(val / binW), nBins - 1);
      if (bin >= 0) counts[bin]++;
    }
    // Normalize to density
    const density = counts.map((c) => c / (nSamples * binW));
    return { density, binW, binMax };
  }, [showSum, sumN, beta, xMax]);

  // Gamma(sumN, beta) overlay for sum comparison
  const sumOverlayCurve = useMemo(() => {
    if (!showSum) return null;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfGamma(x, sumN, beta) });
    }
    return data;
  }, [showSum, sumN, beta, xMax]);

  const maxPDF = useMemo(() => {
    let m = Math.max(...gammaCurve.map((d) => d.y));
    if (expCurve) m = Math.max(m, ...expCurve.map((d) => d.y));
    if (chi2Curve) m = Math.max(m, ...chi2Curve.map((d) => d.y));
    if (sumOverlayCurve) m = Math.max(m, ...sumOverlayCurve.map((d) => d.y));
    if (sumHistogram) m = Math.max(m, ...sumHistogram.density);
    return Math.max(m, 0.01);
  }, [gammaCurve, expCurve, chi2Curve, sumOverlayCurve, sumHistogram]);

  const buildPath = useCallback((data: { x: number; y: number }[]) => {
    return data
      .map((d, i) => {
        const px = MARGIN.left + (d.x / xMax) * plotW;
        const py = MARGIN.top + plotH - (d.y / (maxPDF * 1.15)) * plotH;
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  }, [xMax, plotW, plotH, maxPDF]);

  const mean = expectationGamma(alpha, beta);
  const variance = varianceGamma(alpha, beta);
  const mode = alpha >= 1 ? (alpha - 1) / beta : 0;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Gamma Family Explorer
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">α (shape)</span>
          <input type="range" min={0.1} max={15} step={0.1} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-28" />
          <span className="w-8 tabular-nums">{alpha.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">β (rate)</span>
          <input type="range" min={0.1} max={5} step={0.1} value={beta} onChange={(e) => setBeta(Number(e.target.value))} className="w-28" />
          <span className="w-8 tabular-nums">{beta.toFixed(1)}</span>
        </label>
      </div>

      {/* Overlay toggles */}
      <div className="flex flex-wrap gap-4 justify-center mb-4 text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showExp} onChange={(e) => setShowExp(e.target.checked)} />
          <span style={{ color: '#16a34a' }}>Exp(β) overlay</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showChi2} onChange={(e) => setShowChi2(e.target.checked)} />
          <span style={{ color: '#9333ea' }}>χ²({Math.round(alpha * 2)}) overlay</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showSum} onChange={(e) => setShowSum(e.target.checked)} />
          <span style={{ color: '#DC2626' }}>Sum of Exp(β)</span>
        </label>
        {showSum && (
          <label className="flex items-center gap-2">
            <span className="font-medium">n =</span>
            <input type="range" min={1} max={15} step={1} value={sumN} onChange={(e) => setSumN(Number(e.target.value))} className="w-20" />
            <span className="w-6 tabular-nums">{sumN}</span>
          </label>
        )}
      </div>

      {/* Chart */}
      <svg width={chartW} height={chartH} className="block mx-auto">
        {/* Y grid */}
        {[0, maxPDF * 0.25, maxPDF * 0.5, maxPDF * 0.75].map((v, i) => (
          <g key={i}>
            <line
              x1={MARGIN.left} y1={MARGIN.top + plotH - (v / (maxPDF * 1.15)) * plotH}
              x2={chartW - MARGIN.right} y2={MARGIN.top + plotH - (v / (maxPDF * 1.15)) * plotH}
              stroke="currentColor" strokeOpacity={0.08}
            />
          </g>
        ))}

        {/* Sum histogram */}
        {sumHistogram && sumHistogram.density.map((d, i) => {
          const x0 = i * sumHistogram.binW;
          const px = MARGIN.left + (x0 / xMax) * plotW;
          const pw = (sumHistogram.binW / xMax) * plotW;
          const ph = (d / (maxPDF * 1.15)) * plotH;
          return (
            <rect
              key={i}
              x={px}
              y={MARGIN.top + plotH - ph}
              width={Math.max(1, pw - 1)}
              height={ph}
              fill="#DC2626"
              opacity={0.2}
            />
          );
        })}

        {/* Exponential overlay */}
        {expCurve && <path d={buildPath(expCurve)} fill="none" stroke="#16a34a" strokeWidth={1.5} strokeDasharray="6,3" />}

        {/* Chi-squared overlay */}
        {chi2Curve && <path d={buildPath(chi2Curve)} fill="none" stroke="#9333ea" strokeWidth={1.5} strokeDasharray="4,4" />}

        {/* Sum overlay */}
        {sumOverlayCurve && <path d={buildPath(sumOverlayCurve)} fill="none" stroke="#DC2626" strokeWidth={2} strokeDasharray="6,3" />}

        {/* Main Gamma curve */}
        <path d={buildPath(gammaCurve)} fill="none" stroke="#d97706" strokeWidth={2.5} />

        {/* E[X] marker */}
        {isFinite(mean) && mean > 0 && mean < xMax && (
          <>
            <line
              x1={MARGIN.left + (mean / xMax) * plotW} y1={MARGIN.top}
              x2={MARGIN.left + (mean / xMax) * plotW} y2={MARGIN.top + plotH}
              stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4,3"
            />
          </>
        )}
      </svg>

      {/* Readout */}
      <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
        <span><strong>E[X]</strong> = α/β = {mean.toFixed(3)}</span>
        <span><strong>Var(X)</strong> = α/β² = {variance.toFixed(3)}</span>
        <span><strong>Mode</strong> = {alpha >= 1 ? `(α−1)/β = ${mode.toFixed(3)}` : '0 (α < 1)'}</span>
      </div>
    </div>
  );
}
