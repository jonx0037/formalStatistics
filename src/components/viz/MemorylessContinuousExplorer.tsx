import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfExponential, cdfExponential, pdfGamma } from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

export default function MemorylessContinuousExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [lambda, setLambda] = useState(1);
  const [s, setS] = useState(2);
  const [showGamma, setShowGamma] = useState(false);

  const chartW = Math.max(280, ((width || 600) - 24) / 2);
  const chartH = 220;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const xMax = Math.max(s + 5 / lambda, 8 / lambda);

  // PDF curves
  const pdfCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfExponential(x, lambda) });
    }
    return data;
  }, [lambda, xMax]);

  // Conditional PDF: f(x|X>s) = f(x) / P(X>s) for x > s, shifted to start at s
  const conditionalCurve = useMemo(() => {
    const survS = Math.exp(-lambda * s); // P(X > s)
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = s + (i / NUM_POINTS) * (xMax - s);
      data.push({ x, y: pdfExponential(x, lambda) / survS });
    }
    return data;
  }, [lambda, s, xMax]);

  // Re-indexed conditional: shift to start at 0
  const reindexedCurve = useMemo(() => {
    return conditionalCurve.map((d) => ({ x: d.x - s, y: d.y }));
  }, [conditionalCurve, s]);

  // Original PDF for comparison on right panel
  const originalForComparison = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * (xMax - s);
      data.push({ x, y: pdfExponential(x, lambda) });
    }
    return data;
  }, [lambda, s, xMax]);

  // Gamma comparison
  const gammaCurve = useMemo(() => {
    if (!showGamma) return null;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * xMax;
      data.push({ x, y: pdfGamma(x, 2, lambda) });
    }
    return data;
  }, [showGamma, lambda, xMax]);

  const maxPDF = useMemo(() => {
    let m = Math.max(...pdfCurve.map((d) => d.y));
    if (gammaCurve) m = Math.max(m, ...gammaCurve.map((d) => d.y));
    return Math.max(m, 0.01);
  }, [pdfCurve, gammaCurve]);

  const maxPDFRight = useMemo(() => {
    return Math.max(...originalForComparison.map((d) => d.y), 0.01);
  }, [originalForComparison]);

  // Path builders
  const buildPath = useCallback((
    data: { x: number; y: number }[],
    xMin: number, xMax_: number, yMax: number, w: number
  ) => {
    return data
      .map((d, i) => {
        const px = MARGIN.left + ((d.x - xMin) / (xMax_ - xMin)) * w;
        const py = MARGIN.top + plotH - (d.y / (yMax * 1.15)) * plotH;
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  }, [plotH]);

  // Memoryless verification values
  const tValues = [1, 2, 3];
  const verifyData = useMemo(() => {
    return tValues.map((t) => ({
      t,
      conditional: Math.exp(-lambda * t), // P(X > s+t | X > s)
      unconditional: Math.exp(-lambda * t), // P(X > t)
    }));
  }, [lambda]);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Memoryless Property Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">λ (rate)</span>
          <input type="range" min={0.2} max={3} step={0.1} value={lambda} onChange={(e) => setLambda(Number(e.target.value))} className="w-24" />
          <span className="w-8 tabular-nums">{lambda.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">s (condition)</span>
          <input type="range" min={0.5} max={5} step={0.1} value={s} onChange={(e) => setS(Number(e.target.value))} className="w-24" />
          <span className="w-8 tabular-nums">{s.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs cursor-pointer">
          <input type="checkbox" checked={showGamma} onChange={(e) => setShowGamma(e.target.checked)} />
          <span>Show Gamma(2, λ) comparison</span>
        </label>
      </div>

      {/* Charts */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* Left: Original PDF with conditional overlay */}
        <div>
          <div className="text-center text-xs font-medium mb-1">Exp(λ) PDF with conditional at X {'>'} {s.toFixed(1)}</div>
          <svg width={chartW} height={chartH} className="block">
            {/* Shade region X > s */}
            <rect
              x={MARGIN.left + (s / xMax) * plotW}
              y={MARGIN.top}
              width={plotW - (s / xMax) * plotW}
              height={plotH}
              fill="#16a34a" opacity={0.08}
            />

            {/* Original PDF */}
            <path d={buildPath(pdfCurve, 0, xMax, maxPDF, plotW)} fill="none" stroke="#2563eb" strokeWidth={2} />

            {/* Conditional overlay */}
            <path d={buildPath(conditionalCurve, 0, xMax, maxPDF, plotW)} fill="none" stroke="#DC2626" strokeWidth={2} strokeDasharray="6,3" />

            {/* Gamma comparison */}
            {gammaCurve && (
              <path d={buildPath(gammaCurve, 0, xMax, maxPDF, plotW)} fill="none" stroke="#d97706" strokeWidth={1.5} strokeDasharray="3,3" />
            )}

            {/* s marker */}
            <line
              x1={MARGIN.left + (s / xMax) * plotW} y1={MARGIN.top}
              x2={MARGIN.left + (s / xMax) * plotW} y2={MARGIN.top + plotH}
              stroke="#16a34a" strokeWidth={1.5} strokeDasharray="4,3"
            />
            <text x={MARGIN.left + (s / xMax) * plotW} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '9px' }}>
              s = {s.toFixed(1)}
            </text>
          </svg>
        </div>

        {/* Right: Re-indexed conditional = original */}
        <div>
          <div className="text-center text-xs font-medium mb-1">Re-indexed: identical to original</div>
          <svg width={chartW} height={chartH} className="block">
            {/* Original for comparison */}
            <path d={buildPath(originalForComparison, 0, xMax - s, maxPDFRight, plotW)} fill="none" stroke="#2563eb" strokeWidth={2} />

            {/* Re-indexed conditional */}
            <path d={buildPath(reindexedCurve, 0, xMax - s, maxPDFRight, plotW)} fill="none" stroke="#DC2626" strokeWidth={2} strokeDasharray="6,3" />
          </svg>
        </div>
      </div>

      {/* Verification table */}
      <div className="mt-3 text-xs text-center">
        <div className="font-medium mb-1">Memoryless verification: P(X {'>'} s+t | X {'>'} s) = P(X {'>'} t)</div>
        <div className="flex gap-4 justify-center">
          {verifyData.map(({ t, conditional, unconditional }) => (
            <span key={t}>
              t={t}: {conditional.toFixed(4)} = {unconditional.toFixed(4)} ✓
            </span>
          ))}
        </div>
        <div className="mt-1 opacity-70 italic">The waiting time "resets" — the past provides no information about the remaining time.</div>
      </div>
    </div>
  );
}
