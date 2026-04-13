import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  sampleGaussianCopula, cdfStudentT,
  quantileStdNormal, quantileExponential, quantileBeta, quantileUniform,
  pdfExponential, pdfBeta, pdfUniform, pdfStdNormal,
  cholesky2x2,
} from './shared/distributions';
import { copulaMarginals } from '../../data/multivariate-distributions-data';
import type { MarginalKey } from '../../data/multivariate-distributions-data';

// ── Layout constants ──────────────────────────────────────────────────────

const SCATTER_MARGIN = { top: 8, right: 8, bottom: 32, left: 40 };
const PDF_HEIGHT = 60;
const PDF_GAP = 4;
const NUM_PDF_POINTS = 150;
const NUM_SAMPLES = 500;

const COPULA_TYPES = ['independence', 'gaussian', 'student-t'] as const;
type CopulaType = typeof COPULA_TYPES[number];

const COPULA_LABELS: Record<CopulaType, string> = {
  independence: 'Independence',
  gaussian: 'Gaussian',
  'student-t': 'Student-t',
};

// ── Axis ranges per marginal ──────────────────────────────────────────────

const MARGINAL_RANGE: Record<MarginalKey, [number, number]> = {
  exponential: [0, 5],
  normal: [-3.5, 3.5],
  beta: [0, 1],
  uniform: [0, 1],
};

// ── Marginal helpers ──────────────────────────────────────────────────────

function evalMarginalPDF(key: MarginalKey, x: number): number {
  switch (key) {
    case 'exponential': return pdfExponential(x, 1);
    case 'normal': return pdfStdNormal(x);
    case 'beta': return pdfBeta(x, 2, 5);
    case 'uniform': return pdfUniform(x, 0, 1);
  }
}

function applyQuantile(key: MarginalKey, u: number): number {
  switch (key) {
    case 'exponential': return quantileExponential(u, 1);
    case 'normal': return quantileStdNormal(u);
    case 'beta': return quantileBeta(u, 2, 5);
    case 'uniform': return quantileUniform(u, 0, 1);
  }
}

// ── Student-t copula sampler ──────────────────────────────────────────────

function sampleStudentTCopula(n: number, rho: number, nu: number, rng = Math.random): number[][] {
  const R = [[1, rho], [rho, 1]];
  const L = cholesky2x2(R);
  const samples: number[][] = [];

  for (let i = 0; i < n; i++) {
    // Standard normals via Box-Muller
    const u1 = rng(), u2 = rng();
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

    // Correlate via Cholesky
    const y1 = L[0][0] * z1;
    const y2 = L[1][0] * z1 + L[1][1] * z2;

    // Chi-squared(nu) / nu
    let chiSq = 0;
    for (let j = 0; j < nu; j++) {
      const a = rng(), b = rng();
      const w = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
      chiSq += w * w;
    }
    const v = chiSq / nu;
    const sqrtV = Math.sqrt(v);

    // Student-t marginals
    const t1 = y1 / sqrtV;
    const t2 = y2 / sqrtV;

    // Apply Student-t CDF to get uniforms
    const u_1 = cdfStudentT(t1, nu);
    const u_2 = cdfStudentT(t2, nu);

    samples.push([u_1, u_2]);
  }
  return samples;
}

// ── Correlation statistics ────────────────────────────────────────────────

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i]; sumY += y[i]; sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i]; sumY2 += y[i] * y[i];
  }
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return den > 0 ? num / den : 0;
}

function ranks(arr: number[]): number[] {
  const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const result = new Array(arr.length);
  for (let i = 0; i < sorted.length; i++) result[sorted[i].i] = i + 1;
  return result;
}

function spearmanCorrelation(x: number[], y: number[]): number {
  return pearsonCorrelation(ranks(x), ranks(y));
}

// ── Component ─────────────────────────────────────────────────────────────

export default function CopulaExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls
  const [marginalX, setMarginalX] = useState<MarginalKey>('exponential');
  const [marginalY, setMarginalY] = useState<MarginalKey>('beta');
  const [copula, setCopula] = useState<CopulaType>('gaussian');
  const [rho, setRho] = useState(0.5);
  const [nu, setNu] = useState(5);
  const [seed, setSeed] = useState(0);

  // Responsive layout
  const containerW = Math.max(300, (width || 620) - 16);
  const scatterSize = Math.min(360, containerW - PDF_HEIGHT - PDF_GAP - 60);
  const plotW = scatterSize - SCATTER_MARGIN.left - SCATTER_MARGIN.right;
  const plotH = scatterSize - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom;

  // ── Seeded RNG (simple mulberry32) ────────────────────────────────────
  const makeRng = useCallback((s: number) => {
    let state = s | 0;
    if (state === 0) state = 1;
    return () => {
      state |= 0;
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }, []);

  // ── Sample generation ─────────────────────────────────────────────────
  const samples = useMemo(() => {
    const rng = makeRng(seed * 7919 + 42);

    let uniforms: number[][];
    switch (copula) {
      case 'independence': {
        uniforms = [];
        for (let i = 0; i < NUM_SAMPLES; i++) {
          uniforms.push([rng(), rng()]);
        }
        break;
      }
      case 'gaussian': {
        uniforms = sampleGaussianCopula(NUM_SAMPLES, [[1, rho], [rho, 1]], rng);
        break;
      }
      case 'student-t': {
        uniforms = sampleStudentTCopula(NUM_SAMPLES, rho, nu, rng);
        break;
      }
    }

    const xs = uniforms.map(u => applyQuantile(marginalX, u[0]));
    const ys = uniforms.map(u => applyQuantile(marginalY, u[1]));
    return { xs, ys };
  }, [copula, rho, nu, marginalX, marginalY, seed, makeRng]);

  // ── Correlation readouts ──────────────────────────────────────────────
  const pearsonR = useMemo(() => pearsonCorrelation(samples.xs, samples.ys), [samples]);
  const spearmanR = useMemo(() => spearmanCorrelation(samples.xs, samples.ys), [samples]);

  // ── Axis domains ──────────────────────────────────────────────────────
  const [xMin, xMax] = MARGINAL_RANGE[marginalX];
  const [yMin, yMax] = MARGINAL_RANGE[marginalY];
  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;

  // ── Scale helpers ─────────────────────────────────────────────────────
  const toScatterX = useCallback(
    (x: number) => SCATTER_MARGIN.left + ((x - xMin) / xSpan) * plotW,
    [xMin, xSpan, plotW],
  );
  const toScatterY = useCallback(
    (y: number) => SCATTER_MARGIN.top + plotH - ((y - yMin) / ySpan) * plotH,
    [yMin, ySpan, plotH],
  );

  // ── X-axis ticks ──────────────────────────────────────────────────────
  const xTicks = useMemo(() => {
    const nice = [0.1, 0.2, 0.25, 0.5, 1, 2, 5];
    let step = 1;
    for (const n of nice) {
      if (xSpan / n <= 7) { step = n; break; }
    }
    const ticks: number[] = [];
    const start = Math.ceil(xMin / step) * step;
    for (let v = start; v <= xMax + 1e-9; v += step) {
      ticks.push(Math.round(v * 1000) / 1000);
    }
    return ticks;
  }, [xMin, xMax, xSpan]);

  const yTicks = useMemo(() => {
    const nice = [0.1, 0.2, 0.25, 0.5, 1, 2, 5];
    let step = 1;
    for (const n of nice) {
      if (ySpan / n <= 7) { step = n; break; }
    }
    const ticks: number[] = [];
    const start = Math.ceil(yMin / step) * step;
    for (let v = start; v <= yMax + 1e-9; v += step) {
      ticks.push(Math.round(v * 1000) / 1000);
    }
    return ticks;
  }, [yMin, yMax, ySpan]);

  // ── Marginal PDF curves ───────────────────────────────────────────────
  // These depend ONLY on the marginal selection, not the copula.
  const xPdfCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_PDF_POINTS; i++) {
      const x = xMin + (i / NUM_PDF_POINTS) * xSpan;
      data.push({ x, y: evalMarginalPDF(marginalX, x) });
    }
    return data;
  }, [marginalX, xMin, xSpan]);

  const yPdfCurve = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_PDF_POINTS; i++) {
      const x = yMin + (i / NUM_PDF_POINTS) * ySpan;
      data.push({ x, y: evalMarginalPDF(marginalY, x) });
    }
    return data;
  }, [marginalY, yMin, ySpan]);

  const xPdfMax = useMemo(() => Math.max(...xPdfCurve.map(d => d.y), 0.01), [xPdfCurve]);
  const yPdfMax = useMemo(() => Math.max(...yPdfCurve.map(d => d.y), 0.01), [yPdfCurve]);

  // ── SVG path builders for marginal PDFs ───────────────────────────────
  const xPdfPath = useMemo(() => {
    return xPdfCurve.map((d, i) => {
      const px = SCATTER_MARGIN.left + ((d.x - xMin) / xSpan) * plotW;
      const py = PDF_HEIGHT - (d.y / (xPdfMax * 1.1)) * (PDF_HEIGHT - 6);
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }, [xPdfCurve, xMin, xSpan, plotW, xPdfMax]);

  // Y marginal PDF: rotated 90 degrees — "x" of the PDF maps to scatter Y,
  // "height" of the PDF maps horizontally into the right panel.
  const yPdfPath = useMemo(() => {
    return yPdfCurve.map((d, i) => {
      const py = SCATTER_MARGIN.top + plotH - ((d.x - yMin) / ySpan) * plotH;
      const px = (d.y / (yPdfMax * 1.1)) * (PDF_HEIGHT - 6);
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }, [yPdfCurve, yMin, ySpan, plotH, yPdfMax]);

  // ── Marginal display name ─────────────────────────────────────────────
  const marginalName = (key: MarginalKey) =>
    copulaMarginals.find(m => m.key === key)?.name ?? key;

  // ── Regenerate handler ────────────────────────────────────────────────
  const handleRegenerate = useCallback(() => setSeed(s => s + 1), []);

  // ── Total SVG dimensions ──────────────────────────────────────────────
  const totalSvgW = scatterSize + PDF_GAP + PDF_HEIGHT;
  const totalSvgH = PDF_HEIGHT + PDF_GAP + scatterSize;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Copula Explorer &mdash; Sklar&rsquo;s Theorem
      </div>

      {/* Controls row 1: marginal dropdowns */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">X marginal</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={marginalX}
            onChange={e => setMarginalX(e.target.value as MarginalKey)}
          >
            {copulaMarginals.map(m => (
              <option key={m.key} value={m.key}>{m.name}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Y marginal</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={marginalY}
            onChange={e => setMarginalY(e.target.value as MarginalKey)}
          >
            {copulaMarginals.map(m => (
              <option key={m.key} value={m.key}>{m.name}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Controls row 2: copula tabs + regenerate */}
      <div className="flex flex-wrap gap-2 justify-center mb-3 items-center">
        {COPULA_TYPES.map(type => (
          <button
            key={type}
            onClick={() => setCopula(type)}
            className="px-3 py-1 rounded text-xs font-medium transition-colors"
            style={{
              background: copula === type ? '#2563eb' : 'var(--color-bg)',
              color: copula === type ? '#fff' : 'inherit',
              border: copula === type ? '1px solid #2563eb' : '1px solid var(--color-border)',
            }}
          >
            {COPULA_LABELS[type]}
          </button>
        ))}
        <button
          onClick={handleRegenerate}
          className="px-3 py-1 rounded text-xs font-medium border transition-colors ml-2"
          style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
        >
          Regenerate
        </button>
      </div>

      {/* Controls row 3: sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {copula !== 'independence' && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">{'\u03C1'}</span>
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={rho}
              onChange={e => setRho(Number(e.target.value))}
              className="w-28"
            />
            <span className="w-10 tabular-nums text-right">{rho.toFixed(2)}</span>
          </label>
        )}
        {copula === 'student-t' && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">{'\u03BD'}</span>
            <input
              type="range"
              min={2}
              max={30}
              step={1}
              value={nu}
              onChange={e => setNu(Number(e.target.value))}
              className="w-28"
            />
            <span className="w-10 tabular-nums text-right">{nu}</span>
          </label>
        )}
      </div>

      {/* Main visualization */}
      <div className="flex justify-center">
        <svg
          width={totalSvgW}
          height={totalSvgH}
          className="block"
          style={{ overflow: 'visible' }}
        >
          {/* ── Top margin: X marginal PDF ────────────────────────── */}
          <g transform={`translate(0, 0)`}>
            <path
              d={xPdfPath}
              fill="none"
              stroke="#374151"
              strokeWidth={1.8}
            />
            <text
              x={SCATTER_MARGIN.left + plotW / 2}
              y={12}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              {marginalName(marginalX)} PDF
            </text>
          </g>

          {/* ── Center: Scatter plot ──────────────────────────────── */}
          <g transform={`translate(0, ${PDF_HEIGHT + PDF_GAP})`}>
            {/* Background */}
            <rect
              x={SCATTER_MARGIN.left}
              y={SCATTER_MARGIN.top}
              width={plotW}
              height={plotH}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.15}
            />

            {/* Grid lines — X */}
            {xTicks.map((v, i) => {
              const px = toScatterX(v);
              return (
                <line
                  key={`xg-${i}`}
                  x1={px} y1={SCATTER_MARGIN.top}
                  x2={px} y2={SCATTER_MARGIN.top + plotH}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
              );
            })}

            {/* Grid lines — Y */}
            {yTicks.map((v, i) => {
              const py = toScatterY(v);
              return (
                <line
                  key={`yg-${i}`}
                  x1={SCATTER_MARGIN.left} y1={py}
                  x2={SCATTER_MARGIN.left + plotW} y2={py}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
              );
            })}

            {/* Scatter dots */}
            {samples.xs.map((x, i) => {
              const px = toScatterX(x);
              const py = toScatterY(samples.ys[i]);
              // Clip to plot area
              if (px < SCATTER_MARGIN.left || px > SCATTER_MARGIN.left + plotW) return null;
              if (py < SCATTER_MARGIN.top || py > SCATTER_MARGIN.top + plotH) return null;
              return (
                <circle
                  key={i}
                  cx={px}
                  cy={py}
                  r={2.5}
                  fill="rgba(59, 130, 246, 0.3)"
                  stroke="rgba(59, 130, 246, 0.5)"
                  strokeWidth={0.5}
                />
              );
            })}

            {/* X-axis line */}
            <line
              x1={SCATTER_MARGIN.left}
              y1={SCATTER_MARGIN.top + plotH}
              x2={SCATTER_MARGIN.left + plotW}
              y2={SCATTER_MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* Y-axis line */}
            <line
              x1={SCATTER_MARGIN.left}
              y1={SCATTER_MARGIN.top}
              x2={SCATTER_MARGIN.left}
              y2={SCATTER_MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks + labels */}
            {xTicks.map((v, i) => {
              const px = toScatterX(v);
              return (
                <g key={`xt-${i}`}>
                  <line
                    x1={px} y1={SCATTER_MARGIN.top + plotH}
                    x2={px} y2={SCATTER_MARGIN.top + plotH + 4}
                    stroke="currentColor" strokeOpacity={0.4}
                  />
                  <text
                    x={px}
                    y={SCATTER_MARGIN.top + plotH + 15}
                    textAnchor="middle"
                    fontSize={9}
                    fill="currentColor"
                    opacity={0.6}
                  >
                    {Number.isInteger(v) ? v : v.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Y-axis ticks + labels */}
            {yTicks.map((v, i) => {
              const py = toScatterY(v);
              return (
                <g key={`yt-${i}`}>
                  <line
                    x1={SCATTER_MARGIN.left - 4} y1={py}
                    x2={SCATTER_MARGIN.left} y2={py}
                    stroke="currentColor" strokeOpacity={0.4}
                  />
                  <text
                    x={SCATTER_MARGIN.left - 7}
                    y={py + 3}
                    textAnchor="end"
                    fontSize={9}
                    fill="currentColor"
                    opacity={0.6}
                  >
                    {Number.isInteger(v) ? v : v.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* X-axis label */}
            <text
              x={SCATTER_MARGIN.left + plotW / 2}
              y={SCATTER_MARGIN.top + plotH + 28}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              {marginalName(marginalX)}
            </text>

            {/* Y-axis label */}
            <text
              x={10}
              y={SCATTER_MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 10, ${SCATTER_MARGIN.top + plotH / 2})`}
            >
              {marginalName(marginalY)}
            </text>
          </g>

          {/* ── Right margin: Y marginal PDF (rotated 90deg) ─────── */}
          <g transform={`translate(${scatterSize + PDF_GAP}, ${PDF_HEIGHT + PDF_GAP})`}>
            <path
              d={yPdfPath}
              fill="none"
              stroke="#374151"
              strokeWidth={1.8}
            />
            <text
              x={PDF_HEIGHT / 2}
              y={SCATTER_MARGIN.top + plotH + 15}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              {marginalName(marginalY)} PDF
            </text>
          </g>
        </svg>
      </div>

      {/* Bottom info bar */}
      <div className="mt-3 pt-3 border-t text-xs" style={{ borderColor: 'var(--color-border)' }}>
        <div className="flex flex-wrap gap-x-6 gap-y-1 justify-center">
          <span>
            <strong>Pearson {'\u03C1'}</strong> = {pearsonR.toFixed(3)}
          </span>
          <span>
            <strong>Spearman {'\u03C1'}<sub>s</sub></strong> = {spearmanR.toFixed(3)}
          </span>
          <span className="opacity-60">
            ({NUM_SAMPLES} samples &middot; {COPULA_LABELS[copula]} copula)
          </span>
        </div>
      </div>
    </div>
  );
}
