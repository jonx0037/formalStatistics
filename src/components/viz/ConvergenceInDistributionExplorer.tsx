import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalCDFPoints,
  kolmogorovSmirnovStat,
  sampleSequence,
  exponentialSample,
  bernoulliSample,
  uniformSample,
} from './shared/convergence';
import {
  cdfStdNormal,
  pmfPoisson,
  cdfPoisson,
  pmfBinomial,
  cdfBinomial,
  cdfStudentT,
  pdfNormal,
  pdfStudentT,
} from './shared/distributions';
import { cdfConvergenceExamples } from '../../data/modes-of-convergence-data';

// ── Constants ──────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 50 };
const CDF_COLOR = '#2563eb';
const LIMIT_COLOR = '#111827';
const FAMILY_ALPHAS = [0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.30, 0.50];
const FAMILY_N_VALUES = [1, 2, 5, 10, 20, 50, 100, 200];
const CLT_UNDERLYING = ['Exponential(1)', 'Uniform(0,1)', 'Bernoulli(0.3)'] as const;

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Compute Binomial CDF as a step function for plotting. */
function binomialCDFSteps(n: number, p: number, xMax: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  pts.push({ x: -0.5, y: 0 });
  const upper = Math.min(n, Math.ceil(xMax));
  for (let k = 0; k <= upper; k++) {
    const cdfVal = cdfBinomial(k, n, p);
    pts.push({ x: k, y: cdfVal });
  }
  return pts;
}

/** Compute Poisson CDF as a step function for plotting. */
function poissonCDFSteps(lambda: number, xMax: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  pts.push({ x: -0.5, y: 0 });
  const upper = Math.ceil(xMax);
  for (let k = 0; k <= upper; k++) {
    pts.push({ x: k, y: cdfPoisson(k, lambda) });
  }
  return pts;
}

/** Compute a continuous CDF curve over a range. */
function continuousCDFCurve(
  cdf: (x: number) => number,
  xMin: number,
  xMax: number,
  numPts: number = 300,
): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i <= numPts; i++) {
    const x = xMin + (i / numPts) * (xMax - xMin);
    pts.push({ x, y: cdf(x) });
  }
  return pts;
}

/** Build an SVG path string from step-function data (discrete CDF). */
function buildStepPath(
  data: { x: number; y: number }[],
  toSvgX: (x: number) => number,
  toSvgY: (y: number) => number,
): string {
  if (data.length === 0) return '';
  const parts: string[] = [];
  for (let i = 0; i < data.length; i++) {
    const px = toSvgX(data[i].x);
    const py = toSvgY(data[i].y);
    if (i === 0) {
      parts.push(`M${px.toFixed(1)},${py.toFixed(1)}`);
    } else {
      // Horizontal then vertical (step-after)
      const prevY = toSvgY(data[i - 1].y);
      parts.push(`L${px.toFixed(1)},${prevY.toFixed(1)}`);
      parts.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
    }
  }
  // Extend to right edge
  const last = data[data.length - 1];
  const finalX = toSvgX(last.x + 1);
  parts.push(`L${finalX.toFixed(1)},${toSvgY(last.y).toFixed(1)}`);
  return parts.join(' ');
}

/** Build an SVG path string from smooth curve data. */
function buildSmoothPath(
  data: { x: number; y: number }[],
  toSvgX: (x: number) => number,
  toSvgY: (y: number) => number,
): string {
  return data
    .map((d, i) => {
      const px = toSvgX(d.x);
      const py = toSvgY(d.y);
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(' ');
}

/** Compute KS distance between two discrete step CDFs evaluated at integer points. */
function discreteKS(
  cdf1: (k: number) => number,
  cdf2: (k: number) => number,
  maxK: number,
): number {
  let maxDiff = 0;
  for (let k = 0; k <= maxK; k++) {
    maxDiff = Math.max(maxDiff, Math.abs(cdf1(k) - cdf2(k)));
  }
  return maxDiff;
}

/** Compute nice tick values for an axis range. */
function computeTicks(min: number, max: number, maxCount: number = 6): number[] {
  const span = max - min;
  if (span <= 0) return [min];
  const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
  let step = 1;
  for (const n of nice) {
    if (span / n <= maxCount) { step = n; break; }
  }
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max; v += step) {
    ticks.push(Math.round(v * 1000) / 1000);
  }
  return ticks;
}

// ── CLT simulation cache ──────────────────────────────────────────────────────

interface CLTSimResult {
  ecdfPts: { x: number; Fn: number }[];
  ksStat: number;
  histBins: { lo: number; hi: number; count: number; density: number }[];
}

function runCLTSimulation(
  n: number,
  underlying: string,
  _seed?: number,
): CLTSimResult {
  // Sampler, mean, variance for the underlying distribution
  let sampler: () => number;
  let mu: number;
  let sigma: number;
  if (underlying === 'Exponential(1)') {
    sampler = () => exponentialSample(1);
    mu = 1;
    sigma = 1;
  } else if (underlying === 'Uniform(0,1)') {
    sampler = () => uniformSample(0, 1);
    mu = 0.5;
    sigma = Math.sqrt(1 / 12);
  } else {
    sampler = () => bernoulliSample(0.3);
    mu = 0.3;
    sigma = Math.sqrt(0.3 * 0.7);
  }

  const M = Math.min(1000, Math.ceil(200000 / n));
  const zValues: number[] = [];
  for (let sim = 0; sim < M; sim++) {
    const samples = sampleSequence(sampler, n);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += samples[i];
    const xbar = sum / n;
    const z = (xbar - mu) / (sigma / Math.sqrt(n));
    zValues.push(z);
  }

  // Empirical CDF
  const ecdfPts = empiricalCDFPoints(zValues);
  const ksStat = kolmogorovSmirnovStat(zValues, cdfStdNormal);

  // Histogram bins
  const sorted = [...zValues].sort((a, b) => a - b);
  const lo = Math.min(sorted[0], -4);
  const hi = Math.max(sorted[sorted.length - 1], 4);
  const numBins = 30;
  const binWidth = (hi - lo) / numBins;
  const histBins: { lo: number; hi: number; count: number; density: number }[] = [];
  for (let i = 0; i < numBins; i++) {
    histBins.push({ lo: lo + i * binWidth, hi: lo + (i + 1) * binWidth, count: 0, density: 0 });
  }
  for (const z of zValues) {
    const idx = Math.min(Math.floor((z - lo) / binWidth), numBins - 1);
    if (idx >= 0) histBins[idx].count++;
  }
  for (const bin of histBins) {
    bin.density = bin.count / (M * binWidth);
  }

  return { ecdfPts, ksStat, histBins };
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ConvergenceInDistributionExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls
  const [exampleIdx, setExampleIdx] = useState(0);
  const [n, setN] = useState(10);
  const [lambda, setLambda] = useState(5);
  const [showFamily, setShowFamily] = useState(false);
  const [cltDist, setCltDist] = useState<string>(CLT_UNDERLYING[0]);
  const [cltSeed, setCltSeed] = useState(0);

  const example = cdfConvergenceExamples[exampleIdx];
  const exampleId = example.id;

  // Responsive layout
  const containerW = Math.max(300, (width || 700) - 16);
  const isNarrow = containerW < 640;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = 240;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── X-axis range ──────────────────────────────────────────────────────────

  const xRange = useMemo<[number, number]>(() => {
    if (exampleId === 'poisson-limit') {
      return [-0.5, Math.max(lambda + 4 * Math.sqrt(lambda), 15) + 0.5];
    }
    if (exampleId === 'student-normal') {
      return [-5, 5];
    }
    return [-4.5, 4.5]; // CLT
  }, [exampleId, lambda]);

  const [xMin, xMax] = xRange;

  // ── Scale helpers ─────────────────────────────────────────────────────────

  const toSvgX = useCallback(
    (x: number) => MARGIN.left + ((x - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const toSvgXRight = useCallback(
    (x: number) => MARGIN.left + ((x - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const cdfToSvgY = useCallback(
    (y: number) => MARGIN.top + plotH - y * plotH,
    [plotH],
  );

  // ── Poisson limit data ────────────────────────────────────────────────────

  const poissonData = useMemo(() => {
    if (exampleId !== 'poisson-limit') return null;
    const upper = Math.ceil(xMax);

    // Limit CDF: Poisson(lambda)
    const limitCDF = poissonCDFSteps(lambda, upper);

    // Xn CDF: Binomial(n, lambda/n)
    const p = Math.min(lambda / n, 1);
    const xnCDF = binomialCDFSteps(n, p, upper);

    // Family CDFs for "show all"
    const familyCDFs = FAMILY_N_VALUES.filter(k => k <= 200).map(k => {
      const pk = Math.min(lambda / k, 1);
      return { n: k, cdf: binomialCDFSteps(k, pk, upper) };
    });

    // KS distance
    const ks = discreteKS(
      (k: number) => cdfBinomial(k, n, p),
      (k: number) => cdfPoisson(k, lambda),
      upper,
    );

    // PMF data for right panel
    const pmfLimit = Array.from({ length: upper + 1 }, (_, k) => ({
      k,
      pn: pmfBinomial(k, n, p),
      plimit: pmfPoisson(k, lambda),
    }));

    return { limitCDF, xnCDF, familyCDFs, ks, pmfLimit };
  }, [exampleId, n, lambda, xMax]);

  // ── Student-t data ────────────────────────────────────────────────────────

  const studentData = useMemo(() => {
    if (exampleId !== 'student-normal') return null;

    const limitCDF = continuousCDFCurve(cdfStdNormal, xMin, xMax, 300);
    const xnCDF = continuousCDFCurve((x) => cdfStudentT(x, n), xMin, xMax, 300);

    const familyCDFs = FAMILY_N_VALUES.map(k => ({
      n: k,
      cdf: continuousCDFCurve((x) => cdfStudentT(x, k), xMin, xMax, 200),
    }));

    // KS approximation: evaluate at 200 grid points
    let ks = 0;
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * (xMax - xMin);
      ks = Math.max(ks, Math.abs(cdfStudentT(x, n) - cdfStdNormal(x)));
    }

    // PDF data for right panel
    const pdfLimit = continuousCDFCurve(
      (x) => pdfNormal(x, 0, 1),
      xMin, xMax, 200,
    );
    const pdfXn = continuousCDFCurve(
      (x) => pdfStudentT(x, n),
      xMin, xMax, 200,
    );

    return { limitCDF, xnCDF, familyCDFs, ks, pdfLimit, pdfXn };
  }, [exampleId, n, xMin, xMax]);

  // ── CLT data ──────────────────────────────────────────────────────────────

  const cltData = useMemo(() => {
    if (exampleId !== 'clt-preview') return null;
    const sim = runCLTSimulation(n, cltDist, cltSeed);
    const limitCDF = continuousCDFCurve(cdfStdNormal, -4.5, 4.5, 300);
    return { ...sim, limitCDF };
  }, [exampleId, n, cltDist, cltSeed]);

  // ── KS stat ───────────────────────────────────────────────────────────────

  const ksStat = useMemo(() => {
    if (poissonData) return poissonData.ks;
    if (studentData) return studentData.ks;
    if (cltData) return cltData.ksStat;
    return 0;
  }, [poissonData, studentData, cltData]);

  // ── Right panel max Y ─────────────────────────────────────────────────────

  const rightMaxY = useMemo(() => {
    if (poissonData) {
      return Math.max(...poissonData.pmfLimit.map(d => Math.max(d.pn, d.plimit)), 0.01);
    }
    if (studentData) {
      return Math.max(
        ...studentData.pdfLimit.map(d => d.y),
        ...studentData.pdfXn.map(d => d.y),
        0.01,
      );
    }
    if (cltData) {
      const maxHist = Math.max(...cltData.histBins.map(b => b.density), 0.01);
      const maxPdf = pdfNormal(0, 0, 1);
      return Math.max(maxHist, maxPdf) * 1.1;
    }
    return 0.5;
  }, [poissonData, studentData, cltData]);

  const rightToSvgY = useCallback(
    (y: number) => MARGIN.top + plotH - (y / (rightMaxY * 1.15)) * plotH,
    [plotH, rightMaxY],
  );

  // ── Axis ticks ────────────────────────────────────────────────────────────

  const xTicks = useMemo(() => computeTicks(xMin, xMax, 7), [xMin, xMax]);
  const cdfYTicks = [0, 0.25, 0.5, 0.75, 1.0];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Convergence in Distribution Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-end mb-4">
        {/* Example selector */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Example</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={exampleIdx}
            onChange={(e) => {
              const idx = Number(e.target.value);
              setExampleIdx(idx);
              setN(10);
              setShowFamily(false);
              if (cdfConvergenceExamples[idx].defaultLambda) {
                setLambda(cdfConvergenceExamples[idx].defaultLambda!);
              }
            }}
          >
            {cdfConvergenceExamples.map((ex, i) => (
              <option key={ex.id} value={i}>{ex.name}</option>
            ))}
          </select>
        </label>

        {/* n slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n = {n}</span>
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-32"
          />
        </label>

        {/* Lambda slider (Poisson only) */}
        {exampleId === 'poisson-limit' && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">{'\u03BB'} = {lambda}</span>
            <input
              type="range"
              min={1}
              max={10}
              step={0.5}
              value={lambda}
              onChange={(e) => setLambda(Number(e.target.value))}
              className="w-24"
            />
          </label>
        )}

        {/* CLT underlying distribution */}
        {exampleId === 'clt-preview' && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">Underlying</span>
            <select
              className="rounded border px-2 py-1 text-xs"
              style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
              value={cltDist}
              onChange={(e) => setCltDist(e.target.value)}
            >
              {CLT_UNDERLYING.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </label>
        )}

        {/* Show CDF family toggle */}
        <label className="flex items-center gap-2 text-xs cursor-pointer">
          <input
            type="checkbox"
            checked={showFamily}
            onChange={(e) => setShowFamily(e.target.checked)}
          />
          <span className="font-medium">Show CDF family</span>
        </label>

        {/* Resample button for CLT */}
        {exampleId === 'clt-preview' && (
          <button
            className="rounded border px-2 py-1 text-xs font-medium"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            onClick={() => setCltSeed(s => s + 1)}
          >
            Resample
          </button>
        )}
      </div>

      {/* Description */}
      <div className="text-center text-xs mb-3" style={{ color: 'var(--color-text-muted)' }}>
        {example.description}
        {' \u2014 '}
        KS distance: <span className="font-mono font-semibold">{ksStat.toFixed(4)}</span>
      </div>

      {/* Panels */}
      <div className={`flex ${isNarrow ? 'flex-col' : 'flex-row'} gap-4 justify-center`}>
        {/* Left panel: CDF comparison */}
        <div>
          <div className="text-center text-xs font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
            CDF Comparison
          </div>
          <svg width={panelW} height={chartH} className="block">
            {/* Y-axis grid and labels */}
            {cdfYTicks.map(v => {
              const y = cdfToSvgY(v);
              return (
                <g key={`cy-${v}`}>
                  <line
                    x1={MARGIN.left} x2={MARGIN.left + plotW}
                    y1={y} y2={y}
                    stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
                  />
                  <text
                    x={MARGIN.left - 6} y={y + 3}
                    textAnchor="end" fontSize={9} fill="var(--color-text-muted)"
                  >
                    {v.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* X-axis ticks and labels */}
            {xTicks.map(v => {
              const x = toSvgX(v);
              return (
                <g key={`cx-${v}`}>
                  <line
                    x1={x} x2={x}
                    y1={MARGIN.top} y2={MARGIN.top + plotH}
                    stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
                  />
                  <text
                    x={x} y={chartH - 5}
                    textAnchor="middle" fontSize={9} fill="var(--color-text-muted)"
                  >
                    {v}
                  </text>
                </g>
              );
            })}

            {/* Axes */}
            <line
              x1={MARGIN.left} x2={MARGIN.left + plotW}
              y1={MARGIN.top + plotH} y2={MARGIN.top + plotH}
              stroke="var(--color-text-muted)" strokeWidth={1}
            />
            <line
              x1={MARGIN.left} x2={MARGIN.left}
              y1={MARGIN.top} y2={MARGIN.top + plotH}
              stroke="var(--color-text-muted)" strokeWidth={1}
            />

            {/* Family CDFs (translucent) */}
            {showFamily && exampleId === 'poisson-limit' && poissonData?.familyCDFs.map((fc, i) => (
              <path
                key={`fam-${fc.n}`}
                d={buildStepPath(fc.cdf, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={1}
                opacity={FAMILY_ALPHAS[i]}
              />
            ))}
            {showFamily && exampleId === 'student-normal' && studentData?.familyCDFs.map((fc, i) => (
              <path
                key={`fam-${fc.n}`}
                d={buildSmoothPath(fc.cdf, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={1}
                opacity={FAMILY_ALPHAS[i]}
              />
            ))}
            {showFamily && exampleId === 'clt-preview' && FAMILY_N_VALUES.map((fN, i) => {
              const sim = runCLTSimulation(fN, cltDist, cltSeed + fN);
              const pts = sim.ecdfPts;
              // Build step path from empirical CDF points
              const path = pts
                .map((p, j) => {
                  const px = toSvgX(p.x);
                  const py = cdfToSvgY(p.Fn);
                  if (j === 0) return `M${px.toFixed(1)},${py.toFixed(1)}`;
                  const prevY = cdfToSvgY(pts[j - 1].Fn);
                  return `L${px.toFixed(1)},${prevY.toFixed(1)} L${px.toFixed(1)},${py.toFixed(1)}`;
                })
                .join(' ');
              return (
                <path
                  key={`fam-clt-${fN}`}
                  d={path}
                  fill="none"
                  stroke={CDF_COLOR}
                  strokeWidth={1}
                  opacity={FAMILY_ALPHAS[i]}
                />
              );
            })}

            {/* Limit CDF (dashed black) */}
            {exampleId === 'poisson-limit' && poissonData && (
              <path
                d={buildStepPath(poissonData.limitCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={LIMIT_COLOR}
                strokeWidth={1.5}
                strokeDasharray="5,3"
              />
            )}
            {exampleId === 'student-normal' && studentData && (
              <path
                d={buildSmoothPath(studentData.limitCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={LIMIT_COLOR}
                strokeWidth={1.5}
                strokeDasharray="5,3"
              />
            )}
            {exampleId === 'clt-preview' && cltData && (
              <path
                d={buildSmoothPath(cltData.limitCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={LIMIT_COLOR}
                strokeWidth={1.5}
                strokeDasharray="5,3"
              />
            )}

            {/* Current Xn CDF (solid blue) */}
            {!showFamily && exampleId === 'poisson-limit' && poissonData && (
              <path
                d={buildStepPath(poissonData.xnCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={2}
              />
            )}
            {!showFamily && exampleId === 'student-normal' && studentData && (
              <path
                d={buildSmoothPath(studentData.xnCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={2}
              />
            )}
            {!showFamily && exampleId === 'clt-preview' && cltData && (() => {
              const pts = cltData.ecdfPts;
              const path = pts
                .map((p, j) => {
                  const px = toSvgX(p.x);
                  const py = cdfToSvgY(p.Fn);
                  if (j === 0) return `M${px.toFixed(1)},${py.toFixed(1)}`;
                  const prevY = cdfToSvgY(pts[j - 1].Fn);
                  return `L${px.toFixed(1)},${prevY.toFixed(1)} L${px.toFixed(1)},${py.toFixed(1)}`;
                })
                .join(' ');
              return (
                <path
                  d={path}
                  fill="none"
                  stroke={CDF_COLOR}
                  strokeWidth={2}
                />
              );
            })()}

            {/* When showFamily, also show current n highlighted */}
            {showFamily && exampleId === 'poisson-limit' && poissonData && (
              <path
                d={buildStepPath(poissonData.xnCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={2}
                opacity={0.85}
              />
            )}
            {showFamily && exampleId === 'student-normal' && studentData && (
              <path
                d={buildSmoothPath(studentData.xnCDF, toSvgX, cdfToSvgY)}
                fill="none"
                stroke={CDF_COLOR}
                strokeWidth={2}
                opacity={0.85}
              />
            )}

            {/* KS annotation */}
            <text
              x={MARGIN.left + plotW - 4}
              y={MARGIN.top + 14}
              textAnchor="end"
              fontSize={10}
              fontFamily="monospace"
              fill="var(--color-text-muted)"
            >
              KS = {ksStat.toFixed(4)}
            </text>

            {/* Legend */}
            <line
              x1={MARGIN.left + 8} x2={MARGIN.left + 24}
              y1={MARGIN.top + plotH - 30} y2={MARGIN.top + plotH - 30}
              stroke={CDF_COLOR} strokeWidth={2}
            />
            <text
              x={MARGIN.left + 28} y={MARGIN.top + plotH - 27}
              fontSize={9} fill="var(--color-text-muted)"
            >
              F_Xn
            </text>
            <line
              x1={MARGIN.left + 8} x2={MARGIN.left + 24}
              y1={MARGIN.top + plotH - 18} y2={MARGIN.top + plotH - 18}
              stroke={LIMIT_COLOR} strokeWidth={1.5} strokeDasharray="5,3"
            />
            <text
              x={MARGIN.left + 28} y={MARGIN.top + plotH - 15}
              fontSize={9} fill="var(--color-text-muted)"
            >
              F_X (limit)
            </text>
          </svg>
        </div>

        {/* Right panel */}
        <div>
          <div className="text-center text-xs font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
            {exampleId === 'poisson-limit' ? 'PMF Comparison' :
             exampleId === 'student-normal' ? 'PDF Comparison' : 'Histogram vs N(0,1)'}
          </div>
          <svg width={panelW} height={chartH} className="block">
            {/* Y-axis grid */}
            {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
              const val = frac * rightMaxY;
              const y = rightToSvgY(val);
              if (y < MARGIN.top || y > MARGIN.top + plotH) return null;
              return (
                <g key={`ry-${frac}`}>
                  <line
                    x1={MARGIN.left} x2={MARGIN.left + plotW}
                    y1={y} y2={y}
                    stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
                  />
                  <text
                    x={MARGIN.left - 6} y={y + 3}
                    textAnchor="end" fontSize={9} fill="var(--color-text-muted)"
                  >
                    {val.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* X-axis ticks */}
            {xTicks.map(v => {
              const x = toSvgXRight(v);
              return (
                <g key={`rx-${v}`}>
                  <line
                    x1={x} x2={x}
                    y1={MARGIN.top} y2={MARGIN.top + plotH}
                    stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
                  />
                  <text
                    x={x} y={chartH - 5}
                    textAnchor="middle" fontSize={9} fill="var(--color-text-muted)"
                  >
                    {v}
                  </text>
                </g>
              );
            })}

            {/* Axes */}
            <line
              x1={MARGIN.left} x2={MARGIN.left + plotW}
              y1={MARGIN.top + plotH} y2={MARGIN.top + plotH}
              stroke="var(--color-text-muted)" strokeWidth={1}
            />
            <line
              x1={MARGIN.left} x2={MARGIN.left}
              y1={MARGIN.top} y2={MARGIN.top + plotH}
              stroke="var(--color-text-muted)" strokeWidth={1}
            />

            {/* Poisson: PMF bars */}
            {exampleId === 'poisson-limit' && poissonData?.pmfLimit.map(d => {
              const barW = Math.max(plotW / (poissonData.pmfLimit.length * 2.5), 3);
              const cx = toSvgXRight(d.k);
              // Binomial bar (blue)
              const hXn = Math.max((d.pn / (rightMaxY * 1.15)) * plotH, 0);
              // Poisson bar (gray)
              const hLim = Math.max((d.plimit / (rightMaxY * 1.15)) * plotH, 0);
              return (
                <g key={`pmf-${d.k}`}>
                  <rect
                    x={cx - barW - 0.5}
                    y={MARGIN.top + plotH - hXn}
                    width={barW}
                    height={hXn}
                    fill={CDF_COLOR}
                    opacity={0.6}
                  />
                  <rect
                    x={cx + 0.5}
                    y={MARGIN.top + plotH - hLim}
                    width={barW}
                    height={hLim}
                    fill={LIMIT_COLOR}
                    opacity={0.35}
                  />
                </g>
              );
            })}

            {/* Student-t: PDF curves */}
            {exampleId === 'student-normal' && studentData && (
              <>
                <path
                  d={buildSmoothPath(studentData.pdfXn, toSvgXRight, rightToSvgY)}
                  fill="none"
                  stroke={CDF_COLOR}
                  strokeWidth={2}
                />
                <path
                  d={buildSmoothPath(studentData.pdfLimit, toSvgXRight, rightToSvgY)}
                  fill="none"
                  stroke={LIMIT_COLOR}
                  strokeWidth={1.5}
                  strokeDasharray="5,3"
                />
              </>
            )}

            {/* CLT: Histogram + N(0,1) PDF */}
            {exampleId === 'clt-preview' && cltData && (
              <>
                {cltData.histBins.map((bin, i) => {
                  const x1 = toSvgXRight(bin.lo);
                  const x2 = toSvgXRight(bin.hi);
                  const h = Math.max((bin.density / (rightMaxY * 1.15)) * plotH, 0);
                  return (
                    <rect
                      key={`hist-${i}`}
                      x={x1}
                      y={MARGIN.top + plotH - h}
                      width={Math.max(x2 - x1 - 0.5, 0.5)}
                      height={h}
                      fill={CDF_COLOR}
                      opacity={0.4}
                    />
                  );
                })}
                {/* N(0,1) PDF overlay */}
                <path
                  d={continuousCDFCurve((x) => pdfNormal(x, 0, 1), xMin, xMax, 200)
                    .map((d, i) => {
                      const px = toSvgXRight(d.x);
                      const py = rightToSvgY(d.y);
                      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke={LIMIT_COLOR}
                  strokeWidth={1.5}
                  strokeDasharray="5,3"
                />
              </>
            )}

            {/* Right panel legend */}
            <line
              x1={MARGIN.left + plotW - 80} x2={MARGIN.left + plotW - 64}
              y1={MARGIN.top + 10} y2={MARGIN.top + 10}
              stroke={CDF_COLOR} strokeWidth={2}
            />
            <text
              x={MARGIN.left + plotW - 60} y={MARGIN.top + 13}
              fontSize={9} fill="var(--color-text-muted)"
            >
              X_n
            </text>
            <line
              x1={MARGIN.left + plotW - 80} x2={MARGIN.left + plotW - 64}
              y1={MARGIN.top + 22} y2={MARGIN.top + 22}
              stroke={LIMIT_COLOR} strokeWidth={1.5} strokeDasharray="5,3"
            />
            <text
              x={MARGIN.left + plotW - 60} y={MARGIN.top + 25}
              fontSize={9} fill="var(--color-text-muted)"
            >
              limit
            </text>
          </svg>
        </div>
      </div>

      {/* Footer annotation */}
      <div className="text-center text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>
        {exampleId === 'poisson-limit' && (
          <>Bin(n, {'\u03BB'}/n) {'\u2192'} Poisson({'\u03BB'}) as n {'\u2192'} {'\u221E'} &mdash; the Poisson limit theorem</>
        )}
        {exampleId === 'student-normal' && (
          <>t(n) {'\u2192'} N(0,1) as n {'\u2192'} {'\u221E'} &mdash; heavier tails shrink toward the standard normal</>
        )}
        {exampleId === 'clt-preview' && (
          <>({'\u0058\u0304'}_n &minus; {'\u03BC'})/({'\u03C3'}/{'\u221A'}n) {'\u2192'}_d N(0,1) &mdash; Central Limit Theorem</>
        )}
      </div>
    </div>
  );
}
