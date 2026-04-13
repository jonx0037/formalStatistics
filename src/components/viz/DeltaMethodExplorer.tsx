import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  sampleSequence,
  sampleMean,
  normalSample,
  exponentialSample,
  poissonSample,
  uniformSample,
} from './shared/convergence';
import {
  deltaMethodTransformations,
  deltaMethodDistributions,
} from '../../data/modes-of-convergence-data';
import type { DeltaMethodDistribution } from '../../data/modes-of-convergence-data';

// ── Constants ──────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 30, left: 50 };
const HIST_COLOR = '#2563eb';
const CURVE_COLOR = '#DC2626';
const TRANSFORM_COLOR = '#2563eb';
const TANGENT_COLOR = '#DC2626';
const NUM_CURVE_POINTS = 200;
const NUM_BINS = 40;

// ── Samplers ───────────────────────────────────────────────────────────────

function samplerForDistribution(dist: DeltaMethodDistribution): () => number {
  switch (dist.name) {
    case 'Normal(5, 4)':
      return () => normalSample(5, 2);
    case 'Exponential(1)':
      return () => exponentialSample(1);
    case 'Poisson(5)':
      return () => poissonSample(5);
    case 'Uniform(0, 10)':
      return () => uniformSample(0, 10);
    default:
      return () => normalSample(dist.mu, Math.sqrt(dist.sigmaSquared));
  }
}

// ── Domain-safe transformation application ─────────────────────────────────

function safeApplyG(
  g: (x: number) => number,
  transformName: string,
  xBar: number,
): number | null {
  let safeX = xBar;

  if (transformName === 'log(x)' || transformName === '√x') {
    safeX = Math.max(1e-10, xBar);
  } else if (transformName === '1/x') {
    safeX = Math.max(1e-10, Math.abs(xBar)) * Math.sign(xBar);
    if (safeX === 0) safeX = 1e-10;
  }

  const result = g(safeX);

  if (transformName === 'e^x' && result > 1e10) {
    return null; // discard
  }

  if (!isFinite(result) || isNaN(result)) {
    return null;
  }

  return result;
}

// ── Normal PDF ─────────────────────────────────────────────────────────────

function normalPDF(x: number, mu: number, variance: number): number {
  if (variance <= 0) return 0;
  const coeff = 1 / Math.sqrt(2 * Math.PI * variance);
  const exponent = -((x - mu) ** 2) / (2 * variance);
  return coeff * Math.exp(exponent);
}

// ── Histogram binning ──────────────────────────────────────────────────────

interface HistBin {
  x0: number;
  x1: number;
  count: number;
  density: number;
}

function computeHistogram(values: number[], numBins: number): HistBin[] {
  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;

  if (range === 0) {
    return [{ x0: min - 0.5, x1: max + 0.5, count: values.length, density: values.length }];
  }

  const binWidth = range / numBins;
  const bins: HistBin[] = [];

  for (let i = 0; i < numBins; i++) {
    bins.push({
      x0: min + i * binWidth,
      x1: min + (i + 1) * binWidth,
      count: 0,
      density: 0,
    });
  }

  for (const v of values) {
    let idx = Math.floor((v - min) / binWidth);
    if (idx >= numBins) idx = numBins - 1;
    if (idx < 0) idx = 0;
    bins[idx].count++;
  }

  // Convert counts to density (so the histogram integrates to 1)
  const n = values.length;
  for (const bin of bins) {
    bin.density = bin.count / (n * binWidth);
  }

  return bins;
}

// ── Logarithmic slider helpers ─────────────────────────────────────────────

function logSliderToN(sliderVal: number): number {
  // sliderVal in [0, 1] -> n in [10, 5000] on log scale
  const minLog = Math.log(10);
  const maxLog = Math.log(5000);
  return Math.round(Math.exp(minLog + sliderVal * (maxLog - minLog)));
}

function nToLogSlider(n: number): number {
  const minLog = Math.log(10);
  const maxLog = Math.log(5000);
  return (Math.log(n) - minLog) / (maxLog - minLog);
}

// ── Nice tick generation ───────────────────────────────────────────────────

function niceTicks(min: number, max: number, approxCount: number): number[] {
  const range = max - min;
  if (range === 0) return [min];

  const rawStep = range / approxCount;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const niceSteps = [1, 2, 5, 10];
  let step = mag;
  for (const ns of niceSteps) {
    if (ns * mag >= rawStep) {
      step = ns * mag;
      break;
    }
  }

  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.01; v += step) {
    ticks.push(Math.round(v * 1e10) / 1e10);
  }
  return ticks;
}

// ── Format number ──────────────────────────────────────────────────────────

function fmt(v: number, decimals = 4): string {
  if (!isFinite(v)) return 'N/A';
  if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(2);
  if (Math.abs(v) >= 10000) return v.toExponential(2);
  return v.toFixed(decimals);
}

// ── Component ──────────────────────────────────────────────────────────────

export default function DeltaMethodExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [transformIndex, setTransformIndex] = useState(0);
  const [distIndex, setDistIndex] = useState(0);
  const [n, setN] = useState(100);

  const transform = deltaMethodTransformations[transformIndex];
  const dist = deltaMethodDistributions[distIndex];
  const { mu, sigmaSquared } = dist;
  const { g, gPrime, name: transformName } = transform;

  // ── Simulation ─────────────────────────────────────────────────────────

  const simData = useMemo(() => {
    const numReps = Math.min(2000, Math.ceil(400000 / n));
    const sampler = samplerForDistribution(dist);
    const gXbarValues: number[] = [];

    for (let rep = 0; rep < numReps; rep++) {
      const samples = sampleSequence(sampler, n);
      const xBar = sampleMean(samples);
      const gXbar = safeApplyG(g, transformName, xBar);
      if (gXbar !== null) {
        gXbarValues.push(gXbar);
      }
    }

    // Simulation statistics
    const simMean = gXbarValues.length > 0
      ? gXbarValues.reduce((s, v) => s + v, 0) / gXbarValues.length
      : 0;
    const simVariance = gXbarValues.length > 1
      ? gXbarValues.reduce((s, v) => s + (v - simMean) ** 2, 0) / (gXbarValues.length - 1)
      : 0;

    // Delta method quantities
    const gMu = g(mu);
    const gPrimeMu = gPrime(mu);
    const deltaVar = gPrimeMu ** 2 * sigmaSquared / n;

    // Histogram
    const histogram = computeHistogram(gXbarValues, NUM_BINS);

    return {
      gXbarValues,
      histogram,
      simMean,
      simVariance,
      gMu,
      gPrimeMu,
      deltaVar,
      numReps,
      numValid: gXbarValues.length,
    };
  }, [transformIndex, distIndex, n]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Layout ─────────────────────────────────────────────────────────────

  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 624;
  const chartW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = 240;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Left panel: Histogram + Normal overlay ─────────────────────────────

  const histPanel = useMemo(() => {
    const { histogram, gMu, deltaVar } = simData;
    if (histogram.length === 0) {
      return { xMin: 0, xMax: 1, yMax: 1, xTicks: [0, 0.5, 1], normalCurve: [] };
    }

    const histXMin = histogram[0].x0;
    const histXMax = histogram[histogram.length - 1].x1;

    // Extend domain to cover delta method normal curve
    const deltaSd = Math.sqrt(Math.max(deltaVar, 1e-12));
    const normXMin = gMu - 4 * deltaSd;
    const normXMax = gMu + 4 * deltaSd;

    const xMin = Math.min(histXMin, normXMin);
    const xMax = Math.max(histXMax, normXMax);

    // Compute max density across histogram and normal curve
    let maxDensity = 0;
    for (const bin of histogram) {
      if (bin.density > maxDensity) maxDensity = bin.density;
    }

    // Normal overlay curve
    const normalCurve: { x: number; y: number }[] = [];
    if (deltaVar > 0 && isFinite(deltaVar)) {
      const peakDensity = normalPDF(gMu, gMu, deltaVar);
      if (peakDensity > maxDensity) maxDensity = peakDensity;
      for (let i = 0; i <= NUM_CURVE_POINTS; i++) {
        const x = xMin + (i / NUM_CURVE_POINTS) * (xMax - xMin);
        normalCurve.push({ x, y: normalPDF(x, gMu, deltaVar) });
      }
    }

    const yMax = maxDensity * 1.15 || 1;
    const xTicks = niceTicks(xMin, xMax, 5);

    return { xMin, xMax, yMax, xTicks, normalCurve };
  }, [simData]);

  // ── Right panel: Transformation plot ───────────────────────────────────

  const transformPanel = useMemo(() => {
    const sigma = Math.sqrt(sigmaSquared);
    const seN = sigma / Math.sqrt(n);

    // x-range: show at least mu +/- 3*sigma/sqrt(n), but widen for visibility
    const halfRange = Math.max(3 * seN, sigma * 0.5, Math.abs(mu) * 0.1, 0.5);
    const xMin = mu - halfRange;
    const xMax = mu + halfRange;

    // Compute g(x) curve
    const curvePoints: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_CURVE_POINTS; i++) {
      const x = xMin + (i / NUM_CURVE_POINTS) * (xMax - xMin);
      const safeVal = safeApplyG(g, transformName, x);
      if (safeVal !== null) {
        curvePoints.push({ x, y: safeVal });
      }
    }

    if (curvePoints.length === 0) {
      return { xMin, xMax, yMin: -1, yMax: 1, curvePoints: [], tangentPoints: [], shadeX: [0, 0], shadeY: [0, 0], xTicks: [], yTicks: [] };
    }

    // Tangent line at mu: y = g(mu) + g'(mu)*(x - mu)
    const gMu = g(mu);
    const gPrimeMu = gPrime(mu);
    const tangentPoints = [
      { x: xMin, y: gMu + gPrimeMu * (xMin - mu) },
      { x: xMax, y: gMu + gPrimeMu * (xMax - mu) },
    ];

    // Shading region for X̄ concentration
    const shadeLeft = mu - 2 * seN;
    const shadeRight = mu + 2 * seN;
    const shadeYLeft = gMu + gPrimeMu * (shadeLeft - mu);
    const shadeYRight = gMu + gPrimeMu * (shadeRight - mu);

    // Y-range from all points
    const allYs = [
      ...curvePoints.map((p) => p.y),
      ...tangentPoints.map((p) => p.y),
    ].filter((v) => isFinite(v));

    let yMin = Math.min(...allYs);
    let yMax = Math.max(...allYs);
    const yPad = (yMax - yMin) * 0.1 || 0.5;
    yMin -= yPad;
    yMax += yPad;

    const xTicks = niceTicks(xMin, xMax, 5);
    const yTicks = niceTicks(yMin, yMax, 4);

    return {
      xMin,
      xMax,
      yMin,
      yMax,
      curvePoints,
      tangentPoints,
      shadeX: [shadeLeft, shadeRight],
      shadeY: [shadeYLeft, shadeYRight],
      gMu,
      xTicks,
      yTicks,
    };
  }, [mu, sigmaSquared, n, g, gPrime, transformName]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── SVG coordinate helpers (histogram panel) ───────────────────────────

  const hToSvgX = (x: number) =>
    MARGIN.left + ((x - histPanel.xMin) / (histPanel.xMax - histPanel.xMin || 1)) * plotW;
  const hToSvgY = (y: number) =>
    MARGIN.top + plotH - (y / histPanel.yMax) * plotH;

  // ── SVG coordinate helpers (transform panel) ───────────────────────────

  const tXRange = transformPanel.xMax - transformPanel.xMin || 1;
  const tYRange = transformPanel.yMax - transformPanel.yMin || 1;
  const tToSvgX = (x: number) =>
    MARGIN.left + ((x - transformPanel.xMin) / tXRange) * plotW;
  const tToSvgY = (y: number) =>
    MARGIN.top + plotH - ((y - transformPanel.yMin) / tYRange) * plotH;

  // ── Build SVG paths ────────────────────────────────────────────────────

  const normalPath = histPanel.normalCurve.length > 0
    ? histPanel.normalCurve
        .map((d, i) => {
          const px = hToSvgX(d.x);
          const py = hToSvgY(d.y);
          return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
        })
        .join(' ')
    : '';

  const transformPath = transformPanel.curvePoints.length > 0
    ? transformPanel.curvePoints
        .map((d, i) => {
          const px = tToSvgX(d.x);
          const py = tToSvgY(d.y);
          // Clamp to chart area
          const clampedPy = Math.max(MARGIN.top - 5, Math.min(MARGIN.top + plotH + 5, py));
          return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${clampedPy.toFixed(1)}`;
        })
        .join(' ')
    : '';

  const tangentPath = transformPanel.tangentPoints.length === 2
    ? (() => {
        const p0 = transformPanel.tangentPoints[0];
        const p1 = transformPanel.tangentPoints[1];
        const x0 = tToSvgX(p0.x);
        const y0 = Math.max(MARGIN.top - 5, Math.min(MARGIN.top + plotH + 5, tToSvgY(p0.y)));
        const x1 = tToSvgX(p1.x);
        const y1 = Math.max(MARGIN.top - 5, Math.min(MARGIN.top + plotH + 5, tToSvgY(p1.y)));
        return `M${x0.toFixed(1)},${y0.toFixed(1)} L${x1.toFixed(1)},${y1.toFixed(1)}`;
      })()
    : '';

  // ── Shade polygon: show X̄ concentration region projected through tangent ─

  const shadePolygon = (() => {
    const [sxL, sxR] = transformPanel.shadeX;
    const [syL, syR] = transformPanel.shadeY;
    if (!isFinite(sxL) || !isFinite(sxR) || !isFinite(syL) || !isFinite(syR)) return '';

    const xL = tToSvgX(sxL);
    const xR = tToSvgX(sxR);
    const yBase = MARGIN.top + plotH;
    const yL = Math.max(MARGIN.top, Math.min(yBase, tToSvgY(syL)));
    const yR = Math.max(MARGIN.top, Math.min(yBase, tToSvgY(syR)));

    // Polygon: bottom-left, top-left (tangent), top-right (tangent), bottom-right
    return `${xL.toFixed(1)},${yBase} ${xL.toFixed(1)},${yL.toFixed(1)} ${xR.toFixed(1)},${yR.toFixed(1)} ${xR.toFixed(1)},${yBase}`;
  })();

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Delta Method Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {/* Transformation selector */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">g(x)</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={transformIndex}
            onChange={(e) => setTransformIndex(Number(e.target.value))}
          >
            {deltaMethodTransformations.map((t, i) => (
              <option key={t.name} value={i}>{t.name}</option>
            ))}
          </select>
        </label>

        {/* Distribution selector */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={distIndex}
            onChange={(e) => setDistIndex(Number(e.target.value))}
          >
            {deltaMethodDistributions.map((d, i) => (
              <option key={d.name} value={i}>{d.name}</option>
            ))}
          </select>
        </label>

        {/* Sample size slider (logarithmic) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={nToLogSlider(n)}
            onChange={(e) => setN(logSliderToN(Number(e.target.value)))}
            className="w-28"
          />
          <span className="w-12 tabular-nums text-right">{n}</span>
        </label>
      </div>

      {/* Two-panel chart area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* Left panel: Simulation histogram */}
        <div style={isNarrow ? { width: '100%' } : { width: chartW }}>
          <div className="text-center text-xs opacity-60 mb-1">
            Histogram of g(X&#772;<sub>n</sub>) &mdash; {simData.numValid} replications
          </div>
          <svg width={chartW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {[0.25, 0.5, 0.75].map((frac) => {
              const y = MARGIN.top + plotH * (1 - frac);
              return (
                <line
                  key={frac}
                  x1={MARGIN.left}
                  y1={y}
                  x2={chartW - MARGIN.right}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity={0.08}
                />
              );
            })}

            {/* Histogram bars */}
            {simData.histogram.map((bin, i) => {
              const x = hToSvgX(bin.x0);
              const w = hToSvgX(bin.x1) - hToSvgX(bin.x0);
              const y = hToSvgY(bin.density);
              const h = hToSvgY(0) - y;
              return (
                <rect
                  key={i}
                  x={x}
                  y={y}
                  width={Math.max(0, w - 0.5)}
                  height={Math.max(0, h)}
                  fill={HIST_COLOR}
                  fillOpacity={0.3}
                  stroke={HIST_COLOR}
                  strokeOpacity={0.5}
                  strokeWidth={0.5}
                />
              );
            })}

            {/* Delta method Normal density overlay */}
            {normalPath && (
              <path
                d={normalPath}
                fill="none"
                stroke={CURVE_COLOR}
                strokeWidth={2}
              />
            )}

            {/* g(mu) marker on x-axis */}
            {isFinite(simData.gMu) && (
              <line
                x1={hToSvgX(simData.gMu)}
                y1={MARGIN.top}
                x2={hToSvgX(simData.gMu)}
                y2={MARGIN.top + plotH}
                stroke={CURVE_COLOR}
                strokeWidth={1}
                strokeDasharray="3,3"
                strokeOpacity={0.7}
              />
            )}

            {/* Simulation mean marker */}
            {isFinite(simData.simMean) && (
              <line
                x1={hToSvgX(simData.simMean)}
                y1={MARGIN.top}
                x2={hToSvgX(simData.simMean)}
                y2={MARGIN.top + plotH}
                stroke={HIST_COLOR}
                strokeWidth={1}
                strokeDasharray="6,3"
                strokeOpacity={0.7}
              />
            )}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={chartW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks */}
            {histPanel.xTicks.map((v, i) => (
              <text
                key={i}
                x={hToSvgX(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {Math.abs(v) >= 100 ? v.toExponential(1) : Number.isInteger(v) ? v : v.toFixed(2)}
              </text>
            ))}

            {/* Y-axis labels */}
            {[0.5, 1.0].map((frac) => {
              const val = histPanel.yMax * frac;
              return (
                <text
                  key={frac}
                  x={MARGIN.left - 6}
                  y={hToSvgY(val) + 3}
                  textAnchor="end"
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.6}
                >
                  {val < 0.01 ? val.toExponential(1) : val.toFixed(2)}
                </text>
              );
            })}

            {/* Axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              g(X&#772;)
            </text>
            <text
              x={12}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
            >
              density
            </text>

            {/* Legend */}
            <line x1={chartW - MARGIN.right - 80} y1={MARGIN.top + 8} x2={chartW - MARGIN.right - 65} y2={MARGIN.top + 8} stroke={HIST_COLOR} strokeWidth={2} strokeDasharray="6,3" />
            <text x={chartW - MARGIN.right - 62} y={MARGIN.top + 11} fontSize={8} fill="currentColor" opacity={0.7}>sim mean</text>
            <line x1={chartW - MARGIN.right - 80} y1={MARGIN.top + 20} x2={chartW - MARGIN.right - 65} y2={MARGIN.top + 20} stroke={CURVE_COLOR} strokeWidth={2} />
            <text x={chartW - MARGIN.right - 62} y={MARGIN.top + 23} fontSize={8} fill="currentColor" opacity={0.7}>delta N</text>
          </svg>
        </div>

        {/* Right panel: Transformation plot */}
        <div style={isNarrow ? { width: '100%' } : { width: chartW }}>
          <div className="text-center text-xs opacity-60 mb-1">
            g(x) = {transform.label} with tangent at x = {mu.toFixed(1)}
          </div>
          <svg width={chartW} height={chartH} className="block mx-auto">
            {/* Grid lines */}
            {transformPanel.yTicks.map((v, i) => (
              <line
                key={`yg-${i}`}
                x1={MARGIN.left}
                y1={tToSvgY(v)}
                x2={chartW - MARGIN.right}
                y2={tToSvgY(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}
            {transformPanel.xTicks.map((v, i) => (
              <line
                key={`xg-${i}`}
                x1={tToSvgX(v)}
                y1={MARGIN.top}
                x2={tToSvgX(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Shaded X̄ concentration region */}
            {shadePolygon && (
              <polygon
                points={shadePolygon}
                fill={TRANSFORM_COLOR}
                fillOpacity={0.08}
                stroke={TRANSFORM_COLOR}
                strokeOpacity={0.2}
                strokeWidth={0.5}
              />
            )}

            {/* g(x) curve */}
            {transformPath && (
              <path
                d={transformPath}
                fill="none"
                stroke={TRANSFORM_COLOR}
                strokeWidth={2.5}
              />
            )}

            {/* Tangent line (dashed red) */}
            {tangentPath && (
              <path
                d={tangentPath}
                fill="none"
                stroke={TANGENT_COLOR}
                strokeWidth={1.5}
                strokeDasharray="6,4"
              />
            )}

            {/* mu marker on x-axis */}
            <line
              x1={tToSvgX(mu)}
              y1={MARGIN.top}
              x2={tToSvgX(mu)}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="2,2"
              strokeOpacity={0.3}
            />

            {/* g(mu) dot */}
            {'gMu' in transformPanel && isFinite(transformPanel.gMu as number) && (
              <circle
                cx={tToSvgX(mu)}
                cy={tToSvgY(transformPanel.gMu as number)}
                r={4}
                fill={TANGENT_COLOR}
                stroke="white"
                strokeWidth={1.5}
              />
            )}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={chartW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks */}
            {transformPanel.xTicks.map((v, i) => (
              <text
                key={i}
                x={tToSvgX(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {Number.isInteger(v) ? v : v.toFixed(2)}
              </text>
            ))}

            {/* Y-axis labels */}
            {transformPanel.yTicks.map((v, i) => (
              <text
                key={i}
                x={MARGIN.left - 6}
                y={tToSvgY(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {Math.abs(v) >= 100 ? v.toExponential(1) : Number.isInteger(v) ? v : v.toFixed(2)}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={MARGIN.left + plotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              x
            </text>
            <text
              x={12}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
            >
              g(x)
            </text>

            {/* Legend */}
            <line x1={chartW - MARGIN.right - 85} y1={MARGIN.top + 8} x2={chartW - MARGIN.right - 70} y2={MARGIN.top + 8} stroke={TRANSFORM_COLOR} strokeWidth={2.5} />
            <text x={chartW - MARGIN.right - 67} y={MARGIN.top + 11} fontSize={8} fill="currentColor" opacity={0.7}>{transform.label}</text>
            <line x1={chartW - MARGIN.right - 85} y1={MARGIN.top + 20} x2={chartW - MARGIN.right - 70} y2={MARGIN.top + 20} stroke={TANGENT_COLOR} strokeWidth={1.5} strokeDasharray="6,4" />
            <text x={chartW - MARGIN.right - 67} y={MARGIN.top + 23} fontSize={8} fill="currentColor" opacity={0.7}>tangent</text>
          </svg>
        </div>
      </div>

      {/* Readout table */}
      <div className="mt-4 overflow-x-auto">
        <table className="mx-auto text-xs border-collapse" style={{ borderColor: 'var(--color-border)' }}>
          <thead>
            <tr>
              <th className="px-3 py-1 text-left border-b font-medium" style={{ borderColor: 'var(--color-border)' }}>Quantity</th>
              <th className="px-3 py-1 text-right border-b font-medium" style={{ borderColor: 'var(--color-border)' }}>Delta Method</th>
              <th className="px-3 py-1 text-right border-b font-medium" style={{ borderColor: 'var(--color-border)' }}>Simulation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>Mean of g(X&#772;)</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }}>{fmt(simData.gMu)}</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }}>{fmt(simData.simMean)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>Variance of g(X&#772;)</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }}>{fmt(simData.deltaVar)}</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }}>{fmt(simData.simVariance)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>&mu;</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }} colSpan={2}>{fmt(mu)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>&sigma;&sup2;</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }} colSpan={2}>{fmt(sigmaSquared)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>g(&mu;)</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }} colSpan={2}>{fmt(simData.gMu)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1 border-b" style={{ borderColor: 'var(--color-border)' }}>g&prime;(&mu;)</td>
              <td className="px-3 py-1 text-right tabular-nums border-b" style={{ borderColor: 'var(--color-border)' }} colSpan={2}>{fmt(simData.gPrimeMu)}</td>
            </tr>
            <tr>
              <td className="px-3 py-1">[g&prime;(&mu;)]&sup2; &sigma;&sup2; / n</td>
              <td className="px-3 py-1 text-right tabular-nums" colSpan={2}>{fmt(simData.deltaVar)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
