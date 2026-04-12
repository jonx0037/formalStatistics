import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBinomial,
  pdfNormal,
  pdfExponential,
  pdfUniform,
  cdfNormal,
  cdfExponential,
  trapezoidalIntegral,
} from './shared/distributions';
import {
  expectationDiscrete,
  varianceDiscrete,
  expectationContinuous,
  varianceContinuous,
  markovBound,
  chebyshevBound,
  stdDev,
} from './shared/moments';
import type { DiscreteDistribution, ContinuousDistribution } from './shared/moments';
import {
  inequalityPresets,
  jensenPresets,
} from '../../data/expectation-moments-data';
import type { InequalityPreset, JensenPreset } from '../../data/expectation-moments-data';
import { distributionColors } from './shared/colorScales';

// ── Types ───────────────────────────────────────────────────────────────────

type Mode = 'markov' | 'chebyshev' | 'jensen';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 30, left: 50 };
const SVG_HEIGHT = 280;
const BAR_CHART_HEIGHT = 100;
const N_CURVE_POINTS = 300;

// ── Distribution helpers ────────────────────────────────────────────────────

function evalPDF(preset: InequalityPreset, x: number): number {
  const p = preset.params;
  switch (preset.distribution) {
    case 'Exponential':
      return pdfExponential(x, p.lambda);
    case 'Normal':
      return pdfNormal(x, p.mu, p.sigma2);
    case 'Uniform':
      return pdfUniform(x, p.a, p.b);
    default:
      return 0;
  }
}

function evalPMF(preset: InequalityPreset, k: number): number {
  const p = preset.params;
  switch (preset.distribution) {
    case 'Binomial':
      return pmfBinomial(k, p.n, p.p);
    default:
      return 0;
  }
}

/** X-axis range for plotting a given preset. */
function getXRange(preset: InequalityPreset): [number, number] {
  const p = preset.params;
  switch (preset.distribution) {
    case 'Exponential':
      return [0, 6 / p.lambda + 1];
    case 'Normal': {
      const sd = Math.sqrt(p.sigma2);
      return [p.mu - 4.5 * sd, p.mu + 4.5 * sd];
    }
    case 'Uniform':
      return [p.a - 1, p.b + 1];
    case 'Binomial':
      return [-1, p.n + 1];
    default:
      return [0, 10];
  }
}

/** Discrete support values for a preset. */
function getDiscreteRange(preset: InequalityPreset): number[] {
  const p = preset.params;
  switch (preset.distribution) {
    case 'Binomial':
      return Array.from({ length: p.n + 1 }, (_, i) => i);
    default:
      return [];
  }
}

/** Build a DiscreteDistribution object for moments.ts functions. */
function toDiscreteDistribution(preset: InequalityPreset): DiscreteDistribution {
  const range = getDiscreteRange(preset);
  return {
    values: range,
    probabilities: range.map((k) => evalPMF(preset, k)),
  };
}

/** Build a ContinuousDistribution object for moments.ts functions. */
function toContinuousDistribution(preset: InequalityPreset): ContinuousDistribution {
  const [lo, hi] = getXRange(preset);
  return {
    pdf: (x: number) => evalPDF(preset, x),
    support: [lo, hi],
  };
}

/**
 * Exact or numerical tail probability P(X >= a).
 * Uses closed-form where available; falls back to trapezoidal integration.
 */
function tailProbability(preset: InequalityPreset, a: number): number {
  const p = preset.params;

  if (preset.type === 'discrete') {
    // Binomial: sum PMF from ceil(a) to n
    const startK = Math.max(0, Math.ceil(a));
    let sum = 0;
    for (let k = startK; k <= p.n; k++) {
      sum += pmfBinomial(k, p.n, p.p);
    }
    return Math.min(1, Math.max(0, sum));
  }

  switch (preset.distribution) {
    case 'Exponential':
      // P(X >= a) = e^(-lambda * a) for a >= 0
      if (a <= 0) return 1;
      return Math.exp(-p.lambda * a);
    case 'Normal':
      // P(X >= a) = 1 - CDF(a)
      return 1 - cdfNormal(a, p.mu, p.sigma2);
    case 'Uniform':
      // P(X >= a) = (b - a) / (b - a_param) clamped
      if (a <= p.a) return 1;
      if (a >= p.b) return 0;
      return (p.b - a) / (p.b - p.a);
    default: {
      // Fallback: numerical integration
      const [, hi] = getXRange(preset);
      return trapezoidalIntegral((x) => evalPDF(preset, x), a, hi, 1000);
    }
  }
}

/**
 * Two-tail probability P(|X - mu| >= epsilon).
 * For Chebyshev: epsilon = k * sigma.
 */
function twoTailProbability(
  preset: InequalityPreset,
  mean: number,
  epsilon: number,
): number {
  const p = preset.params;

  if (preset.type === 'discrete') {
    // Binomial: sum PMF where |k - mean| >= epsilon
    let sum = 0;
    for (let k = 0; k <= p.n; k++) {
      if (Math.abs(k - mean) >= epsilon) {
        sum += pmfBinomial(k, p.n, p.p);
      }
    }
    return Math.min(1, Math.max(0, sum));
  }

  const lowerBound = mean - epsilon;
  const upperBound = mean + epsilon;

  switch (preset.distribution) {
    case 'Exponential': {
      // P(X < lower) + P(X > upper)
      const lower = lowerBound <= 0 ? 0 : cdfExponential(lowerBound, p.lambda);
      const upper = upperBound <= 0 ? 1 : 1 - cdfExponential(upperBound, p.lambda);
      return Math.min(1, Math.max(0, lower + upper));
    }
    case 'Normal': {
      const lower = cdfNormal(lowerBound, p.mu, p.sigma2);
      const upper = 1 - cdfNormal(upperBound, p.mu, p.sigma2);
      return Math.min(1, Math.max(0, lower + upper));
    }
    case 'Uniform': {
      const width = p.b - p.a;
      if (width <= 0) return 0;
      // P(X < lowerBound) + P(X > upperBound)
      const leftTail = lowerBound <= p.a ? 0 : lowerBound >= p.b ? 1 : (lowerBound - p.a) / width;
      const rightTail = upperBound >= p.b ? 0 : upperBound <= p.a ? 1 : (p.b - upperBound) / width;
      return Math.min(1, Math.max(0, leftTail + rightTail));
    }
    default: {
      const [lo, hi] = getXRange(preset);
      const leftTail = trapezoidalIntegral(
        (x) => evalPDF(preset, x),
        lo,
        lowerBound,
        500,
      );
      const rightTail = trapezoidalIntegral(
        (x) => evalPDF(preset, x),
        upperBound,
        hi,
        500,
      );
      return Math.min(1, Math.max(0, leftTail + rightTail));
    }
  }
}


// ── Slider range helpers ────────────────────────────────────────────────────

function getMarkovRange(preset: InequalityPreset): { min: number; max: number; step: number } {
  const p = preset.params;
  switch (preset.distribution) {
    case 'Exponential':
      return { min: 0.5, max: 6 / p.lambda, step: 0.1 };
    case 'Normal': {
      const sd = Math.sqrt(p.sigma2);
      // Markov requires X >= 0, but Normal can go negative.
      // Still useful pedagogically for showing when bounds break.
      return { min: 0.5, max: p.mu + 4 * sd, step: 0.1 };
    }
    case 'Uniform':
      return { min: Math.max(0.1, p.a), max: p.b, step: 0.1 };
    case 'Binomial':
      return { min: 1, max: p.n, step: 1 };
    default:
      return { min: 0.5, max: 10, step: 0.1 };
  }
}

// ── Jensen helpers ──────────────────────────────────────────────────────────

/**
 * E[g(X)] for X ~ Normal(mu, sigma2) via numerical integration.
 * For -log(x), we restrict the integration domain to avoid log(0).
 */
function jensenEgX(
  g: (x: number) => number,
  mu: number,
  sigma2: number,
): number {
  const sd = Math.sqrt(sigma2);
  // For -log(x) we need x > 0
  const lo = Math.max(mu - 5 * sd, 0.001);
  const hi = mu + 5 * sd;
  return trapezoidalIntegral(
    (x) => g(x) * pdfNormal(x, mu, sigma2),
    lo,
    hi,
    2000,
  );
}

// ── Sub-components ──────────────────────────────────────────────────────────

/** Horizontal comparison bar chart for true probability vs bound. */
function ComparisonBars({
  trueProb,
  bound,
  label,
  boundLabel,
  boundColor,
  width,
}: {
  trueProb: number;
  bound: number;
  label: string;
  boundLabel: string;
  boundColor: string;
  width: number;
}) {
  const barW = Math.max(width - 120, 100);
  const barH = 22;
  const gap = 8;
  const labelW = 110;
  const maxVal = Math.max(bound, trueProb, 0.01);
  const scale = barW / Math.min(maxVal * 1.15, 1);

  return (
    <svg width={width} height={BAR_CHART_HEIGHT} className="block">
      {/* True probability */}
      <text x={4} y={20} className="fill-current" style={{ fontSize: '11px', fontWeight: 500 }}>
        {label}
      </text>
      <rect
        x={labelW}
        y={8}
        width={Math.max(0, Math.min(trueProb * scale, barW))}
        height={barH}
        fill="var(--color-danger, #DC2626)"
        opacity={0.8}
        rx={3}
      >
        <animate attributeName="width" from="0" to={Math.max(0, Math.min(trueProb * scale, barW))} dur="0.3s" fill="freeze" />
      </rect>
      <text
        x={labelW + Math.max(0, Math.min(trueProb * scale, barW)) + 6}
        y={24}
        className="fill-current"
        style={{ fontSize: '11px', fontFamily: 'monospace' }}
      >
        {trueProb.toFixed(4)}
      </text>

      {/* Bound */}
      <text x={4} y={20 + barH + gap + 12} className="fill-current" style={{ fontSize: '11px', fontWeight: 500 }}>
        {boundLabel}
      </text>
      <rect
        x={labelW}
        y={barH + gap + 8}
        width={Math.max(0, Math.min(bound * scale, barW))}
        height={barH}
        fill={boundColor}
        opacity={0.8}
        rx={3}
      >
        <animate attributeName="width" from="0" to={Math.max(0, Math.min(bound * scale, barW))} dur="0.3s" fill="freeze" />
      </rect>
      <text
        x={labelW + Math.max(0, Math.min(bound * scale, barW)) + 6}
        y={barH + gap + 24}
        className="fill-current"
        style={{ fontSize: '11px', fontFamily: 'monospace' }}
      >
        {bound.toFixed(4)}
      </text>

      {/* Tightness ratio */}
      <text x={4} y={2 * (barH + gap) + 22} className="fill-current" style={{ fontSize: '11px' }}>
        Bound / True ={' '}
        <tspan style={{ fontFamily: 'monospace', fontWeight: 600 }}>
          {trueProb > 1e-8 ? (bound / trueProb).toFixed(2) : '--'}x
        </tspan>
        <tspan style={{ fontSize: '10px', opacity: 0.6 }}>
          {' '}(1.00x = tight)
        </tspan>
      </text>
    </svg>
  );
}

// ── Markov tab ──────────────────────────────────────────────────────────────

function MarkovPanel({
  preset,
  svgW,
}: {
  preset: InequalityPreset;
  svgW: number;
}) {
  const [threshold, setThreshold] = useState(preset.defaultMarkovThreshold);
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;
  const [xMin, xMax] = getXRange(preset);

  // Compute mean
  const mean = useMemo(() => {
    if (preset.type === 'discrete') {
      return expectationDiscrete(toDiscreteDistribution(preset));
    }
    return expectationContinuous(toContinuousDistribution(preset));
  }, [preset]);

  const markov = markovBound(mean, threshold);
  const trueProb = useMemo(
    () => tailProbability(preset, threshold),
    [preset, threshold],
  );

  // Scales
  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const yScaleMax = useMemo(() => {
    if (preset.type === 'discrete') {
      const range = getDiscreteRange(preset);
      return Math.max(...range.map((k) => evalPMF(preset, k)), 0.01);
    }
    let maxY = 0;
    const step = (xMax - xMin) / N_CURVE_POINTS;
    for (let i = 0; i <= N_CURVE_POINTS; i++) {
      maxY = Math.max(maxY, evalPDF(preset, xMin + i * step));
    }
    return maxY || 0.01;
  }, [preset, xMin, xMax]);

  const yScale = useCallback(
    (val: number) => MARGIN.top + plotH - (val / (yScaleMax * 1.15)) * plotH,
    [plotH, yScaleMax],
  );

  // Build PDF path + shaded tail region
  const { curvePath, shadedPath } = useMemo(() => {
    if (preset.type === 'discrete') return { curvePath: '', shadedPath: '' };
    const step = (xMax - xMin) / N_CURVE_POINTS;
    const points = Array.from({ length: N_CURVE_POINTS + 1 }, (_, i) => {
      const x = xMin + i * step;
      return { x, y: evalPDF(preset, x) };
    });
    const curve = points
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');
    // Shaded tail: x >= threshold
    const tailPts = points.filter((d) => d.x >= threshold);
    if (tailPts.length < 2) return { curvePath: curve, shadedPath: '' };
    const baseline = yScale(0);
    let shaded = `M${xScale(tailPts[0].x).toFixed(1)},${baseline}`;
    for (const d of tailPts) {
      shaded += ` L${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`;
    }
    shaded += ` L${xScale(tailPts[tailPts.length - 1].x).toFixed(1)},${baseline} Z`;
    return { curvePath: curve, shadedPath: shaded };
  }, [preset, xMin, xMax, threshold, xScale, yScale]);

  // Discrete bars data
  const discreteData = useMemo(() => {
    if (preset.type !== 'discrete') return [];
    return getDiscreteRange(preset).map((k) => ({ k, p: evalPMF(preset, k) }));
  }, [preset]);

  const barWidth = useMemo(() => {
    if (discreteData.length <= 1) return plotW * 0.6;
    const range = discreteData[discreteData.length - 1].k - discreteData[0].k || 1;
    return Math.max(4, Math.min(24, (plotW / (range + 1)) * 0.65));
  }, [discreteData, plotW]);

  const sliderRange = getMarkovRange(preset);

  // X-axis ticks
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const span = xMax - xMin;
    const step = Math.pow(10, Math.floor(Math.log10(span))) / 2;
    let t = Math.ceil(xMin / step) * step;
    while (t <= xMax) {
      ticks.push(t);
      t += step;
    }
    return ticks;
  }, [xMin, xMax]);

  return (
    <div>
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium" htmlFor="markov-threshold">
            Threshold a
          </label>
          <input
            id="markov-threshold"
            type="range"
            min={sliderRange.min}
            max={sliderRange.max}
            step={sliderRange.step}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-36"
            aria-label={`Threshold a = ${threshold}`}
          />
          <span className="w-14 text-sm font-mono">{threshold.toFixed(preset.type === 'discrete' ? 0 : 1)}</span>
        </div>
      </div>

      {/* SVG */}
      <svg width={svgW} height={SVG_HEIGHT} className="block" role="img" aria-label="Markov inequality visualization">
        {/* Axes */}
        <line x1={MARGIN.left} y1={yScale(0)} x2={svgW - MARGIN.right} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />

        {/* X-axis ticks */}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={xScale(t)} y1={yScale(0)} x2={xScale(t)} y2={yScale(0) + 5} stroke="currentColor" strokeOpacity={0.3} />
            <text x={xScale(t)} y={yScale(0) + 18} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }}>
              {Number.isInteger(t) ? t : t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Continuous: shaded tail */}
        {preset.type === 'continuous' && shadedPath && (
          <path d={shadedPath} fill="var(--color-danger, #DC2626)" opacity={0.3} />
        )}

        {/* Continuous: PDF curve */}
        {preset.type === 'continuous' && curvePath && (
          <path d={curvePath} fill="none" stroke={distributionColors.pdf} strokeWidth={2} />
        )}

        {/* Discrete: bars */}
        {preset.type === 'discrete' && discreteData.map((d) => {
          const inTail = d.k >= threshold;
          return (
            <rect
              key={d.k}
              x={xScale(d.k) - barWidth / 2}
              y={yScale(d.p)}
              width={barWidth}
              height={Math.max(yScale(0) - yScale(d.p), 0)}
              fill={inTail ? 'var(--color-danger, #DC2626)' : distributionColors.pdf}
              opacity={inTail ? 0.7 : 0.5}
              stroke={inTail ? 'var(--color-danger, #DC2626)' : distributionColors.pdf}
              strokeWidth={1}
            />
          );
        })}

        {/* Discrete: x-axis labels */}
        {preset.type === 'discrete' && discreteData.filter((_, i) => i % Math.max(1, Math.floor(discreteData.length / 12)) === 0).map((d) => (
          <text
            key={d.k}
            x={xScale(d.k)}
            y={yScale(0) + 16}
            textAnchor="middle"
            className="fill-current"
            style={{ fontSize: '9px' }}
          >
            {d.k}
          </text>
        ))}

        {/* Threshold line */}
        <line
          x1={xScale(threshold)}
          y1={MARGIN.top}
          x2={xScale(threshold)}
          y2={yScale(0)}
          stroke="var(--color-danger, #DC2626)"
          strokeWidth={2}
          strokeDasharray="6,3"
        />
        <text
          x={xScale(threshold)}
          y={MARGIN.top - 4}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '11px', fontWeight: 600 }}
        >
          a = {threshold.toFixed(preset.type === 'discrete' ? 0 : 1)}
        </text>

        {/* Mean marker */}
        <line
          x1={xScale(mean)}
          y1={yScale(0) - 8}
          x2={xScale(mean)}
          y2={yScale(0) + 8}
          stroke={distributionColors.cdf}
          strokeWidth={2.5}
        />
        <text
          x={xScale(mean)}
          y={yScale(0) + 26}
          textAnchor="middle"
          style={{ fontSize: '10px', fill: distributionColors.cdf, fontWeight: 500 }}
        >
          E[X]={mean.toFixed(2)}
        </text>

        {/* Y-axis label */}
        <text
          x={14}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '11px' }}
          transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
        >
          {preset.type === 'discrete' ? 'P(X = k)' : 'f(x)'}
        </text>
      </svg>

      {/* Comparison bars */}
      <div className="mt-3">
        <ComparisonBars
          trueProb={trueProb}
          bound={markov}
          label={`True P(X \u2265 ${threshold.toFixed(preset.type === 'discrete' ? 0 : 1)})`}
          boundLabel="Markov: E[X]/a"
          boundColor="var(--color-warning, #D97706)"
          width={svgW}
        />
      </div>

      {/* Formula readout */}
      <div className="mt-2 rounded border px-3 py-2 text-sm font-mono" style={{ borderColor: 'var(--color-border)' }}>
        <div>E[X] = {mean.toFixed(4)}</div>
        <div>Markov: P(X {'\u2265'} {threshold.toFixed(preset.type === 'discrete' ? 0 : 1)}) {'\u2264'} E[X]/a = {mean.toFixed(4)} / {threshold.toFixed(preset.type === 'discrete' ? 0 : 1)} = <strong>{markov.toFixed(4)}</strong></div>
        <div>True:   P(X {'\u2265'} {threshold.toFixed(preset.type === 'discrete' ? 0 : 1)}) = <strong>{trueProb.toFixed(4)}</strong></div>
      </div>
    </div>
  );
}

// ── Chebyshev tab ───────────────────────────────────────────────────────────

function ChebyshevPanel({
  preset,
  svgW,
}: {
  preset: InequalityPreset;
  svgW: number;
}) {
  const [k, setK] = useState(preset.defaultChebyshevK);
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;
  const [xMin, xMax] = getXRange(preset);

  // Compute mean and variance
  const { mean, variance } = useMemo(() => {
    if (preset.type === 'discrete') {
      const dist = toDiscreteDistribution(preset);
      return { mean: expectationDiscrete(dist), variance: varianceDiscrete(dist) };
    }
    const dist = toContinuousDistribution(preset);
    return { mean: expectationContinuous(dist), variance: varianceContinuous(dist) };
  }, [preset]);

  const sigma = stdDev(variance);
  const epsilon = k * sigma;
  const chebyshev = chebyshevBound(variance, epsilon);
  const trueProb = useMemo(
    () => twoTailProbability(preset, mean, epsilon),
    [preset, mean, epsilon],
  );

  // Scales
  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const yScaleMax = useMemo(() => {
    if (preset.type === 'discrete') {
      const range = getDiscreteRange(preset);
      return Math.max(...range.map((v) => evalPMF(preset, v)), 0.01);
    }
    let maxY = 0;
    const step = (xMax - xMin) / N_CURVE_POINTS;
    for (let i = 0; i <= N_CURVE_POINTS; i++) {
      maxY = Math.max(maxY, evalPDF(preset, xMin + i * step));
    }
    return maxY || 0.01;
  }, [preset, xMin, xMax]);

  const yScale = useCallback(
    (val: number) => MARGIN.top + plotH - (val / (yScaleMax * 1.15)) * plotH,
    [plotH, yScaleMax],
  );

  // Build PDF path + two shaded tails
  const { curvePath, leftShadedPath, rightShadedPath } = useMemo(() => {
    if (preset.type === 'discrete') return { curvePath: '', leftShadedPath: '', rightShadedPath: '' };
    const step = (xMax - xMin) / N_CURVE_POINTS;
    const points = Array.from({ length: N_CURVE_POINTS + 1 }, (_, i) => {
      const x = xMin + i * step;
      return { x, y: evalPDF(preset, x) };
    });
    const curve = points
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');

    const lowerBound = mean - epsilon;
    const upperBound = mean + epsilon;
    const baseline = yScale(0);

    // Left tail
    const leftPts = points.filter((d) => d.x <= lowerBound);
    let leftPath = '';
    if (leftPts.length >= 2) {
      leftPath = `M${xScale(leftPts[0].x).toFixed(1)},${baseline}`;
      for (const d of leftPts) {
        leftPath += ` L${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`;
      }
      leftPath += ` L${xScale(leftPts[leftPts.length - 1].x).toFixed(1)},${baseline} Z`;
    }

    // Right tail
    const rightPts = points.filter((d) => d.x >= upperBound);
    let rightPath = '';
    if (rightPts.length >= 2) {
      rightPath = `M${xScale(rightPts[0].x).toFixed(1)},${baseline}`;
      for (const d of rightPts) {
        rightPath += ` L${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`;
      }
      rightPath += ` L${xScale(rightPts[rightPts.length - 1].x).toFixed(1)},${baseline} Z`;
    }

    return { curvePath: curve, leftShadedPath: leftPath, rightShadedPath: rightPath };
  }, [preset, xMin, xMax, mean, epsilon, xScale, yScale]);

  // Discrete data
  const discreteData = useMemo(() => {
    if (preset.type !== 'discrete') return [];
    return getDiscreteRange(preset).map((v) => ({ k: v, p: evalPMF(preset, v) }));
  }, [preset]);

  const barWidth = useMemo(() => {
    if (discreteData.length <= 1) return plotW * 0.6;
    const range = discreteData[discreteData.length - 1].k - discreteData[0].k || 1;
    return Math.max(4, Math.min(24, (plotW / (range + 1)) * 0.65));
  }, [discreteData, plotW]);

  // X-axis ticks
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const span = xMax - xMin;
    const step = Math.pow(10, Math.floor(Math.log10(span))) / 2;
    let t = Math.ceil(xMin / step) * step;
    while (t <= xMax) {
      ticks.push(t);
      t += step;
    }
    return ticks;
  }, [xMin, xMax]);

  return (
    <div>
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium" htmlFor="chebyshev-k">
            k (std devs)
          </label>
          <input
            id="chebyshev-k"
            type="range"
            min={0.5}
            max={5}
            step={0.1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-36"
            aria-label={`k = ${k.toFixed(1)} standard deviations`}
          />
          <span className="w-14 text-sm font-mono">{k.toFixed(1)}</span>
        </div>
      </div>

      {/* SVG */}
      <svg width={svgW} height={SVG_HEIGHT} className="block" role="img" aria-label="Chebyshev inequality visualization">
        {/* Axes */}
        <line x1={MARGIN.left} y1={yScale(0)} x2={svgW - MARGIN.right} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />

        {/* X-axis ticks */}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={xScale(t)} y1={yScale(0)} x2={xScale(t)} y2={yScale(0) + 5} stroke="currentColor" strokeOpacity={0.3} />
            <text x={xScale(t)} y={yScale(0) + 18} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }}>
              {Number.isInteger(t) ? t : t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Continuous: shaded tails */}
        {preset.type === 'continuous' && leftShadedPath && (
          <path d={leftShadedPath} fill="var(--color-warning, #D97706)" opacity={0.3} />
        )}
        {preset.type === 'continuous' && rightShadedPath && (
          <path d={rightShadedPath} fill="var(--color-warning, #D97706)" opacity={0.3} />
        )}

        {/* Continuous: PDF curve */}
        {preset.type === 'continuous' && curvePath && (
          <path d={curvePath} fill="none" stroke={distributionColors.pdf} strokeWidth={2} />
        )}

        {/* Discrete: bars */}
        {preset.type === 'discrete' && discreteData.map((d) => {
          const inTail = Math.abs(d.k - mean) >= epsilon;
          return (
            <rect
              key={d.k}
              x={xScale(d.k) - barWidth / 2}
              y={yScale(d.p)}
              width={barWidth}
              height={Math.max(yScale(0) - yScale(d.p), 0)}
              fill={inTail ? 'var(--color-warning, #D97706)' : distributionColors.pdf}
              opacity={inTail ? 0.7 : 0.5}
              stroke={inTail ? 'var(--color-warning, #D97706)' : distributionColors.pdf}
              strokeWidth={1}
            />
          );
        })}

        {/* Discrete: x-axis labels */}
        {preset.type === 'discrete' && discreteData.filter((_, i) => i % Math.max(1, Math.floor(discreteData.length / 12)) === 0).map((d) => (
          <text
            key={d.k}
            x={xScale(d.k)}
            y={yScale(0) + 16}
            textAnchor="middle"
            className="fill-current"
            style={{ fontSize: '9px' }}
          >
            {d.k}
          </text>
        ))}

        {/* Boundary lines: mu - k*sigma and mu + k*sigma */}
        {[mean - epsilon, mean + epsilon].map((bound, i) => (
          <g key={i}>
            <line
              x1={xScale(bound)}
              y1={MARGIN.top}
              x2={xScale(bound)}
              y2={yScale(0)}
              stroke="var(--color-warning, #D97706)"
              strokeWidth={2}
              strokeDasharray="6,3"
            />
            <text
              x={xScale(bound)}
              y={MARGIN.top - 4}
              textAnchor="middle"
              className="fill-current"
              style={{ fontSize: '10px', fontWeight: 500 }}
            >
              {i === 0 ? `\u03BC\u2212${k.toFixed(1)}\u03C3` : `\u03BC+${k.toFixed(1)}\u03C3`}
            </text>
          </g>
        ))}

        {/* Mean marker */}
        <line
          x1={xScale(mean)}
          y1={yScale(0) - 8}
          x2={xScale(mean)}
          y2={yScale(0) + 8}
          stroke={distributionColors.cdf}
          strokeWidth={2.5}
        />
        <text
          x={xScale(mean)}
          y={yScale(0) + 26}
          textAnchor="middle"
          style={{ fontSize: '10px', fill: distributionColors.cdf, fontWeight: 500 }}
        >
          {'\u03BC'}={mean.toFixed(2)}
        </text>

        {/* Y-axis label */}
        <text
          x={14}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '11px' }}
          transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
        >
          {preset.type === 'discrete' ? 'P(X = k)' : 'f(x)'}
        </text>
      </svg>

      {/* Comparison bars */}
      <div className="mt-3">
        <ComparisonBars
          trueProb={trueProb}
          bound={chebyshev}
          label={`True P(|X\u2212\u03BC| \u2265 ${k.toFixed(1)}\u03C3)`}
          boundLabel={`Chebyshev: 1/k\u00B2`}
          boundColor="var(--color-warning, #D97706)"
          width={svgW}
        />
      </div>

      {/* Formula readout */}
      <div className="mt-2 rounded border px-3 py-2 text-sm font-mono" style={{ borderColor: 'var(--color-border)' }}>
        <div>{'\u03BC'} = {mean.toFixed(4)}, {'\u03C3'}{'\u00B2'} = {variance.toFixed(4)}, {'\u03C3'} = {sigma.toFixed(4)}</div>
        <div>Chebyshev: P(|X {'\u2212'} {'\u03BC'}| {'\u2265'} {k.toFixed(1)}{'\u03C3'}) {'\u2264'} 1/k{'\u00B2'} = 1/{(k * k).toFixed(2)} = <strong>{chebyshev.toFixed(4)}</strong></div>
        <div>True:      P(|X {'\u2212'} {'\u03BC'}| {'\u2265'} {k.toFixed(1)}{'\u03C3'}) = <strong>{trueProb.toFixed(4)}</strong></div>
      </div>
    </div>
  );
}

// ── Jensen tab ──────────────────────────────────────────────────────────────

function JensenPanel({
  svgW,
}: {
  svgW: number;
}) {
  const [presetIdx, setPresetIdx] = useState(0);
  const [mu, setMu] = useState(2);
  const [sigma2, setSigma2] = useState(1);
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const jensenPreset: JensenPreset = jensenPresets[presetIdx];
  const g = jensenPreset.g;
  const gPrime = jensenPreset.gPrime;

  // Determine x-range. For -log(x) we keep x > 0.
  const sd = Math.sqrt(sigma2);
  const isLog = presetIdx === 2; // -log(x)
  const xMin = isLog ? Math.max(0.01, mu - 3.5 * sd) : mu - 4 * sd;
  const xMax = mu + 4 * sd;

  // g(E[X]) and E[g(X)]
  const gOfMean = g(mu);
  const eOfG = useMemo(() => jensenEgX(g, mu, sigma2), [g, mu, sigma2]);
  const jensenGap = eOfG - gOfMean;

  // g(x) curve data
  const gData = useMemo(() => {
    const step = (xMax - xMin) / N_CURVE_POINTS;
    return Array.from({ length: N_CURVE_POINTS + 1 }, (_, i) => {
      const x = xMin + i * step;
      return { x, y: g(x) };
    });
  }, [g, xMin, xMax]);

  // Y range from g data
  const yMin = useMemo(() => Math.min(...gData.map((d) => d.y)), [gData]);
  const yMax = useMemo(() => Math.max(...gData.map((d) => d.y)), [gData]);
  const yPad = (yMax - yMin) * 0.1 || 1;

  // Scales
  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const yScale = useCallback(
    (val: number) => MARGIN.top + plotH - ((val - (yMin - yPad)) / (yMax - yMin + 2 * yPad)) * plotH,
    [plotH, yMin, yMax, yPad],
  );

  // Tangent line at x = mu
  const tangentSlope = gPrime(mu);
  const tangentFn = (x: number) => gOfMean + tangentSlope * (x - mu);

  // g(x) curve path
  const gPath = useMemo(() => {
    return gData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');
  }, [gData, xScale, yScale]);

  // Tangent line path
  const tangentPath = useMemo(() => {
    const x0 = xMin;
    const x1 = xMax;
    const y0 = tangentFn(x0);
    const y1 = tangentFn(x1);
    return `M${xScale(x0).toFixed(1)},${yScale(y0).toFixed(1)} L${xScale(x1).toFixed(1)},${yScale(y1).toFixed(1)}`;
  }, [xMin, xMax, tangentFn, xScale, yScale]);

  // X-axis ticks
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const span = xMax - xMin;
    const step = Math.pow(10, Math.floor(Math.log10(span))) / 2;
    let t = Math.ceil(xMin / step) * step;
    while (t <= xMax) {
      ticks.push(t);
      t += step;
    }
    return ticks;
  }, [xMin, xMax]);

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const span = yMax - yMin + 2 * yPad;
    const step = Math.pow(10, Math.floor(Math.log10(span))) / 2;
    const lo = yMin - yPad;
    const hi = yMax + yPad;
    let t = Math.ceil(lo / step) * step;
    while (t <= hi) {
      ticks.push(t);
      t += step;
    }
    return ticks;
  }, [yMin, yMax, yPad]);

  return (
    <div>
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium" htmlFor="jensen-fn">
            g(x)
          </label>
          <select
            id="jensen-fn"
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {jensenPresets.map((p, i) => (
              <option key={p.name} value={i}>{p.name}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium" htmlFor="jensen-mu">{'\u03BC'}</label>
          <input
            id="jensen-mu"
            type="range"
            min={isLog ? 0.5 : -3}
            max={5}
            step={0.1}
            value={mu}
            onChange={(e) => setMu(Number(e.target.value))}
            className="w-24"
            aria-label={`Mean mu = ${mu}`}
          />
          <span className="w-12 text-sm font-mono">{mu.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium" htmlFor="jensen-sigma2">{'\u03C3'}{'\u00B2'}</label>
          <input
            id="jensen-sigma2"
            type="range"
            min={0.1}
            max={4}
            step={0.1}
            value={sigma2}
            onChange={(e) => setSigma2(Number(e.target.value))}
            className="w-24"
            aria-label={`Variance sigma squared = ${sigma2}`}
          />
          <span className="w-12 text-sm font-mono">{sigma2.toFixed(1)}</span>
        </div>
      </div>

      {/* SVG */}
      <svg width={svgW} height={SVG_HEIGHT} className="block" role="img" aria-label="Jensen inequality visualization">
        {/* Axes */}
        <line x1={MARGIN.left} y1={yScale(0)} x2={svgW - MARGIN.right} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.15} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />

        {/* X-axis ticks */}
        {xTicks.map((t) => (
          <g key={`x-${t}`}>
            <line
              x1={xScale(t)}
              y1={MARGIN.top + plotH}
              x2={xScale(t)}
              y2={MARGIN.top + plotH + 5}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <text
              x={xScale(t)}
              y={MARGIN.top + plotH + 18}
              textAnchor="middle"
              className="fill-current"
              style={{ fontSize: '10px' }}
            >
              {Number.isInteger(t) ? t : t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Y-axis ticks */}
        {yTicks.map((t) => (
          <g key={`y-${t}`}>
            <line
              x1={MARGIN.left - 4}
              y1={yScale(t)}
              x2={MARGIN.left}
              y2={yScale(t)}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <text
              x={MARGIN.left - 8}
              y={yScale(t) + 4}
              textAnchor="end"
              className="fill-current"
              style={{ fontSize: '9px' }}
            >
              {Math.abs(t) < 100 ? t.toFixed(1) : t.toFixed(0)}
            </text>
          </g>
        ))}

        {/* Tangent line (drawn first, behind the curve) */}
        <path
          d={tangentPath}
          fill="none"
          stroke="var(--color-warning, #D97706)"
          strokeWidth={1.5}
          strokeDasharray="6,4"
          opacity={0.7}
        />

        {/* g(x) curve */}
        <path d={gPath} fill="none" stroke={distributionColors.pdf} strokeWidth={2.5} />

        {/* Vertical line at E[X] = mu */}
        <line
          x1={xScale(mu)}
          y1={MARGIN.top}
          x2={xScale(mu)}
          y2={MARGIN.top + plotH}
          stroke={distributionColors.cdf}
          strokeWidth={1.5}
          strokeDasharray="4,3"
          opacity={0.5}
        />
        <text
          x={xScale(mu) + 4}
          y={MARGIN.top + 12}
          className="fill-current"
          style={{ fontSize: '10px', fontWeight: 500, fill: distributionColors.cdf }}
        >
          E[X]={mu.toFixed(1)}
        </text>

        {/* Horizontal dotted line for g(E[X]) */}
        <line
          x1={MARGIN.left}
          y1={yScale(gOfMean)}
          x2={xScale(mu)}
          y2={yScale(gOfMean)}
          stroke={distributionColors.cdf}
          strokeWidth={1.5}
          strokeDasharray="3,3"
          opacity={0.6}
        />
        {/* g(E[X]) label on y-axis */}
        <text
          x={MARGIN.left - 4}
          y={yScale(gOfMean) - 6}
          textAnchor="end"
          style={{ fontSize: '10px', fill: distributionColors.cdf, fontWeight: 500 }}
        >
          g(E[X])
        </text>

        {/* Horizontal dotted line for E[g(X)] */}
        <line
          x1={MARGIN.left}
          y1={yScale(eOfG)}
          x2={xScale(mu)}
          y2={yScale(eOfG)}
          stroke="var(--color-danger, #DC2626)"
          strokeWidth={1.5}
          strokeDasharray="3,3"
          opacity={0.6}
        />
        {/* E[g(X)] label on y-axis */}
        <text
          x={MARGIN.left - 4}
          y={yScale(eOfG) - 6}
          textAnchor="end"
          style={{ fontSize: '10px', fill: 'var(--color-danger, #DC2626)', fontWeight: 500 }}
        >
          E[g(X)]
        </text>

        {/* Dot at (mu, g(mu)) = g(E[X]) */}
        <circle
          cx={xScale(mu)}
          cy={yScale(gOfMean)}
          r={5}
          fill={distributionColors.cdf}
          stroke="white"
          strokeWidth={1.5}
        />

        {/* Dot at (mu, E[g(X)]) for the Jensen gap arrow */}
        <circle
          cx={xScale(mu)}
          cy={yScale(eOfG)}
          r={5}
          fill="var(--color-danger, #DC2626)"
          stroke="white"
          strokeWidth={1.5}
        />

        {/* Jensen gap arrow: vertical arrow from g(E[X]) up to E[g(X)] */}
        {jensenGap > 0.01 && (
          <g>
            {/* Arrow shaft */}
            <line
              x1={xScale(mu) + 16}
              y1={yScale(gOfMean)}
              x2={xScale(mu) + 16}
              y2={yScale(eOfG)}
              stroke="var(--color-danger, #DC2626)"
              strokeWidth={2}
              markerEnd="url(#arrowhead-jensen)"
            />
            {/* Arrow label */}
            <text
              x={xScale(mu) + 22}
              y={(yScale(gOfMean) + yScale(eOfG)) / 2 + 4}
              style={{ fontSize: '10px', fill: 'var(--color-danger, #DC2626)', fontWeight: 600 }}
            >
              gap = {jensenGap.toFixed(3)}
            </text>
          </g>
        )}

        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead-jensen"
            markerWidth="8"
            markerHeight="6"
            refX="4"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 8 3, 0 6" fill="var(--color-danger, #DC2626)" />
          </marker>
        </defs>

        {/* Y-axis label */}
        <text
          x={14}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '11px' }}
          transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
        >
          g(x)
        </text>

        {/* Legend */}
        <g transform={`translate(${svgW - MARGIN.right - 130}, ${MARGIN.top + 4})`}>
          <line x1={0} y1={0} x2={16} y2={0} stroke={distributionColors.pdf} strokeWidth={2.5} />
          <text x={20} y={4} className="fill-current" style={{ fontSize: '10px' }}>
            {jensenPreset.name}
          </text>
          <line x1={0} y1={16} x2={16} y2={16} stroke="var(--color-warning, #D97706)" strokeWidth={1.5} strokeDasharray="6,4" />
          <text x={20} y={20} className="fill-current" style={{ fontSize: '10px' }}>
            Tangent at E[X]
          </text>
        </g>
      </svg>

      {/* Numerical readout */}
      <div className="mt-3 rounded border px-3 py-2 text-sm font-mono" style={{ borderColor: 'var(--color-border)' }}>
        <div>X ~ N({'\u03BC'}={mu.toFixed(1)}, {'\u03C3'}{'\u00B2'}={sigma2.toFixed(1)})</div>
        <div>g(E[X]) = g({mu.toFixed(1)}) = <span style={{ color: distributionColors.cdf, fontWeight: 600 }}>{gOfMean.toFixed(4)}</span></div>
        <div>E[g(X)] = <span style={{ color: 'var(--color-danger, #DC2626)', fontWeight: 600 }}>{eOfG.toFixed(4)}</span></div>
        <div>
          Jensen gap: E[g(X)] {'\u2212'} g(E[X]) ={' '}
          <strong>{jensenGap.toFixed(4)}</strong>{' '}
          {jensenGap >= -1e-6
            ? <span style={{ color: distributionColors.cdf }}>{'\u2265'} 0 (convex)</span>
            : <span style={{ color: 'var(--color-danger, #DC2626)' }}>&lt; 0 (concave)</span>}
        </div>
      </div>

      {/* Pedagogical note */}
      <div className="mt-2 text-sm" style={{ opacity: 0.7 }}>
        The tangent line at E[X] always lies below the curve for convex g. This geometric fact
        is why E[g(X)] {'\u2265'} g(E[X]) -- the curve bends upward from the tangent, and
        averaging over a spread-out distribution only increases the function value.
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function InequalityExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>('markov');
  const [presetIdx, setPresetIdx] = useState(0);

  const svgW = Math.max(width, 320);
  const preset = inequalityPresets[presetIdx];

  const handlePresetChange = (idx: number) => {
    setPresetIdx(idx);
  };

  const tabs: { id: Mode; label: string }[] = [
    { id: 'markov', label: 'Markov' },
    { id: 'chebyshev', label: 'Chebyshev' },
    { id: 'jensen', label: 'Jensen' },
  ];

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Tab bar */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div
          className="flex rounded-lg border"
          style={{ borderColor: 'var(--color-border)' }}
          role="tablist"
          aria-label="Inequality type"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={mode === tab.id}
              aria-controls={`panel-${tab.id}`}
              onClick={() => setMode(tab.id)}
              className={
                mode === tab.id
                  ? 'px-4 py-1.5 text-sm font-medium bg-blue-600 text-white'
                  : 'px-4 py-1.5 text-sm font-medium bg-transparent'
              }
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Distribution dropdown (Markov/Chebyshev modes only) */}
        {mode !== 'jensen' && (
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium" htmlFor="ineq-dist-select">
              Distribution
            </label>
            <select
              id="ineq-dist-select"
              value={presetIdx}
              onChange={(e) => handlePresetChange(Number(e.target.value))}
              className="rounded border px-2 py-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
            >
              {inequalityPresets.map((p, i) => (
                <option key={p.name} value={i}>{p.name}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Panels */}
      <div id="panel-markov" role="tabpanel" hidden={mode !== 'markov'}>
        {mode === 'markov' && <MarkovPanel preset={preset} svgW={svgW} />}
      </div>
      <div id="panel-chebyshev" role="tabpanel" hidden={mode !== 'chebyshev'}>
        {mode === 'chebyshev' && <ChebyshevPanel preset={preset} svgW={svgW} />}
      </div>
      <div id="panel-jensen" role="tabpanel" hidden={mode !== 'jensen'}>
        {mode === 'jensen' && <JensenPanel svgW={svgW} />}
      </div>
    </div>
  );
}
