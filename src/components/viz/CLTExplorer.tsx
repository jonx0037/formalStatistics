import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cltReplications,
  normalSample,
  uniformSample,
  bernoulliSample,
  exponentialSample,
  poissonSample,
  gammaSample,
  betaSample,
  chiSquaredSample,
  tSample,
} from './shared/convergence';
import {
  pdfStdNormal,
  cdfStdNormal,
  quantileStdNormal,
} from './shared/distributions';
import {
  cltDistributions,
  type CLTDistribution,
} from '../../data/central-limit-theorem-data';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 18, bottom: 38, left: 48 };
const CHART_H = 260;
const HIST_BINS = 40;

const HIST_COLOR = '#7C3AED'; // violet (matches Track 3 domain)
const NORMAL_COLOR = '#059669'; // emerald
const HIST_COMPARE = '#D97706'; // amber (for compare mode)
const QQ_LINE = '#DC2626'; // red (reference line)

const N_SNAPS = [1, 2, 5, 10, 20, 50, 100, 200];

// ── Sampler Registry ────────────────────────────────────────────────────────

function makeSampler(id: string): () => number {
  switch (id) {
    case 'uniform':
      return () => uniformSample(0, 1);
    case 'exponential':
      return () => exponentialSample(1);
    case 'bernoulli':
      return () => bernoulliSample(0.3);
    case 'poisson':
      return () => poissonSample(5);
    case 'beta':
      return () => betaSample(2, 5);
    case 'chi-squared':
      return () => chiSquaredSample(3);
    case 't5':
      return () => tSample(5);
    case 'gamma':
      return () => gammaSample(2, 1);
    case 'dice':
      return () => 1 + Math.floor(Math.random() * 6);
    default:
      return () => normalSample(0, 1);
  }
}

// ── Statistics Helpers ──────────────────────────────────────────────────────

function sampleSkewness(values: number[]): number {
  const n = values.length;
  if (n < 3) return 0;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  let m2 = 0;
  let m3 = 0;
  for (const v of values) {
    const d = v - mean;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n;
  m3 /= n;
  return m3 / Math.pow(m2, 1.5);
}

function sampleExcessKurtosis(values: number[]): number {
  const n = values.length;
  if (n < 4) return 0;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  let m2 = 0;
  let m4 = 0;
  for (const v of values) {
    const d = v - mean;
    const d2 = d * d;
    m2 += d2;
    m4 += d2 * d2;
  }
  m2 /= n;
  m4 /= n;
  return m4 / (m2 * m2) - 3;
}

/** KS distance against N(0,1). */
function ksAgainstStdNormal(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  let maxDiff = 0;
  for (let i = 0; i < n; i++) {
    const Fx = cdfStdNormal(sorted[i]);
    const ecdfR = (i + 1) / n;
    const ecdfL = i / n;
    const d = Math.max(Math.abs(ecdfR - Fx), Math.abs(ecdfL - Fx));
    if (d > maxDiff) maxDiff = d;
  }
  return maxDiff;
}

// ── Component ───────────────────────────────────────────────────────────────

function CLTExplorer() {
  const [distAId, setDistAId] = useState<string>('exponential');
  const [distBId, setDistBId] = useState<string>('uniform');
  const [n, setN] = useState<number>(30);
  const [M, setM] = useState<number>(5000);
  const [compare, setCompare] = useState<boolean>(false);
  const [animating, setAnimating] = useState<boolean>(false);
  const animTimer = useRef<number | null>(null);

  const distA = useMemo<CLTDistribution>(
    () => cltDistributions.find((d) => d.id === distAId) ?? cltDistributions[0],
    [distAId],
  );
  const distB = useMemo<CLTDistribution>(
    () => cltDistributions.find((d) => d.id === distBId) ?? cltDistributions[0],
    [distBId],
  );

  // Generate standardized replications for distribution A (always) and B (when comparing).
  const repsA = useMemo(() => {
    const sampler = makeSampler(distA.id);
    return cltReplications(sampler, n, M, distA.mu, distA.sigma);
  }, [distA, n, M]);
  const repsB = useMemo(() => {
    if (!compare) return null;
    const sampler = makeSampler(distB.id);
    return cltReplications(sampler, n, M, distB.mu, distB.sigma);
  }, [distB, n, M, compare]);

  const statsA = useMemo(
    () => ({
      ks: ksAgainstStdNormal(repsA),
      skew: sampleSkewness(repsA),
      kurt: sampleExcessKurtosis(repsA),
    }),
    [repsA],
  );
  const statsB = useMemo(
    () =>
      repsB
        ? {
            ks: ksAgainstStdNormal(repsB),
            skew: sampleSkewness(repsB),
            kurt: sampleExcessKurtosis(repsB),
          }
        : null,
    [repsB],
  );

  // Animate: step n through N_SNAPS.
  useEffect(() => {
    if (!animating) return;
    let idx = N_SNAPS.indexOf(n);
    if (idx < 0) idx = 0;
    animTimer.current = window.setInterval(() => {
      idx = (idx + 1) % N_SNAPS.length;
      setN(N_SNAPS[idx]);
    }, 900);
    return () => {
      if (animTimer.current !== null) window.clearInterval(animTimer.current);
    };
  }, [animating, n]);

  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(260, width);

  // Stack panels vertically on small widths; show three columns when wide.
  const threeCol = width >= 900;
  const panelW = threeCol ? (w - 24) / 3 : w;

  return (
    <div
      className="my-8 rounded-lg border p-5"
      style={{
        borderColor: 'var(--color-border)',
        background: 'var(--color-bg-subtle, transparent)',
      }}
    >
      <Controls
        distAId={distAId}
        setDistAId={setDistAId}
        distBId={distBId}
        setDistBId={setDistBId}
        n={n}
        setN={setN}
        M={M}
        setM={setM}
        compare={compare}
        setCompare={setCompare}
        animating={animating}
        setAnimating={setAnimating}
      />

      <div
        ref={containerRef}
        className={
          threeCol ? 'mt-4 grid grid-cols-3 gap-3' : 'mt-4 flex flex-col gap-4'
        }
      >
        <HistogramPanel
          width={panelW}
          reps={repsA}
          distA={distA}
          repsB={repsB}
          distB={compare ? distB : null}
        />
        <CDFPanel
          width={panelW}
          reps={repsA}
          distA={distA}
          repsB={repsB}
          distB={compare ? distB : null}
        />
        <QQPanel
          width={panelW}
          reps={repsA}
          distA={distA}
          repsB={repsB}
          distB={compare ? distB : null}
        />
      </div>

      <Readout
        distA={distA}
        distB={compare ? distB : null}
        n={n}
        M={M}
        statsA={statsA}
        statsB={statsB}
      />
    </div>
  );
}

// ── Controls ───────────────────────────────────────────────────────────────

function Controls(props: {
  distAId: string;
  setDistAId: (id: string) => void;
  distBId: string;
  setDistBId: (id: string) => void;
  n: number;
  setN: (n: number) => void;
  M: number;
  setM: (m: number) => void;
  compare: boolean;
  setCompare: (c: boolean) => void;
  animating: boolean;
  setAnimating: (a: boolean) => void;
}) {
  const {
    distAId,
    setDistAId,
    distBId,
    setDistBId,
    n,
    setN,
    M,
    setM,
    compare,
    setCompare,
    animating,
    setAnimating,
  } = props;

  const onNSlider = (idx: number) => setN(N_SNAPS[idx]);
  const nIdx = Math.max(0, N_SNAPS.indexOf(n));

  return (
    <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
      <label className="flex flex-col text-xs">
        <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
          Distribution
        </span>
        <select
          className="rounded border p-1 text-sm"
          style={{ borderColor: 'var(--color-border)' }}
          value={distAId}
          onChange={(e) => setDistAId(e.target.value)}
        >
          {cltDistributions.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col text-xs">
        <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
          Sample size n = {n}
        </span>
        <input
          type="range"
          min={0}
          max={N_SNAPS.length - 1}
          step={1}
          value={nIdx}
          onChange={(e) => onNSlider(parseInt(e.target.value, 10))}
        />
      </label>

      <label className="flex flex-col text-xs">
        <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
          Replications M = {M.toLocaleString()}
        </span>
        <input
          type="range"
          min={500}
          max={10000}
          step={500}
          value={M}
          onChange={(e) => setM(parseInt(e.target.value, 10))}
        />
      </label>

      <div className="flex flex-col gap-2">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={compare}
            onChange={(e) => setCompare(e.target.checked)}
          />
          <span className="font-medium uppercase tracking-wide opacity-70">
            Compare two distributions
          </span>
        </label>
        {compare && (
          <select
            className="rounded border p-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            value={distBId}
            onChange={(e) => setDistBId(e.target.value)}
          >
            {cltDistributions.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        )}
        <button
          type="button"
          onClick={() => setAnimating(!animating)}
          className="rounded border px-3 py-1 text-xs font-medium"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {animating ? '■ Stop' : '▶ Animate n'}
        </button>
      </div>
    </div>
  );
}

// ── Histogram Panel ────────────────────────────────────────────────────────

function HistogramPanel(props: {
  width: number;
  reps: number[];
  distA: CLTDistribution;
  repsB: number[] | null;
  distB: CLTDistribution | null;
}) {
  const { width, reps, distA, repsB, distB } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Domain: clip to [-4, 4] for standardized variables
    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
    const xBins = d3
      .bin<number, number>()
      .domain([-4, 4])
      .thresholds(HIST_BINS)(reps.filter((v) => v >= -4 && v <= 4));

    const density = (b: d3.Bin<number, number>) =>
      b.length / (reps.length * (b.x1! - b.x0!));

    const yMax = Math.max(
      d3.max(xBins, density) ?? 0.5,
      pdfStdNormal(0) * 1.05,
    );
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Distribution A histogram bars
    g.append('g')
      .attr('fill', HIST_COLOR)
      .attr('fill-opacity', 0.55)
      .selectAll('rect')
      .data(xBins)
      .join('rect')
      .attr('x', (b) => xScale(b.x0!))
      .attr('y', (b) => yScale(density(b)))
      .attr('width', (b) => Math.max(0, xScale(b.x1!) - xScale(b.x0!) - 1))
      .attr('height', (b) => innerH - yScale(density(b)));

    // Compare distribution B (outlined bars)
    if (repsB && distB) {
      const binsB = d3
        .bin<number, number>()
        .domain([-4, 4])
        .thresholds(HIST_BINS)(repsB.filter((v) => v >= -4 && v <= 4));
      g.append('g')
        .attr('fill', 'none')
        .attr('stroke', HIST_COMPARE)
        .attr('stroke-width', 1.5)
        .selectAll('path')
        .data(binsB)
        .join('path')
        .attr(
          'd',
          (b) =>
            `M${xScale(b.x0!)},${innerH} L${xScale(b.x0!)},${yScale(density(b))} L${xScale(b.x1!)},${yScale(density(b))} L${xScale(b.x1!)},${innerH}`,
        );
    }

    // N(0,1) reference curve
    const xs = d3.range(-4, 4.01, 0.05);
    const line = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(pdfStdNormal(x)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', NORMAL_COLOR)
      .attr('stroke-width', 2)
      .attr('d', line);

    // Axes
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('font-size', 10);

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('√n(X̄ₙ − μ)/σ');

    // Title
    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Histogram vs N(0,1)');
  }, [width, reps, repsB, distA, distB]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

// ── CDF Panel ──────────────────────────────────────────────────────────────

function CDFPanel(props: {
  width: number;
  reps: number[];
  distA: CLTDistribution;
  repsB: number[] | null;
  distB: CLTDistribution | null;
}) {
  const { width, reps, repsB } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Empirical CDF for A — subsample for rendering performance
    const sortedA = [...reps].sort((a, b) => a - b);
    const stepA = Math.max(1, Math.floor(sortedA.length / 400));
    const pointsA: Array<[number, number]> = [];
    for (let i = 0; i < sortedA.length; i += stepA) {
      pointsA.push([sortedA[i], (i + 1) / sortedA.length]);
    }
    const lineA = d3
      .line<[number, number]>()
      .x(([x]) => xScale(Math.max(-4, Math.min(4, x))))
      .y(([, y]) => yScale(y))
      .curve(d3.curveStepAfter);
    g.append('path')
      .datum(pointsA)
      .attr('fill', 'none')
      .attr('stroke', HIST_COLOR)
      .attr('stroke-width', 2)
      .attr('d', lineA);

    if (repsB) {
      const sortedB = [...repsB].sort((a, b) => a - b);
      const stepB = Math.max(1, Math.floor(sortedB.length / 400));
      const pointsB: Array<[number, number]> = [];
      for (let i = 0; i < sortedB.length; i += stepB) {
        pointsB.push([sortedB[i], (i + 1) / sortedB.length]);
      }
      g.append('path')
        .datum(pointsB)
        .attr('fill', 'none')
        .attr('stroke', HIST_COMPARE)
        .attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '4 2')
        .attr('d', lineA);
    }

    // Φ(x) reference
    const xs = d3.range(-4, 4.01, 0.05);
    const phi = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(cdfStdNormal(x)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', NORMAL_COLOR)
      .attr('stroke-width', 2)
      .attr('d', phi);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).attr('font-size', 10);

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('z');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Empirical CDF vs Φ');
  }, [width, reps, repsB]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

// ── QQ Panel ───────────────────────────────────────────────────────────────

function QQPanel(props: {
  width: number;
  reps: number[];
  distA: CLTDistribution;
  repsB: number[] | null;
  distB: CLTDistribution | null;
}) {
  const { width, reps, repsB } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Build quantile pairs for A. Use up to 200 points for clarity.
    const nPoints = Math.min(200, reps.length);
    const sortedA = [...reps].sort((a, b) => a - b);
    const stepA = sortedA.length / nPoints;
    const qpA: Array<[number, number]> = [];
    for (let i = 0; i < nPoints; i++) {
      const idx = Math.floor(i * stepA);
      const p = (i + 0.5) / nPoints;
      qpA.push([quantileStdNormal(p), sortedA[idx]]);
    }

    const xScale = d3.scaleLinear().domain([-3.5, 3.5]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([-3.5, 3.5]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // 45° reference line
    g.append('line')
      .attr('x1', xScale(-3.5))
      .attr('y1', yScale(-3.5))
      .attr('x2', xScale(3.5))
      .attr('y2', yScale(3.5))
      .attr('stroke', QQ_LINE)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3 3');

    // Points for A
    g.append('g')
      .selectAll('circle')
      .data(qpA)
      .join('circle')
      .attr('cx', ([x]) => xScale(Math.max(-3.5, Math.min(3.5, x))))
      .attr('cy', ([, y]) => yScale(Math.max(-3.5, Math.min(3.5, y))))
      .attr('r', 2)
      .attr('fill', HIST_COLOR)
      .attr('fill-opacity', 0.75);

    if (repsB) {
      const sortedB = [...repsB].sort((a, b) => a - b);
      const stepB = sortedB.length / nPoints;
      const qpB: Array<[number, number]> = [];
      for (let i = 0; i < nPoints; i++) {
        const idx = Math.floor(i * stepB);
        const p = (i + 0.5) / nPoints;
        qpB.push([quantileStdNormal(p), sortedB[idx]]);
      }
      g.append('g')
        .selectAll('circle')
        .data(qpB)
        .join('circle')
        .attr('cx', ([x]) => xScale(Math.max(-3.5, Math.min(3.5, x))))
        .attr('cy', ([, y]) => yScale(Math.max(-3.5, Math.min(3.5, y))))
        .attr('r', 2)
        .attr('fill', HIST_COMPARE)
        .attr('fill-opacity', 0.75);
    }

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).attr('font-size', 10);

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Theoretical N(0,1) quantile');

    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Empirical quantile');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('QQ plot vs N(0,1)');
  }, [width, reps, repsB]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

// ── Readout ────────────────────────────────────────────────────────────────

function Readout(props: {
  distA: CLTDistribution;
  distB: CLTDistribution | null;
  n: number;
  M: number;
  statsA: { ks: number; skew: number; kurt: number };
  statsB: { ks: number; skew: number; kurt: number } | null;
}) {
  const { distA, distB, n, M, statsA, statsB } = props;
  const fmt = (x: number, digits = 3) => x.toFixed(digits);
  return (
    <div className="mt-4 grid gap-3 text-xs sm:grid-cols-2">
      <div className="rounded border p-3" style={{ borderColor: HIST_COLOR }}>
        <div className="mb-1 font-semibold" style={{ color: HIST_COLOR }}>
          {distA.name}
        </div>
        <div className="opacity-70">{distA.description}</div>
        <div className="mt-2 grid grid-cols-3 gap-2">
          <Stat label="KS vs Φ" value={fmt(statsA.ks, 4)} />
          <Stat label="Sample skew" value={fmt(statsA.skew, 3)} />
          <Stat label="Excess kurtosis" value={fmt(statsA.kurt, 3)} />
        </div>
        <div className="mt-2 opacity-70">
          n = {n} · M = {M.toLocaleString()} · true skewness{' '}
          {distA.skewness.toFixed(2)} · ρ = {distA.thirdMomentRatio.toFixed(2)}
        </div>
      </div>
      {distB && statsB && (
        <div className="rounded border p-3" style={{ borderColor: HIST_COMPARE }}>
          <div className="mb-1 font-semibold" style={{ color: HIST_COMPARE }}>
            {distB.name}
          </div>
          <div className="opacity-70">{distB.description}</div>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <Stat label="KS vs Φ" value={fmt(statsB.ks, 4)} />
            <Stat label="Sample skew" value={fmt(statsB.skew, 3)} />
            <Stat label="Excess kurtosis" value={fmt(statsB.kurt, 3)} />
          </div>
          <div className="mt-2 opacity-70">
            true skewness {distB.skewness.toFixed(2)} · ρ ={' '}
            {distB.thirdMomentRatio.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide opacity-60">{label}</span>
      <span className="font-mono text-sm">{value}</span>
    </div>
  );
}

export default CLTExplorer;
