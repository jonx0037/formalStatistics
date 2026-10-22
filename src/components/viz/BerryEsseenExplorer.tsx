import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cltReplications,
  berryEsseenBound,
  normalSample,
  uniformSample,
  bernoulliSample,
  exponentialSample,
  chiSquaredSample,
} from './shared/convergence';
import { cdfStdNormal } from './shared/distributions';
import {
  berryEsseenPresets,
  type BerryEsseenPreset,
} from '../../data/central-limit-theorem-data';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 18, bottom: 38, left: 55 };
const CHART_H = 280;
const BOUND_COLOR = '#111827';

// n-grid used for the empirical sup-deviation curve (log spacing).
const N_GRID = [5, 10, 20, 40, 80, 150, 300, 500];

const M_REPS = 3000; // replications per n — keep moderate for responsiveness

// ── Sampler Registry ────────────────────────────────────────────────────────

function makeSamplerAndStats(
  id: string,
): { sampler: () => number; mu: number; sigma: number } {
  switch (id) {
    case 'uniform':
      return { sampler: () => uniformSample(0, 1), mu: 0.5, sigma: Math.sqrt(1 / 12) };
    case 'normal':
      return { sampler: () => normalSample(0, 1), mu: 0, sigma: 1 };
    case 'bernoulli-half':
      return { sampler: () => bernoulliSample(0.5), mu: 0.5, sigma: 0.5 };
    case 'exponential':
      return { sampler: () => exponentialSample(1), mu: 1, sigma: 1 };
    case 'chi-squared-1':
      return { sampler: () => chiSquaredSample(1), mu: 1, sigma: Math.sqrt(2) };
    case 'bernoulli-01':
      return {
        sampler: () => bernoulliSample(0.1),
        mu: 0.1,
        sigma: Math.sqrt(0.09),
      };
    default:
      return { sampler: () => normalSample(0, 1), mu: 0, sigma: 1 };
  }
}

function supDeviation(reps: number[]): number {
  const sorted = [...reps].sort((a, b) => a - b);
  const n = sorted.length;
  let d = 0;
  for (let i = 0; i < n; i++) {
    const Fx = cdfStdNormal(sorted[i]);
    const ecdfR = (i + 1) / n;
    const ecdfL = i / n;
    const diff = Math.max(Math.abs(ecdfR - Fx), Math.abs(ecdfL - Fx));
    if (diff > d) d = diff;
  }
  return d;
}

// ── Component ───────────────────────────────────────────────────────────────

function BerryEsseenExplorer() {
  const [presetAId, setPresetAId] = useState<string>('uniform');
  const [presetBId, setPresetBId] = useState<string>('exponential');
  const [n, setN] = useState<number>(50);
  const [compare, setCompare] = useState<boolean>(true);

  const presetA = useMemo<BerryEsseenPreset>(
    () => berryEsseenPresets.find((p) => p.id === presetAId) ?? berryEsseenPresets[0],
    [presetAId],
  );
  const presetB = useMemo<BerryEsseenPreset>(
    () => berryEsseenPresets.find((p) => p.id === presetBId) ?? berryEsseenPresets[0],
    [presetBId],
  );

  // Sweep: empirical sup-deviation curve for each preset across N_GRID.
  const sweepA = useMemo(() => computeSweep(presetA), [presetA]);
  const sweepB = useMemo(
    () => (compare ? computeSweep(presetB) : null),
    [presetB, compare],
  );

  // Fn vs Φ deviation as a function of x for current n (just preset A).
  const devProfile = useMemo(() => {
    const { sampler, mu, sigma } = makeSamplerAndStats(presetA.id);
    const reps = cltReplications(sampler, n, M_REPS, mu, sigma);
    const sorted = [...reps].sort((a, b) => a - b);
    const xs = d3.range(-3.5, 3.51, 0.1);
    const points: Array<{ x: number; dev: number }> = xs.map((x) => {
      // Empirical F at x: fraction of sorted <= x
      const idx = d3.bisectRight(sorted, x);
      const Fn = idx / sorted.length;
      return { x, dev: Fn - cdfStdNormal(x) };
    });
    return points;
  }, [presetA, n]);

  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(260, width);
  const twoCol = width >= 760;
  const panelW = twoCol ? (w - 12) / 2 : w;

  return (
    <div
      className="my-8 rounded-lg border p-5"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            Distribution A
          </span>
          <select
            className="rounded border p-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            value={presetAId}
            onChange={(e) => setPresetAId(e.target.value)}
          >
            {berryEsseenPresets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            Current n = {n} (for deviation profile)
          </span>
          <input
            type="range"
            min={5}
            max={500}
            step={5}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
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
              Compare with B
            </span>
          </label>
          {compare && (
            <select
              className="rounded border p-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
              value={presetBId}
              onChange={(e) => setPresetBId(e.target.value)}
            >
              {berryEsseenPresets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      <div
        ref={containerRef}
        className={twoCol ? 'mt-4 grid grid-cols-2 gap-3' : 'mt-4 flex flex-col gap-4'}
      >
        <SupDeviationPanel
          width={panelW}
          sweepA={sweepA}
          sweepB={sweepB}
          presetA={presetA}
          presetB={compare ? presetB : null}
        />
        <DeviationProfilePanel width={panelW} points={devProfile} presetA={presetA} n={n} />
      </div>

      <div className="mt-4 grid gap-3 text-xs sm:grid-cols-2">
        <Summary preset={presetA} />
        {compare && <Summary preset={presetB} />}
      </div>
    </div>
  );
}

// ── Sweep Computation ───────────────────────────────────────────────────────

interface SweepPoint {
  n: number;
  empirical: number;
  bound: number;
}

function computeSweep(preset: BerryEsseenPreset): SweepPoint[] {
  const { sampler, mu, sigma } = makeSamplerAndStats(preset.id);
  return N_GRID.map((n) => {
    const reps = cltReplications(sampler, n, M_REPS, mu, sigma);
    return {
      n,
      empirical: supDeviation(reps),
      bound: berryEsseenBound(preset.rho, n),
    };
  });
}

// ── Panels ─────────────────────────────────────────────────────────────────

function SupDeviationPanel(props: {
  width: number;
  sweepA: SweepPoint[];
  sweepB: SweepPoint[] | null;
  presetA: BerryEsseenPreset;
  presetB: BerryEsseenPreset | null;
}) {
  const { width, sweepA, sweepB, presetA, presetB } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3
      .scaleLog()
      .domain([N_GRID[0], N_GRID[N_GRID.length - 1]])
      .range([0, innerW]);

    const allY = [
      ...sweepA.flatMap((p) => [p.empirical, p.bound]),
      ...(sweepB ? sweepB.flatMap((p) => [p.empirical, p.bound]) : []),
    ];
    const yMax = Math.min(1.05, Math.max(...allY, 0.01) * 1.2);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const empLine = d3
      .line<SweepPoint>()
      .x((p) => xScale(p.n))
      .y((p) => yScale(p.empirical));
    const boundLine = d3
      .line<SweepPoint>()
      .x((p) => xScale(p.n))
      .y((p) => yScale(p.bound));

    // Draw for A
    g.append('path')
      .datum(sweepA)
      .attr('fill', 'none')
      .attr('stroke', presetA.color)
      .attr('stroke-width', 2)
      .attr('d', empLine);
    g.append('g')
      .selectAll('circle')
      .data(sweepA)
      .join('circle')
      .attr('cx', (p) => xScale(p.n))
      .attr('cy', (p) => yScale(p.empirical))
      .attr('r', 3)
      .attr('fill', presetA.color);
    g.append('path')
      .datum(sweepA)
      .attr('fill', 'none')
      .attr('stroke', presetA.color)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3')
      .attr('d', boundLine);

    if (sweepB && presetB) {
      g.append('path')
        .datum(sweepB)
        .attr('fill', 'none')
        .attr('stroke', presetB.color)
        .attr('stroke-width', 2)
        .attr('d', empLine);
      g.append('g')
        .selectAll('circle')
        .data(sweepB)
        .join('circle')
        .attr('cx', (p) => xScale(p.n))
        .attr('cy', (p) => yScale(p.empirical))
        .attr('r', 3)
        .attr('fill', presetB.color);
      g.append('path')
        .datum(sweepB)
        .attr('fill', 'none')
        .attr('stroke', presetB.color)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 3')
        .attr('d', boundLine);
    }

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickValues(N_GRID)
          .tickFormat((d) => `${d}`),
      )
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('n (log scale)');
    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('sup|Fₙ(x) − Φ(x)|');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Empirical sup deviation vs Berry–Esseen bound (dashed)');
  }, [width, sweepA, sweepB, presetA, presetB]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function DeviationProfilePanel(props: {
  width: number;
  points: Array<{ x: number; dev: number }>;
  presetA: BerryEsseenPreset;
  n: number;
}) {
  const { width, points, presetA, n } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([-3.5, 3.5]).range([0, innerW]);
    const absMax = Math.max(...points.map((p) => Math.abs(p.dev)), 0.01);
    const yScale = d3
      .scaleLinear()
      .domain([-absMax * 1.2, absMax * 1.2])
      .range([innerH, 0])
      .nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Zero reference line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', BOUND_COLOR)
      .attr('stroke-opacity', 0.3);

    const line = d3
      .line<{ x: number; dev: number }>()
      .x((p) => xScale(p.x))
      .y((p) => yScale(p.dev));
    g.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', presetA.color)
      .attr('stroke-width', 2)
      .attr('d', line);

    // Horizontal dashed bound lines at ±C·ρ/√n
    const bnd = berryEsseenBound(presetA.rho, n);
    const bndClamped = Math.min(bnd, absMax * 1.2);
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', yScale(bndClamped))
      .attr('y2', yScale(bndClamped))
      .attr('stroke', BOUND_COLOR)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3');
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', yScale(-bndClamped))
      .attr('y2', yScale(-bndClamped))
      .attr('stroke', BOUND_COLOR)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3');

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
      .text('x');
    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Fₙ(x) − Φ(x)');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(`Deviation Fₙ(x) − Φ(x) at n = ${n}`);
  }, [width, points, presetA, n]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function Summary({ preset }: { preset: BerryEsseenPreset }) {
  return (
    <div className="rounded border p-3" style={{ borderColor: preset.color }}>
      <div className="mb-1 font-semibold" style={{ color: preset.color }}>
        {preset.name}
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <StatCell label="ρ = E[|X|³]/σ³" value={preset.rho.toFixed(2)} />
        <StatCell label="skewness" value={preset.skewness.toFixed(2)} />
        <StatCell label="speed" value={preset.convergenceSpeed} />
      </div>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide opacity-60">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  );
}

export default BerryEsseenExplorer;
