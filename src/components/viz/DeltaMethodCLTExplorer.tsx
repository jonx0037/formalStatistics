import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cltReplications,
  normalSample,
  exponentialSample,
  uniformSample,
  poissonSample,
} from './shared/convergence';
import { pdfNormal } from './shared/distributions';
import {
  deltaMethodTransformations,
  deltaMethodDistributions,
  type DeltaMethodDistribution,
  type DeltaMethodTransformation,
} from '../../data/modes-of-convergence-data';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 18, bottom: 38, left: 52 };
const CHART_H = 260;
const HIST_BINS = 30;
const M_REPS = 3000;

const CLT_COLOR = '#7C3AED';
const DELTA_COLOR = '#059669';
const THEORY_COLOR = '#DC2626';

// ── Sampler Factory ─────────────────────────────────────────────────────────

function makeSampler(dist: DeltaMethodDistribution): () => number {
  // Match by name since the dataset doesn't carry id.
  if (dist.name.startsWith('Normal')) {
    const sigma = Math.sqrt(dist.sigmaSquared);
    return () => normalSample(dist.mu, sigma);
  }
  if (dist.name.startsWith('Exponential')) {
    return () => exponentialSample(1 / dist.mu);
  }
  if (dist.name.startsWith('Poisson')) {
    return () => poissonSample(dist.mu);
  }
  if (dist.name.startsWith('Uniform')) {
    // Uniform(0, 10) has mu = 5, sigmaSquared = 100/12; endpoints a = 0, b = 10.
    const half = Math.sqrt(3 * dist.sigmaSquared);
    return () => uniformSample(dist.mu - half, dist.mu + half);
  }
  return () => normalSample(dist.mu, Math.sqrt(dist.sigmaSquared));
}

// ── Component ───────────────────────────────────────────────────────────────

function DeltaMethodCLTExplorer() {
  const [distIdx, setDistIdx] = useState<number>(0);
  const [transIdx, setTransIdx] = useState<number>(0);
  const [n, setN] = useState<number>(100);

  const dist = deltaMethodDistributions[distIdx];
  const trans = deltaMethodTransformations[transIdx];

  const mu = dist.mu;
  const sigma2 = dist.sigmaSquared;
  const sigma = Math.sqrt(sigma2);
  const gMu = useMemo(() => trans.g(mu), [trans, mu]);
  const gPrimeMu = useMemo(() => trans.gPrime(mu), [trans, mu]);
  const deltaVariance = gPrimeMu * gPrimeMu * sigma2; // asymptotic variance of √n(g(X̄) − g(μ))

  const sampler = useMemo(() => makeSampler(dist), [dist]);

  // Generate √n(X̄ − μ) directly (CLT scale, not standardized to unit variance).
  const cltScaled = useMemo(() => {
    const rawStandardized = cltReplications(sampler, n, M_REPS, mu, sigma); // gives √n(X̄ − μ)/σ
    return rawStandardized.map((z) => z * sigma); // scale up to √n(X̄ − μ)
  }, [sampler, n, mu, sigma]);

  // Generate √n(g(X̄) − g(μ)) via running the sampler directly (need actual g(X̄)).
  const deltaScaled = useMemo(() => {
    const out = new Array<number>(M_REPS);
    const sqrtN = Math.sqrt(n);
    for (let r = 0; r < M_REPS; r++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += sampler();
      const xbar = sum / n;
      // Guard against undefined values (e.g., log(0) if uniform includes 0)
      const gx = trans.g(xbar);
      out[r] = isFinite(gx) ? sqrtN * (gx - gMu) : 0;
    }
    return out;
  }, [sampler, n, trans, gMu]);

  // Variance comparison: direct simulation variance of √n(g(X̄) − g(μ)) vs [g'(μ)]²σ².
  const simVariance = useMemo(() => {
    let mean = 0;
    for (const v of deltaScaled) mean += v;
    mean /= deltaScaled.length;
    let v = 0;
    for (const x of deltaScaled) v += (x - mean) * (x - mean);
    return v / deltaScaled.length;
  }, [deltaScaled]);

  // Curve: predicted vs simulated variance over a grid of n.
  const varSweep = useMemo(() => {
    const nGrid = [10, 25, 50, 100, 200, 500, 1000];
    return nGrid.map((nn) => {
      // Quick 500-rep simulation at each n for the curve (cheap)
      const reps = 500;
      const sqrtN = Math.sqrt(nn);
      const vals: number[] = [];
      for (let r = 0; r < reps; r++) {
        let sum = 0;
        for (let i = 0; i < nn; i++) sum += sampler();
        const xbar = sum / nn;
        const gx = trans.g(xbar);
        if (isFinite(gx)) vals.push(sqrtN * (gx - gMu));
      }
      let mean = 0;
      for (const v of vals) mean += v;
      mean /= vals.length;
      let vr = 0;
      for (const x of vals) vr += (x - mean) * (x - mean);
      vr /= vals.length;
      return { n: nn, sim: vr, theory: deltaVariance };
    });
  }, [sampler, trans, gMu, deltaVariance]);

  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(260, width);
  const threeCol = width >= 900;
  const panelW = threeCol ? (w - 24) / 3 : w;

  const degenerate = Math.abs(gPrimeMu) < 1e-6;

  return (
    <div
      className="my-8 rounded-lg border p-5"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            Distribution
          </span>
          <select
            className="rounded border p-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            value={distIdx}
            onChange={(e) => setDistIdx(parseInt(e.target.value, 10))}
          >
            {deltaMethodDistributions.map((d, i) => (
              <option key={d.name} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            Transformation g
          </span>
          <select
            className="rounded border p-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            value={transIdx}
            onChange={(e) => setTransIdx(parseInt(e.target.value, 10))}
          >
            {deltaMethodTransformations.map((t, i) => (
              <option key={t.name} value={i}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            n = {n}
          </span>
          <input
            type="range"
            min={10}
            max={1000}
            step={10}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
          />
        </label>
      </div>

      {degenerate && (
        <div
          className="mt-3 rounded border p-2 text-xs"
          style={{ borderColor: THEORY_COLOR, color: THEORY_COLOR }}
        >
          Note: g′(μ) = 0 → delta method fails at first order. The limit is a
          scaled chi-squared at rate 1/n, not a Normal at rate 1/√n.
        </div>
      )}

      <div
        ref={containerRef}
        className={threeCol ? 'mt-4 grid grid-cols-3 gap-3' : 'mt-4 flex flex-col gap-4'}
      >
        <CLTPanel width={panelW} values={cltScaled} sigma2={sigma2} />
        <DeltaPanel width={panelW} values={deltaScaled} predVar={deltaVariance} />
        <VariancePanel width={panelW} sweep={varSweep} n={n} />
      </div>

      <div className="mt-4 grid gap-3 text-xs sm:grid-cols-4">
        <Stat label="μ" value={mu.toFixed(3)} />
        <Stat label="σ²" value={sigma2.toFixed(3)} />
        <Stat label="g(μ)" value={isFinite(gMu) ? gMu.toFixed(3) : 'undef'} />
        <Stat label="g′(μ)" value={gPrimeMu.toFixed(3)} />
        <Stat label="[g′(μ)]² σ² (theory)" value={deltaVariance.toFixed(4)} color={THEORY_COLOR} />
        <Stat label="simulated var" value={simVariance.toFixed(4)} color={DELTA_COLOR} />
        <Stat
          label="ratio sim/theory"
          value={deltaVariance === 0 ? '—' : (simVariance / deltaVariance).toFixed(3)}
        />
        <Stat label="M replications" value={M_REPS.toLocaleString()} />
      </div>
    </div>
  );
}

// ── Panels ─────────────────────────────────────────────────────────────────

function HistogramWithNormal({
  width,
  values,
  variance,
  title,
  color,
}: {
  width: number;
  values: number[];
  variance: number;
  title: string;
  color: string;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const finite = values.filter((v) => isFinite(v));
    if (finite.length === 0) return;
    const extent = d3.extent(finite) as [number, number];
    const range = extent[1] - extent[0] || 1;
    const pad = range * 0.05;
    const dom: [number, number] = [extent[0] - pad, extent[1] + pad];

    const xScale = d3.scaleLinear().domain(dom).range([0, innerW]);
    const bins = d3.bin<number, number>().domain(dom).thresholds(HIST_BINS)(finite);
    const density = (b: d3.Bin<number, number>) =>
      b.length / (finite.length * (b.x1! - b.x0!));
    const yMax = Math.max(
      d3.max(bins, density) ?? 0.01,
      pdfNormal(0, 0, variance) * 1.05,
      0.01,
    );
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('fill', color)
      .attr('fill-opacity', 0.55)
      .selectAll('rect')
      .data(bins)
      .join('rect')
      .attr('x', (b) => xScale(b.x0!))
      .attr('y', (b) => yScale(density(b)))
      .attr('width', (b) => Math.max(0, xScale(b.x1!) - xScale(b.x0!) - 1))
      .attr('height', (b) => innerH - yScale(density(b)));

    const steps = 200;
    const xs = d3.range(steps + 1).map((i) => dom[0] + (i / steps) * (dom[1] - dom[0]));
    const line = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(pdfNormal(x, 0, variance)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', THEORY_COLOR)
      .attr('stroke-width', 2)
      .attr('d', line);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('font-size', 10);

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(title);
  }, [width, values, variance, title, color]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function CLTPanel({
  width,
  values,
  sigma2,
}: {
  width: number;
  values: number[];
  sigma2: number;
}) {
  return (
    <HistogramWithNormal
      width={width}
      values={values}
      variance={sigma2}
      title="√n(X̄ₙ − μ) vs N(0, σ²)"
      color={CLT_COLOR}
    />
  );
}

function DeltaPanel({
  width,
  values,
  predVar,
}: {
  width: number;
  values: number[];
  predVar: number;
}) {
  return (
    <HistogramWithNormal
      width={width}
      values={values}
      variance={predVar}
      title="√n(g(X̄) − g(μ)) vs N(0, [g′(μ)]²σ²)"
      color={DELTA_COLOR}
    />
  );
}

function VariancePanel({
  width,
  sweep,
  n,
}: {
  width: number;
  sweep: Array<{ n: number; sim: number; theory: number }>;
  n: number;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3
      .scaleLog()
      .domain([sweep[0].n, sweep[sweep.length - 1].n])
      .range([0, innerW]);

    const all = [...sweep.map((s) => s.sim), ...sweep.map((s) => s.theory)];
    const yMax = Math.max(...all, 0.001) * 1.2;
    const yMin = Math.min(...all, 0.001) * 0.8;
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const simLine = d3
      .line<{ n: number; sim: number }>()
      .x((d) => xScale(d.n))
      .y((d) => yScale(d.sim));
    const theoryLine = d3
      .line<{ n: number; theory: number }>()
      .x((d) => xScale(d.n))
      .y((d) => yScale(d.theory));

    g.append('path')
      .datum(sweep)
      .attr('fill', 'none')
      .attr('stroke', THEORY_COLOR)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4 3')
      .attr('d', theoryLine);

    g.append('path')
      .datum(sweep)
      .attr('fill', 'none')
      .attr('stroke', DELTA_COLOR)
      .attr('stroke-width', 2)
      .attr('d', simLine);
    g.append('g')
      .selectAll('circle')
      .data(sweep)
      .join('circle')
      .attr('cx', (d) => xScale(d.n))
      .attr('cy', (d) => yScale(d.sim))
      .attr('r', 3)
      .attr('fill', DELTA_COLOR);

    // Marker at current n
    if (n >= sweep[0].n && n <= sweep[sweep.length - 1].n) {
      g.append('line')
        .attr('x1', xScale(n))
        .attr('x2', xScale(n))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#6B7280')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '2 2');
    }

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickValues(sweep.map((s) => s.n))
          .tickFormat((d) => `${d}`),
      )
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('n (log)');
    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Var of √n(g(X̄)−g(μ))');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Simulation (green) vs theory (red dashed)');
  }, [width, sweep, n]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div
      className="rounded border p-2"
      style={{ borderColor: color ?? 'var(--color-border)' }}
    >
      <div className="text-[10px] uppercase tracking-wide opacity-60">{label}</div>
      <div className="font-mono text-sm" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export default DeltaMethodCLTExplorer;
