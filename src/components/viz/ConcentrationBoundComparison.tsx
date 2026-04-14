import { useState, useMemo, useRef, useEffect, useDeferredValue } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  chebyshevBound,
  hoeffdingBound,
  bernsteinBound,
  subGaussianBound,
  uniformSample,
  bernoulliSample,
  normalSample,
  exponentialSample,
} from './shared/convergence';
import {
  concentrationDistributions,
  type ConcentrationDistribution,
} from '../../data/large-deviations-data';

const MARGIN = { top: 10, right: 16, bottom: 36, left: 50 };
const H = 260;
const MC_REPLICATIONS = 50_000;

// ── Sampler registry ────────────────────────────────────────────────────────

function makeSampler(d: ConcentrationDistribution): () => number {
  switch (d.id) {
    case 'bernoulli-05':
      return () => bernoulliSample(0.5);
    case 'bernoulli-01':
      return () => bernoulliSample(0.1);
    case 'uniform':
      return () => uniformSample(0, 1);
    case 'normal-truncated':
      return () => {
        let x = normalSample(0, 1);
        while (x < -3 || x > 3) x = normalSample(0, 1);
        return x;
      };
    case 'exponential':
      return () => exponentialSample(1);
    default:
      return () => uniformSample(0, 1);
  }
}

// Estimate P(|X̄ₙ − μ| ≥ ε) via MC. Heavy; memoized outside the render loop.
function mcTailProb(
  sampler: () => number,
  mu: number,
  n: number,
  epsilons: number[],
  M: number
): number[] {
  // For each epsilon we count replications deviating by >= epsilon.
  const counts = new Uint32Array(epsilons.length);
  for (let rep = 0; rep < M; rep++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += sampler();
    const dev = Math.abs(sum / n - mu);
    for (let j = 0; j < epsilons.length; j++) {
      if (dev >= epsilons[j]) counts[j]++;
    }
  }
  const out = new Array<number>(epsilons.length);
  for (let j = 0; j < epsilons.length; j++) out[j] = counts[j] / M;
  return out;
}

// ── Component ───────────────────────────────────────────────────────────────

export default function ConcentrationBoundComparison() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 640, 320);

  const [distId, setDistId] = useState<string>('bernoulli-01');
  const [n, setN] = useState(100);
  const [showExact, setShowExact] = useState(true);
  const deferredN = useDeferredValue(n);

  const dist =
    concentrationDistributions.find((d) => d.id === distId) ??
    concentrationDistributions[0];

  const epsilons = useMemo(() => {
    const pts: number[] = [];
    for (let i = 1; i <= 60; i++) pts.push(0.005 * i);
    return pts;
  }, []);

  const exactTail = useMemo(() => {
    if (!showExact) return null;
    const sampler = makeSampler(dist);
    return mcTailProb(sampler, dist.mu, deferredN, epsilons, MC_REPLICATIONS);
  }, [distId, deferredN, epsilons, showExact, dist]);

  const boundsData = useMemo(() => {
    const markov = epsilons.map(() => 1); // Markov of |X̄ₙ − μ| needs E[|X̄ₙ − μ|]; treat as ≥ 1 to show looseness
    const chebyshev = epsilons.map((e) =>
      Math.min(chebyshevBound(deferredN, dist.sigma2, e), 1)
    );
    const hoeffding =
      dist.range !== undefined
        ? epsilons.map((e) =>
            hoeffdingBound(deferredN, dist.range![0], dist.range![1], e)
          )
        : null;
    const bernstein =
      dist.range !== undefined && dist.M !== undefined
        ? epsilons.map((e) =>
            bernsteinBound(deferredN, dist.sigma2, dist.M!, e)
          )
        : null;
    const subGauss =
      dist.subGaussianParam !== undefined
        ? epsilons.map((e) =>
            subGaussianBound(deferredN, dist.subGaussianParam!, e)
          )
        : null;
    return { markov, chebyshev, hoeffding, bernstein, subGauss };
  }, [deferredN, dist, epsilons]);

  const curves = useMemo(() => {
    const out: Array<{ label: string; color: string; data: number[] }> = [];
    out.push({ label: 'Markov (≥1)', color: '#6b7280', data: boundsData.markov });
    out.push({
      label: 'Chebyshev',
      color: '#d97706',
      data: boundsData.chebyshev,
    });
    if (boundsData.hoeffding)
      out.push({
        label: 'Hoeffding',
        color: '#059669',
        data: boundsData.hoeffding,
      });
    if (boundsData.bernstein)
      out.push({
        label: 'Bernstein',
        color: '#2563eb',
        data: boundsData.bernstein,
      });
    if (boundsData.subGauss)
      out.push({
        label: 'Sub-Gaussian',
        color: '#7c3aed',
        data: boundsData.subGauss,
      });
    if (showExact && exactTail)
      out.push({
        label: 'MC (exact)',
        color: '#dc2626',
        data: exactTail.map((v) => Math.max(v, 1e-10)),
      });
    return out;
  }, [boundsData, exactTail, showExact]);

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = w - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const xScale = d3.scaleLinear().domain(d3.extent(epsilons) as [number, number]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([1e-10, 1]).range([innerH, 0]).clamp(true);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(
        d3
          .axisLeft(yScale)
          .tickValues([1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 1])
          .tickFormat((d) => (d as number).toExponential(0))
      )
      .selectAll('text')
      .style('font-size', '10px');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('ε');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('P(|X̄ₙ − μ| ≥ ε)');

    for (const c of curves) {
      const line = d3
        .line<{ x: number; y: number }>()
        .defined((d) => d.y > 0 && isFinite(d.y))
        .x((d) => xScale(d.x))
        .y((d) => yScale(Math.max(d.y, 1e-10)));
      const pts = epsilons.map((e, i) => ({ x: e, y: c.data[i] }));
      g.append('path')
        .attr('d', line(pts) ?? '')
        .attr('fill', 'none')
        .attr('stroke', c.color)
        .attr('stroke-width', c.label === 'MC (exact)' ? 1.5 : 2)
        .attr('stroke-dasharray', c.label === 'MC (exact)' ? '3 3' : null);
    }
  }, [curves, w, epsilons]);

  return (
    <div
      ref={containerRef}
      style={{
        fontFamily: 'system-ui, sans-serif',
        border: '1px solid var(--color-border)',
        borderRadius: '0.75rem',
        padding: '1.25rem',
        margin: '1.5rem 0',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          alignItems: 'center',
          marginBottom: '0.75rem',
        }}
      >
        <strong>Concentration-Bound Comparison</strong>
        <select
          value={distId}
          onChange={(e) => setDistId(e.target.value)}
          style={{ padding: '0.2rem', marginLeft: 'auto' }}
        >
          {concentrationDistributions.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
        <label style={{ fontSize: '0.8125rem' }}>
          <input
            type="checkbox"
            checked={showExact}
            onChange={(e) => setShowExact(e.target.checked)}
            style={{ marginRight: '0.3rem' }}
          />
          Show MC (exact)
        </label>
      </div>

      <label style={{ fontSize: '0.8125rem', display: 'block', marginBottom: '0.5rem' }}>
        Sample size n = {n}
        <input
          type="range"
          min={10}
          max={1000}
          step={10}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </label>

      <svg ref={svgRef} width={w} height={H} />

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          fontSize: '0.75rem',
          marginTop: '0.5rem',
        }}
      >
        {curves.map((c) => (
          <span key={c.label} style={{ color: c.color }}>
            ● {c.label}
          </span>
        ))}
      </div>
      <div
        style={{
          fontSize: '0.75rem',
          color: 'var(--color-text-muted)',
          marginTop: '0.5rem',
        }}
      >
        {dist.description}
      </div>
    </div>
  );
}
