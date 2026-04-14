import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ── Distribution presets with closed-form MGF and PMF/PDF ───────────────────

type DistId = 'bernoulli-03' | 'bernoulli-05' | 'uniform-01' | 'exp-1';

interface MgfDist {
  id: DistId;
  name: string;
  mu: number;
  /** Log-MGF Λ(t) = log Mₓ(t). */
  logMgf: (t: number) => number;
  /** Discrete support points (for Bernoulli-like) or undefined for continuous. */
  support?: number[];
  /** PMF at each support point (discrete case). */
  pmf?: number[];
  /** PDF (continuous case), evaluated on a grid. */
  pdf?: (x: number) => number;
  /** x range for continuous plotting */
  xRange?: [number, number];
}

const DISTS: Record<DistId, MgfDist> = {
  'bernoulli-03': {
    id: 'bernoulli-03',
    name: 'Bernoulli(0.3)',
    mu: 0.3,
    logMgf: (t) => Math.log(0.7 + 0.3 * Math.exp(t)),
    support: [0, 1],
    pmf: [0.7, 0.3],
  },
  'bernoulli-05': {
    id: 'bernoulli-05',
    name: 'Bernoulli(0.5)',
    mu: 0.5,
    logMgf: (t) => Math.log(0.5 + 0.5 * Math.exp(t)),
    support: [0, 1],
    pmf: [0.5, 0.5],
  },
  'uniform-01': {
    id: 'uniform-01',
    name: 'Uniform(0, 1)',
    mu: 0.5,
    logMgf: (t) => (Math.abs(t) < 1e-8 ? 0 : Math.log((Math.exp(t) - 1) / t)),
    pdf: () => 1,
    xRange: [0, 1],
  },
  'exp-1': {
    id: 'exp-1',
    name: 'Exponential(1)',
    mu: 1,
    logMgf: (t) => (t < 1 ? -Math.log(1 - t) : Infinity),
    pdf: (x) => (x >= 0 ? Math.exp(-x) : 0),
    xRange: [0, 5],
  },
};

const MARGIN = { top: 10, right: 16, bottom: 36, left: 48 };
const H = 240;

export default function ChernoffOptimizationExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 640, 320);

  const [distId, setDistId] = useState<DistId>('bernoulli-03');
  const [epsilon, setEpsilon] = useState(0.1);
  const [n, setN] = useState(50);
  const [t, setT] = useState(1.0);

  const dist = DISTS[distId];

  // Find optimal t* via golden-section on the log-objective
  const { tStar, bound } = useMemo(() => {
    const obj = (s: number) => n * (-s * (dist.mu + epsilon) + dist.logMgf(s));
    const phi = (Math.sqrt(5) - 1) / 2;
    let a = 0.001;
    let b = 5;
    let c = b - phi * (b - a);
    let d = a + phi * (b - a);
    for (let i = 0; i < 80; i++) {
      if (obj(c) < obj(d)) b = d;
      else a = c;
      c = b - phi * (b - a);
      d = a + phi * (b - a);
      if (Math.abs(b - a) < 1e-7) break;
    }
    const ts = (a + b) / 2;
    return { tStar: ts, bound: Math.min(Math.exp(obj(ts)), 1) };
  }, [dist, epsilon, n]);

  // Objective curve as a function of t
  const curve = useMemo(() => {
    const pts: Array<{ t: number; y: number }> = [];
    const tMin = 0.001;
    const tMax = 5;
    const steps = 200;
    for (let i = 0; i <= steps; i++) {
      const tt = tMin + (i / steps) * (tMax - tMin);
      const y = Math.exp(n * (-tt * (dist.mu + epsilon) + dist.logMgf(tt)));
      if (isFinite(y)) pts.push({ t: tt, y });
    }
    return pts;
  }, [dist, epsilon, n]);

  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const panelW = w > 700 ? (w - 16) / 2 : w;

  useEffect(() => {
    if (!leftSvgRef.current || curve.length === 0) return;
    const svg = d3.select(leftSvgRef.current);
    svg.selectAll('*').remove();
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const yPositive = curve.filter((d) => d.y > 0);
    const [yMinRaw, yMaxRaw] = d3.extent(yPositive, (d) => d.y);
    if (yMinRaw === undefined || yMaxRaw === undefined) return;
    const xScale = d3.scaleLinear().domain([0.001, 5]).range([0, innerW]);
    const yScale = d3
      .scaleLog()
      .domain([Math.max(1e-12, yMinRaw), Math.max(1, yMaxRaw)])
      .range([innerH, 0])
      .clamp(true);
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(
        d3
          .axisLeft(yScale)
          .tickValues([1e-9, 1e-6, 1e-3, 1e-1, 1])
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
      .text('t (tilting parameter)');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('e^{n(−t(μ+ε) + Λ(t))}');

    const line = d3
      .line<{ t: number; y: number }>()
      .defined((d) => d.y > 0 && isFinite(d.y))
      .x((d) => xScale(d.t))
      .y((d) => yScale(Math.max(d.y, 1e-12)));
    g.append('path')
      .attr('d', line(curve) ?? '')
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2);

    // Current t (vertical line)
    g.append('line')
      .attr('x1', xScale(t))
      .attr('x2', xScale(t))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 1.5);

    // Optimal t* (star)
    g.append('circle')
      .attr('cx', xScale(tStar))
      .attr('cy', yScale(Math.max(bound, 1e-12)))
      .attr('r', 5)
      .attr('fill', '#059669')
      .attr('stroke', 'white')
      .attr('stroke-width', 1.5);
  }, [curve, panelW, t, tStar, bound]);

  // Right panel: PMF/PDF overlay with tilted distribution
  useEffect(() => {
    if (!rightSvgRef.current) return;
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    if (dist.support && dist.pmf) {
      // Discrete
      const xScale = d3
        .scaleBand()
        .domain(dist.support.map((s) => s.toString()))
        .range([0, innerW])
        .padding(0.3);
      const Mt = Math.exp(dist.logMgf(tStar));
      const tilted = dist.support.map((x, i) => (dist.pmf![i] * Math.exp(tStar * x)) / Mt);
      const maxY = Math.max(...dist.pmf, ...tilted);
      const yScale = d3.scaleLinear().domain([0, maxY * 1.1]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');

      // Original PMF (blue)
      dist.support.forEach((x, i) => {
        g.append('rect')
          .attr('x', (xScale(x.toString()) ?? 0))
          .attr('y', yScale(dist.pmf![i]))
          .attr('width', xScale.bandwidth() / 2)
          .attr('height', innerH - yScale(dist.pmf![i]))
          .attr('fill', '#2563eb')
          .attr('opacity', 0.7);
      });
      // Tilted PMF (red)
      dist.support.forEach((x, i) => {
        g.append('rect')
          .attr('x', (xScale(x.toString()) ?? 0) + xScale.bandwidth() / 2)
          .attr('y', yScale(tilted[i]))
          .attr('width', xScale.bandwidth() / 2)
          .attr('height', innerH - yScale(tilted[i]))
          .attr('fill', '#dc2626')
          .attr('opacity', 0.7);
      });
    } else if (dist.pdf && dist.xRange) {
      // Continuous
      const [xMin, xMax] = dist.xRange;
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
      const steps = 120;
      const origPts: Array<{ x: number; y: number }> = [];
      const tiltPts: Array<{ x: number; y: number }> = [];
      const Mt = Math.exp(dist.logMgf(tStar));
      for (let i = 0; i <= steps; i++) {
        const x = xMin + (i / steps) * (xMax - xMin);
        const p = dist.pdf!(x);
        origPts.push({ x, y: p });
        tiltPts.push({ x, y: (p * Math.exp(tStar * x)) / Mt });
      }
      const maxY = Math.max(
        d3.max(origPts, (d) => d.y) ?? 1,
        d3.max(tiltPts, (d) => d.y) ?? 1
      );
      const yScale = d3.scaleLinear().domain([0, maxY * 1.1]).range([innerH, 0]);
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');

      const line = d3
        .line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));
      g.append('path')
        .attr('d', line(origPts) ?? '')
        .attr('fill', 'none')
        .attr('stroke', '#2563eb')
        .attr('stroke-width', 2);
      g.append('path')
        .attr('d', line(tiltPts) ?? '')
        .attr('fill', 'none')
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4 3');
    }

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('x');
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 10)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#2563eb')
      .text('original');
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 22)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#dc2626')
      .text(`tilted by e^(t*x), t*=${tStar.toFixed(2)}`);
  }, [dist, tStar, panelW]);

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
        <strong>Chernoff Optimization Explorer</strong>
        <select
          value={distId}
          onChange={(e) => setDistId(e.target.value as DistId)}
          style={{ padding: '0.2rem', marginLeft: 'auto' }}
        >
          {Object.values(DISTS).map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: '0.75rem',
          marginBottom: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <label>
          ε = {epsilon.toFixed(3)}
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
        <label>
          n = {n}
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
        <label>
          t = {t.toFixed(2)}
          <input
            type="range"
            min={0.01}
            max={5}
            step={0.01}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 700 ? '1fr 1fr' : '1fr',
          gap: '0.75rem',
        }}
      >
        <svg ref={leftSvgRef} width={panelW} height={H} />
        <svg ref={rightSvgRef} width={panelW} height={H} />
      </div>
      <div
        style={{
          fontSize: '0.8125rem',
          marginTop: '0.5rem',
          color: 'var(--color-text-muted)',
        }}
      >
        Optimal t* = {tStar.toFixed(3)}; Chernoff bound at t* = {bound.toExponential(3)}.
        The tilted distribution (red) shifts mass toward the threshold μ + ε, making the rare event typical.
      </div>
    </div>
  );
}
