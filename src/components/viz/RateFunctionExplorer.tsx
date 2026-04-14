import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { bernoulliKL } from './shared/convergence';

// ── Types ──────────────────────────────────────────────────────────────────

type RateDistId = 'bernoulli' | 'normal' | 'exponential' | 'poisson';

interface RateDist {
  id: RateDistId;
  name: string;
  support: [number, number];
  mean: number;
  /** Closed-form rate function I(x). Returns Infinity outside support. */
  I: (x: number) => number;
  /** Closed-form log-MGF Λ(t) = log Mₓ(t). */
  logMgf: (t: number) => number;
  /** Optimal t*(x) for the Legendre transform (closed form). */
  optimalT: (x: number) => number;
}

function bernoulliRate(p: number): RateDist {
  return {
    id: 'bernoulli',
    name: `Bernoulli(${p})`,
    support: [0, 1],
    mean: p,
    I: (x) => {
      if (x < 0 || x > 1) return Infinity;
      return bernoulliKL(x, p);
    },
    logMgf: (t) => Math.log(1 - p + p * Math.exp(t)),
    optimalT: (x) => {
      if (x <= 0 || x >= 1) return 0;
      return Math.log((x * (1 - p)) / (p * (1 - x)));
    },
  };
}

function normalRate(mu: number, sigma2: number): RateDist {
  return {
    id: 'normal',
    name: `Normal(${mu}, ${sigma2})`,
    support: [mu - 4 * Math.sqrt(sigma2), mu + 4 * Math.sqrt(sigma2)],
    mean: mu,
    I: (x) => ((x - mu) * (x - mu)) / (2 * sigma2),
    logMgf: (t) => mu * t + (sigma2 * t * t) / 2,
    optimalT: (x) => (x - mu) / sigma2,
  };
}

function exponentialRate(lambda: number): RateDist {
  return {
    id: 'exponential',
    name: `Exponential(${lambda})`,
    support: [0.01, 5 / lambda],
    mean: 1 / lambda,
    I: (x) => {
      if (x <= 0) return Infinity;
      return lambda * x - 1 - Math.log(lambda * x);
    },
    logMgf: (t) => (t < lambda ? -Math.log(1 - t / lambda) : Infinity),
    optimalT: (x) => (x > 0 ? lambda - 1 / x : 0),
  };
}

function poissonRate(lambda: number): RateDist {
  return {
    id: 'poisson',
    name: `Poisson(${lambda})`,
    support: [0.1, 2 * lambda],
    mean: lambda,
    I: (x) => {
      if (x <= 0) return Infinity;
      return x * Math.log(x / lambda) - x + lambda;
    },
    logMgf: (t) => lambda * (Math.exp(t) - 1),
    optimalT: (x) => (x > 0 ? Math.log(x / lambda) : 0),
  };
}

const DIST_PRESETS: Record<RateDistId, RateDist> = {
  bernoulli: bernoulliRate(0.3),
  normal: normalRate(0, 1),
  exponential: exponentialRate(1),
  poisson: poissonRate(5),
};

// ── Component ──────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 16, bottom: 36, left: 48 };
const H = 220;

export default function RateFunctionExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 640, 320);

  const [distId, setDistId] = useState<RateDistId>('bernoulli');
  const [n, setN] = useState(50);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const dist = DIST_PRESETS[distId];

  const xPoints = useMemo(() => {
    const [xMin, xMax] = dist.support;
    const pts: number[] = [];
    const steps = 200;
    for (let i = 0; i <= steps; i++) {
      pts.push(xMin + (i / steps) * (xMax - xMin));
    }
    return pts;
  }, [dist.support[0], dist.support[1], distId]);

  const rateData = useMemo(
    () => xPoints.map((x) => ({ x, I: dist.I(x) })).filter((d) => isFinite(d.I)),
    [xPoints, dist]
  );

  const panelW = w > 700 ? (w - 16) / 2 : w;

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
          gap: '0.5rem',
          alignItems: 'center',
          marginBottom: '0.75rem',
        }}
      >
        <strong>Cramér Rate Function I(x)</strong>
        <select
          value={distId}
          onChange={(e) => setDistId(e.target.value as RateDistId)}
          style={{ padding: '0.2rem', marginLeft: 'auto' }}
        >
          <option value="bernoulli">Bernoulli(0.3)</option>
          <option value="normal">Normal(0, 1)</option>
          <option value="exponential">Exponential(1)</option>
          <option value="poisson">Poisson(5)</option>
        </select>
      </div>

      <label style={{ fontSize: '0.8125rem', display: 'block', marginBottom: '0.5rem' }}>
        Sample size n = {n}
        <input
          type="range"
          min={1}
          max={500}
          step={1}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </label>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 700 ? '1fr 1fr' : '1fr',
          gap: '1rem',
        }}
      >
        <RatePanel
          width={panelW}
          data={rateData}
          mean={dist.mean}
          n={n}
          hoverX={hoverX}
          setHoverX={setHoverX}
          iAt={dist.I}
        />
        <LogMgfPanel
          width={panelW}
          logMgf={dist.logMgf}
          hoverX={hoverX}
          dist={dist}
        />
      </div>

      {hoverX !== null && (
        <div
          style={{
            marginTop: '0.75rem',
            fontSize: '0.8125rem',
            color: 'var(--color-text-muted)',
          }}
        >
          At x = {hoverX.toFixed(3)}: I(x) = {dist.I(hoverX).toFixed(4)}, t* ={' '}
          {dist.optimalT(hoverX).toFixed(3)}. Chernoff bound P(X̄ₙ ≥ x) ≤ e^(−nI(x)) ={' '}
          {Math.exp(-n * dist.I(hoverX)).toExponential(3)}.
        </div>
      )}
    </div>
  );
}

// ── Rate panel ─────────────────────────────────────────────────────────────

function RatePanel({
  width,
  data,
  mean,
  n,
  hoverX,
  setHoverX,
  iAt,
}: {
  width: number;
  data: Array<{ x: number; I: number }>;
  mean: number;
  n: number;
  hoverX: number | null;
  setHoverX: (x: number | null) => void;
  iAt: (x: number) => number;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  // Keep the x/y scales stable across re-renders so the hover-only effect can
  // translate hoverX → pixel coords without rebuilding the chart.
  const scalesRef = useRef<{
    xScale: d3.ScaleLinear<number, number>;
    yScale: d3.ScaleLinear<number, number>;
    innerH: number;
  } | null>(null);

  // Static draw — runs only when data/width/mean change. Never wipes on hover.
  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const [xMin, xMax] = d3.extent(data, (d) => d.x);
    if (xMin === undefined || xMax === undefined) return;
    const iMax = d3.max(data, (d) => d.I) ?? 1;

    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, iMax]).range([innerH, 0]);
    scalesRef.current = { xScale, yScale, innerH };

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('x');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('I(x)');

    const line = d3
      .line<{ x: number; I: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.I));
    g.append('path')
      .attr('d', line(data) ?? '')
      .attr('fill', 'none')
      .attr('stroke', '#7c3aed')
      .attr('stroke-width', 2);

    g.append('line')
      .attr('x1', xScale(mean))
      .attr('x2', xScale(mean))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#059669')
      .attr('stroke-dasharray', '3 3')
      .attr('stroke-width', 1);
    g.append('text')
      .attr('x', xScale(mean) + 4)
      .attr('y', 12)
      .style('font-size', '10px')
      .style('fill', '#059669')
      .text(`μ = ${mean.toFixed(2)}`);

    // Dedicated hover layer, mutated by the second effect without rebuilds.
    g.append('g').attr('class', 'hover-layer');

    const overlay = g
      .append('rect')
      .attr('width', innerW)
      .attr('height', innerH)
      .attr('fill', 'transparent')
      .attr('pointer-events', 'all');
    overlay.on('mousemove', (event) => {
      const [mx] = d3.pointer(event);
      setHoverX(xScale.invert(mx));
    });
    overlay.on('mouseleave', () => setHoverX(null));
  }, [data, width, mean, setHoverX]);

  // Hover-only update — redraws the two-element hover layer without touching
  // the rest of the chart. iAt is the closed-form rate function, so the
  // annotation is exact, not a tolerance-search fallback.
  useEffect(() => {
    if (!svgRef.current || !scalesRef.current) return;
    const svg = d3.select(svgRef.current);
    const layer = svg.select<SVGGElement>('g.hover-layer');
    layer.selectAll('*').remove();
    if (hoverX === null) return;
    const { xScale, yScale, innerH } = scalesRef.current;
    const iVal = iAt(hoverX);
    if (!isFinite(iVal)) return;
    const hx = xScale(hoverX);
    layer
      .append('line')
      .attr('x1', hx)
      .attr('x2', hx)
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 1);
    layer
      .append('text')
      .attr('x', hx + 5)
      .attr('y', yScale(iVal) - 4)
      .style('font-size', '10px')
      .style('fill', '#dc2626')
      .text(`e^(-${(n * iVal).toFixed(2)}) = ${Math.exp(-n * iVal).toExponential(2)}`);
  }, [hoverX, n, iAt]);

  return <svg ref={svgRef} width={width} height={H} />;
}

// ── Log-MGF panel with tangent line ────────────────────────────────────────

function LogMgfPanel({
  width,
  logMgf,
  hoverX,
  dist,
}: {
  width: number;
  logMgf: (t: number) => number;
  hoverX: number | null;
  dist: RateDist;
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    const tMin = -2;
    const tMax = 2;
    const pts: Array<{ t: number; lambda: number }> = [];
    const steps = 200;
    for (let i = 0; i <= steps; i++) {
      const t = tMin + (i / steps) * (tMax - tMin);
      const val = logMgf(t);
      if (isFinite(val)) pts.push({ t, lambda: val });
    }

    const yExt = d3.extent(pts, (d) => d.lambda) as [number, number];
    const xScale = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);
    const yScale = d3
      .scaleLinear()
      .domain([Math.min(yExt[0], -0.5), Math.max(yExt[1], 2)])
      .range([innerH, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('t');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Λ(t) = log Mₓ(t)');

    const line = d3
      .line<{ t: number; lambda: number }>()
      .x((d) => xScale(d.t))
      .y((d) => yScale(d.lambda));
    g.append('path')
      .attr('d', line(pts) ?? '')
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2);

    // Origin
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.2)
      .attr('stroke-dasharray', '2 2');

    // Tangent line at t*(hoverX): y = t*x - I(x) passes through (t*, Λ(t*))
    if (hoverX !== null) {
      const tStar = dist.optimalT(hoverX);
      const I = dist.I(hoverX);
      if (isFinite(tStar) && isFinite(I)) {
        // tangent slope = x (the value being queried), intercept = -I
        const yAt = (t: number) => t * hoverX - I;
        g.append('line')
          .attr('x1', xScale(tMin))
          .attr('x2', xScale(tMax))
          .attr('y1', yScale(yAt(tMin)))
          .attr('y2', yScale(yAt(tMax)))
          .attr('stroke', '#dc2626')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '4 3');
        // dot at (t*, Λ(t*))
        if (isFinite(logMgf(tStar))) {
          g.append('circle')
            .attr('cx', xScale(tStar))
            .attr('cy', yScale(logMgf(tStar)))
            .attr('r', 4)
            .attr('fill', '#dc2626');
        }
      }
    }
  }, [width, logMgf, hoverX, dist]);

  return <svg ref={svgRef} width={width} height={H} />;
}
