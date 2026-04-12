import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cdfBinomial, cdfNormal, cdfUniform, cdfExponential, cdfGeometric, cdfPoisson,
} from './shared/distributions';
import { distributionColors } from './shared/colorScales';

const DISCRETE_DISTS = [
  { name: 'Binomial', params: { n: 10, p: 0.3 } },
  { name: 'Geometric', params: { p: 0.3 } },
  { name: 'Poisson', params: { lambda: 5 } },
];

const CONTINUOUS_DISTS = [
  { name: 'Normal', params: { mu: 0, sigma2: 1 } },
  { name: 'Uniform', params: { a: 0, b: 1 } },
  { name: 'Exponential', params: { lambda: 1 } },
];

// ── CDF evaluators ─────────────────────────────────────────────────────────

function evalCDF(name: string, x: number, params: Record<string, number>): number {
  switch (name) {
    case 'Binomial': return cdfBinomial(x, params.n, params.p);
    case 'Geometric': return cdfGeometric(x, params.p);
    case 'Poisson': return cdfPoisson(x, params.lambda);
    case 'Normal': return cdfNormal(x, params.mu, params.sigma2);
    case 'Uniform': return cdfUniform(x, params.a, params.b);
    case 'Exponential': return cdfExponential(x, params.lambda);
    default: return 0;
  }
}

function getXRange(name: string, params: Record<string, number>): [number, number] {
  switch (name) {
    case 'Binomial': return [-1, params.n + 1];
    case 'Geometric': return [0, Math.ceil(6 / params.p) + 1];
    case 'Poisson': return [-1, Math.ceil(params.lambda + 4 * Math.sqrt(params.lambda)) + 2];
    case 'Normal': {
      const sd = Math.sqrt(params.sigma2);
      return [params.mu - 4 * sd, params.mu + 4 * sd];
    }
    case 'Uniform': return [params.a - 1, params.b + 1];
    case 'Exponential': return [-0.5, 5 / params.lambda + 1];
    default: return [0, 1];
  }
}

// ── Layout ─────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 30, left: 45 };

function CDFPanel({
  type, distName, params, panelWidth, queryX, onQueryChange,
}: {
  type: 'discrete' | 'continuous';
  distName: string;
  params: Record<string, number>;
  panelWidth: number;
  queryX: number;
  onQueryChange: (x: number) => void;
}) {
  const panelH = 240;
  const plotW = panelWidth - MARGIN.left - MARGIN.right;
  const plotH = panelH - MARGIN.top - MARGIN.bottom;
  const [xMin, xMax] = getXRange(distName, params);

  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );
  const yScale = useCallback(
    (val: number) => MARGIN.top + plotH - val * plotH,
    [plotH],
  );
  const xInverse = useCallback(
    (px: number) => xMin + ((px - MARGIN.left) / plotW) * (xMax - xMin),
    [xMin, xMax, plotW],
  );

  // CDF data
  const cdfData = useMemo(() => {
    if (type === 'discrete') {
      // Step function points
      const range = getXRange(distName, params);
      const steps: { x: number; y: number }[] = [];
      steps.push({ x: range[0], y: 0 });
      const maxK = distName === 'Binomial' ? params.n : Math.ceil(range[1]);
      const startK = distName === 'Geometric' ? 1 : 0;
      for (let k = startK; k <= maxK; k++) {
        const cdfVal = evalCDF(distName, k, params);
        steps.push({ x: k, y: evalCDF(distName, k - 0.001, params) }); // just before jump
        steps.push({ x: k, y: cdfVal }); // at jump
      }
      steps.push({ x: range[1], y: 1 });
      return steps;
    }
    // Smooth curve
    const nPoints = 200;
    const step = (xMax - xMin) / nPoints;
    return Array.from({ length: nPoints + 1 }, (_, i) => {
      const x = xMin + i * step;
      return { x, y: evalCDF(distName, x, params) };
    });
  }, [type, distName, params, xMin, xMax]);

  // Build CDF path
  const cdfPath = useMemo(() => {
    return cdfData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');
  }, [cdfData, xScale, yScale]);

  const fxValue = evalCDF(distName, queryX, params);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const xVal = xInverse(px);
    onQueryChange(Math.max(xMin, Math.min(xMax, xVal)));
  };

  return (
    <div>
      <div className="mb-1 text-center text-sm font-medium">
        {type === 'discrete' ? 'Discrete' : 'Continuous'}: {distName}
      </div>
      <svg
        width={panelWidth}
        height={panelH}
        className="block cursor-crosshair"
        onMouseMove={handleMouseMove}
      >
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={MARGIN.left}
              y1={yScale(t)}
              x2={panelWidth - MARGIN.right}
              y2={yScale(t)}
              stroke="currentColor"
              strokeOpacity={0.1}
            />
            <text x={MARGIN.left - 6} y={yScale(t) + 4} textAnchor="end" className="fill-current" style={{ fontSize: '9px' }}>
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {/* CDF curve */}
        <path d={cdfPath} fill="none" stroke={distributionColors.cdf} strokeWidth={2} />

        {/* Query line */}
        <line
          x1={xScale(queryX)}
          y1={MARGIN.top}
          x2={xScale(queryX)}
          y2={yScale(0)}
          stroke={distributionColors.sample}
          strokeWidth={1.5}
          strokeDasharray="4,3"
        />
        {/* Horizontal line to y-axis */}
        <line
          x1={MARGIN.left}
          y1={yScale(fxValue)}
          x2={xScale(queryX)}
          y2={yScale(fxValue)}
          stroke={distributionColors.sample}
          strokeWidth={1}
          strokeDasharray="4,3"
          opacity={0.5}
        />
        {/* Dot at intersection */}
        <circle cx={xScale(queryX)} cy={yScale(fxValue)} r={4} fill={distributionColors.sample} />

        {/* Labels */}
        <text x={xScale(queryX)} y={yScale(0) + 16} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }}>
          x = {queryX.toFixed(type === 'discrete' ? 0 : 2)}
        </text>

        {/* CDF properties annotation */}
        <text x={panelWidth - MARGIN.right} y={MARGIN.top + 12} textAnchor="end" className="fill-current" style={{ fontSize: '9px', opacity: 0.6 }}>
          non-decreasing ✓
        </text>
        <text x={panelWidth - MARGIN.right} y={MARGIN.top + 24} textAnchor="end" className="fill-current" style={{ fontSize: '9px', opacity: 0.6 }}>
          {type === 'discrete' ? 'right-continuous ✓' : 'continuous ✓'}
        </text>
      </svg>
    </div>
  );
}

export default function CDFExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [discreteDistIdx, setDiscreteDistIdx] = useState(0);
  const [continuousDistIdx, setContinuousDistIdx] = useState(0);
  const [queryXDiscrete, setQueryXDiscrete] = useState(3);
  const [queryXContinuous, setQueryXContinuous] = useState(0);

  const discreteDist = DISCRETE_DISTS[discreteDistIdx];
  const continuousDist = CONTINUOUS_DISTS[continuousDistIdx];

  const compact = width < 600;
  const panelWidth = compact ? Math.max(width - 16, 280) : Math.floor((width - 32) / 2);

  const fxDiscrete = evalCDF(discreteDist.name, queryXDiscrete, discreteDist.params);
  const fxContinuous = evalCDF(continuousDist.name, queryXContinuous, continuousDist.params);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Discrete:</label>
          <select
            value={discreteDistIdx}
            onChange={(e) => setDiscreteDistIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {DISCRETE_DISTS.map((d, i) => (
              <option key={d.name} value={i}>{d.name}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Continuous:</label>
          <select
            value={continuousDistIdx}
            onChange={(e) => setContinuousDistIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {CONTINUOUS_DISTS.map((d, i) => (
              <option key={d.name} value={i}>{d.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Side-by-side panels */}
      <div className={compact ? 'space-y-4' : 'flex gap-4'}>
        <CDFPanel
          type="discrete"
          distName={discreteDist.name}
          params={discreteDist.params}
          panelWidth={panelWidth}
          queryX={queryXDiscrete}
          onQueryChange={setQueryXDiscrete}
        />
        <CDFPanel
          type="continuous"
          distName={continuousDist.name}
          params={continuousDist.params}
          panelWidth={panelWidth}
          queryX={queryXContinuous}
          onQueryChange={setQueryXContinuous}
        />
      </div>

      {/* Readout */}
      <div className="mt-3 flex flex-wrap gap-6 text-sm font-mono">
        <div>
          <span className="text-gray-500">Discrete:</span>{' '}
          F({queryXDiscrete.toFixed(0)}) = P(X ≤ {queryXDiscrete.toFixed(0)}) = {fxDiscrete.toFixed(4)}
        </div>
        <div>
          <span className="text-gray-500">Continuous:</span>{' '}
          F({queryXContinuous.toFixed(2)}) = P(X ≤ {queryXContinuous.toFixed(2)}) = {fxContinuous.toFixed(4)}
        </div>
      </div>
    </div>
  );
}
