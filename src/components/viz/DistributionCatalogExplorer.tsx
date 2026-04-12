import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBernoulli, pmfBinomial, pmfGeometric, pmfPoisson,
  pmfNegativeBinomial, pmfHypergeometric, pmfDiscreteUniform,
  cdfBinomial, cdfGeometric, cdfPoisson,
  cdfNegativeBinomial, cdfHypergeometric, cdfDiscreteUniform,
  expectationBernoulli, expectationBinomial, expectationGeometric,
  expectationNegBin, expectationPoisson, expectationHypergeometric,
  expectationDiscreteUniform,
  varianceBernoulli, varianceBinomial, varianceGeometric,
  varianceNegBin, variancePoisson, varianceHypergeometric,
  varianceDiscreteUniform,
} from './shared/distributions';
import { distributionConfigs } from '../../data/discrete-distributions-data';

// ── PMF/CDF dispatch ──────────────────────────────────────────────────────

function evalPMF(key: string, k: number, p: Record<string, number>): number {
  switch (key) {
    case 'Bernoulli': return pmfBernoulli(k, p.p);
    case 'Binomial': return pmfBinomial(k, p.n, p.p);
    case 'Geometric': return pmfGeometric(k, p.p);
    case 'NegBin': return pmfNegativeBinomial(k, p.r, p.p);
    case 'Poisson': return pmfPoisson(k, p.lambda);
    case 'Hypergeometric': return pmfHypergeometric(k, p.N, p.K, p.n);
    case 'DiscreteUniform': return pmfDiscreteUniform(k, p.a, p.b);
    default: return 0;
  }
}

// CDF wrappers — some functions don't exist with that exact name yet,
// so we build from PMF summation where needed.
function evalCDF(key: string, k: number, p: Record<string, number>): number {
  switch (key) {
    case 'Bernoulli': return k < 0 ? 0 : k >= 1 ? 1 : 1 - p.p;
    case 'Binomial': return cdfBinomial(k, p.n, p.p);
    case 'Geometric': return cdfGeometric(k, p.p);
    case 'NegBin': return cdfNegativeBinomial(k, p.r, p.p);
    case 'Poisson': return cdfPoisson(k, p.lambda);
    case 'Hypergeometric': return cdfHypergeometric(k, p.N, p.K, p.n);
    case 'DiscreteUniform': return cdfDiscreteUniform(k, p.a, p.b);
    default: return 0;
  }
}

function getExpectation(key: string, p: Record<string, number>): number {
  switch (key) {
    case 'Bernoulli': return expectationBernoulli(p.p);
    case 'Binomial': return expectationBinomial(p.n, p.p);
    case 'Geometric': return expectationGeometric(p.p);
    case 'NegBin': return expectationNegBin(p.r, p.p);
    case 'Poisson': return expectationPoisson(p.lambda);
    case 'Hypergeometric': return expectationHypergeometric(p.N, p.K, p.n);
    case 'DiscreteUniform': return expectationDiscreteUniform(p.a, p.b);
    default: return 0;
  }
}

function getVariance(key: string, p: Record<string, number>): number {
  switch (key) {
    case 'Bernoulli': return varianceBernoulli(p.p);
    case 'Binomial': return varianceBinomial(p.n, p.p);
    case 'Geometric': return varianceGeometric(p.p);
    case 'NegBin': return varianceNegBin(p.r, p.p);
    case 'Poisson': return variancePoisson(p.lambda);
    case 'Hypergeometric': return varianceHypergeometric(p.N, p.K, p.n);
    case 'DiscreteUniform': return varianceDiscreteUniform(p.a, p.b);
    default: return 0;
  }
}

// ── Layout ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };

export default function DistributionCatalogExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [distIndex, setDistIndex] = useState(0);
  const [params, setParams] = useState<Record<string, number>>(() => {
    const cfg = distributionConfigs[0];
    const init: Record<string, number> = {};
    cfg.params.forEach((pd) => { init[pd.name] = pd.default; });
    return init;
  });
  const [queryX, setQueryX] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const config = distributionConfigs[distIndex];
  const panelW = Math.max(200, (width || 600) / 2 - 8);
  const chartH = 240;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Compute support range
  const [supportMin, supportMax] = useMemo(
    () => config.supportRange(params),
    [config, params],
  );

  // Generate PMF bars
  const pmfData = useMemo(() => {
    const data: { k: number; p: number }[] = [];
    for (let k = supportMin; k <= supportMax; k++) {
      data.push({ k, p: evalPMF(config.key, k, params) });
    }
    return data;
  }, [config.key, params, supportMin, supportMax]);

  const maxP = useMemo(() => Math.max(...pmfData.map((d) => d.p), 0.01), [pmfData]);

  // Moments
  const mean = useMemo(() => getExpectation(config.key, params), [config.key, params]);
  const variance = useMemo(() => getVariance(config.key, params), [config.key, params]);
  const sigma = Math.sqrt(variance);

  // Mode: k with highest PMF
  const mode = useMemo(() => {
    let bestK = supportMin;
    let bestP = -1;
    pmfData.forEach(({ k, p }) => { if (p > bestP) { bestP = p; bestK = k; } });
    return bestK;
  }, [pmfData, supportMin]);

  // Scales
  const xScale = useCallback(
    (v: number) => MARGIN.left + ((v - supportMin + 0.5) / (supportMax - supportMin + 1)) * plotW,
    [supportMin, supportMax, plotW],
  );
  const yScalePMF = useCallback(
    (v: number) => MARGIN.top + plotH - (v / (maxP * 1.15)) * plotH,
    [plotH, maxP],
  );
  const yScaleCDF = useCallback(
    (v: number) => MARGIN.top + plotH - v * plotH,
    [plotH],
  );
  const xInverse = useCallback(
    (px: number) => {
      const frac = (px - MARGIN.left) / plotW;
      return Math.round(supportMin + frac * (supportMax - supportMin + 1) - 0.5);
    },
    [supportMin, supportMax, plotW],
  );

  const barW = Math.max(2, plotW / (supportMax - supportMin + 1) - 2);

  // CDF step data
  const cdfData = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    pts.push({ x: supportMin - 1, y: 0 });
    for (let k = supportMin; k <= supportMax; k++) {
      const cdfBefore = evalCDF(config.key, k - 0.5, params);
      const cdfAt = evalCDF(config.key, k, params);
      pts.push({ x: k, y: cdfBefore });
      pts.push({ x: k, y: cdfAt });
    }
    pts.push({ x: supportMax + 1, y: 1 });
    return pts;
  }, [config.key, params, supportMin, supportMax]);

  const cdfPath = useMemo(() => {
    const xScaleCDF = (v: number) => MARGIN.left + ((v - supportMin + 1) / (supportMax - supportMin + 2)) * plotW;
    return cdfData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScaleCDF(d.x).toFixed(1)},${yScaleCDF(d.y).toFixed(1)}`)
      .join(' ');
  }, [cdfData, yScaleCDF, supportMin, supportMax, plotW]);

  // Distribution change handler
  const handleDistChange = (idx: number) => {
    setDistIndex(idx);
    const cfg = distributionConfigs[idx];
    const init: Record<string, number> = {};
    cfg.params.forEach((pd) => { init[pd.name] = pd.default; });
    setParams(init);
    setQueryX(null);
    setShowTable(false);
  };

  const handleParamChange = (name: string, value: number) => {
    setParams((prev) => {
      const next = { ...prev, [name]: value };
      // Clamp dependent params for Hypergeometric
      if (config.key === 'Hypergeometric') {
        next.K = Math.min(Math.max(0, next.K), next.N);
        next.n = Math.min(Math.max(0, next.n), next.N);
      }
      // Clamp DiscreteUniform: ensure b > a regardless of which slider changed
      if (config.key === 'DiscreteUniform') {
        if (name === 'a') next.b = Math.max(next.b, value + 1);
        if (name === 'b') next.b = Math.max(next.b, next.a + 1);
      }
      return next;
    });
  };

  const handlePMFMouse = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const k = xInverse(px);
    if (k >= supportMin && k <= supportMax) setQueryX(k);
  };

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Distribution Catalog Explorer
      </div>

      {/* Distribution selector */}
      <div className="flex flex-wrap gap-1.5 justify-center mb-4">
        {distributionConfigs.map((cfg, i) => (
          <button
            key={cfg.key}
            onClick={() => handleDistChange(i)}
            className="px-2.5 py-1 text-xs rounded-full transition-colors"
            style={{
              backgroundColor: i === distIndex ? cfg.color : 'transparent',
              color: i === distIndex ? 'white' : 'currentColor',
              border: `1px solid ${cfg.color}`,
            }}
          >
            {cfg.name}
          </button>
        ))}
      </div>

      {/* Parameter sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {config.params.map((pd) => (
          <label key={pd.name} className="flex items-center gap-2 text-xs">
            <span className="font-medium w-4 text-right">{pd.label}</span>
            <input
              type="range"
              min={pd.min}
              max={pd.max}
              step={pd.step}
              value={params[pd.name] ?? pd.default}
              onChange={(e) => handleParamChange(pd.name, Number(e.target.value))}
              className="w-24"
            />
            <span className="w-8 tabular-nums">{(params[pd.name] ?? pd.default).toFixed(pd.step < 1 ? 2 : 0)}</span>
          </label>
        ))}
      </div>

      {/* Charts */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* PMF panel */}
        <div>
          <div className="text-center text-xs font-medium mb-1">PMF: P(X = k)</div>
          <svg
            width={panelW}
            height={chartH}
            className="block cursor-crosshair"
            onMouseMove={handlePMFMouse}
            onMouseLeave={() => setQueryX(null)}
          >
            {/* Y grid */}
            {[0, maxP * 0.25, maxP * 0.5, maxP * 0.75, maxP].map((v, i) => (
              <g key={i}>
                <line
                  x1={MARGIN.left} y1={yScalePMF(v)} x2={panelW - MARGIN.right} y2={yScalePMF(v)}
                  stroke="currentColor" strokeOpacity={0.08}
                />
                <text x={MARGIN.left - 4} y={yScalePMF(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(v < 0.01 ? 3 : 2)}
                </text>
              </g>
            ))}

            {/* Bars */}
            {pmfData.map(({ k, p }) => (
              <rect
                key={k}
                x={xScale(k) - barW / 2}
                y={yScalePMF(p)}
                width={barW}
                height={Math.max(0, yScalePMF(0) - yScalePMF(p))}
                fill={config.color}
                opacity={queryX !== null && k <= queryX ? 0.85 : 0.6}
                rx={1}
              />
            ))}

            {/* E[X] marker */}
            <line
              x1={xScale(mean)} y1={MARGIN.top} x2={xScale(mean)} y2={yScalePMF(0)}
              stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4,3"
            />
            <polygon
              points={`${xScale(mean)},${yScalePMF(0) + 2} ${xScale(mean) - 5},${yScalePMF(0) + 10} ${xScale(mean) + 5},${yScalePMF(0) + 10}`}
              fill="#DC2626"
            />

            {/* Query indicator */}
            {queryX !== null && (
              <text x={xScale(queryX)} y={MARGIN.top - 2} textAnchor="middle" className="fill-current" style={{ fontSize: '9px', fontWeight: 600 }}>
                P(X={queryX}) = {evalPMF(config.key, queryX, params).toFixed(4)}
              </text>
            )}

            {/* X axis labels */}
            {pmfData.filter((_, i) => pmfData.length <= 20 || i % Math.ceil(pmfData.length / 15) === 0).map(({ k }) => (
              <text key={k} x={xScale(k)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                {k}
              </text>
            ))}
          </svg>
        </div>

        {/* CDF panel */}
        <div>
          <div className="text-center text-xs font-medium mb-1">CDF: F(x) = P(X ≤ x)</div>
          <svg width={panelW} height={chartH} className="block">
            {/* Y grid */}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line
                  x1={MARGIN.left} y1={yScaleCDF(v)} x2={panelW - MARGIN.right} y2={yScaleCDF(v)}
                  stroke="currentColor" strokeOpacity={0.08}
                />
                <text x={MARGIN.left - 4} y={yScaleCDF(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* CDF step function */}
            <path d={cdfPath} fill="none" stroke="#059669" strokeWidth={2} />

            {/* E[X] marker on CDF */}
            {(() => {
              const xScaleCDF = (v: number) => MARGIN.left + ((v - supportMin + 1) / (supportMax - supportMin + 2)) * plotW;
              return (
                <line
                  x1={xScaleCDF(mean)} y1={MARGIN.top} x2={xScaleCDF(mean)} y2={yScaleCDF(0)}
                  stroke="#DC2626" strokeWidth={1} strokeDasharray="3,3" opacity={0.5}
                />
              );
            })()}
          </svg>
        </div>
      </div>

      {/* Moments readout */}
      <div className="flex flex-wrap gap-6 justify-center mt-3 text-xs">
        <span><b>E[X]</b> = {mean.toFixed(4)}</span>
        <span><b>Var(X)</b> = {variance.toFixed(4)}</span>
        <span><b>σ</b> = {sigma.toFixed(4)}</span>
        <span><b>Mode</b> = {mode}</span>
        <span className="opacity-60">{config.mgfFormula}</span>
      </div>

      {/* Exponential family badge */}
      <div className="text-center mt-2">
        <span className={`inline-block px-2 py-0.5 text-xs rounded-full ${config.expFamily ? 'bg-emerald-100 text-emerald-800' : 'bg-gray-100 text-gray-600'}`}>
          {config.expFamily ? 'Exponential family member' : 'Not exponential family'}
        </span>
      </div>

      {/* PMF table toggle */}
      <div className="text-center mt-3">
        <button
          onClick={() => setShowTable(!showTable)}
          className="text-xs px-3 py-1 rounded border hover:bg-gray-50 transition-colors"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {showTable ? 'Hide PMF table' : 'Show PMF table'}
        </button>
      </div>
      {showTable && (
        <div className="mt-2 max-h-48 overflow-auto text-xs">
          <table className="w-full text-center">
            <thead>
              <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                <th className="px-2 py-1">k</th>
                <th className="px-2 py-1">P(X = k)</th>
                <th className="px-2 py-1">F(k)</th>
              </tr>
            </thead>
            <tbody>
              {pmfData.map(({ k, p }) => (
                <tr key={k} className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                  <td className="px-2 py-0.5 tabular-nums">{k}</td>
                  <td className="px-2 py-0.5 tabular-nums">{p.toFixed(6)}</td>
                  <td className="px-2 py-0.5 tabular-nums">{evalCDF(config.key, k, params).toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
