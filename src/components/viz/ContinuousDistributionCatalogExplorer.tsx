import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfUniform, pdfNormal, pdfExponential, pdfGamma, pdfBeta,
  pdfChi2, pdfStudentT, pdfF,
  cdfUniform, cdfNormal, cdfExponential, cdfGamma, cdfBeta,
  cdfChi2,
  cdfStdNormal,
  expectationUniform, expectationNormal, expectationExponential,
  expectationGamma, expectationBeta, expectationChi2,
  expectationStudentT, expectationF,
  varianceUniform, varianceNormal, varianceExponential,
  varianceGamma, varianceBeta, varianceChi2,
  varianceStudentT, varianceF,
} from './shared/distributions';
import { continuousDistributionConfigs } from '../../data/continuous-distributions-data';

// ── PDF/CDF dispatch ──────────────────────────────────────────────────────

function evalPDF(key: string, x: number, p: Record<string, number>): number {
  switch (key) {
    case 'Uniform': return pdfUniform(x, p.a, p.b);
    case 'Normal': return pdfNormal(x, p.mu, p.sigma2);
    case 'Exponential': return pdfExponential(x, p.lambda);
    case 'Gamma': return pdfGamma(x, p.alpha, p.beta);
    case 'Beta': return pdfBeta(x, p.a, p.b);
    case 'Chi2': return pdfChi2(x, p.k);
    case 'StudentT': return pdfStudentT(x, p.nu);
    case 'F': return pdfF(x, p.d1, p.d2);
    default: return 0;
  }
}

function evalCDF(key: string, x: number, p: Record<string, number>): number {
  switch (key) {
    case 'Uniform': return cdfUniform(x, p.a, p.b);
    case 'Normal': return cdfNormal(x, p.mu, p.sigma2);
    case 'Exponential': return cdfExponential(x, p.lambda);
    case 'Gamma': return cdfGamma(x, p.alpha, p.beta);
    case 'Beta': return cdfBeta(x, p.a, p.b);
    case 'Chi2': return cdfChi2(x, p.k);
    case 'StudentT': {
      // Numerical CDF via trapezoidal for t distribution
      let sum = 0.5;
      const n = 200;
      const h = x / n;
      if (x > 0) {
        for (let i = 1; i <= n; i++) sum += pdfStudentT(i * h, p.nu) * h;
      } else {
        for (let i = 1; i <= Math.abs(n); i++) sum -= pdfStudentT(-i * Math.abs(h), p.nu) * Math.abs(h);
      }
      return Math.max(0, Math.min(1, sum));
    }
    case 'F': return cdfGamma(x, p.d1 / 2, p.d2 / (2 * (p.d2 > 0 ? 1 : 1))); // Approximate via Gamma
    default: return 0;
  }
}

function getExpectation(key: string, p: Record<string, number>): number {
  switch (key) {
    case 'Uniform': return expectationUniform(p.a, p.b);
    case 'Normal': return expectationNormal(p.mu);
    case 'Exponential': return expectationExponential(p.lambda);
    case 'Gamma': return expectationGamma(p.alpha, p.beta);
    case 'Beta': return expectationBeta(p.a, p.b);
    case 'Chi2': return expectationChi2(p.k);
    case 'StudentT': return expectationStudentT(p.nu);
    case 'F': return expectationF(p.d1, p.d2);
    default: return 0;
  }
}

function getVariance(key: string, p: Record<string, number>): number {
  switch (key) {
    case 'Uniform': return varianceUniform(p.a, p.b);
    case 'Normal': return varianceNormal(p.sigma2);
    case 'Exponential': return varianceExponential(p.lambda);
    case 'Gamma': return varianceGamma(p.alpha, p.beta);
    case 'Beta': return varianceBeta(p.a, p.b);
    case 'Chi2': return varianceChi2(p.k);
    case 'StudentT': return varianceStudentT(p.nu);
    case 'F': return varianceF(p.d1, p.d2);
    default: return 0;
  }
}

// ── Layout ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

export default function ContinuousDistributionCatalogExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [distIndex, setDistIndex] = useState(1); // default to Normal
  const [params, setParams] = useState<Record<string, number>>(() => {
    const cfg = continuousDistributionConfigs[1];
    const init: Record<string, number> = {};
    cfg.params.forEach((pd) => { init[pd.name] = pd.default; });
    return init;
  });
  const [queryX, setQueryX] = useState<number | null>(null);

  const config = continuousDistributionConfigs[distIndex];
  const panelW = Math.max(200, (width || 600) / 2 - 8);
  const chartH = 240;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const [supportMin, supportMax] = useMemo(
    () => config.supportRange(params),
    [config, params],
  );

  // Generate PDF curve points
  const pdfData = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    const step = (supportMax - supportMin) / NUM_POINTS;
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = supportMin + i * step;
      data.push({ x, y: evalPDF(config.key, x, params) });
    }
    return data;
  }, [config.key, params, supportMin, supportMax]);

  const maxPDF = useMemo(() => Math.max(...pdfData.map((d) => d.y), 0.01), [pdfData]);

  // CDF curve points
  const cdfData = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    const step = (supportMax - supportMin) / NUM_POINTS;
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = supportMin + i * step;
      data.push({ x, y: evalCDF(config.key, x, params) });
    }
    return data;
  }, [config.key, params, supportMin, supportMax]);

  // Moments
  const mean = useMemo(() => getExpectation(config.key, params), [config.key, params]);
  const variance = useMemo(() => getVariance(config.key, params), [config.key, params]);
  const sigma = Math.sqrt(Math.abs(variance));

  // Scales
  const xScale = useCallback(
    (v: number) => MARGIN.left + ((v - supportMin) / (supportMax - supportMin)) * plotW,
    [supportMin, supportMax, plotW],
  );
  const yScalePDF = useCallback(
    (v: number) => MARGIN.top + plotH - (v / (maxPDF * 1.15)) * plotH,
    [plotH, maxPDF],
  );
  const yScaleCDF = useCallback(
    (v: number) => MARGIN.top + plotH - v * plotH,
    [plotH],
  );
  const xInverse = useCallback(
    (px: number) => supportMin + ((px - MARGIN.left) / plotW) * (supportMax - supportMin),
    [supportMin, supportMax, plotW],
  );

  // SVG paths
  const pdfPath = useMemo(() => {
    return pdfData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScalePDF(d.y).toFixed(1)}`)
      .join(' ');
  }, [pdfData, xScale, yScalePDF]);

  const cdfPath = useMemo(() => {
    return cdfData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScaleCDF(d.y).toFixed(1)}`)
      .join(' ');
  }, [cdfData, xScale, yScaleCDF]);

  // Area fill under PDF up to queryX
  const pdfFillPath = useMemo(() => {
    if (queryX === null) return '';
    const pts = pdfData.filter((d) => d.x <= queryX);
    if (pts.length < 2) return '';
    let path = pts
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScalePDF(d.y).toFixed(1)}`)
      .join(' ');
    path += ` L${xScale(pts[pts.length - 1].x).toFixed(1)},${yScalePDF(0).toFixed(1)}`;
    path += ` L${xScale(pts[0].x).toFixed(1)},${yScalePDF(0).toFixed(1)} Z`;
    return path;
  }, [queryX, pdfData, xScale, yScalePDF]);

  // Handlers
  const handleDistChange = (idx: number) => {
    setDistIndex(idx);
    const cfg = continuousDistributionConfigs[idx];
    const init: Record<string, number> = {};
    cfg.params.forEach((pd) => { init[pd.name] = pd.default; });
    setParams(init);
    setQueryX(null);
  };

  const handleParamChange = (name: string, value: number) => {
    setParams((prev) => {
      const next = { ...prev, [name]: value };
      if (config.key === 'Uniform') {
        if (name === 'a') next.b = Math.max(next.b, value + 0.5);
        if (name === 'b') next.b = Math.max(next.b, next.a + 0.5);
      }
      return next;
    });
  };

  const handlePDFMouse = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = xInverse(px);
    if (x >= supportMin && x <= supportMax) setQueryX(x);
  };

  // X-axis ticks
  const xTicks = useMemo(() => {
    const range = supportMax - supportMin;
    const step = range <= 2 ? 0.5 : range <= 10 ? 1 : range <= 50 ? 5 : 10;
    const ticks: number[] = [];
    const start = Math.ceil(supportMin / step) * step;
    for (let t = start; t <= supportMax; t += step) {
      ticks.push(t);
    }
    return ticks;
  }, [supportMin, supportMax]);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Continuous Distribution Catalog Explorer
      </div>

      {/* Distribution selector */}
      <div className="flex flex-wrap gap-1.5 justify-center mb-4">
        {continuousDistributionConfigs.map((cfg, i) => (
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
            <span className="font-medium">{pd.label}</span>
            <input
              type="range"
              min={pd.min}
              max={pd.max}
              step={pd.step}
              value={params[pd.name] ?? pd.default}
              onChange={(e) => handleParamChange(pd.name, Number(e.target.value))}
              className="w-24"
            />
            <span className="w-10 tabular-nums">{(params[pd.name] ?? pd.default).toFixed(pd.step < 1 ? 1 : 0)}</span>
          </label>
        ))}
      </div>

      {/* Charts */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* PDF panel */}
        <div>
          <div className="text-center text-xs font-medium mb-1">PDF: f(x)</div>
          <svg
            width={panelW}
            height={chartH}
            className="block cursor-crosshair"
            onMouseMove={handlePDFMouse}
            onMouseLeave={() => setQueryX(null)}
          >
            {/* Y grid */}
            {[0, maxPDF * 0.25, maxPDF * 0.5, maxPDF * 0.75, maxPDF].map((v, i) => (
              <g key={i}>
                <line
                  x1={MARGIN.left} y1={yScalePDF(v)} x2={panelW - MARGIN.right} y2={yScalePDF(v)}
                  stroke="currentColor" strokeOpacity={0.08}
                />
                <text x={MARGIN.left - 4} y={yScalePDF(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(v < 0.1 ? 2 : 1)}
                </text>
              </g>
            ))}

            {/* X-axis ticks */}
            {xTicks.map((t) => (
              <text key={t} x={xScale(t)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                {Number.isInteger(t) ? t : t.toFixed(1)}
              </text>
            ))}

            {/* Area fill */}
            {pdfFillPath && (
              <path d={pdfFillPath} fill={config.color} opacity={0.2} />
            )}

            {/* PDF curve */}
            <path d={pdfPath} fill="none" stroke={config.color} strokeWidth={2} />

            {/* E[X] marker */}
            {isFinite(mean) && mean >= supportMin && mean <= supportMax && (
              <>
                <line
                  x1={xScale(mean)} y1={MARGIN.top} x2={xScale(mean)} y2={yScalePDF(0)}
                  stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4,3"
                />
                <polygon
                  points={`${xScale(mean)},${yScalePDF(0) + 2} ${xScale(mean) - 5},${yScalePDF(0) + 10} ${xScale(mean) + 5},${yScalePDF(0) + 10}`}
                  fill="#DC2626"
                />
              </>
            )}

            {/* Query indicator */}
            {queryX !== null && (
              <>
                <line
                  x1={xScale(queryX)} y1={MARGIN.top} x2={xScale(queryX)} y2={yScalePDF(0)}
                  stroke="currentColor" strokeWidth={1} strokeDasharray="2,2" strokeOpacity={0.5}
                />
                <text x={xScale(queryX)} y={MARGIN.top - 2} textAnchor="middle" className="fill-current" style={{ fontSize: '9px', fontWeight: 600 }}>
                  f({queryX.toFixed(2)}) = {evalPDF(config.key, queryX, params).toFixed(4)}
                </text>
              </>
            )}
          </svg>
        </div>

        {/* CDF panel */}
        <div>
          <div className="text-center text-xs font-medium mb-1">CDF: F(x) = P(X ≤ x)</div>
          <svg width={panelW} height={chartH} className="block">
            {/* Y grid */}
            {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
              <g key={i}>
                <line
                  x1={MARGIN.left} y1={yScaleCDF(v)} x2={panelW - MARGIN.right} y2={yScaleCDF(v)}
                  stroke="currentColor" strokeOpacity={0.08}
                />
                <text x={MARGIN.left - 4} y={yScaleCDF(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* X-axis ticks */}
            {xTicks.map((t) => (
              <text key={t} x={xScale(t)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                {Number.isInteger(t) ? t : t.toFixed(1)}
              </text>
            ))}

            {/* CDF curve */}
            <path d={cdfPath} fill="none" stroke={config.color} strokeWidth={2} />

            {/* Query line */}
            {queryX !== null && (
              <>
                <line
                  x1={xScale(queryX)} y1={MARGIN.top} x2={xScale(queryX)} y2={yScaleCDF(0)}
                  stroke="currentColor" strokeWidth={1} strokeDasharray="2,2" strokeOpacity={0.5}
                />
                <text x={xScale(queryX)} y={MARGIN.top - 2} textAnchor="middle" className="fill-current" style={{ fontSize: '9px', fontWeight: 600 }}>
                  F({queryX.toFixed(2)}) = {evalCDF(config.key, queryX, params).toFixed(4)}
                </text>
              </>
            )}
          </svg>
        </div>
      </div>

      {/* Moments readout */}
      <div className="flex flex-wrap gap-6 justify-center mt-3 text-xs">
        <span><strong>E[X]</strong> = {isFinite(mean) ? mean.toFixed(4) : 'undefined'}</span>
        <span><strong>Var(X)</strong> = {isFinite(variance) && !isNaN(variance) ? variance.toFixed(4) : 'undefined'}</span>
        <span><strong>σ</strong> = {isFinite(sigma) && !isNaN(sigma) ? sigma.toFixed(4) : 'undefined'}</span>
        <span><strong>Exp. family?</strong> {config.expFamily ? 'Yes' : 'No'}</span>
      </div>
    </div>
  );
}
