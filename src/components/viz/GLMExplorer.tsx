import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  FAMILIES,
  LINKS,
  glmFit,
  predictGLM,
  pearsonResiduals,
  devianceResiduals,
  coefCIProfileGLM,
  type GLMFit,
  type GLMFamily,
  type LinkFunction,
} from './shared/regression';
import { standardNormalCDF } from './shared/testing';
import { glmFamilyColors } from './shared/colorScales';
import {
  GLM_EXPLORER_PRESETS,
  EXAMPLE_7_GLM_DATA,
} from '../../data/regression-data';

const MARGIN = { top: 24, right: 24, bottom: 44, left: 56 };
const TOP_H = 280;
const AXIS_COLOR = '#9CA3AF';

type FamilyKey = 'bernoulli' | 'poisson' | 'gamma' | 'normal';
type PresetKey = keyof typeof GLM_EXPLORER_PRESETS;

const FAMILY_LINKS: Record<FamilyKey, string[]> = {
  bernoulli: ['logit', 'probit', 'cloglog'],
  poisson: ['log', 'identity', 'sqrt'],
  gamma: ['log', 'inverse', 'identity'],
  normal: ['identity', 'log', 'inverse'],
};

const PRESET_LABELS: Record<PresetKey, { label: string; family: FamilyKey }> = {
  creditDefault: { label: 'Credit default (logistic)', family: 'bernoulli' },
  insuranceFrequency: { label: 'Insurance frequency (Poisson + offset)', family: 'poisson' },
  insuranceAmounts: { label: 'Insurance amounts (Gamma)', family: 'gamma' },
  toyBernoulli: { label: 'Toy Bernoulli (n=30)', family: 'bernoulli' },
  toyPoisson: { label: 'Toy Poisson (n=30)', family: 'poisson' },
  toyGamma: { label: 'Toy Gamma (n=30)', family: 'gamma' },
};

export default function GLMExplorer() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 760;

  const [presetKey, setPresetKey] = useState<PresetKey>('creditDefault');
  const initialFamily = PRESET_LABELS[presetKey].family;
  const [familyKey, setFamilyKey] = useState<FamilyKey>(initialFamily);
  const [linkKey, setLinkKey] = useState<string>(FAMILIES[initialFamily].canonicalLink.name);
  const [residualKind, setResidualKind] = useState<'pearson' | 'deviance'>('pearson');
  const [includeOffset, setIncludeOffset] = useState<boolean>(true);
  const [hoveredCoef, setHoveredCoef] = useState<number | null>(null);

  const preset = GLM_EXPLORER_PRESETS[presetKey];
  const X = preset.X;
  const y = preset.y;

  const isPoissonFreq = presetKey === 'insuranceFrequency';
  const offset =
    isPoissonFreq && includeOffset ? EXAMPLE_7_GLM_DATA.offset : undefined;

  const onPresetChange = (key: PresetKey) => {
    setPresetKey(key);
    const fk = PRESET_LABELS[key].family;
    setFamilyKey(fk);
    setLinkKey(FAMILIES[fk].canonicalLink.name);
    setHoveredCoef(null);
  };

  const onFamilyChange = (fk: FamilyKey) => {
    setFamilyKey(fk);
    setLinkKey(FAMILIES[fk].canonicalLink.name);
  };

  // Fit
  const fit = useMemo<GLMFit | null>(() => {
    const family: GLMFamily = FAMILIES[familyKey];
    const link: LinkFunction = LINKS[linkKey] ?? family.canonicalLink;
    try {
      return glmFit(X, y, family, link, { offset, maxIter: 50 });
    } catch {
      return null;
    }
  }, [X, y, familyKey, linkKey, offset]);

  // Coefficient table (β, SE, z, p) using naive vcov.
  const coefTable = useMemo(() => {
    if (!fit) return [];
    const ses = fit.vcov.map((row, j) => Math.sqrt(Math.max(row[j], 0)));
    return fit.beta.map((b, j) => {
      const z = b / Math.max(ses[j], 1e-12);
      return {
        j,
        name: j === 0 ? 'intercept' : `β${j}`,
        beta: b,
        se: ses[j],
        z,
        p: 2 * (1 - standardNormalCDF(Math.abs(z))),
      };
    });
  }, [fit]);

  // Residuals
  const residuals = useMemo(() => {
    if (!fit) return [];
    return residualKind === 'pearson' ? pearsonResiduals(fit) : devianceResiduals(fit);
  }, [fit, residualKind]);

  // Hovered coefficient profile-LRT CI
  const hoveredCI = useMemo(() => {
    if (!fit || hoveredCoef === null) return null;
    try {
      return coefCIProfileGLM(fit, hoveredCoef, 0.05, { maxBisect: 30 });
    } catch {
      return null;
    }
  }, [fit, hoveredCoef]);

  // Top-left scatter: (x_1, y) + fitted curve over a grid.
  // x_1 is row[1] (skip the intercept column).
  const xs1 = useMemo(() => X.map((row) => row[1] ?? 0), [X]);
  const xRange: [number, number] = useMemo(() => {
    const lo = Math.min(...xs1);
    const hi = Math.max(...xs1);
    const pad = (hi - lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }, [xs1]);
  const yRange: [number, number] = useMemo(() => {
    const lo = Math.min(...y);
    const hi = Math.max(...y);
    const pad = (hi - lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }, [y]);

  const panelW = isStacked ? width - 24 : (width - 280) / 2;
  const xScale = useMemo(
    () => d3.scaleLinear().domain(xRange).range([MARGIN.left, panelW - MARGIN.right]),
    [xRange, panelW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain(yRange).range([MARGIN.top + TOP_H, MARGIN.top]),
    [yRange],
  );

  // Build grid for fitted curve. Hold other predictors at their means.
  const fittedCurve = useMemo(() => {
    if (!fit) return { line: '', upper: '', lower: '' };
    const p = X[0].length;
    const means: number[] = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < X.length; i++) s += X[i][j];
      means[j] = s / X.length;
    }
    means[0] = 1; // intercept

    const M = 100;
    const Xnew: number[][] = [];
    const xs: number[] = [];
    for (let k = 0; k <= M; k++) {
      const t = xRange[0] + ((xRange[1] - xRange[0]) * k) / M;
      xs.push(t);
      const row = means.slice();
      row[1] = t;
      Xnew.push(row);
    }
    const offsetNew = isPoissonFreq && includeOffset
      ? new Array(Xnew.length).fill(EXAMPLE_7_GLM_DATA.offset.reduce((a, b) => a + b, 0) / EXAMPLE_7_GLM_DATA.offset.length)
      : undefined;
    const muHat = predictGLM(fit, Xnew, { offsetNew });
    const etaHat = predictGLM(fit, Xnew, { offsetNew, returnEta: true });

    // Pointwise SE for η via x^T V x.
    const seEta = Xnew.map((row) => {
      let s = 0;
      for (let j = 0; j < row.length; j++) {
        for (let k = 0; k < row.length; k++) {
          s += row[j] * fit.vcov[j][k] * row[k];
        }
      }
      return Math.sqrt(Math.max(s, 0));
    });
    const z = 1.96;
    const upperEta = etaHat.map((e, i) => e + z * seEta[i]);
    const lowerEta = etaHat.map((e, i) => e - z * seEta[i]);
    const upperMu = upperEta.map((e) => fit.link.gInv(e));
    const lowerMu = lowerEta.map((e) => fit.link.gInv(e));

    const line = d3.line<number>().x((_, i) => xScale(xs[i])).y((_, i) => yScale(muHat[i]))(muHat) ?? '';
    const upper = d3.line<number>().x((_, i) => xScale(xs[i])).y((_, i) => yScale(upperMu[i]))(upperMu) ?? '';
    const lower = d3.line<number>().x((_, i) => xScale(xs[i])).y((_, i) => yScale(lowerMu[i]))(lowerMu) ?? '';
    return { line, upper, lower };
  }, [fit, X, isPoissonFreq, includeOffset, xRange, xScale, yScale]);

  // Right panel: residuals vs fitted
  const fittedRange: [number, number] = useMemo(() => {
    if (!fit) return [0, 1];
    const lo = Math.min(...fit.mu);
    const hi = Math.max(...fit.mu);
    const pad = (hi - lo) * 0.05 || 0.1;
    return [lo - pad, hi + pad];
  }, [fit]);
  const residRange: [number, number] = useMemo(() => {
    if (residuals.length === 0) return [-1, 1];
    const lo = Math.min(...residuals);
    const hi = Math.max(...residuals);
    const pad = (hi - lo) * 0.1 || 0.5;
    return [lo - pad, hi + pad];
  }, [residuals]);

  const fittedXScale = useMemo(
    () => d3.scaleLinear().domain(fittedRange).range([MARGIN.left, panelW - MARGIN.right]),
    [fittedRange, panelW],
  );
  const residYScale = useMemo(
    () => d3.scaleLinear().domain(residRange).range([MARGIN.top + TOP_H, MARGIN.top]),
    [residRange],
  );

  const familyColor = glmFamilyColors[familyKey];

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-surface, #ffffff)',
      }}
    >
      {/* Preset bar */}
      <div
        className="flex flex-wrap gap-2 border-b px-3 py-2 text-sm"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        {(Object.keys(GLM_EXPLORER_PRESETS) as PresetKey[]).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => onPresetChange(key)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              presetKey === key
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            {PRESET_LABELS[key].label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 p-3 md:grid-cols-[1fr_240px]">
        <div>
          {/* Top row: scatter+fit, residuals */}
          <div className={`flex gap-3 ${isStacked ? 'flex-col' : 'flex-row'}`}>
            {/* Scatter + fitted curve */}
            <div className={isStacked ? 'w-full' : 'w-1/2'}>
              <p className="mb-1 text-xs text-slate-600">
                Fit:{' '}
                <span style={{ color: familyColor }}>
                  {familyKey} · {linkKey}
                </span>
              </p>
              <svg
                width="100%"
                viewBox={`0 0 ${panelW} ${MARGIN.top + TOP_H + 30}`}
                style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
              >
                {/* Axes */}
                <g transform={`translate(0, ${MARGIN.top + TOP_H})`}>
                  <line x1={MARGIN.left} x2={panelW - MARGIN.right} stroke={AXIS_COLOR} />
                  {xScale.ticks(5).map((t) => (
                    <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                      <line y1={0} y2={4} stroke={AXIS_COLOR} />
                      <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
                        {t.toFixed(1)}
                      </text>
                    </g>
                  ))}
                </g>
                <g>
                  <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + TOP_H} stroke={AXIS_COLOR} />
                  {yScale.ticks(5).map((t) => (
                    <g key={t} transform={`translate(${MARGIN.left}, ${yScale(t)})`}>
                      <line x1={-4} x2={0} stroke={AXIS_COLOR} />
                      <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
                        {Number.isFinite(t) ? t.toFixed(t > 100 ? 0 : 2) : ''}
                      </text>
                    </g>
                  ))}
                </g>
                <text
                  x={(MARGIN.left + panelW - MARGIN.right) / 2}
                  y={MARGIN.top + TOP_H + 28}
                  textAnchor="middle"
                  fontSize={11}
                  fill={AXIS_COLOR}
                >
                  x₁
                </text>

                {/* CI band */}
                {fittedCurve.upper && fittedCurve.lower && (
                  <g>
                    <path d={fittedCurve.upper} fill="none" stroke={familyColor} strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
                    <path d={fittedCurve.lower} fill="none" stroke={familyColor} strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
                  </g>
                )}
                {/* Fitted curve */}
                {fittedCurve.line && (
                  <path d={fittedCurve.line} fill="none" stroke={familyColor} strokeWidth={2.5} />
                )}

                {/* Data points */}
                {X.map((row, i) => (
                  <circle
                    key={i}
                    cx={xScale(row[1] ?? 0)}
                    cy={yScale(y[i])}
                    r={3}
                    fill="#1F2937"
                    opacity={0.55}
                  />
                ))}
              </svg>
            </div>

            {/* Residual plot */}
            <div className={isStacked ? 'w-full' : 'w-1/2'}>
              <div className="mb-1 flex items-center justify-between">
                <p className="text-xs text-slate-600">Residuals vs fitted</p>
                <div className="flex gap-1">
                  {(['pearson', 'deviance'] as const).map((k) => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setResidualKind(k)}
                      className={`rounded px-2 py-0.5 text-[10px] ${
                        residualKind === k
                          ? 'bg-slate-900 text-white'
                          : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {k}
                    </button>
                  ))}
                </div>
              </div>
              <svg
                width="100%"
                viewBox={`0 0 ${panelW} ${MARGIN.top + TOP_H + 30}`}
                style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
              >
                {/* zero line */}
                <line
                  x1={MARGIN.left}
                  x2={panelW - MARGIN.right}
                  y1={residYScale(0)}
                  y2={residYScale(0)}
                  stroke={AXIS_COLOR}
                  strokeDasharray="3 3"
                />
                {/* Axes */}
                <g transform={`translate(0, ${MARGIN.top + TOP_H})`}>
                  <line x1={MARGIN.left} x2={panelW - MARGIN.right} stroke={AXIS_COLOR} />
                  {fittedXScale.ticks(5).map((t) => (
                    <g key={t} transform={`translate(${fittedXScale(t)}, 0)`}>
                      <line y1={0} y2={4} stroke={AXIS_COLOR} />
                      <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
                        {t.toFixed(2)}
                      </text>
                    </g>
                  ))}
                </g>
                <g>
                  <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + TOP_H} stroke={AXIS_COLOR} />
                  {residYScale.ticks(5).map((t) => (
                    <g key={t} transform={`translate(${MARGIN.left}, ${residYScale(t)})`}>
                      <line x1={-4} x2={0} stroke={AXIS_COLOR} />
                      <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
                        {t.toFixed(1)}
                      </text>
                    </g>
                  ))}
                </g>
                <text
                  x={(MARGIN.left + panelW - MARGIN.right) / 2}
                  y={MARGIN.top + TOP_H + 28}
                  textAnchor="middle"
                  fontSize={11}
                  fill={AXIS_COLOR}
                >
                  fitted μ̂
                </text>

                {/* Residual points */}
                {fit &&
                  fit.mu.map((muI, i) => (
                    <circle
                      key={i}
                      cx={fittedXScale(muI)}
                      cy={residYScale(residuals[i] ?? 0)}
                      r={3}
                      fill={familyColor}
                      opacity={0.65}
                    />
                  ))}
              </svg>
            </div>
          </div>

          {/* Coefficient table */}
          <div className="mt-2 rounded border" style={{ borderColor: 'var(--color-border, #e5e7eb)' }}>
            <table className="w-full text-xs" style={{ fontVariantNumeric: 'tabular-nums' }}>
              <thead className="bg-slate-100 text-slate-700">
                <tr>
                  <th className="px-2 py-1 text-left">name</th>
                  <th className="px-2 py-1 text-right">β̂</th>
                  <th className="px-2 py-1 text-right">SE</th>
                  <th className="px-2 py-1 text-right">z</th>
                  <th className="px-2 py-1 text-right">p</th>
                </tr>
              </thead>
              <tbody>
                {coefTable.map((c) => (
                  <tr
                    key={c.j}
                    onMouseEnter={() => setHoveredCoef(c.j)}
                    onMouseLeave={() => setHoveredCoef(null)}
                    className={`cursor-pointer ${hoveredCoef === c.j ? 'bg-slate-50' : ''}`}
                  >
                    <td className="px-2 py-1">{c.name}</td>
                    <td className="px-2 py-1 text-right">{c.beta.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right">{c.se.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right">{c.z.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right">{formatP(c.p)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {hoveredCoef !== null && hoveredCI && (
              <div className="border-t bg-slate-50 px-2 py-1 text-[11px] text-slate-700">
                Profile-LRT 95% CI for β{hoveredCoef}: [
                {hoveredCI.lower.toFixed(3)}, {hoveredCI.upper.toFixed(3)}]
              </div>
            )}
          </div>
        </div>

        {/* Right rail controls */}
        <div
          className="flex flex-col gap-3 rounded-md p-3 text-sm"
          style={{
            backgroundColor: 'var(--color-surface-alt, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <div>
            <p className="mb-1 text-xs font-medium text-slate-700">Family</p>
            <div className="flex flex-col gap-1">
              {(['bernoulli', 'poisson', 'gamma', 'normal'] as FamilyKey[]).map((fk) => (
                <label key={fk} className="flex items-center gap-2 text-xs">
                  <input
                    type="radio"
                    name="family"
                    checked={familyKey === fk}
                    onChange={() => onFamilyChange(fk)}
                  />
                  <span
                    className="inline-block h-3 w-3 rounded-full"
                    style={{ backgroundColor: glmFamilyColors[fk] }}
                  />
                  <span>{fk}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-1 text-xs font-medium text-slate-700">Link</p>
            <select
              value={linkKey}
              onChange={(e) => setLinkKey(e.target.value)}
              className="w-full rounded border px-2 py-1 text-xs"
              style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
            >
              {FAMILY_LINKS[familyKey].map((lk) => (
                <option key={lk} value={lk}>
                  {lk}
                  {FAMILIES[familyKey].canonicalLink.name === lk ? ' (canonical)' : ''}
                </option>
              ))}
            </select>
          </div>

          {isPoissonFreq && (
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={includeOffset}
                onChange={(e) => setIncludeOffset(e.target.checked)}
              />
              <span>Include offset = log(policy-years)</span>
            </label>
          )}

          {fit && (
            <div className="border-t pt-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-600">Deviance D</span>
                <span>{fit.deviance.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">df</span>
                <span>{fit.X.length - fit.beta.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">φ̂</span>
                <span>{fit.phi.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">IRLS iter</span>
                <span>
                  {fit.nIter}
                  {fit.converged ? '' : ' (no conv)'}
                </span>
              </div>
            </div>
          )}
          <p className="text-[11px] leading-snug text-slate-600">
            Hover a coefficient row to compute its profile-LRT CI.
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatP(p: number): string {
  if (!Number.isFinite(p)) return '—';
  if (p < 1e-4) return p.toExponential(1);
  return p.toFixed(4);
}
