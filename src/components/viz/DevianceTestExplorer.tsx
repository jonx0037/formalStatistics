import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  FAMILIES,
  glmFit,
  devianceTestNested,
  coefCIProfileGLM,
  coefCIWaldSandwich,
  type GLMFamily,
  type LinkFunction,
} from './shared/regression';
import {
  chiSquaredPDF,
  chiSquaredInvCDF,
  nonCentralChiSquaredCDF,
} from './shared/testing';
import {
  DEVIANCE_TEST_EXPLORER_PRESETS,
  EXAMPLE_8_GLM_DATA,
  EXAMPLE_6_GLM_DATA,
} from '../../data/regression-data';
import { glmFamilyColors } from './shared/colorScales';

const MARGIN = { top: 24, right: 24, bottom: 44, left: 56 };
const PANEL_H = 280;
const AXIS_COLOR = '#9CA3AF';
const NULL_COLOR = '#2563EB';
const REJECT_COLOR = '#DC2626';
const OBSERVED_COLOR = '#059669';
const PROFILE_COLOR = '#059669';
const WALD_COLOR = '#2563EB';

type FamilyKey = 'bernoulli' | 'poisson' | 'gamma';
type PresetKey = keyof typeof DEVIANCE_TEST_EXPLORER_PRESETS;

const PRESET_LABELS: Record<PresetKey, string> = {
  nullTrue: 'H₀ true (β₂ = 0)',
  smallEffect: 'Small effect (β₂ = 0.2)',
  mediumEffect: 'Medium effect (β₂ = 0.5)',
  largeEffect: 'Large effect (β₂ = 1.0)',
};

export default function DevianceTestExplorer() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 800;

  const [presetKey, setPresetKey] = useState<PresetKey>('mediumEffect');
  const [familyKey, setFamilyKey] = useState<FamilyKey>('poisson');
  const [k, setK] = useState<number>(1);
  const [alpha, setAlpha] = useState<number>(0.05);

  const family: GLMFamily = FAMILIES[familyKey];
  const link: LinkFunction = family.canonicalLink;

  // Preset gives Poisson nested data; for non-Poisson, fall back to a
  // family-appropriate dataset (Bernoulli → EXAMPLE_6 with intercept reduced;
  // Gamma → EXAMPLE_8 with intercept reduced).
  const dataset = useMemo(() => {
    if (familyKey === 'poisson') {
      const p = DEVIANCE_TEST_EXPLORER_PRESETS[presetKey];
      return { Xfull: p.Xfull, Xreduced: p.Xreduced, y: p.y, offset: p.offset };
    }
    if (familyKey === 'bernoulli') {
      const X = EXAMPLE_6_GLM_DATA.X;
      const Xreduced = X.map((row) => [row[0], row[1]]); // drop col 2 (income_z)
      return { Xfull: X, Xreduced, y: EXAMPLE_6_GLM_DATA.y, offset: new Array(X.length).fill(0) };
    }
    // gamma
    const X = EXAMPLE_8_GLM_DATA.X;
    const Xreduced = X.map((row) => [row[0]]);
    return { Xfull: X, Xreduced, y: EXAMPLE_8_GLM_DATA.y, offset: new Array(X.length).fill(0) };
  }, [familyKey, presetKey]);

  // Effective k = number of dropped columns (read from data). Slider lets the
  // user override the χ² df axis to see the test under a hypothetical k.
  const dataK = dataset.Xfull[0].length - dataset.Xreduced[0].length;

  // Run the deviance test
  const testResult = useMemo(() => {
    try {
      return devianceTestNested(dataset.Xfull, dataset.Xreduced, dataset.y, family, link, alpha);
    } catch {
      return null;
    }
  }, [dataset, family, link, alpha]);

  // Full fit (for profile vs Wald CI on β_j where j = first dropped index)
  const fullFit = useMemo(() => {
    try {
      return glmFit(dataset.Xfull, dataset.y, family, link, { offset: dataset.offset });
    } catch {
      return null;
    }
  }, [dataset, family, link]);

  // Profile and Wald CIs for the *first* tested coefficient.
  const targetJ = dataset.Xreduced[0].length; // first index that is in full but not reduced

  const waldCI = useMemo(() => {
    if (!fullFit) return null;
    try {
      const cis = coefCIWaldSandwich(fullFit, alpha, 'HC0');
      return cis[targetJ] ?? null;
    } catch {
      return null;
    }
  }, [fullFit, targetJ, alpha]);

  const profileCI = useMemo(() => {
    if (!fullFit) return null;
    try {
      return coefCIProfileGLM(fullFit, targetJ, alpha, { maxBisect: 30 });
    } catch {
      return null;
    }
  }, [fullFit, targetJ, alpha]);

  const panelW = isStacked ? width - 24 : (width - 36) / 3;

  // ── PANEL 1: χ²_k null density + observed deviance + rejection region ────
  const chi2Critical = useMemo(() => chiSquaredInvCDF(1 - alpha, k), [alpha, k]);
  const observedDiff = testResult?.diff ?? 0;

  const chi2Domain: [number, number] = useMemo(() => {
    const xMax = Math.max(chi2Critical * 2.5, observedDiff * 1.4, 12);
    return [0, xMax];
  }, [chi2Critical, observedDiff]);

  const chi2Density = useMemo(() => {
    const N = 200;
    const out: { x: number; y: number }[] = [];
    for (let i = 0; i <= N; i++) {
      const x = chi2Domain[0] + ((chi2Domain[1] - chi2Domain[0]) * i) / N;
      out.push({ x, y: chiSquaredPDF(x, k) });
    }
    return out;
  }, [chi2Domain, k]);

  const chi2YMax = Math.max(...chi2Density.map((d) => d.y), 0.05);

  const chi2XScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(chi2Domain)
        .range([MARGIN.left, panelW - MARGIN.right]),
    [chi2Domain, panelW],
  );
  const chi2YScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, chi2YMax * 1.1])
        .range([MARGIN.top + PANEL_H, MARGIN.top]),
    [chi2YMax],
  );

  const chi2DensityPath = useMemo(() => {
    return d3.line<{ x: number; y: number }>()
      .x((d) => chi2XScale(d.x))
      .y((d) => chi2YScale(d.y))(chi2Density);
  }, [chi2Density, chi2XScale, chi2YScale]);

  const chi2RejectPath = useMemo(() => {
    const startIdx = chi2Density.findIndex((d) => d.x >= chi2Critical);
    if (startIdx < 0) return '';
    const pts: { x: number; y: number }[] = [{ x: chi2Critical, y: 0 }];
    for (let i = startIdx; i < chi2Density.length; i++) pts.push(chi2Density[i]);
    pts.push({ x: chi2Density[chi2Density.length - 1].x, y: 0 });
    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => chi2XScale(d.x))
      .y((d) => chi2YScale(d.y));
    return (line(pts) ?? '') + ' Z';
  }, [chi2Density, chi2Critical, chi2XScale, chi2YScale]);

  // ── PANEL 2: power curve P(reject | λ) for k df ──────────────────────────
  const lambdaDomain: [number, number] = [0, 20];
  const powerCurve = useMemo(() => {
    const N = 80;
    const out: { x: number; y: number }[] = [];
    const crit = chiSquaredInvCDF(1 - alpha, k);
    for (let i = 0; i <= N; i++) {
      const lam = lambdaDomain[0] + ((lambdaDomain[1] - lambdaDomain[0]) * i) / N;
      const cdf = nonCentralChiSquaredCDF(crit, k, lam);
      out.push({ x: lam, y: 1 - cdf });
    }
    return out;
  }, [alpha, k]);

  const powerXScale = useMemo(
    () => d3.scaleLinear().domain(lambdaDomain).range([MARGIN.left, panelW - MARGIN.right]),
    [panelW],
  );
  const powerYScale = useMemo(
    () => d3.scaleLinear().domain([0, 1]).range([MARGIN.top + PANEL_H, MARGIN.top]),
    [],
  );
  const powerPath = useMemo(
    () =>
      d3
        .line<{ x: number; y: number }>()
        .x((d) => powerXScale(d.x))
        .y((d) => powerYScale(d.y))(powerCurve),
    [powerCurve, powerXScale, powerYScale],
  );

  // ── PANEL 3: profile likelihood D(β_j^0) vs β_j^0 ─────────────────────────
  const profileCurve = useMemo(() => {
    if (!fullFit) return [];
    const center = fullFit.beta[targetJ] ?? 0;
    const seWald = Math.sqrt(Math.max(fullFit.vcov[targetJ]?.[targetJ] ?? 0, 1e-12));
    const lo = center - 4 * seWald;
    const hi = center + 4 * seWald;
    const N = 30;
    // Loop invariants — pulled out of the bj loop to avoid 30 redundant
    // O(n·p) array allocations per render (PR #25 perf review).
    const Xred = fullFit.X.map((row) => row.filter((_, kk) => kk !== targetJ));
    const baseOff = fullFit.offset ?? new Array<number>(fullFit.X.length).fill(0);
    const startBeta = fullFit.beta.filter((_, kk) => kk !== targetJ);
    const out: { x: number; y: number }[] = [];
    for (let i = 0; i <= N; i++) {
      const bj = lo + ((hi - lo) * i) / N;
      // Profile deviance: refit with β_j fixed at bj.
      try {
        const newOff = baseOff.map((o, ii) => o + bj * fullFit.X[ii][targetJ]);
        const refit = glmFit(Xred, fullFit.y, family, link, {
          offset: newOff,
          startBeta,
          maxIter: 30,
          skipNullDeviance: true,
        });
        out.push({ x: bj, y: refit.deviance });
      } catch {
        out.push({ x: bj, y: NaN });
      }
    }
    return out;
  }, [fullFit, targetJ, family, link]);

  const profileDomain: [number, number] = useMemo(() => {
    if (profileCurve.length === 0) return [-1, 1];
    const xs = profileCurve.map((d) => d.x);
    return [Math.min(...xs), Math.max(...xs)];
  }, [profileCurve]);

  const profileYDomain: [number, number] = useMemo(() => {
    if (profileCurve.length === 0 || !fullFit) return [0, 1];
    const ys = profileCurve.map((d) => d.y).filter((v) => Number.isFinite(v));
    if (ys.length === 0) return [0, 1];
    const lo = Math.min(...ys, fullFit.deviance);
    const hi = Math.max(...ys, fullFit.deviance + chiSquaredInvCDF(1 - alpha, 1) * 1.5);
    return [lo - 0.5, hi + 0.5];
  }, [profileCurve, fullFit, alpha]);

  const profileXScale = useMemo(
    () => d3.scaleLinear().domain(profileDomain).range([MARGIN.left, panelW - MARGIN.right]),
    [profileDomain, panelW],
  );
  const profileYScale = useMemo(
    () => d3.scaleLinear().domain(profileYDomain).range([MARGIN.top + PANEL_H, MARGIN.top]),
    [profileYDomain],
  );
  const profilePath = useMemo(() => {
    const line = d3
      .line<{ x: number; y: number }>()
      .defined((d) => Number.isFinite(d.y))
      .x((d) => profileXScale(d.x))
      .y((d) => profileYScale(d.y));
    return line(profileCurve);
  }, [profileCurve, profileXScale, profileYScale]);

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
        {(Object.keys(DEVIANCE_TEST_EXPLORER_PRESETS) as PresetKey[]).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setPresetKey(key)}
            disabled={familyKey !== 'poisson'}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              presetKey === key && familyKey === 'poisson'
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200 disabled:opacity-50'
            }`}
          >
            {PRESET_LABELS[key]}
          </button>
        ))}
      </div>

      <div className={`flex gap-3 p-3 ${isStacked ? 'flex-col' : 'flex-row'}`}>
        {/* PANEL 1: χ² null density */}
        <Panel title={`χ²_${k} null + observed Δ`} width={panelW}>
          {chi2RejectPath && (
            <path d={chi2RejectPath} fill={REJECT_COLOR} opacity={0.25} />
          )}
          {chi2DensityPath && (
            <path d={chi2DensityPath} fill="none" stroke={NULL_COLOR} strokeWidth={2.5} />
          )}
          {/* Critical line */}
          <line
            x1={chi2XScale(chi2Critical)}
            x2={chi2XScale(chi2Critical)}
            y1={MARGIN.top}
            y2={MARGIN.top + PANEL_H}
            stroke={REJECT_COLOR}
            strokeDasharray="4 4"
          />
          <text
            x={chi2XScale(chi2Critical)}
            y={MARGIN.top + 12}
            textAnchor="middle"
            fontSize={10}
            fill={REJECT_COLOR}
          >
            χ²_crit = {chi2Critical.toFixed(2)}
          </text>
          {/* Observed Δ */}
          {testResult && (
            <>
              <line
                x1={chi2XScale(Math.min(observedDiff, chi2Domain[1]))}
                x2={chi2XScale(Math.min(observedDiff, chi2Domain[1]))}
                y1={MARGIN.top}
                y2={MARGIN.top + PANEL_H}
                stroke={OBSERVED_COLOR}
                strokeWidth={2}
              />
              <text
                x={chi2XScale(Math.min(observedDiff, chi2Domain[1]))}
                y={MARGIN.top + 26}
                textAnchor="middle"
                fontSize={10}
                fill={OBSERVED_COLOR}
              >
                Δ = {observedDiff.toFixed(2)}
              </text>
            </>
          )}
          <Axes
            xScale={chi2XScale}
            yScale={chi2YScale}
            panelW={panelW}
            xLabel={`χ²_${k}`}
            yTickFormat={(v) => v.toFixed(2)}
          />
        </Panel>

        {/* PANEL 2: power curve */}
        <Panel title={`Power vs λ (k=${k}, α=${alpha.toFixed(3)})`} width={panelW}>
          {powerPath && (
            <path d={powerPath} fill="none" stroke={NULL_COLOR} strokeWidth={2.5} />
          )}
          {/* α horizontal line */}
          <line
            x1={powerXScale(0)}
            x2={powerXScale(lambdaDomain[1])}
            y1={powerYScale(alpha)}
            y2={powerYScale(alpha)}
            stroke={REJECT_COLOR}
            strokeDasharray="3 3"
          />
          <text
            x={powerXScale(lambdaDomain[1]) - 4}
            y={powerYScale(alpha) - 4}
            textAnchor="end"
            fontSize={10}
            fill={REJECT_COLOR}
          >
            α = {alpha.toFixed(3)}
          </text>
          <Axes
            xScale={powerXScale}
            yScale={powerYScale}
            panelW={panelW}
            xLabel="λ"
            yTickFormat={(v) => v.toFixed(2)}
          />
        </Panel>

        {/* PANEL 3: profile vs Wald CI */}
        <Panel title={`Profile D(β_j) vs Wald-z (j=${targetJ})`} width={panelW}>
          {/* Threshold line at D(β̂) + χ²_{1, 1-α} */}
          {fullFit && (
            <>
              <line
                x1={profileXScale(profileDomain[0])}
                x2={profileXScale(profileDomain[1])}
                y1={profileYScale(fullFit.deviance + chiSquaredInvCDF(1 - alpha, 1))}
                y2={profileYScale(fullFit.deviance + chiSquaredInvCDF(1 - alpha, 1))}
                stroke={REJECT_COLOR}
                strokeDasharray="4 4"
              />
              <text
                x={profileXScale(profileDomain[1]) - 4}
                y={profileYScale(fullFit.deviance + chiSquaredInvCDF(1 - alpha, 1)) - 4}
                textAnchor="end"
                fontSize={10}
                fill={REJECT_COLOR}
              >
                D̂ + χ²₁,₁₋ₐ
              </text>
            </>
          )}
          {profilePath && (
            <path d={profilePath} fill="none" stroke={PROFILE_COLOR} strokeWidth={2.5} />
          )}
          {/* β̂ vertical */}
          {fullFit && (
            <line
              x1={profileXScale(fullFit.beta[targetJ])}
              x2={profileXScale(fullFit.beta[targetJ])}
              y1={MARGIN.top}
              y2={MARGIN.top + PANEL_H}
              stroke={AXIS_COLOR}
              strokeDasharray="2 4"
            />
          )}
          {/* Profile CI */}
          {profileCI && (
            <line
              x1={profileXScale(profileCI.lower)}
              x2={profileXScale(profileCI.upper)}
              y1={MARGIN.top + PANEL_H - 14}
              y2={MARGIN.top + PANEL_H - 14}
              stroke={PROFILE_COLOR}
              strokeWidth={4}
            />
          )}
          {/* Wald CI */}
          {waldCI && (
            <line
              x1={profileXScale(waldCI.lower)}
              x2={profileXScale(waldCI.upper)}
              y1={MARGIN.top + PANEL_H - 4}
              y2={MARGIN.top + PANEL_H - 4}
              stroke={WALD_COLOR}
              strokeWidth={4}
            />
          )}
          <Axes
            xScale={profileXScale}
            yScale={profileYScale}
            panelW={panelW}
            xLabel={`β${targetJ}`}
            yTickFormat={(v) => v.toFixed(0)}
          />
        </Panel>
      </div>

      {/* Controls + readout */}
      <div
        className="grid gap-3 border-t p-3 text-sm md:grid-cols-[1fr_240px]"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-xs">
              <span className="text-slate-700">Family:</span>
              <select
                value={familyKey}
                onChange={(e) => setFamilyKey(e.target.value as FamilyKey)}
                className="rounded border px-2 py-0.5 text-xs"
                style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
              >
                <option value="bernoulli">Bernoulli</option>
                <option value="poisson">Poisson</option>
                <option value="gamma">Gamma</option>
              </select>
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: familyColor }}
              />
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="text-slate-700">k = {k}</span>
              <input
                type="range"
                min={1}
                max={4}
                step={1}
                value={k}
                onChange={(e) => setK(parseInt(e.target.value, 10))}
              />
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="text-slate-700">α</span>
              <select
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="rounded border px-2 py-0.5 text-xs"
                style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
              >
                <option value={0.01}>0.01</option>
                <option value={0.05}>0.05</option>
                <option value={0.10}>0.10</option>
              </select>
            </label>
          </div>
          <p className="text-[11px] text-slate-600">
            Data k = {dataK} (slider just changes the χ²_k reference axis).
          </p>
        </div>

        <div
          className="flex flex-col gap-1 rounded-md p-2 text-xs"
          style={{
            backgroundColor: 'var(--color-surface-alt, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {testResult && (
            <>
              <div className="flex justify-between">
                <span className="text-slate-600">D₀ (reduced)</span>
                <span>{testResult.devianceReduced.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">D₁ (full)</span>
                <span>{testResult.devianceFull.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Δ on {testResult.df} df</span>
                <span>{testResult.diff.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">p-value</span>
                <span>{testResult.pValue.toExponential(2)}</span>
              </div>
            </>
          )}
          {profileCI && (
            <div className="border-t pt-1">
              <span className="text-slate-600">Profile CI:</span>{' '}
              [{profileCI.lower.toFixed(3)}, {profileCI.upper.toFixed(3)}]
            </div>
          )}
          {waldCI && (
            <div>
              <span className="text-slate-600">Wald CI:</span>{' '}
              [{waldCI.lower.toFixed(3)}, {waldCI.upper.toFixed(3)}]
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Subcomponents ────────────────────────────────────────────────────────────

function Panel({ title, width, children }: { title: string; width: number; children: React.ReactNode }) {
  return (
    <div className="flex-1">
      <p className="mb-1 text-xs text-slate-600">{title}</p>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${MARGIN.top + PANEL_H + 30}`}
        style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
      >
        {children}
      </svg>
    </div>
  );
}

function Axes({
  xScale,
  yScale,
  panelW,
  xLabel,
  yTickFormat,
}: {
  xScale: d3.ScaleLinear<number, number>;
  yScale: d3.ScaleLinear<number, number>;
  panelW: number;
  xLabel: string;
  yTickFormat?: (v: number) => string;
}) {
  return (
    <>
      <g transform={`translate(0, ${MARGIN.top + PANEL_H})`}>
        <line x1={MARGIN.left} x2={panelW - MARGIN.right} stroke={AXIS_COLOR} />
        {xScale.ticks(5).map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, 0)`}>
            <line y1={0} y2={4} stroke={AXIS_COLOR} />
            <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
              {t.toFixed(t > 100 ? 0 : 2)}
            </text>
          </g>
        ))}
      </g>
      <g>
        <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + PANEL_H} stroke={AXIS_COLOR} />
        {yScale.ticks(5).map((t) => (
          <g key={t} transform={`translate(${MARGIN.left}, ${yScale(t)})`}>
            <line x1={-4} x2={0} stroke={AXIS_COLOR} />
            <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
              {yTickFormat ? yTickFormat(t) : t.toFixed(2)}
            </text>
          </g>
        ))}
      </g>
      <text
        x={(MARGIN.left + panelW - MARGIN.right) / 2}
        y={MARGIN.top + PANEL_H + 28}
        textAnchor="middle"
        fontSize={11}
        fill={AXIS_COLOR}
      >
        {xLabel}
      </text>
    </>
  );
}
