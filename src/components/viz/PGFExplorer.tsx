import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pgfBernoulli, pgfBinomial, pgfGeometric, pgfPoisson,
  expectationBernoulli, expectationBinomial, expectationGeometric, expectationPoisson,
  varianceBernoulli, varianceBinomial, varianceGeometric, variancePoisson,
} from './shared/distributions';
import { pgfPresets, compoundPresets } from '../../data/discrete-distributions-data';

// ── PGF dispatch ──────────────────────────────────────────────────────────

function evalPGF(dist: string, s: number, p: Record<string, number>): number {
  switch (dist) {
    case 'Bernoulli': return pgfBernoulli(s, p.p);
    case 'Binomial': return pgfBinomial(s, p.n, p.p);
    case 'Geometric': return pgfGeometric(s, p.p);
    case 'Poisson': return pgfPoisson(s, p.lambda);
    default: return 0;
  }
}

function getMean(dist: string, p: Record<string, number>): number {
  switch (dist) {
    case 'Bernoulli': return expectationBernoulli(p.p);
    case 'Binomial': return expectationBinomial(p.n, p.p);
    case 'Geometric': return expectationGeometric(p.p);
    case 'Poisson': return expectationPoisson(p.lambda);
    default: return 0;
  }
}

function getVar(dist: string, p: Record<string, number>): number {
  switch (dist) {
    case 'Bernoulli': return varianceBernoulli(p.p);
    case 'Binomial': return varianceBinomial(p.n, p.p);
    case 'Geometric': return varianceGeometric(p.p);
    case 'Poisson': return variancePoisson(p.lambda);
    default: return 0;
  }
}

// Numerical derivative via central difference
function numericalDerivative(f: (x: number) => number, x: number, h = 1e-6): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

function numericalSecondDerivative(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const N_POINTS = 150;

export default function PGFExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [mode, setMode] = useState<'single' | 'compound'>('single');
  const [compIdx, setCompIdx] = useState(0);

  const preset = pgfPresets[presetIdx];
  const compound = compoundPresets[compIdx];

  const panelW = Math.max(250, ((width || 600) - 32) / 2);
  const chartH = 260;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Single distribution mode ────────────────────────────────────────────

  const pgfCurve = useMemo(() => {
    return Array.from({ length: N_POINTS + 1 }, (_, i) => {
      const s = i / N_POINTS; // s in [0, 1]
      return { s, g: evalPGF(preset.distribution, s, preset.params) };
    });
  }, [preset]);

  const maxG = useMemo(() => Math.max(...pgfCurve.map((d) => d.g), 1.05), [pgfCurve]);

  // Moments via PGF derivatives at s=1
  const pgfFunc = useCallback(
    (s: number) => evalPGF(preset.distribution, s, preset.params),
    [preset],
  );
  const gPrime1 = numericalDerivative(pgfFunc, 1 - 1e-4); // G'(1) = E[X]
  const gDoublePrime1 = numericalSecondDerivative(pgfFunc, 1 - 1e-4); // G''(1) = E[X(X-1)]
  const varFromPGF = gDoublePrime1 + gPrime1 - gPrime1 * gPrime1;

  // Closed-form moments for comparison
  const closedMean = getMean(preset.distribution, preset.params);
  const closedVar = getVar(preset.distribution, preset.params);

  // Tangent line at s=1: y = G(1) + G'(1)(s - 1) = 1 + E[X](s - 1)
  const tangentAt = (s: number) => 1 + gPrime1 * (s - 1);

  // ── Compound mode ───────────────────────────────────────────────────────

  const compoundCurves = useMemo(() => {
    if (mode !== 'compound') return { gN: [], gX: [], gS: [] };
    const gN: { s: number; g: number }[] = [];
    const gX: { s: number; g: number }[] = [];
    const gS: { s: number; g: number }[] = [];

    for (let i = 0; i <= N_POINTS; i++) {
      const s = i / N_POINTS;
      const gXVal = evalPGF(compound.X.distribution, s, compound.X.params);
      const gNVal = evalPGF(compound.N.distribution, s, compound.N.params);
      const gSVal = evalPGF(compound.N.distribution, gXVal, compound.N.params); // G_N(G_X(s))
      gN.push({ s, g: gNVal });
      gX.push({ s, g: gXVal });
      gS.push({ s, g: gSVal });
    }
    return { gN, gX, gS };
  }, [mode, compound]);

  const maxGComp = useMemo(() => {
    if (mode !== 'compound') return 1.05;
    return Math.max(
      ...compoundCurves.gN.map((d) => d.g),
      ...compoundCurves.gX.map((d) => d.g),
      ...compoundCurves.gS.map((d) => d.g),
      1.05,
    );
  }, [mode, compoundCurves]);

  // Scales
  const xScale = useCallback(
    (s: number) => MARGIN.left + s * plotW,
    [plotW],
  );
  const yScale = useCallback(
    (g: number, maxVal: number) => MARGIN.top + plotH - (g / (maxVal * 1.1)) * plotH,
    [plotH],
  );

  const buildPath = (data: { s: number; g: number }[], maxVal: number) => {
    return data
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.s).toFixed(1)},${yScale(d.g, maxVal).toFixed(1)}`)
      .join(' ');
  };

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Probability-Generating Functions
      </div>

      {/* Mode toggle */}
      <div className="flex justify-center gap-2 mb-4">
        <button
          onClick={() => setMode('single')}
          className="px-3 py-1 text-xs rounded-full transition-colors"
          style={{
            backgroundColor: mode === 'single' ? 'var(--color-primary, #2563EB)' : 'transparent',
            color: mode === 'single' ? 'white' : 'currentColor',
            border: '1px solid var(--color-primary, #2563EB)',
          }}
        >
          Single Distribution
        </button>
        <button
          onClick={() => setMode('compound')}
          className="px-3 py-1 text-xs rounded-full transition-colors"
          style={{
            backgroundColor: mode === 'compound' ? '#7C3AED' : 'transparent',
            color: mode === 'compound' ? 'white' : 'currentColor',
            border: '1px solid #7C3AED',
          }}
        >
          Compound Distribution
        </button>
      </div>

      {mode === 'single' ? (
        <>
          {/* Distribution selector */}
          <div className="flex flex-wrap gap-1.5 justify-center mb-4">
            {pgfPresets.map((p, i) => (
              <button
                key={p.name}
                onClick={() => setPresetIdx(i)}
                className="px-2.5 py-1 text-xs rounded-full transition-colors"
                style={{
                  backgroundColor: i === presetIdx ? 'var(--color-primary, #2563EB)' : 'transparent',
                  color: i === presetIdx ? 'white' : 'currentColor',
                  border: '1px solid var(--color-border)',
                }}
              >
                {p.name}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-4 justify-center">
            {/* PGF curve */}
            <div>
              <div className="text-center text-xs font-medium mb-1">G(s) = E[s^X]</div>
              <svg width={panelW} height={chartH} className="block">
                {/* Grid */}
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={v}>
                    <line x1={MARGIN.left} y1={yScale(v, maxG)} x2={panelW - MARGIN.right} y2={yScale(v, maxG)} stroke="currentColor" strokeOpacity={0.08} />
                    <text x={MARGIN.left - 4} y={yScale(v, maxG) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                      {v.toFixed(2)}
                    </text>
                  </g>
                ))}

                {/* PGF curve */}
                <path d={buildPath(pgfCurve, maxG)} fill="none" stroke="var(--color-primary, #2563EB)" strokeWidth={2} />

                {/* Tangent at s=1 */}
                {(() => {
                  const s0 = Math.max(0, 1 - 0.3);
                  const s1 = 1;
                  return (
                    <line
                      x1={xScale(s0)} y1={yScale(tangentAt(s0), maxG)}
                      x2={xScale(s1)} y2={yScale(tangentAt(s1), maxG)}
                      stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4,3"
                    />
                  );
                })()}

                {/* G(1) = 1 marker */}
                <circle cx={xScale(1)} cy={yScale(1, maxG)} r={4} fill="var(--color-primary, #2563EB)" />

                {/* Slope annotation */}
                <text x={xScale(0.85)} y={yScale(tangentAt(0.85), maxG) - 6} className="fill-current" style={{ fontSize: '9px', fill: '#DC2626' }}>
                  slope = E[X] = {gPrime1.toFixed(3)}
                </text>

                {/* X axis labels */}
                {[0, 0.25, 0.5, 0.75, 1].map((s) => (
                  <text key={s} x={xScale(s)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                    {s.toFixed(2)}
                  </text>
                ))}
              </svg>
            </div>

            {/* Moments table */}
            <div className="flex flex-col justify-center">
              <div className="text-xs font-medium mb-2 text-center">Moments from PGF</div>
              <table className="text-xs">
                <thead>
                  <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                    <th className="px-3 py-1 text-left">Quantity</th>
                    <th className="px-3 py-1">PGF</th>
                    <th className="px-3 py-1">Closed-form</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                    <td className="px-3 py-0.5">G'(1) = E[X]</td>
                    <td className="px-3 py-0.5 tabular-nums">{gPrime1.toFixed(4)}</td>
                    <td className="px-3 py-0.5 tabular-nums">{closedMean.toFixed(4)}</td>
                  </tr>
                  <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                    <td className="px-3 py-0.5">G''(1) = E[X(X−1)]</td>
                    <td className="px-3 py-0.5 tabular-nums">{gDoublePrime1.toFixed(4)}</td>
                    <td className="px-3 py-0.5 tabular-nums opacity-50">—</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-0.5">Var(X)</td>
                    <td className="px-3 py-0.5 tabular-nums">{varFromPGF.toFixed(4)}</td>
                    <td className="px-3 py-0.5 tabular-nums">{closedVar.toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="text-[10px] text-gray-500 mt-2 text-center">
                Var = G''(1) + G'(1) − [G'(1)]²
              </div>
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Compound mode selector */}
          <div className="flex flex-wrap gap-1.5 justify-center mb-4">
            {compoundPresets.map((c, i) => (
              <button
                key={c.name}
                onClick={() => setCompIdx(i)}
                className="px-2.5 py-1 text-xs rounded-full transition-colors"
                style={{
                  backgroundColor: i === compIdx ? '#7C3AED' : 'transparent',
                  color: i === compIdx ? 'white' : 'currentColor',
                  border: '1px solid #7C3AED',
                }}
              >
                {c.name}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-4 justify-center">
            {/* Compound PGF chart */}
            <div>
              <div className="text-center text-xs font-medium mb-1">G_S(s) = G_N(G_X(s))</div>
              <svg width={panelW} height={chartH} className="block">
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={v}>
                    <line x1={MARGIN.left} y1={yScale(v, maxGComp)} x2={panelW - MARGIN.right} y2={yScale(v, maxGComp)} stroke="currentColor" strokeOpacity={0.08} />
                    <text x={MARGIN.left - 4} y={yScale(v, maxGComp) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                      {v.toFixed(2)}
                    </text>
                  </g>
                ))}

                {/* G_N(s) dashed */}
                <path d={buildPath(compoundCurves.gN, maxGComp)} fill="none" stroke="#059669" strokeWidth={1.5} strokeDasharray="4,3" />
                {/* G_X(s) dashed */}
                <path d={buildPath(compoundCurves.gX, maxGComp)} fill="none" stroke="#D97706" strokeWidth={1.5} strokeDasharray="4,3" />
                {/* G_S(s) solid */}
                <path d={buildPath(compoundCurves.gS, maxGComp)} fill="none" stroke="#7C3AED" strokeWidth={2.5} />

                {/* Legend */}
                <line x1={panelW - 140} y1={MARGIN.top + 8} x2={panelW - 120} y2={MARGIN.top + 8} stroke="#059669" strokeWidth={1.5} strokeDasharray="4,3" />
                <text x={panelW - 116} y={MARGIN.top + 12} className="fill-current" style={{ fontSize: '9px' }}>G_N(s)</text>
                <line x1={panelW - 140} y1={MARGIN.top + 22} x2={panelW - 120} y2={MARGIN.top + 22} stroke="#D97706" strokeWidth={1.5} strokeDasharray="4,3" />
                <text x={panelW - 116} y={MARGIN.top + 26} className="fill-current" style={{ fontSize: '9px' }}>G_X(s)</text>
                <line x1={panelW - 140} y1={MARGIN.top + 36} x2={panelW - 120} y2={MARGIN.top + 36} stroke="#7C3AED" strokeWidth={2.5} />
                <text x={panelW - 116} y={MARGIN.top + 40} className="fill-current" style={{ fontSize: '9px' }}>G_S(s)</text>

                {[0, 0.25, 0.5, 0.75, 1].map((s) => (
                  <text key={s} x={xScale(s)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                    {s.toFixed(2)}
                  </text>
                ))}
              </svg>
            </div>

            {/* Compound info */}
            <div className="flex flex-col justify-center text-xs">
              <div className="font-medium mb-2">Compound S = X₁ + ... + X_N</div>
              <div className="space-y-1">
                <div><b>N</b> ~ {compound.N.distribution}({Object.entries(compound.N.params).map(([k, v]) => `${k}=${v}`).join(', ')})</div>
                <div><b>Xᵢ</b> ~ {compound.X.distribution}({Object.entries(compound.X.params).map(([k, v]) => `${k}=${v}`).join(', ')})</div>
                <div className="mt-2 pt-2 border-t" style={{ borderColor: 'var(--color-border)' }}>
                  <b>Result:</b> <span style={{ color: '#7C3AED', fontWeight: 600 }}>{compound.result}</span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
