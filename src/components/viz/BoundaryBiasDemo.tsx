/**
 * BoundaryBiasDemo — Topic 30 §30.5 (optional).
 *
 * Two side-by-side KDE panels on a bounded-support preset:
 *   Left  — naive Gaussian KDE, which smooths across the support boundary
 *           and places mass where the true density is zero. The "mass leaked
 *           below 0" annotation computes this numerically for Beta-on-[0,1].
 *   Right — reflection-corrected KDE (Jones 1993's simplest scheme): reflect
 *           each Xᵢ to 2c − Xᵢ at each support boundary c, then compute the
 *           KDE on the augmented sample and restrict to [a, b]. The factor-of-2
 *           normalization falls out of symmetry.
 *
 * Presets: Beta(2, 5), Beta(½, ½) (arcsine — diverges at both ends),
 * and Exp(1) (single boundary at 0). Ships optional per brief §5.4; Fig 4
 * carries the pedagogy even if the component isn't rendered.
 */
import { useMemo, useState } from 'react';

import {
  gaussianKernel,
  kdeEvaluate,
  silvermanBandwidth,
} from './shared/nonparametric';
import { betaPreset, expPreset, seededUniform, type DistributionPreset } from '../../data/nonparametric-data';
import { kdeColors } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Inline Beta(½, ½) preset — the arcsine distribution. f diverges at both
// boundaries so the boundary-bias effect is extreme.
// ────────────────────────────────────────────────────────────────────────────

const beta05Preset: DistributionPreset = {
  name: 'Beta(½, ½)',
  key: 'beta',
  pdf: (x) => (x > 0 && x < 1 ? 1 / (Math.PI * Math.sqrt(x * (1 - x))) : 0),
  cdf: (x) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return (2 / Math.PI) * Math.asin(Math.sqrt(x));
  },
  quantile: (p) => Math.sin((Math.PI * p) / 2) ** 2,
  sampler: (rng) => Math.sin((Math.PI * rng()) / 2) ** 2,
  domain: [0, 1],
  support: [0, 1],
};

const PRESETS: ReadonlyArray<DistributionPreset & { id: string }> = [
  { ...betaPreset, id: 'beta25' },
  { ...beta05Preset, id: 'beta05' },
  { ...expPreset, id: 'exp' },
];

const NS = [100, 200, 500, 1000] as const;

// Reflect the sample about a support boundary and return a second KDE
// contribution at x. Total reflected-KDE is the naive KDE at x plus these.
function kdeReflected(
  sample: number[],
  x: number,
  h: number,
  support: [number, number],
): number {
  if (x < support[0] || x > support[1]) return 0;
  let val = kdeEvaluate(sample, x, h, gaussianKernel);
  if (Number.isFinite(support[0])) {
    const refl = sample.map((xi) => 2 * support[0] - xi);
    val += kdeEvaluate(refl, x, h, gaussianKernel);
  }
  if (Number.isFinite(support[1])) {
    const refl = sample.map((xi) => 2 * support[1] - xi);
    val += kdeEvaluate(refl, x, h, gaussianKernel);
  }
  return val;
}

const WIDTH = 340;
const HEIGHT = 280;
const MARGIN = { top: 20, right: 14, bottom: 40, left: 44 };
const GRID_POINTS = 250;

function linspace(a: number, b: number, n: number): number[] {
  const step = (b - a) / (n - 1);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return out;
}

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

export default function BoundaryBiasDemo() {
  const [presetId, setPresetId] = useState('beta25');
  const [n, setN] = useState<(typeof NS)[number]>(500);
  const [seed, setSeed] = useState(42);
  const [hScale, setHScale] = useState(1.0); // multiplier on Silverman's
  const [showReflection, setShowReflection] = useState(true);

  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0];
  const support = preset.support ?? [-Infinity, Infinity];

  const sample = useMemo(() => {
    const rng = seededUniform(seed);
    const out: number[] = new Array(n);
    for (let i = 0; i < n; i++) out[i] = preset.sampler(rng);
    return out;
  }, [preset, n, seed]);

  const h = useMemo(() => silvermanBandwidth(sample) * hScale, [sample, hScale]);

  // Plot extends beyond the support so the reader can see leaked mass.
  const [plotMin, plotMax] = useMemo(() => {
    const pad = 3 * h;
    return [Math.max(preset.domain[0] - pad, -2), Math.min(preset.domain[1] + pad, 10)];
  }, [preset.domain, h]);

  const curves = useMemo(() => {
    const grid = linspace(plotMin, plotMax, GRID_POINTS);
    const truth = grid.map((x) => preset.pdf(x));
    const naive = grid.map((x) => kdeEvaluate(sample, x, h, gaussianKernel));
    const reflected = grid.map((x) => kdeReflected(sample, x, h, support));
    let yMax = 0;
    for (const arr of [truth, naive, reflected]) {
      for (const v of arr) if (v > yMax) yMax = v;
    }
    return { grid, truth, naive, reflected, yMax: yMax * 1.1 };
  }, [sample, h, support, preset, plotMin, plotMax]);

  // Numerically integrate the naive KDE's mass outside [support[0], support[1]].
  const leaked = useMemo(() => {
    const dx = curves.grid[1] - curves.grid[0];
    let below = 0;
    let above = 0;
    for (let i = 0; i < curves.grid.length; i++) {
      const x = curves.grid[i];
      if (x < support[0]) below += curves.naive[i] * dx;
      else if (x > support[1]) above += curves.naive[i] * dx;
    }
    return { below, above, total: below + above };
  }, [curves, support]);

  const xScale = scaleLinear(plotMin, plotMax, MARGIN.left, WIDTH - MARGIN.right);
  const yScale = scaleLinear(0, curves.yMax, HEIGHT - MARGIN.bottom, MARGIN.top);

  const pathFor = (ys: number[]): string => {
    const pts = curves.grid.map((x, i) => `${xScale(x).toFixed(1)},${yScale(ys[i]).toFixed(1)}`);
    return 'M' + pts.join('L');
  };

  const renderPanel = (
    title: string,
    ys: number[],
    stroke: string,
    showLeakShading: boolean,
  ) => (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="w-full h-auto max-w-full"
      role="img"
      aria-label={`${title} KDE panel`}
    >
      {/* Shade leaked-mass region under naive KDE, if any */}
      {showLeakShading && Number.isFinite(support[0]) && (
        <rect
          x={xScale(plotMin)}
          y={MARGIN.top}
          width={xScale(support[0]) - xScale(plotMin)}
          height={HEIGHT - MARGIN.bottom - MARGIN.top}
          fill={kdeColors.bias}
          opacity={0.08}
        />
      )}
      {showLeakShading && Number.isFinite(support[1]) && (
        <rect
          x={xScale(support[1])}
          y={MARGIN.top}
          width={xScale(plotMax) - xScale(support[1])}
          height={HEIGHT - MARGIN.bottom - MARGIN.top}
          fill={kdeColors.bias}
          opacity={0.08}
        />
      )}
      {/* Support boundaries */}
      {Number.isFinite(support[0]) && (
        <line
          x1={xScale(support[0])}
          y1={MARGIN.top}
          x2={xScale(support[0])}
          y2={HEIGHT - MARGIN.bottom}
          stroke="#94a3b8"
          strokeDasharray="2 3"
        />
      )}
      {Number.isFinite(support[1]) && (
        <line
          x1={xScale(support[1])}
          y1={MARGIN.top}
          x2={xScale(support[1])}
          y2={HEIGHT - MARGIN.bottom}
          stroke="#94a3b8"
          strokeDasharray="2 3"
        />
      )}
      {/* Axes */}
      <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
      <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
      {linspace(plotMin, plotMax, 5).map((x, i) => (
        <g key={`xt-${i}`}>
          <text x={xScale(x)} y={HEIGHT - MARGIN.bottom + 16} textAnchor="middle" className="text-[10px] fill-slate-600">
            {x.toFixed(1)}
          </text>
        </g>
      ))}
      {/* True density */}
      <path d={pathFor(curves.truth)} fill="none" stroke="#0f172a" strokeWidth={2} opacity={0.9} />
      {/* KDE curve (naive or reflected) */}
      <path d={pathFor(ys)} fill="none" stroke={stroke} strokeWidth={2} />
      <text x={MARGIN.left} y={MARGIN.top - 4} className="text-[11px] fill-slate-500">{title}</text>
    </svg>
  );

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6">
      <div className="mb-3 text-sm text-slate-600">
        <span className="font-semibold text-slate-800">Boundary-bias demo</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          Densities on bounded support leak KDE mass across the edge. Jones 1993 reflection patches it.
        </span>
      </div>

      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <div className="text-xs text-slate-500 mb-1">Density</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => setPresetId(p.id)}
                className={`px-2.5 py-1 text-xs rounded ${
                  presetId === p.id
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">n</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5">
            {NS.map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setN(v)}
                className={`px-2 py-1 text-xs rounded ${
                  n === v
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">h = {fmt(h, 3)} (× Silverman)</div>
          <input
            type="range"
            min={0.3}
            max={2.5}
            step={0.01}
            value={hScale}
            onChange={(e) => setHScale(Number(e.target.value))}
            className="w-36"
            aria-label="Bandwidth scale"
          />
        </div>

        <label className="inline-flex items-center gap-1.5 text-xs text-slate-600">
          <input
            type="checkbox"
            checked={showReflection}
            onChange={(e) => setShowReflection(e.target.checked)}
          />
          show reflection panel
        </label>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700"
        >
          Resample
        </button>
      </div>

      <div className={`grid gap-4 ${showReflection ? 'md:grid-cols-2' : 'md:grid-cols-1'}`}>
        <div>
          {renderPanel('Naive Gaussian KDE', curves.naive, kdeColors.kde, true)}
          <div className="mt-1.5 text-[11px] text-slate-600 tabular-nums">
            Mass leaked outside support:{' '}
            <span className="font-medium" style={{ color: kdeColors.bias }}>
              {fmt(leaked.total, 4)}
            </span>
            {leaked.below > 0 && <> &nbsp;·&nbsp; below: {fmt(leaked.below, 4)}</>}
            {leaked.above > 0 && <> &nbsp;·&nbsp; above: {fmt(leaked.above, 4)}</>}
          </div>
        </div>
        {showReflection && (
          <div>
            {renderPanel(
              'Reflection-corrected KDE',
              curves.reflected,
              kdeColors.kernels.epanechnikov,
              false,
            )}
            <div className="mt-1.5 text-[11px] text-slate-600">
              Reflection about the support boundaries; mass restored to [{preset.domain[0].toFixed(0)},{' '}
              {preset.domain[1].toFixed(0)}].
            </div>
          </div>
        )}
      </div>

      <p className="mt-3 text-xs text-slate-500 leading-relaxed">
        Reflection is the simplest of the boundary fixes — Fan–Gijbels 1996's local-linear estimator
        is the modern treatment, delivering the correction and the regression extension together.
        See the Topic 30 §30.10 pointer to formalml <span className="italic">local-regression</span>.
      </p>
    </div>
  );
}
