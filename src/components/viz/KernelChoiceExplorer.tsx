/**
 * KernelChoiceExplorer — Topic 30 §30.3.
 *
 * Five standard kernels overlaid on a single sample at the same bandwidth:
 * the pedagogical payoff is that the KDE curves are visually near-identical,
 * sensitizing the reader to kernel-choice negligibility. A side table carries
 * the closed-form kernel constants (μ₂(K), R(K), efficiency) and the current
 * AMISE(h) for each visible kernel — so the reader can also see numerically
 * why Epanechnikov is the theoretical optimum but only by fractions of a percent.
 *
 * Two bandwidth modes:
 *   • equal h   — a single slider drives every kernel at the same bandwidth
 *                 (default h = Silverman's rule on the current sample).
 *   • canonical — each kernel gets its own AMISE-optimal h* from the oracle
 *                 formula h* = (R(K) / (n μ₂² R(f")))^{1/5}. No slider.
 *
 * Uses the direct-React-ref SVG + Tailwind pattern established by Topic 29's
 * QuantileAsymptoticsExplorer — no D3 selections, no useD3 hook.
 */
import { useMemo, useState } from 'react';

import {
  gaussianKernel,
  epanechnikovKernel,
  biweightKernel,
  triangularKernel,
  uniformKernel,
  kernelProperties,
  kdeEvaluateGrid,
  silvermanBandwidth,
  amiseOptimalBandwidth,
  type KernelFn,
  type KernelName,
} from './shared/nonparametric';
import {
  normalPreset,
  expPreset,
  bimodalNormalPreset,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { kdeColors } from './shared/colorScales';

const PRESETS: readonly DistributionPreset[] = [
  normalPreset,
  expPreset,
  bimodalNormalPreset,
];
const KERNELS: readonly KernelName[] = [
  'gaussian',
  'epanechnikov',
  'biweight',
  'triangular',
  'uniform',
];
const KERNEL_FN: Record<KernelName, KernelFn> = {
  gaussian: gaussianKernel,
  epanechnikov: epanechnikovKernel,
  biweight: biweightKernel,
  triangular: triangularKernel,
  uniform: uniformKernel,
};

const GRID_POINTS = 300;
const WIDTH = 640;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 48 };

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

/**
 * AMISE(h) for kernel K at sample size n against true density f:
 *   AMISE = R(K) / (n h)  +  h⁴ · μ₂(K)² · R(f") / 4.
 * Requires preset.R_f_dd (all Topic 30 kdePresets carry it).
 */
function amise(
  kernelName: KernelName,
  n: number,
  h: number,
  R_f_dd: number,
): number {
  const { R, mu2 } = kernelProperties(kernelName);
  return R / (n * h) + (Math.pow(h, 4) * mu2 * mu2 * R_f_dd) / 4;
}

export default function KernelChoiceExplorer() {
  const [presetKey, setPresetKey] = useState<DistributionPreset['key']>('normal');
  const [n, setN] = useState(200);
  const [bwMode, setBwMode] = useState<'equal' | 'canonical'>('equal');
  const [hEqual, setHEqual] = useState<number | null>(null); // null → use Silverman default
  const [visible, setVisible] = useState<Set<KernelName>>(new Set(KERNELS));
  const [seed, setSeed] = useState(42);

  const preset = PRESETS.find((p) => p.key === presetKey) ?? normalPreset;

  // Draw a fresh sample whenever seed/preset/n change.
  const sample = useMemo(() => {
    const rng = seededUniform(seed);
    const out: number[] = new Array(n);
    for (let i = 0; i < n; i++) out[i] = preset.sampler(rng);
    return out;
  }, [preset, n, seed]);

  // Silverman default bandwidth for the current sample (used by 'equal' mode default slider).
  const hSilverman = useMemo(() => silvermanBandwidth(sample), [sample]);

  // Bandwidth actually used by each kernel under each mode.
  const hPerKernel: Record<KernelName, number> = useMemo(() => {
    const R_f_dd = preset.R_f_dd ?? 0.21;
    if (bwMode === 'equal') {
      const h = hEqual ?? hSilverman;
      return {
        gaussian: h,
        epanechnikov: h,
        biweight: h,
        triangular: h,
        uniform: h,
      };
    }
    // canonical mode: each kernel's own AMISE-optimal h*.
    return {
      gaussian: amiseOptimalBandwidth(R_f_dd, n, 'gaussian'),
      epanechnikov: amiseOptimalBandwidth(R_f_dd, n, 'epanechnikov'),
      biweight: amiseOptimalBandwidth(R_f_dd, n, 'biweight'),
      triangular: amiseOptimalBandwidth(R_f_dd, n, 'triangular'),
      uniform: amiseOptimalBandwidth(R_f_dd, n, 'uniform'),
    };
  }, [bwMode, hEqual, hSilverman, preset.R_f_dd, n]);

  // Plot grid + true density + KDE curve per visible kernel.
  const curves = useMemo(() => {
    const [xMin, xMax] = preset.domain;
    const grid = linspace(xMin, xMax, GRID_POINTS);
    const truth = grid.map((x) => preset.pdf(x));
    const byKernel: Partial<Record<KernelName, number[]>> = {};
    for (const k of KERNELS) {
      if (!visible.has(k)) continue;
      byKernel[k] = kdeEvaluateGrid(sample, grid, hPerKernel[k], KERNEL_FN[k]);
    }
    let yMax = Math.max(...truth);
    for (const ys of Object.values(byKernel)) {
      if (ys) yMax = Math.max(yMax, ...ys);
    }
    return { grid, truth, byKernel, yMax: yMax * 1.08 };
  }, [sample, hPerKernel, preset, visible]);

  const xScale = scaleLinear(preset.domain[0], preset.domain[1], MARGIN.left, WIDTH - MARGIN.right);
  const yScale = scaleLinear(0, curves.yMax, HEIGHT - MARGIN.bottom, MARGIN.top);

  const pathFor = (ys: number[]): string => {
    const pts = curves.grid.map((x, i) => `${xScale(x).toFixed(1)},${yScale(ys[i]).toFixed(1)}`);
    return 'M' + pts.join('L');
  };

  const toggleKernel = (k: KernelName) => {
    const next = new Set(visible);
    if (next.has(k)) next.delete(k);
    else next.add(k);
    setVisible(next);
  };

  const R_f_dd = preset.R_f_dd ?? NaN;

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6">
      <div className="mb-3 text-sm text-slate-600">
        <span className="font-semibold text-slate-800">Kernel choice explorer</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          Five kernels against the true density. Watch how close the curves stay — the efficiency column shows why.
        </span>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap gap-4 mb-4 items-end">
        {/* Preset */}
        <div>
          <div className="text-xs text-slate-500 mb-1">Density</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5">
            {PRESETS.map((p) => (
              <button
                key={p.key}
                type="button"
                onClick={() => {
                  setPresetKey(p.key);
                  setHEqual(null);
                }}
                className={`px-2.5 py-1 text-xs rounded transition-colors ${
                  presetKey === p.key
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* n slider — log-scale 50..2000 */}
        <div>
          <div className="text-xs text-slate-500 mb-1">n = {n}</div>
          <input
            type="range"
            min={Math.log(50)}
            max={Math.log(2000)}
            step={0.01}
            value={Math.log(n)}
            onChange={(e) => {
              const next = Math.round(Math.exp(Number(e.target.value)));
              setN(Math.max(50, Math.min(2000, next)));
              setHEqual(null);
            }}
            className="w-36"
            aria-label="Sample size"
          />
        </div>

        {/* Bandwidth mode */}
        <div>
          <div className="text-xs text-slate-500 mb-1">Bandwidth</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5">
            <button
              type="button"
              onClick={() => setBwMode('equal')}
              className={`px-2.5 py-1 text-xs rounded ${
                bwMode === 'equal'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              equal h
            </button>
            <button
              type="button"
              onClick={() => setBwMode('canonical')}
              className={`px-2.5 py-1 text-xs rounded ${
                bwMode === 'canonical'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              canonical h*
            </button>
          </div>
        </div>

        {bwMode === 'equal' && (
          <div>
            <div className="text-xs text-slate-500 mb-1">h = {fmt(hEqual ?? hSilverman, 3)}</div>
            <input
              type="range"
              min={0.01}
              max={2.0}
              step={0.01}
              value={hEqual ?? hSilverman}
              onChange={(e) => setHEqual(Number(e.target.value))}
              className="w-36"
              aria-label="Bandwidth"
            />
          </div>
        )}

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700"
        >
          Resample
        </button>
      </div>

      {/* Plot */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full h-auto max-w-full"
          role="img"
          aria-label="KDE curves for five kernels overlaid on the sample and true density"
        >
          {/* Axes */}
          <line
            x1={MARGIN.left}
            y1={HEIGHT - MARGIN.bottom}
            x2={WIDTH - MARGIN.right}
            y2={HEIGHT - MARGIN.bottom}
            stroke="#64748b"
          />
          <line
            x1={MARGIN.left}
            y1={MARGIN.top}
            x2={MARGIN.left}
            y2={HEIGHT - MARGIN.bottom}
            stroke="#64748b"
          />
          {/* x-axis ticks */}
          {linspace(preset.domain[0], preset.domain[1], 6).map((x, i) => (
            <g key={`xt-${i}`}>
              <line x1={xScale(x)} y1={HEIGHT - MARGIN.bottom} x2={xScale(x)} y2={HEIGHT - MARGIN.bottom + 4} stroke="#64748b" />
              <text x={xScale(x)} y={HEIGHT - MARGIN.bottom + 16} textAnchor="middle" className="text-[10px] fill-slate-600">
                {x.toFixed(1)}
              </text>
            </g>
          ))}
          {/* Rug marks — sample points */}
          {sample.slice(0, 200).map((x, i) => {
            const cx = xScale(x).toFixed(1);
            return (
              <line
                key={`r-${i}`}
                x1={cx}
                y1={HEIGHT - MARGIN.bottom}
                x2={cx}
                y2={HEIGHT - MARGIN.bottom - 4}
                stroke="#94a3b8"
                strokeWidth={0.5}
                opacity={0.6}
              />
            );
          })}
          {/* True density */}
          <path d={pathFor(curves.truth)} fill="none" stroke="#0f172a" strokeWidth={2} opacity={0.9} />
          {/* KDE curves per kernel */}
          {KERNELS.map((k) => {
            const ys = curves.byKernel[k];
            if (!ys) return null;
            return (
              <path
                key={k}
                d={pathFor(ys)}
                fill="none"
                stroke={kdeColors.kernels[k]}
                strokeWidth={1.6}
                opacity={0.85}
              />
            );
          })}
          <text x={MARGIN.left} y={MARGIN.top - 6} className="text-[11px] fill-slate-500">
            {preset.name} &nbsp;·&nbsp; n = {n} &nbsp;·&nbsp; mode = {bwMode === 'equal' ? 'equal h' : 'canonical h*'}
          </text>
        </svg>
      </div>

      {/* Kernel table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-slate-500">
            <tr>
              <th className="text-left font-medium py-1.5 pr-3">Show</th>
              <th className="text-left font-medium py-1.5 pr-3">Kernel</th>
              <th className="text-right font-medium py-1.5 px-3">μ₂(K)</th>
              <th className="text-right font-medium py-1.5 px-3">R(K)</th>
              <th className="text-right font-medium py-1.5 px-3">eff</th>
              <th className="text-right font-medium py-1.5 px-3">h used</th>
              <th className="text-right font-medium py-1.5 pl-3">AMISE(h)</th>
            </tr>
          </thead>
          <tbody>
            {KERNELS.map((k) => {
              const spec = kernelProperties(k);
              const h = hPerKernel[k];
              const a = Number.isFinite(R_f_dd) ? amise(k, n, h, R_f_dd) : NaN;
              return (
                <tr key={k} className="border-t border-slate-200">
                  <td className="py-1.5 pr-3">
                    <input
                      type="checkbox"
                      checked={visible.has(k)}
                      onChange={() => toggleKernel(k)}
                      aria-label={`Toggle ${spec.name} kernel`}
                    />
                  </td>
                  <td className="py-1.5 pr-3 font-medium flex items-center gap-1.5">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-sm"
                      style={{ backgroundColor: kdeColors.kernels[k] }}
                      aria-hidden
                    />
                    {spec.name}
                  </td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(spec.mu2)}</td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(spec.R)}</td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(spec.efficiency)}</td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(h)}</td>
                  <td className="py-1.5 pl-3 text-right tabular-nums">{fmt(a, 4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-slate-500 leading-relaxed">
        The five kernels track the true density so closely that in any given sample you rarely see
        a meaningful difference. Epanechnikov is AMISE-optimal (efficiency = 1), but the next three
        are within a percent; Uniform trails by ~7%. In practice, the bandwidth <span className="italic">h</span>{' '}
        carries all the action.
      </p>
    </div>
  );
}
