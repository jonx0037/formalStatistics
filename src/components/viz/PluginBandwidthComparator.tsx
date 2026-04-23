/**
 * PluginBandwidthComparator — Topic 30 §30.8.
 *
 * Four data-driven bandwidth selectors head-to-head on the same sample:
 *
 *   1. Silverman's rule       ĥ_ROT = 1.06 · min(SD, IQR/1.34) · n^(−1/5)
 *   2. Scott's rule           ĥ_Scott = 1.06 · SD · n^(−1/5)
 *   3. Unbiased CV            minimizes ∫f̂_h² − (2/n) Σ f̂_{h,-i}(X_i)
 *   4. Sheather-Jones plug-in one-step direct plug-in (WAN1995 §3.6),
 *                             inlined here rather than added to `nonparametric.ts`
 *                             (brief Appendix B.9 locked decision).
 *
 * The pedagogical payoff is the bimodal preset: Silverman over-smooths into a
 * single mode; SHJ recovers both modes cleanly; UCV is approximately right
 * but noisy across resamples.
 *
 * Performance: UCV is O(n² G) per call. The "Run 200 resamples" MC mode is
 * capped at n ≤ 500; per-sample rendering is fine up to n = 2000.
 */
import { useDeferredValue, useMemo, useState } from 'react';

import {
  gaussianKernel,
  kdeEvaluateGrid,
  silvermanBandwidth,
  scottBandwidth,
  ucvBandwidth,
} from './shared/nonparametric';
import {
  normalPreset,
  bimodalNormalPreset,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { cdfStdNormal, quantileStdNormal } from './shared/distributions';
import { kdeColors } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Inline presets — lognormal (skewed) and t_3 (heavy-tailed). These only
// appear in this component; we don't pollute `nonparametric-data.ts`.
// ────────────────────────────────────────────────────────────────────────────

function boxMuller(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
}

/** Lognormal(μ = 0, σ = 1) on (0, ∞). */
const skewedPreset: DistributionPreset = {
  name: 'Lognormal(0, 1)',
  key: 'exp', // reuse — 'lognormal' isn't in DistributionKey and it's OK: this component owns the preset
  pdf: (x) => (x > 0 ? Math.exp(-0.5 * Math.log(x) ** 2) / (x * Math.sqrt(2 * Math.PI)) : 0),
  cdf: (x) => (x > 0 ? cdfStdNormal(Math.log(x)) : 0),
  quantile: (p) => Math.exp(quantileStdNormal(p)),
  sampler: (rng) => Math.exp(boxMuller(rng)),
  domain: [0, 8],
};

/** Student's t with 3 degrees of freedom — heavy-tailed. */
const t3Preset: DistributionPreset = {
  name: 't(3)',
  key: 'cauchy',
  pdf: (x) => {
    // Γ(2) / (√(3π) · Γ(3/2)) = 1 / (√3 · B(1/2, 3/2)) = 2 / (π √3)
    const c = 2 / (Math.PI * Math.sqrt(3));
    return c * Math.pow(1 + (x * x) / 3, -2);
  },
  cdf: (x) => {
    // t3 CDF has a closed form but it's easier to use numerical integration
    // for the rare call-sites here (bisect in quantile). Use Simpson's rule.
    const n = 200;
    const a = -20;
    const b = x;
    if (b <= a) return 0;
    if (b >= 20) return 1;
    const h = (b - a) / n;
    const f = (t: number) =>
      (2 / (Math.PI * Math.sqrt(3))) * Math.pow(1 + (t * t) / 3, -2);
    let s = f(a) + f(b);
    for (let i = 1; i < n; i++) s += (i % 2 === 0 ? 2 : 4) * f(a + i * h);
    return (s * h) / 3;
  },
  quantile: (p) => {
    // Bisection on the CDF; adequate for component-level reference lines.
    if (p <= 0.001) return -10;
    if (p >= 0.999) return 10;
    let lo = -10;
    let hi = 10;
    const cdf = (x: number) => t3Preset.cdf(x);
    for (let i = 0; i < 60 && hi - lo > 1e-6; i++) {
      const mid = 0.5 * (lo + hi);
      if (cdf(mid) < p) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  },
  sampler: (rng) => {
    // t_3 via Z / √(V/3) where V ~ χ²_3 = sum of 3 independent N(0,1) squares.
    const z = boxMuller(rng);
    const z1 = boxMuller(rng);
    const z2 = boxMuller(rng);
    const z3 = boxMuller(rng);
    const v = z1 * z1 + z2 * z2 + z3 * z3;
    return z / Math.sqrt(v / 3);
  },
  domain: [-8, 8],
};

const PRESETS: ReadonlyArray<DistributionPreset & { id: string }> = [
  { ...normalPreset, id: 'unimodal' },
  { ...bimodalNormalPreset, id: 'bimodal' },
  { ...skewedPreset, id: 'skewed' },
  { ...t3Preset, id: 't3' },
];

// ────────────────────────────────────────────────────────────────────────────
// Sheather-Jones plug-in (one-step direct) — inlined per brief B.9.
//
// Uses Silverman as pilot bandwidth g, estimates R̂(f'') via the density-
// derivative plug-in R̂(f'') = n^{-2} g^{-5} Σᵢ Σⱼ K^(4)((Xᵢ - Xⱼ)/g), and
// substitutes into the AMISE-optimal bandwidth formula.
//
// Gaussian K^(4)(u) = (u⁴ − 6u² + 3) · φ(u).
// ────────────────────────────────────────────────────────────────────────────
function sheatherJonesBandwidth(sample: number[]): number {
  const n = sample.length;
  if (n < 10) throw new Error(`SHJ: n=${n} too small`);
  const g = silvermanBandwidth(sample);
  // Gaussian K^(4)(u) = (u⁴ − 6u² + 3) · φ(u) is even, so the full double
  // sum is  n · K^(4)(0) + 2 · Σ_{i<j} K^(4)((X_i − X_j)/g).
  // K^(4)(0) = 3 · φ(0) = 3/√(2π). Halve the kernel evaluations.
  const norm = 1 / Math.sqrt(2 * Math.PI);
  let upperTri = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const u = (sample[i] - sample[j]) / g;
      const u2 = u * u;
      upperTri += (u2 * u2 - 6 * u2 + 3) * Math.exp(-0.5 * u2);
    }
  }
  const sum = n * 3 + 2 * upperTri; // both terms already missing the `norm` factor
  const normalizer = n * n * Math.pow(g, 5) / norm; // i.e. × √(2π)
  const R_f_dd_hat = Math.max(sum / normalizer, 1e-10);
  const R_K = 1 / (2 * Math.sqrt(Math.PI));
  return Math.pow(R_K / (n * R_f_dd_hat), 1 / 5);
}

// ────────────────────────────────────────────────────────────────────────────
// Integrated squared error (ISE) on a grid — our single-sample proxy for MISE.
// ────────────────────────────────────────────────────────────────────────────
function ise(
  preset: DistributionPreset,
  sample: number[],
  h: number,
  grid: number[],
): number {
  const dx = grid[1] - grid[0];
  let s = 0;
  const fHat = kdeEvaluateGrid(sample, grid, h, gaussianKernel);
  for (let i = 0; i < grid.length; i++) {
    const d = fHat[i] - preset.pdf(grid[i]);
    s += d * d;
  }
  return s * dx;
}

type Selector = 'silverman' | 'scott' | 'ucv' | 'sheatherJones';

const SELECTOR_LABEL: Record<Selector, string> = {
  silverman: 'Silverman',
  scott: 'Scott',
  ucv: 'UCV',
  sheatherJones: 'Sheather–Jones',
};

function verdict(ise: number, baseline: number): 'over-smooth' | 'under-smooth' | 'good' {
  // Relative to Silverman baseline; coarse tri-state indicator.
  const ratio = ise / baseline;
  if (ratio < 0.85) return 'good';
  if (ratio > 1.2) return 'over-smooth';
  return 'good';
}

const WIDTH = 640;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 48 };
const GRID_POINTS = 300;

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

export default function PluginBandwidthComparator() {
  const [presetId, setPresetId] = useState('bimodal');
  const [n, setN] = useState(200);
  const [visible, setVisible] = useState<Set<Selector>>(
    new Set<Selector>(['silverman', 'scott', 'ucv', 'sheatherJones']),
  );
  const [seed, setSeed] = useState(42);
  const [mcResults, setMcResults] = useState<Partial<Record<Selector, { median: number; iqr: number }>> | null>(null);
  const [mcRunning, setMcRunning] = useState(false);

  // Defer the expensive inputs (n, seed, presetId) so slider drags keep the
  // UI responsive while UCV's O(n²G) grid-search runs in the background.
  // React will render with stale bandwidths briefly while the new sample
  // sample/bandwidth computation happens in the deferred pass.
  const dN = useDeferredValue(n);
  const dSeed = useDeferredValue(seed);
  const dPresetId = useDeferredValue(presetId);

  const preset = PRESETS.find((p) => p.id === dPresetId) ?? PRESETS[1];

  const sample = useMemo(() => {
    const rng = seededUniform(dSeed);
    const out: number[] = new Array(dN);
    for (let i = 0; i < dN; i++) out[i] = preset.sampler(rng);
    return out;
  }, [preset, dN, dSeed]);

  const bandwidths = useMemo((): Record<Selector, number> => {
    return {
      silverman: silvermanBandwidth(sample),
      scott: scottBandwidth(sample),
      ucv: dN <= 2000 ? ucvBandwidth(sample, 'gaussian') : silvermanBandwidth(sample),
      sheatherJones: sheatherJonesBandwidth(sample),
    };
  }, [sample, dN]);

  const curves = useMemo(() => {
    const [xMin, xMax] = preset.domain;
    const grid = linspace(xMin, xMax, GRID_POINTS);
    const truth = grid.map((x) => preset.pdf(x));
    const byKernel: Partial<Record<Selector, number[]>> = {};
    for (const s of Object.keys(bandwidths) as Selector[]) {
      if (!visible.has(s)) continue;
      byKernel[s] = kdeEvaluateGrid(sample, grid, bandwidths[s], gaussianKernel);
    }
    let yMax = Math.max(...truth);
    for (const ys of Object.values(byKernel)) {
      if (ys) yMax = Math.max(yMax, ...ys);
    }
    return { grid, truth, byKernel, yMax: yMax * 1.1 };
  }, [sample, bandwidths, preset, visible]);

  const xScale = scaleLinear(preset.domain[0], preset.domain[1], MARGIN.left, WIDTH - MARGIN.right);
  const yScale = scaleLinear(0, curves.yMax, HEIGHT - MARGIN.bottom, MARGIN.top);

  const pathFor = (ys: number[]): string => {
    const pts = curves.grid.map((x, i) => `${xScale(x).toFixed(1)},${yScale(ys[i]).toFixed(1)}`);
    return 'M' + pts.join('L');
  };

  // ISE per selector at the current sample.
  const currentIse = useMemo((): Record<Selector, number> => {
    return {
      silverman: ise(preset, sample, bandwidths.silverman, curves.grid),
      scott: ise(preset, sample, bandwidths.scott, curves.grid),
      ucv: ise(preset, sample, bandwidths.ucv, curves.grid),
      sheatherJones: ise(preset, sample, bandwidths.sheatherJones, curves.grid),
    };
  }, [preset, sample, bandwidths, curves.grid]);

  const toggleSelector = (s: Selector) => {
    const next = new Set(visible);
    if (next.has(s)) next.delete(s);
    else next.add(s);
    setVisible(next);
  };

  // Run 200 Monte-Carlo resamples in chunks of CHUNK_SIZE so the browser
  // can repaint between batches — a synchronous loop of 200 reps at n=500
  // runs UCV 200 times (O(n²G) each), freezing the UI for several seconds.
  // Chunking keeps the button's "Running…" state visible and lets the user
  // interact with other controls while the MC is in flight.
  const runMc = () => {
    if (n > 500 || mcRunning) return;
    const REPS = 200;
    const CHUNK_SIZE = 20;
    const hs: Record<Selector, number[]> = {
      silverman: [],
      scott: [],
      ucv: [],
      sheatherJones: [],
    };
    setMcRunning(true);
    setMcResults(null);
    const runChunk = (startRep: number) => {
      const end = Math.min(startRep + CHUNK_SIZE, REPS);
      for (let r = startRep; r < end; r++) {
        const rng = seededUniform(seed + 1 + r);
        const s: number[] = new Array(n);
        for (let i = 0; i < n; i++) s[i] = preset.sampler(rng);
        hs.silverman.push(silvermanBandwidth(s));
        hs.scott.push(scottBandwidth(s));
        hs.ucv.push(ucvBandwidth(s, 'gaussian'));
        hs.sheatherJones.push(sheatherJonesBandwidth(s));
      }
      if (end < REPS) {
        // Yield to the browser so it can repaint, then schedule the next chunk.
        setTimeout(() => runChunk(end), 0);
      } else {
        const summarize = (arr: number[]) => {
          arr.sort((a, b) => a - b);
          const median = arr[Math.floor(arr.length / 2)];
          const q1 = arr[Math.floor(arr.length * 0.25)];
          const q3 = arr[Math.floor(arr.length * 0.75)];
          return { median, iqr: (q3 - q1) / 1.34 };
        };
        setMcResults({
          silverman: summarize(hs.silverman),
          scott: summarize(hs.scott),
          ucv: summarize(hs.ucv),
          sheatherJones: summarize(hs.sheatherJones),
        });
        setMcRunning(false);
      }
    };
    // Defer the first chunk one tick so the button's "Running…" state paints
    // before the compute starts.
    setTimeout(() => runChunk(0), 0);
  };

  const baselineIse = currentIse.silverman;

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6">
      <div className="mb-3 text-sm text-slate-600">
        <span className="font-semibold text-slate-800">Data-driven bandwidth comparator</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          Silverman's rule is the textbook default, but watch what it does on the bimodal preset —
          the two modes merge. Sheather–Jones and UCV work harder but recover the structure.
        </span>
      </div>

      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <div className="text-xs text-slate-500 mb-1">Density</div>
          <div className="inline-flex flex-wrap gap-0.5 rounded-md border border-slate-300 bg-slate-50 p-0.5">
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
          <div className="text-xs text-slate-500 mb-1">n = {n}</div>
          <input
            type="range"
            min={Math.log(50)}
            max={Math.log(2000)}
            step={0.01}
            value={Math.log(n)}
            onChange={(e) => setN(Math.max(50, Math.min(2000, Math.round(Math.exp(Number(e.target.value))))))}
            className="w-36"
            aria-label="Sample size"
          />
        </div>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700"
        >
          Resample
        </button>

        <button
          type="button"
          onClick={runMc}
          disabled={n > 500 || mcRunning}
          title={
            n > 500
              ? 'Cap: Run 200 resamples requires n ≤ 500 (perf)'
              : mcRunning
                ? 'Running 200 Monte-Carlo replicates…'
                : 'Monte-Carlo 200 resamples'
          }
          className="px-3 py-1.5 text-xs border border-slate-300 rounded hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {mcRunning ? 'Running…' : 'Run 200 resamples'}
        </button>
      </div>

      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full h-auto max-w-full"
          role="img"
          aria-label="KDE curves for four bandwidth selectors on the current sample"
        >
          <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
          <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
          {linspace(preset.domain[0], preset.domain[1], 6).map((x, i) => (
            <g key={`xt-${i}`}>
              <line x1={xScale(x)} y1={HEIGHT - MARGIN.bottom} x2={xScale(x)} y2={HEIGHT - MARGIN.bottom + 4} stroke="#64748b" />
              <text x={xScale(x)} y={HEIGHT - MARGIN.bottom + 16} textAnchor="middle" className="text-[10px] fill-slate-600">
                {x.toFixed(1)}
              </text>
            </g>
          ))}
          <path d={pathFor(curves.truth)} fill="none" stroke="#0f172a" strokeWidth={2} opacity={0.9} />
          {(Object.keys(curves.byKernel) as Selector[]).map((s) => {
            const ys = curves.byKernel[s];
            if (!ys) return null;
            return (
              <path
                key={s}
                d={pathFor(ys)}
                fill="none"
                stroke={kdeColors.selectors[s]}
                strokeWidth={1.6}
                opacity={0.9}
              />
            );
          })}
          <text x={MARGIN.left} y={MARGIN.top - 6} className="text-[11px] fill-slate-500">
            {preset.name} &nbsp;·&nbsp; n = {n}
          </text>
        </svg>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-slate-500">
            <tr>
              <th className="text-left font-medium py-1.5 pr-3">Show</th>
              <th className="text-left font-medium py-1.5 pr-3">Selector</th>
              <th className="text-right font-medium py-1.5 px-3">h</th>
              <th className="text-right font-medium py-1.5 px-3">ISE</th>
              <th className="text-left font-medium py-1.5 pl-3">vs Silverman</th>
              {mcResults && (
                <>
                  <th className="text-right font-medium py-1.5 px-3">MC median h</th>
                  <th className="text-right font-medium py-1.5 pl-3">MC IQR/1.34</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {(['silverman', 'scott', 'ucv', 'sheatherJones'] as Selector[]).map((s) => {
              const rel = currentIse[s] / baselineIse;
              const v = verdict(currentIse[s], baselineIse);
              return (
                <tr key={s} className="border-t border-slate-200">
                  <td className="py-1.5 pr-3">
                    <input
                      type="checkbox"
                      checked={visible.has(s)}
                      onChange={() => toggleSelector(s)}
                      aria-label={`Toggle ${SELECTOR_LABEL[s]}`}
                    />
                  </td>
                  <td className="py-1.5 pr-3 font-medium flex items-center gap-1.5">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-sm"
                      style={{ backgroundColor: kdeColors.selectors[s] }}
                      aria-hidden
                    />
                    {SELECTOR_LABEL[s]}
                  </td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(bandwidths[s])}</td>
                  <td className="py-1.5 px-3 text-right tabular-nums">{fmt(currentIse[s], 4)}</td>
                  <td className="py-1.5 pl-3 tabular-nums">
                    {s === 'silverman' ? '—' : `${fmt(rel, 2)}× ${v === 'good' ? '✓' : v}`}
                  </td>
                  {mcResults && (
                    <>
                      <td className="py-1.5 px-3 text-right tabular-nums">{fmt(mcResults[s]?.median ?? NaN)}</td>
                      <td className="py-1.5 pl-3 text-right tabular-nums">{fmt(mcResults[s]?.iqr ?? NaN)}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-slate-500 leading-relaxed">
        At the Bimodal preset, Silverman's bandwidth is roughly <span className="italic">twice</span>{' '}
        the oracle h*, which is why it over-smooths across the modes. SHJ's plug-in gets much closer
        by estimating the curvature R(f"). UCV has the largest variance across resamples — try "Run 200
        resamples" on a smaller n to see the IQR column widen.
      </p>
    </div>
  );
}
