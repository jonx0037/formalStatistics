/**
 * NPMLEFitter — interactive for Topic 33 §33.6.
 *
 * Watch g-modeling happen: fixed-grid EM (LAI1978) moves prior mass on a
 * grid, Lindsay's gradient function D(θ) certifies optimality (Thm 6c:
 * D ≤ 1 everywhere, = 1 on the support), and the resulting Bayes rule
 * t_Ĝ is monotone because it is a genuine posterior mean (Thm 3iii).
 *
 * EM steps call npmleGridEM with tol = 0 and initialWeights = the current
 * weights, so "Step ×n" resumes exactly where the last step stopped.
 * D(θ) and the log-likelihood are computed from likelihood matrices cached
 * per (sample, grid): the same formula as npmleGradientFunction, which
 * recomputes f̂(x_i) on every call and is too slow to evaluate at 301
 * points per frame.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  sampleNormalMixture,
  npmleGridEM,
  posteriorMeanDiscretePrior,
  posteriorMeanNormalMixturePrior,
  tweedieKDE,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { npmleFitterPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 14, right: 14, bottom: 38, left: 44 };
const MOBILE_BREAKPOINT = 640;
const NARROW_BREAKPOINT = 480;
const SIGMA_SQ = 1;
const GRID_SIZES = [51, 101, 181];
const NARROW_GRID_SIZE = 101;
const N_PLOT = 301;
const N_RULE = 241;
const CERT_TOL = 1.001;
const RUN_CHUNK = 200;
const RUN_MAX = 20000;
const SUPPORT_EPS = 1e-3;
const PHI_NORM = 1 / Math.sqrt(2 * Math.PI * SIGMA_SQ);

function linspace(lo: number, hi: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
}

function phi(u: number): number {
  return PHI_NORM * Math.exp((-0.5 * u * u) / SIGMA_SQ);
}

/** Fitted marginal f̂(x_i) = Σ_k w_k φ(x_i − ϑ_k) from the cached N×K matrix. */
function fittedMarginal(L: Float64Array, n: number, K: number, w: number[]): Float64Array {
  const f = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = i * K;
    for (let k = 0; k < K; k++) s += L[row + k] * w[k];
    f[i] = s;
  }
  return f;
}

/** Gradient function D(ϑ_k) = (1/N) Σ_i φ(x_i − ϑ_k)/f̂(x_i) at every grid point. */
function gradientOnGrid(L: Float64Array, n: number, K: number, f: Float64Array): Float64Array {
  const D = new Float64Array(K);
  for (let i = 0; i < n; i++) {
    const inv = 1 / f[i];
    const row = i * K;
    for (let k = 0; k < K; k++) D[k] += L[row + k] * inv;
  }
  for (let k = 0; k < K; k++) D[k] /= n;
  return D;
}

function maxOf(a: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

function linePath(xs: number[], ys: ArrayLike<number>, sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

interface EmState {
  key: string;
  weights: number[];
  iter: number;
}

function Panel({
  width,
  height,
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  xTicks,
  yTicks,
  children,
}: {
  width: number;
  height: number;
  label: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: number[];
  yTicks: number[];
  children: (sx: (v: number) => number, sy: (v: number) => number) => React.ReactNode;
}) {
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const axisColor = 'var(--color-text-muted)';
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label}>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {t}
            </text>
          </g>
        ))}
        {children(sx, sy)}
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={plotH} x2={sx(t)} y2={plotH + 4} stroke={axisColor} />
            <text x={sx(t)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
              {t}
            </text>
          </g>
        ))}
        <text x={plotW / 2} y={plotH + 31} textAnchor="middle" fontSize={11} fill="var(--color-text)">
          {xLabel}
        </text>
        <text
          x={-32}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          transform={`rotate(-90 ${-32} ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

function niceTicks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(3)));
  return out;
}

export default function NPMLEFitter() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = npmleFitterPresets[presetIdx];
  const [N, setN] = useState(preset.N);
  const [seed, setSeed] = useState(preset.seed);
  const [gridSize, setGridSize] = useState(preset.grid.size);
  const [em, setEm] = useState<EmState>({ key: '', weights: [], iter: 0 });
  const [running, setRunning] = useState(false);
  const rafRef = useRef<number | null>(null);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const narrow = containerW < NARROW_BREAKPOINT;
  const K = narrow ? NARROW_GRID_SIZE : gridSize;
  const { lo, hi } = preset.grid;

  const selectPreset = (i: number) => {
    setPresetIdx(i);
    setN(npmleFitterPresets[i].N);
    setSeed(npmleFitterPresets[i].seed);
  };

  const sample = useMemo(() => {
    const rng = createSeededRng(seed);
    const theta = sampleNormalMixture(preset.comps, N, rng);
    return theta.map((t) => t + Math.sqrt(SIGMA_SQ) * rng.normal());
  }, [preset, N, seed]);

  const grid = useMemo(() => linspace(lo, hi, K), [lo, hi, K]);
  const thetaPlot = useMemo(() => linspace(lo, hi, N_PLOT), [lo, hi]);
  const ruleXs = useMemo(() => linspace(lo, hi, N_RULE), [lo, hi]);
  const dataKey = `${preset.id}|${N}|${seed}|${K}`;

  // Likelihood matrices, cached per (sample, grid): L[i, k] = φ(x_i − ϑ_k), P[j, i] = φ(x_i − θ_j).
  const mats = useMemo(() => {
    const n = sample.length;
    const L = new Float64Array(n * K);
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) L[i * K + k] = phi(sample[i] - grid[k]);
    const P = new Float64Array(N_PLOT * n);
    for (let j = 0; j < N_PLOT; j++) for (let i = 0; i < n; i++) P[j * n + i] = phi(sample[i] - thetaPlot[j]);
    return { L, P, n };
  }, [sample, grid, K, thetaPlot]);

  const uniform = useMemo(() => new Array<number>(K).fill(1 / K), [K]);
  const current: EmState = em.key === dataKey ? em : { key: dataKey, weights: uniform, iter: 0 };

  const diag = useMemo(() => {
    const { L, P, n } = mats;
    const f = fittedMarginal(L, n, K, current.weights);
    let logLik = 0;
    for (let i = 0; i < n; i++) logLik += Math.log(f[i]);
    const Dgrid = gradientOnGrid(L, n, K, f);
    const Dplot = new Float64Array(N_PLOT);
    for (let j = 0; j < N_PLOT; j++) {
      let s = 0;
      const row = j * n;
      for (let i = 0; i < n; i++) s += P[row + i] / f[i];
      Dplot[j] = s / n;
    }
    const support = current.weights.filter((w) => w > SUPPORT_EPS).length;
    return { logLik, Dgrid, Dplot, maxD: maxOf(Dgrid), support };
  }, [mats, K, current.weights]);

  const npmleRule = useMemo(
    () => ruleXs.map((v) => posteriorMeanDiscretePrior(v, grid, current.weights, SIGMA_SQ)),
    [ruleXs, grid, current.weights],
  );
  const fixedRules = useMemo(
    () => ({
      oracle: ruleXs.map((v) => posteriorMeanNormalMixturePrior(v, preset.comps, SIGMA_SQ)),
      kde: tweedieKDE(sample, ruleXs, SIGMA_SQ),
    }),
    [ruleXs, preset, sample],
  );

  const stopRun = () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setRunning(false);
  };
  // A new sample or grid invalidates a run in progress; so does unmounting.
  useEffect(() => stopRun, [dataKey]);

  const step = (n: number) => {
    const r = npmleGridEM(sample, grid, SIGMA_SQ, { maxIter: n, tol: 0, initialWeights: current.weights });
    setEm({ key: dataKey, weights: r.weights, iter: current.iter + r.iterations });
  };

  const runToCertificate = () => {
    if (running) {
      stopRun();
      return;
    }
    const { L, n } = mats;
    const certMax = (w: number[]) => maxOf(gradientOnGrid(L, n, K, fittedMarginal(L, n, K, w)));
    let w = current.weights;
    let it = current.iter;
    if (certMax(w) <= CERT_TOL || it >= RUN_MAX) return;
    const key = dataKey;
    setRunning(true);
    const tick = () => {
      const r = npmleGridEM(sample, grid, SIGMA_SQ, { maxIter: RUN_CHUNK, tol: 0, initialWeights: w });
      w = r.weights;
      it += r.iterations;
      setEm({ key, weights: w, iter: it });
      if (certMax(w) <= CERT_TOL || it >= RUN_MAX) {
        rafRef.current = null;
        setRunning(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const reset = () => {
    stopRun();
    setEm({ key: dataKey, weights: uniform, iter: 0 });
  };

  // True prior for the left panel: atoms as diamonds, continuous parts as
  // density × grid spacing (mass per grid cell, comparable to the stems).
  const spacing = (hi - lo) / (K - 1);
  const atoms = preset.comps.filter((c) => c.variance === 0);
  const continuous = preset.comps.filter((c) => c.variance > 0);
  const trueCurve = continuous.length
    ? thetaPlot.map((t) =>
        continuous.reduce(
          (s, c) => s + (c.weight * spacing * Math.exp((-0.5 * (t - c.mean) ** 2) / c.variance)) / Math.sqrt(2 * Math.PI * c.variance),
          0,
        ),
      )
    : null;
  const gMax =
    1.12 *
    Math.max(
      maxOf(current.weights),
      ...atoms.map((a) => a.weight),
      trueCurve ? maxOf(trueCurve) : 0,
      0.02,
    );
  // EM smears each atom's mass over neighboring grid points (Rem 13), so for
  // all-atomic priors report the Ĝ mass in each atom's nearest-atom cell.
  const clusterMass =
    atoms.length > 0 && atoms.length === preset.comps.length
      ? atoms.map((_, j) =>
          grid.reduce((s, g, k) => {
            let nearest = 0;
            atoms.forEach((a, m) => {
              if (Math.abs(g - a.mean) < Math.abs(g - atoms[nearest].mean)) nearest = m;
            });
            return nearest === j ? s + current.weights[k] : s;
          }, 0),
        )
      : null;
  const dMax = Math.max(1.15, 1.05 * maxOf(diag.Dplot));
  const certified = diag.maxD <= CERT_TOL;

  const panelW = Math.max(260, isMobile ? containerW : (containerW - 16) / 2);
  const ruleW = Math.max(260, containerW);
  const xTicks = niceTicks(lo, hi, hi - lo > 8 ? 2 : 1);
  const btn = 'rounded border px-3 py-1.5 text-xs transition disabled:opacity-40';

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">True prior g:</span>
        {npmleFitterPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => selectPreset(i)}
            className={`rounded px-3 py-1 text-xs transition ${
              i === presetIdx
                ? 'bg-[var(--color-shrink)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 text-sm sm:grid-cols-[1fr_auto_auto]">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Observations N</span>
            <span className="font-mono text-xs">{N.toLocaleString('en-US')}</span>
          </span>
          <input
            type="range"
            min={100}
            max={3000}
            step={100}
            value={N}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-g-model)]"
          />
        </label>
        {!narrow && (
          <div className="flex items-center gap-1 text-xs">
            <span className="mr-1">Grid</span>
            {GRID_SIZES.map((g) => (
              <button
                key={g}
                onClick={() => setGridSize(g)}
                className={`rounded px-2 py-1 ${
                  g === gridSize ? 'bg-[var(--color-shrink)] text-white' : 'bg-[var(--color-surface)]'
                }`}
                style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
              >
                {g}
              </button>
            ))}
          </div>
        )}
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="rounded px-3 py-1.5 text-xs font-medium text-white"
          style={{ background: bayesianColors.shrink }}
        >
          Resample (seed {seed})
        </button>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
        {[1, 10, 100].map((n) => (
          <button key={n} onClick={() => step(n)} disabled={running} className={btn} style={{ borderColor: 'var(--color-border)' }}>
            EM step ×{n}
          </button>
        ))}
        <button
          onClick={runToCertificate}
          className="rounded px-3 py-1.5 text-xs font-medium text-white"
          style={{ background: bayesianColors.gModel }}
        >
          {running ? 'Stop' : 'Run to certificate'}
        </button>
        <button
          onClick={reset}
          className={`${btn} col-span-2 sm:col-span-1`}
          style={{ borderColor: 'var(--color-border)' }}
        >
          Reset to uniform
        </button>
      </div>

      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>iteration {current.iter.toLocaleString('en-US')}</span>
        <span>log-lik {diag.logLik.toFixed(2)}</span>
        <span>
          mass &gt; 0.001 on {diag.support} of {K} grid points
        </span>
        <span style={{ color: certified ? 'var(--color-true-parameter)' : 'var(--color-text-muted)' }}>
          max D = {diag.maxD.toFixed(4)} — {certified ? 'certificate satisfied ✓' : 'not yet optimal'}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: bayesianColors.gModel }}>| NPMLE Ĝ (grid weights)</span>
            {atoms.length > 0 && <span style={{ color: bayesianColors.true }}> · ◆ true atoms</span>}
            {trueCurve && <span style={{ color: bayesianColors.true }}> · — true density × grid spacing</span>}
          </div>
          <Panel
            width={panelW}
            height={230}
            label="Estimated prior weights on the grid against the true prior"
            xLabel="θ"
            yLabel="prior mass"
            xDomain={[lo, hi]}
            yDomain={[0, gMax]}
            xTicks={xTicks}
            yTicks={niceTicks(0, gMax, gMax > 0.4 ? 0.2 : gMax > 0.15 ? 0.05 : 0.01)}
          >
            {(sx, sy) => (
              <>
                {trueCurve && (
                  <path d={linePath(thetaPlot, trueCurve, sx, sy)} fill="none" stroke={bayesianColors.true} strokeWidth={1.5} />
                )}
                {grid.map((g, k) =>
                  current.weights[k] > 1e-5 ? (
                    <line
                      key={k}
                      x1={sx(g)}
                      y1={sy(0)}
                      x2={sx(g)}
                      y2={sy(current.weights[k])}
                      stroke={bayesianColors.gModel}
                      strokeWidth={K > 120 ? 1.5 : 2}
                    />
                  ) : null,
                )}
                {atoms.map((a, j) => (
                  <g key={a.mean}>
                    <path
                      d={`M${sx(a.mean)},${sy(a.weight) - 6}L${sx(a.mean) + 5},${sy(a.weight)}L${sx(a.mean)},${sy(a.weight) + 6}L${sx(a.mean) - 5},${sy(a.weight)}Z`}
                      fill={bayesianColors.true}
                      stroke="var(--color-surface)"
                      strokeWidth={1}
                    />
                    {clusterMass && (
                      <text x={sx(a.mean)} y={sy(a.weight) - 10} textAnchor="middle" fontSize={10} fill="var(--color-text)">
                        Ĝ {clusterMass[j].toFixed(2)} · true {a.weight.toFixed(2)}
                      </text>
                    )}
                  </g>
                ))}
              </>
            )}
          </Panel>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: bayesianColors.gModel }}>— gradient function D(θ)</span> · - - D = 1 · ● support points
          </div>
          <Panel
            width={panelW}
            height={230}
            label="Lindsay gradient function of the current fit"
            xLabel="θ"
            yLabel="D(θ)"
            xDomain={[lo, hi]}
            yDomain={[0, dMax]}
            xTicks={xTicks}
            yTicks={niceTicks(0, dMax, dMax > 3 ? 1 : 0.5)}
          >
            {(sx, sy) => (
              <>
                <line x1={sx(lo)} y1={sy(1)} x2={sx(hi)} y2={sy(1)} stroke="var(--color-text-muted)" strokeDasharray="5 3" />
                <path d={linePath(thetaPlot, diag.Dplot, sx, sy)} fill="none" stroke={bayesianColors.gModel} strokeWidth={2} />
                {grid.map((g, k) =>
                  current.weights[k] > SUPPORT_EPS ? (
                    <circle key={k} cx={sx(g)} cy={sy(diag.Dgrid[k])} r={2.5} fill={bayesianColors.gModel} />
                  ) : null,
                )}
              </>
            )}
          </Panel>
        </div>
      </div>

      <div className="mt-4 min-w-0">
        <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <span style={{ color: bayesianColors.mle }}>- - identity</span>
          <span style={{ color: bayesianColors.oracle }}>— oracle Bayes</span>
          <span style={{ color: bayesianColors.fModel }}>- - Tweedie–KDE (f-modeling)</span>
          <span style={{ color: bayesianColors.gModel }}>—● NPMLE rule t_Ĝ (g-modeling)</span>
        </div>
        <Panel
          width={ruleW}
          height={isMobile ? 250 : 280}
          label="Bayes rules: identity, oracle, Tweedie with a KDE marginal, and the NPMLE rule"
          xLabel="observation x"
          yLabel="estimate of θ"
          xDomain={[lo, hi]}
          yDomain={[lo, hi]}
          xTicks={xTicks}
          yTicks={xTicks}
        >
          {(sx, sy) => (
            <>
              <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke={bayesianColors.mle} strokeWidth={1.5} strokeDasharray="5 3" />
              <path d={linePath(ruleXs, fixedRules.oracle, sx, sy)} fill="none" stroke={bayesianColors.oracle} strokeWidth={2} />
              <path
                d={linePath(ruleXs, fixedRules.kde.map((v) => Math.min(hi, Math.max(lo, v))), sx, sy)}
                fill="none"
                stroke={bayesianColors.fModel}
                strokeWidth={2}
                strokeDasharray="5 3"
              />
              <path d={linePath(ruleXs, npmleRule, sx, sy)} fill="none" stroke={bayesianColors.gModel} strokeWidth={2} />
              {ruleXs
                .filter((_, i) => i % 20 === 0)
                .map((v, j) => (
                  <circle key={v} cx={sx(v)} cy={sy(npmleRule[j * 20])} r={3} fill={bayesianColors.gModel} />
                ))}
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
