/**
 * RobbinsPoissonExplorer — interactive for Topic 33 §33.3.
 *
 * Robbins' formula Ê[θ | x] = (x + 1) y_{x+1} / y_x (Thm 2, with f replaced
 * by empirical frequencies) next to its parametric cousin, the Gamma–Poisson
 * empirical-Bayes rule (Ex 4). Clicking a count bar exposes ±1 buttons that
 * perturb that count: Robbins' estimates at x − 1 and x jump, while the
 * Gamma fit — which borrows strength across every x — barely moves (Rem 5).
 *
 * Datasets: the CASI Table 6.1 insurance counts, or a simulated book of
 * policies with θ ~ Gamma(0.7, 0.3), where the true posterior mean
 * (x + 0.7)·0.3/1.3 is known. Simulated counts above 7 are pooled into the
 * 7 cell for display only; the estimators use the full table.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  insuranceClaimsData,
  robbinsEstimate,
  gammaPoissonEB,
} from './shared/bayes';
import { sampleGammaShape } from './shared/distributions';
import { poissonSample } from './shared/convergence';
import { bayesianColors } from './shared/colorScales';
import { robbinsExplorerPresets, type RobbinsExplorerPreset } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 14, right: 16, bottom: 40, left: 44 };
const INSET_MARGIN = { top: 14, right: 8, bottom: 36, left: 40 };
const MOBILE_BREAKPOINT = 640;
const X_EST_MAX = 6; // estimates shown for x = 0, …, 6 (Robbins needs y_{x+1})
const X_BAR_MAX = 7; // count bars for x = 0, …, 7
const LOG_N_MIN = Math.log10(500);
const LOG_N_MAX = 5;
const INSET_W = 250;
const EST_XS = Array.from({ length: X_EST_MAX + 1 }, (_, x) => x);
const BAR_XS = Array.from({ length: X_BAR_MAX + 1 }, (_, x) => x);
const COUNT_TICKS = [1, 10, 100, 1000, 10000, 100000];

/** Simulate N policy holders: θ ~ Gamma(shape, scale), x | θ ~ Poisson(θ); tabulate. */
function simulateCounts(preset: RobbinsExplorerPreset, N: number, seed: number): number[] {
  const shape = preset.gammaShape ?? 0.7;
  const scale = preset.gammaScale ?? 0.3;
  const rng = createSeededRng(seed);
  const u = () => rng.random();
  const tab = new Array<number>(X_BAR_MAX + 2).fill(0);
  for (let i = 0; i < N; i++) {
    const x = poissonSample(sampleGammaShape(shape, u) * scale, u);
    while (tab.length <= x) tab.push(0);
    tab[x]++;
  }
  return tab;
}

/** SVG path through the finite points; NaN breaks the line. */
function brokenPath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  let d = '';
  let pen = false;
  xs.forEach((x, i) => {
    const y = ys[i];
    if (!Number.isFinite(y)) {
      pen = false;
      return;
    }
    d += `${pen ? 'L' : 'M'}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`;
    pen = true;
  });
  return d;
}

function fmt(v: number | undefined): string {
  return v !== undefined && Number.isFinite(v) ? v.toFixed(3) : '—';
}

export default function RobbinsPoissonExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetId, setPresetId] = useState<RobbinsExplorerPreset['id']>('insurance');
  const simPreset = robbinsExplorerPresets.find((p) => p.id === 'gamma-sim')!;
  const [logN, setLogN] = useState(Math.log10(simPreset.nDefault ?? 10000));
  const [seed, setSeed] = useState(simPreset.seed ?? 3303);
  const [deltas, setDeltas] = useState<number[]>(() => new Array(X_BAR_MAX + 1).fill(0));
  const [selected, setSelected] = useState<number | null>(5);
  const [showAll, setShowAll] = useState(false);

  const isSim = presetId === 'gamma-sim';
  const N = Math.round(10 ** logN);
  const resetDeltas = () => setDeltas(new Array(X_BAR_MAX + 1).fill(0));

  const baseCounts = useMemo(
    () => (isSim ? simulateCounts(simPreset, N, seed) : insuranceClaimsData().counts),
    [isSim, simPreset, N, seed],
  );

  // Perturbed table (only x = 0, …, 7 can be perturbed).
  const counts = useMemo(
    () => baseCounts.map((c, x) => Math.max(0, c + (x <= X_BAR_MAX ? deltas[x] : 0))),
    [baseCounts, deltas],
  );

  const robbins = useMemo(() => robbinsEstimate(counts), [counts]);
  const gamma = useMemo(() => gammaPoissonEB(counts), [counts]);
  const shape = simPreset.gammaShape ?? 0.7;
  const scale = simPreset.gammaScale ?? 0.3;
  const truth = EST_XS.map((x) => ((x + shape) * scale) / (1 + scale));

  const robbinsYs = EST_XS.map((x) => robbins[x] ?? NaN);
  const gammaYs = EST_XS.map((x) => gamma.posteriorMean[x] ?? NaN);
  const displayCounts = BAR_XS.map((x) =>
    x < X_BAR_MAX ? counts[x] ?? 0 : counts.slice(X_BAR_MAX).reduce((a, c) => a + c, 0),
  );

  // Layout.
  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const mainW = Math.max(260, isMobile ? containerW : containerW - INSET_W - 16);
  const mainH = isMobile ? 280 : 320;
  const insetW = isMobile ? Math.max(260, containerW) : INSET_W;
  const insetH = 200;
  const plotW = mainW - MARGIN.left - MARGIN.right;
  const plotH = mainH - MARGIN.top - MARGIN.bottom;

  let yMax = X_EST_MAX + 0.5;
  for (const v of [...robbinsYs, ...gammaYs]) if (Number.isFinite(v)) yMax = Math.max(yMax, v);
  yMax = Math.min(14, yMax * 1.08);
  const sx = (x: number) => ((x + 0.4) / (X_EST_MAX + 0.8)) * plotW;
  const sy = (v: number) => plotH - (Math.min(v, yMax) / yMax) * plotH;
  const yTicks = Array.from({ length: Math.floor(yMax) + 1 }, (_, i) => i).filter((t) => t % 2 === 0);

  const insetPlotW = insetW - INSET_MARGIN.left - INSET_MARGIN.right;
  const insetPlotH = insetH - INSET_MARGIN.top - INSET_MARGIN.bottom;
  const maxCount = Math.max(1, ...displayCounts);
  const logTop = Math.log10(1 + maxCount) * 1.08;
  const syInset = (c: number) => insetPlotH - (Math.log10(1 + c) / logTop) * insetPlotH;
  const band = insetPlotW / BAR_XS.length;

  const perturb = (x: number, d: number) =>
    setDeltas((prev) => {
      const next = [...prev];
      if ((baseCounts[x] ?? 0) + next[x] + d < 0) return prev;
      next[x] += d;
      return next;
    });
  const perturbed = deltas.some((d) => d !== 0);
  const affected = (x: number) => selected !== null && deltas[selected] !== 0 && (x === selected || x === selected - 1);

  const tableXs = isMobile && !showAll ? EST_XS.slice(0, 5) : EST_XS;
  const axisColor = 'var(--color-text-muted)';

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Dataset:</span>
        {robbinsExplorerPresets.map((p) => (
          <button
            key={p.id}
            onClick={() => {
              setPresetId(p.id);
              resetDeltas();
            }}
            className={`rounded px-3 py-1 text-xs transition ${
              p.id === presetId
                ? 'bg-[var(--color-shrink)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {isSim && (
        <div className="mb-3 grid grid-cols-1 items-end gap-3 text-sm sm:grid-cols-[1fr_auto]">
          <label className="flex flex-col gap-1">
            <span className="flex items-center justify-between">
              <span>Policy holders N</span>
              <span className="font-mono text-xs">{N.toLocaleString('en-US')}</span>
            </span>
            <input
              type="range"
              min={LOG_N_MIN}
              max={LOG_N_MAX}
              step={0.01}
              value={logN}
              onChange={(e) => {
                setLogN(parseFloat(e.target.value));
                resetDeltas();
              }}
              className="w-full accent-[var(--color-f-model)]"
              aria-label="Number of simulated policy holders (log scale)"
            />
          </label>
          <button
            onClick={() => {
              setSeed((s) => s + 1);
              resetDeltas();
            }}
            className="rounded px-3 py-1.5 text-xs font-medium text-white"
            style={{ background: bayesianColors.shrink }}
          >
            Resample (seed {seed})
          </button>
        </div>
      )}

      <div className="flex flex-col gap-4 md:flex-row">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: bayesianColors.mle }}>- - MLE: θ̂ = x</span>
            <span style={{ color: bayesianColors.fModel }}>● Robbins (x+1)y₍ₓ₊₁₎/yₓ</span>
            <span style={{ color: bayesianColors.shrink }}>■ Gamma–Poisson EB</span>
            {isSim && <span style={{ color: bayesianColors.oracle }}>- - true E[θ | x]</span>}
          </div>
          <svg width={mainW} height={mainH} className="block" role="img" aria-label="Estimated next-year claim rate against this year's claim count">
            <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
              {yTicks.map((t) => (
                <line key={t} x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
              ))}
              <line
                x1={sx(0)}
                y1={sy(0)}
                x2={sx(Math.min(X_EST_MAX, yMax))}
                y2={sy(Math.min(X_EST_MAX, yMax))}
                stroke={bayesianColors.mle}
                strokeWidth={1.5}
                strokeDasharray="5 3"
              />
              {isSim && (
                <path
                  d={brokenPath(EST_XS, truth, sx, sy)}
                  fill="none"
                  stroke={bayesianColors.oracle}
                  strokeWidth={2}
                  strokeDasharray="2 3"
                />
              )}
              <path d={brokenPath(EST_XS, gammaYs, sx, sy)} fill="none" stroke={bayesianColors.shrink} strokeWidth={2} />
              {EST_XS.map((x) =>
                Number.isFinite(gammaYs[x]) ? (
                  <rect key={`g${x}`} x={sx(x) - 3.5} y={sy(gammaYs[x]) - 3.5} width={7} height={7} fill={bayesianColors.shrink} />
                ) : null,
              )}
              <path d={brokenPath(EST_XS, robbinsYs, sx, sy)} fill="none" stroke={bayesianColors.fModel} strokeWidth={2} />
              {EST_XS.map((x) => {
                const v = robbinsYs[x];
                if (!Number.isFinite(v)) return null;
                const clipped = v > yMax;
                return (
                  <g key={`r${x}`}>
                    <circle
                      cx={sx(x)}
                      cy={sy(v)}
                      r={affected(x) ? 6 : 4.5}
                      fill={bayesianColors.fModel}
                      stroke={affected(x) ? 'var(--color-text)' : 'none'}
                      strokeWidth={1.5}
                    />
                    {(clipped || x === 5 || affected(x)) && (
                      <text x={sx(x) + 8} y={sy(v) + 4} fontSize={10} fill={bayesianColors.fModel}>
                        {v.toFixed(2)}
                        {clipped ? ' ↑' : ''}
                      </text>
                    )}
                  </g>
                );
              })}
              <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
              <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
              {EST_XS.map((x) => (
                <g key={x}>
                  <line x1={sx(x)} y1={plotH} x2={sx(x)} y2={plotH + 4} stroke={axisColor} />
                  <text x={sx(x)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                    {x}
                  </text>
                </g>
              ))}
              {yTicks.map((t) => (
                <text key={t} x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
                  {t}
                </text>
              ))}
              <text x={plotW / 2} y={plotH + 32} textAnchor="middle" fontSize={11} fill="var(--color-text)">
                claims this year, x
              </text>
              <text
                x={-30}
                y={plotH / 2}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text)"
                transform={`rotate(-90 ${-30} ${plotH / 2})`}
              >
                Ê[θ | x]
              </text>
            </g>
          </svg>
        </div>

        <div className="shrink-0" style={{ width: isMobile ? '100%' : INSET_W }}>
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Counts yₓ (log scale) — click a bar to perturb it
          </div>
          <svg width={insetW} height={insetH} className="block" role="img" aria-label="Claim counts by number of claims, log scale">
            <g transform={`translate(${INSET_MARGIN.left}, ${INSET_MARGIN.top})`}>
              {COUNT_TICKS.filter((t) => Math.log10(1 + t) <= logTop).map((t) => (
                <g key={t}>
                  <line x1={0} y1={syInset(t)} x2={insetPlotW} y2={syInset(t)} stroke="var(--color-viz-grid)" />
                  <text x={-5} y={syInset(t) + 3} textAnchor="end" fontSize={9} fill={axisColor}>
                    {t >= 1000 ? `${t / 1000}k` : t}
                  </text>
                </g>
              ))}
              {BAR_XS.map((x) => {
                const c = displayCounts[x];
                const isSel = selected === x;
                return (
                  <g key={x} onClick={() => setSelected(x)} style={{ cursor: 'pointer' }}>
                    <rect x={x * band} y={0} width={band} height={insetPlotH} fill="transparent" />
                    <rect
                      x={x * band + band * 0.15}
                      y={syInset(c)}
                      width={band * 0.7}
                      height={Math.max(0, insetPlotH - syInset(c))}
                      fill={bayesianColors.fModel}
                      opacity={isSel ? 0.95 : 0.45}
                      stroke={isSel ? 'var(--color-text)' : 'none'}
                      strokeWidth={1.5}
                    />
                    <text x={x * band + band / 2} y={syInset(c) - 3} textAnchor="middle" fontSize={8.5} fill={axisColor}>
                      {c}
                    </text>
                    <text x={x * band + band / 2} y={insetPlotH + 13} textAnchor="middle" fontSize={10} fill={axisColor}>
                      {x === X_BAR_MAX && isSim ? '7+' : x}
                    </text>
                  </g>
                );
              })}
              <line x1={0} y1={insetPlotH} x2={insetPlotW} y2={insetPlotH} stroke={axisColor} />
              <text x={insetPlotW / 2} y={insetPlotH + 28} textAnchor="middle" fontSize={10} fill="var(--color-text)">
                x
              </text>
            </g>
          </svg>
          {selected !== null && (
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono">
                y<sub>{selected}</sub> = {counts[selected] ?? 0}
                {deltas[selected] !== 0 && ` (${deltas[selected] > 0 ? '+' : ''}${deltas[selected]})`}
              </span>
              <button
                onClick={() => perturb(selected, -1)}
                disabled={(counts[selected] ?? 0) <= 0}
                className="rounded border px-2 py-0.5 disabled:opacity-40"
                style={{ borderColor: 'var(--color-border)' }}
                aria-label={`Decrease count at x = ${selected}`}
              >
                −1
              </button>
              <button
                onClick={() => perturb(selected, 1)}
                className="rounded border px-2 py-0.5"
                style={{ borderColor: 'var(--color-border)' }}
                aria-label={`Increase count at x = ${selected}`}
              >
                +1
              </button>
              <button
                onClick={resetDeltas}
                disabled={!perturbed}
                className="rounded border px-2 py-0.5 disabled:opacity-40"
                style={{ borderColor: 'var(--color-border)' }}
              >
                Reset counts
              </button>
            </div>
          )}
          <div className="mt-2 font-mono text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Gamma fit: ν̂ = {gamma.shape.toFixed(3)}, ŝ = {gamma.scale.toFixed(3)}
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr style={{ color: 'var(--color-text-muted)' }}>
              <th className="py-1 text-left font-normal">x</th>
              <th className="py-1 text-right font-normal">yₓ</th>
              <th className="py-1 text-right font-normal" style={{ color: bayesianColors.fModel }}>
                Robbins
              </th>
              <th className="py-1 text-right font-normal" style={{ color: bayesianColors.shrink }}>
                Gamma EB
              </th>
              {isSim && (
                <th className="py-1 text-right font-normal" style={{ color: bayesianColors.oracle }}>
                  Truth
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {tableXs.map((x) => (
              <tr
                key={x}
                style={{
                  borderTop: '1px solid var(--color-border)',
                  background: affected(x) ? 'var(--color-surface-alt)' : undefined,
                }}
              >
                <td className="py-1">{x}</td>
                <td className="py-1 text-right">{counts[x] ?? 0}</td>
                <td className="py-1 text-right">{fmt(robbinsYs[x])}</td>
                <td className="py-1 text-right">{fmt(gammaYs[x])}</td>
                {isSim && <td className="py-1 text-right">{fmt(truth[x])}</td>}
              </tr>
            ))}
          </tbody>
        </table>
        {isMobile && (
          <button onClick={() => setShowAll((v) => !v)} className="mt-1 text-xs underline" style={{ color: 'var(--color-text-muted)' }}>
            {showAll ? 'Show fewer rows' : 'Show all rows'}
          </button>
        )}
      </div>
    </div>
  );
}
