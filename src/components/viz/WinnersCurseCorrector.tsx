/**
 * WinnersCurseCorrector — interactive for Topic 33 §33.5.
 *
 * The ML payoff of Thm 5 (the Bayes rule is immune to selection): score K
 * model configurations once each on a noisy validation set, rank them, and
 * compare the top k's observed scores with their Tweedie–KDE corrections
 * (fit to the K scores alone, g unknown), the oracle Bayes rule (g known),
 * and the true scores θ.
 *
 * Readouts: mean overstatement among the top k for each estimator, and how
 * often the observed #1 is truly in the top 10% — a Monte Carlo frequency
 * over 200 fresh leaderboards drawn with the same settings.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  sampleNormalMixture,
  tweedieKDE,
  posteriorMeanNormalMixturePrior,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { winnersCursePresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 14, right: 16, bottom: 40, left: 46 };
const NARROW_BREAKPOINT = 480;
const TOP_K_MAX = 50;
const TOP_K_MAX_NARROW = 20;
const MC_REPS = 200;

function signed(v: number): string {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
}

function Marker({ shape, cx, cy, color }: { shape: 'circle' | 'triangle' | 'diamond' | 'square'; cx: number; cy: number; color: string }) {
  switch (shape) {
    case 'circle':
      return <circle cx={cx} cy={cy} r={4} fill={color} />;
    case 'triangle':
      return <path d={`M${cx},${cy - 5}L${cx + 4.5},${cy + 3.5}L${cx - 4.5},${cy + 3.5}Z`} fill={color} />;
    case 'diamond':
      return <path d={`M${cx},${cy - 5}L${cx + 4.5},${cy}L${cx},${cy + 5}L${cx - 4.5},${cy}Z`} fill={color} />;
    case 'square':
      return <rect x={cx - 3.5} y={cy - 3.5} width={7} height={7} fill={color} />;
  }
}

function LegendMarker({ shape, color, label }: { shape: 'circle' | 'triangle' | 'diamond' | 'square'; color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={12} height={12} aria-hidden="true">
        <Marker shape={shape} cx={6} cy={6.5} color={color} />
      </svg>
      <span>{label}</span>
    </span>
  );
}

export default function WinnersCurseCorrector() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = winnersCursePresets[presetIdx];
  const [K, setK] = useState(preset.K);
  const [topK, setTopK] = useState(preset.topK);
  const [sigma, setSigma] = useState(preset.sigma);
  const [seed, setSeed] = useState(preset.seed);
  const [showOracle, setShowOracle] = useState(true);

  const selectPreset = (i: number) => {
    const p = winnersCursePresets[i];
    setPresetIdx(i);
    setK(p.K);
    setTopK(p.topK);
    setSigma(p.sigma);
    setSeed(p.seed);
  };

  const sigmaSq = sigma * sigma;

  // One leaderboard: θ_i ~ g, x_i = θ_i + σ ε_i; the top TOP_K_MAX by x.
  const board = useMemo(() => {
    const rng = createSeededRng(seed);
    const theta = sampleNormalMixture(preset.comps, K, rng);
    const x = theta.map((t) => t + sigma * rng.normal());
    const order = x.map((_, i) => i).sort((a, b) => x[b] - x[a]);
    const top = order.slice(0, Math.min(TOP_K_MAX, K));
    const topX = top.map((i) => x[i]);
    // Tweedie–KDE fit to all K scores (Silverman bandwidth), evaluated at the leaders.
    const tweedie = tweedieKDE(x, topX, sigmaSq);
    const oracle = topX.map((v) => posteriorMeanNormalMixturePrior(v, preset.comps, sigmaSq));
    const leaderTheta = theta[order[0]];
    const atLeastShare = theta.filter((t) => t >= leaderTheta).length / K;
    return { theta, x, top, topX, tweedie, oracle, leaderTheta, atLeastShare };
  }, [preset, K, sigma, sigmaSq, seed]);

  // How often is the observed #1 truly in the top 10%? (ties at the spike count against it)
  const leaderHitRate = useMemo(() => {
    let hits = 0;
    for (let r = 0; r < MC_REPS; r++) {
      const rng = createSeededRng(seed * 1009 + r + 1);
      const theta = sampleNormalMixture(preset.comps, K, rng);
      let best = -Infinity;
      let bestTheta = 0;
      for (let i = 0; i < K; i++) {
        const xi = theta[i] + sigma * rng.normal();
        if (xi > best) {
          best = xi;
          bestTheta = theta[i];
        }
      }
      let atLeast = 0;
      for (const t of theta) if (t >= bestTheta) atLeast++;
      if (atLeast <= 0.1 * K) hits++;
    }
    return hits / MC_REPS;
  }, [preset, K, sigma, seed]);

  // Layout.
  const containerW = width || 700;
  const narrow = containerW < NARROW_BREAKPOINT;
  const kShown = Math.min(topK, narrow ? TOP_K_MAX_NARROW : TOP_K_MAX, board.top.length);
  const chartW = Math.max(260, containerW);
  const chartH = narrow ? 300 : 340;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const ranks = Array.from({ length: kShown }, (_, r) => r);
  const obs = ranks.map((r) => board.topX[r]);
  const tw = ranks.map((r) => board.tweedie[r]);
  const orc = ranks.map((r) => board.oracle[r]);
  const tru = ranks.map((r) => board.theta[board.top[r]]);
  const all = [...obs, ...tw, ...tru, ...(showOracle ? orc : [])];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = 0.08 * (hi - lo || 1);
  const yMin = lo - pad;
  const yMax = hi + pad;
  const band = plotW / Math.max(1, kShown);
  const sx = (r: number) => (r + 0.5) * band;
  const sy = (v: number) => plotH - ((v - yMin) / (yMax - yMin)) * plotH;
  const step = yMax - yMin > 6 ? 2 : yMax - yMin > 2.5 ? 1 : 0.5;
  const yTicks: number[] = [];
  for (let t = Math.ceil(yMin / step) * step; t <= yMax; t += step) yTicks.push(Number(t.toFixed(2)));
  const xLabelEvery = kShown > 25 ? 5 : kShown > 10 ? 2 : 1;

  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
  const over = {
    naive: mean(ranks.map((r) => obs[r] - tru[r])),
    tweedie: mean(ranks.map((r) => tw[r] - tru[r])),
    oracle: mean(ranks.map((r) => orc[r] - tru[r])),
  };

  const axisColor = 'var(--color-text-muted)';
  const off = Math.min(3, band * 0.15);

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Scenario:</span>
        {winnersCursePresets.map((p, i) => (
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

      <div className="mb-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Configurations K</span>
            <span className="font-mono text-xs">{K.toLocaleString('en-US')}</span>
          </span>
          <input
            type="range"
            min={50}
            max={2000}
            step={10}
            value={K}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-f-model)]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Top k shown</span>
            <span className="font-mono text-xs">{kShown}</span>
          </span>
          <input
            type="range"
            min={1}
            max={narrow ? TOP_K_MAX_NARROW : TOP_K_MAX}
            step={1}
            value={Math.min(topK, narrow ? TOP_K_MAX_NARROW : TOP_K_MAX)}
            onChange={(e) => setTopK(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-f-model)]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Validation-set noise σ</span>
            <span className="font-mono text-xs">{sigma.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={sigma}
            onChange={(e) => setSigma(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-f-model)]"
          />
        </label>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="rounded px-3 py-1.5 text-xs font-medium text-white"
          style={{ background: bayesianColors.shrink }}
        >
          Resample (seed {seed})
        </button>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={showOracle} onChange={(e) => setShowOracle(e.target.checked)} />
          Show oracle (g known)
        </label>
      </div>

      <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        <LegendMarker shape="circle" color={bayesianColors.mle} label="observed score x" />
        <LegendMarker shape="triangle" color={bayesianColors.fModel} label="Tweedie–KDE (g unknown)" />
        {showOracle && <LegendMarker shape="diamond" color={bayesianColors.oracle} label="oracle E[θ | x]" />}
        <LegendMarker shape="square" color={bayesianColors.true} label="true score θ" />
      </div>
      <svg width={chartW} height={chartH} className="block" role="img" aria-label="Top configurations by observed score, with corrected and true scores">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
              <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
                {t}
              </text>
            </g>
          ))}
          {ranks.map((r) => (
            <line
              key={`c${r}`}
              x1={sx(r)}
              y1={sy(obs[r])}
              x2={sx(r)}
              y2={sy(tru[r])}
              stroke="var(--color-text-muted)"
              strokeWidth={1}
              opacity={0.5}
            />
          ))}
          {ranks.map((r) => (
            <g key={`m${r}`}>
              <Marker shape="square" cx={sx(r) + off} cy={sy(tru[r])} color={bayesianColors.true} />
              {showOracle && <Marker shape="diamond" cx={sx(r) + off} cy={sy(orc[r])} color={bayesianColors.oracle} />}
              <Marker shape="triangle" cx={sx(r) - off} cy={sy(tw[r])} color={bayesianColors.fModel} />
              <Marker shape="circle" cx={sx(r) - off} cy={sy(obs[r])} color={bayesianColors.mle} />
            </g>
          ))}
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
          <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
          {ranks
            .filter((r) => r === 0 || (r + 1) % xLabelEvery === 0)
            .map((r) => (
              <text key={r} x={sx(r)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                {r + 1}
              </text>
            ))}
          <text x={plotW / 2} y={plotH + 32} textAnchor="middle" fontSize={11} fill="var(--color-text)">
            rank by observed validation score
          </text>
          <text
            x={-32}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-32} ${plotH / 2})`}
          >
            score
          </text>
        </g>
      </svg>

      <div className="mt-3 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Mean overstatement, top {kShown}
          </div>
          <table className="mt-2 w-full font-mono text-xs">
            <tbody>
              <tr>
                <td style={{ color: bayesianColors.mle }}>Naive (x)</td>
                <td className="text-right">{signed(over.naive)}</td>
              </tr>
              <tr>
                <td style={{ color: bayesianColors.fModel }}>Tweedie–KDE</td>
                <td className="text-right">{signed(over.tweedie)}</td>
              </tr>
              <tr>
                <td style={{ color: bayesianColors.oracle }}>Oracle Bayes</td>
                <td className="text-right">{signed(over.oracle)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="rounded border p-3 text-xs" style={{ borderColor: 'var(--color-border)' }}>
          <div className="uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Is the leader really a leader?
          </div>
          <div className="mt-2">
            This leaderboard's #1 truly scores <span className="font-mono">θ = {board.leaderTheta.toFixed(2)}</span>;{' '}
            <span className="font-mono">{(100 * board.atLeastShare).toFixed(1)}%</span> of the configurations score at least
            that well.
          </div>
          <div className="mt-1">
            Across {MC_REPS} simulated leaderboards like this one, the observed #1 is truly in the top 10% in{' '}
            <span className="font-mono">{Math.round(100 * leaderHitRate)}%</span> of them.
          </div>
        </div>
      </div>
    </div>
  );
}
