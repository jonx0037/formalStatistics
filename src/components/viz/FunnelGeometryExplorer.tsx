/**
 * FunnelGeometryExplorer — interactive for Topic 28 §28.9 Thm 7.
 *
 * Reader toggles between centered (θ, log τ) and non-centered (θ̃, log τ)
 * parameterizations of Neal's funnel, adjusts HMC step-size ε, and runs
 * a live 500-iteration HMC trace. Centered mode exhibits the funnel
 * pathology — divergences concentrate at small τ; non-centered mode
 * gives a well-mixed spherical Gaussian.
 *
 * Target densities:
 *   centered:     log p(θ, log τ) = log 𝒩(θ; 0, e^{2 log τ})
 *                                  + log 𝒩(log τ; 0, 1)
 *   non-centered: log p(θ̃, log τ) = log 𝒩(θ̃; 0, 1) + log 𝒩(log τ; 0, 1)
 *
 * HMC runs locally: small-L leapfrog with finite-difference gradients
 * and energy-deviation divergence detection (threshold ΔH > 1000).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { funnelLogDensity, createSeededRng } from './shared/bayes';
import { bayesianColors } from './shared/colorScales';

const MARGIN = { top: 32, right: 24, bottom: 44, left: 56 };
const MOBILE_BREAKPOINT = 640;
const LOG_TAU_MIN = -4;
const LOG_TAU_MAX = 3;
const THETA_MIN = -8;
const THETA_MAX = 8;
const EPS_MIN_LOG = Math.log(0.001);
const EPS_MAX_LOG = Math.log(1);
const HMC_STEPS = 500;
const LEAPFROG_L = 20;
const DIVERGENCE_THRESHOLD = 1000;

type Mode = 'centered' | 'noncentered';
type TracePoint = { q0: number; q1: number; divergent: boolean };

// Negative log-density (potential energy) + gradient for each parameterization.
// q = [θ_or_thetaTilde, logTau]
function potentialAndGrad(mode: Mode, q: [number, number]): { U: number; grad: [number, number] } {
  const theta = q[0];
  const logTau = q[1];
  if (mode === 'centered') {
    // U = -log p(θ, log τ) for centered Neal's funnel.
    const U = -funnelLogDensity(theta, logTau);
    // ∂U/∂θ = θ · e^{-2 log τ};  ∂U/∂(log τ) = 1 − θ² · e^{-2 log τ} + log τ
    // (derived from grad of -log N(θ;0,e^{2 log τ}) - log N(log τ;0,1))
    const eInv = Math.exp(-2 * logTau);
    const dTheta = theta * eInv;
    const dLogTau = 1 - theta * theta * eInv + logTau;
    return { U, grad: [dTheta, dLogTau] };
  }
  // Non-centered: N(0,1) × N(0,1). Spherical.
  const U = 0.5 * (theta * theta + logTau * logTau) + Math.log(2 * Math.PI);
  return { U, grad: [theta, logTau] };
}

function leapfrog(
  mode: Mode,
  q0: [number, number],
  p0: [number, number],
  eps: number,
  L: number,
): {
  q: [number, number];
  p: [number, number];
  divergent: boolean;
  deltaH: number;
} {
  const q: [number, number] = [q0[0], q0[1]];
  const p: [number, number] = [p0[0], p0[1]];
  const { U: U0, grad: g0 } = potentialAndGrad(mode, q);
  const H0 = U0 + 0.5 * (p[0] * p[0] + p[1] * p[1]);
  // half step
  p[0] -= 0.5 * eps * g0[0];
  p[1] -= 0.5 * eps * g0[1];
  for (let l = 0; l < L; l++) {
    q[0] += eps * p[0];
    q[1] += eps * p[1];
    const { grad } = potentialAndGrad(mode, q);
    const halfOrFull = l === L - 1 ? 0.5 : 1;
    p[0] -= halfOrFull * eps * grad[0];
    p[1] -= halfOrFull * eps * grad[1];
    // Defensive divergence detection mid-trajectory.
    if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) {
      return { q: q0, p: p0, divergent: true, deltaH: Infinity };
    }
  }
  const { U: U1 } = potentialAndGrad(mode, q);
  const H1 = U1 + 0.5 * (p[0] * p[0] + p[1] * p[1]);
  const deltaH = H1 - H0;
  const divergent = !Number.isFinite(H1) || Math.abs(deltaH) > DIVERGENCE_THRESHOLD;
  return { q, p, divergent, deltaH };
}

export default function FunnelGeometryExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>('centered');
  const [epsSlider, setEpsSlider] = useState(
    (Math.log(0.1) - EPS_MIN_LOG) / (EPS_MAX_LOG - EPS_MIN_LOG),
  );
  const eps = Math.exp(EPS_MIN_LOG + epsSlider * (EPS_MAX_LOG - EPS_MIN_LOG));
  const [trace, setTrace] = useState<TracePoint[]>([]);
  const [running, setRunning] = useState(false);
  const rafRef = useRef<number | null>(null);
  const cancelRef = useRef(false);

  // Precompute density heatmap on the current mode.
  const HEAT_N = 48;
  const heatmap = useMemo(() => {
    const g: number[][] = [];
    let vMax = -Infinity;
    let vMin = Infinity;
    for (let i = 0; i < HEAT_N; i++) {
      const row: number[] = [];
      const logTau = LOG_TAU_MIN + (i / (HEAT_N - 1)) * (LOG_TAU_MAX - LOG_TAU_MIN);
      for (let j = 0; j < HEAT_N; j++) {
        const theta = THETA_MIN + (j / (HEAT_N - 1)) * (THETA_MAX - THETA_MIN);
        const v =
          mode === 'centered'
            ? funnelLogDensity(theta, logTau)
            : -0.5 * (theta * theta + logTau * logTau) - Math.log(2 * Math.PI);
        row.push(v);
        if (v > vMax) vMax = v;
        if (v < vMin) vMin = v;
      }
      g.push(row);
    }
    return { g, vMax, vMin };
  }, [mode]);

  const runHMC = async () => {
    cancelRef.current = false;
    setRunning(true);
    setTrace([]);
    const rng = createSeededRng(12345);
    // Starting point: roughly in the middle of the density's support.
    let q: [number, number] = [0.5, 0];
    const pts: TracePoint[] = [];

    const stepBatch = () => {
      if (cancelRef.current) {
        setRunning(false);
        return;
      }
      const BATCH = 20;
      for (let b = 0; b < BATCH && pts.length < HMC_STEPS; b++) {
        const p: [number, number] = [rng.normal(), rng.normal()];
        const { q: qNew, divergent, deltaH } = leapfrog(mode, q, p, eps, LEAPFROG_L);
        if (divergent) {
          pts.push({ q0: q[0], q1: q[1], divergent: true });
          continue;
        }
        // Metropolis-Hastings accept/reject on Δ𝐻 — promotes unadjusted
        // leapfrog into a genuine HMC kernel that targets p(q) exactly.
        // (Copilot PR-32; mirrors shared/bayes.ts hamiltonianMonteCarlo.)
        const accept = rng.random() < Math.min(1, Math.exp(-deltaH));
        if (accept) q = qNew;
        pts.push({ q0: q[0], q1: q[1], divergent: false });
      }
      setTrace([...pts]);
      if (pts.length < HMC_STEPS) {
        rafRef.current = requestAnimationFrame(stepBatch);
      } else {
        setRunning(false);
      }
    };
    rafRef.current = requestAnimationFrame(stepBatch);
  };

  const stopHMC = () => {
    cancelRef.current = true;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setRunning(false);
  };

  useEffect(() => {
    return () => {
      cancelRef.current = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    cancelRef.current = true;
    setTrace([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const divergences = trace.filter((t) => t.divergent).length;

  // Layout.
  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(620, (width ?? 620) - 16));
  const chartH = isMobile ? 340 : 420;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;
  const cellW = plotW / (HEAT_N - 1);
  const cellH = plotH / (HEAT_N - 1);

  const xOf = (theta: number) =>
    ((theta - THETA_MIN) / (THETA_MAX - THETA_MIN)) * plotW;
  const yOf = (logTau: number) =>
    plotH - ((logTau - LOG_TAU_MIN) / (LOG_TAU_MAX - LOG_TAU_MIN)) * plotH;

  // Heatmap color: white (low) → slate-700 (high).
  const heatColor = (v: number) => {
    const t = Math.max(0, Math.min(1, (v - heatmap.vMin) / (heatmap.vMax - heatmap.vMin || 1)));
    const rr = Math.round(248 + (51 - 248) * t);
    const gg = Math.round(250 + (65 - 250) * t);
    const bb = Math.round(252 + (85 - 252) * t);
    return `rgb(${rr},${gg},${bb})`;
  };

  // Memoize the 2,304 heatmap <rect>s so React doesn't reconcile them on every
  // HMC-trace frame (≈60 fps). Background depends only on mode + layout.
  // (gemini PR-32)
  const heatmapRects = useMemo(
    () =>
      heatmap.g.flatMap((row, i) =>
        row.map((v, j) => (
          <rect
            key={`${i}-${j}`}
            x={j * cellW - cellW / 2}
            y={plotH - i * cellH - cellH / 2}
            width={cellW + 1}
            height={cellH + 1}
            fill={heatColor(v)}
          />
        )),
      ),
    // heatColor closes over heatmap.vMin/vMax, so listing heatmap alone
    // is sufficient; cellW/cellH/plotH may change under resize.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [heatmap, cellW, cellH, plotH],
  );

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Mode toggle */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Parameterization:</span>
        <div
          className="inline-flex overflow-hidden rounded border text-xs"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {(['centered', 'noncentered'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              disabled={running}
              className={`px-3 py-1.5 transition ${
                mode === m
                  ? 'bg-[var(--color-shrink)] text-white'
                  : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
              }`}
            >
              {m === 'centered' ? 'Centered (θ, log τ)' : 'Non-centered (θ̃, log τ)'}
            </button>
          ))}
        </div>
      </div>

      {/* ε slider + controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>HMC step size ε (log scale)</span>
            <span className="font-mono">{eps.toFixed(eps < 0.01 ? 4 : eps < 0.1 ? 3 : 2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={epsSlider}
            disabled={running}
            onChange={(e) => setEpsSlider(parseFloat(e.target.value))}
            className="accent-[var(--color-shrink)]"
          />
        </label>
        <div className="flex items-end gap-2">
          <button
            onClick={running ? stopHMC : runHMC}
            className="rounded px-3 py-1.5 text-xs font-medium text-white transition"
            style={{ background: running ? bayesianColors.divergence : bayesianColors.shrink }}
          >
            {running ? 'Stop' : `Run HMC (${HMC_STEPS} steps)`}
          </button>
          {trace.length > 0 && (
            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              Drawn: {trace.length} · Divergences:{' '}
              <span style={{ color: bayesianColors.divergence, fontWeight: 600 }}>
                {divergences}
              </span>
            </div>
          )}
        </div>
      </div>

      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Density heatmap (memoized — see heatmapRects) */}
          {heatmapRects}

          {/* Trace (faded where older; opaque where latest) */}
          {trace.map((pt, i) => {
            const age = i / Math.max(1, trace.length - 1);
            if (pt.divergent) {
              return (
                <g key={`div-${i}`}>
                  <circle
                    cx={xOf(pt.q0)}
                    cy={yOf(pt.q1)}
                    r={5}
                    fill={bayesianColors.divergence}
                    stroke="white"
                    strokeWidth={1}
                  />
                  <line
                    x1={xOf(pt.q0) - 4}
                    y1={yOf(pt.q1) - 4}
                    x2={xOf(pt.q0) + 4}
                    y2={yOf(pt.q1) + 4}
                    stroke="white"
                    strokeWidth={1.5}
                  />
                  <line
                    x1={xOf(pt.q0) + 4}
                    y1={yOf(pt.q1) - 4}
                    x2={xOf(pt.q0) - 4}
                    y2={yOf(pt.q1) + 4}
                    stroke="white"
                    strokeWidth={1.5}
                  />
                </g>
              );
            }
            return (
              <circle
                key={`tr-${i}`}
                cx={xOf(pt.q0)}
                cy={yOf(pt.q1)}
                r={1.8}
                fill={bayesianColors.shrink}
                opacity={0.25 + 0.65 * age}
              />
            );
          })}

          {/* Axis frame */}
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="var(--color-text-muted)" />
          <line x1={0} y1={0} x2={0} y2={plotH} stroke="var(--color-text-muted)" />
          {/* x ticks */}
          {[-6, -4, -2, 0, 2, 4, 6].map((t) => (
            <g key={t}>
              <line x1={xOf(t)} y1={plotH} x2={xOf(t)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text
                x={xOf(t)}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-text-muted)"
              >
                {t}
              </text>
            </g>
          ))}
          <text
            x={plotW / 2}
            y={plotH + 34}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
          >
            {mode === 'centered' ? 'θ (group-1 mean)' : 'θ̃ (group-1 non-centered)'}
          </text>
          {/* y ticks */}
          {[-4, -2, 0, 2].map((t) => (
            <g key={t}>
              <line x1={-4} y1={yOf(t)} x2={0} y2={yOf(t)} stroke="var(--color-text-muted)" />
              <text
                x={-8}
                y={yOf(t) + 4}
                textAnchor="end"
                fontSize={10}
                fill="var(--color-text-muted)"
              >
                {t}
              </text>
            </g>
          ))}
          <text
            x={-42}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-42} ${plotH / 2})`}
          >
            log τ (group SD)
          </text>
        </g>
      </svg>

      <div className="mt-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        {mode === 'centered' ? (
          <span>
            Centered: the pinched funnel at small τ forces θ → 0, producing divergent
            trajectories under large step sizes. Try raising ε to trigger more divergences.
          </span>
        ) : (
          <span>
            Non-centered: the prior on θ̃ is 𝒩(0,1) independent of τ — a spherical Gaussian
            in this joint space. HMC mixes freely regardless of ε (§28.9 Thm 7).
          </span>
        )}
      </div>
    </div>
  );
}
