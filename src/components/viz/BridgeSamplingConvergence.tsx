/**
 * BridgeSamplingConvergence — interactive for Topic 27 §27.6.
 *
 * Trace the iterative bridge-sampling fixed point converging to the
 * closed-form Beta-Binomial log marginal likelihood on three preset
 * problems, with the naive-IS estimator overlaid for contrast.
 *
 * Pedagogy: the iteration converges in < 5 steps when the proposal has
 * reasonable overlap with the target; the naive IS point estimator
 * (horizontal line) has higher variance despite the same draw budget.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  betaBinomialLogMarginal,
  bridgeSamplingEstimate,
  importanceSamplingEstimate,
} from './shared/bayes';
import { sampleGammaShape, lnGamma, pdfBeta } from './shared/distributions';
import { bayesianColors } from './shared/colorScales';
import { bridgeSamplingPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 20, right: 24, bottom: 40, left: 56 };
const MOBILE_BREAKPOINT = 640;
const N_DRAWS = 4000;
const MAX_ITER = 10;

type BridgeTrace = {
  closedForm: number;
  bridgeTrace: number[];
  isEstimate: number;
  finalLogM: number;
  iterations: number;
};

/** Run one bridge iteration at a time to capture the convergence trace. */
function runBridgeTrace(
  n: number,
  k: number,
  alpha: number,
  beta: number,
  nDraws: number,
  seed: number,
): BridgeTrace {
  const rng = createSeededRng(seed);
  const aPost = alpha + k;
  const bPost = beta + n - k;
  const aProp = aPost * 0.9 + 1;
  const bProp = bPost * 0.9 + 1;

  const sampleBeta = (a: number, b: number): number => {
    const x = sampleGammaShape(a, () => rng.random());
    const y = sampleGammaShape(b, () => rng.random());
    return x / (x + y);
  };

  const posteriorDraws: number[][] = [];
  const proposalDraws: number[][] = [];
  for (let i = 0; i < nDraws; i++) posteriorDraws.push([sampleBeta(aPost, bPost)]);
  for (let i = 0; i < nDraws; i++) proposalDraws.push([sampleBeta(aProp, bProp)]);

  const logBinom = lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
  const logBPrior =
    lnGamma(alpha) + lnGamma(beta) - lnGamma(alpha + beta);
  const logUnnormPost = (theta: number[]): number => {
    const t = theta[0];
    return (
      logBinom
      + k * Math.log(t)
      + (n - k) * Math.log1p(-t)
      + (alpha - 1) * Math.log(t)
      + (beta - 1) * Math.log1p(-t)
      - logBPrior
    );
  };
  const logProposal = (theta: number[]): number =>
    Math.log(pdfBeta(theta[0], aProp, bProp));

  // Capture iteration trace by calling bridgeSamplingEstimate with increasing maxIter.
  const bridgeTrace: number[] = [];
  let prev = 0;
  for (let iter = 1; iter <= MAX_ITER; iter++) {
    const r = bridgeSamplingEstimate({
      posteriorDraws,
      proposalDraws,
      logUnnormPost,
      logProposal,
      maxIter: iter,
      tol: 0,
    });
    bridgeTrace.push(r.logMarginal);
    if (iter > 1 && Math.abs(r.logMarginal - prev) < 1e-8) break;
    prev = r.logMarginal;
  }
  const final = bridgeSamplingEstimate({
    posteriorDraws,
    proposalDraws,
    logUnnormPost,
    logProposal,
  });

  const is = importanceSamplingEstimate({
    proposalDraws,
    logUnnormPost,
    logProposal,
  });

  return {
    closedForm: betaBinomialLogMarginal(k, n, alpha, beta),
    bridgeTrace,
    isEstimate: is.logMarginal,
    finalLogM: final.logMarginal,
    iterations: final.iterations,
  };
}

export default function BridgeSamplingConvergence() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [seed, setSeed] = useState(42);

  const preset = bridgeSamplingPresets[presetIdx];
  const trace = useMemo(
    () => runBridgeTrace(preset.n, preset.k, preset.alpha, preset.beta, N_DRAWS, seed),
    [preset, seed],
  );

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(720, (width ?? 720) - 16));
  const chartH = isMobile ? 280 : 340;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // y-scale: envelope bridge trace + IS + closed-form
  const allYs = [...trace.bridgeTrace, trace.isEstimate, trace.closedForm];
  let yMin = Math.min(...allYs);
  let yMax = Math.max(...allYs);
  const yPad = Math.max(0.05, (yMax - yMin) * 0.4);
  yMin -= yPad;
  yMax += yPad;
  const nPoints = trace.bridgeTrace.length;

  const xOf = (iter: number) => (iter / Math.max(1, nPoints - 1)) * plotW;
  const yOf = (val: number) => plotH * (1 - (val - yMin) / (yMax - yMin));

  const bridgePath = trace.bridgeTrace
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xOf(i).toFixed(2)} ${yOf(v).toFixed(2)}`)
    .join(' ');

  const xTicks = Array.from({ length: nPoints }, (_, i) => i);
  const yRange = yMax - yMin;
  const yTickStep = yRange > 0.5 ? 0.2 : yRange > 0.1 ? 0.05 : 0.01;
  const yTicks: number[] = [];
  const yStart = Math.ceil(yMin / yTickStep) * yTickStep;
  for (let v = yStart; v <= yMax; v += yTickStep) yTicks.push(v);

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {bridgeSamplingPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => setPresetIdx(i)}
            className={`rounded px-3 py-1 text-xs transition ${
              i === presetIdx
                ? 'bg-[var(--color-posterior)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded bg-[var(--color-surface)] px-3 py-1 text-xs hover:bg-[var(--color-surface-hover)]"
          style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
        >
          New seed ({seed})
        </button>
      </div>

      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* y gridlines + labels */}
          {yTicks.map((v) => (
            <g key={`y-${v}`}>
              <line x1={0} y1={yOf(v)} x2={plotW} y2={yOf(v)} stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2 2" />
              <text x={-6} y={yOf(v) + 4} textAnchor="end" fontSize={10} fill="var(--color-text-muted)">
                {v.toFixed(2)}
              </text>
            </g>
          ))}
          {/* x gridlines + labels */}
          {xTicks.map((i) => (
            <g key={`x-${i}`}>
              <line x1={xOf(i)} y1={plotH} x2={xOf(i)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text x={xOf(i)} y={plotH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">
                {i + 1}
              </text>
            </g>
          ))}
          {/* Closed-form reference */}
          <line
            x1={0}
            y1={yOf(trace.closedForm)}
            x2={plotW}
            y2={yOf(trace.closedForm)}
            stroke={bayesianColors.true}
            strokeWidth={1.5}
            strokeDasharray="6 3"
          />
          <text x={plotW - 4} y={yOf(trace.closedForm) - 4} textAnchor="end" fontSize={10} fill={bayesianColors.true}>
            closed form
          </text>
          {/* IS estimate */}
          <line
            x1={0}
            y1={yOf(trace.isEstimate)}
            x2={plotW}
            y2={yOf(trace.isEstimate)}
            stroke={bayesianColors.likelihood}
            strokeWidth={1.2}
            strokeDasharray="2 3"
          />
          <text x={plotW - 4} y={yOf(trace.isEstimate) + 12} textAnchor="end" fontSize={10} fill={bayesianColors.likelihood}>
            naive IS
          </text>
          {/* Bridge trace */}
          <path d={bridgePath} fill="none" stroke={bayesianColors.posterior} strokeWidth={2} />
          {trace.bridgeTrace.map((v, i) => (
            <circle key={i} cx={xOf(i)} cy={yOf(v)} r={3.5} fill={bayesianColors.posterior} />
          ))}
          {/* axis labels */}
          <text x={plotW / 2} y={plotH + 34} textAnchor="middle" fontSize={11} fill="var(--color-text)">
            Iteration
          </text>
          <text
            x={-36}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-36} ${plotH / 2})`}
          >
            log m̂(y)
          </text>
        </g>
      </svg>

      <div className="mt-3 space-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        <div>
          <span className="font-semibold" style={{ color: 'var(--color-text)' }}>Closed-form log m:</span>{' '}
          {trace.closedForm.toFixed(6)}
        </div>
        <div>
          <span className="font-semibold" style={{ color: 'var(--color-posterior)' }}>Bridge estimate:</span>{' '}
          {trace.finalLogM.toFixed(6)} (converged in {trace.iterations} iter)
        </div>
        <div>
          <span className="font-semibold" style={{ color: 'var(--color-likelihood)' }}>Naive IS:</span>{' '}
          {trace.isEstimate.toFixed(6)}
        </div>
        <div className="mt-2 italic">{preset.description}</div>
      </div>
    </div>
  );
}
