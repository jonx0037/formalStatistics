import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  computeMLE,
  digamma,
  trigamma,
  logGamma,
  logLikelihood,
  logLikelihoodCurve,
  newtonRaphsonStep,
  fisherScoringStep,
} from './shared/estimation';
import {
  exponentialSample,
  poissonSample,
  sampleSequence,
  gammaSample,
} from './shared/convergence';

const MARGIN = { top: 20, right: 16, bottom: 40, left: 56 };
const H = 280;

type PresetId = 'gamma' | 'poisson';

interface Preset {
  id: PresetId;
  name: string;
  trueParam: number;
  otherParams: Record<string, number>;
  thetaRange: [number, number];
  defaultInit: number;
  defaultN: number;
  description: string;
  sample(n: number, rng: () => number): number[];
  logPdf(x: number, theta: number): number;
  score(x: number, theta: number): number; // per-observation score
  hessian(x: number, theta: number): number; // per-observation Hessian of log f
  fisherInfo(theta: number): number; // per-observation expected Fisher info
}

const PRESETS: Preset[] = [
  {
    id: 'gamma',
    name: 'Gamma(α=3, β=2) shape — no closed form',
    trueParam: 3,
    otherParams: { beta: 2 },
    thetaRange: [0.3, 7],
    defaultInit: 1.0,
    defaultN: 50,
    description: 'Score involves the digamma function ψ(α); Newton typically converges in 4–6 steps.',
    sample: (n, rng) => sampleSequence(() => gammaSample(3, 2, rng), n),
    logPdf: (x, alpha) => {
      if (alpha <= 0 || x <= 0) return -Infinity;
      return alpha * Math.log(2) - logGamma(alpha) + (alpha - 1) * Math.log(x) - 2 * x;
    },
    score: (x, alpha) => Math.log(2) - digamma(alpha) + Math.log(Math.max(x, 1e-300)),
    hessian: (_x, alpha) => -trigamma(alpha),
    fisherInfo: (alpha) => trigamma(alpha),
  },
  {
    id: 'poisson',
    name: 'Poisson(λ=3) rate — one-step convergence',
    trueParam: 3,
    otherParams: {},
    thetaRange: [0.5, 7],
    defaultInit: 1.5,
    defaultN: 50,
    description: 'Closed-form MLE λ̂ = X̄. From any start, Newton converges in ~1 step.',
    sample: (n, rng) => sampleSequence(() => poissonSample(3, rng), n),
    logPdf: (x, lambda) => {
      if (lambda <= 0) return -Infinity;
      return x * Math.log(lambda) - lambda - logGamma(x + 1);
    },
    score: (x, lambda) => x / lambda - 1,
    hessian: (x, lambda) => -x / (lambda * lambda),
    fisherInfo: (lambda) => 1 / lambda,
  },
];

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export default function NewtonRaphsonExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const [n, setN] = useState(PRESETS[0].defaultN);
  const [init, setInit] = useState(PRESETS[0].defaultInit);
  const [iterates, setIterates] = useState<number[]>([PRESETS[0].defaultInit]);
  const [useFisher, setUseFisher] = useState(false);
  const [seed, setSeed] = useState(1);

  const leftSvgRef = useRef<SVGSVGElement | null>(null);
  const rightSvgRef = useRef<SVGSVGElement | null>(null);

  const preset = PRESETS[presetIndex];
  const rng = useMemo(() => makeLCG(seed * 7919 + n), [seed, n]);
  const data = useMemo(() => preset.sample(n, rng), [preset, n, rng]);

  // True MLE (for the convergence reference θ̂).
  const mleRef = useMemo(() => {
    const family = preset.id === 'gamma' ? 'Gamma' : 'Poisson';
    const paramName = preset.id === 'gamma' ? 'alpha' : 'lambda';
    return computeMLE(data, family, paramName, preset.otherParams);
  }, [data, preset]);

  // Reset iterates when preset / n / seed changes.
  useEffect(() => {
    setInit(preset.defaultInit);
    setIterates([preset.defaultInit]);
  }, [presetIndex, n, seed]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = useCallback(() => {
    setIterates((path) => {
      const theta = path[path.length - 1];
      const result = useFisher
        ? fisherScoringStep(theta, data, preset.score, preset.fisherInfo)
        : newtonRaphsonStep(theta, data, preset.score, preset.hessian);
      const next = result.nextTheta;
      if (!Number.isFinite(next)) return path;
      // Clamp to range to avoid drawing off-chart.
      const clamped = Math.min(preset.thetaRange[1], Math.max(preset.thetaRange[0], next));
      return path.concat(clamped);
    });
  }, [data, preset, useFisher]);

  const runToConvergence = useCallback(() => {
    setIterates((path) => {
      let theta = path[path.length - 1];
      const out = [...path];
      for (let k = 0; k < 25; k++) {
        const result = useFisher
          ? fisherScoringStep(theta, data, preset.score, preset.fisherInfo)
          : newtonRaphsonStep(theta, data, preset.score, preset.hessian);
        const next = result.nextTheta;
        if (!Number.isFinite(next)) break;
        const clamped = Math.min(preset.thetaRange[1], Math.max(preset.thetaRange[0], next));
        out.push(clamped);
        if (Math.abs(clamped - theta) < 1e-10) break;
        theta = clamped;
      }
      return out;
    });
  }, [data, preset, useFisher]);

  const reset = useCallback(() => {
    setIterates([init]);
  }, [init]);

  // Log-likelihood curve.
  const curve = useMemo(() => {
    const [lo, hi] = preset.thetaRange;
    const grid: number[] = [];
    const G = 181;
    for (let i = 0; i < G; i++) grid.push(lo + ((hi - lo) * i) / (G - 1));
    const { logLiks } = logLikelihoodCurve(data, preset.logPdf, grid);
    return { thetas: grid, logLiks };
  }, [data, preset]);

  // Current iteration diagnostics.
  const diagnostics = useMemo(() => {
    const theta = iterates[iterates.length - 1];
    const result = useFisher
      ? fisherScoringStep(theta, data, preset.score, preset.fisherInfo)
      : newtonRaphsonStep(theta, data, preset.score, preset.hessian);
    const info = useFisher ? result.expectedInfo : result.observedInfo;
    const logLik = logLikelihood(data, preset.logPdf, theta);
    return { theta, score: result.score, info, stepSize: result.stepSize, logLik };
  }, [iterates, data, preset, useFisher]);

  // ── Left panel: ℓ(θ) with tangent at current iterate ────────────────────
  useEffect(() => {
    if (!leftSvgRef.current) return;
    const svg = d3.select(leftSvgRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.58) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain(preset.thetaRange).range([0, innerW]);
    const yVals = curve.logLiks.concat([diagnostics.logLik]);
    const yTop = (d3.max(yVals) ?? 0) + 1;
    const yBot = yTop - 15;
    const y = d3.scaleLinear().domain([yBot, yTop]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`θ (${preset.id === 'gamma' ? 'α' : 'λ'})`);

    // Log-likelihood curve
    const points = curve.thetas.map((t, i) => ({ t, l: curve.logLiks[i] }));
    g.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line<{ t: number; l: number }>()
          .x((d) => x(d.t))
          .y((d) => y(Math.max(d.l, yBot))),
      );

    // True MLE θ̂ reference
    g.append('line')
      .attr('x1', x(mleRef.mle))
      .attr('x2', x(mleRef.mle))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2 2');

    // Iteration path: dots at each iterate, connecting segments.
    iterates.forEach((theta, i) => {
      const ell = logLikelihood(data, preset.logPdf, theta);
      g.append('circle')
        .attr('cx', x(theta))
        .attr('cy', y(Math.max(ell, yBot)))
        .attr('r', i === iterates.length - 1 ? 5 : 3)
        .attr('fill', i === iterates.length - 1 ? '#dc2626' : '#f59e0b')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1);
      g.append('text')
        .attr('x', x(theta))
        .attr('y', y(Math.max(ell, yBot)) - 8)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('fill', '#dc2626')
        .text(`θ${toSuper(i)}`);
    });

    // Tangent line at current iterate: slope = score, intercept so passes through (θ_t, ℓ(θ_t))
    const tEnd = iterates[iterates.length - 1];
    const ellAtT = diagnostics.logLik;
    const slope = diagnostics.score;
    // Draw across the zoom window [θ_t − 2 step | θ_t + 2 step]
    const halfSpan =
      diagnostics.info > 0 ? 2 * Math.abs(diagnostics.stepSize) + 0.3 : 0.5;
    const x1 = Math.max(preset.thetaRange[0], tEnd - halfSpan);
    const x2 = Math.min(preset.thetaRange[1], tEnd + halfSpan);
    const y1 = ellAtT + slope * (x1 - tEnd);
    const y2 = ellAtT + slope * (x2 - tEnd);
    g.append('line')
      .attr('x1', x(x1))
      .attr('x2', x(x2))
      .attr('y1', y(Math.max(Math.min(y1, yTop), yBot)))
      .attr('y2', y(Math.max(Math.min(y2, yTop), yBot)))
      .attr('stroke', '#16a34a')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5 3');

    // Annotation: θ̂
    g.append('text')
      .attr('x', x(mleRef.mle))
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text('θ̂');
  }, [curve, iterates, diagnostics, mleRef.mle, preset, w, isWide, data]);

  // ── Right panel: convergence plot |θ_t − θ̂| vs iteration, log scale ─────
  useEffect(() => {
    if (!rightSvgRef.current) return;
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.42) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const errors = iterates.map((t) => Math.max(Math.abs(t - mleRef.mle), 1e-16));
    const maxIter = Math.max(iterates.length - 1, 1);
    const x = d3.scaleLinear().domain([0, maxIter]).range([0, innerW]);
    const yMin = Math.max(Math.min(...errors) / 10, 1e-16);
    const yMax = Math.max(...errors, 1);
    const y = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(Math.min(maxIter, 8)).tickFormat(d3.format('d')))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(y).ticks(4, '.0e'))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('iteration t');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -44)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('|θ⁽ᵗ⁾ − θ̂|');

    // Curve
    g.append('path')
      .datum(errors.map((e, i) => ({ i, e })))
      .attr('fill', 'none')
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line<{ i: number; e: number }>()
          .x((d) => x(d.i))
          .y((d) => y(d.e)),
      );
    g.selectAll('circle.err')
      .data(errors.map((e, i) => ({ i, e })))
      .enter()
      .append('circle')
      .attr('class', 'err')
      .attr('cx', (d) => x(d.i))
      .attr('cy', (d) => y(d.e))
      .attr('r', 3)
      .attr('fill', '#dc2626');
  }, [iterates, mleRef.mle, w, isWide]);

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
  const fmtExp = (v: number) =>
    Number.isFinite(v) && v > 0 ? v.toExponential(2) : '—';

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: Newton-Raphson Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Step through Newton-Raphson iterates on the log-likelihood. The green dashed line is the
          tangent at θ⁽ᵗ⁾; its slope is the score S(θ⁽ᵗ⁾), and the next iterate θ⁽ᵗ⁺¹⁾ is the
          point where this tangent crosses zero slope (for a concave log-likelihood). The right
          panel plots |θ⁽ᵗ⁾ − θ̂| on a log scale — quadratic convergence shows up as a roughly
          straight line of steep negative slope.
        </div>
      </div>

      <div
        className="flex flex-wrap gap-3 items-center"
        style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}
      >
        <label>
          Preset:{' '}
          <select
            value={presetIndex}
            onChange={(e) => {
              setPresetIndex(Number(e.target.value));
              setSeed((s) => s + 1);
            }}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {PRESETS.map((p, i) => (
              <option key={p.id} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          n = {n}{' '}
          <input
            type="range"
            min={20}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 110, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
        <label>
          init = {init.toFixed(2)}{' '}
          <input
            type="range"
            min={preset.thetaRange[0]}
            max={preset.thetaRange[1]}
            step={0.05}
            value={init}
            onChange={(e) => {
              const v = Number(e.target.value);
              setInit(v);
              setIterates([v]);
            }}
            style={{ width: 110, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={useFisher}
            onChange={(e) => setUseFisher(e.target.checked)}
            style={{ marginRight: 4 }}
          />
          Fisher scoring
        </label>
        <button
          onClick={step}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          Step
        </button>
        <button
          onClick={runToConvergence}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          Run
        </button>
        <button
          onClick={reset}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          Reset
        </button>
        <button
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          ↻ New sample
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isWide ? '58fr 42fr' : '1fr',
          gap: '0.5rem',
        }}
      >
        <svg
          ref={leftSvgRef}
          width={isWide ? Math.floor(w * 0.58) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={rightSvgRef}
          width={isWide ? Math.floor(w * 0.42) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
      </div>

      <div
        className="grid"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '0.25rem 0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.8125rem',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <div>iteration t {iterates.length - 1}</div>
        <div>θ⁽ᵗ⁾ {fmt(diagnostics.theta, 4)}</div>
        <div>θ̂ {fmt(mleRef.mle, 4)}</div>
        <div>S(θ⁽ᵗ⁾) {fmt(diagnostics.score, 3)}</div>
        <div>
          {useFisher ? 'nI(θ⁽ᵗ⁾)' : 'J(θ⁽ᵗ⁾)'} {fmt(diagnostics.info, 3)}
        </div>
        <div>|θ⁽ᵗ⁾ − θ̂| {fmtExp(Math.abs(diagnostics.theta - mleRef.mle))}</div>
      </div>
    </div>
  );
}

/** Render a non-negative integer as Unicode superscript digits. */
function toSuper(k: number): string {
  const digits = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  if (k === 0) return digits[0];
  let s = '';
  let x = k;
  while (x > 0) {
    s = digits[x % 10] + s;
    x = Math.floor(x / 10);
  }
  return s;
}
