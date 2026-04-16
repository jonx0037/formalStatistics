import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { computeMLE, fisherInformation, logLikelihoodCurve, logGamma } from './shared/estimation';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  poissonSample,
  sampleSequence,
} from './shared/convergence';
import {
  mleDistributionPresets,
  type MLEDistributionPreset,
} from '../../data/maximum-likelihood-data';

const MARGIN = { top: 18, right: 16, bottom: 40, left: 48 };
const H = 280;
const N_STEPS = [5, 10, 25, 50, 100, 200, 500] as const;

type PlottablePreset = MLEDistributionPreset;

/** Per-family single-observation log-pdf used for plotting ℓ(θ). */
function logPdfFor(preset: PlottablePreset): (x: number, theta: number) => number {
  switch (preset.family) {
    case 'Normal': {
      const sigma2 = preset.otherParams?.sigma2 ?? 1;
      return (x, mu) => {
        const z = x - mu;
        return -0.5 * Math.log(2 * Math.PI * sigma2) - (z * z) / (2 * sigma2);
      };
    }
    case 'Bernoulli':
      return (x, p) => {
        if (p <= 0 || p >= 1) return -Infinity;
        return x * Math.log(p) + (1 - x) * Math.log(1 - p);
      };
    case 'Exponential':
      return (x, lam) => {
        if (lam <= 0) return -Infinity;
        return Math.log(lam) - lam * x;
      };
    case 'Poisson':
      return (x, lam) => {
        if (lam <= 0) return -Infinity;
        return x * Math.log(lam) - lam - logGamma(x + 1);
      };
    default:
      return () => 0; // Gamma omitted — LikelihoodSurfaceExplorer uses 4 closed-form families
  }
}

/** Sampler closure (n) → number[] for a preset's true parameters. */
function makeSampler(preset: PlottablePreset, rng: () => number): (n: number) => number[] {
  switch (preset.family) {
    case 'Normal': {
      const sigma = Math.sqrt(preset.otherParams?.sigma2 ?? 1);
      return (n) => sampleSequence(() => normalSample(preset.trueParam, sigma, rng), n);
    }
    case 'Bernoulli':
      return (n) => sampleSequence(() => bernoulliSample(preset.trueParam, rng), n);
    case 'Exponential':
      return (n) => sampleSequence(() => exponentialSample(preset.trueParam, rng), n);
    case 'Poisson':
      return (n) => sampleSequence(() => poissonSample(preset.trueParam, rng), n);
    default:
      return () => [];
  }
}

/** Deterministic LCG for reproducible regeneration. */
function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export default function LikelihoodSurfaceExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  // Filter to the 4 closed-form families the brief calls for.
  const presets = useMemo(
    () => mleDistributionPresets.filter((p) => p.family !== 'Gamma'),
    [],
  );

  const [presetIndex, setPresetIndex] = useState(0);
  const [n, setN] = useState(50);
  const [showQuadratic, setShowQuadratic] = useState(true);
  const [seed, setSeed] = useState(1);
  const [animating, setAnimating] = useState(false);
  const animRef = useRef<number | null>(null);
  const leftSvgRef = useRef<SVGSVGElement | null>(null);
  const rightSvgRef = useRef<SVGSVGElement | null>(null);

  const preset = presets[presetIndex];
  const logPdf = useMemo(() => logPdfFor(preset), [preset]);

  // Stable seed: does NOT include n, so a sample of size n is a prefix of
  // size n+1. This makes the likelihood "sharpen" smoothly as n grows
  // rather than jumping randomly with each slider tick.
  const data = useMemo(() => {
    const rng = makeLCG(seed * 7919);
    const sampler = makeSampler(preset, rng);
    return sampler(n);
  }, [preset, seed, n]);

  const mleResult = useMemo(
    () => computeMLE(data, preset.family, preset.paramName, preset.otherParams),
    [data, preset],
  );

  // Fisher info at θ̂ (per-observation) for Wald CI + quadratic overlay.
  const fisherFn = useMemo(
    () => fisherInformation(preset.family, preset.paramName),
    [preset],
  );
  const fisherAtMLE = useMemo(
    () => fisherFn(mleResult.mle, preset.otherParams),
    [fisherFn, mleResult.mle, preset.otherParams],
  );
  const waldSE = Math.sqrt(1 / (n * Math.max(fisherAtMLE, 1e-12)));
  const wald95 = [mleResult.mle - 1.96 * waldSE, mleResult.mle + 1.96 * waldSE];

  // Grid of θ values for the log-likelihood curve.
  const curve = useMemo(() => {
    const [lo, hi] = preset.thetaRange;
    const grid: number[] = [];
    const G = 201;
    for (let i = 0; i < G; i++) grid.push(lo + ((hi - lo) * i) / (G - 1));
    const { logLiks } = logLikelihoodCurve(data, logPdf, grid);
    return { thetas: grid, logLiks };
  }, [data, logPdf, preset.thetaRange]);

  // Quadratic approximation around θ̂: ℓ(θ) ≈ ℓ(θ̂) − (n I(θ̂) / 2)(θ − θ̂)².
  // When the MLE log-lik is non-finite (e.g., degenerate Normal σ² with a
  // constant sample), anchor the parabola on the finite maximum of the
  // plotted curve so the overlay doesn't collapse to the plot floor.
  const quadraticCurve = useMemo(() => {
    const nInfo = n * fisherAtMLE;
    const finiteLogLiks = curve.logLiks.filter((v) => Number.isFinite(v));
    const fallback = finiteLogLiks.length > 0 ? Math.max(...finiteLogLiks) : 0;
    const anchor = Number.isFinite(mleResult.logLik) ? mleResult.logLik : fallback;
    return curve.thetas.map((t) => anchor - 0.5 * nInfo * (t - mleResult.mle) ** 2);
  }, [curve.thetas, curve.logLiks, mleResult, n, fisherAtMLE]);

  // Animation: cycle n through N_STEPS.
  useEffect(() => {
    if (!animating) return;
    let i = N_STEPS.indexOf(n as (typeof N_STEPS)[number]);
    if (i < 0) i = 0;
    const tick = () => {
      i = (i + 1) % N_STEPS.length;
      setN(N_STEPS[i]);
      animRef.current = window.setTimeout(tick, 1100);
    };
    animRef.current = window.setTimeout(tick, 1100);
    return () => {
      if (animRef.current !== null) clearTimeout(animRef.current);
    };
  }, [animating]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Left panel: full ℓ(θ) curve with true θ₀ and MLE θ̂ ────────────────────
  useEffect(() => {
    if (!leftSvgRef.current) return;
    const svg = d3.select(leftSvgRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.6) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3
      .scaleLinear()
      .domain(preset.thetaRange)
      .range([0, innerW])
      .nice();

    // Y-range: show the top portion of ℓ (from ℓ̂ down ~10 units). Fall
    // back to the finite maximum of the plotted curve when the MLE log-lik
    // is not representable as a finite number — the clearest case is a
    // degenerate Normal σ² MLE on a constant sample where σ̂² = 0 and the
    // log-likelihood diverges. (Bernoulli boundary MLEs p̂∈{0,1} are
    // finite under our 0·log 0 = 0 convention, so they hit the fast path.)
    const finiteCurveLogLiks = curve.logLiks.filter((v) => Number.isFinite(v));
    const curveMax = finiteCurveLogLiks.length > 0 ? Math.max(...finiteCurveLogLiks) : 0;
    const anchor = Number.isFinite(mleResult.logLik) ? mleResult.logLik : curveMax;
    const yMax = anchor + 2;
    const yMin = anchor - 10;
    const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`θ (${preset.paramName})`);
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('ℓ(θ)');

    // Log-likelihood curve (clamped to visible range)
    const line = d3
      .line<{ t: number; l: number }>()
      .x((d) => x(d.t))
      .y((d) => y(Math.max(d.l, yMin)));
    const points = curve.thetas.map((t, i) => ({ t, l: curve.logLiks[i] }));

    g.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Quadratic overlay
    if (showQuadratic) {
      const quadPoints = curve.thetas.map((t, i) => ({ t, l: quadraticCurve[i] }));
      g.append('path')
        .datum(quadPoints)
        .attr('fill', 'none')
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 3')
        .attr('d', line);
    }

    // True θ₀ (solid black)
    g.append('line')
      .attr('x1', x(preset.trueParam))
      .attr('x2', x(preset.trueParam))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5);

    // MLE θ̂ (dashed red)
    g.append('line')
      .attr('x1', x(mleResult.mle))
      .attr('x2', x(mleResult.mle))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3');

    g.append('text')
      .attr('x', x(preset.trueParam))
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text('θ₀');
    g.append('text')
      .attr('x', x(mleResult.mle))
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#dc2626')
      .text('θ̂');
  }, [curve, quadraticCurve, showQuadratic, mleResult, preset, w, isWide]);

  // ── Right panel: zoomed view around θ̂ with Wald-CI bar ────────────────────
  useEffect(() => {
    if (!rightSvgRef.current) return;
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.4) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Zoom window: ±4 standard errors around θ̂, clamped to the preset's
    // valid parameter range so the zoom never crosses outside (for
    // Bernoulli p this means [0.02, 0.98] — otherwise a boundary MLE
    // p̂ ∈ {0, 1} would push xLo < 0 / xHi > 1 where logPdf returns −∞
    // and the zoom curve shows an artificial plunge at θ̂).
    const zoomHalf = Math.max(4 * waldSE, (preset.thetaRange[1] - preset.thetaRange[0]) / 50);
    const xLo = Math.max(preset.thetaRange[0], mleResult.mle - zoomHalf);
    const xHi = Math.min(preset.thetaRange[1], mleResult.mle + zoomHalf);

    const x = d3.scaleLinear().domain([xLo, xHi]).range([0, innerW]);

    // Fine grid for the zoom
    const G = 101;
    const zoomGrid: number[] = [];
    for (let i = 0; i < G; i++) zoomGrid.push(xLo + ((xHi - xLo) * i) / (G - 1));
    const zoomLogLik = zoomGrid.map((t) => {
      let s = 0;
      for (let i = 0; i < data.length; i++) s += logPdf(data[i], t);
      return s;
    });

    // Anchor y on the finite log-lik at the MLE. When the MLE log-lik is
    // not finite (e.g., degenerate Normal σ² MLE), fall back to the max
    // finite value observed on the zoom curve itself — that keeps the
    // local shape visible rather than clamping it to a fixed floor of 0.
    const finiteZoomLogLiks = zoomLogLik.filter((v) => Number.isFinite(v));
    const zoomMax = finiteZoomLogLiks.length > 0 ? Math.max(...finiteZoomLogLiks) : 0;
    const anchorR = Number.isFinite(mleResult.logLik) ? mleResult.logLik : zoomMax;
    const yMin = anchorR - 4;
    const yMax = anchorR + 1;
    const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('zoom around θ̂');

    const zoomQuad = zoomGrid.map(
      (t) => anchorR - 0.5 * n * fisherAtMLE * (t - mleResult.mle) ** 2,
    );

    const line = d3
      .line<{ t: number; l: number }>()
      .x((d) => x(d.t))
      .y((d) => y(Math.max(d.l, yMin)));

    g.append('path')
      .datum(zoomGrid.map((t, i) => ({ t, l: zoomLogLik[i] })))
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr('d', line);

    if (showQuadratic) {
      g.append('path')
        .datum(zoomGrid.map((t, i) => ({ t, l: zoomQuad[i] })))
        .attr('fill', 'none')
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 3')
        .attr('d', line);
    }

    // Wald 95% CI as a horizontal bar at y = ℓ(θ̂) − 1/2 (log-likelihood "width" convention).
    const barY = mleResult.logLik - 0.5;
    if (barY >= yMin && barY <= yMax) {
      g.append('line')
        .attr('x1', x(Math.max(wald95[0], xLo)))
        .attr('x2', x(Math.min(wald95[1], xHi)))
        .attr('y1', y(barY))
        .attr('y2', y(barY))
        .attr('stroke', '#16a34a')
        .attr('stroke-width', 3);
      g.append('text')
        .attr('x', x(mleResult.mle))
        .attr('y', y(barY) - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', '#16a34a')
        .text('95% Wald CI');
    }

    // θ̂ marker
    g.append('line')
      .attr('x1', x(mleResult.mle))
      .attr('x2', x(mleResult.mle))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3');
  }, [data, mleResult, waldSE, fisherAtMLE, showQuadratic, logPdf, n, preset, w, isWide]);

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: Likelihood Surface Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Watch the log-likelihood <em>sharpen</em> around the MLE as <em>n</em> grows. The amber
          dashed curve is the quadratic approximation at θ̂, whose curvature is the observed Fisher
          information; the green bar on the right is the 95% Wald confidence interval.
        </div>
      </div>

      <div
        className="flex flex-wrap gap-3 items-center"
        style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}
      >
        <label>
          Family:{' '}
          <select
            value={presetIndex}
            onChange={(e) => {
              setPresetIndex(Number(e.target.value));
              setSeed((s) => s + 1);
            }}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {presets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          n = {n}{' '}
          <input
            type="range"
            min={5}
            max={500}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 140, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
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
        <label>
          <input
            type="checkbox"
            checked={showQuadratic}
            onChange={(e) => setShowQuadratic(e.target.checked)}
            style={{ marginRight: 4 }}
          />
          Quadratic overlay
        </label>
        <button
          onClick={() => setAnimating((a) => !a)}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: animating ? '#2563eb' : 'transparent',
            color: animating ? 'white' : 'inherit',
            cursor: 'pointer',
          }}
        >
          {animating ? '⏸ Pause' : '▶ Animate n'}
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isWide ? '3fr 2fr' : '1fr',
          gap: '0.5rem',
        }}
      >
        <svg
          ref={leftSvgRef}
          width={isWide ? Math.floor(w * 0.6) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={rightSvgRef}
          width={isWide ? Math.floor(w * 0.4) : w}
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
        <div>
          θ̂ <strong>{fmt(mleResult.mle)}</strong>
        </div>
        <div>θ₀ {fmt(preset.trueParam)}</div>
        <div>ℓ(θ̂) {fmt(mleResult.logLik, 2)}</div>
        <div>I(θ̂) {fmt(fisherAtMLE, 4)}</div>
        <div>SE {fmt(waldSE, 4)}</div>
        <div>
          95% CI [{fmt(wald95[0])}, {fmt(wald95[1])}]
        </div>
      </div>
    </div>
  );
}
