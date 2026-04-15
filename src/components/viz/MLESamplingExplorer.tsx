import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  computeMLE,
  fisherInformation,
  sampleMean,
  sampleVariance,
  computeBias,
  computeMSE,
} from './shared/estimation';
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

const MARGIN = { top: 18, right: 16, bottom: 40, left: 52 };
const H = 300;

function makeSampler(preset: MLEDistributionPreset): (n: number) => number[] {
  switch (preset.family) {
    case 'Normal': {
      const sigma = Math.sqrt(preset.otherParams?.sigma2 ?? 1);
      return (n) => sampleSequence(() => normalSample(preset.trueParam, sigma), n);
    }
    case 'Bernoulli':
      return (n) => sampleSequence(() => bernoulliSample(preset.trueParam), n);
    case 'Exponential':
      return (n) => sampleSequence(() => exponentialSample(preset.trueParam), n);
    case 'Poisson':
      return (n) => sampleSequence(() => poissonSample(preset.trueParam), n);
    default:
      return () => [];
  }
}

export default function MLESamplingExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  // 4 closed-form families; Gamma omitted (slower + already covered by Newton explorer).
  const presets = useMemo(
    () => mleDistributionPresets.filter((p) => p.family !== 'Gamma'),
    [],
  );

  const [presetIndex, setPresetIndex] = useState(0);
  const [n, setN] = useState(50);
  const [estimates, setEstimates] = useState<number[]>([]);
  const [pendingDraws, setPendingDraws] = useState(0);
  const [standardized, setStandardized] = useState(false);
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const preset = presets[presetIndex];
  const sampler = useMemo(() => makeSampler(preset), [preset]);

  const fisherAtTrue = useMemo(() => {
    const fn = fisherInformation(preset.family, preset.paramName);
    return fn(preset.trueParam, preset.otherParams);
  }, [preset]);

  // Reset whenever the experimental setup changes.
  useEffect(() => {
    setEstimates([]);
    setPendingDraws(0);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [presetIndex, n]);

  // Batched MC draws via rAF.
  useEffect(() => {
    if (pendingDraws <= 0) return;
    rafRef.current = requestAnimationFrame(() => {
      const batch = Math.min(40, pendingDraws);
      const next: number[] = [];
      for (let i = 0; i < batch; i++) {
        const sample = sampler(n);
        const { mle } = computeMLE(sample, preset.family, preset.paramName, preset.otherParams);
        if (Number.isFinite(mle)) next.push(mle);
      }
      setEstimates((prev) => prev.concat(next));
      setPendingDraws((d) => d - batch);
    });
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [pendingDraws, preset, sampler, n]);

  const summary = useMemo(() => {
    if (estimates.length === 0) {
      return { mean: NaN, bias: NaN, variance: NaN, mse: NaN, efficiency: NaN };
    }
    const mean = sampleMean(estimates);
    const bias = computeBias(estimates, preset.trueParam);
    const variance = sampleVariance(estimates, 1);
    const mse = computeMSE(estimates, preset.trueParam);
    const crlb = 1 / (n * fisherAtTrue);
    return { mean, bias, variance, mse, efficiency: crlb / variance };
  }, [estimates, preset.trueParam, n, fisherAtTrue]);

  // ── Histogram + Normal overlay ───────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const innerW = w - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const sd = Math.sqrt(1 / (n * Math.max(fisherAtTrue, 1e-12)));
    const centered = standardized
      ? estimates.map((e) => Math.sqrt(n) * (e - preset.trueParam))
      : estimates;
    const center = standardized ? 0 : preset.trueParam;
    const spread = standardized ? Math.sqrt(1 / Math.max(fisherAtTrue, 1e-12)) : sd;
    const xMin = center - 4 * spread;
    const xMax = center + 4 * spread;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]).nice();
    const bins = d3
      .bin<number, number>()
      .domain(x.domain() as [number, number])
      .thresholds(32)(centered);

    const binWidth = bins.length > 0 ? (bins[0].x1 ?? 0) - (bins[0].x0 ?? 0) : 0;
    const N = centered.length;
    // Normalize histogram to density so the Normal overlay matches.
    const yDensityMax =
      N > 0 && binWidth > 0
        ? (d3.max(bins, (b) => b.length / (N * binWidth)) ?? 1)
        : 1;

    // Theoretical peak (1 / (σ √2π))
    const theoPeak = 1 / (spread * Math.sqrt(2 * Math.PI));
    const yMax = Math.max(yDensityMax, theoPeak) * 1.1 || 1;
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(standardized ? '√n(θ̂ − θ₀)' : 'θ̂');

    // Histogram (density scale)
    g.selectAll('rect.bin')
      .data(bins)
      .enter()
      .append('rect')
      .attr('class', 'bin')
      .attr('x', (d) => x(d.x0 ?? 0) + 1)
      .attr(
        'y',
        (d) => y(N > 0 && binWidth > 0 ? d.length / (N * binWidth) : 0),
      )
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
      .attr(
        'height',
        (d) =>
          innerH - y(N > 0 && binWidth > 0 ? d.length / (N * binWidth) : 0),
      )
      .attr('fill', '#60a5fa')
      .attr('opacity', 0.65);

    // Normal overlay with mean `center` and sd `spread`.
    const curve = d3.range(101).map((i) => {
      const xi = xMin + ((xMax - xMin) * i) / 100;
      const pdf =
        (1 / (spread * Math.sqrt(2 * Math.PI))) *
        Math.exp(-((xi - center) ** 2) / (2 * spread * spread));
      return { x: xi, y: pdf };
    });
    g.append('path')
      .datum(curve)
      .attr('fill', 'none')
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line<{ x: number; y: number }>()
          .x((d) => x(d.x))
          .y((d) => y(d.y)),
      );

    // Vertical lines: true parameter (or 0 in standardized mode)
    g.append('line')
      .attr('x1', x(center))
      .attr('x2', x(center))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3');

    if (N > 0) {
      const mcMean = standardized
        ? sampleMean(centered) // mean of standardized is close to 0
        : summary.mean;
      g.append('line')
        .attr('x1', x(mcMean))
        .attr('x2', x(mcMean))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 1.5);
    }

    // Legend
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 12)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text(
        standardized
          ? 'N(0, 1/I(θ₀)) overlay'
          : `N(θ₀, 1/(nI(θ₀))) overlay — σ = ${spread.toFixed(3)}`,
      );
  }, [estimates, w, preset, n, fisherAtTrue, standardized, summary.mean]);

  const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : '—');

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: MLE Sampling-Distribution Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Draw Monte-Carlo samples, compute the MLE on each, and watch the sampling distribution
          converge to the asymptotic Normal of Theorem 4. Toggle the <em>standardized</em> view to
          see <em>√n(θ̂ − θ₀)</em> collapse onto <em>N(0, 1/I(θ₀))</em>.
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
            onChange={(e) => setPresetIndex(Number(e.target.value))}
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
            style={{ width: 120, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={standardized}
            onChange={(e) => setStandardized(e.target.checked)}
            style={{ marginRight: 4 }}
          />
          Standardize √n(θ̂ − θ₀)
        </label>
        {[1, 100, 1000].map((k) => (
          <button
            key={k}
            onClick={() => setPendingDraws((d) => d + k)}
            style={{
              padding: '0.25rem 0.65rem',
              fontSize: '0.8125rem',
              border: '1px solid var(--color-border)',
              borderRadius: '0.375rem',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            +{k.toLocaleString()} draw{k === 1 ? '' : 's'}
          </button>
        ))}
        <button
          onClick={() => {
            setEstimates([]);
            setPendingDraws(0);
          }}
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
      </div>

      <svg ref={svgRef} width={w} height={H} style={{ overflow: 'visible' }} />

      <div
        className="grid"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '0.25rem 0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.8125rem',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <div>MC replicates {estimates.length.toLocaleString()}</div>
        <div>E[θ̂] {fmt(summary.mean)}</div>
        <div>Bias {fmt(summary.bias)}</div>
        <div>Var(θ̂) {fmt(summary.variance, 5)}</div>
        <div>CRLB = 1/(nI(θ₀)) {fmt(1 / (n * fisherAtTrue), 5)}</div>
        <div>Efficiency {fmt(summary.efficiency, 3)}</div>
      </div>
    </div>
  );
}
