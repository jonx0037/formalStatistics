import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  sampleMean,
  sampleMedian,
  trimmedMean,
  sampleVariance,
  computeBias,
  computeMSE,
  computeSE,
} from './shared/estimation';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  uniformSample,
  sampleSequence,
} from './shared/convergence';
import {
  distributionPresets,
  estimatorPresets,
  type DistributionPreset,
  type EstimatorPreset,
} from '../../data/point-estimation-data';

type EstimatorId = EstimatorPreset['id'];

const MARGIN = { top: 12, right: 20, bottom: 40, left: 52 };
const H = 280;

/** Return a sampler closure (n: number) => number[] for a given preset. */
function makeSampler(preset: DistributionPreset): (n: number) => number[] {
  const p = preset.params;
  switch (preset.family) {
    case 'Normal': {
      const sigma = Math.sqrt(p.sigma2);
      return (n) => sampleSequence(() => normalSample(p.mu, sigma), n);
    }
    case 'Exponential':
      return (n) => sampleSequence(() => exponentialSample(p.lambda), n);
    case 'Bernoulli':
      return (n) => sampleSequence(() => bernoulliSample(p.p), n);
    case 'Uniform':
      return (n) => sampleSequence(() => uniformSample(p.a, p.b), n);
    default:
      return () => [];
  }
}

/** Evaluate an estimator on a sample. */
function applyEstimator(id: EstimatorId, sample: number[]): number {
  switch (id) {
    case 'mean':
      return sampleMean(sample);
    case 'median':
      return sampleMedian(sample);
    case 'trimmed':
      return trimmedMean(sample, 0.1);
    case 'var_biased':
      return sampleVariance(sample, 0);
    case 'var_unbiased':
      return sampleVariance(sample, 1);
  }
}

/** Population median for each family. */
function populationMedian(preset: DistributionPreset): number {
  const p = preset.params;
  switch (preset.family) {
    case 'Normal':
      return p.mu;
    case 'Exponential':
      return Math.log(2) / p.lambda;
    case 'Bernoulli':
      return p.p >= 0.5 ? 1 : 0;
    case 'Uniform':
      return (p.a + p.b) / 2;
  }
  return 0;
}

/** Target (true) parameter the estimator is trying to hit. */
function trueTarget(estimatorId: EstimatorId, preset: DistributionPreset): number {
  if (estimatorId === 'median') return populationMedian(preset);
  if (estimatorId === 'var_biased' || estimatorId === 'var_unbiased') {
    return preset.trueParam.variance;
  }
  return preset.trueParam.mean;
}

/** Analytic theoretical bias & variance where known; NaN if unknown. */
function theoretical(
  estimatorId: EstimatorId,
  preset: DistributionPreset,
  n: number,
): { bias: number; variance: number; mse: number } {
  const sigma2 = preset.trueParam.variance;
  switch (estimatorId) {
    case 'mean': {
      const variance = sigma2 / n;
      return { bias: 0, variance, mse: variance };
    }
    case 'var_biased': {
      // Bias: E[S²_n] = (n-1)/n · σ², so Bias = -σ²/n.
      // Exact variance is distribution-specific; for Normal use Var(S²_n) = 2(n-1)/n² · σ⁴.
      const bias = -sigma2 / n;
      const variance =
        preset.family === 'Normal' ? (2 * (n - 1) * sigma2 * sigma2) / (n * n) : NaN;
      const mse = isFinite(variance) ? bias * bias + variance : NaN;
      return { bias, variance, mse };
    }
    case 'var_unbiased': {
      const variance = preset.family === 'Normal' ? (2 * sigma2 * sigma2) / (n - 1) : NaN;
      const mse = isFinite(variance) ? variance : NaN;
      return { bias: 0, variance, mse };
    }
    case 'median': {
      // Asymptotic variance 1/(4n f(median)²). Density at median differs per family.
      let fAtMedian = NaN;
      if (preset.family === 'Normal') {
        fAtMedian = 1 / Math.sqrt(2 * Math.PI * sigma2);
      } else if (preset.family === 'Exponential') {
        fAtMedian = preset.params.lambda * Math.exp(-Math.log(2));
      } else if (preset.family === 'Uniform') {
        fAtMedian = 1 / (preset.params.b - preset.params.a);
      }
      const asymptVar = 1 / (4 * n * fAtMedian * fAtMedian);
      // The sample median is finite-sample unbiased only for symmetric
      // populations (Normal, Uniform). On skewed families (Exponential,
      // Bernoulli) E[median] ≠ population median at finite n, so report NaN
      // rather than overstating the guarantee. The MC estimate still reveals
      // the empirical bias directly.
      const symmetric = preset.family === 'Normal' || preset.family === 'Uniform';
      const bias = symmetric ? 0 : NaN;
      const mse = isFinite(bias) && isFinite(asymptVar) ? bias * bias + asymptVar : NaN;
      return { bias, variance: asymptVar, mse };
    }
    case 'trimmed': {
      // Rough approximation: trimmed mean efficiency ≈ 0.95 of the mean for Normal.
      // As with the median, the trimmed mean is only unbiased for μ under
      // symmetry; on skewed families the trimmed expectation differs from E[X].
      const approx = preset.family === 'Normal' ? (sigma2 / n) * 1.05 : NaN;
      const symmetric = preset.family === 'Normal' || preset.family === 'Uniform';
      const bias = symmetric ? 0 : NaN;
      const mse = isFinite(bias) && isFinite(approx) ? bias * bias + approx : NaN;
      return { bias, variance: approx, mse };
    }
  }
}

export default function EstimatorSamplingExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [distIndex, setDistIndex] = useState(0);
  const [estimatorId, setEstimatorId] = useState<EstimatorId>('mean');
  const [n, setN] = useState(25);
  const [estimates, setEstimates] = useState<number[]>([]);
  const [pendingDraws, setPendingDraws] = useState(0);
  const rafRef = useRef<number | null>(null);

  const svgRef = useRef<SVGSVGElement | null>(null);

  const preset = distributionPresets[distIndex];
  const estPreset = estimatorPresets.find((e) => e.id === estimatorId)!;
  const sampler = useMemo(() => makeSampler(preset), [preset]);
  const target = useMemo(() => trueTarget(estimatorId, preset), [estimatorId, preset]);
  const theo = useMemo(() => theoretical(estimatorId, preset, n), [estimatorId, preset, n]);

  // Reset accumulated estimates whenever the experimental setup changes.
  useEffect(() => {
    setEstimates([]);
    setPendingDraws(0);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [distIndex, estimatorId, n]);

  // rAF batching loop: 50 MC draws per frame until pendingDraws reaches zero.
  useEffect(() => {
    if (pendingDraws <= 0) return;
    rafRef.current = requestAnimationFrame(() => {
      const batch = Math.min(50, pendingDraws);
      const newEst: number[] = [];
      for (let i = 0; i < batch; i++) {
        newEst.push(applyEstimator(estimatorId, sampler(n)));
      }
      setEstimates((prev) => prev.concat(newEst));
      setPendingDraws((d) => d - batch);
    });
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [pendingDraws, estimatorId, sampler, n]);

  const summary = useMemo(() => {
    if (estimates.length === 0) {
      return { mean: NaN, bias: NaN, variance: NaN, mse: NaN, se: NaN };
    }
    const m = sampleMean(estimates);
    const bias = computeBias(estimates, target);
    const variance = sampleVariance(estimates, 1);
    const mse = computeMSE(estimates, target);
    const se = computeSE(estimates);
    return { mean: m, bias, variance, mse, se };
  }, [estimates, target]);

  // ── D3 histogram with theoretical Normal overlay ────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = w - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // X-domain: center on the target with a generous window scaled to theoretical SD
    const theoSd = isFinite(theo.variance) ? Math.sqrt(Math.max(theo.variance, 1e-10)) : NaN;
    const pad = isFinite(theoSd) ? 4 * theoSd : 1;
    const baseCenter = estimates.length > 0 ? summary.mean : target;
    const xMin = Math.min(target, baseCenter) - pad;
    const xMax = Math.max(target, baseCenter) + pad;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]).nice();

    // Bins
    const bins = d3
      .bin<number, number>()
      .domain(x.domain() as [number, number])
      .thresholds(32)(estimates);

    const yMax =
      bins.length > 0
        ? d3.max(bins, (b) => b.length) ?? 1
        : 1;
    const y = d3
      .scaleLinear()
      .domain([0, Math.max(yMax, 1)])
      .range([innerH, 0])
      .nice();

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`${estPreset.name} on ${preset.name}`);

    // Histogram bars
    g.selectAll('rect.bin')
      .data(bins)
      .enter()
      .append('rect')
      .attr('class', 'bin')
      .attr('x', (d) => x(d.x0 ?? 0) + 1)
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
      .attr('height', (d) => innerH - y(d.length))
      .attr('fill', '#60a5fa')
      .attr('opacity', 0.65);

    // Theoretical Normal overlay scaled to histogram area
    if (isFinite(theo.variance) && theo.variance > 0 && estimates.length > 2) {
      const binWidth = bins.length > 0 ? (bins[0].x1 ?? 0) - (bins[0].x0 ?? 0) : 0;
      const N = estimates.length;
      const mu = target + theo.bias;
      const sd = Math.sqrt(theo.variance);
      const pts = d3.range(80).map((i) => {
        const xi = xMin + ((xMax - xMin) * i) / 79;
        const pdf =
          (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-((xi - mu) ** 2) / (2 * sd * sd));
        return { x: xi, y: pdf * N * binWidth };
      });
      g.append('path')
        .datum(pts)
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
    }

    // Vertical lines: true target (black dashed) and MC mean (red solid)
    g.append('line')
      .attr('x1', x(target))
      .attr('x2', x(target))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3');

    if (isFinite(summary.mean)) {
      g.append('line')
        .attr('x1', x(summary.mean))
        .attr('x2', x(summary.mean))
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
      .text(`true θ = ${target.toFixed(3)} (dashed)`);
    if (isFinite(summary.mean)) {
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', 26)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', '#dc2626')
        .text(`MC mean = ${summary.mean.toFixed(3)}`);
    }
    if (isFinite(theo.variance) && theo.variance > 0) {
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', 40)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', '#dc2626')
        .text('theoretical Normal overlay');
    }
  }, [estimates, w, preset, estPreset, target, theo, summary.mean]);

  const fmt = (v: number) => (isFinite(v) ? v.toFixed(4) : '—');

  return (
    <div
      ref={containerRef}
      style={{
        fontFamily: 'system-ui, sans-serif',
        border: '1px solid var(--color-border)',
        borderRadius: '0.75rem',
        padding: '1.25rem',
        margin: '1.5rem 0',
      }}
    >
      <div style={{ marginBottom: '0.75rem' }}>
        <strong>Interactive: Estimator Sampling Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Draw samples of size <em>n</em>, compute the chosen estimator on each, and watch the
          sampling distribution build up. Histogram in blue, theoretical Normal overlay in red
          (when analytic variance is available).
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 640 ? '1fr 1fr' : '1fr',
          gap: '0.75rem',
          marginBottom: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <label>
          Distribution:{' '}
          <select
            value={distIndex}
            onChange={(e) => setDistIndex(Number(e.target.value))}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {distributionPresets.map((d, i) => (
              <option key={d.name} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Estimator:{' '}
          <select
            value={estimatorId}
            onChange={(e) => setEstimatorId(e.target.value as EstimatorId)}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {estimatorPresets.map((est) => (
              <option key={est.id} value={est.id}>
                {est.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr auto',
          alignItems: 'center',
          gap: '0.5rem',
          marginBottom: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <span>n = {n}</span>
        <input
          type="range"
          min={5}
          max={500}
          step={1}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
        <span style={{ color: 'var(--color-text-muted)' }}>M = {estimates.length} draws</span>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
        <button
          onClick={() => setPendingDraws((d) => d + 1)}
          style={{ padding: '0.35rem 0.75rem' }}
        >
          Draw 1
        </button>
        <button
          onClick={() => setPendingDraws((d) => d + 100)}
          style={{ padding: '0.35rem 0.75rem' }}
        >
          Run 100
        </button>
        <button
          onClick={() => setPendingDraws((d) => d + 1000)}
          style={{ padding: '0.35rem 0.75rem' }}
        >
          Run 1000
        </button>
        <button
          onClick={() => {
            setEstimates([]);
            setPendingDraws(0);
          }}
          style={{
            padding: '0.35rem 0.75rem',
            marginLeft: 'auto',
            color: 'var(--color-text-muted)',
          }}
        >
          Reset
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 900 ? '2fr 1fr' : '1fr',
          gap: '1rem',
          alignItems: 'start',
        }}
      >
        <svg ref={svgRef} width={w} height={H} style={{ maxWidth: '100%' }} />

        <div
          style={{
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.5rem',
            padding: '0.75rem',
          }}
        >
          <div style={{ marginBottom: '0.25rem', fontWeight: 600 }}>Sampling distribution</div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                <th style={{ textAlign: 'left', padding: 2 }}></th>
                <th style={{ textAlign: 'right', padding: 2 }}>MC</th>
                <th style={{ textAlign: 'right', padding: 2 }}>Theory</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>E[θ̂]</td>
                <td style={{ textAlign: 'right' }}>{fmt(summary.mean)}</td>
                <td style={{ textAlign: 'right' }}>{fmt(target + theo.bias)}</td>
              </tr>
              <tr>
                <td>Bias</td>
                <td style={{ textAlign: 'right' }}>{fmt(summary.bias)}</td>
                <td style={{ textAlign: 'right' }}>{fmt(theo.bias)}</td>
              </tr>
              <tr>
                <td>Variance</td>
                <td style={{ textAlign: 'right' }}>{fmt(summary.variance)}</td>
                <td style={{ textAlign: 'right' }}>{fmt(theo.variance)}</td>
              </tr>
              <tr>
                <td>MSE</td>
                <td style={{ textAlign: 'right' }}>{fmt(summary.mse)}</td>
                <td style={{ textAlign: 'right' }}>{fmt(theo.mse)}</td>
              </tr>
              <tr>
                <td>SE</td>
                <td style={{ textAlign: 'right' }}>{fmt(summary.se)}</td>
                <td style={{ textAlign: 'right' }}>{fmt(Math.sqrt(theo.variance))}</td>
              </tr>
            </tbody>
          </table>
          <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
            True θ = {target.toFixed(4)} ({estPreset.isUnbiased ? 'unbiased' : 'biased'} estimator).
            Theory cells show "—" where no closed form applies.
          </div>
        </div>
      </div>
    </div>
  );
}
