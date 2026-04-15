import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { sampleMean, sampleVariance } from './shared/estimation';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  uniformSample,
  sampleSequence,
  runningMean,
  runningVariance,
} from './shared/convergence';
import {
  distributionPresets,
  consistencyPresets,
  type DistributionPreset,
  type ConsistencyPreset,
} from '../../data/point-estimation-data';

type EstimatorId = ConsistencyPreset['estimator'];
type EnvelopeMode = 'chebyshev' | 'clt';

const MARGIN = { top: 14, right: 20, bottom: 40, left: 52 };
const H = 300;
const MAX_RENDER_POINTS = 400;

const PATH_COLOR = '#2563eb';
const PATH_OPACITY = 0.18;

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

/** True target for each consistency-preset estimator on a given distribution. */
function trueTarget(id: EstimatorId, preset: DistributionPreset): number {
  if (id === 'mean') return preset.trueParam.mean;
  if (id === 'variance') return preset.trueParam.variance;
  if (id === 'first') return preset.trueParam.mean; // the mean the estimator "tries" to hit
  return 7; // constant preset
}

/** Trajectory generator per estimator (length n). */
function trajectory(id: EstimatorId, data: number[]): number[] {
  switch (id) {
    case 'mean':
      return runningMean(data);
    case 'variance':
      return runningVariance(data).map((v, i) => (i === 0 ? 0 : (v * (i + 1)) / i)); // Bessel-corrected
    case 'first':
      return new Array<number>(data.length).fill(data[0]);
    case 'constant':
      return new Array<number>(data.length).fill(7);
  }
}

/** Downsample a full-length trajectory to a lightweight array of {n, value} points. */
function downsample(traj: number[]): { n: number; v: number }[] {
  if (traj.length <= MAX_RENDER_POINTS) {
    return traj.map((v, i) => ({ n: i + 1, v }));
  }
  const idxs: number[] = [];
  // log-spaced indices so the x-axis on a log scale has even visual density
  const logStart = Math.log(1);
  const logEnd = Math.log(traj.length);
  for (let i = 0; i < MAX_RENDER_POINTS; i++) {
    const t = logStart + ((logEnd - logStart) * i) / (MAX_RENDER_POINTS - 1);
    idxs.push(Math.max(0, Math.min(traj.length - 1, Math.round(Math.exp(t) - 1))));
  }
  return idxs.map((i) => ({ n: i + 1, v: traj[i] }));
}

export default function ConsistencyExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [estId, setEstId] = useState<EstimatorId>('mean');
  const [distIndex, setDistIndex] = useState(0);
  const [n, setN] = useState(5000);
  const [numPaths, setNumPaths] = useState(12);
  const [envelopeMode, setEnvelopeMode] = useState<EnvelopeMode>('clt');
  const [seed, setSeed] = useState(0);

  const preset = distributionPresets[distIndex];
  const sampler = useMemo(() => makeSampler(preset), [preset]);
  const theta = useMemo(() => trueTarget(estId, preset), [estId, preset]);
  const popSigma2 = preset.trueParam.variance;

  const paths = useMemo(() => {
    const out: { n: number; v: number }[][] = [];
    const currentN = Math.max(10, Math.round(n));
    for (let p = 0; p < numPaths; p++) {
      const data = sampler(currentN);
      const traj = trajectory(estId, data);
      out.push(downsample(traj));
    }
    return out;
    // `seed` is included to force regeneration when the user hits Resample
  }, [sampler, n, numPaths, estId, seed]);

  // Final values at the last index, for the right-panel histogram
  const finalValues = useMemo(() => paths.map((p) => p[p.length - 1].v), [paths]);

  const leftRef = useRef<SVGSVGElement | null>(null);
  const rightRef = useRef<SVGSVGElement | null>(null);

  // Left panel: trajectories
  useEffect(() => {
    if (!leftRef.current) return;
    const svg = d3.select(leftRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.65 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLog().domain([1, Math.max(n, 10)]).range([0, innerW]);

    const allVals = paths.flatMap((p) => p.map((d) => d.v));
    let yMin = d3.min(allVals) ?? 0;
    let yMax = d3.max(allVals) ?? 1;
    // Always include θ and some padding
    yMin = Math.min(yMin, theta);
    yMax = Math.max(yMax, theta);
    const span = yMax - yMin || 1;
    yMin -= span * 0.1;
    yMax += span * 0.1;
    const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6, '.0s'))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('sample size n (log scale)');

    // Envelope: shaded band around θ. Chebyshev: σ/√n bound scaled by √(1/ε²). We use ε = SD/√n with a fixed factor.
    const gridXs = d3.range(50).map((i) => Math.pow(10, Math.log10(1) + (Math.log10(Math.max(n, 10)) * i) / 49));
    const envelope = gridXs.map((nn) => {
      if (envelopeMode === 'clt') {
        const width = 1.96 * Math.sqrt(popSigma2 / nn);
        return { n: nn, hi: theta + width, lo: theta - width };
      }
      // Chebyshev: P(|θ̂ - θ| ≥ ε) ≤ σ²/(nε²). At confidence 0.95, ε = σ/√(n·0.05).
      const width = Math.sqrt(popSigma2 / (nn * 0.05));
      return { n: nn, hi: theta + width, lo: theta - width };
    });

    const area = d3
      .area<{ n: number; hi: number; lo: number }>()
      .x((d) => x(d.n))
      .y0((d) => y(d.lo))
      .y1((d) => y(d.hi));
    g.append('path').datum(envelope).attr('fill', '#bfdbfe').attr('opacity', 0.45).attr('d', area);

    // Dashed horizontal at θ
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', y(theta))
      .attr('y2', y(theta))
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5 3');

    // Trajectory paths
    const lineGen = d3
      .line<{ n: number; v: number }>()
      .x((d) => x(d.n))
      .y((d) => y(d.v));

    paths.forEach((p) => {
      g.append('path')
        .datum(p)
        .attr('fill', 'none')
        .attr('stroke', PATH_COLOR)
        .attr('stroke-opacity', PATH_OPACITY)
        .attr('stroke-width', 1.2)
        .attr('d', lineGen);
    });

    // Label
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 12)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text(`θ = ${theta.toFixed(3)} (dashed)`);
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 26)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#1e40af')
      .text(envelopeMode === 'clt' ? '±1.96σ/√n envelope' : 'Chebyshev 95% envelope');
  }, [paths, w, theta, n, envelopeMode, popSigma2]);

  // Right panel: histogram of final values
  useEffect(() => {
    if (!rightRef.current) return;
    const svg = d3.select(rightRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.35 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const vExtent = d3.extent(finalValues) as [number, number];
    const span = Math.max((vExtent[1] ?? 0) - (vExtent[0] ?? 0), 0.1);
    const xMin = Math.min(vExtent[0], theta) - span * 0.2;
    const xMax = Math.max(vExtent[1], theta) + span * 0.2;
    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]).nice();

    const bins = d3
      .bin<number, number>()
      .domain(x.domain() as [number, number])
      .thresholds(Math.min(20, Math.max(5, Math.floor(numPaths / 2))))(finalValues);
    const yMax = d3.max(bins, (b) => b.length) ?? 1;
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(4))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

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
      .attr('opacity', 0.75);

    g.append('line')
      .attr('x1', x(theta))
      .attr('x2', x(theta))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-dasharray', '4 3');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`θ̂ at n = ${n}`);
  }, [finalValues, theta, w, n, numPaths]);

  const finalMean = finalValues.length > 0 ? sampleMean(finalValues) : NaN;
  const finalVar = finalValues.length > 1 ? sampleVariance(finalValues, 1) : NaN;
  const consistentText = consistencyPresets.find((p) => p.estimator === estId)?.consistent
    ? 'Consistent — trajectories collapse to θ'
    : 'Inconsistent — trajectories do not converge';

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
        <strong>Interactive: Consistency Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Simulate {numPaths} independent sample paths and watch the running estimator
          trajectories. Consistent estimators collapse to θ as n grows; inconsistent
          estimators do not.
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 720 ? 'repeat(3, 1fr) auto' : '1fr 1fr',
          gap: '0.5rem',
          fontSize: '0.8125rem',
          marginBottom: '0.5rem',
        }}
      >
        <label>
          Estimator:{' '}
          <select
            value={estId}
            onChange={(e) => setEstId(e.target.value as EstimatorId)}
            style={{ padding: '0.2rem' }}
          >
            {consistencyPresets.map((p) => (
              <option key={p.estimator} value={p.estimator}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Distribution:{' '}
          <select
            value={distIndex}
            onChange={(e) => setDistIndex(Number(e.target.value))}
            style={{ padding: '0.2rem' }}
          >
            {distributionPresets.map((d, i) => (
              <option key={d.name} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Envelope:{' '}
          <select
            value={envelopeMode}
            onChange={(e) => setEnvelopeMode(e.target.value as EnvelopeMode)}
            style={{ padding: '0.2rem' }}
          >
            <option value="clt">CLT (±1.96σ/√n)</option>
            <option value="chebyshev">Chebyshev (95%)</option>
          </select>
        </label>
        <button onClick={() => setSeed((s) => s + 1)} style={{ padding: '0.3rem 0.75rem' }}>
          Resample
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.8125rem',
          marginBottom: '0.5rem',
        }}
      >
        <span>n = {n}</span>
        <input
          type="range"
          min={100}
          max={50000}
          step={100}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
        <span>paths = {numPaths}</span>
        <input
          type="range"
          min={1}
          max={30}
          step={1}
          value={numPaths}
          onChange={(e) => setNumPaths(Number(e.target.value))}
        />
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 900 ? '2fr 1fr' : '1fr',
          gap: '0.75rem',
          alignItems: 'start',
        }}
      >
        <svg ref={leftRef} width={w > 900 ? (w - 24) * 0.65 : w} height={H} style={{ maxWidth: '100%' }} />
        <svg ref={rightRef} width={w > 900 ? (w - 24) * 0.35 : w} height={H} style={{ maxWidth: '100%' }} />
      </div>

      <div style={{ marginTop: '0.5rem', fontSize: '0.8125rem' }}>
        <span style={{ fontWeight: 600 }}>{consistentText}.</span>{' '}
        <span style={{ color: 'var(--color-text-muted)' }}>
          Sample mean across paths at n={n}: {isFinite(finalMean) ? finalMean.toFixed(4) : '—'}.
          Sample variance across paths: {isFinite(finalVar) ? finalVar.toFixed(6) : '—'}.
          True θ = {theta.toFixed(4)}.
        </span>
      </div>
    </div>
  );
}
