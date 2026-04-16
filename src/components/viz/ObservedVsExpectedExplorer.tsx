import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  computeMLE,
  fisherInformation,
  observedInformation,
  sampleMean,
  sampleVariance,
} from './shared/estimation';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  poissonSample,
  sampleSequence,
} from './shared/convergence';
import {
  observedVsExpectedPresets,
  type ObservedVsExpectedPreset,
} from '../../data/maximum-likelihood-data';

const REPLICATES = 200;
const MARGIN = { top: 20, right: 16, bottom: 40, left: 52 };
const H = 270;

function makeSampler(preset: ObservedVsExpectedPreset): (n: number) => number[] {
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

export default function ObservedVsExpectedExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const [n, setN] = useState(50);
  const [runId, setRunId] = useState(0); // bump to regenerate

  const scatterRef = useRef<SVGSVGElement | null>(null);
  const histRef = useRef<SVGSVGElement | null>(null);

  const preset = observedVsExpectedPresets[presetIndex];
  const sampler = useMemo(() => makeSampler(preset), [preset]);

  const fisherFn = useMemo(
    () => fisherInformation(preset.family, preset.paramName as 'mu' | 'p' | 'lambda'),
    [preset],
  );
  const observedFn = useMemo(
    () => observedInformation(preset.family, preset.paramName as 'mu' | 'p' | 'lambda'),
    [preset],
  );
  const nITrue = n * fisherFn(preset.trueParam, preset.otherParams);

  // Run MC replicates whenever preset / n / runId changes.
  const mc = useMemo(() => {
    const obs: number[] = [];
    const exp: number[] = [];
    const mles: number[] = [];
    for (let r = 0; r < REPLICATES; r++) {
      const data = sampler(n);
      const { mle } = computeMLE(data, preset.family, preset.paramName, preset.otherParams);
      if (!Number.isFinite(mle)) continue;
      const J = observedFn(mle, data, preset.otherParams);
      const nI = n * fisherFn(mle, preset.otherParams);
      if (Number.isFinite(J) && Number.isFinite(nI)) {
        obs.push(J);
        exp.push(nI);
        mles.push(mle);
      }
    }
    return { obs, exp, mles };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampler, n, preset, runId, observedFn, fisherFn]);

  // ── Scatter (J(θ̂), nI(θ̂)) with identity line ──────────────────────────
  useEffect(() => {
    if (!scatterRef.current) return;
    const svg = d3.select(scatterRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.5) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const all = mc.obs.concat(mc.exp).concat([nITrue]);
    const vmin = d3.min(all) ?? 0;
    const vmax = d3.max(all) ?? 1;
    const pad = Math.max((vmax - vmin) * 0.1, 1);
    const lo = vmin - pad;
    const hi = vmax + pad;
    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]).nice();
    const y = d3.scaleLinear().domain([lo, hi]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(4))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('J(θ̂) vs nI(θ̂) across MC replicates');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Observed J(θ̂)');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Expected nI(θ̂)');

    // Identity line
    const dDomain = x.domain();
    g.append('line')
      .attr('x1', x(dDomain[0]))
      .attr('x2', x(dDomain[1]))
      .attr('y1', y(dDomain[0]))
      .attr('y2', y(dDomain[1]))
      .attr('stroke', '#9ca3af')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3');

    // Theoretical anchor nI(θ₀)
    g.append('circle')
      .attr('cx', x(nITrue))
      .attr('cy', y(nITrue))
      .attr('r', 5)
      .attr('fill', 'none')
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5);
    g.append('text')
      .attr('x', x(nITrue) + 8)
      .attr('y', y(nITrue) - 8)
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text('nI(θ₀)');

    // Scatter points
    g.selectAll('circle.pt')
      .data(mc.obs.map((o, i) => ({ o, e: mc.exp[i] })))
      .enter()
      .append('circle')
      .attr('class', 'pt')
      .attr('cx', (d) => x(d.o))
      .attr('cy', (d) => y(d.e))
      .attr('r', 2.5)
      .attr('fill', '#2563eb')
      .attr('opacity', 0.55);
  }, [mc, w, isWide, nITrue]);

  // ── Histogram overlay: J(θ̂) (blue) and nI(θ̂) (orange) ───────────────────
  useEffect(() => {
    if (!histRef.current) return;
    const svg = d3.select(histRef.current);
    svg.selectAll('*').remove();
    const panelW = isWide ? Math.floor(w * 0.5) : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const all = mc.obs.concat(mc.exp).concat([nITrue]);
    const lo = d3.min(all) ?? 0;
    const hi = d3.max(all) ?? 1;
    const pad = Math.max((hi - lo) * 0.1, 1);
    const x = d3
      .scaleLinear()
      .domain([lo - pad, hi + pad])
      .range([0, innerW])
      .nice();

    const bins = d3
      .bin<number, number>()
      .domain(x.domain() as [number, number])
      .thresholds(24);
    const obsBins = bins(mc.obs);
    const expBins = bins(mc.exp);
    const yMax = Math.max(
      d3.max(obsBins, (b) => b.length) ?? 1,
      d3.max(expBins, (b) => b.length) ?? 1,
    );
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

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
      .text('Histograms of observed vs expected info');

    // Observed (blue, translucent)
    g.selectAll('rect.obs')
      .data(obsBins)
      .enter()
      .append('rect')
      .attr('class', 'obs')
      .attr('x', (d) => x(d.x0 ?? 0) + 1)
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
      .attr('height', (d) => innerH - y(d.length))
      .attr('fill', '#2563eb')
      .attr('opacity', 0.5);

    // Expected (orange, translucent)
    g.selectAll('rect.exp')
      .data(expBins)
      .enter()
      .append('rect')
      .attr('class', 'exp')
      .attr('x', (d) => x(d.x0 ?? 0) + 1)
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
      .attr('height', (d) => innerH - y(d.length))
      .attr('fill', '#f59e0b')
      .attr('opacity', 0.5);

    // Vertical anchor at nI(θ₀)
    g.append('line')
      .attr('x1', x(nITrue))
      .attr('x2', x(nITrue))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 3');
    g.append('text')
      .attr('x', x(nITrue))
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#111827')
      .text('nI(θ₀)');

    // Legend
    g.append('rect')
      .attr('x', innerW - 110)
      .attr('y', 10)
      .attr('width', 10)
      .attr('height', 10)
      .attr('fill', '#2563eb')
      .attr('opacity', 0.5);
    g.append('text')
      .attr('x', innerW - 95)
      .attr('y', 19)
      .style('font-size', '10px')
      .style('fill', 'currentColor')
      .text('J(θ̂)');
    g.append('rect')
      .attr('x', innerW - 60)
      .attr('y', 10)
      .attr('width', 10)
      .attr('height', 10)
      .attr('fill', '#f59e0b')
      .attr('opacity', 0.5);
    g.append('text')
      .attr('x', innerW - 45)
      .attr('y', 19)
      .style('font-size', '10px')
      .style('fill', 'currentColor')
      .text('nI(θ̂)');
  }, [mc, w, isWide, nITrue]);

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');

  const obsMean = mc.obs.length > 0 ? sampleMean(mc.obs) : NaN;
  const expMean = mc.exp.length > 0 ? sampleMean(mc.exp) : NaN;
  const obsVar = mc.obs.length > 1 ? sampleVariance(mc.obs, 1) : NaN;
  const expVar = mc.exp.length > 1 ? sampleVariance(mc.exp, 1) : NaN;

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: Observed vs Expected Fisher Information</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Each of 200 Monte-Carlo replicates produces one pair (J(θ̂), nI(θ̂)). For the
          natural-parameter exponential families below, the two quantities coincide <em>at the
          MLE</em> so the scatter sits exactly on the identity line; what varies sample-to-sample
          is where on that line we land, via θ̂. For Normal with known σ², J(μ̂) = n/σ² is
          non-random and the scatter collapses to a single point.
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
              setRunId((r) => r + 1);
            }}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {observedVsExpectedPresets.map((p, i) => (
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
            min={10}
            max={500}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 120, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
        <button
          onClick={() => setRunId((r) => r + 1)}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          ↻ Rerun 200 replicates
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isWide ? '1fr 1fr' : '1fr',
          gap: '0.5rem',
        }}
      >
        <svg
          ref={scatterRef}
          width={isWide ? Math.floor(w * 0.5) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={histRef}
          width={isWide ? Math.floor(w * 0.5) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
      </div>

      <div
        className="grid"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: '0.25rem 0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.8125rem',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <div>nI(θ₀) {fmt(nITrue)}</div>
        <div>mean J(θ̂) {fmt(obsMean)}</div>
        <div>mean nI(θ̂) {fmt(expMean)}</div>
        <div>var J(θ̂) {fmt(obsVar, 4)}</div>
        <div>var nI(θ̂) {fmt(expVar, 4)}</div>
        <div>replicates {mc.obs.length}</div>
      </div>
    </div>
  );
}
