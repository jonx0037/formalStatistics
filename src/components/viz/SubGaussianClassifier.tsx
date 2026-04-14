import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  normalSample,
  exponentialSample,
  chiSquaredSample,
  uniformSample,
  bernoulliSample,
  poissonSample,
  tSample,
  paretoSample,
} from './shared/convergence';
import {
  subGaussianPresets,
  type SubGaussianPreset,
} from '../../data/large-deviations-data';

const MARGIN = { top: 10, right: 16, bottom: 36, left: 50 };
const H = 260;
const NUM_SAMPLES = 100_000; // 10⁵ instead of 10⁶ for responsiveness; still gives clean envelopes

function cauchySample(): number {
  const u = Math.random();
  return Math.tan(Math.PI * (u - 0.5));
}

function makeSampler(id: string): () => number {
  switch (id) {
    case 'normal':
      return () => normalSample(0, 1);
    case 'uniform-symmetric':
      return () => uniformSample(-1, 1);
    case 'rademacher':
      return () => (bernoulliSample(0.5) === 1 ? 1 : -1);
    case 'exponential':
      return () => exponentialSample(1) - 1;
    case 'chi-squared':
      return () => chiSquaredSample(1) - 1;
    case 'poisson':
      return () => poissonSample(5) - 5;
    case 't3':
      return () => tSample(3);
    case 't5':
      return () => tSample(5);
    case 'cauchy':
      return cauchySample;
    case 'pareto':
      return () => paretoSample(2) - 2;
    default:
      return () => normalSample(0, 1);
  }
}

const BADGE_COLOR: Record<string, string> = {
  'sub-gaussian': '#059669',
  'sub-exponential': '#d97706',
  'heavy-tailed': '#dc2626',
};

const BADGE_BG: Record<string, string> = {
  'sub-gaussian': 'rgba(5, 150, 105, 0.15)',
  'sub-exponential': 'rgba(217, 119, 6, 0.15)',
  'heavy-tailed': 'rgba(220, 38, 38, 0.15)',
};

export default function SubGaussianClassifier() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 640, 320);

  const [distId, setDistId] = useState<string>('normal');
  const [isLoading, setIsLoading] = useState(true);
  const samplesRef = useRef<Float64Array | null>(null);
  const [_, setBump] = useState(0); // trigger re-render after sample generation

  const preset: SubGaussianPreset =
    subGaussianPresets.find((p) => p.id === distId) ?? subGaussianPresets[0];

  useEffect(() => {
    setIsLoading(true);
    samplesRef.current = null;
    // Run generation in next tick so loading state renders first
    const handle = setTimeout(() => {
      const sampler = makeSampler(distId);
      const buf = new Float64Array(NUM_SAMPLES);
      for (let i = 0; i < NUM_SAMPLES; i++) buf[i] = sampler();
      samplesRef.current = buf;
      setIsLoading(false);
      setBump((b) => b + 1);
    }, 10);
    return () => clearTimeout(handle);
  }, [distId]);

  const tailData = useMemo(() => {
    if (!samplesRef.current) return null;
    const samples = samplesRef.current;
    // Use |X|, sort, compute empirical survival function at a grid
    const abs = new Float64Array(samples.length);
    for (let i = 0; i < samples.length; i++) abs[i] = Math.abs(samples[i]);
    const sorted = Array.from(abs).sort((a, b) => a - b);
    const n = sorted.length;
    const cap = sorted[Math.floor(n * 0.999)]; // cap at 99.9% to trim outliers
    const pts: Array<{ t: number; surv: number }> = [];
    const steps = 80;
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * Math.min(cap, 8);
      // empirical survival S(t) = P(|X| > t) = fraction of sorted > t
      // binary search in sorted for t
      let lo = 0;
      let hi = n;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (sorted[mid] <= t) lo = mid + 1;
        else hi = mid;
      }
      const surv = 1 - lo / n;
      if (surv > 0) pts.push({ t, surv });
    }
    return pts;
  }, [isLoading, distId]);

  const stats = useMemo(() => {
    if (!samplesRef.current) return null;
    const s = samplesRef.current;
    // Sample variance
    let mean = 0;
    for (let i = 0; i < s.length; i++) mean += s[i];
    mean /= s.length;
    let m2 = 0;
    for (let i = 0; i < s.length; i++) {
      const d = s[i] - mean;
      m2 += d * d;
    }
    const variance = m2 / s.length;
    // ψ_2 Orlicz norm estimate: find smallest c such that mean(exp(X²/c²)) ≤ 2
    // Rough bisection
    let lo = 0.1;
    let hi = 20;
    for (let iter = 0; iter < 40; iter++) {
      const c = (lo + hi) / 2;
      let sum = 0;
      for (let i = 0; i < s.length; i++) {
        sum += Math.exp((s[i] * s[i]) / (c * c));
        if (!isFinite(sum)) break;
      }
      const avg = sum / s.length;
      if (!isFinite(avg) || avg > 2) lo = c;
      else hi = c;
    }
    const psi2 = hi;
    return { variance, psi2 };
  }, [isLoading, distId]);

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !tailData || tailData.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const innerW = w - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const tMax = d3.max(tailData, (d) => d.t) ?? 5;
    const xScale = d3.scaleLinear().domain([0, tMax]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([1e-5, 1]).range([innerH, 0]).clamp(true);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g')
      .call(
        d3
          .axisLeft(yScale)
          .tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
          .tickFormat((d) => (d as number).toExponential(0))
      )
      .selectAll('text')
      .style('font-size', '10px');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('t');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('P(|X| > t)');

    // Empirical tail (purple)
    const line = d3
      .line<{ t: number; surv: number }>()
      .defined((d) => d.surv > 1e-6)
      .x((d) => xScale(d.t))
      .y((d) => yScale(Math.max(d.surv, 1e-6)));
    g.append('path')
      .attr('d', line(tailData) ?? '')
      .attr('fill', 'none')
      .attr('stroke', '#7c3aed')
      .attr('stroke-width', 2);

    // Sub-Gaussian envelope 2 exp(−t²/(2σ²))
    if (preset.subGaussianParam !== undefined) {
      const sigma = preset.subGaussianParam;
      const steps = 100;
      const pts: Array<{ t: number; y: number }> = [];
      for (let i = 0; i <= steps; i++) {
        const t = (i / steps) * tMax;
        pts.push({ t, y: 2 * Math.exp(-(t * t) / (2 * sigma * sigma)) });
      }
      const sgLine = d3
        .line<{ t: number; y: number }>()
        .defined((d) => d.y > 1e-6)
        .x((d) => xScale(d.t))
        .y((d) => yScale(Math.max(d.y, 1e-6)));
      g.append('path')
        .attr('d', sgLine(pts) ?? '')
        .attr('fill', 'none')
        .attr('stroke', '#059669')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 3');
    }

    // Sub-exponential envelope 2 exp(−t/b)
    if (preset.subExponentialParams) {
      const { b } = preset.subExponentialParams;
      const steps = 100;
      const pts: Array<{ t: number; y: number }> = [];
      for (let i = 0; i <= steps; i++) {
        const t = (i / steps) * tMax;
        pts.push({ t, y: 2 * Math.exp(-t / b) });
      }
      const seLine = d3
        .line<{ t: number; y: number }>()
        .defined((d) => d.y > 1e-6)
        .x((d) => xScale(d.t))
        .y((d) => yScale(Math.max(d.y, 1e-6)));
      g.append('path')
        .attr('d', seLine(pts) ?? '')
        .attr('fill', 'none')
        .attr('stroke', '#d97706')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '2 2');
    }
  }, [tailData, w, preset]);

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
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          alignItems: 'center',
          marginBottom: '0.75rem',
        }}
      >
        <strong>Sub-Gaussian Classifier</strong>
        <select
          value={distId}
          onChange={(e) => setDistId(e.target.value)}
          style={{ padding: '0.2rem', marginLeft: 'auto' }}
        >
          {subGaussianPresets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div
        style={{
          display: 'inline-block',
          padding: '0.35rem 0.75rem',
          borderRadius: '0.375rem',
          background: BADGE_BG[preset.classification],
          color: BADGE_COLOR[preset.classification],
          border: `1px solid ${BADGE_COLOR[preset.classification]}`,
          fontWeight: 600,
          fontSize: '0.875rem',
          marginBottom: '0.75rem',
        }}
      >
        {preset.classification === 'sub-gaussian'
          ? `Sub-Gaussian${preset.subGaussianParam ? ` (σ_sg = ${preset.subGaussianParam})` : ''}`
          : preset.classification === 'sub-exponential'
          ? `Sub-Exponential${preset.subExponentialParams ? ` (ν², b) = (${preset.subExponentialParams.nu2}, ${preset.subExponentialParams.b})` : ''}`
          : 'Heavy-Tailed'}
      </div>

      {isLoading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--color-text-muted)' }}>
          Generating {NUM_SAMPLES.toLocaleString()} samples…
        </div>
      ) : (
        <svg ref={svgRef} width={w} height={H} />
      )}

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          fontSize: '0.75rem',
          marginTop: '0.5rem',
        }}
      >
        <span style={{ color: '#7c3aed' }}>● empirical tail</span>
        {preset.subGaussianParam !== undefined && (
          <span style={{ color: '#059669' }}>-- sub-Gaussian envelope 2e^(−t²/2σ²)</span>
        )}
        {preset.subExponentialParams && (
          <span style={{ color: '#d97706' }}>·· sub-exponential envelope 2e^(−t/b)</span>
        )}
      </div>

      {stats && (
        <div
          style={{
            fontSize: '0.75rem',
            color: 'var(--color-text-muted)',
            marginTop: '0.5rem',
          }}
        >
          Sample variance: {stats.variance.toFixed(4)}; estimated ψ₂ Orlicz norm:{' '}
          {isFinite(stats.psi2) && stats.psi2 < 19 ? stats.psi2.toFixed(3) : '∞ (heavy tail)'}.
          <br />
          {preset.description}
        </div>
      )}
    </div>
  );
}
