import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { runningMean } from './shared/convergence';
import { runningMedian } from './shared/estimation';
import { cauchySample, paretoSampleArray, normalSample, sampleSequence } from './shared/convergence';
import { pathologyPresets } from '../../data/method-of-moments-data';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 48 };
const H = 220;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function generateSample(family: string, params: Record<string, number>, n: number, rng: () => number): number[] {
  switch (family) {
    case 'cauchy':
      return cauchySample(n, params.location, params.scale, rng);
    case 'pareto':
      return paretoSampleArray(n, params.alpha, params.xm, rng);
    case 'normal':
      return sampleSequence(() => normalSample(params.mu, Math.sqrt(params.sigma2), rng), n);
    default:
      return [];
  }
}

// Two-heap O(n log n) running median for streaming use up to n=10,000 (gotcha #9).
function fastRunningMedian(data: number[]): number[] {
  const out = new Array<number>(data.length);
  // Sorted insertion using bisect — O(n²) worst case for the *insert* but O(log n) for the search,
  // adequate for n ≤ 10,000 and avoids the nested-sort O(n²) of the naive runningMedian.
  const sorted: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    sorted.splice(lo, 0, x);
    const m = sorted.length;
    out[i] = m % 2 === 0 ? 0.5 * (sorted[m / 2 - 1] + sorted[m / 2]) : sorted[(m - 1) / 2];
  }
  return out;
}

export default function CauchyPathologyExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = pathologyPresets[presetIndex];

  const [n, setN] = useState(1000);
  const [seed, setSeed] = useState(1);

  const sample = useMemo(() => {
    const rng = makeLCG(seed * 7 + n);
    return generateSample(preset.family, preset.params, n, rng);
  }, [preset, n, seed]);

  const rmean = useMemo(() => runningMean(sample), [sample]);
  const rmedian = useMemo(() => fastRunningMedian(sample), [sample]);

  // Max excursion of the running mean — the headline statistic.
  const maxAbsMean = useMemo(() => {
    let m = 0;
    for (let i = 0; i < rmean.length; i++) {
      const a = Math.abs(rmean[i]);
      if (a > m) m = a;
    }
    return m;
  }, [rmean]);

  const meanRef = useRef<SVGSVGElement | null>(null);
  const medianRef = useRef<SVGSVGElement | null>(null);

  const panelW = isWide ? Math.floor((w - 24) / 2) : w;

  useEffect(() => {
    [{ ref: meanRef, data: rmean, color: '#DC2626', label: 'Running mean' },
     { ref: medianRef, data: rmedian, color: '#059669', label: 'Running median' }
    ].forEach(({ ref, data, color, label }) => {
      const svg = d3.select(ref.current);
      if (!svg.node()) return;
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = H - MARGIN.top - MARGIN.bottom;
      const g = svg.attr('viewBox', `0 0 ${panelW} ${H}`).append('g')
        .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

      const x = d3.scaleLog().domain([Math.max(1, n / 1000), n]).clamp(true).range([0, innerW]);
      // Symmetric y around the population median (or 0)
      const yMax = Math.max(d3.max(data.map(Math.abs)) ?? 1, 1);
      const y = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(x).ticks(5, '~g'))
        .call((sel) => sel.selectAll('text').style('font-size', '10px'));
      g.append('g').call(d3.axisLeft(y).ticks(5))
        .call((sel) => sel.selectAll('text').style('font-size', '10px'));

      // True median reference line.
      g.append('line').attr('x1', 0).attr('x2', innerW)
        .attr('y1', y(preset.trueMedian)).attr('y2', y(preset.trueMedian))
        .attr('stroke', 'var(--color-text-muted)').attr('stroke-dasharray', '4 3');

      const line = d3.line<number>()
        .defined((_, i) => i + 1 >= n / 1000)
        .x((_, i) => x(i + 1))
        .y((d) => y(Math.max(-yMax, Math.min(yMax, d))));

      g.append('path').datum(data).attr('fill', 'none')
        .attr('stroke', color).attr('stroke-width', 1.4).attr('d', line);

      g.append('text').attr('x', 4).attr('y', 12).attr('font-size', 11).attr('fill', color)
        .text(label);
    });
  }, [rmean, rmedian, panelW, n, preset]);

  const reset = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: Running mean vs running median</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          For Cauchy and heavy-tailed Pareto, the running mean never converges (the LLN fails because E[|X|] = ∞), but the running median converges smoothly to the population median. Watch the contrast.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center" style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}>
        <select value={presetIndex} onChange={(e) => { setPresetIndex(Number(e.target.value)); setSeed((s) => s + 1); }}
          style={{ padding: '0.2rem' }}>
          {pathologyPresets.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
        </select>
        <label>n = {n}{' '}
          <input type="range" min={100} max={10000} step={100}
            value={n} onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 130, verticalAlign: 'middle' }} />
        </label>
        <button onClick={reset}
          style={{ padding: '0.25rem 0.65rem', fontSize: '0.8125rem',
            border: '1px solid var(--color-border)', borderRadius: '0.375rem',
            background: 'transparent', cursor: 'pointer' }}>
          New seed
        </button>
      </div>

      <div style={{ display: isWide ? 'flex' : 'block', gap: 12 }}>
        <svg ref={meanRef} width={panelW} height={H} style={{ display: 'block' }} />
        <svg ref={medianRef} width={panelW} height={H} style={{ display: 'block' }} />
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 8, lineHeight: 1.5 }}>
        <strong>Diagnosis.</strong>{' '}
        Population mean exists: <strong>{preset.hasMean ? 'yes' : 'no'}</strong>. Population variance exists: <strong>{preset.hasVariance ? 'yes' : 'no'}</strong>. Population median = <code>{preset.trueMedian.toFixed(3)}</code>. Maximum absolute excursion of the running mean (so far): <code>{maxAbsMean.toFixed(2)}</code>.
        <br />
        {!preset.hasMean
          ? <>The MoM equation X̄<sub>n</sub> = μ<sub>1</sub>(θ̂) has nothing to match — there is no population moment to converge to.</>
          : <>The mean exists and X̄<sub>n</sub> converges, but the variance may not — making higher-moment MoM equations fail even when the first works.</>}
      </div>
    </div>
  );
}
