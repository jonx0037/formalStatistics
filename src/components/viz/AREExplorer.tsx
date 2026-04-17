import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  momGamma,
  momExponential,
  momNormal,
  momUniform,
  mleGammaShape,
  sampleMean,
  sampleVariance,
  areTheoretical,
  areCurveTheoretical,
} from './shared/estimation';
import {
  gammaSample,
  exponentialSample,
  normalSample,
  uniformSample,
  sampleSequence,
} from './shared/convergence';
import { areFamilyPresets } from '../../data/method-of-moments-data';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 48 };
const H = 240;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

interface FamilyOps {
  sample: (n: number, rng: () => number) => number[];
  mom: (data: number[]) => number;
  mle: (data: number[]) => number;
  trueParam: (params: Record<string, number>) => number;
}

function getOps(family: string): FamilyOps {
  switch (family) {
    case 'gamma-shape':
      return {
        sample: (n, rng) => sampleSequence(() => gammaSample(3, 2, rng), n),
        mom: (d) => momGamma(d).alphaHat,
        mle: (d) => {
          const xbar = sampleMean(d);
          const beta0 = momGamma(d).alphaHat / Math.max(xbar, 1e-9);
          return mleGammaShape(d, beta0, { init: momGamma(d).alphaHat, maxIter: 30 }).mle;
        },
        trueParam: (p) => p.alpha,
      };
    case 'exponential-rate':
      return {
        sample: (n, rng) => sampleSequence(() => exponentialSample(1, rng), n),
        mom: (d) => momExponential(d),
        mle: (d) => momExponential(d), // identical to MoM
        trueParam: (p) => p.lambda,
      };
    case 'normal-variance':
      return {
        sample: (n, rng) => sampleSequence(() => normalSample(0, Math.sqrt(4), rng), n),
        mom: (d) => momNormal(d).sigma2Hat,
        mle: (d) => momNormal(d).sigma2Hat, // identical (both biased 1/n)
        trueParam: (p) => p.sigma2,
      };
    case 'uniform-endpoint':
      return {
        sample: (n, rng) => sampleSequence(() => uniformSample(0, 1, rng), n),
        mom: (d) => momUniform(d),
        mle: (d) => Math.max(...d), // X_(n)
        trueParam: (p) => p.theta,
      };
    default:
      throw new Error(`Unknown family ${family}`);
  }
}

export default function AREExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = areFamilyPresets[presetIndex];
  const [n, setN] = useState(100);
  const [reps, setReps] = useState(500);
  const [seed, setSeed] = useState(1);

  const ops = useMemo(() => getOps(preset.family), [preset.family]);
  const truth = preset.defaultParams[Object.keys(preset.defaultParams)[0]];
  const trueParam = ops.trueParam(preset.defaultParams);

  // Run the Monte Carlo sampling-distribution comparison.
  const sim = useMemo(() => {
    const rng = makeLCG(seed * 17 + n + reps);
    const momEst: number[] = [];
    const mleEst: number[] = [];
    for (let r = 0; r < reps; r++) {
      const data = ops.sample(n, rng);
      const m = ops.mom(data);
      const ml = ops.mle(data);
      if (Number.isFinite(m)) momEst.push(m);
      if (Number.isFinite(ml)) mleEst.push(ml);
    }
    const meanMom = momEst.reduce((s, v) => s + v, 0) / momEst.length;
    const meanMle = mleEst.reduce((s, v) => s + v, 0) / mleEst.length;
    const varMom = momEst.reduce((s, v) => s + (v - meanMom) ** 2, 0) / momEst.length;
    const varMle = mleEst.reduce((s, v) => s + (v - meanMle) ** 2, 0) / mleEst.length;
    const empARE = varMle > 0 ? varMle / varMom : NaN;
    const theoreticalARE = preset.family === 'uniform-endpoint'
      ? areTheoretical(preset.family, { ...preset.defaultParams, n })
      : areTheoretical(preset.family, preset.defaultParams);
    return { momEst, mleEst, meanMom, meanMle, varMom, varMle, empARE, theoreticalARE };
  }, [seed, n, reps, ops, preset]);

  // ── Left panel: overlaid histograms of MoM and MLE estimates ─────────────
  const leftRef = useRef<SVGSVGElement | null>(null);
  const leftW = isWide ? Math.floor(w * 0.55) : w;

  useEffect(() => {
    const svg = d3.select(leftRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = leftW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${leftW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const allEst = [...sim.momEst, ...sim.mleEst];
    if (allEst.length === 0) return;
    const lo = (d3.min(allEst) ?? trueParam) * 0.9;
    const hi = (d3.max(allEst) ?? trueParam) * 1.1;
    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
    const bins = d3.bin().domain([lo, hi]).thresholds(30);
    const momBins = bins(sim.momEst);
    const mleBins = bins(sim.mleEst);
    const yMax = (d3.max([...momBins, ...mleBins], (b) => b.length) ?? 0) * 1.05;
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(6))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));

    g.selectAll('rect.mom').data(momBins).enter().append('rect')
      .attr('class', 'mom').attr('x', (b) => x(b.x0!))
      .attr('y', (b) => y(b.length))
      .attr('width', (b) => Math.max(0, x(b.x1!) - x(b.x0!) - 1))
      .attr('height', (b) => innerH - y(b.length))
      .attr('fill', '#2563EB').attr('opacity', 0.4);

    g.selectAll('rect.mle').data(mleBins).enter().append('rect')
      .attr('class', 'mle').attr('x', (b) => x(b.x0!))
      .attr('y', (b) => y(b.length))
      .attr('width', (b) => Math.max(0, x(b.x1!) - x(b.x0!) - 1))
      .attr('height', (b) => innerH - y(b.length))
      .attr('fill', '#D97706').attr('opacity', 0.4);

    g.append('line').attr('x1', x(trueParam)).attr('x2', x(trueParam))
      .attr('y1', 0).attr('y2', innerH).attr('stroke', '#111')
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');

    g.append('text').attr('x', x(trueParam) + 4).attr('y', 12)
      .attr('font-size', 10).attr('fill', '#111').text('truth');

    const lg = g.append('g').attr('transform', `translate(${innerW - 100}, 6)`)
      .style('font-size', '10px').attr('fill', 'var(--color-text)');
    [{ c: '#2563EB', label: 'MoM' }, { c: '#D97706', label: 'MLE' }].forEach((it, i) => {
      lg.append('rect').attr('x', 0).attr('y', i * 14 - 8).attr('width', 12).attr('height', 10)
        .attr('fill', it.c).attr('opacity', 0.5);
      lg.append('text').attr('x', 18).attr('y', i * 14).text(it.label);
    });
  }, [sim, leftW, trueParam]);

  // ── Right panel: theoretical ARE curve over the parameter grid ───────────
  const rightRef = useRef<SVGSVGElement | null>(null);
  const rightW = isWide ? w - leftW - 12 : w;

  useEffect(() => {
    const svg = d3.select(rightRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = rightW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${rightW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const grid = preset.paramGrid;
    const sweepName = preset.sweepParam;
    const fixed = { ...preset.defaultParams };
    const curve = areCurveTheoretical(preset.family, sweepName, grid, fixed);
    const x = d3.scaleLog().domain([Math.max(grid[0], 0.5), grid[grid.length - 1]]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(x).ticks(5, '~g'))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('text').attr('x', innerW / 2).attr('y', innerH + 28)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-text-muted)')
      .text(sweepName + ' (log scale)');
    g.append('text').attr('x', -innerH / 2).attr('y', -34).attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-text-muted)')
      .text('ARE (MoM/MLE)');

    const line = d3.line<number>()
      .defined((d) => Number.isFinite(d))
      .x((_, i) => x(curve.grid[i]))
      .y((d) => y(Math.min(1, d)));

    g.append('path').datum(curve.are).attr('fill', 'none')
      .attr('stroke', '#374151').attr('stroke-width', 2).attr('d', line);

    // Empirical point at the current settings.
    if (Number.isFinite(sim.empARE)) {
      g.append('circle').attr('cx', x(truth)).attr('cy', y(Math.min(1, sim.empARE)))
        .attr('r', 5).attr('fill', '#DC2626').attr('opacity', 0.85);
      g.append('text').attr('x', x(truth) + 8).attr('y', y(Math.min(1, sim.empARE)) - 4)
        .attr('font-size', 10).attr('fill', '#DC2626')
        .text(`empirical: ${sim.empARE.toFixed(3)}`);
    }

    g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', y(1)).attr('y2', y(1))
      .attr('stroke', 'var(--color-text-muted)').attr('stroke-dasharray', '3 3');
  }, [sim, rightW, preset, truth]);

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: ARE(MoM, MLE) — empirical vs theoretical</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Pick a family. The left panel overlays the sampling distributions of MoM (blue) and MLE (orange) over many Monte-Carlo replications; the right panel plots the theoretical ARE curve with the empirical point marked in red.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center" style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}>
        <select value={presetIndex} onChange={(e) => { setPresetIndex(Number(e.target.value)); setSeed((s) => s + 1); }}
          style={{ padding: '0.2rem' }}>
          {areFamilyPresets.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
        </select>
        <label>n = {n}{' '}
          <input type="range" min={20} max={1000} step={10}
            value={n} onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 110, verticalAlign: 'middle' }} />
        </label>
        <label>reps = {reps}{' '}
          <input type="range" min={100} max={3000} step={100}
            value={reps} onChange={(e) => setReps(Number(e.target.value))}
            style={{ width: 110, verticalAlign: 'middle' }} />
        </label>
        <button onClick={() => setSeed((s) => s + 1)}
          style={{ padding: '0.25rem 0.65rem', fontSize: '0.8125rem',
            border: '1px solid var(--color-border)', borderRadius: '0.375rem',
            background: 'transparent', cursor: 'pointer' }}>
          Re-run simulation
        </button>
      </div>

      <div style={{ display: isWide ? 'flex' : 'block', gap: 12 }}>
        <svg ref={leftRef} width={leftW} height={H} style={{ display: 'block' }} />
        <svg ref={rightRef} width={rightW} height={H} style={{ display: 'block' }} />
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 8, lineHeight: 1.4 }}>
        <strong>Empirical Var(MoM):</strong> {sim.varMom.toFixed(5)} ·{' '}
        <strong>Empirical Var(MLE):</strong> {sim.varMle.toFixed(5)} ·{' '}
        <strong>Empirical ARE:</strong> {sim.empARE.toFixed(3)} ·{' '}
        <strong>Theoretical ARE:</strong> {Number.isFinite(sim.theoreticalARE) ? sim.theoreticalARE.toFixed(3) : '—'}{' '}
        <span style={{ color: 'var(--color-text-muted)' }}>
          ({preset.theoreticalAREFn})
        </span>
      </div>
    </div>
  );
}
