import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { tripleComparisonMC } from './shared/estimation';
import { umvueComparatorPresets } from '../../data/sufficient-statistics-data';

const MARGIN = { top: 14, right: 12, bottom: 32, left: 38 };
const H = 220;
const BAR_H = 140;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function trueParamFor(family: string, params: Record<string, number>): number {
  switch (family) {
    case 'normal-variance-unknown-mu': return params.sigma2;
    case 'gamma-scale-with-known-alpha': return params.beta;
    case 'exponential-rate': return params.lambda;
    case 'normal-mean-known-sigma': return params.mu;
    case 'poisson-rate': return params.lambda;
    default: return 0;
  }
}

export default function UMVUEComparator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = umvueComparatorPresets[presetIndex];

  const [n, setN] = useState(preset.defaultN);
  const [M, setM] = useState(2000);
  const [seed, setSeed] = useState(17);

  useEffect(() => {
    setN(umvueComparatorPresets[presetIndex].defaultN);
    setSeed((s) => s + 1);
  }, [presetIndex]);

  const truth = trueParamFor(preset.family, preset.trueParams);

  const mc = useMemo(() => {
    const rng = makeLCG(seed * 7919 + n + M);
    return tripleComparisonMC(preset.family, preset.trueParams, n, M, rng);
  }, [preset.family, preset.trueParams, n, M, seed]);

  // ── Histogram overlays ────────────────────────────────────────────────────
  const histRef = useRef<SVGSVGElement | null>(null);
  const histW = isWide ? Math.floor(w * 0.55) : w;

  useEffect(() => {
    if (!histRef.current) return;
    const svg = d3.select(histRef.current);
    svg.selectAll('*').remove();

    const innerW = histW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const all = [...mc.umvueEstimates, ...mc.mleEstimates, ...mc.momEstimates];
    const finite = all.filter((v) => Number.isFinite(v));
    const lo = d3.quantile(finite.slice().sort(d3.ascending), 0.005) ?? d3.min(finite) ?? 0;
    const hi = d3.quantile(finite.slice().sort(d3.ascending), 0.995) ?? d3.max(finite) ?? 1;
    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);

    const histGen = d3.bin<number, number>().domain([lo, hi]).thresholds(40);
    const u = histGen(mc.umvueEstimates.filter(Number.isFinite));
    const ml = histGen(mc.mleEstimates.filter(Number.isFinite));
    const mo = histGen(mc.momEstimates.filter(Number.isFinite));
    const yMax = Math.max(d3.max(u, (b) => b.length) ?? 0, d3.max(ml, (b) => b.length) ?? 0, d3.max(mo, (b) => b.length) ?? 0);
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).style('font-size', '10px');

    const drawHist = (bins: d3.Bin<number, number>[], color: string, opacity: number) => {
      g.selectAll(null).data(bins).enter().append('rect')
        .attr('x', (b) => x(b.x0 ?? 0))
        .attr('y', (b) => y(b.length))
        .attr('width', (b) => Math.max(1, x(b.x1 ?? 0) - x(b.x0 ?? 0) - 1))
        .attr('height', (b) => innerH - y(b.length))
        .attr('fill', color).attr('opacity', opacity);
    };
    // MoM under MLE (often identical), then MLE, then UMVUE on top
    drawHist(mo, 'rgb(70,130,180)', 0.4);
    drawHist(ml, 'rgb(220,100,70)', 0.45);
    drawHist(u, 'rgb(70,160,90)', 0.5);

    // True parameter line
    g.append('line')
      .attr('x1', x(truth)).attr('x2', x(truth))
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'currentColor').attr('stroke-dasharray', '3,3').attr('opacity', 0.6);
    g.append('text')
      .attr('x', x(truth)).attr('y', -2)
      .attr('text-anchor', 'middle').attr('font-size', 10)
      .attr('fill', 'currentColor').text(`true = ${truth}`);

    const legend = g.append('g').attr('transform', `translate(${innerW - 100},6)`);
    [
      { c: 'rgb(70,160,90)', label: 'UMVUE' },
      { c: 'rgb(220,100,70)', label: 'MLE' },
      { c: 'rgb(70,130,180)', label: 'MoM' },
    ].forEach((row, i) => {
      legend.append('rect').attr('width', 10).attr('height', 10).attr('y', i * 14).attr('fill', row.c).attr('opacity', 0.6);
      legend.append('text').attr('x', 14).attr('y', i * 14 + 9).attr('font-size', 10).attr('fill', 'currentColor').text(row.label);
    });
  }, [mc, histW, truth]);

  // ── Bias / MSE bars ───────────────────────────────────────────────────────
  const barRef = useRef<SVGSVGElement | null>(null);
  const barW = isWide ? Math.floor(w * 0.4) : w;

  useEffect(() => {
    if (!barRef.current) return;
    const svg = d3.select(barRef.current);
    svg.selectAll('*').remove();

    const groups = [
      { name: 'UMVUE', color: 'rgb(70,160,90)', bias: mc.umvueBias, mse: mc.umvueMSE },
      { name: 'MLE', color: 'rgb(220,100,70)', bias: mc.mleBias, mse: mc.mleMSE },
      { name: 'MoM', color: 'rgb(70,130,180)', bias: mc.momBias, mse: mc.momMSE },
    ];

    const innerW = barW - MARGIN.left - MARGIN.right;
    const halfH = (BAR_H - 30) / 2;

    // Bias chart (top)
    const gBias = svg.append('g').attr('transform', `translate(${MARGIN.left},14)`);
    const xb = d3.scaleBand().domain(groups.map((g) => g.name)).range([0, innerW]).padding(0.3);
    const biasMax = Math.max(...groups.map((g) => Math.abs(g.bias)), 1e-3) * 1.2;
    const yb = d3.scaleLinear().domain([-biasMax, biasMax]).range([halfH, 0]);
    gBias.append('g').call(d3.axisLeft(yb).ticks(3)).style('font-size', '9px');
    gBias.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yb(0)).attr('y2', yb(0)).attr('stroke', 'currentColor').attr('opacity', 0.4);
    gBias.selectAll('rect.bias').data(groups).join('rect')
      .attr('x', (g) => xb(g.name) ?? 0)
      .attr('y', (g) => Math.min(yb(0), yb(g.bias)))
      .attr('width', xb.bandwidth())
      .attr('height', (g) => Math.abs(yb(g.bias) - yb(0)))
      .attr('fill', (g) => g.color).attr('opacity', 0.7);
    gBias.append('text').attr('x', innerW / 2).attr('y', -2).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600).attr('fill', 'currentColor').text('Bias');

    // MSE chart (bottom)
    const gMse = svg.append('g').attr('transform', `translate(${MARGIN.left},${20 + halfH + 14})`);
    const mseMax = Math.max(...groups.map((g) => g.mse)) * 1.1;
    const ym = d3.scaleLinear().domain([0, mseMax]).range([halfH, 0]);
    gMse.append('g').call(d3.axisLeft(ym).ticks(3)).style('font-size', '9px');
    gMse.append('g').attr('transform', `translate(0,${halfH})`).call(d3.axisBottom(xb)).style('font-size', '10px');
    gMse.selectAll('rect.mse').data(groups).join('rect')
      .attr('x', (g) => xb(g.name) ?? 0)
      .attr('y', (g) => ym(g.mse))
      .attr('width', xb.bandwidth())
      .attr('height', (g) => halfH - ym(g.mse))
      .attr('fill', (g) => g.color).attr('opacity', 0.7);
    gMse.append('text').attr('x', innerW / 2).attr('y', -2).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600).attr('fill', 'currentColor').text('MSE');
  }, [mc, barW]);

  return (
    <div ref={containerRef} className="not-prose my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}>
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          UMVUE vs MLE vs MoM · §16.11 · Track 4 closer
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {umvueComparatorPresets.map((p, i) => (
            <option key={i} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>n = {n}</span>
          <input type="range" min={10} max={500} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-full" />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>M = {M}</span>
          <input type="range" min={500} max={5000} step={100} value={M} onChange={(e) => setM(Number(e.target.value))} className="w-full" />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-run MC
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-3 gap-4' : 'flex flex-col gap-4'}>
        <div className="col-span-2">
          <svg ref={histRef} width={histW} height={H} style={{ display: 'block' }} />
        </div>
        <div>
          <svg ref={barRef} width={barW} height={BAR_H + 50} style={{ display: 'block' }} />
        </div>
      </div>

      <div className="mt-4 text-sm space-y-2">
        <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
          Estimator formulas
        </div>
        <div className="font-mono text-xs leading-relaxed p-3 rounded space-y-1" style={{ background: 'var(--color-code-bg, rgba(127,127,127,0.08))' }}>
          <div><span style={{ color: 'rgb(70,160,90)' }}>UMVUE:</span> {preset.umvueFormula}</div>
          <div><span style={{ color: 'rgb(220,100,70)' }}>MLE:</span> {preset.mleFormula}</div>
          <div><span style={{ color: 'rgb(70,130,180)' }}>MoM:</span> {preset.momFormula}</div>
        </div>
        <div className="text-xs grid grid-cols-3 gap-2">
          <div>UMVUE bias: <strong>{mc.umvueBias.toFixed(4)}</strong></div>
          <div>MLE bias: <strong>{mc.mleBias.toFixed(4)}</strong></div>
          <div>MoM bias: <strong>{mc.momBias.toFixed(4)}</strong></div>
          <div>UMVUE MSE: <strong>{mc.umvueMSE.toFixed(4)}</strong></div>
          <div>MLE MSE: <strong>{mc.mleMSE.toFixed(4)}</strong></div>
          <div>MoM MSE: <strong>{mc.momMSE.toFixed(4)}</strong></div>
        </div>
        <div className="text-xs italic" style={{ color: 'var(--color-text-muted)' }}>{preset.description}</div>
      </div>
    </div>
  );
}
