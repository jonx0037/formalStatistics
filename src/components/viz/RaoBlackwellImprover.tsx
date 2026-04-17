import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  raoBlackwellClosedForm,
  sampleMean,
  sampleVariance,
} from './shared/estimation';
import {
  bernoulliSample,
  poissonSample,
  normalSample,
  sampleSequence,
} from './shared/convergence';
import { raoBlackwellPresets } from '../../data/sufficient-statistics-data';

const MARGIN = { top: 14, right: 12, bottom: 32, left: 38 };
const H = 220;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function trueValueFor(closedForm: string, params: Record<string, number>): number {
  switch (closedForm) {
    case 'bernoulli-p':
      return params.p;
    case 'bernoulli-variance':
      return params.p * (1 - params.p);
    case 'poisson-zero-prob':
      return Math.exp(-params.lambda);
    case 'normal-tail-prob':
      // Φ((c - μ)/σ) tail. For unknown σ², approximate via std normal.
      const z = (params.c - params.mu) / Math.sqrt(params.sigma2 ?? 1);
      // 1 - Φ(z); use erf approximation
      const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
      const phi = 0.5 * (1 + Math.sign(z) * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2)));
      return 1 - phi;
    default:
      return 0;
  }
}

function drawDataset(closedForm: string, params: Record<string, number>, n: number, rng: () => number): number[] {
  switch (closedForm) {
    case 'bernoulli-p':
    case 'bernoulli-variance':
      return sampleSequence(() => bernoulliSample(params.p, rng), n);
    case 'poisson-zero-prob':
      return sampleSequence(() => poissonSample(params.lambda, rng), n);
    case 'normal-tail-prob':
      return sampleSequence(() => normalSample(params.mu, Math.sqrt(params.sigma2), rng), n);
    default:
      return [];
  }
}

function crudeOf(closedForm: string, data: number[], params: Record<string, number>): number {
  switch (closedForm) {
    case 'bernoulli-p':
      return data[0];
    case 'bernoulli-variance':
      return data[0] * (1 - data[0]); // identically 0
    case 'poisson-zero-prob':
      return data[0] === 0 ? 1 : 0;
    case 'normal-tail-prob':
      return data[0] > params.c ? 1 : 0;
    default:
      return NaN;
  }
}

export default function RaoBlackwellImprover() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = raoBlackwellPresets[presetIndex];

  const [params, setParams] = useState<Record<string, number>>({ ...preset.defaultParams });
  const [n, setN] = useState(preset.defaultParams.n ?? 20);
  const [M, setM] = useState(2000);
  const [seed, setSeed] = useState(7);

  useEffect(() => {
    const p = raoBlackwellPresets[presetIndex];
    setParams({ ...p.defaultParams });
    setN(p.defaultParams.n ?? 20);
    setSeed((s) => s + 1);
  }, [presetIndex]);

  // Run M MC datasets, compute crude and RB'd on each.
  const mc = useMemo(() => {
    const rng = makeLCG(seed * 7919 + n + M);
    const crude: number[] = new Array(M);
    const rb: number[] = new Array(M);
    for (let m = 0; m < M; m++) {
      const data = drawDataset(preset.closedForm, params, n, rng);
      crude[m] = crudeOf(preset.closedForm, data, params);
      rb[m] = raoBlackwellClosedForm(preset.closedForm, data, { threshold: params.c, knownSigma: undefined });
    }
    const tp = trueValueFor(preset.closedForm, params);
    const crudeMSE = crude.reduce((s, v) => s + (v - tp) ** 2, 0) / M;
    const rbMSE = rb.reduce((s, v) => s + (v - tp) ** 2, 0) / M;
    return { crude, rb, crudeMSE, rbMSE, truth: tp };
  }, [preset.closedForm, params, n, M, seed]);

  // ── Histogram of crude vs RB'd ────────────────────────────────────────────
  const histRef = useRef<SVGSVGElement | null>(null);
  const histW = isWide ? Math.floor(w * 0.55) : w;

  useEffect(() => {
    if (!histRef.current) return;
    const svg = d3.select(histRef.current);
    svg.selectAll('*').remove();

    const innerW = histW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const all = [...mc.crude, ...mc.rb];
    const xMin = Math.min(0, d3.min(all) ?? 0) - 0.05;
    const xMax = Math.max(1, d3.max(all) ?? 1) + 0.05;
    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

    const histGen = d3.bin<number, number>().domain([xMin, xMax]).thresholds(40);
    const crudeBins = histGen(mc.crude);
    const rbBins = histGen(mc.rb);
    const yMax = Math.max(d3.max(crudeBins, (b) => b.length) ?? 0, d3.max(rbBins, (b) => b.length) ?? 0);
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).style('font-size', '10px');

    g.selectAll('rect.crude')
      .data(crudeBins)
      .join('rect')
      .attr('class', 'crude')
      .attr('x', (b) => x(b.x0 ?? 0))
      .attr('y', (b) => y(b.length))
      .attr('width', (b) => Math.max(1, x(b.x1 ?? 0) - x(b.x0 ?? 0) - 1))
      .attr('height', (b) => innerH - y(b.length))
      .attr('fill', 'rgb(220,100,70)')
      .attr('opacity', 0.55);

    g.selectAll('rect.rb')
      .data(rbBins)
      .join('rect')
      .attr('class', 'rb')
      .attr('x', (b) => x(b.x0 ?? 0))
      .attr('y', (b) => y(b.length))
      .attr('width', (b) => Math.max(1, x(b.x1 ?? 0) - x(b.x0 ?? 0) - 1))
      .attr('height', (b) => innerH - y(b.length))
      .attr('fill', 'rgb(70,160,90)')
      .attr('opacity', 0.7);

    // True value line
    g.append('line')
      .attr('x1', x(mc.truth)).attr('x2', x(mc.truth))
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'currentColor').attr('stroke-dasharray', '3,3').attr('opacity', 0.6);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 130},2)`);
    legend.append('rect').attr('width', 12).attr('height', 12).attr('fill', 'rgb(220,100,70)').attr('opacity', 0.55);
    legend.append('text').attr('x', 16).attr('y', 10).attr('font-size', 11).attr('fill', 'currentColor').text('Crude');
    legend.append('rect').attr('y', 16).attr('width', 12).attr('height', 12).attr('fill', 'rgb(70,160,90)').attr('opacity', 0.7);
    legend.append('text').attr('x', 16).attr('y', 26).attr('font-size', 11).attr('fill', 'currentColor').text("RB'd");
  }, [mc, histW]);

  const ratio = mc.rbMSE > 0 ? mc.crudeMSE / mc.rbMSE : Infinity;

  return (
    <div ref={containerRef} className="not-prose my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}>
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Rao-Blackwell Improver · §16.5
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {raoBlackwellPresets.map((p, i) => (
            <option key={i} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>n = {n}</span>
          <input type="range" min={5} max={100} value={n} onChange={(e) => { setN(Number(e.target.value)); setParams((p) => ({ ...p, n: Number(e.target.value) })); }} className="w-full" />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>M = {M}</span>
          <input type="range" min={200} max={5000} step={100} value={M} onChange={(e) => setM(Number(e.target.value))} className="w-full" />
        </label>
        <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          MSE(crude) = {mc.crudeMSE.toFixed(4)}<br />
          MSE(RB'd) = {mc.rbMSE.toFixed(4)}<br />
          <strong>Ratio = {ratio.toFixed(1)}×</strong>
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-run MC
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-4'}>
        <svg ref={histRef} width={histW} height={H} style={{ display: 'block' }} />
        <div className="text-sm space-y-2">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Closed-form Rao-Blackwellization
          </div>
          <div className="font-mono text-xs leading-relaxed p-3 rounded space-y-1" style={{ background: 'var(--color-code-bg, rgba(127,127,127,0.08))' }}>
            <div><span style={{ color: 'rgb(220,100,70)' }}>Crude:</span> {preset.crudeFormula}</div>
            <div><span style={{ color: 'rgb(70,160,90)' }}>RB'd:</span> {preset.rbFormula}</div>
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            True value of estimand: <strong>{mc.truth.toFixed(4)}</strong>
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{preset.description}</div>
        </div>
      </div>
    </div>
  );
}
