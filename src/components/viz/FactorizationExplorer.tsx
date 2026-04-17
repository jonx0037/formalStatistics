import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  factorizationForm,
  computeSufficientStatistic,
  type SufficientFamily,
} from './shared/estimation';
import {
  normalSample,
  bernoulliSample,
  poissonSample,
  exponentialSample,
  gammaSample,
  uniformSampleArray,
  sampleSequence,
} from './shared/convergence';
import { factorizationPresets } from '../../data/sufficient-statistics-data';

const MARGIN = { top: 14, right: 12, bottom: 32, left: 38 };
const H = 200;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function drawSample(family: SufficientFamily, params: Record<string, number>, n: number, rng: () => number): number[] {
  switch (family) {
    case 'normal-mu':
    case 'normal-mu-sigma':
      return sampleSequence(() => normalSample(params.mu ?? 0, Math.sqrt(params.sigma2 ?? 1), rng), n);
    case 'bernoulli':
      return sampleSequence(() => bernoulliSample(params.p ?? 0.5, rng), n);
    case 'poisson':
      return sampleSequence(() => poissonSample(params.lambda ?? 1, rng), n);
    case 'exponential':
      return sampleSequence(() => exponentialSample(params.lambda ?? 1, rng), n);
    case 'gamma-scale':
      return sampleSequence(() => gammaSample(params.alpha ?? 2, params.beta ?? 1, rng), n);
    case 'uniform-upper':
      return uniformSampleArray(n, 0, params.theta ?? 1, rng);
  }
}

function paramSliderConfig(family: SufficientFamily): { name: string; key: string; min: number; max: number; step: number } | null {
  switch (family) {
    case 'normal-mu':
    case 'normal-mu-sigma':
      return { name: 'μ', key: 'mu', min: -3, max: 3, step: 0.1 };
    case 'bernoulli':
      return { name: 'p', key: 'p', min: 0.05, max: 0.95, step: 0.05 };
    case 'poisson':
      return { name: 'λ', key: 'lambda', min: 0.5, max: 10, step: 0.5 };
    case 'exponential':
      return { name: 'λ', key: 'lambda', min: 0.2, max: 5, step: 0.1 };
    case 'gamma-scale':
      return { name: 'β', key: 'beta', min: 0.5, max: 5, step: 0.1 };
    case 'uniform-upper':
      return { name: 'θ', key: 'theta', min: 0.5, max: 10, step: 0.5 };
  }
}

export default function FactorizationExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = factorizationPresets[presetIndex];

  const [params, setParams] = useState<Record<string, number>>({ ...preset.defaultParams });
  const [n, setN] = useState(20);
  const [seed, setSeed] = useState(1);

  // Reset on preset change.
  useEffect(() => {
    const p = factorizationPresets[presetIndex];
    setParams({ ...p.defaultParams });
    setSeed((s) => s + 1);
  }, [presetIndex]);

  const sample = useMemo(() => {
    const rng = makeLCG(seed * 9973 + n + Math.floor((params.mu ?? params.p ?? params.lambda ?? params.theta ?? 1) * 1000));
    return drawSample(preset.family, params, n, rng);
  }, [preset.family, params, n, seed]);

  const tValue = useMemo(() => computeSufficientStatistic(preset.family, sample), [preset.family, sample]);
  const factor = useMemo(() => factorizationForm(preset.family), [preset.family]);
  const sliderCfg = paramSliderConfig(preset.family);

  // ── Histogram of the current sample ───────────────────────────────────────
  const histRef = useRef<SVGSVGElement | null>(null);
  const histW = isWide ? Math.floor(w * 0.5) : w;

  useEffect(() => {
    if (!histRef.current || sample.length === 0) return;
    const svg = d3.select(histRef.current);
    svg.selectAll('*').remove();

    const innerW = histW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const isDiscrete = preset.family === 'bernoulli' || preset.family === 'poisson';
    const ext = d3.extent(sample) as [number, number];
    const xMin = isDiscrete ? Math.max(0, Math.floor(ext[0])) : ext[0] - (ext[1] - ext[0]) * 0.05;
    const xMax = isDiscrete ? Math.ceil(ext[1]) + 0.5 : ext[1] + (ext[1] - ext[0]) * 0.05;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

    let bins: { x0: number; x1: number; count: number }[];
    if (isDiscrete) {
      const counts = new Map<number, number>();
      for (const v of sample) counts.set(v, (counts.get(v) ?? 0) + 1);
      bins = [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([k, c]) => ({ x0: k - 0.4, x1: k + 0.4, count: c }));
    } else {
      const histGen = d3.bin<number, number>().domain([xMin, xMax]).thresholds(20);
      const out = histGen(sample);
      bins = out.map((b) => ({ x0: b.x0 ?? 0, x1: b.x1 ?? 0, count: b.length }));
    }

    const yMax = d3.max(bins, (b) => b.count) ?? 1;
    const y = d3.scaleLinear().domain([0, yMax]).nice().range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).style('font-size', '10px');

    g.selectAll('rect.bin')
      .data(bins)
      .join('rect')
      .attr('class', 'bin')
      .attr('x', (b) => x(b.x0))
      .attr('y', (b) => y(b.count))
      .attr('width', (b) => Math.max(1, x(b.x1) - x(b.x0) - 1))
      .attr('height', (b) => innerH - y(b.count))
      .attr('fill', preset.isExpFamily ? 'rgb(70, 130, 180)' : 'rgb(220, 100, 70)')
      .attr('opacity', 0.75);

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -2)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('fill', 'currentColor')
      .text(`Sample (n = ${n})`);
  }, [sample, histW, preset.family, preset.isExpFamily, n]);

  function updateParam(key: string, val: number) {
    setParams((p) => ({ ...p, [key]: val }));
    setSeed((s) => s + 1);
  }

  const tDisplay = Array.isArray(tValue)
    ? `(${tValue.map((v) => v.toFixed(2)).join(', ')})`
    : tValue.toFixed(2);

  return (
    <div ref={containerRef} className="not-prose my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}>
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Factorization Explorer · §16.3
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {factorizationPresets.map((p, i) => (
            <option key={i} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        {sliderCfg && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>{sliderCfg.name} = {(params[sliderCfg.key] ?? 0).toFixed(2)}</span>
            <input
              type="range"
              min={sliderCfg.min} max={sliderCfg.max} step={sliderCfg.step}
              value={params[sliderCfg.key] ?? 0}
              onChange={(e) => updateParam(sliderCfg.key, Number(e.target.value))}
              className="w-full"
            />
          </label>
        )}
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>n = {n}</span>
          <input type="range" min={5} max={100} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-full" />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Draw new sample
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-4'}>
        <div>
          <svg ref={histRef} width={histW} height={H} style={{ display: 'block' }} />
          <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
            <span className="font-mono">T(X) = {tDisplay}</span>
          </div>
        </div>
        <div className="text-sm space-y-2">
          <div>
            <span className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>Factorization</span>
          </div>
          <div className="font-mono text-xs leading-relaxed p-3 rounded" style={{ background: 'var(--color-code-bg, rgba(127,127,127,0.08))' }}>
            <div>f(x; θ) = g(T(x); θ) · h(x)</div>
            <div className="mt-2"><span style={{ color: 'rgb(70,130,180)' }}>T:</span> {factor.tLatex}</div>
            <div><span style={{ color: 'rgb(40,160,80)' }}>g:</span> {factor.gLatex}</div>
            <div><span style={{ color: 'rgb(180,100,40)' }}>h:</span> {factor.hLatex}</div>
          </div>
          <div className="text-xs" style={{ color: preset.isExpFamily ? 'rgb(40,160,80)' : 'rgb(220,100,70)' }}>
            {preset.isExpFamily ? '✓ Exponential family' : '✗ NOT an exponential family'}
          </div>
          {factor.supportNote && (
            <div className="text-xs italic p-2 rounded" style={{ background: 'rgba(220,100,70,0.1)', color: 'rgb(180,80,50)' }}>
              {factor.supportNote}
            </div>
          )}
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{preset.description}</div>
        </div>
      </div>
    </div>
  );
}
