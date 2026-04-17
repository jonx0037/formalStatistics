import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  huberPsi,
  huberPsiPrime,
  tukeyPsi,
  tukeyPsiPrime,
  mEstimatorLocation,
  mEstimatorVariance,
  sampleMean,
  sampleMedian,
  sampleVariance,
} from './shared/estimation';
import { normalSample, sampleSequence } from './shared/convergence';
import { robustContaminationPresets, tuningConstants } from '../../data/method-of-moments-data';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 48 };
const H = 220;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function generateContaminated(
  base: { mu: number; sigma: number },
  contamination: { fraction: number; magnitude: number },
  n: number,
  rng: () => number,
): number[] {
  const out: number[] = [];
  const nContaminated = Math.round(n * contamination.fraction);
  const contamMean = base.mu + contamination.magnitude * base.sigma;
  for (let i = 0; i < n; i++) {
    if (i < nContaminated) {
      // outlier
      const z = sampleSequence(() => normalSample(0, 1, rng), 1)[0];
      out.push(contamMean + z);
    } else {
      const z = sampleSequence(() => normalSample(0, 1, rng), 1)[0];
      out.push(base.mu + base.sigma * z);
    }
  }
  return out;
}

export default function MEstimatorGallery() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = robustContaminationPresets[presetIndex];

  const [k, setK] = useState(tuningConstants.huber.default);
  const [c, setC] = useState(tuningConstants.tukey.default);
  const [contamFrac, setContamFrac] = useState(preset.contamination.fraction);
  const [contamMag, setContamMag] = useState(preset.contamination.magnitude);
  const [n, setN] = useState(200);
  const [seed, setSeed] = useState(1);

  // When preset changes, reset to its contamination defaults.
  useEffect(() => {
    setContamFrac(preset.contamination.fraction);
    setContamMag(preset.contamination.magnitude);
    setSeed((s) => s + 1);
  }, [presetIndex]);

  const data = useMemo(() => {
    const rng = makeLCG(seed * 31 + n + Math.floor(contamFrac * 100) + Math.floor(contamMag * 10));
    return generateContaminated(preset.base, { fraction: contamFrac, magnitude: contamMag }, n, rng);
  }, [seed, n, contamFrac, contamMag, preset.base]);

  const estimates = useMemo(() => {
    const meanEst = sampleMean(data);
    const medianEst = sampleMedian(data);
    const huberEst = mEstimatorLocation(data, (u) => huberPsi(u, k), { init: medianEst });
    const tukeyEst = mEstimatorLocation(data, (u) => tukeyPsi(u, c), { init: medianEst });
    // Sandwich SE for each (using MAD scale internally consistent with mEstimatorLocation):
    const mad = (() => {
      const med = sampleMedian(data);
      return Math.max(sampleMedian(data.map((x) => Math.abs(x - med))) / 0.6745, 1e-9);
    })();
    const meanSE = Math.sqrt(sampleVariance(data, 1) / data.length);
    const medianSE = mad / Math.sqrt(data.length); // approximate; exact uses density at median
    const huberSE = Math.sqrt(Math.max(0, mEstimatorVariance(
      data, huberEst.estimate,
      (u) => huberPsi(u, k),
      (u) => huberPsiPrime(u, k),
      mad,
    )));
    const tukeySE = Math.sqrt(Math.max(0, mEstimatorVariance(
      data, tukeyEst.estimate,
      (u) => tukeyPsi(u, c),
      (u) => tukeyPsiPrime(u, c),
      mad,
    )));
    return [
      { name: 'Sample mean', value: meanEst, se: meanSE, color: '#DC2626', breakdown: '0%' },
      { name: 'Sample median', value: medianEst, se: medianSE, color: '#7C3AED', breakdown: '50%' },
      { name: 'Huber', value: huberEst.estimate, se: huberSE, color: '#2563EB', breakdown: '~5%' },
      { name: 'Tukey', value: tukeyEst.estimate, se: tukeySE, color: '#059669', breakdown: '~50%' },
    ];
  }, [data, k, c]);

  // ── Left panel: ψ-function gallery ───────────────────────────────────────
  const psiRef = useRef<SVGSVGElement | null>(null);
  const psiW = isWide ? Math.floor(w * 0.42) : w;

  useEffect(() => {
    const svg = d3.select(psiRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = psiW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${psiW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const xMax = Math.max(c * 1.2, 6);
    const x = d3.scaleLinear().domain([-xMax, xMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([-xMax * 1.05, xMax * 1.05]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH / 2})`).call(d3.axisBottom(x).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').attr('transform', `translate(${innerW / 2}, 0)`).call(d3.axisLeft(y).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));

    const G = 201;
    const grid: number[] = [];
    for (let i = 0; i < G; i++) grid.push(-xMax + (2 * xMax * i) / (G - 1));

    const series = [
      { label: `MLE score (u)`, color: '#DC2626', vals: grid.map((u) => u) },
      { label: `Huber (k = ${k.toFixed(2)})`, color: '#2563EB', vals: grid.map((u) => huberPsi(u, k)) },
      { label: `Tukey (c = ${c.toFixed(2)})`, color: '#059669', vals: grid.map((u) => tukeyPsi(u, c)) },
      { label: 'sign (median)', color: '#7C3AED', vals: grid.map((u) => Math.sign(u)) },
    ];

    const line = d3.line<number>().x((_, i) => x(grid[i])).y((d) => y(d));

    series.forEach((s) => {
      g.append('path').datum(s.vals).attr('fill', 'none').attr('stroke', s.color)
        .attr('stroke-width', 1.6).attr('d', line);
    });

    const lg = g.append('g').attr('transform', `translate(8, 6)`)
      .style('font-size', '10px').attr('fill', 'var(--color-text)');
    series.forEach((s, i) => {
      lg.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 13).attr('y2', i * 13)
        .attr('stroke', s.color).attr('stroke-width', 2);
      lg.append('text').attr('x', 18).attr('y', i * 13 + 3).text(s.label);
    });
  }, [psiW, k, c]);

  // ── Center panel: histogram of contaminated sample ───────────────────────
  const histRef = useRef<SVGSVGElement | null>(null);
  const histW = isWide ? Math.floor(w * 0.30) : w;

  useEffect(() => {
    const svg = d3.select(histRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = histW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${histW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const lo = (d3.min(data) ?? -2);
    const hi = (d3.max(data) ?? 2);
    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
    const bins = d3.bin().domain([lo, hi]).thresholds(30)(data);
    const yMax = (d3.max(bins, (b) => b.length) ?? 0) * 1.05;
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));

    g.selectAll('rect').data(bins).enter().append('rect')
      .attr('x', (b) => x(b.x0!)).attr('y', (b) => y(b.length))
      .attr('width', (b) => Math.max(0, x(b.x1!) - x(b.x0!) - 1))
      .attr('height', (b) => innerH - y(b.length))
      .attr('fill', 'var(--color-text-muted)').attr('opacity', 0.45);

    g.append('line').attr('x1', x(preset.base.mu)).attr('x2', x(preset.base.mu))
      .attr('y1', 0).attr('y2', innerH).attr('stroke', '#111')
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
  }, [histW, data, preset.base.mu]);

  // ── Right panel: estimator bars with 95% CIs ─────────────────────────────
  const barsRef = useRef<SVGSVGElement | null>(null);
  const barsW = isWide ? w - psiW - histW - 24 : w;

  useEffect(() => {
    const svg = d3.select(barsRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = barsW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${barsW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const trueMu = preset.base.mu;
    const allVals = estimates.flatMap((e) => [e.value - 1.96 * e.se, e.value + 1.96 * e.se]);
    const lo = Math.min(trueMu - 0.5, ...allVals);
    const hi = Math.max(trueMu + 0.5, ...allVals);
    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
    const y = d3.scaleBand().domain(estimates.map((e) => e.name)).range([0, innerH]).padding(0.3);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));

    g.append('line').attr('x1', x(trueMu)).attr('x2', x(trueMu))
      .attr('y1', 0).attr('y2', innerH).attr('stroke', '#111')
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');

    estimates.forEach((est) => {
      const yc = (y(est.name) ?? 0) + y.bandwidth() / 2;
      // CI bar
      g.append('line').attr('x1', x(est.value - 1.96 * est.se)).attr('x2', x(est.value + 1.96 * est.se))
        .attr('y1', yc).attr('y2', yc).attr('stroke', est.color).attr('stroke-width', 2);
      // Caps
      [est.value - 1.96 * est.se, est.value + 1.96 * est.se].forEach((v) => {
        g.append('line').attr('x1', x(v)).attr('x2', x(v))
          .attr('y1', yc - 4).attr('y2', yc + 4).attr('stroke', est.color).attr('stroke-width', 2);
      });
      // Point
      g.append('circle').attr('cx', x(est.value)).attr('cy', yc).attr('r', 4).attr('fill', est.color);
    });
  }, [estimates, barsW, preset.base.mu]);

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: M-estimator gallery — ψ-functions, contamination, sandwich CIs</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          The shape of ψ determines the estimator. Contaminate the data and see how each estimator responds: the mean drifts, the median holds, Huber holds-then-drifts past its breakdown, Tukey holds further still.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center" style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}>
        <select value={presetIndex} onChange={(e) => setPresetIndex(Number(e.target.value))}
          style={{ padding: '0.2rem' }}>
          {robustContaminationPresets.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
        </select>
        <label>k = {k.toFixed(2)}{' '}
          <input type="range" min={0.5} max={3} step={0.05}
            value={k} onChange={(e) => setK(Number(e.target.value))}
            style={{ width: 80, verticalAlign: 'middle' }} />
        </label>
        <label>c = {c.toFixed(2)}{' '}
          <input type="range" min={2} max={8} step={0.05}
            value={c} onChange={(e) => setC(Number(e.target.value))}
            style={{ width: 80, verticalAlign: 'middle' }} />
        </label>
        <label>ε = {(contamFrac * 100).toFixed(0)}%{' '}
          <input type="range" min={0} max={0.4} step={0.02}
            value={contamFrac} onChange={(e) => { setContamFrac(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 90, verticalAlign: 'middle' }} />
        </label>
        <label>magnitude = {contamMag.toFixed(0)}σ{' '}
          <input type="range" min={3} max={20} step={1}
            value={contamMag} onChange={(e) => { setContamMag(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 90, verticalAlign: 'middle' }} />
        </label>
        <label>n = {n}{' '}
          <input type="range" min={30} max={500} step={10}
            value={n} onChange={(e) => { setN(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 90, verticalAlign: 'middle' }} />
        </label>
        <button onClick={() => setSeed((s) => s + 1)}
          style={{ padding: '0.25rem 0.65rem', fontSize: '0.8125rem',
            border: '1px solid var(--color-border)', borderRadius: '0.375rem',
            background: 'transparent', cursor: 'pointer' }}>
          New sample
        </button>
      </div>

      <div style={{ display: isWide ? 'flex' : 'block', gap: 12 }}>
        <svg ref={psiRef} width={psiW} height={H} style={{ display: 'block' }} />
        <svg ref={histRef} width={histW} height={H} style={{ display: 'block' }} />
        <svg ref={barsRef} width={barsW} height={H} style={{ display: 'block' }} />
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 8, lineHeight: 1.5 }}>
        Estimates (true μ = {preset.base.mu}):{' '}
        {estimates.map((e, i) => (
          <span key={e.name} style={{ marginRight: 8, color: e.color }}>
            <strong>{e.name}:</strong> {e.value.toFixed(3)} ± {(1.96 * e.se).toFixed(3)} <span style={{ color: 'var(--color-text-muted)' }}>(breakdown {e.breakdown})</span>{i < estimates.length - 1 ? '·' : ''}
          </span>
        ))}
      </div>
    </div>
  );
}
