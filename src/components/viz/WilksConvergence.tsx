import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  wilksSimulate,
  chiSquaredPDF,
  chiSquaredInvCDF,
} from './shared/testing';
import {
  wilksConvergencePresets,
  type WilksConvergencePreset,
} from '../../data/likelihood-ratio-tests-data';

/**
 * WilksConvergence — the featured Topic 18 component.
 *
 * Visualizes the convergence of −2 log Λₙ to χ²₁ as n grows, under H₀.
 * Left panel: a large histogram for the slider-selected (family, n, M) and
 * the χ²₁ density overlaid. Center panel: empirical-vs-theory moment
 * readouts (mean 1, variance 2, 95th percentile 3.84) with color-coded
 * tolerance indicators. Right panel: 4-panel mini-grid showing the same
 * convergence at n = 10, 50, 200, 1000 simultaneously — the compressed
 * version of figure 18-wilks-convergence.png.
 *
 * Performance: MC at n=1000, M=10000 is 10M samples. `useMemo` caches the
 * main histogram and each mini-panel separately so slider tweaks don't
 * re-run everything. A console warning fires at M > 5000.
 */

const MARGIN = { top: 14, right: 16, bottom: 40, left: 48 };
const H_MAIN = 300;
const H_MINI = 120;
const BINS_MAIN = 40;
const BINS_MINI = 22;

const COLOR_LRT = '#059669';
const COLOR_CHI2 = '#6B7280';
const COLOR_MEAN = '#DC2626';
const COLOR_CRIT = '#D97706';
const MINI_PANEL_NS: readonly [number, number, number, number] = [10, 50, 200, 1000];

interface MomentStats {
  mean: number;
  variance: number;
  q95: number;
}

function computeMoments(samples: number[]): MomentStats {
  const finite = samples.filter((v) => Number.isFinite(v));
  const n = finite.length;
  if (n === 0) return { mean: NaN, variance: NaN, q95: NaN };
  const mean = finite.reduce((a, b) => a + b, 0) / n;
  const variance = finite.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  const sorted = [...finite].sort((a, b) => a - b);
  const q95 = sorted[Math.min(n - 1, Math.floor(0.95 * n))];
  return { mean, variance, q95 };
}

/** Color gauge: green within tol, amber within 2·tol, red otherwise. */
function toleranceColor(got: number, want: number, tol: number): string {
  const rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-6);
  if (rel <= tol) return '#10B981'; // emerald
  if (rel <= 2 * tol) return '#F59E0B'; // amber
  return '#DC2626'; // red
}

/** Bin an array of samples into densities on [0, xMax] with `bins` bins. */
function binDensities(samples: number[], bins: number, xMax: number): number[] {
  const out = new Array(bins).fill(0);
  const step = xMax / bins;
  samples.forEach((v) => {
    if (!Number.isFinite(v) || v < 0 || v >= xMax) return;
    const b = Math.min(bins - 1, Math.floor(v / step));
    out[b]++;
  });
  return out.map((c) => c / (samples.length * step));
}

export default function WilksConvergence() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 900;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: WilksConvergencePreset = wilksConvergencePresets[presetIndex];

  const [theta0, setTheta0] = useState(preset.theta0);
  const [n, setN] = useState(preset.nDefault);
  const [M, setM] = useState(preset.MDefault);
  const [seed, setSeed] = useState(42);

  useEffect(() => {
    setTheta0(preset.theta0);
    setN(preset.nDefault);
    setM(preset.MDefault);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  // Main MC run (slider-controlled n, M, theta0)
  const mainSamples = useMemo(() => {
    const known = preset.family === 'normal-mean-known-sigma' ? (preset.sigma ?? 1) : undefined;
    return wilksSimulate(preset.family, theta0, n, M, known, seed);
  }, [preset, theta0, n, M, seed]);
  const mainMoments = useMemo(() => computeMoments(mainSamples), [mainSamples]);

  // Four fixed-n MC runs for the mini-panel grid. Each caches independently.
  const miniSamples = useMemo(() => {
    const known = preset.family === 'normal-mean-known-sigma' ? (preset.sigma ?? 1) : undefined;
    // Smaller M for mini-panels so they stay snappy
    const miniM = Math.min(M, 2000);
    return MINI_PANEL_NS.map((nVal) =>
      wilksSimulate(preset.family, theta0, nVal, miniM, known, seed + nVal),
    );
    // Intentionally depend on theta0 + seed + preset; not on n (mini panels are fixed)
  }, [preset, theta0, seed, M]);

  const crit = useMemo(() => chiSquaredInvCDF(0.95, 1), []);

  // Main chart
  const mainRef = useRef<SVGSVGElement | null>(null);
  const mainW = isWide ? Math.floor(w * 0.58) : w;

  useEffect(() => {
    const svg = d3.select(mainRef.current);
    svg.selectAll('*').remove();

    const innerW = mainW - MARGIN.left - MARGIN.right;
    const innerH = H_MAIN - MARGIN.top - MARGIN.bottom;

    const finite = mainSamples.filter((v) => Number.isFinite(v));
    if (finite.length === 0) return;

    // Domain: cap at 10 or 99th percentile of the empirical data, whichever is smaller.
    const sorted = [...finite].sort((a, b) => a - b);
    const q99 = sorted[Math.floor(0.99 * sorted.length)] ?? 10;
    const xMax = Math.max(Math.min(q99 * 1.1, 15), 6);
    const x = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);

    const density = binDensities(finite, BINS_MAIN, xMax);
    const step = xMax / BINS_MAIN;

    // χ²₁ reference density
    const chi2Pts: [number, number][] = Array.from({ length: 161 }, (_, i) => {
      const z = (xMax * i) / 160 + 1e-4;
      return [z, chiSquaredPDF(z, 1)];
    });
    const chi2Capped = chi2Pts.filter(([, p]) => Number.isFinite(p) && p < 5);

    const maxY = Math.max(
      ...density,
      ...chi2Capped.map(([, p]) => p),
    ) * 1.15;
    const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${mainW} ${H_MAIN}`)
      .attr('width', mainW)
      .attr('height', H_MAIN)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(x).ticks(7));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')));

    // Histogram bars
    density.forEach((v, i) => {
      g.append('rect')
        .attr('x', x(i * step) + 0.5)
        .attr('y', y(v))
        .attr('width', Math.max(0, innerW / BINS_MAIN - 1))
        .attr('height', innerH - y(v))
        .attr('fill', COLOR_LRT)
        .attr('fill-opacity', 0.7);
    });

    // χ²₁ density curve
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.3));
    g.append('path')
      .datum(chi2Capped)
      .attr('d', line)
      .attr('stroke', COLOR_CHI2)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,3')
      .attr('fill', 'none');

    // Critical value line at χ²₁,.₉₅
    g.append('line')
      .attr('x1', x(crit))
      .attr('x2', x(crit))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', COLOR_CRIT)
      .attr('stroke-dasharray', '3,3');
    g.append('text')
      .attr('x', x(crit))
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .attr('fill', COLOR_CRIT)
      .text(`χ²₁,.₉₅ = ${crit.toFixed(2)}`);

    // Empirical mean line
    if (Number.isFinite(mainMoments.mean)) {
      g.append('line')
        .attr('x1', x(mainMoments.mean))
        .attr('x2', x(mainMoments.mean))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLOR_MEAN)
        .attr('stroke-dasharray', '2,2')
        .attr('stroke-opacity', 0.7);
    }

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 150}, 4)`);
    legend
      .append('rect')
      .attr('width', 10)
      .attr('height', 10)
      .attr('fill', COLOR_LRT)
      .attr('fill-opacity', 0.7);
    legend
      .append('text')
      .attr('x', 14)
      .attr('y', 9)
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .text('−2 log Λₙ (MC)');
    legend
      .append('line')
      .attr('x1', 0)
      .attr('x2', 10)
      .attr('y1', 17)
      .attr('y2', 17)
      .attr('stroke', COLOR_CHI2)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,3');
    legend
      .append('text')
      .attr('x', 14)
      .attr('y', 20)
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .text('χ²₁ density');

    // Axis label
    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H_MAIN - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text(`−2 log Λₙ at n = ${n}, M = ${M}`);
  }, [mainSamples, mainMoments, mainW, crit, n, M]);

  // Mini-panel grid: 4 small charts for n ∈ {10, 50, 200, 1000}
  const miniRef = useRef<SVGSVGElement | null>(null);
  const miniW = isWide ? Math.floor(w * 0.40) : w;

  useEffect(() => {
    const svg = d3.select(miniRef.current);
    svg.selectAll('*').remove();

    const cols = 2;
    const rows = 2;
    const panelW = miniW / cols;
    const panelH = H_MINI;
    const totalH = panelH * rows;

    svg
      .attr('viewBox', `0 0 ${miniW} ${totalH}`)
      .attr('width', miniW)
      .attr('height', totalH);

    const inner = {
      top: 10,
      right: 6,
      bottom: 22,
      left: 30,
    };

    miniSamples.forEach((samples, idx) => {
      const nLabel = MINI_PANEL_NS[idx];
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      const px = col * panelW;
      const py = row * panelH;
      const innerW = panelW - inner.left - inner.right;
      const innerH = panelH - inner.top - inner.bottom;

      const g = svg.append('g').attr('transform', `translate(${px + inner.left}, ${py + inner.top})`);

      const finite = samples.filter((v) => Number.isFinite(v));
      if (finite.length === 0) return;

      const xMax = 10;
      const x = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);

      const density = binDensities(finite, BINS_MINI, xMax);
      const step = xMax / BINS_MINI;

      const chi2Pts: [number, number][] = Array.from({ length: 80 }, (_, i) => {
        const z = (xMax * i) / 79 + 1e-4;
        return [z, chiSquaredPDF(z, 1)];
      });
      const chi2Capped = chi2Pts.filter(([, p]) => Number.isFinite(p) && p < 3);
      const maxY = Math.max(...density, ...chi2Capped.map(([, p]) => p)) * 1.1;
      const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);

      // Light axes
      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(x).ticks(4).tickSize(3))
        .style('font-size', '9px');
      g.append('g')
        .call(d3.axisLeft(y).ticks(3).tickSize(3).tickFormat(d3.format('.1f')))
        .style('font-size', '9px');

      // Bars
      density.forEach((v, i) => {
        g.append('rect')
          .attr('x', x(i * step))
          .attr('y', y(v))
          .attr('width', Math.max(0, innerW / BINS_MINI - 0.5))
          .attr('height', innerH - y(v))
          .attr('fill', COLOR_LRT)
          .attr('fill-opacity', 0.6);
      });

      // χ²₁ density
      const line = d3
        .line<[number, number]>()
        .x((d) => x(d[0]))
        .y((d) => y(d[1]));
      g.append('path')
        .datum(chi2Capped)
        .attr('d', line)
        .attr('stroke', COLOR_CHI2)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '3,2')
        .attr('fill', 'none');

      // Label
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', 10)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('font-weight', 'bold')
        .attr('fill', 'currentColor')
        .text(`n = ${nLabel}`);
    });
  }, [miniSamples, miniW]);

  const MSlow = M > 5000;

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div
          className="text-xs uppercase tracking-wide font-bold"
          style={{ color: 'var(--color-text-muted)' }}
        >
          Wilks Convergence · §18.6
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{
            background: 'var(--color-input-bg)',
            borderColor: 'var(--color-border)',
            color: 'var(--color-text)',
          }}
        >
          {wilksConvergencePresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            θ₀ = <strong>{theta0.toFixed(preset.family === 'bernoulli' ? 2 : 1)}</strong>
          </span>
          <input
            type="range"
            min={preset.family === 'bernoulli' ? 0.05 : -2}
            max={preset.family === 'bernoulli' ? 0.95 : 2}
            step={preset.family === 'bernoulli' ? 0.01 : 0.05}
            value={theta0}
            onChange={(e) => setTheta0(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
          </span>
          <input
            type="range"
            min={preset.nRange[0]}
            max={preset.nRange[1]}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            M = <strong>{M}</strong>
            {MSlow && <span className="ml-1 text-[10px]" style={{ color: '#F59E0B' }}>(slow)</span>}
          </span>
          <input
            type="range"
            min={preset.MRange[0]}
            max={preset.MRange[1]}
            step={500}
            value={M}
            onChange={(e) => setM(Number(e.target.value))}
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="self-end px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-sample
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-[58%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <div className="flex flex-col gap-2">
          <svg ref={mainRef} style={{ display: 'block' }} />
          <div
            className="text-xs p-2 rounded flex flex-wrap gap-x-4 gap-y-1"
            style={{ background: 'var(--color-input-bg)' }}
          >
            <div>
              <span style={{ color: 'var(--color-text-muted)' }}>mean:</span>{' '}
              <strong style={{ color: toleranceColor(mainMoments.mean, 1, 0.05) }}>
                {mainMoments.mean.toFixed(3)}
              </strong>{' '}
              <span style={{ color: 'var(--color-text-muted)' }}>(χ²₁: 1)</span>
            </div>
            <div>
              <span style={{ color: 'var(--color-text-muted)' }}>variance:</span>{' '}
              <strong style={{ color: toleranceColor(mainMoments.variance, 2, 0.1) }}>
                {mainMoments.variance.toFixed(3)}
              </strong>{' '}
              <span style={{ color: 'var(--color-text-muted)' }}>(χ²₁: 2)</span>
            </div>
            <div>
              <span style={{ color: 'var(--color-text-muted)' }}>95th %ile:</span>{' '}
              <strong style={{ color: toleranceColor(mainMoments.q95, 3.84, 0.05) }}>
                {mainMoments.q95.toFixed(2)}
              </strong>{' '}
              <span style={{ color: 'var(--color-text-muted)' }}>(χ²₁: 3.84)</span>
            </div>
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <div
            className="text-xs uppercase font-bold tracking-wide"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Convergence panels — n = 10, 50, 200, 1000
          </div>
          <svg ref={miniRef} style={{ display: 'block' }} />
          <div className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>
            Each mini-panel uses the same θ₀ and seed as the main chart, with
            M = min(main M, 2000) for performance.
          </div>
        </div>
      </div>
    </div>
  );
}
