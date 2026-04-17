import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { basuIndependenceNormal } from './shared/estimation';
import {
  studentTPDF,
  standardNormalPDF,
  tTestStatistic,
} from './shared/testing';
import { normalSample } from './shared/convergence';
import { seededRandom } from './shared/probability';
import {
  tTestBasuPresets,
  type TTestBasuPreset,
} from '../../data/hypothesis-testing-data';

const MARGIN = { top: 14, right: 16, bottom: 36, left: 48 };
const SCATTER_H = 280;
const HIST_H = 260;
const HIST_BINS = 40;

export default function TTestBasuFoundation() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 900;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: TTestBasuPreset = tTestBasuPresets[presetIndex];

  const [n, setN] = useState(preset.defaults.n);
  const [muTrue, setMuTrue] = useState(preset.defaults.muTrue);
  const [sigma, setSigma] = useState(preset.defaults.sigma);
  const [M, setM] = useState(preset.defaults.M);
  const [showNormal, setShowNormal] = useState(true);
  const [seed, setSeed] = useState(11);

  useEffect(() => {
    setN(preset.defaults.n);
    setMuTrue(preset.defaults.muTrue);
    setSigma(preset.defaults.sigma);
    setM(preset.defaults.M);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  // Basu scatter data — (X̄, S²) across M replications.
  const basu = useMemo(() => {
    const rng = seededRandom(seed * 7919 + n + M);
    return basuIndependenceNormal(muTrue, sigma * sigma, n, M, rng);
  }, [muTrue, sigma, n, M, seed]);

  // T-statistic histogram: T_i = √n (X̄_i - mu0) / S_i with μ₀ = preset.defaults.mu0
  const mu0 = preset.defaults.mu0;
  const tValues = useMemo(() => {
    const rng = seededRandom(seed * 104729 + n + M);
    const arr: number[] = [];
    for (let i = 0; i < M; i++) {
      const data = Array.from({ length: n }, () => normalSample(muTrue, sigma, rng));
      arr.push(tTestStatistic(data, mu0));
    }
    return arr;
  }, [muTrue, sigma, n, M, mu0, seed]);

  // Scatter rendered via Canvas for perf at M > 1000.
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasW = isWide ? Math.floor(w * 0.42) : w;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = SCATTER_H * dpr;
    canvas.style.width = `${canvasW}px`;
    canvas.style.height = `${SCATTER_H}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvasW, SCATTER_H);

    const innerW = canvasW - MARGIN.left - MARGIN.right;
    const innerH = SCATTER_H - MARGIN.top - MARGIN.bottom;
    if (basu.xbar.length === 0) return;

    let xMin = basu.xbar[0],
      xMax = basu.xbar[0];
    for (let i = 1; i < basu.xbar.length; i++) {
      if (basu.xbar[i] < xMin) xMin = basu.xbar[i];
      if (basu.xbar[i] > xMax) xMax = basu.xbar[i];
    }
    let yMin = basu.s2[0],
      yMax = basu.s2[0];
    for (let i = 1; i < basu.s2.length; i++) {
      if (basu.s2[i] < yMin) yMin = basu.s2[i];
      if (basu.s2[i] > yMax) yMax = basu.s2[i];
    }
    const xPad = (xMax - xMin) * 0.05 || 0.1;
    const yPad = (yMax - yMin) * 0.05 || 0.1;

    const sx = (v: number) => MARGIN.left + (innerW * (v - (xMin - xPad))) / (xMax + xPad - (xMin - xPad));
    const sy = (v: number) => MARGIN.top + innerH - (innerH * (v - (yMin - yPad))) / (yMax + yPad - (yMin - yPad));

    // Axes
    ctx.strokeStyle = 'currentColor';
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + innerH);
    ctx.lineTo(MARGIN.left + innerW, MARGIN.top + innerH);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Ticks
    ctx.fillStyle = 'currentColor';
    ctx.font = '10px sans-serif';
    ctx.globalAlpha = 0.75;
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const tx = xMin - xPad + (i / 5) * (xMax + xPad - (xMin - xPad));
      ctx.fillText(tx.toFixed(2), sx(tx), MARGIN.top + innerH + 14);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const ty = yMin - yPad + (i / 5) * (yMax + yPad - (yMin - yPad));
      ctx.fillText(ty.toFixed(2), MARGIN.left - 4, sy(ty) + 3);
    }
    ctx.globalAlpha = 1;

    // Points
    ctx.fillStyle = 'rgba(5,150,105,0.35)';
    for (let i = 0; i < basu.xbar.length; i++) {
      const px = sx(basu.xbar[i]);
      const py = sy(basu.s2[i]);
      ctx.fillRect(px - 1, py - 1, 2, 2);
    }

    // Axis labels
    ctx.fillStyle = 'currentColor';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('X̄ (complete sufficient for μ)', MARGIN.left + innerW / 2, SCATTER_H - 6);
    ctx.save();
    ctx.translate(10, MARGIN.top + innerH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('S² (ancillary for μ)', 0, 0);
    ctx.restore();
  }, [basu, canvasW]);

  // T-histogram SVG
  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.42) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = HIST_H - MARGIN.top - MARGIN.bottom;

    // Histogram bins
    const tMin = Math.min(-5, d3.min(tValues) ?? -5);
    const tMax = Math.max(5, d3.max(tValues) ?? 5);
    const step = (tMax - tMin) / HIST_BINS;
    const counts = new Array(HIST_BINS).fill(0);
    tValues.forEach((t) => {
      const b = Math.min(HIST_BINS - 1, Math.max(0, Math.floor((t - tMin) / step)));
      counts[b]++;
    });
    // Convert counts to densities (counts / M / step)
    const density = counts.map((c) => c / (tValues.length * step));
    const maxD = Math.max(...density, 0.5) * 1.15;

    const x = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, maxD]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${HIST_H}`)
      .attr('width', svgW)
      .attr('height', HIST_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Axes
    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(7));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')));

    // Bars
    density.forEach((v, i) => {
      g.append('rect')
        .attr('x', x(tMin + i * step))
        .attr('y', y(v))
        .attr('width', Math.max(0, innerW / HIST_BINS - 1))
        .attr('height', innerH - y(v))
        .attr('fill', '#059669')
        .attr('fill-opacity', 0.55);
    });

    // Exact t_{n-1} density overlay
    const samples = 201;
    const tPts: [number, number][] = Array.from({ length: samples }, (_, i) => {
      const z = tMin + (i * (tMax - tMin)) / (samples - 1);
      return [z, studentTPDF(z, n - 1)];
    });
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.3));
    g.append('path').datum(tPts).attr('d', line).attr('stroke', '#DC2626').attr('stroke-width', 2).attr('fill', 'none');

    if (showNormal) {
      const zPts: [number, number][] = Array.from({ length: samples }, (_, i) => {
        const z = tMin + (i * (tMax - tMin)) / (samples - 1);
        return [z, standardNormalPDF(z)];
      });
      g.append('path')
        .datum(zPts)
        .attr('d', line)
        .attr('stroke', '#6B7280')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3')
        .attr('fill', 'none');
    }

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 100}, 4)`);
    legend.append('rect').attr('x', 0).attr('y', 0).attr('width', 10).attr('height', 10).attr('fill', '#059669').attr('fill-opacity', 0.6);
    legend.append('text').attr('x', 14).attr('y', 9).style('font-size', '10px').attr('fill', 'currentColor').text(`MC (M=${M})`);
    legend.append('line').attr('x1', 0).attr('x2', 10).attr('y1', 20).attr('y2', 20).attr('stroke', '#DC2626').attr('stroke-width', 2);
    legend.append('text').attr('x', 14).attr('y', 23).style('font-size', '10px').attr('fill', 'currentColor').text(`t_{n−1} exact`);
    if (showNormal) {
      legend.append('line').attr('x1', 0).attr('x2', 10).attr('y1', 34).attr('y2', 34).attr('stroke', '#6B7280').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3');
      legend.append('text').attr('x', 14).attr('y', 37).style('font-size', '10px').attr('fill', 'currentColor').text('N(0, 1)');
    }

    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', HIST_H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('t-statistic T = √n(X̄ − μ₀) / S');
  }, [tValues, svgW, n, showNormal, M]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{
        borderColor: 'var(--color-border)',
        background: 'var(--color-card-bg)',
        borderLeftWidth: '4px',
        borderLeftColor: '#DC2626',
      }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: '#B91C1C' }}>
          ★ Featured · t-test via Basu · §17.7
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {tTestBasuPresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
          </span>
          <input type="range" min={4} max={100} value={n} onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            true μ = <strong>{muTrue.toFixed(2)}</strong>
          </span>
          <input type="range" min={-2} max={2} step={0.05} value={muTrue} onChange={(e) => setMuTrue(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            σ = <strong>{sigma.toFixed(2)}</strong>
          </span>
          <input type="range" min={0.5} max={3} step={0.05} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            M = <strong>{M}</strong>
          </span>
          <input type="range" min={200} max={10000} step={100} value={M} onChange={(e) => setM(Number(e.target.value))} />
        </label>
        <div className="flex flex-col gap-1">
          <label className="flex items-center gap-1 text-xs">
            <input type="checkbox" checked={showNormal} onChange={(e) => setShowNormal(e.target.checked)} />
            overlay N(0, 1)
          </label>
          <button
            onClick={() => setSeed((s) => s + 1)}
            className="px-3 py-1 rounded text-xs font-semibold"
            style={{ background: 'var(--color-accent)', color: 'white' }}
          >
            Resample
          </button>
        </div>
      </div>

      <div className={isWide ? 'grid grid-cols-3 gap-4' : 'flex flex-col gap-4'}>
        <div>
          <div className="text-[11px] font-bold mb-1" style={{ color: 'var(--color-text-muted)' }}>
            Basu: (X̄, S²) scatter — decorrelated
          </div>
          <canvas ref={canvasRef} style={{ display: 'block' }} />
          <div className="text-[11px] mt-1" style={{ color: 'var(--color-text-muted)' }}>
            sample ρ = <strong>{basu.correlation.toFixed(4)}</strong> (theoretical: 0)
          </div>
        </div>
        <div>
          <div className="text-[11px] font-bold mb-1" style={{ color: 'var(--color-text-muted)' }}>
            T = √n(X̄ − μ₀)/S — matches t_{n - 1}
          </div>
          <svg ref={svgRef} style={{ display: 'block' }} />
        </div>
        <div
          className="text-sm space-y-3 p-3 rounded"
          style={{
            background: 'rgba(220,38,38,0.05)',
            border: '1px dashed rgba(220,38,38,0.3)',
          }}
        >
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: '#B91C1C' }}>
            Why this works
          </div>
          <p className="text-xs leading-relaxed">
            <a
              href={preset.basuCallback}
              className="font-semibold underline"
              style={{ color: '#B91C1C' }}
            >
              Basu's theorem (Topic 16 §16.9)
            </a>{' '}
            gives{' '}
            <strong className="font-mono">X̄ ⊥⊥ S²</strong>. This independence is what makes the t-ratio have a clean distribution:
          </p>
          <ul className="text-xs space-y-1 list-disc pl-4">
            <li>Numerator <span className="font-mono">√n(X̄ − μ₀)/σ ~ N(0, 1)</span></li>
            <li>Denominator <span className="font-mono">S/σ = √(χ²_{n - 1}/(n − 1))</span></li>
            <li><em>And they are independent.</em></li>
          </ul>
          <p className="text-xs leading-relaxed">
            That's exactly the defining construction of Student's <span className="font-mono">t_{n - 1}</span>. Without Basu's independence, the ratio has a messy joint distribution that depends on (μ, σ²).
          </p>
          <a
            href={preset.basuCallback}
            className="inline-block mt-2 px-3 py-1.5 text-xs font-semibold rounded"
            style={{ background: '#B91C1C', color: 'white', textDecoration: 'none' }}
          >
            → Jump to §16.9
          </a>
          <div className="text-[11px] italic leading-relaxed pt-2" style={{ color: 'var(--color-text-muted)' }}>
            {preset.description}
          </div>
        </div>
      </div>
    </div>
  );
}
