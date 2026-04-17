import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  momGamma,
  mleGammaShape,
  digamma,
  trigamma,
  logGamma,
  sampleMean,
} from './shared/estimation';
import { gammaSample, sampleSequence } from './shared/convergence';
import { warmStartPresets } from '../../data/method-of-moments-data';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 48 };
const H = 240;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

// Run Newton-Raphson on the Gamma shape score from a given starting α₀.
// Records the full path so we can visualize all three trajectories.
function newtonPath(data: number[], beta: number, init: number, maxIter = 30, tol = 1e-9):
  { path: number[]; converged: boolean } {
  const n = data.length;
  let sumLog = 0;
  for (let i = 0; i < n; i++) sumLog += Math.log(Math.max(data[i], 1e-300));
  const path: number[] = [init];
  let alpha = init;
  let converged = false;
  for (let k = 0; k < maxIter; k++) {
    const score = n * Math.log(beta) - n * digamma(alpha) + sumLog;
    const info = n * trigamma(alpha);
    if (!Number.isFinite(info) || info <= 0) break;
    const step = score / info;
    const next = Math.max(alpha + step, 1e-6);
    path.push(next);
    if (Math.abs(step) < tol) {
      converged = true;
      alpha = next;
      break;
    }
    alpha = next;
  }
  return { path, converged };
}

export default function WarmStartExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const preset = warmStartPresets[0];
  const [trueAlpha, setTrueAlpha] = useState(preset.trueAlpha);
  const [n, setN] = useState(preset.defaultN);
  const [seed, setSeed] = useState(1);

  const data = useMemo(() => {
    const rng = makeLCG(seed * 17 + n + Math.floor(trueAlpha * 1000));
    return sampleSequence(() => gammaSample(trueAlpha, preset.trueBeta, rng), n);
  }, [seed, n, trueAlpha, preset.trueBeta]);

  const xbar = useMemo(() => sampleMean(data), [data]);
  const momAlpha = useMemo(() => momGamma(data).alphaHat, [data]);
  const profileBeta = momAlpha > 0 && xbar > 0 ? momAlpha / xbar : preset.trueBeta;

  const trajectories = useMemo(() => {
    const trueMle = mleGammaShape(data, profileBeta, { init: momAlpha, maxIter: 40 }).mle;
    const inits = [
      { label: 'MoM init', color: '#2563EB', start: momAlpha },
      { label: 'Small (0.5)', color: '#DC2626', start: 0.5 },
      { label: 'Large (2 × MoM)', color: '#7C3AED', start: 2 * momAlpha },
    ];
    return inits.map((it) => ({
      ...it,
      ...newtonPath(data, profileBeta, it.start),
      finalMLE: trueMle,
    }));
  }, [data, profileBeta, momAlpha]);

  // ── Left panel: log-likelihood ℓ(α) with iterate markers ────────────────
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

    const allPath = trajectories.flatMap((t) => t.path);
    const lo = Math.max(0.1, Math.min(...allPath, 0.4));
    const hi = Math.max(...allPath, 1.5 * trueAlpha) * 1.1;
    const grid: number[] = [];
    const G = 121;
    for (let i = 0; i < G; i++) grid.push(lo + ((hi - lo) * i) / (G - 1));

    let sumLog = 0;
    for (let i = 0; i < data.length; i++) sumLog += Math.log(Math.max(data[i], 1e-300));
    const sumX = data.reduce((s, x) => s + x, 0);
    const ll = grid.map((alpha) => {
      if (alpha <= 0) return NaN;
      return n * (alpha * Math.log(profileBeta) - logGamma(alpha)) + (alpha - 1) * sumLog - profileBeta * sumX;
    });

    const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
    const llValid = ll.filter(Number.isFinite);
    const yLo = (d3.min(llValid) ?? 0);
    const yHi = (d3.max(llValid) ?? 1);
    const y = d3.scaleLinear().domain([yLo, yHi]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(6))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('text').attr('x', innerW / 2).attr('y', innerH + 28)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-text-muted)')
      .text('α');

    const line = d3.line<number>()
      .defined((d) => Number.isFinite(d))
      .x((_, i) => x(grid[i])).y((d) => y(d));

    g.append('path').datum(ll).attr('fill', 'none').attr('stroke', '#374151')
      .attr('stroke-width', 1.5).attr('d', line);

    // True MLE marker.
    const finalMLE = trajectories[0].finalMLE;
    if (Number.isFinite(finalMLE)) {
      g.append('line').attr('x1', x(finalMLE)).attr('x2', x(finalMLE))
        .attr('y1', 0).attr('y2', innerH).attr('stroke', '#111')
        .attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
      g.append('text').attr('x', x(finalMLE) + 4).attr('y', 12)
        .attr('font-size', 10).attr('fill', '#111').text('α̂_MLE');
    }

    // Iterate markers per trajectory.
    trajectories.forEach((tr) => {
      tr.path.forEach((alpha, i) => {
        if (alpha < lo || alpha > hi || alpha <= 0) return;
        const llVal = n * (alpha * Math.log(profileBeta) - logGamma(alpha)) + (alpha - 1) * sumLog - profileBeta * sumX;
        if (!Number.isFinite(llVal)) return;
        g.append('circle').attr('cx', x(alpha)).attr('cy', y(llVal))
          .attr('r', i === 0 ? 5 : 3).attr('fill', tr.color)
          .attr('opacity', i === 0 ? 1 : 0.7);
      });
    });

    // Legend.
    const lg = g.append('g').attr('transform', `translate(${innerW - 130}, 6)`)
      .style('font-size', '10px').attr('fill', 'var(--color-text)');
    trajectories.forEach((it, i) => {
      lg.append('circle').attr('cx', 5).attr('cy', i * 14).attr('r', 4).attr('fill', it.color);
      lg.append('text').attr('x', 14).attr('y', i * 14 + 3)
        .text(`${it.label} (${it.path.length - 1} steps)`);
    });
  }, [trajectories, leftW, data, n, profileBeta, trueAlpha]);

  // ── Right panel: convergence plot |α_t − α̂| vs iteration on log-y ──────
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

    const finalMLE = trajectories[0].finalMLE;
    const maxLen = Math.max(...trajectories.map((t) => t.path.length));
    const x = d3.scaleLinear().domain([0, Math.max(15, maxLen - 1)]).range([0, innerW]);
    const y = d3.scaleLog().domain([1e-9, 5]).clamp(true).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(6))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(5, '.0e'))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('text').attr('x', innerW / 2).attr('y', innerH + 28)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-text-muted)')
      .text('iteration');
    g.append('text').attr('x', -innerH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-text-muted)')
      .text('|α_t − α̂|');

    const line = d3.line<number>()
      .x((_, i) => x(i))
      .y((d) => y(Math.max(d, 1e-10)));

    trajectories.forEach((tr) => {
      const errs = tr.path.map((a) => Math.abs(a - finalMLE));
      g.append('path').datum(errs).attr('fill', 'none')
        .attr('stroke', tr.color).attr('stroke-width', 1.8)
        .attr('d', line);
      errs.forEach((e, i) => {
        g.append('circle').attr('cx', x(i)).attr('cy', y(Math.max(e, 1e-10)))
          .attr('r', 3).attr('fill', tr.color);
      });
    });
  }, [trajectories, rightW]);

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: MoM warm-start vs arbitrary inits</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Newton-Raphson on the Gamma shape MLE from three starting points. The MoM init lands close to the maximum, converging in 3–5 steps; arbitrary starts wander.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center" style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}>
        <label>true α = {trueAlpha.toFixed(2)}{' '}
          <input type="range" min={0.5} max={10} step={0.1}
            value={trueAlpha} onChange={(e) => { setTrueAlpha(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 100, verticalAlign: 'middle' }} />
        </label>
        <label>n = {n}{' '}
          <input type="range" min={50} max={500} step={10}
            value={n} onChange={(e) => { setN(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 110, verticalAlign: 'middle' }} />
        </label>
        <button onClick={() => setSeed((s) => s + 1)}
          style={{ padding: '0.25rem 0.65rem', fontSize: '0.8125rem',
            border: '1px solid var(--color-border)', borderRadius: '0.375rem',
            background: 'transparent', cursor: 'pointer' }}>
          New sample
        </button>
      </div>

      <div style={{ display: isWide ? 'flex' : 'block', gap: 12 }}>
        <svg ref={leftRef} width={leftW} height={H} style={{ display: 'block' }} />
        <svg ref={rightRef} width={rightW} height={H} style={{ display: 'block' }} />
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 8, lineHeight: 1.4 }}>
        <strong>α̂_MoM = </strong> <code>{momAlpha.toFixed(4)}</code>{' · '}
        <strong>α̂_MLE = </strong> <code>{trajectories[0]?.finalMLE.toFixed(4)}</code>{' · '}
        Steps: MoM init <strong>{trajectories[0]?.path.length - 1}</strong>, small init <strong>{trajectories[1]?.path.length - 1}</strong>, large init <strong>{trajectories[2]?.path.length - 1}</strong>.
      </div>
    </div>
  );
}
