import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  momGamma,
  mleGammaShape,
  sampleMean,
  sampleVariance,
} from './shared/estimation';
import { gammaSample, sampleSequence } from './shared/convergence';
import { pdfGamma } from './shared/distributions';
import { gammaPresets } from '../../data/method-of-moments-data';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 48 };
const H = 240;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export default function MomGammaSolver() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = gammaPresets[presetIndex];

  const [trueAlpha, setTrueAlpha] = useState(preset.trueAlpha);
  const [trueBeta, setTrueBeta] = useState(preset.trueBeta);
  const [n, setN] = useState(preset.defaultN);
  const [seed, setSeed] = useState(1);

  // Reset alpha/beta/n when preset changes.
  useEffect(() => {
    const p = gammaPresets[presetIndex];
    setTrueAlpha(p.trueAlpha);
    setTrueBeta(p.trueBeta);
    setN(p.defaultN);
    setSeed((s) => s + 1);
  }, [presetIndex]);

  const data = useMemo(() => {
    const rng = makeLCG(seed * 7919 + n + Math.floor(trueAlpha * 1000) + Math.floor(trueBeta * 1000));
    return sampleSequence(() => gammaSample(trueAlpha, trueBeta, rng), n);
  }, [seed, n, trueAlpha, trueBeta]);

  const xbar = useMemo(() => sampleMean(data), [data]);
  const s2 = useMemo(() => sampleVariance(data, 0), [data]); // biased 1/n

  const mom = useMemo(() => momGamma(data), [data]);

  const mle = useMemo(() => {
    // Profile β from the MoM α: β = α / X̄ at the MoM. Then iterate α via Newton on the
    // shape-only score, holding β fixed at this profile (standard "shape MLE given β" pattern).
    const beta0 = mom.alphaHat > 0 ? mom.alphaHat / xbar : trueBeta;
    return mleGammaShape(data, beta0, { init: mom.alphaHat, maxIter: 30, tol: 1e-9 });
  }, [data, mom.alphaHat, xbar, trueBeta]);

  const mleBeta = useMemo(() => {
    return mle.mle > 0 && xbar > 0 ? mle.mle / xbar : NaN;
  }, [mle.mle, xbar]);

  // ── Histogram + density overlays (left panel) ────────────────────────────
  const xMax = useMemo(() => {
    const mx = d3.max(data) ?? 1;
    return mx * 1.15;
  }, [data]);
  const histBins = useMemo(() => {
    const bins = d3.bin().domain([0, xMax]).thresholds(28)(data);
    return bins;
  }, [data, xMax]);

  const histRef = useRef<SVGSVGElement | null>(null);
  const histW = isWide ? Math.floor(w * 0.55) : w;

  useEffect(() => {
    const svg = d3.select(histRef.current);
    if (!svg.node()) return;
    svg.selectAll('*').remove();
    const innerW = histW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.attr('viewBox', `0 0 ${histW} ${H}`).append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const x = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);
    // Density y-scale (PDF max from true density)
    const grid = d3.range(0, xMax, xMax / 200);
    const trueDensity = grid.map((g0) => pdfGamma(g0, trueAlpha, trueBeta));
    const momDensity = grid.map((g0) => pdfGamma(g0, mom.alphaHat, mom.betaHat));
    const mleDensity = grid.map((g0) => pdfGamma(g0, mle.mle, mleBeta));
    const yMaxDensity = (d3.max([...trueDensity, ...momDensity, ...mleDensity]) ?? 1) * 1.05;
    // Histogram is rescaled to share the density y-axis (relative-frequency form)
    const totalArea = data.length * (histBins[0]?.x1! - histBins[0]?.x0!);
    const yHistMax = (d3.max(histBins, (b) => b.length / totalArea) ?? 0) * 1.05;
    const yMax = Math.max(yMaxDensity, yHistMax);
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(6))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));
    g.append('g').call(d3.axisLeft(y).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '10px'));

    // histogram bars
    g.selectAll('rect').data(histBins).enter().append('rect')
      .attr('x', (b) => x(b.x0!))
      .attr('y', (b) => y(b.length / totalArea))
      .attr('width', (b) => Math.max(0, x(b.x1!) - x(b.x0!) - 1))
      .attr('height', (b) => innerH - y(b.length / totalArea))
      .attr('fill', 'var(--color-text-muted)')
      .attr('opacity', 0.25);

    const line = d3.line<number>().x((_, i) => x(grid[i])).y((d) => y(d));

    g.append('path').datum(trueDensity).attr('fill', 'none').attr('stroke', '#374151')
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3').attr('d', line);

    if (Number.isFinite(mom.alphaHat) && Number.isFinite(mom.betaHat)) {
      g.append('path').datum(momDensity).attr('fill', 'none').attr('stroke', '#2563EB')
        .attr('stroke-width', 2).attr('d', line);
    }
    if (Number.isFinite(mle.mle) && Number.isFinite(mleBeta)) {
      g.append('path').datum(mleDensity).attr('fill', 'none').attr('stroke', '#D97706')
        .attr('stroke-width', 2).attr('stroke-dasharray', '6 2').attr('d', line);
    }

    // Legend
    const lx = innerW - 130;
    const lg = g.append('g').attr('transform', `translate(${lx}, 6)`)
      .style('font-size', '10px').attr('fill', 'var(--color-text)');
    const items = [
      { c: '#374151', d: '4 3', label: 'true' },
      { c: '#2563EB', d: '', label: 'MoM' },
      { c: '#D97706', d: '6 2', label: 'MLE' },
    ];
    items.forEach((it, i) => {
      lg.append('line').attr('x1', 0).attr('x2', 16).attr('y1', i * 14).attr('y2', i * 14)
        .attr('stroke', it.c).attr('stroke-width', 2).attr('stroke-dasharray', it.d);
      lg.append('text').attr('x', 22).attr('y', i * 14 + 3).text(it.label);
    });
  }, [histBins, histW, data, mom, mle, mleBeta, trueAlpha, trueBeta, xMax]);

  const sideW = isWide ? w - histW - 12 : w;

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: MoM ↔ MLE for Gamma(α, β)</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          MoM gives a closed-form estimate in two lines of algebra; MLE requires Newton-Raphson on the digamma equation. Both target the same true (α, β) at large n.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center" style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          style={{ padding: '0.2rem' }}
        >
          {gammaPresets.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
        </select>
        <label>
          α = {trueAlpha.toFixed(2)}{' '}
          <input type="range" min={0.5} max={10} step={0.1}
            value={trueAlpha} onChange={(e) => { setTrueAlpha(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 90, verticalAlign: 'middle' }} />
        </label>
        <label>
          β = {trueBeta.toFixed(2)}{' '}
          <input type="range" min={0.5} max={5} step={0.1}
            value={trueBeta} onChange={(e) => { setTrueBeta(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 90, verticalAlign: 'middle' }} />
        </label>
        <label>
          n = {n}{' '}
          <input type="range" min={10} max={500} step={1}
            value={n} onChange={(e) => { setN(Number(e.target.value)); setSeed((s) => s + 1); }}
            style={{ width: 110, verticalAlign: 'middle' }} />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          style={{ padding: '0.25rem 0.65rem', fontSize: '0.8125rem',
            border: '1px solid var(--color-border)', borderRadius: '0.375rem',
            background: 'transparent', cursor: 'pointer' }}
        >
          New sample
        </button>
      </div>

      <div style={{ display: isWide ? 'flex' : 'block', gap: 12 }}>
        <svg ref={histRef} width={histW} height={H} style={{ display: 'block' }} />

        <div style={{
          width: sideW,
          fontSize: '0.8125rem',
          color: 'var(--color-text)',
          padding: isWide ? '0' : '0.75rem 0 0',
        }}>
          <div style={{ marginBottom: 8 }}>
            <strong>Sample summary</strong>{' '}
            <span style={{ color: 'var(--color-text-muted)' }}>(n = {n})</span>
            <div>X̄<sub>n</sub> = <code>{xbar.toFixed(4)}</code></div>
            <div>S²<sub>n</sub> (biased) = <code>{s2.toFixed(4)}</code></div>
          </div>

          <div style={{ marginBottom: 8, padding: '6px 8px',
              background: 'rgba(37, 99, 235, 0.06)', borderLeft: '3px solid #2563EB',
              borderRadius: 4 }}>
            <strong>MoM (closed form)</strong>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem', marginTop: 2 }}>
              α̂ = X̄² / S² = {xbar.toFixed(3)}² / {s2.toFixed(3)} = <strong>{mom.alphaHat.toFixed(4)}</strong>
            </div>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem' }}>
              β̂ = X̄ / S² = {xbar.toFixed(3)} / {s2.toFixed(3)} = <strong>{mom.betaHat.toFixed(4)}</strong>
            </div>
          </div>

          <div style={{ marginBottom: 8, padding: '6px 8px',
              background: 'rgba(217, 119, 6, 0.06)', borderLeft: '3px solid #D97706',
              borderRadius: 4 }}>
            <strong>MLE (Newton-Raphson, MoM-warm-started)</strong>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem', marginTop: 2 }}>
              α̂ = <strong>{Number.isFinite(mle.mle) ? mle.mle.toFixed(4) : '—'}</strong>{' '}
              <span style={{ color: 'var(--color-text-muted)' }}>
                ({mle.iterations} iter{mle.iterations === 1 ? '' : 's'}{mle.converged ? ', converged' : ''})
              </span>
            </div>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem' }}>
              β̂ = α̂ / X̄ = <strong>{Number.isFinite(mleBeta) ? mleBeta.toFixed(4) : '—'}</strong>
            </div>
          </div>

          <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', lineHeight: 1.4 }}>
            <strong>Cost:</strong> MoM is O(n); MLE is O(n · iter). At α = 3, β = 2 with n = 100, both estimates land within ~10% of the truth, but MoM extracts ≈ 68% of MLE&apos;s information (ARE ≈ 0.68 — see §15.7).
          </div>
        </div>
      </div>
    </div>
  );
}
