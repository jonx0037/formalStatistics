import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  completenessProbe,
  incompletenessWitnessUniform,
  type SufficientFamily,
} from './shared/estimation';
import { completenessProbePresets } from '../../data/sufficient-statistics-data';

const MARGIN = { top: 14, right: 12, bottom: 32, left: 44 };
const H = 240;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

// Test functions for the complete families. T = ΣXᵢ for the supported families.
function getTestFunctions(family: SufficientFamily, n: number): Record<string, (t: number) => number> {
  return {
    'g₀ ≡ 0': () => 0,
    'g₁ = T − n·θ₀ (linear)': (t) => t - n * 0.5,
    'g₂ = (T − n/2)² / 10 (quadratic)': (t) => Math.pow(t - n / 2, 2) / 10,
    'g₃ = sin(2πT/n)': (t) => Math.sin(2 * Math.PI * t / n),
  };
}

export default function CompletenessProbe() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = completenessProbePresets[presetIndex];

  const [n, setN] = useState(20);
  const [seed, setSeed] = useState(11);

  useEffect(() => { setSeed((s) => s + 1); }, [presetIndex]);

  const isUniformShift = preset.family === 'uniform-shift';

  const thetaGrid = useMemo(() => {
    const [lo, hi] = preset.thetaRange;
    const k = 30;
    const out: number[] = [];
    for (let i = 0; i <= k; i++) out.push(lo + (hi - lo) * i / k);
    return out;
  }, [preset.thetaRange]);

  // For complete families: probe with several test functions.
  // For Uniform(θ, θ+1) (incomplete): show the centered-range witness against θ.
  const probeData = useMemo(() => {
    const rng = makeLCG(seed * 7919 + n);
    if (isUniformShift) {
      const witness = incompletenessWitnessUniform(thetaGrid, n, 1500, rng);
      // Also compute a non-witness contrast: E_θ[X_(1) - 1/(n+1)] grows linearly with θ.
      // Approximate via MC.
      const contrast: number[] = thetaGrid.map((th) => {
        let sum = 0;
        const M = 600;
        for (let m = 0; m < M; m++) {
          let mn = Infinity;
          for (let i = 0; i < n; i++) {
            const x = th + rng();
            if (x < mn) mn = x;
          }
          sum += mn - 1 / (n + 1);
        }
        return sum / M;
      });
      return { witness, contrast };
    }
    const tests = getTestFunctions(preset.family as SufficientFamily, n);
    const result = completenessProbe(preset.family as SufficientFamily, thetaGrid, tests, n, 600, rng);
    return result;
  }, [isUniformShift, preset.family, thetaGrid, n, seed]);

  const plotRef = useRef<SVGSVGElement | null>(null);
  const plotW = isWide ? Math.floor(w * 0.6) : w;

  useEffect(() => {
    if (!plotRef.current) return;
    const svg = d3.select(plotRef.current);
    svg.selectAll('*').remove();

    const innerW = plotW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain([thetaGrid[0], thetaGrid[thetaGrid.length - 1]]).range([0, innerW]);

    const allCurves = isUniformShift
      ? [(probeData as { witness: number[]; contrast: number[] }).witness, (probeData as { witness: number[]; contrast: number[] }).contrast]
      : Object.values(probeData as Record<string, number[]>);
    // d3.extent on an empty array returns [undefined, undefined], which yields
    // a NaN domain and renders nothing. Fall back to a unit domain around 0.
    const flat = allCurves.flat();
    const rawExt = d3.extent(flat);
    const yExt: [number, number] = (rawExt[0] !== undefined && rawExt[1] !== undefined)
      ? [rawExt[0], rawExt[1]]
      : [-1, 1];
    const yPad = Math.max(0.5, (yExt[1] - yExt[0]) * 0.1);
    const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('font-size', '10px');

    // Zero line
    g.append('line')
      .attr('x1', 0).attr('x2', innerW).attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', 'currentColor').attr('stroke-dasharray', '2,3').attr('opacity', 0.5);

    g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'currentColor').text('θ');
    g.append('text').attr('transform', `rotate(-90)`).attr('x', -innerH / 2).attr('y', -34).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'currentColor').text('𝔼_θ[g(T)]');

    const colors = ['rgb(70,160,90)', 'rgb(220,100,70)', 'rgb(70,130,180)', 'rgb(180,120,200)'];
    const line = d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]));

    if (isUniformShift) {
      const w = (probeData as { witness: number[]; contrast: number[] }).witness;
      const c = (probeData as { witness: number[]; contrast: number[] }).contrast;
      const witnessPoints = thetaGrid.map((th, i) => [th, w[i]] as [number, number]);
      const contrastPoints = thetaGrid.map((th, i) => [th, c[i]] as [number, number]);
      g.append('path').datum(witnessPoints).attr('d', line).attr('fill', 'none').attr('stroke', colors[0]).attr('stroke-width', 2);
      g.append('path').datum(contrastPoints).attr('d', line).attr('fill', 'none').attr('stroke', colors[1]).attr('stroke-width', 2);

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 220}, 4)`);
      legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 6).attr('y2', 6).attr('stroke', colors[0]).attr('stroke-width', 2);
      legend.append('text').attr('x', 20).attr('y', 10).attr('font-size', 10).attr('fill', 'currentColor').text('R − (n−1)/(n+1) — witness');
      legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 22).attr('y2', 22).attr('stroke', colors[1]).attr('stroke-width', 2);
      legend.append('text').attr('x', 20).attr('y', 26).attr('font-size', 10).attr('fill', 'currentColor').text('X₍₁₎ − 1/(n+1) — non-witness');
    } else {
      const probe = probeData as Record<string, number[]>;
      const names = Object.keys(probe);
      names.forEach((name, idx) => {
        const pts = thetaGrid.map((th, i) => [th, probe[name][i]] as [number, number]);
        g.append('path').datum(pts).attr('d', line).attr('fill', 'none').attr('stroke', colors[idx % colors.length]).attr('stroke-width', 2);
      });

      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      names.forEach((name, idx) => {
        legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 6 + idx * 16).attr('y2', 6 + idx * 16).attr('stroke', colors[idx % colors.length]).attr('stroke-width', 2);
        legend.append('text').attr('x', 20).attr('y', 10 + idx * 16).attr('font-size', 10).attr('fill', 'currentColor').text(name);
      });
    }
  }, [probeData, thetaGrid, plotW, isUniformShift]);

  return (
    <div ref={containerRef} className="not-prose my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}>
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Completeness Probe · §16.6
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {completenessProbePresets.map((p, i) => (
            <option key={i} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>n = {n}</span>
          <input type="range" min={5} max={50} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-full" />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-evaluate
        </button>
        <div className="text-xs" style={{ color: preset.isComplete ? 'rgb(40,160,80)' : 'rgb(220,100,70)' }}>
          {preset.isComplete ? '✓ Family is COMPLETE' : '✗ Family is INCOMPLETE'}
        </div>
      </div>

      <div className={isWide ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-4'}>
        <svg ref={plotRef} width={plotW} height={H} style={{ display: 'block' }} />
        <div className="text-sm space-y-2">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Reading the plot
          </div>
          {preset.isComplete ? (
            <p className="text-xs">
              For <strong>complete</strong> families, only the constant-zero function g₀ ≡ 0 produces a flat zero curve in 𝔼_θ[g(T)] vs. θ. Any other test function traces a non-trivial θ-dependence.
            </p>
          ) : (
            <p className="text-xs">
              For <strong>Uniform(θ, θ+1)</strong>, the centered range g(T) = R − (n−1)/(n+1) where R = X₍ₙ₎ − X₍₁₎ is a non-trivial witness to incompleteness — its expectation is identically zero across θ (because R is ancillary), yet g is not the zero function. The contrast curve X₍₁₎ − 1/(n+1) traces θ linearly, confirming that not every centered statistic of T has zero expectation.
            </p>
          )}
          {preset.witnessFn && (
            <div className="font-mono text-xs p-2 rounded" style={{ background: 'var(--color-code-bg, rgba(127,127,127,0.08))' }}>
              Witness: {preset.witnessFn}
            </div>
          )}
          <div className="text-xs italic" style={{ color: 'var(--color-text-muted)' }}>{preset.description}</div>
        </div>
      </div>
    </div>
  );
}
