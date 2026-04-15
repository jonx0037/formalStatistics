import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  computeMLE,
  logLikelihoodCurve,
  reparameterizedLogLik,
} from './shared/estimation';
import {
  normalSample,
  bernoulliSample,
  exponentialSample,
  sampleSequence,
} from './shared/convergence';
import {
  invariancePresets,
  type InvariancePreset,
} from '../../data/maximum-likelihood-data';

const MARGIN = { top: 18, right: 16, bottom: 40, left: 52 };
const H = 260;

function makeSampler(preset: InvariancePreset): (n: number, rng: () => number) => number[] {
  switch (preset.baseFamily) {
    case 'Normal': {
      const mu = preset.otherParams?.mu ?? 0;
      const sigma = Math.sqrt(preset.trueParam); // trueParam is σ² here
      return (n, rng) => sampleSequence(() => normalSample(mu, sigma, rng), n);
    }
    case 'Bernoulli':
      return (n, rng) => sampleSequence(() => bernoulliSample(preset.trueParam, rng), n);
    case 'Exponential':
      return (n, rng) => sampleSequence(() => exponentialSample(preset.trueParam, rng), n);
    default:
      return () => [];
  }
}

/** Per-family log-pdf closure for plotting ℓ(θ). */
function makeLogPdf(preset: InvariancePreset): (x: number, theta: number) => number {
  switch (preset.baseFamily) {
    case 'Normal': {
      const mu = preset.otherParams?.mu ?? 0;
      return (x, sigma2) => {
        if (sigma2 <= 0) return -Infinity;
        const z = x - mu;
        return -0.5 * Math.log(2 * Math.PI * sigma2) - (z * z) / (2 * sigma2);
      };
    }
    case 'Bernoulli':
      return (x, p) => {
        if (p <= 0 || p >= 1) return -Infinity;
        return x * Math.log(p) + (1 - x) * Math.log(1 - p);
      };
    case 'Exponential':
      return (x, lam) => {
        if (lam <= 0) return -Infinity;
        return Math.log(lam) - lam * x;
      };
    default:
      return () => 0;
  }
}

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export default function MLEInvarianceExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 760;

  const [presetIndex, setPresetIndex] = useState(0);
  const [n, setN] = useState(50);
  const [seed, setSeed] = useState(1);
  const leftRef = useRef<SVGSVGElement | null>(null);
  const rightRef = useRef<SVGSVGElement | null>(null);

  const preset = invariancePresets[presetIndex];
  const logPdf = useMemo(() => makeLogPdf(preset), [preset]);
  const sampler = useMemo(() => makeSampler(preset), [preset]);
  const rng = useMemo(() => makeLCG(seed * 7919 + n + presetIndex * 131), [seed, n, presetIndex]);
  const data = useMemo(() => sampler(n, rng), [sampler, n, rng]);

  const mleRes = useMemo(
    () => computeMLE(data, preset.baseFamily, preset.baseParam, preset.otherParams),
    [data, preset],
  );
  const thetaHat = mleRes.mle;
  const phiHat = preset.transform(thetaHat);

  // Curve ℓ(θ) on the base parameter.
  const thetaCurve = useMemo(() => {
    const [lo, hi] = preset.thetaRange;
    const grid: number[] = [];
    const G = 161;
    for (let i = 0; i < G; i++) grid.push(lo + ((hi - lo) * i) / (G - 1));
    return logLikelihoodCurve(data, logPdf, grid);
  }, [data, logPdf, preset.thetaRange]);

  // Reparameterized log-likelihood ℓ*(φ) = ℓ(g⁻¹(φ)).
  const phiCurve = useMemo(() => {
    const reparamLogLik = reparameterizedLogLik((theta) => {
      let s = 0;
      for (let i = 0; i < data.length; i++) s += logPdf(data[i], theta);
      return s;
    }, preset.inverse);
    const [lo, hi] = preset.phiRange;
    const G = 161;
    const grid: number[] = [];
    const values: number[] = [];
    for (let i = 0; i < G; i++) {
      const phi = lo + ((hi - lo) * i) / (G - 1);
      grid.push(phi);
      values.push(reparamLogLik(phi));
    }
    return { thetas: grid, logLiks: values };
  }, [data, logPdf, preset]);

  function drawPanel(
    ref: React.RefObject<SVGSVGElement | null>,
    panelW: number,
    grid: { thetas: number[]; logLiks: number[] },
    range: [number, number],
    marker: number,
    labelAxis: string,
    labelMarker: string,
    title: string,
  ) {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain(range).range([0, innerW]);
    const yMax = (d3.max(grid.logLiks) ?? 0) + 1;
    const yMin = yMax - 15;
    const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(title);

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(labelAxis);

    const pts = grid.thetas.map((t, i) => ({ t, l: grid.logLiks[i] }));
    g.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line<{ t: number; l: number }>()
          .x((d) => x(d.t))
          .y((d) => y(Math.max(d.l, yMin))),
      );

    // MLE marker (dashed red)
    if (marker >= range[0] && marker <= range[1]) {
      g.append('line')
        .attr('x1', x(marker))
        .attr('x2', x(marker))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 3');
      g.append('text')
        .attr('x', x(marker))
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', '#dc2626')
        .text(labelMarker);
    }
  }

  useEffect(() => {
    const leftW = isWide ? Math.floor(w / 2) : w;
    const rightW = leftW;
    drawPanel(leftRef, leftW, thetaCurve, preset.thetaRange, thetaHat, `θ (${preset.baseParam})`, `θ̂ = ${thetaHat.toFixed(3)}`, 'ℓ(θ) — base parameterization');
    drawPanel(rightRef, rightW, phiCurve, preset.phiRange, phiHat, `φ = g(θ) = ${preset.transformLabel}`, `g(θ̂) = ${phiHat.toFixed(3)}`, 'ℓ*(φ) — transformed parameter');
  }, [thetaCurve, phiCurve, preset, thetaHat, phiHat, w, isWide]);

  const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : '—');

  return (
    <div
      ref={containerRef}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)', fontFamily: 'system-ui, sans-serif' }}
    >
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>Interactive: Invariance Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          The same likelihood on two parameter scales. Left panel: ℓ(θ) peaks at the MLE θ̂. Right
          panel: ℓ*(φ) = ℓ(g⁻¹(φ)) peaks at g(θ̂). The two maximizers are related by the
          transformation — you do not need a second maximization to transport the estimate.
        </div>
      </div>

      <div
        className="flex flex-wrap gap-3 items-center"
        style={{ fontSize: '0.8125rem', margin: '0.5rem 0 0.75rem' }}
      >
        <label>
          Transform:{' '}
          <select
            value={presetIndex}
            onChange={(e) => {
              setPresetIndex(Number(e.target.value));
              setSeed((s) => s + 1);
            }}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {invariancePresets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          n = {n}{' '}
          <input
            type="range"
            min={10}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 120, marginLeft: 4, verticalAlign: 'middle' }}
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '0.25rem 0.65rem',
            fontSize: '0.8125rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.375rem',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          ↻ New sample
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isWide ? '1fr 1fr' : '1fr',
          gap: '0.5rem',
        }}
      >
        <svg
          ref={leftRef}
          width={isWide ? Math.floor(w / 2) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={rightRef}
          width={isWide ? Math.floor(w / 2) : w}
          height={H}
          style={{ overflow: 'visible' }}
        />
      </div>

      <div
        style={{
          marginTop: '0.5rem',
          fontSize: '0.8125rem',
          fontVariantNumeric: 'tabular-nums',
          color: 'var(--color-text-muted)',
        }}
      >
        θ̂ = <strong>{fmt(thetaHat)}</strong> &nbsp; · &nbsp; g(θ̂) ={' '}
        <strong>{fmt(phiHat)}</strong> &nbsp; · &nbsp; {preset.description}
      </div>
    </div>
  );
}
