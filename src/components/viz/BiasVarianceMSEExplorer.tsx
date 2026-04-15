import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mseShrinkage,
  mseRidge,
  msePolynomial,
  optimalShrinkage,
} from './shared/estimation';
import { shrinkagePresets, type ShrinkagePreset } from '../../data/point-estimation-data';

type Mode = 'shrinkage' | 'regularization' | 'complexity';

const MARGIN = { top: 14, right: 20, bottom: 38, left: 52 };
const H = 260;

interface ModeSpec {
  xLabel: string;
  xRange: [number, number];
  xDefault: number;
  xStep: number;
}

function buildCurve(
  mode: Mode,
  preset: ShrinkagePreset,
  xs: number[],
): { bias2: number[]; variance: number[]; mse: number[] } {
  const bias2: number[] = [];
  const variance: number[] = [];
  const mse: number[] = [];
  for (const x of xs) {
    let r: { bias2: number; variance: number; mse: number };
    if (mode === 'shrinkage') {
      r = mseShrinkage(x, preset.mu, preset.sigma2, preset.n);
    } else if (mode === 'regularization') {
      r = mseRidge(x, preset.mu, preset.sigma2, preset.n);
    } else {
      // Model-complexity mode: x is polynomial degree. Treat the preset mu as
      // the "true degree" of a polynomial signal and sigma2 as the noise variance.
      const trueDeg = Math.max(2, Math.round(preset.mu));
      r = msePolynomial(Math.round(x), trueDeg, preset.n, Math.sqrt(preset.sigma2));
    }
    bias2.push(r.bias2);
    variance.push(r.variance);
    mse.push(r.mse);
  }
  return { bias2, variance, mse };
}

export default function BiasVarianceMSEExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [mode, setMode] = useState<Mode>('shrinkage');
  const [presetIndex, setPresetIndex] = useState(0);

  const preset = shrinkagePresets[presetIndex];

  const spec: ModeSpec = useMemo(() => {
    switch (mode) {
      case 'shrinkage':
        return { xLabel: 'Shrinkage factor c', xRange: preset.cRange, xDefault: 1.0, xStep: 0.01 };
      case 'regularization':
        return { xLabel: 'Ridge penalty λ', xRange: [0, 50], xDefault: 0, xStep: 0.5 };
      case 'complexity':
        return { xLabel: 'Polynomial degree', xRange: [1, 15], xDefault: 3, xStep: 1 };
    }
  }, [mode, preset]);

  const [sliderVal, setSliderVal] = useState(spec.xDefault);

  // Reset slider when mode changes
  useEffect(() => {
    setSliderVal(spec.xDefault);
  }, [spec.xDefault, spec.xRange]);

  // Curve grid (100 points)
  const xs = useMemo(() => {
    const [a, b] = spec.xRange;
    const count = mode === 'complexity' ? b - a + 1 : 101;
    return d3.range(count).map((i) => a + ((b - a) * i) / (count - 1));
  }, [spec.xRange, mode]);
  const curves = useMemo(() => buildCurve(mode, preset, xs), [mode, preset, xs]);

  // Current point on the curve
  const current = useMemo(() => {
    if (mode === 'shrinkage') return mseShrinkage(sliderVal, preset.mu, preset.sigma2, preset.n);
    if (mode === 'regularization') return mseRidge(sliderVal, preset.mu, preset.sigma2, preset.n);
    const trueDeg = Math.max(2, Math.round(preset.mu));
    return msePolynomial(Math.round(sliderVal), trueDeg, preset.n, Math.sqrt(preset.sigma2));
  }, [mode, preset, sliderVal]);

  // Optimal point (closed form only for shrinkage mode; numeric argmin otherwise)
  const optimal = useMemo(() => {
    if (mode === 'shrinkage') {
      return optimalShrinkage(preset.mu, preset.sigma2, preset.n);
    }
    // numeric argmin over the grid
    let bestIdx = 0;
    for (let i = 1; i < curves.mse.length; i++) {
      if (curves.mse[i] < curves.mse[bestIdx]) bestIdx = i;
    }
    return xs[bestIdx];
  }, [mode, preset, curves, xs]);

  const leftRef = useRef<SVGSVGElement | null>(null);
  const rightRef = useRef<SVGSVGElement | null>(null);

  // Left panel: three curves
  useEffect(() => {
    if (!leftRef.current) return;
    const svg = d3.select(leftRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.6 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain(spec.xRange).range([0, innerW]);
    const allVals = curves.bias2.concat(curves.variance, curves.mse);
    const yMax = (d3.max(allVals) ?? 1) * 1.05;
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(spec.xLabel);

    const linePath = (ys: number[], color: string) => {
      const gen = d3
        .line<number>()
        .x((_, i) => x(xs[i]))
        .y((d) => y(d));
      g.append('path').datum(ys).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('d', gen);
    };

    linePath(curves.bias2, '#2563eb');
    linePath(curves.variance, '#dc2626');
    linePath(curves.mse, '#111827');

    // Slider indicator
    g.append('line')
      .attr('x1', x(sliderVal))
      .attr('x2', x(sliderVal))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3');

    // Optimal star
    const optIdx = xs.reduce(
      (best, xi, i) => (Math.abs(xi - optimal) < Math.abs(xs[best] - optimal) ? i : best),
      0,
    );
    g.append('circle')
      .attr('cx', x(optimal))
      .attr('cy', y(curves.mse[optIdx]))
      .attr('r', 5)
      .attr('fill', '#f59e0b')
      .attr('stroke', '#78350f')
      .attr('stroke-width', 1);

    // Legend
    const labels: [string, string][] = [
      ['Bias²', '#2563eb'],
      ['Var', '#dc2626'],
      ['MSE', '#111827'],
    ];
    labels.forEach(([label, color], i) => {
      g.append('circle')
        .attr('cx', innerW - 70)
        .attr('cy', 10 + i * 14)
        .attr('r', 4)
        .attr('fill', color);
      g.append('text')
        .attr('x', innerW - 60)
        .attr('y', 14 + i * 14)
        .style('font-size', '10px')
        .style('fill', 'currentColor')
        .text(label);
    });
  }, [curves, spec, sliderVal, optimal, w, xs]);

  // Right panel: stacked bar Bias² + Var at current slider position
  useEffect(() => {
    if (!rightRef.current) return;
    const svg = d3.select(rightRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.4 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const maxMSE = Math.max(...curves.mse, current.mse);
    const y = d3.scaleLinear().domain([0, maxMSE * 1.05]).range([innerH, 0]).nice();

    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    // Single stacked bar at center
    const barX = innerW / 2 - 30;
    const barW = 60;

    g.append('rect')
      .attr('x', barX)
      .attr('y', y(current.variance + current.bias2))
      .attr('width', barW)
      .attr('height', y(0) - y(current.bias2))
      .attr('fill', '#2563eb');

    g.append('rect')
      .attr('x', barX)
      .attr('y', y(current.variance))
      .attr('width', barW)
      .attr('height', y(0) - y(current.variance))
      .attr('fill', '#dc2626');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('MSE composition');

    // Labels on bar
    if (current.bias2 / current.mse > 0.08) {
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', y(current.mse) + (y(current.variance) - y(current.mse)) / 2 + 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'white')
        .text('Bias²');
    }
    if (current.variance / current.mse > 0.08) {
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', y(current.variance) + (y(0) - y(current.variance)) / 2 + 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'white')
        .text('Var');
    }
  }, [current, curves, w]);

  const pctBias = current.mse > 0 ? (100 * current.bias2) / current.mse : 0;
  const pctVar = current.mse > 0 ? (100 * current.variance) / current.mse : 0;

  const modeLabels: Record<Mode, string> = {
    shrinkage: 'Shrinkage c·X̄',
    regularization: 'Ridge λ (prediction MSE)',
    complexity: 'Model complexity (prediction MSE)',
  };

  return (
    <div
      ref={containerRef}
      style={{
        fontFamily: 'system-ui, sans-serif',
        border: '1px solid var(--color-border)',
        borderRadius: '0.75rem',
        padding: '1.25rem',
        margin: '1.5rem 0',
      }}
    >
      <div style={{ marginBottom: '0.75rem' }}>
        <strong>Interactive: Bias-Variance-MSE Explorer</strong>
        <div
          style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}
        >
          Watch Bias² + Var = MSE trace out its U-shape as you slide across shrinkage,
          ridge penalty, or polynomial degree. Gold star marks the MSE-optimal setting.
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          gap: '0.75rem',
          flexWrap: 'wrap',
          marginBottom: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <label>
          Mode:{' '}
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            <option value="shrinkage">Shrinkage c·X̄ (§13.4)</option>
            <option value="regularization">Regularization (ridge λ)</option>
            <option value="complexity">Model complexity</option>
          </select>
        </label>
        <label>
          Preset:{' '}
          <select
            value={presetIndex}
            onChange={(e) => setPresetIndex(Number(e.target.value))}
            style={{ padding: '0.2rem', marginLeft: 4 }}
          >
            {shrinkagePresets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr auto',
          alignItems: 'center',
          gap: '0.5rem',
          marginBottom: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <span>
          {spec.xLabel} ={' '}
          {mode === 'complexity' ? Math.round(sliderVal) : sliderVal.toFixed(2)}
        </span>
        <input
          type="range"
          min={spec.xRange[0]}
          max={spec.xRange[1]}
          step={spec.xStep}
          value={sliderVal}
          onChange={(e) => setSliderVal(Number(e.target.value))}
        />
        <span style={{ color: 'var(--color-text-muted)' }}>
          optimal {mode === 'complexity' ? Math.round(optimal) : optimal.toFixed(3)}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 900 ? '3fr 2fr' : '1fr',
          gap: '0.75rem',
          alignItems: 'start',
        }}
      >
        <svg ref={leftRef} width={w > 900 ? (w - 24) * 0.6 : w} height={H} style={{ maxWidth: '100%' }} />
        <svg ref={rightRef} width={w > 900 ? (w - 24) * 0.4 : w} height={H} style={{ maxWidth: '100%' }} />
      </div>

      <div
        style={{
          marginTop: '0.5rem',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '0.5rem',
          fontSize: '0.8125rem',
        }}
      >
        <div>
          <div style={{ color: 'var(--color-text-muted)' }}>{modeLabels[mode]}</div>
          <div style={{ fontWeight: 600 }}>
            Bias² = {current.bias2.toFixed(4)} ({pctBias.toFixed(1)}%)
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--color-text-muted)' }}>&nbsp;</div>
          <div style={{ fontWeight: 600 }}>
            Var = {current.variance.toFixed(4)} ({pctVar.toFixed(1)}%)
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--color-text-muted)' }}>&nbsp;</div>
          <div style={{ fontWeight: 600 }}>MSE = {current.mse.toFixed(4)}</div>
        </div>
      </div>
    </div>
  );
}
