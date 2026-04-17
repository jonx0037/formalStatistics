import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  standardNormalPDF,
  standardNormalCDF,
  standardNormalInvCDF,
  twoProportionPower,
} from './shared/testing';
import {
  nullAlternativePresets,
  type NullAlternativePreset,
  type TestSide,
} from '../../data/hypothesis-testing-data';

const MARGIN = { top: 16, right: 16, bottom: 40, left: 44 };
const H = 320;

const COLOR_NULL = '#059669'; // emerald — H₀
const COLOR_ALT = '#DC2626';  // red — H_A
const COLOR_TYPE_I = '#F59E0B';  // amber — Type I area (under H₀, in rejection)
const COLOR_TYPE_II = '#93C5FD'; // sky blue — Type II area (under H_A, not rejected)

/** Standardized effect size δ on the z-statistic scale. The alt density is
 *  approximated as N(δ, 1) for all three families; power readouts use
 *  family-specific closed-form functions from testing.ts. */
function standardizedEffect(preset: NullAlternativePreset, altTheta: number, n: number): number {
  if (preset.family === 'normal-mean-known-sigma') {
    return (Math.sqrt(n) * (altTheta - preset.nullTheta)) / (preset.sigmaKnown ?? 1);
  }
  if (preset.family === 'two-proportion') {
    const pA = preset.baselineP ?? 0.1;
    const pB = pA + altTheta;
    const pBar = (pA + pB) / 2;
    const se = Math.sqrt(2 * pBar * (1 - pBar) / n);
    return (pB - pA) / Math.max(se, 1e-9);
  }
  // exponential-mean: H_A shifts mean from 1/λ₀ to altTheta
  const lambda0 = preset.nullTheta;
  const mean0 = 1 / lambda0;
  const meanA = altTheta;
  return (Math.sqrt(n) * (meanA - mean0)) / mean0;
}

/** Power from the family-specific closed form. */
function computePower(
  preset: NullAlternativePreset,
  altTheta: number,
  n: number,
  alpha: number,
  side: TestSide,
): number {
  if (preset.family === 'normal-mean-known-sigma') {
    const delta = standardizedEffect(preset, altTheta, n);
    if (side === 'right') return 1 - standardNormalCDF(standardNormalInvCDF(1 - alpha) - delta);
    if (side === 'left') return standardNormalCDF(standardNormalInvCDF(alpha) - delta);
    const zHalf = standardNormalInvCDF(1 - alpha / 2);
    return 1 - standardNormalCDF(zHalf - delta) + standardNormalCDF(-zHalf - delta);
  }
  if (preset.family === 'two-proportion') {
    const pA = preset.baselineP ?? 0.1;
    const pB = pA + altTheta;
    return twoProportionPower(pB, pA, n, n, alpha, side);
  }
  // Exponential: use the z-approximation as a serviceable pedagogical reading.
  const delta = standardizedEffect(preset, altTheta, n);
  if (side === 'right') return 1 - standardNormalCDF(standardNormalInvCDF(1 - alpha) - delta);
  if (side === 'left') return standardNormalCDF(standardNormalInvCDF(alpha) - delta);
  const zHalf = standardNormalInvCDF(1 - alpha / 2);
  return 1 - standardNormalCDF(zHalf - delta) + standardNormalCDF(-zHalf - delta);
}

export default function NullAlternativeSimulator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = nullAlternativePresets[presetIndex];

  const [altTheta, setAltTheta] = useState(preset.altThetaDefault);
  const [n, setN] = useState(preset.nDefault);
  const [alpha, setAlpha] = useState(preset.alphaDefault);
  const [side, setSide] = useState<TestSide>('right');

  // Reset params when preset changes
  useEffect(() => {
    setAltTheta(preset.altThetaDefault);
    setN(preset.nDefault);
    setAlpha(preset.alphaDefault);
    if (!preset.sides.includes(side)) setSide(preset.sides[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  const delta = useMemo(() => standardizedEffect(preset, altTheta, n), [preset, altTheta, n]);
  const power = useMemo(() => computePower(preset, altTheta, n, alpha, side), [preset, altTheta, n, alpha, side]);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.62) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // z-axis range — dynamic based on δ
    const xMin = Math.min(-4, delta - 4);
    const xMax = Math.max(4, delta + 4);
    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 0.45]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${H}`)
      .attr('width', svgW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Grid
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(x).ticks(8))
      .call((sel) => sel.selectAll('text').style('font-size', '11px'))
      .call((sel) => sel.selectAll('line,path').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4));
    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .call((sel) => sel.selectAll('text').style('font-size', '11px'))
      .call((sel) => sel.selectAll('line,path').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4));

    // Densities
    const samples = 201;
    const step = (xMax - xMin) / (samples - 1);
    const nullPts = Array.from({ length: samples }, (_, i) => {
      const z = xMin + i * step;
      return [z, standardNormalPDF(z)] as [number, number];
    });
    const altPts = Array.from({ length: samples }, (_, i) => {
      const z = xMin + i * step;
      return [z, standardNormalPDF(z - delta)] as [number, number];
    });
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.3));
    const area = d3
      .area<[number, number]>()
      .x((d) => x(d[0]))
      .y0(y(0))
      .y1((d) => y(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.3));

    // Critical value(s)
    const critR = side === 'two' ? standardNormalInvCDF(1 - alpha / 2) : standardNormalInvCDF(1 - alpha);
    const critL = side === 'two' ? -critR : standardNormalInvCDF(alpha);

    // Shade Type I under null (rejection tails)
    if (side === 'right' || side === 'two') {
      const region = nullPts.filter(([z]) => z >= critR);
      if (region.length > 1) {
        g.append('path')
          .datum(region)
          .attr('d', area)
          .attr('fill', COLOR_TYPE_I)
          .attr('fill-opacity', 0.55);
      }
    }
    if (side === 'left' || side === 'two') {
      const region = nullPts.filter(([z]) => z <= critL);
      if (region.length > 1) {
        g.append('path')
          .datum(region)
          .attr('d', area)
          .attr('fill', COLOR_TYPE_I)
          .attr('fill-opacity', 0.55);
      }
    }

    // Shade Type II under alt (non-rejection middle / tails depending on side)
    if (side === 'right') {
      const region = altPts.filter(([z]) => z <= critR);
      g.append('path').datum(region).attr('d', area).attr('fill', COLOR_TYPE_II).attr('fill-opacity', 0.45);
    } else if (side === 'left') {
      const region = altPts.filter(([z]) => z >= critL);
      g.append('path').datum(region).attr('d', area).attr('fill', COLOR_TYPE_II).attr('fill-opacity', 0.45);
    } else {
      const region = altPts.filter(([z]) => z >= critL && z <= critR);
      g.append('path').datum(region).attr('d', area).attr('fill', COLOR_TYPE_II).attr('fill-opacity', 0.45);
    }

    // Critical value lines
    const critLines: number[] = [];
    if (side === 'right' || side === 'two') critLines.push(critR);
    if (side === 'left' || side === 'two') critLines.push(critL);
    critLines.forEach((c) => {
      g.append('line')
        .attr('x1', x(c))
        .attr('x2', x(c))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', 'currentColor')
        .attr('stroke-dasharray', '4,3')
        .attr('stroke-opacity', 0.6);
      g.append('text')
        .attr('x', x(c))
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .attr('fill', 'currentColor')
        .attr('fill-opacity', 0.75)
        .text(`c = ${c.toFixed(2)}`);
    });

    // Null + alt curves on top
    g.append('path').datum(nullPts).attr('d', line).attr('stroke', COLOR_NULL).attr('stroke-width', 2).attr('fill', 'none');
    g.append('path').datum(altPts).attr('d', line).attr('stroke', COLOR_ALT).attr('stroke-width', 2).attr('fill', 'none');

    // Labels
    g.append('text')
      .attr('x', x(0))
      .attr('y', y(standardNormalPDF(0)) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', COLOR_NULL)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text('H₀');
    g.append('text')
      .attr('x', x(delta))
      .attr('y', y(standardNormalPDF(0)) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', COLOR_ALT)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text('H_A');

    // X-axis title
    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .style('font-size', '11px')
      .text('z-statistic (standardized)');
  }, [preset, delta, alpha, side, svgW]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Null vs Alternative · §17.3
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {nullAlternativePresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            {preset.family === 'two-proportion' ? 'lift p_B − p_A' : preset.family === 'exponential-mean' ? 'true mean' : 'true μ'} ={' '}
            <strong>{altTheta.toFixed(preset.family === 'two-proportion' ? 3 : 2)}</strong>
          </span>
          <input
            type="range"
            min={preset.altThetaRange[0]}
            max={preset.altThetaRange[1]}
            step={preset.altThetaStep}
            value={altTheta}
            onChange={(e) => setAltTheta(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
            {preset.family === 'two-proportion' ? ' / arm' : ''}
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
            α = <strong>{alpha.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0.001}
            max={0.3}
            step={0.001}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
        </label>
        <div className="flex flex-col gap-1 text-xs">
          <span style={{ color: 'var(--color-text-muted)' }}>alternative side</span>
          <div className="flex gap-2">
            {preset.sides.map((s) => (
              <label key={s} className="flex items-center gap-1">
                <input type="radio" name="side" checked={side === s} onChange={() => setSide(s)} />
                {s === 'right' ? 'θ > θ₀' : s === 'left' ? 'θ < θ₀' : 'θ ≠ θ₀'}
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className={isWide ? 'grid grid-cols-[62%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <svg ref={svgRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-3">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Live readouts
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded p-2" style={{ background: 'rgba(245,158,11,0.12)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: '#B45309' }}>Type I (α)</div>
              <div className="text-base font-mono">{alpha.toFixed(3)}</div>
            </div>
            <div className="rounded p-2" style={{ background: 'rgba(147,197,253,0.22)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: '#1D4ED8' }}>Type II (1−β)</div>
              <div className="text-base font-mono">{(1 - power).toFixed(3)}</div>
            </div>
            <div className="rounded p-2" style={{ background: 'rgba(5,150,105,0.12)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: '#047857' }}>Power (β)</div>
              <div className="text-base font-mono">{power.toFixed(3)}</div>
            </div>
            <div className="rounded p-2" style={{ background: 'rgba(220,38,38,0.08)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: '#B91C1C' }}>δ (z-scale)</div>
              <div className="text-base font-mono">{delta.toFixed(2)}</div>
            </div>
          </div>
          <div className="text-xs italic leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
            {preset.description}
          </div>
          <div className="text-[11px] leading-relaxed" style={{ color: 'var(--color-text)' }}>
            Moving H_A further from H₀ increases power; increasing n shrinks both distributions, narrowing their overlap. Shrinking α pushes the critical value out, trading Type I error for Type II.
          </div>
        </div>
      </div>
    </div>
  );
}
