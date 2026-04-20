/**
 * CVvsICComparator — Topic 24 §24.5 component.
 *
 * Empirical demonstration of Stone's CV ≡ AIC equivalence (Thm 4 / Proof 3).
 * Overlays five criteria on the canonical POLY_DGP polynomial-degree axis:
 *
 *   • LOO-CV (hat-matrix shortcut)
 *   • 5-fold CV
 *   • 10-fold CV
 *   • AIC
 *   • Mallows' Cp
 *
 * Each curve is independently toggleable via a checkbox so the reader can
 * isolate the LOO-vs-AIC pair (Stone's identification) from the broader
 * 5-criterion picture. An aria-live readout reports the gap between
 * argmin(LOO-CV) and argmin(AIC) — the empirical Stone identity.
 *
 * Curves are y-axis standardized (each shifted so its min is 0) for visual
 * comparability across criteria with different absolute scales.
 *
 * Data source: POLY_DGP_PRECOMPUTED from model-selection-data.ts (Topic 23
 * IIFE-precompute pattern; component renders in O(1) on toggle changes).
 */

import { useRef, useState, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  informationCriteriaColors,
  informationCriteriaLineStyles,
} from './shared/colorScales';
import { POLY_DGP_PRECOMPUTED } from '../../data/model-selection-data';

const MARGIN = { top: 24, right: 120, bottom: 44, left: 56 };
const PANEL_H = 360;
const AXIS_COLOR = '#94A3B8';
const TICK_COLOR = '#CBD5E1';

type Series = 'looCV' | 'cv5' | 'cv10' | 'aic' | 'cp';
const SERIES: Series[] = ['looCV', 'cv5', 'cv10', 'aic', 'cp'];
const LABEL: Record<Series, string> = {
  looCV: 'LOO-CV',
  cv5: '5-fold CV',
  cv10: '10-fold CV',
  aic: 'AIC',
  cp: 'Cp',
};
/** Map series → palette key. The three CV variants (LOO, 5-fold, 10-fold)
 *  use distinct hues — overlapping a single hue per CV variety would force
 *  reliance on dash patterns alone for distinction, which is harder to read
 *  on the curve-overlay layout this component renders. LOO keeps the
 *  empirical-CV "loo" hue (amber); 5-fold borrows the AICc indigo and
 *  10-fold borrows the BIC violet so each CV curve has its own
 *  hue-and-dash signature. The IC curves keep their own hues. */
const PALETTE: Record<Series, 'aic' | 'aicc' | 'bic' | 'cp' | 'loo'> = {
  looCV: 'loo',
  cv5: 'aicc',
  cv10: 'bic',
  aic: 'aic',
  cp: 'cp',
};

export default function CVvsICComparator() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const data = POLY_DGP_PRECOMPUTED;

  const [visible, setVisible] = useState<Record<Series, boolean>>({
    looCV: true,
    cv5: true,
    cv10: true,
    aic: true,
    cp: true,
  });
  const toggle = (s: Series): void =>
    setVisible((v) => ({ ...v, [s]: !v[s] }));

  // Stone-equivalence readout: |argmin(LOO) - argmin(AIC)|. NaN-safe in case
  // the LOO array's high-d tail is undefined.
  const argLoo = data.argmins.looCV;
  const argAic = data.argmins.aic;
  const stoneGap = Number.isFinite(argLoo) ? Math.abs(argLoo - argAic) : NaN;

  const svgRef = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const w = width;
    const h = PANEL_H;
    const minDeg = data.degrees[0];
    const maxDeg = data.degrees[data.degrees.length - 1];
    const xScale = d3
      .scaleLinear()
      .domain([minDeg, maxDeg])
      .range([MARGIN.left, w - MARGIN.right]);
    // Standardize each visible series to min=0; gather y-extent.
    const shifted: Record<Series, number[]> = {} as Record<Series, number[]>;
    let yMax = 0;
    for (const s of SERIES) {
      if (!visible[s]) continue;
      const v = data.values[s];
      const finite = v.filter(Number.isFinite);
      if (finite.length === 0) continue;
      const m = Math.min(...finite);
      shifted[s] = v.map((x) => (Number.isFinite(x) ? x - m : NaN));
      const localMax = Math.max(...shifted[s].filter(Number.isFinite));
      if (localMax > yMax) yMax = localMax;
    }
    if (yMax === 0) yMax = 1; // all curves hidden — avoid degenerate scale
    const yScale = d3
      .scaleLinear()
      .domain([0, yMax])
      .range([h - MARGIN.bottom, MARGIN.top]);
    // Axes
    svg
      .append('g')
      .attr('transform', `translate(0,${h - MARGIN.bottom})`)
      .call(
        d3
          .axisBottom(xScale)
          .ticks(7)
          .tickFormat((d) => String(d)),
      )
      .selectAll('path,line')
      .style('stroke', AXIS_COLOR);
    svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(6))
      .selectAll('path,line')
      .style('stroke', AXIS_COLOR);
    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', h - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', AXIS_COLOR)
      .text('polynomial degree d');
    svg
      .append('text')
      .attr('x', 12)
      .attr('y', h / 2)
      .attr('transform', `rotate(-90, 12, ${h / 2})`)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', AXIS_COLOR)
      .text('value − min(value)');
    // Argmin guides for AIC and LOO if both visible
    if (visible.aic) {
      svg
        .append('line')
        .attr('x1', xScale(argAic))
        .attr('x2', xScale(argAic))
        .attr('y1', MARGIN.top)
        .attr('y2', h - MARGIN.bottom)
        .attr('stroke', TICK_COLOR)
        .attr('stroke-dasharray', '3 3');
    }
    // Curves
    const lineGen = d3
      .line<{ x: number; y: number }>()
      .defined((d) => Number.isFinite(d.y))
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y));
    for (const s of SERIES) {
      if (!visible[s] || !shifted[s]) continue;
      const pts = data.degrees.map((deg, i) => ({ x: deg, y: shifted[s][i] }));
      svg
        .append('path')
        .datum(pts)
        .attr('fill', 'none')
        .style('stroke', informationCriteriaColors[PALETTE[s]])
        .attr('stroke-width', s === 'aic' || s === 'looCV' ? 2.5 : 2)
        .attr('stroke-dasharray', informationCriteriaLineStyles[PALETTE[s]])
        .attr('d', lineGen);
      // Argmin marker
      const v = data.values[s];
      let argIdx = -1;
      for (let i = 0; i < v.length; i++) {
        if (!Number.isFinite(v[i])) continue;
        if (argIdx === -1 || v[i] < v[argIdx]) argIdx = i;
      }
      if (argIdx >= 0) {
        svg
          .append('circle')
          .attr('cx', xScale(data.degrees[argIdx]))
          .attr('cy', yScale(0))
          .attr('r', 4)
          .style('fill', informationCriteriaColors[PALETTE[s]]);
      }
    }
    // Legend
    const legX = w - MARGIN.right + 6;
    let legY = MARGIN.top;
    for (const s of SERIES) {
      if (!visible[s]) continue;
      svg
        .append('line')
        .attr('x1', legX)
        .attr('x2', legX + 18)
        .attr('y1', legY)
        .attr('y2', legY)
        .style('stroke', informationCriteriaColors[PALETTE[s]])
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', informationCriteriaLineStyles[PALETTE[s]]);
      svg
        .append('text')
        .attr('x', legX + 22)
        .attr('y', legY + 4)
        .style('font-size', '10px')
        .style('fill', '#374151')
        .text(LABEL[s]);
      legY += 14;
    }
  }, [width, data, visible, argAic]);

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3 flex flex-col gap-1">
        <div className="text-sm font-semibold text-gray-900">
          CV vs IC comparator — Stone&apos;s equivalence on POLY_DGP
        </div>
        <div className="text-xs text-gray-600">
          Each curve over polynomial degree d is shifted to its own min so all
          five sit on a common visual scale. Stone&apos;s 1977 theorem: argmin
          of LOO-CV and argmin of AIC coincide asymptotically. Toggle curves to
          isolate any pair.
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        {SERIES.map((s) => (
          <label
            key={s}
            className="flex items-center gap-1 rounded border px-2 py-1 text-xs"
            style={{
              borderColor: informationCriteriaColors[PALETTE[s]],
              background: visible[s] ? '#FFFFFF' : '#F3F4F6',
              color: visible[s]
                ? informationCriteriaColors[PALETTE[s]]
                : '#9CA3AF',
            }}
          >
            <input
              type="checkbox"
              checked={visible[s]}
              onChange={() => toggle(s)}
              aria-label={`toggle ${LABEL[s]}`}
              className="h-3 w-3"
            />
            <span>{LABEL[s]}</span>
          </label>
        ))}
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={PANEL_H}
        style={{
          display: 'block',
          background: 'var(--color-viz-bg, #fff)',
          borderRadius: 6,
        }}
      />

      <div
        className="mt-3 grid gap-2 text-xs"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}
      >
        <div
          className="rounded border px-3 py-2"
          style={{ borderColor: 'var(--color-border)' }}
          aria-live="polite"
        >
          <div className="text-gray-600">
            Stone equivalence empirical gap
          </div>
          <div className="font-mono text-gray-900">
            |argmin(LOO-CV) − argmin(AIC)| ={' '}
            <span style={{ fontWeight: 600 }}>
              {Number.isFinite(stoneGap) ? stoneGap : 'n/a'}
            </span>
            {Number.isFinite(stoneGap) && stoneGap === 0 && (
              <span className="ml-1 text-emerald-700">(coincide)</span>
            )}
          </div>
          <div className="mt-1 text-[11px] text-gray-500">
            argmin(LOO) = d {Number.isFinite(argLoo) ? `= ${argLoo}` : 'undefined'};
            {' '}argmin(AIC) = d = {argAic}
          </div>
        </div>
        <div
          className="rounded border px-3 py-2"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div className="text-gray-600">CV-fold sensitivity check</div>
          <div className="font-mono text-gray-900">
            argmin(5-fold) = d = {data.argmins.cv5}
            {' · '}
            argmin(10-fold) = d = {data.argmins.cv10}
          </div>
          <div className="mt-1 text-[11px] text-gray-500">
            Same DGP, different fold counts. Agreement signals the CV estimate
            is fold-stable; disagreement signals high CV variance.
          </div>
        </div>
      </div>
    </div>
  );
}
