/**
 * ICSelector — Topic 24 §24.3 featured component.
 *
 * Polynomial-degree explorer with five-criterion live ranking on the canonical
 * POLY_DGP. Mirrors Topic 23's RegularizationPathExplorer consumption pattern:
 * the precomputed data object (POLY_DGP_PRECOMPUTED in model-selection-data.ts)
 * is materialized once at module load via an IIFE; this component imports it
 * verbatim and re-renders in O(1) on slider / toggle changes.
 *
 * Panels:
 *   • Left:  scatter of (x, y) with the selected-degree fit overlaid; optional
 *            "show true sin(2πx)" toggle (aria-pressed).
 *   • Right: AIC, AICc, BIC, Mallows' Cp, and 10-fold CV curves over d ∈ [0, 12],
 *            each shifted so its minimum is 0 (preserves argmin and shape, makes
 *            relative differences comparable across criteria with different
 *            absolute scales). Argmin markers per curve.
 *   • Below: per-criterion argmin readout. The pattern AIC/AICc/Cp/CV agree on
 *            d=6 while BIC alone selects d=3 is the canonical Yang signature.
 */

import { useRef, useState, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { polyEval } from './shared/polynomial';
import {
  informationCriteriaColors,
  informationCriteriaLineStyles,
} from './shared/colorScales';
import { POLY_DGP_PRECOMPUTED } from '../../data/model-selection-data';

const MARGIN = { top: 24, right: 96, bottom: 44, left: 56 };
const PANEL_H = 320;
const AXIS_COLOR = '#94A3B8';
const TICK_COLOR = '#CBD5E1';
const TRUE_COLOR = '#dc2626';
const FIT_COLOR = '#0891b2';

type Criterion = 'aic' | 'aicc' | 'bic' | 'cp' | 'loo';
const CRITERIA: Criterion[] = ['aic', 'aicc', 'bic', 'cp', 'loo'];
const CRITERION_LABEL: Record<Criterion, string> = {
  aic: 'AIC',
  aicc: 'AICc',
  bic: 'BIC',
  cp: 'Cp',
  loo: '10-fold CV',
};

export default function ICSelector() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 720;
  const panelW = isStacked ? width - 16 : (width - 24) / 2;

  const data = POLY_DGP_PRECOMPUTED;
  const minDeg = data.degrees[0];
  const maxDeg = data.degrees[data.degrees.length - 1];

  const [selectedDegree, setSelectedDegree] = useState(6);
  const [showTrue, setShowTrue] = useState(true);

  const selectedFit = data.fits[selectedDegree - minDeg];

  // ── Left panel: scatter + fit ────────────────────────────────────────────
  const leftRef = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    if (!leftRef.current) return;
    const svg = d3.select(leftRef.current);
    svg.selectAll('*').remove();
    const w = panelW;
    const h = PANEL_H;
    const xScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([MARGIN.left, w - MARGIN.right]);
    const yMin = d3.min(data.y) ?? -1.5;
    const yMax = d3.max(data.y) ?? 1.5;
    const yPad = 0.1 * (yMax - yMin);
    const yScale = d3
      .scaleLinear()
      .domain([yMin - yPad, yMax + yPad])
      .range([h - MARGIN.bottom, MARGIN.top]);
    svg
      .append('g')
      .attr('transform', `translate(0,${h - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('path,line')
      .style('stroke', AXIS_COLOR);
    svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('path,line')
      .style('stroke', AXIS_COLOR);
    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', h - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', AXIS_COLOR)
      .text('x');
    svg
      .append('text')
      .attr('x', 12)
      .attr('y', h / 2)
      .attr('transform', `rotate(-90, 12, ${h / 2})`)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', AXIS_COLOR)
      .text('y');
    // Data
    svg
      .selectAll('circle.pt')
      .data(data.x.map((xi, i) => ({ x: xi, y: data.y[i] })))
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', 3)
      .attr('fill', '#64748B')
      .attr('opacity', 0.5);
    // True overlay
    if (showTrue) {
      const trueLine = d3
        .line<number>()
        .x((u) => xScale(u))
        .y((u) => yScale(Math.sin(2 * Math.PI * u)))
        .curve(d3.curveBasis);
      const trueGrid = d3.range(0, 1.001, 0.005);
      svg
        .append('path')
        .datum(trueGrid)
        .attr('fill', 'none')
        .attr('stroke', TRUE_COLOR)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 2')
        .attr('opacity', 0.85)
        .attr('d', trueLine);
    }
    // Fit
    const fitLine = d3
      .line<number>()
      .x((u) => xScale(u))
      .y((u) => yScale(polyEval(selectedFit.beta, u, selectedDegree)))
      .curve(d3.curveBasis);
    const fitGrid = d3.range(0, 1.001, 0.005);
    svg
      .append('path')
      .datum(fitGrid)
      .attr('fill', 'none')
      .attr('stroke', FIT_COLOR)
      .attr('stroke-width', 2.5)
      .attr('d', fitLine);
    // Legend
    const legX = w - MARGIN.right + 6;
    const legY = MARGIN.top + 4;
    if (showTrue) {
      svg
        .append('line')
        .attr('x1', legX)
        .attr('x2', legX + 16)
        .attr('y1', legY)
        .attr('y2', legY)
        .attr('stroke', TRUE_COLOR)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4 2');
      svg
        .append('text')
        .attr('x', legX + 20)
        .attr('y', legY + 4)
        .style('font-size', '10px')
        .style('fill', '#374151')
        .text('truth');
    }
    svg
      .append('line')
      .attr('x1', legX)
      .attr('x2', legX + 16)
      .attr('y1', legY + 16)
      .attr('y2', legY + 16)
      .attr('stroke', FIT_COLOR)
      .attr('stroke-width', 2.5);
    svg
      .append('text')
      .attr('x', legX + 20)
      .attr('y', legY + 20)
      .style('font-size', '10px')
      .style('fill', '#374151')
      .text(`d=${selectedDegree}`);
  }, [panelW, data, selectedFit, selectedDegree, showTrue]);

  // ── Right panel: criterion curves ────────────────────────────────────────
  const rightRef = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    if (!rightRef.current) return;
    const svg = d3.select(rightRef.current);
    svg.selectAll('*').remove();
    const w = panelW;
    const h = PANEL_H;
    const xScale = d3
      .scaleLinear()
      .domain([minDeg, maxDeg])
      .range([MARGIN.left, w - MARGIN.right]);
    // Each criterion uses its own y-scale to fit on a common plot. Standardize
    // by shifting each curve's min to 0; this preserves argmin and shape but
    // makes the relative differences visible regardless of absolute scale.
    const series: { c: Criterion; values: number[]; shifted: number[] }[] =
      CRITERIA.map((c) => {
        const v = data.values[c];
        const m = Math.min(...v);
        return { c, values: v, shifted: v.map((x) => x - m) };
      });
    const yMax = d3.max(series.flatMap((s) => s.shifted)) ?? 10;
    const yScale = d3
      .scaleLinear()
      .domain([0, yMax])
      .range([h - MARGIN.bottom, MARGIN.top]);
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
      .call(d3.axisLeft(yScale).ticks(5))
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
      .text('IC − min(IC)');
    // Selected-degree vertical guide
    svg
      .append('line')
      .attr('x1', xScale(selectedDegree))
      .attr('x2', xScale(selectedDegree))
      .attr('y1', MARGIN.top)
      .attr('y2', h - MARGIN.bottom)
      .attr('stroke', TICK_COLOR)
      .attr('stroke-dasharray', '3 3');
    // Curves + argmin markers
    const lineGen = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y));
    series.forEach((s) => {
      const pts = data.degrees.map((deg, i) => ({ x: deg, y: s.shifted[i] }));
      svg
        .append('path')
        .datum(pts)
        .attr('fill', 'none')
        .style('stroke', informationCriteriaColors[s.c])
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', informationCriteriaLineStyles[s.c])
        .attr('d', lineGen);
      // Argmin marker
      const argIdx = s.shifted.indexOf(0);
      const argDeg = data.degrees[argIdx];
      svg
        .append('circle')
        .attr('cx', xScale(argDeg))
        .attr('cy', yScale(0))
        .attr('r', 4)
        .style('fill', informationCriteriaColors[s.c]);
    });
    // Legend
    const legX = w - MARGIN.right + 6;
    let legY = MARGIN.top;
    series.forEach((s) => {
      svg
        .append('line')
        .attr('x1', legX)
        .attr('x2', legX + 16)
        .attr('y1', legY)
        .attr('y2', legY)
        .style('stroke', informationCriteriaColors[s.c])
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', informationCriteriaLineStyles[s.c]);
      svg
        .append('text')
        .attr('x', legX + 20)
        .attr('y', legY + 4)
        .style('font-size', '10px')
        .style('fill', '#374151')
        .text(CRITERION_LABEL[s.c]);
      legY += 14;
    });
  }, [panelW, data, selectedDegree, minDeg, maxDeg]);

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3 flex flex-col gap-1">
        <div className="text-sm font-semibold text-gray-900">
          IC selector — POLY_DGP (sin(2πx) + N(0, 0.25²), n=80)
        </div>
        <div className="text-xs text-gray-600">
          Slide to highlight a polynomial degree; the right panel shows AIC, AICc,
          BIC, Mallows&apos; Cp, and 10-fold CV (each curve shifted so its min is 0
          for comparability of shape). Argmin markers show each criterion&apos;s
          choice.
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex flex-col text-xs text-gray-700">
          <span>
            highlighted degree d ={' '}
            <span className="font-mono">{selectedDegree}</span>
          </span>
          <input
            type="range"
            min={minDeg}
            max={maxDeg}
            step={1}
            value={selectedDegree}
            onChange={(e) => setSelectedDegree(parseInt(e.target.value, 10))}
            aria-label="polynomial degree"
            className="w-72"
          />
        </label>
        <button
          type="button"
          aria-pressed={showTrue}
          onClick={() => setShowTrue((s) => !s)}
          className="rounded border px-2 py-1 text-xs"
          style={{
            borderColor: 'var(--color-border)',
            background: showTrue ? '#fee2e2' : 'transparent',
            color: showTrue ? '#991B1B' : '#374151',
          }}
        >
          {showTrue ? '✓ ' : ''}show true sin(2πx)
        </button>
      </div>

      <div
        className="flex flex-wrap gap-3"
        style={{ flexDirection: isStacked ? 'column' : 'row' }}
      >
        <svg
          ref={leftRef}
          width={panelW}
          height={PANEL_H}
          style={{
            display: 'block',
            background: 'var(--color-viz-bg, #fff)',
            borderRadius: 6,
          }}
        />
        <svg
          ref={rightRef}
          width={panelW}
          height={PANEL_H}
          style={{
            display: 'block',
            background: 'var(--color-viz-bg, #fff)',
            borderRadius: 6,
          }}
        />
      </div>

      <div
        className="mt-3 grid gap-2 text-xs"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}
      >
        {CRITERIA.map((c) => (
          <div
            key={c}
            className="flex items-center justify-between rounded border px-2 py-1"
            style={{
              borderColor: informationCriteriaColors[c],
              background: '#FFFFFF',
            }}
          >
            <span
              style={{ color: informationCriteriaColors[c], fontWeight: 600 }}
            >
              {CRITERION_LABEL[c]} argmin
            </span>
            <span className="font-mono" style={{ color: '#111827' }}>
              d = {data.argmins[c]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
