/**
 * ConsistencyEfficiencyRace — Topic 24 §24.6 component.
 *
 * Empirical demonstration of Yang's 2005 incompatibility theorem (Thm 5):
 * no procedure is both selection-consistent (BIC's property) and
 * minimax-rate-optimal for prediction (AIC / CV's property). Two tabs:
 *
 *   • Tab A (well-specified): truth = degree-3 polynomial. Selection
 *     frequency at the correct d=3 climbs to ≈ 1 for BIC as n grows;
 *     AIC and CV plateau lower (they keep selecting d ≥ 4 a fixed fraction
 *     of the time — asymptotic over-fit). Bar chart per criterion at the
 *     largest sample size in the sweep.
 *
 *   • Tab B (misspecified): truth = sin(2πx); candidate set is polynomials
 *     d ∈ [0, 12] (truth NOT in the candidate set). Mean prediction risk
 *     vs n. AIC and CV adapt the chosen degree to n (minimax-rate optimal);
 *     BIC over-shrinks toward sparser models, sustaining higher risk.
 *
 * Tab UI uses role="tablist" + role="tab" + aria-selected per WAI-ARIA.
 *
 * Data source: YANG_RACE_WELL and YANG_RACE_MIS from model-selection-data.ts
 * (precomputed via IIFE with 25 MC replicates per sample size; component
 * renders in O(1) on tab switch).
 */

import { useRef, useState, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { informationCriteriaColors } from './shared/colorScales';
import {
  YANG_RACE_WELL,
  YANG_RACE_MIS,
} from '../../data/model-selection-data';

const MARGIN = { top: 28, right: 120, bottom: 44, left: 56 };
const PANEL_H = 360;
const AXIS_COLOR = '#94A3B8';
const TICK_COLOR = '#CBD5E1';

type Tab = 'wellSpecified' | 'misspecified';
type Criterion = 'aic' | 'aicc' | 'bic' | 'cv10';
const CRITERIA: Criterion[] = ['aic', 'aicc', 'bic', 'cv10'];
const LABEL: Record<Criterion, string> = {
  aic: 'AIC',
  aicc: 'AICc',
  bic: 'BIC',
  cv10: '10-fold CV',
};
const PALETTE: Record<Criterion, 'aic' | 'aicc' | 'bic' | 'loo'> = {
  aic: 'aic',
  aicc: 'aicc',
  bic: 'bic',
  cv10: 'loo',
};

export default function ConsistencyEfficiencyRace() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);

  const [tab, setTab] = useState<Tab>('wellSpecified');

  const dataset = tab === 'wellSpecified' ? YANG_RACE_WELL : YANG_RACE_MIS;
  // Tab A's "primary view" is the selection-frequency bar chart at the largest
  // sample size in the sweep; Tab B's primary view is the prediction-risk
  // line chart over the full sweep.
  const largestN = dataset.nSweep[dataset.nSweep.length - 1];

  const svgRef = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const w = width;
    const h = PANEL_H;

    if (tab === 'wellSpecified') {
      // ── Tab A: grouped bar chart ─────────────────────────────────────────
      // x: candidate degrees; bars grouped per criterion at the largest n.
      const niMax = dataset.nSweep.length - 1;
      const degrees = dataset.candidateDegrees;
      const trueDeg = dataset.preset.trueCoefs!.length - 1; // = 3 for d3 truth
      const xScale = d3
        .scaleBand<number>()
        .domain(degrees as number[])
        .range([MARGIN.left, w - MARGIN.right])
        .padding(0.15);
      const groupScale = d3
        .scaleBand<string>()
        .domain(CRITERIA)
        .range([0, xScale.bandwidth()])
        .padding(0.05);
      const yScale = d3
        .scaleLinear()
        .domain([0, 1])
        .range([h - MARGIN.bottom, MARGIN.top]);
      // Axes
      svg
        .append('g')
        .attr('transform', `translate(0,${h - MARGIN.bottom})`)
        .call(
          d3
            .axisBottom(xScale)
            .tickFormat((d) => String(d)),
        )
        .selectAll('path,line')
        .style('stroke', AXIS_COLOR);
      svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},0)`)
        .call(
          d3
            .axisLeft(yScale)
            .ticks(5)
            .tickFormat((d) => `${(d as number) * 100}%`),
        )
        .selectAll('path,line')
        .style('stroke', AXIS_COLOR);
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', h - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', AXIS_COLOR)
        .text(`candidate degree (selected at n = ${largestN})`);
      svg
        .append('text')
        .attr('x', 12)
        .attr('y', h / 2)
        .attr('transform', `rotate(-90, 12, ${h / 2})`)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', AXIS_COLOR)
        .text('selection frequency');
      // True-degree highlight
      const trueX = xScale(trueDeg);
      if (trueX !== undefined) {
        svg
          .append('rect')
          .attr('x', trueX - 2)
          .attr('y', MARGIN.top)
          .attr('width', xScale.bandwidth() + 4)
          .attr('height', h - MARGIN.bottom - MARGIN.top)
          .attr('fill', '#FEF3C7')
          .attr('opacity', 0.5);
        svg
          .append('text')
          .attr('x', trueX + xScale.bandwidth() / 2)
          .attr('y', MARGIN.top - 8)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('font-weight', '600')
          .style('fill', '#92400E')
          .text('true d');
      }
      // Bars
      degrees.forEach((d, di) => {
        const x0 = xScale(d) ?? 0;
        CRITERIA.forEach((c) => {
          const freq = dataset.selectionFrequency[c][niMax][di] ?? 0;
          svg
            .append('rect')
            .attr('x', x0 + (groupScale(c) ?? 0))
            .attr('y', yScale(freq))
            .attr('width', groupScale.bandwidth())
            .attr('height', yScale(0) - yScale(freq))
            .style('fill', informationCriteriaColors[PALETTE[c]]);
        });
      });
      // Legend
      const legX = w - MARGIN.right + 6;
      let legY = MARGIN.top;
      CRITERIA.forEach((c) => {
        svg
          .append('rect')
          .attr('x', legX)
          .attr('y', legY - 6)
          .attr('width', 14)
          .attr('height', 10)
          .style('fill', informationCriteriaColors[PALETTE[c]]);
        svg
          .append('text')
          .attr('x', legX + 18)
          .attr('y', legY + 3)
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(LABEL[c]);
        legY += 14;
      });
    } else {
      // ── Tab B: prediction-risk line chart ────────────────────────────────
      // x: log10(n); y: mean test MSE per criterion.
      const nSweep = dataset.nSweep;
      const xScale = d3
        .scaleLog()
        .domain([nSweep[0], nSweep[nSweep.length - 1]])
        .range([MARGIN.left, w - MARGIN.right]);
      const allRisks = CRITERIA.flatMap((c) =>
        dataset.predictionRisk[c].filter(Number.isFinite),
      );
      const yMin = Math.min(...allRisks) * 0.9;
      const yMax = Math.max(...allRisks) * 1.05;
      const yScale = d3
        .scaleLinear()
        .domain([yMin, yMax])
        .range([h - MARGIN.bottom, MARGIN.top]);
      // Axes
      svg
        .append('g')
        .attr('transform', `translate(0,${h - MARGIN.bottom})`)
        .call(
          d3
            .axisBottom(xScale)
            .tickValues(nSweep as number[])
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
        .text('sample size n (log scale)');
      svg
        .append('text')
        .attr('x', 12)
        .attr('y', h / 2)
        .attr('transform', `rotate(-90, 12, ${h / 2})`)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', AXIS_COLOR)
        .text('mean test MSE');
      // Curves
      const lineGen = d3
        .line<{ x: number; y: number }>()
        .defined((d) => Number.isFinite(d.y))
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));
      CRITERIA.forEach((c) => {
        const pts = nSweep.map((n, i) => ({
          x: n,
          y: dataset.predictionRisk[c][i],
        }));
        svg
          .append('path')
          .datum(pts)
          .attr('fill', 'none')
          .style('stroke', informationCriteriaColors[PALETTE[c]])
          .attr('stroke-width', 2.5)
          .attr('d', lineGen);
        // Endpoint markers
        pts.forEach((p) => {
          if (!Number.isFinite(p.y)) return;
          svg
            .append('circle')
            .attr('cx', xScale(p.x))
            .attr('cy', yScale(p.y))
            .attr('r', 3)
            .style('fill', informationCriteriaColors[PALETTE[c]]);
        });
      });
      // Sweep guides
      nSweep.forEach((n) => {
        svg
          .append('line')
          .attr('x1', xScale(n))
          .attr('x2', xScale(n))
          .attr('y1', MARGIN.top)
          .attr('y2', h - MARGIN.bottom)
          .attr('stroke', TICK_COLOR)
          .attr('stroke-dasharray', '2 4')
          .attr('opacity', 0.4);
      });
      // Legend
      const legX = w - MARGIN.right + 6;
      let legY = MARGIN.top;
      CRITERIA.forEach((c) => {
        svg
          .append('line')
          .attr('x1', legX)
          .attr('x2', legX + 18)
          .attr('y1', legY)
          .attr('y2', legY)
          .style('stroke', informationCriteriaColors[PALETTE[c]])
          .attr('stroke-width', 2.5);
        svg
          .append('text')
          .attr('x', legX + 22)
          .attr('y', legY + 4)
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(LABEL[c]);
        legY += 14;
      });
    }
  }, [width, tab, dataset, largestN]);

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3 flex flex-col gap-1">
        <div className="text-sm font-semibold text-gray-900">
          Consistency-vs-efficiency race — Yang&apos;s incompatibility (Thm 5)
        </div>
        <div className="text-xs text-gray-600">
          {tab === 'wellSpecified'
            ? 'Truth is a degree-3 polynomial; the candidate set contains the truth. Selection frequency at the correct d climbs toward 1 for BIC as n grows; AIC and CV keep selecting d ≥ 4 a persistent fraction of the time (asymptotic over-fit).'
            : 'Truth is sin(2πx); candidate set is polynomials d ∈ [0, 12] (truth NOT in the candidate set). Mean test-MSE vs n: AIC and CV adapt the selected degree to n (minimax-rate optimal); BIC over-shrinks toward sparser models, sustaining higher risk.'}
        </div>
      </div>

      <div
        role="tablist"
        aria-label="Yang race scenario"
        className="mb-3 flex flex-wrap gap-2"
      >
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'wellSpecified'}
          onClick={() => setTab('wellSpecified')}
          className="rounded border px-3 py-1 text-xs"
          style={{
            borderColor: 'var(--color-border)',
            background: tab === 'wellSpecified' ? '#DBEAFE' : 'transparent',
            fontWeight: tab === 'wellSpecified' ? 600 : 400,
            color: tab === 'wellSpecified' ? '#1E40AF' : '#374151',
          }}
        >
          Tab A — well-specified (poly-d3)
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'misspecified'}
          onClick={() => setTab('misspecified')}
          className="rounded border px-3 py-1 text-xs"
          style={{
            borderColor: 'var(--color-border)',
            background: tab === 'misspecified' ? '#DBEAFE' : 'transparent',
            fontWeight: tab === 'misspecified' ? 600 : 400,
            color: tab === 'misspecified' ? '#1E40AF' : '#374151',
          }}
        >
          Tab B — misspecified (sin truth)
        </button>
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

      <div className="mt-2 text-[11px] text-gray-500">
        Precomputed at module load: {dataset.mcReps} MC replicates per sample
        size, sample-size sweep {dataset.nSweep.join(' / ')}, candidate degrees
        d ∈ [{dataset.candidateDegrees[0]},{' '}
        {dataset.candidateDegrees[dataset.candidateDegrees.length - 1]}].
      </div>
    </div>
  );
}
