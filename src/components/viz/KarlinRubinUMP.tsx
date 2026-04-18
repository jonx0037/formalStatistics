import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  umpOneSidedBoundary,
  standardNormalCDF,
  chiSquaredCDF,
} from './shared/testing';
import { cdfBinomial, cdfPoisson } from './shared/distributions';
import {
  karlinRubinPresets,
  type KarlinRubinPreset,
} from '../../data/likelihood-ratio-tests-data';

/**
 * KarlinRubinUMP — visualizes the MLR-UMP pipeline of §18.3.
 *
 * Left chart: log-likelihood ratio log[f(x;θ₁)/f(x;θ₀)] plotted against the
 * sufficient statistic T(x). The line is visibly non-decreasing for
 * exponential-family scenarios — that IS MLR. The horizontal line at the
 * NP threshold separates the accept region (low T) from the reject region
 * (high T), showing how MLR converts the LR threshold to a T threshold.
 *
 * Right chart: the power function β(θ) across θ space. It crosses α at
 * θ = θ₀ and rises monotonically. The key Karlin-Rubin payoff is that the
 * rejection region — the vertical dashed line on the left chart — does NOT
 * depend on θ₁: the same test is MP at every alternative simultaneously.
 */

const MARGIN = { top: 14, right: 16, bottom: 40, left: 48 };
const H = 300;

const COLOR_LR = '#2563EB';
const COLOR_BOUNDARY = '#DC2626';
const COLOR_POWER = '#059669';
const COLOR_SIZE = '#F59E0B';

/** Compute β(θ) for the one-sided right-tail UMP test on the given family.
 *  The boundary is fixed (computed once at θ₀); this function only varies θ. */
function powerAt(
  family: KarlinRubinPreset['family'],
  theta: number,
  n: number,
  sigma: number,
  boundary: number,
): number {
  if (family === 'normal-mean-known-sigma') {
    // β(μ) = 1 − Φ((c − μ)/(σ/√n)), c = θ₀ + z_{1-α} σ/√n
    const se = sigma / Math.sqrt(n);
    return 1 - standardNormalCDF((boundary - theta) / se);
  }
  if (family === 'bernoulli') {
    // β(p) = P_p(ΣX ≥ boundary) = 1 − F_Bin(boundary − 1; n, p)
    if (boundary <= 0) return 1;
    if (boundary > n) return 0;
    return 1 - cdfBinomial(boundary - 1, n, theta);
  }
  if (family === 'poisson') {
    // β(λ) = 1 − F_Poi(boundary − 1; nλ)
    if (boundary <= 0) return 1;
    return 1 - cdfPoisson(boundary - 1, n * theta);
  }
  if (family === 'exponential') {
    // β(λ) = P_λ(ΣX > boundary). Using (2λ·ΣX) ~ χ²_{2n}:
    //   P(ΣX > c) = 1 − F_χ²_{2n}(2λ·c).
    // Exact via the shared chi-squared CDF (PR #20 Gemini review).
    if (boundary <= 0) return 1;
    return 1 - chiSquaredCDF(2 * theta * boundary, 2 * n);
  }
  return 0;
}

export default function KarlinRubinUMP() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 900;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: KarlinRubinPreset = karlinRubinPresets[presetIndex];

  const [theta0, setTheta0] = useState(preset.theta0);
  const [theta1, setTheta1] = useState(preset.theta1Default);
  const [n, setN] = useState(preset.n);
  const [alpha, setAlpha] = useState(preset.alpha);

  useEffect(() => {
    setTheta0(preset.theta0);
    setTheta1(preset.theta1Default);
    setN(preset.n);
    setAlpha(preset.alpha);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  const sigma = (preset as { sigma?: number }).sigma ?? 1;

  // UMP boundary — one-sided right (standard convention; exp-family is flipped externally)
  const umpInfo = useMemo(() => {
    // For exponential with theta1 < theta0 (the preset), MLR is reversed ⇒ right-tail.
    // The data file's `mlrDirection` tells us which side to pass.
    const side: 'left' | 'right' =
      preset.mlrDirection === 'increasing-in-T' ? 'right' : 'left';
    return umpOneSidedBoundary(
      preset.family,
      theta0,
      n,
      alpha,
      side,
      preset.family === 'normal-mean-known-sigma' ? sigma : undefined,
    );
  }, [preset, theta0, n, alpha, sigma]);

  // Left chart: log-LR as function of T
  const lrChartRef = useRef<SVGSVGElement | null>(null);
  const leftW = isWide ? Math.floor(w * 0.50) : w;

  useEffect(() => {
    const svg = d3.select(lrChartRef.current);
    svg.selectAll('*').remove();

    const innerW = leftW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // Closed-form log Λ(x) expressed directly in the sufficient statistic T.
    // This replaces an earlier array-allocation + logLikelihoodRatio dispatch that
    // was O(n) per grid point × 80 grid points per render (PR #20 Gemini review).
    let tValues: number[];
    let logLRvals: number[];
    const logRatioAtT = (t: number) => {
      if (preset.family === 'bernoulli') {
        // T = ΣXᵢ ∈ {0,…,n}: log Λ = T·log(p₁/p₀) + (n−T)·log((1−p₁)/(1−p₀))
        return (
          t * Math.log(theta1 / theta0) +
          (n - t) * Math.log((1 - theta1) / (1 - theta0))
        );
      }
      if (preset.family === 'normal-mean-known-sigma') {
        // T = x̄: log Λ = n(θ₁−θ₀)(2x̄ − θ₀ − θ₁) / (2σ²)
        return (
          (n * (theta1 - theta0) * (2 * t - theta0 - theta1)) /
          (2 * sigma * sigma)
        );
      }
      if (preset.family === 'poisson') {
        // T = ΣXᵢ ~ Poisson(n·λ): log Λ = n(λ₀−λ₁) + T·log(λ₁/λ₀)
        return n * (theta0 - theta1) + t * Math.log(theta1 / theta0);
      }
      if (preset.family === 'exponential') {
        // T = ΣXᵢ ~ Gamma(n, λ): log Λ = n·log(λ₁/λ₀) − (λ₁−λ₀)·T
        return n * Math.log(theta1 / theta0) - (theta1 - theta0) * t;
      }
      return 0;
    };

    if (preset.family === 'bernoulli') {
      // T = ΣX ∈ {0, ..., n}
      tValues = Array.from({ length: n + 1 }, (_, i) => i);
      logLRvals = tValues.map(logRatioAtT);
    } else if (preset.family === 'normal-mean-known-sigma') {
      // T = x̄ on a range around [θ₀, θ₁]
      const span = Math.max(Math.abs(theta1 - theta0) * 3, 4 * sigma / Math.sqrt(n));
      const mid = (theta0 + theta1) / 2;
      tValues = Array.from({ length: 80 }, (_, i) => mid - span + (2 * span * i) / 79);
      logLRvals = tValues.map(logRatioAtT);
    } else if (preset.family === 'poisson') {
      // T = ΣX ~ Poisson(nθ)
      const maxT = Math.ceil(n * Math.max(theta0, theta1) + 4 * Math.sqrt(n * Math.max(theta0, theta1)));
      tValues = Array.from({ length: Math.min(maxT + 1, 80) }, (_, i) => (i * maxT) / 79);
      logLRvals = tValues.map(logRatioAtT);
    } else {
      // exponential: T = ΣX > 0
      const mean0 = n / theta0;
      const mean1 = n / theta1;
      const maxT = Math.max(mean0, mean1) * 2.5;
      tValues = Array.from({ length: 80 }, (_, i) => (i * maxT) / 79 + 1e-3);
      logLRvals = tValues.map(logRatioAtT);
    }

    const xExt = d3.extent(tValues) as [number, number];
    const yExt = d3.extent(logLRvals) as [number, number];
    const xSc = d3.scaleLinear().domain(xExt).range([0, innerW]);
    const ySc = d3.scaleLinear().domain([yExt[0], yExt[1] * 1.05]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${leftW} ${H}`)
      .attr('width', leftW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xSc).ticks(6));
    g.append('g').call(d3.axisLeft(ySc).ticks(5));

    const line = d3
      .line<number>()
      .x((_, i) => xSc(tValues[i]))
      .y((d) => ySc(d))
      .curve(d3.curveMonotoneX);
    g.append('path')
      .datum(logLRvals)
      .attr('d', line)
      .attr('stroke', COLOR_LR)
      .attr('stroke-width', 2)
      .attr('fill', 'none');

    // UMP boundary line
    if (Number.isFinite(umpInfo.boundary)) {
      g.append('line')
        .attr('x1', xSc(umpInfo.boundary))
        .attr('x2', xSc(umpInfo.boundary))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLOR_BOUNDARY)
        .attr('stroke-dasharray', '4,3');
      g.append('text')
        .attr('x', xSc(umpInfo.boundary))
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .attr('fill', COLOR_BOUNDARY)
        .text(`UMP boundary: T = ${umpInfo.boundary.toFixed(preset.family === 'bernoulli' ? 0 : 2)}`);
    }

    // Labels
    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text(`T = ${preset.Tform}`);
    svg
      .append('text')
      .attr('x', 4)
      .attr('y', MARGIN.top - 2)
      .attr('text-anchor', 'start')
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('log Λ(x) — monotone in T is MLR');
  }, [preset, theta0, theta1, n, sigma, umpInfo.boundary, leftW]);

  // Right chart: power function β(θ)
  const powerRef = useRef<SVGSVGElement | null>(null);
  const rightW = isWide ? Math.floor(w * 0.48) : w;

  useEffect(() => {
    const svg = d3.select(powerRef.current);
    svg.selectAll('*').remove();

    const innerW = rightW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // θ range: Bernoulli (0.05, 0.95), Normal (θ₀ − 2, θ₀ + 2), Poisson (θ₀*0.3, θ₀*3), Exponential (θ₀*0.3, θ₀*3)
    let thetaMin: number, thetaMax: number;
    if (preset.family === 'bernoulli') {
      thetaMin = 0.05;
      thetaMax = 0.95;
    } else if (preset.family === 'normal-mean-known-sigma') {
      thetaMin = theta0 - 2;
      thetaMax = theta0 + 2;
    } else if (preset.family === 'poisson') {
      thetaMin = Math.max(0.1, theta0 * 0.3);
      thetaMax = theta0 * 3;
    } else {
      thetaMin = Math.max(0.05, theta0 * 0.3);
      thetaMax = theta0 * 3;
    }

    const nPoints = 60;
    const thetaSeq = Array.from({ length: nPoints }, (_, i) => thetaMin + (i * (thetaMax - thetaMin)) / (nPoints - 1));
    const powerSeq = thetaSeq.map((t) => powerAt(preset.family, t, n, sigma, umpInfo.boundary));

    const xSc = d3.scaleLinear().domain([thetaMin, thetaMax]).range([0, innerW]);
    const ySc = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${rightW} ${H}`)
      .attr('width', rightW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xSc).ticks(6));
    g.append('g').call(d3.axisLeft(ySc).ticks(5).tickFormat(d3.format('.1f')));

    // α reference line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', ySc(alpha))
      .attr('y2', ySc(alpha))
      .attr('stroke', COLOR_SIZE)
      .attr('stroke-dasharray', '3,3')
      .attr('stroke-opacity', 0.6);
    g.append('text')
      .attr('x', 6)
      .attr('y', ySc(alpha) - 3)
      .style('font-size', '10px')
      .attr('fill', COLOR_SIZE)
      .text(`α = ${alpha.toFixed(3)}`);

    // θ₀ line
    g.append('line')
      .attr('x1', xSc(theta0))
      .attr('x2', xSc(theta0))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', 'currentColor')
      .attr('stroke-dasharray', '2,2')
      .attr('stroke-opacity', 0.5);
    g.append('text')
      .attr('x', xSc(theta0))
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .text(`θ₀ = ${theta0.toFixed(2)}`);

    // Power curve
    const line = d3
      .line<number>()
      .x((_, i) => xSc(thetaSeq[i]))
      .y((v) => ySc(v))
      .curve(d3.curveMonotoneX);
    g.append('path')
      .datum(powerSeq)
      .attr('d', line)
      .attr('stroke', COLOR_POWER)
      .attr('stroke-width', 2)
      .attr('fill', 'none');

    // θ₁ marker
    const thetaInRange = theta1 >= thetaMin && theta1 <= thetaMax;
    if (thetaInRange) {
      const powerAtTheta1 = powerAt(preset.family, theta1, n, sigma, umpInfo.boundary);
      g.append('circle')
        .attr('cx', xSc(theta1))
        .attr('cy', ySc(powerAtTheta1))
        .attr('r', 4)
        .attr('fill', COLOR_POWER);
      g.append('text')
        .attr('x', xSc(theta1) + 6)
        .attr('y', ySc(powerAtTheta1) - 4)
        .style('font-size', '10px')
        .attr('fill', COLOR_POWER)
        .text(`β(θ₁) = ${powerAtTheta1.toFixed(3)}`);
    }

    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('θ');
    svg
      .append('text')
      .attr('x', 4)
      .attr('y', MARGIN.top - 2)
      .attr('text-anchor', 'start')
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('power β(θ) — UMP at every θ > θ₀');
  }, [preset, theta0, theta1, n, sigma, alpha, umpInfo.boundary, rightW]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div
          className="text-xs uppercase tracking-wide font-bold"
          style={{ color: 'var(--color-text-muted)' }}
        >
          Karlin-Rubin UMP · §18.3
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{
            background: 'var(--color-input-bg)',
            borderColor: 'var(--color-border)',
            color: 'var(--color-text)',
          }}
        >
          {karlinRubinPresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            θ₀ = <strong>{theta0.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={preset.family === 'bernoulli' ? 0.05 : preset.family === 'normal-mean-known-sigma' ? -2 : 0.1}
            max={preset.family === 'bernoulli' ? 0.95 : preset.family === 'normal-mean-known-sigma' ? 2 : 5}
            step={preset.family === 'bernoulli' ? 0.01 : 0.05}
            value={theta0}
            onChange={(e) => setTheta0(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            θ₁ = <strong>{theta1.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={preset.theta1Range[0]}
            max={preset.theta1Range[1]}
            step={preset.family === 'bernoulli' ? 0.01 : 0.05}
            value={theta1}
            onChange={(e) => setTheta1(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
          </span>
          <input
            type="range"
            min={10}
            max={200}
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
            min={0.005}
            max={0.2}
            step={0.005}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
        </label>
      </div>

      <div className={isWide ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-4'}>
        <svg ref={lrChartRef} style={{ display: 'block' }} />
        <svg ref={powerRef} style={{ display: 'block' }} />
      </div>

      <div
        className="mt-3 text-[11px] p-2 rounded font-mono"
        style={{ background: 'var(--color-input-bg)' }}
      >
        UMP rejection region: T &gt; c. Current boundary c = {umpInfo.boundary.toFixed(preset.family === 'bernoulli' ? 0 : 3)},
        T form = {umpInfo.Tform}, exact size = {umpInfo.exactSize.toFixed(4)}. The
        boundary does <strong>not</strong> depend on θ₁ — this is what makes the
        test UMP.
      </div>
    </div>
  );
}
