import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { standardNormalCDF, npCriticalValue } from './shared/testing';
import { cdfBinomial, pmfBinomial } from './shared/distributions';
import {
  npLemmaPresets,
  type NPLemmaPreset,
} from '../../data/likelihood-ratio-tests-data';

/**
 * NPLemmaVisualizer — makes the NP lemma (§18.2) visual.
 *
 * Shows two overlaid densities f(x; θ₀) (null) and f(x; θ₁) (alternative),
 * shades the NP rejection region as a function of a threshold c on the
 * sufficient statistic, and reports live size / power readouts. The
 * threshold slider lets readers see the classic Type-I-vs-Power tradeoff
 * directly: moving c right lowers size AND lowers power; moving it left
 * raises both. The NP lemma says any other region with the same size has
 * at most the shown power.
 */

const MARGIN = { top: 14, right: 16, bottom: 40, left: 48 };
const H = 320;

const COLOR_NULL = '#6B7280';
const COLOR_ALT = '#DC2626';
const COLOR_SIZE = '#F59E0B';
const COLOR_POWER = '#10B981';

/** Evaluate PDF of the sampling distribution of the sufficient statistic T under θ. */
function samplingPDF(family: 'normal-mean-known-sigma' | 'bernoulli' | 'exponential', theta: number, n: number, sigma: number, t: number): number {
  if (family === 'normal-mean-known-sigma') {
    // T = x̄ ~ N(θ, σ²/n)
    const se = sigma / Math.sqrt(n);
    return Math.exp(-0.5 * Math.pow((t - theta) / se, 2)) / (se * Math.sqrt(2 * Math.PI));
  }
  if (family === 'bernoulli') {
    // T = ΣX ~ Binomial(n, θ); returned as a PMF value at integer t
    const k = Math.round(t);
    if (k < 0 || k > n) return 0;
    return pmfBinomial(k, n, theta);
  }
  if (family === 'exponential') {
    // T = ΣX ~ Gamma(n, θ); density = (θ^n / (n-1)!) t^{n-1} e^{-θ t}
    if (t <= 0) return 0;
    // log density = n log θ + (n-1) log t - θ t - lgamma(n)
    // We approximate lgamma(n) = log((n-1)!) via Stirling-refined closed form for small n
    let lgn = 0;
    for (let i = 2; i < n; i++) lgn += Math.log(i);
    const logP = n * Math.log(theta) + (n - 1) * Math.log(t) - theta * t - lgn;
    return Math.exp(logP);
  }
  return 0;
}

/** Right-tail probability P_θ(T > c) for each family. */
function rightTailProb(family: 'normal-mean-known-sigma' | 'bernoulli' | 'exponential', theta: number, n: number, sigma: number, c: number): number {
  if (family === 'normal-mean-known-sigma') {
    const se = sigma / Math.sqrt(n);
    return 1 - standardNormalCDF((c - theta) / se);
  }
  if (family === 'bernoulli') {
    // P(ΣX > c), with c possibly integer
    const k = Math.floor(c);
    if (k < 0) return 1;
    if (k >= n) return 0;
    return 1 - cdfBinomial(k, n, theta);
  }
  if (family === 'exponential') {
    // ΣX ~ Gamma(n, θ) ⇒ (2θΣX) ~ χ²_{2n}
    // P(ΣX > c) = P(χ²_{2n} > 2θc) = 1 - F_chi2_{2n}(2θc)
    // We reuse chiSquaredCDF via a small computed helper
    if (c <= 0) return 1;
    // Inline chiSquaredCDF: regGammaP(n, θc) via series/continued fraction
    // but easier: since samplingPDF handles the PDF, integrate numerically
    // for ≤ 20 steps from c to a large upper bound
    const upper = c + 30 / theta;
    const steps = 200;
    const dx = (upper - c) / steps;
    let s = 0;
    for (let i = 0; i < steps; i++) {
      const t = c + (i + 0.5) * dx;
      s += samplingPDF('exponential', theta, n, sigma, t) * dx;
    }
    return Math.min(1, Math.max(0, s));
  }
  return 0;
}

export default function NPLemmaVisualizer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 900;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: NPLemmaPreset = npLemmaPresets[presetIndex];

  const [theta0, setTheta0] = useState(preset.theta0);
  const [theta1, setTheta1] = useState(preset.theta1);
  const [n, setN] = useState(preset.n);
  const [alpha, setAlpha] = useState(preset.alpha);

  useEffect(() => {
    setTheta0(preset.theta0);
    setTheta1(preset.theta1);
    setN(preset.n);
    setAlpha(preset.alpha);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  const sigma = preset.sigma ?? 1;

  // The three families the NP presets actually expose (narrower than AsymptoticTestFamily).
  const narrowFamily = preset.family as 'normal-mean-known-sigma' | 'bernoulli' | 'exponential';

  // NP threshold at the currently selected α — this is the level-α NP test.
  const npInfo = useMemo(() => {
    return npCriticalValue(preset.family, theta0, theta1, n, alpha, preset.family === 'normal-mean-known-sigma' ? sigma : undefined);
  }, [preset.family, theta0, theta1, n, alpha, sigma]);

  // Size and power at the computed NP threshold
  const sizeAtK = useMemo(() => {
    if (narrowFamily === 'exponential' && theta1 > theta0) {
      // NP rejects for small T in this direction
      return 1 - rightTailProb(narrowFamily, theta0, n, sigma, npInfo.threshold);
    }
    return rightTailProb(narrowFamily, theta0, n, sigma, npInfo.threshold);
  }, [narrowFamily, theta0, theta1, n, sigma, npInfo.threshold]);
  const powerAtK = useMemo(() => {
    if (narrowFamily === 'exponential' && theta1 > theta0) {
      return 1 - rightTailProb(narrowFamily, theta1, n, sigma, npInfo.threshold);
    }
    return rightTailProb(narrowFamily, theta1, n, sigma, npInfo.threshold);
  }, [narrowFamily, theta0, theta1, n, sigma, npInfo.threshold]);

  // Chart
  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.62) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // Determine x-axis range based on family
    let xMin: number, xMax: number, nPoints: number, isDiscrete: boolean;
    if (narrowFamily === 'normal-mean-known-sigma') {
      const center = (theta0 + theta1) / 2;
      const spread = Math.max(4 * sigma / Math.sqrt(n), Math.abs(theta1 - theta0) * 2);
      xMin = center - spread;
      xMax = center + spread;
      nPoints = 200;
      isDiscrete = false;
    } else if (narrowFamily === 'bernoulli') {
      xMin = 0;
      xMax = n;
      nPoints = n + 1;
      isDiscrete = true;
    } else {
      // exponential: ΣX ~ Gamma
      xMin = 0;
      // Use the mean ± 4σ under both hypotheses to set the range
      const mean0 = n / theta0;
      const mean1 = n / theta1;
      const maxMean = Math.max(mean0, mean1);
      xMax = maxMean * 2;
      nPoints = 200;
      isDiscrete = false;
    }

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

    // Build two density arrays
    const step = (xMax - xMin) / (nPoints - 1);
    const nullPts: [number, number][] = [];
    const altPts: [number, number][] = [];
    for (let i = 0; i < nPoints; i++) {
      const t = isDiscrete ? i : xMin + i * step;
      nullPts.push([t, samplingPDF(narrowFamily, theta0, n, sigma, t)]);
      altPts.push([t, samplingPDF(narrowFamily, theta1, n, sigma, t)]);
    }

    const maxY =
      Math.max(
        ...nullPts.map(([, p]) => p),
        ...altPts.map(([, p]) => p),
      ) * 1.1;
    const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${H}`)
      .attr('width', svgW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(x).ticks(7));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')));

    // Rejection region shading
    const threshold = npInfo.threshold;

    const makeArea = (pts: [number, number][], rejectRange: [number, number]) => {
      const [lo, hi] = rejectRange;
      const inRange = pts.filter(([t]) => t >= lo && t <= hi);
      if (inRange.length === 0) return null;
      const area = d3
        .area<[number, number]>()
        .x((d) => x(d[0]))
        .y0(innerH)
        .y1((d) => y(d[1]))
        .curve(isDiscrete ? d3.curveStepAfter : d3.curveCatmullRom.alpha(0.3));
      return area(inRange);
    };

    const isLeftTailRejectLocal = narrowFamily === 'exponential' && theta1 > theta0;
    // Size region (under H₀ in the rejection region)
    const rejRange: [number, number] = isLeftTailRejectLocal
      ? [xMin, threshold]
      : [threshold, xMax];
    const sizeAreaPath = makeArea(nullPts, rejRange);
    if (sizeAreaPath) {
      g.append('path')
        .attr('d', sizeAreaPath)
        .attr('fill', COLOR_SIZE)
        .attr('fill-opacity', 0.4);
    }
    const powerAreaPath = makeArea(altPts, rejRange);
    if (powerAreaPath) {
      g.append('path')
        .attr('d', powerAreaPath)
        .attr('fill', COLOR_POWER)
        .attr('fill-opacity', 0.4);
    }

    // Density lines
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(isDiscrete ? d3.curveStepAfter : d3.curveCatmullRom.alpha(0.3));
    g.append('path')
      .datum(nullPts)
      .attr('d', line)
      .attr('stroke', COLOR_NULL)
      .attr('stroke-width', 2)
      .attr('fill', 'none');
    g.append('path')
      .datum(altPts)
      .attr('d', line)
      .attr('stroke', COLOR_ALT)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,3')
      .attr('fill', 'none');

    // Threshold line
    g.append('line')
      .attr('x1', x(threshold))
      .attr('x2', x(threshold))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', 'currentColor')
      .attr('stroke-dasharray', '3,2')
      .attr('stroke-opacity', 0.7);
    g.append('text')
      .attr('x', x(threshold))
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .text(`c = ${threshold.toFixed(2)}`);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 170}, 0)`);
    const legItems: [string, string, boolean][] = [
      ['f(x; θ₀) — null', COLOR_NULL, false],
      ['f(x; θ₁) — alt', COLOR_ALT, true],
      ['size (Type I)', COLOR_SIZE, false],
      ['power', COLOR_POWER, false],
    ];
    legItems.forEach(([label, color, dashed], i) => {
      if (label.includes('size') || label.includes('power')) {
        legend
          .append('rect')
          .attr('y', i * 14)
          .attr('width', 10)
          .attr('height', 10)
          .attr('fill', color)
          .attr('fill-opacity', 0.4);
      } else {
        legend
          .append('line')
          .attr('x1', 0)
          .attr('x2', 10)
          .attr('y1', i * 14 + 6)
          .attr('y2', i * 14 + 6)
          .attr('stroke', color)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', dashed ? '4,3' : null);
      }
      legend
        .append('text')
        .attr('x', 14)
        .attr('y', i * 14 + 9)
        .style('font-size', '10px')
        .attr('fill', 'currentColor')
        .text(label);
    });

    // X axis label
    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text(
        narrowFamily === 'normal-mean-known-sigma'
          ? 'sample mean x̄'
          : 'sufficient statistic T = ΣXᵢ',
      );
  }, [narrowFamily, theta0, theta1, n, sigma, npInfo.threshold, svgW]);

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
          Neyman-Pearson Lemma · §18.2
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
          {npLemmaPresets.map((p, i) => (
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
            max={preset.family === 'bernoulli' ? 0.95 : preset.family === 'normal-mean-known-sigma' ? 2 : 3}
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
            step={preset.theta1Step}
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

      <div className={isWide ? 'grid grid-cols-[62%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <svg ref={svgRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-3">
          <div
            className="text-xs uppercase font-bold tracking-wide"
            style={{ color: 'var(--color-text-muted)' }}
          >
            NP Test — Level α
          </div>
          <div
            className="text-[11px] space-y-1 p-2 rounded font-mono"
            style={{ background: 'var(--color-input-bg)' }}
          >
            <div>threshold c = {npInfo.threshold.toFixed(3)}</div>
            <div>T = {npInfo.Tform}</div>
            <div>exact size = {npInfo.exactSize.toFixed(4)}</div>
          </div>
          <div className="pt-2 space-y-2">
            <div className="flex items-center justify-between">
              <span style={{ color: 'var(--color-text-muted)' }}>size α:</span>
              <strong style={{ color: COLOR_SIZE }}>{sizeAtK.toFixed(4)}</strong>
            </div>
            <div className="flex items-center justify-between">
              <span style={{ color: 'var(--color-text-muted)' }}>power β(θ₁):</span>
              <strong style={{ color: COLOR_POWER }}>{powerAtK.toFixed(4)}</strong>
            </div>
          </div>
          <div className="text-[11px] pt-2" style={{ color: 'var(--color-text-muted)' }}>
            By the NP lemma (Theorem 1), no other level-α test at θ₁ has
            higher power than the shown region.
          </div>
        </div>
      </div>
    </div>
  );
}
