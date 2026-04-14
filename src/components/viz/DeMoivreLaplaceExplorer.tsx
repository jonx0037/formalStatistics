import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBinomial,
  cdfBinomial,
  pdfStdNormal,
  cdfStdNormal,
} from './shared/distributions';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 18, bottom: 38, left: 48 };
const CHART_H = 280;

const BAR_COLOR = '#7C3AED'; // violet
const NORMAL_COLOR = '#059669'; // emerald
const CDF_BINOMIAL = '#7C3AED';
const CDF_NORMAL = '#059669';
const MAX_DIFF_COLOR = '#DC2626';

// ── Component ───────────────────────────────────────────────────────────────

function DeMoivreLaplaceExplorer() {
  const [n, setN] = useState<number>(40);
  const [p, setP] = useState<number>(0.3);
  const [continuity, setContinuity] = useState<boolean>(false);
  const [animating, setAnimating] = useState<boolean>(false);
  const animTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!animating) return;
    animTimer.current = window.setInterval(() => {
      setN((prev) => {
        if (prev >= 200) {
          setAnimating(false);
          return prev;
        }
        return Math.min(200, prev + 5);
      });
    }, 250);
    return () => {
      if (animTimer.current !== null) window.clearInterval(animTimer.current);
    };
  }, [animating]);

  const mu = useMemo(() => n * p, [n, p]);
  const sigma = useMemo(() => Math.sqrt(n * p * (1 - p)), [n, p]);

  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(260, width);
  const twoCol = width >= 720;
  const panelW = twoCol ? (w - 12) / 2 : w;

  // Compute max |F_Bin − Φ| for the readout.
  const maxDiff = useMemo(() => {
    if (sigma === 0) return 0;
    let d = 0;
    for (let k = 0; k <= n; k++) {
      const bin = cdfBinomial(k, n, p);
      const z = continuity ? (k + 0.5 - mu) / sigma : (k - mu) / sigma;
      const phi = cdfStdNormal(z);
      const diff = Math.abs(bin - phi);
      if (diff > d) d = diff;
    }
    return d;
  }, [n, p, mu, sigma, continuity]);

  return (
    <div
      className="my-8 rounded-lg border p-5"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            n = {n}
          </span>
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
          />
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            p = {p.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={p}
            onChange={(e) => setP(parseFloat(e.target.value))}
          />
        </label>
        <div className="flex items-center justify-between gap-2">
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={continuity}
              onChange={(e) => setContinuity(e.target.checked)}
            />
            <span className="font-medium uppercase tracking-wide opacity-70">
              Continuity correction
            </span>
          </label>
          <button
            type="button"
            onClick={() => setAnimating(!animating)}
            className="rounded border px-3 py-1 text-xs font-medium"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {animating ? '■ Stop' : '▶ Animate n'}
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        className={twoCol ? 'mt-4 grid grid-cols-2 gap-3' : 'mt-4 flex flex-col gap-4'}
      >
        <PMFPanel width={panelW} n={n} p={p} mu={mu} sigma={sigma} />
        <CDFPanel
          width={panelW}
          n={n}
          p={p}
          mu={mu}
          sigma={sigma}
          continuity={continuity}
        />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        <Stat label="np" value={mu.toFixed(2)} />
        <Stat label="√(np(1−p))" value={sigma.toFixed(3)} />
        <Stat
          label="max|F_Bin − Φ|"
          value={maxDiff.toFixed(4)}
          color={MAX_DIFF_COLOR}
        />
        <Stat
          label="continuity"
          value={continuity ? 'on' : 'off'}
          color={continuity ? NORMAL_COLOR : undefined}
        />
      </div>
    </div>
  );
}

// ── Panels ─────────────────────────────────────────────────────────────────

function PMFPanel(props: {
  width: number;
  n: number;
  p: number;
  mu: number;
  sigma: number;
}) {
  const { width, n, p, mu, sigma } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || sigma === 0) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Range: [-4, 4] on standardized axis (k − np)/σ
    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);

    // Compute standardized PMF: density on standardized axis = pmf(k) * σ.
    const bars: Array<{ z: number; density: number; dx: number }> = [];
    const dx = 1 / sigma; // spacing between adjacent k's on standardized axis
    for (let k = 0; k <= n; k++) {
      const z = (k - mu) / sigma;
      if (z < -4.2 || z > 4.2) continue;
      const pk = pmfBinomial(k, n, p);
      bars.push({ z, density: pk * sigma, dx });
    }

    const yMax = Math.max(
      d3.max(bars, (b) => b.density) ?? 0.4,
      pdfStdNormal(0) * 1.05,
    );
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Bars
    const barWidth = Math.max(
      0.5,
      xScale(dx) - xScale(0) - 1,
    );
    g.append('g')
      .attr('fill', BAR_COLOR)
      .attr('fill-opacity', 0.6)
      .selectAll('rect')
      .data(bars)
      .join('rect')
      .attr('x', (b) => xScale(b.z - dx / 2))
      .attr('y', (b) => yScale(b.density))
      .attr('width', barWidth)
      .attr('height', (b) => Math.max(0, innerH - yScale(b.density)));

    // N(0,1) PDF
    const xs = d3.range(-4, 4.01, 0.05);
    const line = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(pdfStdNormal(x)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', NORMAL_COLOR)
      .attr('stroke-width', 2)
      .attr('d', line);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('(k − np)/√(np(1−p))');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Binomial PMF (rescaled) vs N(0,1)');
  }, [width, n, p, mu, sigma]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function CDFPanel(props: {
  width: number;
  n: number;
  p: number;
  mu: number;
  sigma: number;
  continuity: boolean;
}) {
  const { width, n, p, mu, sigma, continuity } = props;
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || sigma === 0) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    // Build step-function CDF in standardized coordinates.
    const cdfPoints: Array<[number, number]> = [];
    let maxDiffAt: { z: number; cdfBin: number; phi: number; gap: number } | null = null;
    for (let k = 0; k <= n; k++) {
      const bin = cdfBinomial(k, n, p);
      const zJump = (k - mu) / sigma;
      cdfPoints.push([zJump, bin]);
      const zForPhi = continuity ? (k + 0.5 - mu) / sigma : zJump;
      const phi = cdfStdNormal(zForPhi);
      const diff = Math.abs(bin - phi);
      if (!maxDiffAt || diff > maxDiffAt.gap) {
        maxDiffAt = { z: zForPhi, cdfBin: bin, phi, gap: diff };
      }
    }

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Φ(x) reference
    const xs = d3.range(-4, 4.01, 0.05);
    const phiLine = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(cdfStdNormal(x)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', CDF_NORMAL)
      .attr('stroke-width', 2)
      .attr('d', phiLine);

    // Binomial step CDF
    const stepLine = d3
      .line<[number, number]>()
      .x(([x]) => xScale(Math.max(-4, Math.min(4, x))))
      .y(([, y]) => yScale(y))
      .curve(d3.curveStepAfter);
    g.append('path')
      .datum(cdfPoints)
      .attr('fill', 'none')
      .attr('stroke', CDF_BINOMIAL)
      .attr('stroke-width', 2)
      .attr('d', stepLine);

    // Highlight the max |F_Bin − Φ| location
    if (maxDiffAt && Math.abs(maxDiffAt.z) <= 4.2) {
      const zClamped = Math.max(-4, Math.min(4, maxDiffAt.z));
      g.append('line')
        .attr('x1', xScale(zClamped))
        .attr('x2', xScale(zClamped))
        .attr('y1', yScale(maxDiffAt.cdfBin))
        .attr('y2', yScale(maxDiffAt.phi))
        .attr('stroke', MAX_DIFF_COLOR)
        .attr('stroke-width', 2);
      g.append('text')
        .attr('x', xScale(zClamped) + 6)
        .attr('y', yScale((maxDiffAt.cdfBin + maxDiffAt.phi) / 2))
        .attr('font-size', 10)
        .attr('fill', MAX_DIFF_COLOR)
        .text(`max gap = ${maxDiffAt.gap.toFixed(4)}`);
    }

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('z');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Binomial CDF vs Φ');
  }, [width, n, p, mu, sigma, continuity]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div
      className="rounded border p-2"
      style={{ borderColor: color ?? 'var(--color-border)' }}
    >
      <div className="text-[10px] uppercase tracking-wide opacity-60">{label}</div>
      <div className="font-mono text-sm" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export default DeMoivreLaplaceExplorer;
