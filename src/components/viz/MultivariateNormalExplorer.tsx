import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBivariateNormal,
  pdfNormal,
  eigenSymmetric2x2,
  cholesky2x2,
} from './shared/distributions';
import { mvnPresets } from '../../data/multivariate-distributions-data';

// ── Layout constants ──────────────────────────────────────────────────────

const MARGIN = { top: 40, right: 40, bottom: 40, left: 50 };
const MARGINAL_SIZE = 50;
const GRID_RES = 70;
const NUM_ELLIPSE_PTS = 120;
const NUM_MARGINAL_PTS = 200;
const SAMPLE_COUNT = 200;

// ── Color helpers ─────────────────────────────────────────────────────────

/** Blue sequential ramp for density heatmap: white -> light blue -> dark blue. */
function densityColor(t: number): string {
  // t in [0, 1] — normalized density
  const r = Math.round(255 * (1 - 0.85 * t));
  const g = Math.round(255 * (1 - 0.63 * t));
  const b = Math.round(255 * (1 - 0.08 * t));
  return `rgb(${r},${g},${b})`;
}

const ELLIPSE_1S_COLOR = '#1d4ed8';
const ELLIPSE_2S_COLOR = '#60a5fa';
const EIGENVECTOR_COLOR = '#dc2626';
const SAMPLE_COLOR = '#f97316';
const MARGINAL_COLOR = '#2563eb';
const AXIS_COLOR = 'currentColor';

// ── Seeded PRNG for stable samples ────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Component ─────────────────────────────────────────────────────────────

export default function MultivariateNormalExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Parameters
  const [sigma1, setSigma1] = useState(1);
  const [sigma2, setSigma2] = useState(1);
  const [rho, setRho] = useState(0);
  const [showSamples, setShowSamples] = useState(false);
  const [showMarginals, setShowMarginals] = useState(false);

  // Responsive layout
  const containerW = Math.max(320, (width || 700) - 16);
  const isNarrow = containerW < 640;
  const plotSide = isNarrow
    ? containerW - 16
    : Math.floor(containerW * 0.62);
  const plotW = plotSide - MARGIN.left - MARGIN.right;
  const plotH = plotW; // square aspect for the contour plot
  const svgW = plotSide;
  const svgH = plotSide;

  // Covariance matrix and derived quantities
  const derived = useMemo(() => {
    const s1sq = sigma1 * sigma1;
    const s2sq = sigma2 * sigma2;
    const cov = rho * sigma1 * sigma2;
    const Sigma = [[s1sq, cov], [cov, s2sq]];
    const det = s1sq * s2sq - cov * cov;
    const { values, vectors } = eigenSymmetric2x2(Sigma);
    const condNum = values[1] > 1e-12 ? values[0] / values[1] : Infinity;
    return { Sigma, det, eigenvalues: values, eigenvectors: vectors, condNum };
  }, [sigma1, sigma2, rho]);

  // Axis range: [-4*max(sigma), 4*max(sigma)]
  const axisRange = useMemo(() => {
    const maxSig = Math.max(sigma1, sigma2);
    const r = 4 * maxSig;
    return { min: -r, max: r };
  }, [sigma1, sigma2]);

  const xSpan = axisRange.max - axisRange.min;

  // Scale helpers
  const toSvgX = useCallback(
    (x: number) => MARGIN.left + ((x - axisRange.min) / xSpan) * plotW,
    [axisRange.min, xSpan, plotW],
  );
  const toSvgY = useCallback(
    (y: number) => MARGIN.top + plotH - ((y - axisRange.min) / xSpan) * plotH,
    [axisRange.min, xSpan, plotH],
  );

  // Density heatmap grid
  const heatmapData = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; color: string }[] = [];
    const step = xSpan / GRID_RES;
    let maxDens = 0;

    // First pass: evaluate density
    const grid: number[][] = [];
    for (let i = 0; i < GRID_RES; i++) {
      const row: number[] = [];
      for (let j = 0; j < GRID_RES; j++) {
        const gx = axisRange.min + (j + 0.5) * step;
        const gy = axisRange.min + (GRID_RES - 1 - i + 0.5) * step;
        const d = pdfBivariateNormal(gx, gy, 0, 0, sigma1, sigma2, rho);
        row.push(d);
        if (d > maxDens) maxDens = d;
      }
      grid.push(row);
    }

    // Second pass: map to colors
    const cellW = plotW / GRID_RES;
    const cellH = plotH / GRID_RES;
    for (let i = 0; i < GRID_RES; i++) {
      for (let j = 0; j < GRID_RES; j++) {
        const t = maxDens > 0 ? grid[i][j] / maxDens : 0;
        if (t > 0.01) {
          cells.push({
            x: MARGIN.left + j * cellW,
            y: MARGIN.top + i * cellH,
            w: cellW + 0.5, // slight overlap to prevent gaps
            h: cellH + 0.5,
            color: densityColor(t),
          });
        }
      }
    }
    return cells;
  }, [sigma1, sigma2, rho, axisRange.min, xSpan, plotW, plotH]);

  // Confidence ellipse path builder
  const buildEllipsePath = useCallback(
    (c: number) => {
      const { eigenvalues, eigenvectors } = derived;
      const [v1, v2] = eigenvectors;
      const r1 = Math.sqrt(c * eigenvalues[0]);
      const r2 = Math.sqrt(c * eigenvalues[1]);
      const pts: string[] = [];
      for (let i = 0; i <= NUM_ELLIPSE_PTS; i++) {
        const theta = (2 * Math.PI * i) / NUM_ELLIPSE_PTS;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const ex = r1 * cosT * v1[0] + r2 * sinT * v2[0];
        const ey = r1 * cosT * v1[1] + r2 * sinT * v2[1];
        const sx = toSvgX(ex);
        const sy = toSvgY(ey);
        pts.push(`${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`);
      }
      return pts.join(' ') + 'Z';
    },
    [derived, toSvgX, toSvgY],
  );

  // Eigenvector arrows
  const eigArrows = useMemo(() => {
    const { eigenvalues, eigenvectors } = derived;
    return eigenvectors.map((v, idx) => {
      const len = Math.sqrt(eigenvalues[idx]);
      return {
        x2: toSvgX(v[0] * len),
        y2: toSvgY(v[1] * len),
        label: `v${idx + 1}`,
      };
    });
  }, [derived, toSvgX, toSvgY]);

  const originX = toSvgX(0);
  const originY = toSvgY(0);

  // Samples via Box-Muller + Cholesky
  const samplePoints = useMemo(() => {
    if (!showSamples) return [];
    const L = cholesky2x2(derived.Sigma);
    const rng = mulberry32(42);
    const pts: { sx: number; sy: number }[] = [];
    for (let i = 0; i < SAMPLE_COUNT; i++) {
      const u1 = rng();
      const u2 = rng();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      const x = L[0][0] * z1 + L[0][1] * z2;
      const y = L[1][0] * z1 + L[1][1] * z2;
      pts.push({ sx: toSvgX(x), sy: toSvgY(y) });
    }
    return pts;
  }, [showSamples, derived.Sigma, toSvgX, toSvgY]);

  // Marginal PDFs
  const marginalsData = useMemo(() => {
    if (!showMarginals) return null;
    const xPdf: { px: number; val: number }[] = [];
    const yPdf: { py: number; val: number }[] = [];
    let maxXPdf = 0;
    let maxYPdf = 0;

    for (let i = 0; i <= NUM_MARGINAL_PTS; i++) {
      const t = axisRange.min + (i / NUM_MARGINAL_PTS) * xSpan;
      const dx = pdfNormal(t, 0, sigma1 * sigma1);
      const dy = pdfNormal(t, 0, sigma2 * sigma2);
      xPdf.push({ px: toSvgX(t), val: dx });
      yPdf.push({ py: toSvgY(t), val: dy });
      if (dx > maxXPdf) maxXPdf = dx;
      if (dy > maxYPdf) maxYPdf = dy;
    }

    // Top marginal (x axis): draw curve from plot top upward
    const topPath = xPdf.map((d, i) => {
      const py = MARGIN.top - (d.val / maxXPdf) * MARGINAL_SIZE;
      return `${i === 0 ? 'M' : 'L'}${d.px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
    const topBaseline = `L${toSvgX(axisRange.max).toFixed(1)},${MARGIN.top} L${toSvgX(axisRange.min).toFixed(1)},${MARGIN.top}Z`;

    // Right marginal (y axis): draw curve from plot right outward
    const rightPath = yPdf.map((d, i) => {
      const px = MARGIN.left + plotW + (d.val / maxYPdf) * MARGINAL_SIZE;
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${d.py.toFixed(1)}`;
    }).join(' ');
    const rightBaseline = `L${(MARGIN.left + plotW).toFixed(1)},${toSvgY(axisRange.min).toFixed(1)} L${(MARGIN.left + plotW).toFixed(1)},${toSvgY(axisRange.max).toFixed(1)}Z`;

    return { topPath, topBaseline, rightPath, rightBaseline };
  }, [showMarginals, sigma1, sigma2, axisRange, xSpan, plotW, toSvgX, toSvgY]);

  // Axis ticks
  const axisTicks = useMemo(() => {
    const ticks: number[] = [];
    const maxSig = Math.max(sigma1, sigma2);
    const step = maxSig <= 1.5 ? 1 : maxSig <= 3 ? 2 : 4;
    const lo = Math.ceil(axisRange.min / step) * step;
    for (let v = lo; v <= axisRange.max; v += step) {
      ticks.push(v);
    }
    return ticks;
  }, [sigma1, sigma2, axisRange]);

  // Preset handler
  const applyPreset = useCallback((p: typeof mvnPresets[number]) => {
    setSigma1(p.sigma1);
    setSigma2(p.sigma2);
    setRho(p.rho);
  }, []);

  // Matrix readout formatting
  const fmt = (v: number) => v.toFixed(3);
  const fmtShort = (v: number) => v.toFixed(2);

  const { Sigma, det, eigenvalues, eigenvectors, condNum } = derived;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Multivariate Normal Explorer
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-2 justify-center mb-3">
        {mvnPresets.map((p) => (
          <button
            key={p.name}
            onClick={() => applyPreset(p)}
            className="rounded border px-2 py-1 text-xs transition-colors hover:bg-blue-50 dark:hover:bg-blue-950"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* Parameter sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-2">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium w-6">σ₁</span>
          <input
            type="range" min={0.5} max={3} step={0.1}
            value={sigma1}
            onChange={(e) => setSigma1(Number(e.target.value))}
            className="w-28"
          />
          <span className="w-8 tabular-nums text-right">{sigma1.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium w-6">σ₂</span>
          <input
            type="range" min={0.5} max={3} step={0.1}
            value={sigma2}
            onChange={(e) => setSigma2(Number(e.target.value))}
            className="w-28"
          />
          <span className="w-8 tabular-nums text-right">{sigma2.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium w-6">ρ</span>
          <input
            type="range" min={-0.95} max={0.95} step={0.05}
            value={rho}
            onChange={(e) => setRho(Number(e.target.value))}
            className="w-28"
          />
          <span className="w-12 tabular-nums text-right">{rho >= 0 ? '+' : ''}{rho.toFixed(2)}</span>
        </label>
      </div>

      {/* Toggles */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        <label className="flex items-center gap-1.5 text-xs cursor-pointer">
          <input
            type="checkbox" checked={showSamples}
            onChange={(e) => setShowSamples(e.target.checked)}
            className="rounded"
          />
          <span>Show samples (n={SAMPLE_COUNT})</span>
        </label>
        <label className="flex items-center gap-1.5 text-xs cursor-pointer">
          <input
            type="checkbox" checked={showMarginals}
            onChange={(e) => setShowMarginals(e.target.checked)}
            className="rounded"
          />
          <span>Show marginals</span>
        </label>
      </div>

      {/* Main two-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* Left panel: contour plot */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: svgW }}>
          <svg width={svgW} height={svgH} className="block mx-auto">
            {/* Clip region for the plot area */}
            <defs>
              <clipPath id="mvn-plot-clip">
                <rect x={MARGIN.left} y={MARGIN.top} width={plotW} height={plotH} />
              </clipPath>
              {/* Arrowhead marker for eigenvectors */}
              <marker
                id="mvn-arrowhead"
                markerWidth={8} markerHeight={6}
                refX={7} refY={3}
                orient="auto"
              >
                <polygon points="0,0 8,3 0,6" fill={EIGENVECTOR_COLOR} />
              </marker>
            </defs>

            {/* Background for plot area */}
            <rect
              x={MARGIN.left} y={MARGIN.top}
              width={plotW} height={plotH}
              fill="white"
              stroke={AXIS_COLOR}
              strokeOpacity={0.15}
            />

            {/* Heatmap cells */}
            <g clipPath="url(#mvn-plot-clip)">
              {heatmapData.map((cell, i) => (
                <rect
                  key={i}
                  x={cell.x} y={cell.y}
                  width={cell.w} height={cell.h}
                  fill={cell.color}
                />
              ))}
            </g>

            {/* Grid lines */}
            {axisTicks.map((v, i) => (
              <g key={i}>
                <line
                  x1={toSvgX(v)} y1={MARGIN.top}
                  x2={toSvgX(v)} y2={MARGIN.top + plotH}
                  stroke={AXIS_COLOR} strokeOpacity={0.1}
                />
                <line
                  x1={MARGIN.left} y1={toSvgY(v)}
                  x2={MARGIN.left + plotW} y2={toSvgY(v)}
                  stroke={AXIS_COLOR} strokeOpacity={0.1}
                />
              </g>
            ))}

            {/* 2σ ellipse (draw first so 1σ sits on top) */}
            <path
              d={buildEllipsePath(4)}
              fill="none"
              stroke={ELLIPSE_2S_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
              clipPath="url(#mvn-plot-clip)"
            />

            {/* 1σ ellipse */}
            <path
              d={buildEllipsePath(1)}
              fill="none"
              stroke={ELLIPSE_1S_COLOR}
              strokeWidth={2}
              clipPath="url(#mvn-plot-clip)"
            />

            {/* Eigenvector arrows */}
            {eigArrows.map((arrow, i) => (
              <line
                key={i}
                x1={originX} y1={originY}
                x2={arrow.x2} y2={arrow.y2}
                stroke={EIGENVECTOR_COLOR}
                strokeWidth={2}
                markerEnd="url(#mvn-arrowhead)"
                clipPath="url(#mvn-plot-clip)"
              />
            ))}

            {/* Sample scatter points */}
            {samplePoints.map((pt, i) => (
              <circle
                key={i}
                cx={pt.sx} cy={pt.sy}
                r={2.2}
                fill={SAMPLE_COLOR}
                fillOpacity={0.55}
                clipPath="url(#mvn-plot-clip)"
              />
            ))}

            {/* Marginal distributions */}
            {marginalsData && (
              <>
                {/* Top marginal (X) */}
                <path
                  d={marginalsData.topPath + marginalsData.topBaseline}
                  fill={MARGINAL_COLOR}
                  fillOpacity={0.12}
                />
                <path
                  d={marginalsData.topPath}
                  fill="none"
                  stroke={MARGINAL_COLOR}
                  strokeWidth={1.5}
                />
                {/* Right marginal (Y) */}
                <path
                  d={marginalsData.rightPath + marginalsData.rightBaseline}
                  fill={MARGINAL_COLOR}
                  fillOpacity={0.12}
                />
                <path
                  d={marginalsData.rightPath}
                  fill="none"
                  stroke={MARGINAL_COLOR}
                  strokeWidth={1.5}
                />
              </>
            )}

            {/* Axes */}
            <line
              x1={MARGIN.left} y1={MARGIN.top + plotH}
              x2={MARGIN.left + plotW} y2={MARGIN.top + plotH}
              stroke={AXIS_COLOR} strokeOpacity={0.3}
            />
            <line
              x1={MARGIN.left} y1={MARGIN.top}
              x2={MARGIN.left} y2={MARGIN.top + plotH}
              stroke={AXIS_COLOR} strokeOpacity={0.3}
            />

            {/* Axis tick labels */}
            {axisTicks.map((v, i) => (
              <g key={i}>
                <text
                  x={toSvgX(v)} y={MARGIN.top + plotH + 14}
                  textAnchor="middle" fontSize={9}
                  fill="currentColor" opacity={0.6}
                >
                  {v}
                </text>
                <text
                  x={MARGIN.left - 6} y={toSvgY(v) + 3}
                  textAnchor="end" fontSize={9}
                  fill="currentColor" opacity={0.6}
                >
                  {v}
                </text>
              </g>
            ))}

            {/* Axis labels */}
            <text
              x={MARGIN.left + plotW / 2}
              y={MARGIN.top + plotH + 30}
              textAnchor="middle" fontSize={11}
              fill="currentColor" opacity={0.6}
            >
              x₁
            </text>
            <text
              x={14} y={MARGIN.top + plotH / 2}
              textAnchor="middle" fontSize={11}
              fill="currentColor" opacity={0.6}
              transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
            >
              x₂
            </text>

            {/* Legend */}
            <g transform={`translate(${MARGIN.left + 6}, ${MARGIN.top + 8})`}>
              <line x1={0} y1={0} x2={14} y2={0} stroke={ELLIPSE_1S_COLOR} strokeWidth={2} />
              <text x={18} y={3.5} fontSize={9} fill="currentColor" opacity={0.7}>1σ</text>
              <line x1={0} y1={12} x2={14} y2={12} stroke={ELLIPSE_2S_COLOR} strokeWidth={1.5} strokeDasharray="4,2" />
              <text x={18} y={15.5} fontSize={9} fill="currentColor" opacity={0.7}>2σ</text>
              <line x1={0} y1={24} x2={14} y2={24} stroke={EIGENVECTOR_COLOR} strokeWidth={2} />
              <text x={18} y={27.5} fontSize={9} fill="currentColor" opacity={0.7}>eigenvectors</text>
            </g>
          </svg>
        </div>

        {/* Right panel: matrix readout */}
        <div
          className={`flex-1 min-w-0 rounded-lg border p-3 flex flex-col justify-center ${isNarrow ? 'w-full' : ''}`}
          style={{ borderColor: 'var(--color-border)', minWidth: isNarrow ? undefined : 200 }}
        >
          {/* Covariance matrix */}
          <div className="text-xs font-semibold mb-2 text-center opacity-70">
            Covariance Matrix
          </div>
          <div className="flex justify-center mb-3">
            <div className="inline-flex items-center gap-1 text-xs">
              <span className="font-medium">Σ =</span>
              <div className="inline-flex flex-col border-l-2 border-r-2 px-2 py-0.5" style={{ borderColor: 'var(--color-border)' }}>
                <div className="flex gap-3 tabular-nums">
                  <span className="w-14 text-right">{fmtShort(Sigma[0][0])}</span>
                  <span className="w-14 text-right">{fmtShort(Sigma[0][1])}</span>
                </div>
                <div className="flex gap-3 tabular-nums">
                  <span className="w-14 text-right">{fmtShort(Sigma[1][0])}</span>
                  <span className="w-14 text-right">{fmtShort(Sigma[1][1])}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Eigenvalues */}
          <div className="text-xs font-semibold mb-1.5 text-center opacity-70">
            Eigendecomposition
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs mb-3">
            <div className="flex items-baseline gap-1">
              <span className="font-medium" style={{ color: EIGENVECTOR_COLOR }}>λ₁ =</span>
              <span className="tabular-nums">{fmt(eigenvalues[0])}</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-medium" style={{ color: EIGENVECTOR_COLOR }}>λ₂ =</span>
              <span className="tabular-nums">{fmt(eigenvalues[1])}</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-medium" style={{ color: EIGENVECTOR_COLOR }}>v₁ =</span>
              <span className="tabular-nums">({fmtShort(eigenvectors[0][0])}, {fmtShort(eigenvectors[0][1])})</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-medium" style={{ color: EIGENVECTOR_COLOR }}>v₂ =</span>
              <span className="tabular-nums">({fmtShort(eigenvectors[1][0])}, {fmtShort(eigenvectors[1][1])})</span>
            </div>
          </div>

          {/* Determinant and condition number */}
          <div className="text-xs font-semibold mb-1.5 text-center opacity-70">
            Matrix Properties
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <div className="flex items-baseline gap-1">
              <span className="font-medium">|Σ| =</span>
              <span className="tabular-nums">{fmt(det)}</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-medium">κ(Σ) =</span>
              <span className="tabular-nums">
                {isFinite(condNum) ? fmt(condNum) : '∞'}
              </span>
            </div>
          </div>

          {/* Interpretation callout */}
          <div
            className="mt-3 pt-2 border-t text-[10px] leading-relaxed opacity-60"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {Math.abs(rho) < 0.05 && Math.abs(sigma1 - sigma2) < 0.15 && (
              <span>Σ is nearly spherical — the contours are approximately circular and the principal axes align with the coordinate axes.</span>
            )}
            {Math.abs(rho) < 0.05 && Math.abs(sigma1 - sigma2) >= 0.15 && (
              <span>ρ ≈ 0 with unequal variances — the ellipse axes align with the coordinates but stretch along the larger-variance direction.</span>
            )}
            {rho >= 0.05 && (
              <span>Positive correlation (ρ = {rho.toFixed(2)}) tilts the ellipse toward the x₁ = x₂ line. The condition number κ = {isFinite(condNum) ? condNum.toFixed(1) : '∞'} measures how elongated the ellipse is.</span>
            )}
            {rho <= -0.05 && (
              <span>Negative correlation (ρ = {rho.toFixed(2)}) tilts the ellipse toward x₁ = −x₂. Higher |ρ| concentrates density along this anti-diagonal.</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
