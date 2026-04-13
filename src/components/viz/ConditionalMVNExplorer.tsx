import { useState, useMemo, useCallback, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBivariateNormal, conditionalNormalParams, pdfNormal,
} from './shared/distributions';

// ── Constants ─────────────────────────────────────────────────────────────

const RANGE = 3.2;
const GRID_N = 60;
const COND_CURVE_PTS = 120;
const JOINT_MARGIN = { top: 30, right: 10, bottom: 32, left: 36 };
const RIGHT_MARGIN = { top: 30, right: 16, bottom: 32, left: 40 };

const COLOR_JOINT = '#2563eb';
const COLOR_COND_MEAN = '#dc2626';
const COLOR_REGRESSION = '#dc2626';
const COLOR_COND_CURVE = '#7c3aed';
const COLOR_COND_FILL = '#c4b5fd';
const COLOR_HIGHLIGHT = '#f59e0b';

// ── Heatmap color interpolation ───────────────────────────────────────────

function densityColor(density: number, maxDensity: number): string {
  const t = Math.min(density / maxDensity, 1);
  // White -> light blue -> blue
  const r = Math.round(255 - t * (255 - 37));
  const g = Math.round(255 - t * (255 - 99));
  const b = Math.round(255 - t * (255 - 235));
  return `rgb(${r},${g},${b})`;
}

// ── Component ─────────────────────────────────────────────────────────────

export default function ConditionalMVNExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState(0.6);
  const [condY, setCondY] = useState(0.0);
  const dragging = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // Fixed parameters
  const mu1 = 0, mu2 = 0, sigma1 = 1, sigma2 = 1;

  // ── Layout calculations ───────────────────────────────────────────────

  const totalW = Math.max(320, (containerWidth || 700) - 16);
  const isNarrow = totalW < 680;

  // Panel widths
  const jointW = isNarrow ? totalW : Math.floor(totalW * 0.4);
  const centerW = isNarrow ? totalW : Math.floor(totalW * 0.25);
  const rightW = isNarrow ? totalW : totalW - jointW - centerW;
  const jointH = isNarrow ? Math.min(jointW, 320) : Math.min(jointW, 360);
  const rightH = jointH;

  const jointPlotW = jointW - JOINT_MARGIN.left - JOINT_MARGIN.right;
  const jointPlotH = jointH - JOINT_MARGIN.top - JOINT_MARGIN.bottom;
  const rightPlotW = rightW - RIGHT_MARGIN.left - RIGHT_MARGIN.right;
  const rightPlotH = rightH - RIGHT_MARGIN.top - RIGHT_MARGIN.bottom;

  // ── Scales ────────────────────────────────────────────────────────────

  const toJointX = useCallback(
    (v: number) => JOINT_MARGIN.left + ((v + RANGE) / (2 * RANGE)) * jointPlotW,
    [jointPlotW],
  );
  const toJointY = useCallback(
    (v: number) => JOINT_MARGIN.top + jointPlotH - ((v + RANGE) / (2 * RANGE)) * jointPlotH,
    [jointPlotH],
  );
  const fromJointY = useCallback(
    (py: number) => {
      const frac = (JOINT_MARGIN.top + jointPlotH - py) / jointPlotH;
      return frac * 2 * RANGE - RANGE;
    },
    [jointPlotH],
  );
  const toRightX = useCallback(
    (v: number) => RIGHT_MARGIN.left + ((v + RANGE) / (2 * RANGE)) * rightPlotW,
    [rightPlotW],
  );
  const toRightY = useCallback(
    (density: number, maxD: number) =>
      RIGHT_MARGIN.top + rightPlotH - (density / (maxD * 1.15)) * rightPlotH,
    [rightPlotH],
  );

  // ── Conditional parameters ────────────────────────────────────────────

  const { condMean, condVar } = useMemo(
    () => conditionalNormalParams(mu1, mu2, sigma1, sigma2, rho, condY),
    [rho, condY],
  );
  const condStd = Math.sqrt(Math.max(condVar, 1e-10));

  // ── Heatmap grid ──────────────────────────────────────────────────────

  const heatmapData = useMemo(() => {
    const step = (2 * RANGE) / GRID_N;
    const cells: { x: number; y: number; px: number; py: number; w: number; h: number; d: number }[] = [];
    let maxD = 0;
    for (let j = 0; j < GRID_N; j++) {
      for (let i = 0; i < GRID_N; i++) {
        const xVal = -RANGE + (i + 0.5) * step;
        const yVal = -RANGE + (j + 0.5) * step;
        const d = pdfBivariateNormal(xVal, yVal, mu1, mu2, sigma1, sigma2, rho);
        if (d > maxD) maxD = d;
        cells.push({ x: xVal, y: yVal, px: i, py: j, w: 1, h: 1, d });
      }
    }
    return { cells, maxD, step };
  }, [rho]);

  // ── Conditional density curve on the joint panel ──────────────────────

  const jointCondCurve = useMemo(() => {
    const pts: { x1: number; density: number }[] = [];
    let maxD = 0;
    for (let i = 0; i <= COND_CURVE_PTS; i++) {
      const x1 = -RANGE + (2 * RANGE * i) / COND_CURVE_PTS;
      const d = pdfNormal(x1, condMean, condVar);
      if (d > maxD) maxD = d;
      pts.push({ x1, density: d });
    }
    return { pts, maxD };
  }, [condMean, condVar]);

  // ── Right panel: conditional density ──────────────────────────────────

  const rightCurve = useMemo(() => {
    const pts: { x: number; d: number }[] = [];
    let maxD = 0;
    for (let i = 0; i <= COND_CURVE_PTS; i++) {
      const x = -RANGE + (2 * RANGE * i) / COND_CURVE_PTS;
      const d = pdfNormal(x, condMean, condVar);
      if (d > maxD) maxD = d;
      pts.push({ x, d });
    }
    return { pts, maxD };
  }, [condMean, condVar]);

  // ── Regression line endpoints ─────────────────────────────────────────
  // For standard BVN with equal variances: μ_{1|2} = ρ * x₂
  const regY1 = -RANGE;
  const regY2 = RANGE;
  const regX1 = rho * regY1;
  const regX2 = rho * regY2;

  // ── Drag handlers ─────────────────────────────────────────────────────

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as Element).setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const py = e.clientY - rect.top;
    const dataY = fromJointY(py);
    const clamped = Math.max(-RANGE + 0.1, Math.min(RANGE - 0.1, dataY));
    setCondY(Math.round(clamped * 20) / 20);
  }, [fromJointY]);

  const handlePointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  // ── Axis ticks ────────────────────────────────────────────────────────

  const axisTicks = [-3, -2, -1, 0, 1, 2, 3];

  // ── Conditional density overlay curve path for joint panel ────────────
  // Scale the density curve to be visible (~20% of plot width at max)

  const condOverlayScale = jointPlotW * 0.18;
  const condOverlayPath = useMemo(() => {
    if (jointCondCurve.maxD === 0) return '';
    return jointCondCurve.pts
      .map((p, i) => {
        const sx = toJointX(p.x1);
        const scaledD = (p.density / jointCondCurve.maxD) * condOverlayScale;
        const sy = toJointY(condY) - scaledD;
        return `${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`;
      })
      .join(' ');
  }, [jointCondCurve, condOverlayScale, toJointX, toJointY, condY]);

  const condOverlayFillPath = condOverlayPath + ' ' +
    jointCondCurve.pts
      .slice()
      .reverse()
      .map((p, i) => {
        const sx = toJointX(p.x1);
        const sy = toJointY(condY);
        return `${i === 0 ? 'L' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`;
      })
      .join(' ') + ' Z';

  // ── Right panel: shaded +-2σ region ───────────────────────────────────

  const rightShadedPath = useMemo(() => {
    if (rightCurve.maxD === 0) return '';
    const lo = condMean - 2 * condStd;
    const hi = condMean + 2 * condStd;
    const filtered = rightCurve.pts.filter((p) => p.x >= lo && p.x <= hi);
    if (filtered.length < 2) return '';
    const top = filtered
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${toRightX(p.x).toFixed(1)},${toRightY(p.d, rightCurve.maxD).toFixed(1)}`)
      .join(' ');
    const bottom = `L${toRightX(filtered[filtered.length - 1].x).toFixed(1)},${toRightY(0, rightCurve.maxD).toFixed(1)} ` +
      `L${toRightX(filtered[0].x).toFixed(1)},${toRightY(0, rightCurve.maxD).toFixed(1)} Z`;
    return top + ' ' + bottom;
  }, [rightCurve, condMean, condStd, toRightX, toRightY]);

  const rightCurvePath = useMemo(() => {
    if (rightCurve.maxD === 0) return '';
    return rightCurve.pts
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${toRightX(p.x).toFixed(1)},${toRightY(p.d, rightCurve.maxD).toFixed(1)}`)
      .join(' ');
  }, [rightCurve, toRightX, toRightY]);

  // ── Heatmap cell dimensions in SVG ────────────────────────────────────

  const cellW = jointPlotW / GRID_N;
  const cellH = jointPlotH / GRID_N;

  // ── Formula values ────────────────────────────────────────────────────

  const sigma12 = rho * sigma1 * sigma2;
  const sigma22 = sigma2 * sigma2;
  const sigma11 = sigma1 * sigma1;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Conditional Multivariate Normal Explorer
      </div>

      {/* Correlation slider */}
      <div className="flex flex-wrap justify-center items-center gap-3 mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Correlation (rho)</span>
          <input
            type="range"
            min={-0.95}
            max={0.95}
            step={0.05}
            value={rho}
            onChange={(e) => setRho(Number(e.target.value))}
            className="w-40"
          />
          <span className="w-12 tabular-nums text-right font-mono">{rho.toFixed(2)}</span>
        </label>
        <span className="text-xs opacity-60">
          Drag the horizontal line on the joint density to condition on x&#x2082;
        </span>
      </div>

      {/* Three-panel layout */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-2 items-start'}>

        {/* ── Left Panel: Joint Density + Conditioning Line ──────────── */}
        <div style={{ width: jointW, flexShrink: 0 }}>
          <svg
            ref={svgRef}
            width={jointW}
            height={jointH}
            className="block"
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            style={{ touchAction: 'none' }}
          >
            {/* Heatmap cells */}
            {heatmapData.cells.map((c, i) => (
              <rect
                key={i}
                x={JOINT_MARGIN.left + c.px * cellW}
                y={JOINT_MARGIN.top + (GRID_N - 1 - c.py) * cellH}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={densityColor(c.d, heatmapData.maxD)}
              />
            ))}

            {/* Regression line: μ_{1|2} = ρ·x₂ */}
            {Math.abs(rho) > 0.01 && (
              <line
                x1={toJointX(Math.max(-RANGE, regX1))}
                y1={toJointY(regY1)}
                x2={toJointX(Math.min(RANGE, regX2))}
                y2={toJointY(regY2)}
                stroke={COLOR_REGRESSION}
                strokeWidth={1.5}
                strokeDasharray="6,4"
                opacity={0.7}
              />
            )}

            {/* Conditional density overlay along conditioning line */}
            <path d={condOverlayFillPath} fill={COLOR_COND_FILL} opacity={0.35} />
            <path d={condOverlayPath} fill="none" stroke={COLOR_COND_CURVE} strokeWidth={1.8} />

            {/* Conditioning line (draggable) */}
            <line
              x1={JOINT_MARGIN.left}
              y1={toJointY(condY)}
              x2={JOINT_MARGIN.left + jointPlotW}
              y2={toJointY(condY)}
              stroke={COLOR_HIGHLIGHT}
              strokeWidth={2.5}
            />
            {/* Wider invisible hit area for dragging */}
            <line
              x1={JOINT_MARGIN.left}
              y1={toJointY(condY)}
              x2={JOINT_MARGIN.left + jointPlotW}
              y2={toJointY(condY)}
              stroke="transparent"
              strokeWidth={16}
              style={{ cursor: 'ns-resize' }}
              onPointerDown={handlePointerDown}
            />

            {/* Conditional mean dot on the conditioning line */}
            <circle
              cx={toJointX(condMean)}
              cy={toJointY(condY)}
              r={5}
              fill={COLOR_COND_MEAN}
              stroke="white"
              strokeWidth={1.5}
            />

            {/* Axes */}
            <line
              x1={JOINT_MARGIN.left}
              y1={JOINT_MARGIN.top + jointPlotH}
              x2={JOINT_MARGIN.left + jointPlotW}
              y2={JOINT_MARGIN.top + jointPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={JOINT_MARGIN.left}
              y1={JOINT_MARGIN.top}
              x2={JOINT_MARGIN.left}
              y2={JOINT_MARGIN.top + jointPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks */}
            {axisTicks.map((v) => (
              <text
                key={`jx-${v}`}
                x={toJointX(v)}
                y={JOINT_MARGIN.top + jointPlotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.55}
              >
                {v}
              </text>
            ))}

            {/* Y-axis ticks */}
            {axisTicks.map((v) => (
              <text
                key={`jy-${v}`}
                x={JOINT_MARGIN.left - 6}
                y={toJointY(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.55}
              >
                {v}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={JOINT_MARGIN.left + jointPlotW / 2}
              y={jointH - 4}
              textAnchor="middle"
              fontSize={11}
              fill="currentColor"
              opacity={0.6}
              fontStyle="italic"
            >
              x&#x2081;
            </text>
            <text
              x={10}
              y={JOINT_MARGIN.top + jointPlotH / 2}
              textAnchor="middle"
              fontSize={11}
              fill="currentColor"
              opacity={0.6}
              fontStyle="italic"
              transform={`rotate(-90, 10, ${JOINT_MARGIN.top + jointPlotH / 2})`}
            >
              x&#x2082;
            </text>

            {/* x₂ value label */}
            <text
              x={JOINT_MARGIN.left + jointPlotW + 2}
              y={toJointY(condY) + 4}
              fontSize={9}
              fill={COLOR_HIGHLIGHT}
              fontWeight={600}
            >
              {condY.toFixed(1)}
            </text>

            {/* Panel label */}
            <text
              x={JOINT_MARGIN.left + jointPlotW / 2}
              y={14}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              fontWeight={600}
            >
              Joint Density f(x&#x2081;, x&#x2082;)
            </text>
          </svg>
        </div>

        {/* ── Center Panel: Block Matrix Diagram ─────────────────────── */}
        <div
          className="flex flex-col justify-center rounded-lg border p-3"
          style={{
            borderColor: 'var(--color-border)',
            width: isNarrow ? '100%' : centerW,
            minHeight: isNarrow ? undefined : jointH,
            flexShrink: 0,
          }}
        >
          <div className="text-[10px] font-semibold text-center opacity-60 mb-2">
            Partition Notation
          </div>

          {/* Mean vector */}
          <div className="text-xs text-center mb-2 font-mono leading-relaxed">
            <span className="opacity-60">&#x03BC;</span> = (0, 0)<sup>T</sup>
          </div>

          {/* Covariance matrix */}
          <div className="text-xs text-center mb-3 font-mono leading-relaxed">
            <div className="opacity-60 mb-0.5">&#x03A3; =</div>
            <div className="inline-flex items-center gap-0.5">
              <span className="text-lg leading-none">[</span>
              <div className="flex flex-col items-center text-[10px]">
                <div className="flex gap-2">
                  <span>{sigma11.toFixed(1)}</span>
                  <span style={{ color: COLOR_JOINT }}>{sigma12.toFixed(2)}</span>
                </div>
                <div className="flex gap-2">
                  <span style={{ color: COLOR_JOINT }}>{sigma12.toFixed(2)}</span>
                  <span>{sigma22.toFixed(1)}</span>
                </div>
              </div>
              <span className="text-lg leading-none">]</span>
            </div>
          </div>

          <div className="border-t my-2" style={{ borderColor: 'var(--color-border)' }} />

          {/* Conditional formulas with live values */}
          <div className="text-[10px] font-semibold text-center opacity-60 mb-1.5">
            Conditional Parameters
          </div>

          {/* Conditional mean */}
          <div className="text-[10.5px] leading-relaxed mb-2">
            <div className="font-mono text-center">
              <span style={{ color: COLOR_COND_MEAN }}>&#x03BC;</span>
              <sub style={{ color: COLOR_COND_MEAN }}>1|2</sub>
              <span className="opacity-60"> = &#x03BC;&#x2081; + </span>
              <span style={{ color: COLOR_JOINT }}>
                (&#x03A3;&#x2081;&#x2082; / &#x03A3;&#x2082;&#x2082;)
              </span>
              <span className="opacity-60">(x&#x2082; &#x2212; &#x03BC;&#x2082;)</span>
            </div>
            <div className="font-mono text-center mt-0.5">
              = 0 + ({sigma12.toFixed(2)} / {sigma22.toFixed(1)})({condY.toFixed(2)} &#x2212; 0)
            </div>
            <div className="font-mono text-center font-semibold" style={{ color: COLOR_COND_MEAN }}>
              = {condMean.toFixed(3)}
            </div>
            <div className="text-center mt-0.5 opacity-60">
              &#x2191; depends on x&#x2082;
            </div>
          </div>

          {/* Conditional variance */}
          <div className="text-[10.5px] leading-relaxed">
            <div className="font-mono text-center">
              <span style={{ color: COLOR_COND_CURVE }}>&#x03C3;&#x00B2;</span>
              <sub style={{ color: COLOR_COND_CURVE }}>1|2</sub>
              <span className="opacity-60"> = &#x03A3;&#x2081;&#x2081; &#x2212; </span>
              <span style={{ color: COLOR_JOINT }}>
                &#x03A3;&#x2081;&#x2082;&#x00B2;
              </span>
              <span className="opacity-60"> / &#x03A3;&#x2082;&#x2082;</span>
            </div>
            <div className="font-mono text-center mt-0.5">
              = {sigma11.toFixed(1)} &#x2212; {sigma12.toFixed(2)}&#x00B2; / {sigma22.toFixed(1)}
            </div>
            <div className="font-mono text-center font-semibold" style={{ color: COLOR_COND_CURVE }}>
              = {condVar.toFixed(3)}
            </div>
            <div className="text-center mt-0.5 opacity-60">
              &#x2191; does NOT depend on x&#x2082;
            </div>
          </div>

          <div className="border-t my-2" style={{ borderColor: 'var(--color-border)' }} />

          <div
            className="text-[10px] text-center px-1 py-1 rounded"
            style={{ background: 'var(--color-bg-secondary, rgba(0,0,0,0.03))' }}
          >
            The conditional mean is <strong>linear</strong> in x&#x2082; (the regression line).
            The conditional variance is <strong>constant</strong> — it never changes as you drag.
          </div>
        </div>

        {/* ── Right Panel: Conditional Density ───────────────────────── */}
        <div style={{ width: isNarrow ? '100%' : rightW, flexShrink: 0 }}>
          <svg width={isNarrow ? totalW : rightW} height={rightH} className="block">
            {/* Shaded ±2σ region */}
            {rightShadedPath && (
              <path d={rightShadedPath} fill={COLOR_COND_FILL} opacity={0.4} />
            )}

            {/* Density curve */}
            <path
              d={rightCurvePath}
              fill="none"
              stroke={COLOR_COND_CURVE}
              strokeWidth={2.5}
            />

            {/* Conditional mean vertical line */}
            <line
              x1={toRightX(condMean)}
              y1={RIGHT_MARGIN.top}
              x2={toRightX(condMean)}
              y2={RIGHT_MARGIN.top + rightPlotH}
              stroke={COLOR_COND_MEAN}
              strokeWidth={1.5}
              strokeDasharray="4,3"
            />

            {/* Mean label */}
            <text
              x={toRightX(condMean)}
              y={RIGHT_MARGIN.top - 4}
              textAnchor="middle"
              fontSize={9}
              fill={COLOR_COND_MEAN}
              fontWeight={600}
            >
              &#x03BC;&#x2081;&#x7C;&#x2082; = {condMean.toFixed(2)}
            </text>

            {/* ±2σ bracket labels */}
            {condMean - 2 * condStd >= -RANGE && (
              <text
                x={toRightX(condMean - 2 * condStd)}
                y={RIGHT_MARGIN.top + rightPlotH + 24}
                textAnchor="middle"
                fontSize={8}
                fill={COLOR_COND_CURVE}
                opacity={0.7}
              >
                &#x2212;2&#x03C3;
              </text>
            )}
            {condMean + 2 * condStd <= RANGE && (
              <text
                x={toRightX(condMean + 2 * condStd)}
                y={RIGHT_MARGIN.top + rightPlotH + 24}
                textAnchor="middle"
                fontSize={8}
                fill={COLOR_COND_CURVE}
                opacity={0.7}
              >
                +2&#x03C3;
              </text>
            )}

            {/* X-axis */}
            <line
              x1={RIGHT_MARGIN.left}
              y1={RIGHT_MARGIN.top + rightPlotH}
              x2={RIGHT_MARGIN.left + rightPlotW}
              y2={RIGHT_MARGIN.top + rightPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={RIGHT_MARGIN.left}
              y1={RIGHT_MARGIN.top}
              x2={RIGHT_MARGIN.left}
              y2={RIGHT_MARGIN.top + rightPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X ticks */}
            {axisTicks.map((v) => (
              <text
                key={`rx-${v}`}
                x={toRightX(v)}
                y={RIGHT_MARGIN.top + rightPlotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.55}
              >
                {v}
              </text>
            ))}

            {/* Y grid lines */}
            {[0.25, 0.5, 0.75].map((frac) => {
              const yVal = rightCurve.maxD * frac;
              return (
                <line
                  key={`rg-${frac}`}
                  x1={RIGHT_MARGIN.left}
                  y1={toRightY(yVal, rightCurve.maxD)}
                  x2={RIGHT_MARGIN.left + rightPlotW}
                  y2={toRightY(yVal, rightCurve.maxD)}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
              );
            })}

            {/* Y-axis labels */}
            {[rightCurve.maxD * 0.5, rightCurve.maxD].map((v, i) => (
              <text
                key={`ry-${i}`}
                x={RIGHT_MARGIN.left - 4}
                y={toRightY(v, rightCurve.maxD) + 3}
                textAnchor="end"
                fontSize={8}
                fill="currentColor"
                opacity={0.5}
              >
                {v.toFixed(2)}
              </text>
            ))}

            {/* Axis label */}
            <text
              x={RIGHT_MARGIN.left + rightPlotW / 2}
              y={rightH - 4}
              textAnchor="middle"
              fontSize={11}
              fill="currentColor"
              opacity={0.6}
              fontStyle="italic"
            >
              x&#x2081;
            </text>
            <text
              x={8}
              y={RIGHT_MARGIN.top + rightPlotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 8, ${RIGHT_MARGIN.top + rightPlotH / 2})`}
            >
              f(x&#x2081; | x&#x2082;)
            </text>

            {/* Panel title */}
            <text
              x={RIGHT_MARGIN.left + rightPlotW / 2}
              y={14}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              fontWeight={600}
            >
              Conditional: X&#x2081; | X&#x2082; = {condY.toFixed(1)}
            </text>
          </svg>

          {/* Annotations below the right panel */}
          <div className="flex flex-col gap-1 mt-1 px-1">
            <div className="text-[10px] flex items-center gap-1.5">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ background: COLOR_COND_MEAN }}
              />
              <span>Mean varies with x&#x2082;: &#x03BC;<sub>1|2</sub> = {condMean.toFixed(3)}</span>
            </div>
            <div className="text-[10px] flex items-center gap-1.5">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ background: COLOR_COND_CURVE }}
              />
              <span>Variance constant: &#x03C3;&#x00B2;<sub>1|2</sub> = {condVar.toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
