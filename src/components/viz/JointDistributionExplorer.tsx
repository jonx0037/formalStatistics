import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBivariateNormal, pdfNormal, conditionalNormalParams,
} from './shared/distributions';
import { bivariateNormalPresets } from '../../data/random-variables-data';
import { distributionColors } from './shared/colorScales';

type Mode = 'discrete' | 'continuous';

// ── Layout ─────────────────────────────────────────────────────────────────

const MARGIN = { top: 50, right: 50, bottom: 40, left: 50 };
const MARGINAL_H = 40; // height for marginal density plots

// ── Discrete: Two dice joint PMF ───────────────────────────────────────────

function DiscreteModePanel({
  panelSize, showMarginals, showConditional, conditionY,
}: {
  panelSize: number;
  showMarginals: boolean;
  showConditional: boolean;
  conditionY: number;
}) {
  const gridSize = 6;
  const plotSize = panelSize - MARGIN.left - MARGIN.right;
  const cellSize = plotSize / gridSize;

  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);

  // Joint PMF: all 1/36
  const jointPMF = 1 / 36;

  // Marginal PMFs: all 1/6
  const marginalPMF = 1 / 6;

  // Conditional PMF: p(X=x | Y=y) = p(X=x, Y=y) / p(Y=y) = (1/36) / (1/6) = 1/6
  const conditionalPMF = 1 / 6;

  return (
    <svg width={panelSize} height={panelSize + (showConditional ? 100 : 0)} className="block">
      {/* Heatmap */}
      {Array.from({ length: gridSize }, (_, row) =>
        Array.from({ length: gridSize }, (_, col) => {
          const x = col + 1;
          const y = row + 1;
          const isConditioned = showConditional && y === conditionY;
          const isHovered = hoveredCell?.x === x && hoveredCell?.y === y;
          return (
            <g key={`${x}-${y}`}>
              <rect
                x={MARGIN.left + col * cellSize}
                y={MARGIN.top + (gridSize - row - 1) * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                fill={isConditioned ? '#2563eb' : '#6b7280'}
                opacity={isHovered ? 0.8 : isConditioned ? 0.5 : 0.2}
                stroke={isHovered ? '#000' : 'none'}
                strokeWidth={isHovered ? 2 : 0}
                rx={2}
                onMouseEnter={() => setHoveredCell({ x, y })}
                onMouseLeave={() => setHoveredCell(null)}
                className="cursor-pointer"
              />
              <text
                x={MARGIN.left + col * cellSize + cellSize / 2}
                y={MARGIN.top + (gridSize - row - 1) * cellSize + cellSize / 2 + 3}
                textAnchor="middle"
                className="fill-current"
                style={{ fontSize: cellSize > 40 ? '9px' : '7px', pointerEvents: 'none' }}
              >
                {(jointPMF).toFixed(3)}
              </text>
            </g>
          );
        })
      )}

      {/* Axis labels */}
      {Array.from({ length: gridSize }, (_, i) => (
        <g key={`label-${i}`}>
          <text
            x={MARGIN.left + i * cellSize + cellSize / 2}
            y={MARGIN.top + plotSize + 16}
            textAnchor="middle"
            className="fill-current"
            style={{ fontSize: '11px' }}
          >
            {i + 1}
          </text>
          <text
            x={MARGIN.left - 10}
            y={MARGIN.top + (gridSize - i - 1) * cellSize + cellSize / 2 + 4}
            textAnchor="end"
            className="fill-current"
            style={{ fontSize: '11px' }}
          >
            {i + 1}
          </text>
        </g>
      ))}
      <text x={MARGIN.left + plotSize / 2} y={MARGIN.top + plotSize + 32} textAnchor="middle" className="fill-current text-xs font-medium">X (Die 1)</text>
      <text x={12} y={MARGIN.top + plotSize / 2} textAnchor="middle" className="fill-current text-xs font-medium" transform={`rotate(-90, 12, ${MARGIN.top + plotSize / 2})`}>Y (Die 2)</text>

      {/* Top marginal bars */}
      {showMarginals && Array.from({ length: gridSize }, (_, i) => (
        <rect
          key={`margX-${i}`}
          x={MARGIN.left + i * cellSize + 2}
          y={MARGIN.top - MARGINAL_H * marginalPMF * 4}
          width={cellSize - 4}
          height={MARGINAL_H * marginalPMF * 4}
          fill={distributionColors.pdf}
          opacity={0.4}
          rx={1}
        />
      ))}

      {/* Right marginal bars */}
      {showMarginals && Array.from({ length: gridSize }, (_, i) => (
        <rect
          key={`margY-${i}`}
          x={MARGIN.left + plotSize + 4}
          y={MARGIN.top + (gridSize - i - 1) * cellSize + 2}
          width={MARGINAL_H * marginalPMF * 4}
          height={cellSize - 4}
          fill={distributionColors.cdf}
          opacity={0.4}
          rx={1}
        />
      ))}

      {/* Conditional row selection */}
      {showConditional && (
        <>
          {/* Highlight the conditioned row */}
          <rect
            x={MARGIN.left - 2}
            y={MARGIN.top + (gridSize - conditionY) * cellSize - 1}
            width={plotSize + 4}
            height={cellSize + 2}
            fill="none"
            stroke="#2563eb"
            strokeWidth={2}
            rx={3}
          />
          {/* Conditional PMF bar chart below */}
          <text
            x={MARGIN.left + plotSize / 2}
            y={panelSize + 10}
            textAnchor="middle"
            className="fill-current text-xs font-medium"
          >
            p(X | Y = {conditionY}) — each bar = {conditionalPMF.toFixed(4)}
          </text>
          {Array.from({ length: gridSize }, (_, i) => (
            <rect
              key={`cond-${i}`}
              x={MARGIN.left + i * cellSize + 4}
              y={panelSize + 20}
              width={cellSize - 8}
              height={60 * conditionalPMF}
              fill="#2563eb"
              opacity={0.6}
              rx={1}
            />
          ))}
        </>
      )}

      {/* Hovered cell readout */}
      {hoveredCell && (
        <text
          x={panelSize / 2}
          y={MARGIN.top - 8}
          textAnchor="middle"
          className="fill-current text-xs font-mono"
        >
          p(X={hoveredCell.x}, Y={hoveredCell.y}) = 1/36 ≈ 0.0278
        </text>
      )}
    </svg>
  );
}

// ── Continuous: Bivariate normal ───────────────────────────────────────────

function ContinuousModePanel({
  panelSize, rho, conditionY, showConditional,
}: {
  panelSize: number;
  rho: number;
  conditionY: number;
  showConditional: boolean;
}) {
  const plotSize = panelSize - MARGIN.left - MARGIN.right;
  const range = 3.5;

  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val + range) / (2 * range)) * plotSize,
    [plotSize],
  );
  const yScale = useCallback(
    (val: number) => MARGIN.top + plotSize - ((val + range) / (2 * range)) * plotSize,
    [plotSize],
  );

  // Contour levels
  const contourData = useMemo(() => {
    const gridN = 80;
    const step = (2 * range) / gridN;
    const grid: number[][] = [];
    for (let j = 0; j <= gridN; j++) {
      const row: number[] = [];
      for (let i = 0; i <= gridN; i++) {
        const x = -range + i * step;
        const y = -range + j * step;
        row.push(pdfBivariateNormal(x, y, 0, 0, 1, 1, rho));
      }
      grid.push(row);
    }
    return { grid, gridN, step };
  }, [rho]);

  // Simple contour rendering: draw filled circles at density thresholds
  const contourCircles = useMemo(() => {
    const levels = [0.01, 0.03, 0.06, 0.1, 0.15];
    const circles: { cx: number; cy: number; density: number }[] = [];
    const { grid, gridN, step } = contourData;
    for (let j = 0; j <= gridN; j += 2) {
      for (let i = 0; i <= gridN; i += 2) {
        const x = -range + i * step;
        const y = -range + j * step;
        const d = grid[j][i];
        if (d > levels[0]) {
          circles.push({ cx: xScale(x), cy: yScale(y), density: d });
        }
      }
    }
    return circles;
  }, [contourData, xScale, yScale]);

  const maxDensity = useMemo(() => {
    return pdfBivariateNormal(0, 0, 0, 0, 1, 1, rho);
  }, [rho]);

  // Marginal PDFs (always standard normal marginals for standard bivariate normal)
  const marginalPoints = useMemo(() => {
    const nPts = 100;
    const pts: { val: number; density: number }[] = [];
    for (let i = 0; i <= nPts; i++) {
      const v = -range + (2 * range * i) / nPts;
      pts.push({ val: v, density: pdfNormal(v, 0, 1) });
    }
    return pts;
  }, []);

  // Conditional distribution
  const conditionalCurve = useMemo(() => {
    if (!showConditional) return null;
    const { condMean, condVar } = conditionalNormalParams(0, 0, 1, 1, rho, conditionY);
    const nPts = 100;
    const pts: { x: number; density: number }[] = [];
    for (let i = 0; i <= nPts; i++) {
      const x = -range + (2 * range * i) / nPts;
      pts.push({ x, density: pdfNormal(x, condMean, condVar) });
    }
    return { pts, condMean, condVar };
  }, [showConditional, rho, conditionY]);

  const isIndependent = Math.abs(rho) < 0.001;

  return (
    <svg width={panelSize} height={panelSize + (showConditional ? 120 : 0)} className="block">
      {/* Contour dots */}
      {contourCircles.map((c, i) => (
        <circle
          key={i}
          cx={c.cx}
          cy={c.cy}
          r={3}
          fill={distributionColors.pdf}
          opacity={0.15 + 0.7 * (c.density / maxDensity)}
        />
      ))}

      {/* Axes */}
      <line x1={MARGIN.left} y1={yScale(0)} x2={MARGIN.left + plotSize} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.2} />
      <line x1={xScale(0)} y1={MARGIN.top} x2={xScale(0)} y2={MARGIN.top + plotSize} stroke="currentColor" strokeOpacity={0.2} />

      {/* Top marginal (X) */}
      <path
        d={marginalPoints
          .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.val).toFixed(1)},${(MARGIN.top - 2 - p.density * MARGINAL_H * 2).toFixed(1)}`)
          .join(' ')}
        fill="none"
        stroke={distributionColors.pdf}
        strokeWidth={1.5}
        opacity={0.6}
      />

      {/* Right marginal (Y) */}
      <path
        d={marginalPoints
          .map((p, i) => `${i === 0 ? 'M' : 'L'}${(MARGIN.left + plotSize + 4 + p.density * MARGINAL_H * 2).toFixed(1)},${yScale(p.val).toFixed(1)}`)
          .join(' ')}
        fill="none"
        stroke={distributionColors.cdf}
        strokeWidth={1.5}
        opacity={0.6}
      />

      {/* Conditioning line */}
      {showConditional && (
        <line
          x1={MARGIN.left}
          y1={yScale(conditionY)}
          x2={MARGIN.left + plotSize}
          y2={yScale(conditionY)}
          stroke={distributionColors.sample}
          strokeWidth={1.5}
          strokeDasharray="6,3"
        />
      )}

      {/* Labels */}
      <text x={MARGIN.left + plotSize / 2} y={MARGIN.top + plotSize + 20} textAnchor="middle" className="fill-current text-xs">X</text>
      <text x={12} y={MARGIN.top + plotSize / 2} textAnchor="middle" className="fill-current text-xs" transform={`rotate(-90, 12, ${MARGIN.top + plotSize / 2})`}>Y</text>

      {/* Independence indicator */}
      <text
        x={MARGIN.left + plotSize}
        y={MARGIN.top + plotSize + 34}
        textAnchor="end"
        className="fill-current"
        style={{ fontSize: '11px', fontWeight: 600 }}
      >
        {isIndependent ? '✓ Independent (ρ = 0)' : `Dependent (ρ = ${rho.toFixed(2)})`}
      </text>

      {/* Conditional distribution subplot */}
      {showConditional && conditionalCurve && (
        <>
          <text
            x={MARGIN.left + plotSize / 2}
            y={panelSize + 10}
            textAnchor="middle"
            className="fill-current text-xs font-medium"
          >
            f(X | Y = {conditionY.toFixed(1)}) ~ N({conditionalCurve.condMean.toFixed(2)}, {conditionalCurve.condVar.toFixed(2)})
          </text>
          <path
            d={conditionalCurve.pts
              .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${(panelSize + 100 - p.density * 80).toFixed(1)}`)
              .join(' ')}
            fill="none"
            stroke={distributionColors.sample}
            strokeWidth={2}
          />
          {/* Baseline for conditional */}
          <line
            x1={MARGIN.left}
            y1={panelSize + 100}
            x2={MARGIN.left + plotSize}
            y2={panelSize + 100}
            stroke="currentColor"
            strokeOpacity={0.2}
          />
        </>
      )}
    </svg>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function JointDistributionExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [mode, setMode] = useState<Mode>('continuous');
  const [rho, setRho] = useState(0.5);
  const [showMarginals, setShowMarginals] = useState(true);
  const [showConditional, setShowConditional] = useState(false);
  const [conditionY, setConditionY] = useState(1.0);

  const panelSize = Math.min(Math.max(width - 32, 280), 450);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border" style={{ borderColor: 'var(--color-border)' }}>
          <button
            onClick={() => setMode('discrete')}
            className={mode === 'discrete' ? 'px-3 py-1 text-sm font-medium bg-blue-600 text-white' : 'px-3 py-1 text-sm font-medium bg-transparent'}
          >
            Discrete (Two Dice)
          </button>
          <button
            onClick={() => setMode('continuous')}
            className={mode === 'continuous' ? 'px-3 py-1 text-sm font-medium bg-blue-600 text-white' : 'px-3 py-1 text-sm font-medium bg-transparent'}
          >
            Continuous (Bivariate Normal)
          </button>
        </div>

        <label className="flex items-center gap-1 text-sm">
          <input
            type="checkbox"
            checked={showMarginals}
            onChange={(e) => setShowMarginals(e.target.checked)}
          />
          Marginals
        </label>

        <label className="flex items-center gap-1 text-sm">
          <input
            type="checkbox"
            checked={showConditional}
            onChange={(e) => setShowConditional(e.target.checked)}
          />
          Conditional
        </label>
      </div>

      {/* ρ slider (continuous mode) */}
      {mode === 'continuous' && (
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <label className="text-sm font-medium">ρ (correlation):</label>
          <input
            type="range"
            min={-0.95}
            max={0.95}
            step={0.05}
            value={rho}
            onChange={(e) => setRho(Number(e.target.value))}
            className="w-48"
          />
          <span className="text-sm font-mono w-12">{rho.toFixed(2)}</span>

          {/* Presets */}
          <div className="flex gap-1">
            {bivariateNormalPresets.map((p) => (
              <button
                key={p.name}
                onClick={() => setRho(p.rho)}
                className="rounded border px-2 py-0.5 text-xs"
                style={{ borderColor: 'var(--color-border)' }}
              >
                ρ={p.rho}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Condition Y slider */}
      {showConditional && (
        <div className="mb-4 flex items-center gap-3">
          <label className="text-sm font-medium">
            Condition on Y =
          </label>
          {mode === 'discrete' ? (
            <select
              value={conditionY}
              onChange={(e) => setConditionY(Number(e.target.value))}
              className="rounded border px-2 py-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
            >
              {[1, 2, 3, 4, 5, 6].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          ) : (
            <>
              <input
                type="range"
                min={-3}
                max={3}
                step={0.1}
                value={conditionY}
                onChange={(e) => setConditionY(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-sm font-mono">{conditionY.toFixed(1)}</span>
            </>
          )}
        </div>
      )}

      {/* Visualization */}
      {mode === 'discrete' ? (
        <DiscreteModePanel
          panelSize={panelSize}
          showMarginals={showMarginals}
          showConditional={showConditional}
          conditionY={conditionY}
        />
      ) : (
        <ContinuousModePanel
          panelSize={panelSize}
          rho={rho}
          conditionY={conditionY}
          showConditional={showConditional}
        />
      )}
    </div>
  );
}
