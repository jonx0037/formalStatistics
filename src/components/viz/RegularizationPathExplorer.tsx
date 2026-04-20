import { useMemo, useRef, useState, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { penaltyColors } from './shared/colorScales';
import {
  PROSTATE_CANCER_DATA,
  PROSTATE_RIDGE_PATH,
  PROSTATE_LASSO_PATH,
} from '../../data/regularization-data';

const MARGIN = { top: 24, right: 96, bottom: 44, left: 56 };
const PANEL_H = 320;
const AXIS_COLOR = '#94A3B8';
const TICK_COLOR = '#CBD5E1';

const FEATURE_COLOR = d3.scaleOrdinal<string, string>().range([
  '#2563EB',
  '#059669',
  '#DC2626',
  '#D97706',
  '#7C3AED',
  '#0891B2',
  '#BE185D',
  '#0F766E',
]);

export default function RegularizationPathExplorer() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 720;
  const panelW = isStacked ? width - 16 : (width - 24) / 2;

  // Slider position is an index into the lambda grid (descending from λ_max).
  const [lambdaIdx, setLambdaIdx] = useState(50);
  const [highlightFeature, setHighlightFeature] = useState<number | null>(null);

  const ridge = PROSTATE_RIDGE_PATH;
  const lasso = PROSTATE_LASSO_PATH;
  const featNames = PROSTATE_CANCER_DATA.featureNames;

  // Use ridge's lambda grid as the slider scale (both paths share log range).
  const nGrid = ridge.lambdas.length;
  const safeIdx = Math.min(Math.max(lambdaIdx, 0), nGrid - 1);
  const currentLambda = ridge.lambdas[safeIdx];

  // Active set at current λ (lasso only; ridge has no exact zeros).
  const lassoBeta = lasso.betas[safeIdx];
  const lassoActive = lassoBeta.reduce(
    (acc, b, j) => (j > 0 && Math.abs(b) > 1e-9 ? acc + 1 : acc),
    0,
  );

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3 flex flex-col gap-1">
        <div className="text-sm font-semibold text-gray-900">
          Regularization paths β̂<sub>j</sub>(λ) — prostate-cancer dataset
        </div>
        <div className="text-xs text-gray-600">
          Drag the slider to sweep λ from large (right, all coefficients shrunk
          to zero) to small (left, recovers OLS). Lasso paths are piecewise
          linear with active-set changes; ridge paths are smooth.
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex flex-col text-xs text-gray-700">
          <span>
            log<sub>10</sub>(λ) ={' '}
            <span className="font-mono">
              {Math.log10(currentLambda).toFixed(2)}
            </span>
          </span>
          <input
            type="range"
            min={0}
            max={nGrid - 1}
            value={safeIdx}
            onChange={(e) => setLambdaIdx(parseInt(e.target.value, 10))}
            aria-label="lambda slider"
            className="w-72"
          />
        </label>
        <div className="rounded bg-gray-50 px-2 py-1 text-xs text-gray-700">
          ridge dof ≈ <span className="font-mono">{ridge.dof[safeIdx].toFixed(2)}</span>
          {'  '}·{'  '}lasso |𝓐| ={' '}
          <span className="font-mono">{lassoActive}</span>
        </div>
      </div>

      <div
        className="flex flex-wrap gap-3"
        style={{ flexDirection: isStacked ? 'column' : 'row' }}
      >
        <Panel
          title="Ridge path"
          color={penaltyColors.ridge}
          path={ridge}
          width={panelW}
          highlightIdx={safeIdx}
          highlightFeature={highlightFeature}
          setHighlightFeature={setHighlightFeature}
          featNames={featNames}
        />
        <Panel
          title="Lasso path"
          color={penaltyColors.lasso}
          path={lasso}
          width={panelW}
          highlightIdx={safeIdx}
          highlightFeature={highlightFeature}
          setHighlightFeature={setHighlightFeature}
          featNames={featNames}
        />
      </div>

      <div className="mt-3 text-xs text-gray-600">
        <strong>Reading the figure.</strong> Each colored curve is one
        coefficient β̂<sub>j</sub>(λ) as λ varies. The vertical line marks
        the current λ. Hover a curve or its legend swatch to highlight one
        feature. Lasso curves enter the active set one at a time (piecewise
        linear, each elbow is an active-set change); ridge curves shrink
        smoothly.
      </div>
    </div>
  );
}

interface PanelProps {
  title: string;
  color: string;
  path: { lambdas: number[]; betas: number[][] };
  width: number;
  highlightIdx: number;
  highlightFeature: number | null;
  setHighlightFeature: (j: number | null) => void;
  featNames: readonly string[];
}

function Panel({
  title,
  color,
  path,
  width,
  highlightIdx,
  highlightFeature,
  setHighlightFeature,
  featNames,
}: PanelProps) {
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 100);
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  const ref = useRef<SVGSVGElement | null>(null);

  const xScale = useMemo(() => {
    const logLambdas = path.lambdas.map((l) => Math.log10(Math.max(l, 1e-12)));
    return d3
      .scaleLinear()
      .domain(d3.extent(logLambdas) as [number, number])
      .range([0, innerW]);
  }, [path.lambdas, innerW]);

  const yScale = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    for (const beta of path.betas) {
      for (let j = 1; j < beta.length; j++) {
        if (beta[j] < min) min = beta[j];
        if (beta[j] > max) max = beta[j];
      }
    }
    const pad = Math.max(0.1 * Math.max(Math.abs(min), Math.abs(max), 0.01), 0.02);
    return d3
      .scaleLinear()
      .domain([min - pad, max + pad])
      .range([innerH, 0])
      .nice();
  }, [path.betas, innerH]);

  // Build line generator for each feature: x = log λ, y = β̂_j(λ).
  const featureLines = useMemo(() => {
    if (path.betas.length === 0) return [];
    const p = path.betas[0].length - 1; // exclude intercept
    const lines: { j: number; d: string }[] = [];
    const lineGen = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y));
    for (let j = 1; j <= p; j++) {
      const data = path.betas.map((beta, i) => ({
        x: Math.log10(Math.max(path.lambdas[i], 1e-12)),
        y: beta[j],
      }));
      const d = lineGen(data) ?? '';
      lines.push({ j: j - 1, d });
    }
    return lines;
  }, [path.betas, path.lambdas, xScale, yScale]);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    const xAxisG = svg.select<SVGGElement>('.x-axis');
    const yAxisG = svg.select<SVGGElement>('.y-axis');
    xAxisG.call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    yAxisG.call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
    xAxisG.selectAll('path,line').attr('stroke', AXIS_COLOR);
    yAxisG.selectAll('path,line').attr('stroke', AXIS_COLOR);
    xAxisG.selectAll('text').attr('fill', '#475569').attr('font-size', 10);
    yAxisG.selectAll('text').attr('fill', '#475569').attr('font-size', 10);
  }, [xScale, yScale]);

  const currentX = xScale(Math.log10(Math.max(path.lambdas[highlightIdx], 1e-12)));

  return (
    <svg
      ref={ref}
      width={width}
      height={PANEL_H}
      role="img"
      aria-label={`${title} for prostate-cancer dataset`}
    >
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {/* Title + sub-title */}
        <text x={0} y={-8} fill={color} fontSize={12} fontWeight={600}>
          {title}
        </text>

        {/* Grid */}
        {yScale.ticks(6).map((t) => (
          <line
            key={`g-${t}`}
            x1={0}
            x2={innerW}
            y1={yScale(t)}
            y2={yScale(t)}
            stroke={TICK_COLOR}
            strokeWidth={0.5}
          />
        ))}

        {/* Zero line */}
        <line
          x1={0}
          x2={innerW}
          y1={yScale(0)}
          y2={yScale(0)}
          stroke="#64748b"
          strokeDasharray="4 3"
          strokeWidth={0.8}
        />

        {/* Coefficient curves */}
        {featureLines.map(({ j, d }) => {
          const isHighlighted = highlightFeature === j;
          const isDimmed = highlightFeature !== null && highlightFeature !== j;
          return (
            <path
              key={`f-${j}`}
              d={d}
              fill="none"
              stroke={FEATURE_COLOR(String(j))}
              strokeWidth={isHighlighted ? 2.6 : 1.6}
              opacity={isDimmed ? 0.18 : 0.95}
              style={{ cursor: 'pointer', transition: 'all 150ms' }}
              onMouseEnter={() => setHighlightFeature(j)}
              onMouseLeave={() => setHighlightFeature(null)}
            />
          );
        })}

        {/* Current-λ vertical line */}
        <line
          x1={currentX}
          x2={currentX}
          y1={0}
          y2={innerH}
          stroke="#0f172a"
          strokeWidth={1.2}
          strokeDasharray="2 2"
        />

        {/* Axes */}
        <g className="x-axis" transform={`translate(0, ${innerH})`} />
        <g className="y-axis" />
        <text
          x={innerW / 2}
          y={innerH + 32}
          textAnchor="middle"
          fontSize={11}
          fill="#64748b"
        >
          log₁₀(λ)
        </text>
        <text
          transform={`translate(-40, ${innerH / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={11}
          fill="#64748b"
        >
          β̂ⱼ
        </text>

        {/* Feature legend at right */}
        <g transform={`translate(${innerW + 8}, 4)`}>
          {featNames.map((name, j) => (
            <g
              key={`legend-${j}`}
              transform={`translate(0, ${j * 14})`}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHighlightFeature(j)}
              onMouseLeave={() => setHighlightFeature(null)}
            >
              <rect
                x={0}
                y={3}
                width={10}
                height={3}
                fill={FEATURE_COLOR(String(j))}
                opacity={highlightFeature !== null && highlightFeature !== j ? 0.3 : 1}
              />
              <text
                x={14}
                y={9}
                fontSize={9}
                fill={
                  highlightFeature !== null && highlightFeature !== j
                    ? '#94a3b8'
                    : '#1f2937'
                }
              >
                {name}
              </text>
            </g>
          ))}
        </g>
      </g>
    </svg>
  );
}
