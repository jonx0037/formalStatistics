import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { expectationMultinomial, covarianceMultinomial } from './shared/distributions';

// ── Constants ─────────────────────────────────────────────────────────────────

const K = 3;
const CATEGORY_COLORS = ['#2563eb', '#f59e0b', '#10b981'];
const CATEGORY_LABELS = ['Cat 1', 'Cat 2', 'Cat 3'];
const NUM_DRAWS = 200;
const VISIBLE_DOTS = 10;
const MARGIN = { top: 12, right: 16, bottom: 34, left: 44 };
const SCATTER_MARGIN = { top: 12, right: 16, bottom: 34, left: 44 };

// ── Multinomial sampling via sequential conditional Binomial ───────────────────

function sampleMultinomial(n: number, p: number[], rng = Math.random): number[] {
  const k = p.length;
  const x = new Array(k).fill(0);
  let remaining = n;
  let pRemaining = 1;
  for (let j = 0; j < k - 1; j++) {
    const prob = pRemaining > 0 ? p[j] / pRemaining : 0;
    for (let i = 0; i < remaining; i++) {
      if (rng() < prob) x[j]++;
    }
    remaining -= x[j];
    pRemaining -= p[j];
  }
  x[k - 1] = remaining;
  return x;
}

// ── Sum-to-1 slider constraint ────────────────────────────────────────────────

function adjustProbs(probs: number[], changedIndex: number, newValue: number): number[] {
  const clamped = Math.max(0.01, Math.min(0.98, newValue));
  const next = [...probs];
  next[changedIndex] = clamped;

  let sumOther = 0;
  for (let i = 0; i < K; i++) {
    if (i !== changedIndex) sumOther += probs[i];
  }

  const remaining = 1 - clamped;
  for (let i = 0; i < K; i++) {
    if (i === changedIndex) continue;
    if (sumOther > 0) {
      next[i] = Math.max(0.01, Math.min(0.98, probs[i] * (remaining / sumOther)));
    } else {
      next[i] = remaining / (K - 1);
    }
  }

  // Normalize to fix floating-point drift
  const total = next.reduce((s, v) => s + v, 0);
  for (let i = 0; i < K; i++) next[i] /= total;

  return next;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function MultinomialExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const [probs, setProbs] = useState<number[]>([1 / 3, 1 / 3, 1 / 3]);
  const [n, setN] = useState(20);
  const [drawSeed, setDrawSeed] = useState(0);

  // Generate all 200 draws whenever parameters or seed change
  const draws = useMemo(() => {
    // drawSeed is used to trigger re-generation
    void drawSeed;
    const result: number[][] = [];
    for (let i = 0; i < NUM_DRAWS; i++) {
      result.push(sampleMultinomial(n, probs));
    }
    return result;
  }, [n, probs, drawSeed]);

  // Theoretical moments
  const expectedCounts = useMemo(() => expectationMultinomial(n, probs), [n, probs]);
  const covMatrix = useMemo(() => covarianceMultinomial(n, probs), [n, probs]);

  // Sample correlation of (X1, X2) from the draws
  const sampleCorrelation = useMemo(() => {
    const x1 = draws.map((d) => d[0]);
    const x2 = draws.map((d) => d[1]);
    const m1 = x1.reduce((s, v) => s + v, 0) / NUM_DRAWS;
    const m2 = x2.reduce((s, v) => s + v, 0) / NUM_DRAWS;
    let cov = 0;
    let var1 = 0;
    let var2 = 0;
    for (let i = 0; i < NUM_DRAWS; i++) {
      const d1 = x1[i] - m1;
      const d2 = x2[i] - m2;
      cov += d1 * d2;
      var1 += d1 * d1;
      var2 += d2 * d2;
    }
    const denom = Math.sqrt(var1 * var2);
    return denom > 0 ? cov / denom : 0;
  }, [draws]);

  // Responsive layout
  const containerW = Math.max(320, (width || 700) - 16);
  const isNarrow = containerW < 620;
  const leftW = isNarrow ? containerW : Math.floor(containerW * 0.55);
  const rightW = isNarrow ? containerW : containerW - leftW - 12;
  const chartH = 240;

  // ── Left panel: Bar chart scales ───────────────────────────────────────────

  const barPlotW = leftW - MARGIN.left - MARGIN.right;
  const barPlotH = chartH - MARGIN.top - MARGIN.bottom;
  const barGroupWidth = barPlotW / K;
  const barWidth = barGroupWidth * 0.5;

  const toBarX = useCallback(
    (j: number) => MARGIN.left + j * barGroupWidth + barGroupWidth / 2,
    [barGroupWidth],
  );
  const toBarY = useCallback(
    (count: number) => MARGIN.top + barPlotH - (count / n) * barPlotH,
    [barPlotH, n],
  );

  // Y-axis ticks for bar chart
  const barYTicks = useMemo(() => {
    const step = n <= 20 ? 5 : n <= 50 ? 10 : 20;
    const ticks: number[] = [];
    for (let v = 0; v <= n; v += step) ticks.push(v);
    if (ticks[ticks.length - 1] !== n) ticks.push(n);
    return ticks;
  }, [n]);

  // ── Right panel: Scatter plot scales ───────────────────────────────────────

  const scatterPlotW = rightW - SCATTER_MARGIN.left - SCATTER_MARGIN.right;
  const scatterPlotH = chartH - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom;

  const toScatterX = useCallback(
    (v: number) => SCATTER_MARGIN.left + (v / n) * scatterPlotW,
    [scatterPlotW, n],
  );
  const toScatterY = useCallback(
    (v: number) => SCATTER_MARGIN.top + scatterPlotH - (v / n) * scatterPlotH,
    [scatterPlotH, n],
  );

  // Scatter axis ticks
  const scatterTicks = useMemo(() => {
    const step = n <= 20 ? 5 : n <= 50 ? 10 : 20;
    const ticks: number[] = [];
    for (let v = 0; v <= n; v += step) ticks.push(v);
    if (ticks[ticks.length - 1] !== n) ticks.push(n);
    return ticks;
  }, [n]);

  // Constraint line: X1 + X2 = n (from (n,0) to (0,n))
  const constraintLine = useMemo(() => {
    return {
      x1: toScatterX(n),
      y1: toScatterY(0),
      x2: toScatterX(0),
      y2: toScatterY(n),
    };
  }, [n, toScatterX, toScatterY]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleProbChange = (j: number, v: number) => {
    setProbs((prev) => adjustProbs(prev, j, v));
  };

  const handleDraw = () => {
    setDrawSeed((s) => s + 1);
  };

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Multinomial Distribution Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-end mb-4">
        {/* Probability sliders */}
        {probs.map((p, j) => (
          <label key={j} className="flex items-center gap-2 text-xs">
            <span className="font-medium" style={{ color: CATEGORY_COLORS[j] }}>
              p<sub>{j + 1}</sub>
            </span>
            <input
              type="range"
              min={0.01}
              max={0.98}
              step={0.01}
              value={p}
              onChange={(e) => handleProbChange(j, Number(e.target.value))}
              className="w-24"
            />
            <span className="w-10 tabular-nums text-right">{p.toFixed(2)}</span>
          </label>
        ))}

        {/* n slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-24"
          />
          <span className="w-8 tabular-nums text-right">{n}</span>
        </label>

        {/* Draw button */}
        <button
          onClick={handleDraw}
          className="px-3 py-1 text-xs font-medium rounded border"
          style={{
            borderColor: 'var(--color-border)',
            background: 'var(--color-bg)',
          }}
        >
          Draw Samples
        </button>
      </div>

      {/* Two-panel layout */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-3'}>
        {/* Left panel: Bar chart */}
        <div style={{ width: leftW }}>
          <svg width={leftW} height={chartH} className="block">
            {/* Y grid */}
            {barYTicks.map((v, i) => (
              <line
                key={i}
                x1={MARGIN.left}
                y1={toBarY(v)}
                x2={leftW - MARGIN.right}
                y2={toBarY(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Expected count bars (outlined) */}
            {expectedCounts.map((ej, j) => {
              const cx = toBarX(j);
              const top = toBarY(ej);
              const bottom = toBarY(0);
              const h = bottom - top;
              return (
                <rect
                  key={`exp-${j}`}
                  x={cx - barWidth / 2}
                  y={top}
                  width={barWidth}
                  height={Math.max(h, 0)}
                  fill="none"
                  stroke={CATEGORY_COLORS[j]}
                  strokeWidth={2}
                  strokeDasharray="5,3"
                  rx={2}
                />
              );
            })}

            {/* E[Xj] labels */}
            {expectedCounts.map((ej, j) => (
              <text
                key={`elabel-${j}`}
                x={toBarX(j)}
                y={toBarY(ej) - 6}
                textAnchor="middle"
                fontSize={9}
                fill={CATEGORY_COLORS[j]}
                fontWeight={600}
              >
                {ej.toFixed(1)}
              </text>
            ))}

            {/* Sample dots: show first VISIBLE_DOTS draws, jittered horizontally */}
            {draws.slice(0, VISIBLE_DOTS).map((draw, di) =>
              draw.map((count, j) => {
                const cx = toBarX(j);
                const jitter = ((di - VISIBLE_DOTS / 2) / VISIBLE_DOTS) * barWidth * 0.8;
                return (
                  <circle
                    key={`dot-${di}-${j}`}
                    cx={cx + jitter}
                    cy={toBarY(count)}
                    r={3}
                    fill={CATEGORY_COLORS[j]}
                    fillOpacity={0.55}
                    stroke={CATEGORY_COLORS[j]}
                    strokeWidth={0.5}
                  />
                );
              }),
            )}

            {/* X axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + barPlotH}
              x2={leftW - MARGIN.right}
              y2={MARGIN.top + barPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {/* Y axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + barPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis category labels */}
            {CATEGORY_LABELS.map((label, j) => (
              <text
                key={j}
                x={toBarX(j)}
                y={MARGIN.top + barPlotH + 16}
                textAnchor="middle"
                fontSize={10}
                fill={CATEGORY_COLORS[j]}
                fontWeight={600}
              >
                {label}
              </text>
            ))}

            {/* Y-axis tick labels */}
            {barYTicks.map((v, i) => (
              <text
                key={i}
                x={MARGIN.left - 6}
                y={toBarY(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {v}
              </text>
            ))}

            {/* Y-axis label */}
            <text
              x={12}
              y={MARGIN.top + barPlotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + barPlotH / 2})`}
            >
              Count
            </text>
          </svg>
          <div className="text-center text-[10px] opacity-50 mt-1">
            Dashed bars = E[X<sub>j</sub>] = np<sub>j</sub>. Dots = individual draws.
          </div>
        </div>

        {/* Right panel: Covariance scatter of (X1, X2) */}
        <div style={{ width: rightW }}>
          <svg width={rightW} height={chartH} className="block">
            {/* Grid */}
            {scatterTicks.map((v, i) => (
              <g key={i}>
                <line
                  x1={SCATTER_MARGIN.left}
                  y1={toScatterY(v)}
                  x2={rightW - SCATTER_MARGIN.right}
                  y2={toScatterY(v)}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
                <line
                  x1={toScatterX(v)}
                  y1={SCATTER_MARGIN.top}
                  x2={toScatterX(v)}
                  y2={SCATTER_MARGIN.top + scatterPlotH}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
              </g>
            ))}

            {/* Constraint region: X1 + X2 <= n (triangular) */}
            <polygon
              points={`${toScatterX(0)},${toScatterY(0)} ${toScatterX(n)},${toScatterY(0)} ${toScatterX(0)},${toScatterY(n)}`}
              fill="currentColor"
              fillOpacity={0.03}
            />

            {/* Constraint line: X1 + X2 = n */}
            <line
              x1={constraintLine.x1}
              y1={constraintLine.y1}
              x2={constraintLine.x2}
              y2={constraintLine.y2}
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeWidth={1.5}
              strokeDasharray="4,3"
            />

            {/* Scatter dots */}
            {draws.map((draw, i) => (
              <circle
                key={i}
                cx={toScatterX(draw[0])}
                cy={toScatterY(draw[1])}
                r={2.5}
                fill={CATEGORY_COLORS[0]}
                fillOpacity={0.35}
                stroke={CATEGORY_COLORS[0]}
                strokeWidth={0.4}
                strokeOpacity={0.5}
              />
            ))}

            {/* Axes */}
            <line
              x1={SCATTER_MARGIN.left}
              y1={SCATTER_MARGIN.top + scatterPlotH}
              x2={rightW - SCATTER_MARGIN.right}
              y2={SCATTER_MARGIN.top + scatterPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={SCATTER_MARGIN.left}
              y1={SCATTER_MARGIN.top}
              x2={SCATTER_MARGIN.left}
              y2={SCATTER_MARGIN.top + scatterPlotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks */}
            {scatterTicks.map((v, i) => (
              <text
                key={i}
                x={toScatterX(v)}
                y={SCATTER_MARGIN.top + scatterPlotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {v}
              </text>
            ))}

            {/* Y-axis ticks */}
            {scatterTicks.map((v, i) => (
              <text
                key={i}
                x={SCATTER_MARGIN.left - 6}
                y={toScatterY(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {v}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={SCATTER_MARGIN.left + scatterPlotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={10}
              fill={CATEGORY_COLORS[0]}
              fontWeight={600}
            >
              X{'\u2081'}
            </text>
            <text
              x={10}
              y={SCATTER_MARGIN.top + scatterPlotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill={CATEGORY_COLORS[1]}
              fontWeight={600}
              transform={`rotate(-90, 10, ${SCATTER_MARGIN.top + scatterPlotH / 2})`}
            >
              X{'\u2082'}
            </text>

            {/* Constraint label */}
            <text
              x={toScatterX(n * 0.55)}
              y={toScatterY(n * 0.55) - 6}
              textAnchor="middle"
              fontSize={8}
              fill="currentColor"
              opacity={0.4}
              transform={`rotate(-45, ${toScatterX(n * 0.55)}, ${toScatterY(n * 0.55) - 6})`}
            >
              X{'\u2081'} + X{'\u2082'} = {n}
            </text>
          </svg>
          <div className="text-center text-[10px] mt-1">
            <span style={{ opacity: 0.5 }}>
              {`\u03C1(X\u2081, X\u2082) = `}
            </span>
            <span className="font-semibold tabular-nums">{sampleCorrelation.toFixed(3)}</span>
            <span style={{ opacity: 0.4 }}> (sample)</span>
          </div>
        </div>
      </div>

      {/* Bottom panel: Moments table */}
      <div className="mt-4 pt-3 border-t overflow-x-auto" style={{ borderColor: 'var(--color-border)' }}>
        <table className="w-full text-xs text-center border-collapse">
          <thead>
            <tr>
              <th className="pb-1.5 px-2 text-left font-semibold opacity-70">Category</th>
              <th className="pb-1.5 px-2 font-semibold opacity-70">E[X<sub>j</sub>] = np<sub>j</sub></th>
              <th className="pb-1.5 px-2 font-semibold opacity-70">Var(X<sub>j</sub>) = np<sub>j</sub>(1-p<sub>j</sub>)</th>
              {[1, 2].map((j) => (
                <th key={j} className="pb-1.5 px-2 font-semibold opacity-70">
                  Cov(X<sub>j</sub>, X<sub>{j + 1}</sub>)
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CATEGORY_LABELS.map((label, j) => (
              <tr key={j} className="border-t" style={{ borderColor: 'var(--color-border)' }}>
                <td className="py-1.5 px-2 text-left font-medium" style={{ color: CATEGORY_COLORS[j] }}>
                  {label}
                </td>
                <td className="py-1.5 px-2 tabular-nums">{expectedCounts[j].toFixed(2)}</td>
                <td className="py-1.5 px-2 tabular-nums">{covMatrix[j][j].toFixed(2)}</td>
                {/* Covariances with the other two categories */}
                {[0, 1, 2]
                  .filter((i) => i !== j)
                  .slice(0, 2)
                  .map((i) => (
                    <td key={i} className="py-1.5 px-2 tabular-nums" style={{ color: '#dc2626' }}>
                      {covMatrix[j][i].toFixed(2)}
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-[10px] opacity-40 text-center mt-1.5">
          Off-diagonal covariances are negative: Cov(X<sub>i</sub>, X<sub>j</sub>) = -np<sub>i</sub>p<sub>j</sub> (the fixed-total constraint).
        </div>
      </div>
    </div>
  );
}
