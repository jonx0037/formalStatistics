import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { benjaminiHochbergSweep, simulatePValues } from './shared/testing';
import { multipleTestingColors } from './shared/colorScales';
import {
  bhAlgorithmPresets,
  type BHAlgorithmPreset,
} from '../../data/multi-testing-data';

/**
 * BHAlgorithmVisualizer — Topic 20 §20.7 interactive artifact.
 *
 * Visualizes the BH step-up search as an animation walking from rank k = m
 * down to k = 1. At each k, the marker lights up red (fail: p_(k) > k·α/m)
 * or green (pass: p_(k) ≤ k·α/m). On the first pass, the search halts,
 * k* = k, and the rejection region shades in (all ranks ≤ k*).
 *
 * The sorted p-values are a scatter; the BH diagonal k·α/m is a green line;
 * the Bonferroni single-step threshold α/m is a dashed red reference.
 */

const MARGIN = { top: 14, right: 14, bottom: 42, left: 58 };
const H = 360;

const C = {
  bh: multipleTestingColors.bh,
  bonf: multipleTestingColors.bonf,
  grid: '#E2E8F0',
  axis: '#64748B',
  text: '#1E293B',
  markerPass: '#10B981',
  markerFail: '#DC2626',
  markerNeutral: '#475569',
  rejectRegion: '#D1FAE5',
  pointNonreject: '#94A3B8',
  pointReject: '#10B981',
};

function formatP(p: number): string {
  if (p < 1e-4) return p.toExponential(2);
  return p.toFixed(4);
}

function resolvePvals(preset: BHAlgorithmPreset): number[] {
  if (preset.pvals && preset.pvals.length > 0) return preset.pvals;
  if (preset.generator) {
    const g = preset.generator;
    return simulatePValues(g.m, g.pi0, g.signalStrength, g.seed);
  }
  return [];
}

export default function BHAlgorithmVisualizer(): React.ReactElement {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 780, 320);
  const isWide = w > 760;

  const [presetIndex, setPresetIndex] = useState<number>(0);
  const preset: BHAlgorithmPreset = bhAlgorithmPresets[presetIndex];
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [speedMs, setSpeedMs] = useState<number>(400);
  const [currentK, setCurrentK] = useState<number | null>(null); // null = not started
  const [halted, setHalted] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const pvals = useMemo(() => resolvePvals(preset), [preset]);
  const m = pvals.length;
  const sweepResult = useMemo(
    () => benjaminiHochbergSweep(pvals, alpha),
    [pvals, alpha],
  );
  const kStar = sweepResult.kStar;
  const sweep = sweepResult.sweep;

  // Reset when preset or alpha changes.
  useEffect(() => {
    setCurrentK(null);
    setHalted(false);
    setIsPlaying(false);
  }, [presetIndex, alpha]);

  // Play loop — one step per speedMs.
  useEffect(() => {
    if (!isPlaying || halted) return;
    const id = window.setTimeout(() => {
      setCurrentK((k) => {
        const next = k === null ? m : k - 1;
        if (next < 1) {
          setHalted(true);
          setIsPlaying(false);
          return 0;
        }
        // Stop at first pass.
        if (sweep[next - 1].reject) {
          setHalted(true);
          setIsPlaying(false);
        }
        return next;
      });
    }, speedMs);
    return () => window.clearTimeout(id);
  }, [isPlaying, halted, currentK, speedMs, sweep, m]);

  const handlePlayPause = (): void => {
    if (halted) {
      // Reset + play from scratch.
      setCurrentK(null);
      setHalted(false);
      setIsPlaying(true);
    } else {
      setIsPlaying((p) => !p);
    }
  };

  const handleStep = (): void => {
    if (halted) return;
    setCurrentK((k) => {
      const next = k === null ? m : k - 1;
      if (next < 1) {
        setHalted(true);
        return 0;
      }
      if (sweep[next - 1].reject) setHalted(true);
      return next;
    });
  };

  const handleReset = (): void => {
    setCurrentK(null);
    setHalted(false);
    setIsPlaying(false);
  };

  const handleRunToEnd = (): void => {
    setCurrentK(kStar > 0 ? kStar : 1);
    setHalted(true);
    setIsPlaying(false);
  };

  // Layout.
  const plotW = isWide ? Math.max(440, w - 320) : w - 16;
  const plotH = H;
  const innerW = plotW - MARGIN.left - MARGIN.right;
  const innerH = plotH - MARGIN.top - MARGIN.bottom;

  const xScale = useMemo(
    () => d3.scaleLinear().domain([0, m + 1]).range([MARGIN.left, MARGIN.left + innerW]),
    [m, innerW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, 1]).range([MARGIN.top + innerH, MARGIN.top]),
    [innerH],
  );

  const xTicks = useMemo(() => xScale.ticks(Math.min(10, m + 1)), [xScale, m]);
  const yTicks = useMemo(() => yScale.ticks(5), [yScale]);

  // Visible rank (the "active" marker) — currentK, unless halted at k* in
  // which case we show the k* highlight.
  const activeK = currentK ?? m + 1;
  const activeEntry = activeK >= 1 && activeK <= m ? sweep[activeK - 1] : null;
  const markerFill = activeEntry
    ? activeEntry.reject
      ? C.markerPass
      : C.markerFail
    : C.markerNeutral;

  // BH diagonal k·α/m as a polyline.
  const diagonalPath = useMemo(() => {
    const pts: [number, number][] = [];
    for (let k = 1; k <= m; k++) pts.push([k, (k * alpha) / m]);
    const line = d3
      .line<[number, number]>()
      .x((d) => xScale(d[0]))
      .y((d) => yScale(Math.min(d[1], 1)));
    return line(pts) ?? '';
  }, [m, alpha, xScale, yScale]);

  // Shaded rejection region — after halt and kStar > 0, rectangle x ∈ [1, kStar].
  const rejectionShade =
    halted && kStar > 0 && activeEntry?.reject
      ? {
          x: xScale(0.5),
          width: xScale(kStar + 0.5) - xScale(0.5),
          y: MARGIN.top,
          height: innerH,
        }
      : null;

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border border-slate-200 bg-slate-50 p-4"
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Preset</span>
          <select
            value={presetIndex}
            onChange={(e) => setPresetIndex(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {bhAlgorithmPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">α = {alpha.toFixed(3)}</span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.005}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">speed</span>
          <input
            type="range"
            min={100}
            max={1000}
            step={50}
            value={speedMs}
            onChange={(e) => setSpeedMs(Number(e.target.value))}
            className="w-24"
          />
          <span className="font-mono text-xs text-slate-500">{speedMs} ms</span>
        </label>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={handlePlayPause}
            className="rounded bg-slate-700 px-3 py-1 text-sm text-white hover:bg-slate-600"
          >
            {isPlaying ? 'Pause' : halted ? 'Replay' : 'Play'}
          </button>
          <button
            type="button"
            onClick={handleStep}
            disabled={halted}
            className="rounded bg-white px-3 py-1 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            Step
          </button>
          <button
            type="button"
            onClick={handleRunToEnd}
            className="rounded bg-white px-3 py-1 text-sm text-slate-700 hover:bg-slate-100"
          >
            Run to k*
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="rounded bg-white px-3 py-1 text-sm text-slate-700 hover:bg-slate-100"
          >
            Reset
          </button>
        </div>
      </div>

      <div className={isWide ? 'flex gap-4' : 'flex flex-col gap-4'}>
        <svg
          width={plotW}
          height={plotH}
          role="img"
          aria-label={`BH step-up algorithm on m=${m} sorted p-values at α=${alpha.toFixed(3)}`}
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {/* Rejection-region shade */}
          {rejectionShade && (
            <rect
              x={rejectionShade.x}
              y={rejectionShade.y}
              width={rejectionShade.width}
              height={rejectionShade.height}
              fill={C.rejectRegion}
              opacity={0.6}
            />
          )}

          {/* Gridlines */}
          {yTicks.map((t) => (
            <line
              key={`yg-${t}`}
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke={C.grid}
              strokeWidth={1}
            />
          ))}

          {/* Bonferroni reference α/m */}
          <line
            x1={MARGIN.left}
            x2={MARGIN.left + innerW}
            y1={yScale(alpha / m)}
            y2={yScale(alpha / m)}
            stroke={C.bonf}
            strokeWidth={1.2}
            strokeDasharray="5 3"
          />
          <text
            x={MARGIN.left + innerW - 8}
            y={yScale(alpha / m) - 5}
            textAnchor="end"
            fontSize={10}
            fill={C.bonf}
          >
            α/m (Bonferroni)
          </text>

          {/* BH diagonal */}
          <path d={diagonalPath} stroke={C.bh} strokeWidth={1.8} fill="none" />
          <text
            x={MARGIN.left + innerW - 8}
            y={yScale(alpha * 0.95) - 5}
            textAnchor="end"
            fontSize={10}
            fill={C.bh}
          >
            k·α/m (BH)
          </text>

          {/* Sorted p-value dots */}
          {sweep.map((entry) => {
            const isRejected = halted && entry.k <= kStar;
            return (
              <circle
                key={`p-${entry.k}`}
                cx={xScale(entry.k)}
                cy={yScale(Math.min(entry.p_k, 1))}
                r={4}
                fill={isRejected ? C.pointReject : C.pointNonreject}
                stroke={entry.k === activeK ? markerFill : 'white'}
                strokeWidth={entry.k === activeK ? 3 : 0.8}
              />
            );
          })}

          {/* Active-k vertical guide */}
          {activeEntry && (
            <line
              x1={xScale(activeK)}
              x2={xScale(activeK)}
              y1={MARGIN.top}
              y2={MARGIN.top + innerH}
              stroke={markerFill}
              strokeWidth={1.4}
              strokeDasharray="3 3"
              opacity={0.8}
            />
          )}

          {/* Axes */}
          <line
            x1={MARGIN.left}
            x2={MARGIN.left + innerW}
            y1={MARGIN.top + innerH}
            y2={MARGIN.top + innerH}
            stroke={C.axis}
          />
          <line
            x1={MARGIN.left}
            x2={MARGIN.left}
            y1={MARGIN.top}
            y2={MARGIN.top + innerH}
            stroke={C.axis}
          />
          {xTicks
            .filter((t) => Number.isInteger(t) && t >= 1 && t <= m)
            .map((t) => (
              <g key={`xt-${t}`}>
                <line
                  x1={xScale(t)}
                  x2={xScale(t)}
                  y1={MARGIN.top + innerH}
                  y2={MARGIN.top + innerH + 4}
                  stroke={C.axis}
                />
                <text
                  x={xScale(t)}
                  y={MARGIN.top + innerH + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill={C.text}
                >
                  {t}
                </text>
              </g>
            ))}
          {yTicks.map((t) => (
            <g key={`yt-${t}`}>
              <line
                x1={MARGIN.left - 4}
                x2={MARGIN.left}
                y1={yScale(t)}
                y2={yScale(t)}
                stroke={C.axis}
              />
              <text
                x={MARGIN.left - 6}
                y={yScale(t) + 4}
                textAnchor="end"
                fontSize={10}
                fill={C.text}
              >
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          <text
            x={MARGIN.left + innerW / 2}
            y={plotH - 8}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
          >
            rank k (1 = smallest p-value)
          </text>
          <text
            x={14}
            y={MARGIN.top + innerH / 2}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
            transform={`rotate(-90, 14, ${MARGIN.top + innerH / 2})`}
          >
            p-value
          </text>
        </svg>

        <div
          className={
            isWide
              ? 'w-[300px] flex-shrink-0 overflow-auto rounded border border-slate-200 bg-white p-3 text-xs'
              : 'overflow-auto rounded border border-slate-200 bg-white p-3 text-xs'
          }
          style={{ maxHeight: plotH }}
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="font-semibold text-slate-800">Step log</span>
            {halted && (
              <span className="font-mono text-slate-600">
                k* ={' '}
                <span style={{ color: kStar > 0 ? C.markerPass : C.markerFail }}>
                  {kStar}
                </span>
              </span>
            )}
          </div>
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-0.5 text-left font-medium">k</th>
                <th className="py-0.5 text-right font-medium">p_(k)</th>
                <th className="py-0.5 text-right font-medium">k·α/m</th>
                <th className="py-0.5 text-right font-medium">dec.</th>
              </tr>
            </thead>
            <tbody>
              {[...sweep].reverse().map((entry) => {
                const visited =
                  activeK !== m + 1 &&
                  entry.k >= (currentK ?? m + 1) &&
                  entry.k <= m;
                const isActive = entry.k === activeK;
                return (
                  <tr
                    key={entry.k}
                    className={`border-b border-slate-100 ${
                      isActive ? 'bg-amber-50 font-semibold' : visited ? '' : 'text-slate-400'
                    }`}
                  >
                    <td className="py-0.5 font-mono">{entry.k}</td>
                    <td className="py-0.5 text-right font-mono">{formatP(entry.p_k)}</td>
                    <td className="py-0.5 text-right font-mono">{formatP(entry.threshold)}</td>
                    <td
                      className="py-0.5 text-right font-mono"
                      style={{
                        color: visited
                          ? entry.reject
                            ? C.markerPass
                            : C.markerFail
                          : undefined,
                      }}
                    >
                      {visited ? (entry.reject ? '✓' : '✗') : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        {preset.description} Expected k* = {preset.expectedKStar}; current k* ={' '}
        {kStar}. The step-up search walks down from k = m, returning the first
        rank where p_(k) ≤ k·α/m.
      </p>
    </div>
  );
}
