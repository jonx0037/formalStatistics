import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { equallyLikelyP, intersection, areIndependent, conditionalP } from './shared/probability';
import { dieIndependencePresets } from '../../data/conditional-probability-data';

type Mode = 'single' | 'double';
type SelectingFor = 'A' | 'B';

/** Generate Omega for single die: {"1","2","3","4","5","6"} */
function singleDieOmega(): Set<string> {
  return new Set(['1', '2', '3', '4', '5', '6']);
}

/** Generate Omega for two dice: {"1,1","1,2",...,"6,6"} */
function doubleDieOmega(): Set<string> {
  const omega = new Set<string>();
  for (let d1 = 1; d1 <= 6; d1++) {
    for (let d2 = 1; d2 <= 6; d2++) {
      omega.add(`${d1},${d2}`);
    }
  }
  return omega;
}

/** Color for an outcome based on membership in A, B, both, or neither. */
function outcomeColor(
  outcome: string,
  eventA: Set<string>,
  eventB: Set<string>,
): { fill: string; opacity: number } {
  const inA = eventA.has(outcome);
  const inB = eventB.has(outcome);
  if (inA && inB) return { fill: '#7c3aed', opacity: 0.35 }; // purple — A∩B
  if (inA) return { fill: '#2563eb', opacity: 0.3 };          // blue — A only
  if (inB) return { fill: '#dc2626', opacity: 0.25 };         // red — B only
  return { fill: 'var(--color-surface)', opacity: 1 };        // neither
}

/** Format a probability as a fraction-like decimal. */
function fmtP(value: number): string {
  return Number.isNaN(value) ? 'undefined' : value.toFixed(4);
}

export default function IndependenceTester() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [mode, setMode] = useState<Mode>('single');
  const [eventA, setEventA] = useState<Set<string>>(new Set());
  const [eventB, setEventB] = useState<Set<string>>(new Set());
  const [selectingFor, setSelectingFor] = useState<SelectingFor>('A');

  // Build sample space based on mode
  const omega = useMemo(() => (mode === 'single' ? singleDieOmega() : doubleDieOmega()), [mode]);

  // Derived probabilities
  const pA = equallyLikelyP(eventA, omega);
  const pB = equallyLikelyP(eventB, omega);
  const aIntersectB = intersection(eventA, eventB);
  const pAB = equallyLikelyP(aIntersectB, omega);
  const pApB = pA * pB;
  const independent = areIndependent(pA, pB, pAB);
  const diff = pAB - pApB;
  const pAgivenB = conditionalP(pAB, pB);

  // Toggle an outcome in the currently selected event
  function toggleOutcome(outcome: string) {
    if (selectingFor === 'A') {
      setEventA((prev) => {
        const next = new Set(prev);
        if (next.has(outcome)) next.delete(outcome);
        else next.add(outcome);
        return next;
      });
    } else {
      setEventB((prev) => {
        const next = new Set(prev);
        if (next.has(outcome)) next.delete(outcome);
        else next.add(outcome);
        return next;
      });
    }
  }

  // Switch mode and clear events
  function switchMode(newMode: Mode) {
    setMode(newMode);
    setEventA(new Set());
    setEventB(new Set());
  }

  // Apply preset
  function applyPreset(preset: (typeof dieIndependencePresets)[number]) {
    if (mode !== 'single') setMode('single');
    setEventA(new Set(preset.eventA.outcomes));
    setEventB(new Set(preset.eventB.outcomes));
  }

  // ── Sizing ───────────────────────────────────────────────────────────
  const singleR = Math.min(Math.max((width - 70) / 6, 28), 48);
  const singleGap = Math.min(singleR * 0.4, 12);

  const gridPad = 8;
  const gridLabelW = 28;
  const availableGrid = width - gridPad * 2 - gridLabelW;
  const cellSize = Math.max(Math.min(Math.floor(availableGrid / 6), 52), 24);
  const gridW = cellSize * 6;
  const gridH = cellSize * 6;

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Mode toggle */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {(['single', 'double'] as const).map((m) => (
          <button
            key={m}
            onClick={() => switchMode(m)}
            className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
            style={{
              background: mode === m ? '#2563eb' : 'var(--color-surface)',
              color: mode === m ? '#ffffff' : 'var(--color-text)',
              borderColor: mode === m ? '#2563eb' : 'var(--color-border)',
            }}
          >
            {m === 'single' ? 'Single Die' : 'Two Dice'}
          </button>
        ))}
      </div>

      {/* Event selector + clear */}
      <div className="flex flex-wrap items-center gap-1.5 mb-3">
        {(['A', 'B'] as const).map((ev) => (
          <button
            key={ev}
            onClick={() => setSelectingFor(ev)}
            className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
            style={{
              background: selectingFor === ev
                ? (ev === 'A' ? '#2563eb' : '#dc2626')
                : 'var(--color-surface)',
              color: selectingFor === ev ? '#ffffff' : 'var(--color-text)',
              borderColor: selectingFor === ev
                ? (ev === 'A' ? '#2563eb' : '#dc2626')
                : 'var(--color-border)',
            }}
          >
            Defining {ev}
          </button>
        ))}
        <button
          onClick={() => setEventA(new Set())}
          className="px-2 py-1 rounded text-xs font-medium transition-colors border cursor-pointer"
          style={{
            background: 'var(--color-surface)',
            color: 'var(--color-text-muted)',
            borderColor: 'var(--color-border)',
          }}
        >
          Clear A
        </button>
        <button
          onClick={() => setEventB(new Set())}
          className="px-2 py-1 rounded text-xs font-medium transition-colors border cursor-pointer"
          style={{
            background: 'var(--color-surface)',
            color: 'var(--color-text-muted)',
            borderColor: 'var(--color-border)',
          }}
        >
          Clear B
        </button>
      </div>

      {/* Presets (single die only) */}
      {mode === 'single' && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          <span className="text-xs self-center" style={{ color: 'var(--color-text-muted)' }}>
            Presets:
          </span>
          {dieIndependencePresets.map((preset) => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset)}
              className="px-2 py-0.5 rounded text-xs font-medium transition-colors border cursor-pointer"
              style={{
                background: 'var(--color-surface)',
                color: 'var(--color-text)',
                borderColor: 'var(--color-border)',
              }}
            >
              {preset.name}
            </button>
          ))}
        </div>
      )}

      {/* SVG outcome grid */}
      <div ref={containerRef} className="w-full">
        {mode === 'single' ? (
          /* ── Single die: row of 6 circles ── */
          <svg
            width={width}
            height={singleR * 2 + 24}
            viewBox={`0 0 ${width} ${singleR * 2 + 24}`}
            style={{ overflow: 'visible' }}
          >
            {[1, 2, 3, 4, 5, 6].map((face, i) => {
              const key = String(face);
              const totalW = 6 * singleR * 2 + 5 * singleGap;
              const offsetX = (width - totalW) / 2;
              const cx = offsetX + singleR + i * (singleR * 2 + singleGap);
              const cy = singleR + 4;
              const color = outcomeColor(key, eventA, eventB);
              return (
                <g
                  key={key}
                  onClick={() => toggleOutcome(key)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={singleR}
                    fill={color.fill}
                    fillOpacity={color.opacity}
                    stroke="var(--color-border-strong)"
                    strokeWidth={1.5}
                  />
                  <text
                    x={cx}
                    y={cy + 1}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={Math.max(singleR * 0.6, 12)}
                    fontWeight={600}
                    fill="var(--color-text-heading)"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >
                    {face}
                  </text>
                </g>
              );
            })}
            {/* Legend below */}
            <g transform={`translate(0, ${singleR * 2 + 14})`}>
              {[
                { label: 'A', fill: '#2563eb', opacity: 0.3 },
                { label: 'B', fill: '#dc2626', opacity: 0.25 },
                { label: 'A∩B', fill: '#7c3aed', opacity: 0.35 },
              ].map((item, idx) => {
                const lx = width / 2 - 90 + idx * 68;
                return (
                  <g key={item.label}>
                    <rect
                      x={lx}
                      y={-5}
                      width={10}
                      height={10}
                      rx={2}
                      fill={item.fill}
                      fillOpacity={item.opacity}
                      stroke={item.fill}
                      strokeWidth={1}
                    />
                    <text
                      x={lx + 14}
                      y={3}
                      fontSize={10}
                      fill="var(--color-text-muted)"
                      dominantBaseline="central"
                    >
                      {item.label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        ) : (
          /* ── Two dice: 6x6 grid ── */
          <svg
            width={width}
            height={gridH + gridLabelW + gridPad * 2 + 20}
            viewBox={`0 0 ${width} ${gridH + gridLabelW + gridPad * 2 + 20}`}
            style={{ overflow: 'visible' }}
          >
            {/* Column headers (Die 2) */}
            {[1, 2, 3, 4, 5, 6].map((d2, j) => {
              const offsetX = (width - gridW - gridLabelW) / 2 + gridLabelW;
              return (
                <text
                  key={`col-${d2}`}
                  x={offsetX + j * cellSize + cellSize / 2}
                  y={gridPad + 10}
                  textAnchor="middle"
                  fontSize={Math.max(cellSize * 0.28, 9)}
                  fill="var(--color-text-muted)"
                  fontWeight={500}
                >
                  {d2}
                </text>
              );
            })}
            {/* Die 2 axis label */}
            <text
              x={(width - gridW - gridLabelW) / 2 + gridLabelW + gridW / 2}
              y={gridPad}
              textAnchor="middle"
              fontSize={10}
              fill="var(--color-text-muted)"
              fontStyle="italic"
            >
              Die 2
            </text>
            {/* Row headers (Die 1) and cells */}
            {[1, 2, 3, 4, 5, 6].map((d1, i) => {
              const offsetX = (width - gridW - gridLabelW) / 2;
              const rowY = gridPad + 18 + i * cellSize;
              return (
                <g key={`row-${d1}`}>
                  {/* Row label */}
                  <text
                    x={offsetX + gridLabelW - 6}
                    y={rowY + cellSize / 2 + 1}
                    textAnchor="end"
                    dominantBaseline="central"
                    fontSize={Math.max(cellSize * 0.28, 9)}
                    fill="var(--color-text-muted)"
                    fontWeight={500}
                  >
                    {d1}
                  </text>
                  {/* Die 1 label on first row */}
                  {i === 0 && (
                    <text
                      x={offsetX}
                      y={rowY + gridH / 2}
                      textAnchor="middle"
                      fontSize={10}
                      fill="var(--color-text-muted)"
                      fontStyle="italic"
                      transform={`rotate(-90, ${offsetX}, ${rowY + gridH / 2})`}
                    >
                      Die 1
                    </text>
                  )}
                  {/* Cells */}
                  {[1, 2, 3, 4, 5, 6].map((d2, j) => {
                    const key = `${d1},${d2}`;
                    const color = outcomeColor(key, eventA, eventB);
                    const cx = offsetX + gridLabelW + j * cellSize;
                    return (
                      <g
                        key={key}
                        onClick={() => toggleOutcome(key)}
                        style={{ cursor: 'pointer' }}
                      >
                        <rect
                          x={cx + 1}
                          y={rowY + 1}
                          width={cellSize - 2}
                          height={cellSize - 2}
                          rx={3}
                          fill={color.fill}
                          fillOpacity={color.opacity}
                          stroke="var(--color-border)"
                          strokeWidth={1}
                        />
                        <text
                          x={cx + cellSize / 2}
                          y={rowY + cellSize / 2 + 1}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fontSize={Math.max(cellSize * 0.22, 8)}
                          fill="var(--color-text-muted)"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {d1},{d2}
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {/* Legend */}
            <g transform={`translate(0, ${gridPad + 18 + gridH + 8})`}>
              {[
                { label: 'A', fill: '#2563eb', opacity: 0.3 },
                { label: 'B', fill: '#dc2626', opacity: 0.25 },
                { label: 'A∩B', fill: '#7c3aed', opacity: 0.35 },
              ].map((item, idx) => {
                const lx = width / 2 - 90 + idx * 68;
                return (
                  <g key={item.label}>
                    <rect
                      x={lx}
                      y={-5}
                      width={10}
                      height={10}
                      rx={2}
                      fill={item.fill}
                      fillOpacity={item.opacity}
                      stroke={item.fill}
                      strokeWidth={1}
                    />
                    <text
                      x={lx + 14}
                      y={3}
                      fontSize={10}
                      fill="var(--color-text-muted)"
                      dominantBaseline="central"
                    >
                      {item.label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>

      {/* Live computation panel */}
      <div
        className="mt-4 rounded border p-3 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs sm:text-sm"
        style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)' }}
      >
        {/* Left column: probabilities */}
        <div className="space-y-0.5" style={{ color: 'var(--color-text)' }}>
          <div>
            <span style={{ color: '#2563eb' }} className="font-semibold">P(A)</span>{' '}
            = {eventA.size}/{omega.size} = {fmtP(pA)}
          </div>
          <div>
            <span style={{ color: '#dc2626' }} className="font-semibold">P(B)</span>{' '}
            = {eventB.size}/{omega.size} = {fmtP(pB)}
          </div>
          <div>
            P(A)·P(B) = {fmtP(pApB)}
          </div>
          <div>
            <span style={{ color: '#7c3aed' }} className="font-semibold">P(A∩B)</span>{' '}
            = {aIntersectB.size}/{omega.size} = {fmtP(pAB)}
          </div>
        </div>

        {/* Right column: verdict + conditional */}
        <div className="space-y-1 pt-1 sm:pt-0">
          {/* Independence verdict */}
          {eventA.size > 0 && eventB.size > 0 ? (
            <div className="font-semibold">
              {independent ? (
                <span style={{ color: '#16a34a' }}>Independent &#x2713;</span>
              ) : (
                <>
                  <span style={{ color: '#dc2626' }}>Dependent &#x2717;</span>
                  <span className="ml-2 font-normal" style={{ color: 'var(--color-text-muted)' }}>
                    Difference: {diff >= 0 ? '+' : ''}{fmtP(diff)}
                  </span>
                </>
              )}
            </div>
          ) : (
            <div className="font-medium" style={{ color: 'var(--color-text-muted)' }}>
              Select outcomes for both A and B
            </div>
          )}

          {/* P(A|B) comparison */}
          <div style={{ color: 'var(--color-text)' }}>
            {pB === 0 ? (
              <span style={{ color: 'var(--color-text-muted)' }}>
                P(B) = 0 — P(A|B) undefined
              </span>
            ) : (
              <>
                P(A|B) = {fmtP(pAgivenB)}{' '}
                <span style={{ color: 'var(--color-text-muted)' }}>
                  vs P(A) = {fmtP(pA)}
                </span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
