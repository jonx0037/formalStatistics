import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { totalProbability } from './shared/probability';

const PALETTE = [
  { base: '#2563eb', dark: '#1d4ed8', label: 'B' },  // blue
  { base: '#16a34a', dark: '#15803d', label: 'B' },  // green
  { base: '#ea580c', dark: '#c2410c', label: 'B' },  // orange
  { base: '#7c3aed', dark: '#6d28d9', label: 'B' },  // purple
];

const PARTITION_SIZES = [2, 3, 4] as const;

/** Default P(Bi) arrays for each partition size, each summing to 1. */
function defaultPB(size: number): number[] {
  if (size === 2) return [0.4, 0.6];
  if (size === 3) return [0.3, 0.5, 0.2];
  return [0.25, 0.35, 0.25, 0.15];
}

/** Default P(A|Bi) arrays for each partition size. */
function defaultPAgivenB(size: number): number[] {
  if (size === 2) return [0.8, 0.3];
  if (size === 3) return [0.8, 0.4, 0.6];
  return [0.8, 0.4, 0.6, 0.5];
}

/**
 * Adjust P(B) values so they sum to 1 after changing index `changedIdx` to `newVal`.
 * Proportionally scales all other entries. If all others are 0, distributes equally.
 */
function adjustPartitionProbs(
  prev: number[],
  changedIdx: number,
  newVal: number,
): number[] {
  const clamped = Math.max(0, Math.min(1, newVal));
  const remaining = 1 - clamped;
  const next = [...prev];
  next[changedIdx] = clamped;

  const othersSum = prev.reduce(
    (s, v, i) => (i === changedIdx ? s : s + v),
    0,
  );

  if (othersSum === 0) {
    // Distribute remaining equally among others
    const otherCount = prev.length - 1;
    for (let i = 0; i < next.length; i++) {
      if (i !== changedIdx) {
        next[i] = otherCount > 0 ? remaining / otherCount : 0;
      }
    }
  } else {
    // Scale proportionally
    const scale = remaining / othersSum;
    for (let i = 0; i < next.length; i++) {
      if (i !== changedIdx) {
        next[i] = prev[i] * scale;
      }
    }
  }

  return next;
}

export default function TotalProbabilityExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [partitionSize, setPartitionSize] = useState<number>(3);
  const [pB, setPB] = useState<number[]>(defaultPB(3));
  const [pAgivenB, setPAgivenB] = useState<number[]>(defaultPAgivenB(3));

  // Compute each term and total
  const terms = useMemo(() => pB.map((pb, i) => pAgivenB[i] * pb), [pB, pAgivenB]);
  const pA = useMemo(() => totalProbability(pAgivenB, pB), [pAgivenB, pB]);

  // Layout: stack panels vertically on narrow screens
  const isNarrow = width > 0 && width < 500;
  const svgPadding = 16;
  const panelGap = isNarrow ? 12 : 24;

  // Panel dimensions
  const panelWidth = isNarrow
    ? width - svgPadding * 2
    : (width - svgPadding * 2 - panelGap) / 2;
  const panelHeight = Math.min(panelWidth * 0.7, 220);
  const svgHeight = isNarrow
    ? panelHeight * 2 + panelGap + svgPadding * 2
    : panelHeight + svgPadding * 2;

  // Panel origins
  const p1x = svgPadding;
  const p1y = svgPadding;
  const p2x = isNarrow ? svgPadding : svgPadding + panelWidth + panelGap;
  const p2y = isNarrow ? svgPadding + panelHeight + panelGap : svgPadding;

  // Event A band: spans middle 40% of the panel height
  const aBandTop = panelHeight * 0.25;
  const aBandHeight = panelHeight * 0.5;

  function handlePartitionChange(size: number) {
    setPartitionSize(size);
    setPB(defaultPB(size));
    setPAgivenB(defaultPAgivenB(size));
  }

  function handlePBChange(idx: number, val: number) {
    setPB((prev) => adjustPartitionProbs(prev, idx, val));
  }

  function handlePAgivenBChange(idx: number, val: number) {
    setPAgivenB((prev) => {
      const next = [...prev];
      next[idx] = Math.max(0, Math.min(1, val));
      return next;
    });
  }

  // Stacked bar chart: max bar height is panelHeight minus labels
  const barLabelSpace = 28;
  const barMaxHeight = panelHeight - barLabelSpace * 2;
  const barWidth = Math.min(panelWidth * 0.35, 80);
  const barX = (panelWidth - barWidth) / 2;

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Partition size toggles */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
          Partition size:
        </span>
        <div className="flex gap-1.5">
          {PARTITION_SIZES.map((size) => (
            <button
              key={size}
              onClick={() => handlePartitionChange(size)}
              className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
              style={{
                background: partitionSize === size ? '#2563eb' : 'var(--color-surface)',
                color: partitionSize === size ? '#ffffff' : 'var(--color-text)',
                borderColor: partitionSize === size ? '#2563eb' : 'var(--color-border)',
              }}
            >
              {size}
            </button>
          ))}
        </div>
      </div>

      {/* SVG panels */}
      <div ref={containerRef} className="w-full">
        {width > 0 && (
          <svg
            width={width}
            height={svgHeight}
            viewBox={`0 0 ${width} ${svgHeight}`}
            style={{ overflow: 'visible' }}
          >
            {/* ── Panel 1: Partition diagram ── */}
            <g transform={`translate(${p1x}, ${p1y})`}>
              {/* Omega rectangle */}
              <rect
                x={0}
                y={0}
                width={panelWidth}
                height={panelHeight}
                rx={6}
                fill="var(--color-surface)"
                stroke="var(--color-border-strong)"
                strokeWidth={1.5}
              />
              <text
                x={8}
                y={16}
                fontSize={13}
                fill="var(--color-text-muted)"
                fontFamily="var(--font-serif)"
              >
                {'\u03A9'}
              </text>

              {/* Partition regions (vertical strips) */}
              {pB.map((pb, i) => {
                const xStart = pB.slice(0, i).reduce((s, v) => s + v, 0) * panelWidth;
                const regionWidth = pb * panelWidth;
                const color = PALETTE[i];

                return (
                  <g key={i}>
                    {/* Full Bi region */}
                    <rect
                      x={xStart}
                      y={0}
                      width={regionWidth}
                      height={panelHeight}
                      fill={color.base}
                      fillOpacity={0.12}
                      stroke={color.base}
                      strokeWidth={1}
                      strokeOpacity={0.4}
                    />
                    {/* A intersection Bi (darker band) */}
                    <rect
                      x={xStart}
                      y={aBandTop}
                      width={regionWidth}
                      height={aBandHeight * pAgivenB[i]}
                      fill={color.dark}
                      fillOpacity={0.4}
                    />
                    {/* Bi label */}
                    <text
                      x={xStart + regionWidth / 2}
                      y={panelHeight - 6}
                      textAnchor="middle"
                      fontSize={regionWidth > 40 ? 12 : 9}
                      fontWeight={600}
                      fill={color.base}
                    >
                      B{String.fromCharCode(0x2081 + i)}
                    </text>
                    {/* A∩Bi label inside the intersection */}
                    {regionWidth > 30 && pAgivenB[i] > 0.1 && (
                      <text
                        x={xStart + regionWidth / 2}
                        y={aBandTop + (aBandHeight * pAgivenB[i]) / 2 + 4}
                        textAnchor="middle"
                        fontSize={regionWidth > 50 ? 10 : 8}
                        fill={color.dark}
                        fontWeight={600}
                      >
                        A{'\u2229'}B{String.fromCharCode(0x2081 + i)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Event A band outline */}
              <rect
                x={0}
                y={aBandTop}
                width={panelWidth}
                height={aBandHeight}
                fill="none"
                stroke="var(--color-text-heading)"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                rx={2}
              />
              <text
                x={panelWidth - 6}
                y={aBandTop - 4}
                textAnchor="end"
                fontSize={13}
                fontWeight={700}
                fill="var(--color-text-heading)"
              >
                A
              </text>
            </g>

            {/* ── Panel 2: Stacked bar chart ── */}
            <g transform={`translate(${p2x}, ${p2y})`}>
              {/* Panel background */}
              <rect
                x={0}
                y={0}
                width={panelWidth}
                height={panelHeight}
                rx={6}
                fill="var(--color-surface)"
                stroke="var(--color-border-strong)"
                strokeWidth={1.5}
              />

              {/* Stacked bar */}
              {(() => {
                const barBaseY = panelHeight - barLabelSpace;
                let cumulativeHeight = 0;

                return terms.map((term, i) => {
                  const segmentHeight = term * barMaxHeight;
                  const segmentY = barBaseY - cumulativeHeight - segmentHeight;
                  cumulativeHeight += segmentHeight;
                  const color = PALETTE[i];

                  return (
                    <g key={i}>
                      <rect
                        x={barX}
                        y={segmentY}
                        width={barWidth}
                        height={Math.max(segmentHeight, 0)}
                        fill={color.base}
                        fillOpacity={0.6}
                        stroke={color.base}
                        strokeWidth={1}
                      />
                      {/* Segment label */}
                      {segmentHeight > 14 && (
                        <text
                          x={barX + barWidth + 6}
                          y={segmentY + segmentHeight / 2 + 4}
                          fontSize={10}
                          fill={color.base}
                          fontWeight={600}
                        >
                          {term.toFixed(3)}
                        </text>
                      )}
                    </g>
                  );
                });
              })()}

              {/* P(A) total label on top */}
              {(() => {
                const barBaseY = panelHeight - barLabelSpace;
                const totalBarHeight = pA * barMaxHeight;
                return (
                  <text
                    x={barX + barWidth / 2}
                    y={barBaseY - totalBarHeight - 6}
                    textAnchor="middle"
                    fontSize={12}
                    fontWeight={700}
                    fill="var(--color-text-heading)"
                  >
                    P(A) = {pA.toFixed(4)}
                  </text>
                );
              })()}

              {/* Axis label */}
              <text
                x={barX + barWidth / 2}
                y={panelHeight - 6}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text-muted)"
              >
                {'\u03A3'} P(A|B{'\u1d62'}) {'\u00b7'} P(B{'\u1d62'})
              </text>
            </g>
          </svg>
        )}
      </div>

      {/* Sliders */}
      <div className="mt-4 space-y-3">
        {/* P(Bi) sliders */}
        <div>
          <div className="text-xs font-medium mb-1.5" style={{ color: 'var(--color-text-muted)' }}>
            Partition probabilities P(B{'\u1d62'})
          </div>
          <div className={'grid gap-3 grid-cols-1 ' + (partitionSize === 2 ? 'sm:grid-cols-2' : partitionSize === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-4')}>
            {pB.map((val, i) => (
              <label key={`pb-${i}`} className="text-xs" style={{ color: PALETTE[i].base }}>
                P(B{String.fromCharCode(0x2081 + i)}) = {val.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={val}
                  onChange={(e) => handlePBChange(i, +e.target.value)}
                  className="w-full"
                />
              </label>
            ))}
          </div>
        </div>

        {/* P(A|Bi) sliders */}
        <div>
          <div className="text-xs font-medium mb-1.5" style={{ color: 'var(--color-text-muted)' }}>
            Conditional probabilities P(A|B{'\u1d62'})
          </div>
          <div className={'grid gap-3 grid-cols-1 ' + (partitionSize === 2 ? 'sm:grid-cols-2' : partitionSize === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-4')}>
            {pAgivenB.map((val, i) => (
              <label key={`pagb-${i}`} className="text-xs" style={{ color: PALETTE[i].base }}>
                P(A|B{String.fromCharCode(0x2081 + i)}) = {val.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={val}
                  onChange={(e) => handlePAgivenBChange(i, +e.target.value)}
                  className="w-full"
                />
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Live readout */}
      <div
        className="mt-4 rounded-md p-3 text-xs sm:text-sm font-mono space-y-1"
        style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
      >
        {terms.map((term, i) => (
          <div key={i} style={{ color: PALETTE[i].base }}>
            P(A|B{String.fromCharCode(0x2081 + i)}) {'\u00b7'} P(B{String.fromCharCode(0x2081 + i)}) = {pAgivenB[i].toFixed(2)} {'\u00b7'} {pB[i].toFixed(2)} = {term.toFixed(4)}
          </div>
        ))}
        <div className="pt-1 border-t font-semibold" style={{ color: 'var(--color-text-heading)', borderColor: 'var(--color-border)' }}>
          P(A) = {'\u03A3'} P(A|B{'\u1d62'}) {'\u00b7'} P(B{'\u1d62'}) = {pA.toFixed(4)}
        </div>
      </div>
    </div>
  );
}
