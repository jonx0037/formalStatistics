import { useState, useId } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { conditionalP } from './shared/probability';

export default function ConditionalProbabilityExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [pA, setPA] = useState(0.5);
  const [pB, setPB] = useState(0.4);
  const [pAB, setPAB] = useState(0.15);
  const [conditioning, setConditioning] = useState(false);

  const maxPAB = Math.min(pA, pB);
  const clampedPAB = Math.min(pAB, maxPAB);

  // Enforce valid Venn diagram: pA + pB - pAB <= 1
  const isValid = pA + pB - clampedPAB <= 1 + 1e-9;

  const pAgivenB = conditionalP(clampedPAB, pB);

  const h = Math.min(width * 0.55, 280);
  const cx1 = width * 0.38;
  const cx2 = width * 0.62;
  const cy = h * 0.5;
  const r = Math.min(width * 0.22, 100);

  // Per-instance clip-path IDs to avoid collisions when multiple instances render
  const instanceId = useId();
  const clipAId = `condClipA-${instanceId}`;
  const clipBId = `condClipB-${instanceId}`;
  const clipIntersectionId = `condClipIntersection-${instanceId}`;

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Conditioning toggle */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        <button
          onClick={() => setConditioning(false)}
          className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
          style={{
            background: !conditioning ? '#2563eb' : 'var(--color-surface)',
            color: !conditioning ? '#ffffff' : 'var(--color-text)',
            borderColor: !conditioning ? '#2563eb' : 'var(--color-border)',
          }}
        >
          Show Full Space
        </button>
        <button
          onClick={() => setConditioning(true)}
          className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
          style={{
            background: conditioning ? '#2563eb' : 'var(--color-surface)',
            color: conditioning ? '#ffffff' : 'var(--color-text)',
            borderColor: conditioning ? '#2563eb' : 'var(--color-border)',
          }}
        >
          Condition on B
        </button>
      </div>

      {/* SVG Venn diagram */}
      <div ref={containerRef} className="w-full">
        <svg
          width={width}
          height={h}
          viewBox={`0 0 ${width} ${h}`}
          style={{ overflow: 'visible' }}
        >
          <defs>
            <clipPath id={clipAId}>
              <circle cx={cx1} cy={cy} r={r} />
            </clipPath>
            <clipPath id={clipBId}>
              <circle cx={cx2} cy={cy} r={r} />
            </clipPath>
            <clipPath id={clipIntersectionId}>
              <circle cx={cx1} cy={cy} r={r} />
            </clipPath>
          </defs>

          {/* Omega rectangle */}
          <rect
            x={4} y={4}
            width={width - 8} height={h - 8}
            rx={8}
            fill="transparent"
            fillOpacity={0}
            stroke="var(--color-border-strong)"
            strokeWidth={1.5}
            style={{
              opacity: conditioning ? 0.1 : 1,
              transition: 'opacity 0.3s ease',
            }}
          />

          {/* Omega label */}
          <text
            x={16} y={24}
            fontSize={14}
            fill="var(--color-text-muted)"
            fontFamily="var(--font-serif)"
            style={{
              opacity: conditioning ? 0.1 : 1,
              transition: 'opacity 0.3s ease',
            }}
          >
            {'\u03A9'}
          </text>

          {/* Only A region (A \ B) */}
          <g clipPath={`url(#${clipAId})`}>
            <rect
              x={0} y={0} width={width} height={h}
              fill="#2563eb"
              fillOpacity={conditioning ? 0.03 : 0.15}
              style={{ transition: 'fill-opacity 0.3s ease' }}
            />
            {/* Carve out B intersection so it can be drawn separately */}
            <circle
              cx={cx2} cy={cy} r={r}
              fill={conditioning ? '#f59e0b' : '#2563eb'}
              fillOpacity={conditioning ? 0.35 : 0.25}
              style={{ transition: 'fill-opacity 0.3s ease, fill 0.3s ease' }}
            />
          </g>

          {/* Only B region (B \ A) */}
          <g clipPath={`url(#${clipBId})`}>
            <rect
              x={0} y={0} width={width} height={h}
              fill={conditioning ? '#f59e0b' : '#7c3aed'}
              fillOpacity={conditioning ? 0.15 : 0.1}
              style={{ transition: 'fill-opacity 0.3s ease, fill 0.3s ease' }}
            />
            {/* Carve out A intersection */}
            <circle
              cx={cx1} cy={cy} r={r}
              fill={conditioning ? '#f59e0b' : '#2563eb'}
              fillOpacity={conditioning ? 0.35 : 0.25}
              style={{ transition: 'fill-opacity 0.3s ease, fill 0.3s ease' }}
            />
          </g>

          {/* Intersection region A∩B — drawn on top */}
          <circle
            cx={cx2} cy={cy} r={r}
            clipPath={`url(#${clipIntersectionId})`}
            fill={conditioning ? '#f59e0b' : '#2563eb'}
            fillOpacity={conditioning ? 0.35 : 0.25}
            style={{ transition: 'fill-opacity 0.3s ease, fill 0.3s ease' }}
          />

          {/* Circle outlines */}
          <circle
            cx={cx1} cy={cy} r={r}
            fill="none"
            stroke="#2563eb"
            strokeWidth={2}
            style={{
              opacity: conditioning ? 0.2 : 1,
              transition: 'opacity 0.3s ease',
            }}
          />
          <circle
            cx={cx2} cy={cy} r={r}
            fill="none"
            stroke={conditioning ? '#f59e0b' : '#7c3aed'}
            strokeWidth={conditioning ? 2.5 : 2}
            style={{ transition: 'stroke 0.3s ease, stroke-width 0.3s ease' }}
          />

          {/* Labels */}
          <text
            x={cx1 - r * 0.5} y={cy}
            textAnchor="middle"
            fontSize={16}
            fontWeight={600}
            fill="#2563eb"
            style={{
              opacity: conditioning ? 0.25 : 1,
              transition: 'opacity 0.3s ease',
            }}
          >
            A
          </text>
          <text
            x={cx2 + r * 0.5} y={cy}
            textAnchor="middle"
            fontSize={16}
            fontWeight={600}
            fill={conditioning ? '#f59e0b' : '#7c3aed'}
            style={{ transition: 'fill 0.3s ease' }}
          >
            B
          </text>

          {/* Conditioning annotation */}
          {conditioning && (
            <text
              x={width / 2} y={h - 12}
              textAnchor="middle"
              fontSize={13}
              fontWeight={600}
              fill="var(--color-text-heading)"
            >
              B is the new sample space
            </text>
          )}
        </svg>
      </div>

      {/* Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
        <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          P(A) = {pA.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={pA} onChange={(e) => setPA(+e.target.value)} className="w-full" />
        </label>
        <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          P(B) = {pB.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={pB} onChange={(e) => setPB(+e.target.value)} className="w-full" />
        </label>
        <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {`P(A\u2229B) = ${clampedPAB.toFixed(2)}`} <span className="opacity-50">(max {maxPAB.toFixed(2)})</span>
          <input type="range" min={0} max={maxPAB} step={0.01} value={clampedPAB} onChange={(e) => setPAB(+e.target.value)} className="w-full" />
        </label>
      </div>

      {/* Validity warning */}
      {!isValid && (
        <div className="mt-2 text-xs font-medium" style={{ color: '#dc2626' }}>
          Invalid: P(A) + P(B) - P(A{'\u2229'}B) {'>'} 1. Adjust sliders so probabilities fit within {'\u03A9'}.
        </div>
      )}

      {/* Readout panel */}
      <div
        className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-sm"
        style={{ color: 'var(--color-text-heading)' }}
      >
        <div>P(A) = {pA.toFixed(4)}</div>
        <div>P(B) = {pB.toFixed(4)}</div>
        <div>P(A{'\u2229'}B) = {clampedPAB.toFixed(4)}</div>
        <div className="font-semibold">
          {pB === 0 ? (
            <span style={{ color: '#dc2626' }}>P(B) = 0 — conditioning undefined</span>
          ) : (
            <>P(A|B) = {clampedPAB.toFixed(2)}/{pB.toFixed(2)} = {pAgivenB.toFixed(4)}</>
          )}
        </div>
      </div>
    </div>
  );
}
