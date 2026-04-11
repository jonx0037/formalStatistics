import { useState, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

type Operation = 'union' | 'intersection' | 'compA' | 'compB' | 'diffAB' | 'compUnion' | 'compIntersection';

const OPERATIONS: { key: Operation; label: string }[] = [
  { key: 'union', label: 'A ∪ B' },
  { key: 'intersection', label: 'A ∩ B' },
  { key: 'compA', label: 'Aᶜ' },
  { key: 'compB', label: 'Bᶜ' },
  { key: 'diffAB', label: 'A \\ B' },
  { key: 'compUnion', label: '(A∪B)ᶜ' },
  { key: 'compIntersection', label: '(A∩B)ᶜ' },
];

export default function EventSetOperationsExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [op, setOp] = useState<Operation>('union');
  const [pA, setPA] = useState(0.5);
  const [pB, setPB] = useState(0.4);
  const [pAB, setPAB] = useState(0.15);
  const svgRef = useRef<SVGSVGElement>(null);

  const maxPAB = Math.min(pA, pB);
  const clampedPAB = Math.min(pAB, maxPAB);

  // Compute P for current operation
  const pResult = (() => {
    switch (op) {
      case 'union': return pA + pB - clampedPAB;
      case 'intersection': return clampedPAB;
      case 'compA': return 1 - pA;
      case 'compB': return 1 - pB;
      case 'diffAB': return pA - clampedPAB;
      case 'compUnion': return 1 - (pA + pB - clampedPAB);
      case 'compIntersection': return 1 - clampedPAB;
    }
  })();

  const h = Math.min(width * 0.55, 280);
  const cx1 = width * 0.38;
  const cx2 = width * 0.62;
  const cy = h * 0.5;
  const r = Math.min(width * 0.22, 100);

  // Region fill logic
  const regionFill = (region: 'onlyA' | 'onlyB' | 'both' | 'outside') => {
    const fillColor = '#2563eb';
    const fillOpacity = 0.25;
    const noFill = 'transparent';

    switch (op) {
      case 'union':
        return region !== 'outside' ? { fill: fillColor, opacity: fillOpacity } : { fill: noFill, opacity: 0 };
      case 'intersection':
        return region === 'both' ? { fill: fillColor, opacity: fillOpacity } : { fill: noFill, opacity: 0 };
      case 'compA':
        return region === 'onlyB' || region === 'outside' ? { fill: '#dc2626', opacity: 0.2 } : { fill: noFill, opacity: 0 };
      case 'compB':
        return region === 'onlyA' || region === 'outside' ? { fill: '#dc2626', opacity: 0.2 } : { fill: noFill, opacity: 0 };
      case 'diffAB':
        return region === 'onlyA' ? { fill: fillColor, opacity: fillOpacity } : { fill: noFill, opacity: 0 };
      case 'compUnion':
        return region === 'outside' ? { fill: '#dc2626', opacity: 0.2 } : { fill: noFill, opacity: 0 };
      case 'compIntersection':
        return region !== 'both' ? { fill: '#dc2626', opacity: 0.2 } : { fill: noFill, opacity: 0 };
    }
  };

  const onlyAFill = regionFill('onlyA');
  const onlyBFill = regionFill('onlyB');
  const bothFill = regionFill('both');
  const outsideFill = regionFill('outside');

  // De Morgan annotation
  const isDeMorgan = op === 'compUnion' || op === 'compIntersection';
  const deMorganNote = op === 'compUnion'
    ? '(A∪B)ᶜ = Aᶜ ∩ Bᶜ'
    : op === 'compIntersection'
      ? '(A∩B)ᶜ = Aᶜ ∪ Bᶜ'
      : '';

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Operation toggles */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {OPERATIONS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setOp(key)}
            className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
            style={{
              background: op === key ? '#2563eb' : 'var(--color-surface)',
              color: op === key ? '#ffffff' : 'var(--color-text)',
              borderColor: op === key ? '#2563eb' : 'var(--color-border)',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* SVG Venn diagram */}
      <div ref={containerRef} className="w-full">
        <svg
          ref={svgRef}
          width={width}
          height={h}
          viewBox={`0 0 ${width} ${h}`}
          style={{ overflow: 'visible' }}
        >
          {/* Omega rectangle with possible fill */}
          <rect
            x={4} y={4}
            width={width - 8} height={h - 8}
            rx={8}
            fill={outsideFill.fill}
            fillOpacity={outsideFill.opacity}
            stroke="var(--color-border-strong)"
            strokeWidth={1.5}
          />
          <text x={16} y={24} fontSize={14} fill="var(--color-text-muted)" fontFamily="var(--font-serif)">Ω</text>

          {/* Clip paths for region isolation */}
          <defs>
            <clipPath id="clipA">
              <circle cx={cx1} cy={cy} r={r} />
            </clipPath>
            <clipPath id="clipB">
              <circle cx={cx2} cy={cy} r={r} />
            </clipPath>
            <clipPath id="clipNotA">
              <rect x={0} y={0} width={width} height={h} />
            </clipPath>
          </defs>

          {/* Only A (A \ B) */}
          <g clipPath="url(#clipA)">
            <rect x={0} y={0} width={width} height={h} fill={onlyAFill.fill} fillOpacity={onlyAFill.opacity} />
            {/* Remove B intersection */}
            <circle cx={cx2} cy={cy} r={r} fill={bothFill.fill} fillOpacity={bothFill.opacity} />
          </g>

          {/* Only B (B \ A) */}
          <g clipPath="url(#clipB)">
            <rect x={0} y={0} width={width} height={h} fill={onlyBFill.fill} fillOpacity={onlyBFill.opacity} />
            {/* Remove A intersection */}
            <circle cx={cx1} cy={cy} r={r} fill={bothFill.fill} fillOpacity={bothFill.opacity} />
          </g>

          {/* Intersection region - draw on top */}
          <g>
            <clipPath id="clipIntersection">
              <circle cx={cx1} cy={cy} r={r} />
            </clipPath>
            <circle
              cx={cx2} cy={cy} r={r}
              clipPath="url(#clipIntersection)"
              fill={bothFill.fill}
              fillOpacity={bothFill.opacity}
            />
          </g>

          {/* Circle outlines */}
          <circle cx={cx1} cy={cy} r={r} fill="none" stroke="#2563eb" strokeWidth={2} />
          <circle cx={cx2} cy={cy} r={r} fill="none" stroke="#7c3aed" strokeWidth={2} />

          {/* Labels */}
          <text x={cx1 - r * 0.5} y={cy} textAnchor="middle" fontSize={16} fontWeight={600} fill="#2563eb">A</text>
          <text x={cx2 + r * 0.5} y={cy} textAnchor="middle" fontSize={16} fontWeight={600} fill="#7c3aed">B</text>

          {isDeMorgan && (
            <text x={width / 2} y={h - 12} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--color-text-heading)">
              {deMorganNote}
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
          P(A∩B) = {clampedPAB.toFixed(2)} <span className="opacity-50">(max {maxPAB.toFixed(2)})</span>
          <input type="range" min={0} max={maxPAB} step={0.01} value={clampedPAB} onChange={(e) => setPAB(+e.target.value)} className="w-full" />
        </label>
      </div>

      {/* Result readout */}
      <div className="mt-3 text-sm font-semibold" style={{ color: 'var(--color-text-heading)' }}>
        P({OPERATIONS.find((o) => o.key === op)?.label}) = {pResult.toFixed(4)}
      </div>
    </div>
  );
}
