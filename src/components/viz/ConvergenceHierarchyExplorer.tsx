import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { convergenceModes, hierarchyArrows } from '../../data/modes-of-convergence-data';
import type { ConvergenceMode, HierarchyArrow } from '../../data/modes-of-convergence-data';
import katex from 'katex';

// ── Colors ────────────────────────────────────────────────────────────────────

const STRENGTH_COLORS: Record<number, string> = {
  1: '#7C3AED', // violet — strongest (a.s. & Lp)
  2: '#2563EB', // blue — middle (in probability)
  3: '#059669', // emerald — weakest (in distribution)
};

const ARROW_COLORS = {
  implication: '#16a34a',
  'non-implication': '#dc2626',
  partial: '#d97706',
} as const;

// ── KaTeX helper ──────────────────────────────────────────────────────────────
// All LaTeX originates from our own static data module (modes-of-convergence-data.ts),
// not from user input. The dangerouslySetInnerHTML usage is safe in this context.

function renderKatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      trust: true,
    });
  } catch {
    return latex;
  }
}

// ── Reachability via BFS ──────────────────────────────────────────────────────

function reachableFrom(startId: string): Set<string> {
  const visited = new Set<string>();
  const queue = [startId];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);
    for (const arrow of hierarchyArrows) {
      if (arrow.from === current && arrow.type === 'implication') {
        queue.push(arrow.to);
      }
    }
  }
  return visited;
}

// ── Node positions (diamond layout) ───────────────────────────────────────────

interface NodePos { cx: number; cy: number }

function nodePositions(w: number, h: number, narrow: boolean): Record<string, NodePos> {
  if (narrow) {
    // Vertical stack for narrow screens
    const cx = w / 2;
    const gap = h / 5;
    return {
      as:   { cx: cx - w * 0.2, cy: gap },
      lp:   { cx: cx + w * 0.2, cy: gap },
      prob: { cx, cy: gap * 2.5 },
      dist: { cx, cy: gap * 4 },
    };
  }
  // Diamond: a.s. top-left, Lp top-right, prob center, dist bottom
  return {
    as:   { cx: w * 0.22, cy: h * 0.22 },
    lp:   { cx: w * 0.78, cy: h * 0.22 },
    prob: { cx: w * 0.50, cy: h * 0.55 },
    dist: { cx: w * 0.50, cy: h * 0.88 },
  };
}

// ── Arrow path with offset from node edges ────────────────────────────────────

const NODE_RX = 70;
const NODE_RY = 24;

function edgePoint(
  from: NodePos,
  to: NodePos,
  rx: number,
  ry: number,
  isStart: boolean,
): { x: number; y: number } {
  const anchor = isStart ? from : to;
  const other = isStart ? to : from;
  const dx = other.cx - anchor.cx;
  const dy = other.cy - anchor.cy;
  const angle = Math.atan2(dy, dx);
  // Ellipse parametric edge
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const denom = Math.sqrt((ry * cos) ** 2 + (rx * sin) ** 2);
  const ex = (rx * ry * cos) / denom;
  const ey = (rx * ry * sin) / denom;
  return { x: anchor.cx + ex, y: anchor.cy + ey };
}

// ── Selection types ───────────────────────────────────────────────────────────

type Selection =
  | { kind: 'node'; mode: ConvergenceMode }
  | { kind: 'arrow'; arrow: HierarchyArrow };

// ── Component ─────────────────────────────────────────────────────────────────

export default function ConvergenceHierarchyExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const [selection, setSelection] = useState<Selection | null>(null);
  const [highlightMode, setHighlightMode] = useState<string>('all');

  // Responsive layout
  const containerW = Math.max(280, (width || 600) - 16);
  const isNarrow = containerW < 624;
  const svgW = containerW;
  const svgH = isNarrow ? 320 : 280;

  const positions = useMemo(
    () => nodePositions(svgW, svgH, isNarrow),
    [svgW, svgH, isNarrow],
  );

  // Reachability set for highlight mode
  const reachable = useMemo(() => {
    if (highlightMode === 'all') return null;
    return reachableFrom(highlightMode);
  }, [highlightMode]);

  // Node opacity based on highlight
  const nodeOpacity = useCallback(
    (id: string) => {
      if (!reachable) return 1;
      return reachable.has(id) ? 1 : 0.2;
    },
    [reachable],
  );

  // Arrow opacity based on highlight
  const arrowOpacity = useCallback(
    (arrow: HierarchyArrow) => {
      if (!reachable) return 1;
      if (arrow.type !== 'implication') return 0.15;
      return reachable.has(arrow.from) && reachable.has(arrow.to) ? 1 : 0.15;
    },
    [reachable],
  );

  // Click handlers
  const handleNodeClick = useCallback(
    (mode: ConvergenceMode) => {
      setSelection((prev) =>
        prev?.kind === 'node' && prev.mode.id === mode.id ? null : { kind: 'node', mode },
      );
    },
    [],
  );

  const handleArrowClick = useCallback(
    (arrow: HierarchyArrow) => {
      setSelection((prev) =>
        prev?.kind === 'arrow' && prev.arrow.from === arrow.from && prev.arrow.to === arrow.to
          ? null
          : { kind: 'arrow', arrow },
      );
    },
    [],
  );

  // Check if a node is selected
  const isNodeSelected = useCallback(
    (id: string) => selection?.kind === 'node' && selection.mode.id === id,
    [selection],
  );

  // Check if an arrow is selected
  const isArrowSelected = useCallback(
    (a: HierarchyArrow) =>
      selection?.kind === 'arrow' && selection.arrow.from === a.from && selection.arrow.to === a.to,
    [selection],
  );

  // Mode lookup
  const modeById = useMemo(() => {
    const map: Record<string, ConvergenceMode> = {};
    for (const m of convergenceModes) map[m.id] = m;
    return map;
  }, []);

  // Marker IDs keyed by type
  const markerIds = useMemo(
    () => ({
      implication: 'arrow-impl',
      'non-implication': 'arrow-non',
      partial: 'arrow-part',
    }),
    [],
  );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{
        borderColor: 'var(--color-border)',
        background: 'var(--color-bg)',
        color: 'var(--color-text)',
      }}
    >
      {/* Header */}
      <div className={`flex ${isNarrow ? 'flex-col gap-2' : 'items-center justify-between'} mb-3`}>
        <h3 className="text-base font-semibold m-0" style={{ color: 'var(--color-text)' }}>
          Convergence Hierarchy
        </h3>
        <label className="flex items-center gap-2 text-sm">
          <span style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>Highlight:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={highlightMode}
            onChange={(e) => setHighlightMode(e.target.value)}
          >
            <option value="all">All</option>
            {convergenceModes.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* SVG Diagram */}
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="block mx-auto"
        style={{ maxWidth: '100%' }}
      >
        {/* Marker definitions */}
        <defs>
          {(['implication', 'non-implication', 'partial'] as const).map((type) => (
            <marker
              key={type}
              id={markerIds[type]}
              viewBox="0 0 10 6"
              refX="9"
              refY="3"
              markerWidth="8"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,3 L0,6 Z" fill={ARROW_COLORS[type]} />
            </marker>
          ))}
        </defs>

        {/* Arrows */}
        {hierarchyArrows.map((arrow) => {
          const fromPos = positions[arrow.from];
          const toPos = positions[arrow.to];
          if (!fromPos || !toPos) return null;

          const start = edgePoint(fromPos, toPos, NODE_RX, NODE_RY, true);
          const end = edgePoint(fromPos, toPos, NODE_RX, NODE_RY, false);

          // For non-implications between a.s. and Lp, curve the path to avoid overlap
          const isBidirectionalPair =
            (arrow.from === 'as' && arrow.to === 'lp') ||
            (arrow.from === 'lp' && arrow.to === 'as');
          // Also curve prob->as non-implication to separate from the partial arrow
          const isProbAsNon = arrow.from === 'prob' && arrow.to === 'as' && arrow.type === 'non-implication';

          let pathD: string;
          if (isBidirectionalPair || isProbAsNon) {
            const mx = (start.x + end.x) / 2;
            const my = (start.y + end.y) / 2;
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const offset = isBidirectionalPair ? 30 : 25;
            // Normal direction: perpendicular, pick consistent side
            const sign = (arrow.from === 'as' && arrow.to === 'lp') || isProbAsNon ? -1 : 1;
            const nx = (-dy / len) * offset * sign;
            const ny = (dx / len) * offset * sign;
            pathD = `M${start.x},${start.y} Q${mx + nx},${my + ny} ${end.x},${end.y}`;
          } else {
            pathD = `M${start.x},${start.y} L${end.x},${end.y}`;
          }

          const dashArray =
            arrow.type === 'non-implication' ? '6,4' : arrow.type === 'partial' ? '3,3' : 'none';

          const selected = isArrowSelected(arrow);
          const opacity = arrowOpacity(arrow);

          return (
            <g key={`${arrow.from}-${arrow.to}-${arrow.type}`}>
              {/* Invisible wider hit area */}
              <path
                d={pathD}
                fill="none"
                stroke="transparent"
                strokeWidth={16}
                style={{ cursor: 'pointer' }}
                onClick={() => handleArrowClick(arrow)}
              />
              <path
                d={pathD}
                fill="none"
                stroke={ARROW_COLORS[arrow.type]}
                strokeWidth={selected ? 3 : 2}
                strokeDasharray={dashArray}
                markerEnd={`url(#${markerIds[arrow.type]})`}
                opacity={opacity}
                style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                onClick={() => handleArrowClick(arrow)}
              />
              {/* Non-implication cross mark at midpoint */}
              {arrow.type === 'non-implication' && (() => {
                const mx = (start.x + end.x) / 2;
                const my = (start.y + end.y) / 2;
                // Shift cross along the curve if needed
                let cx = mx;
                let cy = my;
                if (isBidirectionalPair || isProbAsNon) {
                  const dx = end.x - start.x;
                  const dy = end.y - start.y;
                  const len = Math.sqrt(dx * dx + dy * dy) || 1;
                  const offset = isBidirectionalPair ? 30 : 25;
                  const sign = (arrow.from === 'as' && arrow.to === 'lp') || isProbAsNon ? -1 : 1;
                  cx = mx + (-dy / len) * offset * sign * 0.5;
                  cy = my + (dx / len) * offset * sign * 0.5;
                }
                return (
                  <text
                    x={cx}
                    y={cy}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="14"
                    fontWeight="bold"
                    fill={ARROW_COLORS['non-implication']}
                    opacity={opacity}
                    style={{ pointerEvents: 'none' }}
                  >
                    ✗
                  </text>
                );
              })()}
            </g>
          );
        })}

        {/* Nodes */}
        {convergenceModes.map((mode) => {
          const pos = positions[mode.id];
          if (!pos) return null;
          const selected = isNodeSelected(mode.id);
          const opacity = nodeOpacity(mode.id);
          const color = STRENGTH_COLORS[mode.strength];

          return (
            <g
              key={mode.id}
              style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
              opacity={opacity}
              onClick={() => handleNodeClick(mode)}
            >
              <rect
                x={pos.cx - NODE_RX}
                y={pos.cy - NODE_RY}
                width={NODE_RX * 2}
                height={NODE_RY * 2}
                rx={12}
                ry={12}
                fill={color}
                fillOpacity={selected ? 0.25 : 0.12}
                stroke={color}
                strokeWidth={selected ? 3 : 1.5}
              />
              <text
                x={pos.cx}
                y={pos.cy - 5}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={isNarrow ? '12' : '13'}
                fontWeight="600"
                fill={color}
                style={{ pointerEvents: 'none' }}
              >
                {mode.name}
              </text>
              <text
                x={pos.cx}
                y={pos.cy + 13}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="10"
                fill={color}
                opacity={0.7}
                style={{ pointerEvents: 'none' }}
              >
                ({mode.shortName})
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 justify-center mt-2 mb-1 text-xs" style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>
        <span className="flex items-center gap-1">
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke={ARROW_COLORS.implication} strokeWidth="2" /></svg>
          Implies
        </span>
        <span className="flex items-center gap-1">
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke={ARROW_COLORS['non-implication']} strokeWidth="2" strokeDasharray="6,4" /></svg>
          Does not imply
        </span>
        <span className="flex items-center gap-1">
          <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke={ARROW_COLORS.partial} strokeWidth="2" strokeDasharray="3,3" /></svg>
          Partial (subsequence)
        </span>
      </div>

      {/* Side Panel */}
      {selection && (
        <div
          className="mt-3 rounded-lg border p-4"
          style={{
            borderColor: 'var(--color-border)',
            background: 'var(--color-surface, var(--color-bg))',
          }}
        >
          {selection.kind === 'node' && <NodePanel mode={selection.mode} />}
          {selection.kind === 'arrow' && <ArrowPanel arrow={selection.arrow} modeById={modeById} />}
        </div>
      )}
    </div>
  );
}

// ── Node Panel ────────────────────────────────────────────────────────────────

function NodePanel({ mode }: { mode: ConvergenceMode }) {
  const color = STRENGTH_COLORS[mode.strength];
  // All LaTeX from our own static data module — safe for dangerouslySetInnerHTML
  const symbolHtml = useMemo(() => renderKatex(mode.symbol, true), [mode.symbol]);

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span
          className="inline-block w-3 h-3 rounded-full"
          style={{ background: color }}
        />
        <span className="font-semibold text-sm" style={{ color: 'var(--color-text)' }}>
          {mode.name} Convergence
        </span>
        <span
          className="text-xs px-2 py-0.5 rounded-full"
          style={{
            background: color,
            color: '#fff',
            opacity: 0.85,
          }}
        >
          strength {mode.strength}/3
        </span>
      </div>

      {/* LaTeX definition — safe: source is our own static data module */}
      <div
        className="my-2 overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: symbolHtml }}
      />

      <p className="text-sm mt-2 mb-1" style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>
        <strong>Intuition:</strong> {mode.intuition}
      </p>
      <p className="text-sm mt-1 mb-0" style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>
        <strong>ML use cases:</strong> {mode.mlUseCase}
      </p>
    </div>
  );
}

// ── Arrow Panel ───────────────────────────────────────────────────────────────

function ArrowPanel({
  arrow,
  modeById,
}: {
  arrow: HierarchyArrow;
  modeById: Record<string, ConvergenceMode>;
}) {
  const fromMode = modeById[arrow.from];
  const toMode = modeById[arrow.to];
  const color = ARROW_COLORS[arrow.type];

  const typeLabel =
    arrow.type === 'implication'
      ? 'Implication'
      : arrow.type === 'non-implication'
        ? 'Non-implication'
        : 'Partial implication';

  return (
    <div>
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="font-semibold text-sm" style={{ color: 'var(--color-text)' }}>
          {fromMode?.name ?? arrow.from}
        </span>
        <span style={{ color }} className="font-bold text-sm">
          {arrow.type === 'implication' ? '\u21D2' : arrow.type === 'non-implication' ? '\u21CF' : '\u21E2'}
        </span>
        <span className="font-semibold text-sm" style={{ color: 'var(--color-text)' }}>
          {toMode?.name ?? arrow.to}
        </span>
        <span
          className="text-xs px-2 py-0.5 rounded-full"
          style={{ background: color, color: '#fff' }}
        >
          {typeLabel}
        </span>
        {arrow.theoremNum != null && (
          <span
            className="text-xs px-2 py-0.5 rounded border"
            style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-secondary, var(--color-text))' }}
          >
            Theorem {arrow.theoremNum}
          </span>
        )}
      </div>

      {arrow.caveat && (
        <p className="text-xs italic mt-1 mb-2" style={{ color: ARROW_COLORS.partial }}>
          Caveat: {arrow.caveat}
        </p>
      )}

      {arrow.proof && (
        <div className="text-sm mt-1" style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>
          <strong>Proof sketch:</strong> {arrow.proof}
        </div>
      )}
      {arrow.counterexample && (
        <div className="text-sm mt-1" style={{ color: 'var(--color-text-secondary, var(--color-text))' }}>
          <strong>Counterexample:</strong> {arrow.counterexample}
        </div>
      )}
    </div>
  );
}
