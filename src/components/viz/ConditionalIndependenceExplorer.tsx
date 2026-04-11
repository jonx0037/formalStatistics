import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { areIndependent, areConditionallyIndependent } from './shared/probability';
import { conditionalIndependencePresets } from '../../data/conditional-probability-data';

/** Index mapping: 0→(0,0,0), 1→(0,0,1), 2→(0,1,0), 3→(0,1,1), 4→(1,0,0), 5→(1,0,1), 6→(1,1,0), 7→(1,1,1) */
const INDEX_LABELS: [number, number, number][] = [
  [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
];

const TOLERANCE = 1e-6;

function adjustProbs(
  current: number[],
  locked: boolean[],
  changedIndex: number,
  newValue: number,
): number[] {
  if (locked[changedIndex]) return current;

  const clamped = Math.max(0, Math.min(1, newValue));
  const delta = clamped - current[changedIndex];
  if (Math.abs(delta) < 1e-12) return current;

  // Find unlocked indices excluding the changed one
  const unlocked: number[] = [];
  for (let i = 0; i < 8; i++) {
    if (i !== changedIndex && !locked[i]) unlocked.push(i);
  }
  if (unlocked.length === 0) return current;

  const next = [...current];
  next[changedIndex] = clamped;

  // Distribute -delta proportionally among unlocked cells
  let remaining = -delta;
  const totalUnlocked = unlocked.reduce((sum, i) => sum + next[i], 0);

  if (totalUnlocked <= 1e-12) {
    // All unlocked cells are zero — distribute equally
    const share = remaining / unlocked.length;
    for (const i of unlocked) {
      next[i] = Math.max(0, next[i] + share);
    }
  } else {
    // Proportional distribution with clamping
    let toDistribute = remaining;
    let activeIndices = [...unlocked];
    let iterations = 0;

    while (Math.abs(toDistribute) > 1e-12 && activeIndices.length > 0 && iterations < 10) {
      iterations++;
      const activeTotal = activeIndices.reduce((sum, i) => sum + next[i], 0);
      const nextActive: number[] = [];
      let excess = 0;

      for (const i of activeIndices) {
        const proportion = activeTotal > 1e-12 ? next[i] / activeTotal : 1 / activeIndices.length;
        const adjustment = toDistribute * proportion;
        const adjusted = next[i] + adjustment;

        if (adjusted < 0) {
          excess += adjusted; // negative — needs redistribution
          next[i] = 0;
        } else {
          next[i] = adjusted;
          nextActive.push(i);
        }
      }

      toDistribute = excess;
      activeIndices = nextActive;
    }
  }

  // Normalize to ensure sum = 1 (correct floating-point drift)
  const sum = next.reduce((s, v) => s + v, 0);
  if (sum > 1e-12) {
    const scale = 1 / sum;
    for (let i = 0; i < 8; i++) {
      next[i] *= scale;
    }
  }

  return next;
}

export default function ConditionalIndependenceExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [jointProbs, setJointProbs] = useState<number[]>(
    conditionalIndependencePresets[0].jointProbs,
  );
  const [locked, setLocked] = useState<boolean[]>(Array(8).fill(false));

  // Derived computations
  const stats = useMemo(() => {
    // Marginals
    const pA1 = jointProbs[4] + jointProbs[5] + jointProbs[6] + jointProbs[7];
    const pB1 = jointProbs[2] + jointProbs[3] + jointProbs[6] + jointProbs[7];
    const pA1B1 = jointProbs[6] + jointProbs[7];

    const marginallyIndep = areIndependent(pA1, pB1, pA1B1, TOLERANCE);

    // Conditional given C=1
    const pC1 = jointProbs[1] + jointProbs[3] + jointProbs[5] + jointProbs[7];
    let pA1givenC1 = NaN;
    let pB1givenC1 = NaN;
    let pA1B1givenC1 = NaN;
    let condIndepC1 = false;

    if (pC1 > 1e-12) {
      pA1givenC1 = (jointProbs[5] + jointProbs[7]) / pC1;
      pB1givenC1 = (jointProbs[3] + jointProbs[7]) / pC1;
      pA1B1givenC1 = jointProbs[7] / pC1;
      condIndepC1 = areConditionallyIndependent(pA1B1givenC1, pA1givenC1, pB1givenC1, TOLERANCE);
    }

    // Conditional given C=0
    const pC0 = jointProbs[0] + jointProbs[2] + jointProbs[4] + jointProbs[6];
    let pA1givenC0 = NaN;
    let pB1givenC0 = NaN;
    let pA1B1givenC0 = NaN;
    let condIndepC0 = false;

    if (pC0 > 1e-12) {
      pA1givenC0 = (jointProbs[4] + jointProbs[6]) / pC0;
      pB1givenC0 = (jointProbs[2] + jointProbs[6]) / pC0;
      pA1B1givenC0 = jointProbs[6] / pC0;
      condIndepC0 = areConditionallyIndependent(pA1B1givenC0, pA1givenC0, pB1givenC0, TOLERANCE);
    }

    return {
      pA1, pB1, pA1B1, marginallyIndep,
      pC1, pA1givenC1, pB1givenC1, pA1B1givenC1, condIndepC1,
      pC0, pA1givenC0, pB1givenC0, pA1B1givenC0, condIndepC0,
    };
  }, [jointProbs]);

  const handleSliderChange = (index: number, value: number) => {
    setJointProbs((prev) => adjustProbs(prev, locked, index, value));
  };

  const toggleLock = (index: number) => {
    setLocked((prev) => {
      const next = [...prev];
      next[index] = !next[index];
      return next;
    });
  };

  const applyPreset = (preset: typeof conditionalIndependencePresets[number]) => {
    setJointProbs([...preset.jointProbs]);
    setLocked(Array(8).fill(false));
  };

  const sum = jointProbs.reduce((s, v) => s + v, 0);

  // Graphical model layout
  const graphW = Math.min(width, 200);
  const graphH = 120;
  const nodeR = 18;
  const nodeA = { x: graphW * 0.2, y: graphH * 0.75 };
  const nodeB = { x: graphW * 0.8, y: graphH * 0.75 };
  const nodeC = { x: graphW * 0.5, y: graphH * 0.22 };

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Preset buttons */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {conditionalIndependencePresets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => applyPreset(preset)}
            className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
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

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-6">
        {/* Left: Joint probability table */}
        <div>
          <div className="text-xs font-semibold mb-2" style={{ color: 'var(--color-text-heading)' }}>
            Joint Probability Table
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ color: 'var(--color-text)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                  <th className="py-1 px-1 text-left font-medium" style={{ color: 'var(--color-text-muted)' }}>A</th>
                  <th className="py-1 px-1 text-left font-medium" style={{ color: 'var(--color-text-muted)' }}>B</th>
                  <th className="py-1 px-1 text-left font-medium" style={{ color: 'var(--color-text-muted)' }}>C</th>
                  <th className="py-1 px-1 text-left font-medium" style={{ color: 'var(--color-text-muted)' }}>P(A,B,C)</th>
                  <th className="py-1 px-1 text-center font-medium" style={{ color: 'var(--color-text-muted)', width: '2rem' }}></th>
                  <th className="py-1 px-1 font-medium" style={{ color: 'var(--color-text-muted)', minWidth: '120px' }}></th>
                </tr>
              </thead>
              <tbody>
                {INDEX_LABELS.map(([a, b, c], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--color-border)' }}>
                    <td className="py-1 px-1 font-mono">{a}</td>
                    <td className="py-1 px-1 font-mono">{b}</td>
                    <td className="py-1 px-1 font-mono">{c}</td>
                    <td className="py-1 px-1 font-mono tabular-nums">{jointProbs[i].toFixed(3)}</td>
                    <td className="py-1 px-1 text-center">
                      <button
                        onClick={() => toggleLock(i)}
                        className="cursor-pointer text-sm leading-none"
                        style={{ opacity: locked[i] ? 1 : 0.35 }}
                        aria-label={locked[i] ? `Unlock cell ${i}` : `Lock cell ${i}`}
                      >
                        {locked[i] ? '\uD83D\uDD12' : '\uD83D\uDD13'}
                      </button>
                    </td>
                    <td className="py-1 px-1">
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.001}
                        value={jointProbs[i]}
                        onChange={(e) => handleSliderChange(i, +e.target.value)}
                        disabled={locked[i]}
                        className="w-full"
                        style={{ opacity: locked[i] ? 0.4 : 1 }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={3} className="py-1 px-1 font-semibold" style={{ color: 'var(--color-text-heading)' }}>
                    Sum
                  </td>
                  <td className="py-1 px-1 font-mono font-semibold tabular-nums" style={{ color: Math.abs(sum - 1) < 0.001 ? 'var(--color-text-heading)' : '#dc2626' }}>
                    {sum.toFixed(3)}
                  </td>
                  <td colSpan={2}></td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

        {/* Right: Graphical model diagram */}
        <div className="flex flex-col items-center">
          <div className="text-xs font-semibold mb-2" style={{ color: 'var(--color-text-heading)' }}>
            Dependency Graph
          </div>
          <svg
            width={graphW}
            height={graphH}
            viewBox={`0 0 ${graphW} ${graphH}`}
            style={{ overflow: 'visible' }}
          >
            {/* Edge: A — B (if marginally dependent) */}
            {!stats.marginallyIndep && (
              <line
                x1={nodeA.x} y1={nodeA.y}
                x2={nodeB.x} y2={nodeB.y}
                stroke="#dc2626"
                strokeWidth={2}
                strokeDasharray="6,3"
                style={{ transition: 'opacity 0.3s ease' }}
              />
            )}

            {/* Edges: A — C and B — C (if conditionally dependent given C) */}
            {(!stats.condIndepC0 || !stats.condIndepC1) && (
              <>
                <line
                  x1={nodeA.x} y1={nodeA.y}
                  x2={nodeC.x} y2={nodeC.y}
                  stroke="#2563eb"
                  strokeWidth={2}
                  style={{ transition: 'opacity 0.3s ease' }}
                />
                <line
                  x1={nodeB.x} y1={nodeB.y}
                  x2={nodeC.x} y2={nodeC.y}
                  stroke="#2563eb"
                  strokeWidth={2}
                  style={{ transition: 'opacity 0.3s ease' }}
                />
              </>
            )}

            {/* Node A */}
            <circle cx={nodeA.x} cy={nodeA.y} r={nodeR} fill="#2563eb" fillOpacity={0.15} stroke="#2563eb" strokeWidth={2} />
            <text x={nodeA.x} y={nodeA.y + 1} textAnchor="middle" dominantBaseline="central" fontSize={14} fontWeight={600} fill="#2563eb">A</text>

            {/* Node B */}
            <circle cx={nodeB.x} cy={nodeB.y} r={nodeR} fill="#7c3aed" fillOpacity={0.15} stroke="#7c3aed" strokeWidth={2} />
            <text x={nodeB.x} y={nodeB.y + 1} textAnchor="middle" dominantBaseline="central" fontSize={14} fontWeight={600} fill="#7c3aed">B</text>

            {/* Node C */}
            <circle cx={nodeC.x} cy={nodeC.y} r={nodeR} fill="#059669" fillOpacity={0.15} stroke="#059669" strokeWidth={2} />
            <text x={nodeC.x} y={nodeC.y + 1} textAnchor="middle" dominantBaseline="central" fontSize={14} fontWeight={600} fill="#059669">C</text>
          </svg>

          {/* Legend */}
          <div className="mt-2 text-xs space-y-1" style={{ color: 'var(--color-text-muted)' }}>
            <div className="flex items-center gap-1.5">
              <svg width={20} height={8}><line x1={0} y1={4} x2={20} y2={4} stroke="#dc2626" strokeWidth={2} strokeDasharray="4,2" /></svg>
              <span>Marginal dep.</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg width={20} height={8}><line x1={0} y1={4} x2={20} y2={4} stroke="#2563eb" strokeWidth={2} /></svg>
              <span>Conditional dep.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Independence results */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Section 1: Marginal independence */}
        <div
          className="rounded-md border p-3"
          style={{
            borderColor: stats.marginallyIndep ? '#059669' : '#dc2626',
            background: stats.marginallyIndep ? 'rgba(5,150,105,0.05)' : 'rgba(220,38,38,0.05)',
          }}
        >
          <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--color-text-heading)' }}>
            Marginal Independence
          </div>
          <div className="text-xs space-y-0.5" style={{ color: 'var(--color-text-muted)' }}>
            <div>P(A=1) = {stats.pA1.toFixed(4)}</div>
            <div>P(B=1) = {stats.pB1.toFixed(4)}</div>
            <div>{`P(A=1\u2229B=1)`} = {stats.pA1B1.toFixed(4)}</div>
            <div>P(A=1){'\u00B7'}P(B=1) = {(stats.pA1 * stats.pB1).toFixed(4)}</div>
          </div>
          <div
            className="mt-2 text-xs font-semibold"
            style={{ color: stats.marginallyIndep ? '#059669' : '#dc2626' }}
          >
            {stats.marginallyIndep
              ? 'Marginally Independent \u2713'
              : 'Marginally Dependent \u2717'}
          </div>
        </div>

        {/* Section 2: Conditional independence given C=1 */}
        <div
          className="rounded-md border p-3"
          style={{
            borderColor: stats.pC1 > 1e-12
              ? (stats.condIndepC1 ? '#059669' : '#dc2626')
              : 'var(--color-border)',
            background: stats.pC1 > 1e-12
              ? (stats.condIndepC1 ? 'rgba(5,150,105,0.05)' : 'rgba(220,38,38,0.05)')
              : 'transparent',
          }}
        >
          <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--color-text-heading)' }}>
            Conditional Independence | C=1
          </div>
          {stats.pC1 > 1e-12 ? (
            <>
              <div className="text-xs space-y-0.5" style={{ color: 'var(--color-text-muted)' }}>
                <div>P(A=1|C=1) = {stats.pA1givenC1.toFixed(4)}</div>
                <div>P(B=1|C=1) = {stats.pB1givenC1.toFixed(4)}</div>
                <div>{`P(A=1\u2229B=1|C=1)`} = {stats.pA1B1givenC1.toFixed(4)}</div>
                <div>P(A=1|C=1){'\u00B7'}P(B=1|C=1) = {(stats.pA1givenC1 * stats.pB1givenC1).toFixed(4)}</div>
              </div>
              <div
                className="mt-2 text-xs font-semibold"
                style={{ color: stats.condIndepC1 ? '#059669' : '#dc2626' }}
              >
                {stats.condIndepC1
                  ? 'Cond. Independent given C=1 \u2713'
                  : 'Cond. Dependent given C=1 \u2717'}
              </div>
            </>
          ) : (
            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              P(C=1) = 0 — conditioning undefined
            </div>
          )}
        </div>

        {/* Section 3: Conditional independence given C=0 */}
        <div
          className="rounded-md border p-3"
          style={{
            borderColor: stats.pC0 > 1e-12
              ? (stats.condIndepC0 ? '#059669' : '#dc2626')
              : 'var(--color-border)',
            background: stats.pC0 > 1e-12
              ? (stats.condIndepC0 ? 'rgba(5,150,105,0.05)' : 'rgba(220,38,38,0.05)')
              : 'transparent',
          }}
        >
          <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--color-text-heading)' }}>
            Conditional Independence | C=0
          </div>
          {stats.pC0 > 1e-12 ? (
            <>
              <div className="text-xs space-y-0.5" style={{ color: 'var(--color-text-muted)' }}>
                <div>P(A=1|C=0) = {stats.pA1givenC0.toFixed(4)}</div>
                <div>P(B=1|C=0) = {stats.pB1givenC0.toFixed(4)}</div>
                <div>{`P(A=1\u2229B=1|C=0)`} = {stats.pA1B1givenC0.toFixed(4)}</div>
                <div>P(A=1|C=0){'\u00B7'}P(B=1|C=0) = {(stats.pA1givenC0 * stats.pB1givenC0).toFixed(4)}</div>
              </div>
              <div
                className="mt-2 text-xs font-semibold"
                style={{ color: stats.condIndepC0 ? '#059669' : '#dc2626' }}
              >
                {stats.condIndepC0
                  ? 'Cond. Independent given C=0 \u2713'
                  : 'Cond. Dependent given C=0 \u2717'}
              </div>
            </>
          ) : (
            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              P(C=0) = 0 — conditioning undefined
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
