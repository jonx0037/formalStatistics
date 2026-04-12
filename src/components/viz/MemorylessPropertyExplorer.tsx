import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pmfGeometric, cdfGeometric } from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };

export default function MemorylessPropertyExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [p, setP] = useState(0.25);
  const [s, setS] = useState(5);

  const q = 1 - p;
  const maxK = Math.min(Math.ceil(8 / p), 40);
  const panelW = Math.max(200, ((width || 600) - 32) / 2);
  const chartH = 250;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Full PMF
  const fullPMF = useMemo(() => {
    const data: { k: number; p: number }[] = [];
    for (let k = 1; k <= maxK; k++) {
      data.push({ k, p: pmfGeometric(k, p) });
    }
    return data;
  }, [p, maxK]);

  // Conditional PMF: P(X = k | X > s) = P(X = k) / P(X > s) for k > s
  const condPMF = useMemo(() => {
    const survivalS = Math.pow(q, s); // P(X > s) = q^s
    const data: { k: number; pCond: number }[] = [];
    for (let k = s + 1; k <= maxK; k++) {
      data.push({ k, pCond: pmfGeometric(k, p) / survivalS });
    }
    return data;
  }, [p, q, s, maxK]);

  // Re-indexed: conditional PMF shifted back so k -> k - s
  const reindexedPMF = useMemo(() => {
    return condPMF.map(({ k, pCond }) => ({
      kShifted: k - s,
      pCond,
    }));
  }, [condPMF, s]);

  // Verification table: P(X > s+t | X > s) vs P(X > t) for t = 1..5
  const verification = useMemo(() => {
    return [1, 2, 3, 4, 5].map((t) => ({
      t,
      conditional: Math.pow(q, s + t) / Math.pow(q, s), // P(X > s+t | X > s)
      unconditional: Math.pow(q, t), // P(X > t)
    }));
  }, [q, s]);

  const maxP = useMemo(() => Math.max(...fullPMF.map((d) => d.p), 0.01), [fullPMF]);

  // Scales for left panel (full range)
  const xScaleL = useCallback(
    (v: number) => MARGIN.left + ((v - 0.5) / maxK) * plotW,
    [maxK, plotW],
  );
  const yScaleL = useCallback(
    (v: number) => MARGIN.top + plotH - (v / (maxP * 1.2)) * plotH,
    [plotH, maxP],
  );

  // Scales for right panel (re-indexed)
  const reMaxK = maxK - s;
  const xScaleR = useCallback(
    (v: number) => MARGIN.left + ((v - 0.5) / Math.max(reMaxK, 1)) * plotW,
    [reMaxK, plotW],
  );

  const barW = Math.max(2, plotW / maxK - 1.5);
  const barWR = Math.max(2, plotW / Math.max(reMaxK, 1) - 1.5);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Memoryless Property Explorer
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-6 justify-center mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">p</span>
          <input type="range" min={0.05} max={0.9} step={0.01} value={p} onChange={(e) => setP(Number(e.target.value))} className="w-28" />
          <span className="w-8 tabular-nums">{p.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">s</span>
          <input type="range" min={1} max={15} step={1} value={s} onChange={(e) => setS(Number(e.target.value))} className="w-28" />
          <span className="w-8 tabular-nums">{s}</span>
        </label>
      </div>

      {/* Charts */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* Left: Full PMF with conditional overlay */}
        <div>
          <div className="text-center text-xs font-medium mb-1">Full PMF + Conditional (given X {'>'} {s})</div>
          <svg width={panelW} height={chartH} className="block">
            {/* Full PMF bars */}
            {fullPMF.map(({ k, p: prob }) => (
              <rect
                key={k}
                x={xScaleL(k) - barW / 2}
                y={yScaleL(prob)}
                width={barW}
                height={Math.max(0, yScaleL(0) - yScaleL(prob))}
                fill="var(--color-primary, #2563EB)"
                opacity={k <= s ? 0.3 : 0.5}
                rx={1}
              />
            ))}

            {/* Conditional PMF overlaid as narrower bars */}
            {condPMF.map(({ k, pCond }) => (
              <rect
                key={`c-${k}`}
                x={xScaleL(k) - barW / 4}
                y={yScaleL(pCond)}
                width={barW / 2}
                height={Math.max(0, yScaleL(0) - yScaleL(pCond))}
                fill="#DC2626"
                opacity={0.7}
                rx={1}
              />
            ))}

            {/* Conditioning boundary */}
            <line
              x1={xScaleL(s + 0.5)} y1={MARGIN.top}
              x2={xScaleL(s + 0.5)} y2={yScaleL(0)}
              stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4,3"
            />
            <text x={xScaleL(s + 0.5)} y={MARGIN.top - 2} textAnchor="middle" fill="#DC2626" style={{ fontSize: '9px', fontWeight: 600 }}>
              X {'>'} {s}
            </text>

            {/* Shading for X ≤ s region */}
            <rect
              x={MARGIN.left} y={MARGIN.top}
              width={xScaleL(s + 0.5) - MARGIN.left} height={plotH}
              fill="#DC2626" opacity={0.04}
            />

            {/* X axis ticks */}
            {fullPMF.filter((_, i) => fullPMF.length <= 20 || i % Math.ceil(fullPMF.length / 12) === 0).map(({ k }) => (
              <text key={k} x={xScaleL(k)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                {k}
              </text>
            ))}
          </svg>
        </div>

        {/* Right: Re-indexed conditional PMF vs unconditional */}
        <div>
          <div className="text-center text-xs font-medium mb-1">Re-indexed: same shape as original</div>
          <svg width={panelW} height={chartH} className="block">
            {/* Unconditional PMF (ghost) */}
            {fullPMF.filter(({ k }) => k <= reMaxK).map(({ k, p: prob }) => (
              <rect
                key={`u-${k}`}
                x={xScaleR(k) - barWR / 2}
                y={yScaleL(prob)}
                width={barWR}
                height={Math.max(0, yScaleL(0) - yScaleL(prob))}
                fill="var(--color-primary, #2563EB)"
                opacity={0.25}
                rx={1}
              />
            ))}

            {/* Re-indexed conditional PMF */}
            {reindexedPMF.map(({ kShifted, pCond }) => (
              <rect
                key={`r-${kShifted}`}
                x={xScaleR(kShifted) - barWR / 3}
                y={yScaleL(pCond)}
                width={barWR * 0.65}
                height={Math.max(0, yScaleL(0) - yScaleL(pCond))}
                fill="#DC2626"
                opacity={0.65}
                rx={1}
              />
            ))}

            <text x={panelW / 2} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '9px' }}>
              Re-indexed k (starting from 1)
            </text>
          </svg>
        </div>
      </div>

      {/* Verification table */}
      <div className="mt-4 text-xs text-center">
        <div className="font-medium mb-2">Verification: P(X {'>'} s+t | X {'>'} s) = P(X {'>'} t)</div>
        <table className="mx-auto">
          <thead>
            <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
              <th className="px-3 py-1">t</th>
              <th className="px-3 py-1">P(X {'>'} {s}+t | X {'>'} {s})</th>
              <th className="px-3 py-1">P(X {'>'} t)</th>
              <th className="px-3 py-1">Match?</th>
            </tr>
          </thead>
          <tbody>
            {verification.map(({ t, conditional, unconditional }) => (
              <tr key={t} className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                <td className="px-3 py-0.5 tabular-nums">{t}</td>
                <td className="px-3 py-0.5 tabular-nums">{conditional.toFixed(6)}</td>
                <td className="px-3 py-0.5 tabular-nums">{unconditional.toFixed(6)}</td>
                <td className="px-3 py-0.5">{Math.abs(conditional - unconditional) < 1e-10 ? '=' : '≠'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-2 text-gray-500 italic">The past doesn't help predict the future �� the trials are independent.</div>
      </div>
    </div>
  );
}
