import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBinomial, pmfHypergeometric,
  varianceBinomial, varianceHypergeometric,
  expectationBinomial, expectationHypergeometric,
} from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };

export default function SamplingComparisonExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [N, setN] = useState(50);
  const [pFrac, setPFrac] = useState(0.4); // K/N
  const [sampleN, setSampleN] = useState(10);

  const K = Math.round(N * pFrac);
  const nClamped = Math.min(sampleN, N);

  const panelW = Math.max(200, ((width || 600) - 32) / 2);
  const chartH = 240;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Support range
  const lo = Math.max(0, nClamped - (N - K));
  const hi = Math.min(nClamped, K);

  // PMF data
  const binomPMF = useMemo(() => {
    return Array.from({ length: hi - lo + 1 }, (_, i) => {
      const k = lo + i;
      return { k, p: pmfBinomial(k, nClamped, pFrac) };
    });
  }, [nClamped, pFrac, lo, hi]);

  const hyperPMF = useMemo(() => {
    return Array.from({ length: hi - lo + 1 }, (_, i) => {
      const k = lo + i;
      return { k, p: pmfHypergeometric(k, N, K, nClamped) };
    });
  }, [N, K, nClamped, lo, hi]);

  const maxP = useMemo(() => {
    return Math.max(...binomPMF.map((d) => d.p), ...hyperPMF.map((d) => d.p), 0.01);
  }, [binomPMF, hyperPMF]);

  // Moments
  const eBinom = expectationBinomial(nClamped, pFrac);
  const vBinom = varianceBinomial(nClamped, pFrac);
  const eHyper = expectationHypergeometric(N, K, nClamped);
  const vHyper = varianceHypergeometric(N, K, nClamped);
  const fpc = N > 1 ? (N - nClamped) / (N - 1) : 1;
  const samplingFrac = nClamped / N;

  // Scales
  const barW = Math.max(2, plotW / (hi - lo + 1) - 2);
  const xScale = useCallback(
    (k: number) => MARGIN.left + ((k - lo + 0.5) / (hi - lo + 1)) * plotW,
    [lo, hi, plotW],
  );
  const yScale = useCallback(
    (v: number) => MARGIN.top + plotH - (v / (maxP * 1.15)) * plotH,
    [plotH, maxP],
  );

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: With vs. Without Replacement
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">N</span>
          <input type="range" min={20} max={1000} step={10} value={N} onChange={(e) => setN(Number(e.target.value))} className="w-24" />
          <span className="w-12 tabular-nums">{N}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">K/N</span>
          <input type="range" min={0.1} max={0.9} step={0.01} value={pFrac} onChange={(e) => setPFrac(Number(e.target.value))} className="w-24" />
          <span className="w-10 tabular-nums">{pFrac.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input type="range" min={1} max={Math.min(N, 50)} step={1} value={nClamped} onChange={(e) => setSampleN(Number(e.target.value))} className="w-24" />
          <span className="w-8 tabular-nums">{nClamped}</span>
        </label>
      </div>

      {/* Side-by-side PMF panels */}
      <div className="flex flex-wrap gap-4 justify-center">
        {/* Binomial */}
        <div>
          <div className="text-center text-xs font-medium mb-1" style={{ color: 'var(--color-primary, #2563EB)' }}>
            Binomial(n={nClamped}, p={pFrac.toFixed(2)}) — with replacement
          </div>
          <svg width={panelW} height={chartH} className="block">
            {[0, maxP * 0.5, maxP].map((v, i) => (
              <g key={i}>
                <line x1={MARGIN.left} y1={yScale(v)} x2={panelW - MARGIN.right} y2={yScale(v)} stroke="currentColor" strokeOpacity={0.08} />
                <text x={MARGIN.left - 4} y={yScale(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(3)}
                </text>
              </g>
            ))}
            {binomPMF.map(({ k, p }) => (
              <rect
                key={k}
                x={xScale(k) - barW / 2}
                y={yScale(p)}
                width={barW}
                height={Math.max(0, yScale(0) - yScale(p))}
                fill="var(--color-primary, #2563EB)"
                opacity={0.6}
                rx={1}
              />
            ))}
            {binomPMF.filter((_, i) => binomPMF.length <= 20 || i % Math.ceil(binomPMF.length / 12) === 0).map(({ k }) => (
              <text key={k} x={xScale(k)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>{k}</text>
            ))}
          </svg>
        </div>

        {/* Hypergeometric */}
        <div>
          <div className="text-center text-xs font-medium mb-1" style={{ color: '#0ea5e9' }}>
            Hypergeometric(N={N}, K={K}, n={nClamped}) — without replacement
          </div>
          <svg width={panelW} height={chartH} className="block">
            {[0, maxP * 0.5, maxP].map((v, i) => (
              <g key={i}>
                <line x1={MARGIN.left} y1={yScale(v)} x2={panelW - MARGIN.right} y2={yScale(v)} stroke="currentColor" strokeOpacity={0.08} />
                <text x={MARGIN.left - 4} y={yScale(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
                  {v.toFixed(3)}
                </text>
              </g>
            ))}
            {hyperPMF.map(({ k, p }) => (
              <rect
                key={k}
                x={xScale(k) - barW / 2}
                y={yScale(p)}
                width={barW}
                height={Math.max(0, yScale(0) - yScale(p))}
                fill="#0ea5e9"
                opacity={0.6}
                rx={1}
              />
            ))}
            {hyperPMF.filter((_, i) => hyperPMF.length <= 20 || i % Math.ceil(hyperPMF.length / 12) === 0).map(({ k }) => (
              <text key={k} x={xScale(k)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>{k}</text>
            ))}
          </svg>
        </div>
      </div>

      {/* Stats comparison */}
      <div className="mt-4 text-xs text-center">
        <table className="mx-auto">
          <thead>
            <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
              <th className="px-3 py-1"></th>
              <th className="px-3 py-1">E[X]</th>
              <th className="px-3 py-1">Var(X)</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
              <td className="px-3 py-0.5 font-medium" style={{ color: 'var(--color-primary, #2563EB)' }}>Binomial</td>
              <td className="px-3 py-0.5 tabular-nums">{eBinom.toFixed(4)}</td>
              <td className="px-3 py-0.5 tabular-nums">{vBinom.toFixed(4)}</td>
            </tr>
            <tr>
              <td className="px-3 py-0.5 font-medium" style={{ color: '#0ea5e9' }}>Hypergeometric</td>
              <td className="px-3 py-0.5 tabular-nums">{eHyper.toFixed(4)}</td>
              <td className="px-3 py-0.5 tabular-nums">{vHyper.toFixed(4)}</td>
            </tr>
          </tbody>
        </table>

        <div className="mt-3 flex flex-wrap gap-4 justify-center">
          <span><b>n/N</b> = {samplingFrac.toFixed(4)}</span>
          <span><b>FPC</b> = (N−n)/(N−1) = {fpc.toFixed(4)}</span>
          <span><b>Var ratio</b> = {vBinom > 0 ? (vHyper / vBinom).toFixed(4) : '—'}</span>
        </div>

        <div className="mt-2">
          <span
            className="inline-block px-2 py-0.5 rounded text-xs font-medium"
            style={{
              backgroundColor: samplingFrac < 0.05 ? '#D1FAE5' : '#FEF3C7',
              color: samplingFrac < 0.05 ? '#065F46' : '#92400E',
            }}
          >
            {samplingFrac < 0.05
              ? 'Binomial approximation is excellent (n/N < 5%)'
              : `FPC matters — sampling ${(samplingFrac * 100).toFixed(1)}% of the population`}
          </span>
        </div>
      </div>
    </div>
  );
}
