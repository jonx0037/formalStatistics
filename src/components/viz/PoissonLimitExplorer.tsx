import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBinomial, pmfPoisson,
  expectationBinomial, varianceBinomial,
  expectationPoisson, variancePoisson,
} from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };

export default function PoissonLimitExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [lambda, setLambda] = useState(5);
  const [n, setN] = useState(20);
  const [animating, setAnimating] = useState(false);
  const animRef = useRef<number | null>(null);
  const lastFrameRef = useRef(0);

  const chartW = Math.max(300, (width || 600) - 32);
  const chartH = 280;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const maxK = Math.min(Math.ceil(lambda + 4 * Math.sqrt(lambda)) + 5, 45);

  // PMF data
  const poissonPMF = useMemo(() => {
    return Array.from({ length: maxK + 1 }, (_, k) => ({
      k,
      p: pmfPoisson(k, lambda),
    }));
  }, [lambda, maxK]);

  const binomialPMF = useMemo(() => {
    const p = lambda / n;
    if (p >= 1) return [];
    return Array.from({ length: maxK + 1 }, (_, k) => ({
      k,
      p: k <= n ? pmfBinomial(k, n, p) : 0,
    }));
  }, [lambda, n, maxK]);

  const maxP = useMemo(() => {
    const allP = [...poissonPMF.map((d) => d.p), ...binomialPMF.map((d) => d.p)];
    return Math.max(...allP, 0.01);
  }, [poissonPMF, binomialPMF]);

  // Total variation distance
  const tvDistance = useMemo(() => {
    let sum = 0;
    for (let k = 0; k <= maxK; k++) {
      const pPois = pmfPoisson(k, lambda);
      const pBinom = lambda / n < 1 && k <= n ? pmfBinomial(k, n, lambda / n) : 0;
      sum += Math.abs(pPois - pBinom);
    }
    return sum / 2;
  }, [lambda, n, maxK]);

  // Moments comparison
  const binP = lambda / n;
  const eBinom = binP < 1 ? expectationBinomial(n, binP) : 0;
  const vBinom = binP < 1 ? varianceBinomial(n, binP) : 0;
  const ePois = expectationPoisson(lambda);
  const vPois = variancePoisson(lambda);

  // Scales
  const barGroupW = plotW / (maxK + 1);
  const barW = Math.max(2, barGroupW * 0.35);

  const xScale = useCallback(
    (k: number) => MARGIN.left + (k + 0.5) / (maxK + 1) * plotW,
    [maxK, plotW],
  );
  const yScale = useCallback(
    (v: number) => MARGIN.top + plotH - (v / (maxP * 1.15)) * plotH,
    [plotH, maxP],
  );

  // TV color
  const tvColor = tvDistance < 0.01 ? '#059669' : tvDistance < 0.05 ? '#D97706' : '#DC2626';

  // Animation
  useEffect(() => {
    if (!animating) return;
    const step = (timestamp: number) => {
      if (timestamp - lastFrameRef.current > 100) {
        lastFrameRef.current = timestamp;
        setN((prev) => {
          const next = prev + Math.max(1, Math.floor(prev * 0.08));
          if (next >= 500) {
            setAnimating(false);
            return 500;
          }
          return next;
        });
      }
      animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [animating]);

  // Map n slider to non-linear scale for fast convergence viz
  const nFromSlider = (v: number) => {
    if (v <= 50) return Math.max(5, v);
    return 50 + (v - 50) * 9; // 50..100 maps to 50..500
  };
  const sliderFromN = (nVal: number) => {
    if (nVal <= 50) return nVal;
    return 50 + (nVal - 50) / 9;
  };

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Poisson Limit Theorem
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-6 justify-center mb-4 items-center">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">λ</span>
          <input type="range" min={1} max={20} step={0.5} value={lambda} onChange={(e) => setLambda(Number(e.target.value))} className="w-24" />
          <span className="w-8 tabular-nums">{lambda}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range" min={5} max={100} step={1}
            value={sliderFromN(n)}
            onChange={(e) => { setN(nFromSlider(Number(e.target.value))); setAnimating(false); }}
            className="w-28"
          />
          <span className="w-10 tabular-nums">{n}</span>
        </label>
        <button
          onClick={() => { setN(5); setAnimating(true); lastFrameRef.current = 0; }}
          className="px-3 py-1 text-xs rounded border hover:bg-gray-50 transition-colors"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {animating ? 'Animating...' : 'Animate'}
        </button>
      </div>

      {/* Chart */}
      <svg width={chartW} height={chartH} className="block mx-auto">
        {/* Y grid */}
        {[0, maxP * 0.25, maxP * 0.5, maxP * 0.75, maxP].map((v, i) => (
          <g key={i}>
            <line x1={MARGIN.left} y1={yScale(v)} x2={chartW - MARGIN.right} y2={yScale(v)} stroke="currentColor" strokeOpacity={0.08} />
            <text x={MARGIN.left - 4} y={yScale(v) + 3} textAnchor="end" className="fill-current" style={{ fontSize: '8px' }}>
              {v.toFixed(v < 0.01 ? 3 : 2)}
            </text>
          </g>
        ))}

        {/* Poisson bars (reference — behind) */}
        {poissonPMF.map(({ k, p }) => (
          <rect
            key={`poi-${k}`}
            x={xScale(k) - barW * 0.8}
            y={yScale(p)}
            width={barW * 0.75}
            height={Math.max(0, yScale(0) - yScale(p))}
            fill="#DC2626"
            opacity={0.35}
            rx={1}
          />
        ))}

        {/* Binomial bars (overlay) */}
        {binomialPMF.map(({ k, p }) => (
          <rect
            key={`bin-${k}`}
            x={xScale(k) + barW * 0.05}
            y={yScale(p)}
            width={barW * 0.75}
            height={Math.max(0, yScale(0) - yScale(p))}
            fill="var(--color-primary, #2563EB)"
            opacity={0.55}
            rx={1}
          />
        ))}

        {/* X axis */}
        {poissonPMF.filter((_, i) => poissonPMF.length <= 20 || i % Math.ceil(poissonPMF.length / 15) === 0).map(({ k }) => (
          <text key={k} x={xScale(k)} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
            {k}
          </text>
        ))}

        {/* Legend */}
        <rect x={chartW - 150} y={MARGIN.top + 4} width={10} height={10} fill="#DC2626" opacity={0.5} rx={1} />
        <text x={chartW - 136} y={MARGIN.top + 13} className="fill-current" style={{ fontSize: '9px' }}>Poisson(λ={lambda})</text>
        <rect x={chartW - 150} y={MARGIN.top + 20} width={10} height={10} fill="var(--color-primary, #2563EB)" opacity={0.65} rx={1} />
        <text x={chartW - 136} y={MARGIN.top + 29} className="fill-current" style={{ fontSize: '9px' }}>Binomial(n={n}, p={(lambda / n).toFixed(4)})</text>
      </svg>

      {/* TV distance and moments */}
      <div className="flex flex-wrap gap-6 justify-center mt-3 text-xs">
        <span>
          <b>TV distance:</b>{' '}
          <span style={{ color: tvColor, fontWeight: 600 }}>{tvDistance.toFixed(4)}</span>
          {' '}
          <span className="px-1.5 py-0.5 rounded text-white text-[10px]" style={{ backgroundColor: tvColor }}>
            {tvDistance < 0.01 ? 'Excellent' : tvDistance < 0.05 ? 'Good' : 'Divergent'}
          </span>
        </span>
      </div>

      <div className="mt-2 text-xs text-center">
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
              <td className="px-3 py-0.5 font-medium" style={{ color: '#DC2626' }}>Poisson</td>
              <td className="px-3 py-0.5 tabular-nums">{ePois.toFixed(4)}</td>
              <td className="px-3 py-0.5 tabular-nums">{vPois.toFixed(4)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
