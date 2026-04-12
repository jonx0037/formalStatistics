import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfNormal, cdfNormal } from './shared/distributions';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

type Tab = 'empirical' | 'sum' | 'standardize';

export default function NormalPropertiesExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [tab, setTab] = useState<Tab>('empirical');
  const [mu, setMu] = useState(0);
  const [sigma, setSigma] = useState(1);
  const [mu2, setMu2] = useState(2);
  const [sigma2, setSigma2] = useState(1.5);

  const chartW = Math.max(280, (width || 600) - 16);
  const chartH = 220;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // 68-95-99.7 tab
  const empiricalData = useMemo(() => {
    if (tab !== 'empirical') return null;
    const sigma2Val = sigma * sigma;
    const lo = mu - 4 * sigma;
    const hi = mu + 4 * sigma;
    const step = (hi - lo) / NUM_POINTS;
    const curve: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = lo + i * step;
      curve.push({ x, y: pdfNormal(x, mu, sigma2Val) });
    }
    const maxY = Math.max(...curve.map((d) => d.y));
    const bands = [
      { k: 1, pct: (cdfNormal(mu + sigma, mu, sigma2Val) - cdfNormal(mu - sigma, mu, sigma2Val)) * 100, color: 'rgba(37,99,235,0.3)' },
      { k: 2, pct: (cdfNormal(mu + 2 * sigma, mu, sigma2Val) - cdfNormal(mu - 2 * sigma, mu, sigma2Val)) * 100, color: 'rgba(37,99,235,0.2)' },
      { k: 3, pct: (cdfNormal(mu + 3 * sigma, mu, sigma2Val) - cdfNormal(mu - 3 * sigma, mu, sigma2Val)) * 100, color: 'rgba(37,99,235,0.1)' },
    ];
    return { curve, maxY, lo, hi, bands };
  }, [tab, mu, sigma]);

  // Sum of Normals tab
  const sumData = useMemo(() => {
    if (tab !== 'sum') return null;
    const s1sq = sigma * sigma;
    const s2sq = sigma2 * sigma2;
    const muSum = mu + mu2;
    const varSum = s1sq + s2sq;
    const sigmaSum = Math.sqrt(varSum);

    const lo = Math.min(mu - 4 * sigma, mu2 - 4 * sigma2, muSum - 4 * sigmaSum);
    const hi = Math.max(mu + 4 * sigma, mu2 + 4 * sigma2, muSum + 4 * sigmaSum);
    const step = (hi - lo) / NUM_POINTS;

    const curve1: { x: number; y: number }[] = [];
    const curve2: { x: number; y: number }[] = [];
    const curveSum: { x: number; y: number }[] = [];
    let maxY = 0;
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = lo + i * step;
      const y1 = pdfNormal(x, mu, s1sq);
      const y2 = pdfNormal(x, mu2, s2sq);
      const yS = pdfNormal(x, muSum, varSum);
      curve1.push({ x, y: y1 });
      curve2.push({ x, y: y2 });
      curveSum.push({ x, y: yS });
      maxY = Math.max(maxY, y1, y2, yS);
    }
    return { curve1, curve2, curveSum, maxY, lo, hi, muSum, varSum };
  }, [tab, mu, sigma, mu2, sigma2]);

  // Standardization tab
  const stdData = useMemo(() => {
    if (tab !== 'standardize') return null;
    const s2 = sigma * sigma;
    const loOrig = mu - 4 * sigma;
    const hiOrig = mu + 4 * sigma;
    const step = (hiOrig - loOrig) / NUM_POINTS;

    const curveOrig: { x: number; y: number }[] = [];
    const curveStd: { x: number; y: number }[] = [];
    let maxOrig = 0;
    let maxStd = 0;
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = loOrig + i * step;
      const y = pdfNormal(x, mu, s2);
      curveOrig.push({ x, y });
      maxOrig = Math.max(maxOrig, y);

      const z = -4 + (i / NUM_POINTS) * 8;
      const yStd = pdfNormal(z, 0, 1);
      curveStd.push({ x: z, y: yStd });
      maxStd = Math.max(maxStd, yStd);
    }
    return { curveOrig, curveStd, maxOrig, maxStd, loOrig, hiOrig };
  }, [tab, mu, sigma]);

  // Generic path builder
  const buildPath = useCallback((
    data: { x: number; y: number }[],
    xMin: number, xMax: number, yMax: number
  ) => {
    return data
      .map((d, i) => {
        const px = MARGIN.left + ((d.x - xMin) / (xMax - xMin)) * plotW;
        const py = MARGIN.top + plotH - (d.y / (yMax * 1.15)) * plotH;
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  }, [plotW, plotH]);

  // Band fill path builder for empirical rule
  const buildBandFill = useCallback((
    data: { x: number; y: number }[],
    xMin: number, xMax: number, yMax: number,
    bandLo: number, bandHi: number
  ) => {
    const pts = data.filter((d) => d.x >= bandLo && d.x <= bandHi);
    if (pts.length < 2) return '';
    const xScl = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScl = (v: number) => MARGIN.top + plotH - (v / (yMax * 1.15)) * plotH;
    let path = pts.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScl(d.x).toFixed(1)},${yScl(d.y).toFixed(1)}`).join(' ');
    path += ` L${xScl(pts[pts.length - 1].x).toFixed(1)},${yScl(0).toFixed(1)}`;
    path += ` L${xScl(pts[0].x).toFixed(1)},${yScl(0).toFixed(1)} Z`;
    return path;
  }, [plotW, plotH]);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Normal Properties Explorer
      </div>

      {/* Tabs */}
      <div className="flex gap-1.5 justify-center mb-4">
        {([['empirical', '68-95-99.7 Rule'], ['sum', 'Sum of Normals'], ['standardize', 'Standardization']] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className="px-3 py-1 text-xs rounded-full transition-colors"
            style={{
              backgroundColor: tab === key ? '#2563eb' : 'transparent',
              color: tab === key ? 'white' : 'currentColor',
              border: '1px solid #2563eb',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">μ</span>
          <input type="range" min={-5} max={5} step={0.1} value={mu} onChange={(e) => setMu(Number(e.target.value))} className="w-20" />
          <span className="w-8 tabular-nums">{mu.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">σ</span>
          <input type="range" min={0.2} max={4} step={0.1} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="w-20" />
          <span className="w-8 tabular-nums">{sigma.toFixed(1)}</span>
        </label>
        {tab === 'sum' && (
          <>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">μ₂</span>
              <input type="range" min={-5} max={5} step={0.1} value={mu2} onChange={(e) => setMu2(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{mu2.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">σ₂</span>
              <input type="range" min={0.2} max={4} step={0.1} value={sigma2} onChange={(e) => setSigma2(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{sigma2.toFixed(1)}</span>
            </label>
          </>
        )}
      </div>

      {/* 68-95-99.7 tab */}
      {tab === 'empirical' && empiricalData && (
        <div>
          <svg width={chartW} height={chartH} className="block mx-auto">
            {/* Bands (draw outermost first) */}
            {[...empiricalData.bands].reverse().map((band) => (
              <path
                key={band.k}
                d={buildBandFill(empiricalData.curve, empiricalData.lo, empiricalData.hi, empiricalData.maxY, mu - band.k * sigma, mu + band.k * sigma)}
                fill={band.color}
              />
            ))}
            {/* Curve */}
            <path d={buildPath(empiricalData.curve, empiricalData.lo, empiricalData.hi, empiricalData.maxY)} fill="none" stroke="#2563eb" strokeWidth={2} />
          </svg>
          <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
            {empiricalData.bands.map((band) => (
              <span key={band.k}>μ ± {band.k}σ: <strong>{band.pct.toFixed(2)}%</strong></span>
            ))}
          </div>
        </div>
      )}

      {/* Sum tab */}
      {tab === 'sum' && sumData && (
        <div>
          <svg width={chartW} height={chartH} className="block mx-auto">
            <path d={buildPath(sumData.curve1, sumData.lo, sumData.hi, sumData.maxY)} fill="none" stroke="#2563eb" strokeWidth={1.5} />
            <path d={buildPath(sumData.curve2, sumData.lo, sumData.hi, sumData.maxY)} fill="none" stroke="#16a34a" strokeWidth={1.5} />
            <path d={buildPath(sumData.curveSum, sumData.lo, sumData.hi, sumData.maxY)} fill="none" stroke="#DC2626" strokeWidth={2} strokeDasharray="6,3" />
          </svg>
          <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
            <span style={{ color: '#2563eb' }}>X₁ ~ N({mu}, {(sigma * sigma).toFixed(1)})</span>
            <span style={{ color: '#16a34a' }}>X₂ ~ N({mu2}, {(sigma2 * sigma2).toFixed(1)})</span>
            <span style={{ color: '#DC2626' }}>X₁+X₂ ~ N({sumData.muSum.toFixed(1)}, {sumData.varSum.toFixed(1)})</span>
          </div>
        </div>
      )}

      {/* Standardization tab */}
      {tab === 'standardize' && stdData && (
        <div className="flex flex-wrap gap-4 justify-center">
          <div>
            <div className="text-center text-xs font-medium mb-1">Original: N({mu}, {(sigma * sigma).toFixed(1)})</div>
            <svg width={Math.min(chartW / 2, 300)} height={chartH} className="block">
              <path d={buildPath(stdData.curveOrig, stdData.loOrig, stdData.hiOrig, stdData.maxOrig)} fill="none" stroke="#2563eb" strokeWidth={2} />
            </svg>
          </div>
          <div>
            <div className="text-center text-xs font-medium mb-1">Standardized: Z ~ N(0, 1)</div>
            <svg width={Math.min(chartW / 2, 300)} height={chartH} className="block">
              <path d={buildPath(stdData.curveStd, -4, 4, stdData.maxStd)} fill="none" stroke="#DC2626" strokeWidth={2} />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}
