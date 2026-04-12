import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { logPartitionFunctions, canonicalFormColors } from '../../data/exponential-family-data';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;
const READOUT_W = 160;
const READOUT_GAP = 16;
const DIST_KEYS = ['Bernoulli', 'Geometric', 'Poisson', 'Exponential'] as const;

export default function LogPartitionExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [distKey, setDistKey] = useState<string>('Bernoulli');
  const [eta, setEta] = useState<number>(0);
  const [showDerivatives, setShowDerivatives] = useState(false);
  const circleRef = useRef<SVGCircleElement>(null);

  const lp = logPartitionFunctions[distKey];
  const [domainMin, domainMax] = lp.etaDomain;

  // Reset eta to midpoint when distribution changes
  useEffect(() => {
    const [min, max] = logPartitionFunctions[distKey].etaDomain;
    setEta((min + max) / 2);
  }, [distKey]);

  const containerW = Math.max(280, (width || 600) - 16);
  const isNarrow = containerW < 500;
  const svgW = isNarrow ? containerW : containerW - READOUT_W - READOUT_GAP;
  const chartH = 260;
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Scales
  const xScale = useCallback(
    (v: number) => MARGIN.left + ((v - domainMin) / (domainMax - domainMin)) * plotW,
    [domainMin, domainMax, plotW],
  );
  const xInvert = useCallback(
    (px: number) => domainMin + ((px - MARGIN.left) / plotW) * (domainMax - domainMin),
    [domainMin, domainMax, plotW],
  );

  // Compute all curve data
  const { aCurve, apCurve, appCurve, yMin, yMax } = useMemo(() => {
    const aData: { x: number; y: number }[] = [];
    const apData: { x: number; y: number }[] = [];
    const appData: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = domainMin + (i / NUM_POINTS) * (domainMax - domainMin);
      const av = lp.A(x);
      const apv = lp.Aprime(x);
      const appv = lp.Adoubleprime(x);
      if (isFinite(av)) aData.push({ x, y: av });
      if (isFinite(apv)) apData.push({ x, y: apv });
      if (isFinite(appv)) appData.push({ x, y: appv });
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const d of aData) { lo = Math.min(lo, d.y); hi = Math.max(hi, d.y); }
    if (showDerivatives) {
      for (const d of apData) { lo = Math.min(lo, d.y); hi = Math.max(hi, d.y); }
      for (const d of appData) { lo = Math.min(lo, d.y); hi = Math.max(hi, d.y); }
    }
    const range = hi - lo || 1;
    return {
      aCurve: aData, apCurve: apData, appCurve: appData,
      yMin: lo - range * 0.08, yMax: hi + range * 0.08,
    };
  }, [lp, domainMin, domainMax, showDerivatives]);

  const yScale = useCallback(
    (v: number) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH,
    [yMin, yMax, plotH],
  );

  const buildPath = useCallback(
    (data: { x: number; y: number }[]) =>
      data
        .map((d, i) => {
          const px = xScale(d.x);
          const py = yScale(d.y);
          return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
        })
        .join(' '),
    [xScale, yScale],
  );

  // Current point values
  const aVal = lp.A(eta);
  const apVal = lp.Aprime(eta);
  const appVal = lp.Adoubleprime(eta);

  // Tangent line endpoints (extend across visible x range)
  const tangent = useMemo(() => {
    const slope = apVal;
    const y0 = aVal + slope * (domainMin - eta);
    const y1 = aVal + slope * (domainMax - eta);
    return { x0: domainMin, y0, x1: domainMax, y1 };
  }, [eta, aVal, apVal, domainMin, domainMax]);

  // D3 drag behavior
  useEffect(() => {
    const circle = circleRef.current;
    if (!circle) return;
    const dragBehavior = d3
      .drag<SVGCircleElement, unknown>()
      .on('drag', (event) => {
        const newEta = xInvert(event.x);
        const [min, max] = logPartitionFunctions[distKey].etaDomain;
        setEta(Math.max(min + 0.01, Math.min(max - 0.01, newEta)));
      });
    d3.select(circle).call(dragBehavior);
    return () => { d3.select(circle).on('.drag', null); };
  }, [xInvert, distKey]);

  // Axis ticks
  const xTicks = useMemo(() => {
    const span = domainMax - domainMin;
    const rawStep = span / 6;
    const step = rawStep < 1 ? Math.ceil(rawStep * 10) / 10 : Math.ceil(rawStep);
    const ticks: number[] = [];
    const start = Math.ceil(domainMin / step) * step;
    for (let v = start; v <= domainMax + step * 0.01; v += step) ticks.push(+v.toFixed(4));
    return ticks;
  }, [domainMin, domainMax]);

  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    for (let i = 0; i <= 5; i++) ticks.push(yMin + (i / 5) * (yMax - yMin));
    return ticks;
  }, [yMin, yMax]);

  const rightEdge = MARGIN.left + plotW;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Log-Partition Function Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-center mb-4 text-xs">
        <label className="flex items-center gap-2">
          <span className="font-medium">Distribution</span>
          <select
            value={distKey}
            onChange={(e) => setDistKey(e.target.value)}
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg, #fff)' }}
          >
            {DIST_KEYS.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showDerivatives} onChange={(e) => setShowDerivatives(e.target.checked)} />
          <span>Show A&prime;(&#951;) and A&Prime;(&#951;)</span>
        </label>
      </div>

      {/* Chart + readout */}
      <div className={`flex ${isNarrow ? 'flex-col' : 'flex-row'} gap-4 items-start justify-center`}>
        <svg width={svgW} height={chartH} className="block shrink-0">
          <defs>
            <clipPath id="lp-clip">
              <rect x={MARGIN.left} y={MARGIN.top} width={plotW} height={plotH} />
            </clipPath>
          </defs>

          {/* Grid */}
          {yTicks.map((v, i) => (
            <line key={`yg${i}`} x1={MARGIN.left} y1={yScale(v)} x2={rightEdge} y2={yScale(v)}
              stroke="currentColor" strokeOpacity={0.08} />
          ))}
          {xTicks.map((v, i) => (
            <line key={`xg${i}`} x1={xScale(v)} y1={MARGIN.top} x2={xScale(v)} y2={MARGIN.top + plotH}
              stroke="currentColor" strokeOpacity={0.08} />
          ))}

          {/* Tangent line (clipped) */}
          <line
            x1={xScale(tangent.x0)} y1={yScale(tangent.y0)}
            x2={xScale(tangent.x1)} y2={yScale(tangent.y1)}
            stroke="#9ca3af" strokeWidth={1} strokeDasharray="4,3"
            clipPath="url(#lp-clip)"
          />

          {/* A(η) curve */}
          <path d={buildPath(aCurve)} fill="none" stroke={canonicalFormColors.A} strokeWidth={2.5} />

          {/* Derivative overlays */}
          {showDerivatives && (
            <>
              <path d={buildPath(apCurve)} fill="none" stroke="#16a34a" strokeWidth={2} strokeDasharray="6,3" />
              <path d={buildPath(appCurve)} fill="none" stroke="#7c3aed" strokeWidth={2} strokeDasharray="4,4" />
            </>
          )}

          {/* Draggable point */}
          {isFinite(aVal) && (
            <circle
              ref={circleRef}
              cx={xScale(eta)} cy={yScale(aVal)} r={8}
              fill={canonicalFormColors.A} stroke="white" strokeWidth={2}
              style={{ cursor: 'ew-resize' }}
            />
          )}

          {/* X axis */}
          <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={rightEdge} y2={MARGIN.top + plotH}
            stroke="currentColor" strokeOpacity={0.3} />
          {xTicks.map((v, i) => (
            <text key={`xt${i}`} x={xScale(v)} y={MARGIN.top + plotH + 16}
              textAnchor="middle" fontSize={10} fill="currentColor" fillOpacity={0.5}>
              {v}
            </text>
          ))}
          <text x={MARGIN.left + plotW / 2} y={chartH - 2}
            textAnchor="middle" fontSize={10} fill="currentColor" fillOpacity={0.6}>
            &#951; (natural parameter)
          </text>

          {/* Y axis */}
          <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH}
            stroke="currentColor" strokeOpacity={0.3} />
          {yTicks.map((v, i) => (
            <text key={`yt${i}`} x={MARGIN.left - 6} y={yScale(v) + 3}
              textAnchor="end" fontSize={9} fill="currentColor" fillOpacity={0.5}>
              {v.toFixed(1)}
            </text>
          ))}
          <text x={12} y={MARGIN.top + plotH / 2} textAnchor="middle" fontSize={10}
            fill="currentColor" fillOpacity={0.6}
            transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}>
            A(&#951;)
          </text>
        </svg>

        {/* Readout panel */}
        <div className={`text-xs font-mono space-y-2 ${isNarrow ? 'w-full' : 'w-40'} p-3 rounded-lg border`}
          style={{ borderColor: 'var(--color-border)' }}>
          <div>
            <span className="font-semibold" style={{ color: canonicalFormColors.A }}>&#951;</span>
            {' '}= {eta.toFixed(4)}
          </div>
          <div>
            <span className="font-semibold" style={{ color: canonicalFormColors.A }}>A(&#951;)</span>
            {' '}= {isFinite(aVal) ? aVal.toFixed(4) : '\u2014'}
          </div>
          <div>
            <span className="font-semibold" style={{ color: '#16a34a' }}>A&prime;(&#951;)</span>
            {' '}= E[T(X)] = {isFinite(apVal) ? apVal.toFixed(4) : '\u2014'}
          </div>
          <div>
            <span className="font-semibold" style={{ color: '#7c3aed' }}>A&Prime;(&#951;)</span>
            {' '}= Var(T(X)) = {isFinite(appVal) ? appVal.toFixed(4) : '\u2014'}
          </div>

          {/* Legend */}
          <div className="pt-2 border-t space-y-1" style={{ borderColor: 'var(--color-border)' }}>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-0.5" style={{ background: canonicalFormColors.A }} />
              <span>A(&#951;)</span>
            </div>
            {showDerivatives && (
              <>
                <div className="flex items-center gap-1.5">
                  <span className="inline-block w-4 h-0.5 border-t-2 border-dashed" style={{ borderColor: '#16a34a' }} />
                  <span>A&prime;(&#951;)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="inline-block w-4 h-0.5 border-t-2 border-dashed" style={{ borderColor: '#7c3aed' }} />
                  <span>A&Prime;(&#951;)</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Convexity annotation */}
      <p className="text-center text-xs mt-3 italic" style={{ color: 'var(--color-text-muted, #6b7280)' }}>
        A(&#951;) is convex &#8594; log-likelihood is concave &#8594; MLE exists and is unique
      </p>
    </div>
  );
}
