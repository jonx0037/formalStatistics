import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { birthdayProbability, mcBirthday, seededRandom } from './shared/probability';

export default function BirthdayProblemExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(23);
  const [mcTrials, setMcTrials] = useState(0);
  const [mcMatches, setMcMatches] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const rngRef = useRef(seededRandom(42));
  const trialsRef = useRef(0);

  // Exact curve data
  const curveData = useMemo(() => {
    const points: { n: number; p: number }[] = [];
    for (let i = 1; i <= 80; i++) {
      points.push({ n: i, p: birthdayProbability(i) });
    }
    return points;
  }, []);

  const exactP = birthdayProbability(n);
  const mcP = mcTrials > 0 ? mcMatches / mcTrials : 0;

  // SVG dimensions — responsive: full width on mobile, half on sm+
  const chartH = Math.min(width * 0.45, 260);
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotW = Math.max(0, width < 640
    ? width - pad.left - pad.right
    : (width - pad.left - pad.right) / 2 - 16);
  const plotH = chartH - pad.top - pad.bottom;

  // Scales
  const xScale = (v: number) => pad.left + (v / 80) * plotW;
  const yScale = (v: number) => pad.top + (1 - v) * plotH;

  // Build SVG path for exact curve
  const pathD = curveData
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.n).toFixed(1)} ${yScale(d.p).toFixed(1)}`)
    .join(' ');

  // MC simulation — uses shared mcBirthday utility
  const runBatch = useCallback(() => {
    const batchSize = 200;
    const matchRate = mcBirthday(n, batchSize, rngRef.current);
    const batchMatches = Math.round(matchRate * batchSize);
    trialsRef.current += batchSize;
    setMcTrials((prev) => prev + batchSize);
    setMcMatches((prev) => prev + batchMatches);
  }, [n]);

  const startSim = useCallback(() => {
    rngRef.current = seededRandom(Date.now());
    trialsRef.current = 0;
    setMcTrials(0);
    setMcMatches(0);
    setIsRunning(true);
  }, []);

  const stopSim = useCallback(() => {
    setIsRunning(false);
  }, []);

  // Animation loop — uses ref for trial count to avoid effect churn
  useEffect(() => {
    if (!isRunning) return;
    let active = true;
    const loop = () => {
      if (!active) return;
      runBatch();
      if (trialsRef.current < 10000) {
        requestAnimationFrame(loop);
      } else {
        setIsRunning(false);
      }
    };
    requestAnimationFrame(loop);
    return () => { active = false; };
  }, [isRunning, runBatch]);

  // Reset MC when n changes
  useEffect(() => {
    setMcTrials(0);
    setMcMatches(0);
    setIsRunning(false);
    trialsRef.current = 0;
  }, [n]);

  // Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
  // X-axis ticks
  const xTicks = [0, 20, 40, 60, 80];

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Slider */}
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <label className="text-sm" style={{ color: 'var(--color-text)' }}>
          Group size <span className="font-semibold">n = {n}</span>
        </label>
        <input
          type="range" min={1} max={80} value={n}
          onChange={(e) => setN(+e.target.value)}
          className="flex-1 min-w-[120px]"
        />
      </div>

      {/* Two-panel layout */}
      <div ref={containerRef} className="w-full grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Left: Exact curve */}
        <div>
          <div className="text-xs font-semibold mb-1" style={{ color: 'var(--color-text-heading)' }}>
            Exact Probability
          </div>
          <svg width={plotW + pad.left + pad.right} height={chartH} className="w-full" viewBox={`0 0 ${plotW + pad.left + pad.right} ${chartH}`}>
            {/* Grid */}
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={pad.left} y1={yScale(t)} x2={pad.left + plotW} y2={yScale(t)}
                  stroke="var(--color-viz-grid)" strokeWidth={1} />
                <text x={pad.left - 8} y={yScale(t) + 4} textAnchor="end" fontSize={10} fill="var(--color-text-muted)">
                  {t.toFixed(2)}
                </text>
              </g>
            ))}
            {xTicks.map((t) => (
              <text key={t} x={xScale(t)} y={chartH - 8} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">
                {t}
              </text>
            ))}
            <text x={pad.left + plotW / 2} y={chartH} textAnchor="middle" fontSize={11} fill="var(--color-text-muted)">
              n (group size)
            </text>

            {/* 50% threshold line */}
            <line x1={pad.left} y1={yScale(0.5)} x2={pad.left + plotW} y2={yScale(0.5)}
              stroke="#dc2626" strokeWidth={1} strokeDasharray="4,3" opacity={0.5} />

            {/* Curve */}
            <path d={pathD} fill="none" stroke="#2563eb" strokeWidth={2} />

            {/* n=23 marker */}
            <line x1={xScale(23)} y1={yScale(0)} x2={xScale(23)} y2={yScale(birthdayProbability(23))}
              stroke="#dc2626" strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />
            <circle cx={xScale(23)} cy={yScale(birthdayProbability(23))} r={3} fill="#dc2626" />
            <text x={xScale(23) + 4} y={yScale(birthdayProbability(23)) - 6} fontSize={9} fill="#dc2626">
              n=23
            </text>

            {/* Current n marker */}
            <line x1={xScale(n)} y1={yScale(0)} x2={xScale(n)} y2={yScale(exactP)}
              stroke="#2563eb" strokeWidth={1.5} strokeDasharray="4,2" />
            <circle cx={xScale(n)} cy={yScale(exactP)} r={4} fill="#2563eb" />
          </svg>
          <div className="text-center text-sm mt-1" style={{ color: 'var(--color-text)' }}>
            P(match) = <span className="font-semibold">{exactP.toFixed(6)}</span>
          </div>
        </div>

        {/* Right: Monte Carlo */}
        <div>
          <div className="text-xs font-semibold mb-1" style={{ color: 'var(--color-text-heading)' }}>
            Monte Carlo Simulation
          </div>
          <div className="flex items-center gap-2 mb-3">
            <button
              onClick={isRunning ? stopSim : startSim}
              className="px-4 py-1.5 rounded text-sm font-medium border cursor-pointer transition-colors"
              style={{
                background: isRunning ? 'var(--color-danger)' : '#2563eb',
                color: '#ffffff',
                borderColor: isRunning ? 'var(--color-danger)' : '#2563eb',
              }}
            >
              {isRunning ? 'Stop' : 'Run simulation'}
            </button>
            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {mcTrials.toLocaleString()} trials
            </span>
          </div>

          {/* MC result bar */}
          <div className="rounded overflow-hidden mb-2" style={{ background: 'var(--color-surface)', height: 32 }}>
            <div
              className="h-full flex items-center px-2 text-xs text-white font-semibold transition-all duration-300"
              style={{
                width: `${Math.max(mcP * 100, 0)}%`,
                background: '#2563eb',
                minWidth: mcTrials > 0 ? '2px' : 0,
              }}
            >
              {mcTrials > 0 ? `${(mcP * 100).toFixed(1)}%` : ''}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div style={{ color: 'var(--color-text-muted)' }}>
              MC estimate: <span className="font-semibold" style={{ color: 'var(--color-text)' }}>
                {mcTrials > 0 ? mcP.toFixed(4) : '—'}
              </span>
            </div>
            <div style={{ color: 'var(--color-text-muted)' }}>
              Exact: <span className="font-semibold" style={{ color: 'var(--color-text)' }}>
                {exactP.toFixed(4)}
              </span>
            </div>
            {mcTrials > 0 && (
              <div className="col-span-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Error: {Math.abs(mcP - exactP).toFixed(4)}
                {mcTrials >= 1000 && Math.abs(mcP - exactP) < 0.02 && (
                  <span style={{ color: 'var(--color-accent)' }}> ✓ converging</span>
                )}
              </div>
            )}
          </div>

          {/* n=23 callout */}
          {n === 23 && (
            <div
              className="mt-3 p-2 rounded text-xs border"
              style={{
                background: 'var(--color-definition-bg)',
                borderColor: 'var(--color-definition-border)',
                color: 'var(--color-text)',
              }}
            >
              With just 23 people, the probability of a shared birthday exceeds 50%!
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
