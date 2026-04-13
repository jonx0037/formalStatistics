import { useState, useRef, useEffect, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { typewriterInterval } from './shared/convergence';

// ── Constants ──────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 50 };
const ACTIVE_COLOR = '#DC2626';
const SAFE_COLOR = '#059669';
const BAR_COLOR_ZERO = '#6B7280';
const BAR_COLOR_NONZERO = '#7C3AED';
const EXPECTATION_COLOR = '#D97706';
const PROB_COLOR = '#2563EB';

type TabId = 'typewriter' | 'escape' | 'dist-expectation';

interface Tab {
  id: TabId;
  label: string;
  shortLabel: string;
}

const TABS: Tab[] = [
  { id: 'typewriter', label: 'Typewriter sequence', shortLabel: 'Typewriter' },
  { id: 'escape', label: 'Escape to infinity', shortLabel: 'Escape' },
  { id: 'dist-expectation', label: 'Dist \u219B expectations', shortLabel: 'Dist \u219B E[X]' },
];

// ── Component ──────────────────────────────────────────────────────────────────

export default function CounterexampleExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [activeTab, setActiveTab] = useState<TabId>('typewriter');

  // Responsive layout
  const containerW = Math.max(300, (width || 600) - 16);
  const chartH = 260;

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Convergence Counterexamples
      </div>

      {/* Tab selector */}
      <div className="flex justify-center gap-1 mb-4">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-white'
                : 'hover:opacity-80'
            }`}
            style={{
              background: activeTab === tab.id ? '#7C3AED' : 'var(--color-bg)',
              color: activeTab === tab.id ? '#ffffff' : 'var(--color-text-muted)',
              border: `1px solid ${activeTab === tab.id ? '#7C3AED' : 'var(--color-border)'}`,
            }}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.shortLabel}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'typewriter' && (
        <TypewriterTab containerW={containerW} chartH={chartH} />
      )}
      {activeTab === 'escape' && (
        <EscapeTab containerW={containerW} chartH={chartH} />
      )}
      {activeTab === 'dist-expectation' && (
        <DistExpectationTab containerW={containerW} chartH={chartH} />
      )}
    </div>
  );
}

// ── Tab 1: Typewriter sequence ─────────────────────────────────────────────────

interface TypewriterState {
  u: number;
  step: number;
  hitCount: number;
  isPlaying: boolean;
  history: boolean[]; // true = hit at step i
}

function TypewriterTab({ containerW, chartH }: { containerW: number; chartH: number }) {
  const [state, setState] = useState<TypewriterState>(() => ({
    u: Math.random(),
    step: 0,
    hitCount: 0,
    isPlaying: false,
    history: [],
  }));
  const [speed, setSpeed] = useState(200);

  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const speedRef = useRef<number>(speed);
  const stateRef = useRef(state);

  // Keep refs synced
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { stateRef.current = state; }, [state]);

  const advance = useCallback(() => {
    setState((prev) => {
      const nextStep = prev.step + 1;
      if (nextStep > 500) return prev; // Cap at 500 steps
      const { a, b } = typewriterInterval(nextStep);
      const hit = prev.u >= a && prev.u < b;
      return {
        ...prev,
        step: nextStep,
        hitCount: prev.hitCount + (hit ? 1 : 0),
        history: [...prev.history, hit],
      };
    });
  }, []);

  const animate = useCallback((timestamp: number) => {
    if (timestamp - lastTimeRef.current >= speedRef.current) {
      advance();
      lastTimeRef.current = timestamp;
    }
    animRef.current = requestAnimationFrame(animate);
  }, [advance]);

  const play = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: true }));
    lastTimeRef.current = performance.now();
    animRef.current = requestAnimationFrame(animate);
  }, [animate]);

  const pause = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: false }));
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    pause();
    setState({
      u: Math.random(),
      step: 0,
      hitCount: 0,
      isPlaying: false,
      history: [],
    });
  }, [pause]);

  // Cleanup on unmount or tab switch
  useEffect(() => {
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Current interval
  const currentInterval = state.step > 0 ? typewriterInterval(state.step) : null;
  const isHit = currentInterval
    ? (state.u >= currentInterval.a && state.u < currentInterval.b)
    : false;

  // Row k for current step
  const currentK = state.step > 0
    ? Math.ceil((-1 + Math.sqrt(1 + 8 * (state.step - 1))) / 2)
    : 0;

  // Bar height of the interval [0,1]
  const barY = 60;
  const barH = 30;
  const barLeft = MARGIN.left;
  const barRight = containerW - MARGIN.right;
  const barW = barRight - barLeft;

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-center mb-3">
        <button
          className="rounded border px-3 py-1 text-xs font-medium"
          style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
          onClick={state.isPlaying ? pause : play}
        >
          {state.isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          className="rounded border px-3 py-1 text-xs font-medium"
          style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
          onClick={reset}
        >
          Reset
        </button>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Speed</span>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-24"
          />
          <span className="font-mono text-[10px]">{speed}ms</span>
        </label>
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-4 justify-center text-xs mb-3">
        <span>
          Step n = <span className="font-mono font-semibold">{state.step}</span>
        </span>
        <span>
          U = <span className="font-mono font-semibold">{state.u.toFixed(4)}</span>
        </span>
        <span>
          Times X_n(U) = 1: <span className="font-mono font-semibold">{state.hitCount}</span>
        </span>
        <span>
          Row k = <span className="font-mono font-semibold">{currentK}</span>
          {currentK > 0 && (
            <span style={{ color: 'var(--color-text-muted)' }}>
              {' '}(interval width 1/{currentK})
            </span>
          )}
        </span>
      </div>

      {/* SVG: unit interval */}
      <svg width={containerW} height={chartH} className="block mx-auto">
        {/* Axis label */}
        <text
          x={containerW / 2}
          y={barY - 30}
          textAnchor="middle"
          fontSize={11}
          fontWeight={600}
          fill="var(--color-text)"
        >
          Unit interval [0, 1]
        </text>

        {/* Background bar */}
        <rect
          x={barLeft} y={barY}
          width={barW} height={barH}
          fill="var(--color-bg)"
          stroke="var(--color-border)"
          strokeWidth={1}
          rx={3}
        />

        {/* Current highlighted interval */}
        {currentInterval && (
          <rect
            x={barLeft + currentInterval.a * barW}
            y={barY}
            width={(currentInterval.b - currentInterval.a) * barW}
            height={barH}
            fill={isHit ? ACTIVE_COLOR : '#BFDBFE'}
            opacity={isHit ? 0.5 : 0.3}
            rx={2}
          />
        )}

        {/* Interval bracket labels */}
        {currentInterval && (
          <>
            <text
              x={barLeft + currentInterval.a * barW}
              y={barY + barH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="var(--color-text-muted)"
            >
              {currentInterval.a.toFixed(3)}
            </text>
            <text
              x={barLeft + currentInterval.b * barW}
              y={barY + barH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="var(--color-text-muted)"
            >
              {currentInterval.b.toFixed(3)}
            </text>
          </>
        )}

        {/* U dot */}
        <circle
          cx={barLeft + state.u * barW}
          cy={barY + barH / 2}
          r={6}
          fill={isHit ? ACTIVE_COLOR : SAFE_COLOR}
          stroke="#ffffff"
          strokeWidth={1.5}
        />
        <text
          x={barLeft + state.u * barW}
          y={barY - 6}
          textAnchor="middle"
          fontSize={9}
          fontWeight={600}
          fill={isHit ? ACTIVE_COLOR : SAFE_COLOR}
        >
          U
        </text>

        {/* 0 and 1 labels */}
        <text x={barLeft} y={barY + barH + 24} textAnchor="middle" fontSize={9} fill="var(--color-text-muted)">0</text>
        <text x={barRight} y={barY + barH + 24} textAnchor="middle" fontSize={9} fill="var(--color-text-muted)">1</text>

        {/* Hit indicator */}
        <text
          x={containerW / 2}
          y={barY + barH + 44}
          textAnchor="middle"
          fontSize={12}
          fontWeight={600}
          fill={isHit ? ACTIVE_COLOR : SAFE_COLOR}
        >
          {state.step === 0 ? 'Press Play to start' : (isHit ? 'X_n(U) = 1 (HIT!)' : 'X_n(U) = 0')}
        </text>

        {/* Recent hit-rate visualization */}
        {state.history.length > 0 && (() => {
          const dotSize = 4;
          const dotsPerRow = Math.floor((barW - 20) / (dotSize + 2));
          const lastN = state.history.slice(-Math.min(state.history.length, dotsPerRow));
          const dotY = barY + barH + 64;
          return (
            <g>
              <text
                x={barLeft}
                y={dotY - 6}
                fontSize={9}
                fill="var(--color-text-muted)"
              >
                Recent steps (red = hit):
              </text>
              {lastN.map((hit, i) => (
                <rect
                  key={i}
                  x={barLeft + i * (dotSize + 2)}
                  y={dotY}
                  width={dotSize}
                  height={dotSize}
                  rx={1}
                  fill={hit ? ACTIVE_COLOR : '#d1d5db'}
                />
              ))}
            </g>
          );
        })()}

        {/* Convergence summary */}
        <text
          x={containerW / 2}
          y={chartH - 20}
          textAnchor="middle"
          fontSize={10}
          fill="var(--color-text-muted)"
        >
          P(X_n {'\u2260'} 0) = 1/k {'\u2192'} 0 (convergence in probability {'\u2713'})
        </text>
        <text
          x={containerW / 2}
          y={chartH - 6}
          textAnchor="middle"
          fontSize={10}
          fill={ACTIVE_COLOR}
        >
          But X_n(U) = 1 infinitely often for every U (NOT a.s. convergence)
        </text>
      </svg>
    </div>
  );
}

// ── Tab 2: Escape to infinity ──────────────────────────────────────────────────

function EscapeTab({ containerW, chartH: baseH }: { containerW: number; chartH: number }) {
  const [n, setN] = useState(10);

  const chartH = baseH + 40;
  const plotW = containerW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // X_n: P(X_n = 0) = 1 - 1/n, P(X_n = n) = 1/n
  const pZero = 1 - 1 / n;
  const pN = 1 / n;
  const eXn = 1; // E[X_n] = n * (1/n) = 1 always

  // For the bar chart, we show two bars side by side
  const barW = Math.min(plotW * 0.15, 50);
  const barGap = barW * 0.5;
  const centerX = MARGIN.left + plotW / 2;

  // Y scale: probability on left bars
  const maxProb = 1.05;
  const toSvgY = (p: number) => MARGIN.top + plotH - (p / maxProb) * plotH;

  // For the E[Xn] and P(Xn != 0) curves, use a separate right-side chart
  const nValues = Array.from({ length: 100 }, (_, i) => i + 1);

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n = {n}</span>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <svg width={containerW} height={chartH} className="block mx-auto">
        {/* Distribution bars */}
        <text
          x={containerW / 2}
          y={MARGIN.top + 2}
          textAnchor="middle"
          fontSize={11}
          fontWeight={600}
          fill="var(--color-text)"
        >
          Distribution of X_{n} (n = {n})
        </text>

        {/* Y-axis grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = toSvgY(v);
          return (
            <g key={`ey-${v}`}>
              <line
                x1={MARGIN.left} x2={MARGIN.left + plotW}
                y1={y} y2={y}
                stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
              />
              <text
                x={MARGIN.left - 6} y={y + 3}
                textAnchor="end" fontSize={9} fill="var(--color-text-muted)"
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Axes */}
        <line
          x1={MARGIN.left} x2={MARGIN.left + plotW}
          y1={MARGIN.top + plotH} y2={MARGIN.top + plotH}
          stroke="var(--color-text-muted)" strokeWidth={1}
        />
        <line
          x1={MARGIN.left} x2={MARGIN.left}
          y1={MARGIN.top + 14} y2={MARGIN.top + plotH}
          stroke="var(--color-text-muted)" strokeWidth={1}
        />

        {/* P(X_n = 0) bar */}
        {(() => {
          const h = (pZero / maxProb) * plotH;
          const x = centerX - barGap / 2 - barW;
          return (
            <g>
              <rect
                x={x} y={toSvgY(pZero)}
                width={barW} height={h}
                fill={BAR_COLOR_ZERO}
                opacity={0.6}
                rx={2}
              />
              <text
                x={x + barW / 2} y={toSvgY(pZero) - 6}
                textAnchor="middle" fontSize={9} fontWeight={600}
                fill={BAR_COLOR_ZERO}
              >
                {pZero.toFixed(4)}
              </text>
              <text
                x={x + barW / 2} y={MARGIN.top + plotH + 14}
                textAnchor="middle" fontSize={10}
                fill="var(--color-text)"
              >
                x = 0
              </text>
            </g>
          );
        })()}

        {/* P(X_n = n) bar */}
        {(() => {
          const h = Math.max((pN / maxProb) * plotH, 2);
          const x = centerX + barGap / 2;
          return (
            <g>
              <rect
                x={x} y={MARGIN.top + plotH - h}
                width={barW} height={h}
                fill={BAR_COLOR_NONZERO}
                opacity={0.7}
                rx={2}
              />
              <text
                x={x + barW / 2} y={MARGIN.top + plotH - h - 6}
                textAnchor="middle" fontSize={9} fontWeight={600}
                fill={BAR_COLOR_NONZERO}
              >
                {pN.toFixed(4)}
              </text>
              <text
                x={x + barW / 2} y={MARGIN.top + plotH + 14}
                textAnchor="middle" fontSize={10}
                fill="var(--color-text)"
              >
                x = {n}
              </text>
            </g>
          );
        })()}

        {/* E[Xn] = 1 dashed line */}
        <line
          x1={MARGIN.left + 4} x2={MARGIN.left + plotW - 4}
          y1={toSvgY(eXn / n)} y2={toSvgY(eXn / n)}
          stroke={EXPECTATION_COLOR}
          strokeWidth={0}
          strokeDasharray="6,3"
        />

        {/* P(Xn != 0) curve (small inset in top-right) */}
        {(() => {
          const insetW = Math.min(plotW * 0.35, 160);
          const insetH = 70;
          const insetX = MARGIN.left + plotW - insetW - 4;
          const insetY = MARGIN.top + 18;
          const maxN = 100;
          const toIx = (ni: number) => insetX + ((ni - 1) / (maxN - 1)) * insetW;
          const toIy = (v: number) => insetY + insetH - v * insetH;

          // P(Xn != 0) = 1/n
          const probPath = nValues.map((ni, i) => {
            const px = toIx(ni);
            const py = toIy(1 / ni);
            return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
          }).join(' ');

          // E[Xn] = 1 (constant, but scale to 1 on this axis)
          const ePath = `M${toIx(1).toFixed(1)},${toIy(1).toFixed(1)} L${toIx(maxN).toFixed(1)},${toIy(1).toFixed(1)}`;

          return (
            <g>
              {/* Inset background */}
              <rect
                x={insetX - 4} y={insetY - 14}
                width={insetW + 8} height={insetH + 28}
                fill="var(--color-bg)"
                stroke="var(--color-border)"
                strokeWidth={0.5}
                rx={4}
                opacity={0.95}
              />
              <text
                x={insetX + insetW / 2} y={insetY - 3}
                textAnchor="middle" fontSize={8} fontWeight={600}
                fill="var(--color-text-muted)"
              >
                As n grows
              </text>

              {/* P(Xn != 0) curve */}
              <path d={probPath} fill="none" stroke={PROB_COLOR} strokeWidth={1.5} />

              {/* E[Xn] = 1 line */}
              <path d={ePath} fill="none" stroke={EXPECTATION_COLOR} strokeWidth={1.5} strokeDasharray="4,2" />

              {/* Current n marker */}
              <circle
                cx={toIx(n)} cy={toIy(1 / n)}
                r={3} fill={PROB_COLOR}
              />

              {/* Labels */}
              <text
                x={insetX + insetW + 2} y={toIy(1 / Math.max(n, 2)) + 3}
                fontSize={7} fill={PROB_COLOR}
              >
                P
              </text>
              <text
                x={insetX + insetW + 2} y={toIy(1) + 3}
                fontSize={7} fill={EXPECTATION_COLOR}
              >
                E
              </text>

              {/* Axis labels */}
              <text
                x={insetX} y={insetY + insetH + 10}
                fontSize={7} fill="var(--color-text-muted)"
              >
                1
              </text>
              <text
                x={insetX + insetW} y={insetY + insetH + 10}
                textAnchor="end" fontSize={7} fill="var(--color-text-muted)"
              >
                {maxN}
              </text>
            </g>
          );
        })()}

        {/* Annotation */}
        <text
          x={containerW / 2}
          y={chartH - 18}
          textAnchor="middle"
          fontSize={10}
          fill="var(--color-text-muted)"
        >
          Converges in probability {'\u2713'} (P(X_n {'\u2260'} 0) = 1/n {'\u2192'} 0)
        </text>
        <text
          x={containerW / 2}
          y={chartH - 4}
          textAnchor="middle"
          fontSize={10}
          fill={ACTIVE_COLOR}
        >
          But E[X_n] = n {'\u00B7'} (1/n) = 1 always {'\u2014'} NOT L{'\u00B9'} convergence
        </text>
      </svg>
    </div>
  );
}

// ── Tab 3: Dist does not imply expectations ────────────────────────────────────

function DistExpectationTab({ containerW, chartH: baseH }: { containerW: number; chartH: number }) {
  const [n, setN] = useState(5);

  const chartH = baseH + 40;
  const plotW = containerW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // X_n: P(X_n = 0) = 1 - 1/n, P(X_n = n^2) = 1/n
  const pZero = 1 - 1 / n;
  const pN2 = 1 / n;

  // Bar chart
  const barW = Math.min(plotW * 0.15, 50);
  const barGap = barW * 0.5;
  const centerX = MARGIN.left + plotW / 2;

  const maxProb = 1.05;
  const toSvgY = (p: number) => MARGIN.top + plotH - (p / maxProb) * plotH;

  const nValues = Array.from({ length: 50 }, (_, i) => i + 1);

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center items-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n = {n}</span>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <svg width={containerW} height={chartH} className="block mx-auto">
        {/* Title */}
        <text
          x={containerW / 2}
          y={MARGIN.top + 2}
          textAnchor="middle"
          fontSize={11}
          fontWeight={600}
          fill="var(--color-text)"
        >
          Distribution of X_{n} (n = {n})
        </text>

        {/* Y-axis grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = toSvgY(v);
          return (
            <g key={`dy-${v}`}>
              <line
                x1={MARGIN.left} x2={MARGIN.left + plotW}
                y1={y} y2={y}
                stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2,3"
              />
              <text
                x={MARGIN.left - 6} y={y + 3}
                textAnchor="end" fontSize={9} fill="var(--color-text-muted)"
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Axes */}
        <line
          x1={MARGIN.left} x2={MARGIN.left + plotW}
          y1={MARGIN.top + plotH} y2={MARGIN.top + plotH}
          stroke="var(--color-text-muted)" strokeWidth={1}
        />
        <line
          x1={MARGIN.left} x2={MARGIN.left}
          y1={MARGIN.top + 14} y2={MARGIN.top + plotH}
          stroke="var(--color-text-muted)" strokeWidth={1}
        />

        {/* P(X_n = 0) bar */}
        {(() => {
          const h = (pZero / maxProb) * plotH;
          const x = centerX - barGap / 2 - barW;
          return (
            <g>
              <rect
                x={x} y={toSvgY(pZero)}
                width={barW} height={h}
                fill={BAR_COLOR_ZERO}
                opacity={0.6}
                rx={2}
              />
              <text
                x={x + barW / 2} y={toSvgY(pZero) - 6}
                textAnchor="middle" fontSize={9} fontWeight={600}
                fill={BAR_COLOR_ZERO}
              >
                {pZero.toFixed(4)}
              </text>
              <text
                x={x + barW / 2} y={MARGIN.top + plotH + 14}
                textAnchor="middle" fontSize={10}
                fill="var(--color-text)"
              >
                x = 0
              </text>
            </g>
          );
        })()}

        {/* P(X_n = n^2) bar */}
        {(() => {
          const h = Math.max((pN2 / maxProb) * plotH, 2);
          const x = centerX + barGap / 2;
          return (
            <g>
              <rect
                x={x} y={MARGIN.top + plotH - h}
                width={barW} height={h}
                fill={BAR_COLOR_NONZERO}
                opacity={0.7}
                rx={2}
              />
              <text
                x={x + barW / 2} y={MARGIN.top + plotH - h - 6}
                textAnchor="middle" fontSize={9} fontWeight={600}
                fill={BAR_COLOR_NONZERO}
              >
                {pN2.toFixed(4)}
              </text>
              <text
                x={x + barW / 2} y={MARGIN.top + plotH + 14}
                textAnchor="middle" fontSize={10}
                fill="var(--color-text)"
              >
                x = {n * n}
              </text>
            </g>
          );
        })()}

        {/* E[Xn] inset: grows with n */}
        {(() => {
          const insetW = Math.min(plotW * 0.35, 160);
          const insetH = 70;
          const insetX = MARGIN.left + plotW - insetW - 4;
          const insetY = MARGIN.top + 18;
          const maxN = 50;
          const maxE = 50; // E[Xn] = n, so max is 50
          const toIx = (ni: number) => insetX + ((ni - 1) / (maxN - 1)) * insetW;
          const toIy = (v: number) => insetY + insetH - (v / maxE) * insetH;

          // E[Xn] = n (growing line)
          const ePath = nValues.map((ni, i) => {
            const px = toIx(ni);
            const py = toIy(ni);
            return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
          }).join(' ');

          // P(Xn != 0) = 1/n (decaying)
          const pPath = nValues.map((ni, i) => {
            const px = toIx(ni);
            const py = toIy((1 / ni) * maxE); // scale to fit
            return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
          }).join(' ');

          return (
            <g>
              {/* Inset background */}
              <rect
                x={insetX - 4} y={insetY - 14}
                width={insetW + 8} height={insetH + 28}
                fill="var(--color-bg)"
                stroke="var(--color-border)"
                strokeWidth={0.5}
                rx={4}
                opacity={0.95}
              />
              <text
                x={insetX + insetW / 2} y={insetY - 3}
                textAnchor="middle" fontSize={8} fontWeight={600}
                fill="var(--color-text-muted)"
              >
                As n grows
              </text>

              {/* E[Xn] = n line */}
              <path d={ePath} fill="none" stroke={EXPECTATION_COLOR} strokeWidth={1.5} />

              {/* P(Xn != 0) scaled */}
              <path d={pPath} fill="none" stroke={PROB_COLOR} strokeWidth={1.5} strokeDasharray="4,2" />

              {/* Current n marker on E[Xn] */}
              <circle
                cx={toIx(n)} cy={toIy(n)}
                r={3} fill={EXPECTATION_COLOR}
              />

              {/* Labels */}
              <text
                x={insetX + insetW + 2} y={toIy(maxN) + 3}
                fontSize={7} fill={EXPECTATION_COLOR}
              >
                E[X_n]
              </text>
              <text
                x={insetX + insetW + 2} y={insetY + insetH - 2}
                fontSize={7} fill={PROB_COLOR}
              >
                P(X_n{'\u2260'}0)
              </text>

              {/* Axis labels */}
              <text
                x={insetX} y={insetY + insetH + 10}
                fontSize={7} fill="var(--color-text-muted)"
              >
                1
              </text>
              <text
                x={insetX + insetW} y={insetY + insetH + 10}
                textAnchor="end" fontSize={7} fill="var(--color-text-muted)"
              >
                {maxN}
              </text>
            </g>
          );
        })()}

        {/* Annotation */}
        <text
          x={containerW / 2}
          y={chartH - 18}
          textAnchor="middle"
          fontSize={10}
          fill="var(--color-text-muted)"
        >
          X_n {'\u2192'}_d 0 (mass concentrates at 0 as n {'\u2192'} {'\u221E'})
        </text>
        <text
          x={containerW / 2}
          y={chartH - 4}
          textAnchor="middle"
          fontSize={10}
          fill={ACTIVE_COLOR}
        >
          But E[X_n] = n{'\u00B2'} {'\u00B7'} (1/n) = n {'\u2192'} {'\u221E'} {'\u2014'} convergence in distribution does NOT preserve expectations
        </text>
      </svg>
    </div>
  );
}
