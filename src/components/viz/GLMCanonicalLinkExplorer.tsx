import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 200;

// ── Deterministic LCG PRNG ────────────────────────────────────────────────

function createPRNG(initial: number) {
  let seed = initial;
  return () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return Math.max((seed >>> 0) / 0x100000000, Number.EPSILON);
  };
}

// Box-Muller transform for Normal samples
function normalSample(nextRand: () => number, mu: number, sigma: number): number {
  const u1 = nextRand();
  const u2 = nextRand();
  return mu + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Poisson via inverse CDF (Knuth)
function poissonSample(nextRand: () => number, lambda: number): number {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= nextRand();
  } while (p > L);
  return k - 1;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// ── Tab configuration ─────────────────────────────────────────────────────

interface TabConfig {
  label: string;
  color: string;
  activeClass: string;
  annotation: string;
  linkName: string;
  varianceFunc: string;
}

const TAB_CONFIGS: TabConfig[] = [
  {
    label: 'Normal / Identity',
    color: '#2563eb',
    activeClass: 'bg-blue-600 text-white',
    annotation: '\u03B7 = \u03B2\u2080 + \u03B2\u2081x  \u2192  \u03BC = \u03B7 (identity)  \u2192  Y ~ N(\u03BC, \u03C3\u00B2)',
    linkName: 'g(\u03BC) = \u03BC  (identity)',
    varianceFunc: 'V(\u03BC) = \u03C3\u00B2  (constant)',
  },
  {
    label: 'Bernoulli / Logit',
    color: '#ea580c',
    activeClass: 'bg-orange-600 text-white',
    annotation: '\u03B7 = \u03B2\u2080 + \u03B2\u2081x  \u2192  \u03BC = 1/(1+e\u207B\u1D51)  (sigmoid)  \u2192  Y ~ Bernoulli(\u03BC)',
    linkName: 'g(\u03BC) = log(\u03BC/(1\u2212\u03BC))  (logit)',
    varianceFunc: 'V(\u03BC) = \u03BC(1\u2212\u03BC)',
  },
  {
    label: 'Poisson / Log',
    color: '#16a34a',
    activeClass: 'bg-green-600 text-white',
    annotation: '\u03B7 = \u03B2\u2080 + \u03B2\u2081x  \u2192  \u03BC = e\u1D51  (exp)  \u2192  Y ~ Poisson(\u03BC)',
    linkName: 'g(\u03BC) = log(\u03BC)  (log)',
    varianceFunc: 'V(\u03BC) = \u03BC',
  },
];

// ── Component ─────────────────────────────────────────────────────────────

export default function GLMCanonicalLinkExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const [activeTab, setActiveTab] = useState<0 | 1 | 2>(0);

  // Per-tab parameters (state preserved across tab switches)
  const [beta0N, setBeta0N] = useState(1);
  const [beta1N, setBeta1N] = useState(0.8);
  const [beta0B, setBeta0B] = useState(0);
  const [beta1B, setBeta1B] = useState(1.5);
  const [beta0P, setBeta0P] = useState(0.5);
  const [beta1P, setBeta1P] = useState(0.5);

  // Generate synthetic data ONCE (stable seed, no deps)
  const datasets = useMemo(() => {
    const rng0 = createPRNG(42);
    const normalData: { x: number; y: number }[] = [];
    for (let i = 0; i < 20; i++) {
      const x = (i / 19) * 5;
      const y = 1 + 0.8 * x + normalSample(rng0, 0, 0.5);
      normalData.push({ x, y });
    }

    const rng1 = createPRNG(137);
    const bernoulliData: { x: number; y: number }[] = [];
    for (let i = 0; i < 40; i++) {
      const x = -3 + (i / 39) * 6;
      const p = sigmoid(0 + 1.5 * x);
      const y = rng1() < p ? 1 : 0;
      bernoulliData.push({ x, y });
    }

    const rng2 = createPRNG(271);
    const poissonData: { x: number; y: number }[] = [];
    for (let i = 0; i < 20; i++) {
      const x = (i / 19) * 3;
      const lambda = Math.exp(0.5 + 0.5 * x);
      poissonData.push({ x, y: poissonSample(rng2, lambda) });
    }

    return [normalData, bernoulliData, poissonData] as const;
  }, []);

  // Current beta values
  const betas: [number, number][] = [
    [beta0N, beta1N],
    [beta0B, beta1B],
    [beta0P, beta1P],
  ];
  const [b0, b1] = betas[activeTab];
  const data = datasets[activeTab];
  const cfg = TAB_CONFIGS[activeTab];

  // Slider setters dispatch
  const setB0 = [setBeta0N, setBeta0B, setBeta0P][activeTab];
  const setB1 = [setBeta1N, setBeta1B, setBeta1P][activeTab];

  // Slider ranges per tab
  const sliderRanges: { b0: [number, number]; b1: [number, number] }[] = [
    { b0: [-3, 5], b1: [-2, 3] },
    { b0: [-3, 3], b1: [0.1, 5] },
    { b0: [-1, 2], b1: [0.1, 2] },
  ];
  const range = sliderRanges[activeTab];

  // Layout
  const containerW = Math.max(280, (width || 600) - 16);
  const chartW = containerW;
  const chartH = 240;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // X domain per tab
  const xDomain: [number, number] = activeTab === 0
    ? [-0.3, 5.3]
    : activeTab === 1
      ? [-3.5, 3.5]
      : [-0.3, 3.3];

  // Y domain per tab (depends on data + current fit)
  const yDomain: [number, number] = useMemo(() => {
    if (activeTab === 1) return [-0.1, 1.1];
    let maxY = 0;
    for (const d of data) maxY = Math.max(maxY, d.y);
    // Also check fitted curve extremes
    const [xMin, xMax] = xDomain;
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = xMin + (i / NUM_POINTS) * (xMax - xMin);
      let fy: number;
      if (activeTab === 0) fy = b0 + b1 * x;
      else fy = Math.exp(b0 + b1 * x);
      maxY = Math.max(maxY, fy);
    }
    return [Math.min(0, ...data.map((d) => d.y)) - 0.5, maxY * 1.15];
  }, [activeTab, data, b0, b1, xDomain]);

  const [yMin, yMax] = yDomain;
  const [xMin, xMax] = xDomain;
  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;

  const toSvgX = useCallback(
    (x: number) => MARGIN.left + ((x - xMin) / xSpan) * plotW,
    [xMin, xSpan, plotW],
  );
  const toSvgY = useCallback(
    (y: number) => MARGIN.top + plotH - ((y - yMin) / ySpan) * plotH,
    [yMin, ySpan, plotH],
  );

  // Fitted curve path
  const curvePath = useMemo(() => {
    const parts: string[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = xMin + (i / NUM_POINTS) * xSpan;
      let y: number;
      if (activeTab === 0) y = b0 + b1 * x;
      else if (activeTab === 1) y = sigmoid(b0 + b1 * x);
      else y = Math.exp(b0 + b1 * x);
      const px = toSvgX(x);
      const py = toSvgY(y);
      parts.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`);
    }
    return parts.join(' ');
  }, [activeTab, b0, b1, xMin, xSpan, toSvgX, toSvgY]);

  // Grid ticks
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const nice = [0.5, 1, 2, 5];
    let step = 1;
    for (const n of nice) {
      if (xSpan / n <= 8) { step = n; break; }
    }
    const start = Math.ceil(xMin / step) * step;
    for (let v = start; v <= xMax; v += step) {
      ticks.push(Math.round(v * 100) / 100);
    }
    return ticks;
  }, [xMin, xMax, xSpan]);

  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    let step = 1;
    for (const n of nice) {
      if (ySpan / n <= 6) { step = n; break; }
    }
    const start = Math.ceil(yMin / step) * step;
    for (let v = start; v <= yMax; v += step) {
      ticks.push(Math.round(v * 100) / 100);
    }
    return ticks;
  }, [yMin, yMax, ySpan]);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: GLM Canonical Link Explorer
      </div>

      {/* Tab buttons */}
      <div className="flex flex-wrap gap-1.5 justify-center mb-4">
        {TAB_CONFIGS.map((t, i) => (
          <button
            key={i}
            onClick={() => setActiveTab(i as 0 | 1 | 2)}
            className={`rounded-md px-3 py-1 text-sm font-medium transition-colors ${
              activeTab === i ? t.activeClass : 'bg-gray-100 text-gray-600'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">{'\u03B2\u2080'}</span>
          <input
            type="range"
            min={range.b0[0]}
            max={range.b0[1]}
            step={0.1}
            value={b0}
            onChange={(e) => setB0(Number(e.target.value))}
            className="w-28"
          />
          <span className="w-10 tabular-nums text-right">{b0.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">{'\u03B2\u2081'}</span>
          <input
            type="range"
            min={range.b1[0]}
            max={range.b1[1]}
            step={0.1}
            value={b1}
            onChange={(e) => setB1(Number(e.target.value))}
            className="w-28"
          />
          <span className="w-10 tabular-nums text-right">{b1.toFixed(1)}</span>
        </label>
      </div>

      {/* SVG chart */}
      <svg width={chartW} height={chartH} className="block mx-auto">
        {/* Grid lines */}
        {yTicks.map((v, i) => (
          <line
            key={`yg-${i}`}
            x1={MARGIN.left}
            y1={toSvgY(v)}
            x2={chartW - MARGIN.right}
            y2={toSvgY(v)}
            stroke="currentColor"
            strokeOpacity={0.08}
          />
        ))}
        {xTicks.map((v, i) => (
          <line
            key={`xg-${i}`}
            x1={toSvgX(v)}
            y1={MARGIN.top}
            x2={toSvgX(v)}
            y2={MARGIN.top + plotH}
            stroke="currentColor"
            strokeOpacity={0.08}
          />
        ))}

        {/* Decision boundary for Bernoulli tab */}
        {activeTab === 1 && (
          <line
            x1={MARGIN.left}
            y1={toSvgY(0.5)}
            x2={chartW - MARGIN.right}
            y2={toSvgY(0.5)}
            stroke="#9ca3af"
            strokeWidth={1}
            strokeDasharray="4,4"
            opacity={0.6}
          />
        )}

        {/* Data points */}
        {data.map((d, i) => {
          let fill: string;
          if (activeTab === 1) {
            fill = d.y === 1 ? '#ea580c' : '#9ca3af';
          } else if (activeTab === 2) {
            fill = '#16a34a';
          } else {
            fill = '#2563eb';
          }
          return (
            <circle
              key={i}
              cx={toSvgX(d.x)}
              cy={toSvgY(d.y)}
              r={3.5}
              fill={fill}
              fillOpacity={0.6}
            />
          );
        })}

        {/* Fitted curve */}
        <path
          d={curvePath}
          fill="none"
          stroke={cfg.color}
          strokeWidth={2.5}
        />

        {/* Axes */}
        <line
          x1={MARGIN.left}
          y1={MARGIN.top + plotH}
          x2={chartW - MARGIN.right}
          y2={MARGIN.top + plotH}
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        <line
          x1={MARGIN.left}
          y1={MARGIN.top}
          x2={MARGIN.left}
          y2={MARGIN.top + plotH}
          stroke="currentColor"
          strokeOpacity={0.3}
        />

        {/* X-axis labels */}
        {xTicks.map((v, i) => (
          <text
            key={`xl-${i}`}
            x={toSvgX(v)}
            y={MARGIN.top + plotH + 14}
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            opacity={0.6}
          >
            {Number.isInteger(v) ? v : v.toFixed(1)}
          </text>
        ))}

        {/* Y-axis labels */}
        {yTicks.map((v, i) => (
          <text
            key={`yl-${i}`}
            x={MARGIN.left - 6}
            y={toSvgY(v) + 3}
            textAnchor="end"
            fontSize={9}
            fill="currentColor"
            opacity={0.6}
          >
            {Number.isInteger(v) ? v : v.toFixed(1)}
          </text>
        ))}

        {/* Axis labels */}
        <text
          x={MARGIN.left + plotW / 2}
          y={chartH - 2}
          textAnchor="middle"
          fontSize={10}
          fill="currentColor"
          opacity={0.5}
        >
          x
        </text>
        <text
          x={12}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          fontSize={10}
          fill="currentColor"
          opacity={0.5}
          transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
        >
          y
        </text>
      </svg>

      {/* Bottom annotation */}
      <div className="mt-3 pt-3 border-t text-xs text-center space-y-1" style={{ borderColor: 'var(--color-border)' }}>
        <div className="font-medium" style={{ color: cfg.color }}>
          {cfg.annotation}
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 justify-center opacity-70">
          <span>Link: {cfg.linkName}</span>
          <span>Variance: {cfg.varianceFunc}</span>
        </div>
      </div>
    </div>
  );
}
