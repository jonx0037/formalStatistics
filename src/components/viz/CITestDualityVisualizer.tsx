import { useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  standardNormalInvCDF,
  wilsonInterval,
  scoreCI,
} from './shared/testing';
import { dualityPresets, type DualityPreset } from '../../data/confidence-intervals-data';

/**
 * CITestDualityVisualizer — the featured component for Topic 19 §19.2.
 *
 * Two-plane diagram of the test–CI duality. The shaded (θ₀, T)-region is the
 * joint acceptance set {(θ₀, T) : the level-α test of H₀: θ = θ₀ at data T
 * does not reject}. Two draggable lines generate the two slicings:
 *
 *   • Horizontal slice at observed T_obs → CI C(T_obs) on the θ₀-axis.
 *   • Vertical slice at null θ₀ → acceptance region A(θ₀) on the T-axis.
 *
 * Scenarios: Normal mean σ known (cleanest), Bernoulli proportion, Poisson
 * rate. Mobile-responsive via a stacked layout below ~640 px.
 */

const MARGIN = { top: 16, right: 16, bottom: 44, left: 52 };
const H = 360;

const C = {
  accept: '#E0E7FF',
  boundary: '#475569',
  pivot: '#2563EB',
  ci: '#DC2626',
  accRegion: '#10B981',
  axis: '#64748B',
  grid: '#E2E8F0',
  text: '#1E293B',
};

interface Range {
  lo: number;
  hi: number;
}

function paramRange(preset: DualityPreset): Range {
  if (preset.family === 'normal-mean') return { lo: -1.5, hi: 1.5 };
  if (preset.family === 'bernoulli') return { lo: 0, hi: 1 };
  return { lo: 0.1, hi: 5 };
}

function statRange(preset: DualityPreset): Range {
  if (preset.family === 'normal-mean') return { lo: -1.5, hi: 1.5 };
  if (preset.family === 'bernoulli') return { lo: 0, hi: 1 };
  return { lo: 0.1, hi: 5 };
}

/** Acceptance region A(θ₀) = [L_T(θ₀), U_T(θ₀)] for the two-sided z/score test. */
function acceptanceRegion(
  preset: DualityPreset,
  theta0: number,
  n: number,
  alpha: number,
): [number, number] {
  const z = standardNormalInvCDF(1 - alpha / 2);
  if (preset.family === 'normal-mean') {
    const sigma = preset.sigmaKnown ?? 1;
    const half = (z * sigma) / Math.sqrt(n);
    return [theta0 - half, theta0 + half];
  }
  if (preset.family === 'bernoulli') {
    const half = z * Math.sqrt((theta0 * (1 - theta0)) / n);
    return [Math.max(0, theta0 - half), Math.min(1, theta0 + half)];
  }
  // poisson
  const half = z * Math.sqrt(theta0 / n);
  return [Math.max(0, theta0 - half), theta0 + half];
}

/** CI C(T_obs) — invert the family's default test. */
function confidenceInterval(
  preset: DualityPreset,
  tObs: number,
  n: number,
  alpha: number,
): [number, number] {
  const z = standardNormalInvCDF(1 - alpha / 2);
  if (preset.family === 'normal-mean') {
    const sigma = preset.sigmaKnown ?? 1;
    const half = (z * sigma) / Math.sqrt(n);
    return [tObs - half, tObs + half];
  }
  if (preset.family === 'bernoulli') {
    const x = Math.round(tObs * n);
    const { lower, upper } = wilsonInterval(x, n, alpha);
    return [lower, upper];
  }
  // poisson — invert the score test in closed form
  const data = Array(n).fill(tObs);
  const { lower, upper } = scoreCI('poisson', data, alpha);
  return [lower, upper];
}

function formatNum(x: number): string {
  if (Math.abs(x) < 0.01) return x.toFixed(4);
  if (Math.abs(x) < 10) return x.toFixed(3);
  return x.toFixed(2);
}

export default function CITestDualityVisualizer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 720;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = dualityPresets[presetIndex];
  const [n, setN] = useState<number>(preset.n);
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [tObs, setTObs] = useState<number>(preset.observedStatistic);
  const [theta0, setTheta0] = useState<number>(preset.trueParam);

  const thetaRange = paramRange(preset);
  const tRange = statRange(preset);

  // When switching preset, reset sliders to preset defaults.
  const handlePreset = (idx: number): void => {
    const p = dualityPresets[idx];
    setPresetIndex(idx);
    setN(p.n);
    setAlpha(p.alpha);
    setTObs(p.observedStatistic);
    setTheta0(p.trueParam);
  };

  // Shared scales (same θ-axis shared across the left plane and the right CI panel).
  const plotW = isWide ? Math.max(420, w - 360) : w - 16;
  const plotH = H;

  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([thetaRange.lo, thetaRange.hi])
        .range([MARGIN.left, plotW - MARGIN.right]),
    [thetaRange.lo, thetaRange.hi, plotW],
  );

  const yScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([tRange.lo, tRange.hi])
        .range([plotH - MARGIN.bottom, MARGIN.top]),
    [tRange.lo, tRange.hi, plotH],
  );

  // Sample the acceptance region on a fine θ₀-grid and produce an SVG path for the band.
  const acceptBand = useMemo(() => {
    const N = 120;
    const lowers: [number, number][] = [];
    const uppers: [number, number][] = [];
    for (let i = 0; i <= N; i++) {
      const t = thetaRange.lo + (i / N) * (thetaRange.hi - thetaRange.lo);
      const [lo, hi] = acceptanceRegion(preset, t, n, alpha);
      lowers.push([t, lo]);
      uppers.push([t, hi]);
    }
    const line = d3
      .line<[number, number]>()
      .x((d) => xScale(d[0]))
      .y((d) => yScale(d[1]));
    const area = d3
      .area<[number, number]>()
      .x((d) => xScale(d[0]))
      .y0((_, i) => yScale(lowers[i][1]))
      .y1((_, i) => yScale(uppers[i][1]));
    return {
      lower: line(lowers) ?? '',
      upper: line(uppers) ?? '',
      band: area(uppers) ?? '',
    };
  }, [preset, n, alpha, thetaRange.lo, thetaRange.hi, xScale, yScale]);

  // Draggable horizontal line: observed T_obs.
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<'horiz' | 'vert' | null>(null);

  const handlePointerDown = (which: 'horiz' | 'vert') => (
    ev: React.PointerEvent<SVGLineElement>,
  ): void => {
    (ev.target as Element).setPointerCapture(ev.pointerId);
    setDragging(which);
  };
  const handlePointerMove = (ev: React.PointerEvent<SVGSVGElement>): void => {
    if (!svgRef.current || !dragging) return;
    const { left, top } = svgRef.current.getBoundingClientRect();
    const px = ev.clientX - left;
    const py = ev.clientY - top;
    if (dragging === 'horiz') {
      const v = yScale.invert(py);
      setTObs(Math.max(tRange.lo, Math.min(tRange.hi, v)));
    } else {
      const v = xScale.invert(px);
      setTheta0(Math.max(thetaRange.lo, Math.min(thetaRange.hi, v)));
    }
  };
  const handlePointerUp = (): void => setDragging(null);

  const ci = confidenceInterval(preset, tObs, n, alpha);
  const accept = acceptanceRegion(preset, theta0, n, alpha);

  // Axis tick values
  const xTicks = useMemo(() => xScale.ticks(5), [xScale]);
  const yTicks = useMemo(() => yScale.ticks(5), [yScale]);

  return (
    <div ref={containerRef} className="my-8 rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Scenario</span>
          <select
            value={presetIndex}
            onChange={(e) => handlePreset(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {dualityPresets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">n = {n}</span>
          <input
            type="range"
            min={5}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-32"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">α = {alpha.toFixed(3)}</span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.005}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-32"
          />
        </label>
      </div>

      <svg
        ref={svgRef}
        width={plotW}
        height={plotH}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        role="img"
        aria-label="Test–CI duality two-plane diagram"
        style={{ touchAction: 'none', maxWidth: '100%', height: 'auto' }}
      >
        {/* Grid */}
        {xTicks.map((t) => (
          <line
            key={`xg-${t}`}
            x1={xScale(t)}
            x2={xScale(t)}
            y1={MARGIN.top}
            y2={plotH - MARGIN.bottom}
            stroke={C.grid}
            strokeWidth={1}
          />
        ))}
        {yTicks.map((t) => (
          <line
            key={`yg-${t}`}
            x1={MARGIN.left}
            x2={plotW - MARGIN.right}
            y1={yScale(t)}
            y2={yScale(t)}
            stroke={C.grid}
            strokeWidth={1}
          />
        ))}

        {/* Acceptance band */}
        <path d={acceptBand.band} fill={C.accept} opacity={0.55} />
        <path d={acceptBand.lower} stroke={C.boundary} strokeWidth={1.3} fill="none" />
        <path d={acceptBand.upper} stroke={C.boundary} strokeWidth={1.3} fill="none" />

        {/* Vertical line — draggable; θ₀ */}
        <line
          x1={xScale(theta0)}
          x2={xScale(theta0)}
          y1={MARGIN.top}
          y2={plotH - MARGIN.bottom}
          stroke={C.accRegion}
          strokeWidth={2.2}
          style={{ cursor: 'ew-resize' }}
          onPointerDown={handlePointerDown('vert')}
        />
        {/* Acceptance region slice (green bar on the current y-slice) */}
        <line
          x1={xScale(theta0)}
          x2={xScale(theta0)}
          y1={yScale(accept[0])}
          y2={yScale(accept[1])}
          stroke={C.accRegion}
          strokeWidth={5}
          strokeLinecap="butt"
        />

        {/* Horizontal line — draggable; T_obs */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={yScale(tObs)}
          y2={yScale(tObs)}
          stroke={C.pivot}
          strokeWidth={2.2}
          style={{ cursor: 'ns-resize' }}
          onPointerDown={handlePointerDown('horiz')}
        />
        {/* CI slice (red bar along the current x-slice) */}
        <line
          y1={yScale(tObs)}
          y2={yScale(tObs)}
          x1={xScale(Math.max(thetaRange.lo, ci[0]))}
          x2={xScale(Math.min(thetaRange.hi, ci[1]))}
          stroke={C.ci}
          strokeWidth={5}
          strokeLinecap="butt"
        />

        {/* Axes */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={plotH - MARGIN.bottom}
          y2={plotH - MARGIN.bottom}
          stroke={C.axis}
        />
        <line
          x1={MARGIN.left}
          x2={MARGIN.left}
          y1={MARGIN.top}
          y2={plotH - MARGIN.bottom}
          stroke={C.axis}
        />
        {xTicks.map((t) => (
          <g key={`xt-${t}`}>
            <line
              x1={xScale(t)}
              x2={xScale(t)}
              y1={plotH - MARGIN.bottom}
              y2={plotH - MARGIN.bottom + 4}
              stroke={C.axis}
            />
            <text
              x={xScale(t)}
              y={plotH - MARGIN.bottom + 16}
              textAnchor="middle"
              fontSize={11}
              fill={C.text}
            >
              {formatNum(t)}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`yt-${t}`}>
            <line
              x1={MARGIN.left - 4}
              x2={MARGIN.left}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke={C.axis}
            />
            <text
              x={MARGIN.left - 6}
              y={yScale(t) + 4}
              textAnchor="end"
              fontSize={11}
              fill={C.text}
            >
              {formatNum(t)}
            </text>
          </g>
        ))}
        <text
          x={(MARGIN.left + plotW - MARGIN.right) / 2}
          y={plotH - 6}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
        >
          θ₀ (null value)
        </text>
        <text
          x={14}
          y={(MARGIN.top + plotH - MARGIN.bottom) / 2}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
          transform={`rotate(-90, 14, ${(MARGIN.top + plotH - MARGIN.bottom) / 2})`}
        >
          T (statistic)
        </text>
      </svg>

      <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
        <div className="rounded bg-white p-3 shadow-sm">
          <div className="mb-1 font-semibold" style={{ color: C.ci }}>
            Horizontal slicing → CI C(T_obs)
          </div>
          <div>
            At T = {formatNum(tObs)}, the set of θ₀ the test does NOT reject is
            <span className="font-mono"> [{formatNum(ci[0])}, {formatNum(ci[1])}]</span>.
            This is the{' '}
            {preset.family === 'bernoulli'
              ? 'Wilson'
              : preset.family === 'normal-mean'
                ? 'z'
                : 'score'}{' '}
            CI for {preset.family === 'normal-mean' ? 'μ' : preset.family === 'bernoulli' ? 'p' : 'λ'}.
          </div>
        </div>
        <div className="rounded bg-white p-3 shadow-sm">
          <div className="mb-1 font-semibold" style={{ color: C.accRegion }}>
            Vertical slicing → acceptance region A(θ₀)
          </div>
          <div>
            At θ₀ = {formatNum(theta0)}, the test does NOT reject for T ∈
            <span className="font-mono"> [{formatNum(accept[0])}, {formatNum(accept[1])}]</span>.
            Outside this range, H₀: θ = {formatNum(theta0)} would be rejected at level α.
          </div>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Drag the blue horizontal line to change T_obs and watch the CI update; drag the green vertical
        line to change θ₀ and watch the acceptance region update. Same shaded region, two slicings.
      </p>
    </div>
  );
}
