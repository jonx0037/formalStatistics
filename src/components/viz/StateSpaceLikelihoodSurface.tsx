/**
 * StateSpaceLikelihoodSurface — interactive for Topic 36 §36.8 (mode A) and
 * §36.4 Ex 7 (mode B).
 *
 * Learning and exactness as landscapes. Mode A is the Nile local level's
 * diffuse log-likelihood over (log₁₀ q, log₁₀ r): the direct Nelder–Mead MLE
 * sits on the ridge as a star, and Shumway–Stoffer EM (Thm 7) climbs from a
 * clickable start, one closed-form M-step at a time. Every step raises ℓ (the
 * ascent property, Topic 34 Thm 9), but the climb slows to a crawl along the
 * ridge: EM converges linearly (Rem 20, Ex 14), and the side plot shows the
 * gap ℓ̂ − ℓ(ϑ_k) shrinking geometrically on a log axis. "Use DK gaps" refits
 * with years 20–39 and 60–79 missing; the filter skips them and the M-step
 * averages R′ over observed times only (§36.5).
 *
 * Mode B is the profile exact ARMA(1,1) log-likelihood over (φ, θ) with σ²
 * concentrated out (Rem 9), computed in O(n) by the Kalman filter on Harvey's
 * state-space form (Def 2, Thm 3). The exact MLE, CSS and Whittle optima sit
 * on it with the truth (0.7, 0.4). At n = 100 they visibly spread; at n = 1000
 * they nearly coincide, the asymptotic equivalence of Thm 8 and Rem 10.
 *
 * Grids are computed in row chunks off idle time and cached (mode A per data
 * toggle, mode B per n). Filled bands sit at max − {0.5, 1, 2, 3, 5, 10, 20,
 * 40}; iso-lines come from a local marching-squares helper.
 */
import { useEffect, useId, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  armaCSS,
  armaMLE,
  armaProfileLogLik,
  kalmanLogLik,
  localLevel,
  localLevelMLE,
  ssmEMStep,
  whittleFit,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { likelihoodSurfacePresets, type LikelihoodSurfacePreset } from '../../data/time-series-data';
import { nileStartYear, nileValues } from '../../data/nile';
import { arma11Fixture } from '../../data/ssm-fixtures';

const MARGIN = { top: 10, right: 12, bottom: 36, left: 48 };
const MOBILE_BREAKPOINT = 640;
const BAND_DROPS = [0.5, 1, 2, 3, 5, 10, 20, 40];
/** Opacity of the shading for band b (b = 0: within 0.5 of the max; b = 8: more than 40 below). */
const BAND_OPACITY = [0.62, 0.52, 0.43, 0.35, 0.28, 0.21, 0.15, 0.09, 0.03];
const LABELED_DROPS = [2, 10, 40];
const EM_RTOL = 1e-8;
const EM_MAX_ITER = 500;
const EM_INTERVAL_MS = 50; // 20 iterations per second
/** Durbin & Koopman's artificial gaps: 0-based years 20..39 and 60..79 missing. */
const DK_GAPS: [number, number][] = [
  [20, 40],
  [60, 80],
];
const N_OPTIONS = [100, 250, 1000];
const TRUTH_B = { phi: 0.7, theta: 0.4 };

/** §36.4 Ex 7 (Cell 8, T36.12) at n = 1000, shown while the in-browser fits compute. */
const PUBLISHED_B1000: Record<EstimatorId, Fit> = {
  mle: { phi: 0.637421, theta: 0.40419, sigma2: 1.000176, loglik: -1419.605827 },
  css: { phi: 0.638968, theta: 0.402674, sigma2: 1.001488, loglik: -1419.607146 },
  whittle: { phi: 0.641025, theta: 0.38991, sigma2: 1.011371, loglik: -1419.678996 },
};

type EstimatorId = 'mle' | 'css' | 'whittle';
interface Fit {
  phi: number;
  theta: number;
  sigma2: number;
  /** Exact (profile) log-likelihood at (φ̂, θ̂). */
  loglik: number;
}
interface FitsB {
  fits: Record<EstimatorId, Fit>;
  /** S_t/σ² (the σ² = 1 innovation variances) at the exact MLE, t = 1..5. */
  S: number[];
}
interface Grid {
  z: Float64Array;
  nx: number;
  ny: number;
  x: [number, number];
  y: [number, number];
  /** Reference maximum for the bands: max(grid max, ℓ at the MLE). */
  max: number;
  fitsB?: FitsB;
}
interface EMPoint {
  q: number;
  r: number;
  ll: number;
}
interface EMState {
  pts: EMPoint[];
  done: null | 'converged' | 'maxIter';
}

const ESTIMATORS: { id: EstimatorId; label: string; symbol: string; color: string }[] = [
  { id: 'mle', label: 'Exact MLE (Kalman)', symbol: '★', color: timeSeriesColors.filtered },
  { id: 'css', label: 'CSS', symbol: '■', color: timeSeriesColors.forecast },
  { id: 'whittle', label: 'Whittle', symbol: '▲', color: timeSeriesColors.smoothed },
];

// ── Pure helpers ─────────────────────────────────────────────────────────────

/**
 * Marching squares: line segments of the level set {z = level} on a regular
 * grid, z[j·nx + i] at node (i, j). Returns a flat [i₁, j₁, i₂, j₂, …] array in
 * fractional grid-index coordinates. Saddles (codes 5 and 10) are resolved by
 * the cell-center average. Cells touching a non-finite value are skipped.
 */
function isoSegments(z: Float64Array, nx: number, ny: number, level: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = z[j * nx + i]; // (i, j)
      const b = z[j * nx + i + 1]; // (i+1, j)
      const c = z[(j + 1) * nx + i + 1]; // (i+1, j+1)
      const d = z[(j + 1) * nx + i]; // (i, j+1)
      if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Number.isFinite(d))) continue;
      const code = (a >= level ? 1 : 0) | (b >= level ? 2 : 0) | (c >= level ? 4 : 0) | (d >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      // Crossing points on the four edges (only evaluated on edges that cross).
      const bot = () => [i + (level - a) / (b - a), j];
      const rgt = () => [i + 1, j + (level - b) / (c - b)];
      const top = () => [i + (level - d) / (c - d), j + 1];
      const lft = () => [i, j + (level - a) / (d - a)];
      const seg = (p: number[], q: number[]) => out.push(p[0], p[1], q[0], q[1]);
      switch (code) {
        case 1:
        case 14:
          seg(lft(), bot());
          break;
        case 2:
        case 13:
          seg(bot(), rgt());
          break;
        case 3:
        case 12:
          seg(lft(), rgt());
          break;
        case 4:
        case 11:
          seg(rgt(), top());
          break;
        case 6:
        case 9:
          seg(bot(), top());
          break;
        case 7:
        case 8:
          seg(lft(), top());
          break;
        case 5: {
          // a, c above; b, d below.
          const centerAbove = (a + b + c + d) / 4 >= level;
          if (centerAbove) {
            seg(bot(), rgt());
            seg(lft(), top());
          } else {
            seg(lft(), bot());
            seg(rgt(), top());
          }
          break;
        }
        case 10: {
          // b, d above; a, c below.
          const centerAbove = (a + b + c + d) / 4 >= level;
          if (centerAbove) {
            seg(lft(), bot());
            seg(rgt(), top());
          } else {
            seg(bot(), rgt());
            seg(lft(), top());
          }
          break;
        }
        default:
          break;
      }
    }
  }
  return out;
}

/** Band index of value v: 0 if within 0.5 of max, …, 8 if more than 40 below. */
function bandOf(v: number, max: number): number {
  const drop = max - v;
  for (let b = 0; b < BAND_DROPS.length; b++) if (drop <= BAND_DROPS[b]) return b;
  return BAND_DROPS.length;
}

/** Row runs of equal band (fewer SVG nodes than one rect per cell). */
function bandRuns(g: Grid): { j: number; i0: number; i1: number; band: number }[] {
  const runs: { j: number; i0: number; i1: number; band: number }[] = [];
  for (let j = 0; j < g.ny; j++) {
    let i0 = 0;
    let cur = bandOf(g.z[j * g.nx], g.max);
    for (let i = 1; i <= g.nx; i++) {
      const b = i < g.nx ? bandOf(g.z[j * g.nx + i], g.max) : -1;
      if (b !== cur) {
        runs.push({ j, i0, i1: i - 1, band: cur });
        i0 = i;
        cur = b;
      }
    }
  }
  return runs;
}

function applyGaps(x: number[], gaps: [number, number][]): number[] {
  return x.map((v, t) => (gaps.some(([a, b]) => t >= a && t < b) ? Number.NaN : v));
}

function demeanedPrefix(x: number[], n: number): number[] {
  const s = x.slice(0, n);
  const m = s.reduce((acc, v) => acc + v, 0) / s.length;
  return s.map((v) => v - m);
}

function computeFitsB(x: number[]): FitsB {
  const exact = armaMLE(x, 1, 1, { estimateMean: false });
  const css = armaCSS(x, 1, 1);
  const wh = whittleFit(x, 1, 1);
  const at = (phi: number, theta: number, sigma2: number): Fit => ({
    phi,
    theta,
    sigma2,
    loglik: armaProfileLogLik(x, [phi], [theta]).loglik,
  });
  return {
    fits: {
      mle: at(exact.phi[0], exact.theta[0], exact.sigma2),
      css: at(css.phi[0], css.theta[0], css.sigma2),
      whittle: at(wh.phi[0], wh.theta[0], wh.sigma2),
    },
    S: armaProfileLogLik(x, exact.phi, exact.theta).S.slice(0, 5),
  };
}

function initEM(x: number[], start: [number, number]): EMState {
  return { pts: [{ q: start[0], r: start[1], ll: kalmanLogLik(x, localLevel(start[0], start[1])) }], done: null };
}

/** One EM iteration (Thm 7). Stops when the relative increase < 1e-8 or after 500 M-steps (ssmEM's rule). */
function emAdvance(x: number[], s: EMState): EMState {
  if (s.done) return s;
  const cur = s.pts[s.pts.length - 1];
  const step = ssmEMStep(x, localLevel(cur.q, cur.r));
  const next: EMPoint = { q: step.model.Q[0][0], r: step.model.R[0][0], ll: kalmanLogLik(x, step.model) };
  const pts = [...s.pts, next];
  const inc = next.ll - cur.ll;
  const done = inc < EM_RTOL * Math.abs(cur.ll) ? 'converged' : pts.length - 1 >= EM_MAX_ITER ? 'maxIter' : null;
  return { pts, done };
}

function niceTicks(lo: number, hi: number, step: number): { v: number; label: string }[] {
  const out: { v: number; label: string }[] = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) {
    const v = Number(t.toFixed(6));
    out.push({ v, label: String(Number(v.toPrecision(3))) });
  }
  return out;
}

function decadeTicks(lo: number, hi: number): { v: number; label: string }[] {
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const stride = b - a > 6 ? 2 : 1;
  const out: { v: number; label: string }[] = [];
  for (let k = a; k <= b; k += stride) out.push({ v: 10 ** k, label: k === 0 ? '1' : `1e${k}` });
  return out;
}

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function starPoints(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let k = 0; k < 10; k++) {
    const rad = k % 2 === 0 ? r : r * 0.45;
    const ang = -Math.PI / 2 + (k * Math.PI) / 5;
    pts.push(`${(cx + rad * Math.cos(ang)).toFixed(2)},${(cy + rad * Math.sin(ang)).toFixed(2)}`);
  }
  return pts.join(' ');
}

function fmtExp(v: number): string {
  if (!Number.isFinite(v)) return '—';
  return v.toExponential(2).replace('e-', 'e−');
}

// ── Frames ───────────────────────────────────────────────────────────────────

function Axes({
  plotW,
  plotH,
  sx,
  sy,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
  grid = true,
}: {
  plotW: number;
  plotH: number;
  sx: (v: number) => number;
  sy: (v: number) => number;
  xTicks: { v: number; label: string }[];
  yTicks: { v: number; label: string }[];
  xLabel: string;
  yLabel: string;
  grid?: boolean;
}) {
  const axisColor = 'var(--color-text-muted)';
  return (
    <>
      {yTicks.map((t) => (
        <g key={`y${t.label}`}>
          {grid && <line x1={0} y1={sy(t.v)} x2={plotW} y2={sy(t.v)} stroke="var(--color-viz-grid)" />}
          <line x1={-4} y1={sy(t.v)} x2={0} y2={sy(t.v)} stroke={axisColor} />
          <text x={-6} y={sy(t.v) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
            {t.label}
          </text>
        </g>
      ))}
      <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
      <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
      {xTicks.map((t) => (
        <g key={`x${t.label}`}>
          <line x1={sx(t.v)} y1={plotH} x2={sx(t.v)} y2={plotH + 4} stroke={axisColor} />
          <text x={sx(t.v)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
            {t.label}
          </text>
        </g>
      ))}
      <text x={plotW / 2} y={plotH + 31} textAnchor="middle" fontSize={11} fill="var(--color-text)">
        {xLabel}
      </text>
      <text x={-38} y={plotH / 2} textAnchor="middle" fontSize={11} fill="var(--color-text)" transform={`rotate(-90 ${-38} ${plotH / 2})`}>
        {yLabel}
      </text>
    </>
  );
}

function SurfacePlot({
  size,
  grid,
  spec,
  progress,
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  label,
  onPick,
  children,
}: {
  size: number;
  grid: Grid | null;
  spec: LikelihoodSurfacePreset['grid'];
  progress: number | null;
  xLabel: string;
  yLabel: string;
  xTicks: { v: number; label: string }[];
  yTicks: { v: number; label: string }[];
  label: string;
  onPick?: (gx: number, gy: number) => void;
  children: (sx: (v: number) => number, sy: (v: number) => number) => React.ReactNode;
}) {
  const clipId = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const plotW = Math.max(10, size - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, size - MARGIN.top - MARGIN.bottom);
  const [x0, x1] = spec.x;
  const [y0, y1] = spec.y;
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * plotW;
  const sy = (v: number) => plotH - ((v - y0) / (y1 - y0)) * plotH;

  const runs = useMemo(() => (grid ? bandRuns(grid) : []), [grid]);
  const contours = useMemo(
    () => (grid ? BAND_DROPS.map((drop) => ({ drop, segs: isoSegments(grid.z, grid.nx, grid.ny, grid.max - drop) })) : []),
    [grid],
  );

  const shapes = useMemo(() => {
    if (!grid) return null;
    const dx = (x1 - x0) / (grid.nx - 1);
    const dy = (y1 - y0) / (grid.ny - 1);
    const px = (i: number) => sx(x0 + i * dx);
    const py = (j: number) => sy(y0 + j * dy);
    const rects = runs.map((rn) => {
      const left = px(rn.i0 - 0.5);
      const right = px(rn.i1 + 0.5);
      const topY = py(rn.j + 0.5);
      const botY = py(rn.j - 0.5);
      return {
        key: `${rn.j}-${rn.i0}`,
        x: left,
        y: topY,
        w: Math.max(0, right - left + 0.5),
        h: Math.max(0, botY - topY + 0.5),
        band: rn.band,
      };
    });
    const lines = contours.map(({ drop, segs }) => {
      let d = '';
      let best = -1;
      let bestScore = -Infinity;
      for (let k = 0; k < segs.length; k += 4) {
        d += `M${px(segs[k]).toFixed(1)},${py(segs[k + 1]).toFixed(1)}L${px(segs[k + 2]).toFixed(1)},${py(segs[k + 3]).toFixed(1)}`;
        // Label anchor: the crossing nearest the bottom-right corner, kept off the edges.
        const mi = (segs[k] + segs[k + 2]) / 2 / (grid.nx - 1);
        const mj = (segs[k + 1] + segs[k + 3]) / 2 / (grid.ny - 1);
        if (mi < 0.06 || mi > 0.94 || mj < 0.06 || mj > 0.94) continue;
        const score = mi - mj;
        if (score > bestScore) {
          bestScore = score;
          best = k;
        }
      }
      const lab =
        best >= 0 && LABELED_DROPS.includes(drop)
          ? { x: px((segs[best] + segs[best + 2]) / 2), y: py((segs[best + 1] + segs[best + 3]) / 2) }
          : null;
      return { drop, d, lab };
    });
    return { rects, lines };
    // sx/sy are recreated each render but depend only on plotW/plotH and the spec ranges listed here.
  }, [grid, runs, contours, plotW, plotH, x0, x1, y0, y1]);

  const handleClick = (ev: React.MouseEvent<SVGSVGElement>) => {
    if (!onPick) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const px = ev.clientX - rect.left - MARGIN.left;
    const py = ev.clientY - rect.top - MARGIN.top;
    if (px < 0 || px > plotW || py < 0 || py > plotH) return;
    onPick(x0 + (px / plotW) * (x1 - x0), y0 + (1 - py / plotH) * (y1 - y0));
  };

  return (
    <svg
      width={size}
      height={size}
      className="block"
      role="img"
      aria-label={label}
      onClick={handleClick}
      style={onPick ? { cursor: 'crosshair', touchAction: 'manipulation' } : undefined}
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={plotW} height={plotH} />
        </clipPath>
      </defs>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        <rect x={0} y={0} width={plotW} height={plotH} fill="var(--color-surface)" />
        <g clipPath={`url(#${clipId})`}>
          {shapes?.rects.map((r) => (
            <rect
              key={r.key}
              x={r.x}
              y={r.y}
              width={r.w}
              height={r.h}
              // Opaque mix, so the 0.5 px overlaps between neighboring cells cannot double the alpha into seams.
              style={{ fill: `color-mix(in srgb, ${timeSeriesColors.neutral} ${(BAND_OPACITY[r.band] * 100).toFixed(1)}%, var(--color-surface))` }}
            />
          ))}
          {shapes?.lines.map((l) => (
            <path
              key={l.drop}
              d={l.d}
              fill="none"
              stroke="var(--color-text-muted)"
              strokeWidth={LABELED_DROPS.includes(l.drop) ? 1.1 : 0.6}
              strokeOpacity={0.9}
            />
          ))}
          {shapes?.lines.map(
            (l) =>
              l.lab && (
                <text
                  key={`lab${l.drop}`}
                  x={l.lab.x}
                  y={l.lab.y + 3}
                  textAnchor="middle"
                  fontSize={9}
                  fill="var(--color-text)"
                  stroke="var(--color-surface)"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  −{l.drop}
                </text>
              ),
          )}
          {children(sx, sy)}
        </g>
        {!grid && (
          <text x={plotW / 2} y={plotH / 2} textAnchor="middle" fontSize={12} fill="var(--color-text-muted)">
            computing the surface… {progress !== null ? `${Math.round(progress * 100)}%` : ''}
          </text>
        )}
        <Axes plotW={plotW} plotH={plotH} sx={sx} sy={sy} xTicks={xTicks} yTicks={yTicks} xLabel={xLabel} yLabel={yLabel} grid={false} />
      </g>
    </svg>
  );
}

function GapPlot({ width, height, pts, mle }: { width: number; height: number; pts: EMPoint[]; mle: number }) {
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
  const gaps = pts.map((p) => Math.max(mle - p.ll, 1e-12));
  const kMax = Math.max(20, Math.ceil((pts.length - 1) / 50) * 50);
  const lo = 10 ** Math.floor(Math.log10(Math.min(...gaps, 1e-4)));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...gaps, 10)));
  const ly0 = Math.log10(lo);
  const ly1 = Math.log10(hi);
  const sx = (v: number) => (v / kMax) * plotW;
  const sy = (v: number) => plotH - ((Math.log10(Math.max(v, lo)) - ly0) / (ly1 - ly0)) * plotH;
  const ks = gaps.map((_, k) => k);
  return (
    <svg width={width} height={height} className="block" role="img" aria-label="Gap between the maximum log-likelihood and the EM iterate's log-likelihood against the iteration, log scale">
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        <Axes
          plotW={plotW}
          plotH={plotH}
          sx={sx}
          sy={sy}
          xTicks={niceTicks(0, kMax, kMax / 4)}
          yTicks={decadeTicks(lo, hi)}
          xLabel="EM iteration k"
          yLabel="ℓ̂ − ℓ(ϑ_k)"
        />
        <line x1={0} x2={plotW} y1={sy(0.01)} y2={sy(0.01)} stroke={timeSeriesColors.truth} strokeDasharray="4 4" strokeOpacity={0.6} />
        <text x={plotW - 2} y={sy(0.01) - 3} textAnchor="end" fontSize={9} fill="var(--color-text-muted)">
          0.01
        </text>
        <path d={linePath(ks, gaps, sx, sy)} fill="none" stroke={timeSeriesColors.forecast} strokeWidth={2} />
        {pts.length > 0 && <circle cx={sx(pts.length - 1)} cy={sy(gaps[gaps.length - 1])} r={3.5} fill={timeSeriesColors.forecast} />}
      </g>
    </svg>
  );
}

// ── Component ────────────────────────────────────────────────────────────────

export default function StateSpaceLikelihoodSurface() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const presetA = likelihoodSurfacePresets.find((p) => p.mode === 'localLevel') ?? likelihoodSurfacePresets[0];
  const presetB = likelihoodSurfacePresets.find((p) => p.mode === 'arma11') ?? likelihoodSurfacePresets[1];
  const defaultStart: [number, number] = presetA.emStart ?? [10000, 5000];

  const [mode, setMode] = useState<'A' | 'B'>('A');
  const [useGaps, setUseGaps] = useState(false);
  const [nB, setNB] = useState(1000);
  const [grids, setGrids] = useState<Record<string, Grid>>({});
  const [progress, setProgress] = useState<number | null>(null);
  const [start, setStart] = useState<[number, number]>(defaultStart);
  const [em, setEm] = useState<EMState>(() => initEM(nileValues, defaultStart));
  const [running, setRunning] = useState(false);

  const xA = useMemo(() => (useGaps ? applyGaps(nileValues, DK_GAPS) : nileValues), [useGaps]);
  const mleA = useMemo(() => localLevelMLE(xA), [xA]);
  const xB = useMemo(() => demeanedPrefix(arma11Fixture, nB), [nB]);

  const gridKey = mode === 'A' ? `A:${useGaps ? 'gaps' : 'full'}` : `B:${nB}`;
  const grid = grids[gridKey] ?? null;
  const haveGrid = grid !== null;

  // Chunked grid computation (idle time, ~12 ms slices), cached per key.
  useEffect(() => {
    if (haveGrid) {
      setProgress(null);
      return;
    }
    const isA = mode === 'A';
    const spec = isA ? presetA.grid : presetB.grid;
    const nx = spec.n;
    const ny = spec.n;
    const dx = (spec.x[1] - spec.x[0]) / (nx - 1);
    const dy = (spec.y[1] - spec.y[0]) / (ny - 1);
    const z = new Float64Array(nx * ny);
    const evalAt = isA
      ? (gx: number, gy: number) => kalmanLogLik(xA, localLevel(10 ** gx, 10 ** gy))
      : (gx: number, gy: number) => armaProfileLogLik(xB, [gx], [gy]).loglik;
    let fitsB: FitsB | undefined;
    let row = 0;
    let cancelled = false;
    const hasIdle = typeof window.requestIdleCallback === 'function';
    let handle = 0;
    const work = () => {
      if (cancelled) return;
      const t0 = performance.now();
      if (!isA && !fitsB) fitsB = computeFitsB(xB);
      while (row < ny && performance.now() - t0 < 12) {
        const gy = spec.y[0] + row * dy;
        for (let i = 0; i < nx; i++) z[row * nx + i] = evalAt(spec.x[0] + i * dx, gy);
        row++;
      }
      if (row < ny) {
        setProgress(row / ny);
        schedule();
        return;
      }
      let max = -Infinity;
      for (let k = 0; k < z.length; k++) if (z[k] > max) max = z[k];
      max = Math.max(max, isA ? mleA.loglik : (fitsB?.fits.mle.loglik ?? max));
      setGrids((prev) => ({ ...prev, [gridKey]: { z, nx, ny, x: spec.x, y: spec.y, max, fitsB } }));
      setProgress(null);
    };
    const schedule = () => {
      handle = hasIdle ? window.requestIdleCallback(work, { timeout: 120 }) : window.setTimeout(work, 0);
    };
    setProgress(0);
    schedule();
    return () => {
      cancelled = true;
      if (hasIdle) window.cancelIdleCallback(handle);
      else window.clearTimeout(handle);
    };
  }, [gridKey, haveGrid, mode, xA, xB, mleA, presetA.grid, presetB.grid]);

  // EM animation: 20 iterations per second until the stop rule fires.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setEm((s) => emAdvance(xA, s)), EM_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [running, xA]);
  useEffect(() => {
    if (em.done && running) setRunning(false);
  }, [em.done, running]);

  const resetEM = (x: number[], s: [number, number]) => {
    setRunning(false);
    setEm(initEM(x, s));
  };
  const toggleGaps = () => {
    const next = !useGaps;
    setUseGaps(next);
    resetEM(next ? applyGaps(nileValues, DK_GAPS) : nileValues, start);
  };
  const pickStart = (gx: number, gy: number) => {
    const s: [number, number] = [10 ** gx, 10 ** gy];
    setStart(s);
    resetEM(xA, s);
  };
  const switchMode = (m: 'A' | 'B') => {
    setRunning(false);
    setMode(m);
  };

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const fullW = Math.max(240, containerW);
  const plotSize = isMobile ? fullW : Math.round(Math.min(560, Math.max(260, (fullW - 16) * 0.58)));
  const sideW = isMobile ? fullW : Math.max(200, fullW - 16 - plotSize);
  const sideH = isMobile ? 220 : Math.max(220, Math.min(300, plotSize * 0.6));

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-filtered)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const btnStyle = { borderWidth: 1, borderColor: 'var(--color-border)' };

  // ── Mode A readout ────────────────────────────────────────────────────────
  const cur = em.pts[em.pts.length - 1];
  const steps = em.pts.length - 1;
  const lastInc = steps > 0 ? cur.ll - em.pts[steps - 1].ll : null;
  const gapToMle = mleA.loglik - cur.ll;
  const isDefaultStart = start[0] === defaultStart[0] && start[1] === defaultStart[1];

  // ── Mode B readout ────────────────────────────────────────────────────────
  const fitsB = mode === 'B' ? (grid?.fitsB ?? null) : null;
  const shownFits = fitsB?.fits ?? (nB === 1000 ? PUBLISHED_B1000 : null);

  const progressBar = progress !== null && (
    <div
      className="mb-2 h-2 w-full overflow-hidden rounded"
      style={{ background: 'var(--color-surface-alt)' }}
      role="progressbar"
      aria-label="Surface computation progress"
      aria-valuenow={Math.round(progress * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full" style={{ width: `${progress * 100}%`, background: timeSeriesColors.filtered }} />
    </div>
  );

  const bandLegend = (
    <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
      <span>ℓ − max ℓ:</span>
      {BAND_OPACITY.map((op, b) => (
        <span key={b} className="inline-flex items-center gap-0.5">
          <span className="relative inline-block h-2.5 w-3 rounded-sm" style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}>
            <span className="absolute inset-0" style={{ background: timeSeriesColors.neutral, opacity: op }} />
          </span>
          {b < BAND_DROPS.length ? `≥ −${BAND_DROPS[b]}` : `< −40`}
        </span>
      ))}
    </div>
  );

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Surface:</span>
        <button onClick={() => switchMode('A')} className={btn(mode === 'A')} style={btnStyle} aria-pressed={mode === 'A'}>
          A: Nile local level (q, r)
        </button>
        <button onClick={() => switchMode('B')} className={btn(mode === 'B')} style={btnStyle} aria-pressed={mode === 'B'}>
          B: ARMA(1,1) (φ, θ)
        </button>
      </div>

      {mode === 'A' ? (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
            <button
              onClick={() => setRunning((v) => !v)}
              disabled={em.done !== null}
              className="rounded px-3 py-1 text-xs font-medium text-white disabled:opacity-40"
              style={{ background: timeSeriesColors.forecast }}
            >
              {running ? 'Pause' : 'Run EM'}
            </button>
            <button
              onClick={() => setEm((s) => emAdvance(xA, s))}
              disabled={running || em.done !== null}
              className={`${btn(false)} disabled:opacity-40`}
              style={btnStyle}
            >
              Step
            </button>
            <button onClick={() => resetEM(xA, start)} className={btn(false)} style={btnStyle}>
              Reset
            </button>
            {!isDefaultStart && (
              <button
                onClick={() => {
                  setStart(defaultStart);
                  resetEM(xA, defaultStart);
                }}
                className={btn(false)}
                style={btnStyle}
              >
                Default start (10000, 5000)
              </button>
            )}
            <button onClick={toggleGaps} className={btn(useGaps)} style={btnStyle} aria-pressed={useGaps}>
              Use DK gaps
            </button>
          </div>
          <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
            <span>
              k = {steps} M-steps ({em.pts.length} ℓ evaluations)
            </span>
            <span>
              (q, r) = ({cur.q.toFixed(2)}, {cur.r.toFixed(2)})
            </span>
            <span>ℓ = {cur.ll.toFixed(6)}</span>
            <span>ℓ̂ − ℓ = {fmtExp(gapToMle)}</span>
            <span>
              last increase ={' '}
              {lastInc === null ? (
                '—'
              ) : lastInc > 0 ? (
                <span style={{ color: timeSeriesColors.smoothed }}>+{fmtExp(lastInc)} (ascent ✓)</span>
              ) : (
                <>
                  {fmtExp(lastInc)}{' '}
                  <span className="rounded px-1.5 py-0.5 font-sans font-semibold text-white" style={{ background: timeSeriesColors.warn }}>
                    ascent violated
                  </span>
                </>
              )}
            </span>
            {em.done && (
              <span className="font-sans">
                {em.done === 'converged' ? 'stopped: relative increase < 10⁻⁸·|ℓ|' : `stopped: ${EM_MAX_ITER} iterations`}
              </span>
            )}
          </div>
          <div className="mb-3 font-mono text-xs" style={{ color: 'var(--color-text-muted)' }}>
            direct MLE (Nelder–Mead): q̂ = {mleA.q.toFixed(2)}, r̂ = {mleA.r.toFixed(2)}, ℓ̂ = {mleA.loglik.toFixed(6)}
            {useGaps && <span className="font-sans"> · Nile {nileStartYear + 20}–{nileStartYear + 39} and {nileStartYear + 60}–{nileStartYear + 79} missing</span>}
          </div>
        </>
      ) : (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          <span className="text-xs font-semibold">sample size n:</span>
          {N_OPTIONS.map((n) => (
            <button key={n} onClick={() => setNB(n)} className={btn(n === nB)} style={btnStyle} aria-pressed={n === nB}>
              {n}
            </button>
          ))}
          <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            first n values of the ARMA(1,1) fixture, demeaned; truth φ = 0.7, θ = 0.4
          </span>
        </div>
      )}

      {progressBar}

      {width > 0 ? (
        <div className={isMobile ? 'flex flex-col gap-4' : 'flex items-start gap-4'}>
          <div className="min-w-0" style={{ width: plotSize }}>
            <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {mode === 'A' ? (
                <>
                  <span style={{ color: timeSeriesColors.filtered }}>★ direct MLE</span> ·{' '}
                  <span style={{ color: timeSeriesColors.forecast }}>○ EM start, — EM path, ● current</span> · click or tap to start EM there
                </>
              ) : (
                <>
                  <span style={{ color: timeSeriesColors.truth }}>+ truth</span>
                  {ESTIMATORS.map((e) => (
                    <span key={e.id} style={{ color: e.color }}>
                      {' '}
                      · {e.symbol} {e.label}
                    </span>
                  ))}
                </>
              )}
            </div>
            {mode === 'A' ? (
              <SurfacePlot
                size={plotSize}
                grid={grid}
                spec={presetA.grid}
                progress={progress}
                xLabel="log₁₀ q (level variance)"
                yLabel="log₁₀ r (observation variance)"
                xTicks={niceTicks(presetA.grid.x[0], presetA.grid.x[1], 0.5)}
                yTicks={niceTicks(presetA.grid.y[0], presetA.grid.y[1], 0.2)}
                label="Contours of the Nile local-level log-likelihood over log q and log r, with the direct MLE and the EM path"
                onPick={pickStart}
              >
                {(sx, sy) => {
                  const lx = em.pts.map((p) => Math.log10(p.q));
                  const lyv = em.pts.map((p) => Math.log10(p.r));
                  const mx = sx(Math.log10(mleA.q));
                  const my = sy(Math.log10(mleA.r));
                  return (
                    <>
                      <path d={linePath(lx, lyv, sx, sy)} fill="none" stroke={timeSeriesColors.forecast} strokeWidth={2} />
                      {em.pts.length < 220 &&
                        em.pts.map((_, k) => <circle key={k} cx={sx(lx[k])} cy={sy(lyv[k])} r={1.6} fill={timeSeriesColors.forecast} />)}
                      <circle cx={sx(lx[0])} cy={sy(lyv[0])} r={5.5} fill="var(--color-surface)" stroke={timeSeriesColors.forecast} strokeWidth={2} />
                      <polygon points={starPoints(mx, my, 9)} fill={timeSeriesColors.filtered} stroke="var(--color-surface)" strokeWidth={1.2} />
                      <circle
                        cx={sx(lx[lx.length - 1])}
                        cy={sy(lyv[lyv.length - 1])}
                        r={4.5}
                        fill={timeSeriesColors.forecast}
                        stroke="var(--color-surface)"
                        strokeWidth={1.2}
                      />
                    </>
                  );
                }}
              </SurfacePlot>
            ) : (
              <SurfacePlot
                size={plotSize}
                grid={grid}
                spec={presetB.grid}
                progress={progress}
                xLabel="φ (AR coefficient)"
                yLabel="θ (MA coefficient)"
                xTicks={niceTicks(presetB.grid.x[0], presetB.grid.x[1], 0.1)}
                yTicks={niceTicks(presetB.grid.y[0], presetB.grid.y[1], 0.1)}
                label="Contours of the profile exact ARMA(1,1) log-likelihood over phi and theta, with the truth and the exact, CSS and Whittle estimates"
              >
                {(sx, sy) => {
                  const tx = sx(TRUTH_B.phi);
                  const ty = sy(TRUTH_B.theta);
                  const f = fitsB?.fits;
                  return (
                    <>
                      <g stroke={timeSeriesColors.truth} strokeWidth={2.2}>
                        <line x1={tx - 7} x2={tx + 7} y1={ty} y2={ty} />
                        <line x1={tx} x2={tx} y1={ty - 7} y2={ty + 7} />
                      </g>
                      {f && (
                        <>
                          <polygon
                            points={(() => {
                              const cx = sx(f.whittle.phi);
                              const cy = sy(f.whittle.theta);
                              return `${cx},${cy - 7} ${cx + 6.5},${cy + 5} ${cx - 6.5},${cy + 5}`;
                            })()}
                            fill={timeSeriesColors.smoothed}
                            stroke="var(--color-surface)"
                            strokeWidth={1}
                          />
                          <rect
                            x={sx(f.css.phi) - 5}
                            y={sy(f.css.theta) - 5}
                            width={10}
                            height={10}
                            fill={timeSeriesColors.forecast}
                            stroke="var(--color-surface)"
                            strokeWidth={1}
                          />
                          <polygon
                            points={starPoints(sx(f.mle.phi), sy(f.mle.theta), 9)}
                            fill={timeSeriesColors.filtered}
                            stroke="var(--color-surface)"
                            strokeWidth={1.2}
                          />
                        </>
                      )}
                    </>
                  );
                }}
              </SurfacePlot>
            )}
            {bandLegend}
          </div>

          <div className="min-w-0" style={{ width: isMobile ? fullW : sideW }}>
            {mode === 'A' ? (
              <>
                <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  Gap to the direct MLE, log axis: EM's linear convergence (Rem 20)
                </div>
                <GapPlot width={isMobile ? fullW : sideW} height={sideH} pts={em.pts} mle={mleA.loglik} />
              </>
            ) : (
              <>
                <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  Estimates at n = {nB}; ℓ is the exact profile log-likelihood at (φ̂, θ̂)
                  {!fitsB && shownFits && ' (published Ex 7 values while computing)'}
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-xs">
                    <thead>
                      <tr style={{ color: 'var(--color-text-muted)' }}>
                        <th className="py-1 pr-2 text-left font-sans font-normal">estimator</th>
                        <th className="px-1 text-right font-normal">φ̂</th>
                        <th className="px-1 text-right font-normal">θ̂</th>
                        <th className="px-1 text-right font-normal">σ̂²</th>
                        <th className="pl-1 text-right font-normal">exact ℓ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ESTIMATORS.map((e) => {
                        const fit = shownFits?.[e.id];
                        return (
                          <tr key={e.id} style={{ borderTop: '1px solid var(--color-border)' }}>
                            <td className="py-1 pr-2 font-sans">
                              <span style={{ color: e.color }}>{e.symbol}</span> {e.label}
                            </td>
                            <td className="px-1 text-right">{fit ? fit.phi.toFixed(4) : '…'}</td>
                            <td className="px-1 text-right">{fit ? fit.theta.toFixed(4) : '…'}</td>
                            <td className="px-1 text-right">{fit ? fit.sigma2.toFixed(4) : '…'}</td>
                            <td className="pl-1 text-right">{fit ? fit.loglik.toFixed(3) : '…'}</td>
                          </tr>
                        );
                      })}
                      <tr style={{ borderTop: '1px solid var(--color-border)' }}>
                        <td className="py-1 pr-2 font-sans">
                          <span style={{ color: timeSeriesColors.truth }}>+</span> truth
                        </td>
                        <td className="px-1 text-right">{TRUTH_B.phi.toFixed(4)}</td>
                        <td className="px-1 text-right">{TRUTH_B.theta.toFixed(4)}</td>
                        <td className="px-1 text-right">1.0000</td>
                        <td className="pl-1 text-right">—</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="mt-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  S_t/σ² at the exact MLE, t = 1…5 (Thm 3: S_t = v_(t−1), so S_t/σ² falls to 1 at rate ≈ θ̂² per step)
                </div>
                <div className="font-mono text-xs">{fitsB ? fitsB.S.map((s) => s.toFixed(6)).join(', ') : 'computing…'}</div>
                {fitsB && (
                  <div className="mt-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Spread of φ̂ across the three estimators:{' '}
                    <span className="font-mono">
                      {(
                        Math.max(fitsB.fits.mle.phi, fitsB.fits.css.phi, fitsB.fits.whittle.phi) -
                        Math.min(fitsB.fits.mle.phi, fitsB.fits.css.phi, fitsB.fits.whittle.phi)
                      ).toFixed(4)}
                    </span>
                    ; of θ̂:{' '}
                    <span className="font-mono">
                      {(
                        Math.max(fitsB.fits.mle.theta, fitsB.fits.css.theta, fitsB.fits.whittle.theta) -
                        Math.min(fitsB.fits.mle.theta, fitsB.fits.css.theta, fitsB.fits.whittle.theta)
                      ).toFixed(4)}
                    </span>
                    . They shrink as n grows (Thm 8, Rem 10).
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      ) : (
        <div style={{ height: 560 }} aria-hidden="true" />
      )}
    </div>
  );
}
