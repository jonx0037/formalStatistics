import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { penaltyColors } from './shared/colorScales';

const PANEL_H = 360;
const MARGIN = { top: 24, right: 24, bottom: 36, left: 36 };
const ELLIPSE_COLOR = '#475569';
const FILL_TINT = '#e2e8f0';

type PenaltyKind = 'ridge' | 'lasso' | 'elasticnet';
const TABS: { key: PenaltyKind; label: string; symbol: string }[] = [
  { key: 'ridge', label: 'Ridge (L²)', symbol: '||β||₂² ≤ t' },
  { key: 'lasso', label: 'Lasso (L¹)', symbol: '||β||₁ ≤ t' },
  { key: 'elasticnet', label: 'Elastic net', symbol: 'α||β||₁ + (1−α)||β||₂²/2 ≤ t' },
];

export default function LevelSetExplorer() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 600, 320);
  const panelW = Math.min(width - 16, 420);
  const innerW = panelW - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  const sz = Math.min(innerW, innerH);

  const [tab, setTab] = useState<PenaltyKind>('ridge');
  const [t, setT] = useState(0.6); // constraint radius
  const [alpha, setAlpha] = useState(0.5); // elastic-net mix

  // OLS solution and elliptical loss contours.
  const olsX = 0.7;
  const olsY = 0.5;
  const a = 0.45; // ellipse semi-major
  const b = 0.18; // ellipse semi-minor
  const tilt = -25 * (Math.PI / 180);

  // The constrained solution: for ridge/lasso/elasticnet, find the closest
  // point of the elliptical contour to the OLS solution intersected with the
  // constraint set. Use a numerical sweep over angles for ridge, vertex check
  // for lasso, and a hybrid for elastic net. Cheap and visualization-only.
  const constrainedSolution = useMemo(() => {
    const candidates: { x: number; y: number; loss: number }[] = [];

    const lossAt = (x: number, y: number): number => {
      // Squared distance in rotated coords.
      const dx = x - olsX;
      const dy = y - olsY;
      const xr = dx * Math.cos(-tilt) - dy * Math.sin(-tilt);
      const yr = dx * Math.sin(-tilt) + dy * Math.cos(-tilt);
      return (xr * xr) / (a * a) + (yr * yr) / (b * b);
    };

    const inside = (x: number, y: number): boolean => {
      if (tab === 'ridge') return x * x + y * y <= t * t + 1e-9;
      if (tab === 'lasso') return Math.abs(x) + Math.abs(y) <= t + 1e-9;
      const l1 = Math.abs(x) + Math.abs(y);
      const l2 = (x * x + y * y) / 2;
      return alpha * l1 + (1 - alpha) * l2 <= t + 1e-9;
    };

    // If OLS itself is inside the constraint, return it.
    if (inside(olsX, olsY)) return { x: olsX, y: olsY };

    // Sample 720 points on the boundary of the constraint set, pick min loss.
    const n = 720;
    for (let k = 0; k < n; k++) {
      const theta = (2 * Math.PI * k) / n;
      let xb = 0;
      let yb = 0;
      if (tab === 'ridge') {
        xb = t * Math.cos(theta);
        yb = t * Math.sin(theta);
      } else if (tab === 'lasso') {
        // Diamond: parameterize by angle then project to L¹ unit ball.
        const cx = Math.cos(theta);
        const sy = Math.sin(theta);
        const denom = Math.abs(cx) + Math.abs(sy);
        xb = (t * cx) / denom;
        yb = (t * sy) / denom;
      } else {
        // Elastic-net level set: interpolate radius via bisection.
        const cx = Math.cos(theta);
        const sy = Math.sin(theta);
        let lo = 0;
        let hi = 5;
        for (let it = 0; it < 24; it++) {
          const mid = (lo + hi) / 2;
          const x = mid * cx;
          const y = mid * sy;
          const l1 = Math.abs(x) + Math.abs(y);
          const l2 = (x * x + y * y) / 2;
          if (alpha * l1 + (1 - alpha) * l2 < t) lo = mid;
          else hi = mid;
        }
        xb = lo * cx;
        yb = lo * sy;
      }
      candidates.push({ x: xb, y: yb, loss: lossAt(xb, yb) });
    }
    candidates.sort((p1, p2) => p1.loss - p2.loss);
    return { x: candidates[0].x, y: candidates[0].y };
  }, [tab, t, alpha, olsX, olsY, a, b, tilt]);

  // Convert math coordinates [-1, 1] → SVG pixels.
  const toSvgX = (x: number): number => MARGIN.left + ((x + 1) / 2) * sz;
  const toSvgY = (y: number): number => MARGIN.top + ((1 - y) / 2) * sz;

  // Build constraint-region path.
  const constraintPath = useMemo(() => {
    const n = 120;
    const pts: [number, number][] = [];
    for (let k = 0; k < n; k++) {
      const theta = (2 * Math.PI * k) / n;
      if (tab === 'ridge') {
        pts.push([t * Math.cos(theta), t * Math.sin(theta)]);
      } else if (tab === 'lasso') {
        const cx = Math.cos(theta);
        const sy = Math.sin(theta);
        const denom = Math.abs(cx) + Math.abs(sy);
        pts.push([(t * cx) / denom, (t * sy) / denom]);
      } else {
        const cx = Math.cos(theta);
        const sy = Math.sin(theta);
        let lo = 0;
        let hi = 5;
        for (let it = 0; it < 22; it++) {
          const mid = (lo + hi) / 2;
          const x = mid * cx;
          const y = mid * sy;
          const l1 = Math.abs(x) + Math.abs(y);
          const l2 = (x * x + y * y) / 2;
          if (alpha * l1 + (1 - alpha) * l2 < t) lo = mid;
          else hi = mid;
        }
        pts.push([lo * cx, lo * sy]);
      }
    }
    return (
      'M ' +
      pts.map(([x, y]) => `${toSvgX(x).toFixed(2)},${toSvgY(y).toFixed(2)}`).join(' L ') +
      ' Z'
    );
  }, [tab, t, alpha, sz]);

  // Loss-contour ellipses at three levels — rotated.
  const ellipseLevels = [0.4, 1.0, 1.8];
  const lossContours = ellipseLevels.map((scale) => {
    const cx = toSvgX(olsX);
    const cy = toSvgY(olsY);
    const rx = (a * scale * sz) / 2;
    const ry = (b * scale * sz) / 2;
    return { cx, cy, rx, ry };
  });

  const isAtVertex =
    tab === 'lasso' &&
    (Math.abs(constrainedSolution.x) < 1e-2 || Math.abs(constrainedSolution.y) < 1e-2);

  const tabColor =
    tab === 'ridge'
      ? penaltyColors.ridge
      : tab === 'lasso'
      ? penaltyColors.lasso
      : penaltyColors.elasticnet;

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3">
        <div className="text-sm font-semibold text-gray-900">
          Level-set geometry — why ridge shrinks smoothly and lasso selects
        </div>
        <div className="mt-1 text-xs text-gray-600">
          Loss contours of an OLS quadratic (rotated ellipse) intersect the
          penalty's constraint set. Ridge: tangent lands smoothly. Lasso: tangent
          lands at a vertex of the diamond — a coordinate is exactly zero.
        </div>
      </div>

      <div className="mb-3 flex flex-wrap gap-3">
        <div className="flex gap-1 rounded-md border bg-gray-50 p-1" role="tablist">
          {TABS.map((tabDef) => (
            <button
              key={tabDef.key}
              role="tab"
              aria-pressed={tab === tabDef.key}
              onClick={() => setTab(tabDef.key)}
              className={
                'rounded px-3 py-1 text-xs transition-colors ' +
                (tab === tabDef.key
                  ? 'bg-white font-semibold text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900')
              }
            >
              {tabDef.label}
            </button>
          ))}
        </div>

        <label className="flex flex-col text-xs text-gray-700">
          <span>
            constraint t = <span className="font-mono">{t.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.02}
            value={t}
            onChange={(e) => setT(parseFloat(e.target.value))}
            aria-label="constraint radius"
            className="w-44"
          />
        </label>

        {tab === 'elasticnet' && (
          <label className="flex flex-col text-xs text-gray-700">
            <span>
              α = <span className="font-mono">{alpha.toFixed(2)}</span>{' '}
              <span className="text-gray-500">(0=ridge, 1=lasso)</span>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              aria-label="elastic-net mixing alpha"
              className="w-44"
            />
          </label>
        )}
      </div>

      <svg width={panelW} height={PANEL_H} role="img" aria-label="Level set diagram">
        {/* Axes */}
        <line
          x1={toSvgX(-1)}
          x2={toSvgX(1)}
          y1={toSvgY(0)}
          y2={toSvgY(0)}
          stroke="#94a3b8"
          strokeWidth={0.8}
        />
        <line
          x1={toSvgX(0)}
          x2={toSvgX(0)}
          y1={toSvgY(-1)}
          y2={toSvgY(1)}
          stroke="#94a3b8"
          strokeWidth={0.8}
        />

        {/* Constraint region */}
        <path d={constraintPath} fill={FILL_TINT} stroke={tabColor} strokeWidth={2} />

        {/* Loss contours (ellipses, rotated) */}
        {lossContours.map((c, i) => (
          <ellipse
            key={`ell-${i}`}
            cx={c.cx}
            cy={c.cy}
            rx={c.rx}
            ry={c.ry}
            transform={`rotate(${(tilt * 180) / Math.PI} ${c.cx} ${c.cy})`}
            fill="none"
            stroke={ELLIPSE_COLOR}
            strokeWidth={0.8}
            opacity={0.5}
          />
        ))}

        {/* OLS solution */}
        <circle cx={toSvgX(olsX)} cy={toSvgY(olsY)} r={4} fill="#0f172a" />
        <text x={toSvgX(olsX) + 8} y={toSvgY(olsY) - 6} fontSize={10} fill="#0f172a">
          β̂_OLS
        </text>

        {/* Constrained solution */}
        <circle
          cx={toSvgX(constrainedSolution.x)}
          cy={toSvgY(constrainedSolution.y)}
          r={5}
          fill={tabColor}
          stroke="white"
          strokeWidth={1.5}
        />
        <text
          x={toSvgX(constrainedSolution.x) + 8}
          y={toSvgY(constrainedSolution.y) + 14}
          fontSize={10}
          fill={tabColor}
          fontWeight={600}
        >
          β̂(λ)
        </text>

        {/* Axis labels */}
        <text x={toSvgX(1) - 4} y={toSvgY(0) - 4} fontSize={10} fill="#64748b" textAnchor="end">
          β₁
        </text>
        <text x={toSvgX(0) + 4} y={toSvgY(1) + 12} fontSize={10} fill="#64748b">
          β₂
        </text>
      </svg>

      <div className="mt-2 text-xs text-gray-600">
        <strong>{TABS.find((tabDef) => tabDef.key === tab)?.label}:</strong>{' '}
        constraint region {TABS.find((tabDef) => tabDef.key === tab)?.symbol}.{' '}
        {isAtVertex ? (
          <span className="text-amber-700">
            β̂ landed on a vertex — coordinate exactly zero (variable selected
            out).
          </span>
        ) : tab === 'ridge' ? (
          'Ridge produces a smooth shrinkage; both coordinates are nonzero unless OLS is inside the ball.'
        ) : (
          'Slide t smaller to push β̂ onto a vertex of the diamond.'
        )}
      </div>
    </div>
  );
}
