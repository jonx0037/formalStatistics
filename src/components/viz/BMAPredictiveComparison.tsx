/**
 * BMAPredictiveComparison — interactive for Topic 27 §27.7.
 *
 * Three polynomial models (degrees 1, 2, 3) fit by OLS to a fixed synthetic
 * dataset; user-adjustable weights produce the BMA predictive mixture. Shows
 * per-model predictive fans and the mixture predictive for direct comparison.
 *
 * Pedagogy: watching the BMA predictive widen under uniform weights vs
 * collapse to the cubic under BIC-weighted weights makes the model-uncertainty
 * contribution concrete. The wider band under flat weights is not pessimism —
 * it's honest accounting of cross-model disagreement.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { bayesianColors } from './shared/colorScales';
import {
  bmaPolynomialData,
  bmaWeightPresets,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 20, right: 24, bottom: 40, left: 52 };
const MOBILE_BREAKPOINT = 640;
const DEGREES = [1, 2, 3] as const;

type PolyFit = {
  degree: number;
  coef: number[];      // [β₀, β₁, ..., β_d]
  sigma2: number;      // residual variance estimate
  xtxInv: number[][];  // (X'X)^-1 for predictive variance
};

/** Fit a polynomial of given degree by ordinary least squares on (x, y). */
function fitPoly(x: readonly number[], y: readonly number[], degree: number): PolyFit {
  const n = x.length;
  const k = degree + 1;
  // Design matrix X (n × k)
  const X: number[][] = x.map((xi) => {
    const row = new Array(k);
    let p = 1;
    for (let j = 0; j < k; j++) {
      row[j] = p;
      p *= xi;
    }
    return row;
  });
  // X'X (k × k)
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += X[r][i] * X[r][j];
      XtX[i][j] = s;
    }
  }
  // X'y
  const Xty: number[] = new Array(k).fill(0);
  for (let i = 0; i < k; i++) {
    let s = 0;
    for (let r = 0; r < n; r++) s += X[r][i] * y[r];
    Xty[i] = s;
  }
  // Solve (X'X) β = X'y via Gauss–Jordan with partial pivoting.
  const aug: number[][] = XtX.map((row, i) => [...row, Xty[i]]);
  for (let i = 0; i < k; i++) {
    // Pivot
    let piv = i;
    for (let r = i + 1; r < k; r++) {
      if (Math.abs(aug[r][i]) > Math.abs(aug[piv][i])) piv = r;
    }
    if (piv !== i) [aug[i], aug[piv]] = [aug[piv], aug[i]];
    const d = aug[i][i];
    if (Math.abs(d) < 1e-12) {
      return { degree, coef: new Array(k).fill(0), sigma2: 1, xtxInv: Array.from({ length: k }, () => new Array(k).fill(0)) };
    }
    for (let j = i; j <= k; j++) aug[i][j] /= d;
    for (let r = 0; r < k; r++) {
      if (r === i) continue;
      const f = aug[r][i];
      for (let j = i; j <= k; j++) aug[r][j] -= f * aug[i][j];
    }
  }
  const coef = aug.map((row) => row[k]);

  // Residual variance
  let rss = 0;
  for (let r = 0; r < n; r++) {
    let pred = 0;
    for (let j = 0; j < k; j++) pred += coef[j] * X[r][j];
    rss += (y[r] - pred) ** 2;
  }
  const sigma2 = rss / Math.max(1, n - k);

  // (X'X)^-1 via same augmented matrix approach on [XtX | I]
  const invAug: number[][] = XtX.map((row, i) => {
    const out = [...row];
    for (let j = 0; j < k; j++) out.push(i === j ? 1 : 0);
    return out;
  });
  for (let i = 0; i < k; i++) {
    let piv = i;
    for (let r = i + 1; r < k; r++) {
      if (Math.abs(invAug[r][i]) > Math.abs(invAug[piv][i])) piv = r;
    }
    if (piv !== i) [invAug[i], invAug[piv]] = [invAug[piv], invAug[i]];
    const d = invAug[i][i];
    if (Math.abs(d) < 1e-12) continue;
    for (let j = i; j < 2 * k; j++) invAug[i][j] /= d;
    for (let r = 0; r < k; r++) {
      if (r === i) continue;
      const f = invAug[r][i];
      for (let j = i; j < 2 * k; j++) invAug[r][j] -= f * invAug[i][j];
    }
  }
  const xtxInv: number[][] = invAug.map((row) => row.slice(k));

  return { degree, coef, sigma2, xtxInv };
}

/** Predict mean at x for a polynomial fit. */
function predictMean(fit: PolyFit, x: number): number {
  let pred = 0;
  let p = 1;
  for (let j = 0; j < fit.coef.length; j++) {
    pred += fit.coef[j] * p;
    p *= x;
  }
  return pred;
}

/** Predictive standard deviation at x: √(σ² · (1 + x⁺ (X'X)⁻¹ x)). */
function predictSd(fit: PolyFit, x: number): number {
  const k = fit.coef.length;
  const xVec: number[] = new Array(k);
  let p = 1;
  for (let j = 0; j < k; j++) {
    xVec[j] = p;
    p *= x;
  }
  let q = 0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) q += xVec[i] * fit.xtxInv[i][j] * xVec[j];
  }
  return Math.sqrt(fit.sigma2 * (1 + q));
}

export default function BMAPredictiveComparison() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);        // BIC-weighted default
  const [weights, setWeights] = useState<number[]>([0, 0, 1]); // BIC-weighted initial

  // Apply preset weights
  const applyPreset = (idx: number) => {
    setPresetIdx(idx);
    setWeights([...bmaWeightPresets[idx].weights]);
  };

  // Fit all three models once (data is fixed).
  const fits = useMemo(
    () => DEGREES.map((d) => fitPoly(bmaPolynomialData.x, bmaPolynomialData.y, d)),
    [],
  );

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(780, (width ?? 780) - 16));
  const chartH = isMobile ? 340 : 420;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const xOf = (x: number) => (x / 10) * plotW;

  // y range from data + true f
  const yMinData = Math.min(...bmaPolynomialData.y, 0);
  const yMaxData = Math.max(...bmaPolynomialData.y, 4);
  const yPad = 0.3;
  const yMin = yMinData - yPad;
  const yMax = yMaxData + yPad;
  const yOf = (y: number) => plotH * (1 - (y - yMin) / (yMax - yMin));

  // Compute per-model predictions and BMA mixture on testX grid.
  const predictions = useMemo(() => {
    const totalW = weights.reduce((a, b) => a + b, 0);
    const wNorm = totalW > 0 ? weights.map((w) => w / totalW) : [1 / 3, 1 / 3, 1 / 3];
    return bmaPolynomialData.testX.map((x) => {
      const meanPerModel = fits.map((fit) => predictMean(fit, x));
      const sdPerModel = fits.map((fit) => predictSd(fit, x));
      // BMA predictive mean: Σ w_d * μ_d
      const bmaMean = wNorm.reduce((acc, w, d) => acc + w * meanPerModel[d], 0);
      // BMA predictive variance: Σ w_d (σ²_d + (μ_d - bmaMean)²)  (mixture variance)
      const bmaVar = wNorm.reduce(
        (acc, w, d) => acc + w * (sdPerModel[d] ** 2 + (meanPerModel[d] - bmaMean) ** 2),
        0,
      );
      return {
        x,
        meanPerModel,
        sdPerModel,
        bmaMean,
        bmaSd: Math.sqrt(bmaVar),
      };
    });
  }, [fits, weights]);

  // BMA predictive band (±2 SD)
  const bmaBandTop = predictions
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.x).toFixed(2)} ${yOf(p.bmaMean + 2 * p.bmaSd).toFixed(2)}`)
    .join(' ');
  const bmaBandBot = predictions
    .slice()
    .reverse()
    .map((p) => `L ${xOf(p.x).toFixed(2)} ${yOf(p.bmaMean - 2 * p.bmaSd).toFixed(2)}`)
    .join(' ');
  const bmaBandPath = `${bmaBandTop} ${bmaBandBot} Z`;
  const bmaMeanPath = predictions
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.x).toFixed(2)} ${yOf(p.bmaMean).toFixed(2)}`)
    .join(' ');

  // Per-model mean curves (index-aligned with DEGREES).
  const modelPaths = DEGREES.map((_, idx) =>
    predictions
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.x).toFixed(2)} ${yOf(p.meanPerModel[idx]).toFixed(2)}`)
      .join(' '),
  );

  // True f for reference
  const truePath = bmaPolynomialData.testX
    .map((x, i) => {
      const y = bmaPolynomialData.trueF(x);
      return `${i === 0 ? 'M' : 'L'} ${xOf(x).toFixed(2)} ${yOf(y).toFixed(2)}`;
    })
    .join(' ');

  const modelColors = [bayesianColors.chains[0], bayesianColors.chains[1], bayesianColors.chains[2]];

  // Controlled weight sliders (first two free, third auto-normalizes).
  const setW = (idx: number, val: number) => {
    const w = [...weights];
    w[idx] = Math.max(0, Math.min(1, val));
    // Renormalize remaining so total ≤ 1
    const remaining = Math.max(0, 1 - w[idx]);
    const otherIdxs = [0, 1, 2].filter((i) => i !== idx);
    const otherSum = otherIdxs.reduce((a, i) => a + w[i], 0);
    if (otherSum > 0) {
      otherIdxs.forEach((i) => { w[i] = (w[i] / otherSum) * remaining; });
    } else {
      otherIdxs.forEach((i) => { w[i] = remaining / 2; });
    }
    setWeights(w);
    setPresetIdx(-1);
  };

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Preset selector */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {bmaWeightPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => applyPreset(i)}
            className={`rounded px-3 py-1 text-xs transition ${
              i === presetIdx
                ? 'bg-[var(--color-posterior)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Weight sliders */}
      <div className="mb-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
        {DEGREES.map((d, idx) => (
          <label key={d} className="flex flex-col gap-1">
            <span className="flex items-center justify-between">
              <span style={{ color: modelColors[idx] }}>Degree {d} weight</span>
              <span className="font-mono">{weights[idx].toFixed(3)}</span>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={weights[idx]}
              onChange={(e) => setW(idx, parseFloat(e.target.value))}
              style={{ accentColor: modelColors[idx] }}
            />
          </label>
        ))}
      </div>

      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* y gridlines */}
          {[0, 1, 2, 3, 4].filter((v) => v >= yMin && v <= yMax).map((v) => (
            <g key={v}>
              <line x1={0} y1={yOf(v)} x2={plotW} y2={yOf(v)} stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2 2" />
              <text x={-6} y={yOf(v) + 4} textAnchor="end" fontSize={10} fill="var(--color-text-muted)">{v}</text>
            </g>
          ))}
          {/* x ticks */}
          {[0, 2, 4, 6, 8, 10].map((v) => (
            <g key={v}>
              <line x1={xOf(v)} y1={plotH} x2={xOf(v)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text x={xOf(v)} y={plotH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">{v}</text>
            </g>
          ))}
          {/* BMA predictive band */}
          <path d={bmaBandPath} fill={bayesianColors.posterior} opacity={0.15} />
          {/* Per-model mean curves */}
          {modelPaths.map((p, idx) => (
            <path
              key={idx}
              d={p}
              fill="none"
              stroke={modelColors[idx]}
              strokeWidth={weights[idx] > 0.01 ? 1.2 : 0.6}
              opacity={weights[idx] > 0.01 ? 0.75 : 0.3}
              strokeDasharray="4 2"
            />
          ))}
          {/* True f */}
          <path d={truePath} fill="none" stroke={bayesianColors.true} strokeWidth={1.2} strokeDasharray="2 4" opacity={0.8} />
          {/* BMA mean */}
          <path d={bmaMeanPath} fill="none" stroke={bayesianColors.posterior} strokeWidth={2.5} />
          {/* Data points */}
          {bmaPolynomialData.x.map((xi, i) => (
            <circle key={i} cx={xOf(xi)} cy={yOf(bmaPolynomialData.y[i])} r={3.5} fill={bayesianColors.likelihood} stroke="white" strokeWidth={0.8} />
          ))}
          {/* Axis labels */}
          <text x={plotW / 2} y={plotH + 34} textAnchor="middle" fontSize={11} fill="var(--color-text)">x</text>
          <text x={-38} y={plotH / 2} textAnchor="middle" fontSize={11} fill="var(--color-text)" transform={`rotate(-90 ${-38} ${plotH / 2})`}>y</text>
          {/* Legend */}
          <g transform={`translate(${plotW - 170}, 8)`}>
            <rect x={0} y={0} width={160} height={98} fill="var(--color-surface)" stroke="var(--color-border)" opacity={0.95} />
            {DEGREES.map((d, idx) => (
              <g key={d}>
                <line x1={8} y1={14 + 14 * idx} x2={28} y2={14 + 14 * idx} stroke={modelColors[idx]} strokeDasharray="4 2" strokeWidth={1.5} />
                <text x={34} y={18 + 14 * idx} fontSize={10} fill="var(--color-text)">Degree {d} mean</text>
              </g>
            ))}
            <line x1={8} y1={58} x2={28} y2={58} stroke={bayesianColors.posterior} strokeWidth={2.5} />
            <text x={34} y={62} fontSize={10} fill="var(--color-text)">BMA mean</text>
            <line x1={8} y1={74} x2={28} y2={74} stroke={bayesianColors.true} strokeDasharray="2 4" strokeWidth={1.2} />
            <text x={34} y={78} fontSize={10} fill="var(--color-text)">true f</text>
            <circle cx={18} cy={90} r={3} fill={bayesianColors.likelihood} stroke="white" strokeWidth={0.8} />
            <text x={34} y={94} fontSize={10} fill="var(--color-text)">observed (n=20)</text>
          </g>
        </g>
      </svg>

      <div className="mt-3 text-xs italic" style={{ color: 'var(--color-text-muted)' }}>
        {presetIdx >= 0
          ? bmaWeightPresets[presetIdx].description
          : 'Custom weights — drag sliders to see how BMA predictive band responds to weight changes.'}
      </div>
    </div>
  );
}
