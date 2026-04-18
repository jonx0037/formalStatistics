import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  logLikelihoodNormal2D,
  logLikelihoodGamma2D,
  profileNuisanceOptimizerNormal,
  profileNuisanceOptimizerGamma,
  profileLikelihoodCI,
  chiSquaredInvCDF,
  tCINormalMean,
} from './shared/testing';
import { seededRandom } from './shared/probability';
import { profilePresets, type ProfilePreset } from '../../data/confidence-intervals-data';

/**
 * ProfileLikelihoodExplorer — visualizes §19.7.
 *
 * Traces the profile log-likelihood ℓ_P(θ) = sup_ψ ℓ(θ, ψ) along the axis of
 * the parameter of interest, draws the χ²₁-threshold band, and shades the
 * resulting profile CI. For the Normal preset, the Wald CI (from the
 * quadratic approximation) and the exact t-CI are overlaid — revealing the
 * asymptotic gap between χ²₁ and t²_{n−1} as a visible width difference.
 */

const MARGIN = { top: 16, right: 20, bottom: 44, left: 56 };
const H = 360;

const C = {
  profile: '#10B981',
  threshold: '#DC2626',
  ciBand: '#A7F3D0',
  waldBand: '#FEF3C7',
  tBand: '#DBEAFE',
  axis: '#64748B',
  grid: '#E2E8F0',
  text: '#1E293B',
  mark: '#475569',
};

/** Generate a synthetic sample for a preset. Deterministic via seededRandom. */
function generateSample(preset: ProfilePreset): number[] {
  const rng = seededRandom(42 + preset.n + Math.round(preset.observedThetaHat * 10));
  const data: number[] = [];
  if (preset.scenario === 'normal-2d') {
    // Normal(μ = θ_hat, σ = nuisance)
    for (let i = 0; i < preset.n; i++) {
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data.push(preset.observedThetaHat + preset.observedNuisance * z);
    }
  } else {
    // Gamma(shape = theta, rate = nuisance) via sum of exponentials for integer-ish shape
    const shape = Math.max(1, Math.round(preset.observedThetaHat));
    for (let i = 0; i < preset.n; i++) {
      let g = 0;
      for (let k = 0; k < shape; k++) g += -Math.log(rng()) / preset.observedNuisance;
      data.push(g);
    }
  }
  return data;
}

export default function ProfileLikelihoodExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = profilePresets[presetIndex];
  const [n, setN] = useState(preset.n);
  const [alpha, setAlpha] = useState(preset.alpha);

  const handlePreset = (idx: number): void => {
    setPresetIndex(idx);
    const p = profilePresets[idx];
    setN(p.n);
    setAlpha(p.alpha);
  };

  const effectivePreset: ProfilePreset = { ...preset, n, alpha };
  const data = useMemo(() => generateSample(effectivePreset), [effectivePreset]);

  const logLik = useMemo(() => {
    if (preset.scenario === 'normal-2d') {
      return (theta: number, psi: number): number => logLikelihoodNormal2D(theta, psi, data);
    }
    return (theta: number, psi: number): number => logLikelihoodGamma2D(theta, psi, data);
  }, [preset.scenario, data]);

  const psiOpt = useMemo(() => {
    if (preset.scenario === 'normal-2d') {
      return (theta: number): number => profileNuisanceOptimizerNormal(theta, data);
    }
    return (theta: number): number => profileNuisanceOptimizerGamma(theta, data);
  }, [preset.scenario, data]);

  // Grid + CI computation.
  const gridN = 220;
  const profileCI = useMemo(() => {
    const thetaHat = preset.scenario === 'normal-2d'
      ? data.reduce((s, x) => s + x, 0) / data.length
      : preset.observedThetaHat;
    return profileLikelihoodCI(
      logLik,
      psiOpt,
      thetaHat,
      preset.thetaRange,
      alpha,
      gridN,
    );
  }, [logLik, psiOpt, preset.scenario, preset.thetaRange, preset.observedThetaHat, alpha, data]);

  const maxLL = profileCI.profileCurve.reduce((m, p) => Math.max(m, p[1]), -Infinity);
  const crit = chiSquaredInvCDF(1 - alpha, 1);
  const threshold = maxLL - crit / 2;

  // Y-range for the profile plot.
  const minLL = profileCI.profileCurve.reduce((m, p) => Math.min(m, p[1]), Infinity);
  const yLo = Math.max(minLL, threshold - (maxLL - threshold) * 0.6);
  const yHi = maxLL + (maxLL - yLo) * 0.1;

  const plotW = Math.min(w, 900);

  const xScale = d3
    .scaleLinear()
    .domain(preset.thetaRange)
    .range([MARGIN.left, plotW - MARGIN.right]);
  const yScale = d3.scaleLinear().domain([yLo, yHi]).range([H - MARGIN.bottom, MARGIN.top]);

  const lineGen = d3
    .line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]));
  const profilePath = lineGen(profileCI.profileCurve) ?? '';

  // Wald approximation for Normal preset: quadratic expansion of ℓ_P at the MLE.
  const waldCIQuad: [number, number] | null = useMemo(() => {
    if (preset.scenario !== 'normal-2d') return null;
    const xbar = data.reduce((s, x) => s + x, 0) / data.length;
    const ss = data.reduce((s, x) => s + (x - xbar) * (x - xbar), 0);
    const sigHat = Math.sqrt(ss / data.length);
    const se = sigHat / Math.sqrt(data.length);
    const z = Math.sqrt(crit);
    return [xbar - z * se, xbar + z * se];
  }, [preset.scenario, data, crit]);

  const exactTCI: [number, number] | null = useMemo(() => {
    if (preset.scenario !== 'normal-2d') return null;
    const { lower, upper } = tCINormalMean(data, alpha);
    return [lower, upper];
  }, [preset.scenario, data, alpha]);

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
            {profilePresets.map((p, i) => (
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
            min={10}
            max={200}
            step={5}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-40"
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
        width={plotW}
        height={H}
        role="img"
        aria-label="Profile likelihood with χ² threshold and CI"
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        {/* Grid */}
        {xScale.ticks(6).map((t) => (
          <line
            key={`xg-${t}`}
            x1={xScale(t)}
            x2={xScale(t)}
            y1={MARGIN.top}
            y2={H - MARGIN.bottom}
            stroke={C.grid}
            strokeWidth={1}
          />
        ))}
        {yScale.ticks(6).map((t) => (
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

        {/* Profile CI shaded band */}
        <rect
          x={xScale(Math.max(preset.thetaRange[0], profileCI.lower))}
          width={Math.max(
            0,
            xScale(Math.min(preset.thetaRange[1], profileCI.upper)) -
              xScale(Math.max(preset.thetaRange[0], profileCI.lower)),
          )}
          y={MARGIN.top}
          height={H - MARGIN.bottom - MARGIN.top}
          fill={C.ciBand}
          opacity={0.45}
        />

        {/* χ² threshold line */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={yScale(threshold)}
          y2={yScale(threshold)}
          stroke={C.threshold}
          strokeWidth={1.6}
          strokeDasharray="5,4"
        />
        <text
          x={plotW - MARGIN.right - 8}
          y={yScale(threshold) - 6}
          textAnchor="end"
          fontSize={11}
          fill={C.threshold}
        >
          ℓ_P(θ̂) − χ²₁₋α / 2
        </text>

        {/* Profile curve */}
        <path d={profilePath} stroke={C.profile} strokeWidth={2} fill="none" />

        {/* Wald-quadratic + exact-t overlay ticks (Normal only) */}
        {waldCIQuad && (
          <g>
            <line
              x1={xScale(waldCIQuad[0])}
              x2={xScale(waldCIQuad[1])}
              y1={H - MARGIN.bottom - 28}
              y2={H - MARGIN.bottom - 28}
              stroke="#D97706"
              strokeWidth={5}
              strokeLinecap="butt"
            />
            <text
              x={xScale((waldCIQuad[0] + waldCIQuad[1]) / 2)}
              y={H - MARGIN.bottom - 34}
              textAnchor="middle"
              fontSize={11}
              fill="#D97706"
            >
              Wald
            </text>
          </g>
        )}
        {exactTCI && (
          <g>
            <line
              x1={xScale(exactTCI[0])}
              x2={xScale(exactTCI[1])}
              y1={H - MARGIN.bottom - 12}
              y2={H - MARGIN.bottom - 12}
              stroke="#2563EB"
              strokeWidth={5}
              strokeLinecap="butt"
            />
            <text
              x={xScale((exactTCI[0] + exactTCI[1]) / 2)}
              y={H - MARGIN.bottom - 18}
              textAnchor="middle"
              fontSize={11}
              fill="#2563EB"
            >
              Exact t
            </text>
          </g>
        )}

        {/* Axes */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={H - MARGIN.bottom}
          y2={H - MARGIN.bottom}
          stroke={C.axis}
        />
        <line
          x1={MARGIN.left}
          x2={MARGIN.left}
          y1={MARGIN.top}
          y2={H - MARGIN.bottom}
          stroke={C.axis}
        />
        {xScale.ticks(6).map((t) => (
          <g key={`xt-${t}`}>
            <text
              x={xScale(t)}
              y={H - MARGIN.bottom + 16}
              textAnchor="middle"
              fontSize={11}
              fill={C.text}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {yScale.ticks(5).map((t) => (
          <g key={`yt-${t}`}>
            <text
              x={MARGIN.left - 6}
              y={yScale(t) + 4}
              textAnchor="end"
              fontSize={11}
              fill={C.text}
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          x={(MARGIN.left + plotW - MARGIN.right) / 2}
          y={H - 6}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
        >
          {preset.scenario === 'normal-2d' ? 'μ (Normal mean)' : 'α (Gamma shape)'}
        </text>
        <text
          x={14}
          y={(MARGIN.top + H - MARGIN.bottom) / 2}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
          transform={`rotate(-90, 14, ${(MARGIN.top + H - MARGIN.bottom) / 2})`}
        >
          ℓ_P(θ)
        </text>
      </svg>

      <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
        <div className="rounded bg-white p-3 shadow-sm" style={{ borderLeft: `4px solid ${C.profile}` }}>
          <div className="mb-1 font-semibold" style={{ color: C.profile }}>
            Profile CI (Wilks)
          </div>
          <div>
            <span className="font-mono">
              [{profileCI.lower.toFixed(3)}, {profileCI.upper.toFixed(3)}]
            </span>{' '}
            — threshold-crossings of the χ²₁ level curve.
          </div>
        </div>
        {waldCIQuad && exactTCI && (
          <div className="rounded bg-white p-3 shadow-sm">
            <div className="mb-1 font-semibold" style={{ color: '#D97706' }}>
              Wald (quadratic) vs Exact t
            </div>
            <div>
              Wald:{' '}
              <span className="font-mono">
                [{waldCIQuad[0].toFixed(3)}, {waldCIQuad[1].toFixed(3)}]
              </span>
              , t-CI:{' '}
              <span className="font-mono">
                [{exactTCI[0].toFixed(3)}, {exactTCI[1].toFixed(3)}]
              </span>
              . The Wald-to-t gap shrinks as 1/n; at large n all three agree.
            </div>
          </div>
        )}
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Profile curve is ℓ_P(θ) = sup_ψ ℓ(θ, ψ). The shaded band is {`{`}θ : 2[ℓ_P(θ̂) − ℓ_P(θ)] ≤ χ²₁₋α{`}`} —
        the CI obtained by inverting the generalized LRT with ψ profiled out (Wilks' theorem applies).
      </p>
    </div>
  );
}
