import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  FAMILIES,
  glmFit,
  sandwichSE,
  simulateGLM,
  simulateOverdispersedPoisson,
  simulateClusteredBernoulli,
  type GLMFamily,
  type LinkFunction,
} from './shared/regression';
import { standardNormalInvCDF } from './shared/testing';
import { glmFamilyColors } from './shared/colorScales';
import { SANDWICH_COVERAGE_PRESETS } from '../../data/regression-data';

const MARGIN = { top: 24, right: 24, bottom: 44, left: 56 };
const TOP_H = 280;
const BOTTOM_H = 140;
const AXIS_COLOR = '#9CA3AF';
const COVERED_COLOR = '#10B981';
const MISSED_COLOR = '#DC2626';
const NAIVE_COLOR = '#2563EB';
const HC3_COLOR = '#D97706';

type FamilyKey = 'poisson' | 'gamma' | 'bernoulli';
type PresetKey = keyof typeof SANDWICH_COVERAGE_PRESETS;

const PRESET_FOR_FAMILY: Record<FamilyKey, PresetKey> = {
  poisson: 'overdispersedPoisson',
  gamma: 'misspecifiedGamma',
  bernoulli: 'clusteredBinomial',
};

const FAMILY_LABEL: Record<FamilyKey, string> = {
  poisson: 'Poisson (overdispersion)',
  gamma: 'Gamma (variance misspec)',
  bernoulli: 'Bernoulli (clustered)',
};

interface SimRecord {
  iter: number;
  naiveCovered: boolean;
  hc3Covered: boolean;
  naiveCI: { lo: number; hi: number };
  hc3CI: { lo: number; hi: number };
  betaHat: number;
}

export default function SandwichCoverageSimulator() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 760;

  const [familyKey, setFamilyKey] = useState<FamilyKey>('poisson');
  const [misspec, setMisspec] = useState<number>(2.0);
  const [n, setN] = useState<number>(100);
  const [records, setRecords] = useState<SimRecord[]>([]);
  const [running, setRunning] = useState<boolean>(false);
  const runRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  const family: GLMFamily = FAMILIES[familyKey];
  const link: LinkFunction = family.canonicalLink;

  // Build a fixed design (and true β) for each family.
  const design = useMemo(() => {
    const preset = SANDWICH_COVERAGE_PRESETS[PRESET_FOR_FAMILY[familyKey]];
    // Truncate / repeat to match desired n.
    const Xfull = preset.X;
    const X: number[][] = [];
    for (let i = 0; i < n; i++) X.push(Xfull[i % Xfull.length].slice());
    const betaTrue = (preset as { betaTrue: number[] }).betaTrue;
    return { X, betaTrue };
  }, [familyKey, n]);

  // Reset whenever controls change.
  useEffect(() => {
    runRef.current.cancelled = true;
    setRunning(false);
    setRecords([]);
    runRef.current = { cancelled: false };
  }, [familyKey, misspec, n]);

  // Chunked runner — calls glmFit + sandwichSE per simulation, paced via setTimeout.
  const runChunk = (totalToRun: number) => {
    runRef.current.cancelled = false;
    setRunning(true);
    let done = 0;
    let baseSeed = Date.now() % 100000;
    let workingRecords: SimRecord[] = [];
    setRecords((prev) => {
      workingRecords = prev.slice();
      return prev;
    });

    const oneStep = () => {
      if (runRef.current.cancelled) {
        setRunning(false);
        return;
      }
      // Run a small batch per tick to keep UI responsive
      const batch = Math.min(10, totalToRun - done);
      for (let s = 0; s < batch; s++) {
        // Per-attempt seed (PR #25 review): bumping with `done + s` advances
        // the seed even when an iteration is skipped (failed fit / non-
        // convergence), so the next attempt sees fresh data instead of
        // looping on the same failing draw.
        const seed = baseSeed + done + s;
        let yi: number[];
        try {
          if (familyKey === 'poisson') {
            yi = simulateOverdispersedPoisson(design.X, design.betaTrue, misspec, seed);
          } else if (familyKey === 'gamma') {
            yi = simulateGLM(design.X, design.betaTrue, family, link, seed, {
              dispersion: 0.5 * misspec,
            });
          } else {
            // Bernoulli "clustered" — uses simulateClusteredBernoulli so
            // within-cluster latent correlation actually shows up in the
            // sandwich-vs-naive coverage gap. (Previously called
            // simulateGLM, which gave i.i.d. draws and defeated the demo.)
            yi = simulateClusteredBernoulli(design.X, design.betaTrue, seed, {
              clusterSize: 5,
              rho: 0.3,
            });
          }
        } catch {
          continue;
        }

        let fit;
        try {
          fit = glmFit(design.X, yi, family, link, { maxIter: 30 });
        } catch {
          continue;
        }
        if (!fit.converged) continue;

        const z = standardNormalInvCDF(1 - 0.05 / 2);
        const targetJ = 1; // first slope
        const betaHat = fit.beta[targetJ];
        const seNaive = Math.sqrt(Math.max(fit.vcov[targetJ][targetJ], 0));
        let seHC3: number;
        try {
          const ses = sandwichSE(fit, 'HC3');
          seHC3 = ses[targetJ];
        } catch {
          continue;
        }
        const naiveCI = { lo: betaHat - z * seNaive, hi: betaHat + z * seNaive };
        const hc3CI = { lo: betaHat - z * seHC3, hi: betaHat + z * seHC3 };
        const betaTrue = design.betaTrue[targetJ];
        workingRecords.push({
          iter: workingRecords.length,
          naiveCovered: naiveCI.lo <= betaTrue && betaTrue <= naiveCI.hi,
          hc3Covered: hc3CI.lo <= betaTrue && betaTrue <= hc3CI.hi,
          naiveCI,
          hc3CI,
          betaHat,
        });
      }
      done += batch;
      // Snapshot to React.
      setRecords([...workingRecords]);
      if (done < totalToRun && !runRef.current.cancelled) {
        window.setTimeout(oneStep, 30);
      } else {
        setRunning(false);
      }
    };
    oneStep();
  };

  const runMany = (count: number) => runChunk(count);
  const reset = () => {
    runRef.current.cancelled = true;
    setRunning(false);
    setRecords([]);
  };

  // Coverage tallies
  const naiveCov = records.length === 0 ? 0 : records.filter((r) => r.naiveCovered).length / records.length;
  const hc3Cov = records.length === 0 ? 0 : records.filter((r) => r.hc3Covered).length / records.length;

  // Simulation panel (top): visualize the most-recent ~30 sims as horizontal CI bars.
  const showRecent = records.slice(-30);
  const targetBeta = design.betaTrue[1];

  const xRangeTop: [number, number] = useMemo(() => {
    if (showRecent.length === 0) return [targetBeta - 1, targetBeta + 1];
    const all: number[] = [];
    for (const r of showRecent) {
      all.push(r.naiveCI.lo, r.naiveCI.hi, r.hc3CI.lo, r.hc3CI.hi);
    }
    const lo = Math.min(...all, targetBeta);
    const hi = Math.max(...all, targetBeta);
    const pad = (hi - lo) * 0.08 || 0.5;
    return [lo - pad, hi + pad];
  }, [showRecent, targetBeta]);

  // Single chart-width derivation used for the SVG viewBox, axis x-extent,
  // and the scale range — previously these used `width - MARGIN.right` and
  // `chartW` inconsistently, pushing plotted x-coords past the viewBox
  // when the right rail was present (PR #25 review).
  const RAIL_W = 280;
  const chartW = isStacked ? width : width - RAIL_W;
  const topXScale = useMemo(
    () => d3.scaleLinear().domain(xRangeTop).range([MARGIN.left, chartW - MARGIN.right]),
    [xRangeTop, chartW],
  );

  const rowH = Math.max(8, (TOP_H - 20) / Math.max(showRecent.length, 1));

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-surface, #ffffff)',
      }}
    >
      {/* Family selector bar */}
      <div
        className="flex flex-wrap gap-2 border-b px-3 py-2 text-sm"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        {(['poisson', 'gamma', 'bernoulli'] as FamilyKey[]).map((fk) => (
          <button
            key={fk}
            type="button"
            onClick={() => setFamilyKey(fk)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              familyKey === fk
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            <span
              className="mr-1 inline-block h-2 w-2 rounded-full align-middle"
              style={{ backgroundColor: glmFamilyColors[fk] }}
            />
            {FAMILY_LABEL[fk]}
          </button>
        ))}
      </div>

      <div className={`flex gap-3 p-3 ${isStacked ? 'flex-col' : 'flex-row'}`}>
        <div className="flex-1">
          {/* TOP: per-sim CI bars (most recent) */}
          <p className="mb-1 text-xs text-slate-600">
            Latest {showRecent.length} simulations · 95% CIs · green = covers β_true,
            red = misses · upper bar naive, lower bar HC3
          </p>
          <svg
            width="100%"
            viewBox={`0 0 ${chartW} ${MARGIN.top + TOP_H + 30}`}
            style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
          >
            {/* β_true vertical line */}
            <line
              x1={topXScale(targetBeta)}
              x2={topXScale(targetBeta)}
              y1={MARGIN.top}
              y2={MARGIN.top + TOP_H}
              stroke="#111827"
              strokeDasharray="3 3"
            />
            <text
              x={topXScale(targetBeta)}
              y={MARGIN.top - 4}
              textAnchor="middle"
              fontSize={10}
              fill="#111827"
            >
              β_true = {targetBeta.toFixed(2)}
            </text>

            {/* Per-sim bars */}
            {showRecent.map((r, i) => {
              const yBase = MARGIN.top + 10 + i * rowH;
              return (
                <g key={r.iter}>
                  <line
                    x1={topXScale(r.naiveCI.lo)}
                    x2={topXScale(r.naiveCI.hi)}
                    y1={yBase}
                    y2={yBase}
                    stroke={r.naiveCovered ? COVERED_COLOR : MISSED_COLOR}
                    strokeWidth={2}
                    opacity={0.85}
                  />
                  <line
                    x1={topXScale(r.hc3CI.lo)}
                    x2={topXScale(r.hc3CI.hi)}
                    y1={yBase + rowH * 0.45}
                    y2={yBase + rowH * 0.45}
                    stroke={r.hc3Covered ? COVERED_COLOR : MISSED_COLOR}
                    strokeWidth={2}
                    opacity={0.55}
                    strokeDasharray="2 2"
                  />
                </g>
              );
            })}

            {/* X axis */}
            <g transform={`translate(0, ${MARGIN.top + TOP_H})`}>
              <line x1={MARGIN.left} x2={chartW - MARGIN.right} stroke={AXIS_COLOR} />
              {topXScale.ticks(6).map((t) => (
                <g key={t} transform={`translate(${topXScale(t)}, 0)`}>
                  <line y1={0} y2={4} stroke={AXIS_COLOR} />
                  <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
                    {t.toFixed(2)}
                  </text>
                </g>
              ))}
            </g>
            <text
              x={(MARGIN.left + chartW - MARGIN.right) / 2}
              y={MARGIN.top + TOP_H + 28}
              textAnchor="middle"
              fontSize={11}
              fill={AXIS_COLOR}
            >
              β₁ value
            </text>
          </svg>

          {/* BOTTOM: running coverage bars vs target line */}
          <p className="mt-3 mb-1 text-xs text-slate-600">
            Running coverage (after {records.length} sims) · target = 0.95
          </p>
          <svg
            width="100%"
            viewBox={`0 0 ${chartW} ${MARGIN.top + BOTTOM_H + 30}`}
            style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
          >
            {/* Target 0.95 line */}
            <line
              x1={MARGIN.left}
              x2={chartW - MARGIN.right}
              y1={MARGIN.top + BOTTOM_H * (1 - 0.95)}
              y2={MARGIN.top + BOTTOM_H * (1 - 0.95)}
              stroke="#111827"
              strokeDasharray="4 4"
            />
            <text
              x={chartW - MARGIN.right - 4}
              y={MARGIN.top + BOTTOM_H * (1 - 0.95) - 4}
              textAnchor="end"
              fontSize={10}
              fill="#111827"
            >
              target 0.95
            </text>
            {/* Naive bar */}
            <CovBar
              label="Naive"
              x={MARGIN.left + 60}
              value={naiveCov}
              color={NAIVE_COLOR}
            />
            {/* HC3 bar */}
            <CovBar
              label="HC3"
              x={MARGIN.left + 220}
              value={hc3Cov}
              color={HC3_COLOR}
            />
            {/* Y axis */}
            <g>
              <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + BOTTOM_H} stroke={AXIS_COLOR} />
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <g key={t} transform={`translate(${MARGIN.left}, ${MARGIN.top + BOTTOM_H * (1 - t)})`}>
                  <line x1={-4} x2={0} stroke={AXIS_COLOR} />
                  <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
                    {t.toFixed(2)}
                  </text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        {/* Right rail controls */}
        <div
          className="flex flex-col gap-3 rounded-md p-3 text-sm md:w-[260px]"
          style={{
            backgroundColor: 'var(--color-surface-alt, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-slate-700">
              Misspec ratio = {misspec.toFixed(2)}{' '}
              {familyKey === 'bernoulli' && <span className="text-slate-400">(ignored)</span>}
            </span>
            <input
              type="range"
              min={1.0}
              max={4.0}
              step={0.1}
              value={misspec}
              onChange={(e) => setMisspec(parseFloat(e.target.value))}
              disabled={familyKey === 'bernoulli'}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-slate-700">n = {n}</span>
            <input
              type="range"
              list="n-ticks"
              min={30}
              max={1000}
              step={10}
              value={n}
              onChange={(e) => setN(parseInt(e.target.value, 10))}
            />
            <datalist id="n-ticks">
              <option value="30" />
              <option value="100" />
              <option value="300" />
              <option value="1000" />
            </datalist>
          </label>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => runMany(100)}
              disabled={running}
              className="rounded bg-slate-900 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-50"
            >
              Run 100
            </button>
            <button
              type="button"
              onClick={() => runMany(500)}
              disabled={running}
              className="rounded bg-slate-700 px-3 py-1 text-xs text-white hover:bg-slate-500 disabled:opacity-50"
            >
              Run 500
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded bg-slate-100 px-3 py-1 text-xs text-slate-800 hover:bg-slate-200"
            >
              Reset
            </button>
          </div>

          <div className="border-t pt-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-600">After k sims</span>
              <span>{records.length}</span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: NAIVE_COLOR }}>Naive coverage</span>
              <span>{records.length > 0 ? naiveCov.toFixed(3) : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: HC3_COLOR }}>HC3 coverage</span>
              <span>{records.length > 0 ? hc3Cov.toFixed(3) : '—'}</span>
            </div>
          </div>
          <p className="text-[11px] leading-snug text-slate-600">
            Naive Wald undercovers under variance misspecification; HC3 sandwich
            recovers nominal coverage (Topic 22 §22.9 Rem 19).
          </p>
        </div>
      </div>
    </div>
  );
}

function CovBar({
  label,
  x,
  value,
  color,
}: {
  label: string;
  x: number;
  value: number;
  color: string;
}) {
  const h = BOTTOM_H * value;
  return (
    <g>
      <rect
        x={x}
        y={MARGIN.top + (BOTTOM_H - h)}
        width={50}
        height={h}
        fill={color}
        opacity={0.85}
      />
      <text
        x={x + 25}
        y={MARGIN.top + BOTTOM_H + 16}
        textAnchor="middle"
        fontSize={11}
        fill={color}
      >
        {label}
      </text>
      <text
        x={x + 25}
        y={MARGIN.top + (BOTTOM_H - h) - 4}
        textAnchor="middle"
        fontSize={11}
        fill={color}
        fontWeight={600}
      >
        {value.toFixed(3)}
      </text>
    </g>
  );
}
