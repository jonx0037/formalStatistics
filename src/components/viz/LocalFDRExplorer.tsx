/**
 * LocalFDRExplorer — interactive for Topic 27 §27.9.
 *
 * Visualize how BH and local-FDR procedures differ on synthetic two-groups
 * data (Efron 2010): a mixture of null z-scores from N(0, 1) and alternative
 * z-scores from N(μ_alt, σ_alt²). User-adjustable thresholds for both
 * methods; disagreement region shaded.
 *
 * The pedagogical point: BH controls the *expected* false-discovery
 * proportion over the rejection set (tail-FDR); local FDR gives the
 * *posterior probability* that each individual z is null — they answer
 * different questions and can sensibly rank observations differently.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng, localFdr } from './shared/bayes';
import { cdfStdNormal, pdfStdNormal, pdfNormal } from './shared/distributions';
import { bayesianColors } from './shared/colorScales';
import { localFdrPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 20, right: 24, bottom: 44, left: 56 };
const MOBILE_BREAKPOINT = 640;
const Z_MIN = -4;
const Z_MAX = 6;
const HIST_BINS = 50;

/** Sample a standard normal via Box-Muller from an LCG. */
function sampleStdNormal(rng: { random: () => number }): number {
  const u1 = rng.random();
  const u2 = rng.random();
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12))) * Math.cos(2 * Math.PI * u2);
}

export default function LocalFDRExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(1);           // default: moderate signal
  const [bhQ, setBhQ] = useState(0.10);
  const [fdrThresh, setFdrThresh] = useState(0.20);
  const [seed, setSeed] = useState(42);

  const preset = localFdrPresets[presetIdx];

  // Generate z-scores from the two-groups mixture.
  const zScores = useMemo(() => {
    const rng = createSeededRng(seed);
    const m = preset.m;
    const zs: number[] = [];
    for (let i = 0; i < m; i++) {
      const isAlt = rng.random() < preset.piOne;
      if (isAlt) {
        zs.push(preset.muAlt + preset.sigmaAlt * sampleStdNormal(rng));
      } else {
        zs.push(sampleStdNormal(rng));
      }
    }
    return zs;
  }, [preset, seed]);

  // BH rejection threshold (on |z|).
  const bhCutoff = useMemo(() => {
    const m = zScores.length;
    const pVals = zScores.map((z) => 2 * (1 - cdfStdNormal(Math.abs(z))));
    const sorted = pVals.slice().sort((a, b) => a - b);
    let kStar = -1;
    for (let k = 0; k < m; k++) {
      if (sorted[k] <= ((k + 1) / m) * bhQ) kStar = k;
    }
    if (kStar < 0) return Number.POSITIVE_INFINITY;
    const pStar = sorted[kStar];
    // Convert two-sided p-value back to |z|: p = 2(1 - Φ(|z|)) ⇒ |z| = Φ⁻¹(1 - p/2)
    // Use iterative inversion via standard-normal-quantile.
    const zCutoff = inverseTwoSidedP(pStar);
    return zCutoff;
  }, [zScores, bhQ]);

  // Local-FDR rejection boundary (closed-form under theoretical null + known alternative).
  const zGrid = useMemo(
    () =>
      Array.from({ length: 200 }, (_, i) => Z_MIN + (i * (Z_MAX - Z_MIN)) / 199),
    [],
  );
  const fdrGrid = useMemo(() => {
    return localFdr({
      zScores,
      zGrid,
      theoreticalNull: true,
      alternative: {
        piOne: preset.piOne,
        muAlt: preset.muAlt,
        sigmaAlt: preset.sigmaAlt,
      },
    }).fdrGrid;
  }, [preset, zGrid, zScores]);

  // Find local-FDR threshold crossing (rightmost z where fdr = threshold)
  const fdrCutoff = useMemo(() => {
    for (let i = zGrid.length - 1; i > 0; i--) {
      if (fdrGrid[i - 1] > fdrThresh && fdrGrid[i] <= fdrThresh) {
        return zGrid[i];
      }
    }
    return Number.POSITIVE_INFINITY;
  }, [fdrGrid, zGrid, fdrThresh]);

  // Histogram binning.
  const hist = useMemo(() => {
    const bins = new Array(HIST_BINS).fill(0);
    const binW = (Z_MAX - Z_MIN) / HIST_BINS;
    for (const z of zScores) {
      if (z < Z_MIN || z >= Z_MAX) continue;
      const i = Math.floor((z - Z_MIN) / binW);
      bins[Math.min(HIST_BINS - 1, i)] += 1;
    }
    const total = zScores.length * binW;
    return bins.map((c, i) => ({
      z: Z_MIN + (i + 0.5) * binW,
      density: c / total,
    }));
  }, [zScores]);

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(780, (width ?? 780) - 16));
  const chartH = isMobile ? 300 : 380;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const xOf = (z: number) => ((z - Z_MIN) / (Z_MAX - Z_MIN)) * plotW;
  const yMax = Math.max(...hist.map((h) => h.density), 0.5);
  const yOf = (d: number) => plotH * (1 - d / yMax);

  const histBars = hist.map((h, i) => {
    const xStart = xOf(h.z - (Z_MAX - Z_MIN) / (2 * HIST_BINS));
    const xEnd = xOf(h.z + (Z_MAX - Z_MIN) / (2 * HIST_BINS));
    return (
      <rect
        key={i}
        x={xStart}
        y={yOf(h.density)}
        width={xEnd - xStart - 0.5}
        height={plotH - yOf(h.density)}
        fill="var(--color-surface)"
        stroke="var(--color-border)"
        strokeWidth={0.5}
      />
    );
  });

  // Overlaid mixture density curve
  const nullPath = zGrid
    .map((z, i) => {
      const d = (1 - preset.piOne) * pdfStdNormal(z);
      return `${i === 0 ? 'M' : 'L'} ${xOf(z).toFixed(2)} ${yOf(d).toFixed(2)}`;
    })
    .join(' ');
  const altPath = zGrid
    .map((z, i) => {
      const d = preset.piOne * pdfNormal(z, preset.muAlt, preset.sigmaAlt * preset.sigmaAlt);
      return `${i === 0 ? 'M' : 'L'} ${xOf(z).toFixed(2)} ${yOf(d).toFixed(2)}`;
    })
    .join(' ');
  const mixPath = zGrid
    .map((z, i) => {
      const d =
        (1 - preset.piOne) * pdfStdNormal(z)
        + preset.piOne * pdfNormal(z, preset.muAlt, preset.sigmaAlt * preset.sigmaAlt);
      return `${i === 0 ? 'M' : 'L'} ${xOf(z).toFixed(2)} ${yOf(d).toFixed(2)}`;
    })
    .join(' ');

  const bhCutoffX = Number.isFinite(bhCutoff) ? xOf(bhCutoff) : plotW;
  const fdrCutoffX = Number.isFinite(fdrCutoff) ? xOf(fdrCutoff) : plotW;

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {localFdrPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => setPresetIdx(i)}
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
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded bg-[var(--color-surface)] px-3 py-1 text-xs hover:bg-[var(--color-surface-hover)]"
          style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
        >
          New seed ({seed})
        </button>
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>BH q threshold</span>
            <span className="font-mono">{bhQ.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={bhQ}
            onChange={(e) => setBhQ(parseFloat(e.target.value))}
            className="accent-[var(--color-mle)]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Local-FDR threshold</span>
            <span className="font-mono">{fdrThresh.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={fdrThresh}
            onChange={(e) => setFdrThresh(parseFloat(e.target.value))}
            className="accent-[var(--color-posterior)]"
          />
        </label>
      </div>

      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Rejection region shading: BH (light mle) */}
          {Number.isFinite(bhCutoff) && (
            <rect
              x={bhCutoffX}
              y={0}
              width={plotW - bhCutoffX}
              height={plotH}
              fill={bayesianColors.mle}
              opacity={0.15}
            />
          )}
          {Number.isFinite(fdrCutoff) && (
            <rect
              x={fdrCutoffX}
              y={0}
              width={plotW - fdrCutoffX}
              height={plotH}
              fill={bayesianColors.posterior}
              opacity={0.15}
            />
          )}
          {/* Histogram */}
          {histBars}
          {/* Null component */}
          <path d={nullPath} fill="none" stroke={bayesianColors.prior} strokeWidth={1.5} strokeDasharray="4 3" />
          {/* Alt component */}
          <path d={altPath} fill="none" stroke={bayesianColors.likelihood} strokeWidth={1.5} strokeDasharray="4 3" />
          {/* Mixture */}
          <path d={mixPath} fill="none" stroke={bayesianColors.posterior} strokeWidth={2} />
          {/* BH cutoff line */}
          {Number.isFinite(bhCutoff) && (
            <line
              x1={bhCutoffX}
              y1={0}
              x2={bhCutoffX}
              y2={plotH}
              stroke={bayesianColors.mle}
              strokeWidth={2}
            />
          )}
          {Number.isFinite(fdrCutoff) && (
            <line
              x1={fdrCutoffX}
              y1={0}
              x2={fdrCutoffX}
              y2={plotH}
              stroke={bayesianColors.posterior}
              strokeWidth={2}
              strokeDasharray="5 3"
            />
          )}
          {/* X ticks */}
          {[-4, -2, 0, 2, 4, 6].filter((t) => t >= Z_MIN && t <= Z_MAX).map((t) => (
            <g key={t}>
              <line x1={xOf(t)} y1={plotH} x2={xOf(t)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text x={xOf(t)} y={plotH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">
                {t}
              </text>
            </g>
          ))}
          {/* Axis labels */}
          <text x={plotW / 2} y={plotH + 34} textAnchor="middle" fontSize={11} fill="var(--color-text)">
            z-score
          </text>
          <text
            x={-42}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-42} ${plotH / 2})`}
          >
            density
          </text>
          {/* Legend */}
          <g transform={`translate(${plotW - 150}, 10)`}>
            <rect x={0} y={0} width={140} height={80} fill="var(--color-surface)" stroke="var(--color-border)" opacity={0.95} />
            <line x1={8} y1={14} x2={28} y2={14} stroke={bayesianColors.prior} strokeDasharray="4 3" strokeWidth={1.5} />
            <text x={34} y={18} fontSize={10} fill="var(--color-text)">null ((1−π₁) φ)</text>
            <line x1={8} y1={32} x2={28} y2={32} stroke={bayesianColors.likelihood} strokeDasharray="4 3" strokeWidth={1.5} />
            <text x={34} y={36} fontSize={10} fill="var(--color-text)">alt (π₁ f₁)</text>
            <line x1={8} y1={50} x2={28} y2={50} stroke={bayesianColors.posterior} strokeWidth={2} />
            <text x={34} y={54} fontSize={10} fill="var(--color-text)">mixture</text>
            <line x1={8} y1={68} x2={28} y2={68} stroke={bayesianColors.mle} strokeWidth={2} />
            <text x={34} y={72} fontSize={10} fill="var(--color-text)">BH cutoff</text>
          </g>
        </g>
      </svg>

      <div className="mt-3 space-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        <div>
          <span className="font-semibold" style={{ color: bayesianColors.mle }}>BH |z| cutoff:</span>{' '}
          {Number.isFinite(bhCutoff) ? `±${bhCutoff.toFixed(3)}` : 'no rejections'}
        </div>
        <div>
          <span className="font-semibold" style={{ color: bayesianColors.posterior }}>Local-FDR z cutoff (right tail):</span>{' '}
          {Number.isFinite(fdrCutoff) ? fdrCutoff.toFixed(3) : 'no rejections'}
        </div>
        <div className="mt-2 italic">{preset.description}</div>
      </div>
    </div>
  );
}

/** Invert the two-sided p-value p = 2 (1 − Φ(|z|)) via binary search on |z|. */
function inverseTwoSidedP(p: number): number {
  if (!(p > 0 && p < 1)) return Number.POSITIVE_INFINITY;
  let lo = 0;
  let hi = 10;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    const pMid = 2 * (1 - cdfStdNormal(mid));
    if (pMid > p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}
