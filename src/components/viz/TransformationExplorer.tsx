import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfNormal, pdfUniform, pdfExponential,
} from './shared/distributions';
import { transformationPresets } from '../../data/random-variables-data';
import { distributionColors } from './shared/colorScales';

// ── Source PDF evaluators ──────────────────────────────────────────────────

function evalSourcePDF(source: string, x: number): number {
  switch (source) {
    case 'Normal(0,1)': return pdfNormal(x, 0, 1);
    case 'Uniform(0,1)': return pdfUniform(x, 0, 1);
    case 'Exponential(1)': return pdfExponential(x, 1);
    default: return 0;
  }
}

function getSourceRange(source: string): [number, number] {
  switch (source) {
    case 'Normal(0,1)': return [-4, 4];
    case 'Uniform(0,1)': return [-0.5, 1.5];
    case 'Exponential(1)': return [-0.5, 6];
    default: return [0, 1];
  }
}

// ── Seeded RNG for Monte Carlo ─────────────────────────────────────────────

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

function sampleFromSource(source: string, rng: () => number): number {
  switch (source) {
    case 'Normal(0,1)': {
      // Box-Muller transform
      const u1 = rng();
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    case 'Uniform(0,1)': return rng();
    case 'Exponential(1)': return -Math.log(rng());
    default: return 0;
  }
}

// ── Layout ─────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 30, left: 40 };
const SOURCES = ['Normal(0,1)', 'Uniform(0,1)', 'Exponential(1)'];

export default function TransformationExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [transformIdx, setTransformIdx] = useState(0);
  const [sourceIdx, setSourceIdx] = useState(0);
  const [showMC, setShowMC] = useState(false);
  const [nSamples, setNSamples] = useState(1000);
  const [linearA, setLinearA] = useState(2);
  const [linearB, setLinearB] = useState(1);

  const transform = transformationPresets[transformIdx];
  const source = SOURCES[sourceIdx];

  const compact = width < 600;
  const panelW = compact ? Math.max(width - 32, 260) : Math.floor((width - 48) / 2);
  const panelH = 200;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = panelH - MARGIN.top - MARGIN.bottom;

  // ── Source PDF data ─────────────────────────────────────────────────────

  const [srcMin, srcMax] = useMemo(() => getSourceRange(source), [source]);
  const nPts = 200;

  const sourcePDFData = useMemo(() => {
    const step = (srcMax - srcMin) / nPts;
    return Array.from({ length: nPts + 1 }, (_, i) => {
      const x = srcMin + i * step;
      return { x, y: evalSourcePDF(source, x) };
    });
  }, [source, srcMin, srcMax]);

  const maxSourcePDF = useMemo(() => Math.max(...sourcePDFData.map((d) => d.y), 0.01), [sourcePDFData]);

  // ── Transform function ──────────────────────────────────────────────────

  const gFunc = useCallback(
    (x: number): number => {
      if (transform.hasParams) {
        return transform.gFunc(x, linearA, linearB);
      }
      return transform.gFunc(x);
    },
    [transform, linearA, linearB],
  );

  // ── Transformed PDF via Monte Carlo histogram ───────────────────────────

  const mcData = useMemo(() => {
    if (!showMC) return null;
    const rng = seededRandom(42);
    const samples: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const x = sampleFromSource(source, rng);
      samples.push(gFunc(x));
    }
    return samples;
  }, [showMC, nSamples, source, gFunc]);

  // Compute histogram bins for Y
  const histogram = useMemo(() => {
    if (!mcData) return null;
    const validSamples = mcData.filter((v) => isFinite(v) && Math.abs(v) < 100);
    if (validSamples.length < 2) return null;

    const min = Math.min(...validSamples);
    const max = Math.max(...validSamples);
    const nBins = 40;
    const binWidth = (max - min) / nBins || 1;
    const bins = Array.from({ length: nBins }, (_, i) => ({
      x: min + i * binWidth,
      width: binWidth,
      count: 0,
    }));

    for (const v of validSamples) {
      const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
      if (idx >= 0) bins[idx].count++;
    }

    // Normalize to density
    const total = validSamples.length;
    const maxDensity = Math.max(...bins.map((b) => b.count / (total * binWidth)), 0.01);
    return { bins, min, max, binWidth, maxDensity, total };
  }, [mcData]);

  // ── Y range and theoretical PDF ─────────────────────────────────────────

  const [yMin, yMax] = useMemo(() => {
    if (histogram) return [histogram.min - 0.5, histogram.max + 0.5];
    // Approximate
    const testVals = sourcePDFData.map((d) => gFunc(d.x)).filter((v) => isFinite(v));
    const min = Math.min(...testVals);
    const max = Math.max(...testVals);
    const pad = (max - min) * 0.1 || 1;
    return [min - pad, max + pad];
  }, [histogram, sourcePDFData, gFunc]);

  // ── Scales ──────────────────────────────────────────────────────────────

  const xScaleSrc = useCallback(
    (val: number) => MARGIN.left + ((val - srcMin) / (srcMax - srcMin)) * plotW,
    [srcMin, srcMax, plotW],
  );
  const yScaleSrc = useCallback(
    (val: number) => MARGIN.top + plotH - (val / (maxSourcePDF * 1.1)) * plotH,
    [maxSourcePDF, plotH],
  );

  const xScaleY = useCallback(
    (val: number) => MARGIN.left + ((val - yMin) / (yMax - yMin)) * plotW,
    [yMin, yMax, plotW],
  );

  // ── Source PDF path ─────────────────────────────────────────────────────

  const srcPath = useMemo(() => {
    return sourcePDFData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScaleSrc(d.x).toFixed(1)},${yScaleSrc(d.y).toFixed(1)}`)
      .join(' ');
  }, [sourcePDFData, xScaleSrc, yScaleSrc]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Source:</label>
          <select
            value={sourceIdx}
            onChange={(e) => setSourceIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {SOURCES.map((s, i) => (
              <option key={s} value={i}>{s}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Transform:</label>
          <select
            value={transformIdx}
            onChange={(e) => setTransformIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {transformationPresets.map((t, i) => (
              <option key={t.name} value={i}>{t.name}</option>
            ))}
          </select>
        </div>

        <label className="flex items-center gap-1 text-sm">
          <input
            type="checkbox"
            checked={showMC}
            onChange={(e) => setShowMC(e.target.checked)}
          />
          Monte Carlo
        </label>
      </div>

      {/* Linear transform params */}
      {transform.hasParams && (
        <div className="mb-4 flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">a:</label>
            <input
              type="range"
              min={-3}
              max={3}
              step={0.1}
              value={linearA}
              onChange={(e) => setLinearA(Number(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono w-10">{linearA.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">b:</label>
            <input
              type="range"
              min={-5}
              max={5}
              step={0.1}
              value={linearB}
              onChange={(e) => setLinearB(Number(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono w-10">{linearB.toFixed(1)}</span>
          </div>
        </div>
      )}

      {/* Sample size slider */}
      {showMC && (
        <div className="mb-4 flex items-center gap-3">
          <label className="text-sm font-medium">Samples:</label>
          <input
            type="range"
            min={100}
            max={10000}
            step={100}
            value={nSamples}
            onChange={(e) => setNSamples(Number(e.target.value))}
            className="w-36"
          />
          <span className="text-sm font-mono">{nSamples.toLocaleString()}</span>
        </div>
      )}

      {/* Two-panel display */}
      <div className={compact ? 'space-y-2' : 'flex gap-4'}>
        {/* Left: Source PDF */}
        <div>
          <div className="mb-1 text-center text-xs font-medium">
            PDF of X ~ {source}
          </div>
          <svg width={panelW} height={panelH} className="block">
            <line x1={MARGIN.left} y1={yScaleSrc(0)} x2={panelW - MARGIN.right} y2={yScaleSrc(0)} stroke="currentColor" strokeOpacity={0.2} />
            <path d={srcPath} fill="none" stroke={distributionColors.pdf} strokeWidth={2} />
            <text x={MARGIN.left + plotW / 2} y={panelH - 6} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }}>x</text>
            <text x={10} y={MARGIN.top + plotH / 2} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }} transform={`rotate(-90, 10, ${MARGIN.top + plotH / 2})`}>f(x)</text>
          </svg>
        </div>

        {/* Right: Transformed histogram */}
        <div>
          <div className="mb-1 text-center text-xs font-medium">
            {transform.name}{source === transform.source ? `: ${transform.formula}` : ''}
          </div>
          <svg width={panelW} height={panelH} className="block">
            {/* Baseline */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={panelW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.2}
            />

            {/* Histogram bars */}
            {histogram && histogram.bins.map((bin, i) => {
              const density = bin.count / (histogram.total * histogram.binWidth);
              const barH = (density / histogram.maxDensity) * plotH;
              return (
                <rect
                  key={i}
                  x={xScaleY(bin.x)}
                  y={MARGIN.top + plotH - barH}
                  width={Math.max(1, (bin.width / (yMax - yMin)) * plotW - 1)}
                  height={barH}
                  fill={distributionColors.cdf}
                  opacity={0.4}
                />
              );
            })}

            {/* No MC message */}
            {!showMC && (
              <text x={panelW / 2} y={MARGIN.top + plotH / 2} textAnchor="middle" className="fill-current" style={{ fontSize: '11px', opacity: 0.5 }}>
                Enable Monte Carlo to see Y distribution
              </text>
            )}

            <text x={MARGIN.left + plotW / 2} y={panelH - 6} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }}>y</text>
            <text x={10} y={MARGIN.top + plotH / 2} textAnchor="middle" className="fill-current" style={{ fontSize: '10px' }} transform={`rotate(-90, 10, ${MARGIN.top + plotH / 2})`}>density</text>
          </svg>
        </div>
      </div>

      {/* Summary */}
      <div className="mt-3 text-sm">
        <p>
          <span className="font-medium">Transformation:</span>{' '}
          <span className="font-mono">
            {transform.hasParams
              ? `Y = ${linearA.toFixed(1)}X + ${linearB.toFixed(1)}`
              : transform.name}
          </span>
          {' — '}
          <span className="text-gray-500">{transform.formula}</span>
        </p>
        {showMC && histogram && (
          <p className="mt-1 text-xs text-gray-500">
            {nSamples.toLocaleString()} samples drawn; histogram approximates the PDF of Y
          </p>
        )}
      </div>
    </div>
  );
}
