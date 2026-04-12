import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  conditionalExpectationBVN, conditionalVarianceBVN,
} from './shared/moments';
import { mixtureModelPresets } from '../../data/expectation-moments-data';
import type { MixtureSegment } from '../../data/expectation-moments-data';

type ViewMode = 'mixture' | 'bivariate';

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };
const SEGMENT_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444'];

// ── Mixture model computation ─────────────────────────────────────────────

function computeMixtureStats(segments: MixtureSegment[]) {
  // E[X] = Σ wᵢ μᵢ  (tower property)
  const overallMean = segments.reduce((s, seg) => s + seg.weight * seg.mean, 0);

  // E[Var(X|Y)] = Σ wᵢ σᵢ²  (within-group)
  const withinVar = segments.reduce((s, seg) => s + seg.weight * seg.std * seg.std, 0);

  // Var(E[X|Y]) = Σ wᵢ (μᵢ − μ)²  (between-group)
  const betweenVar = segments.reduce(
    (s, seg) => s + seg.weight * (seg.mean - overallMean) * (seg.mean - overallMean),
    0,
  );

  // Var(X) = within + between  (Eve's law)
  const totalVar = withinVar + betweenVar;

  return { overallMean, withinVar, betweenVar, totalVar };
}

// ── Generate samples from mixture for histogram ─────────────────────────

function generateMixtureSamples(segments: MixtureSegment[], n: number = 2000): number[] {
  // Seeded pseudo-random for deterministic visuals
  let seed = 42;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  // Box-Muller transform
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };

  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    const u = rand();
    let cumWeight = 0;
    let segIdx = 0;
    for (let j = 0; j < segments.length; j++) {
      cumWeight += segments[j].weight;
      if (u <= cumWeight) { segIdx = j; break; }
    }
    const seg = segments[segIdx];
    samples.push(seg.mean + seg.std * randn());
  }
  return samples;
}

// ── Histogram binning ───────────────────────────────────────────────────

function binSamples(samples: number[], nBins: number = 40) {
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const range = max - min || 1;
  const binWidth = range / nBins;

  const bins = Array.from({ length: nBins }, (_, i) => ({
    x0: min + i * binWidth,
    x1: min + (i + 1) * binWidth,
    count: 0,
  }));

  for (const s of samples) {
    const idx = Math.min(Math.floor((s - min) / binWidth), nBins - 1);
    bins[idx].count++;
  }

  return bins;
}

// ── Bivariate normal samples ────────────────────────────────────────────

function generateBVNSamples(
  muX: number, muY: number, sigX: number, sigY: number, rho: number, n: number = 500,
) {
  let seed = 123;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0x7fffffff; return seed / 0x7fffffff; };
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };

  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    const z1 = randn();
    const z2 = randn();
    const x = muX + sigX * z1;
    const y = muY + sigY * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
    pts.push({ x, y });
  }
  return pts;
}

export default function LawOfTotalExpectationExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgWidth = Math.max(300, containerWidth - 32);
  const svgHeight = 300;
  const plotW = svgWidth - MARGIN.left - MARGIN.right;
  const plotH = svgHeight - MARGIN.top - MARGIN.bottom;

  const [viewMode, setViewMode] = useState<ViewMode>('mixture');
  const [presetIdx, setPresetIdx] = useState(0);

  // Editable segments (initialized from preset)
  const [segments, setSegments] = useState<MixtureSegment[]>(
    mixtureModelPresets[0].segments.map((s) => ({ ...s })),
  );

  // BVN params
  const [bvnRho, setBvnRho] = useState(0.7);
  const [bvnYSlice, setBvnYSlice] = useState(0);

  const handlePresetChange = useCallback((idx: number) => {
    setPresetIdx(idx);
    setSegments(mixtureModelPresets[idx].segments.map((s) => ({ ...s })));
  }, []);

  const updateSegment = useCallback((idx: number, field: keyof MixtureSegment, value: number) => {
    setSegments((prev) => {
      const next = prev.map((s) => ({ ...s }));
      (next[idx] as Record<string, unknown>)[field] = value;
      // Re-normalize weights
      if (field === 'weight') {
        const total = next.reduce((s, seg) => s + seg.weight, 0);
        if (total > 0) {
          for (const seg of next) seg.weight = seg.weight / total;
        }
      }
      return next;
    });
  }, []);

  // ── Mixture mode data ─────────────────────────────────────────────────

  const mixtureStats = useMemo(() => computeMixtureStats(segments), [segments]);
  const mixtureSamples = useMemo(() => generateMixtureSamples(segments, 2000), [segments]);
  const bins = useMemo(() => binSamples(mixtureSamples, 50), [mixtureSamples]);

  // ── BVN data ──────────────────────────────────────────────────────────

  const bvnData = useMemo(() => {
    const muX = 0, muY = 0, sigX = 1, sigY = 1;
    const pts = generateBVNSamples(muX, muY, sigX, sigY, bvnRho, 500);
    const condMean = conditionalExpectationBVN(muX, muY, sigX, sigY, bvnRho, bvnYSlice);
    const condVar = conditionalVarianceBVN(sigX, bvnRho);
    return { pts, condMean, condVar, muX, muY, sigX, sigY };
  }, [bvnRho, bvnYSlice]);

  // ── Render mixture histogram ──────────────────────────────────────────

  const renderMixtureHistogram = () => {
    const maxCount = Math.max(...bins.map((b) => b.count));
    if (maxCount === 0) return null;

    const xMin = bins[0].x0;
    const xMax = bins[bins.length - 1].x1;

    const xScale = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (c: number) => MARGIN.top + plotH - (c / maxCount) * plotH;

    return (
      <svg width={svgWidth} height={svgHeight} className="block">
        <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={svgWidth - MARGIN.right} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />

        {/* Histogram bars */}
        {bins.map((b, i) => (
          <rect
            key={i}
            x={xScale(b.x0)}
            y={yScale(b.count)}
            width={Math.max(1, xScale(b.x1) - xScale(b.x0) - 1)}
            height={yScale(0) - yScale(b.count)}
            fill="var(--color-primary, #3b82f6)"
            opacity={0.4}
          />
        ))}

        {/* Segment conditional means */}
        {segments.map((seg, i) => (
          <g key={`seg-${i}`}>
            <line
              x1={xScale(seg.mean)} y1={MARGIN.top}
              x2={xScale(seg.mean)} y2={MARGIN.top + plotH}
              stroke={SEGMENT_COLORS[i % SEGMENT_COLORS.length]}
              strokeWidth={2} strokeDasharray="6,3"
            />
            <text x={xScale(seg.mean)} y={MARGIN.top + 14 + i * 14} textAnchor="middle" className="text-xs font-medium" fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]}>
              E[X|{seg.label}]={seg.mean.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Overall mean */}
        <line
          x1={xScale(mixtureStats.overallMean)} y1={MARGIN.top}
          x2={xScale(mixtureStats.overallMean)} y2={MARGIN.top + plotH}
          stroke="var(--color-danger, #ef4444)" strokeWidth={2.5}
        />
        <text x={xScale(mixtureStats.overallMean)} y={MARGIN.top + plotH + 16} textAnchor="middle" className="text-xs font-bold" fill="var(--color-danger, #ef4444)">
          E[X]={mixtureStats.overallMean.toFixed(2)}
        </text>

        {/* x-axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const val = xMin + frac * (xMax - xMin);
          return <text key={frac} x={xScale(val)} y={MARGIN.top + plotH + 30} textAnchor="middle" className="text-xs fill-current opacity-50">{val.toFixed(0)}</text>;
        })}
      </svg>
    );
  };

  // ── Eve's law bar chart ───────────────────────────────────────────────

  const renderEvesLawBar = () => {
    const { withinVar, betweenVar, totalVar } = mixtureStats;
    const maxVal = totalVar * 1.1;
    const barH = 28;
    const chartW = Math.min(300, svgWidth - 100);

    const scale = (v: number) => (v / maxVal) * chartW;

    return (
      <svg width={chartW + 80} height={120} className="block">
        {/* Within */}
        <rect x={60} y={5} width={scale(withinVar)} height={barH} fill={SEGMENT_COLORS[0]} opacity={0.6} rx={3} />
        <text x={55} y={23} textAnchor="end" className="text-xs fill-current opacity-70">Within</text>
        <text x={62 + scale(withinVar)} y={23} className="text-xs font-mono fill-current">{withinVar.toFixed(1)}</text>

        {/* Between */}
        <rect x={60} y={40} width={scale(betweenVar)} height={barH} fill={SEGMENT_COLORS[1]} opacity={0.6} rx={3} />
        <text x={55} y={58} textAnchor="end" className="text-xs fill-current opacity-70">Between</text>
        <text x={62 + scale(betweenVar)} y={58} className="text-xs font-mono fill-current">{betweenVar.toFixed(1)}</text>

        {/* Total */}
        <rect x={60} y={80} width={scale(totalVar)} height={barH} fill="var(--color-danger, #ef4444)" opacity={0.5} rx={3} />
        <text x={55} y={98} textAnchor="end" className="text-xs fill-current opacity-70">Total</text>
        <text x={62 + scale(totalVar)} y={98} className="text-xs font-mono fill-current">{totalVar.toFixed(1)}</text>
      </svg>
    );
  };

  // ── BVN scatter ───────────────────────────────────────────────────────

  const renderBVNScatter = () => {
    const { pts, condMean, muX, muY, sigX, sigY } = bvnData;
    const pad = 3.5;

    const xScale = (v: number) => MARGIN.left + ((v - (muX - pad)) / (2 * pad)) * plotW;
    const yScale = (v: number) => MARGIN.top + plotH - ((v - (muY - pad)) / (2 * pad)) * plotH;

    // Regression line: E[X|Y=y] = muX + rho*(sigX/sigY)*(y - muY)
    const yVals = [-3, 3];
    const regLine = yVals.map((y) => ({
      x: conditionalExpectationBVN(muX, muY, sigX, sigY, bvnRho, y),
      y,
    }));

    return (
      <svg width={svgWidth} height={svgHeight} className="block">
        <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={svgWidth - MARGIN.right} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />

        {/* Scatter points */}
        {pts.map((p, i) => (
          <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={2} fill="var(--color-primary, #3b82f6)" opacity={0.3} />
        ))}

        {/* Regression line */}
        <line
          x1={xScale(regLine[0].x)} y1={yScale(regLine[0].y)}
          x2={xScale(regLine[1].x)} y2={yScale(regLine[1].y)}
          stroke="var(--color-danger, #ef4444)" strokeWidth={2}
        />

        {/* Y-slice line */}
        <line
          x1={MARGIN.left} y1={yScale(bvnYSlice)}
          x2={svgWidth - MARGIN.right} y2={yScale(bvnYSlice)}
          stroke="var(--color-warning, #f59e0b)" strokeDasharray="4,4" strokeWidth={1.5}
        />

        {/* Conditional mean dot */}
        <circle cx={xScale(condMean)} cy={yScale(bvnYSlice)} r={6} fill="var(--color-danger, #ef4444)" />

        {/* Labels */}
        <text x={svgWidth - MARGIN.right - 5} y={yScale(bvnYSlice) - 8} textAnchor="end" className="text-xs font-medium" fill="var(--color-warning, #f59e0b)">
          Y={bvnYSlice.toFixed(1)}
        </text>
        <text x={xScale(condMean) + 10} y={yScale(bvnYSlice) - 8} className="text-xs font-medium" fill="var(--color-danger, #ef4444)">
          E[X|Y]={condMean.toFixed(3)}
        </text>

        {/* Axis labels */}
        <text x={svgWidth / 2} y={svgHeight - 4} textAnchor="middle" className="text-xs fill-current opacity-60">X</text>
        <text x={14} y={svgHeight / 2} textAnchor="middle" className="text-xs fill-current opacity-60" transform={`rotate(-90, 14, ${svgHeight / 2})`}>Y</text>
      </svg>
    );
  };

  return (
    <div ref={containerRef} className="my-8 rounded-xl border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <h3 className="mb-3 text-lg font-semibold">Law of Total Expectation Explorer</h3>

      {/* View mode tabs */}
      <div className="mb-4 flex gap-2">
        {(['mixture', 'bivariate'] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setViewMode(m)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              viewMode === m ? 'text-white' : 'bg-transparent opacity-60 hover:opacity-100'
            }`}
            style={viewMode === m ? { backgroundColor: 'var(--color-primary, #3b82f6)' } : {}}
          >
            {m === 'mixture' ? 'Mixture Model' : 'Bivariate Normal'}
          </button>
        ))}
      </div>

      {viewMode === 'mixture' && (
        <>
          {/* Preset selector */}
          <div className="mb-3">
            <select
              value={presetIdx}
              onChange={(e) => handlePresetChange(Number(e.target.value))}
              className="rounded border px-2 py-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
            >
              {mixtureModelPresets.map((p, i) => (
                <option key={i} value={i}>{p.name}</option>
              ))}
            </select>
            <span className="ml-2 text-xs opacity-60">{mixtureModelPresets[presetIdx].description}</span>
          </div>

          {/* Segment controls */}
          <div className="mb-4 space-y-2">
            {segments.map((seg, i) => (
              <div key={i} className="flex flex-wrap items-center gap-3 text-sm">
                <span className="w-20 font-medium" style={{ color: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}>
                  {seg.label}
                </span>
                <label className="flex items-center gap-1">
                  <span className="opacity-60">w:</span>
                  <input type="range" min={0.05} max={0.95} step={0.05} value={seg.weight} onChange={(e) => updateSegment(i, 'weight', Number(e.target.value))} className="w-16" />
                  <span className="w-8 font-mono text-xs">{seg.weight.toFixed(2)}</span>
                </label>
                <label className="flex items-center gap-1">
                  <span className="opacity-60">μ:</span>
                  <input type="range" min={0} max={200} step={1} value={seg.mean} onChange={(e) => updateSegment(i, 'mean', Number(e.target.value))} className="w-16" />
                  <span className="w-8 font-mono text-xs">{seg.mean.toFixed(0)}</span>
                </label>
                <label className="flex items-center gap-1">
                  <span className="opacity-60">σ:</span>
                  <input type="range" min={1} max={80} step={1} value={seg.std} onChange={(e) => updateSegment(i, 'std', Number(e.target.value))} className="w-16" />
                  <span className="w-8 font-mono text-xs">{seg.std.toFixed(0)}</span>
                </label>
              </div>
            ))}
          </div>

          {/* Histogram */}
          {renderMixtureHistogram()}

          {/* Eve's law decomposition */}
          <div className="mt-4">
            <h4 className="mb-2 text-sm font-semibold opacity-80">Eve's Law: Var(X) = E[Var(X|Y)] + Var(E[X|Y])</h4>
            {renderEvesLawBar()}
          </div>

          {/* Stats readout */}
          <div className="mt-3 grid grid-cols-2 gap-3 rounded-lg p-3 text-sm sm:grid-cols-5" style={{ backgroundColor: 'var(--color-surface, #f8fafc)' }}>
            <div>
              <span className="opacity-60">E[X]</span>
              <div className="font-mono font-semibold">{mixtureStats.overallMean.toFixed(2)}</div>
            </div>
            <div>
              <span className="opacity-60">E[E[X|Y]]</span>
              <div className="font-mono font-semibold">{mixtureStats.overallMean.toFixed(2)}</div>
            </div>
            <div>
              <span className="opacity-60">E[Var(X|Y)]</span>
              <div className="font-mono font-semibold">{mixtureStats.withinVar.toFixed(2)}</div>
            </div>
            <div>
              <span className="opacity-60">Var(E[X|Y])</span>
              <div className="font-mono font-semibold">{mixtureStats.betweenVar.toFixed(2)}</div>
            </div>
            <div>
              <span className="opacity-60">Var(X)</span>
              <div className="font-mono font-semibold">{mixtureStats.totalVar.toFixed(2)}</div>
            </div>
          </div>
        </>
      )}

      {viewMode === 'bivariate' && (
        <>
          <div className="mb-4 flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm">
              <span className="opacity-60">ρ:</span>
              <input type="range" min={-0.99} max={0.99} step={0.01} value={bvnRho} onChange={(e) => setBvnRho(Number(e.target.value))} className="w-32" />
              <span className="w-12 font-mono text-xs">{bvnRho.toFixed(2)}</span>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <span className="opacity-60">Y slice:</span>
              <input type="range" min={-3} max={3} step={0.1} value={bvnYSlice} onChange={(e) => setBvnYSlice(Number(e.target.value))} className="w-32" />
              <span className="w-12 font-mono text-xs">{bvnYSlice.toFixed(1)}</span>
            </label>
          </div>

          {renderBVNScatter()}

          <div className="mt-3 grid grid-cols-3 gap-3 rounded-lg p-3 text-sm" style={{ backgroundColor: 'var(--color-surface, #f8fafc)' }}>
            <div>
              <span className="opacity-60">E[X|Y={bvnYSlice.toFixed(1)}]</span>
              <div className="font-mono font-semibold">{bvnData.condMean.toFixed(4)}</div>
            </div>
            <div>
              <span className="opacity-60">Var(X|Y)</span>
              <div className="font-mono font-semibold">{bvnData.condVar.toFixed(4)}</div>
            </div>
            <div>
              <span className="opacity-60">ρ</span>
              <div className="font-mono font-semibold">{bvnRho.toFixed(2)}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
