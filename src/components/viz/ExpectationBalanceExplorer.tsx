import { useState, useMemo, useCallback, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfNormal, pdfExponential, pdfUniform,
  trapezoidalIntegral,
} from './shared/distributions';
import {
  expectationDiscrete, varianceDiscrete,
  expectationContinuous, varianceContinuous,
  expectationDiscreteG, stdDev,
} from './shared/moments';
import { expectationPresets } from '../../data/expectation-moments-data';
import type { DiscreteDistribution, ContinuousDistribution } from './shared/moments';

type Mode = 'discrete' | 'continuous' | 'custom';

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

function evalContinuousPDF(name: string, x: number, params: Record<string, number>): number {
  switch (name) {
    case 'Normal': return pdfNormal(x, params.mu, params.sigma2);
    case 'Exponential': return pdfExponential(x, params.lambda);
    case 'Uniform': return pdfUniform(x, params.a, params.b);
    default: return 0;
  }
}

function getContinuousRange(name: string, params: Record<string, number>): [number, number] {
  switch (name) {
    case 'Normal': {
      const s = Math.sqrt(params.sigma2);
      return [params.mu - 4 * s, params.mu + 4 * s];
    }
    case 'Exponential': return [0, 5 / params.lambda + 1];
    case 'Uniform': return [params.a - 0.5, params.b + 0.5];
    default: return [0, 1];
  }
}

export default function ExpectationBalanceExplorer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { width: containerWidth } = useResizeObserver(containerRef);
  const svgWidth = Math.max(300, containerWidth - 32);
  const svgHeight = 280;
  const plotW = svgWidth - MARGIN.left - MARGIN.right;
  const plotH = svgHeight - MARGIN.top - MARGIN.bottom;

  const [mode, setMode] = useState<Mode>('discrete');
  const [discreteIdx, setDiscreteIdx] = useState(0);
  const [continuousIdx, setContinuousIdx] = useState(0);

  // Custom mode state
  const [customValues, setCustomValues] = useState([
    { value: 1, prob: 0.2 },
    { value: 3, prob: 0.5 },
    { value: 7, prob: 0.3 },
  ]);

  // Continuous parameter overrides
  const [contParams, setContParams] = useState<Record<string, number>>({});

  const activeContPreset = expectationPresets.continuous[continuousIdx];
  const mergedContParams = useMemo(
    () => ({ ...activeContPreset.params, ...contParams }),
    [activeContPreset, contParams],
  );

  // Reset params when preset changes
  const handleContinuousChange = useCallback((idx: number) => {
    setContinuousIdx(idx);
    setContParams({});
  }, []);

  // ── Discrete computation ──────────────────────────────────────────────────

  const discreteData = useMemo(() => {
    if (mode === 'discrete') {
      const preset = expectationPresets.discrete[discreteIdx];
      return { values: preset.values, probabilities: preset.probabilities };
    }
    if (mode === 'custom') {
      const totalProb = customValues.reduce((s, c) => s + c.prob, 0);
      const norm = totalProb > 0 ? totalProb : 1;
      return {
        values: customValues.map((c) => c.value),
        probabilities: customValues.map((c) => c.prob / norm),
      };
    }
    return null;
  }, [mode, discreteIdx, customValues]);

  const discreteStats = useMemo(() => {
    if (!discreteData) return null;
    const dist: DiscreteDistribution = discreteData;
    const mean = expectationDiscrete(dist);
    const ex2 = expectationDiscreteG(dist, (x) => x * x);
    const variance = varianceDiscrete(dist);
    return { mean, ex2, variance };
  }, [discreteData]);

  // ── Continuous computation ────────────────────────────────────────────────

  const continuousStats = useMemo(() => {
    if (mode !== 'continuous') return null;
    const range = getContinuousRange(activeContPreset.distribution, mergedContParams);
    const dist: ContinuousDistribution = {
      pdf: (x) => evalContinuousPDF(activeContPreset.distribution, x, mergedContParams),
      support: range,
    };
    const mean = expectationContinuous(dist);
    const ex2 = trapezoidalIntegral(
      (x) => x * x * dist.pdf(x),
      range[0], range[1], 1000,
    );
    const variance = varianceContinuous(dist);
    return { mean, ex2, variance, range };
  }, [mode, activeContPreset, mergedContParams]);

  // ── Render helpers ────────────────────────────────────────────────────────

  const renderDiscreteSVG = () => {
    if (!discreteData || !discreteStats) return null;
    const { values, probabilities } = discreteData;
    const { mean } = discreteStats;

    const xMin = Math.min(...values) - 1;
    const xMax = Math.max(...values) + 1;
    const yMax = Math.max(...probabilities) * 1.2;

    const xScale = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (p: number) => MARGIN.top + plotH - (p / yMax) * plotH;
    const barWidth = Math.max(8, Math.min(40, plotW / values.length / 2));

    return (
      <svg width={svgWidth} height={svgHeight} className="block">
        {/* x-axis */}
        <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={svgWidth - MARGIN.right} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />
        {/* y-axis */}
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />

        {/* Bars */}
        {values.map((v, i) => (
          <rect
            key={i}
            x={xScale(v) - barWidth / 2}
            y={yScale(probabilities[i])}
            width={barWidth}
            height={yScale(0) - yScale(probabilities[i])}
            style={{ fill: 'var(--color-primary, #3b82f6)' }}
            opacity={0.7}
          />
        ))}

        {/* x-axis labels */}
        {values.map((v, i) => (
          <text key={`label-${i}`} x={xScale(v)} y={MARGIN.top + plotH + 18} textAnchor="middle" className="text-xs fill-current opacity-60">{v}</text>
        ))}

        {/* E[X] fulcrum */}
        <line x1={xScale(mean)} y1={MARGIN.top} x2={xScale(mean)} y2={MARGIN.top + plotH} stroke="var(--color-danger, #ef4444)" strokeDasharray="4,4" strokeWidth={2} />
        <polygon
          points={`${xScale(mean)},${MARGIN.top + plotH + 2} ${xScale(mean) - 8},${MARGIN.top + plotH + 14} ${xScale(mean) + 8},${MARGIN.top + plotH + 14}`}
          fill="var(--color-danger, #ef4444)"
        />
        <text x={xScale(mean)} y={MARGIN.top + plotH + 28} textAnchor="middle" className="text-xs font-semibold" fill="var(--color-danger, #ef4444)">
          E[X]={mean.toFixed(2)}
        </text>
      </svg>
    );
  };

  const renderContinuousSVG = () => {
    if (!continuousStats) return null;
    const { mean, range } = continuousStats;
    const [xMin, xMax] = range;

    // Evaluate PDF at many points
    const nPts = 200;
    const step = (xMax - xMin) / nPts;
    const points: { x: number; y: number }[] = [];
    let yMax = 0;
    for (let i = 0; i <= nPts; i++) {
      const x = xMin + i * step;
      const y = evalContinuousPDF(activeContPreset.distribution, x, mergedContParams);
      points.push({ x, y });
      if (y > yMax) yMax = y;
    }
    yMax *= 1.1;

    const xScale = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (p: number) => MARGIN.top + plotH - (p / yMax) * plotH;

    const pathD = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`)
      .join(' ');

    // Fill area
    const areaD = pathD + ` L${xScale(xMax).toFixed(1)},${yScale(0).toFixed(1)} L${xScale(xMin).toFixed(1)},${yScale(0).toFixed(1)} Z`;

    return (
      <svg width={svgWidth} height={svgHeight} className="block">
        <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={svgWidth - MARGIN.right} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke="currentColor" strokeOpacity={0.3} />

        {/* Filled area */}
        <path d={areaD} fill="var(--color-primary, #3b82f6)" opacity={0.15} />
        {/* PDF curve */}
        <path d={pathD} fill="none" stroke="var(--color-primary, #3b82f6)" strokeWidth={2} />

        {/* E[X] fulcrum */}
        <line x1={xScale(mean)} y1={MARGIN.top} x2={xScale(mean)} y2={MARGIN.top + plotH} stroke="var(--color-danger, #ef4444)" strokeDasharray="4,4" strokeWidth={2} />
        <polygon
          points={`${xScale(mean)},${MARGIN.top + plotH + 2} ${xScale(mean) - 8},${MARGIN.top + plotH + 14} ${xScale(mean) + 8},${MARGIN.top + plotH + 14}`}
          fill="var(--color-danger, #ef4444)"
        />
        <text x={xScale(mean)} y={MARGIN.top + plotH + 28} textAnchor="middle" className="text-xs font-semibold" fill="var(--color-danger, #ef4444)">
          E[X]={mean.toFixed(4)}
        </text>

        {/* Axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const val = xMin + frac * (xMax - xMin);
          return (
            <text key={frac} x={xScale(val)} y={MARGIN.top + plotH + 16} textAnchor="middle" className="text-xs fill-current opacity-60">
              {val.toFixed(1)}
            </text>
          );
        })}
      </svg>
    );
  };

  // ── Stats readout ─────────────────────────────────────────────────────────

  const stats = mode === 'continuous' ? continuousStats : discreteStats;

  return (
    <div ref={containerRef} className="my-8 rounded-xl border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <h3 className="mb-3 text-lg font-semibold">Expectation Balance Explorer</h3>

      {/* Mode tabs */}
      <div className="mb-4 flex gap-2">
        {(['discrete', 'continuous', 'custom'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              mode === m
                ? 'text-white'
                : 'bg-transparent opacity-60 hover:opacity-100'
            }`}
            style={mode === m ? { backgroundColor: 'var(--color-primary, #3b82f6)' } : {}}
          >
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="mb-4">
        {mode === 'discrete' && (
          <select
            value={discreteIdx}
            onChange={(e) => setDiscreteIdx(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
          >
            {expectationPresets.discrete.map((p, i) => (
              <option key={i} value={i}>{p.name}</option>
            ))}
          </select>
        )}

        {mode === 'continuous' && (
          <div className="flex flex-wrap items-center gap-4">
            <select
              value={continuousIdx}
              onChange={(e) => handleContinuousChange(Number(e.target.value))}
              className="rounded border px-2 py-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
            >
              {expectationPresets.continuous.map((p, i) => (
                <option key={i} value={i}>{p.name}</option>
              ))}
            </select>
            {Object.entries(activeContPreset.params).map(([key, defaultVal]) => (
              <label key={key} className="flex items-center gap-2 text-sm">
                <span className="opacity-70">{key}:</span>
                <input
                  type="range"
                  min={key === 'lambda' ? 0.1 : key === 'sigma2' ? 0.1 : key === 'a' ? -5 : -5}
                  max={key === 'lambda' ? 5 : key === 'sigma2' ? 5 : key === 'b' ? 10 : 5}
                  step={0.1}
                  value={mergedContParams[key] ?? defaultVal}
                  onChange={(e) => setContParams((p) => ({ ...p, [key]: Number(e.target.value) }))}
                  className="w-24"
                />
                <span className="w-10 text-right font-mono text-xs">{(mergedContParams[key] ?? defaultVal).toFixed(1)}</span>
              </label>
            ))}
          </div>
        )}

        {mode === 'custom' && (
          <div className="space-y-2">
            {customValues.map((cv, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <label className="opacity-70">Value:</label>
                <input
                  type="number"
                  value={cv.value}
                  onChange={(e) => {
                    const next = [...customValues];
                    next[i] = { ...next[i], value: Number(e.target.value) };
                    setCustomValues(next);
                  }}
                  className="w-20 rounded border px-2 py-0.5"
                  style={{ borderColor: 'var(--color-border)' }}
                />
                <label className="opacity-70">Prob:</label>
                <input
                  type="number"
                  value={cv.prob}
                  step={0.05}
                  min={0}
                  max={1}
                  onChange={(e) => {
                    const next = [...customValues];
                    next[i] = { ...next[i], prob: Number(e.target.value) };
                    setCustomValues(next);
                  }}
                  className="w-20 rounded border px-2 py-0.5"
                  style={{ borderColor: 'var(--color-border)' }}
                />
                {customValues.length > 2 && (
                  <button
                    onClick={() => setCustomValues(customValues.filter((_, j) => j !== i))}
                    className="text-xs opacity-50 hover:opacity-100"
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            {customValues.length < 8 && (
              <button
                onClick={() => setCustomValues([...customValues, { value: 0, prob: 0.1 }])}
                className="text-sm opacity-60 hover:opacity-100"
              >
                + Add value
              </button>
            )}
          </div>
        )}
      </div>

      {/* SVG visualization */}
      {(mode === 'discrete' || mode === 'custom') && renderDiscreteSVG()}
      {mode === 'continuous' && renderContinuousSVG()}

      {/* Stats readout */}
      {stats && (
        <div className="mt-4 grid grid-cols-3 gap-4 rounded-lg p-3 text-sm" style={{ backgroundColor: 'var(--color-surface, #f8fafc)' }}>
          <div>
            <span className="opacity-60">E[X]</span>
            <div className="font-mono font-semibold">{stats.mean.toFixed(4)}</div>
          </div>
          <div>
            <span className="opacity-60">E[X²]</span>
            <div className="font-mono font-semibold">{stats.ex2.toFixed(4)}</div>
          </div>
          <div>
            <span className="opacity-60">Var(X)</span>
            <div className="font-mono font-semibold">{stats.variance.toFixed(4)}</div>
          </div>
        </div>
      )}
    </div>
  );
}
