import { useState, useMemo, useCallback, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBernoulli, pmfBinomial, pmfGeometric, pmfPoisson,
  pdfUniform, pdfNormal, pdfExponential,
  trapezoidalIntegral,
} from './shared/distributions';
import {
  discreteDistributionPresets,
  continuousDistributionPresets,
} from '../../data/random-variables-data';
import { distributionColors } from './shared/colorScales';

type Mode = 'discrete' | 'continuous';

// ── PMF evaluators ─────────────────────────────────────────────────────────

function evalPMF(name: string, k: number, params: Record<string, number>): number {
  switch (name) {
    case 'Bernoulli': return pmfBernoulli(k, params.p);
    case 'Binomial': return pmfBinomial(k, params.n, params.p);
    case 'Geometric': return pmfGeometric(k, params.p);
    case 'Poisson': return pmfPoisson(k, params.lambda);
    default: return 0;
  }
}

function getDiscreteRange(name: string, params: Record<string, number>): number[] {
  switch (name) {
    case 'Bernoulli': return [0, 1];
    case 'Binomial': return Array.from({ length: params.n + 1 }, (_, i) => i);
    case 'Geometric': {
      const maxK = Math.min(Math.ceil(5 / params.p) + 1, 40);
      return Array.from({ length: maxK }, (_, i) => i + 1);
    }
    case 'Poisson': {
      const maxK = Math.min(Math.ceil(params.lambda + 4 * Math.sqrt(params.lambda)) + 1, 40);
      return Array.from({ length: maxK + 1 }, (_, i) => i);
    }
    default: return [];
  }
}

// ── PDF evaluators ─────────────────────────────────────────────────────────

function evalPDF(name: string, x: number, params: Record<string, number>): number {
  switch (name) {
    case 'Uniform': return pdfUniform(x, Math.min(params.a, params.b), Math.max(params.a, params.b));
    case 'Normal': return pdfNormal(x, params.mu, params.sigma2);
    case 'Exponential': return pdfExponential(x, params.lambda);
    default: return 0;
  }
}

function getContinuousRange(name: string, params: Record<string, number>): [number, number] {
  switch (name) {
    case 'Uniform': {
      const lo = Math.min(params.a, params.b);
      const hi = Math.max(params.a, params.b);
      return [lo - 0.5, hi + 0.5];
    }
    case 'Normal': {
      const sd = Math.sqrt(params.sigma2);
      return [params.mu - 4 * sd, params.mu + 4 * sd];
    }
    case 'Exponential': return [0, 5 / params.lambda + 1];
    default: return [0, 1];
  }
}

// ── Margins & layout ──────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

export default function PMFPDFExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [mode, setMode] = useState<Mode>('discrete');
  const [discreteIdx, setDiscreteIdx] = useState(1); // Binomial default
  const [continuousIdx, setContinuousIdx] = useState(1); // Normal default
  const [discreteParams, setDiscreteParams] = useState(discreteDistributionPresets[1].params);
  const [continuousParams, setContinuousParams] = useState(continuousDistributionPresets[1].params);
  const [selectedBar, setSelectedBar] = useState<number | null>(null);
  const [dragRange, setDragRange] = useState<[number, number] | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const svgW = Math.max(width, 300);
  const svgH = 300;
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = svgH - MARGIN.top - MARGIN.bottom;

  // ── Discrete data ───────────────────────────────────────────────────────

  const discretePreset = discreteDistributionPresets[discreteIdx];
  const discreteRange = useMemo(
    () => getDiscreteRange(discretePreset.name, discreteParams),
    [discretePreset.name, discreteParams],
  );
  const discreteData = useMemo(
    () => discreteRange.map((k) => ({ k, p: evalPMF(discretePreset.name, k, discreteParams) })),
    [discreteRange, discretePreset.name, discreteParams],
  );
  const maxPMF = useMemo(() => Math.max(...discreteData.map((d) => d.p), 0.01), [discreteData]);

  // ── Continuous data ─────────────────────────────────────────────────────

  const continuousPreset = continuousDistributionPresets[continuousIdx];
  const [xMin, xMax] = useMemo(
    () => getContinuousRange(continuousPreset.name, continuousParams),
    [continuousPreset.name, continuousParams],
  );
  const nPoints = 300;
  const continuousData = useMemo(() => {
    const step = (xMax - xMin) / nPoints;
    return Array.from({ length: nPoints + 1 }, (_, i) => {
      const x = xMin + i * step;
      return { x, y: evalPDF(continuousPreset.name, x, continuousParams) };
    });
  }, [xMin, xMax, continuousPreset.name, continuousParams]);
  const maxPDF = useMemo(() => Math.max(...continuousData.map((d) => d.y), 0.01), [continuousData]);
  const densityExceeds1 = maxPDF > 1;

  // ── Area computation for drag range ─────────────────────────────────────

  const areaInfo = useMemo(() => {
    if (!dragRange || mode !== 'continuous') return null;
    const [a, b] = dragRange[0] < dragRange[1] ? dragRange : [dragRange[1], dragRange[0]];
    const area = trapezoidalIntegral(
      (x) => evalPDF(continuousPreset.name, x, continuousParams),
      a, b, 500,
    );
    return { a, b, area };
  }, [dragRange, mode, continuousPreset.name, continuousParams]);

  // ── Scales ──────────────────────────────────────────────────────────────

  const xScale = useCallback(
    (val: number): number => {
      if (mode === 'discrete') {
        const min = discreteRange[0];
        const max = discreteRange[discreteRange.length - 1];
        const range = max - min || 1;
        return MARGIN.left + ((val - min) / range) * plotW;
      }
      return MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW;
    },
    [mode, discreteRange, xMin, xMax, plotW],
  );

  const yScale = useCallback(
    (val: number): number => {
      const maxVal = mode === 'discrete' ? maxPMF : maxPDF;
      return MARGIN.top + plotH - (val / (maxVal * 1.1)) * plotH;
    },
    [mode, maxPMF, maxPDF, plotH],
  );

  const xInverse = useCallback(
    (px: number): number => {
      return xMin + ((px - MARGIN.left) / plotW) * (xMax - xMin);
    },
    [xMin, xMax, plotW],
  );

  // ── Drag handlers ───────────────────────────────────────────────────────

  const isDragging = useRef(false);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (mode !== 'continuous') return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const xVal = xInverse(px);
      isDragging.current = true;
      setDragRange([xVal, xVal]);
    },
    [mode, xInverse],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDragging.current || mode !== 'continuous') return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const xVal = xInverse(px);
      setDragRange((prev) => prev ? [prev[0], xVal] : null);
    },
    [mode, xInverse],
  );

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  // ── Param update ────────────────────────────────────────────────────────

  const updateDiscreteParam = (key: string, value: number) => {
    setDiscreteParams((prev) => ({ ...prev, [key]: key === 'n' ? Math.round(value) : value }));
    setSelectedBar(null);
  };

  const updateContinuousParam = (key: string, value: number) => {
    setContinuousParams((prev) => ({ ...prev, [key]: value }));
    setDragRange(null);
  };

  const handleDiscreteChange = (idx: number) => {
    setDiscreteIdx(idx);
    setDiscreteParams(discreteDistributionPresets[idx].params);
    setSelectedBar(null);
  };

  const handleContinuousChange = (idx: number) => {
    setContinuousIdx(idx);
    setContinuousParams(continuousDistributionPresets[idx].params);
    setDragRange(null);
  };

  // ── Render ──────────────────────────────────────────────────────────────

  const currentParams = mode === 'discrete' ? discreteParams : continuousParams;
  const currentRanges = mode === 'discrete' ? discretePreset.paramRanges : continuousPreset.paramRanges;

  // Build PDF path
  const pdfPath = useMemo(() => {
    if (mode !== 'continuous') return '';
    return continuousData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');
  }, [mode, continuousData, xScale, yScale]);

  // Build shaded area path for drag selection
  const shadedPath = useMemo(() => {
    if (!areaInfo || mode !== 'continuous') return '';
    const filtered = continuousData.filter((d) => d.x >= areaInfo.a && d.x <= areaInfo.b);
    if (filtered.length < 2) return '';
    const baseline = yScale(0);
    let path = `M${xScale(filtered[0].x).toFixed(1)},${baseline}`;
    for (const d of filtered) {
      path += ` L${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`;
    }
    path += ` L${xScale(filtered[filtered.length - 1].x).toFixed(1)},${baseline} Z`;
    return path;
  }, [areaInfo, mode, continuousData, xScale, yScale]);

  // Discrete bar width
  const barWidth = useMemo(() => {
    if (discreteRange.length <= 1) return plotW * 0.6;
    const min = discreteRange[0];
    const max = discreteRange[discreteRange.length - 1];
    return Math.max(4, Math.min(30, (plotW / (max - min + 1)) * 0.7));
  }, [discreteRange, plotW]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      {/* Mode toggle */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border" style={{ borderColor: 'var(--color-border)' }}>
          <button
            onClick={() => setMode('discrete')}
            className={mode === 'discrete' ? 'px-3 py-1 text-sm font-medium bg-blue-600 text-white' : 'px-3 py-1 text-sm font-medium bg-transparent'}
          >
            Discrete (PMF)
          </button>
          <button
            onClick={() => setMode('continuous')}
            className={mode === 'continuous' ? 'px-3 py-1 text-sm font-medium bg-blue-600 text-white' : 'px-3 py-1 text-sm font-medium bg-transparent'}
          >
            Continuous (PDF)
          </button>
        </div>

        <select
          value={mode === 'discrete' ? discreteIdx : continuousIdx}
          onChange={(e) => {
            const idx = Number(e.target.value);
            mode === 'discrete' ? handleDiscreteChange(idx) : handleContinuousChange(idx);
          }}
          className="rounded border px-2 py-1 text-sm"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {(mode === 'discrete' ? discreteDistributionPresets : continuousDistributionPresets).map((p, i) => (
            <option key={p.name} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      {/* Parameter sliders */}
      <div className="mb-4 flex flex-wrap gap-4">
        {Object.entries(currentRanges).map(([key, [min, max]]) => (
          <div key={key} className="flex items-center gap-2">
            <label className="text-sm font-medium" style={{ minWidth: '20px' }}>
              {key === 'sigma2' ? 'σ²' : key === 'lambda' ? 'λ' : key === 'mu' ? 'μ' : key}
            </label>
            <input
              type="range"
              min={min}
              max={max}
              step={key === 'n' ? 1 : 0.01}
              value={currentParams[key]}
              onChange={(e) => {
                const val = Number(e.target.value);
                mode === 'discrete' ? updateDiscreteParam(key, val) : updateContinuousParam(key, val);
              }}
              className="w-24"
            />
            <span className="text-sm font-mono w-12">
              {key === 'n' ? currentParams[key] : currentParams[key].toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      {/* Warning for density > 1 */}
      {mode === 'continuous' && densityExceeds1 && (
        <div className="mb-3 rounded border border-amber-400 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-900/20 dark:text-amber-200">
          ⚠ Density exceeds 1 (max ≈ {maxPDF.toFixed(2)}) — this is fine! The density is NOT a probability. Only the <strong>area</strong> under the curve gives probabilities.
        </div>
      )}

      {/* SVG Chart */}
      <svg
        ref={svgRef}
        width={svgW}
        height={svgH}
        className="block cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Axes */}
        <line x1={MARGIN.left} y1={yScale(0)} x2={svgW - MARGIN.right} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.3} />

        {/* Y-axis ticks */}
        {[0.25, 0.5, 0.75, 1.0].map((t) => {
          const maxVal = mode === 'discrete' ? maxPMF : maxPDF;
          const val = t * maxVal * 1.1;
          const yPos = yScale(val);
          if (yPos < MARGIN.top) return null;
          return (
            <g key={t}>
              <line x1={MARGIN.left - 4} y1={yPos} x2={MARGIN.left} y2={yPos} stroke="currentColor" strokeOpacity={0.3} />
              <text x={MARGIN.left - 8} y={yPos + 4} textAnchor="end" className="fill-current" style={{ fontSize: '10px' }}>
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Discrete: bars */}
        {mode === 'discrete' && discreteData.map((d) => {
          const barX = xScale(d.k) - barWidth / 2;
          const barY = yScale(d.p);
          const barH = yScale(0) - barY;
          const isSelected = selectedBar === d.k;
          return (
            <g key={d.k} onClick={() => setSelectedBar(d.k)} className="cursor-pointer">
              <rect
                x={barX}
                y={barY}
                width={barWidth}
                height={Math.max(barH, 0)}
                fill={distributionColors.pdf}
                opacity={isSelected ? 0.9 : 0.6}
                stroke={distributionColors.pdf}
                strokeWidth={isSelected ? 2 : 1}
              />
              <text
                x={xScale(d.k)}
                y={yScale(0) + 14}
                textAnchor="middle"
                className="fill-current"
                style={{ fontSize: '10px' }}
              >
                {d.k}
              </text>
            </g>
          );
        })}

        {/* Continuous: shaded area */}
        {shadedPath && (
          <path d={shadedPath} fill={distributionColors.area} opacity={0.5} />
        )}

        {/* Continuous: PDF curve */}
        {mode === 'continuous' && pdfPath && (
          <path d={pdfPath} fill="none" stroke={distributionColors.pdf} strokeWidth={2} />
        )}

        {/* y-axis label */}
        <text
          x={14}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '12px' }}
          transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
        >
          {mode === 'discrete' ? 'P(X = x)' : 'f(x)'}
        </text>
      </svg>

      {/* Readout */}
      <div className="mt-3 text-sm">
        {mode === 'discrete' && selectedBar !== null && (
          <p className="font-mono">
            P(X = {selectedBar}) = {evalPMF(discretePreset.name, selectedBar, discreteParams).toFixed(4)}{' '}
            <span className="text-green-600 dark:text-green-400">← this IS a probability</span>
          </p>
        )}
        {mode === 'continuous' && areaInfo && (
          <p className="font-mono">
            P({areaInfo.a.toFixed(2)} ≤ X ≤ {areaInfo.b.toFixed(2)}) = {areaInfo.area.toFixed(4)}{' '}
            <span className="text-green-600 dark:text-green-400">← the AREA is the probability</span>
          </p>
        )}
        {mode === 'discrete' && selectedBar === null && (
          <p className="text-gray-500 italic">Click a bar to see its probability value</p>
        )}
        {mode === 'continuous' && !areaInfo && (
          <p className="text-gray-500 italic">Click and drag to select an interval and compute the probability (area)</p>
        )}
      </div>
    </div>
  );
}
