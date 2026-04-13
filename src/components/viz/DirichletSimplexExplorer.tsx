import { useState, useMemo, useCallback } from 'react';
import { sampleDirichlet, expectationDirichlet, varianceDirichlet } from './shared/distributions';
import { useResizeObserver } from './shared/useResizeObserver';
import { dirichletPresets, ternaryCoords, simplexVertices } from '../../data/multivariate-distributions-data';

type Mode = 'explore' | 'conjugate';

const NUM_SAMPLES = 300;
const PADDING = 30;
const LABEL_OFFSET = 18;

// ── Multinomial sampling via sequential binomials ─────────────────────────

function sampleMultinomial(n: number, p: number[]): number[] {
  const k = p.length;
  const x = new Array(k).fill(0);
  let remaining = n;
  let pRemaining = 1;
  for (let j = 0; j < k - 1; j++) {
    const prob = pRemaining > 0 ? p[j] / pRemaining : 0;
    for (let i = 0; i < remaining; i++) {
      if (Math.random() < prob) x[j]++;
    }
    remaining -= x[j];
    pRemaining -= p[j];
  }
  x[k - 1] = remaining;
  return x;
}

// ── Simplex 2D projection helpers ─────────────────────────────────────────

function projectToSvg(
  p: [number, number, number],
  scale: number,
  offsetX: number,
  offsetY: number,
): { x: number; y: number } {
  const tc = ternaryCoords(p);
  return {
    x: offsetX + tc.x * scale,
    y: offsetY - tc.y * scale,  // SVG y-axis is inverted
  };
}

function starPath(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const angle = (Math.PI / 2) + (i * Math.PI) / 5;
    const radius = i % 2 === 0 ? r : r * 0.4;
    pts.push(`${cx + radius * Math.cos(angle)},${cy - radius * Math.sin(angle)}`);
  }
  return pts.join(' ');
}

// ── Component ─────────────────────────────────────────────────────────────

export default function DirichletSimplexExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // State
  const [mode, setMode] = useState<Mode>('explore');
  const [alpha, setAlpha] = useState([1, 1, 1]);
  const [observedData, setObservedData] = useState<number[]>([0, 0, 0]);
  const [totalObservations, setTotalObservations] = useState(0);
  const [sampleSeed, setSampleSeed] = useState(0);

  // Responsive layout
  const containerW = Math.max(280, (width || 600) - 16);
  const isNarrow = containerW < 600;
  const simplexSize = isNarrow ? containerW - 20 : Math.min(380, Math.floor(containerW * 0.58));
  const svgH = simplexSize + 2 * PADDING + 20;
  const svgW = simplexSize + 2 * PADDING;
  const scale = simplexSize - 2 * LABEL_OFFSET;
  const offsetX = PADDING + LABEL_OFFSET;
  const offsetY = svgH - PADDING - 10;

  // Derived alpha for the posterior in conjugate mode
  const posteriorAlpha = useMemo(
    () => alpha.map((a, i) => a + observedData[i]),
    [alpha, observedData],
  );

  // Active alpha: in conjugate mode with data, use posterior; otherwise use prior
  const hasData = totalObservations > 0;

  // Samples for explore mode
  const exploreSamples = useMemo(() => {
    void sampleSeed; // depend on seed for re-generation
    return Array.from({ length: NUM_SAMPLES }, () =>
      sampleDirichlet(alpha) as [number, number, number],
    );
  }, [alpha, sampleSeed]);

  // Samples for conjugate mode
  const priorSamples = useMemo(() => {
    void sampleSeed;
    return Array.from({ length: NUM_SAMPLES }, () =>
      sampleDirichlet(alpha) as [number, number, number],
    );
  }, [alpha, sampleSeed]);

  const posteriorSamples = useMemo(() => {
    if (!hasData) return [];
    void sampleSeed;
    return Array.from({ length: NUM_SAMPLES }, () =>
      sampleDirichlet(posteriorAlpha) as [number, number, number],
    );
  }, [posteriorAlpha, hasData, sampleSeed]);

  // Statistics
  const alpha0 = alpha.reduce((s, a) => s + a, 0);
  const priorMean = expectationDirichlet(alpha);
  const posteriorMean = hasData ? expectationDirichlet(posteriorAlpha) : null;
  const posteriorAlpha0 = posteriorAlpha.reduce((s, a) => s + a, 0);
  const mle = hasData
    ? observedData.map((x) => x / totalObservations)
    : null;
  const exploreVar = varianceDirichlet(alpha);

  // Project a simplex point to SVG coordinates
  const project = useCallback(
    (p: [number, number, number]) => projectToSvg(p, scale, offsetX, offsetY),
    [scale, offsetX, offsetY],
  );

  // Triangle vertices in SVG space
  const triVertices = simplexVertices.map((v) =>
    project([
      v.label === 'p\u2081' ? 1 : 0,
      v.label === 'p\u2082' ? 1 : 0,
      v.label === 'p\u2083' ? 1 : 0,
    ] as [number, number, number]),
  );
  const trianglePath = triVertices.map((v, i) => `${i === 0 ? 'M' : 'L'}${v.x},${v.y}`).join(' ') + 'Z';

  // Label positions (slightly outside the triangle)
  const labelPositions = [
    { ...triVertices[0], dx: -12, dy: 14, label: 'p\u2081' },
    { ...triVertices[1], dx: 6, dy: 14, label: 'p\u2082' },
    { ...triVertices[2], dx: 0, dy: -10, label: 'p\u2083' },
  ];

  // Handlers
  const handleAlphaChange = (index: number, value: number) => {
    setAlpha((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  const handlePreset = (preset: typeof dirichletPresets[number]) => {
    setAlpha([...preset.alpha]);
    setSampleSeed((s) => s + 1);
  };

  const handleObserveData = () => {
    const currentMean = hasData ? expectationDirichlet(posteriorAlpha) : priorMean;
    const x = sampleMultinomial(10, currentMean);
    setObservedData((prev) => prev.map((d, i) => d + x[i]));
    setTotalObservations((prev) => prev + 10);
    setSampleSeed((s) => s + 1);
  };

  const handleReset = () => {
    setObservedData([0, 0, 0]);
    setTotalObservations(0);
    setSampleSeed((s) => s + 1);
  };

  const handleModeChange = (m: Mode) => {
    setMode(m);
    handleReset();
  };

  // Format a number for display
  const fmt = (v: number, d = 4) => v.toFixed(d);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Dirichlet Simplex Explorer
      </div>

      {/* Mode toggle */}
      <div className="flex justify-center gap-2 mb-3">
        {(['explore', 'conjugate'] as const).map((m) => (
          <button
            key={m}
            onClick={() => handleModeChange(m)}
            className="rounded px-3 py-1 text-xs font-medium transition-colors"
            style={{
              background: mode === m ? '#2563eb' : 'var(--color-bg)',
              color: mode === m ? '#fff' : 'currentColor',
              borderWidth: 1,
              borderStyle: 'solid',
              borderColor: mode === m ? '#2563eb' : 'var(--color-border)',
            }}
          >
            {m === 'explore' ? 'Explore' : 'Conjugate Updating'}
          </button>
        ))}
      </div>

      {/* Alpha sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {alpha.map((a, i) => (
          <label key={i} className="flex items-center gap-2 text-xs">
            <span className="font-medium">{`\u03B1${'\u2081\u2082\u2083'[i]}`}</span>
            <input
              type="range"
              min={0.1}
              max={20}
              step={0.1}
              value={a}
              onChange={(e) => {
                handleAlphaChange(i, Number(e.target.value));
                setSampleSeed((s) => s + 1);
              }}
              className="w-24"
            />
            <span className="w-8 tabular-nums text-right">{a.toFixed(1)}</span>
          </label>
        ))}
      </div>

      {/* Preset buttons (explore mode) or data buttons (conjugate mode) */}
      <div className="flex flex-wrap gap-2 justify-center mb-4">
        {mode === 'explore' ? (
          dirichletPresets.map((preset) => (
            <button
              key={preset.name}
              onClick={() => handlePreset(preset)}
              className="rounded border px-2 py-1 text-xs transition-colors hover:bg-blue-50 dark:hover:bg-blue-950"
              style={{ borderColor: 'var(--color-border)' }}
            >
              {preset.name}
            </button>
          ))
        ) : (
          <>
            <button
              onClick={handleObserveData}
              className="rounded px-3 py-1 text-xs font-medium text-white transition-colors"
              style={{ background: '#7c3aed' }}
            >
              Observe Data (n = 10)
            </button>
            <button
              onClick={handleReset}
              className="rounded border px-3 py-1 text-xs font-medium transition-colors hover:bg-gray-50 dark:hover:bg-gray-900"
              style={{ borderColor: 'var(--color-border)' }}
            >
              Reset
            </button>
          </>
        )}
      </div>

      {/* Main panel: simplex + statistics side by side */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* Simplex triangle SVG */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: svgW }}>
          <svg width={svgW} height={svgH} className="block mx-auto">
            {/* Triangle outline */}
            <path
              d={trianglePath}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeWidth={1.5}
            />

            {/* Explore mode: all samples in blue */}
            {mode === 'explore' && exploreSamples.map((p, i) => {
              const pt = project(p);
              return (
                <circle
                  key={i}
                  cx={pt.x}
                  cy={pt.y}
                  r={2.5}
                  fill="rgba(59, 130, 246, 0.25)"
                />
              );
            })}

            {/* Conjugate mode: prior samples (light) */}
            {mode === 'conjugate' && priorSamples.map((p, i) => {
              const pt = project(p);
              return (
                <circle
                  key={`prior-${i}`}
                  cx={pt.x}
                  cy={pt.y}
                  r={2.5}
                  fill="rgba(59, 130, 246, 0.15)"
                />
              );
            })}

            {/* Conjugate mode: posterior samples (darker) */}
            {mode === 'conjugate' && posteriorSamples.map((p, i) => {
              const pt = project(p);
              return (
                <circle
                  key={`post-${i}`}
                  cx={pt.x}
                  cy={pt.y}
                  r={2.5}
                  fill="rgba(139, 92, 246, 0.25)"
                />
              );
            })}

            {/* Conjugate mode: prior mean dot */}
            {mode === 'conjugate' && (() => {
              const pt = project(priorMean as [number, number, number]);
              return (
                <g>
                  <circle cx={pt.x} cy={pt.y} r={5} fill="#2563eb" stroke="#fff" strokeWidth={1.5} />
                  <text
                    x={pt.x + 8}
                    y={pt.y - 8}
                    fontSize={10}
                    fill="#2563eb"
                    fontWeight={600}
                  >
                    Prior mean
                  </text>
                </g>
              );
            })()}

            {/* Conjugate mode: posterior mean dot */}
            {mode === 'conjugate' && posteriorMean && (() => {
              const pt = project(posteriorMean as [number, number, number]);
              return (
                <g>
                  <circle cx={pt.x} cy={pt.y} r={5} fill="#7c3aed" stroke="#fff" strokeWidth={1.5} />
                  <text
                    x={pt.x + 8}
                    y={pt.y + 4}
                    fontSize={10}
                    fill="#7c3aed"
                    fontWeight={600}
                  >
                    Posterior mean
                  </text>
                </g>
              );
            })()}

            {/* Conjugate mode: MLE star */}
            {mode === 'conjugate' && mle && (() => {
              const pt = project(mle as [number, number, number]);
              return (
                <g>
                  <polygon
                    points={starPath(pt.x, pt.y, 7)}
                    fill="#dc2626"
                    stroke="#fff"
                    strokeWidth={0.8}
                  />
                  <text
                    x={pt.x + 9}
                    y={pt.y + 12}
                    fontSize={10}
                    fill="#dc2626"
                    fontWeight={600}
                  >
                    MLE
                  </text>
                </g>
              );
            })()}

            {/* Explore mode: mean dot */}
            {mode === 'explore' && (() => {
              const pt = project(priorMean as [number, number, number]);
              return (
                <g>
                  <circle cx={pt.x} cy={pt.y} r={5} fill="#2563eb" stroke="#fff" strokeWidth={1.5} />
                  <text
                    x={pt.x + 8}
                    y={pt.y - 8}
                    fontSize={10}
                    fill="#2563eb"
                    fontWeight={600}
                  >
                    E[P]
                  </text>
                </g>
              );
            })()}

            {/* Vertex labels */}
            {labelPositions.map((lp, i) => (
              <text
                key={i}
                x={lp.x + lp.dx}
                y={lp.y + lp.dy}
                fontSize={12}
                fill="currentColor"
                textAnchor="middle"
                fontWeight={500}
              >
                {lp.label}
              </text>
            ))}
          </svg>
        </div>

        {/* Side panel: statistics */}
        <div
          className={`flex-1 min-w-0 rounded-lg border p-3 flex flex-col ${isNarrow ? 'w-full' : ''}`}
          style={{ borderColor: 'var(--color-border)', minWidth: isNarrow ? undefined : 200 }}
        >
          {mode === 'explore' ? (
            /* Explore mode statistics */
            <>
              <div className="text-xs font-semibold mb-2 text-center opacity-70">
                Dirichlet Statistics
              </div>
              <div className="space-y-2 text-xs">
                <div>
                  <span className="font-medium">{'\u03B1'}</span>{' = '}
                  ({alpha.map((a) => a.toFixed(1)).join(', ')})
                </div>
                <div>
                  <span className="font-medium">{'\u03B1\u2080 = \u03A3\u03B1\u2C7C'}</span>{' = '}
                  <span className="tabular-nums">{fmt(alpha0, 2)}</span>
                </div>
                <div className="border-t pt-2 mt-2" style={{ borderColor: 'var(--color-border)' }}>
                  <div className="font-medium mb-1">Marginal moments:</div>
                  {[0, 1, 2].map((j) => (
                    <div key={j} className="ml-2 mb-1">
                      <span className="opacity-70">{`P${'\u2081\u2082\u2083'[j]}:`}</span>{' '}
                      E = <span className="tabular-nums">{fmt(priorMean[j])}</span>,{' '}
                      Var = <span className="tabular-nums">{fmt(exploreVar[j])}</span>
                    </div>
                  ))}
                </div>
                <div className="border-t pt-2 mt-2 opacity-60" style={{ borderColor: 'var(--color-border)' }}>
                  {alpha0 > 3
                    ? 'High concentration \u2014 samples cluster near the mean.'
                    : alpha0 < 1
                      ? 'Sparse \u2014 mass concentrates at the simplex corners.'
                      : alpha0 <= 3
                        ? 'Moderate concentration \u2014 broad coverage of the simplex.'
                        : ''}
                </div>
              </div>
            </>
          ) : (
            /* Conjugate mode statistics */
            <>
              <div className="text-xs font-semibold mb-2 text-center opacity-70">
                Conjugate Updating
              </div>
              <div className="space-y-2 text-xs">
                <div>
                  <span className="font-medium">Prior {'\u03B1'}</span>{' = '}
                  ({alpha.map((a) => a.toFixed(1)).join(', ')})
                </div>
                {hasData && (
                  <>
                    <div>
                      <span className="font-medium">Observed x</span>{' = '}
                      ({observedData.join(', ')}),{' '}
                      <span className="opacity-70">n = {totalObservations}</span>
                    </div>
                    <div>
                      <span className="font-medium">Posterior {'\u03B1\u2032'}</span>{' = \u03B1 + x = '}
                      ({posteriorAlpha.map((a) => a.toFixed(1)).join(', ')})
                    </div>
                    <div className="border-t pt-2 mt-1" style={{ borderColor: 'var(--color-border)' }}>
                      <div className="mb-1">
                        <span style={{ color: '#2563eb' }}>{'\u25CF'}</span>{' '}
                        <span className="font-medium">Prior mean:</span>{' '}
                        ({priorMean.map((v) => fmt(v, 3)).join(', ')})
                      </div>
                      <div className="mb-1">
                        <span style={{ color: '#7c3aed' }}>{'\u25CF'}</span>{' '}
                        <span className="font-medium">Posterior mean:</span>{' '}
                        {posteriorMean && `(${posteriorMean.map((v) => fmt(v, 3)).join(', ')})`}
                      </div>
                      <div className="mb-1">
                        <span style={{ color: '#dc2626' }}>{'\u2605'}</span>{' '}
                        <span className="font-medium">MLE (x/n):</span>{' '}
                        {mle && `(${mle.map((v) => fmt(v, 3)).join(', ')})`}
                      </div>
                    </div>
                    <div
                      className="border-t pt-2 mt-1 opacity-60"
                      style={{ borderColor: 'var(--color-border)' }}
                    >
                      Effective sample size: {'\u03B1\u2080 = '}{fmt(alpha0, 1)} (prior)
                      {' + '}{totalObservations} (data){' = '}{fmt(posteriorAlpha0, 1)} (posterior).
                      {posteriorAlpha0 > 2 * alpha0
                        ? ' Data is dominating the prior.'
                        : ' Prior still has substantial influence.'}
                    </div>
                  </>
                )}
                {!hasData && (
                  <div className="opacity-60 mt-2">
                    Click &ldquo;Observe Data&rdquo; to generate Multinomial samples and watch
                    the posterior update.
                  </div>
                )}
              </div>
            </>
          )}

          {/* Legend */}
          <div className="mt-auto pt-3 border-t flex flex-wrap gap-3 text-[10px]" style={{ borderColor: 'var(--color-border)' }}>
            {mode === 'explore' ? (
              <>
                <span><span style={{ color: 'rgba(59, 130, 246, 0.7)' }}>{'\u25CF'}</span> Dir({'\u03B1'}) samples</span>
                <span><span style={{ color: '#2563eb' }}>{'\u25CF'}</span> E[P]</span>
              </>
            ) : (
              <>
                <span><span style={{ color: 'rgba(59, 130, 246, 0.5)' }}>{'\u25CF'}</span> Prior</span>
                <span><span style={{ color: 'rgba(139, 92, 246, 0.7)' }}>{'\u25CF'}</span> Posterior</span>
                <span><span style={{ color: '#2563eb' }}>{'\u25CF'}</span> Prior mean</span>
                <span><span style={{ color: '#7c3aed' }}>{'\u25CF'}</span> Post. mean</span>
                <span><span style={{ color: '#dc2626' }}>{'\u2605'}</span> MLE</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
