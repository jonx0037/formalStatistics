import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { bayesTheorem, totalProbability, diagnosticTest } from './shared/probability';
import { medicalTestPresets } from '../../data/conditional-probability-data';

type Mode = 'abstract' | 'medical';

const STEP_LABELS = [
  'Prior',
  'Multiply by Likelihood',
  'Unnormalized Product',
  'Normalize (Posterior)',
] as const;

const STEP_DESCRIPTIONS = [
  'We start with our prior beliefs: P(A) and P(Aᶜ), which must sum to 1.',
  'Multiply each prior by the corresponding likelihood: P(B|A) and P(B|Aᶜ).',
  'The unnormalized products P(B|A)·P(A) and P(B|Aᶜ)·P(Aᶜ) sum to P(B).',
  'Divide each product by P(B) to get the posterior: P(A|B) and P(Aᶜ|B), which sum to 1.',
] as const;

// ── Colors ─────────────────────────────────────────────────────────────────

const COLOR_A = '#2563eb';
const COLOR_AC = '#6b7280';
const COLOR_TP = '#16a34a';
const COLOR_FP = '#dc2626';
const COLOR_TN = '#9ca3af';
const COLOR_FN = '#ea580c';

export default function BayesTheoremExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  // ── Mode ───────────────────────────────────────────────────────────────
  const [mode, setMode] = useState<Mode>('abstract');

  // ── Abstract mode state ────────────────────────────────────────────────
  const [pA, setPA] = useState(0.3);
  const [pBgivenA, setPBgivenA] = useState(0.8);
  const [pBgivenAc, setPBgivenAc] = useState(0.2);
  const [animStep, setAnimStep] = useState(0);

  // ── Medical mode state ─────────────────────────────────────────────────
  const [prevalence, setPrevalence] = useState(0.01);
  const [sensitivity, setSensitivity] = useState(0.99);
  const [specificity, setSpecificity] = useState(0.95);

  // ── Abstract computed values ───────────────────────────────────────────
  const abstractValues = useMemo(() => {
    const pAc = 1 - pA;
    const pB = totalProbability([pBgivenA, pBgivenAc], [pA, pAc]);
    const pAgivenB = pB === 0 ? 0 : bayesTheorem(pBgivenA, pA, pB);
    const pAcGivenB = pB === 0 ? 0 : bayesTheorem(pBgivenAc, pAc, pB);
    return { pAc, pB, pAgivenB, pAcGivenB };
  }, [pA, pBgivenA, pBgivenAc]);

  // Bar values at each animation step
  const stepBars = useMemo(() => {
    const { pAc, pAgivenB, pAcGivenB } = abstractValues;
    return [
      { a: pA, ac: 1 - pA },                                 // Step 0: Prior
      { a: pBgivenA * pA, ac: pBgivenAc * pAc },             // Step 1: Multiply
      { a: pBgivenA * pA, ac: pBgivenAc * pAc },             // Step 2: Unnormalized
      { a: pAgivenB, ac: pAcGivenB },                        // Step 3: Posterior
    ];
  }, [pA, pBgivenA, pBgivenAc, abstractValues]);

  // ── Medical computed values ────────────────────────────────────────────
  const medicalValues = useMemo(() => {
    return diagnosticTest(prevalence, sensitivity, specificity, 1000);
  }, [prevalence, sensitivity, specificity]);

  // ── Handlers ───────────────────────────────────────────────────────────
  const handleNextStep = () => setAnimStep((s) => Math.min(s + 1, 3));
  const handleReset = () => setAnimStep(0);

  const applyPreset = (idx: number) => {
    const preset = medicalTestPresets[idx];
    setPrevalence(preset.prevalence);
    setSensitivity(preset.sensitivity);
    setSpecificity(preset.specificity);
  };

  // ── SVG dimensions ────────────────────────────────────────────────────
  const svgWidth = Math.max(width, 200);
  const svgHeight = Math.min(svgWidth * 0.5, 220);
  const barAreaLeft = svgWidth * 0.15;
  const barAreaRight = svgWidth * 0.85;
  const barAreaTop = 24;
  const barAreaBottom = svgHeight - 28;
  const barAreaHeight = barAreaBottom - barAreaTop;
  const barWidth = Math.min((barAreaRight - barAreaLeft) * 0.3, 80);
  const barGap = (barAreaRight - barAreaLeft - 2 * barWidth) / 3;

  // Medical bar dimensions
  const medBarTop = 40;
  const medBarHeight = 36;
  const medBarLeft = svgWidth * 0.05;
  const medBarFullWidth = svgWidth * 0.9;

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Mode toggle */}
      <div className="flex gap-1.5 mb-4">
        {(['abstract', 'medical'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className="px-3 py-1 rounded text-xs sm:text-sm font-medium transition-colors border cursor-pointer"
            style={{
              background: mode === m ? '#2563eb' : 'var(--color-surface)',
              color: mode === m ? '#ffffff' : 'var(--color-text)',
              borderColor: mode === m ? '#2563eb' : 'var(--color-border)',
            }}
          >
            {m === 'abstract' ? 'Abstract' : 'Medical Testing'}
          </button>
        ))}
      </div>

      <div ref={containerRef} className="w-full">
        {/* ── Abstract Mode ─────────────────────────────────────────── */}
        {mode === 'abstract' && (
          <>
            {/* Step label */}
            <div className="mb-2">
              <span
                className="text-xs font-semibold uppercase tracking-wider"
                style={{ color: 'var(--color-text-muted)' }}
              >
                Step {animStep + 1} of 4:
              </span>{' '}
              <span className="text-sm font-semibold" style={{ color: 'var(--color-text-heading)' }}>
                {STEP_LABELS[animStep]}
              </span>
              <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                {STEP_DESCRIPTIONS[animStep]}
              </p>
            </div>

            {/* SVG bar chart */}
            <svg
              width={svgWidth}
              height={svgHeight}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ overflow: 'visible' }}
            >
              {(() => {
                const bars = stepBars[animStep];
                const maxVal = Math.max(bars.a, bars.ac, 0.01);
                const scale = barAreaHeight / Math.max(maxVal, 0.01);

                const x1 = barAreaLeft + barGap;
                const x2 = x1 + barWidth + barGap;

                const h1 = Math.max(bars.a * scale, 1);
                const h2 = Math.max(bars.ac * scale, 1);
                const y1 = barAreaBottom - h1;
                const y2 = barAreaBottom - h2;

                return (
                  <>
                    {/* Bar A */}
                    <rect
                      x={x1} y={y1} width={barWidth} height={h1}
                      rx={3}
                      fill={COLOR_A} fillOpacity={0.7}
                    />
                    <text
                      x={x1 + barWidth / 2} y={y1 - 6}
                      textAnchor="middle" fontSize={12} fontWeight={600}
                      fill={COLOR_A}
                    >
                      {bars.a.toFixed(4)}
                    </text>
                    <text
                      x={x1 + barWidth / 2} y={barAreaBottom + 16}
                      textAnchor="middle" fontSize={12} fontWeight={600}
                      fill="var(--color-text)"
                    >
                      A
                    </text>

                    {/* Bar Aᶜ */}
                    <rect
                      x={x2} y={y2} width={barWidth} height={h2}
                      rx={3}
                      fill={COLOR_AC} fillOpacity={0.7}
                    />
                    <text
                      x={x2 + barWidth / 2} y={y2 - 6}
                      textAnchor="middle" fontSize={12} fontWeight={600}
                      fill={COLOR_AC}
                    >
                      {bars.ac.toFixed(4)}
                    </text>
                    <text
                      x={x2 + barWidth / 2} y={barAreaBottom + 16}
                      textAnchor="middle" fontSize={12} fontWeight={600}
                      fill="var(--color-text)"
                    >
                      A&#x1D9C;
                    </text>

                    {/* P(B) annotation for step 2 */}
                    {animStep === 2 && (
                      <text
                        x={svgWidth / 2} y={barAreaTop - 2}
                        textAnchor="middle" fontSize={11} fontWeight={600}
                        fill="var(--color-text-heading)"
                      >
                        Sum = P(B) = {abstractValues.pB.toFixed(4)}
                      </text>
                    )}

                    {/* Sum = 1 annotation for steps 0 and 3 */}
                    {(animStep === 0 || animStep === 3) && (
                      <text
                        x={svgWidth / 2} y={barAreaTop - 2}
                        textAnchor="middle" fontSize={11} fontWeight={600}
                        fill="var(--color-text-heading)"
                      >
                        Sum = 1
                      </text>
                    )}
                  </>
                );
              })()}
            </svg>

            {/* Step-through buttons */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={handleReset}
                disabled={animStep === 0}
                className="px-3 py-1 rounded text-xs font-medium border cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                  background: 'var(--color-surface)',
                  color: 'var(--color-text)',
                  borderColor: 'var(--color-border)',
                }}
              >
                Reset
              </button>
              <button
                onClick={handleNextStep}
                disabled={animStep === 3}
                className="px-3 py-1 rounded text-xs font-medium border cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                  background: animStep < 3 ? '#2563eb' : 'var(--color-surface)',
                  color: animStep < 3 ? '#ffffff' : 'var(--color-text)',
                  borderColor: animStep < 3 ? '#2563eb' : 'var(--color-border)',
                }}
              >
                Next Step
              </button>
            </div>

            {/* Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                P(A) = {pA.toFixed(2)}
                <input
                  type="range" min={0.01} max={0.99} step={0.01}
                  value={pA}
                  onChange={(e) => { setPA(+e.target.value); setAnimStep(0); }}
                  className="w-full"
                />
              </label>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                P(B|A) = {pBgivenA.toFixed(2)}
                <input
                  type="range" min={0} max={1} step={0.01}
                  value={pBgivenA}
                  onChange={(e) => { setPBgivenA(+e.target.value); setAnimStep(0); }}
                  className="w-full"
                />
              </label>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                P(B|A&#x1D9C;) = {pBgivenAc.toFixed(2)}
                <input
                  type="range" min={0} max={1} step={0.01}
                  value={pBgivenAc}
                  onChange={(e) => { setPBgivenAc(+e.target.value); setAnimStep(0); }}
                  className="w-full"
                />
              </label>
            </div>

            {/* Live readout */}
            <div
              className="mt-3 text-xs sm:text-sm space-y-1 rounded p-3 border"
              style={{
                background: 'var(--color-surface)',
                borderColor: 'var(--color-border)',
                color: 'var(--color-text)',
              }}
            >
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Prior:</span>{' '}
                P(A) = <strong>{pA.toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Likelihood:</span>{' '}
                P(B|A) = <strong>{pBgivenA.toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Evidence (total probability):</span>{' '}
                P(B) = P(B|A)&middot;P(A) + P(B|A&#x1D9C;)&middot;P(A&#x1D9C;) ={' '}
                <strong>{abstractValues.pB.toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Posterior (Bayes):</span>{' '}
                P(A|B) = P(B|A)&middot;P(A) / P(B) ={' '}
                <strong style={{ color: COLOR_A }}>{abstractValues.pAgivenB.toFixed(4)}</strong>
              </div>
            </div>
          </>
        )}

        {/* ── Medical Testing Mode ──────────────────────────────────── */}
        {mode === 'medical' && (
          <>
            {/* Preset buttons */}
            <div className="flex flex-wrap gap-1.5 mb-4">
              {medicalTestPresets.map((preset, idx) => (
                <button
                  key={preset.name}
                  onClick={() => applyPreset(idx)}
                  className="px-3 py-1 rounded text-xs font-medium transition-colors border cursor-pointer"
                  style={{
                    background: 'var(--color-surface)',
                    color: 'var(--color-text)',
                    borderColor: 'var(--color-border)',
                  }}
                  title={preset.description}
                >
                  {preset.name}
                </button>
              ))}
            </div>

            {/* Natural frequency stacked bar */}
            <svg
              width={svgWidth}
              height={110}
              viewBox={`0 0 ${svgWidth} 110`}
              style={{ overflow: 'visible' }}
            >
              {(() => {
                const { truePositives, falsePositives, trueNegatives, falseNegatives } = medicalValues;
                const total = truePositives + falsePositives + trueNegatives + falseNegatives;
                if (total === 0) return null;

                const segments = [
                  { label: 'TP', count: truePositives, color: COLOR_TP },
                  { label: 'FP', count: falsePositives, color: COLOR_FP },
                  { label: 'FN', count: falseNegatives, color: COLOR_FN },
                  { label: 'TN', count: trueNegatives, color: COLOR_TN },
                ];

                let xOffset = medBarLeft;
                return (
                  <>
                    <text
                      x={svgWidth / 2} y={16}
                      textAnchor="middle" fontSize={12} fontWeight={600}
                      fill="var(--color-text-heading)"
                    >
                      Population of 1,000 people
                    </text>
                    <text
                      x={svgWidth / 2} y={30}
                      textAnchor="middle" fontSize={10}
                      fill="var(--color-text-muted)"
                    >
                      Test results breakdown
                    </text>
                    {segments.map((seg) => {
                      const w = (seg.count / total) * medBarFullWidth;
                      const x = xOffset;
                      xOffset += w;
                      if (seg.count === 0) return null;
                      return (
                        <g key={seg.label}>
                          <rect
                            x={x} y={medBarTop} width={Math.max(w, 0.5)} height={medBarHeight}
                            fill={seg.color} fillOpacity={0.75}
                            rx={0}
                          />
                          {w > 30 && (
                            <>
                              <text
                                x={x + w / 2} y={medBarTop + medBarHeight / 2 - 4}
                                textAnchor="middle" fontSize={11} fontWeight={700}
                                fill="#fff"
                              >
                                {seg.label}
                              </text>
                              <text
                                x={x + w / 2} y={medBarTop + medBarHeight / 2 + 10}
                                textAnchor="middle" fontSize={10}
                                fill="#fff"
                              >
                                {seg.count}
                              </text>
                            </>
                          )}
                        </g>
                      );
                    })}
                    {/* Legend below bar */}
                    {(() => {
                      const legendY = medBarTop + medBarHeight + 20;
                      const legendItems = segments.filter((s) => s.count > 0);
                      const itemWidth = Math.min(svgWidth / legendItems.length, 120);
                      const totalLegendWidth = itemWidth * legendItems.length;
                      const legendStartX = (svgWidth - totalLegendWidth) / 2;
                      return legendItems.map((seg, i) => (
                        <g key={`legend-${seg.label}`}>
                          <rect
                            x={legendStartX + i * itemWidth}
                            y={legendY - 6}
                            width={10} height={10} rx={2}
                            fill={seg.color} fillOpacity={0.75}
                          />
                          <text
                            x={legendStartX + i * itemWidth + 14}
                            y={legendY + 3}
                            fontSize={10}
                            fill="var(--color-text-muted)"
                          >
                            {seg.label}: {seg.count}
                          </text>
                        </g>
                      ));
                    })()}
                  </>
                );
              })()}
            </svg>

            {/* PPV / NPV readout */}
            <div className="grid grid-cols-2 gap-3 mt-3">
              <div
                className="rounded p-3 border text-center"
                style={{
                  background: 'var(--color-surface)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  PPV (Positive Predictive Value)
                </div>
                <div className="text-lg sm:text-xl font-bold" style={{ color: COLOR_TP }}>
                  {isNaN(medicalValues.ppv) ? 'N/A' : (medicalValues.ppv * 100).toFixed(1) + '%'}
                </div>
                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  TP / (TP + FP) = {medicalValues.truePositives} / {medicalValues.truePositives + medicalValues.falsePositives}
                </div>
              </div>
              <div
                className="rounded p-3 border text-center"
                style={{
                  background: 'var(--color-surface)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  NPV (Negative Predictive Value)
                </div>
                <div className="text-lg sm:text-xl font-bold" style={{ color: COLOR_TN }}>
                  {isNaN(medicalValues.npv) ? 'N/A' : (medicalValues.npv * 100).toFixed(1) + '%'}
                </div>
                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                  TN / (TN + FN) = {medicalValues.trueNegatives} / {medicalValues.trueNegatives + medicalValues.falseNegatives}
                </div>
              </div>
            </div>

            {/* Base rate fallacy warning */}
            {prevalence < 0.05 && medicalValues.ppv < 0.5 && !isNaN(medicalValues.ppv) && (
              <div
                className="mt-3 rounded p-3 border text-xs sm:text-sm"
                style={{
                  background: '#fef3c7',
                  borderColor: '#f59e0b',
                  color: '#92400e',
                }}
              >
                <strong>Base rate fallacy:</strong> Even with high sensitivity ({(sensitivity * 100).toFixed(0)}%) and
                specificity ({(specificity * 100).toFixed(0)}%), the low prevalence ({(prevalence * 100).toFixed(1)}%)
                means most positive results are false positives. Only{' '}
                <strong>{(medicalValues.ppv * 100).toFixed(1)}%</strong> of positive tests indicate
                true disease. This is why screening low-prevalence populations produces
                surprisingly many false alarms.
              </div>
            )}

            {/* Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Prevalence = {(prevalence * 100).toFixed(1)}%
                <input
                  type="range" min={0.001} max={0.5} step={0.001}
                  value={prevalence}
                  onChange={(e) => setPrevalence(+e.target.value)}
                  className="w-full"
                />
              </label>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Sensitivity = {(sensitivity * 100).toFixed(1)}%
                <input
                  type="range" min={0} max={1} step={0.01}
                  value={sensitivity}
                  onChange={(e) => setSensitivity(+e.target.value)}
                  className="w-full"
                />
              </label>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Specificity = {(specificity * 100).toFixed(1)}%
                <input
                  type="range" min={0} max={1} step={0.01}
                  value={specificity}
                  onChange={(e) => setSpecificity(+e.target.value)}
                  className="w-full"
                />
              </label>
            </div>

            {/* Live readout */}
            <div
              className="mt-3 text-xs sm:text-sm space-y-1 rounded p-3 border"
              style={{
                background: 'var(--color-surface)',
                borderColor: 'var(--color-border)',
                color: 'var(--color-text)',
              }}
            >
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Prevalence:</span>{' '}
                <strong>{(prevalence * 100).toFixed(2)}%</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Sensitivity (TPR):</span>{' '}
                <strong>{(sensitivity * 100).toFixed(1)}%</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>Specificity (TNR):</span>{' '}
                <strong>{(specificity * 100).toFixed(1)}%</strong>
              </div>
              <div className="pt-1 border-t" style={{ borderColor: 'var(--color-border)' }}>
                <span style={{ color: 'var(--color-text-muted)' }}>TP:</span> {medicalValues.truePositives}{' '}
                <span style={{ color: 'var(--color-text-muted)' }}>FP:</span> {medicalValues.falsePositives}{' '}
                <span style={{ color: 'var(--color-text-muted)' }}>TN:</span> {medicalValues.trueNegatives}{' '}
                <span style={{ color: 'var(--color-text-muted)' }}>FN:</span> {medicalValues.falseNegatives}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
