import { useState, useMemo } from 'react';
import katex from 'katex';
import {
  derivationSteps,
  canonicalFormColors,
  exponentialFamilyMembers,
} from '../../data/exponential-family-data';

// ── Distribution keys in the order they appear in the data module ──────────
const distKeys = exponentialFamilyMembers.map((m) => m.distribution);
const distNames = exponentialFamilyMembers.map((m) => m.name);

// ── Highlight labels mapped to their canonical component names ─────────────
const highlightLabels: Record<string, string> = {
  h: 'h(x)',
  eta: '\u03B7(\u03B8)',
  T: 'T(x)',
  A: 'A(\u03B7)',
};

// ── KaTeX render helper ────────────────────────────────────────────────────
// All LaTeX strings originate from derivationSteps in exponential-family-data.ts,
// not from user input. The dangerouslySetInnerHTML usage is safe in this context
// and follows the same pattern as ExponentialFamilyExplorer.tsx.
function renderLatex(latex: string): string {
  try {
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode: true,
    });
  } catch {
    return latex;
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function CanonicalFormConverter() {
  const [distKey, setDistKey] = useState('Bernoulli');
  const [currentStep, setCurrentStep] = useState(0);
  const [showAll, setShowAll] = useState(false);

  const steps = derivationSteps[distKey] ?? [];

  // Reset state when distribution changes
  const handleDistChange = (key: string) => {
    setDistKey(key);
    setCurrentStep(0);
    setShowAll(false);
  };

  // Step through: advance by one, or wrap to 0
  const handleStep = () => {
    setCurrentStep((prev) => (prev >= steps.length - 1 ? 0 : prev + 1));
  };

  // Pre-render all LaTeX strings (sourced from exponential-family-data.ts)
  const renderedSteps = useMemo(
    () => steps.map((s) => ({ ...s, html: renderLatex(s.latex) })),
    [steps],
  );

  return (
    <div
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <h3 className="text-lg font-semibold mb-3">
        Interactive: Canonical Form Converter
      </h3>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        {/* Distribution dropdown */}
        <label className="flex items-center gap-2 text-sm">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-bg)',
            }}
            value={distKey}
            onChange={(e) => handleDistChange(e.target.value)}
          >
            {distKeys.map((key, i) => (
              <option key={key} value={key}>
                {distNames[i]}
              </option>
            ))}
          </select>
        </label>

        {/* Step Through button */}
        <button
          onClick={handleStep}
          className="rounded border px-3 py-1 text-sm font-medium transition-colors hover:opacity-80"
          style={{
            borderColor: 'var(--color-border)',
            background: 'var(--color-bg)',
          }}
        >
          {currentStep >= steps.length - 1 && !showAll
            ? 'Restart \u21BB'
            : 'Step Through \u25B6'}
        </button>

        {/* Show All checkbox */}
        <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => setShowAll(e.target.checked)}
            className="accent-blue-600"
          />
          Show All
        </label>
      </div>

      {/* Color legend */}
      <div className="flex flex-wrap gap-3 mb-3 text-xs opacity-70">
        <span style={{ color: canonicalFormColors.h }}>
          {'\u25A0'} h(x)
        </span>
        <span style={{ color: canonicalFormColors.eta }}>
          {'\u25A0'} {'\u03B7(\u03B8)'}
        </span>
        <span style={{ color: canonicalFormColors.T }}>
          {'\u25A0'} T(x)
        </span>
        <span style={{ color: canonicalFormColors.A }}>
          {'\u25A0'} A({'\u03B7'})
        </span>
      </div>

      {/* Derivation steps */}
      <div className="space-y-2">
        {renderedSteps.map((step, i) => {
          const visible = showAll || i <= currentStep;
          const highlightColor = step.highlight
            ? canonicalFormColors[step.highlight]
            : undefined;

          return (
            <div
              key={`${distKey}-${i}`}
              className="rounded border p-3"
              style={{
                borderColor: visible
                  ? (highlightColor ?? 'var(--color-border)')
                  : 'var(--color-border)',
                opacity: visible ? 1 : 0.15,
                transition: 'opacity 0.3s ease-in-out',
              }}
            >
              {/* Step description */}
              <div className="text-sm font-medium mb-1">
                <span className="opacity-50 mr-1.5">
                  Step {i + 1}:
                </span>
                {highlightColor ? (
                  <span style={{ color: highlightColor }}>
                    {step.description}
                    {step.highlight && (
                      <span className="ml-1 text-xs opacity-70">
                        [{highlightLabels[step.highlight]}]
                      </span>
                    )}
                  </span>
                ) : (
                  <span>{step.description}</span>
                )}
              </div>

              {/* Rendered LaTeX equation — all strings from exponential-family-data.ts */}
              <div
                className="text-center overflow-x-auto"
                // SAFETY: LaTeX source is our own static derivationSteps data,
                // not user input. Same pattern as ExponentialFamilyExplorer.tsx.
                dangerouslySetInnerHTML={{ __html: step.html }}
              />
            </div>
          );
        })}
      </div>

      {/* Footer note */}
      <div className="mt-3 text-xs opacity-50 text-center">
        {steps.length} steps &middot; Use &ldquo;Step Through&rdquo; to
        reveal one step at a time, or check &ldquo;Show All&rdquo; to see
        the full derivation.
      </div>
    </div>
  );
}
