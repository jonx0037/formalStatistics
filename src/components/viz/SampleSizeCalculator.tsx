import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { sampleSizeRequirements } from './shared/convergence';
import { quantileStdNormal } from './shared/distributions';

// ── Types & Presets ─────────────────────────────────────────────────────────

type DistClass =
  | 'bounded'
  | 'known-variance'
  | 'sub-gaussian'
  | 'general';

const BOUND_COLORS: Record<string, string> = {
  chebyshev: '#d97706',
  hoeffding: '#059669',
  bernstein: '#2563eb',
  subGaussian: '#7c3aed',
  clt: '#dc2626',
};

const BOUND_LABEL: Record<string, string> = {
  chebyshev: 'Chebyshev',
  hoeffding: 'Hoeffding',
  bernstein: 'Bernstein',
  subGaussian: 'Sub-Gaussian',
  clt: 'CLT (approx)',
};

// ── Component ───────────────────────────────────────────────────────────────

export default function SampleSizeCalculator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 640, 320);

  const [epsilon, setEpsilon] = useState(0.05);
  const [delta, setDelta] = useState(0.05);
  const [distClass, setDistClass] = useState<DistClass>('bounded');

  // Bounded-variable params
  const [aBound, setABound] = useState(0);
  const [bBound, setBBound] = useState(1);
  // Variance/M params
  const [sigma2, setSigma2] = useState(0.25);
  const [M, setM] = useState(0.5);
  // Sub-Gaussian param
  const [sgParam, setSgParam] = useState(0.5);

  const results = useMemo(() => {
    const zScore = quantileStdNormal(1 - delta / 2);
    const params: Parameters<typeof sampleSizeRequirements>[2] = { zScore };
    if (distClass === 'bounded' || distClass === 'known-variance') {
      params.range = [aBound, bBound];
    }
    if (distClass === 'known-variance' || distClass === 'sub-gaussian' || distClass === 'general') {
      params.sigma2 = sigma2;
    }
    if (distClass === 'known-variance') {
      params.M = M;
    }
    if (distClass === 'sub-gaussian') {
      params.subGaussianParam = sgParam;
    }
    if (distClass === 'bounded') {
      const range = bBound - aBound;
      params.sigma2 = (range * range) / 4;
      params.M = range / 2;
    }
    return sampleSizeRequirements(epsilon, delta, params);
  }, [epsilon, delta, distClass, aBound, bBound, sigma2, M, sgParam]);

  const entries = useMemo(
    () =>
      Object.entries(results).filter(([, n]) => n !== undefined && isFinite(n)) as Array<
        [string, number]
      >,
    [results]
  );

  const sortedEntries = useMemo(
    () => [...entries].sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0)),
    [entries]
  );

  const tightest = useMemo(() => {
    let best: [string, number] | null = null;
    for (const [k, v] of entries) {
      if (best === null || v < best[1]) best = [k, v];
    }
    return best;
  }, [entries]);

  const maxN = sortedEntries.length > 0 ? sortedEntries[0][1] : 1;

  return (
    <div
      ref={containerRef}
      style={{
        fontFamily: 'system-ui, sans-serif',
        border: '1px solid var(--color-border)',
        borderRadius: '0.75rem',
        padding: '1.25rem',
        margin: '1.5rem 0',
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: '0.75rem' }}>
        Sample-Size Calculator
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 700 ? '1fr 1fr' : '1fr',
          gap: '1rem',
        }}
      >
        {/* Controls column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <LabeledRange
            label={`Target accuracy ε = ${epsilon.toFixed(3)}`}
            min={0.005}
            max={0.2}
            step={0.005}
            value={epsilon}
            onChange={setEpsilon}
          />
          <LabeledRange
            label={`Failure probability δ = ${delta.toFixed(3)}`}
            min={0.005}
            max={0.2}
            step={0.005}
            value={delta}
            onChange={setDelta}
          />

          <label style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
            Distribution class:
            <select
              value={distClass}
              onChange={(e) => setDistClass(e.target.value as DistClass)}
              style={{ marginLeft: '0.5rem', padding: '0.2rem' }}
            >
              <option value="bounded">Bounded [a, b]</option>
              <option value="known-variance">Bounded + known variance</option>
              <option value="sub-gaussian">Sub-Gaussian(σ_sg)</option>
              <option value="general">General (mean + variance)</option>
            </select>
          </label>

          {(distClass === 'bounded' || distClass === 'known-variance') && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <LabeledNumber
                label="a"
                value={aBound}
                onChange={setABound}
                step={0.1}
              />
              <LabeledNumber
                label="b"
                value={bBound}
                onChange={setBBound}
                step={0.1}
              />
            </div>
          )}
          {(distClass === 'known-variance' || distClass === 'sub-gaussian' || distClass === 'general') && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <LabeledNumber
                label="σ²"
                value={sigma2}
                onChange={setSigma2}
                step={0.01}
              />
              {(distClass === 'known-variance') && (
                <LabeledNumber
                  label="M"
                  value={M}
                  onChange={setM}
                  step={0.05}
                />
              )}
            </div>
          )}
          {distClass === 'sub-gaussian' && (
            <LabeledNumber
              label="σ_sg"
              value={sgParam}
              onChange={setSgParam}
              step={0.05}
            />
          )}
        </div>

        {/* Results column */}
        <div>
          <div
            style={{
              fontSize: '0.95rem',
              marginBottom: '0.75rem',
              padding: '0.5rem',
              backgroundColor: 'rgba(5, 150, 105, 0.08)',
              borderRadius: '0.375rem',
              border: '1px solid rgba(5, 150, 105, 0.3)',
            }}
          >
            {tightest ? (
              <>
                <strong>You need n ≥ {tightest[1].toLocaleString()}</strong>{' '}
                under the tightest applicable bound ({BOUND_LABEL[tightest[0]]}).
              </>
            ) : (
              <em>No applicable bound — specify more parameters above.</em>
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {sortedEntries.map(([key, n]) => (
              <div
                key={key}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '110px 1fr 90px',
                  alignItems: 'center',
                  fontSize: '0.8125rem',
                  gap: '6px',
                }}
              >
                <span style={{ color: BOUND_COLORS[key] }}>
                  {BOUND_LABEL[key]}
                </span>
                <div
                  style={{
                    height: '14px',
                    width: `${Math.max(2, (n / maxN) * 100)}%`,
                    background: BOUND_COLORS[key],
                    borderRadius: '3px',
                    opacity: 0.85,
                  }}
                />
                <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                  {n.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
          <div
            style={{
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)',
              marginTop: '0.75rem',
            }}
          >
            The CLT row uses the asymptotic z = Φ⁻¹(1 − δ/2) formula and is approximate;
            the others are non-asymptotic.
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────

function LabeledRange({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ fontSize: '0.8125rem' }}>
      {label}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '100%' }}
      />
    </label>
  );
}

function LabeledNumber({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
}) {
  return (
    <label style={{ fontSize: '0.8125rem', flex: 1 }}>
      {label}:{' '}
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '5rem', padding: '0.15rem' }}
      />
    </label>
  );
}
