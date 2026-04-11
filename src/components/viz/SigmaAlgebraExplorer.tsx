import { useState, useMemo, useCallback } from 'react';
import { sigmaAlgebraPresets } from '../../data/sample-spaces-data';
import { isValidSigmaAlgebra } from './shared/probability';

const OMEGA = ['1', '2', '3', '4'];

/** Generate all 16 subsets of {1,2,3,4}. */
function allSubsets(): string[][] {
  const result: string[][] = [];
  for (let mask = 0; mask < 16; mask++) {
    const subset: string[] = [];
    for (let i = 0; i < 4; i++) {
      if (mask & (1 << i)) subset.push(OMEGA[i]);
    }
    result.push(subset);
  }
  return result;
}

function subsetLabel(s: string[]): string {
  if (s.length === 0) return '∅';
  if (s.length === 4) return 'Ω';
  return `{${s.join(',')}}`;
}

function subsetKey(s: string[]): string {
  return [...s].sort().join(',');
}

export default function SigmaAlgebraExplorer() {
  const subsets = useMemo(() => allSubsets(), []);
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(
    new Set(sigmaAlgebraPresets[0].events.map(subsetKey)),
  );

  const toggleSubset = useCallback((key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const loadPreset = useCallback((presetIdx: number) => {
    const preset = sigmaAlgebraPresets[presetIdx];
    setSelectedKeys(new Set(preset.events.map(subsetKey)));
  }, []);

  // Build Event[] from selected keys
  const omega = new Set(OMEGA);
  const collection = subsets
    .filter((s) => selectedKeys.has(subsetKey(s)))
    .map((s) => new Set(s));

  const validation = isValidSigmaAlgebra(omega, collection);

  // Individual checks
  const hasOmega = selectedKeys.has(subsetKey(OMEGA));

  // Check complement closure separately for display
  const complementIssues = validation.violations.filter((v) => v.startsWith('Complement'));
  const unionIssues = validation.violations.filter((v) => v.startsWith('Union'));

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <span className="text-sm font-semibold" style={{ color: 'var(--color-text-heading)' }}>
          Presets:
        </span>
        {sigmaAlgebraPresets.map((preset, i) => (
          <button
            key={preset.name}
            onClick={() => loadPreset(i)}
            className="px-3 py-1 rounded text-xs font-medium border transition-colors cursor-pointer"
            style={{
              background: 'var(--color-surface)',
              borderColor: 'var(--color-border)',
              color: 'var(--color-text)',
            }}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Omega display */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
          &Omega; =
        </span>
        {OMEGA.map((el) => (
          <span
            key={el}
            className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold border"
            style={{
              background: '#dbeafe',
              borderColor: '#2563eb',
              color: '#1e40af',
            }}
          >
            {el}
          </span>
        ))}
      </div>

      {/* Subset checkboxes */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 mb-4">
        {subsets.map((s) => {
          const key = subsetKey(s);
          const checked = selectedKeys.has(key);
          return (
            <label
              key={key}
              className="flex items-center gap-1.5 text-xs sm:text-sm font-mono rounded px-2 py-1 cursor-pointer transition-colors"
              style={{
                background: checked ? 'var(--color-definition-bg)' : 'var(--color-surface)',
                color: 'var(--color-text)',
              }}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleSubset(key)}
                className="rounded"
              />
              {subsetLabel(s)}
            </label>
          );
        })}
      </div>

      {/* Validation */}
      <div
        className="rounded-lg border p-4"
        style={{
          background: validation.valid ? 'var(--color-theorem-bg)' : 'var(--color-surface)',
          borderColor: validation.valid ? 'var(--color-theorem-border)' : 'var(--color-border)',
        }}
      >
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3 text-sm">
          <div>
            <span className="mr-1">{hasOmega ? '✓' : '✗'}</span>
            <span style={{ color: hasOmega ? 'var(--color-accent)' : 'var(--color-danger)' }}>
              &Omega; &isin; &Fscr;
            </span>
          </div>
          <div>
            <span className="mr-1">{complementIssues.length === 0 && hasOmega ? '✓' : '✗'}</span>
            <span style={{ color: complementIssues.length === 0 && hasOmega ? 'var(--color-accent)' : 'var(--color-danger)' }}>
              Closed under complements
            </span>
          </div>
          <div>
            <span className="mr-1">{unionIssues.length === 0 && hasOmega ? '✓' : '✗'}</span>
            <span style={{ color: unionIssues.length === 0 && hasOmega ? 'var(--color-accent)' : 'var(--color-danger)' }}>
              Closed under unions
            </span>
          </div>
        </div>

        <div
          className="text-sm font-semibold"
          style={{ color: validation.valid ? 'var(--color-accent)' : 'var(--color-danger)' }}
        >
          {validation.valid
            ? `✓ Valid σ-algebra (${collection.length} events)`
            : `✗ Not a valid σ-algebra`}
        </div>

        {!validation.valid && validation.violations.length > 0 && (
          <ul className="mt-2 text-xs space-y-0.5" style={{ color: 'var(--color-text-muted)' }}>
            {validation.violations.slice(0, 5).map((v, i) => (
              <li key={i}>- {v}</li>
            ))}
            {validation.violations.length > 5 && (
              <li>...and {validation.violations.length - 5} more</li>
            )}
          </ul>
        )}
      </div>
    </div>
  );
}
