import { useState, useMemo } from 'react';
import { sampleSpacePresets } from '../../data/sample-spaces-data';
import { equallyLikelyP, complement } from './shared/probability';

export default function SampleSpaceExplorer() {
  const [presetIdx, setPresetIdx] = useState(1); // Die Roll default
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [showComplement, setShowComplement] = useState(false);

  const preset = sampleSpacePresets[presetIdx];
  const omega = useMemo(() => new Set(preset.outcomes), [preset]);

  const toggleOutcome = (o: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(o)) next.delete(o);
      else next.add(o);
      return next;
    });
  };

  const handlePresetChange = (idx: number) => {
    setPresetIdx(idx);
    setSelected(new Set());
    setShowComplement(false);
  };

  const eventA = selected;
  const comp = complement(eventA, omega);
  const pA = equallyLikelyP(eventA, omega);

  const isGrid = preset.layout === 'grid';
  const cols = preset.gridCols ?? 6;

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <label className="text-sm font-semibold" style={{ color: 'var(--color-text-heading)' }}>
          Experiment:
        </label>
        <select
          value={presetIdx}
          onChange={(e) => handlePresetChange(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{
            background: 'var(--color-surface)',
            borderColor: 'var(--color-border)',
            color: 'var(--color-text)',
          }}
        >
          {sampleSpacePresets.map((p, i) => (
            <option key={p.name} value={i}>
              {p.name}
            </option>
          ))}
        </select>
        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {preset.description}
        </span>
      </div>

      {/* Outcome grid */}
      <div className="mb-4">
        <div className="text-xs font-semibold mb-2" style={{ color: 'var(--color-text-muted)' }}>
          Click outcomes to select event A (|&Omega;| = {preset.outcomes.length})
        </div>
        <div
          className={isGrid ? 'grid gap-1' : 'flex flex-wrap gap-1.5'}
          style={isGrid ? { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` } : undefined}
        >
          {preset.outcomes.map((o) => {
            const inA = eventA.has(o);
            const inComp = showComplement && comp.has(o);
            let bg = 'var(--color-surface)';
            let border = 'var(--color-border)';
            let textColor = 'var(--color-text)';
            if (inA) {
              bg = '#dbeafe';
              border = '#2563eb';
              textColor = '#1e40af';
            } else if (inComp) {
              bg = '#fee2e2';
              border = '#dc2626';
              textColor = '#991b1b';
            }
            return (
              <button
                key={o}
                onClick={() => toggleOutcome(o)}
                className="rounded border text-xs sm:text-sm font-mono px-2 py-1 transition-colors cursor-pointer hover:opacity-80"
                style={{ background: bg, borderColor: border, color: textColor }}
              >
                {o}
              </button>
            );
          })}
        </div>
      </div>

      {/* Controls and readout */}
      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-1.5 text-sm cursor-pointer" style={{ color: 'var(--color-text)' }}>
          <input
            type="checkbox"
            checked={showComplement}
            onChange={(e) => setShowComplement(e.target.checked)}
            className="rounded"
          />
          Show A<sup>c</sup>
        </label>

        <div className="flex-1" />

        <div className="text-sm" style={{ color: 'var(--color-text)' }}>
          <span style={{ color: 'var(--color-text-muted)' }}>|A| = </span>
          <span className="font-semibold">{eventA.size}</span>
          <span style={{ color: 'var(--color-text-muted)' }}> &nbsp;|&Omega;| = </span>
          <span className="font-semibold">{omega.size}</span>
          <span style={{ color: 'var(--color-text-muted)' }}> &nbsp;P(A) = </span>
          <span className="font-semibold">{omega.size > 0 ? pA.toFixed(4) : '—'}</span>
        </div>
      </div>
    </div>
  );
}
