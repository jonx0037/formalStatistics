import { useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { inclusionExclusion2, inclusionExclusion3 } from './shared/probability';

type Mode = '2-event' | '3-event';

interface Term {
  label: string;
  value: number;
  sign: '+' | '-';
  color: string;
}

export default function InclusionExclusionExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>('2-event');
  const [step, setStep] = useState(0);
  const [pA, setPA] = useState(0.5);
  const [pB, setPB] = useState(0.4);
  const [pC, setPC] = useState(0.35);
  const [pAB, setPAB] = useState(0.15);
  const [pAC, setPAC] = useState(0.12);
  const [pBC, setPBC] = useState(0.1);
  const [pABC] = useState(0.04);

  const terms2: Term[] = [
    { label: 'P(A)', value: pA, sign: '+', color: '#2563eb' },
    { label: 'P(B)', value: pB, sign: '+', color: '#7c3aed' },
    { label: 'P(A∩B)', value: pAB, sign: '-', color: '#059669' },
  ];

  const terms3: Term[] = [
    { label: 'P(A)', value: pA, sign: '+', color: '#2563eb' },
    { label: 'P(B)', value: pB, sign: '+', color: '#7c3aed' },
    { label: 'P(C)', value: pC, sign: '+', color: '#d97706' },
    { label: 'P(A∩B)', value: pAB, sign: '-', color: '#059669' },
    { label: 'P(A∩C)', value: pAC, sign: '-', color: '#0891b2' },
    { label: 'P(B∩C)', value: pBC, sign: '-', color: '#be185d' },
    { label: 'P(A∩B∩C)', value: pABC, sign: '+', color: '#6b7280' },
  ];

  const terms = mode === '2-event' ? terms2 : terms3;
  const maxStep = terms.length;
  const currentStep = Math.min(step, maxStep);

  // Running total
  let runningTotal = 0;
  for (let i = 0; i < currentStep; i++) {
    runningTotal += terms[i].sign === '+' ? terms[i].value : -terms[i].value;
  }

  const exact = mode === '2-event'
    ? inclusionExclusion2(pA, pB, pAB)
    : inclusionExclusion3(pA, pB, pC, pAB, pAC, pBC, pABC);

  // Venn diagram geometry
  const h = Math.min(width * 0.5, 240);
  const centerX = width / 2;
  const centerY = h / 2;
  const r = Math.min(width * 0.18, 80);
  const offset = r * 0.55;

  const circles = mode === '2-event'
    ? [
      { cx: centerX - offset, cy: centerY, label: 'A', color: '#2563eb' },
      { cx: centerX + offset, cy: centerY, label: 'B', color: '#7c3aed' },
    ]
    : [
      { cx: centerX, cy: centerY - offset * 0.7, label: 'A', color: '#2563eb' },
      { cx: centerX - offset * 0.9, cy: centerY + offset * 0.5, label: 'B', color: '#7c3aed' },
      { cx: centerX + offset * 0.9, cy: centerY + offset * 0.5, label: 'C', color: '#d97706' },
    ];

  return (
    <div
      className="my-6 rounded-lg border p-4 sm:p-6"
      style={{ background: 'var(--color-surface-alt)', borderColor: 'var(--color-border)' }}
    >
      {/* Mode toggle */}
      <div className="flex items-center gap-3 mb-4">
        {(['2-event', '3-event'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); setStep(0); }}
            className="px-3 py-1 rounded text-sm font-medium border transition-colors cursor-pointer"
            style={{
              background: mode === m ? '#2563eb' : 'var(--color-surface)',
              color: mode === m ? '#ffffff' : 'var(--color-text)',
              borderColor: mode === m ? '#2563eb' : 'var(--color-border)',
            }}
          >
            {m === '2-event' ? '2 Events' : '3 Events'}
          </button>
        ))}
      </div>

      {/* Venn diagram */}
      <div ref={containerRef} className="w-full">
        <svg width={width} height={h} viewBox={`0 0 ${width} ${h}`}>
          <rect x={2} y={2} width={width - 4} height={h - 4} rx={8}
            fill="none" stroke="var(--color-border-strong)" strokeWidth={1} />
          {circles.map((c, i) => (
            <g key={i}>
              <circle
                cx={c.cx} cy={c.cy} r={r}
                fill={i < currentStep ? c.color : 'transparent'}
                fillOpacity={i < currentStep ? 0.15 : 0}
                stroke={c.color}
                strokeWidth={2}
              />
              <text
                x={c.cx + (mode === '2-event' ? (i === 0 ? -r * 0.5 : r * 0.5) : 0)}
                y={c.cy + (mode === '3-event' && i === 0 ? -r * 0.4 : mode === '3-event' ? r * 0.4 : 0)}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={16} fontWeight={700} fill={c.color}
              >
                {c.label}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Step controls */}
      <div className="flex items-center gap-3 mt-4 mb-3">
        <button
          onClick={() => setStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-3 py-1 rounded text-sm border cursor-pointer disabled:opacity-30"
          style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          &larr; Back
        </button>
        <button
          onClick={() => setStep(Math.min(maxStep, currentStep + 1))}
          disabled={currentStep === maxStep}
          className="px-3 py-1 rounded text-sm border cursor-pointer disabled:opacity-30"
          style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          Next &rarr;
        </button>
        <button
          onClick={() => setStep(0)}
          className="px-3 py-1 rounded text-xs border cursor-pointer"
          style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}
        >
          Reset
        </button>
        <span className="text-xs ml-auto" style={{ color: 'var(--color-text-muted)' }}>
          Step {currentStep} / {maxStep}
        </span>
      </div>

      {/* Terms list */}
      <div className="space-y-1 mb-3">
        {terms.map((t, i) => (
          <div
            key={i}
            className="flex items-center gap-2 text-sm font-mono rounded px-2 py-1 transition-opacity"
            style={{
              opacity: i < currentStep ? 1 : 0.3,
              background: i < currentStep ? 'var(--color-surface)' : 'transparent',
            }}
          >
            <span style={{ color: t.sign === '+' ? 'var(--color-accent)' : 'var(--color-danger)' }}>
              {t.sign}
            </span>
            <span style={{ color: t.color, fontWeight: 600 }}>{t.label}</span>
            <span style={{ color: 'var(--color-text-muted)' }}>=</span>
            <span style={{ color: 'var(--color-text)' }}>{t.value.toFixed(4)}</span>
          </div>
        ))}
      </div>

      {/* Result */}
      <div className="flex flex-wrap items-center gap-4 text-sm" style={{ color: 'var(--color-text)' }}>
        <span>
          Running total: <span className="font-semibold">{runningTotal.toFixed(4)}</span>
        </span>
        <span style={{ color: 'var(--color-text-muted)' }}>|</span>
        <span>
          Exact P(A{mode === '2-event' ? '∪B' : '∪B∪C'}): <span className="font-semibold">{exact.toFixed(4)}</span>
        </span>
        {currentStep === maxStep && (
          <span style={{ color: Math.abs(runningTotal - exact) < 0.0001 ? 'var(--color-accent)' : 'var(--color-danger)' }}>
            {Math.abs(runningTotal - exact) < 0.0001 ? '✓ Match!' : '✗ Mismatch'}
          </span>
        )}
      </div>

      {/* Sliders */}
      <details className="mt-4">
        <summary className="text-xs cursor-pointer" style={{ color: 'var(--color-text-muted)' }}>
          Adjust probabilities
        </summary>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-2">
          <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            P(A) = {pA.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={pA} onChange={(e) => setPA(+e.target.value)} className="w-full" />
          </label>
          <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            P(B) = {pB.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={pB} onChange={(e) => setPB(+e.target.value)} className="w-full" />
          </label>
          {mode === '3-event' && (
            <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              P(C) = {pC.toFixed(2)}
              <input type="range" min={0} max={1} step={0.01} value={pC} onChange={(e) => setPC(+e.target.value)} className="w-full" />
            </label>
          )}
          <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            P(A∩B) = {pAB.toFixed(2)}
            <input type="range" min={0} max={Math.min(pA, pB)} step={0.01} value={pAB} onChange={(e) => setPAB(+e.target.value)} className="w-full" />
          </label>
          {mode === '3-event' && (
            <>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                P(A∩C) = {pAC.toFixed(2)}
                <input type="range" min={0} max={Math.min(pA, pC)} step={0.01} value={pAC} onChange={(e) => setPAC(+e.target.value)} className="w-full" />
              </label>
              <label className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                P(B∩C) = {pBC.toFixed(2)}
                <input type="range" min={0} max={Math.min(pB, pC)} step={0.01} value={pBC} onChange={(e) => setPBC(+e.target.value)} className="w-full" />
              </label>
            </>
          )}
        </div>
      </details>
    </div>
  );
}
