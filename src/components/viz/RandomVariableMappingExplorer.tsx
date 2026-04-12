import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { rvMappingPresets } from '../../data/random-variables-data';

// ── Colors ─────────────────────────────────────────────────────────────────

const TARGET_COLORS = [
  '#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed',
  '#0891b2', '#be185d', '#4f46e5', '#ea580c', '#16a34a',
  '#6d28d9', '#0d9488', '#c2410c', '#4338ca', '#0369a1',
];

function getTargetColor(idx: number): string {
  return TARGET_COLORS[idx % TARGET_COLORS.length];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function computePreImages(mapping: Record<string, number>): Map<number, string[]> {
  const preImages = new Map<number, string[]>();
  for (const [omega, value] of Object.entries(mapping)) {
    const existing = preImages.get(value) ?? [];
    existing.push(omega);
    preImages.set(value, existing);
  }
  return new Map([...preImages.entries()].sort((a, b) => a[0] - b[0]));
}

export default function RandomVariableMappingExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [presetIdx, setPresetIdx] = useState(0);
  const [mappingName, setMappingName] = useState<string>('');

  const preset = rvMappingPresets[presetIdx];
  const mappingNames = Object.keys(preset.mappings);
  const activeMappingName = mappingName && mappingNames.includes(mappingName)
    ? mappingName
    : mappingNames[0];
  const mapping = preset.mappings[activeMappingName];

  const preImages = useMemo(() => computePreImages(mapping), [mapping]);
  const targetValues = useMemo(() => [...preImages.keys()], [preImages]);
  const targetColorMap = useMemo(() => {
    const m = new Map<number, string>();
    targetValues.forEach((v, i) => m.set(v, getTargetColor(i)));
    return m;
  }, [targetValues]);

  const omegaSize = preset.omega.length;
  const isTwoDice = preset.name === 'Two Dice';

  // ── Layout ──────────────────────────────────────────────────────────────

  const compact = width < 500;
  const svgW = Math.max(width, 300);
  const svgH = isTwoDice ? (compact ? 340 : 300) : (compact ? 260 : 220);

  const leftCx = svgW * 0.22;
  const rightCx = svgW * 0.78;
  const centerY = svgH * 0.5;

  // ── Compute Ω positions ─────────────────────────────────────────────────

  const omegaPositions = useMemo(() => {
    if (isTwoDice) {
      // 6×6 grid
      const gridSize = compact ? 24 : 28;
      const gap = compact ? 2 : 3;
      const startX = leftCx - 3 * (gridSize + gap) + (gridSize + gap) / 2;
      const startY = centerY - 3 * (gridSize + gap) + (gridSize + gap) / 2;
      return preset.omega.map((label, idx) => {
        const row = Math.floor(idx / 6);
        const col = idx % 6;
        return {
          label,
          x: startX + col * (gridSize + gap),
          y: startY + row * (gridSize + gap),
        };
      });
    }
    // Arrange in a vertical column
    const spacing = Math.min(36, (svgH - 40) / omegaSize);
    const startY = centerY - ((omegaSize - 1) * spacing) / 2;
    return preset.omega.map((label, idx) => ({
      label,
      x: leftCx,
      y: startY + idx * spacing,
    }));
  }, [preset, leftCx, centerY, svgH, omegaSize, isTwoDice, compact]);

  // ── Compute target positions ────────────────────────────────────────────

  const targetPositions = useMemo(() => {
    const count = targetValues.length;
    const spacing = Math.min(36, (svgH - 40) / count);
    const startY = centerY - ((count - 1) * spacing) / 2;
    return targetValues.map((val, idx) => ({
      value: val,
      x: rightCx,
      y: startY + idx * spacing,
    }));
  }, [targetValues, rightCx, centerY, svgH]);

  // ── Readout ─────────────────────────────────────────────────────────────

  const readoutData = useMemo(() => {
    return targetValues.map((val) => {
      const preImg = preImages.get(val) ?? [];
      const prob = preImg.length / omegaSize;
      return { value: val, preImage: preImg, probability: prob };
    });
  }, [targetValues, preImages, omegaSize]);

  // ── Handlers ────────────────────────────────────────────────────────────

  const handlePresetChange = (idx: number) => {
    setPresetIdx(idx);
    setMappingName('');
  };

  return (
    <div ref={containerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium">Experiment:</label>
        <select
          value={presetIdx}
          onChange={(e) => handlePresetChange(Number(e.target.value))}
          className="rounded border px-2 py-1 text-sm"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {rvMappingPresets.map((p, i) => (
            <option key={p.name} value={i}>{p.name}</option>
          ))}
        </select>

        <label className="text-sm font-medium">Mapping:</label>
        <select
          value={activeMappingName}
          onChange={(e) => setMappingName(e.target.value)}
          className="rounded border px-2 py-1 text-sm"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {mappingNames.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {/* SVG Mapping Diagram */}
      <svg width={svgW} height={svgH} className="block">
        <defs>
          {targetValues.map((val) => (
            <marker
              key={`arrow-${val}`}
              id={`arrow-${val}`}
              markerWidth="8"
              markerHeight="6"
              refX="8"
              refY="3"
              orient="auto"
            >
              <polygon
                points="0,0 8,3 0,6"
                fill={targetColorMap.get(val)}
                opacity={0.7}
              />
            </marker>
          ))}
        </defs>

        {/* Labels */}
        <text x={leftCx} y={16} textAnchor="middle" className="fill-current text-sm font-semibold">Ω</text>
        <text x={rightCx} y={16} textAnchor="middle" className="fill-current text-sm font-semibold">ℝ</text>

        {/* Arrows */}
        {omegaPositions.map((op) => {
          const val = mapping[op.label];
          const target = targetPositions.find((t) => t.value === val);
          if (!target) return null;
          const color = targetColorMap.get(val) ?? '#888';
          return (
            <line
              key={`arrow-${op.label}`}
              x1={op.x + (isTwoDice ? 14 : 20)}
              y1={op.y}
              x2={target.x - 20}
              y2={target.y}
              stroke={color}
              strokeWidth={1.5}
              opacity={0.5}
              markerEnd={`url(#arrow-${val})`}
            />
          );
        })}

        {/* Ω outcomes */}
        {isTwoDice ? (
          omegaPositions.map((op) => {
            const val = mapping[op.label];
            const color = targetColorMap.get(val) ?? '#888';
            const cellSize = compact ? 24 : 28;
            return (
              <g key={`omega-${op.label}`}>
                <rect
                  x={op.x - cellSize / 2}
                  y={op.y - cellSize / 2}
                  width={cellSize}
                  height={cellSize}
                  fill={color}
                  opacity={0.15}
                  stroke={color}
                  strokeWidth={1}
                  rx={2}
                />
              </g>
            );
          })
        ) : (
          omegaPositions.map((op) => {
            const val = mapping[op.label];
            const color = targetColorMap.get(val) ?? '#888';
            return (
              <g key={`omega-${op.label}`}>
                <circle cx={op.x} cy={op.y} r={14} fill={color} opacity={0.15} stroke={color} strokeWidth={1.5} />
                <text x={op.x} y={op.y + 4} textAnchor="middle" className="fill-current" style={{ fontSize: '11px' }}>
                  {op.label}
                </text>
              </g>
            );
          })
        )}

        {/* Target values */}
        {targetPositions.map((tp) => {
          const color = targetColorMap.get(tp.value) ?? '#888';
          return (
            <g key={`target-${tp.value}`}>
              <circle cx={tp.x} cy={tp.y} r={16} fill={color} opacity={0.2} stroke={color} strokeWidth={2} />
              <text x={tp.x} y={tp.y + 5} textAnchor="middle" className="fill-current font-semibold" style={{ fontSize: '12px' }}>
                {tp.value}
              </text>
            </g>
          );
        })}

        {/* X mapping label */}
        <text x={svgW / 2} y={svgH - 8} textAnchor="middle" className="fill-current" style={{ fontSize: '13px', fontStyle: 'italic' }}>
          X : Ω → ℝ
        </text>
      </svg>

      {/* Readout Panel */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm" style={{ borderColor: 'var(--color-border)' }}>
          <thead>
            <tr className="border-b" style={{ borderColor: 'var(--color-border)' }}>
              <th className="px-2 py-1 text-left">Value x</th>
              <th className="px-2 py-1 text-left">Pre-image X⁻¹({'{'}{'{'}x{'}'}{'}'})</th>
              <th className="px-2 py-1 text-left">P(X = x)</th>
            </tr>
          </thead>
          <tbody>
            {readoutData.map((row) => (
              <tr key={row.value} className="border-b" style={{ borderColor: 'var(--color-border)' }}>
                <td className="px-2 py-1 font-mono" style={{ color: targetColorMap.get(row.value) }}>
                  {row.value}
                </td>
                <td className="px-2 py-1 font-mono text-xs">
                  {'{'}
                  {isTwoDice && row.preImage.length > 4
                    ? `${row.preImage.slice(0, 3).join(', ')}, ... (${row.preImage.length} total)`
                    : row.preImage.join(', ')}
                  {'}'}
                </td>
                <td className="px-2 py-1 font-mono">
                  {row.preImage.length}/{omegaSize} = {(row.probability).toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
