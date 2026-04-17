import { useState, useMemo, useRef, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  basuIndependenceNormal,
  basuDependenceExponentialShift,
} from './shared/estimation';
import { basuPresets } from '../../data/sufficient-statistics-data';

const MARGIN = { top: 14, right: 16, bottom: 36, left: 48 };
const H = 280;

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export default function BasuIndependence() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset = basuPresets[presetIndex];

  const [n, setN] = useState(30);
  const [M, setM] = useState(3000);
  const [seed, setSeed] = useState(13);

  useEffect(() => { setSeed((s) => s + 1); }, [presetIndex]);

  const mc = useMemo(() => {
    const rng = makeLCG(seed * 7919 + n + M);
    if (preset.family === 'normal') {
      const r = basuIndependenceNormal(preset.defaultParams.mu ?? 0, preset.defaultParams.sigma2 ?? 1, n, M, rng);
      return { x: r.xbar, y: r.s2, correlation: r.correlation, xLabel: 'X̄ (complete sufficient for μ)', yLabel: 'S² (ancillary for μ)' };
    }
    const r = basuDependenceExponentialShift(preset.defaultParams.mu ?? 0, n, M, rng);
    return { x: r.min, y: r.xbar, correlation: r.correlation, xLabel: 'X₍₁₎ (complete sufficient for μ)', yLabel: 'X̄ (NOT ancillary — E[X̄] = μ + 1)' };
  }, [preset.family, preset.defaultParams, n, M, seed]);

  // Canvas scatter (>1000 points) for perf per brief
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasW = isWide ? Math.floor(w * 0.6) : w;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${canvasW}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, canvasW, H);

    const innerW = canvasW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // Defensive guard: empty arrays make Math.min(...[]) = +Infinity and spread
    // on very large arrays can exhaust the call stack. Compute extents with a
    // single pass and fall back to a sane unit domain when the array is empty.
    if (mc.x.length === 0 || mc.y.length === 0) return;
    let xMin = mc.x[0];
    let xMax = mc.x[0];
    for (let i = 1; i < mc.x.length; i++) {
      if (mc.x[i] < xMin) xMin = mc.x[i];
      if (mc.x[i] > xMax) xMax = mc.x[i];
    }
    let yMin = mc.y[0];
    let yMax = mc.y[0];
    for (let i = 1; i < mc.y.length; i++) {
      if (mc.y[i] < yMin) yMin = mc.y[i];
      if (mc.y[i] > yMax) yMax = mc.y[i];
    }
    const xPad = (xMax - xMin) * 0.05 || 0.1;
    const yPad = (yMax - yMin) * 0.05 || 0.1;

    const sx = (v: number) => MARGIN.left + (innerW * (v - (xMin - xPad))) / ((xMax + xPad) - (xMin - xPad));
    const sy = (v: number) => MARGIN.top + innerH - (innerH * (v - (yMin - yPad))) / ((yMax + yPad) - (yMin - yPad));

    // Axes
    ctx.strokeStyle = 'currentColor';
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + innerH);
    ctx.lineTo(MARGIN.left + innerW, MARGIN.top + innerH);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Tick labels
    ctx.fillStyle = 'currentColor';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.globalAlpha = 0.75;
    for (let i = 0; i <= 5; i++) {
      const tx = xMin - xPad + (i / 5) * ((xMax + xPad) - (xMin - xPad));
      ctx.fillText(tx.toFixed(2), sx(tx), MARGIN.top + innerH + 14);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const ty = yMin - yPad + (i / 5) * ((yMax + yPad) - (yMin - yPad));
      ctx.fillText(ty.toFixed(2), MARGIN.left - 4, sy(ty) + 3);
    }
    ctx.globalAlpha = 1;

    // Points
    const color = preset.basuHolds ? 'rgba(70,160,90,0.35)' : 'rgba(220,100,70,0.35)';
    ctx.fillStyle = color;
    for (let i = 0; i < mc.x.length; i++) {
      const px = sx(mc.x[i]);
      const py = sy(mc.y[i]);
      ctx.fillRect(px - 1, py - 1, 2, 2);
    }

    // Axis labels
    ctx.fillStyle = 'currentColor';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(mc.xLabel, MARGIN.left + innerW / 2, H - 6);
    ctx.save();
    ctx.translate(10, MARGIN.top + innerH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(mc.yLabel, 0, 0);
    ctx.restore();
  }, [mc, canvasW, preset.basuHolds]);

  const rho = mc.correlation;

  return (
    <div ref={containerRef} className="not-prose my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}>
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Basu Independence · §16.9
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {basuPresets.map((p, i) => (
            <option key={i} value={i}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>n = {n}</span>
          <input type="range" min={5} max={200} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-full" />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>M = {M}</span>
          <input type="range" min={500} max={8000} step={500} value={M} onChange={(e) => setM(Number(e.target.value))} className="w-full" />
        </label>
        <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          Sample ρ = <strong style={{ color: preset.basuHolds ? 'rgb(40,160,80)' : 'rgb(220,100,70)' }}>{rho.toFixed(4)}</strong><br />
          {preset.basuHolds ? 'Theoretical: 0 (Basu)' : `Theoretical: ≈ 1/√n = ${(1 / Math.sqrt(n)).toFixed(3)}`}
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-run MC
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-4'}>
        <canvas ref={canvasRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-2">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            What Basu's theorem says
          </div>
          <p className="text-xs">
            If T is complete sufficient for θ and A is ancillary for θ, then T ⊥⊥ A under every P_θ. Visualized as: their joint MC scatter should be decorrelated — sample ρ ≈ 0.
          </p>
          <div className="text-xs p-2 rounded" style={{ background: preset.basuHolds ? 'rgba(70,160,90,0.12)' : 'rgba(220,100,70,0.12)', color: preset.basuHolds ? 'rgb(40,140,70)' : 'rgb(180,80,50)' }}>
            {preset.basuHolds ? (
              <>
                ✓ <strong>Basu applies.</strong> X̄ is complete sufficient for μ (σ² fixed), S² is ancillary for μ — so they are independent. This is the foundation of the Student's t-distribution: t = √n(X̄ − μ)/S has a well-defined distribution because its numerator and denominator are independent.
              </>
            ) : (
              <>
                ✗ <strong>Basu does NOT apply.</strong> X̄ is not ancillary for μ — its mean shifts with μ. So X̄ and X₍₁₎ are correlated, with ρ ≈ 1/√n. The plausible-looking X̄ − X₍₁₎ would actually BE ancillary by exponential memorylessness, so pairing it with X₍₁₎ would NOT violate Basu.
              </>
            )}
          </div>
          <div className="text-xs italic" style={{ color: 'var(--color-text-muted)' }}>{preset.description}</div>
        </div>
      </div>
    </div>
  );
}
