import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  zTestPower,
  tTestPower,
  twoProportionPower,
  binomialExactPower,
  requiredSampleSize,
  standardNormalInvCDF,
} from './shared/testing';
import {
  powerCurvePresets,
  type PowerCurvePreset,
  type PowerCurveScenario,
  type TestSide,
} from '../../data/hypothesis-testing-data';

const MARGIN = { top: 16, right: 24, bottom: 44, left: 52 };
const H = 320;

const CURVE_COLORS = ['#2563EB', '#059669', '#D97706', '#DC2626'];
const TARGET_POWER = 0.8;

type VaryAxis = 'effect' | 'n' | 'alpha';

/** Power at a given (scenario, δ, n, α, side). */
function computePower(
  scenario: PowerCurveScenario,
  delta: number,
  n: number,
  alpha: number,
  side: TestSide,
  opts: { sigma: number; baselineP: number; p0: number },
): number {
  if (scenario === 'z-one-sample') return zTestPower(delta, 0, opts.sigma, Math.round(n), alpha, side);
  if (scenario === 't-one-sample') return tTestPower(delta, 0, opts.sigma, Math.round(n), alpha, side);
  if (scenario === 'z-two-proportion') {
    const pA = opts.baselineP;
    const pB = pA + delta;
    if (pB <= 0 || pB >= 1) return NaN;
    return twoProportionPower(pB, pA, Math.round(n), Math.round(n), alpha, side);
  }
  // binomial-exact: delta is interpreted as pTrue − p₀
  const pTrue = Math.min(Math.max(opts.p0 + delta, 0.001), 0.999);
  return binomialExactPower(Math.round(n), opts.p0, pTrue, alpha, side === 'left' ? 'left' : 'right');
}

/** Required sample size via testing.ts for the three closed-form cases, or a
 *  bisection-style search for binomial-exact. */
function requiredN(
  scenario: PowerCurveScenario,
  delta: number,
  alpha: number,
  targetPower: number,
  side: TestSide,
  opts: { sigma: number; baselineP: number; p0: number },
): number {
  if (delta === 0) return -1;
  try {
    if (scenario === 'z-one-sample')
      return requiredSampleSize('z-one-sample', { delta, sigma: opts.sigma, alpha, power: targetPower }, side);
    if (scenario === 't-one-sample')
      return requiredSampleSize('t-one-sample', { delta, sigma: opts.sigma, alpha, power: targetPower }, side);
    if (scenario === 'z-two-proportion')
      return requiredSampleSize(
        'z-two-proportion',
        { delta, alpha, power: targetPower, baselineP: opts.baselineP },
        side,
      );
  } catch {
    return -1;
  }
  // binomial-exact: search incrementally
  for (let n = 5; n <= 5000; n++) {
    const p = computePower(scenario, delta, n, alpha, side, opts);
    if (p >= targetPower) return n;
  }
  return -1;
}

export default function PowerCurveExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 860;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: PowerCurvePreset = powerCurvePresets[presetIndex];

  const [refN, setRefN] = useState(preset.defaults.n);
  const [alpha, setAlpha] = useState(preset.defaults.alpha);
  const [sigma, setSigma] = useState(preset.defaults.sigma ?? 1);
  const [baselineP, setBaselineP] = useState(preset.defaults.baselineP ?? 0.1);
  const [p0, setP0] = useState(preset.defaults.p0 ?? 0.5);
  const [effect, setEffect] = useState(
    preset.scenario === 'z-two-proportion' ? (preset.defaults.lift ?? 0.02) : 0.5,
  );
  const [side, setSide] = useState<TestSide>('right');
  const [varyAxis, setVaryAxis] = useState<VaryAxis>('effect');

  useEffect(() => {
    setRefN(preset.defaults.n);
    setAlpha(preset.defaults.alpha);
    setSigma(preset.defaults.sigma ?? 1);
    setBaselineP(preset.defaults.baselineP ?? 0.1);
    setP0(preset.defaults.p0 ?? 0.5);
    setEffect(preset.scenario === 'z-two-proportion' ? (preset.defaults.lift ?? 0.02) : 0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  const opts = { sigma, baselineP, p0 };

  // Build four curves parameterized by the "other" axis
  const curves = useMemo(() => {
    const steps = 81;
    if (varyAxis === 'effect') {
      const nValues = [Math.max(5, Math.round(refN * 0.33)), refN, Math.round(refN * 2), Math.round(refN * 4)];
      const xRange: [number, number] =
        preset.scenario === 'z-two-proportion'
          ? [-0.1, 0.1]
          : preset.scenario === 'binomial-exact'
          ? [-0.5, 0.5]
          : [-2, 2];
      return nValues.map((nVal, i) => {
        const pts: [number, number][] = [];
        for (let k = 0; k < steps; k++) {
          const d = xRange[0] + (k * (xRange[1] - xRange[0])) / (steps - 1);
          pts.push([d, computePower(preset.scenario, d, nVal, alpha, side, opts)]);
        }
        return { label: `n = ${nVal}`, pts, color: CURVE_COLORS[i], xLabel: 'effect size δ', xDomain: xRange };
      });
    }
    if (varyAxis === 'n') {
      const effects =
        preset.scenario === 'z-two-proportion'
          ? [0.01, 0.02, 0.04, 0.08]
          : preset.scenario === 'binomial-exact'
          ? [0.1, 0.2, 0.3, 0.4]
          : [0.2, 0.5, 0.8, 1.2];
      const xRange: [number, number] = [5, Math.max(refN * 6, 200)];
      return effects.map((eff, i) => {
        const pts: [number, number][] = [];
        for (let k = 0; k < steps; k++) {
          const nv = xRange[0] + (k * (xRange[1] - xRange[0])) / (steps - 1);
          pts.push([nv, computePower(preset.scenario, eff, nv, alpha, side, opts)]);
        }
        return { label: `δ = ${eff}`, pts, color: CURVE_COLORS[i], xLabel: 'sample size n', xDomain: xRange };
      });
    }
    // vary α
    const effects =
      preset.scenario === 'z-two-proportion'
        ? [0.01, 0.02, 0.04, 0.08]
        : preset.scenario === 'binomial-exact'
        ? [0.1, 0.2, 0.3, 0.4]
        : [0.2, 0.5, 0.8, 1.2];
    const xRange: [number, number] = [0.001, 0.3];
    return effects.map((eff, i) => {
      const pts: [number, number][] = [];
      for (let k = 0; k < steps; k++) {
        const a = xRange[0] + (k * (xRange[1] - xRange[0])) / (steps - 1);
        pts.push([a, computePower(preset.scenario, eff, refN, a, side, opts)]);
      }
      return { label: `δ = ${eff}`, pts, color: CURVE_COLORS[i], xLabel: 'level α', xDomain: xRange };
    });
  }, [preset, varyAxis, refN, alpha, side, sigma, baselineP, p0]);

  const nReq = useMemo(
    () => requiredN(preset.scenario, effect, alpha, TARGET_POWER, side, opts),
    [preset.scenario, effect, alpha, side, sigma, baselineP, p0],
  );

  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.64) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    const xDomain = curves[0].xDomain;
    const x = d3.scaleLinear().domain(xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${H}`)
      .attr('width', svgW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(x).ticks(7))
      .call((sel) => sel.selectAll('text').style('font-size', '11px'))
      .call((sel) => sel.selectAll('line,path').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4));
    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
      .call((sel) => sel.selectAll('text').style('font-size', '11px'))
      .call((sel) => sel.selectAll('line,path').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4));

    // Target-power horizontal rule
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', y(TARGET_POWER))
      .attr('y2', y(TARGET_POWER))
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-dasharray', '3,3');
    g.append('text')
      .attr('x', innerW - 2)
      .attr('y', y(TARGET_POWER) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.6)
      .text('1 − β = 0.8');

    // Level-α horizontal rule
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', y(alpha))
      .attr('y2', y(alpha))
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.2)
      .attr('stroke-dasharray', '2,4');

    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(Math.min(Math.max(d[1], 0), 1)))
      .curve(d3.curveCatmullRom.alpha(0.5));

    curves.forEach((c) => {
      g.append('path')
        .datum(c.pts.filter(([, v]) => Number.isFinite(v)))
        .attr('d', line)
        .attr('stroke', c.color)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('fill-opacity', 0);
    });

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 100}, 6)`);
    curves.forEach((c, i) => {
      legend
        .append('rect')
        .attr('x', 0)
        .attr('y', i * 14)
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', c.color);
      legend
        .append('text')
        .attr('x', 14)
        .attr('y', i * 14 + 9)
        .style('font-size', '10px')
        .attr('fill', 'currentColor')
        .text(c.label);
    });

    // Axis titles
    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text(curves[0].xLabel);
    svg
      .append('text')
      .attr('transform', `translate(14, ${MARGIN.top + innerH / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('power β(θ)');
  }, [curves, alpha, svgW]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Power Curves · §17.4
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {powerCurvePresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            reference n = <strong>{refN}</strong>
          </span>
          <input type="range" min={5} max={preset.scenario === 'z-two-proportion' ? 10000 : 500} step={1} value={refN} onChange={(e) => setRefN(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            α = <strong>{alpha.toFixed(3)}</strong>
          </span>
          <input type="range" min={0.001} max={0.3} step={0.001} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
        </label>
        {(preset.scenario === 'z-one-sample' || preset.scenario === 't-one-sample') && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              σ = <strong>{sigma.toFixed(2)}</strong>
            </span>
            <input type="range" min={0.1} max={5} step={0.05} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
          </label>
        )}
        {preset.scenario === 'z-two-proportion' && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              baseline p_A = <strong>{baselineP.toFixed(3)}</strong>
            </span>
            <input type="range" min={0.01} max={0.5} step={0.005} value={baselineP} onChange={(e) => setBaselineP(Number(e.target.value))} />
          </label>
        )}
        {preset.scenario === 'binomial-exact' && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              p₀ = <strong>{p0.toFixed(2)}</strong>
            </span>
            <input type="range" min={0.05} max={0.95} step={0.01} value={p0} onChange={(e) => setP0(Number(e.target.value))} />
          </label>
        )}
        <div className="flex flex-col gap-1 text-xs">
          <span style={{ color: 'var(--color-text-muted)' }}>vary</span>
          <div className="flex gap-2 flex-wrap">
            {(['effect', 'n', 'alpha'] as VaryAxis[]).map((v) => (
              <label key={v} className="flex items-center gap-1">
                <input type="radio" name="varyAxis" checked={varyAxis === v} onChange={() => setVaryAxis(v)} />
                {v === 'effect' ? 'δ' : v === 'n' ? 'n' : 'α'}
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className={isWide ? 'grid grid-cols-[64%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <svg ref={svgRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-3">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Sample-size calculator
          </div>
          <div className="text-xs">
            Target power <strong>1 − β = {TARGET_POWER}</strong> at level <strong>α = {alpha.toFixed(3)}</strong>
          </div>
          <label className="flex flex-col gap-1 text-xs">
            <span style={{ color: 'var(--color-text-muted)' }}>
              effect size δ = <strong>{effect.toFixed(3)}</strong>
              {preset.scenario === 'z-two-proportion'
                ? ' (lift)'
                : preset.scenario === 'binomial-exact'
                ? ' (pTrue − p₀)'
                : ''}
            </span>
            <input
              type="range"
              min={preset.scenario === 'z-two-proportion' ? 0.005 : 0.05}
              max={preset.scenario === 'z-two-proportion' ? 0.2 : 2}
              step={preset.scenario === 'z-two-proportion' ? 0.005 : 0.025}
              value={effect}
              onChange={(e) => setEffect(Number(e.target.value))}
            />
          </label>
          <div className="rounded p-3 text-xs font-mono" style={{ background: 'rgba(37,99,235,0.08)' }}>
            Required{' '}
            <span style={{ color: 'var(--color-text-muted)' }}>
              {preset.scenario === 'z-two-proportion' ? 'n per arm' : 'n'}
            </span>
            :{' '}
            <strong className="text-base">{nReq > 0 ? nReq : '—'}</strong>
          </div>
          <div className="text-[11px] leading-relaxed" style={{ color: 'var(--color-text)' }}>
            <strong>Closed form:</strong>{' '}
            <code className="text-[10px]">{preset.closedForm}</code>
          </div>
          <div className="text-[11px] leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
            {preset.sampleSizeFormula}
          </div>
          {preset.scenario === 'binomial-exact' && (
            <div
              className="text-[11px] leading-relaxed p-2 rounded"
              style={{ background: 'rgba(217,119,6,0.1)' }}
            >
              The binomial exact sample-size uses numerical search over n — not the Normal approximation — consistent with §17.6 Example 11. The resulting n can be noticeably larger than a naive z-based calculation suggests, because the exact test is conservative.
            </div>
          )}
          <div
            className="text-[10px]"
            style={{ color: 'var(--color-text-muted)' }}
          >
            critical z at α/2: ±{standardNormalInvCDF(1 - alpha / 2).toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
