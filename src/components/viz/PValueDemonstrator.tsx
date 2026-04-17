import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  standardNormalPDF,
  studentTPDF,
  zTestStatistic,
  tTestStatistic,
  twoProportionZStatistic,
  zTestPValue,
  tTestPValue,
  twoProportionPValue,
  binomialExactPValue,
} from './shared/testing';
import { normalSample, bernoulliSample } from './shared/convergence';
import { seededRandom } from './shared/probability';
import {
  pValueScenarios,
  type PValueScenario,
  type TestSide,
} from '../../data/hypothesis-testing-data';

const MARGIN = { top: 16, right: 16, bottom: 40, left: 48 };
const H = 300;

type Mode = 'single' | 'mc-h0' | 'mc-ha';

/** Generate one sample and return its test statistic and p-value for the scenario. */
function drawStatAndP(
  scenario: PValueScenario,
  rng: () => number,
  n: number,
  trueTheta: number,
  side: TestSide,
): { t: number; p: number } {
  if (scenario.family === 'normal-z') {
    const data = Array.from({ length: n }, () => normalSample(trueTheta, 1, rng));
    const t = zTestStatistic(data, 0, 1);
    return { t, p: zTestPValue(t, side) };
  }
  if (scenario.family === 'normal-t') {
    const data = Array.from({ length: n }, () => normalSample(trueTheta, 1, rng));
    const t = tTestStatistic(data, 0);
    return { t, p: tTestPValue(t, n - 1, side) };
  }
  if (scenario.family === 'two-proportion') {
    // trueTheta = lift (p_B − p_A). `twoProportionZStatistic(s1, n1, s2, n2)`
    // computes (p̂₁ − p̂₂)/SE, so we pass B first to make a positive lift
    // produce a positive z — matching the UI's "lift" framing and the
    // selected tail. Copilot review #3103512348.
    const pA = 0.1;
    const pB = pA + trueTheta;
    const a = Array.from({ length: n }, () => bernoulliSample(pA, rng));
    const b = Array.from({ length: n }, () => bernoulliSample(pB, rng));
    const sA = a.reduce((s, x) => s + x, 0);
    const sB = b.reduce((s, x) => s + x, 0);
    const t = twoProportionZStatistic(sB, n, sA, n);
    return { t, p: twoProportionPValue(sB, n, sA, n, side) };
  }
  // binomial-exact: trueTheta in [0,1] is the true p
  const n0 = (scenario.nullParams.n ?? 20);
  const p0 = (scenario.nullParams.p ?? 0.5);
  const data = Array.from({ length: n0 }, () => bernoulliSample(trueTheta, rng));
  const x = data.reduce((s, v) => s + v, 0);
  return { t: x, p: binomialExactPValue(x, n0, p0, side) };
}

export default function PValueDemonstrator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 820;

  const [presetIndex, setPresetIndex] = useState(0);
  const scenario = pValueScenarios[presetIndex];
  const [mode, setMode] = useState<Mode>('single');
  const [n, setN] = useState(30);
  const [trueTheta, setTrueTheta] = useState(0);
  const [M, setM] = useState(3000);
  const [side, setSide] = useState<TestSide>('two');
  const [seed, setSeed] = useState(7);

  useEffect(() => {
    setN(scenario.family === 'binomial-exact' ? (scenario.nullParams.n ?? 20) : 30);
    setTrueTheta(0);
    setMode('single');
    if (!scenario.sides.includes(side)) setSide(scenario.sides[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  // Single-sample mode: one draw, one tail
  const singleDraw = useMemo(() => {
    if (mode !== 'single') return null;
    const rng = seededRandom(seed);
    return drawStatAndP(scenario, rng, n, trueTheta, side);
  }, [scenario, seed, n, trueTheta, side, mode]);

  // MC mode: M draws, build p-value histogram
  const mcPValues = useMemo(() => {
    if (mode === 'single') return [] as number[];
    const rng = seededRandom(seed * 31);
    // Under H₀ use trueTheta = 0 (two-proportion: lift = 0; binomial: p = p₀).
    // Under H_A use the slider's trueTheta directly.
    const theta =
      mode === 'mc-h0'
        ? scenario.family === 'binomial-exact'
          ? scenario.nullParams.p ?? 0.5
          : 0
        : scenario.family === 'binomial-exact'
        ? trueTheta
        : trueTheta;
    const out: number[] = [];
    for (let i = 0; i < M; i++) {
      out.push(drawStatAndP(scenario, rng, n, theta, side).p);
    }
    return out;
  }, [scenario, seed, n, trueTheta, M, side, mode]);

  const rejectionRate = useMemo(() => {
    if (mcPValues.length === 0) return 0;
    let r = 0;
    for (const p of mcPValues) if (p <= 0.05) r++;
    return r / mcPValues.length;
  }, [mcPValues]);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.6) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${H}`)
      .attr('width', svgW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    if (mode === 'single' && singleDraw) {
      // Plot null density with observed T marked and tail shaded
      const isT = scenario.family === 'normal-t';
      const isBinomial = scenario.family === 'binomial-exact';
      const xMin = isBinomial ? 0 : -4.5;
      const xMax = isBinomial ? (scenario.nullParams.n ?? 20) : 4.5;
      const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
      const samples = 181;
      const step = (xMax - xMin) / (samples - 1);

      if (isBinomial) {
        // PMF bars
        const n0 = scenario.nullParams.n ?? 20;
        const p0 = scenario.nullParams.p ?? 0.5;
        const bars: [number, number][] = [];
        for (let k = 0; k <= n0; k++) {
          // Compute binomial pmf on the fly (imported from distributions is cleaner, but this is trivial).
          const logC = (() => {
            let s = 0;
            for (let j = 1; j <= k; j++) s += Math.log((n0 - j + 1) / j);
            return s;
          })();
          const pmf = Math.exp(logC + k * Math.log(p0) + (n0 - k) * Math.log(1 - p0));
          bars.push([k, pmf]);
        }
        const maxY = Math.max(...bars.map((b) => b[1])) * 1.1;
        const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);
        bars.forEach(([k, v]) => {
          const isTail = side === 'right' ? k >= singleDraw.t : side === 'left' ? k <= singleDraw.t : v <= bars[Math.round(singleDraw.t)]?.[1] + 1e-12;
          g.append('rect')
            .attr('x', x(k) - 3)
            .attr('y', y(v))
            .attr('width', 6)
            .attr('height', innerH - y(v))
            .attr('fill', isTail ? '#F59E0B' : '#059669')
            .attr('fill-opacity', 0.75);
        });
        g.append('line')
          .attr('x1', x(singleDraw.t))
          .attr('x2', x(singleDraw.t))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', '#DC2626')
          .attr('stroke-width', 2);
        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(10));
        g.append('g').call(d3.axisLeft(y).ticks(5));
      } else {
        const nullDensity = isT ? (z: number) => studentTPDF(z, n - 1) : standardNormalPDF;
        const pts: [number, number][] = Array.from({ length: samples }, (_, i) => {
          const z = xMin + i * step;
          return [z, nullDensity(z)];
        });
        const y = d3.scaleLinear().domain([0, 0.45]).range([innerH, 0]);
        const line = d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveCatmullRom.alpha(0.3));
        const area = d3.area<[number, number]>().x((d) => x(d[0])).y0(y(0)).y1((d) => y(d[1])).curve(d3.curveCatmullRom.alpha(0.3));

        // Shade p-value tail
        const t = singleDraw.t;
        let tail: [number, number][] = [];
        if (side === 'right') tail = pts.filter(([z]) => z >= t);
        else if (side === 'left') tail = pts.filter(([z]) => z <= t);
        else {
          const abs = Math.abs(t);
          tail = pts.filter(([z]) => Math.abs(z) >= abs);
        }
        // Split tail area for two-sided into two path segments to avoid connecting line through the middle
        if (side === 'two') {
          const left = tail.filter(([z]) => z < 0);
          const right = tail.filter(([z]) => z > 0);
          if (left.length > 1) g.append('path').datum(left).attr('d', area).attr('fill', '#F59E0B').attr('fill-opacity', 0.55);
          if (right.length > 1) g.append('path').datum(right).attr('d', area).attr('fill', '#F59E0B').attr('fill-opacity', 0.55);
        } else if (tail.length > 1) {
          g.append('path').datum(tail).attr('d', area).attr('fill', '#F59E0B').attr('fill-opacity', 0.55);
        }

        // Main curve
        g.append('path').datum(pts).attr('d', line).attr('stroke', '#059669').attr('stroke-width', 2).attr('fill', 'none');

        // Observed T
        g.append('line')
          .attr('x1', x(t))
          .attr('x2', x(t))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', '#DC2626')
          .attr('stroke-width', 2);
        g.append('text')
          .attr('x', x(t))
          .attr('y', 12)
          .attr('text-anchor', 'middle')
          .attr('fill', '#DC2626')
          .style('font-size', '10px')
          .style('font-weight', '600')
          .text(`T = ${t.toFixed(2)}`);

        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(7));
        g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')));
      }
    } else if (mode !== 'single') {
      // Histogram of p-values on [0, 1]
      const bins = 20;
      const counts = new Array(bins).fill(0);
      mcPValues.forEach((p) => {
        const b = Math.min(bins - 1, Math.floor(p * bins));
        counts[b]++;
      });
      const maxC = Math.max(...counts);
      const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0, maxC * 1.15]).range([innerH, 0]);

      // Uniform reference line
      const expected = mcPValues.length / bins;
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', y(expected))
        .attr('y2', y(expected))
        .attr('stroke', '#1F2937')
        .attr('stroke-opacity', 0.5)
        .attr('stroke-dasharray', '4,4');
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', y(expected) - 4)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .attr('fill', 'currentColor')
        .attr('fill-opacity', 0.6)
        .text('Uniform(0, 1)');

      counts.forEach((c, i) => {
        g.append('rect')
          .attr('x', x(i / bins))
          .attr('y', y(c))
          .attr('width', innerW / bins - 1)
          .attr('height', innerH - y(c))
          .attr('fill', mode === 'mc-h0' ? '#059669' : '#DC2626')
          .attr('fill-opacity', 0.65);
      });
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
    }

    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text(mode === 'single' ? (scenario.family === 'binomial-exact' ? 'sum X_i' : 'test statistic T') : 'p-value');
  }, [svgW, mode, singleDraw, mcPValues, scenario, n, side]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          P-values · §17.5
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {pValueScenarios.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 flex items-center gap-2 text-xs">
        <span style={{ color: 'var(--color-text-muted)' }}>mode</span>
        {(['single', 'mc-h0', 'mc-ha'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className="px-3 py-1 rounded text-xs font-semibold"
            style={{
              background: mode === m ? 'var(--color-accent)' : 'var(--color-input-bg)',
              color: mode === m ? 'white' : 'var(--color-text)',
              border: '1px solid var(--color-border)',
            }}
          >
            {m === 'single' ? 'single sample' : m === 'mc-h0' ? 'MC under H₀' : 'MC under H_A'}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
          </span>
          <input
            type="range"
            min={5}
            max={500}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            disabled={scenario.family === 'binomial-exact'}
          />
        </label>
        {mode !== 'mc-h0' && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              {scenario.family === 'two-proportion' ? 'lift' : scenario.family === 'binomial-exact' ? 'true p' : 'true μ'} ={' '}
              <strong>{trueTheta.toFixed(3)}</strong>
            </span>
            <input
              type="range"
              min={scenario.family === 'binomial-exact' ? 0 : scenario.family === 'two-proportion' ? -0.1 : -1}
              max={scenario.family === 'binomial-exact' ? 1 : scenario.family === 'two-proportion' ? 0.1 : 2}
              step={scenario.family === 'binomial-exact' ? 0.01 : scenario.family === 'two-proportion' ? 0.005 : 0.05}
              value={trueTheta}
              onChange={(e) => setTrueTheta(Number(e.target.value))}
            />
          </label>
        )}
        {mode !== 'single' && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              M = <strong>{M}</strong>
            </span>
            <input type="range" min={200} max={10000} step={100} value={M} onChange={(e) => setM(Number(e.target.value))} />
          </label>
        )}
        <div className="flex flex-col gap-1 text-xs">
          <span style={{ color: 'var(--color-text-muted)' }}>side</span>
          <div className="flex gap-2">
            {scenario.sides.map((s) => (
              <label key={s} className="flex items-center gap-1">
                <input type="radio" name="pv-side" checked={side === s} onChange={() => setSide(s)} />
                {s === 'right' ? 'right' : s === 'left' ? 'left' : 'two'}
              </label>
            ))}
          </div>
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="self-end px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          {mode === 'single' ? 'Draw again' : 'Re-run MC'}
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-[60%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <svg ref={svgRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-3">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            {mode === 'single' ? 'Observed draw' : 'Monte Carlo summary'}
          </div>
          {mode === 'single' && singleDraw && (
            <div className="space-y-2 text-xs">
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>observed T =</span>{' '}
                <strong className="font-mono">{singleDraw.t.toFixed(3)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>p-value =</span>{' '}
                <strong className="font-mono" style={{ color: singleDraw.p <= 0.05 ? '#DC2626' : '#059669' }}>
                  {singleDraw.p.toFixed(4)}
                </strong>
              </div>
              <div className="text-[11px]" style={{ color: singleDraw.p <= 0.05 ? '#B91C1C' : '#047857' }}>
                {singleDraw.p <= 0.001
                  ? '⇒ reject at α = 0.001'
                  : singleDraw.p <= 0.01
                  ? '⇒ reject at α = 0.01 (not at 0.001)'
                  : singleDraw.p <= 0.05
                  ? '⇒ reject at α = 0.05 (not at 0.01)'
                  : '⇒ fail to reject at α = 0.05'}
              </div>
            </div>
          )}
          {mode !== 'single' && (
            <div className="space-y-2 text-xs">
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>M = </span>
                <strong>{mcPValues.length}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--color-text-muted)' }}>empirical rejection rate at α = 0.05:</span>
              </div>
              <div className="text-lg font-mono" style={{ color: mode === 'mc-h0' ? '#059669' : '#DC2626' }}>
                {rejectionRate.toFixed(3)}
              </div>
              <div className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>
                {mode === 'mc-h0'
                  ? 'Under H₀ the rate should equal α = 0.05 (continuous case; Thm 3). For discrete nulls it is ≤ α (conservative).'
                  : 'Under H_A this is power — the probability of correctly rejecting. Larger than α by the effect size.'}
              </div>
            </div>
          )}
          <div className="text-[11px] italic leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
            {scenario.description}
          </div>
        </div>
      </div>
    </div>
  );
}
