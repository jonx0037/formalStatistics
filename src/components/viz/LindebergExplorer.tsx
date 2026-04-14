import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  normalSample,
  exponentialSample,
  uniformSample,
  bernoulliSample,
} from './shared/convergence';
import { pdfStdNormal, cdfStdNormal } from './shared/distributions';
import {
  lindebergPresets,
  type LindebergPreset,
} from '../../data/central-limit-theorem-data';

// ── Constants ───────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 18, bottom: 38, left: 48 };
const CHART_H = 260;
const HIST_BINS = 36;
const M_REPS = 3000;

const HIST_COLOR = '#7C3AED';
const NORMAL_COLOR = '#059669';
const FAIL_COLOR = '#DC2626';
const VAR_BAR_COLOR = '#2563EB';

// ── Summand Construction ────────────────────────────────────────────────────

interface Summand {
  sampler: () => number;
  mean: number;
  variance: number;
  label: string;
}

function buildSummands(preset: LindebergPreset, n: number): Summand[] {
  if (preset.id === 'mixed-equal-var') {
    const base: Summand[] = [
      { sampler: () => normalSample(0, 1), mean: 0, variance: 1, label: 'N(0,1)' },
      {
        sampler: () => exponentialSample(1) - 1,
        mean: 0,
        variance: 1,
        label: 'Exp(1)−1',
      },
      {
        sampler: () => uniformSample(-Math.sqrt(3), Math.sqrt(3)),
        mean: 0,
        variance: 1,
        label: 'U(−√3, √3)',
      },
    ];
    // Cycle through the 3 base types up to n
    const out: Summand[] = [];
    for (let k = 0; k < n; k++) out.push(base[k % 3]);
    return out;
  }
  if (preset.id === 'increasing-var') {
    const out: Summand[] = [];
    for (let k = 1; k <= n; k++) {
      const variance = k;
      const sigma = Math.sqrt(variance);
      out.push({
        sampler: () => normalSample(0, sigma),
        mean: 0,
        variance,
        label: `N(0, ${variance})`,
      });
    }
    return out;
  }
  if (preset.id === 'one-dominant') {
    const out: Summand[] = [];
    const bigVar = n * n;
    out.push({
      sampler: () => normalSample(0, Math.sqrt(bigVar)),
      mean: 0,
      variance: bigVar,
      label: `N(0, n²)`,
    });
    for (let k = 2; k <= n; k++) {
      out.push({
        sampler: () => normalSample(0, 1),
        mean: 0,
        variance: 1,
        label: 'N(0, 1)',
      });
    }
    return out;
  }
  if (preset.id === 'bernoulli-varying-p') {
    const out: Summand[] = [];
    for (let k = 1; k <= n; k++) {
      const p = 1 / (k + 1);
      const mean = p;
      const variance = p * (1 - p);
      out.push({
        sampler: () => bernoulliSample(p) - p,
        mean: 0,
        variance,
        label: `Bern(${p.toFixed(2)})−p`,
      });
    }
    return out;
  }
  return [];
}

// Lindeberg fraction L(ε) — use numerical integration / Monte Carlo.
function lindebergFraction(summands: Summand[], epsilon: number, sN: number): number {
  if (sN === 0) return 0;
  const threshold = epsilon * sN;
  let total = 0;
  // Use 1500 samples per summand to estimate E[X² · 1(|X| > ε sN)]
  // (compromise between precision and interactive responsiveness).
  const samplesPer = 1500;
  for (const s of summands) {
    let sum = 0;
    for (let i = 0; i < samplesPer; i++) {
      const x = s.sampler();
      if (Math.abs(x) > threshold) sum += x * x;
    }
    total += sum / samplesPer;
  }
  return total / (sN * sN);
}

// ── Component ───────────────────────────────────────────────────────────────

function LindebergExplorer() {
  const [presetId, setPresetId] = useState<string>('mixed-equal-var');
  const [n, setN] = useState<number>(30);

  const preset = useMemo<LindebergPreset>(
    () => lindebergPresets.find((p) => p.id === presetId) ?? lindebergPresets[0],
    [presetId],
  );

  const summands = useMemo(() => buildSummands(preset, n), [preset, n]);

  const sN2 = useMemo(
    () => summands.reduce((a, s) => a + s.variance, 0),
    [summands],
  );
  const sN = Math.sqrt(sN2);

  const maxVarShare = useMemo(
    () => (sN2 > 0 ? Math.max(...summands.map((s) => s.variance / sN2)) : 0),
    [summands, sN2],
  );

  // Replications of S_n / s_n.
  const standardizedSums = useMemo(() => {
    const out: number[] = [];
    for (let r = 0; r < M_REPS; r++) {
      let s = 0;
      for (const sm of summands) s += sm.sampler();
      out.push(s / sN);
    }
    return out;
  }, [summands, sN]);

  // Lindeberg fraction for a grid of epsilons.
  const lindebergCurve = useMemo(() => {
    const epsilons = [0.05, 0.1, 0.25, 0.5, 1, 2];
    return epsilons.map((eps) => ({
      eps,
      L: lindebergFraction(summands, eps, sN),
    }));
  }, [summands, sN]);

  const ksStat = useMemo(() => {
    const sorted = [...standardizedSums].sort((a, b) => a - b);
    const n = sorted.length;
    let d = 0;
    for (let i = 0; i < n; i++) {
      const Fx = cdfStdNormal(sorted[i]);
      const ecdfR = (i + 1) / n;
      const ecdfL = i / n;
      const diff = Math.max(Math.abs(ecdfR - Fx), Math.abs(ecdfL - Fx));
      if (diff > d) d = diff;
    }
    return d;
  }, [standardizedSums]);

  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(260, width);
  const threeCol = width >= 900;
  const panelW = threeCol ? (w - 24) / 3 : w;

  return (
    <div
      className="my-8 rounded-lg border p-5"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            Scenario
          </span>
          <select
            className="rounded border p-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            value={presetId}
            onChange={(e) => setPresetId(e.target.value)}
          >
            {lindebergPresets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-medium uppercase tracking-wide opacity-70">
            n summands = {n}
          </span>
          <input
            type="range"
            min={5}
            max={120}
            step={5}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
          />
        </label>
        <div className="rounded border p-2 text-xs" style={{ borderColor: 'var(--color-border)' }}>
          <div className="font-semibold" style={{ color: preset.lindebergHolds ? NORMAL_COLOR : FAIL_COLOR }}>
            Lindeberg: {preset.lindebergHolds ? 'holds' : 'fails'}
          </div>
          <div className="opacity-70">{preset.description}</div>
        </div>
      </div>

      <div
        ref={containerRef}
        className={threeCol ? 'mt-4 grid grid-cols-3 gap-3' : 'mt-4 flex flex-col gap-4'}
      >
        <HistPanel width={panelW} sums={standardizedSums} />
        <LindebergCurvePanel width={panelW} curve={lindebergCurve} />
        <VarShareBars width={panelW} summands={summands} sN2={sN2} />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        <Stat label="s_n²" value={sN2.toFixed(2)} />
        <Stat label="max Var(Xₖ)/s_n²" value={maxVarShare.toFixed(3)} color={maxVarShare > 0.5 ? FAIL_COLOR : NORMAL_COLOR} />
        <Stat label="KS vs N(0,1)" value={ksStat.toFixed(4)} />
        <Stat
          label="status"
          value={preset.lindebergHolds ? 'Lindeberg OK' : 'Lindeberg fails'}
          color={preset.lindebergHolds ? NORMAL_COLOR : FAIL_COLOR}
        />
      </div>
    </div>
  );
}

// ── Panels ─────────────────────────────────────────────────────────────────

function HistPanel({ width, sums }: { width: number; sums: number[] }) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
    const bins = d3.bin<number, number>().domain([-4, 4]).thresholds(HIST_BINS)(
      sums.filter((v) => v >= -4 && v <= 4),
    );
    const density = (b: d3.Bin<number, number>) =>
      b.length / (sums.length * (b.x1! - b.x0!));
    const yMax = Math.max(
      d3.max(bins, density) ?? 0.4,
      pdfStdNormal(0) * 1.05,
    );
    const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .attr('fill', HIST_COLOR)
      .attr('fill-opacity', 0.55)
      .selectAll('rect')
      .data(bins)
      .join('rect')
      .attr('x', (b) => xScale(b.x0!))
      .attr('y', (b) => yScale(density(b)))
      .attr('width', (b) => Math.max(0, xScale(b.x1!) - xScale(b.x0!) - 1))
      .attr('height', (b) => innerH - yScale(density(b)));

    const xs = d3.range(-4, 4.01, 0.05);
    const line = d3
      .line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(pdfStdNormal(x)));
    g.append('path')
      .datum(xs)
      .attr('fill', 'none')
      .attr('stroke', NORMAL_COLOR)
      .attr('stroke-width', 2)
      .attr('d', line);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Sₙ / sₙ');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Standardized sum vs N(0,1)');
  }, [width, sums]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function LindebergCurvePanel({
  width,
  curve,
}: {
  width: number;
  curve: Array<{ eps: number; L: number }>;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const xScale = d3
      .scaleLog()
      .domain([curve[0].eps, curve[curve.length - 1].eps])
      .range([0, innerW]);
    const yMax = Math.max(d3.max(curve, (d) => d.L) ?? 0.1, 0.1);
    const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]).nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const line = d3
      .line<{ eps: number; L: number }>()
      .x((d) => xScale(d.eps))
      .y((d) => yScale(d.L));
    g.append('path')
      .datum(curve)
      .attr('fill', 'none')
      .attr('stroke', HIST_COLOR)
      .attr('stroke-width', 2)
      .attr('d', line);
    g.append('g')
      .selectAll('circle')
      .data(curve)
      .join('circle')
      .attr('cx', (d) => xScale(d.eps))
      .attr('cy', (d) => yScale(d.L))
      .attr('r', 3)
      .attr('fill', HIST_COLOR);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).tickValues(curve.map((c) => c.eps)).tickFormat(d3.format('.2g')))
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('font-size', 10);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('ε (log)');
    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('L_n(ε)');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Lindeberg fraction L_n(ε)');
  }, [width, curve]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function VarShareBars({
  width,
  summands,
  sN2,
}: {
  width: number;
  summands: Summand[];
  sN2: number;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current || sN2 === 0) return;
    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = CHART_H - MARGIN.top - MARGIN.bottom;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Sort by variance descending
    const sorted = summands
      .map((s, i) => ({ ...s, idx: i + 1, share: s.variance / sN2 }))
      .sort((a, b) => b.share - a.share);

    const xScale = d3
      .scaleBand<number>()
      .domain(sorted.map((_, i) => i))
      .range([0, innerW])
      .padding(0.1);
    const yScale = d3
      .scaleLinear()
      .domain([0, Math.max(...sorted.map((s) => s.share), 0.01) * 1.1])
      .range([innerH, 0])
      .nice();

    const g = svg
      .attr('viewBox', `0 0 ${width} ${CHART_H}`)
      .attr('width', width)
      .attr('height', CHART_H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g')
      .selectAll('rect')
      .data(sorted)
      .join('rect')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', (s) => yScale(s.share))
      .attr('width', xScale.bandwidth())
      .attr('height', (s) => innerH - yScale(s.share))
      .attr('fill', (s) => (s.share > 0.3 ? FAIL_COLOR : VAR_BAR_COLOR))
      .attr('fill-opacity', 0.75);

    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickValues(
            xScale
              .domain()
              .filter((_, i) => i === 0 || i === sorted.length - 1 || i % Math.ceil(sorted.length / 6) === 0),
          )
          .tickFormat((i) => `#${(i as number) + 1}`),
      )
      .attr('font-size', 10);
    g.append('g').call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.0%'))).attr('font-size', 10);

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('summand (rank by variance)');
    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', 10)
      .attr('opacity', 0.8)
      .text('Var(Xₖ) / s_n²');

    svg
      .append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top + 10)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Variance share per summand');
  }, [width, summands, sN2]);

  return <svg ref={ref} style={{ display: 'block', width }} />;
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div
      className="rounded border p-2"
      style={{ borderColor: color ?? 'var(--color-border)' }}
    >
      <div className="text-[10px] uppercase tracking-wide opacity-60">{label}</div>
      <div className="font-mono text-sm" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export default LindebergExplorer;
