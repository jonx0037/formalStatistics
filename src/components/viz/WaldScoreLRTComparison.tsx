import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  waldStatistic,
  scoreStatistic,
  lrtStatistic,
  chiSquaredPDF,
  chiSquaredInvCDF,
  type AsymptoticTestFamily,
} from './shared/testing';
import { normalSample, bernoulliSample, poissonSample } from './shared/convergence';
import { seededRandom } from './shared/probability';
import {
  waldScoreLRTPresets,
  type WaldScoreLRTPreset,
} from '../../data/hypothesis-testing-data';

const MARGIN = { top: 14, right: 16, bottom: 40, left: 48 };
const H = 300;
const BINS = 35;

const COLOR_WALD = '#D97706';
const COLOR_SCORE = '#7C3AED';
const COLOR_LRT = '#059669';
const COLOR_CHI2 = '#6B7280';

function sampleFamily(
  family: AsymptoticTestFamily,
  n: number,
  theta: number,
  rng: () => number,
  knownParam?: number,
): number[] {
  switch (family) {
    case 'bernoulli':
      return Array.from({ length: n }, () => bernoulliSample(theta, rng));
    case 'normal-mean-known-sigma':
      return Array.from({ length: n }, () => normalSample(theta, knownParam ?? 1, rng));
    case 'normal-mean':
      return Array.from({ length: n }, () => normalSample(theta, 1, rng));
    case 'poisson':
      return Array.from({ length: n }, () => poissonSample(theta, rng));
  }
}

export default function WaldScoreLRTComparison() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);
  const isWide = w > 900;

  const [presetIndex, setPresetIndex] = useState(0);
  const preset: WaldScoreLRTPreset = waldScoreLRTPresets[presetIndex];

  const [mode, setMode] = useState<'H0' | 'HA'>('H0');
  const [theta0, setTheta0] = useState(preset.nullTheta);
  const [trueTheta, setTrueTheta] = useState(preset.defaultAltTheta);
  const [n, setN] = useState(preset.nDefault);
  const [M, setM] = useState(preset.MDefault);
  const [seed, setSeed] = useState(3);

  useEffect(() => {
    setTheta0(preset.nullTheta);
    setTrueTheta(preset.defaultAltTheta);
    setN(preset.nDefault);
    setM(preset.MDefault);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIndex]);

  const effectiveTrueTheta = mode === 'H0' ? theta0 : trueTheta;

  const mc = useMemo(() => {
    const rng = seededRandom(seed * 7919 + n + M + (mode === 'H0' ? 1 : 2));
    const wald: number[] = [];
    const score: number[] = [];
    const lrt: number[] = [];
    const known = preset.family === 'normal-mean-known-sigma' ? (preset.sigmaKnown ?? 1) : undefined;
    for (let i = 0; i < M; i++) {
      const data = sampleFamily(preset.family, n, effectiveTrueTheta, rng, known);
      wald.push(waldStatistic(preset.family, data, theta0, known));
      score.push(scoreStatistic(preset.family, data, theta0, known));
      lrt.push(lrtStatistic(preset.family, data, theta0, known));
    }
    return { wald, score, lrt };
  }, [preset, n, M, theta0, effectiveTrueTheta, seed, mode]);

  const crit = useMemo(() => chiSquaredInvCDF(0.95, 1), []);
  const rejectionRates = useMemo(() => {
    const fracAbove = (arr: number[]) => arr.reduce((s, v) => s + (v > crit ? 1 : 0), 0) / arr.length;
    return {
      wald: fracAbove(mc.wald),
      score: fracAbove(mc.score),
      lrt: fracAbove(mc.lrt),
    };
  }, [mc, crit]);

  const maxPairwiseDiff = Math.max(
    Math.abs(rejectionRates.wald - rejectionRates.score),
    Math.abs(rejectionRates.score - rejectionRates.lrt),
    Math.abs(rejectionRates.wald - rejectionRates.lrt),
  );

  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgW = isWide ? Math.floor(w * 0.58) : w;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = svgW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;

    // Clip display range to avoid extreme outliers dominating
    const allValues = [...mc.wald, ...mc.score, ...mc.lrt].filter((v) => Number.isFinite(v));
    if (allValues.length === 0) return;
    const q99 = d3.quantile(allValues.slice().sort(d3.ascending), 0.99) ?? 10;
    const xMax = Math.max(q99 * 1.1, 6);
    const x = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);

    const step = xMax / BINS;
    const binCounts = (arr: number[]) => {
      const out = new Array(BINS).fill(0);
      arr.forEach((v) => {
        if (!Number.isFinite(v) || v < 0 || v >= xMax) return;
        const b = Math.min(BINS - 1, Math.floor(v / step));
        out[b]++;
      });
      return out.map((c) => c / (arr.length * step));
    };

    const waldDensity = binCounts(mc.wald);
    const scoreDensity = binCounts(mc.score);
    const lrtDensity = binCounts(mc.lrt);

    // χ²_1 reference
    const samples = 161;
    const chi2Pts: [number, number][] = Array.from({ length: samples }, (_, i) => {
      const z = (xMax * i) / (samples - 1) + 1e-4;
      return [z, chiSquaredPDF(z, 1)];
    });

    const maxY = Math.max(
      ...waldDensity,
      ...scoreDensity,
      ...lrtDensity,
      ...chi2Pts.slice(1).map(([, p]) => p),
    ) * 1.1;
    const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);

    const g = svg
      .attr('viewBox', `0 0 ${svgW} ${H}`)
      .attr('width', svgW)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(7));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')));

    const drawHist = (density: number[], color: string, offsetRatio: number) => {
      const barW = Math.max(0, innerW / BINS - 1);
      const third = barW / 3;
      density.forEach((v, i) => {
        g.append('rect')
          .attr('x', x(i * step) + offsetRatio * third)
          .attr('y', y(v))
          .attr('width', third)
          .attr('height', innerH - y(v))
          .attr('fill', color)
          .attr('fill-opacity', 0.7);
      });
    };
    drawHist(waldDensity, COLOR_WALD, 0);
    drawHist(scoreDensity, COLOR_SCORE, 1);
    drawHist(lrtDensity, COLOR_LRT, 2);

    // χ²_1 density curve
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.3));
    g.append('path')
      .datum(chi2Pts.filter(([, p]) => Number.isFinite(p) && p < maxY * 1.5))
      .attr('d', line)
      .attr('stroke', COLOR_CHI2)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,3')
      .attr('fill', 'none');

    // Critical value line
    g.append('line')
      .attr('x1', x(crit))
      .attr('x2', x(crit))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#DC2626')
      .attr('stroke-dasharray', '3,3');
    g.append('text')
      .attr('x', x(crit))
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .attr('fill', '#DC2626')
      .attr('fill-opacity', 0.85)
      .text(`χ²₁,.₉₅ = ${crit.toFixed(2)}`);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${innerW - 130}, 4)`);
    const items: [string, string][] = [
      ['Wald', COLOR_WALD],
      ['Score', COLOR_SCORE],
      ['LRT', COLOR_LRT],
    ];
    items.forEach(([label, color], i) => {
      legend.append('rect').attr('x', 0).attr('y', i * 14).attr('width', 10).attr('height', 10).attr('fill', color).attr('fill-opacity', 0.7);
      legend.append('text').attr('x', 14).attr('y', i * 14 + 9).style('font-size', '10px').attr('fill', 'currentColor').text(label);
    });
    legend.append('line').attr('x1', 0).attr('x2', 10).attr('y1', 3 * 14 + 5).attr('y2', 3 * 14 + 5).attr('stroke', COLOR_CHI2).attr('stroke-width', 2).attr('stroke-dasharray', '4,3');
    legend.append('text').attr('x', 14).attr('y', 3 * 14 + 8).style('font-size', '10px').attr('fill', 'currentColor').text('χ²₁');

    svg
      .append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', H - 6)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('fill', 'currentColor')
      .attr('fill-opacity', 0.7)
      .text('test statistic');
  }, [mc, svgW, crit]);

  return (
    <div
      ref={containerRef}
      className="not-prose my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', background: 'var(--color-card-bg)' }}
    >
      <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wide font-bold" style={{ color: 'var(--color-text-muted)' }}>
          Wald / Score / LRT · §17.9
        </div>
        <select
          value={presetIndex}
          onChange={(e) => setPresetIndex(Number(e.target.value))}
          className="text-sm rounded border px-2 py-1"
          style={{ background: 'var(--color-input-bg)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
        >
          {waldScoreLRTPresets.map((p, i) => (
            <option key={p.id} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-3 flex items-center gap-2 text-xs">
        <span style={{ color: 'var(--color-text-muted)' }}>mode</span>
        {(['H0', 'HA'] as const).map((m) => (
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
            {m === 'H0' ? 'H₀ (true = null)' : 'H_A (true ≠ null)'}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            θ₀ = <strong>{theta0.toFixed(preset.family === 'bernoulli' ? 2 : 1)}</strong>
          </span>
          <input
            type="range"
            min={preset.altRange[0]}
            max={preset.altRange[1]}
            step={preset.family === 'bernoulli' ? 0.01 : 0.05}
            value={theta0}
            onChange={(e) => setTheta0(Number(e.target.value))}
          />
        </label>
        {mode === 'HA' && (
          <label className="flex flex-col gap-1">
            <span style={{ color: 'var(--color-text-muted)' }}>
              true θ = <strong>{trueTheta.toFixed(preset.family === 'bernoulli' ? 2 : 1)}</strong>
            </span>
            <input
              type="range"
              min={preset.altRange[0]}
              max={preset.altRange[1]}
              step={preset.family === 'bernoulli' ? 0.01 : 0.05}
              value={trueTheta}
              onChange={(e) => setTrueTheta(Number(e.target.value))}
            />
          </label>
        )}
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            n = <strong>{n}</strong>
          </span>
          <input type="range" min={10} max={1000} value={n} onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span style={{ color: 'var(--color-text-muted)' }}>
            M = <strong>{M}</strong>
          </span>
          <input type="range" min={500} max={10000} step={500} value={M} onChange={(e) => setM(Number(e.target.value))} />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="self-end px-3 py-1.5 rounded text-sm font-semibold"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          Re-run MC
        </button>
      </div>

      <div className={isWide ? 'grid grid-cols-[58%_minmax(0,1fr)] gap-4' : 'flex flex-col gap-4'}>
        <svg ref={svgRef} style={{ display: 'block' }} />
        <div className="text-sm space-y-3">
          <div className="text-xs uppercase font-bold tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Formulas — {preset.name}
          </div>
          <div className="text-[11px] space-y-1 p-2 rounded font-mono" style={{ background: 'var(--color-input-bg)' }}>
            <div><span style={{ color: COLOR_WALD }}>■</span> Wald: {preset.formulas.wald}</div>
            <div><span style={{ color: COLOR_SCORE }}>■</span> Score: {preset.formulas.score}</div>
            <div><span style={{ color: COLOR_LRT }}>■</span> LRT: {preset.formulas.lrt}</div>
          </div>
          <div className="text-xs uppercase font-bold tracking-wide pt-1" style={{ color: 'var(--color-text-muted)' }}>
            Rejection rate at α = 0.05
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="rounded p-2" style={{ background: 'rgba(217,119,6,0.12)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: COLOR_WALD }}>Wald</div>
              <div className="text-base font-mono">{rejectionRates.wald.toFixed(3)}</div>
            </div>
            <div className="rounded p-2" style={{ background: 'rgba(124,58,237,0.12)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: COLOR_SCORE }}>Score</div>
              <div className="text-base font-mono">{rejectionRates.score.toFixed(3)}</div>
            </div>
            <div className="rounded p-2" style={{ background: 'rgba(5,150,105,0.12)' }}>
              <div className="text-[10px] uppercase font-bold" style={{ color: COLOR_LRT }}>LRT</div>
              <div className="text-base font-mono">{rejectionRates.lrt.toFixed(3)}</div>
            </div>
          </div>
          <div className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>
            max pairwise diff = <strong>{maxPairwiseDiff.toFixed(3)}</strong>
          </div>
          <div className="text-[11px] leading-relaxed" style={{ color: 'var(--color-text)' }}>
            {mode === 'H0'
              ? 'Under H₀ all three should approximate α = 0.05. Differences shrink as n grows — Wilks, Rao, Wald all agree asymptotically (Topic 18 proves the LRT case in full).'
              : 'Under H_A the three rejection rates are power — all three should exceed α, with finite-sample differences visible. Topic 18 dissects when each dominates.'}
          </div>
          <div className="text-[11px] italic leading-relaxed pt-2" style={{ color: 'var(--color-text-muted)' }}>
            {preset.description}
          </div>
        </div>
      </div>
    </div>
  );
}
