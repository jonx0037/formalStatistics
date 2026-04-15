import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  scoreFunction,
  fisherInformation,
  cramerRaoBound,
  logLikelihoodCurve,
} from './shared/estimation';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  poissonSample,
  sampleSequence,
} from './shared/convergence';
import { fisherInfoPresets, type FisherInfoPreset } from '../../data/point-estimation-data';

type Family = FisherInfoPreset['family'];

const MARGIN = { top: 14, right: 18, bottom: 36, left: 48 };
const H = 220;

interface FamilySpec {
  family: Family;
  paramName: 'mu' | 'p' | 'lambda';
  paramLabel: string;
  paramRange: [number, number];
  paramDefault: number;
  paramStep: number;
  /** Sampler returning n iid obs given the scalar parameter. */
  sampler: (theta: number, n: number) => number[];
  /** Log-density for a single observation. */
  logPdf: (x: number, theta: number) => number;
  /** Extra fixed params (e.g. Normal's σ²) to pass into score/Fisher closures. */
  otherParams: Record<string, number>;
  /** X-range on the score panel (sample values, not θ). */
  xRange: [number, number];
}

const NORMAL_SIGMA2 = 1; // Fix σ² = 1 for the Normal-w.r.t.-μ example

const familySpecs: Record<Family, FamilySpec> = {
  Normal: {
    family: 'Normal',
    paramName: 'mu',
    paramLabel: 'μ',
    paramRange: [-3, 3],
    paramDefault: 0,
    paramStep: 0.01,
    sampler: (mu, n) => sampleSequence(() => normalSample(mu, Math.sqrt(NORMAL_SIGMA2)), n),
    logPdf: (x, mu) =>
      -0.5 * Math.log(2 * Math.PI * NORMAL_SIGMA2) - (x - mu) ** 2 / (2 * NORMAL_SIGMA2),
    otherParams: { sigma2: NORMAL_SIGMA2 },
    xRange: [-3, 3],
  },
  Bernoulli: {
    family: 'Bernoulli',
    paramName: 'p',
    paramLabel: 'p',
    paramRange: [0.02, 0.98],
    paramDefault: 0.5,
    paramStep: 0.01,
    sampler: (p, n) => sampleSequence(() => bernoulliSample(p), n),
    logPdf: (x, p) => {
      const xi = x > 0.5 ? 1 : 0;
      return xi * Math.log(p) + (1 - xi) * Math.log(1 - p);
    },
    otherParams: {},
    xRange: [0, 1],
  },
  Exponential: {
    family: 'Exponential',
    paramName: 'lambda',
    paramLabel: 'λ',
    paramRange: [0.2, 4],
    paramDefault: 1.0,
    paramStep: 0.01,
    sampler: (lambda, n) => sampleSequence(() => exponentialSample(lambda), n),
    logPdf: (x, lambda) => Math.log(lambda) - lambda * x,
    otherParams: {},
    xRange: [0, 6],
  },
  Poisson: {
    family: 'Poisson',
    paramName: 'lambda',
    paramLabel: 'λ',
    paramRange: [0.5, 10],
    paramDefault: 3,
    paramStep: 0.1,
    sampler: (lambda, n) => sampleSequence(() => poissonSample(lambda), n),
    logPdf: (x, lambda) => {
      // x is an integer (or near it); use log-gamma for the factorial
      const k = Math.round(x);
      let logFact = 0;
      for (let i = 2; i <= k; i++) logFact += Math.log(i);
      return k * Math.log(lambda) - lambda - logFact;
    },
    otherParams: {},
    xRange: [0, 15],
  },
};

export default function FisherInformationExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [family, setFamily] = useState<Family>('Normal');
  const spec = familySpecs[family];

  const [theta, setTheta] = useState(spec.paramDefault);
  const [n, setN] = useState(50);
  const [seed, setSeed] = useState(0);

  useEffect(() => {
    setTheta(spec.paramDefault);
  }, [family, spec.paramDefault]);

  const scoreFn = useMemo(() => scoreFunction(family, spec.paramName), [family, spec.paramName]);
  const fiFn = useMemo(() => fisherInformation(family, spec.paramName), [family, spec.paramName]);
  const iTheta = fiFn(theta, spec.otherParams);
  const crlb = cramerRaoBound(n, iTheta);

  const sample = useMemo(
    () => spec.sampler(theta, n),
    // include seed so the user can resample
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [family, n, theta, seed],
  );

  // Log-likelihood grid
  const llGrid = useMemo(() => {
    const [a, b] = spec.paramRange;
    const grid = d3.range(100).map((i) => a + ((b - a) * i) / 99);
    return logLikelihoodCurve(sample, (x, th) => spec.logPdf(x, th), grid);
  }, [sample, spec]);

  // Score-function panel refs
  const scoreRef = useRef<SVGSVGElement | null>(null);
  const llRef = useRef<SVGSVGElement | null>(null);
  const fiRef = useRef<SVGSVGElement | null>(null);

  // Score function panel
  useEffect(() => {
    if (!scoreRef.current) return;
    const svg = d3.select(scoreRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 1050 ? (w - 32) / 3 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Plot s(θ; xᵢ) as a function of θ, for 5 sample x-values
    const xSamples = d3.range(5).map(
      (i) => spec.xRange[0] + ((spec.xRange[1] - spec.xRange[0]) * (i + 1)) / 6,
    );
    const [paramA, paramB] = spec.paramRange;
    const xScale = d3.scaleLinear().domain([paramA, paramB]).range([0, innerW]);

    // Compute scores across the θ grid for each sample x
    const thetaGrid = d3.range(80).map((i) => paramA + ((paramB - paramA) * i) / 79);
    const allCurves = xSamples.map((xi) =>
      thetaGrid.map((th) => ({ th, s: scoreFn(xi, th, spec.otherParams) })),
    );
    const allVals = allCurves.flat().map((d) => d.s);
    const yMin = d3.min(allVals) ?? -1;
    const yMax = d3.max(allVals) ?? 1;
    const yAbs = Math.max(Math.abs(yMin), Math.abs(yMax), 1);
    const yScale = d3.scaleLinear().domain([-yAbs, yAbs]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`Score s(${spec.paramLabel}; x)`);

    // Zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#9ca3af')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3');

    const palette = d3.schemeBlues[7].slice(2);
    allCurves.forEach((curve, i) => {
      const gen = d3
        .line<{ th: number; s: number }>()
        .x((d) => xScale(d.th))
        .y((d) => yScale(d.s));
      g.append('path')
        .datum(curve)
        .attr('fill', 'none')
        .attr('stroke', palette[i % palette.length])
        .attr('stroke-width', 1.5)
        .attr('d', gen);
    });

    // θ marker
    g.append('line')
      .attr('x1', xScale(theta))
      .attr('x2', xScale(theta))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 1.5);
  }, [scoreFn, spec, theta, w]);

  // Log-likelihood panel
  useEffect(() => {
    if (!llRef.current) return;
    const svg = d3.select(llRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 1050 ? (w - 32) / 3 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const [paramA, paramB] = spec.paramRange;
    const xScale = d3.scaleLinear().domain([paramA, paramB]).range([0, innerW]);
    const ys = llGrid.logLiks;
    const ysFinite = ys.filter((v) => isFinite(v));
    if (ysFinite.length === 0) return;
    const yMin = d3.min(ysFinite) ?? 0;
    const yMax = d3.max(ysFinite) ?? 1;
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`Log-likelihood ℓ(${spec.paramLabel})`);

    const gen = d3
      .line<number>()
      .defined((d) => isFinite(d))
      .x((_, i) => xScale(llGrid.thetas[i]))
      .y((d) => yScale(d));
    g.append('path').datum(ys).attr('fill', 'none').attr('stroke', '#111827').attr('stroke-width', 2).attr('d', gen);

    // Quadratic approximation centered at the MLE (argmax on the grid)
    let mleIdx = 0;
    for (let i = 1; i < ys.length; i++) if (ys[i] > ys[mleIdx]) mleIdx = i;
    const thetaHat = llGrid.thetas[mleIdx];
    const thetaHatLL = ys[mleIdx];
    const iHat = fiFn(thetaHat, spec.otherParams);
    const quad = llGrid.thetas.map((th) => thetaHatLL - 0.5 * n * iHat * (th - thetaHat) ** 2);
    const qGen = d3
      .line<number>()
      .x((_, i) => xScale(llGrid.thetas[i]))
      .y((d) => yScale(d));
    g.append('path').datum(quad).attr('fill', 'none').attr('stroke', '#dc2626').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3').attr('d', qGen);

    // Current θ marker
    g.append('line')
      .attr('x1', xScale(theta))
      .attr('x2', xScale(theta))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#6b7280')
      .attr('stroke-dasharray', '3 3');

    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 12)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#dc2626')
      .text('quadratic −½nI(θ̂)(θ−θ̂)²');
  }, [llGrid, spec, theta, fiFn, n, w]);

  // Fisher-info panel
  useEffect(() => {
    if (!fiRef.current) return;
    const svg = d3.select(fiRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 1050 ? (w - 32) / 3 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const [paramA, paramB] = spec.paramRange;
    const xScale = d3.scaleLinear().domain([paramA, paramB]).range([0, innerW]);
    const grid = d3.range(60).map((i) => paramA + ((paramB - paramA) * i) / 59);
    const vals = grid.map((th) => fiFn(th, spec.otherParams));
    const yMax = d3.max(vals.filter(isFinite)) ?? 1;
    const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`Fisher info I(${spec.paramLabel})`);

    const gen = d3
      .line<number>()
      .defined((d) => isFinite(d))
      .x((_, i) => xScale(grid[i]))
      .y((d) => yScale(d));
    g.append('path').datum(vals).attr('fill', 'none').attr('stroke', '#059669').attr('stroke-width', 2).attr('d', gen);

    // Marker at current θ
    g.append('circle')
      .attr('cx', xScale(theta))
      .attr('cy', yScale(isFinite(iTheta) ? iTheta : 0))
      .attr('r', 4)
      .attr('fill', '#059669')
      .attr('stroke', '#064e3b')
      .attr('stroke-width', 1);
  }, [fiFn, spec, theta, iTheta, w]);

  const preset = fisherInfoPresets.find((p) => p.family === family)!;

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
      <div style={{ marginBottom: '0.75rem' }}>
        <strong>Interactive: Fisher Information Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Pick a family, adjust the parameter, and see the score function, log-likelihood
          curvature, and Fisher information side by side.
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 720 ? 'repeat(3, 1fr) auto' : '1fr 1fr',
          gap: '0.5rem',
          fontSize: '0.8125rem',
          marginBottom: '0.5rem',
        }}
      >
        <label>
          Family:{' '}
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Family)}
            style={{ padding: '0.2rem' }}
          >
            {(Object.keys(familySpecs) as Family[]).map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        <label>
          {spec.paramLabel} = {theta.toFixed(3)}
          <input
            type="range"
            min={spec.paramRange[0]}
            max={spec.paramRange[1]}
            step={spec.paramStep}
            value={theta}
            onChange={(e) => setTheta(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
        <label>
          n = {n}
          <input
            type="range"
            min={10}
            max={500}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
        <button onClick={() => setSeed((s) => s + 1)} style={{ padding: '0.3rem 0.75rem' }}>
          Resample
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 1050 ? 'repeat(3, 1fr)' : '1fr',
          gap: '0.5rem',
          alignItems: 'start',
        }}
      >
        <svg ref={scoreRef} width={w > 1050 ? (w - 32) / 3 : w} height={H} style={{ maxWidth: '100%' }} />
        <svg ref={llRef} width={w > 1050 ? (w - 32) / 3 : w} height={H} style={{ maxWidth: '100%' }} />
        <svg ref={fiRef} width={w > 1050 ? (w - 32) / 3 : w} height={H} style={{ maxWidth: '100%' }} />
      </div>

      <div style={{ marginTop: '0.5rem', fontSize: '0.8125rem', lineHeight: 1.5 }}>
        <span style={{ fontWeight: 600 }}>{preset.formula}</span>.{' '}
        I({spec.paramLabel} = {theta.toFixed(3)}) ={' '}
        <strong>{isFinite(iTheta) ? iTheta.toFixed(4) : '∞'}</strong>. Cramér–Rao bound
        1/(n·I) ={' '}
        <strong>{isFinite(crlb) ? crlb.toFixed(5) : '—'}</strong>.{' '}
        <span style={{ color: 'var(--color-text-muted)' }}>{preset.description}</span>
      </div>
    </div>
  );
}
