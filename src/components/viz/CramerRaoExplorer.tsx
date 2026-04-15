import { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cramerRaoBound,
  fisherInformation,
  asymptVarMean,
  asymptVarMedian,
  riskMLE,
  riskJamesStein,
} from './shared/estimation';

type Family = 'Normal' | 'Bernoulli' | 'Exponential' | 'Poisson';
type EstimatorId = 'mean' | 'median' | 'trimmed';

interface CREFamilySpec {
  name: string;
  family: Family;
  paramName: 'mu' | 'p' | 'lambda';
  paramValue: number;
  otherParams: Record<string, number>;
  trueMean: number;
  trueVariance: number;
  fAtMedian: number; // for asymptVarMedian
  supportsJamesStein: boolean;
  note: string;
}

const FAMILY_SPECS: Record<Family, CREFamilySpec> = {
  Normal: {
    name: 'Normal(μ=0, σ²=1)',
    family: 'Normal',
    paramName: 'mu',
    paramValue: 0,
    otherParams: { sigma2: 1 },
    trueMean: 0,
    trueVariance: 1,
    fAtMedian: 1 / Math.sqrt(2 * Math.PI),
    supportsJamesStein: true,
    note: 'Mean is efficient (variance = 1/n = CRLB). Median ≈ 0.637 efficient — π/2 inflation.',
  },
  Bernoulli: {
    name: 'Bernoulli(p=0.3)',
    family: 'Bernoulli',
    paramName: 'p',
    paramValue: 0.3,
    otherParams: {},
    trueMean: 0.3,
    trueVariance: 0.3 * 0.7,
    fAtMedian: 0, // degenerate for Bernoulli — median is not smooth
    supportsJamesStein: false,
    note: 'p̂ = X̄ achieves the CRLB for Bernoulli (efficient).',
  },
  Exponential: {
    name: 'Exponential(λ=1)',
    family: 'Exponential',
    paramName: 'lambda',
    paramValue: 1,
    otherParams: {},
    trueMean: 1,
    trueVariance: 1,
    fAtMedian: Math.exp(-Math.log(2)), // λ·e^{-λ·median} with λ=1, median=log2
    supportsJamesStein: false,
    note: 'Sample mean estimates 1/λ (not λ). Median estimates log(2)/λ — different target.',
  },
  Poisson: {
    name: 'Poisson(λ=3)',
    family: 'Poisson',
    paramName: 'lambda',
    paramValue: 3,
    otherParams: {},
    trueMean: 3,
    trueVariance: 3,
    fAtMedian: 0, // discrete
    supportsJamesStein: false,
    note: 'λ̂ = X̄ achieves the CRLB for Poisson.',
  },
};

const MARGIN = { top: 14, right: 20, bottom: 36, left: 52 };
const H = 260;

export default function CramerRaoExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [family, setFamily] = useState<Family>('Normal');
  const [selected, setSelected] = useState<Set<EstimatorId>>(new Set(['mean', 'median']));
  const [n, setN] = useState(100);
  const [showJamesStein, setShowJamesStein] = useState(false);
  const [jsDimension, setJsDimension] = useState(5);

  const spec = FAMILY_SPECS[family];
  const fiFn = useMemo(
    () => fisherInformation(family, spec.paramName),
    [family, spec.paramName],
  );
  const I = fiFn(spec.paramValue, spec.otherParams);
  const crlbCurrent = cramerRaoBound(n, I);

  // Variance curves across n from 5 to 500
  const curves = useMemo(() => {
    const ns = d3.range(5, 501, 5);
    const variances: Record<string, number[]> = {};
    variances.CRLB = ns.map((nn) => cramerRaoBound(nn, I));
    if (selected.has('mean')) {
      variances.mean = ns.map((nn) => asymptVarMean(spec.trueVariance, nn));
    }
    if (selected.has('median') && spec.fAtMedian > 0) {
      variances.median = ns.map((nn) => asymptVarMedian(spec.fAtMedian, nn));
    }
    if (selected.has('trimmed')) {
      // Approximation: ~5% worse than the mean for Normal, ~10% worse otherwise
      const factor = family === 'Normal' ? 1.05 : 1.15;
      variances.trimmed = ns.map((nn) => asymptVarMean(spec.trueVariance, nn) * factor);
    }
    return { ns, variances };
  }, [selected, I, spec, family]);

  const efficiencies = useMemo(() => {
    const vals: { name: string; eff: number; color: string }[] = [];
    const bound = cramerRaoBound(n, I);
    if (selected.has('mean')) {
      vals.push({
        name: 'Mean',
        eff: bound / asymptVarMean(spec.trueVariance, n),
        color: '#2563eb',
      });
    }
    if (selected.has('median') && spec.fAtMedian > 0) {
      vals.push({
        name: 'Median',
        eff: bound / asymptVarMedian(spec.fAtMedian, n),
        color: '#dc2626',
      });
    }
    if (selected.has('trimmed')) {
      const factor = family === 'Normal' ? 1.05 : 1.15;
      vals.push({
        name: 'Trimmed',
        eff: bound / (asymptVarMean(spec.trueVariance, n) * factor),
        color: '#f59e0b',
      });
    }
    return vals;
  }, [selected, n, I, spec, family]);

  const leftRef = useRef<SVGSVGElement | null>(null);
  const rightRef = useRef<SVGSVGElement | null>(null);

  // Left: variance vs n curves with CRLB envelope
  useEffect(() => {
    if (!leftRef.current) return;
    const svg = d3.select(leftRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.6 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const ns = curves.ns;
    const x = d3.scaleLog().domain([5, 500]).range([0, innerW]);

    const allVals = Object.values(curves.variances).flat().filter(isFinite);
    const yMax = d3.max(allVals) ?? 1;
    const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([innerH, 0]).nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5, '~s'))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('n (log scale)');

    const paint = (key: string, color: string, dash?: string) => {
      const vals = curves.variances[key];
      if (!vals) return;
      const gen = d3
        .line<number>()
        .defined((d) => isFinite(d))
        .x((_, i) => x(ns[i]))
        .y((d) => y(d));
      const path = g.append('path').datum(vals).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('d', gen);
      if (dash) path.attr('stroke-dasharray', dash);
    };

    paint('CRLB', '#059669', '4 2');
    paint('mean', '#2563eb');
    paint('median', '#dc2626');
    paint('trimmed', '#f59e0b');

    // current-n vertical line
    g.append('line')
      .attr('x1', x(n))
      .attr('x2', x(n))
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#6b7280')
      .attr('stroke-dasharray', '3 3');

    // Legend
    let li = 0;
    const leg = (label: string, color: string, dash?: string) => {
      const ly = 10 + li * 14;
      g.append('line')
        .attr('x1', innerW - 110)
        .attr('x2', innerW - 90)
        .attr('y1', ly)
        .attr('y2', ly)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', dash ?? '');
      g.append('text')
        .attr('x', innerW - 86)
        .attr('y', ly + 4)
        .style('font-size', '10px')
        .style('fill', 'currentColor')
        .text(label);
      li++;
    };
    leg('CRLB', '#059669', '4 2');
    if (selected.has('mean')) leg('Mean', '#2563eb');
    if (selected.has('median')) leg('Median', '#dc2626');
    if (selected.has('trimmed')) leg('Trimmed', '#f59e0b');
  }, [curves, selected, n, w]);

  // Right: efficiency bar chart + optional JS dominance
  useEffect(() => {
    if (!rightRef.current) return;
    const svg = d3.select(rightRef.current);
    svg.selectAll('*').remove();

    const panelW = w > 900 ? (w - 24) * 0.4 : w;
    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3
      .scaleBand()
      .domain(efficiencies.map((e) => e.name))
      .range([0, innerW])
      .padding(0.2);
    const y = d3.scaleLinear().domain([0, 1.1]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .style('font-size', '10px');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('font-size', '10px');

    g.selectAll('rect.eff')
      .data(efficiencies)
      .enter()
      .append('rect')
      .attr('class', 'eff')
      .attr('x', (d) => x(d.name) ?? 0)
      .attr('y', (d) => y(Math.min(d.eff, 1.1)))
      .attr('width', x.bandwidth())
      .attr('height', (d) => innerH - y(Math.min(d.eff, 1.1)))
      .attr('fill', (d) => d.color)
      .attr('opacity', 0.85);

    // Efficient reference line at 1.0
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', y(1))
      .attr('y2', y(1))
      .attr('stroke', '#059669')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5 3');

    g.append('text')
      .attr('x', innerW)
      .attr('y', y(1) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#059669')
      .text('efficient');

    // Efficiency labels
    g.selectAll('text.effLabel')
      .data(efficiencies)
      .enter()
      .append('text')
      .attr('class', 'effLabel')
      .attr('x', (d) => (x(d.name) ?? 0) + x.bandwidth() / 2)
      .attr('y', (d) => y(Math.min(d.eff, 1.1)) - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', 'currentColor')
      .text((d) => d.eff.toFixed(3));

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text(`Efficiency at n = ${n}`);
  }, [efficiencies, w, n]);

  const toggleEstimator = (id: EstimatorId) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const jsRisk = useMemo(() => {
    if (!showJamesStein) return { mle: 0, js: 0 };
    const thetaNorm2 = 1; // fixed ||θ||² for illustration
    return {
      mle: riskMLE(jsDimension, spec.trueVariance / n),
      js: riskJamesStein(jsDimension, spec.trueVariance / n, thetaNorm2),
    };
  }, [showJamesStein, jsDimension, n, spec.trueVariance]);

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
        <strong>Interactive: Cramér–Rao Explorer</strong>
        <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
          Compare estimator variances to the CRLB 1/(n·I(θ)). Efficient estimators (bar = 1.0)
          sit on the green envelope; inefficient ones rise above it.
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 720 ? 'repeat(4, auto) 1fr' : '1fr 1fr',
          gap: '0.5rem',
          fontSize: '0.8125rem',
          marginBottom: '0.5rem',
          alignItems: 'center',
        }}
      >
        <label>
          Family:{' '}
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Family)}
            style={{ padding: '0.2rem' }}
          >
            {(Object.keys(FAMILY_SPECS) as Family[]).map((f) => (
              <option key={f} value={f}>
                {FAMILY_SPECS[f].name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <input type="checkbox" checked={selected.has('mean')} onChange={() => toggleEstimator('mean')} /> Mean
        </label>
        <label>
          <input type="checkbox" checked={selected.has('median')} onChange={() => toggleEstimator('median')} /> Median
        </label>
        <label>
          <input type="checkbox" checked={selected.has('trimmed')} onChange={() => toggleEstimator('trimmed')} /> Trimmed
        </label>
        <div style={{ textAlign: 'right', color: 'var(--color-text-muted)' }}>
          I(θ) = {isFinite(I) ? I.toFixed(4) : '∞'}; CRLB ={' '}
          {isFinite(crlbCurrent) ? crlbCurrent.toFixed(5) : '—'}
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.8125rem',
          marginBottom: '0.5rem',
        }}
      >
        <span>n = {n}</span>
        <input
          type="range"
          min={5}
          max={500}
          step={1}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: w > 900 ? '3fr 2fr' : '1fr',
          gap: '0.75rem',
          alignItems: 'start',
        }}
      >
        <svg ref={leftRef} width={w > 900 ? (w - 24) * 0.6 : w} height={H} style={{ maxWidth: '100%' }} />
        <svg ref={rightRef} width={w > 900 ? (w - 24) * 0.4 : w} height={H} style={{ maxWidth: '100%' }} />
      </div>

      <div style={{ marginTop: '0.5rem', fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
        {spec.note}
      </div>

      {spec.supportsJamesStein && (
        <div
          style={{
            marginTop: '1rem',
            padding: '0.75rem',
            border: '1px dashed var(--color-border)',
            borderRadius: '0.5rem',
            fontSize: '0.8125rem',
          }}
        >
          <label>
            <input
              type="checkbox"
              checked={showJamesStein}
              onChange={(e) => setShowJamesStein(e.target.checked)}
            />{' '}
            Show James–Stein dominance (Normal only)
          </label>
          {showJamesStein && (
            <div style={{ marginTop: '0.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <label>
                d = {jsDimension}{' '}
                <input
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={jsDimension}
                  onChange={(e) => setJsDimension(Number(e.target.value))}
                />
              </label>
              <span>
                MLE risk = <strong>{jsRisk.mle.toFixed(4)}</strong>; JS risk ={' '}
                <strong>{jsRisk.js.toFixed(4)}</strong>
                {jsDimension >= 3 ? (
                  <span style={{ color: '#059669' }}> — JS dominates ✓</span>
                ) : (
                  <span style={{ color: 'var(--color-text-muted)' }}> (JS = MLE for d ≤ 2)</span>
                )}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
