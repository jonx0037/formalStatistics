import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  olsFit,
  leverage,
  studentizedResiduals,
  cooksDistance,
  qqPlotData,
} from './shared/regression';
import {
  REGRESSION_DIAGNOSTICS_PRESETS,
  generateLinearModelData,
  type DiagnosticKind,
} from '../../data/regression-data';

// ── Layout ───────────────────────────────────────────────────────────────────
const PANEL_W = 420;
const PANEL_H = 280;
const PANEL_MARGIN = { top: 28, right: 20, bottom: 40, left: 48 };

// ── Palette ──────────────────────────────────────────────────────────────────
const POINT_COLOR = '#2563EB'; // blue
const AXIS_COLOR = '#9CA3AF';
const REFERENCE_COLOR = '#D97706'; // amber for reference lines
const COOK_HI = '#DC2626'; // red for high Cook's distance

// ── DGP seed (fixed so "Fit the model" changes it, but preset swap keeps it) ─
const DEFAULT_SEED = 424242;

export default function RegressionDiagnosticsExplorer() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 900, 320);

  const [activePreset, setActivePreset] = useState<DiagnosticKind>('ideal');
  const [seed, setSeed] = useState<number>(DEFAULT_SEED);

  const preset = REGRESSION_DIAGNOSTICS_PRESETS[activePreset];

  // Each diagnostic scenario fits a simple intercept + slope model on n=80 points.
  const fit = useMemo(() => {
    const { X, y } = generateLinearModelData({
      n: 80,
      p: 1,
      beta: [1, 2],
      sigma: 0.8,
      designKind: preset.kind,
      seed,
    });
    try {
      return { fit: olsFit(X, y), X, y };
    } catch {
      return null;
    }
  }, [preset.kind, seed]);

  const diag = useMemo(() => {
    if (!fit) return null;
    const lev = leverage(fit.X);
    const stu = studentizedResiduals(fit.fit, fit.X);
    const cook = cooksDistance(fit.fit, fit.X);
    const qq = qqPlotData(fit.fit, fit.X);
    const pPlusOne = fit.fit.p + 1;
    return { lev, stu, cook, qq, pPlusOne, leverageThreshold: (2 * pPlusOne) / fit.fit.n };
  }, [fit]);

  if (!fit || !diag) {
    return <div className="my-8 p-3 text-sm text-slate-500">Diagnostic fit failed.</div>;
  }

  // Shared sizing — 2×2 on desktop, stacked on mobile via CSS grid.
  const layoutClass = width < 720 ? 'grid-cols-1' : 'grid-cols-2';

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-surface, #ffffff)',
      }}
    >
      {/* Preset selector */}
      <div
        className="flex flex-wrap items-center gap-2 border-b px-3 py-2 text-sm"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        {Object.values(REGRESSION_DIAGNOSTICS_PRESETS).map((p) => (
          <button
            key={p.kind}
            type="button"
            onClick={() => setActivePreset(p.kind)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              activePreset === p.kind
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            {p.label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded bg-slate-100 px-3 py-1 text-xs hover:bg-slate-200"
        >
          Fit the model (resample)
        </button>
      </div>

      {/* Plot grid */}
      <div className={`grid gap-3 p-3 ${layoutClass}`}>
        <PanelResidualsVsFitted fit={fit.fit} />
        <PanelQQ qq={diag.qq} />
        <PanelScaleLocation fit={fit.fit} stu={diag.stu} />
        <PanelLeverageCook stu={diag.stu} lev={diag.lev} cook={diag.cook} thr={diag.leverageThreshold} />
      </div>

      {/* Diagnosis */}
      <div
        className="border-t px-3 py-3 text-sm leading-relaxed"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Diagnosis</div>
        {preset.diagnosis}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Plot panels — small focused components, each doing one diagnostic view.
// ─────────────────────────────────────────────────────────────────────────────

function PanelResidualsVsFitted({ fit }: { fit: ReturnType<typeof olsFit> }) {
  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(fit.fitted) as [number, number])
    .nice()
    .range([PANEL_MARGIN.left, PANEL_W - PANEL_MARGIN.right]);
  const residExt = d3.extent(fit.residuals) as [number, number];
  const residAbs = Math.max(Math.abs(residExt[0]), Math.abs(residExt[1]));
  const yScale = d3
    .scaleLinear()
    .domain([-residAbs * 1.1, residAbs * 1.1])
    .range([PANEL_H - PANEL_MARGIN.bottom, PANEL_MARGIN.top]);

  return (
    <PanelChrome title="Residuals vs Fitted">
      <svg viewBox={`0 0 ${PANEL_W} ${PANEL_H}`} width="100%">
        <Axes xScale={xScale} yScale={yScale} xLabel="fitted" yLabel="residual" />
        <line
          x1={PANEL_MARGIN.left}
          x2={PANEL_W - PANEL_MARGIN.right}
          y1={yScale(0)}
          y2={yScale(0)}
          stroke={REFERENCE_COLOR}
          strokeDasharray="3 3"
        />
        {fit.fitted.map((f, i) => (
          <circle
            key={i}
            cx={xScale(f)}
            cy={yScale(fit.residuals[i])}
            r={3}
            fill={POINT_COLOR}
            opacity={0.7}
          />
        ))}
      </svg>
    </PanelChrome>
  );
}

function PanelQQ({ qq }: { qq: { theoretical: number[]; sample: number[] } }) {
  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(qq.theoretical) as [number, number])
    .nice()
    .range([PANEL_MARGIN.left, PANEL_W - PANEL_MARGIN.right]);
  const yScale = d3
    .scaleLinear()
    .domain(d3.extent(qq.sample) as [number, number])
    .nice()
    .range([PANEL_H - PANEL_MARGIN.bottom, PANEL_MARGIN.top]);

  // Identity reference line — draw only within the plot interior.
  const lo = Math.max(xScale.domain()[0], yScale.domain()[0]);
  const hi = Math.min(xScale.domain()[1], yScale.domain()[1]);

  return (
    <PanelChrome title="Normal Q-Q">
      <svg viewBox={`0 0 ${PANEL_W} ${PANEL_H}`} width="100%">
        <Axes xScale={xScale} yScale={yScale} xLabel="theoretical quantile" yLabel="sample quantile" />
        <line
          x1={xScale(lo)}
          y1={yScale(lo)}
          x2={xScale(hi)}
          y2={yScale(hi)}
          stroke={REFERENCE_COLOR}
          strokeDasharray="3 3"
        />
        {qq.theoretical.map((t, i) => (
          <circle
            key={i}
            cx={xScale(t)}
            cy={yScale(qq.sample[i])}
            r={3}
            fill={POINT_COLOR}
            opacity={0.7}
          />
        ))}
      </svg>
    </PanelChrome>
  );
}

function PanelScaleLocation({
  fit,
  stu,
}: {
  fit: ReturnType<typeof olsFit>;
  stu: number[];
}) {
  const sqrtAbs = stu.map((r) => Math.sqrt(Math.abs(r)));
  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(fit.fitted) as [number, number])
    .nice()
    .range([PANEL_MARGIN.left, PANEL_W - PANEL_MARGIN.right]);
  const yScale = d3
    .scaleLinear()
    .domain([0, (d3.max(sqrtAbs) ?? 1) * 1.1])
    .range([PANEL_H - PANEL_MARGIN.bottom, PANEL_MARGIN.top]);

  return (
    <PanelChrome title="Scale-Location (√|r| vs Fitted)">
      <svg viewBox={`0 0 ${PANEL_W} ${PANEL_H}`} width="100%">
        <Axes xScale={xScale} yScale={yScale} xLabel="fitted" yLabel="√|studentized r|" />
        {fit.fitted.map((f, i) => (
          <circle
            key={i}
            cx={xScale(f)}
            cy={yScale(sqrtAbs[i])}
            r={3}
            fill={POINT_COLOR}
            opacity={0.7}
          />
        ))}
      </svg>
    </PanelChrome>
  );
}

function PanelLeverageCook({
  stu,
  lev,
  cook,
  thr,
}: {
  stu: number[];
  lev: number[];
  cook: number[];
  thr: number;
}) {
  const levExt = d3.extent(lev) as [number, number];
  const xScale = d3
    .scaleLinear()
    .domain([0, Math.max(levExt[1], thr * 2) * 1.1])
    .range([PANEL_MARGIN.left, PANEL_W - PANEL_MARGIN.right]);
  const stuAbs = Math.max(Math.abs(d3.min(stu) ?? 0), Math.abs(d3.max(stu) ?? 0));
  const yScale = d3
    .scaleLinear()
    .domain([-stuAbs * 1.1, stuAbs * 1.1])
    .range([PANEL_H - PANEL_MARGIN.bottom, PANEL_MARGIN.top]);

  return (
    <PanelChrome title="Residuals vs Leverage (colored by Cook's D)">
      <svg viewBox={`0 0 ${PANEL_W} ${PANEL_H}`} width="100%">
        <Axes xScale={xScale} yScale={yScale} xLabel="leverage h_ii" yLabel="studentized r" />
        <line
          x1={xScale(thr)}
          x2={xScale(thr)}
          y1={yScale(yScale.domain()[0])}
          y2={yScale(yScale.domain()[1])}
          stroke={REFERENCE_COLOR}
          strokeDasharray="3 3"
        />
        <text
          x={xScale(thr) + 4}
          y={PANEL_MARGIN.top + 10}
          fontSize={10}
          fill={REFERENCE_COLOR}
        >
          2(p+1)/n = {thr.toFixed(3)}
        </text>
        {lev.map((h, i) => {
          const highCook = cook[i] > 0.5;
          return (
            <circle
              key={i}
              cx={xScale(h)}
              cy={yScale(stu[i])}
              r={highCook ? 5 : 3}
              fill={highCook ? COOK_HI : POINT_COLOR}
              opacity={highCook ? 0.9 : 0.7}
            />
          );
        })}
      </svg>
    </PanelChrome>
  );
}

// ── Shared chrome / axes ─────────────────────────────────────────────────────

function PanelChrome({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-md border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-muted-bg, #f8fafc)',
      }}
    >
      <div className="border-b px-3 py-1.5 text-xs font-medium text-slate-700" style={{ borderColor: 'var(--color-border, #e5e7eb)' }}>
        {title}
      </div>
      <div className="p-1">{children}</div>
    </div>
  );
}

function Axes({
  xScale,
  yScale,
  xLabel,
  yLabel,
}: {
  xScale: d3.ScaleLinear<number, number>;
  yScale: d3.ScaleLinear<number, number>;
  xLabel: string;
  yLabel: string;
}) {
  return (
    <>
      {/* X axis */}
      <g transform={`translate(0, ${PANEL_H - PANEL_MARGIN.bottom})`}>
        <line x1={PANEL_MARGIN.left} x2={PANEL_W - PANEL_MARGIN.right} stroke={AXIS_COLOR} />
        {xScale.ticks(5).map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, 0)`}>
            <line y1={0} y2={3} stroke={AXIS_COLOR} />
            <text y={14} textAnchor="middle" fontSize={9} fill={AXIS_COLOR}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          x={(PANEL_MARGIN.left + PANEL_W - PANEL_MARGIN.right) / 2}
          y={30}
          textAnchor="middle"
          fontSize={10}
          fill={AXIS_COLOR}
        >
          {xLabel}
        </text>
      </g>
      {/* Y axis */}
      <g transform={`translate(${PANEL_MARGIN.left}, 0)`}>
        <line y1={PANEL_MARGIN.top} y2={PANEL_H - PANEL_MARGIN.bottom} stroke={AXIS_COLOR} />
        {yScale.ticks(5).map((t) => (
          <g key={t} transform={`translate(0, ${yScale(t)})`}>
            <line x1={-3} x2={0} stroke={AXIS_COLOR} />
            <text x={-5} dy="0.32em" textAnchor="end" fontSize={9} fill={AXIS_COLOR}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          transform={`rotate(-90) translate(${-(PANEL_H / 2)}, ${-36})`}
          textAnchor="middle"
          fontSize={10}
          fill={AXIS_COLOR}
        >
          {yLabel}
        </text>
      </g>
    </>
  );
}
