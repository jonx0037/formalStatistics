import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { domainColorScale } from './shared/colorScales';
import graphData from '../../data/curriculum-graph.json';

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  domain: string;
  status: 'published' | 'planned';
  url: string;
}

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
}

const DOMAIN_ORDER = [
  'foundations-of-probability',
  'core-distributions',
  'convergence-limit-theorems',
  'statistical-estimation',
  'hypothesis-testing-confidence',
  'regression-linear-models',
  'bayesian-statistics',
  'high-dimensional-nonparametric',
];

const DOMAIN_LABELS: Record<string, string> = {
  'foundations-of-probability': 'Probability',
  'core-distributions': 'Distributions',
  'convergence-limit-theorems': 'Convergence',
  'statistical-estimation': 'Estimation',
  'hypothesis-testing-confidence': 'Testing',
  'regression-linear-models': 'Regression',
  'bayesian-statistics': 'Bayesian',
  'high-dimensional-nonparametric': 'Nonparametric',
};

export default function CurriculumGraph() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const height = Math.min(Math.max(width * 0.65, 400), 650);
  const svgRef = useRef<SVGSVGElement>(null);

  // Deep-copy nodes/edges so D3 mutation doesn't corrupt the module-level data
  const { nodes, edges } = useMemo(() => {
    const n: GraphNode[] = graphData.nodes.map((d) => ({ ...d }) as GraphNode);
    const e: GraphEdge[] = graphData.edges.map((d) => ({ ...d }));
    return { nodes: n, edges: e };
  }, []);

  useEffect(() => {
    if (!svgRef.current || width <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    svg.call(zoom);

    // Initial positions: arrange by domain columns, topic order within column
    const colWidth = width / (DOMAIN_ORDER.length + 1);
    nodes.forEach((node) => {
      const colIdx = DOMAIN_ORDER.indexOf(node.domain);
      const domainNodes = nodes.filter((n) => n.domain === node.domain);
      const rowIdx = domainNodes.indexOf(node);
      const rowHeight = height / (domainNodes.length + 1);
      node.x = (colIdx + 1) * colWidth;
      node.y = (rowIdx + 1) * rowHeight;
    });

    // Force simulation
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force('link', d3.forceLink<GraphNode, GraphEdge>(edges)
        .id((d) => d.id)
        .distance(80)
        .strength(0.4))
      .force('charge', d3.forceManyBody().strength(-250))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('x', d3.forceX<GraphNode>((d) => {
        const colIdx = DOMAIN_ORDER.indexOf(d.domain);
        return (colIdx + 1) * colWidth;
      }).strength(0.3))
      .force('y', d3.forceY(height / 2).strength(0.05))
      .force('collision', d3.forceCollide(30));

    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .style('fill', 'var(--color-border-strong)');

    // Edges
    const links = g.selectAll('.link')
      .data(edges)
      .join('line')
      .attr('class', 'link')
      .style('stroke', 'var(--color-border-strong)')
      .style('stroke-width', 1.2)
      .style('opacity', 0.5)
      .attr('marker-end', 'url(#arrow)');

    // Node groups — use SVG <a> for published topics
    const nodeGroups = g.selectAll('.node')
      .data(nodes)
      .join('a')
      .attr('class', 'node')
      .attr('href', (d) => d.status === 'published' ? d.url : null)
      .style('cursor', (d) => d.status === 'published' ? 'pointer' : 'default')
      .on('click', (event, d) => {
        if (d.status !== 'published') {
          event.preventDefault();
        }
      });

    // Node circles
    nodeGroups.append('circle')
      .attr('r', (d) => d.status === 'published' ? 10 : 7)
      .style('fill', (d) => d.status === 'published' ? domainColorScale(d.domain) : 'var(--color-surface-alt)')
      .style('stroke', (d) => domainColorScale(d.domain))
      .style('stroke-width', (d) => d.status === 'published' ? 2.5 : 1.5)
      .style('opacity', (d) => d.status === 'published' ? 1 : 0.5);

    // Node labels
    nodeGroups.append('text')
      .text((d) => {
        const maxLen = width < 600 ? 18 : 30;
        return d.label.length > maxLen ? d.label.slice(0, maxLen - 1) + '\u2026' : d.label;
      })
      .attr('dy', -14)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', width < 600 ? '8px' : '10px')
      .style('font-weight', (d) => d.status === 'published' ? '600' : '400')
      .style('opacity', (d) => d.status === 'published' ? 1 : 0.6)
      .style('pointer-events', 'none');

    // Drag behavior
    const drag = d3.drag<SVGAElement, GraphNode>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodeGroups.call(drag);

    // Tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d) => (d.source as GraphNode).x!)
        .attr('y1', (d) => (d.source as GraphNode).y!)
        .attr('x2', (d) => (d.target as GraphNode).x!)
        .attr('y2', (d) => (d.target as GraphNode).y!);

      nodeGroups.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });

    // Start simulation
    simulation.alpha(1).restart();

    return () => {
      simulation.stop();
    };
  }, [width, height, nodes, edges]);

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mb-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        {DOMAIN_ORDER.map((domain) => (
          <span key={domain} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ background: domainColorScale(domain) }}
            />
            {DOMAIN_LABELS[domain]}
          </span>
        ))}
        <span className="ml-2 italic">Drag nodes &middot; Scroll to zoom</span>
      </div>

      <div
        className="rounded-lg border overflow-hidden"
        style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)' }}
      >
        <svg
          ref={svgRef}
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
        />
      </div>
    </div>
  );
}
