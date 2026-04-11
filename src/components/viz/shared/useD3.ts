import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

/**
 * Hook that creates a D3-managed SVG element.
 * Calls renderFn whenever deps change, clearing previous content first.
 */
export function useD3(
  renderFn: (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => void,
  deps: unknown[],
): React.RefObject<SVGSVGElement | null> {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    renderFn(svg);
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  return ref;
}
