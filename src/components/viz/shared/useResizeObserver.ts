import { useRef, useState, useEffect } from 'react';

interface Dimensions {
  width: number;
  height: number;
}

/**
 * Hook that observes the size of a container element.
 * Returns { ref, width, height } — attach ref to a container <div>.
 */
export function useResizeObserver<T extends HTMLElement>(): {
  ref: React.RefObject<T | null>;
  width: number;
  height: number;
} {
  const ref = useRef<T>(null);
  const [dimensions, setDimensions] = useState<Dimensions>({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  return { ref, ...dimensions };
}
