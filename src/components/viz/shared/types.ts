/**
 * Shared types for statistics visualization components.
 */

export interface Point {
  x: number;
  y: number;
}

export interface DataPoint extends Point {
  label?: string;
}

export interface DistributionParams {
  name: string;
  params: Record<string, number>;
}
