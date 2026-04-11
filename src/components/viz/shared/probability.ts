/**
 * probability.ts — Track 1 (Foundations of Probability) shared utility module.
 *
 * Pure, deterministic functions for probability computations used across
 * interactive components in Topics 1–4. Extended by later topics.
 */

// ── Types ───────────────────────────────────────────────────────────────────

/** A finite sample space represented as a set of string labels. */
export type SampleSpace = Set<string>;

/** An event is a subset of the sample space. */
export type Event = Set<string>;

// ── Set operations ──────────────────────────────────────────────────────────

/** Equally-likely probability: P(A) = |A| / |Ω|. */
export function equallyLikelyP(event: Event, omega: SampleSpace): number {
  if (omega.size === 0) return 0;
  return event.size / omega.size;
}

/** Complement of A in Ω. */
export function complement(a: Event, omega: SampleSpace): Event {
  const result = new Set<string>();
  for (const x of omega) {
    if (!a.has(x)) result.add(x);
  }
  return result;
}

/** Union of two events. */
export function union(a: Event, b: Event): Event {
  return new Set([...a, ...b]);
}

/** Intersection of two events. */
export function intersection(a: Event, b: Event): Event {
  const result = new Set<string>();
  for (const x of a) {
    if (b.has(x)) result.add(x);
  }
  return result;
}

/** Set difference A \ B. */
export function setDifference(a: Event, b: Event): Event {
  const result = new Set<string>();
  for (const x of a) {
    if (!b.has(x)) result.add(x);
  }
  return result;
}

/** Check if two events are disjoint. */
export function areDisjoint(a: Event, b: Event): boolean {
  for (const x of a) {
    if (b.has(x)) return false;
  }
  return true;
}

/** Power set of a finite set. Returns array of all subsets. */
export function powerSet<T>(s: Set<T>): Set<T>[] {
  const elements = [...s];
  const n = elements.length;
  const result: Set<T>[] = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    const subset = new Set<T>();
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) subset.add(elements[i]);
    }
    result.push(subset);
  }
  return result;
}

// ── Sigma-algebra validation ────────────────────────────────────────────────

/** Serialize a Set to a canonical string key for comparison. */
function serializeSet(s: Set<string>): string {
  return [...s].sort().join(',');
}

/**
 * Check if a collection of subsets forms a valid sigma-algebra over omega.
 * Returns { valid, violations } where violations lists specific failures.
 */
export function isValidSigmaAlgebra(
  omega: SampleSpace,
  collection: Event[],
): { valid: boolean; violations: string[] } {
  const violations: string[] = [];
  const serialized = new Set(collection.map(serializeSet));
  const omegaKey = serializeSet(omega);
  const emptyKey = serializeSet(new Set<string>());

  // 0. Every event must be a subset of Ω
  for (const event of collection) {
    for (const el of event) {
      if (!omega.has(el)) {
        const eventStr = `{${[...event].sort().join(',')}}`;
        violations.push(`${eventStr} contains element "${el}" not in Ω`);
        break;
      }
    }
  }

  // 1. Ω must be in F
  if (!serialized.has(omegaKey)) {
    violations.push('Ω is not in the collection');
  }

  // 2. ∅ must be in F (consequence of Ω ∈ F + complement closure)
  if (!serialized.has(emptyKey)) {
    violations.push('∅ is not in the collection');
  }

  // 3. Closed under complements
  for (const event of collection) {
    const comp = complement(event, omega);
    const compKey = serializeSet(comp);
    if (!serialized.has(compKey)) {
      const eventStr = event.size === 0 ? '∅' : `{${[...event].sort().join(',')}}`;
      const compStr = comp.size === 0 ? '∅' : `{${[...comp].sort().join(',')}}`;
      violations.push(`Complement of ${eventStr} = ${compStr} is missing`);
    }
  }

  // 4. Closed under pairwise unions (sufficient for finite case)
  for (let i = 0; i < collection.length; i++) {
    for (let j = i + 1; j < collection.length; j++) {
      const u = union(collection[i], collection[j]);
      const uKey = serializeSet(u);
      if (!serialized.has(uKey)) {
        const aStr = collection[i].size === 0 ? '∅' : `{${[...collection[i]].sort().join(',')}}`;
        const bStr = collection[j].size === 0 ? '∅' : `{${[...collection[j]].sort().join(',')}}`;
        const uStr = u.size === 0 ? '∅' : `{${[...u].sort().join(',')}}`;
        violations.push(`Union ${aStr} ∪ ${bStr} = ${uStr} is missing`);
      }
    }
  }

  return { valid: violations.length === 0, violations };
}

// ── Counting ────────────────────────────────────────────────────────────────

/** n! (factorial). Returns Infinity for n > 170. */
export function factorial(n: number): number {
  if (n < 0) return NaN;
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

/** P(n, k) = n! / (n-k)! (permutations). */
export function permutations(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) result *= n - i;
  return result;
}

/** C(n, k) = n! / (k!(n-k)!) (combinations / binomial coefficient). */
export function combinations(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  // Use the smaller k for efficiency
  const kk = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < kk; i++) {
    result = (result * (n - i)) / (i + 1);
  }
  return Math.round(result);
}

// ── Birthday problem ────────────────────────────────────────────────────────

/** P(at least one shared birthday) for n people, d days (default 365). */
export function birthdayProbability(n: number, d: number = 365): number {
  if (n <= 1) return 0;
  if (n > d) return 1;
  let pNoMatch = 1;
  for (let k = 0; k < n; k++) {
    pNoMatch *= (d - k) / d;
  }
  return 1 - pNoMatch;
}

/**
 * Count derangements D(n) using inclusion-exclusion.
 * D(n) = n! * Σ_{k=0}^{n} (-1)^k / k!
 */
export function derangementCount(n: number): number {
  if (n === 0) return 1;
  if (n === 1) return 0;
  const nFact = factorial(n);
  let sum = 0;
  for (let k = 0; k <= n; k++) {
    sum += (k % 2 === 0 ? 1 : -1) / factorial(k);
  }
  return Math.round(nFact * sum);
}

/** P(derangement) = D(n) / n!. */
export function derangementProbability(n: number): number {
  if (n <= 0) return 1;
  let sum = 0;
  for (let k = 0; k <= n; k++) {
    sum += (k % 2 === 0 ? 1 : -1) / factorial(k);
  }
  return sum;
}

// ── Inclusion-exclusion ─────────────────────────────────────────────────────

/** Inclusion-exclusion for 2 events: P(A∪B) = P(A) + P(B) - P(A∩B). */
export function inclusionExclusion2(
  pA: number,
  pB: number,
  pAB: number,
): number {
  return pA + pB - pAB;
}

/** Inclusion-exclusion for 3 events. */
export function inclusionExclusion3(
  pA: number,
  pB: number,
  pC: number,
  pAB: number,
  pAC: number,
  pBC: number,
  pABC: number,
): number {
  return pA + pB + pC - pAB - pAC - pBC + pABC;
}

// ── Monte Carlo ─────────────────────────────────────────────────────────────

/**
 * Seeded PRNG for reproducible simulations.
 * Uses a simple linear congruential generator.
 */
export function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    return (state >>> 0) / 0x100000000;
  };
}

/**
 * Monte Carlo birthday simulation: returns fraction of trials with a match.
 * @param n - group size
 * @param nTrials - number of simulation trials
 * @param rng - random number generator (use seededRandom for reproducibility)
 * @param d - number of days (default 365)
 */
export function mcBirthday(
  n: number,
  nTrials: number,
  rng: () => number,
  d: number = 365,
): number {
  let matches = 0;
  for (let trial = 0; trial < nTrials; trial++) {
    const birthdays = new Set<number>();
    let hasMatch = false;
    for (let i = 0; i < n; i++) {
      const day = Math.floor(rng() * d);
      if (birthdays.has(day)) {
        hasMatch = true;
        break;
      }
      birthdays.add(day);
    }
    if (hasMatch) matches++;
  }
  return matches / nTrials;
}
