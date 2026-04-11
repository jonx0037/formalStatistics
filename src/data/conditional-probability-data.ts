/**
 * conditional-probability-data.ts — Preset data for Topic 2 interactive components.
 */

/** Medical testing presets for BayesTheoremExplorer. */
export const medicalTestPresets = [
  {
    name: 'Mammography',
    prevalence: 0.01,
    sensitivity: 0.90,
    specificity: 0.93,
    description: 'Breast cancer screening in the general population',
  },
  {
    name: 'COVID Rapid Test',
    prevalence: 0.05,
    sensitivity: 0.85,
    specificity: 0.995,
    description: 'Rapid antigen test during moderate prevalence',
  },
  {
    name: 'Rare Disease Screen',
    prevalence: 0.001,
    sensitivity: 0.99,
    specificity: 0.99,
    description: 'Screening for a condition affecting 1 in 1000',
  },
  {
    name: 'HIV ELISA',
    prevalence: 0.003,
    sensitivity: 0.998,
    specificity: 0.998,
    description: 'HIV screening in the general US population',
  },
];

/** Preset events for IndependenceTester (single die). */
export const dieIndependencePresets = [
  {
    name: 'Even & ≥ 5',
    eventA: { name: 'Even', outcomes: ['2', '4', '6'] },
    eventB: { name: '≥ 5', outcomes: ['5', '6'] },
    independent: false,
  },
  {
    name: 'Even & ≤ 3',
    eventA: { name: 'Even', outcomes: ['2', '4', '6'] },
    eventB: { name: '≤ 3', outcomes: ['1', '2', '3'] },
    independent: true,
  },
  {
    name: 'Prime & Odd',
    eventA: { name: 'Prime', outcomes: ['2', '3', '5'] },
    eventB: { name: 'Odd', outcomes: ['1', '3', '5'] },
    independent: false,
  },
];

/** Conditional independence presets for ConditionalIndependenceExplorer. */
export const conditionalIndependencePresets = [
  {
    name: 'Confounding (Marg. Dep., Cond. Indep.)',
    description:
      'A and B are both caused by C — marginally dependent but conditionally independent given C',
    // P(A=a, B=b, C=c) for (a,b,c) in {0,1}³, ordered as 000,001,010,011,100,101,110,111
    jointProbs: [0.36, 0.04, 0.04, 0.16, 0.04, 0.16, 0.16, 0.04],
  },
  {
    name: 'Explaining Away (Marg. Indep., Cond. Dep.)',
    description:
      'A and B are independent causes of C — marginally independent but conditionally dependent given C',
    jointProbs: [0.36, 0.09, 0.09, 0.01, 0.09, 0.16, 0.01, 0.19],
  },
  {
    name: 'Both Independent',
    description: 'A, B, C are mutually independent',
    jointProbs: [0.216, 0.144, 0.144, 0.096, 0.144, 0.096, 0.096, 0.064],
  },
];

/** Monty Hall presets. */
export const montyHallSetup = {
  doors: 3,
  description:
    'Car behind one door, goats behind the other two. You pick, host reveals a goat, you decide: switch or stay.',
};
