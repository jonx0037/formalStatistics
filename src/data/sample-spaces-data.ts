/** Preset sample spaces for the SampleSpaceExplorer. */
export interface SampleSpacePreset {
  name: string;
  outcomes: string[];
  description: string;
  layout?: 'list' | 'grid';
  gridCols?: number;
}

export const sampleSpacePresets: SampleSpacePreset[] = [
  {
    name: 'Coin Flip',
    outcomes: ['H', 'T'],
    description: 'Flip a fair coin',
    layout: 'list',
  },
  {
    name: 'Die Roll',
    outcomes: ['1', '2', '3', '4', '5', '6'],
    description: 'Roll a six-sided die',
    layout: 'list',
  },
  {
    name: 'Two Coins',
    outcomes: ['HH', 'HT', 'TH', 'TT'],
    description: 'Flip two coins (ordered)',
    layout: 'list',
  },
  {
    name: 'Two Dice',
    outcomes: (() => {
      const pairs: string[] = [];
      for (let i = 1; i <= 6; i++) {
        for (let j = 1; j <= 6; j++) {
          pairs.push(`(${i},${j})`);
        }
      }
      return pairs;
    })(),
    description: 'Roll two dice (ordered pairs)',
    layout: 'grid',
    gridCols: 6,
  },
];

/** Preset sigma-algebras for the SigmaAlgebraExplorer. */
export interface SigmaAlgebraPreset {
  name: string;
  omega: string[];
  events: string[][]; // each inner array is an event (list of outcomes)
}

export const sigmaAlgebraPresets: SigmaAlgebraPreset[] = [
  {
    name: 'Trivial',
    omega: ['1', '2', '3', '4'],
    events: [[], ['1', '2', '3', '4']],
  },
  {
    name: 'Partition {1,2} | {3,4}',
    omega: ['1', '2', '3', '4'],
    events: [[], ['1', '2'], ['3', '4'], ['1', '2', '3', '4']],
  },
  {
    name: 'Discrete (Power Set)',
    omega: ['1', '2', '3', '4'],
    events: (() => {
      const omega = ['1', '2', '3', '4'];
      const result: string[][] = [];
      for (let mask = 0; mask < 16; mask++) {
        const subset: string[] = [];
        for (let i = 0; i < 4; i++) {
          if (mask & (1 << i)) subset.push(omega[i]);
        }
        result.push(subset);
      }
      return result;
    })(),
  },
];

/** Named events for die roll examples. */
export const dieEvents: Record<string, { name: string; outcomes: string[] }> = {
  even: { name: 'Even', outcomes: ['2', '4', '6'] },
  odd: { name: 'Odd', outcomes: ['1', '3', '5'] },
  atLeast5: { name: '≥ 5', outcomes: ['5', '6'] },
  prime: { name: 'Prime', outcomes: ['2', '3', '5'] },
};
