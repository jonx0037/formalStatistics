/**
 * Annual flow of the Nile at Aswan, 1871–1970 (n = 100), in 10⁸ m³ — Topic 36,
 * State-Space Models & Kalman Smoothing (§36.1 Exs 1–2, §36.3 Ex 5, §36.5
 * Exs 9–10, §36.6 Exs 11–12, §36.8 Ex 14; KalmanSmootherExplorer,
 * StateSpaceLikelihoodSurface).
 *
 * Cobb, "The Problem of the Nile" (COB1978), and Durbin & Koopman's running
 * example (DUR2012); public domain, as bundled in statsmodels
 * (`sm.datasets.nile`) and written by notebook Cell 4
 * (notebooks/state-space-models-and-kalman-smoothing/36_state_space_models.ipynb,
 * data/nile.json). See docs/formalstatistics-state-space-models-and-kalman-smoothing-handoff-brief.md §7.
 */

export const nileStartYear = 1871;

export const nileValues: number[] = [
  1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
  995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
  1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
  874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
  831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
  768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
  781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
  649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
  744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
  1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
];
