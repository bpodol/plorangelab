# Exact enumeration vs adaptive Monte Carlo findings

## Methodology

Paired deterministic inputs were used for both methods. For each mode, street, and player count, the board is fixed and balanced ranges use nested hand prefixes as sizes grow. The 76 scenarios cover heads-up PLO4/PLO5 preflop, flop, turn, and river; small workloads through about 1.2 million estimates; low, moderate, and high collision profiles; nonuniform weights; unbalanced 1xN ranges; and representative three-way cases. A coarse sweep was followed by intermediate turn and river points wherever the median winner changed.

Both timings include validation and range preparation; scenario generation and first-use Numba compilation are excluded. Exact uses `run_exact_equity(max_showdowns=None)` only in the temporary harness. Adaptive MC uses target MOE 0.001, minimum 2,000, maximum 1,000,000, with a fresh RNG per independent seed. Estimates above 2,000,000 are skipped for exact under the benchmark budget; production limits are unchanged. Each scenario has a 45-second budget and borderline repeats add up to 15 seconds. Exact timing uses five repetitions, seven at close points; MC uses three seeds in the coarse/extras sweep, five in refinements, and four additional seeds for borderline points.

Hardware: Intel64 Family 6 Model 158 Stepping 13, GenuineIntel; 8 logical CPUs; 8 Numba threads; Windows-11-10.0.26200-SP0.
Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0; timing repetitions use median and MAD.

## Crossover summary

Thresholds use balanced, low-collision heads-up ranges. They are approximate: the report interpolates log(exact time / MC time) between adjacent clear winners. Borderline results remain labeled inconclusive. The clear/inconclusive timing label is a MAD-based noise screen, not a formal significance test.

| Mode | Street | Approximate crossover / measured bound |
|---|---|---|
| PLO4 | preflop | Adaptive MC clear from 1,086,008; lower crossover unmeasured |
| PLO4 | flop | Exact clear through 1,184,080; no crossover measured |
| PLO4 | turn | ~1,039,998 estimated (bracket 806,560-1,128,960) |
| PLO4 | river | ~976,892 estimated (bracket 498,436-1,132,096) |
| PLO5 | preflop | Adaptive MC clear from 850,668; lower crossover unmeasured |
| PLO5 | flop | Exact clear through 1,185,600; no crossover measured |
| PLO5 | turn | ~887,637 estimated (bracket 493,848-1,034,550) |
| PLO5 | river | ~1,012,081 estimated (bracket 810,000-1,132,096) |

## Per-scenario results

`Collision` is the unweighted / weight-adjusted hand-tuple collision rate. Exact and MC times are median +/- MAD in milliseconds. MC simulation counts, stop reasons, and maximum estimated MOE are ranges across measured independent seeds. Equity difference is the maximum absolute player-equity difference from exact. Speedup is the slower median divided by the faster median.

| Scenario | Estimated / actual valid | Collision u/w | Exact ms | MC ms | MC sims; stop | Max MOE | Max abs equity diff | Speedup | Result |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| PLO4 preflop 1x1 low uniform | 1,086,008 / 1,086,008 | 0.0% / 0.0% | 3620.51 +/- 5.86 | 1998.14 +/- 1.48 | 962,000-962,000; target_reached; 3/3 seeds | 0.000998-0.000998 | 0.000868 | 1.81x | adaptive MC |
| PLO4 preflop 2x2 low uniform | 4,344,032 / 4,344,032 | 0.0% / 0.0% | skipped: estimate exceeds 2,000,000 budget | 2035.69 +/- 15.73 | 942,000-942,000; target_reached; 3/3 seeds | 0.000999-0.000999 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO4 flop 1x1 low uniform | 820 / 820 | 0.0% / 0.0% | 3.89 +/- 0.07 | 1092.23 +/- 7.29 | 752,000-752,000; target_reached; 3/3 seeds | 0.000996-0.000997 | 0.001004 | 280.88x | exact |
| PLO4 flop 3x3 low uniform | 7,380 / 7,380 | 0.0% / 0.0% | 7.71 +/- 0.08 | 1450.43 +/- 13.97 | 962,000-962,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000595 | 188.23x | exact |
| PLO4 flop 11x11 low uniform | 99,220 / 99,220 | 0.0% / 0.0% | 65.70 +/- 1.72 | 1315.98 +/- 17.20 | 832,000-832,000; target_reached; 3/3 seeds | 0.000997-0.000998 | 0.000500 | 20.03x | exact |
| PLO4 flop 38x38 low uniform | 1,184,080 / 1,184,080 | 0.0% / 0.0% | 790.25 +/- 14.16 | 1471.58 +/- 19.70 | 942,000-942,000; target_reached; 3/3 seeds | 0.001000-0.001000 | 0.000378 | 1.86x | exact |
| PLO4 turn 1x1 low uniform | 40 / 40 | 0.0% / 0.0% | 0.66 +/- 0.04 | 944.07 +/- 22.30 | 812,000-812,000; target_reached; 3/3 seeds | 0.000996-0.000997 | 0.000924 | 1439.13x | exact |
| PLO4 turn 16x16 low uniform | 10,240 / 10,240 | 0.0% / 0.0% | 7.67 +/- 0.36 | 929.54 +/- 5.98 | 732,000-732,000; target_reached; 3/3 seeds | 0.000995-0.000996 | 0.000481 | 121.21x | exact |
| PLO4 turn 50x50 low uniform | 100,000 / 100,000 | 0.0% / 0.0% | 69.23 +/- 1.20 | 869.59 +/- 4.46 | 652,000-662,000; target_reached; 3/3 seeds | 0.000993-0.000999 | 0.001190 | 12.56x | exact |
| PLO4 turn 173x173 low uniform | 1,197,160 / 1,197,160 | 0.0% / 0.0% | 808.22 +/- 11.37 | 687.03 +/- 4.34 | 502,000-502,000; target_reached; 3/3 seeds | 0.000993-0.000996 | 0.000777 | 1.18x | adaptive MC |
| PLO4 river 1x1 low uniform | 1 / 1 | 0.0% / 0.0% | 0.45 +/- 0.02 | 2.63 +/- 0.32 | 2,000-2,000; target_reached; 3/3 seeds | 0.000000-0.000000 | 0.000000 | 5.81x | exact |
| PLO4 river 100x100 low uniform | 10,000 / 10,000 | 0.0% / 0.0% | 11.30 +/- 0.14 | 1021.42 +/- 21.73 | 942,000-942,000; target_reached; 3/3 seeds | 0.000998-0.000999 | 0.000981 | 90.36x | exact |
| PLO4 river 316x316 low uniform | 99,856 / 99,856 | 0.0% / 0.0% | 110.34 +/- 0.84 | 1190.19 +/- 63.89 | 952,000-952,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000595 | 10.79x | exact |
| PLO4 river 1095x1095 low uniform | 1,199,025 / 1,199,025 | 0.0% / 0.0% | 1318.08 +/- 29.18 | 1150.82 +/- 56.15 | 922,000-922,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000578 | 1.15x | adaptive MC (inconclusive) |
| PLO5 preflop 1x1 low uniform | 850,668 / 850,668 | 0.0% / 0.0% | 4686.32 +/- 48.47 | 2324.20 +/- 9.49 | 862,000-862,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000619 | 2.02x | adaptive MC |
| PLO5 preflop 2x2 low uniform | 3,402,672 / 3,402,672 | 0.0% / 0.0% | skipped: estimate exceeds 2,000,000 budget | 2477.70 +/- 23.48 | 922,000-922,000; target_reached; 3/3 seeds | 0.000999-0.000999 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO5 flop 1x1 low uniform | 741 / 741 | 0.0% / 0.0% | 5.71 +/- 0.11 | 1626.86 +/- 20.32 | 772,000-772,000; target_reached; 3/3 seeds | 0.000997-0.000998 | 0.000700 | 284.67x | exact |
| PLO5 flop 4x4 low uniform | 11,856 / 11,856 | 0.0% / 0.0% | 13.89 +/- 0.20 | 1961.15 +/- 3.89 | 932,000-932,000; target_reached; 3/3 seeds | 0.000997-0.000998 | 0.000461 | 141.14x | exact |
| PLO5 flop 12x12 low uniform | 106,704 / 106,704 | 0.0% / 0.0% | 134.07 +/- 9.03 | 2058.60 +/- 34.62 | 952,000-952,000; target_reached; 3/3 seeds | 0.000995-0.000995 | 0.001123 | 15.35x | exact |
| PLO5 flop 40x40 low uniform | 1,185,600 / 1,185,600 | 0.0% / 0.0% | 1323.12 +/- 40.62 | 2098.34 +/- 22.17 | 952,000-952,000; target_reached; 3/3 seeds | 0.000999-0.000999 | 0.000578 | 1.59x | exact |
| PLO5 turn 1x1 low uniform | 38 / 38 | 0.0% / 0.0% | 0.80 +/- 0.02 | 1048.73 +/- 43.85 | 592,000-592,000; target_reached; 3/3 seeds | 0.000998-0.000999 | 0.000642 | 1316.01x | exact |
| PLO5 turn 16x16 low uniform | 9,728 / 9,728 | 0.0% / 0.0% | 11.74 +/- 0.67 | 1019.47 +/- 23.23 | 542,000-542,000; target_reached; 3/3 seeds | 0.000995-0.000996 | 0.000701 | 86.86x | exact |
| PLO5 turn 51x51 low uniform | 98,838 / 98,838 | 0.0% / 0.0% | 127.71 +/- 13.54 | 1032.89 +/- 23.40 | 522,000-522,000; target_reached; 3/3 seeds | 0.000996-0.001000 | 0.001177 | 8.09x | exact |
| PLO5 turn 178x178 low uniform | 1,203,992 / 1,203,992 | 0.0% / 0.0% | 1364.45 +/- 32.61 | 898.63 +/- 31.64 | 462,000-462,000; target_reached; 3/3 seeds | 0.000994-0.000997 | 0.000460 | 1.52x | adaptive MC |
| PLO5 river 1x1 low uniform | 1 / 1 | 0.0% / 0.0% | 0.53 +/- 0.04 | 3.80 +/- 0.57 | 2,000-2,000; target_reached; 3/3 seeds | 0.000000-0.000000 | 0.000000 | 7.20x | exact |
| PLO5 river 100x100 low uniform | 10,000 / 10,000 | 0.0% / 0.0% | 16.77 +/- 0.80 | 1414.40 +/- 18.00 | 912,000-912,000; target_reached; 3/3 seeds | 0.000999-0.001000 | 0.000721 | 84.32x | exact |
| PLO5 river 316x316 low uniform | 99,856 / 99,856 | 0.0% / 0.0% | 157.52 +/- 2.38 | 1259.60 +/- 13.86 | 802,000-802,000; target_reached; 3/3 seeds | 0.000998-0.001000 | 0.000957 | 8.00x | exact |
| PLO5 river 1095x1095 low uniform | 1,199,025 / 1,199,025 | 0.0% / 0.0% | 1772.92 +/- 43.41 | 1505.23 +/- 35.02 | 942,000-942,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000860 | 1.18x | adaptive MC |
| PLO4 river 706x706 low uniform | 498,436 / 498,436 | 0.0% / 0.0% | 544.22 +/- 12.42 | 1079.93 +/- 31.39 | 952,000-952,000; target_reached; 5/5 seeds | 0.000995-0.000996 | 0.001457 | 1.98x | exact |
| PLO4 river 900x900 low uniform | 810,000 / 810,000 | 0.0% / 0.0% | 933.62 +/- 49.52 | 1079.25 +/- 42.73 | 922,000-922,000; target_reached; 9/9 seeds | 0.000998-0.000999 | 0.000588 | 1.16x | exact (inconclusive) |
| PLO4 river 1017x1017 low uniform | 1,034,289 / 1,034,289 | 0.0% / 0.0% | 1184.23 +/- 36.09 | 1103.10 +/- 27.35 | 922,000-922,000; target_reached; 9/9 seeds | 0.000996-0.000996 | 0.000580 | 1.07x | adaptive MC (inconclusive) |
| PLO4 river 1064x1064 low uniform | 1,132,096 / 1,132,096 | 0.0% / 0.0% | 1305.55 +/- 21.49 | 1045.32 +/- 36.28 | 922,000-922,000; target_reached; 9/9 seeds | 0.000996-0.000997 | 0.000874 | 1.25x | adaptive MC |
| PLO4 river 1083x1083 low uniform | 1,172,889 / 1,172,889 | 0.0% / 0.0% | 1434.62 +/- 41.52 | 1070.71 +/- 23.69 | 922,000-922,000; target_reached; 5/5 seeds | 0.000996-0.000996 | 0.001071 | 1.34x | adaptive MC |
| PLO5 river 706x706 low uniform | 498,436 / 498,436 | 0.0% / 0.0% | 767.52 +/- 11.46 | 1521.97 +/- 11.68 | 942,000-942,000; target_reached; 5/5 seeds | 0.000997-0.000997 | 0.001701 | 1.98x | exact |
| PLO5 river 900x900 low uniform | 810,000 / 810,000 | 0.0% / 0.0% | 1201.70 +/- 26.50 | 1524.82 +/- 13.61 | 942,000-942,000; target_reached; 5/5 seeds | 0.000997-0.000998 | 0.001614 | 1.27x | exact |
| PLO5 river 1017x1017 low uniform | 1,034,289 / 1,034,289 | 0.0% / 0.0% | 1592.60 +/- 61.38 | 1540.14 +/- 21.11 | 942,000-942,000; target_reached; 9/9 seeds | 0.000998-0.000998 | 0.001435 | 1.03x | adaptive MC (inconclusive) |
| PLO5 river 1064x1064 low uniform | 1,132,096 / 1,132,096 | 0.0% / 0.0% | 1808.26 +/- 36.25 | 1569.77 +/- 26.08 | 942,000-942,000; target_reached; 9/9 seeds | 0.000997-0.000998 | 0.001543 | 1.15x | adaptive MC |
| PLO5 river 1083x1083 low uniform | 1,172,889 / 1,172,889 | 0.0% / 0.0% | 1853.26 +/- 15.01 | 1522.36 +/- 36.98 | 942,000-942,000; target_reached; 9/9 seeds | 0.000997-0.000997 | 0.000727 | 1.22x | adaptive MC |
| PLO4 preflop 2x2 moderate nonuniform | 4,344,032 / 3,258,024 | 25.0% / 9.9% | skipped: estimate exceeds 2,000,000 budget | 1999.23 +/- 38.03 | 912,000-912,000; target_reached; 3/3 seeds | 0.000995-0.000996 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO4 preflop 2x2 high uniform | 4,344,032 / 1,086,008 | 75.0% / 75.0% | skipped: estimate exceeds 2,000,000 budget | 2846.36 +/- 3.42 | 952,000-952,000; target_reached; 3/3 seeds | 0.000996-0.000997 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO4 preflop 1x2 low nonuniform | 2,172,016 / 2,172,016 | 0.0% / 0.0% | skipped: estimate exceeds 2,000,000 budget | 2055.26 +/- 4.28 | 962,000-962,000; target_reached; 3/3 seeds | 0.000996-0.000996 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO4 flop 11x11 moderate nonuniform | 99,220 / 55,760 | 43.8% / 41.7% | 51.69 +/- 1.90 | 1817.35 +/- 11.74 | 942,000-942,000; target_reached; 3/3 seeds | 0.000996-0.000997 | 0.000411 | 35.16x | exact |
| PLO4 flop 11x11 high uniform | 99,220 / 46,740 | 52.9% / 52.9% | 40.59 +/- 1.67 | 2088.66 +/- 67.32 | 962,000-962,000; target_reached; 3/3 seeds | 0.000997-0.000998 | 0.001169 | 51.46x | exact |
| PLO4 flop 1x122 low nonuniform | 100,040 / 100,040 | 0.0% / 0.0% | 69.00 +/- 1.92 | 1568.83 +/- 9.70 | 962,000-962,000; target_reached; 3/3 seeds | 0.000997-0.000997 | 0.000446 | 22.74x | exact |
| PLO4 turn 50x50 moderate nonuniform | 100,000 / 57,200 | 42.8% / 42.8% | 43.28 +/- 0.91 | 1547.36 +/- 9.28 | 902,000-902,000; target_reached; 3/3 seeds | 0.000995-0.000995 | 0.001001 | 35.75x | exact |
| PLO4 turn 50x50 high uniform | 100,000 / 37,960 | 62.0% / 62.0% | 45.99 +/- 2.69 | 1975.01 +/- 27.25 | 952,000-962,000; target_reached; 3/3 seeds | 0.000995-0.001000 | 0.000910 | 42.95x | exact |
| PLO4 turn 1x2500 low nonuniform | 100,000 / 100,000 | 0.0% / 0.0% | 72.04 +/- 0.66 | 1097.74 +/- 26.42 | 742,000-742,000; target_reached; 3/3 seeds | 0.000998-0.000999 | 0.000903 | 15.24x | exact |
| PLO4 river 316x316 moderate nonuniform | 99,856 / 56,524 | 43.4% / 43.5% | 90.89 +/- 1.78 | 1360.76 +/- 45.74 | 952,000-952,000; target_reached; 3/3 seeds | 0.001000-0.001000 | 0.000875 | 14.97x | exact |
| PLO4 river 316x316 high uniform | 99,856 / 37,423 | 62.5% / 62.5% | 78.96 +/- 1.99 | 1625.07 +/- 19.34 | 962,000-962,000; target_reached; 3/3 seeds | 0.000997-0.000997 | 0.000742 | 20.58x | exact |
| PLO4 river 1x100000 low nonuniform | 100,000 / 100,000 | 0.0% / 0.0% | 148.87 +/- 2.27 | 302.24 +/- 10.34 | 222,000-232,000; target_reached; 3/3 seeds | 0.000980-0.000997 | 0.000765 | 2.03x | exact |
| PLO5 preflop 2x2 moderate nonuniform | 3,402,672 / 850,668 | 75.0% / 90.1% | skipped: estimate exceeds 2,000,000 budget | 4884.00 +/- 2.16 | 928,313-931,212; target_reached; 3/3 seeds | 0.000995-0.000997 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO5 preflop 2x2 high uniform | 3,402,672 / 850,668 | 75.0% / 75.0% | skipped: estimate exceeds 2,000,000 budget | 3599.20 +/- 59.61 | 962,000-962,000; target_reached; 3/3 seeds | 0.000996-0.000996 | n/a (exact skipped) | - | not compared (skipped or incomplete) |
| PLO5 preflop 1x2 low nonuniform | 1,701,336 / 1,701,336 | 0.0% / 0.0% | 4864.19 +/- 42.72 | 2367.50 +/- 7.06 | 872,000-872,000; target_reached; 3/3 seeds | 0.000996-0.000996 | 0.000684 | 2.05x | adaptive MC |
| PLO5 flop 12x12 moderate nonuniform | 106,704 / 37,050 | 65.3% / 63.2% | 53.79 +/- 0.88 | 2381.09 +/- 1.73 | 872,000-872,000; target_reached; 3/3 seeds | 0.000997-0.000997 | 0.001160 | 44.26x | exact |
| PLO5 flop 12x12 high uniform | 106,704 / 22,230 | 79.2% / 79.2% | 33.43 +/- 0.19 | 3183.28 +/- 2.48 | 952,000-952,000; target_reached; 3/3 seeds | 0.001000-0.001000 | 0.000379 | 95.22x | exact |
| PLO5 flop 1x135 low nonuniform | 100,035 / 100,035 | 0.0% / 0.0% | 113.91 +/- 1.29 | 1890.08 +/- 29.50 | 872,000-872,000; target_reached; 3/3 seeds | 0.000996-0.000997 | 0.000673 | 16.59x | exact |
| PLO5 turn 51x51 moderate nonuniform | 98,838 / 39,368 | 60.2% / 60.3% | 44.00 +/- 0.44 | 2300.28 +/- 59.91 | 922,000-922,000; target_reached; 3/3 seeds | 0.000995-0.000996 | 0.000626 | 52.28x | exact |
| PLO5 turn 51x51 high uniform | 98,838 / 19,456 | 80.3% / 80.3% | 24.13 +/- 1.14 | 3267.13 +/- 2.15 | 962,000-962,000; target_reached; 3/3 seeds | 0.000995-0.000995 | 0.000448 | 135.40x | exact |
| PLO5 turn 1x2632 low nonuniform | 100,016 / 100,016 | 0.0% / 0.0% | 112.34 +/- 0.90 | 1220.67 +/- 45.87 | 642,000-642,000; target_reached; 3/3 seeds | 0.000994-0.000995 | 0.000799 | 10.87x | exact |
| PLO5 river 316x316 moderate nonuniform | 99,856 / 39,864 | 60.1% / 60.0% | 94.74 +/- 1.18 | 2086.36 +/- 22.95 | 952,000-952,000; target_reached; 3/3 seeds | 0.000998-0.000998 | 0.001108 | 22.02x | exact |
| PLO5 river 316x316 high uniform | 99,856 / 19,278 | 80.7% / 80.7% | 73.39 +/- 3.53 | 3038.55 +/- 15.80 | 962,000-962,000; target_reached; 3/3 seeds | 0.000998-0.000998 | 0.001247 | 41.40x | exact |
| PLO5 river 1x100000 low nonuniform | 100,000 / 100,000 | 0.0% / 0.0% | 188.01 +/- 1.88 | 1155.74 +/- 10.40 | 672,000-672,000; target_reached; 3/3 seeds | 0.000998-0.001000 | 0.000418 | 6.15x | exact |
| PLO4 flop 8x8x8 high nonuniform | 340,992 / 18,648 | 94.5% / 96.3% | 40.28 +/- 1.42 | 9971.18 +/- 11.83 | 735,365-736,110; attempt_limit; 3/3 seeds | 0.001115-0.001116 | 0.001351 | 247.57x | exact |
| PLO4 river 30x30x30 low uniform | 27,000 / 27,000 | 0.0% / 0.0% | 46.00 +/- 0.33 | 1535.51 +/- 51.24 | 952,000-952,000; target_reached; 3/3 seeds | 0.000998-0.000998 | 0.001303 | 33.38x | exact |
| PLO5 flop 8x8x8 high nonuniform | 287,232 / 8,415 | 97.1% / 97.8% | 24.09 +/- 0.20 | 9803.87 +/- 31.74 | 443,537-444,324; attempt_limit; 3/3 seeds | 0.001447-0.001448 | 0.001492 | 407.05x | exact |
| PLO5 river 30x30x30 low uniform | 27,000 / 27,000 | 0.0% / 0.0% | 60.68 +/- 0.99 | 1934.57 +/- 56.36 | 882,000-882,000; target_reached; 3/3 seeds | 0.000995-0.000995 | 0.000418 | 31.88x | exact |
| PLO4 turn 112x112 low uniform | 501,760 / 501,760 | 0.0% / 0.0% | 349.20 +/- 17.71 | 577.06 +/- 4.72 | 422,000-422,000; target_reached; 5/5 seeds | 0.000991-0.000995 | 0.000569 | 1.65x | exact |
| PLO4 turn 142x142 low uniform | 806,560 / 806,560 | 0.0% / 0.0% | 530.27 +/- 2.29 | 714.64 +/- 18.05 | 522,000-522,000; target_reached; 5/5 seeds | 0.000994-0.000996 | 0.000552 | 1.35x | exact |
| PLO4 turn 161x161 low uniform | 1,036,840 / 1,036,840 | 0.0% / 0.0% | 711.15 +/- 26.14 | 718.11 +/- 24.45 | 492,000-502,000; target_reached; 9/9 seeds | 0.000990-0.000999 | 0.001208 | 1.01x | exact (inconclusive) |
| PLO4 turn 168x168 low uniform | 1,128,960 / 1,128,960 | 0.0% / 0.0% | 763.04 +/- 17.94 | 681.02 +/- 19.93 | 492,000-492,000; target_reached; 9/9 seeds | 0.000991-0.000994 | 0.000814 | 1.12x | adaptive MC |
| PLO4 turn 171x171 low uniform | 1,169,640 / 1,169,640 | 0.0% / 0.0% | 796.04 +/- 17.50 | 689.56 +/- 27.67 | 492,000-492,000; target_reached; 9/9 seeds | 0.000995-0.000999 | 0.000734 | 1.15x | adaptive MC |
| PLO5 turn 114x114 low uniform | 493,848 / 493,848 | 0.0% / 0.0% | 530.98 +/- 4.05 | 1003.68 +/- 1.50 | 522,000-522,000; target_reached; 5/5 seeds | 0.000994-0.000997 | 0.001068 | 1.89x | exact |
| PLO5 turn 146x146 low uniform | 810,008 / 810,008 | 0.0% / 0.0% | 911.30 +/- 14.08 | 971.69 +/- 18.04 | 482,000-492,000; target_reached; 9/9 seeds | 0.000990-0.000999 | 0.000651 | 1.07x | exact (inconclusive) |
| PLO5 turn 165x165 low uniform | 1,034,550 / 1,034,550 | 0.0% / 0.0% | 1160.84 +/- 11.85 | 915.40 +/- 14.18 | 462,000-472,000; target_reached; 5/5 seeds | 0.000990-0.001000 | 0.000784 | 1.27x | adaptive MC |
| PLO5 turn 173x173 low uniform | 1,137,302 / 1,137,302 | 0.0% / 0.0% | 1291.58 +/- 21.08 | 875.78 +/- 5.87 | 462,000-462,000; target_reached; 5/5 seeds | 0.000995-0.000999 | 0.001003 | 1.47x | adaptive MC |
| PLO5 turn 176x176 low uniform | 1,177,088 / 1,177,088 | 0.0% / 0.0% | 1344.96 +/- 18.52 | 907.12 +/- 29.50 | 462,000-462,000; target_reached; 5/5 seeds | 0.000992-0.000997 | 0.000862 | 1.48x | adaptive MC |

## Interpretation

Among 69 scenarios with timed exact results, 69 were fully timed; 7 exact scenarios were skipped by the 2,000,000 estimate budget. Clear/inconclusive winners use the timing rule above.

The 308 MC seed runs used 2,000-962,000 simulations; 302 stopped at target and 6 hit the attempt limit. The maximum reported MOE was 0.001448. 23/308 MC runs with exact references had a maximum absolute equity difference above that run's reported maximum MOE; the largest difference was 0.001701. The plug-in MOE is approximate, and collision-heavy workloads can fail to reach the target within the attempt budget.

## Initial method=auto policy recommendation

Keep the current exact limit unchanged for a first implementation: choose adaptive MC preflop; otherwise choose exact when the conservative estimate is at most 100,000 and MC above that cap. This rule is intentionally conservative and follows the current production guard. If exact capacity is separately expanded later, use the measured per-street crossover bands as provisional ceilings: PLO4 turn about 1.04M and river about 0.98M; PLO5 turn about 0.89M and river about 1.01M; flop exact remained faster through the tested 1.18M. Adjust the exact estimate for compatible hand tuples before applying those ceilings: collision-heavy cases strongly favored exact despite an estimate near 100,000. Near a crossover, keep the decision marked borderline and prefer exact only when within its permitted budget.

Exact remained faster beyond the production 100,000 limit in balanced flop cases through about 1.18M and in turn cases through about 0.49-0.81M, with some higher turn points inconclusive. River exact remained faster at about 0.50M for PLO4 and 0.81M for PLO5. The experiment does not justify raising production limits by itself.

## Initial method=auto policy recommendation

Keep the current exact limit unchanged for a first implementation: choose adaptive MC preflop; otherwise choose exact when the conservative estimate is at most 100,000 and MC above that cap. This rule is intentionally conservative and follows the current production guard. If exact capacity is separately expanded later, use the measured per-street crossover bands as provisional ceilings: PLO4 turn about 1.04M and river about 0.98M; PLO5 turn about 0.89M and river about 1.01M; flop exact remained faster through the tested 1.18M. Adjust the exact estimate for compatible hand tuples before applying those ceilings: collision-heavy cases strongly favored exact despite an estimate near 100,000. Near a crossover, keep the decision marked borderline and prefer exact only when within its permitted budget.

Exact remained faster beyond the production 100,000 limit in balanced flop cases through about 1.18M and in turn cases through about 0.49-0.81M, with some higher turn points inconclusive. River exact remained faster at about 0.50M for PLO4 and 0.81M for PLO5. The experiment does not justify raising production limits by itself.
