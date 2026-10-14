# Evaluation Results

**Total runs:** 252
**Total tokens:** 4,206,559
**Total cost:** $0.6979
**Safety violations:** 0

## Summary Table

| Scenario | Target | Planner | Memory | Runs | Success | False-found | Mean steps | Median steps | Mean path (m) | LLM calls | Tokens | Cost ($) | Safety |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absent_target | red cube | heuristic | off | 2 | 100.0% | 0.0% | 16.0 | 16 | 4.04 | 0 | 0 | 0.0000 | 0 |
| absent_target | red cube | heuristic | on | 2 | 100.0% | 0.0% | 16.0 | 16 | 4.04 | 0 | 0 | 0.0000 | 0 |
| absent_target | red cube | llm | off | 2 | 100.0% | 0.0% | 32.5 | 32 | 8.82 | 20 | 39653 | 0.0065 | 0 |
| absent_target | red cube | llm | on | 2 | 50.0% | 0.0% | 24.5 | 24 | 9.58 | 29 | 59122 | 0.0098 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 20 | 100.0% | 0.0% | 6.0 | 5 | 2.06 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 20 | 100.0% | 0.0% | 6.0 | 5 | 2.04 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 20 | 100.0% | 0.0% | 6.5 | 5 | 2.25 | 40 | 79558 | 0.0131 | 0 |
| apartment_v1 | blue sphere | llm | on | 20 | 95.0% | 0.0% | 8.2 | 5 | 2.89 | 73 | 162707 | 0.0269 | 0 |
| apartment_v1 | red cube | heuristic | off | 20 | 60.0% | 0.0% | 84.8 | 86 | 35.74 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 20 | 55.0% | 0.0% | 84.0 | 87 | 35.40 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | llm | off | 20 | 40.0% | 0.0% | 38.4 | 29 | 16.87 | 625 | 1328106 | 0.2212 | 0 |
| apartment_v1 | red cube | llm | on | 20 | 25.0% | 0.0% | 44.6 | 32 | 19.26 | 598 | 1388145 | 0.2306 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 20 | 25.0% | 0.0% | 44.6 | 51 | 17.94 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 20 | 15.0% | 0.0% | 45.5 | 51 | 18.33 | 0 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 20 | 85.0% | 0.0% | 14.9 | 13 | 6.60 | 177 | 360506 | 0.0594 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 20 | 70.0% | 0.0% | 25.9 | 24 | 11.07 | 349 | 788762 | 0.1304 | 0 |
| living_room_v1 | blue sphere | heuristic | off | 1 | 0.0% | 0.0% | 21.0 | 21 | 5.15 | 0 | 0 | 0.0000 | 0 |
| living_room_v1 | blue sphere | heuristic | on | 1 | 0.0% | 0.0% | 21.0 | 21 | 5.15 | 0 | 0 | 0.0000 | 0 |
| living_room_v1 | red cube | heuristic | off | 1 | 100.0% | 0.0% | 6.0 | 6 | 1.97 | 0 | 0 | 0.0000 | 0 |
| living_room_v1 | red cube | heuristic | on | 1 | 100.0% | 0.0% | 6.0 | 6 | 1.97 | 0 | 0 | 0.0000 | 0 |


## Plots

### Memory Comparison

![memory_comparison](plots/memory_comparison.png)

### Planner Comparison

![planner_comparison](plots/planner_comparison.png)

### Success Rates

![success_rates](plots/success_rates.png)


## Per-Run Details

| Scenario | Target | Planner | Memory | Seed | Outcome | Steps | Path (m) | LLM calls | Cost ($) | Safety |
|---|---|---|---|---|---|---|---|---|---|---|
| absent_target | red cube | heuristic | off | 1 | not_found | 21 | 5.15 | 0 | 0.0000 | 0 |
| absent_target | red cube | heuristic | off | 2 | not_found | 11 | 2.94 | 0 | 0.0000 | 0 |
| absent_target | red cube | heuristic | on | 1 | not_found | 21 | 5.15 | 0 | 0.0000 | 0 |
| absent_target | red cube | heuristic | on | 2 | not_found | 11 | 2.94 | 0 | 0.0000 | 0 |
| absent_target | red cube | llm | off | 1 | not_found | 54 | 14.71 | 10 | 0.0032 | 0 |
| absent_target | red cube | llm | off | 2 | not_found | 11 | 2.93 | 10 | 0.0032 | 0 |
| absent_target | red cube | llm | on | 1 | error | 38 | 16.02 | 19 | 0.0065 | 0 |
| absent_target | red cube | llm | on | 2 | not_found | 11 | 3.13 | 10 | 0.0033 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 1 | found | 5 | 1.53 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 2 | found | 8 | 2.95 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 3 | found | 8 | 2.95 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 4 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 5 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 6 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 7 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 8 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 9 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 10 | found | 5 | 1.69 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 11 | found | 9 | 3.45 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 12 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 13 | found | 6 | 2.00 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 14 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 15 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 16 | found | 13 | 5.99 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 17 | found | 5 | 1.53 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 18 | found | 6 | 1.92 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 19 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | off | 20 | found | 9 | 3.30 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 1 | found | 5 | 1.53 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 2 | found | 8 | 2.95 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 3 | found | 8 | 2.95 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 4 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 5 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 6 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 7 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 8 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 9 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 10 | found | 5 | 1.69 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 11 | found | 9 | 3.06 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 12 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 13 | found | 6 | 2.00 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 14 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 15 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 16 | found | 13 | 5.99 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 17 | found | 5 | 1.53 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 18 | found | 6 | 1.92 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 19 | found | 5 | 1.52 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | heuristic | on | 20 | found | 9 | 3.30 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 1 | found | 5 | 1.53 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 2 | found | 8 | 2.51 | 5 | 0.0016 | 0 |
| apartment_v1 | blue sphere | llm | off | 3 | found | 8 | 2.92 | 4 | 0.0013 | 0 |
| apartment_v1 | blue sphere | llm | off | 4 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 5 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 6 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 7 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 8 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 9 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 10 | found | 5 | 1.69 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 11 | found | 6 | 2.56 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 12 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 13 | found | 6 | 2.00 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 14 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | off | 15 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 16 | found | 25 | 11.57 | 13 | 0.0047 | 0 |
| apartment_v1 | blue sphere | llm | off | 17 | found | 5 | 1.53 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 18 | found | 9 | 2.40 | 4 | 0.0013 | 0 |
| apartment_v1 | blue sphere | llm | off | 19 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | off | 20 | found | 7 | 2.46 | 4 | 0.0013 | 0 |
| apartment_v1 | blue sphere | llm | on | 1 | found | 5 | 1.53 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 2 | found | 7 | 2.46 | 4 | 0.0014 | 0 |
| apartment_v1 | blue sphere | llm | on | 3 | found | 8 | 2.83 | 4 | 0.0014 | 0 |
| apartment_v1 | blue sphere | llm | on | 4 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | on | 5 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 6 | found | 4 | 1.19 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | on | 7 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 8 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 9 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | on | 10 | found | 5 | 1.69 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | on | 11 | found | 10 | 3.88 | 4 | 0.0014 | 0 |
| apartment_v1 | blue sphere | llm | on | 12 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 13 | found | 6 | 2.00 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 14 | found | 4 | 1.20 | 0 | 0.0000 | 0 |
| apartment_v1 | blue sphere | llm | on | 15 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 16 | found | 15 | 5.98 | 7 | 0.0026 | 0 |
| apartment_v1 | blue sphere | llm | on | 17 | found | 5 | 1.53 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 18 | found | 8 | 2.91 | 4 | 0.0014 | 0 |
| apartment_v1 | blue sphere | llm | on | 19 | found | 5 | 1.52 | 1 | 0.0003 | 0 |
| apartment_v1 | blue sphere | llm | on | 20 | not_found | 49 | 19.08 | 41 | 0.0157 | 0 |
| apartment_v1 | red cube | heuristic | off | 1 | not_found | 31 | 12.62 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 2 | not_found | 31 | 12.71 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 3 | found | 86 | 36.78 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 4 | found | 86 | 36.77 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 5 | found | 86 | 35.91 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 6 | not_found | 101 | 40.65 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 7 | not_found | 101 | 42.52 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 8 | found | 86 | 35.93 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 9 | not_found | 101 | 43.58 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 10 | found | 56 | 23.18 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 11 | not_found | 121 | 51.41 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 12 | found | 76 | 31.43 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 13 | not_found | 121 | 55.38 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 14 | found | 87 | 36.45 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 15 | found | 75 | 31.10 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 16 | found | 83 | 36.83 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 17 | found | 85 | 35.53 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 18 | not_found | 121 | 47.28 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 19 | found | 89 | 37.97 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | off | 20 | found | 74 | 30.78 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 1 | not_found | 31 | 12.62 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 2 | not_found | 31 | 12.62 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 3 | found | 87 | 36.45 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 4 | not_found | 101 | 42.53 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 5 | found | 85 | 36.49 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 6 | not_found | 101 | 43.02 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 7 | not_found | 101 | 43.61 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 8 | found | 90 | 39.18 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 9 | not_found | 101 | 42.70 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 10 | found | 60 | 25.21 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 11 | not_found | 121 | 49.35 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 12 | found | 76 | 31.43 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 13 | not_found | 121 | 51.28 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 14 | found | 87 | 37.16 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 15 | found | 75 | 31.12 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 16 | found | 56 | 23.20 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 17 | found | 85 | 36.48 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 18 | found | 87 | 36.46 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 19 | not_found | 121 | 51.30 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | heuristic | on | 20 | found | 63 | 25.68 | 0 | 0.0000 | 0 |
| apartment_v1 | red cube | llm | off | 1 | error | 27 | 13.02 | 24 | 0.0081 | 0 |
| apartment_v1 | red cube | llm | off | 2 | not_found | 31 | 12.87 | 24 | 0.0081 | 0 |
| apartment_v1 | red cube | llm | off | 3 | error | 19 | 9.72 | 17 | 0.0057 | 0 |
| apartment_v1 | red cube | llm | off | 4 | error | 21 | 10.98 | 18 | 0.0061 | 0 |
| apartment_v1 | red cube | llm | off | 5 | found | 21 | 9.64 | 16 | 0.0053 | 0 |
| apartment_v1 | red cube | llm | off | 6 | error | 45 | 21.62 | 40 | 0.0143 | 0 |
| apartment_v1 | red cube | llm | off | 7 | error | 42 | 14.95 | 38 | 0.0134 | 0 |
| apartment_v1 | red cube | llm | off | 8 | error | 24 | 11.87 | 20 | 0.0069 | 0 |
| apartment_v1 | red cube | llm | off | 9 | error | 21 | 11.31 | 19 | 0.0064 | 0 |
| apartment_v1 | red cube | llm | off | 10 | error | 19 | 9.75 | 17 | 0.0057 | 0 |
| apartment_v1 | red cube | llm | off | 11 | error | 103 | 40.47 | 94 | 0.0341 | 0 |
| apartment_v1 | red cube | llm | off | 12 | found | 12 | 4.84 | 9 | 0.0029 | 0 |
| apartment_v1 | red cube | llm | off | 13 | error | 94 | 41.53 | 78 | 0.0288 | 0 |
| apartment_v1 | red cube | llm | off | 14 | error | 62 | 29.63 | 52 | 0.0190 | 0 |
| apartment_v1 | red cube | llm | off | 15 | found | 13 | 5.15 | 9 | 0.0029 | 0 |
| apartment_v1 | red cube | llm | off | 16 | found | 13 | 5.39 | 6 | 0.0019 | 0 |
| apartment_v1 | red cube | llm | off | 17 | found | 78 | 32.64 | 59 | 0.0217 | 0 |
| apartment_v1 | red cube | llm | off | 18 | found | 46 | 19.67 | 27 | 0.0095 | 0 |
| apartment_v1 | red cube | llm | off | 19 | found | 45 | 20.30 | 34 | 0.0117 | 0 |
| apartment_v1 | red cube | llm | off | 20 | found | 31 | 11.96 | 24 | 0.0084 | 0 |
| apartment_v1 | red cube | llm | on | 1 | error | 23 | 8.26 | 15 | 0.0055 | 0 |
| apartment_v1 | red cube | llm | on | 2 | error | 29 | 12.60 | 26 | 0.0097 | 0 |
| apartment_v1 | red cube | llm | on | 3 | error | 25 | 9.85 | 14 | 0.0051 | 0 |
| apartment_v1 | red cube | llm | on | 4 | error | 30 | 13.98 | 15 | 0.0055 | 0 |
| apartment_v1 | red cube | llm | on | 5 | error | 31 | 15.30 | 25 | 0.0095 | 0 |
| apartment_v1 | red cube | llm | on | 6 | found | 33 | 14.32 | 27 | 0.0099 | 0 |
| apartment_v1 | red cube | llm | on | 7 | error | 26 | 10.29 | 15 | 0.0054 | 0 |
| apartment_v1 | red cube | llm | on | 8 | error | 52 | 23.47 | 33 | 0.0128 | 0 |
| apartment_v1 | red cube | llm | on | 9 | error | 18 | 9.74 | 14 | 0.0052 | 0 |
| apartment_v1 | red cube | llm | on | 10 | error | 23 | 11.36 | 19 | 0.0072 | 0 |
| apartment_v1 | red cube | llm | on | 11 | found | 52 | 19.49 | 33 | 0.0125 | 0 |
| apartment_v1 | red cube | llm | on | 12 | error | 71 | 32.93 | 31 | 0.0121 | 0 |
| apartment_v1 | red cube | llm | on | 13 | found | 64 | 23.43 | 53 | 0.0209 | 0 |
| apartment_v1 | red cube | llm | on | 14 | error | 71 | 26.87 | 53 | 0.0210 | 0 |
| apartment_v1 | red cube | llm | on | 15 | error | 62 | 30.25 | 48 | 0.0191 | 0 |
| apartment_v1 | red cube | llm | on | 16 | error | 81 | 35.33 | 57 | 0.0231 | 0 |
| apartment_v1 | red cube | llm | on | 17 | found | 28 | 11.44 | 22 | 0.0079 | 0 |
| apartment_v1 | red cube | llm | on | 18 | error | 77 | 31.90 | 53 | 0.0212 | 0 |
| apartment_v1 | red cube | llm | on | 19 | error | 72 | 33.22 | 28 | 0.0109 | 0 |
| apartment_v1 | red cube | llm | on | 20 | found | 25 | 11.22 | 17 | 0.0062 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 1 | found | 7 | 2.66 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 2 | not_found | 31 | 12.61 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 3 | not_found | 51 | 20.55 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 4 | not_found | 51 | 20.03 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 5 | not_found | 51 | 20.52 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 6 | not_found | 51 | 20.39 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 7 | not_found | 51 | 20.54 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 8 | found | 7 | 2.51 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 9 | found | 49 | 20.43 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 10 | not_found | 51 | 20.05 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 11 | not_found | 51 | 20.68 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 12 | not_found | 51 | 20.04 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 13 | not_found | 51 | 20.40 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 14 | found | 38 | 16.73 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 15 | not_found | 51 | 20.02 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 16 | not_found | 51 | 20.53 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 17 | not_found | 51 | 20.09 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 18 | not_found | 51 | 20.03 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 19 | not_found | 51 | 20.05 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | off | 20 | found | 46 | 19.88 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 1 | found | 7 | 2.66 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 2 | not_found | 31 | 12.62 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 3 | not_found | 51 | 20.51 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 4 | not_found | 51 | 20.00 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 5 | not_found | 51 | 20.50 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 6 | not_found | 51 | 20.52 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 7 | not_found | 51 | 20.05 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 8 | found | 7 | 2.51 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 9 | not_found | 51 | 20.89 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 10 | not_found | 51 | 20.53 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 11 | not_found | 51 | 23.18 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 12 | not_found | 51 | 20.02 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 13 | not_found | 51 | 21.01 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 14 | found | 49 | 20.66 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 15 | not_found | 51 | 20.00 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 16 | not_found | 51 | 20.03 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 17 | not_found | 51 | 20.39 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 18 | not_found | 51 | 20.02 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 19 | not_found | 51 | 20.03 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | heuristic | on | 20 | not_found | 51 | 20.53 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 1 | found | 7 | 2.66 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 2 | error | 21 | 9.06 | 13 | 0.0045 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 3 | found | 13 | 5.04 | 7 | 0.0023 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 4 | error | 23 | 13.88 | 20 | 0.0068 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 5 | found | 9 | 3.92 | 6 | 0.0019 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 6 | found | 9 | 3.65 | 6 | 0.0019 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 7 | found | 10 | 4.24 | 6 | 0.0019 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 8 | found | 7 | 2.78 | 3 | 0.0009 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 9 | found | 9 | 3.31 | 4 | 0.0013 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 10 | found | 18 | 7.28 | 10 | 0.0033 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 11 | not_found | 51 | 23.16 | 35 | 0.0129 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 12 | found | 19 | 8.29 | 10 | 0.0033 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 13 | found | 10 | 4.23 | 6 | 0.0019 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 14 | found | 9 | 3.32 | 4 | 0.0013 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 15 | found | 13 | 5.58 | 7 | 0.0023 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 16 | found | 23 | 11.87 | 12 | 0.0040 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 17 | found | 9 | 3.92 | 6 | 0.0019 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 18 | found | 13 | 5.57 | 7 | 0.0023 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 19 | found | 13 | 5.58 | 7 | 0.0023 | 0 |
| apartment_v1 | yellow cylinder | llm | off | 20 | found | 13 | 4.67 | 8 | 0.0026 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 1 | found | 7 | 2.66 | 0 | 0.0000 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 2 | not_found | 31 | 14.55 | 22 | 0.0083 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 3 | error | 24 | 10.21 | 15 | 0.0054 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 4 | found | 27 | 11.53 | 20 | 0.0073 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 5 | error | 28 | 11.28 | 23 | 0.0087 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 6 | found | 11 | 5.38 | 7 | 0.0025 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 7 | found | 14 | 5.96 | 9 | 0.0032 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 8 | found | 10 | 3.83 | 4 | 0.0014 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 9 | found | 23 | 8.96 | 16 | 0.0059 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 10 | error | 41 | 15.62 | 34 | 0.0133 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 11 | found | 46 | 19.74 | 29 | 0.0110 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 12 | found | 25 | 10.94 | 18 | 0.0067 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 13 | found | 13 | 4.80 | 8 | 0.0029 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 14 | not_found | 51 | 27.30 | 36 | 0.0142 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 15 | not_found | 50 | 18.53 | 35 | 0.0136 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 16 | found | 26 | 10.32 | 16 | 0.0057 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 17 | found | 16 | 5.76 | 10 | 0.0035 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 18 | found | 24 | 9.96 | 13 | 0.0046 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 19 | found | 30 | 14.12 | 21 | 0.0076 | 0 |
| apartment_v1 | yellow cylinder | llm | on | 20 | found | 22 | 9.97 | 13 | 0.0046 | 0 |
| living_room_v1 | blue sphere | heuristic | off | 1 | not_found | 21 | 5.15 | 0 | 0.0000 | 0 |
| living_room_v1 | blue sphere | heuristic | on | 1 | not_found | 21 | 5.15 | 0 | 0.0000 | 0 |
| living_room_v1 | red cube | heuristic | off | 1 | found | 6 | 1.97 | 0 | 0.0000 | 0 |
| living_room_v1 | red cube | heuristic | on | 1 | found | 6 | 1.97 | 0 | 0.0000 | 0 |
