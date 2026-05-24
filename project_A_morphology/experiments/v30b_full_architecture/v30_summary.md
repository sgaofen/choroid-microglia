# v30 summary — somaness-based cell filtering

Two iterations of the GPT Pro round-3 recipe:
v30a (minimal hard-threshold filter) → v30b (full somaness_score + graph merge).

## Cross-genotype counts and mean endpoints per cell

| image  | v29 cells | v30a cells | v30b cells | v29 mean | v30a mean | v30b mean |
|--------|-----------|------------|------------|----------|-----------|-----------|
| F_WT_2 |    2395   |    2013    |    1556    |   2.22   |   2.45    |   3.33    |
| F_HET_1|    2086   |    1634    |    1380    |   2.39   |   2.67    |   3.37    |
| F_HET_3|    1955   |    1539    |    1324    |   2.62   |   2.82    |   3.62    |

(v29 row counts only include cells that endpoint-attribution reached;
v27 raw soma_cores had ~3000 labels per image.)

## Distribution shift

v30b distribution peaks at 3 endpoints/cell, matching published microglia
process counts. v29 had a huge spike at 0-1 endpoints (process peaks
masquerading as cells).

## Visual verification: 7 cells per image, then 4 region comparisons

* **process peaks correctly rejected**: long-process beads, vessel
  thickenings, junction artifacts.
* **isolated small cells correctly kept** (v30b fix over v30a):
  Kolmer-like compact blobs, comma-shaped cells.
* **dense regions still imperfect** but much closer to truth.
* **vessel-aligned cells**: v30b correctly merges the 3-5 over-split
  process-peaks into 1-2 cells, matches Stephen's visual judgment.

## What v30b changes over v30a

| concern | v30a (hard thresholds) | v30b (graph + score) |
|---------|------------------------|----------------------|
| score | binary (ecc>0.85 OR tube>blob) | weighted multi-component 0..1 |
| small isolated cell | rejected by both rules | kept as low_confidence_soma |
| vessel beads | partly kept | merged via thin-neck test |
| touching cells | indistinguishable | flagged ambiguous_neighbor |

## Known limits (per GPT Pro)

* dense clusters still ambiguous from single-channel 2D max projection
* no vessel marker means vessel-following cells cannot be distinguished
  from one elongated cell with high confidence
* ambiguous_neighbor_pairs flagged but kept (not collapsed to integer)

## Files

```
v30b_full_architecture/
  v30b_run.py                       — pipeline
  v30b_viz_individual.py            — per-cell 7-sample crops
  single_compare.py                 — 4-region 3-way compares
  final_compare.py                  — cross-genotype summary
  F_*_seeds_v30b.json               — per-seed scores + types
  F_*_endpoint_counts_v30b.json     — final endpoints per accepted cell
  F_*_accepted_labels.npy           — accepted soma labels
  compare2x2_*.png                  — orig + v29 + v30a + v30b panels
```
