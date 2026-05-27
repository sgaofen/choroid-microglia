# Results (current pipeline output)

3 images: F_WT_2 = WT/control; F_HET_1, F_HET_3 = HET/disease (Alzheimer's
model, replicates). All metrics computed on the FULL image. Aggregate metrics
normalized by foreground (signal) area in mm². Per-segment = each skeleton
branch between junctions/endpoints. Regional = 400-px (~83 µm) tiles, CV taken
across tiles as a spatial-heterogeneity measure.

| metric | WT_2 | HET_1 | HET_3 | HET vs WT |
|---|---|---|---|---|
| mean segment length (µm) | 3.26 | 3.17 | 3.23 | −1.8% |
| median segment length (µm) | 2.57 | 2.46 | 2.50 | −3.5% |
| p90 segment length (µm) | 6.82 | 6.69 | 6.70 | −1.8% |
| CV segment length | 0.861 | 0.865 | 0.886 | +1.7% |
| mean thickness/diameter (µm) | 1.84 | 1.91 | 1.92 | +4.3% |
| CV thickness | 0.396 | 0.433 | 0.583 | +28.3% |
| branch points / mm² | 24725 | 21352 | 23750 | −8.8% |
| endpoints / mm² | 87146 | 94297 | 95255 | +8.8% |
| skeleton length (µm) / mm² | 577798 | 531911 | 569694 | −4.7% |
| ramification (branches / 100µm skeleton) | 4.28 | 4.01 | 4.17 | −4.4% |
| regional CV (skeleton density) | 0.078 | 0.068 | 0.108 | +12.8% |
| regional CV (thickness) | 0.11 | 0.12 | 0.24 | +63.2% |

## Apparent direction (HET vs WT)
- **De-ramification**: fewer branch points (−8.8%), lower ramification index
  (−4.4%), less total skeleton (−4.7%); more endpoints (+8.8%) = more
  fragmented/terminal tips.
- **Processes slightly thicker (+4.3%) and more variable (+28% CV)** — possible
  swelling toward activated/amoeboid.
- **More spatially heterogeneous** (regional CVs up, thickness regional CV
  +63%) — consistent with patchy disease response.
- Segment length itself barely differs (−1.8%).

## Caveats we already know
- n = 1 WT vs 2 HET images — underpowered; no image-level statistics possible.
- Per-cell counting is NOT done (dense/touching). These are aggregate +
  per-segment + regional metrics, deliberately avoiding single-cell segmentation.
- Binarization is a single global threshold (Otsu×0.7); raising it fragments
  processes, lowering it floods background — see PROMPT_for_GPT_Pro.md.
