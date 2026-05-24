# Prompt for GPT Pro / Deep Research

I'm an undergraduate doing summer research in a wet-lab that studies the **mouse choroid plexus** (CSF-producing tissue in the brain ventricles). My PI is a new assistant professor; her postdoctoral training was in the Lehtinen lab (Boston Children's / Harvard). She gave me three sample fluorescence microscopy images — single-channel max-projected confocal, 20× tile-scanned, ~655 µm field, 0.21 µm/px, 16-bit. The signal channel is a microglia marker (most likely Iba1 or CX3CR1-GFP). One sample is WT, two are HET of an undisclosed allele.

**Biological question (as she framed it in our meeting):** detect every microglia in each image, classify each on a 0 → 5 ordinal scale of ramification (0 = round soma, no processes; 5 = highly branched with many processes), then compare the distribution between WT and HET. She acknowledged that even she has trouble defining the morphology boundary by eye, especially in dense regions.

I've attached a snapshot of where I'm stuck. Please read it.

## What I've tried

| Pipeline | WT_2 | HET_1 | HET_3 | Notes |
|---|---|---|---|---|
| Cellpose cyto3 (default) | 1099 | 1211 | 1293 | Clean somata, misses faint-soma ramified cells (the most ramified end of the population) |
| Cellpose cyto3 (cellprob_threshold = −2.0) | 1800 | 1848 | 1772 | Catches more, masks grow larger |
| v4 = above + distance-based supplementary detector | 2752 | 2472 | 2330 | Tries to catch missed ramified cells; ~40–60% of supplementary additions are false positives in dense regions |
| Fiji classical (Otsu + watershed + size 30–3000 px) | 9272 | 7780 | 6644 | Over-fragments, especially along processes |

For each cell I have area, circularity (4πA/P²), eccentricity, solidity, and a source flag (Cellpose vs supplementary). CSV is in `03_data/`.

## The fundamental problems I cannot solve alone

1. **Visual ambiguity.** I cannot distinguish a real cell from a process-fragment or process-convergence node in dense regions by eye. I lack the biological intuition that my PI has. The textbook "compact soma + radiating processes" is rarely visible cleanly — most structures in this tissue are tangled.

2. **Detection failure modes I can identify:**
   - Cellpose systematically misses cells whose soma is faint relative to their processes (i.e., the most-ramified cells — exactly the population that matters for the morphology question)
   - My distance-based supplementary detector helps in sparse regions but adds garbage in dense regions
   - Fiji classical over-fragments because watershed seeds on local maxima of the distance transform, which fire repeatedly along thick processes

3. **The morphology metric is unstable across pipelines.** A "process count" depends on how the soma boundary is defined: tight soma → high process count, loose soma → low process count. With the v4 supplementary detector that includes processes inside the mask, process counts collapse toward zero. There's no objective ground truth.

4. **No literature precedent.** I searched. As of mid-2026, every published choroid-plexus immune cell paper still uses manual counting in ImageJ + DAPI normalization in QuPath. No one has published an automated microglia pipeline specifically for choroid plexus tissue. The closest off-the-shelf solution, MicrogliaMorphology (Ciernia lab, eNeuro 2024), is threshold-based and almost certainly suffers the same over-fragmentation problem as my Fiji baseline on this tissue.

## What I'm asking

Please give me your honest, technical, actionable take on:

1. **Detection pipeline.** Given the data and my failure modes, what's the best computational approach you'd recommend that I haven't tried? Be specific. If it's "fine-tune Cellpose on hand-annotations," say how many annotations and which loss. If it's "use μSAM / SAM 2 with point prompts seeded from local maxima," tell me which model. If it's "give up on per-cell and go patch-level," argue why.

2. **The soma vs. process boundary.** Is there a principled way to fix the "process count depends on soma definition" instability? E.g., compute morphology features on the full cell territory (Cellpose + dilation to nearest process tip via watershed) rather than on the Cellpose mask alone?

3. **MicrogliaMorphology / MicrogliaMorphologyR by the Ciernia lab.** Worth trying despite the threshold-based concern? Or skip and go straight to a custom Cellpose fine-tune?

4. **A defensible analysis strategy if detection cannot be made perfect.** Suppose I accept that 10–20 % of cells are mis-detected. Is there a way to design the WT vs HET comparison so the bias is symmetric and the comparison is still valid? E.g., focus on cell density rather than per-cell morphology, or use only the brightest 50 % of cells where detection is reliable.

5. **Single most useful next experiment** — if I had one week of compute and could only do one new thing, what should it be?

6. **What I might be missing about choroid plexus biology** that would change my framing. Published references that would help me think about Kolmer cells vs stromal microglia in this tissue, the difference in expected morphology between the two populations, and what genotype effects in this tissue have been reported.

## What I do NOT need

- Encouragement or "this is great progress" — I want criticism and concrete next steps
- A summary of what I already wrote — I read it
- Generic computer-vision advice that doesn't engage with this specific tissue's structure
- Reminders to talk to my PI — I'm doing that; this is for a parallel external opinion

## Files attached

- `01_samples/` — three full images (auto-contrasted previews)
- `02_crops/` — native-resolution crops with three detection overlays (red Cellpose / green supplementary / yellow Fiji), paired with unmarked originals and centroid-only dot versions
- `03_data/all_cells_v4.csv` — 7554 cells with per-cell metrics
- `04_writeups/` — chronological writeups of what I tried and what I saw

Thanks. Be direct.
