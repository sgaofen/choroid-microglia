# Prompt for GPT Pro / Deep Research

Round 2 of consultation on the same choroid-plexus microglia morphology project. Your previous advice (round 1) led us to abandon Cellpose entirely and build a pure-algorithm pipeline. The pipeline is now end-to-end deterministic (threshold → erode → watershed → skeleton → boundary-rooted process count). Read `README.md` and then `04_writeups/01_current_pipeline_and_findings.md` for full context.

I have three concrete problems I need your honest take on. Be direct.

## Issue 1 — Over-split is severe (primary concern)

Current watershed step finds 6163 somata in WT_2. Of these, **850 groups (~14 % of cells) have at least 2 somata within 15 px (~3 µm) of each other**. Many of these "two cells" are visibly one biological cell — the watershed split it because the erosion + distance-transform step found two local maxima inside one elongated soma.

See `03_issues/oversplit_examples_6groups.png`. Each panel shows 6 cases where 2–4 colors are tagging what visually appears to be one cell.

Questions:

- Is there a principled way to merge over-split somata? Distance threshold is the obvious heuristic but feels brittle.
- Should I be using a different seed-finding strategy than `peak_local_max` on the erosion's distance transform? `min_distance=8` was an arbitrary pick.
- Is there a known method that does "soma detection without over-split" specifically on dense neuronal-like tissue?
- Would it be more principled to detect soma centers via Hessian-based blob detection (LoG / DoG) followed by single-seeded growing, rather than erosion + DT + local maxima?

## Issue 2 — Normalization choice flips the per-cell morphology result

Three different normalization + threshold choices give three different directions for WT vs HET per-cell process count:

| Approach | WT mean | HET mean | Direction |
|---|---|---|---|
| Original Otsu (too strict, gives gappy skeleton) | 1.68 | 1.08–1.17 | WT > HET (×1.5) |
| Standard pct[1, 99.5] + Otsu × 0.7 (current) | 1.98 | 1.82–2.00 | ≈ |
| Shipley pct[20, 90] + Otsu × 0.7 | 2.45 | 3.57–4.94 | HET > WT (×2) |

See `03_issues/normalization_changes_pattern_direction.png` for visualization of normalization difference.

The image is the same. The biology is the same. But the answer flips depending on a methodology knob.

Questions:

- Is there a normalization-invariant per-cell metric? Something that doesn't depend on where we set the brightness range?
- For cross-image comparison in this kind of variable-staining context, should I be normalizing per-image (current) or to a fixed reference (e.g., common 99th percentile across all 3)?
- Is the right answer that **per-cell morphology is just not measurable without ground truth**, and we should report only density?

## Issue 3 — Touching cells (under-merge)

Counterpart of issue 1: sometimes two distinct cells whose somata physically touch end up as one mask after erosion. There's no signal break between them, so no watershed split. This under-counts cells and inflates per-cell process count for the merged pair.

I don't have a clear example to show because I can't tell them apart from real single cells. But the issue must exist statistically given how dense the tissue is.

Question:

- Is there a principled way to detect "this single mask is actually two cells"? E.g., look for elongated/non-round soma shape? Bimodal intensity?

## Bonus — what's actually the right downstream framing

Given:
1. Over-split confounds per-cell counts upward
2. Under-merge confounds them downward
3. Normalization sensitivity flips the direction
4. n = 1 vs 2 anyway

The honest read is that **per-cell morphology comparison between WT and HET is currently uninterpretable**. The only stable signal is **cell density** (WT > HET ~25 %, robust across all 3 normalizations).

Question for you:

- Should I just drop the "process count distribution" analysis entirely from what I report to the PI, and focus only on density?
- Or is there a different morphology summary I should be reporting (e.g., total skeleton length per tissue area — patch-level) that bypasses the per-cell issues?
- For the morphology question specifically, what's the right next experiment?

## What I'd like back

For each of the 3 issues, a **specific** answer with one of:
(a) "Try this technique, here's why, here's how it differs from what you did"
(b) "Can't be solved with current data, here's the right move instead"

Don't tell me to talk to my PI — I'm doing that.
Don't list every CV technique that exists — I want your pick.
Don't summarize what I wrote.

The previous round's answer was excellent because it caught the supp-asymmetry artifact I missed and pointed at MicrogliaMorphology, μSAM, and proper analysis design. Looking for that level of sharpness again.
