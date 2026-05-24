# Personal audit — what I actually saw on full-image native scans (2026-05-21)

I scanned four 200×200 native crops of WT_2 spread across the tissue, plus pulled out 9 individual cells across different morphology categories. Below is what I saw, region by region.

## Visual rubric — what counts as a cell

A microglia in this dataset has at least one of:

1. **A compact bright central region** — the cell body (soma). Looks like an oval or round bright spot, roughly 15–30 px across at native resolution (3–6 µm). Brightness is clearly higher than surrounding tissue.
2. **Bright filamentous extensions** — processes, thin lines emerging from the soma. Usually dimmer than the soma but still clearly above background.

A "Kolmer cell" has only criterion 1 (oval soma, no processes). A "ramified microglia" has both. Either qualifies as a cell.

**What is NOT a cell**:
- Single bright pixels with no surrounding structure — imaging noise / hot pixels
- Long continuous bright lines without a thicker node — probably vasculature, not a single cell
- Diffuse glow with no central peak — out-of-focus or autofluorescence
- Bright tissue edges or folds — geometric brightening artifact

**The hard cases**:
- A process clearly extending from one cell into another's territory: which cell owns it? Watershed by soma centroid is our convention but it's not biologically perfect.
- A bright filament without any visible soma in the frame: this is probably a process from a cell whose soma is outside the crop (or hidden by overlap). Counts as "evidence of a cell" but the actual cell may not be detectable in this 2D max projection.
- A "donut" — outlined region with darker interior. Common pattern when the soma cytoplasm is membrane-localized and the nucleus center is unstained. Counts as a real cell.

## Region-by-region findings on WT_2

### Region A (y=500–700, x=600–800) — upper tissue
- **14 cells outlined.**
- All outlines correspond to genuine bright structures. No outlines floating in empty space.
- One outline in the lower-middle is suspiciously round and the interior is mostly dim — could be a very faint cell or a borderline false positive. Worth Stephen looking at this one specifically.
- A few small bright spots and short filaments in the gaps are not outlined — these are probably process fragments from outlined cells, not separate missed cells.

### Region B (y=1600–1800, x=1200–1400) — deep dense interior
- **13 cells outlined.**
- **Clearest missed cell observed in the entire audit: a large, classically ramified microglia in the bottom-center with bright soma + ~5 radiating processes is NOT outlined by Cellpose.** This is exactly the "soma faint relative to processes" case I expected. Its processes get watershed-assigned to neighbors.
- One outline on the bottom-left looks like 2–3 cells merged into a single mask.
- Several fine filaments running through the middle of the region are not outlined; these are likely processes from cells in adjacent crops.

### Region C (y=2500–2700, x=1300–1500) — lower lobe
- **15 cells outlined.** (Not screenshotted, similar in character to A.)

### Region D (y=1400–1600, x=2300–2500) — right edge area, densest of the four
- **22 cells outlined** — highest density.
- All outlines have visible signal underneath; no false positives in empty space.
- Many "donut" outlines (dark interior, bright ring) — these are cells where the staining is on the cell body periphery; they are real cells.
- Some outlines clearly include parts of adjacent cells — the boundary between touching cells is fuzzy.
- The dense cluster on the left has 4–5 cells whose outlines run into each other; could be over-counting from accidental splits or under-counting from merges. Hard to tell without ground truth.

## Summary of detection failure modes

After looking at the data systematically, the three real error modes:

| Failure mode | Frequency | Severity |
|---|---|---|
| **Missed cells** with faint soma relative to processes | A few per 200×200 crop, ~5–10% of true cells | High — these are the most-ramified cells, which is exactly the population the morphology question is about |
| **Merged cells** in dense regions where processes interlock | Hard to estimate without ground truth; visually 2–5% of outlines look like merges | Medium — inflates "ramified count" for some outlines, can be tolerated for distribution-level statistics |
| **False positives** (outlines around mostly empty area) | Very rare — 0–1 per 200×200 crop | Low |

The over-detection threshold (cellprob_threshold=−2) is **not** producing junk false positives. What it gave us is real additional cells that the default threshold missed, plus a few ambiguous edge cases.

## Honest answer to "is my detection trustworthy"

**For counting** (cells per image, gross density): yes, trustworthy at ~5–10% noise level. The 1100 → 1800 jump from baseline → extended is real cells, not garbage.

**For morphology classification**: not yet. The systematic under-detection of faint-soma ramified cells means the most-ramified end of the distribution is incomplete. This is the bottleneck.

**To fix**: either (a) hand-correct ~30–50 missed cells in one image to fine-tune Cellpose, or (b) add a complementary blob detector that runs after Cellpose to pick up the faint-soma cases.

## Files

- `rubric/rubric_gallery.png` — 9-panel labeled gallery of canonical cases
- `audit_scan/audit_{A,B,C,D}_*.png` — the four region overlays I scanned
- `cellpose_extended/all_cells_extended.json` — per-cell metrics for all images
- `review_crops/` — the 9 paired crops + CSV from earlier
