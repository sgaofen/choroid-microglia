# Autonomous research log — 2026-05-24 overnight

Stephen went to sleep around 2026-05-23 night. This log captures my work
overnight with Codex. Each section is a self-contained finding so you can
read in any order.

## 2026-05-25 — methodology survey + direction reset

Stepped back from algorithm tuning to check the literature (should have done
this first). Findings:

- **Field standard = Young & Morrison 2018** (binarize → skeletonize → ImageJ
  AnalyzeSkeleton → endpoints/junctions/branch-length). Aggregate ROI readout,
  no per-cell segmentation, zero ML. Reproduced faithfully in Python with
  `skan` at `experiments/imagej_skeleton_baseline/`. Huixin confirmed ImageJ
  already extracts this → she likely wants the aggregate paradigm.
- Aggregate WT vs HET (density-normalized, per mm² signal): endpoints +8.8%,
  junctions −8.2%, branch length −5.0%, mean branch len −1.9%. Same direction
  as the v30f per-cell numbers (HET less ramified). n=1 WT vs 2 HET → underpowered.
- Per-cell tools exist too (3DMorph MATLAB, MicrogliaMorphology) but all built
  for sparse parenchymal / 3D; none handle dense 2D choroid-plexus whole-mount.
  That density is the only genuine open gap, and the only place a model helps.
- **Shipley 2020 (Huixin's own paper) has NO static morphology methods** — it's
  a calcium-dynamics / motility study. No rubric to borrow.
- **Key blind spot**: every skeleton-based method structurally misses amoeboid
  (round, process-less) microglia — a degenerate skeleton. But amoeboid
  shift IS the classic disease signal, so skeleton methods may miss exactly the
  phenotype that matters. Argues for a soma-center detector (CNN heatmap, like
  the pupa TinyUNet) that doesn't depend on processes; processes then come from
  skeleton attribution, optionally + a learned endpoint head.

**Blocker (Tue meeting with Huixin)**: confirm (1) aggregate avg vs per-cell
0–5 distribution, (2) exact cell/process definition (ideally she hand-annotates
~10 cells), (3) normalization. Until then, building detectors is premature.

Infra: project moved from `~/Documents/choroid-microglia` to
`~/choroid-microglia` (background bash can't read macOS-protected `~/Documents`).

## ⭐ Morning brief

**Working version: v30f** (`experiments/v30f_trunk_gate/v30f_run.py`).
Counts: WT_2 1594, HET_1 1356, HET_3 1263. Trunk metrics computed for
every accepted cell. Validation by Codex G: 3 BETTER / 2 NEUTRAL / 1
WORSE across 6 audited regions; the single WORSE is a "compact blobby
no-process" cell that v30g (small-cell detector) is built to recover.

**Best biological signal so far** (pooled per-cell, exploratory):
- n_trunks: WT 2.75, HET 2.74 — identical
- n_local_branches: WT 11.47, HET 10.36 (-9.7%, MWU p=2e-7)
- skel_len_local: WT 101, HET 94 (-7.5%, MWU p=3e-11)

Direction: HET microglia keep the same primary-process count but show
**less ramification** (fewer secondary branches, shorter total skeleton).
Visual at `v30f_trunk_gate/wt_vs_het_distributions.png`. Real claim needs
N≥3 images/genotype with image-level summary; currently 1+2.

**Open items**:
- ~~Codex H validation of v30g~~ DONE: **0/30 CELL across all 90 crops.**
  v30g detects almost entirely vessel-wall fragments and noise — its
  isolation criterion (dist ≥ 5 px to long branches + 8 px exclusion from
  v30f) forces it to land in vessel/empty zones where there are no real
  small cells. **v30h NOT activated.** The 17.6% HET_1 ratio is an artifact
  of more vessel structure in that slice, not a biology signal. Small-cell
  recovery is therefore an OPEN PROBLEM, not solved by v30g.
- HET_1 dense bottom-left over-split still present in v30f (next iteration)

---

## Version chain so far

| version | core idea | accepted cells (WT/HET1/HET3) | mean endpoint/cell |
|---|---|---|---|
| v29 | h-maxima seeds, soft pruning | 2395 / 2086 / 1955 | 2.22 / 2.39 / 2.62 |
| v30a | min(ecc) AND tube>blob hard filter | 2013 / 1634 / 1539 | 2.45 / 2.67 / 2.82 |
| v30b | full somaness score + graph merge | 1556 / 1380 / 1324 | 3.33 / 3.37 / 3.62 |
| v30c | + snap-to-peak + blob/density hard gate | 1396 / 1259 / 1117 | 3.57 / 3.42 / 3.82 |
| v30d | + compact-component gate + strong-rprom/n_dirs | 1355 / 1225 / 1104 | 3.67 / 3.52 / 3.87 |
| v30e | merge-bug fixes from Codex A | 1928 / 1666 / 1472 | 2.58 / 2.59 / 2.90 |
| v30f | weak floor 0.22→0.26, trunk-gate, fg_dens off | 1594 / 1356 / 1263 | 3.06 / 3.07 / 3.29 |

## Critical findings

### 1. v30c WT_2 audit (Claude + Codex cross-check)

I audited 15 top-score cells in WT_2. Initial verdict: 0 clear ✓, 9 wrong (60%).
Codex cross-checked and was even harsher — agreed on 9 wrong + flagged 5 more
I had marked "marginal" as actually NOT_A_CELL. On 3 disagreements I went back
and re-verified — Codex was right on all 3 (cells 8 L879, 11 L840, 14 L705
are all process beads / vessel junctions, not somas).

Two systemic bugs identified:
- **Endpoint over-attribution**: algorithm says 12 endpoints for a cell that
  visually has 3-4. BFS-based attribution is greedy in dense regions.
- **fg_dens not discriminative**: vessels and tubes have fg_dens=1.0 too,
  so the foreground-density gate only catches "empty space" centers, not
  vessel/tube centers.

### 2. v30d compact-component gate

Added per-Codex's recommendation: in a 13×13 px box around the snapped
center, check the connected fg component for area / eccentricity / solidity.
First attempt (18-px box, touches_border check) over-killed: only 149 cells
left in WT_2. After tuning down to 13-px box + removing touches_border check,
got back to 1355 cells.

But Codex B's dense-region audit found v30d **under-counts** badly:
- WT_2 dense: 7 cells visible, 7 accepted (balanced)
- **HET_1 dense: 7 visible, only 3 accepted**
- **HET_3 dense: 8-9 visible, only 5 accepted**

The bias flipped from over-split (v29) to under-count (v30d).

### 3. Codex A code review → 3 confirmed merging bugs

1. **Strong-strong default merge**: two real strong cells 15 px apart,
   short thick bridge between them → lower-score cell wrongly merged into
   higher-score.
2. **Weak demote ignores thin_run**: weak seed near strong with `neck_ratio
   < 0.60` was demoted, even if the path had NO thin section. Comment said
   "thin path" but code didn't check thin_run.
3. **Multi-weak collapse**: a connected component with only weak seeds (no
   strong) would keep only the BEST weak — all other weak seeds in that
   component became process_peak. Real Kolmer clusters got reduced to 1
   cell each.

### 4. Codex C merge-audit: 13/30 (43%) merges were WRONG

Looked at 30 actual merge cases (10 per image) and found WT_2 5/10,
HET_1 4/10, HET_3 4/10 wrongly merged two real cells.

### 5. v30e: applied the merging fixes

- Strong-strong: only merge if centers within 8 px (1 soma diameter); else
  keep both as ambiguous neighbors.
- Weak demote: requires both `neck_ratio < 0.60` AND `thin_run >= 5`.
- Multi-weak: keep ALL as low_confidence_soma.

Result: 0 merges produced anywhere (centers within 8 px almost never happen
since v27's earlier merging already handled close duplicates). Cell counts
recovered to 1928 / 1666 / 1472. mean endpoint dropped to 2.58-2.90.

### 6. Codex D v30e validation (2026-05-24)

- WT_2: WORSE. New L markers on central vertical process + top-right edge
  structure look like over-split (one morphology getting multiple cells).
- HET_1: BETTER but bottom-left cluster over-split.
- HET_3: BETTER, slightly too permissive.

Verdict: v30e fixed the HET_1/HET_3 under-count but introduces over-
acceptance.

### 7. Process count fundamental concern (Codex audit + biology)

Algorithm counts ENDPOINTS (distal tips), but for WT/HET comparison the
biologically meaningful quantities are:
- **primary trunk count** (3-5 typical) — number of processes leaving the
  soma, measured at a soma+collar annulus
- **total branch points** — ramification complexity
- **total skeleton length** in microns
- (current) total endpoints — over-attributed in dense regions

**Solved**: `primary_trunk_counter.py` implemented. Reports:
- WT_2: mean_trunks=2.46, mean_branches=10.57, mean_skel=95
- HET_1: mean_trunks=2.36, mean_branches=8.85, mean_skel=83
- HET_3: mean_trunks=2.53, mean_branches=10.41, mean_skel=92

HET_1 distinctly lower in branches/skel_len than WT_2 — potential biology
signal worth following up.

### 8. Threshold sweep (task #69) — 80 combos × 3 images

Pre-merge sweep on (BLOB_HARD_MIN, FG_DENSITY_MIN, SCORE_WEAK) using cached
per-seed metrics in seeds_v30e.json. Key findings:
- **FG_DENSITY_MIN is useless**: varying 0.35→0.65 changes accepted count
  by <10 cells. Real soma neighborhoods are uniformly dense; this gate
  fires almost never. Disabled in v30f.
- **BLOB_HARD_MIN is a bad lever**: raising 0.20→0.25 cuts ~140 cells per
  image but only ~1.5% of those cuts hit cells with n_trunks ≤ 1. Most
  cells removed are real microglia. Kept at 0.20.
- **SCORE_WEAK is the principled lever**: raising 0.22→0.26 cuts ~140-190
  cells per image. Of those cuts, 23-38% have n_trunks ≤ 1 (vessel pieces
  / process beads). Adopted in v30f.

### 9b. Skeleton quality (task #71)

v29's pruned skeleton stats per image:
| metric | WT_2 | HET_1 | HET_3 |
|---|---|---|---|
| skel/fg | 0.119 | 0.109 | 0.117 |
| endpoints/fg | 0.00370 | 0.00400 | 0.00402 |
| short spur frac (<5 px) | 0.003 | 0.004 | 0.003 |
| mean branch len | 15.9 | 15.8 | 16.1 |
| fg cov within 8 px | 0.961 | 0.941 | 0.944 |

Short-spur fraction is 0.3-0.4% — pruning is clean. 8-px fg coverage 94-96% —
skeleton reaches almost every cell-body region. No skeleton-quality issues.
HET_1 has slightly less dense skeleton, consistent with the lower process
density observed in trunk metrics — likely real biology, not a skel artifact.

### 9c. Codex F blob triage (task #70)

20 sampled isolated blobs (from 342 total in WT_2): 3 CELL, 4 DEBRIS,
3 VESSEL_FRAGMENT, 6 NOISE, 4 AMBIGUOUS. Roughly 15-25% are likely real
small cells (Kolmer epiplexus). Current pipeline rejects all of them because
they don't pass the annulus trunk-count or the n_dirs>=3 strong criterion.

Implication: ~50 missed cells per image. Need a separate "small round cell"
detector (task #74) with conservative criteria: compact filled body, bright
core + faint halo, separation from linear vessel/wall structures, area >= 15
px, low eccentricity.

### 9. v30f trunk-gate (task #73)

After SCORE_WEAK=0.26, added an explicit trunk-gate: weak/low_confidence
cells must show at least 1 annulus skeleton-trunk to survive. Strong cells
are exempt (they already pass n_dirs >= 3).

Effects (v30e → v30f):
- WT_2: 1928 → 1594 (-17.3%), mean endpoints 2.58 → 3.06
- HET_1: 1666 → 1356 (-18.6%), mean endpoints 2.59 → 3.07
- HET_3: 1472 → 1263 (-14.2%), mean endpoints 2.90 → 3.29

Trunk-gate alone demoted 148 / 140 / 98 cells in WT/HET1/HET3 — these are
weak somas with no annulus trunk visible, i.e. just a bright dot with no
attached process at the expected primary-process distance. Combined with
the score-floor tightening, total reduction is principled.

Codex G dispatched to audit whether v30f killed the right cells (the
Codex-D-flagged over-acceptances) without losing legit ones.

### 9d. Codex G v30f validation

Codex G's 6-region audit:
| region | verdict |
|---|---|
| WT_2 dense | BETTER (removed over-split markers, kept all somas) |
| HET_1 dense | NEUTRAL (didn't fix the bottom-left cluster over-split) |
| HET_3 vessel dense | BETTER (removed edge/dim cyan Ls correctly) |
| WT_2 tight | NEUTRAL (1 good over-split removal, 1 ambiguous compact soma loss) |
| **HET_1 tight** | **WORSE** (lost an upper-right blobby compact soma) |
| HET_3 tight | BETTER |

Score: 3 BETTER / 2 NEUTRAL / 1 WORSE. The single WORSE is exactly the cell
type v30g (small-round-cell detector) is designed to catch — a compact body
with no clear primary process trunk. The combined pipeline v30f + v30g
should restore this cell while keeping v30f's other gains.

### 9e. WT vs HET stats (exploratory, n=1+2 images)

Per-cell pooled comparison on v30f trunk metrics:
| metric | WT_mean | HET_mean | delta | Cohen_d | MWU_p |
|---|---|---|---|---|---|
| n_trunks | 2.75 | 2.74 | -0.6% | 0.02 | 0.21 |
| n_local_branches | 11.47 | 10.36 | -9.7% | 0.14 | 2.1e-7 |
| skel_len_local | 101.08 | 93.50 | -7.5% | 0.19 | 2.6e-11 |
| score | 0.39 | 0.39 | -0.8% | 0.04 | 0.33 |

Direction: HET keeps the same number of PRIMARY processes but has fewer
sub-branches and less total skeleton — i.e. less ramified. Effect sizes are
small (d 0.14-0.19) but distributionally clear. Real claim needs N>=3 images
per genotype with IMAGE-LEVEL summaries, not pooled per-cell.

## Files written so far

```
v30c_snap_and_blob_gate/
  v30c_run.py
  codex_audit.md                          — Codex's 15-cell audit
  audit_F_WT_2_WTtight_cell{00-14}_*.png  — per-cell verification crops

v30d_compact_gate/
  v30d_run.py
  codex_review_merging.md                 — Codex A: 3 merging bugs
  codex_audit_dense.md                    — Codex B: dense under-count
  codex_audit_merge_and_het.md            — Codex C: 43% wrong merges
  audit_full.py                           — generates all audit crops

v30e_merge_fixes/
  v30e_run.py                             — previous version
  codex_audit_dense_v30e.md               — Codex D: HET fixed, WT regressed
  primary_trunk_counter.py                — annulus trunk counter
  trunk_metrics_v30e.json (per image)     — n_trunks/n_branches/skel_len
  threshold_sweep.py                      — 80-combo pre-merge sweep
  threshold_vs_trunks.py                  — which threshold cuts hit low-trunk
  sweep_F_*.csv                           — full sweep CSV per image
  extract_blob_candidates.py              — find isolated bright blobs
  blob_candidates/blob_01..20.png         — sampled blob crops
  codex_blob_triage.md                    — Codex F: classify blobs (running)

v30f_trunk_gate/
  v30f_run.py                             — current best
  *_seeds_v30f.json                       — per-seed records
  cmp_F_*_*.png                           — 6 side-by-side v30e vs v30f
  compare_v30e_v30f.py                    — comparison generator
  codex_audit_v30f_vs_v30e.md             — Codex G (running)
```

### 9f. v30g small-cell proposer + Codex H (task #74)

Conservative criteria (area 15-80, ecc<=0.7, sol>=0.85, peak/mean>=1.25,
dist>=5 to long branches, not within 8 px of v30f). Candidates per image:

| image | v30f cells | v30g candidates | v30g/v30f ratio |
|---|---|---|---|
| F_WT_2 | 1594 | 153 | 9.6% |
| F_HET_1 | 1356 | 238 | **17.6%** |
| F_HET_3 | 1263 | 167 | 13.2% |

If v30g precision is similar across images, HET_1's elevated small-cell
ratio (17.6% vs WT_2's 9.6%) suggests a real morphology shift toward
amoeboid/Kolmer-like cells in this slice. Consistent with the previously
observed lower n_branches in HET_1.

Codex H dispatched to classify 30 sampled crops per image. Codex H first
attempt bailed early; H2 re-dispatched.

**Codex H2 verdict: 0/30 CELL across all 90 crops.** Almost entirely
VESSEL_FRAGMENT (60-80% per image) and NOISE (15-35%), with 0% CELL. Three
independent samples (30 each) gave the same answer. Strong rejection of
v30g's current design.

**Root cause** (Claude's analysis): The v30g criteria require
`min_dist_to_long_branch >= 5 px` AND exclude 8 px around every v30f
accepted cell. After both filters, the surviving candidates can only be in
regions that are FAR from any process or accepted soma — i.e., in
inter-cell empty zones. Real small microglia (Kolmer-like) live ON the
choroid plexus epithelium and near other cells; they are NOT in isolated
empty zones. What v30g selects instead is **isolated bright pieces of the
choroid plexus vasculature/vessel walls** — fragments of the same wall
structures the long_branch_mask was already filtering for. The mask
detects long branches as cells' processes but does not detect the wall
fragments that are not part of any long branch.

**Implication for HET_1**: the elevated 17.6% small-cell ratio in HET_1
is NOT a biology signal of more amoeboid cells. It is simply more wall
fragmentation in that slice, presumably reflecting different scanning
slice depth, fixation, or genuine vascular density differences. **The
n_branches/skel_len differences in v30f trunk metrics remain the
genotype signal worth following up**; the v30g count does not.

**Conclusion**: v30g rejected. v30h NOT activated. Future small-cell
recovery (if Kolmer cells matter for the genotype comparison) needs a
different design: don't require isolation, but require a positive cell
signature (e.g., compact connected fg with intensity pattern matching
canonical Kolmer cell, validated with a small ground-truth set).

### 9g. v30h scaffold (NOT ACTIVATED)

`v30h_integrated/v30h_combine.py` was written for the union-with-dedup
case. After Codex H2's 0% verdict, NOT ACTIVATED. File kept as scaffold
for a future small-cell detector with a different design.

## Codex agents this session

- **D** done — v30e validation (HET fixed, WT regressed)
- **F** done — blob triage (15% CELL, recommend small-cell detector)
- **G** done — v30f validation (3 BETTER / 2 NEUTRAL / 1 WORSE)
- **H** bailed early without writing output → **H2 re-dispatched** (still running)

## Suggested morning decisions for Stephen

1. **Adopt v30f as the working version.** Threshold sweep + trunk-gate gave
   the cleanest cell population so far (mean endpoints 3.0+, mean trunks
   2.7+) with WT vs HET ramification difference clearly visible.

2. ~~v30g/v30h~~ **REJECTED** — Codex H2 found 0/30 CELL in 90 crops.
   The "isolation" filter pushed v30g into selecting vessel-wall
   fragments instead of small cells. The HET_1 elevated small-cell ratio
   (17.6%) was therefore NOT a biology signal. Do not activate v30h. If
   small-cell recovery becomes important for your final analysis, it
   needs a positive-signature detector (small ground-truth set) rather
   than the current negative-isolation approach.

3. **Open HET_1 dense bottom-left over-split as a future iteration.** v30f
   didn't fix it. Possibilities: cluster-compactness gate (multiple weak
   seeds on the same fg component within radius R → keep only the best),
   or graph-cut style cell partitioning. Both require careful audit because
   merging logic has been the most error-prone area.

4. **Acquire more WT and HET images.** The biology direction is clear from
   per-cell pooled stats but image-level inference needs N >= 3 per
   genotype. Right now we have 1 WT, 2 HET.

5. **Consider Cellpose 2 finetune as a future track.** v30f's classification
   could serve as weak labels (strong=high-confidence positive). Sample
   ~500 crops with mixed positive/negative labels, finetune Cellpose 2 on
   them, compare cell counts head-to-head against v30f.

## Files written this session

```
v30e_merge_fixes/
  threshold_sweep.py + sweep_F_*.csv      — 80-combo sweep
  threshold_vs_trunks.py                  — discriminates kill-low-trunk
  primary_trunk_counter.py + trunk_metrics_v30e.json
  extract_blob_candidates.py + blob_candidates/blob_01..20.png
  codex_blob_triage.md                    — F: 15% CELL

v30f_trunk_gate/
  v30f_run.py                             — WORKING VERSION
  *_seeds_v30f.json                       — per-image seed records
  trunk_metrics_v30f.py + *_trunk_metrics_v30f.json
  skeleton_quality.py + skeleton_quality.json
  wt_vs_het_stats.py                      — stat scaffold
  wt_vs_het_plot.py + wt_vs_het_distributions.png
  cmp_F_*.png (6)                         — side-by-side v30e vs v30f
  codex_audit_v30f_vs_v30e.md             — G: 3B/2N/1W
  compare_v30e_v30f.py                    — comparison generator

v30g_small_cells/
  v30g_small_cell_proposals.py            — conservative criteria
  *_small_cell_candidates_v30g.json       — per-image candidates
  small_cell_crops_F_*/small_cell_NN.png  — 30 audit crops per image
  codex_audit_v30g.md                     — H2: pending

v30h_integrated/
  v30h_combine.py                         — dry-runs; activate after H2
```
