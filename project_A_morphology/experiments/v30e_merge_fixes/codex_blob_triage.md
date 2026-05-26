# TL;DR

In this 20-crop sample, 3/20 (15%) look like likely real compact cells, 13/20 (65%) look like clear non-cell artifacts/noise, and 4/20 (20%) are too low-resolution or context-contaminated to call confidently. Extrapolated to 342 total candidates, that suggests roughly 50 likely real small cells, with a plausible higher range around 15-25% if some ambiguous crops prove real; most candidates should still be ignored unless they pass stricter morphology checks. v30f should include a separate small-round-cell detector, but it should be conservative rather than counting every unattached blob candidate.

# Classification Table

| Blob ID | Classification | Key Features | Confidence (high/med/low) |
|---|---|---|---|
| blob_01 | AMBIGUOUS | Off-center compact bright lower blob near the crop boundary; nearby bright linear/edge structures make it unclear whether this is a rounded cell or a fragment. | low |
| blob_02 | DEBRIS | Faint irregular central smudge with uneven brightness and no round body; separate bright fragments near crop edges. | med |
| blob_03 | VESSEL_FRAGMENT | Crowded crop with bright linear/branching wall-like structures; centered object appears as an elongated stub among broken linear fragments. | high |
| blob_04 | VESSEL_FRAGMENT | Thin vertical bright stub through the center with tapered/linear shape; large bright wall-like structure at upper left. | med |
| blob_05 | NOISE | Tiny isolated central bright dot, only a few pixels across, with no visible halo or internal structure; edge structures are separate. | med |
| blob_06 | NOISE | Small compact central speck without a resolved body or stubs; unrelated bright structures sit around the crop edges. | med |
| blob_07 | NOISE | Center signal is essentially a tiny speck, while the obvious bright structure is a separate vessel/branch-like complex off to the right/lower side. | high |
| blob_08 | NOISE | Center candidate is only one to a few bright pixels; no resolved body, with a separate vessel/wall fragment along the left edge. | high |
| blob_09 | AMBIGUOUS | Small isolated round-ish central spot with a bright core, but it is too few pixels to separate a tiny cell from a bright dot. | low |
| blob_10 | DEBRIS | Asymmetric short dash/comma near the center with faint adjoining linear signal; not a smooth round cell body. | med |
| blob_11 | NOISE | Tiny vertical dash/point at center with no resolved body or halo; brighter edge fragment is separate. | med |
| blob_12 | CELL | Compact oval central cluster with a bright core and faint surrounding halo; not visibly attached to the nearby right-side fragment. | med |
| blob_13 | DEBRIS | Jagged central cluster adjacent to a short broken linear fragment; lacks a smooth rounded outline. | med |
| blob_14 | DEBRIS | Small central bright core with a short stub/secondary fragment; surrounding crop contains additional linear fragments. | med |
| blob_15 | CELL | Small compact rounded/ovoid bright body at center with a smooth-ish outline and no visible attachment. | med |
| blob_16 | AMBIGUOUS | Small central bright cluster close to brighter linear structures on the right/bottom; cannot tell isolated cell from fragment/noise. | low |
| blob_17 | VESSEL_FRAGMENT | Bright candidate region sits within a larger curved/linear wall-like mass near the lower-right crop edge, not an isolated round body. | med |
| blob_18 | NOISE | Center signal is a tiny vertical speck with no resolved body or internal structure; larger bright material is elsewhere in the crop. | high |
| blob_19 | CELL | Compact bright oval/clump at center with a filled body and faint halo, visually separated from edge structures; slightly elongated. | med |
| blob_20 | AMBIGUOUS | Central object has a compact bright head plus a short curved/downward extension; could be a small cell with a stub or a debris fragment. | low |

# Per-Blob Brief Rationale

blob_01: The visible candidate-like object is a compact bright lower blob, but it is off-center and near a crop/image boundary. Nearby bright linear structures make the morphology too contaminated to confidently call CELL or fragment.

blob_02: The central object is a faint, irregular smudge rather than a filled round body. Its uneven brightness and small broken shape fit debris better than a cell.

blob_03: The crop is dominated by bright branching or wall-like structures. The centered signal is elongated and stub-like, so this reads as a vessel/wall fragment rather than an isolated small cell.

blob_04: The central feature is a narrow vertical segment with a tapered linear profile. That morphology is more consistent with a short vessel/branch fragment than a rounded cell.

blob_05: The central signal is a very small bright dot with no visible cellular body, halo, or internal structure. I would treat this as noise rather than count it.

blob_06: The central spot is compact but only a few pixels across and lacks any resolved body. The crop contains other unrelated bright structures, but the candidate itself looks like a speck.

blob_07: The center contains only a minimal bright speck. The prominent bright material is off-center and linear/fragmented, so the candidate should be ignored as noise.

blob_08: The center is essentially a tiny point signal. The large left-edge structure appears separate and vessel/wall-like, leaving no visible cell morphology at the candidate.

blob_09: The central object is round-ish and isolated, but it is extremely small. Because it could be either a tiny cell-like dot or random bright noise at this resolution, I marked it ambiguous.

blob_10: The central feature is an asymmetric dash with faint linear continuation. It does not show a compact rounded body, so debris is the best fit.

blob_11: The candidate is a tiny vertical dash/point with no discernible structure. This is below what I would count as a real cell from the crop alone.

blob_12: The center shows a compact oval cluster with a bright core and slight halo, visually separated from the right-side fragment. That is one of the clearer likely small-cell morphologies in the sample.

blob_13: The central material is jagged and sits next to a broken short linear fragment. The shape is irregular rather than round, supporting debris.

blob_14: The center has a small bright core but also a stub-like/secondary piece. With additional linear fragments nearby and no clean rounded outline, debris is more likely than a cell.

blob_15: The central object is compact, rounded to ovoid, and not visibly attached to a linear structure. The crop resolution limits confidence, but this is a likely small cell.

blob_16: The central cluster is small and close to brighter linear material on the right/lower side. The crop does not provide enough separation or shape detail for a confident call.

blob_17: The candidate region appears embedded in or adjacent to a larger curved/linear bright mass near the crop edge. It does not look like a separate rounded cell.

blob_18: The center signal is a tiny vertical speck with no resolved body. The larger bright fragment lower/right is separate from the center candidate.

blob_19: The central clump is bright, compact, and filled, with a faint surrounding halo and no obvious attachment. It is slightly elongated, so confidence is medium rather than high.

blob_20: The object has a compact bright head but also a short downward/curved extension. That could be a small cell with a stub, but from this crop alone it is not distinct enough from debris.

# Summary Counts

| Category | Count out of 20 |
|---|---:|
| CELL | 3 |
| DEBRIS | 4 |
| VESSEL_FRAGMENT | 3 |
| NOISE | 6 |
| AMBIGUOUS | 4 |

# Recommendation

v30f should add a separate small-round-cell detector because a real minority of these unattached candidates appear countable: about 15% in this sample, possibly closer to 20-25% if some ambiguous crops are true cells. The detector should not simply count all unattached blobs; most sampled candidates are specks, debris, or vessel/wall fragments.

The visible features that best distinguish likely cells here are a compact filled oval/round body, a bright core with a faint halo, separation from linear wall/branch structures, and no obvious elongated tail. Artifacts tend to be one-to-few-pixel specks, jagged broken fragments, thin linear stubs, or candidates embedded in larger curved vessel/wall-like structures.
