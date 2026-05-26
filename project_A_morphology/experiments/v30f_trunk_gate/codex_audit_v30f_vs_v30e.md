# v30f vs v30e Audit

## Dense regions

### WT_2 dense (cmp_F_WT_2_WTdense.png)
- ORIG soma count: 4 clear somas. I count the upper-center horizontal soma, the large mid-right soma, the lower-left soma, and the lower-right edge soma; the top-right edge structures and center vertical strands look process/vessel-like rather than separate somas.
- v30e accepted: 11 (S=1, L=10, A=0 visible)
- v30f accepted: 9 (S=1, L=8, A=0 visible)
- Removed markers analysis: v30f drops the upper-left cyan L on the horizontal arm of the upper-center cell and the isolated lower-middle cyan L in a low-signal/dark area. Both removals look like over-splits or debris rather than legitimate soma loss. v30f still leaves multiple cyan L markers on the same upper-center morphology and on the top-right vessel-like edge, so the over-acceptance is reduced but not eliminated.
- Verdict: v30f BETTER

### HET_1 dense (cmp_F_HET_1_HET1dense.png)
- ORIG soma count: 6 clear somas. I count three in the upper interconnected group, one central/lower soma, one bottom-left cluster soma, and one bottom-center soma; partial bottom-edge fragments are not counted.
- v30e accepted: 9 (S=4, L=5, A=0 visible)
- v30f accepted: 9 (S=4, L=5, A=0 visible)
- Removed markers analysis: I do not see any accepted marker removed between v30e and v30f. The bottom-left cluster still has three accepted markers around one local morphology, and the upper/mid-left connected structure still has markers on process-like segments. v30f therefore does not address the HET_1 dense over-split pattern, but it also does not newly remove a legitimate soma in this crop.
- Verdict: v30f NEUTRAL

### HET_3 vessel dense (cmp_F_HET_3_HET3vessel_dense.png)
- ORIG soma count: 5 clear somas, with one additional possible soma along the central horizontal bright structure. The top-left edge fragment and the long horizontal vessel-like ridge are difficult, but I do not count the top-left edge fragment as a soma.
- v30e accepted: 10 (S=3, L=7, A=0 visible)
- v30f accepted: 8 (S=3, L=5, A=0 visible)
- Removed markers analysis: v30f drops the upper-left cyan L on the edge/vessel-like fragment and the isolated lower-left cyan L in a dim area below the main structures. Neither removed marker has a convincing round/blobby soma in ORIG. The retained S markers correspond better to real soma-like bodies in the central/lower cluster, although v30f still keeps several cyan markers on vessel/process-like segments.
- Verdict: v30f BETTER

## Tight regions

### WT_2 tight (cmp_F_WT_2_WTtight.png)
- ORIG soma count: 8, including one low-confidence compact soma at center-right. The clear somas are upper-left/partial top, upper-center/right, left-middle, center vertical, mid-lower right, lower-right, and lower-center; the center-right compact bright body is soma-like but has little visible process.
- v30e accepted: 14 (S=2, L=12, A=0 visible)
- v30f accepted: 12 (S=2, L=10, A=0 visible)
- Removed markers analysis: v30f removes the center-right cyan L on the compact bright body and a right-middle cyan L on a process branch. The right-middle removal is a good over-split removal. The center-right removal is ambiguous: the ORIG panel shows a small blobby object there, but it lacks a clear trunk/process, so v30f may have removed a legitimate compact soma or a low-confidence debris object. Overall this crop shows a useful reduction in over-splits, with some risk of losing a real weak/compact soma.
- Verdict: v30f NEUTRAL

### HET_1 tight (cmp_F_HET_1_HET1tight.png)
- ORIG soma count: 5 clear somas. I count the top-left edge cell, the central vertical cell, the upper-right compact/blobby cell, the lower-center cell, and the lower-right cell; bottom-edge specks are not counted.
- v30e accepted: 8 (S=2, L=6, A=0 visible)
- v30f accepted: 5 (S=2, L=3, A=0 visible)
- Removed markers analysis: v30f drops three cyan L markers: a top-center/left small speck near the top-left process, the upper-right cyan L on a clear blobby soma, and one lower-edge cyan L near a partial/low-signal fragment. The top-center/left and lower-edge removals look like plausible false positives, but the upper-right removal appears to be a legitimate soma loss. This is the clearest normal-density regression in the six crops.
- Verdict: v30f WORSE

### HET_3 tight (cmp_F_HET_3_HET3tight.png)
- ORIG soma count: 8 clear somas. I count the top-left partial cell, top-middle cell, top-right partial cell, mid-right cell, lower-left cell, lower-middle-left cell, lower-middle-right cell, and lower-right edge cell.
- v30e accepted: 9 (S=2, L=7, A=0 visible)
- v30f accepted: 8 (S=2, L=6, A=0 visible)
- Removed markers analysis: v30f removes one isolated lower-middle cyan L in a dark/low-signal area below the central cells. There is no convincing soma at that location in ORIG, so this looks like a good weak false-positive removal. The upper-left partial cell still has multiple markers on connected morphology, but the v30e-to-v30f change itself does not appear to lose a legitimate soma.
- Verdict: v30f BETTER

## Overall verdict
v30f partly did its job: in WT_2 dense, HET_3 vessel dense, and HET_3 tight, the removed markers are preferentially weak non-soma/process/vessel-like acceptances. It did not improve the HET_1 dense over-split case at all, and it introduces a real regression in HET_1 tight by removing an upper-right soma-like cell. WT_2 tight is mixed because one removed marker is a good process over-split, while the other is an ambiguous compact soma. Overall, v30f is better at suppressing weak false positives, but the trunk gate is not clean enough to claim "no legitimate cells lost."

## One-line verdicts
- WT_2 dense: v30f BETTER
- HET_1 dense: v30f NEUTRAL
- HET_3 vessel dense: v30f BETTER
- WT_2 tight: v30f NEUTRAL
- HET_1 tight: v30f WORSE
- HET_3 tight: v30f BETTER
