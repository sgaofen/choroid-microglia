# audit_F_WT_2_WTdense_region.png

- Estimated true count: ~7-8 real cells; algorithm accepted count: 7 lime/cyan markers. Comparison: roughly balanced, possibly slight under-split if the top-center pair are two cells.
- Dominant error mode: wrong-position / ambiguous splitting.
- Bad cases:
  - (~82,47) and (~108,46): two cyan L markers sit on a connected top-left/top-center structure. This may be two adjacent somas, but if the bright horizontal blob is one soma plus process, it is over-split.
  - (~179,77): lime S is close to a bright soma/process junction but appears offset onto the process neck rather than centered on the compact soma.
  - (~221,222) and (~240,242): two cyan L markers in the lower-right branched object may split one ramified cell/process field; low confidence because the bottom edge is crowded.
- Yellow M merges: no yellow M markers are visible, so no merge calls to judge.
- Completely missed cells: no clear high-confidence missed soma; a few dim compact blobs near (~137,204) and lower edge are low-confidence.

# audit_F_HET_1_HET1dense_region.png

- Estimated true count: ~7 real cells; algorithm accepted count: 3 lime/cyan markers. Comparison: under-split.
- Dominant error mode: missed / under-split.
- Bad cases:
  - (~59,58): bright compact left-top soma is rejected as red P and has no accepted marker.
  - (~143,42): yellow M lies on a bright top-center soma-like body; it looks more like a separate cell than a duplicate of the nearby right structure.
  - (~128,134): lower central bright soma/process junction has only red P markers nearby and no accepted S/L marker.
- Yellow M merges: the single visible M does not look clearly legitimate; it likely absorbed a real adjacent soma in the dense top cluster.
- Completely missed cells: yes, likely the left-top soma (~59,58), right-top soma (~166,55), and lower-central soma (~128,134).

# audit_F_HET_3_HET3vessel_dense_region.png

- Estimated true count: ~8-9 real cells, with several low-confidence because of vessel-like bright background; algorithm accepted count: 5 lime/cyan markers. Comparison: under-split.
- Dominant error mode: missed / wrong-position near vessel-like structures.
- Bad cases:
  - (~111,131): central compact branch point is red P, while the nearest accepted S at (~130,140) does not fully cover it; likely missed or absorbed.
  - (~165,130) and (~204,132): bright compact peaks along the horizontal structure are rejected as P. These may be real somas, though confidence is lower because they sit on a vessel-like band.
  - (~150,203), (~190,205), and (~145,238): bottom dense network has several compact bright bodies marked P or not accepted, suggesting under-counting.
- Yellow M merges: the M near (~229,233) is suspect; it sits on a compact bright bottom-right cluster and is not obviously a duplicate.
- Completely missed cells: yes, likely central (~111,131) and multiple bottom-network cells; some are low-confidence due to overlap with vessel/background signal.

# Summary

Worst dense region: `audit_F_HET_1_HET1dense_region.png`, because only 3 accepted markers cover about 7 visible cells and several clear somas are rejected or unmarked. Overall bias across these dense crops is under-split / under-count, with additional wrong-position errors where accepted markers land on process junctions rather than soma centers.
