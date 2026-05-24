# Visual audit: 18 cases of endpoint-near-branchpoint (≤3 px)

Looked at 9 WT_2 cases (random.seed=42) + 9 HET_1 cases (random.seed=123). All confirmed same 3 artifact types.

## Type 1: TIP THICKENING (~30% cases)
Process ends in a slightly wider tip. Skeletonize creates star pattern at the end. 1 real endpoint becomes [endpoint + branchpoint + extra endpoint] within 2-3 px.
- WT_2 cases: 3, 5, 7, 9
- HET_1 cases: 8

## Type 2: JUNCTION ARTIFACT (~50% cases)
Real Y/multi-junction. Skeleton creates small additional spurs around the junction. Multiple endpoints clustered within 2-3 px of the branchpoint.
- WT_2 cases: 4, 6, 8
- HET_1 cases: 3, 4, 6, 7, 9

## Type 3: CURVE ARTIFACT (~15% cases)
Wavy/curving line creates tiny Y-forks at bends. The branchpoints are "fake" — should be regular degree-2 pixels.
- WT_2 cases: 1, 2
- HET_1 cases: 1, 2

## Type 4: LOOP (rare, ~5%)
Skeleton forms closed loop from thick foreground region. Endpoint dangles off the loop.
- HET_1 cases: 5

## General fix

```python
For each endpoint E:
  Walk along skeleton from E until hit (branchpoint OR another endpoint OR > 3 steps)
  If reached a branchpoint AND path_length ≤ 3:
    Remove path from skeleton (E and 1-2 intermediate pixels)
    Branchpoint may demote to degree-2 (no longer junction)
```

## Expected impact
- 707 endpoints (WT) + 672 (HET_1) within 3px of branchpoint = ~11% of all endpoints
- After fix: ~10-15% endpoint reduction
- Per-cell endpoint count drops by 1-3 on average
- Distribution shape preserved (Kolmer cells still 0, ramified still high)

## Will NOT touch
- Spurs >3 px long: those are real short branches, keep
- Loops (Type 4): handle separately if needed

## Decision: implement spur_prune with max_len=3 only
This is conservative — doesn't risk removing real branches. Only kills obvious tip+junction artifacts.
