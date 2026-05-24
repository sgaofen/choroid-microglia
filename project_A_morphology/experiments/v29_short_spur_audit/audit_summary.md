# Short-spur prune audit — final summary

## Visual audit scope

**Examined 27 cases total**:
- 9 WT_2 cases (seed=42) — confirmed 3 artifact types
- 9 HET_1 cases (seed=123) — confirmed same 3 types
- 9 HET_3 cases (seed=999) — confirmed same 3 types after prune

Each case viewed once, observations cached in text, image deleted from disk.

## Artifact pattern (confirmed across 27 cases)

| Type | % of cases | Visual signature |
|---|---|---|
| Junction artifact | ~55% | Real Y-junction + 1-2 extra small spurs within 2-3 px |
| Tip thickening | ~30% | Process tip thicker → skeleton creates star at end |
| Curve artifact | ~15% | Skeleton along bend creates tiny Y in wavy line |

## Fix applied: spur_prune(max_len=3)

Walks each endpoint along skeleton ≤3 px. If reaches a branchpoint, deletes path.

## Impact on data
```
                Endpoints       Branchpoints    Skeleton pixels
WT_2            6427 → 5352     18761 → 17846   174534 → 172174
HET_1           5965 → 5000     14268 → 13426   138733 → 136725
HET_3           6094 → 5163     15994 → 15199   151922 → 149967
```

- Endpoints: -15 to -17%
- Branchpoints: -5%
- Skeleton: -1.5%

**The numbers fit the expected pattern**: skeleton mass barely changes (real structure preserved), endpoint count drops significantly (artifacts removed).

## Before/After visual verification

Looked at all 27 same locations after prune:
- **27/27 cases visually improved**: artifact endpoints gone, real structure intact
- No real branches were removed
- Some branchpoints demoted to degree-2 (former junctions become regular skeleton)

## Conclusion

`max_len=3` is a safe threshold. Aggressive enough to clean ALL observed artifact types (tip, junction, curve), conservative enough to preserve real short branches (≥5 px).
