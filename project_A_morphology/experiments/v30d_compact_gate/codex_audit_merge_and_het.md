## Section 1: Merge audit

| image | crop | verdict |
|---|---:|---|
| WT_2 | merged00 | WRONG_MERGE |
| WT_2 | merged01 | WRONG_MERGE |
| WT_2 | merged02 | OK_MERGE |
| WT_2 | merged03 | WRONG_MERGE |
| WT_2 | merged04 | OK_MERGE |
| WT_2 | merged05 | AMBIGUOUS |
| WT_2 | merged06 | OK_MERGE |
| WT_2 | merged07 | WRONG_MERGE |
| WT_2 | merged08 | WRONG_MERGE |
| WT_2 | merged09 | OK_MERGE |
| HET_1 | merged00 | OK_MERGE |
| HET_1 | merged01 | WRONG_MERGE |
| HET_1 | merged02 | WRONG_MERGE |
| HET_1 | merged03 | OK_MERGE |
| HET_1 | merged04 | OK_MERGE |
| HET_1 | merged05 | WRONG_MERGE |
| HET_1 | merged06 | AMBIGUOUS |
| HET_1 | merged07 | OK_MERGE |
| HET_1 | merged08 | OK_MERGE |
| HET_1 | merged09 | WRONG_MERGE |
| HET_3 | merged00 | AMBIGUOUS |
| HET_3 | merged01 | AMBIGUOUS |
| HET_3 | merged02 | OK_MERGE |
| HET_3 | merged03 | WRONG_MERGE |
| HET_3 | merged04 | WRONG_MERGE |
| HET_3 | merged05 | WRONG_MERGE |
| HET_3 | merged06 | OK_MERGE |
| HET_3 | merged07 | OK_MERGE |
| HET_3 | merged08 | OK_MERGE |
| HET_3 | merged09 | WRONG_MERGE |

Wrong merges per image: WT_2 5/10; HET_1 4/10; HET_3 4/10.

Examples: WT_2 merged01 has yellow on an upper bright soma/branch complex and lime on a lower-right compact object, with a clear dark gap and different local arbors. HET_3 merged05 has yellow on a top compact cell with radiating processes while lime is on a separate lower elongated structure, not the same soma.

## Section 2: HET per-cell

HET_1:

| cell | verdict |
|---:|---|
| 00 | OK |
| 01 | OK |
| 02 | OK |
| 03 | OK |
| 04 | OK |
| 05 | OK |
| 06 | OK |
| 07 | NOT_A_CELL |
| 08 | NOT_A_CELL |
| 09 | OK |
| 10 | OK |
| 11 | NOT_A_CELL |
| 12 | OK |
| 13 | OK |
| 14 | OK |

Totals: OK 12; CENTER_OFF 0; NOT_A_CELL 3; AMBIGUOUS 0.

HET_3:

| cell | verdict |
|---:|---|
| 00 | OK |
| 01 | OK |
| 02 | NOT_A_CELL |
| 03 | OK |
| 04 | OK |
| 05 | OK |
| 06 | OK |
| 07 | OK |
| 08 | NOT_A_CELL |
| 09 | OK |
| 10 | OK |
| 11 | NOT_A_CELL |
| 12 | OK |
| 13 | NOT_A_CELL |
| 14 | NOT_A_CELL |

Totals: OK 10; CENTER_OFF 0; NOT_A_CELL 5; AMBIGUOUS 0.

## Section 3: top fix recommendation

Tighten the merge rule so a weaker seed can be absorbed only when the two centers lie on the same connected bright soma/process component with an uninterrupted high-intensity bridge and no dark gap between compact blobs; otherwise keep both seeds until a later duplicate-suppression pass.
