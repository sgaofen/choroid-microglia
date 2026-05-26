# v30c Independent Audit

| cell_id | verdict | pipeline_class | visual_process_count | algo_endpoint_count | brief_note |
|---|---|---:|---:|---:|---|
| cell00 L947 | OK | strong | 3-4 | 12 | Real soma-like blob; center OK; endpoints over-attributed. |
| cell01 L448 | AMBIGUOUS | strong | 0 | 0 | Isolated compact dot; Kolmer cell vs debris. |
| cell02 L612 | AMBIGUOUS | strong | 0 | 7 | Isolated round dot; no attached processes visible. |
| cell03 L518 | OK | strong | ~4 | 4 | Bright soma/junction; marker still on soma mass. |
| cell04 L523 | OK | strong | 5-6 | 6 | Ramified soma-like hub; center slightly low but acceptable. |
| cell05 L507 | OK | strong | ~5 | 6 | Clear soma with multiple thin processes. |
| cell06 L803 | AMBIGUOUS | strong | 0 | 0 | Isolated punctum; plausible small cell but no processes. |
| cell07 L657 | AMBIGUOUS | strong | 1-2 | 2 | Small bright bead/soma candidate on sparse signal. |
| cell08 L879 | NOT_A_CELL process bead | strong | 0-1 | 0 | Tiny bead on faint filament; no soma. |
| cell09 L637 | NOT_A_CELL junction | strong | 2-3 | 3 | Marker on process/vessel junction, not compact soma. |
| cell10 L585 | NOT_A_CELL filament | strong | ~2 | 5 | Linear tube/segment; no soma despite high fg density. |
| cell11 L840 | NOT_A_CELL vessel/junction | strong | 3-4 | 5 | Thick segment complex; marker not on discrete soma. |
| cell12 L687 | NOT_A_CELL process segment | strong | 1-2 | 3 | Faint elongated bead/segment, not a compact body. |
| cell13 L667 | AMBIGUOUS | strong | 0 | 0 | Isolated dot; possible Kolmer cell vs debris. |
| cell14 L705 | NOT_A_CELL beaded process | strong | 2-3 | 6 | Beaded vertical process; snap lands on lower bead. |

## Definition Validity

A. The soma definition is broadly reasonable for choroid plexus microglia, but isolated 5 px dots should be accepted only as low-confidence Kolmer candidates when not lying on a filament/chain. Dots on chains look like process beads. "Process count" should mean primary skeleton trunks leaving a soma collar, not total distal endpoints attributed through the graph.

B. The weights over-reward compact/blob/radius-prominence and under-penalize linear context. `fg_dens` is not discriminative because tubes and vessels are dense too. `sholl` should be based on primary annular exits around a validated soma, not just nearby skeleton directions.

C. Biggest remaining failure mode: bright process beads and vessel/process junctions survive as `strong`, then endpoint attribution assigns surrounding network branches to them.

## Top Recommended Fix

Add one post-snap soma-validation hard gate in `score_seed`: validate a compact local foreground component around the snapped center and count primary skeleton exits through an annulus. Keep true compact bodies, demote isolated valid 5 px dots to low-confidence Kolmer candidates, and reject linear through-going or beaded/junction components as `process_peak`.
