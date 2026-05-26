**TL;DR**
- Yes: two strong soma seeds in one foreground component can be merged when their skeleton path lacks a strict thin neck (`v30d_run.py:503-511`).
- A 0.45 vs 0.55 pair is not always safely absorbed: if both are strong, the lower score can become `merged_into_strong`; if the 0.45 seed is weak, it can become `process_peak` (`v30d_run.py:483-487`, `v30d_run.py:505-511`).
- `bfs_path()` returning `None` is handled safely; the risks are false merging/skipping and order dependence, not a crash (`v30d_run.py:470-472`, `v30d_run.py:498-500`).

**Confirmed bugs**
- Strong-strong valid somas are merged by default when no neck is proven. Trace: two real cells 15 px apart, both initially `strong` (`v30d_run.py:311-312`), same foreground component and with snaps (`v30d_run.py:436-442`), path exists and is under `GEODESIC_LIMIT=80` (`v30d_run.py:496-500`). A short thick bridge gives `has_neck=False` because `neck_ratio < 0.55` and `thin_run >= 5` are not both true (`v30d_run.py:503-504`). The lower-score seed is set to `merged_into_strong` (`v30d_run.py:505-511`). For the stated biology, this should not be an automatic merge when both seeds have valid somas.
- Weak-strong demotion ignores `thin_run`. Trace: seed A scores 0.45 but fails `strong_ok`, so it is `weak` (`v30d_run.py:305-314`); seed B scores 0.55 and is `strong`. If their best path has `neck_ratio=0.57` and `thin_run=0`, Step A still demotes A because the only condition is `stats['neck_ratio'] < NECK_RATIO_DEMOTE` (`v30d_run.py:481-487`). This contradicts the comment saying “thin path” (`v30d_run.py:482`) and can erase a real weak soma.
- Multiple weak cells in one component with no strong collapse to one kept cell. Trace: two small Kolmer-like weak seeds share a binary component, `strongs` is empty (`v30d_run.py:450-453`), the best weak becomes `low_confidence_soma`, and every other weak becomes `process_peak` with `demoted_to` (`v30d_run.py:454-462`). Since `low_confidence_soma` is accepted (`v30d_run.py:522-524`), this keeps only one real small cell.

**Likely bugs**
- Hypothesis: snap-to-skeleton can leak off-soma in dense meshwork. The score center first snaps to a local intensity peak within 5 px (`v30d_run.py:139-149`, `v30d_run.py:260-262`), then `nearest_skel_pix()` searches skeleton pixels out to `max_search=10` without checking soma-core membership or foreground-component membership of the skeleton pixel (`v30d_run.py:357-370`, `v30d_run.py:431-434`).
- Hypothesis: Step B can reach order-dependent survivors. The group is score-sorted (`v30d_run.py:446-447`), then the first non-neck pair is merged and the loop restarts (`v30d_run.py:494-518`). Example: A-B no neck, B-C no neck, A-C has neck. If A-B is encountered first, B is removed before B-C is tested, so C survives.
- No confirmed re-merge after a type change: the dropped seed is removed from `kept_strong` (`v30d_run.py:509-512`). However, confirmed pairs before a later merge can be appended again after restart (`v30d_run.py:515-518`).

**Threshold concerns**
- The split rule is strict for 5-10 px soma diameters: `THIN_LEN_REQUIRED=5`, `PAIR_TRIM=4`, and `NECK_RATIO_SPLIT=0.55` (`v30d_run.py:70-75`) plus `thin_thr = min(0.5 * rmin_end, 2.5)` (`v30d_run.py:411-422`) require five consecutive skeleton pixels with radius roughly below 1.25-2.5 px. In a few-pixel inter-soma gap, that evidence may not exist even for separate cells.
- `NECK_RATIO_DEMOTE=0.60` is even looser for weak-strong demotion and has no `thin_run` guard (`v30d_run.py:70`, `v30d_run.py:483-487`).

**One concrete fix**
Apply the conservative soma-preservation fix first: in Step B, make `not has_neck` mark the pair ambiguous and keep both strong seeds, rather than merging the lower-score seed. Only merge strong-strong seeds when there is explicit duplicate evidence outside this neck test, such as the same snapped skeleton location or overlapping soma cores. Then add a `thin_run` requirement to Step A before demoting weak seeds.
