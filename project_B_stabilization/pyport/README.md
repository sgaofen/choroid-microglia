# cpstab

Python port of the Shipley 2020 two-photon z-stack stabilization pipeline
(MATLAB). Two versions, one switch.

- **replicate** — same algorithm as the original MATLAB pipeline.
- **improved** — four corrections on top: larger usable field, sharper frames,
  less residual drift.

No MATLAB, no Java, no Fiji. Python 3.9 + numpy/scipy/tifffile.

## Install

```bash
pip install numpy scipy scikit-image tifffile
export PYTHONPATH=/path/to/pyport
```

## Run

```bash
# one experiment run: .oir -> stabilized z-projection + metrics report
scripts/process_run.sh <oir-chunk-dir> <out-dir>
```

Or step by step:

```bash
python scripts/relayout.py --in RAW.ome.tif --out RAW.tzcyx.npy
python fast_run.py --raw RAW.tzcyx.npy --out-dir OUT --workers 10 [--mode improved]
python -m cpstab.metrics --stride 8 run=OUT/*_mean_zproj.tif
```

Library use:

```python
from cpstab import RegistrationConfig, run_pipeline
zproj = run_pipeline(RegistrationConfig(input_path="RAW.tzcyx.npy", mode="replicate"))
```

## Results

On a 1500-volume run (65 GB, 512x512x41z x 2 channels, 55 min of imaging):

| | original MATLAB setup | replicate | improved |
|---|---|---|---|
| wall clock (compute) | est. 1-6 h | **2.2 min** | 3.8 min |
| + one-off relayout | — | 0.9 min | 0.9 min |
| intermediate files on disk | 3-4 full copies | 0 | 0 |
| residual motion (steady state) | 0.33 px | 0.28 px | **0.19 px** |
| usable field (black border) | 9.9% | 9.5% | **3.4%** |

Timings are on an 18-core laptop, 10 worker processes, with the volume store
already in the page cache; a cold read adds a minute or two. `replicate`
reproduces the reference port output **bit for bit** at every scale tested;
against the lab's own MATLAB product it matches structurally (r = 0.987) with
sub-pixel trajectory differences that do not affect measurements.

## Layout

```
cpstab/           the library (see cpstab/README.md for the full technical doc)
fast_run.py       multiprocessing driver
scripts/          relayout, batch driver, benchmark ladder
matlab_bench/     package for timing the ORIGINAL pipeline on a MATLAB machine
```

## Tests

```bash
python cpstab/tests/test_synthetic.py     # end-to-end on generated data
python -m pytest cpstab/tests/ -q         # full suite
```
