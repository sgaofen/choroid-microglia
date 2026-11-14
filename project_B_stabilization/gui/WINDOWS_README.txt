Shipley stabilizer (Windows)
============================

Stabilizes two-photon z-stack time series (.oir) with the Shipley 2020
pipeline, ported to Python. Nothing to install: no MATLAB, Fiji or Java.

HOW TO USE
1. Copy this whole folder to the computer (a local drive, not a network drive).
2. Double-click "Start Shipley Stabilizer.bat".
3. Click "Add..." and pick the main .oir file of each run
   (e.g. 20230620_FAD-F_1.oir). The _00001, _00002 ... files next to it are
   read automatically, so keep them in the same folder.
4. Pick an output folder, then click Start.

OUTPUT  (one subfolder per run)
  <run>_replicate_zproj.tif   stabilized z-projection, ImageJ hyperstack
                              (T, C, Y, X) uint16; opens in Fiji/ImageJ
  <run>_replicate_shifts.npz  registration shifts (for re-use / checking)
  <run>_replicate_traj.npz    frame-to-frame trajectory of the projection
  <run>_replicate_log.txt     what happened, with timings
("improved" replaces "replicate" in the names when that mode is chosen.)

MODES
  Replicate  same algorithm as the original MATLAB pipeline.
  Improved   less residual drift and a larger usable field of view.

NEEDS
  Disk: about 1.1 x the size of the .oir run, free in the output folder,
        while it runs (a temporary copy, deleted when done; the result
        itself is ~1.6 GB for a 1500-frame run).
  RAM:  about 5.5 GB per parallel worker, and 17 GB at the final step.
        The worker count is preset from the free memory. If the computer
        slows to a crawl, close other programs or lower the worker number.

If something goes wrong, run "Start (show errors).bat" instead; it keeps a
window with the error messages open. Cancelled or failed runs keep their
temporary files in <run>\_work and pick up from there on the next Start.

Checked on a full 1500-frame run (61 .oir files, 65 GB): the output is
bit-for-bit identical to the validated Bio-Formats + Python route.
