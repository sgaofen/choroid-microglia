# Shipley stabilizer — desktop app

A one-window front end for `../pyport` (cpstab): pick `.oir` runs, pick an
output folder, press Start. Reads Olympus `.oir` directly in Python
(`oirread.py`, no Java/Bio-Formats needed).

**Windows users:** download `ShipleyStabilizer.zip` from the
[Releases page](https://github.com/sgaofen/choroid-microglia/releases),
unzip, double-click `Start Shipley Stabilizer.bat`. Details in
`WINDOWS_README.txt`.

**From source (any OS):** `pip install numpy scipy tifffile`, then
`python app.py` (or `python shipley_run.py RUN.oir OUT_DIR` without a window).

Checks done on a full 1500-frame run (61 `.oir` files, 65 GB):
- `.oir` read directly → replicate z-projection is bit-for-bit identical to
  the validated Bio-Formats route;
- cancel / resume / skip-finished tested through the real window (`test_gui.py`);
- memory: ~5.3 GB per worker (registration), ~17 GB peak in the main process
  at the end; the app presets the worker count from free RAM.

`build_windows.sh` rebuilds the portable Windows bundle from this folder.
