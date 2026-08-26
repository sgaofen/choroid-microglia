"""Compute-precision control for the cpstab float domain (PORT EXTENSION).

No MATLAB counterpart: MATLAB computed every image operation in `double`
(class-preserving casts back to uint16 aside), so the *replicate* path of this
port is hard-wired to float64 and stays there. This module adds an OPT-IN
"fast mode" in which the float working class of the pipeline becomes float32:
memory traffic halves and numpy/scipy pick the single-precision kernels
(np.fft.fft2 of a float32 array returns complex64, scipy.ndimage
interpolates in float32), at the cost of small numeric differences.

THE RULE
--------
    Fast mode lowers the class of PIXELS. It never lowers the class of the
    correlation arithmetic that DECIDES A SHIFT.

So `get_compute_dtype()` governs image storage, interpolation, accumulators
and the projection; `get_correlation_dtype()` / `get_complex_dtype()` govern
the DFT registration engine and are float64 / complex128 in BOTH modes.

That split is not conservatism, it is forced by a measurement on the
validation subset (FAD-F_1_T0-39, 128x128 downsampled frames, 41 planes x 20
volumes per chunk). Every shift in this pipeline is an ARGMAX over a phase
correlation surface, and on this data that surface is pathologically flat:

    DC pedestal / peak height ................. 0.909  (median)
    winning peak -> runner-up, relative gap ... 4.5e-04 (median)
                                                5.6e-06 (worst seen)
    float32 summation noise over the 256x256
    padded inverse transform .................. ~3e-05 relative

i.e. single-precision noise is of the same order as the margin that picks the
winner. The frames are dim (max 669 counts, mean 22) so almost all of the
correlation magnitude is the DC term and the actual alignment signal lives in
the last four digits. Running the engine in complex64 therefore does not
perturb a shift by an ulp — it flips the argmax to a different grid cell, and
the shift jumps by a whole 1/usfac step. Measured end-to-end with the engine
in complex64 (everything else identical):

    RS/CS per-plane shifts .... max |delta| 15.5 px (of a 512 px frame),
                                 68% of entries changed
    output projection ......... max |delta| 126 counts, Pearson r = 0.756

Compare the shipped boundary (engine kept in double), same subset:

    RS/CS/ZS + chunk corrections .. bit-identical to the float64 run
    zproj_mean, float level ....... max |delta| 6.9e-06 on a 0..226 range
                                    (5.4e-08 relative), r = 0.999999999999977
    written uint16 projection ..... bit-identical
    wall clock .................... 55.9 s -> 40.0 s  (1.33-1.40x)

Both configurations are reproduced by cpstab/tests/test_f32.py, which
measures the shipped one and asserts the complex64 one is NOT what ships.

WHAT THIS MODULE GOVERNS
------------------------
`get_compute_dtype()` is the dtype every cpstab module uses when raw image
data CROSSES INTO the float domain (`np.zeros(...)` accumulators, the
`astype(float64)` at the head of an interpolation, the class handed out by
VolumeSource.get_volume). It deliberately does NOT govern

  * the DFT correlation engine — see THE RULE above;
  * the shift bookkeeping (RS/CS/ZS and their chunk corrections, the
    intermediate per-plane shift matrices, the .dftshifts.npz payload). Those
    are O(Nz x Nt) scalars — no measurable cost, and they are the file the
    apply stage replays, so they stay float64 in both modes;
  * the final `matlab_uint16()` cast of the projection (pipeline.py), which
    stays float64: it is a one-shot cast whose half-away-from-zero rounding
    must not inherit float32 representation error on top of the already
    float32 pixel values;
  * MATLAB's own class-preserving casts (matlab_cast_like): a uint16 chain
    stays uint16 in both modes, which is what reproduces MATLAB's
    re-quantization points.

CONTRACT
--------
`get_compute_dtype()` is float64 unless a caller changed it. When it IS
float64 every dtype-aware call site below reduces to the literal expression
it replaced (`np.zeros(shape)` == `np.zeros(shape, np.float64)`,
`np.asarray(x, float64)` unchanged), so the replicate path is bit-for-bit
what it was — that is the iron law and it is enforced by the two regressions
in cpstab/tests/test_f32.py.

The setting is a MODULE-LEVEL GLOBAL, not a threaded argument. Rationale:
the float boundary is ~20 call sites spread over six modules and three
private helpers that fast_run.py calls DIRECTLY (orchestrator._process_chunk,
apply_project._process_volume/_apply_shifts_volume/_project); threading a
dtype through all of them would have meant changing every one of those
mirrored MATLAB signatures. A global is also the right granularity for the
multiprocessing driver: each worker process sets it once on entry
(`set_compute_dtype(name)`) and every module in that process follows.

NOT thread-safe by design (the pipeline is process-parallel, never
thread-parallel). Use `compute_dtype_scope()` to set-and-restore.
"""

import contextlib

import numpy as np

__all__ = [
    "SUPPORTED_DTYPES",
    "resolve_compute_dtype",
    "get_compute_dtype",
    "get_correlation_dtype",
    "get_complex_dtype",
    "set_compute_dtype",
    "compute_dtype_scope",
    "as_float",
    "as_correlation",
    "zeros",
    "work_dtype_for",
]

#: The two working classes the port supports. float64 = the MATLAB replicate
#: precision; float32 = the fast mode.
SUPPORTED_DTYPES = ("float64", "float32")

_VALID = (np.dtype(np.float64), np.dtype(np.float32))

#: Class of the phase-correlation arithmetic. Fixed, in BOTH modes -- see
#: THE RULE in the module docstring. Kept as named constants rather than
#: inlined float64/complex128 so that the (measured) reason they are pinned
#: has exactly one place to live, and so a future caller with well-conditioned
#: data has one obvious knob to reconsider.
_CORR_DTYPE = np.dtype(np.float64)
_CORR_COMPLEX = np.dtype(np.complex128)

_DTYPE = np.dtype(np.float64)


def resolve_compute_dtype(dtype):
    """Normalize a user-supplied compute dtype to a np.dtype, or raise.

    Accepts 'float64'/'float32', np.float64/np.float32, np.dtype objects.
    """
    d = np.dtype(dtype)
    if d not in _VALID:
        raise ValueError(
            "compute dtype must be one of %s (got %r); float64 is the "
            "MATLAB-replicate precision, float32 the fast mode."
            % (list(SUPPORTED_DTYPES), dtype))
    return d


def get_compute_dtype():
    """The float class raw image data is promoted to (default float64)."""
    return _DTYPE


def get_correlation_dtype():
    """Float class the DFT engine's FFT inputs are promoted to. ALWAYS float64.

    This is the second half of THE RULE (module docstring): a shift is an
    argmax over a correlation surface whose winning margin, on this data, is
    of the same order as float32 summation noise, so the engine keeps double
    in fast mode too. Everything it returns is a shift, never a pixel, so
    nothing downstream is promoted by this.
    """
    return _CORR_DTYPE


def get_complex_dtype():
    """Complex class of the DFT engine. ALWAYS complex128 (see above)."""
    return _CORR_COMPLEX


def set_compute_dtype(dtype):
    """Set the process-wide compute dtype; returns the PREVIOUS np.dtype."""
    global _DTYPE
    prev = _DTYPE
    _DTYPE = resolve_compute_dtype(dtype)
    return prev


@contextlib.contextmanager
def compute_dtype_scope(dtype):
    """`with compute_dtype_scope('float32'):` — set on entry, restore on exit.

    Restoration happens even on exception, so a failed run cannot leave the
    interpreter in fast mode for the next one.
    """
    prev = set_compute_dtype(dtype)
    try:
        yield get_compute_dtype()
    finally:
        set_compute_dtype(prev)


def as_float(a):
    """`np.asarray(a, dtype=compute_dtype)` — the float-domain entry point.

    Replaces the literal `np.asarray(x, dtype=np.float64)` at the head of
    every interpolation / FFT / projection helper. No copy when `a` already
    has the compute dtype.
    """
    return np.asarray(a, dtype=_DTYPE)


def as_correlation(a):
    """`np.asarray(a, float64)` at the head of a registration FFT.

    The counterpart of as_float() for the estimation side. In replicate mode
    the two are the same call; in fast mode this is the ONE place a float32
    pixel array is promoted back to double, and it is deliberate.
    """
    return np.asarray(a, dtype=_CORR_DTYPE)


def zeros(shape):
    """`np.zeros(shape, dtype=compute_dtype)` — float-domain accumulators."""
    return np.zeros(shape, dtype=_DTYPE)


def work_dtype_for(in_dtype):
    """Working class for a class-PRESERVING MATLAB op (imresize & friends).

    MATLAB computes imresize/imtranslate/imwarp in double and casts the
    result back to the input class. Here the internal arithmetic follows the
    compute dtype — EXCEPT when the input is already float64, which is then
    kept. That exception is load-bearing rather than cosmetic: the only
    float64-in/float64-out use of matlab_imresize in this pipeline is the
    'nearest' stretch of the inter-chunk shift vectors (DFT_warp_3D_2.m
    L127-129, orchestrator.dft_warp_3d_2), i.e. shift BOOKKEEPING, which this
    module deliberately leaves in double in both modes. Image data arrives as
    uint16 or as an array already in the compute dtype, so it follows the
    setting as intended.
    """
    d = np.dtype(in_dtype)
    if d == np.dtype(np.float64):
        return d
    return _DTYPE


# DESIGN NOTES
# ------------
# 1. Why a global instead of `cfg` drill-down: see the module docstring. The
#    decisive constraint is fast_run.py, which bypasses run_pipeline and calls
#    the private per-chunk / per-volume helpers directly in worker processes;
#    those helpers mirror MATLAB signatures verbatim (package rule 8) and
#    adding a dtype parameter to each would have broken that mirror in eight
#    places. RegistrationConfig.compute_dtype remains the user-facing knob —
#    pipeline.run_pipeline simply installs it through compute_dtype_scope().
# 2. Why float64 is the DEFAULT of the global and not merely of the config:
#    so that anything importing cpstab modules directly (validate.py, the
#    scratch_*.py probes, tests, third-party callers) keeps replicate numerics
#    without knowing this module exists.
# 3. Scope is per-PROCESS. ProcessPoolExecutor workers start with the default
#    float64 and must call set_compute_dtype() themselves — fast_run.py ships
#    the dtype name inside each job tuple and sets it at the top of the
#    worker. Forgetting that would silently run the workers in float64 (slow
#    but correct), never the reverse.
# 4. Not thread-safe: a second thread flipping the global mid-run would
#    produce a mixed-precision chunk. The pipeline is process-parallel only,
#    so this is a documented non-goal rather than a latent bug.
