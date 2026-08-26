"""Mode switchboard for the cpstab 'improved' path (PORT EXTENSION).

No MATLAB counterpart. This module owns ONE question — "is this run allowed to
diverge from the MATLAB original, and in which of the four documented ways?" —
and answers it for every other module in the package.

    replicate (DEFAULT)   bit-for-bit the validated port. Every `use_*()` below
                          returns False, every guarded branch takes the literal
                          code that was there before this feature existed, and
                          the two iron-law regressions (tests/test_synthetic.py
                          7/7 and the 40-volume subset vs
                          reportA/port_run/FAD-F_1_T0-39_mean_zproj.tif) hold.
    improved              all four corrections below, on.

THE FOUR CORRECTIONS
--------------------
1. `global_median` — apply_project.make_sbxall (MakeSBXall.m L29-L35).
   MATLAB centres RS_total/CS_total with `median()` at its DEFAULT dim, which
   on the (Nz, Nt) shift matrices is a PER-COLUMN (per-timepoint) median. Every
   shift term that is CONSTANT ALONG THE PLANE AXIS is therefore subtracted
   back out, exactly and identically, at every timepoint:

       RS_total - median(RS_total, dim=1)
         = (RS + RS_chunk) - median(RS + RS_chunk, dim=1)
         = RS - median(RS, dim=1)                       (RS_chunk drops out)

   and RS itself is RS0*scale + RS1*scale + repmat(RS2', Nz, 1) — so the
   tiled RS2 term drops out the same way. Two of the three registration stages
   are thus algebraically annihilated before a single pixel moves:

       * the per-volume 3-D registration against ref2 (DFT_warp_3D_2.m L71-79,
         RS2/CS2), and
       * the inter-chunk stitch that anchors every chunk to chunk 1
         (DFT_warp_3D_2.m L110-118, RS_chunk/CS_chunk).

   Verified on the 40-volume validation subset, where RS_chunk/CS_chunk are
   plane-constant by construction (they are an imresize 'nearest' stretch of a
   1 x Nchunks vector) and the cancellation is exact to 3.6e-15 px. Measured
   consequence on that subset: the volume-level RS trace has a real -12..-15 px
   excursion over t=15..24 (the subject moves); per-column centring turns it
   into a +10.8 px excursion of the opposite sign, and flattens the CS
   excursion from -12.7 px to -0.1 px.

   The fix centres on a GLOBAL scalar median over the whole matrix, which
   removes the arbitrary absolute origin (the only thing centring is for) while
   leaving every relative motion intact. ZS is unaffected in either mode: it is
   a (1, Nt) row vector, where MATLAB's default-dim median is already the
   scalar median of all elements.

2. `fourier_shift` — the APPLY side of the XY translation
   (apply_project._apply_shifts_volume, and the final per-frame translation of
   zproj_reg). Bilinear interpolation is a low-pass filter: every fractional
   shift costs resolution, and the pipeline applies one per plane per volume.
   Replaced by a phase-ramp translation carried out in the SQRT domain
   (cpstab/fourier_shift.py fshift2_vst). The ESTIMATION side (DFT_rect /
   DetermineXYShiftsFBS's internal imtranslate) is deliberately NOT touched,
   so the shifts this mode estimates are the same ones replicate estimates
   (correction 1 and 3 aside).

   WHY THE SQRT DOMAIN AND NOT THE PLAIN RAMP
   ..........................................
   The plain ramp is the exact band-limited translation, and this data is not
   band-limited: photon counts on a near-zero background carry real energy at
   and above Nyquist, so the ideal reconstruction rings around every
   single-pixel feature. Measured on one interior frame of the validation
   subset, a 0.37/-1.62 px ramp shift leaves 25.6% of the pixels NEGATIVE, the
   most negative at -373 counts, with |negative mass| = 20% of the frame's
   total intensity — and 93% of those negatives are more than 20 px from the
   border, i.e. this is not the wrap seam that fourier_shift.py's original
   docstring measured (that measurement used a Gaussian-SMOOTHED frame, which
   had already removed the content that rings).

   Most of it cancels in the z-projection, but not all: 0.145% of the
   projection's pixels come out negative, and the uint16 cast turns each one
   into a black speckle surrounded by real data. Isolated internal zeros on
   the 40-volume subset, 80 frames: replicate 3, plain ramp 33480, sqrt-domain
   ramp 5712 (deep interior, >10 px from the black band: 0 / 18398 / 340).

   Nonnegativity cannot be restored afterwards. Clipping the plane at 0 costs
   +14.0% of the projection's total intensity, clamping to the 2x2 source
   bracket +10.7%, and a spectral taper wide enough to remove the negatives is
   a worse low-pass than the bilinear being replaced — because a kernel that
   cannot make a negative from nonnegative input IS a nonnegative kernel, and
   a nonnegative kernel is a low-pass. Shifting sqrt(x) and squaring back is
   nonlinear, so it escapes that; it is also the variance-stabilizing
   transform for Poisson data and it compresses exactly the local contrast
   that drives the ringing. Cost on the subset: resid px median 0.0224 ->
   0.0316 and sharpness 0.0176 -> 0.0126 against the plain ramp (replicate:
   0.0636 / 0.0063), total intensity preserved to 0.08%.

   This is not a separate switch — see DESIGN NOTE 6, and fourier_shift.py
   DESIGN NOTES 7 and 8.

3. `chain_refine` — orchestrator._process_chunk (DFT_warp_3D_2.m L52-57).
   DFT_rect rectifies a volume by CHAINING: plane i is registered to the
   translated plane i-1, outward from the centre plane. Chained estimates
   random-walk — each link's error is added to every link after it, so the
   planes at the two ends of the stack drift away from the anchor with no
   restoring force. One global pass (every plane vs the mean of the
   chain-aligned volume, dftregistration_alex usfac=4) supplies that restoring
   force; the correction folds into RS0/CS0 and the volume is rebuilt with ONE
   interpolation from the original planes at the corrected total shift.

   THE TRUST GATE (`chain_refine_cap` / `chain_refine_min_ncc`)
   ...........................................................
   A restoring force is only a force if the measurement it is built on means
   something. On real stacks the DEEP planes have almost no content in common
   with the volume mean — measured on FAD-F_1 (scale 4, 41 planes): Pearson
   between a plane and the volume mean runs 0.30-0.76 for z <= 23 and
   0.039-0.245 for z >= 24, where the correlation surface is essentially flat
   (peak-to-sidelobe 1.01-1.04). An argmax over a flat surface is a uniform
   draw from the whole +-N/2 search domain, and the refinement above used to
   fold that draw into RS0/CS0 unconditionally: at t=1300 (a QUIET timepoint)
   17 of 41 planes were "corrected" by 95-225 full-resolution px, blowing the
   volume's RS range out to [-10.4, +218.4] px against replicate's
   [-9.2, +19.0]. The zero-fill bands of those planes are the rectangular
   seams and ghosts visible in the projection, and the damage compounds: the
   volume is rebuilt at the bogus shift, so DetermineXYShiftsFBS then measures
   a mostly-black frame and CS1's spread went 4.1 px -> 189 px. Over the full
   run 35.3% of (plane, timepoint) cells exceeded 50 px, in 1492 of 1500
   timepoints.

   So each plane's correction is now ACCEPTED ONLY IF it is both small and
   backed by a real peak; otherwise the plane keeps its chain value:

       accept  iff  max(|dR|, |dC|) <= chain_refine_cap
                    and NCC_zm(plane, volume mean) >= chain_refine_min_ncc

   Both thresholds are read from this module by
   orchestrator._refine_chain_to_volume_mean, and both are set from
   RegistrationConfig (cfg.chain_refine_cap / cfg.chain_refine_min_ncc).

   Units: `chain_refine_cap` is in REGISTRATION-GRID px, the units of the
   dftregistration_alex output at that call site (a 128x128 grid at
   scale=4); multiply by cfg.scale for full-resolution px.

   Where the default 3.0 comes from — the same 4100-sample survey (5 chunks
   of FAD-F_1), correction magnitude in grid px:

       trustworthy planes (Pearson >= 0.40, n=2141) ... p99 1.0, max 2.0
       trustworthy planes (Pearson >= 0.30, n=2504) ... p99 1.0, max 8.5 (1)
       runaway cluster (n=1218, 29.7%) ............... min 12.75, median 46.9

   i.e. the distribution is bimodal with an empty valley between 2.0 and
   12.75 grid px; 3.0 is 1.5x the largest honest correction ever observed and
   still sits deep inside the valley (only 29 of 4100 samples fall in
   (2.0, 12.75), every one of them a deep low-correlation plane). 4.0 would
   change nothing. The NCC gate is the second, independent line: it separates
   cleanly at 0.30 (runaway Pearson max 0.296) and is what catches a mislock
   that happens to land NEAR the origin, which a pure magnitude cap cannot.

   REJECTION MEANS dR = dC = 0 — keep DFT_rect's chain value — and not
   clamping to the cap: clamping still injects a fabricated shift of cap
   magnitude in a random direction. Falling back to the chain is the only
   fallback with a proof attached, because it is exactly what replicate does
   for that plane, so a fully-rejected volume degrades to the MATLAB-faithful
   result and can never be worse than the status quo.

4. `subplane_z` — the APPLY side of the Z shift
   (apply_project._apply_shifts_volume, MakeSBXall.m L111-L118). MATLAB rounds
   ZS to a whole plane and circshifts, discarding up to half a plane of axial
   registration that dftregistration3D measured at usfac=2 (i.e. to 0.5 plane),
   and then zeroes the wrapped band ASYMMETRICALLY (Z<0 clears one plane more
   than it wrapped). Replaced by integer circshift + linear interpolation
   between the two neighbouring planes, with honest zero fill at both ends.

USAGE
-----
    RegistrationConfig(..., mode='improved')      # the user-facing knob
    with improved.mode_scope('improved'): ...     # what run_pipeline installs

For ABLATION, each correction has an independent override that beats the mode
in both directions:

    with improved.feature_scope(chain_refine=False):   # improved minus #3
    with improved.feature_scope(global_median=True):   # replicate plus #1

Like cpstab/precision.py this is a per-PROCESS module global rather than a
threaded argument, and for the same reason: fast_run.py bypasses run_pipeline
and calls the private per-chunk / per-volume helpers directly in worker
processes, whose signatures mirror MATLAB's (package rule 8). Each worker
installs the mode once on entry. NOT thread-safe by design.
"""

import contextlib

__all__ = [
    "MODES",
    "FEATURES",
    "resolve_mode",
    "get_mode",
    "set_mode",
    "mode_scope",
    "enabled",
    "set_features",
    "feature_scope",
    "use_global_median",
    "use_fourier_shift",
    "use_chain_refine",
    "use_subplane_z",
    "CHAIN_REFINE_CAP",
    "CHAIN_REFINE_MIN_NCC",
    "resolve_chain_refine_cap",
    "resolve_chain_refine_min_ncc",
    "chain_refine_cap",
    "chain_refine_min_ncc",
    "set_chain_refine_guard",
    "chain_refine_guard_scope",
]

#: The two pipelines this package can run. 'replicate' is the default and is
#: what the validation report covers.
MODES = ("replicate", "improved")

#: The four corrections, each independently switchable for ablation. Order is
#: the order they appear in the pipeline, not the order of the docstring.
FEATURES = ("global_median", "chain_refine", "fourier_shift", "subplane_z")

_MODE = "replicate"
_OVERRIDES = {}

#: Correction 3's trust gate, in REGISTRATION-GRID px (x cfg.scale for
#: full-resolution px). A per-plane refinement larger than this is a mislock,
#: not a measurement -- see the derivation in the module docstring. Set
#: `float('inf')` to disable the magnitude half of the gate.
CHAIN_REFINE_CAP = 3.0

#: Correction 3's second gate: the zero-mean normalized correlation between a
#: plane and the volume mean it is being measured against. Below this the
#: correlation surface is flat and the argmax is noise. 0.0 disables it.
CHAIN_REFINE_MIN_NCC = 0.30

_CHAIN_REFINE_CAP = CHAIN_REFINE_CAP
_CHAIN_REFINE_MIN_NCC = CHAIN_REFINE_MIN_NCC


def resolve_mode(mode):
    """Normalize/validate a mode name, or raise.

    Kept here rather than in config.py so the accepted spellings cannot drift
    between the config validator and the thing that consumes them (the same
    arrangement precision.resolve_compute_dtype has with cfg.compute_dtype).
    """
    if mode is None:
        return "replicate"
    m = str(mode).strip().lower()
    if m not in MODES:
        raise ValueError(
            "mode must be one of %s (got %r); 'replicate' is the MATLAB-"
            "faithful default, 'improved' enables the four corrections "
            "documented in cpstab/improved.py." % (list(MODES), mode))
    return m


def get_mode():
    """The process-wide mode (default 'replicate')."""
    return _MODE


def set_mode(mode):
    """Set the process-wide mode; returns the PREVIOUS mode name.

    Does NOT touch per-feature overrides — an ablation scope set up around a
    mode change survives it, which is what makes feature_scope composable.
    """
    global _MODE
    prev = _MODE
    _MODE = resolve_mode(mode)
    return prev


@contextlib.contextmanager
def mode_scope(mode):
    """`with mode_scope('improved'):` — set on entry, restore on exit.

    Restoration happens on exception too, so a failed improved run cannot
    leave the interpreter in improved mode for a later replicate run in the
    same process. This is what run_pipeline wraps a run in.
    """
    prev = set_mode(mode)
    try:
        yield get_mode()
    finally:
        set_mode(prev)


def enabled(feature):
    """Is `feature` active right now?

    An explicit override wins over the mode, in BOTH directions, so an
    ablation can subtract one correction from 'improved' or add one to
    'replicate'. With no override the answer is simply "are we improved".
    """
    if feature not in FEATURES:
        raise KeyError("unknown feature %r; known: %s" % (feature, list(FEATURES)))
    if feature in _OVERRIDES:
        return bool(_OVERRIDES[feature])
    return _MODE == "improved"


def set_features(**overrides):
    """Pin individual features on/off; None clears an override. Returns the
    PREVIOUS override dict (pass it back to set_features to restore)."""
    global _OVERRIDES
    prev = dict(_OVERRIDES)
    new = dict(_OVERRIDES)
    for k, v in overrides.items():
        if k not in FEATURES:
            raise KeyError("unknown feature %r; known: %s" % (k, list(FEATURES)))
        if v is None:
            new.pop(k, None)
        else:
            new[k] = bool(v)
    _OVERRIDES = new
    return prev


@contextlib.contextmanager
def feature_scope(**overrides):
    """`with feature_scope(chain_refine=False):` — ablate one correction."""
    global _OVERRIDES
    prev = set_features(**overrides)
    try:
        yield
    finally:
        _OVERRIDES = prev


def use_global_median():
    """Correction 1: scalar median centring of RS_total/CS_total."""
    return enabled("global_median")


def use_chain_refine():
    """Correction 3: global refinement pass after the DFT_rect chain."""
    return enabled("chain_refine")


def use_fourier_shift():
    """Correction 2: phase-ramp translation on the apply side."""
    return enabled("fourier_shift")


def use_subplane_z():
    """Correction 4: fractional-plane Z shift on the apply side."""
    return enabled("subplane_z")


# There is deliberately NO correction 5. A trust gate on the movie-level
# trajectory (flag deviations from a running median, rebuild by neighbour
# interpolation) was implemented and falsified the same day (2026-08-25):
# the 50-64 px single-frame trajectory deviations it targeted turned out to
# be REAL 1-2-frame tissue excursions that the estimator tracks to ~8 px of
# output residual; interpolating through them raised the residual to 41-46 px.
# Distribution-level plausibility is not output-level proof -- gate proposals
# for this trajectory must be validated against measured output residuals
# (see apply_project.zproj_reg, the comment above `tf = ...`).


# ---------------------------------------------------------------------------
# correction 3's trust gate (parameters, not a switch of their own)
# ---------------------------------------------------------------------------

def resolve_chain_refine_cap(value):
    """Normalize/validate a cap in registration-grid px, or raise.

    Same arrangement as resolve_mode above: the module that CONSUMES the
    setting owns its validator, so config.py and a direct
    set_chain_refine_guard() caller can never disagree about what is legal.
    """
    if value is None:
        return CHAIN_REFINE_CAP
    if isinstance(value, bool):
        raise TypeError("chain_refine_cap must be a number of grid px, got a bool")
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(
            "chain_refine_cap must be a number of registration-grid px "
            "(got %r)" % (value,)) from e
    if not (v > 0) or v != v:
        raise ValueError(
            "chain_refine_cap must be > 0 grid px (got %r); use float('inf') "
            "to disable the magnitude gate, not 0." % (value,))
    return v


def resolve_chain_refine_min_ncc(value):
    """Normalize/validate the correlation floor (0..1), or raise."""
    if value is None:
        return CHAIN_REFINE_MIN_NCC
    if isinstance(value, bool):
        raise TypeError("chain_refine_min_ncc must be a number in [0, 1), got a bool")
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(
            "chain_refine_min_ncc must be a number in [0, 1) (got %r)"
            % (value,)) from e
    if not (0.0 <= v < 1.0):
        raise ValueError(
            "chain_refine_min_ncc must be in [0, 1) (got %r); 0.0 disables "
            "the correlation gate." % (value,))
    return v


def chain_refine_cap():
    """The process-wide correction-3 magnitude cap, in grid px."""
    return _CHAIN_REFINE_CAP


def chain_refine_min_ncc():
    """The process-wide correction-3 correlation floor."""
    return _CHAIN_REFINE_MIN_NCC


def set_chain_refine_guard(cap=None, min_ncc=None):
    """Set the gate process-wide; returns the PREVIOUS (cap, min_ncc).

    None leaves that half alone (it does NOT reset it to the default), so a
    caller can move one threshold without knowing the other. Pass the returned
    tuple straight back to restore.

    ATOMIC: both arguments are validated BEFORE either global moves, so a bad
    second argument leaves the process-wide gate exactly as it was. It used to
    write the cap first and validate min_ncc second, which on a raise left the
    cap changed while `prev` was never returned -- an unrecoverable leak for
    the caller, and one chain_refine_guard_scope could not undo either (it
    calls this OUTSIDE its try, so its finally never ran). Same shape as
    set_mode / set_features above, which compute first and assign last.
    """
    global _CHAIN_REFINE_CAP, _CHAIN_REFINE_MIN_NCC
    prev = (_CHAIN_REFINE_CAP, _CHAIN_REFINE_MIN_NCC)
    new_cap = prev[0] if cap is None else resolve_chain_refine_cap(cap)
    new_ncc = (prev[1] if min_ncc is None
               else resolve_chain_refine_min_ncc(min_ncc))
    _CHAIN_REFINE_CAP = new_cap
    _CHAIN_REFINE_MIN_NCC = new_ncc
    return prev


@contextlib.contextmanager
def chain_refine_guard_scope(cap=None, min_ncc=None):
    """`with chain_refine_guard_scope(cap=8.0):` — set on entry, restore on
    exit (including on an exception), like mode_scope."""
    prev_cap, prev_ncc = set_chain_refine_guard(cap, min_ncc)
    try:
        yield (chain_refine_cap(), chain_refine_min_ncc())
    finally:
        set_chain_refine_guard(prev_cap, prev_ncc)


# DESIGN NOTES
# ------------
# 1. Why a global instead of drilling `cfg` down: identical to the reasoning in
#    cpstab/precision.py DESIGN NOTES #1. fast_run.py calls
#    orchestrator._process_chunk and apply_project._process_volume /
#    _apply_shifts_volume / _project directly from worker processes; those
#    helpers mirror MATLAB signatures verbatim and threading a mode through
#    each would break that mirror in eight places for no gain.
#    RegistrationConfig.mode stays the user-facing knob and run_pipeline
#    installs it with mode_scope().
# 2. Why 'replicate' is the default of the GLOBAL and not merely of the config:
#    so anything importing cpstab modules directly (validate.py, the scratch
#    probes, the test suites, third-party callers) keeps MATLAB-faithful
#    numerics without having to know this module exists. A missed
#    set_mode() in a worker degrades to replicate — wrong-but-obvious — never
#    to a silently half-improved run.
# 3. Scope is per-PROCESS, exactly like the compute dtype. fast_run.py ships
#    the mode name inside each job tuple and installs it on the worker's first
#    line, next to set_compute_dtype().
# 4. The overrides dict is deliberately separate from the mode rather than
#    being four more mode names ('improved_no_chain', ...): the ablation space
#    is 2^4 and enumerating it as modes would put sixteen spellings into the
#    config validator. feature_scope() also composes with mode_scope(), so a
#    test can hold one ablation across several runs.
# 5. enabled() lets an override turn a feature ON in replicate mode. That is
#    not a footgun to be closed off — it is how the per-correction attribution
#    in tests/test_improved.py works (each correction measured against
#    replicate on its own). The IRON LAW is about the DEFAULT path: mode
#    'replicate' with no overrides, which is what both regressions exercise.
# 6. The chain-refine gate is PARAMETERS of correction 3, not a fifth entry in
#    FEATURES. There is no "correction 3 without the gate" worth shipping —
#    ungated it is the bug documented above, so making it switchable would be
#    offering a mode whose only distinction is being broken. Ablation is still
#    possible for anyone reproducing that measurement (cap=float('inf'),
#    min_ncc=0.0 restores the old behaviour exactly), which is why the
#    thresholds are numbers with an explicit disable value rather than an
#    on/off flag. They live here, next to the mode, because
#    _refine_chain_to_volume_mean is reached from fast_run.py's workers, which
#    bypass run_pipeline (DESIGN NOTE 1) — the same reason the mode itself is
#    a process global.
