"""Run configuration for the cpstab stabilization pipeline.

Mirrors: registration/RegistrationMasterPipeline.m L8-27 (the hardcoded
identification + parameter block) restructured as an explicit config object,
following the design of clean/run_registration.m L14-43 and
clean/config_example.m (the MATLAB clean-up already replaced the
pipe.lab.* hostname->drive path chain with explicit cfg paths; this dataclass
is the Python analogue).

All numeric defaults are the values hardcoded in RegistrationMasterPipeline.m:
scale=4 (L22), chunksize=20 (L23), proj_type='mean' (L26), opttype='none'
(L14), refchannel=1 (L15).
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["RegistrationConfig", "matlab_round"]


def matlab_round(x):
    """MATLAB round(): round half AWAY FROM ZERO, returned as int.

    MATLAB round(7.5) == 8, round(22.5) == 23, round(-2.5) == -3.
    Python's built-in round() and np.round() are half-to-even
    (np.round(22.5) == 22), which silently changes chunk counts and
    projection ranges — hence this helper. Used for:
      * Nchunks = round(Nt/chunksize)            (RegistrationMasterPipeline.m L24)
      * round(0.25*Nz):round(0.75*Nz) proj range (MakeSBXall.m L13)

    Implemented by exact fractional-part comparison, NOT floor(abs(x)+0.5):
    for doubles just below a half-integer (e.g. nextafter(0.5, 0)) the
    addition itself IEEE-rounds up before floor, adding 1 (double rounding).
    a - floor(a) is exact (Sterbenz), so the >= 0.5 test is a true-value
    comparison. The two pipeline call sites above cannot hit that edge
    (integer/integer quotients and 0.25/0.75 products are exact doubles),
    but the helper is exported and must be correct for arbitrary input.
    (Verified against an exact-rational oracle: tests/scratch_f1_rounding.py.)
    """
    a = abs(x)
    f = math.floor(a)
    if a - f >= 0.5:
        f += 1
    return int(f) if x >= 0 else -int(f)


@dataclass
class RegistrationConfig:
    """Configuration for one stabilization run.

    Mirrors RegistrationMasterPipeline.m L8-27 / clean/config_example.m.

    Attributes
    ----------
    input_path : str
        Path to the raw stack (the port's VolumeSource decides what it
        accepts: an .sbx with its sidecar .mat next to it, or a TIFF
        export). Replaces the pipe.lab.datapath(...) resolution
        (RegistrationMasterPipeline.m L17-18).
    refchannel : int
        Registration reference channel, **1-based to preserve the MATLAB
        habit** (1 = red vessels / IV dye, 2 = green;
        RegistrationMasterPipeline.m L15). Conversion to 0-based happens
        inside the IO layer, never here — every module-boundary signature in
        this package carries the MATLAB 1-based value.
    scale : int
        Spatial downsample factor used during shift *estimation*
        (RegistrationMasterPipeline.m L22; shifts are scaled back up by
        DFT_warp_3D_2.m L77-84).
    chunksize : int
        Volumes per registration chunk; the original comment says
        "don't go over 20" (RegistrationMasterPipeline.m L23) and we
        enforce that here.
    proj_type : str
        'mean' | 'max' | 'median' (RegistrationMasterPipeline.m L26).
        NOTE: in the original this only affects the output *filename*; the
        projection actually computed is always the mean (MakeSBXall.m L120).
        See PORTING NOTES.
    proj_range : str | sequence of int
        Which z-planes enter the projection.
        'quarter' -> round(0.25*Nz)..round(0.75*Nz) inclusive — the
        MakeSBXall.m L13 default, i.e. what the published pipeline
        actually used.
        'full'    -> 1..Nz — the range the master script computed (L25)
        but never passed (CODEMAP §13 Q9).
        A sequence is taken as explicit 1-based inclusive plane indices.
    opttype : str
        'none' (piezo) | 'affine' | 'rigid' (optotune lens)
        (RegistrationMasterPipeline.m L14). Only 'none' is implemented in
        the port; the other two required Fiji/MultiStackReg/TurboReg.
    write_registered : bool
        Whether to also write the full registered stack (the original
        unconditionally wrote '<base>.sbxall', which nothing in the repo
        ever reads — CODEMAP §5 item G). Default False: the projection
        TIFF is the deliverable.
    out_dir : str, optional
        Directory for outputs (.dftshifts.npz shift file, the projection
        TIFF, and the optional registered stack). Defaults to the
        directory of input_path, matching where '<rundir>.*' outputs
        landed in the original.
    mode : str
        Which pipeline to run. PORT EXTENSION — no MATLAB counterpart.

        'replicate' (DEFAULT)
            Bit-for-bit the validated port of the MATLAB original. This is
            the only mode the ground-truth comparison covers, and the two
            iron-law regressions (tests/test_synthetic.py 7/7 and the
            40-volume subset vs reportA/port_run) assert it stays that way.
        'improved'
            Four documented corrections to the original algorithm, all on.
            In one line each — the full derivation, the measurements and the
            per-correction ablation switches live in cpstab/improved.py:
              1. RS/CS centring uses a GLOBAL scalar median instead of
                 MATLAB's per-timepoint one. The per-timepoint median
                 algebraically cancels every plane-constant shift term, which
                 silently discards BOTH the per-volume 3-D registration
                 (RS2/CS2) and the inter-chunk stitch (RS_chunk/CS_chunk) —
                 verified exact to 3.6e-15 px on the validation subset.
              2. The apply-side XY translation is an exact Fourier phase ramp
                 (cpstab/fourier_shift.py) instead of bilinear interpolation,
                 which is a low-pass filter applied once per plane.
              3. The DFT_rect plane-to-plane chain gets one global refinement
                 pass against the chain-aligned volume mean, supplying the
                 restoring force a pure chain lacks.
              4. The apply-side Z shift interpolates between neighbouring
                 planes instead of rounding to a whole plane, keeping the
                 half-plane precision dftregistration3D actually measured.
            Corrections 1 and 3 change the SHIFTS, so an improved run's
            .dftshifts.npz and its projection both differ from replicate;
            2 and 4 change only how those shifts are applied.
    chain_refine_cap : float
        Correction 3's trust gate, part 1 (PORT EXTENSION; IGNORED in
        'replicate' mode, where correction 3 does not run). Largest per-plane
        refinement that will be believed, in REGISTRATION-GRID px — the units
        of the estimate itself, i.e. full-resolution px / cfg.scale. A plane
        whose correction exceeds it keeps its DFT_rect chain value.

        Default 3.0 (= 12 full-resolution px at scale 4). The correction
        magnitudes on real data are bimodal: honest ones (plane-vs-mean
        correlation >= 0.40) have p99 1.0 and max 2.0 grid px, while the
        mislocks of the deep planes start at 12.75 and run to 61 grid px, with
        an all-but-empty valley between. Ungated, those mislocks moved planes
        by up to 225 full-resolution px at QUIET timepoints and put the
        rectangular seams into the projection — the full measurement is in
        cpstab/improved.py correction 3. float('inf') restores the ungated
        behaviour.
    chain_refine_min_ncc : float
        Correction 3's trust gate, part 2 (PORT EXTENSION; IGNORED in
        'replicate'). Minimum zero-mean normalized correlation between a plane
        and the volume mean for its refinement to be believed, in [0, 1).

        Default 0.30. This is the gate that catches a mislock landing NEAR the
        origin, which no magnitude cap can: on FAD-F_1 every runaway plane
        scored <= 0.296 while the planes with real shared content scored
        0.30-0.76. 0.0 disables it.
    compute_dtype : str
        Float working class of the pipeline's image domain. PORT EXTENSION —
        no MATLAB counterpart (MATLAB computed everything in `double`).

        'float64' (DEFAULT)
            The replicate precision: bit-for-bit identical to the validated
            port output, which is the only precision the ground-truth
            comparison was done in. Do not change it for a run whose numbers
            are meant to be compared against the MATLAB pipeline or against
            an earlier port run.
        'float32'
            Fast mode. Every raw frame enters the float domain as float32, so
            VolumeSource reads, scipy.ndimage interpolation, the projection
            accumulators and the zproj_reg refinement all run at half the
            memory traffic. The DFT correlation that decides each shift is
            deliberately NOT lowered — see cpstab/precision.py, which owns
            that boundary and documents the measurement that forced it.

            Measured on the 40-volume validation subset (FAD-F_1_T0-39,
            refchannel=1 scale=4 chunksize=20 proj_range=quarter, macOS /
            numpy 2.0.2, single process):

                wall clock ......... 55.9 s -> 40.0 s   (1.33-1.40x over
                                     repeated runs; the apply stage gains
                                     ~1.5x, the registration stage ~1.12x)
                .dftshifts.npz ..... bit-identical to the float64 run
                zproj_mean (float) . max|diff| 6.9e-06 on a 0..226 range,
                                     max relative diff 5.4e-08,
                                     Pearson r = 0.999999999999977,
                                     NRMSE 3.1e-09
                written uint16 TIFF  BIT-IDENTICAL (0 of 20,971,520 pixels)

            The bit-identity of the written TIFF is a RESULT on this dataset,
            not a promise: the float-level difference is ~5e-8 relative, so a
            pixel sitting within that of a .5 rounding tie can still come out
            one count different — expect O(1) such pixels per 20 M, and do
            not use fast mode where exact reproduction is the requirement.
            Nor is the identical shift file a guarantee: the discrete parts
            of the algorithm (argmax peak picking, the round(ZS) plane
            circshift) could tip differently at another SNR, and that
            difference would be a whole plane, not a count. Validate on a new
            dataset before trusting fast mode there.

        Shift bookkeeping (RS/CS/ZS and the .dftshifts.npz payload) stays
        float64 in BOTH modes — see cpstab/precision.py, which owns the
        boundary and is what run_pipeline installs this field into.
    """

    input_path: str
    refchannel: int = 1
    scale: int = 4
    chunksize: int = 20
    proj_type: str = "mean"
    proj_range: Union[str, Sequence[int]] = "quarter"
    opttype: str = "none"
    write_registered: bool = False
    out_dir: Optional[str] = None
    mode: str = "replicate"
    compute_dtype: str = "float64"
    # Correction 3's trust gate. Appended AFTER compute_dtype on purpose: the
    # field order of this dataclass is also a positional signature, so new
    # settings go on the end rather than shifting the existing ones.
    chain_refine_cap: float = 3.0
    chain_refine_min_ncc: float = 0.30

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if not self.input_path:
            raise ValueError("cfg.input_path is required (see config_example.m).")
        if not isinstance(self.refchannel, int) or self.refchannel < 1:
            raise ValueError("cfg.refchannel is 1-based (MATLAB habit): 1 = red, 2 = green.")
        if not isinstance(self.scale, int) or self.scale < 1:
            raise ValueError("cfg.scale must be a positive integer (original used 4).")
        if not isinstance(self.chunksize, int) or not (1 <= self.chunksize <= 20):
            raise ValueError(
                "cfg.chunksize must be in 1..20 "
                "(RegistrationMasterPipeline.m L23: \"don't go over 20\")."
            )
        if self.proj_type not in ("mean", "max", "median"):
            raise ValueError("cfg.proj_type must be 'mean' | 'max' | 'median'.")
        if isinstance(self.proj_range, str) and self.proj_range not in ("quarter", "full"):
            raise ValueError(
                "cfg.proj_range must be 'quarter', 'full', or an explicit "
                "1-based sequence of plane indices."
            )
        if self.opttype not in ("none", "affine", "rigid"):
            raise ValueError("cfg.opttype must be 'none' | 'affine' | 'rigid'.")
        # Both of the port-extension settings below are validated through the
        # module that OWNS them, so the accepted spellings can never drift
        # from what set_compute_dtype / set_mode accept, and both are stored
        # as canonical NAME STRINGS so the dataclass stays trivially
        # serializable (asdict / repr / pickling into fast_run.py's worker
        # job tuples).
        from .precision import resolve_compute_dtype
        self.compute_dtype = resolve_compute_dtype(self.compute_dtype).name
        from .improved import (resolve_chain_refine_cap,
                               resolve_chain_refine_min_ncc, resolve_mode)
        self.mode = resolve_mode(self.mode)
        # Parameters of correction 3 (not a mode of their own -- see
        # cpstab/improved.py DESIGN NOTE 6), validated by the same module that
        # consumes them and stored as plain floats so the dataclass stays
        # trivially serializable.
        self.chain_refine_cap = resolve_chain_refine_cap(self.chain_refine_cap)
        self.chain_refine_min_ncc = resolve_chain_refine_min_ncc(
            self.chain_refine_min_ncc)

    # ------------------------------------------------------------------ #
    # Derived values (the master script computed these inline)
    # ------------------------------------------------------------------ #
    def nchunks(self, nt):
        """RegistrationMasterPipeline.m L24: Nchunks = round(Nt/chunksize).

        MATLAB round is half-away-from-zero; see matlab_round().
        """
        return matlab_round(nt / self.chunksize)

    def proj_planes_1based(self, nz):
        """Resolve proj_range to explicit **1-based inclusive** plane indices.

        'quarter': MakeSBXall.m L13 default
            round(0.25*Nz):round(0.75*Nz)   (MATLAB colon, both ends included)
        'full': RegistrationMasterPipeline.m L25
            1:Nz
        Returns an int64 numpy vector of 1-based indices, mirroring the
        MATLAB p.proj_range vector. The apply layer converts to 0-based.
        """
        if isinstance(self.proj_range, str):
            if self.proj_range == "quarter":
                lo = matlab_round(0.25 * nz)
                hi = matlab_round(0.75 * nz)
                planes = np.arange(lo, hi + 1, dtype=np.int64)
            else:  # 'full'
                planes = np.arange(1, nz + 1, dtype=np.int64)
        else:
            planes = np.asarray(list(self.proj_range), dtype=np.int64)
        if planes.size == 0 or planes.min() < 1 or planes.max() > nz:
            # For tiny Nz, 'quarter' can produce index 0 (e.g. Nz=1 ->
            # round(0.25)=0); MATLAB would error on the 0 index too.
            raise ValueError(
                "proj_range resolves to planes %s outside 1..%d "
                "(1-based, MATLAB convention)." % (planes.tolist(), nz)
            )
        return planes

    # ------------------------------------------------------------------ #
    # Output paths (replace the pipe.lab.rundir(...) string concatenations)
    # ------------------------------------------------------------------ #
    def out_base(self):
        """Output path stem, analogous to pipe.lab.rundir(...) in the master
        script (L37/L43): '<out_dir>/<input stem>'."""
        d = self.out_dir
        if d is None:
            d = os.path.dirname(os.path.abspath(self.input_path)) or "."
        stem = os.path.basename(self.input_path.rstrip("/\\"))
        stem = os.path.splitext(stem)[0]
        return os.path.join(d, stem)

    def shiftpath(self):
        """RegistrationMasterPipeline.m L37: '<rundir>.dftshifts'.

        DIVERGENCE (container only): the original .dftshifts is a MAT-file
        (DFT_warp_3D_2.m L140 save(...,'-mat')); the port stores the same
        arrays in an .npz. Keys/semantics are the orchestrator module's
        contract.
        """
        return self.out_base() + ".dftshifts.npz"

    def zproj_tiff_path(self):
        """RegistrationMasterPipeline.m L43:
        '<rundir>_<proj_type>_zproj.tif'."""
        return "%s_%s_zproj.tif" % (self.out_base(), self.proj_type)

    def registered_stack_path(self):
        """Where the optional registered stack lands when
        write_registered=True.

        The apply layer (apply_project.make_sbxall) byte-mirrors MATLAB's
        RegWriter: it writes '<input stem>.sbxall' NEXT TO input_path
        (MakeSBXall.m L137/L193), ignoring out_dir — this helper just
        reports that path. The only original-vs-port difference is that the
        write is opt-in (write_registered, default False) instead of
        unconditional.
        """
        base = os.path.splitext(os.path.abspath(self.input_path))[0]
        return base + ".sbxall"


# PORTING NOTES
# 1. proj_range default: the master script computes proj_range = 1:Nz (L25)
#    but NEVER passes it to MakeSBXall, whose own default
#    round(0.25*Nz):round(0.75*Nz) (MakeSBXall.m L13) is what actually ran
#    (CODEMAP §13 Q9). Default 'quarter' therefore replicates the published
#    behavior; 'full' replicates the script's apparent (unused) intent.
#    Until Q9 is answered by ground-truth comparison, treat 'quarter' as
#    canonical.
# 2. refchannel stays 1-based everywhere above the IO layer. Chosen because
#    every mirrored MATLAB signature carries the 1-based value; converting
#    at each boundary would multiply off-by-one risk.
# 3. matlab_round: MATLAB round() is half-away-from-zero; np.round is
#    half-to-even (np.round(22.5)==22 but MATLAB round(22.5)==23). Both
#    Nchunks and the 'quarter' projection bounds hit .5 cases for
#    real-world Nt/Nz values, so this is load-bearing, not pedantry.
# 4. chunksize<=20 is enforced here although MATLAB only had a comment;
#    exceeding it in the original silently degrades registration (larger
#    chunks drift within-chunk). Deliberate hardening, flagged as such.
# 5. shiftpath uses '.dftshifts.npz' (not a MAT-file). Pure container
#    divergence; the arrays inside must still mirror
#    RS/CS/ZS/RS_chunk/CS_chunk/ZS_chunk/scale/tforms_optotune_full of
#    DFT_warp_3D_2.m L140. ref_all and intermediate_shifts are NOT required
#    downstream (MakeSBXall reads only the fields above — CODEMAP §5 F) so
#    the orchestrator may drop them.
# 6. registered_stack_path reports the '<base>.sbxall' the apply layer
#    actually writes (byte-mirror of MATLAB RegWriter, next to input_path;
#    an earlier draft of this file claimed a TIFF — stale, fixed with review
#    F3). Field write_registered defaults False because '<base>.sbxall' was
#    write-only bloat (CODEMAP §5 G / §6); MATLAB wrote it unconditionally
#    (review F2 — deliberate default divergence, documented in
#    apply_project PORTING NOTES #15).
# 7b. mode is a PORT EXTENSION with no MATLAB counterpart, and the ONE field
#    that is allowed to change the numbers the pipeline produces. It is
#    consumed exactly like compute_dtype: run_pipeline installs it process-wide
#    with improved.mode_scope() and every module reads the global, so an
#    improved run cannot leak into a later replicate run in the same
#    interpreter — including on an exception. Default 'replicate' is what the
#    VALIDATION_REPORT covers and what both iron-law regressions assert.
#    Per-correction ablation is improved.feature_scope(), not a config field:
#    the space is 2^4 and enumerating it here would put sixteen spellings into
#    this validator.
# 7a. compute_dtype is a PORT EXTENSION with no MATLAB counterpart (MATLAB is
#    double throughout). It is stored as a NAME string rather than a np.dtype
#    so RegistrationConfig stays trivially serializable (asdict / repr /
#    pickling into fast_run.py's worker job tuples). run_pipeline installs it
#    with precision.compute_dtype_scope(), so the process-wide setting is
#    restored when the run ends — a fast run cannot contaminate a subsequent
#    replicate run in the same interpreter.
# 7. No 'reftype' field: the master hardcodes reftype='mean' at its
#    DFT_warp_3D_2 call (L38, overriding that function's own 'median'
#    default). pipeline.run_pipeline hardcodes it the same way rather than
#    exposing a knob the original run never varied.
