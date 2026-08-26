"""Port of Shipley2020 2-D shift estimation/application helpers.

Source files (ground truth, read line-by-line):
  registration/DFT_rect.m              (L1-L29)  -> dft_rect
  registration/DFT_reg.m               (L1-L17)  -> dft_reg
  registration/DetermineXYShiftsFBS.m  (L1-L49)  -> determine_xy_shifts_fbs
  registration/ApplyXYShiftsFBS.m      (L1-L26)  -> apply_xy_shifts_fbs
  registration/defineReference.m       (L1-L39)  -> define_reference

Array conventions (project-wide, mirrors MATLAB dim order):
  single frame  [Y, X]        (MATLAB [row, col])
  volume        [Y, X, Z]
  time block    [Y, X, Z, T]

Shift conventions:
  dftregistration (cpstab.dftreg) returns [row_shift, col_shift] = the
  translation that must be APPLIED to the moving image to align it to the
  reference (mirrors dftregistrationAlex.m output=[row_shift,col_shift]).
  MATLAB imtranslate takes [tx, ty] == [col_shift, row_shift]; the public
  `imtranslate` here keeps that exact [C, R] argument order.
"""

import math

import numpy as np
from scipy import ndimage as _ndi

from .bilinear import bilinear_shift2 as _fast_shift2
from .dftreg import dftregistration_alex as dftregistration
# Canonical MATLAB double->class cast lives in matlab_compat (shared with
# matlab_imresize / matlab_imwarp_affine2d so all class-preserving ops
# quantize identically); this is the same code that used to be local here.
from .matlab_compat import matlab_cast_like as _cast_back_like_matlab
# Float working class of the image domain: float64 (replicate, default) or
# float32 (fast mode). See cpstab/precision.py and PORTING NOTES 12 below.
from .precision import as_correlation, work_dtype_for, zeros as _fzeros

__all__ = [
    "imtranslate",
    "dft_rect",
    "dft_reg",
    "determine_xy_shifts_fbs",
    "apply_xy_shifts_fbs",
    "define_reference",
]


def _fft2(a):
    """fft2 for the registration engine: ALWAYS float64 -> complex128.

    MATLAB's fft2 implicitly casts its (uint16) argument to double, and the
    literal port of that was `np.fft.fft2(uint16_slice)`, which numpy also
    promotes to float64. Routing through as_correlation() only makes that
    promotion explicit -- the value is unchanged in the default mode.

    It is explicit because in float32 fast mode the ARGUMENT may already be
    float32 (dft_reg / determine_xy_shifts_fbs are handed single-precision
    stacks), and numpy would then hand pocketfft a complex64 transform.
    That is exactly the thing precision.py's rule forbids: every caller of
    this function feeds an argmax that picks a shift, and on this data the
    winning correlation sample beats the runner-up by a median 4.5e-4
    relative -- inside float32's noise floor for a 256x256 transform. See
    cpstab/precision.py for the measurement and what complex64 did to the
    output (Pearson r 0.756).
    """
    return np.fft.fft2(as_correlation(a))


def imtranslate(img, translation):
    """MATLAB imtranslate(A,[tx ty]) equivalent: 2-D, 'linear', FillValues=0.

    Parameters mirror MATLAB: `translation` is **[C, R] = [x, y]** order —
    translation[0] moves content toward +X (increasing column index),
    translation[1] toward +Y (increasing row index). OutputView 'same'.

    Implemented as scipy.ndimage.shift(order=1, mode='grid-constant', cval=0):
    verified bilinear interpolation over a zero-padded grid, identical to
    MATLAB's separable fill-padded linear translation (fill value blends into
    the <=1 px boundary ring for fractional shifts; integer shifts are exact).
    Preserves the input class like MATLAB (integer outputs are rounded
    half-away-from-zero and saturated).
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError("imtranslate: expected a 2-D image [Y, X]")
    tx = float(translation[0])
    ty = float(translation[1])
    w = a.astype(work_dtype_for(a.dtype), copy=False)
    # cpstab.bilinear.bilinear_shift2 is a bit-exact stand-in for exactly
    # this scipy call (its docstring carries the two arithmetic details that
    # make it exact and the validation it was held to); it is ~2.5x faster,
    # and this call is ~10% of a registration chunk.  Anything it does not
    # cover -- non-float working classes -- falls through to scipy.
    if w.dtype == np.float64:
        out = _fast_shift2(w, ty, tx)
    elif w.dtype == np.float32:
        out = _fast_shift2(np.asarray(w, dtype=np.float64),
                           ty, tx).astype(np.float32)
    else:
        out = _ndi.shift(
            w,
            (ty, tx),  # scipy shift is axis order [row, col]
            order=1,
            mode="grid-constant",
            cval=0.0,
            prefilter=False,  # no-op at order=1
        )
    return _cast_back_like_matlab(out, a.dtype)


def _imgaussfilt(img, sigma):
    """MATLAB imgaussfilt(A,sigma) equivalent (defaults).

    MATLAB defaults: FilterSize = 2*ceil(2*sigma)+1 (i.e. radius
    ceil(2*sigma)), Padding='replicate', normalized truncated Gaussian
    sampled at integer offsets -> scipy gaussian_filter(mode='nearest',
    radius=ceil(2*sigma)), whose 1-D kernel formula is identical.
    Preserves the input class like MATLAB.
    """
    a = np.asarray(img)
    sigma = float(sigma)
    if not sigma > 0:
        # MATLAB imgaussfilt validates sigma as positive; scipy would treat
        # sigma=0 as a silent identity filter.
        raise ValueError("_imgaussfilt: sigma must be > 0 (MATLAB imgaussfilt "
                         "requires a positive sigma)")
    radius = int(math.ceil(2.0 * sigma))
    out = _ndi.gaussian_filter(
        a.astype(work_dtype_for(a.dtype), copy=False),
        float(sigma),
        mode="nearest",
        radius=radius,
    )
    return _cast_back_like_matlab(out, a.dtype)


def dft_rect(vol, start, upscale):
    """registration/DFT_rect.m L1-L29: plane-to-plane axial rectification.

    [R,C,reg] = DFT_rect(vol, start, upscale)

    Parameters
    ----------
    vol : ndarray [Y, X, Z]
        Volume to rectify.
    start : int, **1-based** (MATLAB convention, kept so the orchestrator's
        `dft_rect(vol, round(Nz/2), 4)` transliterates verbatim).
        Anchor plane; forward chain runs start..Nz, backward chain start..1.
    upscale : int
        usfac passed to dftregistration.

    Returns
    -------
    R, C : ndarray (Nz,) float64  — per-plane row/col shifts (MATLAB Nz-by-1);
        float64 in both precision modes (shift bookkeeping, note 12).
    reg : ndarray [Y, X, Z] — chained-registered planes, in the compute class
        (float64 by default, float32 in fast mode).

    Chaining semantics (as in MATLAB): each plane i is registered to the
    *translated previous plane* (the running `target`), not to the anchor;
    the anchor plane itself is registered to itself in both loops (shift 0).
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError("dft_rect: expected vol of shape [Y, X, Z]")
    nz = vol.shape[2]  # L2: Nz = size(vol,3)
    if not float(start).is_integer():
        # MATLAB vol(:,:,start) errors on a non-integer index (L4);
        # int() would silently truncate.
        raise ValueError("dft_rect: start must be an integer (1-based plane "
                         "index), got %r" % (start,))
    start = int(start)
    if not 1 <= start <= nz:
        # MATLAB vol(:,:,start) errors out of range; numpy would silently wrap
        raise IndexError("dft_rect: start must be in 1..Nz (1-based)")
    s0 = start - 1

    reg = _fzeros(vol.shape)                     # L3 (MATLAB zeros() = double)
    target = vol[:, :, s0]                       # L4
    r = np.zeros(nz, dtype=np.float64)           # L6
    c = np.zeros(nz, dtype=np.float64)           # L7

    # forward (L10-L17): i = start:Nz
    for i in range(s0, nz):
        source = vol[:, :, i]
        s = dftregistration(_fft2(target), _fft2(source), upscale)  # L12
        r[i] = s[0]
        c[i] = s[1]
        target = imtranslate(source, (s[1], s[0]))  # L15: [C, R] order
        reg[:, :, i] = target

    # backwards (L20-L28): i = flip(1:start) = start, start-1, ..., 1
    target = vol[:, :, s0]
    for i in range(s0, -1, -1):
        source = vol[:, :, i]
        s = dftregistration(_fft2(target), _fft2(source), upscale)
        r[i] = s[0]
        c[i] = s[1]
        target = imtranslate(source, (s[1], s[0]))
        reg[:, :, i] = target

    return r, c, reg


def dft_reg(stack, target, upscale):
    """registration/DFT_reg.m L1-L17: register each plane to a fixed target.

    [R,C,reg] = DFT_reg(stack, target, upscale)

    Parameters
    ----------
    stack : ndarray [Y, X, N]
    target : ndarray [Y, X]   — fixed reference frame (NOT updated per plane).
    upscale : int             — usfac for dftregistration.

    Returns
    -------
    R, C : ndarray (N,) float64 (both modes — shift bookkeeping, note 12)
    reg : ndarray [Y, X, N] — translated planes, in the compute class.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("dft_reg: expected stack of shape [Y, X, N]")
    n = stack.shape[2]                            # L2
    reg = _fzeros(stack.shape)                    # L3
    r = np.zeros(n, dtype=np.float64)             # L5
    c = np.zeros(n, dtype=np.float64)             # L6

    # MATLAB recomputes fft2(target) every iteration (L11); hoisting is
    # numerically identical (pure function of an unchanged array).
    target_ft = _fft2(np.asarray(target))
    for i in range(n):  # L9-L15
        source = stack[:, :, i]
        s = dftregistration(target_ft, _fft2(source), upscale)
        r[i] = s[0]
        c[i] = s[1]
        reg[:, :, i] = imtranslate(source, (s[1], s[0]))  # L14: [C, R]

    return r, c, reg


def determine_xy_shifts_fbs(full_vol, blur_factor, keeping_factor,
                            reference_volume):
    """registration/DetermineXYShiftsFBS.m L1-L49: per-plane XY shifts.

    [RowShifts,ColumnShifts] = DetermineXYShiftsFBS(full_vol, BlurFactor,
                                                    KeepingFactor, ReferenceVolume)

    Parameters
    ----------
    full_vol : ndarray [Y, X, Z, T]
    blur_factor : float      — sigma of the Gaussian blur (MATLAB imgaussfilt).
    keeping_factor : float   — 0 < k <= ~1, central crop fraction (e.g. 0.95).
    reference_volume : ndarray [Y, X, Z, T/n] — moving reference
        (defineReference output); cropped with the SAME bounds computed from
        full_vol's size, exactly as in MATLAB.

    Returns
    -------
    row_shifts, column_shifts : ndarray (Z, T) float64 (both modes)
        Shifts to APPLY to each plane to align it to its reference
        (feed directly to apply_xy_shifts_fbs).

    Registration runs at usfac=100 on the blurred central crops.
    """
    full_vol = np.asarray(full_vol)
    reference_volume = np.asarray(reference_volume)
    if full_vol.ndim != 4 or reference_volume.ndim != 4:
        raise ValueError(
            "determine_xy_shifts_fbs: expected 4-D arrays [Y, X, Z, T]")

    size1 = full_vol.shape[0]   # L20: Size = size(full_vol)
    size2 = full_vol.shape[1]
    s3 = full_vol.shape[2]      # L21
    s4 = full_vol.shape[3]
    keep = float(keeping_factor)  # L22

    # L23-L26 crop bounds, 1-based inclusive; float-op order copied verbatim
    # so ceil() lands on the same integers as MATLAB.
    r_lo = math.ceil(size1 * (1 - keep) / 2)
    r_hi = math.ceil(size1 * (1 - (1 - keep) / 2))
    c_lo = math.ceil(size2 * (1 - keep) / 2)
    c_hi = math.ceil(size2 * (1 - (1 - keep) / 2))
    if r_lo < 1 or c_lo < 1:
        # MATLAB: index 0 -> hard error. Guard so numpy never wraps to -1.
        raise IndexError(
            "determine_xy_shifts_fbs: crop lower bound < 1 "
            "(KeepingFactor too close to / above 1)")
    if (reference_volume.shape[0] < r_hi or reference_volume.shape[1] < c_hi):
        # MATLAB would error; numpy slicing would silently truncate.
        raise IndexError(
            "determine_xy_shifts_fbs: ReferenceVolume smaller than the crop "
            "bounds computed from full_vol")

    red_vol = full_vol[r_lo - 1:r_hi, c_lo - 1:c_hi, :, :]          # L23-L26
    chunck = math.floor(s4 / reference_volume.shape[3])             # L27
    ref_crop = reference_volume[r_lo - 1:r_hi, c_lo - 1:c_hi, :, :]  # L28-L32

    row_shifts = np.zeros((s3, s4), dtype=np.float64)     # L33
    column_shifts = np.zeros((s3, s4), dtype=np.float64)  # L34

    for t in range(s4):  # L36: t = 1:S4
        # L37: reft = ReferenceVolume(:,:,:,ceil(t/chunck))
        reft = ref_crop[:, :, :, math.ceil((t + 1) / chunck) - 1]
        for i in range(s3):  # L38
            ref = reft[:, :, i]
            output = dftregistration(                     # L40-L42
                _fft2(_imgaussfilt(ref, blur_factor)),
                _fft2(_imgaussfilt(red_vol[:, :, i, t], blur_factor)),
                100)
            row_shifts[i, t] = output[0]
            column_shifts[i, t] = output[1]

    return row_shifts, column_shifts


def apply_xy_shifts_fbs(unshifted_volume, row_shifts, column_shifts):
    """registration/ApplyXYShiftsFBS.m L1-L26: apply per-plane XY shifts.

    correctedVolume = ApplyXYShiftsFBS(unshiftedVolume, RowShifts, ColumnShifts)

    Parameters
    ----------
    unshifted_volume : ndarray [Y, X, Z, T]
    row_shifts, column_shifts : ndarray (Z, T)

    Returns
    -------
    corrected_volume : ndarray [Y, X, Z, T] in the compute class
        Each plane translated by imtranslate(slice, [C, R]) — MATLAB's
        "careful order!": x-translation = ColumnShifts, y = RowShifts.
    """
    unshifted_volume = np.asarray(unshifted_volume)
    if unshifted_volume.ndim != 4:
        raise ValueError(
            "apply_xy_shifts_fbs: expected 4-D array [Y, X, Z, T]")
    row_shifts = np.asarray(row_shifts)
    column_shifts = np.asarray(column_shifts)

    nbplanes = unshifted_volume.shape[2]                       # L14
    corrected = _fzeros(unshifted_volume.shape)                # L16
    for t in range(unshifted_volume.shape[3]):                 # L17
        for i in range(nbplanes):                              # L18
            r_ = row_shifts[i, t]
            c_ = column_shifts[i, t]
            slice_ = unshifted_volume[:, :, i, t]
            corrected[:, :, i, t] = imtranslate(slice_, (c_, r_))  # L22
    return corrected


def define_reference(volume, n, type):  # noqa: A002 - mirrors MATLAB arg name
    """registration/defineReference.m L1-L39: build the moving reference.

    ref = defineReference(volume, n, type)

    Parameters
    ----------
    volume : ndarray [Y, X, Z, T] (double per the MATLAB header contract)
    n : int — frames averaged per reference volume; must divide T.
    type : {'median', 'mean'} — temporal projection.

    Returns
    -------
    ref : ndarray [Y, X, Z, T/n] in the compute class
    """
    volume = np.asarray(volume)
    if volume.ndim != 4:
        raise ValueError("define_reference: expected 4-D array [Y, X, Z, T]")
    x = volume.shape[0]   # L13-L16
    y = volume.shape[1]
    z = volume.shape[2]
    t = volume.shape[3]

    if not float(n).is_integer():
        # MATLAB: mod(t, n) == 0 can pass for non-integer n (e.g. t=10,
        # n=2.5), but the loop then dies indexing (i-1)*n+1:i*n with a
        # non-integer subscript (L26). int() would silently truncate n.
        raise ValueError("define_reference: n must be an integer, got %r"
                         % (n,))
    n = int(n)
    if n <= 0 or t % n != 0:  # L18-L20
        raise ValueError("n does not divide the number of frames")
    nref = t // n

    ref = _fzeros((x, y, z, nref))                    # L22

    for i in range(nref):        # L24: i = 1:t/n
        for zi in range(z):      # L25
            # L26-L27: volume(:,:,z,(i-1)*n+1:i*n) -> squeeze -> [Y, X, n]
            a = volume[:, :, zi, i * n:(i + 1) * n]
            if type == "median":       # L28-L29 (strcmp: exact match)
                a = np.median(a, axis=2)
            elif type == "mean":       # L30-L31
                a = np.mean(a, axis=2)
            else:                      # L32-L33
                raise ValueError("specify type of projection for reference")
            ref[:, :, zi, i] = a       # L35
    return ref


# PORTING NOTES
# -------------
# 1. dftregistration contract: assumed to mirror dftregistrationAlex.m, i.e.
#    it returns a 2-element sequence [row_shift, col_shift] (Alex's variant
#    strips Guizar's error/diffphase fields; dftregistrationAlex.m L128).
#    This module only reads out[0]/out[1], so it also tolerates the 4-element
#    original-Guizar layout ONLY IF that layout is [row, col, ...] — it is
#    NOT ([error, diffphase, row, col]), so dftreg.py MUST keep Alex's order.
# 2. imtranslate boundary model: MATLAB imtranslate('linear', FillValues=0,
#    OutputView 'same') does separable translation over a fill-padded grid,
#    so for fractional shifts the fill value blends into the <=1 px border
#    ring. scipy mode='constant' does NOT blend (out-of-domain sample points
#    snap wholly to cval); mode='grid-constant' does, and was verified here
#    (against a manual bilinear-over-zero-padding reference) to be exact
#    bilinear + zero padding, with integer shifts bit-exact. Chose
#    'grid-constant'. Not validated against a live MATLAB (none on this
#    machine); worst-case divergence, if MATLAB in fact snaps rather than
#    blends, is confined to the 1-px border ring for fractional shifts.
# 3. dtype semantics (RESOLVED by review F2): in the original MATLAB run
#    DFT_rect received uint16 (sbxRead -> imresize -> imwarp all preserve
#    uint16), and MATLAB imtranslate then QUANTIZED each chained target back
#    to uint16 (round-half-away + saturate) before the next fft2.
#    imtranslate/_imgaussfilt here replicate that class-preserving behavior,
#    and the orchestrator now feeds dft_rect the NATIVE-class (uint16)
#    volume, so the quantization chain is reproduced. The cast helper is the
#    shared matlab_compat.matlab_cast_like (imported above), the same
#    routine matlab_imresize/matlab_imwarp_affine2d use.
#    Float32 inputs used to diverge differently (MATLAB computes in single
#    end-to-end, we computed in double and cast down). Under note 12 that is
#    no longer true in fast mode: work_dtype_for() puts the internal
#    arithmetic in single for a float32 input, which is what MATLAB would
#    have done. In replicate mode the old divergence stands and is still
#    unreachable (the pipeline feeds uint16 here, never float32).
#    Additionally, in uint16 mode a pixel whose bilinear true value is exactly
#    x.5 can round to either side (±1 count): scipy's separable two-pass
#    evaluation and a direct 4-term sum can differ by 1 ulp around the tie,
#    and which side MATLAB lands on depends on its internal operation order
#    (undecidable without a live MATLAB). R/C shift estimates are unaffected;
#    the float64 project boundary never quantizes, so it has no such ties.
# 4. _imgaussfilt: MATLAB imgaussfilt defaults = radius ceil(2*sigma)
#    (FilterSize 2*ceil(2*sigma)+1), Padding 'replicate', normalized truncated
#    Gaussian sampled at integer offsets. scipy gaussian_filter(mode='nearest',
#    radius=ceil(2*sigma)) uses the identical 1-D kernel formula. MATLAB
#    FilterDomain='auto' may switch to frequency-domain filtering for large
#    kernels — numerically equivalent to spatial convolution to ~1e-15.
# 5. dft_rect keeps MATLAB's 1-BASED `start` so the orchestrator's
#    DFT_rect(vol, round(Nz/2), 4) transliterates with MATLAB round()
#    (half away from zero). Out-of-range start raises instead of numpy's
#    silent negative-index wraparound; non-integer start raises instead of
#    int() truncation (MATLAB errors on non-integer indices). Likewise
#    define_reference rejects non-integer n and _imgaussfilt rejects
#    sigma <= 0 (MATLAB errors in both cases; the naive numpy/scipy paths
#    would silently truncate / no-op).
# 6. DetermineXYShiftsFBS crop: the 1-based inclusive bounds
#    ceil(M*(1-k)/2) .. ceil(M*(1-(1-k)/2)) become the half-open slice
#    [lo-1 : hi]. Float expression order copied verbatim so ceil() lands on
#    the same integers. Guards raise where MATLAB would hard-error (lower
#    bound 0, or a ReferenceVolume smaller than bounds computed from
#    full_vol's size) because numpy would otherwise wrap/truncate silently.
#    ReferenceVolume is cropped with full_vol's size, exactly as in MATLAB.
# 7. chunck = floor(S4 / size(ref,4)): if ref has more time points than
#    full_vol, chunck = 0 and MATLAB dies indexing at Inf; here ceil((t+1)/0)
#    raises ZeroDivisionError. If chunck*refT < S4, ceil overruns the ref
#    time axis: IndexError here, index-out-of-bounds error in MATLAB. Both
#    faithful "crash the same way" cases.
# 8. define_reference: np.median matches MATLAB median for double input
#    (even n -> mean of the two middle values, NaN propagates). MATLAB
#    median on INTEGER classes returns the same class (rounds the .5 case);
#    not replicated — the .m header and every call site use double. The
#    permute/squeeze dance in MATLAB (L27) is a no-op reshape to [Y, X, n];
#    indexed directly here. MATLAB's shadowing of loop var `z` (L25) over
#    the size var (L15) is harmless and not mirrored (separate names).
# 9. Strictness: MATLAB tolerates trailing singleton dims (e.g. a 3-D
#    full_vol acts as T=1). These ports require full-rank arrays ([Y,X,Z] /
#    [Y,X,Z,T]) and raise otherwise — every production call site passes
#    full-rank arrays, and silent axis mix-ups are worse than a hard error.
# 10. dft_reg hoists fft2(target) out of the loop (MATLAB recomputes it per
#    iteration); pure function of an unchanged array, numerically identical.
# 11. All fft inputs are cast to float64 -> complex128, matching MATLAB's
#    implicit double cast in fft2(uint16). The cast is explicit (_fft2 wraps
#    as_correlation) rather than left to numpy's own promotion, because under
#    note 12 the argument can arrive as float32 and numpy would then silently
#    give the engine a complex64 transform.
# 12. FLOAT32 FAST MODE (port extension, cpstab/precision.py). With
#    cfg.compute_dtype='float32' every `zeros(...)` accumulator here and the
#    internal class of imtranslate/_imgaussfilt become single precision, while
#    _fft2 and the dftregistration engine behind it stay double -- fast mode
#    lowers PIXELS, never the correlation that decides a shift. The default
#    float64 mode is bit-for-bit what it was (`_fzeros(shape)` ==
#    `np.zeros(shape, np.float64)`; as_correlation is a no-op there).
#    Two module-specific consequences:
#      * dft_rect's CHAIN (each plane registered to the translated previous
#        plane, then quantized back to uint16 by imtranslate's class
#        preservation) can differ by +-1 count per step from the single-
#        precision bilinear. Each step re-quantizes to integers, so the
#        difference cannot accumulate past the rounding tie -- it does not
#        drift -- but on frames this dim (mean ~22 counts) a 1-count change
#        IS a percent-level perturbation, which is the residual mechanism by
#        which fast-mode shift estimates can still differ from replicate ones.
#        Measured on the validation subset: they did not (RS/CS bit-identical).
#      * row_shifts / column_shifts in determine_xy_shifts_fbs stay float64 in
#        BOTH modes: they are shift bookkeeping (O(Nz*T) scalars), and they
#        end up in the .dftshifts.npz that the apply stage replays.
