"""Subpixel DFT registration engine (Guizar-Sicairos 2008), ported line-for-line.

MATLAB sources (ground truth):
  registration/dftregistrationAlex.m  (202 lines) -> dftregistration_alex, dftups, ftpad
  registration/dftregistration3D.m    ( 69 lines) -> dftregistration3d, ftpad3

Array conventions (project rules):
  Single frame  [Y, X]        == MATLAB [row, col]
  Volume        [Y, X, Z]     == MATLAB [row, col, plane]
  All Fourier inputs are the *unshifted* FFTs (DC at index [0, 0(, 0)]),
  i.e. exactly what np.fft.fft2 / np.fft.fftn produce -- do NOT fftshift.
  Complex math is done in complex128 throughout.

Shift sign convention (verified empirically in tests/test_dftreg.py):
  If moving = circshift(reference, (+dy, +dx)) -- i.e. moving(n) = reference(n - d)
  -- then dftregistration_alex(fft2(reference), fft2(moving), usfac)
  returns approximately (-dy, -dx): the translation to APPLY to the moving
  image to align it onto the reference. Same convention for the 3D variant.

See "# PORTING NOTES" at the end of this file for every deliberate
divergence / disambiguation from the MATLAB source.
"""

import numpy as np

from .precision import get_complex_dtype

__all__ = ["dftregistration_alex", "dftregistration3d", "dftups", "ftpad", "ftpad3"]


# ---------------------------------------------------------------------------
# Small helpers (MATLAB semantics)
# ---------------------------------------------------------------------------

def _matlab_round(x):
    """MATLAB round(): round half AWAY from zero (scalar).

    np.round is half-to-even and np.floor(x+0.5) is wrong for negative
    half-integers, so neither is usable directly.
    """
    return float(np.trunc(x + np.copysign(0.5, x)))


def _argmax_first_colmajor(a):
    """0-based subscripts of the FIRST maximum in column-major (Fortran) scan.

    Mirrors MATLAB `[~,I] = max(A(:))` / `find(A == max(A(:)), 1, 'first')`,
    both of which scan linear (column-major) indices and return the first hit.
    np.argmax alone scans C-order, which breaks ties differently.

    NaN guard: np.argmax treats NaN as maximal, so any NaN in `a` wins and
    the port would silently return shifts computed from a NaN peak. MATLAB
    max() skips NaNs and `x == NaN` is all-false, so the same input makes
    find() return EMPTY and MATLAB errors downstream. Raise instead
    (fail-loudly mapping, PORTING NOTES 12); O(1) check on the winner only,
    which is sufficient because NaN always wins np.argmax.
    """
    idx = int(np.argmax(a.ravel(order="F")))
    sub = np.unravel_index(idx, a.shape, order="F")
    if np.isnan(a[sub]):
        raise ValueError(
            "cross-correlation peak is NaN (non-finite registration input); "
            "MATLAB find()==max() returns empty here and errors downstream")
    return tuple(int(v) for v in sub)


def _ifftshift_range(n):
    """MATLAB `ifftshift(-fix(n/2):ceil(n/2)-1)` for positive integer n.

    Maps 0-based array index i -> signed frequency/shift value
    [0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1].
    """
    return np.fft.ifftshift(np.arange(-(n // 2), (n + 1) // 2))


def _ftpad_paste(im_ft_shifted, Nout):
    """Shared centered paste-with-clipping used by both FTpad variants.

    Implements (0-based, half-open) exactly the MATLAB 1-based inclusive
    ranges:
      dest   max(cc+1,1)  : min(cc+Nin,  Nout)
      source max(-cc+1,1) : min(-cc+Nout, Nin)
    with cc = cenout_cen = (floor(Nout/2)+1) - (floor(Nin/2)+1)
            = floor(Nout/2) - floor(Nin/2)   (the +1s cancel).
    Handles both padding (Nout > Nin) and cropping (Nout < Nin).
    """
    Nin = im_ft_shifted.shape
    out = np.zeros(Nout, dtype=im_ft_shifted.dtype)
    dst, src = [], []
    for ni, no in zip(Nin, Nout):
        cc = no // 2 - ni // 2
        dst.append(slice(max(cc, 0), min(cc + ni, no)))
        src.append(slice(max(-cc, 0), min(-cc + no, ni)))
    out[tuple(dst)] = im_ft_shifted[tuple(src)]
    return out


# ---------------------------------------------------------------------------
# dftregistrationAlex.m
# ---------------------------------------------------------------------------

def dftregistration_alex(buf1ft, buf2ft, usfac=1):
    """registration/dftregistrationAlex.m L1-L129 (function dftregistrationAlex).

    Efficient subpixel image registration by cross-correlation
    (Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 33, 156-158, 2008),
    as modified by Alex: returns ONLY [row_shift, col_shift].
    (The MATLAB header comment still advertises
    [error, diffphase, net_row_shift, net_col_shift] + optional Greg, but the
    code at L128 returns just the two shifts -- the port mirrors the code.)

    Parameters
    ----------
    buf1ft : (nr, nc) complex ndarray
        FFT of the REFERENCE image, DC at [0, 0] (do not fftshift).
    buf2ft : (nr, nc) complex ndarray
        FFT of the image to register (moving), DC at [0, 0].
    usfac : int, optional (default 1, as in MATLAB L68-L70)
        Upsampling factor. Registration is to within 1/usfac of a pixel.
        0  -> no registration, returns [0, 0]
        1  -> integer-pixel registration (plain IFFT cross-correlation peak)
        2  -> half-pixel via 2x zero-padded FFT
        >2 -> 2x estimate refined by matrix-multiply DFT (dftups)

    Returns
    -------
    output : (2,) float64 ndarray
        [row_shift, col_shift] == [dy, dx]: translation to apply to the
        moving image to align it to the reference (see module docstring).
    """
    # complex128 in BOTH precision modes -- see PORTING NOTES 16; the callers
    # promote their (possibly float32) pixel arrays before transforming, so
    # this asarray is normally a no-op.
    cdt = get_complex_dtype()
    buf1ft = np.asarray(buf1ft, dtype=cdt)
    buf2ft = np.asarray(buf2ft, dtype=cdt)
    nr, nc = buf2ft.shape  # L72

    # L73-L74 (used only by the usfac == 1 branch)
    Nr = _ifftshift_range(nr)
    Nc = _ifftshift_range(nc)

    if usfac == 0:
        # L76-L79: no registration
        row_shift = 0.0
        col_shift = 0.0
    elif usfac == 1:
        # L80-L87: single-pixel registration
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        CCabs = np.abs(CC)
        # MATLAB L84 uses find() WITHOUT 'first' (returns all ties);
        # port takes the first max in column-major order (see PORTING NOTES 2).
        r, c = _argmax_first_colmajor(CCabs)
        row_shift = float(Nr[r])
        col_shift = float(Nc[c])
    elif usfac > 1:
        # L88-L97: start with usfac == 2 (2x zero-padded FFT)
        CC = np.fft.ifft2(ftpad(buf1ft * np.conj(buf2ft), (2 * nr, 2 * nc)))
        CCabs = np.abs(CC)
        r, c = _argmax_first_colmajor(CCabs)  # L92: find(...,1,'first')
        Nr2 = np.fft.ifftshift(np.arange(-nr, nr))  # L94: -fix(nr):ceil(nr)-1
        Nc2 = np.fft.ifftshift(np.arange(-nc, nc))  # L95
        row_shift = float(Nr2[r]) / 2.0
        col_shift = float(Nc2[c]) / 2.0

        if usfac > 2:
            # L98-L114: refine with matrix-multiply DFT
            # L102-L103: initial estimate on the upsampled grid
            # (MATLAB round = half away from zero; hits exact .5 for odd usfac)
            row_shift = _matlab_round(row_shift * usfac) / usfac
            col_shift = _matlab_round(col_shift * usfac) / usfac
            # L104: dftshift = fix(ceil(usfac*1.5)/2); output center at
            # MATLAB index dftshift+1 == 0-based index dftshift
            nlarge = int(np.ceil(usfac * 1.5))
            dftshift = nlarge // 2
            # L106-L107: note the operand swap (buf2*conj(buf1)) + outer conj
            CC = np.conj(dftups(buf2ft * np.conj(buf1ft), nlarge, nlarge, usfac,
                                dftshift - row_shift * usfac,
                                dftshift - col_shift * usfac))
            CCabs = np.abs(CC)
            rloc, cloc = _argmax_first_colmajor(CCabs)  # L110, 0-based
            # L111-L112: MATLAB rloc(1-based) - dftshift - 1 == rloc0 - dftshift
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            # L113-L114
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac

        # L117-L124: singleton dimension -> shift has no effect, force 0.
        # (Only in this usfac > 1 branch, exactly as in MATLAB.)
        if nr == 1:
            row_shift = 0.0
        if nc == 1:
            col_shift = 0.0
    else:
        # 0 < usfac < 1 (or negative): MATLAB falls through every branch and
        # dies at L128 with "Undefined ... 'row_shift'". Fail loudly instead.
        raise ValueError("usfac must be 0, 1, or > 1 (got %r)" % (usfac,))

    # L128
    return np.array([row_shift, col_shift], dtype=np.float64)


def dftups(in_, nor=None, noc=None, usfac=1, roff=0, coff=0):
    """registration/dftregistrationAlex.m L131-L167 (subfunction dftups).

    Upsampled DFT over a small window by matrix multiplies. Equivalent to:
    embed `in_` (DC at [0,0]) in a usfac-times-larger zero array centered on
    the spectrum, FFT the large array, and read out the [nor, noc] window
    whose element (p, q) is the large-FFT bin at linear frequency
    (p - roff, q - coff) (mod usfac*n).

    Parameters mirror MATLAB dftups(in, nor, noc, usfac, roff, coff):
    `in` is renamed `in_` (Python keyword); nor/noc default to in_.shape,
    usfac=1, roff=0, coff=0 (MATLAB L156-L162).

    Returns (nor, noc) complex128 ndarray.
    """
    cdt = get_complex_dtype()
    in_ = np.asarray(in_, dtype=cdt)
    nr, nc = in_.shape  # L156
    if roff is None:
        roff = 0
    if coff is None:
        coff = 0
    if noc is None:
        noc = nc
    if nor is None:
        nor = nr
    nor = int(nor)
    noc = int(noc)
    # L164: kernc = exp((-1i*2*pi/(nc*usfac)) *
    #                  (ifftshift(0:nc-1).' - floor(nc/2)) * ((0:noc-1) - coff))
    # The two kernels are pure functions of the geometry: evaluated in double
    # and cast to the engine's complex class (currently always complex128, so
    # the cast is a no-op -- it exists so the class has ONE source of truth).
    fc = np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2.0)
    kernc = np.exp((-1j * 2.0 * np.pi / (nc * usfac))
                   * np.outer(fc, np.arange(noc) - coff)).astype(cdt, copy=False)
    # L165: kernr = exp((-1i*2*pi/(nr*usfac)) *
    #                  ((0:nor-1).' - roff) * (ifftshift(0:nr-1) - floor(nr/2)))
    fr = np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2.0)
    kernr = np.exp((-1j * 2.0 * np.pi / (nr * usfac))
                   * np.outer(np.arange(nor) - roff, fr)).astype(cdt, copy=False)
    # L166
    # np.errstate: Apple Accelerate BLAS raises spurious divide/overflow/
    # invalid FP flags on complex matmul for some shapes (same phenomenon as
    # matlab_compat._resize_along_dim, where it was verified bit-identical to
    # np.einsum; a matmul performs no division, so the flags are bogus).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return kernr @ in_ @ kernc


def ftpad(im_ft, outsize):
    """registration/dftregistrationAlex.m L170-L203 (subfunction FTpad), 2D.

    Pad or crop a Fourier transform (DC at [0,0]) to `outsize`, keeping zero
    frequency in the right place for subsequent FFT/IFFT, and scale amplitude
    by prod(outsize)/prod(insize). No Nyquist-bin splitting is performed --
    the fftshifted spectrum is embedded/cropped as-is, exactly like MATLAB.

    Parameters
    ----------
    im_ft : (nr, nc) complex ndarray, DC at [0, 0]
    outsize : length-2 sequence of int, output shape (ny, nx)

    Returns (outsize) complex128 ndarray, DC at [0, 0].
    """
    im_ft = np.asarray(im_ft, dtype=get_complex_dtype())
    if im_ft.ndim != 2:
        # L185-L187: MATLAB errors via ~ismatrix
        raise ValueError("Maximum number of array dimensions is 2")
    Nout = tuple(int(v) for v in outsize)
    Nin = im_ft.shape
    out = _ftpad_paste(np.fft.fftshift(im_ft), Nout)  # L190-L200
    # L202: imFTout = ifftshift(imFTout)*Nout(1)*Nout(2)/(Nin(1)*Nin(2))
    # (same left-to-right evaluation order as MATLAB)
    return np.fft.ifftshift(out) * Nout[0] * Nout[1] / (Nin[0] * Nin[1])


# ---------------------------------------------------------------------------
# dftregistration3D.m
# ---------------------------------------------------------------------------

def dftregistration3d(buf1ft, buf2ft, usfac):
    """registration/dftregistration3D.m L1-L35 (function dftregistration3D).

    3D coarse DFT registration: zero-pad the 3D cross-power spectrum by
    `usfac` in every dimension, inverse FFT, take the integer argmax.
    No matrix-multiply refinement -- resolution is exactly 1/usfac voxel.
    The pipeline always calls it with usfac=2 (half-pixel XY, half-plane Z).

    Parameters
    ----------
    buf1ft : (nr, nc, np_) complex ndarray
        FFTN of the REFERENCE volume [Y, X, Z], DC at [0, 0, 0].
        np_ must be >= 2 (see Raises below).
    buf2ft : (nr, nc, np_) complex ndarray
        FFTN of the volume to register, DC at [0, 0, 0].
    usfac : number
        Upsampling factor (REQUIRED -- the MATLAB function has no default).
        Need NOT be an integer: MATLAB only requires zeros([usfac*nr,
        usfac*nc, usfac*np]) to succeed, i.e. every product must be a
        positive integer (usfac=1.5 on all-even dims runs fine in MATLAB
        and is supported here bit-identically).

    Returns
    -------
    output : (3,) float64 ndarray
        [row_shift, col_shift, pl_shift] == [dy, dx, dz]: translation to
        apply to the moving volume to align it to the reference.
        NOTE: singleton nr or nc forces that component to 0 (L27-L32).

    Raises
    ------
    ValueError
        If np_ == 1 or usfac*np_ == 1: MATLAB CRASHES inside FTpad on these
        (trailing-singleton size() semantics -- see PORTING NOTES 8), so the
        port refuses them too instead of returning a made-up pl_shift.
        Also if usfac <= 0 or any usfac*dim is non-integer (MATLAB zeros()
        errors / returns empty -- PORTING NOTES 6).
    """
    cdt = get_complex_dtype()
    buf1ft = np.asarray(buf1ft, dtype=cdt)
    buf2ft = np.asarray(buf2ft, dtype=cdt)
    nr, nc, np_ = buf2ft.shape  # L5
    if np_ == 1:
        # MATLAB errors here: size() of an (nr, nc, 1) array is [nr nc], so
        # inside FTpad `center` is 1x2 while `centerout` is 1x3 and L62
        # `centerout - center` is a dimension mismatch (usfac==1 dies at
        # cenout_cen(3) instead). Fail loudly like MATLAB; PORTING NOTES 8.
        raise ValueError(
            "dftregistration3d: single-plane volume (np == 1); MATLAB "
            "dftregistration3D errors inside FTpad on such input -- "
            "register single planes with dftregistration_alex instead")
    usfac = float(usfac)
    out3 = (nr * usfac, nc * usfac, np_ * usfac)
    if usfac <= 0 or any(v != int(v) for v in out3):
        # MATLAB zeros() errors on non-integer sizes and returns empties for
        # usfac <= 0 (empty output would crash the caller); PORTING NOTES 6.
        raise ValueError(
            "usfac must be > 0 with usfac*nr, usfac*nc, usfac*np all "
            "integers (got usfac=%r for shape %r)" % (usfac, (nr, nc, np_)))
    outsize = tuple(int(v) for v in out3)
    if outsize[2] == 1:
        # zeros([m, n, 1]) is 2-D in MATLAB -> same FTpad crash as np_ == 1.
        raise ValueError(
            "dftregistration3d: usfac*np == 1 makes MATLAB's zeros() drop "
            "the plane dimension and FTpad error; refused (usfac=%r, np=%d)"
            % (usfac, np_))
    # (MATLAB L6-L8 compute Nr/Nc/Np that are never used -- not ported.)

    # L10-L13
    CC = np.fft.ifftn(ftpad3(buf1ft * np.conj(buf2ft), outsize))
    CCabs = np.abs(CC)
    # [~,I] = max(CCabs(:)); ind2sub(...)  -> first max, column-major
    r, c, p = _argmax_first_colmajor(CCabs)

    # L16-L18: -fix(nr*usfac/2):ceil(nr*usfac/2)-1 with nr*usfac == outsize[0]
    Nr2 = _ifftshift_range(outsize[0])
    Nc2 = _ifftshift_range(outsize[1])
    Np2 = _ifftshift_range(outsize[2])

    # L20-L22
    row_shift = float(Nr2[r]) / usfac
    col_shift = float(Nc2[c]) / usfac
    pl_shift = float(Np2[p]) / usfac

    # L27-L32 zero singleton row/col shifts. (np_ == 1 never reaches here:
    # rejected above because MATLAB itself crashes on it -- PORTING NOTES 8.)
    if nr == 1:
        row_shift = 0.0
    if nc == 1:
        col_shift = 0.0

    # L34
    return np.array([row_shift, col_shift, pl_shift], dtype=np.float64)


def ftpad3(im_ft, outsize):
    """registration/dftregistration3D.m L37-L70 (subfunction FTpad), 3D.

    Same algorithm as :func:`ftpad` generalized to 3 dimensions; amplitude
    scale prod(outsize)/prod(insize). The MATLAB subfunction has no
    dimensionality check (it just index-errors on non-3D input); the port
    raises ValueError for anything but a 3D array.
    """
    im_ft = np.asarray(im_ft, dtype=get_complex_dtype())
    if im_ft.ndim != 3:
        raise ValueError("ftpad3 requires a 3D array")
    Nout = tuple(int(v) for v in outsize)
    Nin = im_ft.shape
    out = _ftpad_paste(np.fft.fftshift(im_ft), Nout)  # L52-L67
    # L69 (same left-to-right evaluation order as MATLAB)
    return (np.fft.ifftshift(out)
            * Nout[0] * Nout[1] * Nout[2] / (Nin[0] * Nin[1] * Nin[2]))


# ---------------------------------------------------------------------------
# PORTING NOTES
# ---------------------------------------------------------------------------
# 1. Stale MATLAB header: dftregistrationAlex.m's comment block (L31-L37)
#    advertises output = [error, diffphase, net_row_shift, net_col_shift] and
#    an optional Greg, inherited from Guizar's original dftregistration.m.
#    Alex's actual code computes none of those: L128 is
#    `output=[row_shift,col_shift]`. The port mirrors the CODE (2 values).
#    Callers (DFT_rect L12/L23, DFT_reg L11, DetermineXYShiftsFBS L40) index
#    only S(1)/S(2), confirming this.
# 2. usfac == 1 branch, MATLAB L84: `find(CCabs == max(CCabs(:)))` has no
#    `1,'first'`, so with exactly-tied maxima MATLAB returns VECTORS and the
#    function output becomes a >2-element vector (a latent bug, never
#    triggered on real data). Port always takes the first maximum in
#    column-major order -- identical whenever the maximum is unique.
#    Caller-visible consequence when maxima DO tie (e.g. an all-black /
#    constant frame, as from a dropped frame) at usfac == 1: MATLAB callers
#    read S(1)/S(2), and with the long output S(2) is the SECOND tied
#    ROW-shift, not a col-shift at all (4x4 constant image: MATLAB output
#    starts [0 1 -2 -1 ...], so S(1:2) = [0 1]; the port returns [0 0]) --
#    the two versions then differ by 1 px in the column component. Not
#    reachable in the pipeline (DetermineXYShiftsFBS hardwires usfac=100;
#    DFT_rect/DFT_reg take a caller-chosen upscale > 1), but any caller
#    running upscale == 1 on degenerate frames should know.
# 3. Argmax tie-breaking everywhere: MATLAB max/find scan linear indices in
#    column-major order; np.argmax scans C-order. Port ravels in order='F'
#    before argmax so ties resolve exactly like MATLAB.
# 4. MATLAB round() (L102-L103) is half-away-from-zero; np.round is
#    half-to-even and np.floor(x+0.5) is wrong for negative half-integers.
#    Port uses trunc(x + copysign(0.5, x)). The .5 case genuinely occurs:
#    row_shift is a multiple of 1/2, so row_shift*usfac is an exact .5 for
#    odd usfac (binary-exact product, no representation fuzz).
# 5. dftups: MATLAB argument `in` renamed `in_` (Python keyword). Argument
#    order/defaults preserved (nor=noc=size(in), usfac=1, roff=coff=0).
# 6. Error behavior mapping: FTpad(2D)'s ~ismatrix error -> ValueError with
#    the same message; FTpad(3D) has no check in MATLAB (would index-error on
#    non-3D input) -> explicit ValueError; usfac strictly between 0 and 1 in
#    dftregistrationAlex makes MATLAB die at L128 with an undefined-variable
#    error -> explicit ValueError. dftregistration3d usfac: MATLAB has no
#    validation at all -- zeros([usfac*nr,usfac*nc,usfac*np]) errors on
#    non-integer sizes and silently returns empties for usfac <= 0 (the
#    caller then dies) -> port raises ValueError for those; any usfac > 0
#    whose three products are integers RUNS in MATLAB (e.g. usfac=1.5 on
#    even dims) and is accepted and bit-matched by the port.
# 7. dftregistration3D L6-L8 compute Nr/Nc/Np that are never used (only
#    Nr2/Nc2/Np2 are); dead locals not ported.
# 8. dftregistration3D on a single-plane volume (np == 1): MATLAB does NOT
#    run -- it CRASHES inside FTpad. size() drops trailing singleton dims,
#    so the (nr, nc, 1) product array is 2-D to MATLAB: `center` is 1x2,
#    `centerout` is 1x3 (outsize [2nr 2nc 2] is truly 3-D) and L62
#    `centerout - center` is a dimension-mismatch error; with usfac == 1
#    it dies indexing cenout_cen(3) instead. The port raises ValueError for
#    np_ == 1, and likewise for usfac*np == 1 (zeros([m,n,1]) is 2-D ->
#    same crash). Singleton row/col zeroing (L27-L32) is ported as-is; the
#    plane component is never zeroed because MATLAB never gets that far on
#    single-plane input.
# 9. Nyquist handling: FTpad embeds/crops the fftshifted spectrum verbatim --
#    no Nyquist-bin splitting/halving for even sizes (unlike some other
#    Fourier-resampling codes). Deliberately NOT "fixed"; bit-compatible with
#    MATLAB is the goal.
# 10. FTpad amplitude scale is applied with MATLAB's left-to-right operation
#    order (array * Nout1 * Nout2 [* Nout3] / prod(Nin)) to match rounding.
# 11. Shift sign convention verified in tests/test_dftreg.py: for
#    moving = np.roll(reference, (dy, dx)), the return is (-dy, -dx), i.e.
#    the shift to apply to the moving image. Downstream, DetermineXYShiftsFBS
#    /ApplyXYShiftsFBS feed these into imtranslate as [Col, Row] -- that
#    ordering is those modules' responsibility, not this one's.
# 12. NaN inputs: MATLAB max() skips NaNs and `CCabs == NaN` is all-false,
#    so a NaN anywhere in either FFT (it spreads through the IFFT to the
#    whole correlation) makes find() return EMPTY: MATLAB propagates empties
#    / errors and the caller dies indexing S(1)/S(2). np.argmax instead
#    treats NaN as maximal and the port used to silently return numeric
#    shifts ([0, 0] / [-0.75, -0.75]-style). _argmax_first_colmajor now
#    raises ValueError when the winning peak is NaN (O(1) check on the
#    winner only -- sufficient since NaN always wins np.argmax). usfac == 0
#    never inspects the data and still returns [0, 0], like MATLAB.
#    Unreachable from the pipeline (uint16-sourced finite images); the guard
#    exists so upstream regressions fail loudly instead of registering to
#    garbage.
# 13. Degenerate all-ties inputs (e.g. a constant image) with usfac > 2
#    return a NONZERO shift of -dftshift/usfac (-0.75 for usfac=4) because
#    the refinement window ties everywhere and find-first lands on its
#    corner. That is exactly what MATLAB does; reproduced, not "fixed"
#    (covered by tests/test_dftreg.py::test_2d_constant_image_tie).
# 14. On macOS Accelerate/OpenBLAS builds, the complex matmul in dftups
#    emits spurious "divide by zero/overflow/invalid in matmul"
#    RuntimeWarnings. Originally observed only for degenerate 1xN inputs and
#    left unsuppressed; integration showed it fires on ORDINARY 2-D
#    refinement windows on every pipeline run (macOS 15/numpy 2.0
#    Accelerate). Verified benign: inputs finite, BLAS output finite and
#    matches einsum to ~3e-12; a matmul performs no division, so the flags
#    are definitionally bogus. Now suppressed with np.errstate around the
#    matmul only, matching matlab_compat._resize_along_dim (the package-wide
#    convention for this Accelerate artifact).
# 16. THIS ENGINE STAYS IN DOUBLE, including in the float32 fast mode (port
#    extension, cpstab/precision.py). get_complex_dtype() returns complex128
#    unconditionally and the callers (shifts2d._fft2, orchestrator._fftn,
#    apply_project._fft2) promote their possibly-float32 pixel arrays back to
#    float64 before transforming. That is not caution, it is a measurement:
#    every result here is an ARGMAX, and on the validation data the phase
#    correlation surface is nearly flat -- the DC term is 91% of the peak
#    height and the winning sample beats the runner-up by a median 4.5e-4
#    relative (worst 5.6e-6), while float32 summation noise over the 256x256
#    padded inverse transform is ~3e-5 relative. Single precision therefore
#    does not perturb a shift by an ulp, it flips which grid cell wins and the
#    shift jumps by a whole 1/usfac step. Measured with this engine forced to
#    complex64 and nothing else changed: RS/CS moved by up to 15.5 px and the
#    output projection fell to Pearson r = 0.756. With the engine pinned to
#    double, the same fast-mode run reproduced the float64 shift file and the
#    written uint16 projection BIT FOR BIT. The full numbers and the
#    reproduction live in cpstab/precision.py and cpstab/tests/test_f32.py.
#    Corollary: everything the engine returns is float64 in both modes -- the
#    shifts are exact multiples of 1/usfac and feed float64 bookkeeping -- so
#    no caller inherits a dtype from here.
# 15. dftups brute-force equivalence (tested): out[p, q] equals the FFT of
#    the usfac-times-larger centered embedding of `in_`, sampled at bin
#    ((p - roff) mod usfac*nr, (q - coff) mod usfac*nc). The MATLAB comment's
#    "starting with the [roff+1 coff+1] element" describes the offset with
#    the opposite sign of what the kernels (L164-L165: indices MINUS
#    roff/coff) implement; the port follows the kernels (and the test proves
#    the kernel form is what makes the registration correct).
