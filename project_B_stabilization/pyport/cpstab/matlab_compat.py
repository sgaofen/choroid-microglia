"""MATLAB numeric-compatibility helpers shared across the cpstab port.

This file is owned by the orchestrator port (DFT_warp_3D_2) but is the single
shared home for MATLAB-equivalent numerics: the dftreg / shifts2d / IO modules
must import `matlab_imresize` (and friends) from here instead of rolling their
own, so every module downsamples identically.

Every routine reproduces the exact numeric semantics of its MATLAB namesake
(R2018-era MATLAB + Image Processing Toolbox), as required for line-by-line
equivalence with the Shipley2020 registration pipeline.

Contents
--------
matlab_round            MATLAB round(): half away from zero.
matlab_cast_like        MATLAB double->class cast (integer: round half away
                        from zero + saturate); shared by every class-
                        preserving image op in the package.
matlab_imresize         MATLAB imresize(): cubic a=-0.5 kernel, antialiasing
                        on downscale, 'nearest' method, scalar scale or
                        explicit output size, N-D input (first two dims only).
matlab_imtranslate      MATLAB imtranslate(A, [C, R]): 2-D, 'linear',
                        FillValues=0, same-size output.
matlab_imwarp_affine2d  MATLAB imwarp(A, affine2d(T), 'OutputView',
                        imref2d(size(A))): 'linear', FillValues=0.

All functions compute internally in float64 and, like their MATLAB
namesakes, PRESERVE the input class: integer input comes back in the same
integer class (rounded half-away-from-zero and saturated, i.e. MATLAB's
double->intN cast); float64 passes through. This class round-trip is
essential for numerical equivalence: in the MATLAB pipeline the uint16
frames stay uint16 through imresize/imwarp/imtranslate and each op
re-quantizes its output before the next FFT sees it.
"""

import math

import numpy as np
from scipy import ndimage

from .precision import work_dtype_for

__all__ = [
    "matlab_round",
    "matlab_cast_like",
    "matlab_imresize",
    "matlab_imtranslate",
    "matlab_imwarp_affine2d",
]


def matlab_round(x):
    """MATLAB round(): round half AWAY FROM ZERO (not banker's rounding).

    np.round rounds half to even; MATLAB round(2.5)=3, round(-2.5)=-3.
    Works elementwise on arrays and on scalars.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    if out.ndim == 0:
        return float(out)
    return out


def matlab_cast_like(result_f64, dtype):
    """Cast a float64 result back to `dtype` the way MATLAB casts double->intN.

    MATLAB integer casts round half away from zero and saturate; float casts
    are plain precision casts. No-op for float64. (Canonical shared copy —
    shifts2d's imtranslate/_imgaussfilt use this same routine.)
    """
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return result_f64
    if np.issubdtype(dtype, np.integer):
        rounded = np.sign(result_f64) * np.floor(np.abs(result_f64) + 0.5)
        info = np.iinfo(dtype)
        return np.clip(rounded, info.min, info.max).astype(dtype)
    # float32 etc.: MATLAB would have computed in single throughout. In
    # REPLICATE mode we compute in double and cast down (documented
    # divergence, shifts2d PORTING NOTES #3). In float32 FAST mode the
    # callers already computed in single (work_dtype_for), so this is a
    # no-op astype and the divergence does not arise.
    return result_f64.astype(dtype)


# ---------------------------------------------------------------------------
# imresize -- faithful reimplementation of MATLAB's imresize.m
# ---------------------------------------------------------------------------
#
# Mirrors MATLAB's "contributions" algorithm exactly:
#   * output size for a scalar scale:      ceil(inputSize * scale)
#   * sample mapping (1-based):            u = x/scale + 0.5*(1 - 1/scale)
#   * left = floor(u - kernel_width/2);  P = ceil(kernel_width) + 2
#   * antialiasing (scale < 1, non-nearest): h(x) = scale*kernel(scale*x),
#     kernel support widened to kernel_width/scale
#   * weights normalized to sum 1 per output pixel BEFORE boundary mirroring
#   * boundary: symmetric mirror  aux = [1:n, n:-1:1]; idx = aux[mod(idx-1,2n)]
#   * cubic kernel = Keys bicubic with a = -0.5 (MATLAB's exact polynomials)
#   * 'nearest' kernel = box on [-0.5, 0.5), antialiasing always off
#
# Deliberate implementation difference (numerically negligible, documented):
# MATLAB may reorder which dimension is resized first as a speed optimization.
# Here dim 0 (rows) is always resized before dim 1 (cols). Separable resampling
# commutes exactly in exact arithmetic; in float64 the difference is O(1e-16)
# and both uses in this pipeline (equal scales; 'nearest', which is exact
# selection with no arithmetic) are insensitive to the order.


def _cubic_kernel(x):
    """MATLAB imresize 'cubic' kernel (Keys, a = -0.5), support width 4."""
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    f = (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1)
    f += (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0) * ((absx > 1) & (absx <= 2))
    return f


def _nearest_kernel(x):
    """MATLAB imresize 'nearest' kernel: box on [-0.5, 0.5), support width 1."""
    return ((x >= -0.5) & (x < 0.5)).astype(np.float64)


_KERNELS = {
    "bicubic": (_cubic_kernel, 4.0),
    "cubic": (_cubic_kernel, 4.0),
    "nearest": (_nearest_kernel, 1.0),
}


def _contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Port of the `contributions` subfunction inside MATLAB imresize.m.

    Returns (weights, indices): both (out_length, P); indices are 1-based and
    already boundary-mirrored. Weights are normalized per row.
    """
    if scale < 1.0 and antialiasing:
        # Widened, rescaled kernel for antialiased shrinking.
        def h(x):
            return scale * kernel(scale * x)

        kw = kernel_width / scale
    else:
        h = kernel
        kw = kernel_width

    # Output pixel coordinates (1-based), mapped into input space.
    x = np.arange(1, out_length + 1, dtype=np.float64)[:, None]
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kw / 2.0)
    P = int(math.ceil(kw)) + 2
    indices = (left + np.arange(P, dtype=np.float64)).astype(np.int64)  # 1-based, unmirrored
    weights = h(u - indices)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    # Symmetric mirror boundary (MATLAB: aux = [1:n, n:-1:1]).
    aux = np.concatenate(
        [np.arange(1, in_length + 1, dtype=np.int64), np.arange(in_length, 0, -1, dtype=np.int64)]
    )
    indices = aux[np.mod(indices - 1, aux.size)]
    return weights, indices


def _resize_along_dim(A, dim, weights, indices, in_length):
    """Apply one dimension's contributions via a dense resampling matrix.

    Equivalent to MATLAB's per-dim weighted gather; duplicate (mirrored)
    indices fold by addition, which equals MATLAB's weighted sum over the
    gathered samples.
    """
    out_length, P = weights.shape
    # The resampling matrix is folded in float64 (the kernel weights are a
    # pure function of the geometry, so there is no reason to make THEM less
    # accurate) and cast once to A's working class, so the matmul below runs
    # single-precision in fast mode instead of promoting A back to float64.
    W = np.zeros((out_length, in_length), dtype=np.float64)
    rows = np.repeat(np.arange(out_length), P)
    np.add.at(W, (rows, (indices - 1).ravel()), weights.ravel())
    W = W.astype(A.dtype, copy=False)
    moved = np.moveaxis(A, dim, 0)
    shp = moved.shape
    # np.errstate: Apple Accelerate BLAS raises spurious divide/overflow/
    # invalid FP flags on this matmul for some shapes (verified bit-identical
    # to np.einsum on this machine; matmul performs no division, so the flags
    # are definitionally bogus for this operation).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = W @ moved.reshape(in_length, -1)
    out = out.reshape((out_length,) + shp[1:])
    return np.moveaxis(out, 0, dim)


def matlab_imresize(A, scale=None, output_shape=None, method="bicubic", antialiasing=True):
    """MATLAB imresize() equivalent. Resizes the FIRST TWO dims of an N-D array.

    Parameters
    ----------
    A : ndarray, N-D (N >= 2). Computed in float64 internally; the INPUT
        CLASS IS PRESERVED like MATLAB imresize (integer input -> rounded
        half-away-from-zero, saturated, same integer class on output; the
        pipeline relies on this uint16 re-quantization, DFT_warp_3D_2.m L41).
        Dims 3+ pass through untouched (matches MATLAB: "imresize only
        resizes the first two dimensions").
    scale : float, optional. Scalar scale factor for both dims; output size is
        ceil(inputSize * scale), exactly as MATLAB.
    output_shape : tuple (rows, cols), optional. Explicit output size for the
        first two dims (MATLAB imresize(A, [numrows numcols])). Exactly one of
        `scale` / `output_shape` must be given.
    method : 'bicubic' (default, == MATLAB default) or 'nearest'.
    antialiasing : bool, default True (MATLAB default for non-nearest).
        Only takes effect when shrinking; forced off for 'nearest'
        (MATLAB's default for nearest -- the pipeline never overrides it).

    Returns
    -------
    ndarray of A's input class with dims (out_rows, out_cols, *A.shape[2:]).
    """
    A = np.asarray(A)
    in_dtype = A.dtype
    # Internal working class: float64 in replicate mode, float32 in fast mode
    # (a float64 INPUT always stays float64 -- see precision.work_dtype_for;
    # that is what keeps the inter-chunk shift-vector stretch in double).
    A = A.astype(work_dtype_for(in_dtype), copy=False)
    if A.ndim < 2:
        raise ValueError("matlab_imresize expects an array with >= 2 dims")
    in_shape = A.shape[:2]

    if (scale is None) == (output_shape is None):
        raise ValueError("give exactly one of `scale` or `output_shape`")
    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        out_shape = tuple(int(math.ceil(s * scale)) for s in in_shape)
        scales = (float(scale), float(scale))
    else:
        out_shape = (int(output_shape[0]), int(output_shape[1]))
        scales = (out_shape[0] / in_shape[0], out_shape[1] / in_shape[1])

    method = method.lower()
    if method not in _KERNELS:
        raise ValueError("method must be one of %s" % sorted(_KERNELS))
    kernel, kernel_width = _KERNELS[method]
    if method == "nearest":
        antialiasing = False

    B = A
    for dim in (0, 1):
        weights, indices = _contributions(
            in_shape[dim], out_shape[dim], scales[dim], kernel, kernel_width, antialiasing
        )
        B = _resize_along_dim(B, dim, weights, indices, in_shape[dim])
    # MATLAB imresize preserves the input class: integer classes are rounded
    # half-away-from-zero and saturated AFTER the (double) resampling. This
    # quantization feeds the MATLAB registration chain (uint16 downsampled
    # frames) and must not be skipped.
    return matlab_cast_like(B, in_dtype)


# ---------------------------------------------------------------------------
# imtranslate / imwarp -- linear interpolation with MATLAB fill semantics
# ---------------------------------------------------------------------------
#
# MATLAB imwarp/imtranslate ('linear', FillValues=0) run through
# images.internal.interp2d, which PADS the image with one ring of the fill
# value and interpolates over the padded grid: a query point within one pixel
# outside the domain BLENDS interior pixels with the fill (the familiar soft
# edge of imrotate 'bilinear'); only queries beyond the padded ring are pure
# fill. scipy.ndimage's mode='grid-constant' has exactly this
# pad-then-interpolate semantics (mode='constant' does NOT: it snaps edge
# queries wholly to cval), so the functions below use mode='grid-constant'
# and need no extra masking -- far-outside queries interpolate between
# padding values and come out as the fill on their own. This matches
# shifts2d.imtranslate, which was verified against a manual
# bilinear-over-zero-padding reference (shifts2d PORTING NOTES #2).
# [Review F3: the previous hard out-of-domain mask here contradicted
# shifts2d on the <=1 px boundary ring and was the wrong side of MATLAB's
# semantics; final confirmation against a live MATLAB
# imtranslate(ones(4)*100,[0.5 0]) is still pending on a machine that has
# one -- expected edge value 50, not 0.]


def matlab_imtranslate(img, translation):
    """MATLAB imtranslate(img, [C, R]) for a 2-D image.

    Parameters
    ----------
    img : 2-D ndarray [Y, X] (rows, cols). Computed in float64; the input
        class is preserved like MATLAB (integer classes are re-quantized
        with round-half-away-from-zero + saturation).
    translation : sequence (C, R) -- MATLAB [x, y] order: C shifts columns
        (positive = content moves right), R shifts rows (positive = down).
        CAREFUL: this is the [Col, Row] argument order of the MATLAB call
        sites (DFT_rect.m L15/L26, ApplyXYShiftsFBS.m L22), NOT [row, col].

    Interpolation 'linear', FillValues=0, same-size output ('same').
    Integer translations are exact (no interpolation error). Fractional
    translations blend the fill value into the <=1 px boundary ring
    (see the module comment above on MATLAB's padded-grid semantics).
    """
    img = np.asarray(img)
    in_dtype = img.dtype
    imgf = img.astype(work_dtype_for(in_dtype), copy=False)
    if imgf.ndim != 2:
        raise ValueError("matlab_imtranslate expects a 2-D image")
    c_shift = float(translation[0])
    r_shift = float(translation[1])
    H, W = imgf.shape
    r_in = np.arange(H, dtype=np.float64)[:, None] - r_shift  # broadcast rows
    c_in = np.arange(W, dtype=np.float64)[None, :] - c_shift  # broadcast cols
    r_grid = np.broadcast_to(r_in, (H, W))
    c_grid = np.broadcast_to(c_in, (H, W))
    out = ndimage.map_coordinates(
        imgf, [r_grid, c_grid], order=1, mode="grid-constant", cval=0.0
    )
    return matlab_cast_like(out, in_dtype)


def matlab_imwarp_affine2d(img, T):
    """MATLAB imwarp(img, affine2d(T), 'OutputView', imref2d(size(img))).

    Parameters
    ----------
    img : 2-D ndarray [Y, X]. Computed in float64; the input class is
        preserved like MATLAB imwarp (integer classes re-quantized with
        round-half-away-from-zero + saturation).
    T : (3, 3) MATLAB affine2d T matrix, ROW-VECTOR convention:
        [x_out y_out 1] = [x_in y_in 1] * T, so translations live in T[2, 0:2]
        and the last COLUMN must be [0, 0, 1].

    'linear' interpolation, FillValues=0 (both MATLAB defaults), output view
    identical to the input frame (world == intrinsic coordinates). The output
    grid is inverse-mapped through T and sampled bilinearly over the
    fill-padded grid; queries near/beyond the border blend with / become the
    fill value (see module comment on MATLAB's padded-grid fill semantics).
    """
    img = np.asarray(img)
    in_dtype = img.dtype
    imgf = img.astype(work_dtype_for(in_dtype), copy=False)
    if imgf.ndim != 2:
        raise ValueError("matlab_imwarp_affine2d expects a 2-D image")
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (3, 3) or not np.allclose(T[:, 2], [0.0, 0.0, 1.0]):
        raise ValueError("T must be a 3x3 affine2d matrix with last column [0,0,1]")

    # Column-vector forward map A = T'; inverse-map output points to input.
    a = np.linalg.inv(T.T)
    H, W = imgf.shape
    # Output intrinsic coords (1-based): x = col+1, y = row+1.
    c0, r0 = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_out = c0 + 1.0
    y_out = r0 + 1.0
    x_in = a[0, 0] * x_out + a[0, 1] * y_out + a[0, 2]
    y_in = a[1, 0] * x_out + a[1, 1] * y_out + a[1, 2]
    r_in = y_in - 1.0
    c_in = x_in - 1.0
    out = ndimage.map_coordinates(
        imgf, [r_in, c_in], order=1, mode="grid-constant", cval=0.0
    )
    return matlab_cast_like(out, in_dtype)


# PORTING NOTES
# -------------
# * matlab_imresize follows imresize.m's `contributions` algorithm verbatim:
#   u = x/scale + 0.5*(1-1/scale); left = floor(u - kw/2); P = ceil(kw)+2;
#   weights normalized BEFORE symmetric-mirror index folding (that ordering is
#   what MATLAB does and it matters at the borders). The per-row weighted sum
#   is realized as a dense (out x in) matrix multiply; mirrored duplicate
#   indices fold by addition, which is algebraically identical to MATLAB's
#   gather-then-dot. Residual float differences vs MATLAB are only summation-
#   order effects, O(eps).
# * Dimension processing order is fixed rows-then-cols; MATLAB may pick the
#   order for speed. Exact-arithmetic equivalent; O(1e-16) in float64; both
#   call sites in this pipeline (equal scales / nearest) are insensitive.
# * 'nearest' via the box kernel reduces to floor(u+0.5) (round half up) with
#   MATLAB's inclusive-left/exclusive-right box — verified this matches
#   imresize nearest, including the exact-half tie direction.
# * matlab_imtranslate / matlab_imwarp_affine2d boundary model (review F3):
#   MATLAB imwarp/imtranslate go through images.internal.interp2d, which pads
#   the image with one ring of FillValues and interpolates over the padded
#   grid -- the fill BLENDS into the <=1 px boundary ring for fractional
#   coordinates (imrotate 'bilinear' soft edge). scipy mode='grid-constant'
#   implements exactly that; the earlier hard out-of-domain mask here was the
#   opposite (snap-to-fill) semantics and contradicted the independently
#   verified shifts2d.imtranslate on the boundary ring. Interior pixels are
#   unchanged by this fix. One live-MATLAB spot check
#   (imtranslate(ones(4)*100,[0.5 0]) -> edge 50) remains outstanding.
# * Class preservation (review F2): imresize/imtranslate/imwarp preserve the
#   input class exactly as MATLAB does, via matlab_cast_like (round half away
#   from zero + saturate for integer classes). The registration chain feeds
#   uint16 through imresize and imwarp before DFT_rect, and MATLAB's
#   re-quantization at those two spots measurably changes the estimated
#   subpixel shifts at realistic SNR -- do not "simplify" back to
#   float64-everywhere.
# * Only 2-D images are supported by imtranslate/imwarp here because every
#   MATLAB call site in this pipeline warps/translates single slices.
# * Internal arithmetic is float64 by default (MATLAB computes these ops in
#   double before casting back to the input class). The float32 FAST MODE
#   (cfg.compute_dtype='float32', cpstab/precision.py) lowers that internal
#   class to single via work_dtype_for(): a uint16 input is promoted to
#   float32 instead of float64, resampled, and cast back to uint16 exactly as
#   before. Two consequences worth knowing:
#     - the CLASS-PRESERVING contract is unchanged in both modes (uint16 in,
#       uint16 out), so MATLAB's re-quantization points still exist; only the
#       value under the round-half-away tie can differ, by +-1 count;
#     - a float64 input is never demoted (work_dtype_for), which is why the
#       'nearest' stretch of the inter-chunk shift vectors (DFT_warp_3D_2.m
#       L127-129) keeps full double precision even in fast mode.
#   The resampling matrix W is always folded in float64 and cast to the
#   working class once, so the kernel weights themselves are mode-independent.
