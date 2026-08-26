# -*- coding: utf-8 -*-
"""apply_project — port of registration/MakeSBXall.m + registration/zproj_reg.m.

Shipley 2020 choroid-plexus z-stack stabilization, "apply + project" stage:
read the raw .sbx volume-by-volume, apply the optotune warp + per-plane XY
translation + per-volume Z circshift estimated by DFT_warp_3D_2, accumulate a
z-projection time series, refine it with a 3-pass DFT time-series
stabilization (zproj_reg), and optionally write the registered stack to
'<base>.sbxall'.

Public functions (signatures mirror the MATLAB originals; see rule 8):

    make_sbxall(path, shiftpath, ...)   <-> registration/MakeSBXall.m  L1-L204
    zproj_reg(k, nt, pmt, otrange, ...) <-> registration/zproj_reg.m   L1-L66

registration/zproj.m (113 lines) is DEAD CODE (its only caller is the
bypassed `pipe.zproj` branch of zproj_reg; MakeSBXall always supplies
zproj_raw).  It is NOT ported; its NaN/omitnan projection semantics are
however reproduced here as the optional `omitnan=True` mode of the fused
projection (see make_sbxall docstring).

Single-pass design (per module brief):
  * PASS1 (MakeSBXall.m L72-123) + the zproj_reg refinement (L125) keep the
    original semantics and ordering.
  * PASS2 (MakeSBXall.m L139-202, the .sbxall write) is behind
    `write_registered=False` (the file has no consumer inside the repo).
    When enabled, the raw stack is by default read ONCE: the
    lineshift+crop+warp intermediate (`warpvol`, which PASS2 recomputes
    bit-identically from the same raw bytes) is cached during PASS1 and the
    write phase only REPLAYS the shift application with the refined totals.
    This preserves the ordering dependency (zproj_reg adjusts RS/CS used by
    the write) and is numerically identical to the MATLAB two-pass loop.

Array conventions (rule 9): single frame [Y, X] (= MATLAB [row, col]);
volume [Y, X, Z]; time block [Y, X, Z, T].  The channel axis, where present,
is a LEADING axis mirroring MATLAB's (Nc, Nx, Ny, ...) layout; note MATLAB's
"Nx" is info.sz(1) = rows = Y and "Ny" is info.sz(2) = cols = X.

Inter-module calls: the IO layer (sbx_info / imread / spoof_sbx_info_3d /
RegWriter) and the DFT engine (dft_reg / dft_rect) are ported by sibling
modules of this package.  They are resolved lazily by scanning the package
for the snake_case mirror names; every one can also be injected via
keyword arguments, and faithful internal fallbacks are provided for the DFT
engine, SpoofSBXinfo3D and RegWriter (clearly marked, used only when no
sibling is found).  On the CURRENT package layout the resolver binds
shifts2d.dft_reg / shifts2d.dft_rect and io_rw.imread / io_rw.sbx_info, so
the internal fallbacks are dead code in practice; they are kept for layout
independence and are verified bit-identical to the siblings
(tests/scratch_apply_project.py A2/A3).

Target: Python 3.9, numpy + scipy only (skimage/tifffile not needed here).
"""

import importlib
import os
import pkgutil
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor as _ThreadPool


def _tick(_state=[None]):
    """Env-gated stopwatch for the serial tail (CPSTAB_TIMING=1).

    Pure instrumentation: it only reads a clock and prints. No array is
    touched, so it cannot move a pixel.
    """
    if not os.environ.get("CPSTAB_TIMING"):
        return lambda _label: None
    _state[0] = _time.time()

    def lap(label):
        now = _time.time()
        print("[timing]   %-22s %6.1fs" % (label, now - _state[0]),
              flush=True)
        _state[0] = now
    return lap

import numpy as np
import scipy.ndimage as _ndi

# Algorithm mode: 'replicate' (default, MATLAB-faithful) or 'improved'.  Three
# of the four corrections live in this module -- see PORTING NOTES #17.
from . import improved as _improved
from .bilinear import bilinear_shift2 as _fast_shift2
from .fourier_shift import fshift2_vst as _fshift2_vst
# Float working class of the image domain: float64 (replicate, default) or
# float32 (fast mode).  See cpstab/precision.py and PORTING NOTES #16.
from .precision import (as_correlation, as_float, get_complex_dtype,
                        work_dtype_for, zeros as _fzeros)


def _fft2(a):
    """fft2 for the registration engine: ALWAYS float64 -> complex128.

    Mirror of shifts2d._fft2 for this module's internal fallback engine, so
    the fallback stays bit-identical to the sibling in fast mode too. See
    cpstab/precision.py: fast mode lowers pixels, never the correlation that
    decides a shift.
    """
    return np.fft.fft2(as_correlation(a))

__all__ = ["make_sbxall", "zproj_reg"]

_U16MAX = 65535.0


# =========================================================================
# MATLAB-semantics primitives
# =========================================================================

def _matlab_round(x):
    """MATLAB round(): nearest integer, ties away from zero (scalar or array)."""
    a = np.asarray(x, dtype=np.float64)
    r = np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))
    if r.ndim == 0:
        return float(r)
    return r


def _matlab_truthy(v):
    """Truthiness of a MATLAB `if` on a parameter (MakeSBXall.m L82 `if p.optotune`).

    MATLAB: char arrays are true iff nonempty (so BOTH 'true' AND 'false' are
    true!); numerics/logicals are true iff nonempty and all nonzero.
    """
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return len(v) > 0
    a = np.asarray(v)
    return a.size > 0 and bool(np.all(a != 0))


def _median_matlab_default(x):
    """MATLAB median(X) with default dim (MakeSBXall.m L29/L32/L35).

    Vector (or 1-row matrix) -> scalar median of all elements.
    Matrix with >1 row      -> per-column median (row vector), which then
    broadcasts over rows exactly like MATLAB implicit expansion.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim <= 1 or a.shape[0] == 1:
        return float(np.median(a))
    return np.median(a, axis=0)


def _median_centering(x):
    """The value subtracted from a shift matrix to centre it (MakeSBXall.m
    L29/L32/L35).  Mode-dependent -- cpstab/improved.py correction 1.

    replicate: MATLAB's median() at its default dim, i.e. a PER-COLUMN
      (per-timepoint) median for the (Nz, Nt) RS/CS matrices.  That is the
      MATLAB behaviour and it is also the bug: any shift term that is constant
      along the plane axis is subtracted straight back out.  Both the
      per-volume 3-D registration (RS2/CS2, tiled over planes at
      DFT_warp_3D_2.m L83-84) and the inter-chunk stitch (RS_chunk/CS_chunk, an
      imresize 'nearest' stretch of a 1 x Nchunks vector) are exactly such
      terms, so both are annihilated:

          (RS + RS_chunk) - median(RS + RS_chunk, dim=1)
              == RS - median(RS, dim=1)

      measured exact to 3.6e-15 px on the validation subset.

    improved: a GLOBAL scalar median over the whole matrix.  Centring exists
      only to remove the arbitrary absolute origin of the shift estimates; one
      scalar does that and leaves every relative motion -- between planes,
      between volumes, between chunks -- intact.

    ZS is unaffected in either mode: it is a (1, Nt) row vector, where MATLAB's
    default-dim median is already the scalar median of all elements, so both
    branches return the same number for it.
    """
    if _improved.use_global_median():
        return float(np.median(np.asarray(x, dtype=np.float64)))
    return _median_matlab_default(x)


def _matlab_prctile(x, p):
    """MATLAB prctile(x(:), p) (zproj_reg.m L31).

    Sorted data points sit at percentile positions 100*(i-0.5)/n; linear
    interpolation in between; clamped to the extremes outside.  (numpy's
    default 'linear' percentile uses different positions — do not swap.)
    """
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = v.size
    if n == 0:
        raise ValueError("prctile of empty array")
    if n == 1:
        return float(v[0])
    pos = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    if p <= pos[0]:
        return float(v[0])
    if p >= pos[-1]:
        return float(v[-1])
    return float(np.interp(p, pos, v))


def _matlab_rescale(a, inmin, inmax):
    """MATLAB rescale(A,'InputMin',L,'InputMax',U) -> [0,1] (zproj_reg.m L32).

    Input is clamped to [L, U] first, then mapped linearly to [0, 1].
    Degenerate U <= L (near-constant slice): returns zeros — see PORTING
    NOTES (#7).
    """
    a = as_float(a)
    if not (inmax > inmin):
        return np.zeros_like(a)
    c = np.clip(a, inmin, inmax)
    return (c - inmin) / (inmax - inmin)


def _quantize_u16(v):
    """MATLAB cast of nonnegative doubles to uint16: round half away from
    zero + saturate to [0, 65535].  Returned as a float of the COMPUTE class
    holding integer values (they are subsequently assigned into double
    arrays, mirroring e.g. MakeSBXall.m L106 where a uint16 slice lands in a
    double reg_vol).

    The rounding stays exact in float32 fast mode: every value it produces is
    an integer <= 65535 (17 bits, well inside float32's 24-bit significand)
    and the tie point 65535.5 is representable, so floor(x + 0.5) loses
    nothing.  Only the bilinear x handed to it can differ by an ulp from the
    float64 run — that is the mechanism behind the +-1-count pixels.
    """
    return np.clip(np.floor(as_float(v) + 0.5), 0.0, _U16MAX)


def _imtranslate_float(img, c_shift, r_shift):
    """MATLAB imtranslate(img,[C,R]) on a double image (linear, FillValues=0,
    'OutputView' 'same').  NOTE argument order: MATLAB takes [Col, Row] = [x, y];
    scipy.ndimage.shift takes (row, col) — swapped here, order=1, cval=0.
    output(y, x) = input(y - R, x - C).
    mode='grid-constant' (NOT 'constant'): MATLAB pads with the fill value
    and interpolates across the boundary, so the fill blends into the <=1 px
    edge ring on fractional shifts; scipy's plain 'constant' would clamp
    that ring to 0 instead (verified: shift [10,20,30] by 0.5 -> MATLAB
    [5,15,25], 'grid-constant' [5,15,25], 'constant' [0,15,25]).
    """
    img = as_float(img)
    if img.ndim == 2:
        if img.dtype == np.float64:
            return _fast_shift2(img, float(r_shift), float(c_shift))
        if img.dtype == np.float32:
            # scipy accumulates in double whatever the input class is and
            # rounds ONCE on the store, so fast mode is that same double
            # accumulation with one cast at the end -- not float32
            # arithmetic, which would round four more times and disagree.
            # Verified bitwise against scipy over 3324 float32 cases.
            return _fast_shift2(np.asarray(img, dtype=np.float64),
                                float(r_shift),
                                float(c_shift)).astype(np.float32)
    return _ndi.shift(img, (r_shift, c_shift),
                      order=1, mode="grid-constant", cval=0.0, prefilter=False)


def _imtranslate_u16(img, c_shift, r_shift):
    """MATLAB imtranslate on a uint16 image: bilinear in float, then cast back
    to uint16 (round half away + saturate).  This quantization is part of the
    ground-truth numerics (MakeSBXall.m L101/L168 operate on uint16 slices)."""
    return _quantize_u16(_imtranslate_float(img, c_shift, r_shift))


def _is_identity_tform(t):
    return t is None or np.array_equal(np.asarray(t, dtype=np.float64), np.eye(3))


def _imwarp_affine2d_u16(img_u16, t):
    """MATLAB imwarp(slice, affine2d(T), 'OutputView', imref2d(size(slice)))
    on a uint16 slice (MakeSBXall.m L86/L153): linear interp, fill 0,
    output = input grid; then cast back to uint16.

    affine2d convention: [x_out y_out 1] = [x_in y_in 1] * T (x = col,
    y = row, 1-based pixel-center coordinates).  Inverse mapping: for each
    output pixel, sample input at [x_o y_o 1] * inv(T).
    Exact-identity transforms short-circuit to a copy (bit-exact, and what
    MATLAB effectively produces for uint16 identity warps).
    """
    if _is_identity_tform(t):
        return as_float(img_u16)
    t = np.asarray(t, dtype=np.float64)
    tinv = np.linalg.inv(t)
    rows, cols = img_u16.shape
    xo, yo = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                         np.arange(1, rows + 1, dtype=np.float64))
    xi = xo * tinv[0, 0] + yo * tinv[1, 0] + tinv[2, 0]
    yi = xo * tinv[0, 1] + yo * tinv[1, 1] + tinv[2, 1]
    sampled = _ndi.map_coordinates(as_float(img_u16),
                                   [yi - 1.0, xi - 1.0],
                                   order=1, mode="grid-constant", cval=0.0,
                                   prefilter=False)  # grid-constant: MATLAB
    # blends FillValues into the boundary ring (see _imtranslate_float).
    return _quantize_u16(sampled)


# -------------------------------------------------------------------------
# MATLAB imresize (bicubic a=-0.5, antialiasing on when shrinking)
# -------------------------------------------------------------------------

def _imresize_cubic_kernel(x):
    """MATLAB images.internal.resize cubic kernel (Keys, a = -0.5)."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    ax2 = ax * ax
    ax3 = ax2 * ax
    f = np.where(ax <= 1, 1.5 * ax3 - 2.5 * ax2 + 1.0, 0.0)
    f = np.where((ax > 1) & (ax <= 2), -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0, f)
    return f


def _imresize_contributions(in_len, out_len, scale, antialias):
    """Weights/indices exactly as MATLAB imresize 'contributions':
    u = x/scale + 0.5*(1 - 1/scale) with 1-based x; symmetric boundary."""
    kw = 4.0
    if antialias and scale < 1.0:
        def h(x):
            return scale * _imresize_cubic_kernel(scale * x)
        kw = kw / scale
    else:
        h = _imresize_cubic_kernel
    x = np.arange(1, out_len + 1, dtype=np.float64)[:, None]     # (out,1), 1-based
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kw / 2.0)                                # 1-based
    p = int(np.ceil(kw)) + 2
    indices = left + np.arange(p, dtype=np.float64)[None, :]     # (out,P), 1-based
    weights = h(u - indices)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    idx0 = aux[np.mod(indices.astype(np.int64) - 1, aux.size)]   # 0-based, mirrored
    return weights, idx0


def _resize_along_axis(a, weights, idx0, axis):
    taken = np.take(a, idx0, axis=axis)                # axis -> (out, P)
    shape = [1] * taken.ndim
    shape[axis] = weights.shape[0]
    shape[axis + 1] = weights.shape[1]
    # The kernel weights are always computed in float64 (they are a pure
    # function of the geometry); cast them to the data's working class so a
    # float32 fast-mode stack is not promoted back to double here.
    w = weights.astype(taken.dtype, copy=False)
    return np.sum(taken * w.reshape(shape), axis=axis + 1)


# -------------------------------------------------------------------------
# Serial-tail scheduling helpers.
#
# zproj_reg runs ONCE, in the parent, after the worker pool has been shut
# down, on a (Nc, Y, X, Nt) stack -- 5.9 GB at the production size.  Its
# per-frame work is embarrassingly parallel and was measured at 46 of the
# 59 s tail (36.6 s of shifts + 9.6 s of imresize), so the parent sitting on
# one core there is the single largest idle-core window in the run.
#
# THREADS, not processes: the stack is already in the parent's address
# space, shipping it to workers would cost more than the work, and both the
# scipy interpolation and the numpy ufuncs underneath release the GIL.
# CPSTAB_TAIL_THREADS overrides the width -- set it to 1 in any context that
# is itself already inside a pool.
# -------------------------------------------------------------------------
def _tail_threads():
    n = os.environ.get("CPSTAB_TAIL_THREADS")
    if n:
        try:
            return max(1, int(n))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 18))


def _tail_block_len(nt, per_thread=3):
    """Frames per block: enough blocks to even out the tails, not so many
    that each one re-pays the transpose setup."""
    nb = max(1, _tail_threads() * int(per_thread))
    return max(1, int(np.ceil(float(nt) / nb)))


def _run_blocks(fn, nt, block):
    """fn(t0, t1) over [0, nt) in `block`-sized pieces, on a thread pool."""
    spans = [(t0, min(t0 + block, nt)) for t0 in range(0, nt, block)]
    nw = min(_tail_threads(), len(spans))
    if nw <= 1 or len(spans) == 1:
        for t0, t1 in spans:
            fn(t0, t1)
        return
    with _ThreadPool(max_workers=nw) as ex:
        list(ex.map(lambda s: fn(s[0], s[1]), spans))


def _imresize_over_t(a, scale):
    """_matlab_imresize on a (Y, X, Nt) stack, one T block per thread.

    Bit-exact by construction: _matlab_imresize resizes the FIRST TWO axes
    and treats every trailing index as an independent channel, so each
    output element is a fixed weighted sum over one frame's pixels, in an
    order that does not depend on how many frames sit beside it.  Verified
    bitwise against the one-shot call on the production shape.

    It is also the difference between a 14 GB transient and a bounded one:
    the row pass materializes np.take(a, idx0, axis=0) with P = ceil(4/scale)
    + 2 = 18 taps, i.e. 18x the stack, which at (512, 512, 1500) doubles is
    14 GB in a single temporary.
    """
    a = as_float(a)
    if a.ndim != 3 or a.shape[2] < 2:
        return _matlab_imresize(a, scale)
    nt = a.shape[2]
    out = [None] * nt
    block = _tail_block_len(nt)
    spans = [(t0, min(t0 + block, nt)) for t0 in range(0, nt, block)]
    pieces = [None] * len(spans)

    def one(k):
        t0, t1 = spans[k]
        pieces[k] = _matlab_imresize(np.ascontiguousarray(a[:, :, t0:t1]),
                                     scale)

    nw = min(_tail_threads(), len(spans))
    if nw <= 1:
        for k in range(len(spans)):
            one(k)
    else:
        with _ThreadPool(max_workers=nw) as ex:
            list(ex.map(one, range(len(spans))))
    return np.concatenate(pieces, axis=2)


def _matlab_imresize(a, scale):
    """MATLAB imresize(A, scale) for 2-D or 3-D A (first two dims resized,
    trailing dim(s) treated as channels — zproj_reg.m L26 resizes a
    [Y, X, Nt] stack).  Bicubic a=-0.5; antialiasing=true when scale<1
    (MATLAB's default for bicubic); output size = ceil(scale * insize);
    symmetric boundary padding.  Written by hand because scipy/skimage have
    no equivalent of the antialiased MATLAB kernel (porting trap #2).
    Equal row/col scales -> rows resized first, then cols (MATLAB order for
    uniform scale).

    Unlike matlab_compat.matlab_imresize this one is NOT class-preserving and
    is only ever handed image data (the zproj stack, zproj_reg.m L26), so it
    works in the compute class unconditionally — there is no shift-vector
    call site here to protect."""
    a = as_float(a)
    out0 = int(np.ceil(a.shape[0] * scale))
    out1 = int(np.ceil(a.shape[1] * scale))
    w0, i0 = _imresize_contributions(a.shape[0], out0, float(scale), True)
    w1, i1 = _imresize_contributions(a.shape[1], out1, float(scale), True)
    r = _resize_along_axis(a, w0, i0, axis=0)
    r = _resize_along_axis(r, w1, i1, axis=1)
    return r


# =========================================================================
# Internal fallback DFT engine (used only when no sibling module provides
# dft_reg / dft_rect).  Faithful ports of dftregistrationAlex.m,
# DFT_reg.m, DFT_rect.m.
# =========================================================================

def _argmax_colmajor(a):
    """MATLAB find(A == max(A(:)), 1, 'first'): first maximum in
    COLUMN-major order.  Returns 0-based (row, col)."""
    idx = int(np.argmax(a.ravel(order="F")))
    return np.unravel_index(idx, a.shape, order="F")


def _ftpad(im_ft, outsize):
    """dftregistrationAlex.m L170-L203 (FTpad): pad/crop a DC-in-(1,1)
    Fourier transform to `outsize`, energy-rescaled."""
    nin = im_ft.shape
    nout = tuple(int(n) for n in outsize)
    a = np.fft.fftshift(im_ft)
    center = (nin[0] // 2, nin[1] // 2)          # MATLAB floor(n/2)+1, 1-based
    centerout = (nout[0] // 2, nout[1] // 2)
    co0 = centerout[0] - center[0]
    co1 = centerout[1] - center[1]
    out = np.zeros(nout, dtype=a.dtype)
    dr0, dr1 = max(co0 + 1, 1), min(co0 + nin[0], nout[0])
    dc0, dc1 = max(co1 + 1, 1), min(co1 + nin[1], nout[1])
    sr0, sr1 = max(-co0 + 1, 1), min(-co0 + nout[0], nin[0])
    sc0, sc1 = max(-co1 + 1, 1), min(-co1 + nout[1], nin[1])
    out[dr0 - 1:dr1, dc0 - 1:dc1] = a[sr0 - 1:sr1, sc0 - 1:sc1]
    return np.fft.ifftshift(out) * (nout[0] * nout[1]) / (nin[0] * nin[1])


def _dftups(inp, nor, noc, usfac, roff, coff):
    """dftregistrationAlex.m L131-L167 (dftups): matrix-multiply upsampled DFT."""
    nr, nc = inp.shape
    cdt = np.asarray(inp).dtype
    kernc = np.exp((-2j * np.pi / (nc * usfac)) *
                   np.outer(np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2.0),
                            np.arange(noc) - coff)).astype(cdt, copy=False)
    kernr = np.exp((-2j * np.pi / (nr * usfac)) *
                   np.outer(np.arange(nor) - roff,
                            np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2.0))
                   ).astype(cdt, copy=False)
    # Some BLAS backends (macOS Accelerate) emit spurious divide/overflow
    # RuntimeWarnings on complex GEMM; the result is verified finite below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = kernr @ inp @ kernc
    if not np.isfinite(out).all():
        raise FloatingPointError("dftups produced non-finite values")
    return out


def _dftregistration_alex(buf1ft, buf2ft, usfac=1):
    """dftregistrationAlex.m L1-L129: Guizar subpixel DFT registration.

    buf1ft/buf2ft: fft2 of reference / moving image (DC at [0,0]).
    Returns np.array([row_shift, col_shift]) such that translating the
    moving image BY (row_shift, col_shift) (imtranslate([col, row])) aligns
    it to the reference.  complex128 throughout.
    """
    cdt = get_complex_dtype()
    buf1ft = np.asarray(buf1ft, dtype=cdt)
    buf2ft = np.asarray(buf2ft, dtype=cdt)
    nr, nc = buf2ft.shape
    if usfac == 0:
        return np.array([0.0, 0.0])
    if usfac == 1:
        cc = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        r, c = _argmax_colmajor(np.abs(cc))
        nrv = np.fft.ifftshift(np.arange(-(nr // 2), int(np.ceil(nr / 2.0))))
        ncv = np.fft.ifftshift(np.arange(-(nc // 2), int(np.ceil(nc / 2.0))))
        return np.array([float(nrv[r]), float(ncv[c])])
    # usfac > 1: zero-padded 2x FFT first
    cc = np.fft.ifft2(_ftpad(buf1ft * np.conj(buf2ft), (2 * nr, 2 * nc)))
    r, c = _argmax_colmajor(np.abs(cc))
    nr2 = np.fft.ifftshift(np.arange(-nr, nr))
    nc2 = np.fft.ifftshift(np.arange(-nc, nc))
    row_shift = nr2[r] / 2.0
    col_shift = nc2[c] / 2.0
    if usfac > 2:
        row_shift = _matlab_round(row_shift * usfac) / usfac
        col_shift = _matlab_round(col_shift * usfac) / usfac
        upn = int(np.ceil(usfac * 1.5))
        dftshift = int(np.fix(upn / 2.0))
        cc = np.conj(_dftups(buf2ft * np.conj(buf1ft), upn, upn, usfac,
                             dftshift - row_shift * usfac,
                             dftshift - col_shift * usfac))
        r, c = _argmax_colmajor(np.abs(cc))
        rloc = (r + 1) - dftshift - 1          # back to MATLAB 1-based math
        cloc = (c + 1) - dftshift - 1
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac
    if nr == 1:
        row_shift = 0.0
    if nc == 1:
        col_shift = 0.0
    return np.array([float(row_shift), float(col_shift)])


def _fallback_dft_reg(stack, target, upscale):
    """DFT_reg.m L1-L17 (fallback): register every slice of [Y,X,N] `stack`
    to `target`; returns (R, C, reg) with reg the translated (double,
    unquantized) stack."""
    stack = as_float(stack)
    n = stack.shape[2]
    reg = np.zeros_like(stack)
    rr = np.zeros(n)
    cc = np.zeros(n)
    tft = _fft2(target)
    for i in range(n):
        src = stack[:, :, i]
        s = _dftregistration_alex(tft, _fft2(src), upscale)
        rr[i] = s[0]
        cc[i] = s[1]
        reg[:, :, i] = _imtranslate_float(src, s[1], s[0])
    return rr, cc, reg


def _fallback_dft_rect(vol, start, upscale):
    """DFT_rect.m L1-L29 (fallback): plane-to-plane sequential rectification
    outward from 1-BASED plane `start` (MATLAB semantics; the translated
    source becomes the next target).  Index `start` is visited by both the
    forward and backward loops, as in the original."""
    vol = as_float(vol)
    nz = vol.shape[2]
    reg = np.zeros_like(vol)
    rr = np.zeros(nz)
    cc = np.zeros(nz)
    s0 = int(start) - 1
    if not (0 <= s0 < nz):
        raise ValueError("DFT_rect start index out of range (1-based): %r" % (start,))
    target = vol[:, :, s0]
    for i in range(s0, nz):                    # MATLAB: for i = start:Nz
        src = vol[:, :, i]
        s = _dftregistration_alex(_fft2(target), _fft2(src), upscale)
        rr[i] = s[0]
        cc[i] = s[1]
        target = _imtranslate_float(src, s[1], s[0])
        reg[:, :, i] = target
    target = vol[:, :, s0]
    for i in range(s0, -1, -1):                # MATLAB: for i = flip(1:start)
        src = vol[:, :, i]
        s = _dftregistration_alex(_fft2(target), _fft2(src), upscale)
        rr[i] = s[0]
        cc[i] = s[1]
        target = _imtranslate_float(src, s[1], s[0])
        reg[:, :, i] = target
    return rr, cc, reg


# =========================================================================
# Sibling-module resolution (parallel-port seam)
# =========================================================================

_SIBLING_CACHE = {}


def _resolve_sibling(candidates, fallback=None, what=""):
    """Find a function/class exported by a sibling module of this package
    under any of `candidates` (snake_case mirrors of the MATLAB names).
    Falls back to `fallback` when no sibling provides it; raises ImportError
    when there is no fallback either."""
    key = tuple(candidates)
    if key in _SIBLING_CACHE:
        return _SIBLING_CACHE[key]
    found = None
    pkgname = __package__ if __package__ else None
    myname = os.path.splitext(os.path.basename(__file__))[0]
    if pkgname:
        try:
            pkg = importlib.import_module(pkgname)
            modnames = sorted({m.name for m in pkgutil.iter_modules(getattr(pkg, "__path__", []))})
            for mn in modnames:
                if mn == myname:
                    continue
                try:
                    mod = importlib.import_module(pkgname + "." + mn)
                except Exception:
                    continue
                for cand in candidates:
                    obj = getattr(mod, cand, None)
                    if obj is not None:
                        found = obj
                        break
                if found is not None:
                    break
        except Exception:
            pass
    if found is None:
        found = fallback
    if found is None:
        raise ImportError(
            "apply_project: no sibling module of package %r provides %s "
            "(looked for %s) and no internal fallback exists; pass it "
            "explicitly via the corresponding *_fn / *_cls keyword." %
            (pkgname, what or candidates[0], ", ".join(candidates)))
    _SIBLING_CACHE[key] = found
    return found


def _info_get(info, name, default=None):
    """Field access on an info object that may be a dict, a SimpleNamespace,
    or a scipy.io mat_struct."""
    if isinstance(info, dict):
        return info.get(name, default)
    return getattr(info, name, default)


class _MatStruct(dict):
    """dict with attribute access, for the fallback spoofed info struct."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


# =========================================================================
# Fallback IO pieces (SpoofSBXinfo3D + RegWriter) — used only if no sibling
# IO module is importable; byte contract copied line-by-line.
# =========================================================================

def _fallback_spoof_sbx_info_3d(y_dim, x_dim, z_dim, t_dim, nchan):
    """SpoofSBXinfo3D.m L1-L35 (fallback): fabricate a Scanbox info struct.
    Field quirks preserved verbatim (sz=[yDim,xDim], height=sz(2),
    width=sz(1), recordsPerBuffer=width, channels: 1 if nchan==2 else 2)."""
    info = _MatStruct()
    info.resfreq = 7930
    info.postTriggerSamples = 5000
    info.nchannels = 1
    info.abort_bit = 0
    info.scanbox_version = 2
    info.volscan = 1
    info.opto2pow = []
    info.power_depth_link = 0
    info.area_line = True
    info.sz = np.array([y_dim, x_dim])
    info.height = int(info.sz[1])
    info.width = int(info.sz[0])
    info.nframes = int(z_dim * t_dim)
    info.nchan = int(nchan)
    info.max_idx = info.nframes - 1
    info.recordsPerBuffer = info.width
    info.bytesPerBuffer = info.postTriggerSamples * info.width * 2 * info.nchan
    info.nsamples = info.width * info.height * 2 * info.nchan
    info.channels = 1 if nchan == 2 else 2
    info.scanmode = 1
    info.optotune_used = 1
    info.otlevels = int(z_dim)
    info.otwave = np.arange(1, int(z_dim) + 1)
    return info


class _FallbackRegWriter(object):
    """RegWriter.m L1-L111 (fallback, write-path subset).

    Byte contract mirrored exactly: data (nchan, Y=sz1, X=sz2, T) uint16 is
    permuted [1 3 2 4] (numpy axes (0, 2, 1, 3)), inverted as
    65535 - value, flattened COLUMN-major, written little-endian uint16.
    """

    def __init__(self, path, info, extension=".sbxreg", force=False):
        if not extension.startswith("."):
            extension = "." + extension
        base, name = os.path.split(path)
        name = os.path.splitext(name)[0]
        self.path = os.path.join(base, name + extension)
        if os.path.exists(self.path) and not force:
            raise IOError("Cannot overwrite an existing file unless forced: %s" % self.path)
        self.info = info
        self.curframe = 0
        self.fid = open(self.path, "wb")

    def write(self, data):
        data = np.asarray(data)
        nchan = int(_info_get(self.info, "nchan"))
        sz = np.asarray(_info_get(self.info, "sz")).ravel()
        if data.ndim == 2 and nchan == 1:
            data = data.reshape((1,) + data.shape + (1,))
        elif data.ndim == 3 and nchan == 1:
            data = data.reshape((1,) + data.shape)
        elif data.ndim == 3 and nchan == 2:
            data = data.reshape(data.shape + (1,))
        if data.ndim != 4 or data.shape[0] != nchan \
                or data.shape[1] != int(sz[0]) or data.shape[2] != int(sz[1]):
            raise ValueError("Data must match that declared in info file. "
                             "Check declared nchan! got %r vs nchan=%d sz=%r"
                             % (data.shape, nchan, tuple(sz)))
        nframes = int(_info_get(self.info, "nframes"))
        if data.shape[3] + self.curframe > nframes:
            raise ValueError("writing more frames than declared in info")
        if data.dtype != np.uint16:
            d = np.asarray(data, dtype=np.float64)
            d = np.clip(_matlab_round(d), 0.0, _U16MAX)
            data = d.astype(np.uint16)
        self.curframe += data.shape[3]
        inv = (np.uint16(65535) - data).transpose(0, 2, 1, 3)
        self.fid.write(np.ascontiguousarray(inv.ravel(order="F"),
                                            dtype="<u2").tobytes())

    def close(self):
        self.fid.close()


# =========================================================================
# Shift-file loading
# =========================================================================

def _load_shifts(shiftpath):
    """Load the '.dftshifts' payload written by DFT_warp_3D_2 (MAT-file, per
    DFT_warp_3D_2.m L140) or by its Python port (.npz with the same keys).

    Returns dict with at least RS, CS, ZS, RS_chunk, CS_chunk, ZS_chunk
    (float64 arrays), plus 'tforms_optotune_full' as a numeric array of
    MATLAB-convention T matrices, or None when absent or undecodable.  When
    the variable is PRESENT but cannot be decoded to numerics (MATLAB
    affine2d objects in a v7 MAT-file are opaque to scipy.io.loadmat) the
    extra key 'tforms_optotune_undecodable' is True — make_sbxall decides
    whether that is fatal (it is, whenever the optotune warp would actually
    be applied; silently substituting identity was review finding F1).
    """
    d = None
    try:
        z = np.load(shiftpath, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    except Exception:
        try:
            import scipy.io as _sio
            d = _sio.loadmat(shiftpath, squeeze_me=False, struct_as_record=False)
        except NotImplementedError:
            raise IOError("shift file %r looks like a MATLAB v7.3 (HDF5) MAT-file;"
                          " re-save as v7 or convert to .npz" % (shiftpath,))
    required = ["RS", "CS", "ZS", "RS_chunk", "CS_chunk", "ZS_chunk"]
    out = {}
    for k in required:
        if k not in d:
            raise KeyError("shift file %r is missing variable %r" % (shiftpath, k))
        out[k] = np.atleast_2d(np.asarray(d[k], dtype=np.float64))
    tf = None
    undecodable = False
    if "tforms_optotune_full" in d:
        try:
            tf = np.asarray(d["tforms_optotune_full"], dtype=np.float64)
        except Exception:
            tf = None
            undecodable = True
    out["tforms_optotune_full"] = tf
    out["tforms_optotune_undecodable"] = undecodable
    return out


def _normalize_tforms(tf, nz):
    """Coerce transforms to a length-nz list of (3,3) MATLAB-convention T
    matrices ([x y 1] * T).  Accepts (nz,3,3), (3,3,nz), (3,3) or None.
    None / a single matrix is replicated to all planes (MakeSBXall.m L61-63
    builds a SINGLE identity affine2d in that case — which in MATLAB would
    actually error when indexed with j>1; see PORTING NOTES #5)."""
    if tf is None:
        return [None] * nz
    a = np.asarray(tf, dtype=np.float64)
    if a.shape == (3, 3):
        return [a] * nz

    def _looks_like_t_list(arr):  # arr: (n,3,3), third COLUMN of each == [0,0,1]
        return all(np.allclose(arr[i][:, 2], [0.0, 0.0, 1.0]) for i in range(arr.shape[0]))

    if a.ndim == 3 and a.shape[1:] == (3, 3) and _looks_like_t_list(a):
        lst = [a[i] for i in range(a.shape[0])]
    elif a.ndim == 3 and a.shape[:2] == (3, 3):
        b = np.moveaxis(a, 2, 0)
        lst = [b[i] for i in range(b.shape[0])]
    elif a.ndim == 3 and a.shape[1:] == (3, 3):
        lst = [a[i] for i in range(a.shape[0])]
    else:
        raise ValueError("cannot interpret tforms_optotune_full with shape %r" % (a.shape,))
    if len(lst) == 1:
        lst = lst * nz
    if len(lst) < nz:
        raise ValueError("tforms_optotune_full has %d transforms for Nz=%d" % (len(lst), nz))
    return lst[:nz]


# =========================================================================
# zproj_reg  (registration/zproj_reg.m)
# =========================================================================

def zproj_reg(k, nt, pmt, otrange, pathz=None, proj_type="mean", mtype=".sbx",
              reg_suffix=None, align=False, scale=4, regtype="DFT", refchan=1,
              write_unreg=False, zproj_raw=None,
              dft_reg_fn=None, dft_rect_fn=None):
    """zproj_reg.m L1-L66 — 3-pass DFT time-series stabilization of a
    z-projection stack.

    Parameters mirror the MATLAB function.  `zproj_raw` must be supplied as
    a (Nc, Y, X, Nt) float array ((Y, X, Nt) is accepted and treated as
    Nc=1): the `pipe.zproj` fallback branch (zproj_reg.m L17-18) is dead in
    this pipeline (MakeSBXall always supplies zproj_raw) and zproj.m itself
    is dead code, so that branch is not ported.  `k`, `otrange`, `pathz`,
    `proj_type`, `mtype`, `reg_suffix`, `align`, `write_unreg` are kept for
    signature fidelity; they are only consumed by the dead branch.

    regtype='Affine' (zproj_reg.m L43-47) is intentionally REMOVED: the
    branch references an undefined variable `pathz` (bare name at L44, not
    p.pathz) so it errors in the original too, and its only would-be caller
    path is never taken (default 'DFT').  Passing it raises.

    Algorithm (all on the refchan channel, in the compute class -- float64
    by default, float32 in fast mode; PORTING NOTES #16):
      L26      raw_ref = imresize(zproj_ref, 1/scale)   (MATLAB bicubic+antialias)
      L28-34   per-frame contrast normalization: clamp to [median, prctile99] -> [0,1]
      L37-38   pass 1: DFT_reg against the mean of the first min(Nt,50) frames
      L39      pass 2: DFT_rect chained from frame round(Nt/2)  (1-based)
      L40-41   pass 3: DFT_reg against the temporal median
      L52-53   R = (R1+R2+R3)*scale, C likewise
      L56-64   apply [C(i), R(i)] to every channel/frame of the FULL-res stack

    Returns (zproj_reg, R, C, TF): the stabilized stack (Nc, Y, X, Nt) in
    the compute class, the row/col shift vectors (Nt,) -- always float64,
    they are shift bookkeeping -- and TF as an (Nt, 3, 3) stack
    of identity matrices (mirror of the repmat(affine2d(eye(3))) at L49; the
    subsequent imwarp with an exact identity is a no-op and is skipped).
    """
    if zproj_raw is None:
        raise NotImplementedError(
            "zproj_reg: the pipe.zproj branch (zproj_reg.m L17-18) is dead "
            "code in this pipeline and was not ported; pass zproj_raw.")
    if regtype == "Affine":
        raise NotImplementedError(
            "zproj_reg: the 'Affine' branch (zproj_reg.m L43-47) is dead and "
            "broken in the original (undefined variable `pathz` at L44); it "
            "was removed in the port. Use regtype='DFT'.")
    if regtype != "DFT":
        raise ValueError("regtype must be 'DFT' (got %r)" % (regtype,))

    _lap0 = _tick()
    zproj = as_float(zproj_raw)
    _lap0("as_float")
    if zproj.ndim == 3:
        zproj = zproj[np.newaxis]
    if zproj.ndim != 4:
        raise ValueError("zproj_raw must be (Nc, Y, X, Nt); got shape %r" % (zproj.shape,))
    nt = int(nt)
    if zproj.shape[3] < nt:
        raise ValueError("nt=%d exceeds zproj_raw time dimension %d" % (nt, zproj.shape[3]))
    if not (1 <= int(refchan) <= zproj.shape[0]):
        raise IndexError("refchan=%r out of range for Nc=%d (refchan is 1-based)"
                         % (refchan, zproj.shape[0]))

    if dft_reg_fn is None:
        dft_reg_fn = _resolve_sibling(["dft_reg"], fallback=_fallback_dft_reg,
                                      what="DFT_reg")
    if dft_rect_fn is None:
        dft_rect_fn = _resolve_sibling(["dft_rect"], fallback=_fallback_dft_rect,
                                       what="DFT_rect")

    _lap = _tick()
    zproj_ref = zproj[int(refchan) - 1]                       # (Y, X, Nt)
    raw_ref = _imresize_over_t(zproj_ref, 1.0 / float(scale))  # (y, x, Nt)
    _lap("imresize 1/scale")

    h, w = raw_ref.shape[0], raw_ref.shape[1]
    raw_adj = _fzeros((h, w, nt))
    for i in range(nt):
        sl = raw_ref[:, :, i]
        lo = float(np.median(sl))
        hi = _matlab_prctile(sl, 99.0)
        raw_adj[:, :, i] = _matlab_rescale(sl, lo, hi)
    _lap("contrast norm loop")

    target1 = np.mean(raw_adj[:, :, :min(nt, 50)], axis=2)
    r1, c1, reg1 = dft_reg_fn(raw_adj, target1, scale)
    _lap("pass1 DFT_reg")
    r2, c2, reg2 = dft_rect_fn(reg1, int(_matlab_round(nt / 2.0)), scale)  # 1-based start
    _lap("pass2 DFT_rect")
    target3 = np.median(reg2, axis=2)
    r3, c3, reg3 = dft_reg_fn(reg2, target3, scale)
    _lap("pass3 DFT_reg")

    rr = (np.ravel(r1) + np.ravel(r2) + np.ravel(r3)) * float(scale)
    cc = (np.ravel(c1) + np.ravel(c2) + np.ravel(c3)) * float(scale)
    # DO NOT gate this trajectory against its own running median. It was
    # tried ("correction 5", 2026-08-25) on the data-plausible theory that
    # its two 50-64 px single-frame deviations were estimator mislocks; the
    # output-level measurement falsified it: applying the raw estimates
    # leaves those frames at ~8 px residual (the estimator TRACKS the lurch),
    # while neighbour interpolation left them at 41-46 px (it cut the corner
    # of a real 1-2-frame, ~30 um tissue excursion). A trajectory outlier
    # here is real motion until proven otherwise.
    tf = np.tile(np.eye(3), (nt, 1, 1))

    out = np.zeros_like(zproj)
    # L56-64.  This is the LAST resampling the deliverable ever sees, applied
    # to the finished projection, so improved mode's phase-ramp shift
    # (cpstab/improved.py correction 2) matters more here than anywhere else:
    # in replicate mode every output frame carries one bilinear low-pass on
    # top of the per-plane one already baked into the projection.
    # The improved branch goes through the SQRT-DOMAIN ramp (fshift2_vst, not
    # the raw fshift2): a raw ramp of this projection rings around every bright
    # structure and drives 0.145% of its pixels negative, which the uint16 cast
    # at RegistrationMasterPipeline.m L44 turns into black speckle in the
    # middle of the image -- the whole argument, with the numbers and the three
    # rejected alternatives, is in cpstab/fourier_shift.py.  The input here is a
    # mean of nonnegative planes, which is what fshift2_vst requires.
    _shift2 = _fshift2_vst if _improved.use_fourier_shift() else None

    # Nc*Nt independent 2-D shifts, one per (channel, frame), each a pure
    # function of one frame -- so the only thing the schedule can change is
    # how long it takes.  Two things are changed and neither touches a value:
    #   * a block of frames is transposed to (T, Y, X) first.  zproj is
    #     (Nc, Y, X, Nt) and zproj[c, :, :, i] therefore steps Nt doubles
    #     (12 kB at Nt=1500) between neighbouring columns; contiguous frames
    #     cost one sequential pass instead.
    #   * blocks run on a thread pool.  scipy's shift and numpy's ufuncs both
    #     release the GIL, so these are real cores (measured ~11x at 14
    #     threads on the scipy call alone).
    def _block(t0, t1):
        for c in range(zproj.shape[0]):
            src = np.ascontiguousarray(np.moveaxis(zproj[c, :, :, t0:t1], 2, 0))
            dst = np.empty_like(src)
            for k in range(t1 - t0):
                i = t0 + k
                if _shift2 is None:
                    dst[k] = _imtranslate_float(src[k], cc[i], rr[i])
                else:
                    dst[k] = _shift2(src[k], rr[i], cc[i])
                # zproj_reg.m L60 imwarp with TF(i) == identity: exact no-op.
            out[c, :, :, t0:t1] = np.moveaxis(dst, 0, 2)

    _run_blocks(_block, nt, _tail_block_len(nt))
    _lap("final shift apply")
    return out, rr, cc, tf


# =========================================================================
# make_sbxall  (registration/MakeSBXall.m)
# =========================================================================

def _resolve_proj_range(proj_range, nz):
    """Return 0-based plane indices for the projection window.

    'quarter' (MATLAB default, MakeSBXall.m L13): round(0.25*Nz):round(0.75*Nz),
    MATLAB round (half away from zero), 1-based INCLUSIVE both ends.
    'full': 1:Nz.
    Explicit sequences are interpreted as MATLAB-style 1-BASED inclusive
    indices (mirror of the proj_range parameter); 0 in the sequence is
    rejected to catch accidental 0-based input.
    """
    if isinstance(proj_range, str):
        key = proj_range.lower()
        if key == "quarter":
            lo = int(_matlab_round(0.25 * nz))
            hi = int(_matlab_round(0.75 * nz))
            idx1 = np.arange(lo, hi + 1)
        elif key == "full":
            idx1 = np.arange(1, nz + 1)
        else:
            raise ValueError("proj_range must be 'quarter', 'full', or a "
                             "sequence of 1-based plane indices; got %r" % (proj_range,))
    else:
        arr = np.asarray(proj_range, dtype=np.float64).ravel()
        if arr.size and not np.all(arr == np.floor(arr)):
            # MATLAB errors on a non-integer subscript; a bare int64 cast
            # would silently truncate ([1.5, 2.5] -> [1, 2]) instead.
            raise ValueError("proj_range must contain integer plane indices "
                             "(MATLAB errors on non-integer subscripts); "
                             "got %r" % (proj_range,))
        idx1 = arr.astype(np.int64)
    if idx1.size == 0:
        raise ValueError("proj_range is empty")
    if np.any(idx1 < 1) or np.any(idx1 > nz):
        raise IndexError(
            "proj_range indices must be 1-based within [1, Nz=%d]; got %r. "
            "(A 0 here usually means 0-based indices were passed.)" % (nz, idx1))
    return idx1 - 1


def _project(sub, proj_type, omitnan):
    """Z-projection of (Nc, Y, X, Nplanes) -> (Nc, Y, X).

    proj_type='mean', omitnan=False is MakeSBXall.m L120 verbatim
    (plain mean; translation/circshift fill zeros ARE averaged in).
    omitnan=True reproduces the dead zproj.m semantics (L59/L61-82):
    zeros -> NaN, NaN-omitting reduction, NaN -> 0.
    """
    if omitnan:
        sub = np.array(sub, dtype=work_dtype_for(np.asarray(sub).dtype),
                       copy=True)
        sub[sub == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if proj_type == "mean":
                out = np.nanmean(sub, axis=3)
            elif proj_type == "max":
                out = np.nanmax(sub, axis=3)
            elif proj_type == "median":
                out = np.nanmedian(sub, axis=3)
            else:
                raise ValueError("invalid projection type %r" % (proj_type,))
        out[np.isnan(out)] = 0.0
        return out
    if proj_type == "mean":
        return np.mean(sub, axis=3)
    if proj_type == "max":
        return np.max(sub, axis=3)
    if proj_type == "median":
        return np.median(sub, axis=3)
    raise ValueError("invalid projection type %r" % (proj_type,))


def _normalize_movie(mov, nc, nz):
    """Coerce a sibling imread result to (Nc, Y, X, Nz) — the MATLAB
    sbxRead contract is (nchan, rows, cols, N) for pmt=-1 with 2 channels,
    else squeezed (rows, cols, N).  MakeSBXall.m L75's reshape is a no-op
    for edges=[0 0 0 0] (see PORTING NOTES #4), so only axis bookkeeping
    happens here."""
    mov = np.asarray(mov)
    if mov.ndim == 2:
        mov = mov[np.newaxis, :, :, np.newaxis]
    elif mov.ndim == 3:
        if nc == 2 and mov.shape[0] == 2 and nz == 1:
            mov = mov[:, :, :, np.newaxis]
        else:
            mov = mov[np.newaxis]
    if mov.ndim != 4 or mov.shape[0] != nc or mov.shape[3] != nz:
        raise ValueError("imread returned shape %r; expected (Nc=%d, Y, X, Nz=%d)"
                         % (mov.shape, nc, nz))
    return mov


def _process_volume(raw4, lineshift, edges, tforms, optotune_on):
    """MakeSBXall.m L77-L91 (identical in both passes): line-shift the odd
    scan rows, crop by edges, apply the per-plane optotune warp.

    raw4: (Nc, Ys, Xs, Nz) uint16-valued.  Returns (Nc, Y, X, Nz) in the
    compute class (float64 by default, float32 in fast mode) holding
    uint16-quantized values (the MATLAB arrays stay uint16 here).
    """
    v = np.asarray(raw4)
    if lineshift != 0:
        v = v.copy()
        # MATLAB L77: raw_vol(:,1:2:end,:,:) = circshift(..., lineshift, 3)
        # rows 1,3,5,... (1-based odd) == 0-based ::2 ; dim 3 == axis 2 (X);
        # pure wrap-around, no zero clearing.
        v[:, ::2, :, :] = np.roll(v[:, ::2, :, :], int(lineshift), axis=2)
    e1, e2, e3, e4 = (int(e) for e in edges)
    ys, xs = v.shape[1], v.shape[2]
    # MATLAB L80: raw_vol(:, e3+1:end-e4, e1+1:end-e2, :)
    v = v[:, e3:ys - e4, e1:xs - e2, :]
    v = as_float(v)
    if optotune_on:
        nc, nz = v.shape[0], v.shape[3]
        # opttype 'none' (the production setting) makes EVERY tform an exact
        # identity, and the loop below then copies v into out plane by plane
        # through two strided views -- 172 MB of pure memcpy per volume, 13%
        # of the apply stage's profile, for a result equal to v element for
        # element.  Skipping it is bit-exact by construction; the copy is
        # still made the moment any plane has a real warp.  (v is already a
        # fresh array here: as_float above materializes the uint16 -> float
        # cast, so no caller aliases it.)
        if (v.flags.c_contiguous
                and all(_is_identity_tform(tforms[j]) for j in range(nz))):
            return v
        out = np.empty_like(v)
        for j in range(nz):
            t = tforms[j]
            if _is_identity_tform(t):
                out[:, :, :, j] = v[:, :, :, j]
            else:
                for c in range(nc):
                    out[c, :, :, j] = _imwarp_affine2d_u16(v[c, :, :, j], t)
        v = out
    return v


def _z_shift_for(zs_value):
    """The Z shift `_apply_shifts_volume` should be given, per mode
    (MakeSBXall.m L111).

    replicate: MATLAB's round(ZS_total(i)) -- a whole plane, half away from
      zero.
    improved: the raw float (cpstab/improved.py correction 4), which the
      sub-plane branch of _apply_shifts_volume interpolates.
    """
    if _improved.use_subplane_z():
        return float(zs_value)
    return int(_matlab_round(zs_value))


def _apply_z_shift_subplane(reg, z):
    """Fractional-plane Z translation (cpstab/improved.py correction 4).

    Replaces MakeSBXall.m L111-L118 (round -> circshift -> clear the wrapped
    band) in improved mode.  ZS comes from dftregistration3D at usfac=2, i.e.
    it is measured to HALF a plane; rounding it to a whole plane throws away
    up to 0.5 plane of the axial registration that was just computed, which on
    a 41-plane stack with ~1 plane of respiratory/heartbeat sway is the
    dominant residual axial error.

    out[..., j] = in[..., j - z], linearly interpolated between the two
    planes bracketing the fractional position:

        zi = floor(z),  f = z - zi  in [0, 1)
        out[j] = (1 - f) * in[j - zi]  +  f * in[j - zi - 1]

    (out(j) should sample in(j - zi - f); that lies a fraction f of the way
    from in(j - zi) toward in(j - zi - 1), since increasing f moves the sample
    point toward SMALLER input index.)

    Planes whose source index falls outside [0, Nz) contribute nothing -- the
    output is zero-filled there.  That is honest and symmetric, unlike the
    original's circshift-then-clear, which wraps real planes around the stack
    and then zeroes a band of the wrong width on one side (MATLAB clears
    -Z + 1 planes for Z < 0, one MORE than it wrapped; see PORTING NOTES #3).
    A whole-number z reduces to a pure zero-filled plane shift, so even then
    improved mode differs from replicate by dropping that off-by-one.

    NOTE the boundary plane is left PARTIALLY weighted rather than cleared:
    for z = 0.5 the first output plane is 0.5 * in[0], because its other
    contributor is off the end of the stack.  This differs on purpose from
    what fshift2 does at the XY border, where the contaminated band is zeroed
    outright -- there the contaminant is WRAPPED far-edge content, i.e. wrong
    data, while here it is a clean zero and the plane is merely dim.  It is
    also unreachable on the deliverable: the projection window is the middle
    half of the stack (MakeSBXall.m L13) and |ZS| is ~1 plane after median
    centring, so a dimmed end plane never enters the projection.
    """
    nz = reg.shape[3]
    zi = int(np.floor(z))
    f = float(z - zi)
    if not abs(z) < nz:
        # Every source index would fall outside the stack, so the honest
        # answer is an all-zero volume -- which is exactly why this raises
        # instead of returning it.  Unreachable on real data (ZS is
        # median-centred upstream); the replicate branch below fails loudly on
        # the same input and this branch must not be the quiet one.
        raise IndexError(
            "Z shift %r exceeds Nz=%d; every plane would be zero-filled "
            "(improved sub-plane Z path)" % (z, nz))
    out = np.zeros_like(reg)
    # term A: weight (1-f) from plane j-zi ; term B: weight f from j-zi-1
    for k, w in ((zi, 1.0 - f), (zi + 1, f)):
        if w == 0.0:
            continue
        lo = max(0, k)              # first output plane whose source exists
        hi = min(nz, nz + k)        # one past the last
        if hi > lo:
            out[:, :, :, lo:hi] += w * reg[:, :, :, lo - k:hi - k]
    return out


def _apply_shifts_volume(warp4, r_col, c_col, z_shift, plane_major=False):
    """MakeSBXall.m L93-L118 (identical in both passes): per-plane XY
    translation, then the whole-volume Z shift.

    warp4: (Nc, Y, X, Nz) in the compute class (uint16-valued), or
    (Nc, Nz, Y, X) with plane_major=True -- see _PLANE_MAJOR_NOTE below.
    z_shift: what _z_shift_for() returned -- a MATLAB-rounded int in replicate
    mode, the raw float in improved mode.  Returns reg_vol in warp4's class.

    Mode differences (cpstab/improved.py corrections 2 and 4):
      XY  replicate does MATLAB imtranslate([C,R]) with the uint16
          re-quantization the original's uint16 arrays imposed; improved does
          a Fourier phase-ramp shift in the SQRT domain
          (fourier_shift.fshift2_vst) and does NOT re-quantize -- dropping a
          bilinear low-pass AND a 0.5-count rounding, both applied once per
          plane, is the whole point of that correction.  The sqrt domain is
          not decoration: a raw ramp leaves 25.6% of a plane's pixels
          NEGATIVE (down to -373 counts) by ringing around single-pixel
          content, and the residue that survives the z-projection becomes
          black speckle in the uint16 deliverable.  Numbers, and the clip /
          clamp / spectral-taper alternatives that were measured and
          rejected, in cpstab/fourier_shift.py.  warp4 is uint16-valued, i.e.
          nonnegative, which is what that function requires.
      Z   replicate does round -> circshift -> asymmetric clear; improved
          interpolates between neighbouring planes with zero fill.

    _PLANE_MAJOR_NOTE.  plane_major=True says warp4 (and the result) is
    (Nc, Nz, Y, X) instead of (Nc, Y, X, Nz).  Nothing arithmetic changes:
    the same _imtranslate_u16 runs on the same contiguous 2-D plane, and the
    Z bookkeeping is the same roll-and-clear on whichever axis holds Nz.
    What it buys is the two transposes the (Nc, Y, X, Nz) form has to make
    to get contiguous planes and put them back -- 344 MB of shuffling per
    production volume, measured 0.249 -> 0.133 s/volume for the whole
    per-volume chain.  The caller is responsible for having built the stack
    in that layout and for un-transposing the projection window afterwards
    (see fast_run._apply_worker).  Only the replicate branch takes it;
    improved mode's sub-plane Z helper is written against axis 3, and there
    is no reason to fork it for a mode that is not the benchmark.
    """
    if plane_major:
        if _improved.use_subplane_z() or _improved.use_fourier_shift():
            raise ValueError("plane_major is a replicate-mode layout only")
        nc, nz = warp4.shape[0], warp4.shape[1]
        zaxis = 1
    else:
        nc, nz = warp4.shape[0], warp4.shape[3]
        zaxis = 3
    reg = np.zeros_like(warp4)
    if _improved.use_fourier_shift():
        for c in range(nc):
            for j in range(nz):
                reg[c, :, :, j] = _fshift2_vst(warp4[c, :, :, j],
                                               r_col[j], c_col[j])
    else:
        # warp4 is (Nc, Y, X, Nz) C-contiguous, so warp4[c, :, :, j] steps
        # Nz doubles between neighbouring columns: every interpolation reads
        # and writes with a 328-byte stride and the shift spends most of its
        # time missing cache.  Transposing one channel to (Nz, Y, X) once,
        # doing the 41 shifts on contiguous planes, and transposing back
        # moves the same bytes twice but sequentially -- measured 0.727 ->
        # 0.158 s/volume together with _bilinear_shift2, bitwise identical
        # (each shift is an independent, deterministic function of one
        # plane's values; only the memory layout it reads them from changed).
        for c in range(nc):
            if plane_major:
                src, dst = warp4[c], reg[c]          # already (Nz, Y, X)
            else:
                src = np.ascontiguousarray(np.moveaxis(warp4[c], 2, 0))
                dst = np.empty_like(src)
            for j in range(nz):
                dst[j] = _imtranslate_u16(src[j], c_col[j], r_col[j])
            if not plane_major:
                reg[c] = np.moveaxis(dst, 0, 2)
    if _improved.use_subplane_z():
        return _apply_z_shift_subplane(reg, float(z_shift))
    z = int(z_shift)
    if z != 0:
        if z > 0 and z > nz:
            # MATLAB would NOT error here: reg_vol(:,:,:,1:Z)=0 with Z>Nz
            # GROWS the array to Z planes and zeroes 1:Z (a superset of all
            # real planes), so the projection would be silently all-zero.
            # Pathological input (ZS is median-centred upstream, |Z|<Nz on
            # any real run); the port fails loudly instead.
            raise IndexError("Z shift %d exceeds Nz=%d (MATLAB would "
                             "silently grow reg_vol and zero every real "
                             "plane; the port raises instead)" % (z, nz))
        if z < 0 and nz + z - 1 < 0:
            # Here MATLAB DOES error: end+Z:end with Nz+Z < 1 is a
            # zero/negative subscript.
            raise IndexError("Z shift %d exceeds Nz=%d (MATLAB errors at "
                             "reg_vol(:,:,:,end+Z:end)=0 too)" % (z, nz))
        reg = np.roll(reg, z, axis=zaxis)
        # (slice(None),) * zaxis is "index the axes ahead of Nz, whichever
        # they are" -- the same two MATLAB statements, addressed by axis.
        lead = (slice(None),) * zaxis
        if z > 0:
            # MATLAB L114-115: reg_vol(:,:,:,1:Z) = 0  -> exactly Z planes
            reg[lead + (slice(None, z),)] = 0.0
        else:
            # MATLAB L116-117: reg_vol(:,:,:,end+Z:end) = 0
            # 1-based Nz+Z .. Nz  ->  0-based [Nz+Z-1 :], i.e. -Z+1 planes:
            # ONE MORE than actually wrapped. Faithful to the original.
            reg[lead + (slice(nz + z - 1, None),)] = 0.0
    return reg


def make_sbxall(path, shiftpath, edges=(0, 0, 0, 0), nt=None, pmt=None,
                lineshift=0, optotune="true", refchannel=2,
                proj_range="quarter",
                proj_type="mean", omitnan=False,
                write_registered=False, cache_warped=True,
                tforms_optotune_full=None, verbose=False,
                imread_fn=None, sbx_info_fn=None, spoof_info_fn=None,
                reg_writer_cls=None, dft_reg_fn=None, dft_rect_fn=None):
    """MakeSBXall.m L1-L204 — apply the DFT shifts, build the stabilized
    z-projection time series, optionally write the registered '.sbxall'.

    Mirrored parameters (same defaults as the MATLAB inputParser, L4-L13):
      path        .sbx file path (with its .mat sidecar readable by sbx_info)
      shiftpath   the '.dftshifts' file from DFT_warp_3D_2 (MAT v7 or .npz)
      edges       [e1,e2,e3,e4]; rows cropped by e3(top)/e4(bottom), cols by
                  e1(left)/e2(right)   (default [0,0,0,0])
      nt          number of volumes (default: all registered, = CS.shape[1]).
                  NB (faithful crash parity): any 1 < nt < CS.shape[1] dies
                  at the refinement fold (MakeSBXall.m L130 / the broadcast
                  below) in BOTH implementations — MATLAB implicit expansion
                  rejects (Nz, Nreg) + (Nz, nt), numpy rejects the same
                  broadcast; nt=1 broadcasts the scalar refinement over all
                  registered columns in both (equivalent). Only nt=None /
                  nt=CS.shape[1] (and nt=1) complete.
      pmt         None -> all channels (-1 if the file has 2); -1/1/2 as in
                  sbxRead (1-based channel select)
      lineshift   circshift of odd scan rows along X (default 0)
      optotune    MATLAB-truthy flag (default 'true'; note MATLAB treats ANY
                  nonempty string, including 'false', as true — mirrored)
      refchannel  1-based channel used by the zproj_reg refinement (default 2,
                  as in the MATLAB source; the master pipeline passes 1)
      proj_range  'quarter' (default; MATLAB round(0.25*Nz):round(0.75*Nz),
                  1-based inclusive), 'full' (1:Nz), or an explicit sequence
                  of 1-BASED plane indices

    Port extensions (keyword-only in spirit; keep them named):
      proj_type / omitnan   projection statistic for the PASS1 accumulate
                  (MakeSBXall.m L120 hardcodes plain 'mean'; 'max'/'median'
                  and the omitnan semantics of the dead zproj.m are offered
                  as options — defaults reproduce the original exactly)
      write_registered      False (default): skip the .sbxall write (PASS2,
                  L139-202) entirely. DELIBERATE default-behavior DIVERGENCE
                  (review F2): MakeSBXall.m runs PASS2 UNCONDITIONALLY; the
                  port makes it opt-in because the file has no consumer in
                  the repo (write-only disk bloat, CODEMAP §5-G). For full
                  MATLAB parity pass write_registered=True — the emitted
                  byte stream is then verified identical to the MATLAB
                  contract (tests/scratch_apply_project.py, section B).
                  True: after the zproj_reg refinement adjusts RS/CS
                  (L127-131), the shifts are applied a second time and the
                  volume stream is written via RegWriter, exactly like the
                  original ordering. The zproj_mean return value is
                  unaffected either way.
      cache_warped          True (default): single-read fusion — cache the
                  deterministic lineshift+crop+warp intermediate from PASS1
                  (as uint16, ~2 bytes/voxel of the cropped stack) and only
                  replay the shift application for the write; False: re-read
                  the raw file like MATLAB PASS2. Both are numerically
                  identical.
      tforms_optotune_full  override for the per-plane affine T matrices
                  ((Nz,3,3), (3,3,Nz) or (3,3)); default: taken from the
                  shift file. Absent from the shift file -> identity
                  (mirrors MakeSBXall.m L61-63). PRESENT but undecodable
                  (MATLAB affine2d objects, opaque to scipy) -> ValueError
                  when the optotune warp branch is live (review F1: silently
                  substituting identity for real warps diverges pixel-wise);
                  pass this argument or a falsy `optotune` to proceed.
      *_fn / *_cls          dependency injection for the sibling IO / DFT
                  modules (default: resolved from this package).

    Returns zproj_mean: the zproj_reg-stabilized projection time series,
    (Nc, Y, X, Nt) in the compute class — float64, mirroring the single
    MATLAB return value, or float32 under cfg.compute_dtype='float32'.

    Pipeline (line refs into MakeSBXall.m):
      L26-35   load shifts; totals = local + chunk, median-centred
               (RS/CS: per-timepoint median across planes when Nz>1;
                ZS: scalar median — MATLAB median() default-dim semantics)
      L72-123  PASS1 per volume: read raw -> lineshift -> crop -> warp ->
               per-plane imtranslate([C,R]) (uint16 quantized) ->
               Z circshift + clear -> projection over proj_range
      L125     zproj_reg refinement (3-pass DFT on the projection series)
      L127-131 fold the refinement into RS_total / CS_total
      L134-202 (write_registered) spoof info, RegWriter '.sbxall', re-apply
               with refined shifts, pad by edges, uint16, stream out
    """
    # --- info / geometry (L11-L23) ---------------------------------------
    if sbx_info_fn is None:
        sbx_info_fn = _resolve_sibling(["sbx_info", "sbxinfo"], fallback=None,
                                       what="pipe.io.sbxInfo")
    info = sbx_info_fn(path)
    otwave = np.asarray(_info_get(info, "otwave"))
    nz = int(otwave.size)                       # L12: size(info.otwave, 2)
    sz = np.asarray(_info_get(info, "sz")).ravel()
    e1, e2, e3, e4 = (int(e) for e in edges)
    ny_rows = int(sz[0]) - e3 - e4              # L22 "Nx" (rows / Y)
    nx_cols = int(sz[1]) - e1 - e2              # L23 "Ny" (cols / X)

    # --- shifts (L26-L35) -------------------------------------------------
    shifts = _load_shifts(shiftpath)
    rs, cs, zs = shifts["RS"], shifts["CS"], shifts["ZS"]
    # _median_centering is MATLAB's default-dim median() in replicate mode and
    # a global scalar in improved mode (cpstab/improved.py correction 1). For
    # ZS -- a (1, Nt) row vector -- the two are the same number.
    zs_total = zs + shifts["ZS_chunk"]
    zs_total = zs_total - _median_centering(zs_total)   # row vector -> scalar med
    zs_total = np.ravel(zs_total)
    rs_total = rs + shifts["RS_chunk"]
    rs_total = rs_total - _median_centering(rs_total)   # per-column med (Nz>1)
    cs_total = cs + shifts["CS_chunk"]
    cs_total = cs_total - _median_centering(cs_total)
    rs_total = np.atleast_2d(np.asarray(rs_total, dtype=np.float64))
    cs_total = np.atleast_2d(np.asarray(cs_total, dtype=np.float64))
    if rs_total.shape[0] not in (1, nz):
        raise ValueError("RS has %d planes but Nz=%d" % (rs_total.shape[0], nz))
    if rs_total.shape[0] == 1 and nz > 1:
        rs_total = np.repeat(rs_total, nz, axis=0)
        cs_total = np.repeat(cs_total, nz, axis=0)

    # --- Nt (L37-L42) -----------------------------------------------------
    n_registered = int(cs.shape[1])
    if nt is None:
        nt = n_registered
    elif int(nt) > n_registered:
        # MATLAB prints 'Trying to save more volumes...' and silently
        # returns []; the port raises instead (see PORTING NOTES #10).
        raise ValueError("nt=%d exceeds the %d registered volumes" % (nt, n_registered))
    nt = int(nt)

    # --- pmt / Nc (L44-L58) ----------------------------------------------
    if pmt is None:
        nc = int(_info_get(info, "nchan"))
        pmt = -1 if nc == 2 else 1
    elif pmt == -1:
        nc = 2
    elif pmt in (1, 2):
        nc = 1
    else:
        raise ValueError("please enter valid pmt (got %r)" % (pmt,))

    # --- optotune transforms (L61-L63) -----------------------------------
    optotune_on = _matlab_truthy(optotune)
    if tforms_optotune_full is None:
        tforms_optotune_full = shifts["tforms_optotune_full"]
        if tforms_optotune_full is None and shifts.get("tforms_optotune_undecodable"):
            # Review F1: the shift file DOES carry transforms (MATLAB
            # affine2d objects, opaque to scipy.io.loadmat) which MakeSBXall.m
            # L86/L153 would apply. Substituting identity silently would
            # produce pixel-level different output, so this is fatal whenever
            # the warp branch is live (an explicit tforms_optotune_full=
            # argument bypasses this, as does a falsy optotune).
            if optotune_on:
                raise ValueError(
                    "apply_project: shift file %r contains "
                    "'tforms_optotune_full' but it cannot be decoded to a "
                    "numeric array (MATLAB affine2d objects in a MAT-file "
                    "are opaque to scipy.io.loadmat), and optotune=%r is "
                    "MATLAB-truthy, so MakeSBXall.m L86 would apply these "
                    "warps — refusing to substitute identity transforms "
                    "silently. Fix: pass tforms_optotune_full=... explicitly "
                    "(numeric (Nz,3,3), (3,3,Nz) or (3,3) MATLAB-convention "
                    "T matrices), or re-save the shift file with a numeric T "
                    "stack (the Python orchestrator's .npz already is), or "
                    "pass a falsy optotune (e.g. optotune=0) if the "
                    "transforms are known to be identity."
                    % (shiftpath, optotune))
            warnings.warn(
                "apply_project: 'tforms_optotune_full' in %r is not "
                "decodable (MATLAB affine2d objects are opaque to scipy); "
                "harmless here because optotune=%r is falsy and the warp "
                "branch is skipped." % (shiftpath, optotune))
    tforms = _normalize_tforms(tforms_optotune_full, nz)

    # --- projection window (L13) -----------------------------------------
    idx0 = _resolve_proj_range(proj_range, nz)

    if imread_fn is None:
        imread_fn = _resolve_sibling(["imread", "sbx_read", "sbxread"],
                                     fallback=None, what="pipe.imread")

    # --- PASS 1 (L72-L123) ------------------------------------------------
    zproj_raw = _fzeros((nc, ny_rows, nx_cols, nt))
    cache = [] if (write_registered and cache_warped) else None
    for i in range(nt):
        k = nz * i + 1                                        # MATLAB Nz*(i-1)+1
        raw = _normalize_movie(imread_fn(path, k, nz, pmt, None), nc, nz)
        warp4 = _process_volume(raw, lineshift, edges, tforms, optotune_on)
        if cache is not None:
            cache.append(warp4.astype(np.uint16))             # values are integers
        z = _z_shift_for(zs_total[i])                         # L111
        reg = _apply_shifts_volume(warp4, rs_total[:, i], cs_total[:, i], z)
        zproj_raw[:, :, :, i] = _project(reg[:, :, :, idx0], proj_type, omitnan)
        if verbose and (i + 1) % 10 == 0:
            print("projected %d/%d volumes" % (i + 1, nt))

    # --- zproj_reg refinement (L125-L131) ---------------------------------
    zproj_mean, r_zproj, c_zproj, _tf = zproj_reg(
        1, nt, pmt, idx0 + 1, refchan=refchannel, zproj_raw=zproj_raw,
        dft_reg_fn=dft_reg_fn, dft_rect_fn=dft_rect_fn)
    # L127-131: repmat over planes + transpose == broadcast row over Nz
    rs_total = rs_total + np.asarray(r_zproj, dtype=np.float64)[np.newaxis, :nt]
    cs_total = cs_total + np.asarray(c_zproj, dtype=np.float64)[np.newaxis, :nt]

    # --- PASS 2: .sbxall write (L134-L202), optional ----------------------
    if write_registered:
        if spoof_info_fn is None:
            spoof_info_fn = _resolve_sibling(
                ["spoof_sbx_info_3d", "spoof_sbxinfo_3d", "spoof_sbxinfo3d"],
                fallback=_fallback_spoof_sbx_info_3d, what="SpoofSBXinfo3D")
        if reg_writer_cls is None:
            reg_writer_cls = _resolve_sibling(
                ["RegWriter", "reg_writer"],
                fallback=_FallbackRegWriter, what="pipe.io.RegWriter")
        info_out = spoof_info_fn(ny_rows, nx_cols, nz, nt, nc)   # L134
        rw = reg_writer_cls(path, info_out, ".sbxall", True)     # L137
        try:
            for i in range(nt):
                if cache is not None:
                    warp4 = as_float(cache[i])   # same class PASS1 used
                else:
                    k = nz * i + 1
                    raw = _normalize_movie(imread_fn(path, k, nz, pmt, None), nc, nz)
                    warp4 = _process_volume(raw, lineshift, edges, tforms, optotune_on)
                z = _z_shift_for(zs_total[i])                    # L177
                reg = _apply_shifts_volume(warp4, rs_total[:, i], cs_total[:, i], z)
                # L188-190: un-crop (zero border of the edge sizes), uint16
                masked = _fzeros((nc, reg.shape[1] + e3 + e4,
                                  reg.shape[2] + e1 + e2, reg.shape[3]))
                masked[:, e3:e3 + reg.shape[1], e1:e1 + reg.shape[2], :] = reg
                masked_u16 = np.clip(np.floor(masked + 0.5), 0, _U16MAX).astype(np.uint16)
                rw.write(masked_u16)                             # L193
                if verbose and (i + 1) % 10 == 0:
                    print("written %d volumes to .sbxall" % (i + 1))
        finally:
            close = getattr(rw, "close", None)
            if close is not None:
                close()   # MATLAB relies on the RegWriter destructor; explicit here.

    return zproj_mean


# =========================================================================
# PORTING NOTES
# =========================================================================
# 1.  Shift-array shapes (verified against DFT_warp_3D_2.m L83-129): RS/CS
#     are (Nz, Nt), ZS is (1, Nt); the *_chunk arrays are imresize'd
#     ('nearest') to the SAME shapes before saving. Hence MATLAB median()
#     default-dim semantics differ: median(RS_total) is a per-column (per-
#     timepoint) median across planes when Nz>1, while median(ZS_total) on a
#     row vector is a scalar. _median_matlab_default reproduces both,
#     including the Nz==1 corner where RS collapses to vector semantics.
# 2.  uint16 quantization chain (the least obvious ground-truth semantics):
#     in both passes the slices fed to imtranslate (L101/L168) — and to
#     imwarp in the optotune branch — are uint16, so MATLAB rounds each
#     bilinear result half-away-from-zero and saturates to [0, 65535]
#     BEFORE it is accumulated into the double reg_vol. The port keeps
#     float64 arrays but applies _quantize_u16 at exactly those points.
#     In zproj_reg (L59) the stack is double, so NO quantization there.
#     Caveat: values landing exactly on .5 after bilinear interpolation can
#     round differently than MATLAB at the ~1e-15 level of fft/interp noise;
#     unavoidable for any reimplementation.  Translation/warp boundary
#     handling uses scipy mode='grid-constant' (NOT 'constant'): MATLAB
#     imtranslate/imwarp pad with FillValues=0 and interpolate across the
#     boundary, so the fill blends into the <=1 px edge ring on fractional
#     shifts; plain 'constant' would zero that ring instead. The <=1 px
#     edge-ring blend model is consistent across all three ported modules
#     and with R2018a semantics (imtranslate always pads; imwarp predates
#     the R2019b SmoothEdges=false default; the master script pins R2018a
#     via its javaaddpath), but a live-MATLAB spot check of a fractional
#     shift is still pending (adversarial review F4).
# 3.  Z circshift clearing is asymmetric in the original (L113-118): Z>0
#     clears exactly Z planes (1:Z), Z<0 clears end+Z:end which is -Z+1
#     planes — one MORE than actually wrapped. Mirrored verbatim. Out-of-
#     range Z (unreachable: ZS is median-centred) raises here; MATLAB is
#     asymmetric about it — Z>Nz GROWS the array and silently zeroes every
#     real plane, Z<=-Nz errors on the zero/negative subscript.
# 4.  MakeSBXall.m L75 reshapes the raw read to the EDGE-CROPPED dims
#     (Nc,Nx,Ny,Nz) and L80 crops again; that reshape only runs without
#     error when edges=[0,0,0,0] (element-count mismatch otherwise), so the
#     MATLAB function de facto requires zero edges. The port reshapes to the
#     full frame and crops once — identical for edges=0, and implements the
#     evident intent (rather than the crash) for edges!=0. Same for the
#     RegWriter size check against the spoofed (cropped-size) info.
# 5.  L61-63 fallback when the shift file lacks tforms_optotune_full builds
#     a SINGLE identity affine2d; MATLAB would then error at
#     tforms_optotune_full(j) for j>1 (indexing a scalar object). The port
#     replicates the identity across all planes instead (identical for
#     Nz==1; the crash is not reproduced). Exact-identity transforms skip
#     imwarp entirely — for uint16 inputs MATLAB's identity imwarp is also
#     an exact copy after rounding, so this is equivalence, not divergence.
#     Note MATLAB truthiness of the `optotune` flag is mirrored: any
#     nonempty string (including 'false') enables the warp branch.
# 6.  MATLAB affine2d objects inside a v7 .dftshifts MAT-file are opaque to
#     scipy.io.loadmat. Review F1: when 'tforms_optotune_full' is PRESENT
#     but undecodable AND the optotune flag is MATLAB-truthy, the port now
#     RAISES instead of silently substituting identity (MakeSBXall.m L86
#     would apply the real warps; identity would diverge pixel-wise).
#     Escape hatches: pass tforms_optotune_full explicitly (numeric
#     MATLAB-convention T: [x y 1] * T), use an .npz shift file with the
#     numeric (Nz,3,3) stack (the Python orchestrator writes exactly that),
#     or pass a falsy optotune. A variable that is ABSENT still maps to
#     identity (mirrors MakeSBXall.m L61-63; correct for the piezo/'none'
#     path). The general _imwarp_affine2d_u16 path implements imwarp
#     (linear, fill 0, OutputView=input grid, 1-based intrinsic coords) but
#     has NOT been validated against MATLAB on non-identity transforms —
#     flagged as DIVERGENCE-risk, unexercised on the production path.
# 7.  rescale (zproj_reg.m L32): clamp to [InputMin, InputMax] then map to
#     [0,1]. The degenerate U<=L case (constant resized frame) returns
#     zeros here; MATLAB's exact behavior in that corner (0/0) was not
#     verified — it cannot occur on real data where prctile99 > median.
#     prctile is implemented with MATLAB's midpoint positions
#     100*(i-0.5)/n + linear interpolation (numpy's default percentile
#     positions DIFFER — do not substitute).
# 8.  imresize (zproj_reg.m L26): hand-written MATLAB algorithm — cubic
#     kernel a=-0.5, antialiasing (kernel widened by 1/scale) since
#     scale=1/4<1, output size ceil(scale*size), u = x/scale+0.5*(1-1/scale),
#     P = ceil(kw)+2 taps, symmetric boundary via the [1:m, m:-1:1] index
#     mirror, per-row weight normalization; rows resized before cols (the
#     MATLAB order for equal scales; separable passes commute up to ~1e-16
#     anyway).
# 9.  dftregistrationAlex fallback: max-peak located in COLUMN-major first-
#     occurrence order (MATLAB find(...,1,'first')), FTpad/dftups index math
#     kept in 1-based form and converted at the slice boundary, complex128
#     throughout. usfac==1/0 regimes are implemented but unused here
#     (upscale=scale=4 on this path). MATLAB's usfac==1 branch with tied
#     maxima returns vectors (find without 'first'); the port returns the
#     first column-major peak — divergence only on exact ties in a branch
#     this pipeline never takes.
# 10. Error-path divergences (deliberate, non-numeric): MATLAB prints and
#     silently returns [] for nt too large / invalid pmt; the port raises
#     ValueError. The dead pipe.zproj branch of zproj_reg (only reachable
#     when zproj_raw is empty; requires the dead zproj.m) raises
#     NotImplementedError. The broken 'Affine' branch (undefined `pathz`,
#     zproj_reg.m L44) is removed and raises NotImplementedError.
# 11. Cross-module call assumptions (parallel port): sibling functions are
#     expected under snake_case mirrors — sbx_info(path); imread(path, k,
#     N, pmt, optolevel) with k 1-BASED and the sbxRead return layout
#     ((nchan, rows, cols, N) for pmt=-1 & 2 channels, squeezed otherwise,
#     values already 65535-inverted uint16); dft_reg(stack, target,
#     upscale) -> (R, C, reg); dft_rect(vol, start, upscale) with start
#     1-BASED -> (R, C, reg); spoof_sbx_info_3d(y, x, z, t, nchan);
#     RegWriter(path, info, extension, force) with .write/.close. If a
#     sibling deviates (e.g. 0-based k or start), inject the correct
#     adapter via the *_fn/*_cls keywords. Internal fallbacks exist for the
#     DFT engine, SpoofSBXinfo3D and RegWriter, but NOT for sbx_info/imread
#     (the byte-contract owner is the IO module). Resolution reality on the
#     current package (review F3): dft_reg/dft_rect bind to shifts2d,
#     imread/sbx_info to io_rw — the internal fallbacks are dead code in
#     practice, verified bit-identical to the siblings.
# 12. proj_range: 'quarter' default mirrors MakeSBXall's own
#     round(0.25*Nz):round(0.75*Nz) (MATLAB round = half away from zero;
#     e.g. Nz=30 -> planes 8..23 1-based). Explicit sequences are 1-BASED
#     (MATLAB-style) and validated; Nz==1 with 'quarter' raises just as the
#     MATLAB index 0:1 would.
# 13. zproj_raw is (Nc, Y, X, Nt) in the compute class (float64 by default,
#     see note 16); MakeSBXall.m L120 accumulates a
#     PLAIN mean including translation/circshift zero fills — proj_type=
#     'mean', omitnan=False reproduces that exactly and is the default; the
#     omitnan=True variants implement the zeros->NaN->reduce->0 semantics of
#     the dead zproj.m for the optional statistics.
# 14. MATLAB never closes the RegWriter explicitly (destructor closes on
#     function exit); the port closes it deterministically in a finally
#     block. The .sbxall byte contract (permute [1 3 2 4], 65535-value,
#     column-major uint16 stream) is mirrored in the fallback writer.
# 16. FLOAT32 FAST MODE (port extension; cpstab/precision.py, cfg.
#     compute_dtype). MATLAB is double everywhere, so float64 stays the
#     default and the replicate path is bit-for-bit unchanged (every
#     `_fzeros(shape)` / `as_float(x)` below reduces to the
#     `np.zeros(shape)` / `np.asarray(x, np.float64)` it replaced). In fast
#     mode the whole image domain of this module drops to single precision:
#     the PASS1 zproj_raw accumulator, _process_volume's promoted volume,
#     _apply_shifts_volume's reg (via zeros_like), the bilinear
#     imtranslate/imwarp, the projection reduction, and zproj_reg's resized
#     + contrast-normalized stack. What deliberately does NOT follow:
#       * _load_shifts / _median_matlab_default / the RS_total, CS_total,
#         ZS_total bookkeeping and _matlab_round of the Z shift — float64 in
#         both modes; the shift file is one precision no matter who wrote it;
#       * _matlab_prctile and the per-frame median of zproj_reg.m L29-31 —
#         they run on the (small) resized frame in double, because they set
#         the contrast window every later step depends on;
#       * the FallbackRegWriter's uint16 quantization of the .sbxall stream.
#     Two failure modes are worth naming, because they are DISCRETE rather
#     than an ulp: `z = round(ZS_total[i])` picks a whole-plane circshift, and
#     every argmax in the DFT engine picks a whole 1/usfac grid step. A
#     float32 perturbation that lands on one of those ties changes the output
#     by a plane or by a 1/usfac shift, not by a count. Neither was observed
#     on the validation subset (see cfg.compute_dtype's docstring for the
#     measured numbers, and tests/test_f32.py which re-measures them), but
#     they are the reason fast mode is opt-in and must be validated per
#     dataset instead of assumed.
# 17. IMPROVED MODE (port extension; cpstab/improved.py, cfg.mode). Three of
#     the four corrections land in this module, each behind its own
#     `_improved.use_*()` guard so that in the DEFAULT 'replicate' mode every
#     branch below takes the literal code that was there before the feature
#     existed — that is what the two iron-law regressions assert.
#       * correction 1 (`_median_centering`, the L29-35 centring): MATLAB's
#         default-dim median is per-timepoint, which algebraically cancels
#         every plane-constant shift term. RS2/CS2 (tiled over planes at
#         DFT_warp_3D_2.m L83-84) and RS_chunk/CS_chunk (a 'nearest' stretch of
#         a 1 x Nchunks vector) are both exactly that, so the per-volume 3-D
#         registration AND the inter-chunk stitch contribute NOTHING to the
#         applied shifts in replicate mode. Measured on the validation subset:
#         dropping the chunk term changes the centred matrix by 3.6e-15 px,
#         i.e. not at all. Improved mode centres on a global scalar instead.
#       * correction 2 (`_apply_shifts_volume` XY, and the final translation in
#         `zproj_reg`): exact Fourier phase ramp instead of bilinear, and NO
#         uint16 re-quantization of the intermediate planes. The quantization
#         is MATLAB-faithful — its arrays really were uint16 there — but it
#         costs up to 0.5 counts per plane on data whose mean is ~22 counts,
#         and it exists only because of the storage class, not the algorithm.
#         Ringing/wraparound trade-offs are measured in cpstab/fourier_shift.py.
#       * correction 4 (`_z_shift_for` + `_apply_z_shift_subplane`): ZS is
#         measured at usfac=2, i.e. to half a plane; replicate rounds it to a
#         whole one and then clears an asymmetric band (note 3). Improved
#         interpolates between neighbouring planes with symmetric zero fill.
#     Correction 3 is in orchestrator._process_chunk, not here. Note that 1
#     and 3 change the SHIFTS, so an improved run's .dftshifts.npz differs from
#     replicate's; 2 and 4 change only how a given shift is applied. Anything
#     bypassing run_pipeline (fast_run.py) must install the mode itself.
# 15. write_registered=False default (review F2): MakeSBXall.m runs PASS2
#     (the .sbxall write, L139-202) UNCONDITIONALLY; the port defaults it
#     OFF because nothing in the repo consumes the file. This is the one
#     deliberate default-behavior change in this module; zproj_mean is
#     unaffected, and with write_registered=True the byte stream is verified
#     identical (tests/scratch_apply_project.py section B). Callers needing
#     full MATLAB parity must pass write_registered=True explicitly.
