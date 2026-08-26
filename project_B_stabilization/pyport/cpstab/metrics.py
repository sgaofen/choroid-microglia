# -*- coding: utf-8 -*-
"""Quality metrics for a stabilized z-projection time series (PORT EXTENSION).

No MATLAB counterpart. The original pipeline shipped no way to answer "did
this run stabilize better than that one", which is exactly the question the
'improved' mode (cpstab/improved.py) raises. This module answers it with three
numbers per channel, chosen so that they cannot all be gamed at once.

    residual_px_median / residual_px_p95
        HOW MUCH MOTION IS LEFT. Sampled frames are registered against the
        temporal mean by subpixel phase correlation (dftregistration_alex,
        usfac=100); the reported numbers are the median and 95th percentile of
        the residual displacement magnitude, in full-resolution pixels. Lower
        is better, and this is the metric that actually measures the job the
        pipeline was hired to do. It is the primary one.

    sharpness
        HOW MUCH THE TEMPORAL MEAN WAS BLURRED BY THE MOTION THAT IS LEFT.
        Mean squared gradient of the time-averaged image, normalized by its
        mean intensity squared (so it is dimensionless and survives a global
        gain change). Residual motion smears the temporal mean, which costs
        gradient energy; better stabilization keeps it. Higher is better.

    field_noise_ratio
        THE CONTROL ON `sharpness`. Gradient energy is also produced by NOISE,
        so a run that merely preserved more high-frequency noise scores higher
        without being better. This measures the power at the exact vertical
        Nyquist frequency of the time-averaged image (the odd/even scan-line
        alternation of a resonant scanner, which is instrument artifact, not
        structure) relative to the median power in the upper half of the
        vertical band. 1.0 means no field artifact; large means the image is
        carrying a strong line-alternation component.

WHY THE THIRD METRIC IS NOT OPTIONAL
------------------------------------
Every resampling choice in this pipeline trades resolution against smoothing,
and the two cheap metrics point opposite ways under that trade. Bilinear
interpolation (replicate mode) is a low-pass filter: it destroys Nyquist
content, which LOWERS both `sharpness` and `field_noise_ratio`. An exact
Fourier shift (improved mode, correction 2) preserves Nyquist content, which
RAISES both. Reading `sharpness` alone would credit improved mode for
preserving line noise. Reading the two together separates the cases:

    sharpness up, field ratio flat ... real structure was recovered
    sharpness up, field ratio up ..... some of the gain is scan-line artifact
    sharpness down ................... the run blurred, whatever the cause

`residual_px_*` is independent of all of that -- it measures displacement, not
contrast -- which is why it is the one to trust when they disagree.

MEASUREMENT HYGIENE
-------------------
All three metrics are computed on a CENTRAL CROP (default 80% per axis).
Stabilization opens a black border of |shift| pixels that moves from frame to
frame; including it would let a run with LARGER shifts score better simply by
having more zeros (zeros are smooth, and they correlate with each other). The
crop is applied identically to every input so comparisons stay fair.

Before phase correlation each frame additionally gets its mean subtracted and
a separable Hann window applied. Mean subtraction removes the DC pedestal,
which on this data is ~91% of the correlation peak height
(cpstab/precision.py) and would otherwise swamp the alignment signal; the
window removes the edge discontinuity the DFT's periodic assumption creates.
Both are standard for phase correlation, both are applied identically to every
input, and neither touches the frames the other two metrics see.

USAGE
-----
    from cpstab.metrics import stabilization_metrics
    m = stabilization_metrics("run_mean_zproj.tif", sample_stride=4)

    # markdown comparison table, several runs at once
    python -m cpstab.metrics replicate=a_zproj.tif improved=b_zproj.tif \\
           --stride 4
"""

import os

import numpy as np

from .dftreg import dftregistration_alex

__all__ = ["stabilization_metrics", "markdown_table"]

_DEFAULT_CROP = 0.8
_USFAC = 100


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------

def _from_tiff(path, dtype=np.float64):
    """Read a zproj TIFF to (Nc, Nt, Y, X) using its own axes tag.

    write_zproj_tiff emits an ImageJ hyperstack whose series axes read 'TCYX'
    (Z is singleton and dropped); plain multi-page files report 'IYX' or 'QYX'
    for the unlabelled page axis, which is interpreted as T with one channel.

    `dtype=None` returns the pages in their STORED class (uint16 for anything
    this package writes) and is what stabilization_metrics uses: it casts
    after subsampling instead, which is exact (uint16 -> float64 loses
    nothing) and much cheaper. This function used to cast the whole file to
    float64 on the way in and then allocate a SECOND full-size float64 for a
    mean over an axis of length 1. On the production projection
    ((1500, 2, 512, 512) uint16, series.axes 'TCYX') that was 1.46 + 5.86 +
    5.86 = 13.21 GiB resident per input, measured, before --stride got a look
    in -- and bench_full.sh feeds four inputs to one process.
    """
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = str(series.axes).upper()
        arr = series.asarray()
    if arr.ndim == 2:
        return np.asarray(arr, dtype=dtype)[None, None]
    # unlabelled page axis -> treat as time
    axes = axes.replace("I", "T").replace("Q", "T")
    if len(axes) != arr.ndim:
        raise ValueError("tifffile reported axes %r for shape %r in %s"
                         % (axes, arr.shape, path))
    for ax in "CZT":
        if ax not in axes:
            arr = arr[..., None]
            axes = axes + ax
    arr = np.transpose(arr, [axes.index(a) for a in "CTZYX"])   # a view
    # collapse Z. A zproj has none, so the ordinary case is a size-1 axis:
    # INDEX it (a view) rather than averaging it, which allocated a full
    # second copy to divide by one. The averaging branch stays for a file
    # that really does carry Z, and keeps the float64 accumulation the mean
    # always had.
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    else:
        arr = np.asarray(arr, dtype=np.float64).mean(axis=2)
    return arr if dtype is None else np.asarray(arr, dtype=dtype)


def _as_ctyx(src, dtype=np.float64):
    """Normalize an input to (Nc, Nt, Y, X) float64.

    Accepts a path to a TIFF, or an in-memory array in the pipeline's own
    layout -- (Nc, Y, X, Nt) as returned by run_pipeline / make_sbxall, or
    (Y, X, Nt) for a single channel. The pipeline layout is assumed for 4-D
    arrays because that is the ONLY 4-D array this package produces; a TIFF
    read back from disk carries its own axes tag and is handled above.

    `dtype=None` skips the float64 cast and hands back the stored class; the
    caller is then responsible for casting before any arithmetic. See
    _from_tiff for why.
    """
    if isinstance(src, (str, bytes, os.PathLike)):
        return _from_tiff(src, dtype=dtype)
    a = np.asarray(src) if dtype is None else np.asarray(src, dtype=dtype)
    if a.ndim == 3:                      # (Y, X, Nt)
        a = a[None]
    if a.ndim != 4:
        raise ValueError(
            "metrics input must be a TIFF path, an (Nc, Y, X, Nt) array (the "
            "run_pipeline layout) or a (Y, X, Nt) array; got shape %r"
            % (a.shape,))
    return np.transpose(a, (0, 3, 1, 2))    # (Nc, Y, X, Nt) -> (Nc, Nt, Y, X)


def _central_crop(a, frac):
    """Central crop of the last two axes to `frac` of each, rounded to an EVEN
    row count (the field metric needs a real Nyquist bin along y)."""
    ny, nx = a.shape[-2], a.shape[-1]
    cy = max(4, int(round(ny * frac)))
    cx = max(4, int(round(nx * frac)))
    cy -= cy % 2
    cx -= cx % 2
    y0 = (ny - cy) // 2
    x0 = (nx - cx) // 2
    return a[..., y0:y0 + cy, x0:x0 + cx]


# ---------------------------------------------------------------------------
# the three metrics
# ---------------------------------------------------------------------------

def _hann2(shape):
    """Separable Hann window (periodic-safe outer product)."""
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _residual_motion(frames):
    """Per-frame residual displacement vs the temporal mean, in pixels.

    `frames` is (Nt, Y, X), already cropped. Returns an (Nt, 2) array of
    (dy, dx) as dftregistration_alex reports them: the shift that would have
    to be APPLIED to the frame to bring it onto the reference, i.e. exactly
    the motion the pipeline failed to remove.

    The reference is the mean of the SAME frames the residuals are measured
    for, so this is self-referential by construction -- the residuals are
    guaranteed to have near-zero mean and what carries the information is
    their spread. That is the intended reading (median/p95 of the magnitude),
    and it is also why the metric needs no external ground truth.
    """
    win = _hann2(frames.shape[1:])
    prepped = np.empty(frames.shape, dtype=np.float64)
    for i in range(frames.shape[0]):
        f = frames[i]
        prepped[i] = (f - f.mean()) * win
    ref_ft = np.fft.fft2(prepped.mean(axis=0))
    out = np.zeros((frames.shape[0], 2))
    for i in range(frames.shape[0]):
        out[i] = dftregistration_alex(ref_ft, np.fft.fft2(prepped[i]), _USFAC)
    return out


def _sharpness(mean_img):
    """Mean squared gradient of the time-averaged image / mean intensity^2.

    Forward differences on the common interior so gy and gx cover the same
    pixels. Normalizing by mean^2 makes the number dimensionless: a run that
    is uniformly brighter does not score higher for it.
    """
    gy = np.diff(mean_img, axis=0)[:, :-1]
    gx = np.diff(mean_img, axis=1)[:-1, :]
    mu = float(mean_img.mean())
    if not mu > 0:
        return float("nan")
    return float(np.mean(gy * gy + gx * gx) / (mu * mu))


def _field_noise_ratio(mean_img):
    """Vertical-Nyquist power / median upper-band power of the time mean.

    A resonant scanner writes alternate lines in opposite sweep directions, so
    any residual gain/timing mismatch between them appears as a pure
    alternation from row to row -- i.e. energy at exactly the vertical Nyquist
    frequency, and nowhere else. Taking the rFFT down each column and
    averaging the power over columns isolates that bin cleanly.

    Returned as a ratio against the MEDIAN power of the upper half of the
    vertical band (excluding the Nyquist bin itself), so it is normalized
    against whatever broadband high-frequency content the image happens to
    have. 1.0 = the Nyquist bin is unremarkable; 10 = a tenfold spike there.
    """
    m = mean_img - mean_img.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(m, axis=0)) ** 2
    p = p.mean(axis=1)                      # average over columns
    if p.size < 4:
        return float("nan")
    band = p[p.size // 2:-1]                # upper half, Nyquist excluded
    denom = float(np.median(band))
    if not denom > 0:
        return float("nan")
    return float(p[-1] / denom)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def stabilization_metrics(src, sample_stride=1, crop=_DEFAULT_CROP,
                          label=None):
    """Measure stabilization quality of one z-projection time series.

    Parameters
    ----------
    src : str or ndarray
        Path to a zproj TIFF, or an (Nc, Y, X, Nt) / (Y, X, Nt) array in the
        pipeline's layout (see _as_ctyx).
    sample_stride : int, default 1
        Use every Nth timepoint for the residual-motion estimate. The
        expensive part is one 2-D FFT pair per sampled frame; a stride of 4-8
        is plenty for a run of thousands of volumes. The temporal MEAN images
        that `sharpness` and `field_noise_ratio` use are always built from the
        sampled frames too, so a stride change moves all three numbers
        slightly -- compare runs at the SAME stride.
    crop : float, default 0.8
        Central fraction of each spatial axis to measure on; see MEASUREMENT
        HYGIENE in the module docstring.
    label : str, optional
        Name for this run in a markdown table (defaults to the file's stem).

    Returns
    -------
    dict with keys 'label', 'source', 'shape' (Nc, Nt, Y, X), 'sample_stride',
    'frames_used', 'crop', and 'channels' -- a list of one dict per channel
    holding 'channel' (1-based), 'residual_px_median', 'residual_px_p95',
    'residual_dy_median', 'residual_dx_median', 'sharpness' and
    'field_noise_ratio'.
    """
    if label is None:
        label = (os.path.splitext(os.path.basename(str(src)))[0]
                 if isinstance(src, (str, bytes, os.PathLike)) else "array")
    # Read in the STORED class and cast only what survives the stride and the
    # crop. Both of those are pure indexing, and uint16 -> float64 is exact,
    # so every number below is bit-for-bit what the old cast-everything-first
    # order produced -- it just stops materializing 5.9 GiB of float64 that
    # `idx` was about to throw 7/8 of away.
    a = _as_ctyx(src, dtype=None)
    nc, nt = a.shape[0], a.shape[1]
    stride = max(1, int(sample_stride))
    idx = np.arange(0, nt, stride)
    if idx.size < 3:
        raise ValueError(
            "need at least 3 sampled timepoints for a residual-motion "
            "estimate; Nt=%d with stride=%d gives %d" % (nt, stride, idx.size))
    a = np.asarray(_central_crop(a[:, idx], crop), dtype=np.float64)

    channels = []
    for c in range(nc):
        frames = a[c]
        res = _residual_motion(frames)
        mag = np.hypot(res[:, 0], res[:, 1])
        mean_img = frames.mean(axis=0)
        channels.append({
            "channel": c + 1,
            "residual_px_median": float(np.median(mag)),
            "residual_px_p95": float(np.percentile(mag, 95)),
            "residual_dy_median": float(np.median(res[:, 0])),
            "residual_dx_median": float(np.median(res[:, 1])),
            "sharpness": _sharpness(mean_img),
            "field_noise_ratio": _field_noise_ratio(mean_img),
        })
    return {
        "label": label,
        "source": str(src) if isinstance(src, (str, bytes, os.PathLike))
                  else "<array>",
        "shape": tuple(int(v) for v in (nc, nt, a.shape[2], a.shape[3])),
        "sample_stride": stride,
        "frames_used": int(idx.size),
        "crop": float(crop),
        "channels": channels,
    }


_COLUMNS = [
    ("residual_px_median", "resid px (median)", "%.4f", "lower"),
    ("residual_px_p95", "resid px (p95)", "%.4f", "lower"),
    ("sharpness", "sharpness", "%.5f", "higher"),
    ("field_noise_ratio", "field noise ratio", "%.3f", "control"),
]


def markdown_table(results):
    """Render stabilization_metrics() dicts as a markdown comparison table.

    One row per (run, channel). The arrow column says which direction is
    better so the table is readable without the module docstring; the field
    noise ratio deliberately has no direction -- it is the control on
    `sharpness`, not a score (see WHY THE THIRD METRIC IS NOT OPTIONAL).
    """
    if isinstance(results, dict):
        results = [results]
    lines = []
    head = ["run", "ch"] + [c[1] for c in _COLUMNS]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in results:
        for ch in r["channels"]:
            row = [r["label"], str(ch["channel"])]
            for key, _name, fmt, _dir in _COLUMNS:
                v = ch[key]
                row.append("n/a" if v != v else fmt % v)
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("better: " + ", ".join(
        "%s = %s" % (c[1], c[3]) for c in _COLUMNS))
    if results:
        r0 = results[0]
        lines.append("frames used: %d (stride %d), central crop %.0f%%, "
                     "measured region %dx%d px"
                     % (r0["frames_used"], r0["sample_stride"],
                        100 * r0["crop"], r0["shape"][2], r0["shape"][3]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m cpstab.metrics",
        description="Stabilization quality metrics for zproj TIFFs. Give one "
                    "or more inputs as PATH or LABEL=PATH.")
    p.add_argument("inputs", nargs="+", metavar="[LABEL=]PATH")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth timepoint (default 1); compare runs at "
                        "the same stride")
    p.add_argument("--crop", type=float, default=_DEFAULT_CROP,
                   help="central fraction of each spatial axis to measure on "
                        "(default %.2f)" % _DEFAULT_CROP)
    a = p.parse_args(argv)

    results = []
    for item in a.inputs:
        label, path = (item.split("=", 1) if "=" in item else (None, item))
        results.append(stabilization_metrics(path, sample_stride=a.stride,
                                             crop=a.crop, label=label))
    print(markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# DESIGN NOTES
# ------------
# 1. The residual-motion reference is the temporal mean of the same frames
#    being measured, not an external truth. That is deliberate: there IS no
#    ground truth for a real run, and a self-referential estimate still
#    answers the question that matters -- how much do the frames move relative
#    to each other. The consequence is that the residual MEAN is ~0 by
#    construction and only the spread is informative, which is why the metric
#    reports median/p95 of the magnitude and (for diagnosis) the median signed
#    dy/dx, never the mean.
# 2. usfac=100 for the residual estimate, versus the 2 and 4 the pipeline uses
#    to PRODUCE shifts. A metric may be more precise than the thing it
#    measures; using the pipeline's own coarse factors would quantize the
#    residuals to 0.25 px and hide exactly the sub-pixel differences these
#    corrections are about.
# 3. Why not report an SNR or a contrast-to-noise number: they are dominated
#    by acquisition (laser power, dwell time, dye), not by registration, so
#    they do not discriminate between two runs over the same raw data. Every
#    metric here is invariant to a global gain and responds only to how the
#    frames were aligned and resampled.
# 4. Nc is kept separate throughout rather than averaged. The two channels of
#    this dataset are a vessel-dye reference and a functional channel with
#    very different SNR; a pooled number would be dominated by whichever is
#    brighter and would hide a channel-specific regression.
# 5. field_noise_ratio needs an even crop height to have a true Nyquist bin,
#    which _central_crop enforces. It is computed on the TIME MEAN, not on
#    individual frames, because the scan-line artifact is fixed-pattern: it
#    survives averaging while shot noise does not, so the time mean is where
#    it stands out most clearly.
