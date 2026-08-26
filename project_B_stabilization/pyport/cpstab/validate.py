"""Validation harness: ported pipeline vs original MATLAB ground truth.

Runs cpstab.run_pipeline on a raw stack (bfconvert-produced OME-TIFF or
ImageJ TIFF), then compares the resulting z-projection time series against
the original pipeline's zproj TIFF:

  * per-frame Pearson r and NRMSE (per channel),
  * residual-offset probing: phase cross-correlation (dftregistrationAlex,
    usfac=50) truth-vs-port on a spread of frames — a consistent nonzero
    median indicates a systematic misalignment (wrong proj_range, axis swap,
    refchannel mismatch ...) rather than noise,
  * a markdown report + per-frame CSV + key-frame comparison PNGs.

Usage
-----
    python -m cpstab.validate \
        --raw   /data/run002_bfconvert.ome.tif \
        --truth /data/run002_mean_zproj.tif \
        --out   /data/validate_run002 \
        [--t-range 0:100] [--probe-frames 7] [--truth-axes TCYX] \
        [--params refchannel=1 proj_range=quarter proj_type=mean \
                  scale=4 chunksize=20]

Notes
-----
  * The pipeline ALWAYS runs on the full raw stack (registration chunking
    needs every volume); --t-range only subsets the COMPARISON, using
    Python-style 0-based half-open "start:stop" frame indices.
  * --params k=v pairs are forwarded to RegistrationConfig: refchannel/
    scale/chunksize (int), proj_range ('quarter' | 'full' | comma list of
    1-based planes), proj_type ('mean'|'max'|'median'), opttype,
    write_registered (true/false).
  * The truth TIFF axis order is auto-detected from tifffile series axes
    (ImageJ hyperstacks report e.g. TCYX). Plain multi-page TIFFs that
    report 'I' (unlabelled page axis) are interpreted as T with a warning;
    override with --truth-axes if that is wrong.
  * Outputs land in --out: cpstab_validation.md, per_frame_metrics.csv,
    frame_*.png, and port_run/ (the port's own .dftshifts.npz + zproj TIFF).
"""

import argparse
import csv
import os
import sys

import numpy as np
import tifffile

if __package__ in (None, ""):  # ran as a plain script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cpstab.config import RegistrationConfig
    from cpstab.pipeline import run_pipeline, matlab_uint16
    from cpstab.dftreg import dftregistration_alex
else:
    from .config import RegistrationConfig
    from .pipeline import run_pipeline, matlab_uint16
    from .dftreg import dftregistration_alex


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

_INT_KEYS = {"refchannel", "scale", "chunksize"}
_BOOL_KEYS = {"write_registered"}


def _parse_params(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit("--params entries must be key=value, got %r" % p)
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in _INT_KEYS:
            out[k] = int(v)
        elif k in _BOOL_KEYS:
            out[k] = v.lower() in ("1", "true", "yes")
        elif k == "proj_range" and "," in v:
            out[k] = [int(x) for x in v.split(",") if x.strip()]
        else:
            out[k] = v
    return out


def _parse_t_range(spec, nt):
    if not spec:
        return 0, nt
    try:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else nt
    except ValueError:
        raise SystemExit("--t-range must be 'start:stop' (0-based half-open)")
    a = max(0, a)
    b = min(nt, b)
    if a >= b:
        raise SystemExit("--t-range %r selects no frames of 0..%d" % (spec, nt))
    return a, b


# ---------------------------------------------------------------------------
# truth loading
# ---------------------------------------------------------------------------

def load_truth(path, axes_override=None, warnings_out=None):
    """Load the ground-truth zproj TIFF as (C, Y, X, T) float64.

    Detects the axis order from the tifffile series (ImageJ hyperstack
    metadata); tolerates singleton extra axes; maps an unlabelled page axis
    ('I'/'Q') to T with a warning.
    """
    warn = (warnings_out.append if warnings_out is not None else
            lambda m: print("WARNING:", m, file=sys.stderr))
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = (axes_override or series.axes).upper()
        arr = series.asarray()
    if len(axes) != arr.ndim:
        raise SystemExit("truth axes %r do not match data ndim %d"
                         % (axes, arr.ndim))

    # squeeze unknown singleton axes; relabel unlabelled page axis as T
    keep_axes = ""
    for i, ax in enumerate(axes):
        if ax in "TZCYX":
            keep_axes += ax
        elif arr.shape[i] == 1:
            arr = np.squeeze(arr, axis=len(keep_axes))
        elif ax in "IQS" and "T" not in axes:
            warn("truth axis %r (size %d) interpreted as T; pass "
                 "--truth-axes to override" % (ax, arr.shape[i]))
            keep_axes += "T"
        else:
            raise SystemExit("cannot interpret truth axis %r of size %d "
                             "(axes %r); pass --truth-axes"
                             % (ax, arr.shape[i], axes))
    axes = keep_axes
    if "Z" in axes:
        zi = axes.index("Z")
        if arr.shape[zi] == 1:
            arr = np.squeeze(arr, axis=zi)
            axes = axes.replace("Z", "")
        else:
            raise SystemExit("truth file has a real Z axis (size %d) — not a "
                             "z-projection?" % arr.shape[zi])
    for ax in "YX":
        if ax not in axes:
            raise SystemExit("truth axes %r lack %s" % (axes, ax))
    # add missing T/C as singletons, then order to (C, Y, X, T)
    for ax in "TC":
        if ax not in axes:
            arr = arr[None]
            axes = ax + axes
    order = [axes.index(a) for a in "CYXT"]
    return np.transpose(arr, order).astype(np.float64), axes


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def pearson_r(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return np.nan
    return float((a * b).sum() / den)


def nrmse(truth, port):
    truth = truth.astype(np.float64)
    port = port.astype(np.float64)
    rng = truth.max() - truth.min()
    if rng == 0:
        return np.nan
    return float(np.sqrt(np.mean((truth - port) ** 2)) / rng)


def residual_shift(truth, port, usfac=50):
    """(row, col) shift that would align `port` to `truth`."""
    s = dftregistration_alex(np.fft.fft2(truth.astype(np.float64)),
                             np.fft.fft2(port.astype(np.float64)), usfac)
    return float(s[0]), float(s[1])


# ---------------------------------------------------------------------------
# report pieces
# ---------------------------------------------------------------------------

def save_comparison_png(path, truth, port, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    diff = np.abs(truth.astype(np.float64) - port.astype(np.float64))
    lo, hi = np.percentile(truth, [1, 99.5])
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, name, kw in (
            (axs[0], truth, "truth (MATLAB)", dict(vmin=lo, vmax=hi)),
            (axs[1], port, "port (cpstab)", dict(vmin=lo, vmax=hi)),
            (axs[2], diff, "|diff|", {})):
        im = ax.imshow(img, cmap="gray", **kw)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True,
                    help="raw input stack (bfconvert OME-TIFF / ImageJ TIFF)")
    ap.add_argument("--truth", required=True,
                    help="original MATLAB *_zproj.tif ground truth")
    ap.add_argument("--out", required=True, help="report/output directory")
    ap.add_argument("--t-range", default=None,
                    help="comparison frame subset, 0-based half-open 'a:b'")
    ap.add_argument("--params", nargs="*", default=[],
                    help="RegistrationConfig overrides, key=value "
                         "(refchannel, scale, chunksize, proj_range, "
                         "proj_type, opttype, write_registered)")
    ap.add_argument("--truth-axes", default=None,
                    help="override truth TIFF axis order (e.g. TCYX)")
    ap.add_argument("--probe-frames", type=int, default=7,
                    help="frames probed for residual offset (default 7)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    port_dir = os.path.join(args.out, "port_run")
    os.makedirs(port_dir, exist_ok=True)
    warns = []

    params = _parse_params(args.params)
    cfg = RegistrationConfig(input_path=args.raw, out_dir=port_dir, **params)

    print("running ported pipeline on %s ..." % args.raw, file=sys.stderr)
    zproj = run_pipeline(cfg)                      # (C, Y, X, T) float64
    port = matlab_uint16(zproj).astype(np.float64)

    truth, truth_axes = load_truth(args.truth, args.truth_axes, warns)

    # shape reconciliation
    if truth.shape[1:3] != port.shape[1:3]:
        raise SystemExit("frame size mismatch: truth %r vs port %r — wrong "
                         "raw file, crop, or axis order?"
                         % (truth.shape[1:3], port.shape[1:3]))
    nc = min(truth.shape[0], port.shape[0])
    if truth.shape[0] != port.shape[0]:
        warns.append("channel count differs (truth %d, port %d); comparing "
                     "first %d" % (truth.shape[0], port.shape[0], nc))
    nt = min(truth.shape[3], port.shape[3])
    if truth.shape[3] != port.shape[3]:
        warns.append("frame count differs (truth %d, port %d); comparing "
                     "first %d" % (truth.shape[3], port.shape[3], nt))
    t0, t1 = _parse_t_range(args.t_range, nt)
    frames = list(range(t0, t1))

    refc = int(params.get("refchannel", cfg.refchannel)) - 1
    refc = min(max(refc, 0), nc - 1)

    # per-frame metrics
    metrics = []  # rows: t, channel, r, nrmse
    for t in frames:
        for c in range(nc):
            metrics.append((t, c + 1,
                            pearson_r(truth[c, :, :, t], port[c, :, :, t]),
                            nrmse(truth[c, :, :, t], port[c, :, :, t])))
    csv_path = os.path.join(args.out, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "channel", "pearson_r", "nrmse"])
        w.writerows(metrics)

    ref_rows = [m for m in metrics if m[1] == refc + 1]
    rs = np.array([m[2] for m in ref_rows])
    ns = np.array([m[3] for m in ref_rows])

    # residual-offset probing on the reference channel
    n_probe = max(1, min(args.probe_frames, len(frames)))
    probe_ts = [frames[i] for i in
                np.unique(np.linspace(0, len(frames) - 1, n_probe).astype(int))]
    probes = [(t,) + residual_shift(truth[refc, :, :, t], port[refc, :, :, t])
              for t in probe_ts]
    pr = np.array([[p[1], p[2]] for p in probes])
    med_shift = np.median(pr, axis=0)

    # key-frame PNGs: worst r, median r, best r
    order = np.argsort(rs)
    picks = sorted({frames[order[0]], frames[order[len(order) // 2]],
                    frames[order[-1]]})
    png_ok, pngs = True, []
    for t in picks:
        r_t = rs[frames.index(t)]
        name = "frame_t%04d.png" % t
        ok = save_comparison_png(
            os.path.join(args.out, name), truth[refc, :, :, t],
            port[refc, :, :, t],
            "t=%d  channel=%d  r=%.5f" % (t, refc + 1, r_t))
        png_ok &= ok
        if ok:
            pngs.append((t, name))
    if not png_ok:
        warns.append("matplotlib unavailable — comparison PNGs skipped")

    # ---------------- markdown report ----------------
    lines = []
    ap_ = lines.append
    ap_("# cpstab validation report\n")
    ap_("- raw: `%s`" % os.path.abspath(args.raw))
    ap_("- truth: `%s` (axes detected: `%s`, normalized to CYXT %r)"
        % (os.path.abspath(args.truth), truth_axes, truth.shape))
    ap_("- port zproj: `%s` %r" % (cfg.zproj_tiff_path(), port.shape))
    ap_("- config: refchannel=%d scale=%d chunksize=%d proj_range=%r "
        "proj_type=%s opttype=%s" % (cfg.refchannel, cfg.scale, cfg.chunksize,
                                     cfg.proj_range, cfg.proj_type, cfg.opttype))
    ap_("- compared frames: t=%d..%d (%d frames), %d channel(s); metrics "
        "channel %d\n" % (t0, t1 - 1, len(frames), nc, refc + 1))
    if warns:
        ap_("## Warnings\n")
        for w_ in warns:
            ap_("- %s" % w_)
        ap_("")

    ap_("## Summary (reference channel)\n")
    ap_(_md_table(
        ["metric", "value"],
        [["median Pearson r", "%.6f" % np.median(rs)],
         ["min Pearson r", "%.6f (t=%d)" % (rs.min(), frames[int(np.argmin(rs))])],
         ["median NRMSE", "%.5f" % np.median(ns)],
         ["max NRMSE", "%.5f (t=%d)" % (ns.max(), frames[int(np.argmax(ns))])],
         ["median residual shift (row, col) px",
          "(%.3f, %.3f)" % (med_shift[0], med_shift[1])],
         ["max |residual shift| px", "%.3f" % np.abs(pr).max()]]))
    ap_("")
    ap_("Reading: r ~ 1 and NRMSE ~ 0 mean pixelwise agreement; a CONSISTENT "
        "nonzero residual shift across probes points at a systematic "
        "misalignment (proj_range / refchannel / axis order), while "
        "scattered sub-pixel residuals are interpolation-level noise.\n")

    ap_("## Residual-offset probes (channel %d)\n" % (refc + 1))
    ap_(_md_table(["t", "row shift (px)", "col shift (px)"],
                  [(t, "%.3f" % r_, "%.3f" % c_) for t, r_, c_ in probes]))
    ap_("")

    if pngs:
        ap_("## Key-frame comparisons\n")
        for t, name in pngs:
            ap_("### t=%d\n" % t)
            ap_("![t=%d](%s)\n" % (t, name))

    ap_("## Worst 10 frames by Pearson r (channel %d)\n" % (refc + 1))
    worst = sorted(ref_rows, key=lambda m: m[2])[:10]
    ap_(_md_table(["t", "pearson r", "nrmse"],
                  [(m[0], "%.6f" % m[2], "%.5f" % m[3]) for m in worst]))
    ap_("")
    ap_("Full per-frame metrics: `per_frame_metrics.csv`.")
    if len(frames) <= 60:
        ap_("\n## All frames (channel %d)\n" % (refc + 1))
        ap_(_md_table(["t", "pearson r", "nrmse"],
                      [(m[0], "%.6f" % m[2], "%.5f" % m[3])
                       for m in ref_rows]))

    report = os.path.join(args.out, "cpstab_validation.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report: %s" % report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
