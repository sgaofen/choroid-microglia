"""Top-level driver for the cpstab z-stack stabilization pipeline.

Mirrors: registration/RegistrationMasterPipeline.m (L1-44, the whole entry
script) restructured as run_pipeline(cfg), following the config-driven design
of clean/run_registration.m (L1-115).

Stage order is preserved exactly:

    (dims)            GetDimensions                 master L20
    (chunking)        Nchunks = round(Nt/chunksize) master L24
    (optotune)        CalculateOptotuneWarp         master L34   ('none' -> identity)
    (estimate)        DFT_warp_3D_2                 master L38   -> shift file
    (apply+project)   MakeSBXall                    master L41   -> zproj_mean
    (write)           write2chanTiff(uint16(...))   master L44   -> *_zproj.tif

Dropped relative to the master script (see README.md):
    * javaaddpath mij/ij/TurboReg (L3-5)     — no JVM; TIFF via tifffile
    * pipe.lab.datedir/datapath (L17-18)     — explicit cfg paths
    * ConvertOIR_SBX branch (L29-32)         — ingest handled by VolumeSource /
                                               external bfconvert
"""

import importlib
import os
import time
import warnings

import numpy as np

from .config import RegistrationConfig, matlab_round
from .improved import chain_refine_guard_scope, mode_scope
from .precision import compute_dtype_scope

__all__ = ["run_pipeline", "matlab_uint16"]


# ---------------------------------------------------------------------------
# MATLAB-semantics helpers
# ---------------------------------------------------------------------------
def matlab_uint16(x):
    """MATLAB uint16(x) cast: round half-away-from-zero, saturate to
    [0, 65535], NaN -> 0.

    Mirrors the uint16(zproj_mean) cast at RegistrationMasterPipeline.m L44.
    A bare np.astype(np.uint16) would TRUNCATE toward zero and WRAP
    negatives/overflow — both wrong. MATLAB: uint16(2.5)==3, uint16(-3.7)==0,
    uint16(70000)==65535, uint16(NaN)==0.

    Rounding is done by exact fractional-part comparison, NOT
    floor(abs(x)+0.5): for doubles just below a half-integer (e.g.
    np.nextafter(0.5, 0), exact value < 0.5) the addition itself IEEE-rounds
    up to the integer before floor, adding 1 (double rounding). With
    f = floor(a), a - f is exact (Sterbenz), so the >= 0.5 test below is the
    true-value comparison MATLAB's cast performs.
    (Verified against an exact-rational oracle: tests/scratch_f1_rounding.py.)
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isnan(x), 0.0, x)
    a = np.minimum(np.abs(x), 65536.0)  # tames +/-Inf; >= 65535.5 saturates anyway
    f = np.floor(a)
    x = np.sign(x) * np.where(a - f >= 0.5, f + 1.0, f)  # round half away from zero
    return np.clip(x, 0.0, 65535.0).astype(np.uint16)


# ---------------------------------------------------------------------------
# Sibling-module resolution (INTEGRATION SURFACE)
#
# The other cpstab modules are being ported in parallel against the rule
# "mirror the MATLAB signature, snake_case name". Their exact module file
# names are not fixed yet, so every cross-module symbol is resolved lazily
# here, in one place, from a candidate list. Integration only has to touch
# these tables.
# ---------------------------------------------------------------------------
def _try_resolve(candidates):
    """Return the first importable (module, attr) from candidates, else None."""
    pkg = __package__ or "cpstab"
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module("." + mod_name, pkg)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None:
            return fn
    return None


def _resolve(candidates, what, matlab_ref):
    fn = _try_resolve(candidates)
    if fn is None:
        raise ImportError(
            "cpstab integration: could not resolve %s (mirror of %s). "
            "Tried: %s. Add the real (module, attr) to the candidate table in "
            "cpstab/pipeline.py." % (what, matlab_ref, candidates)
        )
    return fn


_VOLUME_SOURCE_CANDIDATES = [
    ("io_rw", "VolumeSource"),  # the IO port landed as io_rw.py (review F1)
    ("io_sbx", "VolumeSource"),
    ("io", "VolumeSource"),
    ("volume_source", "VolumeSource"),
    ("sbx", "VolumeSource"),
]
_GET_DIMENSIONS_CANDIDATES = [
    # NB: io_rw.get_dimensions mirrors GetDimensions.m's (path, fdir, fbase)
    # signature, not the source-object call _dims() makes; it is deliberately
    # NOT listed here. VolumeSource/SbxFile expose Nz/Nt attributes, which
    # _dims() reads directly.
    ("io_sbx", "get_dimensions"),
    ("io", "get_dimensions"),
    ("volume_source", "get_dimensions"),
]
_DFT_WARP_CANDIDATES = [
    ("orchestrator", "dft_warp_3d_2"),  # strict snake_case of DFT_warp_3D_2.m
    ("orchestrator", "dft_warp_3d"),
]
_APPLY_CANDIDATES = [
    # The port landed under the MATLAB-mirror name (package rule 8), taking
    # the .sbx PATH first plus *_fn injection seams — NOT a VolumeSource
    # first argument (review F3; see _apply_io_adapter).
    ("apply_project", "make_sbxall"),
]
_WRITER_CANDIDATES = [
    ("io_rw", "write_zproj_tiff"),  # the IO port landed as io_rw.py (review F1)
    ("writer", "write_zproj_tiff"),
    ("io_tiff", "write_zproj_tiff"),
    ("io", "write_zproj_tiff"),
    ("tiff_out", "write_zproj_tiff"),
    ("io_rw", "write2chan_tiff"),   # strict snake_case of write2chanTiff.m
    ("writer", "write2chan_tiff"),
    ("io_tiff", "write2chan_tiff"),
]
_OPTOTUNE_CANDIDATES = [
    ("optotune", "calculate_optotune_warp"),
    ("orchestrator", "calculate_optotune_warp"),
    ("apply_project", "calculate_optotune_warp"),
]


def _dims(source):
    """[Nchan,Nx,Ny,Nz,Nt] <- GetDimensions.m L8-38 (sbx branch L10-17).

    Only Nz and Nt are consumed by this driver (Nx/Ny/Nchan fed the dropped
    ConvertOIR_SBX branch). Note GetDimensions L16 uses
    Nt = floor(nframes/otlevels) — floor, not round.
    Returns (nz, nt).

    Nz CONTRACT for the IO module: derive nz as len(info.otwave), the way
    the live MATLAB path does (MakeSBXall.m L12, DFT_warp_3D_2.m L21,
    CalculateOptotuneWarp.m L18 all use size(info.otwave,2)) — NOT
    info.otlevels as GetDimensions.m L15 does. sbxInfo.m L105-108 sets
    otlevels = length(otwave) only when optotune_used; for a plane-scan file
    (volscan == 0) with non-empty otwave the two diverge (1 vs length), and
    the original MATLAB pipeline is itself broken there (chunking scrambles).
    On the supported demo path (volscan > 0 / SpoofSBXinfo3D) they are
    identical. This driver takes nz from ONE place and feeds it to both the
    identity-tform count and the default proj_range, so the IO module's
    choice is the single source of truth — it must match the otwave-length
    convention.
    """
    for a_nz, a_nt in (("nz", "nt"), ("Nz", "Nt"), ("n_planes", "n_volumes")):
        if hasattr(source, a_nz) and hasattr(source, a_nt):
            return int(getattr(source, a_nz)), int(getattr(source, a_nt))
    fn = _try_resolve(_GET_DIMENSIONS_CANDIDATES)
    if fn is not None:
        nchan, nx, ny, nz, nt = fn(source)
        return int(nz), int(nt)
    raise AttributeError(
        "cpstab integration: VolumeSource exposes neither nz/nt attributes "
        "nor a get_dimensions() mirror (GetDimensions.m). Extend _dims() in "
        "cpstab/pipeline.py."
    )


def _apply_io_adapter(source):
    """Bridge the driver's IO seam to apply_project.make_sbxall (review F3).

    make_sbxall mirrors MakeSBXall.m: first argument is the .sbx PATH, and
    its IO happens through the injectable sbx_info_fn / imread_fn seams
    (defaults resolve io_rw.sbx_info / io_rw.imread, which need a real .sbx
    with sidecar). The driver, however, already holds an open source object
    (io_rw.VolumeSource for TIFF, io_rw.SbxFile for .sbx) — so wire the
    seams to that object instead of re-opening by path.

    Returns (sbx_info_fn, imread_fn) with the MATLAB contracts make_sbxall
    assumes (PORTING NOTES #11 there): sbx_info(path) -> info with
    otwave/sz/nchan; imread(path, k, N, pmt, optolevel) with k 1-BASED and
    the sbxRead return layout ((nchan, rows, cols, N) for pmt=-1, squeezed
    (rows, cols, N) otherwise), values NOT inverted (the .sbx write/read
    pair cancels its own 65535-inversion).
    """
    # SbxFile-like: sbxRead-compatible .read plus a ready info dict.
    if hasattr(source, "read") and hasattr(source, "info"):
        def sbx_info_fn(_path):
            return source.info

        def imread_fn(_path, k, n, pmt, optolevel):
            return source.read(k, n, pmt, optolevel)

        return sbx_info_fn, imread_fn

    # VolumeSource-like: 0-based get_volume(t, channel) -> [Y, X, Z] float64
    # holding the exact integer values of the file's native class.
    if hasattr(source, "get_volume"):
        nchan, nx, ny, nz, nt = (int(v) for v in source.metadata)
        info = {
            "sz": np.array([nx, ny], dtype=np.float64),  # [rows, cols]
            # only numel(otwave) is consumed downstream (Nz); synthesize the
            # standard (1, Nz) row vector (sbxInfo shape).
            "otwave": np.arange(1, nz + 1, dtype=np.float64)[None, :],
            "nchan": nchan,
            "nframes": nz * nt,
        }

        def sbx_info_fn(_path):
            return info

        def imread_fn(_path, k, n, pmt, optolevel):
            k0 = int(k) - 1
            if k0 % nz or int(n) != nz:
                raise ValueError(
                    "volume-source reads must be single whole volumes: "
                    "first frame %r / count %r with Nz=%d" % (k, n, nz))
            t = k0 // nz
            chans = range(nchan) if pmt == -1 else [int(pmt) - 1]
            vols = np.stack([source.get_volume(t, channel=c) for c in chans])
            return vols if pmt == -1 else vols[0]

        return sbx_info_fn, imread_fn

    raise TypeError(
        "cpstab integration: source %r exposes neither read/info (SbxFile) "
        "nor get_volume (VolumeSource); extend _apply_io_adapter in "
        "cpstab/pipeline.py." % type(source))


def _optotune_tforms(cfg, source, nz):
    """CalculateOptotuneWarp.m L1-50, restricted to the demo path.

    'none' (piezo): the .m returns repmat(affine2d(eye(3)),[1,Nz]) and
    returns early BEFORE touching Fiji (CalculateOptotuneWarp.m L20-23).
    Ported inline as a list of Nz identity 3x3 float64 matrices.

    'affine'/'rigid' (optotune lens): the .m shells out to Fiji
    MultiStackReg/TurboReg (L35-43) — not ported. If a sibling module ever
    provides calculate_optotune_warp, it is delegated to; otherwise
    NotImplementedError. (The master passed save='true' — a string, one of
    the original's truthy-string quirks; the port uses a real bool.)
    """
    if cfg.opttype == "none":
        return [np.eye(3, dtype=np.float64) for _ in range(nz)]
    fn = _try_resolve(_OPTOTUNE_CANDIDATES)
    if fn is None:
        raise NotImplementedError(
            "opttype=%r requires the Fiji MultiStackReg/TurboReg path "
            "(CalculateOptotuneWarp.m L35-43), which is not ported. Only "
            "'none' (piezo) is supported." % cfg.opttype
        )
    return fn(source, cfg.refchannel, cfg.scale, regtype=cfg.opttype, save=True)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg):
    """RegistrationMasterPipeline.m L1-44 — one end-to-end stabilization run.

    Parameters
    ----------
    cfg : RegistrationConfig or dict
        See cpstab.config.RegistrationConfig. A dict is accepted for
        MATLAB-struct-style convenience and converted.

    Returns
    -------
    zproj_mean : ndarray, float64
        The registered z-projection time series, exactly as MakeSBXall
        returned it (a double in MATLAB; master L41). The uint16-cast copy
        is what gets written to '<out_base>_<proj_type>_zproj.tif'
        (master L43-44).
    """
    if isinstance(cfg, dict):
        cfg = RegistrationConfig(**cfg)
    # PORT EXTENSIONS (no MATLAB counterpart): install this run's algorithm
    # mode (cpstab/improved.py) and float working class (cpstab/precision.py)
    # process-wide for the duration of the call, restoring both on the way out
    # -- including on an exception, so a failed improved/fast run cannot leave
    # a later replicate run in the wrong mode or in float32. The defaults
    # (cfg.mode='replicate', cfg.compute_dtype='float64') make both no-ops,
    # which is what keeps the default path bit-for-bit the validated port.
    # The chain-refine gate rides along in the same scope: it is a parameter
    # of correction 3, so it is only read when cfg.mode enables that
    # correction, and installing it here keeps the "one process-wide setting,
    # restored on the way out" rule the other two follow.
    with mode_scope(cfg.mode), compute_dtype_scope(cfg.compute_dtype), \
            chain_refine_guard_scope(cfg.chain_refine_cap,
                                     cfg.chain_refine_min_ncc):
        return _run_stages(cfg)


def _run_stages(cfg):
    """The master script's stage sequence proper (RegistrationMasterPipeline.m
    L17-44), split out of run_pipeline only so the mode / compute-dtype scopes
    can wrap it without re-indenting every stage. Assumes cfg is a real
    RegistrationConfig and that both process-wide settings are installed.

    The stage sequence itself is mode-independent: 'improved' changes what
    happens INSIDE the estimate and apply stages (orchestrator, apply_project),
    never their order or their arguments."""
    t0 = time.time()  # master L1: timerval = tic (original never calls toc)

    # ---- input + dimensions (master L17-20) -----------------------------
    VolumeSource = _resolve(
        _VOLUME_SOURCE_CANDIDATES, "VolumeSource", "pipe.io.sbxInfo/sbxRead"
    )
    source = VolumeSource(cfg.input_path)
    nz, nt = _dims(source)  # GetDimensions.m; only Nz, Nt used downstream

    # ---- chunking + projection parameters (master L22-26) ---------------
    nchunks = cfg.nchunks(nt)  # L24: round(Nt/chunksize), half-away-from-zero
    if nchunks < 1:
        # MATLAB would proceed and die later with a cryptic division/index
        # error inside DFT_warp_3D_2 (chunkframes = Nz*floor(Nt/0)); fail
        # early instead. Deviation in error behavior only.
        raise ValueError(
            "Nt=%d with chunksize=%d gives Nchunks=%d; need at least one "
            "chunk (>= chunksize/2 volumes)." % (nt, cfg.chunksize, nchunks)
        )
    proj_planes = cfg.proj_planes_1based(nz)  # 1-based, see config.py
    if cfg.proj_type != "mean":
        # Faithful quirk: MakeSBXall.m L120 always computes a MEAN
        # projection; the master's proj_type only renames the output file.
        warnings.warn(
            "proj_type=%r only affects the output filename; the projection "
            "computed is still the mean (MakeSBXall.m L120), exactly as in "
            "the original." % cfg.proj_type
        )

    out_dir = os.path.dirname(cfg.out_base()) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---- (master L29-32: ConvertOIR_SBX) --------------------------------
    # Dropped: ingest is VolumeSource's job; FluoView-export/.oir inputs are
    # converted externally (bfconvert). See README.md.

    # ---- optotune warp (master L34) --------------------------------------
    tforms_optotune = _optotune_tforms(cfg, source, nz)

    # ---- DFT shift estimation (master L37-38) ----------------------------
    shiftpath = cfg.shiftpath()  # L37: '<rundir>.dftshifts'
    dft_warp = _resolve(_DFT_WARP_CANDIDATES, "dft_warp_3d_2", "DFT_warp_3D_2.m")
    # Master L38 passes reftype='mean', overriding DFT_warp_3D_2's own
    # 'median' default (DFT_warp_3D_2.m L7) — must stay explicit here.
    dft_warp(
        source,
        shiftpath,
        cfg.refchannel,
        cfg.scale,
        nchunks,
        tforms_optotune,
        reftype="mean",
    )

    # ---- apply shifts + z-project (master L41) ---------------------------
    make_sbxall = _resolve(_APPLY_CANDIDATES, "make_sbxall", "MakeSBXall.m")
    sbx_info_fn, imread_fn = _apply_io_adapter(source)
    zproj_mean = make_sbxall(
        cfg.input_path,                     # path, as in MakeSBXall(path,...);
                                            # names the '<stem>.sbxall' output
        shiftpath,
        refchannel=cfg.refchannel,          # master L41 (default in .m was 2)
        proj_range=proj_planes,             # 1-based; MakeSBXall.m L13 default
        write_registered=cfg.write_registered,  # .sbxall write, now opt-in
        sbx_info_fn=sbx_info_fn,            # IO seam: reuse the open source
        imread_fn=imread_fn,
    )

    # ---- write projection TIFF (master L43-44) ---------------------------
    savepath = cfg.zproj_tiff_path()
    write_tiff = _resolve(_WRITER_CANDIDATES, "write_zproj_tiff", "write2chanTiff.m")
    write_tiff(matlab_uint16(zproj_mean), savepath)  # L44: write2chanTiff(uint16(...))

    print(
        "cpstab.run_pipeline: wrote %s  (%.1f min)"
        % (savepath, (time.time() - t0) / 60.0)
    )
    return zproj_mean


# PORTING NOTES
# 1. GLUE (pinned at integration, review F3): all cross-module symbols
#    resolve lazily through the candidate tables above; the real signatures
#    are now:
#      io_rw.VolumeSource(input_path)                            [pipe.io.*]
#      orchestrator.dft_warp_3d_2(source_or_path, shiftpath, refchannel,
#                    scale, nchunks, tforms_optotune, *, reftype=...)
#                                                     [DFT_warp_3D_2.m L1]
#      apply_project.make_sbxall(path, shiftpath, *, refchannel=...,
#                    proj_range=..., write_registered=...,
#                    sbx_info_fn=..., imread_fn=...)  [MakeSBXall.m L1]
#      io_rw.write_zproj_tiff(uint16_array, path)     [write2chanTiff.m L1]
#    dft_warp_3d_2 accepts the source OBJECT directly (its _resolve_source
#    seam); make_sbxall keeps the MATLAB path-first signature, so this
#    driver bridges the open source object into its sbx_info_fn/imread_fn
#    injection seams via _apply_io_adapter. The path passed to make_sbxall
#    still names the optional '<stem>.sbxall' output, which lands next to
#    the input exactly like MATLAB's RegWriter(path, ...).
# 2. refchannel and proj_range cross this boundary 1-BASED (MATLAB
#    convention), per the package rule that the IO/apply layers convert.
#    apply_and_project must subtract 1 from proj_range before slicing.
# 3. reftype='mean' is hardcoded at the dft_warp call because the master
#    script hardcodes it (L38); DFT_warp_3D_2's own default is 'median'
#    (L7). Do not "simplify" by dropping the keyword.
# 4. proj_type: in the original, zproj is ALWAYS the mean (MakeSBXall.m
#    L120); proj_type only appears in the filename (master L43). The port
#    reproduces this (warning emitted for 'max'/'median') rather than
#    implementing projections the original never computed.
# 5. uint16 cast (matlab_uint16): MATLAB uint16() rounds half-away-from-zero
#    and SATURATES; NaN -> 0. numpy astype truncates and wraps. This cast is
#    in the master script (L44), i.e. this module's responsibility, not the
#    writer's — the writer receives ready-made uint16, as in MATLAB.
# 6. nchunks guard: added a fail-fast for Nchunks < 1; MATLAB crashes later
#    with a cryptic error. Error-path deviation only, numerics untouched.
# 7. Optotune identity: for 'none' the .m early-returns Nz identity affine2d
#    objects (CalculateOptotuneWarp.m L20-23); ported as Nz np.eye(3)
#    matrices. MATLAB affine2d.T uses the ROW-VECTOR convention with the
#    translation in T(3,1:2) — irrelevant for the identity, but whichever
#    module implements apply/scale of these matrices (DFT_warp_3D_2.m
#    L133-136 scales T(3,1:2) by `scale`) must index [2, 0:2] in Python.
#    Also note MakeSBXall runs its warp loop even for identity transforms
#    (p.optotune defaults to the truthy STRING 'true', MakeSBXall.m L8);
#    identity imwarp with linear interp on an integer grid is a no-op, so
#    skipping it in the port is numerically equivalent.
# 8. Timing: original does tic (L1) with no matching toc; the print here
#    follows clean/run_registration.m L101 instead. Cosmetic.
# 9. zproj_mean is passed to the writer opaquely: MATLAB shape was
#    [Nc, Nx(rows), Ny(cols), Nt] (MakeSBXall.m L70). The project convention
#    says the channel dim is handled at the IO layer; whatever layout
#    apply_and_project returns, this driver only casts it — layout parity is
#    the apply/writer modules' contract to verify at integration.
# 10. os.makedirs(out_dir) is new (MATLAB assumed the run directory
#    existed). Benign deviation.
# 13. mode (PORT EXTENSION, cpstab/improved.py): run_pipeline is the ONLY
#     place cfg.mode is consumed — it installs the setting with mode_scope()
#     and the orchestrator / apply_project read the global at the four call
#     sites that differ. Consequences for this file:
#       * the stage list above is unchanged in both modes, so the ordering
#         comments and MATLAB line refs stay accurate;
#       * matlab_uint16() at L44 is applied in both modes. Improved mode does
#         NOT quantize its intermediate planes (that is correction 2), but the
#         deliverable is still a uint16 TIFF and still rounds half away from
#         zero exactly once, at the end;
#       * anything bypassing run_pipeline (fast_run.py, validate.py, a REPL)
#         starts in 'replicate' and must install the scope itself.
# 12. compute_dtype (PORT EXTENSION, cpstab/precision.py): run_pipeline is the
#     ONLY place cfg.compute_dtype is consumed — it installs the setting with
#     compute_dtype_scope() and every module then reads the global. Two
#     consequences to keep in mind when editing this file:
#       * matlab_uint16() below deliberately stays float64 in both modes. It
#         is the master script's L44 cast, applied once to the finished
#         projection; doing its half-away-from-zero rounding in float32 would
#         add representation error ON TOP of already-float32 pixel values and
#         move ties for no speed gain worth having.
#       * anything that bypasses run_pipeline (fast_run.py, validate.py, a
#         REPL) starts in float64 and must install the scope itself.
# 11. MATLAB's GetDimensions names Nx=sz(1) which is the ROW count — the
#    original's "Nx" is vertical. This driver never touches Nx/Ny so no
#    swap risk here, but keep it in mind reading the .m sources.
