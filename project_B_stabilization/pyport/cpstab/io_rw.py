# -*- coding: utf-8 -*-
"""io_rw.py — IO layer of the Shipley2020 choroid-plexus z-stack stabilizer port.

Mirrors (MATLAB ground truth in references/Shipley2020/):
    registration/ConvertOIR_SBX.m   -> convert_oir_sbx
    registration/SpoofSBXinfo3D.m   -> spoof_sbx_info_3d
    data/sbxRead.m                  -> sbx_read
    data/sbxInfo.m                  -> sbx_info
    data/RegWriter.m                -> RegWriter (class)
    data/imread.m  (SBX version)    -> imread
    data/GetDimensions.m            -> get_dimensions
    data/write2chanTiff.m           -> write2chan_tiff
    data/load_tiff.m                -> load_tiff
    data/load_tiff_nobar.m          -> load_tiff_nobar

New-design classes (no .m counterpart, see class docstrings):
    VolumeSource  — reads bfconvert OME-TIFF / ImageJ big-TIFF, OR the T-major
                    (T,Z,C,Y,X) `.npy` store written by scripts/relayout.py;
                    yields [Y,X,Z] float64 single-channel volumes indexed by
                    T (0-based). Both containers return the SAME values bit
                    for bit (PORTING NOTES #16).
    SbxFile       — legacy `.sbx` byte-contract reader with the same
                    get_volume interface (memmap-backed).
    write_zproj_tiff — general ImageJ-hyperstack writer aligned with the
                    MIJ output of write2chanTiff.

AXIS / NAMING CONVENTIONS (read this first)
-------------------------------------------
* Single frame is [Y, X] == MATLAB [row, col]. Volume [Y, X, Z]. Block
  [Y, X, Z, T]. Dimension order matches MATLAB; we do NOT reorder to
  C-contiguous-friendly [T, Z, Y, X].
* The MATLAB pipeline's "Nx" is the number of ROWS (our Y length) and
  "Ny" is the number of COLUMNS (our X length). This is confusing but it
  is what GetDimensions.m returns (Nx = size(im,1)) and what every caller
  assumes. We keep the MATLAB names verbatim to preserve signatures.
* Mirrored functions keep MATLAB 1-based indices (frame k, pmt, optolevel).
  New-design classes (VolumeSource / SbxFile.get_volume) are 0-based and
  say so in their docstrings.

THE 65535-INVERSION IS A ROUND-TRIP IDENTITY (hidden trap)
----------------------------------------------------------
RegWriter.write stores `65535 - value` (Scanbox convention: the PMT signal
is recorded inverted). sbxRead returns `65535 - raw`. Therefore
    sbx_read(RegWriter.write(v)) == v
and the pipeline math only ever sees the ORIGINAL FluoView TIFF values.
Consequently the new OME-TIFF direct path (VolumeSource) must apply NO
inversion at all: bfconvert output already holds the original values, the
same values the MATLAB pipeline computed on. The inversion exists only
inside the raw `.sbx` bytes; it is observable nowhere else.

SBX BYTE CONTRACT (extracted line-by-line from sbxRead/sbxInfo/RegWriter/
SpoofSBXinfo3D)
----------------------------------------------------------------------
* uint16, native little-endian, no header; the sidecar `<base>.mat` holds
  the `info` struct.
* One "frame"/record = one 2D slice with ALL channels interleaved.
  In-file order fastest -> slowest:  channel, column (X), row (Y); frames
  (z-slices, then time) slowest.  I.e. as a C-ordered numpy view the file
  is  (nframes, rows, cols, nchan)  of raw (inverted) uint16.
* info.sz == [rows, cols]; info.recordsPerBuffer == rows == info.width;
  info.height == cols == sz[1] (yes, "height"/"width" are swapped relative
  to their plain-English meaning; we mirror them anyway).
* record size in bytes: nsamples = sz[1]*recordsPerBuffer*2*nchan.
* nframes is derived from the ACTUAL file size (sbxInfo), not from the
  sidecar's nframes field.
* channels<->nchan inverse coding: channels==1 -> nchan=2 (both PMTs),
  channels==2|3 -> nchan=1. SpoofSBXinfo3D writes channels = 1 if
  nchan==2 else 2, so the coding round-trips.
* CODEMAP §13 Q8 (nsamples self-consistency) — RESOLVED: SpoofSBXinfo3D
  sets nsamples = width*height*2*nchan = sz[0]*sz[1]*2*nchan, and sbxInfo
  recomputes nsamples = sz[1]*recordsPerBuffer*2*nchan with
  recordsPerBuffer = width = sz[0]; both equal sz[0]*sz[1]*2*nchan, so the
  spoofed contract is self-consistent and `.sbx` files round-trip
  byte-exactly. (bytesPerBuffer, built from postTriggerSamples=5000, is
  only consumed by the scanbox_version<2 branch, which the spoof — with
  scanbox_version=2 — never takes.)
* KNOWN UPSTREAM QUIRK: ConvertOIR_SBX.m L38 calls
  SpoofSBXinfo3D(Ny, Nx, ...) (cols, rows) while MakeSBXall.m L134 calls
  SpoofSBXinfo3D(Nx, Ny, ...) (rows, cols). RegWriter's shape check
  (sz[0]==size(data,2), sz[1]==size(data,3)) can only pass for BOTH
  conventions when frames are square, which all Shipley2020 data is
  (512x512). We mirror both call sites verbatim, swap included.

Only numpy / scipy / skimage / tifffile are used (Python 3.9).
"""

import os
import re
import glob
import shutil

import numpy as np
import scipy.io as sio
import tifffile

# Float working class handed OUT of get_volume: float64 (replicate, default)
# or float32 (fast mode). uint16 -> either is exact, so this is a container
# choice, not a value change. See cpstab/precision.py and PORTING NOTES #18.
from .precision import get_compute_dtype

__all__ = [
    "spoof_sbx_info_3d",
    "sbx_info",
    "sbx_read",
    "imread",
    "RegWriter",
    "get_dimensions",
    "load_tiff",
    "load_tiff_nobar",
    "convert_oir_sbx",
    "write2chan_tiff",
    "write_zproj_tiff",
    "matlab_uint16",
    "SbxFile",
    "VolumeSource",
]

UINT16_MAX = np.uint16(65535)


# ---------------------------------------------------------------------------
# small MATLAB-semantics helpers
# ---------------------------------------------------------------------------

def matlab_uint16(x):
    """MATLAB `uint16(x)` for real input: round half-AWAY-from-zero, saturate.

    np.round would round half-to-even; MATLAB rounds 0.5 up (away from 0).
    For x >= 0, floor(x+0.5) is exactly half-away-from-zero; negatives
    saturate to 0 either way, so clipping after floor is equivalent.

    NaN maps to 0 (MATLAB `uint16(NaN)` == 0, a documented, deterministic
    conversion). NB np.clip does NOT remove NaN, and float->uint16 casting
    of NaN is C undefined behavior (platform-dependent + RuntimeWarning),
    so the explicit replacement below is load-bearing, not cosmetic.
    """
    x = np.asarray(x)
    if x.dtype == np.uint16:
        return x
    xf = np.floor(np.asarray(x, dtype=np.float64) + 0.5)
    xf = np.where(np.isnan(xf), 0.0, xf)    # MATLAB: uint16(NaN) == 0
    return np.clip(xf, 0, 65535).astype(np.uint16)


def _rescale(x):
    """MATLAB `rescale(X)` (R2017b+): (X - min) / (max - min), elementwise
    over the WHOLE array, output in [0, 1]. Degenerate max==min input would
    produce NaN here (see PORTING NOTES #9)."""
    x = np.asarray(x, dtype=np.float64)
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo)


_NAT_TOKEN = re.compile(r"(\d+)")


def _sort_nat(names):
    """Minimal port of data/sort_nat.m default mode (ascending natural order):
    digit runs compare numerically, text runs lexically. Only used by
    get_dimensions to pick the LAST C###Z###T### file, where any correct
    natural sort agrees (needed because T can exceed the %03d padding,
    e.g. T1200 > T999 numerically but not lexically)."""
    def key(name):
        parts = _NAT_TOKEN.split(name)
        return tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts)
    return sorted(names, key=key)


def _mat_to_dict(obj):
    """Normalize scipy.io.loadmat output (mat_struct / record arrays /
    0-d arrays) into a plain dict of python scalars & numpy arrays."""
    if isinstance(obj, sio.matlab.mat_struct):
        return {f: _mat_to_dict(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size == 1:
            return _mat_to_dict(obj.item())
        if obj.dtype.names:  # record array (struct_as_record=True fallback)
            o = obj.squeeze()
            return {n: _mat_to_dict(o[n].item() if o[n].dtype == object else o[n])
                    for n in obj.dtype.names}
        if obj.size == 1:
            return obj.reshape(-1)[0].item() if obj.dtype.kind in "biufc" else obj
        return obj
    return obj


def _load_info_mat(ipath):
    """Load the `info` struct from a `.mat` sidecar via scipy.io.loadmat.

    NOTE: the original ConvertOIR_SBX.m saves the sidecar with '-v7.3'
    (HDF5), which scipy CANNOT read and h5py is absent from the target
    venv. Our own convert_oir_sbx writes v5 sidecars instead (identical
    content, readable here). For legacy v7.3 sidecars use
    SbxFile(path, dims=(Nchan, Nx, Ny, Nz, Nt)) to bypass the sidecar.
    """
    try:
        mat = sio.loadmat(ipath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        raise IOError(
            "%s is a MATLAB v7.3 (HDF5) file; scipy.io.loadmat cannot read "
            "it and h5py is not available. Re-save it as v5, or open the "
            ".sbx with SbxFile(path, dims=(Nchan, Nx, Ny, Nz, Nt))." % ipath
        ) from e
    if "info" not in mat:
        raise KeyError("no `info` variable in %s" % ipath)
    info = _mat_to_dict(mat["info"])
    if not isinstance(info, dict):
        raise TypeError("`info` in %s is not a struct" % ipath)
    return info


def _isempty(v):
    if v is None:
        return True
    a = np.asarray(v)
    return a.size == 0


# ---------------------------------------------------------------------------
# SpoofSBXinfo3D.m
# ---------------------------------------------------------------------------

def spoof_sbx_info_3d(y_dim, x_dim, z_dim, t_dim, nchan):
    """registration/SpoofSBXinfo3D.m L1-34 (whole file).

    Fabricates a Scanbox `info` struct (as a dict) for a spoofed `.sbx`.

    ARGUMENT-ORDER TRAP: despite the parameter names, ConvertOIR_SBX.m L38
    passes (Ny, Nx, ...) = (cols, rows) whereas MakeSBXall.m L134 passes
    (Nx, Ny, ...) = (rows, cols). For MakeSBXall's (correct) order,
    sz = [rows, cols] and the byte contract in the module docstring holds
    literally; for ConvertOIR's order the two entries of sz are swapped —
    harmless for the square frames this pipeline is used on, and mirrored
    verbatim here.
    """
    info = {}
    info["resfreq"] = 7930                              # L4
    info["postTriggerSamples"] = 5000                   # L5
    info["nchannels"] = 1                               # L6
    info["abort_bit"] = 0                               # L7
    info["scanbox_version"] = 2                         # L8
    info["volscan"] = 1                                 # L9
    info["opto2pow"] = np.zeros((0, 0))                 # L10  ([] in MATLAB)
    info["power_depth_link"] = 0                        # L11
    info["area_line"] = True                            # L12

    info["sz"] = np.array([y_dim, x_dim], dtype=np.float64)   # L15
    info["height"] = int(info["sz"][1])                 # L16 (= x_dim!)
    info["width"] = int(info["sz"][0])                  # L17 (= y_dim!)
    info["nframes"] = int(z_dim) * int(t_dim)           # L18
    info["nchan"] = int(nchan)                          # L19
    info["max_idx"] = info["nframes"] - 1               # L20
    info["recordsPerBuffer"] = info["width"]            # L21
    info["bytesPerBuffer"] = (info["postTriggerSamples"]
                              * info["width"] * 2 * info["nchan"])  # L22
    info["nsamples"] = (info["width"] * info["height"]
                        * 2 * info["nchan"])            # L23
    if nchan == 2:                                      # L24-28
        info["channels"] = 1
    else:
        info["channels"] = 2
    info["scanmode"] = 1                                # L29

    info["optotune_used"] = 1                           # L32
    info["otlevels"] = int(z_dim)                       # L33
    info["otwave"] = np.arange(1, info["otlevels"] + 1) # L34
    return info


def save_sbx_info(mat_path, info):
    """Mirror of `save(strcat(...,'.mat'),'info','-v7.3')` in
    ConvertOIR_SBX.m L39.

    DIVERGENCE: written as MAT v5 (scipy has no v7.3 writer). Content is
    identical; v5 is what our own sbx_info reader consumes.
    """
    sio.savemat(mat_path, {"info": info})


# ---------------------------------------------------------------------------
# sbxInfo.m
# ---------------------------------------------------------------------------

# Mirrors the MATLAB globals `info_loaded` / `info` (sbxInfo.m L6).
_INFO_CACHE = {"base": None, "info": None}


def sbx_info(path, force=False, use_cache=True):
    """data/sbxInfo.m L1-121 (whole file).

    Loads the `.mat` sidecar next to a `.sbx*` file and derives read
    geometry. Returns a dict with the same field names as the MATLAB
    struct, plus info['fid'] = an OPEN binary file handle (mirroring the
    MATLAB fid; the previous cached handle is closed when a new file is
    loaded, exactly like L35-51 — but unlike MATLAB we never leak it on
    interpreter exit thanks to GC).

    `use_cache=False` (port-only escape hatch, default preserves MATLAB
    behavior) bypasses the module-level cache so independent readers do
    not stomp each other's fid.
    """
    parent, base_name = os.path.split(path)
    fname, ext = os.path.splitext(base_name)

    # LEGACY split-channel catch (L14-25)
    splitchan = False
    if ext in (".sbxreg", ".sbxclean", ".sbxregclean"):
        if "_reg" in fname:
            fname = fname[:fname.index("_reg")]
            splitchan = True
        elif "_xyreg" in fname:
            fname = fname[:fname.index("_xyreg")]
            splitchan = True

    # L27-31. (MATLAB concatenates a literal '\'; we join portably — the
    # hardcoded backslash is one of the CODEMAP §9 #6 mac-breaking bugs.)
    base = os.path.join(parent, fname) if parent else fname
    ipath = base + ".mat"
    if ext == "":
        path = path + ".sbx"

    if force and use_cache:                             # L34-42
        if _INFO_CACHE["base"] is not None:
            try:
                _INFO_CACHE["info"]["fid"].close()
            except Exception:
                pass
            _INFO_CACHE["base"] = None

    if use_cache and _INFO_CACHE["base"] == base:       # L45
        # Shallow copy: MATLAB's `nginfo = info` (L119) VALUE-copies the
        # struct, so caller mutations never poison the global. dict(...)
        # gives the same key-level immunity; the fid file object stays
        # shared, exactly like MATLAB's shared numeric file handle.
        return dict(_INFO_CACHE["info"])
    if use_cache and _INFO_CACHE["base"] is not None:   # L46-51
        try:
            _INFO_CACHE["info"]["fid"].close()
        except Exception:
            pass

    info = _load_info_mat(ipath)                        # L54

    if splitchan:                                       # L58-61
        info["nchan"] = 1
        info["channels"] = 2

    # L67-69: `if(~isfield(info,'sz')) sz = [512 796]; end` assigns a LOCAL
    # variable, not info.sz — a genuine no-op bug in the original. Mirrored
    # as a no-op; a truly sz-less info errors below, same as MATLAB would.

    info["scanmode"] = int(info["scanmode"])
    if info["scanmode"] == 0:                           # L72-75
        info["recordsPerBuffer"] = int(info["recordsPerBuffer"]) * 2
    else:
        info["recordsPerBuffer"] = int(info["recordsPerBuffer"])

    channels = int(info["channels"])                    # L79-89
    if channels == 1:
        info["nchan"] = 2   # both PMT0 & 1
        factor = 1
    elif channels == 2:
        info["nchan"] = 1   # PMT 0
        factor = 2
    elif channels == 3:
        info["nchan"] = 1   # PMT 1
        factor = 2
    else:
        raise ValueError("info.channels=%r not in {1,2,3}" % channels)

    info["fid"] = open(path, "rb")                      # L91
    nbytes = os.path.getsize(path)                      # L92 (d = dir(path))
    sz = np.asarray(info["sz"], dtype=np.float64).reshape(-1)
    info["sz"] = sz
    sz2 = int(sz[1])
    rpb = int(info["recordsPerBuffer"])
    info["nsamples"] = sz2 * rpb * 2 * info["nchan"]    # L93 bytes per record

    if info.get("scanbox_version", 0) and int(info["scanbox_version"]) >= 2:
        # L96: d.bytes/recordsPerBuffer/sz(2)*factor/4 - 1
        # == bytes/nsamples - 1 when the division is exact; MATLAB keeps a
        # double (fractional for truncated files) — we floor, see NOTES #8.
        info["max_idx"] = nbytes * factor // (rpb * sz2 * 4) - 1
        info["nsamples"] = sz2 * rpb * 2 * info["nchan"]        # L97 (dup)
    else:
        info["max_idx"] = nbytes * factor // int(info["bytesPerBuffer"]) - 1  # L99

    info["nframes"] = int(info["max_idx"]) + 1          # L103
    info["optotune_used"] = False                       # L104
    info["otlevels"] = 1                                # L105
    if "volscan" in info and not _isempty(info["volscan"]) \
            and float(np.asarray(info["volscan"]).reshape(-1)[0]) > 0:  # L106
        info["optotune_used"] = True
    if "volscan" not in info and not _isempty(info.get("otwave")):      # L107
        info["optotune_used"] = True
    if info["optotune_used"]:                           # L108
        info["otlevels"] = int(np.asarray(info["otwave"]).size)

    info["framerate"] = 15.49                           # L110-113
    if info["scanmode"] == 0:
        info["framerate"] = 30.98

    info["height"] = sz2                                # L115
    info["width"] = rpb                                 # L116

    if use_cache:
        _INFO_CACHE["base"] = base                      # L64
        _INFO_CACHE["info"] = info
        return dict(info)   # value-copy out of the cache (sbxInfo.m L119)
    return info


# ---------------------------------------------------------------------------
# sbxRead.m
# ---------------------------------------------------------------------------

def sbx_read(path, k=1, N=-1, pmt=1, optolevel=None):
    """data/sbxRead.m L1-90 (whole file).

    Reads frames k .. k+N-1 (k is 1-BASED, like the MATLAB function after
    its own 0->1 correction) from a `.sbx*` file, returning
    `65535 - raw` as uint16 (the write-time inversion is undone here; see
    module docstring).

    Shapes (after MATLAB squeeze semantics; trailing singleton frame axis
    dropped): nchan==1 or pmt>-1 -> [rows, cols, N] (or [rows, cols] if
    N==1); pmt==-1 with nchan==2 -> [2, rows, cols, N].
    N<0 or None reads to the end. `optolevel` (1-based) picks one z level
    of the optotune cycle.
    """
    info = sbx_info(path, True)                         # L17 (force reload)

    k = int(k) - 1                                      # L21 (to 0-based)
    if N is None or N < 0:                              # L24-25
        N = info["nframes"] - k
    N = int(N)
    if optolevel is not None and not info["optotune_used"]:   # L36-38
        raise ValueError("Optotune was not used for this file.")
    if N > info["nframes"] - k:                         # L33-34
        N = info["nframes"] - k

    fid = info.get("fid")                               # L41-43
    if fid is None or fid.closed:
        raise IOError("Cannot read file" + str(path))

    nsamples = int(info["nsamples"])
    nchan = int(info["nchan"])
    sz2 = int(info["sz"][1])
    rpb = int(info["recordsPerBuffer"])
    per_rec = nsamples // 2                             # uint16 per record

    if optolevel is None:                               # L48-55
        fid.seek(k * nsamples, os.SEEK_SET)             # L50
        x = np.fromfile(fid, dtype="<u2", count=per_rec * N)  # L51
        if x.size != per_rec * N:                       # L53-54 (catch)
            raise ValueError(
                "Cannot read frame. Index range likely outside of bounds.")
        x = x.reshape((nchan, sz2, rpb, N), order="F")  # L52
    else:                                               # L56-73
        optocycle = int(np.asarray(info["otwave"]).size)      # L57
        k = k * optocycle + (int(optolevel) - 1)        # L58
        if k + N * optocycle > info["nframes"]:         # L61-64
            N = int(np.floor((info["max_idx"] - k) / optocycle)) + 1  # L63
        bufwidth = nchan * sz2 * rpb                    # L65
        x = np.zeros(bufwidth * N, dtype=np.uint16)     # L66
        for n in range(N):                              # L67-70
            fid.seek((k + n * optocycle) * nsamples, os.SEEK_SET)
            chunk = np.fromfile(fid, dtype="<u2", count=per_rec)
            if chunk.size != per_rec:
                raise ValueError(
                    "Cannot read frame. Index range likely outside of bounds.")
            x[n * bufwidth:(n + 1) * bufwidth] = chunk
        x = x.reshape((nchan, sz2, rpb, N), order="F")  # L72

    # L75: invert and swap dims 2<->3 -> [nchan, rows, cols, N]
    x = UINT16_MAX - np.transpose(x, (0, 2, 1, 3))

    # L78-90: channel selection + MATLAB squeeze (trailing singletons drop)
    # NB the N<=1 else-branch mirrors MATLAB's THREE-subscript squeeze(
    # x(pmt,:,:)) on the 4-D array: the last subscript spans dims 3-4
    # merged column-major -> [rows, cols*N] (N==1: [rows,cols]; N==0:
    # [rows,0], a 2-D empty).
    if nchan == 1:
        x = x[0]                                        # -> [rows, cols, N]
        if N == 1:
            x = x[:, :, 0]                              # -> [rows, cols]
        elif N == 0:
            x = x.reshape((x.shape[0], 0))              # -> [rows, 0]
    elif pmt > -1:
        # MATLAB x(pmt,...) errors unless pmt is a positive integer <=
        # nchan; pmt==0 in particular must NOT silently wrap to x[-1]
        # (the last channel) — sbxRead is 1-indexed by its own contract.
        p = int(pmt)
        if p != pmt or not (1 <= p <= nchan):
            raise IndexError(
                "pmt=%r is not a valid 1-based channel index (nchan=%d); "
                "MATLAB x(pmt,...) errors here" % (pmt, nchan))
        x = x[p - 1]                                    # -> [rows, cols, N]
        if N == 1:
            x = x[:, :, 0]
        elif N == 0:
            x = x.reshape((x.shape[0], 0))
    else:
        if N == 1:                                      # MATLAB size drops
            x = x[:, :, :, 0]                           # trailing singleton
    return x


# ---------------------------------------------------------------------------
# imread.m  (SBX-only imread that shadows the builtin in MATLAB)
# ---------------------------------------------------------------------------

def imread(path, k=1, N=-1, pmt=1, optolevel=None,
           mtype=None, register=False, registration_path=None):
    """data/imread.m L1-74 (whole file). SBX-only movie reader (pipe.imread).

    In MATLAB this file SHADOWS the builtin imread (CODEMAP §9 #5); in
    Python it is safely namespaced as io_rw.imread.

    The register=True branch resolves to pipe.reg.aligned (dead code,
    CODEMAP §10) and is not ported — it raises NotImplementedError, except
    for the pre-existing `.sbxreg` redirect (L44-47) which is honored.
    """
    # L22: `isempty(N) || N < 0` — a 0-d / 1-element array N is a scalar
    # to MATLAB (NOT empty, so N=2 wrapped in an array still reads 2
    # frames); only genuinely empty or negative N maps to -1.
    if N is None or _isempty(N) or \
            (np.asarray(N).size == 1 and float(np.asarray(N).reshape(-1)[0]) < 0):
        N = -1
    ext = os.path.splitext(path)[1]                     # L39

    if register and (ext.lower() != ".sbxreg" or registration_path):  # L40
        if registration_path is None:
            base, name = os.path.split(path)
            name = os.path.splitext(name)[0]
            sbxreg = os.path.join(base, name + ".sbxreg")
            if os.path.exists(sbxreg):                  # L44-47
                return imread(sbxreg, k, N, pmt, optolevel)
        raise NotImplementedError(
            "read-time registration (pipe.reg.aligned) is dead code in the "
            "production path and was not ported")       # L50-63

    if (mtype is not None and str(mtype).lower() == "sbx") or \
            (len(ext) > 3 and ext[:4].lower() == ".sbx"):   # L68
        return sbx_read(path, k, N, pmt, optolevel)     # L69
    raise ValueError("Cannot read movie type.")         # L71


# ---------------------------------------------------------------------------
# RegWriter.m
# ---------------------------------------------------------------------------

class RegWriter(object):
    """data/RegWriter.m L1-111 (whole classdef). Streams volumes into a
    `.sbx`-family file, applying the `65535 - value` inversion at write
    time (see module docstring: write+read inversions cancel).

    Accepted `write` shapes (MATLAB dims, i.e. [C, rows, cols, frames]):
        nchan==1: [rows, cols] | [rows, cols, F]
        nchan==2: [2, rows, cols] | [2, rows, cols, F]
    Non-uint16 input is clamped to [0, 65535] and rounded MATLAB-style.

    Faithful quirks kept: the single-frame (3D) branch does NOT check the
    nframes bound (RegWriter.m L70-89 has no such check); shape checks
    compare against info.sz exactly as MATLAB does, so a swapped spoof
    (ConvertOIR_SBX) only passes for square frames. The warndlg calls
    (L54-55/75-76) are GUI-only no-ops and are dropped.
    """

    def __init__(self, path, info, extension=".sbxreg", force=False):
        # L12-30
        if not extension.startswith("."):
            extension = "." + extension                 # L19
        base, name = os.path.split(path)
        name = os.path.splitext(name)[0]
        path = os.path.join(base, name + extension)     # L20-21
        if os.path.exists(path) and not force:          # L23-25
            raise IOError("Cannot overwrite an existing file unless forced.")
        self.info = dict(info)                          # L27 (MATLAB
        #        obj.info = info value-copies the struct; keep that immunity)
        self.path = path                                # L28
        self.curframe = 0                               # L6 (fbs 9/17/18)
        self.fid = open(path, "wb")                     # L29

    # -- helpers -----------------------------------------------------------
    def _clamp_uint16(self, data):
        # L53-59 / L74-80: clamp then MATLAB uint16() rounding
        if data.dtype != np.uint16:
            data = np.clip(np.asarray(data, dtype=np.float64), 0, 65535)
            data = matlab_uint16(data)
        return data

    def _emit(self, d4):
        # L62-65 / L83-86: invert, permute [1 3 2 4], column-major flatten
        out = UINT16_MAX - np.transpose(d4, (0, 2, 1, 3))
        flat = np.asarray(out, dtype="<u2").flatten(order="F")
        self.fid.write(flat.tobytes())

    # -- API ---------------------------------------------------------------
    def write(self, data):
        """RegWriter.m L32-92."""
        data = np.squeeze(np.asarray(data))             # L40
        nchan = int(self.info["nchan"])
        sz = np.asarray(self.info["sz"]).reshape(-1)
        sz1, sz2 = int(sz[0]), int(sz[1])
        nframes = int(self.info["nframes"])

        nd = data.ndim
        # L41-47 reshapes. NB: in MATLAB a trailing singleton does not raise
        # ndims, so nchan==1&2D and nchan==2&3D land in the 3D branch below.
        if nchan == 1 and nd == 2:
            d4, matlab_nd = data[None, :, :, None], 3
        elif nchan == 1 and nd == 3:
            d4, matlab_nd = data[None, :, :, :], 4
        elif nchan == 2 and nd == 3:
            d4, matlab_nd = data[:, :, :, None], 3
        elif nd == 4:
            d4, matlab_nd = data, 4
        else:
            d4, matlab_nd = None, nd

        # L49-68: full-block branch (with nframes bound check)
        if matlab_nd == 4 and d4.shape[0] == nchan \
                and sz1 == d4.shape[1] and sz2 == d4.shape[2] \
                and d4.shape[3] + self.curframe <= nframes:
            d4 = self._clamp_uint16(d4)
            self.curframe += d4.shape[3]                # L61
            self._emit(d4)
            return

        # L70-89: frame-by-frame branch (fbs 20181108; no nframes check)
        if matlab_nd == 3 and d4 is not None and d4.shape[0] == nchan \
                and sz1 == d4.shape[1] and sz2 == d4.shape[2]:
            d4 = self._clamp_uint16(d4)
            self.curframe += d4.shape[3]                # L82 (size(data,4)=1)
            self._emit(d4)
            return

        raise ValueError(
            "Data must match that declared in info file. Check declared nchan!")

    def close(self):
        """RegWriter.m L94-103 (prints curframe like the MATLAB disp)."""
        print(self.curframe)                            # L96
        if self.fid is None:
            raise IOError("No file to close.")          # L97
        self.fid.close()

    def delete(self):
        """RegWriter.m L105-109 (destructor; silent close).
        ConvertOIR_SBX calls this, not close()."""
        if self.fid is not None and not self.fid.closed:
            self.fid.close()

    # pythonic conveniences (no MATLAB counterpart)
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.delete()
        return False


# ---------------------------------------------------------------------------
# GetDimensions.m
# ---------------------------------------------------------------------------

def get_dimensions(path, fdir, fbase):
    """data/GetDimensions.m L8-39 (function body).

    Returns (Nchan, Nx, Ny, Nz, Nt). REMINDER: Nx = ROWS, Ny = COLS
    (MATLAB pipeline convention; see module docstring).

    Try-branch reads the sbx sidecar; catch-branch scans the
    `<fbase>.tif.frames` FluoView export and parses the last (natural
    sort) file name `..._C###Z###T###.tif`. The MATLAB catch is a bare
    catch-all; mirrored here as `except Exception`.
    """
    try:                                                # L10-16
        info = sbx_info(path)
        nchan = int(info["nchan"])
        nx = int(info["sz"][0])
        ny = int(info["sz"][1])
        nz = int(info["otlevels"])
        nt = int(np.floor(info["nframes"] / info["otlevels"]))
        return nchan, nx, ny, nz, nt
    except Exception as sidecar_err:                    # L18-36
        tifdir = fdir + os.sep + str(fbase) + ".tif.frames"     # L19
        tiflist = sorted(glob.glob(os.path.join(tifdir, "*.tif")))  # L20
        # (L20 hardcodes '\*.tif' — CODEMAP §9 #6; glob is the portable
        # equivalent; sorted() mirrors MATLAB dir()'s alphabetical order)
        if not tiflist:
            # Legacy data (MATLAB-era v7.3 sidecar + already-deleted
            # .tif.frames dir) lands here: MATLAB's native load() reads
            # the v7.3 sidecar fine, but scipy cannot — do NOT degrade
            # that descriptive error into a bare IndexError on tiflist[0].
            raise IOError(
                "get_dimensions: reading the sbx sidecar failed (%s) and "
                "no .tif frames were found in %s. For a legacy v7.3 "
                "sidecar, open the .sbx with SbxFile(path, dims=(Nchan, "
                "Nx, Ny, Nz, Nt)) instead." % (sidecar_err, tifdir)
            ) from sidecar_err
        first_im = load_tiff(tiflist[0])                # L21
        names = [os.path.basename(f) for f in tiflist]
        tiflist_sort = _sort_nat(names)                 # L23
        last_name = tiflist_sort[-1]                    # L24
        parts = re.split(r"[_.]", last_name)            # L27
        a = parts[-2]                                   # L28 (C###Z###T###)
        limits = re.findall(r"[\d.]+", a)               # L29
        nchan = int(float(limits[0]))                   # L31 (str2num)
        nx = first_im.shape[0]                          # L32 (rows)
        ny = first_im.shape[1]                          # L33 (cols)
        nz = int(float(limits[1]))                      # L34
        nt = int(float(limits[2]))                      # L35
        return nchan, nx, ny, nz, nt


# ---------------------------------------------------------------------------
# load_tiff.m / load_tiff_nobar.m
# ---------------------------------------------------------------------------

def load_tiff(fpath):
    """data/load_tiff.m L1-26 (whole file).

    Loads a TIFF as float64 [height, width, frames]; a single-page file
    comes back 2D [height, width] (MATLAB drops the trailing singleton).
    The waitbar/progress chrome (L14, L18-24) is GUI-only and dropped.

    Non-grayscale pages (RGB / extra samples) raise: MATLAB's
    `img(:,:,i) = imread(...)` (L17) errors on a [h,w,3] page, and a
    single-page RGB file must NOT be mistaken for a 3-page grayscale
    stack (tifffile returns (Y,X,3) for it — same ndim as (F,Y,X)).
    Grayscale-ness is judged by the page's samplesperpixel, not by ndim.
    """
    with tifffile.TiffFile(fpath) as tf:
        spp = int(getattr(tf.pages[0], "samplesperpixel", 1) or 1)
        img = tf.asarray()
    if spp != 1:
        raise ValueError(
            "non-grayscale TIFF (%d samples/pixel) %s: MATLAB load_tiff's "
            "img(:,:,i) assignment errors on such pages too" % (spp, fpath))
    img = np.asarray(img, dtype=np.float64)             # zeros(...) is double
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return np.transpose(img, (1, 2, 0))             # (F,Y,X)->(Y,X,F)
    raise ValueError("unexpected TIFF dimensionality %r for %s"
                     % (img.shape, fpath))


def load_tiff_nobar(fpath):
    """data/load_tiff_nobar.m L1-17 (whole file; the .m even declares the
    function name `load_tiff` — a copy-paste of load_tiff.m minus the
    waitbar, byte-identical in output). Alias of load_tiff here."""
    return load_tiff(fpath)


# ---------------------------------------------------------------------------
# ConvertOIR_SBX.m
# ---------------------------------------------------------------------------

def convert_oir_sbx(mouse, date, run, fdir, fbase, Nx, Ny, Nz, Nt, Nchan,
                    lineshift=False, delete_frames=True, progress=True):
    """registration/ConvertOIR_SBX.m L1-107 (whole file).

    Converts a FluoView `<fbase>.tif.frames/` export into a monolithic
    `.sbx` + `.mat` sidecar. Returns the estimated line shift (an int) when
    lineshift=True, else the MATLAB `shift = []` -> None.

    Faithful details:
      * savename = `{mouse}_{date}_{run:03d}`; file `<fdir>/<savename>.sbx`.
      * L32-35 early exit when the frames dir is gone but the .sbx exists
        (returns None; MATLAB would error on the unassigned output).
      * L38 passes (Ny, Nx, ...) to SpoofSBXinfo3D — the swapped order,
        mirrored verbatim (square-frame-safe; see spoof_sbx_info_3d).
      * L39 sidecar save (v5 here instead of v7.3 — see save_sbx_info).
      * L42-45 copies `<fbase>.txt` into fdir if present (try/except pass).
      * Line-shift estimation L49-81: first 10 timepoints; every other ROW
        (1-based odd rows) circularly shifted along the COLUMN axis
        (MATLAB dim 3) by R in -5..5 — pure circshift, no zero clearing;
        per-channel `rescale` to [0,1]; channel-mean; row-derivative
        energy; argmin over R. Indexes channel 2 unconditionally, so
        Nchan==1 raises IndexError exactly as MATLAB would.
      * Write loop L90-101: per t builds vol[Nchan, Nx, Ny, Nz] (float64)
        from single-page TIFFs named `{fbase}_C%03dZ%03dT%03d.tif`, then
        writes MATLAB-`uint16(vol)` through RegWriter (which applies the
        65535-value inversion).
      * L106 deletes the frames folder; guarded by delete_frames=True
        (port-only safety valve, default preserves MATLAB behavior).
    parfor is replaced by a serial loop (bit-identical result).
    """
    tifdir = fdir + os.sep + str(fbase) + ".tif.frames"         # L26
    savename = "%s_%s_%03d" % (mouse, date, int(run))           # L27
    path = fdir + os.sep + savename + ".sbx"                    # L28

    if not os.path.exists(tifdir) and os.path.exists(path):     # L32-35
        print("Tiff directory does not exist. May have been deleted already")
        return None

    info = spoof_sbx_info_3d(Ny, Nx, Nz, Nt, Nchan)             # L38 (swap!)
    save_sbx_info(fdir + os.sep + savename + ".mat", info)      # L39

    try:                                                        # L42-45
        path_txt = tifdir + os.sep + str(fbase) + ".txt"
        shutil.copy(path_txt, fdir)
    except Exception:
        pass

    def _frame(c, z, t):
        tempname = "%s_C%03dZ%03dT%03d.tif" % (fbase, c, z, t)  # L54/L94
        return load_tiff_nobar(tifdir + os.sep + tempname)

    # -- line shift (L48-81) ----------------------------------------------
    shift = None                                                # L48
    if lineshift:
        A = np.zeros((Nchan, Nx, Ny, Nz, 10))                   # L50
        for t in range(1, 11):                                  # L51
            for z in range(1, Nz + 1):                          # L52
                for c in range(1, Nchan + 1):                   # L53
                    A[c - 1, :, :, z - 1, t - 1] = _frame(c, z, t)  # L55
        A = A.reshape((Nchan, Nx, Ny, -1), order="F")           # L59

        R = np.arange(-5, 6)                                    # L62
        B = np.zeros((Nx - 1, Ny, R.size))
        I = np.zeros((2, Nx, Ny))
        for i in range(R.size):                                 # L63-73
            temp = A.copy()
            temp[:, ::2, :, :] = np.roll(temp[:, ::2, :, :],
                                         int(R[i]), axis=2)     # L65 (dim 3)
            temp = temp.mean(axis=3)                            # L66
            I[0] = _rescale(temp[0])                            # L68
            I[1] = _rescale(temp[1])                            # L69 (Nchan==1
            #                                       -> IndexError, as MATLAB)
            Im = I.mean(axis=0)                                 # L70
            Im = _rescale(Im)                                   # L71
            B[:, :, i] = np.abs(np.diff(Im, 1, axis=0))         # L72
        trace = B.sum(axis=(0, 1))                              # L75
        pkidx = int(np.argmin(trace))                           # L79
        shift = int(R[pkidx])                                   # L80

    # -- conversion (L86-103) ---------------------------------------------
    rw = RegWriter(path, info, ".sbx", True)                    # L86
    try:
        for t in range(1, Nt + 1):                              # L90
            vol = np.zeros((Nchan, Nx, Ny, Nz))                 # L91
            for z in range(1, Nz + 1):                          # L92
                for c in range(1, Nchan + 1):                   # L93
                    vol[c - 1, :, :, z - 1] = _frame(c, z, t)   # L96
            rw.write(matlab_uint16(vol))                        # L99
            if progress and (t % 10 == 0 or t == Nt):           # L100 (bar)
                print("converting to sbx... %d/%d" % (t, Nt))
    finally:
        rw.delete()                                             # L102

    if delete_frames:
        shutil.rmtree(tifdir)                                   # L106
    return shift


# ---------------------------------------------------------------------------
# write2chanTiff.m  and the general hyperstack writer
# ---------------------------------------------------------------------------

def _matlab_stack_to_pages(mov):
    """Reproduce write2chanTiff.m's permute+reshape+MIJ page order.

    MATLAB: mov(C, rows, cols, [Z,] T) -> permute rows/cols first ->
    reshape merges the trailing dims COLUMN-MAJOR, so the ImageJ stack
    slice index runs C fastest, then Z, then T — which is exactly the
    'Stack to Hyperstack... order=xyczt(default)' interpretation AND the
    page order ImageJ uses when saving a hyperstack TIFF (c fastest,
    z middle, t slowest).

    DISPATCH USES MATLAB ndims SEMANTICS: MATLAB drops TRAILING singleton
    dims, so a [C,Y,X,1] (4-D, T==1) array has ndims 3 -> falls to the
    disp-only branch (nothing written — a genuine upstream quirk), and a
    [C,Y,X,Z,1] (5-D, T==1) array has ndims 4 -> takes the 4-D branch,
    which labels the hyperstack channels=C, slices=1, frames=Z (Z is
    relabeled as T; the page pixel order is unchanged). Mirrored verbatim
    here — see PORTING NOTES #15.

    Returns pages shaped (T, Z, C, Y, X) — tifffile's imagej axes order —
    plus (Nc, Nz, Nt); (None, ...) means the MATLAB else-branch (no write).
    """
    shape = list(mov.shape)
    while len(shape) > 2 and shape[-1] == 1:            # MATLAB ndims()
        shape.pop()
    eff = mov.reshape(shape)
    if eff.ndim == 4:                                   # L2-15
        nc, nx, ny, nt = eff.shape
        pages = np.transpose(eff, (3, 0, 1, 2))[:, None, :, :, :]  # (T,1,C,Y,X)
        return pages, nc, 1, nt
    elif eff.ndim == 5:                                 # L16-30
        nc, nx, ny, nz, nt = eff.shape
        pages = np.transpose(eff, (4, 3, 0, 1, 2))      # (T,Z,C,Y,X)
        return pages, nc, nz, nt
    return None, None, None, None


def write2chan_tiff(mov, path):
    """data/write2chanTiff.m L1-34 (whole file).

    mov: [C, rows(Y), cols(X), T] (4D) or [C, Y, X, Z, T] (5D), uint16
    (the master script does `uint16(zproj_mean)` before calling — use
    matlab_uint16 for that conversion).

    T==1 TRAP (faithful mirror of a MATLAB quirk, PORTING NOTES #15):
    a 4D input with T==1 writes NOTHING (MATLAB ndims==3 -> disp branch),
    and a 5D input with T==1 is written with frames=Nz, slices=1 (Z
    relabeled as T). Use write_zproj_tiff to write such data on purpose.

    Replaces the MIJ/ImageJ round trip (Miji -> MIJ.createImage ->
    'Stack to Hyperstack... order=xyczt(default)' -> Save Tiff) with
    tifffile's ImageJ-format writer. Page order derivation is in
    _matlab_stack_to_pages: slices run C fastest, Z, then T slowest —
    identical to an ImageJ hyperstack save. `display=Composite` maps to
    ImageJ metadata mode='composite' (display-only; pixel data unaffected).
    Exact byte-level equality with a MIJ-written file is NOT guaranteed
    (tag layout differs); pixel/axis-level equality is, pending
    calibration against a real ground-truth file (see write_zproj_tiff's
    axes parameters).
    """
    mov = np.asarray(mov)
    pages, nc, nz, nt = _matlab_stack_to_pages(mov)
    if pages is None:                                   # L31-32 (disp, no error)
        print("Movie dim must be 4 (CXYT) or 5 (CXYZT) for writing")
        return
    if pages.dtype not in (np.uint8, np.uint16, np.float32):
        raise TypeError(
            "ImageJ TIFF supports uint8/uint16/float32; got %s. The MATLAB "
            "caller converts with uint16() first — use matlab_uint16()."
            % pages.dtype)
    tifffile.imwrite(path, np.ascontiguousarray(pages), imagej=True,
                     metadata={"axes": "TZCYX", "mode": "composite"})


def write_zproj_tiff(mov, path, axes=None, page_axis_order="TZC",
                     imagej=True):
    """New-design generalized writer for the pipeline deliverable
    (`<rundir>_mean_zproj.tif`), aligned with write2chanTiff's MIJ result.

    mov  : ndarray whose layout is described by `axes` (default 'CYXT' for
           4D, 'CYXZT' for 5D — the write2chanTiff conventions; Y=rows).
    path : output file.
    page_axis_order : order of the non-YX axes from SLOWEST to FASTEST
           page index. Default 'TZC' == ImageJ hyperstack order (t slowest,
           c fastest) as derived from the MIJ call chain. Adjustable so the
           page order can be calibrated against a real MIJ-produced file
           later; any non-'TZC' order cannot be a spec-conformant ImageJ
           hyperstack, so it is written as a plain TIFF with an axes
           metadata tag instead.
    """
    mov = np.asarray(mov)
    if axes is None:
        axes = {4: "CYXT", 5: "CYXZT", 3: "CYX", 2: "YX"}.get(mov.ndim)
        if axes is None:
            raise ValueError("cannot infer axes for ndim=%d" % mov.ndim)
    axes = axes.upper()
    if len(axes) != mov.ndim or set(axes) - set("CYXZT"):
        raise ValueError("bad axes %r for shape %r" % (axes, mov.shape))

    # expand to full CYXZT
    full = "CYXZT"
    arr = mov
    for ax in full:
        if ax not in axes:
            arr = arr[..., None]
            axes = axes + ax
    order = [axes.index(ax) for ax in full]
    arr = np.transpose(arr, order)                      # (C, Y, X, Z, T)

    page_axis_order = page_axis_order.upper()
    if sorted(page_axis_order) != ["C", "T", "Z"]:
        raise ValueError("page_axis_order must be a permutation of 'TZC'")
    src = {"C": 0, "Y": 1, "X": 2, "Z": 3, "T": 4}
    perm = [src[a] for a in page_axis_order] + [src["Y"], src["X"]]
    pages = np.ascontiguousarray(np.transpose(arr, perm))

    if imagej and page_axis_order == "TZC":
        if pages.dtype not in (np.uint8, np.uint16, np.float32):
            raise TypeError("ImageJ TIFF needs uint8/uint16/float32, got %s"
                            % pages.dtype)
        tifffile.imwrite(path, pages, imagej=True,
                         metadata={"axes": "TZCYX", "mode": "composite"})
    else:
        tifffile.imwrite(path, pages,
                         metadata={"axes": page_axis_order + "YX"})


# ---------------------------------------------------------------------------
# New-design readers
# ---------------------------------------------------------------------------

class SbxFile(object):
    """Legacy `.sbx` reader implementing the byte contract from the module
    docstring (uint16, value = 65535 - raw, record layout channel-fastest ->
    cols -> rows, sidecar `.mat` via scipy.io.loadmat). Memmap-backed: no
    full load of multi-GB files.

    New-design class (no single .m counterpart; contract extracted from
    sbxRead.m/sbxInfo.m/RegWriter.m/SpoofSBXinfo3D.m — see the mirrored
    functions above for the line-by-line ports).

    Parameters
    ----------
    path : str            `.sbx` (or `.sbxall`, ...) file.
    dims : tuple, optional (Nchan, Nx, Ny, Nz, Nt) — bypasses the sidecar
           (needed for legacy v7.3 sidecars scipy cannot read; Nx=rows).
           Nt may be None to derive it from the file size.
    info : dict, optional  a ready sbx_info-style dict (advanced use).

    get_volume(t, channel=0) — 0-BASED t and channel — returns the
    [Y, X, Z] float64 volume (inversion undone), matching
    `double(pipe.imread(path, Nz*t+1, Nz, channel+1, []))`.
    """

    def __init__(self, path, dims=None, info=None):
        self.path = path
        if info is not None:
            self.info = dict(info)
        elif dims is not None:
            nchan, nx, ny, nz, nt = dims
            # MakeSBXall's (correct, unswapped) spoof order: sz=[rows, cols]
            self.info = spoof_sbx_info_3d(nx, ny, nz, 1 if nt is None else nt,
                                          nchan)
        else:
            self.info = sbx_info(path, force=True, use_cache=False)
            self.info["fid"].close()  # we use a memmap instead

        nchan = int(self.info["nchan"])
        sz = np.asarray(self.info["sz"]).reshape(-1)
        rows, cols = int(sz[0]), int(sz[1])
        nsamples = cols * rows * 2 * nchan          # bytes per record
        nbytes = os.path.getsize(path)
        nframes = nbytes // nsamples                # from file size (sbxInfo)
        self.info["nsamples"] = nsamples
        self.info["nframes"] = nframes
        self.info["max_idx"] = nframes - 1

        self.Nchan = nchan
        self.Nx = rows                               # rows (MATLAB Nx)
        self.Ny = cols                               # cols (MATLAB Ny)
        self.Nz = int(self.info.get("otlevels", 1))
        self.Nt = int(np.floor(nframes / self.Nz))   # GetDimensions.m L16
        # C-order view of the on-disk layout (see module docstring):
        self._mm = np.memmap(path, dtype="<u2", mode="r",
                             shape=(nframes, rows, cols, nchan))

    @property
    def metadata(self):
        """(Nchan, Nx, Ny, Nz, Nt) with the MATLAB meanings (Nx=rows)."""
        return (self.Nchan, self.Nx, self.Ny, self.Nz, self.Nt)

    def get_volume(self, t, channel=0):
        """[Y, X, Z] float64 volume at 0-based time t, 0-based channel."""
        if not (0 <= t < self.Nt):
            raise IndexError("t=%d out of range [0, %d)" % (t, self.Nt))
        if not (0 <= channel < self.Nchan):
            raise IndexError("channel=%d out of range" % channel)
        raw = self._mm[t * self.Nz:(t + 1) * self.Nz, :, :, channel]
        # undo inversion; float64 by default, float32 in fast mode (both hold
        # the uint16 values exactly, so 65535.0 - x is exact either way)
        vol = 65535.0 - np.asarray(raw, dtype=get_compute_dtype())
        return np.ascontiguousarray(np.transpose(vol, (1, 2, 0)))  # (Y,X,Z)

    def read(self, k=1, N=-1, pmt=1, optolevel=None):
        """sbxRead-compatible access (1-BASED k/pmt/optolevel, uint16
        output, MATLAB squeeze semantics) backed by the memmap; element-
        for-element identical to sbx_read(self.path, ...)."""
        k0 = int(k) - 1
        nframes = self.info["nframes"]
        if N is None or N < 0:
            N = nframes - k0
        N = int(N)
        if optolevel is not None and not self.info.get("optotune_used", False):
            raise ValueError("Optotune was not used for this file.")
        if N > nframes - k0:
            N = nframes - k0
        if N < 0:
            # k starts beyond the end: sbx_read's fread-size check raises
            # here (MATLAB's catch does too); an empty array would diverge.
            raise ValueError(
                "Cannot read frame. Index range likely outside of bounds.")
        if optolevel is None:
            if k0 < 0 or k0 + N > nframes:
                raise ValueError(
                    "Cannot read frame. Index range likely outside of bounds.")
            raw = self._mm[k0:k0 + N]                   # (N, rows, cols, C)
        else:
            optocycle = int(np.asarray(self.info["otwave"]).size)
            k0 = k0 * optocycle + (int(optolevel) - 1)
            if k0 + N * optocycle > nframes:
                N = int(np.floor((self.info["max_idx"] - k0) / optocycle)) + 1
            idx = k0 + optocycle * np.arange(N)
            raw = self._mm[idx]
        x = UINT16_MAX - np.transpose(np.asarray(raw), (3, 1, 2, 0))
        if self.Nchan == 1:
            x = x[0]
            if N == 1:
                x = x[:, :, 0]
            elif N == 0:
                x = x.reshape((x.shape[0], 0))          # MATLAB [rows, 0]
        elif pmt > -1:
            p = int(pmt)
            if p != pmt or not (1 <= p <= self.Nchan):  # see sbx_read
                raise IndexError(
                    "pmt=%r is not a valid 1-based channel index (nchan=%d); "
                    "MATLAB x(pmt,...) errors here" % (pmt, self.Nchan))
            x = x[p - 1]
            if N == 1:
                x = x[:, :, 0]
            elif N == 0:
                x = x.reshape((x.shape[0], 0))
        elif N == 1:
            x = x[:, :, :, 0]
        return x

    def close(self):
        self._mm = None

    def __len__(self):
        return self.Nt

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class VolumeSource(object):
    """Primary input reader of the new pipeline: a bfconvert-produced
    OME-TIFF (or ImageJ big TIFF) replacing the FluoView-export -> .sbx
    conversion entirely; also reads the T-major `.npy` store produced by
    scripts/relayout.py (same interface, same values — see PORTING NOTES
    #16 and the '.npy' paragraph below).

    New-design class (no .m counterpart). Because the .sbx write/read pair
    cancels its own 65535-inversion (module docstring), the values this
    class returns are exactly what the MATLAB pipeline computed on — NO
    inversion is applied here, and none must ever be added.

    get_volume(t, channel=0) — 0-BASED — returns [Y, X, Z] float64, the
    drop-in equivalent of `double(pipe.imread(path, Nz*t+1, Nz,
    channel+1, []))` after `reshape(...,Nx,Ny,Nz)`.

    Metadata: (Nchan, Nx, Ny, Nz, Nt) with MATLAB meanings — Nx = rows
    (image height, our Y), Ny = cols (width, our X).

    Reading strategy: tifffile memmap when the series is stored
    contiguously (uncompressed), else per-page reads via
    TiffFile.asarray(key=...) — either way a 60 GB stack is never loaded
    whole. Missing axes (no T, no C, ...) are treated as size 1.

    `.npy` INPUT (T-major store, scripts/relayout.py)
    -------------------------------------------------
    A path ending in '.npy' is opened with np.load(mmap_mode='r') and must
    hold a 5-D (T, Z, C, Y, X) C-order array — the layout in which ONE
    get_volume() call is a single contiguous byte range instead of Nz
    seeks scattered through a Z-major OME-TIFF. Everything this class
    exposes (metadata / dtype / Nchan / Nx / Ny / Nz / Nt / get_volume /
    get_frame / len / context manager) is identical to the TIFF path, and
    get_volume returns the SAME float64 values bit for bit: relayout copies
    the integers verbatim and neither path performs arithmetic.
    """

    _KNOWN = "TZCYX"

    def __init__(self, path, prefer_memmap=True, axes=None):
        self.path = path
        self._npy = None
        if str(path).lower().endswith(".npy"):
            self._init_npy(path, axes)
            return
        self._tif = tifffile.TiffFile(path)
        self._series = self._tif.series[0]
        s_axes = (axes or self._series.axes).upper()
        shape = tuple(self._series.shape)
        if len(s_axes) != len(shape):
            raise ValueError("axes %r do not match shape %r" % (s_axes, shape))

        # tolerate unknown singleton axes (e.g. 'S'/'Q' of size 1)
        sizes = {}
        keep = []
        for ax, n in zip(s_axes, shape):
            if ax in self._KNOWN:
                sizes[ax] = n
                keep.append(ax)
            elif n == 1:
                keep.append(ax.lower())     # marker: squeeze at index time
            else:
                raise ValueError(
                    "unsupported axis %r of size %d in %s (axes %r)"
                    % (ax, n, path, s_axes))
        self._axes = "".join(keep)          # lowercase = singleton unknown
        if "Y" not in sizes or "X" not in sizes:
            raise ValueError("series has no Y/X axes: %r" % s_axes)
        if not self._axes.upper().endswith(("YX",)):
            raise ValueError(
                "expected Y,X as the last axes, got %r — pass axes= to "
                "override if the file is nonstandard" % s_axes)

        self.Nchan = sizes.get("C", 1)
        self.Nz = sizes.get("Z", 1)
        self.Nt = sizes.get("T", 1)
        self.Nx = sizes["Y"]                # rows  (MATLAB Nx)
        self.Ny = sizes["X"]                # cols  (MATLAB Ny)

        # page bookkeeping: pages are ordered C-style over the non-YX axes
        self._page_axes = [a for a in self._axes if a.upper() not in "YX"]
        self._page_dims = [sizes.get(a, 1) for a in self._page_axes]

        self._mm = None
        if prefer_memmap:
            try:
                self._mm = tifffile.memmap(path, mode="r")
            except (ValueError, NotImplementedError):
                self._mm = None             # not contiguous; use page reads

    def _init_npy(self, path, axes=None):
        """Open a relayout.py T-major store: (T, Z, C, Y, X) .npy memmap.

        New-design ingest path (no .m counterpart), added to cure the
        read-scatter bottleneck of the Z-major bfconvert OME-TIFF (see the
        class docstring). Sets exactly the same attributes the TIFF branch
        sets, so no downstream code can tell the two apart.
        """
        if axes is not None and axes.upper() != "TZCYX":
            raise ValueError(
                "a .npy volume store is TZCYX by construction "
                "(scripts/relayout.py); got axes=%r" % (axes,))
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 5:
            raise ValueError(
                "%s is %d-D %r; a T-major volume store must be 5-D "
                "(T, Z, C, Y, X) — regenerate it with scripts/relayout.py"
                % (path, arr.ndim, tuple(arr.shape)))
        self._npy = arr
        self._tif = None
        self._series = None
        self._mm = None
        self.Nt, self.Nz, self.Nchan, self.Nx, self.Ny = (
            int(n) for n in arr.shape)      # Nx = rows (Y), Ny = cols (X)
        self._axes = "TZCYX"
        self._page_axes = ["T", "Z", "C"]
        self._page_dims = [self.Nt, self.Nz, self.Nchan]

    @property
    def metadata(self):
        """(Nchan, Nx, Ny, Nz, Nt); Nx = rows, Ny = cols (MATLAB naming)."""
        return (self.Nchan, self.Nx, self.Ny, self.Nz, self.Nt)

    @property
    def dtype(self):
        """Native storage dtype of the underlying TIFF series (e.g. uint16).

        get_volume always returns a FLOAT (float64, or float32 in fast mode),
        but the values are an exact image
        of this storage class. The orchestrator uses this to restore the
        MATLAB uint16 quantization chain (imresize/imwarp/imtranslate on
        uint16) when the file is integer-typed — the cast back is lossless
        because no arithmetic touches the values in between.

        For a .npy store this is the stored array's dtype, which relayout
        copies straight from the TIFF series — same value either way.
        """
        if self._npy is not None:
            return np.dtype(self._npy.dtype)
        return np.dtype(self._series.dtype)

    def _check(self, t, channel):
        if not (0 <= t < self.Nt):
            raise IndexError("t=%d out of range [0, %d)" % (t, self.Nt))
        if not (0 <= channel < self.Nchan):
            raise IndexError("channel=%d out of range [0, %d)"
                             % (channel, self.Nchan))

    def get_volume(self, t, channel=0):
        """[Y, X, Z] volume at 0-based time t, 0-based channel.

        dtype is the compute class — float64 by default (the replicate
        precision), float32 when the pipeline runs in fast mode. The VALUES
        are identical in both cases: the store is uint16 and no arithmetic
        happens here, so both classes hold them exactly (PORTING NOTES #18).
        """
        self._check(t, channel)
        fdt = get_compute_dtype()
        if self._npy is not None:
            # (T, Z, C, Y, X)[t, :, c] -> (Z, Y, X), ONE contiguous read of
            # Nz strided C-blocks; identical values to the TIFF branch.
            vol = np.asarray(self._npy[t, :, channel], dtype=fdt)
            return np.ascontiguousarray(np.transpose(vol, (1, 2, 0)))
        if self._mm is not None:
            idx = []
            for a in self._axes:
                au = a.upper()
                if au == "T":
                    idx.append(t)
                elif au == "C":
                    idx.append(channel)
                elif au == "Z" or au in "YX":
                    idx.append(slice(None))
                else:                        # unknown singleton
                    idx.append(0)
            vol = np.asarray(self._mm[tuple(idx)], dtype=fdt)
            if "Z" not in self._axes:
                vol = vol[None]              # (1, Y, X)
        else:
            sel = {"T": t, "C": channel, "Z": None}
            fixed = []
            for a in self._page_axes:
                au = a.upper()
                v = sel.get(au, 0)
                fixed.append(np.arange(self._page_dims[self._page_axes.index(a)])
                             if v is None else np.array([v]))
            # C-order flat page indices for all Z at this (t, channel)
            grids = np.meshgrid(*fixed, indexing="ij")
            flat = np.ravel_multi_index([g.ravel() for g in grids],
                                        self._page_dims, order="C")
            data = self._tif.asarray(key=[int(i) for i in flat], series=0)
            vol = np.asarray(data, dtype=fdt)
            if vol.ndim == 2:
                vol = vol[None]              # single page -> (1, Y, X)
        # (Z, Y, X) -> [Y, X, Z]
        return np.ascontiguousarray(np.transpose(vol, (1, 2, 0)))

    def read_block(self, t0, t1):
        """Whole-run handle on volumes [t0, t1) — T-MAJOR .npy ONLY.

        New-design method (no .m counterpart), added for the parallel
        driver's apply stage (fast_run.py, PORTING NOTES #19). In the
        (T, Z, C, Y, X) store a RUN of volumes is one contiguous byte
        range, so a slab can be addressed as a single array.

        Returns
        -------
        ndarray or None
            (t1-t0, Z, C, Y, X) in the store's NATIVE dtype — a VIEW on
            the read-only memmap, NOT an in-RAM copy — or None when this
            source is not a .npy store (callers then fall back to
            get_volume(), which is what every other reader does anyway).

        A VIEW, deliberately, and the docstring used to claim the opposite.
        np.asarray() on an np.memmap only drops the subclass; it copies
        nothing (OWNDATA False, np.shares_memory(blk, self._npy) True), so
        the bytes are still faulted in on demand when the caller touches
        them. That was measured, not assumed, and forcing a real copy with
        np.array() was measured too: on this machine it is not a win. One
        process, a 50-volume production slab: copy 0.53-0.82 s + 1.61 s
        consume = 2.15-2.43 s total, against 1.83-2.00 s for the view. Ten
        concurrent workers (the case PORTING NOTES #19 argues for): copy
        5.03 s wall, view 4.60 s and 5.71 s — the two view runs straddle
        the copy, i.e. the gap is inside the run-to-run noise. So the copy
        buys no measurable overlap here while making ~2.1 GB per worker
        genuinely resident, and the view stays.

        The float cast get_volume() performs is deliberately NOT done
        here: the caller slices one volume out of the block and casts it
        exactly as get_volume() would, so the two routes are bit-for-bit
        the same array (no arithmetic on either). Aliasing is a non-issue
        because the mmap is opened read-only and every caller only reads.
        """
        if self._npy is None:
            return None
        t0, t1 = int(t0), int(t1)
        if not (0 <= t0 <= t1 <= self.Nt):
            raise IndexError("block [%d, %d) out of range [0, %d)"
                             % (t0, t1, self.Nt))
        return np.asarray(self._npy[t0:t1])

    def get_frame(self, t, z, channel=0):
        """[Y, X] single slice (compute-class float), all indices 0-based."""
        vol = self.get_volume(t, channel)    # simple & correct; optimize later
        return vol[:, :, z]

    def close(self):
        self._mm = None
        self._npy = None                     # drops the np.load memmap
        if self._tif is not None:
            self._tif.close()

    def __len__(self):
        return self.Nt

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ===========================================================================
# PORTING NOTES
# ===========================================================================
# 1. Nx/Ny naming. The MATLAB pipeline's Nx is the ROW count (image height,
#    our Y axis) and Ny the COLUMN count — pinned by GetDimensions.m L32-33
#    (Nx = size(first_im,1)) and by every reshape(raw, Nx, Ny, Nz, []) call
#    site. All ports keep the MATLAB names; [Y, X, Z] volumes therefore have
#    shape (Nx, Ny, Nz) under these names. Documented everywhere it bites.
# 2. SpoofSBXinfo3D argument swap. ConvertOIR_SBX.m L38 passes (Ny, Nx, ...)
#    but MakeSBXall.m L134 passes (Nx, Ny, ...). With ConvertOIR's order,
#    info.sz = [cols, rows]; RegWriter's checks (sz(1)==size(data,2)==rows)
#    then only pass when rows==cols, and GetDimensions would return Nx/Ny
#    swapped for non-square data. All Shipley2020 data is 512x512 square, so
#    this never fired. Mirrored verbatim (both orders); SbxFile(dims=...)
#    uses the MakeSBXall (semantically correct) order.
# 3. 65535-inversion identity. RegWriter writes 65535-v, sbxRead returns
#    65535-raw; composition is the identity, so pipeline math always sees
#    original FluoView values, and the OME-TIFF direct path (VolumeSource)
#    performs NO inversion. Verified against RegWriter.m L62/L83 and
#    sbxRead.m L75. uint16 arithmetic cannot overflow here (raw <= 65535).
# 4. v7.3 sidecars. ConvertOIR_SBX.m L39 saves the .mat with '-v7.3'
#    (HDF5); scipy.io.loadmat cannot read those and h5py is absent from the
#    target venv. Choice: our save_sbx_info writes v5 (identical content);
#    for legacy v7.3 sidecars sbx_info raises a descriptive IOError and
#    SbxFile accepts dims=(Nchan,Nx,Ny,Nz,Nt) to reconstruct the info.
# 5. MATLAB squeeze / trailing singletons. sbxRead returns [rows, cols]
#    when N==1 because MATLAB drops trailing singleton dims; mirrored by
#    explicitly dropping the frame axis when N==1 (including the pmt==-1
#    4D case). RegWriter's branch dispatch depends on MATLAB's
#    trailing-singleton ndims semantics (a [2,r,c,1] reshape still has
#    ndims 3), so nchan==2 single frames take the L70-89 branch, which has
#    NO nframes bound check — quirk preserved.
# 6. GUI surfaces dropped: waitbar/parfor_progressbar -> print;
#    RegWriter's warndlg clamp warnings dropped (the clamping itself is
#    kept; note the MATLAB `if min(data) < 0` on an array is effectively
#    broken anyway — `if` on an array requires ALL elements true).
# 7. write2chanTiff page order. permute[2,3,1,4]+column-major reshape makes
#    the ImageJ stack index run C fastest then (Z) then T; 'Stack to
#    Hyperstack... order=xyczt(default)' interprets it identically, and an
#    ImageJ hyperstack TIFF saves pages in the same czt order. tifffile's
#    imagej=True writer with (T, Z, C, Y, X) input reproduces that page
#    order. Byte-level tag layout will differ from a MIJ file; pixel and
#    axis semantics should not. write_zproj_tiff exposes axes/
#    page_axis_order for calibration against a real MIJ-produced file.
# 8. sbxInfo max_idx arithmetic. MATLAB computes
#    d.bytes/recordsPerBuffer/sz(2)*factor/4 - 1 in doubles; with
#    factor/(4) == 1/(2*nchan) this equals bytes/nsamples - 1 and is
#    integral for well-formed files. We use integer floor division —
#    identical for whole files, and for a truncated file we floor where
#    MATLAB would propagate a fractional frame count into fread (and fail
#    later anyway). Also mirrored: nframes comes from the ACTUAL file
#    size, never from the sidecar's nframes field.
# 9. MATLAB rescale edge case: rescale(X) with max==min divides 0/0; our
#    _rescale reproduces the plain formula (NaN result). Never hit on real
#    two-channel image data (only used in the lineshift estimator).
#    OPEN QUESTION (review F6, unresolved — no MATLAB on this machine):
#    MATLAB's rescale MAY guard constant input and return the lower bound
#    (0) instead of NaN. If so, a constant channel (e.g. dead PMT) gives
#    MATLAB a real lineshift argmin while our NaN trace degenerates to
#    shift=-5. To settle: run `rescale(ones(3))` in any MATLAB; if it
#    returns zeros, add `if hi == lo: return np.zeros_like(x)` to _rescale.
# 10. dir() ordering. GetDimensions L21 loads tiflist(1) in MATLAB dir()
#    order (OS alphabetical); we use sorted(glob(...)). For the FluoView
#    zero-padded names the first file is the same; only its SHAPE is used.
#    The 'last' file for the C/Z/T limits uses natural sort (sort_nat) in
#    both, so T>999 (e.g. T1200) parses correctly.
# 11. MATLAB catch-all in GetDimensions is mirrored as `except Exception`,
#    so a missing/corrupt sidecar falls through to the TIFF-name scan just
#    like MATLAB (including path=None / path='').
# 12. Indexing bases. Mirrored functions keep MATLAB 1-based frame/pmt/
#    optolevel arguments (their docstrings say so). New-design classes
#    (VolumeSource, SbxFile.get_volume) are 0-based. Do not mix them up
#    when wiring the orchestration module.
# 13. convert_oir_sbx extras: delete_frames=True guards the destructive
#    rmdir (default preserves MATLAB), progress=True replaces the parfor
#    progress bar. The duplicated inputParser block (ConvertOIR_SBX L2-5 /
#    L20-23) is a no-op in MATLAB and has no port. The early-return path
#    (L32-35) returns None where MATLAB would error on an unassigned
#    output when the caller requests it.
# 14. sbxRead's fid sanity check (L41: `info.fid == 1`) tests against
#    MATLAB's stdout handle; the port checks for a missing/closed handle,
#    which is the intended semantics.
# 15. write2chanTiff T==1 (review F2). MATLAB ndims() drops TRAILING
#    singletons, so a [C,Y,X,1] stack (single-timepoint run) has ndims 3
#    and falls through to the disp-only else — NOTHING is written (an
#    upstream quirk that silently skips the `_mean_zproj.tif` deliverable
#    for T==1 runs). A [C,Y,X,Z,1] stack has ndims 4 and is written by the
#    4-D branch with frames=Nz, slices=1 (Z relabeled as T; page pixel
#    order identical). write2chan_tiff mirrors BOTH faithfully via
#    _matlab_stack_to_pages' trailing-singleton strip. To write T==1 data
#    intentionally, call write_zproj_tiff (new-design, dispatches on
#    explicit axes, no such quirk).
# 16. T-major .npy input (new-design, no .m counterpart). bfconvert writes
#     the OIR series as ZTCYX — z SLOWEST — so one get_volume() is Nz page
#     reads scattered over the whole file (82 seeks in 114 GB for the
#     production stack), and parallel workers then thrash the drive's
#     request queue (measured 2.4x on 10 processes). scripts/relayout.py
#     rewrites the stack once as a plain (T, Z, C, Y, X) uint16 .npy, in
#     which a volume is one contiguous byte range; VolumeSource accepts
#     that path via _init_npy and serves it through the SAME interface
#     (metadata/dtype/Nchan/Nx/Ny/Nz/Nt/get_volume/get_frame/__len__/
#     context manager), so pipeline.py, orchestrator._resolve_source and
#     pipeline._apply_io_adapter need no changes at all — they only ever
#     touch those attributes.
#     BIT-EXACTNESS (the replicate-path iron law): relayout copies pixels
#     verbatim (same dtype, no scaling, no inversion — see #3) and
#     get_volume does np.asarray(..., float64) + transpose on both paths.
#     No arithmetic touches the values, so uint16 -> float64 is exact and
#     the two paths agree bit for bit; cpstab/tests/test_tmajor.py asserts
#     that for every (t, c) of the 40-frame regression subset AND asserts
#     the full-pipeline output is np.array_equal to the TIFF-path run.
#     The .npy carries no axis metadata beyond its shape, hence the strict
#     ndim == 5 check and the refusal of any axes= override: a 5-D array
#     that is NOT (T, Z, C, Y, X) cannot be detected, so relayout.py is the
#     only sanctioned producer.
# 18. FLOAT32 FAST MODE (port extension, cpstab/precision.py). The float class
#     get_volume / get_frame hand out follows the process-wide compute dtype:
#     float64 by default (unchanged replicate path), float32 under
#     cfg.compute_dtype='float32'. This is a CONTAINER choice, not a value
#     change — the store is uint16 (<= 65535, 17 bits) and neither branch does
#     any arithmetic beyond the .sbx 65535-x inversion, which is exact in
#     float32 too, so both classes hold the same numbers exactly. It matters
#     because get_volume is the single largest allocation in the apply stage
#     (one [Y, X, Z] volume per channel per timepoint) and everything
#     downstream inherits its dtype.
#     Deliberately NOT affected: sbx_read / imread / RegWriter and the TIFF
#     writers, which are byte-contract code and stay uint16; and load_tiff,
#     which mirrors a MATLAB double() cast.
# 19. VolumeSource.read_block (new-design, no .m counterpart). #16 removed the
#     SEEKS; this addresses a whole run of volumes as one array instead of
#     Nz*C strided slices per volume. The T-major layout is what makes that
#     possible at all (in the Z-major TIFF a "run of volumes" is not a run of
#     bytes).
#     WHAT IT IS NOT: it is not a bulk READ. It returns a memmap VIEW, so the
#     pages are still faulted in on demand exactly as get_volume() faulted
#     them, and the earlier claim here — "a single sequential read the kernel
#     prefetches", with one worker's read overlapping the others' compute —
#     was never true of this code. Both halves were then measured (numbers in
#     read_block's docstring): making the copy real is inside the noise at 10
#     workers and a loss single-process, so the view is kept and the claim is
#     withdrawn rather than the code changed. What the method does still buy
#     is the flat loop shape in fast_run._apply_worker and one bounds check
#     per slab instead of per volume; treat --read-mb as loop granularity,
#     not as a memory budget that is actually being spent.
#     It returns None for non-.npy sources ON PURPOSE rather than emulating
#     the bulk read on TIFF: the caller's fallback is the ordinary
#     get_volume() path, so there is exactly one code path to trust, and
#     which one runs cannot change any value. Bit-exactness holds by
#     construction — the block is the stored integers verbatim, and slicing
#     block[j, :, c] then casting is literally what get_volume(t0+j, c) does
#     to self._npy[t, :, c] (cpstab/tests/test_tmajor.py test_7 asserts it
#     for every (t, c) of the regression subset, in both compute dtypes).
