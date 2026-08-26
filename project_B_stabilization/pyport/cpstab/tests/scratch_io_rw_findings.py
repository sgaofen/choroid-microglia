"""Pre/post-fix probes for the io_rw.py review findings F1-F9.

--pre  : verify the PLAUSIBLE findings actually reproduce on this box
         (F3 NaN-through-clip UB) and document the pre-fix behaviors.
--post : assert every applied fix behaves per the .m ground truth.
"""
import os
import sys
import warnings
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

# through the PACKAGE: io_rw.py carries a relative import since the float32
# fast mode landed (`from .precision import ...`), so a flat import fails.
import cpstab.io_rw as io_rw  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "--post"
failures = []


def check(name, ok, detail=""):
    print("%-8s %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        failures.append(name)


def make_sbx(tmp, nchan=2, rows=4, cols=5, nz=1, nt=3):
    """Build a tiny valid .sbx + v5 sidecar; return (path, original_data).

    original_data has MATLAB dims [C, rows, cols, F] (pre-inversion values).
    """
    rng = np.random.RandomState(0)
    data = rng.randint(0, 65535, size=(nchan, rows, cols, nz * nt)).astype(np.uint16)
    info = io_rw.spoof_sbx_info_3d(rows, cols, nz, nt, nchan)  # MakeSBXall order
    base = os.path.join(tmp, "m_000_001")
    io_rw.save_sbx_info(base + ".mat", info)
    rw = io_rw.RegWriter(base + ".sbx", info, ".sbx", True)
    rw.write(data)
    rw.delete()
    return base + ".sbx", data


# ---------------------------------------------------------------- F3 probe
def probe_f3_pre():
    x = np.array([1.5, np.nan, 7.0])
    c = np.clip(x, 0, 65535)
    check("F3.clip-keeps-nan", np.isnan(c[1]),
          "np.clip(NaN)=%r (finding requires NaN to survive clip)" % c[1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = c.astype(np.uint16)
        warned = any("invalid value" in str(x.message) for x in w)
    check("F3.cast-is-ub", warned or v[1] != 0,
          "NaN->uint16 gave %r, RuntimeWarning=%r (UB either way)" % (v[1], warned))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = io_rw.matlab_uint16(np.array([np.nan]))
        warned = any("invalid value" in str(x.message) for x in w)
    check("F3.matlab_uint16-pre", warned,
          "matlab_uint16(NaN)=%r with warning=%r" % (out[0], warned))


def probe_f3_post():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning -> failure
        out = io_rw.matlab_uint16(np.array([np.nan, -np.inf, np.inf, 1.5, -3.0]))
    ok = (out == np.array([0, 0, 65535, 2, 0], dtype=np.uint16)).all()
    check("F3.matlab_uint16-post", ok, "got %r" % (out,))
    # RegWriter path: NaN pixels must land as raw 65535 (65535 - uint16(NaN)=0)
    with tempfile.TemporaryDirectory() as tmp:
        info = io_rw.spoof_sbx_info_3d(2, 2, 1, 1, 1)
        base = os.path.join(tmp, "n_000_001")
        io_rw.save_sbx_info(base + ".mat", info)
        rw = io_rw.RegWriter(base + ".sbx", info, ".sbx", True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rw.write(np.array([[np.nan, 1.0], [2.0, 3.0]]))
        rw.delete()
        got = io_rw.sbx_read(base + ".sbx", 1, 1, 1)
        check("F3.regwriter-nan", got[0, 0] == 0 and got[0, 1] == 1,
              "read back %r" % (got,))


# ---------------------------------------------------------------- F1 post
def probe_f1_post():
    with tempfile.TemporaryDirectory() as tmp:
        path, data = make_sbx(tmp, nchan=2)
        # pmt=0 must raise (MATLAB: x(0,...) errors)
        for bad in (0, 0.5, -0.5, 1.5):
            try:
                io_rw.sbx_read(path, 1, 1, bad)
                check("F1.sbx_read-pmt=%r" % (bad,), False, "no error raised")
            except IndexError:
                check("F1.sbx_read-pmt=%r" % (bad,), True)
        # valid pmt values still work and select the right channel
        g1 = io_rw.sbx_read(path, 1, 1, 1)
        g2 = io_rw.sbx_read(path, 1, 1, 2)
        check("F1.sbx_read-pmt-valid",
              (g1 == data[0, :, :, 0]).all() and (g2 == data[1, :, :, 0]).all())
        # pmt <= -1 returns both channels (MATLAB: no correction branch)
        gall = io_rw.sbx_read(path, 1, 1, -1)
        check("F1.sbx_read-pmt=-1", gall.shape == (2, 4, 5)
              and (gall == data[:, :, :, 0]).all())
        # SbxFile.read same contract
        sf = io_rw.SbxFile(path)
        try:
            sf.read(1, 1, 0)
            check("F1.SbxFile-pmt=0", False, "no error raised")
        except IndexError:
            check("F1.SbxFile-pmt=0", True)
        check("F1.SbxFile-pmt-valid",
              (sf.read(1, 1, 2) == data[1, :, :, 0]).all())
        sf.close()


# ---------------------------------------------------------------- F2 post
def probe_f2_post():
    with tempfile.TemporaryDirectory() as tmp:
        # 4D, T==1  ->  MATLAB ndims==3 -> disp branch, NO file written
        p = os.path.join(tmp, "a.tif")
        io_rw.write2chan_tiff(np.zeros((2, 4, 5, 1), np.uint16), p)
        check("F2.4d-T1-nowrite", not os.path.exists(p))
        # 3D -> disp branch in both
        io_rw.write2chan_tiff(np.zeros((4, 5, 3), np.uint16), p)
        check("F2.3d-nowrite", not os.path.exists(p))
        # 5D, T==1 -> MATLAB ndims==4 -> 4D branch: frames=Nz, slices=1
        import tifffile
        mov = np.arange(2 * 4 * 5 * 3, dtype=np.uint16).reshape(2, 4, 5, 3, 1)
        io_rw.write2chan_tiff(mov, p)
        with tifffile.TiffFile(p) as tf:
            ij = tf.imagej_metadata
            ok = (ij.get("frames", 1) == 3 and ij.get("slices", 1) == 1
                  and ij.get("channels", 1) == 2)
            arr = tf.asarray()
        check("F2.5d-T1-as-4d", os.path.exists(p) and ok,
              "imagej meta %r" % {k: ij.get(k) for k in ("frames", "slices", "channels")})
        # page pixel order unchanged: c fastest then z(->t)
        pages = arr.reshape(-1, 4, 5)
        want = np.transpose(mov[:, :, :, :, 0], (3, 0, 1, 2)).reshape(-1, 4, 5)
        check("F2.5d-T1-pages", (pages == want).all())
        # normal 4D and 5D still written identically to before
        p2 = os.path.join(tmp, "b.tif")
        mov4 = np.arange(2 * 4 * 5 * 3, dtype=np.uint16).reshape(2, 4, 5, 3)
        io_rw.write2chan_tiff(mov4, p2)
        with tifffile.TiffFile(p2) as tf:
            ij = tf.imagej_metadata
            arr = tf.asarray()
        check("F2.4d-normal", ij.get("frames", 1) == 3 and ij.get("slices", 1) == 1
              and (arr.reshape(-1, 4, 5)
                   == np.transpose(mov4, (3, 0, 1, 2)).reshape(-1, 4, 5)).all())
        p3 = os.path.join(tmp, "c.tif")
        mov5 = np.arange(2 * 4 * 5 * 3 * 2, dtype=np.uint16).reshape(2, 4, 5, 3, 2)
        io_rw.write2chan_tiff(mov5, p3)
        with tifffile.TiffFile(p3) as tf:
            ij = tf.imagej_metadata
            arr = tf.asarray()
        want5 = np.transpose(mov5, (4, 3, 0, 1, 2)).reshape(-1, 4, 5)
        check("F2.5d-normal", ij.get("frames") == 2 and ij.get("slices") == 3
              and (arr.reshape(-1, 4, 5) == want5).all())


# ---------------------------------------------------------------- F4 post
def probe_f4_post():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = make_sbx(tmp)
        i1 = io_rw.sbx_info(path)
        i1["nframes"] = 99999
        i2 = io_rw.sbx_info(path)
        check("F4.cache-immune", i2["nframes"] == 3, "got %r" % i2["nframes"])
        check("F4.fid-shared", i1["fid"] is i2["fid"])
        # RegWriter must copy info too
        info = io_rw.spoof_sbx_info_3d(4, 5, 1, 3, 2)
        rw = io_rw.RegWriter(os.path.join(tmp, "z_000_001.sbx"), info, ".sbx", True)
        rw.info["nframes"] = 7
        check("F4.regwriter-copy", info["nframes"] == 3)
        rw.delete()


# ---------------------------------------------------------------- F5 post
def probe_f5_post():
    with tempfile.TemporaryDirectory() as tmp:
        # no sidecar, no .tif.frames dir -> descriptive IOError, not IndexError
        try:
            io_rw.get_dimensions(os.path.join(tmp, "gone.sbx"), tmp, "gone")
            check("F5.raises", False, "no error")
        except IndexError as e:
            check("F5.raises", False, "still bare IndexError: %s" % e)
        except IOError as e:
            msg = str(e)
            check("F5.raises", "SbxFile" in msg and ".tif.frames" in msg,
                  "IOError text: %s" % msg[:120])
        # the TIFF-scan fallback still works when frames exist
        import tifffile
        fdir = os.path.join(tmp, "d")
        tifdir = os.path.join(fdir, "fb.tif.frames")
        os.makedirs(tifdir)
        for c in (1, 2):
            for z in (1, 2):
                for t in (1, 2):
                    tifffile.imwrite(
                        os.path.join(tifdir, "fb_C%03dZ%03dT%03d.tif" % (c, z, t)),
                        np.zeros((4, 5), np.uint16))
        dims = io_rw.get_dimensions(None, fdir, "fb")
        check("F5.fallback-ok", dims == (2, 4, 5, 2, 2), "got %r" % (dims,))


# ---------------------------------------------------------------- F7 post
def probe_f7_post():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = make_sbx(tmp, nchan=2, rows=4, cols=5, nz=1, nt=3)
        # (a) k one past the end -> N==0 -> MATLAB squeeze(x(pmt,:,:)) = [rows, 0]
        out = io_rw.sbx_read(path, 4, -1, 1)
        check("F7a.sbx_read-N0", out.shape == (4, 0), "shape %r" % (out.shape,))
        sf = io_rw.SbxFile(path)
        out = sf.read(4, -1, 1)
        check("F7a.SbxFile-N0", out.shape == (4, 0), "shape %r" % (out.shape,))
        # pmt=-1, N==0: both keep the 4-D empty (MATLAB no squeeze branch)
        out = io_rw.sbx_read(path, 4, -1, -1)
        check("F7a.sbx_read-N0-pmt-1", out.shape == (2, 4, 5, 0),
              "shape %r" % (out.shape,))
        # (b) k beyond the end -> both must raise like MATLAB's fread catch
        try:
            io_rw.sbx_read(path, 5, -1, 1)
            check("F7b.sbx_read-oob", False, "no error")
        except ValueError:
            check("F7b.sbx_read-oob", True)
        try:
            sf.read(5, -1, 1)
            check("F7b.SbxFile-oob", False, "no error (was empty array)")
        except ValueError:
            check("F7b.SbxFile-oob", True)
        # equivalence on normal reads is untouched
        check("F7.normal-eq",
              (io_rw.sbx_read(path, 2, 2, -1) == sf.read(2, 2, -1)).all())
        sf.close()


# ---------------------------------------------------------------- F8 post
def probe_f8_post():
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        # single-page RGB must raise (MATLAB: img(:,:,i) = [h,w,3] errors)
        p = os.path.join(tmp, "rgb.tif")
        tifffile.imwrite(p, np.zeros((4, 5, 3), np.uint8), photometric="rgb")
        try:
            io_rw.load_tiff(p)
            check("F8.rgb-raises", False, "no error")
        except ValueError:
            check("F8.rgb-raises", True)
        # grayscale single- and multi-page still fine
        p2 = os.path.join(tmp, "g1.tif")
        tifffile.imwrite(p2, np.arange(20, dtype=np.uint16).reshape(4, 5))
        check("F8.gray-1p", io_rw.load_tiff(p2).shape == (4, 5))
        p3 = os.path.join(tmp, "g3.tif")
        tifffile.imwrite(p3, np.arange(60, dtype=np.uint16).reshape(3, 4, 5),
                         photometric="minisblack")
        out = io_rw.load_tiff(p3)
        check("F8.gray-3p", out.shape == (4, 5, 3)
              and (out[:, :, 1] == np.arange(20, 40).reshape(4, 5)).all())


# ---------------------------------------------------------------- F9 post
def probe_f9_post():
    with tempfile.TemporaryDirectory() as tmp:
        path, data = make_sbx(tmp, nchan=2, rows=4, cols=5, nz=1, nt=3)
        # 0-d array N=2 must read 2 frames (MATLAB str: isempty(N) false)
        out = io_rw.imread(path, 1, np.array(2), 1)
        check("F9.0d-N", out.shape == (4, 5, 2), "shape %r" % (out.shape,))
        out = io_rw.imread(path, 1, np.array([2]), 1)
        check("F9.1elem-N", out.shape == (4, 5, 2), "shape %r" % (out.shape,))
        # empty N -> whole file (MATLAB isempty -> -1)
        out = io_rw.imread(path, 1, np.array([]), 1)
        check("F9.empty-N", out.shape == (4, 5, 3), "shape %r" % (out.shape,))
        out = io_rw.imread(path, 1, -1, 1)
        check("F9.neg-N", out.shape == (4, 5, 3), "shape %r" % (out.shape,))
        out = io_rw.imread(path, 1, 2, 1)
        check("F9.int-N", out.shape == (4, 5, 2), "shape %r" % (out.shape,))


if MODE == "--pre":
    probe_f3_pre()
else:
    probe_f3_post()
    probe_f1_post()
    probe_f2_post()
    probe_f4_post()
    probe_f5_post()
    probe_f7_post()
    probe_f8_post()
    probe_f9_post()

print("\n%d failures" % len(failures))
sys.exit(1 if failures else 0)
