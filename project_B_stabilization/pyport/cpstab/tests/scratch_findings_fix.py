"""Pre/post-fix probes for review findings F1/F3/F4 on dftreg.py.

Run with --pre  (before the fix): documents the port's silent-return behavior.
Run with --post (after the fix): asserts the new guards + F3 relaxation
bit-match the 1-based MATLAB reference in scratch_dftreg.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# through the PACKAGE: dftreg.py carries a relative import since the float32
# fast mode landed (`from .precision import ...`), so a flat import fails.
import cpstab.dftreg as dftreg  # noqa: E402
from cpstab.dftreg import dftregistration_alex, dftregistration3d  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "--post"
failures = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("%s  %s %s" % (tag, name, detail))
    if not cond:
        failures.append(name)


def raises_valueerror(fn):
    try:
        out = fn()
    except ValueError as e:
        return True, str(e)
    return False, repr(out)


rng = np.random.RandomState(0)

# --------------------------------------------------------------- F1: np_==1
vol = rng.rand(6, 6, 1)
ref = np.roll(vol, (1, 2, 0), axis=(0, 1, 2))
f1_call = lambda: dftregistration3d(np.fft.fftn(ref), np.fft.fftn(vol), 2)
if MODE == "--pre":
    print("F1 pre-fix: (6,6,1) usfac=2 ->", f1_call())
else:
    ok, msg = raises_valueerror(f1_call)
    check("F1 np_==1 raises", ok, msg[:90])
    # output plane count 1 via fractional usfac (np_=2, usfac=0.5):
    # MATLAB zeros([m,n,1]) is 2-D -> same FTpad crash; must raise too.
    v2 = rng.rand(4, 4, 2)
    ok, msg = raises_valueerror(
        lambda: dftregistration3d(np.fft.fftn(v2), np.fft.fftn(v2), 0.5))
    check("F1 out-plane==1 raises", ok, msg[:90])

# --------------------------------------------- F4: NaN -> silent vs raising
# numpy argmax property the guard relies on: NaN wins argmax.
check("argmax lands on NaN", int(np.argmax(np.array([5.0, np.nan, 3.0]))) == 1)

img = rng.rand(8, 8)
bad = img.copy()
bad[3, 4] = np.nan
F = np.fft.fft2(img)
G = np.fft.fft2(bad)  # NaN spreads through fft2 -> CC all-NaN
volb = rng.rand(8, 8, 4)
volb[1, 1, 1] = np.nan
cases = [
    ("usfac=1", lambda: dftregistration_alex(F, G, 1)),
    ("usfac=2", lambda: dftregistration_alex(F, G, 2)),
    ("usfac=4", lambda: dftregistration_alex(F, G, 4)),
    ("3d", lambda: dftregistration3d(np.fft.fftn(volb),
                                     np.fft.fftn(volb), 2)),
]
for name, fn in cases:
    if MODE == "--pre":
        with np.errstate(invalid="ignore"):
            print("F4 pre-fix %-8s ->" % name, fn())
    else:
        with np.errstate(invalid="ignore"):
            ok, msg = raises_valueerror(fn)
        check("F4 NaN raises (%s)" % name, ok, msg[:80])

if MODE == "--post":
    # usfac=0 never touches the data: NaN input must still return [0,0]
    # (bit-faithful to MATLAB L76-L79).
    check("F4 usfac=0 untouched",
          np.all(dftregistration_alex(F, G, 0) == 0.0))
    # finite inputs unaffected
    sh = dftregistration_alex(np.fft.fft2(img),
                              np.fft.fft2(np.roll(img, (2, -1), (0, 1))), 1)
    check("finite path intact", np.array_equal(sh, [-2.0, 1.0]), repr(sh))

# ------------------------------- F3: fractional usfac w/ integer products
if MODE == "--post":
    from scratch_dftreg import m_dftreg_3d  # 1-based MATLAB reference

    ok_all = True
    for shape in [(4, 4, 2), (6, 4, 4), (8, 6, 2)]:
        for usfac in [1.5, 2.5, 0.5, 3]:
            m = [d * usfac for d in shape]
            if any(x != int(x) for x in m) or int(m[2]) == 1:
                continue  # MATLAB errors on these; port raises (checked above)
            v = rng.rand(*shape)
            w = np.roll(v, (1, -1, 1), axis=(0, 1, 2))
            got = dftregistration3d(np.fft.fftn(v), np.fft.fftn(w), usfac)
            want = np.asarray(m_dftreg_3d(np.fft.fftn(v), np.fft.fftn(w),
                                          usfac), dtype=np.float64)
            same = np.array_equal(got, want)
            ok_all &= same
            if not same:
                print("   mismatch", shape, usfac, got, want)
    check("F3 fractional usfac bit-match vs MATLAB reference", ok_all)
    # integer usfac still bit-identical to the reference (regression)
    v = rng.rand(9, 7, 5)
    w = np.roll(v, (2, 1, -1), axis=(0, 1, 2))
    got = dftregistration3d(np.fft.fftn(v), np.fft.fftn(w), 2)
    want = np.asarray(m_dftreg_3d(np.fft.fftn(v), np.fft.fftn(w), 2),
                      dtype=np.float64)
    check("F3 integer usfac regression", np.array_equal(got, want),
          "%s vs %s" % (got, want))
    # reviewer's concrete probe: usfac must NOT be int()-truncated
    ok, msg = raises_valueerror(
        lambda: dftregistration3d(np.fft.fftn(rng.rand(5, 5, 3)),
                                  np.fft.fftn(rng.rand(5, 5, 3)), 1.5))
    check("F3 non-integer products still raise", ok, msg[:80])

print("\n%s: %d failures" % (MODE, len(failures)))
sys.exit(1 if failures else 0)
