"""F1 verification: double-rounding edge in matlab_uint16 / matlab_round.

Review claim: `np.floor(np.abs(x) + 0.5)` double-rounds for doubles just
below a half-integer, because abs(x)+0.5 is IEEE-rounded UP to the integer
before floor. Canonical case: x = nextafter(0.5, 0) = 0.49999999999999994.
Its exact real value is < 0.5, so correct round-half-away-from-zero -> 0,
but abs(x)+0.5 = 1-2^-54 rounds (tie-to-even) to 1.0 -> floor -> 1.

Oracle: exact rational arithmetic (fractions.Fraction over the double's
exact value) implementing round-half-away-from-zero, i.e. the C99 round()
semantics MATLAB's uint16()/round() use.

Run: python scratch_f1_rounding.py
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # pyport/

from cpstab.pipeline import matlab_uint16
from cpstab.config import matlab_round

FAIL = 0


def check(name, got, want):
    global FAIL
    ok = got == want
    print("%-58s got=%-8r want=%-8r %s" % (name, got, want, "PASS" if ok else "FAIL"))
    if not ok:
        FAIL += 1


def oracle_round_half_away(x):
    """Exact round-half-away-from-zero of the double's exact value."""
    if np.isnan(x):
        return 0  # only used via uint16 oracle
    f = Fraction(float(x))
    a = abs(f)
    n = a.numerator // a.denominator
    frac = a - n
    if frac >= Fraction(1, 2):
        n += 1
    return n if f >= 0 else -n


def oracle_uint16(x):
    if np.isnan(x):
        return 0
    if np.isposinf(x):
        return 65535
    if np.isneginf(x):
        return 0
    r = oracle_round_half_away(x)
    return min(max(r, 0), 65535)


# --- 1. The review's canonical edge case ------------------------------------
e = np.nextafter(0.5, 0.0)  # 0.49999999999999994, exact value < 1/2
print("edge value: %.17g  (exact < 0.5: %s)" % (e, Fraction(e) < Fraction(1, 2)))
print()
print("== current implementations on the edge case ==")
check("matlab_uint16(nextafter(0.5,0)) [oracle]", int(matlab_uint16(e)), oracle_uint16(e))
check("matlab_round(nextafter(0.5,0))  [oracle]", matlab_round(e), oracle_round_half_away(e))
check("matlab_round(-nextafter(0.5,0)) [oracle]", matlab_round(-e), oracle_round_half_away(-e))

# --- 2. Reachability shape: does the edge exist at other magnitudes? --------
# k + (0.5 - ulp) is only representable (distinct from k+0.5) for small k;
# still, sweep every half-integer neighborhood representable below 2^52.
print()
print("== sweep: nextafter(k+0.5, k) for k in 0..70000 (uint16 path) ==")
bad_cur, bad_at = 0, []
for k in list(range(0, 2049)) + [4095, 8191, 16383, 32767, 65534, 65535, 70000]:
    v = np.nextafter(k + 0.5, k)
    if v == k + 0.5:  # not representable distinctly; no edge exists here
        continue
    want = oracle_uint16(v)
    got = int(matlab_uint16(v))
    if got != want:
        bad_cur += 1
        if len(bad_at) < 8:
            bad_at.append((k, v, got, want))
print("current matlab_uint16 deviates from oracle at %d half-integer edges" % bad_cur)
for k, v, got, want in bad_at:
    print("   k=%-6d x=%.17g got=%d want=%d" % (k, v, got, want))

# --- 3. Review's fixed algorithm: exact-fraction comparison -----------------
def fixed_uint16(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isnan(x), 0.0, x)
    a = np.minimum(np.abs(x), 65536.0)  # kills Inf; >=65535.5 saturates anyway
    f = np.floor(a)
    r = np.where(a - f >= 0.5, f + 1.0, f)  # a-f exact (Sterbenz) -> no double round
    r = np.sign(x) * r
    return np.clip(r, 0.0, 65535.0).astype(np.uint16)


def fixed_round(x):
    import math
    a = abs(x)
    f = math.floor(a)
    if a - f >= 0.5:
        f += 1
    return int(f) if x >= 0 else -int(f)


print()
print("== fixed algorithms vs oracle ==")
# main-semantics vector from the review (must stay identical)
vec = [2.5, -3.7, 70000.0, np.nan, 0.5, 1.5, -0.4, 65534.5, np.inf, -np.inf]
want_vec = [3, 0, 65535, 0, 1, 2, 0, 65535, 65535, 0]
check("fixed_uint16(main vector)", fixed_uint16(vec).tolist(), want_vec)
check("current matlab_uint16(main vector)", matlab_uint16(vec).tolist(), want_vec)
check("fixed_uint16(edge)", int(fixed_uint16(e)), oracle_uint16(e))
check("fixed_round(edge)", fixed_round(e), oracle_round_half_away(e))
check("fixed_round(-edge)", fixed_round(-e), oracle_round_half_away(-e))

bad_fix = 0
for k in list(range(0, 2049)) + [4095, 8191, 16383, 32767, 65534, 65535, 70000]:
    for v in (np.nextafter(k + 0.5, k), k + 0.5, np.nextafter(k + 0.5, k + 1), float(k)):
        if int(fixed_uint16(v)) != oracle_uint16(v):
            bad_fix += 1
print("fixed_uint16 deviates from oracle at %d sweep points" % bad_fix)
if bad_fix:
    FAIL += 1

# fixed_round parity sweep incl. negatives and the review-verified round cases
bad_r = 0
rng = np.random.default_rng(0)
pts = list(rng.uniform(-1e6, 1e6, 20000)) + [7.5, 22.5, -2.5, 10.5, 0.0, -0.5]
for k in range(-2049, 2049):
    pts += [k + 0.5, np.nextafter(k + 0.5, k), np.nextafter(k + 0.5, k + 1)]
for v in pts:
    if fixed_round(float(v)) != oracle_round_half_away(float(v)):
        bad_r += 1
print("fixed_round deviates from oracle at %d of %d points" % (bad_r, len(pts)))
if bad_r:
    FAIL += 1

# Nchunks / proj-range call sites must be byte-identical old vs new
bad_call = 0
for nt in range(1, 3000):
    for cs in range(1, 21):
        if fixed_round(nt / cs) != matlab_round(nt / cs):
            bad_call += 1
for nz in range(1, 129):
    for q in (0.25, 0.75):
        if fixed_round(q * nz) != matlab_round(q * nz):
            bad_call += 1
print("call-site parity old-vs-new (Nchunks + quarter bounds): %d diffs" % bad_call)
if bad_call:
    FAIL += 1

# fixed_uint16 must not emit warnings on inf
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")
    fixed_uint16([np.inf, -np.inf, np.nan, 1e300, -1e300])
print("fixed_uint16 inf/nan path: no warnings")

print()
print("RESULT: %s (%d failures)" % ("ALL PASS" if FAIL == 0 else "FAILURES", FAIL))
sys.exit(1 if FAIL else 0)
