# Differential test: independent transliteration of the 5 .m files vs shifts2d port.
# Independent primitives written fresh from the MATLAB sources (incl. dftregistrationAlex).
import math
import numpy as np

import cpstab.dftreg as dreg
if not hasattr(dreg, "dftregistration"):
    dreg.dftregistration = dreg.dftregistration_alex
import cpstab.shifts2d as s2

rng = np.random.default_rng(42)
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


# ---------------- independent MATLAB primitives ----------------
def m_cast(x, dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        r = np.sign(x) * np.floor(np.abs(x) + 0.5)
        info = np.iinfo(dtype)
        return np.clip(r, info.min, info.max).astype(dtype)
    return x.astype(dtype)


def ind_imtranslate(img, tvec):  # MATLAB imtranslate(A,[tx ty]),'linear',Fill 0
    tx, ty = float(tvec[0]), float(tvec[1])
    a = np.asarray(img)
    af = a.astype(float)
    H, W = af.shape
    pad = int(max(abs(tx), abs(ty))) + 2
    ap = np.zeros((H + 2 * pad, W + 2 * pad))
    ap[pad:pad + H, pad:pad + W] = af
    yy = np.arange(H)[:, None] - ty + pad
    xx = np.arange(W)[None, :] - tx + pad
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    fy = yy - y0; fx = xx - x0
    out = (ap[y0, x0] * (1 - fy) * (1 - fx) + ap[y0 + 1, x0] * fy * (1 - fx)
           + ap[y0, x0 + 1] * (1 - fy) * fx + ap[y0 + 1, x0 + 1] * fy * fx)
    return m_cast(out, a.dtype)


def ind_imgaussfilt(a, sigma):
    a = np.asarray(a)
    r = math.ceil(2 * sigma)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()
    ap = np.pad(a.astype(float), r, mode="edge")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, ap)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, tmp)
    return m_cast(out, a.dtype)


def ind_ifftshift_range(lo, hi):  # ifftshift(lo:hi) as vector
    return np.fft.ifftshift(np.arange(lo, hi + 1))


def ind_find_first_colmajor(mask):
    idx = np.argmax(mask.flatten(order="F"))  # first True in column-major
    return np.unravel_index(idx, mask.shape, order="F")


def ind_ftpad(imFT, outsize):  # FTpad, L170-L203
    Nout = np.array(outsize); Nin = np.array(imFT.shape)
    imFT = np.fft.fftshift(imFT)
    center = np.floor(Nin / 2).astype(int) + 1
    imFTout = np.zeros(tuple(Nout), dtype=complex)
    centerout = np.floor(Nout / 2).astype(int) + 1
    cc = centerout - center
    imFTout[max(cc[0] + 1, 1) - 1:min(cc[0] + Nin[0], Nout[0]),
            max(cc[1] + 1, 1) - 1:min(cc[1] + Nin[1], Nout[1])] = \
        imFT[max(-cc[0] + 1, 1) - 1:min(-cc[0] + Nout[0], Nin[0]),
             max(-cc[1] + 1, 1) - 1:min(-cc[1] + Nout[1], Nin[1])]
    return np.fft.ifftshift(imFTout) * Nout[0] * Nout[1] / (Nin[0] * Nin[1])


def ind_dftups(in_, nor, noc, usfac, roff, coff):  # dftups, L131-L166
    nr, nc = in_.shape
    kernc = np.exp((-1j * 2 * np.pi / (nc * usfac))
                   * (np.fft.ifftshift(np.arange(nc))[:, None] - math.floor(nc / 2))
                   * (np.arange(noc)[None, :] - coff))
    kernr = np.exp((-1j * 2 * np.pi / (nr * usfac))
                   * (np.arange(nor)[:, None] - roff)
                   * (np.fft.ifftshift(np.arange(nr))[None, :] - math.floor(nr / 2)))
    return kernr @ in_ @ kernc


def matlab_round(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def ind_dftreg_alex(buf1ft, buf2ft, usfac):  # dftregistrationAlex L68-L128
    nr, nc = buf2ft.shape
    if usfac == 1:
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        CCabs = np.abs(CC)
        r, c = ind_find_first_colmajor(CCabs == CCabs.max())
        Nr = ind_ifftshift_range(-(nr // 2) if nr % 2 == 0 else -(nr // 2), 0)  # unused path
        raise NotImplementedError
    CC = np.fft.ifft2(ind_ftpad(buf1ft * np.conj(buf2ft), [2 * nr, 2 * nc]))
    CCabs = np.abs(CC)
    r, c = ind_find_first_colmajor(CCabs == CCabs.max())
    Nr2 = np.fft.ifftshift(np.arange(-int(np.fix(nr)), math.ceil(nr)))
    Nc2 = np.fft.ifftshift(np.arange(-int(np.fix(nc)), math.ceil(nc)))
    row_shift = Nr2[r] / 2
    col_shift = Nc2[c] / 2
    if usfac > 2:
        row_shift = matlab_round(row_shift * usfac) / usfac
        col_shift = matlab_round(col_shift * usfac) / usfac
        dftshift = int(np.fix(math.ceil(usfac * 1.5) / 2))
        CC = np.conj(ind_dftups(buf2ft * np.conj(buf1ft),
                                math.ceil(usfac * 1.5), math.ceil(usfac * 1.5), usfac,
                                dftshift - row_shift * usfac, dftshift - col_shift * usfac))
        CCabs = np.abs(CC)
        rloc, cloc = ind_find_first_colmajor(CCabs == CCabs.max())
        rloc = (rloc + 1) - dftshift - 1  # 1-based find -> back to offsets
        cloc = (cloc + 1) - dftshift - 1
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac
    if nr == 1:
        row_shift = 0.0
    if nc == 1:
        col_shift = 0.0
    return np.array([row_shift, col_shift], dtype=float)


# ---------------- independent transliterations of the 5 files ----------------
def ind_DFT_rect(vol, start, upscale):
    Nz = vol.shape[2]
    reg = np.zeros(vol.shape)
    target = vol[:, :, start - 1]
    R = np.zeros(Nz); C = np.zeros(Nz)
    for i in range(start, Nz + 1):
        source = vol[:, :, i - 1]
        S = ind_dftreg_alex(np.fft.fft2(target), np.fft.fft2(source), upscale)
        R[i - 1] = S[0]; C[i - 1] = S[1]
        target = ind_imtranslate(source, [S[1], S[0]])
        reg[:, :, i - 1] = target
    target = vol[:, :, start - 1]
    for i in range(start, 0, -1):
        source = vol[:, :, i - 1]
        S = ind_dftreg_alex(np.fft.fft2(target), np.fft.fft2(source), upscale)
        R[i - 1] = S[0]; C[i - 1] = S[1]
        target = ind_imtranslate(source, [S[1], S[0]])
        reg[:, :, i - 1] = target
    return R, C, reg


def ind_DFT_reg(stack, target, upscale):
    N = stack.shape[2]
    reg = np.zeros(stack.shape)
    R = np.zeros(N); C = np.zeros(N)
    for i in range(1, N + 1):
        source = stack[:, :, i - 1]
        S = ind_dftreg_alex(np.fft.fft2(target), np.fft.fft2(source), upscale)
        R[i - 1] = S[0]; C[i - 1] = S[1]
        reg[:, :, i - 1] = ind_imtranslate(source, [S[1], S[0]])
    return R, C, reg


def ind_Determine(full_vol, Blur, Keep, RefVol):
    S1, S2, S3, S4 = full_vol.shape
    rlo = math.ceil(S1 * (1 - Keep) / 2); rhi = math.ceil(S1 * (1 - (1 - Keep) / 2))
    clo = math.ceil(S2 * (1 - Keep) / 2); chi = math.ceil(S2 * (1 - (1 - Keep) / 2))
    red = full_vol[rlo - 1:rhi, clo - 1:chi, :, :]
    chunck = math.floor(S4 / RefVol.shape[3])
    RefVol = RefVol[rlo - 1:rhi, clo - 1:chi, :, :]
    RS = np.zeros((S3, S4)); CS = np.zeros((S3, S4))
    for t in range(1, S4 + 1):
        reft = RefVol[:, :, :, math.ceil(t / chunck) - 1]
        for i in range(1, S3 + 1):
            ref = reft[:, :, i - 1]
            out = ind_dftreg_alex(np.fft.fft2(ind_imgaussfilt(ref, Blur)),
                                  np.fft.fft2(ind_imgaussfilt(red[:, :, i - 1, t - 1], Blur)), 100)
            RS[i - 1, t - 1] = out[0]; CS[i - 1, t - 1] = out[1]
    return RS, CS


def ind_Apply(vol, RS, CS):
    out = np.zeros(vol.shape)
    for t in range(vol.shape[3]):
        for i in range(vol.shape[2]):
            out[:, :, i, t] = ind_imtranslate(vol[:, :, i, t], [CS[i, t], RS[i, t]])
    return out


def ind_defineReference(volume, n, type_):
    x, y, z, t = volume.shape
    ref = np.zeros((x, y, z, t // n))
    for i in range(1, t // n + 1):
        for zz in range(1, z + 1):
            a = volume[:, :, zz - 1, (i - 1) * n:i * n]
            a = np.median(a, axis=2) if type_ == "median" else np.mean(a, axis=2)
            ref[:, :, zz - 1, i - 1] = a
    return ref


# ---------------- differential runs ----------------
def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


# D0: primitive-level fuzz of dftreg_alex vs port on adversarial banded/rolled inputs
base = ind_imgaussfilt(rng.random((64, 63)) * 100, 2.0)
adv = base.copy(); adv[0, :] = 0; adv[:, -1] = 0
cases = []
for dy, dx in [(-2, 1), (3, -2), (0, 0), (-31, 30)]:
    mov = np.roll(base, (dy, dx), axis=(0, 1))
    cases.append((adv, mov))
    cases.append((base, mov))
ok = True
for usfac in [2, 4, 100]:
    for f, g in cases:
        a = dreg.dftregistration_alex(np.fft.fft2(f), np.fft.fft2(g), usfac)
        b = ind_dftreg_alex(np.fft.fft2(f), np.fft.fft2(g), usfac)
        if maxdiff(a, b) > 1e-9:
            ok = False
            print(f"  D0 usfac={usfac}: port {a} vs indep {b}")
check("D0 dftreg_alex == independent transliteration (banded inputs)", ok)

# D1: DFT_rect differential, float64 and uint16, odd dims, edge starts
volf = rng.random((32, 33, 5)) * 1000
volu = (rng.random((32, 33, 5)) * 60000).astype(np.uint16)
ok = True
for vol in (volf, volu):
    for start in (1, 3, 5):
        for ups in (4, 100):
            R1, C1, G1 = s2.dft_rect(vol, start, ups)
            R2, C2, G2 = ind_DFT_rect(vol, start, ups)
            d = max(maxdiff(R1, R2), maxdiff(C1, C2), maxdiff(G1, G2))
            if d > 1e-9:
                ok = False
                print(f"  D1 dtype={vol.dtype} start={start} ups={ups} maxdiff={d:.3e}")
                print(f"     R port={R1}\n     R ind ={R2}")
check("D1 dft_rect differential", ok)

# D2: DFT_reg differential
tgt = rng.random((32, 33)) * 1000
R1, C1, G1 = s2.dft_reg(volf, tgt, 100)
R2, C2, G2 = ind_DFT_reg(volf, tgt, 100)
check("D2 dft_reg differential", max(maxdiff(R1, R2), maxdiff(C1, C2), maxdiff(G1, G2)) < 1e-9)

# D3: Determine differential (float64 and uint16, keep=0.95/0.9, T=4, refT=2)
fv_f = rng.random((40, 41, 2, 4)) * 1000
fv_u = (rng.random((40, 41, 2, 4)) * 60000).astype(np.uint16)
ok = True
for fv in (fv_f, fv_u):
    refv = ind_defineReference(fv.astype(float), 2, "mean")  # double ref, like pipeline
    for keep in (0.95, 0.9):
        RS1, CS1 = s2.determine_xy_shifts_fbs(fv, 1.0, keep, refv)
        RS2, CS2 = ind_Determine(fv, 1.0, keep, refv)
        d = max(maxdiff(RS1, RS2), maxdiff(CS1, CS2))
        if d > 1e-9:
            ok = False
            print(f"  D3 dtype={fv.dtype} keep={keep} maxdiff={d:.3e}")
            print(f"     RS port={RS1.ravel()}\n     RS ind ={RS2.ravel()}")
check("D3 determine_xy_shifts_fbs differential", ok)

# D3b: uint16 ReferenceVolume as well (MATLAB header: 'uint')
refu = ind_defineReference(fv_u.astype(float), 2, "mean").astype(np.uint16)
RS1, CS1 = s2.determine_xy_shifts_fbs(fv_u, 1.0, 0.95, refu)
RS2, CS2 = ind_Determine(fv_u, 1.0, 0.95, refu)
check("D3b determine uint16 ref differential", max(maxdiff(RS1, RS2), maxdiff(CS1, CS2)) < 1e-9)

# D4: Apply differential (fractional shifts, uint16 volume)
RSx = rng.uniform(-3, 3, (2, 4)); CSx = rng.uniform(-3, 3, (2, 4))
A1 = s2.apply_xy_shifts_fbs(fv_u, RSx, CSx)
A2 = ind_Apply(fv_u, RSx, CSx)
check("D4 apply_xy_shifts_fbs differential (uint16)", maxdiff(A1, A2) < 1e-9)
A1 = s2.apply_xy_shifts_fbs(fv_f, RSx, CSx)
A2 = ind_Apply(fv_f, RSx, CSx)
check("D4b apply_xy_shifts_fbs differential (float64)", maxdiff(A1, A2) < 1e-9)

# D5: defineReference differential
for typ in ("mean", "median"):
    r1 = s2.define_reference(fv_f, 2, typ)
    r2 = ind_defineReference(fv_f, 2, typ)
    check(f"D5 define_reference {typ}", maxdiff(r1, r2) == 0.0)

print()
print("FAILURES:", fails if fails else "none")
