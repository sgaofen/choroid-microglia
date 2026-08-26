# Probe dftregistration_alex directly with periodic (circular) shifts.
import numpy as np
import cpstab.dftreg as dreg

rng = np.random.default_rng(3)
base = rng.random((64, 64)) * 100
# smooth it periodically so subpixel interp is well-behaved
F = np.fft.fft2(base)
ky = np.fft.fftfreq(64)[:, None]
kx = np.fft.fftfreq(64)[None, :]
F *= np.exp(-((ky * 64) ** 2 + (kx * 64) ** 2) / (2 * 8 ** 2))
base = np.real(np.fft.ifft2(F))

def circshift_sub(img, dy, dx):
    # periodic subpixel shift via Fourier: content moves +dy, +dx
    F = np.fft.fft2(img)
    ph = np.exp(-2j * np.pi * (ky * dy * 64 / 64 * 1 + kx * dx))
    ph = np.exp(-2j * np.pi * (np.fft.fftfreq(64)[:, None] * dy + np.fft.fftfreq(64)[None, :] * dx))
    return np.real(np.fft.ifft2(F * ph))

print("integer roll cases (moving = roll(ref,(dy,dx))), expect (-dy,-dx):")
for dy, dx in [(3, -2), (-5, 7), (0, 1)]:
    mov = np.roll(base, (dy, dx), axis=(0, 1))
    out = dreg.dftregistration_alex(np.fft.fft2(base), np.fft.fft2(mov), 100)
    print(f"  dy,dx=({dy},{dx}) -> {out}")

print("subpixel periodic cases, expect (-dy,-dx):")
for dy, dx in [(1.25, -2.5), (-3.41, 2.48), (0.33, 0.67)]:
    mov = circshift_sub(base, dy, dx)
    out = dreg.dftregistration_alex(np.fft.fft2(base), np.fft.fft2(mov), 100)
    print(f"  dy,dx=({dy},{dx}) -> {out}")

print("usfac=4:")
for dy, dx in [(1.25, -2.5), (2.0, -1.0)]:
    mov = circshift_sub(base, dy, dx)
    out = dreg.dftregistration_alex(np.fft.fft2(base), np.fft.fft2(mov), 4)
    print(f"  dy,dx=({dy},{dx}) -> {out}")
