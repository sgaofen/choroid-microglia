"""oirread.py — pure-Python reader for Olympus FluoView .oir (no Java).

Follows the block layout used by Bio-Formats OIRReader:
    pixel block = int32 checkLen | int32 3 | 8 bytes | int32 uidLen | uid
                  | int32 nbytes | 4 bytes | raw little-endian pixels
uid = z###t###_..._<channelId>_<block>   (block = horizontal strip index).
Multi-file runs: <name>.oir + <name>_00001, _00002, ... in the same folder.
Duplicate uids: the last one in file order wins (same as Bio-Formats).
"""
import mmap
import os
import re
import struct

import numpy as np

_HDR = re.compile(rb'(?s)(.{4})\x03\x00\x00\x00.{8}(.{4})([lzt]\d{3})')
_UID = re.compile(r'^(?:l(\d{3}))?(?:z(\d{3}))?(?:t(\d+))?_.*_([0-9a-fA-F-]+)_(\d+)$')


def run_files(path):
    """Master .oir plus its _00001.. companions, in order."""
    path = os.path.abspath(path)
    d, name = os.path.split(path)
    if name.lower().endswith('.oir'):
        prefix = name[:-4]
    else:
        prefix = name[:name.rfind('_')]
        cand = [f for f in os.listdir(d) if f.lower() == (prefix + '.oir').lower()]
        if not cand:
            raise FileNotFoundError('no master .oir for %s' % path)
        path = os.path.join(d, cand[0])
    extra = sorted(f for f in os.listdir(d)
                   if f.startswith(prefix + '_') and len(f) == len(prefix) + 6
                   and f[-5:].isdigit())
    return [path] + [os.path.join(d, f) for f in extra]


def _scan_file(fn, blocks, order):
    with open(fn, 'rb') as fh:
        if os.fstat(fh.fileno()).st_size < 64:
            return
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm[:16] != b'OLYMPUSRAWFORMAT':
                raise ValueError('%s is not an OIR file' % fn)
            n = len(mm)
            pos = 16
            while pos < n - 32:
                m = _HDR.search(mm, pos)
                if not m:
                    break
                off = m.start()
                chk = struct.unpack_from('<i', mm, off)[0]
                ulen = struct.unpack_from('<i', mm, off + 16)[0]
                if chk != ulen + 12 or not (8 <= ulen <= 512):
                    pos = off + 1
                    continue
                uid = mm[off + 20: off + 20 + ulen].decode('ascii', 'replace')
                um = _UID.match(uid)
                if not um:
                    pos = off + 1
                    continue
                nb = struct.unpack_from('<i', mm, off + 20 + ulen)[0]
                data = off + 28 + ulen
                if nb <= 0 or data + nb > n:
                    pos = off + 1
                    continue
                blocks[uid] = (fn, data, nb)
                order.append(uid)
                pos = data + nb
        finally:
            mm.close()


def _channel_ids(master):
    with open(master, 'rb') as fh:
        head = fh.read(8 << 20)
    ids = []
    for m in re.finditer(rb'<commonphase:channel[^>]*?id="([0-9a-fA-F-]+)"[^>]*?order="(\d+)"', head):
        cid, o = m.group(1).decode(), int(m.group(2))
        if cid not in [c for c, _ in ids]:
            ids.append((cid, o))
    ids.sort(key=lambda x: x[1])
    return [c for c, _ in ids]


def _xml_int(master, tag):
    with open(master, 'rb') as fh:
        head = fh.read(8 << 20)
    m = re.search(rb'<base:' + tag.encode() + rb'>(\d+)<', head)
    return int(m.group(1)) if m else None


class OIRRun:
    """Index of one acquisition. .shape = (T, Z, C, Y, X), dtype uint16."""

    def __init__(self, path, progress=None):
        self.files = run_files(path)
        blocks, order = {}, []
        for i, fn in enumerate(self.files):
            _scan_file(fn, blocks, order)
            if progress:
                progress(i + 1, len(self.files))
        if not blocks:
            raise ValueError('no pixel blocks found')
        chans = _channel_ids(self.files[0])
        nx = _xml_int(self.files[0], 'width')
        planes = {}
        for uid, (fn, off, nb) in blocks.items():
            l, z, t, cid, b = _UID.match(uid).groups()
            if cid not in chans:
                chans.append(cid)
            key = (int(t or 1) - 1, int(z or 1) - 1, chans.index(cid) + int(l or 1) - 1)
            planes.setdefault(key, {})[int(b)] = (fn, off, nb)
        nbpp = max(len(v) for v in planes.values())
        full = {k: v for k, v in planes.items() if len(v) == nbpp}
        ts = sorted({k[0] for k in full}); zs = sorted({k[1] for k in full})
        cs = sorted({k[2] for k in planes})
        self.t0, self.z0 = ts[0], zs[0]
        nt, nz, nc = ts[-1] - ts[0] + 1, zs[-1] - zs[0] + 1, len(cs)
        any_plane = next(iter(full.values()))
        tot = sum(v[2] for v in any_plane.values())
        if not nx:
            nx = 512
        ny = tot // (2 * nx)
        self.shape = (nt, nz, nc, ny, nx)
        self.planes = planes
        self.channel_ids = chans
        self.missing = [(t, z, c) for t in range(nt) for z in range(nz) for c in range(nc)
                        if len(planes.get((t + self.t0, z + self.z0, c), {})) != nbpp]

    def read_plane(self, t, z, c, handles=None):
        ny, nx = self.shape[3:]
        out = np.zeros(ny * nx, np.uint16)
        segs = self.planes.get((t + self.t0, z + self.z0, c))
        if not segs:
            return out.reshape(ny, nx)
        pos = 0
        for b in sorted(segs):
            fn, off, nb = segs[b]
            fh = handles.get(fn) if handles is not None else None
            if fh is None:
                fh = open(fn, 'rb')
                if handles is not None:
                    handles[fn] = fh
            fh.seek(off)
            a = np.frombuffer(fh.read(nb), '<u2')
            out[pos:pos + a.size] = a
            pos += a.size
            if handles is None:
                fh.close()
        return out.reshape(ny, nx)

    def to_npy(self, out_path, progress=None, cancel=None):
        """Write a T-major (T,Z,C,Y,X) uint16 .npy (the layout fast_run reads).

        Plain sequential writes, not a memmap: a 60 GB dirty mapping makes
        Windows (and macOS) count the whole thing against RAM.
        """
        T, Z, C, Y, X = self.shape
        handles = {}
        try:
            with open(out_path, 'wb') as f:
                np.lib.format.write_array_header_1_0(
                    f, {'descr': '<u2', 'fortran_order': False, 'shape': self.shape})
                vol = np.empty((Z, C, Y, X), np.uint16)
                for t in range(T):
                    for z in range(Z):
                        for c in range(C):
                            vol[z, c] = self.read_plane(t, z, c, handles)
                    f.write(vol.tobytes())
                    if cancel and cancel():
                        raise KeyboardInterrupt('cancelled')
                    if progress:
                        progress(t + 1, T)
        finally:
            for fh in handles.values():
                fh.close()
