"""shipley_run.py — one .oir run -> stabilized z-projection, used by the GUI.

Steps for each run:
  1 index the .oir (+ its _00001.. companion files)
  2 write a T-major .npy volume store into <out>/<name>/_work/
  3 run fast_run.py (cpstab, the Python port of the Shipley 2020 pipeline)
  4 move the result to <out>/<name>/<name>_<mode>_zproj.tif, delete _work/

Also usable from the command line:
  python shipley_run.py RUN.oir OUT_DIR [--mode replicate|improved] [--workers N]
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# fast_run.py + cpstab/ sit next to this file in the Windows bundle; in the
# git repo they live in ../pyport
ENGINE = HERE if os.path.exists(os.path.join(HERE, 'fast_run.py')) else \
    os.path.normpath(os.path.join(HERE, '..', 'pyport'))
sys.path.insert(0, HERE)

from oirread import OIRRun  # noqa: E402

# overall progress share of each step
W_INDEX, W_CONVERT, W_REG, W_APPLY, W_TAIL = 0.02, 0.30, 0.30, 0.30, 0.08


class Cancelled(Exception):
    pass


def run_name(oir):
    b = os.path.basename(oir)
    return b[:-4] if b.lower().endswith('.oir') else b


def output_path(oir, out_root, mode):
    name = run_name(oir)
    return os.path.join(out_root, name, '%s_%s_zproj.tif' % (name, mode))


def _kill_tree(p):
    if p.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def process_one(oir, out_root, mode='replicate', workers=8, keep_work=False,
                log=print, progress=None, cancelled=lambda: False):
    """Returns the path of the written z-projection TIFF."""
    prog = progress or (lambda f: None)
    name = run_name(oir)
    out_dir = os.path.join(out_root, name)
    final = output_path(oir, out_root, mode)
    if os.path.exists(final):
        log('already done, skipping: %s' % final)
        prog(1.0)
        return final
    work = os.path.join(out_dir, '_work')
    os.makedirs(work, exist_ok=True)
    logf = open(os.path.join(out_dir, '%s_%s_log.txt' % (name, mode)), 'a',
                encoding='utf-8')

    def say(msg):
        line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
        log(line)
        logf.write(line + '\n')
        logf.flush()

    try:
        t0 = time.time()
        say('== %s  (mode=%s, workers=%d)' % (name, mode, workers))
        # 1 index
        run = OIRRun(oir, progress=lambda i, n: prog(W_INDEX * i / n))
        T, Z, C, Y, X = run.shape
        say('found %d files; T=%d Z=%d C=%d %dx%d' % (len(run.files), T, Z, C, Y, X))
        if run.missing:
            say('WARNING: %d planes missing in the .oir, filled with zeros'
                % len(run.missing))
        if C != 2:
            say('WARNING: expected 2 channels (red vessel dye + green cells), got %d' % C)
        npy_bytes = T * Z * C * Y * X * 2
        out_bytes = T * C * Y * X * 2
        npy = os.path.join(work, name + '.tzcyx.npy')
        have_npy = os.path.exists(npy) and os.path.exists(npy + '.ok')
        need = (0 if have_npy else npy_bytes) + out_bytes + (2 << 30)
        free = shutil.disk_usage(out_dir).free
        if free < need:
            raise RuntimeError('not enough disk space in %s: need %.0f GB, have %.0f GB'
                               % (out_root, need / 1e9, free / 1e9))
        # 2 convert
        if have_npy:
            say('reusing converted volumes from an earlier attempt')
        else:
            say('reading .oir -> %.1f GB temporary volume file' % (npy_bytes / 1e9))
            tc = time.time()
            try:
                run.to_npy(npy, progress=lambda i, n: prog(W_INDEX + W_CONVERT * i / n),
                           cancel=cancelled)
            except KeyboardInterrupt:
                raise Cancelled()
            open(npy + '.ok', 'w').close()
            say('read done in %.1f min' % ((time.time() - tc) / 60))
        prog(W_INDEX + W_CONVERT)
        if cancelled():
            raise Cancelled()
        # 3 stabilize
        exe = sys.executable
        if os.path.basename(exe).lower() == 'pythonw.exe':  # GUI launcher on Windows
            exe = os.path.join(os.path.dirname(exe), 'python.exe')
        cmd = [exe, '-u', os.path.join(ENGINE, 'fast_run.py'),
               '--raw', npy, '--out-dir', work, '--workers', str(workers),
               '--mode', mode]
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([HERE, ENGINE, env.get('PYTHONPATH', '')])
        env['PYTHONIOENCODING'] = 'utf-8'  # Windows pipes default to cp1252
        env['PYTHONUTF8'] = '1'
        kw = {}
        if os.name == 'nt':
            kw['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
        else:
            kw['start_new_session'] = True
        say('stabilizing (this is the long step)')
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=ENGINE, env=env, text=True, encoding='utf-8',
                             errors='replace', bufsize=1, **kw)
        base = W_INDEX + W_CONVERT
        tail = []

        def _watch():  # cancel even while fast_run prints nothing
            while p.poll() is None:
                if cancelled():
                    _kill_tree(p)
                    return
                time.sleep(0.5)
        threading.Thread(target=_watch, daemon=True).start()
        try:
            for line in p.stdout:
                line = line.rstrip()
                tail.append(line)
                tail = tail[-40:]
                logf.write(line + '\n')
                m = re.match(r'reg chunk (\d+)/(\d+)', line)
                if m:
                    prog(base + W_REG * int(m.group(1)) / int(m.group(2)))
                    continue
                m = re.match(r'apply slab (\d+)/(\d+)', line)
                if m:
                    prog(base + W_REG + W_APPLY * int(m.group(1)) / int(m.group(2)))
                    continue
                if line.startswith(('fast_run', 'registration', 'apply parallel',
                                    'Traceback')) or 'Error' in line:
                    say('  ' + line)
            rc = p.wait()
        finally:
            _kill_tree(p)
        if cancelled():
            raise Cancelled()
        if rc != 0:
            raise RuntimeError('stabilization failed (exit code %d). Last output:\n%s'
                               % (rc, '\n'.join(tail[-15:])))
        # 4 collect
        stem = name + '.tzcyx'
        produced = os.path.join(work, stem + '_mean_zproj.tif')
        if not os.path.exists(produced):
            raise RuntimeError('fast_run finished but %s is missing' % produced)
        os.replace(produced, final)
        for suf, new in (('.dftshifts.npz', '_shifts.npz'), ('.zprojtraj.npz', '_traj.npz')):
            src = os.path.join(work, stem + suf)
            if os.path.exists(src):
                os.replace(src, os.path.join(out_dir, '%s_%s%s' % (name, mode, new)))
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
            say('temporary files deleted')
        prog(1.0)
        say('DONE in %.1f min -> %s' % ((time.time() - t0) / 60, final))
        return final
    except Cancelled:
        say('cancelled (temporary files kept in %s; next start resumes from them)' % work)
        raise
    except Exception as e:
        say('FAILED: %s' % e)
        raise
    finally:
        logf.close()


def _print(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('oir', nargs='+')
    ap.add_argument('out')
    ap.add_argument('--mode', default='replicate', choices=['replicate', 'improved'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--keep-work', action='store_true')
    a = ap.parse_args()
    for f in a.oir:
        process_one(f, a.out, a.mode, a.workers, a.keep_work, log=_print)


if __name__ == '__main__':
    main()
