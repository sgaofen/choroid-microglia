"""Drive the real GUI end to end: add file, set output, Start, (optionally Cancel),
wait for 'done', screenshot the window. Usage:
  python test_gui.py OIR OUT [--cancel-after SEC] [--shot PNG]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tkinter as tk  # noqa: E402
import app  # noqa: E402

pa = argparse.ArgumentParser()
pa.add_argument('oir')
pa.add_argument('out')
pa.add_argument('--cancel-after', type=float, default=0)
pa.add_argument('--shot', default='')
pa.add_argument('--workers', type=int, default=0)
pa.add_argument('--mode', default='replicate')
a = pa.parse_args()

root = tk.Tk()
ui = app.App(root)
ui.files.insert('end', a.oir)
ui.out.set(a.out)
if a.workers:
    ui.workers.set(a.workers)
ui.mode.set(a.mode)
t0 = time.time()
shots = []


def shoot(tag):
    if not a.shot:
        return
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    fn = a.shot.replace('.png', '_%s.png' % tag)
    subprocess.run(['screencapture', '-x', '-R%d,%d,%d,%d' % (x, y, w, h), fn])
    shots.append(fn)


def tick():
    el = time.time() - t0
    if a.cancel_after and el > a.cancel_after and not ui.cancel_flag.is_set():
        shoot('before_cancel')
        print('GUI test: pressing Cancel at %.0fs, status=%r' % (el, ui.status.get()), flush=True)
        ui.cancel()
    if ui.worker is not None and not ui.worker.is_alive() and str(ui.b_start['state']) == 'normal':
        shoot('end')
        print('GUI test: finished at %.0fs, status=%r, bar=%s' % (el, ui.status.get(), ui.bar['value']), flush=True)
        print('--- log box ---\n' + ui.log.get('1.0', 'end'), flush=True)
        root.after(500, root.destroy)
        return
    if int(el) % 30 == 0:
        print('GUI test: %.0fs status=%r bar=%s' % (el, ui.status.get(), ui.bar['value']), flush=True)
    root.after(1000, tick)


def go():
    shoot('start')
    ui.start()
    root.after(1000, tick)


root.after(800, go)
root.mainloop()
print('shots:', shots)
