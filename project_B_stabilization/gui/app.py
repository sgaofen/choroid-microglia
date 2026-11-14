"""Shipley stabilizer — minimal window around shipley_run.process_one."""
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shipley_run  # noqa: E402

SETTINGS = os.path.join(os.path.expanduser('~'), '.shipley_gui_last_output.txt')


GB_PER_WORKER = 5.5   # measured: 5.3 GB per worker in the registration step
GB_PARENT_PEAK = 17   # measured: main process at the final z-projection step


def ram_gb():
    """(total, available) physical memory in GB; None if unknown."""
    try:
        if os.name == 'nt':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
        total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
        return total, total * 0.7
    except Exception:
        return None


def default_workers():
    n = max(1, (os.cpu_count() or 4) - 2)
    r = ram_gb()
    if r:
        n = min(n, int((r[1] - 2) // GB_PER_WORKER))
    return max(1, min(n, 10))


class App:
    def __init__(self, root):
        self.root = root
        root.title('Shipley stabilizer')
        root.minsize(640, 520)
        self.q = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker = None

        pad = dict(padx=8, pady=4)
        f = ttk.Frame(root, padding=10)
        f.pack(fill='both', expand=True)
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text='1. OIR files (pick the main .oir of each run; '
                          'its _00001… parts are found automatically)').grid(
            row=0, column=0, columnspan=2, sticky='w')
        lf = ttk.Frame(f)
        lf.grid(row=1, column=0, sticky='nsew', **pad)
        f.rowconfigure(1, weight=1)
        self.files = tk.Listbox(lf, height=6, selectmode='extended')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.files.yview)
        self.files.configure(yscrollcommand=sb.set)
        self.files.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        bf = ttk.Frame(f)
        bf.grid(row=1, column=1, sticky='n', **pad)
        self.b_add = ttk.Button(bf, text='Add…', command=self.add_files)
        self.b_add.pack(fill='x', pady=2)
        self.b_rm = ttk.Button(bf, text='Remove', command=self.remove_files)
        self.b_rm.pack(fill='x', pady=2)

        ttk.Label(f, text='2. Output folder').grid(row=2, column=0, sticky='w')
        of = ttk.Frame(f)
        of.grid(row=3, column=0, columnspan=2, sticky='ew', **pad)
        of.columnconfigure(0, weight=1)
        self.out = tk.StringVar(value=self._last_output())
        ttk.Entry(of, textvariable=self.out).grid(row=0, column=0, sticky='ew')
        self.b_out = ttk.Button(of, text='Browse…', command=self.pick_out)
        self.b_out.grid(row=0, column=1, padx=(6, 0))

        ttk.Label(f, text='3. Options').grid(row=4, column=0, sticky='w')
        opt = ttk.Frame(f)
        opt.grid(row=5, column=0, columnspan=2, sticky='w', **pad)
        self.mode = tk.StringVar(value='replicate')
        ttk.Radiobutton(opt, text='Replicate (same algorithm as the original MATLAB)',
                        variable=self.mode, value='replicate').grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Radiobutton(opt, text='Improved (less residual drift, sharper)',
                        variable=self.mode, value='improved').grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Label(opt, text='Parallel workers:').grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.workers = tk.IntVar(value=default_workers())
        ttk.Spinbox(opt, from_=1, to=max(1, os.cpu_count() or 4), width=4,
                    textvariable=self.workers).grid(row=2, column=1, sticky='w', pady=(6, 0))
        r = ram_gb()
        hint = ('about %.1f GB RAM each; this computer has %.0f GB, %.0f GB free now'
                % (GB_PER_WORKER, r[0], r[1])) if r else 'about %.1f GB RAM each' % GB_PER_WORKER
        ttk.Label(opt, text=hint, foreground='gray').grid(
            row=2, column=2, sticky='w', padx=6, pady=(6, 0))

        rf = ttk.Frame(f)
        rf.grid(row=6, column=0, columnspan=2, sticky='ew', **pad)
        self.b_start = ttk.Button(rf, text='Start', command=self.start)
        self.b_start.pack(side='left')
        self.b_cancel = ttk.Button(rf, text='Cancel', command=self.cancel, state='disabled')
        self.b_cancel.pack(side='left', padx=6)
        ttk.Button(rf, text='Open output folder', command=self.open_out).pack(side='right')

        self.status = tk.StringVar(value='Ready.')
        ttk.Label(f, textvariable=self.status).grid(row=7, column=0, columnspan=2, sticky='w', padx=8)
        self.bar = ttk.Progressbar(f, maximum=1000)
        self.bar.grid(row=8, column=0, columnspan=2, sticky='ew', **pad)

        self.log = tk.Text(f, height=10, wrap='word', state='disabled')
        self.log.grid(row=9, column=0, columnspan=2, sticky='nsew', **pad)
        f.rowconfigure(9, weight=2)

        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(150, self.poll)

    # ---- inputs ----------------------------------------------------------
    def _last_output(self):
        try:
            with open(SETTINGS, encoding='utf-8') as fh:
                return fh.read().strip()
        except OSError:
            return ''

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title='Choose .oir files',
            filetypes=[('Olympus OIR', '*.oir'), ('All files', '*.*')])
        have = set(self.files.get(0, 'end'))
        for p in paths:
            p = os.path.normpath(p)
            if p not in have:
                self.files.insert('end', p)
        if paths and not self.out.get():
            self.out.set(os.path.join(os.path.dirname(paths[0]), 'stabilized'))

    def remove_files(self):
        for i in reversed(self.files.curselection()):
            self.files.delete(i)

    def pick_out(self):
        d = filedialog.askdirectory(title='Choose output folder')
        if d:
            self.out.set(os.path.normpath(d))

    def open_out(self):
        d = self.out.get()
        if not d or not os.path.isdir(d):
            messagebox.showinfo('Shipley stabilizer', 'The output folder does not exist yet.')
            return
        if os.name == 'nt':
            os.startfile(d)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', d])
        else:
            subprocess.Popen(['xdg-open', d])

    # ---- run -------------------------------------------------------------
    def start(self):
        files = list(self.files.get(0, 'end'))
        out = self.out.get().strip()
        if not files:
            messagebox.showwarning('Shipley stabilizer', 'Add at least one .oir file.')
            return
        if not out:
            messagebox.showwarning('Shipley stabilizer', 'Choose an output folder.')
            return
        try:
            workers = int(self.workers.get())
        except (tk.TclError, ValueError):
            workers = default_workers()
        r = ram_gb()
        if r and max(workers * GB_PER_WORKER, GB_PARENT_PEAK) > r[1] + 1:
            if not messagebox.askyesno(
                    'Shipley stabilizer',
                    'This needs about %.0f GB of memory but only %.0f GB is free.\n'
                    'Close other programs or use fewer workers.\n\nStart anyway?'
                    % (max(workers * GB_PER_WORKER, GB_PARENT_PEAK), r[1])):
                return
        os.makedirs(out, exist_ok=True)
        try:
            with open(SETTINGS, 'w', encoding='utf-8') as fh:
                fh.write(out)
        except OSError:
            pass
        self.cancel_flag.clear()
        self._busy(True)
        self.worker = threading.Thread(target=self._run, daemon=True,
                                       args=(files, out, self.mode.get(), workers))
        self.worker.start()

    def _run(self, files, out, mode, workers):
        n = len(files)
        ok, failed = [], []
        for i, f in enumerate(files):
            if self.cancel_flag.is_set():
                break
            self.q.put(('status', 'Run %d of %d: %s' % (i + 1, n, os.path.basename(f))))
            try:
                shipley_run.process_one(
                    f, out, mode, workers,
                    log=lambda m: self.q.put(('log', m)),
                    progress=lambda p, i=i: self.q.put(('prog', (i + p) / n)),
                    cancelled=self.cancel_flag.is_set)
                ok.append(f)
            except shipley_run.Cancelled:
                break
            except Exception:
                failed.append(f)
                self.q.put(('log', traceback.format_exc()))
        self.q.put(('done', (ok, failed, self.cancel_flag.is_set())))

    def cancel(self):
        self.cancel_flag.set()
        self.status.set('Cancelling…')
        self.b_cancel.configure(state='disabled')

    def _busy(self, busy):
        s = 'disabled' if busy else 'normal'
        for b in (self.b_start, self.b_add, self.b_rm, self.b_out):
            b.configure(state=s)
        self.b_cancel.configure(state='normal' if busy else 'disabled')

    def _append(self, msg):
        self.log.configure(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == 'log':
                    self._append(val)
                elif kind == 'status':
                    self.status.set(val)
                    self._t0 = time.time()
                elif kind == 'prog':
                    self.bar['value'] = int(val * 1000)
                elif kind == 'done':
                    ok, failed, cancelled = val
                    self._busy(False)
                    if cancelled:
                        msg = 'Cancelled. Finished %d run(s).' % len(ok)
                    elif failed:
                        msg = 'Finished %d, FAILED %d (see log).' % (len(ok), len(failed))
                    else:
                        msg = 'All done: %d run(s).' % len(ok)
                        self.bar['value'] = 1000
                    self.status.set(msg)
                    self._append(msg)
                    if failed and not cancelled:
                        messagebox.showerror('Shipley stabilizer', msg)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno('Shipley stabilizer',
                                       'Processing is running. Stop it and quit?'):
                return
            self.cancel_flag.set()
            self.worker.join(timeout=15)
        self.root.destroy()


def main():
    if sys.stderr is None:  # pythonw.exe has no console: keep errors somewhere
        import tempfile
        sys.stderr = sys.stdout = open(os.path.join(tempfile.gettempdir(),
                                                    'shipley_gui_error.log'), 'a')
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista' if os.name == 'nt' else 'clam')
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
