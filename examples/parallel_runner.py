"""
Cross-state parallel runner for the replication pipeline.

Produces byte-for-byte the same two workbooks as applied_methods.py, by
running the state-windows in separate processes instead of one after another.

Why across states and not within one: the bottleneck is numpy's argsort in the
dominance sweep (measured ~370 ms per 2e6 elements on a 4-core i5-1135G7,
against ~20 ms for the matmul), and numpy's sort is single-threaded. A
parallel merge sort is real work for roughly a 2x return. The thirteen
state-windows, by contrast, are completely independent -- no shared state, no
ordering, no communication -- so process-level parallelism gets the full core
count for a few dozen lines.

Memory is the thing that can bite. Each worker holds its own lattice, so peak
is (workers x per-state peak). The worker count is therefore capped by
available RAM, not by core count: raise n_points later and the cap drops by
itself. Use --probe to measure per-state peak on this machine before choosing.

    python parallel_runner.py                 # auto worker count
    python parallel_runner.py --workers 6
    python parallel_runner.py --probe         # measure, choose nothing
    python parallel_runner.py --serial        # reference path, for comparison
"""
import argparse
import contextlib
import io
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output'

R_F = 0.00
# Rough peak resident bytes for one state at n_points=1e6, measured with
# --probe on a 16 GB i5-1135G7. Scale it if you change n_points.
PEAK_PER_STATE_BYTES = 1_100_000_000
RAM_HEADROOM = 0.70          # never plan to use more than this share of free RAM


# ── helpers lifted from applied_methods.py so both paths format identically ──
def _parse_rows(text):
    rows = []
    for line in text.split('\n'):
        s = line.strip()
        if not s or all(c in '-| ' for c in s):
            continue
        if '|' in s:
            cells = [c.strip() for c in s.split('|') if c.strip()]
            if len(cells) >= 2 and cells[0] not in ('asset', 'stat'):
                rows.append(cells)
    return rows


def _rows_to_df(rows):
    COLS = ['Max r / Same sd', 'Min sd / Same r', 'Min Var',
            'Max Return', 'Max Sharpe', 'EF Min Diss']
    records = []
    for row in rows:
        rec = {'Stat': row[0], 'w_o': row[1] if len(row) > 1 else ''}
        for i, col in enumerate(COLS):
            cell = row[i + 2] if i + 2 < len(row) else ''
            parts = cell.split()
            rec[col] = parts[0] if parts else ''
            rec[col + ' (delta)'] = parts[1] if len(parts) >= 2 else ''
        records.append(rec)
    return pd.DataFrame(records)


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# ── workers: module level, so Windows spawn can pickle them ─────────────────
def _work_primary(task):
    """Full treatment for one primary state. Returns the three verbose blocks."""
    import frontier_segments.frontier_segments as fs
    abb, mu, sigma, w_o = task
    cloud = fs.compute_cloud(mu, sigma)
    out = {}
    _, out[f'{abb}_absolute'] = _capture(
        fs.absolute_performance, cloud, sd=True, weights=w_o, verbose=True, rf=R_F)
    _, out[f'{abb}_quasi'] = _capture(
        fs.quasi_relative_performance, cloud, sd=True, weights=w_o, verbose=True, rf=R_F)
    _, out[f'{abb}_relative'] = _capture(
        fs.relative_performance, cloud, weights=w_o, reference=True, verbose=True, rf=R_F)
    return abb, out


def _work_secondary(task):
    """Observed allocation only. Returns the summary row."""
    import frontier_segments.frontier_segments as fs
    abb, mu, sigma, w_o, cfg = task
    cloud = fs.compute_cloud(mu, sigma)
    with contextlib.redirect_stdout(io.StringIO()):
        ap = fs.absolute_performance(cloud, w_o, sd=True, rf=R_F, reference=False)
        qr = fs.quasi_relative_performance(cloud, w_o, sd=True, rf=R_F, reference=False)
        rp = fs.relative_performance(cloud, w_o, rf=R_F, reference=False)
    return {
        'State': cfg['name'], 'Abbreviation': abb,
        'Start Year': cfg['start_year'], 'End Year': cfg['end_year'],
        'r_w': ap['r_w'], 'sd_w': ap['sd_w'], 'sharpe_w': ap['sharpe_w'],
        'rho_r': qr['rho_r'], 'rho_sigma': qr['rho_sigma'],
        'gamma_r': qr['gamma_r'], 'gamma_sigma': qr['gamma_sigma'],
        'gamma_sharpe': qr['gamma_sharpe'],
        'P_r_minus': rp['P_r_minus'], 'P_sigma_plus': rp['P_sigma_plus'],
        'P_sharpe_minus': rp['P_sharpe_minus'],
        'A_i': rp['A_i'], 'F_i': rp['F_i'], 'Q_A': rp['Q_A'], 'Q_F': rp['Q_F'],
    }


# ── resources ───────────────────────────────────────────────────────────────
def _free_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    if sys.platform == 'win32':                     # no psutil: ask Windows
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MS(); st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullAvailPhys)
    return 4 * 1024 ** 3                            # conservative fallback


def plan_workers(requested=None, peak=PEAK_PER_STATE_BYTES):
    """
    Worker count, capped by memory rather than by cores.

    Oversubscribing cores costs time; oversubscribing RAM costs the run. The
    memory cap is the binding one, and it tightens automatically if n_points
    is raised later.
    """
    cores = os.cpu_count() or 1
    free = _free_ram_bytes()
    by_ram = max(1, int(free * RAM_HEADROOM // peak))
    n = min(cores, by_ram) if requested is None else requested
    print(f"  cores={cores}  free RAM={free/1e9:.1f} GB  "
          f"peak/state~{peak/1e9:.1f} GB  ->  RAM allows {by_ram}, using {n}")
    if requested is not None and requested > by_ram:
        print(f"  WARNING: {requested} workers exceeds the {by_ram} that RAM allows; "
              f"expect swapping or a MemoryError.")
    return max(1, n)


def probe(primary, secondary, STATES_SECONDARY):
    """Measure one state's peak so PEAK_PER_STATE_BYTES can be set honestly."""
    import tracemalloc
    import frontier_segments.frontier_segments as fs
    abb = max(STATES_SECONDARY, key=lambda a: len(secondary['mu'][a]))
    print(f"probing on {abb} (N={len(secondary['mu'][abb])}, the widest state)")
    tracemalloc.start()
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        cloud = fs.compute_cloud(secondary['mu'][abb], secondary['sigma'][abb])
        fs.relative_performance(cloud, secondary['w'][abb], rf=R_F, reference=False)
    el = time.time() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  python-traced peak {peak/1e9:.2f} GB in {el:.1f}s")
    print(f"  tracemalloc misses numpy temporaries, so that is a FLOOR, not the")
    print(f"  real peak. Planning below uses the conservative constant instead.")
    print()
    print(f"  plan on the measured floor ({peak/1e9:.2f} GB/state) -- OPTIMISTIC:")
    plan_workers(peak=max(peak, 1))
    print(f"  plan on PEAK_PER_STATE_BYTES ({PEAK_PER_STATE_BYTES/1e9:.1f} GB/state) -- USE THIS:")
    plan_workers(peak=PEAK_PER_STATE_BYTES)
    return peak


# ── driver ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--serial', action='store_true', help='reference path')
    ap.add_argument('--probe', action='store_true', help='measure and exit')
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print("loading inputs ...")
    with contextlib.redirect_stdout(io.StringIO()):
        tp = runpy.run_path(str(HERE / 'tax_portfolio.py'))
    STATES_PRIMARY = tp['STATES_PRIMARY']
    STATES_SECONDARY = tp['STATES_SECONDARY']
    primary, secondary = tp['primary'], tp['secondary']

    if args.probe:
        probe(primary, secondary, STATES_SECONDARY)
        return

    n = 1 if args.serial else plan_workers(args.workers)
    t0 = time.time()

    prim_tasks = [(abb, primary['mu'][abb], primary['sigma'][abb], primary['w'][abb])
                  for abb in STATES_PRIMARY]
    sec_tasks = [(abb, secondary['mu'][abb], secondary['sigma'][abb],
                  secondary['w'][abb], STATES_SECONDARY[abb])
                 for abb in STATES_SECONDARY]

    if n == 1:
        prim_out = [_work_primary(t) for t in prim_tasks]
        sec_rows = [_work_secondary(t) for t in sec_tasks]
    else:
        # one pool for both sections: the primary states are the long poles, so
        # submitting them first keeps the tail short
        with ProcessPoolExecutor(max_workers=n) as ex:
            fp = [ex.submit(_work_primary, t) for t in prim_tasks]
            fs_ = [ex.submit(_work_secondary, t) for t in sec_tasks]
            prim_out = [f.result() for f in fp]
            sec_rows = [f.result() for f in fs_]

    captured = {}
    for abb, blocks in prim_out:
        captured.update(blocks)

    out1 = OUT_DIR / 'applied_methods_primary.xlsx'
    with pd.ExcelWriter(out1, engine='openpyxl') as wr:
        for sheet, text in captured.items():
            _rows_to_df(_parse_rows(text)).to_excel(wr, sheet_name=sheet, index=False)

    out2 = OUT_DIR / 'applied_methods_secondary.xlsx'
    pd.DataFrame(sec_rows).to_excel(out2, index=False,
                                    sheet_name='Secondary (observed only)')

    el = time.time() - t0
    print(f"\n{len(prim_tasks)} primary + {len(sec_tasks)} secondary states "
          f"on {n} worker(s) in {el/60:.1f} min")
    print(f"  {out1}\n  {out2}")
    print("\nFigures are NOT produced here -- run applied_methods.py for those.")


if __name__ == '__main__':
    main()
