"""
Replication driver for Bartsch (2026), "Evaluating State Tax Portfolio
Performance" -- see examples/README.md.

Section 1 (primary)   : Illinois and Louisiana, 2010-2015, full treatment
                        including all six reference portfolios.
Section 2 (secondary) : eleven states with 4-5 year stable-policy windows,
                        observed allocation only (reference=False).

Run from anywhere; all inputs and outputs resolve relative to this file.
Outputs land in examples/output/ and are not tracked by git.
"""
import runpy
import contextlib
import numpy as np
import frontier_segments.frontier_segments as fs
import io, sys, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    old = sys.stdout
    class _Tee:
        def write(self, s): old.write(s); buf.write(s)
        def flush(self): old.flush()
    sys.stdout = _Tee()
    result = fn(*args, **kwargs)
    sys.stdout = old
    return result, buf.getvalue()

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
    COLS = ['Max r / Same sd', 'Min sd / Same r', 'Min Var', 'Max Return', 'Max Sharpe', 'EF Min Diss']
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


# ── Load inputs (suppress tax_portfolio prints) ───────────────────────────────

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output'
OUT_DIR.mkdir(exist_ok=True)

with contextlib.redirect_stdout(io.StringIO()):
    _tp = runpy.run_path(str(HERE / 'tax_portfolio.py'))

STATES_PRIMARY   = _tp['STATES_PRIMARY']
STATES_SECONDARY = _tp['STATES_SECONDARY']
primary          = _tp['primary']
secondary        = _tp['secondary']

r_f = 0.00

# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Primary: Illinois & Louisiana, 2010-2015 (6-year overlap)
# Full treatment: absolute / quasi-relative / relative performance against
# the 6 standard reference portfolios, plus frontier + annual-weights plots.
# Directly replaces the old FL/GA analysis.
# ═══════════════════════════════════════════════════════════════════════════

_captured = {}

xlim = (0, 35)
ylim = None   # auto-scale -- no hand-tuned bounds yet for IL/LA (unlike the
              # old FL/GA ylim dict, which was tuned by eye against those plots)

for abb in STATES_PRIMARY:
    cloud = fs.compute_cloud(primary['mu'][abb], primary['sigma'][abb])
    w_o   = primary['w'][abb]

    _, _captured[f'{abb}_absolute'] = _capture(
        fs.absolute_performance, cloud, sd=True, weights=w_o, verbose=True, rf=r_f)

    _, _captured[f'{abb}_quasi'] = _capture(
        fs.quasi_relative_performance, cloud, sd=True, weights=w_o, verbose=True, rf=r_f)

    _, _captured[f'{abb}_relative'] = _capture(
        fs.relative_performance, cloud, weights=w_o, determine=True, rf=r_f, reference=True, verbose=True)

    # Frontiers only, with each year's actual portfolio (from revenue shares
    # that year) plotted as a marker -- no asset or reference-portfolio markers.
    order    = STATES_PRIMARY[abb]['cats'] + ['Other']
    years    = primary['years'][abb]
    wide_ann = primary['wide_ann'][abb]
    fig_ann, ax_ann = fs.plot_cloud(sd=True, cloud_dict=cloud, show_assets=False,
                                     xlim=xlim, ylim=ylim, tick_step=(2.5, 2),
                                     label_size=16, title_size=0, axis_title_size=16,
                                     percent=True, lw=2, show_legend=False, bw=True,
                                     ytitle="Growth Rate (%)", show=False)
    for year in years:
        w_year   = wide_ann.loc[order, year].values / wide_ann.loc['TOTAL', year]
        r_year   = float(primary['mu'][abb] @ w_year)
        sig_year = float(np.sqrt(max(w_year @ primary['sigma'][abb] @ w_year, 0.0)))
        ax_ann.scatter(sig_year * 100, r_year * 100, marker='o', color='black',
                        s=40, zorder=4)
        if year in (years[0], years[-1]):
            ax_ann.annotate(str(year), (sig_year * 100, r_year * 100),
                             textcoords="offset points", xytext=(6, 6), fontsize=11)
    fig_ann.tight_layout()
    fig_ann.savefig(str(OUT_DIR / f'annual_weights_{abb}.png'), dpi=150)
    plt.show()

_out1 = OUT_DIR / 'applied_methods_primary.xlsx'
with pd.ExcelWriter(_out1, engine='openpyxl') as _writer:
    for _sheet, _text in _captured.items():
        _rows_to_df(_parse_rows(_text)).to_excel(_writer, sheet_name=_sheet, index=False)
print(f"\nExported to {_out1}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Secondary: 4-5-year overlap states, observed portfolio only.
# reference=False on all three functions -- no reference-portfolio columns,
# just each state's own allocation's absolute/quasi-relative/relative stats,
# collected into one summary table (one row per state).
# ═══════════════════════════════════════════════════════════════════════════

_sec_rows = []
for abb in STATES_SECONDARY:
    cloud = fs.compute_cloud(secondary['mu'][abb], secondary['sigma'][abb])
    w_o   = secondary['w'][abb]

    ap = fs.absolute_performance(cloud, w_o, sd=True, rf=r_f, reference=False, verbose=True)
    qr = fs.quasi_relative_performance(cloud, w_o, sd=True, rf=r_f, reference=False, verbose=True)
    rp = fs.relative_performance(cloud, w_o, rf=r_f, reference=False, verbose=True)

    cfg = STATES_SECONDARY[abb]
    _sec_rows.append({
        'State': cfg['name'], 'Abbreviation': abb,
        'Start Year': cfg['start_year'], 'End Year': cfg['end_year'],
        'r_w': ap['r_w'], 'sd_w': ap['sd_w'], 'sharpe_w': ap['sharpe_w'],
        'rho_r': qr['rho_r'], 'rho_sigma': qr['rho_sigma'],
        'gamma_r': qr['gamma_r'], 'gamma_sigma': qr['gamma_sigma'], 'gamma_sharpe': qr['gamma_sharpe'],
        'P_r_minus': rp['P_r_minus'], 'P_sigma_plus': rp['P_sigma_plus'], 'P_sharpe_minus': rp['P_sharpe_minus'],
        'A_i': rp['A_i'], 'F_i': rp['F_i'], 'Q_A': rp['Q_A'], 'Q_F': rp['Q_F'],
    })

_sec_df = pd.DataFrame(_sec_rows)
_out2 = OUT_DIR / 'applied_methods_secondary.xlsx'
_sec_df.to_excel(_out2, index=False, sheet_name='Secondary (observed only)')
print(f"\nExported to {_out2}")
print(_sec_df.to_string(index=False))
