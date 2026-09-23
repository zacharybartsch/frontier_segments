"""
Tax portfolio inputs: builds the expected-return vector, covariance matrix and
observed weight vector for each state-window analyzed in Bartsch (2026),
"Evaluating State Tax Portfolio Performance".

Returns are annual percent changes in revenue; the covariance matrix is built
from same-quarter-prior-year changes, which removes seasonality without
differencing away the level. Observed weights are each state's own start-year
revenue shares.

Run directly to export the arrays, or import STATES_PRIMARY, STATES_SECONDARY,
primary and secondary to use them. See examples/README.md.

Author: Zachary Bartsch, 2026
"""
import pandas as pd
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
from pathlib import Path

HERE      = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "state_tax_data.csv"
OUT_DIR   = HERE / "output"

# DATA_PATH points at the trimmed replication extract shipped with this repo
# (11 states, 2010-2017 -- exactly the windows the examples use). Point it at
# a full Census QTAX extract with the same columns and everything below runs
# unchanged; see examples/README.md for the column contract.

USE_LW    = False   # ← toggle Ledoit-Wolf shrinkage on/off
EXPORT_NP = True    # ← write mu/sigma/w arrays to Excel when True

# ── State groups ─────────────────────────────────────────────────────────────
# Each state's window is its longest interval free of substantial statutory
# tax changes, so measured volatility reflects revenue behavior rather than
# policy changes. 'cats' = the largest tax categories by
# start-year revenue share, kept separate up to whichever prefix's cumulative
# share lands CLOSEST to 90% (may run slightly over, per the confirmed rule);
# everything else is folded into 'Other'. Initial portfolio weights come from
# each state's own start_year revenue shares.
#
# One exception is applied to these lists below, by drop_nonpositive_cats():
# a category with a non-positive revenue reading anywhere in its state's
# window is folded into 'Other' regardless of its size, because percent
# changes off a zero or negative base are undefined rather than volatile.
# The lists here are the pre-exclusion ones, so the recorded cumulative
# shares are the shares before any such fold.
#
# Primary section: the one state pair with a full 6-year overlap (2010-2015)
# — directly replaces the old FL/GA example.
STATES_PRIMARY = {
    'IL': {'name': 'Illinois',  'start_year': 2010, 'end_year': 2015,
           'cats': ['T40', 'T09', 'T41', 'T15', 'T19', 'T24', 'T13']},  # cum 89.84%
    'LA': {'name': 'Louisiana', 'start_year': 2010, 'end_year': 2015,
           'cats': ['T09', 'T40', 'T53', 'T11', 'T13', 'T12']},         # cum 92.15%
}

# Secondary section: states with 4-5 years of stable-policy overlap (rather
# than the primary pair's 6), each given its LONGEST available window from
# the same overlaps file. Illinois and Louisiana reappear here too, one year
# shorter (2010-2014) than in the primary section — same start year, so same
# category list — alongside states that only have a 4-5-year overlap at all.
STATES_SECONDARY = {
    # 2010-2014 (5-year overlap)
    'AL': {'name': 'Alabama',     'start_year': 2010, 'end_year': 2014,
           'cats': ['T40', 'T09', 'T15', 'T13', 'T41', 'T01', 'T12', 'T19']},  # cum 88.86%
    'IL': {'name': 'Illinois',    'start_year': 2010, 'end_year': 2014,
           'cats': ['T40', 'T09', 'T41', 'T15', 'T19', 'T24', 'T13']},         # cum 89.84%
    'KY': {'name': 'Kentucky',    'start_year': 2010, 'end_year': 2014,
           'cats': ['T40', 'T09', 'T13', 'T19', 'T01', 'T41', 'T53']},         # cum 88.91%
    'LA': {'name': 'Louisiana',   'start_year': 2010, 'end_year': 2014,
           'cats': ['T09', 'T40', 'T53', 'T11', 'T13', 'T12']},                # cum 92.15%
    'MS': {'name': 'Mississippi', 'start_year': 2010, 'end_year': 2014,
           'cats': ['T09', 'T40', 'T13', 'T19', 'T41', 'T12', 'T16', 'T11']},  # cum 90.65%
    'TX': {'name': 'Texas',       'start_year': 2010, 'end_year': 2014,
           'cats': ['T09', 'T19', 'T22', 'T13', 'T53', 'T24', 'T16']},         # cum 89.46%
    #   ^ T22 is dropped into 'Other' by drop_nonpositive_cats() (negative
    #     quarters in 2010-2011), leaving 80.72% held separately. Texas is
    #     therefore the one state whose 'Other' carries well over 10%.
    # 2010-2013 (4-year overlap; states not already covered by the 5yr group)
    'GA': {'name': 'Georgia',     'start_year': 2010, 'end_year': 2013,
           'cats': ['T40', 'T09', 'T13', 'T41']},                              # cum 90.69%
    'MO': {'name': 'Missouri',    'start_year': 2010, 'end_year': 2013,
           'cats': ['T40', 'T09', 'T13', 'T11', 'T12', 'T24']},                # cum 91.00%
    'NJ': {'name': 'New Jersey',  'start_year': 2010, 'end_year': 2013,
           'cats': ['T40', 'T09', 'T41', 'T15', 'T16', 'T50', 'T24', 'T12']},  # cum 90.81%
    'OK': {'name': 'Oklahoma',    'start_year': 2010, 'end_year': 2013,
           'cats': ['T40', 'T09', 'T53', 'T24', 'T13', 'T16']},                # cum 89.36%
    # 2014-2017 (4-year overlap; disjoint window)
    'AR': {'name': 'Arkansas',    'start_year': 2014, 'end_year': 2017,
           'cats': ['T09', 'T40', 'T01', 'T13', 'T41', 'T19', 'T16']},         # cum 90.79%
}


# ── 1. Load raw data (once; filtered per-state/window inside process_states) ─
raw_full = pd.read_csv(DATA_PATH)
raw_full.columns = raw_full.columns.str.lower().str.strip()
raw_full = raw_full[raw_full['state'].notna() & (raw_full['state'] != '')]


# ── 1b. Fold non-positive categories into 'Other' ───────────────────────
def drop_nonpositive_cats(states, label):
    """
    Remove from each state's 'cats' any category that reports a non-positive
    annual or quarterly value anywhere in that state's window, so it lands in
    'Other' instead of being held separately.

    Returns are same-quarter-prior-year percent changes, so a zero or negative
    base does not make a category volatile -- it makes its return undefined.
    Texas T22 (Corporations In General License) is the only such case in the
    current windows: it flips sign through zero twice, giving 2010Q4 =
    -240/30 - 1 = -900% and 2011Q1 = 200/-17 - 1 = -1276%, and an annualized
    sd of 6.73 against 0.57 for the next-most-volatile Texas category. Those
    are artifacts of the denominator, not of tax revenue behavior.

    'Other' absorbs them safely because it is a sum over many categories and
    stays comfortably positive (checked by the warning in process_states).

    This does not make category membership time-varying. The list is resolved
    once, here, from the state's full window, and is then held fixed for every
    period of that window -- nothing enters or leaves 'Other' partway through.
    """
    for abb, cfg in states.items():
        years = list(range(cfg['start_year'], cfg['end_year'] + 1))
        sr = raw_full[(raw_full['state'] == cfg['name'])
                      & (raw_full['year'].between(years[0], years[-1]))].copy()
        sr['period'] = sr['year'] * 4 + sr['quarter']
        qtr = sr.groupby(['period', 'cat_tax'], as_index=False)['rev'].sum()
        ann = sr.groupby(['year',   'cat_tax'], as_index=False)['rev'].sum()

        dropped = [c for c in cfg['cats']
                   if (qtr.loc[qtr['cat_tax'] == c, 'rev'] <= 0).any()
                   or (ann.loc[ann['cat_tax'] == c, 'rev'] <= 0).any()]
        if dropped:
            cfg['cats'] = [c for c in cfg['cats'] if c not in dropped]
            print(f"[{label}:{abb}] folded into 'Other' "
                  f"(non-positive revenue in window): {dropped}")
    return states

drop_nonpositive_cats(STATES_PRIMARY,   'Primary')
drop_nonpositive_cats(STATES_SECONDARY, 'Secondary')


# ── 2. Wide format helpers ────────────────────────────────────────────────────
def make_wide_annual(df_ann, sub_cats, years):
    keep = sub_cats + ['TOTAL']
    wide = (
        df_ann[df_ann['cat_tax'].isin(keep)]
        .pivot_table(index='cat_tax', columns='year', values='rev', aggfunc='sum')
    )[years]
    wide.loc['Other'] = wide.loc['TOTAL'] - wide.loc[sub_cats].sum()
    return wide

def make_wide_quarterly(df_qtr, sub_cats, periods):
    keep = sub_cats + ['TOTAL']
    wide = (
        df_qtr[df_qtr['cat_tax'].isin(keep)]
        .pivot_table(index='cat_tax', columns='period', values='rev', aggfunc='sum')
    )[periods]
    wide.loc['Other'] = wide.loc['TOTAL'] - wide.loc[sub_cats].sum()
    return wide.drop(index='TOTAL')


# ── 3. Annual pct changes & summary stats (Table II) ─────────────────────────
def add_stats(wide, chg_years):
    pch = pd.DataFrame(index=wide.index)
    for y in chg_years:
        pch[y] = wide[y] / wide[y - 1] - 1
    pch['ret_a'] = pch[chg_years].mean(axis=1)
    pch['sd_a']  = pch[chg_years].std(axis=1, ddof=1)
    return pch


# ── 4. Quarterly pct changes for sigma ────────────────────────────────────────
def quarterly_pch(wide, periods):
    pch = pd.DataFrame(index=wide.index)
    for i in range(4, len(periods)):          # same quarter, prior year — removes seasonality
        pch[periods[i]] = wide[periods[i]] / wide[periods[i - 4]] - 1
    return pch


# ── 5. Return vectors & covariance matrices ───────────────────────────────────
def mu_sigma(ann_pch, qtr_pch, order, chg_years, state_name='', lw=False):
    mu       = ann_pch.loc[order, chg_years].mean(axis=1).values
    qtr_data = qtr_pch.loc[order].values

    if lw:
        from sklearn.covariance import LedoitWolf as _LW
        _lw   = _LW().fit(qtr_data.T)
        sigma = _lw.covariance_ * 4
        cond  = float(np.linalg.cond(np.cov(qtr_data)))
        print(f"[{state_name}] Sample cov condition number: {cond:.2e}")
        print(f"[{state_name}] Ledoit-Wolf applied: α = {_lw.shrinkage_:.4f}\n")
    else:
        sigma = np.cov(qtr_data)              # YoY returns are already annual-scale; no ×4

    return mu, sigma

def weights_initial(wide, order, weight_year):
    total = wide.loc['TOTAL', weight_year]
    return wide.loc[order, weight_year].values / total

def unconstrained_mvp(sigma, names):
    inv  = np.linalg.inv(sigma)
    ones = np.ones(len(names))
    w    = inv @ ones / (ones @ inv @ ones)
    return dict(zip(names, w))

def print_np(name, arr):
    if arr.ndim == 1:
        vals = ', '.join(f'{x:.8f}' for x in arr)
        print(f'{name} = np.array([{vals}])')
    else:
        rows = ',\n '.join(
            '[' + ', '.join(f'{x:.8f}' for x in row) + ']'
            for row in arr
        )
        print(f'{name} = np.array([\n {rows}])')
    print()


# ── 6. Per-group pipeline: each state uses its own start_year/end_year ───────
def process_states(states, label):
    """
    Run the full annual/quarterly/mu-sigma pipeline for one state group,
    each state using its own window (cfg['start_year']..cfg['end_year']).
    Returns a dict of {field: {abb: value}} for: wide_ann, wide_qtr, ann_pch,
    qtr_pch, mu, sigma, w, order, years, chg_years.
    """
    result = {k: {} for k in
               ('wide_ann', 'wide_qtr', 'ann_pch', 'qtr_pch',
                'mu', 'sigma', 'w', 'order', 'years', 'chg_years')}

    for abb, cfg in states.items():
        years     = list(range(cfg['start_year'], cfg['end_year'] + 1))
        chg_years = years[1:]
        order     = cfg['cats'] + ['Other']

        state_raw = raw_full[(raw_full['state'] == cfg['name'])
                              & (raw_full['year'].between(years[0], years[-1]))].copy()

        df_ann = state_raw.groupby(['year', 'cat_tax'], as_index=False)['rev'].sum()

        state_raw['period'] = state_raw['year'] * 4 + state_raw['quarter']
        df_qtr  = state_raw.groupby(['period', 'cat_tax'], as_index=False)['rev'].sum()
        periods = sorted(df_qtr['period'].unique())

        wide_ann = make_wide_annual(df_ann, cfg['cats'], years)
        wide_qtr = make_wide_quarterly(df_qtr, cfg['cats'], periods)
        pch      = add_stats(wide_ann, chg_years)
        qpch     = quarterly_pch(wide_qtr, periods)

        if (wide_qtr <= 0).any().any():
            bad = list(wide_qtr.index[(wide_qtr <= 0).any(axis=1)])
            print(f"[{label}:{abb}] WARNING — zero/negative quarterly revenue in: {bad}")
        if ~np.isfinite(qpch.values).all():
            bad = list(qpch.index[(~np.isfinite(qpch.values)).any(axis=1)])
            print(f"[{label}:{abb}] WARNING — NaN/inf in quarterly pct changes for: {bad}")

        mu, sigma = mu_sigma(pch, qpch, order, chg_years, f'{label}:{abb}', lw=USE_LW)
        w = weights_initial(wide_ann, order, cfg['start_year'])

        result['wide_ann'][abb] = wide_ann
        result['wide_qtr'][abb] = wide_qtr
        result['ann_pch'][abb]  = pch
        result['qtr_pch'][abb]  = qpch
        result['mu'][abb]       = mu
        result['sigma'][abb]    = sigma
        result['w'][abb]        = w
        result['order'][abb]    = order
        result['years'][abb]    = years
        result['chg_years'][abb] = chg_years

    return result


primary   = process_states(STATES_PRIMARY,   'Primary')
secondary = process_states(STATES_SECONDARY, 'Secondary')


# ── 7. Reporting ──────────────────────────────────────────────────────────────
def report_group(states, res, label):
    for abb, cfg in states.items():
        order = cfg['cats'] + ['Other', 'TOTAL']
        print("=" * 60)
        print(f"TABLE II — {label}: {cfg['name']} ({abb}, {cfg['start_year']}-{cfg['end_year']})"
              f"  (cat_tax | ret_a | sd_a)")
        print("=" * 60)
        print(res['ann_pch'][abb][['ret_a', 'sd_a']].reindex(order).to_string(float_format="{:.6f}".format))
        print()

    print(f"Unconstrained MVP weights — {label}:")
    for abb, cfg in states.items():
        mvp = unconstrained_mvp(res['sigma'][abb], res['order'][abb])
        print(f"  {cfg['name']}: { {k: f'{v:.4f}' for k, v in mvp.items()} }")
    print()

    for abb, cfg in states.items():
        print("=" * 60)
        print(f"{label}: {cfg['name']} ({abb}, {cfg['start_year']}-{cfg['end_year']})"
              f" — categories: {res['order'][abb]}")
        print("=" * 60)
        print_np(f'mu_{abb}',    res['mu'][abb])
        print_np(f'sigma_{abb}', res['sigma'][abb])
        print_np(f'w_{abb}',     res['w'][abb])

report_group(STATES_PRIMARY,   primary,   'Primary')
report_group(STATES_SECONDARY, secondary, 'Secondary')


# ── 8. Excel export of mu/sigma/w arrays ──────────────────────────────────────
def export_group(states, res, out_path):
    with pd.ExcelWriter(out_path, engine='openpyxl') as _writer:
        for abb, cfg in states.items():
            order   = res['order'][abb]
            order_t = order + ['TOTAL']
            mu, sigma, w = res['mu'][abb], res['sigma'][abb], res['w'][abb]
            chg_years    = res['chg_years'][abb]

            # TOTAL mu: mean of its annual pct changes (already in ann_pch)
            mu_total = float(res['ann_pch'][abb].loc['TOTAL', chg_years].mean())
            # TOTAL YoY quarterly pct changes (consistent with quarterly_pch)
            _total_qtr = res['wide_qtr'][abb].sum(axis=0)
            _periods_n = len(_total_qtr)
            _total_pch = np.array([_total_qtr.iloc[i] / _total_qtr.iloc[i - 4] - 1
                                   for i in range(4, _periods_n)])
            # Augmented sigma: append TOTAL as last row/col
            _aug_data = np.vstack([res['qtr_pch'][abb].loc[order].values, _total_pch])
            sigma_t   = np.cov(_aug_data)

            mu_arr = np.append(mu, mu_total).astype(float)          # shape (N+1,)
            pd.DataFrame(mu_arr.reshape(1, -1)).to_excel(
                _writer, sheet_name=abb, startrow=1, startcol=1,
                index=False, header=False)

            pd.DataFrame(w.astype(float).reshape(1, -1)).to_excel(
                _writer, sheet_name=abb, startrow=4, startcol=1,
                index=False, header=False)

            pd.DataFrame(sigma_t.astype(float)).to_excel(
                _writer, sheet_name=abb, startrow=7, startcol=1,
                index=False, header=False)

            ws = _writer.sheets[abb]
            ws.cell(1, 1, 'mu_' + abb)
            for c, lbl in enumerate(order_t, 2):
                ws.cell(1, c, lbl)
            ws.cell(4, 1, 'w_' + abb)
            for c, lbl in enumerate(order, 2):
                ws.cell(4, c, lbl)
            ws.cell(7, 1, 'sigma_' + abb)
            for c, lbl in enumerate(order_t, 2):
                ws.cell(7, c, lbl)
            for r, lbl in enumerate(order_t, 8):
                ws.cell(r, 1, lbl)

    print(f"\nExported arrays to {out_path}")

if EXPORT_NP:
    OUT_DIR.mkdir(exist_ok=True)
    export_group(STATES_PRIMARY,   primary,   OUT_DIR / 'tax_portfolio_arrays_primary.xlsx')
    export_group(STATES_SECONDARY, secondary, OUT_DIR / 'tax_portfolio_arrays_secondary.xlsx')
