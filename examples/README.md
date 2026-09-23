# Replication: State Tax Portfolio Performance

End-to-end replication of Bartsch (2026), *Evaluating State Tax Portfolio Performance*, using `frontier_segments`. Two scripts take raw quarterly tax revenue and produce every frontier, reference allocation, and performance measure the paper reports.

Everything resolves relative to this directory, so the scripts run from anywhere with no path editing.

```bash
pip install -e ..          # or: pip install frontier_segments
python applied_methods.py  # runs tax_portfolio.py first, then the analysis
```

Outputs land in `examples/output/` (git-ignored). Expect roughly 20 minutes — most of it is the ~900,000-point simplex lattice for each state's `Q_A`/`Q_F`.

---

## What the two scripts do

**`tax_portfolio.py`** turns the raw revenue panel into portfolio inputs. For each state it consolidates tax categories into assets, computes annual percent changes for expected returns and same-quarter-prior-year changes for the covariance matrix, and takes initial weights from the state's own start-year revenue shares. Import it directly (`STATES_PRIMARY`, `STATES_SECONDARY`, `primary`, `secondary`) if you only want the μ, Σ and w arrays.

**`applied_methods.py`** runs the analysis in two sections:

| Section | States | Window | Treatment |
|---|---|---|---|
| Primary | Illinois, Louisiana | 2010–2015 | Full — `reference=True`, all six reference allocations, frontier and annual-weight figures |
| Secondary | 11 states | 2010–2013, 2010–2014, 2014–2017 | Observed allocation only — `reference=False`, one summary row per state |

Each state's window is its longest stable-tax-policy overlap, so measured volatility reflects revenue behavior rather than statutory changes.

---

## Data

`data/state_tax_data.csv` — 8,497 rows, 11 states, 2010–2017, 568 KB.

Source: U.S. Census Bureau, **Quarterly Summary of State and Local Government Tax Revenue (QTAX)**. `cat_tax` holds Census government-finance classification codes (`T01` property, `T09` general sales, `T40` individual income, `T41` corporation net income, `T53` severance, and so on), with `TOTAL` the state's all-category total. Revenue is in thousands of dollars.

This is a trimmed extract covering exactly the state-windows the examples use. It is not a separate dataset: the scripts read any file with these eight columns, so pointing `DATA_PATH` at a full QTAX extract works unchanged.

```
date, state, state_abb, cat_tax, cat_str, rev, year, quarter
```

Verified against the full 170,797-row panel: every one of the 1,014 numeric cells in the exported μ/Σ/w arrays matches to 0.0.

---

## Two rules worth knowing before reading the output

**Category selection is fixed at the start year.** For each state the largest categories by start-year revenue share are held as separate assets up to whichever prefix's cumulative share lands closest to 90%; everything else is summed into `Other`. That list is resolved once and reused for every period in the window — nothing migrates into or out of `Other` partway through.

**Categories with non-positive revenue are folded into `Other`.** Returns are same-quarter-prior-year percent changes, so a zero or negative base does not make a category volatile, it makes its return undefined. Texas `T22` (Corporations In General License) is the only such case in these windows, and it is instructive: it flips sign through zero twice, giving

```
2010Q4:  -240 /   30 - 1  =   -900%
2011Q1:   200 /  -17 - 1  =  -1276%
```

and an annualized σ of 6.73 against 0.57 for the next-most-volatile Texas category. Those are artifacts of the denominator. `drop_nonpositive_cats()` applies the rule generally rather than special-casing Texas. The consequence is that Texas holds 80.7% of revenue in separate assets rather than ~90%, and its `Other` carries 19.3% — the one state where `Other` is well above 10%.

---

## Methodological notes

Two findings from validating this replication changed the package itself, and any earlier output differs accordingly.

**The frontier is anchored at the long-only global minimum variance portfolio**, the minimum of the constrained curve φ²ₛ(r), not at the unconstrained B/C. The two coincide only when the unconstrained solution happens to be non-negative, which fails for most of these state-windows. Using the wrong anchor mislabeled part of the genuine efficient frontier as SW, inflated `gamma_sigma`, and in two states produced a starting active set with no feasible return interval at all.

**`A_i`, `F_i`, `Q_A` and `Q_F` are computed by exact dominance counting on the lattice** (`method="count"`, the default) rather than by quadrature. The quadrature path spread `n_quad` as a total budget across N−2 dimensions, leaving 3 nodes per dimension at N ≥ 7 — coarse enough that thin dominating regions integrated to exactly zero. It reported Illinois and Missouri as sitting precisely on their efficient frontiers when Illinois is 1.70pp inside its. See the main [README](../README.md#relative_performance) for the comparison.

---

## Output

In `examples/output/`:

| File | Contents |
|---|---|
| `tax_portfolio_arrays_primary.xlsx` | μ, Σ, w per primary state, one sheet each |
| `tax_portfolio_arrays_secondary.xlsx` | Same, for the eleven secondary states |
| `applied_methods_primary.xlsx` | Absolute, quasi-relative and relative tables for IL and LA |
| `applied_methods_secondary.xlsx` | One summary row per secondary state |
| `annual_weights_IL.png`, `annual_weights_LA.png` | Observed allocation's path through risk-return space, year by year |

---

[← Back to README](../README.md) · Worked single-state walkthroughs: [Georgia](../Example_Georgia.md) · [Florida](../Example_Florida.md)
