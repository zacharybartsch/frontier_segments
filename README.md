# frontier_segments

**Portfolio Efficient Frontiers & Diagnostics for Python**

A Python package for exact Markowitz mean-variance frontier computation and portfolio performance analysis. Given a vector of expected returns and a covariance matrix, `frontier_segments` traces all three frontier branches — the northwest efficient frontier (NW EF), the southwest minimum-variance frontier (SW), and the east variance-maximizing frontier (EA) — and provides a suite of absolute, quasi-relative, and relative performance measures for any observed portfolio.

---

## Installation

```bash
pip install git+https://github.com/zacharybartsch/frontier_segments.git
```

**Dependencies:** `numpy`, `scipy`, `matplotlib` (also `pandas` and `scipy.stats` if you use `q_plot`'s `stats=True` export)

---

## Quick Start

The example below uses Georgia's actual 2013 state tax revenue allocation — five consolidated tax sources (General Sales and Gross Receipts, Motor Fuels, Individual Income, Corporation Net Income, and a residual "Other" category), replicated from Bartsch (2026), *Evaluating State Tax Portfolio Performance* (Tables I–II).

```python
import numpy as np
import frontier_segments.frontier_segments as fs

# Georgia, 2013 — five consolidated tax sources
mu_GA = np.array([0.04321064, 0.09432582, 0.05994018, 0.04744774, 0.05847559])
Sigma_GA = np.array([
    [ 0.00117476, -0.00005078, -0.00047024, -0.00156685, -0.00228487],
    [-0.00005078,  0.02358429,  0.00088930, -0.00595364,  0.00263467],
    [-0.00047024,  0.00088930,  0.00245991, -0.00055711, -0.00010694],
    [-0.00156685, -0.00595364, -0.00055711,  0.03806298, -0.00113346],
    [-0.00228487,  0.00263467, -0.00010694, -0.00113346,  0.01611126],
])
w_o = np.array([0.27856823, 0.06649115, 0.49736724, 0.04979834, 0.10777504])  # observed 2013 allocation

# Step 1: compute frontiers once
cloud = fs.compute_cloud(mu_GA, Sigma_GA)

# Step 2: run diagnostics
ap = fs.absolute_performance(cloud, w_o, verbose=True)
qr = fs.quasi_relative_performance(cloud, w_o, verbose=True)
rp = fs.relative_performance(cloud, w_o, reference=True, verbose=True)

# Step 3: visualize
fs.plot_cloud(cloud, weights=w_o)

# Step 4: distribution of a relative measure over the feasible simplex
fs.q_plot(cloud, weights=w_o, stat="A")
```

---

## The Markowitz Cloud

`frontier_segments` partitions the feasible portfolio space into three frontier branches:

| Branch | Flag key | Description |
|--------|----------|-------------|
| **NW efficient frontier** | `ef_frontier` | Minimizes variance at each return level above the global minimum |
| **SW frontier** | `low_frontier` | Minimizes variance at each return level *below* the global minimum |
| **East (EA) frontier** | `ea_frontier` | Maximizes variance at each return level (outer boundary of the feasible cloud) |

Each branch is represented as a list of piecewise-parabolic segments. A segment records its active asset set, return interval `[lower_r, upper_r]`, and the parabola coefficients `a·r² + b·r + c = σ²`.

---

## API Reference

### `compute_cloud`

```python
cloud = fs.compute_cloud(mu, Sigma,
                      ef=True,            # compute NW efficient frontier
                      swf=True,           # compute SW frontier
                      east_mode="exact",  # "exact" | "grid" | False
                      east_K=200,         # grid resolution when east_mode="grid"
                      verbose=False)
```

Computes frontier segments and returns a `cloud_dict` for reuse across all diagnostics. Pass `verbose=True` to print a segment table.

**Returns** a dict with keys: `segments`, `mu`, `Sigma`, `r_global`, `N`, `chol_L`.

- `r_global` — the global minimum-variance portfolio return (MVP).
- `east_mode="exact"` enumerates all C(N, 2) asset pairs analytically; `"grid"` uses `east_K` evenly-spaced return points (faster for large N).

---

### `absolute_performance`

```python
ap = fs.absolute_performance(cloud, weights,
                          sd=True,                  # True → report std dev; False → variance
                          rf=0.0,                    # risk-free rate for Sharpe
                          reference_weights=None,    # optional benchmark
                          verbose=False)
```

Locates the observed portfolio relative to the frontier. Four frontier reference points are computed by search, plus three always-on-the-EF anchor points:

| Key | Description |
|-----|-------------|
| `frontier_same_var` | Highest-return frontier point at the same variance as `w_o` |
| `frontier_same_r` | Lowest-variance frontier point at the same return as `w_o` |
| `nearest_ef` | Nearest NW EF point in (return, σ) space (golden-section search) |
| `closest_ef_weights` | NW EF point with minimum portfolio-weight dissimilarity *D* |
| `min_var` | Global minimum-variance portfolio (`r_global`) |
| `max_return` | Single highest-return asset |
| `max_sharpe` | Tangency (max-Sharpe) portfolio at `rf` |

**Returns** a dict with keys: `r_w, var_w, sd_w, sharpe_w, rf, frontier_same_var, frontier_same_r, nearest_ef, closest_ef_weights, min_var, max_return, max_sharpe`.

**Verbose output** prints a comparison table of `w_o` against six reference portfolios (Max r\|Same σ, Min σ\|Same r, Min Var, Max Return, Max Sharpe, EF Min Diss, and optionally a reference portfolio), showing r, σ, Sharpe, and asset weights with signed deltas.

---

### `quasi_relative_performance`

```python
qr = fs.quasi_relative_performance(cloud, weights,
                                sd=True,
                                rf=0.0,
                                w_ref=None,
                                verbose=False)
```

Measures how the observed portfolio is positioned *within* the feasible return-risk cloud, returning scores in [0, 1] where **1 = best**.

**Conditional measures (rho)** — conditioned on the portfolio's own risk or return level:

| Key | Formula | Interpretation |
|-----|---------|----------------|
| `rho_r` | (r\_o − r\_min) / (r\_max − r\_min) at σ\_o | Return rank within all feasible returns at the observed σ; 1 = on EF |
| `rho_sigma` | (σ\_max − σ\_o) / (σ\_max − σ\_min) at r\_o | Risk rank within feasible σ range at the observed return; 1 = on EF |

**Unconditional measures (gamma)** — anchored to the global feasible extremes across the entire cloud:

| Key | Formula | 1 = | 0 = |
|-----|---------|-----|-----|
| `gamma_r` | (r\_o − min(μ)) / (max(μ) − min(μ)) | highest-return single asset | lowest-return single asset |
| `gamma_sigma` | (σ\_max\_global − σ\_o) / (σ\_max\_global − σ\_min\_global) | MVP | highest-variance single asset |
| `gamma_sharpe` | (SR\_o − SR\_min) / (SR\_max − SR\_min) | tangency portfolio | worst single-asset Sharpe |

where σ\_min\_global = MVP σ, σ\_max\_global = max√(diag(Σ)), SR\_max = tangency Sharpe (at `rf`), SR\_min = worst single-asset Sharpe.

**Returns** a dict with keys:

```
r_w, var_w, sd_w, sharpe_w,
rho_r, r_min_at_sigma, r_max_at_sigma,
rho_sigma, sd_min_at_r, sd_max_at_r,
gamma_r, r_min_global, r_max_global,
gamma_sigma, sd_min_global, sd_max_global,
gamma_sharpe, sharpe_min_global, sharpe_max_global,
ref_rho_r, ref_rho_sigma,
ref_gamma_r, ref_gamma_sigma, ref_gamma_sharpe,
dissim_w_ref
```

Any key is `None` when not applicable or the required frontier was not computed.

---

### `relative_performance`

```python
rp = fs.relative_performance(cloud, weights,
                          rf=0.0,
                          n_points=1_000_000,  # target lattice size for Q_A / Q_F distribution
                          lattice_k=None,      # explicit barycentric lattice resolution (overrides n_points)
                          method="count",      # "count" → exact lattice dominance counting; "quad" → legacy GL quadrature
                          determine=True,      # deprecated spelling; determine=False ≡ method="count"
                          n_quad=200,          # GL nodes per outer dimension (method="quad" only)
                          reference=False,     # also compute the 6 reference-portfolio columns
                          coarse=False,        # boundary refinement for Q_A/Q_F (method="quad" only)
                          w_ref=None,
                          verbose=False)
```

Evaluates the observed portfolio against the uniform distribution over all long-only portfolios on the probability simplex W\_s = {w ≥ 0, **1**'w = 1}.

**Univariate statistics** (each in [0, 1]; higher = better):

| Key | Definition |
|-----|-----------|
| `P_r_minus` | Pr\_w(r(w) < r\_o) — fraction of simplex beaten in return |
| `P_sigma_plus` | Pr\_w(σ(w) > σ\_o) — fraction of simplex beaten in risk |
| `P_sharpe_minus` | Pr\_w(SR(w) < SR(w\_o)) — fraction of simplex beaten in Sharpe ratio |

**Domination-region statistics:**

| Key | Definition | Interpretation |
|-----|-----------|----------------|
| `A_i` | Pr\_w(r(w) > r\_o **and** σ(w) < σ\_o) | Area of simplex that *dominates* w\_o; 0 on the EF |
| `F_i` | Pr\_w(r(w) < r\_o **and** σ(w) > σ\_o) | Area of simplex that w\_o *dominates* |
| `Q_A` | Pr\_w(A(w) ≥ A(w\_o)) | Upper-tail rank of A\_i; → 1 means w\_o is near the EF |
| `Q_F` | Pr\_w(F(w) ≤ F(w\_o)) | Lower-tail rank of F\_i; → 1 means w\_o dominates most portfolios |

`P_r_minus` and `P_sigma_plus` are computed analytically (exact). `P_sharpe_minus` uses exact closed-form formulas for N ≤ 4 and Gauss-Legendre quadrature for N > 4.

**`method`** selects how `A_i`, `F_i`, `Q_A` and `Q_F` are computed. It changes none of the definitions above.

- **`"count"` (default)** — `A` and `F` are evaluated on the deterministic barycentric lattice itself by exact 2-D dominance counting: one O(M log M) sweep yields both for all M points at once, and `w_o` (which is not a lattice point) is scored against the same lattice. The only error is the lattice's own resolution; it shrinks predictably in `k` and cannot collapse a thin dominating region to a spurious zero. `A(w)` is a function *of* the lattice and `Q_A` a rank *within* it, so numerator and population are one consistent object.
- **`"quad"`** — the legacy iterated Gauss-Legendre path. Its inner 1-D length is exact, but the outer (N−2)-dimensional integrand is non-smooth (kinks plus a compact support boundary), so Gauss-Legendre has no advantage there, and `n_quad` is a *total* node budget spread as `n_quad**(1/(N−2))` per dimension — only 3 nodes per dimension once N ≥ 7. At that resolution thin dominating regions integrate to exactly zero. Retained for validating levels at high `n_quad` and for reproducing earlier results; **not recommended for `Q_A`**.

On the Georgia example the two differ materially: `A_i` 0.000926 (`"quad"`) against 0.000153 (`"count"`, converged), and `Q_A` 0.8820 against 0.9776. `F_i` and `Q_F` agree closely (`Q_F` 0.7778 vs 0.7765) — the quadrature handled `F` well and only `A` broke. Note that a counted `A_i` is quantized to multiples of 1/M, so near-frontier levels carry about two significant figures; the rank statistics do not have this limitation.

**`reference=False`** (default) computes only `w_o`'s own measures — the cheap path. **`reference=True`** additionally computes the 6 reference-portfolio columns (max-return-conditional, min-variance-conditional, global min variance, global max return, max Sharpe, min-D EF allocation), mirroring `absolute_performance`, and adds a `reference_columns` key (a list of per-portfolio dicts) to the returned result. `verbose` only controls printing — it's independent of `reference`. The reference allocations themselves are exact critical-line-algorithm objects under either `method`: a lattice measures a distribution but cannot locate a frontier, so these are never read off the grid. Under `method="count"` each is scored against the same lattice in O(M) and ranked in the same population as `w_o`, so every column of the table is one consistent object; the four that sit on the NW EF by construction return `A_i = 0` on their own rather than by assertion.

**`coarse`** applies to `method="quad"` only, where it is an opt-in alternative to full-lattice evaluation for a single threshold's `Q_A`/`Q_F`: a coarse sub-lattice is classified above/below the threshold first, and only fine lattice points near the resulting contour are evaluated exactly. `True` uses a default coarse resolution derived from the fine lattice; an `int` sets it explicitly; validate a chosen resolution with `validate_coarse_halving` before trusting it for reporting. Under `method="count"` it is unnecessary and ignored (with a notice) — a single sweep already produces `A` and `F` at every lattice point.

**Returns** a dict with keys: `r_w, var_w, sd_w, P_r_minus, P_sigma_plus, P_sharpe_minus, A_i, F_i, Q_A, Q_F`, and (only when `reference=True`) `reference_columns`.

---

### `plot_cloud`

```python
fs.plot_cloud(cloud,
           weights=None,        # observed portfolio — plotted as an open circle
           ref_weights=None,    # benchmark portfolio — plotted as a filled circle
           sd=True,             # x-axis: std dev (True) or variance
           num_points=200,      # curve resolution
           show_assets=True,    # show individual asset markers
           show_targets=True,   # scatter the 6 reference portfolios when weights is given
           xlim=None, ylim=None,
           percent=False,       # multiply axis values by 100
           bw=False,            # black-and-white mode
           lw=2, asset_size=36, target_size=80, target_color="black",
           title_size=None, axis_title_size=None, label_size=None, tick_step=None,
           xtitle=None, ytitle=None,
           show_legend=True, show=True,
           save=None, dpi=150,  # save the figure to a file
           rf=0.0)              # risk-free rate, used to locate the Max Sharpe target
```

Plots the three frontier branches and (optionally) the observed and reference portfolios on a return-vs-risk diagram. Returns `(fig, ax)`.

- **Blue solid** — NW efficient frontier
- **Green dashed** — SW frontier
- **Red dotted** — East (EA) frontier
- **■** — individual assets
- **○** — observed portfolio
- **●** — reference (benchmark) portfolio
- **+** — the 6 reference portfolios (only when `weights` and `show_targets=True`)

> **Known issue:** the `+` target markers (`show_targets=True`) currently render oversized — `target_size` is passed straight through to `markersize`, a linear point-size unit, rather than the area-like unit used for `asset_size`/`target_size` elsewhere via `ax.scatter`. Pass `show_targets=False` until this is fixed, or expect very large crosses.

---

### `q_plot`

```python
fs.q_plot(cloud, weights,
      stat="A",             # "A" | "F" | "return" | "sigma" | "sharpe"
      n_points=1_000_000,   # target lattice size; ignored when lattice_k is given
      lattice_k=None,       # explicit barycentric lattice resolution
      method="count",       # "count" → exact lattice dominance counting; "quad" → legacy GL quadrature
      determine=True,       # deprecated spelling; determine=False ≡ method="count"
      n_quad=200,           # method="quad" only
      rf=0.0,                # only used when stat="sharpe"
      bins=30, width=None,   # bin count, or explicit bin width (overrides bins)
      xlim=None, ylim=None,
      percent=True,          # multiply values by 100 (ignored for "sharpe")
      bw=False, lw=2,
      title_size=None, axis_title_size=None, label_size=None, tick_step=None,
      target_color="black", xtitle=None, ytitle=None,
      show=True, show_legend=True,
      save=None, dpi=150,     # save the figure to a file
      graph=True,             # draw/show/save the histogram
      stats=False,            # export descriptive statistics to an Excel workbook
      stats_save=None,        # workbook path (defaults from save, or "q_plot_stats.xlsx")
      stats_sheet=None)       # sheet name (defaults to "{STAT} distribution")
```

Histograms a portfolio statistic sampled over the same barycentric lattice used by `relative_performance`'s `Q_A`/`Q_F`, drawn as a frequency polygon (a line through each bin's midpoint) rather than bars, with the observed portfolio's own value marked as a vertical line. `stat="A"`/`"F"` histogram the same domination-region measures as `relative_performance`; `"return"`, `"sigma"`, and `"sharpe"` histogram the portfolio's own return, standard deviation, or Sharpe ratio across the simplex. Returns `(fig, ax)`, or `(None, None)` when `graph=False`.

Set `stats=True` to additionally export N, Min, the 10th–90th percentiles (deciles), Max, Mean, Std Dev, Skewness, and excess Kurtosis (normal = 0) to an Excel sheet, computed on the same values shown in the histogram. Calling `q_plot` multiple times with the same `stats_save` path (e.g. once per state in a loop) accumulates each call onto its own sheet in one shared workbook — give each call a distinct `stats_sheet` name to avoid collisions.

---

## Examples

Full worked examples with real data, verbose output, and figures — each replicates results from Bartsch (2026), *Evaluating State Tax Portfolio Performance*:

- **[Example_Georgia.md](Example_Georgia.md)** — 5-asset feasible set (General Sales, Motor Fuels, Individual Income, Corporation Net Income, Other)
- **[Example_Florida.md](Example_Florida.md)** — 7-asset feasible set (General Sales, Motor Fuels, Public Utilities, Motor Vehicles License, Corporation Net Income, Documentary and Stock Transfer, Other)

---

## Technical Notes

### Frontier Computation

The west frontier (NW EF + SW) is computed by a piecewise critical-line algorithm. Starting from the full-asset active set at the global MVP, assets enter and exit the active set at breakpoint returns where a weight hits zero or a shadow price changes sign. Each active set yields an exact parabolic segment in (return, variance) space.

The east frontier is computed by finding, at each return level, the two-asset combination with the *highest* variance. With `east_mode="exact"` all C(N, 2) asset pairs are examined analytically; with `east_mode="grid"` a K-point return grid is used.

### Dissimilarity

Portfolio dissimilarity between w\_a and w\_b is defined as:

> D(w\_a, w\_b) = ½ · Σ |w\_a,i − w\_b,i|

This equals the total rebalancing required to move from one portfolio to the other (one-way turnover). It lies in [0, 1] for long-only portfolios.

### Simplex Geometry

All relative performance measures are computed analytically (no Monte Carlo sampling) for any number of assets N:

- **P\_r\_minus** — exact polytope volume via `scipy.spatial.ConvexHull`.
- **P\_sigma\_plus** and **P\_sharpe\_minus** — Gauss-Legendre quadrature via the Duffy transform. The σ² and Sharpe-ratio conditions each reduce to a quadratic inequality in the innermost simplex coordinate, solved analytically at each quadrature node.
- **A\_i**, **F\_i**, **Q\_A**, **Q\_F** — exact 2-D dominance counting on a deterministic barycentric lattice (`method="count"`, the default). Sorting by return and sweeping with a merge-based counter gives, for every lattice point at once, the exact number of lattice points that strictly dominate it and that it strictly dominates, in O(M log M). The lattice includes the simplex's vertices, edges and faces — where the frontier's extreme allocations actually live — and guarantees coverage at a known scale: no region of volume much above k^−(N−1) can be missed. The error is deterministic bias rather than random variance, so the same inputs always return the same number; it is bounded by refining `k` and watching convergence rather than by a confidence interval. The legacy quadrature path (`method="quad"`) remains available for validating levels — see the `relative_performance` section above.

---

## Citation

> Bartsch, Zachary. 2026. "Portfolio Efficient Frontiers & Diagnostics for Python."  
> Ave Maria University. https://github.com/zacharybartsch/frontier_segments

The worked example above replicates data and results from:

> Bartsch, Zachary. 2026. "Evaluating State Tax Portfolio Performance." Ave Maria University.

---

## License

MIT
