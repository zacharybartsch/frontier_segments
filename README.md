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
                          n_points=4_194_304,  # Sobol points (rounded down to a power of 2)
                          method="analytic",   # "analytic" → exact/quadrature P-measures; "sobol" → all from the point set
                          n_quad_p=10_000_000, # GL node budget for P_sigma_plus / P_sharpe_minus
                          reference=False,     # also compute the 6 reference-portfolio columns
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

`P_r_minus` is exact — a polytope volume via `scipy.spatial.ConvexHull`, no quadrature. `P_sigma_plus` and `P_sharpe_minus` reduce to an (N−2)-dimensional integral whose inner condition is a quadratic solved in closed form at each node, and are evaluated by Gauss-Legendre quadrature over that outer simplex.

**`n_quad_p`** is the total outer-node budget for those two, default 10,000,000. They are evaluated once per portfolio rather than once per point, so the budget can be generous. It is a *ceiling*: nodes per dimension is the largest K with K^(N−2) ≤ `n_quad_p`, capped at 64 because convergence is in K and nothing moves past that. The grid is generated in blocks, so memory stays bounded no matter how large K^(N−2) is.

Resolution matters more than it looks, and increasingly with N. Going from the old default of 200 to 10,000,000:

| state | N | nodes/dim | `P_sigma_plus` 200 → 10M | shift |
|---|---|---|---|---|
| Georgia | 5 | 6 → 64 | 0.960539 → 0.963684 | 0.31pp |
| Florida | 7 | 3 → 25 | 0.955791 → 0.957105 | 0.13pp |
| Mississippi | 9 | 3 → 10 | 0.953277 → 0.932211 | **2.1pp** |
| Alabama | 9 | 3 → 10 | 0.864783 → 0.806653 | **5.8pp** |

Past the default there is nothing left to gain: between 10M and 60M nodes the Georgia and Florida values move by about 0.003pp, two orders of magnitude below anything reportable, at twenty times the cost. Cost at the default is well under a second through N=5 and roughly fifteen seconds at N=9.

Convergence is in nodes per dimension, and it arrives early. Four N=7 states, which share the same K at every budget, are all converged by 20,000 nodes (K=7); between 10M and 60M they move by 0.002pp or less. At N=9, K=10 is likewise converged -- New Jersey sits at 0.2297 and Alabama at 0.807 across a threefold range of K, with last steps of 0.07pp and 0.12pp.

The entire correction therefore happened between K=3 and K=7. Three Gauss-Legendre nodes cannot locate a curved level set in five or more dimensions at all, so the old default was not under-resolved so much as arbitrary, which is why its error ranged from 0.4pp (Georgia) to 33pp (Illinois) with no relation to N. What governed the size was where the observed portfolio sat in the distribution: error in the measure is roughly the error in locating the boundary times the density of mass there. Georgia's sigma_o sits in the far left tail at density 0.80 and barely moved; Illinois and Alabama sit at density 5.7 and 6.3 and moved by 33pp and 19pp.

Lattice counting, used for the dominance measures below, is the cruder estimator for these particular regions -- at N=9 it is still falling steeply at k=20 where the quadrature has settled. The two methods are each applied where they win: Gauss-Legendre where the region has one smooth boundary, counting where it does not.

**`A_i`, `F_i`, `Q_A` and `Q_F` are computed on a Sobol point set** — an unscrambled, low-discrepancy sequence mapped to the simplex by sorted spacings. It is *not* Monte Carlo: there is no seed and no randomness, and the same N and point count give bit-identical output forever.

Why Sobol rather than a uniform lattice. `A(w)` depends on `w` only through `(r, σ)`, so the integrand is **two-dimensional however many assets there are** — the integral is over the joint density of `(r, σ)` induced by `w ~ Uniform(Δ)`. Sobol's low-dimensional projections are well distributed by the (t,m,s)-net property; a uniform lattice's are not, and an N-asset grid projected onto the `(r, σ)` plane clumps badly. A lattice's spacing also improves only as M^(−1/(N−1)), so doubling the points buys 9% at N=9.

Measured against values known independently — exact convex-hull `P_r_minus`, converged quadrature `P_sigma_plus` — at matched or smaller point counts:

| state | N | lattice (~900k points) | Sobol (262k points) |
|---|---|---|---|
| GA | 5 | +0.0066 | −0.00008 |
| LA | 7 | +0.0146 | +0.00015 |
| IL | 8 | −0.0274 | −0.00045 |
| NJ | 9 | **−0.0167 / +0.1504** | +0.00012 / +0.00039 |

New Jersey's lattice `P_sigma_plus` was wrong by 15 percentage points; Sobol reaches 0.0004 with 3.5× fewer points and runs 20× faster. The lattice's failure is structural: at N=9, k=16 it puts **99.1% of its points on the simplex boundary**, a set of measure zero in the continuum.

**`method`** controls only the three `P` measures. The default `"analytic"` keeps them in the forms the literature supports: `P_r_minus` is an exact convex-hull polytope volume, and `P_sigma_plus` / `P_sharpe_minus` are Gauss-Legendre over an analytic reduction, both converged at high node counts. `method="sobol"` recomputes all three from the point set instead, as a cross-check. `A_i`, `F_i`, `Q_A` and `Q_F` come from Sobol either way.

**`n_points`** is rounded down to a power of two (Sobol's net property holds on full 2^m blocks); the default is 2^22 = 4,194,304. At that size, measured across N = 5 to 9: levels converge to 10⁻⁵–10⁻⁴ and `Q_A` to 10⁻⁴–6×10⁻⁴, at 70–105 s per state. 2^24 buys roughly another 5× on `Q_A` for 4–5× the time.

**Error reporting.** Every call returns `M` and a `convergence` dict holding |stat(M) − stat(M/4)| for each statistic. The first M/4 Sobol points are a strict prefix of the first M, so the coarse run costs an extra sweep but no extra point generation — about 25%. This is a measured step, not a bound: there is no sampling variance to quote, because nothing is sampled. Also returned is `sobol_gauge`, which compares the point set's `P_r_minus` against its exactly known value — a directly measured error on a comparable region.


**`reference=False`** (default) computes only `w_o`'s own measures — the cheap path. **`reference=True`** additionally computes the 6 reference-portfolio columns (max-return-conditional, min-variance-conditional, global min variance, global max return, max Sharpe, min-D EF allocation), mirroring `absolute_performance`, and adds a `reference_columns` key (a list of per-portfolio dicts) to the returned result. `verbose` only controls printing — it's independent of `reference`. The reference allocations themselves are exact critical-line-algorithm objects: a point set measures a distribution but cannot locate a frontier, so these are never read off it. Each is scored against the same Sobol set in O(M) and ranked in the same population as `w_o`, so every column of the table is one consistent object; the four that sit on the NW EF by construction return `A_i = 0` on their own rather than by assertion.

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
      n_points=4_194_304,   # Sobol points (rounded down to a power of 2)
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

Histograms a portfolio statistic over the same Sobol point set used by `relative_performance`'s `Q_A`/`Q_F`, drawn as a frequency polygon (a line through each bin's midpoint) rather than bars, with the observed portfolio's own value marked as a vertical line. `stat="A"`/`"F"` histogram the same domination-region measures as `relative_performance`; `"return"`, `"sigma"`, and `"sharpe"` histogram the portfolio's own return, standard deviation, or Sharpe ratio across the simplex. Returns `(fig, ax)`, or `(None, None)` when `graph=False`.

Set `stats=True` to additionally export N, Min, the 10th–90th percentiles (deciles), Max, Mean, Std Dev, Skewness, and excess Kurtosis (normal = 0) to an Excel sheet, computed on the same values shown in the histogram. Calling `q_plot` multiple times with the same `stats_save` path (e.g. once per state in a loop) accumulates each call onto its own sheet in one shared workbook — give each call a distinct `stats_sheet` name to avoid collisions.

---

## Examples

Full worked examples with real data, verbose output, and figures — each replicates results from Bartsch (2026), *Evaluating State Tax Portfolio Performance*:

- **[Example_Georgia.md](Example_Georgia.md)** — 5-asset feasible set (General Sales, Motor Fuels, Individual Income, Corporation Net Income, Other)
- **[Example_Florida.md](Example_Florida.md)** — 7-asset feasible set (General Sales, Motor Fuels, Public Utilities, Motor Vehicles License, Corporation Net Income, Documentary and Stock Transfer, Other)

For the full pipeline — raw quarterly tax revenue in, every frontier and performance measure out — see **[examples/](examples/README.md)**. Two runnable scripts and a trimmed Census QTAX extract reproduce all thirteen state-windows in the paper:

```bash
pip install -e .
python examples/applied_methods.py
```

`examples/tax_portfolio.py` builds the μ, Σ and w arrays from the revenue panel and can be imported on its own; `examples/applied_methods.py` runs the analysis. Outputs go to `examples/output/`.

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
- **P\_sigma\_plus** and **P\_sharpe\_minus** — Gauss-Legendre quadrature via the Duffy transform. The σ² and Sharpe-ratio conditions each reduce to a quadratic inequality in the innermost simplex coordinate, solved analytically at each quadrature node. Each region is bounded by a single smooth surface cut by the simplex, so Gauss-Legendre converges well here — unlike the dominance regions below, which are intersections of a half-space with an ellipsoid interior and carry corners, kinks, and arbitrarily thin slivers. Same reduction, different integrand geometry, different right tool.
- **A\_i**, **F\_i**, **Q\_A**, **Q\_F** — exact 2-D dominance counting over an unscrambled Sobol point set. Sorting by return and sweeping with a merge-based counter gives, for every point at once, the exact number of points that strictly dominate it and that it strictly dominates, in O(M log M). The counts are exact with respect to the point set; the only error is how well that set represents the simplex, and because `A` depends on `w` only through `(r, σ)` the relevant discrepancy is Sobol's 2-dimensional one rather than its N-dimensional one. Deterministic throughout — no seed, no sampling variance, bit-identical on every run — so error is reported as a measured convergence step rather than a confidence interval.

### Approaches that were tried and rejected

Recorded because each cost real measurement to rule out.

**Quadrature for A and F.** The inner 1-D length is exact, but the outer
(N−2)-dimensional integrand is non-smooth — kinks plus a compact support
boundary — so Gauss-Legendre has no advantage, and `n_quad` spread as a total
budget gave 3 nodes per dimension at N ≥ 7. At that resolution thin dominating
regions integrate to exactly zero: 10% of Georgia's simplex was reported as
sitting on the efficient frontier.

**A uniform barycentric lattice.** Spacing improves only as M^(−1/(N−1)), so at
N=9 doubling the points buys 9%. Worse, at N=9 and k=16 it puts 99.1% of its
points on the simplex boundary — a set of measure zero in the continuum — and
boundary portfolios are less diversified, so every region defined by σ > σ_o is
biased. New Jersey's `P_sigma_plus` was wrong by 15 percentage points.

**Evaluation-reduction schemes.** Three were built or costed against the
vectorized sweep and none paid:

- *Line-wise closed-form classification.* Correct and validated exactly against
  brute force, but slower than the sweep it would replace. Per-line cost in
  Python is ~25 µs against ~1 µs per point for NumPy, and the ceiling is
  points-per-line, which is 2.5 at N=9.
- *Face certification.* Reaches 0.000–0.223% of points (median 0.03%). A face
  certifies only when every portfolio on it qualifies, which needs every
  individual asset to be less volatile than the diversified observed portfolio.
- *Branch-and-bound over the composition tree.* Inherits both failures — looser
  bounds than the exact face tests, same points-per-line ceiling.

**Staggered or offset lattices, and unions with midpoints.** For `A_o` alone,
averaging offset grids is sound. For `Q_A` it is not: `A` is defined relative to
the point set, so shifting changes both the population and the function being
ranked. A lattice unioned with its midpoints is just the lattice at 2k, which
costs 2^(N−1) times more points — 104× at N=9.

---

## Citation

> Bartsch, Zachary. 2026. "Portfolio Efficient Frontiers & Diagnostics for Python."  
> Ave Maria University. https://github.com/zacharybartsch/frontier_segments

The worked example above replicates data and results from:

> Bartsch, Zachary. 2026. "Evaluating State Tax Portfolio Performance." Ave Maria University.

---

## License

MIT
