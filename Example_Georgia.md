# Example: Georgia, 2013 Tax Revenue Portfolio

[← Back to README](README.md) · See also: [Example_Florida.md](Example_Florida.md)

Full worked example using Georgia's actual 2013 state tax revenue allocation — five consolidated tax sources (General Sales and Gross Receipts, Motor Fuels, Individual Income, Corporation Net Income, and a residual "Other" category), replicated from Bartsch (2026), *Evaluating State Tax Portfolio Performance* (Tables I–II). Every output block below was produced by actually running the current code against this data — see the [API Reference](README.md#api-reference) for what each argument and return value means.

```python
import numpy as np
import frontier_segments.frontier_segments as fs

mu_GA = np.array([0.04321064, 0.09432582, 0.05994018, 0.04744774, 0.05847559])
Sigma_GA = np.array([
    [ 0.00117476, -0.00005078, -0.00047024, -0.00156685, -0.00228487],
    [-0.00005078,  0.02358429,  0.00088930, -0.00595364,  0.00263467],
    [-0.00047024,  0.00088930,  0.00245991, -0.00055711, -0.00010694],
    [-0.00156685, -0.00595364, -0.00055711,  0.03806298, -0.00113346],
    [-0.00228487,  0.00263467, -0.00010694, -0.00113346,  0.01611126],
])
w_o = np.array([0.27856823, 0.06649115, 0.49736724, 0.04979834, 0.10777504])

cloud = fs.compute_cloud(mu_GA, Sigma_GA, verbose=True)
```

Assets, in order: **T09** General Sales and Gross Receipts Taxes, **T13** Motor Fuels Sales Tax, **T40** Individual Income Taxes, **T41** Corporation Net Income Taxes, **Other**.

**`compute_cloud` verbose output:**

```
N         : 5
r_global  : 0.049239
mu        : [0.04321064 0.09432582 0.05994018 0.04744774 0.05847559]
Sigma     :
[[ 1.174760e-03 -5.078000e-05 -4.702400e-04 -1.566850e-03 -2.284870e-03]
 [-5.078000e-05  2.358429e-02  8.893000e-04 -5.953640e-03  2.634670e-03]
 [-4.702400e-04  8.893000e-04  2.459910e-03 -5.571100e-04 -1.069400e-04]
 [-1.566850e-03 -5.953640e-03 -5.571100e-04  3.806298e-02 -1.133460e-03]
 [-2.284870e-03  2.634670e-03 -1.069400e-04 -1.133460e-03  1.611126e-02]]
chol_L    :
[[ 0.03427477  0.          0.          0.          0.        ]
 [-0.00148156  0.15356463  0.          0.          0.        ]
 [-0.01371971  0.00565868  0.04732503  0.          0.        ]
 [-0.04571438 -0.03921065 -0.02033633  0.1844509   0.        ]
 [-0.06666332  0.0165136  -0.02356019 -0.02175403  0.10181475]]
segments  : 11 total
  active_set           lower_r     upper_r   ef   sw   ea
  -------------------------------------------------------
  (0, 3)              0.043211    0.043393    0    1    0
  (0, 3, 4)           0.043393    0.044985    0    1    0
  (0, 2, 3, 4)        0.044985    0.049076    0    1    0
  (0, 1, 2, 3, 4)     0.049076    0.049239    0    1    0
  (0, 1, 2, 3, 4)     0.049239    0.067958    1    0    0
  (1, 2, 3, 4)        0.067958    0.076171    1    0    0
  (1, 2, 3)           0.076171    0.084663    1    0    0
  (1, 2)              0.084663    0.094326    1    0    0
  (0, 3)              0.043211    0.047448    0    0    1
  (1, 3)              0.047448    0.080324    0    0    1
  (0, 1)              0.080324    0.094326    0    0    1
```

```python
ap = fs.absolute_performance(cloud, w_o, verbose=True)
```

**`absolute_performance` verbose output:**

```
--- absolute_performance ---
  rf = 0.000000
           asset |   w_o    |     Max r|Same sd          Min sd|Same r             Min Var              Max Return             Max Sharpe             EF Min Diss      
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
               r | 0.056786 |  0.057460  +0.000674 |  0.056786  +0.000000 |  0.049239  -0.007547 |  0.094326  +0.037540 |  0.050042  -0.006744 |  0.059341  +0.002555 |
              sd | 0.027903 |  0.027903  +0.000000 |  0.026483  -0.001420 |  0.016950  -0.010953 |  0.153572  +0.125669 |  0.017087  -0.010815 |  0.032081  +0.004178 |
          sharpe | 2.035166 |  2.059305  +0.024139 |  2.144257  +0.109091 |  2.905021  +0.869855 |  0.614213  -1.420952 |  2.928601  +0.893436 |  1.849741  -0.185425 |
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  weights      0 | 0.278568 |  0.339401  +0.060833 |  0.361176  +0.082608 |  0.605161  +0.326593 |  0.000000  -0.278568 |  0.579214  +0.300646 |  0.278568  -0.000000 |
               1 | 0.066491 |  0.113104  +0.046613 |  0.104017  +0.037526 |  0.002198  -0.064293 |  1.000000  +0.933509 |  0.013026  -0.053465 |  0.138491  +0.071999 |
               2 | 0.497367 |  0.414764  -0.082604 |  0.400881  -0.096486 |  0.245330  -0.252037 |  0.000000  -0.497367 |  0.261872  -0.235495 |  0.453547  -0.043820 |
               3 | 0.049798 |  0.045079  -0.004720 |  0.044630  -0.005169 |  0.039601  -0.010197 |  0.000000  -0.049798 |  0.040136  -0.009662 |  0.046332  -0.003466 |
               4 | 0.107775 |  0.087653  -0.020122 |  0.089296  -0.018479 |  0.107710  -0.000065 |  0.000000  -0.107775 |  0.105751  -0.002024 |  0.083062  -0.024713 |
```

```python
qr = fs.quasi_relative_performance(cloud, w_o, verbose=True)
```

**`quasi_relative_performance` verbose output:**

```
--- quasi_relative_performance ---
            stat |   w_o    |     Max r|Same sd          Min sd|Same r             Min Var              Max Return             Max Sharpe             EF Min Diss      
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
           rho_r | 0.949825 |  1.000000  +0.050175 |  1.000000  +0.050175 |  1.000000  +0.050175 |  1.000000  +0.050175 |  1.000000  +0.050175 |  1.000000  +0.050175 |
          rho_sd | 0.988790 |  1.000000  +0.011210 |  1.000000  +0.011210 |  1.000000  +0.011210 |  1.000000  +0.011210 |  1.000000  +0.011210 |  1.000000  +0.011210 |
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
         gamma_r | 0.265589 |  0.278766  +0.013177 |  0.265589  +0.000000 |  0.117941  -0.147647 |  1.000000  +0.734411 |  0.133643  -0.131945 |  0.315579  +0.049990 |
        gamma_sd | 0.938518 |  0.938518  +0.000000 |  0.946487  +0.007969 |  1.000000  +0.061482 |  0.233096  -0.705422 |  0.999228  +0.060709 |  0.915063  -0.023455 |
    gamma_sharpe | 0.667299 |  0.676288  +0.008989 |  0.707923  +0.040624 |  0.991219  +0.323920 |  0.138159  -0.529140 |  1.000000  +0.332701 |  0.598250  -0.069049 |
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
   dissimilarity | 0.000000 |  0.107446            |  0.120134            |  0.326593            |  0.933509            |  0.300646            |  0.071999            |
```

**Reading the table.** Each column is a reference portfolio. Signed deltas show how that portfolio's score differs from `w_o`. The `rho` rows are each 1.0 for every frontier portfolio by construction — they serve as a sanity check that the frontier point was found correctly. The `gamma` rows are unconditional: `gamma_r = 1` only for the single asset with the highest expected return, `gamma_sd = 1` only for the MVP, and `gamma_sharpe = 1` only for the tangency (Max Sharpe) portfolio. Georgia's 2013 allocation scores `gamma_r = 0.266` — its revenue growth was far below what was achievable elsewhere in the feasible set — despite scoring `gamma_sd = 0.939` (risk close to the global minimum) and a moderate `gamma_sharpe = 0.667`, illustrating that low absolute growth need not reflect a policy defect: greater growth was possible, but only at higher volatilities than were realized.

```python
rp = fs.relative_performance(cloud, w_o, reference=True, verbose=True)
```

**`relative_performance` verbose output:**

```
--- relative_performance ---
            stat |   w_o    |     Max r|Same sd          Min sd|Same r             Min Var              Max Return             Max Sharpe             EF Min Diss      
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
       P_r_minus | 0.336172 |  0.378136  +0.041964 |  0.336172  +0.000000 |  0.023505  -0.312667 |  1.000000  +0.663828 |  0.037717  -0.298454 |  0.492899  +0.156727 |
    P_sigma_plus | 0.960539 |  0.960539  +0.000000 |  0.973057  +0.012517 |  1.000000  +0.039461 |  0.001475  -0.959065 |  0.999929  +0.039390 |  0.935979  -0.024560 |
  P_sharpe_minus | 0.971592 |  0.973959  +0.002367 |  0.984606  +0.013015 |  1.000000  +0.028408 |  0.089430  -0.882162 |  1.000000  +0.028408 |  0.953474  -0.018118 |
             A_i | 0.000153 |  0.000000  -0.000153 |  0.000000  -0.000153 |  0.000000  -0.000153 |  0.000000  -0.000153 |  0.000000  -0.000153 |  0.000000  -0.000153 |
             F_i | 0.308075 |  0.348474  +0.040400 |  0.316063  +0.007989 |  0.030059  -0.278016 |  0.002537  -0.305538 |  0.045886  -0.262189 |  0.430293  +0.122219 |
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
             Q_A | 0.977603 |  1.000000  +0.022397 |  1.000000  +0.022397 |  1.000000  +0.022397 |  1.000000  +0.022397 |  1.000000  +0.022397 |  1.000000  +0.022397 |
             Q_F | 0.776498 |  0.843761  +0.067263 |  0.790439  +0.013940 |  0.103260  -0.673238 |  0.013621  -0.762877 |  0.151427  -0.625071 |  0.943331  +0.166832 |
```

`reference=True` picked a lattice resolution of `k=67` (`_simplex_grid: k=67, points=971635`), auto-derived from the default `n_points=1_000_000` for this 5-asset feasible set — matching the resolution reported in Bartsch (2026), Table V. Georgia's 2013 allocation is dominated by only `A_i = 0.015%` of the feasible simplex, and itself dominates `F_i = 30.8%` of it — the 2013 allocation sits almost exactly on the efficient frontier (visible in the plot below), while still being strictly superior to nearly a third of all other feasible tax-source weightings. At this resolution `A_i` is about 149 of the 971,635 lattice points, so read it as two significant figures; `Q_A` and `Q_F` are ranks over the full lattice and carry no such limitation.

```python
fs.plot_cloud(cloud, weights=w_o)
```

**Plot:**

![Markowitz Cloud](Figure_1.png)

```python
fs.q_plot(cloud, weights=w_o, stat="A")
```

**Plot — distribution of `A(w)` over the feasible simplex, with Georgia's 2013 `A_i` marked:**

![Distribution of A(w)](Figure_2.png)

About 0.09% of the simplex has `A(w) = 0` (i.e. also sits on the EF), and that share keeps shrinking as the lattice is refined — it is the genuine efficient set, not an artifact of resolution. The observed allocation's own `A_i ≈ 0.015%` lands at the very left edge of the distribution, consistent with the near-zero value in the table above.

---

[← Back to README](README.md) · See also: [Example_Florida.md](Example_Florida.md)
