# Porftolio Efficient Frontiers & Diagnostics for Python
## Version 0.2 (2025-11-25)
This version is a beta release. If you find an error, please tell me about it. 


# Installation from your command shell or terminal:
```
py -m pip install git+https://github.com/zacharybartsch/frontier_segments.git
```
or
```
python -m pip install git+https://github.com/zacharybartsch/frontier_segments.git
```
# Python Import
```
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from frontier_segments import compute_frontiers
```


# Authors
- Zachary Bartsch (Ave Maria University), *maintainer*
# License and Citation
You are free to use this package under the terms of its [license](LICENSE). If you use it, please cite the software package in your work:
- Bartsch, Zachary. 2025. “Porftolio Efficient Frontiers & Diagnostics for Python.” [Software] Available at https://github.com/zacharybartsch/frontier_segments/tree/main

# Introduction
Diversification is a fundamental concept in finance. A key result is that a set of assets, or a portfolio, can have a lower variance than its components. This works because variance is nonlinearly related to the proportions or weights that assets consistitute in a portfolio. Portfolio Returns, in contrast, are linear combinations of the asset returns. So, a portfolio's return never exceeds or fails to meet the returns of its assets. Beloware the return and covariance matrices for a portfolio of assets. Descriptions can be found here:
- https://economistwritingeveryday.com/2025/11/07/optimal-portfolio-weights/
- https://economistwritingeveryday.com/2025/11/14/portfolio-efficient-frontier/

The below function calculates the 3 possible portfolio frontiers and provides diagnostics to evaluate a specific portfolio.

# Syntax
```
compute_frontiers(mu, Sigma, weights=w,
      ef_frontier=True, low_frontier=True, east_mode=True, east_K=200,
      verbose=False, sd=True, w_deltas=True, graph=True)
```
# Arguments
- **mu**: A list of lists or a numpy array of N returns.
- **sigma**: A list of lists or a numpy array of the asset covariance matrix.
-**weights (optional)**: A list of lists or a numpy array of the asset weights matrix that describes a portfolio. Needed for portfolio diagnostics.
- **ef_frontier (optional)**: Accepts True/False to toggle the variance-minimizing frontier for returns that occur above the global variance-minimzing return. Default value is True.
- **low_frontier (optional)**: Accepts True/False to toggle the variance-minimizing frontier for returns that occur below the global variance-minimzing return. Default value is True.
- **east_mode (optional)**:  Accepts True/grid/False to toggle the variance-maximizing frontier. A value of True calculates the precise frontier with 2^N possible segments. A value of ```grid```  is used with ```east_K``` do define an approximate east frontier. Default value is True.
- **east_K (optional)**: Accepts integer, eg 200, to estimate the east frontier with a resolution defined by the user.
- **verbose (optional)**: Prints the steps and defines the underlying parabolas of the frontier segments.  Default value is False.
- **sd (optional)**: Replaces variance with standard deviation in graph and the specific portfolio diagnostics.  Default value is False.
- **w_deltas (optional)**: Requires ```weights```. Describes 1)  the frontier point with the maximum return at the same variance. Also includes the difference and the necessary change in weights to achieve it. 2) the frontier point with the minimum variance at the same return. Also includes the difference and the necessary change in weights to achieve it. 3) the nearest efficient frontier point (rarely used). Also includes the difference and the necessary change in weights to achieve it. 4) the  efficient frontier point achievable with the minimum change in weights. Also includes the difference and the necessary change in weights to achieve it. 
- **graph (optional)**: Accepts True/False to toggle the graph. Default value is True.

# Replication
There are three assets that are described by their returns and covariance matrices. A specific portfolio described by w is evaluated.
```
mu = np.array([0.2044, 0.1579, 0.095])
Sigma = np.array([
    [0.00024086, 0.00005642, 0.00008801],
    [0.00005642, 0.00011336, 0.00006400],
    [0.00008801, 0.00006400, 0.00015271]
])
w = np.array([0.6,0.00,0.4])

out = compute_frontiers(mu, Sigma, weights=w,ef_frontier=False, low_frontier=True, east_mode=True, east_K=200, verbose=False, sd=True, w_deltas=True, graph=True)

```
**Output:**
<img src="https://github.com/user-attachments/assets/a3982dc6-b737-4f5c-9251-83ffa6f3aa34"
     align="right"
     width="300">

```
=== Portfolio Diagnostics ===
Portfolio r_w = 0.16064
Portfolio sd_w = 0.012384990916427835
Portfolio w  = [0.6, 0.0, 0.4]

1) Frontier at same variance (same sd as w):
   r_frontier      = 0.19075250864468404
   r_diff          = 0.03011250864468404
   dissimilarity D = 0.3999999999999987

2) Frontier at same return r_w:
   frontier source = EF
   sd_frontier     = 0.009714263423226663
   sd_diff         = 0.002670727493201172
   dissimilarity D = 0.6395367908104868

3) Nearest EF point in (r, sd) space:
   r_ef            = 0.16069641602984358
   sd_ef           = 0.009715453211638733
   r_diff          = 5.641602984357563e-05
   sd_diff         = -0.0026695377047891017
   distance        = 0.0026701337655095064
   dissimilarity D = 0.6397796590869596

4) Closest EF point in WEIGHT space (min D):
   r_ef            = 0.1859313443141879
   sd_ef           = 0.011507702823987553
   r_diff          = 0.02529134431418789
   sd_diff         = -0.0008772880924402815
   dissimilarity D = 0.3999999999999987

=== Weight Details (w_deltas=True) ===

[1] w_f & Δw to same-variance frontier:
   w_f: [0.7065055622512704, 0.29349443774872697, 0.0]
   Δw:  [0.10650556225127039, 0.29349443774872697, -0.4]

[2] w_f & Δw to same-return frontier:
   w_f: [0.23229557457057015, 0.6395367908104864, 0.12816763461894287]
   Δw:  [-0.3677044254294298, 0.6395367908104864, -0.27183236538105715]

[3] w_f & Δw to nearest EF (r,sd) portfolio:
   w_f: [0.2326716222419909, 0.6397796590869593, 0.12754871867104933]
   Δw:  [-0.3673283777580091, 0.6397796590869593, -0.2724512813289507]

[4] w_f & Δw to closest-in-weights EF portfolio:
   w_f: [0.6028246089072669, 0.3971753910927305, 0.0]
   Δw:  [0.002824608907266879, 0.3971753910927305, -0.4]
============================
```
Diagnostics 1-4 provide the frontier details, the difference from the portfolio specified by ```weights```, the total change in weights necessary to achieve the point.
Weight Details 1-4 provide the weights at each corresponding diagnostic and how each asset's weight would need to change in order to achieve the point.
The graph prints last and is generated even when specific portfolio weights are omitted.

# What's Next?
- similar processes can be used to compare the performance of portfolios of different assets.
- Auto retrieving asset information. This might be a premium option that would allow the user to either generate the covariance and return matrices from a dataframe or file, or to simply list the asset details for the function to retrieve directly.
