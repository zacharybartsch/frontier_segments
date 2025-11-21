"""
frontier_segments.py
By Zachary Bartsch
2025-11-21

Compute long-only Markowitz frontier segments (EF + low frontier)
for an arbitrary covariance matrix and expected returns vector.

Main entry:
    frontier_segments(mu, sigma, verbose=False, graph=False)

to load:
    from  frontier_segments import frontier_segments
"""

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt


def frontier_segments(mu, sigma, verbose=False, graph=False):
    """
    Parameters
    ----------
    mu : array-like, shape (N,)
        Expected returns for N assets.
    sigma : array-like, shape (N, N)
        Covariance matrix for N assets.
    verbose : bool, optional
        If True, print intermediate step outputs.
    graph : bool, optional
        If True, plot:
          - EF frontier segments in blue
          - Other frontier segments in red
          - Global minimum-variance point in green
          - Asset (return, variance) points in black squares

    Returns
    -------
    segments : np.ndarray, shape (K, 10)
        Columns:
          0: parabola_idx
          1: lower_r
          2: upper_r
          3: feasible (1.0)
          4: ef_frontier
          5: low_frontier
          6: a_scaled
          7: b_scaled
          8: c_scaled
          9: d_scaled
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)   # ensure 1D vector
    sigma = np.asarray(sigma, dtype=float)         # ensure 2D array

    N = mu.shape[0]
    if sigma.shape != (N, N):
        raise ValueError(
            f"sigma must be an {N}x{N} covariance matrix; got {sigma.shape}"
        )    # -------------------------------------------------------
    # STEP 1: Coerce inputs
    # -------------------------------------------------------
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    N = len(mu)
    assert sigma.shape == (N, N), "sigma must be N x N"

    eps = 1e-12

    # =======================================================
    # STEP 2: Unconstrained parabolas for all subsets (size ≥ 2)
    #   A0 = μ'Σ⁻¹μ
    #   B0 = 1'Σ⁻¹μ
    #   C0 = 1'Σ⁻¹1
    #   Δ  = A0*C0 - B0²
    #   σ²(r) = (C0 r² - 2 B0 r + A0) / Δ
    #   Normalized:
    #      An = A0 / Δ, Bn = (2 B0) / Δ, Cn = C0 / Δ
    #      σ²(r) = Cn r² - Bn r + An
    # =======================================================

    parabolas = []
    idx = 1

    for r in range(2, N + 1):  # subsets of size ≥ 2
        for subset in itertools.combinations(range(N), r):

            subset = list(subset)
            mu_s = mu[subset]
            sigma_s = sigma[np.ix_(subset, subset)]

            inv_sigma_s = np.linalg.inv(sigma_s)
            ones = np.ones(len(subset))

            A0 = mu_s @ inv_sigma_s @ mu_s
            B0 = ones @ inv_sigma_s @ mu_s
            C0 = ones @ inv_sigma_s @ ones
            Delta0 = A0 * C0 - B0 ** 2
            if abs(Delta0) < eps:
                continue  # degenerate subset; skip

            An = A0 / Delta0
            Bn = (2.0 * B0) / Delta0
            Cn = C0 / Delta0

            parabolas.append({
                "idx": idx,
                "subset": subset,
                "An": An, "Bn": Bn, "Cn": Cn,
                "mu_s": mu_s,
                "sigma_s": sigma_s,
                "inv_sigma": inv_sigma_s,
                "ones": ones,
                "A0": A0, "B0": B0, "C0": C0, "Delta0": Delta0
            })

            idx += 1

    if verbose:
        print("=== STEP 2: PARABOLA COEFFICIENTS ===")
        print("idx | subset | A0 | B0 | C0 | Delta0")
        for p in parabolas:
            print(p["idx"], p["subset"], p["A0"], p["B0"], p["C0"], p["Delta0"])

    if not parabolas:
        return np.zeros((0, 10))

    # =======================================================
    # STEP 3: Intersections, long-only domains, global min-var
    # =======================================================

    # 3a: intersections with filters

    # Build (idx, a, b, c) for y = a r^2 + b r + c
    coeffs = []
    for p in parabolas:
        a = p["Cn"]          # Cn = C0/Δ
        b = -p["Bn"]         # -Bn = -2B0/Δ
        c = p["An"]          # An = A0/Δ
        coeffs.append({"idx": p["idx"], "a": a, "b": b, "c": c})

    parabola_by_idx = {p["idx"]: p for p in parabolas}

    # subset return ranges
    subset_ranges_step3 = {}
    for p in parabolas:
        idx_p = p["idx"]
        sub_mu = mu[p["subset"]]
        r_min_p = float(np.min(sub_mu))
        r_max_p = float(np.max(sub_mu))
        subset_ranges_step3[idx_p] = (r_min_p, r_max_p)

    # variance ceiling (long-only)
    max_asset_var = float(np.max(np.diag(sigma)))

    intersections = {p["idx"]: [] for p in parabolas}
    eps_disc = 1e-12
    tol_var = 1e-12

    def maybe_store_root(x, idx1, idx2):
        rmin1, rmax1 = subset_ranges_step3[idx1]
        rmin2, rmax2 = subset_ranges_step3[idx2]

        rmin_overlap = max(rmin1, rmin2)
        rmax_overlap = min(rmax1, rmax2)

        if x < rmin_overlap - eps or x > rmax_overlap + eps:
            return

        p1 = parabola_by_idx[idx1]
        An = p1["An"]
        Bn = p1["Bn"]
        Cn = p1["Cn"]
        var_x = Cn * x * x - Bn * x + An

        if var_x > max_asset_var + tol_var:
            return

        intersections[idx1].append(x)
        intersections[idx2].append(x)

    for i in range(len(coeffs)):
        for j in range(i + 1, len(coeffs)):
            p1, p2 = coeffs[i], coeffs[j]
            idx1, idx2 = p1["idx"], p2["idx"]

            A = p1["a"] - p2["a"]
            B = p1["b"] - p2["b"]
            C = p1["c"] - p2["c"]

            if abs(A) < eps:
                if abs(B) < eps:
                    continue
                x = -C / B
                maybe_store_root(x, idx1, idx2)
            else:
                disc = B * B - 4 * A * C
                if disc < -eps_disc:
                    continue
                elif abs(disc) <= eps_disc:
                    x = -B / (2.0 * A)
                    maybe_store_root(x, idx1, idx2)
                else:
                    rdisc = math.sqrt(max(disc, 0.0))
                    x1 = (-B + rdisc) / (2.0 * A)
                    x2 = (-B - rdisc) / (2.0 * A)
                    maybe_store_root(x1, idx1, idx2)
                    maybe_store_root(x2, idx1, idx2)

    # 3b: build [r_min, r_max] segments (pre-long-only)
    segments = []
    for p in parabolas:
        idx_p = p["idx"]
        r_min_p, r_max_p = subset_ranges_step3[idx_p]

        xs = sorted(set(intersections[idx_p]))
        xs = [x for x in xs if (r_min_p - eps) <= x <= (r_max_p + eps)]

        bounds = [r_min_p] + xs + [r_max_p]
        for k in range(len(bounds) - 1):
            lower = bounds[k]
            upper = bounds[k + 1]
            feasible = 1.0
            segments.append([idx_p, lower, upper, feasible])

    segments = np.array(segments, dtype=float)

    # 3c: enforce long-only weights by trimming to r where w(r) >= 0

    tol_w = 1e-12
    long_only_ranges = {}

    for p in parabolas:
        idx_p = p["idx"]
        mu_s = p["mu_s"]
        ones = p["ones"]
        inv_sigma_s = p["inv_sigma"]
        A0 = p["A0"]
        B0 = p["B0"]
        C0 = p["C0"]
        Delta0 = p["Delta0"]

        v1 = (C0 * mu_s - B0 * ones) / Delta0
        v0 = (-B0 * mu_s + A0 * ones) / Delta0
        beta = inv_sigma_s @ v1
        alpha = inv_sigma_s @ v0

        r_lo = -np.inf
        r_hi = np.inf
        infeasible_all = False

        for a_i, b_i in zip(alpha, beta):
            if abs(b_i) < tol_w:
                if a_i < -tol_w:
                    infeasible_all = True
                    break
                else:
                    continue
            r_zero = -a_i / b_i
            if b_i > 0:
                r_lo = max(r_lo, r_zero)
            else:
                r_hi = min(r_hi, r_zero)

        if infeasible_all or r_hi <= r_lo - eps:
            long_only_ranges[idx_p] = (np.inf, -np.inf)
        else:
            long_only_ranges[idx_p] = (r_lo, r_hi)

    pruned_segments = []
    for row in segments:
        idx_p, lower_r, upper_r, feasible = row
        idx_p = int(idx_p)
        r_lo_long, r_hi_long = long_only_ranges[idx_p]

        if r_hi_long <= r_lo_long - eps:
            continue

        new_lower = max(lower_r, r_lo_long)
        new_upper = min(upper_r, r_hi_long)

        if new_upper <= new_lower + eps:
            continue

        pruned_segments.append([idx_p, new_lower, new_upper, 1.0])

    segments = np.array(pruned_segments, dtype=float)

    # 3d: global min-variance and split the segment that contains it

    global_min_var = float('inf')
    global_min_r = None
    global_min_seg_idx = None

    for i in range(len(segments)):
        idx_p, lower_r, upper_r, feasible = segments[i]
        idx_p = int(idx_p)
        if feasible == 0.0:
            continue

        p = parabola_by_idx[idx_p]
        An = p["An"]
        Bn = p["Bn"]
        Cn = p["Cn"]

        r_star = Bn / (2.0 * Cn)
        r_seg = max(min(r_star, upper_r), lower_r)

        var_seg = Cn * r_seg * r_seg - Bn * r_seg + An

        if var_seg < global_min_var:
            global_min_var = var_seg
            global_min_r = r_seg
            global_min_seg_idx = i

    new_segments = []
    for i in range(len(segments)):
        idx_p, lower_r, upper_r, feasible = segments[i]
        if i != global_min_seg_idx:
            new_segments.append([idx_p, lower_r, upper_r, feasible])
            continue

        if global_min_r is None:
            new_segments.append([idx_p, lower_r, upper_r, feasible])
            continue

        if global_min_r <= lower_r + eps or global_min_r >= upper_r - eps:
            new_segments.append([idx_p, lower_r, upper_r, feasible])
        else:
            new_segments.append([idx_p, lower_r, global_min_r, feasible])
            new_segments.append([idx_p, global_min_r, upper_r, feasible])

    segments = np.array(new_segments, dtype=float)

    if verbose:
        print("\n=== STEP 3: SEGMENTS AFTER LONG-ONLY + MIN-VAR SPLIT ===")
        print("parabola_idx | lower_r | upper_r | feasible")
        for row in segments:
            print(int(row[0]), row[1], row[2], int(row[3]))

    # =======================================================
    # STEP 4 (optimized): EF and low-frontier flags
    # =======================================================

    num_segments = len(segments)
    if num_segments == 0:
        return np.zeros((0, 10))

    seg_idx_arr = segments[:, 0].astype(int)
    seg_low_arr = segments[:, 1]
    seg_high_arr = segments[:, 2]
    seg_feas_arr = (segments[:, 3] == 1.0)
    r_mid_arr = 0.5 * (seg_low_arr + seg_high_arr)

    max_idx = max(seg_idx_arr)
    An_arr = np.zeros(max_idx + 1)
    Bn_arr = np.zeros(max_idx + 1)
    Cn_arr = np.zeros(max_idx + 1)
    for p in parabolas:
        i = p["idx"]
        An_arr[i] = p["An"]
        Bn_arr[i] = p["Bn"]
        Cn_arr[i] = p["Cn"]

    ef_flags = np.zeros(num_segments, dtype=float)
    low_flags = np.zeros(num_segments, dtype=float)
    tol = 1e-10

    for i in range(num_segments):

        if not seg_feas_arr[i]:
            continue

        r_mid = r_mid_arr[i]

        cover_mask = (
            (seg_feas_arr) &
            (seg_low_arr - eps <= r_mid) &
            (r_mid <= seg_high_arr + eps)
        )

        if not np.any(cover_mask):
            continue

        idx_qs = seg_idx_arr[cover_mask]
        vars_q = Cn_arr[idx_qs] * r_mid * r_mid - Bn_arr[idx_qs] * r_mid + An_arr[idx_qs]

        v_min = vars_q.min()
        v_max = vars_q.max()

        covered_indices = np.where(cover_mask)[0]
        below_min = (global_min_r is not None) and (r_mid < global_min_r - eps)

        for k, v in zip(covered_indices, vars_q):

            if (not below_min) and abs(v - v_min) <= tol:
                ef_flags[k] = 1.0

            if below_min:
                if abs(v - v_min) <= tol or abs(v - v_max) <= tol:
                    low_flags[k] = 1.0
            else:
                if abs(v - v_max) <= tol:
                    low_flags[k] = 1.0

    segments = np.hstack([
        segments,
        ef_flags.reshape(-1, 1),
        low_flags.reshape(-1, 1)
    ])

    if verbose:
        print("\n=== STEP 4: FRONTIER SEGMENTS (RAW) ===")
        print("parabola_idx | lower_r | upper_r | feasible | ef_frontier | low_frontier")
        for row in segments:
            print(int(row[0]), row[1], row[2], int(row[3]), int(row[4]), int(row[5]))

    # =======================================================
    # STEP 5: (unchanged name) – merge & keep only frontier
    # =======================================================

    # (We called it Step 6 earlier; keep that numbering here.)

    # A) keep only segments that are on some frontier
    mask_frontier = (segments[:, 4] == 1.0) | (segments[:, 5] == 1.0)
    frontier_segments = segments[mask_frontier]

    if len(frontier_segments) == 0:
        if verbose:
            print("\n=== STEP 5/6: NO FRONTIER SEGMENTS FOUND ===")
        # add dummy a,b,c,d columns
        return np.zeros((0, 10))

    # Normalize intervals and drop degenerate ones
    eps_zero = 1e-12
    normalized = []
    for row in frontier_segments:
        idx_p, low, high, feas, ef, lowf = row
        low = float(low)
        high = float(high)
        if high < low:
            low, high = high, low
        if high - low < eps_zero:
            continue
        normalized.append([idx_p, low, high, feas, ef, lowf])

    if not normalized:
        if verbose:
            print("\n=== STEP 5/6: NO NON-DEGENERATE FRONTIER SEGMENTS ===")
        return np.zeros((0, 10))

    frontier_segments = np.array(normalized, dtype=float)

    # Sort by (parabola_idx, lower_r)
    order = np.lexsort((frontier_segments[:, 1], frontier_segments[:, 0]))
    frontier_segments = frontier_segments[order]

    merged = []
    eps_merge = 1e-10

    i = 0
    n = len(frontier_segments)
    while i < n:
        current = frontier_segments[i].copy()
        j = i + 1
        while j < n:
            row = frontier_segments[j]
            same_idx = int(row[0]) == int(current[0])
            same_ef = row[4] == current[4]
            same_low = row[5] == current[5]
            contiguous = abs(row[1] - current[2]) <= eps_merge

            if same_idx and same_ef and same_low and contiguous:
                current[2] = row[2]
                j += 1
            else:
                break

        merged.append(current)
        i = j

    segments = np.array(merged, dtype=float)

    # Append scaled a,b,c,d

    seg_idx_arr = segments[:, 0].astype(int)

    max_idx = max(p["idx"] for p in parabolas)
    An_arr = np.zeros(max_idx + 1)
    Bn_arr = np.zeros(max_idx + 1)
    Cn_arr = np.zeros(max_idx + 1)
    for p in parabolas:
        i = p["idx"]
        An_arr[i] = p["An"]
        Bn_arr[i] = p["Bn"]
        Cn_arr[i] = p["Cn"]

    a_scaled = Cn_arr[seg_idx_arr]          # a = Cn
    b_scaled = -Bn_arr[seg_idx_arr]         # b = -Bn
    c_scaled = An_arr[seg_idx_arr]          # c = An
    d_scaled = np.ones_like(a_scaled)       # d = 1

    segments = np.hstack([
        segments,
        a_scaled.reshape(-1, 1),
        b_scaled.reshape(-1, 1),
        c_scaled.reshape(-1, 1),
        d_scaled.reshape(-1, 1),
    ])

    if verbose:
        print("\n=== STEP 6: MERGED FRONTIER SEGMENTS ===")
        print("parabola_idx | lower_r | upper_r | feasible | "
              "ef_frontier | low_frontier | a_scaled | b_scaled | c_scaled | d_scaled")
        for row in segments:
            print(int(row[0]), row[1], row[2],
                  int(row[3]), int(row[4]), int(row[5]),
                  row[6], row[7], row[8], row[9])

    # =======================================================
    # Optional graph (transposed: variance on x-axis, return on y-axis)
    # =======================================================
    if graph and segments.size > 0:
        fig, ax = plt.subplots()

        # To build a clean legend later
        ef_line = None
        low_line = None

        # Plot frontier segments
        for row in segments:
            idx_p, r_low, r_high, feas, ef, lowf, a_s, b_s, c_s, d_s = row

            r_vals  = np.linspace(r_low, r_high, 100)
            var_vals = a_s * r_vals**2 + b_s * r_vals + c_s

            # Transposed: x = variance, y = return
            if ef == 1.0:
                ef_line = ax.plot(var_vals, r_vals, color='b', linewidth=2)[0]
            else:
                low_line = ax.plot(var_vals, r_vals, color='r', linewidth=2)[0]

        # Global minimum-variance point (green)
        if global_min_r is not None and np.isfinite(global_min_var):
            ax.scatter(
                [global_min_var], [global_min_r],
                color='g', marker='o', s=60,
                label="Global minimum variance"
            )

        # Asset points in black squares
        asset_vars = np.diag(sigma)
        ax.scatter(
            asset_vars, mu,
            color='k', marker='s', s=40,
            label="Assets"
        )

        # Construct legend with colored line samples
        legend_handles = []
        legend_labels  = []

        if ef_line is not None:
            legend_handles.append(ef_line)
            legend_labels.append("Efficient Frontier (blue)")

        if low_line is not None:
            legend_handles.append(low_line)
            legend_labels.append("Cloud Border (red)")

        # Always include global min + assets (only once)
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)

        ax.legend(
            legend_handles,
            legend_labels,
            loc='lower right',
            frameon=True
        )

        ax.set_xlabel("Variance $\sigma^2$")
        ax.set_ylabel("Return r")
        ax.set_title("Long-only Frontier (variance on x-axis)")

        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return segments








# Simple example usage
if __name__ == "__main__":
    mu_ex = np.array([0.2044, 0.1579, 0.095])
    sigma_ex = np.array([
        [0.00024086, 0.00005642, 0.00008801],
        [0.00005642, 0.00011336, 0.00006400],
        [0.00008801, 0.00006400, 0.00015271]
    ])

    segs = frontier_segments(mu_ex, sigma_ex, verbose=False, graph=True)
    print("\nFinal segments array shape:", segs.shape)

